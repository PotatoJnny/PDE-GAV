"""Empirical Cartesian velocity quadrature for signed source functions.

There are no Oracle calls or case assumptions in this module.  ``fit`` learns
separate velocity spaces from fully sampled source matrices, selects Cartesian
sampling nodes, and integrates the interpolant.  It exactly integrates the
chosen monomials on the retained tensor-product space (up to roundoff).
This is not a positive quadrature and does not certify unresolved functions.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy.linalg import qr


def _axes(values, nx, ny, name):
    if isinstance(values, tuple) and len(values) == 2:
        x, y = (np.asarray(z, dtype=float) for z in values)
    else:
        x = y = np.asarray(values, dtype=float)
    if x.shape != (nx,) or y.shape != (ny,):
        raise ValueError(f"{name} must contain one vector or a pair of axis vectors")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError(f"nonfinite {name}")
    return x, y


def _pod(z, axis):
    # Eigh of a small axis covariance avoids the much larger unfolding SVD.
    if axis == 0:
        c = np.einsum("rxy,rzy->xz", z, z, optimize=True)
    else:
        c = np.einsum("rxy,rxz->yz", z, z, optimize=True)
    s2, u = np.linalg.eigh(c)
    s2 = np.maximum(s2[::-1], 0.)
    return u[:, ::-1], s2


def _rank(s2, relative_tolerance, cap):
    if s2.sum() <= np.finfo(float).tiny:
        return 1
    tail = np.r_[np.cumsum(s2[::-1])[::-1], 0.]
    acceptable = np.flatnonzero(tail <= relative_tolerance**2 * s2.sum())
    return max(1, min(int(acceptable[0]), int(cap), len(s2)))


def _select(u, count):
    """Q-DEIM followed by determinant-greedy oversampling."""
    rank = u.shape[1]
    selected = list(qr(u.T, pivoting=True, mode="economic")[2][:rank])
    while len(selected) < count:
        b = u[selected]
        gram_inverse = np.linalg.pinv(b.T @ b, rcond=1e-13)
        leverage = np.einsum("ij,jk,ik->i", u, gram_inverse, u)
        leverage[selected] = -np.inf
        selected.append(int(np.argmax(leverage)))
    return np.asarray(selected, dtype=int)


@dataclass
class TensorCubature:
    """Prepared rule; arrays are ordinary NumPy host values.

    ``sampled = full_values[..., rule.index_x, :][..., rule.index_y]``
    (or ``rule.sample(full_values)``) has the shape expected by ``moments``.
    ``moment_pairs`` uses degree ordering: (0,0),(1,0),(0,1),(2,0),....
    """
    nodes_x: np.ndarray
    nodes_y: np.ndarray
    integration_x: np.ndarray
    integration_y: np.ndarray
    index_x: np.ndarray
    index_y: np.ndarray
    interpolation_x: np.ndarray
    interpolation_y: np.ndarray
    moment_pairs: tuple
    moment_weights: np.ndarray
    diagnostics: dict[str, Any]

    @property
    def vx(self):
        return self.nodes_x[self.index_x]

    @property
    def vy(self):
        return self.nodes_y[self.index_y]

    @property
    def node_count(self):
        return len(self.index_x) * len(self.index_y)

    def sample(self, full_values):
        return np.take(np.take(full_values, self.index_x, axis=-2), self.index_y, axis=-1)

    def moments(self, values):
        """Return (..., n_moments), integrated over the full parent grid."""
        values = np.asarray(values)
        if values.shape[-2:] != (len(self.index_x), len(self.index_y)):
            raise ValueError("values do not have the rule's Cartesian node shape")
        return np.einsum("...ij,mij->...m", values, self.moment_weights, optimize=True)

    def reconstruct(self, values):
        """Reconstruct the parent grid; useful for diagnostics, not required."""
        return np.einsum("xi,...ij,yj->...xy", self.interpolation_x,
                         np.asarray(values), self.interpolation_y, optimize=True)

    def functional_weights(self, functional):
        """Weights for arbitrary scalar function(s), including phase mixing.

        ``functional`` has shape (...,nx,ny) and does not include the parent
        integration weights.  Complex Fourier transport functions are allowed.
        The resulting shape is (...,selected_nx,selected_ny).
        """
        f = np.asarray(functional)
        if f.shape[-2:] != (len(self.nodes_x), len(self.nodes_y)):
            raise ValueError("functional must be defined on the full velocity grid")
        fw = f * self.integration_x[:, None] * self.integration_y[None, :]
        return np.einsum("xi,...xy,yj->...ij", self.interpolation_x, fw,
                         self.interpolation_y, optimize=True)

    def validate(self, full_values):
        """RMS diagnostics against the supplied full-grid numerical quadrature.

        This method does not assert a full-velocity or PDE accuracy bound.
        All ratios use aggregate RMS errors, never pointwise relative errors.
        Near-zero individual moments can naturally have large relative error.
        """
        f = np.asarray(full_values, dtype=float)
        if f.ndim != 3 or f.shape[1:] != (len(self.nodes_x), len(self.nodes_y)):
            raise ValueError("validation requires (nrows,nx,ny) full matrices")
        full_weights = np.stack([
            (self.integration_x * self.nodes_x**i)[:, None] *
            (self.integration_y * self.nodes_y**j)[None, :]
            for i, j in self.moment_pairs])
        target = np.einsum("rxy,mxy->rm", f, full_weights, optimize=True)
        sampled = self.sample(f)
        pred = self.moments(sampled)
        rms_error = np.sqrt(np.mean((pred-target)**2, axis=0))
        rms_target = np.sqrt(np.mean(target**2, axis=0))
        tiny = max(float(np.max(rms_target)), 1.) * 1e-14
        relative = rms_error / np.maximum(rms_target, tiny)
        reconstructed = self.reconstruct(sampled)
        area = self.integration_x[:, None] * self.integration_y[None, :]
        e2 = np.sum((reconstructed-f)**2 * area)
        f2 = np.sum(f*f * area)
        # Degree groups use their natural polynomial powers; x/y have same
        # physical units, so sums within a degree have meaningful ratios.
        groups = {}
        for degree in range(max(i+j for i, j in self.moment_pairs)+1):
            mask = np.asarray([i+j == degree for i, j in self.moment_pairs])
            groups[str(degree)] = float(np.linalg.norm(rms_error[mask]) /
                max(float(np.linalg.norm(rms_target[mask])), tiny))
        return {
            "rows": len(f), "weighted_reconstruction_relative_rms":
                float(np.sqrt(e2 / max(f2, np.finfo(float).tiny))),
            "moment_pairs": [list(p) for p in self.moment_pairs],
            "moment_rms_error": rms_error.tolist(),
            "moment_reference_rms": rms_target.tolist(),
            "moment_relative_rms": relative.tolist(),
            "degree_relative_rms": groups,
            "max_moment_relative_rms": float(np.max(relative)),
        }


def fit(samples, nodes, integration_weights, max_rank=40, *,
        relative_tolerance=1e-7, max_nodes=None, oversample=0,
        degree=4, velocity_scale=None, tail_power=4,
        normalize_rows=True, rcond=1e-12, validation=None):
    """Fit a bounded-cost tensor rule from ``samples[ntrain,nvx,nvy]``.

    ``max_rank`` is an integer or pair of axis caps.  ``max_nodes`` bounds the
    Cartesian node count and may lower ranks below the requested tolerance.
    ``oversample`` adds nodes per axis and reduces alias/noise amplification.
    A scale-weighted POD represents tails; ``tail_power=0`` gives ordinary
    quadrature-weighted POD.  ``velocity_scale`` is a positive scalar or pair.
    The full-grid integration supplied here must itself be sufficiently
    resolved; heldout diagnostics only test compression relative to that grid.

    To limit training expense callers should supply at most a few hundred
    representative matrices.  This routine forms only nx-by-nx and ny-by-ny
    covariance matrices, never a (nx*ny)-square covariance.
    """
    f = np.asarray(samples, dtype=float)
    if f.ndim != 3 or min(f.shape) < 1 or not np.all(np.isfinite(f)):
        raise ValueError("finite nonempty samples[ntrain,nvx,nvy] required")
    if degree < 0 or int(degree) != degree:
        raise ValueError("degree must be a nonnegative integer")
    nx, ny = f.shape[1:]
    vx, vy = _axes(nodes, nx, ny, "nodes")
    qx, qy = _axes(integration_weights, nx, ny, "integration_weights")
    if min(qx.min(), qy.min()) <= 0:
        raise ValueError("strictly positive parent integration weights required")
    if np.isscalar(max_rank):
        caps = [min(nx, int(max_rank)), min(ny, int(max_rank))]
    else:
        caps = [min(nx, int(max_rank[0])), min(ny, int(max_rank[1]))]
    if min(caps) < 1 or relative_tolerance <= 0 or oversample < 0:
        raise ValueError("positive rank/tolerance and nonnegative oversample required")
    if velocity_scale is None:
        f2 = np.sum(f*f, axis=0) * qx[:, None] * qy[None, :]
        denom = max(float(f2.sum()), np.finfo(float).tiny)
        scales = [max(1., float(np.sqrt(np.sum(f2*vx[:, None]**2)/denom))),
                  max(1., float(np.sqrt(np.sum(f2*vy[None, :]**2)/denom)))]
    elif np.isscalar(velocity_scale):
        scales = [float(velocity_scale), float(velocity_scale)]
    else:
        scales = [float(s) for s in velocity_scale]
    if min(scales) <= 0:
        raise ValueError("positive velocity scales required")
    ax = np.sqrt(qx) * (1. + np.abs(vx/scales[0])**tail_power)
    ay = np.sqrt(qy) * (1. + np.abs(vy/scales[1])**tail_power)
    z = f * ax[None, :, None] * ay[None, None, :]
    if normalize_rows:
        norms = np.linalg.norm(z.reshape(len(z), -1), axis=1)
        z /= np.maximum(norms[:, None, None], np.finfo(float).tiny)
    ux, sx2 = _pod(z, 0)
    uy, sy2 = _pod(z, 1)
    ranks = [_rank(sx2, relative_tolerance, caps[0]),
             _rank(sy2, relative_tolerance, caps[1])]
    full_ranks = list(ranks)
    extra = int(oversample)
    def counts(rs):
        return [min(nx, rs[0]+extra), min(ny, rs[1]+extra)]
    if max_nodes is not None:
        if max_nodes < 1:
            raise ValueError("max_nodes must be positive")
        # Reduce the axis with the least weighted retained energy per saved
        # Cartesian point, retaining at least one mode on each axis.
        while np.prod(counts(ranks)) > int(max_nodes):
            candidates = []
            for a, eigenvalues in enumerate((sx2, sy2)):
                if ranks[a] > 1:
                    reduced = list(ranks); reduced[a] -= 1
                    saved = np.prod(counts(ranks)) - np.prod(counts(reduced))
                    if saved > 0:
                        candidates.append((eigenvalues[ranks[a]-1]/max(eigenvalues.sum(),1e-300)/saved, a))
            if candidates:
                ranks[min(candidates)[1]] -= 1
            elif extra:
                extra -= 1
            else:
                raise ValueError("unable to satisfy node budget")
    selected = []; matrices = []; conditions = []; singular_min = []
    for u, a, rank, count in zip((ux, uy), (ax, ay), ranks, counts(ranks)):
        b = u[:, :rank]
        ind = _select(b, count)
        sampled_b = b[ind]
        pinv = np.linalg.pinv(sampled_b, rcond=rcond)
        interpolation = (b @ pinv) * a[ind][None, :] / a[:, None]
        selected.append(ind); matrices.append(interpolation)
        conditions.append(float(np.linalg.cond(sampled_b)))
        singular_min.append(float(np.linalg.svd(sampled_b, compute_uv=False)[-1]))
    pairs = tuple((d-j, j) for d in range(int(degree)+1) for j in range(d+1))
    wx = np.stack([(qx*vx**i) @ matrices[0] for i in range(int(degree)+1)])
    wy = np.stack([(qy*vy**j) @ matrices[1] for j in range(int(degree)+1)])
    weights = np.stack([wx[i, :, None] * wy[j, None, :] for i, j in pairs])
    amplification = []
    for k, (i, j) in enumerate(pairs):
        full_l1 = np.sum(qx*np.abs(vx)**i) * np.sum(qy*np.abs(vy)**j)
        amplification.append(float(np.sum(np.abs(weights[k])) / max(full_l1,1e-300)))
    diag = {
        "ranks": ranks, "ranks_before_joint_cap": full_ranks,
        "nodes_per_axis": [len(s) for s in selected],
        "node_count": int(np.prod([len(s) for s in selected])),
        "basis_sample_condition": conditions, "basis_sample_min_singular": singular_min,
        "axis_relative_projection_tail": [float(np.sqrt(s[r:].sum()/max(s.sum(),1e-300)))
                                          for s, r in zip((sx2,sy2),ranks)],
        "quadrature_l1_amplification": amplification,
        "velocity_scale": scales, "tail_power": tail_power,
        "normalize_rows": bool(normalize_rows), "training_rows": len(f),
        "rcond": rcond,
    }
    result = TensorCubature(vx.copy(), vy.copy(), qx.copy(), qy.copy(),
        selected[0], selected[1], matrices[0], matrices[1], pairs, weights, diag)
    if validation is not None:
        result.diagnostics["heldout"] = result.validate(validation)
    return result


def regularize_moments(rule, training, ridge=1e-3, *, baseline_fraction=1.,
                       validation=None):
    """Fit regularized moment weights on the existing Cartesian nodes.

    Default weights start at the tensor rule, fitting its residual moments by
    a minimum-norm ridge correction. ``baseline_fraction=0`` instead fits all
    moments directly from sampled values.  No nodes are changed.  A new rule
    is returned; the baseline is retained untouched for independent comparison.

    Heldout checks are essential: training precision is not a generalization
    guarantee.  Reconstruction and arbitrary functional weights remain the
    original tensor interpolant; ONLY ``moments`` uses the corrected rule.
    ``ridge`` is dimensionless relative to the largest normalized training
    singular value squared.  Oracle values need not be positive.
    """
    f = np.asarray(training, dtype=float)
    if f.ndim != 3 or f.shape[1:] != (len(rule.nodes_x),len(rule.nodes_y)):
        raise ValueError("training requires (rows,nx,ny) full matrices")
    if not np.all(np.isfinite(f)) or ridge <= 0:
        raise ValueError("finite training and positive ridge required")
    values = rule.sample(f).reshape(len(f), -1)
    full_weights = np.stack([
        (rule.integration_x*rule.nodes_x**i)[:,None] *
        (rule.integration_y*rule.nodes_y**j)[None,:]
        for i,j in rule.moment_pairs])
    target = np.einsum("rxy,mxy->rm",f,full_weights,optimize=True)
    baseline = baseline_fraction*rule.moment_weights.reshape(len(rule.moment_pairs),-1).T
    residual = target-values@baseline
    # Row normalization keeps a few high-amplitude source rows from entirely
    # determining the correction while preserving source zeros consistently.
    norm = np.linalg.norm(values,axis=1)
    floor = max(float(np.max(norm))*1e-12, np.finfo(float).tiny)
    norm = np.maximum(norm,floor)
    x = values/norm[:,None]
    b = residual/norm[:,None]
    u,s,vt = np.linalg.svd(x,full_matrices=False)
    lam = ridge*max(float(s[0]**2),np.finfo(float).tiny)
    delta = vt.T @ ((s/(s*s+lam))[:,None]*(u.T@b))
    weights = (baseline+delta).T.reshape(rule.moment_weights.shape)
    diagnostics = dict(rule.diagnostics)
    diagnostics.pop("heldout",None)
    full_l1 = np.array([np.sum(rule.integration_x*np.abs(rule.nodes_x)**i)*
                       np.sum(rule.integration_y*np.abs(rule.nodes_y)**j)
                       for i,j in rule.moment_pairs])
    old_l1 = np.sum(np.abs(rule.moment_weights),axis=(1,2))
    new_l1 = np.sum(np.abs(weights),axis=(1,2))
    diagnostics["correction"] = {
        "ridge":float(ridge), "baseline_fraction":float(baseline_fraction),
        "effective_degrees_of_freedom":float(np.sum(s*s/(s*s+lam))),
        "training_matrix_condition":float(s[0]/max(float(s[-1]),np.finfo(float).tiny)),
        "weight_l1_ratio_to_tensor":(new_l1/np.maximum(old_l1,1e-300)).tolist(),
        "delta_l2_ratio_to_tensor":(np.linalg.norm(delta,axis=0)/
            np.maximum(np.linalg.norm(rule.moment_weights.reshape(len(rule.moment_pairs),-1),axis=1),1e-300)).tolist(),
        "training_rows":len(f),
    }
    diagnostics["quadrature_l1_amplification"] = (new_l1/np.maximum(full_l1,1e-300)).tolist()
    result = replace(rule,moment_weights=weights,diagnostics=diagnostics)
    if validation is not None:
        result.diagnostics["heldout"] = result.validate(validation)
    return result

