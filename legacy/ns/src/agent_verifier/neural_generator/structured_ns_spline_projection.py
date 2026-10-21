"""Opt-in joint velocity fitting with basis-independent Sobolev penalties.

The current quadrature implementation covers rectangular manufactured-trace
fields without obstacles. Other geometries fail explicitly; their established
separable projector remains available. No dataset name selects this fitter.
"""
from __future__ import annotations
from dataclasses import replace
import numpy as np
import torch
from .structured_ns_answer_first import (
    StructuredNSProjectionConfig, StructuredNSCoefficientProjector,
    _projection_spatial_operators, _summary, STRUCTURED_NS_GENERATOR_SCHEMA,
)
from .continuous_geometry import continuous_geometry_spec_from_contract
from .structured_ns_basis import axis_basis_derivatives


def axis_quadrature(config, axis, *, device='cpu'):
    kind = config.temporal_basis if axis == 't' else config.spatial_basis
    if kind == 'bspline':
        knots = config.temporal_knots if axis == 't' else getattr(config, f'spatial_knots_{axis}')
        edges = np.unique(knots)
        degree = config.temporal_degree if axis == 't' else config.spatial_degree
        z, w = np.polynomial.legendre.leggauss(degree+1)
        lo, hi = edges[:-1, None], edges[1:, None]
        points = (lo+(z+1)*(hi-lo)/2).ravel()
        weights = np.broadcast_to(w*(hi-lo)/2, (len(lo), len(w))).ravel()
    else:
        count = config.time_coefficient_count if axis == 't' else getattr(config, f'spatial_mode_{axis}')+1
        z, w = np.polynomial.legendre.leggauss(max(64, 4*count))
        points, weights = (z+1)/2, w/2
    return (torch.as_tensor(points, dtype=torch.float64, device=device),
            torch.as_tensor(weights.copy(), dtype=torch.float64, device=device))


class JointSmoothNSCoefficientProjector:
    """Solve the joint quadratic problem by batched preconditioned CG.

    Equal weighting of u/v; normalized domain integrals. Spatial roughness is
    Lref^4 * integral |Hessian(velocity)|_F^2 / 2 with Lref=sqrt(Lx*Ly).
    Temporal roughness is integral |d_tau^2 velocity|^2 / 2. These definitions
    are identical for cosine/Chebyshev and B-spline bases. Ridge_relative does
    not enter this objective. The streamfunction constant coefficient
    direction is projected out, leaving the physical velocity unchanged.
    """
    def __init__(self, *, geometry, t, config=None, device='cpu'):
        self.config = config or StructuredNSProjectionConfig(fitting_method='joint_smoothing')
        self.config.validate()
        if self.config.fitting_method != 'joint_smoothing':
            raise ValueError('joint projector requires an explicit joint_smoothing configuration')
        self.device, self.geometry = torch.device(device), geometry
        self.geometry_spec = continuous_geometry_spec_from_contract(geometry)
        spec = self.geometry_spec
        if spec.outer_boundary != 'manufactured_trace' or spec.obstacle_polygons:
            raise ValueError('joint Sobolev quadrature currently requires an obstacle-free manufactured-trace rectangle')
        self.times = torch.as_tensor(t, dtype=torch.float64, device=self.device)
        if (self.times.ndim != 1 or not torch.isfinite(self.times).all()
                or len(self.times) <= self.config.time_order or not torch.all(torch.diff(self.times) > 0)):
            raise ValueError('joint fitting requires increasing finite times and fewer basis coefficients than frames')
        self.fluid = torch.as_tensor(np.asarray(geometry.fluid_mask, dtype=bool), device=self.device)
        self.spatial_operators = _projection_spatial_operators(torch, geometry=geometry,
            geometry_spec=spec, config=self.config, device=self.device)
        tau = (self.times-self.times[0])/(self.times[-1]-self.times[0])
        self.temporal_design = axis_basis_derivatives(tau, config=self.config, axis='t')[0]
        grams = {}
        for axis in ('x', 'y', 't'):
            q, w = axis_quadrature(self.config, axis, device=self.device)
            jets = axis_basis_derivatives(q, config=self.config, axis=axis, order=2 if axis == 't' else 3)
            grams[axis] = [b.T @ (w[:, None]*b) for b in jets]
        gx, gy, gt = grams['x'], grams['y'], grams['t']
        lx, ly = spec.domain_bounds[1]-spec.domain_bounds[0], spec.domain_bounds[3]-spec.domain_bounds[2]
        kron = torch.kron
        self.velocity_mass = .5*(kron(gx[0], gy[1])/ly**2+kron(gx[1], gy[0])/lx**2)
        self.spatial_roughness = .5*(lx*ly)**2*(
            3*kron(gx[2], gy[1])/(lx**4*ly**2)+3*kron(gx[1], gy[2])/(lx**2*ly**4)
            +kron(gx[0], gy[3])/ly**6+kron(gx[3], gy[0])/lx**6)
        self.time_mass, self.time_roughness = gt[0], gt[2]
        size = self.config.spatial_coefficient_count
        self.gauge = torch.ones(size, dtype=torch.float64, device=self.device)
        if self.config.spatial_basis == 'cosine':
            self.gauge.zero_(); self.gauge[0] = 1.
        self.gauge /= self.gauge.norm()
        self.objective = {
            'data': 'mean squared error over selected time, fluid sites, and u/v',
            'spatial': 'Lref^4/2 integral normalized_box normalized_time sum_components |physical_Hessian(velocity)|_F^2',
            'temporal': '1/2 integral normalized_box normalized_time sum_components |d_tau^2 velocity|^2',
            'Lref': float(np.sqrt(lx*ly)), 'time_scale': float(self.times[-1]-self.times[0]),
            'gauge': 'coefficient Euclidean projection against the constant streamfunction direction',
            'ridge_relative_used_in_velocity_objective': False,
        }

    def _gauge_fix(self, values):
        return values-(values@self.gauge)[..., None]*self.gauge

    def prepare_system(self, *, time_mask=None, spatial_mask=None):
        nt, nf = len(self.times), int(self.fluid.sum())
        tm = torch.ones(nt, dtype=torch.bool, device=self.device) if time_mask is None else torch.as_tensor(time_mask, dtype=torch.bool, device=self.device)
        sm = torch.ones(nf, dtype=torch.bool, device=self.device) if spatial_mask is None else torch.as_tensor(spatial_mask, dtype=torch.bool, device=self.device)
        if tm.shape != (nt,) or sm.shape != (nf,) or not tm.any() or not sm.any():
            raise ValueError('holdout masks must retain times and fluid sites')
        T = self.temporal_design[tm]
        V = self.spatial_operators['velocity_design'][torch.cat((sm, sm))]
        return {'time_mask': tm, 'spatial_mask': sm, 'T': T, 'V': V,
                'Gt': T.T@T/len(T), 'Gv': V.T@V/len(V)}

    def solve(self, target, *, system=None, smoothing_spatial=None, smoothing_temporal=None,
              tolerance=1e-8, max_iterations=500):
        """target is [batch,time,2,fluid_sites]; excluded values are never read."""
        s = system or self.prepare_system()
        ys = torch.as_tensor(target, dtype=torch.float64, device=self.device)
        if ys.ndim != 4 or tuple(ys.shape[1:]) != (len(self.times), 2, int(self.fluid.sum())):
            raise ValueError('joint target must have shape [batch,time,2,fluid_sites]')
        observed = ys[:, s['time_mask']][:, :, :, s['spatial_mask']].reshape(len(ys), len(s['T']), -1)
        if not torch.isfinite(observed).all():
            raise ValueError('nonfinite observed velocity')
        ls = self.config.smoothing_spatial if smoothing_spatial is None else float(smoothing_spatial)
        lt = self.config.smoothing_temporal if smoothing_temporal is None else float(smoothing_temporal)
        if not np.isfinite([ls, lt]).all() or min(ls, lt) < 0:
            raise ValueError('invalid smoothing strengths')
        gt, gv, ms, rs, mt, rt = s['Gt'], s['Gv'], self.velocity_mass, self.spatial_roughness, self.time_mass, self.time_roughness
        rhs = self._gauge_fix(s['T'].T @ observed @ s['V']/(len(s['T'])*len(s['V'])))
        def operator(c):
            return self._gauge_fix(gt@c@gv + ls*(mt@c@rs) + lt*(rt@c@ms))
        # Jitter is applied only to the preconditioner, never to the objective.
        ps = gv+ls*rs
        gauge_scale = ps.diag().mean().clamp_min(1e-30)
        ps = ps+gauge_scale*torch.outer(self.gauge, self.gauge)
        pt = gt+lt*rt
        ps_jitter = float(gauge_scale)*1e-12
        pt_jitter = float(pt.diag().mean())*1e-12
        cs = torch.linalg.cholesky((ps+ps.T)/2+torch.eye(len(ps), device=self.device)*ps_jitter)
        ct = torch.linalg.cholesky((pt+pt.T)/2+torch.eye(len(pt), device=self.device)*pt_jitter)
        def precondition(r):
            z = torch.cholesky_solve(r, ct)
            return self._gauge_fix(torch.cholesky_solve(z.transpose(-2,-1), cs).transpose(-2,-1))
        c = torch.zeros_like(rhs); r = rhs.clone(); z = precondition(r); p = z.clone()
        dot = lambda a,b: (a*b).sum(dim=(-2,-1))
        rz = dot(r,z); norm = rhs.flatten(1).norm(dim=1).clamp_min(1e-30)
        relative = r.flatten(1).norm(dim=1)/norm
        iterations = torch.zeros(len(ys), dtype=torch.int64, device=self.device)
        for iteration in range(max_iterations):
            active = relative > tolerance
            if not active.any():
                break
            ap = operator(p); denominator = dot(p,ap)
            if ((denominator <= 0) & active).any():
                raise RuntimeError('joint normal operator lost positive curvature')
            alpha = torch.where(active, rz/denominator.clamp_min(1e-300), 0.)
            c += alpha[:,None,None]*p; r -= alpha[:,None,None]*ap
            iterations[active] += 1
            relative = r.flatten(1).norm(dim=1)/norm
            z = precondition(r); rz_new = dot(r,z)
            beta = torch.where(active, rz_new/rz.clamp_min(1e-300), 0.)
            p = z+beta[:,None,None]*p; rz = rz_new
        actual = (rhs-operator(c)).flatten(1).norm(dim=1)/norm
        if (actual > max(tolerance*10, 1e-7)).any():
            raise RuntimeError(f'joint fit did not converge: normal residual {float(actual.max()):.3g}')
        diag = {'iterations': iterations.tolist(), 'normal_relative_residual': actual.tolist(),
                'smoothing_spatial': ls, 'smoothing_temporal': lt,
                'preconditioner_jitter_spatial': ps_jitter, 'preconditioner_jitter_temporal': pt_jitter,
                'retained_times': len(s['T']), 'retained_sites': len(s['V'])//2}
        return self._gauge_fix(c), diag

    def predict(self, psi):
        return (self.temporal_design @ psi @ self.spatial_operators['velocity_design'].T).reshape(
            len(psi), len(self.times), 2, -1)

    def project(self, *, velocity, pressure):
        v = torch.as_tensor(velocity, dtype=torch.float64, device=self.device)
        p = torch.as_tensor(pressure, dtype=torch.float64, device=self.device)
        if v.ndim == 4: v = v[None]
        if p.ndim == 3: p = p[None]
        if (v.ndim != 5 or v.shape[2] != 2 or tuple(v.shape[-2:]) != tuple(self.fluid.shape)
                or tuple(p.shape) != (len(v), v.shape[1], *self.fluid.shape) or not torch.isfinite(p).all()):
            raise ValueError('velocity/pressure grid shapes or finite values disagree')
        target = v[:, :, :, self.fluid]
        system = self.prepare_system()
        fitted, diagnostics, rows = [], [], []
        relative = lambda a,b: float((a-b).norm()/b.norm().clamp_min(1e-12))
        for start in range(0, len(target), 32):
            batch = target[start:start+32]
            chunk, diagnostic = self.solve(batch, system=system)
            fitted.append(chunk); diagnostics.append(diagnostic)
            rows.extend({'velocity_relative_l2': relative(a,b), 'initial_velocity_relative_l2': relative(a[0],b[0])}
                        for a,b in zip(self.predict(chunk),batch))
        psi = torch.cat(fitted)
        diagnostic = dict(diagnostics[0])
        for key in ('iterations', 'normal_relative_residual'):
            diagnostic[key] = [value for item in diagnostics for value in item[key]]
        diagnostic['batch_cases'] = 32
        pc = torch.zeros_like(psi)
        pressure_method = 'identically_zero'
        if torch.count_nonzero(p):
            old = StructuredNSCoefficientProjector(geometry=self.geometry, t=self.times,
                config=replace(self.config, fitting_method='separable_ridge', smoothing_spatial=0., smoothing_temporal=0.), device=self.device)
            pressure_target = p[:, :, self.fluid]
            pressure_target = pressure_target-pressure_target.mean(-1, keepdim=True)
            pc = old.temporal_projection @ (pressure_target @ old.pressure_projection.T)
            pressure_method = 'original_separable_pressure_ridge; velocity Sobolev penalty does not apply to pressure'
        coefficients = torch.stack((psi,pc),dim=1).reshape(len(v),2,self.config.time_coefficient_count,
            self.config.spatial_mode_x+1,self.config.spatial_mode_y+1)
        return coefficients, {'schema': 'ns2d_structured_joint_sobolev_projection_v1',
            'generator_schema': STRUCTURED_NS_GENERATOR_SCHEMA, 'projection': 'joint_velocity_sobolev_pcg',
            'config': self.config.to_dict(), 'objective': self.objective, 'pressure_method': pressure_method,
            'operator_diagnostics': diagnostic, 'cases': rows,
            'summary': {key: _summary([row[key] for row in rows]) for key in rows[0]}}
