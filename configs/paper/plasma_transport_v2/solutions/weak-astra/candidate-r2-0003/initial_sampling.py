"""Solver-owned initial quadrature for bulk-centered triangular raw moments."""
from __future__ import annotations

import numpy as np
from pde_agent_verifier_solver.sdk import open_public_case_session


def moment_pairs(degree=5):
    """Total degree ascending, x degree descending within each total degree."""
    if type(degree) is not int or degree < 0:
        raise ValueError("degree must be a nonnegative integer")
    return tuple((a, total-a) for total in range(degree+1)
                 for a in range(total, -1, -1))


def public_geometry(case):
    question = case.public_data["question"]
    periods = np.asarray(question["grid"]["x_domain"], dtype=float)
    interval = np.asarray(question["time_domain"], dtype=float)
    if (periods.shape != (2,) or not np.isfinite(periods).all()
            or np.any(periods <= 0) or interval.shape != (2,)
            or not np.isfinite(interval).all() or interval[1] <= interval[0]):
        raise ValueError("positive public periods and ordered time interval required")
    limit = question.get("numerical_access", {}).get("max_points_per_request", 262144)
    if type(limit) is not int or limit <= 0:
        raise ValueError("positive public request limit required")
    return periods, interval, min(limit, 262144)


def public_bulk_center(case):
    """Read the initial preview only to center the velocity quadrature."""
    preview = case.public_data.get("arrays", {})
    if "rho0" in preview and "momentum0" in preview:
        rho = np.asarray(preview["rho0"], dtype=float)
        momentum = np.asarray(preview["momentum0"], dtype=float)
        if (rho.ndim == 2 and momentum.shape == rho.shape+(2,)
                and np.isfinite(rho).all() and np.isfinite(momentum).all()
                and np.mean(rho) > 0):
            return np.mean(momentum, axis=(0, 1))/np.mean(rho)
    return np.zeros(2)


def sample_initial(case, check_budget=None, nx=32, nv=80, vmax=16.,
                   degree=5, session=None):
    """Integrate g0, normalize by public rho0, and return continuous modes.

    Every monomial is in w=v-center. The returned dictionary is candidate-owned
    state, not a query-specific answer. No source values are requested here.
    """
    if type(nx) is not int or nx < 4 or type(nv) is not int or nv < 4:
        raise ValueError("nx and nv must be integers at least four")
    if not np.isfinite(vmax) or vmax <= 0:
        raise ValueError("vmax must be finite and positive")
    if check_budget is None:
        check_budget = lambda: None
    periods, interval, limit = public_geometry(case)
    center = public_bulk_center(case)
    pairs = moment_pairs(degree)
    axes = tuple(np.arange(nx)*length/nx for length in periods)
    xy = np.column_stack([a.ravel() for a in np.meshgrid(*axes, indexing="ij")])
    velocity_axis = (np.arange(nv)-nv//2)*2*vmax/nv
    relative_nodes = np.column_stack([
        a.ravel() for a in np.meshgrid(velocity_axis, velocity_axis, indexing="ij")])
    vx, vy = relative_nodes.T
    weights = np.asarray([vx**a*vy**b for a, b in pairs])*(2*vmax/nv)**2
    velocity_nodes = relative_nodes+center[None, :]
    if session is None:
        session = open_public_case_session(case)
    check_budget()
    density_parts = []
    for start in range(0, len(xy), limit):
        batch = xy[start:start+limit]
        values = np.asarray(session.rho0(batch), dtype=float)
        if (values.shape != (len(batch),) or not np.isfinite(values).all()
                or np.any(values <= 0)):
            raise ValueError("initial density must be finite and positive")
        density_parts.append(values)
        check_budget()
    density = np.concatenate(density_parts)
    moments = np.empty((len(xy), len(pairs)))
    blocks = max(1, limit//len(velocity_nodes))
    for start in range(0, len(xy), blocks):
        cells = xy[start:start+blocks]
        phase_points = np.concatenate([
            np.repeat(cells, len(velocity_nodes), axis=0),
            np.tile(velocity_nodes, (len(cells), 1))], axis=1)
        parts = []
        for offset in range(0, len(phase_points), limit):
            batch = phase_points[offset:offset+limit]
            values = np.asarray(session.g0(batch), dtype=float)
            if values.shape != (len(batch),) or not np.isfinite(values).all():
                raise ValueError("initial velocity density must be finite")
            parts.append(values)
            check_budget()
        distribution = np.concatenate(parts).reshape(len(cells), len(velocity_nodes))
        local = distribution@weights.T
        if np.any(local[:, 0] <= 0):
            raise ValueError("positive initial velocity mass required")
        local *= (density[start:start+len(cells)]/local[:, 0])[:, None]
        local[:, 0] = density[start:start+len(cells)]
        moments[start:start+len(cells)] = local
    moments = moments.T.reshape(len(pairs), nx, nx)
    modes = np.fft.fft2(moments)/nx**2
    if nx % 2 == 0:
        modes[..., nx//2, :] = 0.
        modes[..., :, nx//2] = 0.
    check_budget()
    return {"moments": moments, "modes": modes, "periods": periods,
            "interval": interval, "center": center, "pairs": pairs,
            "axes": axes, "degree": degree}
