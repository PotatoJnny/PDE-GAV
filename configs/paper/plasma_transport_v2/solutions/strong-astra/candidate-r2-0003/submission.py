"""Fourier--Hermite discretization of forced VPFP on the full velocity space.

The initial data and forcing are integrated with solver-owned quadrature.
Only the computed low moments and their continuous Fourier/time interpolant
are retained for projection after preparation.
"""
from __future__ import annotations

import math
import importlib.util
from pathlib import Path
import numpy as np
from scipy.fft import fft2
from scipy.interpolate import CubicSpline, make_interp_spline
from pde_agent_verifier_solver.sdk import AnswerView, QueryKind, open_public_case_session
from pde_agent_verifier_solver.sdk.solve_budget import candidate_solve_clock

FIELDS = ("rho", "electric_x", "electric_y", "momentum_x", "momentum_y",
          "pressure_xx", "pressure_xy", "pressure_yy", "heat_flux_x", "heat_flux_y")
NX = 41
DEGREE = 16
NQ = 24
NT = 769


def _bundle_module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_gpu = _bundle_module("gpu_integrator_bspline")
_temporal = _bundle_module("temporal_sampling")
_quadrature = _bundle_module("quadrature_select")
_corners = _bundle_module("corner_sampling")
_piecewise = _bundle_module("source_spline_piecewise")
_refinement = _bundle_module("corner_refinement")
_degree_selection = _bundle_module("spline_degree_selection")
_corner_proposals = _bundle_module("corner_proposals")
_spatial_source = _bundle_module("spatial_source_guard")


def hermite(z, degree):
    h = np.empty((degree + 1, len(z)))
    h[0] = 1.
    h[1] = z
    for k in range(1, degree):
        h[k + 1] = (z * h[k] - math.sqrt(k) * h[k - 1]) / math.sqrt(k + 1)
    return h


class Prepared:
    def __init__(self, times, modes, periods, variance, corner_times=()):
        breaks = np.flatnonzero(np.isin(times, corner_times))
        self.interpolant = _piecewise.make_piecewise_cubic_spline(times, modes, breaks)
        self.periods = periods
        self.interval = (times[0], times[-1])
        self.variance = variance
        self.n = modes.shape[-1]
        k = np.fft.fftfreq(self.n) * self.n
        self.kx, self.ky = np.meshgrid(2 * np.pi * k / periods[0],
                                     2 * np.pi * k / periods[1], indexing="ij")
        k2 = self.kx ** 2 + self.ky ** 2
        self.poisson_x = np.divide(-1j * self.kx, k2, out=np.zeros_like(k2, dtype=complex), where=k2 > 0)
        self.poisson_y = np.divide(-1j * self.ky, k2, out=np.zeros_like(k2, dtype=complex), where=k2 > 0)

    def evaluate(self, query):
        if query.kind is not QueryKind.POINTS:
            raise ValueError("POINTS queries required")
        p = np.asarray([(row["t"], row["x"], row["y"]) for row in query.points], dtype=float)
        if not np.isfinite(p).all() or np.any(p[:, 0] < self.interval[0]) or np.any(p[:, 0] > self.interval[1]):
            raise ValueError("query outside prepared domain")
        coeff = self.interpolant(p[:, 0])
        phase = np.exp(1j * (p[:, 1, None, None] * self.kx + p[:, 2, None, None] * self.ky)) / self.n ** 2
        h = np.einsum("ncxy,nxy->cn", coeff, phase).real
        ex = np.einsum("nxy,nxy,xy->n", coeff[:, 0], phase, self.poisson_x).real
        ey = np.einsum("nxy,nxy,xy->n", coeff[:, 0], phase, self.poisson_y).real
        # Order: (00),(10),(01),(20),(11),(02),(30),(21),(12),(03).
        rho = h[0]
        scale = math.sqrt(self.variance)
        jx, jy = scale * h[1], scale * h[2]
        ux, uy = jx / np.maximum(rho, 1.e-8), jy / np.maximum(rho, 1.e-8)
        rxx = self.variance * (rho + math.sqrt(2) * h[3])
        rxy = self.variance * h[4]
        ryy = self.variance * (rho + math.sqrt(2) * h[5])
        qx = .5 * scale ** 3 * (math.sqrt(6) * h[6] + math.sqrt(2) * h[8] + 4 * h[1])
        qy = .5 * scale ** 3 * (math.sqrt(6) * h[9] + math.sqrt(2) * h[7] + 4 * h[2])
        qx += -(rxx * ux + rxy * uy) - .5 * (rxx + ryy) * ux + rho * (ux ** 2 + uy ** 2) * ux
        qy += -(rxy * ux + ryy * uy) - .5 * (rxx + ryy) * uy + rho * (ux ** 2 + uy ** 2) * uy
        vals = dict(zip(FIELDS, (rho, ex, ey, jx, jy, rxx - jx * ux, rxy - jx * uy, ryy - jy * uy, qx, qy)))
        return AnswerView(kind=query.kind, fields={k: tuple(map(float, vals[k])) for k in query.fields},
                          points=query.points, shape=(len(p),))

    def close(self):
        self.interpolant = None


def prepare_solution(case, solve_budget=None):
    started = candidate_solve_clock()
    budget = 850. if solve_budget is None else float(solve_budget)
    question = case.public_data["question"]
    periods = np.asarray(question["grid"]["x_domain"], dtype=float)
    t0, t1 = map(float, question["time_domain"])
    nu, theta = float(question["collision_nu"]), float(question["temperature"])
    preview = case.public_data.get("arrays", {})
    if "kinetic_energy0" in preview:
        variance = max(theta, float(np.mean(preview["kinetic_energy0"])))
    else:
        variance = 2 * theta
    variance = min(max(variance, theta), 4 * theta)
    scale = math.sqrt(variance)
    n, degree = NX, DEGREE
    indices = [(a, d - a) for d in range(degree + 1) for a in range(d, -1, -1)]
    count = len(indices)
    axes = [np.arange(n) * length / n for length in periods]
    x, y = np.meshgrid(*axes, indexing="ij")
    xy = np.column_stack((x.ravel(), y.ravel()))
    session = open_public_case_session(case)
    limit = min(262144, question.get("numerical_access", {}).get("max_points_per_request", 262144))

    class SamplingDeadline(Exception):
        pass

    chunk_cost = .5

    def get(method, points):
        nonlocal chunk_cost
        chunks = []
        for i in range(0, len(points), limit):
            remaining = budget - (candidate_solve_clock() - started)
            if remaining < 65 + 3 * chunk_cost:
                raise SamplingDeadline("reserve remaining budget for integration and projection")
            mark = candidate_solve_clock()
            values = np.asarray(getattr(session, method)(points[i:i + limit]), dtype=float)
            chunk_cost = max(.1, .7 * chunk_cost + .3 * (candidate_solve_clock() - mark))
            chunks.append(values)
        return np.concatenate(chunks)

    nq = _quadrature.select_quadrature(get, periods=periods, t_interval=(t0, t1),
                                      theta=theta, variance=variance, base_order=NQ, degree=degree)
    rho = get("rho0", xy)
    z, w = np.polynomial.hermite_e.hermegauss(nq)
    quad_scale = math.sqrt(theta)
    v = z * quad_scale
    weights = w * np.exp(z ** 2 / 2) * quad_scale
    hx = hermite(v / scale, degree)
    basis = np.array([(hx[a, :, None] * hx[b, None, :] * weights[:, None] * weights[None, :]).ravel()
                      for a, b in indices])
    vx, vy = np.meshgrid(v, v, indexing="ij")
    phase = np.empty((n * n, nq * nq, 4))
    phase[:, :, :2] = xy[:, None, :]
    phase[:, :, 2] = vx.ravel()
    phase[:, :, 3] = vy.ravel()
    phase = phase.reshape(-1, 4)
    g = get("g0", phase).reshape(n * n, -1)
    initial = (g @ basis.T).T.reshape(count, n, n) * rho.reshape(n, n)
    initial[0] = rho.reshape(n, n)
    state = fft2(initial, axes=(-2, -1))
    state[0, 0, 0] = n * n
    source_points = np.empty((len(phase), 5))
    source_points[:, 1:] = phase
    pilot_bank = {}
    spatial_guard = None
    small_source_points = None
    trial_guard = None
    trial_source_points = None
    small_n = min(33, n)
    trial_n = min(29, small_n)
    small_after = t0 + 4. / nu if nu > 0 else math.inf

    def full_forcing(t):
        source_points[:, 0] = t
        values = get("source", source_points).reshape(n * n, -1)
        moments = (values @ basis.T).T.reshape(count, n, n)
        modes = fft2(moments, axes=(-2, -1))
        modes[0, 0, 0] = 0.
        return modes

    def forcing(t):
        nonlocal small_source_points, trial_source_points
        if (small_n >= n or t < small_after or spatial_guard is None
                or not spatial_guard.enabled or t not in pilot_bank):
            return full_forcing(t)
        if trial_guard is not None and trial_guard.enabled:
            if trial_source_points is None:
                ax, ay = np.meshgrid(np.arange(trial_n)*periods[0]/trial_n,
                                     np.arange(trial_n)*periods[1]/trial_n, indexing="ij")
                trial_xy = np.column_stack((ax.ravel(), ay.ravel()))
                trial_source_points = np.empty((trial_n*trial_n, nq*nq, 5))
                trial_source_points[:, :, 1:3] = trial_xy[:, None, :]
                trial_source_points[:, :, 3] = vx.ravel()
                trial_source_points[:, :, 4] = vy.ravel()
                trial_source_points = trial_source_points.reshape(-1, 5)
            trial_source_points[:, 0] = t
            values = get("source", trial_source_points).reshape(trial_n*trial_n, -1)
            moments = (values @ basis.T).T.reshape(count, trial_n, trial_n)
            modes = _spatial_source.pad_fourier_modes(fft2(moments, axes=(-2, -1)), n)
            modes[0, 0, 0] = 0.
            accepted = trial_guard.accept(modes, pilot_bank[t])
            if trial_guard.rejected >= 3:
                trial_guard.enabled = False
            if accepted:
                return modes
        if small_source_points is None:
            ax, ay = np.meshgrid(np.arange(small_n)*periods[0]/small_n,
                                 np.arange(small_n)*periods[1]/small_n,indexing="ij")
            small_xy = np.column_stack((ax.ravel(),ay.ravel()))
            small_source_points = np.empty((small_n*small_n,nq*nq,5))
            small_source_points[:,:,1:3] = small_xy[:,None,:]
            small_source_points[:,:,3] = vx.ravel()
            small_source_points[:,:,4] = vy.ravel()
            small_source_points = small_source_points.reshape(-1,5)
        small_source_points[:,0] = t
        values = get("source",small_source_points).reshape(small_n*small_n,-1)
        moments = (values@basis.T).T.reshape(count,small_n,small_n)
        modes = _spatial_source.pad_fourier_modes(fft2(moments,axes=(-2,-1)),n)
        modes[0,0,0] = 0.
        if spatial_guard.accept(modes,pilot_bank[t]):
            return modes
        return full_forcing(t)

    # Retain a complete time interval before spending the remaining budget on
    # pilot discovery and source refinement.
    samples = {t0: forcing(t0), t1: forcing(t1)}
    corner_times = np.empty(0, dtype=float)
    pilot_times = pilot_values = corner_indices = None

    def finish():
        nonlocal source_points, phase, g, basis, state, initial, small_source_points, trial_source_points
        times = np.array(sorted(samples))
        sample_values = np.array([samples[t] for t in times])
        samples.clear()
        break_indices = np.flatnonzero(np.isin(times, corner_times))
        source_degree = 3
        if pilot_values is not None:
            selected = np.searchsorted(pilot_times, times)
            if np.all(selected < len(pilot_times)) and np.array_equal(pilot_times[selected], times):
                source_degree, _ = _degree_selection.choose_source_spline_degree(
                    pilot_times, pilot_values, selected, corner_indices, theta, nu,
                    _piecewise.make_piecewise_cubic_spline,
                    _piecewise.make_piecewise_quintic_spline)
        builder = (_piecewise.make_piecewise_quintic_spline if source_degree == 5
                   else _piecewise.make_piecewise_cubic_spline)
        source_spline = builder(times, sample_values, break_indices)
        del sample_values, source_points, phase, g, basis, small_source_points, trial_source_points
        pilot_bank.clear()
        save_times, saved = _gpu.integrate_gpu(state, source_spline.c, source_spline.t,
            n=n, degree=degree, variance=variance, theta=theta, nu=nu,
            periods=periods, t_interval=(t0, t1), solve_budget=budget, started=started,
            time_knots=times, source_degree=source_degree)
        del source_spline, state, initial
        result = Prepared(save_times, saved, periods, variance, corner_times)
        result.source_spatial_diagnostics = (None if spatial_guard is None
                                            else spatial_guard.diagnostics())
        result.source_trial_diagnostics = (None if trial_guard is None
                                          else trial_guard.diagnostics())
        return result

    # Cheap spatial probes identify time intervals that need the expensive
    # complete spatial/velocity mesh. Probe coordinates never depend on queries.
    pilot_fraction = np.unique(np.r_[np.linspace(0., 1., 2049),
                                     np.geomspace(1.e-9, 1. / 256, 256)])
    pilot_times = t0 + (t1 - t0) * pilot_fraction
    probe_index = np.arange(1, 17)
    pilot_xy = np.column_stack((np.mod(probe_index * .6180339887498949, 1),
                                np.mod(probe_index * .4142135623730951, 1))) * periods
    hp = hermite(v / math.sqrt(theta), 3)
    pilot_basis = np.array([(hp[a, :, None] * hp[b, None, :] * weights[:, None] * weights[None, :]).ravel()
                           for a, b in indices[:10]])
    def sample_pilot(request_times):
        values = np.empty((len(request_times), len(pilot_xy), 10))
        for first in range(0, len(request_times), 32):
            part_times = request_times[first:first + 32]
            points = np.empty((len(part_times), len(pilot_xy), nq * nq, 5))
            points[:, :, :, 0] = part_times[:, None, None]
            points[:, :, :, 1:3] = pilot_xy[None, :, None, :]
            points[:, :, :, 3] = vx.ravel()
            points[:, :, :, 4] = vy.ravel()
            raw = get("source", points.reshape(-1, 5)).reshape(len(part_times), len(pilot_xy), nq * nq)
            values[first:first + len(part_times)] = raw @ pilot_basis.T
            full_moments = raw @ basis.T
            for i, pilot_time in enumerate(part_times):
                pilot_bank[float(pilot_time)] = full_moments[i].copy()
        return values

    try:
        pilot_values = sample_pilot(pilot_times)
    except SamplingDeadline:
        return finish()
    corner_indices = _corners.detect_corner_indices(pilot_times, pilot_values)
    proposed = np.union1d(corner_indices,
        _corner_proposals.propose_weak_corner_indices(pilot_times, pilot_values))
    try:
        pilot_times, pilot_values, corner_indices = _refinement.refine_corner_pilot(
            pilot_times, pilot_values, proposed, sample_pilot, rounds=4,
            trusted_corner_indices=corner_indices)
    except SamplingDeadline:
        pass
    corner_times = pilot_times[corner_indices]
    if t1 in pilot_bank and small_n < n:
        spatial_guard = _spatial_source.SpatialSourceGuard(
            n,pilot_xy,periods,degree,samples[t1],pilot_bank[t1])
        if trial_n < small_n:
            trial_guard = _spatial_source.SpatialSourceGuard(
                n, pilot_xy, periods, degree, samples[t1], pilot_bank[t1])
    order = _corners.select_corner_knot_order(
        pilot_times, pilot_values, theta, nu, NT, corners=corner_indices)
    selected_times = pilot_times[order]
    del pilot_basis
    for selected_time in selected_times:
        t = float(selected_time)
        if t in samples:
            continue
        try:
            samples[t] = forcing(t)
        except SamplingDeadline:
            break
    return finish()
