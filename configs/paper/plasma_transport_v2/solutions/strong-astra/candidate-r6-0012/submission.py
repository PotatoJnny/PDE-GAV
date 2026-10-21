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
NT = 1537
FULL_MIN = 1538
FULL_MAX = 1537
LOW_DEGREE = 6


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
_stable_source = _bundle_module("source_spline_stable")
_initial_projection = _bundle_module("initial_projection")
_initial_normalization = _bundle_module("initial_normalization")
_separable_projection = _bundle_module("separable_projection")
_multirate_sampling = _bundle_module("multirate_sampling")
_multirate_gate = _bundle_module("multirate_gate")
_pilot_density = _bundle_module("pilot_density")
_stable_quintic = _bundle_module("stable_quintic")
_bounded_cubic = _bundle_module("bounded_cubic")
_late_source = _bundle_module("late_source_plan")
_piecewise.make_piecewise_cubic_spline = _bounded_cubic.make_piecewise_cubic_spline
_piecewise.make_piecewise_quintic_spline = _stable_quintic.make_piecewise_quintic_spline
_pilot_density.PiecewiseCubic = _bounded_cubic.GuardedPiecewiseCubic


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
        self.interpolant = _stable_source.make_piecewise_cubic_spline(times, modes, breaks)
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
    finish_extra = 0.

    def get(method, points):
        nonlocal chunk_cost
        chunks = []
        for i in range(0, len(points), limit):
            remaining = budget - (candidate_solve_clock() - started)
            if remaining < 110 + finish_extra + 3 * chunk_cost:
                raise SamplingDeadline("reserve remaining budget for integration and projection")
            mark = candidate_solve_clock()
            values = np.asarray(getattr(session, method)(points[i:i + limit]), dtype=float)
            chunk_cost = max(.1, .7 * chunk_cost + .3 * (candidate_solve_clock() - mark))
            chunks.append(values)
        return np.concatenate(chunks)

    nq = _quadrature.select_quadrature(get, periods=periods, t_interval=(t0, t1),
                                      theta=theta, variance=variance, base_order=NQ, degree=degree)
    initial_order, initial_rule_diagnostics = _initial_projection.initial_quadrature_order(
        get, periods=periods, theta=theta, variance=variance, degree=degree, base_order=nq)
    normalize_initial, initial_normalization_diagnostics = _initial_normalization.choose_normalization(
        get, arrays=preview, theta=theta, variance=variance, order=initial_order,
        remaining_time=lambda: budget - (candidate_solve_clock() - started))
    state, initial_spatial_diagnostics = _initial_projection.project_initial(
        get, periods=periods, state_n=n, theta=theta, variance=variance,
        degree=degree, order=initial_order, sample_n=61, max_points=limit,
        normalize=normalize_initial)
    z, w = np.polynomial.hermite_e.hermegauss(nq)
    quad_scale = math.sqrt(theta)
    v = z * quad_scale
    weights = w * np.exp(z ** 2 / 2) * quad_scale
    hx = hermite(v / scale, degree)
    basis = np.array([(hx[a, :, None] * hx[b, None, :] * weights[:, None] * weights[None, :]).ravel()
                      for a, b in indices])
    source_projector = _separable_projection.SeparableHermiteProjector(hx * weights[None, :])
    vx, vy = np.meshgrid(v, v, indexing="ij")
    phase = np.empty((n * n, nq * nq, 4))
    phase[:, :, :2] = xy[:, None, :]
    phase[:, :, 2] = vx.ravel()
    phase[:, :, 3] = vy.ravel()
    phase = phase.reshape(-1, 4)
    source_points = np.empty((len(phase), 5))
    source_points[:, 1:] = phase
    pilot_bank = {}
    spatial_guard = None
    small_source_points = None
    trial_guard = None
    late_plan = None
    trial_source_points = None
    small_n = min(33, n)
    trial_n = min(29, small_n)
    small_after = t0 + 4. / nu if nu > 0 else math.inf

    def full_forcing(t):
        source_points[:, 0] = t
        values = get("source", source_points).reshape(n * n, -1)
        moments = source_projector.project(values).T.reshape(count, n, n)
        modes = fft2(moments, axes=(-2, -1))
        modes[0, 0, 0] = 0.
        return modes

    def forcing(t):
        nonlocal small_source_points, trial_source_points
        if late_plan is not None and t >= small_after and t in pilot_bank:
            candidate_modes = late_plan.sample(t, pilot_bank[t])
            if candidate_modes is not None:
                return candidate_modes
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
            moments = source_projector.project(values).T.reshape(count, trial_n, trial_n)
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
        moments = source_projector.project(values).T.reshape(count,small_n,small_n)
        modes = _spatial_source.pad_fourier_modes(fft2(moments,axes=(-2,-1)),n)
        modes[0,0,0] = 0.
        if spatial_guard.accept(modes,pilot_bank[t]):
            return modes
        return full_forcing(t)

    # Retain a complete time interval before spending the remaining budget on
    # pilot discovery and source refinement.
    samples = {t0: forcing(t0), t1: forcing(t1)}
    low_samples = None
    low_guard = None
    multirate_diagnostics = []
    low_rule_diagnostics = None
    corner_times = np.empty(0, dtype=float)
    pilot_times = pilot_values = corner_indices = None

    def finish():
        nonlocal source_points, phase, basis, state, small_source_points, trial_source_points
        times = np.array(sorted(samples))
        sample_values = np.array([samples[t] for t in times])
        density_values = None
        density_diagnostics = {'enabled': False, 'reason': 'incomplete pilot or insufficient full snapshots'}
        remaining = budget - (candidate_solve_clock() - started)
        if (pilot_values is not None and len(times) >= 12 and remaining > 93. + finish_extra
                and np.all(np.isin(corner_times, times))
                and np.all(np.isin(times, pilot_times))):
            mark = candidate_solve_clock()
            model = None
            try:
                model = _pilot_density.prepare_density_correction(
                    times, sample_values[:, 0], pilot_times, pilot_xy, pilot_values[:, :, 0],
                    periods=periods, corner_times=corner_times, first_passing=True,
                    should_continue=lambda: budget - (candidate_solve_clock() - started) > 90. + finish_extra)
                density_diagnostics = model.diagnostics
                if model.enabled and budget - (candidate_solve_clock() - started) > 90. + finish_extra:
                    density_values = model.evaluate(pilot_times)[:, None, :, :]
                elif model.enabled:
                    density_diagnostics = dict(density_diagnostics, enabled=False,
                        reason='reserve ordinary finishing time before dense projection')
            except (ValueError, ArithmeticError, np.linalg.LinAlgError, MemoryError) as error:
                density_diagnostics = {'enabled': False, 'reason': str(error)}
                density_values = None
            finally:
                model = None
            density_diagnostics['attempt_seconds'] = candidate_solve_clock() - mark
        elif remaining <= 93. + finish_extra:
            density_diagnostics['reason'] = 'insufficient reserve for density correction'
        samples.clear()
        break_indices = np.flatnonzero(np.isin(times, corner_times))
        source_degree = 3
        if pilot_values is not None and low_samples is None and density_values is None:
            selected = np.searchsorted(pilot_times, times)
            if np.all(selected < len(pilot_times)) and np.array_equal(pilot_times[selected], times):
                source_degree, _ = _degree_selection.choose_source_spline_degree(
                    pilot_times, pilot_values, selected, corner_indices, theta, nu,
                    _piecewise.make_piecewise_cubic_spline,
                    _piecewise.make_piecewise_quintic_spline)
        builder = (_piecewise.make_piecewise_quintic_spline if source_degree == 5
                   else _piecewise.make_piecewise_cubic_spline)
        source_spline = builder(times, sample_values, break_indices)
        low_arguments = {}
        if density_values is not None:
            low_breaks = np.flatnonzero(np.isin(pilot_times, corner_times))
            low_spline = _piecewise.make_piecewise_cubic_spline(pilot_times, density_values, low_breaks)
            del density_values
            low_arguments = dict(low_source_coefficients=low_spline.c,
                                 low_source_times=low_spline.t,
                                 low_source_degree=3, low_count=1)
            times = np.union1d(times, pilot_times)
        elif low_samples is not None:
            low_times = np.array(sorted(low_samples))
            low_values = np.array([low_samples[t] for t in low_times])
            low_samples.clear()
            low_breaks = np.flatnonzero(np.isin(low_times, corner_times))
            low_spline = _piecewise.make_piecewise_cubic_spline(low_times, low_values, low_breaks)
            del low_values
            low_arguments = dict(low_source_coefficients=low_spline.c,
                                 low_source_times=low_spline.t,
                                 low_source_degree=3, low_count=low_spline.c.shape[1])
            times = np.union1d(times, low_times)
        del sample_values, source_points, phase, basis, small_source_points, trial_source_points
        pilot_bank.clear()
        save_times, saved = _gpu.integrate_gpu(state, source_spline.c, source_spline.t,
            n=n, degree=degree, variance=variance, theta=theta, nu=nu,
            periods=periods, t_interval=(t0, t1), solve_budget=budget, started=started,
            time_knots=times, source_degree=source_degree, **low_arguments)
        del source_spline, state
        result = Prepared(save_times, saved, periods, variance, corner_times)
        result.source_spatial_diagnostics = (None if spatial_guard is None
                                            else spatial_guard.diagnostics())
        result.source_trial_diagnostics = (None if trial_guard is None
                                          else trial_guard.diagnostics())
        result.initial_rule_diagnostics = initial_rule_diagnostics
        result.initial_spatial_diagnostics = initial_spatial_diagnostics
        result.initial_normalization_diagnostics = initial_normalization_diagnostics
        result.multirate_diagnostics = multirate_diagnostics
        result.low_rule_diagnostics = low_rule_diagnostics
        result.low_source_diagnostics = None if low_guard is None else low_guard.diagnostics()
        result.pilot_density_diagnostics = density_diagnostics
        result.late_source_diagnostics = None if late_plan is None else late_plan.diagnostics()
        return result

    # Cheap spatial probes identify time intervals that need the expensive
    # complete spatial/velocity mesh. Probe coordinates never depend on queries.
    pilot_fraction = np.unique(np.r_[np.linspace(0., 1., 2049),
                                     np.geomspace(1.e-9, 1. / 256, 256)])
    physical_pilot = np.clip(t0 + (t1 - t0) * pilot_fraction, t0, t1)
    physical_pilot[0], physical_pilot[-1] = t0, t1
    pilot_times = np.unique(physical_pilot)
    probe_index = np.arange(1, 17)
    plastic = 1.324717957244746
    pilot_xy = np.column_stack((np.mod(probe_index / plastic, 1),
                                np.mod(probe_index / plastic ** 2, 1))) * periods
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
        try:
            late_plan = None if small_after >= t1 else _late_source.LateSourcePlan(get, periods=periods,
                theta=theta, variance=variance, degree=degree, state_n=n,
                full_order=nq, pilot_xy=pilot_xy, reference_modes=samples[t1],
                reference_values=pilot_bank[t1], reference_time=t1,
                guard_factory=_spatial_source.SpatialSourceGuard,
                rule_factory=_separable_projection.weighted_hermite_rule,
                projector_factory=_separable_projection.SeparableHermiteProjector,
                pad_modes=_spatial_source.pad_fourier_modes)
        except SamplingDeadline:
            return finish()
    order = _corners.select_corner_knot_order(
        pilot_times, pilot_values, theta, nu, NT, corners=corner_indices)
    selected_times = pilot_times[order]
    del pilot_basis
    low_degree = min(LOW_DEGREE, degree)
    low_count = (low_degree + 1) * (low_degree + 2) // 2
    low_points = {}
    low_basis = low_velocity = None

    def low_forcing(t):
        def at_mesh(m):
            if m not in low_points:
                ax, ay = np.meshgrid(np.arange(m)*periods[0]/m,
                                     np.arange(m)*periods[1]/m, indexing="ij")
                p = np.empty((m*m, len(low_velocity), 5))
                p[:, :, 1:3] = np.column_stack((ax.ravel(), ay.ravel()))[:, None, :]
                p[:, :, 3:] = low_velocity
                low_points[m] = p.reshape(-1, 5)
            p = low_points[m]
            p[:, 0] = t
            values = get("source", p).reshape(m*m, -1)
            moments = (values @ low_basis.T).T.reshape(low_count, m, m)
            modes = _spatial_source.pad_fourier_modes(fft2(moments, axes=(-2, -1)), n)
            modes[0, 0, 0] = 0.
            return modes
        if t >= small_after and low_guard is not None and low_guard.enabled and small_n < n:
            modes = at_mesh(small_n)
            if low_guard.accept(modes, pilot_bank[t][:, :low_count]):
                return modes
        return at_mesh(n)

    for selected_time in selected_times:
        if low_samples is None and len(samples) >= FULL_MAX:
            break
        t = float(selected_time)
        if t in (samples if low_samples is None else low_samples):
            continue
        try:
            if low_samples is not None:
                low_samples[t] = low_forcing(t)
                continue
            samples[t] = forcing(t)
            finish_extra = .045 * max(0, len(samples) - 769)
            if len(samples) >= FULL_MIN and (len(samples)-FULL_MIN) % 64 == 0:
                selected = np.searchsorted(pilot_times, np.array(sorted(samples)))
                all_values = np.array([pilot_bank[float(pt)] for pt in pilot_times])
                assessment = _multirate_gate.assess_high_moment_temporal_error(
                    pilot_times, all_values, selected, corner_indices, nu,
                    degree=degree, bands=((low_degree+1, min(10, degree)), (11, degree)),
                    thresholds=(2.e-6, 1.e-5))
                del all_values
                multirate_diagnostics.append(dict(full_count=len(samples), **assessment))
                if assessment["passed"]:
                    low_order, low_velocity, low_basis, low_rule_diagnostics = _multirate_sampling.choose_low_rule(
                        get, pilot_times, pilot_xy, pilot_bank, theta=theta,
                        variance=variance, full_order=nq, degree=low_degree)
                    low_samples = {pt: modes[:low_count].copy() for pt, modes in samples.items()}
                    low_guard = _spatial_source.SpatialSourceGuard(
                        n, pilot_xy, periods, low_degree, samples[t1][:low_count],
                        pilot_bank[t1][:, :low_count])
                    low_rule_diagnostics = {"order": low_order, "probes": low_rule_diagnostics}
        except SamplingDeadline:
            break
    return finish()
