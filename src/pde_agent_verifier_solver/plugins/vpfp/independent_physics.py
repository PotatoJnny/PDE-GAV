"""Local independent quadrature/coordinate-FD checks of frozen VPFP answers.

This is NOT a population audit or an admission gate.  The actual native f is
integrated, and S is evaluated by the old analytic or the A autodiff backend;
S is never reconstructed from the moment balances being checked.  For A, the
Gaussian carrier is the ORIGINAL bath Gaussian pushed through the continuous
affine map, not a Gaussian approximation with A's generated covariance.

N+3/N+5 rules integrate moments through kinetic energy (and f's energy flux).
The optional L2 rule instead uses the squared-Gaussian weight, with 2N+4/2N+6
nodes per axis.  Polynomial exactness in real arithmetic is not a certificate
for finite-precision decoder evaluation, source cancellation, or all t,x,y.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from hashlib import sha256
import importlib
import json
import math

import numpy as np
import torch

from .moving_frame import MovingFrameDecoder, source_local_moments
from .moving_frame_backend import manufacture_moving_frame_source
from .perturbation_backend import _legacy_api


SCHEMA = "vpfp_independent_local_physics_v1"
MOMENT_NAMES = ("mass", "jx", "jy", "K", "Pxx", "Pxy", "Pyy", "Qx", "Qy")
SOURCE_MOMENT_NAMES = ("source_mass", "source_jx", "source_jy", "source_K")


def _numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().double().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _finite(value):
    """JSON conversion rejects, rather than hides, nonfinite computed values."""
    if isinstance(value, Mapping):
        return {str(k): _finite(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_finite(v) for v in value]
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return _finite(_numpy(value).tolist())
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(value):
            raise FloatingPointError("nonfinite independent-physics result")
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _ratio(numerator, denominator):
    """No epsilon/clipping: undefined divisions remain explicit JSON nulls."""
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.broadcast_to(np.asarray(denominator, dtype=np.float64), numerator.shape)
    defined = np.isfinite(denominator) & (denominator > 0)
    out = np.zeros_like(numerator)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        np.divide(numerator, denominator, out=out, where=defined)
    defined &= np.isfinite(out)
    flat = [float(v) if ok else None for v, ok in zip(out.ravel(), defined.ravel())]
    return np.asarray(flat, dtype=object).reshape(out.shape).tolist()


class _Runtime:
    def __init__(self, decoder, answer, condition, chunk_size):
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        self.decoder, self.moving = decoder, isinstance(decoder, MovingFrameDecoder)
        self.base = decoder.base_decoder if self.moving else decoder
        self.cfg, self.parameter = self.base.config, next(self.base.parameters())
        self.device = self.parameter.device
        self.degree = int(self.cfg.hermite_total_degree)
        if not 0 <= self.degree <= 14:
            raise ValueError("this fixed audit protocol supports native Hermite degrees 0..14")
        theta = _numpy(condition)
        if theta.shape == (1, 2):
            theta = theta[0]
        if theta.shape != (2,) or not np.isfinite(theta).all() or theta[0] < 0 or theta[1] <= 0:
            raise ValueError("one finite physical [nu>=0, temperature>0] condition required")
        # Match the frozen decoder's actual condition arithmetic; report it.
        self.condition = torch.as_tensor(theta, device=self.device, dtype=self.parameter.dtype)
        self.nu, self.temperature = map(float, _numpy(self.condition))
        if not math.isfinite(self.nu) or not math.isfinite(self.temperature) or self.nu < 0 or self.temperature <= 0:
            raise ValueError("condition is not finite/resolvable in the frozen decoder dtype")
        self.background = float(self.cfg.equation.background_density)
        self.domain = np.asarray(self.cfg.equation.x_domain, dtype=np.float64)
        self.time_domain = tuple(map(float, self.cfg.time_domain))
        if self.background <= 0 or not np.isfinite(self.background) or self.domain.shape != (2,) or not np.isfinite(self.domain).all() or not (self.domain > 0).all():
            raise ValueError("positive physical background and two spatial lengths required")
        if not self.time_domain[1] > self.time_domain[0]:
            raise ValueError("nondegenerate physical time interval required")
        if self.moving:
            if not isinstance(answer, Mapping) or set(answer) != {"m", "z"}:
                raise ValueError("A requires the complete {'m','z'} carrier, never z alone")
            self.answer = {}
            for key, dim in (("m", decoder.macro_config.macro_dimension), ("z", decoder.value_config.latent_dimension)):
                v = torch.as_tensor(answer[key], device=self.device).detach()
                if v.ndim == 1:
                    v = v[None]
                if v.shape != (1, dim) or not v.is_floating_point() or not bool(torch.isfinite(v).all()):
                    raise ValueError(f"one finite complete native A {key} required")
                self.answer[key] = v
        else:
            if isinstance(answer, Mapping):
                raise ValueError("the old decoder requires its complete full latent, not an A carrier")
            v = torch.as_tensor(answer, device=self.device, dtype=self.parameter.dtype).detach()
            if v.ndim == 1:
                v = v[None]
            if v.shape != (1, self.cfg.latent_dim) or not bool(torch.isfinite(v).all()):
                raise ValueError("one finite complete old full latent required")
            self.answer = v
        _legacy_api()
        self.analytic = importlib.import_module("agent_verifier.vpfp.analytic_source")
        self.macros = importlib.import_module("agent_verifier.vpfp.continuous_macros")
        self.analytic.require_analytic_structured_source(self.base)
        self.chunk_size = chunk_size
        self.counts = {"f_phase_points": 0, "source_phase_points": 0, "macro_points": 0}

    @contextmanager
    def measuring(self):
        was_training = self.decoder.training
        self.decoder.eval()
        try:
            with self.analytic.ieee_float32_coordinate_matmul(self.base):
                yield
        finally:
            self.decoder.train(was_training)

    def points(self, points):
        values = _numpy(points)
        if values.ndim != 2 or values.shape[1] != 3 or not len(values) or not np.isfinite(values).all():
            raise ValueError("finite nonempty points_txy [P,3] required")
        if (values[:, 0] < self.time_domain[0]).any() or (values[:, 0] > self.time_domain[1]).any():
            raise ValueError("time query outside native interval")
        return values

    def fields(self, points):
        phase = torch.as_tensor(points, device=self.device, dtype=torch.float64)
        with torch.no_grad():
            result = self.decoder.fields(phase, self.answer, equation_condition=self.condition)
        self.counts["macro_points"] += len(phase)
        return result

    def reference(self, point):
        q = np.r_[point, 0., 0.][None]
        fields = self.fields(q)
        rho = fields["rho"].double()
        if not bool((rho > 0).all()):
            raise ValueError("nonpositive represented density")
        if self.moving:
            j, k = fields["momentum"], fields["kinetic_energy"]
        else:
            c = fields["hermite_coefficients"].double()
            # The native fallback/normalization has ALREADY been selected. The
            # recurrence helper's tiny guard is inactive for this valid input.
            if bool((c.square().sum(-1) <= torch.finfo(torch.float64).tiny).any()):
                raise ValueError("invalid native normalized amplitude")
            moments = self.macros.torch_squared_hermite_moments(torch, c,
                modes=self.base.hermite_modes, temperature=fields["equation_condition"][:, 1].double(),
                floor_mixture=float(self.cfg.velocity_floor_mixture))
            j, k = rho[:, None]*moments["mean_velocity"], rho*moments["kinetic_energy"]
        result = {"moments": np.r_[_numpy(rho)[0], _numpy(j)[0], _numpy(k)[0]],
                  "potential_fields": np.r_[_numpy(rho)[0], _numpy(fields["phi"])[0], _numpy(fields["electric"])[0]]}
        _finite(result)
        return result

    def phase_values(self, phase, include_source):
        collected, disagreements, dtypes = [], [], []
        # Autodiff sources have much larger working memory than f alone.
        chunk = self.chunk_size if include_source else max(256, self.chunk_size)
        for start in range(0, len(phase), chunk):
            q = phase[start:start+chunk]
            if self.moving and include_source:
                fields = manufacture_moving_frame_source(self.decoder, q, self.answer, self.condition)
                actual_f, source = fields["f"], fields["source"]
                disagreement = 0.
            else:
                with torch.no_grad():
                    native = self.decoder.fields(q, self.answer, equation_condition=self.condition)
                    actual_f = native["f"]
                    fields = self.analytic.manufacture_standard_vpfp_source_analytic(
                        self.base, q, self.answer, equation_condition=self.condition) if include_source else None
                    source = fields["source"] if include_source else torch.zeros_like(actual_f)
                    disagreement = float((fields["f"].double()-actual_f.double()).abs().max()) if include_source else 0.
            if not bool(torch.isfinite(actual_f).all()) or not bool(torch.isfinite(source).all()):
                raise FloatingPointError("nonfinite actual f/source at a full-velocity quadrature node")
            if bool((actual_f < 0).any()):
                raise ValueError("negative actual f at a quadrature node")
            # Cast BEFORE squaring/weighting: FP32 f**2 would lose valid tails.
            collected.append(np.column_stack((_numpy(actual_f), _numpy(source))))
            disagreements.append(disagreement)
            dtypes.append(actual_f.dtype)
            self.counts["f_phase_points"] += len(q)
            self.counts["source_phase_points"] += len(q) if include_source else 0
        if len(set(dtypes)) != 1:
            raise ValueError("decoder output dtype changed within one quadrature")
        return np.concatenate(collected), max(disagreements), dtypes[0]


def _rule(runtime, point, order, power):
    if type(order) is not int or not 2 <= order <= 64 or power not in (1, 2):
        raise ValueError("GH order must be an integer in [2,64], power 1 or 2")
    minimum = runtime.degree+3 if power == 1 else 2*runtime.degree+4
    if order < minimum:
        raise ValueError(f"order {order} is below this protocol's polynomial-degree minimum {minimum}")
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    mesh = np.stack(np.meshgrid(nodes, nodes, indexing="ij"), -1).reshape(-1, 2)
    source_v = torch.as_tensor(mesh*math.sqrt(2*runtime.temperature/power), device=runtime.device, dtype=torch.float64)
    jacobian = 1.
    if runtime.moving:
        q0 = torch.as_tensor(np.r_[point, 0., 0.][None], device=runtime.device, dtype=torch.float64)
        with torch.no_grad():
            native = runtime.decoder.source_fields(q0, runtime.answer["z"], runtime.condition)
            original = source_local_moments(native, runtime.base)
            target = runtime.decoder.macro_at(q0, runtime.answer["m"], native)
            standardized = torch.linalg.solve_triangular(original["factor"][0],
                (source_v-original["mean"][0]).T, upper=False).T
            velocity = target["mean"][0]+standardized@target["factor"][0].T
            jacobian = float(torch.linalg.det(target["factor"][0])/torch.linalg.det(original["factor"][0]))
    else:
        velocity = source_v
    if not math.isfinite(jacobian) or jacobian <= 0:
        raise ValueError("nonfinite/nonpositive continuous carrier affine determinant")
    exponent = np.square(mesh).sum(1)
    measure = np.outer(weights, weights).ravel()*np.exp(exponent)*(2*runtime.temperature/power)*jacobian
    if not np.isfinite(measure).all() or not (measure > 0).all() or not bool(torch.isfinite(velocity).all()):
        raise FloatingPointError("nonfinite velocity nodes/importance measure; no clipping")
    phase = torch.cat((torch.as_tensor(point, device=runtime.device, dtype=torch.float64)[None].expand(len(mesh), -1), velocity), 1)
    return phase, measure, {"affine_determinant": jacobian, "gaussian_exponent_max": float(exponent.max()/power),
        "maximum_lebesgue_weight": float(measure.max()), "carrier_gaussian_temperature": runtime.temperature/power,
        "map": "u+L_target*inverse(L_source)*(w-mu_source)" if runtime.moving else "v=w",
        "weight": "physicists_GH_wx_wy*exp(zx^2+zy^2)*(2*theta/power)*det(dv/dw)"}


def _integrate(runtime, points, *, order, power, include_source):
    rows, diagnostic = [], []
    for point in runtime.points(points):
        phase, measure, rule = _rule(runtime, point, order, power)
        values, disagreement, dtype = runtime.phase_values(phase, include_source)
        f, source = values.T
        v = _numpy(phase[:, 3:5])
        energy = .5*np.square(v).sum(1)
        if power == 1:
            basis = np.column_stack((np.ones(len(v)), v, energy, v[:, 0]**2,
                v[:, 0]*v[:, 1], v[:, 1]**2, energy[:, None]*v))
            terms = (f*measure)[:, None]*basis
            row = {"moments": terms.sum(0)}
            if include_source:
                source_terms = (source*measure)[:, None]*basis[:, :4]
                row.update(source_moments=source_terms.sum(0), source_absolute_weighted_term_sums=np.abs(source_terms).sum(0))
        else:
            row = {"f_l2_squared": np.sum(np.square(f)*measure)}
            if include_source:
                row["source_l2_squared"] = np.sum(np.square(source)*measure)
        if any(not np.isfinite(value).all() for value in row.values()):
            raise FloatingPointError("nonfinite weighted integral, retained as failed not zero")
        rho = runtime.reference(point)["moments"][0]
        floor = float(runtime.cfg.velocity_floor_mixture)
        floor_bound = (math.log(rho)+math.log(floor)-math.log(2*math.pi*runtime.temperature)
            -rule["gaussian_exponent_max"]-math.log(rule["affine_determinant"])) if floor > 0 else None
        diagnostic.append({**rule, "evaluated_f_dtype": str(dtype), "evaluated_f_zero_count": int(np.count_nonzero(f == 0)),
            "evaluated_nonzero_f_below_normal_count": int(np.count_nonzero((f > 0) & (f < torch.finfo(dtype).tiny))),
            "nominal_floor_log_lower_bound": floor_bound,
            "nominal_positive_floor_above_dtype_normal": None if floor_bound is None else bool(floor_bound > math.log(torch.finfo(dtype).tiny)),
            "native_vs_analytic_f_max_absolute_difference": disagreement if not runtime.moving and include_source else None,
            "zero_source_is_not_itself_underflow_evidence": True,
            "underflow_and_roundoff_fully_certified": False})
        rows.append(row)
    return {**{key: np.asarray([row[key] for row in rows]) for key in rows[0]}, "quadrature": diagnostic}


def evaluate_velocity_moments(decoder, answer, condition, points_txy, *, order, chunk_size=32, include_source=True):
    """Return NumPy physical moment arrays at fixed txy; never write/generate.

    ``moments[P,9]`` order is MOMENT_NAMES; source_moments[P,4] follows
    SOURCE_MOMENT_NAMES.  The complete frozen native dtype/fallback remains in
    effect.  These are full R2 integrals at selected points, NOT all spacetime.
    """
    runtime = _Runtime(decoder, answer, condition, chunk_size)
    with runtime.measuring():
        return _integrate(runtime, points_txy, order=order, power=1, include_source=include_source)


def evaluate_velocity_l2(decoder, answer, condition, points_txy, *, order, chunk_size=32, include_source=True):
    """Full-R2 local f²/S² with the ORIGINAL Gaussian of temperature theta/2.

    At N14 use 32/34, not 17/19.  Actual f/S are evaluated first, converted to
    FP64, then squared; a Gaussian factor is never reconstructed from an
    already underflowed f.  Two-order agreement is only a numerical diagnostic.
    """
    runtime = _Runtime(decoder, answer, condition, chunk_size)
    with runtime.measuring():
        return _integrate(runtime, points_txy, order=order, power=2, include_source=include_source)


def _coordinate_fd(evaluate, point, axis, step, *, time_domain, spatial_domain, derivative=1):
    """Independent second-order coordinate FD; periodic x/y, bounded t."""
    point = np.asarray(point, dtype=np.float64)
    if not math.isfinite(step) or step <= 0 or derivative not in (1, 2):
        raise ValueError("positive finite FD step and derivative 1/2 required")
    if axis not in (0, 1, 2) or (axis == 0 and derivative == 2):
        raise ValueError("time FD here is first derivative only")
    if axis == 0 and (point[0]-step < time_domain[0] or point[0]+step > time_domain[1]):
        sign = 1 if point[0]+2*step <= time_domain[1] else -1
        if not time_domain[0] <= point[0]+2*sign*step <= time_domain[1]:
            raise ValueError("FD time step cannot fit the frozen interval")
        shifts, weights = [0., sign*step, 2*sign*step], np.asarray([-3., 4., -1.])/(2*sign*step)
    else:
        shifts = [-step, 0., step] if derivative == 2 else [-step, step]
        weights = np.asarray([1., -2., 1.])/step**2 if derivative == 2 else np.asarray([-1., 1.])/(2*step)
    result = None
    for shift, weight in zip(shifts, weights):
        query = point.copy(); query[axis] += shift
        if axis:
            query[axis] %= spatial_domain[axis-1]
        value = np.asarray(evaluate(query), dtype=np.float64)*weight
        result = value if result is None else result+value
    return result


def _balances(runtime, point, center, source, order, relative_step, cache):
    def moments(q):
        key = (order, tuple(q))
        if key not in cache:
            cache[key] = _integrate(runtime, q[None], order=order, power=1, include_source=False)["moments"][0]
        return cache[key]
    def potential(q):
        return runtime.reference(q)["potential_fields"]
    extent = np.r_[runtime.time_domain[1]-runtime.time_domain[0], runtime.domain]
    kwargs = {"time_domain": runtime.time_domain, "spatial_domain": runtime.domain}
    derivative = [_coordinate_fd(moments, point, axis, relative_step*extent[axis], **kwargs) for axis in range(3)]
    dt, dx, dy = derivative
    represented = potential(point)
    rho, j, k, electric = center[0], center[1:3], center[3], represented[2:4]
    lhs = np.r_[dt[0]+dx[1]+dy[2],
        dt[1:3]+np.asarray([dx[4]+dy[5], dx[5]+dy[6]])-rho*electric+runtime.nu*j,
        dt[3]+dx[7]+dy[8]-electric@j+2*runtime.nu*(k-runtime.temperature*rho)]
    ex = _coordinate_fd(potential, point, 1, relative_step*extent[1], **kwargs)
    ey = _coordinate_fd(potential, point, 2, relative_step*extent[2], **kwargs)
    lap = sum(_coordinate_fd(potential, point, axis, relative_step*extent[axis], derivative=2, **kwargs)[1] for axis in (1, 2))
    residual = lhs-source
    source_scales = runtime.nu*runtime.background*np.asarray([1., math.sqrt(runtime.temperature), math.sqrt(runtime.temperature), runtime.temperature])
    result = {"relative_step": relative_step, "physical_steps_txy": relative_step*extent,
        "balance_lhs": lhs, "source_integrated_independently": source, "balance_residual": residual,
        "balance_residual_collision_normalized": _ratio(residual, source_scales),
        "poisson_divE_minus_integrated_rho": ex[2]+ey[3]-(rho-runtime.background),
        "poisson_minus_laplacian_phi_minus_integrated_rho": -lap-(rho-runtime.background),
        "electric_plus_gradient_phi": electric+np.asarray([ex[1], ey[1]]),
        "represented_rho_minus_integrated_rho": represented[0]-rho,
        "poisson_divE_over_background": (ex[2]+ey[3]-(rho-runtime.background))/runtime.background,
        "poisson_laplacian_over_background": (-lap-(rho-runtime.background))/runtime.background,
        "electric_gradient_normalized_by_background_times_length": (electric+np.asarray([ex[1], ey[1]]))/(runtime.background*runtime.domain)}
    _finite(result)
    return result


def _spatial_mass(runtime, time, size, orders):
    if type(size) is not int or size < 3 or size % 2 != 1:
        raise ValueError("periodic spatial mass grids must be odd integers >=3")
    xy = np.stack(np.meshgrid(*(np.arange(size)*length/size for length in runtime.domain), indexing="ij"), -1).reshape(-1, 2)
    points = np.column_stack((np.full(len(xy), time), xy))
    represented = np.asarray([runtime.reference(point)["moments"][0] for point in points])
    values = [_integrate(runtime, points, order=order, power=1, include_source=False)["moments"][:, 0] for order in orders]
    volume = float(runtime.domain.prod())
    return _finite({"time": float(time), "grid_size": size, "velocity_orders": list(orders),
        "mean_integrated_f_by_velocity_order": [float(row.mean()) for row in values],
        "total_mass_by_velocity_order": [float(row.mean()*volume) for row in values],
        "mean_represented_rho": float(represented.mean()), "background_total_mass": runtime.background*volume,
        "mean_f_minus_background_over_background": [float(row.mean()/runtime.background-1) for row in values],
        "mean_rho_minus_background_over_background": float(represented.mean()/runtime.background-1),
        "pointwise_f_minus_rho_rms": [float(np.sqrt(np.mean(np.square(row-represented)))) for row in values],
        "fine_minus_coarse_velocity_mean_mass": float(values[1].mean()-values[0].mean()),
        "spatial_global_source_mass_balance_measured": False, "is_spatial_convergence_certificate": False})


def audit_independent_physics(decoder, answer, condition, *, points_txy, spatial_times=None,
        spatial_grids=(17, 33), chunk_size=32, relative_fd_step=2.**-8, include_l2=True):
    """Finite JSON ledger for ONE frozen answer; retains per-probe failures.

    The caller freezes IDs, points, grids and code/checkpoint lineage.  No file
    access, new answers, PDE solve, reference fitting, thresholds or admission
    occur here.  N14 uses 17/19 moment and (optionally) 32/34 L2 quadrature.
    Spatial mass grids evaluate f only, not a costly full-grid source oracle.
    Runtime is intentionally not estimated from tiny synthetic fixtures.
    """
    report = {"schema": SCHEMA, "execution_passed": False, "stage": "validate", "admitted": False,
        "numerically_uncertain": True, "physical_regime_qualified": False, "points": [], "spatial_mass": [],
        "scope": "fixed_local_txy_full_velocity_integrals_and_periodic_mass_grids_not_all_spacetime",
        "independence": "velocity_node_evaluation_and_coordinate_FD_vs_analytic_or_A_autodiff_source; shared_decoder_not_independent_law_truth",
        "pending": ["a_priori_calibrated_numerical_error_budget", "independent_refined_physical_reference_calibration",
                    "population_confirmation", "global_spatial_source_mass_balance", "all_spacetime_source_norm"]}
    try:
        runtime = _Runtime(decoder, answer, condition, chunk_size)
        points = runtime.points(points_txy)
        if not math.isfinite(relative_fd_step) or not 0 < relative_fd_step <= 1/16:
            raise ValueError("pre-registered relative FD step must lie in (0,1/16]")
        grids = tuple(spatial_grids)
        if any(type(size) is not int or size < 3 or size % 2 != 1 for size in grids) or tuple(sorted(set(grids))) != grids:
            raise ValueError("spatial grids must be strictly increasing odd integers >=3")
        times = sorted(set(float(v) for v in (points[:, 0] if spatial_times is None else spatial_times)))
        if not all(math.isfinite(t) and runtime.time_domain[0] <= t <= runtime.time_domain[1] for t in times):
            raise ValueError("spatial mass time outside frozen interval")
        orders, l2orders = (runtime.degree+3, runtime.degree+5), (2*runtime.degree+4, 2*runtime.degree+6)
        raw_parts = runtime.answer if isinstance(runtime.answer, Mapping) else {"full": runtime.answer}
        digest = sha256()
        for key, value in sorted({**raw_parts, "condition": runtime.condition}.items()):
            array = value.detach().cpu().contiguous().numpy()
            digest.update(key.encode()+str(array.dtype).encode()+str(array.shape).encode()+array.tobytes())
        report.update(answer_measurement_sha256=digest.hexdigest(), condition_native=_numpy(runtime.condition),
            decoder_config_sha256=sha256(json.dumps(runtime.cfg.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
            moving_frame_identity=runtime.decoder.identity() if runtime.moving else None,
            decoder_kind="moving_frame_A" if runtime.moving else "old_full_B", degree=runtime.degree,
            moment_names=MOMENT_NAMES, source_moment_names=SOURCE_MOMENT_NAMES, velocity_orders=orders,
            l2_velocity_orders=l2orders if include_l2 else [], relative_fd_steps=[relative_fd_step, relative_fd_step/2],
            scales={"rho": runtime.background, "thermal_speed": math.sqrt(runtime.temperature), "nu": runtime.nu,
                "source_moments": "nu*rho_background*[1,sqrt(theta),sqrt(theta),theta]",
                "local_source_L2": "nu*rho_background/sqrt(theta)",
                "relative_source_L2": "norm_R2(S)/(nu*norm_R2(f)); undefined for zero denominator",
                "units": "physical v,t,x,y; K=one_half_integral_v_squared_f"})
        with runtime.measuring():
            cache = {}
            for index, point in enumerate(points):
                entry = {"index": index, "txy": point, "execution_passed": False, "numerically_uncertain": True, "stage": "moments"}
                fraction = (point[0]-runtime.time_domain[0])/(runtime.time_domain[1]-runtime.time_domain[0])
                entry.update(time_fraction=float(fraction), time_region="early_fraction_le_0_25" if fraction <= .25 else "late_fraction_ge_0_5" if fraction >= .5 else "middle")
                try:
                    integrals, entry["quadrature_by_order"] = [], []
                    for order in orders:
                        value = _integrate(runtime, point[None], order=order, power=1, include_source=True)
                        integrals.append(value)
                        entry["quadrature_by_order"].append({"order": order, **value})
                    entry["moment_fine_minus_coarse"] = integrals[1]["moments"][0]-integrals[0]["moments"][0]
                    entry["source_moment_fine_minus_coarse"] = integrals[1]["source_moments"][0]-integrals[0]["source_moments"][0]
                    entry["source_moments_collision_normalized"] = _ratio(integrals[1]["source_moments"][0],
                        runtime.nu*runtime.background*np.asarray([1., math.sqrt(runtime.temperature), math.sqrt(runtime.temperature), runtime.temperature]))
                    reference = runtime.reference(point)["moments"]
                    entry["represented_macro_moments"] = reference
                    entry["integrated_minus_represented_moments"] = integrals[1]["moments"][0, :4]-reference
                    entry["stage"] = "coordinate_finite_differences"
                    checks = entry["finite_difference_checks"] = []
                    for order, integral in zip(orders, integrals):
                        cache[(order, tuple(point))] = integral["moments"][0]
                        for step in (relative_fd_step, relative_fd_step/2):
                            checks.append({"velocity_order": order, **_balances(runtime, point, integral["moments"][0],
                                integral["source_moments"][0], order, step, cache)})
                    entry["fine_h_minus_coarse_h_balance_by_velocity_order"] = [checks[2*i+1]["balance_residual"]-checks[2*i]["balance_residual"] for i in range(2)]
                    if include_l2:
                        entry["stage"] = "full_velocity_local_L2"
                        norms, entry["L2_by_order"] = [], []
                        for order in l2orders:
                            value = _integrate(runtime, point[None], order=order, power=2, include_source=True)
                            norms.append(value)
                            entry["L2_by_order"].append({"order": order, **value})
                        entry["L2_squared_fine_minus_coarse"] = {name: float(norms[1][name][0]-norms[0][name][0]) for name in ("f_l2_squared", "source_l2_squared")}
                        fnorm = math.sqrt(float(norms[-1]["f_l2_squared"][0]))
                        snorm = math.sqrt(float(norms[-1]["source_l2_squared"][0]))
                        entry["local_velocity_L2_norm_f"] = fnorm
                        entry["local_velocity_L2_norm_source"] = snorm
                        entry["source_L2_collision_background_normalized"] = _ratio(snorm, runtime.nu*runtime.background/math.sqrt(runtime.temperature))
                        entry["source_L2_relative_to_nu_f_L2"] = _ratio(snorm, runtime.nu*fnorm)
                    entry.update(stage="complete", execution_passed=True)
                except Exception as exc:
                    entry["error"] = {"type": type(exc).__name__, "message": str(exc)}
                report["points"].append(_finite(entry))
            report["stage"] = "spatial_mass"
            for time in times:
                for size in grids:
                    entry = {"time": time, "grid_size": size, "execution_passed": False}
                    try:
                        entry.update(_spatial_mass(runtime, time, size, orders), execution_passed=True)
                    except Exception as exc:
                        entry["error"] = {"type": type(exc).__name__, "message": str(exc)}
                    report["spatial_mass"].append(_finite(entry))
            report["evaluation_counts"] = runtime.counts
        successes = sum(row["execution_passed"] for row in report["points"])
        report.update(attempted_point_count=len(points), completed_point_count=successes,
            failed_point_count=len(points)-successes, local_velocity_source_moments_completed=successes == len(points),
            local_velocity_source_L2_completed=include_l2 and successes == len(points),
            independent_spatial_mass_measured=bool(grids and times) and all(row["execution_passed"] for row in report["spatial_mass"]))
        report["execution_passed"] = successes == len(points) and all(row["execution_passed"] for row in report["spatial_mass"])
        report["stage"] = "complete" if report["execution_passed"] else "partial_failure"
        if not include_l2:
            report["pending"].append("full_velocity_local_source_L2")
        if not grids or not times:
            report["pending"].append("spatial_mean_mass_crosscheck")
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    report = _finite(report)
    json.dumps(report, allow_nan=False)
    return report
