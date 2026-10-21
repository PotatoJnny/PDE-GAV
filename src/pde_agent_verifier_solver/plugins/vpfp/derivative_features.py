"""Ordered physical-coordinate derivative probes of complete VPFP answers.

This is an independent finite-feature audit, not a score loss or a Sobolev-law
certificate. Time differentiation is with respect to physical t, never the
diffusion noise level sigma. All velocity moments are exact full-space squared
Hermite moments of the pinned continuous decoder.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path

import numpy as np

from .diffusion_features import _check_conditions
from .perturbation_backend import _legacy_api


_QUANTITIES = ("f", "rho", "momentum_x", "momentum_y", "kinetic_energy")
_COMPONENTS = ("value", "dt", "dx", "dy")
_UNITS = {"f": "density/velocity^2", "rho": "density",
          "momentum_x": "density*velocity", "momentum_y": "density*velocity",
          "kinetic_energy": "density*velocity^2"}


@dataclass(frozen=True)
class VPFPDerivativeFeaturePlan:
    # Several midpoints of the production 33-control representation, including
    # its first interval. Endpoints retain boundary/initial-value information.
    time_fractions: tuple[float, ...] = (0., .015625, .046875, .140625, .265625, .515625, .765625, 1.)
    macro_probe_count: int = 8
    phase_probe_count: int = 8
    probe_seed: int = 2026090417
    chunk_size: int = 64
    tail_quantiles: tuple[float, ...] = (.9, .99)

    def validate(self, config):
        for name in ("macro_probe_count", "phase_probe_count", "probe_seed", "chunk_size"):
            value = getattr(self, name)
            minimum = 0 if name == "probe_seed" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        times = np.asarray(self.time_fractions)
        if (times.dtype.kind not in "fiu" or times.ndim != 1 or len(times) < 3
                or not np.isfinite(times).all() or times[0] != 0. or times[-1] != 1.
                or not np.all(np.diff(times) > 0)):
            raise ValueError("time fractions must strictly increase from 0 to 1 and include interior times")
        domain = np.asarray(config.time_domain, dtype=float)
        lengths = np.asarray(config.equation.x_domain, dtype=float)
        if domain.shape != (2,) or not np.isfinite(domain).all() or domain[1] <= domain[0]:
            raise ValueError("physical time domain must be finite and increasing")
        if lengths.shape != (2,) or not np.isfinite(lengths).all() or np.any(lengths <= 0):
            raise ValueError("periodic spatial domain lengths must be finite and positive")
        if config.structured_time_basis == "piecewise_cubic_hermite_v1":
            controls = config.structured_time_control_count
            if controls < 2:
                raise ValueError("piecewise representation requires at least two controls")
            locations = times[1:-1] * (controls - 1)
            if np.isclose(locations, np.rint(locations), atol=1.e-10, rtol=0.).all():
                raise ValueError("derivative probes must include a control-interval interior, not only nodes")
        quantiles = np.asarray(self.tail_quantiles)
        if (quantiles.dtype.kind not in "fiu" or quantiles.ndim != 1 or not len(quantiles)
                or not np.isfinite(quantiles).all() or np.any(quantiles <= 0.)
                or np.any(quantiles >= 1.) or np.any(np.diff(quantiles) <= 0)):
            raise ValueError("tail quantiles must strictly increase inside (0,1)")

    @property
    def digest(self):
        return sha256(json.dumps(asdict(self), sort_keys=True, allow_nan=False).encode()).hexdigest()


def _require_compute(parameter):
    if parameter.device.type == "cuda" and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("CUDA physical-derivative extraction requires a Slurm compute allocation")


def _probe_points(config, plan, dtype):
    random = np.random.default_rng(plan.probe_seed)
    times = config.time_domain[0] + np.asarray(plan.time_fractions) * (config.time_domain[1] - config.time_domain[0])
    macro_space = random.uniform(size=(plan.macro_probe_count, 2)) * np.asarray(config.equation.x_domain)
    phase_space = random.uniform(size=(plan.phase_probe_count, 2)) * np.asarray(config.equation.x_domain)
    velocity = random.normal(size=(plan.phase_probe_count, 2)) * np.where(
        np.arange(plan.phase_probe_count)[:, None] % 2, 3., 1.)

    def product(space, velocity_values):
        spatial_velocity = np.column_stack((space, velocity_values))
        return np.asarray(np.column_stack((np.repeat(times, len(space)),
            np.tile(spatial_velocity, (len(times), 1)))), dtype=dtype)

    return product(macro_space, np.zeros_like(macro_space)), product(phase_space, velocity)


def _point_jets(model, points, latent, condition, *, macro, macros):
    """Coordinate autodiff, valid because this pinned decoder is pointwise.

    Summing each scalar output over query rows extracts all diagonal point
    derivatives in one backward pass. There is no cross-query normalization.
    No parameter .grad or second-order graph is accumulated.
    """
    torch = model.torch
    parameter = next(model.parameters())
    with torch.enable_grad():
        query = torch.tensor(points, dtype=parameter.dtype, device=parameter.device, requires_grad=True)
        fields = model.fields(query, latent, equation_condition=condition)
        rho = fields["rho"]
        if bool(torch.any(~torch.isfinite(rho)).item()) or bool(torch.any(rho <= 0.).item()):
            raise ValueError("sampled physical density positivity/finite hard gate failed")
        if macro:
            moments = macros.torch_squared_hermite_moments(
                torch, fields["hermite_coefficients"], modes=model.hermite_modes,
                temperature=fields["equation_condition"][:, 1],
                floor_mixture=model.config.velocity_floor_mixture)
            quantities = {"rho": rho, "momentum_x": rho * moments["mean_velocity"][:, 0],
                          "momentum_y": rho * moments["mean_velocity"][:, 1],
                          "kinetic_energy": rho * moments["kinetic_energy"]}
        else:
            quantities = {"f": fields["f"]}
        result = {}
        for name, values in quantities.items():
            if tuple(values.shape) != (len(points),) or not bool(torch.isfinite(values).all().item()):
                raise ValueError(f"invalid decoded {name} values")
            if name in ("f", "kinetic_energy") and bool(torch.any(values < 0.).item()):
                raise ValueError(f"sampled {name} nonnegativity hard gate failed")
            if not values.requires_grad:
                raise ValueError(f"physical-coordinate derivative graph is disconnected for {name}")
            gradient = torch.autograd.grad(values.sum(), query, retain_graph=True,
                                           create_graph=False, allow_unused=False)[0]
            joined = torch.cat((values[:, None], gradient[:, :3]), dim=1)
            if not bool(torch.isfinite(joined).all().item()):
                raise ValueError(f"nonfinite physical derivative for {name}")
            result[name] = joined.detach().cpu().numpy()
        return result


def _joint_features(jets, plan):
    """Keep ordered values and derivatives BEFORE optional marginal summaries."""
    if set(jets) != set(_QUANTITIES):
        raise ValueError("all five physical quantities are required")
    descriptors, parts = [], []
    for name in _QUANTITIES:
        array = np.asarray(jets[name])
        if array.dtype.kind not in "fiu" or array.ndim != 2 or array.shape[1] != 4 or not len(array) or not np.isfinite(array).all():
            raise ValueError(f"invalid ordered derivative feature array: {name}")
        for probe in range(len(array)):
            for axis, component in enumerate(_COMPONENTS):
                unit = _UNITS[name] if axis == 0 else f"({_UNITS[name]})/{'time' if axis == 1 else 'length'}"
                descriptors.append({"name": f"{name}.probe{probe}.{component}",
                    "group": "physical_value_derivative_joint", "unit": unit, "scale_floor": 1.e-6})
        parts.extend(array.reshape(-1).tolist())
    joint_count = len(parts)
    for name in _QUANTITIES:
        array = np.asarray(jets[name], dtype=np.float64)
        for axis, component in enumerate(_COMPONENTS):
            unit = _UNITS[name] if axis == 0 else f"({_UNITS[name]})/{'time' if axis == 1 else 'length'}"
            # Scale before squaring to avoid a gratuitous overflow in RMS.
            maximum = float(np.max(np.abs(array[:, axis])))
            rms = 0. if maximum == 0. else maximum * float(np.sqrt(np.mean((array[:, axis] / maximum)**2)))
            summaries = [("rms", rms)] + [(f"abs_q{q:g}", float(np.quantile(np.abs(array[:, axis]), q))) for q in plan.tail_quantiles]
            for label, value in summaries:
                descriptors.append({"name": f"{name}.{component}.{label}",
                    "group": "physical_derivative_marginal_summary", "unit": unit, "scale_floor": 1.e-6})
                parts.append(value)
    row = np.asarray(parts, dtype=np.float64)
    if not np.isfinite(row).all():
        raise ValueError("nonfinite derivative feature summary")
    return row, tuple(descriptors), joint_count


def extract_vpfp_derivative_features(model, latents, conditions, *, plan=None):
    """Audit complete production FP32 answers at fixed physical probes.

    Higher-precision caller arrays are explicitly rounded to float32 before
    either hashing or evaluating an answer. Probe coordinates and coordinate
    derivatives use the physical model's float32/float64 precision. CPU tiny
    fixtures are supported; CUDA execution requires an existing Slurm job.
    """
    plan = plan or VPFPDerivativeFeaturePlan()
    if not isinstance(plan, VPFPDerivativeFeaturePlan):
        raise TypeError("plan must be VPFPDerivativeFeaturePlan")
    cfg, torch = model.config, model.torch
    plan.validate(cfg)
    parameter = next(model.parameters())
    _require_compute(parameter)
    if parameter.dtype not in (torch.float32, torch.float64):
        raise ValueError("physical coordinate derivatives require float32 or float64 model precision")
    if torch.is_inference_mode_enabled():
        raise ValueError("physical derivatives cannot be extracted under torch.inference_mode")
    values, conditions = np.asarray(latents), np.asarray(conditions)
    if (values.dtype.kind not in "fiu" or conditions.dtype.kind not in "fiu"
            or values.ndim != 2 or values.shape[1] != cfg.latent_dim or not len(values)
            or conditions.shape != (len(values), 2)):
        raise ValueError("complete real coefficient rows and aligned nu/temperature rows are required")
    _check_conditions(conditions, cfg)
    _legacy_api()  # Reject changed pinned physical module bytes.
    analytic = importlib.import_module("agent_verifier.vpfp.analytic_source")
    macros = importlib.import_module("agent_verifier.vpfp.continuous_macros")
    hermite = importlib.import_module("agent_verifier.vpfp.continuous_hermite")
    analytic.require_analytic_structured_source(model)
    dtype = np.float32 if parameter.dtype == torch.float32 else np.float64
    macro_points, phase_points = _probe_points(cfg, plan, dtype)
    rows, descriptors, cases = [], None, []
    was_training = model.training
    model.eval()
    try:
        with analytic.ieee_float32_coordinate_matmul(model):
            for raw_latent, raw_condition in zip(values, conditions):
                with np.errstate(over="ignore", invalid="ignore"):
                    latent = np.array(raw_latent, dtype="<f4", copy=True)
                    condition = np.array(raw_condition, dtype="<f4", copy=True)
                if not np.isfinite(latent).all():
                    raise ValueError("nonfinite canonical float32 coefficients")
                _check_conditions(condition[None], cfg)
                digest = sha256(latent.tobytes() + condition.tobytes()).hexdigest()
                latent.flags.writeable = False
                condition.flags.writeable = False
                # torch.tensor explicitly copies the immutable NumPy snapshot;
                # legacy as_tensor therefore never aliases read-only storage.
                latent_tensor = torch.tensor(latent, dtype=parameter.dtype, device=parameter.device)
                condition_tensor = torch.tensor(condition, dtype=parameter.dtype, device=parameter.device)
                chunks = {name: [] for name in _QUANTITIES}
                for macro, probes in ((False, phase_points), (True, macro_points)):
                    for start in range(0, len(probes), plan.chunk_size):
                        block = _point_jets(model, probes[start:start + plan.chunk_size],
                            latent_tensor, condition_tensor, macro=macro, macros=macros)
                        for name, array in block.items():
                            chunks[name].append(array)
                jets = {name: np.concatenate(parts) for name, parts in chunks.items()}
                row, names, joint_count = _joint_features(jets, plan)
                if descriptors is not None and descriptors != names:
                    raise RuntimeError("ordered derivative feature schema changed between trajectories")
                descriptors = names
                rows.append(row)
                cases.append({"input_answer_sha256": digest, "input_equation_condition": condition.tolist(),
                    "minimum_sampled_density": float(jets["rho"][:, 0].min()),
                    "minimum_sampled_f": float(jets["f"][:, 0].min())})
    finally:
        model.train(was_training)
    return np.stack(rows), descriptors, {
        "schema": "vpfp_physical_coordinate_derivative_features_v1", "admitted": False,
        "status": "partial_finite_feature_audit", "plan": asdict(plan), "plan_sha256": plan.digest,
        "coordinate_derivative_backend": "torch_pointwise_coordinate_autograd_exact_Hermite_moments",
        "coordinate_order": ["physical_t", "physical_x", "physical_y"],
        "time_derivative": "d/d physical_t; not d/d normalized_tau or diffusion_sigma",
        "time_domain": list(cfg.time_domain), "spatial_domain_lengths": list(cfg.equation.x_domain),
        "decoder_config": cfg.to_dict(),
        "decoder_config_sha256": sha256(json.dumps(cfg.to_dict(), sort_keys=True, allow_nan=False).encode()).hexdigest(),
        "macro_points_txy": macro_points[:, :3].tolist(), "phase_points_txyvv": phase_points.tolist(),
        "probe_order": "time_major_then_fixed_probe_index", "joint_component_order": list(_COMPONENTS),
        "quantity_order": list(_QUANTITIES), "joint_feature_count": joint_count,
        "marginal_summaries_replace_joint_features": False,
        "input_answer_hash_encoding": "sha256(little_endian_float32_full_coefficients || little_endian_float32_nu_temperature)",
        "input_answer_snapshot_precision": "canonical_float32_even_for_higher_precision_caller_arrays",
        "coordinate_precision": str(parameter.dtype), "per_case": cases,
        "physical_module_sha256": {name: sha256(Path(module.__file__).read_bytes()).hexdigest()
                                   for name, module in (("continuous_macros", macros), ("analytic_source", analytic),
                                                       ("continuous_hermite", hermite))},
        "statistical_unit": "one_complete_trajectory_per_row", "score_loss_integration": False,
        "global_positivity_certificate": False, "high_fidelity_validation": False,
        "sobolev_norm_or_distribution_equivalence_established": False,
        "limitations": ["finite fixed probes do not determine the full function or its Sobolev law",
                        "probe RMS and tails are descriptive, not converged domain-integral norms",
                        "boundary time derivatives use the represented endpoint/one-sided in-domain derivative",
                        "no physical equivalence margins, power or resolution convergence established"],
    }


__all__ = ["VPFPDerivativeFeaturePlan", "extract_vpfp_derivative_features"]
