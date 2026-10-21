"""Physical, whole-answer features for the VPFP diffusion research audit.

No learned embedding or per-mode phase quotient is fitted here. These finite
features diagnose the declared numerical-answer distribution, not equality of
infinite-dimensional laws or fidelity to experimental plasma observations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import importlib
import json
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class VPFPFeaturePlan:
    grid_size: int = 17
    time_fractions: tuple[float, ...] = (0., .015625, .0625, .125, .25, .5, .75, 1.)
    phase_probe_count: int = 64
    probe_seed: int = 2026090407
    chunk_size: int = 512

    def validate(self, rho_max_mode=6, local_max_mode=4):
        for name in ("grid_size", "phase_probe_count", "probe_seed", "chunk_size"):
            value = getattr(self, name)
            minimum = 0 if name == "probe_seed" else 1
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.grid_size < 2 * max(rho_max_mode, local_max_mode) + 1:
            raise ValueError("feature grid aliases the represented spatial modes")
        times = np.asarray(self.time_fractions, dtype=np.float64)
        if (times.ndim != 1 or not times.size or not np.isfinite(times).all()
                or np.any(times < 0) or np.any(times > 1) or np.any(np.diff(times) <= 0)):
            raise ValueError("feature times must be strictly increasing points in the declared horizon")
        if self.phase_probe_count < 4:
            raise ValueError("invalid physical feature probe settings")

    @property
    def digest(self):
        return sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def _descriptor(name, group, unit):
    return {"name": name, "group": group, "unit": unit, "scale_floor": 1.e-6}


def _check_conditions(conditions, cfg):
    if not np.isfinite(conditions).all():
        raise ValueError("feature input is nonfinite")
    if np.any(conditions[:, 0] < 0) or np.any(conditions[:, 1] <= 0):
        raise ValueError("physical conditions require nonnegative nu and positive temperature")
    for axis, bounds in enumerate((cfg.collision_nu_range, cfg.temperature_range)):
        # A float32 condition can round just beyond a double configured
        # endpoint; permit only that endpoint's actual float32 encoding.
        encoded = np.asarray(bounds, dtype=conditions.dtype if conditions.dtype.kind == "f" else np.float64)
        if np.any(conditions[:, axis] < encoded[0]) or np.any(conditions[:, axis] > encoded[1]):
            raise ValueError("feature conditions are outside the decoder's declared regime")


def summarize_continuous_fields(
    fields: Mapping[str, np.ndarray], amplitude: np.ndarray,
    hermite_modes: Sequence[tuple[int, int]], plan: VPFPFeaturePlan,
):
    """Summarize actual decoded fields on each complete trajectory's grid.

    The bispectra preserve cross-mode phase coupling while respecting a common
    physical translation; individual Fourier phases are not independently erased.
    """
    plan.validate(rho_max_mode=0, local_max_mode=0)
    expected = (len(plan.time_fractions), plan.grid_size, plan.grid_size)
    required = {"rho", "momentum_x", "momentum_y", "kinetic_energy", "electric_x", "electric_y"}
    if set(fields) != required:
        raise ValueError("all declared VPFP macro and field quantities are required")
    if any(np.shape(value) != expected or np.asarray(value).dtype.kind not in "fiu"
           or not np.isfinite(value).all() for value in fields.values()):
        raise ValueError("physical fields have invalid shapes or nonfinite values")
    if np.any(np.asarray(fields["rho"]) <= 0):
        raise ValueError("physical density positivity hard gate failed")
    if np.any(np.asarray(fields["kinetic_energy"]) < 0):
        raise ValueError("physical kinetic-energy nonnegativity hard gate failed")
    if (len(hermite_modes) == 0 or any(len(pair) != 2 or any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0
            for value in pair) for pair in hermite_modes)
            or len(set(tuple(pair) for pair in hermite_modes)) != len(hermite_modes)):
        raise ValueError("Hermite modes must be distinct nonnegative integer pairs")
    if np.asarray(amplitude).dtype.kind not in "fiu":
        raise ValueError("normalized Hermite amplitude must be real numeric values")
    amplitude = np.asarray(amplitude, dtype=np.float64)
    if amplitude.shape != (*expected, len(hermite_modes)) or not np.isfinite(amplitude).all():
        raise ValueError("normalized Hermite amplitude has invalid shape or values")
    names, values = [], []

    def add(name, group, unit, value):
        if not np.isfinite(value):
            raise ValueError(f"nonfinite feature: {name}")
        names.append(_descriptor(name, group, unit))
        values.append(float(value))

    frequencies = np.fft.fftfreq(plan.grid_size) * plan.grid_size
    kx, ky = np.meshgrid(frequencies, frequencies, indexing="ij")
    radius = kx * kx + ky * ky
    units = {"rho": "density", "momentum_x": "density*velocity", "momentum_y": "density*velocity",
             "kinetic_energy": "density*velocity^2", "electric_x": "acceleration", "electric_y": "acceleration"}
    for name in sorted(fields):
        data = np.asarray(fields[name], dtype=np.float64)
        unit = units[name]
        for index, frame in enumerate(data):
            prefix = f"{name}.time{index}"
            centered = frame - frame.mean()
            for label, value in (("mean", frame.mean()), ("std", frame.std()),
                                  ("rms", np.sqrt(np.mean(frame * frame))),
                                  ("minimum", frame.min()), ("maximum", frame.max())):
                add(f"{prefix}.{label}", "macros", unit, value)
            add(f"{prefix}.third_central_moment", "phase_coupling", f"({unit})^3", np.mean(centered**3))
            spectrum = np.fft.fft2(frame) / frame.size
            for band, mask in (("low", (radius > 0) & (radius <= 4)),
                               ("middle", (radius > 4) & (radius <= 16)),
                               ("high", radius > 16)):
                add(f"{prefix}.spectrum_{band}", "spatial_spectra", f"({unit})^2",
                    np.square(np.abs(spectrum[mask])).sum())
            if name in {"rho", "kinetic_energy"} and plan.grid_size >= 7:
                for axis, first, second, third in (
                    ("x112", (1, 0), (1, 0), (2, 0)), ("x123", (1, 0), (2, 0), (3, 0)),
                    ("y112", (0, 1), (0, 1), (0, 2)), ("y123", (0, 1), (0, 2), (0, 3)),
                    ("xy", (1, 0), (0, 1), (1, 1)),
                ):
                    value = spectrum[first] * spectrum[second] * spectrum[third].conjugate()
                    add(f"{prefix}.bispectrum_{axis}_real", "phase_coupling", f"({unit})^3", value.real)
                    add(f"{prefix}.bispectrum_{axis}_imag", "phase_coupling", f"({unit})^3", value.imag)
        for index in range(1, len(data)):
            difference = data[index] - data[index - 1]
            add(f"{name}.time_difference{index}.rms", "temporal_shapes", unit,
                np.sqrt(np.mean(difference * difference)))
            add(f"{name}.cross_time{index}.mean_product", "temporal_shapes", f"({unit})^2",
                np.mean(data[index] * data[index - 1]))
    degrees = np.asarray([sum(pair) for pair in hermite_modes])
    for index, frame in enumerate(amplitude):
        for degree in sorted(set(degrees.tolist())):
            add(f"amplitude_degree{degree}.time{index}.mean_energy", "hermite_spectra", "dimensionless",
                np.square(frame[..., degrees == degree]).sum(axis=-1).mean())
    return np.asarray(values), tuple(names)


def extract_vpfp_features(model, latents, conditions, *, plan=None, progress=None):
    """Evaluate fixed physical features with the pinned decoder on a compute node.

    Macro velocity integrals use exact squared-Hermite recurrences. Spatial
    quadrature of nonlinear quantities still needs independent refinement; a
    band-resolving grid alone is not a convergence certificate.

    This audit targets production float32 coefficients/conditions. Each whole
    trajectory is copied to a read-only canonical float32 snapshot before any
    field/source call. The per-case input hash identifies that SAME snapshot.
    Higher-precision caller arrays are explicitly rounded to float32 here;
    this function is not a precision-preserving float64 answer audit.
    """
    plan = plan or VPFPFeaturePlan()
    cfg = model.config
    plan.validate(max(cfg.rho_max_mode_x, cfg.rho_max_mode_y), cfg.structured_local_spatial_mode)
    values = np.asarray(latents)
    conditions = np.asarray(conditions)
    if values.ndim != 2 or values.shape[1] != cfg.latent_dim or conditions.shape != (len(values), 2):
        raise ValueError("feature extraction requires complete latents and their physical conditions")
    if values.dtype.kind not in "fiu" or conditions.dtype.kind not in "fiu":
        raise ValueError("feature input must contain real numeric coefficients and conditions")
    _check_conditions(conditions, cfg)
    torch = model.torch
    macros = importlib.import_module("agent_verifier.vpfp.continuous_macros")
    analytic = importlib.import_module("agent_verifier.vpfp.analytic_source")
    times = cfg.time_domain[0] + np.asarray(plan.time_fractions) * (cfg.time_domain[1] - cfg.time_domain[0])
    x = np.arange(plan.grid_size) * cfg.equation.x_domain[0] / plan.grid_size
    y = np.arange(plan.grid_size) * cfg.equation.x_domain[1] / plan.grid_size
    txy = np.stack(np.meshgrid(times, x, y, indexing="ij"), axis=-1).reshape(-1, 3)
    phase_grid = np.column_stack((txy, np.zeros((len(txy), 2))))
    rng = np.random.default_rng(plan.probe_seed)
    probes = np.column_stack((
        rng.uniform(*cfg.time_domain, size=plan.phase_probe_count),
        rng.uniform(0, cfg.equation.x_domain[0], size=plan.phase_probe_count),
        rng.uniform(0, cfg.equation.x_domain[1], size=plan.phase_probe_count),
        rng.normal(size=(plan.phase_probe_count, 2)) * np.where(
            np.arange(plan.phase_probe_count)[:, None] % 2, 3., 1.),
    ))
    rows, descriptors, sanity = [], None, []
    grid_shape = (len(times), plan.grid_size, plan.grid_size)
    model.eval()
    with torch.no_grad(), analytic.ieee_float32_coordinate_matmul(model):
        for index, (latent, condition) in enumerate(zip(values, conditions)):
            # Scan one trajectory at a time: a full corpus boolean temporary
            # would defeat the memory-mapped collection/training boundary.
            with np.errstate(over="ignore", invalid="ignore"):
                latent = np.array(latent, dtype="<f4", copy=True)
                condition = np.array(condition, dtype="<f4", copy=True)
            if not np.isfinite(latent).all():
                raise ValueError("feature input is nonfinite")
            _check_conditions(condition[None], cfg)
            input_digest = sha256(latent.tobytes() + condition.tobytes()).hexdigest()
            latent.flags.writeable = False
            condition.flags.writeable = False
            accumulated = {key: [] for key in ("rho", "momentum_x", "momentum_y", "kinetic_energy", "electric_x", "electric_y", "amplitude")}
            for start in range(0, len(phase_grid), plan.chunk_size):
                fields = model.fields(phase_grid[start:start + plan.chunk_size], latent, equation_condition=condition)
                coeff = fields["hermite_coefficients"]
                moment = macros.torch_squared_hermite_moments(
                    torch, coeff, modes=model.hermite_modes, temperature=fields["equation_condition"][:, 1],
                    floor_mixture=cfg.velocity_floor_mixture)
                density = fields["rho"].reshape(-1)
                arrays = {"rho": density, "momentum_x": density * moment["mean_velocity"][:, 0],
                    "momentum_y": density * moment["mean_velocity"][:, 1],
                    "kinetic_energy": density * moment["kinetic_energy"],
                    "electric_x": fields["electric"][:, 0], "electric_y": fields["electric"][:, 1], "amplitude": coeff}
                for key, array in arrays.items():
                    accumulated[key].append(array.detach().cpu().numpy())
            amplitude = np.concatenate(accumulated.pop("amplitude")).reshape(*grid_shape, -1)
            sampled_fields = {key: np.concatenate(chunks).reshape(grid_shape) for key, chunks in accumulated.items()}
            row, description = summarize_continuous_fields(sampled_fields, amplitude, model.hermite_modes, plan)
            source = analytic.manufacture_standard_vpfp_source_analytic(model, probes, latent, equation_condition=condition)
            source_values = source["source"].detach().cpu().numpy().reshape(-1)
            f_values = source["f"].detach().cpu().numpy().reshape(-1)
            if not np.isfinite(source_values).all() or not np.isfinite(f_values).all():
                raise ValueError("generated/reference continuous answer or induced source is nonfinite")
            if source_values.shape != (plan.phase_probe_count,) or f_values.shape != (plan.phase_probe_count,):
                raise ValueError("analytic source returned unexpected phase-probe shapes")
            if np.any(f_values < 0):
                raise ValueError("continuous distribution nonnegativity hard gate failed")
            extras = np.concatenate((f_values, source_values))
            extra_description = tuple(_descriptor(f"f.probe{i}", "kinetic_point_values", "density/velocity^2") for i in range(len(f_values)))
            extra_description += tuple(_descriptor(f"induced_source.probe{i}", "source_shape", "density/(velocity^2*time)") for i in range(len(source_values)))
            description += extra_description
            if descriptors is not None and descriptors != description:
                raise RuntimeError("physical feature ordering changed between trajectories")
            descriptors = description
            rows.append(np.concatenate((row, extras)))
            sanity.append({"input_answer_sha256": input_digest,
                "input_equation_condition": condition.tolist(),
                "minimum_density": float(sampled_fields["rho"].min()),
                "minimum_sampled_f": float(f_values.min()), "source_rms": float(np.sqrt(np.mean(source_values**2))),
                "source_maximum_absolute": float(np.max(np.abs(source_values))),
                "sampled_amplitude_normalization_max_error": float(np.max(np.abs(np.sum(amplitude**2, axis=-1) - 1.)))})
            if progress:
                progress(index + 1, len(values))
    if not rows:
        raise ValueError("at least one complete answer is required")
    return np.stack(rows), descriptors, {"schema": "vpfp_complete_answer_physical_features_v1",
        "plan": asdict(plan), "plan_sha256": plan.digest, "phase_points": probes.tolist(),
        "input_answer_hash_encoding": "sha256(little_endian_float32_full_coefficients || little_endian_float32_nu_temperature)",
        "input_answer_snapshot_precision": "canonical_float32_even_for_higher_precision_caller_arrays",
        "spatial_quadrature_convergence_qualified": False, "high_fidelity_validation": False,
        "sampled_density_positivity_passed": True, "sampled_f_nonnegativity_passed": True,
        "positivity_probe_scope": "declared grids and phase probes only, not a global numerical certificate",
        "amplitude_spectrum_meaning": "normalized squared-Hermite amplitude energy, not the spectrum of f",
        "source_meaning": "same-answer induced source, never residual to source-free VPFP",
        "per_case": sanity}


__all__ = ["VPFPFeaturePlan", "summarize_continuous_fields", "extract_vpfp_features"]
