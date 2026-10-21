from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from agent_verifier.neural_generator.structured_ns_answer_first import (
    ContinuousStructuredNSMMS,
    StructuredNSCaseCoefficients,
    StructuredNSProjectionConfig,
)


PERTURBATION_FAMILIES = (
    "pca_tangent",
    "pca_normal",
    "spectral_complement",
    "spline_refinement",
)


def _require_torch():
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("PyTorch is required for structured NS robustness") from exc
    return torch


@dataclass(frozen=True)
class StructuredNSPerturbation:
    family: str
    config: StructuredNSProjectionConfig
    base_coefficients: Any
    direction_coefficients: Any
    metadata: dict[str, Any]


def coefficient_tensor_from_field(field: ContinuousStructuredNSMMS):
    """Return the canonical `[2,R,Kx,Ky]` coefficient tensor."""

    torch = _require_torch()
    config = field.config
    shape = (
        2,
        int(config.time_order) + 1,
        int(config.spatial_mode_x) + 1,
        int(config.spatial_mode_y) + 1,
    )
    return torch.stack(
        (field.coefficients.psi, field.coefficients.pressure),
        dim=0,
    ).reshape(shape)


def make_perturbation(
    field: ContinuousStructuredNSMMS,
    *,
    family: str,
    seed: int,
    pca=None,
    spectral_max_mode: int = 20,
) -> StructuredNSPerturbation:
    if family not in PERTURBATION_FAMILIES:
        raise ValueError(f"unsupported perturbation family: {family!r}")
    rng = np.random.default_rng(int(seed))
    if family == "spline_refinement":
        return _make_spline_refinement(field, rng=rng, seed=int(seed))
    if family == "pca_tangent":
        if pca is None:
            raise ValueError("pca_tangent requires a PCA checkpoint")
        return _make_pca_tangent(field, pca=pca, rng=rng, seed=int(seed))
    if family == "pca_normal":
        if pca is None:
            raise ValueError("pca_normal requires a PCA checkpoint")
        return _make_pca_normal(field, pca=pca, rng=rng, seed=int(seed))
    return _make_spectral_complement(
        field,
        rng=rng,
        seed=int(seed),
        spectral_max_mode=int(spectral_max_mode),
    )


def _pca_scale_and_components(pca, *, dtype, device):
    scale = pca.field_scale.to(device=device, dtype=dtype)
    components = pca.components.to(device=device, dtype=dtype)
    return scale, components


def _validate_pca_shape(field, pca) -> tuple[int, ...]:
    shape = tuple(int(value) for value in coefficient_tensor_from_field(field).shape)
    if tuple(int(value) for value in pca.coefficient_shape) != shape:
        raise ValueError(
            f"PCA coefficient shape {pca.coefficient_shape} does not match {shape}"
        )
    return shape


def _make_pca_tangent(field, *, pca, rng, seed: int) -> StructuredNSPerturbation:
    torch = _require_torch()
    base = coefficient_tensor_from_field(field).detach().cpu().double()
    shape = _validate_pca_shape(field, pca)
    direction_z = torch.as_tensor(
        rng.standard_normal(int(pca.rank)),
        dtype=torch.float64,
    )
    direction_z = direction_z / direction_z.norm().clamp_min(1.0e-15)
    scale, components = _pca_scale_and_components(
        pca,
        dtype=torch.float64,
        device="cpu",
    )
    score_scale = pca.score_scale.detach().cpu().double()
    balanced = (direction_z * score_scale) @ components
    direction = balanced.reshape(1, *shape) * scale
    direction = direction[0]
    return StructuredNSPerturbation(
        family="pca_tangent",
        config=field.config,
        base_coefficients=base,
        direction_coefficients=direction,
        metadata={
            "seed": int(seed),
            "definition": "unit_random_whitened_score_direction_decoded_through_pca",
            "pca_rank": int(pca.rank),
            "whitened_score_direction_l2": float(direction_z.norm()),
            "balanced_direction_l2": float(balanced.norm()),
        },
    )


def _make_pca_normal(field, *, pca, rng, seed: int) -> StructuredNSPerturbation:
    torch = _require_torch()
    base = coefficient_tensor_from_field(field).detach().cpu().double()
    shape = _validate_pca_shape(field, pca)
    dimension = int(np.prod(shape))
    raw = torch.zeros(dimension, dtype=torch.float64)
    psi_dimension = int(np.prod(shape[1:]))
    raw[:psi_dimension] = torch.as_tensor(
        rng.standard_normal(psi_dimension),
        dtype=torch.float64,
    )
    scale, components = _pca_scale_and_components(
        pca,
        dtype=torch.float64,
        device="cpu",
    )
    balanced = raw - components.T @ (components @ raw)
    # A second pass removes float32 checkpoint roundoff from the projection.
    balanced = balanced - components.T @ (components @ balanced)
    balanced = balanced / balanced.norm().clamp_min(1.0e-15)
    orthogonality = (components @ balanced).norm()
    direction = (balanced.reshape(1, *shape) * scale)[0]
    return StructuredNSPerturbation(
        family="pca_normal",
        config=field.config,
        base_coefficients=base,
        direction_coefficients=direction,
        metadata={
            "seed": int(seed),
            "definition": "psi_seeded_random_balanced_coefficient_direction_projected_out_of_pca_span",
            "pca_rank": int(pca.rank),
            "ambient_dimension": int(dimension),
            "normal_dimension": int(dimension - int(pca.rank)),
            "balanced_direction_l2": float(balanced.norm()),
            "pca_projection_l2": float(orthogonality),
        },
    )


def _make_spectral_complement(
    field,
    *,
    rng,
    seed: int,
    spectral_max_mode: int,
) -> StructuredNSPerturbation:
    if getattr(field.config, "spatial_basis", "cosine") != "cosine":
        raise ValueError("spectral_complement requires cosine space; select an explicit compatible perturbation family")
    torch = _require_torch()
    old = coefficient_tensor_from_field(field).detach().cpu().double()
    old_x = int(field.config.spatial_mode_x)
    old_y = int(field.config.spatial_mode_y)
    if spectral_max_mode <= max(old_x, old_y):
        raise ValueError("spectral_max_mode must exceed the represented spatial band")
    if int(field.config.time_order) < 2:
        raise ValueError("tau-squared spectral perturbations require time_order >= 2")
    config = replace(
        field.config,
        spatial_mode_x=int(spectral_max_mode),
        spatial_mode_y=int(spectral_max_mode),
    )
    shape = (
        2,
        int(config.time_order) + 1,
        int(config.spatial_mode_x) + 1,
        int(config.spatial_mode_y) + 1,
    )
    base = torch.zeros(shape, dtype=torch.float64)
    base[
        :,
        : old.shape[1],
        : old.shape[2],
        : old.shape[3],
    ] = old
    kx, ky = np.meshgrid(
        np.arange(shape[2]),
        np.arange(shape[3]),
        indexing="ij",
    )
    shell = (kx > old_x) | (ky > old_y)
    spatial = np.zeros((shape[2], shape[3]), dtype=np.float64)
    spatial[shell] = rng.standard_normal(int(np.count_nonzero(shell)))
    spatial /= max(float(np.linalg.norm(spatial)), 1.0e-15)
    direction = torch.zeros(shape, dtype=torch.float64)
    spatial_tensor = torch.as_tensor(spatial, dtype=torch.float64)
    # tau^2 = 3/8 T_0 + 1/2 T_1 + 1/8 T_2.  The perturbation and
    # its first time derivative therefore vanish at the initial time.
    if getattr(config, "temporal_basis", "chebyshev") == "chebyshev":
        # Preserve the original arithmetic and coefficient normalization exactly.
        for order, weight in enumerate((3.0 / 8.0, 1.0 / 2.0, 1.0 / 8.0)):
            direction[0, order] = float(weight) * spatial_tensor
    else:
        direction[0] = torch.from_numpy(_tau_squared_coefficients(config))[:, None, None] * spatial_tensor[None]
    direction = direction / direction.norm().clamp_min(1.0e-15)
    low_band_leakage = direction[
        :, :, : old.shape[2], : old.shape[3]
    ].abs().max()
    return StructuredNSPerturbation(
        family="spectral_complement",
        config=config,
        base_coefficients=base,
        direction_coefficients=direction,
        metadata={
            "seed": int(seed),
            "definition": "streamfunction_spatial_cosine_complement_with_tau_squared_envelope",
            "represented_spatial_modes": [int(old_x), int(old_y)],
            "perturbed_spatial_mode_max": int(spectral_max_mode),
            "shell_coefficient_count": int(np.count_nonzero(shell)),
            "time_envelope": "tau_squared",
            "initial_streamfunction_and_time_derivative_vanish": True,
            "low_band_coefficient_leakage_max_abs": float(low_band_leakage),
            **({"temporal_basis": "bspline", "time_polynomial_representation": "exact_bspline_blossom",
                "amplitude_convention": "l2_normalized_joint_streamfunction_coefficients",
                "amplitude_equivalent_to_cosine": False}
               if getattr(config, "temporal_basis", "chebyshev") == "bspline" else {}),
            **({"cosine_expansion_extension": float(config.spatial_extension),
                "cosine_expansion_domain_role": "basis_only_physical_measurement_domain_unchanged"}
               if config.spatial_extension else {}),
        },
    )


def _midpoint_spline_refinement(knots, degree):
    """Exact coarse-to-fine coefficient map; insert each interval midpoint once."""
    from scipy.interpolate import BSpline

    knots = np.asarray(knots, dtype=np.float64)
    count = len(knots) - int(degree) - 1
    spline = BSpline(knots, np.eye(count), int(degree), extrapolate=False)
    unique = np.unique(knots)
    for point in (unique[:-1] + unique[1:]) / 2.0:
        spline = spline.insert_knot(float(point))
    fine_count = len(spline.t) - int(degree) - 1
    return tuple(float(value) for value in spline.t), np.asarray(spline.c[:fine_count])


def _tau_squared_coefficients(config):
    """Represent the same global tau**2 polynomial exactly in the chosen time basis."""
    if getattr(config, "temporal_basis", "chebyshev") == "chebyshev":
        if int(config.time_order) < 2:
            raise ValueError("tau-squared perturbations require time_order >= 2")
        values = np.zeros(int(config.time_coefficient_count))
        values[:3] = (3.0 / 8.0, 1.0 / 2.0, 1.0 / 8.0)
        return values
    degree = int(config.temporal_degree)
    if degree < 2:
        raise ValueError("tau-squared perturbations require spline degree >= 2")
    knots = np.asarray(config.temporal_knots, dtype=np.float64)
    # The degree-p blossom of t**2 is the mean of its pairwise argument products.
    # Evaluating it on knots i+1,...,i+p gives the B-spline coefficient; this is
    # polynomial reproduction, not a sampled temporal least-squares fit.
    result = []
    for index in range(int(config.time_coefficient_count)):
        nodes = knots[index + 1:index + degree + 1]
        result.append(sum(nodes[i] * nodes[j] for i in range(degree)
                          for j in range(i + 1, degree)) / (degree * (degree - 1) / 2))
    return np.asarray(result, dtype=np.float64)


def _make_spline_refinement(field, *, rng, seed: int) -> StructuredNSPerturbation:
    """Coefficient-Euclidean complement in a fixed midpoint-refined spline space.

    This is a new perturbation definition. Its coefficient-L2 amplitude is not
    physically interchangeable with the old cosine-shell amplitude. The caller
    freezes an explicit epsilon after an offline physical-delta assessment.
    """
    if getattr(field.config, "spatial_basis", "cosine") != "bspline":
        raise ValueError("spline_refinement requires an explicitly selected spatial B-spline basis")
    torch = _require_torch()
    old = coefficient_tensor_from_field(field).detach().cpu().double()
    degree = int(field.config.spatial_degree)
    knots_x, rx = _midpoint_spline_refinement(field.config.spatial_knots_x, degree)
    knots_y, ry = _midpoint_spline_refinement(field.config.spatial_knots_y, degree)
    config = replace(field.config, spatial_mode_x=rx.shape[0] - 1,
                     spatial_mode_y=ry.shape[0] - 1,
                     spatial_knots_x=knots_x, spatial_knots_y=knots_y)
    # Projection onto range(Rx tensor Ry), without materializing its large
    # Kronecker matrix. QR uses the explicitly declared coefficient inner product.
    qx, _ = np.linalg.qr(rx, mode="reduced")
    qy, _ = np.linalg.qr(ry, mode="reduced")
    spatial = rng.standard_normal((rx.shape[0], ry.shape[0]))
    for _ in range(2):
        spatial -= qx @ (qx.T @ spatial @ qy) @ qy.T
    norm = float(np.linalg.norm(spatial))
    if not np.isfinite(norm) or norm <= 1.0e-14:
        raise ValueError("spline refinement produced no finite nonzero detail direction")
    spatial /= norm
    temporal = _tau_squared_coefficients(config)
    base = torch.einsum("ik,ctkl,jl->ctij", torch.from_numpy(rx), old, torch.from_numpy(ry))
    direction = torch.zeros_like(base)
    direction[0] = torch.from_numpy(temporal[:, None, None] * spatial[None])
    direction /= direction.norm()
    return StructuredNSPerturbation(
        family="spline_refinement", config=config, base_coefficients=base,
        direction_coefficients=direction,
        metadata={
            "seed": int(seed),
            "definition": "streamfunction_midpoint_spline_refinement_coefficient_orthogonal_detail",
            "refinement": "insert_each_spatial_interval_midpoint_once",
            "coarse_spatial_shape": list(old.shape[-2:]),
            "fine_spatial_shape": list(base.shape[-2:]),
            "complement_inner_product": "euclidean_fine_spline_coefficients",
            "coarse_projection_max_abs": float(np.abs(qx.T @ spatial @ qy).max()),
            "direction_coefficient_l2": float(direction.norm()),
            "amplitude_convention": "l2_normalized_joint_streamfunction_coefficients",
            "amplitude_equivalent_to_cosine": False,
            "time_envelope": "tau_squared",
            "initial_streamfunction_and_time_derivative_vanish": True,
        },
    )


def field_at_amplitude(
    base_field: ContinuousStructuredNSMMS,
    perturbation: StructuredNSPerturbation,
    *,
    amplitude: float,
    device: str | Any | None = None,
) -> ContinuousStructuredNSMMS:
    coefficients = (
        perturbation.base_coefficients
        + float(amplitude) * perturbation.direction_coefficients
    )
    config = perturbation.config
    coefficient_shape = (
        int(config.time_order) + 1,
        (int(config.spatial_mode_x) + 1) * (int(config.spatial_mode_y) + 1),
    )
    geometry_arguments = (
        {
            "geometry_spec": base_field.geometry_spec,
            "continuous_mms_abi": base_field.continuous_mms_abi,
        }
        if base_field.geometry_spec is not None
        else {
            "canonical_geometry": base_field.canonical_geometry,
            "continuous_mms_abi": base_field.continuous_mms_abi,
        }
    )
    return ContinuousStructuredNSMMS(
        StructuredNSCaseCoefficients(
            psi=coefficients[0].reshape(coefficient_shape),
            pressure=coefficients[1].reshape(coefficient_shape),
        ),
        config=config,
        **geometry_arguments,
        domain_bounds=base_field.domain_bounds,
        viscosity=float(base_field.viscosity),
        t0=float(base_field.t0),
        t1=float(base_field.t1),
        device=base_field.device if device is None else device,
    )


def structured_perturbation_velocity_metrics(base_field, perturbation, *, amplitude,
                                         spatial_points=17, time_points=5):
    """Explicit offline diagnostic on a fixed uniform physical query grid.

    No amplitude selection is performed here. This function is never invoked by
    the online perturber. RMS uses equal weights over fluid grid sites and both
    velocity components; it is a reproducible diagnostic, not an exact integral.
    """
    if perturbation.family not in {"spline_refinement", "spectral_complement"}:
        raise ValueError("these diagnostics require a unit coefficient-L2 complement perturbation")
    if not np.isfinite(float(amplitude)):
        raise ValueError("diagnostic amplitude must be finite")
    for name, value in (("spatial_points", spatial_points), ("time_points", time_points)):
        if isinstance(value, bool) or int(value) != value or int(value) < 2:
            raise ValueError(f"{name} must be an integer >= 2")
    torch = _require_torch()
    child = field_at_amplitude(base_field, perturbation, amplitude=float(amplitude))
    xmin, xmax, ymin, ymax = base_field.domain_bounds
    tt, xx, yy = np.meshgrid(np.linspace(base_field.t0, base_field.t1, int(time_points)),
                            np.linspace(xmin, xmax, int(spatial_points)),
                            np.linspace(ymin, ymax, int(spatial_points)), indexing="ij")
    points = np.column_stack((xx.ravel(), yy.ravel(), tt.ravel()))
    normalized = np.column_stack(((points[:, 0]-xmin)/(xmax-xmin),
                                  (points[:, 1]-ymin)/(ymax-ymin),
                                  (points[:, 2]-base_field.t0)/(base_field.t1-base_field.t0)))
    _, fluid = base_field.geometry_masks(torch.as_tensor(normalized, dtype=base_field.dtype,
                                                         device=base_field.device))
    selected = fluid.detach().cpu().numpy() > 0
    if not np.any(selected):
        raise ValueError("diagnostic grid contains no fluid sites")
    def velocity(field):
        # Dense offline diagnostic grids must not build one coordinate-autograd
        # graph for all points at once (especially at high spatial capacity).
        chunks = []
        batch = int(field.config.evaluation_batch_points)
        for start in range(0, len(points), batch):
            u, v, _ = field.velocity_pressure(points[start:start + batch])
            chunks.append(torch.stack((u, v), dim=-1).detach().cpu().numpy())
        return np.concatenate(chunks, axis=0)
    original, changed = velocity(base_field), velocity(child)
    delta = changed - original
    base_rms = float(np.sqrt(np.mean(original[selected] ** 2)))
    delta_rms = float(np.sqrt(np.mean(delta[selected] ** 2)))
    initial = selected & (points[:, 2] == base_field.t0)
    return {"diagnostic": "offline_uniform_fluid_grid_velocity_rms",
            "amplitude": float(amplitude), "amplitude_convention": "l2_normalized_joint_streamfunction_coefficients",
            "grid_shape": [int(time_points), int(spatial_points), int(spatial_points)],
            "fluid_query_count": int(selected.sum()), "base_velocity_rms": base_rms,
            "delta_velocity_rms": delta_rms,
            "delta_velocity_relative_rms": delta_rms/base_rms if base_rms > 0 else None,
            "initial_velocity_delta_max_abs": float(np.max(np.abs(delta[initial])))}


def spline_perturbation_velocity_metrics(base_field, perturbation, *, amplitude,
                                         spatial_points=17, time_points=5):
    """Backward-compatible spline-only entry point for the offline diagnostic."""
    if perturbation.family != "spline_refinement":
        raise ValueError("these diagnostics require spline_refinement")
    return structured_perturbation_velocity_metrics(base_field, perturbation, amplitude=amplitude,
                                                   spatial_points=spatial_points, time_points=time_points)


def masked_vector_rms(values, fluid_mask) -> float:
    array = np.asarray(values, dtype=np.float64)
    fluid = np.asarray(fluid_mask, dtype=bool)
    if array.ndim == 3:
        array = array[None]
    if array.ndim != 4 or int(array.shape[1]) != 2:
        raise ValueError(f"vector field must be [T,2,X,Y] or [2,X,Y], got {array.shape}")
    if tuple(array.shape[-2:]) != tuple(fluid.shape):
        raise ValueError("vector field and fluid mask shapes do not match")
    selected = array[:, :, fluid]
    return float(np.sqrt(np.mean(np.square(selected))))


def question_delta_metrics(
    *,
    base_velocity,
    base_forcing,
    perturbed_velocity,
    perturbed_forcing,
    fluid_mask,
    velocity_scale: float,
    forcing_scale: float,
) -> dict[str, float]:
    if min(float(velocity_scale), float(forcing_scale)) <= 0.0:
        raise ValueError("reference velocity and forcing scales must be positive")
    velocity_delta = np.asarray(perturbed_velocity) - np.asarray(base_velocity)
    forcing_delta = np.asarray(perturbed_forcing) - np.asarray(base_forcing)
    initial_rms = masked_vector_rms(velocity_delta[0], fluid_mask)
    forcing_rms = masked_vector_rms(forcing_delta, fluid_mask)
    truth_rms = masked_vector_rms(velocity_delta, fluid_mask)
    initial_scaled = initial_rms / float(velocity_scale)
    forcing_scaled = forcing_rms / float(forcing_scale)
    return {
        "initial_velocity_delta_rms": float(initial_rms),
        "forcing_delta_rms": float(forcing_rms),
        "truth_velocity_delta_rms": float(truth_rms),
        "initial_velocity_delta_reference_scaled": float(initial_scaled),
        "forcing_delta_reference_scaled": float(forcing_scaled),
        "truth_velocity_delta_reference_scaled": float(
            truth_rms / float(velocity_scale)
        ),
        "question_delta_norm": float(max(initial_scaled, forcing_scaled)),
    }


def paired_solver_metrics(
    *,
    base_prediction,
    base_truth,
    perturbed_prediction,
    perturbed_truth,
    fluid_mask,
    velocity_scale: float,
) -> dict[str, Any]:
    base_prediction = np.asarray(base_prediction, dtype=np.float64)
    base_truth = np.asarray(base_truth, dtype=np.float64)
    perturbed_prediction = np.asarray(perturbed_prediction, dtype=np.float64)
    perturbed_truth = np.asarray(perturbed_truth, dtype=np.float64)
    if not (
        base_prediction.shape
        == base_truth.shape
        == perturbed_prediction.shape
        == perturbed_truth.shape
    ):
        raise ValueError("paired prediction and truth shapes do not match")
    fluid = np.asarray(fluid_mask, dtype=bool)

    def selected(array):
        return np.asarray(array, dtype=np.float64)[:, :, fluid]

    def norm(array):
        return float(np.linalg.norm(selected(array).reshape(-1)))

    def rms(array):
        values = selected(array)
        return float(np.sqrt(np.mean(np.square(values))))

    base_error = base_prediction - base_truth
    perturbed_error = perturbed_prediction - perturbed_truth
    truth_delta = perturbed_truth - base_truth
    prediction_delta = perturbed_prediction - base_prediction
    response_mismatch = prediction_delta - truth_delta
    base_relative = norm(base_error) / max(norm(base_truth), 1.0e-15)
    perturbed_relative = norm(perturbed_error) / max(norm(perturbed_truth), 1.0e-15)
    truth_delta_norm = norm(truth_delta)
    reference_floor = (
        float(velocity_scale)
        * float(np.sqrt(selected(truth_delta).size))
        * 1.0e-8
    )
    per_time = []
    for index in range(int(truth_delta.shape[0])):
        delta_at_time = truth_delta[index : index + 1]
        mismatch_at_time = response_mismatch[index : index + 1]
        per_time.append(
            norm(mismatch_at_time)
            / max(norm(delta_at_time), reference_floor / np.sqrt(truth_delta.shape[0]))
        )
    return {
        "base_relative_l2": float(base_relative),
        "perturbed_relative_l2": float(perturbed_relative),
        "relative_l2_delta": float(perturbed_relative - base_relative),
        "relative_l2_ratio": float(
            perturbed_relative / max(base_relative, 1.0e-12)
        ),
        "truth_delta_rms": rms(truth_delta),
        "prediction_delta_rms": rms(prediction_delta),
        "paired_error_delta_rms": rms(perturbed_error - base_error),
        "paired_response_mismatch_rms": rms(response_mismatch),
        "paired_response_mismatch_rel": float(
            norm(response_mismatch) / max(truth_delta_norm, reference_floor)
        ),
        "paired_response_mismatch_rel_by_t": [float(value) for value in per_time],
        "paired_response_mismatch_rel_terminal": float(per_time[-1]),
        "paired_response_mismatch_rel_q90_time": float(np.quantile(per_time, 0.9)),
    }
