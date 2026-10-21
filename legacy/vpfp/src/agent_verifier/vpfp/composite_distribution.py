from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from agent_verifier.fusion_simulation_first.coverage import (
    prdc_metrics,
    sliced_wasserstein,
)
from agent_verifier.mast_solution_first.coverage_gfid import (
    diversity_audit,
    fit_train_only_whitened_pca,
    frechet_gaussian_distance,
    polynomial_kid,
)
from agent_verifier.vpfp.continuous_diffusion import denormalize_vectors
from agent_verifier.vpfp.continuous_macros import torch_squared_hermite_moments
from agent_verifier.vpfp.structured_diffusion import full_latent_from_active


COMPOSITE_POLICY_SCHEMA = "vpfp_diffusion_composite_distribution_policy_v1"
COMPOSITE_METRIC_IMPLEMENTATION = (
    "vpfp_composite_metrics_v2_c2st_shared_split_explicit_status"
)


@dataclass(frozen=True)
class VPFPCompositeDistributionConfig:
    pca_max_dimension: int = 24
    pca_variance_fraction: float = 0.99
    preprojection_dimension: int = 512
    bootstrap_repeats: int = 24
    bootstrap_fraction: float = 0.80
    sliced_projections: int = 128
    sliced_quantiles: int = 128
    prdc_neighbors: int = 5
    c2st_repeats: int = 8
    c2st_steps: int = 100
    c2st_hidden: int = 64
    null_upper_quantile: float = 0.95
    null_lower_quantile: float = 0.05
    minimum_compatible_families: int = 3
    physical_feature_compatible_fraction: float = 0.80
    physical_stratum_compatible_fraction: float = 0.75
    catastrophic_variance_ratio: float = 0.10
    catastrophic_duplicate_rate: float = 0.25
    catastrophic_spectrum_ratio_low: float = 0.10
    catastrophic_spectrum_ratio_high: float = 10.0
    rbf_bandwidth_multipliers: tuple[float, ...] = (0.5, 1.0, 2.0)
    primary_embedding_blocks: tuple[str, ...] = (
        "density_fourier",
        "kinetic_spectral",
    )

    def validate(self) -> None:
        positive = (
            self.pca_max_dimension,
            self.preprojection_dimension,
            self.bootstrap_repeats,
            self.sliced_projections,
            self.sliced_quantiles,
            self.prdc_neighbors,
            self.c2st_repeats,
            self.c2st_steps,
            self.c2st_hidden,
            self.minimum_compatible_families,
        )
        if min(int(value) for value in positive) < 1:
            raise ValueError("composite audit counts must be positive")
        fractions = (
            self.pca_variance_fraction,
            self.bootstrap_fraction,
            self.null_upper_quantile,
            self.physical_feature_compatible_fraction,
            self.physical_stratum_compatible_fraction,
        )
        if any(not 0.0 < float(value) <= 1.0 for value in fractions):
            raise ValueError("composite audit fractions must lie in (0,1]")
        if not 0.0 <= float(self.null_lower_quantile) < float(
            self.null_upper_quantile
        ):
            raise ValueError("composite audit null quantiles are invalid")
        if not self.rbf_bandwidth_multipliers or any(
            float(value) <= 0.0 for value in self.rbf_bandwidth_multipliers
        ):
            raise ValueError("RBF bandwidth multipliers must be positive")
        if len(set(self.primary_embedding_blocks)) != len(
            self.primary_embedding_blocks
        ):
            raise ValueError("primary embedding blocks must be unique")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["rbf_bandwidth_multipliers"] = list(
            self.rbf_bandwidth_multipliers
        )
        result["primary_embedding_blocks"] = list(
            self.primary_embedding_blocks
        )
        return result


def composite_embedding_audit(
    calibration: Mapping[str, Any],
    reference_a: Mapping[str, Any],
    reference_b: Mapping[str, Any],
    generated: Mapping[str, Any],
    *,
    config: VPFPCompositeDistributionConfig | None = None,
    seed: int = 20260807,
    c2st_device: str = "cpu",
) -> dict[str, Any]:
    """Apply five calibrated two-sample families to frozen sample sets."""

    cfg = config or VPFPCompositeDistributionConfig()
    cfg.validate()
    names = set(calibration)
    if (
        not names
        or names != set(reference_a)
        or names != set(reference_b)
        or names != set(generated)
    ):
        raise ValueError("composite embedding blocks must match")
    missing = set(cfg.primary_embedding_blocks) - names
    if missing:
        raise ValueError(f"missing primary embedding blocks: {sorted(missing)}")

    blocks = {}
    for block_index, name in enumerate(sorted(names)):
        blocks[name] = _audit_embedding_block(
            calibration[name],
            reference_a[name],
            reference_b[name],
            generated[name],
            config=cfg,
            seed=int(seed) + 100_003 * block_index,
            c2st_device=c2st_device,
        )

    family_names = (
        "frechet",
        "mmd_kid",
        "sliced_wasserstein",
        "c2st",
        "prdc",
    )
    families = {}
    for family in family_names:
        primary = {
            name: bool(blocks[name]["family_compatibility"][family])
            for name in cfg.primary_embedding_blocks
        }
        families[family] = {
            "compatible": all(primary.values()),
            "primary_blocks": primary,
            "diagnostic_blocks": {
                name: bool(block["family_compatibility"][family])
                for name, block in blocks.items()
                if name not in cfg.primary_embedding_blocks
            },
        }

    compatible_family_count = sum(
        int(row["compatible"]) for row in families.values()
    )
    corroborated = {}
    for block_name in cfg.primary_embedding_blocks:
        failed = [
            family
            for family, row in families.items()
            if not row["primary_blocks"][block_name]
        ]
        if len(failed) >= 2:
            corroborated[block_name] = failed

    for family_name, row in families.items():
        primary_status = {}
        for block_name, compatible in row["primary_blocks"].items():
            if compatible:
                status = "pass"
            elif family_name in corroborated.get(block_name, ()):
                status = "corroborated_failure"
            else:
                status = "warning"
            primary_status[block_name] = status
        row["primary_block_status"] = primary_status
        row["diagnostic_block_status"] = {
            block_name: "pass" if compatible else "warning"
            for block_name, compatible in row["diagnostic_blocks"].items()
        }
        if any(
            value == "corroborated_failure"
            for value in primary_status.values()
        ):
            row["status"] = "corroborated_failure"
        elif bool(row["compatible"]):
            row["status"] = "pass"
        else:
            row["status"] = "warning"

    catastrophes = []
    for name, block in blocks.items():
        diversity = block["diversity"]
        if (
            float(diversity["effective_rank_ratio"])
            < float(cfg.catastrophic_variance_ratio)
        ):
            catastrophes.append(f"{name}:effective_rank_collapse")
        if (
            float(diversity["pairwise_distance_median_ratio"])
            < float(cfg.catastrophic_variance_ratio)
        ):
            catastrophes.append(f"{name}:pairwise_distance_collapse")
        if (
            float(diversity["near_duplicate_generated_rate"])
            > float(cfg.catastrophic_duplicate_rate)
        ):
            catastrophes.append(f"{name}:near_duplicate_rate")

    soft_gate_passed = (
        compatible_family_count >= int(cfg.minimum_compatible_families)
        and not corroborated
        and not catastrophes
    )
    return {
        "schema": "vpfp_composite_embedding_audit_v1",
        "policy_schema": COMPOSITE_POLICY_SCHEMA,
        "metric_implementation": COMPOSITE_METRIC_IMPLEMENTATION,
        "config": cfg.to_dict(),
        "blocks": blocks,
        "families": families,
        "compatible_family_count": int(compatible_family_count),
        "required_compatible_family_count": int(
            cfg.minimum_compatible_families
        ),
        "corroborated_failures": corroborated,
        "warning_families": [
            name
            for name, row in families.items()
            if row["status"] == "warning"
        ],
        "corroborated_failure_families": [
            name
            for name, row in families.items()
            if row["status"] == "corroborated_failure"
        ],
        "catastrophic_failures": catastrophes,
        "soft_distribution_gate_passed": bool(soft_gate_passed),
        "single_metric_is_veto": False,
        "fit_boundary": (
            "PCA, whitening, RBF bandwidths, and feature scaling use only "
            "the disjoint reference calibration split"
        ),
        "generated_samples_used_for_fit": False,
    }


def mechanical_development_decision(
    *,
    hard_gates_passed: bool,
    physical_statistics_passed: bool,
    embedding_audit: Mapping[str, Any],
    hard_gates_pending: bool = False,
) -> dict[str, Any]:
    """Mechanically combine pre-registered hard and composite soft gates."""

    compatible = int(embedding_audit.get("compatible_family_count", 0))
    required = int(embedding_audit.get("required_compatible_family_count", 3))
    corroborated = dict(embedding_audit.get("corroborated_failures") or {})
    catastrophic = list(embedding_audit.get("catastrophic_failures") or [])
    soft_passed = (
        bool(embedding_audit.get("soft_distribution_gate_passed"))
        and bool(physical_statistics_passed)
        and compatible >= required
        and not corroborated
        and not catastrophic
    )
    admitted = bool(hard_gates_passed) and soft_passed and not hard_gates_pending
    reasons = []
    if hard_gates_pending:
        reasons.append("hard_answer_first_and_physical_gates_pending")
    elif not hard_gates_passed:
        reasons.append("hard_gate_failure")
    if not physical_statistics_passed:
        reasons.append("physical_statistics_family_failed")
    if compatible < required:
        reasons.append("fewer_than_three_distribution_families_compatible")
    if corroborated:
        reasons.append("independent_metric_families_corroborate_failure")
    if catastrophic:
        reasons.append("catastrophic_distribution_failure")
    return {
        "schema": "vpfp_composite_development_decision_v1",
        "policy_schema": COMPOSITE_POLICY_SCHEMA,
        "hard_gates_passed": bool(hard_gates_passed),
        "hard_gates_pending": bool(hard_gates_pending),
        "physical_statistics_passed": bool(physical_statistics_passed),
        "soft_distribution_gate_passed": bool(soft_passed),
        "development_admitted": bool(admitted),
        "claim_grade_admitted": False,
        "decision_rationale": reasons or ["development_policy_passed"],
    }


def continuous_physics_features(
    decoder,
    normalized_active_latents,
    equation_conditions,
    *,
    active_normalization: Mapping[str, Any],
    canonical_config,
    time_count: int = 6,
    spatial_count: int = 6,
) -> dict[str, Any]:
    """Evaluate compact VPFP statistics without a velocity-space grid."""

    if min(int(time_count), int(spatial_count)) < 3:
        raise ValueError("physics feature grids need at least three points per axis")
    torch = decoder.torch
    parameter = next(decoder.parameters())
    active = denormalize_vectors(
        torch.as_tensor(
            normalized_active_latents,
            dtype=parameter.dtype,
            device=parameter.device,
        ),
        active_normalization,
    )
    conditions = torch.as_tensor(
        equation_conditions,
        dtype=parameter.dtype,
        device=parameter.device,
    )
    count = int(active.shape[0])
    if tuple(active.shape) != (count, int(canonical_config.active_latent_dim)):
        raise ValueError("normalized active latent matrix has the wrong shape")
    if tuple(conditions.shape) != (count, 2):
        raise ValueError("equation conditions must have shape [N,2]")
    full = full_latent_from_active(active, canonical_config)

    t0, t1 = (float(value) for value in decoder.config.time_domain)
    lx, ly = (float(value) for value in decoder.config.equation.x_domain)
    t_axis = np.linspace(t0, t1, int(time_count), dtype=np.float64)
    x_axis = np.linspace(0.0, lx, int(spatial_count), endpoint=False)
    y_axis = np.linspace(0.0, ly, int(spatial_count), endpoint=False)
    mesh = np.meshgrid(t_axis, x_axis, y_axis, indexing="ij")
    txy = np.stack(mesh, axis=-1).reshape(-1, 3)
    points = np.column_stack((txy, np.zeros((txy.shape[0], 2))))
    points_tensor = torch.as_tensor(
        points, dtype=parameter.dtype, device=parameter.device
    )
    shape = (int(time_count), int(spatial_count), int(spatial_count))
    degrees = np.asarray(
        [first + second for first, second in decoder.hermite_modes],
        dtype=np.int64,
    )
    rows = []
    minimum_rho = float("inf")
    maximum_abs = 0.0
    decoder.eval()
    with torch.no_grad():
        for index in range(count):
            fields = decoder.fields(
                points_tensor,
                full[index],
                equation_condition=conditions[index],
            )
            moments = torch_squared_hermite_moments(
                torch,
                fields["hermite_coefficients"],
                modes=decoder.hermite_modes,
                temperature=fields["equation_condition"][:, 1],
                floor_mixture=float(decoder.config.velocity_floor_mixture),
            )
            rho = fields["rho"].reshape(shape).detach().cpu().numpy()
            electric = (
                fields["electric"].reshape(*shape, 2).detach().cpu().numpy()
            )
            mean_velocity = (
                moments["mean_velocity"]
                .reshape(*shape, 2)
                .detach()
                .cpu()
                .numpy()
            )
            raw_second = (
                moments["raw_second_moment"]
                .reshape(*shape, 3)
                .detach()
                .cpu()
                .numpy()
            )
            coefficients = (
                fields["hermite_coefficients"]
                .reshape(*shape, -1)
                .detach()
                .cpu()
                .numpy()
            )
            rows.append(
                _one_continuous_physics_row(
                    rho,
                    electric,
                    mean_velocity,
                    raw_second,
                    coefficients,
                    degrees,
                )
            )
            minimum_rho = min(minimum_rho, float(np.min(rho)))
            maximum_abs = max(
                maximum_abs,
                float(np.max(np.abs(rho))),
                float(np.max(np.abs(electric))),
                float(np.max(np.abs(mean_velocity))),
                float(np.max(np.abs(raw_second))),
            )
    names = list(rows[0]) if rows else []
    values = np.asarray(
        [[row[name] for name in names] for row in rows], dtype=np.float64
    )
    if values.shape != (count, len(names)) or not np.isfinite(values).all():
        raise FloatingPointError("continuous physics features are non-finite")
    return {
        "schema": "vpfp_continuous_physics_features_v1",
        "values": values,
        "feature_names": names,
        "diagnostics": {
            "case_count": count,
            "time_count": int(time_count),
            "spatial_count_per_axis": int(spatial_count),
            "velocity_grid_constructed": False,
            "velocity_moments": "exact_squared_Hermite_recurrence_on_R2",
            "g_normalization": "analytic_Parseval_identity",
            "g_normalization_max_abs_error": 0.0,
            "minimum_rho": float(minimum_rho),
            "maximum_abs_macro_value": float(maximum_abs),
            "finite": bool(np.isfinite(values).all()),
        },
    }


def calibrated_physics_audit(
    calibration_features,
    reference_a_features,
    reference_b_features,
    generated_features,
    *,
    feature_names: Sequence[str],
    calibration_conditions,
    reference_a_conditions,
    reference_b_conditions,
    generated_conditions,
    config: VPFPCompositeDistributionConfig | None = None,
    seed: int = 20260807,
) -> dict[str, Any]:
    cfg = config or VPFPCompositeDistributionConfig()
    cfg.validate()
    calibration, first, second, candidate = _aligned_four_matrices(
        calibration_features,
        reference_a_features,
        reference_b_features,
        generated_features,
        name="physics features",
    )
    names = [str(value) for value in feature_names]
    if len(names) != calibration.shape[1] or len(set(names)) != len(names):
        raise ValueError("physics feature names are invalid")
    mean = calibration.mean(axis=0)
    raw_scale = calibration.std(axis=0)
    positive_scale = raw_scale[raw_scale > 0.0]
    typical_scale = (
        float(np.median(positive_scale)) if positive_scale.size else 1.0
    )
    floor = max(typical_scale * 1.0e-4, 1.0e-8)
    scale = np.maximum(raw_scale, floor)
    informative_feature_mask = raw_scale > floor
    calibration_z, first_z, second_z, candidate_z = (
        (value - mean[None]) / scale[None]
        for value in (calibration, first, second, candidate)
    )
    random = np.random.default_rng(int(seed))
    feature_rows = []
    for feature_index, name in enumerate(names):
        null_mean = []
        candidate_mean = []
        null_scale = []
        candidate_scale = []
        for _ in range(int(cfg.bootstrap_repeats)):
            ia, ib, ig = _bootstrap_indices(random, first.shape[0])
            left = first_z[ia, feature_index]
            right = second_z[ib, feature_index]
            fake = candidate_z[ig, feature_index]
            null_mean.append(abs(float(np.mean(left) - np.mean(right))))
            candidate_mean.append(abs(float(np.mean(left) - np.mean(fake))))
            null_scale.append(_log_scale_gap(left, right))
            candidate_scale.append(_log_scale_gap(left, fake))
        mean_status = _upper_compatibility(
            null_mean, candidate_mean, config=cfg
        )
        scale_status = _upper_compatibility(
            null_scale, candidate_scale, config=cfg
        )
        feature_rows.append(
            {
                "feature": name,
                "mean": mean_status,
                "scale": scale_status,
                "compatible": bool(
                    mean_status["compatible"] and scale_status["compatible"]
                ),
            }
        )
        if not bool(informative_feature_mask[feature_index]):
            feature_rows[-1].update(
                {
                    "compatible": None,
                    "status": "unknown_reference_calibration_degenerate",
                    "calibration_standard_deviation": float(
                        raw_scale[feature_index]
                    ),
                    "calibration_scale_floor": float(floor),
                }
            )
    informative_rows = [
        row for row in feature_rows if row["compatible"] is not None
    ]
    compatible_fraction = (
        float(np.mean([row["compatible"] for row in informative_rows]))
        if informative_rows
        else 0.0
    )

    condition_report = _condition_strata_audit(
        calibration_z[:, informative_feature_mask],
        first_z[:, informative_feature_mask],
        second_z[:, informative_feature_mask],
        candidate_z[:, informative_feature_mask],
        calibration_conditions,
        reference_a_conditions,
        reference_b_conditions,
        generated_conditions,
        config=cfg,
        seed=int(seed) + 700_001,
    )
    feature_selection = {
        "total_feature_count": int(len(names)),
        "used_feature_count": int(np.count_nonzero(informative_feature_mask)),
        "excluded_reference_degenerate_features": [
            name
            for name, used in zip(names, informative_feature_mask, strict=True)
            if not bool(used)
        ],
        "criterion": "reference_calibration_std_above_registered_scale_floor",
        "scale_floor": float(floor),
    }
    condition_report["feature_selection"] = feature_selection
    variance_ratio = (
        float(
            np.var(candidate_z[:, informative_feature_mask], axis=0).sum()
            / max(
                float(
                    np.var(first_z[:, informative_feature_mask], axis=0).sum()
                ),
                1.0e-12,
            )
        )
        if bool(np.any(informative_feature_mask))
        else 0.0
    )
    spectral_ratios = []
    catastrophes = []
    for index, name in enumerate(names):
        if not any(
            token in name
            for token in ("hermite", "fourier", "field_energy", "spectrum")
        ):
            continue
        reference_value = float(np.mean(np.abs(first[:, index])))
        candidate_value = float(np.mean(np.abs(candidate[:, index])))
        if reference_value <= 1.0e-10:
            continue
        ratio = candidate_value / reference_value
        spectral_ratios.append(
            {"feature": name, "generated_to_reference_ratio": float(ratio)}
        )
        if (
            ratio < float(cfg.catastrophic_spectrum_ratio_low)
            or ratio > float(cfg.catastrophic_spectrum_ratio_high)
        ):
            catastrophes.append(f"{name}:order_of_magnitude_spectrum_gap")
    if variance_ratio < float(cfg.catastrophic_variance_ratio):
        catastrophes.append("physical_feature_variance_collapse")
    passed = (
        compatible_fraction >= float(cfg.physical_feature_compatible_fraction)
        and float(condition_report["compatible_fraction"])
        >= float(cfg.physical_stratum_compatible_fraction)
        and not condition_report["missing_regimes"]
        and not catastrophes
    )
    return {
        "schema": "vpfp_calibrated_physics_statistics_audit_v1",
        "policy_schema": COMPOSITE_POLICY_SCHEMA,
        "feature_rows": feature_rows,
        "feature_selection": feature_selection,
        "compatible_feature_fraction": compatible_fraction,
        "required_compatible_feature_fraction": float(
            cfg.physical_feature_compatible_fraction
        ),
        "condition_strata": condition_report,
        "generated_to_reference_variance_ratio": variance_ratio,
        "spectral_ratios": spectral_ratios,
        "catastrophic_failures": catastrophes,
        "physical_statistics_passed": bool(passed),
        "fit_boundary": "calibration/reference only",
        "generated_samples_used_for_fit": False,
    }


def _audit_embedding_block(
    calibration,
    reference_a,
    reference_b,
    generated,
    *,
    config: VPFPCompositeDistributionConfig,
    seed: int,
    c2st_device: str,
) -> dict[str, Any]:
    raw_sets = _aligned_four_matrices(
        calibration, reference_a, reference_b, generated, name="embedding"
    )
    raw_dimension = int(raw_sets[0].shape[1])
    (
        (calibration, first, second, candidate),
        preprojection,
    ) = _fixed_count_sketch_sets(
        raw_sets,
        maximum_dimension=int(config.preprojection_dimension),
        seed=int(seed) + 50_021,
    )
    embedding = fit_train_only_whitened_pca(
        calibration,
        max_dimension=min(
            int(config.pca_max_dimension), calibration.shape[0] - 1
        ),
        target_variance_fraction=float(config.pca_variance_fraction),
    )
    embedding_metadata = embedding.to_dict()
    embedding_metadata.update(
        {
            "schema": "vpfp_reference_calibration_whitened_pca_v1",
            "fit_split": "disjoint_reference_calibration_only",
            "generated_or_evaluation_used_for_fit": False,
        }
    )
    calibration_z, first_z, second_z, candidate_z = (
        embedding.transform(value)
        for value in (calibration, first, second, candidate)
    )
    bandwidth = _reference_bandwidth(calibration_z)
    repeat_count = int(config.bootstrap_repeats)
    subset_count = max(
        8,
        min(
            first.shape[0],
            int(round(float(config.bootstrap_fraction) * first.shape[0])),
        ),
    )
    random = np.random.default_rng(int(seed))
    values: dict[str, dict[str, list[float]]] = {
        "frechet": {"null": [], "candidate": []},
        "kid": {"null": [], "candidate": []},
        "sliced_wasserstein": {"null": [], "candidate": []},
    }
    for multiplier in config.rbf_bandwidth_multipliers:
        values[f"mmd_{float(multiplier):g}"] = {
            "null": [],
            "candidate": [],
        }
    prdc_values = {
        key: {"null": [], "candidate": []}
        for key in ("precision", "recall", "density", "coverage")
    }
    for repeat in range(repeat_count):
        ia = random.choice(first.shape[0], size=subset_count, replace=False)
        ib = random.choice(second.shape[0], size=subset_count, replace=False)
        ig = random.choice(candidate.shape[0], size=subset_count, replace=False)
        left = first_z[ia]
        right = second_z[ib]
        fake = candidate_z[ig]
        values["frechet"]["null"].append(
            frechet_gaussian_distance(left, right)["distance"]
        )
        values["frechet"]["candidate"].append(
            frechet_gaussian_distance(left, fake)["distance"]
        )
        values["kid"]["null"].append(polynomial_kid(left, right))
        values["kid"]["candidate"].append(polynomial_kid(left, fake))
        projection_seed = int(seed) + 10_007 * repeat
        values["sliced_wasserstein"]["null"].append(
            sliced_wasserstein(
                left,
                right,
                projections=int(config.sliced_projections),
                quantiles=int(config.sliced_quantiles),
                seed=projection_seed,
            )
        )
        values["sliced_wasserstein"]["candidate"].append(
            sliced_wasserstein(
                left,
                fake,
                projections=int(config.sliced_projections),
                quantiles=int(config.sliced_quantiles),
                seed=projection_seed,
            )
        )
        for multiplier in config.rbf_bandwidth_multipliers:
            key = f"mmd_{float(multiplier):g}"
            scaled = bandwidth * float(multiplier)
            values[key]["null"].append(_rbf_mmd(left, right, scaled))
            values[key]["candidate"].append(_rbf_mmd(left, fake, scaled))
        null_prdc = prdc_metrics(
            left, right, neighbors=int(config.prdc_neighbors)
        )
        candidate_prdc = prdc_metrics(
            left, fake, neighbors=int(config.prdc_neighbors)
        )
        for key in prdc_values:
            prdc_values[key]["null"].append(float(null_prdc[key]))
            prdc_values[key]["candidate"].append(float(candidate_prdc[key]))

    metric_rows = {
        name: _upper_compatibility(
            row["null"], row["candidate"], config=config
        )
        for name, row in values.items()
    }
    prdc_rows = {
        name: _lower_compatibility(
            row["null"], row["candidate"], config=config
        )
        for name, row in prdc_values.items()
    }
    c2st = _c2st_comparison(
        first_z,
        second_z,
        candidate_z,
        config=config,
        seed=int(seed) + 800_011,
        device=c2st_device,
    )
    mmd_keys = [
        "kid",
        *(
            f"mmd_{float(value):g}"
            for value in config.rbf_bandwidth_multipliers
        ),
    ]
    mmd_passes = sum(
        int(metric_rows[key]["compatible"]) for key in mmd_keys
    )
    prdc_passes = sum(int(row["compatible"]) for row in prdc_rows.values())
    family_compatibility = {
        "frechet": bool(metric_rows["frechet"]["compatible"]),
        "mmd_kid": mmd_passes >= math.ceil(0.75 * len(mmd_keys)),
        "sliced_wasserstein": bool(
            metric_rows["sliced_wasserstein"]["compatible"]
        ),
        "c2st": bool(c2st["compatible"]),
        "prdc": prdc_passes >= 3,
    }
    return {
        "sample_count_per_set": int(first.shape[0]),
        "raw_feature_dimension": raw_dimension,
        "fixed_preprojection": preprojection,
        "reference_only_embedding": embedding_metadata,
        "reference_bandwidth": float(bandwidth),
        "metrics": metric_rows,
        "prdc": prdc_rows,
        "c2st": c2st,
        "family_compatibility": family_compatibility,
        "diversity": diversity_audit(first_z, candidate_z),
        "resampling": {
            "method": "repeated_without_replacement_subsamples",
            "repeat_count": repeat_count,
            "rows_per_repeat": subset_count,
            "confidence_interval_quantiles": [
                float(config.null_lower_quantile),
                float(config.null_upper_quantile),
            ],
        },
    }


def _c2st_comparison(
    reference_a,
    reference_b,
    generated,
    *,
    config: VPFPCompositeDistributionConfig,
    seed: int,
    device: str,
) -> dict[str, Any]:
    null = []
    candidate = []
    for repeat in range(int(config.c2st_repeats)):
        repeat_seed = int(seed) + 104_729 * repeat
        null.append(
            _c2st_auc_once(
                reference_a,
                reference_b,
                seed=repeat_seed,
                steps=int(config.c2st_steps),
                hidden=int(config.c2st_hidden),
                device=device,
            )
        )
        candidate.append(
            _c2st_auc_once(
                reference_a,
                generated,
                seed=repeat_seed + 17,
                steps=int(config.c2st_steps),
                hidden=int(config.c2st_hidden),
                device=device,
            )
        )
    row = _upper_compatibility(null, candidate, config=config)
    row.update(
        {
            "direction_symmetric_auc": True,
            "split_strategy": (
                "shared_index_split_preserves_condition_pairing_and_"
                "prevents_cross_split_duplicate_leakage"
            ),
            "repeats": int(config.c2st_repeats),
            "classifier_steps": int(config.c2st_steps),
            "classifier_hidden_width": int(config.c2st_hidden),
            "preprocessing": "reference_calibration_only_whitened_PCA",
        }
    )
    return row


def _c2st_auc_once(
    first,
    second,
    *,
    seed: int,
    steps: int,
    hidden: int,
    device: str,
) -> float:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("PyTorch is required for C2ST") from exc
    left = torch.as_tensor(first, dtype=torch.float32, device=device)
    right = torch.as_tensor(second, dtype=torch.float32, device=device)
    count = min(int(left.shape[0]), int(right.shape[0]))
    if count < 8 or left.ndim != 2 or right.shape[1] != left.shape[1]:
        raise ValueError("C2ST requires aligned matrices with at least eight rows")
    random = torch.Generator(device=left.device)
    random.manual_seed(int(seed))
    order = torch.randperm(count, generator=random, device=left.device)
    left = left[:count][order]
    right = right[:count][order]
    split = max(4, int(0.70 * count))
    train = torch.cat((left[:split], right[:split]), dim=0)
    train_labels = torch.cat(
        (left.new_zeros(split), left.new_ones(split)), dim=0
    )
    test = torch.cat((left[split:], right[split:]), dim=0)
    labels = torch.cat(
        (left.new_zeros(count - split), left.new_ones(count - split)), dim=0
    )
    rng_devices = (
        [] if left.device.type == "cpu" else [int(left.device.index or 0)]
    )
    with torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(int(seed) + 31)
        model = torch.nn.Sequential(
            torch.nn.Linear(left.shape[1], int(hidden), device=left.device),
            torch.nn.SiLU(),
            torch.nn.Linear(int(hidden), 1, device=left.device),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=2.0e-3)
        for _ in range(int(steps)):
            logits = model(train).reshape(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, train_labels
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            scores = model(test).reshape(-1)
    positive = scores[labels > 0.5]
    negative = scores[labels <= 0.5]
    comparison = positive[:, None] - negative[None, :]
    directional = float(
        ((comparison > 0.0).float() + 0.5 * (comparison == 0.0).float())
        .mean()
        .cpu()
    )
    return max(directional, 1.0 - directional)


def _one_continuous_physics_row(
    rho,
    electric,
    mean_velocity,
    raw_second,
    coefficients,
    degrees,
) -> dict[str, float]:
    ux = mean_velocity[..., 0]
    uy = mean_velocity[..., 1]
    central_xx = np.maximum(raw_second[..., 0] - ux**2, 0.0)
    central_xy = raw_second[..., 1] - ux * uy
    central_yy = np.maximum(raw_second[..., 2] - uy**2, 0.0)
    temperature = 0.5 * (central_xx + central_yy)
    kinetic_energy_density = 0.5 * rho * (
        raw_second[..., 0] + raw_second[..., 2]
    )
    momentum = rho[..., None] * mean_velocity
    field_energy = 0.5 * np.sum(electric**2, axis=-1)
    coefficient_energy = np.square(coefficients)
    coefficient_energy /= np.maximum(
        np.sum(coefficient_energy, axis=-1, keepdims=True), 1.0e-30
    )
    low = np.sum(coefficient_energy[..., degrees <= 2], axis=-1)
    mid = np.sum(
        coefficient_energy[..., (degrees > 2) & (degrees <= 5)], axis=-1
    )
    high = np.sum(coefficient_energy[..., degrees > 5], axis=-1)
    entropy = -np.sum(
        coefficient_energy
        * np.log(np.maximum(coefficient_energy, 1.0e-30)),
        axis=-1,
    ) / math.log(max(int(coefficient_energy.shape[-1]), 2))
    rho_fluctuation = rho - np.mean(rho, axis=(1, 2), keepdims=True)
    return {
        "density_mean": float(np.mean(rho)),
        "density_std": float(np.std(rho)),
        "density_rms": float(np.sqrt(np.mean(rho**2))),
        "density_temporal_relative_change": _relative_change(rho),
        "density_spatial_fourier_high_fraction": _spatial_high_fraction(
            rho_fluctuation
        ),
        "density_temporal_spectrum_high_fraction": _temporal_high_fraction(
            np.sqrt(np.mean(rho_fluctuation**2, axis=(1, 2)))
        ),
        "density_decorrelation_time_fraction": _decorrelation_time_fraction(
            rho_fluctuation
        ),
        "momentum_x_mean_abs": float(np.mean(np.abs(momentum[..., 0]))),
        "momentum_y_mean_abs": float(np.mean(np.abs(momentum[..., 1]))),
        "momentum_rms": float(np.sqrt(np.mean(momentum**2))),
        "kinetic_energy_mean": float(np.mean(kinetic_energy_density)),
        "kinetic_energy_std": float(np.std(kinetic_energy_density)),
        "temperature_mean": float(np.mean(temperature)),
        "temperature_std": float(np.std(temperature)),
        "velocity_anisotropy_rms": float(
            np.sqrt(
                np.mean(
                    (
                        (central_xx - central_yy)
                        / np.maximum(2.0 * temperature, 1.0e-12)
                    )
                    ** 2
                )
            )
        ),
        "velocity_shear_rms": float(
            np.sqrt(
                np.mean(
                    (
                        central_xy
                        / np.maximum(2.0 * temperature, 1.0e-12)
                    )
                    ** 2
                )
            )
        ),
        "field_energy_mean": float(np.mean(field_energy)),
        "field_energy_temporal_relative_change": _relative_change(field_energy),
        "hermite_low_degree_energy_fraction": float(np.mean(low)),
        "hermite_mid_degree_energy_fraction": float(np.mean(mid)),
        "hermite_high_degree_energy_fraction": float(np.mean(high)),
        "hermite_high_degree_final_initial_ratio": float(
            np.mean(high[-1]) / max(float(np.mean(high[0])), 1.0e-12)
        ),
        "hermite_high_degree_spatial_fourier_fraction": _spatial_high_fraction(
            high
        ),
        "hermite_spectral_entropy_proxy": float(np.mean(entropy)),
    }


def _condition_strata_audit(
    calibration_features,
    reference_a_features,
    reference_b_features,
    generated_features,
    calibration_conditions,
    reference_a_conditions,
    reference_b_conditions,
    generated_conditions,
    *,
    config: VPFPCompositeDistributionConfig,
    seed: int,
) -> dict[str, Any]:
    condition_sets = [
        _finite_matrix(value, "condition matrix")
        for value in (
            calibration_conditions,
            reference_a_conditions,
            reference_b_conditions,
            generated_conditions,
        )
    ]
    if any(value.shape[1] != 2 for value in condition_sets):
        raise ValueError("condition matrices must have collision and temperature")
    if [value.shape[0] for value in condition_sets] != [
        calibration_features.shape[0],
        reference_a_features.shape[0],
        reference_b_features.shape[0],
        generated_features.shape[0],
    ]:
        raise ValueError("condition rows do not align with physics features")
    if int(calibration_features.shape[1]) < 1:
        return {
            "rows": [],
            "compatible_fraction": 0.0,
            "missing_regimes": ["no_reference_variable_solution_shape_features"],
            "strata_fit_split": "reference_calibration_only",
        }
    calibration, first_conditions, second_conditions, candidate_conditions = (
        condition_sets
    )
    rows = []
    missing = []
    random = np.random.default_rng(int(seed))
    for condition_index, name in enumerate(("collision_nu", "temperature")):
        threshold = float(np.median(calibration[:, condition_index]))
        for side in ("low", "high"):
            selector = (
                (lambda value: value <= threshold)
                if side == "low"
                else (lambda value: value > threshold)
            )
            masks = [
                selector(values[:, condition_index])
                for values in (
                    first_conditions,
                    second_conditions,
                    candidate_conditions,
                )
            ]
            counts = [int(np.count_nonzero(mask)) for mask in masks]
            if min(counts) < 8:
                missing.append(f"{name}:{side}")
                rows.append(
                    {
                        "condition": name,
                        "stratum": side,
                        "threshold": threshold,
                        "counts": counts,
                        "compatible": False,
                        "reason": "fewer_than_eight_cases",
                    }
                )
                continue
            null = []
            candidate = []
            for _ in range(int(config.bootstrap_repeats)):
                selected = []
                for mask in masks:
                    indices = np.flatnonzero(mask)
                    selected.append(
                        random.choice(indices, size=indices.size, replace=True)
                    )
                left = reference_a_features[selected[0]]
                right = reference_b_features[selected[1]]
                fake = generated_features[selected[2]]
                null.append(
                    float(
                        np.sqrt(
                            np.mean(
                                (left.mean(axis=0) - right.mean(axis=0)) ** 2
                            )
                        )
                    )
                )
                candidate.append(
                    float(
                        np.sqrt(
                            np.mean(
                                (left.mean(axis=0) - fake.mean(axis=0)) ** 2
                            )
                        )
                    )
                )
            status = _upper_compatibility(null, candidate, config=config)
            rows.append(
                {
                    "condition": name,
                    "stratum": side,
                    "threshold": threshold,
                    "counts": counts,
                    **status,
                }
            )
    compatible_fraction = (
        float(np.mean([bool(row.get("compatible")) for row in rows]))
        if rows
        else 0.0
    )
    return {
        "rows": rows,
        "compatible_fraction": compatible_fraction,
        "missing_regimes": missing,
        "strata_fit_split": "reference_calibration_only",
    }


def _upper_compatibility(null, candidate, *, config) -> dict[str, Any]:
    null_summary = _summary(null)
    candidate_summary = _summary(candidate)
    threshold = float(
        np.quantile(
            np.asarray(null, dtype=np.float64), config.null_upper_quantile
        )
    )
    return {
        "null": null_summary,
        "generated_reference": candidate_summary,
        "null_upper_threshold": threshold,
        "normalized_to_null_median": float(
            candidate_summary["median"]
            / max(abs(null_summary["median"]), 1.0e-12)
        ),
        "compatible": bool(candidate_summary["median"] <= threshold),
        "direction": "smaller_is_more_compatible",
    }


def _lower_compatibility(null, candidate, *, config) -> dict[str, Any]:
    null_summary = _summary(null)
    candidate_summary = _summary(candidate)
    threshold = float(
        np.quantile(
            np.asarray(null, dtype=np.float64), config.null_lower_quantile
        )
    )
    return {
        "null": null_summary,
        "generated_reference": candidate_summary,
        "null_lower_threshold": threshold,
        "compatible": bool(candidate_summary["median"] >= threshold),
        "direction": "larger_is_more_compatible",
    }


def _summary(values) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 1 or not np.isfinite(array).all():
        raise ValueError("metric samples must be a finite nonempty vector")
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "q05": float(np.quantile(array, 0.05)),
        "q95": float(np.quantile(array, 0.95)),
        "samples": [float(value) for value in array],
    }


def _reference_bandwidth(values) -> float:
    distances = _squared_distances(values, values)
    positive = distances[distances > 0.0]
    return math.sqrt(max(float(np.median(positive)), 1.0e-12))


def _rbf_mmd(first, second, bandwidth: float) -> float:
    first = _finite_matrix(first, "MMD first")
    second = _finite_matrix(second, "MMD second")
    if (
        first.shape[1] != second.shape[1]
        or min(first.shape[0], second.shape[0]) < 2
    ):
        raise ValueError("MMD matrices are incompatible")
    denominator = 2.0 * max(float(bandwidth) ** 2, 1.0e-12)
    first_kernel = np.exp(-_squared_distances(first, first) / denominator)
    second_kernel = np.exp(-_squared_distances(second, second) / denominator)
    cross_kernel = np.exp(-_squared_distances(first, second) / denominator)
    first_term = (first_kernel.sum() - np.trace(first_kernel)) / (
        first.shape[0] * (first.shape[0] - 1)
    )
    second_term = (second_kernel.sum() - np.trace(second_kernel)) / (
        second.shape[0] * (second.shape[0] - 1)
    )
    return float(first_term + second_term - 2.0 * cross_kernel.mean())


def _squared_distances(first, second) -> np.ndarray:
    first = _finite_matrix(first, "distance first")
    second = _finite_matrix(second, "distance second")
    if first.shape[1] != second.shape[1]:
        raise ValueError("distance matrices have different widths")
    return np.maximum(
        np.sum(first**2, axis=1)[:, None]
        + np.sum(second**2, axis=1)[None]
        - 2.0 * first @ second.T,
        0.0,
    )


def _fixed_count_sketch_sets(values, *, maximum_dimension: int, seed: int):
    matrices = tuple(_finite_matrix(value, "preprojection input") for value in values)
    if not matrices or len({value.shape[1] for value in matrices}) != 1:
        raise ValueError("preprojection inputs must have matching widths")
    input_dimension = int(matrices[0].shape[1])
    output_dimension = min(int(maximum_dimension), input_dimension)
    if output_dimension < 1:
        raise ValueError("preprojection dimension must be positive")
    if output_dimension == input_dimension:
        return matrices, {
            "schema": "vpfp_fixed_preprojection_v1",
            "method": "identity",
            "input_dimension": input_dimension,
            "output_dimension": output_dimension,
            "seed": None,
            "fit_split": "none",
            "generated_samples_used_for_fit": False,
        }

    random = np.random.default_rng(int(seed))
    buckets = random.integers(
        0, output_dimension, size=input_dimension, dtype=np.int64
    )
    signs = 2.0 * random.integers(
        0, 2, size=input_dimension, dtype=np.int8
    ) - 1.0
    counts = np.bincount(buckets, minlength=output_dimension).astype(np.float64)
    normalization = np.sqrt(np.maximum(counts, 1.0))

    def transform(matrix):
        result = np.empty((matrix.shape[0], output_dimension), dtype=np.float64)
        for row_index, row in enumerate(matrix):
            result[row_index] = np.bincount(
                buckets,
                weights=signs * row,
                minlength=output_dimension,
            )
        return result / normalization[None]

    return tuple(transform(value) for value in matrices), {
        "schema": "vpfp_fixed_preprojection_v1",
        "method": "signed_count_sketch",
        "input_dimension": input_dimension,
        "output_dimension": output_dimension,
        "seed": int(seed),
        "fit_split": "none_fixed_random_map",
        "generated_samples_used_for_fit": False,
    }


def _aligned_four_matrices(calibration, first, second, candidate, *, name):
    values = [
        _finite_matrix(value, name)
        for value in (calibration, first, second, candidate)
    ]
    widths = {value.shape[1] for value in values}
    if len(widths) != 1:
        raise ValueError(f"{name} widths differ")
    if len({value.shape[0] for value in values[1:]}) != 1:
        raise ValueError(f"{name} reference A/B/generated counts must match")
    if values[0].shape[0] < 3 or values[1].shape[0] < 16:
        raise ValueError(f"{name} sample sets are too small")
    return values


def _finite_matrix(value, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if (
        matrix.ndim != 2
        or min(matrix.shape) < 1
        or not np.isfinite(matrix).all()
    ):
        raise ValueError(f"{name} must be a finite nonempty matrix")
    return matrix


def _bootstrap_indices(random: np.random.Generator, count: int):
    return tuple(
        random.integers(0, int(count), size=int(count), dtype=np.int64)
        for _ in range(3)
    )


def _log_scale_gap(first, second) -> float:
    floor = 1.0e-6
    return abs(
        math.log(
            max(float(np.std(first)), floor)
            / max(float(np.std(second)), floor)
        )
    )


def _relative_change(values) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(
        np.sqrt(np.mean((array - array[:1]) ** 2))
        / max(float(np.sqrt(np.mean(array**2))), 1.0e-12)
    )


def _spatial_high_fraction(values) -> float:
    array = np.asarray(values, dtype=np.float64)
    spectrum = np.fft.fft2(array, axes=(-2, -1))
    power = np.abs(spectrum) ** 2
    nx, ny = array.shape[-2:]
    kx = np.fft.fftfreq(nx) * nx
    ky = np.fft.fftfreq(ny) * ny
    radius = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
    high = radius >= max(2.0, 0.35 * float(np.max(radius)))
    return float(
        np.sum(power[..., high]) / max(float(np.sum(power)), 1.0e-30)
    )


def _temporal_high_fraction(values) -> float:
    array = np.asarray(values, dtype=np.float64)
    power = np.abs(np.fft.rfft(array - np.mean(array))) ** 2
    if power.size <= 2:
        return 0.0
    return float(
        np.sum(power[2:]) / max(float(np.sum(power[1:])), 1.0e-30)
    )


def _decorrelation_time_fraction(values) -> float:
    array = np.asarray(values, dtype=np.float64)
    flat = array.reshape(array.shape[0], -1)
    first = flat[0]
    denominator = max(float(np.mean(first**2)), 1.0e-30)
    correlation = np.mean(flat * first[None], axis=1) / denominator
    below = np.flatnonzero(correlation <= math.exp(-1.0))
    if below.size < 1:
        return 1.0
    return float(below[0] / max(array.shape[0] - 1, 1))
