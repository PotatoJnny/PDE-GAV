from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from agent_verifier.vpfp.io import load_case

Array = np.ndarray


@dataclass(frozen=True)
class VPFPEmbeddingConfig:
    """Fixed-size, grid-independent summaries of a materialized VPFP function."""

    time_samples: int = 8
    spectrum_time_samples: int = 4
    spatial_band_edges: tuple[float, ...] = (0.5, 1.5, 2.5, 3.5, 5.5)
    velocity_moment_degree: int = 6
    public_sketch_width: int = 64

    def validate(self) -> None:
        if min(
            self.time_samples,
            self.spectrum_time_samples,
            self.velocity_moment_degree,
            self.public_sketch_width,
        ) < 1:
            raise ValueError("embedding dimensions must be positive")
        if tuple(sorted(self.spatial_band_edges)) != self.spatial_band_edges:
            raise ValueError("spatial band edges must be increasing")


@dataclass(frozen=True)
class VPFPGeneratorQualityGateConfig:
    """Provisional gates calibrated against held-out reference/reference splits."""

    min_reference_cases: int = 32
    min_candidate_cases: int = 32
    repeat_count: int = 12
    knn_k: int = 3
    pca_max_dim: int = 32
    precision_min: float = 0.50
    recall_min: float = 0.50
    density_min: float = 0.40
    coverage_min: float = 0.50
    relative_to_self_min: float = 0.70
    cap_absolute_minima_by_self_median: bool = False
    frechet_self_multiplier: float = 3.0
    mmd_self_multiplier: float = 3.0
    effective_rank_ratio_min: float = 0.50
    nearest_recall_ratio_max: float = 2.5
    shortcut_r2_gap_max: float = 0.25
    shortcut_r2_gate_floor: float = 0.50
    required_reference_equation_family: str = "spatially_conditioned_gaussian_packets_v3"
    representation_rho_rel_l2_p90_max: float = 0.15
    representation_f_rel_l2_p90_max: float = 0.30

    def validate(self) -> None:
        if min(
            self.min_reference_cases,
            self.min_candidate_cases,
            self.repeat_count,
            self.knn_k,
            self.pca_max_dim,
        ) < 1:
            raise ValueError("quality gate counts must be positive")
        if not 0.0 < float(self.relative_to_self_min) <= 1.0:
            raise ValueError(
                "relative_to_self_min must lie in (0, 1]"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def embed_case(
    case_dir: str | Path,
    *,
    config: VPFPEmbeddingConfig | None = None,
) -> dict[str, Any]:
    """Embed hidden answer functions and their public question without grid flattening."""

    cfg = config or VPFPEmbeddingConfig()
    cfg.validate()
    case = load_case(case_dir)
    public = case["public_arrays"]
    hidden = case["hidden_arrays"]
    rho = np.asarray(hidden["rho_true"], dtype=np.float64)
    g = np.asarray(hidden["g_true"], dtype=np.float64)
    electric = np.asarray(hidden["E_true"], dtype=np.float64)
    weights = np.asarray(public["velocity_weights"], dtype=np.float64)
    vx = np.asarray(public["vx"], dtype=np.float64)
    vy = np.asarray(public["vy"], dtype=np.float64)
    t = np.asarray(public["t"], dtype=np.float64)
    _validate_case_arrays(rho, g, electric, weights, vx, vy, t)

    normalized_g, mass = _normalize_window_density(g, weights)
    moment_fields = _kinetic_fields(normalized_g, weights, vx, vy)
    spatial_fields = {
        "rho_anomaly": rho / np.maximum(np.mean(rho, axis=(1, 2), keepdims=True), 1.0e-12) - 1.0,
        "u_x": moment_fields["u_x"],
        "u_y": moment_fields["u_y"],
        "temperature": moment_fields["temperature"],
        "normal_stress": moment_fields["normal_stress"],
        "shear_stress": moment_fields["shear_stress"],
        "electric_x": electric[:, 0],
        "electric_y": electric[:, 1],
    }
    answer_values: list[float] = []
    answer_names: list[str] = []
    for field_name, field in spatial_fields.items():
        summaries = {
            "mean": np.mean(field, axis=(1, 2)),
            "std": np.std(field, axis=(1, 2)),
            "rms": np.sqrt(np.mean(field**2, axis=(1, 2))),
        }
        for summary_name, trajectory in summaries.items():
            sampled = _resample_trajectory(trajectory, cfg.time_samples)
            answer_values.extend(sampled.tolist())
            answer_names.extend(
                f"{field_name}.{summary_name}.t{index:02d}"
                for index in range(cfg.time_samples)
            )

    spectrum_indices = _sample_indices(rho.shape[0], cfg.spectrum_time_samples)
    for field_name, field in spatial_fields.items():
        for time_slot, time_index in enumerate(spectrum_indices):
            powers = _spatial_band_powers(
                field[int(time_index)], edges=cfg.spatial_band_edges
            )
            answer_values.extend(np.log1p(powers).tolist())
            answer_names.extend(
                f"{field_name}.spectrum.t{time_slot:02d}.band{band:02d}"
                for band in range(len(powers))
            )

    velocity_moments = _velocity_moment_fields(
        normalized_g,
        weights,
        vx,
        vy,
        max_degree=cfg.velocity_moment_degree,
    )
    for name, field in velocity_moments.items():
        trajectory = np.mean(field, axis=(1, 2))
        values = (
            float(np.mean(trajectory)),
            float(np.std(field)),
            float(_temporal_relative_change(trajectory)),
        )
        answer_values.extend(values)
        answer_names.extend((f"{name}.mean", f"{name}.std", f"{name}.temporal_change"))

    radius = np.sqrt(vx[:, None] ** 2 + vy[None, :] ** 2)
    for threshold in (2.0, 3.0, 4.0):
        tail = np.sum(
            normalized_g * weights * (radius >= threshold), axis=(-2, -1)
        )
        answer_values.extend((float(np.mean(tail)), float(np.std(tail))))
        answer_names.extend((f"tail_ge_{threshold:g}.mean", f"tail_ge_{threshold:g}.std"))
    answer_values.extend(
        (
            float(np.mean(np.abs(mass - 1.0))),
            float(np.max(np.abs(mass - 1.0))),
            float(t[-1] - t[0]),
        )
    )
    answer_names.extend(("window_mass.mean_abs_error", "window_mass.max_abs_error", "time_horizon"))

    public_embedding, public_names = _public_question_embedding(case, cfg)
    answer = np.asarray(answer_values, dtype=np.float64)
    if not np.all(np.isfinite(answer)) or not np.all(np.isfinite(public_embedding)):
        raise ValueError(f"non-finite function embedding for {case_dir}")
    return {
        "case_id": case["case_id"],
        "case_dir": str(Path(case_dir).resolve()),
        "source_origin": case["public_question"].get("source_origin"),
        "equation_family": (
            case["case_spec"].get("continuous_equation_spec", {}).get("family")
            or case["case_spec"].get("family")
        ),
        "answer": answer,
        "answer_feature_names": answer_names,
        "public": public_embedding,
        "public_feature_names": public_names,
    }


def embed_solution_arrays(
    *,
    rho: Array,
    g: Array,
    electric: Array,
    velocity_weights: Array,
    vx: Array,
    vy: Array,
    t: Array,
    config: VPFPEmbeddingConfig | None = None,
) -> dict[str, Any]:
    """Embed an in-memory VPFP solution without materializing a case bundle."""

    cfg = config or VPFPEmbeddingConfig()
    cfg.validate()
    rho = np.asarray(rho, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    electric = np.asarray(electric, dtype=np.float64)
    weights = np.asarray(velocity_weights, dtype=np.float64)
    vx = np.asarray(vx, dtype=np.float64)
    vy = np.asarray(vy, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    _validate_case_arrays(rho, g, electric, weights, vx, vy, t)

    normalized_g, mass = _normalize_window_density(g, weights)
    moment_fields = _kinetic_fields(normalized_g, weights, vx, vy)
    spatial_fields = {
        "rho_anomaly": rho
        / np.maximum(np.mean(rho, axis=(1, 2), keepdims=True), 1.0e-12)
        - 1.0,
        "u_x": moment_fields["u_x"],
        "u_y": moment_fields["u_y"],
        "temperature": moment_fields["temperature"],
        "normal_stress": moment_fields["normal_stress"],
        "shear_stress": moment_fields["shear_stress"],
        "electric_x": electric[:, 0],
        "electric_y": electric[:, 1],
    }
    answer_values: list[float] = []
    answer_names: list[str] = []
    for field_name, field in spatial_fields.items():
        summaries = {
            "mean": np.mean(field, axis=(1, 2)),
            "std": np.std(field, axis=(1, 2)),
            "rms": np.sqrt(np.mean(field**2, axis=(1, 2))),
        }
        for summary_name, trajectory in summaries.items():
            sampled = _resample_trajectory(trajectory, cfg.time_samples)
            answer_values.extend(sampled.tolist())
            answer_names.extend(
                f"{field_name}.{summary_name}.t{index:02d}"
                for index in range(cfg.time_samples)
            )

    spectrum_indices = _sample_indices(rho.shape[0], cfg.spectrum_time_samples)
    for field_name, field in spatial_fields.items():
        for time_slot, time_index in enumerate(spectrum_indices):
            powers = _spatial_band_powers(
                field[int(time_index)], edges=cfg.spatial_band_edges
            )
            answer_values.extend(np.log1p(powers).tolist())
            answer_names.extend(
                f"{field_name}.spectrum.t{time_slot:02d}.band{band:02d}"
                for band in range(len(powers))
            )

    velocity_moments = _velocity_moment_fields(
        normalized_g,
        weights,
        vx,
        vy,
        max_degree=cfg.velocity_moment_degree,
    )
    for name, field in velocity_moments.items():
        trajectory = np.mean(field, axis=(1, 2))
        answer_values.extend(
            (
                float(np.mean(trajectory)),
                float(np.std(field)),
                float(_temporal_relative_change(trajectory)),
            )
        )
        answer_names.extend(
            (f"{name}.mean", f"{name}.std", f"{name}.temporal_change")
        )

    radius = np.sqrt(vx[:, None] ** 2 + vy[None, :] ** 2)
    for threshold in (2.0, 3.0, 4.0):
        tail = np.sum(
            normalized_g * weights * (radius >= threshold),
            axis=(-2, -1),
        )
        answer_values.extend((float(np.mean(tail)), float(np.std(tail))))
        answer_names.extend(
            (f"tail_ge_{threshold:g}.mean", f"tail_ge_{threshold:g}.std")
        )
    answer_values.extend(
        (
            float(np.mean(np.abs(mass - 1.0))),
            float(np.max(np.abs(mass - 1.0))),
            float(t[-1] - t[0]),
        )
    )
    answer_names.extend(
        (
            "window_mass.mean_abs_error",
            "window_mass.max_abs_error",
            "time_horizon",
        )
    )
    answer = np.asarray(answer_values, dtype=np.float64)
    if not np.all(np.isfinite(answer)):
        raise ValueError("non-finite in-memory VPFP solution embedding")
    return {
        "answer": answer,
        "answer_feature_names": answer_names,
    }


def embed_case_dirs(
    case_dirs: Iterable[str | Path],
    *,
    config: VPFPEmbeddingConfig | None = None,
) -> dict[str, Any]:
    rows = [embed_case(path, config=config) for path in case_dirs]
    if not rows:
        raise ValueError("cannot embed an empty case collection")
    answer_names = rows[0]["answer_feature_names"]
    public_names = rows[0]["public_feature_names"]
    if any(row["answer_feature_names"] != answer_names for row in rows):
        raise ValueError("answer embedding schema changed across cases")
    if any(row["public_feature_names"] != public_names for row in rows):
        raise ValueError("public embedding schema changed across cases")
    return {
        "case_ids": [row["case_id"] for row in rows],
        "case_dirs": [row["case_dir"] for row in rows],
        "source_origins": [row["source_origin"] for row in rows],
        "equation_families": [row["equation_family"] for row in rows],
        "answer": np.stack([row["answer"] for row in rows]),
        "answer_feature_names": answer_names,
        "public": np.stack([row["public"] for row in rows]),
        "public_feature_names": public_names,
    }


def audit_reference_discretization_evidence(
    case_dirs: Iterable[str | Path],
    *,
    required_max_tail_mass_fraction: float = 1.0e-3,
) -> dict[str, Any]:
    """Require auditable velocity-window evidence for every equation-first case."""

    required = float(required_max_tail_mass_fraction)
    if not np.isfinite(required) or required <= 0.0:
        raise ValueError("required velocity-tail threshold must be finite and positive")
    rows = []
    failures: list[str] = []
    for raw_path in case_dirs:
        case_dir = Path(raw_path).resolve()
        case_id = case_dir.name
        row_failures: list[str] = []
        audit_path = case_dir / "high_fidelity_audit.json"
        diagnostics_path = case_dir / "diagnostics.json"
        audit: dict[str, Any] = {}
        diagnostics: dict[str, Any] = {}
        if not audit_path.is_file():
            row_failures.append("high_fidelity_audit_missing")
        else:
            try:
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                row_failures.append("high_fidelity_audit_malformed")
        if not diagnostics_path.is_file():
            row_failures.append("diagnostics_missing")
        else:
            try:
                diagnostics = json.loads(
                    diagnostics_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                row_failures.append("diagnostics_malformed")

        if audit and audit.get("schema") != "vpfp_high_fidelity_case_audit_v1":
            row_failures.append("high_fidelity_audit_schema")
        if audit and audit.get("accepted") is not True:
            row_failures.append("high_fidelity_audit_not_accepted")
        audited_tail = audit.get("velocity_tail_mass_fraction") if audit else None
        audited_threshold = (
            audit.get("thresholds", {}).get("max_velocity_tail_mass_fraction")
            if audit
            else None
        )
        recomputed_tail = None
        try:
            kinetic = diagnostics["kinetic"]
            recomputed_tail = float(kinetic["velocity_tail_mass"]["max"]) / max(
                abs(float(kinetic["mass"]["min"])), 1.0e-12
            )
        except (KeyError, TypeError, ValueError):
            row_failures.append("velocity_tail_diagnostics_missing")
        audited_tail_value = None
        if audited_tail is None:
            row_failures.append("velocity_tail_audit_value_missing")
        else:
            try:
                audited_tail_value = float(audited_tail)
            except (TypeError, ValueError):
                row_failures.append("velocity_tail_audit_value_nonfinite")
            else:
                if not np.isfinite(audited_tail_value):
                    row_failures.append("velocity_tail_audit_value_nonfinite")
        audited_threshold_value = None
        if audited_threshold is None:
            row_failures.append("velocity_tail_audit_threshold_missing")
        else:
            try:
                audited_threshold_value = float(audited_threshold)
            except (TypeError, ValueError):
                row_failures.append("velocity_tail_audit_threshold_too_loose")
            else:
                if (
                    not np.isfinite(audited_threshold_value)
                    or audited_threshold_value > required * (1.0 + 1.0e-12)
                ):
                    row_failures.append("velocity_tail_audit_threshold_too_loose")
        if recomputed_tail is not None:
            if not np.isfinite(recomputed_tail) or recomputed_tail > required:
                row_failures.append("velocity_tail_mass_fraction")
            if audited_tail_value is not None and np.isfinite(audited_tail_value):
                tolerance = 1.0e-12 + 1.0e-9 * abs(recomputed_tail)
                if abs(audited_tail_value - recomputed_tail) > tolerance:
                    row_failures.append("velocity_tail_audit_value_mismatch")
        failures.extend(f"{case_id}:{value}" for value in row_failures)
        rows.append(
            {
                "case_id": case_id,
                "case_dir": str(case_dir),
                "accepted": not row_failures,
                "failures": row_failures,
                "audited_tail_mass_fraction": audited_tail,
                "recomputed_tail_mass_fraction": recomputed_tail,
                "audited_threshold": audited_threshold,
            }
        )
    if not rows:
        failures.append("reference_cases_empty")
    return {
        "schema": "vpfp_reference_discretization_evidence_v1",
        "accepted": not failures,
        "required_max_velocity_tail_mass_fraction": required,
        "case_count": len(rows),
        "failures": failures,
        "cases": rows,
    }


def evaluate_generator_quality(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    gate_config: VPFPGeneratorQualityGateConfig | None = None,
    seed: int = 20260713,
    representation_report: dict[str, Any] | None = None,
    reference_discretization_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare generated support with equation-first support using calibrated splits."""

    cfg = gate_config or VPFPGeneratorQualityGateConfig()
    cfg.validate()
    failures: list[str] = []
    warnings: list[str] = []
    reference_count = int(np.asarray(reference["answer"]).shape[0])
    candidate_count = int(np.asarray(candidate["answer"]).shape[0])
    reference_families = sorted(
        {str(value) for value in reference.get("equation_families", [])}
    )
    reference_origins = sorted(
        {str(value) for value in reference.get("source_origins", [])}
    )
    if reference_origins != ["randomized_public_equation"]:
        failures.append(
            "reference_source_origin:expected randomized_public_equation, "
            f"got {reference_origins}"
        )
    if reference_families != [cfg.required_reference_equation_family]:
        failures.append(
            "reference_equation_family:expected "
            f"{cfg.required_reference_equation_family}, got {reference_families}"
        )
    if reference_count < cfg.min_reference_cases:
        failures.append(
            f"reference_case_count:{reference_count}<{cfg.min_reference_cases}"
        )
    if candidate_count < cfg.min_candidate_cases:
        failures.append(
            f"candidate_case_count:{candidate_count}<{cfg.min_candidate_cases}"
        )

    answer_distribution = calibrated_distribution_comparison(
        np.asarray(reference["answer"], dtype=np.float64),
        np.asarray(candidate["answer"], dtype=np.float64),
        gate_config=cfg,
        seed=seed,
    )
    public_distribution = calibrated_distribution_comparison(
        np.asarray(reference["public"], dtype=np.float64),
        np.asarray(candidate["public"], dtype=np.float64),
        gate_config=cfg,
        seed=seed + 101,
    )
    failures.extend(f"answer_distribution:{value}" for value in answer_distribution["failures"])
    failures.extend(f"public_distribution:{value}" for value in public_distribution["failures"])

    reference_probe = cross_validated_shortcut_probe(
        np.asarray(reference["public"], dtype=np.float64),
        np.asarray(reference["answer"], dtype=np.float64),
        seed=seed + 211,
    )
    candidate_probe = cross_validated_shortcut_probe(
        np.asarray(candidate["public"], dtype=np.float64),
        np.asarray(candidate["answer"], dtype=np.float64),
        seed=seed + 307,
    )
    shortcut_gap = float(candidate_probe["r2"] - reference_probe["r2"])
    shortcut_failed = bool(
        candidate_probe["r2"] >= cfg.shortcut_r2_gate_floor
        and shortcut_gap > cfg.shortcut_r2_gap_max
    )
    if shortcut_failed:
        failures.append(
            "shortcut_probe:candidate source-to-answer inversion is materially easier "
            "than the equation-first reference"
        )
    if reference_count < 2 * max(4, cfg.knn_k + 1):
        warnings.append("reference set is too small for stable split-calibrated metrics")

    if reference_discretization_report is None:
        failures.append("reference_discretization:not_run")
    elif not bool(reference_discretization_report.get("accepted", False)):
        failures.extend(
            f"reference_discretization:{value}"
            for value in reference_discretization_report.get(
                "failures", ["failed"]
            )
        )

    if representation_report is None:
        failures.append("representation_coverage:not_run")
    elif not bool(representation_report.get("accepted", False)):
        failures.extend(
            f"representation_coverage:{value}"
            for value in representation_report.get("failures", ["failed"])
        )

    return {
        "schema": "vpfp_generator_quality_audit_v1",
        "ready_for_agent_iteration": not failures,
        "claim_scope": "generator_distribution_fidelity_coverage_and_shortcut_audit",
        "reference_count": reference_count,
        "candidate_count": candidate_count,
        "reference_equation_families": reference_families,
        "reference_source_origins": reference_origins,
        "gate_config": cfg.to_dict(),
        "failures": failures,
        "warnings": warnings,
        "answer_distribution": answer_distribution,
        "public_distribution": public_distribution,
        "shortcut_probe": {
            "reference": reference_probe,
            "candidate": candidate_probe,
            "candidate_minus_reference_r2": shortcut_gap,
            "failed": shortcut_failed,
            "interpretation": (
                "This is a relative shortcut diagnostic, not a claim that the physical "
                "source-to-solution map should be unpredictable."
            ),
        },
        "reference_discretization": reference_discretization_report,
        "representation_coverage": representation_report,
    }


def calibrated_distribution_comparison(
    reference: Array,
    candidate: Array,
    *,
    gate_config: VPFPGeneratorQualityGateConfig | None = None,
    seed: int = 20260713,
) -> dict[str, Any]:
    cfg = gate_config or VPFPGeneratorQualityGateConfig()
    reference = _as_matrix(reference, "reference")
    candidate = _as_matrix(candidate, "candidate")
    if reference.shape[1] != candidate.shape[1]:
        raise ValueError("reference and candidate embedding dimensions differ")
    minimum = max(4, cfg.knn_k + 1)
    evaluation_count = min(candidate.shape[0], reference.shape[0] // 2)
    if evaluation_count < minimum:
        return {
            "accepted": False,
            "failures": [f"insufficient_split_size:{evaluation_count}<{minimum}"],
            "evaluation_count": int(evaluation_count),
            "embedding_dim": int(reference.shape[1]),
            "self": {},
            "candidate": {},
            "thresholds": {},
        }

    rng = np.random.default_rng(int(seed))
    self_rows: list[dict[str, float]] = []
    candidate_rows: list[dict[str, float]] = []
    for _ in range(int(cfg.repeat_count)):
        ref_order = rng.permutation(reference.shape[0])
        fit = reference[ref_order[:evaluation_count]]
        held = reference[ref_order[evaluation_count : 2 * evaluation_count]]
        cand_indices = rng.choice(
            candidate.shape[0], size=evaluation_count, replace=False
        )
        cand = candidate[cand_indices]
        transform = _ReferenceTransform.fit(fit, max_dim=cfg.pca_max_dim)
        fit_z = transform.apply(fit)
        held_z = transform.apply(held)
        cand_z = transform.apply(cand)
        self_rows.append(_distribution_metrics(fit_z, held_z, k=cfg.knn_k))
        candidate_rows.append(_distribution_metrics(fit_z, cand_z, k=cfg.knn_k))

    self_summary = _metric_summary(self_rows)
    candidate_summary = _metric_summary(candidate_rows)
    failures: list[str] = []
    thresholds: dict[str, float | bool] = {}
    for metric, absolute_minimum in (
        ("precision", cfg.precision_min),
        ("recall", cfg.recall_min),
        ("density", cfg.density_min),
        ("coverage", cfg.coverage_min),
    ):
        self_relative_p05 = (
            float(self_summary[metric]["p05"])
            * float(cfg.relative_to_self_min)
        )
        self_relative_median = (
            float(self_summary[metric]["median"])
            * float(cfg.relative_to_self_min)
        )
        effective_absolute_minimum = float(absolute_minimum)
        if bool(cfg.cap_absolute_minima_by_self_median):
            effective_absolute_minimum = min(
                effective_absolute_minimum,
                self_relative_median,
            )
        threshold = max(
            effective_absolute_minimum,
            self_relative_p05,
        )
        thresholds[f"{metric}_min"] = threshold
        thresholds[f"{metric}_registered_absolute_min"] = float(
            absolute_minimum
        )
        thresholds[f"{metric}_self_median_relative_min"] = self_relative_median
        if float(candidate_summary[metric]["median"]) < threshold:
            failures.append(metric)

    thresholds["absolute_minima_capped_by_self_median"] = bool(
        cfg.cap_absolute_minima_by_self_median
    )

    frechet_max = max(
        1.0e-8,
        float(self_summary["frechet_distance"]["p95"])
        * float(cfg.frechet_self_multiplier),
    )
    mmd_max = max(
        1.0e-10,
        float(self_summary["polynomial_mmd2_biased"]["p95"])
        * float(cfg.mmd_self_multiplier),
    )
    thresholds["frechet_distance_max"] = frechet_max
    thresholds["polynomial_mmd2_biased_max"] = mmd_max
    if float(candidate_summary["frechet_distance"]["median"]) > frechet_max:
        failures.append("frechet_distance")
    if float(candidate_summary["polynomial_mmd2_biased"]["median"]) > mmd_max:
        failures.append("polynomial_mmd2_biased")

    rank_ratio_min = float(cfg.effective_rank_ratio_min)
    nearest_ratio_max = float(cfg.nearest_recall_ratio_max)
    thresholds["effective_rank_ratio_min"] = rank_ratio_min
    thresholds["reference_to_candidate_nn_ratio_max"] = nearest_ratio_max
    if float(candidate_summary["effective_rank_ratio"]["median"]) < rank_ratio_min:
        failures.append("effective_rank_ratio")
    if (
        float(candidate_summary["reference_to_candidate_nn_ratio"]["median"])
        > nearest_ratio_max
    ):
        failures.append("reference_to_candidate_nn_ratio")
    return {
        "accepted": not failures,
        "failures": failures,
        "evaluation_count": int(evaluation_count),
        "embedding_dim": int(reference.shape[1]),
        "repeat_count": int(cfg.repeat_count),
        "self": self_summary,
        "candidate": candidate_summary,
        "thresholds": thresholds,
    }


def cross_validated_shortcut_probe(
    public_embeddings: Array,
    answer_embeddings: Array,
    *,
    seed: int = 20260713,
    fold_count: int = 5,
    target_dim: int = 8,
    ridge_alpha: float = 0.1,
) -> dict[str, Any]:
    """Measure fixed-distribution linear inversion with case-held-out folds."""

    x = _as_matrix(public_embeddings, "public_embeddings")
    y = _as_matrix(answer_embeddings, "answer_embeddings")
    if x.shape[0] != y.shape[0]:
        raise ValueError("shortcut probe inputs have different case counts")
    if x.shape[0] < 6:
        return {
            "case_count": int(x.shape[0]),
            "r2": float("nan"),
            "normalized_rmse": float("nan"),
            "fold_count": 0,
            "status": "insufficient_cases",
        }
    y_centered = y - np.mean(y, axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(y_centered, full_matrices=False)
    dimension = min(int(target_dim), vh.shape[0], max(1, x.shape[0] - 1))
    targets = y_centered @ vh[:dimension].T
    rng = np.random.default_rng(int(seed))
    folds = np.array_split(rng.permutation(x.shape[0]), min(int(fold_count), x.shape[0]))
    squared_error = 0.0
    baseline_error = 0.0
    prediction_count = 0
    for test_indices in folds:
        train_mask = np.ones(x.shape[0], dtype=bool)
        train_mask[test_indices] = False
        train_indices = np.flatnonzero(train_mask)
        x_train, x_test = x[train_indices], x[test_indices]
        y_train, y_test = targets[train_indices], targets[test_indices]
        x_mean = np.mean(x_train, axis=0, keepdims=True)
        x_std = np.std(x_train, axis=0, keepdims=True)
        active = np.ravel(x_std > 1.0e-8)
        x_scale = np.maximum(x_std[:, active], 1.0e-8)
        x_train_z = (x_train[:, active] - x_mean[:, active]) / x_scale
        x_test_z = (x_test[:, active] - x_mean[:, active]) / x_scale
        x_train_z /= np.sqrt(max(1, x_train_z.shape[1]))
        x_test_z /= np.sqrt(max(1, x_test_z.shape[1]))
        y_mean = np.mean(y_train, axis=0, keepdims=True)
        centered_targets = y_train - y_mean
        kernel = x_train_z @ x_train_z.T
        dual = np.linalg.solve(
            kernel + float(ridge_alpha) * np.eye(kernel.shape[0]), centered_targets
        )
        predicted = y_mean + (x_test_z @ x_train_z.T) @ dual
        squared_error += float(np.sum((predicted - y_test) ** 2))
        baseline_error += float(np.sum((y_test - y_mean) ** 2))
        prediction_count += int(y_test.size)
    denominator = max(baseline_error, 1.0e-12)
    return {
        "case_count": int(x.shape[0]),
        "public_dim": int(x.shape[1]),
        "target_dim": int(dimension),
        "fold_count": len(folds),
        "ridge_alpha": float(ridge_alpha),
        "r2": float(1.0 - squared_error / denominator),
        "normalized_rmse": float(np.sqrt(squared_error / denominator)),
        "prediction_scalar_count": int(prediction_count),
        "status": "ok",
    }


def numerical_continuous_feature_grid(case_dir: str | Path) -> Array:
    """Build the historical seven-channel encoder input from a numerical case."""

    case = load_case(case_dir)
    rho = np.asarray(case["hidden_arrays"]["rho_true"], dtype=np.float64)
    g = np.asarray(case["hidden_arrays"]["g_true"], dtype=np.float64)
    public = case["public_arrays"]
    weights = np.asarray(public["velocity_weights"], dtype=np.float64)
    vx = np.asarray(public["vx"], dtype=np.float64)
    vy = np.asarray(public["vy"], dtype=np.float64)
    normalized_g, _ = _normalize_window_density(g, weights)
    fields = _kinetic_fields(normalized_g, weights, vx, vy)
    background = float(case["public_question"].get("background_density", 1.0))
    return np.stack(
        (
            np.log(np.maximum(rho, 1.0e-12)),
            rho / background - 1.0,
            fields["u_x"],
            fields["u_y"],
            fields["temperature"],
            fields["normal_stress"],
            fields["shear_stress"],
        ),
        axis=0,
    )


def audit_continuous_autoencoder_representation(
    case_dirs: Iterable[str | Path],
    checkpoint: str | Path,
    *,
    device: str = "cuda",
    sample_count: int = 8192,
    latent_fit_steps: int = 100,
    latent_fit_lr: float = 0.03,
    gate_config: VPFPGeneratorQualityGateConfig | None = None,
) -> dict[str, Any]:
    """Test whether the current decoder can cover equation-first hidden functions."""

    import torch

    cfg = gate_config or VPFPGeneratorQualityGateConfig()
    checkpoint_payload = torch.load(
        Path(checkpoint), map_location="cpu", weights_only=False
    )
    checkpoint_schema = checkpoint_payload.get("schema")
    del checkpoint_payload
    point_set_schemas = {
        "vpfp_continuous_flow_autoencoder_checkpoint_v1",
        "vpfp_continuous_hermite_autoencoder_checkpoint_v1",
    }
    if checkpoint_schema in point_set_schemas:
        return _audit_flow_autoencoder_representation(
            case_dirs,
            checkpoint,
            checkpoint_schema=str(checkpoint_schema),
            device=device,
            sample_count=sample_count,
            latent_fit_steps=latent_fit_steps,
            latent_fit_lr=latent_fit_lr,
            gate_config=cfg,
        )
    if checkpoint_schema != "vpfp_continuous_autoencoder_checkpoint_v1":
        raise ValueError(
            f"unexpected VPFP autoencoder checkpoint schema: {checkpoint_schema!r}"
        )

    from agent_verifier.vpfp.continuous_training import (
        load_continuous_autoencoder_checkpoint,
        normalize_feature_grid,
    )

    encoder, decoder, payload = load_continuous_autoencoder_checkpoint(
        checkpoint, device=device
    )
    parameter = next(decoder.parameters())
    for value in decoder.parameters():
        value.requires_grad_(False)
    rows: list[dict[str, Any]] = []
    for case_index, case_path in enumerate(case_dirs):
        case = load_case(case_path)
        features = numerical_continuous_feature_grid(case_path)
        feature_tensor = torch.as_tensor(
            features[None], dtype=parameter.dtype, device=parameter.device
        )
        normalized = normalize_feature_grid(
            feature_tensor, payload["feature_normalization"]
        )
        with torch.no_grad():
            initial_latent = encoder(normalized)[0]
        points, target_rho, target_g = _sample_case_phase_points(
            case, sample_count=sample_count, seed=20260713 + case_index
        )
        points_t = torch.as_tensor(points, dtype=parameter.dtype, device=parameter.device)
        rho_t = torch.as_tensor(target_rho, dtype=parameter.dtype, device=parameter.device)
        g_t = torch.as_tensor(target_g, dtype=parameter.dtype, device=parameter.device)
        initial_metrics = _decoder_sample_metrics(
            decoder, points_t, initial_latent, rho_t, g_t
        )
        fitted_latent = initial_latent.detach().clone()
        if int(latent_fit_steps) > 0:
            fitted_latent.requires_grad_(True)
            optimizer = torch.optim.Adam([fitted_latent], lr=float(latent_fit_lr))
            for _ in range(int(latent_fit_steps)):
                optimizer.zero_grad(set_to_none=True)
                predicted = decoder.fields(points_t, fitted_latent)
                pred_rho = predicted["rho"].clamp_min(1.0e-10)
                pred_g = predicted["g"].clamp_min(1.0e-12)
                target_f = rho_t * g_t
                pred_f = pred_rho * pred_g
                loss = (
                    torch.mean((torch.log(pred_rho) - torch.log(rho_t.clamp_min(1.0e-10))) ** 2)
                    + 0.5
                    * torch.mean((torch.log(pred_g) - torch.log(g_t.clamp_min(1.0e-12))) ** 2)
                    + 0.25
                    * torch.mean((pred_f - target_f) ** 2)
                    / torch.mean(target_f**2).clamp_min(1.0e-12)
                )
                loss.backward()
                optimizer.step()
        fitted_metrics = _decoder_sample_metrics(
            decoder, points_t, fitted_latent.detach(), rho_t, g_t
        )
        rows.append(
            {
                "case_id": case["case_id"],
                "case_dir": str(Path(case_path).resolve()),
                "initial": initial_metrics,
                "latent_fitted": fitted_metrics,
            }
        )
    rho_values = np.asarray(
        [row["latent_fitted"]["rho_rel_l2"] for row in rows], dtype=float
    )
    f_values = np.asarray(
        [row["latent_fitted"]["f_rel_l2"] for row in rows], dtype=float
    )
    rho_p90 = float(np.quantile(rho_values, 0.9)) if rows else float("inf")
    f_p90 = float(np.quantile(f_values, 0.9)) if rows else float("inf")
    failures = []
    if not rows:
        failures.append("empty_reference_set")
    if rho_p90 > cfg.representation_rho_rel_l2_p90_max:
        failures.append("rho_rel_l2_p90")
    if f_p90 > cfg.representation_f_rel_l2_p90_max:
        failures.append("f_rel_l2_p90")
    return {
        "schema": "vpfp_continuous_representation_coverage_v1",
        "accepted": not failures,
        "failures": failures,
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_train_claim_scope": payload.get("train_config", {}).get("claim_scope"),
        "case_count": len(rows),
        "sample_count_per_case": int(sample_count),
        "latent_fit_steps": int(latent_fit_steps),
        "thresholds": {
            "rho_rel_l2_p90_max": cfg.representation_rho_rel_l2_p90_max,
            "f_rel_l2_p90_max": cfg.representation_f_rel_l2_p90_max,
        },
        "metrics": {
            "rho_rel_l2": _scalar_summary(rho_values),
            "f_rel_l2": _scalar_summary(f_values),
        },
        "cases": rows,
    }


def _point_set_representation_backend(
    checkpoint_schema: str,
    decoder_config: dict[str, Any],
) -> str:
    if checkpoint_schema == "vpfp_continuous_flow_autoencoder_checkpoint_v1":
        return "v3_conditional_fullspace_velocity_flow"
    if (
        checkpoint_schema
        != "vpfp_continuous_hermite_autoencoder_checkpoint_v1"
    ):
        raise ValueError(
            f"unsupported point-set checkpoint schema: {checkpoint_schema!r}"
        )
    if not bool(decoder_config.get("structured_spectral_latent", False)):
        return "v4_global_squared_hermite_fullspace"
    if bool(decoder_config.get("structured_direct_coefficients", False)):
        return "v5_structured_direct_squared_hermite_fullspace"
    return "v5_structured_nonlinear_squared_hermite_fullspace"


def _audit_flow_autoencoder_representation(
    case_dirs: Iterable[str | Path],
    checkpoint: str | Path,
    *,
    checkpoint_schema: str,
    device: str,
    sample_count: int,
    latent_fit_steps: int,
    latent_fit_lr: float,
    gate_config: VPFPGeneratorQualityGateConfig,
) -> dict[str, Any]:
    import torch

    from agent_verifier.vpfp.continuous_flow_training import (
        fit_structured_direct_case_latent,
        load_flow_autoencoder_checkpoint,
        load_flow_reference_case,
        load_hermite_autoencoder_checkpoint,
        normalize_observation_features,
        prepare_hermite_coefficient_targets,
        prepare_structured_direct_projection_operators,
    )

    if checkpoint_schema == "vpfp_continuous_flow_autoencoder_checkpoint_v1":
        loader = load_flow_autoencoder_checkpoint
    elif (
        checkpoint_schema
        == "vpfp_continuous_hermite_autoencoder_checkpoint_v1"
    ):
        loader = load_hermite_autoencoder_checkpoint
    else:
        raise ValueError(
            f"unsupported point-set checkpoint schema: {checkpoint_schema!r}"
        )

    encoder, decoder, payload = loader(checkpoint, device=device)
    representation_backend = _point_set_representation_backend(
        checkpoint_schema,
        payload.get("decoder_config", {}),
    )
    parameter = next(decoder.parameters())
    for value in decoder.parameters():
        value.requires_grad_(False)
    latent_mean = torch.as_tensor(
        payload["latent_normalization"]["mean"],
        dtype=parameter.dtype,
        device=parameter.device,
    )
    latent_std = torch.as_tensor(
        payload["latent_normalization"]["std"],
        dtype=parameter.dtype,
        device=parameter.device,
    )
    analytic_direct = representation_backend == (
        "v5_structured_direct_squared_hermite_fullspace"
    )
    effective_latent_fit_steps = (
        0 if analytic_direct else int(latent_fit_steps)
    )
    projection_operator_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for case_index, case_path in enumerate(case_dirs):
        case = load_case(case_path)
        flow_case = load_flow_reference_case(case_path)
        condition = torch.as_tensor(
            flow_case["equation_condition"],
            dtype=parameter.dtype,
            device=parameter.device,
        )
        projection_report = None
        if analytic_direct:
            prepare_hermite_coefficient_targets(
                [flow_case],
                max_total_degree=int(decoder.config.hermite_total_degree),
            )
            grid_key = tuple(
                (
                    key,
                    np.asarray(flow_case[key]).shape,
                    np.asarray(flow_case[key]).tobytes(),
                )
                for key in ("t", "x", "y")
            )
            projection_operators = projection_operator_cache.get(grid_key)
            if projection_operators is None:
                projection_operators = (
                    prepare_structured_direct_projection_operators(
                        decoder,
                        flow_case,
                    )
                )
                projection_operator_cache[grid_key] = projection_operators
            projected, projection_report = (
                fit_structured_direct_case_latent(
                    decoder,
                    flow_case,
                    projection_operators=projection_operators,
                    rho_feasible_fraction=float(
                        payload.get("train_config", {}).get(
                            "hermite_direct_rho_feasible_fraction",
                            0.95,
                        )
                    ),
                )
            )
            initial_latent = torch.as_tensor(
                projected,
                dtype=parameter.dtype,
                device=parameter.device,
            )
            initialization_backend = (
                "deterministic_chebyshev_fourier_hermite_projection"
            )
        else:
            coordinates = torch.as_tensor(
                flow_case["observation_coordinates"][None],
                dtype=parameter.dtype,
                device=parameter.device,
            )
            features = torch.as_tensor(
                flow_case["observation_features"][None],
                dtype=parameter.dtype,
                device=parameter.device,
            )
            normalized = normalize_observation_features(
                features, payload["feature_normalization"]
            )
            with torch.no_grad():
                normalized_latent = encoder(
                    coordinates, normalized, condition
                )[0]
                initial_latent = normalized_latent * latent_std + latent_mean
            initialization_backend = "neural_point_set_encoder"
        points, target_rho, target_g = _sample_case_phase_points(
            case, sample_count=sample_count, seed=20260713 + case_index
        )
        points_t = torch.as_tensor(
            points, dtype=parameter.dtype, device=parameter.device
        )
        rho_t = torch.as_tensor(
            target_rho, dtype=parameter.dtype, device=parameter.device
        )
        g_t = torch.as_tensor(
            target_g, dtype=parameter.dtype, device=parameter.device
        )
        initial_metrics = _decoder_sample_metrics(
            decoder,
            points_t,
            initial_latent,
            rho_t,
            g_t,
            equation_condition=condition,
        )
        fitted_latent = initial_latent.detach().clone()
        if effective_latent_fit_steps > 0:
            fitted_latent.requires_grad_(True)
            optimizer = torch.optim.Adam(
                [fitted_latent], lr=float(latent_fit_lr)
            )
            for _ in range(effective_latent_fit_steps):
                optimizer.zero_grad(set_to_none=True)
                predicted = decoder.fields(
                    points_t,
                    fitted_latent,
                    equation_condition=condition,
                )
                pred_rho = predicted["rho"].clamp_min(1.0e-10)
                pred_g = predicted["g"].clamp_min(1.0e-12)
                target_f = rho_t * g_t
                pred_f = pred_rho * pred_g
                loss = (
                    torch.mean(
                        (
                            torch.log(pred_rho)
                            - torch.log(rho_t.clamp_min(1.0e-10))
                        )
                        ** 2
                    )
                    + 0.5
                    * torch.mean(
                        (
                            torch.log(pred_g)
                            - torch.log(g_t.clamp_min(1.0e-12))
                        )
                        ** 2
                    )
                    + 0.25
                    * torch.mean((pred_f - target_f) ** 2)
                    / torch.mean(target_f**2).clamp_min(1.0e-12)
                )
                loss.backward()
                optimizer.step()
        fitted_metrics = _decoder_sample_metrics(
            decoder,
            points_t,
            fitted_latent.detach(),
            rho_t,
            g_t,
            equation_condition=condition,
        )
        rows.append(
            {
                "case_id": case["case_id"],
                "case_dir": str(Path(case_path).resolve()),
                "initialization_backend": initialization_backend,
                "projection": projection_report,
                "initial": initial_metrics,
                "latent_fitted": fitted_metrics,
            }
        )

    rho_values = np.asarray(
        [row["latent_fitted"]["rho_rel_l2"] for row in rows], dtype=float
    )
    f_values = np.asarray(
        [row["latent_fitted"]["f_rel_l2"] for row in rows], dtype=float
    )
    rho_p90 = float(np.quantile(rho_values, 0.9)) if rows else float("inf")
    f_p90 = float(np.quantile(f_values, 0.9)) if rows else float("inf")
    failures = []
    if not rows:
        failures.append("empty_reference_set")
    if rho_p90 > gate_config.representation_rho_rel_l2_p90_max:
        failures.append("rho_rel_l2_p90")
    if f_p90 > gate_config.representation_f_rel_l2_p90_max:
        failures.append("f_rel_l2_p90")
    return {
        "schema": "vpfp_continuous_representation_coverage_v1",
        "accepted": not failures,
        "failures": failures,
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_schema": payload.get("schema"),
        "representation_backend": representation_backend,
        "checkpoint_train_claim_scope": payload.get("train_config", {}).get(
            "claim_scope"
        ),
        "case_count": len(rows),
        "sample_count_per_case": int(sample_count),
        "latent_fit_steps": int(latent_fit_steps),
        "requested_latent_fit_steps": int(latent_fit_steps),
        "effective_latent_fit_steps": effective_latent_fit_steps,
        "latent_initialization_backend": (
            "deterministic_chebyshev_fourier_hermite_projection"
            if analytic_direct
            else "neural_point_set_encoder"
        ),
        "thresholds": {
            "rho_rel_l2_p90_max": (
                gate_config.representation_rho_rel_l2_p90_max
            ),
            "f_rel_l2_p90_max": gate_config.representation_f_rel_l2_p90_max,
        },
        "metrics": {
            "rho_rel_l2": _scalar_summary(rho_values),
            "f_rel_l2": _scalar_summary(f_values),
        },
        "cases": rows,
    }


def validate_generator_quality_report(report: dict[str, Any] | None) -> list[str]:
    if report is None:
        return ["generator quality report was not provided"]
    failures = []
    if report.get("schema") != "vpfp_generator_quality_audit_v1":
        failures.append("unexpected generator quality report schema")
    if report.get("claim_scope") != "generator_distribution_fidelity_coverage_and_shortcut_audit":
        failures.append("unexpected generator quality claim scope")
    if report.get("ready_for_agent_iteration") is not True:
        failures.append("generator quality audit did not pass")
    if report.get("failures"):
        failures.append("generator quality report contains failures")
    discretization = report.get("reference_discretization")
    if not isinstance(discretization, dict):
        failures.append("generator quality report lacks reference discretization evidence")
    elif discretization.get("schema") != "vpfp_reference_discretization_evidence_v1":
        failures.append("unexpected generator quality reference discretization schema")
    elif discretization.get("accepted") is not True:
        failures.append("generator quality reference discretization evidence failed")
    if not isinstance(report.get("representation_coverage"), dict):
        failures.append("generator quality report lacks representation coverage")
    return failures


class _ReferenceTransform:
    def __init__(self, mean: Array, scale: Array, basis: Array):
        self.mean = mean
        self.scale = scale
        self.basis = basis

    @classmethod
    def fit(cls, values: Array, *, max_dim: int) -> "_ReferenceTransform":
        mean = np.mean(values, axis=0, keepdims=True)
        scale = np.std(values, axis=0, keepdims=True)
        positive_scale = scale[scale > 1.0e-8]
        scale_floor = max(
            float(np.median(positive_scale)) * 1.0e-3
            if positive_scale.size
            else 0.0,
            1.0e-6,
        )
        scale = np.maximum(scale, scale_floor)
        dimension = max(1, min(int(max_dim), values.shape[1]))
        if dimension == values.shape[1]:
            basis = np.eye(dimension, dtype=float)
        else:
            rng = np.random.default_rng(
                104729 + 131 * values.shape[1] + 17 * dimension
            )
            random_matrix = rng.normal(size=(values.shape[1], dimension))
            basis = np.linalg.qr(random_matrix, mode="reduced")[0].T
        return cls(mean, scale, basis)

    def apply(self, values: Array) -> Array:
        standardized = np.clip(
            (values - self.mean) / self.scale,
            -25.0,
            25.0,
        )
        return standardized @ self.basis.T


def _distribution_metrics(reference: Array, candidate: Array, *, k: int) -> dict[str, float]:
    distances = _pairwise_distances(reference, candidate)
    ref_self = _pairwise_distances(reference, reference)
    cand_self = _pairwise_distances(candidate, candidate)
    np.fill_diagonal(ref_self, np.inf)
    np.fill_diagonal(cand_self, np.inf)
    k_eff = min(int(k), reference.shape[0] - 1, candidate.shape[0] - 1)
    ref_radius = np.partition(ref_self, k_eff - 1, axis=1)[:, k_eff - 1]
    cand_radius = np.partition(cand_self, k_eff - 1, axis=1)[:, k_eff - 1]
    precision = float(np.mean(np.any(distances <= ref_radius[:, None], axis=0)))
    recall = float(np.mean(np.any(distances <= cand_radius[None, :], axis=1)))
    density = float(np.mean(np.sum(distances <= ref_radius[:, None], axis=0) / k_eff))
    coverage = float(np.mean(np.min(distances, axis=1) <= ref_radius))
    reference_nn_scale = max(float(np.median(ref_radius)), 1.0e-12)
    return {
        "frechet_distance": _frechet_distance(reference, candidate),
        "polynomial_mmd2_biased": _polynomial_mmd2_biased(reference, candidate),
        "precision": precision,
        "recall": recall,
        "density": density,
        "coverage": coverage,
        "candidate_to_reference_nn_ratio": float(
            np.median(np.min(distances, axis=0)) / reference_nn_scale
        ),
        "reference_to_candidate_nn_ratio": float(
            np.median(np.min(distances, axis=1)) / reference_nn_scale
        ),
        "effective_rank_ratio": float(
            _effective_rank(candidate) / max(_effective_rank(reference), 1.0e-12)
        ),
    }


def _frechet_distance(first: Array, second: Array) -> float:
    mean_gap = np.sum((np.mean(first, axis=0) - np.mean(second, axis=0)) ** 2)
    cov_first = _covariance(first)
    cov_second = _covariance(second)
    root_first = _psd_sqrt(cov_first)
    middle = root_first @ cov_second @ root_first
    trace_cross = float(np.trace(_psd_sqrt(middle)))
    value = mean_gap + np.trace(cov_first) + np.trace(cov_second) - 2.0 * trace_cross
    return float(max(value, 0.0))


def _polynomial_mmd2_biased(first: Array, second: Array) -> float:
    dimension = max(1, first.shape[1])
    k_xx = (first @ first.T / dimension + 1.0) ** 3
    k_yy = (second @ second.T / dimension + 1.0) ** 3
    k_xy = (first @ second.T / dimension + 1.0) ** 3
    return float(max(np.mean(k_xx) + np.mean(k_yy) - 2.0 * np.mean(k_xy), 0.0))


def _effective_rank(values: Array) -> float:
    covariance = _covariance(values)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    total = float(np.sum(eigenvalues))
    if total <= 1.0e-12:
        return 0.0
    probabilities = eigenvalues[eigenvalues > 0.0] / total
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def _covariance(values: Array) -> Array:
    centered = values - np.mean(values, axis=0, keepdims=True)
    return centered.T @ centered / max(1, values.shape[0] - 1)


def _psd_sqrt(matrix: Array) -> Array:
    symmetric = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(symmetric)
    return (vectors * np.sqrt(np.maximum(values, 0.0))) @ vectors.T


def _pairwise_distances(first: Array, second: Array) -> Array:
    squared = (
        np.sum(first**2, axis=1, keepdims=True)
        + np.sum(second**2, axis=1)[None, :]
        - 2.0 * first @ second.T
    )
    return np.sqrt(np.maximum(squared, 0.0))


def _metric_summary(rows: list[dict[str, float]]) -> dict[str, Any]:
    return {
        name: _scalar_summary(np.asarray([row[name] for row in rows], dtype=float))
        for name in rows[0]
    }


def _scalar_summary(values: Array) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {"n": 0, "mean": float("nan"), "p05": float("nan"), "median": float("nan"), "p90": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def _public_question_embedding(case: dict[str, Any], config: VPFPEmbeddingConfig) -> tuple[Array, list[str]]:
    public = case["public_arrays"]
    blocks = (
        ("rho0", np.log(np.maximum(np.asarray(public["rho0"], dtype=float), 1.0e-12)), (8, 8), 11),
        ("g0", np.log(np.maximum(np.asarray(public["g0"], dtype=float), 1.0e-12)), (8, 8, 7, 7), 23),
        ("source", np.sign(public["source"]) * np.log1p(np.abs(public["source"])), (4, 8, 8, 7, 7), 37),
    )
    values: list[float] = [float(case["collision_nu"]), float(case["time"].t1 - case["time"].t0)]
    names = ["collision_nu", "time_horizon"]
    for block_name, block, shape, seed in blocks:
        sampled = _subsample_to_shape(np.asarray(block, dtype=float), shape)
        sketch = _count_sketch(sampled.ravel(), config.public_sketch_width, seed=seed)
        values.extend(sketch.tolist())
        values.extend((float(np.mean(sampled)), float(np.std(sampled)), float(np.sqrt(np.mean(sampled**2)))))
        names.extend(f"{block_name}.sketch{index:03d}" for index in range(config.public_sketch_width))
        names.extend((f"{block_name}.mean", f"{block_name}.std", f"{block_name}.rms"))
    return np.asarray(values, dtype=np.float64), names


def _count_sketch(values: Array, width: int, *, seed: int) -> Array:
    rng = np.random.default_rng(int(seed))
    buckets = rng.integers(0, int(width), size=values.size)
    signs = 2.0 * rng.integers(0, 2, size=values.size) - 1.0
    counts = np.bincount(buckets, minlength=int(width)).astype(float)
    sums = np.bincount(buckets, weights=signs * values, minlength=int(width))
    return sums / np.sqrt(np.maximum(counts, 1.0))


def _subsample_to_shape(values: Array, target_shape: tuple[int, ...]) -> Array:
    if values.ndim != len(target_shape):
        raise ValueError(f"cannot subsample shape {values.shape} to {target_shape}")
    output = values
    for axis, count in enumerate(target_shape):
        indices = _sample_indices(output.shape[axis], int(count))
        output = np.take(output, indices, axis=axis)
    return output


def _kinetic_fields(g: Array, weights: Array, vx: Array, vy: Array) -> dict[str, Array]:
    vx_grid = vx[None, None, None, :, None]
    vy_grid = vy[None, None, None, None, :]
    ux = np.sum(g * weights * vx_grid, axis=(-2, -1))
    uy = np.sum(g * weights * vy_grid, axis=(-2, -1))
    centered_x = vx_grid - ux[..., None, None]
    centered_y = vy_grid - uy[..., None, None]
    var_x = np.sum(g * weights * centered_x**2, axis=(-2, -1))
    var_y = np.sum(g * weights * centered_y**2, axis=(-2, -1))
    covariance = np.sum(g * weights * centered_x * centered_y, axis=(-2, -1))
    return {
        "u_x": ux,
        "u_y": uy,
        "temperature": 0.5 * (var_x + var_y),
        "normal_stress": var_x - var_y,
        "shear_stress": covariance,
    }


def _velocity_moment_fields(
    g: Array,
    weights: Array,
    vx: Array,
    vy: Array,
    *,
    max_degree: int,
) -> dict[str, Array]:
    output = {}
    for degree in range(1, int(max_degree) + 1):
        for power_x in range(degree + 1):
            power_y = degree - power_x
            monomial = vx[:, None] ** power_x * vy[None, :] ** power_y
            output[f"velocity_moment_{power_x}_{power_y}"] = np.sum(
                g * weights * monomial, axis=(-2, -1)
            )
    return output


def _normalize_window_density(g: Array, weights: Array) -> tuple[Array, Array]:
    mass = np.sum(g * weights, axis=(-2, -1))
    return g / np.maximum(mass[..., None, None], 1.0e-12), mass


def _spatial_band_powers(field: Array, *, edges: tuple[float, ...]) -> Array:
    nx, ny = field.shape
    kx = np.fft.fftfreq(nx) * nx
    ky = np.fft.fftfreq(ny) * ny
    radius = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
    power = np.abs(np.fft.fft2(field) / np.sqrt(nx * ny)) ** 2
    bounds = (-np.inf, *edges, np.inf)
    return np.asarray(
        [np.mean(power[(radius > low) & (radius <= high)]) if np.any((radius > low) & (radius <= high)) else 0.0 for low, high in zip(bounds[:-1], bounds[1:])],
        dtype=float,
    )


def _resample_trajectory(values: Array, count: int) -> Array:
    source = np.linspace(0.0, 1.0, np.asarray(values).shape[0])
    target = np.linspace(0.0, 1.0, int(count))
    return np.interp(target, source, np.asarray(values, dtype=float))


def _temporal_relative_change(values: Array) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean((array - array[:1]) ** 2)) / max(np.sqrt(np.mean(array**2)), 1.0e-12))


def _sample_indices(size: int, count: int) -> Array:
    if int(size) < 1:
        raise ValueError("cannot sample an empty axis")
    return np.rint(np.linspace(0, int(size) - 1, int(count))).astype(int)


def _sample_case_phase_points(
    case: dict[str, Any], *, sample_count: int, seed: int
) -> tuple[Array, Array, Array]:
    public = case["public_arrays"]
    rho = np.asarray(case["hidden_arrays"]["rho_true"], dtype=float)
    g = np.asarray(case["hidden_arrays"]["g_true"], dtype=float)
    total = int(g.size)
    count = min(int(sample_count), total)
    rng = np.random.default_rng(int(seed))
    flat = np.sort(rng.choice(total, size=count, replace=False))
    indices = np.unravel_index(flat, g.shape)
    points = np.stack(
        (
            np.asarray(public["t"])[indices[0]],
            np.asarray(public["x"])[indices[1]],
            np.asarray(public["y"])[indices[2]],
            np.asarray(public["vx"])[indices[3]],
            np.asarray(public["vy"])[indices[4]],
        ),
        axis=1,
    )
    return points, rho[indices[:3]], g[indices]


def _decoder_sample_metrics(
    decoder,
    points,
    latent,
    target_rho,
    target_g,
    *,
    equation_condition=None,
) -> dict[str, float]:
    import torch

    field_kwargs = {}
    if equation_condition is not None:
        field_kwargs["equation_condition"] = equation_condition
    with torch.no_grad():
        prediction = decoder.fields(points, latent, **field_kwargs)
    pred_rho = prediction["rho"]
    pred_g = prediction["g"]
    target_f = target_rho * target_g
    pred_f = pred_rho * pred_g

    def rel(first, second):
        return float(
            torch.linalg.vector_norm(first - second)
            / torch.linalg.vector_norm(second).clamp_min(1.0e-12)
        )

    return {
        "rho_rel_l2": rel(pred_rho, target_rho),
        "g_rel_l2": rel(pred_g, target_g),
        "f_rel_l2": rel(pred_f, target_f),
        "log_g_rmse": float(
            torch.sqrt(
                torch.mean(
                    (
                        torch.log(pred_g.clamp_min(1.0e-12))
                        - torch.log(target_g.clamp_min(1.0e-12))
                    )
                    ** 2
                )
            )
        ),
    }


def _validate_case_arrays(
    rho: Array,
    g: Array,
    electric: Array,
    weights: Array,
    vx: Array,
    vy: Array,
    t: Array,
) -> None:
    if g.shape != rho.shape + (vx.size, vy.size):
        raise ValueError(f"inconsistent rho/g shape: {rho.shape}, {g.shape}")
    if electric.shape != (rho.shape[0], 2, rho.shape[1], rho.shape[2]):
        raise ValueError(f"inconsistent electric shape: {electric.shape}")
    if weights.shape != (vx.size, vy.size):
        raise ValueError(f"inconsistent velocity weight shape: {weights.shape}")
    if t.size != rho.shape[0]:
        raise ValueError("time axis does not match hidden arrays")


def _as_matrix(values: Array, name: str) -> Array:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array
