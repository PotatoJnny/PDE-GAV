#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent_verifier.fusion_simulation_first.coverage import (
    effective_rank,
    prdc_metrics,
    rbf_mmd,
    sliced_wasserstein,
    standardize_features,
)
from agent_verifier.neural_generator.continuous_geometry import (
    continuous_geometry_spec_from_contract,
)
from agent_verifier.neural_generator.structured_ns_answer_first import (
    STRUCTURED_NS_MMS_ABI_VERSION,
    ContinuousStructuredNSMMS,
    StructuredNSCaseCoefficients,
    StructuredNSProjectionConfig,
    structured_ns_mms_provenance,
    write_structured_continuous_case_state,
)
from agent_verifier.neural_generator.structured_ns_distribution import (
    StructuredCoefficientPCA,
    compatible_projection_signatures,
    load_structured_coefficient_corpus,
)
from agent_verifier.neural_generator.triplane_dataset import (
    build_kite_baffle_geometry,
    case_metrics,
    write_case,
    write_geometry_contract,
)
from agent_verifier.neural_generator.triplane_verifier import (
    continuous_mms_case_contract_failures,
    generated_fields_mms_residual_rel,
)


PHYSICAL_METRICS = (
    "velocity_rms",
    "forcing_rms",
    "time_derivative_rms",
    "vorticity_rms",
    "velocity_laplacian_rms",
    "pressure_gradient_rms",
    "pressure_rms",
    "repeat_initial_difficulty",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize structured-flow coefficient samples as exact continuous "
            "Navier-Stokes MMS cases and compare them with held-out D0 projections."
        )
    )
    parser.add_argument("--sample-audit", required=True)
    parser.add_argument("--reference-manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--generated-cases", type=int, default=64)
    parser.add_argument("--reference-cases", type=int, default=64)
    parser.add_argument("--case-id-prefix", default="structured_flow")
    parser.add_argument("--selection-seed", type=int, default=20260811)
    parser.add_argument("--baseline-repeats", type=int, default=32)
    parser.add_argument("--source-batch-points", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if min(int(args.generated_cases), int(args.source_batch_points)) < 1:
        raise SystemExit("generated case and batch sizes must be positive")
    if int(args.reference_cases) < 0:
        raise SystemExit("reference case count cannot be negative")
    if int(args.baseline_repeats) < 2:
        raise SystemExit("empirical D0 baseline needs at least two repeats")
    if not str(args.case_id_prefix).strip():
        raise SystemExit("case ID prefix cannot be empty")
    output = Path(args.out).resolve()
    if output.exists():
        raise SystemExit(f"refusing to reuse structured flow materialization: {output}")
    output.mkdir(parents=True)
    generated_root = output / "generated_cases"
    generated_root.mkdir()
    reference_pca_root = output / "reference_pca_cases"
    if int(args.reference_cases) > 0:
        reference_pca_root.mkdir()

    sample_report_path = Path(args.sample_audit).resolve()
    sample_report = json.loads(sample_report_path.read_text())
    if sample_report.get("schema") != "ns2d_structured_coefficient_flow_sample_audit_v1":
        raise SystemExit("unexpected structured flow sample audit schema")
    sample_path = sample_report_path.parent / str(sample_report["sample_file"])
    with np.load(sample_path) as arrays:
        generated_coefficients = np.asarray(
            arrays["coefficients"],
            dtype=np.float32,
        )
    if generated_coefficients.ndim != 5 or generated_coefficients.shape[1] != 2:
        raise SystemExit("generated coefficients must have shape [N,2,R,Kx,Ky]")
    if int(args.generated_cases) > generated_coefficients.shape[0]:
        raise SystemExit("requested more generated cases than the sample file contains")

    device = _device(str(args.device))
    pca_path = Path(str(sample_report["pca_checkpoint"])).resolve()
    pca = StructuredCoefficientPCA.load(pca_path, map_location=device).with_rank(
        int(sample_report["pca_rank"])
    )
    reference = load_structured_coefficient_corpus(
        args.reference_manifest,
        max_cases=0 if int(args.reference_cases) > 0 else 1,
        device="cpu",
    )
    if not compatible_projection_signatures(
        pca.source_signature,
        dict(reference.manifest["encoding_signature"]),
    ):
        raise SystemExit("held-out coefficients are incompatible with the flow PCA")
    if int(args.reference_cases) > reference.case_count:
        raise SystemExit("requested more reference cases than the corpus contains")

    rng = np.random.default_rng(int(args.selection_seed))
    generated_indices = np.sort(
        rng.choice(
            generated_coefficients.shape[0],
            size=int(args.generated_cases),
            replace=False,
        )
    )
    reference_indices = (
        np.sort(
            rng.choice(
                reference.case_count,
                size=int(args.reference_cases),
                replace=False,
            )
        )
        if int(args.reference_cases) > 0
        else np.asarray([], dtype=np.int64)
    )
    generated_selected = generated_coefficients[generated_indices]
    if int(args.reference_cases) > 0:
        reference_full = reference.coefficients[reference_indices].to(device=device)
        with _torch().no_grad():
            reference_pca = pca.reconstruct(reference_full).float().cpu().numpy()
        reference_full = reference_full.float().cpu().numpy()
    else:
        reference_full = np.empty(
            (0, *pca.coefficient_shape),
            dtype=np.float32,
        )
        reference_pca = reference_full.copy()

    projection_config = StructuredNSProjectionConfig.from_dict(
        dict(pca.source_signature["projection_config"])
    )
    projection_config.validate()
    expected_shape = (
        2,
        projection_config.time_coefficient_count,
        int(projection_config.spatial_mode_x) + 1,
        int(projection_config.spatial_mode_y) + 1,
    )
    for name, values in (
        ("generated", generated_selected),
        ("reference_full", reference_full),
        ("reference_pca", reference_pca),
    ):
        if tuple(values.shape[1:]) != expected_shape:
            raise SystemExit(
                f"{name} coefficient shape {tuple(values.shape[1:])} "
                f"does not match {expected_shape}"
            )

    geometry = build_kite_baffle_geometry()
    geometry_spec = continuous_geometry_spec_from_contract(geometry)
    t, viscosity, source_case = _source_case_metadata(
        pca.source_signature,
        preferred_case_id=(
            reference.case_ids[int(reference_indices[0])]
            if int(args.reference_cases) > 0
            else reference.case_ids[0]
        ),
        expected_geometry_hash=geometry.contract_hash,
    )
    write_geometry_contract(output / "geometry_contract.json", geometry)
    started = time.perf_counter()
    generated_rows = _evaluate_group(
        generated_selected,
        group="generated",
        ids=[
            f"{str(args.case_id_prefix)}_{index:06d}"
            for index in range(len(generated_selected))
        ],
        source_indices=[int(index) for index in generated_indices],
        output_root=generated_root,
        config=projection_config,
        geometry=geometry,
        geometry_spec=geometry_spec,
        t=t,
        viscosity=viscosity,
        source_batch_points=int(args.source_batch_points),
        device=device,
        sample_report=sample_report,
    )
    reference_ids = [reference.case_ids[int(index)] for index in reference_indices]
    reference_full_rows = (
        _evaluate_group(
            reference_full,
            group="reference_full_projected",
            ids=reference_ids,
            source_indices=[int(index) for index in reference_indices],
            output_root=None,
            config=projection_config,
            geometry=geometry,
            geometry_spec=geometry_spec,
            t=t,
            viscosity=viscosity,
            source_batch_points=int(args.source_batch_points),
            device=device,
            sample_report=sample_report,
        )
        if int(args.reference_cases) > 0
        else []
    )
    reference_pca_rows = (
        _evaluate_group(
            reference_pca,
            group="reference_pca_reconstructed",
            ids=reference_ids,
            source_indices=[int(index) for index in reference_indices],
            output_root=reference_pca_root,
            config=projection_config,
            geometry=geometry,
            geometry_spec=geometry_spec,
            t=t,
            viscosity=viscosity,
            source_batch_points=int(args.source_batch_points),
            device=device,
            sample_report=sample_report,
        )
        if int(args.reference_cases) > 0
        else []
    )

    physical_distribution = _physical_distribution(
        generated_rows=generated_rows,
        reference_full_rows=reference_full_rows,
        reference_pca_rows=reference_pca_rows,
        seed=int(args.selection_seed),
        baseline_repeats=int(args.baseline_repeats),
    )
    contract_rows = [
        {
            "case_id": row["case_id"],
            "failures": list(row["continuous_contract_failures"]),
        }
        for row in generated_rows
    ]
    report = {
        "schema": "ns2d_structured_coefficient_flow_materialization_audit_v1",
        "sample_audit": str(sample_report_path),
        "sample_audit_sha256": _file_sha256(sample_report_path),
        "sample_file": str(sample_path),
        "flow_checkpoint": str(sample_report["flow_checkpoint"]),
        "flow_step": int(sample_report["flow_step"]),
        "flow_state": str(sample_report["flow_state"]),
        "sampling_policy": str(
            sample_report.get("sampling_policy", "rectified_flow")
        ),
        "pca_checkpoint": str(pca_path),
        "pca_rank": int(pca.rank),
        "generator_representation": str(
            sample_report.get("generator_representation", "pca_whitened_scores")
        ),
        "generator_feature_dim": int(
            sample_report.get("generator_feature_dim", sample_report["pca_rank"])
        ),
        "pca_used_for_training": bool(
            sample_report.get("pca_used_for_training", True)
        ),
        "full_rank_diagonal_velocity": bool(
            sample_report.get("full_rank_diagonal_velocity", False)
        ),
        "pca_role": str(
            sample_report.get("pca_role", "generator_training_and_decoding")
        ),
        "direct_standardizer": sample_report.get("direct_standardizer"),
        "reference_manifest": str(Path(args.reference_manifest).resolve()),
        "reference_source_case": source_case,
        "projection_config": projection_config.to_dict(),
        "generated_cases": len(generated_rows),
        "reference_cases": len(reference_full_rows),
        "reference_pca_cases_root": (
            str(reference_pca_root) if int(args.reference_cases) > 0 else None
        ),
        "case_id_prefix": str(args.case_id_prefix),
        "selection_seed": int(args.selection_seed),
        "source_batch_points": int(args.source_batch_points),
        "elapsed_seconds": float(time.perf_counter() - started),
        "continuous_contract_passed": sum(
            not row["failures"] for row in contract_rows
        ),
        "reference_pca_contract_passed": sum(
            not row["continuous_contract_failures"]
            for row in reference_pca_rows
        ),
        "continuous_contract": contract_rows,
        "continuous_hard_metrics": _hard_summary(generated_rows),
        "physical_distribution": physical_distribution,
        "continuous_exact_physical_statistics": {
            name: {
                group_label: physical_distribution["groups"][group_name][name]
                for group_label, group_name in (
                    (
                        "answer_first_continuous_exact",
                        "generated",
                    ),
                    (
                        "reference_full_projected",
                        "reference_full_projected",
                    ),
                    (
                        "reference_pca_reconstructed",
                        "reference_pca_reconstructed",
                    ),
                )
                if group_name in physical_distribution["groups"]
            }
            for name in PHYSICAL_METRICS
        },
        "cases": generated_rows,
        "reference_full_projected_cases": reference_full_rows,
        "reference_pca_reconstructed_cases": reference_pca_rows,
    }
    (output / "answer_first_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (output / "materialization_audit.md").write_text(_markdown(report))
    print(
        json.dumps(
            {
                "out": str(output),
                "flow_step": report["flow_step"],
                "generated_cases": report["generated_cases"],
                "reference_cases": report["reference_cases"],
                "continuous_contract_passed": report[
                    "continuous_contract_passed"
                ],
                "reference_pca_contract_passed": report[
                    "reference_pca_contract_passed"
                ],
                "joint_physical_distribution": physical_distribution[
                    "joint_log_metric_distribution"
                ],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


def _evaluate_group(
    coefficients: np.ndarray,
    *,
    group: str,
    ids: list[str],
    source_indices: list[int],
    output_root: Path | None,
    config,
    geometry,
    geometry_spec,
    t,
    viscosity: float,
    source_batch_points: int,
    device,
    sample_report,
) -> list[dict[str, Any]]:
    rows = []
    for local_index, (case_id, source_index, coefficient) in enumerate(
        zip(ids, source_indices, coefficients, strict=True)
    ):
        case_started = time.perf_counter()
        field = _field_from_coefficients(
            coefficient,
            config=config,
            geometry_spec=geometry_spec,
            viscosity=viscosity,
            t=t,
            device=device,
        )
        decoded = field.sample_grid(
            x=geometry.x,
            y=geometry.y,
            t=t,
            batch_points=int(source_batch_points),
        )
        exact = dict(decoded.metadata["continuous_exact_physical_statistics"])
        finite_difference = case_metrics(
            velocity=decoded.velocity,
            pressure=decoded.pressure,
            forcing=decoded.forcing,
            t=decoded.t,
            geometry=geometry,
        )
        hard = _continuous_hard_metrics(decoded, geometry=geometry)
        contract_failures: list[str] = []
        if output_root is not None:
            case_dir = output_root / case_id
            finite_difference = write_case(
                case_dir,
                case_id=case_id,
                geometry=geometry,
                t=decoded.t,
                velocity=decoded.velocity,
                pressure=decoded.pressure,
                forcing=decoded.forcing,
                viscosity=float(viscosity),
                case_spec={
                    "case_id": case_id,
                    "family": (
                        "d0_learned_structured_coefficient_flow"
                        if group == "generated"
                        else "heldout_d0_pca_reconstruction"
                    ),
                    "private_sample_index": int(source_index),
                },
                diagnostics={
                    **finite_difference,
                    **{f"continuous_exact_{name}": value for name, value in exact.items()},
                    "mms_residual_rel": generated_fields_mms_residual_rel(decoded),
                    "public_forcing_derivation": (
                        "continuous_hidden_solution_coordinate_autodiff"
                    ),
                },
                precomputed_metrics=finite_difference,
                case_source="d0_learned_structured_flow_continuous_mms",
            )
            write_structured_continuous_case_state(case_dir, field=field)
            metadata = {
                **(
                    _generator_provenance(config)
                    if group == "generated"
                    else _reference_provenance(config)
                ),
                "pca_checkpoint": str(sample_report["pca_checkpoint"]),
                "pca_rank": int(sample_report["pca_rank"]),
                "source_index": int(source_index),
                "continuous_exact_physical_statistics": exact,
                "continuous_contract_failures": [],
            }
            if group == "generated":
                metadata.update(
                    {
                        "generator_representation": str(
                            sample_report.get(
                                "generator_representation",
                                "pca_whitened_scores",
                            )
                        ),
                        "generator_feature_dim": int(
                            sample_report.get(
                                "generator_feature_dim",
                                sample_report["pca_rank"],
                            )
                        ),
                        "pca_used_for_training": bool(
                            sample_report.get("pca_used_for_training", True)
                        ),
                        "full_rank_diagonal_velocity": bool(
                            sample_report.get(
                                "full_rank_diagonal_velocity", False
                            )
                        ),
                        "pca_role": str(
                            sample_report.get(
                                "pca_role",
                                "generator_training_and_decoding",
                            )
                        ),
                        "flow_checkpoint": str(sample_report["flow_checkpoint"]),
                        "flow_step": int(sample_report["flow_step"]),
                        "flow_state": str(sample_report["flow_state"]),
                        "sampling_policy": str(
                            sample_report.get(
                                "sampling_policy",
                                "rectified_flow",
                            )
                        ),
                    }
                )
                if sample_report.get("direct_standardizer"):
                    metadata["direct_standardizer"] = str(
                        sample_report["direct_standardizer"]
                    )
                    metadata["direct_standardizer_sha256"] = str(
                        sample_report["direct_standardizer_sha256"]
                    )
            else:
                metadata["heldout_source_case_id"] = case_id
            metadata_path = case_dir / "generator_metadata.json"
            metadata_path.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n"
            )
            contract_failures = continuous_mms_case_contract_failures(case_dir)
            metadata["continuous_contract_failures"] = contract_failures
            metadata_path.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n"
            )
        rows.append(
            {
                "case_id": case_id,
                "group": group,
                "source_index": int(source_index),
                "continuous_exact_metrics": exact,
                "materialized_finite_difference_metrics": finite_difference,
                "continuous_hard_metrics": hard,
                "continuous_contract_failures": contract_failures,
                "elapsed_seconds": float(time.perf_counter() - case_started),
            }
        )
        print(
            "[structured-flow-materialize] "
            f"group={group} completed={local_index + 1}/{len(ids)} "
            f"case={case_id} velocity_rms={exact['velocity_rms']:.6g} "
            f"elapsed_seconds={rows[-1]['elapsed_seconds']:.3f}",
            flush=True,
        )
        del decoded, field
    return rows


def _field_from_coefficients(
    coefficients,
    *,
    config,
    geometry_spec,
    viscosity: float,
    t,
    device,
):
    values = np.asarray(coefficients, dtype=np.float64)
    return ContinuousStructuredNSMMS(
        StructuredNSCaseCoefficients(
            psi=values[0].reshape(config.time_coefficient_count, -1),
            pressure=values[1].reshape(config.time_coefficient_count, -1),
        ),
        config=config,
        geometry_spec=geometry_spec,
        continuous_mms_abi=STRUCTURED_NS_MMS_ABI_VERSION,
        domain_bounds=geometry_spec.domain_bounds,
        viscosity=float(viscosity),
        t0=float(t[0]),
        t1=float(t[-1]),
        device=device,
    )


def _source_case_metadata(
    source_signature: dict[str, Any],
    *,
    preferred_case_id: str,
    expected_geometry_hash: str,
) -> tuple[np.ndarray, float, str]:
    manifest_path = Path(str(source_signature["split_manifest"])).resolve()
    manifest = json.loads(manifest_path.read_text())
    rows = [
        row
        for split_rows in manifest.get("splits", {}).values()
        for row in split_rows
        if str(row.get("case_id")) == str(preferred_case_id)
    ]
    if not rows:
        raise ValueError(f"could not find held-out case {preferred_case_id}")
    case_dir = _resolve_case_dir(rows[0], split_manifest_path=manifest_path)
    with np.load(case_dir / "hidden_truth_cache.npz") as hidden:
        t = np.asarray(hidden["t"], dtype=np.float64)
    question = json.loads((case_dir / "public_question.json").read_text())
    actual_hash = str(question.get("geometry", {}).get("contract_hash", ""))
    if actual_hash != str(expected_geometry_hash):
        raise ValueError("held-out source case uses a different geometry contract")
    return t, float(question["viscosity"]), str(case_dir)


def _resolve_case_dir(row: dict[str, Any], *, split_manifest_path: Path) -> Path:
    case_id = str(row["case_id"])
    raw = Path(str(row.get("case_dir", "")))
    candidates = (
        raw,
        ROOT / raw,
        split_manifest_path.parent / raw,
        split_manifest_path.parent / "cases" / case_id,
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(case_id)


def _generator_provenance(config) -> dict[str, Any]:
    return {
        **structured_ns_mms_provenance(
            config,
            continuous_mms_abi=STRUCTURED_NS_MMS_ABI_VERSION,
        ),
        "continuous_mms_abi": STRUCTURED_NS_MMS_ABI_VERSION,
        "generator_schema": "ns2d_structured_coefficient_flow_materialization_v1",
        "generator_backend_and_version": (
            "d0_learned_structured_coefficient_rectified_flow_continuous_mms_v1"
        ),
        "truth_primary": "continuous_hidden_streamfunction_pressure",
        "training_source": "D0_train_split_structured_coefficients",
        "learned_distribution_model_used": True,
        "rectified_flow_model_used": True,
        "diffusion_model_used": False,
        "learned_decoder_used": False,
        "diagnostic_field_stack_allowed": False,
        "velocity_projection_enabled": False,
        "D0_distribution_gate_enabled": False,
    }


def _reference_provenance(config) -> dict[str, Any]:
    return {
        **structured_ns_mms_provenance(
            config,
            continuous_mms_abi=STRUCTURED_NS_MMS_ABI_VERSION,
        ),
        "continuous_mms_abi": STRUCTURED_NS_MMS_ABI_VERSION,
        "generator_schema": "ns2d_structured_coefficient_reference_materialization_v1",
        "generator_backend_and_version": (
            "heldout_d0_pca_reconstruction_continuous_mms_v1"
        ),
        "truth_primary": "continuous_hidden_streamfunction_pressure",
        "training_source": "heldout_D0_structured_coefficients",
        "learned_distribution_model_used": False,
        "rectified_flow_model_used": False,
        "diffusion_model_used": False,
        "learned_decoder_used": False,
        "diagnostic_field_stack_allowed": False,
        "velocity_projection_enabled": False,
        "D0_distribution_gate_enabled": False,
    }


def _continuous_hard_metrics(decoded, *, geometry) -> dict[str, float]:
    velocity = np.asarray(decoded.velocity, dtype=np.float64)
    forcing = np.asarray(decoded.forcing, dtype=np.float64)
    residual = np.asarray(decoded.residual, dtype=np.float64)
    outer = np.asarray(geometry.outer_boundary_mask, dtype=bool)
    obstacle = np.asarray(geometry.obstacle_mask, dtype=bool)
    return {
        "divergence_abs_max": float(np.max(np.abs(decoded.divergence))),
        "residual_abs_max": float(np.max(np.abs(residual))),
        "outer_boundary_velocity_abs_max": float(
            np.max(np.abs(velocity[:, :, outer]))
        ),
        "obstacle_velocity_abs_max": float(
            np.max(np.abs(velocity[:, :, obstacle]))
        ),
        "obstacle_forcing_abs_max": float(
            np.max(np.abs(forcing[:, :, obstacle]))
        ),
    }


def _physical_distribution(
    *,
    generated_rows,
    reference_full_rows,
    reference_pca_rows,
    seed: int,
    baseline_repeats: int,
) -> dict[str, Any]:
    groups = {"generated": generated_rows}
    if reference_full_rows:
        groups["reference_full_projected"] = reference_full_rows
    if reference_pca_rows:
        groups["reference_pca_reconstructed"] = reference_pca_rows
    summaries = {
        group: {
            name: _summary(
                [float(row["continuous_exact_metrics"][name]) for row in rows]
            )
            for name in PHYSICAL_METRICS
        }
        for group, rows in groups.items()
    }
    comparisons = {}
    generated_matrix = _physical_matrix(generated_rows)
    for reference_name in (
        "reference_full_projected",
        "reference_pca_reconstructed",
    ):
        if reference_name not in groups:
            continue
        reference_rows = groups[reference_name]
        reference_matrix = _physical_matrix(reference_rows)
        comparisons[reference_name] = {
            name: {
                "median_ratio": _safe_ratio(
                    summaries["generated"][name]["median"],
                    summaries[reference_name][name]["median"],
                ),
                "mean_ratio": _safe_ratio(
                    summaries["generated"][name]["mean"],
                    summaries[reference_name][name]["mean"],
                ),
                "iqr_ratio": _safe_ratio(
                    summaries["generated"][name]["iqr"],
                    summaries[reference_name][name]["iqr"],
                ),
                "quantile_wasserstein_log": _quantile_wasserstein(
                    np.log(np.maximum(generated_matrix[:, index], 1.0e-12)),
                    np.log(np.maximum(reference_matrix[:, index], 1.0e-12)),
                ),
            }
            for index, name in enumerate(PHYSICAL_METRICS)
        }
    joint = None
    if reference_pca_rows:
        reference_log = np.log(
            np.maximum(_physical_matrix(reference_pca_rows), 1.0e-12)
        )
        generated_log = np.log(np.maximum(generated_matrix, 1.0e-12))
        reference_z, generated_z = standardize_features(
            reference_log,
            generated_log,
        )
        comparison_count = min(
            generated_z.shape[0],
            reference_z.shape[0] // 2,
        )
        if comparison_count < 2:
            raise ValueError(
                "physical empirical baseline needs two D0 groups with at "
                "least two cases each"
            )
        comparison_rng = np.random.default_rng(int(seed))
        comparison_indices = comparison_rng.permutation(
            reference_z.shape[0]
        )[:comparison_count]
        reference_comparison = reference_z[comparison_indices]
        generated_comparison = generated_z[:comparison_count]
        empirical_trials = _empirical_two_sample_trials(
            reference_z,
            sample_count=comparison_count,
            repeats=int(baseline_repeats),
            seed=int(seed),
        )
        first_baseline = empirical_trials[0]
        joint = {
            "metrics": list(PHYSICAL_METRICS),
            "comparison_case_count": int(comparison_count),
            "empirical_baseline_repeats": int(baseline_repeats),
            "empirical_d0_trials": empirical_trials,
            "generated_vs_reference_pca_prdc": prdc_metrics(
                reference_comparison,
                generated_comparison,
            ),
            "generated_vs_reference_pca_rbf_mmd": rbf_mmd(
                reference_comparison,
                generated_comparison,
            ),
            "generated_vs_reference_pca_sliced_wasserstein": sliced_wasserstein(
                reference_comparison,
                generated_comparison,
                seed=int(seed),
            ),
            "effective_rank": {
                "generated": effective_rank(generated_z),
                "reference_pca": effective_rank(reference_z),
            },
            "reference_half_baseline_prdc": {
                name: first_baseline[name]
                for name in ("precision", "recall", "density", "coverage")
            },
            "reference_half_baseline_rbf_mmd": first_baseline["mmd"],
            "reference_half_baseline_sliced_wasserstein": (
                first_baseline["sliced_wasserstein"]
            ),
        }
    return {
        "groups": summaries,
        "generated_over_reference": comparisons,
        "joint_log_metric_distribution": joint,
    }


def _physical_matrix(rows) -> np.ndarray:
    return np.asarray(
        [
            [float(row["continuous_exact_metrics"][name]) for name in PHYSICAL_METRICS]
            for row in rows
        ],
        dtype=np.float64,
    )


def _hard_summary(rows) -> dict[str, Any]:
    names = tuple(rows[0]["continuous_hard_metrics"])
    return {
        name: _summary(
            [float(row["continuous_hard_metrics"][name]) for row in rows]
        )
        for name in names
    }



def _empirical_two_sample_trials(
    reference: np.ndarray,
    *,
    sample_count: int,
    repeats: int,
    seed: int,
) -> list[dict[str, float]]:
    """Measure natural D0-vs-D0 variation at the candidate sample size."""

    values = np.asarray(reference, dtype=np.float64)
    if values.ndim != 2 or 2 * int(sample_count) > values.shape[0]:
        raise ValueError("empirical D0 trials need two disjoint equal-size groups")
    random = np.random.default_rng(int(seed))
    rows = []
    for repeat_index in range(int(repeats)):
        indices = random.permutation(values.shape[0])
        left = values[indices[:sample_count]]
        right = values[indices[sample_count : 2 * sample_count]]
        prdc = prdc_metrics(left, right)
        rows.append(
            {
                "repeat": repeat_index,
                "mmd": float(rbf_mmd(left, right)),
                "sliced_wasserstein": float(
                    sliced_wasserstein(
                        left,
                        right,
                        seed=int(seed) + repeat_index,
                    )
                ),
                "precision": float(prdc["precision"]),
                "recall": float(prdc["recall"]),
                "density": float(prdc["density"]),
                "coverage": float(prdc["coverage"]),
            }
        )
    return rows

def _summary(values) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "q10": float(np.quantile(array, 0.1)),
        "q25": float(np.quantile(array, 0.25)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "q90": float(np.quantile(array, 0.9)),
        "max": float(np.max(array)),
        "iqr": float(np.quantile(array, 0.75) - np.quantile(array, 0.25)),
    }


def _quantile_wasserstein(first, second, *, quantiles: int = 256) -> float:
    probabilities = np.linspace(0.0, 1.0, int(quantiles))
    return float(
        np.mean(
            np.abs(
                np.quantile(np.asarray(first), probabilities)
                - np.quantile(np.asarray(second), probabilities)
            )
        )
    )


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if abs(float(denominator)) <= 1.0e-12:
        return None
    return float(numerator) / float(denominator)


def _markdown(report: dict[str, Any]) -> str:
    distribution = report["physical_distribution"]
    lines = [
        "# Structured NS Flow Materialization Audit",
        "",
        f"- flow step: `{report['flow_step']}`",
        f"- generated cases: `{report['generated_cases']}`",
        f"- held-out reference cases: `{report['reference_cases']}`",
        (
            "- continuous contract passed: "
            f"`{report['continuous_contract_passed']}/{report['generated_cases']}`"
        ),
    ]
    if "reference_pca_reconstructed" not in distribution["groups"]:
        lines.extend(
            [
                "",
                "| metric | generated median |",
                "|---|---:|",
            ]
        )
        for name in PHYSICAL_METRICS:
            generated = distribution["groups"]["generated"][name]["median"]
            lines.append(f"| {name} | {generated:.6g} |")
        return "\n".join(lines) + "\n"
    lines.extend(
        [
            "",
            "| metric | generated median | reference PCA median | ratio |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in PHYSICAL_METRICS:
        generated = distribution["groups"]["generated"][name]["median"]
        reference = distribution["groups"]["reference_pca_reconstructed"][name][
            "median"
        ]
        ratio = distribution["generated_over_reference"][
            "reference_pca_reconstructed"
        ][name]["median_ratio"]
        lines.append(f"| {name} | {generated:.6g} | {reference:.6g} | {ratio:.6g} |")
    return "\n".join(lines) + "\n"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _torch():
    import torch

    return torch


def _device(requested: str):
    torch = _torch()
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {requested}, but CUDA is unavailable")
    return device


if __name__ == "__main__":
    main()
