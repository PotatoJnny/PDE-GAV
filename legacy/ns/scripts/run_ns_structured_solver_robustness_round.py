from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent_verifier.neural_generator.structured_ns_answer_first import (
    STRUCTURED_NS_MMS_ABI_VERSION,
    load_structured_continuous_case,
    structured_ns_mms_provenance,
    write_structured_continuous_case_state,
)
from agent_verifier.neural_generator.structured_ns_distribution import (
    StructuredCoefficientPCA,
)
from agent_verifier.neural_generator.structured_ns_robustness import (
    PERTURBATION_FAMILIES,
    field_at_amplitude,
    make_perturbation,
    paired_solver_metrics,
    question_delta_metrics,
)
from agent_verifier.neural_generator.triplane_agent_solver import (
    _call_solver,
    _extract_velocity,
    _load_public_question,
    _load_submission_module,
    _velocity_error_metrics,
)
from agent_verifier.neural_generator.triplane_dataset import (
    build_kite_baffle_geometry,
    case_metrics,
    write_case,
    write_geometry_contract,
)
from agent_verifier.neural_generator.triplane_verifier import (
    generated_fields_mms_residual_rel,
    require_continuous_mms_case_contract,
)


SCHEMA = "ns2d_structured_solver_orthogonal_robustness_round_v1"
PERTURBATION_SCHEMA = "ns2d_structured_solver_perturbation_case_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize three controlled perturbations for one completed structured "
            "NS adversarial round and evaluate the round/final solvers pairwise."
        )
    )
    parser.add_argument("--source-run-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--round-index", type=int, required=True)
    parser.add_argument("--pca-checkpoint", required=True)
    parser.add_argument("--expected-cases", type=int, default=1024)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--target-question-norm", type=float, default=0.01)
    parser.add_argument("--reference-velocity-scale", type=float, required=True)
    parser.add_argument("--reference-forcing-scale", type=float, required=True)
    parser.add_argument("--spectral-max-mode", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--source-batch-points", type=int, default=8192)
    parser.add_argument("--calibration-tolerance", type=float, default=0.02)
    parser.add_argument("--calibration-iterations", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--streaming-batch-size",
        type=int,
        default=0,
        help="Materialize/evaluate this many source cases at a time; 0 disables streaming.",
    )
    parser.add_argument(
        "--drop-hidden-cache-after-evaluation",
        action="store_true",
        help=(
            "After both solver roles are recorded, remove each perturbed case's "
            "regenerable hidden_truth_cache.npz while retaining canonical truth."
        ),
    )
    parser.add_argument("--materialize-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_args(args)
    loop_root = _resolve_loop_root(Path(args.source_run_root))
    round_index = int(args.round_index)
    round_name = f"round_{round_index:03d}"
    source_round = loop_root / "rounds" / round_name
    final_round = loop_root / "rounds" / "round_004"
    base_cases_root = (
        source_round
        / "structured_generator_materialization"
        / "generated_cases"
    )
    round_solver = source_round / "submission.py"
    final_solver = final_round / "submission.py"
    for path in (
        loop_root / "final_summary.json",
        base_cases_root,
        round_solver,
        final_solver,
        Path(args.pca_checkpoint),
    ):
        if not path.exists():
            raise SystemExit(f"missing robustness input: {path}")
    final_summary = json.loads((loop_root / "final_summary.json").read_text())
    if int(final_summary.get("rounds_completed", -1)) != 5:
        raise SystemExit("source adversarial loop is not a completed five-round run")

    base_case_dirs = sorted(
        path
        for path in base_cases_root.iterdir()
        if path.is_dir() and (path / "hidden_continuous_case.pt").is_file()
    )
    if len(base_case_dirs) != int(args.expected_cases):
        raise SystemExit(
            f"expected {args.expected_cases} source cases in {round_name}, "
            f"found {len(base_case_dirs)}"
        )
    if int(args.max_cases) > 0:
        base_case_dirs = base_case_dirs[: int(args.max_cases)]

    output_root = Path(args.output_root)
    round_root = output_root / "rounds" / round_name
    round_root.mkdir(parents=True, exist_ok=True)
    receipt = _round_receipt(
        args=args,
        loop_root=loop_root,
        source_round=source_round,
        base_case_dirs=base_case_dirs,
        round_solver=round_solver,
        final_solver=final_solver,
    )
    _write_or_validate_json(round_root / "round_config.json", receipt)

    pca = StructuredCoefficientPCA.load(args.pca_checkpoint, map_location="cpu")
    geometry = build_kite_baffle_geometry()
    write_geometry_contract(round_root / "geometry_contract.json", geometry)

    if bool(args.drop_hidden_cache_after_evaluation):
        materialization, evaluations = _run_streaming_round(
            args=args,
            round_index=round_index,
            round_root=round_root,
            base_case_dirs=base_case_dirs,
            pca=pca,
            geometry=geometry,
            round_solver=round_solver,
            final_solver=final_solver,
        )
    else:
        if not bool(args.evaluate_only):
            _materialize_round(
                args=args,
                round_index=round_index,
                round_root=round_root,
                base_case_dirs=base_case_dirs,
                pca=pca,
                geometry=geometry,
            )
        materialization = _materialization_summary(
            round_root=round_root,
            base_case_dirs=base_case_dirs,
        )
        _write_json_atomic(round_root / "materialization_summary.json", materialization)
        if not bool(args.materialize_only):
            evaluations = _evaluate_round(
                args=args,
                round_index=round_index,
                round_root=round_root,
                base_case_dirs=base_case_dirs,
                round_solver=round_solver,
                final_solver=final_solver,
            )
        else:
            evaluations = []
    summary = {
        "schema": SCHEMA,
        "round": int(round_index),
        "round_name": round_name,
        "source_run_root": str(loop_root.parent),
        "source_loop_root": str(loop_root),
        "case_count": len(base_case_dirs),
        "perturbation_families": list(PERTURBATION_FAMILIES),
        "target_question_norm": float(args.target_question_norm),
        "materialization": materialization,
        "evaluations": evaluations,
        "complete": bool(
            materialization["complete"]
            and (bool(args.materialize_only) or all(item["complete"] for item in evaluations))
        ),
    }
    _write_json_atomic(round_root / "round_robustness_summary.json", summary)
    print(json.dumps({
        "round": round_index,
        "case_count": len(base_case_dirs),
        "materialized": materialization["materialized_case_count"],
        "evaluations": [item["solver_roles"] for item in evaluations],
        "complete": summary["complete"],
    }, indent=2, sort_keys=True))


def _validate_args(args) -> None:
    if not 0 <= int(args.round_index) < 5:
        raise SystemExit("round-index must lie in [0,4]")
    if int(args.expected_cases) < 1 or int(args.max_cases) < 0:
        raise SystemExit("case counts must be non-negative and expected-cases positive")
    if not 0.0 < float(args.target_question_norm) < 1.0:
        raise SystemExit("target-question-norm must lie in (0,1)")
    if min(
        float(args.reference_velocity_scale),
        float(args.reference_forcing_scale),
        float(args.calibration_tolerance),
    ) <= 0.0:
        raise SystemExit("reference scales and calibration tolerance must be positive")
    if int(args.calibration_iterations) < 1:
        raise SystemExit("calibration-iterations must be positive")
    if int(args.streaming_batch_size) < 0:
        raise SystemExit("streaming-batch-size must be non-negative")
    if bool(args.materialize_only) and bool(args.evaluate_only):
        raise SystemExit("materialize-only and evaluate-only are mutually exclusive")
    if bool(args.drop_hidden_cache_after_evaluation):
        if int(args.streaming_batch_size) < 1:
            raise SystemExit(
                "drop-hidden-cache-after-evaluation requires streaming-batch-size >= 1"
            )
        if bool(args.materialize_only) or bool(args.evaluate_only):
            raise SystemExit(
                "cache compaction requires the combined materialize/evaluate workflow"
            )


def _run_streaming_round(
    *,
    args,
    round_index: int,
    round_root: Path,
    base_case_dirs: list[Path],
    pca,
    geometry,
    round_solver: Path,
    final_solver: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    batch_size = int(args.streaming_batch_size)
    for start in range(0, len(base_case_dirs), batch_size):
        stop = min(start + batch_size, len(base_case_dirs))
        batch = base_case_dirs[start:stop]
        print(
            f"round {round_index}: streaming source cases {start + 1}-{stop}/"
            f"{len(base_case_dirs)}",
            flush=True,
        )
        _materialize_round(
            args=args,
            round_index=round_index,
            round_root=round_root,
            base_case_dirs=batch,
            pca=pca,
            geometry=geometry,
            case_index_offset=start,
        )
        _evaluate_round(
            args=args,
            round_index=round_index,
            round_root=round_root,
            base_case_dirs=batch,
            round_solver=round_solver,
            final_solver=final_solver,
            finalize=False,
        )
        for base_case_dir in batch:
            for family in PERTURBATION_FAMILIES:
                _compact_hidden_cache(
                    _perturbed_case_dir(round_root, base_case_dir.name, family)
                )
        print(
            f"round {round_index}: compacted perturbed cases through source {stop}/"
            f"{len(base_case_dirs)}",
            flush=True,
        )

    materialization = _materialization_summary(
        round_root=round_root,
        base_case_dirs=base_case_dirs,
        expect_compacted=True,
    )
    _write_json_atomic(round_root / "materialization_summary.json", materialization)
    if not materialization["complete"]:
        raise RuntimeError("streaming materialization or cache compaction is incomplete")
    evaluations = _evaluate_round(
        args=args,
        round_index=round_index,
        round_root=round_root,
        base_case_dirs=base_case_dirs,
        round_solver=round_solver,
        final_solver=final_solver,
        finalize=True,
    )
    return materialization, evaluations


def _compact_hidden_cache(case_dir: Path) -> None:
    cache_path = case_dir / "hidden_truth_cache.npz"
    receipt_path = case_dir / "cache_compaction_receipt.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        if (
            receipt.get("schema") != "continuous_truth_cache_compaction_v1"
            or receipt.get("case_id") != case_dir.name
        ):
            raise RuntimeError(f"invalid cache compaction receipt in {case_dir}")
        if cache_path.is_file():
            cache_path.unlink()
        return
    if not cache_path.is_file():
        raise RuntimeError(
            f"hidden cache is absent without a compaction receipt in {case_dir}"
        )
    require_continuous_mms_case_contract(case_dir)
    receipt = {
        "schema": "continuous_truth_cache_compaction_v1",
        "case_id": case_dir.name,
        "removed_cache": cache_path.name,
        "removed_cache_size_bytes": int(cache_path.stat().st_size),
        "removed_cache_sha256": _sha256(cache_path),
        "canonical_truth": "hidden_continuous_case.pt",
        "canonical_truth_sha256": _sha256(case_dir / "hidden_continuous_case.pt"),
        "public_arrays_sha256": _sha256(case_dir / "public_arrays.npz"),
        "regeneration": (
            "evaluate hidden_continuous_case.pt on the public x/y/t query grid"
        ),
        "cache_removed": True,
    }
    hidden_manifest_path = case_dir / "hidden_truth_manifest.json"
    hidden_manifest = json.loads(hidden_manifest_path.read_text())
    hidden_manifest.update(
        {
            "hidden_arrays": {},
            "hidden_arrays_role": (
                "not_persisted_regenerate_from_canonical_continuous_truth"
            ),
            "hidden_truth_cache_compacted": True,
        }
    )
    _write_json_atomic(hidden_manifest_path, hidden_manifest)
    continuous_manifest_path = case_dir / "hidden_continuous_manifest.json"
    continuous_manifest = json.loads(continuous_manifest_path.read_text())
    continuous_manifest.update(
        {
            "sampled_grid_cache": (
                "not persisted; regenerate from hidden_continuous_case.pt"
            ),
            "sampled_grid_cache_compacted": True,
        }
    )
    _write_json_atomic(continuous_manifest_path, continuous_manifest)
    require_continuous_mms_case_contract(case_dir)
    _write_json_atomic(receipt_path, receipt)
    cache_path.unlink()


def _resolve_loop_root(source: Path) -> Path:
    candidate = source / "agent_adversarial_loop"
    return candidate if candidate.is_dir() else source


def _round_receipt(
    *,
    args,
    loop_root: Path,
    source_round: Path,
    base_case_dirs: list[Path],
    round_solver: Path,
    final_solver: Path,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "source_loop_root": str(loop_root.resolve()),
        "source_final_summary_sha256": _sha256(loop_root / "final_summary.json"),
        "round": int(args.round_index),
        "source_round": str(source_round.resolve()),
        "source_case_count": len(base_case_dirs),
        "source_first_case": base_case_dirs[0].name,
        "source_last_case": base_case_dirs[-1].name,
        "pca_checkpoint": str(Path(args.pca_checkpoint).resolve()),
        "pca_checkpoint_sha256": _sha256(Path(args.pca_checkpoint)),
        "round_solver": str(round_solver.resolve()),
        "round_solver_sha256": _sha256(round_solver),
        "final_solver": str(final_solver.resolve()),
        "final_solver_sha256": _sha256(final_solver),
        "perturbation_families": list(PERTURBATION_FAMILIES),
        "target_question_norm": float(args.target_question_norm),
        "question_norm": (
            "max(initial_velocity_delta_rms/reference_velocity_scale,"
            "forcing_delta_rms/reference_forcing_scale)"
        ),
        "reference_velocity_scale": float(args.reference_velocity_scale),
        "reference_forcing_scale": float(args.reference_forcing_scale),
        "spectral_max_mode": int(args.spectral_max_mode),
        "seed": int(args.seed),
        "source_batch_points": int(args.source_batch_points),
        "calibration_tolerance": float(args.calibration_tolerance),
        "calibration_iterations": int(args.calibration_iterations),
        "streaming_batch_size": int(args.streaming_batch_size),
        "drop_hidden_cache_after_evaluation": bool(
            args.drop_hidden_cache_after_evaluation
        ),
        "retained_perturbed_case_artifacts": [
            "public_arrays.npz",
            "hidden_continuous_case.pt",
            "metadata_and_contract_json",
        ],
    }


def _materialize_round(
    *,
    args,
    round_index: int,
    round_root: Path,
    base_case_dirs: list[Path],
    pca,
    geometry,
    case_index_offset: int = 0,
) -> None:
    total = len(base_case_dirs) * len(PERTURBATION_FAMILIES)
    completed = 0
    for local_case_index, base_case_dir in enumerate(base_case_dirs):
        case_index = int(case_index_offset) + int(local_case_index)
        missing = [
            family
            for family in PERTURBATION_FAMILIES
            if not _perturbation_case_complete(
                _perturbed_case_dir(round_root, base_case_dir.name, family),
                base_case_id=base_case_dir.name,
                family=family,
                target_question_norm=float(args.target_question_norm),
            )
        ]
        completed += len(PERTURBATION_FAMILIES) - len(missing)
        if not missing:
            continue
        with np.load(base_case_dir / "public_arrays.npz") as arrays:
            base_public = {name: np.asarray(arrays[name]) for name in arrays.files}
        with np.load(base_case_dir / "hidden_truth_cache.npz") as arrays:
            base_velocity = np.asarray(arrays["u_true"], dtype=np.float64)
        _validate_geometry(base_public, geometry=geometry)
        base_forcing = np.asarray(base_public["forcing"], dtype=np.float64)
        times = np.asarray(base_public["t"], dtype=np.float64)
        base_field = load_structured_continuous_case(
            base_case_dir,
            device=str(args.device),
        )
        for family in missing:
            seed = _direction_seed(
                base_seed=int(args.seed),
                round_index=int(round_index),
                case_index=int(case_index),
                family=family,
            )
            perturbation = make_perturbation(
                base_field,
                family=family,
                seed=seed,
                pca=pca,
                spectral_max_mode=int(args.spectral_max_mode),
            )
            field, decoded, amplitude, calibration = _calibrate(
                args=args,
                base_field=base_field,
                perturbation=perturbation,
                base_velocity=base_velocity,
                base_forcing=base_forcing,
                fluid_mask=geometry.fluid_mask,
                x=geometry.x,
                y=geometry.y,
                t=times,
            )
            target_case_dir = _perturbed_case_dir(
                round_root,
                base_case_dir.name,
                family,
            )
            _write_perturbation_case(
                target_case_dir=target_case_dir,
                base_case_dir=base_case_dir,
                family=family,
                seed=seed,
                amplitude=amplitude,
                perturbation=perturbation,
                field=field,
                decoded=decoded,
                calibration=calibration,
                geometry=geometry,
                target_question_norm=float(args.target_question_norm),
            )
            completed += 1
            if completed % 16 == 0 or completed == total:
                print(
                    f"round {round_index}: materialized {completed}/{total}",
                    flush=True,
                )
        del base_field
        _empty_cuda_cache()


def _calibrate(
    *,
    args,
    base_field,
    perturbation,
    base_velocity,
    base_forcing,
    fluid_mask,
    x,
    y,
    t,
):
    probe = field_at_amplitude(base_field, perturbation, amplitude=1.0)
    probe_decoded = probe.sample_grid(
        x=x,
        y=y,
        t=t,
        batch_points=int(args.source_batch_points),
    )
    probe_metrics = question_delta_metrics(
        base_velocity=base_velocity,
        base_forcing=base_forcing,
        perturbed_velocity=probe_decoded.velocity,
        perturbed_forcing=probe_decoded.forcing,
        fluid_mask=fluid_mask,
        velocity_scale=float(args.reference_velocity_scale),
        forcing_scale=float(args.reference_forcing_scale),
    )
    probe_norm = float(probe_metrics["question_delta_norm"])
    if not np.isfinite(probe_norm) or probe_norm <= 0.0:
        raise RuntimeError("perturbation probe has zero or non-finite public-question norm")
    amplitude = float(args.target_question_norm) / probe_norm
    history = [{"amplitude": 1.0, **probe_metrics}]
    decoded = None
    final_metrics = None
    for _ in range(int(args.calibration_iterations)):
        field = field_at_amplitude(
            base_field,
            perturbation,
            amplitude=float(amplitude),
        )
        decoded = field.sample_grid(
            x=x,
            y=y,
            t=t,
            batch_points=int(args.source_batch_points),
        )
        final_metrics = question_delta_metrics(
            base_velocity=base_velocity,
            base_forcing=base_forcing,
            perturbed_velocity=decoded.velocity,
            perturbed_forcing=decoded.forcing,
            fluid_mask=fluid_mask,
            velocity_scale=float(args.reference_velocity_scale),
            forcing_scale=float(args.reference_forcing_scale),
        )
        history.append({"amplitude": float(amplitude), **final_metrics})
        actual = float(final_metrics["question_delta_norm"])
        relative_miss = abs(actual - float(args.target_question_norm)) / float(
            args.target_question_norm
        )
        if relative_miss <= float(args.calibration_tolerance):
            break
        if not np.isfinite(actual) or actual <= 0.0:
            raise RuntimeError("perturbation calibration produced an invalid norm")
        amplitude *= float(args.target_question_norm) / actual
    assert decoded is not None and final_metrics is not None
    relative_miss = abs(
        float(final_metrics["question_delta_norm"])
        - float(args.target_question_norm)
    ) / float(args.target_question_norm)
    if relative_miss > float(args.calibration_tolerance):
        raise RuntimeError(
            f"perturbation calibration missed target by {relative_miss:.3%}"
        )
    if perturbation.family == "spectral_complement":
        initial_scaled = float(
            final_metrics["initial_velocity_delta_reference_scaled"]
        )
        if initial_scaled > max(1.0e-10, float(args.target_question_norm) * 1.0e-6):
            raise RuntimeError(
                "tau-squared spectral perturbation changed the initial condition"
            )
    return field, decoded, float(amplitude), {
        "target_question_norm": float(args.target_question_norm),
        "tolerance": float(args.calibration_tolerance),
        "history": history,
        "final": final_metrics,
    }


def _write_perturbation_case(
    *,
    target_case_dir: Path,
    base_case_dir: Path,
    family: str,
    seed: int,
    amplitude: float,
    perturbation,
    field,
    decoded,
    calibration,
    geometry,
    target_question_norm: float,
) -> None:
    if target_case_dir.exists():
        raise RuntimeError(f"refusing to overwrite incomplete case: {target_case_dir}")
    target_case_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{target_case_dir.name}.tmp.",
            dir=target_case_dir.parent,
        )
    )
    case_id = target_case_dir.name
    exact = dict(decoded.metadata["continuous_exact_physical_statistics"])
    finite = case_metrics(
        velocity=decoded.velocity,
        pressure=decoded.pressure,
        forcing=decoded.forcing,
        t=decoded.t,
        geometry=geometry,
    )
    hard = _continuous_hard_metrics(decoded, geometry=geometry)
    metadata = {
        "schema": PERTURBATION_SCHEMA,
        "base_case_id": base_case_dir.name,
        "base_case_dir": str(base_case_dir.resolve()),
        "perturbation_family": family,
        "perturbation_seed": int(seed),
        "coefficient_amplitude": float(amplitude),
        "target_question_norm": float(target_question_norm),
        "direction": perturbation.metadata,
        "calibration": calibration,
        "continuous_exact_physical_statistics": exact,
        "continuous_hard_metrics": hard,
        "continuous_mms_abi": STRUCTURED_NS_MMS_ABI_VERSION,
        "generator_schema": "ns2d_structured_solver_orthogonal_robustness_v1",
        "generator_backend_and_version": (
            "structured_ns_orthogonal_perturbation_continuous_mms_v1"
        ),
        "truth_primary": "continuous_hidden_streamfunction_pressure",
        "learned_distribution_model_used": False,
        "rectified_flow_model_used": False,
        "diffusion_model_used": False,
        "learned_decoder_used": False,
        "diagnostic_field_stack_allowed": False,
        "velocity_projection_enabled": False,
        "D0_distribution_gate_enabled": False,
        **structured_ns_mms_provenance(perturbation.config),
    }
    try:
        write_case(
            temporary,
            case_id=case_id,
            geometry=geometry,
            t=decoded.t,
            velocity=decoded.velocity,
            pressure=decoded.pressure,
            forcing=decoded.forcing,
            viscosity=float(decoded.viscosity),
            case_spec={
                "case_id": case_id,
                "family": "structured_ns_local_orthogonal_robustness",
                "base_case_id": base_case_dir.name,
                "perturbation_family": family,
                "perturbation_seed": int(seed),
                "target_question_norm": float(target_question_norm),
            },
            diagnostics={
                **finite,
                **{f"continuous_exact_{name}": value for name, value in exact.items()},
                **{f"continuous_hard_{name}": value for name, value in hard.items()},
                "mms_residual_rel": generated_fields_mms_residual_rel(decoded),
                "public_forcing_derivation": (
                    "continuous_hidden_solution_coordinate_autodiff"
                ),
            },
            precomputed_metrics=finite,
            case_source="structured_ns_local_orthogonal_robustness_mms",
        )
        write_structured_continuous_case_state(temporary, field=field)
        (temporary / "generator_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        (temporary / "perturbation_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        require_continuous_mms_case_contract(temporary)
        os.replace(temporary, target_case_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _continuous_hard_metrics(decoded, *, geometry) -> dict[str, float]:
    velocity = np.asarray(decoded.velocity, dtype=np.float64)
    forcing = np.asarray(decoded.forcing, dtype=np.float64)
    residual = np.asarray(decoded.residual, dtype=np.float64)
    outer = np.asarray(geometry.outer_boundary_mask, dtype=bool)
    obstacle = np.asarray(geometry.obstacle_mask, dtype=bool)
    return {
        "divergence_abs_max": float(np.max(np.abs(decoded.divergence))),
        "residual_abs_max": float(np.max(np.abs(residual))),
        "outer_boundary_velocity_abs_max": float(np.max(np.abs(velocity[:, :, outer]))),
        "obstacle_velocity_abs_max": float(np.max(np.abs(velocity[:, :, obstacle]))),
        "obstacle_forcing_abs_max": float(np.max(np.abs(forcing[:, :, obstacle]))),
    }


def _perturbed_case_dir(round_root: Path, base_case_id: str, family: str) -> Path:
    return (
        round_root
        / "perturbed_cases"
        / family
        / f"{base_case_id}__robust_{family}"
    )


def _perturbation_case_complete(
    case_dir: Path,
    *,
    base_case_id: str,
    family: str,
    target_question_norm: float,
) -> bool:
    metadata_path = case_dir / "perturbation_metadata.json"
    if not metadata_path.is_file():
        return False
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("schema") != PERTURBATION_SCHEMA:
        raise RuntimeError(f"unexpected perturbation schema in {case_dir}")
    expected = {
        "base_case_id": base_case_id,
        "perturbation_family": family,
        "target_question_norm": float(target_question_norm),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"perturbation receipt mismatch for {key} in {case_dir}")
    require_continuous_mms_case_contract(case_dir)
    return True


def _materialization_summary(
    *,
    round_root: Path,
    base_case_dirs: list[Path],
    expect_compacted: bool = False,
) -> dict[str, Any]:
    rows = []
    missing = []
    compacted = 0
    retained_hidden_caches = 0
    for base_case_dir in base_case_dirs:
        for family in PERTURBATION_FAMILIES:
            case_dir = _perturbed_case_dir(round_root, base_case_dir.name, family)
            metadata_path = case_dir / "perturbation_metadata.json"
            if (case_dir / "cache_compaction_receipt.json").is_file():
                compacted += 1
            if (case_dir / "hidden_truth_cache.npz").is_file():
                retained_hidden_caches += 1
            if not metadata_path.is_file():
                missing.append(str(case_dir))
                continue
            rows.append(json.loads(metadata_path.read_text()))
    summaries = {
        family: _summarize_numeric_rows(
            [
                {
                    **row["calibration"]["final"],
                    "coefficient_amplitude": row["coefficient_amplitude"],
                    **row["continuous_hard_metrics"],
                }
                for row in rows
                if row["perturbation_family"] == family
            ]
        )
        for family in PERTURBATION_FAMILIES
    }
    expected = len(base_case_dirs) * len(PERTURBATION_FAMILIES)
    storage_complete = (
        not bool(expect_compacted)
        or (compacted == expected and retained_hidden_caches == 0)
    )
    return {
        "expected_case_count": int(expected),
        "materialized_case_count": len(rows),
        "missing_case_count": len(missing),
        "missing_cases": missing[:32],
        "compacted_hidden_cache_count": int(compacted),
        "retained_hidden_cache_count": int(retained_hidden_caches),
        "expect_compacted": bool(expect_compacted),
        "families": summaries,
        "complete": len(rows) == expected and not missing and storage_complete,
    }


def _evaluate_round(
    *,
    args,
    round_index: int,
    round_root: Path,
    base_case_dirs: list[Path],
    round_solver: Path,
    final_solver: Path,
    finalize: bool = True,
) -> list[dict[str, Any]]:
    round_hash = _sha256(round_solver)
    final_hash = _sha256(final_solver)
    if round_hash == final_hash:
        specs = [
            {
                "name": "round_and_final_solver",
                "roles": ["round_solver", "final_solver"],
                "path": round_solver,
                "sha256": round_hash,
            }
        ]
    else:
        specs = [
            {
                "name": "round_solver",
                "roles": ["round_solver"],
                "path": round_solver,
                "sha256": round_hash,
            },
            {
                "name": "final_solver",
                "roles": ["final_solver"],
                "path": final_solver,
                "sha256": final_hash,
            },
        ]
    return [
        _evaluate_solver_pairs(
            args=args,
            round_index=round_index,
            round_root=round_root,
            base_case_dirs=base_case_dirs,
            solver_spec=spec,
            finalize=bool(finalize),
        )
        for spec in specs
    ]


def _evaluate_solver_pairs(
    *,
    args,
    round_index: int,
    round_root: Path,
    base_case_dirs: list[Path],
    solver_spec: dict[str, Any],
    finalize: bool,
) -> dict[str, Any]:
    evaluation_root = round_root / "evaluations"
    evaluation_root.mkdir(parents=True, exist_ok=True)
    rows_path = evaluation_root / f"{solver_spec['name']}.jsonl"
    rows_by_id = _read_jsonl_by_key(rows_path, key="base_case_id")
    pending = [path for path in base_case_dirs if path.name not in rows_by_id]
    module = (
        _load_submission_module(Path(solver_spec["path"]))
        if pending
        else None
    )
    processed = 0
    with rows_path.open("a", encoding="utf-8") as handle:
        for base_case_dir in base_case_dirs:
            if base_case_dir.name in rows_by_id:
                continue
            if module is None:
                raise RuntimeError("solver module was not loaded for pending cases")
            row = {
                "base_case_id": base_case_dir.name,
                "round": int(round_index),
                "solver_name": solver_spec["name"],
                "solver_roles": list(solver_spec["roles"]),
                "solver_path": str(Path(solver_spec["path"]).resolve()),
                "solver_sha256": solver_spec["sha256"],
                "families": {},
            }
            base = _solve_case(
                module,
                base_case_dir,
                device=str(args.device),
                batch_points=int(args.source_batch_points),
            )
            row["base"] = _without_arrays(base)
            for family in PERTURBATION_FAMILIES:
                perturbed_case_dir = _perturbed_case_dir(
                    round_root,
                    base_case_dir.name,
                    family,
                )
                perturbed = _solve_case(
                    module,
                    perturbed_case_dir,
                    device=str(args.device),
                    batch_points=int(args.source_batch_points),
                )
                family_row = {
                    "family": family,
                    "perturbed_case_id": perturbed_case_dir.name,
                    "perturbed": _without_arrays(perturbed),
                    "perturbation": json.loads(
                        (perturbed_case_dir / "perturbation_metadata.json").read_text()
                    )["calibration"]["final"],
                }
                if base["status"] == "ok" and perturbed["status"] == "ok":
                    family_row.update(
                        paired_solver_metrics(
                            base_prediction=base["prediction"],
                            base_truth=base["truth"],
                            perturbed_prediction=perturbed["prediction"],
                            perturbed_truth=perturbed["truth"],
                            fluid_mask=base["fluid_mask"],
                            velocity_scale=float(args.reference_velocity_scale),
                        )
                    )
                    family_row["status"] = "ok"
                else:
                    family_row["status"] = "solver_failure"
                row["families"][family] = family_row
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            rows_by_id[base_case_dir.name] = row
            processed += 1
            if processed % 16 == 0 or processed == len(pending):
                print(
                    f"round {round_index} {solver_spec['name']}: evaluated "
                    f"{processed}/{len(pending)} pending cases; "
                    f"JSONL total={len(rows_by_id)}",
                    flush=True,
                )
    rows = [rows_by_id[path.name] for path in base_case_dirs]
    family_summaries = {}
    for family in PERTURBATION_FAMILIES:
        family_rows = [row["families"][family] for row in rows]
        family_summaries[family] = {
            "status_counts": _status_counts(family_rows),
            "metrics": _summarize_numeric_rows(family_rows),
        }
    result = {
        "schema": "ns2d_structured_solver_orthogonal_robustness_evaluation_v1",
        "round": int(round_index),
        "solver_name": solver_spec["name"],
        "solver_roles": list(solver_spec["roles"]),
        "solver_path": str(Path(solver_spec["path"]).resolve()),
        "solver_sha256": solver_spec["sha256"],
        "case_count": len(rows),
        "base_metrics": _summarize_numeric_rows(
            [row["base"].get("metrics", {}) for row in rows]
        ),
        "families": family_summaries,
        "results": rows,
        "complete": len(rows) == len(base_case_dirs),
    }
    output_path = evaluation_root / f"{solver_spec['name']}.json"
    if bool(finalize):
        _write_json_atomic(output_path, result)
    return {
        key: result[key]
        for key in (
            "solver_name",
            "solver_roles",
            "solver_path",
            "solver_sha256",
            "case_count",
            "base_metrics",
            "families",
            "complete",
        )
    } | {
        "report": str(output_path) if bool(finalize) else None,
        "finalized": bool(finalize),
    }


def _solve_case(
    module,
    case_dir: Path,
    *,
    device: str = "auto",
    batch_points: int = 8192,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        question = _load_public_question(case_dir)
        public = question["_arrays"]["public_arrays"]
        query_grid = {
            name: np.asarray(public[name], dtype=np.float64)
            for name in ("x", "y", "t")
        }
        truth, truth_source = _load_truth_velocity(
            case_dir,
            public=public,
            device=device,
            batch_points=batch_points,
        )
        output = _call_solver(module, case_dir, question, query_grid)
        prediction = _extract_velocity(output, target_shape=truth.shape)
        fluid = np.asarray(public["fluid_mask"], dtype=bool)
        metrics = _velocity_error_metrics(
            prediction,
            truth,
            fluid,
            velocity_std=None,
        )
        return {
            "status": "ok",
            "metrics": {key: float(value) for key, value in metrics.items()},
            "wall_time_seconds": float(time.perf_counter() - started),
            "truth_source": truth_source,
            "prediction": prediction,
            "truth": truth,
            "fluid_mask": fluid,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "exception",
            "exception": repr(exc),
            "wall_time_seconds": float(time.perf_counter() - started),
        }


def _load_truth_velocity(
    case_dir: Path,
    *,
    public: dict[str, Any],
    device: str,
    batch_points: int,
) -> tuple[np.ndarray, str]:
    cache_path = case_dir / "hidden_truth_cache.npz"
    if cache_path.is_file():
        with np.load(cache_path) as hidden:
            return (
                np.asarray(hidden["u_true"], dtype=np.float64),
                "hidden_truth_cache",
            )
    canonical_path = case_dir / "hidden_continuous_case.pt"
    if not canonical_path.is_file():
        raise FileNotFoundError(
            f"case has neither sampled nor canonical hidden truth: {case_dir}"
        )
    field = load_structured_continuous_case(case_dir, device=device)
    x = np.asarray(public["x"], dtype=np.float64)
    y = np.asarray(public["y"], dtype=np.float64)
    t = np.asarray(public["t"], dtype=np.float64)
    tt, xx, yy = np.meshgrid(t, x, y, indexing="ij")
    coordinates = np.stack((xx, yy, tt), axis=-1).reshape(-1, 3)
    u_parts = []
    v_parts = []
    chunk = max(1, int(batch_points))
    for start in range(0, int(coordinates.shape[0]), chunk):
        u, v, pressure = field.velocity_pressure(coordinates[start : start + chunk])
        u_parts.append(u.detach().cpu().numpy().astype(np.float64))
        v_parts.append(v.detach().cpu().numpy().astype(np.float64))
        del u, v, pressure
    shape = (len(t), len(x), len(y))
    u_values = np.concatenate(u_parts).reshape(shape)
    v_values = np.concatenate(v_parts).reshape(shape)
    del field
    _empty_cuda_cache()
    return (
        np.stack((u_values, v_values), axis=1),
        "canonical_continuous_truth_regenerated",
    )


def _without_arrays(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {"prediction", "truth", "fluid_mask"}
    }


def _validate_geometry(public: dict[str, np.ndarray], *, geometry) -> None:
    for name, expected in geometry.public_arrays().items():
        if name not in public:
            raise RuntimeError(f"source public arrays are missing {name}")
        if not np.array_equal(np.asarray(public[name]), np.asarray(expected)):
            raise RuntimeError(f"source geometry differs from frozen contract: {name}")


def _direction_seed(
    *,
    base_seed: int,
    round_index: int,
    case_index: int,
    family: str,
) -> int:
    family_index = PERTURBATION_FAMILIES.index(family)
    value = (
        int(base_seed)
        + 1_000_003 * int(round_index)
        + 10_007 * int(case_index)
        + 101 * int(family_index)
    )
    return int(value % (2**63 - 1))


def _summarize_numeric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    result = {"n": len(rows)}
    for key in keys:
        values = np.asarray(
            [
                float(row[key])
                for row in rows
                if isinstance(row.get(key), (int, float))
                and np.isfinite(float(row[key]))
            ],
            dtype=np.float64,
        )
        if not values.size:
            continue
        result[key] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "q10": float(np.quantile(values, 0.10)),
            "q90": float(np.quantile(values, 0.90)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return result


def _status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def _read_jsonl_by_key(path: Path, *, key: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return rows
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid JSONL at {path}:{line_number}") from exc
        value = str(row[key])
        if value in rows:
            raise RuntimeError(f"duplicate {key}={value!r} in {path}")
        rows[value] = row
    return rows


def _write_or_validate_json(path: Path, payload: dict[str, Any]) -> None:
    if path.is_file():
        existing = json.loads(path.read_text())
        if existing != payload:
            raise RuntimeError(f"existing run receipt does not match requested config: {path}")
        return
    _write_json_atomic(path, payload)


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        return


if __name__ == "__main__":
    main()
