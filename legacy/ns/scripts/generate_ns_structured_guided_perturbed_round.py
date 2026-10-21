#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agent_verifier.neural_generator.structured_ns_answer_first import (  # noqa: E402
    load_structured_continuous_case,
)
from agent_verifier.neural_generator.structured_ns_distribution import (  # noqa: E402
    StructuredCoefficientPCA,
)
from agent_verifier.neural_generator.structured_ns_robustness import (  # noqa: E402
    make_perturbation,
)
from agent_verifier.neural_generator.triplane_dataset import (  # noqa: E402
    build_kite_baffle_geometry,
    write_geometry_contract,
)
from agent_verifier.neural_generator.triplane_verifier import (  # noqa: E402
    CONTINUOUS_MMS_TRUTH_CONTRACT,
    require_continuous_mms_case_contract,
)
from scripts.run_ns_structured_solver_robustness_round import (  # noqa: E402
    _calibrate,
    _direction_seed,
    _empty_cuda_cache,
    _load_truth_velocity,
    _validate_geometry,
    _write_perturbation_case,
)


SCHEMA = "ns2d_structured_guided_perturbed_round_v1"
DEFAULT_FAMILY_COUNTS = {
    "pca_tangent": 48,
    "spectral_complement": 96,
    "pca_normal": 48,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one structured NS round with a fixed-budget mixture of "
            "base/guided cases and random answer-first perturbations."
        )
    )
    parser.add_argument("--flow-checkpoint", required=True)
    parser.add_argument("--pca-checkpoint", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--reference-manifest", required=True)
    parser.add_argument("--integration-steps", type=int, default=256)
    parser.add_argument("--sample-batch", type=int, default=256)
    parser.add_argument("--train-audit-cases", type=int, default=80000)
    parser.add_argument("--validation-audit-cases", type=int, default=2048)
    parser.add_argument("--metric-rank", type=int, default=64)
    parser.add_argument("--paired-base-audit-cases", type=int, default=128)
    parser.add_argument("--guidance-strength", type=float, default=0.10)
    parser.add_argument("--guidance-max-flow-ratio", type=float, default=0.25)
    parser.add_argument("--guided-fraction", type=float, default=1.0)
    parser.add_argument("--source-batch-points", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--selection-seed", type=int, default=2026081601)
    parser.add_argument("--perturbation-seed", type=int, default=2026081602)
    parser.add_argument("--unperturbed-cases", type=int, default=832)
    parser.add_argument("--perturbation-cases", type=int, default=192)
    parser.add_argument("--replay-perturbation-cases", type=int, default=96)
    parser.add_argument("--pca-tangent-cases", type=int, default=48)
    parser.add_argument("--spectral-complement-cases", type=int, default=96)
    parser.add_argument("--pca-normal-cases", type=int, default=48)
    parser.add_argument("--target-question-norm", type=float, default=0.01)
    parser.add_argument(
        "--reference-velocity-scale",
        type=float,
        default=0.005531590859331513,
    )
    parser.add_argument(
        "--reference-forcing-scale",
        type=float,
        default=0.39948812205030987,
    )
    parser.add_argument("--spectral-max-mode", type=int, default=20)
    parser.add_argument("--calibration-tolerance", type=float, default=0.02)
    parser.add_argument("--calibration-iterations", type=int, default=3)
    parser.add_argument(
        "--cuda-cache-clear-every",
        type=int,
        default=1,
        help="Clear unused CUDA blocks after this many perturbations; zero disables it.",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    round_index = _required_int_env("AVF_ROUND_INDEX")
    final_case_count = _required_int_env("AVF_CASES_PER_ROUND")
    round_dir = Path(_required_env("AVF_ROUND_DIR")).resolve()
    linked_root = Path(_required_env("AVF_ROUND_GENERATED_CASES")).resolve()
    family_counts = {
        "pca_tangent": int(args.pca_tangent_cases),
        "spectral_complement": int(args.spectral_complement_cases),
        "pca_normal": int(args.pca_normal_cases),
    }
    _validate_args(
        args,
        round_index=round_index,
        final_case_count=final_case_count,
        family_counts=family_counts,
    )
    if any(linked_root.iterdir()):
        raise SystemExit(f"final generated-case root is not empty: {linked_root}")

    unmixed_root = round_dir / "structured_generator_unmixed_cases"
    if unmixed_root.exists():
        raise SystemExit(f"refusing to reuse unmixed generator root: {unmixed_root}")
    _run_base_generator(
        args,
        round_index=round_index,
        round_dir=round_dir,
        linked_root=unmixed_root,
        case_count=int(args.unperturbed_cases),
    )
    base_receipt_path = round_dir / "structured_generator_round.json"
    base_receipt = json.loads(base_receipt_path.read_text())
    _write_json(round_dir / "base_generator_round.json", base_receipt)

    current_case_dirs = _case_dirs(unmixed_root)
    replay_case_dirs = _previous_unperturbed_cases(
        round_dir=round_dir,
        round_index=round_index,
    )
    plan = plan_round_composition(
        current_case_ids=[path.name for path in current_case_dirs],
        replay_case_ids=[path.name for path in replay_case_dirs],
        unperturbed_cases=int(args.unperturbed_cases),
        family_counts=family_counts,
        replay_perturbation_cases=(
            0 if round_index == 0 else int(args.replay_perturbation_cases)
        ),
        selection_seed=int(args.selection_seed) + 100_003 * round_index,
    )
    current_by_id = {path.name: path for path in current_case_dirs}
    replay_by_id = {path.name: path for path in replay_case_dirs}
    unperturbed_ids = set(plan["unperturbed_case_ids"])
    current_anchor_ids = {
        row["base_case_id"]
        for row in plan["perturbations"]
        if row["source_kind"] == "current"
    }

    case_rows: list[dict[str, Any]] = []
    for case_id in plan["unperturbed_case_ids"]:
        source = current_by_id[case_id]
        target = linked_root / case_id
        _link_case(source, target)
        case_rows.append(
            {
                "case_id": case_id,
                "private_case_dir": str(source.resolve()),
                "role": (
                    "current_unperturbed_anchor"
                    if case_id in current_anchor_ids
                    else "current_unperturbed_guided"
                ),
                "source_round": int(round_index),
            }
        )

    selections = []
    for index, row in enumerate(plan["perturbations"]):
        source_index = current_by_id if row["source_kind"] == "current" else replay_by_id
        selections.append(
            {
                **row,
                "base_case_dir": source_index[row["base_case_id"]],
                "target_case_id": (
                    f"structured_guided_r{round_index:03d}_pert_"
                    f"{index:06d}_{row['family']}"
                ),
            }
        )
    perturbation_rows, perturbation_summary = _materialize_perturbations(
        args=args,
        round_index=round_index,
        round_dir=round_dir,
        selections=selections,
    )
    for row in perturbation_rows:
        source = Path(row["private_case_dir"])
        _link_case(source, linked_root / row["case_id"])
        case_rows.append(row)

    linked_cases = _case_dirs(linked_root)
    if len(linked_cases) != final_case_count:
        raise RuntimeError(
            f"mixed round linked {len(linked_cases)} cases; expected {final_case_count}"
        )
    if len({path.name for path in linked_cases}) != len(linked_cases):
        raise RuntimeError("mixed round contains duplicate case IDs")
    if not unperturbed_ids.issubset({path.name for path in linked_cases}):
        raise RuntimeError("mixed round lost an unperturbed selected case")

    role_counts = Counter(str(row["role"]) for row in case_rows)
    composition = {
        "fixed_total_budget": True,
        "case_count": int(final_case_count),
        "base_generated_candidate_count": len(current_case_dirs),
        "unused_base_generated_candidate_count": (
            len(current_case_dirs) - int(args.unperturbed_cases)
        ),
        "unperturbed_case_count": int(args.unperturbed_cases),
        "perturbation_case_count": int(args.perturbation_cases),
        "current_unperturbed_anchor_count": int(
            role_counts["current_unperturbed_anchor"]
        ),
        "current_unperturbed_non_anchor_count": int(
            role_counts["current_unperturbed_guided"]
        ),
        "perturbation_source_counts": dict(
            Counter(str(row["source_kind"]) for row in plan["perturbations"])
        ),
        "perturbation_family_counts": dict(
            Counter(str(row["family"]) for row in plan["perturbations"])
        ),
        "role_counts": dict(role_counts),
        "target_question_norm": float(args.target_question_norm),
    }
    receipt = {
        "schema": "precomputed_continuous_contract_receipt_v1",
        "generator_schema": SCHEMA,
        "truth_contract": CONTINUOUS_MMS_TRUTH_CONTRACT,
        "contract_passed": True,
        "linked_cases": int(final_case_count),
        "round": int(round_index),
        "sampling_policy": base_receipt.get("sampling_policy"),
        "base_generator": {
            "receipt": str((round_dir / "base_generator_round.json").resolve()),
            "receipt_sha256": _sha256(round_dir / "base_generator_round.json"),
            "candidate_count": len(current_case_dirs),
            "guidance_checkpoint": base_receipt.get("guidance_checkpoint"),
            "guidance_checkpoint_sha256": base_receipt.get(
                "guidance_checkpoint_sha256"
            ),
        },
        "composition": composition,
        "selection_seed": int(args.selection_seed) + 100_003 * round_index,
        "perturbation_seed": int(args.perturbation_seed),
        "perturbation_materialization": perturbation_summary,
        "guidance_policy": {
            "base_generator_guidance_uses_perturbation_cases": False,
            "reason": "avoid_double_guidance_and_keep_rf_pca_definition_frozen",
        },
        "cases": case_rows,
    }
    _write_json(round_dir / "precomputed_continuous_contract.json", receipt)
    _write_json(round_dir / "structured_generator_round.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


def plan_round_composition(
    *,
    current_case_ids: list[str],
    replay_case_ids: list[str],
    unperturbed_cases: int,
    family_counts: dict[str, int],
    replay_perturbation_cases: int,
    selection_seed: int,
) -> dict[str, Any]:
    perturbation_cases = sum(int(value) for value in family_counts.values())
    if perturbation_cases <= 0:
        raise ValueError("perturbation family counts must be positive")
    if len(current_case_ids) < int(unperturbed_cases):
        raise ValueError("current candidate pool is smaller than the unperturbed quota")
    if len(set(current_case_ids)) != len(current_case_ids):
        raise ValueError("current candidate IDs are not unique")
    if len(set(replay_case_ids)) != len(replay_case_ids):
        raise ValueError("replay candidate IDs are not unique")
    replay_count = int(replay_perturbation_cases)
    if replay_count < 0 or replay_count > perturbation_cases:
        raise ValueError("replay perturbation quota is outside the total quota")
    if replay_count > len(replay_case_ids):
        raise ValueError("replay candidate pool is smaller than its perturbation quota")
    current_perturbation_count = perturbation_cases - replay_count
    if current_perturbation_count > int(unperturbed_cases):
        raise ValueError("current perturbation anchors exceed the unperturbed quota")

    rng = np.random.default_rng(int(selection_seed))
    current_order = rng.permutation(len(current_case_ids))
    unperturbed = [
        current_case_ids[int(index)]
        for index in current_order[: int(unperturbed_cases)]
    ]
    current_anchors = unperturbed[:current_perturbation_count]
    replay_order = rng.permutation(len(replay_case_ids))
    replay_anchors = [
        replay_case_ids[int(index)] for index in replay_order[:replay_count]
    ]

    replay_family_counts = {}
    for family, count in family_counts.items():
        numerator = int(count) * replay_count
        if numerator % perturbation_cases != 0:
            raise ValueError(
                "replay quota must split exactly across perturbation families"
            )
        replay_family_counts[family] = numerator // perturbation_cases
    current_family_counts = {
        family: int(family_counts[family]) - int(replay_family_counts[family])
        for family in family_counts
    }

    def shuffled_labels(counts: dict[str, int]) -> list[str]:
        labels = [
            family
            for family, count in counts.items()
            for _ in range(int(count))
        ]
        rng.shuffle(labels)
        return labels

    current_labels = shuffled_labels(current_family_counts)
    replay_labels = shuffled_labels(replay_family_counts)
    perturbations = [
        {
            "base_case_id": case_id,
            "family": family,
            "source_kind": "current",
        }
        for case_id, family in zip(current_anchors, current_labels, strict=True)
    ]
    perturbations.extend(
        {
            "base_case_id": case_id,
            "family": family,
            "source_kind": "replay",
        }
        for case_id, family in zip(replay_anchors, replay_labels, strict=True)
    )
    rng.shuffle(perturbations)
    return {
        "unperturbed_case_ids": unperturbed,
        "current_anchor_case_ids": current_anchors,
        "replay_anchor_case_ids": replay_anchors,
        "perturbations": perturbations,
        "current_family_counts": current_family_counts,
        "replay_family_counts": replay_family_counts,
    }


def _materialize_perturbations(
    *,
    args: argparse.Namespace,
    round_index: int,
    round_dir: Path,
    selections: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    materialization_root = round_dir / "structured_perturbation_materialization"
    if materialization_root.exists():
        raise SystemExit(
            f"refusing to reuse perturbation materialization: {materialization_root}"
        )
    materialization_root.mkdir(parents=True)
    pca = StructuredCoefficientPCA.load(args.pca_checkpoint, map_location="cpu")
    geometry = build_kite_baffle_geometry()
    write_geometry_contract(materialization_root / "geometry_contract.json", geometry)
    rows: list[dict[str, Any]] = []
    final_question_norms: dict[str, list[float]] = {
        family: [] for family in DEFAULT_FAMILY_COUNTS
    }
    for case_index, selection in enumerate(selections):
        base_case_dir = Path(selection["base_case_dir"])
        family = str(selection["family"])
        with np.load(base_case_dir / "public_arrays.npz") as arrays:
            base_public = {
                name: np.asarray(arrays[name]) for name in arrays.files
            }
        _validate_geometry(base_public, geometry=geometry)
        base_velocity, truth_source = _load_truth_velocity(
            base_case_dir,
            public=base_public,
            device=str(args.device),
            batch_points=int(args.source_batch_points),
        )
        base_forcing = np.asarray(base_public["forcing"], dtype=np.float64)
        times = np.asarray(base_public["t"], dtype=np.float64)
        base_field = load_structured_continuous_case(
            base_case_dir,
            device=str(args.device),
        )
        direction_seed = _direction_seed(
            base_seed=int(args.perturbation_seed),
            round_index=int(round_index),
            case_index=int(case_index),
            family=family,
        )
        perturbation = make_perturbation(
            base_field,
            family=family,
            seed=direction_seed,
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
        target_case_dir = (
            materialization_root
            / "perturbed_cases"
            / family
            / str(selection["target_case_id"])
        )
        _write_perturbation_case(
            target_case_dir=target_case_dir,
            base_case_dir=base_case_dir,
            family=family,
            seed=direction_seed,
            amplitude=amplitude,
            perturbation=perturbation,
            field=field,
            decoded=decoded,
            calibration=calibration,
            geometry=geometry,
            target_question_norm=float(args.target_question_norm),
        )
        require_continuous_mms_case_contract(target_case_dir)
        final_question_norms[family].append(
            float(calibration["final"]["question_delta_norm"])
        )
        rows.append(
            {
                "case_id": target_case_dir.name,
                "private_case_dir": str(target_case_dir.resolve()),
                "role": "structured_random_perturbation",
                "source_kind": str(selection["source_kind"]),
                "base_case_id": base_case_dir.name,
                "base_case_dir": str(base_case_dir.resolve()),
                "base_truth_source": truth_source,
                "perturbation_family": family,
                "perturbation_seed": int(direction_seed),
                "source_round": int(round_index),
            }
        )
        del base_field, field, decoded, perturbation
        completed = case_index + 1
        clear_every = int(args.cuda_cache_clear_every)
        if clear_every > 0 and (
            completed % clear_every == 0 or completed == len(selections)
        ):
            _empty_cuda_cache()
        if completed % 16 == 0 or completed == len(selections):
            print(
                f"round {round_index}: materialized {completed}/{len(selections)} "
                "mixed-training perturbations",
                flush=True,
            )
    family_summary = {}
    for family, values in final_question_norms.items():
        array = np.asarray(values, dtype=np.float64)
        family_summary[family] = {
            "n": int(array.size),
            "question_delta_norm_mean": float(np.mean(array)),
            "question_delta_norm_min": float(np.min(array)),
            "question_delta_norm_max": float(np.max(array)),
        }
    summary = {
        "case_count": len(rows),
        "family_counts": dict(
            Counter(str(row["perturbation_family"]) for row in rows)
        ),
        "source_counts": dict(Counter(str(row["source_kind"]) for row in rows)),
        "families": family_summary,
        "complete": len(rows) == len(selections),
    }
    _write_json(materialization_root / "materialization_summary.json", summary)
    return rows, summary


def _run_base_generator(
    args: argparse.Namespace,
    *,
    round_index: int,
    round_dir: Path,
    linked_root: Path,
    case_count: int,
) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "generate_ns_structured_guided_round.py"),
        "--flow-checkpoint",
        str(Path(args.flow_checkpoint).resolve()),
        "--pca-checkpoint",
        str(Path(args.pca_checkpoint).resolve()),
        "--train-manifest",
        str(Path(args.train_manifest).resolve()),
        "--validation-manifest",
        str(Path(args.validation_manifest).resolve()),
        "--reference-manifest",
        str(Path(args.reference_manifest).resolve()),
        "--integration-steps",
        str(args.integration_steps),
        "--sample-batch",
        str(args.sample_batch),
        "--train-audit-cases",
        str(args.train_audit_cases),
        "--validation-audit-cases",
        str(args.validation_audit_cases),
        "--metric-rank",
        str(args.metric_rank),
        "--paired-base-audit-cases",
        str(args.paired_base_audit_cases),
        "--guidance-strength",
        str(args.guidance_strength),
        "--guidance-max-flow-ratio",
        str(args.guidance_max_flow_ratio),
        "--guided-fraction",
        str(args.guided_fraction),
        "--source-batch-points",
        str(args.source_batch_points),
        "--seed",
        str(args.seed),
        "--device",
        str(args.device),
    ]
    env = os.environ.copy()
    env.update(
        {
            "AVF_EXPLICIT_CASE_MIX": "1",
            "AVF_ROUND_INDEX": str(round_index),
            "AVF_ROUND_DIR": str(round_dir),
            "AVF_ROUND_GENERATED_CASES": str(linked_root),
            "AVF_CASES_PER_ROUND": str(case_count),
        }
    )
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def _previous_unperturbed_cases(*, round_dir: Path, round_index: int) -> list[Path]:
    if int(round_index) == 0:
        return []
    prior_receipt_path = (
        round_dir.parent
        / f"round_{round_index - 1:03d}"
        / "structured_generator_round.json"
    )
    if not prior_receipt_path.is_file():
        raise SystemExit(f"missing prior mixed-round receipt: {prior_receipt_path}")
    payload = json.loads(prior_receipt_path.read_text())
    if payload.get("generator_schema") != SCHEMA:
        raise SystemExit(f"unexpected prior mixed-round schema: {prior_receipt_path}")
    paths = [
        Path(row["private_case_dir"])
        for row in payload.get("cases", [])
        if str(row.get("role", "")).startswith("current_unperturbed_")
    ]
    if len(paths) != int(payload["composition"]["unperturbed_case_count"]):
        raise RuntimeError("prior receipt does not enumerate all unperturbed cases")
    for path in paths:
        if not path.is_dir():
            raise FileNotFoundError(path)
    return paths


def _validate_args(
    args: argparse.Namespace,
    *,
    round_index: int,
    final_case_count: int,
    family_counts: dict[str, int],
) -> None:
    if round_index < 0:
        raise SystemExit("round index cannot be negative")
    if min(int(value) for value in family_counts.values()) <= 0:
        raise SystemExit("all perturbation family quotas must be positive")
    if sum(family_counts.values()) != int(args.perturbation_cases):
        raise SystemExit("perturbation family quotas do not sum to the total")
    if int(args.unperturbed_cases) + int(args.perturbation_cases) != final_case_count:
        raise SystemExit("unperturbed and perturbation quotas must equal the round budget")
    if not 0 <= int(args.replay_perturbation_cases) <= int(args.perturbation_cases):
        raise SystemExit("invalid replay perturbation quota")
    if not 0.0 < float(args.target_question_norm) < 1.0:
        raise SystemExit("target question norm must lie in (0,1)")
    if int(args.cuda_cache_clear_every) < 0:
        raise SystemExit("CUDA cache clear interval cannot be negative")
    if not 0.0 <= float(args.guided_fraction) <= 1.0:
        raise SystemExit("guided fraction must lie in [0,1]")
    if min(
        float(args.reference_velocity_scale),
        float(args.reference_forcing_scale),
        float(args.calibration_tolerance),
    ) <= 0.0:
        raise SystemExit("reference scales and calibration tolerance must be positive")
    for label in (
        "flow_checkpoint",
        "pca_checkpoint",
        "train_manifest",
        "validation_manifest",
        "reference_manifest",
    ):
        if not Path(getattr(args, label)).is_file():
            raise SystemExit(f"missing {label}: {getattr(args, label)}")


def _case_dirs(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "hidden_continuous_case.pt").is_file()
    )


def _link_case(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        raise RuntimeError(f"refusing to replace mixed-round link: {target}")
    target.symlink_to(source.resolve(), target_is_directory=True)


def _required_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise SystemExit(f"{name} is required")
    return value


def _required_int_env(name: str) -> int:
    value = int(_required_env(name))
    if value < 0:
        raise SystemExit(f"{name} cannot be negative")
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main(sys.argv[1:])
