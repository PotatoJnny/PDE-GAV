#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent_verifier.neural_generator.triplane_verifier import (
    CONTINUOUS_MMS_TRUTH_CONTRACT,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one base or solver-guided continuous structured NS round."
    )
    parser.add_argument("--flow-checkpoint", required=True)
    parser.add_argument("--pca-checkpoint", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--reference-manifest", required=True)
    parser.add_argument("--integration-steps", type=int, default=256)
    parser.add_argument("--sample-batch", type=int, default=256)
    parser.add_argument("--train-audit-cases", type=int, default=4096)
    parser.add_argument("--validation-audit-cases", type=int, default=1024)
    parser.add_argument("--metric-rank", type=int, default=64)
    parser.add_argument("--paired-base-audit-cases", type=int, default=128)
    parser.add_argument("--guidance-strength", type=float, default=0.10)
    parser.add_argument("--guidance-max-flow-ratio", type=float, default=0.25)
    parser.add_argument(
        "--guided-fraction",
        type=float,
        default=1.0,
        help="Fraction of generated cases that use previous-round guidance.",
    )
    parser.add_argument("--source-batch-points", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    round_index = _required_int_env("AVF_ROUND_INDEX")
    case_count = _required_int_env("AVF_CASES_PER_ROUND")
    round_dir = Path(_required_env("AVF_ROUND_DIR")).resolve()
    linked_root = Path(_required_env("AVF_ROUND_GENERATED_CASES")).resolve()
    if os.environ.get("AVF_EXPLICIT_CASE_MIX") == "1":
        _run_explicit_case_mix(
            args,
            round_index=round_index,
            case_count=case_count,
            round_dir=round_dir,
            linked_root=linked_root,
        )
        return
    loop_root = round_dir.parents[1]
    sample_root = round_dir / "structured_generator_sample"
    materialization_root = round_dir / "structured_generator_materialization"
    if sample_root.exists() or materialization_root.exists():
        raise SystemExit("refusing to overwrite an existing structured round")

    guidance_path = None
    if round_index > 0:
        guidance_path = (
            loop_root
            / "rounds"
            / f"round_{round_index - 1:03d}"
            / "guidance_update"
            / "reward_guidance.pt"
        )
        if not guidance_path.is_file():
            raise SystemExit(f"missing previous-round reward guidance: {guidance_path}")

    seed = int(args.seed) + 100_003 * int(round_index)
    sample_command = [
        sys.executable,
        str(ROOT / "scripts" / "sample_ns_structured_coefficient_flow.py"),
        "--flow-checkpoint",
        str(Path(args.flow_checkpoint).resolve()),
        "--pca-checkpoint",
        str(Path(args.pca_checkpoint).resolve()),
        "--train-manifest",
        str(Path(args.train_manifest).resolve()),
        "--validation-manifest",
        str(Path(args.validation_manifest).resolve()),
        "--out",
        str(sample_root),
        "--samples",
        str(case_count),
        "--sample-batch",
        str(args.sample_batch),
        "--integration-steps",
        str(args.integration_steps),
        "--train-audit-cases",
        str(args.train_audit_cases),
        "--validation-audit-cases",
        str(args.validation_audit_cases),
        "--metric-rank",
        str(args.metric_rank),
        "--paired-base-audit-cases",
        str(args.paired_base_audit_cases),
        "--seed",
        str(seed),
        "--state",
        "raw",
        "--device",
        str(args.device),
    ]
    if guidance_path is not None:
        sample_command.extend(
            [
                "--guidance-checkpoint",
                str(guidance_path),
                "--guidance-strength",
                str(args.guidance_strength),
                "--guidance-max-flow-ratio",
                str(args.guidance_max_flow_ratio),
            ]
        )
    _run(sample_command)

    materialize_command = [
        sys.executable,
        str(ROOT / "scripts" / "materialize_ns_structured_flow_samples.py"),
        "--sample-audit",
        str(sample_root / "sample_audit.json"),
        "--reference-manifest",
        str(Path(args.reference_manifest).resolve()),
        "--out",
        str(materialization_root),
        "--generated-cases",
        str(case_count),
        "--reference-cases",
        "0",
        "--case-id-prefix",
        f"structured_guided_r{round_index:03d}",
        "--selection-seed",
        str(seed + 1),
        "--source-batch-points",
        str(args.source_batch_points),
        "--device",
        str(args.device),
    ]
    _run(materialize_command)

    sources = sorted((materialization_root / "generated_cases").iterdir())
    if len(sources) != case_count:
        raise SystemExit(
            f"materializer produced {len(sources)} cases; expected {case_count}"
        )
    linked_root.mkdir(parents=True, exist_ok=True)
    for source in sources:
        target = linked_root / source.name
        if target.exists() or target.is_symlink():
            raise SystemExit(f"refusing to replace generated case link: {target}")
        target.symlink_to(source, target_is_directory=True)

    sample_audit = sample_root / "sample_audit.json"
    materialization_audit = materialization_root / "answer_first_report.json"
    receipt = {
        "schema": "precomputed_continuous_contract_receipt_v1",
        "truth_contract": CONTINUOUS_MMS_TRUTH_CONTRACT,
        "contract_passed": True,
        "linked_cases": case_count,
        "round": round_index,
        "sampling_policy": (
            "base_rectified_flow" if guidance_path is None else "solver_reward_guided_rectified_flow"
        ),
        "sample_audit": str(sample_audit),
        "sample_audit_sha256": _sha256(sample_audit),
        "materialization_audit": str(materialization_audit),
        "materialization_audit_sha256": _sha256(materialization_audit),
        "guidance_checkpoint": str(guidance_path) if guidance_path else None,
        "guidance_checkpoint_sha256": _sha256(guidance_path) if guidance_path else None,
    }
    (round_dir / "precomputed_continuous_contract.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    (round_dir / "structured_generator_round.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


def _run_explicit_case_mix(
    args: argparse.Namespace,
    *,
    round_index: int,
    case_count: int,
    round_dir: Path,
    linked_root: Path,
) -> None:
    """Generate an exact base/guided mixture for one public or hidden cohort."""

    fraction = float(args.guided_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise SystemExit("--guided-fraction must lie in [0, 1]")
    if case_count <= 0:
        raise SystemExit("explicit case mix needs at least one generated case")
    guided_count = (
        min(case_count, max(0, int(round(case_count * fraction))))
        if round_index > 0
        else 0
    )
    base_count = case_count - guided_count
    loop_root = Path(
        os.environ.get("AVF_LOOP_ROOT", str(round_dir.parents[1]))
    ).resolve()
    guidance_path = None
    if guided_count:
        guidance_path = (
            loop_root
            / "rounds"
            / f"round_{round_index - 1:03d}"
            / "guidance_update"
            / "reward_guidance.pt"
        )
        if not guidance_path.is_file():
            raise SystemExit(f"missing previous-round reward guidance: {guidance_path}")

    linked_root.mkdir(parents=True, exist_ok=True)
    if any(linked_root.iterdir()):
        raise SystemExit(f"generated case root is not empty: {linked_root}")
    case_prefix = os.environ.get("AVF_CASE_ID_PREFIX", "structured")
    seed = int(args.seed) + 100_003 * int(round_index)
    components = []
    for name, count, component_guidance, seed_offset in (
        ("base", base_count, None, 0),
        ("guided", guided_count, guidance_path, 10_000_019),
    ):
        if count <= 0:
            continue
        sample_root = round_dir / f"structured_generator_sample_{name}"
        materialization_root = (
            round_dir / f"structured_generator_materialization_{name}"
        )
        if sample_root.exists() or materialization_root.exists():
            raise SystemExit(
                f"refusing to overwrite existing {name} Generator output"
            )
        component_seed = seed + seed_offset
        sample_command = [
            sys.executable,
            str(ROOT / "scripts" / "sample_ns_structured_coefficient_flow.py"),
            "--flow-checkpoint", str(Path(args.flow_checkpoint).resolve()),
            "--pca-checkpoint", str(Path(args.pca_checkpoint).resolve()),
            "--train-manifest", str(Path(args.train_manifest).resolve()),
            "--validation-manifest", str(Path(args.validation_manifest).resolve()),
            "--out", str(sample_root),
            "--samples", str(count),
            "--sample-batch", str(args.sample_batch),
            "--integration-steps", str(args.integration_steps),
            "--train-audit-cases", str(args.train_audit_cases),
            "--validation-audit-cases", str(args.validation_audit_cases),
            "--metric-rank", str(args.metric_rank),
            "--paired-base-audit-cases", str(
                min(int(args.paired_base_audit_cases), count)
                if component_guidance is not None
                else 0
            ),
            "--seed", str(component_seed),
            "--state", "raw",
            "--device", str(args.device),
        ]
        if component_guidance is not None:
            sample_command.extend(
                (
                    "--guidance-checkpoint", str(component_guidance),
                    "--guidance-strength", str(args.guidance_strength),
                    "--guidance-max-flow-ratio", str(args.guidance_max_flow_ratio),
                )
            )
        _run(sample_command)

        materialize_command = [
            sys.executable,
            str(ROOT / "scripts" / "materialize_ns_structured_flow_samples.py"),
            "--sample-audit", str(sample_root / "sample_audit.json"),
            "--reference-manifest", str(Path(args.reference_manifest).resolve()),
            "--out", str(materialization_root),
            "--generated-cases", str(count),
            "--reference-cases", "0",
            "--case-id-prefix", f"{case_prefix}_{name}_r{round_index:03d}",
            "--selection-seed", str(component_seed + 1),
            "--source-batch-points", str(args.source_batch_points),
            "--device", str(args.device),
        ]
        _run(materialize_command)
        sources = sorted((materialization_root / "generated_cases").iterdir())
        if len(sources) != count:
            raise SystemExit(
                f"{name} materializer produced {len(sources)} cases; expected {count}"
            )
        for source in sources:
            target = linked_root / source.name
            if target.exists() or target.is_symlink():
                raise SystemExit(f"duplicate generated case ID: {target.name}")
            target.symlink_to(source, target_is_directory=True)
        sample_audit = sample_root / "sample_audit.json"
        materialization_audit = materialization_root / "answer_first_report.json"
        components.append(
            {
                "kind": name,
                "cases": count,
                "sample_audit": str(sample_audit),
                "sample_audit_sha256": _sha256(sample_audit),
                "materialization_audit": str(materialization_audit),
                "materialization_audit_sha256": _sha256(materialization_audit),
            }
        )

    receipt = {
        "schema": "precomputed_continuous_contract_receipt_v1",
        "truth_contract": CONTINUOUS_MMS_TRUTH_CONTRACT,
        "contract_passed": True,
        "linked_cases": case_count,
        "round": round_index,
        "sampling_policy": "explicit_base_guided_mixture",
        "case_mix": {"base": base_count, "guided": guided_count},
        "guided_fraction_requested": fraction,
        "components": components,
        "guidance_checkpoint": str(guidance_path) if guidance_path else None,
        "guidance_checkpoint_sha256": (
            _sha256(guidance_path) if guidance_path else None
        ),
    }
    for name in ("precomputed_continuous_contract.json", "structured_generator_round.json"):
        (round_dir / name).write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main(sys.argv[1:])
