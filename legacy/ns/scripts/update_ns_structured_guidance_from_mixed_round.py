#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FILTER_SCHEMA = "ns2d_structured_guidance_perturbation_filter_v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove answer-first perturbation rows from the cumulative reward "
            "table, then fit the next structured NS generator guidance model."
        )
    )
    parser.add_argument("--flow-checkpoint", required=True)
    parser.add_argument("--pca-checkpoint", required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--time-width", type=int, default=128)
    parser.add_argument("--minimum-cases", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    round_index = int(_required_env("AVF_ROUND_INDEX"))
    round_dir = Path(_required_env("AVF_ROUND_DIR")).resolve()
    reward_table = Path(_required_env("AVF_REWARD_TABLE")).resolve()
    solver = Path(_required_env("AVF_SOLVER_SUBMISSION")).resolve()
    loop_root = round_dir.parents[1]
    filter_root = round_dir / "guidance_filter"
    filtered_reward_table, report = filter_reward_table(
        reward_table=reward_table,
        loop_root=loop_root,
        output_dir=filter_root,
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)

    output = round_dir / "guidance_update"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "train_ns_structured_reward_guidance.py"),
        "--reward-table",
        str(filtered_reward_table),
        "--loop-root",
        str(loop_root),
        "--pca-checkpoint",
        str(Path(args.pca_checkpoint).resolve()),
        "--flow-checkpoint",
        str(Path(args.flow_checkpoint).resolve()),
        "--solver-submission",
        str(solver),
        "--out",
        str(output),
        "--steps",
        str(args.steps),
        "--batch-size",
        str(args.batch_size),
        "--width",
        str(args.width),
        "--depth",
        str(args.depth),
        "--time-width",
        str(args.time_width),
        "--minimum-cases",
        str(args.minimum_cases),
        "--lr",
        str(args.lr),
        "--seed",
        str(int(args.seed) + 100_003 * round_index),
        "--device",
        str(args.device),
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def filter_reward_table(
    *,
    reward_table: Path,
    loop_root: Path,
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    reward_table = reward_table.resolve()
    loop_root = loop_root.resolve()
    output_dir = output_dir.resolve()
    if not reward_table.is_file():
        raise FileNotFoundError(reward_table)
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse guidance filter output: {output_dir}")

    case_index = _private_case_index(loop_root)
    rows = _read_jsonl(reward_table)
    retained: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    excluded_families: Counter[str] = Counter()
    original_excluded = 0
    missing: list[str] = []
    for row in rows:
        case_id = str(row.get("case_id") or "")
        case_dir = case_index.get(case_id)
        if case_dir is None:
            missing.append(case_id or "<empty>")
            continue
        if not (case_dir / "hidden_continuous_case.pt").is_file():
            original_excluded += 1
            excluded.append(row)
            continue
        metadata_path = case_dir / "perturbation_metadata.json"
        if not metadata_path.is_file():
            retained.append(row)
            continue
        metadata = json.loads(metadata_path.read_text())
        family = str(metadata.get("perturbation_family") or "unknown")
        excluded_families[family] += 1
        excluded.append(row)
    if missing:
        preview = ", ".join(missing[:8])
        raise ValueError(
            f"could not map {len(missing)} reward rows to private cases: {preview}"
        )
    if not retained:
        raise ValueError("guidance filter removed every reward row")

    output_dir.mkdir(parents=True)
    filtered_path = output_dir / "filtered_reward_table.jsonl"
    _write_jsonl(filtered_path, retained)
    report = {
        "schema": FILTER_SCHEMA,
        "source_reward_table": str(reward_table),
        "filtered_reward_table": str(filtered_path),
        "perturbations_excluded_from_generator_guidance": True,
        "reason": "avoid_double_guidance_and_keep_rf_pca_definition_frozen",
        "input_rows": len(rows),
        "original_dataset_rows_excluded": int(original_excluded),
        "retained_rows": len(retained),
        "excluded_rows": len(excluded),
        "excluded_by_family": dict(sorted(excluded_families.items())),
        "input_status_counts": _status_counts(rows),
        "retained_status_counts": _status_counts(retained),
        "excluded_status_counts": _status_counts(excluded),
    }
    (output_dir / "reward_filter_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return filtered_path, report


def _private_case_index(loop_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    manifests = sorted(
        loop_root.glob("rounds/round_*/private_eval/generated_private_manifest.json")
    )
    if not manifests:
        raise ValueError(f"no generated private manifests under {loop_root}")
    for manifest_path in manifests:
        payload = json.loads(manifest_path.read_text())
        for row in payload.get("cases", []):
            case_dir = Path(str(row.get("private_case_dir") or ""))
            if not case_dir.is_dir():
                raise FileNotFoundError(
                    f"manifest references missing private case: {case_dir}"
                )
            case_id = case_dir.name
            prior = index.get(case_id)
            if prior is not None and prior.resolve() != case_dir.resolve():
                raise ValueError(f"ambiguous private case id: {case_id}")
            index[case_id] = case_dir
    return index


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
        rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(
        sorted(Counter(str(row.get("status", "unknown")) for row in rows).items())
    )


def _required_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise SystemExit(f"{name} is required")
    return value


if __name__ == "__main__":
    main(sys.argv[1:])
