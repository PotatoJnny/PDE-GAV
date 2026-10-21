#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


PUBLIC_CASE_FILES = ("public_question.json", "public_arrays.npz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample public VPFP questions from this session's frozen case reservoir."
    )
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def _regular_file(path: Path) -> Path:
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"expected a regular public file: {path}")
    return path


def main() -> None:
    args = parse_args()
    if int(args.count) <= 0:
        raise SystemExit("--count must be positive")

    workspace = Path(__file__).resolve().parent
    default_manifest = (
        "generator_reservoir_manifest.json"
        if (workspace / "generator_reservoir_manifest.json").is_file()
        else "case_manifest.json"
    )
    manifest_path = (workspace / (args.manifest or default_manifest)).resolve()
    if manifest_path.parent != workspace:
        raise SystemExit("--manifest must name a file in the current workspace")
    manifest = json.loads(_regular_file(manifest_path).read_text(encoding="utf-8"))
    rows = list(manifest.get("cases") or [])
    if int(args.count) > len(rows):
        raise SystemExit(
            f"requested {int(args.count)} cases from a reservoir of {len(rows)}"
        )

    rng = random.Random(int(args.seed))
    selected = rng.sample(rows, int(args.count))
    default_out = Path("generator_samples") / f"seed_{int(args.seed):010d}"
    output = (workspace / (args.out or default_out)).resolve()
    if output == workspace or workspace not in output.parents:
        raise SystemExit("--out must stay inside the current workspace")
    if output.exists():
        raise SystemExit(f"sample output already exists: {output}")

    cases_out = output / "cases"
    cases_out.mkdir(parents=True)
    sampled_rows = []
    for index, row in enumerate(selected):
        relative = Path(str(row.get("case_dir") or ""))
        source = (workspace / relative).resolve()
        if workspace not in source.parents or source.parent.name != "cases":
            raise SystemExit(f"invalid public case path in manifest: {relative}")
        destination_id = f"sample_{index:06d}"
        destination = cases_out / destination_id
        destination.mkdir()
        for name in PUBLIC_CASE_FILES:
            shutil.copy2(_regular_file(source / name), destination / name)
        question = json.loads(
            (destination / "public_question.json").read_text(encoding="utf-8")
        )
        question["case_id"] = destination_id
        (destination / "public_question.json").write_text(
            json.dumps(question, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        sampled_rows.append(
            {
                "case_id": destination_id,
                "case_dir": f"cases/{destination_id}",
            }
        )

    sample_manifest = {
        "schema": "vpfp_public_generator_sample_v1",
        "sampling_contract": "frozen_public_reservoir_from_current_controller_session",
        "seed": int(args.seed),
        "case_count": len(sampled_rows),
        "cases": sampled_rows,
    }
    (output / "sample_manifest.json").write_text(
        json.dumps(sample_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), **sample_manifest}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
