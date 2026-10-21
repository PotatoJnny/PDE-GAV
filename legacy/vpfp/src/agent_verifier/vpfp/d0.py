from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from agent_verifier.vpfp.basis import basis_projection_metrics
from agent_verifier.vpfp.io import load_case, write_case
from agent_verifier.vpfp.numerical import VPFPNumericalConfig, make_numerical_d0_case
from agent_verifier.vpfp.operators import VPFPGrid, VPFPTime
from agent_verifier.vpfp.stats import summarize_case_dirs
from agent_verifier.vpfp.verifier import verify_case_dir


def build_numerical_d0(
    *,
    out_dir: str | Path,
    case_count: int,
    seed: int,
    grid: VPFPGrid,
    time: VPFPTime,
    numerical: VPFPNumericalConfig,
    start_index: int = 0,
    split_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    out = Path(out_dir)
    cases_root = out / "cases"
    cases_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(seed))
    rows = []
    for local_idx in range(int(case_count)):
        case_number = int(start_index) + int(local_idx)
        sample_seed = int(rng.integers(0, 2**31 - 1))
        case_id = f"vpfp_d0_{case_number:07d}"
        case = make_numerical_d0_case(
            case_id=case_id,
            grid=grid,
            time=time,
            config=numerical,
            rng=np.random.default_rng(sample_seed),
            generator_metadata={"sample_seed": sample_seed, "case_number": case_number},
        )
        case_dir = cases_root / case_id
        write_case(case_dir, case)
        rows.append({
            "case_id": case_id,
            "case_dir": str(case_dir),
            "sample_seed": sample_seed,
            "status": "accepted",
            "source_origin": "randomized_public_equation",
            "repeat_initial_difficulty": case.diagnostics.get("repeat_initial_difficulty"),
            "basis_f_projection_rel_l2": case.diagnostics.get("basis_f_projection_rel_l2"),
        })
    manifest = {
        "backend": "deterministic_coarse_vpfp_grid_solver",
        "case_schema": "vpfp_factorized_public_hidden_v1",
        "case_count": int(case_count),
        "seed": int(seed),
        "start_index": int(start_index),
        "grid": grid.to_dict(),
        "time": time.to_dict(),
        "numerical_config": numerical.to_dict(),
        "cases": rows,
    }
    if split_counts and sum(int(v) for v in split_counts.values()) > 0:
        manifest["splits"] = make_splits(rows, split_counts)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if "splits" in manifest:
        (out / "split_manifest.json").write_text(json.dumps(_split_manifest(manifest), indent=2, sort_keys=True) + "\n")
    return manifest


def make_splits(rows: list[dict[str, Any]], split_counts: dict[str, int]) -> dict[str, list[dict[str, Any]]]:
    requested = {name: int(split_counts.get(name, 0)) for name in ("train", "validation", "d0_distribution_test", "adversarial_guard")}
    if sum(requested.values()) <= 0:
        requested["train"] = len(rows)
    if sum(requested.values()) > len(rows):
        raise ValueError(f"split counts {requested} exceed available cases {len(rows)}")
    splits: dict[str, list[dict[str, Any]]] = {}
    cursor = 0
    for name, count in requested.items():
        splits[name] = rows[cursor : cursor + count]
        cursor += count
    if cursor < len(rows):
        splits.setdefault("train", []).extend(rows[cursor:])
    return splits


def audit_numerical_d0(d0_root: str | Path) -> dict[str, Any]:
    root = Path(d0_root)
    case_dirs = iter_case_dirs(root / "cases")
    rows = []
    for case_dir in case_dirs:
        verification = verify_case_dir(case_dir)
        case = load_case(case_dir)
        hidden = case["hidden_arrays"]
        basis_metrics = basis_projection_metrics(hidden["rho_true"], hidden["g_true"], case["grid"], case["public_question"].get("basis", {}).get("config"))
        rows.append({
            "case_id": case["case_id"],
            "accepted": bool(verification["accepted"]),
            "failures": verification["failures"],
            "source_origin": case["public_question"].get("source_origin"),
            **verification["metrics"],
            **basis_metrics,
        })
    accepted = [row for row in rows if row["accepted"]]
    split_sizes = {}
    split_path = root / "split_manifest.json"
    if split_path.exists():
        split_manifest = json.loads(split_path.read_text())
        split_sizes = {name: len(values) for name, values in split_manifest.get("splits", {}).items()}
    physical_summary = summarize_case_dirs(case_dirs) if case_dirs else {"physical_all_accepted": False, "physical_accepted_cases": 0, "metrics": {}, "failure_counts": {}}
    summary = {
        "case_count": len(rows),
        "accepted_cases": len(accepted),
        "all_accepted": len(rows) == len(accepted),
        "physical_accepted_cases": physical_summary.get("physical_accepted_cases", 0),
        "physical_all_accepted": bool(physical_summary.get("physical_all_accepted", False)),
        "physical_failure_counts": physical_summary.get("failure_counts", {}),
        "split_sizes": split_sizes,
        "mean_basis_f_projection_rel_l2": _mean(rows, "basis_f_projection_rel_l2"),
        "max_g_normalization_error": _max(rows, "g_normalization_max_abs_error"),
        "mean_repeat_initial_difficulty": _mean(rows, "repeat_initial_difficulty"),
        "physical_metrics": physical_summary.get("metrics", {}),
        "cases": rows,
    }
    (root / "d0_audit.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (root / "d0_audit.md").write_text(_audit_markdown(summary))
    return summary


def assemble_sharded_d0(*, shards_root: str | Path, out_dir: str | Path, split_counts: dict[str, int]) -> dict[str, Any]:
    shards = sorted(Path(shards_root).glob("*/cases/*"))
    out = Path(out_dir)
    cases_root = out / "cases"
    cases_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for src in shards:
        if not src.is_dir():
            continue
        dst = cases_root / src.name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        rows.append({"case_id": dst.name, "case_dir": str(dst), "source_origin": "randomized_public_equation", "status": "accepted"})
    rows = sorted(rows, key=lambda row: row["case_id"])
    manifest = {
        "backend": "assembled_deterministic_coarse_vpfp_grid_solver",
        "case_schema": "vpfp_factorized_public_hidden_v1",
        "case_count": len(rows),
        "cases": rows,
        "splits": make_splits(rows, split_counts),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (out / "split_manifest.json").write_text(json.dumps(_split_manifest(manifest), indent=2, sort_keys=True) + "\n")
    return manifest


def iter_case_dirs(root: str | Path) -> list[Path]:
    path = Path(root)
    if not path.exists():
        return []
    return sorted(item for item in path.iterdir() if item.is_dir() and (item / "public_question.json").exists())


def _split_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_schema": manifest.get("case_schema", "vpfp_factorized_public_hidden_v1"),
        "source_manifest": str(manifest.get("manifest_path", "manifest.json")),
        "splits": manifest.get("splits", {}),
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [float(row[key]) for row in rows if key in row and np.isfinite(float(row[key]))]
    return float(np.mean(vals)) if vals else float("inf")


def _max(rows: list[dict[str, Any]], key: str) -> float:
    vals = [float(row[key]) for row in rows if key in row and np.isfinite(float(row[key]))]
    return float(np.max(vals)) if vals else float("inf")


def _audit_markdown(summary: dict[str, Any]) -> str:
    return "\n".join([
        "# VPFP D0 Audit",
        "",
        f"- case count: `{summary['case_count']}`",
        f"- accepted cases: `{summary['accepted_cases']}`",
        f"- all accepted: `{summary['all_accepted']}`",
        f"- physical accepted cases: `{summary.get('physical_accepted_cases')}`",
        f"- physical all accepted: `{summary.get('physical_all_accepted')}`",
        f"- physical failure counts: `{summary.get('physical_failure_counts', {})}`",
        f"- mean basis f projection rel L2: `{summary['mean_basis_f_projection_rel_l2']}`",
        f"- max g normalization error: `{summary['max_g_normalization_error']}`",
        f"- mean repeat-initial difficulty: `{summary['mean_repeat_initial_difficulty']}`",
    ]) + "\n"
