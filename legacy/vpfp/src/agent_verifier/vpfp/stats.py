from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from agent_verifier.vpfp.basis import basis_projection_metrics
from agent_verifier.vpfp.io import load_case
from agent_verifier.vpfp.operators import EPS, f_from_rho_g, make_axes, poisson_electric_field, relative_l2

Array = np.ndarray


@dataclass(frozen=True)
class VPFPPhysicalGateConfig:
    g_normalization_max: float = 5.0e-4
    rho_min: float = -1.0e-8
    g_min: float = -1.0e-10
    poisson_field_rel_l2_max: float = 1.0e-7
    basis_f_projection_rel_l2_max: float = 0.20
    max_abs_value: float = 1.0e8
    min_repeat_initial_difficulty: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "g_normalization_max": float(self.g_normalization_max),
            "rho_min": float(self.rho_min),
            "g_min": float(self.g_min),
            "poisson_field_rel_l2_max": float(self.poisson_field_rel_l2_max),
            "basis_f_projection_rel_l2_max": float(self.basis_f_projection_rel_l2_max),
            "max_abs_value": float(self.max_abs_value),
            "min_repeat_initial_difficulty": float(self.min_repeat_initial_difficulty),
        }


DEFAULT_SUMMARY_STATS_KEYS = (
    "rho_mean",
    "rho_std",
    "rho_rms",
    "f_rms",
    "source_rms",
    "electric_rms",
    "field_energy_mean",
    "repeat_initial_difficulty",
    "rho_temporal_rel_l2",
    "f_temporal_rel_l2",
    "mean_abs_u_vx",
    "mean_abs_u_vy",
    "thermal_energy_mean",
    "entropy_mean",
    "basis_f_projection_rel_l2",
)

# Generated adversarial cases are answer-first and derive public source by residual,
# while D0 cases are equation-first with randomized public source. Compare hidden
# solution shape here; keep source/projection checks in the physical audit.
DEFAULT_STATS_KEYS = (
    "rho_mean",
    "rho_std",
    "rho_rms",
    "f_rms",
    "electric_rms",
    "field_energy_mean",
    "repeat_initial_difficulty",
    "rho_temporal_rel_l2",
    "f_temporal_rel_l2",
    "mean_abs_u_vx",
    "mean_abs_u_vy",
    "thermal_energy_mean",
    "entropy_mean",
)


def summarize_case(case_dir: str | Path, *, gate_config: VPFPPhysicalGateConfig | None = None) -> dict[str, Any]:
    case = load_case(case_dir)
    grid = case["grid"]
    hidden = case["hidden_arrays"]
    public = case["public_arrays"]
    rho = np.asarray(hidden["rho_true"], dtype=float)
    g = np.asarray(hidden["g_true"], dtype=float)
    source = np.asarray(public["source"], dtype=float)
    electric = np.asarray(hidden["E_true"], dtype=float)
    phi = np.asarray(hidden["phi_true"], dtype=float)
    f = f_from_rho_g(rho, g)
    axes = make_axes(grid, case["time"])
    vx = axes["vx"][None, None, None, :, None]
    vy = axes["vy"][None, None, None, None, :]
    weights = float(grid.velocity_cell_area)
    g_mass = np.sum(g, axis=(-2, -1)) * weights
    ux = np.sum(g * vx, axis=(-2, -1)) * weights
    uy = np.sum(g * vy, axis=(-2, -1)) * weights
    thermal = np.sum(g * ((vx - ux[..., None, None]) ** 2 + (vy - uy[..., None, None]) ** 2), axis=(-2, -1)) * weights
    entropy = -np.sum(g * np.log(np.maximum(g, EPS)), axis=(-2, -1)) * weights
    _phi_re, electric_re = poisson_electric_field(rho, grid)
    repeat = _repeat_initial_difficulty(f)
    basis_metrics = {}
    if case["public_question"].get("basis"):
        basis_metrics = basis_projection_metrics(rho, g, grid, case["public_question"].get("basis", {}).get("config"))
    row: dict[str, Any] = {
        "case_id": case["case_id"],
        "case_dir": str(case["case_dir"]),
        "source_origin": case["public_question"].get("source_origin"),
        "finite": bool(np.all(np.isfinite(rho)) and np.all(np.isfinite(g)) and np.all(np.isfinite(source)) and np.all(np.isfinite(electric))),
        "rho_min": float(np.min(rho)),
        "rho_max": float(np.max(rho)),
        "rho_mean": float(np.mean(rho)),
        "rho_std": float(np.std(rho)),
        "rho_rms": float(np.sqrt(np.mean(rho**2))),
        "g_min": float(np.min(g)),
        "g_max": float(np.max(g)),
        "g_normalization_max_abs_error": float(np.max(np.abs(g_mass - 1.0))),
        "g_normalization_mean_abs_error": float(np.mean(np.abs(g_mass - 1.0))),
        "f_rms": float(np.sqrt(np.mean(f**2))),
        "source_rms": float(np.sqrt(np.mean(source**2))),
        "source_abs_max": float(np.max(np.abs(source))),
        "electric_rms": float(np.sqrt(np.mean(electric**2))),
        "field_energy_mean": float(0.5 * np.mean(np.sum(electric**2, axis=1))),
        "poisson_field_rel_l2": relative_l2(electric - electric_re, electric_re),
        "phi_rms": float(np.sqrt(np.mean(phi**2))),
        "repeat_initial_difficulty": repeat,
        "rho_temporal_rel_l2": relative_l2(rho - rho[:1], rho),
        "f_temporal_rel_l2": relative_l2(f - f[:1], f),
        "mass_initial": float(np.mean(rho[0])),
        "mass_final": float(np.mean(rho[-1])),
        "mass_relative_change": float(abs(np.mean(rho[-1]) - np.mean(rho[0])) / (abs(np.mean(rho[0])) + EPS)),
        "mean_abs_u_vx": float(np.mean(np.abs(ux))),
        "mean_abs_u_vy": float(np.mean(np.abs(uy))),
        "thermal_energy_mean": float(np.mean(thermal)),
        "thermal_energy_min": float(np.min(thermal)),
        "entropy_mean": float(np.mean(entropy)),
        "max_abs_value": float(max(np.max(np.abs(rho)), np.max(np.abs(g)), np.max(np.abs(source)), np.max(np.abs(electric)))),
        **basis_metrics,
    }
    row["physical_failures"] = physical_failures(row, gate_config or VPFPPhysicalGateConfig())
    row["physical_accepted"] = not row["physical_failures"]
    return row


def summarize_case_dirs(case_dirs: Iterable[str | Path], *, gate_config: VPFPPhysicalGateConfig | None = None) -> dict[str, Any]:
    rows = [summarize_case(path, gate_config=gate_config) for path in case_dirs]
    return summarize_rows(rows, gate_config=gate_config)


def summarize_rows(rows: list[dict[str, Any]], *, gate_config: VPFPPhysicalGateConfig | None = None) -> dict[str, Any]:
    cfg = gate_config or VPFPPhysicalGateConfig()
    aggregate = {
        "case_count": len(rows),
        "physical_accepted_cases": sum(1 for row in rows if row.get("physical_accepted")),
        "physical_all_accepted": bool(rows) and all(row.get("physical_accepted") for row in rows),
        "gate_config": cfg.to_dict(),
        "metrics": {},
        "failure_counts": {},
        "cases": rows,
    }
    for row in rows:
        for failure in row.get("physical_failures", []):
            aggregate["failure_counts"][failure] = int(aggregate["failure_counts"].get(failure, 0)) + 1
    keys = sorted({key for row in rows for key, value in row.items() if isinstance(value, (int, float)) and np.isfinite(float(value))})
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows if key in row and isinstance(row[key], (int, float)) and np.isfinite(float(row[key]))], dtype=float)
        if values.size:
            aggregate["metrics"][key] = _stats(values)
    return aggregate


def compare_summaries(reference: dict[str, Any], candidate: dict[str, Any], *, keys: tuple[str, ...] = DEFAULT_STATS_KEYS, z_threshold: float = 3.0) -> dict[str, Any]:
    rows = []
    for key in keys:
        ref_stats = reference.get("metrics", {}).get(key)
        cand_stats = candidate.get("metrics", {}).get(key)
        if not ref_stats or not cand_stats:
            continue
        ref_mean = float(ref_stats["mean"])
        cand_mean = float(cand_stats["mean"])
        ref_std = max(float(ref_stats.get("std", 0.0)), abs(ref_mean) * 0.05, EPS)
        z_gap = abs(cand_mean - ref_mean) / ref_std
        rel_gap = abs(cand_mean - ref_mean) / (abs(ref_mean) + EPS)
        rows.append({
            "metric": key,
            "reference_mean": ref_mean,
            "candidate_mean": cand_mean,
            "reference_std_for_gap": ref_std,
            "z_gap": float(z_gap),
            "relative_gap": float(rel_gap),
            "within_threshold": bool(z_gap <= float(z_threshold)),
        })
    max_z = max([row["z_gap"] for row in rows], default=float("inf"))
    mean_z = float(np.mean([row["z_gap"] for row in rows])) if rows else float("inf")
    return {
        "comparison": "candidate_vs_reference_summary_stats",
        "reference_case_count": reference.get("case_count", 0),
        "candidate_case_count": candidate.get("case_count", 0),
        "z_threshold": float(z_threshold),
        "max_z_gap": float(max_z),
        "mean_z_gap": float(mean_z),
        "accepted": bool(rows) and max_z <= float(z_threshold) and candidate.get("physical_all_accepted", False),
        "metric_rows": rows,
    }


def physical_failures(row: dict[str, Any], config: VPFPPhysicalGateConfig) -> list[str]:
    failures = []
    if not bool(row.get("finite", False)):
        failures.append("nonfinite")
    if float(row.get("rho_min", -float("inf"))) < float(config.rho_min):
        failures.append("rho_min")
    if float(row.get("g_min", -float("inf"))) < float(config.g_min):
        failures.append("g_min")
    if float(row.get("g_normalization_max_abs_error", float("inf"))) > float(config.g_normalization_max):
        failures.append("g_normalization")
    if float(row.get("poisson_field_rel_l2", float("inf"))) > float(config.poisson_field_rel_l2_max):
        failures.append("poisson_field_consistency")
    if float(row.get("basis_f_projection_rel_l2", 0.0)) > float(config.basis_f_projection_rel_l2_max):
        failures.append("basis_projection")
    if float(row.get("max_abs_value", float("inf"))) > float(config.max_abs_value):
        failures.append("max_abs_value")
    if float(row.get("repeat_initial_difficulty", 0.0)) < float(config.min_repeat_initial_difficulty):
        failures.append("repeat_initial_too_easy")
    return failures


def case_dirs_from_root(root: str | Path) -> list[Path]:
    path = Path(root)
    if not path.exists():
        return []
    return sorted(item for item in path.iterdir() if item.is_dir() and (item / "public_question.json").exists())


def case_dirs_from_split_manifest(path: str | Path, split: str) -> list[Path]:
    data = json.loads(Path(path).read_text())
    return [Path(row["case_dir"]) for row in data.get("splits", {}).get(split, [])]


def write_summary(summary: dict[str, Any], out_dir: str | Path, *, name: str = "vpfp_stats") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (out / f"{name}.md").write_text(_markdown_summary(summary))


def write_comparison(comparison: dict[str, Any], out_dir: str | Path, *, name: str = "vpfp_stats_comparison") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    lines = ["# VPFP Statistics Comparison", "", f"- accepted: `{comparison['accepted']}`", f"- reference cases: `{comparison['reference_case_count']}`", f"- candidate cases: `{comparison['candidate_case_count']}`", f"- max z gap: `{comparison['max_z_gap']}`", f"- mean z gap: `{comparison['mean_z_gap']}`", "", "| metric | reference mean | candidate mean | z gap | relative gap |", "| --- | ---: | ---: | ---: | ---: |"]
    for row in comparison.get("metric_rows", []):
        lines.append(f"| `{row['metric']}` | {row['reference_mean']:.6g} | {row['candidate_mean']:.6g} | {row['z_gap']:.3g} | {row['relative_gap']:.3g} |")
    (out / f"{name}.md").write_text("\n".join(lines) + "\n")


def _repeat_initial_difficulty(f: Array) -> float:
    pred = np.repeat(f[:1], f.shape[0], axis=0)
    return relative_l2(pred - f, f)


def _stats(values: Array) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p05": float(np.quantile(arr, 0.05)),
        "p50": float(np.quantile(arr, 0.50)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
    }


def _markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# VPFP Physical Statistics",
        "",
        f"- case count: `{summary['case_count']}`",
        f"- physical accepted cases: `{summary['physical_accepted_cases']}`",
        f"- physical all accepted: `{summary['physical_all_accepted']}`",
        f"- failure counts: `{summary.get('failure_counts', {})}`",
        "",
        "| metric | mean | std | min | p50 | p95 | max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in DEFAULT_SUMMARY_STATS_KEYS:
        stats = summary.get("metrics", {}).get(key)
        if not stats:
            continue
        lines.append(f"| `{key}` | {stats['mean']:.6g} | {stats['std']:.6g} | {stats['min']:.6g} | {stats['p50']:.6g} | {stats['p95']:.6g} | {stats['max']:.6g} |")
    return "\n".join(lines) + "\n"
