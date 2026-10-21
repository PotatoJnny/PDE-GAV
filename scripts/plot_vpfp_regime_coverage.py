"""Plot completed, SHA-bound VPFP regime audits without scoring a winner."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path

AXES = ("source_fixed_probe_relative_to_collision", "net_momentum_injection_relative_to_collision",
        "net_energy_injection_relative_to_collision")
TITLES = ("Fixed-probe source budget", "Net momentum injection budget", "Net energy injection budget")


def file_sha(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("regular completed audit file required")
    return sha256(path.read_bytes()).hexdigest()


def load_completed(path, expected):
    path = Path(path).absolute()
    result = json.loads(path.read_text())
    if result.get("execution_passed") is not True or result.get("stage") != "complete":
        raise ValueError("only a completed descriptive audit can be plotted")
    frozen, reports = {str(path): expected}, []
    for target, item in sorted(result["targets"].items()):
        if item.get("status") != "descriptive_audit_completed":
            raise ValueError("every requested target must have completed")
        paths = [("A", item["A_audit_path"]), ("B", item["B_audit_path"])] if "A_audit_path" in item else [("native", item["audit_path"])]
        for view, name in paths:
            source = Path(name).absolute()
            if not source.is_relative_to(path.parent):
                raise ValueError("audit files must belong to the completed result directory")
            frozen[str(source)] = file_sha(source)
            audit = json.loads(source.read_text())
            if audit.get("execution_passed") is not True or audit.get("admitted") is not False:
                raise ValueError("expected completed non-admitting research audit")
            parts = Path(item["directory"]).parts
            index = next((i for i, part in enumerate(parts) if part.startswith("seed-")), None)
            label = target if index is None else "/".join(parts[index:index+2])
            step = next((part for part in parts if part.startswith("step-") and part[5:].isdigit()), None)
            if step is not None: label += "/"+str(int(step[5:])//1000)+"k"
            if view != "native": label += "/"+view
            reports.append((label, audit))
    if not reports:
        raise ValueError("no completed model report")
    identities = {(row["geometry"]["fit_identity_sha256"], row.get("source_manifest_sha256"),
                   row.get("reference_source_manifest_sha256")) for _, row in reports}
    if len(identities) != 1:
        raise ValueError("plots cannot compare different reference geometries/sources")
    return reports, frozen


def coverage_vector(report, role="generated"):
    rows = report["comparisons"][role]["joint_strata"]
    if [row["stratum"] for row in rows] != list(range(8)):
        raise ValueError("expected all eight frozen condition strata")
    return [row["neighbors"]["5"].get("reference_coverage") for row in rows]


def budget_display_limits(reports):
    """One nonnegative display range from every registered curve, not a cutoff."""
    maximum = 0.0
    for label, report in reports:
        for name in AXES:
            curve = report["budget_curves"][name]["curve"]
            if not isinstance(curve, list) or not curve:
                raise ValueError(f"nonempty registered budget curve required: {label}/{name}")
            budgets = []
            for point in curve:
                value = point.get("maximum_dimensionless_budget")
                if type(value) not in (int, float):
                    raise ValueError("budget values must be finite nonnegative numbers")
                try:
                    value = float(value)
                except OverflowError as error:
                    raise ValueError("budget values must be finite nonnegative numbers") from error
                if not math.isfinite(value) or value < 0:
                    raise ValueError("budget values must be finite nonnegative numbers")
                budgets.append(value)
            if any(right <= left for left, right in zip(budgets, budgets[1:])):
                raise ValueError("registered budget curve must be strictly increasing")
            maximum = max(maximum, max(budgets))
    if maximum <= 0:
        # Matplotlib expands identical limits; never invent a negative range or
        # an unregistered positive endpoint for a degenerate all-zero curve.
        raise ValueError("a positive registered budget is required for a nondegenerate display range")
    return (0.0, maximum)


def plot(path, expected, output):
    reports, frozen = load_completed(path, expected)
    budget_limits = budget_display_limits(reports)
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("new figure output required; no overwrite")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    output.mkdir(parents=True)
    fig, panels = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey="row", constrained_layout=True)
    palette = plt.get_cmap("tab20").colors
    for color, (label, report) in enumerate(reports):
        for column, axis in enumerate(AXES):
            curve = report["budget_curves"][axis]["curve"]
            x = [p["maximum_dimensionless_budget"] for p in curve]
            y = [np.mean(p["k5_reference_coverage_by_stratum"]) if all(v is not None for v in p["k5_reference_coverage_by_stratum"]) else np.nan for p in curve]
            panels[0, column].plot(x, y, label=label, color=palette[color % len(palette)])
            panels[1, column].plot(x, [p["retained_fraction_all_attempts"] for p in curve], color=palette[color % len(palette)])
    for column, title in enumerate(TITLES):
        panels[0, column].set_title(title)
        for axis in panels[:, column]:
            axis.set_xscale("symlog", linthresh=1.e-4)
            axis.set_xlim(*budget_limits)
            axis.set_ylim(-.02, 1.02); axis.grid(alpha=.2)
        panels[1, column].set_xlabel("Maximum dimensionless budget (not admission threshold)")
    panels[0, 0].set_ylabel("Mean k5 reference coverage\n(equal weight for 8 strata)")
    panels[1, 0].set_ylabel("Retained / all generation attempts")
    handles, labels = panels[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.01), ncol=min(4, len(reports)))
    fig.suptitle("VPFP: separate source-budget and reference-coverage diagnostics\nUnresolved numerical checks are not physical failures; retained samples are not admitted", fontsize=12)
    fig.savefig(output/"fidelity-coverage-curves.png", dpi=160, bbox_inches="tight"); plt.close(fig)

    reference = coverage_vector(reports[0][1], "reference_null")
    values = np.asarray([reference]+[coverage_vector(row) for _, row in reports], dtype=float)
    fig, axis = plt.subplots(figsize=(11, max(3.5, .65*len(values)+1.7)), constrained_layout=True)
    display = axis.imshow(values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    axis.set_yticks(range(len(values)), ["Independent reference B (development null)"]+[name for name, _ in reports])
    axis.set_xticks(range(8), [f"nu bin {i//2+1}, T bin {i%2+1}" for i in range(8)], rotation=30, ha="right")
    for i in range(len(values)):
        for j in range(8):
            if np.isfinite(values[i, j]):
                axis.text(j, i, f"{100*values[i,j]:.1f}%", ha="center", va="center", color="white" if values[i,j]<.55 else "black")
    axis.set_title("Unrestricted k5 reference-neighborhood coverage\nSupport coverage is not physical validity; these are not trajectory matches")
    fig.colorbar(display, ax=axis, label="Reference neighborhood coverage")
    fig.savefig(output/"condition-coverage.png", dpi=160); plt.close(fig)
    receipt = {"schema":"vpfp_regime_coverage_figures_v1", "execution_passed":True, "admitted":False,
        "files":frozen, "models":[name for name, _ in reports], "new_field_evaluations":0,
        "new_generated_answers":0, "automatic_candidate_selection":False,
        "single_budget_axis_does_not_constrain_the_other_axes":True,
        "budget_display_range":list(budget_limits),
        "budget_display_range_shared_by_all_panels":True,
        "numerical_status_is_in_original_reports_not_certified_by_plot":True,
        "figures":["fidelity-coverage-curves.png", "condition-coverage.png"]}
    (output/"figure-provenance.json").write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False)+"\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-result", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.audit_result, args.expected_sha256, args.output)
