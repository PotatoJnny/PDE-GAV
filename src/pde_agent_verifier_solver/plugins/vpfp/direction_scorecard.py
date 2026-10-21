"""Cheap, two-sided VPFP development-direction screens of saved features.

The user compares a representation candidate with a SAME-objective baseline.
This consumes real generated-trajectory features and already computed audits;
it never samples a model, trains, reruns a test, or admits a checkpoint. Opened
reference banks are development data, not independent confirmation evidence.
"""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Mapping, Sequence

import numpy as np


SCHEMA = "vpfp_development_direction_scorecard_v1"
_DESIGN = {
    "joint_sw_required_relative_reduction": 0.20,
    "require_reduction_in_both_physical_and_derivative_joint_sw": True,
    "mean_gap_regression_tolerance": "max(abs(reference_A_mean-reference_B_mean), 0.1*reference_RMS, feature_scale_floor)",
    "reference_center": "equal_weight_mean_of_reference_A_and_B_means",
    "reference_scale": "equal_weight_RMS_of_reference_A_and_B_case_statistics",
    "early_time_fraction_maximum": 0.25,
    "late_time_fraction_minimum": 0.5,
    "momentum_amplitude_collapse_ratio": 0.5,
    "momentum_collapse_guard": "candidate probe-value RMS below half reference center; disabled only below feature floor",
    "mmd_use": "descriptive_effect_only; saturation_does_not_establish_agreement",
    "joint_effects": "saved_audits_only; no_new_permutations_or_bootstrap",
}


def _real(value, label, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{label} must be a finite real number")
    value = float(value)
    if not np.isfinite(value) or value < 0 or (positive and value <= 0):
        raise ValueError(f"{label} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def _integer(value, label, *, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return int(value)


def _name(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _schema(descriptors):
    result = {}
    for index, item in enumerate(descriptors):
        name = item.get("name") if isinstance(item, Mapping) else getattr(item, "name", None)
        floor = item.get("scale_floor") if isinstance(item, Mapping) else getattr(item, "scale_floor", None)
        _name(name, "feature name")
        if name in result:
            raise ValueError(f"duplicate feature name: {name}")
        result[name] = (index, _real(floor, f"{name} scale floor", positive=True))
    if not result:
        raise ValueError("feature schema must not be empty")
    return result


def _times(values, label):
    raw = np.asarray(values)
    if (raw.dtype.kind not in "fiu" or raw.ndim != 1 or len(raw) < 3
            or not np.isfinite(raw).all() or raw[0] != 0 or raw[-1] != 1
            or not np.all(np.diff(raw) > 0)):
        raise ValueError(f"{label} must increase strictly from 0 to 1 with interior probes")
    return raw.astype(np.float64)


def _rms(values, axis=None):
    """Scale first: low variance/zero fields never create unstable ratios."""
    values = np.asarray(values, dtype=np.float64)
    maximum = np.max(np.abs(values), axis=axis, keepdims=True)
    safe = np.where(maximum > 0, maximum, 1.)
    answer = maximum * np.sqrt(np.mean(np.square(values / safe), axis=axis, keepdims=True))
    return np.squeeze(answer, axis=axis) if axis is not None else float(answer.reshape(-1)[0])


def _mean(values):
    maximum = float(np.max(np.abs(values)))
    return 0. if maximum == 0 else maximum * float(np.mean(values / maximum))


def _metric_columns(physical_schema, derivative_schema, physical_times, derivative_times,
                    macro_probe_count, phase_probe_count):
    """Fixed selectors; time-major probe ordering is the extractor contract."""
    definitions = []

    def add(name, kind, columns, *, reduction="identity", window=None, time=None):
        schema = physical_schema if kind == "physical" else derivative_schema
        missing = [column for column in columns if column not in schema]
        if missing:
            raise ValueError(f"missing required {kind} features: {missing[:3]}")
        definitions.append({"name": name, "kind": kind,
            "indices": [schema[column][0] for column in columns],
            "scale_floor": max(schema[column][1] for column in columns),
            "reduction": reduction, "window": window, "time_fraction": time})

    for index, time in enumerate(physical_times):
        window = "early" if time <= .25 else "late" if time >= .5 else None
        if window is None:
            continue
        for statistic in ("mean", "std", "spectrum_low", "spectrum_middle", "spectrum_high"):
            column = f"kinetic_energy.time{index}.{statistic}"
            add(f"physical.{column}", "physical", [column], window=window, time=float(time))
    for quantity, component in (("f", "dt"), ("f", "dx"), ("f", "dy"),
            ("rho", "value"), ("rho", "dt"),
            ("kinetic_energy", "dt"), ("kinetic_energy", "dx"), ("kinetic_energy", "dy"),
            ("momentum_x", "value"), ("momentum_y", "value"),
            ("momentum_x", "dt"), ("momentum_y", "dt")):
        count = phase_probe_count if quantity == "f" else macro_probe_count
        for window in ("all", "early", "late"):
            selected = [index for index, time in enumerate(derivative_times)
                        if window == "all" or (window == "early" and time <= .25)
                        or (window == "late" and time >= .5)]
            columns = [f"{quantity}.probe{time * count + probe}.{component}"
                       for time in selected for probe in range(count)]
            add(f"derivative.{quantity}.{component}.{window}.rms", "derivative", columns,
                reduction="probe_rms", window=window)
    return definitions


def _saved_effect(report, role, statistic):
    try:
        value = report["comparisons"][role]["statistics"][statistic]["effect"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"missing saved audit effect: {role}/{statistic}") from exc
    return _real(value, f"saved {role}/{statistic}")


def build_vpfp_direction_scorecard(*, regime: str, seed: int, objective: str,
        baseline_arm: str, samples: Mapping, physical_schema: Sequence,
        derivative_schema: Sequence, joint_reports: Mapping,
        physical_time_fractions: Sequence, derivative_time_fractions: Sequence,
        macro_probe_count: int, phase_probe_count: int):
    """Compare candidate arms to one same-objective baseline, without writes.

    ``samples[role]['physical'|'derivative']`` are saved [trajectory, feature]
    arrays. Roles include reference_a, reference_b, baseline_arm, and candidates.
    Reference and generated row counts may differ; within-role row identities,
    checkpoint freezes, condition strata and audit/schema bindings are checked
    by the calling experiment, not inferred from feature values here.
    ``joint_reports[arm][kind]`` are the existing complete distribution audits.
    """
    regime, objective = _name(regime, "regime"), _name(objective, "objective")
    seed = _integer(seed, "seed")
    baseline_arm = _name(baseline_arm, "baseline arm")
    if baseline_arm in ("reference_a", "reference_b"):
        raise ValueError("baseline must be a generated arm")
    if not isinstance(samples, Mapping) or not {"reference_a", "reference_b", baseline_arm}.issubset(samples):
        raise ValueError("samples require reference_a, reference_b, and baseline arm")
    for role in samples:
        _name(role, "sample role")
    arms = sorted(set(samples) - {"reference_a", "reference_b"})
    if len(arms) < 2:
        raise ValueError("at least one candidate arm is required")
    for arm in arms:
        _name(arm, "arm")
    if not isinstance(joint_reports, Mapping) or set(joint_reports) != set(arms):
        raise ValueError("joint reports must match the generated arms exactly")
    schemas = {"physical": _schema(physical_schema), "derivative": _schema(derivative_schema)}
    physical_times = _times(physical_time_fractions, "physical time fractions")
    derivative_times = _times(derivative_time_fractions, "derivative time fractions")
    definitions = _metric_columns(schemas["physical"], schemas["derivative"],
        physical_times, derivative_times, _integer(macro_probe_count, "macro probe count", minimum=1),
        _integer(phase_probe_count, "phase probe count", minimum=1))
    matrices = {}
    for role, data in samples.items():
        matrices[role] = {}
        for kind, schema in schemas.items():
            if not isinstance(data, Mapping) or kind not in data:
                raise ValueError(f"{role} needs physical and derivative arrays")
            array = np.asarray(data[kind])
            if (array.dtype.kind not in "fiu" or array.ndim != 2 or array.shape[0] < 2
                    or array.shape[1] != len(schema) or not np.isfinite(array).all()):
                raise ValueError(f"{role}/{kind} must be finite real [N>=2,{len(schema)}] features")
            matrices[role][kind] = np.asarray(array, dtype=np.float64)
        if len(matrices[role]["physical"]) != len(matrices[role]["derivative"]):
            raise ValueError(f"{role} physical/derivative trajectory counts disagree")
    summaries = {}
    for definition in definitions:
        summaries[definition["name"]] = {}
        for role, data in matrices.items():
            selected = data[definition["kind"]][:, definition["indices"]]
            values = _rms(selected, axis=1) if definition["reduction"] == "probe_rms" else selected[:, 0]
            if ((definition["reduction"] == "probe_rms" or definition["name"].endswith(
                    (".std", ".spectrum_low", ".spectrum_middle", ".spectrum_high")))
                    and np.any(values < 0)):
                raise ValueError(f"negative nonnegative statistic: {definition['name']}")
            summaries[definition["name"]][role] = {"mean": _mean(values), "rms": _rms(values)}
    arm_reports = {}
    for arm in arms:
        metrics, regressions, vetoes = {}, [], []
        for definition in definitions:
            name, floor = definition["name"], definition["scale_floor"]
            summary = summaries[name]
            a, b = summary["reference_a"], summary["reference_b"]
            center = a["mean"] / 2 + b["mean"] / 2
            scale = max(_rms([a["rms"], b["rms"]]), floor)
            noise_gap = abs(a["mean"] - b["mean"])
            tolerance = max(noise_gap, .1 * scale, floor)
            base_gap = abs(summary[baseline_arm]["mean"] - center)
            signed_gap = summary[arm]["mean"] - center
            gap = abs(signed_gap)
            regression = gap > base_gap + tolerance
            momentum = name.startswith(("derivative.momentum_x.value.", "derivative.momentum_y.value."))
            collapsed = bool(momentum and center > floor and summary[arm]["mean"] < .5 * center)
            if regression:
                regressions.append(name)
            if collapsed:
                vetoes.append(name)
            metrics[name] = {"reference_a_mean": a["mean"], "reference_b_mean": b["mean"],
                "reference_center": center, "reference_scale": scale,
                "reference_mean_gap": noise_gap, "baseline_mean": summary[baseline_arm]["mean"],
                "candidate_mean": summary[arm]["mean"], "signed_mean_gap": signed_gap,
                "absolute_mean_gap": gap, "baseline_absolute_mean_gap": base_gap,
                "gap_change_from_baseline": gap - base_gap, "allowed_gap_regression": tolerance,
                "normalized_signed_mean_gap": signed_gap / scale,
                "regression_beyond_reference_tolerance": bool(regression),
                "momentum_amplitude_collapse_veto": collapsed,
                "window": definition["window"], "time_fraction": definition["time_fraction"]}
        joint = {}
        for kind in schemas:
            try:
                baseline_report = joint_reports[baseline_arm][kind]
                candidate_report = joint_reports[arm][kind]
            except (KeyError, TypeError) as exc:
                raise ValueError(f"missing saved {kind} audit") from exc
            joint[kind] = {}
            for statistic in ("sliced_wasserstein_max", "multiband_mmd_max"):
                base = _saved_effect(baseline_report, "generated", statistic)
                candidate = _saved_effect(candidate_report, "generated", statistic)
                reference = _saved_effect(baseline_report, "reference_null", statistic)
                if reference != _saved_effect(candidate_report, "reference_null", statistic):
                    raise ValueError("paired arms must use the same saved reference-null effects")
                joint[kind][statistic] = {"baseline_effect": base, "candidate_effect": candidate,
                    "reference_null_effect": reference,
                    "baseline_above_reference_noise": bool(base > reference),
                    "relative_reduction": None if base == 0 else 1 - candidate / base,
                    "twenty_percent_reduction": bool(base > reference and candidate <= .8 * base)}
        joint_pass = all(joint[kind]["sliced_wasserstein_max"]["twenty_percent_reduction"] for kind in joint)
        candidate_pass = arm != baseline_arm and joint_pass and not regressions and not vetoes
        arm_reports[arm] = {"role": "baseline" if arm == baseline_arm else "candidate",
            "direction_supported": bool(candidate_pass), "joint_sw_reduction_pass": bool(joint_pass),
            "mean_gap_nonregression_pass": not regressions, "momentum_collapse_veto": bool(vetoes),
            "regressed_metrics": regressions, "momentum_veto_metrics": vetoes,
            "joint_effects": joint, "metrics": metrics}
    feature_plan = {"physical_time_fractions": physical_times.tolist(),
        "derivative_time_fractions": derivative_times.tolist(),
        "macro_probe_count": int(macro_probe_count), "phase_probe_count": int(phase_probe_count),
        "schema_name_order_and_floor_sha256": sha256(json.dumps(schemas, sort_keys=True).encode()).hexdigest()}
    result = {"schema": SCHEMA, "regime": regime, "seed": seed, "objective": objective,
        "baseline_arm": baseline_arm, "design": dict(_DESIGN), "arms": arm_reports,
        "feature_plan": feature_plan,
        "sample_counts": {role: len(data["physical"]) for role, data in matrices.items()},
        "status": "development_screen_not_independent_confirmation", "admitted": False,
        "recommend_scale_up": False, "recommend_fresh_confirmation": False,
        "limitations": ["heuristic direction screen, not an equivalence or significance test",
            "opened reference bank is development data; no independent confirmation",
            "reference A-B mean gap is a descriptive noise baseline, not a confidence bound",
            "RMS reductions alone are not improvement; signed and absolute reference gaps are retained",
            "momentum veto detects severe amplitude suppression, not every form of diversity collapse",
            "time-window probe RMS is not a converged domain-integral norm",
            "caller must bind sample identities, condition strata, feature plans, objective and checkpoint freezes",
            "both regimes and at least two distinct training seeds are required before recommending fresh confirmation"]}
    try:
        json.dumps(result, allow_nan=False)
    except (ValueError, OverflowError) as exc:
        raise ValueError("scorecard statistics exceeded finite arithmetic range") from exc
    return result


def aggregate_vpfp_direction_scorecards(reports, *, required_regimes=("weak", "strong"), minimum_seeds=2):
    """Require every available replicate, separately in both frozen regimes.

    Even a positive development screen recommends only fresh confirmation, never
    a large experiment or production admission. Different objectives may not be
    pooled, and duplicate regime/seed entries do not count as replication.
    """
    reports = list(reports)
    minimum_seeds = _integer(minimum_seeds, "minimum seeds", minimum=2)
    required_regimes = tuple(required_regimes)
    if len(required_regimes) < 2 or len(set(required_regimes)) != len(required_regimes):
        raise ValueError("at least two distinct required regimes must be frozen")
    for regime in required_regimes:
        _name(regime, "required regime")
    if not reports:
        raise ValueError("at least one development scorecard is required")
    first = reports[0]
    candidates = {name for name, arm in first.get("arms", {}).items() if arm.get("role") == "candidate"}
    if not candidates:
        raise ValueError("scorecards must contain candidate arms")
    seen = set()
    for report in reports:
        if (report.get("schema") != SCHEMA or report.get("design") != _DESIGN
                or report.get("objective") != first.get("objective")
                or report.get("baseline_arm") != first.get("baseline_arm")
                or report.get("feature_plan") != first.get("feature_plan")
                or {name for name, arm in report.get("arms", {}).items() if arm.get("role") == "candidate"} != candidates):
            raise ValueError("aggregate needs same-objective, same-design, same-arm scorecards")
        key = (report.get("regime"), _integer(report.get("seed"), "seed"))
        if key[0] not in required_regimes or key in seen:
            raise ValueError("unexpected regime or duplicate regime/seed; duplicates are not replication")
        seen.add(key)
    results = {}
    for candidate in sorted(candidates):
        regimes = {}
        for regime in required_regimes:
            selected = [report for report in reports if report["regime"] == regime]
            complete = len(selected) >= minimum_seeds
            supported = complete and all(report["arms"][candidate]["direction_supported"] for report in selected)
            regimes[regime] = {"seeds": sorted(report["seed"] for report in selected),
                "replication_complete": complete, "direction_supported": bool(supported),
                "failed_seeds": sorted(report["seed"] for report in selected
                                       if not report["arms"][candidate]["direction_supported"])}
        results[candidate] = {"by_regime": regimes,
            "recommend_fresh_confirmation": all(item["direction_supported"] for item in regimes.values()),
            "recommend_scale_up": False, "admitted": False}
    return {"schema": "vpfp_development_direction_replication_v1", "objective": first["objective"],
        "baseline_arm": first["baseline_arm"], "status": "development_screen_not_independent_confirmation",
        "required_regimes": list(required_regimes), "minimum_distinct_seeds_per_regime": minimum_seeds,
        "candidates": results, "recommend_scale_up": False, "admitted": False}
