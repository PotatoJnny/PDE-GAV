"""Paired held-out guidance effects, separate from execution/fidelity claims.

The fixed design is 32 independent roots, four in each of eight condition
strata. No optimizer, model loading, sampling, filesystem IO or selection is
performed here. Failed and missing attempts never receive fabricated errors.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import math

import numpy as np


SCHEMA = "vpfp_fixed_guidance_paired_effect_v1"
EXPECTED_ROOTS = 32
STRATUM_COUNT = 8
ROOTS_PER_STRATUM = 4
ABSOLUTE_EQUIVALENCE_BAND = 1e-6
RELATIVE_EQUIVALENCE_BAND = 1e-5
_STATUSES = frozenset(("pending", "running", "interrupted", "failed", "completed"))
_TERMINAL = frozenset(("failed", "completed"))


def _integer(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _arm(value):
    if value is None:
        return {"status": "missing", "valid": False, "error": None,
                "attempt_count": 0, "terminal": False, "usable": False}
    if not isinstance(value, Mapping):
        raise ValueError("each arm must be a mapping or missing")
    status = value.get("status", "pending")
    if status not in _STATUSES:
        raise ValueError("unknown arm status")
    valid = value.get("valid", False)
    if type(valid) is not bool:
        raise ValueError("arm validity must be an explicit boolean")
    attempts = _integer(value.get("attempt_count", 0 if status == "pending" else 1),
                        "attempt_count")
    if (status == "pending" and attempts != 0) or (status != "pending" and attempts < 1):
        raise ValueError("pending arms have zero attempts; started arms need an attempt")
    error = None
    if valid:
        raw = value.get("error")
        if (isinstance(raw, (bool, np.bool_)) or not isinstance(raw, (int, float, np.number))
                or np.iscomplexobj(raw) or not math.isfinite(float(raw)) or float(raw) < 0):
            raise ValueError("valid arm error must be finite, real and nonnegative")
        if status == "failed":
            raise ValueError("a failed arm cannot be declared valid")
        error = float(raw)
    return {"status": status, "valid": valid, "error": error,
            "attempt_count": attempts, "terminal": status in _TERMINAL,
            "usable": status == "completed" and valid}


def _mean(values):
    if not values:
        return None
    # Divide before summing to avoid an unnecessary sum overflow for large
    # but finite nonnegative error values.
    return math.fsum(float(value)/len(values) for value in values)


def _finite_ratio(numerator, denominator):
    if denominator is None or denominator <= 0:
        return None
    value = numerator/denominator
    return value if math.isfinite(value) else None


def _pair_statistics(pairs):
    base = [row["base"]["error"] for row in pairs]
    guided = [row["guided"]["error"] for row in pairs]
    delta = [g-b for b, g in zip(base, guided)]
    band = [max(ABSOLUTE_EQUIVALENCE_BAND, RELATIVE_EQUIVALENCE_BAND*abs(b)) for b in base]
    count = len(pairs)
    mean_delta, base_mean = _mean(delta), _mean(base)
    raw_wins = sum(d > 0 for d in delta)
    raw_ties = sum(d == 0 for d in delta)
    wins = sum(d > tolerance for d, tolerance in zip(delta, band))
    losses = sum(d < -tolerance for d, tolerance in zip(delta, band))
    ties = count-wins-losses
    return {
        "paired_valid_roots": count,
        "base_mean_error": base_mean,
        "guided_mean_error": _mean(guided),
        "mean_delta": mean_delta,
        "median_delta": float(np.median(delta)) if count else None,
        "relative_gain": _finite_ratio(mean_delta, base_mean) if count else None,
        "relative_gain_definition": "paired_mean_delta / paired_base_mean_error; not mean of per-root ratios",
        "mean_equivalence_band": _mean(band),
        "wins": wins, "ties": ties, "losses": losses,
        "win_rate": wins/count if count else None,
        "tie_rate": ties/count if count else None,
        "win_tie_definition": "delta greater than band; absolute(delta) <= band; delta less than negative band",
        "raw_wins": raw_wins, "raw_ties": raw_ties,
        "raw_losses": count-raw_wins-raw_ties,
        "raw_win_rate": raw_wins/count if count else None,
        "raw_tie_rate": raw_ties/count if count else None,
    }


def _bootstrap(pairs, *, seed, replicates):
    grouped = [[row for row in pairs if row["stratum"] == s] for s in range(STRATUM_COUNT)]
    result = {
        "seed": seed, "replicates": replicates, "confidence_level": .95,
        "method": "percentile_stratified_paired_whole_root_bootstrap",
        "statistical_unit": "one independent root containing both paired arms",
        "root_counts_by_stratum": [len(group) for group in grouped],
        "scope": "paired valid subset; failures/missing roots are not imputed or resampled as valid",
        "mean_delta_ci95": None, "relative_gain_ci95": None,
        "relative_gain_finite_replicates": 0,
    }
    if not pairs:
        return result
    rng = np.random.default_rng(seed)
    base_mean = np.zeros(replicates, dtype=np.float64)
    delta_mean = np.zeros(replicates, dtype=np.float64)
    count = len(pairs)
    for group in grouped:
        if not group:
            continue
        base = np.asarray([row["base"]["error"] for row in group])
        delta = np.asarray([row["guided"]["error"]-row["base"]["error"] for row in group])
        # One index matrix selects both arms together; all roots of a stratum
        # remain together as pairs and every replicate retains its root count.
        index = rng.integers(0, len(group), size=(replicates, len(group)))
        base_mean += np.sum(base[index]/count, axis=1)
        delta_mean += np.sum(delta[index]/count, axis=1)
    if not np.isfinite(delta_mean).all() or not np.isfinite(base_mean).all():
        raise ValueError("bootstrap arithmetic produced a nonfinite mean")
    result["mean_delta_ci95"] = [float(x) for x in np.quantile(delta_mean, [.025, .975])]
    relative = np.full(replicates, np.nan)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        np.divide(delta_mean, base_mean, out=relative, where=base_mean > 0)
    finite = np.isfinite(relative)
    result["relative_gain_finite_replicates"] = int(finite.sum())
    if finite.all():
        result["relative_gain_ci95"] = [float(x) for x in np.quantile(relative, [.025, .975])]
    else:
        result["relative_gain_ci_unavailable_reason"] = "some resampled baseline means are zero or ratios are nonfinite; undefined replicates were not silently dropped"
    return result


def summarize_pairs(rows, *, bootstrap_seed, bootstrap_replicates=10000):
    """Summarize one preregistered 32-root baseline/guided confirmation.

    Rows contain root_seed, stratum and base/guided mappings with status,
    valid, error and optional attempt_count. A missing arm may be None or
    absent. ``incomplete`` means unresolved planned slots; completed invalid
    trials are ``not_demonstrated``, never synthetic zero or high rewards.
    No result here establishes physical-regime validity or distribution
    equivalence. The caller owns asset, seed/initial-state pairing and query
    protocol verification before calling this pure statistical function.
    """
    seed = _integer(bootstrap_seed, "bootstrap_seed")
    replicates = _integer(bootstrap_replicates, "bootstrap_replicates", minimum=1)
    if replicates > 1_000_000:
        raise ValueError("bootstrap_replicates exceeds the explicit memory bound")
    if isinstance(rows, (str, bytes, Mapping)):
        raise ValueError("rows must be a sequence of paired root records")
    prepared = []
    seen = set()
    strata = Counter()
    for item in rows:
        if not isinstance(item, Mapping):
            raise ValueError("root row must be a mapping")
        root = _integer(item.get("root_seed"), "root_seed")
        stratum = _integer(item.get("stratum"), "stratum")
        if stratum >= STRATUM_COUNT or root in seen:
            raise ValueError("duplicate root or invalid stratum")
        seen.add(root)
        strata[stratum] += 1
        if strata[stratum] > ROOTS_PER_STRATUM:
            raise ValueError("registered design allows exactly four roots per stratum")
        prepared.append({"root_seed": root, "stratum": stratum,
                         "base": _arm(item.get("base")), "guided": _arm(item.get("guided"))})
    if len(prepared) > EXPECTED_ROOTS:
        raise ValueError("confirmation permits exactly 32 independent roots")
    # Stable root order makes bootstrap output independent of receipt loading
    # order, not just deterministic for one caller's dictionary iteration.
    prepared.sort(key=lambda row: (row["stratum"], row["root_seed"]))
    pairs = [row for row in prepared if row["base"]["usable"] and row["guided"]["usable"]]
    missing_roots = EXPECTED_ROOTS-len(prepared)
    arms = {}
    for name in ("base", "guided"):
        arm_rows = [row[name] for row in prepared]
        status_counts = Counter(row["status"] for row in arm_rows)
        status_counts["missing"] += missing_roots
        arms[name] = {
            "attempt_count": sum(row["attempt_count"] for row in arm_rows),
            "attempted_root_slots": sum(row["attempt_count"] > 0 for row in arm_rows),
            "valid_roots": sum(row["usable"] for row in arm_rows),
            "terminal_roots": sum(row["terminal"] for row in arm_rows),
            "failed_or_invalid_roots": sum(row["terminal"] and not row["usable"] for row in arm_rows),
            "missing_or_unresolved_roots": missing_roots+sum(not row["terminal"] for row in arm_rows),
            "status_counts": {status: status_counts[status] for status in ("missing", "pending", "running", "interrupted", "failed", "completed")},
            "valid_yield_per_attempt": (sum(row["usable"] for row in arm_rows)/sum(row["attempt_count"] for row in arm_rows)
                                        if sum(row["attempt_count"] for row in arm_rows) else None),
        }
    lost = [row["root_seed"] for row in prepared
            if row["base"]["usable"] and row["guided"]["terminal"] and not row["guided"]["usable"]]
    unpaired_base = [row["root_seed"] for row in prepared
                     if row["base"]["usable"] and not row["guided"]["usable"]]
    inventory_complete = len(prepared) == EXPECTED_ROOTS and all(strata[s] == ROOTS_PER_STRATUM for s in range(STRATUM_COUNT))
    all_terminal = inventory_complete and all(row[arm]["terminal"] for row in prepared for arm in ("base", "guided"))
    all_valid = all_terminal and len(pairs) == EXPECTED_ROOTS and not lost
    statistics = _pair_statistics(pairs)
    bootstrap = _bootstrap(pairs, seed=seed, replicates=replicates)
    lower = bootstrap["mean_delta_ci95"][0] if bootstrap["mean_delta_ci95"] is not None else None
    exceeds_band = (lower is not None and statistics["mean_equivalence_band"] is not None
                    and lower > statistics["mean_equivalence_band"])
    if not all_terminal:
        conclusion, reason = "incomplete", "at least one preregistered root/arm is missing or unresolved"
    elif not all_valid:
        conclusion, reason = "not_demonstrated", "the full 32-root paired-valid yield condition was not met"
    elif exceeds_band:
        conclusion, reason = "harder", "paired mean-error bootstrap lower bound exceeds the preregistered mean numerical-equivalence band"
    else:
        conclusion, reason = "not_demonstrated", "paired mean-error lower bound does not exceed the fixed numerical-equivalence band"
    per_stratum = []
    for stratum in range(STRATUM_COUNT):
        group = [row for row in prepared if row["stratum"] == stratum]
        paired = [row for row in pairs if row["stratum"] == stratum]
        per_stratum.append({"stratum": stratum, "expected_roots": ROOTS_PER_STRATUM,
            "registered_roots": len(group),
            "base_valid_roots": sum(row["base"]["usable"] for row in group),
            "guided_valid_roots": sum(row["guided"]["usable"] for row in group),
            **_pair_statistics(paired)})
    return {
        "schema": SCHEMA, "conclusion": conclusion, "reason": reason,
        "expected_roots": EXPECTED_ROOTS, "registered_roots": len(prepared),
        "missing_root_slots": missing_roots, "expected_arm_slots": 2*EXPECTED_ROOTS,
        "all_attempts": sum(arm["attempt_count"] for arm in arms.values()),
        "attempt_count_semantics": "sum of supplied arm attempt_count; repeated attempts are not additional independent roots",
        "arms": arms, "root_inventory_complete": inventory_complete,
        "all_arms_terminal": all_terminal, "all_32_pairs_valid": all_valid,
        "lost_valid_base_roots": len(lost), "lost_valid_base_root_seeds": lost,
        "unpaired_valid_base_roots": len(unpaired_base),
        "unpaired_valid_base_root_seeds": unpaired_base,
        "paired": statistics, "bootstrap": bootstrap, "per_stratum": per_stratum,
        "decision": {"requires_all_32_pairs_valid": True,
            "requires_no_lost_valid_base_roots": True,
            "mean_delta_ci_lower_exceeds_mean_band": bool(exceeds_band),
            "fixed_per_root_band": "max(1e-6, 1e-5*abs(base_error))",
            "absolute_band": ABSOLUTE_EQUIVALENCE_BAND,
            "relative_band": RELATIVE_EQUIVALENCE_BAND,
            "strict_comparison": "lower_bound > mean_band"},
        "effect_scope": "actual error of this frozen solver under the caller-verified fixed public-input protocol",
        "selection_or_refitting_performed": False,
        "physical_regime_qualification": False,
        "distribution_matching_required": False,
        "not_demonstrated_is_not_equivalence_proof": True,
        "failed_or_missing_errors_imputed": False,
    }


__all__ = ["summarize_pairs"]
