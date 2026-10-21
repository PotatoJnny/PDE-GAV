"""Fixed development-set statistics for continuous VPFP reward regression.

This is an engineering screen, not physical admission or evidence that a
frozen solver became harder. No tensors, model loading, or solver calls.
"""

from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
import math
from numbers import Real


SIGMAS = (.002, .05, .2, 1., 2.)
ROOT_IDS = tuple(f"validation-{index:04d}" for index in range(16))
VARIANTS = ("original", "scale", "initialization", "combined")
SCHEMA = "vpfp_continuous_guidance_calibration_metrics_v1"


def _number(value, name, *, unit_interval=True):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    value = float(value)
    if not math.isfinite(value) or (unit_interval and not 0. <= value <= 1.):
        raise ValueError(f"{name} must be finite" + (" and in [0,1]" if unit_interval else ""))
    return value


def _mean(values):
    return math.fsum(values) / len(values)


def _ranks(values):
    """Average ranks for exact ties, without adding random jitter."""
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = (start + 1 + stop) / 2.
        for index in order[start:stop]:
            ranks[index] = rank
        start = stop
    return ranks


def _spearman(first, second):
    a, b = _ranks(first), _ranks(second)
    am, bm = _mean(a), _mean(b)
    aa = math.fsum((x-am)**2 for x in a)
    bb = math.fsum((x-bm)**2 for x in b)
    if aa == 0. or bb == 0.:
        return None
    value = math.fsum((x-am)*(y-bm) for x, y in zip(a, b)) / math.sqrt(aa*bb)
    return max(-1., min(1., value))


def _stratum_means(values):
    if isinstance(values, Mapping):
        if len(values) != 8:
            raise ValueError("fit_stratum_means requires all eight strata exactly once")
        result = {}
        for key, value in values.items():
            if type(key) is int and 0 <= key < 8:
                index = key
            elif isinstance(key, str) and key in tuple(str(i) for i in range(8)):
                index = int(key)
            else:
                raise ValueError("invalid fit_stratum_means key")
            if index in result:
                raise ValueError("duplicate fit_stratum_means stratum")
            result[index] = _number(value, "fit_stratum_mean")
        return result
    if not isinstance(values, (list, tuple)) or len(values) != 8:
        raise ValueError("fit_stratum_means requires eight fixed training-only means")
    return {index: _number(value, "fit_stratum_mean") for index, value in enumerate(values)}


def _root_means(records):
    grouped = defaultdict(list)
    for row in records:
        grouped[row["root_id"]].append(row)
    return [{"root_id": root, "stratum": rows[0]["stratum"], "target": rows[0]["target"],
             "mean_reward": _mean([r["reward"] for r in rows]),
             "mean_utility": _mean([r["reward"]*r["valid_probability"] for r in rows]),
             "mean_valid_probability": _mean([r["valid_probability"] for r in rows])}
            for root, rows in sorted(grouped.items())]


def _metrics(records, fit_mean, fit_stratum_means):
    roots = _root_means(records)
    targets = [r["target"] for r in roots]
    return {"record_count": len(records), "root_count": len(roots),
            # Square each independent noisy prediction BEFORE averaging.
            "reward_mse": _mean([(r["reward"]-r["target"])**2 for r in records]),
            "utility_mse": _mean([(r["reward"]*r["valid_probability"]-r["target"])**2 for r in records]),
            "fit_mean_mse": _mean([(fit_mean-r["target"])**2 for r in records]),
            "fit_stratum_mse": _mean([(fit_stratum_means[r["stratum"]]-r["target"])**2 for r in records]),
            "reward_spearman": _spearman(targets, [r["mean_reward"] for r in roots]),
            "utility_spearman": _spearman(targets, [r["mean_utility"] for r in roots])}


def _concordance(roots):
    pairs = []
    for stratum in range(8):
        first, second = [r for r in roots if r["stratum"] == stratum]
        target_delta = second["target"] - first["target"]
        row = {"stratum": stratum, "root_ids": [first["root_id"], second["root_id"]],
               "target_tie": target_delta == 0.}
        for name in ("reward", "utility"):
            delta = second["mean_"+name] - first["mean_"+name]
            row[name] = (None if target_delta == 0. else .5 if delta == 0.
                         else 1. if target_delta*delta > 0. else 0.)
        pairs.append(row)
    comparable = [r for r in pairs if not r["target_tie"]]
    return {"pairs": pairs, "comparable_pairs": len(comparable),
            "target_ties": 8-len(comparable),
            "reward": _mean([r["reward"] for r in comparable]) if comparable else None,
            "utility": _mean([r["utility"] for r in comparable]) if comparable else None,
            "tie_policy": "target ties excluded; prediction ties score one half",
            "diagnostic_only": True}


def summarize_predictions(records, *, fit_mean, fit_stratum_means):
    """Summarize exactly 16 roots x five sigmas x two noise replicates.

    Malformed, missing, duplicate, or nonfinite observations are rejected;
    callers retain that failure and must not substitute rows or score zeros.
    ``displacement_nonzero`` describes the actual finite FP32 denoiser-plus-
    guidance displacement, not a threshold on an abstract gradient norm.
    """
    fit_mean = _number(fit_mean, "fit_mean")
    stratum_means = _stratum_means(fit_stratum_means)
    if isinstance(records, (Mapping, str, bytes)):
        raise ValueError("records must be the fixed observation sequence")
    records = list(records)
    if len(records) != 160:
        raise ValueError("exactly 160 observations are required; failed rows cannot be dropped")
    checked, seen, identity = [], set(), {}
    for index, source in enumerate(records):
        if not isinstance(source, Mapping):
            raise ValueError("each observation must be a mapping")
        required = {"root_id", "stratum", "sigma", "replicate", "target", "reward",
                    "valid_probability", "displacement_nonzero", "finite"}
        if not required.issubset(source):
            raise ValueError("observation fields missing")
        root, stratum, replicate = source["root_id"], source["stratum"], source["replicate"]
        if not isinstance(root, str) or root not in ROOT_IDS:
            raise ValueError("only the fixed validation-0000 through validation-0015 roots are allowed")
        if type(stratum) is not int or not 0 <= stratum < 8:
            raise ValueError("stratum must be an integer in 0..7")
        if type(replicate) is not int or replicate not in (0, 1):
            raise ValueError("replicate must be integer zero or one")
        sigma = _number(source["sigma"], "sigma", unit_interval=False)
        if sigma not in SIGMAS:
            raise ValueError("observation sigma is outside the fixed guidance window probes")
        key = root, sigma, replicate
        if key in seen:
            raise ValueError("duplicate root/sigma/replicate observation")
        seen.add(key)
        if type(source["finite"]) is not bool or source["finite"] is not True:
            raise ValueError(f"observation {index} is not explicitly finite; no substituted metrics")
        if type(source["displacement_nonzero"]) is not bool:
            raise ValueError("displacement_nonzero must be boolean")
        row = {"root_id": root, "stratum": stratum, "sigma": sigma, "replicate": replicate,
               "displacement_nonzero": source["displacement_nonzero"], "finite": True}
        row.update({name: _number(source[name], name) for name in ("target", "reward", "valid_probability")})
        current = stratum, row["target"]
        if root in identity and identity[root] != current:
            raise ValueError("a physical root changed target or stratum across noise observations")
        identity[root] = current
        checked.append(row)
    expected = {(root, sigma, replicate) for root in ROOT_IDS for sigma in SIGMAS for replicate in (0, 1)}
    if seen != expected or any(sum(s == index for s, _ in identity.values()) != 2 for index in range(8)):
        raise ValueError("fixed root/noise inventory and two independent roots per stratum required")
    checked.sort(key=lambda r: (r["root_id"], r["sigma"], r["replicate"]))
    roots = _root_means(checked)
    for root in roots:
        root["sigma1_both_displacements_nonzero"] = all(
            r["displacement_nonzero"] for r in checked if r["root_id"] == root["root_id"] and r["sigma"] == 1.)
    result = _metrics(checked, fit_mean, stratum_means)
    effective = sum(r["sigma1_both_displacements_nonzero"] for r in roots)
    reasons = []
    if not result["reward_mse"] < result["fit_mean_mse"]:
        reasons.append("reward_mse_not_better_than_training_mean_baseline")
    for name in ("reward_spearman", "utility_spearman"):
        if result[name] is None or result[name] <= 0.:
            reasons.append(name+"_not_positive")
    if effective < 12:
        reasons.append("fewer_than_12_sigma1_effective_roots")
    if any(r["mean_valid_probability"] <= .5 for r in roots):
        reasons.append("root_mean_valid_probability_not_above_half")
    result.update(schema=SCHEMA, eligible=not reasons, eligibility_reasons=reasons,
                  all_records_finite=True, sigma1_effective_roots=effective, roots=roots,
                  fit_mean=fit_mean, fit_stratum_means={str(k): v for k, v in stratum_means.items()},
                  per_sigma={format(sigma, "g"): _metrics([r for r in checked if r["sigma"] == sigma],
                             fit_mean, stratum_means) for sigma in SIGMAS},
                  within_stratum_concordance=_concordance(roots),
                  mse_aggregation="mean of squared errors over roots, fixed sigmas, and noise replicates",
                  rank_aggregation="Spearman of per-root mean predictions; exact average ranks for ties",
                  gate_scope="preregistered development engineering screen; not physical admission or solver-effect evidence",
                  scientific_admission=False, solver_effectiveness_established=False)
    return result


def select_variant(results):
    """Select from complete eligible two-seed variants; original is control only.

    Missing/failed validation summaries keep their original run ineligible.
    Solver outcomes are neither accepted as a score nor inspected here.
    """
    if not isinstance(results, (list, tuple)):
        raise ValueError("results must be a sequence of planned variant/seed runs")
    grouped = {name: [] for name in VARIANTS}
    seen, seeds = set(), set()
    for row in results:
        if not isinstance(row, Mapping) or row.get("variant") not in VARIANTS or type(row.get("seed")) is not int:
            raise ValueError("each result needs a registered variant and integer seed")
        key = row["variant"], row["seed"]
        if key in seen:
            raise ValueError("duplicate variant/seed result; no repeated seed selection")
        seen.add(key)
        seeds.add(row["seed"])
        validation = row.get("validation")
        if validation is not None:
            if not isinstance(validation, Mapping) or type(validation.get("eligible")) is not bool:
                raise ValueError("validation summary requires explicit engineering eligibility")
            _number(validation.get("reward_mse"), "validation reward_mse")
        grouped[row["variant"]].append(row)
    if len(seeds) > 2:
        raise ValueError("all variants must use the same two preregistered seeds")
    inventory, candidates = {}, []
    for variant in VARIANTS:
        rows = sorted(grouped[variant], key=lambda r: r["seed"])
        reasons = []
        if len(rows) != 2:
            reasons.append("missing_two_seed_results")
        if any(r.get("validation") is None for r in rows):
            reasons.append("missing_or_failed_validation_summary")
        if any(r.get("validation") is not None and not r["validation"]["eligible"] for r in rows):
            reasons.append("one_or_more_seeds_failed_engineering_screen")
        mean_mse = (_mean([float(r["validation"]["reward_mse"]) for r in rows])
                    if len(rows) == 2 and all(r.get("validation") is not None for r in rows) else None)
        if variant == "original":
            reasons.append("original_is_control_only")
        inventory[variant] = {"seeds": [r["seed"] for r in rows], "run_count": len(rows),
                              "mean_reward_mse": mean_mse, "eligible": not reasons, "reasons": reasons}
        if not reasons:
            candidates.append((mean_mse, variant, rows))
    selected, variant = None, None
    if candidates:
        _, variant, rows = min(candidates, key=lambda item: (item[0], item[1]))
        selected = deepcopy(min(rows, key=lambda r: (r["validation"]["reward_mse"], r["seed"])))
    return {"selected": selected, "selected_variant": variant,
            "eligible_variants": sorted(item[1] for item in candidates), "variant_results": inventory,
            "reason": ("lowest_two_seed_mean_reward_mse_then_lower_development_mse_seed" if selected is not None
                       else "no_complete_two_seed_engineering_eligible_candidate"),
            "selection_scope": "development labels only; independent fixed-solver confirmation still required",
            "scientific_admission": False, "solver_results_used": False}
