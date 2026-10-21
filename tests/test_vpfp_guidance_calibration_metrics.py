"""Synthetic statistics contracts, not real Guidance validation evidence."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest


_PATH = Path(__file__).resolve().parents[1] / "scripts/vpfp_guidance_calibration_metrics.py"
_SPEC = importlib.util.spec_from_file_location("vpfp_guidance_calibration_metrics_test", _PATH)
metrics = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(metrics)


def _records():
    return [{"root_id": f"validation-{index:04d}", "stratum": index % 8,
             "sigma": sigma, "replicate": replicate, "target": .2+.03*index,
             "reward": .2+.03*index, "valid_probability": 1.,
             "displacement_nonzero": True, "finite": True}
            for index in range(16) for sigma in metrics.SIGMAS for replicate in (0, 1)]


def _summary(rows=None):
    result = metrics.summarize_predictions(_records() if rows is None else rows,
        fit_mean=.425, fit_stratum_means={index: .32+.03*index for index in range(8)})
    json.dumps(result, allow_nan=False)
    return result


def test_exact_inventory_perfect_continuous_regression_and_no_mutation():
    rows = _records()
    before = deepcopy(rows)
    result = _summary(rows)
    assert result["eligible"] and result["record_count"] == 160 and result["root_count"] == 16
    assert result["reward_mse"] == result["utility_mse"] == 0.
    assert result["reward_spearman"] == result["utility_spearman"] == 1.
    assert result["sigma1_effective_roots"] == 16
    assert result["within_stratum_concordance"]["comparable_pairs"] == 8
    assert result["within_stratum_concordance"]["reward"] == 1.
    assert all(row["record_count"] == 32 for row in result["per_sigma"].values())
    assert _summary(list(reversed(rows))) == result and rows == before
    assert not result["scientific_admission"] and not result["solver_effectiveness_established"]


def test_square_each_noisy_prediction_before_averaging_and_keep_utility_separate():
    rows = _records()
    for row in rows:
        row["reward"] += .05 if row["replicate"] else -.05
        row["valid_probability"] = .75
    result = _summary(rows)
    assert result["reward_mse"] == pytest.approx(.05**2)
    assert result["utility_mse"] > result["reward_mse"]
    assert result["reward_spearman"] == 1.
    assert all(row["reward_mse"] == pytest.approx(.05**2) for row in result["per_sigma"].values())


def test_average_tie_ranks_and_constant_null_spearman():
    assert metrics._spearman([1, 2, 3], [1, 1, 2]) == pytest.approx(3**.5/2)
    rows = _records()
    for row in rows:
        row["reward"] = .425
    result = _summary(rows)
    assert result["reward_spearman"] is None and result["utility_spearman"] is None
    assert not result["eligible"]
    assert result["within_stratum_concordance"]["reward"] == .5
    for row in rows:
        row["target"] = .425
    result = _summary(rows)
    assert result["within_stratum_concordance"]["comparable_pairs"] == 0
    assert result["within_stratum_concordance"]["reward"] is None


@pytest.mark.parametrize("damage", ["drop", "duplicate", "wrong_sigma", "root", "stratum", "target", "nan", "flag", "range", "replicate"])
def test_incomplete_or_invalid_observations_fail_closed(damage):
    rows = _records()
    if damage == "drop":
        rows.pop()
    elif damage == "duplicate":
        rows[1] = deepcopy(rows[0])
    elif damage == "wrong_sigma":
        rows[0]["sigma"] = 10.
    elif damage == "root":
        rows[0]["root_id"] = "fit-0000"
    elif damage == "stratum":
        rows[0]["stratum"] = 1
    elif damage == "target":
        rows[0]["target"] += .01
    elif damage == "nan":
        rows[0]["reward"] = float("nan")
    elif damage == "flag":
        rows[0]["finite"] = False
    elif damage == "range":
        rows[0]["valid_probability"] = 1.01
    else:
        rows[0]["replicate"] = True
    with pytest.raises(ValueError):
        _summary(rows)


def test_actual_displacement_requires_both_replicates_at_sigma_one():
    rows = _records()
    for row in rows:
        if row["root_id"] in metrics.ROOT_IDS[:4] and row["sigma"] == 1. and row["replicate"] == 0:
            row["displacement_nonzero"] = False
    result = _summary(rows)
    assert result["sigma1_effective_roots"] == 12 and result["eligible"]
    next(row for row in rows if row["root_id"] == metrics.ROOT_IDS[4] and row["sigma"] == 1.)["displacement_nonzero"] = False
    result = _summary(rows)
    assert result["sigma1_effective_roots"] == 11 and not result["eligible"]


def test_every_root_needs_mean_validity_strictly_above_half():
    rows = _records()
    for row in rows:
        if row["root_id"] == metrics.ROOT_IDS[0]:
            row["valid_probability"] = .5
    result = _summary(rows)
    assert not result["eligible"]
    assert "root_mean_valid_probability_not_above_half" in result["eligibility_reasons"]


def test_utility_ranking_must_be_positive_even_when_reward_regression_is_perfect():
    rows = _records()
    for row in rows:
        index = int(row["root_id"].rsplit("-", 1)[1])
        row["target"] = row["reward"] = .4+.005*index
        row["valid_probability"] = 1.-.025*index
    result = _summary(rows)
    assert result["reward_mse"] == 0. and result["reward_spearman"] == 1.
    assert result["utility_spearman"] == -1.
    assert all(root["mean_valid_probability"] > .5 for root in result["roots"])
    assert not result["eligible"] and "utility_spearman_not_positive" in result["eligibility_reasons"]


def test_stratum_baseline_is_diagnostic_not_an_extra_admission_gate():
    rows = _records()
    for row in rows:
        row["reward"] += .13
    result = _summary(rows)
    assert result["fit_stratum_mse"] < result["reward_mse"] < result["fit_mean_mse"]
    assert result["eligible"]


def _run(variant, seed, *, mse=.001, eligible=True):
    validation = _summary()
    validation.update(reward_mse=mse, eligible=eligible)
    return {"variant": variant, "seed": seed, "validation": validation,
            "checkpoint": {"path": f"/{variant}-{seed}.pt", "sha256": "a"*64}, "training_updates": 2000}


def test_selection_uses_two_seed_variant_mean_then_seed_score_and_never_control():
    rows = [_run("original", seed, mse=0.) for seed in (11, 22)]
    rows += [_run("scale", 11, mse=.001), _run("scale", 22, mse=.003)]
    rows += [_run("combined", 11, mse=.002), _run("combined", 22, mse=.001)]
    before = deepcopy(rows)
    result = metrics.select_variant(rows)
    assert result["selected_variant"] == "combined" and result["selected"]["seed"] == 22
    assert result["selected"]["checkpoint"] == rows[-1]["checkpoint"]
    assert rows == before
    result["selected"]["checkpoint"]["path"] = "changed"
    assert rows == before


def test_selection_ties_are_variant_name_then_seed_and_input_order_independent():
    rows = [_run(variant, seed) for variant in ("scale", "combined") for seed in (22, 11)]
    result = metrics.select_variant(rows)
    assert result["selected_variant"] == "combined" and result["selected"]["seed"] == 11
    assert metrics.select_variant(list(reversed(rows))) == result


@pytest.mark.parametrize("failure", ["missing", "ineligible", "failed"])
def test_missing_or_failed_second_seed_never_selects_variant(failure):
    rows = [_run("scale", 11)]
    if failure != "missing":
        rows.append(_run("scale", 22, eligible=False))
        if failure == "failed":
            rows[-1]["validation"] = None
    result = metrics.select_variant(rows)
    assert result["selected"] is None and not result["eligible_variants"]
    assert not result["variant_results"]["scale"]["eligible"]


def test_selection_rejects_duplicate_or_additional_seed_instead_of_cherry_picking():
    with pytest.raises(ValueError, match="duplicate"):
        metrics.select_variant([_run("scale", 11), _run("scale", 11)])
    with pytest.raises(ValueError, match="same two"):
        metrics.select_variant([_run("scale", 11), _run("scale", 22), _run("combined", 33)])
