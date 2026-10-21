"""Synthetic statistics contracts, not actual guidance-effect evidence."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


_PATH = Path(__file__).resolve().parents[1]/"scripts/vpfp_guidance_effect_summary.py"
_SPEC = importlib.util.spec_from_file_location("vpfp_guidance_effect_summary_test", _PATH)
summary = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(summary)


def _rows(delta=.1, base=1.):
    return [{"root_seed": 1000+i, "stratum": i % 8,
             "base": {"valid": True, "error": base, "status": "completed"},
             "guided": {"valid": True, "error": base+delta, "status": "completed"}}
            for i in range(32)]


def _summarize(rows, *, seed=12345, replicates=1000):
    result = summary.summarize_pairs(rows, bootstrap_seed=seed, bootstrap_replicates=replicates)
    json.dumps(result, allow_nan=False)
    return result


def test_positive_effect_requires_full_paired_yield_and_fixed_margin():
    result = _summarize(_rows())
    assert result["conclusion"] == "harder"
    assert result["all_32_pairs_valid"] and result["all_attempts"] == 64
    assert result["paired"]["mean_delta"] == pytest.approx(.1)
    assert result["paired"]["median_delta"] == pytest.approx(.1)
    assert result["paired"]["relative_gain"] == pytest.approx(.1)
    assert result["paired"]["wins"] == 32 and result["paired"]["ties"] == 0
    assert result["bootstrap"]["mean_delta_ci95"] == pytest.approx([.1, .1])
    assert result["bootstrap"]["relative_gain_ci95"] == pytest.approx([.1, .1])
    assert result["paired"]["mean_equivalence_band"] == pytest.approx(1e-5)
    assert all(row["paired_valid_roots"] == 4 for row in result["per_stratum"])
    assert not result["selection_or_refitting_performed"]
    assert not result["physical_regime_qualification"]


@pytest.mark.parametrize("delta", [0., 1e-10, -1e-10, 5e-7])
def test_zero_or_submargin_float_noise_is_not_difficulty_evidence(delta):
    result = _summarize(_rows(delta=delta, base=.03))
    assert result["conclusion"] == "not_demonstrated"
    assert result["paired"]["ties"] == 32
    assert result["paired"]["wins"] == result["paired"]["losses"] == 0
    assert result["paired"]["mean_equivalence_band"] == pytest.approx(1e-6)
    assert result["not_demonstrated_is_not_equivalence_proof"]
    if delta > 0:
        assert result["paired"]["raw_wins"] == 32


def test_band_has_preregistered_absolute_and_relative_branches():
    rows = _rows(delta=1e-6, base=100.)
    result = _summarize(rows)
    assert result["paired"]["mean_equivalence_band"] == pytest.approx(.001)
    assert result["conclusion"] == "not_demonstrated"
    assert result["decision"]["absolute_band"] == 1e-6
    assert result["decision"]["relative_band"] == 1e-5


@pytest.mark.parametrize("failure_status", ["failed", "completed"])
def test_invalid_guided_attempt_is_counted_without_becoming_zero_or_high_error(failure_status):
    rows = _rows(delta=.2)
    rows[0]["guided"] = {"valid": False, "error": float("nan"),
                         "status": failure_status, "attempt_count": 2}
    result = _summarize(rows)
    assert result["conclusion"] == "not_demonstrated"
    assert result["all_arms_terminal"] and not result["all_32_pairs_valid"]
    assert result["lost_valid_base_roots"] == 1
    assert result["all_attempts"] == 65
    assert result["arms"]["guided"]["failed_or_invalid_roots"] == 1
    assert result["arms"]["guided"]["valid_yield_per_attempt"] == pytest.approx(31/33)
    assert result["paired"]["paired_valid_roots"] == 31
    assert result["paired"]["mean_delta"] == pytest.approx(.2)
    assert not result["failed_or_missing_errors_imputed"]


def test_invalid_baseline_is_not_silently_counted_as_a_guidance_win():
    rows = _rows(delta=.2)
    rows[0]["base"] = {"valid": False, "error": {"type": "SolverFailed"}, "status": "failed"}
    result = _summarize(rows)
    assert result["conclusion"] == "not_demonstrated"
    assert result["paired"]["paired_valid_roots"] == 31
    assert result["paired"]["wins"] == 31
    assert result["arms"]["base"]["failed_or_invalid_roots"] == 1


@pytest.mark.parametrize("missing", ["root", "arm", "pending", "running", "interrupted"])
def test_missing_or_unresolved_work_is_incomplete_even_if_existing_effect_is_large(missing):
    rows = _rows(delta=10.)
    if missing == "root":
        rows.pop()
    elif missing == "arm":
        rows[-1].pop("guided")
    else:
        rows[-1]["guided"] = {"status": missing, "valid": False, "error": None}
    result = _summarize(rows)
    assert result["conclusion"] == "incomplete"
    assert not result["all_32_pairs_valid"] and not result["all_arms_terminal"]
    assert result["paired"]["paired_valid_roots"] == 31
    assert result["arms"]["guided"]["missing_or_unresolved_roots"] == 1
    assert result["arms"]["guided"]["failed_or_invalid_roots"] == 0
    assert result["lost_valid_base_roots"] == 0


def test_empty_inventory_reports_all_missing_without_nan_or_false_equivalence():
    result = _summarize([])
    assert result["conclusion"] == "incomplete"
    assert result["all_attempts"] == 0 and result["missing_root_slots"] == 32
    assert result["paired"]["mean_delta"] is None
    assert result["paired"]["win_rate"] is None
    assert result["bootstrap"]["mean_delta_ci95"] is None
    assert result["arms"]["base"]["missing_or_unresolved_roots"] == 32


def test_large_outlier_mean_does_not_masquerade_as_typical_pair_improvement():
    rows = _rows(delta=-.1)
    rows[0]["guided"]["error"] = 11.
    result = _summarize(rows, replicates=2000)
    assert result["paired"]["mean_delta"] > 0
    assert result["paired"]["median_delta"] == pytest.approx(-.1)
    assert result["paired"]["wins"] == 1
    assert result["paired"]["raw_win_rate"] == pytest.approx(1/32)
    assert result["bootstrap"]["mean_delta_ci95"][0] < 0
    assert result["conclusion"] == "not_demonstrated"


def test_stratified_bootstrap_resamples_whole_pairs_not_arms_or_condition_frequencies():
    rows = _rows()
    for index, row in enumerate(rows):
        row["base"]["error"] = 100.+10.*index
        row["guided"]["error"] = row["base"]["error"] + .2*(row["stratum"]+1)
    result = _summarize(rows)
    # Constant within-stratum paired deltas make stratified mean delta exactly
    # .9. Independent-arm or unstratified-root resampling would vary it.
    assert result["bootstrap"]["root_counts_by_stratum"] == [4]*8
    assert result["bootstrap"]["mean_delta_ci95"] == pytest.approx([.9, .9])
    assert result["paired"]["relative_gain"] == pytest.approx(.9/255.)


def test_bootstrap_is_deterministic_order_independent_and_does_not_mutate_inputs():
    rows = _rows()
    for index, row in enumerate(rows):
        row["guided"]["error"] += .03*index
    before = deepcopy(rows)
    first = _summarize(rows, seed=123)
    second = _summarize(list(reversed(rows)), seed=123)
    assert first == second and rows == before
    changed = _summarize(rows, seed=321)
    assert first["paired"] == changed["paired"]
    assert first["bootstrap"]["seed"] != changed["bootstrap"]["seed"]


def test_zero_baseline_has_defined_absolute_effect_but_no_invented_relative_gain():
    result = _summarize(_rows(delta=.01, base=0.))
    assert result["conclusion"] == "harder"
    assert result["paired"]["relative_gain"] is None
    assert result["bootstrap"]["relative_gain_ci95"] is None
    assert result["bootstrap"]["relative_gain_finite_replicates"] == 0


@pytest.mark.parametrize("bad", ["duplicate", "stratum", "fifth_in_stratum", "nonfinite_valid",
                                  "negative_error", "bool_error", "status", "attempt_count"])
def test_bad_receipts_fail_closed_before_scientific_summary(bad):
    rows = _rows()
    if bad == "duplicate": rows[1]["root_seed"] = rows[0]["root_seed"]
    elif bad == "stratum": rows[0]["stratum"] = 8
    elif bad == "fifth_in_stratum": rows[1]["stratum"] = 0
    elif bad == "nonfinite_valid": rows[0]["guided"]["error"] = np.inf
    elif bad == "negative_error": rows[0]["guided"]["error"] = -.1
    elif bad == "bool_error": rows[0]["guided"]["error"] = True
    elif bad == "status": rows[0]["guided"]["status"] = "maybe_finished"
    elif bad == "attempt_count": rows[0]["guided"]["attempt_count"] = 0
    with pytest.raises(ValueError):
        _summarize(rows)


def test_default_bootstrap_count_is_not_silently_reduced():
    result = summary.summarize_pairs(_rows(), bootstrap_seed=123)
    assert result["bootstrap"]["replicates"] == 10000
