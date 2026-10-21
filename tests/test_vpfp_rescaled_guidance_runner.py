"""Pure small runner contracts, not numerical/physical Guidance validation."""
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load(name, filename):
    path = Path(__file__).resolve().parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


effect = _load("vpfp_rescaled_effect_runner_test", "run_vpfp_rescaled_guidance_effect.py")
training = _load("vpfp_rescaled_training_runner_test", "run_vpfp_rescaled_guidance_training.py")


@pytest.fixture
def config(tmp_path):
    prior = []
    for index in range(2):
        path = tmp_path / f"prior-{index}.json"
        path.write_text(json.dumps({"synthetic_fixture": True, "index": index}))
        prior.append({"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()})
    return {
        "schema": "vpfp_rescaled_guidance_experiment_v1",
        "reference_effect_config": "configs/vpfp_calibrated_guidance_effect_20260908.json",
        "steps": 2000, "seed": 3120260908, "beta_std": .5,
        "calibration_seed": 7120260908, "validation_seed": 7220260908,
        "fresh_split_seeds": {"fit": 9320260908, "validation": 9420260908,
            "selection": 9520260908, "confirmation": 9620260908, "training": 9720260908},
        "bootstrap_seed": 9820260908, "bootstrap_replicates": 10000,
        "rescaling": {"target_ratio": .1, "gain_max": 1.e6, "max_flow_ratio": .25,
            "minimum_gradient_rms": 1.e-12, "minimum_base_drift_rms": 1.e-8,
            "random_seed": 9920260908,
            "fixed_gain_rule": "median_target_over_natural_flow_ratio_first_fit_root_per_stratum",
            "calibration_roots": 8, "arms": list(effect.ARMS)},
        "physics": {"light_probe_every_attempt": True,
            "local_audit_first_root_per_stratum": True, "hard_admission_gate": False},
        "prior_effects": prior,
    }


def test_fixed_eight_arm_rescaling_and_non_admission_protocol(config):
    original = deepcopy(config)
    assert effect.validate_rescaling_config(config) == original
    assert config == original
    assert len(effect.ARMS) == len(set(effect.ARMS)) == 8
    assert effect.PRIMARY == (("base", "exp_relative"), ("random_relative", "exp_relative"))


@pytest.mark.parametrize("change", ["schema", "steps", "beta", "ratio", "gain", "cap", "arms", "audit"])
def test_outcome_based_protocol_changes_rejected(config, change):
    if change == "schema": config["schema"] += "changed"
    if change == "steps": config["steps"] = 1999
    if change == "beta": config["beta_std"] = 1.
    if change == "ratio": config["rescaling"]["target_ratio"] = .03
    if change == "gain": config["rescaling"]["gain_max"] = 1.e4
    if change == "cap": config["rescaling"]["max_flow_ratio"] = 1.
    if change == "arms": config["rescaling"]["arms"].remove("random_relative")
    if change == "audit": config["physics"]["hard_admission_gate"] = True
    with pytest.raises(ValueError): effect.validate_rescaling_config(config)


def test_arm_order_balances_positions_and_does_not_fix_order_by_stratum():
    orders = [effect.arm_order(index) for index in range(32)]
    assert all(set(order) == set(effect.ARMS) and len(order) == 8 for order in orders)
    for position in range(8):
        assert Counter(order[position] for order in orders) == Counter({arm: 4 for arm in effect.ARMS})
    # Registered root index % 8 is the condition stratum. Merely rotating by
    # index % 8 would leave all four roots of a stratum with the same order.
    for stratum in range(8):
        assert len({orders[index] for index in range(stratum, 32, 8)}) == 4


@pytest.mark.parametrize("index", [-1, True, .5, "1"])
def test_invalid_root_order_index_rejected(index):
    with pytest.raises(ValueError): effect.arm_order(index)


def _roots():
    return [{"index": index, "stratum": index % 8, "root_seed": 1000+index,
             "conditions": [10.+index%8, 1.]} for index in range(32)]


def test_first_calibration_root_per_stratum_uses_registration_not_input_order():
    rows = _roots()[::-1]
    original = deepcopy(rows)
    selected = effect.first_per_stratum(rows)
    assert [row["index"] for row in selected] == list(range(8))
    assert rows == original
    assert effect.first_per_stratum(_roots()) == selected
    with pytest.raises(ValueError): effect.first_per_stratum([row for row in rows if row["stratum"] != 3])
    duplicate = _roots()
    duplicate[1]["root_seed"] = duplicate[0]["root_seed"]
    with pytest.raises(ValueError): effect.first_per_stratum(duplicate)


def _record(ratio, **kwargs):
    return {"skip_reason": None, "raw_gradient_rms": 1e-4, "base_drift_rms": 1.,
            "applied_flow_ratio": ratio, **kwargs}


def test_fixed_gain_is_median_of_per_call_gains_not_gain_of_median_ratio(config):
    records = [_record(.01), _record(.1)]
    original = deepcopy(records)
    result = effect.fixed_gain_from_records(records, config["rescaling"])
    assert result["fixed_gain"] == pytest.approx(5.5)
    assert result["fixed_gain"] != pytest.approx(.1/.055)
    assert result["active_calibration_calls"] == result["all_calibration_calls"] == 2
    assert result["solver_calls_for_calibration"] == 0
    assert not result["confirmation_outcomes_used"] and not result["empty_direction_fallback"]
    assert records == original


def test_fixed_gain_filters_numerically_empty_calls_and_enforces_registered_cap(config):
    records = [_record(1e-8), _record(0.), _record(None), _record(float("nan")),
        _record(float("inf")), _record(-1.), _record(.01, skip_reason="outside_sigma_window"),
        _record(.01, raw_gradient_rms=1e-30), _record(.01, base_drift_rms=0.)]
    result = effect.fixed_gain_from_records(records, config["rescaling"])
    assert result["active_calibration_calls"] == 1 and result["all_calibration_calls"] == len(records)
    assert result["uncapped_median_gain"] == pytest.approx(1e7)
    assert result["fixed_gain"] == 1e6 and result["gain_capped"]
    empty = effect.fixed_gain_from_records(records[1:], config["rescaling"])
    assert empty["fixed_gain"] == 1. and empty["empty_direction_fallback"]
    assert empty["natural_ratio_median"] is None and empty["active_calibration_calls"] == 0
    json.dumps(empty, allow_nan=False)


def test_comparison_pairs_preserve_missing_failed_attempts_and_are_detached():
    row = {"root_seed": 181, "stratum": 4,
        "base": {"status": "completed", "valid": True, "error": .1, "attempt_count": 1},
        "exp_relative": {"status": "failed", "valid": False, "error": None, "attempt_count": 1,
                         "failure": {"reason": "synthetic draw failure"}},
        "random_relative": {"status": "pending", "valid": False, "error": None, "attempt_count": 0}}
    original = deepcopy(row)
    pairs = effect.comparison_pairs([row], "base", "exp_relative")
    assert pairs == [{"root_seed": 181, "stratum": 4, "base": row["base"], "guided": row["exp_relative"]}]
    pairs[0]["guided"]["failure"]["reason"] = "changed copy"
    assert row == original
    pending = effect.comparison_pairs([row], "random_relative", "base")[0]
    assert pending["base"]["attempt_count"] == 0 and pending["base"]["error"] is None
    assert pending["guided"]["error"] == .1
    for left, right in (("base", "base"), ("unknown", "base")):
        with pytest.raises(ValueError): effect.comparison_pairs([row], left, right)


def test_interrupted_before_draws_still_has_incomplete_registered_root_summaries(config):
    rows = [{**root, **{arm: {"status": "pending", "valid": False,
        "error": None, "attempt_count": 0} for arm in effect.ARMS}} for root in _roots()]
    summaries = effect.summarize_comparisons(rows, config)
    assert summaries and sum(item["preregistered_primary"] for item in summaries.values()) == 2
    assert all(item["conclusion"] == "incomplete" and item["registered_roots"] == 32
        and item["all_attempts"] == 0 and item["paired"]["paired_valid_roots"] == 0
        for item in summaries.values())
    assert all(not item["automatic_candidate_admission"] for item in summaries.values())


def test_training_load_uses_fixed_recipe_and_real_sha_checks_only(config, tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    original = path.read_bytes()
    assert training.load_config(path) == config
    assert path.read_bytes() == original
    Path(config["prior_effects"][0]["path"]).write_text("modified synthetic evidence")
    with pytest.raises(ValueError, match="evidence changed"):
        training.load_config(path)


@pytest.mark.parametrize("change", ["steps", "float_seed", "split", "duplicate_history", "beta", "reference"])
def test_training_rejects_changed_recipe_roots_and_duplicate_history(config, tmp_path, change):
    if change == "steps": config["steps"] = 1999
    if change == "float_seed": config["seed"] = float(config["seed"])
    if change == "split": config["fresh_split_seeds"]["confirmation"] += 1
    if change == "duplicate_history": config["prior_effects"][1] = deepcopy(config["prior_effects"][0])
    if change == "beta": config["beta_std"] = 2.
    if change == "reference": config["reference_effect_config"] = "unregistered.json"
    path = tmp_path / "bad-config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError): training.load_config(path)


@pytest.mark.parametrize("runner", [effect, training])
def test_runner_refuses_login_without_reading_assets(monkeypatch, runner):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_TMPDIR", raising=False)
    # No args attributes exist: a late allocation check would already fail.
    with pytest.raises(ValueError, match="Slurm compute allocation"):
        runner.run(SimpleNamespace())
