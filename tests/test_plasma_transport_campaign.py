"""Synthetic configuration tests; no real model or calibration claims."""
from copy import deepcopy
import json

import pytest

from scripts import plasma_gpu_campaign as campaign
from scripts import prepare_plasma_transport_v2 as migration
from tests.test_plasma_gpu_campaign import config
from tests.test_plasma_transport_scoring import _calibration


def upgraded(tmp_path, equation="strong", arm="sol"):
    old = config(equation, arm)
    path = tmp_path / f"{equation}-{arm}.json"
    path.write_text(json.dumps(old))
    return old, migration.transform(old, name=f"{equation}-{arm}", source_path=path,
                                    calibration=_calibration(), run_tag="unit-test-only")


@pytest.mark.parametrize("equation", ["strong", "weak", "artemis"])
@pytest.mark.parametrize("arm", ["sol", "astra"])
def test_six_arms_share_objective_without_changing_budgets_or_models(tmp_path, equation, arm):
    old, new = upgraded(tmp_path, equation, arm)
    assert new["schema"] == campaign.TRANSPORT_SCHEMA
    assert new["backend"] == old["backend"]
    assert new["execution"] == old["execution"]
    for name in ("official", "terminal", "development", "promotion", "master_seed", "round_count"):
        assert new[name] == old[name]
    recipe = campaign.recipe(new, object())
    assert recipe.generator.settings["metric_calibration"] == recipe.task_settings["metric_calibration"]
    fields = recipe.evaluator.default_query.fields
    assert len(fields) == (5 if equation == "artemis" else 10)
    assert all("900" in row["prompt"] and "NO separate preparation cap" in row["prompt"] for row in new["agents"])
    assert all("plasma_macro_transport_v2" in row["prompt"] for row in new["agents"])
    assert new["source_experiment"]["previous_guidance_imported"] is False
    assert "metric_profile" not in old["task"]


def test_legacy_per_host_limit_migrates_once(tmp_path):
    old = config("weak")
    old["protocol"].pop("parallel_evaluation")
    old["execution"]["oracle_max_workers"] = 1
    path = tmp_path / "weak.json"
    path.write_text(json.dumps(old))
    new = migration.transform(old, name="weak-sol", source_path=path,
                              calibration=_calibration(), run_tag="unit-test")
    assert new["execution"]["oracle_max_workers"] == 8
    assert old["execution"]["oracle_max_workers"] == 1


@pytest.mark.parametrize("kind", ["missing", "hash", "guide", "legacy"])
def test_mixed_or_implicit_objective_rejected(tmp_path, kind):
    _, new = upgraded(tmp_path)
    if kind == "missing":
        new["task"].pop("metric_calibration")
    elif kind == "hash":
        new["protocol"]["scoring_objective_hash"] = "0" * 64
    elif kind == "guide":
        new["generator"]["settings"].pop("metric_profile")
    else:
        new["schema"] = campaign.SCHEMA
    with pytest.raises(ValueError):
        campaign.validate_config(new)


def test_wrong_model_arm_does_not_generate_config(tmp_path):
    old = config("strong", "astra")
    path = tmp_path / "source.json"
    path.write_text(json.dumps(old))
    with pytest.raises(ValueError, match="model arm"):
        migration.transform(old, name="strong-sol", source_path=path,
                            calibration=_calibration(), run_tag="unit-test")


def test_migration_never_submits_or_imports_history():
    import inspect
    source = inspect.getsource(migration)
    assert "subprocess" not in source
    assert "sbatch" not in source
    assert "scancel" not in source
