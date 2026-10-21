"""Bounded recipe and admission-boundary tests; no actual assets or jobs."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
loader = importlib.util.spec_from_file_location("rho_next_under_test",ROOT/"scripts/run_vpfp_strong_rho_continuation.py")
driver = importlib.util.module_from_spec(loader)
loader.loader.exec_module(driver)


def test_exact_two_seed_recipe_and_fixed_steps(tmp_path):
    bound = {"last":{"path":str(tmp_path/"last.pt"),"sha256":"a"*64},
             "best":{"path":str(tmp_path/"best.pt"),"sha256":"b"*64}}
    for replica in (0,1):
        args = driver.training_arguments(tmp_path,replica,bound,"c"*40,3210.)
        assert (args.seed,args.batch_size,args.learning_rate,args.validate_every) == (
            1620260906+10000*replica,4,.0002,1000)
        assert (args.width,args.depth,args.steps,args.checkpoint_steps) == (256,4,40000,(30000,40000))
        assert args.resume_checkpoint == tmp_path/"last.pt"
        assert args.expected_resume_sha256 == "a"*64
        assert args.resume_best_checkpoint == tmp_path/"best.pt"
        assert args.train_seconds == 3210. and args.ablation_metadata["normalizer_refit"] is False


def fixture_audit(tmp_path,monkeypatch):
    unit_path = tmp_path/"unit.json"
    unit_path.write_text(json.dumps({"execution_passed":True}))
    slots = driver.completion.old.registered_slots()
    units = driver.completion.registered_units({"selected":slots})
    for unit in units:
        unit.update(status="completed",attempts=[{"status":"completed","report":str(unit_path),
            "sha256":driver.shared.file_sha(unit_path)}])
    history = tmp_path/"historical-plan.json"; history.write_text(json.dumps({"selected":slots}))
    historical_result = tmp_path/"historical-result.json"; historical_result.write_text("{}")
    history_sha = driver.shared.file_sha(history); result_sha = driver.shared.file_sha(historical_result)
    monkeypatch.setattr(driver.completion,"HISTORY_PLAN_SHA",history_sha)
    monkeypatch.setattr(driver.completion,"HISTORY_RESULT_SHA",result_sha)
    plan = {"historical_plan":{"path":str(history),"sha256":history_sha},
        "historical_result":{"path":str(historical_result),"sha256":result_sha},
        "selected":slots,"design":driver.completion.completion_design(),"files":{}}
    plan_path = tmp_path/"frozen-plan.json"; plan_path.write_text(json.dumps(plan))
    audit = {"schema":"vpfp_strong_initial_layer_completion_v1","execution_passed":True,
        "completed_carrier_count":40,"final_frozen_recheck_passed":True,"units":units,
        "historical_result_sha256":result_sha,"frozen_plan_sha256":driver.shared.file_sha(plan_path)}
    return audit,unit_path


@pytest.mark.parametrize("change",[
    {"execution_passed":False},{"completed_carrier_count":26},
    {"final_frozen_recheck_passed":False},{"units":[]},{"schema":"wrong"}])
def test_incomplete_or_different_audit_cannot_start_training(tmp_path,change,monkeypatch):
    audit,_ = fixture_audit(tmp_path,monkeypatch)
    audit.update(change)
    path = tmp_path/"result.json"; path.write_text(json.dumps(audit))
    with pytest.raises(ValueError,match="complete40"):
        driver.require_completed_audit(path,driver.shared.file_sha(path))


def test_successful_execution_is_not_scientific_admission(tmp_path,monkeypatch):
    audit,unit_path = fixture_audit(tmp_path,monkeypatch)
    audit["scientific_admission"] = False
    path = tmp_path/"result.json"; path.write_text(json.dumps(audit))
    assert driver.require_completed_audit(path,driver.shared.file_sha(path)) == audit
    unit_path.write_text(json.dumps({"execution_passed":False}))
    with pytest.raises(ValueError,match="changed|SHA mismatch"):
        driver.require_completed_audit(path,driver.shared.file_sha(path))


@pytest.mark.parametrize("change",["omitted","duplicate"])
def test_exact_unit_identity_not_completed_label(tmp_path,monkeypatch,change):
    audit,_ = fixture_audit(tmp_path,monkeypatch)
    assert len(audit["units"]) == 77
    if change == "omitted": audit["units"] = audit["units"][:1]
    else: audit["units"][-1] = audit["units"][0]
    path = tmp_path/"result.json"; path.write_text(json.dumps(audit))
    with pytest.raises(ValueError,match="exact77"):
        driver.require_completed_audit(path,driver.shared.file_sha(path))


def test_input_symlink_rejected(tmp_path):
    path = tmp_path/"original.json"; path.write_text("{}")
    link = tmp_path/"link.json"; link.symlink_to(path)
    with pytest.raises(ValueError,match="regular"):
        driver.checked_json(link,driver.shared.file_sha(path))
