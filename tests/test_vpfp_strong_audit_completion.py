"""Synthetic completion/resume receipts, not real physical validation."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("test_strong_completion_driver",ROOT/"scripts/run_vpfp_strong_audit_completion.py")
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)


def tiny_plan():
    selected=[];arrays={}
    for i,slot in enumerate(driver.old.registered_slots()):
        row,values=driver.old._input_row(slot["group"],i,np.array([i,2.,3.],dtype=np.float32),
            np.array([20.,1.],dtype=np.float32),"synthetic-root-"+str(i),i)
        row.update(id=slot["id"],stratum_id=slot["stratum_id"])
        selected.append(row);arrays[row["id"]]=values
    plan={"schema":driver.old.SCHEMA,"selected":selected,"files":{},"design":driver.old.design(),
        "refinement_selected":selected[7],"same_seed_models_paired":True,"source_manifest_sha256":driver.old.SOURCE_SHA,
        "checkpoint_candidates":driver.old.candidates(),"generation_designs":{"synthetic":True}}
    return plan,arrays


def test_exact_denominators_stages_and_new_time_protocol():
    plan,_=tiny_plan();units=driver.registered_units(plan);design=driver.completion_design()
    assert len(units)==77 and len({u["id"] for u in units})==77
    assert sum(u["stage"]=="remaining" for u in units)==28
    assert sum(u["stage"]=="temporal" for u in units)==24
    assert sum(u["stage"]=="t0" for u in units)==24
    assert units[0]["stage"]=="refinement"
    assert {u["carrier_id"] for u in units if u["stage"]=="remaining"}=={r["id"] for r in plan["selected"][26:]}
    assert design["extended_time_counts"]==[257,513] and design["t0_relative_fd_steps"]==[2.**-9,2.**-10]
    assert not design["automatic_downsampling"] and "not40_independent_roots" in design["statistical_unit"]


@pytest.fixture
def fake_run(tmp_path,monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID","12345");monkeypatch.setenv("SLURM_TMPDIR",str(tmp_path))
    monkeypatch.setattr(driver.old,"SCRATCH",tmp_path)
    original,arrays=tiny_plan();events=[];domain={"fixture":True}
    plan={"schema":driver.SCHEMA,"selected":original["selected"],"current_input_plan":original,"files":{},
        "historical_slots":driver.old.registered_slots(),"historical_refinement_failure":{"status":"failed","reason":"old domain guard"},
        "domain_coordinate_receipt":domain,"design":driver.completion_design()}
    def cheap(): events.append("cheap_domain");return domain
    def freeze(check): check();events.append("freeze40_plus_old52reports");return deepcopy(plan),arrays,{},object()
    def runtime(*args):
        events.append("runtime_after_freeze")
        return SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(hermite_total_degree=14)),frozen_files={})
    def measure(kind,raw): events.append((kind,int(raw[0])));return {"execution_passed":True,"fixture":True,"numerically_uncertain":True}
    def refine(*args): events.append("refinement");return {"execution_passed":True,"new_pde_solves":2,"fixture":True}
    monkeypatch.setattr(driver,"cheap_domain_preflight",cheap)
    monkeypatch.setattr(driver,"freeze_completion",freeze)
    monkeypatch.setattr(driver.old.existing,"_runtime",runtime)
    monkeypatch.setattr(driver.old.existing,"check_frozen",lambda _:events.append("check_frozen"))
    monkeypatch.setattr(driver.old,"_tensors",lambda model,values:values)
    monkeypatch.setattr(driver.old,"_independent",lambda model,raw,theta,d:measure("independent",raw))
    monkeypatch.setattr(driver.old,"native_temporal_frequency",lambda model,raw,theta,**k:measure("old_temporal",raw))
    monkeypatch.setattr(driver,"extended_temporal",lambda model,raw,theta,**k:measure("extended_temporal",raw))
    monkeypatch.setattr(driver,"t0_balances",lambda model,raw,theta:measure("t0",raw))
    monkeypatch.setattr(driver,"_refine",refine)
    return SimpleNamespace(output=tmp_path/"out",events=events,plan=plan,arrays=arrays)


def test_all_reuses26_only_measures14_then_fixed24_and_never_claims_admission(fake_run):
    result=driver.run(fake_run.output,device="cpu",wall_seconds=20000)
    assert result["execution_passed"] and result["requested_stage_execution_passed"]
    assert result["completed_carrier_count"]==40 and len(result["units"])==77
    assert [e[1] for e in fake_run.events if isinstance(e,tuple) and e[0]=="independent"]==list(range(26,40))
    assert [e[1] for e in fake_run.events if isinstance(e,tuple) and e[0]=="extended_temporal"]==list(range(24))
    assert [e[1] for e in fake_run.events if isinstance(e,tuple) and e[0]=="t0"]==list(range(24))
    assert fake_run.events.index("cheap_domain")<fake_run.events.index("freeze40_plus_old52reports")<fake_run.events.index("runtime_after_freeze")
    assert fake_run.events.index("refinement")<fake_run.events.index(("independent",26))
    assert not result["admitted"] and result["numerically_uncertain"] and not result["confirmation_512_completed"]
    assert result["historical_terminal"]["exit_code"]=="124" and result["historical_refinement_failure"]["status"]=="failed"


def test_stage_resume_skips_complete_and_keeps_report_bytes(fake_run):
    first=driver.run(fake_run.output,device="cpu",stage="remaining",wall_seconds=20000)
    assert first["requested_stage_execution_passed"] and not first["execution_passed"]
    assert first["completed_carrier_count"]==40
    frozen={p:p.read_bytes() for p in fake_run.output.glob("units/**/attempt-*.json")}
    fake_run.events.clear()
    second=driver.run(fake_run.output,device="cpu",resume=True,wall_seconds=20000)
    assert second["execution_passed"] and len(second["invocations"])==2
    assert not any(isinstance(e,tuple) and e[0] in ("independent","old_temporal") for e in fake_run.events)
    assert all(p.read_bytes()==data for p,data in frozen.items())


def test_deadline_records_unfinished_unit_then_resume_only_that_and_later(fake_run,monkeypatch):
    action=driver.old._independent
    def stop(model,raw,theta,d):
        if raw[0]==28: raise TimeoutError("synthetic allocation deadline")
        return action(model,raw,theta,d)
    monkeypatch.setattr(driver.old,"_independent",stop)
    result=driver.run(fake_run.output,device="cpu",wall_seconds=20000)
    assert not result["execution_passed"] and result["completed_carrier_count"]==28
    assert len(result["units"])==77 and result["failure"]["type"]=="TimeoutError"
    monkeypatch.setattr(driver.old,"_independent",action);fake_run.events.clear()
    resumed=driver.run(fake_run.output,device="cpu",resume=True,wall_seconds=20000)
    assert resumed["execution_passed"]
    assert [e[1] for e in fake_run.events if isinstance(e,tuple) and e[0]=="independent"]==list(range(28,40))
    unit=next(u for u in resumed["units"] if u.get("carrier_id")==fake_run.plan["selected"][28]["id"] and u["kind"]=="independent_physics")
    assert [a["status"] for a in unit["attempts"]]==["incomplete","completed"]
    assert resumed["failures"] # successful resumption does not delete prior failure traces


def test_failed_measurement_not_retried_without_explicit_flag(fake_run,monkeypatch):
    action=driver.old._independent
    monkeypatch.setattr(driver.old,"_independent",lambda m,r,t,d:{"execution_passed":False} if r[0]==26 else action(m,r,t,d))
    first=driver.run(fake_run.output,device="cpu",stage="remaining",wall_seconds=20000)
    assert not first["requested_stage_execution_passed"]
    monkeypatch.setattr(driver.old,"_independent",action);fake_run.events.clear()
    second=driver.run(fake_run.output,device="cpu",stage="remaining",resume=True,wall_seconds=20000)
    assert not second["requested_stage_execution_passed"] and not any(isinstance(e,tuple) for e in fake_run.events)
    third=driver.run(fake_run.output,device="cpu",stage="remaining",resume=True,retry_failed=True,wall_seconds=20000)
    assert third["requested_stage_execution_passed"] and third["failures"]


def test_resume_changed_plan_or_report_rejected_without_erasing_progress(fake_run,monkeypatch):
    driver.run(fake_run.output,device="cpu",stage="remaining",wall_seconds=20000)
    path=fake_run.output/"result.json";original=path.read_bytes()
    fake_run.plan["design"]["extended_time_counts"]=[129,257]
    result=driver.run(fake_run.output,device="cpu",resume=True,wall_seconds=20000)
    assert not result["execution_passed"] and "changed" in result["failure"]["reason"]
    assert path.read_bytes()==original and list(fake_run.output.glob("resume-preflight-*.json"))


def test_resume_changed_complete_attempt_sha_is_not_reused(fake_run):
    first=driver.run(fake_run.output,device="cpu",stage="remaining",wall_seconds=20000)
    entry=next(u for u in first["units"] if u["status"]=="completed")["attempts"][-1]
    Path(entry["report"]).write_text('{"execution_passed":true,"changed":true}')
    original=(fake_run.output/"result.json").read_bytes()
    result=driver.run(fake_run.output,device="cpu",resume=True,wall_seconds=20000)
    assert "attempt report changed" in result["failure"]["reason"]
    assert (fake_run.output/"result.json").read_bytes()==original


def test_cheap_preflight_no_runtime_no_freeze_and_no_false_scientific_pass(fake_run):
    report=driver.run(fake_run.output,device="cpu",preflight_only=True)
    assert fake_run.events==["cheap_domain"]
    assert report["requested_stage_execution_passed"] and not report["execution_passed"]
    assert not report["full_input_freeze_completed"] and not report["physics_executed"]
    assert len(report["units"])==77


def test_failed_domain_preflight_keeps_denominators_and_never_constructs_runtime(fake_run,monkeypatch):
    monkeypatch.setattr(driver,"cheap_domain_preflight",lambda:(_ for _ in ()).throw(ValueError("wrong domain")))
    result=driver.run(fake_run.output,device="cpu")
    assert not fake_run.events and result["failure"]["reason"]=="wrong domain"
    assert result["registered_carrier_count"]==40 and len(result["units"])==77


@pytest.mark.parametrize("seconds",[0.,-1.,float("inf"),float("nan")])
def test_invalid_budget_rejected_but_no_old_three_hour_cap(fake_run,seconds):
    with pytest.raises(ValueError):driver.run(fake_run.output,device="cpu",wall_seconds=seconds)


def test_output_overlap_overwrite_and_login_guards(fake_run,monkeypatch):
    fake_run.output.mkdir()
    with pytest.raises(ValueError,match="fresh"):driver.run(fake_run.output,device="cpu")
    monkeypatch.delenv("SLURM_JOB_ID")
    with pytest.raises(ValueError,match="allocation"):driver.run(fake_run.output.parent/"other",device="cpu")


@pytest.fixture
def frozen_history(tmp_path,monkeypatch):
    history=tmp_path/"history";history.mkdir();monkeypatch.setattr(driver,"HISTORY",history)
    plan,arrays=tiny_plan()
    old_code=tmp_path/"old_tree/scripts/run_vpfp_strong_initial_layer_audit.py"
    old_code.parent.mkdir(parents=True);old_code.write_text("# synthetic frozen old source")
    plan["files"]={str(old_code):driver.old.existing.file_sha(old_code)}
    (history/"frozen-plan.json").write_text(json.dumps(plan))
    plan_sha=driver.old.existing.file_sha(history/"frozen-plan.json");monkeypatch.setattr(driver,"HISTORY_PLAN_SHA",plan_sha)
    slots=driver.old.registered_slots()
    for i,(slot,row) in enumerate(zip(slots,plan["selected"])):
        slot["input"]=row
        if i>=26: continue
        slot["status"]="completed"
        for kind in driver.KINDS:
            path=history/slot["id"]/(kind+".json");path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps({"execution_passed":True,"input_full_sha256":row["native_full_sha256"],"source_provenance":row}))
            slot[kind]={"status":"completed","report":str(path),"sha256":driver.old.existing.file_sha(path)}
    result={"schema":driver.old.SCHEMA,"execution_passed":False,"registered_carrier_count":40,"completed_carrier_count":26,
        "frozen_plan_sha256":plan_sha,"design":driver.old.design(),"slots":slots,"same_seed_models_paired":True,
        "refinement":{"status":"failed","reason":"domain"}}
    (history/"result.json").write_text(json.dumps(result))
    monkeypatch.setattr(driver,"HISTORY_RESULT_SHA",driver.old.existing.file_sha(history/"result.json"))
    current=deepcopy(plan);current["files"]={}
    manifest={"provenance":{"generation_config":{"nx":16,"ny":16,"velocity_grid_size":21,"nt":33,"t1":1.,
        "velocity_max":8.,"internal_dt_max":.001,"dtype":"float32"},"decoder_config":{
            "time_domain":[0.,1.],"equation":{"x_domain":[6.2831853330135345]*2}}}}
    monkeypatch.setattr(driver.old,"freeze_inputs",lambda check:(current,arrays,manifest,object()))
    return SimpleNamespace(history=history,plan=plan,current=current,result=result)


def test_history_26_report_hashes_and_original40_input_rows_bound(frozen_history):
    plan,_,_,_=driver.freeze_completion()
    assert plan["historical_result"]["sha256"]==driver.HISTORY_RESULT_SHA
    assert len(plan["historical_slots"])==40 and plan["selected"]==frozen_history.plan["selected"]
    assert len([p for p in plan["files"] if p.endswith("/independent_physics.json")])==26
    assert len([p for p in plan["files"] if p.endswith("/temporal_frequency.json")])==26
    assert plan["historical_terminal"]["exit_code"]=="124"


def test_history_mutated_report_refuses_reuse(frozen_history):
    path=Path(frozen_history.result["slots"][0]["independent_physics"]["report"])
    path.write_text('{"execution_passed":true,"changed":true}')
    with pytest.raises(ValueError,match="historical SHA"):driver.freeze_completion()


def test_current_new_draw_or_seed_cannot_replace_original_slot(frozen_history):
    frozen_history.current["selected"][26]["source_seed"]+=1
    with pytest.raises(ValueError,match="original selection"):driver.freeze_completion()


def test_historical_completed_count_and_candidate_ancestry_cannot_change(frozen_history):
    result=deepcopy(frozen_history.result);plan=deepcopy(frozen_history.plan)
    result["completed_carrier_count"]=27
    with pytest.raises(ValueError):driver.validate_history(result,plan)
    result=deepcopy(frozen_history.result);plan["same_seed_models_paired"]=False
    with pytest.raises(ValueError):driver.validate_history(result,plan)


def test_t0_wrapper_uses_actual_independent_forward_fd_and_no_outside_points(monkeypatch):
    from pde_agent_verifier_solver.plugins.vpfp import independent_physics
    captured={}
    monkeypatch.setattr(driver.old,"physical_points",lambda *a:(None,np.array([[.2,.3]])))
    def audit(*args,**kwargs):captured.update(kwargs);return {"execution_passed":True}
    monkeypatch.setattr(independent_physics,"audit_independent_physics",audit)
    model=SimpleNamespace(config=SimpleNamespace(time_domain=(2.,3.)))
    result=driver.t0_balances(model,None,None)
    np.testing.assert_array_equal(captured["points_txy"],[[2.,.2,.3]])
    assert captured["relative_fd_step"]==2.**-9 and captured["spatial_grids"]==() and captured["spatial_times"]==[]
    assert not captured["include_l2"] and "no outside-time query" in result["time_FD_stencil"]


def test_extended257_513_actual_tiny_native_decoder_source_derivative_is_fd():
    """Tiny degree1/native33 fixture, not corpus or scientific validation."""
    from dataclasses import replace
    import torch
    from tests.test_vpfp_moving_frame import tiny
    from pde_agent_verifier_solver.plugins.vpfp.perturbation_backend import _legacy_api
    from pde_agent_verifier_solver.plugins.vpfp.moving_frame import value_config_from_decoder
    from pde_agent_verifier_solver.plugins.vpfp.value_control_diffusion import lift_value_controls
    before=torch.get_num_threads();torch.set_num_threads(1)
    try:
        source,_,_,_,condition=tiny(dtype="float64")
        config=replace(source.config,structured_time_control_count=33)
        config=replace(config,latent_dim=config.structured_latent_layout()["total_latent_dim"])
        model=_legacy_api().build_deterministic_direct_model(config,device="cpu")
        value=value_config_from_decoder(model)
        z=torch.zeros((1,value.latent_dimension),dtype=torch.float64)
        kinetic=z[:,value.rho_dimension:].reshape(1,value.hermite_channels,33,9)
        kinetic[:,0,:,0]=1.
        kinetic[:,1,:,0]=.05*torch.linspace(0,1,33,dtype=torch.float64)
        full=lift_value_controls(z,value)
        training=model.training
        result=driver.extended_temporal(model,full,condition,row={"origin":"synthetic_fixture"})
        assert result["execution_passed"] and result["time_counts"]==[257,513]
        assert model.training==training and not result["uniform_convergence_established"]
        a,b=result["grids"]
        np.testing.assert_array_equal(np.asarray(a["times"]),np.asarray(b["times"])[::2])
        for grid in (a,b):
            source_values=np.asarray(grid["values"]["source"])
            assert source_values.shape==(grid["count"],1)
            np.testing.assert_array_equal(grid["dt_values"]["source"],
                np.gradient(source_values,grid["times"],axis=0,edge_order=2))
            assert set(grid["t0_pointwise_one_sided_FD_minus_analytic_dt"])=={"rho","jx","jy","K","f"}
        json.dumps(result,allow_nan=False)
    finally: torch.set_num_threads(before)
