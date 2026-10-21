"""Synthetic orchestration only; numerical execution belongs on compute nodes."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("strong_initial_layer_test_driver",ROOT/"scripts/run_vpfp_strong_initial_layer_audit.py")
driver=importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def completed():
    return {"schema":"vpfp_frozen_regime_fidelity_coverage_run_v3","execution_passed":True,
        "all_requested_physical_adapters_completed":True,"confirmation_512_completed":False,
        "admitted":False,"physical_regime_qualified":False,"new_training":False,
        "new_generated_answers":0,"pde_solves":0,"sample_count":128,"stage":"complete",
        "source_manifest_sha256":driver.SOURCE_SHA,"model_source_manifest_sha256":driver.SOURCE_SHA,
        "reference_source_manifest_sha256":driver.SOURCE_SHA,
        "evidence_role":"previously_opened_development_answers_not_blind_confirmation",
        "targets":{label:{"directory":path,"status":"descriptive_audit_completed","attempted_count":128,
            "failed_count":0,"admitted":False,"audit_path":str(driver.AUDIT_ROOT/(label+"-audit.json"))}
            for label,path in driver.all_audited_states().items()}}


def test_fixed40_plus_one_refinement_protocol_not512():
    d=driver.design();slots=driver.registered_slots()
    assert len(slots)==40 and len({s["id"] for s in slots})==40
    assert [s["stratum_id"] for s in slots[:8]]==list(range(8))
    assert d["time_counts"]==[65,129] and d["spatial_mass_grids"]==[17,33]
    assert d["time_fractions"]==[.0078125,.0234375,.7578125]
    h=d["relative_fd_step"]
    for fraction in d["time_fractions"]:
        assert 0<fraction-h<fraction+h<1
        assert int((fraction-h)*32)==int((fraction+h)*32)
    assert h==2.**-9 and d["velocity_moment_orders"]==[17,19] and d["velocity_L2_orders"]==[32,34]
    assert d["refinement"]["stratum_id"]==7 and d["refinement"]["root_count"]==1
    assert not d["automatic_downsampling"] and not d["scientific_admission"]
    assert "t0_one_sided_FD" in d["pending"]
    assert [(c["replica"],c["arm"],c["step"]) for c in driver.candidates()]==[
        (0,"initial_conditioned",40000),(1,"initial_conditioned",40000),
        (0,"initial_conditioned_rho_cov",20000),(1,"initial_conditioned_rho_cov",20000)]


@pytest.mark.parametrize("bad",["missing_state","checkpoint_directory","failure","count512","admitted","source"])
def test_completed14_original_audit_binding_rejects_changes(bad):
    value=completed();driver.validate_completed_audit(value)
    if bad=="missing_state":del value["targets"]["target-01"]
    elif bad=="checkpoint_directory":value["targets"]["target-05"]["directory"]="other_model"
    elif bad=="failure":value["targets"]["target-05"]["failed_count"]=1
    elif bad=="count512":value["sample_count"]=512
    elif bad=="admitted":value["admitted"]=True
    elif bad=="source":value["reference_source_manifest_sha256"]="b"*64
    with pytest.raises(ValueError):driver.validate_completed_audit(value)


def test_first_metadata_row_selection_does_not_consult_success_or_features():
    edges=driver.existing._edges([10.,15.,20.,25.,30.],[.8,1.,1.2])
    conditions=np.asarray([[11.,.9],[11.,1.1],[16.,.9],[16.,1.1],[21.,.9],[21.,1.1],[26.,.9],[26.,1.1]]*2,dtype=np.float32)
    rows,bins=driver.select_first_rows(conditions,edges)
    assert rows==list(range(8)) and list(bins)==list(range(8))*2
    # Invalid raw is retained with its original row/hash, never replaced by row8.
    record,_=driver._input_row("synthetic",0,np.asarray([np.nan],dtype=np.float32),conditions[0],"fixture",3)
    assert not record["input_finite"] and record["row"]==0
    with pytest.raises(ValueError):driver.select_first_rows(conditions[:7],edges)


def test_measurement_local_rows_map_back_to_original_indices():
    permutation=list(reversed(range(128)))
    measure={"cases":[{"row":i,"input_answer_sha256":f"sha-{i}"} for i in range(128)]}
    result=driver.measured_original_rows(measure,{"selected_rows":permutation},{"selected_rows":permutation})
    assert result[127]=="sha-0" and result[0]=="sha-127"
    with pytest.raises(ValueError):driver.measured_original_rows(measure,{"selected_rows":permutation},{"selected_rows":list(range(128))})


def test_known_initial_and_future_streams_use_existing_native_tags():
    seeds=list(range(128))
    initial=driver.existing._native_stream("vpfp-initial-conditioned-prior",seeds)
    prior={"known_initial_seed_applied":True,"design":{"sampling_seeds":seeds,"known_initial_seeds":initial}}
    _,future=driver.validate_generation_design(prior)
    assert future==driver.existing._native_stream("vpfp-initial-conditioned-future",seeds)
    assert future!=initial
    prior["design"]["known_initial_seeds"][0]+=1
    with pytest.raises(ValueError):driver.validate_generation_design(prior)


@pytest.fixture
def fake_run(monkeypatch,tmp_path):
    """Mock all asset/runtime/numerical calls; this proves ordering, not physics."""
    monkeypatch.setenv("SLURM_JOB_ID","123456")
    monkeypatch.setenv("SLURM_TMPDIR",str(tmp_path))
    monkeypatch.setattr(driver,"SCRATCH",tmp_path)
    selected=[];arrays={};events=[];output=tmp_path/"out"
    for i,slot in enumerate(driver.registered_slots()):
        record,value=driver._input_row(slot["group"],i,np.asarray([float(i),2.,3.],dtype=np.float32),
            np.asarray([20.,1.],dtype=np.float32),"fixture-root-"+str(i),i)
        record.update(id=slot["id"],stratum_id=slot["stratum_id"])
        selected.append(record);arrays[record["id"]]=value
    plan={"schema":driver.SCHEMA,"files":{},"selected":selected,"design":driver.design(),
        "refinement_selected":selected[7],"same_seed_models_paired":True}
    def freeze(check):
        check();events.append("freeze_all40")
        initial=json.loads((output/"result.json").read_text())
        assert len(initial["slots"])==40 and initial["refinement"]["status"]=="pending"
        return plan,arrays,{},object()
    def runtime(*args):
        assert len(json.loads((output/"frozen-plan.json").read_text())["selected"])==40
        events.append("runtime_after_plan")
        return SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(hermite_total_degree=14)),frozen_files={})
    def independent(model,raw,condition,design):
        events.append(("physics",int(raw[0])))
        return {"execution_passed":True,"numerically_uncertain":True,"fixture":True}
    def temporal(model,raw,condition,**kwargs):
        events.append(("temporal",int(raw[0])))
        assert kwargs["design"]["time_counts"]==[65,129]
        return {"execution_passed":True,"fixture":True}
    def refine(plan,model,out,device):
        assert plan["refinement_selected"]["stratum_id"]==7
        events.append("refinement_stratum7")
        return {"execution_passed":True,"new_pde_solves":2,"new_pde_solve_attempts":2,"fixture":True}
    monkeypatch.setattr(driver,"freeze_inputs",freeze)
    monkeypatch.setattr(driver.existing,"_runtime",runtime)
    monkeypatch.setattr(driver.existing,"check_frozen",lambda _:events.append("frozen_checked"))
    monkeypatch.setattr(driver,"_tensors",lambda model,values:values)
    monkeypatch.setattr(driver,"_independent",independent)
    monkeypatch.setattr(driver,"native_temporal_frequency",temporal)
    monkeypatch.setattr(driver,"_refinement",refine)
    return SimpleNamespace(output=output,plan=plan,arrays=arrays,events=events)


def test_pipeline_all_frozen_before_measurement_and_refinement_before_candidates(fake_run):
    result=driver.run(fake_run.output,device="cpu",wall_seconds=100)
    assert result["execution_passed"] and result["completed_carrier_count"]==40
    events=fake_run.events
    assert events.index("freeze_all40")<events.index("runtime_after_plan")<events.index(("physics",0))
    assert events.index(("temporal",7))<events.index("refinement_stratum7")<events.index(("physics",8))
    assert [e[1] for e in events if isinstance(e,tuple) and e[0]=="physics"]==list(range(40))
    assert result["same_seed_models_paired"] and "not40_independent_roots" in result["statistical_unit"]
    assert result["first_reference_benchmark"]["is_extra_root"] is False
    assert result["refinement"]["new_pde_solves"]==2
    assert result["numerically_uncertain"] and not result["confirmation_512_completed"] and not result["admitted"]


def test_measurement_failure_kept_and_other_registered_roots_continue(fake_run,monkeypatch):
    original=driver._independent
    def one_failure(model,raw,condition,design):
        if raw[0]==2:return {"execution_passed":False,"error":"synthetic quadrature failure"}
        return original(model,raw,condition,design)
    monkeypatch.setattr(driver,"_independent",one_failure)
    result=driver.run(fake_run.output,device="cpu",wall_seconds=100)
    assert not result["execution_passed"] and result["completed_carrier_count"]==39
    assert result["slots"][2]["status"]=="measurement_failed"
    failed=json.loads(Path(result["slots"][2]["independent_physics"]["report"]).read_text())
    assert failed["error"]=="synthetic quadrature failure"
    assert result["refinement"]["status"]=="completed" and ("physics",39) in fake_run.events


def test_deadline_keeps_all40_denominators_and_unstarted_refinement(fake_run,monkeypatch):
    original=driver._independent
    def stop(model,raw,condition,design):
        if raw[0]==2:raise TimeoutError("synthetic fixed deadline")
        return original(model,raw,condition,design)
    monkeypatch.setattr(driver,"_independent",stop)
    result=driver.run(fake_run.output,device="cpu",wall_seconds=100)
    assert not result["execution_passed"] and result["completed_carrier_count"]==2
    assert len(result["slots"])==40 and result["refinement"]["status"]=="pending"
    assert result["slots"][2]["status"]=="incomplete"
    assert all(s["status"]=="pending" for s in result["slots"][3:])
    assert result["failure"]["type"]=="TimeoutError" and result["design"]["time_counts"]==[65,129]


def test_bad_freeze_prevents_runtime_and_keeps_registered_slots(fake_run,monkeypatch):
    monkeypatch.setattr(driver,"freeze_inputs",lambda check:(_ for _ in ()).throw(ValueError("synthetic wrong SHA")))
    result=driver.run(fake_run.output,device="cpu",wall_seconds=100)
    assert result["failure"]["reason"]=="synthetic wrong SHA" and not fake_run.events
    assert all(slot["status"]=="pending" for slot in result["slots"])


@pytest.mark.parametrize("seconds",[0.,10801.,float("nan"),float("inf")])
def test_no_wall_budget_expansion(fake_run,seconds):
    with pytest.raises(ValueError):driver.run(fake_run.output,device="cpu",wall_seconds=seconds)
    assert not fake_run.output.exists()


def test_no_output_overwrite_or_login_execution(fake_run,monkeypatch):
    fake_run.output.mkdir();sentinel=fake_run.output/"keep";sentinel.write_text("original")
    with pytest.raises(ValueError):driver.run(fake_run.output,device="cpu",wall_seconds=100)
    assert sentinel.read_text()=="original"
    monkeypatch.delenv("SLURM_JOB_ID")
    with pytest.raises(ValueError,match="compute"):driver.run(fake_run.output.parent/"other",device="cpu",wall_seconds=100)
