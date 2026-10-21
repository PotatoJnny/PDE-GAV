"""Pure orchestration and tiny native mechanics; pytest only on compute nodes."""
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("independent_physics_cli_tests",ROOT/"scripts/run_vpfp_independent_physics.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


@pytest.fixture
def plan():
    rows = [{"stratum_id":i%8,"case_id":f"root-{i}","equation_seed":i} for i in range(512)]
    candidates = [{"id":"strong-model","regime":"strong","rows":deepcopy(rows)}]
    references = {"strong":{"expected_splits":{"calibration":rows}}}
    return {"references":references,"candidates":candidates,"master_seed":17,
            "physics_design":driver.physics_design(references,candidates,17)}


def test_design_frozen8_original_rows_even_if_failed_or_extreme(plan):
    design = plan["physics_design"]
    assert design["reference_rows"]["strong"] == list(range(8))
    assert design["candidate_rows"]["strong-model"] == list(range(8))
    plan["candidates"][0]["rows"][0].update(failed=True,score=1e99)
    assert driver.physics_design(plan["references"],plan["candidates"],17) == design
    assert design["time_fractions"] == [.0703125,.7578125] and design["time_counts"] == [65,129]
    assert all((t*32)%1==.25 for t in design["time_fractions"])
    assert design["spatial_mass_grids"] == [17,33]
    assert design["refinement"]["max_cases"] == 1 and not design["refinement"]["refined_512_population"]
    assert not design["scientific_admission"]


def test_exact_six_physical_drive_levels_and_fixed_frequency_law(plan):
    controls = plan["physics_design"]["controls"]
    drives = [item for item in controls if item["kind"]=="drive_velocity"]
    assert [(item["global_amplitude"][0],item["cycles"]) for item in drives] == [
        (.125,1.),(.125,8.),(.5,1.),(.5,8.),(2.,1.),(2.,8.)]
    for item in drives:
        assert item["local_amplitude"] == [item["global_amplitude"][0]/2,0.]
        assert item["amplitude_units"] == "physical_velocity_not_thermal_scaled"
    weight = next(item for item in controls if item["id"]=="reweighted")
    np.testing.assert_allclose(weight["source_sampling_weights"],np.exp(-np.arange(8))/np.exp(-np.arange(8)).sum(),rtol=0,atol=0)
    assert weight["indices"] == [0,0,0,0,0,1,1,1,2,2,3,3,4,5,6,7]
    assert sorted(set(weight["indices"])) == list(range(8))
    assert next(item for item in controls if item["id"]=="collapse")["indices"] == [0]*8


@pytest.mark.parametrize("rows", [[{"stratum_id":i} for i in range(7)], [{"stratum_id":True}], [{"stratum_id":8}]])
def test_missing_or_malformed_stratum_not_silently_replaced(rows):
    with pytest.raises(ValueError): driver.first_stratum_rows(rows)


def sample(count=8, *, repeated=False):
    from pde_agent_verifier_solver.plugins.vpfp.regime_fidelity_coverage import RegimeAuditSample
    roots = tuple("same" if repeated else f"root-{i}" for i in range(count))
    hashes = tuple("a"*64 if repeated else f"{i:064x}" for i in range(count))
    return RegimeAuditSample(np.zeros((count,16)),np.tile([.03,.9],(count,1)),roots,hashes,
        np.ones(count,dtype=bool),np.zeros(count,dtype=bool),np.zeros((count,3)),
        budget_defined_mask=np.ones((count,3),dtype=bool))


def test_literal_collapse_retains8_outputs_but_one_statistical_root():
    original = sample(repeated=True)
    effective,receipt = driver.independent_control_sample(original)
    assert len(effective.root_ids)==1 and effective.root_ids==("same",)
    assert receipt["control_rows"]==8 and receipt["distinct_physical_roots"]==1
    assert receipt["independent_statistical_rows"]==[0] and receipt["same_root_replicas_are_not_new_N"]
    assert len(original.root_ids)==8


@pytest.mark.parametrize("field",["features","conditions","answer_sha256"])
def test_different_descendants_cannot_bypass_cluster_accounting(field):
    original = sample(repeated=True)
    if field=="answer_sha256": original = replace(original,answer_sha256=("b"*64,*original.answer_sha256[1:]))
    else:
        values = np.array(getattr(original,field),copy=True); values[1,0]+=1
        original = replace(original,**{field:values})
    with pytest.raises(ValueError,match="cluster estimator"): driver.independent_control_sample(original)


def test_subset_remaps_failed_rows_and_keeps_undefined_budget_mask():
    original = replace(sample(),failures=({"row":3,"reason":"fixed failure"},{"row":5,"reason":"excluded"}))
    result = driver.subset_sample(original,[7,3])
    assert result.root_ids==("root-7","root-3")
    assert result.failures==({"row":1,"reason":"fixed failure"},)
    assert result.budget_defined_mask.shape==(2,3)


def test_partial_progress_retains_failure_then_continues_independent_unit(tmp_path,plan):
    progress = driver._Progress(tmp_path/"out","a"*64,plan["physics_design"],"strong",20.)
    def failure(): raise ValueError("kept original error")
    assert progress.unit("first",failure) is None
    progress.unit("second",lambda:{"execution_passed":True,"value":3})
    value = json.loads((tmp_path/"out/result.json").read_text())
    assert value["units"]["first"]["failure"]["reason"]=="kept original error"
    assert value["units"]["second"]["status"]=="completed"
    assert all(row["status"]=="pending" for row in value["controls"].values())
    assert value["execution_passed"] is False and (tmp_path/"out/second.json").is_file()


def test_failed_physics_report_is_not_execution_pass(tmp_path,plan):
    progress = driver._Progress(tmp_path/"out","a"*64,plan["physics_design"],"strong",20.)
    progress.unit("root",lambda:{"execution_passed":False,"points":[{"reason":"nonfinite"}]})
    assert progress.report["units"]["root"]["status"]=="measurement_failed"


def test_nonempty_carrier_does_not_hide_MMS_or_measurement_failure():
    units = {"carrier":{"status":"completed"},"MMS":{"status":"measurement_failed"}}
    assert driver.control_status(units,list(units),True)=="completed_with_measurement_failures"
    assert driver.control_status(units,list(units),False)=="failed"
    units["MMS"]["status"]="completed"
    assert driver.control_status(units,list(units),True)=="completed"


def test_deadline_does_not_start_or_mark_unrun_control_complete(tmp_path,monkeypatch,plan):
    monkeypatch.setattr(driver.time,"monotonic",lambda:10.)
    progress = driver._Progress(tmp_path/"out","a"*64,plan["physics_design"],"strong",1.)
    monkeypatch.setattr(driver.time,"monotonic",lambda:12.)
    with pytest.raises(TimeoutError): progress.unit("must_not_run",lambda:pytest.fail("budget exhausted"))
    assert "must_not_run" not in progress.report["units"]
    assert all(item["status"]=="pending" for item in progress.report["controls"].values())


def test_reference_cache_uses512_and_restores_tuple_plan_fields(monkeypatch,tmp_path):
    from pde_agent_verifier_solver.plugins.vpfp import diffusion_data
    from pde_agent_verifier_solver.plugins.vpfp.regime_fidelity_coverage import VPFPRegimeFeaturePlan
    from dataclasses import asdict
    audit,corpus = tmp_path/"audit",tmp_path/"corpus"
    audit.mkdir(); corpus.mkdir()
    conditions = np.tile([.03,.9],(512,1))
    split = SimpleNamespace(conditions=conditions,case_ids=np.asarray([f"r-{i}" for i in range(512)]))
    monkeypatch.setattr(diffusion_data,"load_split",lambda *args,**kwargs:split)
    cases = [{"row":i,"input_answer_sha256":f"{i:064x}","hard_valid":True,"numerically_uncertain":True,
              "budgets":[0.,0.,0.],"budget_defined_mask":[True]*3} for i in range(512)]
    documents = {"result.json":{"execution_passed":True,"sample_count":512},"descriptors.json":[{"name":"x"}],
        "frozen-plan.json":{"reference_source":str(corpus),"plan":asdict(VPFPRegimeFeaturePlan()),
          "reference_rows":{role:list(range(512)) for role in ("calibration","reference_a","reference_b")}}}
    for name,value in documents.items(): (audit/name).write_text(json.dumps(value))
    for role in ("calibration","reference_a","reference_b"):
        np.save(audit/(role+"-features.npy"),np.zeros((512,16)))
        (audit/(role+"-measurements.json")).write_text(json.dumps({"cases":cases}))
    frozen = {}
    cached,_,_,plan,_ = driver._reference_cache(audit,corpus,{},frozen)
    assert all(len(value.root_ids)==512 for value in cached.values())
    assert isinstance(plan.grid_sizes,tuple) and isinstance(plan.time_fractions,tuple)
    assert len(frozen)==9
    (audit/"result.json").write_text(json.dumps({"execution_passed":True,"sample_count":128}))
    with pytest.raises(ValueError,match="all512"): driver._reference_cache(audit,corpus,{},frozen)


def test_native_B_time_probe_tiny_actual_decoder_no_A_substitution():
    import torch
    from tests.test_vpfp_moving_frame import tiny
    from pde_agent_verifier_solver.plugins.vpfp.value_control_diffusion import lift_value_controls
    old = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        base,value,_,z,condition = tiny(dtype="float64")
        was_training = base.training
        full = lift_value_controls(z,value)
        design = {"time_counts":[5,9],"time_fractions":[.0625,.75],
            "spatial_probe_fractions":[[.173,.389]],"physical_velocity_probes":[[0.,0.]]}
        result = driver.native_temporal_frequency(base,full,condition,root_id="synthetic-root",
            provenance={"source_manifest_sha256":"a"*64,"split":"fixture","data_origin":"synthetic_fixture"},design=design)
        assert result["decoder_kind"]=="old_full_B" and result["time_counts"]==[5,9]
        assert result["execution_passed"] and not result["scientific_regime_qualified"]
        for grid in result["grids"]:
            np.testing.assert_allclose(grid["values"]["jx"],0.,atol=1e-12)
            np.testing.assert_allclose(grid["values"]["source"],0.,atol=1e-12)
            np.testing.assert_allclose(grid["dt_values"]["K"],0.,atol=1e-12)
        assert base.training is was_training
        json.dumps(result,allow_nan=False)
    finally:
        torch.set_num_threads(old)


def test_physics_script_has_no_training_sampling_or_child_scheduler():
    source = (ROOT/"scripts/run_vpfp_independent_physics.py").read_text()
    for token in ("sbatch","srun","shared.train(","sample_conditions(","model.sample("):
        assert token not in source
    assert "refine_reference_samples" in source and "audit_independent_physics(" in source
    assert "extract_moving_frame_regime_features(" in source and "audit_regime_fidelity_coverage(" in source


@pytest.fixture(params=["strong_native","weak_native","weak_hierarchy"])
def bound_candidate(tmp_path,request):
    """Upstream native validation is mocked; test the NEW schedule boundary.

    All file/receipt hashes are real and self-consistent. Both substitute
    checkpoint headers retain the identical training-source SHA. Their toy
    bytes are not presented as real trained assets or physical validation.
    """
    def asset(name,content):
        path = tmp_path/name
        path.write_text(json.dumps(content))
        return {"path":str(path),"sha256":driver.file_sha(path)}
    header = {"corpus_manifest_sha256":"c"*64,"base_decoder_sha256":"d"*64,"synthetic_upstream_validated":True}
    original = asset("original.pt",{**header,"model":"original"})
    other = asset("other.pt",{**header,"model":"another_legal_model_same_source"})
    directory = tmp_path/"generated"; directory.mkdir()
    rows = [{"seed":100+i,"condition_seed":1000+i,"conditions":[.03,.9],"stratum_id":i//64} for i in range(512)]
    kind = request.param
    candidate = {"id":"selected","kind":kind,"regime":"strong" if kind=="strong_native" else "weak",
        "asset":original,"rows":rows,"generated_dir":str(directory),
        "model_source":{"corpus":str(tmp_path/"training"),"manifest_sha256":"c"*64}}
    provenance = {"selection_sha256":"e"*64,"seeds":[r["seed"] for r in rows],
        "condition_seeds":[r["condition_seed"] for r in rows],"model_source_manifest_sha256":"c"*64}
    target = {"status":"ready_native_direct","conditions":np.asarray([r["conditions"] for r in rows],dtype=np.float32),"files":{}}
    if kind in ("strong_native","weak_native"):
        provenance.update(checkpoint=original,checkpoint_sha256=original["sha256"])
    else:
        selected_manifest = {"schema":"vpfp_moving_frame_native_model_manifest_v1",
            "base_decoder":asset("base.pt",header),"macro_checkpoint":original,
            "conditional_checkpoint":asset("conditional.pt",header)}
        candidate["asset"] = asset("selected-native.json",selected_manifest)
        bindings = {role:{"checkpoint":selected_manifest[role+"_checkpoint"]["path"],
            "checkpoint_sha256":selected_manifest[role+"_checkpoint"]["sha256"]} for role in ("macro","conditional")}
        provenance["checkpoint_bindings"] = deepcopy(bindings)
        target.update(status="ready_paired_A_B",checkpoint_bindings=bindings)
    plan = {"selection_sha256":"e"*64,"candidates":[candidate]}
    def publish():
        path = directory/"generated-provenance.json"
        path.write_text(json.dumps(provenance))
        target["files"][str(path)] = driver.file_sha(path)
    publish()
    return SimpleNamespace(candidate=candidate,plan=plan,target=target,provenance=provenance,
        other=other,publish=publish)


def test_native_validity_plus_exact_registered_schedule_accepted(bound_candidate):
    fixture = bound_candidate
    result = driver.validate_candidate_schedule(fixture.candidate,fixture.plan,fixture.target)
    assert result["execution_passed"] and result["registered_attempt_count"]==512
    assert result["root_and_condition_seed_lists_equal"] and result["condition_fp32_bits_equal"]


def test_candidate_schedule_rejects_kind_from_other_regime_before_measurement(bound_candidate):
    fixture = bound_candidate
    fixture.candidate["regime"] = "weak" if fixture.candidate["regime"]=="strong" else "strong"
    with pytest.raises(ValueError,match="collision regime"):
        driver.validate_candidate_schedule(fixture.candidate,fixture.plan,fixture.target)


@pytest.mark.parametrize("change",["selection","roots","condition_seeds","conditions","bool_seed"])
def test_self_consistent_other512_schedule_rejected_even_same_source(bound_candidate,change):
    fixture = bound_candidate
    if change=="selection": fixture.provenance["selection_sha256"]="f"*64
    if change=="roots": fixture.provenance["seeds"]=[seed+10000 for seed in fixture.provenance["seeds"]]
    if change=="condition_seeds": fixture.provenance["condition_seeds"]=[seed+10000 for seed in fixture.provenance["condition_seeds"]]
    if change=="conditions": fixture.target["conditions"][511,1]=np.nextafter(np.float32(.9),np.float32(1.))
    if change=="bool_seed": fixture.provenance["seeds"][0]=True
    fixture.publish()  # Receipt remains self-consistent and SHA-frozen.
    assert fixture.provenance["model_source_manifest_sha256"]==fixture.candidate["model_source"]["manifest_sha256"]
    with pytest.raises(ValueError,match="registered|selection"):
        driver.validate_candidate_schedule(fixture.candidate,fixture.plan,fixture.target)


def test_other_legal_native_model_same_corpus_rejected(bound_candidate):
    fixture = bound_candidate
    if fixture.candidate["kind"] in ("strong_native","weak_native"):
        fixture.provenance.update(checkpoint=fixture.other,checkpoint_sha256=fixture.other["sha256"])
    else:
        other = {"checkpoint":fixture.other["path"],"checkpoint_sha256":fixture.other["sha256"]}
        fixture.provenance["checkpoint_bindings"]["macro"]=other
        fixture.target["checkpoint_bindings"]["macro"]=deepcopy(other)
    fixture.publish()
    with pytest.raises(ValueError,match="explicitly selected"):
        driver.validate_candidate_schedule(fixture.candidate,fixture.plan,fixture.target)


def test_condition_comparison_really_uses_FP32_bits(bound_candidate):
    fixture = bound_candidate
    fixture.candidate["rows"][0]["conditions"][0]=0.
    fixture.target["conditions"][0,0]=-0.
    assert float(fixture.target["conditions"][0,0])==0.
    with pytest.raises(ValueError,match="FP32 condition bits"):
        driver.validate_candidate_schedule(fixture.candidate,fixture.plan,fixture.target)


def test_all_target_freeze_binds_schedule_before_any_physical_runtime(bound_candidate,monkeypatch):
    import ast
    import run_vpfp_regime_fidelity_coverage as audit
    fixture = bound_candidate
    calls=[]
    monkeypatch.setattr(audit,"freeze_generated",lambda *args,**kwargs: calls.append((args,kwargs)) or fixture.target)
    frozen={}; checks=[]
    targets,receipts = driver.freeze_candidate_targets(fixture.plan,fixture.candidate["regime"],{},None,None,
        frozen=frozen,check_deadline=lambda:checks.append("checked"))
    assert checks==["checked"] and calls[0][1]["count"]==512
    assert targets["selected"] is fixture.target and receipts["selected"]["execution_passed"]
    assert frozen==fixture.target["files"]
    fixture.provenance["seeds"]=[seed+10000 for seed in fixture.provenance["seeds"]]; fixture.publish()
    with pytest.raises(ValueError,match="registered512"):
        driver.freeze_candidate_targets(fixture.plan,fixture.candidate["regime"],{},None,None,frozen={})
    source = ast.parse((ROOT/"scripts/run_vpfp_independent_physics.py").read_text())
    function = next(node for node in source.body if isinstance(node,ast.FunctionDef) and node.name=="run")
    calls = [(node.func.id,node.lineno) for node in ast.walk(function)
             if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)]
    assert min(line for name,line in calls if name=="freeze_candidate_targets") < min(line for name,line in calls if name=="_runtime")
    assert not any(name=="freeze_generated" for name,_ in calls)  # No hours-late reloading of a different target.
