"""Synthetic namespace, source-count and orchestration checks; compute-only pytest."""
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("independent_confirmation_tests",ROOT/"scripts/run_vpfp_independent_confirmation.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def asset(path,value):
    path.write_text(json.dumps(value))
    return {"path":str(path),"sha256":sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def selection(tmp_path):
    checkpoint = asset(tmp_path/"model.pt",{"synthetic_only":True})
    development = asset(tmp_path/"development.json",{"seeds":[11,12],"design":{"known_initial_seeds":[21,22],"conditions":[[.02,.9]]}})
    source = {"corpus":str(tmp_path/"source/corpus"),"manifest_sha256":"c"*64}
    candidate = {"id":"strong-one","regime":"strong","kind":"strong_native","asset":checkpoint,
        "model_source":source,"development_provenance":[development]}
    value = {"schema":driver.SELECTION_SCHEMA,"review":"Explicit synthetic mechanics fixture only",
        "scientific_admission":False,"candidates":[candidate]}
    path = tmp_path/"selection.json"
    item = asset(path,value)
    return SimpleNamespace(path=path,item=item,value=value,candidate=candidate)


def test_selection_is_explicit_hash_bound_and_excludes_recorded_ancestry(selection):
    actual,frozen,forbidden = driver.load_selection(selection.path,selection.item["sha256"])
    assert actual==selection.value and forbidden=={11,12,21,22}
    assert len(frozen)==3
    selection.path.write_text("changed")
    with pytest.raises(ValueError,match="SHA"):
        driver.load_selection(selection.path,selection.item["sha256"])


@pytest.mark.parametrize("change",["duplicate","admit","no_review","missing_development","wrong_kind","unknown"])
def test_selection_rejects_implicit_or_malformed_candidate_policy(selection,change):
    value = deepcopy(selection.value)
    if change=="duplicate": value["candidates"].append(deepcopy(value["candidates"][0]))
    if change=="admit": value["scientific_admission"]=True
    if change=="no_review": value["review"]=" "
    if change=="missing_development": value["candidates"][0]["development_provenance"]=[]
    if change=="wrong_kind": value["candidates"][0]["kind"]="weak_hierarchy"
    if change=="unknown": value["auto_select_best"]=True
    item = asset(selection.path,value)
    with pytest.raises(ValueError): driver.load_selection(selection.path,item["sha256"])


def test_weak_hierarchy_freezes_all_three_assets(tmp_path,selection):
    value = deepcopy(selection.value)
    manifest = {"schema":"vpfp_moving_frame_native_model_manifest_v1","regime_id":"weak_fixture","sampler_steps":64,
        **{role:asset(tmp_path/(role+".pt"),{"synthetic":role}) for role in ("base_decoder","macro_checkpoint","conditional_checkpoint")}}
    candidate = value["candidates"][0]
    candidate.update(id="weak-one",regime="weak",kind="weak_hierarchy",asset=asset(tmp_path/"native.json",manifest))
    item = asset(selection.path,value)
    _,frozen,_ = driver.load_selection(selection.path,item["sha256"])
    assert all(entry["path"] in frozen for role,entry in manifest.items() if isinstance(entry,dict))


def test_weak_native_selection_keeps_explicit_checkpoint_and_development_ancestry(selection):
    value = deepcopy(selection.value)
    candidate = value["candidates"][0]
    candidate.update(id="weak-w0",regime="weak",kind="weak_native")
    selected = asset(selection.path,value)
    actual,frozen,forbidden = driver.load_selection(selection.path,selected["sha256"])
    assert actual==value and len(frozen)==3 and forbidden=={11,12,21,22}
    assert frozen[candidate["asset"]["path"]]==candidate["asset"]["sha256"]


@pytest.mark.parametrize(("regime","kind"),[
    ("strong","weak_native"),("strong","weak_hierarchy"),("weak","strong_native")])
def test_candidate_kind_wrong_regime_rejected_by_selection_and_plan(selection,tmp_path,regime,kind):
    value = deepcopy(selection.value)
    value["candidates"][0].update(regime=regime,kind=kind)
    selected = asset(selection.path,value)
    with pytest.raises(ValueError,match="collision regime"):
        driver.load_selection(selection.path,selected["sha256"])
    # The direct plan entry also rejects this before imports, assets or seeds.
    with pytest.raises(ValueError,match="collision regime"):
        driver.prepare_plan(value,selected["sha256"],tmp_path/"unused",7,{},set())


@pytest.mark.parametrize("kind",["strong_native","weak_native"])
@pytest.mark.parametrize("change",[None,"source","decoder","extra_field"])
def test_native_plan_uses_same_seven_field_source_binding_before_reference_planning(
        monkeypatch,tmp_path,selection,kind,change):
    import torch
    from pde_agent_verifier_solver.plugins.vpfp import diffusion_data
    value = deepcopy(selection.value)
    candidate = value["candidates"][0]
    regime = "weak" if kind=="weak_native" else "strong"
    candidate.update(regime=regime,kind=kind)
    corpus = Path(candidate["model_source"]["corpus"]); corpus.mkdir(parents=True)
    decoder = asset(tmp_path/"base.pt",{"synthetic_decoder":True})
    seed_plan = asset(corpus/"seed-plan.json",{"synthetic_only":True})
    manifest = {"manifest_sha256":candidate["model_source"]["manifest_sha256"],"regime_id":regime+"_fixture",
        "base_decoder_sha256":decoder["sha256"],"seed_plan":{"path":"seed-plan.json","sha256":seed_plan["sha256"]},
        "model_spec":{label+suffix:decoder[key] for label in ("base_decoder","expanded_decoder")
            for suffix,key in (("_checkpoint","path"),("_sha256","sha256"))}}
    asset(corpus/"manifest.json",manifest)
    payload = {"schema":"synthetic_native_header","config":{},"model":{},"normalizer":{},
        "base_decoder_sha256":decoder["sha256"],"corpus_manifest_sha256":manifest["manifest_sha256"],"training":{}}
    if change=="source": payload["corpus_manifest_sha256"]="d"*64
    if change=="decoder": payload["base_decoder_sha256"]="e"*64
    if change=="extra_field": payload["continuation"]={}
    torch.save(payload,candidate["asset"]["path"])
    candidate["asset"]["sha256"]=driver.file_sha(candidate["asset"]["path"])
    monkeypatch.setattr(diffusion_data,"read_manifest",lambda *args,**kwargs:manifest)
    monkeypatch.setattr(diffusion_data,"_source_exclusion_seeds",lambda _:set())
    class ReferencePlanningReached(Exception): pass
    def stop_reference_planning(_):
        raise ReferencePlanningReached("header checked; no real reference planning or solves")
    monkeypatch.setattr(diffusion_data,"VPFPOnlineModelSpec",SimpleNamespace(from_mapping=stop_reference_planning))
    expected = ReferencePlanningReached if change is None else ValueError
    message = "header checked" if change is None else "seven-field"
    with pytest.raises(expected,match=message):
        driver.prepare_plan(value,"a"*64,tmp_path/"unused",7,{},set())


@pytest.mark.parametrize("value",[{"seeds":[True]},{"seed":-1},{"known_prior_seeds":"12"},{"equation_seed":2**63}])
def test_seed_ancestry_never_coerces_ambiguous_ids(value):
    with pytest.raises(ValueError): driver.seed_ids(value)


def test_512_conditions_and_every_actual_derived_stream_disjoint_and_deterministic(selection):
    manifest = {"collision_nu_range":[.02,.08],"temperature_range":[.8,1.2]}
    rows,used = driver.generation_plan(selection.candidate,manifest,"d"*64,531,{11,12})
    assert len(rows)==512
    assert rows==driver.generation_plan(selection.candidate,manifest,"d"*64,531,{11,12})[0]
    assert np.bincount([r["stratum_id"] for r in rows]).tolist()==[64]*8
    names = ("seed","condition_seed","known_initial_seed","future_noise_seed","macro_noise_seed","micro_noise_seed")
    flat = [row[key] for row in rows for key in names]
    assert len(flat)==len(set(flat))==512*6 and set(flat).isdisjoint({11,12})
    for row in rows:
        assert row["known_initial_seed"]==driver.stream("vpfp-initial-conditioned-prior",row["seed"])
        assert row["future_noise_seed"]==driver.stream("vpfp-initial-conditioned-future",row["seed"])
        assert row["macro_noise_seed"]==driver.stream("vpfp-coherent-macro",row["seed"])
        assert row["micro_noise_seed"]==driver.stream("vpfp-coherent-micro",row["seed"])
    second,_ = driver.generation_plan({**selection.candidate,"id":"strong-two"},manifest,"d"*64,531,used)
    assert set(flat).isdisjoint(row[key] for row in second for key in names)
    forbidden = {rows[0]["known_initial_seed"]}
    revised,_ = driver.generation_plan(selection.candidate,manifest,"d"*64,531,forbidden)
    assert revised[0]!=rows[0]
    assert forbidden.isdisjoint(row[key] for row in revised for key in names)


def test_collect_stage_requires_exact_preplanned_512_reference_rows(monkeypatch,tmp_path):
    from pde_agent_verifier_solver.plugins.vpfp import diffusion_data
    reference = tmp_path/"fresh"
    reference.mkdir()
    rows = {role:[{"equation_seed":i+offset*512} for i in range(512)] for offset,role in enumerate(driver.ROLES)}
    plan = {"reference_protocol":"original_coarse_NOT_refined","references":{"strong":{
        "model_source":{"corpus":"original","manifest_sha256":"a"*64},"reference_source":str(reference),
        "master_seed":42,"expected_splits":rows}}}
    calls=[]
    def collect(*args,**kwargs):
        calls.append((args,kwargs))
        (reference/"seed_plan.json").write_text(json.dumps({"splits":rows}))
        return {"manifest_sha256":"b"*64,"seed_plan":{"path":"seed_plan.json"}}
    monkeypatch.setattr(diffusion_data,"collect_fresh_evaluation_corpus",collect)
    result = driver.collect_stage(plan,"strong","cpu")
    assert calls[0][1]["evaluation_count"]==512 and result["count_per_role"]==512 and result["refined_reference"] is False
    changed = deepcopy(plan)
    changed["references"]["strong"]["expected_splits"]["reference_a"][0]["equation_seed"]+=1
    with pytest.raises(ValueError,match="pre-solve"): driver.collect_stage(changed,"strong","cpu")


@pytest.mark.parametrize("kind",["strong_native","weak_native"])
def test_native_generation_keeps_failed_rows_and_all512_conditions_without_retry(monkeypatch,tmp_path,selection,kind):
    import torch
    from pde_agent_verifier_solver.plugins.vpfp import structured_diffusion
    source_sha = selection.candidate["model_source"]["manifest_sha256"]
    torch.save({"schema":"synthetic_native_header","config":{},"model":{},"normalizer":{},
        "base_decoder_sha256":"b"*64,"corpus_manifest_sha256":source_sha,"training":{}},selection.candidate["asset"]["path"])
    item = deepcopy(selection.candidate)
    item.update(kind=kind,regime="weak" if kind=="weak_native" else "strong")
    item["asset"]["sha256"] = driver.file_sha(item["asset"]["path"])
    rows,_ = driver.generation_plan(item,{"collision_nu_range":[.02,.08],"temperature_range":[.8,1.2]},"d"*64,9,set())
    item.update(rows=rows,generated_dir=str(tmp_path/"generated"))
    calls=[]
    def sample(conditions,seeds,*,steps):
        assert steps==64
        calls.append(seeds[0])
        if seeds[0]==rows[13]["seed"]: raise RuntimeError("deliberate failed attempt, no replacement")
        return torch.ones((1,6),dtype=torch.float32)
    sampler = SimpleNamespace(identity={"latent_dimension":6},model=SimpleNamespace(sample=sample),
        normalizer=SimpleNamespace(decode=lambda x:x))
    loads=[]
    def load(path,digest,*,device,steps):
        loads.append((path,digest,device,steps))
        return sampler
    monkeypatch.setattr(structured_diffusion,"load_vpfp_diffusion_sampler",load)
    plan = {"candidates":[item],"selection_sha256":"d"*64,"reference_protocol":"original_coarse_NOT_refined"}
    result = driver.generate_stage(plan,item["id"],"cpu")
    directory = Path(item["generated_dir"])
    raw = np.load(directory/"generated-coefficients.npy")
    receipt = json.loads((directory/"generated-provenance.json").read_text())
    assert len(calls)==len(set(calls))==512 and result["completed_attempts"]==512 and result["failed_attempts"]==1
    assert raw.shape==(512,6) and np.isnan(raw[13]).all() and np.isfinite(np.delete(raw,13,axis=0)).all()
    assert receipt["attempted"]==512 and receipt["successful_attempts"]==511 and receipt["failed_attempts"]==1
    assert len(receipt["condition_seeds"])==len(receipt["future_noise_seeds"])==512
    assert receipt["future_noise_seeds"]==receipt["seeds"]  # Plain native EDM directly uses those fresh root streams.
    assert loads==[(item["asset"]["path"],item["asset"]["sha256"],"cpu",64)]
    assert receipt["schema"]=="vpfp_independent_native_generation_v1"
    assert receipt["checkpoint"]==item["asset"] and receipt["checkpoint_sha256"]==item["asset"]["sha256"]
    assert receipt["model_source_manifest_sha256"]==source_sha
    assert not (directory/"generated-macro.npy").exists()
    assert len((directory/"generation-progress.jsonl").read_text().splitlines())==512


@pytest.fixture
def orchestration(tmp_path,monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID","synthetic-no-real-compute")
    monkeypatch.setenv("SLURM_TMPDIR",str(tmp_path))
    monkeypatch.setattr(driver,"SCRATCH_ROOT",tmp_path)
    output = tmp_path/"output"
    candidates = [{"id":"strong-one","regime":"strong","kind":"strong_native","model_source":{"corpus":str(tmp_path/"model/corpus")},
        "generated_dir":str(output/"generated/strong-one")}]
    plan = {"schema":driver.SCHEMA,"sample_count":512,"sampler_steps":64,"frozen_files":{},"candidates":candidates,
        "physics_design":{"physics_population_scope":"8 fixed synthetic roots not all512"},
        "references":{"strong":{"model_source":{"corpus":str(tmp_path/"model/corpus")},"reference_source":str(output/"strong-reference/corpus")}}}
    monkeypatch.setattr(driver,"load_selection",lambda *args:({"candidates":candidates},{},set()))
    monkeypatch.setattr(driver,"prepare_plan",lambda *args:plan)
    calls=[]
    def child(command,log,deadline):
        calls.append(list(command))
        if "generate" in command:
            path = Path(candidates[0]["generated_dir"])
            path.mkdir(parents=True)
            (path/"generated-provenance.json").write_text("{}")
        if "physics_controls" in command:
            path = output/"physics-strong"
            path.mkdir()
            (path/"result.json").write_text(json.dumps({"execution_passed":True}))
        return {"exit_code":0,"timed_out":False,"actual_wall_seconds":.1,"command":command}
    monkeypatch.setattr(driver,"run_child",child)
    args = SimpleNamespace(output=output,selection=tmp_path/"selection.json",selection_sha256="a"*64,seed=7,wall_seconds=120.,device="cpu")
    return args,calls,plan


def test_complete_handoff_never_claims_full_physical_or_refined_confirmation(orchestration):
    args,calls,_ = orchestration
    result = driver.run(args)
    assert result["execution_passed"] and result["data_handoff_complete"]
    assert not result["confirmation_512_completed"] and not result["physical_regime_qualified"] and not result["scientific_admission"]
    assert len(calls)==4 and calls[0][calls[0].index("--internal-stage")+1]=="collect"
    assert calls[1][calls[1].index("--internal-stage")+1]=="generate"
    audit = calls[2]
    assert audit[audit.index("--model-source")+1]!=audit[audit.index("--reference-source")+1]
    assert audit[audit.index("--sample-count")+1]=="512"
    assert calls[3][calls[3].index("--internal-stage")+1]=="physics_controls"
    assert result["descriptive_coverage_512_completed"] and result["physical_subset_execution_completed"]
    assert result["reference_refinement_subset_completed"] and "strong" in result["physics_subsets"]


@pytest.mark.parametrize("timed_out",[False,True])
def test_stage_failure_retains_partial_and_never_downgrades_count(orchestration,monkeypatch,timed_out):
    args,calls,_ = orchestration
    def failed(command,log,deadline):
        calls.append(command)
        Path(log).write_text("retained partial collector failure")
        return {"exit_code":-15 if timed_out else 1,"timed_out":timed_out,"actual_wall_seconds":.1}
    monkeypatch.setattr(driver,"run_child",failed)
    result = driver.run(args)
    assert not result["execution_passed"] and not result["data_handoff_complete"] and result["sample_count"]==512
    assert len(calls)==1 and result["failure"] and (args.output/"result.json").exists()
    assert not (args.output/"independent-physics-handoff.json").exists()


def test_global_deadline_prevents_starting_a_new_process(monkeypatch,tmp_path):
    monkeypatch.setattr(driver.time,"monotonic",lambda:10.)
    monkeypatch.setattr(driver.subprocess,"Popen",lambda *args,**kwargs:pytest.fail("must not launch"))
    with pytest.raises(TimeoutError): driver.run_child(["unused"],tmp_path/"log",9.)
    assert not (tmp_path/"log").exists()


def test_timeout_only_terminates_owned_process_group_and_keeps_log(monkeypatch,tmp_path):
    class Process:
        pid=1729
        def __init__(self): self.calls=0
        def wait(self,timeout=None):
            self.calls+=1
            if self.calls==1: raise subprocess.TimeoutExpired("synthetic",timeout)
            return -15
    process = Process(); groups=[]
    monkeypatch.setattr(driver.subprocess,"Popen",lambda *args,**kwargs:process)
    monkeypatch.setattr(driver.os,"killpg",lambda *args:groups.append(args))
    monkeypatch.setattr(driver.time,"monotonic",lambda:10.)
    result = driver.run_child(["synthetic-command"],tmp_path/"log",11.)
    assert result["timed_out"] and result["exit_code"]==-15
    assert groups==[(1729,driver.signal.SIGTERM)] and (tmp_path/"log").exists()


def test_driver_has_no_scheduler_or_training_action():
    source = (ROOT/"scripts/run_vpfp_independent_confirmation.py").read_text()
    assert "sbatch" not in source and "srun" not in source and "shared.train(" not in source
    assert "COUNT = 512" in source and '"--sample-count",str(COUNT)' in source
