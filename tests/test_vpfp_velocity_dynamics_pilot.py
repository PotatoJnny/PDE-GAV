"""Tiny shared-trainer and orchestration tests; run only in compute gates."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
loader=importlib.util.spec_from_file_location("velocity_pilot_tests",ROOT/"scripts/run_vpfp_velocity_dynamics_pilot.py")
pilot=importlib.util.module_from_spec(loader); loader.loader.exec_module(pilot)
loader=importlib.util.spec_from_file_location("velocity_macro_fixture",Path(__file__).with_name("test_vpfp_coherent_velocity_diffusion.py"))
fixture=importlib.util.module_from_spec(loader); loader.loader.exec_module(fixture)
from pde_agent_verifier_solver.plugins.vpfp.diffusion_data import CorpusSplit


def same(a,b):
    if torch.is_tensor(a): assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a: same(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b): same(x,y)
    else: assert a==b


def test_cli_registered_plan_and_domains():
    args=pilot.parser().parse_args(["--config","fixture.json","--output","/scratch/fixture","--seed","73","--expected-preflight-sha256","a"*64])
    assert (args.steps,args.replicas,args.batch_size,args.generated_answers,args.sample_steps)==(5000,2,4,128,64)
    assert pilot.ARMS==("nodes33","drive4","drive8")
    values={pilot.seed_for(domain,73,index) for domain in ("generation","other") for index in range(128)}
    assert len(values)==256


@pytest.mark.parametrize("problem",[None,"sha","schema","incomplete","solves","strata","case"])
def test_raw_reference_execution_receipt_is_not_scientific_approval(tmp_path,problem):
    report={"schema":"vpfp_weak_two_root_raw_time_velocity_preflight_v1","execution_passed":True,
        "completed_pde_solves":6,"scientific_admission":False,
        "cases":[{"root":{"stratum_id":i},"execution_passed":True} for i in (0,7)]}
    if problem=="schema": report["schema"]="wrong"
    if problem=="incomplete": report["execution_passed"]=False
    if problem=="solves": report["completed_pde_solves"]=5
    if problem=="strata": report["cases"][1]["root"]["stratum_id"]=6
    if problem=="case": report["cases"][1]["execution_passed"]=False
    path=tmp_path/"receipt.json"; path.write_text(json.dumps(report))
    digest="0"*64 if problem=="sha" else pilot.shared.file_sha(path)
    if problem:
        with pytest.raises(ValueError): pilot.require_raw_reference(path,digest)
    else:
        result=pilot.require_raw_reference(path,digest)
        assert result["execution_prerequisite_only"] and not result["scientific_approval_inferred"]
    assert pilot.require_raw_reference(None,None,prepare_only=True) is None
    with pytest.raises(ValueError,match="requires"): pilot.require_raw_reference(None,None)


def test_baseline_factory_keeps_original_EDM_loss():
    cfg,_,macro,_,parent=fixture.fixture()
    options=pilot.training_factories(cfg.base_macro_config,{"train":macro.numpy()},parent,"/synthetic")
    assert options["model_factory"] is fixture.VPFPCoherentMacroEDM
    assert "training_metric_factory" not in options
    assert "validation_metric_factory" not in options


@pytest.mark.parametrize("device",["cpu","cuda"])
@pytest.mark.parametrize("degree",[4,8])
def test_real_shared_train_continuous_equals_split_all_state(monkeypatch,tmp_path,degree,device):
    if device=="cuda" and not torch.cuda.is_available(): pytest.skip("CUDA exact continuation requires compute GPU")
    old_threads=torch.get_num_threads(); torch.set_num_threads(1)
    try:
        cfg,raw,macro,condition,parent=fixture.fixture(degree)
        parent=fixture.VPFPCoherentMacroNormalizer.fit(cfg.base_macro_config,macro[:4],sample_ids=[0,1,2,3])
        manifest={"latent_dimension":cfg.full_dimension,"manifest_sha256":"c"*64,"base_decoder_sha256":"b"*64}
        splits={}
        for role,sl in (("train",slice(0,4)),("validation",slice(4,6))):
            ids=np.arange(6)[sl]
            splits[role]=CorpusSplit(macro[sl].numpy(),condition[sl].numpy(),ids,ids.astype(str),np.zeros(len(ids),dtype=int),manifest)
        arrays={"train":raw[:4].numpy(),"validation":raw[4:].numpy()}
        monkeypatch.setattr(pilot,"load_split",lambda path,role,**kw:splits[role])
        monkeypatch.setattr(pilot.shared,"read_manifest",lambda *a,**kw:manifest)
        monkeypatch.setattr(pilot.shared,"source_fingerprints",lambda:{"fixture":"d"*64})
        monkeypatch.setenv("SLURM_JOB_ID","synthetic-test")
        def run(name,steps,resume=None):
            output=tmp_path/name; output.mkdir()
            args=SimpleNamespace(seed=83,device=device,batch_size=2,steps=steps,train_seconds=10000,
                validate_every=1,learning_rate=2.e-4,source_revision="synthetic",
                ablation_metadata={"data_transform_sha256":"e"*64})
            if resume:
                args.resume_checkpoint=resume/"checkpoints/last.pt"
                args.resume_best_checkpoint=resume/"checkpoints/best.pt"
                args.expected_resume_sha256=pilot.shared.file_sha(args.resume_checkpoint)
                args.expected_resume_best_sha256=pilot.shared.file_sha(args.resume_best_checkpoint)
            options=pilot.training_factories(cfg,arrays,parent,tmp_path/"corpus")
            if resume:
                def no_refit(*a,**kw): raise AssertionError("resume must load frozen train normalizer")
                options["normalizer_factory"]=no_refit
            pilot.shared.train(args,SimpleNamespace(base_decoder_sha256="b"*64),tmp_path/"corpus",output,**options)
            return output
        whole=run("whole",4); first=run("first",2); resumed=run("resumed",4,first)
        left=torch.load(whole/"checkpoints/last.pt",weights_only=True,map_location="cpu")
        right=torch.load(resumed/"checkpoints/last.pt",weights_only=True,map_location="cpu")
        for key in ("model","normalizer","continuation_state"): same(left[key],right[key])
        assert right["training"]["step"]==4 and right["training"]["continuation"]["normalizer_refitted"] is False
    finally: torch.set_num_threads(old_threads)


def test_original_cubic_and_OU_are_reported_separately():
    cfg,raw,macro,condition,_=fixture.fixture(8)
    report=pilot.velocity_diagnostics(macro[:1],condition[:1],cfg.base_macro_config,raw=raw[:1],dynamics_config=cfg)
    assert set(("65","129"))<=report.keys()
    assert "ou_vs_cubic_drive_rms" in report["129"]
    assert report["scientific_admission"] is False
    assert "not independently integrated" in report["source_interpretation"]


def test_unallocated_execution_fails_before_preparation(monkeypatch,tmp_path):
    monkeypatch.delenv("SLURM_JOB_ID",raising=False)
    with pytest.raises(ValueError,match="Slurm"):
        pilot.run(SimpleNamespace(output=tmp_path/"unused"))
    assert not (tmp_path/"unused").exists()


@pytest.mark.parametrize("failure",["draw","deadline"])
def test_128_attempt_contract_preserves_failures_and_unrun_rows(monkeypatch,tmp_path,failure):
    cfg,_,_,_,_=fixture.fixture()
    base=cfg.base_macro_config
    class Model:
        def sample(self,condition,seeds,steps):
            if failure=="draw" and seeds==[3]: raise RuntimeError("synthetic draw failure")
            return torch.zeros((1,base.latent_dimension))
    monkeypatch.setattr(pilot,"load_fixed",lambda *a:(Model(),SimpleNamespace(decode=lambda x:x)))
    monkeypatch.setattr(pilot,"velocity_diagnostics",lambda *a,**kw:{"synthetic_fixture":True})
    count=0
    def guard():
        nonlocal count
        count+=1
        if failure=="deadline" and count==3: raise TimeoutError("synthetic deadline")
    output=tmp_path/"draws"
    args=SimpleNamespace(device="cpu")
    call=lambda:pilot.sample_attempts(args,{"checkpoint":"synthetic"},base,{},
        np.tile(np.array([[.03,1.]],dtype=np.float32),(128,1)),list(range(128)),output,guard)
    if failure=="deadline":
        with pytest.raises(TimeoutError): call()
    else: call()
    receipt=json.loads((output/"generated-receipt.json").read_text())
    assert receipt["attempted_count"]==128 and receipt["registered_attempt_root_count"]==128
    assert len(set(receipt["root_ids"]))==128
    if failure=="draw":
        assert receipt["valid_count"]==127 and receipt["failed_count"]==1
        assert receipt["rows"][3]["failure"]["reason"]=="synthetic draw failure"
        assert np.isnan(np.load(output/"macro.npy")[3]).all()
    else:
        assert receipt["completed_attempt_count"]==2 and receipt["unrun_attempt_count"]==126
        assert receipt["independent_valid_root_count"]==2


def test_parent_registers_six_groups_and_links_timed_out_child(tmp_path):
    entries=[{"seed_index":i,"arm":arm} for i in range(2) for arm in pilot.ARMS]
    result={}; saved=[]
    def callback(entry):
        path=tmp_path/f"seed-{entry['seed_index']}"/entry["arm"]/"generated/generated-receipt.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"attempted_count":128,"completed_attempt_count":2,
            "valid_count":1,"failed_count":1,"unrun_attempt_count":126}))
        raise TimeoutError("synthetic child timeout")
    with pytest.raises(TimeoutError):
        pilot.sample_registered_groups(entries,tmp_path,callback,result,lambda:saved.append(deepcopy(result)))
    assert len(result["generated"])==6
    first=result["generated"][0]
    assert first["status"]=="partial_or_failed" and first["unrun_attempt_count"]==126
    assert Path(first["receipt_path"]).is_file() and len(first["receipt_sha256"])==64
    assert all(x["status"]=="pending" and x["unrun_attempt_count"]==128 for x in result["generated"][1:])
    assert all(x["status"]=="pending" for x in saved[0]["generated"])
