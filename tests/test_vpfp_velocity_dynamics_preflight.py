"""Tiny synthetic cache mechanics; no real corpus, job query or PDE solve."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("velocity_dynamics_preflight_tests",ROOT/"scripts/run_vpfp_velocity_dynamics_preflight.py")
driver=importlib.util.module_from_spec(spec); spec.loader.exec_module(driver)


@pytest.fixture
def cache(tmp_path,monkeypatch):
    source=tmp_path/"source/corpus"; source.mkdir(parents=True)
    cache=tmp_path/"prior/macro-targets"; cache.mkdir(parents=True)
    monkeypatch.setattr(driver,"COUNTS",{"train":2,"validation":1})
    monkeypatch.setattr(driver,"DEGREES",(4,))
    monkeypatch.setattr(driver,"SCRATCH_ROOT",tmp_path)
    manifest={"regime_id":"long_time_weak_collision_v2","manifest_sha256":"a"*64,
        "condition_order":["collision_nu","temperature"],"time_domain":[0.,1.],"splits":{}}
    (source/"seed-plan.json").write_text("{}")
    manifest["seed_plan"]={"path":"seed-plan.json","sha256":driver.file_sha(source/"seed-plan.json")}
    prior={"arguments":{"source":str(source.parent)},"source_manifest_sha256":"a"*64,"macro_targets":{},
        "source_file_sha256":{"src/pde_agent_verifier_solver/plugins/vpfp/moving_frame.py":
            driver.file_sha(ROOT/"src/pde_agent_verifier_solver/plugins/vpfp/moving_frame.py")}}
    for role,count in driver.COUNTS.items():
        (source/role).mkdir()
        ids=list(range(10 if role=="train" else 20,(10 if role=="train" else 20)+count))
        roots=["synthetic:"+str(value) for value in ids]
        arrays={"conditions":np.tile(np.asarray([.02,1.],dtype=np.float32),(count,1)),
            "equation_seeds":np.asarray(ids,dtype=np.int64),"case_ids":np.asarray(roots),
            "stratum_ids":np.zeros(count,dtype=np.int64)}
        files={}
        for key,value in arrays.items():
            path=source/role/(key+".npy"); np.save(path,value,allow_pickle=False)
            files[key]={"path":str(path.relative_to(source)),"sha256":driver.file_sha(path)}
        manifest["splits"][role]={"count":count,"files":files}
        macro=np.zeros((count,5*33*169),dtype=np.float32)
        times=np.linspace(0,1,33,dtype=np.float32)
        for row in range(count):
            macro[row,:66]=np.stack((times*.2+row*.1,1-times*.1)).reshape(-1)
        path=cache/(role+".npy"); np.save(path,macro,allow_pickle=False)
        receipt={"file":str(path),"sha256":driver.file_sha(path),"case_ids":roots,"sample_ids":ids,
            "extraction":[{"schema":"vpfp_moving_frame_macro_extraction_v1","control_count":33,"maximum_mode":6,
                "fit_grid":33,"physical_velocity_units":True} for _ in ids]}
        (cache/(role+".json")).write_text(json.dumps(receipt)); prior["macro_targets"][role]=receipt
    (source/"manifest.json").write_text(json.dumps(manifest))
    (cache.parent/"frozen-plan.json").write_text(json.dumps(prior))
    monkeypatch.setattr(driver,"read_manifest",lambda path,verify_hashes:deepcopy(manifest))
    return cache,source,manifest,prior


def test_bound_macro_cache_reads_only_expected_roots_and_layout(cache):
    target,source,manifest,_=cache
    data,actual,frozen=driver.frozen_inputs(target,source)
    assert actual["manifest_sha256"]==manifest["manifest_sha256"]
    assert data["train"]["macro"].shape==(2,27885)
    assert data["validation"]["receipt"]["sample_ids"]==[20]
    assert str(target/"train.npy") in frozen
    assert not any("coefficients.npy" in path for path in frozen)


@pytest.mark.parametrize("change",["root","sample_id","physical_units","count","cache_hash","moving_source","source"])
def test_self_consistent_wrong_cache_or_provenance_rejected(cache,change):
    target,source,manifest,prior=cache
    if change=="root": prior["macro_targets"]["train"]["case_ids"][0]="other-root"
    elif change=="sample_id": prior["macro_targets"]["train"]["sample_ids"][0]=99
    elif change=="physical_units": prior["macro_targets"]["train"]["extraction"][0]["physical_velocity_units"]=False
    elif change=="count": manifest["splits"]["train"]["count"]=3
    elif change=="cache_hash": prior["macro_targets"]["train"]["sha256"]="b"*64
    elif change=="moving_source": prior["source_file_sha256"]["src/pde_agent_verifier_solver/plugins/vpfp/moving_frame.py"]="b"*64
    elif change=="source": prior["source_manifest_sha256"]="b"*64
    (target/"train.json").write_text(json.dumps(prior["macro_targets"]["train"]))
    (target.parent/"frozen-plan.json").write_text(json.dumps(prior))
    with pytest.raises(ValueError): driver.frozen_inputs(target,source)


def test_actual_old_surrogate_derivative_for_linear_data():
    cfg=driver.VPFPVelocityDynamicsConfig()
    t=torch.linspace(0,1,33,dtype=torch.float64)
    data=torch.stack((1+2*t,3-t))[None,:]
    off=torch.tensor([.25/32,.5/32,.75/32,31.75/32],dtype=torch.float64)
    value,dt=driver.surrogate(data,off,cfg)
    torch.testing.assert_close(value[0],torch.stack((1+2*off,3-off)),rtol=1.e-13,atol=1.e-13)
    torch.testing.assert_close(dt[0],torch.stack((torch.ones_like(off)*2,-torch.ones_like(off))),rtol=1.e-12,atol=1.e-12)


def test_small_end_to_end_preserves_counts_and_train_only_normalizer(cache,tmp_path,monkeypatch):
    target,source,_,_=cache
    monkeypatch.setenv("SLURM_JOB_ID","123"); monkeypatch.setenv("SLURM_TMPDIR",str(tmp_path))
    args=driver.parser().parse_args(["--cache",str(target),"--source",str(source),"--output",str(tmp_path/"output"),"--device","cpu","--batch-size","2"])
    report=driver.run(args)
    assert report["execution_passed"],report.get("failure")
    assert report["degrees"]["4"]["splits"]["train"]["completed_rows"]==2
    assert report["degrees"]["4"]["splits"]["validation"]["completed_rows"]==1
    normal=json.loads((tmp_path/"output/degree-4/normalizer.json").read_text())
    assert normal["fit_metadata"]["sample_ids"]==["10","11"]
    assert report["off_node_comparison_is_surrogate_only"] and not report["scientific_admission"]
    assert not report["new_training"] and report["new_PDE_solves"]==0


def test_partial_failure_keeps_first_batch_and_reason(cache,tmp_path,monkeypatch):
    target,source,_,_=cache
    monkeypatch.setenv("SLURM_JOB_ID","123"); monkeypatch.setenv("SLURM_TMPDIR",str(tmp_path))
    original=driver.analyze_batch; calls=[]
    def once(*args):
        calls.append(1)
        if len(calls)>1: raise RuntimeError("synthetic second batch failure")
        return original(*args)
    monkeypatch.setattr(driver,"analyze_batch",once)
    args=driver.parser().parse_args(["--cache",str(target),"--source",str(source),"--output",str(tmp_path/"partial"),"--batch-size","1"])
    result=driver.run(args)
    assert not result["execution_passed"] and result["failure"]["reason"]=="synthetic second batch failure"
    assert result["degrees"]["4"]["splits"]["train"]["completed_rows"]==1
    assert (tmp_path/"partial/result.json").is_file()


def test_minimal_launcher_defaults_and_compute_requirement(tmp_path,monkeypatch):
    args=driver.parser().parse_args(["--output",str(tmp_path/"not-run"),"--device","cpu"])
    assert args.cache==driver.DEFAULT_CACHE and args.source==driver.DEFAULT_SOURCE
    monkeypatch.delenv("SLURM_JOB_ID",raising=False)
    with pytest.raises(ValueError,match="Slurm"): driver.run(args)
