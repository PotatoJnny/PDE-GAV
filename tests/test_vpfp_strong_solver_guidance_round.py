"""Tiny real-guide train/restore and saved-rollout ABI tests.

Synthetic mechanics only; run in CPU/GPU compute gates, never on login.
No real checkpoint, solver call, PDE solve, or scheduler query is used here.
"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("strong_guidance_round_tests",ROOT/"scripts/run_vpfp_strong_solver_guidance_round.py")
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
spec=importlib.util.spec_from_file_location("strong_guidance_round_rollouts",ROOT/"scripts/vpfp_strong_guidance_rollouts.py")
rollouts=importlib.util.module_from_spec(spec);spec.loader.exec_module(rollouts)


@pytest.fixture(autouse=True)
def one_thread():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config():
    return json.loads((ROOT/"configs/vpfp_strong_solver_guidance_20260908.json").read_text())


def test_registered_config_is_consistent_with_real_rollout_contract():
    cfg=config()
    assert driver.validate_config(cfg) is cfg
    roots=rollouts.registered_roots(cfg)
    assert {key:len(value) for key,value in roots.items()}==cfg["counts"]
    assert sum(map(len,roots.values()))==128
    assert len({r["root_seed"] for rows in roots.values() for r in rows})==128
    assert cfg["base_weights_trainable"] is False and cfg["solver_weights_or_code_trainable"] is False


@pytest.mark.parametrize("bad",["schema","count","base","solver","failed_reward","old_labels",
    "negative_strength","duplicate_strength","missing_baseline","preflight_zero"])
def test_bad_round_contract_is_rejected(bad):
    cfg=config()
    if bad=="schema":cfg["schema"]="wrong"
    elif bad=="count":cfg["counts"]["fit"]=7
    elif bad=="base":cfg["base_weights_trainable"]=True
    elif bad=="solver":cfg["solver_weights_or_code_trainable"]=True
    elif bad=="failed_reward":cfg["failed_solver_outputs_are_high_rewards"]=True
    elif bad=="old_labels":cfg["old_scores_used_as_training_labels"]=True
    elif bad=="negative_strength":cfg["sampling"]["selection_strengths"]=[0.,-1.]
    elif bad=="duplicate_strength":cfg["sampling"]["selection_strengths"]=[0.,.5,.5]
    elif bad=="missing_baseline":cfg["sampling"]["selection_strengths"]=[.5,2.]
    elif bad=="preflight_zero":cfg["preflight_fit_count"]=0
    with pytest.raises(ValueError):driver.validate_config(cfg)


def test_selection_uses_paired_actual_error_without_rewarding_failed_or_dropped_roots():
    baseline={"a":{"valid":True,"error":1.},"b":{"valid":True,"error":2.},
        "c":{"valid":False,"error":{"type":"SolverFailed"}}}
    results={"0":baseline,
        "0.5":{"a":{"valid":True,"error":1.4},"b":{"valid":True,"error":2.4},
            "c":{"valid":False,"error":{"type":"SolverFailed"}}},
        "2":{"a":{"valid":True,"error":100.},"b":{"valid":False,"error":{"type":"Timeout"}},
            "c":{"valid":True,"error":1000.}}}
    report=driver.select_strength(results,[0.,.5,2.])
    assert report["selected_strength"]==.5 and not report["scientific_admission"]
    assert report["rows"][1]["paired_mean_error_increase"]==pytest.approx(.4)
    assert report["rows"][2]["eligible_without_lost_baseline_roots"] is False
    assert report["rows"][2]["paired_valid"]==1
    # Even arbitrarily good results on a changed root inventory cannot win.
    results["2"]["b"]={"valid":True,"error":100.};results["2"]["new"]={"valid":True,"error":1000.}
    assert driver.select_strength(results,[0.,.5,2.])["selected_strength"]==.5


def test_selection_retains_zero_if_no_positive_eligible_gain_or_no_valid_baseline():
    for valid in (True,False):
        baseline={"a":{"valid":valid,"error":1. if valid else None}}
        result=driver.select_strength({"0":baseline,"1":{"a":{"valid":True,"error":.9}}},[0.,1.])
        assert result["selected_strength"]==0.
        assert result["rows"][1]["eligible_without_lost_baseline_roots"] is valid


def saved_labels(tmp_path, *, valid_count=8, include_finite_failure=True, bad_input=False):
    """Persist the actual private-draw field inventory, not a mock loader."""
    from tests.test_vpfp_initial_conditioned_diffusion import fixture_config
    cfg=fixture_config();entries={}
    count=valid_count+int(include_finite_failure)+int(bad_input)
    rows=rollouts.registered_roots(config())["fit"][:count]
    for index,row in enumerate(rows):
        path=tmp_path/row["slot_id"];path.mkdir()
        sample={"schema":rollouts.SCHEMA,"row":row,"guidance":None,
            "full":torch.zeros((1,cfg.full_dimension)),
            "initial_context":torch.full((1,cfg.context_dimension),float(index)),
            "future_white":torch.full((1,cfg.latent_dimension),float(index)+.5),
            "equation_conditions":torch.tensor([row["conditions"]],dtype=torch.float32),
            "root_seeds":[row["root_seed"]],
            "known_prior_seeds":[rollouts._seed(f"vpfp-initial-conditioned-prior:{row['root_seed']}")],
            "future_noise_seeds":[rollouts._seed(f"vpfp-initial-conditioned-future:{row['root_seed']}")]}
        if bad_input and index==count-1:sample["future_white"][0,0]=float("nan")
        torch.save(sample,path/"private-draw.pt")
        valid=index<valid_count
        report={"schema":rollouts.SCHEMA,"row":row,"stage":"completed" if valid else "solver_failed",
            "execution_passed":valid,"valid":valid,"reward_eligible":valid,"scientific_admission":False,
            "validity_label_available":True,
            "label_kind":"valid_solver_error" if valid else "numerical_solver_failure",
            "error":float(index+1) if valid else {"type":"SolverFailed","message":"synthetic retained failure"},
            "private_draw":{"path":str(path/"private-draw.pt"),"sha256":driver.file_sha(path/"private-draw.pt")}}
        driver.write(path/"result.json",report)
        entries[row["slot_id"]]={"result_path":str(path/"result.json"),"result_sha256":driver.file_sha(path/"result.json")}
    return entries


def test_saved_private_draw_rows_keep_finite_failed_inputs_for_validity_only(tmp_path):
    entries=saved_labels(tmp_path)
    data=driver.load_training_rows(entries)
    assert data["root_ids"]==sorted(entries)
    assert data["valid"].tolist()==[True]*8+[False]
    assert data["errors"][:8].tolist()==list(range(1,9)) and torch.isnan(data["errors"][-1])
    assert data["future_white"].shape[0]==9 and torch.isfinite(data["future_white"]).all()
    assert data["conditions"].shape==(9,2)
    assert data["initial_context"][3,0]==3.


def test_failed_nonfinite_draw_is_retained_but_excluded_from_learnable_inputs(tmp_path):
    entries=saved_labels(tmp_path,bad_input=True)
    data=driver.load_training_rows(entries)
    assert data["registered_count"]==10 and data["used_input_count"]==9
    assert len(data["excluded_rows"])==1 and data["excluded_rows"][0]["slot_id"]==sorted(entries)[-1]
    assert data["excluded_rows"][0]["reason"]
    assert torch.isfinite(data["future_white"]).all() and data["valid"].tolist()==[True]*8+[False]
    # Exclusion is not deletion of the underlying failed artifact.
    assert all(Path(item["result_path"]).is_file() for item in entries.values())


def test_changed_private_input_and_insufficient_valid_labels_are_rejected(tmp_path):
    first=tmp_path/"first";first.mkdir()
    entries=saved_labels(first)
    report=json.loads(Path(entries[sorted(entries)[0]]["result_path"]).read_text())
    Path(report["private_draw"]["path"]).write_bytes(b"synthetic modified private draw")
    with pytest.raises(ValueError,match="private.*changed"):driver.load_training_rows(entries)
    second=tmp_path/"second";second.mkdir()
    with pytest.raises(ValueError,match="at least8"):
        driver.load_training_rows(saved_labels(second,valid_count=7))


def test_infrastructure_failure_is_retained_but_not_a_solver_validity_label(tmp_path):
    entries=saved_labels(tmp_path)
    slot_id=sorted(entries)[-1];entry=entries[slot_id]
    path=Path(entry["result_path"]);report=json.loads(path.read_text())
    report.update(validity_label_available=False,label_kind="unavailable",
        label_unavailable_reason="public_materialization_failed")
    driver.write(path,report);entry["result_sha256"]=driver.file_sha(path)
    data=driver.load_training_rows(entries)
    assert data["registered_count"]==9 and data["used_input_count"]==8
    assert data["valid"].tolist()==[True]*8
    assert data["excluded_rows"]==[{"slot_id":slot_id,"reason":"public_materialization_failed"}]
    assert path.is_file() and Path(report["private_draw"]["path"]).is_file()


@pytest.mark.parametrize("available",[True,False])
def test_completed_label_survives_allocation_stop_without_turning_it_into_failure(tmp_path,available):
    entries=saved_labels(tmp_path,include_finite_failure=False)
    measured=entries[sorted(entries)[0]].copy()
    path=Path(measured["result_path"]);report=json.loads(path.read_text())
    report.update(validity_label_available=available,resident_released=True)
    driver.write(path,report);measured.update(report,result_sha256=driver.file_sha(path))
    error=TimeoutError("allocation ending");error.rollout_result=measured
    slot={"status":"running"};attempt={"status":"running"}
    driver.retain_interrupted_rollout(slot,attempt,error)
    assert slot["status"]==("completed" if available else "interrupted")
    assert attempt["status"]==("completed_before_allocation_interruption" if available else "interrupted")
    if available:
        assert slot["valid"] is True and slot["error"]==1.
        driver.verify_bound_result(slot)


def assert_same(left,right):
    if isinstance(left,torch.Tensor):assert torch.equal(left.cpu(),right.cpu())
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left:assert_same(left[key],right[key])
    elif isinstance(left,(tuple,list)):
        assert len(left)==len(right)
        for a,b in zip(left,right):assert_same(a,b)
    else:assert left==right


@pytest.mark.parametrize("device",["cpu","cuda"])
def test_actual_tiny_guide_continuous_equals_interrupted_restore_all_state(tmp_path,device):
    if device=="cuda" and not torch.cuda.is_available():pytest.skip("requires allocated CUDA device")
    from tests.test_vpfp_initial_conditioned_diffusion import data as initial_data
    from pde_agent_verifier_solver.plugins.vpfp.initial_conditioned_diffusion import VPFPInitialConditionedEDM
    from pde_agent_verifier_solver.plugins.vpfp.solver_error_guidance import _module_sha
    cfg0,raw,norm=initial_data(dtype=torch.float32)
    base=SimpleNamespace(model=VPFPInitialConditionedEDM(cfg0).to(device).eval().requires_grad_(False),normalizer=norm.to(device))
    context,future=norm.encode(raw.to(device))
    theta=torch.tensor([[12.,.9],[20.,1.],[28.,1.1],[15.,1.]]*2,device=device)
    fit={"initial_context":context.detach().cpu(),"future_white":future.detach().cpu(),"conditions":theta.cpu(),
        "errors":torch.tensor([0.,.1,.4,1.,2.,3.,4.,float("nan")],dtype=torch.float64),
        "valid":torch.tensor([True]*7+[False]),"root_ids":[f"synthetic-fit-{i}" for i in range(8)]}
    validation={"initial_context":context[:4].detach().cpu()+.03,"future_white":future[:4].detach().cpu()-.02,
        "conditions":theta[:4].cpu(),"errors":torch.tensor([.1,.3,1.,float("nan")],dtype=torch.float64),
        "valid":torch.tensor([True,True,True,False]),"root_ids":[f"synthetic-validation-{i}" for i in range(4)]}
    cfg=config();cfg["training"].update(width=4,depth=1,steps=4,checkpoint_every=2,batch_size=2)
    before_base=_module_sha(base.model);before_norm=driver.json_sha(norm.to_payload())
    whole=tmp_path/"whole"
    guide=driver.train_surrogate(cfg,base,fit,validation,whole,"e"*64,lambda:None)
    assert all(not p.requires_grad for p in guide.parameters())
    split=tmp_path/"split";calls=0
    def interrupt():
        nonlocal calls
        calls+=1
        if calls==3:raise TimeoutError("synthetic interruption after saved step2")
    with pytest.raises(TimeoutError):driver.train_surrogate(cfg,base,fit,validation,split,"e"*64,interrupt)
    midpoint=torch.load(split/"continuation.pt",map_location="cpu",weights_only=True)
    assert midpoint["step"]==2 and (split/"best.pt").is_file() and (split/"last.pt").is_file()
    restored=driver.train_surrogate(cfg,base,fit,validation,split,"e"*64,lambda:None)
    left=torch.load(whole/"continuation.pt",map_location="cpu",weights_only=True)
    right=torch.load(split/"continuation.pt",map_location="cpu",weights_only=True)
    assert_same(left,right)
    for name in ("best.pt","last.pt"):
        assert_same(torch.load(whole/name,map_location="cpu",weights_only=True),torch.load(split/name,map_location="cpu",weights_only=True))
    assert_same(guide.state_dict(),restored.state_dict())
    assert left["step"]==4 and len(left["history"])==2
    assert _module_sha(base.model)==before_base and driver.json_sha(norm.to_payload())==before_norm
    assert all(p.grad is None for p in base.model.parameters())
    with pytest.raises(ValueError,match="current-label/config"):
        driver.train_surrogate(cfg,base,fit,validation,split,"f"*64,lambda:None)


def test_ambient_float64_trains_reloads_and_guides_the_frozen_fp32_base(tmp_path,monkeypatch):
    from tests.test_vpfp_solver_error_guidance import native
    from pde_agent_verifier_solver.plugins.vpfp import solver_error_guidance as sg
    previous=torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        base,_,context,future,theta,_=native(monkeypatch)
        fit={"initial_context":context,"future_white":future,"conditions":theta,
            "errors":torch.tensor([.1,1.],dtype=torch.float64),"valid":torch.tensor([True,True]),
            "root_ids":["dtype-fit-0","dtype-fit-1"]}
        validation={**fit,"initial_context":context+.01,"root_ids":["dtype-val-0","dtype-val-1"]}
        cfg=config();cfg["base_checkpoint_sha256"]="e"*64
        cfg["training"].update(width=4,depth=1,steps=2,checkpoint_every=1,batch_size=2)
        base_sha=sg._module_sha(base.model);norm_sha=driver.json_sha(base.normalizer.to_payload())
        torch.set_default_dtype(torch.float64)
        directory=tmp_path/"ambient64"
        guide=driver.train_surrogate(cfg,base,fit,validation,directory,"a"*64,lambda:None)
        assert torch.get_default_dtype()==torch.float64
        assert {p.dtype for p in guide.parameters()}=={torch.float32}
        assert {b.dtype for b in guide.transform.buffers()}=={torch.float64}
        assert all(not p.requires_grad for p in guide.parameters())
        guided=sg.GuidedIC40Sampler(base,guide,strength=.1)
        sample=guided.sample_future(context[:1],theta[:1],[211])
        assert sample.dtype==torch.float32 and torch.isfinite(sample).all()
        assert guided.last_report["nonzero_gradient_calls"]>0
        assert sg._module_sha(base.model)==base_sha and driver.json_sha(base.normalizer.to_payload())==norm_sha
        assert all(p.grad is None for p in base.model.parameters())
        continuation=torch.load(directory/"continuation.pt",map_location="cpu",weights_only=True)
        assert continuation["step"]==2
        continuation["model"]["input.weight"]=continuation["model"]["input.weight"].double()
        torch.save(continuation,directory/"continuation.pt")
        with pytest.raises(ValueError,match="consistent floating dtype"):
            driver.train_surrogate(cfg,base,fit,validation,directory,"a"*64,lambda:None)
        assert torch.get_default_dtype()==torch.float64
    finally:
        torch.set_default_dtype(previous)
