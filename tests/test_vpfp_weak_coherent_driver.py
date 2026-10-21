"""Data-flow checks: never replace generated macros by held-out oracle values."""
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("weak_coherent_driver_test",ROOT/"scripts/run_vpfp_weak_coherent.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


@dataclass
class Split:
    coefficients: object
    conditions: object
    case_ids: object
    sample_ids: object = None
    @property
    def latents(self):
        return self.coefficients
    def __len__(self):
        return len(self.coefficients)


def test_training_views_preserve_rows_and_conditions(monkeypatch,tmp_path):
    split = Split(np.arange(12).reshape(3,4),np.array([[.03,1.],[.04,1.],[.05,1.]]),["a","b","c"])
    monkeypatch.setattr(driver,"load_split",lambda *a,**k:split)
    m = {"train":np.arange(18).reshape(3,6).astype("float32")}
    macro = driver.loader_with_view(tmp_path,m,mode="macro")(tmp_path,"train")
    np.testing.assert_equal(macro.coefficients,m["train"])
    np.testing.assert_equal(macro.conditions,split.conditions)
    class Normalizer:
        def encode(self,x):
            return 2*x
    conditional_loader = driver.loader_with_view(tmp_path,m,mode="conditional",normalizer=Normalizer())
    conditional = conditional_loader(tmp_path,"train")
    np.testing.assert_equal(conditional.coefficients,split.coefficients)
    np.testing.assert_equal(conditional.conditions[:,:2],split.conditions)
    np.testing.assert_equal(conditional.conditions[:,2:],2*m["train"])
    assert conditional.case_ids == split.case_ids
    with pytest.raises(ValueError,match="train/validation"):
        conditional_loader(tmp_path,"reference_a")


def test_representation_selection_requires_bound_real_mechanics(monkeypatch,tmp_path):
    monkeypatch.setattr(driver,"read_manifest",lambda *a,**k:{"manifest_sha256":"a"*64})
    evidence = tmp_path/"real-preflight.json"
    evidence.write_text(json.dumps({"numerical_mechanics_passed":True,"source_manifest_sha256":"a"*64}))
    choice = {"schema":"vpfp_moving_frame_representation_selection_v1","mechanics_passed":True,
        "selected_mode":6,"selected_fit_grid":33,"compression_review":"reviewed node and quarter-point errors separately",
        "corpus_manifest_sha256":"a"*64,"preflight_result":str(evidence),
        "preflight_result_sha256":driver.shared.file_sha(evidence)}
    path = tmp_path/"selection.json"
    path.write_text(json.dumps(choice))
    assert driver.require_selection(path,tmp_path)["selected_mode"] == 6
    evidence.write_text(json.dumps({"numerical_mechanics_passed":False,"source_manifest_sha256":"a"*64}))
    with pytest.raises(ValueError,match="changed"):
        driver.require_selection(path,tmp_path)
    choice["preflight_result_sha256"] = driver.shared.file_sha(evidence)
    path.write_text(json.dumps(choice))
    with pytest.raises(ValueError,match="did not pass"):
        driver.require_selection(path,tmp_path)


def test_campaign_gpu_reservations_are_48_and_preflight_is_not_extra():
    plan = json.loads((ROOT/"configs/vpfp_regime_coherent_campaign_20260907.json").read_text())
    assert sum(plan["stage_gpu_hour_caps"].values()) == plan["maximum_gpu_hours"] == 48
    assert plan["stage_gpu_hour_caps"]["preflight"]+plan["stage_gpu_hour_caps"]["audit_screen"] == 8
    assert plan["maximum_concurrent_gpus"] == 2
    assert not plan["scientific_admission"] and not plan["replace_online_assets"]


def test_screen_cli_has_fixed_recipe_and_explicit_selection():
    args = driver.parser().parse_args(["--config","config.json","--source","old", "--output","new",
        "--seed","1520260906","--representation-selection","selection.json"])
    assert (args.steps,args.replicas,args.batch_size,args.generated_answers,args.sample_steps) == (20000,2,4,128,64)


def test_nonregistered_batch_size_rejected_before_target_extraction(tmp_path,monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID","synthetic-gate-fixture")
    monkeypatch.setenv("SLURM_TMPDIR",str(tmp_path))
    monkeypatch.setattr(driver,"require_selection",lambda *a:pytest.fail("invalid recipe opened training data"))
    with pytest.raises(SystemExit) as raised:
        driver.main(["--config","unused.json","--source","old","--output","/scratch/vpfp-uncreated-mechanics-fixture",
            "--seed","1520260906","--representation-selection","unused-selection.json","--batch-size","2"])
    assert raised.value.code == 2


@pytest.mark.parametrize("bound_sha",[None,"0"*64])
def test_pair_rejects_unbound_or_replaced_macro_checkpoint(tmp_path,bound_sha):
    """Matching normalization alone must not permit a different macro prior."""
    from pde_agent_verifier_solver.plugins.vpfp.structured_diffusion import VPFPStructuredEDMConfig
    full = VPFPStructuredEDMConfig(control_count=3,hermite_channels=3,rho_max_mode=1,kinetic_max_mode=1,
        inactive_dim=0,grid_size=5,width=16,depth=1)
    value = driver.VPFPValueEDMConfig(full)
    macro = driver.VPFPCoherentMacroConfig(full,maximum_mode=2,coherent_mode=1)
    entry = {}
    for role,cfg,model in (("macro",macro,driver.VPFPCoherentMacroEDM),
                           ("conditional",value,driver.VPFPMacroConditionalEDM)):
        ablation = {"macro_normalizer_sha256":driver.json_sha({})}
        if bound_sha is not None:
            ablation["macro_checkpoint_sha256"] = bound_sha
        payload = {"schema":model.checkpoint_schema,"config":cfg.to_dict(),"model":{},"normalizer":{},
            "base_decoder_sha256":"b"*64,"corpus_manifest_sha256":"c"*64,
            "training":{"ablation":ablation}}
        checkpoint = tmp_path/f"{role}.pt"
        torch.save(payload,checkpoint)
        entry[role] = {"checkpoint":str(checkpoint),"checkpoint_sha256":driver.shared.file_sha(checkpoint)}
    with pytest.raises(ValueError,match="bound to this frozen macro checkpoint"):
        driver.load_pair(entry,macro,value,"cpu")


def test_native_manifest_handoff_binds_three_exact_files_without_admission(tmp_path):
    entry = {}
    for role in ("base","macro","conditional"):
        path = tmp_path/f"{role}.pt"
        path.write_bytes(f"synthetic identity fixture, not a model: {role}".encode())
        entry[role] = {"checkpoint":str(path),"checkpoint_sha256":driver.shared.file_sha(path)}
    spec = SimpleNamespace(regime_id="weak-fixture",base_decoder_checkpoint=entry["base"]["checkpoint"],
        base_decoder_sha256=entry["base"]["checkpoint_sha256"])
    receipt = driver.write_native_manifest(spec,entry,tmp_path)
    payload = json.loads(Path(receipt["manifest"]).read_text())
    assert set(payload) == {"schema","regime_id","sampler_steps","base_decoder","macro_checkpoint","conditional_checkpoint"}
    assert payload["macro_checkpoint"]["sha256"] == entry["macro"]["checkpoint_sha256"]
    assert not receipt["native_loading_and_campaign_smoke_completed"] and not receipt["scientific_admission"]
    with pytest.raises(ValueError,match="overwrite"):
        driver.write_native_manifest(spec,entry,tmp_path)


@pytest.mark.parametrize("arm",["macro","conditional"])
def test_shared_trainer_accepts_explicit_hierarchical_view(tmp_path,monkeypatch,arm):
    from pde_agent_verifier_solver.plugins.vpfp.structured_diffusion import VPFPStructuredEDMConfig
    from pde_agent_verifier_solver.plugins.vpfp.value_control_diffusion import lift_value_controls, VPFPValueControlNormalizer
    full = VPFPStructuredEDMConfig(control_count=3,hermite_channels=3,rho_max_mode=1,kinetic_max_mode=1,
        inactive_dim=0,grid_size=5,width=16,depth=1)
    value = driver.VPFPValueEDMConfig(full)
    macro = driver.VPFPCoherentMacroConfig(full,maximum_mode=2,coherent_mode=1)
    random = torch.Generator().manual_seed(119)
    raw = lift_value_controls(torch.randn(6,value.latent_dimension,generator=random),value).numpy()
    conditions = np.tile(np.array([[.04,1.]],dtype="float32"),(6,1))
    splits = {"train":Split(raw[:4],conditions[:4],["t0","t1","t2","t3"],np.arange(4)),
              "validation":Split(raw[4:],conditions[4:],["v0","v1"],np.arange(4,6))}
    arrays = {role:torch.randn(len(split),macro.latent_dimension,generator=random).numpy() for role,split in splits.items()}
    monkeypatch.setattr(driver,"load_split",lambda _,role,**kw:splits[role])
    monkeypatch.setattr(driver.shared,"read_manifest",lambda *a,**kw:{"manifest_sha256":"c"*64,
        "base_decoder_sha256":"b"*64,"latent_dimension":full.latent_dimension})
    monkeypatch.setattr(driver.shared,"source_fingerprints",lambda:{})
    monkeypatch.setenv("SLURM_JOB_ID","synthetic-mechanics-fixture")
    args = SimpleNamespace(seed=7,device="cpu",steps=2,train_seconds=60,batch_size=2,validate_every=1,
        learning_rate=2e-4,source_revision="synthetic-test",checkpoint_steps=(2,),
        ablation_metadata={"data_transform_sha256":"d"*64})
    if arm == "macro":
        cfg,make,fit = macro,driver.VPFPCoherentMacroEDM,driver.VPFPCoherentMacroNormalizer.fit
        view = driver.loader_with_view(tmp_path,arrays,mode="macro")
    else:
        norm = driver.VPFPCoherentMacroNormalizer.fit(macro,torch.tensor(arrays["train"]),sample_ids=list(range(4)))
        cfg,make,fit = value,lambda c:driver.VPFPMacroConditionalEDM(c,macro),VPFPValueControlNormalizer.fit
        view = driver.loader_with_view(tmp_path,arrays,mode="conditional",normalizer=norm)
    output = tmp_path/arm
    output.mkdir()
    _,report = driver.shared.train(args,SimpleNamespace(base_decoder_sha256="b"*64),tmp_path,output,
        model_config_factory=lambda *a:cfg,model_factory=make,normalizer_factory=fit,split_loader=view)
    assert report["final_step"] == 2
    payload = torch.load(output/"checkpoints/fixed-step-000002.pt",weights_only=True)
    assert "continuation_state" not in payload
    assert payload["training"]["presentations"] == 4
