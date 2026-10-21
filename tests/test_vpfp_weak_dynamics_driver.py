"""Small lineage/orchestration/native-mechanics fixtures; compute-only pytest."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("weak_dynamics_driver_tests",
    ROOT/"scripts/run_vpfp_weak_dynamics_diagnostic.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def tiny_split():
    groups = np.asarray([7,0,3,5,1,6,2,4]*2,dtype=np.int64)
    return SimpleNamespace(stratum_ids=groups,
        conditions=np.asarray([[.03+g*.001,.85+g*.02] for g in groups],dtype=np.float32),
        latents=np.arange(16*4,dtype=np.float32).reshape(16,4),
        case_ids=np.asarray([f"fixture-root-{i}" for i in range(16)]),
        sample_ids=np.arange(100,116,dtype=np.int64))


def test_fixed_first_roots_keep_actual_condition_bits_and_nonindependent_interventions():
    split = tiny_split()
    rows = driver.select_validation_rows(split)
    assert [r["row"] for r in rows]==[1,4,6,2,7,3,5,0]
    for row in rows:
        assert row["condition_fp32_bits"]==split.conditions[row["row"]].view(np.uint32).tolist()
    plan = driver.fixed_noise_plan(rows)
    assert plan==driver.fixed_noise_plan(rows) and len(plan)==16
    assert len({r["micro_noise_seed"] for r in plan})==16
    assert {r["root_id"] for r in plan}=={r["root_id"] for r in rows}
    assert plan[0]["conditions"]==plan[8]["conditions"]
    assert plan[0]["micro_noise_seed"]!=plan[8]["micro_noise_seed"]


@pytest.mark.parametrize("change",["missing_stratum","float_stratum","condition_dtype","condition_nan","duplicate_root"])
def test_bad_validation_lineage_is_not_replaced(change):
    split = tiny_split()
    if change=="missing_stratum": split.stratum_ids[split.stratum_ids==7]=0
    if change=="float_stratum": split.stratum_ids=split.stratum_ids.astype(float)
    if change=="condition_dtype": split.conditions=split.conditions.astype(np.float64)
    if change=="condition_nan": split.conditions[1,0]=np.nan
    if change=="duplicate_root": split.case_ids[4]=split.case_ids[1]
    with pytest.raises(ValueError):
        driver.select_validation_rows(split)


@pytest.mark.parametrize("change",["condition","signed_zero","full"])
def test_selected_row_state_and_condition_bits_rechecked(change):
    split=tiny_split()
    if change=="signed_zero": split.conditions[1,0]=0.
    row=driver.select_validation_rows(split)[0]
    if change=="condition": split.conditions[1,1]=np.nextafter(split.conditions[1,1],np.float32(2.))
    if change=="signed_zero": split.conditions[1,0]=-0.
    if change=="full": split.latents[1,0]+=1
    with pytest.raises(ValueError,match="changed"):
        driver.condition_for_row(split,row,"cpu")


def test_real_native_value_retains_highest_H_and_rejects_slope_tamper():
    from tests.test_vpfp_moving_frame import tiny
    from pde_agent_verifier_solver.plugins.vpfp.value_control_diffusion import lift_value_controls
    base,value,config,z,condition=tiny(degree=14,dtype="float32")
    z[:,value.rho_dimension:].reshape(1,value.hermite_channels,3,9)[:,-1,:,0]=.4
    full=lift_value_controls(z,value)
    actual=driver.check_full_to_value(full,value)
    assert torch.equal(actual,z)
    bad=full.clone()
    bad[:,value.full_config.rho_dimension//2]+=1.
    with pytest.raises(ValueError):
        driver.check_full_to_value(bad,value)


def test_actual_macro_micro_sampler_uses_original_normalizers_and_identical_micro_noise():
    from tests.test_vpfp_coherent_diffusion import configs
    from pde_agent_verifier_solver.plugins.vpfp.coherent_diffusion import (
        VPFPCoherentMacroEDM,VPFPCoherentMacroNormalizer,VPFPMacroConditionalEDM,sample_hierarchical)
    from pde_agent_verifier_solver.plugins.vpfp.temporal_covariance_diffusion import VPFPTemporalCovarianceNormalizer
    from pde_agent_verifier_solver.plugins.vpfp.value_control_diffusion import lift_value_controls
    value,macro_cfg=configs()
    torch.manual_seed(14)
    raw=lift_value_controls(torch.randn(6,value.latent_dimension),value)
    macro_norm=VPFPCoherentMacroNormalizer.fit(macro_cfg,torch.randn(6,macro_cfg.latent_dimension),sample_ids=list(range(6)))
    micro_norm=VPFPTemporalCovarianceNormalizer.fit(value,raw,sample_ids=list(range(6)),
        relative_floor=.05,center_kinetic_dc=True,rho_policy="identity",covariance_strength=.5)
    sampler=SimpleNamespace(macro=VPFPCoherentMacroEDM(macro_cfg).eval(),macro_norm=macro_norm,
        micro=VPFPMacroConditionalEDM(value,macro_cfg).eval(),micro_norm=micro_norm)
    row=driver.fixed_noise_plan(driver.select_validation_rows(tiny_split()))[0]
    condition=torch.tensor(row["conditions"],dtype=torch.float32)
    m,white=driver.sample_macro(sampler,condition,row)
    z=driver.sample_micro(sampler,condition,white,row)
    original=sample_hierarchical(sampler.macro,macro_norm,sampler.micro,micro_norm,
        condition[None],[row["sampling_root_seed"]],steps=64)
    assert torch.equal(m,original["macro_coordinates"])
    assert torch.equal(lift_value_controls(z,value),original["full_coefficients"])
    calls=[]
    old_sample=sampler.micro.sample
    def record(packed,seeds,*,steps):
        calls.append((packed.detach().clone(),list(seeds),steps))
        return old_sample(packed,seeds,steps=steps)
    sampler.micro.sample=record
    real_white=macro_norm.encode(m+.2)
    driver.sample_micro(sampler,condition,real_white,row)
    driver.sample_micro(sampler,condition,white,row)
    assert calls[0][1:]==calls[1][1:]==([row["micro_noise_seed"]],64)
    assert torch.equal(calls[0][0][:,:2],calls[1][0][:,:2])
    assert torch.equal(calls[0][0][:,2:],real_white) and torch.equal(calls[1][0][:,2:],white)


def test_actual_complete_A_carrier_saved_with_condition_and_root_hashes(tmp_path):
    from tests.test_vpfp_moving_frame import tiny,constant_macro
    from pde_agent_verifier_solver.plugins.vpfp.moving_frame import MovingFrameDecoder
    base,value,cfg,z,condition=tiny(dtype="float32")
    decoder=MovingFrameDecoder(base,cfg,value_config=value)
    carrier={"m":constant_macro(cfg).float(),"z":z}
    provenance={"root_id":"physical-root","view":"generated_m_real_z","micro_conditioning_macro":None}
    receipt=driver.save_carrier(tmp_path/"carrier",decoder,carrier,condition,provenance)
    assert np.array_equal(np.load(receipt["files"]["z"]["path"]),z.numpy())
    assert receipt["complete_floor_preserved"] and not receipt["paired_truth_claimed"]
    assert receipt["root_id"]=="physical-root" and len(receipt["carrier_sha256"])==64
    with pytest.raises(FileExistsError):
        driver.save_carrier(tmp_path/"carrier",decoder,carrier,condition,provenance)


def test_freeze_pins_all_validation_files_assets_rows_and_code_before_sampling(tmp_path,monkeypatch):
    from pde_agent_verifier_solver.plugins.vpfp import diffusion_data
    source=tmp_path/"corpus"; source.mkdir()
    modelroot=tmp_path/"models"; modelroot.mkdir()
    coderoot=tmp_path/"code"; (coderoot/"scripts").mkdir(parents=True)
    for name in ("run_vpfp_independent_confirmation.py","run_vpfp_independent_physics.py","diagnose_vpfp_known_initial.py"):
        (coderoot/"scripts"/name).write_text("# fixture\n")
    def asset(path,value):
        path.write_text(json.dumps(value))
        return {"path":str(path),"sha256":driver.file_sha(path)}
    files={}
    split=tiny_split()
    arrays={"coefficients":split.latents,"conditions":split.conditions,"equation_seeds":split.sample_ids,
        "case_ids":split.case_ids,"stratum_ids":split.stratum_ids}
    for name,value in arrays.items():
        path=source/(name+".npy"); np.save(path,value)
        files[name]={"path":path.name,"sha256":driver.file_sha(path)}
    seeds=asset(source/"seed.json",{"synthetic_only":True})
    base=asset(modelroot/"base.pt",{"synthetic":True})
    manifest={"manifest_sha256":"a"*64,"regime_id":"long_time_weak_collision_v2",
        "base_decoder_sha256":base["sha256"],"seed_plan":{"path":"seed.json","sha256":seeds["sha256"]},
        "splits":{"validation":{"files":files}}}
    asset(source/"manifest.json",manifest)
    natives=[]
    for i in range(2):
        native={"schema":"vpfp_moving_frame_native_model_manifest_v1","sampler_steps":64,
            "regime_id":manifest["regime_id"],"base_decoder":base,
            "macro_checkpoint":asset(modelroot/f"macro-{i}.pt",{"synthetic":i}),
            "conditional_checkpoint":asset(modelroot/f"micro-{i}.pt",{"synthetic":i})}
        natives.append(asset(modelroot/f"native-{i}.json",native))
    monkeypatch.setattr(driver,"SOURCE",source); monkeypatch.setattr(driver,"SOURCE_SHA","a"*64)
    monkeypatch.setattr(driver,"ROOT",coderoot); monkeypatch.setattr(driver,"MODEL_MANIFESTS",natives)
    monkeypatch.setattr(driver.subprocess,"check_output",lambda command,**kwargs:"" if "status" in command else "a"*40)
    monkeypatch.setattr(diffusion_data,"read_manifest",lambda *args,**kwargs:manifest)
    monkeypatch.setattr(diffusion_data,"load_split",lambda *args,**kwargs:split)
    output=tmp_path/"output"; output.mkdir()
    plan,actual=driver.freeze_inputs(output,{"synthetic_design":True})
    assert actual is split and len(plan["selected_roots"])==8 and len(plan["noise_plan"])==16
    assert (output/"frozen-plan.json").is_file()
    for entry in files.values(): assert str(source/entry["path"]) in plan["frozen_files"]
    for item in natives: assert plan["frozen_files"][item["path"]]==item["sha256"]
    assert plan["all_inputs_and_noise_frozen_before_sampling"]
    (source/"conditions.npy").write_bytes(b"tampered")
    with pytest.raises(ValueError,match="changed"): driver.check_frozen(plan)


@pytest.mark.parametrize("fail_macro",[False,True])
def test_cross_orchestration_preserves_pair_conditions_noise_deep_subset_and_failures(tmp_path,monkeypatch,fail_macro):
    from pde_agent_verifier_solver.plugins.vpfp import moving_frame_binding,moving_frame,weak_dynamics_diagnostic as helper
    monkeypatch.setenv("SLURM_JOB_ID","synthetic-compute"); monkeypatch.setenv("SLURM_TMPDIR",str(tmp_path))
    monkeypatch.setattr(driver,"SCRATCH_ROOT",tmp_path)
    split=tiny_split(); rows=driver.select_validation_rows(split)
    plan={"models":[{"manifest":{"path":f"fixture-{i}","sha256":str(i)*64}} for i in range(2)],
        "noise_plan":driver.fixed_noise_plan(rows),"frozen_files":{}}
    output=tmp_path/"run"
    def freeze(where,design):
        driver.write_json(where/"frozen-plan.json",{**plan,"design":design})
        return plan,split
    monkeypatch.setattr(driver,"freeze_inputs",freeze)
    norm=SimpleNamespace(encode=lambda x:x*2,to_payload=lambda:{"synthetic":True})
    sampler=SimpleNamespace(identity={"corpus_manifest_sha256":driver.SOURCE_SHA},
        macro_norm=norm,micro_norm=norm,micro=SimpleNamespace(config=None))
    decoder=SimpleNamespace(base_decoder=object(),macro_config=SimpleNamespace(control_count=33,max_mode=6))
    loads=[]
    def load(*args,**kwargs):
        assert (output/"frozen-plan.json").is_file(); loads.append(args)
        if len(loads)==1:
            ledger=json.loads((output/"result.json").read_text())
            pending=[name for name,item in ledger["units"].items() if item["status"]=="pending"]
            assert len(pending)==64 and ledger["completed_carrier_count"]==0
        return sampler,decoder
    monkeypatch.setattr(moving_frame_binding,"load_moving_frame_sampler",load)
    monkeypatch.setattr(driver,"check_full_to_value",lambda full,cfg:full.clone())
    monkeypatch.setattr(moving_frame,"extract_macro_controls",
        lambda base,z,condition,cfg,**kwargs:(torch.ones((1,3))*float(z[0,0]),{"execution_passed":True}))
    draws=[]
    def macro(sampler,condition,row):
        draws.append(row["macro_noise_seed"])
        if fail_macro and len(draws)==1: raise RuntimeError("fixture macro failure, no redraw")
        return torch.ones((1,3))*99,torch.ones((1,3))*198
    monkeypatch.setattr(driver,"sample_macro",macro)
    micros=[]
    def micro(sampler,condition,white,row):
        micros.append((row["micro_noise_seed"],condition.detach().clone(),white.detach().clone()))
        return torch.ones((1,4))*float(white[0,0])
    monkeypatch.setattr(driver,"sample_micro",micro)
    def save(directory,decoder,carrier,condition,provenance):
        directory.mkdir()
        return dict(provenance)
    monkeypatch.setattr(driver,"save_carrier",save)
    measured=[]; clean=[]
    def measure(decoder,carrier,condition,**kwargs):
        p=kwargs["provenance"]; row=p["noise_plan"]
        assert condition.numpy().view(np.uint32).tolist()==row["condition_fp32_bits"]
        measured.append((p["model_replica"],p["stratum_id"],p["view"],kwargs["deep"],
            carrier["m"].clone(),carrier["z"].clone()))
        driver.write_json(kwargs["output"],{"execution_passed":True,"view":p["view"]})
        return {"execution_passed":True}
    def compression(decoder,carrier,condition,**kwargs):
        assert kwargs["provenance"]["view"]=="real_m_real_z"
        assert torch.equal(carrier["z"],kwargs["reference_full"])
        clean.append(kwargs["root_id"])
        return {"execution_passed":True}
    monkeypatch.setattr(helper,"measure_dynamics_carrier",measure)
    monkeypatch.setattr(helper,"measure_clean_compression",compression)
    result=driver.run(output,device="cpu",wall_seconds=60)
    assert result["execution_passed"] is (not fail_macro)
    assert len(loads)==2 and len(draws)==16 and len(clean)==16
    assert len(measured)==(62 if fail_macro else 64)
    assert result["completed_carrier_count"]==len(measured)
    assert result["first_completed_carrier_benchmark"]["deep_included"]
    assert sum(item[3] for item in measured)==(15 if fail_macro else 16)
    assert all(not deep or (replica==0 and view in (driver.ARMS[0],driver.ARMS[3]))
        for replica,group,view,deep,m,z in measured)
    if fail_macro:
        assert result["failures"] and result["units"]["seed-0/root-0/generated_m_generated_z"]["status"]=="not_run"
    # All successful conditional pair calls share physical bits and micro noise.
    for seed in {x[0] for x in micros}:
        calls=[x for x in micros if x[0]==seed]
        if len(calls)==2: assert torch.equal(calls[0][1],calls[1][1])
    assert (output/"result.json").is_file()


def test_allocation_fresh_output_and_wall_guards(tmp_path,monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID",raising=False)
    with pytest.raises(ValueError,match="allocation"): driver.run(tmp_path/"new")
    monkeypatch.setenv("SLURM_JOB_ID","fixture"); monkeypatch.setenv("SLURM_TMPDIR",str(tmp_path))
    with pytest.raises(ValueError,match="6h"): driver.run(tmp_path/"new",wall_seconds=21601)
    monkeypatch.setattr(driver,"SCRATCH_ROOT",tmp_path)
    with pytest.raises(ValueError,match="fresh"): driver.run(tmp_path)


def test_deadline_stops_without_silent_protocol_reduction_and_keeps_partial(tmp_path,monkeypatch):
    now=[100.]
    monkeypatch.setattr(driver.time,"monotonic",lambda:now[0])
    progress=driver.Progress(tmp_path,10)
    progress.unit("completed",lambda:{"execution_passed":True})
    now[0]=111.
    with pytest.raises(TimeoutError,match="no shortened"):
        progress.unit("must-not-start",lambda:pytest.fail("deadline"))
    report=json.loads((tmp_path/"result.json").read_text())
    assert report["units"]["completed"]["status"]=="completed"
    assert "must-not-start" not in report["units"]
