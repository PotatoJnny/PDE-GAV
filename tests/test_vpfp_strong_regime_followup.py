"""Mock scheduler/processes and tiny file inventories only; no real job calls."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("strong_followup_tests",ROOT/"scripts/run_vpfp_strong_regime_followup.py")
driver = importlib.util.module_from_spec(spec); spec.loader.exec_module(driver)


def budget_receipt(elapsed=18000):
    return {"predecessor":{"job_id":"5268586","state":"COMPLETED","exit_code":"0:0",
        "elapsed_seconds":elapsed,"allocated_gpus":1},"cap_gpu_seconds":28800,
        "remaining_floor_minutes":max(0,28800-elapsed)//60}


def scheduler(monkeypatch, *, predecessor=None, current=None, control=None):
    calls=[]
    predecessor = predecessor or "5268586|COMPLETED|0:0|18000|cpu=8,gres/gpu=1,gres/gpu:l40s=1|480|\n"
    current = current or "777|RUNNING|0:0|60|cpu=8,gres/gpu=1|180|\n"
    control = control or "JobId=777 JobState=RUNNING TimeLimit=03:00:00 RunTime=00:01:03"
    def query(command):
        calls.append(command)
        if command[0]==driver.SCONTROL: return control
        return predecessor if command[command.index("-j")+1]=="5268586" else current
    monkeypatch.setattr(driver,"query",query)
    return calls


def test_exact14_fixed_paths_not_best_or_last():
    rows = driver.expected_models(Path("/synthetic/run"))
    assert len(rows)==14
    for replica in range(2):
        assert len([v for v in rows.values() if v["replica"]==replica])==7
        for arm in ("cov_base","initial_conditioned"):
            assert rows[f"seed-{replica}/{arm}/step-020000"]["checkpoint_path"].endswith("/initial-step-020000.pt")
            for step in (30000,40000):
                assert rows[f"seed-{replica}/{arm}/step-{step:06d}"]["checkpoint_path"].endswith(f"/checkpoints/fixed-step-{step:06d}.pt")
        assert rows[f"seed-{replica}/initial_conditioned_rho_cov/step-020000"]["checkpoint_path"].endswith("/checkpoints/fixed-step-020000.pt")
    assert not any("best.pt" in row["checkpoint_path"] or "last.pt" in row["checkpoint_path"] for row in rows.values())


@pytest.mark.parametrize("tres,count",[("cpu=8,gres/gpu=1",1),("gres/gpu=1,gres/gpu:l40s=1",1),("gres/gpu:l40s=1",1),
    ("gres/gpu=2,gres/gpu:l40s=2",2)])
def test_allocated_gpu_aliases_not_double_counted(tres,count):
    assert driver.gpu_count(tres)==count


@pytest.mark.parametrize("tres",["cpu=8","gres/gpu=1,gres/gpu:l40s=2","gres/gpu=1,gres/gpu=1","gres/gpu=one"])
def test_ambiguous_tres_rejected(tres):
    with pytest.raises(ValueError): driver.gpu_count(tres)


def test_terminal_allocation_elapsed_not_batch_sum_and_real_current_limit(monkeypatch):
    calls = scheduler(monkeypatch,predecessor="5268586|COMPLETED|0:0|18000|gres/gpu=1|480|\n5268586.batch|COMPLETED|0:0|17990|gres/gpu=1|480|\n")
    value = driver.scheduler_budget(budget_receipt(),"777")
    assert value["remaining_strong_gpu_seconds"]==10800
    assert value["remaining_current_seconds"]==10800-63
    assert len(calls)==3 and calls[0][0]==driver.SACCT and calls[-1][0]==driver.SCONTROL


@pytest.mark.parametrize("change",["running","failed","elapsed","two_gpu","current_two_gpu","cap","floor","long_limit","default_limit","control_mismatch"])
def test_budget_overrides_or_nonterminal_parent_fail_closed(monkeypatch,change):
    receipt = budget_receipt(); kwargs={}
    if change=="running": kwargs["predecessor"]="5268586|RUNNING|0:0|18000|gres/gpu=1|480|"
    if change=="failed": kwargs["predecessor"]="5268586|FAILED|1:0|18000|gres/gpu=1|480|"
    if change=="elapsed": receipt["predecessor"]["elapsed_seconds"]=17900
    if change=="two_gpu": kwargs["predecessor"]="5268586|COMPLETED|0:0|18000|gres/gpu=2|480|"
    if change=="current_two_gpu": kwargs["current"]="777|RUNNING|0:0|60|gres/gpu=2|180|"
    if change=="cap": receipt["cap_gpu_seconds"]=57600
    if change=="floor": receipt["remaining_floor_minutes"]=181
    if change=="long_limit":
        kwargs.update(current="777|RUNNING|0:0|0|gres/gpu=1|181|",control="JobId=777 JobState=RUNNING TimeLimit=03:01:00 RunTime=00:00:00")
    if change=="default_limit":
        kwargs.update(current="777|RUNNING|0:0|0|gres/gpu=1|1|",control="JobId=777 JobState=RUNNING TimeLimit=00:01:00 RunTime=00:00:00")
    if change=="control_mismatch": kwargs["control"]="JobId=777 JobState=RUNNING TimeLimit=03:01:00 RunTime=00:00:00"
    scheduler(monkeypatch,**kwargs)
    with pytest.raises(ValueError): driver.scheduler_budget(receipt,"777")


def test_remaining_minutes_are_floor_of_actual_terminal_seconds(monkeypatch):
    scheduler(monkeypatch,predecessor="5268586|COMPLETED|0:0|18001|gres/gpu=1|480|",
        current="777|RUNNING|0:0|1|gres/gpu=1|179|",
        control="JobId=777 JobState=RUNNING TimeLimit=02:59:00 RunTime=00:00:01")
    assert driver.scheduler_budget(budget_receipt(18001),"777")["current_limit_seconds"]==10740


@pytest.fixture
def inventory(tmp_path,monkeypatch):
    scratch = tmp_path/"scratch"; scratch.mkdir()
    run,source,repo = scratch/"run",scratch/"source/corpus",tmp_path/"repo"
    run.mkdir(); source.mkdir(parents=True); (repo/"configs").mkdir(parents=True)
    monkeypatch.setattr(driver,"RUN_ROOT",run); monkeypatch.setattr(driver,"SOURCE",source)
    monkeypatch.setattr(driver,"ROOT",repo); monkeypatch.setattr(driver,"SCRATCH",scratch)
    def asset(path,value):
        path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value))
        return {"path":str(path),"sha256":driver.file_sha(path)}
    source_asset = asset(source/"manifest.json",{"manifest_sha256":"c"*64,"base_decoder_sha256":"d"*64})
    config = asset(repo/"configs/vpfp_unified_live_strong_collision.json",{"synthetic":True})
    models = {}
    prior_models = {}
    for key,item in driver.expected_models(run).items():
        checkpoint = asset(Path(item["checkpoint_path"]),{"synthetic":"SHA-bound native header placeholder"})
        models[key] = {k:v for k,v in item.items() if k!="checkpoint_path"}; models[key]["checkpoint"]=checkpoint
        prior_models[key] = {"checkpoint":checkpoint["path"],"sha256":checkpoint["sha256"],
            **{k:item[k] for k in ("arm","replica","step")}}
    prior_plan = {"schema":"vpfp_source_aware_strong_continuation_and_rho_control_v1", "source":str(source.parent),
        "source_corpus_manifest_sha256":"c"*64,"config":config["path"],
        "frozen_input_sha256":{source_asset["path"]:source_asset["sha256"],config["path"]:config["sha256"]}}
    artifacts = {"result":asset(run/"result.json",{"execution_passed":True,"preflight_only":False,"regime":"strong","audits":{k:{} for k in models}}),
        "plan":asset(run/"frozen-plan.json",prior_plan),
        "checkpoints":asset(run/"frozen-checkpoints.json",{"models":prior_models,
            "all_models_frozen_before_feature_reading":True,"generation_selection":"fixed_updates_only",
            "sha256":{x["checkpoint"]["path"]:x["checkpoint"]["sha256"] for x in models.values()}})}
    receipt = {"schema":driver.SCHEMA,**budget_receipt(),"run_root":str(run),"source":{"corpus":str(source),"manifest_sha256":"c"*64},
        "artifacts":artifacts,"models":models,"config":config,"scientific_admission":False}
    path = tmp_path/"receipt.json"
    def publish(): return asset(path,receipt)
    item = publish()
    return SimpleNamespace(receipt=receipt,path=path,item=item,publish=publish,asset=asset,scratch=scratch)


def test_human_receipt_binds_real14_checkpoint_files_and_three_terminal_artifacts(inventory):
    receipt,_,frozen,manifest = driver.load_receipt(inventory.path,inventory.item["sha256"])
    assert len(receipt["models"])==14 and manifest["manifest_sha256"]=="c"*64
    for model in receipt["models"].values(): assert frozen[model["checkpoint"]["path"]]==model["checkpoint"]["sha256"]
    Path(next(iter(receipt["models"].values()))["checkpoint"]["path"]).write_text("changed")
    with pytest.raises(ValueError,match="SHA mismatch"): driver.load_receipt(inventory.path,inventory.item["sha256"])


@pytest.mark.parametrize("change",["missing_model","best_path","wrong_step","config","unfinished"])
def test_wrong_state_or_nonterminal_inventory_rejected(inventory,change):
    receipt=inventory.receipt
    key=next(iter(receipt["models"]))
    if change=="missing_model": receipt["models"].pop(key)
    if change=="best_path": receipt["models"][key]["checkpoint"]["path"]="/wrong/best.pt"
    if change=="wrong_step": receipt["models"][key]["step"]=19000
    if change=="config": receipt["config"]["sha256"]="a"*64
    if change=="unfinished": receipt["artifacts"]["result"]=inventory.asset(Path(receipt["artifacts"]["result"]["path"]),{"execution_passed":False})
    item=inventory.publish()
    with pytest.raises(ValueError): driver.load_receipt(inventory.path,item["sha256"])


def test_one14_directory_audit_then14_independent_smokes_even_audit_failure(monkeypatch,inventory):
    monkeypatch.setenv("SLURM_JOB_ID","777"); monkeypatch.setenv("SLURM_TMPDIR",str(inventory.scratch))
    (driver.ROOT/"scripts").mkdir(); (driver.ROOT/"src/pde_agent_verifier_solver/plugins/vpfp").mkdir(parents=True)
    for name in ("run_vpfp_regime_fidelity_coverage.py","run_plasma_canonical_smoke.py","run_vpfp_independent_confirmation.py"):
        (driver.ROOT/"scripts"/name).write_text("# synthetic script binding")
    monkeypatch.setattr(driver,"scheduler_budget",lambda *a:{"remaining_current_seconds":5000})
    monkeypatch.setattr(driver,"validate_generated",lambda *a:None)
    calls=[]
    def child(command,log,deadline):
        calls.append(command); Path(log).write_text("synthetic process only")
        if command[1].endswith("run_vpfp_regime_fidelity_coverage.py"):
            return {"exit_code":1,"timed_out":False,"actual_wall_seconds":.1}
        path=Path(command[command.index("--output")+1]); path.mkdir(parents=True)
        report={"status":"passed","diffusion_checkpoint_sha256":command[command.index("--diffusion-checkpoint-sha256")+1],
            "config_sha256":inventory.receipt["config"]["sha256"],"online_pde_calls_during_learned_sampling":0,
            "online_pde_guard_restored":True,"full_solution_prediction_required":False}
        (path/"report.json").write_text(json.dumps(report))
        return {"exit_code":0,"timed_out":False,"actual_wall_seconds":.1}
    monkeypatch.setattr(driver,"run_child",child)
    result=driver.run(SimpleNamespace(receipt=inventory.path,receipt_sha256=inventory.item["sha256"],output=inventory.scratch/"out",device="cuda"))
    assert len(calls)==15 and calls[0].count("--generated-dir")==14
    assert calls[0][calls[0].index("--sample-count")+1]=="128" and calls[0][calls[0].index("--bootstrap-repeats")+1]=="49"
    assert result["audit"]["status"]=="failed" and all(row["status"]=="passed" for row in result["smokes"].values())
    assert not result["execution_passed"] and not result["scientific_admission"]
    assert all(command[command.index("--physics")+1]=="vpfp" for command in calls[1:])


def test_scheduler_failure_precedes_large_inventory_validation(monkeypatch,inventory):
    monkeypatch.setenv("SLURM_JOB_ID","777"); monkeypatch.setenv("SLURM_TMPDIR",str(inventory.scratch))
    def failure(*args): raise ValueError("wrong current time limit")
    monkeypatch.setattr(driver,"scheduler_budget",failure)
    monkeypatch.setattr(driver,"load_receipt",lambda *a:pytest.fail("do not hash/load14 assets after scheduler failure"))
    result=driver.run(SimpleNamespace(receipt=inventory.path,receipt_sha256=inventory.item["sha256"],output=inventory.scratch/"out",device="cuda"))
    assert result["failure"]["reason"]=="wrong current time limit"
    assert all(row["status"]=="pending" for row in result["smokes"].values())


@pytest.mark.parametrize("timeout",[False,True])
def test_scheduler_error_keeps_stdout_stderr_in_json(monkeypatch,inventory,timeout):
    monkeypatch.setenv("SLURM_JOB_ID","777"); monkeypatch.setenv("SLURM_TMPDIR",str(inventory.scratch))
    error = (subprocess.TimeoutExpired([driver.SACCT],30,output=b"partial accounting",stderr=b"configuration lookup failed")
        if timeout else subprocess.CalledProcessError(1,[driver.SACCT],output="partial accounting",stderr="configuration lookup failed"))
    def failure(*args): raise error
    monkeypatch.setattr(driver,"scheduler_budget",failure)
    result=driver.run(SimpleNamespace(receipt=inventory.path,receipt_sha256=inventory.item["sha256"],output=inventory.scratch/"out",device="cuda"))
    assert result["failure"]["command_stdout"]=="partial accounting"
    assert result["failure"]["command_stderr"]=="configuration lookup failed"
    assert json.loads((inventory.scratch/"out/result.json").read_text())["failure"]==result["failure"]


@pytest.fixture
def generated_inventory(inventory):
    # Actual tiny torch/NPY files exercise the metadata reader, not a trained
    # native network or a synthetic claim of complete decoder validation.
    import numpy as np
    import torch
    designs={f"seed-{i}":{"conditions":[[.7,1.2] for _ in range(128)],
        "sampling_seeds":list(range(1000+i*1000,1128+i*1000)),
        "known_initial_seeds":list(range(4000+i*1000,4128+i*1000))} for i in range(2)}
    frozen={}
    for item in inventory.receipt["models"].values():
        checkpoint={"schema":"synthetic_header_only","config":{},"model":{},"normalizer":{},
            "training":{"step":item["step"]},"base_decoder_sha256":"d"*64,"corpus_manifest_sha256":"c"*64}
        path=Path(item["checkpoint"]["path"]); torch.save(checkpoint,path)
        item["checkpoint"]["sha256"]=driver.file_sha(path)
        directory=Path(item["evaluation_directory"]); directory.mkdir(parents=True)
        np.save(directory/"generated-coefficients.npy",np.zeros((128,4),dtype=np.float32))
        design=designs[f"seed-{item['replica']}"]
        np.save(directory/"generated-conditions.npy",np.asarray(design["conditions"],dtype=np.float32))
        provenance={"experiment_arm":item["arm"],"global_update":item["step"],
            "checkpoint_sha256":item["checkpoint"]["sha256"],
            "coefficient_sha256":driver.file_sha(directory/"generated-coefficients.npy"),
            "design":design,"known_initial_seed_applied":item["arm"]!="cov_base",
            "sampler":{"checkpoint_sha256":item["checkpoint"]["sha256"],
                "corpus_manifest_sha256":"c"*64,"base_decoder_sha256":"d"*64}}
        for name,value in (("generated-provenance.json",provenance),("generated-physical.json",{}),
                           ("physical-distribution-audit.json",{})):
            (directory/name).write_text(json.dumps(value))
    return inventory.receipt,{"plan":{"generation_design":designs}},frozen,{"base_decoder_sha256":"d"*64}


def test_actual_header_and_fp32_generation_metadata_reader(generated_inventory):
    driver.validate_generated(*generated_inventory)
    assert len(generated_inventory[2])==14*5


@pytest.mark.parametrize("change",["checkpoint_step","checkpoint_source","checkpoint_extra_key","sampler_source",
    "sampler_checkpoint","generation_step","paired_seed","known_prior_flag","condition_bit","conditions_dtype","coefficient_hash"])
def test_generated_metadata_substitution_rejected(generated_inventory,change):
    import numpy as np
    import torch
    receipt,documents,frozen,manifest=generated_inventory
    item=next(iter(receipt["models"].values())); directory=Path(item["evaluation_directory"])
    checkpoint_path=Path(item["checkpoint"]["path"])
    provenance_path=directory/"generated-provenance.json"
    value=json.loads(provenance_path.read_text())
    if change.startswith("checkpoint_"):
        payload=torch.load(checkpoint_path,weights_only=True,map_location="cpu")
        if change=="checkpoint_step": payload["training"]["step"]=19000
        elif change=="checkpoint_source": payload["corpus_manifest_sha256"]="e"*64
        else: payload["continuation_state"]={}
        torch.save(payload,checkpoint_path)
    elif change=="sampler_source": value["sampler"]["corpus_manifest_sha256"]="e"*64
    elif change=="sampler_checkpoint": value["sampler"]["checkpoint_sha256"]="e"*64
    elif change=="generation_step": value["global_update"]=19000
    elif change=="paired_seed": value["design"]["sampling_seeds"][0]+=1
    elif change=="known_prior_flag": value["known_initial_seed_applied"]=True
    elif change=="coefficient_hash": value["coefficient_sha256"]="e"*64
    elif change in ("condition_bit","conditions_dtype"):
        conditions=np.load(directory/"generated-conditions.npy",allow_pickle=False)
        if change=="condition_bit": conditions.view(np.uint32)[0,0]+=1
        else: conditions=conditions.astype(np.float64)
        np.save(directory/"generated-conditions.npy",conditions)
    provenance_path.write_text(json.dumps(value))
    with pytest.raises(ValueError): driver.validate_generated(receipt,documents,frozen,manifest)


def test_budget_exhaustion_preserves_all14_unrun_smokes_without_shrinking_audit(monkeypatch,inventory):
    monkeypatch.setenv("SLURM_JOB_ID","777"); monkeypatch.setenv("SLURM_TMPDIR",str(inventory.scratch))
    (driver.ROOT/"scripts").mkdir(); (driver.ROOT/"src/pde_agent_verifier_solver/plugins/vpfp").mkdir(parents=True)
    for name in ("run_vpfp_regime_fidelity_coverage.py","run_plasma_canonical_smoke.py","run_vpfp_independent_confirmation.py"):
        (driver.ROOT/"scripts"/name).write_text("# fixture")
    clock=[0.]
    monkeypatch.setattr(driver.time,"monotonic",lambda:clock[0])
    monkeypatch.setattr(driver,"scheduler_budget",lambda *a:{"remaining_current_seconds":100})
    monkeypatch.setattr(driver,"validate_generated",lambda *a:None)
    calls=[]
    def child(command,log,deadline):
        calls.append(command); clock[0]=71.
        return {"exit_code":-15,"timed_out":True,"actual_wall_seconds":71.}
    monkeypatch.setattr(driver,"run_child",child)
    result=driver.run(SimpleNamespace(receipt=inventory.path,receipt_sha256=inventory.item["sha256"],output=inventory.scratch/"out",device="cuda"))
    assert len(calls)==1 and calls[0].count("--generated-dir")==14
    assert calls[0][calls[0].index("--sample-count")+1]=="128"
    assert all(row["status"]=="not_run_budget_exhausted" for row in result["smokes"].values())
    assert not result["execution_passed"]


def test_launcher_short_default_formal_manual_binding_and_no_children():
    path=ROOT/"scripts/run_vpfp_strong_regime_followup.slurm"
    subprocess.run(["bash","-n",str(path)],check=True)
    source=path.read_text()
    for text in ("#SBATCH --time=00:01:00","#SBATCH --gres=gpu:l40s:1","--cpus-per-task=8","--mem=64G",
                 "VPFP_STRONG_FOLLOWUP_RECEIPT_SHA256","HEAD^{commit}","SLURM_CONF","launch-receipt.tsv.partial"):
        assert text in source
    assert not re.search(r"^\s*(sbatch|srun|scancel|scontrol)\b",source,re.M)
    assert "--time=" not in source.split("set -euo pipefail",1)[1]
