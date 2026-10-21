"""Manual14-state strong regime audit and native smoke inside the ORIGINAL8h.

Requires a human-frozen terminal-job receipt. No training, PDE reference solve,
population resampling, checkpoint selection, submission or budget enlargement.
The existing canonical smoke alone generates its tiny interface cases.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src")); sys.path.insert(0,str(ROOT/"scripts"))
from run_vpfp_independent_confirmation import checked_asset, file_sha, write_json, run_child, assert_frozen

SCHEMA = "vpfp_strong_regime_followup_receipt_v1"
PREDECESSOR = "5268586"
CAP = 28800
SCRATCH = Path("${PDE_GAV_RUN_ROOT}")
RUN_ROOT = SCRATCH/("vpfp-strong-regime-coherent-"+PREDECESSOR)
SOURCE = SCRATCH/"vpfp-strong-main-5244223/corpus"
SACCT = "sacct"
SCONTROL = "scontrol"
FIELDS = {"schema","config","model","normalizer","base_decoder_sha256","corpus_manifest_sha256","training"}


def expected_models(run_root):
    rows = {}
    for replica in range(2):
        for arm in ("cov_base","initial_conditioned","initial_conditioned_rho_cov"):
            for step in ((20000,) if arm.endswith("rho_cov") else (20000,30000,40000)):
                base = Path(run_root)/f"seed-{replica}"/arm
                checkpoint = (base/"initial-step-020000.pt" if step==20000 and not arm.endswith("rho_cov")
                              else base/"checkpoints"/f"fixed-step-{step:06d}.pt")
                key = f"seed-{replica}/{arm}/step-{step:06d}"
                rows[key] = {"replica":replica,"arm":arm,"step":step,"checkpoint_path":str(checkpoint),
                    "evaluation_directory":str(Path(run_root)/key/"evaluation")}
    return rows


def gpu_count(tres):
    pairs = {}
    for part in tres.split(","):
        if "=" in part:
            key,value = part.split("=",1)
            if key in pairs: raise ValueError("duplicate accounting TRES key")
            pairs[key] = value
    values = [value for key,value in pairs.items() if key=="gres/gpu" or key.startswith("gres/gpu:")]
    if not values or any(not value.isdigit() for value in values):
        raise ValueError("explicit integer allocated GPU TRES required")
    typed = sum(int(value) for key,value in pairs.items() if key.startswith("gres/gpu:"))
    generic = int(pairs["gres/gpu"]) if "gres/gpu" in pairs else typed
    if typed and typed != generic: raise ValueError("generic and typed allocated GPU counts disagree")
    return generic  # Do not double count generic and typed aliases.


def duration_seconds(value):
    match = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d{2}):(\d{2})",value)
    if match is None: raise ValueError("finite scheduler D-HH:MM:SS time required")
    days,hours,minutes,seconds = [int(v or 0) for v in match.groups()]
    if minutes>=60 or seconds>=60: raise ValueError("malformed scheduler duration")
    return days*86400+hours*3600+minutes*60+seconds


def accounting_row(text,job_id):
    matches = []
    for line in text.splitlines():
        values = line.strip().split("|")
        if values and values[-1]=="": values.pop()
        if values and values[0]==str(job_id): matches.append(values)
    if len(matches)!=1 or len(matches[0])!=6:
        raise ValueError("one exact parent allocation accounting row required; never sum batch/extern steps")
    identity,state,exit_code,elapsed,tres,limit = matches[0]
    if not elapsed.isdigit() or not limit.isdigit(): raise ValueError("integer Slurm elapsed and timelimit minutes required")
    return {"job_id":identity,"state":state,"exit_code":exit_code,"elapsed_seconds":int(elapsed),
        "allocated_gpus":gpu_count(tres),"time_limit_minutes":int(limit),"alloc_tres":tres}


def query(command):
    result = subprocess.run(command,check=True,text=True,capture_output=True,timeout=30)
    return result.stdout


def scheduler_budget(receipt,current_job):
    command = lambda job:[SACCT,"-X","-n","-P","-j",str(job),"--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES%200,TimelimitRaw"]
    predecessor = accounting_row(query(command(PREDECESSOR)),PREDECESSOR)
    expected = receipt["predecessor"]
    if (set(expected)!={"job_id","state","exit_code","elapsed_seconds","allocated_gpus"}
            or type(expected["allocated_gpus"]) is not int
            or any(predecessor[key]!=value for key,value in expected.items())
            or predecessor["state"]!="COMPLETED" or predecessor["exit_code"]!="0:0"
            or predecessor["allocated_gpus"]!=1):
        raise ValueError("predecessor receipt differs from successful terminal one-GPU sacct allocation")
    remaining = CAP-predecessor["elapsed_seconds"]
    if (receipt["cap_gpu_seconds"]!=CAP or type(receipt["cap_gpu_seconds"]) is not int
            or type(expected["elapsed_seconds"]) is not int or expected["elapsed_seconds"]<=0
            or type(receipt["remaining_floor_minutes"]) is not int
            or receipt["remaining_floor_minutes"]!=max(0,remaining)//60):
        raise ValueError("remaining budget must be floor((28800-terminal Slurm elapsed)/60), never training wall")
    current = accounting_row(query(command(current_job)),current_job)
    control_text = query([SCONTROL,"--oneliner","show","job",str(current_job)])
    control = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S+)",control_text))
    if control.get("JobId")!=str(current_job) or control.get("JobState")!="RUNNING":
        raise ValueError("current scheduler allocation must be the running caller job")
    limit = duration_seconds(control.get("TimeLimit",""))
    elapsed = duration_seconds(control.get("RunTime",""))
    if (current["allocated_gpus"]!=1 or current["state"]!="RUNNING" or limit<=60
            or limit!=current["time_limit_minutes"]*60 or limit>receipt["remaining_floor_minutes"]*60):
        raise ValueError("explicit current one-GPU TimeLimit must exceed default1min and not exceed original strong remaining cap")
    return {"predecessor":predecessor,"current":current,"current_control":control,
        "remaining_strong_gpu_seconds":remaining,"current_limit_seconds":limit,
        "remaining_current_seconds":limit-max(elapsed,current["elapsed_seconds"]),
        "budget_source":"terminal parent allocation ElapsedRaw; current scheduler TimeLimit/RunTime; no batch-step sum",
        "manual_campaign_constraint":"parent coordinates total48 GPUh and at most2 concurrent GPUs; no child submission"}


def load_receipt(path,digest):
    frozen = {}
    receipt = checked_asset({"path":str(Path(path).absolute()),"sha256":digest},frozen=frozen,json_only=True)
    if (set(receipt)!={"schema","predecessor","cap_gpu_seconds","remaining_floor_minutes","run_root","source","artifacts","models","config","scientific_admission"}
            or receipt["schema"]!=SCHEMA or receipt["scientific_admission"] is not False
            or receipt["predecessor"].get("job_id")!=PREDECESSOR
            or Path(receipt["run_root"])!=RUN_ROOT or Path(receipt["source"]["corpus"])!=SOURCE):
        raise ValueError("explicit fixed strong predecessor/source/14-state receipt required")
    expected = expected_models(RUN_ROOT)
    if set(receipt["models"])!=set(expected): raise ValueError("receipt must bind exactly14 registered states, not best-score selection")
    if set(receipt["artifacts"])!={"result","plan","checkpoints"}: raise ValueError("three terminal artifact bindings required")
    documents = {}
    for role,name in (("result","result.json"),("plan","frozen-plan.json"),("checkpoints","frozen-checkpoints.json")):
        item = receipt["artifacts"][role]
        if Path(item["path"])!=RUN_ROOT/name: raise ValueError("terminal artifact path differs from predecessor run")
        documents[role] = checked_asset(item,frozen=frozen,json_only=True)
    result,plan,checkpoint_set = [documents[key] for key in ("result","plan","checkpoints")]
    if (result.get("execution_passed") is not True or result.get("preflight_only") is not False
            or result.get("regime")!="strong" or set(result.get("audits",{}))!=set(expected)
            or checkpoint_set.get("all_models_frozen_before_feature_reading") is not True
            or checkpoint_set.get("generation_selection")!="fixed_updates_only"
            or set(checkpoint_set.get("models",{}))!=set(expected)
            or plan.get("schema")!="vpfp_source_aware_strong_continuation_and_rho_control_v1"
            or Path(plan["source"])/"corpus"!=SOURCE
            or plan["source_corpus_manifest_sha256"]!=receipt["source"]["manifest_sha256"]):
        raise ValueError("successful fixed14-state predecessor and original source required")
    original_hashes = plan["frozen_input_sha256"]
    source_manifest = checked_asset({"path":str(SOURCE/"manifest.json"),"sha256":original_hashes[str(SOURCE/"manifest.json")]},
        frozen=frozen,json_only=True)
    if source_manifest["manifest_sha256"]!=receipt["source"]["manifest_sha256"]:
        raise ValueError("original source manifest identity changed")
    config = receipt["config"]
    if (Path(config["path"])!=ROOT/"configs/vpfp_unified_live_strong_collision.json"
            or config["sha256"]!=original_hashes[plan["config"]]):
        raise ValueError("native smoke config must be the same hash as strong training config")
    checked_asset(config,frozen=frozen)
    for key,layout in expected.items():
        item = receipt["models"][key]
        if (set(item)!={"replica","arm","step","checkpoint","evaluation_directory"}
                or type(item["replica"]) is not int or type(item["step"]) is not int
                or any(item[name]!=layout[name] for name in ("replica","arm","step","evaluation_directory"))
                or item["checkpoint"]["path"]!=layout["checkpoint_path"]):
            raise ValueError("fixed state path/step/arm differs from the registered14-state layout")
        prior = checkpoint_set["models"][key]
        if (any(prior[name]!=item[name] for name in ("replica","arm","step"))
                or prior["checkpoint"]!=item["checkpoint"]["path"] or prior["sha256"]!=item["checkpoint"]["sha256"]
                or checkpoint_set["sha256"].get(prior["checkpoint"])!=prior["sha256"]):
            raise ValueError("actual fixed checkpoint differs from predecessor frozen set")
        checked_asset(item["checkpoint"],frozen=frozen)
    return receipt,documents,frozen,source_manifest


def validate_generated(receipt,documents,frozen,source_manifest):
    import numpy as np
    import torch
    for key,item in receipt["models"].items():
        checkpoint = torch.load(item["checkpoint"]["path"],map_location="cpu",weights_only=True)
        if (not isinstance(checkpoint,dict) or set(checkpoint)!=FIELDS or checkpoint["training"].get("step")!=item["step"]
                or checkpoint["corpus_manifest_sha256"]!=receipt["source"]["manifest_sha256"]
                or checkpoint["base_decoder_sha256"]!=source_manifest["base_decoder_sha256"]):
            raise ValueError("fixed state is not a seven-field native asset with matching global step/source")
        del checkpoint
        directory = Path(item["evaluation_directory"])
        for name in ("generated-coefficients.npy","generated-conditions.npy","generated-provenance.json",
                     "generated-physical.json","physical-distribution-audit.json"):
            path = directory/name
            if path.is_symlink(): raise ValueError("generated audit inputs must be regular files")
            frozen[str(path)] = file_sha(path)
        provenance = json.loads((directory/"generated-provenance.json").read_text())
        design = documents["plan"]["generation_design"][f"seed-{item['replica']}"]
        if (provenance.get("experiment_arm")!=item["arm"] or provenance.get("global_update")!=item["step"]
                or provenance.get("checkpoint_sha256")!=item["checkpoint"]["sha256"]
                or provenance.get("coefficient_sha256")!=frozen[str(directory/"generated-coefficients.npy")]
                or provenance.get("design")!=design
                or provenance.get("known_initial_seed_applied") is not (item["arm"]!="cov_base")):
            raise ValueError("generation receipt differs from fixed model/global-step/paired learning-curve design")
        identity = provenance["sampler"]
        if (identity.get("checkpoint_sha256")!=item["checkpoint"]["sha256"]
                or identity.get("corpus_manifest_sha256")!=receipt["source"]["manifest_sha256"]
                or identity.get("base_decoder_sha256")!=source_manifest["base_decoder_sha256"]):
            raise ValueError("native sampler identity differs from checkpoint/source")
        actual = np.load(directory/"generated-conditions.npy",allow_pickle=False)
        expected = np.asarray(design["conditions"],dtype=np.float32)
        if (actual.shape!=(128,2) or actual.dtype!=np.dtype("float32") or expected.shape!=actual.shape
                or not np.array_equal(actual.view(np.uint32),expected.view(np.uint32))):
            raise ValueError("all128 FP32 conditions must replay the frozen paired design")
    # The reused audit additionally replays all input hashes/root IDs and
    # validates the complete decoder ABI, reference ancestry and split hashes.


def run(arguments):
    job = os.environ.get("SLURM_JOB_ID","")
    if not job.isdigit() or not os.environ.get("SLURM_TMPDIR") or job==PREDECESSOR:
        raise ValueError("a separate Slurm compute allocation is required; never reuse the running parent")
    output = Path(arguments.output).absolute()
    if (output.exists() or output.is_symlink() or not output.is_relative_to(SCRATCH)
            or any(output==path or output in path.parents or path in output.parents for path in (RUN_ROOT,SOURCE))):
        raise ValueError("fresh independent scratch output outside frozen inputs required")
    output.mkdir(parents=True)
    started = time.monotonic()
    report = {"schema":"vpfp_strong_regime_followup_run_v1","execution_passed":False,"scientific_admission":False,
        "stage":"validate_terminal_receipt_and_scheduler_budget","audit":{"status":"pending"},
        "smokes":{key:{"status":"pending"} for key in expected_models(RUN_ROOT)},
        "new_training":False,"new_population_generation":False,"new_reference_PDE_solves":False,
        "evidence_role":"opened128_paired_learning_curve_development_regime_diagnostics_not_fresh512_confirmation",
        "same_seeds_across_states_are_not_independent_samples":True}
    def save():
        report["wall_seconds"] = time.monotonic()-started
        write_json(output/"result.json",report)
    save()
    try:
        # Query the actual scheduler before hashing/loading the14 assets, so
        # malformed --time cannot spend hours before the cap is checked.
        preliminary = checked_asset({"path":str(Path(arguments.receipt).absolute()),
            "sha256":arguments.receipt_sha256},json_only=True)
        budget_started = time.monotonic()
        budget = scheduler_budget(preliminary,job)
        deadline = budget_started+budget["remaining_current_seconds"]-30
        report["scheduler_budget"] = budget
        save()
        if deadline<=time.monotonic(): raise TimeoutError("original strong budget exhausted before input validation")
        receipt,documents,frozen,source_manifest = load_receipt(arguments.receipt,arguments.receipt_sha256)
        validate_generated(receipt,documents,frozen,source_manifest)
        for path in [Path(__file__),ROOT/"scripts/run_vpfp_regime_fidelity_coverage.py",ROOT/"scripts/run_plasma_canonical_smoke.py",
                     ROOT/"scripts/run_vpfp_independent_confirmation.py",*sorted((ROOT/"src/pde_agent_verifier_solver/plugins/vpfp").glob("*.py"))]:
            frozen[str(path)] = file_sha(path)
        write_json(output/"frozen-plan.json",{"receipt":{"path":str(arguments.receipt),"sha256":arguments.receipt_sha256},
            "scheduler_budget":budget,"files":frozen,"ordered_models":list(expected_models(RUN_ROOT)),
            "audit_count_per_state":128,"bootstrap_repeats":49,"canonical_smoke_is_separate_not_a_substitute":True})
        def stage(name,command,path,validator):
            if deadline-time.monotonic()<=5:
                return {"status":"not_run_budget_exhausted"}
            assert_frozen({"frozen_files":frozen})
            report["stage"] = name; save()
            try:
                value = run_child(command,output/(name+".log"),deadline)
                if value["exit_code"]!=0 or value["timed_out"]:
                    return {"status":"failed",**value}
                evidence = json.loads(path.read_text())
                validator(evidence)
                return {"status":"passed",**value,"report":str(path),"report_sha256":file_sha(path)}
            except Exception as error:
                return {"status":"failed","error_type":type(error).__name__,"error":str(error),"traceback":traceback.format_exc()}
        command = [sys.executable,str(ROOT/"scripts/run_vpfp_regime_fidelity_coverage.py"),
            "--model-source",str(SOURCE),"--reference-source",str(SOURCE),"--output",str(output/"regime-audit"),
            "--sample-count","128","--bootstrap-repeats","49","--device",arguments.device]
        for key in expected_models(RUN_ROOT): command.extend(("--generated-dir",receipt["models"][key]["evaluation_directory"]))
        def check_audit(value):
            if value.get("execution_passed") is not True or value.get("all_requested_physical_adapters_completed") is not True:
                raise ValueError("all14 requested128-answer regime audits must complete")
        report["audit"] = stage("regime-audit",command,output/"regime-audit/result.json",check_audit)
        save()
        # Failed distribution measurement never substitutes for or suppresses
        # an independently executable macro-only interface smoke.
        for index,(key,layout) in enumerate(expected_models(RUN_ROOT).items()):
            item = receipt["models"][key]
            target = output/"smokes"/key
            def check_smoke(value,item=item):
                if (value.get("status")!="passed" or value.get("diffusion_checkpoint_sha256")!=item["checkpoint"]["sha256"]
                        or value.get("config_sha256")!=receipt["config"]["sha256"]
                        or value.get("online_pde_calls_during_learned_sampling")!=0
                        or value.get("online_pde_guard_restored") is not True
                        or value.get("full_solution_prediction_required") is not False):
                    raise ValueError("native macro-only smoke failed its asset/online-PDE guard")
            command = [sys.executable,str(ROOT/"scripts/run_plasma_canonical_smoke.py"),"--physics","vpfp",
                "--config",receipt["config"]["path"],"--diffusion-checkpoint",item["checkpoint"]["path"],
                "--diffusion-checkpoint-sha256",item["checkpoint"]["sha256"],"--output",str(target),"--device",arguments.device]
            report["smokes"][key] = stage(f"native-smoke-{index:02d}",command,target/"report.json",check_smoke)
            save()
        assert_frozen({"frozen_files":frozen})
        report.update(execution_passed=report["audit"]["status"]=="passed" and all(x["status"]=="passed" for x in report["smokes"].values()),
            stage="completed" if time.monotonic()<deadline else "budget_exhausted_partial_preserved")
    except Exception as error:
        captured = lambda name: (getattr(error,name).decode("utf-8",errors="replace")
            if isinstance(getattr(error,name,None),bytes) else getattr(error,name,None))
        report["failure"] = {"type":type(error).__name__,"reason":str(error),"traceback":traceback.format_exc(),
            "command_stdout":captured("stdout"),"command_stderr":captured("stderr")}
    finally:
        save()
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--receipt",type=Path,required=True)
    p.add_argument("--receipt-sha256",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",choices=("cuda",),default="cuda")
    return p


if __name__=="__main__":
    raise SystemExit(0 if run(parser().parse_args())["execution_passed"] else 1)
