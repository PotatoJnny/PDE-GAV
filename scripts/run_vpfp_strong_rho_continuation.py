"""Fixed two-seed rho-time-covariance continuation after the completed audit.

Resume genuine last.pt at20k; retain30k/40k native EMA states. Do not refit
normalization, rank by cross-coordinate validation loss, or replace assets.
"""
from __future__ import annotations

import argparse
from copy import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parents[1]
loader = importlib.util.spec_from_file_location("rho_continuation_reuse",ROOT/"scripts/run_vpfp_strong_regime_coherent.py")
old = importlib.util.module_from_spec(loader)
loader.loader.exec_module(old)
audit_loader = importlib.util.spec_from_file_location("rho_completed_audit_binding",ROOT/"scripts/run_vpfp_strong_audit_completion.py")
completion = importlib.util.module_from_spec(audit_loader)
audit_loader.loader.exec_module(completion)
shared, previous = old.shared, old.previous
PARENT = Path("${PDE_GAV_RUN_ROOT}/vpfp-strong-regime-coherent-5268586")
SOURCE = Path("${PDE_GAV_RUN_ROOT}/vpfp-strong-main-5244223")
PARENT_HASHES = {
    "frozen-plan.json":"68ea04b0a9cbbada1682a048957ed4ac0cf072ec2eac848feea24be6af606aee",
    "frozen-checkpoints.json":"618ec5cdee50b2329a4443454751933657bf9094d9eabeca05de21c63b34e9c7",
    "result.json":"77763fb9d487fecc4b85767a68863e8386a708b3b54d11f2c1c9411be85270ce"}
SOURCE_SHA = "4f59a2101eea8eb3ba958250bf8f5bb9faa7d883a658aaa76a6c69f8c010e78d"
STEPS = (30000,40000)
SCHEMA = "vpfp_strong_rho_covariance_fixed_20_to40k_v1"


def checked_json(path,digest):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('JSON input is missing or symlinked')
    return json.loads(path.read_text())


def require_completed_audit(path,digest):
    value = checked_json(path,digest)
    if (value.get("schema") != "vpfp_strong_initial_layer_completion_v1"
            or value.get("execution_passed") is not True
            or value.get("completed_carrier_count") != 40
            or value.get("final_frozen_recheck_passed") is not True
            or not value.get("units")
            or any(unit.get("status") != "completed" for unit in value["units"])):
        raise ValueError("complete40-carrier audit, reference refinement and extended-time/t0 units required before continuation")
    plan_path = Path(path).parent/"frozen-plan.json"
    plan = checked_json(plan_path,value.get("frozen_plan_sha256"))
    history = plan["historical_plan"]
    if (history["sha256"] != completion.HISTORY_PLAN_SHA
            or plan["historical_result"]["sha256"] != completion.HISTORY_RESULT_SHA
            or value.get("historical_result_sha256") != completion.HISTORY_RESULT_SHA):
        raise ValueError("completed audit must retain the exact failed5289599 ancestry")
    historical_plan = checked_json(history["path"],history["sha256"])
    checked_json(plan["historical_result"]["path"],plan["historical_result"]["sha256"])
    if plan["selected"] != historical_plan["selected"] or plan["design"] != completion.completion_design():
        raise ValueError("completed audit fixed roots or extended design changed")
    expected = completion.registered_units({"selected":completion.old.registered_slots()})
    if len(value["units"]) != len(expected) or any(
            any(unit.get(k) != bound.get(k) for k in ("id","kind","stage","carrier_id"))
            for unit,bound in zip(value["units"],expected)):
        raise ValueError("exact77 registered audit identities required, not only a completed count")
    completion.validate_resumed_report(value,plan,value["frozen_plan_sha256"])
    completion.old.existing.check_frozen(plan["files"])
    for unit in value["units"]:
        attempt = unit["attempts"][-1]
        measured = checked_json(attempt["report"],attempt["sha256"])
        if measured.get("execution_passed") is not True:
            raise ValueError("completed audit unit report is not successful")
    return value


def inventory():
    documents = {name:checked_json(PARENT/name,digest) for name,digest in PARENT_HASHES.items()}
    plan, frozen, result = [documents[k] for k in ("frozen-plan.json","frozen-checkpoints.json","result.json")]
    if (result["execution_passed"] is not True or plan["source_corpus_manifest_sha256"] != SOURCE_SHA
            or plan["seed"] != 1620260906 or Path(plan["source"]) != SOURCE):
        raise ValueError("same completed parent and physical corpus required")
    files = {str(PARENT/name):digest for name,digest in PARENT_HASHES.items()}
    inputs = {}
    for replica in (0,1):
        key = f"seed-{replica}/{old.RHO_ARM}"
        summary = result["training"][key]
        if summary["final_step"] != 20000 or summary["stop_reason"] != "step_limit":
            raise ValueError("both original rho states must complete20k")
        bound = {}
        for role in ("last","best"):
            path = PARENT/key/"checkpoints"/(role+".pt")
            digest = frozen["sha256"][str(path)]
            if path.is_symlink():
                raise ValueError('parent training checkpoint differs')
            files[str(path)] = digest
            bound[role] = {"path":str(path),"sha256":digest}
        inputs[str(replica)] = {**bound,"normalizer_sha256":summary["normalizer_sha256"]}
    return inputs, files, plan


def training_arguments(output,replica,bound,revision,wall_seconds):
    args = old.parser().parse_args(["--config",str(ROOT/"configs/vpfp_unified_live_strong_collision.json"),
        "--source",str(SOURCE),"--pilot",str(PARENT),"--output",str(output)])
    args.seed = 1620260906+10000*replica
    args.steps, args.checkpoint_steps = 40000, STEPS
    args.train_seconds = wall_seconds
    args.source_revision = revision
    args.resume_checkpoint = Path(bound["last"]["path"])
    args.expected_resume_sha256 = bound["last"]["sha256"]
    args.resume_best_checkpoint = Path(bound["best"]["path"])
    args.expected_resume_best_sha256 = bound["best"]["sha256"]
    args.ablation_metadata = {"arm":old.RHO_ARM,"replica":replica,
        "rho_policy":"covariance","normalizer_refit":False,"controlled_change":"additional_updates_only",
        "fixed_snapshot_steps":list(STEPS),"cross_normalizer_validation_loss_comparable":False,
        "evidence_role":"opened_development_not_confirmation"}
    return args


def run(arguments):
    if not os.environ.get("SLURM_JOB_ID", "").isdigit() or not os.environ.get("SLURM_TMPDIR"):
        raise ValueError("Slurm compute allocation required")
    if not torch.cuda.is_available(): raise ValueError("CUDA required for identical continuation arithmetic")
    output = Path(arguments.output).absolute()
    if (output.exists() or output.is_symlink() or not output.is_relative_to(Path("${PDE_GAV_RUN_ROOT}"))
            or any(output==p or output.is_relative_to(p) or p.is_relative_to(output) for p in (PARENT,SOURCE))):
        raise ValueError("fresh separate scratch output required")
    if not math.isfinite(arguments.wall_seconds) or arguments.wall_seconds <= 180:
        raise ValueError("allocation must leave checkpoint/receipt time")
    output.mkdir(parents=True)
    started = time.monotonic()
    result = {"schema":SCHEMA,"execution_passed":False,"scientific_admission":False,
        "training":{},"models":{},"audits":{},"failures":[],"new_pde_solves":0,
        "fixed_steps":list(STEPS),"online_assets_replaced":False,"confirmation_512_completed":False}
    def save():
        result["wall_seconds"] = time.monotonic()-started
        shared.write_json(output/"result.json",result)
    def remaining():
        seconds = arguments.wall_seconds-(time.monotonic()-started)-120
        if seconds <= 0: raise TimeoutError("allocation ending; retain exact last states and uncompleted audit rows")
        return seconds
    save()
    try:
        audit = require_completed_audit(arguments.audit_completion,arguments.audit_sha256)
        inputs, frozen, prior = inventory()
        manifest = previous.read_manifest(SOURCE/"corpus",verify_hashes=True)
        if manifest["manifest_sha256"] != SOURCE_SHA: raise ValueError("bound source changed")
        config_path = ROOT/"configs/vpfp_unified_live_strong_collision.json"
        spec = previous.VPFPOnlineModelSpec.from_mapping(json.loads(config_path.read_text())["vpfp"])
        revision = subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
        frozen.update({str(ROOT/k):v for k,v in shared.source_fingerprints().items()})
        frozen[str(Path(__file__))] = shared.file_sha(__file__)
        frozen[str(Path(arguments.audit_completion).absolute())] = arguments.audit_sha256
        shared.write_json(output/"frozen-plan.json",{"schema":SCHEMA,"inputs":inputs,"frozen_files":frozen,
            "generation_design":prior["generation_design"],"source_revision":revision,"source_manifest_sha256":SOURCE_SHA,
            "recipe":{"steps":list(STEPS),"batch_size":4,"learning_rate":.0002,"validate_every":1000,
                "width":256,"depth":4,"normalizer_refitted":False},
            "audit_is_execution_prerequisite_not_physics_admission":True,
            "maintenance_policy":"preserve_periodic_last_no_automatic_cancel_requeue_or_restart"})
        torch.set_num_threads(min(8,int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))))
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        for replica in (0,1):
            key = f"seed-{replica}/{old.RHO_ARM}"
            result["stage"] = "train/"+key; save()
            args = training_arguments(output,replica,inputs[str(replica)],revision,remaining())
            init = {}
            cfg,make,fit = old.factories(args,spec,manifest,old.RHO_ARM,init)
            directory = output/key; directory.mkdir(parents=True)
            _,report = shared.train(args,spec,SOURCE/"corpus",directory,
                model_config_factory=lambda *a,cfg=cfg:cfg,model_factory=make,
                normalizer_factory=fit,normalizer_loader=old.old.VPFPInitialConditionedNormalizer.from_payload)
            result["training"][key] = report; save()
            if (report["final_step"] != 40000 or report["stop_reason"] != "step_limit"
                    or report["normalizer_sha256"] != inputs[str(replica)]["normalizer_sha256"]
                    or report.get("unreached_checkpoint_steps")):
                raise ValueError("fixed40k continuation unfinished or normalization changed")
            for snap in report["step_snapshots"]:
                result["models"][key+f"/step-{snap['step']:06d}"] = {
                    "replica":replica,"arm":old.RHO_ARM,"step":snap["step"],
                    "checkpoint":snap["checkpoint"],"sha256":snap["sha256"]}
            save()
        frozen_models = {x["checkpoint"]:x["sha256"] for x in result["models"].values()}
        shared.write_json(output/"frozen-checkpoints.json",{"models":result["models"],"sha256":frozen_models,
            "generation_selection":"fixed_updates_only","all_models_frozen_before_feature_reading":True})
        args.output = output
        constraint = previous.audit_value_control_corpus(SOURCE/"corpus",batch_size=8,tolerance=1.e-5)
        previous.require_value_control_audit(constraint)
        samples,schemas = old.old.load_references(args,manifest,constraint)
        first = next(iter(result["models"].values()))
        sampler = previous.load_vpfp_diffusion_sampler(first["checkpoint"],first["sha256"],device="cuda",steps=64)
        binding = previous.build_vpfp_online_binding(spec,output_root=output/"audit-runtime",device="cuda",latent_sampler=sampler)
        decoder = binding.plugin.generator.backend.runtime.base_model
        for key,entry in result["models"].items():
            remaining(); result["stage"] = "descriptive128/"+key; save()
            directory = output/key/"evaluation"; directory.mkdir(parents=True)
            eval_args = copy(args); eval_args.seed = 1620260906+10000*entry["replica"]
            _,reports = old.old.sample_and_audit(eval_args,spec,manifest,dict(entry,arm="initial_conditioned"),
                directory,decoder,samples,schemas,prior["generation_design"][f"seed-{entry['replica']}"])
            result["audits"][key] = {"completed_attempts":128,"requested_attempts":128,
                "directory":str(directory),"old_distribution_reports_descriptive_only":True,
                "regime_fidelity_coverage_followup":"pending_not_omitted_or_passed"}
            for kind,report in reports.items():
                report.update(experiment_arm=old.RHO_ARM,global_update=entry["step"],scientific_admission=False)
                shared.write_json(directory/f"{kind}-distribution-audit.json",report)
            save()
        previous.assert_hashes({**frozen,**frozen_models})
        result.update(execution_passed=True,stage="complete",regime_fidelity_coverage_followup="pending")
    except Exception as error:
        result["failures"].append({"stage":result.get("stage"),"type":type(error).__name__,
            "reason":str(error),"traceback":traceback.format_exc()})
    finally:
        save()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--audit-completion",type=Path,required=True)
    parser.add_argument("--audit-sha256",required=True)
    parser.add_argument("--wall-seconds",type=float,default=14400.)
    return 0 if run(parser.parse_args(argv))["execution_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
