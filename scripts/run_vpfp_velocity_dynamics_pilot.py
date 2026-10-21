"""Six fixed5k macro-only development arms; no PDE, micro training or admission.

The old33-node baseline and driven-U degrees4/8 learn GLOBAL and LOCAL
macros jointly. Output physical33-node m is compatible with the existing
micro condition encoder, but its cubic interpolant is NOT the OU process.
"""
from __future__ import annotations

import argparse
from copy import copy
from dataclasses import replace
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
sys.path.insert(0, str(ROOT/"scripts"))
import train_vpfp_structured_diffusion as shared
import run_vpfp_velocity_dynamics_preflight as representation
from pde_agent_verifier_solver.plugins.vpfp.canonical_binding import VPFPOnlineModelSpec
from pde_agent_verifier_solver.plugins.vpfp.diffusion_data import load_split
from pde_agent_verifier_solver.plugins.vpfp.coherent_diffusion import (
    VPFPCoherentMacroConfig, VPFPCoherentMacroNormalizer, VPFPCoherentMacroEDM)
from pde_agent_verifier_solver.plugins.vpfp.coherent_velocity_diffusion import (
    VPFPVelocityMacroConfig, VPFPVelocityMacroEDM, VPFPVelocityMacroNormalizer,
    VelocityCoordinateMetric, fit_macro_coordinates, decode_macro_coordinates,
    load_velocity_macro_checkpoint, json_sha)
from pde_agent_verifier_solver.plugins.vpfp.coherent_velocity_dynamics import (
    VPFPVelocityDynamicsConfig, evaluate_velocity_dynamics)

ARMS = ("nodes33", "drive4", "drive8")
DEFAULT_PREFLIGHT = Path("${PDE_GAV_RUN_ROOT}/vpfp-weak-dynamics-diagnostic-5287092/pipeline/velocity-preflight/result.json")


def seed_for(domain, seed, row):
    return int.from_bytes(sha256(f"vpfp-U-pilot-v1:{domain}:{seed}:{row}".encode()).digest()[:8], "little") % (2**63)


def require_raw_reference(path, digest, *, prepare_only=False):
    """Execution/lineage prerequisite only; scientific interpretation is manual."""
    if path is None and digest is None and prepare_only:
        return None
    if path is None or not isinstance(digest,str) or len(digest)!=64:
        raise ValueError("training requires explicit raw-reference receipt path and SHA")
    path=Path(path).resolve()
    report=json.loads(path.read_text())
    cases=report.get("cases",[])
    if (report.get("schema")!="vpfp_weak_two_root_raw_time_velocity_preflight_v1"
            or report.get("execution_passed") is not True or report.get("completed_pde_solves")!=6
            or len(cases)!=2 or [x.get("root",{}).get("stratum_id") for x in cases]!=[0,7]
            or any(x.get("execution_passed") is not True for x in cases)):
        raise ValueError("completed six-solve two-root strata0/7 raw-reference receipt required")
    return {"path":str(path),"sha256":digest,"roots":[x["root"] for x in cases],
        "execution_prerequisite_only":True,"scientific_approval_inferred":False}


def training_factories(config, arrays, parent, corpus):
    """Ordinary shared.train factories; no private optimizer or lifecycle."""
    def split_loader(path, role, verify_hashes=False):
        if Path(path).resolve() != Path(corpus).resolve() or role not in ("train", "validation"):
            raise ValueError("only the frozen training/validation view may be opened")
        split = load_split(path, role, verify_hashes=verify_hashes)
        if len(split) != len(arrays[role]): raise ValueError("derived macro row count changed")
        return replace(split, coefficients=arrays[role])
    common = {"model_config_factory": lambda *a: config, "split_loader": split_loader}
    if isinstance(config, VPFPVelocityMacroConfig):
        return {**common, "model_factory": VPFPVelocityMacroEDM,
            "normalizer_factory": lambda cfg, raw, sample_ids: VPFPVelocityMacroNormalizer.fit(
                cfg, raw, sample_ids=sample_ids, parent_normalizer=parent),
            "normalizer_loader": VPFPVelocityMacroNormalizer.from_payload,
            "training_metric_factory": VelocityCoordinateMetric,
            "validation_metric_factory": VelocityCoordinateMetric}
    return {**common, "model_factory": VPFPCoherentMacroEDM,
        "normalizer_factory": VPFPCoherentMacroNormalizer.fit,
        "normalizer_loader": VPFPCoherentMacroNormalizer.from_payload}


def named_parameter_sha(model):
    digest = sha256()
    for name, value in model.named_parameters():
        digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_fixed(entry, config, device, manifest):
    path = Path(entry["checkpoint"])
    if isinstance(config, VPFPVelocityMacroConfig):
        model, normalizer, payload = load_velocity_macro_checkpoint(path, entry["sha256"], device=device)
    else:
        payload = torch.load(path, weights_only=True, map_location="cpu")
        if (set(payload) != {"schema", "config", "model", "normalizer", "base_decoder_sha256", "corpus_manifest_sha256", "training"}
                or payload["schema"] != VPFPCoherentMacroEDM.checkpoint_schema or payload["config"] != config.to_dict()):
            raise ValueError("exact original macro seven-field asset required")
        model = VPFPCoherentMacroEDM(config)
        normalizer = VPFPCoherentMacroNormalizer.from_payload(payload["normalizer"])
        buffers = dict(model.named_buffers()); expected = model.state_dict()
        if set(payload["model"]) != set(expected): raise ValueError("baseline tensor inventory differs")
        for name, value in payload["model"].items():
            if (not torch.is_tensor(value) or value.shape != expected[name].shape or value.dtype != expected[name].dtype
                    or not torch.isfinite(value).all() or (name in buffers and not torch.equal(value, expected[name]))):
                raise ValueError("invalid baseline weight/immutable buffer")
        model.load_state_dict(payload["model"], strict=True)
        model, normalizer = model.to(device).eval(), normalizer.to(device)
    if (payload["config"] != config.to_dict() or normalizer.config != config
            or payload["corpus_manifest_sha256"] != manifest["manifest_sha256"]
            or payload["base_decoder_sha256"] != manifest["base_decoder_sha256"]
            or payload["training"]["step"] != 5000
            or json_sha(payload["normalizer"]) != payload["training"]["normalizer_sha256"]):
        raise ValueError("fixed5k checkpoint source/normalizer binding differs")
    return model, normalizer


def velocity_diagnostics(macro, condition, macro_config, *, raw=None, dynamics_config=None):
    """Fixed probes; exact global-U drive proxy is not measured full-f source."""
    u = macro[:, :66].reshape(-1,2,33).double()
    nu = condition[:, 0].double()
    cfg = VPFPVelocityDynamicsConfig(drive_degree=4, t0=macro_config.time_domain[0], t1=macro_config.time_domain[1])
    reports = {}
    for count in (65,129):
        points = torch.linspace(*macro_config.time_domain, count, dtype=torch.float64, device=u.device)
        velocity, dt = representation.surrogate(u, points, cfg)
        acceleration = dt+nu[:,None,None]*velocity
        row = {"U_rms": velocity.square().mean().sqrt().item(),
            "U0_norm": velocity[:,:,0].norm(dim=1).item(), "terminal_U_norm": velocity[:,:,-1].norm(dim=1).item(),
            "dtU_rms": dt.square().mean().sqrt().item(),
            "a_vector_mean_square": acceleration.square().sum(dim=1).mean().item(),
            "a_max_vector_norm": acceleration.norm(dim=1).max().item()}
        collision_scale=(nu*condition[:,1].double().sqrt()).item()
        row["a_rms_over_nu_sqrt_bath_T"]=math.sqrt(row["a_vector_mean_square"])/collision_scale
        row["a_max_over_nu_sqrt_bath_T"]=row["a_max_vector_norm"]/collision_scale
        if dynamics_config is not None:
            native = evaluate_velocity_dynamics(raw[:, :dynamics_config.velocity_dimension], nu,
                points, dynamics_config.dynamics)
            row.update(ou_vs_cubic_U_rms=(native["U"]-velocity).square().mean().sqrt().item(),
                ou_vs_cubic_drive_rms=(native["a"]-acceleration).square().mean().sqrt().item(),
                ou_drive_mean_square=native["drive_mean_square"].item(),
                ou_drive_derivative_mean_square=native["drive_derivative_mean_square"].item())
        reports[str(count)] = row
    # Retain one joint macro descriptor, not an independently generated U law.
    reports["joint_nonU_coordinate_rms"] = macro[:,66:].double().square().mean().sqrt().item()
    reports["source_interpretation"] = "a=Udot+nu*U; acceleration proxy under periodic constant mass, not independently integrated full-f source; collision normalization uses nu*sqrt(bath_T), not a pass threshold"
    reports["interpolant"] = "existing33node_cubic_for_micro_compatibility; OU_continuous_process_reported_separately"
    reports["scientific_admission"] = False
    return reports


def prepare(args, macro_cfg, data, output, clock_guard):
    arrays = {"nodes33": {role:row["macro"] for role,row in data.items()}}
    parent = VPFPCoherentMacroNormalizer.fit(macro_cfg,
        torch.from_numpy(np.array(data["train"]["macro"],copy=True)),
        sample_ids=data["train"]["receipt"]["sample_ids"])
    configs = {"nodes33":macro_cfg}
    receipt = {"parent_normalizer_sha256":json_sha(parent.to_payload()), "arms":{},
        "off_node_truth_available":False, "normalizer_fit_role":"train_only"}
    for arm,degree in (("drive4",4),("drive8",8)):
        cfg = VPFPVelocityMacroConfig(macro_cfg,degree); configs[arm]=cfg; arrays[arm]={}
        receipt["arms"][arm]={}
        for role,values in data.items():
            path=output/f"{arm}-{role}.npy"
            prepared=np.lib.format.open_memmap(path,mode="w+",dtype="float32",
                shape=(len(values["macro"]),cfg.latent_dimension))
            reports=[]
            for first in range(0,len(prepared),16):
                clock_guard()
                raw=torch.from_numpy(np.array(values["macro"][first:first+16],copy=True))
                condition=torch.from_numpy(np.array(values["conditions"][first:first+16],copy=True))
                transformed,report=fit_macro_coordinates(raw,condition,cfg)
                if report["U0_preserved_exactly"] is not True: raise ValueError("initial U was lost in representation fit")
                prepared[first:first+len(raw)]=transformed.numpy(); reports.append(report)
            prepared.flush(); arrays[arm][role]=prepared
            receipt["arms"][arm][role]={"path":str(path),"sha256":shared.file_sha(path),"batch_reports":reports}
    shared.write_json(output/"representation.json",receipt)
    return configs,arrays,parent,receipt


def sample_attempts(args, entry, config, manifest, conditions, noise_seeds, output, clock_guard):
    model,normalizer=load_fixed(entry,config,args.device,manifest)
    macro_cfg=config.base_macro_config if isinstance(config,VPFPVelocityMacroConfig) else config
    output.mkdir()
    samples=np.lib.format.open_memmap(output/"macro.npy",mode="w+",dtype="float32",shape=(128,macro_cfg.latent_dimension))
    coordinates=np.lib.format.open_memmap(output/"coordinates.npy",mode="w+",dtype="float32",shape=(128,config.latent_dimension))
    samples[:]=np.nan; coordinates[:]=np.nan
    receipt={"attempted_count":128,"completed_attempt_count":0,"valid_count":0,"failed_count":0,"rows":[],
        "checkpoint":entry,"conditions_fp32":conditions.tolist(),"noise_seeds":noise_seeds,
        "root_ids":[sha256(f"vpfp-U-pilot-root:{seed}".encode()).hexdigest() for seed in noise_seeds],
        "scientific_admission":False,"registered_attempt_root_count":128,
        "root_reuse_across_arms":"paired; summing three arm counts does not triple independent roots",
        "evidence_role":"same_validation_conditions_new_noise_previously_opened_development"}
    def save():
        samples.flush(); coordinates.flush(); shared.write_json(output/"generated-receipt.json",receipt)
    save()
    try:
        for index,seed in enumerate(noise_seeds):
            clock_guard(); row={"row":index,"root_id":receipt["root_ids"][index],"noise_seed":seed,"status":"failed"}
            condition=torch.from_numpy(np.array(conditions[index:index+1],copy=True)).to(args.device)
            try:
                with torch.no_grad():
                    white=model.sample(condition,[seed],steps=64)
                    raw=normalizer.decode(white)
                    m=decode_macro_coordinates(raw,condition,config) if isinstance(config,VPFPVelocityMacroConfig) else raw
                if not torch.isfinite(m).all(): raise ValueError("nonfinite generated macro")
                samples[index]=m[0].detach().cpu().numpy(); coordinates[index]=raw[0].detach().cpu().numpy()
                row["generation_completed"]=True
                row["diagnostics"]=velocity_diagnostics(m,condition,macro_cfg,raw=raw,
                    dynamics_config=config if isinstance(config,VPFPVelocityMacroConfig) else None)
                row["status"]="completed"; receipt["valid_count"]+=1
            except Exception as error:
                row["failure"]={"type":type(error).__name__,"reason":str(error),
                    "stage":"measurement" if row.get("generation_completed") else "generation"}
                receipt["failed_count"]+=1
            receipt["rows"].append(row); receipt["completed_attempt_count"]+=1; save()
    finally:
        receipt["unrun_attempt_count"]=128-receipt["completed_attempt_count"]
        receipt["independent_valid_root_count"]=receipt["valid_count"]
        receipt["files"]={name:shared.file_sha(output/name) for name in ("macro.npy","coordinates.npy")}
        save()
    return receipt


def sample_registered_groups(entries, output, callback, result, save):
    """Register every128-attempt denominator before any child sampling starts."""
    groups=[{"seed_index":entry["seed_index"],"arm":entry["arm"],"status":"pending",
        "attempted_count":128,"completed_attempt_count":0,"valid_count":0,"failed_count":0,
        "unrun_attempt_count":128,"receipt_path":str(Path(output)/f"seed-{entry['seed_index']}"/entry["arm"]/"generated/generated-receipt.json")}
        for entry in entries]
    result["generated"]=groups; save()
    for entry,group in zip(entries,groups):
        group["status"]="running"; save()
        try:
            receipt=callback(entry)
        except Exception as error:
            path=Path(group["receipt_path"])
            if path.is_file():
                partial=json.loads(path.read_text())
                group.update({key:partial[key] for key in ("completed_attempt_count","valid_count","failed_count","unrun_attempt_count")})
                group["receipt_sha256"]=shared.file_sha(path)
            group.update(status="partial_or_failed",failure={"type":type(error).__name__,"reason":str(error)})
            save(); raise
        group.update({key:receipt[key] for key in ("completed_attempt_count","valid_count","failed_count","unrun_attempt_count")})
        group["receipt_sha256"]=shared.file_sha(group["receipt_path"])
        group["status"]="completed" if receipt["valid_count"]==128 else "completed_with_failures"
        save()
    return groups


def run(args):
    if not os.environ.get("SLURM_JOB_ID","").isdigit() or not os.environ.get("SLURM_TMPDIR"):
        raise ValueError("real preparation/training/sampling requires a Slurm compute allocation")
    if (args.steps!=5000 or args.batch_size!=4 or args.replicas!=2 or args.generated_answers!=128
            or args.sample_steps!=64 or args.width!=256 or args.depth!=4
            or not math.isfinite(args.wall_seconds) or args.wall_seconds<=0
            or args.train_seconds<=0 or args.validate_every!=500):
        raise ValueError("fixed2 seeds x3 arms,5k updates,batch4,width256/depth4,128 attempts,Heun64 required")
    raw_reference=require_raw_reference(args.raw_reference_receipt,args.raw_reference_sha256,
                                         prepare_only=args.prepare_only)
    output=Path(args.output).resolve(); source=Path(args.source).resolve(); cache=Path(args.cache).resolve()
    if output.exists() or not output.is_relative_to(Path("/scratch")) or any(output==x or output.is_relative_to(x) or x.is_relative_to(output) for x in (source,cache)):
        raise ValueError("fresh scratch output outside frozen input trees required")
    output.mkdir(parents=True)
    started=time.monotonic(); deadline=started+args.wall_seconds
    result={"schema":"vpfp_velocity_macro_pilot_v1","execution_passed":False,"scientific_admission":False,
        "new_PDE_solves":0,"micro_training":False,"stage":"prepare","training":[],"generated":[]}
    def save(): result["elapsed_seconds"]=time.monotonic()-started; shared.write_json(output/"result.json",result)
    def clock_guard():
        if time.monotonic()>=deadline: raise TimeoutError("pilot wall deadline reached; pending work is not completed")
    save()
    try:
        data,manifest,frozen=representation.frozen_inputs(cache,source)
        if raw_reference is not None:
            frozen[raw_reference["path"]]=raw_reference["sha256"]
            for root in raw_reference["roots"]:
                row=root["row"]
                if (str(data["validation"]["case_ids"][row])!=root["root_id"]
                        or int(data["validation"]["equation_seeds"][row])!=root["equation_seed"]
                        or int(data["validation"]["stratum_ids"][row])!=root["stratum_id"]):
                    raise ValueError("raw-reference root differs from this frozen validation source")
        preflight_path=Path(args.representation_preflight).resolve()
        prior=json.loads(preflight_path.read_text())
        if prior.get("execution_passed") is not True or prior.get("off_node_comparison_is_surrogate_only") is not True:
            raise ValueError("completed representation mechanics receipt required, not a physics certificate")
        preflight_plan=preflight_path.parent/"frozen-plan.json"; prior_plan=json.loads(preflight_plan.read_text())
        if (prior_plan["source_manifest_sha256"]!=manifest["manifest_sha256"]
                or Path(prior_plan["cache"]).resolve()!=cache): raise ValueError("preflight source/cache differs")
        for path in (Path(__file__),Path(args.config),preflight_path,preflight_plan,
                ROOT/"src/pde_agent_verifier_solver/plugins/vpfp/coherent_velocity_diffusion.py",
                ROOT/"scripts/train_vpfp_structured_diffusion.py"):
            frozen[str(path.resolve())]=shared.file_sha(path)
        spec=VPFPOnlineModelSpec.from_mapping(json.loads(Path(args.config).read_text())["vpfp"])
        if spec.base_decoder_sha256!=manifest["base_decoder_sha256"]: raise ValueError("config/source decoder differs")
        macro_cfg=VPFPCoherentMacroConfig(shared.model_config(args,spec),maximum_mode=6)
        args.source_revision=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        seeds=[args.seed+replica*10000 for replica in range(2)]
        conditions=np.array(data["validation"]["conditions"],copy=True)
        noise={str(seed):[seed_for("generation",seed,row) for row in range(128)] for seed in seeds}
        plan={"schema":"vpfp_velocity_macro_pilot_plan_v1","source_manifest_sha256":manifest["manifest_sha256"],
            "raw_reference_execution_receipt":raw_reference,
            "frozen_inputs":frozen,"arms":list(ARMS),"seeds":seeds,"steps":5000,"generated_attempts":128,
            "conditions_fp32":conditions.tolist(),"generation_noise_seeds":noise,
            "generation_conditions_from":"frozen128_validation; not trajectory pairing",
            "same_seed_not_same_dimension_noise_vectors":True,"equal_compute_claimed":False,
            "validation_losses_not_compared_across_representations":True,
            "weight_control":"new U coordinates fixed positive multiplier66/D_U; old baseline unchanged",
            "physics_preflight_is_surrogate_not_qualification":True,"scientific_admission":False}
        shared.write_json(output/"frozen-plan.json",plan)
        prepared=output/"prepared"; prepared.mkdir()
        configs,arrays,parent,prep=prepare(args,macro_cfg,data,prepared,clock_guard)
        transform_sha=json_sha({"frozen_inputs":frozen,"prepared":prep,"configs":{k:v.to_dict() for k,v in configs.items()}})
        if args.prepare_only:
            result.update(execution_passed=True,stage="prepared_not_trained"); return result
        args.checkpoint_steps=(5000,)
        for replica,seed in enumerate(seeds):
            initial={}
            for arm in ARMS:
                torch.manual_seed(seed+100)
                model=(VPFPCoherentMacroEDM if arm=="nodes33" else VPFPVelocityMacroEDM)(configs[arm])
                initial[arm]={"named_parameter_sha256":named_parameter_sha(model),"parameters":sum(p.numel() for p in model.parameters())}
            if len({row["named_parameter_sha256"] for row in initial.values()})!=1:
                raise ValueError("common macro trunk initialization differs")
            del model
            for arm in ARMS:
                clock_guard(); result["stage"]=f"seed-{replica}/{arm}/training"; save()
                directory=output/f"seed-{replica}"/arm; directory.mkdir(parents=True)
                run_args=copy(args); run_args.seed=seed
                run_args.train_seconds=min(args.train_seconds,max(1.,deadline-time.monotonic()-60))
                run_args.ablation_metadata={"schema":"vpfp_velocity_macro_pilot_ablation_v1","arm":arm,
                    "data_transform_sha256":transform_sha,**initial[arm],"equal_compute_claimed":False}
                entry={"seed_index":replica,"arm":arm,"status":"training_or_partial",
                    "parameters":initial[arm]["parameters"],"directory":str(directory)}
                result["training"].append(entry); save()
                _,trained=shared.train(run_args,spec,source,directory,
                    **training_factories(configs[arm],arrays[arm],parent,source))
                entry.update(summary=trained,last_checkpoint=str(directory/"checkpoints/last.pt"),
                    last_sha256=shared.file_sha(directory/"checkpoints/last.pt")); save()
                if trained["final_step"]!=5000: raise RuntimeError("paired arm stopped before5k; comparison incomplete")
                checkpoint=directory/"checkpoints/fixed-step-005000.pt"
                norm_payload=torch.load(checkpoint,map_location="cpu",weights_only=True)["normalizer"]
                actual_parent=norm_payload if arm=="nodes33" else norm_payload["parent"]
                if actual_parent!=parent.to_payload():
                    raise ValueError("paired arms changed original non-U train-only scales/centering")
                entry.update(status="fixed5k_complete",checkpoint=str(checkpoint),sha256=shared.file_sha(checkpoint))
                shared.write_json(output/"frozen-checkpoints.json",result["training"]); save()
        # Never inspect generated descriptors to select a checkpoint.
        result["stage"]="all_six_states_frozen_sampling"; save()
        def sample_one(entry):
            clock_guard(); arm=entry["arm"]; replica=entry["seed_index"]
            return sample_attempts(args,entry,configs[arm],manifest,conditions,
                noise[str(seeds[replica])],output/f"seed-{replica}"/arm/"generated",clock_guard)
        sample_registered_groups(result["training"],output,sample_one,result,save)
        result["stage"]="reference_validation_macro_diagnostics"; save()
        rows=[]
        for index in range(128):
            clock_guard()
            macro=torch.from_numpy(np.array(data["validation"]["macro"][index:index+1],copy=True)).to(args.device)
            condition=torch.from_numpy(conditions[index:index+1].copy()).to(args.device)
            rows.append(velocity_diagnostics(macro,condition,macro_cfg))
        shared.write_json(output/"validation-macro-diagnostics.json",{"rows":rows,"data_origin":"previously_opened_validation", "scientific_admission":False})
        passed=all(x["valid_count"]==128 and x["completed_attempt_count"]==128 for x in result["generated"])
        result.update(execution_passed=passed,stage="development_pilot_complete" if passed else "completed_with_generation_failures")
    except Exception as error:
        result["failure"]={"type":type(error).__name__,"reason":str(error),"traceback":traceback.format_exc()}
        result["stage"]="failed_or_partial"
    finally: save()
    return result


def parser():
    p=shared.parser()
    p.set_defaults(seed=1520260907,steps=5000,train_seconds=3600,validate_every=500,bootstrap_repeats=49)
    p.add_argument("--source",type=Path,default=representation.DEFAULT_SOURCE)
    p.add_argument("--cache",type=Path,default=representation.DEFAULT_CACHE)
    p.add_argument("--representation-preflight",type=Path,default=DEFAULT_PREFLIGHT)
    p.add_argument("--expected-preflight-sha256",required=True)
    p.add_argument("--raw-reference-receipt",type=Path)
    p.add_argument("--raw-reference-sha256")
    p.add_argument("--replicas",type=int,default=2)
    p.add_argument("--generated-answers",type=int,default=128)
    p.add_argument("--wall-seconds",type=float,default=28800.)
    p.add_argument("--prepare-only",action="store_true")
    return p


if __name__=="__main__":
    args=parser().parse_args(); result=run(args)
    raise SystemExit(0 if result["execution_passed"] else 1)
