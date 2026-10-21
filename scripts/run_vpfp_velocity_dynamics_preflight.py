"""Fixed4/8/16 driven-U compression probe; no model training or PDE solve.

Only cached512 training +128 validation macro rows are used. Off-node errors
are against the existing 33-node cubic macro SURROGATE, never continuous
reference truth. A small error does not establish regime/forcing fidelity.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from pde_agent_verifier_solver.plugins.vpfp.coherent_velocity_dynamics import (
    VPFPVelocityDynamicsConfig,VPFPVelocityDynamicsNormalizer,
    evaluate_velocity_dynamics,fit_velocity_dynamics,
)
from pde_agent_verifier_solver.plugins.vpfp.diffusion_data import read_manifest
from pde_agent_verifier_solver.plugins.vpfp.moving_frame import MovingFrameMacroConfig,interpolate_controls

DEFAULT_CACHE=Path("${PDE_GAV_RUN_ROOT}/vpfp-weak-regime-coherent-5268892/macro-targets")
DEFAULT_SOURCE=Path("${PDE_GAV_RUN_ROOT}/vpfp-scaleup-weak-5241246/corpus")
COUNTS={"train":512,"validation":128}
DEGREES=(4,8,16)
SCRATCH_ROOT=Path("/scratch")


def file_sha(path):
    digest=sha256()
    if Path(path).is_symlink() or not Path(path).is_file():
        raise ValueError("regular immutable input file required")
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""): digest.update(block)
    return digest.hexdigest()


def write_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+".partial")
    temporary.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n")
    temporary.replace(path)


def frozen_inputs(cache,source):
    cache,source=Path(cache).resolve(strict=True),Path(source).resolve(strict=True)
    frozen={}
    def freeze(path,expected=None):
        digest=file_sha(path)
        frozen[str(path)]=digest
    plan_path=cache.parent/"frozen-plan.json"; freeze(plan_path)
    prior=json.loads(plan_path.read_text())
    manifest=read_manifest(source,verify_hashes=False)
    if (Path(prior["arguments"]["source"]).resolve()/"corpus"!=source
            or prior["source_manifest_sha256"]!=manifest["manifest_sha256"]
            or manifest["regime_id"]!="long_time_weak_collision_v2"
            or manifest["condition_order"]!=["collision_nu","temperature"]
            or set(prior["macro_targets"])!=set(COUNTS)):
        raise ValueError("cached macros must bind the original weak source and two split roles")
    freeze(source/"manifest.json")
    freeze(source/manifest["seed_plan"]["path"],manifest["seed_plan"]["sha256"])
    moving=ROOT/"src/pde_agent_verifier_solver/plugins/vpfp/moving_frame.py"
    freeze(moving,prior["source_file_sha256"][str(moving.relative_to(ROOT))])
    freeze(Path(__file__))
    freeze(ROOT/"src/pde_agent_verifier_solver/plugins/vpfp/coherent_velocity_dynamics.py")
    data={}; roots=set(); ids=set()
    for role,count in COUNTS.items():
        path=cache/(role+".json"); freeze(path)
        receipt=json.loads(path.read_text())
        if receipt!=prior["macro_targets"][role] or Path(receipt["file"])!=cache/(role+".npy"):
            raise ValueError("cache receipt differs from frozen macro extraction plan")
        freeze(Path(receipt["file"]),receipt["sha256"])
        extraction=receipt["extraction"]
        if (manifest["splits"][role]["count"]!=count or len(extraction)!=count
                or any(item.get("schema")!="vpfp_moving_frame_macro_extraction_v1"
                    or item.get("control_count")!=33 or item.get("maximum_mode")!=6
                    or item.get("fit_grid")!=33 or item.get("physical_velocity_units") is not True for item in extraction)):
            raise ValueError("fixed512/128, 33-frame, mode6 physical macro cache required")
        arrays={}
        for key in ("conditions","equation_seeds","case_ids","stratum_ids"):
            item=manifest["splits"][role]["files"][key]; path=source/item["path"]
            freeze(path,item["sha256"]); arrays[key]=np.load(path,allow_pickle=False,mmap_mode="r")
        if (list(map(str,arrays["case_ids"]))!=receipt["case_ids"]
                or list(map(int,arrays["equation_seeds"]))!=receipt["sample_ids"]
                or len(set(receipt["case_ids"]))!=count or len(set(receipt["sample_ids"]))!=count
                or roots.intersection(receipt["case_ids"]) or ids.intersection(receipt["sample_ids"])):
            raise ValueError("macro rows must replay distinct source roots/IDs without train-validation overlap")
        roots.update(receipt["case_ids"]); ids.update(receipt["sample_ids"])
        macro=np.load(receipt["file"],mmap_mode="r",allow_pickle=False)
        if macro.shape!=(count,5*33*169) or macro.dtype!=np.dtype("float32"):
            raise ValueError("exact native macro FP32 layout required")
        if (arrays["conditions"].shape!=(count,2) or arrays["conditions"].dtype!=np.dtype("float32")
                or not np.isfinite(arrays["conditions"]).all()):
            raise ValueError("finite native FP32 nu/T conditions required")
        data[role]={"macro":macro,**arrays,"receipt":receipt}
    return data,manifest,frozen


def surrogate(values,points,config):
    """Actual old cubic macro interpolant and its own autograd derivative."""
    batch,count=len(values),len(points)
    times=points.repeat(batch).detach().requires_grad_(True)
    repeated=values[:,None,:,:,None].expand(-1,count,-1,-1,-1).reshape(batch*count,2,33,1)
    macro_config=MovingFrameMacroConfig(control_count=33,max_mode=6,time_domain=(config.t0,config.t1))
    output=interpolate_controls(repeated,times,macro_config)[:,:,0]
    derivatives=torch.stack([torch.autograd.grad(output[:,c].sum(),times,retain_graph=True)[0] for c in range(2)],dim=1)
    return output.reshape(batch,count,2).transpose(1,2).detach(),derivatives.reshape(batch,count,2).transpose(1,2).detach()


def analyze_batch(values,nu,config):
    fitted=fit_velocity_dynamics(values,nu,config)
    # Both halves and quarters inside all32 intervals; no feature-based choice.
    fractions=torch.tensor([.25,.5,.75],device=values.device,dtype=torch.float64)
    points=config.t0+config.duration*(torch.arange(32,device=values.device,dtype=torch.float64)[:,None]+fractions).flatten()/32
    target,target_dt=surrogate(values.double(),points,config)
    predicted=evaluate_velocity_dynamics(fitted.coordinates,nu,points,config)
    h=config.duration/65536
    fd=(evaluate_velocity_dynamics(fitted.coordinates,nu,points+h,config)["U"]
        -evaluate_velocity_dynamics(fitted.coordinates,nu,points-h,config)["U"])/(2*h)
    fd_half=(evaluate_velocity_dynamics(fitted.coordinates,nu,points+h/2,config)["U"]
        -evaluate_velocity_dynamics(fitted.coordinates,nu,points-h/2,config)["U"])/h
    def rms(t): return t.square().mean(dim=(1,2)).sqrt().detach().cpu().tolist()
    diagnostics=fitted.diagnostics
    diagnostics.update(surrogate_off_node_U_rms_gap=rms(predicted["U"]-target),
        surrogate_off_node_dU_rms_gap=rms(predicted["dU_dt"]-target_dt),
        surrogate_off_node_drive_rms_gap=rms(predicted["a"]-target_dt-nu[:,None,None]*target),
        ou_derivative_fd_h_rms_gap=rms(predicted["dU_dt"]-fd),
        ou_derivative_fd_half_h_rms_gap=rms(predicted["dU_dt"]-fd_half),
        surrogate_drive_probe_mean_square=((target_dt+nu[:,None,None]*target).square().mean(dim=(1,2))).cpu().tolist(),
        ou_drive_probe_mean_square=predicted["a"].square().mean(dim=(1,2)).detach().cpu().tolist(),
        fd_h_physical_time=h,
        off_node_reference="old33frame_native_cubic_surrogate_not_continuous_physical_truth")
    return fitted.coordinates.detach().cpu(),diagnostics


def run(arguments):
    if not os.environ.get("SLURM_JOB_ID","").isdigit() or not os.environ.get("SLURM_TMPDIR"):
        raise ValueError("real data fitting and numerical preflight require a Slurm compute allocation")
    if (type(arguments.batch_size) is not int or not 1<=arguments.batch_size<=32
            or not math.isfinite(arguments.wall_seconds) or not 0<arguments.wall_seconds<=3600):
        raise ValueError("bounded batch1..32 and wall seconds in (0,3600] required")
    output=Path(arguments.output).absolute()
    if (output.exists() or output.is_symlink() or not output.is_relative_to(SCRATCH_ROOT)
            or any(output==Path(p).resolve() or output in Path(p).resolve().parents
                or Path(p).resolve() in output.parents for p in (arguments.cache,arguments.source))):
        raise ValueError("fresh scratch output outside frozen source/cache required")
    output.mkdir(parents=True)
    started=time.monotonic(); deadline=started+arguments.wall_seconds
    report={"schema":"vpfp_velocity_dynamics_representation_preflight_v1","execution_passed":False,
        "scientific_admission":False,"new_training":False,"new_PDE_solves":0,
        "stage":"freeze_inputs","degrees":{},"off_node_comparison_is_surrogate_only":True,
        "full_source_coefficients_rehashed":False,"source_identity_checked_by_canonical_manifest_and_cache_extraction_receipt":True}
    def save():
        report["wall_seconds"]=time.monotonic()-started; write_json(output/"result.json",report)
    def clock_guard():
        if time.monotonic()>=deadline: raise TimeoutError("bounded representation preflight exhausted; incomplete rows preserved")
    save()
    try:
        data,manifest,frozen=frozen_inputs(arguments.cache,arguments.source)
        clock_guard()
        write_json(output/"frozen-plan.json",{"files":frozen,"source_manifest_sha256":manifest["manifest_sha256"],
            "cache":str(Path(arguments.cache).resolve()),"source":str(Path(arguments.source).resolve()),
            "counts":COUNTS,"degrees":list(DEGREES),"time_domain":manifest["time_domain"],
            "off_node_interval_fractions":[.25,.5,.75],"original_node_count":33,
            "target_interpolant":"existing moving_frame.interpolate_controls; fixed finite-difference slopes",
            "root_ids":{role:row["receipt"]["case_ids"] for role,row in data.items()},
            "sample_ids":{role:row["receipt"]["sample_ids"] for role,row in data.items()},
            "normalizer_fit_role":"train_only","scientific_admission":False})
        for degree in DEGREES:
            cfg=VPFPVelocityDynamicsConfig(drive_degree=degree,t0=manifest["time_domain"][0],t1=manifest["time_domain"][1])
            directory=output/f"degree-{degree}"; directory.mkdir()
            row_report={"status":"running","splits":{}}
            report["degrees"][str(degree)]=row_report
            coordinates={}
            for role,count in COUNTS.items():
                row_report["splits"][role]={"completed_rows":0,"requested_rows":count,"batch_reports":[]}
                parts=[]
                for first in range(0,count,arguments.batch_size):
                    clock_guard(); stop=min(count,first+arguments.batch_size)
                    report["stage"]=f"degree-{degree}/{role}/{first}:{stop}"; save()
                    values=torch.from_numpy(np.array(data[role]["macro"][first:stop,:66],copy=True)).reshape(-1,2,33).to(arguments.device)
                    nu=torch.from_numpy(np.array(data[role]["conditions"][first:stop,0],copy=True)).double().to(arguments.device)
                    coordinates_batch,diagnostics=analyze_batch(values,nu,cfg)
                    if not diagnostics["U0_preserved_exactly"]: raise ValueError("initial coherent velocity was not retained exactly")
                    parts.append(coordinates_batch)
                    row_report["splits"][role]["batch_reports"].append({"first_row":first,"stop_row":stop,**diagnostics})
                    row_report["splits"][role]["completed_rows"]=stop; save()
                coordinates[role]=torch.cat(parts)
                summaries={}
                for key in ("node_rms_error","node_max_absolute_error","design_condition_number",
                        "quadrature_velocity_max_difference","drive_mean_square","drive_derivative_mean_square",
                        "drive_high_order_energy","surrogate_off_node_U_rms_gap","surrogate_off_node_dU_rms_gap",
                        "surrogate_off_node_drive_rms_gap","ou_derivative_fd_h_rms_gap","ou_derivative_fd_half_h_rms_gap"):
                    values=np.asarray([x for batch in row_report["splits"][role]["batch_reports"] for x in batch[key]],dtype=np.float64)
                    summaries[key]={"mean":float(values.mean()),"median":float(np.median(values)),
                        "p90":float(np.quantile(values,.9)),"maximum":float(values.max())}
                row_report["splits"][role]["summary"]=summaries
                path=directory/f"{role}-coordinates.npy"; np.save(path,coordinates[role].numpy(),allow_pickle=False)
                row_report["splits"][role]["coordinates"]={"path":str(path),"sha256":file_sha(path)}
            clock_guard()
            normalizer=VPFPVelocityDynamicsNormalizer.fit(cfg,coordinates["train"],sample_ids=data["train"]["receipt"]["sample_ids"])
            payload=normalizer.to_payload(); write_json(directory/"normalizer.json",payload)
            row_report["normalizer_sha256"]=payload["sha256"]
            row_report["normalized_coordinates"]={}
            for role,values in coordinates.items():
                encoded=normalizer.encode(values); restored=normalizer.decode(encoded)
                row_report["normalized_coordinates"][role]={"coordinate_rms":encoded.square().mean().sqrt().item(),
                    "roundtrip_max_error":(restored-values).abs().max().item(),"used_for_fit":role=="train"}
            row_report["status"]="completed"; save()
        report.update(execution_passed=True,stage="completed")
    except Exception as error:
        report["failure"]={"type":type(error).__name__,"reason":str(error),"traceback":traceback.format_exc()}
    finally: save()
    return report


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache",type=Path,default=DEFAULT_CACHE)
    p.add_argument("--source",type=Path,default=DEFAULT_SOURCE)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--wall-seconds",type=float,default=1800.)
    return p


if __name__=="__main__":
    raise SystemExit(0 if run(parser().parse_args())["execution_passed"] else 1)
