"""Complete the fixed strong development audit without replacing any old draw.

Uses 5289599's 26 completed carriers as immutable evidence, fills its 14
unfinished slots, replays/refines its original stratum7 reference, and measures
257/513 time grids plus t0 one-sided balances for the same 8 refs + 16 IC draws.
No new generation or training. All science remains numerically unresolved.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src")); sys.path.insert(0,str(ROOT/"scripts"))
import run_vpfp_strong_initial_layer_audit as old
from pde_agent_verifier_solver.plugins.vpfp.reference_refinement import reference_domain_receipt

SCHEMA="vpfp_strong_initial_layer_completion_v1"
HISTORY=old.SCRATCH/"vpfp-strong-initial-layer-audit-5289599/audit"
HISTORY_RESULT_SHA="92af531b0653f541ab9b847c1831633c503b5b19fa054717d848ae5f0aefc535"
HISTORY_PLAN_SHA="e00f61aba29bab1845681089cb796ba5e7eee07ddc9807c206ba9d8e452bc86c"
KINDS=("independent_physics","temporal_frequency")
STAGES=("all","refinement","remaining","temporal","t0")


def _json_sha(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def cheap_domain_preflight():
    manifest=old.existing.read_manifest(old.SOURCE,verify_hashes=False)
    if manifest["manifest_sha256"]!=old.SOURCE_SHA: raise ValueError("original source manifest changed")
    provenance=manifest["provenance"]
    return reference_domain_receipt(provenance["generation_config"],provenance["decoder_config"])


def completion_design():
    return {"historical_design":old.design(),"historical_completed_carriers":26,"remaining_original_carriers":14,
        "extended_time_counts":[257,513],"extended_carrier_ids":[s["id"] for s in old.registered_slots()[:24]],
        "extended_temporal_geometry":"same_one_fixed_paired_sparse_lab_probe_not_full_space",
        "t0_relative_fd_steps":[2.**-9,2.**-10],"t0_stencil":"forward_second_order_[0,h,2h]_and_[0,h/2,h]",
        "t0_velocity_moment_orders":[17,19],"t0_additional_spatial_mass_grids":[],
        "t0_additional_L2":False,"t0_scope":"local_full_velocity_moment_balances_and_Poisson_not_new_global_norm",
        "execution_order":"original_stratum7_refinement_then14_remaining_then24_extended_time_then24_t0",
        "new_generation":False,"new_training":False,"automatic_downsampling":False,
        "same_seed_models_paired":True,"statistical_unit":"8_calibration_roots_plus_paired_model_noise_interventions_not40_independent_roots",
        "pending":["independent_global_spatial_source_momentum_and_energy","uniform_space_time_velocity_convergence",
            "fresh512_population_confirmation","a_priori_physical_numerical_acceptance_calibration"]}


def validate_history(result,plan):
    expected_ids=[s["id"] for s in old.registered_slots()]
    if (result.get("schema")!=old.SCHEMA or result.get("execution_passed") is not False
            or result.get("registered_carrier_count")!=40 or result.get("completed_carrier_count")!=26
            or result.get("frozen_plan_sha256")!=HISTORY_PLAN_SHA or result.get("design")!=old.design()
            or plan.get("design")!=old.design() or plan.get("source_manifest_sha256")!=old.SOURCE_SHA
            or [s["id"] for s in result.get("slots",[])]!=expected_ids
            or [s["id"] for s in plan.get("selected",[])]!=expected_ids):
        raise ValueError("fixed failed5289599 40-slot/26-complete history required")
    for i,(slot,row) in enumerate(zip(result["slots"],plan["selected"])):
        if slot.get("input")!=row: raise ValueError("historical slot input differs from original plan")
        if i<26:
            if slot["status"]!="completed" or any(slot[k].get("status")!="completed" for k in KINDS):
                raise ValueError("exact original26 complete slots required")
        elif slot["status"]=="completed": raise ValueError("unexpected replacement/completed history slot")
    if (plan.get("same_seed_models_paired") is not True or result.get("same_seed_models_paired") is not True
            or plan.get("checkpoint_candidates")!=old.candidates()):
        raise ValueError("paired original checkpoint ancestry changed")


def freeze_completion(check=lambda:None):
    """Rebind all original rows and both generations of code before measuring."""
    files={}
    def read(path,expected):
        check();path=Path(path).absolute()
        digest=old.existing.file_sha(path)
        files[str(path)]=digest
        return json.loads(path.read_text())
    result=read(HISTORY/"result.json",HISTORY_RESULT_SHA)
    history_plan=read(HISTORY/"frozen-plan.json",HISTORY_PLAN_SHA)
    validate_history(result,history_plan)
    old.existing.check_frozen(history_plan["files"])
    files.update(history_plan["files"])
    for slot in result["slots"][:26]:
        for kind in KINDS:
            entry=slot[kind];path=Path(entry["report"])
            if path!=HISTORY/slot["id"]/(kind+".json"):
                raise ValueError("historical report path outside its exact slot")
            payload=read(path,entry["sha256"])
            if payload.get("execution_passed") is not True:
                raise ValueError("historical complete receipt contains unsuccessful report")
            if kind=="temporal_frequency" and (payload.get("input_full_sha256")!=slot["input"]["native_full_sha256"]
                    or payload.get("source_provenance")!=slot["input"]):
                raise ValueError("historical temporal report input/ancestry changed")
    current,arrays,manifest,spec=old.freeze_inputs(check)
    for key in ("selected","generation_designs","checkpoint_candidates","source_manifest_sha256","refinement_selected"):
        if current[key]!=history_plan[key]: raise ValueError("current reconstruction changed original selection/ancestry: "+key)
    # Keep historical paths/SHA unchanged. New code paths have their own freeze;
    # they are never relabelled as the implementation used for old measurements.
    current_files=dict(current["files"])
    current_files[str(Path(__file__).resolve())]=old.existing.file_sha(__file__)
    for path,digest in current_files.items():
        if path in files and files[path]!=digest: raise ValueError("history/current file collision")
    files.update(current_files)
    old_root=next((Path(p).parents[1] for p in history_plan["files"] if p.endswith("/scripts/run_vpfp_strong_initial_layer_audit.py")),None)
    if old_root is None: raise ValueError("historical implementation root absent")
    unchanged=("src/pde_agent_verifier_solver/plugins/vpfp/independent_physics.py",
        "scripts/run_vpfp_independent_physics.py")
    changes=[]
    for path,digest in current_files.items():
        if not Path(path).is_relative_to(ROOT): continue
        relative=str(Path(path).relative_to(ROOT)); prior=history_plan["files"].get(str(old_root/relative))
        if prior is not None and prior!=digest:
            changes.append({"relative_path":relative,"historical_sha256":prior,"current_sha256":digest})
            if relative in unchanged or relative.startswith("legacy/vpfp/"):
                raise ValueError("remaining measurement science/legacy operator changed: "+relative)
    plan={"schema":SCHEMA,"design":completion_design(),"selected":current["selected"],"files":files,
        "historical_files":history_plan["files"],"current_implementation_files":current_files,
        "current_code_changes_from_history":changes,"current_input_plan":current,
        "historical_result":{"path":str(HISTORY/"result.json"),"sha256":HISTORY_RESULT_SHA},
        "historical_plan":{"path":str(HISTORY/"frozen-plan.json"),"sha256":HISTORY_PLAN_SHA},
        "historical_terminal":{"job_id":"5289599","state":"FAILED","exit_code":"124","old_failure_preserved":True},
        "historical_refinement_failure":result["refinement"],"historical_slots":result["slots"],
        "domain_coordinate_receipt":reference_domain_receipt(manifest["provenance"]["generation_config"],manifest["provenance"]["decoder_config"]),
        "evidence_role":"opened_development_continuation_not_fresh512","admitted":False}
    old.existing.check_frozen(files)
    return plan,arrays,manifest,spec


def extended_temporal(model,full,condition,*,row,check=lambda:None):
    """Same native analytic jet/source, finer fixed grids; dtS remains FD."""
    import torch
    from pde_agent_verifier_solver.plugins.vpfp.perturbation_backend import _legacy_api
    from pde_agent_verifier_solver.plugins.vpfp.regime_fidelity_coverage import _macro_values
    from pde_agent_verifier_solver.plugins.vpfp.physical_controls import _time_spectrum,_derivative_energy
    _legacy_api();analytic=importlib.import_module("agent_verifier.vpfp.analytic_source")
    if model.config.structured_time_control_count!=33 or not bool(torch.isfinite(full).all()):
        raise ValueError("unchanged finite native33-control answer required")
    fixed=old.design();_,xy=old.physical_points(model,fixed)
    vv=np.asarray(fixed["physical_velocity_probes"])
    if vv.shape!=xy.shape: raise ValueError("same fixed paired sparse probe required")
    report={"schema":"vpfp_native_B_extended_temporal_v1","execution_passed":True,"grids":[],
        "time_counts":[257,513],"source_provenance":row,"numerically_uncertain":True,"admitted":False,
        "input_full_sha256":sha256(full.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
        "derivatives":{"f_and_macros":"native analytic physical-coordinate first jets",
            "source":"np.gradient(source_samples,times,edge_order=2), not autodiff of S"},
        "scope":"same_one_fixed_paired_phase_probe; finite-window time spectra, not full-space norms",
        "independent_PDE_residual":False,"uniform_convergence_established":False}
    training=model.training;model.eval()
    try:
        with torch.no_grad(),analytic.ieee_float32_coordinate_matmul(model):
            for size in report["time_counts"]:
                times=np.linspace(*model.config.time_domain,size)
                points=np.column_stack((np.repeat(times,len(xy)),np.tile(xy,(size,1)),np.tile(vv,(size,1))))
                values,derivatives={},{}
                for offset in range(0,len(points),16):
                    check();q=points[offset:offset+16]
                    jet=analytic.analytic_structured_macro_jet(model,q[:,:3],full)
                    macros=_macro_values(jet,model.hermite_modes,float(condition[1]),model.config.velocity_floor_mixture)
                    source=analytic.manufacture_standard_vpfp_source_analytic(model,q,full,equation_condition=condition)
                    current={k:macros[k] for k in ("rho","jx","jy","K")}
                    current.update({k:source[k].detach().double().cpu().numpy() for k in ("f","source")})
                    dt={k:macros["dt_"+k] for k in ("jx","jy","K")}
                    dt.update(rho=jet["rho_derivatives"][0].detach().double().cpu().numpy(),
                        f=source["f_gradient"][:,0].detach().double().cpu().numpy())
                    for k,v in current.items(): values.setdefault(k,[]).append(v)
                    for k,v in dt.items(): derivatives.setdefault(k,[]).append(v)
                values={k:np.concatenate(v).reshape(size,len(xy)) for k,v in values.items()}
                derivatives={k:np.concatenate(v).reshape(size,len(xy)) for k,v in derivatives.items()}
                derivatives["source"]=np.gradient(values["source"],times,axis=0,edge_order=2)
                h=times[1]-times[0]
                report["grids"].append({"count":size,"times":times.tolist(),
                    "values":{k:v.tolist() for k,v in values.items()},"dt_values":{k:v.tolist() for k,v in derivatives.items()},
                    "spectra":{k:_time_spectrum(v,times) for k,v in values.items()},
                    "derivative_energy":{k:_derivative_energy(v,times) for k,v in derivatives.items()},
                    "t0_pointwise_one_sided_FD_minus_analytic_dt":{
                        k:((-3*values[k][0]+4*values[k][1]-values[k][2])/(2*h)-derivatives[k][0]).tolist()
                        for k in ("rho","jx","jy","K","f")}})
    finally: model.train(training)
    a,b=report["grids"]
    report["fine_minus_coarse_derivative_energy"]={k:b["derivative_energy"][k]["time_integrated_probe_mean_square"]-
        a["derivative_energy"][k]["time_integrated_probe_mean_square"] for k in a["derivative_energy"]}
    json.dumps(report,allow_nan=False)
    return report


def t0_balances(model,full,condition):
    from pde_agent_verifier_solver.plugins.vpfp.independent_physics import audit_independent_physics
    _,xy=old.physical_points(model,old.design())
    points=np.column_stack((np.full(len(xy),model.config.time_domain[0]),xy))
    result=audit_independent_physics(model,full,condition,points_txy=points,spatial_times=[],spatial_grids=(),
        chunk_size=32,relative_fd_step=2.**-9,include_l2=False)
    result["time_FD_stencil"]="t0: (-3M(t0)+4M(t0+h)-M(t0+2h))/(2h); also h/2; no outside-time query"
    return result


def _refine(plan,output,device):
    from pde_agent_verifier_solver.plugins.vpfp.reference_refinement import refine_reference_samples
    domain=plan["domain_coordinate_receipt"]
    times=domain["time_domain"];lengths=domain["numerical_x_domain"]
    points=[[times[0]+fraction*(times[1]-times[0]),.173*lengths[0],.389*lengths[1],0.,0.]
        for fraction in old.design()["time_fractions"]]
    return refine_reference_samples(old.SOURCE,"calibration",expected_manifest_sha256=old.SOURCE_SHA,
        rows=[plan["current_input_plan"]["refinement_selected"]["row"]],max_cases=1,device=device,
        output=output,points_txyvv=points,resume=(output/"frozen-plan.json").is_file())


def registered_units(plan):
    units=[{"id":"refinement-stratum7","kind":"refinement","stage":"refinement","status":"pending","attempts":[]}]
    for row in plan["selected"][26:]:
        for kind in KINDS:
            units.append({"id":row["id"]+"/"+kind,"carrier_id":row["id"],"kind":kind,"stage":"remaining","status":"pending","attempts":[]})
    for row in plan["selected"][:24]:
        for kind in ("temporal","t0"):
            units.append({"id":row["id"]+"/"+kind,"carrier_id":row["id"],"kind":kind,"stage":kind,"status":"pending","attempts":[]})
    # All time-grid units precede t0 units, keeping explicit stage ordering.
    return sorted(units,key=lambda u:{"refinement":0,"remaining":1,"temporal":2,"t0":3}[u["stage"]])


def validate_resumed_report(report,plan,plan_sha):
    if (report.get("schema")!=SCHEMA or report.get("frozen_plan_sha256")!=plan_sha
            or report.get("historical_result_sha256")!=HISTORY_RESULT_SHA):
        raise ValueError("resume report identity changed")
    expected=registered_units(plan)
    if len(report.get("units",[]))!=len(expected): raise ValueError("resume unit denominator changed")
    for unit,original in zip(report["units"],expected):
        for key in ("id","kind","stage","carrier_id"):
            if unit.get(key)!=original.get(key): raise ValueError("resume fixed unit selection changed")
        if unit["status"]=="completed":
            if not unit.get("attempts") or unit["attempts"][-1]["status"]!="completed":
                raise ValueError("completed unit has no complete attempt receipt")
            payload=json.loads(Path(unit["attempts"][-1]["report"]).read_text())
            if payload.get("execution_passed") is not True: raise ValueError("completed unit report failed")


def run(output,*,device="cuda",wall_seconds=21600.,resume=False,stage="all",preflight_only=False,retry_failed=False):
    if not os.environ.get("SLURM_JOB_ID","").isdigit() or not os.environ.get("SLURM_TMPDIR"):
        raise ValueError("compute allocation required, including CPU preflight")
    if device not in ("cpu","cuda") or stage not in STAGES or not math.isfinite(wall_seconds) or wall_seconds<=0:
        raise ValueError("explicit valid stage/device and positive finite allocation wall budget required")
    output=Path(output).absolute()
    if output.is_symlink() or not output.is_relative_to(old.SCRATCH): raise ValueError("explicit original scratch output required")
    for source in (HISTORY,old.SOURCE,old.MODEL_ROOT,old.AUDIT_ROOT):
        if output==source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("output overlaps frozen historical inputs")
    if resume:
        if preflight_only or not (output/"frozen-plan.json").is_file() or not (output/"result.json").is_file():
            raise ValueError("explicit resume needs existing complete plan and progress receipt")
    elif output.exists(): raise ValueError("fresh output required without --resume")
    else: output.mkdir(parents=True)
    started=time.monotonic();deadline=started+wall_seconds
    # Failed resume validation must not erase the previous progress registry.
    progress_path=output/(f"resume-preflight-{time.time_ns()}.json" if resume else "result.json")
    report={"schema":SCHEMA,"execution_passed":False,"requested_stage_execution_passed":False,"stage":"domain_preflight",
        "historical_result_sha256":HISTORY_RESULT_SHA,"historical_completed_carriers":26,"registered_carrier_count":40,
        "historical_terminal":{"job_id":"5289599","state":"FAILED","exit_code":"124","preserved":True},
        "admitted":False,"physical_regime_qualified":False,"confirmation_512_completed":False,"numerically_uncertain":True,
        "new_training":False,"new_generated_answers":0,"same_seed_models_paired":True,
        "statistical_unit":completion_design()["statistical_unit"],
        "units":registered_units({"selected":old.registered_slots()}),"invocations":[],"failures":[]}
    plan=None
    def check():
        if time.monotonic()>=deadline: raise TimeoutError("allocation deadline; fixed uncompleted units retained for explicit resume")
    def save():
        report["actual_invocation_wall_seconds"]=time.monotonic()-started
        complete={u["carrier_id"] for u in report["units"] if u["stage"]=="remaining" and all(
            v["status"]=="completed" for v in report["units"] if v.get("carrier_id")==u["carrier_id"] and v["stage"]=="remaining")}
        report["completed_carrier_count"]=26+len(complete)
        old.write_json(progress_path,report)
    try:
        domain=cheap_domain_preflight()
        if preflight_only:
            report.update(stage="domain_preflight_complete",domain_coordinate_receipt=domain,requested_stage_execution_passed=True,
                preflight_only=True,physics_executed=False,full_input_freeze_completed=False)
            save();return report
        check();plan,arrays,manifest,spec=freeze_completion(check)
        if domain!=plan["domain_coordinate_receipt"]: raise ValueError("cheap and fully bound domain receipts differ")
        if resume:
            prior_plan=json.loads((output/"frozen-plan.json").read_text())
            if prior_plan!=plan: raise ValueError("resume frozen history/input/current implementation changed")
            report=json.loads((output/"result.json").read_text())
            validate_resumed_report(report,plan,old.existing.file_sha(output/"frozen-plan.json"))
            progress_path=output/"result.json"
        else:
            old.write_json(output/"frozen-plan.json",plan)
            report.update(frozen_plan_sha256=old.existing.file_sha(output/"frozen-plan.json"),
                domain_coordinate_receipt=domain,units=registered_units(plan),design=completion_design(),
                historical_slots=plan["historical_slots"],historical_refinement_failure=plan["historical_refinement_failure"])
        report["invocations"].append({"job_id":os.environ["SLURM_JOB_ID"],"stage":stage,"wall_seconds":wall_seconds,
            "explicit_resume":resume,"retry_failed":retry_failed})
        report.update(execution_passed=False,requested_stage_execution_passed=False,final_frozen_recheck_passed=False,
            stage="runtime_after_freeze");save();check()
        runtime=old.existing._runtime(manifest,spec,device)
        for path,digest in runtime.frozen_files.items():
            if plan["files"].get(path)!=digest: raise ValueError("runtime code absent from premeasurement freeze")
        if runtime.model.config.hermite_total_degree!=14: raise ValueError("fixed N14 model required")
        rows={r["id"]:r for r in plan["selected"]}
        for unit in report["units"]:
            if stage!="all" and unit["stage"]!=stage: continue
            if unit["status"]=="completed": continue
            if unit["status"] in ("failed","measurement_failed") and not retry_failed: continue
            check()
            for attempt in unit["attempts"]:
                if attempt["status"]=="running": attempt["status"]="interrupted_previous_invocation"
            path=output/"units"/unit["id"]/f"attempt-{len(unit['attempts']):03d}.json"
            if path.exists(): raise ValueError("unregistered attempt artifact retained; refusing overwrite")
            path.parent.mkdir(parents=True,exist_ok=True)
            attempt={"status":"running","job_id":os.environ["SLURM_JOB_ID"]}
            unit["attempts"].append(attempt);unit["status"]="running";report["stage"]=unit["id"];save()
            before=time.monotonic()
            try:
                if unit["kind"]=="refinement": value=_refine(plan,output/"reference-refinement",device)
                else:
                    row=rows[unit["carrier_id"]]
                    if not row["input_finite"]: raise ValueError("original selected input nonfinite; no replacement")
                    raw,theta=old._tensors(runtime.model,arrays[row["id"]])
                    if unit["kind"]=="independent_physics": value=old._independent(runtime.model,raw,theta,old.design())
                    elif unit["kind"]=="temporal_frequency": value=old.native_temporal_frequency(runtime.model,raw,theta,
                        root_id=row["root_id"],provenance=row,design=old.design())
                    elif unit["kind"]=="temporal": value=extended_temporal(runtime.model,raw,theta,row=row,check=check)
                    else: value=t0_balances(runtime.model,raw,theta)
                old.write_json(path,value)
                attempt.update(status="completed" if value.get("execution_passed") is True else "measurement_failed",
                    report=str(path),sha256=old.existing.file_sha(path),actual_wall_seconds=time.monotonic()-before)
                unit["status"]=attempt["status"]
                if unit["status"]!="completed": report["failures"].append({"unit":unit["id"],"attempt":len(unit["attempts"])-1,"reason":"execution_passed=false"})
            except Exception as error:
                attempt.update(status="incomplete" if isinstance(error,TimeoutError) else "failed",actual_wall_seconds=time.monotonic()-before,
                    failure={"type":type(error).__name__,"reason":str(error),"traceback":traceback.format_exc()})
                unit["status"]=attempt["status"];report["failures"].append({"unit":unit["id"],**attempt["failure"]})
                if isinstance(error,TimeoutError): raise
            finally: save()
        check();old.existing.check_frozen(plan["files"])
        for row in plan["selected"]:
            if old.existing._input_hash(*arrays[row["id"]])!=row["input_answer_sha256"]:
                raise ValueError("copied original input changed during measurement")
        selected=[u for u in report["units"] if stage=="all" or u["stage"]==stage]
        report.update(stage="invocation_complete",requested_stage_execution_passed=all(u["status"]=="completed" for u in selected),
            execution_passed=all(u["status"]=="completed" for u in report["units"]),final_frozen_recheck_passed=True)
    except Exception as error:
        report.update(execution_passed=False,requested_stage_execution_passed=False,
            failure={"type":type(error).__name__,"reason":str(error),"traceback":traceback.format_exc()})
    finally:
        save()
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    parser.add_argument("--wall-seconds",type=float,default=21600.)
    parser.add_argument("--stage",choices=STAGES,default="all")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--retry-failed",action="store_true")
    parser.add_argument("--preflight-only",action="store_true")
    args=parser.parse_args(argv)
    report=run(args.output,device=args.device,wall_seconds=args.wall_seconds,resume=args.resume,
        stage=args.stage,preflight_only=args.preflight_only,retry_failed=args.retry_failed)
    return 0 if report["requested_stage_execution_passed"] else 1


if __name__=="__main__": raise SystemExit(main())
