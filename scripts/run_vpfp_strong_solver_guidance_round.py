"""Frozen IC40 -> current public-only solver feedback -> actual score guidance.

Research development round, not automatic physical qualification or deployment.
Only runs in a Slurm compute allocation. Resume preserves registered roots,
completed real solver labels and actual optimizer/random state.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(ROOT/"scripts"))
SCHEMA="vpfp_frozen_ic40_actual_solver_guidance_round_v1"


def file_sha(path):
    path=Path(path)
    if not path.is_file() or path.is_symlink(): raise ValueError("regular nonsymlinked input required")
    digest=sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()


def json_sha(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+".partial")
    temporary.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n")
    temporary.replace(path)


def validate_config(cfg):
    from vpfp_strong_portable_binding import portable_config
    portable_config(cfg)
    from vpfp_strong_guidance_rollouts import PUBLIC_ACCESS_MODES
    if cfg.get("public_access_mode", "materialized_legacy_v1") not in PUBLIC_ACCESS_MODES:
        raise ValueError("invalid public access mode")
    if cfg.get("schema")!="vpfp_frozen_ic40_solver_guidance_round_v1": raise ValueError("wrong round schema")
    for name in ("fit","validation","selection","confirmation"):
        n=cfg["counts"][name]
        if type(n) is not int or n<8 or n%8: raise ValueError("positive balanced8-stratum counts required")
    if cfg["base_weights_trainable"] or cfg["solver_weights_or_code_trainable"]:
        raise ValueError("base and solver must remain frozen")
    if cfg["failed_solver_outputs_are_high_rewards"] or cfg["old_scores_used_as_training_labels"]:
        raise ValueError("only new finite solver errors may be rewards")
    strengths=cfg["sampling"]["selection_strengths"]
    if (not strengths or strengths[0]!=0 or len(set(strengths))!=len(strengths)
            or any(not math.isfinite(float(x)) or x<0 for x in strengths)):
        raise ValueError("unique nonnegative strengths including baseline required")
    if cfg["preflight_fit_count"]<1 or cfg["preflight_fit_count"]>cfg["counts"]["fit"]:
        raise ValueError("preflight is a registered prefix, not replacement roots")
    return cfg


def load_round_base(cfg, assets, *, device):
    """An explicit portable choice never falls back to another codec."""
    from vpfp_strong_portable_binding import portable_config, validate_portable_proof, validate_loaded_portable
    entries=assets["assets"]
    if portable_config(cfg) is None:
        from pde_agent_verifier_solver.plugins.vpfp.structured_diffusion import load_vpfp_diffusion_sampler
        return load_vpfp_diffusion_sampler(entries["base_checkpoint"]["path"],
            expected_sha256=cfg["base_checkpoint_sha256"],device=device,steps=cfg["sampling"]["heun_steps"],churn=0.),None
    from pde_agent_verifier_solver.plugins.vpfp.portable_initial_conditioned import load_portable_initial_conditioned_sampler
    binding=validate_portable_proof(cfg,frozen_assets=entries)
    base=load_portable_initial_conditioned_sampler(entries["base_checkpoint"]["path"],
        checkpoint_sha256=cfg["base_checkpoint_sha256"],portable_bundle_path=entries["portable_codec"]["path"],
        portable_bundle_sha256=entries["portable_codec"]["sha256"],device=device,
        sample_steps=cfg["sampling"]["heun_steps"],sample_churn=0.)
    validate_loaded_portable(base,binding)
    return base,binding


def select_strength(results,strengths):
    """Use only selection roots; never reward dropped/failed baseline roots."""
    baseline=results["0"]
    valid_ids={k for k,v in baseline.items() if v.get("valid") is True}
    rows=[];chosen=0.;best_gain=0.
    for strength in strengths:
        current=results[strength_key(strength)]
        paired=sorted(k for k in valid_ids if current.get(k,{}).get("valid") is True)
        eligible=bool(valid_ids) and len(paired)==len(valid_ids) and set(current)==set(baseline)
        gain=(sum(current[k]["error"]-baseline[k]["error"] for k in paired)/len(paired)) if paired else None
        rows.append({"strength":strength,"registered_roots":len(baseline),"baseline_valid":len(valid_ids),
            "candidate_valid":sum(x.get("valid") is True for x in current.values()),"paired_valid":len(paired),
            "eligible_without_lost_baseline_roots":eligible,"paired_mean_error_increase":gain})
        if eligible and gain is not None and gain>best_gain:
            chosen=float(strength);best_gain=gain
    return {"selected_strength":chosen,"rows":rows,"selection_source":"selection_roots_only",
        "criterion":"largest_actual_physical_scale_macro_error_increase_with_no_lost_valid_baseline_root",
        "scientific_admission":False}


def strength_key(value): return format(float(value),"g")


def verify_bound_result(entry):
    path=Path(entry["result_path"])
    report=json.loads(path.read_text())
    if report.get("private_draw"):
        draw=report["private_draw"]
    return report


def retain_interrupted_rollout(slot,attempt,error):
    """Keep a fully scored label while still ending this allocation."""
    measured=getattr(error,"rollout_result",None)
    if (isinstance(measured,dict) and measured.get("validity_label_available") is True
            and measured.get("resident_released") is True):
        verify_bound_result(measured)
        slot.update(measured,status="completed")
        attempt.update(status="completed_before_allocation_interruption",valid=measured["valid"])
    else:
        attempt["status"]="interrupted";slot["status"]="interrupted"


def load_training_rows(entries):
    import torch
    context=[];future=[];condition=[];errors=[];valid=[];ids=[];excluded=[];expected=None
    # Establish layout from a successful validated draw before considering
    # finite inputs whose solver failed; malformed failures cannot set it.
    for slot_id,entry in sorted(entries.items(),key=lambda item:(item[1].get("valid") is not True,item[0])):
        report=verify_bound_result(entry)
        if report.get("validity_label_available") is not True:
            excluded.append({"slot_id":slot_id,"reason":report.get("label_unavailable_reason")
                or "no_verified_solver_numerical_label"});continue
        expected_kind="valid_solver_error" if report.get("valid") is True else "numerical_solver_failure"
        if type(report.get("valid")) is not bool or report.get("label_kind")!=expected_kind:
            raise ValueError("verified solver label kind disagrees with numerical validity")
        if not report.get("private_draw"):
            excluded.append({"slot_id":slot_id,"reason":"no_complete_input_draw"});continue
        sample=torch.load(report["private_draw"]["path"],map_location="cpu",weights_only=True)
        tensors=[sample.get(k) for k in ("initial_context","future_white","equation_conditions")]
        usable=all(isinstance(v,torch.Tensor) and v.ndim==2 and v.shape[0]==1 and v.shape[1]>0
            and v.is_floating_point() and bool(torch.isfinite(v).all()) for v in tensors)
        usable=usable and tuple(tensors[2].shape)==(1,2)
        shapes=tuple(tuple(v.shape) for v in tensors) if usable else None
        if not usable or (expected is not None and shapes!=expected):
            if report.get("valid") is True: raise ValueError("valid solver label has invalid or inconsistent input tensor")
            excluded.append({"slot_id":slot_id,"reason":"nonfinite_or_incompatible_input_draw"});continue
        if expected is None: expected=shapes
        context.append(sample["initial_context"]);future.append(sample["future_white"])
        condition.append(sample["equation_conditions"])
        valid.append(report["valid"] is True);errors.append(report["error"] if report["valid"] else float("nan"));ids.append(slot_id)
    if not future or sum(valid)<8: raise ValueError("at least8 newly measured valid solver labels required")
    return {"initial_context":torch.cat(context),"future_white":torch.cat(future),
        "conditions":torch.cat(condition),"errors":torch.tensor(errors,dtype=torch.float64),
        "valid":torch.tensor(valid,dtype=torch.bool),"root_ids":ids,"excluded_rows":excluded,
        "registered_count":len(entries),"used_input_count":len(ids)}


def train_surrogate(cfg,base,fit,validation,output,label_sha,check):
    import torch
    from pde_agent_verifier_solver.plugins.vpfp.known_initial_prior import _json_sha
    from pde_agent_verifier_solver.plugins.vpfp.solver_error_guidance import (
        SolverErrorGuidanceConfig,make_solver_error_regressor,guidance_payload,regressor_from_payload,
        _validate_guide_state)
    recipe=cfg["training"];parameter=next(base.model.parameters());device=parameter.device
    if (parameter.dtype not in (torch.float32,torch.float64)
            or any(p.dtype!=parameter.dtype or p.device!=device for p in base.model.parameters())):
        raise ValueError("frozen base parameters require one float32/float64 dtype and device")
    torch.manual_seed(cfg["seeds"]["training"])
    gc=SolverErrorGuidanceConfig(base_config=base.model.config,
        base_checkpoint_sha256=cfg["base_checkpoint_sha256"],normalizer_sha256=_json_sha(base.normalizer.to_payload()),
        width=recipe["width"],depth=recipe["depth"],error_scale=recipe["error_scale"],validity_weight=recipe["validity_weight"])
    guide=make_solver_error_regressor(gc,parameter_dtype=parameter.dtype,device=device)
    optimizer=torch.optim.AdamW(guide.parameters(),lr=recipe["learning_rate"])
    rng=torch.Generator(device=device).manual_seed(cfg["seeds"]["training"]+1)
    data=[{k:(v.to(device) if isinstance(v,torch.Tensor) else v) for k,v in rows.items()} for rows in (fit,validation)]
    continuation=output/"continuation.pt";history=[];start=0;best=float("inf")
    if continuation.exists():
        saved=torch.load(continuation,map_location=device,weights_only=True)
        if saved["label_manifest_sha256"]!=label_sha or saved["config_sha256"]!=json_sha(cfg):
            raise ValueError("exact current-label/config training resume required")
        _validate_guide_state(guide,saved["model"])
        guide.load_state_dict(saved["model"],strict=True);optimizer.load_state_dict(saved["optimizer"])
        rng.set_state(saved["training_rng"].cpu());torch.set_rng_state(saved["torch_rng"].cpu())
        if device.type=="cuda": torch.cuda.set_rng_state_all([x.cpu() for x in saved["cuda_rng"]])
        start=saved["step"];best=saved["best_validation"];history=saved["history"]
    def loss(rows,idx,generator):
        return guide.loss(rows["future_white"][idx],rows["conditions"][idx],rows["initial_context"][idx],
            rows["errors"][idx],rows["valid"][idx],generator=generator)
    def snapshot(step,validation_loss):
        return guidance_payload(guide,label_manifest_sha256=label_sha,train_root_ids=fit["root_ids"],
            validation_root_ids=validation["root_ids"],training={"label_source":"actual_public_solver_macro_error",
                "step":step,"validation_loss":validation_loss,"base_trainable":False,"solver_trainable":False})
    output.mkdir(parents=True,exist_ok=True)
    for step in range(start+1,recipe["steps"]+1):
        check();guide.train();idx=torch.randint(len(fit["root_ids"]),(recipe["batch_size"],),generator=rng,device=device)
        optimizer.zero_grad(set_to_none=True);value,_=loss(data[0],idx,rng)
        if not torch.isfinite(value): raise ValueError("nonfinite real-error training loss")
        value.backward();torch.nn.utils.clip_grad_norm_(guide.parameters(),1.0);optimizer.step()
        if step%recipe["checkpoint_every"]==0 or step==recipe["steps"]:
            guide.eval();vrng=torch.Generator(device=device).manual_seed(cfg["seeds"]["training"]+2)
            with torch.no_grad(): v,_=loss(data[1],torch.arange(len(validation["root_ids"]),device=device),vrng)
            v=float(v)
            if not math.isfinite(v): raise ValueError("nonfinite held-out surrogate loss")
            history.append({"step":step,"training_loss":float(value.detach()),"validation_loss":v})
            snap=snapshot(step,v)
            if v<best: best=v;torch.save(snap,output/"best.pt")
            torch.save(snap,output/"last.pt")
            state={"schema":"vpfp_solver_error_training_state_v1","step":step,"model":guide.state_dict(),
                "optimizer":optimizer.state_dict(),"training_rng":rng.get_state(),"torch_rng":torch.get_rng_state(),
                "cuda_rng":torch.cuda.get_rng_state_all() if device.type=="cuda" else [],
                "label_manifest_sha256":label_sha,"config_sha256":json_sha(cfg),"best_validation":best,"history":history}
            torch.save(state,continuation.with_suffix(".pt.partial"));continuation.with_suffix(".pt.partial").replace(continuation)
            write(output/"training.json",{"final_step":step,"target_step":recipe["steps"],"history":history,
                "trained_from_current_solver_labels":True,"base_weights_frozen":True,"label_manifest_sha256":label_sha})
    if not (output/"best.pt").exists(): raise ValueError("no actual trained guidance checkpoint")
    return regressor_from_payload(torch.load(output/"best.pt",map_location="cpu",weights_only=True),device=device)


def run(args):
    import torch
    from vpfp_strong_guidance_rollouts import freeze_round_assets,registered_roots,build_rollout_runtime,evaluate_draw
    from pde_agent_verifier_solver.plugins.vpfp.frozen_solver_reward import load_frozen_solver
    from pde_agent_verifier_solver.plugins.vpfp.solver_error_guidance import draw_frozen_ic40,GuidedIC40Sampler
    if not os.environ.get("SLURM_JOB_ID","").isdigit() or not os.environ.get("SLURM_TMPDIR"):
        raise ValueError("numerical work requires a Slurm compute allocation")
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise ValueError("one visible allocated CUDA device required")
    if not math.isfinite(args.wall_seconds) or args.wall_seconds<=120: raise ValueError("positive usable allocation budget required")
    output=args.output.absolute();scratch=Path("${PDE_GAV_RUN_ROOT}")
    if not output.is_relative_to(scratch) or output==scratch or output.is_symlink():
        raise ValueError("specific separate nonsymlinked scratch directory required")
    if output.exists() and not args.resume: raise ValueError("existing output requires explicit resume")
    cfg=validate_config(json.loads(args.config.read_text()))
    revision=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    def check_source():
        if (subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()!=revision
                or subprocess.check_output(["git","status","--porcelain","--","src","scripts","configs","tests","legacy/vpfp"],cwd=ROOT,text=True).strip()):
            raise ValueError("freeze exact relevant source before numerical work")
    check_source();output.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();deadline=started+args.wall_seconds-45
    def check():
        if time.monotonic()>=deadline: raise TimeoutError("allocation ending; preserve exact labels and training checkpoints")
    report={"schema":SCHEMA,"execution_passed":False,"requested_stage_execution_passed":False,
        "source_revision":revision,"config_sha256":json_sha(cfg),"scientific_admission":False,
        "online_assets_replaced":False,"training_started":False,"invocations":[],"failures":[],"rollouts":{}}
    progress=output/"result.json"
    def save():
        report["actual_invocation_wall_seconds"]=time.monotonic()-started;write(progress,report)
    try:
        if args.resume:
            report=json.loads(progress.read_text())
            if report["schema"]!=SCHEMA or report["source_revision"]!=revision or report["config_sha256"]!=json_sha(cfg):
                raise ValueError("resume requires unchanged source and registered configuration")
        report.update(execution_passed=False,requested_stage_execution_passed=False)
        report["invocations"].append({"job_id":os.environ["SLURM_JOB_ID"],"stage":args.stage,
            "resume":args.resume,"wall_seconds_requested":args.wall_seconds});save();check()
        assets,native_spec=freeze_round_assets(cfg,output)
        roots=registered_roots(cfg)
        all_ids=[row["root_seed"] for rows in roots.values() for row in rows]
        if len(all_ids)!=len(set(all_ids)): raise ValueError("feedback/validation/selection/confirmation roots overlap")
        plan={"schema":SCHEMA,"config":cfg,"source_revision":revision,"assets":assets,"roots":roots,
            "statistical_unit":"independent_root; strengths_paired_within_root; no_prior_scores_as_labels"}
        plan_path=output/"frozen-plan.json"
        if plan_path.exists():
            if json.loads(plan_path.read_text())!=plan: raise ValueError("frozen plan changed")
        else: write(plan_path,plan)
        report["frozen_plan_sha256"]=file_sha(plan_path)
        for split in ("fit","validation","selection"):
            strengths=[0.] if split!="selection" else cfg["sampling"]["selection_strengths"]
            for strength in strengths:
                group=report["rollouts"].setdefault(split,{}).setdefault(strength_key(strength),{})
                for row in roots[split]: group.setdefault(row["slot_id"],{"status":"pending","root":row,"attempts":[]})
        report["registered_roots_by_split"]={s:len(rows) for s,rows in roots.items()};save()
        torch.set_num_threads(min(8,int(os.environ.get("SLURM_CPUS_PER_TASK","8"))))
        torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
        base,portable_binding=load_round_base(cfg,assets,device="cuda")
        if portable_binding is not None:
            if "portable_codec_binding" in report and report["portable_codec_binding"]!=portable_binding:
                raise ValueError("resume cannot change portable codec/normalizer identity")
            report["portable_codec_binding"]=portable_binding
            report["base_sampling_identity"]=base.identity
            save()
        base.model.eval().requires_grad_(False);base.normalizer.eval().requires_grad_(False)
        binding,replay=build_rollout_runtime(native_spec,base,output/"native-runtime",device="cuda")
        runtime=binding.plugin.generator.backend.runtime
        solver=load_frozen_solver(assets["assets"]["solver"]["path"],cfg["solver_sha256"])
        guide=None
        def collect(split,strength,limit=None):
            group=report["rollouts"].setdefault(split,{}).setdefault(strength_key(strength),{})
            chosen_rows=roots[split] if limit is None else roots[split][:limit]
            for row in roots[split]: group.setdefault(row["slot_id"],{"status":"pending","root":row,"attempts":[]})
            sampler=None
            if strength:
                if guide is None: raise ValueError("actual trained guide required before guided rollouts")
                sampling=cfg["sampling"]
                sampler=GuidedIC40Sampler(base,guide,strength=strength,max_gradient_norm=sampling["max_gradient_norm"],
                    sigma_min=sampling["guidance_sigma_min"],sigma_max=sampling["guidance_sigma_max"])
            for row in chosen_rows:
                slot=group[row["slot_id"]]
                if slot["status"]=="completed": verify_bound_result(slot);continue
                check();report["stage"]=f"{split}/strength-{strength_key(strength)}/{row['slot_id']}";save()
                attempt=output/"rollouts"/split/row["slot_id"]/f"strength-{strength_key(strength)}"/f"attempt-{len(slot['attempts']):03d}"
                record={"job_id":os.environ["SLURM_JOB_ID"],"path":str(attempt),"status":"running"}
                slot["attempts"].append(record);slot["status"]="running";save()
                try:
                    theta=torch.tensor([row["conditions"]],device="cuda",dtype=torch.float32)
                    draw=(draw_frozen_ic40(base,[row["root_seed"]],conditions=theta) if sampler is None
                        else sampler.draw([row["root_seed"]],conditions=theta))
                    measured=evaluate_draw(draw,row,runtime=runtime,replay=replay,solver=solver,output=attempt,
                        budget_seconds=cfg["solver_budget_seconds"],check=check,
                        public_access_mode=cfg.get("public_access_mode", "materialized_legacy_v1"))
                    slot.update(measured,status="completed");record.update(status="completed",valid=measured["valid"])
                except TimeoutError as error:
                    retain_interrupted_rollout(slot,record,error);raise
                except Exception as error:
                    record.update(status="failed",failure={"type":type(error).__name__,"reason":str(error)})
                    slot["status"]="failed";raise
                finally: save()
        collect("fit",0.,cfg["preflight_fit_count"] if args.stage=="preflight" else None)
        if args.stage=="preflight":
            done=[report["rollouts"]["fit"]["0"][r["slot_id"]] for r in roots["fit"][:cfg["preflight_fit_count"]]]
            passed=all(x["status"]=="completed" for x in done) and any(x.get("valid") is True for x in done)
            report.update(stage="preflight_complete",requested_stage_execution_passed=passed,
                preflight_finite_labels=sum(x.get("valid") is True for x in done),
                preflight_only=True,execution_passed=False,full_round_completed=False)
        else:
            collect("validation",0.)
            labels={split:report["rollouts"][split]["0"] for split in ("fit","validation")}
            write(output/"label-manifest.json",{"schema":SCHEMA,"config_sha256":json_sha(cfg),"labels":labels,
                "label_source":"actual_current_frozen_solver_predictions_vs_private_macros"})
            label_sha=file_sha(output/"label-manifest.json")
            report.update(stage="guidance_training",training_started=True,label_manifest_sha256=label_sha);save()
            fit,validation=[load_training_rows(labels[k]) for k in ("fit","validation")]
            report["training_input_accounting"]={s:{k:rows[k] for k in ("registered_count","used_input_count","excluded_rows")}
                for s,rows in (("fit",fit),("validation",validation))};save()
            guide=train_surrogate(cfg,base,fit,validation,output/"guidance",label_sha,check)
            report["trained_guidance_checkpoint"]={"path":str(output/"guidance/best.pt"),"sha256":file_sha(output/"guidance/best.pt")}
            for strength in cfg["sampling"]["selection_strengths"]: collect("selection",strength)
            selection=select_strength(report["rollouts"]["selection"],cfg["sampling"]["selection_strengths"])
            write(output/"strength-selection.json",selection);report["selection"]=selection;save()
            selected=selection["selected_strength"]
            for strength in sorted({0.,selected}): collect("confirmation",strength)
            confirmation=select_strength(report["rollouts"]["confirmation"],sorted({0.,selected}))
            confirmation["selected_before_confirmation"]=selected
            confirmation["confirmation_not_used_to_reselect_or_refit"]=True
            report["confirmation"]=confirmation
            report["guidance_improved_actual_heldout_solver_error"]=(selected>0 and confirmation["selected_strength"]==selected)
            report.update(stage="complete",requested_stage_execution_passed=True,execution_passed=True,
                preflight_only=False,full_round_completed=True)
        check_source()
        report["frozen_assets_final_verified"]=True
    except Exception as error:
        report.update(execution_passed=False,requested_stage_execution_passed=False)
        report["failures"].append({"stage":report.get("stage"),"type":type(error).__name__,
            "reason":str(error),"traceback":traceback.format_exc()})
    finally: save()
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/vpfp_strong_solver_guidance_demand_20260908.json")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--stage",choices=("preflight","all"),default="all")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--wall-seconds",type=float,default=21000)
    args=parser.parse_args(argv)
    return 0 if run(args)["requested_stage_execution_passed"] else 1


if __name__=="__main__": raise SystemExit(main())
