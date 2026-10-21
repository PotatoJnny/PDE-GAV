"""Forty frozen strong B carriers: initial-layer checks, not fresh confirmation.

The researcher receives per-root full-velocity integrals, coordinate-FD balance
checks and sparse time spectra. One predeclared calibration root is re-solved
at two resolutions. No training, candidate selection or new generation occurs.
The enclosing Slurm launcher owns the hard three-hour process deadline.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
sys.path.insert(0, str(ROOT/"scripts"))
import run_vpfp_regime_fidelity_coverage as existing
from run_vpfp_independent_confirmation import write_json
from run_vpfp_independent_physics import first_stratum_rows, native_temporal_frequency, physical_points

SCHEMA = "vpfp_frozen_strong_initial_layer_development_audit_v1"
SCRATCH = Path("${PDE_GAV_RUN_ROOT}")
SOURCE = SCRATCH/"vpfp-strong-main-5244223/corpus"
SOURCE_SHA = "4f59a2101eea8eb3ba958250bf8f5bb9faa7d883a658aaa76a6c69f8c010e78d"
MODEL_ROOT = SCRATCH/"vpfp-strong-regime-coherent-5268586"
AUDIT_ROOT = SCRATCH/"vpfp-strong-regime-followup-5286256/followup/regime-audit"
AUDIT_SHA = "6af6ea4418a9ec5eadd6dbc52209e1e5380c4505013bceabe04135551074b5f4"


def candidates():
    return [
        {"id":"seed-0-initial-40k", "target":"target-05", "replica":0,
         "arm":"initial_conditioned", "step":40000,
         "sha256":"255b8f77e418568b4d48b3669b95aa706cd83d94cee76d291806ae4e55794f62"},
        {"id":"seed-1-initial-40k", "target":"target-12", "replica":1,
         "arm":"initial_conditioned", "step":40000,
         "sha256":"fac4b591d13afb75b744ed0775d5cf1eeb142d3b0cb9ef4665b9d50b1f7c3d5a"},
        {"id":"seed-0-rho-cov-20k", "target":"target-06", "replica":0,
         "arm":"initial_conditioned_rho_cov", "step":20000,
         "sha256":"78c60b944bcccf27ca74c455fef1c8b9cb59817dde19a5bf50d10f3454e4a568"},
        {"id":"seed-1-rho-cov-20k", "target":"target-13", "replica":1,
         "arm":"initial_conditioned_rho_cov", "step":20000,
         "sha256":"08fe4015e21627e91d4086363c062bde337f6dab898e2b1fcad6b80983a3542f"}]


def candidate_directory(item):
    return MODEL_ROOT/f"seed-{item['replica']}"/item["arm"]/f"step-{item['step']:06d}"/"evaluation"


def all_audited_states():
    paths = []
    for seed in range(2):
        for arm in ("cov_base", "initial_conditioned", "initial_conditioned_rho_cov"):
            for step in ((20000,) if arm.endswith("rho_cov") else (20000,30000,40000)):
                paths.append(str(MODEL_ROOT/f"seed-{seed}"/arm/f"step-{step:06d}"/"evaluation"))
    return {f"target-{i:02d}": path for i,path in enumerate(paths)}


def design():
    return {"schema":SCHEMA, "time_fractions":[.0078125,.0234375,.7578125],
        "spatial_probe_fractions":[[.173,.389]], "physical_velocity_probes":[[0.,0.]],
        "probe_geometry":"one_fixed_paired_sparse_lab_probe_not_full_space_qualification",
        "relative_fd_step":2.**-9, "spatial_mass_grids":[17,33], "include_l2":True,
        "time_counts":[65,129], "velocity_moment_orders":[17,19], "velocity_L2_orders":[32,34],
        "phase_chunk_size":32, "reference_role":"calibration", "carrier_count":40,
        "reference_root_count":8, "candidate_count":4, "draws_per_candidate":8,
        "selection":"first_original_row_per_actual_condition_stratum_no_failed_row_replacement",
        "execution_order":"8_reference_carriers_then_stratum7_refinement_then32_candidate_carriers",
        "refinement":{"stratum_id":7, "root_count":1, "levels":"same_seed_coarse_and_default_same_domain_refined",
            "maximum_PDE_solve_attempts":2, "raw_vs_projected_comparison_kept_separate":True},
        "first_reference_is_reused_benchmark_not_extra_sample":True,
        "automatic_downsampling":False, "scientific_admission":False,
        "pending":["t0_one_sided_FD", "independent_global_spatial_source_momentum_and_energy_integrals",
            "uniform_space_time_velocity_convergence", "fresh512_population_confirmation"]}


def validate_completed_audit(document):
    expected = {"schema":"vpfp_frozen_regime_fidelity_coverage_run_v3", "execution_passed":True,
        "all_requested_physical_adapters_completed":True, "confirmation_512_completed":False,
        "admitted":False, "physical_regime_qualified":False, "new_training":False,
        "new_generated_answers":0, "pde_solves":0, "sample_count":128,
        "stage":"complete", "source_manifest_sha256":SOURCE_SHA,
        "model_source_manifest_sha256":SOURCE_SHA, "reference_source_manifest_sha256":SOURCE_SHA,
        "evidence_role":"previously_opened_development_answers_not_blind_confirmation"}
    if any(document.get(k) != v for k,v in expected.items()):
        raise ValueError("original completed128 descriptive audit identity/status required")
    rows = document.get("targets", {})
    expected_paths = all_audited_states()
    if set(rows) != set(expected_paths):
        raise ValueError("all14 original audited model states required")
    for label,path in expected_paths.items():
        row = rows[label]
        if (row.get("directory") != path or row.get("status") != "descriptive_audit_completed"
                or row.get("attempted_count") != 128 or row.get("failed_count") != 0
                or row.get("admitted") is not False
                or row.get("audit_path") != str(AUDIT_ROOT/(label+"-audit.json"))):
            raise ValueError("original14-state result does not match fixed model schedule")


def select_first_rows(conditions, edges):
    values = np.asarray(conditions)
    if values.dtype != np.dtype("float32") or values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("original finite FP32 condition rows required before feature-free selection")
    groups = existing._strata(values,edges)
    return first_stratum_rows([{"stratum_id":int(group)} for group in groups]), groups


def registered_slots():
    groups = ["reference-calibration", *(item["id"] for item in candidates())]
    return [{"id":f"{group}/stratum-{stratum}", "group":group, "stratum_id":stratum,
        "status":"pending", "independent_physics":{"status":"pending"},
        "temporal_frequency":{"status":"pending"}} for group in groups for stratum in range(8)]


def validate_generation_design(provenance):
    prior = provenance.get("design",{})
    seeds = prior.get("sampling_seeds")
    if (provenance.get("known_initial_seed_applied") is not True or not isinstance(seeds,list)
            or len(seeds)!=128 or len(set(seeds))!=128
            or any(type(seed) is not int or seed<0 for seed in seeds)
            or prior.get("known_initial_seeds") != existing._native_stream("vpfp-initial-conditioned-prior",seeds)):
        raise ValueError("original known-initial generation streams required")
    return prior,existing._native_stream("vpfp-initial-conditioned-future",seeds)


def measured_original_rows(measure, old_target, target):
    selected = old_target.get("selected_rows",[])
    if selected != target["selected_rows"] or len(selected)!=128 or len(set(selected))!=128:
        raise ValueError("original audited row-selection mapping differs")
    cases = measure.get("cases",[])
    if len(cases)!=128 or {case["row"] for case in cases}!=set(range(128)):
        raise ValueError("all128 original measurement row receipts required")
    return {selected[case["row"]]:case["input_answer_sha256"] for case in cases}


def _input_row(group, row, raw, condition, root_id, seed, *, checkpoint_sha=None, initial_seed=None):
    value,theta = np.array(raw,copy=True),np.array(condition,copy=True)
    if value.ndim != 1 or value.dtype != np.dtype("float32") or theta.dtype != np.dtype("float32") or theta.shape != (2,):
        raise ValueError("one complete native FP32 coefficient/condition row required")
    receipt = {"group":group,"row":int(row),"root_id":str(root_id),"source_manifest_sha256":SOURCE_SHA,
        "data_origin":"physical_reference" if group == "reference-calibration" else "generated_answer",
        "split":"calibration" if group == "reference-calibration" else "previously_opened_generation",
        "source_seed":int(seed),"known_initial_seed":initial_seed,"checkpoint_sha256":checkpoint_sha,
        "conditions":theta.tolist(),"condition_fp32_bits":theta.view(np.uint32).tolist(),
        "input_answer_sha256":existing._input_hash(value,theta),
        "native_full_sha256":sha256(value.tobytes()).hexdigest(),
        "input_finite":bool(np.isfinite(value).all() and np.isfinite(theta).all())}
    return receipt, (value,theta)


def freeze_inputs(check_deadline=lambda:None):
    """Freeze all40 selections, code and four checkpoint bytes before measuring."""
    files = {}
    def freeze(path, expected=None):
        check_deadline()
        path = Path(path).absolute()
        digest = existing.file_sha(path)
        files[str(path)] = digest
        return path
    def document(path, expected=None):
        return json.loads(freeze(path,expected).read_text())
    completed = document(AUDIT_ROOT/"result.json", AUDIT_SHA)
    validate_completed_audit(completed)
    old_plan = document(AUDIT_ROOT/"frozen-plan.json")
    training_plan = document(MODEL_ROOT/"frozen-plan.json")
    if (old_plan.get("source_manifest_sha256") != SOURCE_SHA or old_plan.get("sample_count") != 128
            or old_plan.get("source") != str(SOURCE) or old_plan.get("model_source") != str(SOURCE)
            or old_plan.get("reference_source") != str(SOURCE)):
        raise ValueError("completed audit plan source differs from original strong corpus")
    manifest = existing.read_manifest(SOURCE,verify_hashes=False)
    if (manifest["manifest_sha256"] != SOURCE_SHA or manifest["regime_id"] != "long_time_strong_collision_v2"
            or manifest["condition_order"] != ["collision_nu","temperature"]
            or manifest["latent_dimension"] != 652672 or manifest["splits"]["calibration"]["count"] != 128):
        raise ValueError("fixed original strong native corpus required")
    freeze(SOURCE/"manifest.json")
    freeze(SOURCE/manifest["seed_plan"]["path"],manifest["seed_plan"]["sha256"])
    for item in manifest["splits"]["calibration"]["files"].values():
        freeze(SOURCE/item["path"],item["sha256"])
    spec = existing.VPFPOnlineModelSpec.from_mapping(manifest["model_spec"])
    for item in spec.preflight()["checkpoints"].values():
        freeze(item["path"],item["sha256"])
    edges = existing._edges(np.linspace(*spec.collision_nu_range,5),np.linspace(*spec.temperature_range,3))
    split = existing.load_split(SOURCE,"calibration",verify_hashes=False)
    chosen,groups = select_first_rows(split.conditions,edges)
    if not np.array_equal(groups,np.asarray(split.stratum_ids)):
        raise ValueError("reference strata differ from physical conditions")
    selected,arrays = [],{}
    for group,row in enumerate(chosen):
        receipt,value = _input_row("reference-calibration",row,split.latents[row],split.conditions[row],
            split.case_ids[row],split.sample_ids[row])
        receipt.update(id=f"reference-calibration/stratum-{group}",stratum_id=group)
        selected.append(receipt); arrays[receipt["id"]] = value
    if len({item["root_id"] for item in selected}) != 8 or len({item["source_seed"] for item in selected}) != 8:
        raise ValueError("eight distinct calibration roots required")
    original_registry = {item["directory"]:item for item in old_plan["generated_targets"]}
    if len(original_registry)!=14 or set(original_registry)!=set(all_audited_states().values()):
        raise ValueError("completed plan must contain all14 original target directories")
    generation_designs,paired_designs = {},{}
    for item in candidates():
        directory = candidate_directory(item)
        checkpoint = directory.parent.parent/"checkpoints"/f"fixed-step-{item['step']:06d}.pt"
        freeze(checkpoint,item["sha256"])
        old_target = original_registry.get(str(directory),{})
        if old_target.get("checkpoint_sha256") != item["sha256"]:
            raise ValueError("checkpoint differs from completed descriptive audit registry")
        target = existing.freeze_generated(directory,manifest=manifest,spec=spec,count=128,edges=edges)
        if target["status"] != "ready_native_direct" or len(target["raw"]) != 128 or target["checkpoint_sha256"] != item["sha256"]:
            raise ValueError("fixed128 old-B target/checkpoint required, not a new512 plan or A carrier")
        files.update(target["files"])
        if any(old_plan["files"].get(path)!=digest for path,digest in target["files"].items()):
            raise ValueError("generated input bytes differ from completed audit frozen plan")
        provenance = document(directory/"generated-provenance.json")
        generation,future_seeds = validate_generation_design(provenance)
        seeds = generation["sampling_seeds"]
        replica = f"seed-{item['replica']}"
        if generation != training_plan["generation_design"].get(replica):
            raise ValueError("generation differs from original frozen training-run design")
        if replica in paired_designs and generation != paired_designs[replica]:
            raise ValueError("same-seed model interventions lost their common original noise/initial roots")
        paired_designs[replica] = generation; generation_designs[item["id"]] = generation
        measure = document(AUDIT_ROOT/(item["target"]+"-measurements.json"))
        audit = document(AUDIT_ROOT/(item["target"]+"-audit.json"))
        if audit.get("checkpoint_sha256") != item["sha256"] or audit.get("generated_directory") != str(directory) or audit.get("execution_passed") is not True:
            raise ValueError("completed model audit checkpoint identity differs")
        measured = measured_original_rows(measure,old_target,target)
        rows,bins = select_first_rows(target["conditions"],edges)
        for group,row in enumerate(rows):
            receipt,value = _input_row(item["id"],row,target["raw"][row],target["conditions"][row],
                target["roots"][row],seeds[row],checkpoint_sha=item["sha256"],
                initial_seed=None if target["known_initial_seeds"] is None else target["known_initial_seeds"][row])
            if measured.get(row) != receipt["input_answer_sha256"]:
                raise ValueError("selected native answer differs from completed5286256 measurement")
            receipt.update(id=f"{item['id']}/stratum-{group}",stratum_id=group,
                checkpoint=str(checkpoint),completed_audit_target=item["target"],model_replica=item["replica"],
                future_noise_seed=future_seeds[row],paired_intervention_id=f"{replica}:original-draw:{seeds[row]}",
                future_seed_source="exact_existing_vpfp-initial-conditioned-future_SHA_tag_no_new_RNG")
            selected.append(receipt); arrays[receipt["id"]] = value
    # Only repository-local implementation files; no external arbitrary search.
    for directory in (ROOT/"src/pde_agent_verifier_solver/plugins/vpfp",ROOT/"legacy/vpfp/src/agent_verifier/vpfp"):
        for path in sorted(directory.rglob("*.py")): freeze(path)
    for name in (Path(__file__).name,"run_vpfp_regime_fidelity_coverage.py","run_vpfp_independent_physics.py",
            "run_vpfp_independent_confirmation.py","diagnose_vpfp_known_initial.py","train_vpfp_structured_diffusion.py"):
        freeze(ROOT/"scripts"/name)
    plan = {"schema":SCHEMA,"design":design(),"completed_audit":{"path":str(AUDIT_ROOT/"result.json"),"sha256":AUDIT_SHA},
        "source":str(SOURCE),"source_manifest_sha256":SOURCE_SHA,"files":files,"selected":selected,
        "checkpoint_candidates":candidates(),"refinement_selected":selected[7],
        "generation_designs":generation_designs,"same_seed_models_paired":True,
        "statistical_unit":"8_calibration_roots_plus_paired_model_noise_interventions_not40_independent_roots",
        "evidence_role":"opened128_development_subsets_not_fresh512",
        "same_sampling_or_initial_seeds_across_models_are_not_independent_roots":True,
        "scientific_admission":False,"new_training":False,"new_generation":False}
    if len(selected) != 40 or len(arrays) != 40:
        raise ValueError("exact40 slots must freeze before any measurement")
    existing.check_frozen(files)
    return plan,arrays,manifest,spec


def _independent(model,full,condition,fixed):
    from pde_agent_verifier_solver.plugins.vpfp.independent_physics import audit_independent_physics
    points,_ = physical_points(model,fixed)
    return audit_independent_physics(model,full,condition,points_txy=points,spatial_times=points[:,0],
        spatial_grids=tuple(fixed["spatial_mass_grids"]),chunk_size=fixed["phase_chunk_size"],
        relative_fd_step=fixed["relative_fd_step"],include_l2=True)


def _tensors(model,values):
    import torch
    parameter = next(model.parameters())
    return (torch.tensor(values[0],device=parameter.device,dtype=parameter.dtype)[None],
            torch.tensor(values[1],device=parameter.device,dtype=parameter.dtype))


def _refinement(plan,model,output,device):
    from pde_agent_verifier_solver.plugins.vpfp.reference_refinement import refine_reference_samples
    points,_ = physical_points(model,plan["design"])
    vv = np.tile(plan["design"]["physical_velocity_probes"],(len(plan["design"]["time_fractions"]),1))
    return refine_reference_samples(SOURCE,"calibration",expected_manifest_sha256=SOURCE_SHA,
        output=output,rows=[plan["refinement_selected"]["row"]],max_cases=1,device=device,
        points_txyvv=np.column_stack((points,vv)))


def run(output,*,device="cuda",wall_seconds=10800):
    if not os.environ.get("SLURM_JOB_ID","").isdigit() or not os.environ.get("SLURM_TMPDIR"):
        raise ValueError("separate compute allocation required, including CPU")
    if device not in ("cpu","cuda") or not math.isfinite(wall_seconds) or not 0<wall_seconds<=10800:
        raise ValueError("cpu/cuda and fixed wall budget in (0,10800] seconds required")
    output = Path(output).absolute()
    if output.exists() or output.is_symlink() or not output.is_relative_to(SCRATCH):
        raise ValueError("fresh original scratch output required")
    if any(output==p or output.is_relative_to(p) or p.is_relative_to(output) for p in (SOURCE,MODEL_ROOT,AUDIT_ROOT)):
        raise ValueError("output must not overlap frozen inputs")
    output.mkdir(parents=True)
    started = time.monotonic(); deadline = started+wall_seconds
    report = {"schema":SCHEMA,"execution_passed":False,"stage":"freeze_inputs",
        "admitted":False,"physical_regime_qualified":False,"numerically_uncertain":True,
        "confirmation_512_completed":False,"new_training":False,"new_generated_answers":0,
        "same_seed_models_paired":True,"registered_maximum_new_PDE_solve_attempts":2,
        "statistical_unit":"8_calibration_roots_plus_paired_model_noise_interventions_not40_independent_roots",
        "registered_carrier_count":40,"slots":registered_slots(),"failures":[],
        "refinement":{"status":"pending","stratum_id":7,"registered_root_count":1},
        "design":design(),"wall_budget_seconds":wall_seconds,"evidence_role":"opened_development_not_blind_confirmation"}
    def save():
        report["actual_wall_seconds"] = time.monotonic()-started
        report["completed_carrier_count"] = sum(row["status"]=="completed" for row in report["slots"])
        write_json(output/"result.json",report)
    def check():
        if time.monotonic()>=deadline: raise TimeoutError("fixed3h budget exhausted; unstarted slots remain pending, no replacement")
    def unit(slot,key,action):
        check(); state=slot[key]; state["status"]="running"; report["stage"]=slot["id"]+"/"+key; save()
        begin=time.monotonic()
        path=output/slot["id"]/(key+".json"); path.parent.mkdir(parents=True,exist_ok=True)
        try:
            value=action(); write_json(path,value)
            state.update(status="completed" if value.get("execution_passed") is True else "measurement_failed",
                report=str(path),sha256=existing.file_sha(path),actual_wall_seconds=time.monotonic()-begin)
            if key=="refinement":
                state.update(new_pde_solves=value.get("new_pde_solves"),new_pde_solve_attempts=value.get("new_pde_solve_attempts"))
            if state["status"]!="completed": report["failures"].append({"unit":report["stage"],"reason":"measurement returned execution_passed=false"})
        except Exception as error:
            failure={"type":type(error).__name__,"reason":str(error),"traceback":traceback.format_exc()}
            state.update(status="failed",failure=failure,actual_wall_seconds=time.monotonic()-begin)
            report["failures"].append({"unit":report["stage"],**failure})
            if isinstance(error,TimeoutError): raise
        finally: save()
    save(); plan=None
    try:
        plan,arrays,manifest,spec = freeze_inputs(check)
        write_json(output/"frozen-plan.json",plan)
        report["frozen_plan_sha256"]=existing.file_sha(output/"frozen-plan.json")
        for slot,selected in zip(report["slots"],plan["selected"]):
            if slot["id"]!=selected["id"]: raise ValueError("frozen selection differs from pre-registered40 slots")
            slot["input"]=selected
        save(); check()
        runtime = existing._runtime(manifest,spec,device)
        if int(runtime.model.config.hermite_total_degree)!=14:
            raise ValueError("fixed N14 decoder required")
        for path,digest in runtime.frozen_files.items():
            if plan["files"].get(path)!=digest: raise ValueError("runtime source absent from premeasurement freeze: "+path)
        existing.check_frozen(plan["files"])
        for index,slot in enumerate(report["slots"]):
            if index==8:
                surrogate={"id":"reference-stratum7-refinement","refinement":report["refinement"]}
                unit(surrogate,"refinement",lambda:_refinement(plan,runtime.model,output/"reference-refinement",device))
            check(); row=slot["input"]
            if not row["input_finite"]:
                slot.update(status="input_failed",reason="selected original row is nonfinite; no replacement")
                report["failures"].append({"unit":slot["id"],"reason":slot["reason"]});save();continue
            raw,condition = _tensors(runtime.model,arrays[slot["id"]])
            slot["status"]="running"
            unit(slot,"independent_physics",lambda:_independent(runtime.model,raw,condition,plan["design"]))
            unit(slot,"temporal_frequency",lambda:native_temporal_frequency(runtime.model,raw,condition,
                root_id=row["root_id"],provenance=row,design=plan["design"]))
            slot["status"]="completed" if all(slot[k]["status"]=="completed" for k in ("independent_physics","temporal_frequency")) else "measurement_failed"
            if index==0:
                report["first_reference_benchmark"]={"slot":slot["id"],"is_extra_root":False,
                    "independent_physics_seconds":slot["independent_physics"]["actual_wall_seconds"],
                    "temporal_frequency_seconds":slot["temporal_frequency"]["actual_wall_seconds"]}
            save()
        report["stage"]="recheck_frozen_inputs";save();check()
        existing.check_frozen(plan["files"])
        for row in plan["selected"]:
            raw,theta=arrays[row["id"]]
            if existing._input_hash(raw,theta)!=row["input_answer_sha256"]:
                raise ValueError("selected copied input changed during measurement")
        report.update(stage="complete",execution_passed=not report["failures"] and
            all(x["status"]=="completed" for x in report["slots"]) and report["refinement"]["status"]=="completed")
    except Exception as error:
        report.update(execution_passed=False,failure={"type":type(error).__name__,"reason":str(error),"traceback":traceback.format_exc()})
    finally:
        for slot in report["slots"]:
            if slot["status"]=="running": slot["status"]="incomplete"
        save()
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    parser.add_argument("--wall-seconds",type=float,default=10800.)
    args=parser.parse_args(argv)
    return 0 if run(args.output,device=args.device,wall_seconds=args.wall_seconds)["execution_passed"] else 1


if __name__=="__main__":
    raise SystemExit(main())
