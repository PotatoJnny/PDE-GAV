"""Frozen weak-model, same-condition macro/micro interventions; no training.

All descendants retain the eight selected validation-root identities. A/B
crosses are counterfactual carriers, not independent samples or truth errors.
Run only in a compute allocation; the outer launcher owns the hard wall timeout.
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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
sys.path.insert(0, str(ROOT/"scripts"))
from run_vpfp_independent_confirmation import checked_asset, file_sha, json_sha, write_json

SCHEMA = "vpfp_frozen_weak_dynamics_cross_diagnostic_v1"
SOURCE = Path("${PDE_GAV_RUN_ROOT}/vpfp-scaleup-weak-5241246/corpus")
SOURCE_SHA = "0e322cd70bdd2558e0512911e2199a45acff964499ba3ba6a18b901b432e6db8"
MODEL_ROOT = Path("${PDE_GAV_RUN_ROOT}/vpfp-weak-regime-coherent-5268892")
MODEL_MANIFESTS = (
    {"path":str(MODEL_ROOT/"seed-0/native-moving-frame-model.json"),
     "sha256":"6f8df218e67b3111038d766ea1535be53a3477a2a0205b5002fdcba601bc5c4d"},
    {"path":str(MODEL_ROOT/"seed-1/native-moving-frame-model.json"),
     "sha256":"b516164c7b82c0563786bb567ad6e3c7144238823c29b83277679e86e22b4c56"},
)
ARMS = ("real_m_real_z", "real_m_generated_z", "generated_m_real_z", "generated_m_generated_z")
MASTER_SEED = 2026090801
SCRATCH_ROOT = Path("/scratch")


def array_sha(value):
    value = np.ascontiguousarray(value)
    header = json.dumps({"dtype":value.dtype.str, "shape":list(value.shape)}, sort_keys=True).encode()
    return sha256(header+b"\0"+value.tobytes()).hexdigest()


def tensor_sha(value):
    return array_sha(value.detach().cpu().numpy())


def stream(tag, value):
    return int.from_bytes(sha256(f"{tag}:{value}".encode()).digest()[:8], "little") % 2**63


def select_validation_rows(split):
    first = {}
    for index, group in enumerate(split.stratum_ids):
        if type(group.item() if hasattr(group,"item") else group) is not int or not 0 <= int(group) < 8:
            raise ValueError("validation requires all eight integer condition strata")
        first.setdefault(int(group), index)
    if set(first) != set(range(8)):
        raise ValueError("validation requires all eight condition strata")
    result = []
    for group, index in sorted(first.items()):
        condition = np.asarray(split.conditions[index])
        full = np.asarray(split.latents[index])
        if condition.dtype != np.dtype("float32") or condition.shape != (2,) or not np.isfinite(condition).all():
            raise ValueError("actual validation condition must be finite FP32 [nu,temperature]")
        if condition[0] < 0 or condition[1] <= 0 or full.dtype != np.dtype("float32") or not np.isfinite(full).all():
            raise ValueError("finite complete native validation state and physical condition required")
        result.append({"row":index, "stratum_id":group, "root_id":str(split.case_ids[index]),
            "equation_seed":int(split.sample_ids[index]), "conditions":condition.tolist(),
            "condition_fp32_bits":condition.view(np.uint32).tolist(),
            "condition_array_sha256":array_sha(condition), "full_array_sha256":array_sha(full)})
    if len({r["root_id"] for r in result}) != 8 or len({r["equation_seed"] for r in result}) != 8:
        raise ValueError("selected validation roots must be independent, never duplicated as eight rows")
    return result


def fixed_noise_plan(rows):
    result = []
    for replica in range(2):
        for row in rows:
            root_seed = stream("vpfp-weak-dynamics-root", json_sha(
                [SCHEMA, MASTER_SEED, replica, row["root_id"], row["equation_seed"]]))
            result.append({"replica":replica, **row, "sampling_root_seed":root_seed,
                "macro_noise_seed":stream("vpfp-coherent-macro",root_seed),
                "micro_noise_seed":stream("vpfp-coherent-micro",root_seed)})
    seeds = [r[key] for r in result for key in ("sampling_root_seed","macro_noise_seed","micro_noise_seed")]
    if len(set(seeds)) != len(seeds):
        raise ValueError("registered diagnostic noise streams unexpectedly collide")
    return result


def check_frozen(plan):
    for path, digest in plan["frozen_files"].items():
        if Path(path).is_symlink():
            raise ValueError('frozen input or source changed: ' + path)


def freeze_inputs(output, design):
    from pde_agent_verifier_solver.plugins.vpfp.diffusion_data import read_manifest, load_split
    dirty = subprocess.check_output(["git","status","--porcelain","--untracked-files=all","--",
        "src","scripts","configs","tests","legacy/vpfp"],cwd=ROOT,text=True)
    if dirty.strip():
        raise ValueError("freeze all diagnostic source/tests before compute")
    revision = subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    manifest = read_manifest(SOURCE,verify_hashes=False)
    if manifest["manifest_sha256"] != SOURCE_SHA or manifest["regime_id"] != "long_time_weak_collision_v2":
        raise ValueError("only the pinned weak development corpus is registered")
    frozen = {str(SOURCE/"manifest.json"):file_sha(SOURCE/"manifest.json")}
    files = manifest["splits"]["validation"]["files"]
    for entry in [manifest["seed_plan"], *files.values()]:
        path = (SOURCE/entry["path"]).resolve(strict=True)
        if not path.is_relative_to(SOURCE.resolve()):
            raise ValueError("validation input escaped source corpus")
        checked_asset({"path":str(path),"sha256":entry["sha256"]},frozen=frozen)
    assets = []
    for item in MODEL_MANIFESTS:
        header = checked_asset(item,frozen=frozen,json_only=True)
        if (header.get("schema") != "vpfp_moving_frame_native_model_manifest_v1"
                or header.get("sampler_steps") != 64
                or header.get("regime_id") != manifest["regime_id"]
                or header["base_decoder"]["sha256"] != manifest["base_decoder_sha256"]):
            raise ValueError("pinned two-seed native manifest has wrong weak decoder/recipe")
        for role in ("base_decoder","macro_checkpoint","conditional_checkpoint"):
            checked_asset(header[role],frozen=frozen)
        assets.append({"manifest":dict(item),"assets":header})
    # Freeze only the actual diagnostic/model/physical implementation graphs.
    paths = [*sorted((ROOT/"src/pde_agent_verifier_solver/plugins/vpfp").glob("*.py")),
        *sorted((ROOT/"legacy/vpfp").rglob("*.py")),
        Path(__file__), ROOT/"scripts/run_vpfp_independent_confirmation.py",
        ROOT/"scripts/run_vpfp_independent_physics.py", ROOT/"scripts/diagnose_vpfp_known_initial.py"]
    for path in paths:
        frozen[str(path.resolve())] = file_sha(path)
    split = load_split(SOURCE,"validation",verify_hashes=False)
    selected = select_validation_rows(split)
    plan = {"schema":SCHEMA, "source_revision":revision, "source":str(SOURCE),
        "source_manifest_sha256":SOURCE_SHA, "validation_files":files,
        "validation_collection_origin":manifest["splits"]["validation"].get("collection_origin"),
        "source_split":"validation", "selection":"first_original_row_per_condition_stratum_not_feature_selected",
        "selected_roots":selected, "models":assets, "noise_plan":fixed_noise_plan(selected),
        "frozen_files":frozen, "design":design, "arms":list(ARMS), "macro_fit_grid":33,
        "sampling_steps":64, "master_seed":MASTER_SEED,
        "deep_selection":{"replicas":[0],"arms":[ARMS[0],ARMS[3]],"root_count":8,"carrier_count":16},
        "statistical_unit":"8_previously_opened_validation_roots_with_nested_interventions_not64_independent_answers",
        "condition_rule":"same_actual_validation_FP32_bits_for_all_four_arms_and_both_seed_models",
        "micro_noise_rule":"same_noise_seed_for_p(z|real_m,condition)_and_p(z|generated_m,condition)",
        "z_scope":"complete_native_value_contains_rho_and_all_kinetic_modes_no_truncation_or_projection",
        "all_inputs_and_noise_frozen_before_sampling":True,
        "scientific_admission":False,"confirmation_512_completed":False,"new_training":False,"pde_solves":0}
    check_frozen(plan)
    write_json(output/"frozen-plan.json",plan)
    return plan,split


def condition_for_row(split, row, device):
    import torch
    condition = np.array(split.conditions[row["row"]],copy=True)
    full = np.array(split.latents[row["row"]:row["row"]+1],copy=True)
    if (condition.dtype != np.dtype("float32")
            or condition.view(np.uint32).tolist() != row["condition_fp32_bits"]
            or array_sha(condition) != row["condition_array_sha256"]
            or array_sha(full[0]) != row["full_array_sha256"]):
        raise ValueError("actual validation condition/state changed after frozen selection")
    return torch.from_numpy(condition).to(device),torch.from_numpy(full).to(device)


def check_full_to_value(full, config):
    import torch
    from pde_agent_verifier_solver.plugins.vpfp.value_control_diffusion import extract_value_controls, lift_value_controls
    z = extract_value_controls(full,config,validate_slopes=True)
    if not torch.equal(lift_value_controls(z,config),full):
        raise ValueError("complete native value relift must preserve all original slope bits")
    return z


def sample_macro(sampler, condition, row):
    import torch
    with torch.no_grad():
        white = sampler.macro.sample(condition[None], [row["macro_noise_seed"]], steps=64)
        macro = sampler.macro_norm.decode(white)
    if not torch.isfinite(white).all() or not torch.isfinite(macro).all():
        raise FloatingPointError("nonfinite macro draw; no replacement noise")
    return macro.detach(),white.detach()


def sample_micro(sampler, condition, macro_white, row):
    import torch
    physical = condition[None].to(macro_white)
    if physical.detach().cpu().numpy().view(np.uint32).tolist() != [row["condition_fp32_bits"]]:
        raise ValueError("micro conditioning changed the validation physical FP32 bits")
    with torch.no_grad():
        white = sampler.micro.sample(torch.cat((physical,macro_white),dim=1),[row["micro_noise_seed"]],steps=64)
        full = sampler.micro_norm.decode(white)
        z = check_full_to_value(full,sampler.micro.config)
    if full.dtype != torch.float32 or not torch.isfinite(z).all():
        raise FloatingPointError("nonfinite or wrong-precision complete conditional draw")
    return z.detach()


def save_carrier(directory, decoder, carrier, condition, provenance):
    from pde_agent_verifier_solver.plugins.vpfp.physical_controls import carrier_hashes
    directory.mkdir()
    decoder.carrier_to_payload(carrier)  # Full versioned ABI validation, never flatten A as old z.
    files = {}
    for name,value in carrier.items():
        path = directory/(name+".npy")
        temporary = path.with_suffix(".npy.partial")
        with temporary.open("xb") as stream:
            np.save(stream,value.detach().cpu().numpy(),allow_pickle=False)
        temporary.replace(path)
        files[name] = {"path":str(path),"sha256":file_sha(path),"tensor_sha256":tensor_sha(value)}
    receipt = {**provenance, "carrier_sha256":carrier_hashes(decoder,carrier,condition[None])[0],
        "decoder_identity":decoder.identity(),"files":files,"condition_fp32_bits":condition.detach().cpu().numpy().view(np.uint32).tolist(),
        "no_Hermite_reprojection":True,"complete_floor_preserved":True,
        "paired_truth_claimed":False,"independent_population_sample":False}
    write_json(directory/"carrier-receipt.json",receipt)
    return receipt


class Progress:
    def __init__(self,output,wall_seconds):
        self.output,self.started = output,time.monotonic()
        self.deadline = self.started+wall_seconds
        self.required = []
        self.report = {"schema":SCHEMA,"execution_passed":False,"stage":"freeze",
            "units":{},"failures":[],"new_training":False,"pde_solves":0,"scientific_admission":False,
            "registered_carrier_count":64,"registered_clean_compression_count":16,"registered_deep_carrier_count":16,
            "confirmation_512_completed":False,"evidence_role":"previously_opened_validation_dynamics_diagnostic",
            "wall_seconds_limit":wall_seconds,"deadline_semantics":"checked_between_stages_and_measurement_chunks_outer_launcher_hard_timeout_required"}
        self.save()
    def save(self):
        self.report["elapsed_seconds"] = time.monotonic()-self.started
        if self.required:
            self.report["completed_carrier_count"] = sum(
                self.report["units"].get(name,{}).get("status")=="completed" for name in self.required)
        write_json(self.output/"result.json",self.report)
    def check(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("registered global diagnostic wall budget exhausted; no shortened protocol")
    def unit(self,name,function):
        self.check()
        self.report["stage"] = name
        self.report["units"][name] = {"status":"running"}
        self.save()
        started = time.monotonic()
        try:
            value = function()
            if isinstance(value,dict) and value.get("execution_passed") is False:
                raise ValueError("measurement returned execution_passed=false; original report retained")
            self.report["units"][name] = {"status":"completed","elapsed_seconds":time.monotonic()-started}
            return value
        except Exception as error:
            record = {"status":"failed","elapsed_seconds":time.monotonic()-started,
                "type":type(error).__name__,"reason":str(error),"traceback":traceback.format_exc()}
            self.report["units"][name] = record
            self.report["failures"].append({"unit":name,**record})
            if isinstance(error,TimeoutError):
                raise
            return None
        finally:
            self.save()
    def skip(self,name,reason):
        self.report["units"][name] = {"status":"not_run","reason":reason}
        self.save()


def run(output, *, device="cuda", wall_seconds=21600):
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_TMPDIR"):
        raise ValueError("weak dynamics extraction and sampling require a compute allocation")
    if not math.isfinite(wall_seconds) or not 0 < wall_seconds <= 21600:
        raise ValueError("wall_seconds must be positive and at most the registered6h cap")
    output = Path(output).absolute()
    if output.exists() or output.is_symlink() or not output.is_relative_to(SCRATCH_ROOT):
        raise ValueError("choose a fresh scratch output; never overwrite diagnostic inputs")
    if output.is_relative_to(SOURCE) or output.is_relative_to(MODEL_ROOT):
        raise ValueError("output must be outside the immutable corpus and model tree")
    output.mkdir(parents=True)
    progress = Progress(output,wall_seconds)
    plan = None
    try:
        import torch
        from pde_agent_verifier_solver.plugins.vpfp.moving_frame_binding import load_moving_frame_sampler
        from pde_agent_verifier_solver.plugins.vpfp.moving_frame import extract_macro_controls
        from pde_agent_verifier_solver.plugins.vpfp.weak_dynamics_diagnostic import (
            dynamics_design,measure_dynamics_carrier,measure_clean_compression)
        design = dynamics_design()
        frozen = progress.unit("freeze_inputs",lambda:freeze_inputs(output,design))
        if frozen is None:
            return progress.report
        plan,split = frozen
        progress.required = [f"seed-{replica}/root-{group}/{arm}"
            for replica in range(2) for group in range(8) for arm in ARMS]
        for name in progress.required:
            progress.report["units"][name] = {"status":"pending","reason":"registered before any model sampling"}
        progress.save()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        for replica,item in enumerate(plan["models"]):
            native = progress.unit(f"seed-{replica}/native-load",lambda item=item:
                load_moving_frame_sampler(item["manifest"]["path"],item["manifest"]["sha256"],device=device))
            if native is None:
                for row in (r for r in plan["noise_plan"] if r["replica"]==replica):
                    for arm in ARMS:
                        progress.skip(f"seed-{replica}/root-{row['stratum_id']}/{arm}","native loading failed")
                continue
            sampler,decoder = native
            if (sampler.identity["corpus_manifest_sha256"] != SOURCE_SHA
                    or decoder.macro_config.control_count != 33 or decoder.macro_config.max_mode != 6):
                raise ValueError("loaded model source/macro layout differs from frozen weak protocol")
            identity = {"sampler_identity":sampler.identity,
                "macro_normalizer_sha256":json_sha(sampler.macro_norm.to_payload()),
                "micro_normalizer_sha256":json_sha(sampler.micro_norm.to_payload())}
            seed_dir = output/f"seed-{replica}"; seed_dir.mkdir()
            write_json(seed_dir/"native-identity.json",identity)
            for row in (r for r in plan["noise_plan"] if r["replica"]==replica):
                prefix = f"seed-{replica}/root-{row['stratum_id']}"
                parent = seed_dir/f"root-{row['stratum_id']}"; parent.mkdir()
                inputs = progress.unit(prefix+"/validation-input",lambda row=row:condition_for_row(split,row,device))
                if inputs is None:
                    for arm in ARMS: progress.skip(prefix+"/"+arm,"frozen validation input unavailable")
                    continue
                condition,full = inputs
                real_z = progress.unit(prefix+"/real-z",lambda:check_full_to_value(full,sampler.micro.config))
                real_m = None
                if real_z is not None:
                    def extract():
                        macro,receipt = extract_macro_controls(decoder.base_decoder,real_z,condition,
                            decoder.macro_config,value_config=sampler.micro.config,fit_grid=33)
                        # Original training cached the physical extraction in FP32.
                        macro = macro.detach().to(device=device,dtype=torch.float32)
                        if not torch.isfinite(macro).all(): raise FloatingPointError("nonfinite clean macro controls")
                        receipt.update(stored_dtype="float32",post_projection_tensor_sha256=tensor_sha(macro),
                            old_full_reference_sha256=row["full_array_sha256"])
                        write_json(parent/"real-m-extraction.json",receipt)
                        return macro
                    real_m = progress.unit(prefix+"/real-m",extract)
                generated = progress.unit(prefix+"/macro-draw",lambda:sample_macro(sampler,condition,row))
                generated_m,generated_white = generated if generated is not None else (None,None)
                real_micro,generated_micro,real_white = None,None,None
                if real_m is not None:
                    real_white = progress.unit(prefix+"/normalize-real-m",lambda:sampler.macro_norm.encode(real_m))
                    if real_white is not None:
                        real_micro = progress.unit(prefix+"/micro-given-real-m",lambda:
                            sample_micro(sampler,condition,real_white,row))
                else: progress.skip(prefix+"/micro-given-real-m","real macro extraction failed")
                if generated_white is not None:
                    generated_micro = progress.unit(prefix+"/micro-given-generated-m",lambda:
                        sample_micro(sampler,condition,generated_white,row))
                else: progress.skip(prefix+"/micro-given-generated-m","macro generation failed")
                components = ((real_m,real_z),(real_m,real_micro),(generated_m,real_z),(generated_m,generated_micro))
                for arm,(m,z) in zip(ARMS,components):
                    name = prefix+"/"+arm
                    if m is None or z is None:
                        progress.skip(name,"required upstream macro/micro state failed; no replacement")
                        continue
                    carrier = {"m":m,"z":z}
                    provenance = {"source_manifest_sha256":SOURCE_SHA,"split":"validation",
                        "data_origin":"physical_reference" if arm==ARMS[0] else "generated_answer",
                        "root_id":row["root_id"],"equation_seed":row["equation_seed"],"stratum_id":row["stratum_id"],
                        "model_replica":replica,"arm":arm,"view":arm,"row":row["row"],"noise_plan":row,
                        "native_manifest":item["manifest"],"m_origin":"real" if arm.startswith("real_m") else "generated",
                        "z_origin":"real_full_state" if arm in (ARMS[0],ARMS[2]) else "conditional_generated_full_state",
                        "z_contains_generated_rho_as_well_as_kinetic":arm in (ARMS[1],ARMS[3]),
                        "micro_conditioning_macro":"real" if arm==ARMS[1] else "generated" if arm==ARMS[3] else None,
                        "micro_conditioning_white_sha256":tensor_sha(real_white) if arm==ARMS[1]
                            else tensor_sha(generated_white) if arm==ARMS[3] else None}
                    receipt = progress.unit(name+"/save-carrier",lambda:
                        save_carrier(parent/arm,decoder,carrier,condition,provenance))
                    if receipt is None:
                        progress.skip(name,"carrier save/ABI validation failed"); continue
                    if arm==ARMS[0]:
                        progress.unit(name+"/clean-compression",lambda:
                            measure_clean_compression(decoder,carrier,condition,output=parent/arm/"clean-compression.json",
                                root_id=row["root_id"],provenance=receipt,design=design,
                                check_deadline=progress.check,reference_full=full))
                    measured = progress.unit(name,lambda:measure_dynamics_carrier(decoder,carrier,condition,
                        output=parent/arm/"dynamics.json",root_id=row["root_id"],provenance=receipt,
                        design=design,check_deadline=progress.check,deep=replica==0 and arm in (ARMS[0],ARMS[3])))
                    if measured is not None and "first_completed_carrier_benchmark" not in progress.report:
                        progress.report["first_completed_carrier_benchmark"] = {
                            "unit":name,"elapsed_seconds":progress.report["units"][name]["elapsed_seconds"],
                            "deep_included":replica==0 and arm in (ARMS[0],ARMS[3]),
                            "scope":"this_measurement_only_excludes_loading_sampling_and_clean_compression",
                            "projection_to_remaining_runtime_is_not_guaranteed":True}
                        progress.save()
            del sampler,decoder,native
        progress.unit("recheck-frozen-inputs",lambda:check_frozen(plan) or {"execution_passed":True})
        required = [f"seed-{replica}/root-{group}/{arm}" for replica in range(2) for group in range(8) for arm in ARMS]
        progress.report.update(stage="complete",registered_carrier_count=64,
            completed_carrier_count=sum(progress.report["units"].get(name,{}).get("status")=="completed" for name in required),
            execution_passed=not progress.report["failures"] and all(
                progress.report["units"].get(name,{}).get("status")=="completed" for name in required))
    except Exception as error:
        progress.report.update(stage="stopped",failure={"type":type(error).__name__,"reason":str(error),
            "traceback":traceback.format_exc()},execution_passed=False)
    finally:
        if plan is not None:
            required = [f"seed-{replica}/root-{group}/{arm}" for replica in range(2) for group in range(8) for arm in ARMS]
            for name in required:
                progress.report["units"].setdefault(name,{"status":"not_run",
                    "reason":"diagnostic stopped before this registered carrier; no replacement"})
            progress.report["completed_carrier_count"] = sum(
                progress.report["units"][name]["status"]=="completed" for name in required)
        progress.save()
    return progress.report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--wall-seconds",type=float,default=21600)
    args = parser.parse_args(argv)
    result = run(args.output,device=args.device,wall_seconds=args.wall_seconds)
    print(json.dumps({"execution_passed":result["execution_passed"],"output":str(args.output)}),flush=True)
    return 0 if result["execution_passed"] else 1


if __name__=="__main__":
    raise SystemExit(main())
