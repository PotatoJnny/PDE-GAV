"""Explicit frozen candidates -> fresh coarse references and512 new attempts.

All512 attempts get descriptive coverage; fixed8-root physics, carrier controls
and one-root reference refinement are reported separately, not qualification.
No score-based selection, training, hidden count reduction or automatic jobs.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import importlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT/"scripts"))
SCHEMA = "vpfp_independent_confirmation_handoff_v1"
SELECTION_SCHEMA = "vpfp_independent_confirmation_selection_v1"
ROLES = ("calibration","reference_a","reference_b")
NATIVE_KINDS = ("strong_native","weak_native")
COUNT = 512
SCRATCH_ROOT = Path("/scratch")
SEED_LIST_KEYS = {"seeds","noise_seeds","sampling_seeds","equation_seeds","known_prior_seeds",
    "known_initial_seeds","prior_seeds","future_noise_seeds","macro_noise_seeds","micro_noise_seeds","condition_seeds"}
SEED_KEYS = {"seed","master_seed","condition_seed","equation_seed","known_initial_seed",
    "prior_seed","future_noise_seed","macro_noise_seed","micro_noise_seed"}


def json_sha(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def file_sha(path):
    value = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""): value.update(block)
    return value.hexdigest()


def write_json(path,value):
    path = Path(path)
    temporary = path.with_name(path.name+".partial")
    temporary.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False))
    temporary.replace(path)


def checked_asset(item, *, frozen=None, json_only=False):
    if (not isinstance(item,dict) or set(item)!={"path","sha256"}
            or not isinstance(item["sha256"],str) or not re.fullmatch("[0-9a-f]{64}",item["sha256"])):
        raise ValueError("asset requires exact path and lowercase SHA256")
    path = Path(item["path"])
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError('explicit asset missing, symlinked, nonabsolute or SHA mismatch: ' + str(path))
    if frozen is not None: frozen[str(path)] = item["sha256"]
    return json.loads(path.read_text()) if json_only else path


def seed_ids(value):
    """Read only explicit seed fields, never infer ancestry from physical scores."""
    found = set()
    def add(seed):
        if type(seed) is not int or not 0<=seed<2**63: raise ValueError("invalid declared development seed")
        found.add(seed)
    def visit(item):
        if isinstance(item,dict):
            for key,child in item.items():
                if key in SEED_KEYS: add(child)
                elif key in SEED_LIST_KEYS:
                    if not isinstance(child,list): raise ValueError("development seed field must be a list")
                    for seed in child: add(seed)
                else: visit(child)
        elif isinstance(item,list):
            for child in item: visit(child)
    visit(value)
    return found


def validate_candidate_kind(candidate):
    regime,kind = candidate.get("regime"),candidate.get("kind")
    allowed = {"strong":("strong_native",),"weak":("weak_native","weak_hierarchy")}
    if regime not in ("strong","weak") or kind not in allowed[regime]:
        raise ValueError("candidate kind does not match its explicit collision regime")


def load_selection(path, expected):
    frozen = {}
    selection = checked_asset({"path":str(Path(path).absolute()),"sha256":expected},frozen=frozen,json_only=True)
    if (set(selection)!={"schema","review","scientific_admission","candidates"}
            or selection["schema"]!=SELECTION_SCHEMA or not isinstance(selection["review"],str)
            or not selection["review"].strip() or selection["scientific_admission"] is not False
            or not isinstance(selection["candidates"],list) or not selection["candidates"]):
        raise ValueError("an explicit reviewed candidate selection is required")
    names,assets,forbidden = set(),set(),set()
    for candidate in selection["candidates"]:
        if (set(candidate)!={"id","regime","kind","asset","model_source","development_provenance"}
                or not isinstance(candidate["id"],str) or not re.fullmatch("[a-z][a-z0-9_-]{0,63}",candidate["id"])
                or candidate["id"] in names or candidate["regime"] not in ("strong","weak")):
            raise ValueError("invalid/duplicate explicit candidate identity")
        validate_candidate_kind(candidate)
        names.add(candidate["id"])
        path = checked_asset(candidate["asset"],frozen=frozen)
        if str(path) in assets: raise ValueError("duplicate candidate asset")
        assets.add(str(path))
        source = candidate["model_source"]
        if (not isinstance(source,dict) or set(source)!={"corpus","manifest_sha256"}
                or not Path(source["corpus"]).is_absolute() or Path(source["corpus"]).is_symlink()
                or not re.fullmatch("[0-9a-f]{64}",source["manifest_sha256"])):
            raise ValueError("model_source must explicitly bind its training corpus identity")
        if not isinstance(candidate["development_provenance"],list) or not candidate["development_provenance"]:
            raise ValueError("every candidate needs explicit completed development generation receipts")
        for item in candidate["development_provenance"]:
            document = checked_asset(item,frozen=frozen,json_only=True)
            ids = seed_ids(document)
            if not ids: raise ValueError("development receipt has no auditable seed ancestry")
            forbidden.update(ids)
        if candidate["kind"]=="weak_hierarchy":
            manifest = checked_asset(candidate["asset"],json_only=True)
            if manifest.get("schema")!="vpfp_moving_frame_native_model_manifest_v1" or manifest.get("sampler_steps")!=64:
                raise ValueError("explicit weak hierarchical native manifest required")
            for role in ("base_decoder","macro_checkpoint","conditional_checkpoint"):
                checked_asset(manifest[role],frozen=frozen)
    return selection,frozen,forbidden


def stream(tag,seed):
    return int.from_bytes(sha256(f"{tag}:{seed}".encode()).digest()[:8],"little") % 2**63


def generation_plan(candidate,manifest,selection_sha,master_seed,forbidden):
    """512 fixed8-stratum attempts; no generated/physical features consulted."""
    edges = [np.linspace(*manifest[key],bins+1) for key,bins in (("collision_nu_range",4),("temperature_range",2))]
    used = set(forbidden)
    rows = []
    for index in range(COUNT):
        group = index//(COUNT//8)
        for nonce in range(10000):
            namespace = json_sha([SCHEMA,selection_sha,master_seed,candidate["id"],index,nonce])
            root = stream("vpfp-independent-root",namespace)
            condition_seed = stream("vpfp-independent-condition",namespace)
            streams = {"seed":root,"condition_seed":condition_seed,
                "known_initial_seed":stream("vpfp-initial-conditioned-prior",root),
                "future_noise_seed":stream("vpfp-initial-conditioned-future",root),
                "macro_noise_seed":stream("vpfp-coherent-macro",root),
                "micro_noise_seed":stream("vpfp-coherent-micro",root)}
            ids = set(streams.values())
            if len(ids)!=len(streams) or ids.intersection(used): continue
            rng = np.random.default_rng(condition_seed)
            nu_bin,temp_bin = divmod(group,2)
            condition = np.asarray([rng.uniform(*edges[0][nu_bin:nu_bin+2]),
                rng.uniform(*edges[1][temp_bin:temp_bin+2])],dtype=np.float32)
            actual = [min(np.searchsorted(axis,float(x),side="right")-1,len(axis)-2) for x,axis in zip(condition,edges)]
            if actual!=[nu_bin,temp_bin]: continue
            rows.append({"attempt":index,"stratum_id":group,"conditions":condition.tolist(),**streams})
            used.update(ids)
            break
        else: raise RuntimeError("could not construct a disjoint prescribed generation stream")
    return rows,used


def prepare_plan(selection,selection_sha,output,seed,frozen,forbidden):
    for candidate in selection["candidates"]:
        validate_candidate_kind(candidate)
    import torch
    from pde_agent_verifier_solver.plugins.vpfp import diffusion_data as data
    from pde_agent_verifier_solver.plugins.vpfp.perturbation_backend import _legacy_api
    _legacy_api()
    physical = importlib.import_module("agent_verifier.vpfp.high_fidelity")
    sources,manifests = {},{}
    for candidate in selection["candidates"]:
        regime,source = candidate["regime"],candidate["model_source"]
        if regime in sources and sources[regime]!=source: raise ValueError("one explicit training source per regime required")
        sources[regime] = source
    for regime,source in sources.items():
        corpus = Path(source["corpus"])
        manifest = data.read_manifest(corpus,verify_hashes=False)
        if manifest["manifest_sha256"]!=source["manifest_sha256"]: raise ValueError("model source canonical manifest SHA changed")
        if regime not in manifest["regime_id"].lower(): raise ValueError("candidate/source collision regime differs")
        manifests[regime] = manifest
        frozen[str(corpus/"manifest.json")] = file_sha(corpus/"manifest.json")
        source_plan = checked_asset({"path":str(corpus/manifest["seed_plan"]["path"]),"sha256":manifest["seed_plan"]["sha256"]},frozen=frozen,json_only=True)
        forbidden.update(data._source_exclusion_seeds(source_plan))
        for label in ("base_decoder","expanded_decoder"):
            checked_asset({"path":manifest["model_spec"][label+"_checkpoint"],"sha256":manifest["model_spec"][label+"_sha256"]},frozen=frozen)
    # Fail on an incorrectly selected model BEFORE spending time collecting
    # new references. Full native loader validation remains in generation.
    fields = {"schema","config","model","normalizer","base_decoder_sha256","corpus_manifest_sha256","training"}
    for candidate in selection["candidates"]:
        source = manifests[candidate["regime"]]
        if candidate["kind"] in NATIVE_KINDS:
            payloads = [torch.load(candidate["asset"]["path"],map_location="cpu",weights_only=True)]
        else:
            assets = checked_asset(candidate["asset"],json_only=True)
            payloads = [torch.load(assets[role+"_checkpoint"]["path"],map_location="cpu",weights_only=True) for role in ("macro","conditional")]
            from pde_agent_verifier_solver.plugins.vpfp.moving_frame_binding import validate_hierarchical_payloads
            validate_hierarchical_payloads(*payloads,base_decoder_sha256=source["base_decoder_sha256"],
                macro_checkpoint_sha256=assets["macro_checkpoint"]["sha256"])
        for payload in payloads:
            if (not isinstance(payload,dict) or set(payload)!=fields
                    or payload["corpus_manifest_sha256"]!=source["manifest_sha256"]
                    or payload["base_decoder_sha256"]!=source["base_decoder_sha256"]):
                raise ValueError("candidate is not a native seven-field asset bound to the declared training source")
        del payloads,payload
    references = {}
    for regime,source in sources.items():
        manifest = manifests[regime]
        spec = data.VPFPOnlineModelSpec.from_mapping(manifest["model_spec"])
        source_plan = json.loads((Path(source["corpus"])/manifest["seed_plan"]["path"]).read_text())
        cfg = physical.VPFPHighFidelityConfig.from_dict(manifest["provenance"]["equation_config"])
        count = {**manifest["split_counts"],**{role:COUNT for role in ROLES}}
        master = stream("vpfp-independent-reference-"+regime,json_sha([selection_sha,seed]))
        planned = data._plan_seeds(spec,count,master,(4,2),
            lambda s:physical.sample_continuous_equation_spec(seed=s,config=cfg,defer_rho_log_partition=True),
            split_names=ROLES,excluded_equation_seeds=data._source_exclusion_seeds(source_plan))
        roots = {row["equation_seed"] for role in ROLES for row in planned["splits"][role]}
        if roots.intersection(forbidden): raise ValueError("preplanned fresh reference overlaps development or another regime; choose another explicit master seed")
        forbidden.update(roots)
        references[regime] = {"model_source":source,"reference_source":str(output/(regime+"-reference")/"corpus"),
            "master_seed":master,"expected_splits":planned["splits"],"count_per_role":COUNT}
    candidates = []
    for candidate in selection["candidates"]:
        rows,forbidden = generation_plan(candidate,manifests[candidate["regime"]],selection_sha,seed,forbidden)
        candidates.append({**candidate,"rows":rows,"generated_dir":str(output/"generated"/candidate["id"])})
    # Every module and invoked script is frozen before collection or generation.
    from run_vpfp_independent_physics import physics_design
    for path in [*sorted((ROOT/"src/pde_agent_verifier_solver/plugins/vpfp").glob("*.py")),
            Path(__file__),ROOT/"scripts/run_vpfp_regime_fidelity_coverage.py",
            ROOT/"scripts/run_vpfp_independent_physics.py",ROOT/"scripts/diagnose_vpfp_known_initial.py"]:
        frozen[str(path.resolve())] = file_sha(path)
    return {"schema":SCHEMA,"selection_sha256":selection_sha,"master_seed":seed,
        "references":references,"candidates":candidates,"frozen_files":frozen,
        "reference_protocol":"original_coarse_solve_then_projection_NOT_refined_confirmation",
        "sample_count":COUNT,"sampler_steps":64,"strata":[4,2],"all_seeds_fixed_before_PDE_or_generation":True,
        "scientific_admission":False,"physics_design":physics_design(references,candidates,seed)}


def assert_frozen(plan):
    for path,digest in plan["frozen_files"].items():
        if Path(path).is_symlink():
            raise ValueError('frozen input/source changed: ' + path)


def collect_stage(plan,regime,device):
    from pde_agent_verifier_solver.plugins.vpfp import diffusion_data as data
    item = plan["references"][regime]
    result = data.collect_fresh_evaluation_corpus(item["model_source"]["corpus"],item["reference_source"],
        item["master_seed"],batch_size=8,device=device,evaluation_count=COUNT)
    actual = json.loads((Path(item["reference_source"])/result["seed_plan"]["path"]).read_text())
    if any(actual["splits"][role]!=item["expected_splits"][role] for role in ROLES):
        raise ValueError("actual fresh reference seeds differ from the pre-solve fixed plan")
    if result["manifest_sha256"]==item["model_source"]["manifest_sha256"]:
        raise ValueError("independent reference identity cannot equal training identity")
    return {"reference_manifest_sha256":result["manifest_sha256"],"count_per_role":COUNT,
        "reference_protocol":plan["reference_protocol"],"refined_reference":False}


def generate_stage(plan,identifier,device):
    import torch
    from pde_agent_verifier_solver.plugins.vpfp.structured_diffusion import load_vpfp_diffusion_sampler
    from pde_agent_verifier_solver.plugins.vpfp.initial_conditioned_diffusion import VPFPInitialConditionedBatchSampler
    from pde_agent_verifier_solver.plugins.vpfp.moving_frame_binding import load_moving_frame_sampler
    item = next(row for row in plan["candidates"] if row["id"]==identifier)
    validate_candidate_kind(item)
    directory = Path(item["generated_dir"])
    directory.mkdir(parents=True,exist_ok=False)
    paired = item["kind"]=="weak_hierarchy"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if paired:
        sampler,_ = load_moving_frame_sampler(item["asset"]["path"],item["asset"]["sha256"],device=device)
        model_source = sampler.identity["corpus_manifest_sha256"]
        dimension = sampler.micro.config.full_dimension
    else:
        payload = torch.load(item["asset"]["path"],map_location="cpu",weights_only=True)
        if set(payload)!={"schema","config","model","normalizer","base_decoder_sha256","corpus_manifest_sha256","training"}:
            raise ValueError("native selection must name safe7-field checkpoints, not continuation states")
        model_source = payload["corpus_manifest_sha256"]
        del payload
        sampler = load_vpfp_diffusion_sampler(item["asset"]["path"],item["asset"]["sha256"],device=device,steps=64)
        dimension = sampler.identity["latent_dimension"]
    if model_source!=item["model_source"]["manifest_sha256"]: raise ValueError("model checkpoint belongs to another training corpus")
    rows = item["rows"]
    if len(rows)!=COUNT: raise ValueError("generation plan must retain all512 attempted rows")
    conditions = np.asarray([row["conditions"] for row in rows],dtype=np.float32)
    np.save(directory/"generated-conditions.npy",conditions,allow_pickle=False)
    raw = np.lib.format.open_memmap(directory/"generated-coefficients.npy",mode="w+",dtype="float32",shape=(COUNT,dimension))
    raw[:] = np.nan; raw.flush()
    macro = None
    if paired:
        macro = np.lib.format.open_memmap(directory/"generated-macro.npy",mode="w+",dtype="float32",shape=(COUNT,sampler.macro.config.latent_dimension))
        macro[:] = np.nan; macro.flush()
    failures = []
    for index,row in enumerate(rows):
        error = None
        try:
            if paired:
                from pde_agent_verifier_solver.plugins.vpfp.value_control_diffusion import lift_value_controls
                draw = sampler.sample_conditions(conditions[index:index+1],[row["seed"]])
                value = lift_value_controls(draw["z"],sampler.micro.config)
                macro[index] = draw["m"][0].numpy()
            elif isinstance(sampler,VPFPInitialConditionedBatchSampler):
                value = sampler.sample_conditions(conditions[index:index+1],[row["seed"]])
            else:
                value = sampler.normalizer.decode(sampler.model.sample(conditions[index:index+1],[row["seed"]],steps=64))
            raw[index] = value[0].detach().cpu().numpy()
            if not np.isfinite(raw[index]).all() or (macro is not None and not np.isfinite(macro[index]).all()):
                raise FloatingPointError("nonfinite native generation; row retained, not resampled")
        except (ValueError,RuntimeError,FloatingPointError) as exc:
            raw[index] = np.nan
            if macro is not None: macro[index] = np.nan
            error = {"attempt":index,"seed":row["seed"],"type":type(exc).__name__,"reason":str(exc)}
            failures.append(error)
        raw.flush()
        if macro is not None: macro.flush()
        with (directory/"generation-progress.jsonl").open("a") as log:
            log.write(json.dumps({"attempt":index,"completed_attempts":index+1,"failure":error},allow_nan=False)+"\n")
        write_json(directory/"generation-partial.json",{"attempted":COUNT,"completed_attempts":index+1,"failures":failures,
            "complete":False,"scientific_admission":False})
    conditional = isinstance(sampler,VPFPInitialConditionedBatchSampler)
    receipt = {"schema":"vpfp_coherent_generated_native_pair_v1" if paired else "vpfp_independent_native_generation_v1",
        "attempted":COUNT,"completed_attempts":COUNT,"failed_attempts":len(failures),"successful_attempts":COUNT-len(failures),
        "failures":failures,"seeds":[r["seed"] for r in rows],
        "condition_seeds":[r["condition_seed"] for r in rows],
        "future_noise_seeds":[r["future_noise_seed"] if conditional else r["micro_noise_seed"] if paired else r["seed"] for r in rows],
        "conditions_sha256":file_sha(directory/"generated-conditions.npy"),"coefficients_sha256":file_sha(directory/"generated-coefficients.npy"),
        "model_source_manifest_sha256":model_source,"selection_sha256":plan["selection_sha256"],
        "reference_protocol":plan["reference_protocol"],"sampler_steps":64,"strata":[4,2],
        "generation_seed_rule":"selection_SHA/master_seed/candidate/attempt -> separate condition and root; native macro/micro or prior/future SHA streams",
        "failed_rows_are_retained_and_not_replaced":True,"scientific_admission":False,
        "evidence_role":"independent_attempts_against_new_original_coarse_references_not_refined_confirmation"}
    if conditional: receipt["known_initial_seeds"] = [row["known_initial_seed"] for row in rows]
    if paired:
        assets = sampler.identity["assets"]
        receipt.update(checkpoint_bindings={role:{"checkpoint":assets[role+"_checkpoint"]["path"],
            "checkpoint_sha256":assets[role+"_checkpoint"]["sha256"]} for role in ("macro","conditional")},
            macro_sha256=file_sha(directory/"generated-macro.npy"),macro_config=sampler.macro.config.to_dict(),
            micro_config=sampler.micro.config.to_dict(),macro_noise_seeds=[r["macro_noise_seed"] for r in rows],
            micro_noise_seeds=[r["micro_noise_seed"] for r in rows],
            decoder_views=["B_old_full_decoder","A_continuous_moving_frame"],full_coefficients_are_A_answer=False)
    else:
        receipt.update(checkpoint=dict(item["asset"]),checkpoint_sha256=item["asset"]["sha256"])
    write_json(directory/"generated-provenance.json",receipt)
    write_json(directory/"generation-partial.json",{"attempted":COUNT,"completed_attempts":COUNT,
        "failed_attempts":len(failures),"failures":failures,"complete":True,"scientific_admission":False})
    return {"completed_attempts":COUNT,"failed_attempts":len(failures),"generated_dir":str(directory),
        "provenance_sha256":file_sha(directory/"generated-provenance.json"),"physical_qualification":False}


def run_child(command,log,deadline):
    """One owned process group; timeout preserves files and kills no scheduler job."""
    remaining = deadline-time.monotonic()
    if remaining<=0: raise TimeoutError("global allocation budget exhausted before next stage")
    started = time.monotonic()
    timed_out = False
    with Path(log).open("xb") as stream_log:
        process = subprocess.Popen(command,stdout=stream_log,stderr=subprocess.STDOUT,cwd=ROOT,start_new_session=True)
        try: code = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            timed_out = True
            try: os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError: pass
            try: code = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try: os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError: pass
                code = process.wait()
    return {"command":list(map(str,command)),"log":str(log),"exit_code":int(code),"timed_out":timed_out,
        "actual_wall_seconds":time.monotonic()-started,"log_sha256":file_sha(log)}


def run(arguments):
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_TMPDIR"):
        raise RuntimeError("independent confirmation requires one existing Slurm compute allocation")
    if (type(arguments.seed) is not int or arguments.seed<0 or not math.isfinite(arguments.wall_seconds)
            or not 30<arguments.wall_seconds<=43200): raise ValueError("positive seed and wall_seconds in(30,43200] required")
    output = Path(arguments.output).absolute()
    if output.exists() or output.is_symlink() or output==SCRATCH_ROOT or not output.is_relative_to(SCRATCH_ROOT):
        raise ValueError("fresh /scratch output required; no implicit continuation")
    started = time.monotonic()
    # Reserve cleanup/publication time; total including source freeze uses this deadline.
    deadline = started+arguments.wall_seconds-15
    selection,frozen,forbidden = load_selection(arguments.selection,arguments.selection_sha256)
    for candidate in selection["candidates"]:
        source = Path(candidate["model_source"]["corpus"])
        if source==output or source in output.parents or output in source.parents: raise ValueError("output overlaps model source")
    output.mkdir(parents=True)
    result = {"schema":SCHEMA,"execution_passed":False,"data_handoff_complete":False,"confirmation_512_completed":False,
        "physical_regime_qualified":False,"scientific_admission":False,"new_training":False,
        "reference_protocol":"original_coarse_not_refined","sample_count":COUNT,"stages":[],"stage":"freeze_plan",
        "wall_budget_seconds":arguments.wall_seconds,"independent_physics_status":"registered_subset_pending",
        "descriptive_coverage_512_completed":False,"physical_subset_execution_completed":False,
        "reference_refinement_subset_completed":False,"physics_subsets":{}}
    try:
        plan = prepare_plan(selection,arguments.selection_sha256,output,arguments.seed,frozen,forbidden)
        path = output/"frozen-plan.json"
        write_json(path,plan)
        digest = file_sha(path)
        result["frozen_plan_sha256"] = digest
        def stage(name,command):
            result["stage"] = name; write_json(output/"result.json",result)
            receipt = run_child(command,output/(name+".log"),deadline)
            result["stages"].append({"stage":name,**receipt}); write_json(output/"result.json",result)
            if receipt["exit_code"]!=0 or receipt["timed_out"]: raise RuntimeError("stage failed or deadline reached: "+name)
        for regime in plan["references"]:
            stage("collect-"+regime,[sys.executable,__file__,"--internal-stage","collect","--plan",str(path),"--plan-sha256",digest,"--target",regime,"--device",arguments.device])
        for candidate in plan["candidates"]:
            stage("generate-"+candidate["id"],[sys.executable,__file__,"--internal-stage","generate","--plan",str(path),"--plan-sha256",digest,"--target",candidate["id"],"--device",arguments.device])
        for regime,item in plan["references"].items():
            command = [sys.executable,str(ROOT/"scripts/run_vpfp_regime_fidelity_coverage.py"),
                "--model-source",item["model_source"]["corpus"],"--reference-source",item["reference_source"],
                "--output",str(output/(regime+"-audit")),"--sample-count",str(COUNT),"--bootstrap-repeats","49","--device",arguments.device]
            for candidate in plan["candidates"]:
                if candidate["regime"]==regime: command.extend(("--generated-dir",candidate["generated_dir"]))
            stage("audit-"+regime,command)
        result.update(data_handoff_complete=True,descriptive_coverage_512_completed=True)
        write_json(output/"result.json",result)
        for regime in plan["references"]:
            remaining = deadline-time.monotonic()-5
            if remaining<=0: raise TimeoutError("no budget remains for the registered physics/control stage")
            result["physics_subsets"][regime] = {"path":str(output/("physics-"+regime)/"result.json"),
                "execution_passed":False,"status":"registered_stage_started"}
            stage("physics-controls-"+regime,[sys.executable,__file__,"--internal-stage","physics_controls",
                "--plan",str(path),"--plan-sha256",digest,"--target",regime,"--device",arguments.device,
                "--wall-seconds",str(remaining)])
            report_path = output/("physics-"+regime)/"result.json"
            physical = json.loads(report_path.read_text())
            result["physics_subsets"][regime] = {"path":str(report_path),"sha256":file_sha(report_path),
                "execution_passed":physical["execution_passed"],"scope":plan["physics_design"]["physics_population_scope"]}
            if physical["execution_passed"] is not True: raise RuntimeError("physical/control subset incomplete: "+regime)
        assert_frozen(plan)
        if time.monotonic()>=deadline: raise TimeoutError("total deadline reached before final publication")
        result.update(execution_passed=True,physical_subset_execution_completed=True,reference_refinement_subset_completed=True,
            independent_physics_status="registered_subset_executed_numerical_differences_and_budget_curves_saved",
            stage="all_registered_stages_executed_not_automatic_scientific_admission")
    except Exception as error:
        result["failure"] = {"type":type(error).__name__,"reason":str(error),"traceback":traceback.format_exc()}
    finally:
        result["actual_wall_seconds"] = time.monotonic()-started
        write_json(output/"result.json",result)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selection",type=Path)
    p.add_argument("--selection-sha256")
    p.add_argument("--output",type=Path)
    p.add_argument("--seed",type=int)
    p.add_argument("--wall-seconds",type=float,default=43200.)
    p.add_argument("--device",default="cuda")
    p.add_argument("--internal-stage",choices=("collect","generate","physics_controls"),help=argparse.SUPPRESS)
    p.add_argument("--plan",type=Path,help=argparse.SUPPRESS)
    p.add_argument("--plan-sha256",help=argparse.SUPPRESS)
    p.add_argument("--target",help=argparse.SUPPRESS)
    return p


def main(argv=None):
    p = parser(); args = p.parse_args(argv)
    if args.internal_stage:
        if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_TMPDIR"): p.error("private stages require Slurm")
        plan = checked_asset({"path":str(args.plan),"sha256":args.plan_sha256},json_only=True)
        if plan.get("schema")!=SCHEMA or plan.get("sample_count")!=COUNT or plan.get("sampler_steps")!=64: p.error("invalid fixed plan")
        assert_frozen(plan)
        if args.internal_stage=="physics_controls":
            from run_vpfp_independent_physics import run as run_physics
            result = run_physics(plan,plan_sha256=args.plan_sha256,regime=args.target,
                output=args.plan.parent/("physics-"+args.target),device=args.device,wall_seconds=args.wall_seconds)
        else:
            result = (collect_stage if args.internal_stage=="collect" else generate_stage)(plan,args.target,args.device)
        assert_frozen(plan)
        write_json(args.plan.parent/(args.internal_stage+"-"+args.target+"-receipt.json"),result)
        return 0 if result.get("execution_passed",True) is True else 1
    if any(getattr(args,key) is None for key in ("selection","selection_sha256","output","seed")):
        p.error("--selection --selection-sha256 --output --seed are required")
    return 0 if run(args)["execution_passed"] else 1


if __name__=="__main__": raise SystemExit(main())
