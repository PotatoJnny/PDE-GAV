"""Bounded W0 / hierarchical B,A experiment; requires reviewed real preflight.

Only clean training macro coordinates condition micro training. Both generated
decoder views reuse one joint sample. This driver cannot admit/deploy models.
"""
from __future__ import annotations

import argparse
from copy import copy
from dataclasses import replace
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
loader = importlib.util.spec_from_file_location("vpfp_coherent_shared", ROOT / "scripts/train_vpfp_structured_diffusion.py")
shared = importlib.util.module_from_spec(loader)
loader.loader.exec_module(shared)

from pde_agent_verifier_solver.plugins.vpfp.canonical_binding import VPFPOnlineModelSpec, build_vpfp_online_binding
from pde_agent_verifier_solver.plugins.vpfp.diffusion_data import load_split, read_manifest
from pde_agent_verifier_solver.plugins.vpfp.structured_diffusion import load_vpfp_diffusion_sampler
from pde_agent_verifier_solver.plugins.vpfp.value_control_diffusion import VPFPValueEDMConfig, extract_value_controls
from pde_agent_verifier_solver.plugins.vpfp.temporal_covariance_diffusion import VPFPTemporalCovarianceEDM, VPFPTemporalCovarianceNormalizer
from pde_agent_verifier_solver.plugins.vpfp.coherent_diffusion import (
    VPFPCoherentMacroConfig, VPFPCoherentMacroEDM, VPFPCoherentMacroNormalizer,
    VPFPMacroConditionalEDM, sample_hierarchical,
)


def json_sha(data):
    return sha256(json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def require_selection(path, corpus):
    """A numerical gate plus an explicit interpretation of compression curves."""
    choice = json.loads(Path(path).read_text())
    if (choice.get("schema") != "vpfp_moving_frame_representation_selection_v1"
            or choice.get("mechanics_passed") is not True
            or choice.get("selected_mode") not in (4,6)
            or choice.get("selected_fit_grid") not in (33,65)
            or not choice.get("compression_review")
            or choice.get("corpus_manifest_sha256") != read_manifest(corpus,verify_hashes=False)["manifest_sha256"]):
        raise ValueError("real-data representation preflight and explicit compression review required")
    evidence = Path(choice["preflight_result"])
    report = json.loads(evidence.read_text())
    if report.get("source_manifest_sha256") != choice["corpus_manifest_sha256"]:
        raise ValueError("representation preflight belongs to another source corpus")
    if report.get("numerical_mechanics_passed") is not True:
        raise ValueError("representation preflight did not pass mechanics")
    return choice


def macro_cache(args, decoder, value_cfg, macro_cfg, corpus, output):
    from pde_agent_verifier_solver.plugins.vpfp.moving_frame import MovingFrameMacroConfig, extract_macro_controls
    representation = MovingFrameMacroConfig(control_count=macro_cfg.control_count,
        max_mode=macro_cfg.maximum_mode, time_domain=macro_cfg.time_domain,
        spatial_domain_lengths=tuple(decoder.config.equation.x_domain))
    cache = output / "macro-targets"
    cache.mkdir()
    arrays, receipts = {}, {}
    for role in ("train", "validation"):
        split = load_split(corpus,role,verify_hashes=False)
        path = cache / f"{role}.npy"
        values = np.lib.format.open_memmap(path,mode="w+",dtype="float32",shape=(len(split),macro_cfg.latent_dimension))
        rows = []
        for index in range(len(split)):
            raw = torch.from_numpy(np.array(split.latents[index:index+1],copy=True)).to(args.device)
            z = extract_value_controls(raw,value_cfg)
            condition = torch.from_numpy(np.array(split.conditions[index:index+1],copy=True)).to(args.device)
            m, report = extract_macro_controls(decoder,z,condition,
                representation,fit_grid=args.selected_fit_grid,value_config=value_cfg)
            m = torch.as_tensor(m).detach().cpu().numpy()
            if m.shape != (1,macro_cfg.latent_dimension) or not np.isfinite(m).all():
                raise ValueError("macro target extraction returned an invalid complete row")
            values[index] = m[0]
            rows.append(report)
            if (index+1)%8 == 0:
                values.flush()
                shared.emit("coherent_macro_targets",role=role,count=index+1,total=len(split))
        values.flush()
        receipts[role] = {"file":str(path),"sha256":shared.file_sha(path),
            "case_ids":list(map(str,split.case_ids)),"sample_ids":list(map(int,split.sample_ids)),
            "extraction":rows}
        arrays[role] = values
        shared.write_json(cache/f"{role}.json",receipts[role])
    return arrays, receipts, representation


def loader_with_view(corpus, macro_arrays, *, mode, normalizer=None):
    def load(path, role, verify_hashes=False):
        if Path(path).resolve() != Path(corpus).resolve() or role not in ("train","validation"):
            raise ValueError("training view may open only bound train/validation")
        split = load_split(path,role,verify_hashes=verify_hashes)
        values = np.asarray(macro_arrays[role])
        if len(values) != len(split):
            raise ValueError("macro view row count changed")
        if mode == "macro":
            return replace(split,coefficients=values)
        if mode != "conditional" or normalizer is None:
            raise ValueError("invalid training view")
        m = normalizer.encode(torch.from_numpy(np.array(values,copy=True))).cpu().numpy().astype(np.float32)
        return replace(split,conditions=np.concatenate((split.conditions,m),axis=1))
    return load


def fit_micro(cfg,raw,*,sample_ids):
    return VPFPTemporalCovarianceNormalizer.fit(cfg,raw,sample_ids=sample_ids,
        relative_floor=.05,center_kinetic_dc=True,rho_policy="identity",covariance_strength=.5)


def load_pair(entry, macro_cfg, value_cfg, device):
    payloads = {}
    for role in ("macro","conditional"):
        item = entry[role]
        payloads[role] = torch.load(item["checkpoint"],map_location="cpu",weights_only=True)
    fields = {"schema","config","model","normalizer","base_decoder_sha256","corpus_manifest_sha256","training"}
    for role,cfg,model_type in (("macro",macro_cfg,VPFPCoherentMacroEDM),("conditional",value_cfg,VPFPMacroConditionalEDM)):
        payload = payloads[role]
        if (set(payload) != fields or payload["schema"] != model_type.checkpoint_schema
                or payload["config"] != cfg.to_dict()):
            raise ValueError("hierarchical frozen native asset schema/config mismatch")
    for key in ("base_decoder_sha256","corpus_manifest_sha256"):
        if payloads["macro"][key] != payloads["conditional"][key]:
            raise ValueError("hierarchical pair belongs to different source/decoder identities")
    macro_norm_sha = json_sha(payloads["macro"]["normalizer"])
    if payloads["conditional"]["training"]["ablation"].get("macro_normalizer_sha256") != macro_norm_sha:
        raise ValueError("micro conditioning normalization differs from the frozen macro prior")
    if payloads["conditional"]["training"]["ablation"].get("macro_checkpoint_sha256") != entry["macro"]["checkpoint_sha256"]:
        raise ValueError("micro conditioning is not bound to this frozen macro checkpoint")
    macro = VPFPCoherentMacroEDM(macro_cfg).to(device).eval()
    micro = VPFPMacroConditionalEDM(value_cfg,macro_cfg).to(device).eval()
    for role,model in (("macro",macro),("conditional",micro)):
        expected = model.state_dict()
        state = payloads[role]["model"]
        if set(state) != set(expected):
            raise ValueError("hierarchical checkpoint state keys differ")
        buffers = dict(model.named_buffers())
        for name,tensor in state.items():
            if (not isinstance(tensor,torch.Tensor) or tensor.shape != expected[name].shape
                    or tensor.dtype != expected[name].dtype or not torch.isfinite(tensor).all()
                    or (name in buffers and not torch.equal(tensor,expected[name].cpu()))):
                raise ValueError(f"invalid hierarchical parameter or immutable Fourier buffer: {name}")
        model.load_state_dict(state,strict=True)
    macro_norm = VPFPCoherentMacroNormalizer.from_payload(payloads["macro"]["normalizer"]).to(device)
    micro_norm = VPFPTemporalCovarianceNormalizer.from_payload(payloads["conditional"]["normalizer"]).to(device)
    if (macro_norm.config != macro_cfg or micro_norm.config != value_cfg
            or macro_norm.fit_metadata["sample_ids"] != micro_norm.fit_metadata["sample_ids"]):
        raise ValueError("hierarchical model/normalizer or train lineage mismatch")
    return macro,macro_norm,micro,micro_norm


def write_native_manifest(spec, entry, output):
    """Explicit asset handoff; the native loader still validates every model."""
    from pde_agent_verifier_solver.plugins.vpfp.moving_frame_binding import MANIFEST_SCHEMA
    manifest = {"schema":MANIFEST_SCHEMA,"regime_id":spec.regime_id,"sampler_steps":64,
        "base_decoder":{"path":str(Path(spec.base_decoder_checkpoint).resolve()),"sha256":spec.base_decoder_sha256}}
    for role in ("macro","conditional"):
        manifest[role+"_checkpoint"] = {"path":str(Path(entry[role]["checkpoint"]).resolve()),
            "sha256":entry[role]["checkpoint_sha256"]}
    for role in ("base_decoder","macro_checkpoint","conditional_checkpoint"):
        item = manifest[role]
    path = output/"native-moving-frame-model.json"
    if path.exists() or path.is_symlink():
        raise ValueError("refusing to overwrite native A manifest")
    shared.write_json(path,manifest)
    return {"manifest":str(path),"manifest_sha256":shared.file_sha(path),
        "native_loading_and_campaign_smoke_completed":False,"scientific_admission":False}


def sample_frozen(args, entries, macro_cfg, value_cfg, output):
    """All checkpoints have frozen before opening any generated diagnostics."""
    result = {}
    conditions = shared.balanced_conditions(value_cfg,args.generated_answers,args.seed+200)
    seeds = list(range(args.seed+300000000,args.seed+300000000+len(conditions)))
    for replica, entry in enumerate(entries):
        for arm in ("W0","paired_AB"):
            directory = output/f"seed-{replica}"/arm
            directory.mkdir(exist_ok=True)
            raw = np.lib.format.open_memmap(directory/"generated-coefficients.npy",mode="w+",dtype="float32",
                shape=(len(conditions),value_cfg.full_dimension))
            macro_values = None
            if arm == "W0":
                sampler = load_vpfp_diffusion_sampler(entry["W0"]["checkpoint"],entry["W0"]["checkpoint_sha256"],device=args.device)
            else:
                pair = load_pair(entry,macro_cfg,value_cfg,args.device)
                macro_values = np.lib.format.open_memmap(directory/"generated-macro.npy",mode="w+",dtype="float32",
                    shape=(len(conditions),macro_cfg.latent_dimension))
            failures = []
            for index, (condition,seed) in enumerate(zip(conditions,seeds)):
                try:
                    if arm == "W0":
                        value = sampler.normalizer.decode(sampler.model.sample(condition[None],[seed],steps=args.sample_steps))
                    else:
                        draw = sample_hierarchical(*pair,condition[None],[seed],steps=args.sample_steps)
                        value = draw["full_coefficients"]
                        macro_values[index] = draw["macro_coordinates"][0].cpu().numpy()
                    raw[index] = value[0].cpu().numpy()
                    if not np.isfinite(raw[index]).all() or (macro_values is not None and not np.isfinite(macro_values[index]).all()):
                        raise FloatingPointError("nonfinite generated native answer")
                except (ValueError,RuntimeError,FloatingPointError) as error:
                    raw[index] = np.nan
                    if macro_values is not None:
                        macro_values[index] = np.nan
                    failures.append({"attempt":index,"seed":seed,"type":type(error).__name__,"reason":str(error)})
                if (index+1)%8 == 0:
                    raw.flush()
                    shared.emit("coherent_sample",replica=replica,arm=arm,count=index+1,total=len(conditions))
            raw.flush()
            if macro_values is not None:
                macro_values.flush()
            np.save(directory/"generated-conditions.npy",conditions,allow_pickle=False)
            receipt = {"schema":"vpfp_coherent_generated_native_pair_v1","attempted":len(conditions),
                "failures":failures,"seeds":seeds,"checkpoint_bindings":entry,
                "coefficients_sha256":shared.file_sha(directory/"generated-coefficients.npy"),
                "conditions_sha256":shared.file_sha(directory/"generated-conditions.npy"),
                "macro_sha256":None if macro_values is None else shared.file_sha(directory/"generated-macro.npy"),
                "decoder_views":["old_full_decoder"] if arm == "W0" else ["B_old_full_decoder","A_continuous_moving_frame"],
                "scientific_admission":False,"full_coefficients_are_A_answer":False,
                "macro_config":macro_cfg.to_dict(),"micro_config":value_cfg.to_dict(),
                "audit_status":"pending_independent_physical_and_numerical_audit"}
            shared.write_json(directory/"generated-provenance.json",receipt)
            result[f"seed-{replica}/{arm}"] = receipt
    return result


def parser():
    p = shared.parser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--representation-selection",type=Path,required=True)
    p.add_argument("--generated-answers",type=int,default=128)
    p.add_argument("--replicas",type=int,default=2)
    p.set_defaults(steps=20000,train_seconds=7200,validate_every=1000,batch_size=4,
                   seed=1520260906,sample_steps=64)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    args.output,args.source = args.output.resolve(),args.source.resolve()
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_TMPDIR"):
        p.error("training/target extraction require a Slurm compute allocation")
    if args.output.exists() or not args.output.is_relative_to(Path("/scratch")) or args.output.is_relative_to(args.source):
        p.error("choose a fresh /scratch output outside immutable source")
    if args.replicas != 2 or args.steps != 20000 or args.batch_size != 4 or args.generated_answers != 128 or args.sample_steps != 64:
        p.error("this registered screen requires two seeds,20k steps,batch4,128 attempts and64 Heun steps")
    corpus = args.source/"corpus"
    selection = require_selection(args.representation_selection,corpus)
    args.selected_fit_grid = selection["selected_fit_grid"]
    spec = VPFPOnlineModelSpec.from_mapping(json.loads(args.config.read_text())["vpfp"])
    manifest = read_manifest(corpus,verify_hashes=True)
    args.source_revision = subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    args.output.mkdir()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    args.checkpoint_steps = (20000,)
    full_cfg = shared.model_config(args,spec)
    value_cfg = VPFPValueEDMConfig.from_full_config(full_cfg)
    macro_cfg = VPFPCoherentMacroConfig(full_cfg,maximum_mode=selection["selected_mode"])
    started = time.monotonic()
    result = {"execution_passed":False,"scientific_admission":False,"training":[],"stage":"prepare"}
    try:
        binding = build_vpfp_online_binding(spec,output_root=args.output/"runtime",device=args.device)
        decoder = binding.plugin.generator.backend.runtime.base_model
        macro_arrays,receipts,_ = macro_cache(args,decoder,value_cfg,macro_cfg,corpus,args.output)
        transform_sha = json_sha({"macro_config":macro_cfg.to_dict(),"macro_files":
            {role:receipt["sha256"] for role,receipt in receipts.items()},"selection":selection})
        shared.write_json(args.output/"frozen-plan.json",{"source_revision":args.source_revision,
            "source_file_sha256":shared.source_fingerprints(),"source_manifest_sha256":manifest["manifest_sha256"],
            "representation_selection":selection,"macro_targets":receipts,"arguments":
            {key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()},
            "public_matcher_changed":False,"scientific_admission":False})
        for replica in range(2):
            run = copy(args)
            run.seed = args.seed+replica*10000
            entry = {}
            parent = args.output/f"seed-{replica}"
            parent.mkdir()
            for arm in ("W0","macro","conditional"):
                result["stage"] = f"seed-{replica}/{arm}"
                directory = parent/arm
                directory.mkdir()
                run.ablation_metadata = {"schema":"vpfp_coherent_hierarchy_ablation_v1","arm":arm,
                    "macro_config":macro_cfg.to_dict(),"data_transform_sha256":transform_sha,
                    "same_micro_initialization_seed":run.seed+100,"equal_compute_claimed":False}
                options = dict(model_config_factory=lambda *a:value_cfg,model_factory=VPFPTemporalCovarianceEDM,
                               normalizer_factory=fit_micro)
                if arm == "macro":
                    options = dict(model_config_factory=lambda *a:macro_cfg,model_factory=VPFPCoherentMacroEDM,
                        normalizer_factory=VPFPCoherentMacroNormalizer.fit,
                        split_loader=loader_with_view(corpus,macro_arrays,mode="macro"))
                elif arm == "conditional":
                    macro_payload = torch.load(entry["macro"]["checkpoint"],map_location="cpu",weights_only=True)
                    macro_norm = VPFPCoherentMacroNormalizer.from_payload(macro_payload["normalizer"])
                    run.ablation_metadata["macro_normalizer_sha256"] = json_sha(macro_payload["normalizer"])
                    run.ablation_metadata["macro_checkpoint_sha256"] = entry["macro"]["checkpoint_sha256"]
                    run.ablation_metadata["data_transform_sha256"] = json_sha({"base_transform_sha256":transform_sha,
                        "macro_normalizer_sha256":run.ablation_metadata["macro_normalizer_sha256"],
                        "macro_checkpoint_sha256":run.ablation_metadata["macro_checkpoint_sha256"]})
                    options.update(model_factory=lambda cfg:VPFPMacroConditionalEDM(cfg,macro_cfg),
                        split_loader=loader_with_view(corpus,macro_arrays,mode="conditional",normalizer=macro_norm))
                _,summary = shared.train(run,spec,corpus,directory,**options)
                if summary["final_step"] != 20000:
                    raise RuntimeError(f"{arm} stopped before the fixed20k comparison; no silent shorter run")
                # Use the registered terminal-step state, not post-audit selection.
                checkpoint = directory/"checkpoints/fixed-step-020000.pt"
                entry[arm] = {"checkpoint":str(checkpoint),"checkpoint_sha256":shared.file_sha(checkpoint),"summary":summary}
                shared.write_json(parent/"frozen-checkpoints.json",entry)
            entry["native_A_asset"] = write_native_manifest(spec,entry,parent)
            shared.write_json(parent/"frozen-checkpoints.json",entry)
            result["training"].append(entry)
        result["stage"] = "sample_frozen"
        result["generated"] = sample_frozen(args,result["training"],macro_cfg,value_cfg,args.output)
        read_manifest(corpus,verify_hashes=True)
        result.update(execution_passed=True,stage="training_and_sampling_complete_audit_pending")
    except Exception as error:
        result["failure"] = {"type":type(error).__name__,"reason":str(error),"traceback":traceback.format_exc()}
        raise
    finally:
        result["wall_seconds"] = time.monotonic()-started
        shared.write_json(args.output/"result.json",result)


if __name__ == "__main__":
    main()
