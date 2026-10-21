"""Native hierarchical checkpoint loading and BASE-only VPFP A binding.

The private m/z carrier is explicitly versioned. Nothing labels A as a fixed
Hermite coefficient answer, and no training script is imported at runtime.
"""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import importlib
import json
from pathlib import Path

import numpy as np
import torch

from pde_agent_verifier_solver.sdk import DatasetInput,DatasetKind,PreparedDataset,GeneratorCheckpoint,ProgramInfo,PanelCard
from .canonical_binding import _BoundVPFPContinuousGenerator
from .coherent_diffusion import (VPFPCoherentMacroConfig,VPFPCoherentMacroEDM,VPFPCoherentMacroNormalizer,
    VPFPMacroConditionalEDM,sample_hierarchical)
from .value_control_diffusion import VPFPValueEDMConfig,extract_value_controls,lift_value_controls
from .temporal_covariance_diffusion import VPFPTemporalCovarianceNormalizer
from .structured_diffusion import _sha,_seeds
from .moving_frame import MovingFrameDecoder,MovingFrameMacroConfig
from .moving_frame_backend import MovingFrameBackend
from .perturbation_backend import _legacy_api
from .product import build_vpfp_plugin

MANIFEST_SCHEMA = "vpfp_moving_frame_native_model_manifest_v1"


def json_sha(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def file_sha(path):
    digest = sha256()
    with Path(path).open("rb") as source:
        for part in iter(lambda:source.read(1024*1024),b""):
            digest.update(part)
    return digest.hexdigest()


def _read_asset(item,*,json_only=False):
    if set(item)-{"sha256"}!={"path"}:
        raise ValueError("native asset requires a path")
    path = Path(item["path"]).expanduser()
    if path.is_symlink() or not path.is_file():
        raise ValueError("native asset missing or symlinked")
    value = json.loads(path.read_text()) if json_only else torch.load(path,map_location="cpu",weights_only=True)
    return value


def validate_hierarchical_payloads(macro_payload,conditional_payload,*,base_decoder_sha256,macro_checkpoint_sha256):
    """Lightweight header/binding validation shared with A audit adapters."""
    fields = {"schema","config","model","normalizer","base_decoder_sha256","corpus_manifest_sha256","training"}
    for payload,model_type in ((macro_payload,VPFPCoherentMacroEDM),(conditional_payload,VPFPMacroConditionalEDM)):
        if not isinstance(payload,dict) or set(payload)!=fields or payload["schema"]!=model_type.checkpoint_schema:
            raise ValueError("unexpected hierarchical native checkpoint schema/fields")
        if payload["base_decoder_sha256"]!=_sha(base_decoder_sha256,"base decoder"):
            raise ValueError("hierarchical checkpoint belongs to another base decoder")
        _sha(payload["corpus_manifest_sha256"],"corpus")
        json_sha(payload["training"])
        if (type(payload["training"].get("step")) is not int or payload["training"]["step"]!=20000
                or payload["training"].get("presentations")!=80000):
            raise ValueError("A native research asset requires the registered 20k/80k frozen state")
    macro_cfg = VPFPCoherentMacroConfig.from_dict(macro_payload["config"])
    value_cfg = VPFPValueEDMConfig.from_dict(conditional_payload["config"])
    if (macro_cfg.full_config!=value_cfg.full_config
            or macro_payload["corpus_manifest_sha256"]!=conditional_payload["corpus_manifest_sha256"]):
        raise ValueError("hierarchical physical config/corpus mismatch")
    macro_meta,conditional_meta = macro_payload["training"],conditional_payload["training"]
    for name in ("seed","train_case_ids","validation_case_ids"):
        if macro_meta.get(name)!=conditional_meta.get(name) or macro_meta.get(name) is None:
            raise ValueError("hierarchical training lineage mismatch")
    ablation = conditional_meta.get("ablation") or {}
    if (ablation.get("macro_normalizer_sha256")!=json_sha(macro_payload["normalizer"])
            or ablation.get("macro_checkpoint_sha256")!=_sha(macro_checkpoint_sha256,"macro checkpoint")
            or ablation.get("macro_config")!=macro_cfg.to_dict()):
        raise ValueError("conditional model's actual macro checkpoint/normalizer binding differs")
    return macro_cfg,value_cfg


def build_moving_frame_decoder(base_payload,macro_cfg,value_cfg,device):
    api = _legacy_api()
    hermite = importlib.import_module("agent_verifier.vpfp.continuous_hermite")
    adapter = base_payload.get("robust_answer_first_adapter") or {}
    if (adapter.get("unused_neural_parameters_zeroed") is not True
            or adapter.get("continuous_hidden_answer_is_primary") is not True
            or adapter.get("generator_parameterization_public") is not False):
        raise ValueError("base decoder lacks the pinned deterministic answer-first contract")
    config = hermite.VPFPContinuousHermiteConfig.from_dict(base_payload["decoder_config"])
    if (tuple(config.collision_nu_range)!=tuple(value_cfg.collision_nu_range)
            or tuple(config.temperature_range)!=tuple(value_cfg.temperature_range)):
        raise ValueError("base decoder physical condition ranges differ")
    base = api.build_deterministic_direct_model(config,device=device).eval().requires_grad_(False)
    representation = MovingFrameMacroConfig(control_count=macro_cfg.control_count,max_mode=macro_cfg.maximum_mode,
        time_domain=macro_cfg.time_domain,spatial_domain_lengths=config.equation.x_domain)
    return MovingFrameDecoder(base,representation,value_config=value_cfg)


def _load_state(model,payload):
    expected,state = model.state_dict(),payload["model"]
    if not isinstance(state,dict) or set(state)!=set(expected):
        raise ValueError("native hierarchical state keys differ")
    buffers = dict(model.named_buffers())
    for name,tensor in state.items():
        if (not isinstance(tensor,torch.Tensor) or tensor.layout!=torch.strided
                or tensor.shape!=expected[name].shape or tensor.dtype!=expected[name].dtype
                or not torch.isfinite(tensor).all()
                or (name in buffers and not torch.equal(tensor.cpu(),expected[name].cpu()))):
            raise ValueError("invalid learned tensor or immutable Fourier buffer: "+name)
    model.load_state_dict(state,strict=True)


class MovingFrameSampler:
    def __init__(self,macro,macro_norm,micro,micro_norm,*,identity,steps):
        self.macro,self.macro_norm,self.micro,self.micro_norm = macro,macro_norm,micro,micro_norm
        self.identity,self.steps = dict(identity),steps

    @torch.no_grad()
    def sample_conditions(self,conditions,seeds):
        seeds = _seeds(seeds)
        draw = sample_hierarchical(self.macro,self.macro_norm,self.micro,self.micro_norm,conditions,seeds,steps=self.steps)
        full = draw["full_coefficients"]
        z = extract_value_controls(full,self.micro.config,validate_slopes=True)
        if not torch.equal(lift_value_controls(z,self.micro.config),full):
            raise ValueError("sampled full-to-value carrier lost native slope bits")
        return {"m":draw["macro_coordinates"].detach().cpu(),"z":z.detach().cpu(),
                "macro_noise_seeds":draw["macro_noise_seeds"],"micro_noise_seeds":draw["micro_noise_seeds"]}

    def generate(self,seeds):
        seeds = _seeds(seeds)
        cfg = self.micro.config
        conditions = []
        for seed in seeds:
            stream = int.from_bytes(sha256(f"vpfp-edm-condition:{seed}".encode()).digest()[:8],"little")
            rng = np.random.default_rng(stream)
            conditions.append((rng.uniform(*cfg.collision_nu_range),rng.uniform(*cfg.temperature_range)))
        conditions = np.asarray(conditions,dtype=np.float32)
        draw = self.sample_conditions(conditions,seeds)
        return {"schema":"vpfp_moving_frame_native_answer_batch_v1",**draw,"equation_conditions":conditions,
                "equation_seeds":np.asarray(seeds,dtype=np.int64),"pde_solves":0}


def load_moving_frame_sampler(manifest_path,expected_sha256,*,device="cuda"):
    manifest = _read_asset({"path":str(manifest_path),"sha256":expected_sha256},json_only=True)
    fields = {"schema","regime_id","base_decoder","macro_checkpoint","conditional_checkpoint","sampler_steps"}
    if (set(manifest)!=fields or manifest["schema"]!=MANIFEST_SCHEMA
            or not isinstance(manifest["regime_id"],str) or not manifest["regime_id"].strip()
            or type(manifest["sampler_steps"]) is not int or manifest["sampler_steps"]!=64):
        raise ValueError("invalid native moving-frame model manifest")
    base,macro_payload,micro_payload = (_read_asset(manifest[name]) for name in
        ("base_decoder","macro_checkpoint","conditional_checkpoint"))
    macro_cfg,value_cfg = validate_hierarchical_payloads(macro_payload,micro_payload,
        base_decoder_sha256=manifest["base_decoder"]["sha256"],macro_checkpoint_sha256=manifest["macro_checkpoint"]["sha256"])
    with torch.random.fork_rng(devices=[]):
        macro,micro = VPFPCoherentMacroEDM(macro_cfg),VPFPMacroConditionalEDM(value_cfg,macro_cfg)
    _load_state(macro,macro_payload); _load_state(micro,micro_payload)
    macro_norm = VPFPCoherentMacroNormalizer.from_payload(macro_payload["normalizer"])
    micro_norm = VPFPTemporalCovarianceNormalizer.from_payload(micro_payload["normalizer"])
    if (macro_norm.config!=macro_cfg or micro_norm.config!=value_cfg
            or macro_norm.fit_metadata["sample_ids"]!=micro_norm.fit_metadata["sample_ids"]
            or micro_norm.fit_metadata.get("rho_policy")!="identity"
            or micro_norm.fit_metadata.get("covariance_strength")!=.5
            or micro_norm.fit_metadata.get("centering")!="kinetic_spatial_DC_train_mean_only"
            or micro_norm.fit_metadata.get("relative_floor")!=.05
            or micro_norm.fit_metadata.get("absolute_floor")!=1.e-6
            or micro_norm.covariance_recipe!="mixture"):
        raise ValueError("native hierarchy normalization/lineage/fixed-half geometry differs")
    macro.to(device).eval().requires_grad_(False); micro.to(device).eval().requires_grad_(False)
    macro_norm.to(device); micro_norm.to(device)
    decoder = build_moving_frame_decoder(base,macro_cfg,value_cfg,device)
    identity = {"schema":"vpfp_moving_frame_native_sampler_v1","manifest_sha256":expected_sha256,
        "assets":manifest,"decoder":decoder.identity(),"corpus_manifest_sha256":macro_payload["corpus_manifest_sha256"],
        "macro_dimension":macro_cfg.latent_dimension,"native_value_dimension":value_cfg.latent_dimension,
        "sampling":{"method":"Heun","steps":64,"condition_prior":"uniform_physical_nu_temperature",
                    "joint_streams":"same_m_z_draw; domain_separated_macro_and_micro_noise"},
        "learned_generation":True,"pde_solves_per_sample":0,"scientific_admission":False}
    sampler = MovingFrameSampler(macro,macro_norm,micro,micro_norm,identity=identity,steps=64)
    return sampler,decoder


class MovingFrameGenerator(_BoundVPFPContinuousGenerator):
    info = ProgramInfo("vpfp-moving-frame-answer-first-v1","VPFP moving-frame Generator",
        "Generates continuous kinetic answers with explicit mean velocity and stress; macro-only tasks.")

    def preview(self,record):
        # Only already-public initial moments; never consult m/z or resident truth.
        arrays = record.case.public_data["arrays"]
        return PanelCard(title=f"VPFP case {record.case.case_id}",summary="Initial density, momentum and kinetic energy",
            values={"initial_mean_density":float(np.asarray(arrays["rho0"]).mean()),
                "initial_momentum_rms":float(np.sqrt(np.mean(np.asarray(arrays["momentum0"])**2))),
                "initial_mean_kinetic_energy":float(np.asarray(arrays["kinetic_energy0"]).mean())})


def binding_from_sampler(decoder,sampler,*,output_root,regime_id,**runtime_options):
    """Thin composition also used by synthetic mechanics tests."""
    from pde_agent_verifier_solver.app.backend_run import BackendRunBinding
    backend = MovingFrameBackend(decoder,sampler,output_root=output_root,**runtime_options)
    identity = {"schema":"vpfp_moving_frame_canonical_binding_v1","sampling":sampler.identity,
        "answer_decoder":decoder.identity(),"regime_id":regime_id,"guidance":"disabled","perturbations":"unsupported",
        "source_derivation":"actual_A_FP64_coordinate_autodiff_OU_Poisson_residual",
        "physical_module_sha256":{name:file_sha(Path(__file__).with_name(name)) for name in
            ("moving_frame.py","moving_frame_backend.py","moving_frame_binding.py")},
        "scientific_qualification":"not_established_by_native_or_campaign_smoke"}
    if backend.batched_source or backend.adaptive_source_only:
        identity["source_derivation"] = backend.source_implementation["source_derivation"]
        identity["numerical_implementation"] = backend.source_implementation
        identity["physical_module_sha256"].update(backend.source_implementation["physical_module_sha256"])
    identity["binding_sha256"] = json_sha(identity)
    generator = MovingFrameGenerator(backend,identity)
    plugin = replace(build_vpfp_plugin(),generator=generator)
    checkpoint = GeneratorCheckpoint(generator.info.program_id,identity,report=identity)
    dataset = PreparedDataset(DatasetInput(dataset_id=f"vpfp-{regime_id}-moving-frame",kind=DatasetKind.NUMERICAL,
        problem={"equation":"VPFP","regime_id":regime_id},settings={"online_only":True}),(),(),
        training_summary={"online_only":True,"frozen_checkpoint":True,"scientific_admission":False})
    return BackendRunBinding(plugin=plugin,checkpoint=checkpoint,dataset=dataset,report=identity)


def build_vpfp_moving_frame_binding(manifest_path,expected_sha256,*,output_root,device="cuda"):
    sampler,decoder = load_moving_frame_sampler(manifest_path,expected_sha256,device=device)
    return binding_from_sampler(decoder,sampler,output_root=output_root,regime_id=sampler.identity["assets"]["regime_id"])
