"""Explicit research p(m)p(z|m), with complete innovation inverse and A carrier.

No public native dispatcher registration, training, guidance, PDE solve or
physical admission. A drive4/8 prior and a new conditional score may have
different seeds and normalizers; the SAME physical33 m connects both stages.
"""
from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from .coherent_diffusion import VPFPCoherentMacroConfig, VPFPCoherentMacroNormalizer
from .coherent_velocity_diffusion import (SCHEMA as MACRO_SCHEMA,
    load_velocity_macro_checkpoint, decode_macro_coordinates)
from .innovation_diffusion import (CHECKPOINT_SCHEMA, PRECISION_POLICY, VPFPInnovationConditionalEDM,
    VPFPInnovationNormalizer, decode_complete_innovation_answer)
from .temporal_innovation import VPFPTemporalInnovationTransform, SCHEMA as TEMPORAL_SCHEMA
from .bulk_transport_innovation import VPFPBulkTransportInnovationTransform, SCHEMA as BULK_SCHEMA
from .value_control_diffusion import VPFPValueEDMConfig, extract_value_controls, lift_value_controls
from .structured_diffusion import _seeds
from .moving_frame_binding import (file_sha, json_sha, _read_asset, _load_state,
    build_moving_frame_decoder, binding_from_sampler)

ROOT = Path(__file__).resolve().parents[4]
MANIFEST_SCHEMA = "vpfp_innovation_moving_frame_research_manifest_v2"
CONTEXT_SCHEMA = "vpfp_innovation_conditioning_context_v2"
VIEW_SCHEMA = "vpfp_complete_innovation_training_view_binding_v2"
PREPARED_SCHEMA = "vpfp_prepared_complete_innovation_training_view_v2"
ASSETS = ("base_decoder", "macro_checkpoint", "conditional_checkpoint", "conditioning_context")
SOURCE_GRAPH = tuple("src/pde_agent_verifier_solver/plugins/vpfp/"+name for name in (
    "innovation_diffusion.py", "temporal_innovation.py", "bulk_transport_innovation.py",
    "coherent_diffusion.py", "structured_diffusion.py", "value_control_diffusion.py", "moving_frame.py"
)) + ("scripts/vpfp_innovation_training_view.py", "scripts/train_vpfp_structured_diffusion.py",
      "scripts/vpfp_innovation_precision_gate.py")
MACRO_GRAPH = tuple("src/pde_agent_verifier_solver/plugins/vpfp/"+name for name in (
    "coherent_velocity_diffusion.py", "coherent_velocity_dynamics.py",
    "coherent_diffusion.py", "structured_diffusion.py"))
CHECKPOINT_FIELDS = {"schema", "config", "model", "normalizer", "base_decoder_sha256",
                     "corpus_manifest_sha256", "training"}


def _exact(left, right):
    return json.dumps(left, sort_keys=True, separators=(",", ":"), allow_nan=False) == json.dumps(
        right, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("explicit lowercase SHA256 required")
    return value


def _asset(item):
    if (not isinstance(item, dict) or set(item) != {"path", "sha256"}
            or type(item["path"]) is not str or not Path(item["path"]).is_absolute()):
        raise ValueError("asset requires exact absolute path and SHA256")
    _digest(item["sha256"])
    return deepcopy(item)


def _training(step, seed):
    if type(step) is not int or step < 1 or type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("registered positive fixed step and nonnegative int64 training seed required")
    return {"step": step, "seed": seed}


def make_research_manifest(*, regime_id, base_decoder, macro_checkpoint, conditional_checkpoint,
        conditioning_context, macro_seed, micro_seed, micro_step, macro_step=5000, sampler_steps=64):
    if type(regime_id) is not str or not regime_id.strip() or type(sampler_steps) is not int or sampler_steps != 64:
        raise ValueError("explicit regime and registered 64-step research sampling required")
    if type(macro_step) is not int or macro_step != 5000:
        raise ValueError("this research composition uses the frozen 5000-step drive4/8 prior")
    return {"schema": MANIFEST_SCHEMA, "regime_id": regime_id,
        "base_decoder": _asset(base_decoder), "macro_checkpoint": _asset(macro_checkpoint),
        "conditional_checkpoint": _asset(conditional_checkpoint), "conditioning_context": _asset(conditioning_context),
        "training_contract": {"macro": _training(macro_step, macro_seed), "conditional": _training(micro_step, micro_seed)},
        "sampler_steps": sampler_steps}


def _source_graph(recorded, expected):
    if not isinstance(recorded, dict) or set(recorded) != set(expected):
        raise ValueError("exact registered runtime source graph required")
    for relative in expected:
        path = ROOT/relative
        if path.is_symlink() or not path.is_file() or file_sha(path) != _digest(recorded[relative]):
            raise ValueError("research runtime source graph differs: "+relative)


def _roots(value):
    if not isinstance(value, dict):
        raise ValueError("complete training-view root binding required")
    ids, cases = value.get("sample_ids"), value.get("case_ids")
    if (not isinstance(ids, list) or not ids or any(type(i) is not int or i < 0 for i in ids)
            or len(set(ids)) != len(ids) or not isinstance(cases, list) or len(cases) != len(ids)
            or any(type(c) is not str or not c for c in cases) or len(set(cases)) != len(cases)):
        raise ValueError("distinct complete root sample/case IDs required")
    for key in ("source_manifest_identity_sha256", "base_decoder_sha256", "conditions_sha256", "full_coefficients_sha256"):
        _digest(value.get(key))
    return ids, cases


def _context(context):
    if (not isinstance(context, dict) or set(context) != {"schema", "macro_normalizer", "training_view_binding"}
            or context["schema"] != CONTEXT_SCHEMA):
        raise ValueError("exact new innovation conditioning context required")
    normalizer = VPFPCoherentMacroNormalizer.from_payload(context["macro_normalizer"])
    if not _exact(normalizer.to_payload(), context["macro_normalizer"]):
        raise ValueError("noncanonical micro-conditioning macro normalizer")
    view = context["training_view_binding"]
    expected = {"schema", "corpus", "views", "source_graph_sha256", "coordinate_transform", "macro_config",
        "macro_normalizer_sha256", "view_dtype", "precision_policy", "no_derived_coordinate_is_a_physical_answer",
        "fit_roles", "held_out_role", "independent_unit"}
    if (not isinstance(view, dict) or set(view) != expected or view["schema"] != VIEW_SCHEMA
            or view["view_dtype"] != "float64" or view["precision_policy"] != PRECISION_POLICY
            or view["fit_roles"] != ["train"]
            or view["held_out_role"] != "validation" or view["independent_unit"] != "complete root trajectory"
            or view["no_derived_coordinate_is_a_physical_answer"] is not True
            or type(view["corpus"]) is not str or not Path(view["corpus"]).is_absolute()
            or set(view["views"]) != {"train", "validation"}):
        raise ValueError("invalid complete train/validation view context")
    _source_graph(view["source_graph_sha256"], SOURCE_GRAPH)
    if (not _exact(normalizer.config.to_dict(), view["macro_config"])
            or json_sha(normalizer.to_payload()) != view["macro_normalizer_sha256"]):
        raise ValueError("conditional macro configuration/normalizer binding differs")
    for role in ("train", "validation"):
        item = view["views"][role]
        ids, _ = _roots(item["source"])
        if (item.get("schema") != PREPARED_SCHEMA or item.get("storage_dtype") != "float64"
                or item.get("conditional_inputs_dtype") != "float32"
                or item.get("precision_policy") != PRECISION_POLICY
                or item.get("role") != role or item.get("macro_normalizer_sha256") != view["macro_normalizer_sha256"]
                or item.get("coordinate_transform_sha256") != json_sha(view["coordinate_transform"])
                or type(item.get("count_complete_root_trajectories")) is not int
                or item["count_complete_root_trajectories"] != len(ids)
                or item.get("generated_or_audit_reference_rows_used") is not False
                or item.get("held_out_validation") is not (role == "validation")
                or item.get("is_physical_full_answer") is not False
                or item.get("time_nodes_are_independent_samples") is not False):
            raise ValueError("training-view row/coordinate/role binding differs")
        for key in ("clean_physical_macro_sha256", "coordinates_sha256", "conditional_inputs_sha256"):
            _digest(item.get(key))
    train, validation = (view["views"][role]["source"] for role in ("train", "validation"))
    if (set(train["sample_ids"]) & set(validation["sample_ids"])
            or set(train["case_ids"]) & set(validation["case_ids"])
            or normalizer.fit_metadata.get("sample_ids") != train["sample_ids"]
            or len(train["sample_ids"]) < 2):
        raise ValueError("macro fit and independent train/validation roots differ")
    for key in ("source_manifest_identity_sha256", "base_decoder_sha256"):
        if train[key] != validation[key]: raise ValueError("train/validation source identity differs")
    coordinate = view["coordinate_transform"]
    if coordinate.get("schema") == TEMPORAL_SCHEMA:
        transform = VPFPTemporalInnovationTransform.from_payload(coordinate)
    elif coordinate.get("schema") == BULK_SCHEMA:
        transform = VPFPBulkTransportInnovationTransform.from_payload(coordinate)
    else:
        raise ValueError("unknown full VALUE innovation coordinate schema")
    cfg, macro = transform.config.value_config, normalizer.config
    if (cfg.full_config != macro.full_config
            or any(view["views"][role].get("coordinate_dimension") != cfg.latent_dimension for role in ("train", "validation"))):
        raise ValueError("micro-conditioned full VALUE layout differs")
    return normalizer, transform, view


def make_conditioning_context(macro_normalizer, training_view_binding):
    if not isinstance(macro_normalizer, VPFPCoherentMacroNormalizer):
        raise TypeError("the actual micro conditioning train macro normalizer is required")
    result = {"schema": CONTEXT_SCHEMA, "macro_normalizer": deepcopy(macro_normalizer.to_payload()),
              "training_view_binding": deepcopy(training_view_binding)}
    _context(result)
    return result


@contextmanager
def _fp32_initialization():
    original = torch.get_default_dtype()
    with torch.random.fork_rng(devices=[]):
        try:
            torch.set_default_dtype(torch.float32)
            yield
        finally:
            torch.set_default_dtype(original)


def _header(payload, schema, contract, base_sha, view):
    if not isinstance(payload, dict) or set(payload) != CHECKPOINT_FIELDS or payload["schema"] != schema:
        raise ValueError("exact registered seven-field research checkpoint required; no old micro/continuation")
    train = payload["training"]
    if (not isinstance(train, dict) or any(type(train.get(k)) is not int or train[k] != v for k,v in contract.items())
            or type(train.get("presentations")) is not int or train["presentations"] < contract["step"]
            or payload["base_decoder_sha256"] != base_sha
            or payload["corpus_manifest_sha256"] != view["views"]["train"]["source"]["source_manifest_identity_sha256"]
            or train.get("normalizer_sha256") != json_sha(payload["normalizer"])):
        raise ValueError("checkpoint fixed update/seed/normalizer/source identity differs")
    for role, key in (("train", "train_case_ids"), ("validation", "validation_case_ids")):
        if train.get(key) != view["views"][role]["source"]["case_ids"]:
            raise ValueError("checkpoint complete root lineage differs")
    return train


def _module_sha(module):
    digest = sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(json.dumps([name, list(tensor.shape), str(tensor.dtype)]).encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class InnovationMovingFrameSampler:
    """Research sampler; .macro_norm is MICRO conditioning, not prior whitening."""
    def __init__(self, macro, prior_norm, micro, micro_norm, macro_norm, transform, *, identity, steps=64):
        self.macro, self.macro_prior_norm, self.micro = macro, prior_norm, micro
        self.micro_norm, self.macro_norm, self.transform = micro_norm, macro_norm, transform
        self.config, self.steps, self.identity = micro.config, steps, deepcopy(identity)
        self._fixed_steps = steps
        for model in (macro, micro): model.eval().requires_grad_(False)
        self._models = (_module_sha(macro), _module_sha(micro))
        self._normalizers = tuple(json_sha(n.to_payload()) for n in (prior_norm, micro_norm, macro_norm))
        self._transform_sha, self._identity_sha = json_sha(transform.to_payload()), json_sha(self.identity)

    def assert_frozen(self):
        # Full bytes only at draw boundaries, never in denoise/Heun steps.
        if (self.steps != self._fixed_steps or tuple(_module_sha(m) for m in (self.macro, self.micro)) != self._models
                or any(p.requires_grad for m in (self.macro, self.micro) for p in m.parameters())
                or any(m.training for m in (self.macro, self.micro))
                or tuple(json_sha(n.to_payload()) for n in (self.macro_prior_norm,self.micro_norm,self.macro_norm)) != self._normalizers
                or json_sha(self.transform.to_payload()) != self._transform_sha
                or json_sha(self.identity) != self._identity_sha):
            raise ValueError("loaded research sampler weights, buffers, normalization or identity changed")

    @torch.no_grad()
    def sample_conditions(self, conditions, root_seeds, *, macro_noise_seeds=None, micro_noise_seeds=None):
        self.assert_frozen()
        roots = _seeds(root_seeds)
        if ((macro_noise_seeds is None) != (micro_noise_seeds is None)):
            raise ValueError("provide both explicit noise streams or neither")
        if macro_noise_seeds is None:
            def stream(name):
                return [int.from_bytes(sha256(f"vpfp-innovation-{name}:{s}".encode()).digest()[:8],"little") % 2**63 for s in roots]
            macro_noise_seeds, micro_noise_seeds = stream("macro"), stream("micro")
        macro_seeds, micro_seeds = _seeds(macro_noise_seeds), _seeds(micro_noise_seeds)
        if len(macro_seeds) != len(roots) or len(micro_seeds) != len(roots) or set(macro_seeds) & set(micro_seeds):
            raise ValueError("distinct registered per-root macro/micro streams required")
        if isinstance(conditions, np.ndarray):
            if conditions.dtype != np.dtype("float32"): raise ValueError("exact FP32 nu/T conditions required")
            theta = torch.from_numpy(np.array(conditions,copy=True))
        else:
            theta = conditions
        if (not isinstance(theta,torch.Tensor) or theta.layout != torch.strided or theta.dtype != torch.float32
                or theta.device.type not in ("cpu", "cuda")
                or theta.shape != (len(roots),2) or not bool(torch.isfinite(theta).all())):
            raise ValueError("exact finite FP32 [B,2] physical conditions required")
        device = next(self.micro.parameters()).device
        theta = theta.to(device=device)
        low = theta.new_tensor([self.config.collision_nu_range[0],self.config.temperature_range[0]])
        high = theta.new_tensor([self.config.collision_nu_range[1],self.config.temperature_range[1]])
        if bool(((theta < low) | (theta > high)).any()): raise ValueError("conditions outside frozen physical range")
        white_m = self.macro.sample(theta, macro_seeds, steps=self.steps)
        raw_m = self.macro_prior_norm.decode(white_m)
        m = decode_macro_coordinates(raw_m, theta, self.macro.config)
        condition_m = self.macro_norm.encode(m)
        white_z = self.micro.sample(torch.cat((theta,condition_m),dim=1),micro_seeds,steps=self.steps)
        full = decode_complete_innovation_answer(white_z,self.micro_norm,self.transform,physical_macro=m)
        z = extract_value_controls(full,self.config,validate_slopes=True)
        if not torch.equal(lift_value_controls(z,self.config),full):
            raise ValueError("complete native z lift lost slope/inactive bits")
        if not bool(torch.isfinite(m).all()) or not bool(torch.isfinite(z).all()):
            raise ValueError("nonfinite complete A carrier")
        self.assert_frozen()
        return {"schema":"vpfp_moving_frame_native_answer_batch_v1","m":m.detach().cpu(),"z":z.detach().cpu(),
            "full_coefficients":full.detach().cpu(),"full_coefficients_are_complete_A_answer":False,
            "full_state_role":"complete native z carrier; A additionally requires same m",
            "equation_conditions":theta.cpu().numpy(),"equation_seeds":np.asarray(roots,dtype=np.int64),
            "macro_noise_seeds":list(macro_seeds),"micro_noise_seeds":list(micro_seeds),"pde_solves":0}

    def generate(self, root_seeds):
        roots = _seeds(root_seeds)
        rows = []
        for seed in roots:
            stream = int.from_bytes(sha256(f"vpfp-edm-condition:{seed}".encode()).digest()[:8],"little")
            rng = np.random.default_rng(stream)
            rows.append([rng.uniform(*self.config.collision_nu_range),rng.uniform(*self.config.temperature_range)])
        return self.sample_conditions(np.asarray(rows,dtype=np.float32),roots)


def load_innovation_research_sampler(manifest_path, expected_sha256, *, device="cuda"):
    manifest = _read_asset({"path":str(manifest_path),"sha256":_digest(expected_sha256)},json_only=True)
    expected = make_research_manifest(regime_id=manifest.get("regime_id"),
        **{name:manifest.get(name) for name in ASSETS},
        macro_seed=manifest.get("training_contract",{}).get("macro",{}).get("seed"),
        micro_seed=manifest.get("training_contract",{}).get("conditional",{}).get("seed"),
        macro_step=manifest.get("training_contract",{}).get("macro",{}).get("step"),
        micro_step=manifest.get("training_contract",{}).get("conditional",{}).get("step"),
        sampler_steps=manifest.get("sampler_steps"))
    if not _exact(manifest,expected): raise ValueError("unknown research manifest fields/schema")
    context = _read_asset(manifest["conditioning_context"],json_only=True)
    macro_norm, transform, view = _context(context)
    base = _read_asset(manifest["base_decoder"])
    micro_payload = _read_asset(manifest["conditional_checkpoint"])
    cfg, macro_cfg = transform.config.value_config, macro_norm.config
    if view["views"]["train"]["source"]["base_decoder_sha256"] != manifest["base_decoder"]["sha256"]:
        raise ValueError("training view and actual A base asset differ")
    micro_meta = _header(micro_payload,CHECKPOINT_SCHEMA,manifest["training_contract"]["conditional"],
                         manifest["base_decoder"]["sha256"],view)
    micro_norm = VPFPInnovationNormalizer.from_payload(micro_payload["normalizer"])
    digest = json_sha(view)
    if (not _exact(micro_payload["config"],cfg.to_dict()) or micro_norm.config != cfg
            or micro_norm.fit_metadata.get("sample_ids") != view["views"]["train"]["source"]["sample_ids"]
            or not _exact(micro_norm.fit_metadata["coordinate_transform"],transform.to_payload())
            or micro_norm.fit_metadata.get("training_view_binding_sha256") != digest
            or (micro_meta.get("ablation") or {}).get("data_transform_sha256") != digest):
        raise ValueError("new conditional model/norm/coordinate/view binding differs")
    item = manifest["macro_checkpoint"]
    path = Path(item["path"])
    if path.is_symlink() or not path.is_file() or file_sha(path) != item["sha256"]:
        raise ValueError("frozen macro prior asset SHA differs")
    with _fp32_initialization():
        macro, prior_norm, macro_payload = load_velocity_macro_checkpoint(path,item["sha256"],device=device)
        micro = VPFPInnovationConditionalEDM(cfg,macro_cfg)
    if file_sha(path) != item["sha256"]: raise ValueError("macro prior changed during loading")
    macro_meta = _header(macro_payload,MACRO_SCHEMA,manifest["training_contract"]["macro"],
                         manifest["base_decoder"]["sha256"],view)
    if (macro.config.base_macro_config != macro_cfg
            or prior_norm.fit_metadata.get("sample_ids") != view["views"]["train"]["source"]["sample_ids"]):
        raise ValueError("prior/micro complete macro geometry or train roots differ")
    recorded = macro_meta.get("source_file_sha256") or {}
    _source_graph({key:recorded.get(key) for key in MACRO_GRAPH},MACRO_GRAPH)
    _load_state(micro,micro_payload)
    micro.to(device).eval().requires_grad_(False)
    macro_norm.to(device); micro_norm.to(device)
    with _fp32_initialization():
        decoder = build_moving_frame_decoder(base,macro_cfg,cfg,device)
    if isinstance(transform,VPFPBulkTransportInnovationTransform) and not _exact(
            transform.config.macro_config.to_dict(),decoder.macro_config.to_dict()):
        raise ValueError("bulk predictor differs from actual complete A geometry")
    identity = {"schema":"vpfp_innovation_moving_frame_research_sampler_v2","manifest_sha256":expected_sha256,
        "assets":manifest,"decoder":decoder.identity(),"corpus_manifest_sha256":micro_payload["corpus_manifest_sha256"],
        "training_view_binding_sha256":digest,"coordinate_transform_sha256":json_sha(transform.to_payload()),
        "micro_conditioning_macro_normalizer_sha256":json_sha(macro_norm.to_payload()),
        "precision_policy":deepcopy(PRECISION_POLICY),
        "macro_dimension":macro_cfg.latent_dimension,"native_value_dimension":cfg.latent_dimension,
        "sampling":{"method":"Heun","steps":64,"macro_prior":"frozen_drive4_or_drive8",
            "joint_streams":"domain_separated_or_both_explicit; same_physical33_m","condition_prior":"uniform_physical_nu_temperature"},
        "micro_weights":"new_innovation_checkpoint_not_reused_legacy_micro","learned_generation":True,
        "sampler_source_sha256":file_sha(Path(__file__)),"pde_solves_per_sample":0,"scientific_admission":False}
    sampler = InnovationMovingFrameSampler(macro,prior_norm,micro,micro_norm,macro_norm,transform,identity=identity)
    for name in ASSETS:
        if file_sha(manifest[name]["path"]) != manifest[name]["sha256"]:
            raise ValueError("research asset changed during complete load: "+name)
    if file_sha(manifest_path) != expected_sha256: raise ValueError("research manifest changed during load")
    return sampler,decoder
