"""Typed inference export of the already strictly loaded weak H2/20k Base.

No training-history loader is bypassed: export follows its successful original
load. New runtime reconstruction retains every model tensor, normalizer and
sampling identity. Original versus new interface provenance is checked by the
separate exact-file migration proof. No object pickle or teacher arrays occur.

Keep this module's top level stdlib-only: the exporter loads it by file path
before importing the ORIGINAL frozen package, without contaminating that load.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
SCHEMA = "vpfp_weak_native_inference_portable_base_v1"
CLASSES = {
    "sampler": "pde_agent_verifier_solver.plugins.vpfp.innovation_binding.InnovationMovingFrameSampler",
    "macro": "pde_agent_verifier_solver.plugins.vpfp.hybrid_stress_diffusion.VPFPHybridStressJointMacroEDM",
    "micro": "run_vpfp_weak_replay_probe_screen.ReplayEDM",
    "decoder": "pde_agent_verifier_solver.plugins.vpfp.moving_frame.MovingFrameDecoder"}
PAYLOAD_KEYS = {"schema", "classes", "sampling", "macro", "micro", "normalizers", "coordinate_transform",
                "decoder", "origin_receipt", "provenance"}
# The 33-node Legendre QR is derived geometry, not a learned parameter.  Its
# saved float32 entries are order one.  Allow at most four float32 eps of
# reconstruction drift (zero relative tolerance); do not tune this to assets.
# The saved buffer remains authoritative and is restored byte-for-byte below.
PACKING_QR_ATOL = 2.0 ** -21
PACKING_QR_GRAM_ATOL = 2.0 ** -20


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _plain(value):
    return json.loads(_json(value))


def _typed_plain(value):
    """Copy finite data without changing native list/tuple container types.

    Frozen normalizers compare their reconstructed Python payloads exactly.
    JSON roundtripping changes config tuples to lists even though its SHA is
    unchanged, so it cannot serialize these typed constructor payloads.
    Both list and tuple are safe built-in containers for weights_only loading.
    """
    _json(value)
    def copy_data(item):
        if item is None or type(item) in (bool, int, float, str):
            return item
        if type(item) in (list, tuple):
            return type(item)(copy_data(part) for part in item)
        if type(item) is dict and all(type(key) is str for key in item):
            return {key: copy_data(part) for key, part in item.items()}
        raise ValueError("portable constructor payload requires only finite built-in data")
    return copy_data(value)


def _file_sha(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _class(value):
    return type(value).__module__ + "." + type(value).__qualname__


def _state_sha(state):
    # Same byte definition as the original innovation_binding._module_sha.
    import numpy as np  # noqa: F401 -- frozen NumPy-before-Torch convention
    import torch
    digest = sha256()
    for name, tensor in sorted(state.items()):
        if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
                or not bool(torch.isfinite(tensor).all())):
            raise ValueError("portable state requires finite strided tensors: " + name)
        digest.update(json.dumps([name, list(tensor.shape), str(tensor.dtype)]).encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _validate_provenance(provenance, *, runtime_interface_profile=None):
    path = ROOT / "scripts/vpfp_weak_portability_provenance.py"
    spec = importlib.util.spec_from_file_location("_weak_base_migration_proof", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    result = (module.validate_portability_provenance(provenance, ROOT)
        if runtime_interface_profile is None else
        module.validate_runtime_portability_provenance(provenance, ROOT, runtime_interface_profile))
    if result.get("execution_passed") is not True:
        raise ValueError("exact original-to-canonical source provenance failed")
    return result


def _origin(receipt, provenance):
    if (receipt.get("native_strict_load_passed") is not True or receipt.get("fixed_step") != 20000
            or receipt.get("sampling_steps") != 64 or receipt.get("new_training_steps") != 0
            or receipt.get("new_PDE_solves") != 0 or receipt.get("fresh_joint_draw_uses_H2_not_ancestral_drive4") is not True):
        raise ValueError("original successful native H2/fixed20k load receipt required")
    digest = sha256(_json(receipt).encode()).hexdigest()
    if (provenance.get("origin_receipt_sha256") != digest
            or _json(provenance.get("origin_receipt")) != _json(receipt)):
        raise ValueError("portable payload origin differs from exact migration provenance")


class _InferenceReplayContext:
    """Only the original identity; no training teacher/budget arrays attached."""
    def __init__(self, binding):
        self.binding = deepcopy(binding)

    def guard(self):
        raise RuntimeError("portable accepted Base is inference-only; replay training is unavailable")

    def sample(self, *args, **kwargs):
        self.guard()


def export_payload(sampler, decoder, receipt, *, provenance):
    """Return a tensor/built-in-data payload after the OLD strict loader succeeds."""
    import numpy as np  # noqa: F401
    import torch
    _origin(receipt, provenance); _validate_provenance(provenance)
    actual = {"sampler": _class(sampler), "macro": _class(sampler.macro),
              "micro": _class(sampler.micro), "decoder": _class(decoder)}
    if actual != CLASSES:
        raise ValueError("portable export requires exact H2, ReplayEDM and complete A classes")
    sampler.assert_frozen()
    if (sampler.steps != 64 or sampler.micro.source_step != 0 or sampler.micro.arm != "baseline"
            or sampler.micro.probe_arm != "stratified32" or sampler.micro._physics_context is not None
            or sampler.micro._capture_prediction or sampler.micro._inside_loss
            or _json(sampler.identity) != _json(receipt["sampler_identity"])):
        raise ValueError("only the frozen fixed20k EMA inference state may be exported")
    state = {"macro": _state(sampler.macro), "micro": _state(sampler.micro), "decoder": _state(decoder)}
    if (_state_sha(state["macro"]) != receipt["macro_state_sha256"]
            or _state_sha(state["micro"]) != receipt["model_state_sha256"]
            or sha256(_json(sampler.identity).encode()).hexdigest() != receipt["sampler_identity_sha256"]):
        raise ValueError("live original model/identity differs from its successful receipt")
    item = receipt["sampler_identity"]["assets"]["base_decoder"]
    path = Path(item["path"])
    if path.is_symlink() or _file_sha(path) != item["sha256"]:
        raise ValueError("original deterministic decoder asset changed")
    original_base = torch.load(path, map_location="cpu", weights_only=True)
    base_payload = _typed_plain({key: original_base[key] for key in ("decoder_config", "robust_answer_first_adapter")})
    if (_file_sha(path) != item["sha256"]
            or _json(base_payload["decoder_config"]) != _json(decoder.base_decoder.config.to_dict())):
        raise ValueError("actual decoder differs from its strictly loaded original asset")
    payload = {"schema": SCHEMA, "classes": dict(CLASSES),
        "sampling": {"steps": 64, "identity": _plain(sampler.identity),
            "normalizer_sha256": list(sampler._normalizers), "coordinate_transform_sha256": sampler._transform_sha},
        "macro": {"config": _typed_plain(sampler.macro.config.to_dict()), "state": state["macro"], "state_sha256": _state_sha(state["macro"])},
        "micro": {"config": _typed_plain(sampler.micro.config.to_dict()), "macro_config": _typed_plain(sampler.micro.macro_config.to_dict()),
            "spatial_domain_lengths": list(sampler.micro.spatial_domain_lengths), "probe_arm": sampler.micro.probe_arm,
            "replay_binding": _plain(sampler.micro.replay_context.binding), "identity": _plain(sampler.micro.identity()),
            "state": state["micro"], "state_sha256": _state_sha(state["micro"])},
        "normalizers": {"macro_prior": _typed_plain(sampler.macro_prior_norm.to_payload()),
            "micro": _typed_plain(sampler.micro_norm.to_payload()), "micro_macro": _typed_plain(sampler.macro_norm.to_payload())},
        "coordinate_transform": _typed_plain(sampler.transform.to_payload()),
        "decoder": {"base_payload": base_payload, "macro_config": _typed_plain(decoder.macro_config.to_dict()),
            "value_config": _typed_plain(decoder.value_config.to_dict()), "identity": _plain(decoder.identity()),
            "state": state["decoder"], "state_sha256": _state_sha(state["decoder"])},
        "origin_receipt": _plain(receipt), "provenance": _plain(provenance)}
    sampler.assert_frozen()
    return payload


def _packing_reconstruction_check(model, key, value, expected):
    """Qualify only the known H2 macro's fixed 33-node QR packing.

    This does not waive immutable-buffer checks generally.  A differing QR
    basis must be both close to the typed constructor and independently retain
    its scaled orthonormal geometry.  Hashes still authenticate the entire
    saved state; the constructor's rounding is never written into that state.
    """
    import torch
    from .coherent_velocity_diffusion import VPFPVelocityMacroConfig, VPFPVelocityMacroTransform
    from .hybrid_stress_diffusion import VPFPHybridStressJointMacroEDM

    if (key != "transform.packing" or type(model) is not VPFPHybridStressJointMacroEDM
            or type(model.transform) is not VPFPVelocityMacroTransform
            or type(model.config) is not VPFPVelocityMacroConfig
            or model.transform.config != model.config):
        return None
    columns = model.config.drive_degree + 2
    if (model.config.drive_degree not in (4, 8) or value.dtype != torch.float32
            or expected.dtype != torch.float32 or tuple(value.shape) != (33, columns)
            or tuple(expected.shape) != (33, columns)):
        raise ValueError("portable QR packing constructor geometry differs: " + key)
    stored = value.detach().cpu().double()
    constructed = expected.detach().cpu().double()
    if not bool(torch.isfinite(stored).all() and torch.isfinite(constructed).all()):
        raise ValueError("portable QR packing requires finite stored and constructed geometry: " + key)
    difference = stored - constructed
    max_abs = float(difference.abs().max())
    relative_frobenius = float(torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(constructed))
    identity = torch.eye(columns, dtype=torch.float64)
    stored_gram = float(((stored.T @ stored) * (columns / 33.) - identity).abs().max())
    constructed_gram = float(((constructed.T @ constructed) * (columns / 33.) - identity).abs().max())
    if (max_abs > PACKING_QR_ATOL or stored_gram > PACKING_QR_GRAM_ATOL
            or constructed_gram > PACKING_QR_GRAM_ATOL):
        raise ValueError("portable QR packing reconstruction differs: " + key
            + f" max_abs={max_abs:.17g} atol={PACKING_QR_ATOL:.17g}"
            + f" stored_gram={stored_gram:.17g} constructed_gram={constructed_gram:.17g}"
            + f" gram_atol={PACKING_QR_GRAM_ATOL:.17g}")
    return {"key": key, "shape": list(value.shape), "dtype": str(value.dtype),
        "constructor_class": _class(model.transform), "drive_degree": model.config.drive_degree,
        "constructor_bytes_equal": torch.equal(value.cpu(), expected.cpu()),
        "max_absolute_difference": max_abs, "relative_frobenius_difference": relative_frobenius,
        "absolute_tolerance": PACKING_QR_ATOL, "relative_tolerance": 0.,
        "stored_normalized_gram_max_error": stored_gram,
        "constructed_normalized_gram_max_error": constructed_gram,
        "normalized_gram_absolute_tolerance": PACKING_QR_GRAM_ATOL}


def _load_state(model, saved, *, deterministic=False):
    import torch
    if set(saved) != {"state", "state_sha256"} or _state_sha(saved["state"]) != saved["state_sha256"]:
        raise ValueError("portable model state fingerprint differs")
    expected, buffers = model.state_dict(), dict(model.named_buffers())
    if set(saved["state"]) != set(expected):
        raise ValueError("portable model state inventory differs")
    reconstruction_checks = []
    for key, value in saved["state"].items():
        if value.shape != expected[key].shape or value.dtype != expected[key].dtype:
            raise ValueError("portable tensor/immutable buffer differs: " + key)
        if deterministic or key in buffers:
            check = (_packing_reconstruction_check(model, key, value, expected[key])
                if not deterministic and key == "transform.packing" else None)
            if check is not None:
                reconstruction_checks.append(check)
            elif not torch.equal(value.cpu(), expected[key].cpu()):
                raise ValueError("portable tensor/immutable buffer differs: " + key)
    model.load_state_dict(deepcopy(saved["state"]), strict=True)
    if _state_sha(model.state_dict()) != saved["state_sha256"]:
        raise ValueError("portable strict load did not preserve all original tensor bytes")
    for check in reconstruction_checks:
        check["saved_bytes_restored"] = True
    return reconstruction_checks


def load_portable_base(path, sha, device="cuda", *, runtime_interface_profile=None, migration_evidence=None):
    """Reconstruct exact original inference classes without historical loaders."""
    import numpy as np  # noqa: F401
    import torch
    from vpfp_weak_migration_paths import resolve_asset_path
    path = resolve_asset_path(path)
    if path.is_symlink() or not path.is_file() or _file_sha(path) != sha:
        raise ValueError("portable Base asset SHA mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != PAYLOAD_KEYS or payload["schema"] != SCHEMA or payload["classes"] != CLASSES:
        raise ValueError("exact typed portable Base schema/classes required")
    _origin(payload["origin_receipt"], payload["provenance"])
    if migration_evidence is not None:
        if runtime_interface_profile is not None:
            raise ValueError("PDE-GAV migration proof and historical runtime profiles are distinct")
        from .weak_pde_gav_binding import validate_pde_gav_provenance
        checks = validate_pde_gav_provenance(payload["provenance"], migration_evidence)
    else:
        checks = (_validate_provenance(payload["provenance"]) if runtime_interface_profile is None else
            _validate_provenance(payload["provenance"], runtime_interface_profile=runtime_interface_profile))
    expected_keys = {"sampling": {"steps", "identity", "normalizer_sha256", "coordinate_transform_sha256"}, "macro": {"config", "state", "state_sha256"},
        "micro": {"config", "macro_config", "spatial_domain_lengths", "probe_arm", "replay_binding", "identity", "state", "state_sha256"},
        "normalizers": {"macro_prior", "micro", "micro_macro"},
        "decoder": {"base_payload", "macro_config", "value_config", "identity", "state", "state_sha256"}}
    if any(set(payload[name]) != keys for name, keys in expected_keys.items()):
        raise ValueError("portable constructor inventory differs")
    if payload["sampling"]["steps"] != 64 or payload["micro"]["probe_arm"] != "stratified32":
        raise ValueError("portable sampler steps/replay arm differs")
    from .hybrid_stress_diffusion import VPFPHybridStressJointMacroEDM, VPFPHybridStressMacroNormalizer
    from .coherent_velocity_diffusion import VPFPVelocityMacroConfig
    from .coherent_diffusion import VPFPCoherentMacroConfig, VPFPCoherentMacroNormalizer
    from .innovation_diffusion import VPFPInnovationNormalizer
    from .value_control_diffusion import VPFPValueEDMConfig
    from .temporal_innovation import VPFPTemporalInnovationTransform, SCHEMA as TEMPORAL
    from .bulk_transport_innovation import VPFPBulkTransportInnovationTransform, SCHEMA as BULK
    from .moving_frame_binding import build_moving_frame_decoder
    from .innovation_binding import InnovationMovingFrameSampler
    script_root = str(ROOT / "scripts")
    if script_root not in sys.path: sys.path.insert(0, script_root)
    from run_vpfp_weak_replay_probe_screen import ReplayEDM
    old_dtype = torch.get_default_dtype()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    reconstruction_checks = []
    try:
        torch.set_default_dtype(torch.float32)
        with torch.random.fork_rng(devices=devices):
            macro = VPFPHybridStressJointMacroEDM(VPFPVelocityMacroConfig.from_dict(payload["macro"]["config"]))
            row = payload["micro"]
            micro = ReplayEDM(VPFPValueEDMConfig.from_dict(row["config"]), VPFPCoherentMacroConfig.from_dict(row["macro_config"]),
                spatial_domain_lengths=row["spatial_domain_lengths"], replay=_InferenceReplayContext(row["replay_binding"]),
                probe_arm=row["probe_arm"])
            for key, model in (("macro", macro), ("micro", micro)):
                reconstruction_checks.extend(dict(check, model=key) for check in
                    _load_state(model, {k: payload[key][k] for k in ("state", "state_sha256")}))
                model.to(device).eval().requires_grad_(False)
            if micro.source_step != 0 or _json(micro.identity()) != _json(row["identity"]):
                raise ValueError("portable micro clock/identity changed")
            normal = payload["normalizers"]
            prior_norm = VPFPHybridStressMacroNormalizer.from_payload(normal["macro_prior"]).to(device)
            micro_norm = VPFPInnovationNormalizer.from_payload(normal["micro"]).to(device)
            macro_norm = VPFPCoherentMacroNormalizer.from_payload(normal["micro_macro"]).to(device)
            coordinate = payload["coordinate_transform"]
            transforms = {TEMPORAL: VPFPTemporalInnovationTransform, BULK: VPFPBulkTransportInnovationTransform}
            if coordinate.get("schema") not in transforms: raise ValueError("unknown complete innovation coordinates")
            transform = transforms[coordinate["schema"]].from_payload(coordinate).to(device)
            for norm, key in ((prior_norm, "macro_prior"), (micro_norm, "micro"), (macro_norm, "micro_macro")):
                norm.eval().requires_grad_(False)
                if _typed_plain(norm.to_payload()) != normal[key]:
                    raise ValueError("portable normalization payload changed during reconstruction")
            transform.eval().requires_grad_(False)
            decoder = build_moving_frame_decoder(payload["decoder"]["base_payload"], micro.macro_config, micro.config, device)
            _load_state(decoder, {k: payload["decoder"][k] for k in ("state", "state_sha256")}, deterministic=True)
            decoder.eval().requires_grad_(False)
            if (macro.config.base_macro_config != micro.macro_config or prior_norm.config != macro.config
                    or micro_norm.config != micro.config or macro_norm.config != micro.macro_config
                    or transform.config.value_config != micro.config
                    or _json(decoder.identity()) != _json(payload["decoder"]["identity"])
                    or _json(decoder.macro_config.to_dict()) != _json(payload["decoder"]["macro_config"])
                    or _json(decoder.value_config.to_dict()) != _json(payload["decoder"]["value_config"])):
                raise ValueError("portable hierarchy/decoder constructor geometry differs")
            sampler = InnovationMovingFrameSampler(macro, prior_norm, micro, micro_norm, macro_norm, transform,
                identity=payload["sampling"]["identity"], steps=64)
            if (list(sampler._normalizers) != payload["sampling"]["normalizer_sha256"]
                    or sampler._transform_sha != payload["sampling"]["coordinate_transform_sha256"]):
                raise ValueError("portable normalization/coordinate fingerprints differ")
    finally:
        torch.set_default_dtype(old_dtype)
    receipt = deepcopy(payload["origin_receipt"])
    if (_class(sampler) != CLASSES["sampler"] or _class(macro) != CLASSES["macro"] or _class(micro) != CLASSES["micro"]
            or _class(decoder) != CLASSES["decoder"] or _json(sampler.identity) != _json(receipt["sampler_identity"])
            or _state_sha(macro.state_dict()) != receipt["macro_state_sha256"]
            or _state_sha(micro.state_dict()) != receipt["model_state_sha256"]
            or sha256(_json(sampler.identity).encode()).hexdigest() != receipt["sampler_identity_sha256"]
            or _file_sha(path) != sha):
        raise ValueError("portable Base changed the accepted native identity/state")
    sampler.assert_frozen()
    receipt["portable_base"] = dict(path=str(path.resolve()), sha256=sha, schema=SCHEMA,
        provenance_checks=checks, original_sampling_identity_preserved=True,
        original_model_classes_preserved=True, original_states_exact=True,
        qr_buffer_reconstruction_checks=reconstruction_checks,
        inference_only=True, replay_teacher_arrays_loaded=False, historical_training_loader_called=False,
        pde_gav_inference_wrapper=migration_evidence is not None,
        numerical_equivalence_in_new_runtime_qualified=False if migration_evidence is not None else None)
    return sampler, decoder, receipt
