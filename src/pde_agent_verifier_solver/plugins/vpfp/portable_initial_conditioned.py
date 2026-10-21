"""Explicit frozen-buffer codec for a separately identified strong candidate.

This does not repair, relax, or register the legacy checkpoint loader. The
original weights and original normalization payload remain immutable lineage.
Only a new candidate's fifteen FP64 buffers are hydrated, without eig/eigvalsh.
File/content integrity is not evidence of old/new sample equivalence.

Runtime guards use tensor identity, storage descriptor and mutation version;
full byte checks run at export/load/device migration and on explicit request.
As with PyTorch's version mechanism, deliberately writing through a raw storage
pointer or ``tensor.data`` in place is unsupported; assert_frozen(full=True)
detects such writes before exporting. No tensor copies occur per denoiser step.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import math
from pathlib import Path

import torch
from torch import nn

from . import initial_conditioned_diffusion as original
from .known_initial_prior import KnownInitialPriorSpec, _json_sha
from .noise_geometry import VPFPTemporalCovarianceTransform, SCHEMA as TRANSFORM_SCHEMA
from .structured_diffusion import VPFPCoefficientNormalizer, _seeds, _sha
from .temporal_covariance_diffusion import VPFPTemporalCovarianceNormalizer, NORMALIZER_SCHEMA as TEMPORAL_SCHEMA
from .value_control_diffusion import (VPFPValueControlNormalizer, LIFT_SCHEMA,
    CHECKPOINT_SCHEMA as VALUE_SCHEMA, _sampler_from_value_payload)

SCHEMA = "vpfp_portable_initial_conditioned_frozen_buffers_v1"
SAMPLER_SCHEMA = "vpfp_explicit_portable_initial_conditioned_sampler_v1"
HASH_SCHEMA = "named_tensor_dtype_shape_little_endian_bytes_sha256_v1"
_SCHUR = ("regression", "schur", "eigenvalues", "square_root", "inverse_square_root")
_GEOMETRY = ("second_moment", "raw_second_moment", "kinetic_dc_mean", "eigenvalues",
    "regularized_eigenvalues", "eigenvectors", "eigenvalue_floor", "square_root", "inverse_square_root")
_FIELDS = {"schema", "hash_schema", "source_checkpoint_sha256", "model_state_sha256",
    "original_normalizer", "original_normalizer_sha256", "candidate_normalizer",
    "candidate_normalizer_sha256", "buffers", "buffer_manifest", "buffer_state_sha256",
    "identity", "sha256"}


def tensor_state_manifest(state):
    """Canonical byte hashes, independent of torch.save container encoding."""
    if not isinstance(state, Mapping) or not state or any(not isinstance(k, str) for k in state):
        raise ValueError("tensor state must be a nonempty named mapping")
    result = {}
    for name in sorted(state):
        value = state[name]
        if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                or value.dtype not in (torch.float32, torch.float64) or value.requires_grad
                or not torch.isfinite(value).all()):
            raise ValueError(f"invalid frozen tensor: {name}")
        array = value.detach().cpu().contiguous().numpy()
        raw = array.astype(array.dtype.newbyteorder("<"), copy=False).tobytes(order="C")
        result[name] = {"dtype": str(value.dtype), "shape": list(value.shape), "sha256": sha256(raw).hexdigest()}
    return result


def tensor_state_sha256(state):
    return _json_sha({"schema": HASH_SCHEMA, "tensors": tensor_state_manifest(state)})


def _payload_digest(payload):
    # Buffer contents are authenticated through the verified manifest. Never
    # pickle a Python object merely to derive its identity.
    return _json_sha({key: value for key, value in payload.items() if key not in ("buffers", "sha256")})


def _scope(candidate):
    if (not isinstance(candidate, dict) or set(candidate) !=
            {"schema", "config", "temporal_normalizer", "initial_prior", "fit_metadata"}
            or candidate["schema"] != original.NORMALIZER_SCHEMA):
        raise ValueError("unexpected original/candidate normalizer schema or fields")
    config = original.VPFPInitialConditionedEDMConfig.from_dict(candidate["config"])
    temporal = candidate["temporal_normalizer"]
    if (not isinstance(temporal, dict) or set(temporal) !=
            {"schema", "config", "value_normalizer", "temporal_transform", "fit_metadata"}
            or temporal["schema"] != TEMPORAL_SCHEMA):
        raise ValueError("unexpected temporal normalization fields")
    meta = temporal["fit_metadata"]
    if (meta.get("rho_policy") != "identity" or meta.get("covariance_strength", 1.) != 1.
            or meta.get("covariance_recipe", "mixture") != "mixture"
            or temporal["temporal_transform"]["fit_metadata"].get("center_kinetic_dc") is not True):
        raise ValueError("portable v1 only supports strong identity rho, centered strength-one mixture")
    if config.collision_nu_range[0] < 10.:
        raise ValueError("portable v1 is scoped to the strong-collision codec")
    return config


def _expected_shapes(config):
    g, t, h = 1 + config.hermite_channels, config.full_config.control_count, config.hermite_channels
    result = {"regression": (g, t-1), "eigenvalues": (g, t-1)}
    result.update({key: (g, t-1, t-1) for key in ("schur", "square_root", "inverse_square_root")})
    result["temporal.parent.group_scales"] = (g,)
    for name in _GEOMETRY:
        shape = ((h, t) if name == "kinetic_dc_mean" else (g,) if name == "eigenvalue_floor"
                 else (g, t) if name in ("eigenvalues", "regularized_eigenvalues") else (g, t, t))
        result["temporal.temporal." + name] = shape
    return result


def _same_bits(left, right):
    return (left.shape == right.shape and left.dtype == right.dtype and
            torch.equal(left.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                        right.detach().cpu().contiguous().reshape(-1).view(torch.uint8)))


def _check_parent(parent):
    """The legacy value loader's complete train-only normalization lineage."""
    meta, config = parent.fit_metadata, parent.config
    ids = _seeds(meta.get("sample_ids", ()))
    if (len(ids) < 2 or len(set(ids)) != len(ids)
            or meta.get("schema") != "vpfp_train_value_group_rms_normalization_v1"
            or meta.get("centering") != "none" or meta.get("inactive_policy") != "exact_zero_after_lift"
            or meta.get("slope_lift") != LIFT_SCHEMA or meta.get("delta_tau") != config.slope_interval
            or meta.get("slope_subspace_verified") is not True or meta.get("roundoff_tolerance_ulps") != 8):
        raise ValueError("value normalization provenance/layout invalid")
    payload = meta.get("parent_normalizer_payload")
    if not isinstance(payload, dict) or set(payload) != {"config", "group_scales", "fit_metadata"}:
        raise ValueError("value normalization needs its exact bound parent payload")
    full = VPFPCoefficientNormalizer.from_payload(payload)
    fm = full.fit_metadata
    if (full.config != config.full_config or not _same_bits(full.group_scales[:, 0], parent.group_scales)
            or meta.get("floor_policy") != "inherited_full_representation_for_paired_control"
            or meta.get("scale") != "value_group_rms_with_inherited_full_block_relative_floor"
            or fm.get("schema") != "vpfp_train_group_rms_normalization_v1" or fm.get("centering") != "none"
            or fm.get("scale") != "group_rms_with_block_relative_floor" or fm.get("inactive_policy") != "exact_zero"
            or fm.get("slope_transform") != "d_d_tau_times_delta_tau" or fm.get("delta_tau") != config.slope_interval
            or fm.get("sample_ids") != list(ids)
            or any(meta.get(k) != fm.get(k) for k in ("relative_floor", "absolute_floor", "rho_floor", "kinetic_floor"))):
        raise ValueError("value normalization differs from its paired full-state parent")


def _matrix_close(actual, expected, *, label, tolerance):
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError(f"nonfinite frozen {label}")
    scale = torch.maximum(actual.abs().amax((-2, -1)), expected.abs().amax((-2, -1))).clamp_min(torch.finfo(torch.float64).tiny)
    if torch.any((actual-expected).abs().amax((-2, -1)) > tolerance*scale):
        raise ValueError(f"inconsistent frozen {label}")


def _validate_factors(transform, state):
    """No spectral solve: verify saved factors against their saved statistics.

    These are construction checks, NOT a sample-equivalence tolerance. Temporal
    inverse closure retains the old constructor's absolute 1e-7 limit. Other
    matrix identities use 128*n*eps64 times the actual product magnitude, to
    accommodate normal FP64 reduction differences without changing any buffer.
    """
    t = transform.config.control_count
    eps_budget = 128*t*torch.finfo(torch.float64).eps
    c, raw, v = transform.second_moment, transform.raw_second_moment, transform.eigenvectors
    eig, floored = transform.eigenvalues, transform.regularized_eigenvalues
    for name, value in (("second moment", c), ("raw moment", raw)):
        _matrix_close(value, value.transpose(-1, -2), label=name+" symmetry", tolerance=1.e-10)
    scale = c.abs().amax((-2, -1)).clamp_min(torch.finfo(torch.float64).tiny)
    if torch.any(eig[:, 0] < -1.e-10*scale) or torch.any(eig[:, 1:] < eig[:, :-1]):
        raise ValueError("invalid sorted PSD temporal eigenvalues")
    meta = transform.fit_metadata
    floor = (meta["relative_floor"]**2*c.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(0.)).clamp_min(meta["absolute_floor"]**2)
    if not torch.allclose(floor, transform.eigenvalue_floor, rtol=eps_budget, atol=0.):
        raise ValueError("temporal eigenvalue floor differs")
    if not torch.allclose(torch.maximum(eig, floor[:, None]), floored, rtol=eps_budget, atol=0.) or torch.any(floored <= 0):
        raise ValueError("regularized temporal eigenvalues differ")
    eye = torch.eye(t, dtype=torch.float64).expand_as(c)
    _matrix_close(v@v.transpose(-1, -2), eye, label="temporal orthogonality", tolerance=eps_budget)
    _matrix_close((v*eig[:, None, :])@v.transpose(-1, -2), c, label="temporal eigensystem", tolerance=eps_budget)
    root, inv = transform.square_root, transform.inverse_square_root
    _matrix_close(root, (v*floored.sqrt()[:, None, :])@v.transpose(-1, -2), label="temporal root", tolerance=eps_budget)
    _matrix_close(inv, (v*floored.rsqrt()[:, None, :])@v.transpose(-1, -2), label="temporal inverse", tolerance=eps_budget)
    closure = root@inv-eye
    if not torch.isfinite(closure).all() or float(closure.abs().max()) > 1.e-7:
        raise ValueError("temporal inverse closure exceeds original bound")
    # rho is exactly identity in this version, not an empirical rho factor.
    scoped = torch.cat((eye[:1], root[1:]))
    cov = scoped@scoped.transpose(-1, -2)
    regression = cov[:, 1:, 0]/cov[:, 0, 0, None]
    if not torch.allclose(state["regression"], regression, rtol=eps_budget, atol=eps_budget*float(cov.abs().max())):
        raise ValueError("Schur regression differs from frozen temporal root")
    schur = cov[:, 1:, 1:]-regression[:, :, None]*cov[:, None, 0, 1:]
    schur = (schur+schur.transpose(-1, -2))/2
    _matrix_close(state["schur"], schur, label="Schur covariance", tolerance=eps_budget)
    r, inv = state["square_root"], state["inverse_square_root"]
    _matrix_close(r, r.transpose(-1, -2), label="Schur root symmetry", tolerance=eps_budget)
    if torch.any(torch.linalg.cholesky_ex(r, check_errors=False).info):
        raise ValueError("frozen Schur root must be positive definite")
    _matrix_close(r@r, schur, label="Schur root square", tolerance=eps_budget)
    eye = torch.eye(t-1, dtype=torch.float64).expand_as(schur)
    # Product-scaled gamma bound, separate from the old/new output replay test.
    magnitude = (r.abs()@inv.abs()).amax((-2, -1)).clamp_min(1.)
    closure = r@inv-eye
    if (not torch.isfinite(magnitude).all() or not torch.isfinite(closure).all()
            or torch.any(closure.abs().amax((-2, -1)) > eps_budget*magnitude)):
        raise ValueError("Schur inverse closure differs")
    eig = state["eigenvalues"]
    if torch.any(eig <= 0) or torch.any(eig[:, 1:] < eig[:, :-1]):
        raise ValueError("Schur eigenvalues must be sorted positive")
    if not torch.allclose(eig.sum(-1), schur.diagonal(dim1=-2, dim2=-1).sum(-1), rtol=eps_budget, atol=0.):
        raise ValueError("Schur spectrum trace differs")


def _hydrate(candidate, state):
    config = _scope(candidate)
    shapes = _expected_shapes(config)
    if not isinstance(state, Mapping) or set(state) != set(shapes):
        raise ValueError("portable v1 requires exactly fifteen named buffers")
    for name, shape in shapes.items():
        value = state[name]
        if (not isinstance(value, torch.Tensor) or value.layout != torch.strided or value.dtype != torch.float64
                or tuple(value.shape) != shape or value.requires_grad or not torch.isfinite(value).all()):
            raise ValueError(f"invalid frozen buffer: {name}")
    state = {k: v.detach().cpu().contiguous().clone() for k, v in state.items()}
    temporal_payload = candidate["temporal_normalizer"]
    parent = VPFPValueControlNormalizer.from_payload(temporal_payload["value_normalizer"])
    _check_parent(parent)
    if not _same_bits(state["temporal.parent.group_scales"], parent.group_scales):
        raise ValueError("frozen parent scales differ from source payload")
    payload = temporal_payload["temporal_transform"]
    if (set(payload) != {"schema", "config", "second_moment", "raw_second_moment", "kinetic_dc_mean", "fit_metadata"}
            or payload["schema"] != TRANSFORM_SCHEMA or payload["config"] != config.value_config.to_dict()):
        raise ValueError("invalid frozen temporal transform fields/config")
    meta = payload["fit_metadata"]
    ids = _seeds(meta.get("sample_ids", ()))
    if (meta.get("split") != "train" or meta.get("generated_samples_used_for_fit") is not False
            or meta.get("reference_samples_used_for_fit") is not False or meta.get("center_kinetic_dc") is not True
            or len(ids) < 2 or len(set(ids)) != len(ids)):
        raise ValueError("temporal transform requires distinct training-only identities")
    relative, absolute = meta.get("relative_floor"), meta.get("absolute_floor")
    if (isinstance(relative, bool) or isinstance(absolute, bool) or not isinstance(relative, (float, int))
            or not isinstance(absolute, (float, int)) or not math.isfinite(relative) or not 0 < relative <= 1
            or not math.isfinite(absolute) or absolute <= 0):
        raise ValueError("invalid temporal floor metadata")
    transform = VPFPTemporalCovarianceTransform.__new__(VPFPTemporalCovarianceTransform)
    nn.Module.__init__(transform)
    transform.config, transform.fit_metadata = config.value_config, deepcopy(meta)
    for name in _GEOMETRY:
        value = state["temporal.temporal."+name]
        if name in ("second_moment", "raw_second_moment", "kinetic_dc_mean"):
            if not _same_bits(value, torch.tensor(payload[name], dtype=torch.float64)):
                raise ValueError(f"frozen {name} differs from source payload")
        transform.register_buffer(name, value)
    _validate_factors(transform, state)
    # The default constructor validates full parent/transform lineage but does
    # not reconstruct roots for this strictly limited recipe.
    temporal = VPFPTemporalCovarianceNormalizer(parent, transform, rho_policy="identity",
        covariance_strength=1., covariance_recipe="mixture", fit_metadata=temporal_payload["fit_metadata"])
    if temporal.to_payload() != temporal_payload or temporal.config != config.value_config:
        raise ValueError("temporal payload/config differs from frozen components")
    prior = KnownInitialPriorSpec.from_payload(candidate["initial_prior"])
    prior.check_config(config)
    obj = PortableInitialConditionedNormalizer.__new__(PortableInitialConditionedNormalizer)
    nn.Module.__init__(obj)
    obj.config, obj.temporal, obj.initial_prior = config, temporal, prior
    for name in _SCHUR:
        obj.register_buffer(name, state[name])
    expected = {"schema": original.NORMALIZER_SCHEMA, "split": "train", "sample_ids": list(temporal.fit_metadata["sample_ids"]),
        "generated_samples_used_for_fit": False, "reference_samples_used_for_fit": False,
        "temporal_normalizer_sha256": _json_sha(temporal.to_payload()), "known_initial_prior_sha256": prior.sha256,
        "schur_operator_sha256": obj._operator_sha(), "context": "clean_parent_RMS_t0_values_before_temporal_whitening",
        "context_slopes_or_future_values": False, "loss_target": "future_white_values_only",
        "context_dimension": config.context_dimension, "future_dimension": config.latent_dimension,
        "full_decoder_dimension": config.full_dimension, "slope_lift": LIFT_SCHEMA,
        "future_times": torch.linspace(*config.time_domain, config.full_config.control_count, dtype=torch.float64)[1:].tolist(),
        "additional_schur_eigenvalue_floor": False, "future_noise_full_rank": True,
        "conditional_gaussian_target_assumed": False, "physical_answer_fidelity_certified": False}
    if candidate["fit_metadata"] != expected:
        raise ValueError("conditional candidate metadata or Schur binding differs")
    obj.fit_metadata, obj._metadata_sha = deepcopy(expected), _json_sha(expected)
    return obj


def _identity(payload):
    return {"schema": SAMPLER_SCHEMA, "source_checkpoint_sha256": payload["source_checkpoint_sha256"],
        "model_state_sha256": payload["model_state_sha256"],
        "original_normalizer_sha256": payload["original_normalizer_sha256"],
        "candidate_normalizer_sha256": payload["candidate_normalizer_sha256"],
        "buffer_state_sha256": payload["buffer_state_sha256"],
        "old_schur_operator_sha256": payload["original_normalizer"]["fit_metadata"]["schur_operator_sha256"],
        "candidate_schur_operator_sha256": payload["candidate_normalizer"]["fit_metadata"]["schur_operator_sha256"],
        "runtime_eigendecomposition": False, "old_sampler_equivalence_certified": False,
        "scientific_admission": False}


def export_frozen_normalizer(original_payload, candidate_payload, state_dict, *, source_checkpoint_sha256, model_state_sha256):
    """Convert the explicitly approved probe fields; no fitting or weight edits.

    The caller supplies SHA-verified original checkpoint payloads and its model
    state digest. Only the candidate Schur hash may differ from old metadata.
    Export validation alone grants neither replay equivalence nor deployment.
    """
    old, candidate = deepcopy(original_payload), deepcopy(candidate_payload)
    _scope(old)
    _scope(candidate)
    _sha(old["fit_metadata"]["schur_operator_sha256"], "old Schur operator")
    comparable = deepcopy(candidate)
    comparable["fit_metadata"]["schur_operator_sha256"] = old["fit_metadata"]["schur_operator_sha256"]
    if comparable != old:
        raise ValueError("only the explicitly identified Schur operator hash may differ")
    obj = _hydrate(candidate, state_dict)
    buffers = {name: value.detach().cpu().contiguous().clone() for name, value in obj.named_buffers()}
    manifest = tensor_state_manifest(buffers)
    payload = {"schema": SCHEMA, "hash_schema": HASH_SCHEMA,
        "source_checkpoint_sha256": _sha(source_checkpoint_sha256, "source checkpoint"),
        "model_state_sha256": _sha(model_state_sha256, "original model state"),
        "original_normalizer": old, "original_normalizer_sha256": _json_sha(old),
        "candidate_normalizer": candidate, "candidate_normalizer_sha256": _json_sha(candidate),
        "buffers": buffers, "buffer_manifest": manifest,
        "buffer_state_sha256": _json_sha({"schema": HASH_SCHEMA, "tensors": manifest})}
    payload["identity"] = _identity(payload)
    payload["sha256"] = _payload_digest(payload)
    return payload


class PortableInitialConditionedNormalizer(original.VPFPInitialConditionedNormalizer):
    """Original arithmetic methods; explicit, immutable new asset identity."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use load_frozen_normalizer with an explicit expected content SHA")

    def _metadata(self):
        return {"config": self.config.to_dict(), "fit_metadata": self.fit_metadata,
            "temporal_config": self.temporal.config.to_dict(), "temporal_fit": self.temporal.fit_metadata,
            "parent_config": self.temporal.parent.config.to_dict(), "parent_fit": self.temporal.parent.fit_metadata,
            "transform_config": self.temporal.temporal.config.to_dict(), "transform_fit": self.temporal.temporal.fit_metadata,
            "prior": self.initial_prior.to_payload(), "rho_policy": self.temporal.rho_policy,
            "strength": self.temporal.covariance_strength, "recipe": self.temporal.covariance_recipe}

    @staticmethod
    def _descriptor(value):
        return (value._version, value.data_ptr(), value.storage_offset(), value.shape, value.stride(), value.dtype, value.device)

    def _seal(self):
        self._references = {name: (value, self._descriptor(value)) for name, value in self.named_buffers()}
        self._module_references = {name: module for name, module in self.named_modules()}
        self._frozen_versions = {name: value._version for name, value in self.named_buffers()}
        self.temporal._frozen_buffer_versions = {name: value._version for name, value in self.temporal.named_buffers()}
        self._all_metadata_sha = _json_sha(self._metadata())

    def assert_frozen(self, *, full=False):
        current = dict(self.named_buffers())
        modules = dict(self.named_modules())
        if (set(current) != set(self._references) or set(modules) != set(self._module_references)
                or any(modules[k] is not v for k, v in self._module_references.items())):
            raise ValueError("portable frozen module/buffer identities changed")
        for name, value in current.items():
            saved, descriptor = self._references[name]
            if (value is not saved or self._descriptor(value) != descriptor
                    or value.requires_grad or value.dtype != torch.float64):
                raise ValueError(f"portable frozen buffer identity/version changed: {name}")

    def _check(self):
        self.assert_frozen()
        super()._check()

    def _apply(self, fn, recurse=True):
        self.assert_frozen(full=True)
        super()._apply(fn, recurse=recurse)
        if (any(value.dtype != torch.float64 for value in self.buffers())
                or tensor_state_manifest(dict(self.named_buffers())) != self._header["buffer_manifest"]):
            raise ValueError("portable device migration changed frozen FP64 bytes")
        self._seal()
        return self

    def to_payload(self):
        """Small JSON binding for the inherited sampler/guidance hash contract."""
        self.assert_frozen(full=True)
        return {"schema": SCHEMA, "sha256": self._header["sha256"],
            "identity": deepcopy(self._header["identity"]), "config": self.config.to_dict(),
            "buffer_manifest": deepcopy(self._header["buffer_manifest"])}

    def to_frozen_payload(self):
        """Complete weights_only-safe tensor bundle for an explicit export."""
        self.assert_frozen(full=True)
        return dict(deepcopy(self._header), buffers={k: v.detach().cpu().contiguous().clone() for k, v in self.named_buffers()})

    @classmethod
    def from_frozen_payload(cls, payload, *, expected_sha256, device="cpu"):
        return load_frozen_normalizer(payload, expected_sha256=expected_sha256, device=device)

    @classmethod
    def from_payload(cls, payload):
        raise ValueError("compact portable binding is not a loadable bundle; use from_frozen_payload")

    def __deepcopy__(self, memo):
        result = load_frozen_normalizer(self.to_frozen_payload(), expected_sha256=self._header["sha256"],
            device=next(self.buffers()).device)
        memo[id(self)] = result
        return result

    def report(self):
        self.assert_frozen()
        return {"schema": SCHEMA, "identity": deepcopy(self._header["identity"]),
            "content_sha256": self._header["sha256"], "frozen_buffer_count": len(self._references),
            "random_dimension": self.config.latent_dimension, "full_decoder_dimension": self.config.full_dimension,
            "context_dimension": self.config.context_dimension, "scientific_admission": False}


def load_frozen_normalizer(payload, *, expected_sha256, device="cpu"):
    if (not isinstance(payload, dict) or set(payload) != _FIELDS
            or payload["schema"] != SCHEMA or payload["hash_schema"] != HASH_SCHEMA):
        raise ValueError("invalid portable codec schema/fields")
    # Re-run every metadata/construction check, not only self-reported hashes.
    validated = export_frozen_normalizer(payload["original_normalizer"], payload["candidate_normalizer"], payload["buffers"],
        source_checkpoint_sha256=payload["source_checkpoint_sha256"], model_state_sha256=payload["model_state_sha256"])
    obj = _hydrate(payload["candidate_normalizer"], payload["buffers"])
    obj._header = deepcopy({key: value for key, value in payload.items() if key != "buffers"})
    obj._seal()
    return obj.to(device=device)


def _read_weights_only(path, expected_sha256):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"portable input must be a regular nonsymlink file: {path}")
    raw = path.read_bytes()
    return torch.load(BytesIO(raw), weights_only=True, map_location="cpu")


@contextmanager
def _original_fp32_construction():
    """Construct the frozen FP32 graph without leaking ambient dtype changes.

    This explicit loader is a synchronous asset-loading operation. It must not
    run concurrently with another thread changing torch's process defaults.
    """
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        with torch.device("cpu"):
            yield
    finally:
        torch.set_default_dtype(previous)


def load_portable_initial_conditioned_sampler(original_checkpoint, *, checkpoint_sha256,
        portable_bundle_path, portable_bundle_sha256, device="cpu", sample_steps=64, sample_churn=0.):
    """Explicit two-file factory. No dispatcher registration or legacy fallback."""
    payload = _read_weights_only(original_checkpoint, checkpoint_sha256)
    bundle = _read_weights_only(portable_bundle_path, portable_bundle_sha256)
    fields = {"schema", "config", "model", "normalizer", "base_decoder_sha256", "corpus_manifest_sha256", "training"}
    if not isinstance(payload, dict) or set(payload) != fields or payload["schema"] != original.CHECKPOINT_SCHEMA:
        raise ValueError("unexpected original initial-conditioned checkpoint schema/fields")
    normalizer = load_frozen_normalizer(bundle, expected_sha256=bundle["sha256"])
    if bundle['original_normalizer'] != payload['normalizer']:
        raise ValueError("portable codec is bound to a different original checkpoint/weights/normalizer")
    config = original.VPFPInitialConditionedEDMConfig.from_dict(payload["config"])
    if config != normalizer.config:
        raise ValueError("portable model and normalizer configs differ")
    with _original_fp32_construction():
        model = original.VPFPInitialConditionedEDM(config)
    state, expected = payload["model"], model.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError("conditional model tensor keys differ")
    for name, target in expected.items():
        value = state[name]
        if (not isinstance(value, torch.Tensor) or value.layout != torch.strided or value.dtype != target.dtype
                or value.shape != target.shape or not torch.isfinite(value).all()):
            raise ValueError(f"invalid conditional model tensor: {name}")
        if name.startswith("transform.") and not _same_bits(value, target):
            raise ValueError("checkpoint changed immutable conditional Fourier geometry")
    view = dict(payload, schema=VALUE_SCHEMA, config=config.value_config.to_dict(), normalizer=normalizer.temporal.parent.to_payload(),
        model={k: v for k, v in state.items() if not k.startswith("initial_input.")})
    with _original_fp32_construction():
        checked = _sampler_from_value_payload(view, checkpoint_sha256, "cpu", steps=sample_steps, churn=sample_churn)
    prior, training = normalizer.initial_prior.to_payload(), payload["training"]
    if payload["corpus_manifest_sha256"] != prior["corpus_manifest_sha256"]:
        raise ValueError("conditional checkpoint and initial prior corpus identity differ")
    if "train_case_ids" in training and training["train_case_ids"] != [
            f"{prior['model_spec']['regime_id']}:{seed}" for seed in normalizer.fit_metadata["sample_ids"]]:
        raise ValueError("conditional training case IDs differ")
    model.load_state_dict(state, strict=True)
    if any(not _same_bits(value, state[name]) for name, value in model.state_dict().items()):
        raise ValueError("strict model load did not preserve original tensor bytes")
    model.to(device=device).eval().requires_grad_(False)
    normalizer.to(device=device)
    sampler = original.VPFPInitialConditionedBatchSampler(model, normalizer,
        base_decoder_sha256=checked.identity["base_decoder_sha256"], checkpoint_sha256=checkpoint_sha256,
        steps=sample_steps, churn=sample_churn)
    sampler.identity.update(checkpoint_schema=SAMPLER_SCHEMA, portable_codec=deepcopy(bundle["identity"]),
        portable_bundle_sha256=portable_bundle_sha256, portable_content_sha256=bundle["sha256"], training=deepcopy(training))
    return sampler
