"""EDM for future controls conditional on a clean, full spatial initial field.

Only future white coordinates are random reverse states or denoising targets.
The fixed Schur map is a coordinate preconditioner, not a Gaussian assumption
about the physical conditional law. The original 33 value times and final
value-to-slope lift are unchanged. No nonlinear physical-loss claim is made.
"""
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import math
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .known_initial_prior import KnownInitialPriorSpec, KnownInitialPrior, _json_sha
from .structured_diffusion import VPFPStructuredEDMConfig, VPFPStructuredEDM, VPFPDiffusionBatchSampler, _coefficients, _seeds, _sha
from .value_control_diffusion import (VPFPValueEDMConfig, VPFPValueControlEDM, _value_split,
    _sampler_from_value_payload, CHECKPOINT_SCHEMA as VALUE_SCHEMA, LIFT_SCHEMA)
from .temporal_covariance_diffusion import VPFPTemporalCovarianceNormalizer

CHECKPOINT_SCHEMA = "vpfp_initial_conditioned_future_diffusion_checkpoint_v1"
NORMALIZER_SCHEMA = "vpfp_clean_initial_future_schur_normalizer_v1"


@dataclass(frozen=True)
class VPFPInitialConditionedEDMConfig:
    full_config: VPFPStructuredEDMConfig
    schema: str = "vpfp_clean_initial_conditioned_future_edm_v1"

    def __post_init__(self):
        if (not isinstance(self.full_config, VPFPStructuredEDMConfig)
                or self.full_config.control_count < 3 or self.schema != "vpfp_clean_initial_conditioned_future_edm_v1"):
            raise ValueError("conditional EDM requires its registered full time-control config")

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "full_config"), name)

    @property
    def value_config(self):
        return VPFPValueEDMConfig(self.full_config)

    @property
    def control_count(self):
        return self.full_config.control_count - 1

    @property
    def context_dimension(self):
        return self.rho_modes + self.hermite_channels * self.kinetic_modes

    @property
    def full_dimension(self):
        return self.full_config.latent_dimension

    @property
    def rho_dimension(self):
        return self.control_count * self.rho_modes

    @property
    def latent_dimension(self):
        return self.context_dimension * self.control_count

    @property
    def kinetic_dimension(self):
        return self.latent_dimension - self.rho_dimension

    @property
    def inactive_dim(self):
        return 0

    @property
    def field_channels(self):
        return 1 + self.hermite_channels

    def to_dict(self):
        return {"schema": self.schema, "full_config": self.full_config.to_dict()}

    @classmethod
    def from_dict(cls, payload):
        if not isinstance(payload, dict) or set(payload) != {"schema", "full_config"}:
            raise ValueError("unexpected conditional model config fields")
        return cls(VPFPStructuredEDMConfig.from_dict(payload["full_config"]), payload["schema"])


def _context(values, config, *, prototype=None):
    kwargs = {} if prototype is None else {"device": prototype.device, "dtype": prototype.dtype}
    values = torch.as_tensor(values, **kwargs)
    if (values.ndim != 2 or values.shape[1] != config.context_dimension
            or not values.is_floating_point() or not torch.isfinite(values).all()):
        raise ValueError("clean initial context must contain finite complete t0 value controls")
    return values


class VPFPInitialConditionedNormalizer(nn.Module):
    """Immutable train-only normalization; encode returns (clean t0, future)."""

    def __init__(self, config, temporal, *, initial_prior, fit_metadata=None):
        super().__init__()
        if not isinstance(config, VPFPInitialConditionedEDMConfig) or not isinstance(temporal, VPFPTemporalCovarianceNormalizer):
            raise TypeError("conditional normalizer requires its config and frozen temporal parent")
        if temporal.config != config.value_config:
            raise ValueError("conditional normalizer parent config differs")
        if not isinstance(initial_prior, KnownInitialPriorSpec):
            raise TypeError("conditional native asset requires the frozen known-initial prior")
        initial_prior.check_config(config)
        temporal._check_buffer_dtype()
        self.config, self.temporal, self.initial_prior = config, temporal, initial_prior
        # Derive on CPU FP64 both at fit and reload, independent of target GPU.
        root = getattr(temporal, "mixture_square_root", temporal.temporal.square_root).detach().cpu().double()
        if temporal.rho_policy == "identity":
            root = torch.cat((torch.eye(config.full_config.control_count, dtype=torch.float64)[None], root[1:]))
        covariance = root @ root.transpose(-1, -2)
        regression = covariance[:, 1:, 0] / covariance[:, 0, 0, None]
        schur = covariance[:, 1:, 1:] - regression[:, :, None] * covariance[:, None, 0, 1:]
        schur = (schur + schur.transpose(-1, -2)) / 2
        eigenvalues, vectors = torch.linalg.eigh(schur)
        if not torch.isfinite(eigenvalues).all() or torch.any(eigenvalues <= 0):
            raise ValueError("conditional future Schur covariance must remain strictly SPD")
        for name, buffer in {"regression": regression, "schur": schur, "eigenvalues": eigenvalues,
                "square_root": (vectors * eigenvalues.sqrt()[:, None, :]) @ vectors.transpose(-1, -2),
                "inverse_square_root": (vectors * eigenvalues.rsqrt()[:, None, :]) @ vectors.transpose(-1, -2)}.items():
            self.register_buffer(name, buffer.detach().clone())
        expected = {"schema": NORMALIZER_SCHEMA, "split": "train", "sample_ids": list(temporal.fit_metadata["sample_ids"]),
            "generated_samples_used_for_fit": False, "reference_samples_used_for_fit": False,
            "temporal_normalizer_sha256": _json_sha(temporal.to_payload()), "known_initial_prior_sha256": initial_prior.sha256,
            "schur_operator_sha256": self._operator_sha(), "context": "clean_parent_RMS_t0_values_before_temporal_whitening",
            "context_slopes_or_future_values": False, "loss_target": "future_white_values_only",
            "context_dimension": config.context_dimension, "future_dimension": config.latent_dimension,
            "full_decoder_dimension": config.full_dimension, "slope_lift": LIFT_SCHEMA,
            "future_times": torch.linspace(*config.time_domain, config.full_config.control_count, dtype=torch.float64)[1:].tolist(),
            "additional_schur_eigenvalue_floor": False, "future_noise_full_rank": True,
            "conditional_gaussian_target_assumed": False, "physical_answer_fidelity_certified": False}
        if fit_metadata is not None and fit_metadata != expected:
            raise ValueError("conditional normalization provenance or Schur binding differs")
        self.fit_metadata = deepcopy(expected)
        self._metadata_sha = _json_sha(expected)
        self._frozen_versions = {name: value._version for name, value in self.named_buffers()}

    @classmethod
    def fit(cls, config, full_training_latents, *, sample_ids, initial_prior, **kwargs):
        temporal = VPFPTemporalCovarianceNormalizer.fit(config.value_config, full_training_latents,
            sample_ids=sample_ids, **kwargs)
        return cls(config, temporal, initial_prior=initial_prior)

    def _operator_sha(self):
        return _json_sha({name: getattr(self, name).detach().cpu().tolist() for name in
            ("regression", "schur", "eigenvalues", "square_root", "inverse_square_root")})

    def _check(self):
        self.temporal._check_buffer_dtype()
        if any(buffer.dtype != torch.float64 or buffer.requires_grad or buffer._version != self._frozen_versions[name]
               for name, buffer in self.named_buffers()):
            raise ValueError("frozen conditional normalization buffers changed; use .to(device), not dtype casts")

    def encode_initial(self, raw_initial):
        self._check()
        values = _context(raw_initial, self.config)
        scale = self.temporal.parent.group_scales.to(values)
        return torch.cat((values[:, :self.config.rho_modes] / scale[0],
            (values[:, self.config.rho_modes:].reshape(-1, self.config.hermite_channels, self.config.kinetic_modes)
             / scale[None, 1:, None]).flatten(1)), dim=1)

    def encode(self, full_raw):
        self._check()
        parent = self.temporal.parent.encode(full_raw)
        rho, kinetic = _value_split(parent, self.config.value_config)
        context = torch.cat((rho[:, 0], kinetic[:, :, 0].flatten(1)), dim=1)
        rho, kinetic = rho.double(), kinetic.double().clone()
        kinetic[..., 0] -= self.temporal.temporal.kinetic_dc_mean.to(kinetic)[None]
        regression, inverse = self.regression.to(parent.device), self.inverse_square_root.to(parent.device)
        rf = rho[:, 1:] - regression[0, :, None] * rho[:, :1]
        kf = kinetic[:, :, 1:] - regression[None, 1:, :, None] * kinetic[:, :, :1]
        future = torch.cat((torch.einsum("ts,bsm->btm", inverse[0], rf).flatten(1),
            torch.einsum("hts,bhsm->bhtm", inverse[1:], kf).flatten(1)), dim=1).to(parent.dtype)
        return context, _coefficients(future, self.config)

    def decode(self, encoded, future=None):
        self._check()
        if future is None:
            if not isinstance(encoded, (tuple, list)) or len(encoded) != 2:
                raise ValueError("conditional decoding requires (clean context, future)")
            context, future = encoded
        else:
            context = encoded
        future = _coefficients(future, self.config)
        context = _context(context, self.config, prototype=future)
        if len(context) != len(future):
            raise ValueError("context and future batch counts differ")
        cfg = self.config
        rho0 = context[:, :cfg.rho_modes].double()
        kinetic0 = context[:, cfg.rho_modes:].reshape(-1, cfg.hermite_channels, cfg.kinetic_modes).double().clone()
        mean = self.temporal.temporal.kinetic_dc_mean.to(device=context.device)
        kinetic0[..., 0] -= mean[None, :, 0]
        white_rho, white_kinetic = _value_split(future, cfg)
        root, regression = self.square_root.to(context.device), self.regression.to(context.device)
        rho = torch.cat((rho0[:, None], torch.einsum("ts,bsm->btm", root[0], white_rho.double())
                         + regression[0, :, None] * rho0[:, None]), dim=1)
        kinetic = torch.cat((kinetic0[:, :, None], torch.einsum("hts,bhsm->bhtm", root[1:], white_kinetic.double())
                             + regression[None, 1:, :, None] * kinetic0[:, :, None]), dim=2)
        kinetic[..., 0] += mean[None]
        parent = torch.cat((rho.flatten(1), kinetic.flatten(1)), dim=1).to(context.dtype)
        return self.temporal.parent.decode(parent)

    def to_payload(self):
        self._check()
        temporal = self.temporal.to_payload()
        return {"schema": NORMALIZER_SCHEMA, "config": self.config.to_dict(), "temporal_normalizer": temporal,
                "initial_prior": self.initial_prior.to_payload(), "fit_metadata": deepcopy(self.fit_metadata)}

    @classmethod
    def from_payload(cls, payload):
        fields = {"schema", "config", "temporal_normalizer", "initial_prior", "fit_metadata"}
        if not isinstance(payload, dict) or set(payload) != fields or payload["schema"] != NORMALIZER_SCHEMA:
            raise ValueError("unexpected conditional normalizer fields/schema")
        return cls(VPFPInitialConditionedEDMConfig.from_dict(payload["config"]),
            VPFPTemporalCovarianceNormalizer.from_payload(payload["temporal_normalizer"]),
            initial_prior=KnownInitialPriorSpec.from_payload(payload["initial_prior"]), fit_metadata=payload["fit_metadata"])

    def report(self):
        self._check()
        return {"schema": NORMALIZER_SCHEMA, "fit_metadata": deepcopy(self.fit_metadata),
            "minimum_schur_eigenvalue": float(self.eigenvalues.min()),
            "maximum_group_condition_number": float((self.eigenvalues[:, -1] / self.eigenvalues[:, 0]).max()),
            "scientific_distribution_qualification": False}


class VPFPInitialConditionedEDM(VPFPValueControlEDM):
    checkpoint_schema = CHECKPOINT_SCHEMA

    def __init__(self, config):
        if not isinstance(config, VPFPInitialConditionedEDMConfig):
            raise TypeError("initial-conditioned EDM requires its explicit config")
        # Construct the old trunk first: same seed gives exactly the same
        # common parameter initialization. The only extra learned map is the
        # full spatial initial-field input, not a global-moment summary.
        super().__init__(config.value_config)
        self.config = self.transform.config = config
        self.initial_input = nn.Conv2d(config.field_channels, config.width, 1)

    def initial_field(self, initial_context):
        parameter = next(self.parameters())
        values = _context(initial_context, self.config, prototype=parameter)
        rho = values[:, :self.config.rho_modes]
        kinetic = values[:, self.config.rho_modes:].reshape(-1, self.config.hermite_channels, self.config.kinetic_modes)
        return torch.cat((torch.einsum("bm,mxy->bxy", rho, self.transform.rho_basis.to(values))[:, None],
            torch.einsum("bhm,mxy->bhxy", kinetic, self.transform.kinetic_basis.to(values))), dim=1)

    def denoise(self, noisy_normalized_coefficients, sigma, conditions, *, initial_context):
        parameter = next(self.parameters())
        noisy = _coefficients(noisy_normalized_coefficients, self.config, device=parameter.device, dtype=parameter.dtype)
        initial_context = _context(initial_context, self.config, prototype=noisy)
        sigma = torch.as_tensor(sigma, dtype=noisy.dtype, device=noisy.device)
        if (sigma.shape != (len(noisy),) or not torch.isfinite(sigma).all() or torch.any(sigma <= 0)
                or len(initial_context) != len(noisy)):
            raise ValueError("sigma/context batch differs from conditional future")
        condition = self._conditions(conditions, len(noisy), noisy)
        denominator = (sigma.square() + self.config.sigma_data**2).sqrt()
        c_skip, c_out = self.config.sigma_data**2 / denominator.square(), sigma * self.config.sigma_data / denominator
        field = self.transform.to_field(noisy / denominator[:, None])
        frequency = torch.exp(torch.linspace(0., math.log(100.), 8, dtype=noisy.dtype, device=noisy.device))
        phase = sigma.log()[:, None] * frequency[None]
        embedding = self.context(torch.cat((phase.sin(), phase.cos(), condition), dim=1))
        # Physical future positions are 1/32..1, NOT a new 32-point [0,1].
        tau = torch.linspace(0., 1., self.config.full_config.control_count, dtype=noisy.dtype, device=noisy.device)[1:]
        position = self.time_position(torch.stack((tau, (math.pi*tau).sin(), (math.pi*tau).cos()), dim=1))
        hidden = self.input(field) + self.initial_input(self.initial_field(initial_context))[:, :, None] + position.T[None, :, :, None, None]
        for block in self.blocks:
            hidden = block(hidden, embedding)
        residual = self.transform.to_coefficients(self.output(F.silu(hidden)))
        return c_skip[:, None]*noisy + c_out[:, None]*residual

    def _conditioned_view(self, context):
        # Immutable per-call closure lets the exact existing EDM loss and
        # Heun code operate in future space. No mutable model context, no
        # free initial reverse coordinate, no mid-trajectory projection.
        return SimpleNamespace(config=self.config, parameters=self.parameters, _noise=self._noise,
            _conditions=self._conditions, denoise=lambda noisy, sigma, conditions:
                self.denoise(noisy, sigma, conditions, initial_context=context))

    def loss(self, encoded, conditions, *, generator=None, sigma=None, noise=None, residual_metric=None):
        if residual_metric is not None:
            raise ValueError("old value/Sobolev metrics are not conditional future metrics")
        if not isinstance(encoded, (tuple, list)) or len(encoded) != 2:
            raise ValueError("conditional loss requires (clean context, future target)")
        context, future = encoded
        context = _context(context, self.config, prototype=next(self.parameters()))
        return VPFPStructuredEDM.loss(self._conditioned_view(context), future, conditions,
            generator=generator, sigma=sigma, noise=noise, residual_metric=None)

    @torch.no_grad()
    def sample(self, initial_context, conditions, seeds, **kwargs):
        seeds = _seeds(seeds)
        context = _context(initial_context, self.config, prototype=next(self.parameters()))
        conditions = torch.as_tensor(conditions, dtype=context.dtype, device=context.device)
        if len(context) != len(seeds):
            raise ValueError("one initial context per future noise seed is required")
        self._conditions(conditions, len(seeds), context)
        return torch.cat([VPFPStructuredEDM.sample(self._conditioned_view(context[index:index+1]),
            conditions[index:index+1], (seed,), **kwargs) for index, seed in enumerate(seeds)], dim=0)

    def architecture_report(self):
        return {"schema": self.config.schema, "future_dimension": self.config.latent_dimension,
            "context_dimension": self.config.context_dimension, "initial_condition": "complete_rho_and_all_Hermite_spatial_fields",
            "context_input": "one_by_one_spatial_channel_mixing_broadcast_to_all_future_times",
            "initial_target_or_noise": False, "initial_state_overwritten_during_sampling": False,
            "future_time_fractions": [i/self.config.control_count for i in range(1, self.config.control_count+1)],
            "loss": "original_EDM_future_only_with_fixed_rho_vs_kinetic_group_weight",
            "macro_loss_fraction_is_physical_macro_loss": False, "physical_derivative_distribution_matching": False,
            "learnable_parameters": sum(parameter.numel() for parameter in self.parameters())}


class VPFPInitialConditionedBatchSampler(VPFPDiffusionBatchSampler):
    def __init__(self, model, normalizer, **kwargs):
        if not isinstance(model, VPFPInitialConditionedEDM) or not isinstance(normalizer, VPFPInitialConditionedNormalizer):
            raise TypeError("conditional native sampler needs matching conditional model/normalizer")
        super().__init__(model, normalizer, **kwargs)
        prior = normalizer.initial_prior.to_payload()
        if self.identity["base_decoder_sha256"] != prior["model_spec"]["base_decoder_sha256"]:
            raise ValueError("conditional model and known-initial decoder bindings differ")
        self.prior = KnownInitialPrior(normalizer.initial_prior, device=next(model.parameters()).device)
        self.identity.update(kind="known_initial_prior_then_clean_initial_conditioned_future_edm", checkpoint_schema=CHECKPOINT_SCHEMA,
            latent_dimension=model.config.full_dimension, random_dimension=model.config.latent_dimension,
            initial_context_dimension=model.config.context_dimension, known_initial_prior_sha256=normalizer.initial_prior.sha256,
            corpus_manifest_sha256=prior["corpus_manifest_sha256"], architecture=model.architecture_report(),
            initial_law_fidelity="original_native_projection_of_known_analytic_law_not_lossless_analytic_f0")
        self.identity["sampling"].update(noise_space="future_Schur_white_values_only",
            condition_prior="original_known_law_nu_T_or_explicit_requested_nu_T",
            initial_prior_randomness="domain_separated_Gaussian_packet_parameters_not_EDM_initial_noise",
            decoder_lift_timing="inverse_future_Schur_restore_clean_t0_then_original_value_slope_lift",
            seed_is_network_condition=False, source_is_network_condition=False)

    def _sample(self, seeds, conditions=None):
        seeds = _seeds(seeds)
        stream = lambda tag, seed: int.from_bytes(sha256(f"vpfp-initial-conditioned-{tag}:{seed}".encode()).digest()[:8], "little") % 2**63
        prior_seeds, future_seeds = [stream("prior", seed) for seed in seeds], [stream("future", seed) for seed in seeds]
        initial = self.prior.sample(prior_seeds, conditions)
        parameter = next(self.model.parameters())
        context = self.normalizer.encode_initial(initial["initial_controls"].to(parameter))
        future = self.model.sample(context, initial["equation_conditions"], future_seeds, steps=self.steps, churn=self.churn)
        full = self.normalizer.decode((context, future))
        return full, initial["equation_conditions"], prior_seeds, future_seeds

    @torch.no_grad()
    def sample_conditions(self, conditions, seeds):
        """Full native coefficients for the explicitly supplied physical conditions."""
        return self._sample(seeds, conditions)[0]

    @torch.no_grad()
    def generate(self, seeds):
        seeds = _seeds(seeds)
        full, conditions, prior_seeds, future_seeds = self._sample(seeds)
        return {"schema": "vpfp_learned_answer_batch_v1", "latents": full.detach().cpu().numpy().astype(np.float32),
            "equation_conditions": conditions, "equation_seeds": np.asarray(seeds, dtype=np.int64),
            "diagnostics": {"pde_solves": 0, "learned_generation": True, "known_prior_seeds": prior_seeds,
                "future_noise_seeds": future_seeds, "scientific_qualification": "not_established_by_sampling"}}


def _sampler_from_initial_conditioned_payload(payload, expected_sha256, device, *, steps, churn):
    """For the canonical SHA-verified weights_only loader; no inference here."""
    fields = {"schema", "config", "model", "normalizer", "base_decoder_sha256", "corpus_manifest_sha256", "training"}
    if not isinstance(payload, dict) or set(payload) != fields or payload["schema"] != CHECKPOINT_SCHEMA:
        raise ValueError("unexpected initial-conditioned checkpoint schema/fields")
    config = VPFPInitialConditionedEDMConfig.from_dict(payload["config"])
    normalizer = VPFPInitialConditionedNormalizer.from_payload(payload["normalizer"])
    if config != normalizer.config:
        raise ValueError("conditional model and normalizer configs differ")
    model = VPFPInitialConditionedEDM(config)
    state, expected = payload["model"], model.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError("conditional model tensor keys differ")
    for name, target in expected.items():
        tensor = state[name]
        if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided or tensor.shape != target.shape
                or tensor.dtype != target.dtype or not torch.isfinite(tensor).all()):
            raise ValueError(f"invalid conditional model tensor: {name}")
        if name.startswith("transform.") and not torch.equal(tensor, target):
            raise ValueError("checkpoint changed immutable conditional Fourier geometry")
    # Validate the complete existing parent normalizer lineage and the shared
    # trunk tensors through the old strict checker; extra condition tensors
    # were checked above. This is an in-memory validation view, not an asset.
    view = dict(payload, schema=VALUE_SCHEMA, config=config.value_config.to_dict(),
        normalizer=normalizer.temporal.parent.to_payload(),
        model={name: tensor for name, tensor in state.items() if not name.startswith("initial_input.")})
    checked = _sampler_from_value_payload(view, expected_sha256, "cpu", steps=steps, churn=churn)
    prior = normalizer.initial_prior.to_payload()
    if payload["corpus_manifest_sha256"] != prior["corpus_manifest_sha256"]:
        raise ValueError("conditional checkpoint and initial prior corpus identity differ")
    training = payload["training"]
    if "train_case_ids" in training and training["train_case_ids"] != [
            f"{prior['model_spec']['regime_id']}:{seed}" for seed in normalizer.fit_metadata["sample_ids"]]:
        raise ValueError("conditional training case IDs differ from normalization fit IDs")
    model.load_state_dict(state, strict=True)
    model.to(device=device).eval().requires_grad_(False)
    normalizer.to(device=device)
    sampler = VPFPInitialConditionedBatchSampler(model, normalizer,
        base_decoder_sha256=checked.identity["base_decoder_sha256"], checkpoint_sha256=_sha(expected_sha256, "conditional checkpoint"),
        steps=steps, churn=churn)
    sampler.identity["training"] = deepcopy(training)
    return sampler
