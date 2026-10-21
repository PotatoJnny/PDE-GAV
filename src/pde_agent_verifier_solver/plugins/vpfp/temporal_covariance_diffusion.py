"""Native VPFP diffusion asset in train-whitened temporal value coordinates.

An immutable, full-rank affine coordinate change is fitted on training answers
only. EDM noise and all reverse states live in these white coordinates. After
sampling, its inverse and the existing value-to-slope lift recover the complete
physical-decoder ABI. No high-frequency direction is discarded; no generated
physical derivative distribution is silently imposed on denoiser outputs.
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import math

import torch
from torch import nn

from .noise_geometry import VPFPTemporalCovarianceTransform
from .noise_training import VPFPLowNoiseValueControlEDM
from .structured_diffusion import _coefficients, _seeds, _sha
from .value_control_diffusion import (
    CHECKPOINT_SCHEMA as VALUE_CHECKPOINT_SCHEMA, LIFT_SCHEMA,
    VPFPValueEDMConfig, VPFPValueControlNormalizer, VPFPValueControlEDM,
    VPFPValueDiffusionBatchSampler, _sampler_from_value_payload,
)


CHECKPOINT_SCHEMA = "vpfp_temporal_covariance_diffusion_checkpoint_v1"
NORMALIZER_SCHEMA = "vpfp_native_temporal_covariance_normalization_v1"
MIXTURE_DEFINITION = "(1-alpha)*I+alpha*C_regularized_in_parent_value_coordinates"
FACTORIAL_RECIPES = {
    "trace_isotropic": "tau_g*I; tau_g=trace(C_regularized_g)/control_count",
    "unit_trace_shape": "C_regularized_g/tau_g; tau_g=trace(C_regularized_g)/control_count",
}
TAU_DEFINITION = "mean_regularized_eigenvalues_per_group_after_the_existing_floor_only"


def _json_sha(value):
    try:
        encoded = json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise ValueError("temporal asset metadata must be finite JSON") from error
    return sha256(encoded.encode()).hexdigest()


def _check_transform_parent(parent, temporal):
    """Bind the empirical transform to the same training answers and units."""
    if parent.config != temporal.config:
        raise ValueError("temporal transform and value normalizer configs differ")
    metadata, parent_metadata = temporal.fit_metadata, parent.fit_metadata
    ids = _seeds(metadata.get("sample_ids", ()))
    if (metadata.get("sample_ids") != parent_metadata.get("sample_ids")
            or metadata.get("trajectory_count") != len(ids)
            or metadata.get("fit_coordinate_system") != "caller_supplied_reduced_values"
            or metadata.get("accumulation_dtype") != "float64"
            or metadata.get("second_moment_definition") != "mean_over_cases_and_spatial_modes_of_outer_product_in_time"
            or metadata.get("source_dtype") not in ("torch.float32", "torch.float64")
            or isinstance(metadata.get("batch_size"), bool)
            or not isinstance(metadata.get("batch_size"), int)
            or not 1 <= metadata["batch_size"] <= 8):
        raise ValueError("temporal transform training provenance differs from its value parent")
    # Centering only one (DC) spatial mode changes C by mu mu^T / mode_count.
    # This relationship catches an internally inconsistent raw/centered/mean
    # payload even if someone recalculates its outer JSON hash.
    expected = torch.zeros_like(temporal.second_moment)
    mean = temporal.kinetic_dc_mean
    expected[1:] = mean[:, :, None]*mean[:, None, :] / parent.config.kinetic_modes
    actual = temporal.raw_second_moment - temporal.second_moment
    scale = torch.maximum(temporal.raw_second_moment.abs(), temporal.second_moment.abs()).amax((-2, -1))
    scale = torch.maximum(scale, expected.abs().amax((-2, -1))).clamp_min(torch.finfo(torch.float64).tiny)
    if torch.any((actual-expected).abs().amax((-2, -1)) > 1.e-8*scale):
        raise ValueError("temporal centered/raw moments do not match the kinetic-DC mean")


class VPFPTemporalCovarianceNormalizer(nn.Module):
    """Value RMS followed by fixed symmetric temporal whitening, not a VAE."""

    def __init__(self, parent, temporal, *, rho_policy="identity", covariance_strength=1.,
            covariance_recipe="mixture", fit_metadata=None):
        super().__init__()
        if rho_policy not in ("identity", "covariance"):
            raise ValueError("rho_policy must be identity or covariance")
        if (isinstance(covariance_strength, bool) or not isinstance(covariance_strength, (int, float))
                or not math.isfinite(covariance_strength) or not 0 <= covariance_strength <= 1):
            raise ValueError("covariance_strength must be finite in [0,1]")
        if covariance_strength != 1 and rho_policy != "identity":
            raise ValueError("covariance shrinkage requires the unchanged identity rho_policy")
        if covariance_recipe not in ("mixture", *FACTORIAL_RECIPES):
            raise ValueError("unknown covariance_recipe")
        if covariance_recipe != "mixture" and (covariance_strength != 1 or rho_policy != "identity"):
            raise ValueError("factorial covariance recipes require strength=1 and unchanged identity rho_policy")
        if not isinstance(parent, VPFPValueControlNormalizer) or not isinstance(temporal, VPFPTemporalCovarianceTransform):
            raise TypeError("native temporal normalization needs its fixed value parent and temporal transform")
        _check_transform_parent(parent, temporal)
        self.config = parent.config
        self.parent, self.temporal = parent, temporal
        self.rho_policy = rho_policy
        self.covariance_strength = float(covariance_strength)
        self.covariance_recipe = covariance_recipe
        parent_sha, geometry_sha = _json_sha(parent.to_payload()), _json_sha(temporal.to_payload())
        # The default endpoint intentionally retains the exact legacy payload
        # and sampler identity. New shrinkage recipes bind their alpha and
        # mathematical definition into the same native normalizer receipt.
        kinetic_policy = "covariance" if covariance_strength == 1 else "identity_covariance_mixture"
        mixture = {} if covariance_strength == 1 else {
            "covariance_strength": self.covariance_strength, "covariance_mixture_definition": MIXTURE_DEFINITION}
        if covariance_recipe != "mixture":
            kinetic_policy = covariance_recipe
            mixture = {"covariance_recipe": covariance_recipe,
                "covariance_recipe_definition": FACTORIAL_RECIPES[covariance_recipe],
                "tau_definition": TAU_DEFINITION,
                "kinetic_tau": temporal.regularized_eigenvalues[1:].mean(-1).tolist(),
                "post_recipe_floor_or_clipping": False}
        scope_sha = _json_sha({"value_normalizer_sha256": parent_sha,
            "temporal_transform_sha256": geometry_sha, "rho_policy": rho_policy, "kinetic_policy": kinetic_policy,
            **mixture})
        expected_metadata = {
            "schema": NORMALIZER_SCHEMA, "split": "train",
            "sample_ids": list(parent.fit_metadata["sample_ids"]),
            "generated_samples_used_for_fit": False, "reference_samples_used_for_fit": False,
            "input_coordinate_system": "paired_full_normalizer_value_coordinates",
            "model_coordinate_system": "fixed_scoped_symmetric_temporal_value_coordinates",
            "value_normalizer_sha256": parent_sha, "temporal_transform_sha256": geometry_sha,
            "normalization_scope_sha256": scope_sha,
            "rho_policy": rho_policy, "kinetic_policy": kinetic_policy,
            "rho_temporal_statistics": "retained_diagnostic_only_not_applied" if rho_policy == "identity" else "applied_covariance",
            "centering": "kinetic_spatial_DC_train_mean_only" if temporal.fit_metadata["center_kinetic_dc"] else "none",
            "relative_floor": temporal.fit_metadata["relative_floor"],
            "absolute_floor": temporal.fit_metadata["absolute_floor"],
            "floor_definition": "max(relative_floor_squared_times_group_mean_variance,absolute_floor_squared)",
            "full_rank_affine": True, "subspace_or_frequency_removal": False,
            "inactive_policy": "exact_zero_after_lift", "slope_lift": LIFT_SCHEMA,
            "native_sampler_asset": True, "physical_answer_fidelity_certified": False,
            **mixture,
        }
        if fit_metadata is not None and fit_metadata != expected_metadata:
            raise ValueError("native temporal normalization metadata/bindings differ")
        self.fit_metadata = deepcopy(expected_metadata)
        if covariance_strength != 1 or covariance_recipe != "mixture":
            # C is floored first by the existing training-only transform. No
            # second floor or group rescaling is applied to the mixture. The
            # alpha=0 endpoint is exactly I, not trace(C)/T times I.
            eigenvalues = (1 - covariance_strength) + covariance_strength * temporal.regularized_eigenvalues
            vectors = temporal.eigenvectors
            if covariance_recipe == "unit_trace_shape":
                eigenvalues = temporal.regularized_eigenvalues / temporal.regularized_eigenvalues.mean(-1, keepdim=True)
            if covariance_strength == 0 or covariance_recipe == "trace_isotropic":
                root = torch.eye(self.config.control_count, dtype=torch.float64,
                    device=vectors.device).expand_as(vectors).clone()
                inverse = root.clone()
                if covariance_recipe == "trace_isotropic":
                    tau = temporal.regularized_eigenvalues.mean(-1)
                    # Fresh tensors keep version zero, like a device copy.
                    # In-place construction would falsely trip the frozen
                    # buffer guard after the first CPU -> CUDA migration.
                    root = root * tau.sqrt()[:, None, None]
                    inverse = inverse * tau.rsqrt()[:, None, None]
            else:
                root = (vectors * eigenvalues.sqrt()[:, None, :]) @ vectors.transpose(-1, -2)
                inverse = (vectors * eigenvalues.rsqrt()[:, None, :]) @ vectors.transpose(-1, -2)
            self.register_buffer("mixture_square_root", root)
            self.register_buffer("mixture_inverse_square_root", inverse)
        self._frozen_buffer_versions = {name: buffer._version for name, buffer in self.named_buffers()}

    @classmethod
    def fit(cls, config, full_training_latents, *, sample_ids, split="train",
            relative_floor=.05, absolute_floor=1.e-6, center_kinetic_dc=True, batch_size=8,
            rho_policy="identity", covariance_strength=1., covariance_recipe="mixture"):
        if split != "train":
            raise ValueError("native temporal normalization only accepts the train split")
        if rho_policy not in ("identity", "covariance"):
            raise ValueError("rho_policy must be identity or covariance")
        parent = VPFPValueControlNormalizer.fit(config, full_training_latents,
            sample_ids=sample_ids, split=split)
        reduced = parent.encode(full_training_latents)
        temporal = VPFPTemporalCovarianceTransform.fit(config, reduced, sample_ids=sample_ids,
            split=split, relative_floor=relative_floor, absolute_floor=absolute_floor,
            center_kinetic_dc=center_kinetic_dc, batch_size=batch_size)
        return cls(parent, temporal, rho_policy=rho_policy, covariance_strength=covariance_strength,
            covariance_recipe=covariance_recipe)

    def _check_buffer_dtype(self):
        # Moving devices is supported; silently downcasting covariance factors
        # then recasting them inside the transform would change the asset.
        if any(buffer.dtype != torch.float64 for buffer in self.buffers()):
            raise ValueError("frozen temporal normalization buffers must remain float64; use .to(device) only")
        if any(buffer.requires_grad or buffer._version != self._frozen_buffer_versions[name]
                for name, buffer in self.named_buffers()):
            raise ValueError("frozen temporal normalization buffers were modified in place")
        if self.rho_policy != self.fit_metadata["rho_policy"]:
            raise ValueError("frozen temporal normalization rho_policy changed")
        if self.covariance_strength != self.fit_metadata.get("covariance_strength", 1.):
            raise ValueError("frozen temporal normalization covariance_strength changed")
        if self.covariance_recipe != self.fit_metadata.get("covariance_recipe", "mixture"):
            raise ValueError("frozen temporal normalization covariance_recipe changed")

    def _scoped_temporal(self, values, *, inverse):
        values = _coefficients(values, self.config)
        operation = self.temporal.decode if inverse else self.temporal.encode
        if self.covariance_strength != 1 or self.covariance_recipe != "mixture":
            def operation(values):
                return self.temporal._transform(values,
                    self.mixture_square_root if inverse else self.mixture_inverse_square_root,
                    subtract_mean=not inverse, add_mean=inverse)
        if self.rho_policy == "covariance":
            return operation(values)
        # Do not evaluate the fitted rho operator on actual data/noise when
        # it is outside this ablation's scope. The audited transform sees a
        # zero rho placeholder; the original block is passed through exactly.
        rho = values[:, :self.config.rho_dimension]
        kinetic_input = torch.cat((torch.zeros_like(rho), values[:, self.config.rho_dimension:]), dim=1)
        transformed = operation(kinetic_input)
        return torch.cat((rho, transformed[:, self.config.rho_dimension:]), dim=1)

    def encode(self, full_raw_latents):
        self._check_buffer_dtype()
        return self._scoped_temporal(self.parent.encode(full_raw_latents), inverse=False)

    def decode(self, white_values):
        self._check_buffer_dtype()
        return self.parent.decode(self._scoped_temporal(white_values, inverse=True))

    def report(self):
        self._check_buffer_dtype()
        return {"schema": NORMALIZER_SCHEMA, "fit_metadata": deepcopy(self.fit_metadata),
            "random_dimension": self.config.latent_dimension, "full_decoder_dimension": self.config.full_dimension,
            "native_sampler_asset": True, "full_rank_affine": True,
            "decode_order": ["inverse_temporal_affine", "inverse_value_RMS", "fixed_value_to_slope_lift"],
            "physical_derivative_distribution_matching": False, "physical_distribution_qualification": False}

    def to_payload(self):
        self._check_buffer_dtype()
        parent, temporal = self.parent.to_payload(), self.temporal.to_payload()
        return {"schema": NORMALIZER_SCHEMA, "config": self.config.to_dict(),
            "value_normalizer": parent, "temporal_transform": temporal,
            "fit_metadata": deepcopy(self.fit_metadata)}

    @classmethod
    def from_payload(cls, payload):
        fields = {"schema", "config", "value_normalizer", "temporal_transform", "fit_metadata"}
        if not isinstance(payload, dict) or set(payload) != fields or payload.get("schema") != NORMALIZER_SCHEMA:
            raise ValueError("unexpected native temporal normalizer schema/fields")
        metadata = payload["fit_metadata"]
        if not isinstance(metadata, dict):
            raise ValueError("native temporal normalization metadata must be a dictionary")
        _json_sha(metadata)
        parent = VPFPValueControlNormalizer.from_payload(payload["value_normalizer"])
        temporal = VPFPTemporalCovarianceTransform.from_payload(payload["temporal_transform"])
        normalizer = cls(parent, temporal, rho_policy=metadata.get("rho_policy"),
            covariance_strength=metadata.get("covariance_strength", 1.),
            covariance_recipe=metadata.get("covariance_recipe", "mixture"), fit_metadata=metadata)
        if normalizer.config != VPFPValueEDMConfig.from_dict(payload["config"]):
            raise ValueError("native temporal normalizer config differs from its components")
        return normalizer


class VPFPTemporalCovarianceEDM(VPFPValueControlEDM):
    """Same network/EDM equations, with explicit native white-coordinate asset."""

    checkpoint_schema = CHECKPOINT_SCHEMA

    def loss(self, clean_normalized_coefficients, conditions, *, generator=None,
             sigma=None, noise=None, residual_metric=None):
        if residual_metric is not None:
            raise ValueError("old value/full Sobolev metrics do not apply to temporal-white coordinates")
        return super().loss(clean_normalized_coefficients, conditions, generator=generator,
            sigma=sigma, noise=noise, residual_metric=None)


class VPFPLowNoiseTemporalCovarianceEDM(VPFPLowNoiseValueControlEDM, VPFPTemporalCovarianceEDM):
    """Reuse the training-only sigma override; covariance loss guard follows it."""

    checkpoint_schema = CHECKPOINT_SCHEMA


class VPFPTemporalCovarianceBatchSampler(VPFPValueDiffusionBatchSampler):
    def __init__(self, model, normalizer, **kwargs):
        if not isinstance(normalizer, VPFPTemporalCovarianceNormalizer):
            raise TypeError("temporal sampler needs its native affine normalizer")
        super().__init__(model, normalizer, **kwargs)
        self.identity.update(kind="temporal_covariance_value_control_conditional_edm",
            checkpoint_schema=CHECKPOINT_SCHEMA,
            geometry_binding={key: normalizer.fit_metadata[key] for key in
                ("value_normalizer_sha256", "temporal_transform_sha256", "normalization_scope_sha256", "rho_policy",
                 "kinetic_policy", "centering", "relative_floor", "absolute_floor")})
        self.identity["sampling"].update(noise_space=("independent_gaussian_in_kinetic_whitened_and_original_rho_value_coordinates"
                if normalizer.rho_policy == "identity" else "independent_gaussian_in_temporally_whitened_value_coordinates"),
            physical_value_noise=("kinetic_train_covariance_colored_noise_with_original_rho_noise"
                if normalizer.rho_policy == "identity" else "fixed_train_covariance_colored_temporal_noise"),
            decoder_lift_timing="after_complete_white_Heun_trajectory_and_inverse_temporal_affine")
        if normalizer.covariance_strength != 1:
            self.identity["geometry_binding"].update({key: normalizer.fit_metadata[key]
                for key in ("covariance_strength", "covariance_mixture_definition")})
            self.identity["sampling"].update(
                noise_space="independent_gaussian_in_kinetic_mixture_normalized_and_original_rho_value_coordinates",
                physical_value_noise="kinetic_identity_covariance_mixture_with_original_rho_noise",
                covariance_strength=normalizer.covariance_strength)
        if normalizer.covariance_recipe != "mixture":
            self.identity["geometry_binding"].update({key: normalizer.fit_metadata[key]
                for key in ("covariance_recipe", "covariance_recipe_definition", "tau_definition",
                    "kinetic_tau", "post_recipe_floor_or_clipping")})
            self.identity["sampling"].update(
                noise_space="independent_gaussian_in_kinetic_factorial_normalized_and_original_rho_value_coordinates",
                physical_value_noise=FACTORIAL_RECIPES[normalizer.covariance_recipe] + "; rho_noise_unchanged",
                covariance_recipe=normalizer.covariance_recipe)


def _sampler_from_temporal_payload(payload, expected_sha256, device, *, steps, churn):
    """Canonical hash loader dispatch, reusing existing value-state validation."""
    fields = {"schema", "config", "model", "normalizer", "base_decoder_sha256", "corpus_manifest_sha256", "training"}
    if not isinstance(payload, dict) or set(payload) != fields or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("unexpected temporal diffusion checkpoint schema/fields")
    normalizer = VPFPTemporalCovarianceNormalizer.from_payload(payload["normalizer"])
    if normalizer.config != VPFPValueEDMConfig.from_dict(payload["config"]):
        raise ValueError("temporal checkpoint model/normalizer configs differ")
    # The original validator checks all seven fields, decoder/corpus hashes,
    # finite training metadata, the complete parent-normalizer provenance and
    # every model tensor including immutable Fourier buffers. This in-memory
    # view is not a rewritten checkpoint and does not bypass the outer hash.
    value_view = dict(payload, schema=VALUE_CHECKPOINT_SCHEMA,
        normalizer=normalizer.parent.to_payload())
    checked = _sampler_from_value_payload(value_view, expected_sha256, device,
        steps=steps, churn=churn)
    model = VPFPTemporalCovarianceEDM(normalizer.config)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device=device).eval().requires_grad_(False)
    normalizer.to(device=device)
    sampler = VPFPTemporalCovarianceBatchSampler(model, normalizer,
        base_decoder_sha256=checked.identity["base_decoder_sha256"], checkpoint_sha256=expected_sha256,
        steps=steps, churn=churn)
    sampler.identity.update(corpus_manifest_sha256=checked.identity["corpus_manifest_sha256"],
        training=deepcopy(checked.identity["training"]))
    return sampler


__all__ = ["VPFPTemporalCovarianceNormalizer", "VPFPTemporalCovarianceEDM",
    "VPFPLowNoiseTemporalCovarianceEDM", "VPFPTemporalCovarianceBatchSampler"]
