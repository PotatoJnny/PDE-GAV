"""Train-only, full-rank temporal Gaussian diagnostics in reduced VPFP values.

This experimental transform is NOT a native sampler/checkpoint asset. The
caller defines input units (raw coefficients or old normalized value blocks).
No temporal frequency or coefficient is removed, and no generated/reference
case may be used to fit it. One temporal operator is shared by all spatial
modes of a rho/Hermite group, preserving Fourier translation pairs.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from typing import Mapping

import torch
from torch import nn

from .structured_diffusion import _coefficients, _seeds
from .value_control_diffusion import VPFPValueEDMConfig, _value_split


SCHEMA = "vpfp_train_temporal_covariance_diagnostic_v1"


def _values(values, config):
    values = _coefficients(values, config)
    if values.dtype not in (torch.float32, torch.float64) or len(values) < 1:
        raise ValueError("temporal diagnostics require nonempty float32/float64 reduced values")
    return values


def _dct(count, *, device):
    frequency = torch.arange(count, dtype=torch.float64, device=device)
    location = torch.arange(count, dtype=torch.float64, device=device) + .5
    matrix = torch.cos(math.pi * frequency[:, None] * location[None, :] / count) * math.sqrt(2. / count)
    matrix[0] /= math.sqrt(2.)
    return matrix


def _second_moment(values, config, mean, batch_size):
    count, groups, time = len(values), 1 + config.hermite_channels, config.control_count
    moment = torch.zeros(groups, time, time, dtype=torch.float64, device=values.device)
    for chunk in values.split(batch_size):
        rho, kinetic = _value_split(chunk, config)
        rho, kinetic = rho.detach().double(), kinetic.detach().double().clone()
        if mean is not None:
            kinetic[..., 0] -= mean[None]
        moment[0] += torch.einsum("btm,bsm->ts", rho, rho) / (count * config.rho_modes)
        moment[1:] += torch.einsum("bhtm,bhsm->hts", kinetic, kinetic) / (count * config.kinetic_modes)
    return (moment + moment.transpose(-1, -2)) * .5


def _statistics(moment):
    basis = _dct(moment.shape[-1], device=moment.device)
    spectrum = torch.einsum("kt,gts,ks->gk", basis, moment, basis).clamp_min(0.)
    rms = moment.diagonal(dim1=-2, dim2=-1).clamp_min(0.).sqrt()
    return {"time_rms": rms.detach().cpu().tolist(),
            "dct_mean_energy": spectrum.detach().cpu().tolist(),
            "time_mean_square": moment.diagonal(dim1=-2, dim2=-1).mean(-1).detach().cpu().tolist()}


class VPFPTemporalCovarianceTransform(nn.Module):
    """Fixed symmetric temporal whitening with an exactly defined inverse.

    ``C[g] = mean_{training cases, spatial modes} v(t) v(s)`` is uncentered
    except for the OPTIONAL train-mean subtraction in kinetic spatial DC only.
    Covariance regularization raises eigenvalues to a strictly positive floor;
    it never clips observed values or removes directions. Matmul is evaluated
    in float64 then returned in input dtype, bounding float32 factor-cast error.
    """

    def __init__(self, config, second_moment, raw_second_moment, kinetic_dc_mean, *, fit_metadata):
        super().__init__()
        if not isinstance(config, VPFPValueEDMConfig):
            raise TypeError("temporal transform requires VPFPValueEDMConfig")
        metadata = deepcopy(dict(fit_metadata))
        if (metadata.get("split") != "train" or metadata.get("generated_samples_used_for_fit") is not False
                or metadata.get("reference_samples_used_for_fit") is not False
                or not isinstance(metadata.get("center_kinetic_dc"), bool)):
            raise ValueError("temporal covariance fitting is training-only")
        ids = _seeds(metadata.get("sample_ids", ()))
        if len(ids) < 2 or len(set(ids)) != len(ids):
            raise ValueError("temporal covariance needs distinct training case identities")
        relative, absolute = metadata.get("relative_floor"), metadata.get("absolute_floor")
        if (isinstance(relative, bool) or isinstance(absolute, bool)
                or not isinstance(relative, (int, float)) or not isinstance(absolute, (int, float))
                or not math.isfinite(relative) or not 0 < relative <= 1
                or not math.isfinite(absolute) or absolute <= 0):
            raise ValueError("temporal eigenvalue floors must be finite and positive")
        empirical = torch.as_tensor(second_moment, dtype=torch.float64).detach().clone()
        original = torch.as_tensor(raw_second_moment, dtype=torch.float64, device=empirical.device).detach().clone()
        mean = torch.as_tensor(kinetic_dc_mean, dtype=torch.float64, device=empirical.device).detach().clone()
        shape = (1 + config.hermite_channels, config.control_count, config.control_count)
        if (empirical.shape != shape or original.shape != shape
                or mean.shape != (config.hermite_channels, config.control_count)
                or not all(torch.isfinite(value).all() for value in (empirical, original, mean))):
            raise ValueError("invalid temporal moment/mean shape or finite values")
        for value in (empirical, original):
            if not torch.allclose(value, value.transpose(-1, -2), rtol=1.e-10, atol=1.e-12):
                raise ValueError("temporal second moments must be symmetric")
        if not metadata["center_kinetic_dc"] and torch.count_nonzero(mean):
            raise ValueError("a nonzero mean requires kinetic-DC centering")
        eigenvalues, eigenvectors = torch.linalg.eigh(empirical)
        magnitude = empirical.abs().amax(dim=(-2, -1)).clamp_min(torch.finfo(torch.float64).tiny)
        if torch.any(eigenvalues[:, 0] < -1.e-10 * magnitude):
            raise ValueError("temporal second moment is not positive semidefinite")
        for value in (original,):
            eigen = torch.linalg.eigvalsh(value)
            if torch.any(eigen[:, 0] < -1.e-10 * value.abs().amax(dim=(-2, -1)).clamp_min(torch.finfo(torch.float64).tiny)):
                raise ValueError("raw temporal second moment is not positive semidefinite")
        average_variance = empirical.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(0.)
        floor = (relative ** 2 * average_variance).clamp_min(absolute ** 2)
        floored = torch.maximum(eigenvalues, floor[:, None])
        if not torch.isfinite(floored).all() or torch.any(floored <= 0):
            raise ValueError("requested floor is not representable as a positive finite eigenvalue")
        root = (eigenvectors * floored.sqrt()[:, None, :]) @ eigenvectors.transpose(-1, -2)
        inverse = (eigenvectors * floored.rsqrt()[:, None, :]) @ eigenvectors.transpose(-1, -2)
        if not torch.isfinite(root).all() or not torch.isfinite(inverse).all():
            raise ValueError("temporal covariance factors exceed float64 precision bounds")
        identity = torch.eye(config.control_count, dtype=torch.float64, device=empirical.device)
        factor_error = (root @ inverse - identity).abs().amax()
        if float(factor_error) > 1.e-7:
            raise ValueError("temporal factors are too ill-conditioned for a reliable inverse; increase the floor")
        self.config, self.fit_metadata = config, metadata
        for name, value in (("second_moment", empirical), ("raw_second_moment", original),
                            ("kinetic_dc_mean", mean), ("eigenvalues", eigenvalues),
                            ("regularized_eigenvalues", floored), ("eigenvectors", eigenvectors),
                            ("eigenvalue_floor", floor), ("square_root", root), ("inverse_square_root", inverse)):
            self.register_buffer(name, value)

    @classmethod
    def fit(cls, config, training_values, *, sample_ids, split="train", relative_floor=.05,
            absolute_floor=1.e-6, center_kinetic_dc=False, batch_size=8):
        if split != "train":
            raise ValueError("temporal covariance only accepts the train split")
        if not isinstance(config, VPFPValueEDMConfig):
            raise TypeError("temporal transform requires VPFPValueEDMConfig")
        if not isinstance(center_kinetic_dc, bool):
            raise ValueError("center_kinetic_dc must be boolean")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 8:
            raise ValueError("temporal fitting batch_size must lie in [1,8]")
        values = _values(training_values, config).detach()
        ids = _seeds(sample_ids)
        if len(values) < 2 or len(ids) != len(values) or len(set(ids)) != len(ids):
            raise ValueError("temporal fitting requires >=2 distinct training case identities")
        mean = torch.zeros(config.hermite_channels, config.control_count, dtype=torch.float64, device=values.device)
        if center_kinetic_dc:
            for chunk in values.split(batch_size):
                _, kinetic = _value_split(chunk, config)
                mean += kinetic[..., 0].double().sum(0) / len(values)
        raw = _second_moment(values, config, None, batch_size)
        moment = _second_moment(values, config, mean, batch_size) if center_kinetic_dc else raw.clone()
        return cls(config, moment, raw, mean, fit_metadata={
            "split": "train", "sample_ids": list(ids), "generated_samples_used_for_fit": False,
            "reference_samples_used_for_fit": False, "trajectory_count": len(values),
            "fit_coordinate_system": "caller_supplied_reduced_values",
            "center_kinetic_dc": center_kinetic_dc, "relative_floor": relative_floor,
            "absolute_floor": absolute_floor, "accumulation_dtype": "float64",
            "source_dtype": str(values.dtype), "batch_size": batch_size,
            "second_moment_definition": "mean_over_cases_and_spatial_modes_of_outer_product_in_time",
            "native_sampler_asset": False, "physical_fidelity_certified": False})

    def _transform(self, values, matrices, *, subtract_mean=False, add_mean=False):
        values = _values(values, self.config)
        rho, kinetic = _value_split(values, self.config)
        rho, kinetic = rho.double(), kinetic.double().clone()
        mean = self.kinetic_dc_mean.to(device=values.device, dtype=torch.float64)
        matrix = matrices.to(device=values.device, dtype=torch.float64)
        if subtract_mean:
            kinetic[..., 0] -= mean[None]
        if matrix.ndim == 3:
            rho = torch.einsum("ts,bsm->btm", matrix[0], rho)
            kinetic = torch.einsum("hts,bhsm->bhtm", matrix[1:], kinetic)
        elif matrix.ndim == 4:
            rho = torch.einsum("bts,bsm->btm", matrix[:, 0], rho)
            kinetic = torch.einsum("bhts,bhsm->bhtm", matrix[:, 1:], kinetic)
        else:
            raise ValueError("invalid temporal operator shape")
        if add_mean:
            kinetic[..., 0] += mean[None]
        result = torch.cat((rho.flatten(1), kinetic.flatten(1)), dim=1).to(dtype=values.dtype)
        if not torch.isfinite(result).all():
            raise ValueError("temporal transform overflowed the input dtype")
        return result

    def encode(self, reduced_values):
        return self._transform(reduced_values, self.inverse_square_root, subtract_mean=True)

    def decode(self, white_values):
        return self._transform(white_values, self.square_root, add_mean=True)

    def posterior_mean(self, noisy_values, sigma):
        """Exact fixed-Gaussian mean for y=x+sigma*epsilon in fit coordinates.

        Uses the FLOORED C, not its empirical singular version. This is a
        train-fitted Gaussian baseline, not evidence that the target is Gaussian.
        All spatial modes share C; only kinetic DC can have a nonzero mean.
        """
        noisy_values = _values(noisy_values, self.config)
        levels = torch.as_tensor(sigma, dtype=torch.float64, device=noisy_values.device)
        if levels.ndim == 0:
            levels = levels.expand(len(noisy_values))
        if levels.shape != (len(noisy_values),) or not torch.isfinite(levels).all() or torch.any(levels < 0):
            raise ValueError("sigma must be a finite nonnegative scalar or one level per case")
        eigenvalues = self.regularized_eigenvalues.to(device=noisy_values.device, dtype=torch.float64)
        vectors = self.eigenvectors.to(device=noisy_values.device, dtype=torch.float64)
        # sqrt(lambda)/sigma avoids squaring a potentially huge finite sigma.
        ratios = levels[:, None, None] / eigenvalues.sqrt()[None]
        weights = 1. / (1. + ratios.square())
        matrices = (vectors[None] * weights[:, :, None, :]) @ vectors.transpose(-1, -2)[None]
        return self._transform(noisy_values, matrices, subtract_mean=True, add_mean=True)

    def report(self):
        time = self.config.control_count
        identity = torch.eye(time, device=self.square_root.device, dtype=self.square_root.dtype)
        covariance = self.square_root @ self.square_root.transpose(-1, -2)
        whitened = self.inverse_square_root @ covariance @ self.inverse_square_root.transpose(-1, -2)
        names = ["rho", *(f"kinetic_channel_{index:03d}" for index in range(self.config.hermite_channels))]
        report = {"schema": SCHEMA, "fit_metadata": deepcopy(self.fit_metadata), "group_order": names,
            "full_rank": True, "subspace_or_frequency_removal": False,
            "data_value_clipping": False, "native_sampler_asset": False,
            "spatial_operator": "same_temporal_matrix_for_every_spatial_mode_within_group",
            "centering": "kinetic_spatial_DC_train_mean_only" if self.fit_metadata["center_kinetic_dc"] else "none",
            "kinetic_dc_mean": self.kinetic_dc_mean.detach().cpu().tolist(),
            "raw": _statistics(self.raw_second_moment.double()),
            "centered": _statistics(self.second_moment.double()),
            "dct_convention": "orthonormal_DCT_II_on_discrete_time_controls_not_physical_f_spectrum",
            "eigenvalue_floor_formula": "max(relative_floor^2*trace(C)/T, absolute_floor^2)",
            "eigenvalue_floor": self.eigenvalue_floor.detach().cpu().tolist(),
            "empirical_eigenvalues": self.eigenvalues.detach().cpu().tolist(),
            "regularized_eigenvalues": self.regularized_eigenvalues.detach().cpu().tolist(),
            "regularized_condition_number": (self.regularized_eigenvalues[:, -1] / self.regularized_eigenvalues[:, 0]).detach().cpu().tolist(),
            "factor_inverse_max_abs_error": float((self.square_root @ self.inverse_square_root - identity).abs().max()),
            "whitened_regularized_noise_trace": whitened.diagonal(dim1=-2, dim2=-1).sum(-1).detach().cpu().tolist(),
            "expected_whitened_noise_trace_per_group": time,
            "coordinate_dimension": self.config.latent_dimension,
            "decode_noise_covariance": "regularized_C_temporal_per_spatial_mode",
            "posterior_model": "train_fitted_fixed_temporal_Gaussian_not_true_data_distribution",
            "statistics_are_physical_validation": False}
        json.dumps(report, allow_nan=False)
        return report

    def to_payload(self):
        return {"schema": SCHEMA, "config": self.config.to_dict(),
                "second_moment": self.second_moment.detach().cpu().tolist(),
                "raw_second_moment": self.raw_second_moment.detach().cpu().tolist(),
                "kinetic_dc_mean": self.kinetic_dc_mean.detach().cpu().tolist(),
                "fit_metadata": deepcopy(self.fit_metadata)}

    @classmethod
    def from_payload(cls, payload):
        fields = {"schema", "config", "second_moment", "raw_second_moment", "kinetic_dc_mean", "fit_metadata"}
        if not isinstance(payload, Mapping) or set(payload) != fields or payload["schema"] != SCHEMA:
            raise ValueError("unexpected temporal diagnostic payload fields/schema")
        return cls(VPFPValueEDMConfig.from_dict(payload["config"]), payload["second_moment"],
                   payload["raw_second_moment"], payload["kinetic_dc_mean"], fit_metadata=payload["fit_metadata"])
