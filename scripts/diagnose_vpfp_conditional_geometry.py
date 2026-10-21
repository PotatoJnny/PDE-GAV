"""Stage-zero initial/future coordinates, not a trained conditional sampler.

Split the clean t0 VALUES before any temporal whitening. Fit nothing new:
take a frozen training-only temporal metric, use its Schur complement on
the free future variables, and lift slopes only after restoring all values.
The Gaussian regression here is a coordinate preconditioner, NOT a claim
that the actual conditional VPFP trajectory law is Gaussian.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pde_agent_verifier_solver.plugins.vpfp.diffusion_data import load_split, read_manifest
from pde_agent_verifier_solver.plugins.vpfp.structured_diffusion import load_vpfp_diffusion_sampler
from pde_agent_verifier_solver.plugins.vpfp.temporal_covariance_diffusion import VPFPTemporalCovarianceNormalizer
from pde_agent_verifier_solver.plugins.vpfp.value_control_diffusion import _value_split
from diagnose_vpfp_weak_structure_prior import select_validation_rows, _file_sha, _json_sha, _write_json


class ConditionalFutureCoordinates:
    """Research codec with independent clean context and full-rank future noise.

    This class has no deployed checkpoint schema or sampler registration.
    Context has one value per spatial/Hermite mode. Future has T-1 values
    per mode, retaining the original times t[1:], not a new [0,1] grid.
    """
    def __init__(self, normalizer):
        if not isinstance(normalizer, VPFPTemporalCovarianceNormalizer):
            raise TypeError("conditional geometry requires frozen temporal normalization")
        normalizer._check_buffer_dtype()
        self.normalizer = normalizer
        self.config = cfg = normalizer.config
        self.normalizer_sha256 = _json_sha(normalizer.to_payload())
        root = getattr(normalizer, "mixture_square_root", normalizer.temporal.square_root).detach().clone()
        if normalizer.rho_policy == "identity":
            root[0] = torch.eye(cfg.control_count, dtype=root.dtype, device=root.device)
        self.covariance = root @ root.transpose(-1, -2)
        self.regression = self.covariance[:, 1:, 0] / self.covariance[:, 0, 0, None]
        schur = self.covariance[:, 1:, 1:] - self.regression[:, :, None] * self.covariance[:, None, 0, 1:]
        schur = (schur + schur.transpose(-1, -2)) / 2
        eigenvalues, vectors = torch.linalg.eigh(schur)
        if not torch.isfinite(eigenvalues).all() or torch.any(eigenvalues <= 0):
            raise ValueError("conditional future Schur complement must remain strictly SPD")
        self.eigenvalues = eigenvalues
        self.root = (vectors * eigenvalues.sqrt()[:, None, :]) @ vectors.transpose(-1, -2)
        self.inverse = (vectors * eigenvalues.rsqrt()[:, None, :]) @ vectors.transpose(-1, -2)
        self.schur = schur
        self.context_dimension = cfg.rho_modes + cfg.hermite_channels * cfg.kinetic_modes
        self.future_dimension = self.context_dimension * (cfg.control_count - 1)

    def _centered(self, parent_values):
        rho, kinetic = _value_split(parent_values, self.config)
        kinetic = kinetic.double().clone()
        kinetic[..., 0] -= self.normalizer.temporal.kinetic_dc_mean.to(kinetic)[None]
        return rho.double(), kinetic

    def encode(self, full_raw):
        parent = self.normalizer.parent.encode(full_raw)
        rho, kinetic = _value_split(parent, self.config)
        # No slopes, source, future controls, or whole-time white coordinates
        # enter the context. Parent scales are frozen training statistics.
        context = torch.cat((rho[:, 0], kinetic[:, :, 0].flatten(1)), dim=1)
        rho, kinetic = self._centered(parent)
        regression, inverse = self.regression.to(parent.device), self.inverse.to(parent.device)
        rho_future = rho[:, 1:] - regression[0, :, None] * rho[:, :1]
        kinetic_future = kinetic[:, :, 1:] - regression[None, 1:, :, None] * kinetic[:, :, :1]
        white_rho = torch.einsum("ts,bsm->btm", inverse[0], rho_future)
        white_kinetic = torch.einsum("hts,bhsm->bhtm", inverse[1:], kinetic_future)
        white = torch.cat((white_rho.flatten(1), white_kinetic.flatten(1)), dim=1).to(parent.dtype)
        if not torch.isfinite(white).all():
            raise ValueError("conditional coordinate encoding overflow")
        return context, white

    def decode(self, context, white_future):
        cfg = self.config
        if (context.ndim != 2 or context.shape[1] != self.context_dimension
                or white_future.shape != (len(context), self.future_dimension)
                or context.device != white_future.device or context.dtype != white_future.dtype
                or not torch.isfinite(context).all() or not torch.isfinite(white_future).all()):
            raise ValueError("finite aligned initial context and free future arrays required")
        count, future_count = len(context), cfg.control_count - 1
        rho0 = context[:, :cfg.rho_modes].double()
        kin0 = context[:, cfg.rho_modes:].reshape(count, cfg.hermite_channels, cfg.kinetic_modes).double().clone()
        mean = self.normalizer.temporal.kinetic_dc_mean.to(device=context.device)
        kin0[..., 0] -= mean[None, :, 0]
        rho_end = future_count * cfg.rho_modes
        rwhite = white_future[:, :rho_end].reshape(count, future_count, cfg.rho_modes).double()
        kwhite = white_future[:, rho_end:].reshape(count, cfg.hermite_channels, future_count, cfg.kinetic_modes).double()
        root, regression = self.root.to(context.device), self.regression.to(context.device)
        rho_future = torch.einsum("ts,bsm->btm", root[0], rwhite) + regression[0, :, None] * rho0[:, None]
        kin_future = torch.einsum("hts,bhsm->bhtm", root[1:], kwhite) + regression[None, 1:, :, None] * kin0[:, :, None]
        rho = torch.cat((rho0[:, None], rho_future), dim=1)
        kinetic = torch.cat((kin0[:, :, None], kin_future), dim=2)
        kinetic[..., 0] += mean[None]
        parent = torch.cat((rho.flatten(1), kinetic.flatten(1)), dim=1).to(context.dtype)
        return self.normalizer.parent.decode(parent)

    def report(self):
        identity = torch.eye(self.config.control_count - 1, dtype=self.root.dtype, device=self.root.device)
        whitened = self.inverse @ self.schur @ self.inverse.transpose(-1, -2)
        return {"schema": "vpfp_initial_condition_future_schur_geometry_probe_v1",
            "source_normalizer_sha256": self.normalizer_sha256,
            "context_dimension": self.context_dimension, "future_noise_dimension": self.future_dimension,
            "full_decoder_dimension": self.config.full_dimension,
            "minimum_schur_eigenvalue": float(self.eigenvalues.min()),
            "maximum_group_condition_number": float((self.eigenvalues[:, -1] / self.eigenvalues[:, 0]).max()),
            "whitening_max_abs_error": float((whitened - identity).abs().max()),
            "context": "clean_parent_RMS_t0_values_only_before_temporal_whitening",
            "context_slopes_or_future_values": False, "future_noise_full_rank": True,
            "additional_eigenvalue_floor": False, "train_statistics_refitted": False,
            "future_times": torch.linspace(*self.config.time_domain, self.config.control_count, dtype=torch.float64)[1:].tolist(),
            "decode": "inverse_future_Schur_then_restore_t0_then_original_value_control_slope_lift",
            "conditional_gaussian_target_assumed": False,
            "trained_conditional_model": False, "native_asset_registered": False,
            "scientific_distribution_validation": False}


def run(checkpoint, corpus_root, output, *, device="cuda"):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("real corpus geometry probe requires a Slurm compute allocation")
    raw_output = Path(output)
    output, checkpoint, corpus_root = raw_output.resolve(), Path(checkpoint).resolve(), Path(corpus_root).resolve()
    if raw_output.is_symlink() or output.exists() or output.is_relative_to(corpus_root):
        raise ValueError("choose a fresh probe output outside source corpus")
    output.mkdir(parents=True, exist_ok=False)
    report = {"execution_passed": False, "new_pde_solves": 0, "training_run": False}
    try:
        manifest = read_manifest(corpus_root, verify_hashes=True)
        checkpoint_sha = _file_sha(checkpoint)
        sampler = load_vpfp_diffusion_sampler(checkpoint, checkpoint_sha, device=device)
        if (sampler.identity["corpus_manifest_sha256"] != manifest["manifest_sha256"]
                or sampler.identity["base_decoder_sha256"] != manifest["base_decoder_sha256"]):
            raise ValueError("checkpoint/corpus/decoder identity mismatch")
        train = load_split(corpus_root, "train", verify_hashes=False)
        if sampler.normalizer.fit_metadata["sample_ids"] != list(map(int, train.sample_ids)):
            raise ValueError("normalizer fit IDs differ from corpus training IDs")
        geometry = ConditionalFutureCoordinates(sampler.normalizer)
        report.update(geometry.report(), checkpoint_sha256=checkpoint_sha,
            corpus_manifest_sha256=manifest["manifest_sha256"], splits={})
        edges = (np.linspace(*manifest["collision_nu_range"], 5), np.linspace(*manifest["temperature_range"], 3))
        selected_inputs = {}

        def selected_sha(split, indices):
            digest = sha256()
            for values in (split.latents[indices], split.conditions[indices], split.sample_ids[indices]):
                digest.update(np.ascontiguousarray(values).tobytes())
            return digest.hexdigest()

        for split_name in ("train", "validation"):
            split = train if split_name == "train" else load_split(corpus_root, split_name, verify_hashes=False)
            indices = select_validation_rows(split.conditions, split.sample_ids, edges)
            selected_inputs[split_name] = (indices.copy(), selected_sha(split, indices))
            raw = torch.from_numpy(np.array(split.latents[indices], copy=True)).to(device)
            context, future = geometry.encode(raw)
            restored = geometry.decode(context, future)
            noisy = future + torch.randn(future.shape, generator=torch.Generator(device=device).manual_seed(902606), device=device)
            changed = geometry.decode(context, noisy)
            new_context, _ = geometry.encode(changed)
            relative = float(torch.linalg.vector_norm((restored - raw).double()) / torch.linalg.vector_norm(raw.double()))
            context_error = float((new_context - context).abs().max())
            if relative > 2.e-5 or not torch.allclose(new_context, context, rtol=2.e-6, atol=2.e-6):
                raise ValueError("conditional codec roundtrip or immutable t0 check failed")
            # A physically fixed t0 does NOT fix its finite-difference slope:
            # that slope correctly depends on the sampled next value.
            report["splits"][split_name] = {"equation_ids": list(map(int, split.sample_ids[indices])),
                "selected_raw_conditions_ids_sha256": selected_inputs[split_name][1],
                "full_ABI_roundtrip_relative_l2": relative, "future_noise_t0_max_abs_change_in_parent_units": context_error,
                "actual_future_white_RMS_change": float((noisy - future).double().square().mean().sqrt()),
                "one_case_per_joint_condition_bin": True}
        if read_manifest(corpus_root, verify_hashes=False)["manifest_sha256"] != manifest["manifest_sha256"]:
            raise ValueError("corpus manifest changed during geometry probe")
        report["selected_inputs_revalidated"] = True
        report["execution_passed"] = True
    except Exception as error:
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        _write_json(output / "result.json", report)
        raise
    _write_json(output / "result.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    run(arguments.checkpoint, arguments.corpus, arguments.output, device=arguments.device)
