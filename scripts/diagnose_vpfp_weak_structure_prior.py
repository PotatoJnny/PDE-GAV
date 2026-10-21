"""Small, paired low-noise validation diagnostic; never a generation audit.

Call run_structure_denoising_diagnostic AFTER every experiment best checkpoint
has been frozen. This module does not train, sample trajectories, fit geometry,
read development reference features, or invoke a PDE solver.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import sys
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pde_agent_verifier_solver.plugins.vpfp.diffusion_audit import _strata
from pde_agent_verifier_solver.plugins.vpfp.diffusion_data import load_split, read_manifest
from pde_agent_verifier_solver.plugins.vpfp.structured_diffusion import _fourier_basis, load_vpfp_diffusion_sampler
from pde_agent_verifier_solver.plugins.vpfp.temporal_covariance_diffusion import VPFPTemporalCovarianceNormalizer
from pde_agent_verifier_solver.plugins.vpfp.value_control_diffusion import _value_split, extract_value_controls

SIGMAS = (.03, .1, .3)
SCHEMA = "vpfp_weak_structure_paired_low_sigma_diagnostic_v1"


def _file_sha(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _write_json(path, value):
    # The enclosing output directory is exclusive to this diagnostic.
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _assert_hashes(frozen):
    pass


def select_validation_rows(conditions, sample_ids, bin_edges, *, sample_count=8):
    """Select one smallest equation ID in each of the eight joint bins."""
    if sample_count != 8:
        raise ValueError("this bounded diagnostic requires exactly eight validation cases")
    ids = np.asarray(sample_ids)
    if (ids.ndim != 1 or ids.dtype.kind not in "iu" or len(ids) != len(conditions)
            or np.any(ids < 0) or len(set(map(int, ids))) != len(ids)):
        raise ValueError("validation equation IDs must be unique nonnegative integers")
    if len(bin_edges) != 2 or tuple(len(edge) for edge in bin_edges) != (5, 3):
        raise ValueError("validation diagnostic requires 4 by 2 condition bins")
    bins = _strata(conditions, tuple(np.asarray(edge, dtype=float) for edge in bin_edges))
    rows = []
    for group in range(8):
        selected = np.flatnonzero(bins == group)
        if not len(selected):
            raise ValueError(f"validation has no case in condition stratum {group}")
        rows.append(int(selected[np.argmin(ids[selected])]))
    return np.asarray(rows, dtype=np.int64)


def _mode_bands(maximum_mode, *, constant):
    modes = [(0, 0)] if constant else []
    for kx in range(-maximum_mode, maximum_mode + 1):
        for ky in range(-maximum_mode, maximum_mode + 1):
            if kx > 0 or (kx == 0 and ky > 0):
                modes.extend(((kx, ky), (kx, ky)))
    return {name: torch.tensor([predicate(k) for k in modes], dtype=torch.bool) for name, predicate in (
        ("dc", lambda k: k == (0, 0)),
        ("low_box_non_dc", lambda k: k != (0, 0) and max(abs(k[0]), abs(k[1])) <= 2),
        ("rest", lambda k: max(abs(k[0]), abs(k[1])) > 2))}


def _rms(values, dimensions=None):
    squared = values.double().square()
    return torch.sqrt(squared.mean() if dimensions is None else squared.mean(dim=dimensions))


def _finite_values(prediction, target, config):
    if (prediction.shape != target.shape or prediction.ndim != 2 or not len(prediction)
            or prediction.shape[1] != config.latent_dimension
            or not torch.isfinite(prediction).all() or not torch.isfinite(target).all()):
        raise ValueError("diagnostic requires matching finite nonempty reduced value arrays")


def value_band_diagnostics(prediction, target, config):
    """Per-trajectory errors, with exact source Fourier-pair ordering.

    These are coefficient-value errors, NOT errors in positive decoded f.
    rho has no DC coefficient. No band is dropped from the model or result.
    """
    _finite_values(prediction, target, config)
    predicted, expected = _value_split(prediction, config), _value_split(target, config)
    result = {}
    for index, (group, maximum, constant) in enumerate((
            ("rho", config.rho_max_mode, False), ("kinetic", config.kinetic_max_mode, True))):
        result[group] = {}
        for band, mask in _mode_bands(maximum, constant=constant).items():
            count = int(mask.sum())
            if not count:
                result[group][band] = {"available": False, "fourier_value_count": 0}
                continue
            mask = mask.to(prediction.device)
            p, t = predicted[index][..., mask], expected[index][..., mask]
            error = p - t
            dimensions = tuple(range(1, error.ndim))
            per_time_dimensions = tuple(axis for axis in range(error.ndim) if axis != error.ndim - 2)
            result[group][band] = {"available": True, "fourier_value_count": count,
                "rmse": float(_rms(error)), "target_rms": float(_rms(t)),
                "prediction_rms": float(_rms(p)), "mean_signed_error": float(error.double().mean()),
                "per_case_rmse": _rms(error, dimensions).cpu().tolist(),
                "per_control_time_rmse": _rms(error, per_time_dimensions).cpu().tolist()}
    return result


def amplitude_neighbor_diagnostics(prediction, target, config, conditions, *, decoder_amplitude_scale=1.):
    """Normalized amplitude coupling, NOT complete physical momentum j.

    For each axis, 2 sqrt(T) sum sqrt(n+1) a_n a_(n+e) / sum a_n^2.
    We omit rho and the Maxwellian-floor factor and do NOT evaluate f. Spatial
    points are the minimal kinetic Fourier grid; nonlinear ratios on this
    grid are descriptive, not a velocity/spatial convergence certificate.
    """
    _finite_values(prediction, target, config)
    if not math.isfinite(decoder_amplitude_scale) or decoder_amplitude_scale <= 0:
        raise ValueError("decoder amplitude scale must be finite positive")
    theta = torch.as_tensor(conditions, dtype=prediction.dtype, device=prediction.device)
    if theta.shape != (len(prediction), 2) or not torch.isfinite(theta).all() or torch.any(theta[:, 1] <= 0):
        raise ValueError("finite positive temperatures are required")
    degree = int((math.sqrt(8 * config.hermite_channels + 1) - 3) / 2)
    pairs = tuple((nx, total - nx) for total in range(degree + 1) for nx in range(total + 1))
    if len(pairs) != config.hermite_channels:
        raise ValueError("neighbor diagnostic requires a complete total-degree Hermite layout")
    mapping = {pair: i for i, pair in enumerate(pairs)}
    grid = 2 * config.kinetic_max_mode + 1
    basis = _fourier_basis(config.kinetic_max_mode, grid, constant=True, normalized=True).to(prediction)

    def coupling(values):
        coefficients = _value_split(values, config)[1]
        field = torch.einsum("bhtm,mxy->bhtxy", coefficients, basis).double()
        norm2 = field.square().sum(dim=1)
        zero = norm2 == 0
        fallback = norm2 * decoder_amplitude_scale**2 <= torch.finfo(values.dtype).eps**2
        denominator = norm2.clamp_min(torch.finfo(field.dtype).tiny)
        axes = []
        for axis in range(2):
            left, right, weights = [], [], []
            for i, pair in enumerate(pairs):
                neighbor = list(pair)
                neighbor[axis] += 1
                j = mapping.get(tuple(neighbor))
                if j is not None:
                    left.append(i); right.append(j); weights.append(math.sqrt(pair[axis] + 1))
            if left:
                product = field[:, left] * field[:, right]
                weighted = product * field.new_tensor(weights)[None, :, None, None, None]
                numerator = 2 * theta[:, 1].double().sqrt()[:, None, None, None] * weighted.sum(dim=1)
                # The qualified direct decoder substitutes its ground state
                # below eps; that state has exactly zero first moment.
                axes.append(torch.where(fallback, 0., numerator / denominator))
            else:
                axes.append(torch.zeros_like(norm2))
        return axes, int(zero.sum()), int(fallback.sum())

    predicted, predicted_zero, predicted_fallback = coupling(prediction)
    expected, expected_zero, expected_fallback = coupling(target)
    report = {"definition": "normalized_amplitude_neighbor_velocity_proxy_NOT_physical_j",
        "omitted_factors": ["rho", "1-minus-Maxwellian-floor"], "new_physical_evaluation": False,
        "grid_size": grid, "target_zero_norm_points": expected_zero,
        "prediction_zero_norm_points": predicted_zero,
        "target_decoder_ground_state_fallback_points": expected_fallback,
        "prediction_decoder_ground_state_fallback_points": predicted_fallback,
        "decoder_amplitude_scale_cancels_except_at_fallback": decoder_amplitude_scale, "axes": {}}
    for axis, name in enumerate(("x", "y")):
        p, t = predicted[axis], expected[axis]
        coherent_p, coherent_t = p.mean(dim=(-2, -1)), t.mean(dim=(-2, -1))
        report["axes"][name] = {"pointwise_rmse": float(_rms(p - t)),
            "target_pointwise_rms": float(_rms(t)), "prediction_pointwise_rms": float(_rms(p)),
            "coherent_mean_rmse": float(_rms(coherent_p - coherent_t)),
            "target_coherent_mean_rms": float(_rms(coherent_t)),
            "prediction_coherent_mean_rms": float(_rms(coherent_p)),
            "per_control_time_coherent_mean_rmse": _rms(coherent_p - coherent_t, (0,)).cpu().tolist()}
    return report


def _noise_for_ids(ids, dimension, seed):
    rows, seeds = [], []
    for sample_id in ids:
        digest = sha256(f"{SCHEMA}:{seed}:{int(sample_id)}".encode()).digest()
        local_seed = int.from_bytes(digest[:8], "little") % (2**63 - 1)
        seeds.append(local_seed)
        rows.append(torch.randn(dimension, dtype=torch.float32,
            generator=torch.Generator(device="cpu").manual_seed(local_seed)))
    return torch.stack(rows), seeds


def _normalized_binding(sampler, manifest, train_ids):
    normalizer = sampler.normalizer
    if (not isinstance(normalizer, VPFPTemporalCovarianceNormalizer)
            or normalizer.covariance_strength != .5 or normalizer.rho_policy != "identity"
            or normalizer.temporal.fit_metadata["center_kinetic_dc"] is not True):
        raise ValueError("all diagnostic models must use the frozen centered half covariance geometry")
    if (sampler.identity["corpus_manifest_sha256"] != manifest["manifest_sha256"]
            or sampler.identity["base_decoder_sha256"] != manifest["base_decoder_sha256"]
            or normalizer.fit_metadata["sample_ids"] != list(map(int, train_ids))):
        raise ValueError("checkpoint corpus, decoder, or train-only geometry identity differs")
    return _json_sha(normalizer.to_payload())


def _decoder_amplitude_contract(manifest, config):
    """Qualify the exact direct decoder path; never guess away a base term."""
    decoder = manifest["provenance"]["decoder_config"]
    degree = decoder["hermite_total_degree"]
    if (decoder.get("structured_spectral_latent") is not True
            or decoder.get("structured_direct_coefficients") is not True
            or decoder.get("structured_residual_scale") != 0.
            or decoder.get("structured_time_basis") != "piecewise_cubic_hermite_v1"
            or decoder.get("structured_time_control_count") != config.control_count
            or decoder.get("structured_local_channels") != config.hermite_channels
            or (degree + 1) * (degree + 2) // 2 != config.hermite_channels
            or decoder.get("structured_local_spatial_mode") != config.kinetic_max_mode
            or decoder.get("dtype") != "float32"):
        raise ValueError("amplitude proxy requires the qualified direct zero-residual float32 decoder layout")
    scale = float(decoder["latent_input_scale"]) * float(decoder["structured_local_scale"]) / math.sqrt(config.kinetic_modes)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("invalid direct decoder common amplitude scale")
    return {"decoder_config_sha256": _json_sha(decoder), "raw_control_to_amplitude_common_scale": scale,
        "coefficient_anchor_or_base_residual_added": False, "micro_network_evaluated": False,
        "time_locations": "control_knots_only_cubic_value_basis_equals_identity",
        "scale_in_normalized_neighbor_ratio": "cancels_except_ground_state_fallback",
        "floor_mixture_omitted_from_proxy": decoder["velocity_floor_mixture"]}


def run_structure_denoising_diagnostic(checkpoints, corpus, output, *, seed, device="cuda",
        sample_count=8, batch_size=2, expected_corpus_sha256=None):
    """Diagnose frozen bests, returning and saving a finite development report.

    checkpoints maps unique arm/replica labels to {checkpoint, sha256}. The
    driver must supply ALL frozen bests, after checkpoint selection has ended.
    An unsuccessful run preserves its partial diagnostics and precise stage.
    """
    if Path(output).is_symlink():
        raise ValueError("diagnostic output must not be a symlink")
    corpus, output = Path(corpus).resolve(strict=True), Path(output).resolve()
    if output.exists() or output.is_relative_to(corpus):
        raise ValueError("choose a fresh diagnostic output outside the immutable corpus")
    if (sample_count != 8 or not isinstance(batch_size, int) or isinstance(batch_size, bool)
            or not 1 <= batch_size <= 8 or not isinstance(seed, int) or isinstance(seed, bool) or seed < 0):
        raise ValueError("diagnostic needs eight cases, batch size 1..8, and a nonnegative integer seed")
    if not isinstance(checkpoints, dict) or len(checkpoints) < 2:
        raise ValueError("supply at least two frozen checkpoint entries for a paired comparison")
    inventory = {}
    for label, row in checkpoints.items():
        if (not isinstance(label, str) or not label or not isinstance(row, dict)
                or not {"checkpoint", "sha256"}.issubset(row)):
            raise ValueError("checkpoint entries require a label, checkpoint path, and SHA256")
        path = Path(row["checkpoint"]).resolve(strict=True)
        digest = row["sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid frozen checkpoint SHA256")
        inventory[label] = {"checkpoint": str(path), "sha256": digest}
    output.mkdir(parents=True, exist_ok=False)
    stage, results = "input_binding", {}
    report = {"schema": SCHEMA, "execution_passed": False, "evidence_role": "previously_opened_validation_development_diagnostic",
        "blind_test": False, "admitted": False, "used_for_checkpoint_selection": False,
        "generated_samples": 0, "new_pde_solves": 0, "normalizer_fit_performed": False,
        "sigmas": list(SIGMAS), "sample_count": sample_count, "checkpoints": inventory,
        "caller_contract": "all_experiment_best_checkpoints_frozen_before_call",
        "band_definition": "DC; non-DC axis box max(abs(kx),abs(ky))<=2; complement (NOT physical-audit radial bands)",
        "interpretation": "paired low-noise reconstruction only; conditional-mean shrinkage is not by itself a generative failure",
        "free_generation_comparison": "consult separately saved physical spatial-momentum-diagnostic.json; proxy here is not physical j",
        "models": results}
    try:
        manifest = read_manifest(corpus, verify_hashes=False)
        if expected_corpus_sha256 is not None and manifest["manifest_sha256"] != expected_corpus_sha256:
            raise ValueError("diagnostic corpus differs from frozen experiment plan")
        frozen = {str(corpus / "manifest.json"): _file_sha(corpus / "manifest.json"),
            str(Path(__file__).resolve()): _file_sha(__file__)}
        for entry in manifest["splits"]["validation"]["files"].values():
            path = (corpus / entry["path"]).resolve(strict=True)
            if not path.is_relative_to(corpus):
                raise ValueError("validation input leaves the corpus")
            frozen[str(path)] = entry["sha256"]
        frozen.update({entry["checkpoint"]: entry["sha256"] for entry in inventory.values()})
        _assert_hashes(frozen)
        validation = load_split(corpus, "validation", verify_hashes=False)
        training = load_split(corpus, "train", verify_hashes=False)
        edges = (np.linspace(*manifest["collision_nu_range"], 5), np.linspace(*manifest["temperature_range"], 3))
        rows = select_validation_rows(validation.conditions, validation.sample_ids, edges, sample_count=sample_count)
        ids = validation.sample_ids[rows].tolist()
        raw = torch.from_numpy(np.array(validation.latents[rows], copy=True))
        conditions = np.array(validation.conditions[rows], copy=True)
        if set(map(int, ids)).intersection(map(int, training.sample_ids)):
            raise ValueError("validation and train equation IDs overlap")
        report.update(corpus_manifest_sha256=manifest["manifest_sha256"], base_decoder_sha256=manifest["base_decoder_sha256"],
            validation_rows=rows.tolist(), validation_ids=ids, validation_conditions=conditions.tolist(),
            sample_unit="eight_complete_trajectories_one_per_condition_stratum",
            selected_raw_answers_sha256=sha256(raw.numpy().tobytes() + conditions.tobytes()).hexdigest(),
            frozen_input_sha256=frozen, seed=seed)
        common_geometry = None
        for label, entry in inventory.items():
            stage = "denoise:" + label
            _assert_hashes(frozen)
            sampler = load_vpfp_diffusion_sampler(entry["checkpoint"], entry["sha256"], device=device, steps=2)
            geometry_sha = _normalized_binding(sampler, manifest, training.sample_ids)
            if common_geometry is not None and common_geometry != geometry_sha:
                raise ValueError("paired models do not share identical normalizer payloads")
            common_geometry = geometry_sha
            normalizer, model = sampler.normalizer, sampler.model
            model.eval()
            config = model.config
            decoder_contract = _decoder_amplitude_contract(manifest, config)
            report["decoder_amplitude_contract"] = decoder_contract
            noise, noise_seeds = _noise_for_ids(ids, config.latent_dimension, seed)
            report.update(normalizer_payload_sha256=geometry_sha, noise_seeds=noise_seeds,
                white_noise_sha256=sha256(noise.numpy().tobytes()).hexdigest(),
                noise_policy="same_CPU_float32_standard_normal_per_equation_ID_across_all_models_and_sigmas")
            record = results[label] = {"normalizer_payload_sha256": geometry_sha, "sigmas": {}}
            with torch.inference_mode():
                clean_raw = extract_value_controls(raw, config)
                clean_parent = normalizer.parent.encode(raw.to(device)).cpu()
                clean_white = normalizer.encode(raw.to(device))
                for sigma in SIGMAS:
                    reconstructed, corrupted = [], []
                    for start in range(0, sample_count, batch_size):
                        stop = min(sample_count, start + batch_size)
                        noisy = clean_white[start:stop] + sigma * noise[start:stop].to(device)
                        sigma_rows = noisy.new_full((stop - start,), sigma)
                        prediction = model.denoise(noisy, sigma_rows, conditions[start:stop])
                        if not torch.isfinite(prediction).all():
                            raise FloatingPointError("nonfinite low-sigma model denoising output")
                        reconstructed.append(normalizer.decode(prediction).cpu())
                        corrupted.append(normalizer.decode(noisy).cpu())
                    measures = {}
                    for name, full in (("noisy_input", torch.cat(corrupted)), ("denoised", torch.cat(reconstructed))):
                        values = extract_value_controls(full, config)
                        parent_values = normalizer.parent.encode(full.to(device)).cpu()
                        measures[name] = {"raw_reduced_value": value_band_diagnostics(values, clean_raw, config),
                            "parent_value_coordinates": value_band_diagnostics(parent_values, clean_parent, config),
                            "amplitude_neighbor_proxy": amplitude_neighbor_diagnostics(values, clean_raw, config, conditions,
                                decoder_amplitude_scale=decoder_contract["raw_control_to_amplitude_common_scale"])}
                    record["sigmas"][str(sigma)] = measures
                    _write_json(output / "result.partial.json", report)
            del sampler, model, normalizer
        stage = "final_input_revalidation"
        _assert_hashes(frozen)
        report["execution_passed"] = True
        report["all_inputs_revalidated"] = True
    except Exception as error:
        report.update(failed_stage=stage, error=repr(error), traceback=traceback.format_exc(), partial_artifacts_preserved=True)
    _write_json(output / "result.json", report)
    return report
