"""Observe frozen native VPFP Heun samplers without changing their trajectory.

This is a small, previously-opened-validation mechanism diagnostic, NOT a
physical distribution audit. Intermediate xhat0 values are noisy-state
conditional estimates, not generated answers. Only final sigma-zero states
are samples; the 256-step result is a finer numerical comparison, not truth.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import re
import sys
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from diagnose_vpfp_weak_structure_prior import (
    _assert_hashes, _decoder_amplitude_contract, _file_sha, _json_sha,
    _mode_bands, _normalized_binding, _write_json,
    amplitude_neighbor_diagnostics, select_validation_rows,
    value_band_diagnostics,
)
from pde_agent_verifier_solver.plugins.vpfp.diffusion_data import load_split, read_manifest
from pde_agent_verifier_solver.plugins.vpfp.structured_diffusion import (
    _coefficients, _seeds, load_vpfp_diffusion_sampler,
)
from pde_agent_verifier_solver.plugins.vpfp.value_control_diffusion import (
    _value_split, extract_value_controls, value_control_slopes,
)

SCHEMA = "vpfp_native_reverse_joint_diagnostic_v1"
STEPS = (64, 128, 256)
OBSERVATION_SIGMAS = (20., 10., 3., 1., .3, .1, .03, .01, .002)


def _tensor_sha(value):
    array = value.detach().cpu().contiguous().numpy()
    return sha256(array.tobytes()).hexdigest()


def _noise_seeds(ids, seed):
    return [int.from_bytes(sha256(f"{SCHEMA}:{seed}:{int(case)}".encode()).digest()[:8],
                           "little") % (2**63) for case in ids]


def _schedule(model, steps):
    parameter = next(model.parameters())
    config = model.config
    ramp = torch.linspace(0., 1., steps, device=parameter.device, dtype=parameter.dtype)
    return (config.sigma_max**(1 / config.rho)
            + ramp * (config.sigma_min**(1 / config.rho) - config.sigma_max**(1 / config.rho)))**config.rho


def observe_native_sample(model, conditions, seeds, *, steps, observation_sigmas=OBSERVATION_SIGMAS):
    """Return the unchanged native output plus selected first-evaluation xhat0.

    No extra denoiser evaluations, noise draws, state edits, or sampler-code
    copies are used. The observer is restored even when inference fails.
    Churn is zero, so a per-case seed completely fixes each initial state.
    """
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 2:
        raise ValueError("native observation needs integer steps >=2")
    seed_values = _seeds(seeds)
    if not observation_sigmas or any(isinstance(s, bool) or not isinstance(s, (int, float))
            or not math.isfinite(s) or s <= 0 for s in observation_sigmas):
        raise ValueError("observation sigmas must be finite positive numbers")
    if model.training:
        raise ValueError("native diagnostic requires an eval-mode frozen model")
    levels = _schedule(model, steps)
    # Different native grids are not nested. Save exact actual sigma, not just
    # a rounded requested label; endpoint comparisons are exact, interior ones
    # are descriptive nearest-log-sigma observations only.
    requested = {0: [], steps - 1: []}
    excluded = []
    for sigma in observation_sigmas:
        if not model.config.sigma_min <= sigma <= model.config.sigma_max:
            excluded.append(float(sigma))
            continue
        index = int((levels.double().log() - math.log(sigma)).abs().argmin())
        requested.setdefault(index, []).append(float(sigma))
    selected = {index: [] for index in sorted(requested)}
    initial, calls = [], 0
    calls_per_case = 2 * steps - 1
    had_instance_override = "denoise" in model.__dict__
    old_instance_override = model.__dict__.get("denoise")
    original = model.denoise

    def observed(noisy, sigma, condition):
        nonlocal calls
        case, local_call = divmod(calls, calls_per_case)
        if case >= len(seed_values) or len(noisy) != 1:
            raise ValueError("native sampler no longer follows its per-case Heun ABI")
        output = original(noisy, sigma, condition)
        if local_call % 2 == 0:
            index = local_call // 2
            if sigma.shape != (1,) or not torch.equal(sigma[0], levels[index]):
                raise ValueError("native first-evaluation sigma differs from the registered no-churn grid")
            if index == 0:
                initial.append(noisy.detach().cpu().clone())
            if index in selected:
                if not torch.isfinite(output).all():
                    raise FloatingPointError("nonfinite native conditional estimate")
                selected[index].append(output.detach().cpu().clone())
        calls += 1
        return output  # Return the original object, never its detached copy.

    model.denoise = observed
    try:
        with torch.inference_mode():
            final = model.sample(conditions, seed_values, steps=steps, churn=0.)
    finally:
        if had_instance_override:
            model.denoise = old_instance_override
        else:
            delattr(model, "denoise")
    if calls != len(seed_values) * calls_per_case or len(initial) != len(seed_values):
        raise ValueError("native denoiser evaluation count changed")
    _coefficients(final, model.config)
    return final, {
        "denoiser_evaluations": calls, "evaluations_per_case": calls_per_case,
        "initial_noisy_states_sha256": _tensor_sha(torch.cat(initial)),
        "initial_noisy_state_per_case_sha256": [_tensor_sha(row) for row in initial],
        "sigma_grid": levels.detach().cpu().tolist(),
        "excluded_observation_sigmas": excluded,
        "observation_rule": "nearest_native_log_sigma_first_evaluation_only_not_interpolated",
        "stages": [{"native_step": index, "actual_sigma": float(levels[index]),
                    "requested_sigmas": requested[index], "xhat0": torch.cat(selected[index])}
                   for index in selected],
    }


def coefficient_time_statistics(raw_values, config):
    """Original coefficient d/dphysical-time at knots, NOT dt(f) or dt(j)."""
    values = _coefficients(raw_values, config)
    duration = config.time_domain[1] - config.time_domain[0]
    tau = torch.linspace(0., 1., config.control_count, device=values.device, dtype=torch.float64)
    report = {"definition": "raw_value_coefficients_and_fixed_lift_d_dt_at_control_knots_NOT_physical_f_j_K",
              "physical_time": (config.time_domain[0] + duration * tau).cpu().tolist(), "groups": {}}
    for group, block, maximum, constant in (
            ("rho", _value_split(values, config)[0], config.rho_max_mode, False),
            ("kinetic", _value_split(values, config)[1], config.kinetic_max_mode, True)):
        derivative = value_control_slopes(block.double(), config) / duration
        group_report = report["groups"][group] = {}
        for name, mask in _mode_bands(maximum, constant=constant).items():
            if not int(mask.sum()):
                group_report[name] = {"available": False}
                continue
            value = block[..., mask.to(block.device)].double()
            dt = derivative[..., mask.to(block.device)]
            dimensions = tuple(range(1, dt.ndim))
            time_dimensions = tuple(i for i in range(dt.ndim) if i != dt.ndim - 2)
            entry = {"available": True, "value_rms": float(value.square().mean().sqrt()),
                     "dt_rms": float(dt.square().mean().sqrt()),
                     "per_case_dt_rms": dt.square().mean(dim=dimensions).sqrt().cpu().tolist(),
                     "per_control_time_dt_rms": dt.square().mean(dim=time_dimensions).sqrt().cpu().tolist()}
            for window, selected in (("early", tau <= .25), ("late", tau >= .5)):
                piece = dt[..., selected, :]
                entry[window + "_dt_rms"] = float(piece.square().mean().sqrt())
            group_report[name] = entry
    return report


def _neighbor_summary(values, config, conditions, scale):
    # Self-comparison only reuses the qualified decoder proxy implementation.
    # Remove all reconstruction-error/target labels: generated trajectories do
    # not share hidden answers with the condition-matched validation cases.
    result = amplitude_neighbor_diagnostics(values, values, config, conditions,
                                            decoder_amplitude_scale=scale)
    return {"definition": result["definition"], "omitted_factors": result["omitted_factors"],
            "grid_size": result["grid_size"], "physical_evaluation_performed": False,
            "zero_norm_points": result["prediction_zero_norm_points"],
            "decoder_ground_state_fallback_points": result["prediction_decoder_ground_state_fallback_points"],
            "axes": {axis: {"coherent_mean_rms": row["prediction_coherent_mean_rms"],
                             "pointwise_rms": row["prediction_pointwise_rms"]}
                     for axis, row in result["axes"].items()}}


def _summary(values, config, conditions, scale):
    return {"coefficient_time": coefficient_time_statistics(values, config),
            "amplitude_neighbor_proxy": _neighbor_summary(values, config, conditions, scale)}


def _inventory(checkpoints):
    if not isinstance(checkpoints, dict) or not checkpoints:
        raise ValueError("supply frozen checkpoint entries")
    entries = {}
    for label, row in checkpoints.items():
        if (not isinstance(label, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", label)
                or not isinstance(row, dict) or not {"checkpoint", "sha256"}.issubset(row)
                or not isinstance(row["sha256"], str) or not re.fullmatch(r"[a-f0-9]{64}", row["sha256"])):
            raise ValueError("checkpoint entries need safe labels, paths and lowercase SHA256")
        entries[label] = {"checkpoint": str(Path(row["checkpoint"]).resolve(strict=True)), "sha256": row["sha256"]}
    return entries


def run_reverse_joint_diagnostic(checkpoints, corpus, output, *, seed, device="cuda",
        sample_count=8, steps=STEPS, expected_corpus_sha256=None):
    """Save a bounded paired sampler report in a NEW output directory.

    The caller freezes all checkpoint paths first. All model/noise geometry
    and input files remain unchanged. Failure is saved and returned with
    execution_passed=False; the caller must check it before declaring success.
    """
    if Path(output).is_symlink():
        raise ValueError("diagnostic output must not be a symlink")
    corpus, output = Path(corpus).resolve(strict=True), Path(output).resolve()
    if output.exists() or output.is_relative_to(corpus):
        raise ValueError("choose a fresh output outside the immutable corpus")
    if (sample_count != 8 or isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63
            or tuple(steps) != STEPS or any(type(n) is not int for n in steps)):
        raise ValueError("diagnostic fixes eight validation conditions and 64/128/256 steps, with a nonnegative int64 seed")
    inventory = _inventory(checkpoints)
    output.mkdir(parents=True, exist_ok=False)
    stage = "input_binding"
    report = {"schema": SCHEMA, "execution_passed": False, "blind_test": False, "admitted": False,
              "used_for_checkpoint_selection": False, "normalizer_fit_performed": False, "new_pde_solves": 0,
              "new_physical_audit_performed": False,
              "evidence_role": "previously_opened_validation_conditions_sampler_mechanism_diagnostic",
              "sample_count": sample_count, "seed": seed, "steps": list(steps), "checkpoints": inventory,
              "noise_policy": "same_native_device_per_case_seed_and_initial_state_across_models_and_step_counts_churn_zero",
              "sample_unit": "eight_conditions_one_per_joint_stratum_same_conditions_reused_across_models_and_steps",
              "reference_role": "aggregate_validation_descriptions_NOT_paired_ground_truth_for_generated_answers",
              "stage_role": "first_evaluation_xhat0_conditional_estimates_of_noisy_states_NOT_generated_answers",
              "comparison_role": "256_steps_is_not_truth; interior_native_sigma_nodes_are_not_exactly_matched",
              "causal_limit": "convergence_can_bound_integrator_sensitivity_not_prove_model_capacity_or_high_sigma_cause",
              "models": {}}
    try:
        manifest = read_manifest(corpus, verify_hashes=False)
        if expected_corpus_sha256 is not None and manifest["manifest_sha256"] != expected_corpus_sha256:
            raise ValueError("diagnostic corpus differs from frozen plan")
        frozen = {str(corpus / "manifest.json"): _file_sha(corpus / "manifest.json"),
                  str(Path(__file__).resolve()): _file_sha(__file__),
                  str(ROOT / "scripts/diagnose_vpfp_weak_structure_prior.py"):
                      _file_sha(ROOT / "scripts/diagnose_vpfp_weak_structure_prior.py")}
        # No training-latent scan or train refit. Bind all validation bytes and
        # the small training identity files actually used to check fit lineage.
        for split in ("validation", "train"):
            for name, entry in manifest["splits"][split]["files"].items():
                if split == "train" and name == "coefficients":
                    continue
                path = (corpus / entry["path"]).resolve(strict=True)
                if not path.is_relative_to(corpus):
                    raise ValueError("diagnostic input leaves the corpus")
                frozen[str(path)] = entry["sha256"]
        frozen.update({row["checkpoint"]: row["sha256"] for row in inventory.values()})
        _assert_hashes(frozen)
        validation = load_split(corpus, "validation", verify_hashes=False)
        training = load_split(corpus, "train", verify_hashes=False)
        edges = (np.linspace(*manifest["collision_nu_range"], 5), np.linspace(*manifest["temperature_range"], 3))
        rows = select_validation_rows(validation.conditions, validation.sample_ids, edges, sample_count=sample_count)
        ids = validation.sample_ids[rows].tolist()
        if set(map(int, ids)).intersection(map(int, training.sample_ids)):
            raise ValueError("validation and train IDs overlap")
        conditions = np.array(validation.conditions[rows], copy=True)
        raw = torch.from_numpy(np.array(validation.latents[rows], copy=True))
        noise_seeds = _noise_seeds(ids, seed)
        report.update(corpus_manifest_sha256=manifest["manifest_sha256"], base_decoder_sha256=manifest["base_decoder_sha256"],
                      validation_rows=rows.tolist(), validation_ids=ids, validation_conditions=conditions.tolist(),
                      noise_seeds=noise_seeds, frozen_input_sha256=frozen,
                      selected_raw_answers_sha256=sha256(raw.numpy().tobytes() + conditions.tobytes()).hexdigest())
        common_geometry = common_initial = None
        for label, row in inventory.items():
            stage = "load:" + label
            _assert_hashes(frozen)
            sampler = load_vpfp_diffusion_sampler(row["checkpoint"], row["sha256"], device=device, steps=64, churn=0.)
            geometry = _normalized_binding(sampler, manifest, training.sample_ids)
            if common_geometry is not None and common_geometry != geometry:
                raise ValueError("paired checkpoints have different frozen noise geometry")
            common_geometry = geometry
            model, normalizer = sampler.model, sampler.normalizer
            model.eval()
            config = model.config
            contract = _decoder_amplitude_contract(manifest, config)
            scale = contract["raw_control_to_amplitude_common_scale"]
            report.update(normalizer_payload_sha256=geometry, decoder_amplitude_contract=contract)
            if "condition_matched_validation_summary" not in report:
                values = extract_value_controls(raw.to(device), config)
                report["condition_matched_validation_summary"] = _summary(values, config, conditions, scale)
            entry = report["models"][label] = {"normalizer_payload_sha256": geometry,
                "native_sampler_identity": sampler.identity, "by_steps": {}, "paired_numerical_differences": {}}
            final_values = {}
            with torch.inference_mode():
                for count in steps:
                    stage = f"sample:{label}:{count}"
                    final, observed = observe_native_sample(model, conditions, noise_seeds, steps=count)
                    if common_initial is not None and common_initial != observed["initial_noisy_states_sha256"]:
                        raise ValueError("paired native initial noisy states differ")
                    common_initial = observed["initial_noisy_states_sha256"]
                    record = entry["by_steps"][str(count)] = {key: value for key, value in observed.items() if key != "stages"}
                    record["stages"] = []
                    for observation in observed["stages"]:
                        estimate = observation.pop("xhat0")
                        values = extract_value_controls(normalizer.decode(estimate.to(device)), config)
                        observation.update(role=report["stage_role"], summary=_summary(values, config, conditions, scale))
                        record["stages"].append(observation)
                    values = extract_value_controls(normalizer.decode(final), config)
                    record["final"] = {"role": "native_sigma_zero_generated_answer",
                                       "normalized_sha256": _tensor_sha(final),
                                       "summary": _summary(values, config, conditions, scale)}
                    destination = output / f"{label}-steps-{count}-final-values.npy"
                    np.save(destination, values.detach().cpu().numpy(), allow_pickle=False)
                    record["final"]["raw_reduced_values"] = {"path": str(destination), "sha256": _file_sha(destination)}
                    final_values[count] = values.detach().cpu()
                    _write_json(output / "result.partial.json", report)
                    del observed, values, final
                for left, right in ((64, 128), (128, 256), (64, 256)):
                    a, b = final_values[left].to(device), final_values[right].to(device)
                    entry["paired_numerical_differences"][f"{left}_vs_{right}"] = {
                        "left_steps": left, "right_steps": right, "right_is_truth": False,
                        "raw_value_difference": value_band_diagnostics(a, b, config),
                        "coefficient_dt_difference": coefficient_time_statistics(a - b, config),
                        "amplitude_neighbor_difference": amplitude_neighbor_diagnostics(a, b, config, conditions,
                                                                                       decoder_amplitude_scale=scale)}
                del final_values
            _write_json(output / "result.partial.json", report)
            del sampler, model, normalizer
        stage = "final_input_revalidation"
        _assert_hashes(frozen)
        if read_manifest(corpus, verify_hashes=False)["manifest_sha256"] != manifest["manifest_sha256"]:
            raise ValueError("corpus manifest changed")
        report.update(execution_passed=True, all_inputs_revalidated=True, initial_noisy_states_sha256=common_initial,
                      generated_answers=sample_count * len(steps) * len(inventory))
    except Exception as error:
        report.update(failed_stage=stage, error=repr(error), traceback=traceback.format_exc(), partial_artifacts_preserved=True)
    _write_json(output / "result.json", report)
    return report
