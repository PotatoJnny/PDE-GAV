"""Replay the frozen strong initial law into native t0 controls, without a PDE.

This is a representation gate on eight train and eight validation cases, not
a trained conditional model or distribution qualification. The constant-time
carrier exists only to call the original 33-control projector. Its future
values are neither read from the corpus nor published as physical answers.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import importlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pde_agent_verifier_solver.plugins.vpfp.canonical_binding import VPFPOnlineModelSpec
from pde_agent_verifier_solver.plugins.vpfp.diffusion_data import load_split, read_manifest
from pde_agent_verifier_solver.plugins.vpfp.perturbation_backend import _legacy_api
from pde_agent_verifier_solver.plugins.vpfp.value_control_audit import _layout
from diagnose_vpfp_weak_structure_prior import _file_sha, _json_sha, _write_json, select_validation_rows

SCHEMA = "vpfp_known_initial_representation_audit_v1"
FAMILY = "spatially_conditioned_gaussian_packets_v3"
PHYSICAL_MODULES = ("high_fidelity", "online_training", "torch_reference",
                    "continuous_flow_training", "continuous_hermite")
REPLAY_RTOL, REPLAY_ATOL = 2.e-4, 2.e-5
_SCRATCH_ROOT = Path("/scratch")


def initial_controls(coefficients, row, layout):
    """Copy only t0 VALUES from a read-only full-layout array; never slopes."""
    k, r, h, m = (layout[key] for key in
                 ("control_count", "rho_modes", "hermite_channels", "kinetic_modes"))
    rho_end = 2 * k * r
    kinetic_start = rho_end + layout["inactive_dimension"]
    dimension = kinetic_start + h * 2 * k * m
    if coefficients.ndim != 2 or coefficients.shape[1] != dimension:
        raise ValueError("initial control source has incorrect full decoder dimension")
    # Basic slicing and reshape are views of the mmap. Only the final t0
    # slices are copied, so even nonfinite future/slope controls are not used.
    rho = coefficients[row, :rho_end].reshape(2, k, r)[0, 0]
    kinetic = coefficients[row, kinetic_start:].reshape(h, 2, k, m)[:, 0, 0]
    result = {"rho": np.array(rho, copy=True), "kinetic": np.array(kinetic, copy=True)}
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError("nonfinite t0 value controls")
    return result


def initial_carrier(controls, layout):
    """Synthetic constant carrier for exact t0 queries, NOT a trajectory."""
    k, r, h, m = (layout[key] for key in
                 ("control_count", "rho_modes", "hermite_channels", "kinetic_modes"))
    rho, kinetic = np.asarray(controls["rho"]), np.asarray(controls["kinetic"])
    if (rho.shape != (r,) or kinetic.shape != (h, m)
            or not np.isfinite(rho).all() or not np.isfinite(kinetic).all()):
        raise ValueError("malformed initial control values")
    rho_part = np.zeros((2, k, r), dtype=np.float32)
    kinetic_part = np.zeros((h, 2, k, m), dtype=np.float32)
    rho_part[0] = rho[None]
    kinetic_part[:, 0] = kinetic[:, None]
    return np.concatenate((rho_part.ravel(),
        np.zeros(layout["inactive_dimension"], dtype=np.float32), kinetic_part.ravel()))


def comparison(predicted, reference):
    """Original-unit errors; relative L2 is floored once, not pointwise."""
    predicted, reference = np.asarray(predicted, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if (predicted.shape != reference.shape or not reference.size
            or not np.isfinite(predicted).all() or not np.isfinite(reference).all()):
        raise ValueError("comparison requires aligned nonempty finite arrays")
    difference = predicted - reference
    rms = lambda array: float(np.sqrt(np.mean(array * array)))
    return {"rmse": rms(difference), "max_abs_error": float(np.max(np.abs(difference))),
        "relative_l2": rms(difference) / max(rms(reference), 1.e-12),
        "reference_rms": rms(reference), "prediction_rms": rms(predicted),
        "reference_mean": float(reference.mean()), "prediction_mean": float(predicted.mean()),
        "reference_std": float(reference.std()), "prediction_std": float(predicted.std()),
        "float32_replay_allclose": bool(np.allclose(predicted, reference, rtol=REPLAY_RTOL, atol=REPLAY_ATOL))}


def analytic_initial_macros(spec, grid, physical, axes):
    """Full-velocity Gaussian-mixture moments on the source spatial grid.

    Correlation enters f0 but not the trace of its second moment. No velocity
    quadrature, truncated Hermite moment system, or PDE is used here.
    """
    if spec.get("family") != FAMILY:
        raise ValueError("known initial audit requires the frozen rich Gaussian-packet law")
    x, y = np.asarray(axes["x"])[:, None], np.asarray(axes["y"])[None, :]
    fields = []
    for component in spec["g0_components"]:
        def field(name):
            return physical._spatial_mode_field(component[name], x=x, y=y, grid=grid)
        fields.append((float(component["base_logit"]) + field("weight_modes"),
            float(component["base_mean"][0]) + field("mean_x_modes"),
            float(component["base_mean"][1]) + field("mean_y_modes"),
            np.exp(float(component["base_log_std"][0]) + field("log_std_x_modes")),
            np.exp(float(component["base_log_std"][1]) + field("log_std_y_modes"))))
    logits, mx, my, sx, sy = (np.stack([entry[index] for entry in fields]) for index in range(5))
    weights = np.exp(logits - logits.max(axis=0, keepdims=True))
    weights /= weights.sum(axis=0, keepdims=True)
    rho = physical._evaluate_rho0(spec, grid)
    result = {"rho": rho, "jx": rho * np.sum(weights * mx, axis=0),
        "jy": rho * np.sum(weights * my, axis=0),
        "K": .5 * rho * np.sum(weights * (mx**2 + my**2 + sx**2 + sy**2), axis=0)}
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError("nonfinite analytic Gaussian-mixture moments")
    return result


def discrete_initial_macros(f0, grid, axes):
    f0 = np.asarray(f0, dtype=np.float64)
    vx, vy = np.asarray(axes["vx"])[None, None, :, None], np.asarray(axes["vy"])[None, None, None, :]
    if f0.shape != (grid.nx, grid.ny, grid.nvx, grid.nvy) or not np.isfinite(f0).all() or np.any(f0 < 0):
        raise ValueError("initial f must be finite nonnegative on the original grid")
    integrate = lambda value: np.sum(value, axis=(-2, -1)) * float(grid.dvx) * float(grid.dvy)
    return {"rho": integrate(f0), "jx": integrate(f0 * vx), "jy": integrate(f0 * vy),
            "K": .5 * integrate(f0 * (vx**2 + vy**2))}


def _check_frozen(files):
    for filename, expected in files.items():
        path = Path(filename)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'frozen initial-audit input changed: {path}')


def _runtime(manifest, spec, device):
    """Load SHA-checked original projection/decoder code; never construct a solver."""
    api = _legacy_api()
    modules = {name: importlib.import_module(f"agent_verifier.vpfp.{name}") for name in PHYSICAL_MODULES}
    expected = manifest["provenance"]["physical_module_sha256"]
    if set(expected) != set(modules):
        raise ValueError("frozen physical source bundle fields differ")
    files = {str(Path(module.__file__).absolute()): expected[name] for name, module in modules.items()}
    _check_frozen(files)
    for name in ("robust_perturbation", "analytic_source", "continuous_cases", "continuous_macros", "operators"):
        module = importlib.import_module(f"agent_verifier.vpfp.{name}")
        files[str(Path(module.__file__).absolute())] = _file_sha(module.__file__)
    payload = torch.load(spec.base_decoder_checkpoint, map_location="cpu", weights_only=True)
    provenance = manifest["provenance"]
    if _json_sha(payload["decoder_config"]) != _json_sha(provenance["decoder_config"]):
        raise ValueError("base decoder configuration differs from frozen corpus")
    adapter = payload.get("robust_answer_first_adapter", {})
    if (adapter.get("unused_neural_parameters_zeroed") is not True
            or adapter.get("continuous_hidden_answer_is_primary") is not True
            or adapter.get("generator_parameterization_public") is not False):
        raise ValueError("base decoder lacks the original answer-first identity")
    decoder = modules["continuous_hermite"].VPFPContinuousHermiteConfig.from_dict(payload["decoder_config"])
    generation = modules["online_training"].VPFPOnlineGenerationConfig(**provenance["generation_config"])
    physical = modules["high_fidelity"]
    equation = physical.VPFPHighFidelityConfig.from_dict(provenance["equation_config"])
    equation.validate()
    if equation.equation_family != FAMILY or _json_sha(equation.to_dict()) != _json_sha(provenance["equation_config"]):
        raise ValueError("initial equation law differs from frozen rich-packet configuration")
    for name in ("collision_nu_range", "temperature_range", "time_domain"):
        if tuple(getattr(decoder, name)) != tuple(getattr(spec, name)):
            raise ValueError(f"decoder {name} differs from requested frozen spec")
    for name in ("collision_nu_range", "temperature_range"):
        if tuple(getattr(equation, name)) != tuple(getattr(spec, name)):
            raise ValueError(f"equation {name} differs from requested frozen spec")
    grid, time = generation.grid(), generation.time()
    model = api.build_deterministic_direct_model(decoder, device=device)
    with api.ieee_float32_coordinate_matmul(model):
        projector = modules["online_training"].VPFPTorchStructuredProjector(
            decoder, grid=grid, time=time, device=device, dtype=generation.dtype)
    return SimpleNamespace(api=api, physical=physical, equation=equation, reference=modules["torch_reference"],
        generation=generation, grid=grid, time=time, model=model, projector=projector, device=device,
        axes=api.make_axes(grid), frozen_files=files)


def reconstruct_initial(runtime, seed, condition, layout):
    """Full seed replay preserves source RNG draws, but source is never solved."""
    spec = runtime.physical.sample_continuous_equation_spec(
        seed=int(seed), config=runtime.equation, defer_rho_log_partition=True)
    sampled_condition = np.asarray([spec["collision_nu"], spec["temperature"]], dtype=np.float32)
    if not np.array_equal(sampled_condition, np.asarray(condition, dtype=np.float32)):
        raise ValueError("equation seed does not reproduce the frozen condition")
    reference = runtime.reference
    parameters = reference.pack_torch_equation_specs([spec], dtype="float32")
    # f0 is time-grid independent. The original materializer also computes
    # source frames; use the minimum legal three and discard them, without a
    # solver or a change to the original 33-control projection operators.
    prepared = reference.evaluate_torch_equation_parameter_batch(parameters,
        grid=runtime.grid, time=replace(runtime.time, nt=3), device=runtime.device)
    f0 = prepared["f0"]
    conditions = torch.stack((prepared["collision_nu"], prepared["temperature"]), dim=-1)
    if not np.array_equal(conditions.detach().cpu().numpy()[0], sampled_condition):
        raise ValueError("packed/materialized initial conditions changed")
    del prepared
    # This is the solver's OUTPUT quadrature, including t0; its internal
    # electrostatic field quadrature is trapezoidal and must not replace it.
    rho, g = reference._uniform_rho_g(torch, f0, grid=runtime.grid,
        rho_floor=float(runtime.generation.solver().rho_floor))
    carrier = {"rho": rho[:, None].expand(-1, runtime.time.nt, -1, -1),
               "g": g[:, None].expand(-1, runtime.time.nt, -1, -1, -1, -1)}
    with torch.no_grad(), runtime.api.ieee_float32_coordinate_matmul(runtime.model):
        projected = runtime.projector.project(carrier, conditions)
    controls = initial_controls(projected["latents"].detach().cpu().numpy(), 0, layout)
    return {"controls": controls,
        "analytic": analytic_initial_macros(spec, runtime.grid, runtime.physical, runtime.axes),
        "discrete": discrete_initial_macros(f0.detach().cpu().numpy()[0], runtime.grid, runtime.axes),
        "equation_spec_sha256": _json_sha(spec), "component_count": len(spec["g0_components"]),
        "rho_log_partition_deferred": spec.get("rho_log_partition") is None,
        "condition": sampled_condition.tolist()}


def native_initial_macros(runtime, controls, condition, layout):
    xx, yy = np.meshgrid(runtime.axes["x"], runtime.axes["y"], indexing="ij")
    points = np.column_stack((np.full(xx.size, runtime.time.t0), xx.ravel(), yy.ravel()))
    with runtime.api.ieee_float32_coordinate_matmul(runtime.model):
        raw = runtime.api.evaluate_continuous_hermite_macro_points(runtime.model,
            initial_carrier(controls, layout), points, equation_condition=np.asarray(condition), chunk_size=256)
    shape = (runtime.grid.nx, runtime.grid.ny)
    return {"rho": raw["rho"].reshape(shape), "jx": raw["momentum"][:, 0].reshape(shape),
            "jy": raw["momentum"][:, 1].reshape(shape), "K": raw["kinetic_energy"].reshape(shape)}


def run(config_path, corpus_root, output, device="cuda"):
    if not os.environ.get("SLURM_JOB_ID", "").isdigit():
        raise RuntimeError("real initial projection requires a Slurm compute allocation")
    raw_output = Path(output)
    config_path, corpus_root, output = Path(config_path).resolve(strict=True), Path(corpus_root).resolve(strict=True), raw_output.resolve()
    if (raw_output.is_symlink() or output.exists() or output.is_relative_to(corpus_root)
            or not output.is_relative_to(_SCRATCH_ROOT.resolve()) or output == _SCRATCH_ROOT.resolve()):
        raise ValueError("choose a fresh initial-audit output outside the source corpus")
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": SCHEMA, "execution_passed": False, "mechanical_replay_passed": False,
        "new_pde_solves": 0, "training_run": False, "scientific_distribution_validation": False,
        "physical_projection_qualified": False, "cases": [],
        "evidence_role": "train_validation_initial_representation_gate_not_model_admission"}
    try:
        frozen = {str(config_path): _file_sha(config_path), str(corpus_root / "manifest.json"): _file_sha(corpus_root / "manifest.json")}
        config = json.loads(config_path.read_text())
        spec = VPFPOnlineModelSpec.from_mapping(config["vpfp"])
        spec.preflight()
        manifest = read_manifest(corpus_root, verify_hashes=True)
        if (spec != VPFPOnlineModelSpec.from_mapping(manifest["model_spec"])
                or "strong" not in spec.regime_id or spec.base_decoder_sha256 != manifest["base_decoder_sha256"]):
            raise ValueError("requested strong model spec differs from source corpus")
        layout = _layout(manifest)
        splits = {name: load_split(corpus_root, name, verify_hashes=False) for name in ("train", "validation")}
        edges = (np.linspace(*spec.collision_nu_range, 5), np.linspace(*spec.temperature_range, 3))
        selection = {name: select_validation_rows(split.conditions, split.sample_ids, edges).tolist()
                     for name, split in splits.items()}
        for name in ("base_decoder", "expanded_decoder"):
            frozen[str(Path(getattr(spec, name + "_checkpoint")).resolve())] = getattr(spec, name + "_sha256")
        runtime = _runtime(manifest, spec, device)
        frozen.update(runtime.frozen_files)
        for path in (Path(__file__), ROOT / "scripts/diagnose_vpfp_weak_structure_prior.py",
                     ROOT / "src/pde_agent_verifier_solver/plugins/vpfp/diffusion_data.py",
                     ROOT / "src/pde_agent_verifier_solver/plugins/vpfp/value_control_audit.py"):
            frozen[str(path.resolve())] = _file_sha(path)
        _check_frozen(frozen)
        plan = {"schema": SCHEMA, "model_spec": asdict(spec), "corpus_manifest_sha256": manifest["manifest_sha256"],
            "provenance_sha256": _json_sha(manifest["provenance"]), "frozen_files": frozen, "layout": layout,
            "selection_policy": "minimum_equation_seed_in_each_of_4_by_2_condition_bins_per_split",
            "selection": {name: [{"row": row, "equation_seed": int(splits[name].sample_ids[row]),
                "condition": splits[name].conditions[row].tolist()} for row in rows] for name, rows in selection.items()},
            "numerically_read_corpus_controls": "t0_values_only; no slopes, future, calibration or reference controls",
            "all_source_files_integrity_hashed": True, "projector_carrier": "constant_reconstructed_f0_33_slots_only_internal",
            "published_future_trajectory": False, "source_frames": "three_frames_materialized_then_discarded_without_solving",
            "macro_comparison": "analytic_full_velocity; original_uniform_velocity_grid; projected_native; corpus_native",
            "spatial_probe_grid": [runtime.grid.nx, runtime.grid.ny], "spatial_convergence_qualified": False,
            "full_f_projection_error_evaluated": False, "macro_equivalence_margins": None,
            "replay_tolerance": {"rtol": REPLAY_RTOL, "atol": REPLAY_ATOL},
            "precision": "source_float32_with_IEEE_coordinate_and_projection_matmul",
            "new_pde_solves": 0, "training_run": False, "device": str(device)}
        _write_json(output / "plan.json", plan)
        plan_sha = _file_sha(output / "plan.json")
        report.update(plan_sha256=plan_sha, corpus_manifest_sha256=manifest["manifest_sha256"],
                      base_decoder_sha256=spec.base_decoder_sha256)
        for name, rows in selection.items():
            split = splits[name]
            for row in rows:
                seed, condition = int(split.sample_ids[row]), np.array(split.conditions[row], copy=True)
                reconstructed = reconstruct_initial(runtime, seed, condition, layout)
                stored = initial_controls(split.latents, row, layout)
                projected = native_initial_macros(runtime, reconstructed["controls"], condition, layout)
                corpus = native_initial_macros(runtime, stored, condition, layout)
                entry = {key: reconstructed[key] for key in ("equation_spec_sha256", "component_count", "condition", "rho_log_partition_deferred")}
                entry.update(split=name, row=row, equation_seed=seed,
                    initial_controls_sha256={"reconstructed": _json_sha({key: value.tolist() for key, value in reconstructed["controls"].items()}),
                                             "corpus": _json_sha({key: value.tolist() for key, value in stored.items()})},
                    controls_replay={key: comparison(reconstructed["controls"][key], stored[key]) for key in stored},
                    macros={label: {key: comparison(predicted[key], reference[key]) for key in reference}
                        for label, predicted, reference in (
                            ("discrete_vs_analytic", reconstructed["discrete"], reconstructed["analytic"]),
                            ("projected_vs_discrete", projected, reconstructed["discrete"]),
                            ("projected_vs_analytic", projected, reconstructed["analytic"]),
                            ("corpus_vs_analytic", corpus, reconstructed["analytic"]),
                            ("projected_vs_corpus", projected, corpus))})
                entry["mechanical_replay_passed"] = all(value["float32_replay_allclose"]
                    for group in (entry["controls_replay"], entry["macros"]["projected_vs_corpus"]) for value in group.values())
                report["cases"].append(entry)
                _write_json(output / "result.json", report)
                print(json.dumps({"stage": "known_initial_case", "split": name, "equation_seed": seed,
                                  "mechanical_replay_passed": entry["mechanical_replay_passed"]}), flush=True)
        _check_frozen(frozen)
        report["mechanical_replay_passed"] = all(entry["mechanical_replay_passed"] for entry in report["cases"])
        if not report["mechanical_replay_passed"]:
            raise ValueError("known-law initial projection does not replay corpus t0 within frozen float32 tolerances")
        report["execution_passed"] = True
        report["next_gate"] = "review original-unit macro projection errors before known-prior conditional-future training"
    except Exception as error:
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        _write_json(output / "result.json", report)
        raise
    _write_json(output / "result.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args.config, args.corpus, args.output, args.device)
