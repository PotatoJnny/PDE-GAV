#!/usr/bin/env python3
"""Exercise a frozen measured NS Generator through Base/G/P, resume, and Panel.

This is an integration probe, not a scientific Generator or Agent experiment.
The only optimization is two steps of the existing neural Guidance model on
explicitly synthetic scalar feedback. All Base/Guided sampling uses raw weights
and 256-step Heun. The Panel submits one fixed, existing hold-initial program.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "legacy/ns/src", ROOT / "src"):
    sys.path.insert(0, str(directory))

import numpy as np
import torch

from check_ns_representation_compatibility import inspect_record, plain, sha, write_json
from pde_agent_verifier_solver.plugins.ns.experimental_mms import assemble_experimental_mms, _answer
from pde_agent_verifier_solver.plugins.ns.experimental_mms_project import ExperimentalMMSProblem
from pde_agent_verifier_solver.plugins.ns.structured_guidance_backend import StructuredNSGuidanceConfig
from pde_agent_verifier_solver.engine.guidance_controller import GuidanceProtocolConfig
from pde_agent_verifier_solver.sdk import (
    AnswerQuery, Artifact, CaseFeedback, CaseVisibility, DatasetInput, DatasetKind,
    GenerationKind, GenerationPurpose, GenerationRequest, GeneratorMode,
    GuidanceTrainingBudget, RoundFeedback, TaskKind,
)

RUNTIME = {"device": "cpu", "integration_steps": 256, "integration_method": "heun",
           "flow_state": "raw", "cuda_graph_sampling": False,
           "perturbation_amplitude": 2.861761125944777e-6, "spectral_max_mode": 20}


def event(name, **values):
    print(json.dumps({"event": name, **values}), flush=True)


def request(kind, seed, *, count=1, index=1, anchors=()):
    return GenerationRequest(round_index=index, count=count, mode=GeneratorMode.ONLINE,
        kind=kind, visibility=CaseVisibility.HIDDEN, purpose=GenerationPurpose.PROMOTION,
        seed=seed, anchors=anchors)


def guidance_settings():
    config = StructuredNSGuidanceConfig(training_steps=2, batch_size=2, width=8,
                                       depth=1, time_width=4, minimum_cases=2)
    protocol = GuidanceProtocolConfig(GuidanceTrainingBudget(60., "cpu", 1),
        config.sha256, 2, 2, 2, 120., "cpu", {
            "sampling_policy": "solver_reward_guided_rectified_flow", "flow_state": "raw",
            "integration_steps": 256, "guidance_strength": config.guidance_strength,
            "guidance_max_flow_ratio": config.guidance_max_flow_ratio})
    return config, protocol


def assembly(dataset, flow, pca, out, *, runtime_settings=None):
    config, protocol = guidance_settings()
    return assemble_experimental_mms(dataset, flow_checkpoint=flow, pca_checkpoint=pca,
        generator_settings=RUNTIME if runtime_settings is None else runtime_settings, guidance_artifact_root=out,
        guidance_config=config, guidance_protocol=protocol)


def inspect(assembled, record, label, arrays):
    result = inspect_record(assembled.bundle, record, label, arrays)
    field = _answer(record)
    points = arrays[f"{label}/query_points_txy"]
    # Unlike an initial-only check, this point includes the nonzero tau² P term.
    quantities = field.quantities(points[2:3, (1, 2, 0)])
    for target, names in (("private_velocity", ("u", "v")),
                          ("public_forcing", ("forcing_u", "forcing_v"))):
        reference = np.stack([quantities[name].detach().cpu().numpy() for name in names], axis=-1)
        np.testing.assert_allclose(arrays[f"{label}/{target}"][2:3], reference,
                                   rtol=1e-10, atol=1e-11)
    divergence = float(quantities["divergence"].detach().abs().max().cpu())
    residual = max(float(quantities[key].detach().abs().max().cpu())
                   for key in ("residual_u", "residual_v"))
    if divergence > 1e-8 or residual > 1e-8:
        raise AssertionError("manufactured field consistency check failed")
    result["middle_point_divergence_abs_max"] = divergence
    result["middle_point_manufactured_residual_abs_max"] = residual
    return result


def assert_same(left, right):
    assert plain(left.case) == plain(right.case), "restored Case identity differs"
    a, b = _answer(left), _answer(right)
    assert a.config.to_dict() == b.config.to_dict()
    assert a.geometry_spec.to_dict() == b.geometry_spec.to_dict()
    for name in ("psi", "pressure"):
        np.testing.assert_array_equal(getattr(a.coefficients, name).detach().cpu().numpy(),
                                      getattr(b.coefficients, name).detach().cpu().numpy())


def run_panel(dataset, assembled, out, *, runtime_settings=None):
    from pde_agent_verifier_solver.app import ProductApplication, ProgramBundle, RunStatus
    from pde_agent_verifier_solver.engine import PythonCandidateCodec
    from pde_agent_verifier_solver.engine.evaluator import EvaluationRequest
    from pde_agent_verifier_solver.engine.runtime import AgentOutcome
    from pde_agent_verifier_solver.panel.controller import PanelController
    from pde_agent_verifier_solver.plugins.ns.product import build_ns_plugin

    fixed = PythonCandidateCodec(task_kind=TaskKind.SOLVER).bind_existing(
        ROOT / "configs/candidates/ns_hold_initial_bundle", storage_root=out / "fixed_candidate",
        artifact_id="measured-integration-fixed-initial", name="Fixed integration probe",
        summary="Existing hold-initial program; no search or improvement.", usage="prepare_solution")

    class LocalAgent:
        def __init__(self):
            self.calls = []

        def run(self, context):
            artifact = replace(fixed, artifact_id=f"fixed-initial-r{context.round_plan.round_index}",
                round_index=context.round_plan.round_index, agent_id=context.agent.agent_id)
            response = context.evaluator.evaluate(artifact,
                EvaluationRequest(tuple(case.case_id for case in context.cases)))
            assert np.isfinite(response.primary_error) and response.cases == ()
            self.calls.append({"round": context.round_plan.round_index, "cases": len(context.cases)})
            return AgentOutcome(artifact, observations=("Fixed local interface probe; no Agent search.",))

    agent = LocalAgent()
    plugin = replace(build_ns_plugin(task_id="experimental_mms_solver"), generator=assembled.generator)
    programs = ProgramBundle.from_plugin(plugin, {"local": agent})
    app = ProductApplication.local_development()
    field = _answer(dataset.development_cases[0])
    left, right, bottom, top = field.domain_bounds
    query = AnswerQuery.at_points(fields=("velocity_x", "velocity_y"), points=(
        {"t": (field.t0+field.t1)/2, "x": left+.2*(right-left), "y": bottom+.3*(top-bottom)},
        {"t": field.t1, "x": left+.8*(right-left), "y": bottom+.7*(top-bottom)}))
    panel = PanelController(app, programs, prepared_dataset=dataset,
                            generator_checkpoint=assembled.checkpoint, default_query=query)
    payload = {
        **panel.setup_page()["defaults"], "project_name": "Measured NS integration only",
        "run_name": "Fixed local program ABI check", "round_count": 1,
        "dataset_location": "", "problem_settings": {}, "agents": ["local"],
        "guidance_enabled": False, "perturbations_enabled": True,
        "allowed_sources": ["original", "base"], "sampling_strategy": "fixed_recipe",
        "max_sampled_cases_per_agent_per_round": 3,
        "starter_cases": {"original": 1, "base": 1},
        "fixed_recipe": {"original": 1, "base": 1, "base_perturbed": 1},
        "round_hidden": {"base": 1, "base_perturbed": 1},
        "terminal_evaluation": {"cohorts": [{"id": "base_model", "guidance_source": "base",
            "counts": {"base": 1, "base_perturbed": 1}}],
            "ranking": {"cohort_id": "base_model", "distribution": "base", "statistic": "mean"}},
        "generator_settings": dict(RUNTIME if runtime_settings is None else runtime_settings),
        "task_settings": {"official_point_count": 4, "terminal_point_count": 4},
        "development_point_count": 4, "agent_budget_seconds": 60,
    }
    write_json(out / "panel_payload.json", payload)
    deadline = time.monotonic() + 180
    launched = panel.create_and_launch(payload)
    completed = app.wait(launched["run_id"], timeout=45)
    while completed.status not in (RunStatus.COMPLETED, RunStatus.FAILED):
        event("panel_wait", status=completed.status.value)
        if time.monotonic() > deadline:
            raise TimeoutError("local Panel integration exceeded its time budget")
        completed = app.wait(launched["run_id"], timeout=30)
    if completed.status is not RunStatus.COMPLETED:
        raise AssertionError(f"Panel run failed: {completed.error}")
    page = plain(panel.run_page(launched["run_id"]))
    assert page["status"] == "completed" and page["result"]["artifact_id"]
    assert np.isfinite(page["result"]["primary_error"])
    assert any(item.event_type == "generator_checkpoint_selected" for item in completed.events.snapshot())
    write_json(out / "panel_result.json", page)
    return {"success": True, "run_id": launched["run_id"], "agent_calls": agent.calls,
        "candidate_tree_sha256": fixed.metadata["candidate_bundle_tree_sha256"],
        "scientific_result": False, "hidden_original_cases": 0,
        "scope": "Real Panel create/launch/result with one fixed existing local program; no HTTP/browser rendering or solver search.",
        "resume": "Not available in Panel/App; Generator state restoration is tested separately."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--flow-checkpoint", type=Path)
    parser.add_argument("--skip-panel", action="store_true")
    parser.add_argument("--perturbation-family", choices=("spectral_complement", "spline_refinement"),
                        default="spectral_complement")
    parser.add_argument("--perturbation-amplitude", type=float,
                        help="Explicit coefficient epsilon. Required for spline_refinement; not physically equivalent to the cosine epsilon.")
    parser.add_argument("--spectral-max-mode", type=int, default=20,
                        help="Explicit cosine perturbation upper mode; must exceed the loaded spatial band.")
    args = parser.parse_args()
    if args.perturbation_family == "spline_refinement" and args.perturbation_amplitude is None:
        parser.error("spline_refinement requires an explicit --perturbation-amplitude selected offline")
    if args.perturbation_amplitude is not None and (not np.isfinite(args.perturbation_amplitude) or args.perturbation_amplitude <= 0):
        parser.error("--perturbation-amplitude must be finite and positive")
    if args.spectral_max_mode < 1:
        parser.error("--spectral-max-mode must be positive")
    if args.spectral_max_mode != 20 and args.perturbation_amplitude is None:
        parser.error("a changed --spectral-max-mode requires an explicit offline-selected --perturbation-amplitude")
    runtime_settings = dict(RUNTIME)
    runtime_settings["spectral_max_mode"] = args.spectral_max_mode
    if args.perturbation_family != "spectral_complement":
        runtime_settings["perturbation_family"] = args.perturbation_family
    if args.perturbation_amplitude is not None:
        runtime_settings["perturbation_amplitude"] = args.perturbation_amplitude
    prepared, out = args.prepared.resolve(), args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to reuse integration output: {out}")
    flow = args.flow_checkpoint.resolve() if args.flow_checkpoint else Path(json.loads(
        (prepared / "base_generator/train_report.json").read_text())["final_checkpoint"])
    pca, manifest = prepared / "pca_checkpoint.pt", prepared / "prepared_dataset.json"
    artifacts = {str(path): sha(path) for path in (flow, pca, manifest)}
    torch.set_num_threads(1)
    started = time.perf_counter()
    out.mkdir(parents=True)
    event("load_prepared", path=str(prepared))
    dataset = ExperimentalMMSProblem().prepare_dataset(DatasetInput(
        "measured-generator-integration", DatasetKind.EXPERIMENTAL, str(manifest), settings={}))
    template = _answer(dataset.development_cases[0])
    if not template.config.is_legacy_basis and args.perturbation_amplitude is None:
        parser.error("a nonlegacy basis requires an explicit offline-selected --perturbation-amplitude")
    first = assembly(dataset, flow, pca, out / "guidance", runtime_settings=runtime_settings)
    generator = first.generator
    state = generator.load(first.checkpoint, {})
    bases = generator.sample(state, request(GenerationKind.BASE, 20260917, count=2, index=0))
    child = generator.sample(state, request(GenerationKind.PERTURBED, 20260918, index=0,
                                            anchors=(bases[0].case,)))[0]
    event("base_and_perturbation_ready")
    fake_solver = Artifact("synthetic-feedback-only", TaskKind.SOLVER, "Synthetic interface feedback",
                          SimpleNamespace(), "Never executed", "", 0, "local")
    feedback = RoundFeedback(0, fake_solver, tuple(
        CaseFeedback(row.case, .1+.2*index, {}) for index, row in enumerate(bases)))
    boundary = generator.backend.fit_guidance(state.base, None, feedback, {"controller_seed": 20260920})
    state = replace(state, guidance=boundary)
    guided = generator.sample(state, request(GenerationKind.GUIDED, 20260919))[0]
    assert guided.case.public_data["distribution_lineage"]["kind"] == "frozen_reward_guided"
    assert child.case.parent_case_id == bases[0].case.case_id
    arrays, cases = {}, {}
    for label, record in (("original_development", dataset.development_cases[0]),
                          ("base", bases[0]), ("guided", guided), ("perturbed", child)):
        cases[label] = inspect(first, record, label, arrays)
    np.testing.assert_allclose(arrays["base/public_initial_velocity"],
                               arrays["perturbed/public_initial_velocity"], rtol=1e-11, atol=1e-12)
    np.savez_compressed(out / "query_arrays.npz", **arrays)
    event("base_guidance_perturbation_mms_passed")
    objects = out / "resume_objects"
    runtime = generator.export_runtime_state(objects)
    state_payload = generator.export_generator_state(state, objects)
    saved_cases = [generator.export_case_record(row, objects)
                   for row in (dataset.development_cases[0], bases[0], guided, child)]
    write_json(out / "runtime_state.json", runtime)
    write_json(out / "generator_state.json", state_payload)
    write_json(out / "case_records.json", saved_cases)
    second = assembly(dataset, flow, pca, out / "restored_guidance", runtime_settings=runtime_settings)
    second.generator.restore_runtime_state(json.loads((out / "runtime_state.json").read_text()), objects)
    restored = second.generator.restore_generator_state(json.loads((out / "generator_state.json").read_text()), objects)
    replays = [second.generator.restore_case_record(row, objects)
               for row in json.loads((out / "case_records.json").read_text())]
    for expected, actual in zip((dataset.development_cases[0], bases[0], guided, child), replays):
        assert_same(expected, actual)
    continued = []
    for kind, seed, anchors in ((GenerationKind.BASE, 20260921, ()),
                               (GenerationKind.GUIDED, 20260922, ()),
                               (GenerationKind.PERTURBED, 20260923, (bases[0].case,))):
        query = request(kind, seed, index=0 if anchors else 1, anchors=anchors)
        expected = generator.sample(state, query)[0]
        actual = second.generator.sample(restored, query)[0]
        assert_same(expected, actual)
        old_arrays, new_arrays = {}, {}
        old = inspect(first, expected, kind.value, old_arrays)
        new = inspect(second, actual, kind.value, new_arrays)
        assert old == new
        for key, value in old_arrays.items():
            np.testing.assert_array_equal(value, new_arrays[key])
        continued.append({"kind": kind.value, "case_id": expected.case.case_id})
    event("state_restore_and_future_base_guidance_perturbation_passed")
    panel = {"skipped": True} if args.skip_panel else run_panel(dataset, second, out, runtime_settings=runtime_settings)
    report = {"schema": "ns_measured_generator_integration_v1", "success": True,
        "scope": "Integration only; no scientific quality or Agent-improvement claim.",
        "prepared": str(prepared), "artifacts": artifacts, "runtime_settings": runtime_settings,
        "guidance": {"training_steps": 2, "width": 8, "depth": 1, "batch_size": 2,
            "feedback": "Two synthetic scalar errors (0.1,0.3) on real Base Cases; no solver-derived rewards.",
            "sampling": "Original neural Guidance and 256-step Heun/raw; frozen PCA/Flow unchanged."},
        "cases": cases, "resume": {"success": True, "future_cases": continued,
            "scope": "Actual Generator state and Case export to disk, fresh assembly restore, exact future Base/G/P identities, coefficients and query values."},
        "panel": panel, "elapsed_seconds": time.perf_counter()-started,
        "input_mode": dataset.training_summary.get("input_mode"),
        "pressure_model": "zero", "verifier_final_test_observations_evaluated": False}
    for path, digest in artifacts.items():
        assert sha(path) == digest, f"source artifact changed: {path}"
    write_json(out / "integration_report.json", report)
    event("complete", report=str(out / "integration_report.json"), elapsed_seconds=report["elapsed_seconds"])


if __name__ == "__main__":
    main()
