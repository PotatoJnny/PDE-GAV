#!/usr/bin/env python3
"""Two real-asset canonical rounds with a trusted diagnostic Agent, no LLM.

Use the production Controller, learned Base, round Guidance trainer and
isolated Candidate workers. The Agent only exercises public sampling and
evaluates S0; it submits nothing, so the incumbent remains unchanged. This
checks integration/retraining, not new Solver improvement or LLM behavior.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_canonical_vpfp_campaign import Writer, file_sha, jsonable, load_config, persist_guidance, recipe
from pde_agent_verifier_solver.engine import (
    EventStream, PythonCandidateCodec, PythonTaskWorkerSpec, SubprocessCandidateExecutionService,
)
from pde_agent_verifier_solver.engine.evaluator import EvaluationRequest, HiddenVerifier
from pde_agent_verifier_solver.engine.runtime import AgentOutcome
from pde_agent_verifier_solver.sdk import (
    CaseVisibility, GenerationKind, GeneratorMode, GenerationPurpose, GenerationRequest,
    PairedSampleRequest, SampleCasesRequest, SampleSource, TaskKind, inspect_candidate_bundle,
)

SCHEMA = "vpfp_canonical_two_round_real_execution_check_v1"


def verify_quick_measurements(reconstructions, audits, *, error, retention):
    if len(reconstructions) != 4 or len({row["root_seed"] for row in reconstructions}) != 4:
        raise ValueError("quick check needs four unique actual BASE draws")
    if any(row["bitwise_equal"] is not True or row["maximum_absolute_difference"] != 0
           for row in reconstructions):
        raise ValueError("batch-four captured answer differs from its per-root clean-coordinate decode")
    if not math.isfinite(error) or error < 0:
        raise ValueError("quick S0 isolated evaluation did not produce a valid macro error")
    if len(audits) != 1:
        raise ValueError("quick check requires exactly one isolated Candidate execution")
    row = audits[0]
    if (row["status"] != "completed" or row["return_code"] != 0 or row["error_code"] is not None
            or row["process_group_cleaned"] is not True or row["candidate_process_isolated"] is not True
            or row["controller_truth_shared"] is not False):
        raise ValueError("quick Candidate execution failed or violated isolation/cleanup")
    if set(retention) != {"draws", "pending_draws", "residents", "anchors"} or any(retention.values()):
        raise ValueError("quick check retained private Case/draw state after release")


def run_quick(config_path, output):
    """Load/sample four actual roots and score one S0, never train a Guide."""
    if not os.environ.get("SLURM_JOB_ID", "").isdigit():
        raise RuntimeError("quick VPFP numerical execution requires a Slurm compute allocation")
    config = load_config(config_path)
    source_state = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all"], text=True)
    if source_state.strip():
        raise ValueError("quick execution check requires a clean committed source checkout")
    revision = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    writer = Writer(output)
    started = time.monotonic()
    report = {"schema": SCHEMA + "_quick", "execution_scope": "quick_load_batch4_one_isolated_solver",
        "status": "running", "execution_passed": False, "full_two_round_passed": False,
        "source_revision": revision, "config_sha256": file_sha(config_path),
        "llm_invoked": False, "guidance_training_performed": False, "scientific_admission": False,
        "online_assets_replaced": False, "reconstructions": [], "failures": []}
    binding = candidates = state = None
    records, guard = (), {}
    try:
        import torch
        from pde_agent_verifier_solver.plugins.vpfp.iteration_backend import build_vpfp_strong_iteration_binding
        from pde_agent_verifier_solver.plugins.vpfp.solver_error_guidance import _module_sha, _json_sha
        from run_plasma_canonical_smoke import _forbid_online_vpfp_generation
        binding = build_vpfp_strong_iteration_binding(**config["backend"], output_root=output / "private_backend")
        generator, backend = binding.plugin.generator, binding.plugin.generator.backend
        base = backend.capture.base
        before = (_module_sha(base.model), _json_sha(base.normalizer.to_payload()))
        parameter = next(base.model.parameters())
        state = generator.load(binding.checkpoint, {})
        with _forbid_online_vpfp_generation(guard):
            records = tuple(generator.sample(state, GenerationRequest(round_index=0, count=4,
                mode=GeneratorMode.ONLINE, kind=GenerationKind.BASE, visibility=CaseVisibility.HIDDEN,
                seed=config["master_seed"]+11700000000, purpose=GenerationPurpose.GUIDANCE_FEEDBACK)))
            if len(records) != 4:
                raise ValueError("quick BASE count=4 did not return four Cases")
            for record in records:
                row = backend._draws[record.case.case_id]
                draw = row["draw"]
                with torch.no_grad():
                    decoded = base.normalizer.decode((draw["initial_context"].to(parameter),
                                                       draw["future_white"].to(parameter)))
                    captured = draw["full"].to(parameter)
                    report["reconstructions"].append({"case_id": record.case.case_id,
                        "root_seed": row["root_seed"], "bitwise_equal": torch.equal(decoded, captured),
                        "maximum_absolute_difference": float((decoded-captured).abs().max()),
                        "relative_l2_difference": float((decoded-captured).double().norm()/captured.double().norm()),
                        "coordinate_dtype": str(captured.dtype)})
            writer.write("quick-summary.json", report)
            codec = PythonCandidateCodec(task_kind=TaskKind.OBSERVABLE_MODEL)
            initial = codec.bind_existing((config_path.parent / config["candidate"]["initial_bundle"]).resolve(),
                storage_root=output / "controller_candidates", artifact_id="vpfp-quick-s0", name="Frozen S0",
                summary="Actual public-only strong Solver", usage="prepare then evaluate POINTS")
            plan = recipe(config, initial)
            task = binding.plugin.task("observable_model")
            settings = config["execution"]
            candidates = SubprocessCandidateExecutionService(output / "candidate_execution",
                task=PythonTaskWorkerSpec.from_task_type(type(task)),
                evaluation_settings={**plan.task_settings, "default_query": plan.evaluator.default_query},
                max_workers=1, case_hard_wall_seconds=settings["candidate_case_hard_wall_seconds"],
                development_batch_hard_wall_seconds=settings["development_candidate_batch_hard_wall_seconds"],
                python_executable=sys.executable)
            verifier = HiddenVerifier(task, plan.evaluator.default_query, settings=plan.task_settings,
                seed=config["master_seed"]+11800000000, generator=generator, generator_state=state,
                candidate_execution=candidates)
            scored = verifier.evaluate(initial, records[:1])
            report["solver_primary_error"] = scored.primary_error
            report["solver_metrics"] = dict(scored.metrics)
            report["solver_bundle_sha256"] = initial.program.tree_sha256
            if inspect_candidate_bundle(initial.program.bundle_locator).tree_sha256 != initial.program.tree_sha256:
                raise ValueError("quick S0 bundle changed")
            after = (_module_sha(base.model), _json_sha(base.normalizer.to_payload()))
            if before != after:
                raise ValueError("quick evaluation mutated Base or normalizer")
            report["base_final_verified"] = True
        report["status"] = "measured"
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        report["failures"].append({"type": type(error).__name__, "message": str(error)})
        (output / "failure_traceback.txt").write_text(traceback.format_exc())
    finally:
        try:
            if binding is not None:
                backend = binding.plugin.generator.backend
                if state is not None:
                    # Include any partial generation transaction retained by
                    # the plugin, not just the tuple returned on full success.
                    case_ids = tuple(set(backend._draws) | set(backend.runtime._resident_cases) | set(backend.backend._anchors))
                    binding.plugin.generator.release(state, case_ids)
                report["resident_counts_after_run"] = {"draws": len(backend._draws),
                    "pending_draws": len(backend.capture.pending), "residents": len(backend.runtime._resident_cases),
                    "anchors": len(backend.backend._anchors)}
                if backend.training_audits:
                    raise ValueError("quick-only unexpectedly trained Guidance")
            audits = [] if candidates is None else jsonable(candidates.audits)
            writer.write("candidate_execution_audits.json", audits)
            if report["status"] == "measured":
                verify_quick_measurements(report["reconstructions"], audits,
                    error=report["solver_primary_error"], retention=report["resident_counts_after_run"])
                report.update(status="passed", execution_passed=True)
        except BaseException as error:
            report.update(status="failed", execution_passed=False, error_type=type(error).__name__, error=str(error))
            report["failures"].append({"type": type(error).__name__, "message": str(error)})
        report["wall_seconds"] = time.monotonic()-started
        report["online_coarse_solver_guard"] = guard
        final_revision = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        final_state = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all"], text=True)
        report["source_final_verified"] = final_revision == revision and not final_state.strip()
        if not report["source_final_verified"]:
            report.update(status="failed", execution_passed=False)
            report["failures"].append({"type": "SourceMutation", "message": "source changed during quick check"})
        writer.write("quick-summary.json", report)
        writer.write("status.json", report)
    return report


def diagnostic_config(config):
    """A smaller declared cohort, never a change to the training sample count."""
    value = deepcopy(config)
    value["campaign_id"] += "-real-execution-check"
    value["round_count"] = 2
    value["master_seed"] += 10300000000
    value["development"] = {"max_cases_per_agent_per_round": 3, "starter_base": 0}
    value["terminal"] = {"base": 1, "base_perturbed": 0, "guided_per_round": 1}
    # The scientific Task point counts and actual 64/16, 2000-step Guidance
    # recipe remain identical to the upcoming live Agent pilot.
    return value


class DiagnosticAgent:
    def __init__(self):
        self.rows = []

    def run(self, context):
        if context.incumbent is None or context.sampler is None:
            raise ValueError("two-round execution check requires the frozen S0 and adaptive sampler")
        original = context.sampler.sample_cases(SampleCasesRequest(SampleSource.BASE, 1))
        pair = context.sampler.sample_paired_cases(PairedSampleRequest(SampleSource.GUIDED, 1))
        cases = original.cases + pair.cases
        if len(cases) != 3 or len({case.case_id for case in cases}) != 3:
            raise ValueError("diagnostic expected one Base plus one Guided root/perturbed child")
        if any("source" in case.public_data["arrays"] or "g0" in case.public_data["arrays"] for case in cases):
            raise ValueError("canonical Case eagerly materialized a source/g0 array")
        response = context.evaluator.evaluate(context.incumbent,
            EvaluationRequest(tuple(case.case_id for case in cases)))
        if not math.isfinite(response.primary_error) or response.primary_error < 0:
            raise ValueError("diagnostic incumbent returned an invalid development score")
        if any(row.truth is not None for row in response.cases):
            raise ValueError("diagnostic Agent received forbidden hidden truth")
        self.rows.append({"round_index": context.round_plan.round_index,
            "candidate_id": context.incumbent.artifact_id,
            "case_ids": [case.case_id for case in cases], "base_count": 1,
            "guided_count": 1, "perturbed_count": 1,
            "paired_attempts": jsonable(pair.attempts), "primary_error": response.primary_error,
            "source_or_g0_prematerialized": False, "truth_received": False,
            "nominations": 0})
        return AgentOutcome(observations=(
            "Trusted integration diagnostic evaluated S0 on fresh Base, Guided and paired child Cases.",
            "No candidate nominated; this does not measure Agent learning or Solver improvement.",
        ), round_log=(f"# Round {context.round_plan.round_index} integration check\n\n"
            f"Public-only S0 development error: {response.primary_error}. "
            "One Base, one Guided and its perturbed child evaluated by the formal Candidate service. "
            "No submission; retain incumbent.\n"))


def verify_receipts(training, sampling, manifests, candidate_audits, agent_rows, *, selected_sha):
    """Pure-data checks shared by real execution and synthetic unit tests."""
    if len(training) != 2 or len(manifests) != 2 or len(agent_rows) != 2:
        raise ValueError("exactly two completed Guidance/Agent rounds are required")
    if [row["round_index"] for row in agent_rows] != [1, 2]:
        raise ValueError("diagnostic Agent round sequence differs")
    all_roots, all_case_ids = set(), set()
    inventory = {}
    cohort_counts = Counter()
    for item in sampling:
        if len(item["case_ids"]) != item["count"] or len(item["root_seeds"]) != item["count"]:
            raise ValueError("sampling audit count does not match actual roots/Cases")
        for case_id, root in zip(item["case_ids"], item["root_seeds"], strict=True):
            if case_id in all_case_ids or root in all_roots:
                raise ValueError("fresh sampling reused a physical root or Case ID")
            all_case_ids.add(case_id)
            all_roots.add(root)
            inventory[case_id] = {"root": root, **item}
        cohort_counts[(item["purpose"], item["round_index"], item["source"])] += item["count"]
    for purpose, round_index in (("guidance_feedback", 0), ("promotion", 1), ("promotion", 2)):
        if cohort_counts[(purpose, round_index, "base")] != 80:
            raise ValueError("missing 80-Case initial or round Official clean-Base cohort")
    feedback_roots, feedback_case_ids, snapshots = set(), set(), []
    for index, (audit, manifest) in enumerate(zip(training, manifests, strict=True)):
        if (audit["source_round_index"], audit["target_round_index"]) != (index, index+1):
            raise ValueError("Guidance targets the wrong round")
        if audit["target_solver_sha256"] != selected_sha or not audit["fresh_model_initialization"]:
            raise ValueError("Guidance does not retrain for the selected frozen solver")
        if audit["feedback_case_count"] != 80 or audit["clean_base_case_count"] != 80 or audit["excluded"]:
            raise ValueError("Guidance requires all 80 clean Base feedback Cases")
        report = audit["report"]
        if (report.get("execution_passed") is not True or report.get("final_step") != 2000
                or report.get("target_step") != 2000 or report.get("completed_step") != 2000
                or report.get("base_final_verified") is not True or report.get("feedback_final_verified") is not True
                or report.get("label_source") != "actual_public_solver_macro_error"):
            raise ValueError("real 2000-step Guidance training did not complete unchanged")
        if audit["previous_snapshot_sha256"] != (None if index == 0 else snapshots[-1]):
            raise ValueError("Guidance predecessor snapshot differs")
        if audit["snapshot_sha256"] in snapshots:
            raise ValueError("different rounds silently reused identical Guidance checkpoints")
        snapshots.append(audit["snapshot_sha256"])
        rows = manifest["rows"]
        if len(rows) != 80 or Counter(row["split"] for row in rows) != {"fit": 64, "validation": 16}:
            raise ValueError("Guidance training root split was changed")
        for row in rows:
            case_id, root = row["case_id"], row["root_seed"]
            actual = inventory.get(case_id)
            if (actual is None or actual["root"] != root or actual["source"] != "base"
                    or row["source"] != "base" or row["source_round_index"] != index
                    or row["target_solver_sha256"] != selected_sha):
                raise ValueError("feedback is not the actual clean Base draw for this solver")
            expected_purpose = "guidance_feedback" if index == 0 else "promotion"
            if actual["purpose"] != expected_purpose or actual["round_index"] != index:
                raise ValueError("Guidance consumed feedback from the wrong cohort/round")
            if root in feedback_roots or case_id in feedback_case_ids:
                raise ValueError("round Guidance training reuses a previous feedback root")
            feedback_roots.add(root)
            feedback_case_ids.add(case_id)
    for item in sampling:
        if item["source"] == "guided":
            index = item["round_index"] - 1
            if index not in (0, 1) or item["snapshot_sha256"] != snapshots[index]:
                raise ValueError("guided sampling did not use its round-frozen snapshot")
            if set(item["case_ids"]) & feedback_case_ids:
                raise ValueError("guided Cases leaked into Base training feedback")
    for row in agent_rows:
        if (row["nominations"] != 0 or row["base_count"] != 1 or row["guided_count"] != 1
                or row["perturbed_count"] != 1 or row["truth_received"]
                or row["source_or_g0_prematerialized"]):
            raise ValueError("diagnostic Agent diverged from its public-only integration task")
    if not candidate_audits:
        raise ValueError("formal Candidate subprocess evidence is missing")
    audit_keys = set()
    for row in candidate_audits:
        key = (row["execution_id"], row["case_index"])
        if key in audit_keys:
            raise ValueError("duplicate Candidate audit cannot count as another execution")
        audit_keys.add(key)
        if (row["status"] != "completed" or row["return_code"] != 0 or row["error_code"] is not None
                or row["process_group_cleaned"] is not True or row["candidate_process_isolated"] is not True
                or row["controller_truth_shared"] is not False):
            raise ValueError("Candidate subprocess failed, leaked truth or did not clean up")
    # Initial feedback, two Official cohorts and 2*(Base, Guided, child)
    # development cases are mandatory. Terminal may reuse the identical S0.
    phases = Counter(row["phase"] for row in candidate_audits)
    if phases["official"] < 240 or phases["development"] != 6 or phases["terminal"] < 3:
        raise ValueError("Candidate audit inventory omits required real Solver executions")
    return {"guidance_rounds": 2, "actual_feedback_roots": len(feedback_roots),
        "unique_sampled_roots": len(all_roots), "isolated_candidate_executions": len(candidate_audits),
        "candidate_execution_phase_counts": dict(phases),
        "snapshot_sha256": snapshots, "shared_round_frozen_guidance": True,
        "selected_solver_sha256": selected_sha, "base_guided_perturbed_exercised": True,
        "feedback_roots_cross_round_disjoint": True, "truth_sent_to_candidate": False,
        "llm_invoked": False, "solver_improvement_claimed": False}


def run(config_path, output):
    if not os.environ.get("SLURM_JOB_ID", "").isdigit():
        raise RuntimeError("real two-round VPFP check requires a Slurm compute allocation")
    config = diagnostic_config(load_config(config_path))
    state = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all"], text=True)
    if state.strip():
        raise ValueError("real execution check requires a clean committed source checkout")
    revision = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    writer = Writer(output)
    began = time.monotonic()
    report = {"schema": SCHEMA, "status": "running", "execution_passed": False,
        "source_revision": revision, "config_sha256": file_sha(config_path),
        "llm_invoked": False, "agent_execution": "trusted_local_diagnostic",
        "candidate_execution": "formal_subprocess", "scientific_admission": False,
        "online_assets_replaced": False, "failures": []}
    binding = candidates = None
    agent = DiagnosticAgent()
    guard = {}
    try:
        from pde_agent_verifier_solver.plugins.vpfp.iteration_backend import build_vpfp_strong_iteration_binding
        from pde_agent_verifier_solver.plugins.vpfp.solver_error_guidance import _module_sha, _json_sha
        from run_plasma_canonical_smoke import _forbid_online_vpfp_generation
        binding = build_vpfp_strong_iteration_binding(**config["backend"], output_root=output / "private_backend")
        backend = binding.plugin.generator.backend
        base = backend.capture.base
        before = (_module_sha(base.model), _json_sha(base.normalizer.to_payload()))
        codec = PythonCandidateCodec(task_kind=TaskKind.OBSERVABLE_MODEL)
        initial = codec.bind_existing((config_path.parent / config["candidate"]["initial_bundle"]).resolve(),
            storage_root=output / "controller_candidates", artifact_id="vpfp-check-s0", name="Frozen strong S0",
            summary="Existing solver, query-blind public Oracle adapter", usage="prepare then evaluate POINTS")
        initial_sha = initial.program.tree_sha256
        plan = recipe(config, initial)
        task = binding.plugin.task("observable_model")
        settings = config["execution"]
        candidates = SubprocessCandidateExecutionService(output / "candidate_execution",
            task=PythonTaskWorkerSpec.from_task_type(type(task)),
            evaluation_settings={**plan.task_settings, "default_query": plan.evaluator.default_query},
            max_workers=settings["candidate_case_workers"],
            case_hard_wall_seconds=settings["candidate_case_hard_wall_seconds"],
            development_batch_hard_wall_seconds=settings["development_candidate_batch_hard_wall_seconds"],
            python_executable=sys.executable)
        events = EventStream()
        events.subscribe(writer.event)
        events.subscribe(lambda event: persist_guidance(binding, writer)
                         if event.event_type in ("guidance_ready", "round_finished") else None)
        controller = binding.controller(plan, {config["agents"][0]["agent_id"]: agent},
            candidate_execution=candidates, local_development=True, events=events, task_id="observable_model")
        writer.write("resolved_config.json", config)
        writer.write("input_provenance.json", dict(binding.report))
        with _forbid_online_vpfp_generation(guard):
            result = controller.run()
        writer.write("campaign-result.json", result)
        writer.write("candidate_history.json", result.candidate_history)
        (output / "CAMPAIGN_MEMORY.md").write_text(result.memory.markdown + "\n")
        if len(result.rounds) != 2 or any(round.candidate_evaluations for round in result.rounds):
            raise ValueError("diagnostic unexpectedly nominated candidates or omitted a round")
        if any(round.selected.artifact.program.tree_sha256 != initial_sha for round in result.rounds):
            raise ValueError("diagnostic did not retain the same incumbent program")
        manifests = []
        for audit in backend.training_audits:
            checkpoint = audit["checkpoint_path"]
            item = audit["report"]["feedback_manifest"]
            manifests.append(json.loads(Path(item["path"]).read_text()))
        report["checks"] = verify_receipts(backend.training_audits, backend.sampling_audits,
            manifests, jsonable(candidates.audits), agent.rows, selected_sha=initial_sha)
        after = (_module_sha(base.model), _json_sha(base.normalizer.to_payload()))
        if before != after:
            raise ValueError("frozen Base weights/normalizer changed")
        retention = {"draws": len(backend._draws), "pending_draws": len(backend.capture.pending),
            "residents": len(backend.runtime._resident_cases), "anchors": len(backend.backend._anchors)}
        if any(retention.values()) or backend.capture.guide is not None:
            raise ValueError("Campaign retained private draw/resident/anchor state after release")
        if inspect_candidate_bundle(initial.program.bundle_locator).tree_sha256 != initial_sha:
            raise ValueError("frozen S0 bundle changed")
        report.update(status="passed", execution_passed=True, base_final_verified=True,
            initial_solver_final_verified=True, round_checkpoints_final_verified=True,
            resident_counts_after_run=retention, online_coarse_solver_guard=guard)
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        report["failures"].append({"type": type(error).__name__, "message": str(error)})
        (output / "failure_traceback.txt").write_text(traceback.format_exc())
    finally:
        report["wall_seconds"] = time.monotonic() - began
        report["online_coarse_solver_guard"] = guard
        writer.write("agent-diagnostics.json", agent.rows)
        writer.write("candidate_execution_audits.json", () if candidates is None else candidates.audits)
        persist_guidance(binding, writer)
        final_revision = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        final_state = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all"], text=True)
        report["source_final_verified"] = final_revision == revision and not final_state.strip()
        if not report["source_final_verified"]:
            report.update(status="failed", execution_passed=False)
            report["failures"].append({"type": "SourceMutation", "message": "source changed during real execution check"})
        writer.write("acceptance-summary.json", report)
        writer.write("status.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/vpfp_strong_canonical_2round.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick-only", action="store_true",
        help="Only actual batch-four capture/decode and one isolated S0 evaluation; no Guidance training.")
    args = parser.parse_args(argv)
    report = (run_quick if args.quick_only else run)(args.config.resolve(), args.output.resolve())
    name = "quick-summary.json" if args.quick_only else "acceptance-summary.json"
    print(json.dumps({"status": report["status"], "report": str(args.output / name)}), flush=True)
    return 0 if report["execution_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
