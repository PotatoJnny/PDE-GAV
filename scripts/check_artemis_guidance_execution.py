#!/usr/bin/env python3
"""Bounded real-Artemis Guidance wiring check, not a model-quality gate.

The unchanged canonical Controller runs two rounds with two trusted local
diagnostic Agents. They request Guided/PERT pairs through its public sampler
and evaluate an observation-only wrapper around S0 in actual CPU Candidate
workers. No nominations are made: S1 keeps S0, so only one solver is evaluated
on each private feedback/Official cohort. No LLM API or qualification study is
invoked. Reward fitting and transport retain the configured physical math.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pde_agent_verifier_solver.engine import (
    AgentSpec, CampaignRecipe, CaseCounts, DevelopmentSamplingPolicy,
    EvaluatorPolicy, EventStream, GeneratorRecipe, PythonCandidateCodec,
    PythonTaskWorkerSpec, RoundSettings, SubprocessCandidateExecutionService,
)
from pde_agent_verifier_solver.engine.evaluator import EvaluationRequest
from pde_agent_verifier_solver.engine.runtime import AgentOutcome
from pde_agent_verifier_solver.sdk import (
    AnswerQuery, GenerationKind, HiddenSuiteSpec, OfficialEvaluationSpec,
    PairedSampleRequest, SampleSource, SamplingStrategy, TaskKind,
    write_candidate_bundle_manifest,
)

FIELDS = ("M0", "M1", "M2", "M3", "E")
AGENTS = ("guidance-smoke-a", "guidance-smoke-b")

# Eight diagnostic evaluations exercise source RPC and its causal denial; the
# private feedback and Official cohorts use the original S0 bundle unchanged.
# These checks exist only in this smoke artifact, not in the deployed solver.
PROBE_SOURCE = '''\
import numpy as np
from s0_baseline import prepare_solution as prepare_s0
from pde_agent_verifier_solver.sdk import (
    open_public_case_session, open_development_solution_session,
)

def prepare_solution(case, solve_budget=None):
    domain = case.public_data["equation"]["domain"]
    start, end = domain["time_interval"]
    x = float(case.public_data["arrays"]["x"][0])
    velocity = sum(domain["velocity_query_window"]) / 2
    session = open_public_case_session(case)
    value = np.asarray(session.query_source(np.array([[end, x, velocity]])))
    if value.shape != (1,) or not np.isfinite(value).all():
        raise AssertionError("real source RPC returned an invalid scalar")
    try:
        session.query_source(np.array([[end + max(1., end - start), x, velocity]]))
    except ValueError:
        pass
    else:
        raise AssertionError("future source RPC was accessible")
    try:
        open_development_solution_session(case)
    except (RuntimeError, PermissionError):
        pass
    else:
        raise AssertionError("private completed solution was accessible")
    return prepare_s0(case, solve_budget)
'''


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def smoke_recipe(initial, official_count=96):
    if isinstance(official_count, bool) or official_count not in (48, 96):
        raise ValueError("bounded Guidance smoke official_count must be 48 or 96")
    return CampaignRecipe(
        name="artemis-real-guidance-engineering-smoke", round_count=2,
        agents=tuple(AgentSpec(name, "Trusted bounded Guidance diagnostic") for name in AGENTS),
        default_round=RoundSettings(
            agent_budget_seconds=300,
            development_sampling=DevelopmentSamplingPolicy(
                strategy=SamplingStrategy.AGENT_ADAPTIVE,
                allowed_sources=(SampleSource.BASE, SampleSource.GUIDED),
                max_cases_per_agent_per_round=2, starter=CaseCounts(),
            ),
            official_evaluation=OfficialEvaluationSpec(
                round=HiddenSuiteSpec(base=official_count),
                terminal=HiddenSuiteSpec(base=1, base_perturbed=1),
            ),
        ),
        generator=GeneratorRecipe(guidance_enabled=True, perturbations_enabled=True,
                                  settings={"guidance_maximum_wall_seconds": 120.0}),
        evaluator=EvaluatorPolicy(AnswerQuery.at_points(fields=FIELDS,
            points=({"t": 0.5, "x": 0.0},)), development_point_count=2),
        task_settings={"require_prepared_solution": True, "causal_horizon": True,
                       "candidate_compute_budget": 120., "development_point_count": 2,
                       "official_point_count": 2, "terminal_point_count": 2},
        master_seed=101, initial_candidate=initial,
        promotion={"preset": "minimize_metric", "settings": {
            "primary": {"distribution": "base", "statistic": "mean"}}},
    )


class SessionMeter:
    def __init__(self, target, audit):
        self.target, self.audit = target, audit

    def __getattr__(self, name):
        method = getattr(self.target, name)
        if not callable(method):
            return method
        def call(*args, **kwargs):
            row = self.audit["methods"].setdefault(name, {"calls": 0, "failures": 0})
            row["calls"] += 1
            try:
                return method(*args, **kwargs)
            except BaseException:
                row["failures"] += 1
                raise
        return call

    def close(self):
        self.target.close()
        self.audit["closed"] = True


class GeneratorMeter:
    """Controller-private observation only; all scientific calls delegate."""

    def __init__(self, target):
        self.target = target
        self.sessions, self.generations, self.updates = [], [], []
        self.guidance_states = []

    def __getattr__(self, name):
        return getattr(self.target, name)

    def update_guidance(self, state, feedback, settings, progress=None):
        began = perf_counter()
        updated = self.target.update_guidance(state, feedback, settings, progress)
        self.updates.append({
            "source_round_index": feedback.source_round_index,
            "target_round_index": feedback.target_round_index,
            "previous_guidance_reused": updated.guidance is state.guidance,
            "previous_model_reused": (state.guidance is not None
                                      and updated.guidance.model is state.guidance.model),
            "feedback_target_solver_id": feedback.selected_program_id,
            "updated_target_solver_id": updated.guidance.target_solver_id,
            "base_object_preserved": updated.base is state.base,
            "snapshot_sha256": updated.guidance.snapshot_sha256,
            "wall_seconds": perf_counter() - began,
        })
        self.guidance_states.append(updated.guidance)
        return updated

    def sample(self, state, request, progress=None):
        records = self.target.sample(state, request, progress)
        self.generations.append({
            "round_index": request.round_index, "kind": request.kind.value,
            "purpose": request.purpose.value, "count": len(records),
            "guidance_snapshot_sha256": getattr(state.guidance, "snapshot_sha256", None),
            "cases": [{"case_id": row.case.case_id, "parent_case_id": row.case.parent_case_id}
                      for row in records],
        })
        return records

    def open_case_session(self, state, case, access):
        target = self.target.open_case_session(state, case, access)
        row = {"case_id": case.case_id, "access": access.value, "closed": False, "methods": {}}
        self.sessions.append(row)
        return SessionMeter(target, row)


class GuidedProbeAgent:
    def __init__(self, probe):
        self.probe, self.turns = probe, []

    def run(self, context):
        batch = context.sampler.sample_paired_cases(PairedSampleRequest(SampleSource.GUIDED, 1))
        if batch.count != 2 or batch.remaining != 0:
            raise ValueError("Guided/PERT pair did not spend the shared two-Case allowance")
        result = context.evaluator.evaluate(self.probe, EvaluationRequest(
            tuple(case.case_id for case in batch.cases)))
        if not math.isfinite(result.primary_error) or any(row.truth is not None for row in result.cases):
            raise ValueError("Guided pair failed private macro scoring or leaked truth")
        self.turns.append({"agent_id": context.agent.agent_id,
                           "round_index": context.round_plan.round_index,
                           "case_count": batch.count, "remaining": batch.remaining,
                           "primary_error": result.primary_error})
        return AgentOutcome(observations=(
            "Engineering smoke: real Guided/PERT sampling and private scoring passed; keep incumbent.",))


def bind_candidates(root):
    codec = PythonCandidateCodec(task_kind=TaskKind.OBSERVABLE_MODEL)
    baseline = ROOT / "configs/candidates/artemis_hold_initial_bundle"
    initial = codec.bind_existing(baseline, storage_root=root / "candidates", artifact_id="guidance-smoke-s0",
        name="Existing Artemis S0", summary="Unchanged hold-initial baseline", usage="prepare_solution then evaluate")
    probe_root = root / "probe-source"
    probe_root.mkdir()
    shutil.copyfile(baseline / "submission.py", probe_root / "s0_baseline.py")
    (probe_root / "submission.py").write_text(PROBE_SOURCE)
    write_candidate_bundle_manifest(probe_root, files=("submission.py", "s0_baseline.py"))
    probe = codec.bind_existing(probe_root, storage_root=root / "candidates", artifact_id="guidance-smoke-probe",
        name="S0 with bounded public RPC probe", summary="Same S0 predictions; smoke-only causal source checks",
        usage="prepare_solution then evaluate")
    return initial, probe


def validate_evidence(evidence, official_count):
    """Require engineering completion, not gain, activation or nonzero motion."""
    training, sampling = evidence["training_audits"], evidence["sampling_audits"]
    if evidence["rounds_completed"] != 2 or len(training) != 2:
        raise ValueError("two canonical rounds and two real Guidance fits are required")
    if [row["source_round_index"] for row in training] != [0, 1]:
        raise ValueError("Guidance did not train on S0 then S1 feedback")
    if any(row["feedback_case_count"] != official_count for row in training):
        raise ValueError("Guidance feedback did not match the Controller's Official Base count")
    for row in training:
        counts = row["condition_counts"]
        values = list(counts.values()) if isinstance(counts, dict) else list(counts)
        if (len(values) != 6 or min(values) < evidence["configured_guidance"]["crossfit_fold_count"]
                or sum(values) != official_count):
            raise ValueError("real feedback did not cover all six conditions for configured crossfit folds")
    updates = evidence["generator_updates"]
    if len(updates) != 2 or any(row["previous_guidance_reused"] or not row["base_object_preserved"] for row in updates):
        raise ValueError("Guidance update did not produce a fresh state with the same frozen Base")
    if any(row["previous_model_reused"] for row in updates):
        raise ValueError("Guidance update reused the previous round's trained model object")
    if any(row["feedback_target_solver_id"] != row["updated_target_solver_id"] for row in updates):
        raise ValueError("Guidance model target does not match the current incumbent feedback")
    if len({row["snapshot_sha256"] for row in training}) != 2:
        raise ValueError("two feedback updates produced indistinguishable snapshot identities")
    for round_index in (1, 2):
        active = [row for row in sampling if row["round_index"] == round_index and row["purpose"] == "development"]
        expected = training[round_index - 1]["snapshot_sha256"]
        if sum(row["count"] for row in active) < 2 or any(row["snapshot_sha256"] != expected for row in active):
            raise ValueError("same-round Agents did not use the same frozen Guidance snapshot")
        turns = [row for row in evidence["agent_turns"] if row["round_index"] == round_index]
        if {row["agent_id"] for row in turns} != set(AGENTS) or len(turns) != 2:
            raise ValueError("both deterministic Agents must complete each round")
        if any(row["case_count"] != 2 or row["remaining"] != 0 for row in turns):
            raise ValueError("Guided/PERT sampling did not obey the shared case allowance")
    paired = evidence["paired_sampling_attempts"]
    accepted = [row for row in paired if row["accepted"] and row["source"] == "guided"]
    if len(accepted) != 4 or any(not row["root_case_id"] or not row["child_case_id"] for row in accepted):
        raise ValueError("four Guided-parent perturbation lineages are required")
    audits = evidence["candidate_execution_audits"]
    if not audits or any(row["status"] != "completed" or row["return_code"] != 0
        or not row["candidate_process_isolated"] or row["controller_truth_shared"]
        or not row["process_group_cleaned"] for row in audits):
        raise ValueError("Candidate subprocess/private-scoring/cleanup audit failed")
    sessions = evidence["public_sessions"]
    source_calls = sum(row["methods"].get("query_source", {}).get("calls", 0) for row in sessions)
    if source_calls < 8 or not sessions or any(not row["closed"] for row in sessions):
        raise ValueError("Guided/PERT source RPC or public-session cleanup was not exercised")
    if evidence["resident_roots"] or evidence["resident_records"] or evidence["retained_feedback_coordinates"]:
        raise ValueError("Controller left physical records or Guidance feedback coordinates resident")


def run_smoke(binding, root, official_count):
    initial, probe = bind_candidates(root)
    meter = GeneratorMeter(binding.plugin.generator)
    binding = replace(binding, plugin=replace(binding.plugin, generator=meter))
    guide = meter.backend.guidance_adapter
    if guide is None or guide.config.guidance_strength <= 0:
        raise ValueError("smoke requires the real configured nonzero Guidance adapter")
    if guide.config.model_kind != "ou_hermite_ridge_v1":
        raise ValueError("smoke must retain the existing OU/Hermite Guidance model")
    recipe = smoke_recipe(initial, official_count)
    task = binding.plugin.task("observable_model")
    service = SubprocessCandidateExecutionService(root / "candidate_execution",
        task=PythonTaskWorkerSpec.from_task_type(type(task)),
        evaluation_settings={**recipe.task_settings, "default_query": recipe.evaluator.default_query},
        max_workers=4, case_hard_wall_seconds=180, development_batch_hard_wall_seconds=600,
        python_executable=sys.executable)
    agents = {name: GuidedProbeAgent(probe) for name in AGENTS}
    events = EventStream()
    def progress(event):
        if event.event_type in {"round_started", "guidance_ready", "round_finished", "campaign_completed"}:
            print(json.dumps({"event": event.event_type, "round_index": event.round_index}), flush=True)
    events.subscribe(progress)
    controller = binding.controller(recipe, agents, candidate_execution=service,
        local_development=True, task_id="observable_model", events=events)
    evidence = {"rounds_completed": 0}
    try:
        result = controller.run()
        evidence.update(rounds_completed=len(result.rounds), terminal_round_reports=len(result.terminal_round_reports),
            final_macro_score=result.selected.primary_error,
            paired_sampling_attempts=[{**asdict(row), "source": row.source.value} for row in result.sampling_attempts],
            official_agent_case_counts=[dict(row.agent_case_counts) for row in result.rounds])
    finally:
        source = meter.backend.source
        evidence.update(training_audits=list(guide.training_audits), sampling_audits=list(guide.sampling_audits),
            generator_updates=meter.updates, generations=meter.generations,
            agent_turns=[row for agent in agents.values() for row in agent.turns],
            candidate_execution_audits=[asdict(row) for row in service.audits],
            public_sessions=meter.sessions, resident_roots=len(source._roots),
            resident_records=len(source._issued_records), retained_feedback_coordinates=len(guide._coordinates),
            configured_guidance=guide.config.to_dict())
        write_json(root / "execution-evidence.json", evidence)
    validate_evidence(evidence, official_count)
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--official-count", type=int, choices=(48, 96), default=96)
    args = parser.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_TMPDIR"):
        parser.error("real Guidance smoke requires a Slurm compute allocation")
    root = args.output_root.resolve()
    if root.exists():
        parser.error("output root already exists; preserve it and choose a fresh directory")
    root.mkdir(parents=True)
    began = perf_counter()
    summary = {"schema": "artemis_guidance_execution_smoke_v1", "status": "failed",
               "rounds_completed": 0, "guidance_fit_count": 0, "guided_case_count": 0,
               "model_api_called": False, "scientific_quality_gate": False,
               "difficulty_gain_required": False, "activation_required": False,
               "real_model_training_requested": True, "candidate_case_workers": 4,
               "agent_execution": "trusted_local_diagnostic_not_llm", "candidate_execution": "cpu_subprocess_private_scoring",
               "official_count": args.official_count}
    try:
        summary["canonical_commit"] = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        summary["backend_config_sha256"] = sha256(args.backend_config.read_bytes()).hexdigest()
        config = json.loads(args.backend_config.read_text())
        if config.get("physics") != "artemis" or config["factory_settings"].get("device") != "cuda":
            raise ValueError("real Guidance smoke requires the configured Artemis CUDA backend")
        from pde_agent_verifier_solver.plugins.vp_1d1v.product import load_artemis_backend
        binding = load_artemis_backend(**config["factory_settings"])
        evidence = run_smoke(binding, root, args.official_count)
        summary.update(status="passed", rounds_completed=evidence["rounds_completed"],
            guidance_fit_count=len(evidence["training_audits"]),
            guided_case_count=sum(row["count"] for row in evidence["sampling_audits"]),
            candidate_evaluations=len(evidence["candidate_execution_audits"]),
            final_macro_score=evidence["final_macro_score"], frozen_base_preserved=True,
            same_round_guidance_frozen=True, source_cleanup_confirmed=True)
    except BaseException as error:
        summary.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    summary["wall_seconds"] = perf_counter() - began
    write_json(root / "acceptance-summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
