#!/usr/bin/env python3
"""Real-asset, no-LLM acceptance of R1 regeneration and active solve timing.

Run only in a Slurm compute allocation.  Recreate the canonical 128-Case R1
Official cohort and private plans, then evaluate S0 and the four byte-frozen
nominees on two preselected surviving reference Cases.  This is engineering
acceptance, not replacement Official scores or a scientific ranking.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
from threading import Lock
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from artemis_round_recovery import load_frozen_round_one, stage_frozen_round_one, verify_official_replay
import run_canonical_artemis_campaign as canonical
from pde_agent_verifier_solver.engine import EventStream, PythonTaskWorkerSpec, SubprocessCandidateExecutionService
from pde_agent_verifier_solver.engine.evaluator import HiddenVerifier
from pde_agent_verifier_solver.engine.seeds import SeedLedger
from pde_agent_verifier_solver.sdk import EvaluationPhase, GenerationPurpose

SOURCE_HEAD = "da3cb03449dfd725cc72a145e591c0776f3f44fa"
SOURCE_CONFIG_SHA256 = "61d4a99f99fa67caef04e99fe85dd93a39c4c243c631b7ab4ad977e49d9242cd"
BACKEND_SHA256 = "0433ef1edefce3c49dc0f085ec0f886c467e350eddf9d5c187e6ca8eda1b3651"
SMOKE_INDICES = (0, 55)


def validate_case_metrics(metrics):
    required = ("candidate_solve_wall_seconds", "candidate_solve_active_seconds",
                "candidate_platform_wait_seconds", "candidate_oracle_queue_wait_seconds",
                "candidate_oracle_lock_wait_seconds", "candidate_oracle_timed_requests")
    for name in required:
        value = metrics.get(name)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"missing or invalid real solve-budget metric: {name}")
    wall, active, wait = (metrics[key] for key in required[:3])
    if active > 600.0 or active > wall + 0.02 or abs(wall - active - wait) > 0.02:
        raise ValueError("real solve budget does not equal wall minus platform waiting")
    if abs(wait - metrics[required[3]] - metrics[required[4]]) > 1e-6:
        raise ValueError("real platform wait does not match queue plus lock waiting")


def run_smoke(source_campaign: Path, output_root: Path, writer):
    frozen = load_frozen_round_one(source_campaign, expected_source_head=SOURCE_HEAD,
        expected_config_sha256=SOURCE_CONFIG_SHA256, expected_backend_config_sha256=BACKEND_SHA256)
    frozen = stage_frozen_round_one(frozen, output_root / "recovered_r1_candidates")
    writer.write_json("recovery-source-manifest.json", frozen.manifest)
    config, config_path = canonical._load_config(ROOT / "configs/artemis_canonical_5round.json")
    # Resolved source files append deployment provenance; none is a scientific
    # campaign setting or permission to change the original experiment.
    deployment_keys = {"source_config", "config_sha256", "run_root", "worker_gpu_tokens", "controller_gpu_token"}
    source_campaign_config = {key: value for key, value in frozen.source_config.items() if key not in deployment_keys}
    if canonical._canonical_json(config) != canonical._canonical_json(source_campaign_config):
        raise ValueError("smoke must retain the complete original canonical campaign configuration")
    inputs = canonical._static_inputs(config, config_path)
    events = EventStream()
    controller, _unused_agent_service, provenance = canonical._compose(
        config=config, config_path=config_path, inputs=inputs, run_root=output_root,
        codex_binary=Path("/nonexistent/no-agent-is-launched"),
        auth_file=Path("/nonexistent/no-auth-is-read"), worker_gpus=("1", "2"),
        controller_gpu="0", events=events,
    )
    # _compose is unchanged formal composition.  Its unused executor has never
    # run a task; the acceptance slice intentionally uses only four CPU slots.
    controller.candidate_execution._executor.shutdown(wait=True)
    settings = {**controller.campaign.task_settings,
                "default_query": controller.campaign.evaluator.default_query}
    service = SubprocessCandidateExecutionService(output_root / "smoke_candidate_execution",
        task=PythonTaskWorkerSpec.from_task_type(type(controller.task)), evaluation_settings=settings,
        max_workers=4, case_hard_wall_seconds=900, development_batch_hard_wall_seconds=6900,
        python_executable=sys.executable)
    controller.candidate_execution = service
    # Initialize only the bookkeeping required by the existing canonical
    # loader/sampler.  No Agent, promotion, Guidance fit or Terminal is run.
    controller._seeds = SeedLedger(controller.campaign.master_seed)
    controller._generator_operation_lock = Lock()
    controller._case_samplers = []
    generator_settings = {"basis_size": controller.campaign.generator.basis_size,
                          **controller.campaign.generator.settings,
                          "master_seed": controller.campaign.master_seed,
                          "controller_seed": controller._seeds.derive("generator", "base")}
    checkpoint = controller.campaign.generator.checkpoint
    if checkpoint is None:
        raise ValueError("recovery smoke must use the existing frozen Base checkpoint")
    evidence = {"source_manifest_sha256": frozen.source_manifest_sha256,
                "source_manifest": dict(frozen.manifest), "cases": [], "backend": provenance["backend"],
                "source_state": canonical._source_state(), "smoke_indices": list(SMOKE_INDICES)}
    try:
        state = controller.generator.load(checkpoint, generator_settings,
                                           controller._generator_progress(None, "base_loading"))
        controller._loaded_generator_state = state
        plan = controller.campaign.rounds[0]
        began = time.perf_counter()
        hidden, _, _ = controller._hidden_suite(state, plan, plan.round_hidden,
            GenerationPurpose.PROMOTION, 0, generator_settings)
        if set(hidden) != {"base"}:
            raise ValueError("recovery smoke expects exactly the original R1 clean Base distribution")
        records = hidden["base"]
        provider = controller._hidden_plan_seed_provider("official_round:1", EvaluationPhase.OFFICIAL, hidden)
        verifier = HiddenVerifier(controller.evaluation, controller.campaign.evaluator.default_query,
            controller.campaign.task_settings, phase=EvaluationPhase.OFFICIAL,
            seed=controller._seeds.derive("official_round:1", EvaluationPhase.OFFICIAL.value),
            plan_seed_provider=provider, generator=controller.generator, generator_state=state,
            session_lock=controller._generator_operation_lock, candidate_execution=service)
        private_plans = tuple(verifier._plan_for(record, index) for index, record in enumerate(records))
        evidence["replay_verification"] = verify_official_replay(frozen,
            tuple(record.case for record in records), private_plans)
        evidence["cohort_regeneration_wall_seconds"] = time.perf_counter() - began
        if not set(SMOKE_INDICES) <= {reference.case_index for reference in frozen.references}:
            raise ValueError("smoke indices must be surviving saved numerical reference Cases")
        subset = tuple(records[index] for index in SMOKE_INDICES)
        initial = controller._freeze_initial_candidate(controller.campaign.initial_candidate)
        if initial.program.tree_sha256 != frozen.manifest["initial_bundle_tree_sha256"]:
            raise ValueError("regenerated S0 does not match the original frozen S0")
        writer.write_json("partial-evidence.json", evidence)
        print(json.dumps({"stage": "r1_cohort_verified", **evidence["replay_verification"]}), flush=True)
        for artifact in (initial, *frozen.artifacts):
            evaluated = verifier.evaluate(artifact, subset)
            if len(evaluated.cases) != len(SMOKE_INDICES) or not math.isfinite(evaluated.primary_error):
                raise ValueError("real frozen candidate did not return finite complete smoke results")
            for index, result in zip(SMOKE_INDICES, evaluated.cases, strict=True):
                if not math.isfinite(result.primary_error):
                    raise ValueError("real smoke contains a non-finite scientific error")
                metrics = dict(result.metrics)
                validate_case_metrics(metrics)
                if artifact.round_index == 1 and metrics.get("candidate_budget_clock_compat_aliases", 0) < 1:
                    raise ValueError("frozen R1 candidate legacy budget clock was not adapted")
                evidence["cases"].append({"candidate_id": artifact.artifact_id, "case_index": index,
                    "bundle_tree_sha256": artifact.program.tree_sha256,
                    "primary_error": result.primary_error, "metrics": metrics})
            writer.write_json("partial-evidence.json", evidence)
            print(json.dumps({"stage": "frozen_candidate_smoke_complete", "artifact_id": artifact.artifact_id,
                              "case_count": len(evaluated.cases), "mean_error": evaluated.primary_error}), flush=True)
        evidence["candidate_execution_audits"] = [asdict(row) for row in service.audits]
        if len(service.audits) != 10 or any(row.status != "completed" or not row.process_group_cleaned
                                          or row.error_code is not None for row in service.audits):
            raise ValueError("real smoke worker completion/cleanup audit failed")
        return evidence
    finally:
        controller._close_case_samplers(controller._case_samplers)
        service._executor.shutdown(wait=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-campaign", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_TMPDIR"):
        parser.error("real recovery smoke requires a Slurm compute allocation")
    root = args.output_root.resolve()
    if root.exists():
        parser.error("output root must be new; existing acceptance artifacts are preserved")
    root.mkdir(parents=True, mode=0o700)
    writer = canonical._RunWriter(root)
    started = time.perf_counter()
    summary = {"schema": "artemis_budget_recovery_smoke_v1", "status": "failed",
               "source_state": canonical._source_state(), "canonical_commit": canonical._source_state()["commit"],
               "backend_config_sha256": BACKEND_SHA256, "model_api_called": False,
               "agent_iteration_started": False, "scientific_ranking_claimed": False,
               "candidate_case_workers": 4, "candidate_case_count": 2, "candidate_count": 5,
               "formal_official_replacement": False, "real_model_assets_used": True}
    try:
        if not summary["source_state"]["commit"] or summary["source_state"]["status_porcelain"]:
            raise ValueError("real recovery acceptance requires a committed clean source worktree")
        evidence = run_smoke(args.source_campaign, root, writer)
        writer.write_json("execution-evidence.json", evidence)
        summary.update(status="passed", candidate_evaluations=len(evidence["cases"]),
                       replay_verification=evidence["replay_verification"],
                       source_manifest_sha256=evidence["source_manifest_sha256"],
                       active_wait_accounting_validated=True, immutable_r1_bundles_preserved=True)
    except BaseException as error:
        summary.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    summary["wall_seconds"] = time.perf_counter() - started
    writer.write_json("acceptance-summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
