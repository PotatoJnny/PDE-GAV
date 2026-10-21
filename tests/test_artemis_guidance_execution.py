"""Small bookkeeping checks; real Guidance assets run only in the Slurm smoke."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("artemis_guidance_execution_smoke",
    ROOT / "scripts/check_artemis_guidance_execution.py")
SMOKE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SMOKE)


def evidence_fixture():
    training = [{"source_round_index": index, "feedback_case_count": 48,
                 "condition_counts": [8] * 6, "snapshot_sha256": str(index) * 64,
                 "model_report": {"activation_guard_passed": False}}
                for index in range(2)]
    return {
        "rounds_completed": 2,
        "training_audits": training,
        "configured_guidance": {"crossfit_fold_count": 6},
        "generator_updates": [{"previous_guidance_reused": False, "base_object_preserved": True,
                               "previous_model_reused": False,
                               "feedback_target_solver_id": "incumbent-s0",
                               "updated_target_solver_id": "incumbent-s0"}
                              for _ in range(2)],
        "sampling_audits": [{"round_index": round_index, "purpose": "development", "count": 1,
                             "snapshot_sha256": training[round_index - 1]["snapshot_sha256"],
                             "diagnostics": {"effective_guidance_strength": 0.0}}
                            for round_index in (1, 2) for _ in range(2)],
        "agent_turns": [{"round_index": round_index, "agent_id": name,
                         "case_count": 2, "remaining": 0}
                        for round_index in (1, 2) for name in SMOKE.AGENTS],
        "paired_sampling_attempts": [{"source": "guided", "accepted": True,
                                      "root_case_id": f"root-{index}", "child_case_id": f"child-{index}"}
                                     for index in range(4)],
        "candidate_execution_audits": [{"status": "completed", "return_code": 0,
                                        "candidate_process_isolated": True,
                                        "controller_truth_shared": False, "process_group_cleaned": True}],
        "public_sessions": [{"closed": True, "methods": {"query_source": {"calls": 8}}}],
        "resident_roots": 0, "resident_records": 0, "retained_feedback_coordinates": 0,
    }


def test_recipe_uses_actual_controller_feedback_counts_and_two_guided_agents():
    from pde_agent_verifier_solver.sdk import SampleSource
    recipe = SMOKE.smoke_recipe(None)
    assert recipe.round_count == len(recipe.agents) == 2
    assert recipe.generator.guidance_enabled and recipe.generator.perturbations_enabled
    assert recipe.default_round.official_evaluation.round.base == 96
    assert recipe.default_round.official_evaluation.round.guided == 0
    assert SampleSource.GUIDED in recipe.default_round.development_sampling.allowed_sources
    assert recipe.default_round.development_sampling.max_cases_per_agent_per_round == 2
    assert recipe.evaluator.allow_development_solution_session is False
    assert recipe.evaluator.allow_point_truth is False
    assert recipe.task_settings["require_prepared_solution"] is True
    assert recipe.promotion["settings"]["primary"]["distribution"] == "base"
    with pytest.raises(ValueError, match="48 or 96"):
        SMOKE.smoke_recipe(None, official_count=384)


def test_engineering_smoke_accepts_valid_noop_guidance_without_quality_claim():
    SMOKE.validate_evidence(evidence_fixture(), 48)


@pytest.mark.parametrize("field,value,error", [
    ("rounds_completed", 1, "two canonical rounds"),
    ("resident_roots", 1, "left physical records"),
    ("resident_records", 1, "left physical records"),
    ("retained_feedback_coordinates", 1, "left physical records"),
])
def test_engineering_gate_rejects_incomplete_or_leaking_execution(field, value, error):
    evidence = evidence_fixture()
    evidence[field] = value
    with pytest.raises(ValueError, match=error):
        SMOKE.validate_evidence(evidence, 48)


@pytest.mark.parametrize("mutation,error", [
    (lambda e: e["training_audits"][1].update(source_round_index=0), "S0 then S1"),
    (lambda e: e["training_audits"][0].update(feedback_case_count=96), "Official Base count"),
    (lambda e: e["training_audits"][0].update(condition_counts=[1, 7, 10, 10, 10, 10]), "all six conditions"),
    (lambda e: e["training_audits"][0].update(condition_counts=[5, 7, 9, 9, 9, 9]), "all six conditions"),
    (lambda e: e["generator_updates"][0].update(previous_guidance_reused=True), "fresh state"),
    (lambda e: e["generator_updates"][1].update(previous_model_reused=True), "previous round's trained model"),
    (lambda e: e["generator_updates"][1].update(updated_target_solver_id="stale-solver"), "current incumbent feedback"),
    (lambda e: e["sampling_audits"][1].update(snapshot_sha256="z" * 64), "same-round Agents"),
    (lambda e: e["agent_turns"][0].update(remaining=1), "shared case allowance"),
    (lambda e: e["candidate_execution_audits"][0].update(controller_truth_shared=True), "private-scoring"),
    (lambda e: e["candidate_execution_audits"][0].update(process_group_cleaned=False), "private-scoring"),
    (lambda e: e["public_sessions"][0].update(closed=False), "source RPC"),
])
def test_engineering_gate_rejects_broken_wiring(mutation, error):
    evidence = evidence_fixture()
    mutation(evidence)
    with pytest.raises(ValueError, match=error):
        SMOKE.validate_evidence(evidence, 48)


def test_probe_preserves_original_s0_bytes_and_statically_binds(tmp_path):
    initial, probe = SMOKE.bind_candidates(tmp_path)
    assert initial.program.tree_sha256 != probe.program.tree_sha256
    assert (tmp_path / "probe-source/s0_baseline.py").read_bytes() == (
        ROOT / "configs/candidates/artemis_hold_initial_bundle/submission.py").read_bytes()
    assert "return prepare_s0(case, solve_budget)" in SMOKE.PROBE_SOURCE
    assert "query_source" in SMOKE.PROBE_SOURCE
    assert "future source RPC was accessible" in SMOKE.PROBE_SOURCE
    assert initial.program._loaded is probe.program._loaded is None


def test_main_refuses_real_assets_without_compute_allocation(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(SystemExit) as error:
        SMOKE.main(["--backend-config", "unused.json", "--output-root", str(tmp_path / "smoke")])
    assert error.value.code == 2
    assert not (tmp_path / "smoke").exists()


def test_main_preserves_actual_failure_without_claiming_acceptance(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "fixture-not-real")
    monkeypatch.setenv("SLURM_TMPDIR", str(tmp_path))
    config = tmp_path / "backend.json"
    config.write_text(json.dumps({"physics": "artemis", "factory_settings": {"device": "cpu"}}))
    monkeypatch.setattr(SMOKE.subprocess, "check_output", lambda *args, **kwargs: "abc\n")
    output = tmp_path / "smoke"
    assert SMOKE.main(["--backend-config", str(config), "--output-root", str(output)]) == 1
    summary = json.loads((output / "acceptance-summary.json").read_text())
    assert summary["status"] == "failed" and summary["rounds_completed"] == 0
    assert summary["model_api_called"] is False and summary["scientific_quality_gate"] is False
    assert "CUDA backend" in summary["error"]
