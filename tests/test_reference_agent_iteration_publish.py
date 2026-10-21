from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from pde_agent_verifier_solver.engine import reference_agent_iteration as iteration
from pde_agent_verifier_solver.engine.reference_agent_iteration import (
    _terminal_claim_metrics,
    _terminal_payload,
    _round_payload,
)
from pde_agent_verifier_solver.engine.reference_agent_runtime import (
    load_python_macro_candidate,
    python_bundle_sha256,
)
from pde_agent_verifier_solver.engine.selection import (
    EvidenceStage,
    PairedObjectivePolicy,
)
from pde_agent_verifier_solver.sdk import CandidateRef, CandidateSubmission, ExecutionAssurance
from pde_agent_verifier_solver.sdk.candidate_bundle import (
    inspect_candidate_bundle,
    write_candidate_bundle_manifest,
)
from tests.test_guidance_controller import config as guidance_config


def _candidate(candidate_id: str):
    return SimpleNamespace(
        candidate_id=candidate_id,
        bundle_sha256="a" * 64,
        round_index=1,
        agent_id="agent",
        parent_sha256="b" * 64,
    )


def _batch(statuses, failures):
    return SimpleNamespace(
        scores=tuple(
            SimpleNamespace(status=status, failures=tuple(row_failures))
            for status, row_failures in zip(statuses, failures)
        ),
        root_ids=("root-1", "root-1", "root-2", "root-2"),
    )


def test_invalid_terminal_publishes_failures_and_no_claim_metrics():
    cold_batch = _batch(
        ("valid", "invalid", "rejected", "valid"),
        ((), ("candidate_execution_failed",), ("macro_gate_rejected",), ()),
    )
    selected_batch = _batch(("valid",) * 4, ((),) * 4)
    terminal = SimpleNamespace(
        cohort_id="terminal",
        cohort_sha256="c" * 64,
        cold_start=cold_batch,
        selected=selected_batch,
        cold_start_score=SimpleNamespace(
            errors=(1.0, 1.0e30, 2.0, 1.0),
            eligible=False,
        ),
        selected_score=SimpleNamespace(
            errors=(0.5, 0.5, 0.5, 0.5),
            eligible=True,
        ),
        comparison=None,
        supports_selected_over_cold_start=False,
        cold_start_objective_aggregates=None,
        selected_objective_aggregates=None,
        perturbed_guardrail=None,
    )
    result = SimpleNamespace(
        terminal=terminal,
        cold_start=_candidate("cold"),
        selected=_candidate("selected"),
        paired_perturbations=True,
        paired_objective_policy=SimpleNamespace(
            mode=SimpleNamespace(value="weighted_composite")
        ),
    )

    payload = _terminal_payload(result)

    assert _terminal_claim_metrics(terminal) == (None, None, None)
    assert payload["evaluation_status"] == "invalid"
    assert payload["root_count"] == 2
    assert payload["cold_start"]["eligible"] is False
    assert payload["cold_start"]["failures"] == [
        [],
        ["candidate_execution_failed"],
        ["macro_gate_rejected"],
        [],
    ]
    assert payload["cold_start"]["failure_counts"] == {
        "candidate_execution_failed": 1,
        "macro_gate_rejected": 1,
    }
    assert payload["cold_start"]["valid_member_count"] == 2
    assert payload["cold_start"]["invalid_member_count"] == 1
    assert payload["cold_start"]["rejected_member_count"] == 1
    assert payload["cold_start"]["valid_root_count"] == 0
    assert payload["cold_start"]["invalid_root_count"] == 1
    assert payload["cold_start"]["rejected_root_count"] == 1


def test_primary_terminal_publishes_split_aggregates_and_guardrail():
    batch = _batch(("valid",) * 4, ((),) * 4)

    def aggregate(mean, cvar, maximum):
        return SimpleNamespace(mean=mean, cvar=cvar, max=maximum)

    cold_split = SimpleNamespace(
        generated=aggregate(0.5, 0.6, 0.7),
        perturbed=aggregate(0.6, 0.7, 0.8),
        composite=aggregate(0.55, 0.65, 0.75),
    )
    selected_split = SimpleNamespace(
        generated=aggregate(0.2, 0.3, 0.4),
        perturbed=aggregate(0.61, 0.72, 0.82),
        composite=aggregate(0.405, 0.51, 0.61),
    )
    guardrail = SimpleNamespace(
        root_count=2,
        candidate_mean=0.61,
        incumbent_mean=0.6,
        mean_excess=-0.02,
        standard_error=0.01,
        confidence_lower=-0.036,
        confidence_z=1.6448536269514722,
        practical_relative_margin=0.05,
        passed=True,
        failures=(),
    )
    comparison = SimpleNamespace(
        mean_improvement=0.3,
        confidence_lower=0.2,
        confidence_upper=0.4,
        win_rate=1.0,
        passed=True,
        failures=(
            "candidate_gate:secret-terminal-root-1:physical_gate",
            "candidate_gate:secret-terminal-root-2:physical_gate",
        ),
    )
    terminal = SimpleNamespace(
        cohort_id="terminal",
        cohort_sha256="c" * 64,
        cold_start=batch,
        selected=batch,
        cold_start_score=SimpleNamespace(errors=(0.5, 0.5), eligible=True),
        selected_score=SimpleNamespace(errors=(0.2, 0.2), eligible=True),
        comparison=comparison,
        supports_selected_over_cold_start=True,
        cold_start_objective_aggregates=cold_split,
        selected_objective_aggregates=selected_split,
        perturbed_guardrail=guardrail,
    )
    result = SimpleNamespace(
        terminal=terminal,
        cold_start=_candidate("cold"),
        selected=_candidate("selected"),
        paired_perturbations=True,
        paired_objective_policy=SimpleNamespace(
            mode=SimpleNamespace(
                value="primary_with_perturbed_guardrail"
            )
        ),
    )

    payload = _terminal_payload(result)

    assert payload["case_count"] == 4
    assert payload["member_case_count"] == 4
    assert payload["root_count"] == 2
    assert payload["objective_sample_count"] == 2
    assert payload["active_objective"] == "generated"
    assert (
        payload["distribution_aggregates"]["selected"]["generated"]["mean"]
        == 0.2
    )
    assert (
        payload["distribution_aggregates"]["selected"]["perturbed"]["mean"]
        == 0.61
    )
    assert payload["perturbed_guardrail"]["passed"] is True
    assert payload["perturbed_guardrail"]["root_count"] == 2
    assert payload["comparison"]["failures"] == ["candidate_gate_failure"]
    assert "secret-terminal-root" not in str(payload)


def test_valid_terminal_claim_metrics_are_finite_comparison_values():
    terminal = SimpleNamespace(
        cold_start_score=SimpleNamespace(errors=(2.0, 4.0), eligible=True),
        selected_score=SimpleNamespace(errors=(1.0, 2.0), eligible=True),
    )

    assert _terminal_claim_metrics(terminal) == (1.5, 3.0, 0.5)


def test_round_payload_publishes_aggregate_search_audit_without_bundle_uri():
    selected = _candidate("selected")
    leaderboard_entry = SimpleNamespace(
        rank=1,
        candidate=selected,
        agent_id="agent-a",
        eligible=True,
        mean=0.2,
        cvar=0.3,
        maximum=0.4,
        generated_mean=0.15,
        perturbed_mean=0.25,
        guardrail_passed=True,
        roles=("screen_generated_mean",),
    )
    completion = SimpleNamespace(
        agent_id="agent-a",
        distinct_screened_candidates=24,
        full_stage_candidates=2,
        released_roots=160,
        final_candidate_sha256=selected.bundle_sha256,
        final_completed_root_count=64,
        sufficient=True,
        failures=(),
    )
    feedback = SimpleNamespace(
        evidence_case_count=128,
        confidence_group_size=64,
        paired_mean_improvement=0.5,
        paired_confidence_lower=0.4,
        paired_win_rate=0.9,
        perturbed_guardrail_passed=True,
        perturbed_guardrail_confidence_lower=0.01,
    )
    row = SimpleNamespace(
        round_index=1,
        development_case_ids=("private-case",),
        candidate_refs=(selected,),
        candidate_portfolio_roles={"selected": ("screen_generated_mean",)},
        agent_failures={},
        finalist_evaluations=(),
        challenger=None,
        selected=selected,
        promotion=None,
        promotion_challenger_objective_aggregates=None,
        promotion_incumbent_objective_aggregates=None,
        promotion_perturbed_guardrail=None,
        leaderboard=(leaderboard_entry,),
        search_completion={"agent-a": completion},
        handoff=SimpleNamespace(
            handoff_sha256="c" * 64,
            previous_handoff_sha256=None,
            promoted=False,
            decision_reason="incumbent retained",
            feedback=feedback,
        ),
    )

    payload = _round_payload(row)

    assert payload["leaderboard"][0]["rank"] == 1
    assert payload["leaderboard"][0]["loss"] == {
        "mean": 0.2,
        "cvar": 0.3,
        "max": 0.4,
        "generated_mean": 0.15,
        "perturbed_mean": 0.25,
    }
    assert payload["search_completion"]["agent-a"] == {
        "agent_id": "agent-a",
        "distinct_screened_candidates": 24,
        "full_stage_candidates": 2,
        "released_roots": 160,
        "final_candidate_sha256": "a" * 64,
        "final_completed_root_count": 64,
        "sufficient": True,
        "failures": [],
    }
    serialized = str(payload)
    assert "bundle_uri" not in serialized
    assert "private-case" not in serialized


@pytest.mark.parametrize("explicit_store", (False, True))
def test_reference_iteration_wires_optional_cohort_store_root(
    tmp_path,
    monkeypatch,
    explicit_store,
):
    candidate = tmp_path / "candidate-bundle"
    candidate.mkdir()
    (candidate / "submission.py").write_text(
        "def predict_macros(case, query, budget=None): return None\n",
        encoding="utf-8",
    )
    write_candidate_bundle_manifest(
        candidate,
        entrypoint="submission.py",
        files=("submission.py",),
    )
    launcher_path = tmp_path / "launcher"
    codex_path = tmp_path / "codex"
    auth = tmp_path / "auth.json"
    auth.write_text("{}\n", encoding="utf-8")
    for executable in (launcher_path, codex_path):
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o700)

    store_root = tmp_path / "node-local-cohorts"
    config = iteration.ReferenceIterationConfig(
        name="store-wiring",
        run_root=tmp_path / "run",
        cold_start_candidate=candidate,
        agent_launcher=launcher_path,
        codex_binary=codex_path,
        auth_file=auth,
        prompt="Improve the macro solver.",
        model="test-model",
        reasoning_effort="high",
        agent_timeout_seconds=10.0,
        agent_hard_timeout_seconds=10.0,
        round_count=1,
        development_case_count=2,
        selection_case_count=2,
        evidence_stages=(EvidenceStage(2, confidence_z=0.0),),
        terminal_case_count=2,
        development_case_credits=2,
        seed_base=10,
        seed_stride=20,
        paired_perturbations=False,
        cohort_store_root=store_root if explicit_store else None,
        guidance_protocol=guidance_config(),
    )
    captured = {}

    class FakeAttestor:
        def __init__(self, root):
            self.root = root

        def attest(self, *args, **kwargs):
            return object()

    class FakeRunner:
        sampling_attempts = ()

        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self):
            return SimpleNamespace(sampling_attempts=())

    cold_ref = CandidateRef(
        "cold",
        "a" * 64,
        0,
        "controller",
        bundle_uri=str(candidate),
    )
    monkeypatch.setattr(iteration, "ReferenceFilesystemBundleAttestor", FakeAttestor)
    monkeypatch.setattr(
        iteration,
        "candidate_ref_from_attestation",
        lambda attestation: cold_ref,
    )
    monkeypatch.setattr(
        iteration,
        "_reference_candidate_submission",
        lambda ref: CandidateSubmission(ref, object(), program_factory=object),
    )
    monkeypatch.setattr(
        iteration,
        "SubprocessReferenceAgentLauncher",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        iteration,
        "ReferenceFilesystemMacroAgentProgram",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(iteration, "MacroClosedLoopRunner", FakeRunner)
    monkeypatch.setattr(iteration, "_write_sampling_ledger", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        iteration,
        "_publish_result",
        lambda **kwargs: {"published": True},
    )

    guidance_backend = object()
    result = iteration.run_reference_agent_iteration(
        config=config,
        plugin=SimpleNamespace(plugin_id="toy"),
        provenance={},
        guidance_backend=guidance_backend,
        legacy_macro_reference=True,
    )

    assert result == {"published": True}
    expected_store = store_root.resolve() if explicit_store else None
    assert config.cohort_store_root == expected_store
    assert captured["cohort_store_root"] == expected_store
    assert captured["config"].guidance_protocol is config.guidance_protocol
    assert captured["guidance_backend"] is guidance_backend


def test_guidance_payload_contains_only_safe_aggregate_bindings():
    candidate = _candidate("guided")
    budget = SimpleNamespace(
        identity_payload=lambda: {
            "maximum_wall_seconds": 10.0,
            "compute_capacity": "one-gpu",
            "maximum_concurrency": 1,
        }
    )
    request = SimpleNamespace(
        source_round_index=0,
        target_round_index=1,
        target_solver_sha256="b" * 64,
        base_generator_sha256="c" * 64,
        reward_matcher_definition_sha256="d" * 64,
        guidance_config_sha256="e" * 64,
        training_budget=budget,
        reward_feedback_uri="private://secret-reward",
        controller_training_seed=123456,
    )
    snapshot = SimpleNamespace(
        snapshot_sha256="f" * 64,
        backend_id="test-guide",
        training_request=request,
        guided_artifact_uri="private://secret-model",
    )
    base_metrics = SimpleNamespace(
        cohort_id="round-1-base",
        kind=SimpleNamespace(value="average_base"),
        generator_or_guidance_sha256="c" * 64,
        values={"primary_error_mean": 0.2},
    )
    guided_metrics = SimpleNamespace(
        cohort_id="round-1-guided",
        kind=SimpleNamespace(value="guided_adversarial"),
        generator_or_guidance_sha256="f" * 64,
        values={"primary_error_mean": 0.5},
    )
    value = SimpleNamespace(
        snapshot=snapshot,
        cohort_id="round-1-guided",
        cohort_sha256="1" * 64,
        campaign_guided_generation_wall_seconds=0.25,
        candidates=(
            SimpleNamespace(
                candidate=candidate,
                guided_score=SimpleNamespace(eligible=True),
                guided_aggregate=SimpleNamespace(
                    mean=0.5, cvar=0.6, max=0.7
                ),
                separated_metrics=SimpleNamespace(
                    average_base=base_metrics,
                    guided_adversarial=guided_metrics,
                ),
            ),
        ),
    )

    payload = iteration._guidance_evaluation_payload(value)
    serialized = str(payload)

    assert payload["snapshot"]["target_solver_sha256"] == "b" * 64
    assert payload["cohort_sha256"] == "1" * 64
    assert payload["candidates"][0]["guided_aggregate"]["mean"] == 0.5
    assert "secret-reward" not in serialized
    assert "secret-model" not in serialized
    assert "controller_training_seed" not in serialized
    assert "case_id" not in serialized
    assert "answer" not in serialized
    assert "locator" not in serialized


def test_publish_result_preserves_multifile_candidate_bundle_round_trip(
    tmp_path: Path,
    monkeypatch,
):
    source = tmp_path / "candidate-bundle"
    source.mkdir()
    (source / "submission.py").write_text(
        "from .helper import VALUE\n"
        "def predict_macros(case, query, budget=None):\n"
        "    return VALUE\n",
        encoding="utf-8",
    )
    (source / "helper.py").write_text("VALUE = 7\n", encoding="utf-8")
    write_candidate_bundle_manifest(
        source,
        entrypoint="submission.py",
        files=("submission.py", "helper.py"),
    )
    archive_sha256 = python_bundle_sha256(source)
    selected = CandidateRef(
        "selected",
        archive_sha256,
        1,
        "agent",
        parent_sha256="b" * 64,
        bundle_uri=str(source),
    )

    launcher_path = tmp_path / "launcher"
    codex_path = tmp_path / "codex"
    auth = tmp_path / "auth.json"
    auth.write_text("{}\n", encoding="utf-8")
    for executable in (launcher_path, codex_path):
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o700)
    config = iteration.ReferenceIterationConfig(
        name="multifile-publish",
        run_root=tmp_path / "run",
        cold_start_candidate=source,
        agent_launcher=launcher_path,
        codex_binary=codex_path,
        auth_file=auth,
        prompt="Find the best solution.",
        model="test-model",
        reasoning_effort="high",
        agent_timeout_seconds=10.0,
        agent_hard_timeout_seconds=310.0,
        round_count=1,
        development_case_count=2,
        selection_case_count=2,
        evidence_stages=(EvidenceStage(2, confidence_z=0.0),),
        terminal_case_count=2,
        development_case_credits=2,
        development_release_root_limit=1,
        development_screen_root_stages=(1,),
        development_screen_root_credit_limit=1,
        seed_base=10,
        seed_stride=20,
        paired_perturbations=False,
    )
    config.run_root.mkdir()
    result = SimpleNamespace(
        execution_assurance=ExecutionAssurance.REFERENCE_IN_PROCESS,
        formal_claims_allowed=False,
        selected=selected,
        cold_start=selected,
        terminal=SimpleNamespace(
            supports_selected_over_cold_start=False
        ),
        rounds=(),
        plugin_id="generic-pde",
        public_result_kind="pde_solver",
        public_result_task="pde_solution",
        published_metric_names=(),
        macro_contract_id="solution_error",
        paired_perturbations=False,
        paired_objective_policy=PairedObjectivePolicy(),
        sampling_attempts=(),
        generator_id="fixed-generator",
        generator_sha256="c" * 64,
    )
    monkeypatch.setattr(
        iteration,
        "_terminal_claim_metrics",
        lambda terminal: (None, None, None),
    )
    monkeypatch.setattr(
        iteration,
        "_terminal_payload",
        lambda result: {"evaluation_status": "valid"},
    )

    payload = iteration._publish_result(
        config=config,
        result=result,
        provenance={},
        wall_time_seconds=1.0,
    )

    published_root = config.run_root / "result" / "model"
    published = inspect_candidate_bundle(published_root)
    assert payload["kind"] == "pde_solver"
    assert payload["task"] == "pde_solution"
    assert payload["artifact"]["entrypoint"] == "submission.py"
    assert payload["artifact"]["files"] == ["helper.py", "submission.py"]
    assert payload["artifact"]["bytes"] == published.total_bytes
    assert payload["artifact"]["tree_sha256"] == published.tree_sha256
    assert payload["artifact"]["archive_sha256"] == archive_sha256
    assert payload["artifact"]["single_file"] is False
    assert payload["controller_policy"]["development_credit_scope"] == (
        "per_request_capacity_no_lifetime_quota"
    )
    assert Path(payload["model"]) == published.entrypoint_path
    loaded = load_python_macro_candidate(
        payload["model"],
        expected_sha256=archive_sha256,
    )
    assert loaded.predict_macros(None, None) == 7


def test_process_top_level_keeps_central_guidance_and_worker_config_disables_it(
    tmp_path: Path,
    monkeypatch,
):
    candidate = tmp_path / "candidate.py"
    launcher_path = tmp_path / "launcher"
    codex_path = tmp_path / "codex"
    auth = tmp_path / "auth.json"
    candidate.write_text(
        "def predict_macros(case, query, budget=None): return None\n",
        encoding="utf-8",
    )
    auth.write_text("{}\n", encoding="utf-8")
    for executable in (launcher_path, codex_path):
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o700)
    config = iteration.ReferenceIterationConfig(
        name="process-guidance-wiring",
        run_root=tmp_path / "run",
        cold_start_candidate=candidate,
        agent_launcher=launcher_path,
        codex_binary=codex_path,
        auth_file=auth,
        prompt="Improve the solver.",
        model="test-model",
        reasoning_effort="high",
        agent_timeout_seconds=10.0,
        agent_hard_timeout_seconds=10.0,
        round_count=1,
        development_case_count=2,
        selection_case_count=2,
        evidence_stages=(EvidenceStage(2, confidence_z=0.0),),
        terminal_case_count=2,
        development_case_credits=2,
        seed_base=10,
        seed_stride=20,
        agent_count=1,
        paired_perturbations=False,
        guidance_protocol=guidance_config(),
    )
    captured = {}

    class FakeAttestor:
        def __init__(self, root):
            self.root = root

        def attest(self, *args, **kwargs):
            return object()

    class FakeRunner:
        sampling_attempts = ()

        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self):
            return SimpleNamespace(sampling_attempts=())

    cold_ref = CandidateRef(
        "cold",
        python_bundle_sha256(candidate),
        0,
        "controller",
        bundle_uri=str(candidate),
    )
    monkeypatch.setattr(iteration, "ReferenceFilesystemBundleAttestor", FakeAttestor)
    monkeypatch.setattr(
        iteration, "candidate_ref_from_attestation", lambda value: cold_ref
    )
    monkeypatch.setattr(
        iteration,
        "_reference_candidate_submission",
        lambda ref: CandidateSubmission(ref, object(), program_factory=object),
    )
    monkeypatch.setattr(iteration, "MacroClosedLoopRunner", FakeRunner)
    monkeypatch.setattr(
        iteration, "_write_sampling_ledger", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        iteration, "_publish_result", lambda **kwargs: {"published": True}
    )
    execution = iteration.ReferenceProcessWorkerConfig(
        command=(sys.executable,),
        gpu_devices=("GPU-agent",),
        controller_gpu_device="GPU-controller",
        node_local_root=tmp_path / "workers",
        cpus_per_worker=1,
        worker_timeout_seconds=30.0,
    )
    backend = object()
    plugin = SimpleNamespace(
        plugin_id="toy",
        cases=SimpleNamespace(
            generator_id="toy-generator",
            artifact_sha256="f" * 64,
        ),
    )

    assert iteration.run_reference_agent_iteration(
        config=config,
        plugin=plugin,
        provenance={},
        process_workers=execution,
        guidance_backend=backend,
        legacy_macro_reference=True,
    ) == {"published": True}
    assert captured["config"].guidance_protocol is config.guidance_protocol
    assert captured["guidance_backend"] is backend
    developer = captured["development_executor"]
    assert developer.iteration_config.guidance_protocol is config.guidance_protocol
    worker_config = replace(
        developer.iteration_config,
        run_root=tmp_path / "worker",
        cohort_store_root=tmp_path / "worker-cohorts",
        guidance_protocol=None,
    )
    assert iteration._reference_loop_config(
        worker_config,
        captured["config"].agents,
    ).guidance_protocol is None
