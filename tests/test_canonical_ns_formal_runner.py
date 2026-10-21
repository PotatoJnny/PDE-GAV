from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from pde_agent_verifier_solver.engine.memory import CAMPAIGN_MEMORY_HEADER
from pde_agent_verifier_solver.engine.python_candidate import PythonCandidateCodec
from pde_agent_verifier_solver.engine.runtime import RunEvent
from pde_agent_verifier_solver.engine.unified_runtime import _InteractiveSampler
from pde_agent_verifier_solver.sdk import (
    CandidateBundleError,
    CaseVisibility,
    GenerationKind,
    GenerationPurpose,
    GenerationRequest,
    GeneratorMode,
    SampleCasesRequest,
    SampleSource,
    TaskKind,
    inspect_candidate_bundle,
)


# These small CPU fixtures freeze real continuous fields without GPU or Agent calls.
from tests.test_ns_mixed_evaluation_preparation import cpu_field_factory, prepared_suite

_ROOT = Path(__file__).resolve().parents[1]
_RUNNER_PATH = _ROOT / "scripts/run_canonical_ns_campaign.py"
_CONFIG_PATHS = (
    _ROOT / "configs/ns_canonical_structured_5round.json",
    _ROOT / "configs/ns_canonical_piv_5round.json",
)


@pytest.fixture(scope="module")
def runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_canonical_ns_formal_runner_test",
        _RUNNER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("formal NS runner could not be imported")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("config_path", _CONFIG_PATHS)
def test_formal_ns_configs_freeze_campaign_shape_and_budgets(
    runner: ModuleType,
    config_path: Path,
) -> None:
    config, resolved = runner._load_config(config_path)

    assert resolved == config_path.resolve()
    assert config["schema"] == "canonical_ns_campaign_v1"
    assert config["round_count"] == 5
    assert len(config["agents"]) == 2
    assert config["execution"] == {
        "agent_count": 2,
        "agent_hard_wall_seconds": 14_400,
        "transient_resume_until_deadline": True,
        "controller_rpc_timeout_seconds": 7_200,
        "candidate_case_workers": 28,
        "candidate_device": "cpu_or_cuda",
        "candidate_case_hard_wall_seconds": 900,
        "development_candidate_batch_hard_wall_seconds": 6_900,
        "model": "gpt-5.6-sol",
        "reasoning_effort": "max",
        "nomination_policy": "single_primary_v1",
        "solver_batch_size": 8,
        "gpu_slots_per_device": 2,
        "oracle_max_workers": 8,
        "development_oracle_max_workers": 2,
        "oracle_concurrency_per_device": 2,
        "development_solver_placement": "agent_gpu",
        "gpu_execution": {
            "protocol": "shared_gpu_v1",
            "candidate_devices": ["3"],
            "agent_sandbox_backend": "bwrap-nvidia-v1",
        },
    }
    assert config["development"]["strategy"] == "agent_adaptive"
    assert config["development"]["case_replay"] is False

    starter_only = config["development"]["starter_only_sources"]
    if config["profile"] == "experimental_piv":
        assert starter_only == ["original"]
        assert config["development"]["starter"]["original"] == 10
        assert config["candidate"]["max_bundle_bytes"] == 64 * 1024 * 1024
    else:
        assert config["profile"] == "structured_numerical"
        assert starter_only == []
        per_agent_limit = config["development"][
            "max_cases_per_agent_per_round"
        ]
        assert per_agent_limit is None
        assert runner._development_policy(config["development"]).max_cases_per_agent_per_round is None
        assert config["official"]["base"] == 1_024
        assert (
            config["generator"]["guidance"]["protocol"][
                "reward_feedback_case_count"
            ]
            == 1_024
        )


@pytest.mark.parametrize(
    ("field_path", "value"),
    (
        (("round_count",), 3),
        (("development", "max_cases_per_agent_per_round"), 511),
        (("development", "max_cases_per_agent_per_round"), None),
        (("official", "base"), 1023),
        (("execution", "agent_hard_wall_seconds"), 30000),
        (("execution", "controller_rpc_timeout_seconds"), 7199),
        (("execution", "development_candidate_batch_hard_wall_seconds"), 6899),
        (("execution", "candidate_case_workers"), 3),
        (("execution", "candidate_case_hard_wall_seconds"), 600),
        (("execution", "model"), "configured-backbone"),
        (("execution", "reasoning_effort"), "low"),
        (("generator", "perturbations_enabled"), False),
    ),
)
def test_ns_config_accepts_explicit_experiment_parameters(runner, field_path, value):
    config, _ = runner._load_config(_CONFIG_PATHS[0])
    target = config
    for key in field_path[:-1]:
        target = target[key]
    target[field_path[-1]] = value
    runner._validate_config(config, inspect_initial_candidate=False)


@pytest.mark.parametrize(
    ("field_path", "value", "message"),
    (
        (("execution", "agent_count"), 1, "match"),
        (("execution", "agent_hard_wall_seconds"), 0, "positive and finite"),
        (("execution", "agent_active_budget_seconds"), 60, "complete Agent hard-wall"),
        (("execution", "candidate_case_workers"), 0, "integer"),
        (("execution", "controller_rpc_timeout_seconds"), 6900, "below RPC"),
        (("execution", "model"), "", "nonempty"),
        (("development", "case_replay"), True, "replay"),
        (("development", "max_cases_per_agent_per_round"), 0, "integer"),
        (("development", "max_cases_per_agent_per_round"), -1, "integer"),
        (("development", "max_cases_per_agent_per_round"), "unlimited", "integer"),
        (("generator", "guidance_enabled"), False, "guidance is disabled"),
        (("candidate", "initial_bundle"), None, "before feedback exists"),
    ),
)
def test_ns_config_preserves_protocol_and_resource_safety(runner, field_path, value, message):
    config, _ = runner._load_config(_CONFIG_PATHS[0])
    target = config
    for key in field_path[:-1]:
        target = target[key]
    target[field_path[-1]] = value
    with pytest.raises(ValueError, match=message):
        runner._validate_config(config, inspect_initial_candidate=False)


def test_ns_config_accepts_total_time_only_and_round_total_time_override(runner):
    config, _ = runner._load_config(_CONFIG_PATHS[0])
    config["execution"].pop("agent_active_budget_seconds", None)
    config["execution"]["agent_hard_wall_seconds"] = 14400
    config["round_overrides"] = {"2": {"agent_hard_wall_seconds": 3600}}
    runner._validate_config(config, inspect_initial_candidate=False)
    assert runner._round_overrides(config)[2].agent_budget_seconds == 3600
    config["round_overrides"]["2"]["agent_hard_wall_seconds"] = 14401
    with pytest.raises(ValueError, match="total-time limit"):
        runner._validate_config(config, inspect_initial_candidate=False)


@pytest.mark.parametrize("agents", (1, 3))
def test_gpu_allocation_tracks_configured_agent_count(runner, agents):
    workers = tuple(str(index) for index in range(agents))
    assert runner._gpu_allocation(
        ",".join(workers), str(agents), agent_count=agents,
    ) == (workers, str(agents))
    with pytest.raises(ValueError, match="worker GPUs"):
        runner._gpu_allocation(
            ",".join(workers), str(agents), agent_count=agents + 1,
        )
    with pytest.raises(ValueError, match="distinct"):
        runner._gpu_allocation(
            ",".join(workers), workers[0], agent_count=agents,
        )


def test_piv_candidate_bundle_limit_drift_fails_closed(
    runner: ModuleType,
) -> None:
    source, _ = runner._load_config(_CONFIG_PATHS[1])
    config = deepcopy(source)
    config["candidate"]["max_bundle_bytes"] -= 1

    with pytest.raises(ValueError, match="bundle limit must be 67108864 bytes"):
        runner._validate_config(config)


def test_piv_codec_rejects_oversized_agent_nomination(
    runner: ModuleType,
    tmp_path: Path,
) -> None:
    source, _ = runner._load_config(_CONFIG_PATHS[1])
    candidate_settings = source["candidate"]
    limits = runner._candidate_bundle_limits(
        source["profile"],
        candidate_settings,
    )
    codec = PythonCandidateCodec(task_kind=TaskKind.SOLVER, limits=limits)
    workspace = tmp_path / "agent-turn"
    workspace.mkdir()
    oversized = workspace / "oversized.py"
    with oversized.open("wb") as handle:
        handle.write(b"def prepare_solution(case, solve_budget=None):\n    pass\n")
        handle.seek(limits.max_file_bytes)
        handle.write(b"\n")
    context = SimpleNamespace(
        round_plan=SimpleNamespace(round_index=1),
        agent=SimpleNamespace(agent_id="agent-a"),
        incumbent=None,
    )

    with pytest.raises(CandidateBundleError, match="exceeds max_file_bytes"):
        codec.decode_nomination(
            {"bundle_path": "oversized.py"},
            context,
            workspace,
        )

    assert not (workspace / ".controller-candidates").exists()


@pytest.mark.parametrize("config_path", _CONFIG_PATHS)
def test_formal_ns_config_paths_resolve_to_installed_inputs(
    runner: ModuleType,
    config_path: Path,
) -> None:
    config, resolved = runner._load_config(config_path)
    dataset = runner._config_path(resolved, config["dataset"]["manifest"])
    candidate = runner._config_path(
        resolved,
        config["candidate"]["initial_bundle"],
    )
    worker = runner._config_path(resolved, config["worker"]["script"])
    codex_config = runner._config_path(
        resolved,
        config["worker"]["codex_config"],
    )
    paths = [dataset, worker, codex_config]
    if config["profile"] == "structured_numerical":
        paths.extend(
            runner._config_path(resolved, raw)
            for raw in config["generator"]["artifacts"].values()
        )
    else:
        paths.append(
            runner._config_path(resolved, config["generator"]["checkpoint"])
        )

    assert all(path.is_absolute() for path in (*paths, candidate))
    missing = [str(path) for path in paths if not path.is_file()]
    if not candidate.is_dir():
        missing.append(str(candidate))
    if missing:
        pytest.skip(
            "formal external NS inputs are not installed in this checkout: "
            + ", ".join(missing)
        )
    inspect_candidate_bundle(candidate)
    runner._validate_config(config)


def test_piv_original_cases_are_unavailable_after_the_starter_set(
    runner: ModuleType,
) -> None:
    config, _ = runner._load_config(_CONFIG_PATHS[1])
    development = config["development"]
    allowed = tuple(SampleSource(value) for value in development["allowed_sources"])
    starter_only = tuple(
        SampleSource(value) for value in development["starter_only_sources"]
    )

    class _OriginalPoolMustNotBeCalled:
        called = False

        def sample_cases(self, request):
            del request
            self.called = True
            raise AssertionError("starter-only request crossed into the Case pool")

    inner = _OriginalPoolMustNotBeCalled()
    sampler = _InteractiveSampler(
        inner,
        object(),
        allowed,
        starter_only_sources=starter_only,
        allow_guidance=True,
        allow_perturbations=True,
    )
    sampler.seal_starter()

    with pytest.raises(PermissionError, match="Controller-provided starter set"):
        sampler.sample_cases(SampleCasesRequest(SampleSource.ORIGINAL, 1))
    assert inner.called is False


def test_run_writer_persists_each_memory_round_immediately_without_duplication(
    runner: ModuleType,
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "formal-run"
    run_root.mkdir()
    writer = runner._RunWriter(run_root)
    memory_writes = []
    write_memory = writer._write_text_locked

    def record_memory_write(relative: str, value: str) -> None:
        memory_writes.append((relative, value))
        write_memory(relative, value)

    writer._write_text_locked = record_memory_write
    round_one = "## Round 1\n\n### Controller outcome\n\nSelected candidate A."
    round_two = "## Round 2\n\n### Controller outcome\n\nSelected candidate B."

    writer.record_event(
        RunEvent(
            sequence=1,
            event_type="campaign_memory_updated",
            message="Round 1 memory recorded",
            round_index=1,
            payload={"round_memory": {"merged_markdown": round_one}},
        )
    )
    memory_path = run_root / "CAMPAIGN_MEMORY.md"
    expected_one = f"{CAMPAIGN_MEMORY_HEADER}\n\n{round_one}\n"
    assert memory_path.read_text(encoding="utf-8") == expected_one

    writer.record_event(
        RunEvent(
            sequence=2,
            event_type="campaign_memory_updated",
            message="Round 2 memory recorded",
            round_index=2,
            payload={"round_memory": {"merged_markdown": round_two}},
        )
    )
    expected_two = (
        f"{CAMPAIGN_MEMORY_HEADER}\n\n{round_one}\n\n{round_two}\n"
    )
    assert memory_path.read_text(encoding="utf-8") == expected_two

    writer.record_event(
        RunEvent(
            sequence=3,
            event_type="round_finished",
            message="Round 2 finished",
            round_index=2,
            payload={"artifact_id": "candidate-b"},
        )
    )

    assert memory_path.read_text(encoding="utf-8") == expected_two
    assert memory_writes == [
        ("CAMPAIGN_MEMORY.md", expected_one),
        ("CAMPAIGN_MEMORY.md", expected_two),
    ]
    assert len((run_root / "events.jsonl").read_text().splitlines()) == 3


def test_git_prefix_supports_private_and_standard_repository_layouts(
    runner: ModuleType,
    tmp_path: Path,
) -> None:
    assert runner._git_prefix(tmp_path) == ("git", "-C", str(tmp_path))

    private_git = tmp_path / ".git-real"
    private_git.mkdir()
    assert runner._git_prefix(tmp_path) == (
        "git",
        f"--git-dir={private_git}",
        f"--work-tree={tmp_path}",
    )


@pytest.mark.parametrize("guidance,perturbations", [(False, False), (True, False), (False, True), (True, True)])
def test_ns_hidden_evaluation_capabilities_do_not_enable_development_tools(runner, guidance, perturbations):
    config, _ = runner._load_config(_CONFIG_PATHS[0])
    config["generator"]["guidance_enabled"] = guidance
    config["generator"]["perturbations_enabled"] = perturbations
    config["development"]["allowed_sources"] = ["base"]
    config["development"]["starter"]["base"] = 32
    config["development"]["starter"]["guided"] = 0
    runner._validate_config(config, inspect_initial_candidate=False)

    assert runner._generator_switches(config) == (guidance, perturbations)
    assert runner._effective_generator_switches(config) == (True, True)


def test_ns_disabled_development_perturbation_still_rejects_paired_starters(runner):
    config, _ = runner._load_config(_CONFIG_PATHS[0])
    config["generator"]["perturbations_enabled"] = False
    config["development"]["starter"]["base"] = 1
    config["development"]["starter"]["base_perturbed"] = 1
    with pytest.raises(ValueError, match="perturbed development Cases"):
        runner._validate_config(config, inspect_initial_candidate=False)


def test_ns_hidden_capabilities_include_round_overrides_and_terminal_sources(runner):
    config, _ = runner._load_config(_CONFIG_PATHS[0])
    config["generator"]["guidance_enabled"] = False
    config["generator"]["perturbations_enabled"] = False
    base_suite = {"original": 0, "base": 32, "guided": 0, "base_perturbed": 0, "guided_perturbed": 0}
    config["official"] = dict(base_suite)
    config["terminal"]["cohorts"] = [{
        "cohort_id": "base_model", "guidance_source": "base", "suite": dict(base_suite),
    }]
    config["terminal"]["ranking"]["cohort_id"] = "base_model"
    assert runner._effective_generator_switches(config) == (False, False)

    config["round_overrides"] = {"3": {"official": {**base_suite, "guided": 4, "guided_perturbed": 4}}}
    assert runner._effective_generator_switches(config) == (True, True)
    del config["round_overrides"]
    config["terminal"]["cohorts"][0]["guidance_source"] = "final_guided"
    config["terminal"]["cohorts"][0]["suite"] = {**base_suite, "base": 0, "guided": 32}
    config["terminal"]["ranking"]["distribution"] = "guided"
    assert runner._effective_generator_switches(config) == (True, False)


@pytest.mark.parametrize("guidance,perturbations", [(False, False), (True, False), (False, True), (True, True)])
def test_runner_explicit_shared_suite_resolves_relative_path_and_keeps_development_permissions(
    runner, prepared_suite, tmp_path, guidance, perturbations,
):
    from pde_agent_verifier_solver.plugins.ns.mixed_evaluation import SharedNSMixedEvaluationGenerator
    controller = prepared_suite["controller"]
    original = controller.generator
    original.backend.development_guidance_enabled = guidance
    original.backend.development_perturbations_enabled = perturbations
    config_path = tmp_path / "configuration" / "run.json"
    config_path.parent.mkdir()
    config = {
        "profile": "structured_numerical",
        "generator": {"guidance_enabled": guidance, "perturbations_enabled": perturbations},
        "shared_evaluation_suite_manifest": os.path.relpath(prepared_suite["path"], config_path.parent),
    }
    config_path.write_text(json.dumps(config))
    composition = SimpleNamespace(generator=original, campaign=controller.campaign)
    wrapper = runner._bind_hidden_evaluation_audit(config, config_path, composition, tmp_path / "run")
    assert isinstance(wrapper, SharedNSMixedEvaluationGenerator)
    assert composition.generator is original
    assert runner._generator_switches(config) == (guidance, perturbations)
    assert (original.backend.development_guidance_enabled,
            original.backend.development_perturbations_enabled) == (guidance, perturbations)
    assert wrapper.capabilities.guidance and wrapper.capabilities.perturbations
    audit_path = tmp_path / "run/controller_private/shared_evaluation_case_audit.jsonl"
    audit = json.loads(audit_path.read_text().splitlines()[0])
    assert Path(audit["suite_manifest"]) == prepared_suite["path"]
    assert audit["master_seed"] == controller.campaign.master_seed

    state = wrapper.load(controller.campaign.generator.checkpoint, {})
    first = prepared_suite["suite"]["official_rounds"][0]["sampling_requests"][0]
    records = wrapper.sample(state, GenerationRequest(
        round_index=first["round_index"], count=first["count"], mode=GeneratorMode.ONLINE,
        kind=GenerationKind(first["kind"]), visibility=CaseVisibility.HIDDEN,
        seed=first["seed"], purpose=GenerationPurpose.PROMOTION,
    ))
    assert [row.case.case_id for row in records] == first["case_ids"]
    for kind, permitted in ((GenerationKind.GUIDED, guidance), (GenerationKind.PERTURBED, perturbations)):
        if not permitted:
            with pytest.raises(PermissionError, match="NS development"):
                wrapper.sample(state, GenerationRequest(
                    round_index=1, count=1, mode=GeneratorMode.ONLINE, kind=kind,
                    visibility=CaseVisibility.DEVELOPMENT, seed=22,
                    purpose=GenerationPurpose.DEVELOPMENT,
                    anchors=(records[0].case,) if kind is GenerationKind.PERTURBED else (),
                ))
    wrapper.release(state, tuple(row.case.case_id for row in records))


@pytest.mark.parametrize("change,message", [("seed", "master_seed"), ("task", "Task evaluation settings")])
def test_runner_shared_suite_rejects_changed_seed_or_task_settings(runner, prepared_suite, tmp_path, change, message):
    controller = prepared_suite["controller"]
    campaign = controller.campaign
    if change == "seed":
        campaign = replace(campaign, master_seed=campaign.master_seed + 1)
    else:
        campaign = replace(campaign, task_settings={**campaign.task_settings, "official_point_count": 5})
    with pytest.raises(ValueError, match=message):
        runner._bind_hidden_evaluation_audit(
            {"profile": "structured_numerical", "shared_evaluation_suite_manifest": str(prepared_suite["path"])},
            tmp_path / "config.json", SimpleNamespace(generator=controller.generator, campaign=campaign),
            tmp_path / "run",
        )
    assert not (tmp_path / "run/controller_private/shared_evaluation_case_audit.jsonl").exists()


def test_runner_without_shared_suite_retains_existing_official_audit_path(runner, tmp_path, monkeypatch):
    from pde_agent_verifier_solver.plugins.ns import official_audit
    captured = []
    sentinel = object()
    def factory(*args):
        captured.append(args)
        return sentinel
    monkeypatch.setattr(official_audit, "OfficialAuditGenerator", factory)
    composition = SimpleNamespace(generator=object(), campaign=SimpleNamespace(
        master_seed=101, task_settings={"official_point_count": 4},
        evaluator=SimpleNamespace(default_query="fixed-query")))
    config_path = tmp_path / "configs/run.json"
    assert runner._bind_hidden_evaluation_audit({"profile": "structured_numerical"},
        config_path, composition, tmp_path / "none") is None
    result = runner._bind_hidden_evaluation_audit(
        {"profile": "structured_numerical", "official_audit_manifest": "../control/official.json"},
        config_path, composition, tmp_path / "old",
    )
    assert result is sentinel
    assert captured == [(composition.generator, tmp_path / "control/official.json",
        tmp_path / "old/controller_private/official_case_audit.jsonl", 101,
        {"official_point_count": 4, "default_query": "fixed-query"})]


def test_runner_requires_one_explicit_ns_hidden_evaluation_choice(runner):
    config, _ = runner._load_config(_CONFIG_PATHS[0])
    config["shared_evaluation_suite_manifest"] = "shared/suite.json"
    config["official_audit_manifest"] = "old/official.json"
    with pytest.raises(ValueError, match="mutually exclusive"):
        runner._validate_config(config, inspect_initial_candidate=False)
    del config["official_audit_manifest"]
    runner._validate_config(config, inspect_initial_candidate=False)
    config["profile"] = "experimental_piv"
    with pytest.raises(ValueError, match="structured numerical profile"):
        runner._validate_config(config, inspect_initial_candidate=False)


@pytest.mark.parametrize("policy", ["all", "latest", "single_primary_v1"])
def test_ns_config_accepts_explicit_nomination_policy(runner, policy):
    config, _ = runner._load_config(_CONFIG_PATHS[0])
    config["execution"]["nomination_policy"] = policy
    runner._validate_config(config, inspect_initial_candidate=False)


@pytest.mark.parametrize("policy", [None, True, 1, [], {}, "best", "Latest"])
def test_ns_config_rejects_invalid_nomination_policy_before_launch(runner, policy):
    config, _ = runner._load_config(_CONFIG_PATHS[0])
    config["execution"]["nomination_policy"] = policy
    with pytest.raises(ValueError, match="execution.nomination_policy"):
        runner._validate_config(config, inspect_initial_candidate=False)


@pytest.mark.parametrize("attempts", [None, True, 0, -1, 4, 1.5, "3"])
def test_ns_config_rejects_invalid_resume_bound_before_launch(runner, attempts):
    config, _ = runner._load_config(_CONFIG_PATHS[0])
    config["execution"]["transient_resume_max_attempts"] = attempts
    with pytest.raises(ValueError, match="transient_resume_max_attempts"):
        runner._validate_config(config, inspect_initial_candidate=False)


@pytest.mark.parametrize("attempts", [1, 2, 3])
def test_ns_config_accepts_bounded_transient_resume(runner, attempts):
    config, _ = runner._load_config(_CONFIG_PATHS[0])
    config["execution"]["transient_resume_max_attempts"] = attempts
    runner._validate_config(config, inspect_initial_candidate=False)


@pytest.mark.parametrize("value", [True, False])
def test_ns_config_accepts_deadline_recovery(runner, value):
    config, _ = runner._load_config(_CONFIG_PATHS[0])
    config["execution"]["transient_resume_until_deadline"] = value
    runner._validate_config(config, inspect_initial_candidate=False)


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_ns_config_rejects_invalid_deadline_recovery(runner, value):
    config, _ = runner._load_config(_CONFIG_PATHS[0])
    config["execution"]["transient_resume_until_deadline"] = value
    with pytest.raises(ValueError, match="transient_resume_until_deadline"):
        runner._validate_config(config, inspect_initial_candidate=False)
