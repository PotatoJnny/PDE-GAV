"""Formal composition tests; no model calls, Slurm jobs or physical evidence."""

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pde_agent_verifier_solver.engine import (
    CommandAgentProgram, PythonCandidateCodec, SubprocessAgentExecutionService,
    SubprocessCandidateExecutionService,
)
from pde_agent_verifier_solver.engine.runtime import EventStream, RunEvent
from pde_agent_verifier_solver.engine.memory import CAMPAIGN_MEMORY_HEADER
from pde_agent_verifier_solver.engine.unified_runtime import CampaignController
from pde_agent_verifier_solver.sdk import SampleSource, TaskKind, inspect_candidate_bundle
from tests.test_artemis_canonical import _binding


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/artemis_canonical_5round.json"
SPEC = importlib.util.spec_from_file_location("artemis_formal_runner_fixture", ROOT / "scripts/run_canonical_artemis_campaign.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def _configuration():
    return RUNNER._load_config(CONFIG)


def test_frozen_configuration_and_static_check_do_not_load_assets_or_auth(monkeypatch, capsys):
    from pde_agent_verifier_solver.plugins.vp_1d1v import product
    monkeypatch.setattr(product, "load_artemis_backend", lambda **kwargs: pytest.fail("static check loaded GPU assets"))
    monkeypatch.setattr(RUNNER, "_compose", lambda **kwargs: pytest.fail("static check composed execution"))
    assert RUNNER.main(["--config", str(CONFIG), "--validate-only"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["status"] == "static_configuration_validated"
    assert status["assets_loaded"] is status["model_called"] is False
    config, path = _configuration()
    assert path == CONFIG
    assert config["execution"]["model"] == "gpt-5.6-sol"
    assert config["execution"]["reasoning_effort"] == "max"
    assert config["official"]["base"] == 128
    assert config["official"]["base_perturbed"] == 0
    assert config["execution"]["agent_active_budget_seconds"] == 7200
    assert config["execution"]["agent_hard_wall_seconds"] == 14400
    assert config["terminal"]["suite"]["base"] == config["terminal"]["suite"]["base_perturbed"] == 256
    assert config["generator"]["guidance_enabled"] is True
    assert config["generator"]["feedback_source"] == "canonical_selected_solver_official"
    assert config["generator"]["feedback_case_count"] == config["official"]["base"] == 128
    assert config["generator"]["settings"] == {"guidance_maximum_wall_seconds": 3600.0}
    assert config["development"]["allowed_sources"] == ["base", "guided"]
    inputs = RUNNER._static_inputs(config, path)
    guide = inputs["factory_settings"]["reward_guidance_settings"]
    assert guide["model_kind"] == "ou_hermite_ridge_v1"
    assert guide["condition_count"] == 6
    assert guide["sampling_steps"] == 64
    assert guide["guidance_strength"] == 2.0
    assert guide["activation_positive_fraction"] == 0.75
    assert guide["minimum_uniform_mixture"] == 0.25


@pytest.mark.parametrize("path,value", [
    (("round_count",), 4),
    (("round_count",), True),
    (("execution", "agent_count"), 1),
    (("execution", "agent_active_budget_seconds"), 21599),
    (("execution", "agent_hard_wall_seconds"), 28799),
    (("execution", "controller_rpc_timeout_seconds"), 300),
    (("execution", "model"), "different-model"),
    (("execution", "reasoning_effort"), "high"),
    (("generator", "guidance_enabled"), False),
    (("generator", "perturbations_enabled"), False),
    (("generator", "settings", "guidance_maximum_wall_seconds"), 3601),
    (("generator", "feedback_case_count"), 384),
    (("generator", "feedback_source"), "legacy_independent_cohort"),
    (("development", "max_cases_per_agent_per_round"), 255),
    (("development", "max_cases_per_agent_per_round"), 512),
    (("development", "allowed_sources"), ["base"]),
    (("development", "case_replay"), True),
    (("official", "base_perturbed"), 127),
    (("terminal", "suite", "base"), 255),
    (("promotion", "settings", "guardrail_distributions"), []),
    (("promotion", "settings", "relative_margin"), 0.1),
    (("evaluator", "development_coordinate_mode"), "agent_selected"),
    (("evaluator", "allow_grid_truth"), True),
    (("evaluator", "allow_point_truth"), True),
    (("evaluator", "allow_development_solution_session"), True),
    (("evaluator", "terminal_point_count"), 31),
    (("task", "require_prepared_solution"), False),
    (("task", "causal_horizon"), False),
    (("task", "candidate_compute_budget"), 599),
])
def test_frozen_protocol_drift_is_rejected(path, value):
    config, _ = _configuration()
    node = config
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(ValueError, match="changed"):
        RUNNER._validate_frozen_protocol(config)


def test_duplicate_agents_and_unknown_configuration_keys_are_rejected():
    config, _ = _configuration()
    config["agents"][1]["agent_id"] = config["agents"][0]["agent_id"]
    with pytest.raises(ValueError, match="distinct"):
        RUNNER._validate_frozen_protocol(config)
    config, _ = _configuration()
    config["unexpected"] = True
    with pytest.raises(ValueError, match="unknown"):
        RUNNER._validate_frozen_protocol(config)


def test_backend_bytes_are_verified_before_any_factory_import(tmp_path):
    config, path = _configuration()
    malformed = tmp_path / "backend.json"
    malformed.write_text('{"physics":"artemis","factory_settings":{}}')
    config["backend"]["config"] = str(malformed)
    with pytest.raises(ValueError, match="admitted CUDA backend"):
        RUNNER._static_inputs(config, path)


@pytest.mark.parametrize("workers,controller", [("0", "1"), ("0,1,2", "3"), ("0,0", "1"), ("0,1", "0"), ("0,,1", "2"), ("0,1", ""), ("0,1", "2,3")])
def test_gpu_allocation_rejects_overlap_missing_or_extra_devices(workers, controller):
    with pytest.raises(ValueError, match="GPU"):
        RUNNER._gpu_allocation(workers, controller)


def test_gpu_allocation_preserves_slurm_tokens():
    assert RUNNER._gpu_allocation("GPU-aab1, GPU-cc22", "GPU-dd33") == (("GPU-aab1", "GPU-cc22"), "GPU-dd33")


@pytest.mark.parametrize("reuse_gpus", [False, True])
def test_compose_uses_only_canonical_binding_worker_codec_and_frozen_recipe(tmp_path, monkeypatch, reuse_gpus):
    from pde_agent_verifier_solver.plugins.vp_1d1v import product
    from pde_agent_verifier_solver.plugins.vp_1d1v import ArtemisBackend, build_artemis_plugin
    config, path = _configuration()
    inputs = RUNNER._static_inputs(config, path)
    binding, source = _binding()
    guide = SimpleNamespace(fit_guidance=lambda *args: pytest.fail("compose trained Guidance"),
                            generate=lambda *args: pytest.fail("compose sampled Guidance"))
    backend = ArtemisBackend(source, physical_revision="test-double-not-physical-evidence", guidance_adapter=guide)
    binding = replace(binding, plugin=build_artemis_plugin(backend), checkpoint=backend.checkpoint(),
                      report={"guidance_enabled": True, "perturbations_enabled": True,
                                       "evidence": "synthetic_interface_fixture"})
    called = []
    def factory(**settings):
        called.append(settings)
        return binding
    monkeypatch.setattr(product, "load_artemis_backend", factory)
    controller, execution, provenance = RUNNER._compose(
        config=config, config_path=path, inputs=inputs, run_root=tmp_path,
        codex_binary=Path("/trusted/codex"), auth_file=Path("/trusted/auth-source"),
        worker_gpus=("0", "1"), controller_gpu="2", events=EventStream(),
        reuse_agent_gpus_for_evaluation=reuse_gpus,
    )
    assert called == [inputs["factory_settings"]]
    assert isinstance(controller, CampaignController)
    assert isinstance(execution, SubprocessAgentExecutionService)
    assert controller.agent_execution is execution
    assert execution.hard_wall_seconds == 14400
    assert execution.enforce_active_budget is False
    assert isinstance(controller.candidate_execution, SubprocessCandidateExecutionService)
    assert controller.candidate_execution.max_workers == 28
    assert controller.candidate_execution.case_hard_wall_seconds == 900
    assert controller.candidate_execution.development_batch_hard_wall_seconds == 6900
    assert execution.controller_gpu_devices == ("2",)
    assert len(controller.campaign.rounds) == 5
    assert controller.campaign.generator.checkpoint is binding.checkpoint
    assert controller.campaign.generator.guidance_enabled is True
    assert dict(controller.campaign.generator.settings) == {"guidance_maximum_wall_seconds": 3600.0}
    assert controller._round_needs_guidance(controller.campaign.rounds[0]) is True
    assert controller.campaign.initial_candidate.kind is TaskKind.OBSERVABLE_MODEL
    assert controller.evaluation.require_prepared_solution is True
    assert controller.evaluation.causal_horizon is True
    assert controller.evaluation.solve_budget == 600
    for plan in controller.campaign.rounds:
        assert plan.agent_budget_seconds == 7200
        # Exercise the actual service rule: source request 7200 is now total
        # wall time, not 7200 active plus a separate 14400 wall allowance.
        assert execution._wall_budget(plan.agent_budget_seconds) == 7200
        assert plan.development_sampling.max_cases_per_agent_per_round == 256
        assert plan.development_sampling.starter.base == 16
        assert plan.development_sampling.starter.guided == 0
        assert plan.development_sampling.allowed_sources == (SampleSource.BASE, SampleSource.GUIDED)
        assert plan.round_hidden.base == 128
        assert plan.round_hidden.base_perturbed == 0
    for index, program in enumerate(controller.agents.values()):
        assert isinstance(program, CommandAgentProgram)
        assert program.gpu_devices == (str(index),)
        assert "run_canonical_codex_agent.py" in program.command[1]
        assert "reference" not in program.command[1]
        assert isinstance(program.nomination_decoder.__self__, PythonCandidateCodec)
        assert program.nomination_decoder.__self__.task_kind is TaskKind.OBSERVABLE_MODEL
        assert program.incumbent_encoder.__self__ is program.nomination_decoder.__self__
        assert program.worker_settings["controller_rpc_timeout_seconds"] == 7200
        assert program.worker_settings["candidate_execution"] == {
            "device": "cpu", "cuda_available": False, "solver_batch_size": 1,
            "gpu_slots_per_device": 1, "oracle_concurrency_per_device": 1}
        assert program.environment["PATH"].split(":")[0] == str(Path(RUNNER.sys.executable).parent)
    terminal = controller.campaign.terminal_evaluation
    assert terminal.cohorts[0].suite.base == terminal.cohorts[0].suite.base_perturbed == 256
    assert terminal.cohorts[0].suite.guided == terminal.cohorts[0].suite.guided_perturbed == 0
    assert provenance["guidance_protocol"]["feedback_case_count"] == 128
    assert provenance["guidance_protocol"]["initial_feedback_case_count"] == 128
    assert provenance["guidance_protocol"]["legacy_feedback_case_count"] == 384
    assert provenance["guidance_protocol"]["activation_gate_noop_is_valid"] is True
    assert provenance["assurance"]["candidate_execution_isolated"] is True
    assert provenance["assurance"]["formal_private_scoring_claimed"] is True
    assert provenance["assurance"]["hostile_filesystem_sandbox"] is False
    assert provenance["agent_protocol"] == {
        "agent_count": 2, "time_budget_mode": "hard_wall", "enforce_active_budget": False,
        "requested_budget_seconds": 7200, "effective_hard_wall_seconds": 7200,
        "service_hard_wall_limit_seconds": 14400, "active_budget_seconds": None,
        "includes_controller_tool_wait": True, "model": "gpt-5.6-sol", "reasoning_effort": "max",
    }
    assert provenance["source_execution_config"] == config["execution"]
    assert provenance["source_execution_config"]["agent_active_budget_seconds"] == 7200
    assert provenance["protocol_revision"] == "0e8aeade88bff80da79a8ba008700638f32511b2"
    assert not source.requests and execution.audits == ()


def test_compose_rejects_backend_that_did_not_bind_guidance(tmp_path, monkeypatch):
    from pde_agent_verifier_solver.plugins.vp_1d1v import product
    config, path = _configuration()
    binding, _ = _binding()
    binding = replace(binding, report={"guidance_enabled": False, "perturbations_enabled": True})
    monkeypatch.setattr(product, "load_artemis_backend", lambda **kwargs: binding)
    with pytest.raises(ValueError, match="Base\\+Guidance\\+Perturbation"):
        RUNNER._compose(
            config=config, config_path=path, inputs=RUNNER._static_inputs(config, path),
            run_root=tmp_path, codex_binary=Path("/trusted/codex"), auth_file=Path("/trusted/auth-source"),
            worker_gpus=("0", "1"), controller_gpu="2", events=EventStream(),
        )


def test_guidance_audits_persist_backend_cost_and_snapshot_metadata(tmp_path):
    adapter = SimpleNamespace(
        training_audits=[{"feedback_case_count": 128, "wall_seconds": 0.5, "snapshot_sha256": "g1"}],
        sampling_audits=[{"count": 2, "wall_seconds": 0.2, "snapshot_sha256": "g1"}],
    )
    controller = SimpleNamespace(generator=SimpleNamespace(backend=SimpleNamespace(guidance_adapter=adapter)))
    RUNNER._persist_guidance_audits(controller, RUNNER._RunWriter(tmp_path))
    report = json.loads((tmp_path / "guidance_audits.json").read_text())
    assert report["schema"] == "canonical_artemis_guidance_audits_v1"
    assert report["training"] == adapter.training_audits
    assert report["sampling"] == adapter.sampling_audits


def test_writer_checkpoints_memory_and_deduplicates_round_updates(tmp_path):
    writer = RUNNER._RunWriter(tmp_path)
    for sequence, index, markdown in ((1, 1, "Round one"), (2, 1, "Round one updated"), (3, 2, "Round two")):
        writer.record_event(RunEvent(sequence, "campaign_memory_updated", "memory", round_index=index,
                                    payload={"round_memory": {"merged_markdown": markdown}}))
    assert (tmp_path / "CAMPAIGN_MEMORY.md").read_text() == f"{CAMPAIGN_MEMORY_HEADER}\n\nRound one updated\n\nRound two\n"
    assert len((tmp_path / "events.jsonl").read_text().splitlines()) == 3
    assert json.loads((tmp_path / "status.json").read_text())["round_index"] == 2
    assert (tmp_path / "events.jsonl").stat().st_mode & 0o777 == 0o600


def _mock_main(monkeypatch, tmp_path, *, failure=False):
    config, path = _configuration()
    bundle = ROOT / "configs/candidates/artemis_hold_initial_bundle"
    codec = PythonCandidateCodec(task_kind=TaskKind.OBSERVABLE_MODEL)
    artifact = codec.bind_existing(bundle, storage_root=tmp_path / "frozen", artifact_id="selected",
                                   name="Selected", summary="fixture", usage="prepare/evaluate")
    calls = []
    class Controller:
        campaign = {"round_count": 5}
        candidate_execution = SimpleNamespace(audits=())
        def run(self):
            calls.append("run")
            if failure:
                raise RuntimeError("preserve this exact synthetic failure")
            return SimpleNamespace(campaign_name=config["campaign_id"], memory=SimpleNamespace(markdown="Saved memory"),
                                   candidate_history={"fixture": True}, selected=SimpleNamespace(artifact=artifact),
                                   best_terminal_round=2, rounds=(1, 2, 3, 4, 5))
    monkeypatch.setattr(RUNNER, "_compose", lambda **kwargs: (Controller(), SimpleNamespace(audits=()), {"fixture": True}))
    monkeypatch.setattr(RUNNER, "_source_state", lambda: {"commit": "c" * 40, "status_porcelain": ""})
    auth = tmp_path / "auth-source"
    auth.write_text("credential-must-never-be-read-by-composition")
    original_read = Path.read_text
    def no_auth_read(self, *args, **kwargs):
        if self == auth:
            pytest.fail("composition read credentials")
        return original_read(self, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", no_auth_read)
    output = tmp_path / "run"
    argv = ["--config", str(path), "--run-root", str(output), "--codex-binary", "/bin/true",
            "--auth-file", str(auth), "--worker-gpus", "0,1", "--controller-gpu", "2"]
    return argv, output, calls, artifact


def test_preflight_composes_but_never_runs_agent(monkeypatch, tmp_path):
    argv, output, calls, _ = _mock_main(monkeypatch, tmp_path)
    assert RUNNER.main([*argv, "--preflight-only"]) == 0
    assert calls == []
    assert json.loads((output / "status.json").read_text())["status"] == "preflight_completed"
    assert (output / "source_state.json").is_file()
    assert (output / "compiled_campaign.json").is_file()


def test_main_calls_controller_once_and_publishes_selected_bundle(monkeypatch, tmp_path):
    argv, output, calls, artifact = _mock_main(monkeypatch, tmp_path)
    assert RUNNER.main(argv) == 0
    assert calls == ["run"]
    assert inspect_candidate_bundle(output / "selected_candidate").tree_sha256 == artifact.program.tree_sha256
    assert json.loads((output / "status.json").read_text())["best_terminal_round"] == 2
    assert json.loads((output / "status.json").read_text())["selected_artifact_id"] == "selected"
    for filename in ("result.json", "execution_audits.json", "candidate_execution_audits.json", "guidance_audits.json", "candidate_history.json", "CAMPAIGN_MEMORY.md", "input_provenance.json", "resolved_config.json"):
        assert (output / filename).is_file()


def test_failure_is_not_converted_to_success_and_artifacts_are_kept(monkeypatch, tmp_path):
    argv, output, calls, _ = _mock_main(monkeypatch, tmp_path, failure=True)
    with pytest.raises(RuntimeError, match="exact synthetic failure"):
        RUNNER.main(argv)
    assert calls == ["run"]
    assert json.loads((output / "status.json").read_text())["status"] == "failed"
    assert "exact synthetic failure" in (output / "failure_traceback.txt").read_text()
    assert (output / "execution_audits.json").is_file()
    assert (output / "candidate_execution_audits.json").is_file()
    assert (output / "guidance_audits.json").is_file()
    with pytest.raises(FileExistsError, match="refusing to reuse"):
        RUNNER.main(argv)


def test_dirty_checkout_fails_before_composition_and_retains_reason(monkeypatch, tmp_path):
    argv, output, calls, _ = _mock_main(monkeypatch, tmp_path)
    monkeypatch.setattr(RUNNER, "_source_state", lambda: {"commit": "c" * 40, "status_porcelain": " M user.py"})
    with pytest.raises(ValueError, match="clean, committed"):
        RUNNER.main([*argv, "--preflight-only"])
    assert calls == []
    assert json.loads((output / "source_state.json").read_text())["status_porcelain"] == " M user.py"
    assert json.loads((output / "status.json").read_text())["status"] == "failed"


def test_publication_refuses_existing_destination(monkeypatch, tmp_path):
    _, _, _, artifact = _mock_main(monkeypatch, tmp_path)
    destination = tmp_path / "already-published"
    destination.mkdir()
    with pytest.raises(FileExistsError):
        RUNNER._publish_selected_candidate(artifact, destination)
