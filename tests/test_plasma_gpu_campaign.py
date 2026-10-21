"""CPU-only composition and durable boundary checks; no science or model calls."""
from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import plasma_gpu_campaign as campaign


def config(equation="strong", arm="sol"):
    model, effort = ("gpt-5.6-sol", "max") if arm == "sol" else ("gpt-6-astra", "ultra")
    return {
        "schema": campaign.SCHEMA, "equation": equation, "campaign_id": equation + "-" + arm,
        "round_count": 5, "master_seed": 101,
        "agents": [{"agent_id": x, "name": x, "prompt": "Use the public macro-only task."} for x in ("a", "b")],
        "execution": {"model": model, "reasoning_effort": effort,
            "agent_hard_wall_seconds": 14400, "candidate_case_hard_wall_seconds": 900,
            "candidate_case_workers": 16 if equation == "strong" else 8,
            "controller_rpc_timeout_seconds": 7200, "development_candidate_batch_hard_wall_seconds": 6900,
            "native_session_persistence": True, "transient_resume_max_attempts": 3 if equation == "weak" else 1,
            "development_workers_per_gpu": 4, "workers_per_gpu": 4,
            "solver_batch_size": 1, "gpu_slots_per_device": 1,
            "oracle_concurrency_per_device": 1,
            "oracle_max_workers": {"strong": 12, "weak": 8, "artemis": 3}[equation],
            "gpu_execution": {"protocol": "shared_gpu_v1", "candidate_devices": ["3"],
                "agent_sandbox_backend": "bwrap-nvidia-v1", "agent_runtime_sha256": "a" * 64}},
        "development": {"max_cases_per_agent_per_round": 32, "starter_base": 4},
        "official": deepcopy(campaign.OFFICIAL), "terminal": deepcopy(campaign.TERMINAL),
        "promotion": {"preset": "minimize_metric", "settings": {
            "primary": {"distribution": "overall", "statistic": "mean"}}},
        "protocol": {"official_nomination_policy": {"mode": "single_primary_v1", "start_round": 1},
                     "parallel_evaluation": "service_global_oracle_v1"},
        "task": {"require_prepared_solution": True, "candidate_budget_mode": "ns_remaining_case_v1",
            "development_point_count": 128, "official_point_count": 256, "terminal_point_count": 256,
            **({"scoring_coordinate_mode": "continuous_x_v1", "causal_horizon": True} if equation == "artemis" else {})},
        "worker": {"script": str(campaign.ROOT / "scripts/run_canonical_codex_agent.py"),
                   "codex_config": str(campaign.ROOT / "scripts/canonical_codex.config.toml")},
        "candidate": {"initial_bundle": "/unloaded/candidate", "initial_bundle_sha256": "b" * 64,
                      "initial_name": "S0", "initial_summary": "Frozen S0"},
        "backend": {"factory_settings": {}},
        "migration": {"oracle_execution_options": {"workers_per_gpu": 4,
            "oracle_variant": "ordered_throughput_v1", "source_request_block": 262144, "macro_point_block": 2048}},
    }


@pytest.mark.parametrize("equation", ["strong", "weak", "artemis"])
@pytest.mark.parametrize("arm", ["sol", "astra"])
def test_same_approved_science_and_budget_across_six_arms(equation, arm):
    value = config(equation, arm)
    assert campaign.validate_config(value) is value
    initial = SimpleNamespace()
    recipe = campaign.recipe(value, initial)
    assert recipe.initial_candidate is initial
    assert recipe.default_round.agent_budget_seconds == 14400
    assert recipe.default_round.development_sampling.max_cases_per_agent_per_round == 32
    assert recipe.default_round.development_sampling.starter.base == 4
    assert recipe.default_round.official_evaluation.round.case_count == 256
    assert recipe.evaluator.allow_point_truth is False
    assert recipe.task_settings["candidate_budget_mode"] == "ns_remaining_case_v1"
    assert "candidate_compute_budget" not in recipe.task_settings
    assert len(recipe.terminal_evaluation.cohorts) == 2


@pytest.mark.parametrize("equation", ["strong", "weak", "artemis"])
@pytest.mark.parametrize("arm", ["sol", "astra"])
def test_same_deadline_recovery_does_not_change_six_arm_scientific_budgets(equation, arm):
    value = config(equation, arm)
    value["execution"]["transient_resume_until_deadline"] = True
    assert campaign.validate_config(value) is value
    compiled = campaign.recipe(value, SimpleNamespace())
    assert compiled.default_round.agent_budget_seconds == 14400
    assert compiled.default_round.development_sampling.max_cases_per_agent_per_round == 32
    assert value["execution"]["candidate_case_hard_wall_seconds"] == 900


def test_artemis_starter_shape_normalizes_without_changing_science():
    value = config("artemis")
    value["development"] = {"max_cases_per_agent_per_round": 32,
        "starter": {"base": 4, "guided": 0}, "allowed_sources": ["base", "guided"], "case_replay": False}
    assert campaign.validate_config(value) is value


@pytest.mark.parametrize("section,key,value", [
    ("execution", "agent_hard_wall_seconds", 7200),
    ("execution", "candidate_case_hard_wall_seconds", 180),
    ("task", "candidate_compute_budget", 180),
    ("execution", "model", "unapproved-fallback"),
    ("development", "max_cases_per_agent_per_round", None),
    ("development", "starter_base", 16),
    ("execution", "transient_resume_max_attempts", 3),
])
def test_protocol_drift_fails_before_assets(section, key, value):
    item = config()
    item[section][key] = value
    with pytest.raises(ValueError):
        campaign.validate_config(item)


def test_static_check_keeps_shared_entrypoints_and_verifies_initial_hash(tmp_path):
    from pde_agent_verifier_solver.sdk import inspect_candidate_bundle, write_candidate_bundle_manifest
    item = config()
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "submission.py").write_text("def prepare_solution(case, solve_budget=None):\n    return None\n")
    write_candidate_bundle_manifest(bundle, entrypoint="submission.py", files=["submission.py"])
    item["candidate"].update(initial_bundle=str(bundle), initial_bundle_sha256=inspect_candidate_bundle(bundle).tree_sha256)
    asset = tmp_path / "asset.bin"
    asset.write_bytes(b"CPU fixture, not physical model validation")
    digest = campaign.file_sha(asset)
    item["backend"] = {"base_checkpoint": str(asset), "base_checkpoint_sha256": digest,
        "portable_codec": str(asset), "portable_codec_sha256": digest,
        "native_spec": {"base_decoder_checkpoint": str(asset), "base_decoder_sha256": digest,
                        "expanded_decoder_checkpoint": str(asset), "expanded_decoder_sha256": digest}}
    checked = campaign.static_check(item)
    assert checked["cuda_qualified"] is False
    assert checked["status"] == "static_checks_passed"
    item["worker"]["script"] = "/old/checkout/run_canonical_codex_agent.py"
    with pytest.raises(ValueError, match="shared current"):
        campaign.static_check(item)


class Service:
    def __init__(self, *args, **kwargs):
        self.options = kwargs
        self.evaluation_device_map = kwargs.get("evaluation_device_map")
        self.quarantined_gpu_devices = ()
        self.max_workers = kwargs.get("max_workers", 1)
        self.audits = []
        self.closed = 0

    def close(self):
        self.closed += 1


@pytest.mark.parametrize("equation", ["strong", "weak", "artemis"])
@pytest.mark.parametrize("arm", ["sol", "astra"])
def test_composition_keeps_new_engine_and_separate_gpu_roles(tmp_path, monkeypatch, equation, arm):
    from pde_agent_verifier_solver.engine import agent_execution, candidate_execution, python_candidate
    from pde_agent_verifier_solver.engine.unified_runtime import CampaignController
    from pde_agent_verifier_solver.plugins.vpfp import evaluation_gpu_reuse, weak_parallel_oracle
    from pde_agent_verifier_solver.plugins.vp_1d1v import evaluation_gpu_reuse as artemis_scope
    item = config(equation, arm)
    item["execution"]["gpu_slots_per_device"] = 2
    seen = {}
    class Codec:
        def __init__(self, **kwargs): pass
        def bind_existing(self, *args, **kwargs):
            return SimpleNamespace(program=SimpleNamespace(tree_sha256=item["candidate"]["initial_bundle_sha256"]))
        def decode_nomination(self, *args): pass
        def encode_incumbent(self, *args): pass
    class Binding:
        report = {"scope": "mock_only"}
        plugin = SimpleNamespace(task=lambda task_id: SimpleNamespace())
        def controller(self, plan, programs, **kwargs):
            seen.update(plan=plan, programs=programs, controller_options=kwargs)
            controller = object.__new__(CampaignController)
            controller.agent_execution = kwargs["agent_execution"]
            controller.candidate_execution = kwargs["candidate_execution"]
            controller._official_reports_in_scope = lambda *args, **kw: ("reports", "evaluations")
            return controller
    def scope(binding, execution, **kwargs):
        seen["scope"] = kwargs
        @contextmanager
        def enter(**scope_options):
            seen["scope_enter"] = scope_options
            try:
                yield
            finally:
                seen["scope_closed"] = True
        return enter
    def development(binding, execution, **kwargs):
        seen["development"] = kwargs
        return SimpleNamespace(shutdown=lambda: seen.update(development_closed=True))
    monkeypatch.setenv("SLURM_JOB_GPUS", "0,1,2,3")
    monkeypatch.setattr(campaign, "_binding", lambda *args: (Binding(), {}))
    monkeypatch.setattr(python_candidate, "PythonCandidateCodec", Codec)
    monkeypatch.setattr(agent_execution, "SubprocessAgentExecutionService", Service)
    monkeypatch.setattr(candidate_execution, "SubprocessCandidateExecutionService", Service)
    monkeypatch.setattr(candidate_execution.PythonTaskWorkerSpec, "from_task_type", lambda *args: "mock-task")
    monkeypatch.setattr(evaluation_gpu_reuse, "VPFPHiddenEvaluationScope", scope)
    monkeypatch.setattr(weak_parallel_oracle, "WeakVPFPHiddenEvaluationScope", scope)
    monkeypatch.setattr(weak_parallel_oracle, "install_development_oracle", development)
    monkeypatch.setattr(artemis_scope, "ArtemisHiddenEvaluationScope", scope)
    composed = campaign.compose(item, tmp_path / "config.json", tmp_path / "run",
        codex_binary="/qualified/codex", auth_file="/private/auth",
        evaluation_device_map={"2": "cuda:0", "0": "cuda:1", "1": "cuda:2"})
    assert composed.candidate_execution.options["gpu_devices"] == ("3",)
    assert composed.candidate_execution.options["gpu_slots_per_device"] == 2
    assert composed.candidate_execution.options["solver_batch_size"] == 1
    assert composed.candidate_execution.options["oracle_max_workers"] == item["execution"]["oracle_max_workers"]
    assert seen["scope"]["devices"] == ("2", "0", "1")
    # Exercise the real bootstrap/official Controller boundary: a successful
    # composition must have exactly one owner for hidden-evaluation GPUs.
    composed.agent_execution.quarantined_gpu_devices = ("1",)
    case = object()
    hidden = {"base": (SimpleNamespace(case=case),)}
    assert composed.controller._official_reports(("initial-candidate",), hidden) == ("reports", "evaluations")
    assert seen["scope_enter"] == {"cases": (case,), "unavailable_gpu_devices": ("1",)}
    assert seen["scope_closed"] is True
    assert composed.agent_execution.evaluation_device_map is None
    for program, token in zip(seen["programs"].values(), ("0", "1")):
        assert program.gpu_devices == (token,)
        assert program.worker_settings["model"] == item["execution"]["model"]
        assert program.worker_settings["reasoning_effort"] == item["execution"]["reasoning_effort"]
        assert program.worker_settings["nomination_policy"] == "single_primary_v1"
        assert program.worker_settings["transient_resume_until_deadline"] is True
        assert "official_nomination_policy" not in program.worker_settings
        assert program.worker_settings["native_session_persistence"] is True
    if equation == "strong":
        assert composed.candidate_execution.options["oracle_inline_array_bytes"] == 8 * 1024 * 1024
        assert seen["scope"]["oracle_variant"] == "ordered_throughput_v1"
    if equation == "weak":
        assert seen["development"]["controller_gpu"] == "2"
        assert seen["development"]["workers_per_gpu"] == 4
    composed.close()
    assert composed.candidate_execution.closed == 1


def _run_fixture(tmp_path, monkeypatch, *, fail_cleanup=False):
    from pde_agent_verifier_solver.engine import checkpoint_store
    from pde_agent_verifier_solver.engine.campaign_continuation import CampaignPaused
    value = config()
    native = tmp_path / "native"
    native.write_bytes(b"mock qualified runtime")
    value["execution"]["gpu_execution"]["agent_runtime_sha256"] = campaign.file_sha(native)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(value))
    seen = []
    class Controller:
        def run(self, *, continuation, pause_control):
            seen.append(continuation)
            if continuation is None:
                pause_control.save({"phase": "after_development", "completed_agents": ["a", "b"]})
                raise CampaignPaused(stage="development_completed")
            assert continuation["completed_agents"] == ["a", "b"]
            return SimpleNamespace(selected=SimpleNamespace(artifact="selected"),
                memory=SimpleNamespace(markdown="same campaign"), candidate_history={}, rounds=[None] * 5)
    class Broken(Service):
        def close(self):
            super().close()
            if fail_cleanup:
                raise RuntimeError("fixture cleanup failure")
    def compose(*args, **kwargs):
        generator = SimpleNamespace(export_runtime_state=lambda directory: None)
        return campaign.Composition(Controller(), SimpleNamespace(plugin=SimpleNamespace(generator=generator)),
            Service(), Broken(), None, {"binding_receipt": {"interface_check": {
                "status": "passed", "private_probe_released": True, "scope": "CPU mock, not CUDA"}}})
    monkeypatch.setattr(campaign, "static_check", lambda item: {"status": "mock"})
    monkeypatch.setattr(campaign, "compose", compose)
    monkeypatch.setattr(campaign, "_publish", lambda *args: {"sha256": "c" * 64})
    monkeypatch.setattr(checkpoint_store, "source_identity", lambda root: "frozen-source")
    kwargs = dict(codex_binary=native, auth_file=tmp_path / "unused-auth",
                  allocation_seconds=60, shutdown_margin_seconds=1)
    return path, tmp_path / "run", kwargs, seen


def test_pause_resume_uses_real_checkpoint_without_rerunning_agents(tmp_path, monkeypatch):
    path, root, kwargs, seen = _run_fixture(tmp_path, monkeypatch)
    paused = campaign.run(path, root, **kwargs)
    assert paused["status"] == "paused" and paused["retry_safe"] is True
    assert paused["stage"] == "development_completed" and paused["cleanup_complete"] is True
    assert Path(paused["checkpoint_path"]).is_file()
    first_provenance = (root / "input_provenance.json").read_bytes()
    accepted = (root / "qualification/accepted.json").read_bytes()
    completed = campaign.run(path, root, resume=True, **kwargs)
    assert completed["status"] == "completed" and completed["retry_safe"] is False
    assert completed["rounds_completed"] == 5
    assert seen == [None, {"phase": "after_development", "completed_agents": ["a", "b"]}]
    assert (root / "input_provenance.json").read_bytes() == first_provenance
    assert (root / "qualification/accepted.json").read_bytes() == accepted
    assert json.loads((root / "resume_provenance.json").read_text())["interface_qualification"]["reused_without_resampling"] is True
    with pytest.raises(ValueError, match="clean paused"):
        campaign.run(path, root, resume=True, **kwargs)


def test_unconfirmed_cleanup_never_becomes_retry_safe(tmp_path, monkeypatch):
    path, root, kwargs, seen = _run_fixture(tmp_path, monkeypatch, fail_cleanup=True)
    with pytest.raises(RuntimeError, match="cleanup"):
        campaign.run(path, root, **kwargs)
    status = json.loads((root / "status.json").read_text())
    assert status["status"] == "failed" and status["retry_safe"] is False
    with pytest.raises(ValueError, match="clean paused"):
        campaign.run(path, root, resume=True, **kwargs)


def test_changed_configuration_cannot_restore_existing_checkpoint(tmp_path, monkeypatch):
    path, root, kwargs, seen = _run_fixture(tmp_path, monkeypatch)
    campaign.run(path, root, **kwargs)
    changed = json.loads(path.read_text())
    changed["agents"][0]["prompt"] += " Changed prompt."
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="identity differs"):
        campaign.run(path, root, resume=True, **kwargs)
    assert seen == [None]


def test_writer_lock_rejection_never_overwrites_the_owners_status(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from pde_agent_verifier_solver.engine import checkpoint_store
    path, root, kwargs, seen = _run_fixture(tmp_path, monkeypatch)
    campaign.run(path, root, **kwargs)
    before = (root / "status.json").read_bytes()
    @contextmanager
    def occupied(run_root):
        raise RuntimeError("another writer is already running this Run")
        yield
    monkeypatch.setattr(checkpoint_store, "run_writer_lock", occupied)
    with pytest.raises(RuntimeError, match="another writer"):
        campaign.run(path, root, resume=True, **kwargs)
    assert (root / "status.json").read_bytes() == before
    assert seen == [None]


@pytest.mark.parametrize("change", ["missing", "failed", "altered_evidence"])
def test_resume_requires_exact_passed_original_interface_receipt(tmp_path, monkeypatch, change):
    path, root, kwargs, seen = _run_fixture(tmp_path, monkeypatch)
    campaign.run(path, root, **kwargs)
    receipt = root / "qualification/accepted.json"
    if change == "missing":
        receipt.rename(root / "qualification/removed-for-test.json")
    else:
        contents = json.loads(receipt.read_text())
        if change == "failed":
            contents["report"]["status"] = "failed"
        else:
            contents["report"]["scope"] = "changed evidence"
        receipt.write_text(json.dumps(contents))
    with pytest.raises(ValueError, match="qualification|identity differs"):
        campaign.run(path, root, resume=True, **kwargs)
    assert seen == [None]


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_plasma_rejects_invalid_deadline_recovery_switch(value):
    item = config("weak", "sol")
    item["execution"]["transient_resume_until_deadline"] = value
    with pytest.raises(ValueError, match="must be a boolean"):
        campaign.validate_config(item)
