from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pde_agent_verifier_solver.sdk import inspect_candidate_bundle


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_unified_plasma_agent_iteration.py"
SPEC = importlib.util.spec_from_file_location("unified_ns_launch", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
UNIFIED = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(UNIFIED)

SMOKE = ROOT / "configs" / "ns_unified_parallel_worker_smoke.json"
FORMAL = ROOT / "configs" / "ns_unified_formal_template.json"
COLD = ROOT / "configs" / "candidates" / "ns_hold_initial_bundle"


def _raw(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _localize_agent_and_storage(payload: dict[str, object], tmp_path: Path) -> None:
    storage = tmp_path / "storage"
    storage.mkdir()
    agent = payload["agent"]
    assert isinstance(agent, dict)
    launcher = tmp_path / "launcher.sh"
    launcher.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    launcher.chmod(0o700)
    auth = tmp_path / "auth.json"
    auth.write_text("{}\n", encoding="utf-8")
    agent.update(
        {
            "cold_start_candidate": str(COLD),
            "agent_launcher": str(launcher),
            "codex_binary": "/bin/true",
            "auth_file": str(auth),
        }
    )
    payload["storage_root"] = str(storage)
    payload["run_root"] = str(storage / "run")


def test_formal_launcher_accepts_shared_minimize_metric_preset() -> None:
    specification = {
        "preset": "minimize_metric",
        "settings": {
            "primary": {"distribution": "base", "statistic": "mean"},
            "require_improvement": True,
        },
    }

    assert UNIFIED._promotion_policy(specification) == specification
    with pytest.raises(ValueError, match="unknown promotion preset"):
        UNIFIED._promotion_policy({"preset": "not-a-preset", "settings": {}})


def test_formal_default_promotion_uses_only_the_reported_guided_guardrail() -> None:
    normalized = UNIFIED._promotion_policy({
        "preset": "base_improvement_with_robustness_guardrails",
        "settings": {},
    })

    assert normalized["settings"]["guardrail_distributions"] == ["guided"]


@pytest.mark.parametrize(
    ("distribution", "error"),
    (
        ("guided_perturbed", "formal promotion reports do not provide"),
        ("original", "formal promotion reports do not provide"),
        ("replay", "unknown guardrail distribution"),
    ),
)
def test_formal_promotion_rejects_unreported_guardrail_distributions(
    distribution: str,
    error: str,
) -> None:
    specification = {
        "preset": "base_improvement_with_robustness_guardrails",
        "settings": {"guardrail_distributions": ["guided", distribution]},
    }

    with pytest.raises(ValueError, match=error):
        UNIFIED._promotion_policy(specification)


@pytest.mark.parametrize(
    ("settings", "error"),
    (
        (
            {"primary": {"distribution": "original", "statistic": "mean"}},
            "formal promotion reports do not provide",
        ),
        (
            {"tie_breakers": [{"distribution": "replay", "statistic": "cvar"}]},
            "unknown promotion distribution",
        ),
        (
            {
                "guardrails": [
                    {"distribution": "guided_perturbed", "statistic": "maximum"}
                ]
            },
            "formal promotion reports do not provide",
        ),
    ),
)
def test_formal_minimize_metric_rejects_unreported_distributions(
    settings: dict[str, object],
    error: str,
) -> None:
    specification = {"preset": "minimize_metric", "settings": settings}

    with pytest.raises(ValueError, match=error):
        UNIFIED._promotion_policy(specification)


def test_smoke_is_explicitly_synthetic_and_uses_directory_candidate_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    payload = _raw(SMOKE)
    assert payload["physics"] == "ns"
    assert payload["execution"] == {
        "mode": "process_per_agent_with_central_gpu",
        "worker_gpu_count": 2,
        "controller_gpu_count": 1,
        "cpus_per_worker": 8,
        "timeout_seconds": 1800,
    }
    ns = payload["ns"]
    assert isinstance(ns, dict)
    assert ns == {
        "generator_mode": "synthetic_cpu_chain_test_only",
        "runtime": {"max_points_per_request": 4096},
        "guidance": {"enabled": False},
    }
    assert inspect_candidate_bundle(COLD).manifest_path.is_file()
    assert str(payload["agent"]["cold_start_candidate"]).endswith(
        "configs/candidates/ns_hold_initial_bundle"
    )

    _localize_agent_and_storage(payload, tmp_path)
    monkeypatch.setattr(UNIFIED, "_require_free_bytes", lambda *_args: 10**12)
    audit = UNIFIED._preflight(payload)
    qualification = audit["perturbation_qualification"]
    assert audit["input_count"] == 0
    storage = audit["storage"]
    assert storage["official_selection_terminal_mode"] == (
        "root_pair_streamed_memory_only"
    )
    assert storage["cohort_disk_persistence"] is False
    assert "cohort_store_root" not in storage
    assert "transient_cohort_mode" not in storage
    assert qualification["synthetic_cpu_chain_test_only"] is True
    assert qualification["formal_trained_generator"] is False
    assert qualification["online_audit_or_hashing"] is False
    assert qualification["guidance_enabled"] is False
    assert qualification["cold_start_candidate_bundle"]["entrypoint"] == (
        "submission.py"
    )

    payload["seed_base"] = 2_608_300
    config = UNIFIED._iteration_config(payload, run_root=tmp_path / "iteration")
    assert config.cohort_store_root is None
    assert config.promotion_program.guardrail_distributions == (
        "guided",
    )
    for phrase in (
        "strongest PDE solution method",
        "velocity_x and velocity_y",
        "synthetic_cpu_chain_test_only",
        "not a trained-Generator experiment",
        "prepare_solution(case, solve_budget=None)",
        "relative to the entrypoint's",
        "__file__",
        "solver preparation and point-query evaluation",
        "30.0 seconds",
        "0.05 seconds per requested point",
        "defers its solve until evaluate",
        "No frozen Guidance backend is enabled",
    ):
        assert phrase in config.prompt
    assert "sample_reference_guidance" not in config.prompt


def test_guidance_helpers_are_prompted_only_when_guidance_is_enabled():
    disabled = _raw(SMOKE)
    assert "sample_reference_guidance" not in UNIFIED._ns_prompt(disabled)
    enabled = deepcopy(disabled)
    enabled["ns"]["guidance"]["enabled"] = True
    prompt = UNIFIED._ns_prompt(enabled)
    for phrase in (
        "sample_reference_guidance",
        "query_reference_guided_completed_solution",
        "any number of times for fresh online Guidance samples",
        "not promotion tickets or a fixed released collection",
        "outside candidate preparation and evaluation clocks",
    ):
        assert phrase in prompt


def test_formal_template_cannot_load_with_unresolved_artifact_placeholders(
    monkeypatch: pytest.MonkeyPatch,
):
    for name in ("PLASMA_RUN_ROOT", "PLASMA_STORAGE_ROOT", "UNIFIED_ROOT"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="unresolved environment variables"):
        UNIFIED._load_config(FORMAL)


def test_formal_template_requires_every_transferred_artifact_and_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    payload = _raw(FORMAL)
    promotion = payload["controller_policy"]["promotion"]
    assert promotion["settings"]["guardrail_distributions"] == ["guided"]
    ns = payload["ns"]
    assert isinstance(ns, dict)
    assert ns["generator_mode"] == "trained_structured_coefficient_flow"
    assert ns["guidance"]["enabled"] is True
    enabled, protocol, backend, audit = UNIFIED._ns_guidance_settings(
        ns, mode="trained_structured_coefficient_flow"
    )
    assert enabled is True
    assert protocol.reward_feedback_case_count == 1024
    assert protocol.official_round_case_count == 64
    assert protocol.official_terminal_case_count == 128
    assert protocol.training_budget.maximum_wall_seconds == 7200.0
    assert backend.minimum_cases == 32
    assert protocol.guidance_config_sha256 == backend.sha256
    assert audit["backend_id"] == "ns-structured-reward-gradient-v1"
    assert audit["protocol"]["unlimited_fresh_online_inference"] is True
    assert audit["protocol"]["official_sample_settings"] == protocol.official_sample_settings
    mismatched = deepcopy(ns)
    mismatched["guidance"]["protocol"]["official_sample_settings"]["integration_steps"] = 255
    with pytest.raises(ValueError, match="do not match runtime/backend"):
        UNIFIED._ns_guidance_settings(mismatched, mode="trained_structured_coefficient_flow")
    assert set(ns["artifacts"]) == set(UNIFIED.NS_FORMAL_ARTIFACT_FIELDS)
    assert set(ns["artifact_sha256"]) == set(UNIFIED.NS_FORMAL_ARTIFACT_FIELDS)
    for name in UNIFIED.NS_FORMAL_ARTIFACT_FIELDS:
        assert ns["artifacts"][name] == "${NS_" + name.upper() + "}"
        assert ns["artifact_sha256"][name] == (
            "${NS_" + name.upper() + "_SHA256}"
        )

    _localize_agent_and_storage(payload, tmp_path)
    payload["seed_base"] = 2_608_301
    iteration = UNIFIED._iteration_config(payload, run_root=tmp_path / "formal-run")
    assert iteration.cohort_store_root is None
    assert iteration.guidance_protocol is not None
    assert iteration.guidance_protocol.guidance_config_sha256 == backend.sha256
    assert "sample_reference_guidance" in iteration.prompt
    assert "query_reference_guided_completed_solution" in iteration.prompt
    assert "strongest PDE solution method" in iteration.prompt
    assert iteration.candidate_query_seconds_per_point_limit == 0.05
    assert (
        payload["controller_policy"]["candidate_query_seconds_per_point_limit"]
        == 0.05
    )
    for name in UNIFIED.NS_FORMAL_ARTIFACT_FIELDS:
        ns["artifacts"][name] = str(tmp_path / f"absent-{name}")
        ns["artifact_sha256"][name] = "0" * 64
    ns["runtime"]["perturbation_amplitude"] = "0.02"
    monkeypatch.setattr(UNIFIED, "_require_free_bytes", lambda *_args: 10**12)
    with pytest.raises(FileNotFoundError, match="is not a regular input"):
        UNIFIED._preflight(payload)


def test_formal_builder_passes_one_concrete_guide_to_plugin_and_controller_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    payload = _raw(FORMAL)
    ns = payload["ns"]
    assert isinstance(ns, dict)
    for name in UNIFIED.NS_FORMAL_ARTIFACT_FIELDS:
        ns["artifacts"][name] = str(tmp_path / name)
        ns["artifact_sha256"][name] = "0" * 64
    ns["runtime"]["perturbation_amplitude"] = "0.02"
    monkeypatch.setattr(UNIFIED, "_git_commit", lambda _root: "a" * 40)

    import pde_agent_verifier_solver.plugins.ns as ns_plugin

    calls: dict[str, object] = {}
    guide = SimpleNamespace(
        official_sample_settings=dict(ns["guidance"]["protocol"]["official_sample_settings"])
    )
    plugin = object()

    class Components:
        flow_sampler = SimpleNamespace(artifact_metadata={"pca_rank": 32})

        def build_guidance_backend(self, *, artifact_root, config):
            calls["artifact_root"] = artifact_root
            calls["config"] = config
            return guide

        def bind(self, *, guidance_backend):
            calls["bound_guide"] = guidance_backend
            return plugin

    def factory(**kwargs):
        calls["factory"] = kwargs
        return Components()

    monkeypatch.setattr(ns_plugin, "build_structured_ns_backend", factory)
    built_plugin, provenance, built_guide = UNIFIED._build_ns(payload, tmp_path)
    assert built_plugin is plugin
    assert built_guide is guide
    assert calls["bound_guide"] is guide
    assert calls["artifact_root"] == (
        tmp_path / "controller_private" / "ns_guidance"
    )
    assert calls["config"].sha256 == (
        ns["guidance"]["protocol"]["guidance_config_sha256"]
    )
    assert provenance["artifact_startup"]["guidance_enabled"] is True
    assert provenance["artifact_startup"]["guidance"]["backend_id"] == (
        "ns-structured-reward-gradient-v1"
    )
    assert calls["factory"]["flow_checkpoint"] == Path(
        ns["artifacts"]["flow_checkpoint"]
    )


def test_all_equation_prompts_require_portable_declared_assets():
    payloads_and_prompts = (
        (
            _raw(ROOT / "configs" / "vpfp_unified_live_strong_collision.json"),
            UNIFIED._vpfp_prompt,
        ),
        (
            _raw(ROOT / "configs" / "gkw_unified_competitive_full.json"),
            UNIFIED._gkw_prompt,
        ),
        (_raw(SMOKE), UNIFIED._ns_prompt),
    )
    for payload, prompt_builder in payloads_and_prompts:
        prompt = prompt_builder(payload)
        assert "entrypoint's __file__" in prompt
        assert "Declare every asset in candidate_bundle.json" in prompt
        assert "Do not depend on an absolute workspace" in prompt


def test_formal_report_binding_requires_selected_full_contract(tmp_path: Path):
    def write(name: str, payload: object) -> Path:
        path = tmp_path / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    pca_checkpoint = tmp_path / "pca.pt"
    flow_checkpoint = tmp_path / "flow.pt"
    pca_checkpoint.write_bytes(b"pca")
    flow_checkpoint.write_bytes(b"flow")
    flow_sha = hashlib.sha256(b"flow").hexdigest()
    inputs = {
        "pca_checkpoint": pca_checkpoint,
        "flow_checkpoint": flow_checkpoint,
        "pca_selection": write(
            "pca_selection.json",
            {"schema": "ns_pca_selection_v1", "selected_rank": 4},
        ),
        "pca_audit": write(
            "pca_audit.json",
            {
                "schema": "ns2d_structured_coefficient_pca_audit_v1",
                "ranks": [{"rank": 4}],
            },
        ),
        "flow_selection": write(
            "flow_selection.json",
            {
                "schema": "ns_flow_selection_v1",
                "selected_step": 10,
                "minimum_contract_pass_fraction": 1.0,
                "candidates": [
                    {
                        "step": 10,
                        "eligible": True,
                        "contract_pass_fraction": 1.0,
                    }
                ],
            },
        ),
        "flow_train_report": write(
            "train.json",
            {"schema": "ns2d_structured_coefficient_flow_train_report_v1"},
        ),
        "flow_sample_audit": write(
            "sample.json",
            {
                "schema": "ns2d_structured_coefficient_flow_sample_audit_v1",
                "flow_checkpoint_sha256": flow_sha,
                "flow_step": 10,
                "pca_rank": 4,
                "flow_state": "raw",
            },
        ),
        "answer_first_report": write(
            "answer.json",
            {
                "schema": "ns2d_structured_coefficient_flow_materialization_audit_v1",
                "flow_step": 10,
                "pca_rank": 4,
                "flow_state": "raw",
                "generated_cases": 128,
                "continuous_contract_passed": 128,
            },
        ),
    }
    expected = {name: "0" * 64 for name in inputs}
    sample_audit_sha = hashlib.sha256(
        inputs["flow_sample_audit"].read_bytes()
    ).hexdigest()
    answer_payload = json.loads(inputs["answer_first_report"].read_text())
    answer_payload["sample_audit_sha256"] = sample_audit_sha
    inputs["answer_first_report"].write_text(json.dumps(answer_payload))
    expected["flow_sample_audit"] = sample_audit_sha
    expected["flow_checkpoint"] = flow_sha
    audit = UNIFIED._validate_ns_formal_reports(
        inputs, expected_sha256=expected, flow_state="raw"
    )
    assert audit["formal_report_validation"]["continuous_contract_passed"] == 128

    bound = json.loads(inputs["answer_first_report"].read_text())
    bound["sample_audit_sha256"] = "f" * 64
    inputs["answer_first_report"].write_text(json.dumps(bound))
    with pytest.raises(ValueError, match="not mutually bound"):
        UNIFIED._validate_ns_formal_reports(inputs, expected_sha256=expected, flow_state="raw")
    bound["sample_audit_sha256"] = sample_audit_sha
    inputs["answer_first_report"].write_text(json.dumps(bound))
    flow_payload = json.loads(inputs["flow_sample_audit"].read_text())
    flow_payload["flow_state"] = "ema"
    inputs["flow_sample_audit"].write_text(json.dumps(flow_payload))
    with pytest.raises(ValueError, match="not mutually bound"):
        UNIFIED._validate_ns_formal_reports(inputs, expected_sha256=expected, flow_state="raw")
    flow_payload["flow_state"] = "raw"
    inputs["flow_sample_audit"].write_text(json.dumps(flow_payload))

    bad = json.loads(inputs["answer_first_report"].read_text(encoding="utf-8"))
    bad["continuous_contract_passed"] = 127
    inputs["answer_first_report"].write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="not mutually bound"):
        UNIFIED._validate_ns_formal_reports(
            inputs, expected_sha256=expected, flow_state="raw"
        )


def test_slurm_wrappers_keep_formal_and_synthetic_tiers_separate():
    smoke = (ROOT / "scripts" / "slurm_ns_unified_parallel_worker_smoke.sh").read_text()
    formal = (ROOT / "scripts" / "slurm_ns_unified_formal_template.sh").read_text()
    common = (ROOT / "scripts" / "slurm_ns_unified_live_agent_iteration.sh").read_text()
    for script in (smoke, formal):
        assert "#SBATCH --gres=gpu:l40s:3" in script
        assert "PDE_AGENT_WORKER_GPU_DEVICES" in script
        assert "PDE_CONTROLLER_GPU_DEVICE" in script
    assert "configs/ns_unified_parallel_worker_smoke.json" in smoke
    assert "ns_synth_test_only" in smoke
    assert "REQUIRED_ARTIFACT_ENV" in formal
    assert "No collaborator path or digest is defaulted" in formal
    assert "/scratch/" not in formal and "/home/" not in formal
    assert "NS_PCA_CHECKPOINT:-" not in formal
    assert "legacy/ns/src" in common
    assert "sealed-ns-cohorts-" not in common
    assert "unset PLASMA_COHORT_STORAGE_ROOT" in common
    assert "export PLASMA_COHORT_STORAGE_ROOT" not in common
    assert "--config \"${CONFIG}\"" in common
