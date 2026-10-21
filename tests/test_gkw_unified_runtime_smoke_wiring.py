from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SMOKE_PATH = ROOT / "scripts" / "smoke_gkw_unified_runtime.py"
SLURM_PATH = ROOT / "scripts" / "slurm_gkw_unified_runtime_smoke.sh"
LIVE_SLURM_PATH = ROOT / "scripts" / "slurm_gkw_unified_live_agent_iteration.sh"
PARALLEL_SLURM_PATH = (
    ROOT / "scripts" / "slurm_gkw_unified_parallel_worker_smoke.sh"
)
COMPETITIVE_REHEARSAL_SLURM_PATH = (
    ROOT / "scripts" / "slurm_gkw_unified_competitive_rehearsal.sh"
)
LEGACY_PHYSICAL_PIN = "bda22391d2da32a12f87a2688d67270f51f8c091"
PREVIOUS_QUALIFIED_PHYSICAL_PIN = (
    "5ca50ae3c78d48015f8b9d5ffd1eda5347f52748"
)
CONTINUOUS_PHYSICAL_PIN = "80632f11a39b8577124ffa652556af1521d9b992"
LEGACY_CONFIG_PATHS = tuple(
    ROOT / "configs" / name
    for name in (
        "gkw_unified_fresh_equal_pilot.json",
        "gkw_unified_fresh_primary_guardrail_pilot.json",
        "gkw_unified_lifecycle_memory_smoke.json",
        "gkw_unified_strong_full.json",
        "gkw_unified_strong_rehearsal.json",
    )
)
CONTINUOUS_CONFIG_PATHS = tuple(
    ROOT / "configs" / name
    for name in (
        "gkw_unified_live_agent_iteration.json",
        "gkw_unified_parallel_worker_smoke.json",
        "gkw_unified_competitive_rehearsal.json",
        "gkw_unified_competitive_full.json",
    )
)
RETIRED_RECOVERY_CONFIG = (
    ROOT / "configs" / "gkw_unified_competitive_recovery.json"
)


@pytest.fixture
def smoke_module(monkeypatch):
    agent = ModuleType("agent_verifier")
    agent.__path__ = []
    fusion = ModuleType("agent_verifier.fusion_simulation_first")
    fusion.__path__ = []
    factory = ModuleType(
        "agent_verifier.fusion_simulation_first.gkw_selected_service_factory"
    )
    factory.GKWSelectedServiceArtifacts = object
    factory.build_gkw_selected_online_service = object
    perturbation = ModuleType(
        "agent_verifier.fusion_simulation_first.gkw_smooth_perturbation"
    )
    perturbation.GKWSmoothPerturbationConfig = object
    qualification = ModuleType(
        "agent_verifier.fusion_simulation_first.gkw_perturbation_qualification"
    )
    qualification.gkw_qualification_certificate_from_payload = object
    qualification.gkw_qualification_certificate_payload = object
    modules = {
        "agent_verifier": agent,
        "agent_verifier.fusion_simulation_first": fusion,
        factory.__name__: factory,
        perturbation.__name__: perturbation,
        qualification.__name__: qualification,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location("gkw_smoke_wiring", SMOKE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _required_arguments(module) -> list[str]:
    values = [
        "--run-root",
        "/tmp/run",
        "--storage-root",
        "/tmp",
        "--external-root",
        "/tmp/external",
        "--gyaradax-root",
        "/tmp/gyaradax",
        "--physical-commit",
        "physical",
        "--physical-root",
        "/tmp/physical",
        "--unified-commit",
        "unified",
        "--qualification-certificate",
        "/tmp/qualification.json",
        "--qualification-report",
        "/tmp/qualification_report.json",
        "--generator-manifest",
        "/tmp/generator_manifest.json",
        "--operator-manifest",
        "/tmp/operator_manifest.json",
        "--physical-source-manifest",
        "/tmp/physical_source_manifest.json",
        "--seed-plan",
        "/tmp/seed_plan.json",
    ]
    for name in module._ARTIFACT_FIELDS:
        values.extend(("--" + name.replace("_", "-"), f"/tmp/{name}"))
    return values


def test_smoke_cli_defaults_to_one_development_root_and_accepts_selection(
    smoke_module, monkeypatch
):
    arguments = _required_arguments(smoke_module)
    monkeypatch.setattr(sys, "argv", ["smoke", *arguments])
    development = smoke_module._arguments()
    assert development.cohort == "development"
    assert development.lifecycle_root_count == 1

    monkeypatch.setattr(
        sys,
        "argv",
        ["smoke", *arguments, "--cohort", "selection"],
    )
    assert smoke_module._arguments().cohort == "selection"

    monkeypatch.setattr(
        sys,
        "argv",
        ["smoke", *arguments, "--cohort", "terminal"],
    )
    with pytest.raises(SystemExit):
        smoke_module._arguments()


def test_selection_smoke_uses_controller_private_root_paired_point_queries():
    smoke = SMOKE_PATH.read_text(encoding="utf-8")
    assert "engine.macro_query_plan import (" in smoke
    assert "ControllerMacroQueryPlan," in smoke
    assert "query_plan = ControllerMacroQueryPlan(" in smoke
    assert "controller_query_factory=gkw_macro_point_query" in smoke
    assert "query.kind is not QueryKind.POINTS" in smoke
    assert "public_session_methods=GKW_PUBLIC_SESSION_METHODS" in smoke
    assert "GKW clean/perturbed scoring queries drifted" in smoke
    assert "gkw_native_macro_query" not in smoke
    assert "_validated_panel_audits" not in smoke


def test_continuous_condition_admission_rejects_lookup_and_child_drift(
    smoke_module,
):
    def record(condition):
        scalars = dict(
            zip(smoke_module._PUBLIC_CONDITION_NAMES, condition)
        )
        return SimpleNamespace(
            case=SimpleNamespace(public_data={"scalars": scalars})
        )

    archives = {
        0: SimpleNamespace(rlt_i=0.0, rln_i=1.0, shat=2.0, q=3.0),
        1: SimpleNamespace(rlt_i=2.0, rln_i=3.0, shat=4.0, q=5.0),
    }
    fresh = (1.0, 2.0, 3.0, 4.0)
    passed = smoke_module._continuous_condition_admission(
        (record(fresh),),
        (record(fresh),),
        archives,
        np.ones(4, dtype=np.float64),
    )
    assert passed["passed"] is True
    assert passed["clean_condition_count"] == 1
    assert passed["archive_condition_count"] == 2
    assert passed["standardization"] == "generator_geometry_sidecar"
    assert passed["exact_archive_lookup_hit_count"] == 0
    assert passed["minimum_nearest_standardized_distance"] > 0.0
    assert passed["child_condition_match_count"] == 1

    lookup = smoke_module._continuous_condition_admission(
        (record((0.0, 1.0, 2.0, 3.0)),),
        (record((0.0, 1.0, 2.0, 3.0)),),
        archives,
        np.ones(4, dtype=np.float64),
    )
    assert lookup["passed"] is False
    assert lookup["exact_archive_lookup_hit_count"] == 1

    drift = smoke_module._continuous_condition_admission(
        (record(fresh),),
        (record((1.0, 2.0, 3.0, 4.25)),),
        archives,
        np.ones(4, dtype=np.float64),
    )
    assert drift["passed"] is False
    assert drift["child_condition_match_count"] == 0


def test_smoke_report_and_slurm_wrapper_record_and_forward_cohort():
    smoke = SMOKE_PATH.read_text(encoding="utf-8")
    wrapper = SLURM_PATH.read_text(encoding="utf-8")
    assert '"cohort": args.cohort' in smoke
    assert '"profile": query_profile' in smoke
    assert '"query_plan": {' in smoke
    assert 'SMOKE_COHORT="${SMOKE_COHORT:-development}"' in wrapper
    assert '--cohort "${SMOKE_COHORT}"' in wrapper
    assert 'GKW_PERTURBATION_CERTIFICATE' in wrapper
    assert '--qualification-certificate "${QUALIFICATION_CERTIFICATE}"' in wrapper
    for name in (
        "qualification-report",
        "generator-manifest",
        "operator-manifest",
        "physical-source-manifest",
        "seed-plan",
    ):
        assert f"--{name}" in wrapper


def test_slurm_smoke_is_offline_and_checks_all_pinned_inputs():
    wrapper = SLURM_PATH.read_text(encoding="utf-8")

    for forbidden in (
        "git fetch",
        "git clone",
        "git init",
        "remote add",
        "pip install",
        "https://github.com",
    ):
        assert forbidden not in wrapper
    assert "GKW_PHYSICAL_COMMIT:?Set the final pinned lean physical commit" in wrapper
    assert "GKW_PHYSICAL_ROOT:?Set the clean pinned physical checkout" in wrapper
    assert "neural-gyrokinetics-568411b" in wrapper
    assert "gyaradax-318e3d11" in wrapper
    assert "gkw-perturbation-qualification-python-v1" in wrapper
    assert "568411b33be486262b8c5cfe1dfb3e0fe97f6419" in wrapper
    assert "318e3d11d7326d99e96f2bd5a8998a462261dcd4" in wrapper
    assert wrapper.count("status --porcelain --untracked-files=all") >= 3
    assert "actual = {name: metadata.version(name) for name in expected}" in wrapper
    for distribution, version in {
        "kappamodules": "0.1.112",
        "einops": "0.8.1",
        "omegaconf": "2.3.0",
        "antlr4-python3-runtime": "4.9.3",
        "PyYAML": "6.0.2",
    }.items():
        expected = (
            chr(34) + distribution + chr(34)
            + ": " + chr(34) + version + chr(34)
        )
        assert expected in wrapper


def test_section_timer_records_synchronized_cuda_process_memory_and_io(
    smoke_module,
    monkeypatch,
):
    class FakeCuda:
        def __init__(self):
            self.synchronize_calls = 0
            self.reset_calls = 0

        def synchronize(self):
            self.synchronize_calls += 1

        def reset_peak_memory_stats(self):
            self.reset_calls += 1

        def memory_allocated(self):
            return 11

        def memory_reserved(self):
            return 13

        def max_memory_allocated(self):
            return 17

        def max_memory_reserved(self):
            return 19

    cuda = FakeCuda()
    memory = iter(
        (
            {
                "current_rss_bytes": 101,
                "peak_rss_bytes": 151,
                "ru_maxrss_bytes": 161,
            },
            {
                "current_rss_bytes": 111,
                "peak_rss_bytes": 171,
                "ru_maxrss_bytes": 181,
            },
        )
    )
    process_io = iter(
        (
            {"read_bytes": 20, "write_bytes": 30},
            {"read_bytes": 25, "write_bytes": 42},
        )
    )
    clock = iter((2.0, 2.5))
    monkeypatch.setattr(smoke_module.torch, "cuda", cuda)
    monkeypatch.setattr(
        smoke_module,
        "_process_memory_snapshot",
        lambda: next(memory),
    )
    monkeypatch.setattr(
        smoke_module,
        "_process_io_snapshot",
        lambda: next(process_io),
    )
    monkeypatch.setattr(smoke_module, "perf_counter", lambda: next(clock))

    sections = {}
    with smoke_module._measured_section(sections, "cpu-test"):
        pass

    row = sections["cpu-test"]
    assert row["completed"] is True
    assert row["wall_seconds"] == pytest.approx(0.5)
    assert row["memory_before"]["current_rss_bytes"] == 101
    assert row["memory_after"]["peak_rss_bytes"] == 171
    assert row["memory_after"]["ru_maxrss_bytes"] == 181
    assert row["process_io_delta"] == {
        "read_bytes": 5,
        "write_bytes": 12,
    }
    assert row["cuda"]["peak_allocated_bytes"] == 17
    assert row["cuda"]["peak_reserved_bytes"] == 19
    assert cuda.synchronize_calls == 2
    assert cuda.reset_calls == 1


def test_smoke_uses_case_major_release_and_explicit_report_sections():
    smoke = SMOKE_PATH.read_text(encoding="utf-8")
    assert "evaluations = evaluator.evaluate_many(candidate_programs)" in smoke
    assert "release_cases=source.release_cases" in smoke
    assert 'finally:\n        with _measured_section(sections, "cohort_release"):' in smoke
    assert "cohort.discard()" in smoke
    for name in (
        '"online_generation"',
        '"required_oracle_query"',
        '"offline_qualification"',
        '"qualified_perturbation_phase_timing"',
        '"qualified_perturbation_online_work"',
        '"lean_online_work_zero"',
        '"decoder_rejections_before_physical_audit"',
        '"physical_audits"',
        '"decoder_attempts_derived"',
    ):
        assert name in smoke


def _physical_commit_values(value):
    if isinstance(value, dict):
        rows = [
            item
            for key, item in value.items()
            if key == "physical_commit"
        ]
        for item in value.values():
            rows.extend(_physical_commit_values(item))
        return rows
    if isinstance(value, list):
        rows = []
        for item in value:
            rows.extend(_physical_commit_values(item))
        return rows
    return []


def test_live_configs_pin_commit_and_smoke_requires_explicit_pin():
    smoke_wrapper = SLURM_PATH.read_text(encoding="utf-8")
    live_wrapper = LIVE_SLURM_PATH.read_text(encoding="utf-8")
    assert "GKW_PHYSICAL_COMMIT:?Set the final pinned lean physical commit" in smoke_wrapper
    assert LEGACY_PHYSICAL_PIN not in smoke_wrapper
    assert PREVIOUS_QUALIFIED_PHYSICAL_PIN not in smoke_wrapper
    assert CONTINUOUS_PHYSICAL_PIN not in smoke_wrapper
    for forbidden in (
        "git fetch",
        "git clone",
        "git init",
        "remote add",
        "pip install",
        "https://github.com",
    ):
        assert forbidden not in live_wrapper
    assert (
        f'GKW_PHYSICAL_COMMIT="${{GKW_PHYSICAL_COMMIT:-{CONTINUOUS_PHYSICAL_PIN}}}"'
        in live_wrapper
    )
    for required in (
        "GKW_PHYSICAL_ROOT:?Set the clean pinned physical checkout",
        "neural-gyrokinetics-568411b",
        "gyaradax-318e3d11",
        "gkw-perturbation-qualification-python-v1",
        "GKW_PERTURBATION_CERTIFICATE",
        "GKW_QUALIFICATION_REPORT",
        "GKW_QUALIFICATION_GENERATOR_MANIFEST",
        "GKW_QUALIFICATION_OPERATOR_MANIFEST",
        "GKW_QUALIFICATION_PHYSICAL_SOURCE_MANIFEST",
        "GKW_QUALIFICATION_SEED_PLAN",
        "GKW_QUALIFICATION_REPORT_SHA256",
        "GKW_QUALIFICATION_GENERATOR_MANIFEST_SHA256",
        "GKW_QUALIFICATION_OPERATOR_MANIFEST_SHA256",
        "GKW_QUALIFICATION_PHYSICAL_SOURCE_MANIFEST_SHA256",
        "GKW_QUALIFICATION_SEED_PLAN_MANIFEST_SHA256",
        "568411b33be486262b8c5cfe1dfb3e0fe97f6419",
        "318e3d11d7326d99e96f2bd5a8998a462261dcd4",
    ):
        assert required in live_wrapper
    for distribution, version in (
        ("kappamodules", "0.1.112"),
        ("einops", "0.8.1"),
        ("omegaconf", "2.3.0"),
        ("antlr4-python3-runtime", "4.9.3"),
        ("PyYAML", "6.0.2"),
    ):
        assert f'"{distribution}": "{version}"' in live_wrapper
    assert live_wrapper.count("status --porcelain --untracked-files=all") >= 4
    assert "${GKW_EXTERNAL_ROOT}:${GYARADAX_ROOT}:${DEPS_ROOT}" in live_wrapper
    assert "run_unified_plasma_agent_iteration.py" in live_wrapper
    assert "--legacy-macro-reference" in live_wrapper
    assert "--config \"${CONFIG}\"" in live_wrapper
    assert "gkw-answer-first-support-audit" not in live_wrapper
    assert "gkw-perturbation-qualification-production-20260828-v5" not in (
        live_wrapper
    )
    assert PREVIOUS_QUALIFIED_PHYSICAL_PIN not in live_wrapper

    for path in LEGACY_CONFIG_PATHS:
        values = _physical_commit_values(
            json.loads(path.read_text(encoding="utf-8"))
        )
        assert values
        assert set(values) == {LEGACY_PHYSICAL_PIN}

    expected_campaigns = {
        "gkw_unified_live_agent_iteration.json": (
            "gkw-unified-continuous-condition-live-20260830-v1"
        ),
        "gkw_unified_parallel_worker_smoke.json": (
            "gkw-unified-continuous-condition-parallel-worker-smoke-"
            "20260830-v1"
        ),
        "gkw_unified_competitive_rehearsal.json": (
            "gkw-unified-continuous-condition-competitive-rehearsal-"
            "20260830-v1"
        ),
        "gkw_unified_competitive_full.json": (
            "gkw-unified-continuous-condition-open-search-fresh-"
            "20260830-v1"
        ),
    }
    expected_bindings = {
        "qualification_report_sha256": "$"
        + "{GKW_QUALIFICATION_REPORT_SHA256}",
        "generator_manifest_sha256": "$"
        + "{GKW_QUALIFICATION_GENERATOR_MANIFEST_SHA256}",
        "operator_manifest_sha256": "$"
        + "{GKW_QUALIFICATION_OPERATOR_MANIFEST_SHA256}",
        "physical_source_manifest_sha256": "$"
        + "{GKW_QUALIFICATION_PHYSICAL_SOURCE_MANIFEST_SHA256}",
        "seed_plan_manifest_sha256": "$"
        + "{GKW_QUALIFICATION_SEED_PLAN_MANIFEST_SHA256}",
    }
    for path in CONTINUOUS_CONFIG_PATHS:
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = _physical_commit_values(payload)
        assert values
        assert payload["campaign_id"] == expected_campaigns[path.name]
        assert set(values) == {CONTINUOUS_PHYSICAL_PIN}
        assert payload["agent"]["cold_start_candidate"].endswith(
            "/configs/candidates/gkw_hold_initial_macro_candidate.py"
        )
        qualification = payload["gkw"]["perturbation_qualification"]
        assert {
            name: qualification[name] for name in expected_bindings
        } == expected_bindings

    assert expected_campaigns[
        "gkw_unified_parallel_worker_smoke.json"
    ] in PARALLEL_SLURM_PATH.read_text(encoding="utf-8")
    assert expected_campaigns[
        "gkw_unified_competitive_rehearsal.json"
    ] in COMPETITIVE_REHEARSAL_SLURM_PATH.read_text(encoding="utf-8")

    retired = json.loads(RETIRED_RECOVERY_CONFIG.read_text(encoding="utf-8"))
    assert retired["retired"] is True
    assert retired["physics"] == "gkw_retired_finite_condition_recovery"
    assert set(_physical_commit_values(retired)) == {
        PREVIOUS_QUALIFIED_PHYSICAL_PIN
    }
