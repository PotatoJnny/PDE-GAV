from __future__ import annotations

from copy import deepcopy
import errno
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_unified_plasma_agent_iteration.py"
_SPEC = importlib.util.spec_from_file_location("unified_gkw_strong", SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
UNIFIED = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(UNIFIED)

REHEARSAL = ROOT / "configs" / "gkw_unified_strong_rehearsal.json"
FULL = ROOT / "configs" / "gkw_unified_strong_full.json"
COMPETITIVE_FULL = ROOT / "configs" / "gkw_unified_competitive_full.json"
LIVE = ROOT / "configs" / "gkw_unified_live_agent_iteration.json"
FRESH_EQUAL = ROOT / "configs" / "gkw_unified_fresh_equal_pilot.json"
FRESH_PRIMARY = (
    ROOT / "configs" / "gkw_unified_fresh_primary_guardrail_pilot.json"
)
RECOVERY = ROOT / "configs" / "gkw_unified_competitive_recovery.json"
RECOVERY_WRAPPER = ROOT / "scripts" / "slurm_gkw_unified_competitive_recovery.sh"


def _payload(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def _iteration_config(path: Path, tmp_path: Path):
    payload = _payload(path)
    agent = payload["agent"]
    assert isinstance(agent, dict)
    for name in ("cold_start_candidate", "agent_launcher", "codex_binary", "auth_file"):
        target = tmp_path / name
        target.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        if name in {"agent_launcher", "codex_binary"}:
            target.chmod(0o700)
        agent[name] = str(target)
    payload["seed_base"] = 25_082_500
    return UNIFIED._iteration_config(payload, run_root=tmp_path / "run")


def test_finite_condition_recovery_is_explicitly_retired():
    payload = _payload(RECOVERY)
    assert payload["retired"] is True
    assert "GeneratedCodes v1" in payload["retired_reason"]
    assert payload["physics"] == "gkw_retired_finite_condition_recovery"
    assert payload["gkw"]["physical_commit"] == (
        "5ca50ae3c78d48015f8b9d5ffd1eda5347f52748"
    )

    wrapper = RECOVERY_WRAPPER.read_text(encoding="utf-8")
    message = "finite-condition GeneratedCodes v1"
    assert message in wrapper
    assert wrapper.index("exit 2") < wrapper.index(
        "GKW_RECOVERY_COLD_START"
    )

def test_rehearsal_policy_is_two_fresh_agents_with_root_based_strong_feedback(
    tmp_path: Path,
):
    payload = _payload(REHEARSAL)
    assert "seed_base" not in payload
    policy = UNIFIED._controller_policy(payload)

    assert policy["profile"] == "gkw_strong_rehearsal_v1"
    assert policy["agent_count"] == 2
    assert policy["round_count"] == 1
    assert policy["agent_timeout_seconds"] == 21600.0
    assert policy["agent_hard_timeout_seconds"] == 36000.0
    assert policy["development_pool_multiplier"] == 1
    assert policy["development_root_count"] == 16
    assert policy["development_case_count"] == 32
    assert policy["selection_root_count"] == 8
    assert policy["selection_case_count"] == 16
    assert [stage["root_count"] for stage in policy["evidence_stages"]] == [4, 8]
    assert [stage["case_count"] for stage in policy["evidence_stages"]] == [8, 16]
    assert policy["terminal_root_count"] == 16
    assert policy["terminal_case_count"] == 32
    assert policy["development_release_root_limit"] == 16
    assert policy["development_case_credits"] == 32
    assert policy["development_screen_root_stages"] == [4, 16]
    assert policy["development_screen_root_credit_limit"] == 64
    assert policy["development_screen_confidence_z"] == pytest.approx(
        1.6448536269514722
    )
    assert policy["development_screen_continuation_max_mean_ratio"] == 1.05
    assert policy["development_screen_timeout_seconds"] == 3600.0
    assert policy["allow_development_truth"] is False

    config = _iteration_config(REHEARSAL, tmp_path)
    assert config.agent_count == 2
    assert config.agent_timeout_seconds == 21600.0
    assert config.agent_hard_timeout_seconds == 36000.0
    assert config.development_case_count == 32
    assert config.development_release_root_limit == 16
    assert config.development_case_credits == 32
    assert config.development_screen_root_stages == (4, 16)
    assert config.development_screen_root_credit_limit == 64
    assert config.development_screen_confidence_z == pytest.approx(
        1.6448536269514722
    )
    assert config.development_screen_continuation_max_mean_ratio == 1.05
    assert config.development_screen_timeout_seconds == 3600.0
    assert "public_problem_manifest" in config.prompt
    assert "query_reference_public_oracle" in config.prompt
    assert "no protocol-level lifetime or sample quota" in config.prompt
    assert "screen_reference_candidate" in config.prompt
    assert "requested_root_count=4" in config.prompt
    assert "Request sizes 4, 16 are efficiency recommendations" in config.prompt
    assert "not stages or promotion requirements" in config.prompt
    assert "draws fresh randomness and accumulates evidence" in config.prompt
    assert "no fixed lifetime root quota" in config.prompt
    assert "controller-published 3600.0-second RPC deadline" in config.prompt
    assert "21600.0-second Agent active-time budget" in config.prompt
    assert "36000.0-second hard wall" in config.prompt
    assert "Controller numerical-service time is excluded" in config.prompt
    assert "prior screening is unnecessary" in config.prompt
    assert "prepare_solution(case, solve_budget=None)" in config.prompt
    assert "candidate preparation and point" in config.prompt
    assert "legacy one-shot" in config.prompt
    assert "You may search the web" in config.prompt
    assert "There is no search completion quota" in config.prompt
    assert "aggregate loss/CVaR/component" in config.prompt
    assert "controller-computed bundle digests are opaque" in config.prompt
    assert "component regressions and a finite-sample maximum" in config.prompt
    assert "evaluate_reference_candidate" not in config.prompt
    assert "return_finite_truth" not in config.prompt
    assert "one official nomination" in config.prompt


def test_full_profile_matches_old_main_scale_as_a_configuration_benchmark(
    tmp_path: Path,
):
    payload = _payload(FULL)
    assert "seed_base" not in payload
    policy = UNIFIED._controller_policy(payload)

    assert policy["profile"] == "gkw_strong_full_v1"
    assert policy["agent_count"] == 2
    assert policy["round_count"] == 4
    assert policy["agent_timeout_seconds"] == 21600.0
    assert policy["agent_hard_timeout_seconds"] == 86400.0
    assert policy["development_pool_multiplier"] == 1
    assert policy["development_case_count"] == 64
    assert policy["selection_case_count"] == 256
    assert [stage["root_count"] for stage in policy["evidence_stages"]] == [
        4,
        16,
        64,
        128,
    ]
    assert policy["terminal_case_count"] == 512
    assert policy["development_release_root_limit"] == 32
    assert policy["development_screen_root_stages"] == [4, 16, 64]
    assert policy["development_screen_root_credit_limit"] == 4096
    assert policy["allow_development_truth"] is False
    assert policy["development_screen_timeout_seconds"] == 18000.0

    config = _iteration_config(FULL, tmp_path)
    assert config.round_count == 4
    assert config.agent_timeout_seconds == 21600.0
    assert config.agent_hard_timeout_seconds == 86400.0
    assert config.development_screen_root_stages == (4, 16, 64)
    assert config.development_screen_timeout_seconds == 18000.0
    assert [stage.case_count for stage in config.evidence_stages] == [
        8,
        32,
        128,
        256,
    ]


def test_formal_gkw_config_disables_private_cohort_storage(tmp_path: Path):
    config = _iteration_config(COMPETITIVE_FULL, tmp_path)
    assert config.cohort_store_root is None


def test_recovery_policy_backfills_defaults_but_rejects_explicit_promotion(
    tmp_path: Path,
):
    payload = deepcopy(_payload(LIVE))
    payload["campaign_id"] = "historical-policy-recovery-test"
    payload["seed_base"] = 12345
    controller_policy = UNIFIED._controller_policy(payload)

    config_path = tmp_path / "campaign.json"
    _write_json(config_path, payload)
    config_sha256 = UNIFIED._sha256_file(config_path)
    source_root = tmp_path / "source-run"
    source_root.mkdir()
    source_commit = "a" * 40
    target_commit = "b" * 40

    historical_policy = deepcopy(controller_policy)
    historical_policy.pop("promotion")
    source_seed = {
        "schema": UNIFIED.CAMPAIGN_SEED_SCHEMA,
        "campaign_id": payload["campaign_id"],
        "seed_base": payload["seed_base"],
        "source": "configured",
        "replay": {
            "config_field": "seed_base",
            "value": payload["seed_base"],
        },
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "unified_commit": source_commit,
        "controller_policy": historical_policy,
        "seed_derivation": "macro_loop_seed_schedule_v1",
    }
    source_seed_path = source_root / "campaign_seed.json"
    _write_json(source_seed_path, source_seed)

    manifest = {
        "schema": UNIFIED.RECOVERY_MANIFEST_SCHEMA,
        "phase": UNIFIED.RECOVERY_PHASE,
        "source": {
            "run_root": str(source_root),
            "campaign_seed_sha256": UNIFIED._sha256_file(source_seed_path),
            "config_sha256": config_sha256,
            "unified_commit": source_commit,
            "campaign_id": payload["campaign_id"],
            "seed_base": payload["seed_base"],
            "seed_stride": controller_policy["seed_stride"],
            "plugin_identity": {"plugin_id": "gkw"},
            "cold_start_sha256": "c" * 64,
        },
        "target": {
            "unified_commit": target_commit,
            "plugin_identity": {"plugin_id": "gkw"},
            "official_scoring_protocol": "fresh-common-hidden-v1",
            "fresh_official_evaluation": True,
        },
        "workers": [{"agent_id": "agent-1"}],
        "source_common_evidence_imported": False,
    }
    manifest_path = tmp_path / "recovery-manifest.json"
    _write_json(manifest_path, manifest)

    resolved, record, _ = UNIFIED._resolve_recovery_campaign_seed(
        payload=payload,
        manifest_argument=manifest_path,
        expected_manifest_sha256=UNIFIED._sha256_file(manifest_path),
        config_sha256=config_sha256,
        controller_policy=controller_policy,
        unified_commit=target_commit,
    )
    assert resolved["seed_base"] == payload["seed_base"]
    assert record["source"] == UNIFIED.RECOVERY_SEED_SOURCE

    explicit_payload = deepcopy(payload)
    explicit_payload["controller_policy"]["promotion"] = {
        "preset": "minimize_metric",
        "settings": {
            "primary": {"distribution": "base", "statistic": "mean"},
        },
    }
    source_seed["controller_policy"] = UNIFIED._controller_policy(
        explicit_payload
    )
    _write_json(source_seed_path, source_seed)
    manifest["source"]["campaign_seed_sha256"] = UNIFIED._sha256_file(
        source_seed_path
    )
    _write_json(manifest_path, manifest)

    with pytest.raises(
        ValueError,
        match="recovery source campaign seed record is inconsistent",
    ):
        UNIFIED._resolve_recovery_campaign_seed(
            payload=payload,
            manifest_argument=manifest_path,
            expected_manifest_sha256=UNIFIED._sha256_file(manifest_path),
            config_sha256=config_sha256,
            controller_policy=controller_policy,
            unified_commit=target_commit,
        )


def test_live_profile_is_fresh_primary_guardrail_with_strong_search(
    tmp_path: Path,
):
    payload = _payload(LIVE)
    assert "seed_base" not in payload
    policy = UNIFIED._controller_policy(payload)

    assert policy["profile"] == "gkw_fresh_primary_guardrail_clean_gate_v1"
    assert policy["agent_count"] == 1
    assert policy["round_count"] == 1
    assert policy["development_root_count"] == 32
    assert policy["development_release_root_limit"] == 4096
    assert policy["development_screen_root_stages"] == [4, 16, 64]
    assert policy["development_screen_root_credit_limit"] == 4096
    assert policy["agent_timeout_seconds"] == 21600.0
    assert policy["agent_hard_timeout_seconds"] == 86400.0
    assert (
        policy["paired_objective"]["mode"]
        == "primary_with_perturbed_guardrail"
    )

    config = _iteration_config(LIVE, tmp_path)
    for phrase in (
        "generated/base distribution is the optimization",
        "Fresh releases and screens are random",
        "There is no search completion quota",
        "prepare_solution(case, solve_budget=None)",
        "You may search the web",
        "one common hidden cohort",
    ):
        assert phrase in config.prompt


def test_fresh_objective_arms_have_equal_search_resources_and_distinct_rules(
    tmp_path: Path,
):
    equal = UNIFIED._controller_policy(_payload(FRESH_EQUAL))
    primary = UNIFIED._controller_policy(_payload(FRESH_PRIMARY))

    common_fields = (
        "agent_count",
        "round_count",
        "development_root_count",
        "development_release_root_limit",
        "development_screen_root_stages",
        "development_screen_root_credit_limit",
        "selection_root_count",
        "terminal_root_count",
        "candidate_compute_budget",
        "agent_timeout_seconds",
        "agent_hard_timeout_seconds",
    )
    assert {name: equal[name] for name in common_fields} == {
        name: primary[name] for name in common_fields
    }
    assert equal["agent_count"] == 1
    assert equal["round_count"] == 1
    assert equal["development_release_root_limit"] == 4096
    assert equal["development_screen_root_credit_limit"] == 4096
    assert equal["paired_objective"]["mode"] == "weighted_composite"
    assert (
        primary["paired_objective"]["mode"]
        == "primary_with_perturbed_guardrail"
    )

    config = _iteration_config(FRESH_PRIMARY, tmp_path)
    assert (
        config.paired_objective_policy.mode.value
        == "primary_with_perturbed_guardrail"
    )
    for phrase in (
        "generated/base distribution is the optimization",
        "release_reference_completed_answers",
        "submit_reference_candidate_screen",
        "Fresh releases and screens are random",
        "no protocol-level lifetime or sample quota",
        "prepare_solution(case, solve_budget=None)",
        "one official nomination",
    ):
        assert phrase in config.prompt


def test_fresh_objective_array_maps_two_independent_replicates_per_arm():
    wrapper = (
        ROOT / "scripts" / "slurm_gkw_unified_fresh_objective_ab.sh"
    ).read_text(encoding="utf-8")

    assert "#SBATCH --array=0-3" in wrapper
    assert "#SBATCH --gres=gpu:l40s:1" in wrapper
    assert "#SBATCH --partition=gpubase_l40s_b4" in wrapper
    assert "#SBATCH --time=3-00:00:00" in wrapper
    assert "WATCHDOG_SECONDS=255600" in wrapper
    assert "timeout --verbose --signal=TERM" in wrapper
    assert '--kill-after="${WATCHDOG_GRACE_SECONDS}s"' in wrapper
    assert "controller_watchdog_failure.json" in wrapper
    assert 'exec "${UNIFIED_ROOT}/scripts/' not in wrapper
    assert "gkw_unified_fresh_equal_pilot.json" in wrapper
    assert "gkw_unified_fresh_primary_guardrail_pilot.json" in wrapper
    assert 'REPLICATE="1"' in wrapper
    assert 'REPLICATE="2"' in wrapper
    assert "SLURM_ARRAY_JOB_ID" in wrapper
    assert "sbatch " not in wrapper


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        (
            "development_screen_root_stages",
            [8, 4],
            "screen root stages must strictly increase",
        ),
        (
            "development_screen_root_credit_limit",
            3,
            "credits cannot fund the first stage",
        ),
        (
            "development_screen_continuation_max_mean_ratio",
            0.0,
            "below its numeric bound",
        ),
        (
            "development_screen_timeout_seconds",
            14400.1,
            "hard-wall service allowance must cover one screen timeout",
        ),
        (
            "agent_hard_timeout_seconds",
            21599.0,
            "hard timeout cannot be below its active-time timeout",
        ),
    ),
)
def test_strong_policy_rejects_invalid_screen_contract(field, value, message):
    payload = deepcopy(_payload(REHEARSAL))
    payload["controller_policy"][field] = value

    with pytest.raises(ValueError, match=message):
        UNIFIED._controller_policy(payload)


def test_v1_policy_without_screen_timeout_keeps_legacy_default():
    payload = deepcopy(_payload(REHEARSAL))
    payload["controller_policy"].pop("development_screen_timeout_seconds")

    policy = UNIFIED._controller_policy(payload)

    assert policy["development_screen_timeout_seconds"] == 300.0


def test_strong_configs_reuse_the_verified_live_gkw_artifact_pins():
    expected = _payload(LIVE)["gkw"]["artifact_sha256"]

    for path in (REHEARSAL, FULL):
        actual = _payload(path)["gkw"]["artifact_sha256"]
        assert actual == expected
        assert all(len(digest) == 64 for digest in actual.values())


def test_gkw_qualification_settings_are_fail_closed_and_exact(tmp_path):
    live = _payload(LIVE)["gkw"]
    assert live["physical_commit"] == (
        "80632f11a39b8577124ffa652556af1521d9b992"
    )
    assert live["perturbation_qualification"] == {
        "certificate": "${GKW_PERTURBATION_CERTIFICATE}",
        "qualification_report": "${GKW_QUALIFICATION_REPORT}",
        "generator_manifest": "${GKW_QUALIFICATION_GENERATOR_MANIFEST}",
        "operator_manifest": "${GKW_QUALIFICATION_OPERATOR_MANIFEST}",
        "physical_source_manifest": "${GKW_QUALIFICATION_PHYSICAL_SOURCE_MANIFEST}",
        "seed_plan": "${GKW_QUALIFICATION_SEED_PLAN}",
        "qualification_report_sha256": "$" + "{GKW_QUALIFICATION_REPORT_SHA256}",
        "generator_manifest_sha256": "$" + "{GKW_QUALIFICATION_GENERATOR_MANIFEST_SHA256}",
        "operator_manifest_sha256": "$" + "{GKW_QUALIFICATION_OPERATOR_MANIFEST_SHA256}",
        "physical_source_manifest_sha256": "$" + "{GKW_QUALIFICATION_PHYSICAL_SOURCE_MANIFEST_SHA256}",
        "seed_plan_manifest_sha256": "$" + "{GKW_QUALIFICATION_SEED_PLAN_MANIFEST_SHA256}",
    }
    digest = "a" * 64
    with pytest.raises(ValueError, match="perturbation_qualification"):
        UNIFIED._gkw_qualification_settings({})

    qualification_paths = {
        name: tmp_path / f"{name}.json"
        for name in UNIFIED.GKW_QUALIFICATION_PATH_FIELDS
    }
    qualification = {
        **{name: str(path) for name, path in qualification_paths.items()},
        **{
            name: digest
            for name in UNIFIED.GKW_QUALIFICATION_BINDING_FIELDS
        },
    }
    paths, bindings = UNIFIED._gkw_qualification_settings(
        {"perturbation_qualification": qualification}
    )
    assert paths == {
        name: path.resolve() for name, path in qualification_paths.items()
    }
    assert bindings == {
        name: digest for name in UNIFIED.GKW_QUALIFICATION_BINDING_FIELDS
    }

    incomplete = dict(qualification)
    incomplete.pop("seed_plan_manifest_sha256")
    with pytest.raises(ValueError, match="changed or are incomplete"):
        UNIFIED._gkw_qualification_settings(
            {"perturbation_qualification": incomplete}
        )

    unexpected = dict(qualification, allow_unqualified=True)
    with pytest.raises(ValueError, match="changed or are incomplete"):
        UNIFIED._gkw_qualification_settings(
            {"perturbation_qualification": unexpected}
        )



def test_gkw_generator_identity_binds_old_strength_formal_panel():
    source = SCRIPT.read_text(encoding="utf-8")

    assert '"formal_macro_time_count": 4' in source
    assert "formal_macro_time_count=4" in source


def test_storage_roots_keep_only_durable_gkw_data(tmp_path):
    durable = tmp_path / "durable"
    durable.mkdir()
    payload = {
        "physics": "gkw",
        "storage_root": str(durable),
        "gkw": {},
    }
    run_root = durable / "campaign"
    roots = UNIFIED._storage_roots(payload, run_root)

    assert roots == {"durable": durable}


def test_gkw_wiring_has_no_oracle_replay_storage_contract():
    for path in ROOT.glob("configs/gkw_unified_*.json"):
        settings = _payload(path)["gkw"]
        assert "oracle_replay_root" not in settings
        assert "oracle_replay_storage_root" not in settings

    source = SCRIPT.read_text(encoding="utf-8")
    for obsolete in (
        "MINIMUM_GKW_REPLAY_FREE_BYTES",
        "MINIMUM_GKW_COMBINED_CACHE_FREE_BYTES",
        '"oracle_replay_root"',
        '"oracle_replay_storage_root"',
        '"oracle_replay_free_bytes"',
        '"minimum_oracle_replay_free_bytes"',
        '"combined_cache_free_bytes"',
        '"minimum_combined_cache_free_bytes"',
        '"oracle_replay_role"',
        "MINIMUM_GKW_COHORT_STORE_FREE_BYTES",
        '"cohort_store_root"',
        '"cohort_store_free_bytes"',
        '"minimum_cohort_store_free_bytes"',
        '"cohort_store_role"',
    ):
        assert obsolete not in source
    assert '"official_selection_terminal_mode": (' in source
    assert '"root_pair_streamed_memory_only"' in source
    assert '"cohort_disk_persistence": False' in source


def test_storage_preflight_reports_enospc_before_runtime_build(monkeypatch):
    monkeypatch.setattr(
        UNIFIED.shutil, "disk_usage", lambda root: SimpleNamespace(free=0)
    )
    with pytest.raises(OSError) as exc_info:
        UNIFIED._require_free_bytes(Path("/scratch"), 1, "test storage")
    assert exc_info.value.errno == errno.ENOSPC


def test_slurm_wrapper_is_one_l40s_and_fails_closed_for_full_profile():
    wrapper_path = (
        ROOT / "scripts" / "slurm_gkw_unified_strong_agent_iteration.sh"
    )
    wrapper = wrapper_path.read_text(encoding="utf-8")
    base = (
        ROOT / "scripts" / "slurm_gkw_unified_live_agent_iteration.sh"
    ).read_text(encoding="utf-8")
    smoke = (
        ROOT / "scripts" / "slurm_gkw_unified_runtime_smoke.sh"
    ).read_text(encoding="utf-8")
    competitive = (
        ROOT / "scripts" / "slurm_gkw_unified_competitive_full.sh"
    ).read_text(encoding="utf-8")

    assert "#SBATCH --gres=gpu:l40s:1" in wrapper
    assert "#SBATCH --cpus-per-task=16" in wrapper
    assert "#SBATCH --mem=96G" in wrapper
    assert 'GKW_STRONG_PROFILE="${GKW_STRONG_PROFILE:-rehearsal}"' in wrapper
    assert "gkw_unified_strong_rehearsal.json" in wrapper
    assert "gkw_unified_strong_full.json" in wrapper
    assert "requires a segmented or parallel launcher" in wrapper
    assert "GKW_STRONG_PROFILE must be rehearsal or full" in wrapper
    assert "sbatch " not in wrapper
    assert "GKW_UNIFIED_CONFIG:-" in base
    assert "PLASMA_RUN_ROOT:-" in base
    assert "PLASMA_STORAGE_ROOT" in wrapper
    assert "${PDE_GAV_RUN_ROOT}" in wrapper
    assert "${PDE_GAV_RUN_ROOT}" in base
    assert "${PDE_GAV_RUN_ROOT}" in smoke
    assert "SLURM_TMPDIR" in smoke
    assert "GKW_ORACLE_REPLAY_ROOT" not in base
    assert "GKW_ORACLE_REPLAY_STORAGE_ROOT" not in base
    assert "ORACLE_REPLAY_ROOT" not in smoke
    assert "ORACLE_REPLAY_STORAGE_ROOT" not in smoke
    assert "PLASMA_RECOVERY_MANIFEST:-" in base
    assert "PLASMA_RECOVERY_MANIFEST_SHA256:-" in base
    assert "--recover-development-manifest" in base
    assert "--recover-development-manifest-sha256" in base
    assert (
        'exec "${UNIFIED_ROOT}/scripts/slurm_gkw_unified_live_agent_iteration.sh"'
        in competitive
    )

    rejected = subprocess.run(
        ("bash", str(wrapper_path)),
        check=False,
        capture_output=True,
        text=True,
        env={
            "PDE_ENABLE_LEGACY_MACRO_REFERENCE": "1",
            "GKW_STRONG_PROFILE": "full",
            "UNIFIED_PLASMA_ROOT": str(ROOT),
            "PATH": "/usr/bin:/bin",
        },
    )
    assert rejected.returncode == 2
    assert "requires a segmented or parallel launcher" in rejected.stderr
