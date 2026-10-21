from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
ARRAY_WRAPPER = ROOT / "scripts" / "slurm_gkw_unified_fresh_objective_ab.sh"
LIVE_WRAPPER = ROOT / "scripts" / "slurm_gkw_unified_live_agent_iteration.sh"
SMOKE_WRAPPER = (
    ROOT / "scripts" / "slurm_gkw_unified_lifecycle_memory_smoke.sh"
)
SMOKE_CONFIG = ROOT / "configs" / "gkw_unified_lifecycle_memory_smoke.json"
RUNNER = ROOT / "scripts" / "run_unified_plasma_agent_iteration.py"

_SPEC = importlib.util.spec_from_file_location("gkw_lifecycle_smoke_runner", RUNNER)
assert _SPEC is not None and _SPEC.loader is not None
UNIFIED = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(UNIFIED)


def test_watchdog_uses_private_timeout_diagnostic_and_new_root():
    wrapper = ARRAY_WRAPPER.read_text(encoding="utf-8")
    live = LIVE_WRAPPER.read_text(encoding="utf-8")

    assert "unified-gkw-fresh-objective-ab-20260827-v2" in wrapper
    assert "GKW_FRESH_AB_ROOT" in wrapper
    assert "TIMEOUT_DIAGNOSTIC" in wrapper
    assert "bash -c 'exec 2>&3 3>&-; exec" in wrapper
    assert "2> " in wrapper
    assert (
        "grep -Eq '^timeout: sending signal (TERM|KILL) to command '"
        in wrapper
    )
    assert "WATCHDOG_TRIGGERED" in wrapper
    assert "-eq 124 ||" not in wrapper
    assert "controller_process_failure.json" in wrapper
    assert '"reason": "controller_process_failure"' in wrapper
    assert "timeout_signal_observed" in wrapper
    assert "sealed-macro-cohorts-" not in live
    assert "unset PLASMA_COHORT_STORAGE_ROOT" in live
    assert "export PLASMA_COHORT_STORAGE_ROOT" not in live


def test_non_timeout_sigkill_is_generic_and_child_stderr_stays_live(
    tmp_path: Path,
):
    fake_root = tmp_path / "unified"
    fake_scripts = fake_root / "scripts"
    fake_scripts.mkdir(parents=True)
    fake_controller = fake_scripts / "slurm_gkw_unified_live_agent_iteration.sh"
    fake_controller.write_text(
        "#!/usr/bin/env bash\n"
        "echo child-stderr-remains-live >&2\n"
        "kill -KILL \"$$\"\n",
        encoding="utf-8",
    )
    fake_controller.chmod(0o700)

    slurm_tmp = tmp_path / "slurm-tmp"
    storage = tmp_path / "storage"
    slurm_tmp.mkdir()
    storage.mkdir()
    completed = subprocess.run(
        ("bash", str(ARRAY_WRAPPER)),
        check=False,
        capture_output=True,
        text=True,
        timeout=10.0,
        env={
            "PDE_ENABLE_LEGACY_MACRO_REFERENCE": "1",
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "SLURM_JOB_ID": "9001",
            "SLURM_ARRAY_JOB_ID": "9000",
            "SLURM_ARRAY_TASK_ID": "0",
            "SLURM_TMPDIR": str(slurm_tmp),
            "UNIFIED_PLASMA_ROOT": str(fake_root),
            "PLASMA_STORAGE_ROOT": str(storage),
        },
    )

    assert completed.returncode == 137
    assert "child-stderr-remains-live" in completed.stderr
    run_root = (
        storage
        / "unified-gkw-fresh-objective-ab-20260827-v2"
        / "equal"
        / "replicate-1-array-9000-task-0"
    )
    generic = json.loads(
        (run_root / "controller_process_failure.json").read_text(
            encoding="utf-8"
        )
    )
    assert generic["reason"] == "controller_process_failure"
    assert generic["timeout_signal_observed"] is False
    assert generic["exit_status"] == 137
    assert not (run_root / "controller_watchdog_failure.json").exists()
    assert not list(slurm_tmp.glob("gkw-controller-timeout-*"))


def test_no_codex_lifecycle_smoke_is_small_but_reaches_terminal_result():
    payload = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
    policy = UNIFIED._controller_policy(payload)
    wrapper = SMOKE_WRAPPER.read_text(encoding="utf-8")

    assert "seed_base" not in payload
    assert payload["agent"]["agent_launcher"] == "/usr/bin/false"
    assert policy["agent_count"] == 1
    assert policy["round_count"] == 1
    assert policy["development_root_count"] == 2
    assert policy["development_release_root_limit"] == 2
    assert policy["selection_root_count"] == 8
    assert [stage["root_count"] for stage in policy["evidence_stages"]] == [2, 8]
    assert policy["terminal_root_count"] == 16
    assert policy["selection_case_count"] == 16
    assert policy["terminal_case_count"] == 32
    assert policy["paired_objective"]["mode"] == (
        "primary_with_perturbed_guardrail"
    )

    assert "#SBATCH --gres=gpu:l40s:1" in wrapper
    assert "#SBATCH --mem=32G" in wrapper
    assert "#SBATCH --time=04:00:00" in wrapper
    assert "unified-gkw-lifecycle-memory-smoke-20260827-v1" in wrapper
    assert "/usr/bin/time -v" in wrapper
    assert "result/result.json" in wrapper
    assert "_controller/sampling_ledger.json" in wrapper
    assert 'terminal.get("evaluation_status") == "valid"' in wrapper
    assert "invalid_count == 0" in wrapper
    assert "valid_count == case_count" in wrapper
    assert "MAX_RSS_LIMIT_KB=$((28 * 1024 * 1024))" in wrapper
    assert "sbatch " not in wrapper
