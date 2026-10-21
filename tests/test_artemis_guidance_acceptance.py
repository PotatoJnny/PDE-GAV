"""Bookkeeping fixtures, not model-quality measurements."""

import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("guidance_acceptance", ROOT / "scripts/artemis_guidance_acceptance.py")
HELPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPER)


def _state():
    return {"canonical_commit": "guided-commit", "backend_config_sha256": "backend",
            "campaign_config_sha256": "campaign"}


def _reports():
    return ({"status": "passed", "canonical_commit": HELPER.COLD_COMMIT,
             "backend_config_sha256": HELPER.COLD_BACKEND_SHA256},
            {"status": "passed", **_state()})


def test_launch_reuses_d2_acceptance_and_requires_exact_guidance_revision():
    cold, guidance = _reports()
    report = HELPER.validate_launch(cold, guidance, _state())
    assert report["status"] == "passed"
    assert report["launch_mode"] == "new_full_five_round_bgp_campaign"
    assert report["scientific_quality_gate"] is False


@pytest.mark.parametrize("arm,key", [("cold", "status"), ("cold", "canonical_commit"),
    ("cold", "backend_config_sha256"), ("guidance", "status"),
    ("guidance", "canonical_commit"), ("guidance", "backend_config_sha256"),
    ("guidance", "campaign_config_sha256")])
def test_launch_rejects_failed_or_changed_acceptance(arm, key):
    cold, guidance = _reports()
    {"cold": cold, "guidance": guidance}[arm][key] = "changed"
    with pytest.raises(ValueError):
        HELPER.validate_launch(cold, guidance, _state())


def _fixture(root):
    (root / "smoke").mkdir()
    HELPER.write_report(root, "source-state.json", _state())
    HELPER.write_report(root, "junit-counts.json", {"status": "passed", "totals": {
        "tests": 3, "failures": 0, "errors": 0, "skipped": 0}})
    HELPER.write_report(root / "smoke", "acceptance-summary.json", {
        "status": "passed", **_state(), "rounds_completed": 2,
        "guidance_fit_count": 2, "guided_case_count": 4})


def test_summary_accepts_two_round_engineering_without_hardness_gate(tmp_path, monkeypatch):
    _fixture(tmp_path)
    monkeypatch.setattr(HELPER, "source_state", _state)
    report = HELPER.summarize(tmp_path, 0)
    assert report["status"] == "passed" and not report["problems"]
    assert report["model_api_called"] is False


@pytest.mark.parametrize("problem", ["tests", "missing_smoke", "smoke_failed", "rounds",
                                    "fits", "no_guided", "code_changed", "command_failed"])
def test_summary_preserves_failures(tmp_path, monkeypatch, problem):
    _fixture(tmp_path)
    monkeypatch.setattr(HELPER, "source_state", _state)
    smoke_path = tmp_path / "smoke/acceptance-summary.json"
    smoke = json.loads(smoke_path.read_text())
    if problem == "tests":
        HELPER.write_report(tmp_path, "junit-counts.json", {"status": "failed"})
    elif problem == "missing_smoke":
        smoke_path.unlink()
    elif problem == "code_changed":
        monkeypatch.setattr(HELPER, "source_state", lambda: {**_state(), "canonical_commit": "new"})
    elif problem != "command_failed":
        key, value = {"smoke_failed": ("status", "failed"), "rounds": ("rounds_completed", 1),
                      "fits": ("guidance_fit_count", 1), "no_guided": ("guided_case_count", 0)}[problem]
        smoke[key] = value
        HELPER.write_report(tmp_path / "smoke", "acceptance-summary.json", smoke)
    report = HELPER.summarize(tmp_path, int(problem == "command_failed"))
    assert report["status"] == "failed" and report["problems"]
    assert (tmp_path / "acceptance-summary.json").is_file()


def test_guidance_smoke_is_short_compute_only_and_formal_launch_has_both_gates():
    smoke = ROOT / "scripts/slurm_artemis_guidance_acceptance_20260908.sh"
    formal = ROOT / "scripts/slurm_artemis_canonical_guidance_20260908.sh"
    for path in (smoke, formal):
        subprocess.run(["bash", "-n", str(path)], check=True, capture_output=True)
    text = smoke.read_text()
    assert "#SBATCH --time=00:30:00" in text
    assert "#SBATCH --gres=gpu:l40s:1" in text
    assert "--official-count 96" in text
    assert "trap artemis_guidance_finish EXIT" in text
    assert "--auth-file" not in text and "sbatch " not in text
    assert text.index("junit --root") < text.index("scripts/check_artemis_guidance_execution.py")
    text = formal.read_text()
    assert '--cold-report "$1" --guidance-report "$2"' in text
    assert "artemis_canonical_d2_guided_5round_20260908" in text
    assert "exec bash scripts/slurm_artemis_canonical_5round.sh" in text
