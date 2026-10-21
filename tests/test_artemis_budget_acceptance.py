"""Publication must precede launching the exact accepted recovery revision."""

import subprocess

import pytest

from scripts import artemis_budget_acceptance as gate
from scripts import run_canonical_artemis_campaign as runner


def receipts():
    source = {"commit": "a" * 40, "status_porcelain": ""}
    acceptance = {
        "schema": "artemis_budget_recovery_acceptance_v1", "status": "passed",
        "source_state": dict(source), "backend_config_sha256": gate.BACKEND_CONFIG_SHA256,
        "source_manifest_sha256": "b" * 64,
    }
    publication = {"schema": "artemis_budget_publication_v1", "status": "verified",
                   "commit": source["commit"], "remote": gate.REMOTE, "branch": gate.BRANCH}
    return acceptance, publication, source


def test_exact_acceptance_and_publication_enable_recovery():
    accepted, published, source = receipts()
    report = gate.validate_launch(accepted, published, source)
    assert report["status"] == "passed"
    assert report["candidate_workers"] == 4
    assert report["launch_mode"] == "r1_reevaluation_then_fresh_r2_to_r5"


@pytest.mark.parametrize("target, key, value", [
    ("acceptance", "status", "failed"),
    ("acceptance", "backend_config_sha256", "c" * 64),
    ("publication", "status", "not_pushed"),
    ("publication", "commit", "d" * 40),
    ("publication", "branch", "some-other-branch"),
    ("publication", "remote", "https://example.invalid/not-the-collaborator"),
])
def test_incomplete_or_different_receipts_block_launch(target, key, value):
    accepted, published, source = receipts()
    {"acceptance": accepted, "publication": published}[target][key] = value
    with pytest.raises(ValueError):
        gate.validate_launch(accepted, published, source)


def test_acceptance_source_change_blocks_launch():
    accepted, published, source = receipts()
    source["commit"] = "e" * 40
    with pytest.raises(ValueError, match="exact-revision"):
        gate.validate_launch(accepted, published, source)


def test_dirty_launch_source_is_rejected():
    accepted, published, source = receipts()
    source["status_porcelain"] = " M numerical_code.py"
    with pytest.raises(ValueError, match="exact-revision"):
        gate.validate_launch(accepted, published, source)


@pytest.mark.parametrize("attribute", ["failures", "errors", "skipped"])
def test_junit_does_not_silently_accept_failed_or_skipped_tests(tmp_path, attribute):
    path = tmp_path / "tests.xml"
    path.write_text(f'<testsuites><testsuite tests="1" {attribute}="1"><testcase/></testsuite></testsuites>')
    with pytest.raises(ValueError, match="failure/error/skip"):
        gate.junit_report(path)


def test_junit_reports_subtest_totals_separately_from_testcase_nodes(tmp_path):
    path = tmp_path / "tests.xml"
    path.write_text('<testsuites><testsuite tests="2"><testcase/></testsuite></testsuites>')
    report = gate.junit_report(path)
    assert report["junit_reported_tests"] == 2
    assert report["testcase_nodes"] == 1


def test_worker_override_requires_explicit_recovery():
    with pytest.raises(SystemExit) as result:
        runner.main(["--validate-only", "--candidate-workers", "4"])
    assert result.value.code == 2


def test_recovery_slurm_launch_is_gated_and_scripts_parse():
    scripts = runner.ROOT / "scripts"
    recovery = (scripts / "slurm_artemis_budget_recovery_20260909.sh").read_text()
    assert recovery.index("artemis_budget_acceptance.py launch") < recovery.index("exec bash")
    assert 'ARTEMIS_RECOVERY_CANDIDATE_WORKERS=4' in recovery
    for name in ("slurm_artemis_budget_acceptance_20260909.sh",
                 "slurm_artemis_budget_recovery_20260909.sh", "slurm_artemis_canonical_5round.sh"):
        subprocess.run(("bash", "-n", str(scripts / name)), check=True)
