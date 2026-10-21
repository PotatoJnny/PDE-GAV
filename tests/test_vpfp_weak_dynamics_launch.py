"""No real scheduler calls; fixed-budget and launcher mechanics."""
import copy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import re
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = spec_from_file_location("weak_dynamics_launch_under_test", ROOT / "scripts/run_vpfp_weak_dynamics_launch.py")
launch = module_from_spec(spec)
spec.loader.exec_module(launch)


def accounting():
    prior = {job: {"job_id": job, "state": state, "exit_code": code,
        "elapsed_seconds": elapsed, "allocated_gpus": 1}
        for job, (state, code, elapsed) in launch.PRIOR.items()}
    current = {"job_id": "9999999", "state": "RUNNING", "exit_code": "0:0",
        "elapsed_seconds": 20, "allocated_gpus": 1, "time_limit_minutes": 360}
    control = {"JobId": "9999999", "JobState": "RUNNING", "TimeLimit": "06:00:00", "RunTime": "00:00:25"}
    return prior, current, control


def test_weak_remaining_budget_counts_failed_allocation():
    result = launch.validate_accounting(*accounting())
    assert result["weak_allocated_gpu_seconds_before_this_job"] == 11708
    assert result["remaining_current_seconds"] == 21575
    assert launch.WEAK_CAP_SECONDS - 11708 == 60292


@pytest.mark.parametrize("change", ["missing_failed", "elapsed", "exit", "two_gpu",
    "over_cap", "no_override", "not_running", "wrong_job", "limit_mismatch"])
def test_budget_rejects_changed_receipts_or_allocation(change):
    prior, current, control = accounting()
    if change == "missing_failed": del prior["5268596"]
    elif change == "elapsed": prior["5268892"]["elapsed_seconds"] -= 1
    elif change == "exit": prior["5268596"]["exit_code"] = "0:0"
    elif change == "two_gpu": current["allocated_gpus"] = 2
    elif change == "over_cap": control["TimeLimit"] = "07:00:00"; current["time_limit_minutes"] = 420
    elif change == "no_override": control["TimeLimit"] = "00:01:00"; current["time_limit_minutes"] = 1
    elif change == "not_running": control["JobState"] = "PENDING"
    elif change == "wrong_job": control["JobId"] = "5268892"
    elif change == "limit_mismatch": current["time_limit_minutes"] = 359
    with pytest.raises(ValueError):
        launch.validate_accounting(prior, current, control)


def test_cpu_gate_requires_same_revision_and_success(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "SCRATCH", tmp_path)
    receipt = tmp_path / "receipt.tsv"
    content = "stage\texit_code\nsource_revision\tabc\nmechanics_tests\t0\nlaunch\t0\nrequested_gpus\t0\nscientific_admission\tfalse\n"
    receipt.write_text(content)
    digest = launch.file_sha(receipt)
    launch.validate_cpu_gate(receipt, digest, "abc")
    with pytest.raises(ValueError): launch.validate_cpu_gate(receipt, digest, "def")
    with pytest.raises(ValueError): launch.validate_cpu_gate(receipt, "0" * 64, "abc")
    receipt.write_text(content.replace("mechanics_tests\t0", "mechanics_tests\t1"))
    with pytest.raises(ValueError): launch.validate_cpu_gate(receipt, launch.file_sha(receipt), "abc")


@pytest.mark.parametrize("name,gpus,limit", [
    ("run_vpfp_weak_dynamics_cpu_gate.slurm", "gpu:0", "00:20:00"),
    ("run_vpfp_weak_dynamics_diagnostic.slurm", "gpu:l40s:1", "06:00:00")])
def test_wrappers_are_scheduler_only_bounded_and_frozen(name, gpus, limit):
    path = ROOT / "scripts" / name
    text = path.read_text()
    subprocess.run(["bash", "-n", str(path)], check=True)
    assert "#SBATCH --gres=" + gpus in text
    assert "#SBATCH --time=" + limit in text
    assert "SLURM_TMPDIR" in text and "HEAD^{commit}" in text
    assert "--untracked-files=all -- src scripts configs tests legacy/vpfp" in text
    assert "scientific_admission\\tfalse" in text
    if gpus == "gpu:0":
        assert "VPFP_FINAL_REVISION" in text and "VPFP_FINAL_DIRTY" in text
    assert not re.search(r"^\s*(?:sbatch|scancel|srun)\b", text, re.M)
    for test in re.findall(r"tests/test_vpfp_[a-z_]+\.py", text):
        assert (ROOT / test).is_file(), test
