"""Static launcher scope checks; never contact Slurm."""
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("phase",["cpu","strong","weak","velocity_pilot","rho_continuation"])
def test_launchers_are_bash_and_no_child_submissions(phase):
    path = ROOT/"scripts"/f"run_vpfp_dynamics_next_{phase}.slurm"
    subprocess.run(["bash","-n",str(path)],check=True)
    text = path.read_text()
    assert "#SBATCH --cpus-per-task=8" in text
    assert ("#SBATCH --gres=gpu:0" if phase=="cpu" else "#SBATCH --gres=gpu:l40s:1") in text
    assert "sbatch " not in text and "scancel " not in text and "--requeue" not in text
    assert "run_vpfp_dynamics_next_job.sh "+phase in text


def test_receipts_and_no_automatic_physics_qualification():
    path = ROOT/"scripts/run_vpfp_dynamics_next_job.sh"
    subprocess.run(["bash","-n",str(path)],check=True)
    text = path.read_text()
    assert "VPFP_NEXT_CPU_RECEIPT" in text and "VPFP_NEXT_CPU_SHA256" in text
    assert 'd["source_revision"]==sys.argv[3]' in text
    assert "scientific_admission" in text
    assert "tests/test_vpfp_training_resume.py" in text
    assert "tests/test_vpfp_velocity_dynamics_pilot.py" in text
    assert "scripts/run_vpfp_velocity_raw_reference.py" in text
    assert "scripts/run_vpfp_weak_block_diagnostic.py" in text
    assert 'source "${PDE_GAV_VENV}/bin/activate"' in text
    assert "module load" in text and "scipy-stack pytest" in text
    assert "set +eu\nset +o pipefail\nsource /etc/profile" in text
    assert "VPFP_PROFILE_STATUS=$?\nset -euo pipefail" in text
    assert "pip install" not in text and "scancel " not in text
