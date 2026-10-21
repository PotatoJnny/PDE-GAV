"""Static CPU-only launcher check; no scheduler operation."""
from pathlib import Path
import re
import subprocess


def test_cpu_gate_stays_on_compute_without_claiming_GPU_or_confirmation():
    root = Path(__file__).resolve().parents[1]
    path = root/"scripts/run_vpfp_independent_cpu_gate.slurm"
    text = path.read_text()
    subprocess.run(["bash", "-n", str(path)],check=True)
    assert "#SBATCH --gres=gpu:0" in text and "#SBATCH --cpus-per-task=8" in text
    assert "#SBATCH --time=00:20:00" in text and "#SBATCH --mem=32G" in text
    assert 'export CUDA_VISIBLE_DEVICES=""' in text
    assert "SLURM_TMPDIR" in text and "HEAD^{commit}" in text
    assert "--untracked-files=all -- src scripts configs tests legacy/vpfp" in text
    assert "timeout --signal=TERM --kill-after=15s 1100" in text
    assert "CUDA_mechanics_completed\\tfalse" in text
    assert not re.search(r"^\s*(?:sbatch|srun|scancel|scontrol)\b",text,re.M)
    assert "scripts/run_vpfp_independent_confirmation.py --selection" not in text
    for name in re.findall(r"tests/test_vpfp_[a-z_]+\.py",text):
        assert (root/name).is_file(), name
