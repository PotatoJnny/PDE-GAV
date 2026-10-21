"""Static job boundaries; no scheduler submission or real numeric audit."""
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def test_cpu_analysis_is_zero_gpu_same_frozen_source_and_actual_figures():
    path=ROOT/"scripts/run_vpfp_strong_audit_analysis_cpu.slurm"
    body=path.read_text()
    subprocess.run(["bash","-n",str(path)],check=True)
    assert "#SBATCH --gres=gpu:0" in body and "#SBATCH --time=00:20:00" in body
    assert "scripts/summarize_vpfp_strong_audit.py" in body
    assert "6af6ea4418a9ec5eadd6dbc52209e1e5380c4505013bceabe04135551074b5f4" in body
    assert "Source changed during tests/summary" in body
    assert "CUDA_VISIBLE_DEVICES=\"\"" in body
    for name in re.findall(r"tests/test_vpfp_[a-z_]+\.py",body):
        assert (ROOT/name).is_file(),name


def test_independent_audit_is_bounded_and_requires_successful_gate():
    path=ROOT/"scripts/run_vpfp_strong_initial_layer_audit.slurm"
    body=path.read_text()
    subprocess.run(["bash","-n",str(path)],check=True)
    assert "#SBATCH --gres=gpu:l40s:1" in body and "#SBATCH --time=03:00:00" in body
    assert "TimeLimit=03:00:00" in body and "torch.cuda.device_count()==1" in body
    assert 'r["source_revision"]==sys.argv[3]' in body
    assert 'r["mechanics_tests"]=="0"' in body and 'r["launch"]=="0"' in body
    assert 'r["completed_audit_figures"]=="0"' in body
    assert "tests/test_vpfp_independent_physics.py" in body
    assert "10800-(SECONDS-VPFP_STARTED_SECONDS)-45" in body
    assert "--wall-seconds" in body and "timeout --signal=TERM --kill-after=15s" in body
    assert "prior_audit_preflight_gpu_seconds\\t5076" in body
    assert "scientific_admission\\tfalse" in body and "new_training\\tfalse" in body
    assert "Source changed during independent audit" in body
    assert not re.search(r"^\s*(?:sbatch|scancel|srun)\b",body,re.M)
