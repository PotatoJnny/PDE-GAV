"""Synthetic safety checks; no scheduler, GPU, real training, or PDE solve."""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECKOUT = "${PDE_GAV_WORKTREES_ROOT}/vpfp-weak-noise-device-fix"
SCRATCH = "${PDE_GAV_RUN_ROOT}"
ARMS = {"noise": ("cov_energy_iso", "cov_unit_shape"),
        "structure": ("cov_shrink_half", "branch_identity_control", "lowmode_coherent")}
LABELS = {"noise": "vpfp-weak-noise-factorial", "structure": "vpfp-weak-structure-prior"}
DRIVERS = {"noise": "scripts/run_vpfp_weak_noise_factorial.py", "structure": "scripts/run_vpfp_weak_structure_prior.py"}
loader = importlib.util.spec_from_file_location("weak_prior_mock_runtime", ROOT / "tests/test_vpfp_scaleup_launch.py")
old = importlib.util.module_from_spec(loader)
loader.loader.exec_module(old)
MOCK_RUNTIME = old.MOCK_RUNTIME.replace(
    'args[0] == "scripts/run_vpfp_scaleup_experiment.py"',
    'args[0] in ("scripts/run_vpfp_weak_noise_factorial.py", "scripts/run_vpfp_weak_structure_prior.py")'
).replace('for arm in ("value_base", option("--candidate")):',
    'for arm in (("cov_energy_iso", "cov_unit_shape") if "factorial" in args[0] else '
    '("cov_shrink_half", "branch_identity_control", "lowmode_coherent")):')


@pytest.fixture
def launch(tmp_path):
    checkout, scratch, commands = (tmp_path / name for name in ("checkout", "scratch", "commands"))
    for p in (checkout / "configs", checkout / "scripts", scratch, commands, tmp_path / "local"):
        p.mkdir(parents=True)
    (checkout / "configs/vpfp_unified_live_weak_collision.json").write_text("{}")
    source = scratch / "vpfp-scaleup-weak-5241246"
    (source / "corpus").mkdir(parents=True)
    (source / "corpus/manifest.json").write_text("{}")
    (source / "result.json").write_text("{}")
    anchors = scratch / "vpfp-weak-regime-debug-5244224"
    anchors.mkdir()
    for name in ("result.json", "frozen-checkpoints.json"):
        (anchors / name).write_text("{}")
    for p in (*DRIVERS.values(), "scripts/run_plasma_canonical_smoke.py"):
        (checkout / p).write_text("# synthetic presence sentinel")
    runtime = commands / "runtime"
    runtime.write_text(f"#!{sys.executable}\n" + MOCK_RUNTIME)
    runtime.chmod(0o700)
    for name in ("python", "git", "module", "sha256sum", "sbatch", "srun", "scancel", "scontrol"):
        (commands / name).symlink_to(runtime)
    source = (ROOT / "scripts/run_vpfp_weak_prior_job.sh").read_text()
    for original, replacement in ((CHECKOUT, str(checkout)), (SCRATCH, str(scratch)),
                                  ("${PDE_GAV_PYTHON}", str(commands / "python"))):
        assert original in source
        source = source.replace(original, replacement)
    wrapper = checkout / "scripts/run_vpfp_weak_prior_job.sh"
    wrapper.write_text(source)
    event_path = tmp_path / "events.jsonl"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "MOCK_", "BASH_FUNC_")) and k != "BASH_ENV"}
    env.update(PATH=f"{commands}:/usr/bin:/bin", SLURM_JOB_ID="123456",
        SLURM_TMPDIR=str(tmp_path / "local"), MOCK_EVENTS=str(event_path))
    def events():
        return [json.loads(s) for s in event_path.read_text().splitlines()] if event_path.exists() else []
    def run(kind="structure", **overrides):
        selected = {k:v for k,v in {**env, **overrides}.items() if v is not None}
        result = subprocess.run(["/bin/bash", str(wrapper), kind], env=selected, cwd=tmp_path,
            text=True, capture_output=True, timeout=15, check=False)
        assert not any(e["command"] in ("sbatch", "srun", "scancel", "scontrol") for e in events())
        return result
    def output(kind="structure"):
        return scratch / f"{LABELS[kind]}-123456"
    def receipt(kind="structure"):
        return dict(s.split("\t") for s in Path(str(output(kind))+".launch-exit-codes.tsv").read_text().splitlines()[1:])
    return SimpleNamespace(run=run, output=output, receipt=receipt, events=events, anchors=anchors)


def calls(launch, script):
    return [e["args"] for e in launch.events() if e["command"] == "python" and e["args"][0] == script]


@pytest.mark.parametrize("kind", ARMS)
def test_fixed_budget_and_all_new_seed_zero_native_smokes(launch, kind):
    result = launch.run(kind)
    assert result.returncode == 0, result.stderr
    assert "using the existing explicit venv" in result.stderr
    rows = calls(launch, DRIVERS[kind])
    assert len(rows) == 1
    args = rows[0]
    expected = {"--regime":"weak", "--seed":"1320260905", "--replicas":"2", "--train-seconds":"1800",
        "--batch-size":"4", "--generated-answers":"128", "--sample-steps":"64", "--steps":"100000", "--device":"cuda"}
    assert all(args[args.index(k)+1] == v for k,v in expected.items())
    assert args[args.index("--checkpoint-seconds")+1:args.index("--checkpoint-seconds")+3] == ["360","900"]
    assert ("--anchors" in args) == (kind == "noise")
    smokes = calls(launch, "scripts/run_plasma_canonical_smoke.py")
    assert len(smokes) == len(ARMS[kind])
    assert {Path(s[s.index("--diffusion-checkpoint")+1]).parent.parent.name for s in smokes} == set(ARMS[kind])
    assert set(launch.receipt(kind).values()) == {"0"}


@pytest.mark.parametrize("env", [{"SLURM_JOB_ID":None}, {"SLURM_JOB_ID":"bad"}, {"SLURM_TMPDIR":None}])
def test_allocation_guard(launch, env):
    assert launch.run(**env).returncode == 2
    assert not launch.events()


def test_dirty_unknown_and_missing_anchor_guards(launch):
    assert launch.run("other").returncode == 2
    assert not launch.events()
    assert launch.run(MOCK_DIRTY=" M src/change.py\n").returncode == 2
    assert not calls(launch, DRIVERS["structure"])
    (launch.anchors / "result.json").unlink()
    assert launch.run("noise").returncode == 2
    assert not calls(launch, DRIVERS["noise"])


@pytest.mark.parametrize("kind", ARMS)
def test_gpu_and_training_failures_are_preserved(launch, kind):
    assert launch.run(kind, MOCK_GPU_EXIT="17").returncode == 17
    receipt = launch.receipt(kind)
    assert receipt["experiment"] == "not_run"
    assert all(receipt[f"smoke_{arm}"] == "gpu_preflight_failed" for arm in ARMS[kind])
    assert not calls(launch, DRIVERS[kind])


@pytest.mark.parametrize("kind", ARMS)
def test_training_failure_keeps_partial_results(launch, kind):
    assert launch.run(kind, MOCK_EXPERIMENT_EXIT="23").returncode == 23
    assert (launch.output(kind) / "mock-partial-artifact.txt").exists()
    assert launch.receipt(kind)["launch"] == "23"


@pytest.mark.parametrize("kind,arm", [(k,a) for k,arms in ARMS.items() for a in arms])
@pytest.mark.parametrize("env,code", [("MOCK_HASH_FAIL_ARM","29"), ("MOCK_SMOKE_FAIL_ARM","37"),
                                    ("MOCK_MISSING_CHECKPOINTS","checkpoint_missing")])
def test_each_smoke_failure_prevents_success(launch, kind, arm, env, code):
    assert launch.run(kind, **{env:arm}).returncode == 1
    assert launch.receipt(kind)[f"smoke_{arm}"] == code


@pytest.mark.parametrize("kind", ARMS)
@pytest.mark.parametrize("suffix", ["", ".launch-exit-codes.tsv", ".gpu-preflight.log"])
def test_no_overwrite(launch, kind, suffix):
    p = Path(str(launch.output(kind))+suffix)
    p.write_text("preserve")
    assert launch.run(kind).returncode == 2
    assert p.read_text() == "preserve" and not launch.events()


@pytest.mark.parametrize("kind,name,hours", [("noise","noise_factorial","04"), ("structure","structure_prior","05")])
def test_resource_caps_and_isolated_source(kind, name, hours):
    text = (ROOT / f"scripts/run_vpfp_weak_{name}.slurm").read_text()
    for line in ("#SBATCH --gres=gpu:l40s:1", "#SBATCH --cpus-per-task=8", "#SBATCH --mem=48G",
                 f"#SBATCH --time={hours}:00:00"):
        assert line in text
    assert f"exec bash {CHECKOUT}/scripts/run_vpfp_weak_prior_job.sh {kind}" in text
