"""Synthetic launch mechanics: no scheduler, PDE solve, GPU, or real training."""
from __future__ import annotations

import importlib.util
import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/run_vpfp_regime_job.sh"
CHECKOUT = "${PDE_GAV_WORKTREES_ROOT}/vpfp-regime-main-debug"
SCRATCH = "${PDE_GAV_RUN_ROOT}"
ARMS = {"strong": ("value_base", "cov_base"),
        "weak": ("value_base", "cov_centered_iso", "cov_shrink_half", "cov_base")}
LABELS = {"strong": "vpfp-strong-main", "weak": "vpfp-weak-regime-debug"}
DRIVERS = {"strong": "scripts/run_vpfp_strong_main_experiment.py",
           "weak": "scripts/run_vpfp_weak_regime_debug.py"}

loader = importlib.util.spec_from_file_location("vpfp_regime_old_launch_mock", ROOT / "tests/test_vpfp_scaleup_launch.py")
old = importlib.util.module_from_spec(loader)
loader.loader.exec_module(old)
MOCK_RUNTIME = old.MOCK_RUNTIME.replace(
    'args[0] == "scripts/run_vpfp_scaleup_experiment.py"',
    'args[0] in ("scripts/run_vpfp_strong_main_experiment.py", "scripts/run_vpfp_weak_regime_debug.py")'
).replace(
    'for arm in ("value_base", option("--candidate")):',
    'for arm in (("value_base", "cov_base") if option("--regime") == "strong" else '
    '("value_base", "cov_centered_iso", "cov_shrink_half", "cov_base")):'
)


@pytest.fixture
def launch(tmp_path):
    checkout, scratch, commands = (tmp_path / name for name in ("checkout", "scratch", "commands"))
    for directory in (checkout / "configs", checkout / "scripts", scratch, commands, tmp_path / "local"):
        directory.mkdir(parents=True)
    for regime, job in (("weak", 5241246), ("strong", 5241247)):
        (checkout / f"configs/vpfp_unified_live_{regime}_collision.json").write_text("{}")
        source = scratch / f"vpfp-scaleup-{regime}-{job}"
        (source / "corpus").mkdir(parents=True)
        (source / "corpus/manifest.json").write_text("{}")
        (source / "result.json").write_text("{}")
    for script in (*DRIVERS.values(), "scripts/run_plasma_canonical_smoke.py"):
        (checkout / script).write_text("# Synthetic presence sentinel only.\n")
    runtime = commands / "runtime"
    runtime.write_text(f"#!{sys.executable}\n" + MOCK_RUNTIME)
    runtime.chmod(0o700)
    for command in ("python", "git", "module", "sha256sum", "sbatch", "srun", "scancel", "scontrol"):
        (commands / command).symlink_to(runtime)
    source = WRAPPER.read_text()
    for original, replacement in ((CHECKOUT, str(checkout)), (SCRATCH, str(scratch)),
                                  ("${PDE_GAV_PYTHON}", str(commands / "python"))):
        assert original in source
        source = source.replace(original, replacement)
    wrapper = checkout / "scripts/run_vpfp_regime_job.sh"
    wrapper.write_text(source)
    events_path = tmp_path / "mock-events.jsonl"
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("SLURM_", "MOCK_", "BASH_FUNC_")) and key != "BASH_ENV"}
    env.update(PATH=f"{commands}:/usr/bin:/bin", SLURM_JOB_ID="123456", SLURM_TMPDIR=str(tmp_path / "local"),
               MOCK_EVENTS=str(events_path))

    def events():
        return [json.loads(line) for line in events_path.read_text().splitlines()] if events_path.exists() else []

    def run(regime="strong", **overrides):
        selected = {key: value for key, value in {**env, **overrides}.items() if value is not None}
        result = subprocess.run(["/bin/bash", str(wrapper), regime], env=selected, cwd=tmp_path,
                                text=True, capture_output=True, timeout=15, check=False)
        assert not any(event["command"] in ("sbatch", "srun", "scancel", "scontrol") for event in events())
        return result

    def output(regime="strong"):
        return scratch / f"{LABELS[regime]}-123456"

    def receipt(regime="strong"):
        path = Path(str(output(regime)) + ".launch-exit-codes.tsv")
        return dict(line.split("\t") for line in path.read_text().splitlines()[1:])

    return SimpleNamespace(run=run, events=events, receipt=receipt, output=output, checkout=checkout)


def calls(launch, script):
    return [event["args"] for event in launch.events()
            if event["command"] == "python" and event["args"][0] == script]


def option(args, name):
    return args[args.index(name) + 1]


def driver_parser(regime):
    path = ROOT / DRIVERS[regime]
    tree = ast.parse(path.read_text(), filename=str(path))
    compile(tree, str(path), "exec")
    declarations = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "parser"]
    assert len(declarations) == 1
    namespace = {"argparse": argparse, "Path": Path, "__doc__": "Synthetic launcher CLI check"}
    exec(compile(ast.Module(body=declarations, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["parser"]()


@pytest.mark.parametrize("regime", ["strong", "weak"])
def test_exact_budgets_and_every_seed_zero_native_smoke(launch, regime):
    result = launch.run(regime)
    assert result.returncode == 0, result.stderr
    assert "using the existing explicit venv" in result.stderr
    experiments = calls(launch, DRIVERS[regime])
    assert len(experiments) == 1
    args = experiments[0]
    expected = {"--regime": regime, "--replicas": "2", "--batch-size": "4",
                "--generated-answers": "128", "--sample-steps": "64", "--device": "cuda"}
    if regime == "strong":
        expected.update({"--train-answers": "2048", "--heldout-answers": "128", "--train-seconds": "7200",
                         "--steps": "1000000", "--collect-batch-size": "8", "--seed": "1120260905"})
        assert option(args, "--source-corpus").endswith("vpfp-scaleup-strong-5241247/corpus")
        thresholds = ["1800", "3600"]
    else:
        expected.update({"--train-seconds": "1800", "--steps": "100000", "--seed": "1320260905"})
        assert option(args, "--source").endswith("vpfp-scaleup-weak-5241246")
        thresholds = ["360", "900"]
    assert all(option(args, key) == value for key, value in expected.items())
    index = args.index("--checkpoint-seconds")
    assert args[index + 1:index + 3] == thresholds
    parsed = driver_parser(regime).parse_args(args[1:])
    assert parsed.regime == regime and parsed.replicas == 2
    assert parsed.width == 256 and parsed.depth == 4 and parsed.learning_rate == 2.e-4
    assert parsed.permutations == 99 and parsed.bootstrap_repeats == 49
    assert parsed.covariance_floor == .05
    smokes = calls(launch, "scripts/run_plasma_canonical_smoke.py")
    assert len(smokes) == len(ARMS[regime])
    assert {Path(option(args, "--diffusion-checkpoint")).parent.parent.name for args in smokes} == set(ARMS[regime])
    assert set(launch.receipt(regime).values()) == {"0"}


@pytest.mark.parametrize("overrides", [{"SLURM_JOB_ID": None}, {"SLURM_JOB_ID": "bad"},
                                       {"SLURM_TMPDIR": None}, {"SLURM_TMPDIR": ""}])
def test_allocation_guard_precedes_external_commands(launch, overrides):
    result = launch.run(**overrides)
    assert result.returncode == 2 and not launch.events()


def test_unknown_regime_is_not_an_implicit_default(launch):
    assert launch.run("other").returncode == 2
    assert not launch.events()


def test_dirty_code_blocks_launch_before_python(launch):
    result = launch.run(MOCK_DIRTY=" M src/changed.py\n")
    assert result.returncode == 2 and "Commit/freeze" in result.stderr
    assert [event["command"] for event in launch.events()] == ["git"]


@pytest.mark.parametrize("regime", ["strong", "weak"])
def test_gpu_failure_never_reaches_training_or_smoke(launch, regime):
    result = launch.run(regime, MOCK_GPU_EXIT="17")
    assert result.returncode == 17
    assert not calls(launch, DRIVERS[regime]) and not calls(launch, "scripts/run_plasma_canonical_smoke.py")
    receipt = launch.receipt(regime)
    assert receipt["gpu_preflight"] == "17" and receipt["experiment"] == "not_run"
    assert all(receipt[f"smoke_{arm}"] == "gpu_preflight_failed" for arm in ARMS[regime])


@pytest.mark.parametrize("regime", ["strong", "weak"])
def test_training_failure_preserves_partial_artifacts_and_exit(launch, regime):
    result = launch.run(regime, MOCK_EXPERIMENT_EXIT="23")
    assert result.returncode == 23
    assert (launch.output(regime) / "mock-partial-artifact.txt").exists()
    assert launch.receipt(regime)["experiment"] == "23"
    assert launch.receipt(regime)["launch"] == "23"


@pytest.mark.parametrize("failure,record", [("MOCK_HASH_FAIL_ARM", "29"),
                                           ("MOCK_SMOKE_FAIL_ARM", "37"),
                                           ("MOCK_MISSING_CHECKPOINTS", "checkpoint_missing")])
@pytest.mark.parametrize("regime,arm", [("strong", "cov_base"), ("weak", "cov_centered_iso"),
                                       ("weak", "cov_shrink_half")])
def test_each_arm_failure_prevents_success(launch, failure, record, regime, arm):
    result = launch.run(regime, **{failure: arm})
    assert result.returncode == 1
    assert launch.receipt(regime)[f"smoke_{arm}"] == record


@pytest.mark.parametrize("regime", ["strong", "weak"])
@pytest.mark.parametrize("suffix", ["", ".launch-exit-codes.tsv", ".gpu-preflight.log"])
def test_existing_artifact_is_not_overwritten(launch, regime, suffix):
    path = Path(str(launch.output(regime)) + suffix)
    path.write_text("preserve")
    assert launch.run(regime).returncode == 2
    assert path.read_text() == "preserve" and not launch.events()


@pytest.mark.parametrize("script,regime,time", [("run_vpfp_strong_main.slurm", "strong", "16:00:00"),
                                              ("run_vpfp_weak_regime_debug.slurm", "weak", "06:00:00")])
def test_slurm_resource_caps_and_exact_entrypoint(script, regime, time):
    source = (ROOT / "scripts" / script).read_text()
    for line in ("#SBATCH --gres=gpu:l40s:1", "#SBATCH --cpus-per-task=8", "#SBATCH --mem=48G",
                 f"#SBATCH --time={time}"):
        assert line in source
    assert f"exec bash {CHECKOUT}/scripts/run_vpfp_regime_job.sh {regime}" in source
