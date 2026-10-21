"""Mock launch mechanics only: no real scheduler, GPU, training or selection."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/run_vpfp_scaleup_job.sh"
FROZEN_CHECKOUT = "${PDE_GAV_WORKTREES_ROOT}/vpfp-scaleup-experiment"
FROZEN_SCRATCH = "${PDE_GAV_RUN_ROOT}"

# This executable stands in for every potentially expensive/external command.
# Selection/config placeholders and checkpoint bytes exist only below tmp_path.
MOCK_RUNTIME = r'''
import hashlib
import json
import os
from pathlib import Path
import sys

name, args = Path(sys.argv[0]).name, sys.argv[1:]
event = {"command": name, "args": args}
with open(os.environ["MOCK_EVENTS"], "a") as stream:
    stream.write(json.dumps(event) + "\n")

def option(name):
    return args[args.index(name) + 1]

if name == "git":
    if args == ["status", "--porcelain", "--untracked-files=all", "--", "src", "scripts", "configs", "legacy/vpfp"]:
        print(os.environ.get("MOCK_DIRTY", ""), end="")
    elif args == ["rev-parse", "HEAD"]:
        print("a" * 40)
    else:
        raise SystemExit("unexpected git invocation")
elif name == "module":
    assert args == ["load", "scipy-stack", "pytest"]
    raise SystemExit(1)  # Exercise the documented explicit-venv fallback.
elif name == "sha256sum":
    assert args[0] == "--" and len(args) == 2
    checkpoint = Path(args[1])
    if checkpoint.parent.parent.name == os.environ.get("MOCK_HASH_FAIL_ARM"):
        raise SystemExit(29)
    print(hashlib.sha256(checkpoint.read_bytes()).hexdigest() + "  " + str(checkpoint))
elif name == "python":
    if args[0] == "-c" and "json.load" in args[1]:
        print(json.loads(Path(args[2]).read_text())["candidate"])
    elif args[0] == "-c" and "torch.cuda.is_available" in args[1]:
        print("MOCK CUDA preflight; no GPU queried")
        raise SystemExit(int(os.environ.get("MOCK_GPU_EXIT", "0")))
    elif args[0] == "scripts/run_vpfp_scaleup_experiment.py":
        output = Path(option("--output"))
        output.mkdir()
        (output / "mock-partial-artifact.txt").write_text("mock training was reached")
        missing = os.environ.get("MOCK_MISSING_CHECKPOINTS", "").split(",")
        for replica in (0, 1):
            for arm in ("value_base", option("--candidate")):
                if arm not in missing:
                    checkpoint = output / f"seed-{replica}" / arm / "checkpoints/best.pt"
                    checkpoint.parent.mkdir(parents=True)
                    checkpoint.write_bytes(f"mock checkpoint {replica} {arm}".encode())
        raise SystemExit(int(os.environ.get("MOCK_EXPERIMENT_EXIT", "0")))
    elif args[0] == "scripts/run_plasma_canonical_smoke.py":
        checkpoint = Path(option("--diffusion-checkpoint"))
        assert option("--diffusion-checkpoint-sha256") == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        assert option("--physics") == "vpfp" and option("--device") == "cuda"
        assert Path(option("--output")) == checkpoint.parent.parent / "canonical-campaign-smoke"
        assert checkpoint.parent.parent.parent.name == "seed-0"
        print("MOCK native smoke; no model loaded")
        if checkpoint.parent.parent.name == os.environ.get("MOCK_SMOKE_FAIL_ARM"):
            raise SystemExit(37)
    else:
        raise SystemExit("unexpected Python invocation")
else:
    raise SystemExit("real scheduler action forbidden by mock launch test")
'''


@pytest.fixture
def launch(tmp_path):
    checkout, scratch, commands = (tmp_path / name for name in ("checkout", "scratch", "commands"))
    for directory in (checkout / "configs", checkout / "scripts", scratch, commands, tmp_path / "local"):
        directory.mkdir(parents=True)
    for regime, old_job, pilot_job in (("weak", 5232589, 5240924), ("strong", 5232590, 5240925)):
        (checkout / f"configs/vpfp_unified_live_{regime}_collision.json").write_text("{}")
        (checkout / f"configs/vpfp_scaleup_selection_{regime}.json").write_text(json.dumps({"candidate": "cov_low", "mock_only": True}))
        cache = scratch / f"vpfp-sobolev-matching-{regime}-{old_job}/corpus"
        cache.mkdir(parents=True)
        (cache / "manifest.json").write_text("{}")
        pilot = scratch / f"vpfp-temporal-training-{regime}-{pilot_job}"
        pilot.mkdir()
        (pilot / "result.json").write_text("{}")
    for script in ("run_vpfp_scaleup_experiment.py", "run_plasma_canonical_smoke.py"):
        (checkout / "scripts" / script).write_text("# Mock-only presence sentinel; never executed.\n")
    runtime = commands / "runtime"
    runtime.write_text(f"#!{sys.executable}\n" + MOCK_RUNTIME)
    runtime.chmod(0o700)
    for command in ("python", "git", "module", "sha256sum", "sbatch", "srun", "scancel", "scontrol"):
        (commands / command).symlink_to(runtime)
    source = WRAPPER.read_text()
    for original, replacement in ((FROZEN_CHECKOUT, str(checkout)), (FROZEN_SCRATCH, str(scratch)),
                                  ("${PDE_GAV_PYTHON}", str(commands / "python"))):
        assert original in source, f"update mock path substitution for {original}"
        source = source.replace(original, replacement)
    wrapper = checkout / "scripts/run_vpfp_scaleup_job.sh"
    wrapper.write_text(source)
    events_path = tmp_path / "mock-events.jsonl"
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("SLURM_", "MOCK_", "BASH_FUNC_")) and key != "BASH_ENV"}
    env.update(PATH=f"{commands}:/usr/bin:/bin", SLURM_JOB_ID="123456", SLURM_TMPDIR=str(tmp_path / "local"),
               MOCK_EVENTS=str(events_path))

    def run(regime="weak", **overrides):
        selected = {**env, **overrides}
        selected = {key: value for key, value in selected.items() if value is not None}
        result = subprocess.run(["/bin/bash", str(wrapper), regime], env=selected, cwd=tmp_path,
                                text=True, capture_output=True, timeout=15, check=False)
        if events_path.exists():
            assert not any(json.loads(line)["command"] in ("sbatch", "srun", "scancel", "scontrol")
                           for line in events_path.read_text().splitlines())
        return result

    def events():
        return [json.loads(line) for line in events_path.read_text().splitlines()] if events_path.exists() else []

    def receipt(regime="weak"):
        path = scratch / f"vpfp-scaleup-{regime}-123456.launch-exit-codes.tsv"
        return dict(line.split("\t") for line in path.read_text().splitlines()[1:])

    def set_candidate(candidate, regime="weak"):
        (checkout / f"configs/vpfp_scaleup_selection_{regime}.json").write_text(
            json.dumps({"candidate": candidate, "mock_only": True}))

    return SimpleNamespace(run=run, events=events, receipt=receipt, set_candidate=set_candidate,
                           checkout=checkout, scratch=scratch,
                           output=scratch / "vpfp-scaleup-weak-123456")


def _calls(launch, script):
    return [event["args"] for event in launch.events()
            if event["command"] == "python" and event["args"][0] == script]


def _option(args, name):
    return args[args.index(name) + 1]


def _driver_parser():
    """Use the real argument declarations without importing training modules."""
    path = ROOT / "scripts/run_vpfp_scaleup_experiment.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    compile(tree, str(path), "exec")  # Also reject whole-driver syntax errors.
    declarations = [node for node in tree.body
                    if (isinstance(node, ast.FunctionDef) and node.name == "parser")
                    or (isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                        and target.id == "CANDIDATES" for target in node.targets))]
    assert len(declarations) == 2
    namespace = {"argparse": argparse, "Path": Path, "__doc__": "Mock launch CLI alignment"}
    exec(compile(ast.Module(body=declarations, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["parser"]()


@pytest.mark.parametrize("overrides", [{"SLURM_JOB_ID": None}, {"SLURM_JOB_ID": "not-an-allocation"},
                                       {"SLURM_TMPDIR": None}, {"SLURM_TMPDIR": ""}])
def test_allocation_guard_precedes_every_external_command(launch, overrides):
    result = launch.run(**overrides)
    assert result.returncode == 2
    assert "requires a Slurm compute allocation" in result.stderr
    assert not launch.events() and not launch.output.exists()


def test_unknown_regime_rejected_before_external_commands(launch):
    result = launch.run("other")
    assert result.returncode == 2 and "Unknown VPFP regime" in result.stderr
    assert not launch.events()


def test_missing_reviewed_selection_never_reaches_python(launch):
    (launch.checkout / "configs/vpfp_scaleup_selection_weak.json").unlink()
    result = launch.run()
    assert result.returncode == 2 and "reviewed selection" in result.stderr
    assert not launch.events() and not launch.output.exists()


def test_dirty_relevant_source_never_reaches_python(launch):
    result = launch.run(MOCK_DIRTY=" M configs/vpfp_scaleup_selection_weak.json\n")
    assert result.returncode == 2 and "Commit/freeze relevant source" in result.stderr
    assert [event["command"] for event in launch.events()] == ["git"]
    assert not launch.output.exists()


@pytest.mark.parametrize("candidate", ["value_base", "full_base", "value_time", "", "cov_low extra"])
def test_only_three_reviewed_candidate_names_are_accepted(launch, candidate):
    launch.set_candidate(candidate)
    result = launch.run()
    assert result.returncode == 2 and "Invalid reviewed VPFP candidate" in result.stderr
    assert not _calls(launch, "scripts/run_vpfp_scaleup_experiment.py")
    assert not any("torch.cuda" in str(event) for event in launch.events())


@pytest.mark.parametrize("regime,seed", [("weak", "920260905"), ("strong", "1020260905")])
@pytest.mark.parametrize("candidate", ["value_low", "cov_base", "cov_low"])
def test_success_has_exact_budget_and_two_seed_zero_native_smokes(launch, regime, seed, candidate):
    launch.set_candidate(candidate, regime)
    result = launch.run(regime)
    assert result.returncode == 0, result.stderr
    assert "using the existing explicit venv" in result.stderr
    calls = _calls(launch, "scripts/run_vpfp_scaleup_experiment.py")
    assert len(calls) == 1
    args = calls[0]
    for option, value in {"--replicas": "2", "--train-seconds": "1800", "--steps": "100000",
                          "--batch-size": "4", "--collect-batch-size": "8", "--generated-answers": "128",
                          "--sample-steps": "64", "--device": "cuda", "--regime": regime,
                          "--seed": seed, "--candidate": candidate}.items():
        assert _option(args, option) == value
    index = args.index("--checkpoint-seconds")
    assert args[index + 1:index + 3] == ["360", "900"]
    parsed = _driver_parser().parse_args(args[1:])
    assert parsed.replicas == 2 and parsed.train_seconds == 1800
    assert parsed.checkpoint_seconds == [360.0, 900.0]
    assert parsed.generated_answers == 128 and parsed.collect_batch_size == 8
    assert parsed.replicas * 2 * (1 + len(parsed.checkpoint_seconds)) * parsed.generated_answers == 1536
    assert parsed.width == 256 and parsed.depth == 4 and parsed.learning_rate == 2.e-4
    assert parsed.permutations == 99 and parsed.bootstrap_repeats == 49
    assert _option(args, "--config").endswith(f"vpfp_unified_live_{regime}_collision.json")
    assert _option(args, "--selection").endswith(f"vpfp_scaleup_selection_{regime}.json")
    assert f"vpfp-temporal-training-{regime}-" in _option(args, "--pilot")
    smokes = _calls(launch, "scripts/run_plasma_canonical_smoke.py")
    assert len(smokes) == 2
    assert [Path(_option(args, "--diffusion-checkpoint")).parent.parent.name for args in smokes] == ["value_base", candidate]
    for smoke in smokes:
        checkpoint = Path(_option(smoke, "--diffusion-checkpoint"))
        assert checkpoint.parent.parent.parent.name == "seed-0"
        assert _option(smoke, "--diffusion-checkpoint-sha256") == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        assert (checkpoint.parent.parent / "canonical-campaign-smoke.log").read_text().startswith("MOCK native smoke")
    assert launch.receipt(regime) == {"gpu_preflight": "0", "experiment": "0", "baseline_smoke": "0",
                                       "candidate_smoke": "0", "launch": "0"}


def test_real_driver_candidate_choices_match_wrapper():
    parser = _driver_parser()
    action = next(action for action in parser._actions if action.dest == "candidate")
    assert tuple(action.choices) == ("value_low", "cov_base", "cov_low")
    assert action.required is True


def test_gpu_failure_preserves_log_and_receipt_without_running_training(launch):
    result = launch.run(MOCK_GPU_EXIT="17")
    assert result.returncode == 17
    assert not _calls(launch, "scripts/run_vpfp_scaleup_experiment.py")
    assert not _calls(launch, "scripts/run_plasma_canonical_smoke.py")
    assert not launch.output.exists()
    assert Path(str(launch.output) + ".gpu-preflight.log").read_text().startswith("MOCK CUDA")
    assert launch.receipt() == {"gpu_preflight": "17", "experiment": "not_run",
                                "baseline_smoke": "gpu_preflight_failed", "candidate_smoke": "gpu_preflight_failed", "launch": "17"}


def test_experiment_failure_keeps_partial_artifact_runs_available_smokes_and_receipt(launch):
    result = launch.run(MOCK_EXPERIMENT_EXIT="23")
    assert result.returncode == 23
    assert (launch.output / "mock-partial-artifact.txt").exists()
    assert len(_calls(launch, "scripts/run_plasma_canonical_smoke.py")) == 2
    assert launch.receipt() == {"gpu_preflight": "0", "experiment": "23", "baseline_smoke": "0",
                                "candidate_smoke": "0", "launch": "23"}


@pytest.mark.parametrize("arm,receipt_key", [("value_base", "baseline_smoke"), ("cov_low", "candidate_smoke")])
def test_hash_failure_is_not_erased_by_set_e_and_other_smoke_still_runs(launch, arm, receipt_key):
    result = launch.run(MOCK_HASH_FAIL_ARM=arm)
    assert result.returncode == 1
    assert f"Cannot hash seed-0/{arm}" in result.stderr
    receipt = launch.receipt()
    assert receipt[receipt_key] == "29" and receipt["experiment"] == "0" and receipt["launch"] == "1"
    smokes = _calls(launch, "scripts/run_plasma_canonical_smoke.py")
    assert len(smokes) == 1
    assert Path(_option(smokes[0], "--diffusion-checkpoint")).parent.parent.name != arm
    assert (launch.output / "mock-partial-artifact.txt").exists()


@pytest.mark.parametrize("arm,receipt_key", [("value_base", "baseline_smoke"), ("cov_low", "candidate_smoke")])
def test_smoke_failure_is_preserved_without_skipping_other_arm(launch, arm, receipt_key):
    result = launch.run(MOCK_SMOKE_FAIL_ARM=arm)
    assert result.returncode == 1
    assert len(_calls(launch, "scripts/run_plasma_canonical_smoke.py")) == 2
    receipt = launch.receipt()
    assert receipt[receipt_key] == "37" and receipt["experiment"] == "0" and receipt["launch"] == "1"


def test_missing_checkpoints_cannot_turn_experiment_success_into_launch_success(launch):
    result = launch.run(MOCK_MISSING_CHECKPOINTS="value_base,cov_low")
    assert result.returncode == 1
    assert not _calls(launch, "scripts/run_plasma_canonical_smoke.py")
    assert launch.receipt() == {"gpu_preflight": "0", "experiment": "0", "baseline_smoke": "checkpoint_missing",
                                "candidate_smoke": "checkpoint_missing", "launch": "1"}


@pytest.mark.parametrize("suffix", ["", ".launch-exit-codes.tsv", ".gpu-preflight.log"])
@pytest.mark.parametrize("symlink", [False, True])
def test_existing_artifact_and_dangling_symlink_are_never_overwritten(launch, suffix, symlink):
    artifact = Path(str(launch.output) + suffix)
    if symlink:
        artifact.symlink_to(launch.scratch / "missing-target")
    else:
        artifact.write_text("preserve me")
    result = launch.run()
    assert result.returncode == 2 and "Refusing to overwrite" in result.stderr
    assert not launch.events()
    assert artifact.is_symlink() if symlink else artifact.read_text() == "preserve me"


@pytest.mark.parametrize("regime", ["weak", "strong"])
def test_slurm_resources_bound_the_four_training_runs_and_use_same_wrapper(regime):
    source = (ROOT / f"scripts/run_vpfp_scaleup_{regime}.slurm").read_text()
    directives = dict(re.findall(r"^#SBATCH --([^=\s]+)=(\S+)$", source, re.MULTILINE))
    assert directives["partition"] == "gpubase_l40s_b3"
    assert directives["gres"] == "gpu:l40s:1"
    assert directives["cpus-per-task"] == "8" and directives["mem"] == "48G"
    assert directives["time"] == "04:00:00"
    assert 2 * 2 * 1800 < 4 * 3600  # Headroom for collection, sampling, audit and smokes.
    assert directives["job-name"] == f"vpfp_scaleup_{regime}"
    assert "chdir" not in directives and "account" not in directives  # site options come from sbatch
    assert directives["output"] == "slurm-%x-%j.out"
    assert directives["error"] == "slurm-%x-%j.err"
    assert source.rstrip().endswith(f"exec bash {FROZEN_CHECKOUT}/scripts/run_vpfp_scaleup_job.sh {regime}")
