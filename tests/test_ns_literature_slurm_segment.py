"""Exercise safe continuation without model calls, GPUs, or real Slurm submission."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def segment(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("literature_segment_test", scripts / "run_ns_literature_slurm_segment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for key in tuple(module.os.environ):
        if key.startswith("NS_CONTINUATION_"):
            monkeypatch.delenv(key)
    for key, value in {
        "SLURM_CLUSTER_NAME": "cluster-a", "NS_LITERATURE_EXPECTED_CLUSTER": "cluster-a", "SLURM_JOB_ID": "100",
        "SLURM_CPUS_PER_TASK": "16", "CUDA_VISIBLE_DEVICES": "GPU-first",
        "SLURM_TMPDIR": str(tmp_path / "scratch"),
        "NS_LITERATURE_SEGMENT_STARTED_UNIX": "1000",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(module.time, "time", lambda: 1060)
    config = tmp_path / "config with spaces.json"
    config.write_text('{"method":"stub"}')
    output = tmp_path / "run"
    calls = []

    def sbatch(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="200;cluster-a\n", stderr="")

    monkeypatch.setattr(module.continuation.subprocess, "run", sbatch)
    return module, config, output, calls


def paused(output):
    def call(arguments):
        assert arguments[arguments.index("--output") + 1] == str(output)
        assert arguments[arguments.index("--segment-seconds") + 1] == "10740"
        checkpoint = output / "baseline_checkpoints/checkpoint.json"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text('{"used_seconds":7200,"acquired_cases":17}')
        return 75
    return call


def complete(output):
    def call(arguments):
        output.mkdir(parents=True, exist_ok=True)
        (output / "baseline_result.json").write_text('{"status":"completed"}')
        return 0
    return call


def test_safe_pause_rebind_resume_and_completion(segment, monkeypatch):
    module, config, output, calls = segment
    assert module.run_segment(config, output, baseline_main=paused(output)) == 0
    assert len(calls) == 1
    command, options = calls[0]
    for flag in ("--partition=gpubase_bygpu_b1", "--cpus-per-task=16", "--mem=64G",
                 "--gres=gpu:l40s:1", "--time=03:00:00", "--dependency=afterok:100"):
        assert flag in command
    assert command[-3:] == [str(config), str(output), "12"]
    assert command[-4] == module.sys.executable  # Retain the virtualenv interpreter path.
    for key, value in options["env"].items():
        if key.startswith("NS_CONTINUATION_"):
            monkeypatch.setenv(key, value)
    monkeypatch.setenv("SLURM_JOB_ID", "200")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-second")

    def resumed(arguments):
        assert "--resume" in arguments
        assert arguments[arguments.index("--gpu-token") + 1] == "GPU-second"
        checkpoint = json.loads((output / "baseline_checkpoints/checkpoint.json").read_text())
        assert checkpoint == {"used_seconds": 7200, "acquired_cases": 17}
        return complete(output)(arguments)

    assert module.run_segment(config, output, baseline_main=resumed) == 0
    assert len(calls) == 1
    state = module.continuation.read_state(output.parent / ".run.slurm")
    assert state["status"] == "completed"
    assert [s["status"] for s in state["segments"]] == ["paused", "completed"]


@pytest.mark.parametrize("target", ["config", "checkpoint"])
def test_changed_scientific_input_blocks_resume(segment, monkeypatch, target):
    module, config, output, calls = segment
    module.run_segment(config, output, baseline_main=paused(output))
    for key, value in calls[0][1]["env"].items():
        if key.startswith("NS_CONTINUATION_"):
            monkeypatch.setenv(key, value)
    monkeypatch.setenv("SLURM_JOB_ID", "200")
    changed = config if target == "config" else output / "baseline_checkpoints/checkpoint.json"
    changed.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="differs from the paused"):
        module.run_segment(config, output, baseline_main=lambda _: pytest.fail("must not run"))
    assert len(calls) == 1
    assert module.continuation.read_state(output.parent / ".run.slurm")["status"] == "failed"


@pytest.mark.parametrize("exit_code", [0, 1, 75])
def test_no_successor_on_failure_or_missing_result(segment, exit_code):
    module, config, output, calls = segment
    with pytest.raises((FileNotFoundError, RuntimeError)):
        module.run_segment(config, output, baseline_main=lambda _: exit_code)
    assert calls == []
    assert module.continuation.read_state(output.parent / ".run.slurm")["status"] == "failed"


def test_completion_never_submits(segment):
    module, config, output, calls = segment
    module.run_segment(config, output, baseline_main=complete(output))
    assert calls == []


def test_segment_limit_preserves_pause_without_submission(segment):
    module, config, output, calls = segment
    with pytest.raises(module.continuation.SegmentLimitError):
        module.run_segment(config, output, max_segments=1, baseline_main=paused(output))
    assert calls == []
    assert module.continuation.read_state(output.parent / ".run.slurm")["status"] == "paused"


def test_old_output_is_not_restarted(segment):
    module, config, output, calls = segment
    output.mkdir()
    prior = output / "baseline_result.json"
    prior.write_text('{"status":"completed","old":true}')
    before = prior.read_bytes()
    with pytest.raises(FileExistsError, match="new baseline output"):
        module.run_segment(config, output, baseline_main=lambda _: pytest.fail("must not run"))
    assert prior.read_bytes() == before
    assert calls == []


@pytest.mark.parametrize("key,value", [
    ("SLURM_CLUSTER_NAME", "cluster-b"), ("SLURM_CPUS_PER_TASK", "8"),
    ("CUDA_VISIBLE_DEVICES", "0,1"), ("NS_CONTINUATION_RUN_ROOT", "/tmp/other-run"),
])
def test_wrong_allocation_or_cross_run_environment(segment, monkeypatch, key, value):
    module, config, output, calls = segment
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError):
        module.run_segment(config, output, baseline_main=lambda _: pytest.fail("must not run"))
    assert calls == []
