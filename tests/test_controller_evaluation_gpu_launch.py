"""Controller borrowing respects the allocation visible before CUDA startup."""
from __future__ import annotations

import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import pytest

from pde_agent_verifier_solver.engine.controller_gpu_visibility import (
    configure_controller_evaluation_devices,
)


@pytest.fixture(autouse=True)
def uninitialized_cuda(monkeypatch):
    # These launch tests never import Torch, create a CUDA context, or use a GPU.
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False))
    )


def test_reorders_only_already_visible_devices_and_preserves_other_environment():
    environment = {"CUDA_VISIBLE_DEVICES": "5,9,3", "OMP_NUM_THREADS": "1"}

    devices = configure_controller_evaluation_devices("3", ("5", "9"), environment=environment)

    assert environment == {"CUDA_VISIBLE_DEVICES": "3,5,9", "OMP_NUM_THREADS": "1"}
    assert devices == {"3": "cuda:0", "5": "cuda:1", "9": "cuda:2"}


@pytest.mark.parametrize("worker_count", (1, 2, 4))
def test_maps_each_configured_agent_without_assuming_two_agents(worker_count):
    workers = tuple(f"GPU-worker-{index}" for index in range(worker_count))
    environment = {"CUDA_VISIBLE_DEVICES": ",".join((*workers, "GPU-controller"))}

    devices = configure_controller_evaluation_devices("GPU-controller", workers, environment=environment)

    assert environment["CUDA_VISIBLE_DEVICES"] == ",".join(("GPU-controller", *workers))
    assert devices == {token: f"cuda:{index}" for index, token in enumerate(("GPU-controller", *workers))}


def test_gpu_uuid_and_mig_tokens_remain_opaque():
    controller = "GPU-c0123456-1111-2222-3333-444444444444"
    workers = ("MIG-GPU-a0123456-1111-2222-3333-444444444444/1/0", "MIG-b0123456-1111-2222-3333-444444444444")
    environment = {"CUDA_VISIBLE_DEVICES": ",".join((*workers, controller))}

    devices = configure_controller_evaluation_devices(controller, workers, environment=environment)

    assert devices == {controller: "cuda:0", workers[0]: "cuda:1", workers[1]: "cuda:2"}


def test_legacy_controller_only_visibility_does_not_expose_worker_gpus():
    environment = {"CUDA_VISIBLE_DEVICES": "3", "OMP_NUM_THREADS": "1"}

    devices = configure_controller_evaluation_devices("3", ("5", "9"), environment=environment)

    assert devices == {}
    assert environment == {"CUDA_VISIBLE_DEVICES": "3", "OMP_NUM_THREADS": "1"}


@pytest.mark.parametrize("visible", (None, "", "-1", "5,9", "3,5", "3,7", "7"))
def test_missing_allocation_is_rejected_without_expanding_visibility(visible):
    environment = {} if visible is None else {"CUDA_VISIBLE_DEVICES": visible}
    before = dict(environment)

    with pytest.raises(ValueError):
        configure_controller_evaluation_devices("3", ("5", "9"), environment=environment)

    assert environment == before


@pytest.mark.parametrize(
    ("controller", "workers", "visible"),
    (
        ("", ("5", "9"), "3,5,9"),
        ("   ", ("5", "9"), "3,5,9"),
        ("3", ("", "9"), "3,5,9"),
        ("3", ("5", "   "), "3,5,9"),
        ("3", ("5", "5"), "3,5,9"),
        ("3", ("3", "9"), "3,5,9"),
        ("3,5", ("9",), "3,5,9"),
        ("3", ("5,9",), "3,5,9"),
        ("3", ("5", "9"), "3,5,5,9"),
        ("3", ("5", "9"), "3,,5,9"),
        ("3", ("5", "9"), "3,5,9,"),
    ),
)
def test_duplicate_empty_and_compound_tokens_fail_before_environment_mutation(controller, workers, visible):
    environment = {"CUDA_VISIBLE_DEVICES": visible}

    with pytest.raises(ValueError):
        configure_controller_evaluation_devices(controller, workers, environment=environment)

    assert environment == {"CUDA_VISIBLE_DEVICES": visible}


def test_reordering_after_cuda_initialization_is_rejected(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: True)))
    environment = {"CUDA_VISIBLE_DEVICES": "5,9,3"}

    with pytest.raises(RuntimeError):
        configure_controller_evaluation_devices("3", ("5", "9"), environment=environment)

    assert environment == {"CUDA_VISIBLE_DEVICES": "5,9,3"}


def test_existing_controller_first_order_can_be_used_after_cuda_initialization(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: True)))
    environment = {"CUDA_VISIBLE_DEVICES": "3,5,9"}

    devices = configure_controller_evaluation_devices("3", ("5", "9"), environment=environment)

    assert devices == {"3": "cuda:0", "5": "cuda:1", "9": "cuda:2"}
    assert environment["CUDA_VISIBLE_DEVICES"] == "3,5,9"


def test_legacy_single_gpu_remains_unchanged_after_cuda_initialization(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: True)))
    environment = {"CUDA_VISIBLE_DEVICES": "3"}

    assert configure_controller_evaluation_devices("3", ("5", "9"), environment=environment) == {}
    assert environment["CUDA_VISIBLE_DEVICES"] == "3"


def test_torch_absent_is_not_imported_by_visibility_configuration(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    environment = {"CUDA_VISIBLE_DEVICES": "5,9,3"}

    devices = configure_controller_evaluation_devices("3", ("5", "9"), environment=environment)

    assert devices["3"] == "cuda:0"
    assert "torch" not in sys.modules


def test_default_environment_updates_only_visible_device_order(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5,9,3")
    monkeypatch.setenv("PDE_GPU_LAUNCH_TEST_SENTINEL", "unchanged")

    devices = configure_controller_evaluation_devices("3", ("5", "9"))

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3,5,9"
    assert os.environ["PDE_GPU_LAUNCH_TEST_SENTINEL"] == "unchanged"
    assert devices["9"] == "cuda:2"


def test_worker_tokens_cannot_be_passed_as_a_bare_string():
    environment = {"CUDA_VISIBLE_DEVICES": "3,5,9"}

    with pytest.raises(TypeError):
        configure_controller_evaluation_devices("3", "59", environment=environment)

    assert environment["CUDA_VISIBLE_DEVICES"] == "3,5,9"


def test_extra_visible_devices_are_not_added_to_the_configured_borrowing_pool():
    environment = {"CUDA_VISIBLE_DEVICES": "5,9,3,11"}

    devices = configure_controller_evaluation_devices("3", ("5", "9"), environment=environment)

    assert devices == {"3": "cuda:0", "5": "cuda:1", "9": "cuda:2"}
    assert environment["CUDA_VISIBLE_DEVICES"] == "3,5,9"


@pytest.mark.parametrize(
    ("script", "expected_expression"),
    (
        ("slurm_ns_draft_first_batch.sh", "${JOB_GPU_IDS[2]},${JOB_GPU_IDS[0]},${JOB_GPU_IDS[1]}"),
        ("slurm_ns_canonical_full_5round.sh", "${PDE_CONTROLLER_GPU_DEVICE},${PDE_AGENT_WORKER_GPU_DEVICES},${PDE_CANDIDATE_GPU_DEVICE}"),
        ("slurm_vpfp_strong_formal_5round_20260909.sh", "$PDE_CONTROLLER_GPU_DEVICE,$PDE_AGENT_WORKER_GPU_DEVICES"),
    ),
)
def test_formal_shell_entrypoints_expose_controller_before_allocated_agent_gpus(script, expected_expression):
    # Inspect declarations only: never source a Slurm script or submit a job.
    source = (Path(__file__).resolve().parents[1] / "scripts" / script).read_text()
    assignments = re.findall(r'^\s*export CUDA_VISIBLE_DEVICES="([^"]*)"\s*$', source, re.MULTILINE)

    assert expected_expression in assignments


@pytest.mark.parametrize("mode,expected", [("fixed", "5,9,3"), ("bon", ""), ("aide", "")])
def test_fixed_and_search_modes_preserve_their_current_controller_visibility(mode, expected):
    import subprocess
    source = (Path(__file__).resolve().parents[1] / "scripts/slurm_ns_draft_first_batch.sh").read_text()
    # Execute only the environment-selection block; never source or submit the launcher.
    start = source.index('if [[ "${MODE}" == fixed ]]')
    end = source.index("\nfi", start) + len("\nfi")
    selected = source[start:end]
    environment = {**os.environ, "MODE": mode, "NS_JOB_GPU_TOKENS": "5,9,3"}
    result = subprocess.run(["bash", "-c", selected + '\nprintf "%s" "$CUDA_VISIBLE_DEVICES"'],
                            env=environment, text=True, capture_output=True, check=True)
    assert result.stdout == expected
