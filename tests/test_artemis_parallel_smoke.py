"""Synthetic bookkeeping tests, not real-asset or multi-GPU qualification."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("artemis_parallel_smoke_check",
    ROOT / "scripts/check_artemis_parallel_evaluation.py")
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def _pool(closed=False):
    return {"started": True, "closed": closed, "cleanup_complete": closed,
        "lanes": [{"device": str(index), "pid": 100 + index, "alive": not closed,
            "device_metadata": {"cuda_visible_devices": str(index), "device_count": 1,
                "current_device": 0, "name": "NVIDIA L40S", "uuid": f"GPU-{index}", "cpu_threads": 1},
            "request_count": 20, "source_calls": 12, "distribution_calls": 4}
            for index in range(3)]}


def _run():
    return {"status": "completed", "cases": [
        {"candidate": candidate, "case_key": f"case-{index}", "plan_seed": index,
         "public_sessions": [{"request_identity_sha256": "a" * 64,
            "methods": {name: {"calls": 2, "points": 4096, "failures": 0, "wall_seconds": 1.0}
                        for name in ("query_source", "query_initial_distribution")}}]}
        for candidate in ("R1_fast", "R1_accurate") for index in range(4)]}


def test_exact_three_allocated_devices_keep_original_tokens():
    assert CHECK.allocated_devices("GPU-abc, GPU-def,GPU-012") == ("GPU-abc", "GPU-def", "GPU-012")


@pytest.mark.parametrize("value", ["", "0", "0,1", "0,1,2,3", "0,1,0", "0,,2"])
def test_bad_gpu_allocation_is_rejected(value):
    with pytest.raises(ValueError, match="three distinct"):
        CHECK.allocated_devices(value)


@pytest.mark.parametrize("closed", [False, True])
def test_pool_evidence_checks_all_three_gpu_processes(closed):
    CHECK.validate_pool(_pool(closed), ("0", "1", "2"), closed=closed)


@pytest.mark.parametrize("fault", ["missing", "duplicate_pid", "source", "initial", "requests", "alive", "cleanup", "device"])
def test_pool_cannot_pass_without_real_requests_and_cleanup(fault):
    report = _pool(closed=True)
    if fault == "missing": report["lanes"].pop()
    if fault == "duplicate_pid": report["lanes"][1]["pid"] = report["lanes"][0]["pid"]
    if fault == "source": report["lanes"][1]["source_calls"] = 0
    if fault == "initial": report["lanes"][1]["distribution_calls"] = 0
    if fault == "requests": report["lanes"][1]["request_count"] = 0
    if fault == "alive": report["lanes"][1]["alive"] = True
    if fault == "cleanup": report["cleanup_complete"] = False
    if fault == "device": report["lanes"][1]["device"] = "foreign"
    with pytest.raises(ValueError):
        CHECK.validate_pool(report, ("0", "1", "2"), closed=True)


@pytest.mark.parametrize("key,value", [("cuda_visible_devices", "0"), ("device_count", 3),
    ("current_device", 1), ("name", ""), ("cpu_threads", 16), ("uuid", "GPU-0")])
def test_real_worker_identity_is_checked_once_from_startup_evidence(key, value):
    report = _pool()
    report["lanes"][1]["device_metadata"][key] = value
    with pytest.raises(ValueError):
        CHECK.validate_pool(report, ("0", "1", "2"), closed=False)


def test_older_torch_without_uuid_retains_explicit_mask_evidence():
    report = _pool()
    for lane in report["lanes"]:
        lane["device_metadata"]["uuid"] = ""
    CHECK.validate_pool(report, ("0", "1", "2"), closed=False)


def test_workload_preserves_case_request_order_and_ignores_elapsed_times():
    first, second = _run(), _run()
    for row in second["cases"]:
        row["public_sessions"][0]["methods"]["query_source"]["wall_seconds"] = 9.0
    assert CHECK.public_workload(first) == CHECK.public_workload(second)
    second["cases"].reverse()
    assert CHECK.public_workload(first) != CHECK.public_workload(second)


@pytest.mark.parametrize("fault", ["missing_case", "missing_session", "missing_digest", "missing_source", "failed"])
def test_incomplete_or_failed_workload_is_rejected(fault):
    report = _run()
    if fault == "missing_case": report["cases"].pop()
    if fault == "missing_session": report["cases"][0]["public_sessions"] = []
    if fault == "missing_digest": report["cases"][0]["public_sessions"][0].pop("request_identity_sha256")
    if fault == "missing_source": report["cases"][0]["public_sessions"][0]["methods"].pop("query_source")
    if fault == "failed": report["cases"][0]["public_sessions"][0]["methods"]["query_source"]["failures"] = 1
    with pytest.raises(ValueError):
        CHECK.public_workload(report)


def test_compare_reuses_scientific_comparator_and_does_not_claim_speedup(tmp_path):
    captured = []
    def compare(left, right, **kwargs):
        captured.append(kwargs)
        return {"status": "equivalent", "evaluation_wall_speedup": 3,
                "cases": [{"wall_speedup": 3}]}
    result = CHECK.compare_runs(SimpleNamespace(_comparison_payload=compare), _run(), _run(), tmp_path)
    assert captured == [{"left_root": tmp_path, "right_root": tmp_path, "rtol": 1e-10, "atol": 1e-12}]
    assert result["public_workload"] == "identical"
    assert result["multi_gpu_speedup_claimed"] is False
    assert "evaluation_wall_speedup" not in result and "wall_speedup" not in result["cases"][0]


@pytest.mark.parametrize("fault", ["values", "order", "count", "request_identity"])
def test_comparison_rejects_science_or_query_workload_changes(tmp_path, fault):
    first, second = _run(), _run()
    def compare(*args, **kwargs):
        return {"status": "not_equivalent" if fault == "values" else "equivalent", "problems": []}
    if fault == "order": second["cases"].reverse()
    if fault == "count": second["cases"][0]["public_sessions"][0]["methods"]["query_source"]["points"] += 1
    if fault == "request_identity": second["cases"][0]["public_sessions"][0]["request_identity_sha256"] = "b" * 64
    with pytest.raises(ValueError):
        CHECK.compare_runs(SimpleNamespace(_comparison_payload=compare), first, second, tmp_path)


def _metrics():
    return dict(candidate_solve_wall_seconds=9., candidate_solve_active_seconds=6.,
        candidate_platform_wait_seconds=3., candidate_oracle_queue_wait_seconds=2., candidate_oracle_lock_wait_seconds=1.)


def test_real_solve_budget_arithmetic_is_required():
    CHECK.validate_solve_metrics(_metrics())


@pytest.mark.parametrize("key,value", [("candidate_solve_active_seconds", 601.),
    ("candidate_solve_wall_seconds", 8.), ("candidate_oracle_queue_wait_seconds", 1.),
    ("candidate_platform_wait_seconds", float("nan")), ("candidate_solve_active_seconds", True)])
def test_bad_solve_budget_metrics_fail(key, value):
    values = _metrics()
    values[key] = value
    with pytest.raises(ValueError):
        CHECK.validate_solve_metrics(values)


def test_meter_forwards_lane_lock_instead_of_silently_serializing():
    lock = object()
    class Meter:
        def __init__(self, generator): self.generator = generator
    generator = SimpleNamespace(public_session_operation_lock=lambda state, case: lock)
    meter = CHECK._metered_generator(SimpleNamespace(_MeteredGenerator=Meter), generator)
    assert meter.public_session_operation_lock("state", "case") is lock


def test_smoke_slurm_is_small_and_preserves_all_gpu_tokens():
    script = (ROOT / "scripts/slurm_artemis_parallel_smoke.sh").read_text()
    assert "#SBATCH --gres=gpu:l40s:3" in script
    assert "#SBATCH --time=00:20:00" in script
    assert "${PDE_GAV_RUN_ROOT}/artemis_parallel_smoke_20260910/" in script
    assert "export CUDA_VISIBLE_DEVICES=" not in script
    assert "sbatch" not in script and "scancel" not in script


@pytest.mark.parametrize("bad_visibility", [False, True])
def test_smoke_generates_normally_issued_hidden_pairs_without_relabeling(tmp_path, bad_visibility):
    import numpy as np
    from pde_agent_verifier_solver.sdk import Case, CaseSource, CaseVisibility, GenerationKind, GenerationPurpose
    requests, issued, released = [], [], []
    def sample(state, request):
        requests.append(request)
        assert request.visibility is CaseVisibility.HIDDEN
        assert request.purpose is GenerationPurpose.PROMOTION
        index = len(requests) - 1
        child = request.kind is GenerationKind.PERTURBED
        case = Case(f"fresh-{index}", CaseSource.PERTURBED if child else CaseSource.GENERATED,
            CaseVisibility.DEVELOPMENT if bad_visibility else CaseVisibility.HIDDEN,
            {"arrays": {"t": np.array([0., 1.]), "x": np.array([0., 1.]), "v": np.array([-1., 1.])}},
            round_index=1, parent_case_id=request.anchors[0].case_id if child else None)
        issued.append(SimpleNamespace(case=case))
        return (issued[-1],)
    generator = SimpleNamespace(load=lambda checkpoint, settings: "state", sample=sample,
        release=lambda state, case_ids: released.extend(case_ids))
    binding = SimpleNamespace(checkpoint=object(), plugin=SimpleNamespace(generator=generator))
    benchmark = SimpleNamespace(_seed=lambda master, namespace, position: 10 + position,
        _write_json=lambda path, value: None)
    if bad_visibility:
        with pytest.raises(ValueError, match="normally issued hidden"):
            CHECK._hidden_cohort(benchmark, binding, tmp_path)
        assert released == ["fresh-0"]
    else:
        state, records, descriptors = CHECK._hidden_cohort(benchmark, binding, tmp_path)
        assert state == "state" and len(records) == len(descriptors) == 4
        assert all(actual is original for actual, original in zip(records, issued))
        assert requests[1].anchors == (records[0].case,)
        assert requests[3].anchors == (records[2].case,)
        assert not released
