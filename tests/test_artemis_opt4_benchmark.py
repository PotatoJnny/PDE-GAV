"""Synthetic fixtures only: benchmark validation is not physical evidence."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("_opt4_benchmark_test",
    Path(__file__).resolve().parents[1] / "scripts/benchmark_artemis_opt4.py")
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


def report(transport="direct", wall=2.0):
    result = {"schema": b.SCHEMA, "status": "completed", "transport": transport,
        "master_seed": 2026090903, "parents": 2, "repeats": 2,
        "environment": {name: "8" for name in b.PREVIOUS.THREAD_VARS},
        "gpu": {"name": "synthetic-test-only", "capability": [8,9],
                "total_memory_bytes": 12345678, "torch_version": "fixture", "cuda_version": "fixture"},
        "backend_configuration": {"factory_settings": {"device": "cuda", "expected_physical_revision": "a"*40}},
        "physical_source": {"commit": "a"*40}, "cohort": [], "cases": []}
    for case in range(4):
        result["cohort"].append({"key": str(case), "pair_index": case//2,
            "distribution": "base" if case%2==0 else "perturbed", "generation_seed": case, "public_array_sha256": str(case)})
        for mode in b.MODES:
            values = [float(case)] * 4096
            row = {"case_key": str(case), "mode": mode, "point_count": 4096,
                "tx_count": 16, "velocity_count": 256, "point_sha256": "fixture",
                "points": [[0., 0., 0.]]*4096, "calls": []}
            for index in range(3):
                row["calls"].append({"index": index, "kind": "first" if index==0 else "repeat",
                    "values": list(values), "wall_seconds": wall, "source_wall_seconds": wall,
                    "cuda_event_seconds": wall/2, "allocated_before_bytes": 100,
                    "reserved_before_bytes": 200, "peak_allocated_bytes": 150, "peak_reserved_bytes": 250,
                    "backend_usage": {"source_query_calls": 1, "source_queried_points": 4096,
                                      "query_calls": 1, "queried_points": 4096},
                    **({"sidecar_new_files": []} if transport=="http" else {})})
            result["cases"].append(row)
    return result


@pytest.mark.parametrize("transport", ["direct", "http"])
def test_equivalent_results_and_memory_cuda_timings(transport):
    result = b.compare_micro_payload(report(transport), report(transport, 1.0))
    assert result["status"] == "equivalent", result["problems"]
    assert len(result["calls"]) == 24
    assert all(row["wall_speedup"] == 2 for row in result["calls"])
    assert all(row["cuda_event_speedup"] == 2 for row in result["calls"])
    assert not result["full_iteration_speedup_claimed"]


@pytest.mark.parametrize("fault", ["values", "count", "shape", "points", "gpu", "transport", "pin", "cuda_time", "source_time", "memory", "sidecar", "missing_sidecar", "status"])
def test_invalid_or_changed_workloads_withhold_speedups(fault):
    a, c = report("http"), report("http", 1)
    call = c["cases"][0]["calls"][0]
    if fault == "values": call["values"][0] = 99
    if fault == "count": call["backend_usage"]["source_query_calls"] = 2
    if fault == "shape": c["cases"][0]["point_count"] = 1024
    if fault == "points": c["cases"][0]["point_sha256"] = "changed"
    if fault == "gpu": c["gpu"]["name"] = "other"
    if fault == "transport": c["transport"] = "direct"
    if fault == "pin": c["physical_source"]["commit"] = "b"*40
    if fault == "cuda_time": call["cuda_event_seconds"] = -1
    if fault == "source_time": call["source_wall_seconds"] = 0
    if fault == "memory": call["peak_allocated_bytes"] = 0
    if fault == "sidecar": call["sidecar_new_files"] = ["unexpected.npy"]
    if fault == "missing_sidecar": del call["sidecar_new_files"]
    if fault == "status": c["status"] = "failed"
    result = b.compare_micro_payload(a,c)
    assert result["status"] == "not_equivalent"
    assert result["problems"]
    assert all("wall_speedup" not in row and "cuda_event_speedup" not in row for row in result["calls"])


def test_source_points_are_4096_and_repeated_velocities_deduplicate():
    domain = {"time_interval": (0,1), "x_interval": (0,6), "velocity_query_window": (-4,4)}
    a = b.PREVIOUS.source_points(domain, 31, "grid_repeated_v", 16, 256)
    c = b.PREVIOUS.source_points(domain, 31, "arbitrary_v", 16, 256)
    assert a.shape == c.shape == (4096,3)
    assert len(np.unique(a[:,2])) == 256
    assert len(np.unique(c[:,2])) == 4096
    np.testing.assert_array_equal(a[:,:2], c[:,:2])


@pytest.mark.parametrize("transport", ["direct", "http"])
def test_public_workload_counts_calls_and_no_disk(tmp_path, monkeypatch, transport):
    state = {"source_query_calls": 0, "source_queried_points": 0}
    def source(points):
        state["source_query_calls"] += 1
        state["source_queried_points"] += len(points)
        return points[:,2].copy()
    def measured(call, **kwargs):
        return call(), {"source_wall_seconds": 0.1, "cuda_event_seconds": 0.05,
            "allocated_before_bytes": 0, "reserved_before_bytes": 0,
            "peak_allocated_bytes": 1, "peak_reserved_bytes": 1}
    monkeypatch.setattr(b, "measured_source", measured)
    session = SimpleNamespace(query_source=source, usage_snapshot=state)
    points = np.zeros((4096,3), dtype=np.float64)
    rows = list(b.public_workload(session, "fixture", points, transport=transport,
        sidecar_root=tmp_path/"sidecars", torch=None, device="cuda", repeats=2))
    assert len(rows) == 3 and state["source_query_calls"] == 3
    for actual, row in rows:
        np.testing.assert_array_equal(actual, points[:,2])
        assert row["backend_usage"]["source_query_calls"] == 1
        assert row["backend_usage"]["source_queried_points"] == 4096
        if transport == "http":
            assert row["sidecar_new_files"] == []


def test_offline_value_gate_rejects_wrong_dtype_shape_or_nonfinite():
    assert len(b.checked_values(np.zeros(4096,dtype=np.float64),4096)) == 4096
    for value in (np.zeros(4096,dtype=np.float32), np.zeros((4096,1)), np.full(4096,np.nan)):
        with pytest.raises(ValueError):
            b.checked_values(value,4096)
