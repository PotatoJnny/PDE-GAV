"""Offline harness fixtures only; these tests are not real-model validation."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("artemis_opt3_benchmark", ROOT / "scripts/benchmark_artemis_opt3.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_grid_and_arbitrary_velocity_share_tx_but_not_velocity_arrays():
    domain = {"time_interval": [0, 1], "x_interval": [0, 2], "velocity_query_window": [-8, 8]}
    grid = MOD.source_points(domain, 123, "grid_repeated_v")
    random = MOD.source_points(domain, 123, "arbitrary_v")
    assert grid.shape == random.shape == (1024, 3)
    assert np.array_equal(grid[:, :2], random[:, :2])
    assert np.array_equal(grid[:64, 2], grid[64:128, 2])
    assert not np.array_equal(random[:64, 2], random[64:128, 2])
    assert np.array_equal(grid, MOD.source_points(domain, 123, "grid_repeated_v"))
    assert grid.dtype == np.float64 and grid.flags.c_contiguous


def test_unknown_micro_pattern_is_rejected():
    with pytest.raises(ValueError, match="unknown"):
        MOD.source_points({"time_interval": [0, 1], "x_interval": [0, 1], "velocity_query_window": [-1, 1]}, 1, "invalid")


def fixture_report():
    rows = []
    for case in range(4):
        for mode in ("grid_repeated_v", "arbitrary_v"):
            rows.append({"case_key": str(case), "mode": mode, "point_count": 2,
                "tx_count": 1, "velocity_count": 2, "point_sha256": "points",
                "points": [[0, 0, -1], [0, 0, 1]],
                "calls": [{"index": i, "kind": "first" if i == 0 else "repeat", "values": [0.25, -0.25],
                           "wall_seconds": 0.1, "backend_usage": {"source_query_calls": 1, "source_queried_points": 2}} for i in range(3)]})
    return {"schema": MOD.MICRO_SCHEMA, "status": "completed", "master_seed": 2026090903,
        "parents": 2, "repeats": 2, "cases": rows,
        "cohort": [{"key": str(i), "pair_index": i // 2, "distribution": "base" if i % 2 == 0 else "base_perturbed",
                    "generation_seed": i, "public_array_sha256": {"t": "hash"}} for i in range(4)],
        "environment": {name: "8" for name in MOD.THREAD_VARS},
        "backend_configuration": {"factory_settings": {"device": "cuda", "physical_checkout": "/d2", "oracle_settings": {}}}}


def test_micro_compare_accepts_only_numerically_equivalent_implementation_change():
    left = fixture_report()
    right = copy.deepcopy(left)
    right["backend_configuration"]["factory_settings"].update(physical_checkout="/d3", oracle_settings={"profile": "frozen"})
    right["cases"][0]["calls"][0]["values"][0] += 1e-13
    result = MOD.compare_micro_payload(left, right)
    assert result["status"] == "equivalent" and len(result["calls"]) == 24


@pytest.mark.parametrize("change", ["values", "nonfinite", "points", "count", "calls", "order", "thread", "seed", "science", "cohort", "incomplete", "counter", "timing"])
def test_micro_compare_rejects_semantic_or_workload_drift(change):
    left, right = fixture_report(), fixture_report()
    if change == "values": right["cases"][0]["calls"][0]["values"][0] += 0.1
    if change == "nonfinite": right["cases"][0]["calls"][0]["values"][0] = float("nan")
    if change == "points": right["cases"][0]["points"][0][0] = 0.5
    if change == "count": right["cases"].pop()
    if change == "calls": right["cases"][0]["calls"].pop()
    if change == "order": right["cases"][0]["calls"][0]["index"] = 9
    if change == "thread": right["environment"]["OMP_NUM_THREADS"] = "1"
    if change == "seed": right["master_seed"] += 1
    if change == "science": right["backend_configuration"]["factory_settings"]["device"] = "cpu"
    if change == "cohort": right["cohort"][0]["public_array_sha256"]["t"] = "other"
    if change == "incomplete": right["status"] = "running"
    if change == "counter": right["cases"][0]["calls"][0]["backend_usage"] = {"source_query_calls": 2}
    if change == "timing": right["cases"][0]["calls"][0]["wall_seconds"] = -1
    result = MOD.compare_micro_payload(left, right)
    assert result["status"] == "not_equivalent"
    assert all("wall_speedup" not in row for row in result["calls"])


def test_usage_delta_separates_calls_and_ignores_labels():
    assert MOD.usage_delta({"calls": 1, "wall_seconds": 2}, {"calls": 3, "wall_seconds": 5, "cost_unit": "point"}) == {"calls": 2, "wall_seconds": 3}


def test_workload_summary_counts_public_source_separately_from_evaluation():
    report = {"cases": [{"candidate": "fast", "case_key": "base", "evaluation_wall_seconds": 3,
        "public_sessions": [{"methods": {"query_source": {"calls": 2, "points": 1024, "wall_seconds": 1, "failures": 0}},
                             "backend_usage": {"source_backend_wall_seconds": 0.8, "cost_unit": "point"}}]}]}
    summary = MOD.workload_summary(report)
    assert summary["cases"][0]["oracle_methods"]["query_source"]["points"] == 1024
    assert summary["cases"][0]["backend_usage"] == {"source_backend_wall_seconds": 0.8}


def test_thread_config_rejects_already_imported_numerical_runtime():
    with pytest.raises(RuntimeError, match="before importing"):
        MOD.configure_threads(8)
    with pytest.raises(ValueError):
        MOD.configure_threads(2)


def source_fixture(tmp_path):
    history = [{"artifact_id": aid, "title": title, "round": 1, "agent_id": "artemis-critical-refiner"}
               for _, aid, title, _ in MOD.FROZEN]
    event = {"event_type": "campaign_memory_updated", "round_index": 1, "payload": {"candidate_history": history}}
    (tmp_path / "events.jsonl").write_text(json.dumps(event) + "\n")
    for _, _, _, digest in MOD.FROZEN:
        root = tmp_path / "agent_turns/round-0001-artemis-critical-refiner-frozen/.controller-candidates" / digest
        root.mkdir(parents=True)
        (root / "candidate_bundle.json").write_text("{}")
    return event


def test_frozen_candidate_lookup_selects_only_exact_fast_and_accurate(tmp_path):
    source_fixture(tmp_path)
    result = MOD.frozen_candidates(tmp_path)
    assert [row["artifact_id"] for row in result] == ["candidate-r1-0002", "candidate-r1-0003"]
    assert all(".controller-candidates" in row["path"] for row in result)


def test_frozen_lookup_rejects_changed_original_title(tmp_path):
    event = source_fixture(tmp_path)
    event["payload"]["candidate_history"][0]["title"] = "replacement"
    (tmp_path / "events.jsonl").write_text(json.dumps(event))
    with pytest.raises(ValueError, match="identity"):
        MOD.frozen_candidates(tmp_path)


def test_run_reuses_legacy_harness_with_fixed_workload(tmp_path, monkeypatch):
    args = SimpleNamespace(threads=8, harness=MOD.HARNESS, source_campaign=tmp_path, label="d3",
        canonical_root=ROOT, backend_config=ROOT / "config.json", seed=2026090903,
        horizon="native-end", output_root=tmp_path / "output")
    candidates = [{"name": name, "tree_sha256": digest, "path": str(tmp_path / digest)} for name, _, _, digest in MOD.FROZEN]
    seen = []
    def run(call):
        seen.append(call)
        call.output_root.mkdir()
        (call.output_root / "result.json").write_text(json.dumps({"status": "completed", "candidates": candidates, "cases": []}))
        return 0
    fake = SimpleNamespace(run_variant=run, _write_json=lambda path, value: path.write_text(json.dumps(value)))
    monkeypatch.setattr(MOD, "configure_threads", lambda count: None)
    monkeypatch.setattr(MOD, "frozen_candidates", lambda source: candidates)
    monkeypatch.setattr(MOD, "harness", lambda path: fake)
    monkeypatch.setattr(MOD, "install_request_meter", lambda legacy: None, raising=False)
    assert MOD.run(args) == 0
    assert seen[0].parents == 2 and seen[0].point_count == 32 and seen[0].seed == 2026090903
    assert len(seen[0].candidate) == 2


def test_request_hash_detects_points_order_and_method_changes():
    legacy = MOD.harness()
    MOD.install_request_meter(legacy)
    class Session:
        def query_source(self, points): return np.ones(len(points))
        def query_initial_distribution(self, points): return np.ones(len(points))
    def digest(points, method="query_source"):
        row = {}
        meter = legacy._SessionMeter(Session(), row)
        getattr(meter, method)(points)
        assert row["methods"][method]["calls"] == 1
        return row["request_identity_sha256"]
    points = np.array([[0., 1., 2.], [1., 2., 3.]])
    assert digest(points) == digest(points.copy())
    assert digest(points) != digest(points[::-1])
    assert digest(points) != digest(points, "query_initial_distribution")


def test_micro_command_rejects_login_node_without_loading_backend(tmp_path, monkeypatch):
    monkeypatch.setattr(MOD, "configure_threads", lambda count: None)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="compute allocation"):
        MOD.micro(SimpleNamespace(threads=8, output_root=tmp_path / "no-output"))
    assert not (tmp_path / "no-output").exists()


def test_compare_accepts_verified_physical_pin_without_mutating_saved_inputs(tmp_path, monkeypatch):
    left, right = fixture_report(), fixture_report()
    right["backend_configuration"]["factory_settings"]["expected_physical_revision"] = "a" * 40
    right["physical_source"] = {"commit": "a" * 40}
    lp, rp = tmp_path / "d2.json", tmp_path / "d3.json"
    lp.write_text(json.dumps(left))
    rp.write_text(json.dumps(right))
    before = rp.read_bytes()
    args = SimpleNamespace(command="compare-micro", harness=MOD.HARNESS,
        baseline=lp, optimized=rp, output_root=tmp_path / "comparison", rtol=1e-10, atol=1e-12)
    assert MOD.compare(args) == 0
    assert rp.read_bytes() == before
    right["physical_source"]["commit"] = "b" * 40
    rp.write_text(json.dumps(right))
    with pytest.raises(ValueError, match="physical revision"):
        MOD.compare(args)


def test_micro_runtime_records_first_repeat_and_releases_sessions(tmp_path, monkeypatch):
    legacy = MOD.harness()
    events = []
    domain = {"time_interval": [0, 1], "x_interval": [0, 1], "velocity_query_window": [-8, 8]}
    class Session:
        calls = 0
        @property
        def usage_snapshot(self): return {"source_query_calls": self.calls, "source_queried_points": self.calls * 1024}
        def query_source(self, points):
            self.calls += 1
            return points[:, 2] ** 2
        def close(self): events.append("closed")
    class Generator:
        def open_case_session(self, *args): return Session()
        def release(self, state, ids): events.append(("release", ids))
    binding = SimpleNamespace(plugin=SimpleNamespace(generator=Generator()), report={})
    records = [SimpleNamespace(case=SimpleNamespace(case_id=f"case-{i}", public_data={"equation": {"domain": domain}})) for i in range(4)]
    descriptors = [{"key": f"pair-{i // 2}-{i % 2}", "pair_index": i // 2} for i in range(4)]
    config = {"factory_settings": {"device": "cpu", "physical_checkout": str(tmp_path)}}
    cp = tmp_path / "config.json"
    cp.write_text(json.dumps(config))
    monkeypatch.setattr(MOD, "configure_threads", lambda count: None)
    monkeypatch.setattr(MOD, "harness", lambda path: legacy)
    monkeypatch.setattr(legacy, "_version", lambda path: {"status_porcelain": "", "commit": "x"})
    monkeypatch.setattr(legacy, "_load_binding", lambda *args: (binding, config))
    monkeypatch.setattr(legacy, "_cohort", lambda *args, **kwargs: ("state", records, descriptors))
    monkeypatch.setenv("SLURM_JOB_ID", "fixture-only")
    monkeypatch.setenv("SLURM_TMPDIR", str(tmp_path))
    args = SimpleNamespace(threads=8, harness=MOD.HARNESS, canonical_root=ROOT, backend_config=cp,
        label="fixture", output_root=tmp_path / "micro", seed=123, repeats=2)
    assert MOD.micro(args) == 0
    report = json.loads((args.output_root / "micro.json").read_text())
    assert len(report["cases"]) == 8
    for row in report["cases"]:
        assert [call["kind"] for call in row["calls"]] == ["first", "repeat", "repeat"]
        assert all(call["backend_usage"]["source_query_calls"] == 1 for call in row["calls"])
    assert events.count("closed") == 8
    assert events[-1] == ("release", tuple(f"case-{i}" for i in range(4)))
