#!/usr/bin/env python3
"""Small offline D2/D3 comparison using the existing frozen-candidate harness.

No Agent, Guidance training, Official evaluation, or campaign resume is run.
Each run/micro command belongs in its own process in a compute allocation.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter
import traceback

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT.parent / "artemis-eval-benchmark-20260907/scripts/benchmark_canonical_artemis_oracle.py"
SOURCE = Path("${PDE_GAV_RUN_ROOT}/artemis_canonical_d2_guided_5round_20260908/job-5309385/campaign")
FROZEN = (
    ("R1_fast", "candidate-r1-0002", "Causal density-field hybrid (fast)", "161c939f7194a5e801b9a9ce92589268c5166132c2bc6a2dc399042217e89943"),
    ("R1_accurate", "candidate-r1-0003", "Causal density-field hybrid (accurate)", "04ff2fd9a03c829b17d4e2a4e891d6fb47907bf76403be18c7db8ccf0940a26c"),
)
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
MICRO_SCHEMA = "artemis_opt3_source_micro_v1"


def harness(path=HARNESS):
    spec = importlib.util.spec_from_file_location("artemis_opt3_legacy_harness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_threads(count):
    if count not in (1, 8):
        raise ValueError("use an independent process with one or eight threads")
    if any(name in sys.modules for name in ("numpy", "torch")):
        raise RuntimeError("configure threads before importing numerical libraries")
    for name in THREAD_VARS:
        os.environ[name] = str(count)


def frozen_candidates(source=SOURCE):
    """Resolve only the two original R1 Controller-owned nominations."""
    source = source.resolve(strict=True)
    history = None
    with (source / "events.jsonl").open() as handle:
        for line in handle:
            event = json.loads(line)
            if event.get("event_type") == "campaign_memory_updated" and event.get("round_index") == 1:
                history = {row["artifact_id"]: row for row in event["payload"]["candidate_history"]}
                break
    if history is None:
        raise ValueError("missing original R1 memory boundary")
    result = []
    for name, artifact_id, title, digest in FROZEN:
        row = history.get(artifact_id, {})
        if row.get("title") != title or row.get("round") != 1 or row.get("agent_id") != "artemis-critical-refiner":
            raise ValueError("original fast/accurate nomination identity changed")
        matches = list((source / "agent_turns").glob(f"round-0001-artemis-critical-refiner-*/.controller-candidates/{digest}"))
        if len(matches) != 1 or matches[0].is_symlink():
            raise ValueError("frozen Controller candidate is absent or ambiguous")
        path = matches[0].resolve(strict=True)
        if not path.is_relative_to(source / "agent_turns") or not (path / "candidate_bundle.json").is_file():
            raise ValueError("invalid frozen Controller candidate path")
        result.append({"name": name, "artifact_id": artifact_id, "tree_sha256": digest, "path": str(path)})
    return result


def workload_summary(report):
    rows = []
    for case in report.get("cases", []):
        row = {key: case.get(key) for key in ("candidate", "case_key", "status", "evaluation_wall_seconds")}
        row["oracle_methods"] = {}
        row["backend_usage"] = {}
        row["request_identity_sha256"] = [session.get("request_identity_sha256") for session in case.get("public_sessions", [])]
        for session in case.get("public_sessions", []):
            for name, counters in session.get("methods", {}).items():
                dest = row["oracle_methods"].setdefault(name, {})
                for key, value in counters.items():
                    dest[key] = dest.get(key, 0) + value
            for key, value in session.get("backend_usage", {}).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    row["backend_usage"][key] = row["backend_usage"].get(key, 0) + value
        rows.append(row)
    return {"schema": "artemis_opt3_workload_v1", "status": report.get("status"),
            "environment": report.get("environment"), "cases": rows,
            "evaluation_wall_seconds": report.get("evaluation_wall_seconds"),
            "backend_load_wall_seconds": report.get("backend_load_wall_seconds"),
            "generation_wall_seconds": report.get("generation_wall_seconds"),
            "timing_scope": "serial direct evaluations after generation/public probes; not per-request cold starts or multi-worker throughput"}


def run(args):
    configure_threads(args.threads)
    legacy = harness(args.harness)
    install_request_meter(legacy)
    candidates = frozen_candidates(args.source_campaign)
    call = argparse.Namespace(label=args.label, canonical_root=args.canonical_root,
        backend_config=args.backend_config, candidate=[f"{row['name']}={row['path']}" for row in candidates],
        parents=2, seed=args.seed, point_count=32, horizon=args.horizon, output_root=args.output_root)
    status = legacy.run_variant(call)
    report = json.loads((args.output_root / "result.json").read_text())
    summary = workload_summary(report)
    summary["wrapper_source_sha256"] = sha256(Path(__file__).read_bytes()).hexdigest()
    expected = {row["name"]: row["tree_sha256"] for row in candidates}
    summary["frozen_identity_verified"] = {row["name"]: row["tree_sha256"] for row in report.get("candidates", [])} == expected
    if not summary["frozen_identity_verified"]:
        summary["status"] = "failed"
        status = 1
    legacy._write_json(args.output_root / "workload-summary.json", summary)
    return status


def install_request_meter(legacy):
    """Extend only the offline harness meter, never the Backend or Controller."""
    original = legacy._SessionMeter
    class RequestMeter(original):
        def __init__(self, delegate, row):
            super().__init__(delegate, row)
            self.request_digest = sha256()

        def __getattr__(self, name):
            method = super().__getattr__(name)
            if not callable(method) or name not in ("query_source", "query_initial_distribution"):
                return method
            def measured(*args, **kwargs):
                import numpy as np
                points = np.ascontiguousarray(args[0] if args else kwargs["points"], dtype=np.float64)
                self.request_digest.update(json.dumps([name, list(points.shape)]).encode())
                self.request_digest.update(points.tobytes())
                self.row["request_identity_sha256"] = self.request_digest.hexdigest()
                return method(*args, **kwargs)
            return measured
    legacy._SessionMeter = RequestMeter


def source_points(domain, seed, mode, tx_count=16, velocity_count=64):
    import numpy as np
    rng = np.random.default_rng(seed)
    t0, t1 = domain["time_interval"]
    x0, x1 = domain["x_interval"]
    v0, v1 = domain["velocity_query_window"]
    tx = np.column_stack((rng.uniform(t0, t1, tx_count), rng.uniform(x0, x1, tx_count)))
    if mode == "grid_repeated_v":
        v = np.tile(np.linspace(v0, v1, velocity_count), tx_count)
    elif mode == "arbitrary_v":
        v = rng.uniform(v0, v1, tx_count * velocity_count)
    else:
        raise ValueError("unknown source workload")
    return np.ascontiguousarray(np.column_stack((np.repeat(tx, velocity_count, axis=0), v)), dtype=np.float64)


def usage_delta(before, after):
    return {key: value - before.get(key, 0) for key, value in after.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)}


def micro(args):
    configure_threads(args.threads)
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_TMPDIR"):
        raise RuntimeError("real source microbenchmark requires a compute allocation")
    legacy = harness(args.harness)
    root = legacy._fresh_root(args.output_root)
    state = None
    binding = None
    records = ()
    report = {"schema": MICRO_SCHEMA, "status": "running", "label": args.label,
        "master_seed": args.seed, "parents": 2, "repeats": args.repeats,
        "wrapper_source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "legacy_harness_sha256": sha256(args.harness.read_bytes()).hexdigest(),
        "environment": {name: os.environ.get(name) for name in (*THREAD_VARS, "CUDA_VISIBLE_DEVICES", "SLURM_JOB_ID")},
        "timing_scope": "first query of this session/workload after model loading/generation, then identical repeats; other workloads may already have warmed field constants; not process/field cold startup",
        "agent_invoked": False, "formal_scientific_admission": False, "cases": []}
    try:
        report["canonical_source"] = legacy._version(args.canonical_root)
        if report["canonical_source"]["status_porcelain"]:
            raise ValueError("microbenchmark canonical checkout must be committed and clean")
        began = perf_counter()
        binding, config = legacy._load_binding(args.canonical_root, args.backend_config)
        report.update(backend_load_wall_seconds=perf_counter() - began, backend_configuration=config,
            backend_config_sha256=sha256(args.backend_config.read_bytes()).hexdigest(), backend_report=dict(binding.report))
        report["physical_source"] = legacy._version(Path(config["factory_settings"]["physical_checkout"]))
        if report["physical_source"]["status_porcelain"]:
            raise ValueError("physical source must be committed and clean")
        device = str(config["factory_settings"].get("device", "cpu"))
        state, records, descriptors = legacy._cohort(binding, parents=2, master_seed=args.seed, root=root, device=device)
        report["cohort"] = descriptors
        from pde_agent_verifier_solver.sdk import CaseAccessKind
        import numpy as np
        for record, descriptor in zip(records, descriptors, strict=True):
            for mode in ("grid_repeated_v", "arbitrary_v"):
                points = source_points(record.case.public_data["equation"]["domain"],
                    legacy._seed(args.seed, f"micro/{mode}", descriptor["pair_index"]), mode)
                row = {"case_key": descriptor["key"], "mode": mode,
                    "point_count": len(points), "tx_count": 16, "velocity_count": 64,
                    "point_sha256": sha256(points.tobytes()).hexdigest(), "points": points.tolist(), "calls": []}
                report["cases"].append(row)
                session = binding.plugin.generator.open_case_session(state, record.case, CaseAccessKind.PUBLIC_PROBLEM)
                try:
                    for index in range(args.repeats + 1):
                        before = dict(getattr(session, "usage_snapshot", {}))
                        legacy._synchronize(device)
                        began = perf_counter()
                        values = np.asarray(session.query_source(points))
                        legacy._synchronize(device)
                        elapsed = perf_counter() - began
                        if values.size != len(points) or not np.all(np.isfinite(values)):
                            raise ValueError("source returned invalid/nonfinite values")
                        row["calls"].append({"index": index, "kind": "first" if index == 0 else "repeat",
                            "wall_seconds": elapsed, "values": values.reshape(-1).tolist(),
                            "backend_usage": usage_delta(before, dict(getattr(session, "usage_snapshot", {})))})
                        legacy._write_json(root / "partial-micro.json", report)
                finally:
                    session.close()
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        if state is not None:
            binding.plugin.generator.release(state, tuple(record.case.case_id for record in records))
        legacy._write_json(root / "micro.json", report)
    return 0 if report["status"] == "completed" else 1


def compare_micro_payload(left, right, *, rtol=1e-10, atol=1e-12):
    import numpy as np
    problems, rows = [], []
    for key in ("schema", "master_seed", "parents", "repeats"):
        if left.get(key) != right.get(key):
            problems.append(f"configuration_mismatch:{key}")
    if left.get("schema") != MICRO_SCHEMA or any(row.get("status") != "completed" for row in (left, right)):
        problems.append("both_micro_runs_must_complete")
    def science(report):
        config = dict(report.get("backend_configuration", {}).get("factory_settings", {}))
        for key in ("physical_checkout", "oracle_settings", "expected_physical_revision"):
            config.pop(key, None)
        return config
    if science(left) != science(right):
        problems.append("scientific_factory_configuration_mismatch")
    for key in THREAD_VARS:
        if left.get("environment", {}).get(key) != right.get("environment", {}).get(key):
            problems.append(f"thread_mismatch:{key}")
    for key in ("key", "pair_index", "distribution", "generation_seed", "public_array_sha256"):
        if [row.get(key) for row in left.get("cohort", [])] != [row.get(key) for row in right.get("cohort", [])]:
            problems.append(f"cohort_mismatch:{key}")
    lc, rc = left.get("cases", []), right.get("cases", [])
    if len(lc) != 8 or len(rc) != 8:
        problems.append("expected_two_workloads_on_four_cases")
    for a, b in zip(lc, rc):
        for key in ("case_key", "mode", "point_count", "tx_count", "velocity_count", "point_sha256", "points"):
            if a.get(key) != b.get(key):
                problems.append(f"request_identity_mismatch:{key}")
        ac, bc = a.get("calls", []), b.get("calls", [])
        if len(ac) != left.get("repeats", -1) + 1 or len(ac) != len(bc):
            problems.append("source_call_count_mismatch")
        for x, y in zip(ac, bc):
            u, v = np.asarray(x.get("values", [])), np.asarray(y.get("values", []))
            equal = u.size == a.get("point_count") and u.shape == v.shape and bool(np.allclose(u, v, rtol=rtol, atol=atol, equal_nan=False))
            identity = x.get("index") == y.get("index") and x.get("kind") == y.get("kind")
            for call in (x, y):
                usage = call.get("backend_usage", {})
                if usage.get("source_query_calls") != 1 or usage.get("source_queried_points") != a.get("point_count"):
                    problems.append("source_counters_missing_or_inconsistent")
            for counter in ("source_query_calls", "source_queried_points", "query_calls", "queried_points"):
                if x.get("backend_usage", {}).get(counter) != y.get("backend_usage", {}).get(counter):
                    problems.append(f"source_counter_changed:{counter}")
            if any(not isinstance(call.get("wall_seconds"), (int, float)) or not math.isfinite(call["wall_seconds"])
                   or call["wall_seconds"] <= 0 for call in (x, y)):
                problems.append("invalid_wall_timing")
            if not equal or not identity:
                problems.append("source_values_or_call_order_changed")
            rows.append({"case_key": a.get("case_key"), "mode": a.get("mode"), "kind": x.get("kind"),
                "index": x.get("index"), "equivalent": bool(equal and identity),
                "max_absolute_difference": float(np.max(np.abs(u-v))) if u.size and u.shape == v.shape else None,
                "baseline_wall_seconds": x.get("wall_seconds"), "optimized_wall_seconds": y.get("wall_seconds")})
    if not problems:
        for row in rows:
            row["wall_speedup"] = row["baseline_wall_seconds"] / row["optimized_wall_seconds"]
    return {"schema": "artemis_opt3_source_micro_comparison_v1", "status": "equivalent" if not problems else "not_equivalent",
            "problems": problems, "calls": rows, "rtol": rtol, "atol": atol,
            "formal_scientific_admission": False}


def compare(args):
    legacy = harness(args.harness)
    left_report, right_report = json.loads(args.baseline.read_text()), json.loads(args.optimized.read_text())
    for report in (left_report, right_report):
        explicit = report.get("backend_configuration", {}).get("factory_settings", {}).get("expected_physical_revision")
        if explicit is not None and explicit != report.get("physical_source", {}).get("commit"):
            raise ValueError("explicit physical revision does not match measured source")
    if args.command == "compare":
        legacy._fresh_root(args.output_root)
        # The old comparator predates configurable physical pins. Remove only
        # that already-verified provenance field from in-memory comparison input.
        for value in (left_report, right_report):
            value["backend_configuration"]["factory_settings"].pop("expected_physical_revision", None)
        report = legacy._comparison_payload(left_report, right_report, left_root=args.baseline.parent,
            right_root=args.optimized.parent, rtol=args.rtol, atol=args.atol)
        report.update(baseline_result=str(args.baseline), optimized_result=str(args.optimized))
        status = 0 if report["status"] == "equivalent" else 1
        left, right = workload_summary(left_report), workload_summary(right_report)
        def workload(report):
            return [(row["candidate"], row["case_key"], row["request_identity_sha256"], {name: {key: value for key, value in values.items() if key != "wall_seconds"}
                for name, values in row["oracle_methods"].items()}) for row in report["cases"]]
        path = args.output_root / "comparison.json"
        identities_present = all(row["request_identity_sha256"] and all(row["request_identity_sha256"])
            for side in (left, right) for row in side["cases"])
        report["public_workload_identical"] = identities_present and workload(left) == workload(right)
        if not report["public_workload_identical"]:
            report["status"] = "not_equivalent"
            report["problems"].append("public_calls_points_or_failures_changed")
            status = 1
        if status:
            report["evaluation_wall_speedup"] = None
            for row in report.get("cases", []):
                row["wall_speedup"] = None
        legacy._write_json(path, report)
        return status
    root = legacy._fresh_root(args.output_root)
    report = compare_micro_payload(left_report, right_report, rtol=args.rtol, atol=args.atol)
    legacy._write_json(root / "comparison.json", report)
    return 0 if report["status"] == "equivalent" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    resolve = commands.add_parser("candidates")
    resolve.add_argument("--source-campaign", type=Path, default=SOURCE)
    for name in ("run", "micro"):
        command = commands.add_parser(name)
        command.add_argument("--harness", type=Path, default=HARNESS)
        command.add_argument("--label", required=True)
        command.add_argument("--canonical-root", type=Path, default=ROOT)
        command.add_argument("--backend-config", type=Path, required=True)
        command.add_argument("--threads", type=int, choices=(1, 8), default=8)
        command.add_argument("--seed", type=int, default=2026090903)
        command.add_argument("--output-root", type=Path, required=True)
        if name == "run":
            command.add_argument("--source-campaign", type=Path, default=SOURCE)
            command.add_argument("--horizon", choices=("private-random", "native-end"), default="native-end")
        else:
            command.add_argument("--repeats", type=int, choices=range(1, 6), default=2)
    for name in ("compare", "compare-micro"):
        command = commands.add_parser(name)
        command.add_argument("--harness", type=Path, default=HARNESS)
        command.add_argument("--baseline", type=Path, required=True)
        command.add_argument("--optimized", type=Path, required=True)
        command.add_argument("--output-root", type=Path, required=True)
        command.add_argument("--rtol", type=float, default=1e-10)
        command.add_argument("--atol", type=float, default=1e-12)
    args = parser.parse_args(argv)
    if args.command == "candidates":
        print(json.dumps(frozen_candidates(args.source_campaign), indent=2))
        return 0
    if hasattr(args, "seed") and args.seed < 0:
        parser.error("seed must be nonnegative")
    if hasattr(args, "rtol") and (not all(math.isfinite(v) for v in (args.rtol, args.atol)) or min(args.rtol, args.atol) < 0):
        parser.error("tolerances must be finite and nonnegative")
    return {"run": run, "micro": micro, "compare": compare, "compare-micro": compare}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
