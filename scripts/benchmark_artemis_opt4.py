#!/usr/bin/env python3
"""Offline real-GPU source batching checks; no Agent or Guidance training.

The existing opt3 runner owns the fixed candidate/cohort/numerical comparison.
This adapter adds request-boundary CUDA timing and a separate real HTTP mode.
Neither mode is a full iteration or a concurrent/multi-worker throughput test.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.util
import json
import math
import os
from pathlib import Path
from time import perf_counter
import traceback

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("artemis_opt4_previous", ROOT / "scripts/benchmark_artemis_opt3.py")
PREVIOUS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREVIOUS)
SCHEMA = "artemis_opt4_source_micro_v1"
MODES = ("grid_repeated_v", "arbitrary_v")


def measured_source(call, *, torch, device):
    """Record on the actual scientific caller's stream, including HTTP threads.

    Events bracket the whole source invocation, not individual kernels. Their
    stream timeline includes CPU-dispatch gaps; it is not GPU busy percentage.
    Synchronization/reset occurs only at request boundaries, never per operator.
    The returned public array is checked for finite values by the caller later.
    """
    with torch.cuda.device(device):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        stream = torch.cuda.current_stream(device)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        began = perf_counter()
        start.record(stream)
        result = call()
        end.record(stream)
        end.synchronize()
        elapsed = perf_counter() - began
        return result, {"source_wall_seconds": elapsed,
            "cuda_event_seconds": start.elapsed_time(end) / 1000.0,
            "allocated_before_bytes": allocated, "reserved_before_bytes": reserved,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}


def sidecar_files(root):
    """Called outside timing; do not count benchmark reports/model assets."""
    return {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}


def public_workload(session, case_id, points, *, transport, sidecar_root, torch, device, repeats):
    """Yield one completed request at a time; only HTTP transport owns a host."""
    from pde_agent_verifier_solver.engine.candidate_execution import (
        _CandidateOracleHost, CandidatePublicSessionClient,
    )
    host = None
    measured = []

    def invoke(request_case_id, method, args, kwargs):
        if request_case_id != case_id or method != "query_source" or kwargs or len(args) != 1:
            raise ValueError("source benchmark received an unexpected public request")
        value, timing = measured_source(lambda: session.query_source(args[0]), torch=torch, device=device)
        measured.append(timing)
        return value

    try:
        if transport == "http":
            host = _CandidateOracleHost(sidecar_root, case_ids=(case_id,), methods=("query_source",), invoke=invoke, close=None)
            host.start()
            client = CandidatePublicSessionClient(host.manifest_for_case(case_id))
            call = lambda: client.invoke(case_id, "query_source", (points,), {})
        elif transport == "direct":
            call = lambda: invoke(case_id, "query_source", (points,), {})
        else:
            raise ValueError("unknown source transport")
        for index in range(repeats + 1):
            before = dict(getattr(session, "usage_snapshot", {}))
            files_before = sidecar_files(sidecar_root) if transport == "http" else set()
            old_count = len(measured)
            began = perf_counter()
            values = call()
            outer_wall = perf_counter() - began
            if len(measured) != old_count + 1:
                raise ValueError("one source request must produce exactly one scientific invocation")
            timing = measured[-1]
            row = {"index": index, "kind": "first" if index == 0 else "repeat",
                "wall_seconds": outer_wall if transport == "http" else timing["source_wall_seconds"],
                **timing, "backend_usage": PREVIOUS.usage_delta(before, dict(getattr(session, "usage_snapshot", {})))}
            if transport == "http":
                row["sidecar_new_files"] = sorted(sidecar_files(sidecar_root) - files_before)
            yield values, row
    finally:
        if host is not None:
            host.close()


def checked_values(values, count):
    import numpy as np
    array = np.asarray(values)
    if array.shape != (count,) or array.dtype != np.float64 or not np.all(np.isfinite(array)):
        raise ValueError("source response must be finite float64 [4096]")
    return array.tolist()


def gpu_information(torch, device):
    properties = torch.cuda.get_device_properties(device)
    return {"name": properties.name, "capability": [properties.major, properties.minor],
        "total_memory_bytes": properties.total_memory, "device": str(device),
        "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda}


def micro(args):
    PREVIOUS.configure_threads(args.threads)
    legacy = PREVIOUS.harness(args.harness)
    root = legacy._fresh_root(args.output_root)
    binding = state = None
    records = ()
    report = {"schema": SCHEMA, "status": "running", "label": args.label,
        "transport": args.transport, "master_seed": args.seed, "parents": 2, "repeats": args.repeats,
        "wrapper_source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "legacy_harness_sha256": sha256(args.harness.read_bytes()).hexdigest(),
        "timing_scope": "first query of this session/workload after loading/generation; other workloads may have warmed field constants; not process cold startup",
        "cuda_timing_scope": "events on the scientific callback stream at request boundaries; elapsed timeline includes host-dispatch gaps, not GPU utilization or sum of kernel durations",
        "wall_timing_scope": "complete HTTP client.invoke including encode/decode" if args.transport == "http" else "direct public source call including request-boundary event bookkeeping",
        "memory_scope": "per-request process-wide allocator peaks including resident model/cases; no empty_cache",
        "agent_invoked": False, "formal_scientific_admission": False,
        "full_iteration_speedup_claimed": False, "multi_worker_throughput_claimed": False, "cases": []}
    try:
        if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_TMPDIR"):
            raise RuntimeError("real GPU microbenchmark requires a compute allocation")
        report["environment"] = {name: os.environ.get(name) for name in (*PREVIOUS.THREAD_VARS, "CUDA_VISIBLE_DEVICES", "SLURM_JOB_ID")}
        canonical_root = args.canonical_root.resolve(strict=True)
        report["canonical_source"] = legacy._version(canonical_root)
        if report["canonical_source"]["status_porcelain"]:
            raise ValueError("microbenchmark driver must be committed and clean")
        began = perf_counter()
        binding, config = legacy._load_binding(canonical_root, args.backend_config)
        report.update(backend_load_wall_seconds=perf_counter() - began, backend_configuration=config,
            backend_config_sha256=sha256(args.backend_config.read_bytes()).hexdigest(), backend_report=dict(binding.report))
        report["physical_source"] = legacy._version(Path(config["factory_settings"]["physical_checkout"]))
        if report["physical_source"]["status_porcelain"]:
            raise ValueError("physical source must be committed and clean")
        device = str(config["factory_settings"].get("device", "cpu"))
        import torch
        if not device.startswith("cuda") or not torch.cuda.is_available():
            raise ValueError("this benchmark requires a real CUDA backend")
        report["gpu"] = gpu_information(torch, device)
        state, records, descriptors = legacy._cohort(binding, parents=2, master_seed=args.seed, root=root, device=device)
        report["cohort"] = descriptors
        from pde_agent_verifier_solver.sdk import CaseAccessKind
        for record, descriptor in zip(records, descriptors, strict=True):
            for mode in MODES:
                points = PREVIOUS.source_points(record.case.public_data["equation"]["domain"],
                    legacy._seed(args.seed, f"micro/{mode}", descriptor["pair_index"]), mode, tx_count=16, velocity_count=256)
                row = {"case_key": descriptor["key"], "mode": mode, "point_count": 4096,
                    "tx_count": 16, "velocity_count": 256, "point_sha256": sha256(points.tobytes()).hexdigest(),
                    "points": points.tolist(), "calls": []}
                report["cases"].append(row)
                session = binding.plugin.generator.open_case_session(state, record.case, CaseAccessKind.PUBLIC_PROBLEM)
                samples = None
                try:
                    samples = public_workload(session, record.case.case_id, points, transport=args.transport,
                        sidecar_root=root / "oracle-sidecars" / f"{descriptor['key']}-{mode}", torch=torch, device=device, repeats=args.repeats)
                    for values, call in samples:
                        # Offline response validation and all JSON/filesystem work
                        # are outside both source and HTTP wall/event intervals.
                        row["calls"].append(call)
                        call["values"] = checked_values(values, 4096)
                        if args.transport == "http" and call["sidecar_new_files"]:
                            raise ValueError("4096-point source HTTP unexpectedly created sidecar files")
                        legacy._write_json(root / "partial-micro.json", report)
                finally:
                    try:
                        if samples is not None:
                            samples.close()
                    finally:
                        session.close()
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        try:
            if state is not None:
                binding.plugin.generator.release(state, tuple(record.case.case_id for record in records))
        except Exception as error:
            report.update(status="failed", cleanup_error=f"{type(error).__name__}: {error}")
        legacy._write_json(root / "micro.json", report)
    return 0 if report["status"] == "completed" else 1


def measurement_problems(report):
    problems = []
    if report.get("schema") != SCHEMA or report.get("transport") not in ("direct", "http"):
        problems.append("invalid_gpu_micro_schema_or_transport")
    pin = report.get("backend_configuration", {}).get("factory_settings", {}).get("expected_physical_revision")
    if pin is not None and pin != report.get("physical_source", {}).get("commit"):
        problems.append("measured_physical_revision_mismatch")
    for row in report.get("cases", []):
        if row.get("point_count") != 4096 or row.get("tx_count") != 16 or row.get("velocity_count") != 256:
            problems.append("micro_request_shape_changed")
        for call in row.get("calls", []):
            for key in ("cuda_event_seconds", "source_wall_seconds"):
                value = call.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    problems.append(f"invalid_measurement:{key}")
            for prefix in ("allocated", "reserved"):
                before, peak = call.get(f"{prefix}_before_bytes"), call.get(f"peak_{prefix}_bytes")
                if (isinstance(before, bool) or isinstance(peak, bool) or not isinstance(before, int)
                        or not isinstance(peak, int) or before < 0 or peak < before):
                    problems.append(f"invalid_memory_peak:{prefix}")
            if report.get("transport") == "http" and call.get("sidecar_new_files") != []:
                problems.append("http_source_sidecar_created_or_unverified")
    return problems


def compare_micro_payload(left, right, *, rtol=1e-10, atol=1e-12):
    # Reuse the existing point/value/cohort/counter comparator, never another
    # scientific evaluator. Only adapt its report schema in temporary mappings.
    report = PREVIOUS.compare_micro_payload({**left, "schema": PREVIOUS.MICRO_SCHEMA},
        {**right, "schema": PREVIOUS.MICRO_SCHEMA}, rtol=rtol, atol=atol)
    report["schema"] = "artemis_opt4_source_micro_comparison_v1"
    report["transport"] = left.get("transport")
    report["problems"].extend(measurement_problems(left) + measurement_problems(right))
    if left.get("transport") != right.get("transport"):
        report["problems"].append("cannot_compare_http_to_direct_timing")
    for key in ("name", "capability", "total_memory_bytes", "torch_version", "cuda_version"):
        if not left.get("gpu", {}).get(key) or left.get("gpu", {}).get(key) != right.get("gpu", {}).get(key):
            report["problems"].append(f"gpu_environment_mismatch:{key}")
    pairs = [(a, b) for lrow, rrow in zip(left.get("cases", []), right.get("cases", []))
             for a, b in zip(lrow.get("calls", []), rrow.get("calls", []))]
    for row, (a, b) in zip(report["calls"], pairs):
        for role, call in (("baseline", a), ("optimized", b)):
            for key in ("source_wall_seconds", "cuda_event_seconds", "allocated_before_bytes", "reserved_before_bytes",
                        "peak_allocated_bytes", "peak_reserved_bytes"):
                row[f"{role}_{key}"] = call.get(key)
    report["status"] = "not_equivalent" if report["problems"] else "equivalent"
    if report["problems"]:
        for row in report["calls"]:
            row.pop("wall_speedup", None)
    else:
        for row in report["calls"]:
            row["cuda_event_speedup"] = row["baseline_cuda_event_seconds"] / row["optimized_cuda_event_seconds"]
    report["full_iteration_speedup_claimed"] = False
    report["multi_worker_throughput_claimed"] = False
    return report


def compare_micro(args):
    legacy = PREVIOUS.harness(args.harness)
    root = legacy._fresh_root(args.output_root)
    try:
        report = compare_micro_payload(json.loads(args.baseline.read_text()), json.loads(args.optimized.read_text()), rtol=args.rtol, atol=args.atol)
    except Exception as error:
        report = {"schema": "artemis_opt4_source_micro_comparison_v1", "status": "failed", "error": str(error), "traceback": traceback.format_exc()}
    legacy._write_json(root / "comparison.json", report)
    return 0 if report["status"] == "equivalent" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "micro"):
        command = commands.add_parser(name)
        command.add_argument("--harness", type=Path, default=PREVIOUS.HARNESS)
        command.add_argument("--label", required=True)
        command.add_argument("--canonical-root", type=Path, default=ROOT)
        command.add_argument("--backend-config", type=Path, required=True)
        command.add_argument("--threads", type=int, choices=(1, 8), default=8)
        command.add_argument("--seed", type=int, default=2026090903)
        command.add_argument("--output-root", type=Path, required=True)
        if name == "run":
            command.add_argument("--source-campaign", type=Path, default=PREVIOUS.SOURCE)
            command.add_argument("--horizon", choices=("private-random", "native-end"), default="native-end")
        else:
            command.add_argument("--transport", choices=("direct", "http"), default="direct")
            command.add_argument("--repeats", type=int, choices=range(1, 6), default=2)
    for name in ("compare", "compare-micro"):
        command = commands.add_parser(name)
        command.add_argument("--harness", type=Path, default=PREVIOUS.HARNESS)
        command.add_argument("--baseline", type=Path, required=True)
        command.add_argument("--optimized", type=Path, required=True)
        command.add_argument("--output-root", type=Path, required=True)
        command.add_argument("--rtol", type=float, default=1e-10)
        command.add_argument("--atol", type=float, default=1e-12)
    args = parser.parse_args(argv)
    if hasattr(args, "seed") and args.seed < 0:
        parser.error("seed must be nonnegative")
    if hasattr(args, "rtol") and (not all(math.isfinite(v) for v in (args.rtol, args.atol)) or min(args.rtol, args.atol) < 0):
        parser.error("tolerances must be finite and nonnegative")
    return {"run": PREVIOUS.run, "compare": PREVIOUS.compare, "micro": micro, "compare-micro": compare_micro}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
