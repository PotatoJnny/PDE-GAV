#!/usr/bin/env python3
"""Offline real-loopback HTTP benchmark with a synthetic source, not physics.

Compares the former 64 KiB/scanned-reply codec with opt4 128 KiB/trusted-reply
transport in isolated hosts. No model, GPU, Agent, or formal experiment runs.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pde_agent_verifier_solver.engine.candidate_execution import (  # noqa: E402
    CandidatePublicSessionClient, _CandidateOracleHost,
)

SCHEMA = "artemis-opt4-loopback-rpc-micro-v1"
VARIANTS = (("baseline_64k_scanned", 64 * 1024, True),
            ("opt4_128k_trusted", 128 * 1024, False))


def payload(iteration):
    rows = np.arange(4096)
    points = np.empty((4096, 3), dtype=np.float64)
    points[:, 0] = iteration / 128.0
    points[:, 1] = (rows // 256) / 64.0
    points[:, 2] = (rows % 256) / 32.0 - 4.0
    return points


def synthetic_source(points):
    return points[:, 0] + 0.125 * points[:, 1] - 0.25 * points[:, 2]


def require_equal(actual, expected):
    if (not isinstance(actual, np.ndarray) or actual.dtype != expected.dtype
            or actual.shape != expected.shape or actual.tobytes() != expected.tobytes()):
        raise AssertionError("HTTP RPC changed dtype, shape, numeric values or order")


def summary(values):
    return {"median": float(np.median(values)), "p95": float(np.percentile(values, 95))}


def run_filesystem(parent, iterations):
    parent = Path(parent).resolve(strict=True)
    root = Path(tempfile.mkdtemp(prefix="artemis-opt4-rpc-", dir=parent))
    report = {"parent": str(parent), "workspace": str(root), "status": "running", "variants": {}}
    with ExitStack() as stack:
        clients, calls = {}, {}
        for name, threshold, scan_reply in VARIANTS:
            calls[name] = []
            def invoke(case, method, args, kwargs, log=calls[name]):
                if case != "one" or kwargs or len(args) != 1:
                    raise ValueError("invalid mock invocation")
                points = args[0]
                log.append({"method": method, "shape": list(points.shape)})
                return synthetic_source(points) if method == "query_source" else points
            host = _CandidateOracleHost(root / name, case_ids=("one",),
                methods=("query_source", "echo"), invoke=invoke, close=lambda _: None)
            # Explicit test-only baseline; no deployment settings are changed.
            host.codec.inline_array_bytes = threshold
            host.codec.validate_encoded_array_finite = scan_reply
            stack.callback(host.close)
            host.start()
            client = CandidatePublicSessionClient(host.manifest_for_case("one"))
            client.codec.inline_array_bytes = threshold
            client.codec.validate_decoded_array_finite = scan_reply
            clients[name] = client
            report["variants"][name] = {"inline_array_bytes": threshold,
                "trusted_reply_finite_scans": scan_reply, "samples": []}
        for iteration in range(iterations + 1):
            points = payload(iteration)
            expected = synthetic_source(points)
            order = VARIANTS if iteration % 2 == 0 else tuple(reversed(VARIANTS))
            for name, _, _ in order:
                began = perf_counter()
                result = clients[name].invoke("one", "query_source", (points,), {})
                elapsed = perf_counter() - began
                require_equal(result, expected)
                report["variants"][name]["samples"].append({"iteration": iteration,
                    "kind": "first" if iteration == 0 else "warm_changing_payload",
                    "full_http_roundtrip_seconds": elapsed, "bitwise_equal": True})
        for name, threshold, _ in VARIANTS:
            row = report["variants"][name]
            files = tuple((root / name).glob("*.npy"))
            row.update(measured_sidecar_count=len(files),
                measured_sidecar_bytes=sum(path.stat().st_size for path in files),
                source_calls=len(calls[name]), source_points=(iterations + 1) * 4096,
                first_http_roundtrip_seconds=row["samples"][0]["full_http_roundtrip_seconds"],
                warm_http_roundtrip_seconds=summary([sample["full_http_roundtrip_seconds"]
                    for sample in row["samples"][1:]]))
            expected_sidecars = iterations + 1 if threshold == 64 * 1024 else 0
            if len(files) != expected_sidecars or len(calls[name]) != iterations + 1:
                raise AssertionError("unexpected source-call count or sidecar creation")
            # Separate generic large-array ABI probe, not a legal Artemis source
            # query. Excluded from source timing, counts, and measured disk bytes.
            large = np.arange(8192 * 3, dtype=np.float64).reshape(8192, 3)
            require_equal(clients[name].invoke("one", "echo", (large,), {}), large)
            row["large_array_compatibility_passed"] = len(tuple((root / name).glob("*.npy"))) > len(files)
            if not row["large_array_compatibility_passed"]:
                raise AssertionError("large-array sidecar compatibility was not exercised")
        baseline, optimized = [report["variants"][name] for name, _, _ in VARIANTS]
        report["warm_median_ratio_baseline_over_opt4"] = (
            baseline["warm_http_roundtrip_seconds"]["median"]
            / optimized["warm_http_roundtrip_seconds"]["median"])
        report["status"] = "passed"
    return report


def run(args):
    if not 1 <= args.iterations <= 50:
        raise ValueError("iterations must be between 1 and 50")
    if not os.environ.get("SLURM_JOB_ID"):
        if not args.allow_local_smoke or args.iterations > 2:
            raise RuntimeError("use Slurm, or explicitly allow a local smoke with at most two warm calls")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    report = {"schema": SCHEMA, "status": "running", "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "model_execution_performed": False,
        "scope": "real_loopback_http_with_synthetic_source_not_physical_or_solver_evaluation",
        "timed_scope": "complete client.invoke; both codecs, JSON, HTTP and synthetic source; excludes setup and offline equality checks",
        "first_call_scope": "first RPC on each host after common imports; not independent process cold startup",
        "payload_schedule": "first plus changing entire t layer each warm call, identical sequence for both variants",
        "comparison": "64KiB scanned replies versus 128KiB trusted replies; input validation identical",
        "iterations_after_first": args.iterations, "request_numeric_bytes": 98304,
        "reply_numeric_bytes": 32768, "deployment_changed": False, "filesystems": {}}
    try:
        for name, parent in (("scratch", args.scratch_parent), ("local", args.local_parent)):
            if parent is not None:
                report["filesystems"][name] = run_filesystem(parent, args.iterations)
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return 0 if report["status"] == "passed" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch-parent", type=Path, required=True)
    parser.add_argument("--local-parent", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--allow-local-smoke", action="store_true")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
