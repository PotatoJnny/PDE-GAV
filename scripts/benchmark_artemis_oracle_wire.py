#!/usr/bin/env python3
"""Offline codec-only comparison for Artemis-sized exact-source RPC arrays.

Uses the existing AgentWireCodec at 64/128 KiB thresholds; no engine or Backend
configuration is changed. Payloads are synthetic 4096x3 FP64 coordinates and
4096 FP64 replies, not physical model evaluations. Full roundtrip below means
encode + JSON encode/decode + decode in both directions, with no HTTP, process
IPC, GPU, queueing, or solver work. Artifacts are preserved in new temp folders.
"""

from __future__ import annotations

import argparse
import base64
from hashlib import sha256
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

from pde_agent_verifier_solver.engine.agent_wire import (  # noqa: E402
    AgentWireCodec, AgentWireCodecError,
)


SCHEMA = "artemis-oracle-wire-microbenchmark-v1"
POINT_COUNT = 4096
THRESHOLDS = (64 * 1024, 128 * 1024)


def make_payload(iteration: int) -> tuple[np.ndarray, np.ndarray]:
    """Match one 16x256 source chunk; every iteration has a distinct t layer."""
    rows = np.arange(POINT_COUNT)
    points = np.empty((POINT_COUNT, 3), dtype=np.float64)
    points[:, 0] = iteration / 128.0
    points[:, 1] = (rows // 256) * (2.0 * np.pi / 64.0)
    points[:, 2] = -6.0 + (rows % 256) * (12.0 / 256.0)
    reply = points[:, 0] + 0.125 * points[:, 1] - 0.25 * points[:, 2]
    return points, reply


def _require_exact(actual, expected) -> None:
    if (not isinstance(actual, np.ndarray) or actual.shape != expected.shape
            or actual.dtype != expected.dtype or actual.tobytes() != expected.tobytes()):
        raise AssertionError("codec changed numeric bytes, dtype, shape, or row order")


def _json_dump(value) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True,
                      allow_nan=False).encode("utf-8")


def roundtrip(sender, receiver, points, reply) -> dict[str, object]:
    """Time unchanged codec operations; validate numeric bytes after timing."""
    times = {}

    def timed(name, operation):
        began = perf_counter()
        value = operation()
        times[name + "_seconds"] = perf_counter() - began
        return value

    began = perf_counter()
    request_node = timed("request_encode", lambda: sender.encode((points,)))
    request_bytes = timed("request_json_encode", lambda: _json_dump(request_node))
    request_read = timed("request_json_decode", lambda: json.loads(request_bytes))
    received = timed("request_decode", lambda: receiver.decode(request_read))
    response_node = timed("response_encode", lambda: receiver.encode(reply))
    response_bytes = timed("response_json_encode", lambda: _json_dump(response_node))
    response_read = timed("response_json_decode", lambda: json.loads(response_bytes))
    returned = timed("response_decode", lambda: sender.decode(response_read))
    times["full_roundtrip_seconds"] = perf_counter() - began
    _require_exact(received[0], points)
    _require_exact(returned, reply)
    times["encode_seconds"] = times["request_encode_seconds"] + times["response_encode_seconds"]
    times["decode_seconds"] = times["request_decode_seconds"] + times["response_decode_seconds"]
    return {
        **times, "request_json_bytes": len(request_bytes),
        "response_json_bytes": len(response_bytes), "bitwise_equal": True,
        "request_payload_sha256": sha256(points.tobytes()).hexdigest(),
        "reply_payload_sha256": sha256(reply.tobytes()).hexdigest(),
    }


def verify_safety(directory: Path, threshold: int) -> dict[str, bool]:
    """Exercise finite-value and NPY shape checks outside timed samples."""
    codec = AgentWireCodec(directory, inline_array_bytes=threshold)
    points, reply = make_payload(0)
    nonfinite = points.copy()
    nonfinite[0, 0] = np.nan
    try:
        codec.encode(nonfinite)
    except AgentWireCodecError:
        pass
    else:
        raise AssertionError("codec accepted nonfinite request values")
    response = dict(codec.encode(reply))
    response["data"] = base64.b64encode(np.full_like(reply, np.inf).tobytes()).decode("ascii")
    try:
        codec.decode(response)
    except AgentWireCodecError:
        pass
    else:
        raise AssertionError("codec accepted nonfinite response values")
    # Shape remains part of the wire contract; checksum metadata is optional.
    node = dict(codec.encode(np.arange(32768, dtype=np.float64)))
    node["sha256"] = "stale metadata"
    np.testing.assert_array_equal(codec.decode(node), np.arange(32768, dtype=np.float64))
    node["shape"] = [32767]
    try:
        codec.decode(node)
    except AgentWireCodecError:
        pass
    else:
        raise AssertionError("codec accepted a sidecar with a different shape")
    return {"nonfinite_request_rejected": True, "nonfinite_response_rejected": True,
            "sidecar_shape_mismatch_rejected": True}


def summarize(samples: list[dict[str, object]]) -> dict[str, object]:
    keys = tuple(key for key in samples[0] if key.endswith("_seconds"))
    return {
        key: {"median": float(np.median([row[key] for row in samples])),
              "p95": float(np.percentile([row[key] for row in samples], 95))}
        for key in keys
    }


def benchmark_filesystem(parent: Path, *, iterations: int, warmup: int) -> dict[str, object]:
    """Create only a fresh private child; preserve successes and failed probes."""
    parent = parent.expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix="artemis-oracle-wire-", dir=parent))
    report = {"parent": str(parent), "workspace": str(workspace),
              "filesystem_device": workspace.stat().st_dev, "status": "failed"}
    try:
        variants = {}
        codecs = {}
        for threshold in THRESHOLDS:
            root = workspace / str(threshold)
            safety = verify_safety(root / "safety", threshold)
            warm_sender = AgentWireCodec(root / "warmup", inline_array_bytes=threshold)
            warm_receiver = AgentWireCodec(root / "warmup", inline_array_bytes=threshold)
            for index in range(warmup):
                roundtrip(warm_sender, warm_receiver, *make_payload(-index - 1))
            codecs[threshold] = (
                AgentWireCodec(root / "measured", inline_array_bytes=threshold),
                AgentWireCodec(root / "measured", inline_array_bytes=threshold),
            )
            variants[str(threshold)] = {"inline_array_bytes": threshold,
                                        "safety": safety, "samples": []}
        for index in range(iterations):
            points, reply = make_payload(index)
            # Alternate order to reduce systematic first/second-run bias.
            order = THRESHOLDS if index % 2 == 0 else tuple(reversed(THRESHOLDS))
            for threshold in order:
                sample = roundtrip(*codecs[threshold], points, reply)
                variants[str(threshold)]["samples"].append({"iteration": index, **sample})
        for threshold in THRESHOLDS:
            variant = variants[str(threshold)]
            files = tuple((workspace / str(threshold) / "measured").glob("*.npy"))
            variant.update({
                "timing_seconds": summarize(variant["samples"]),
                "measured_sidecar_count": len(files),
                "measured_sidecar_bytes": sum(path.stat().st_size for path in files),
                "warmup_and_safety_files_excluded": True,
            })
        baseline, comparison = (variants[str(threshold)] for threshold in THRESHOLDS)
        for a, b in zip(baseline["samples"], comparison["samples"]):
            if (a["request_payload_sha256"], a["reply_payload_sha256"]) != (
                b["request_payload_sha256"], b["reply_payload_sha256"]
            ):
                raise AssertionError("threshold variants used different numeric payloads")
        median_a = baseline["timing_seconds"]["full_roundtrip_seconds"]["median"]
        median_b = comparison["timing_seconds"]["full_roundtrip_seconds"]["median"]
        report.update({"status": "passed", "variants": variants,
                       "median_codec_roundtrip_ratio_64_over_128": median_a / median_b})
    except Exception as error:
        report.update({"error_type": type(error).__name__, "error": str(error)})
        if "variants" in locals():
            report["variants"] = variants
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch-parent", type=Path, required=True)
    parser.add_argument("--local-parent", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--allow-local-smoke", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.iterations <= 512 or not 0 <= args.warmup <= 20:
        parser.error("iterations must be 1..512 and warmup 0..20")
    if not os.environ.get("SLURM_JOB_ID"):
        if not args.allow_local_smoke or args.iterations > 2 or args.warmup != 0:
            parser.error("run on a Slurm compute allocation; local smoke requires <=2 iterations and --warmup 0")
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error("refusing to overwrite an existing benchmark report")
    local = args.local_parent or (Path(os.environ["SLURM_TMPDIR"])
                                  if os.environ.get("SLURM_TMPDIR") else None)
    report = {
        "schema": SCHEMA, "status": "failed", "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "python_version": sys.version.split()[0], "numpy_version": np.__version__,
        "benchmark_script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "codec_source_sha256": sha256(
            (ROOT / "src/pde_agent_verifier_solver/engine/agent_wire.py").read_bytes()
        ).hexdigest(),
        "benchmark_scope": "codec_microbenchmark_not_http_or_end_to_end",
        "model_execution_performed": False, "deployment_changed": False,
        "synthetic_payloads": True, "point_count": POINT_COUNT,
        "request_shape": [POINT_COUNT, 3], "response_shape": [POINT_COUNT],
        "dtype": "float64", "request_numeric_bytes": POINT_COUNT * 3 * 8,
        "response_numeric_bytes": POINT_COUNT * 8, "iterations": args.iterations,
        "warmup_iterations_per_variant": args.warmup,
        "payload_schedule": "unique entire t layer each iteration; identical for both thresholds",
        "timed_scope": "both directions: codec encode/decode plus JSON encode/decode; excludes fixture generation, equality checks, safety probes, and directory setup",
        "artifact_policy": "preserved; warmup and deliberately corrupted safety probes are separate from measured files",
        "inline_integrity_scope": "bitwise numeric roundtrip and finite validation; inline protocol has no independent payload digest",
        "filesystems": {},
    }
    for label, parent in (("scratch", args.scratch_parent), ("local", local)):
        if parent is None:
            report["filesystems"][label] = {"status": "skipped", "reason": "SLURM_TMPDIR unset and --local-parent not supplied"}
            continue
        try:
            report["filesystems"][label] = benchmark_filesystem(
                parent, iterations=args.iterations, warmup=args.warmup,
            )
        except Exception as error:
            report["filesystems"][label] = {"status": "failed", "error_type": type(error).__name__, "error": str(error)}
    rows = tuple(report["filesystems"].values())
    report["status"] = "failed" if any(row["status"] == "failed" for row in rows) else "passed"
    if all("filesystem_device" in row for row in rows):
        report["scratch_and_local_same_device"] = rows[0]["filesystem_device"] == rows[1]["filesystem_device"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "report": str(output),
                      "scope": report["benchmark_scope"]}), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
