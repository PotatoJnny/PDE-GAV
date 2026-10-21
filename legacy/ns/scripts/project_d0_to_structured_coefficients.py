#!/usr/bin/env python
from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent_verifier.neural_generator.structured_ns_answer_first import (
    StructuredNSCoefficientProjector,
    StructuredNSProjectionConfig,
)
from agent_verifier.neural_generator.triplane_dataset import (
    build_kite_baffle_geometry,
)


PROJECTION_METRICS = (
    "velocity_relative_l2",
    "initial_velocity_relative_l2",
    "pressure_gauge_relative_l2",
    "velocity_spatial_only_relative_l2",
    "pressure_spatial_only_gauge_relative_l2",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project a frozen D0 split into structured continuous coefficients."
    )
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--split", choices=("train", "validation", "d0_distribution_test"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--batch-cases", type=int, default=32)
    parser.add_argument("--shard-cases", type=int, default=256)
    parser.add_argument("--loader-workers", type=int, default=8)
    parser.add_argument("--spatial-mode-x", type=int, default=16)
    parser.add_argument("--spatial-mode-y", type=int, default=16)
    parser.add_argument("--time-order", type=int, default=12)
    parser.add_argument("--geometry-mask-transition", type=float, default=0.05)
    parser.add_argument("--ridge-relative", type=float, default=1.0e-6)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if min(int(args.batch_cases), int(args.shard_cases), int(args.loader_workers)) < 1:
        raise SystemExit("batch, shard, and loader sizes must be positive")
    manifest_path = Path(args.split_manifest).resolve()
    split_manifest = json.loads(manifest_path.read_text())
    split_rows = list(split_manifest.get("splits", {}).get(str(args.split), []))
    if not split_rows:
        raise SystemExit(f"split {args.split!r} is missing or empty")
    if int(args.max_cases) > 0:
        split_rows = split_rows[: int(args.max_cases)]
    case_ids = [str(row["case_id"]) for row in split_rows]
    if len(case_ids) != len(set(case_ids)):
        raise SystemExit("selected split contains duplicate case IDs")
    case_dirs = [
        _resolve_case_dir(row, split_manifest_path=manifest_path)
        for row in split_rows
    ]
    config = StructuredNSProjectionConfig(
        spatial_mode_x=int(args.spatial_mode_x),
        spatial_mode_y=int(args.spatial_mode_y),
        time_order=int(args.time_order),
        geometry_mask_transition=float(args.geometry_mask_transition),
        ridge_relative=float(args.ridge_relative),
    )
    config.validate()
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    shard_root = output / "shards"
    signature = {
        "schema": "ns2d_structured_coefficient_dataset_signature_v2",
        "split_manifest": str(manifest_path),
        "split_manifest_sha256": _file_sha256(manifest_path),
        "split": str(args.split),
        "selected_case_count": len(case_ids),
        "selected_case_ids_sha256": _strings_sha256(case_ids),
        "projection_config": config.to_dict(),
        "coefficient_shape": [
            2,
            config.time_coefficient_count,
            int(config.spatial_mode_x) + 1,
            int(config.spatial_mode_y) + 1,
        ],
    }
    _require_signature(output / "encoding_signature.json", signature)
    shard_root.mkdir(exist_ok=True)
    completed_manifest = output / "coefficient_manifest.json"
    if completed_manifest.exists():
        payload = json.loads(completed_manifest.read_text())
        if payload.get("encoding_signature") != signature:
            raise SystemExit("completed coefficient manifest has a different signature")
        print(json.dumps({"out": str(output), "reused_complete_manifest": True}, indent=2))
        return

    geometry = build_kite_baffle_geometry()
    first = _load_case(case_dirs[0], expected_geometry_hash=geometry.contract_hash)
    projector = StructuredNSCoefficientProjector(
        geometry=geometry,
        t=first["t"],
        config=config,
        device=_device(str(args.device)),
    )
    coefficient_shape = tuple(signature["coefficient_shape"])
    moments = _StreamingMoments(coefficient_shape)
    metric_values: dict[str, list[float]] = {name: [] for name in PROJECTION_METRICS}
    shard_rows: list[dict[str, Any]] = []
    progress_path = output / "projection_progress.jsonl"
    started = time.perf_counter()
    # Reusing one pool prevents glibc from retaining a new set of large NumPy
    # decompression arenas for every shard.
    loader_pool = ThreadPoolExecutor(
        max_workers=int(args.loader_workers)
    )

    for shard_start in range(0, len(case_dirs), int(args.shard_cases)):
        shard_end = min(shard_start + int(args.shard_cases), len(case_dirs))
        shard_index = shard_start // int(args.shard_cases)
        shard_path = shard_root / f"coefficients_{shard_index:05d}.npz"
        expected_ids = case_ids[shard_start:shard_end]
        if shard_path.exists():
            arrays = _load_ready_shard(
                shard_path,
                expected_ids=expected_ids,
                coefficient_shape=coefficient_shape,
            )
            reused = True
        else:
            arrays = _project_shard(
                projector=projector,
                case_dirs=case_dirs[shard_start:shard_end],
                expected_geometry_hash=geometry.contract_hash,
                batch_cases=int(args.batch_cases),
                loader_pool=loader_pool,
            )
            _atomic_savez(shard_path, **arrays)
            reused = False
        moments.update(arrays["coefficients"])
        for name in PROJECTION_METRICS:
            metric_values[name].extend(float(value) for value in arrays[name])
        shard_row = {
            "shard_index": int(shard_index),
            "start": int(shard_start),
            "end": int(shard_end),
            "count": int(shard_end - shard_start),
            "path": str(Path("shards") / shard_path.name),
            "sha256": _file_sha256(shard_path),
            "reused": bool(reused),
        }
        shard_rows.append(shard_row)
        _append_jsonl(
            progress_path,
            {
                "event": "shard_complete",
                **shard_row,
                "completed_cases": int(shard_end),
                "total_cases": len(case_dirs),
                "elapsed_seconds": float(time.perf_counter() - started),
            },
        )
        print(
            "[structured-coefficients] "
            f"shard={shard_index} completed={shard_end}/{len(case_dirs)} "
            f"reused={reused} elapsed_seconds={time.perf_counter() - started:.3f}",
            flush=True,
        )
        del arrays
        _release_unused_memory(projector)

    loader_pool.shutdown(wait=True)
    mean, std = moments.finalize()
    normalizer_path = output / "coefficient_normalizer.npz"
    _atomic_savez(
        normalizer_path,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        count=np.asarray(moments.count, dtype=np.int64),
    )
    report = {
        "schema": "ns2d_structured_coefficient_dataset_v2",
        "encoding_signature": signature,
        "source_split_size": len(split_manifest["splits"][str(args.split)]),
        "encoded_cases": len(case_dirs),
        "coefficient_shape": list(coefficient_shape),
        "coefficient_count": int(np.prod(coefficient_shape)),
        "coefficient_dtype": "float32",
        "normalizer": {
            "path": normalizer_path.name,
            "sha256": _file_sha256(normalizer_path),
            "fit_split": str(args.split),
            "per_coefficient": True,
            "casewise_normalization": False,
            "std_zero_count": int(np.count_nonzero(std <= 0.0)),
            "std_below_1e_12_count": int(np.count_nonzero(std <= 1.0e-12)),
        },
        "projection": {
            name: _summary(values) for name, values in metric_values.items()
        },
        "operator_diagnostics": projector.operator_diagnostics,
        "shards": shard_rows,
        "elapsed_seconds": float(time.perf_counter() - started),
        "cases_per_second": float(len(case_dirs) / max(time.perf_counter() - started, 1.0e-12)),
    }
    completed_manifest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


def _project_shard(
    *,
    projector,
    case_dirs: list[Path],
    expected_geometry_hash: str,
    batch_cases: int,
    loader_pool: ThreadPoolExecutor,
) -> dict[str, np.ndarray]:
    coefficient_parts = []
    metric_parts: dict[str, list[float]] = {name: [] for name in PROJECTION_METRICS}
    case_ids = []
    families = []
    for start in range(0, len(case_dirs), int(batch_cases)):
        paths = case_dirs[start : start + int(batch_cases)]
        loaded = list(
            loader_pool.map(
                lambda path: _load_case(
                    path,
                    expected_geometry_hash=expected_geometry_hash,
                ),
                paths,
            )
        )
        coefficients, report = projector.project(
            velocity=np.stack([item["velocity"] for item in loaded]),
            pressure=np.stack([item["pressure"] for item in loaded]),
        )
        coefficient_parts.append(
            coefficients.detach().cpu().numpy().astype(np.float32)
        )
        case_ids.extend(item["case_id"] for item in loaded)
        families.extend(item["family"] for item in loaded)
        for row in report["cases"]:
            for name in PROJECTION_METRICS:
                metric_parts[name].append(float(row[name]))
    arrays: dict[str, np.ndarray] = {
        "coefficients": np.concatenate(coefficient_parts),
        "case_ids": np.asarray(case_ids),
        "families": np.asarray(families),
    }
    arrays.update(
        {
            name: np.asarray(values, dtype=np.float32)
            for name, values in metric_parts.items()
        }
    )
    return arrays


def _release_unused_memory(projector: StructuredNSCoefficientProjector) -> None:
    """Return per-shard temporary arrays instead of retaining their arenas."""

    gc.collect()
    if projector.device.type == "cuda":
        projector.torch.cuda.empty_cache()
    try:
        libc = ctypes.CDLL(None)
        malloc_trim = getattr(libc, "malloc_trim", None)
        if malloc_trim is not None:
            malloc_trim(0)
    except OSError:
        # malloc_trim is a Linux/glibc optimization; correctness does not rely on it.
        pass


def _load_case(case_dir: Path, *, expected_geometry_hash: str) -> dict[str, Any]:
    with np.load(case_dir / "hidden_truth_cache.npz") as hidden:
        velocity = np.asarray(hidden["u_true"], dtype=np.float64)
        pressure = np.asarray(hidden["pressure"], dtype=np.float64)
        t = np.asarray(hidden["t"], dtype=np.float64)
    question = json.loads((case_dir / "public_question.json").read_text())
    actual_hash = str(question.get("geometry", {}).get("contract_hash", ""))
    if actual_hash != str(expected_geometry_hash):
        raise ValueError(f"{case_dir.name}: unexpected geometry contract {actual_hash}")
    spec = json.loads((case_dir / "case_spec.json").read_text())
    return {
        "case_id": case_dir.name,
        "family": str(spec.get("family", "unknown")),
        "velocity": velocity,
        "pressure": pressure,
        "t": t,
    }


def _load_ready_shard(
    path: Path,
    *,
    expected_ids: list[str],
    coefficient_shape: tuple[int, ...],
) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    if list(arrays.get("case_ids", ())) != list(expected_ids):
        raise RuntimeError(f"resumable shard has the wrong case IDs: {path}")
    expected_shape = (len(expected_ids), *coefficient_shape)
    if tuple(arrays.get("coefficients", np.empty(0)).shape) != expected_shape:
        raise RuntimeError(f"resumable shard has the wrong coefficient shape: {path}")
    missing = sorted(set(PROJECTION_METRICS) - set(arrays))
    if missing:
        raise RuntimeError(f"resumable shard is missing metrics {missing}: {path}")
    return arrays


class _StreamingMoments:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.count = 0
        self.sum = np.zeros(shape, dtype=np.float64)
        self.sum_squared = np.zeros(shape, dtype=np.float64)

    def update(self, values) -> None:
        array = np.asarray(values, dtype=np.float64)
        self.count += int(array.shape[0])
        self.sum += np.sum(array, axis=0)
        self.sum_squared += np.sum(np.square(array), axis=0)

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count < 1:
            raise ValueError("cannot finalize empty coefficient moments")
        mean = self.sum / float(self.count)
        variance = np.maximum(self.sum_squared / float(self.count) - mean**2, 0.0)
        return mean, np.sqrt(variance)


def _resolve_case_dir(row: dict[str, Any], *, split_manifest_path: Path) -> Path:
    case_id = str(row["case_id"])
    raw = Path(str(row.get("case_dir", "")))
    candidates = [raw, ROOT / raw, split_manifest_path.parent / "cases" / case_id]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"could not resolve D0 case {case_id}")


def _require_signature(path: Path, signature: dict[str, Any]) -> None:
    if path.exists():
        if json.loads(path.read_text()) != signature:
            raise SystemExit(f"existing output has a different encoding signature: {path}")
        return
    if any(path.parent.iterdir()):
        raise SystemExit(f"nonempty output is missing an encoding signature: {path.parent}")
    path.write_text(json.dumps(signature, indent=2, sort_keys=True) + "\n")


def _atomic_savez(path: Path, **arrays) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
            temporary_path = Path(handle.name)
        np.savez_compressed(temporary_path, **arrays)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _summary(values) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "q90": float(np.quantile(array, 0.9)),
        "max": float(np.max(array)),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strings_sha256(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _device(requested: str):
    import torch

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {requested}, but CUDA is unavailable")
    return device


if __name__ == "__main__":
    main()
