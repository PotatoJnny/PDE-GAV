#!/usr/bin/env python
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import sys
import time
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
from agent_verifier.neural_generator.structured_ns_distribution import (
    StructuredCoefficientPCA,
    compatible_projection_signatures,
    load_structured_coefficient_corpus,
)
from agent_verifier.neural_generator.triplane_dataset import (
    build_kite_baffle_geometry,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit train-only PCA on structured NS coefficients and audit held-out "
            "physical reconstruction."
        )
    )
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ranks", default="32,64,128,256,384,512")
    parser.add_argument("--train-cases", type=int, default=0)
    parser.add_argument("--validation-cases", type=int, default=0)
    parser.add_argument("--batch-cases", type=int, default=16)
    parser.add_argument("--loader-workers", type=int, default=8)
    parser.add_argument("--oversample", type=int, default=16)
    parser.add_argument("--power-iterations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    ranks = sorted(
        {
            int(value)
            for value in re.split(r"[,;:]", str(args.ranks))
            if value.strip()
        }
    )
    if not ranks or min(ranks) < 1:
        raise SystemExit("PCA ranks must be positive")
    if min(int(args.batch_cases), int(args.loader_workers)) < 1:
        raise SystemExit("batch size and loader workers must be positive")
    output = Path(args.out).resolve()
    if output.exists():
        raise SystemExit(f"refusing to reuse PCA audit output: {output}")
    output.mkdir(parents=True)
    device = _device(str(args.device))
    started = time.perf_counter()

    train = load_structured_coefficient_corpus(
        args.train_manifest,
        max_cases=int(args.train_cases),
        device=device,
    )
    validation = load_structured_coefficient_corpus(
        args.validation_manifest,
        max_cases=int(args.validation_cases),
        device=device,
    )
    train_signature = dict(train.manifest["encoding_signature"])
    validation_signature = dict(validation.manifest["encoding_signature"])
    if not compatible_projection_signatures(train_signature, validation_signature):
        raise SystemExit("train and validation coefficient projections are incompatible")
    maximum_rank = min(train.case_count - 1, int(train.coefficients[0].numel()))
    if ranks[-1] > maximum_rank:
        raise SystemExit(
            f"largest requested rank {ranks[-1]} exceeds fit bound {maximum_rank}"
        )

    fit_started = time.perf_counter()
    pca = StructuredCoefficientPCA.fit(
        train.coefficients,
        rank=ranks[-1],
        source_signature=train_signature,
        oversample=int(args.oversample),
        power_iterations=int(args.power_iterations),
        seed=int(args.seed),
    )
    fit_seconds = float(time.perf_counter() - fit_started)
    checkpoint = output / "structured_coefficient_pca.pt"
    pca.save(checkpoint)

    projector, d0_rows = _build_physical_audit_context(
        validation,
        device=device,
    )
    rank_reports = []
    for rank in ranks:
        rank_started = time.perf_counter()
        selected = pca.with_rank(rank)
        reconstructed = selected.reconstruct(validation.coefficients)
        coefficient_report = _coefficient_reconstruction(
            validation.coefficients,
            reconstructed,
            validation.families,
        )
        physical_report = _physical_reconstruction(
            original_coefficients=validation.coefficients,
            reconstructed_coefficients=reconstructed,
            case_ids=validation.case_ids,
            families=validation.families,
            projector=projector,
            d0_rows=d0_rows,
            batch_cases=int(args.batch_cases),
            loader_workers=int(args.loader_workers),
            device=device,
        )
        row = {
            "rank": int(rank),
            "explained_variance_fraction": pca.explained_variance_fraction(rank),
            "coefficient_reconstruction": coefficient_report,
            "physical_reconstruction": physical_report,
            "elapsed_seconds": float(time.perf_counter() - rank_started),
        }
        rank_reports.append(row)
        print(
            "[structured-pca] "
            f"rank={rank} explained={row['explained_variance_fraction']:.6f} "
            "manifold_velocity_median="
            f"{physical_report['manifold_velocity_relative_l2']['median']:.6g} "
            "d0_velocity_median="
            f"{physical_report['d0_velocity_relative_l2']['median']:.6g}",
            flush=True,
        )

    train_scores = pca.encode(train.coefficients)
    validation_scores = pca.encode(validation.coefficients)
    report = {
        "schema": "ns2d_structured_coefficient_pca_audit_v1",
        "train_manifest": str(Path(args.train_manifest).resolve()),
        "validation_manifest": str(Path(args.validation_manifest).resolve()),
        "train_cases": train.case_count,
        "validation_cases": validation.case_count,
        "coefficient_shape": list(train.coefficient_shape),
        "field_scale": [float(value) for value in pca.field_scale.reshape(-1).cpu()],
        "fit_seconds": fit_seconds,
        "elapsed_seconds": float(time.perf_counter() - started),
        "checkpoint": checkpoint.name,
        "score_diagnostics": {
            "train": _score_diagnostics(train_scores),
            "validation": _score_diagnostics(validation_scores),
        },
        "ranks": rank_reports,
    }
    (output / "pca_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (output / "pca_audit.md").write_text(_markdown(report))


def _build_physical_audit_context(corpus, *, device):
    signature = corpus.manifest["encoding_signature"]
    split_manifest_path = Path(signature["split_manifest"]).resolve()
    split_manifest = json.loads(split_manifest_path.read_text())
    row_by_id = {
        str(row["case_id"]): row
        for rows in split_manifest.get("splits", {}).values()
        for row in rows
    }
    missing = [case_id for case_id in corpus.case_ids if case_id not in row_by_id]
    if missing:
        raise ValueError(f"D0 split manifest is missing {len(missing)} audit cases")
    d0_rows = {
        case_id: _resolve_case_dir(
            row_by_id[case_id],
            split_manifest_path=split_manifest_path,
        )
        for case_id in corpus.case_ids
    }
    first = _load_d0_case(d0_rows[corpus.case_ids[0]])
    config = StructuredNSProjectionConfig.from_dict(
        dict(signature["projection_config"])
    )
    projector = StructuredNSCoefficientProjector(
        geometry=build_kite_baffle_geometry(),
        t=first["t"],
        config=config,
        device=device,
    )
    return projector, d0_rows


def _coefficient_reconstruction(original, reconstructed, families):
    torch = _torch()
    rows = []
    for field_index, name in enumerate(("psi", "pressure")):
        reference = original[:, field_index].flatten(1)
        error = (reconstructed[:, field_index] - original[:, field_index]).flatten(1)
        values = _relative_l2_rows(error, reference).detach().cpu().numpy()
        rows.append(
            (
                name,
                {
                    "relative_l2": _summary(values),
                    "by_family": _by_family(values, families),
                },
            )
        )
    del torch
    return dict(rows)


def _physical_reconstruction(
    *,
    original_coefficients,
    reconstructed_coefficients,
    case_ids,
    families,
    projector,
    d0_rows,
    batch_cases: int,
    loader_workers: int,
    device,
):
    values: dict[str, list[float]] = {
        "manifold_velocity_relative_l2": [],
        "manifold_initial_velocity_relative_l2": [],
        "manifold_pressure_gauge_relative_l2": [],
        "d0_projection_velocity_relative_l2": [],
        "d0_velocity_relative_l2": [],
        "d0_initial_velocity_relative_l2": [],
        "d0_pressure_gauge_relative_l2": [],
    }
    for start in range(0, len(case_ids), batch_cases):
        end = min(start + batch_cases, len(case_ids))
        with ThreadPoolExecutor(max_workers=loader_workers) as pool:
            d0_cases = list(
                pool.map(
                    _load_d0_case,
                    [d0_rows[case_id] for case_id in case_ids[start:end]],
                )
            )
        original_velocity, original_pressure = _decode_projected_fields(
            original_coefficients[start:end],
            projector=projector,
        )
        reconstructed_velocity, reconstructed_pressure = _decode_projected_fields(
            reconstructed_coefficients[start:end],
            projector=projector,
        )
        d0_velocity = _torch().as_tensor(
            np.stack([case["velocity"] for case in d0_cases]),
            dtype=original_velocity.dtype,
            device=device,
        )
        d0_pressure = _torch().as_tensor(
            np.stack([case["pressure"] for case in d0_cases]),
            dtype=original_pressure.dtype,
            device=device,
        )
        fluid = projector.fluid
        d0_velocity = _stack_fluid_velocity(d0_velocity, fluid)
        d0_pressure = d0_pressure[:, :, fluid]
        d0_pressure = d0_pressure - d0_pressure.mean(dim=-1, keepdim=True)
        metrics = {
            "manifold_velocity_relative_l2": _relative_l2_rows(
                reconstructed_velocity - original_velocity,
                original_velocity,
            ),
            "manifold_initial_velocity_relative_l2": _relative_l2_rows(
                reconstructed_velocity[:, 0] - original_velocity[:, 0],
                original_velocity[:, 0],
            ),
            "manifold_pressure_gauge_relative_l2": _relative_l2_rows(
                reconstructed_pressure - original_pressure,
                original_pressure,
            ),
            "d0_projection_velocity_relative_l2": _relative_l2_rows(
                original_velocity - d0_velocity,
                d0_velocity,
            ),
            "d0_velocity_relative_l2": _relative_l2_rows(
                reconstructed_velocity - d0_velocity,
                d0_velocity,
            ),
            "d0_initial_velocity_relative_l2": _relative_l2_rows(
                reconstructed_velocity[:, 0] - d0_velocity[:, 0],
                d0_velocity[:, 0],
            ),
            "d0_pressure_gauge_relative_l2": _relative_l2_rows(
                reconstructed_pressure - d0_pressure,
                d0_pressure,
            ),
        }
        for name, metric in metrics.items():
            values[name].extend(float(value) for value in metric.detach().cpu())
    return {
        name: {
            **_summary(metric_values),
            "by_family": _by_family(metric_values, families),
        }
        for name, metric_values in values.items()
    }


def _decode_projected_fields(coefficients, *, projector):
    batch_size = int(coefficients.shape[0])
    spatial_count = int(projector.config.spatial_coefficient_count)
    values = coefficients.reshape(batch_size, 2, -1, spatial_count).to(
        dtype=projector.temporal_design.dtype,
        device=projector.device,
    )
    psi_at_times = _torch().einsum(
        "tr,brs->bts",
        projector.temporal_design,
        values[:, 0],
    )
    pressure_at_times = _torch().einsum(
        "tr,brs->bts",
        projector.temporal_design,
        values[:, 1],
    )
    velocity = _torch().einsum(
        "fs,bts->btf",
        projector.spatial_operators["velocity_design"],
        psi_at_times,
    )
    pressure = _torch().einsum(
        "fs,bts->btf",
        projector.spatial_operators["pressure_design_centered"],
        pressure_at_times,
    )
    return velocity, pressure


def _stack_fluid_velocity(velocity, fluid):
    return _torch().cat(
        (velocity[:, :, 0, fluid], velocity[:, :, 1, fluid]),
        dim=-1,
    )


def _load_d0_case(case_dir: Path):
    with np.load(case_dir / "hidden_truth_cache.npz") as hidden:
        return {
            "velocity": np.asarray(hidden["u_true"], dtype=np.float32),
            "pressure": np.asarray(hidden["pressure"], dtype=np.float32),
            "t": np.asarray(hidden["t"], dtype=np.float64),
        }


def _resolve_case_dir(row: dict[str, Any], *, split_manifest_path: Path) -> Path:
    case_id = str(row["case_id"])
    raw = Path(str(row.get("case_dir", "")))
    for candidate in (raw, ROOT / raw, split_manifest_path.parent / "cases" / case_id):
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"could not resolve D0 case {case_id}")


def _relative_l2_rows(error, reference):
    error_flat = error.reshape(error.shape[0], -1)
    reference_flat = reference.reshape(reference.shape[0], -1)
    return _torch().linalg.vector_norm(error_flat, dim=1) / _torch().linalg.vector_norm(
        reference_flat,
        dim=1,
    ).clamp_min(1.0e-12)


def _score_diagnostics(scores) -> dict[str, float]:
    values = scores.detach().float()
    coordinate_mean = values.mean(dim=0)
    coordinate_std = values.std(dim=0, unbiased=False)
    case_rms = values.square().mean(dim=1).sqrt()
    return {
        "coordinate_abs_mean_max": float(coordinate_mean.abs().max().cpu()),
        "coordinate_std_mean": float(coordinate_std.mean().cpu()),
        "coordinate_std_min": float(coordinate_std.min().cpu()),
        "coordinate_std_max": float(coordinate_std.max().cpu()),
        "case_rms_mean": float(case_rms.mean().cpu()),
        "case_rms_iqr": float(
            (_torch().quantile(case_rms, 0.75) - _torch().quantile(case_rms, 0.25)).cpu()
        ),
    }


def _by_family(values, families) -> dict[str, dict[str, float]]:
    array = np.asarray(values, dtype=np.float64)
    family_array = np.asarray(families)
    return {
        family: _summary(array[family_array == family])
        for family in sorted(set(families))
    }


def _summary(values) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "q90": float(np.quantile(array, 0.9)),
        "max": float(np.max(array)),
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Structured NS coefficient PCA audit",
        "",
        f"- train cases: `{report['train_cases']}`",
        f"- validation cases: `{report['validation_cases']}`",
        f"- fit seconds: `{report['fit_seconds']:.3f}`",
        "",
        "| rank | explained | manifold velocity median | D0 velocity median | D0 pressure median |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in report["ranks"]:
        physical = row["physical_reconstruction"]
        lines.append(
            f"| {row['rank']} | {row['explained_variance_fraction']:.6f} | "
            f"{physical['manifold_velocity_relative_l2']['median']:.6g} | "
            f"{physical['d0_velocity_relative_l2']['median']:.6g} | "
            f"{physical['d0_pressure_gauge_relative_l2']['median']:.6g} |"
        )
    return "\n".join(lines) + "\n"


def _torch():
    import torch

    return torch


def _device(requested: str):
    torch = _torch()
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {requested}, but CUDA is unavailable")
    return device


if __name__ == "__main__":
    main()
