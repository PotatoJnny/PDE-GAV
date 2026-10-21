#!/usr/bin/env python
"""Build batched answer-first VPFP development and released-solution banks."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import torch

from agent_verifier.vpfp.continuous_cases import (
    materialize_continuous_case,
    verify_continuous_case_dir,
)
from agent_verifier.vpfp.continuous_flow_training import (
    load_hermite_autoencoder_checkpoint,
)
from agent_verifier.vpfp.continuous_hermite import VPFPContinuousHermiteConfig
from agent_verifier.vpfp.continuous_macros import MACRO_OUTPUT_CONTRACT
from agent_verifier.vpfp.high_fidelity import VPFPHighFidelityConfig
from agent_verifier.vpfp.online_training import (
    VPFPOnlineBatchGenerator,
    VPFPOnlineGenerationConfig,
)
from agent_verifier.vpfp.operators import VPFPGrid, VPFPTime
from agent_verifier.vpfp.runtime_paths import is_persistent_run_path


REPORT_SCHEMA = "vpfp_long_time_answer_first_release_bank_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder-checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--regime-id", required=True)
    parser.add_argument("--collision-nu-min", type=float, required=True)
    parser.add_argument("--collision-nu-max", type=float, required=True)
    parser.add_argument("--temperature-min", type=float, default=0.8)
    parser.add_argument("--temperature-max", type=float, default=1.2)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--development-count", type=int, default=16)
    parser.add_argument("--released-count", type=int, default=32)
    parser.add_argument("--generation-batch-size", type=int, default=16)
    parser.add_argument("--coarse-nx", type=int, default=16)
    parser.add_argument("--coarse-ny", type=int, default=16)
    parser.add_argument("--coarse-nv", type=int, default=21)
    parser.add_argument("--coarse-vmax", type=float, default=8.0)
    parser.add_argument("--internal-dt-max", type=float, default=0.001)
    parser.add_argument("--materialized-nx", type=int, default=33)
    parser.add_argument("--materialized-ny", type=int, default=33)
    parser.add_argument("--materialized-nv", type=int, default=9)
    parser.add_argument("--materialized-nt", type=int, default=13)
    parser.add_argument("--materialized-vmax", type=float, default=5.0)
    parser.add_argument("--materialization-chunk-size", type=int, default=262144)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_TMPDIR"):
        raise SystemExit("VPFP release-bank generation requires a Slurm node")
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise SystemExit("VPFP release-bank generation requires CUDA")
    output = Path(args.out).resolve()
    if output.exists() or not is_persistent_run_path(output):
        raise SystemExit("--out must be fresh and below the configured run storage root")
    total_count = int(args.development_count) + int(args.released_count)
    batch_size = int(args.generation_batch_size)
    if (
        args.development_count < 1
        or not 1 <= args.released_count <= 32
        or batch_size < 1
        or total_count % batch_size != 0
        or args.collision_nu_min < 0.0
        or args.collision_nu_max <= args.collision_nu_min
        or args.temperature_min <= 0.0
        or args.temperature_max <= args.temperature_min
    ):
        raise SystemExit("invalid VPFP release-bank schedule")
    checkpoint = Path(args.decoder_checkpoint).resolve()
    if not checkpoint.is_file() or checkpoint.is_symlink():
        raise FileNotFoundError(checkpoint)

    output.mkdir(parents=True)
    cases_root = output / "cases"
    cases_root.mkdir()
    started = time.perf_counter()
    _encoder, decoder, payload = load_hermite_autoencoder_checkpoint(
        checkpoint,
        device=str(args.device),
    )
    decoder.eval()
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    adapter = dict(payload.get("robust_answer_first_adapter") or {})
    if not (
        adapter.get("unused_neural_parameters_zeroed") is True
        and adapter.get("continuous_hidden_answer_is_primary") is True
        and adapter.get("generator_parameterization_public") is False
    ):
        raise SystemExit("release-bank decoder is not a qualified answer-first adapter")
    config = VPFPContinuousHermiteConfig.from_dict(payload["decoder_config"])
    expected_regime = (
        tuple(float(value) for value in config.collision_nu_range),
        tuple(float(value) for value in config.temperature_range),
    )
    requested_regime = (
        (float(args.collision_nu_min), float(args.collision_nu_max)),
        (float(args.temperature_min), float(args.temperature_max)),
    )
    if expected_regime != requested_regime:
        raise SystemExit("release-bank checkpoint belongs to a different regime")

    equation = VPFPHighFidelityConfig(
        internal_dt_max=float(args.internal_dt_max),
        collision_nu_range=requested_regime[0],
        temperature_range=requested_regime[1],
    )
    generation = VPFPOnlineGenerationConfig(
        batch_size=batch_size,
        nx=int(args.coarse_nx),
        ny=int(args.coarse_ny),
        velocity_grid_size=int(args.coarse_nv),
        nt=int(config.structured_time_control_count),
        t1=float(config.time_domain[1]),
        velocity_max=float(args.coarse_vmax),
        internal_dt_max=float(args.internal_dt_max),
        cache_capacity_cases=max(64, batch_size),
        anchor_case_count=batch_size,
    )
    generator = VPFPOnlineBatchGenerator(
        decoder_config=config,
        generation_config=generation,
        equation_config=equation,
        device=torch.device(args.device),
    )
    grid = VPFPGrid(
        nx=int(args.materialized_nx),
        ny=int(args.materialized_ny),
        nvx=int(args.materialized_nv),
        nvy=int(args.materialized_nv),
        x_domain=tuple(float(value) for value in config.equation.x_domain),
        v_bounds=(
            -float(args.materialized_vmax),
            float(args.materialized_vmax),
            -float(args.materialized_vmax),
            float(args.materialized_vmax),
        ),
    )
    time_grid = VPFPTime(
        t0=float(config.time_domain[0]),
        t1=float(config.time_domain[1]),
        nt=int(args.materialized_nt),
    )
    grid.validate()
    time_grid.validate()
    torch.cuda.reset_peak_memory_stats()

    rows: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []
    seeds = np.arange(
        int(args.seed_start),
        int(args.seed_start) + total_count,
        dtype=np.int64,
    )
    for batch_index, offset in enumerate(range(0, total_count, batch_size)):
        batch_seeds = seeds[offset : offset + batch_size]
        generation_started = time.perf_counter()
        try:
            batch = generator.generate(batch_seeds)
        except Exception as exc:
            _write_json(
                output / "failure.json",
                {
                    "stage": "batch_generation",
                    "batch_index": batch_index,
                    "seeds": batch_seeds.tolist(),
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                },
            )
            raise
        generation_rows.append(
            {
                "batch_index": batch_index,
                "seeds": batch_seeds.tolist(),
                "wall_time_seconds": float(time.perf_counter() - generation_started),
                "solver_wall_time_seconds": float(
                    batch["diagnostics"]["solver"]["solve_wall_time_seconds"]
                ),
                "latent_shape": list(batch["latents"].shape),
            }
        )
        for local_index, seed in enumerate(batch_seeds):
            global_index = offset + local_index
            split = (
                "development"
                if global_index < int(args.development_count)
                else "released_completed"
            )
            case_id = f"{args.regime_id}_{split}_{global_index:06d}"
            case_dir = cases_root / case_id
            row: dict[str, Any] = {
                "index": global_index,
                "seed": int(seed),
                "split": split,
                "case_id": case_id,
                "case_dir": str(case_dir),
                "accepted": False,
            }
            try:
                condition = (
                    batch["equation_conditions"][local_index]
                    .detach()
                    .cpu()
                    .numpy()
                )
                materialize_continuous_case(
                    case_dir,
                    case_id=case_id,
                    model=decoder,
                    decoder_checkpoint=checkpoint,
                    latent=batch["latents"][local_index],
                    grid=grid,
                    time=time_grid,
                    equation_condition=condition,
                    source_origin="generated_continuous_autodiff_residual",
                    source_derivative_backend="analytic",
                    solver_output_contract=MACRO_OUTPUT_CONTRACT,
                    generator_metadata={
                        "generator_family": "coarse_projection_answer_first",
                        "generator_route": "batched_long_time_release_bank_v2",
                        "generator_parameterization_public": False,
                    },
                    case_spec={
                        "regime_id": str(args.regime_id),
                        "split": split,
                        "truth_primary": "continuous_hidden_answer",
                        "grid_arrays_are_access_views": True,
                        "low_fidelity_equation_solve_is_truth": False,
                        "public_source_is_hidden_answer_residual": True,
                    },
                    evaluation_chunk_size=int(args.materialization_chunk_size),
                )
                audit = verify_continuous_case_dir(
                    case_dir,
                    device=str(args.device),
                    truth_sample_count=8,
                    source_sample_count=8,
                )
                row["accepted"] = bool(audit["accepted"])
                row["audit_failures"] = list(audit["failures"])
                row["audit_metrics"] = audit["metrics"]
            except Exception as exc:
                row["exception_type"] = type(exc).__name__
                row["exception_message"] = str(exc)[:2000]
            rows.append(row)
            _write_json(
                output / "progress.json",
                {
                    "schema": "vpfp_long_time_release_bank_progress_v2",
                    "completed_case_count": len(rows),
                    "accepted_case_count": sum(bool(item["accepted"]) for item in rows),
                    "last_row": row,
                },
            )

    development = [
        row["case_dir"]
        for row in rows
        if row["split"] == "development" and row["accepted"]
    ]
    released = [
        row["case_dir"]
        for row in rows
        if row["split"] == "released_completed" and row["accepted"]
    ]
    passed = bool(
        len(development) == int(args.development_count)
        and len(released) == int(args.released_count)
    )
    report = {
        "schema": REPORT_SCHEMA,
        "passed": passed,
        "regime": {
            "regime_id": str(args.regime_id),
            "collision_nu_range": list(requested_regime[0]),
            "temperature_range": list(requested_regime[1]),
            "time_domain": list(config.time_domain),
        },
        "answer_first_contract": {
            "continuous_hidden_answer_is_primary": True,
            "public_source_is_exact_jet_residual": True,
            "coarse_equation_first_solve_is_shape_seed_only": True,
            "grid_arrays_are_numerical_access_views": True,
        },
        "decoder_checkpoint": str(checkpoint),
        "generation_batch_size": batch_size,
        "development_case_dirs": development,
        "released_completed_case_dirs": released,
        "rows": rows,
        "generation_batches": generation_rows,
        "total_wall_time_seconds": float(time.perf_counter() - started),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    _write_json(output / "release_bank_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not passed:
        raise SystemExit("VPFP long-time release bank failed acceptance")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


if __name__ == "__main__":
    main()
