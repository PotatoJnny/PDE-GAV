#!/usr/bin/env python
"""Serve resource-metered continuous answer-first VPFP development cases."""

from __future__ import annotations

import argparse
from collections import deque
import json
import os
import shutil
import signal
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

from agent_verifier.storage_lifecycle import (
    is_within_active_slurm_tmp,
    sha256_file,
)
from agent_verifier.vpfp.answer_first_online_protocol import (
    AGGREGATE_LIMITS_ACCOUNTING_ONLY,
    validate_answer_first_controller_config,
    validate_answer_first_materialization_resolution,
)
from agent_verifier.vpfp.compute_budget import COMPUTE_BUDGET_CONTRACT
from agent_verifier.vpfp.continuous_cases import (
    materialize_continuous_case,
    verify_continuous_case_dir,
)
from agent_verifier.vpfp.guided_scoring import (
    GuidanceSolverCompatibilityError,
    guidance_selection_rewards,
    load_guidance_solver,
    normalize_guidance_solver_budget,
    score_guidance_candidate,
)
from agent_verifier.vpfp.continuous_flow_training import (
    load_hermite_autoencoder_checkpoint,
)
from agent_verifier.vpfp.continuous_macros import MACRO_OUTPUT_CONTRACT
from agent_verifier.vpfp.continuous_oracle import (
    VPFPContinuousPrivateCase,
    VPFPOracleQueryLimits,
)
from agent_verifier.vpfp.high_fidelity import VPFPHighFidelityConfig
from agent_verifier.vpfp.online_generator_protocol import (
    ONLINE_GENERATOR_REQUEST_SCHEMA,
    audit_online_output,
    export_numerical_online_case,
)
from agent_verifier.vpfp.online_training import (
    VPFPOnlineBatchGenerator,
    VPFPOnlineGenerationConfig,
)
from agent_verifier.vpfp.operators import VPFPGrid, VPFPTime
from agent_verifier.vpfp.oracle_training import inspect_solver_bundle
from agent_verifier.vpfp.resolution_convergence import (
    SCALABLE_RESOLUTION_CONTRACT,
    VPFPResolution,
)
from run_vpfp_continuous_online_generator_service import (
    RequestHandler,
    UnixServer,
    _load_points,
    _query_output_path,
    _variants,
    _workspace_file,
)


SOURCE_ORIGIN = "generated_continuous_autodiff_residual"
EXACT_MACRO_FILENAME = "exact_macro_answer.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _initialize_public_output_root(workspace: Path) -> Path:
    output_root = workspace / "online_generator_outputs"
    if output_root.is_symlink():
        physical_root = output_root.resolve(strict=True)
        if (
            not is_within_active_slurm_tmp(physical_root)
            or not physical_root.is_dir()
            or any(physical_root.iterdir())
        ):
            raise RuntimeError(
                "online output alias must target one empty active SLURM_TMPDIR directory"
            )
    else:
        output_root.mkdir(parents=True, exist_ok=False)
    return output_root


def _relative_public_output(path: str | Path, output_root: Path) -> str:
    relative = Path(path).resolve(strict=False).relative_to(
        output_root.resolve(strict=True)
    )
    return str(Path("online_generator_outputs") / relative)


class AnswerFirstOnlineGeneratorState:
    def __init__(self, config_path: Path, *, device: str) -> None:
        self.config_path = config_path.resolve()
        self.config = validate_answer_first_controller_config(
            json.loads(self.config_path.read_text(encoding="utf-8"))
        )
        self.workspace = Path(self.config["workspace"]).resolve()
        self.private_root = Path(self.config["private_root"]).resolve()
        self.private_root.mkdir(parents=True, exist_ok=False)
        self.private_cases = self.private_root / "cases"
        self.private_cases.mkdir()
        self.candidate_root = self.private_root / "candidate_scratch"
        self.candidate_root.mkdir()
        self.output_root = _initialize_public_output_root(self.workspace)
        (self.output_root / "cases").mkdir()
        (self.output_root / "queries").mkdir()
        readme = self.output_root / "README.md"
        readme.write_text(
            "# Online VPFP Development Samples\n\n"
            "Each public equation is manufactured from a fresh continuous hidden "
            "answer. Numerical files are query views only. Cases generated here "
            "are training-only and excluded from private selection and final "
            "evaluation.\n",
            encoding="utf-8",
        )
        self.registered_output_paths: set[Path] = {
            readme,
            self.output_root / "manifest.json",
        }

        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("answer-first online generation requires a GPU")
        torch.set_float32_matmul_precision("high")
        decoder_checkpoint = Path(self.config["decoder_checkpoint"]).resolve()
        _encoder, self.decoder, _payload = load_hermite_autoencoder_checkpoint(
            decoder_checkpoint, device=str(self.device)
        )
        self.decoder.eval()
        for parameter in self.decoder.parameters():
            parameter.requires_grad_(False)
        self.decoder_checkpoint_sha256 = sha256_file(decoder_checkpoint)
        self.decoder_checkpoint = decoder_checkpoint

        generation_payload = dict(self.config["generation_config"])
        generation_payload["cache_capacity_cases"] = max(
            int(generation_payload.get("cache_capacity_cases", 32)),
            int(generation_payload["batch_size"]),
        )
        self.generation_config = VPFPOnlineGenerationConfig(
            **generation_payload
        )
        self.equation_config = VPFPHighFidelityConfig.from_dict(
            dict(self.config["equation_config"])
        )
        if not np.allclose(
            self.decoder.config.time_domain,
            (0.0, float(self.generation_config.t1)),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("decoder and coarse shape generator time domains differ")
        if int(self.decoder.config.structured_time_control_count) != int(
            self.generation_config.nt
        ):
            raise ValueError("decoder time controls do not match coarse frames")
        self.generator = VPFPOnlineBatchGenerator(
            decoder_config=self.decoder.config,
            generation_config=self.generation_config,
            equation_config=self.equation_config,
            device=self.device,
        )
        self.grid = VPFPGrid(**dict(self.config["materialization_grid"]))
        self.time = VPFPTime(**dict(self.config["materialization_time"]))
        self.grid.validate()
        self.time.validate()

        self.target_solver_path = Path(self.config["target_solver"]).resolve()
        target_bundle = inspect_solver_bundle(self.target_solver_path)
        if str(target_bundle["aggregate_sha256"]) != str(
            self.config["target_solver_sha256"]
        ):
            raise ValueError("online controller targets a different solver bundle")
        self.regime = dict(self.config["regime"])
        self.guidance = dict(self.config["guidance"])
        self.guidance_budget = normalize_guidance_solver_budget(
            self.guidance["solver_budget"]
        )
        self.target_solver = load_guidance_solver(self.target_solver_path)
        self.limits = dict(self.config["limits"])
        self.resource_budget = dict(
            self.config.get("development_resource_budget") or {}
        )
        self.resource_mode = bool(self.resource_budget)
        self.resolution_policy = dict(
            self.config.get("resolution_materialization") or {}
        )
        self.compute_budget_contract_enabled = bool(
            self.config.get("compute_budget_contract_enabled")
        )
        self.seed_next = {
            name: int(value)
            for name, value in dict(self.config["seed_starts"]).items()
        }
        self.seed_step = int(self.config.get("seed_step", 1))

        self.sample_calls = 0
        self.sample_count = 0
        self.candidate_answer_count = 0
        self.candidate_answer_generated_count = 0
        self.prefetch_cache_hits = 0
        self.prefetch_generation_calls = 0
        self.prefetch_cache = {
            name: deque() for name in self.seed_next
        }
        self.candidate_materialization_count = 0
        self.materialization_integrity_audit_count = 0
        self.selected_candidate_reuse_count = 0
        self.guidance_score_accepted_count = 0
        self.guidance_score_failed_count = 0
        self.answer_count = 0
        self.query_calls = 0
        self.query_points = 0
        self.resolution_view_count = 0
        self.resolution_target_elements = 0
        self.materialization_generator_calls = 0
        self.private_materialization_eviction_count = 0
        self.raw_trajectory_bytes_discarded = 0
        self.operation_usage = {
            name: {
                "calls": 0,
                "wall_time_seconds": 0.0,
                "process_cpu_seconds": 0.0,
            }
            for name in (
                "sample",
                "materialize",
                "materialize_batch",
                "query",
                "query_batch",
            )
        }
        self.records: dict[str, dict[str, Any]] = {}
        self.private_log_path = self.private_root / "request_log.jsonl"
        self.private_log_path.write_text("", encoding="utf-8")
        import threading

        self.lock = threading.Lock()
        self.started = time.time()
        self.started_monotonic = time.monotonic()
        self.storage_bytes_used = 0
        self.storage_scan_wall_time_seconds = 0.0
        self._refresh_storage_usage()
        self._write_public_manifest()

    def dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        if request.get("schema") != ONLINE_GENERATOR_REQUEST_SCHEMA:
            raise ValueError("unexpected request schema")
        operation = str(request.get("operation") or "")
        if operation == "status":
            return self.status()
        handler = {
            "sample": self.sample,
            "materialize": self.materialize,
            "materialize_batch": self.materialize_batch,
            "query": self.query,
            "query_batch": self.query_batch,
        }.get(operation)
        if handler is None:
            raise ValueError("unsupported operation")
        self._assert_resources_available()
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        try:
            response = handler(request)
        finally:
            usage = self.operation_usage[operation]
            usage["calls"] += 1
            usage["wall_time_seconds"] += time.perf_counter() - wall_started
            usage["process_cpu_seconds"] += time.process_time() - cpu_started
            self._refresh_storage_usage()
        response["budgets"] = self._budget_report()
        self._write_public_manifest()
        return response

    def status(self) -> dict[str, Any]:
        return {
            "target_solver_sha256": self.config["target_solver_sha256"],
            "solver_output_contract_version": MACRO_OUTPUT_CONTRACT,
            "regime": self.regime,
            "budgets": self._budget_report(),
            "available_condition_pools": {
                "train": "fresh disjoint hidden-answer shape seeds",
                "validation": "fresh disjoint hidden-answer shape seeds",
            },
            "dynamic_output_root": "online_generator_outputs",
            "primary_truth": "continuous hidden answer",
            "public_equation_construction": (
                "source is the analytic coordinate-jet residual of that answer"
            ),
            "coarse_numerical_solve_role": "shape seed only; not truth",
            "guided_sampler": (
                "direct frozen-solver error best of fresh continuous-answer pool"
            ),
            "guidance_solver_budget": self.guidance_budget.to_dict(),
            "guided_all_failure_behavior": (
                "reject request as target-solver incompatibility"
            ),
            "model_parameterization_exposed": False,
            "continuous_point_oracle_available": True,
            "exact_macro_point_oracle_available": True,
            "batch_operations_available": True,
            "batch_operations": ["materialize_batch", "query_batch"],
            "compute_budget_contract_enabled": bool(
                self.compute_budget_contract_enabled
            ),
            "resolution_materialization_available": bool(
                self.resolution_policy.get("enabled")
            ),
            "resolution_materialization": self.resolution_policy,
        }

    def sample(self, request: dict[str, Any]) -> dict[str, Any]:
        count = int(request.get("count", 0))
        mode = str(request.get("mode") or "")
        answers = str(request.get("answers") or "")
        pool_name = str(request.get("condition_pool") or "")
        if count < 1 or count > int(self.limits["max_batch"]):
            raise ValueError("sample count exceeds the per-call limit")
        if not self.resource_mode and self.sample_count + count > int(
            self.limits["max_samples"]
        ):
            raise ValueError("total sample budget would be exceeded")
        if mode not in {"base", "guided", "mixed"}:
            raise ValueError("sample mode must be base, guided, or mixed")
        if answers not in {"none", "grid"}:
            raise ValueError("answers must be none or grid")
        if (
            not self.resource_mode
            and answers == "grid"
            and self.answer_count + count > int(self.limits["max_answer_grids"])
        ):
            raise ValueError("answer-grid budget would be exceeded")
        if pool_name not in self.seed_next:
            raise ValueError("unknown condition pool")

        variants = _variants(mode, count)
        pool_size = int(self.guidance["candidate_pool_size"])
        candidate_counts = [
            pool_size if variant == "guided" else 1 for variant in variants
        ]
        offsets = np.concatenate(
            (np.zeros(1, dtype=np.int64), np.cumsum(candidate_counts))
        )
        candidate_count = int(offsets[-1])
        if (
            not self.resource_mode
            and self.candidate_answer_count + candidate_count
            > int(self.limits["max_candidate_answers"])
        ):
            raise ValueError("candidate answer budget would be exceeded")
        seeds, latents, conditions = self._take_candidate_answers(
            pool_name, candidate_count
        )
        self.candidate_answer_count += candidate_count

        rows = []
        request_log = []
        for row_index, variant in enumerate(variants):
            start = int(offsets[row_index])
            stop = int(offsets[row_index + 1])
            scores = []
            score_results = []
            sequence = self.sample_count + 1
            public_id = f"online_{sequence:06d}"
            private_id = f"controller_online_{sequence:06d}"
            private_case = self.private_cases / private_id
            if variant == "guided":
                candidate_dirs = []
                for local_index in range(start, stop):
                    candidate_dir = self.candidate_root / (
                        f"request_{self.sample_calls + 1:04d}_row_{row_index:03d}_"
                        f"candidate_{local_index - start:02d}"
                    )
                    candidate_dirs.append(candidate_dir)
                    self._materialize_private_case(
                        candidate_dir,
                        case_id=private_id,
                        latent=latents[local_index],
                        condition=conditions[local_index].numpy(),
                        grid=self.grid,
                        time_grid=self.time,
                        variant="guided",
                        cohort="agent_online_training",
                        candidate_pool_size=pool_size,
                    )
                    self.candidate_materialization_count += 1
                    result = score_guidance_candidate(
                        self.target_solver,
                        candidate_dir,
                        budget=self.guidance_budget,
                    )
                    score_results.append(result)
                    if result["accepted"]:
                        self.guidance_score_accepted_count += 1
                    else:
                        self.guidance_score_failed_count += 1
                try:
                    scores = guidance_selection_rewards(
                        score_results,
                        context=(
                            f"online request {self.sample_calls + 1}, "
                            f"row {row_index}"
                        ),
                    )
                except GuidanceSolverCompatibilityError as exc:
                    self._private_log(
                        {
                            "operation": "guidance_compatibility_failure",
                            "public_case_id": public_id,
                            "error": str(exc),
                            "results": score_results,
                            "guidance_solver_budget": (
                                self.guidance_budget.to_dict()
                            ),
                        }
                    )
                    for candidate_dir in candidate_dirs:
                        shutil.rmtree(candidate_dir, ignore_errors=True)
                    raise
                selected = start + int(
                    np.argmax(np.asarray(scores, dtype=float))
                )
                for local_index, candidate_dir in zip(
                    range(start, stop), candidate_dirs, strict=True
                ):
                    if local_index == selected:
                        shutil.move(str(candidate_dir), str(private_case))
                    else:
                        shutil.rmtree(candidate_dir)
                self.selected_candidate_reuse_count += 1
            else:
                selected = start
                self._materialize_private_case(
                    private_case,
                    case_id=private_id,
                    latent=latents[selected],
                    condition=conditions[selected].numpy(),
                    grid=self.grid,
                    time_grid=self.time,
                    variant=variant,
                    cohort="agent_online_training",
                    candidate_pool_size=1,
                )
                self.candidate_materialization_count += 1

            selected_latent = latents[selected].contiguous()
            selected_condition = conditions[selected].numpy()
            base_resolution = VPFPResolution.from_grid_time(self.grid, self.time)
            self._attach_resolution_contract(
                private_case,
                root_case_id=public_id,
                parent_case_id=None,
                resolution=base_resolution,
                study_id=f"{public_id}_base",
                dimension="base",
                level=0,
                characteristic_spacing=1.0,
            )
            exported = export_numerical_online_case(
                private_case,
                self.output_root / "cases" / public_id,
                public_case_id=public_id,
                include_answer=answers == "grid",
            )
            self._register_export(exported)
            shutil.rmtree(private_case)
            self.private_materialization_eviction_count += 1
            record = {
                "latent": selected_latent,
                "condition": selected_condition,
                "variant": variant,
                "answers": answers,
                "condition_pool": pool_name,
                "shape_seed": int(seeds[selected]),
                "root_case_id": public_id,
                "parent_case_id": None,
                "resolution_view": False,
                "resolution": base_resolution.to_dict(),
                "study_id": f"{public_id}_base",
                "dimension": "base",
                "level": 0,
                "characteristic_spacing": 1.0,
            }
            self.records[public_id] = record
            self.sample_count += 1
            if answers == "grid":
                self.answer_count += 1
            rows.append(
                self._public_row(
                    public_id,
                    exported=exported,
                    record=record,
                    exact_macro_answer=None,
                    truth_times=None,
                )
            )
            request_log.append(
                {
                    "candidate_shape_seeds": [
                        int(value) for value in seeds[start:stop]
                    ],
                    "selected_shape_seed": int(seeds[selected]),
                    "public_case_id": public_id,
                    "candidate_count": stop - start,
                    "selected_candidate_offset": selected - start,
                    "selection_rewards": scores,
                    "guidance_score_results": score_results,
                    "variant": variant,
                }
            )
        self.sample_calls += 1
        self._private_log(
            {
                "operation": "sample",
                "mode": mode,
                "answers": answers,
                "condition_pool": pool_name,
                "rows": request_log,
            }
        )
        return {"cases": rows}

    def _take_candidate_answers(
        self, pool_name: str, count: int
    ) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """Return compact answer codes while batching coarse solves on the GPU."""

        cache = self.prefetch_cache[pool_name]
        cached_before = len(cache)
        generation_batch = int(self.generation_config.batch_size)
        missing = max(0, count - cached_before)
        if missing:
            generation_count = missing
            if self.resource_mode:
                generation_count = (
                    (missing + generation_batch - 1) // generation_batch
                ) * generation_batch
            seed_start = int(self.seed_next[pool_name])
            maximum_seed = seed_start + (generation_count - 1) * self.seed_step
            if maximum_seed > np.iinfo(np.int64).max:
                raise OverflowError("online shape seed stream exhausted int64")
            generated_seeds = (
                seed_start
                + np.arange(generation_count, dtype=np.int64) * self.seed_step
            )
            self.seed_next[pool_name] = (
                seed_start + generation_count * self.seed_step
            )
            for start in range(0, generation_count, generation_batch):
                batch_seeds = generated_seeds[start : start + generation_batch]
                batch = self.generator.generate(batch_seeds)
                latents = batch["latents"].detach().cpu()
                conditions = batch["equation_conditions"].detach().cpu()
                for offset, seed in enumerate(batch_seeds):
                    cache.append(
                        (
                            int(seed),
                            latents[offset].contiguous(),
                            conditions[offset].contiguous(),
                        )
                    )
                self.raw_trajectory_bytes_discarded += int(
                    batch["diagnostics"]["raw_f_trajectory_bytes_discarded"]
                )
                self.prefetch_generation_calls += 1
            self.candidate_answer_generated_count += generation_count

        self.prefetch_cache_hits += min(count, cached_before)
        selected = [cache.popleft() for _ in range(count)]
        seeds = np.asarray([row[0] for row in selected], dtype=np.int64)
        latents = torch.stack([row[1] for row in selected], dim=0)
        conditions = torch.stack([row[2] for row in selected], dim=0)
        return seeds, latents, conditions

    def materialize(self, request: dict[str, Any]) -> dict[str, Any]:
        return {"cases": self._materialize_requests([request])}

    def materialize_batch(self, request: dict[str, Any]) -> dict[str, Any]:
        raw = request.get("requests")
        if not isinstance(raw, list) or not raw:
            raise ValueError("materialize_batch requires a nonempty request list")
        if len(raw) > int(self.limits["max_batch"]):
            raise ValueError("materialize_batch exceeds the per-call batch limit")
        if not all(isinstance(row, dict) for row in raw):
            raise ValueError("every materialize_batch row must be an object")
        return {"cases": self._materialize_requests([dict(row) for row in raw])}

    def _materialize_requests(
        self, requests: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if self.resolution_policy.get("enabled") is not True:
            raise ValueError("resolution materialization is not enabled")
        plans = []
        pending_answer_count = 0
        pending_elements = 0
        for offset, request in enumerate(requests):
            source_id = str(request.get("case_id") or "")
            if source_id not in self.records:
                raise ValueError("materialize case_id is not an online sample")
            answers = str(request.get("answers") or "")
            if answers not in {"none", "grid"}:
                raise ValueError("answers must be none or grid")
            resolution = validate_answer_first_materialization_resolution(
                dict(request.get("resolution") or {}),
                policy=self.resolution_policy,
            )
            if resolution.nx < 4 * int(self.decoder.config.rho_max_mode_x) + 1:
                raise ValueError("requested nx aliases the hidden rho Fourier support")
            if resolution.ny < 4 * int(self.decoder.config.rho_max_mode_y) + 1:
                raise ValueError("requested ny aliases the hidden rho Fourier support")
            truth_times = _truth_times(
                request.get("truth_times"),
                time_domain=self.decoder.config.time_domain,
            )
            if truth_times is not None and answers != "grid":
                raise ValueError("truth_times require answers=grid")
            source_record = self.records[source_id]
            root_id = str(source_record["root_case_id"])
            sequence = self.resolution_view_count + offset + 1
            public_id = f"online_view_{sequence:06d}"
            dimension = _resolution_dimension(
                VPFPResolution.from_dict(dict(source_record["resolution"])),
                resolution,
            )
            plans.append(
                {
                    "source_id": source_id,
                    "source_record": source_record,
                    "public_id": public_id,
                    "private_id": f"controller_{public_id}",
                    "root_id": root_id,
                    "answers": answers,
                    "resolution": resolution,
                    "truth_times": truth_times,
                    "study_id": _study_id(
                        request.get("study_id"),
                        default=f"{root_id}_materialization",
                    ),
                    "dimension": dimension,
                    "level": _materialization_level(
                        request.get("level"), fallback=sequence
                    ),
                    "characteristic_spacing": _characteristic_spacing(
                        VPFPResolution.from_dict(dict(source_record["resolution"])),
                        resolution,
                        dimension=dimension,
                    ),
                }
            )
            pending_elements += resolution.source_element_count
            pending_answer_count += int(answers == "grid")
        if not self.resource_mode:
            if self.resolution_view_count + len(plans) > int(
                self.resolution_policy["max_views"]
            ):
                raise ValueError("resolution-view budget would be exceeded")
            if self.resolution_target_elements + pending_elements > int(
                self.resolution_policy["max_total_target_source_elements"]
            ):
                raise ValueError("resolution source-element budget would be exceeded")
            if self.answer_count + pending_answer_count > int(
                self.limits["max_answer_grids"]
            ):
                raise ValueError("answer-grid budget would be exceeded")

        rows = []
        self.materialization_generator_calls += 1
        for plan in plans:
            resolution = plan["resolution"]
            grid = VPFPGrid(
                nx=int(resolution.nx),
                ny=int(resolution.ny),
                nvx=int(resolution.nvx),
                nvy=int(resolution.nvy),
                x_domain=tuple(float(value) for value in self.grid.x_domain),
                v_bounds=tuple(float(value) for value in self.grid.v_bounds),
            )
            time_grid = VPFPTime(
                t0=float(self.time.t0),
                t1=float(self.time.t1),
                nt=int(resolution.nt),
            )
            private_case = self.private_cases / plan["private_id"]
            source_record = plan["source_record"]
            self._materialize_private_case(
                private_case,
                case_id=plan["private_id"],
                latent=source_record["latent"],
                condition=source_record["condition"],
                grid=grid,
                time_grid=time_grid,
                variant=str(source_record["variant"]),
                cohort="agent_online_resolution_development",
                candidate_pool_size=1,
            )
            self._attach_resolution_contract(
                private_case,
                root_case_id=plan["root_id"],
                parent_case_id=plan["source_id"],
                resolution=resolution,
                study_id=plan["study_id"],
                dimension=plan["dimension"],
                level=plan["level"],
                characteristic_spacing=plan["characteristic_spacing"],
            )
            exported = export_numerical_online_case(
                private_case,
                self.output_root / "cases" / plan["public_id"],
                public_case_id=plan["public_id"],
                include_answer=plan["answers"] == "grid",
            )
            self._register_export(exported)
            exact_path = None
            if plan["truth_times"] is not None:
                exact_path = self._write_exact_macro_answer(
                    Path(exported["case_dir"]),
                    latent=source_record["latent"],
                    condition=source_record["condition"],
                    grid=grid,
                    truth_times=plan["truth_times"],
                )
                self.registered_output_paths.add(exact_path)
            shutil.rmtree(private_case)
            self.private_materialization_eviction_count += 1
            record = {
                **source_record,
                "answers": plan["answers"],
                "root_case_id": plan["root_id"],
                "parent_case_id": plan["source_id"],
                "resolution_view": True,
                "resolution": resolution.to_dict(),
                "study_id": plan["study_id"],
                "dimension": plan["dimension"],
                "level": plan["level"],
                "characteristic_spacing": plan["characteristic_spacing"],
            }
            self.records[plan["public_id"]] = record
            self.resolution_view_count += 1
            self.resolution_target_elements += resolution.source_element_count
            self.answer_count += int(plan["answers"] == "grid")
            rows.append(
                self._public_row(
                    plan["public_id"],
                    exported=exported,
                    record=record,
                    exact_macro_answer=exact_path,
                    truth_times=plan["truth_times"],
                )
            )
            self._private_log(
                {
                    "operation": "materialize_batch",
                    "source_case_id": plan["source_id"],
                    "public_case_id": plan["public_id"],
                    "resolution": resolution.to_dict(),
                    "answers": plan["answers"],
                    "exact_truth_time_count": (
                        0
                        if plan["truth_times"] is None
                        else len(plan["truth_times"])
                    ),
                    "study_id": plan["study_id"],
                }
            )
        return rows

    def query(self, request: dict[str, Any]) -> dict[str, Any]:
        row = self._query_plan(request, sequence=self.query_calls + 1)
        if not self.resource_mode and self.query_points + row["count"] > int(
            self.limits["max_total_points"]
        ):
            raise ValueError("total point-query budget would be exceeded")
        result = self._execute_query(row)
        self.query_calls += 1
        self.query_points += row["count"]
        return result

    def query_batch(self, request: dict[str, Any]) -> dict[str, Any]:
        raw = request.get("requests")
        if not isinstance(raw, list) or not raw:
            raise ValueError("query_batch requires a nonempty request list")
        if len(raw) > int(self.limits["max_batch"]):
            raise ValueError("query_batch exceeds the per-call batch limit")
        plans = [
            self._query_plan(dict(row), sequence=self.query_calls + index + 1)
            for index, row in enumerate(raw)
            if isinstance(row, dict)
        ]
        if len(plans) != len(raw):
            raise ValueError("every query_batch row must be an object")
        total = sum(int(plan["count"]) for plan in plans)
        if total > int(self.limits["max_points_per_call"]):
            raise ValueError("query_batch exceeds the per-call point limit")
        if not self.resource_mode and self.query_points + total > int(
            self.limits["max_total_points"]
        ):
            raise ValueError("total point-query budget would be exceeded")
        rows = [self._execute_query(plan) for plan in plans]
        self.query_calls += len(plans)
        self.query_points += total
        return {"queries": rows}

    def _query_plan(self, request: dict[str, Any], *, sequence: int) -> dict[str, Any]:
        case_id = str(request.get("case_id") or "")
        field = str(request.get("field") or "")
        if case_id not in self.records:
            raise ValueError("query case_id is not an online sample")
        if field not in {"source", "rho0", "g0", "truth", "macro_truth"}:
            raise ValueError("unsupported query field")
        points_path = _workspace_file(
            self.workspace,
            request.get("points_file"),
            suffixes={".npy", ".npz"},
        )
        points = _load_points(points_path)
        count = int(points.shape[0])
        if count > int(self.limits["max_points_per_call"]):
            raise ValueError("point query exceeds the per-call limit")
        return {
            "case_id": case_id,
            "field": field,
            "points": points,
            "count": count,
            "out": request.get("out"),
            "sequence": int(sequence),
        }

    def _execute_query(self, plan: dict[str, Any]) -> dict[str, Any]:
        record = self.records[plan["case_id"]]
        private = self._private_case(record, max_points=int(plan["count"]))
        field = plan["field"]
        if field == "source":
            values = {"values": private.public_oracle().source(plan["points"])}
        elif field == "rho0":
            values = {"values": private.public_oracle().rho0(plan["points"])}
        elif field == "g0":
            values = {"values": private.public_oracle().g0(plan["points"])}
        elif field == "macro_truth":
            values = private.released_macros(plan["points"])
        else:
            values = private.released_truth(plan["points"])
        output = _query_output_path(
            self.output_root.resolve(strict=True),
            plan["out"],
            sequence=int(plan["sequence"]),
        )
        np.savez_compressed(output, points=plan["points"], **values)
        self.registered_output_paths.add(output)
        self._private_log(
            {
                "operation": "query",
                "case_id": plan["case_id"],
                "field": field,
                "point_count": int(plan["count"]),
                "output": str(output),
            }
        )
        return {
            "case_id": plan["case_id"],
            "field": field,
            "point_count": int(plan["count"]),
            "output": _relative_public_output(output, self.output_root),
            "keys": ["points", *values.keys()],
        }

    def _materialize_private_case(
        self,
        path: Path,
        *,
        case_id: str,
        latent,
        condition,
        grid: VPFPGrid,
        time_grid: VPFPTime,
        variant: str,
        cohort: str,
        candidate_pool_size: int,
        verify: bool = True,
    ) -> Path:
        result = materialize_continuous_case(
            path,
            case_id=case_id,
            model=self.decoder,
            decoder_checkpoint=self.decoder_checkpoint,
            latent=torch.as_tensor(latent).to(device=self.device),
            grid=grid,
            time=time_grid,
            equation_condition=np.asarray(condition, dtype=float),
            source_origin=SOURCE_ORIGIN,
            source_derivative_backend="analytic",
            solver_output_contract=MACRO_OUTPUT_CONTRACT,
            generator_metadata={
                "generator_route": "coarse_shape_to_continuous_answer_first_v1",
                "policy_variant": variant,
                "candidate_pool_size": int(candidate_pool_size),
                "target_solver_sha256": self.config["target_solver_sha256"],
            },
            case_spec={
                "cohort": cohort,
                "training_only": True,
                "eligible_for_private_selection": False,
                "eligible_for_final_evaluation": False,
                "hidden_answer_is_primary": True,
                "coarse_shape_seed_is_not_truth": True,
                "public_source_is_hidden_answer_residual": True,
            },
            evaluation_chunk_size=int(self.config["materialization_chunk_size"]),
        )
        if verify:
            self._verify_private_case(result)
        return result

    def _verify_private_case(self, path: Path) -> None:
        audit = verify_continuous_case_dir(
            path,
            device=str(self.device),
            truth_sample_count=4,
            source_sample_count=4,
        )
        if not audit["accepted"]:
            raise RuntimeError(f"online answer-first materialization failed: {audit}")
        self.materialization_integrity_audit_count += 1

    def _attach_resolution_contract(
        self,
        case_dir: Path,
        *,
        root_case_id: str,
        parent_case_id: str | None,
        resolution: VPFPResolution,
        study_id: str,
        dimension: str,
        level: int,
        characteristic_spacing: float,
    ) -> None:
        question_path = case_dir / "public_question.json"
        question = json.loads(question_path.read_text(encoding="utf-8"))
        request = {
            "schema": SCALABLE_RESOLUTION_CONTRACT,
            "study_id": study_id,
            "dimension": dimension,
            "level": int(level),
            "characteristic_spacing": float(characteristic_spacing),
            "resolution": resolution.to_dict(),
            "semantics": {
                "public_arrays_are_samples_not_primary_truth": True,
                "same_hidden_continuous_case_across_levels": True,
                "source_recomputed_from_same_hidden_answer": True,
                "solver_must_use_actual_case_grid": True,
            },
            "online_root_case_id": root_case_id,
            "parent_case_id": parent_case_id,
        }
        question.update(
            {
                "solver_resolution_contract_version": SCALABLE_RESOLUTION_CONTRACT,
                "resolution_request": request,
                "solver_contract": (
                    "define solve_at_resolution(public_case, resolution) and return "
                    "rho, momentum, and kinetic_energy on the requested grid"
                ),
            }
        )
        if self.compute_budget_contract_enabled:
            question.pop("solver_resolution_contract_version", None)
            development_view = question.pop("resolution_request")
            question.update(
                {
                    "solver_compute_budget_contract_version": COMPUTE_BUDGET_CONTRACT,
                    "time_query_contract": {
                        "query_times": (
                            "arbitrary finite 1D times in the public [t0, t1] interval"
                        ),
                        "output_time_axis": (
                            "exactly follows query_times; candidate owns interpolation"
                        ),
                        "internal_time_discretization": "candidate controlled",
                    },
                    "solver_contract": (
                        "define solve_with_budget(public_case, query_times, budget) "
                        "and return rho, momentum, and kinetic_energy at every requested "
                        "time; define solve(public_case) as a compatibility wrapper"
                    ),
                    "development_view": development_view,
                }
            )
        question_path.write_text(
            json.dumps(question, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _write_exact_macro_answer(
        self,
        public_case_dir: Path,
        *,
        latent,
        condition,
        grid: VPFPGrid,
        truth_times: np.ndarray,
    ) -> Path:
        x = np.linspace(0.0, float(grid.x_domain[0]), int(grid.nx), endpoint=False)
        y = np.linspace(0.0, float(grid.x_domain[1]), int(grid.ny), endpoint=False)
        mesh = np.meshgrid(truth_times, x, y, indexing="ij")
        points = np.column_stack([value.reshape(-1) for value in mesh])
        private = self._private_case(
            {"latent": latent, "condition": condition},
            max_points=max(1, int(points.shape[0])),
        )
        values = private.released_macros(points)
        shape = (len(truth_times), int(grid.nx), int(grid.ny))
        exact = {
            "rho": values["rho"].reshape(shape),
            "momentum": values["momentum"].reshape(*shape, 2),
            "kinetic_energy": values["kinetic_energy"].reshape(shape),
            "raw_second_moment": values["raw_second_moment"].reshape(*shape, 3),
        }
        path = public_case_dir / EXACT_MACRO_FILENAME
        np.savez_compressed(path, **exact)
        return path

    def _private_case(
        self, record: dict[str, Any], *, max_points: int
    ) -> VPFPContinuousPrivateCase:
        return VPFPContinuousPrivateCase(
            case_id="online_numerical_oracle",
            model=self.decoder,
            latent=torch.as_tensor(record["latent"]).to(device=self.device),
            equation_condition=np.asarray(record["condition"], dtype=float),
            limits=VPFPOracleQueryLimits(
                max_points_per_call=max(1, int(max_points)),
                max_total_points=max(1, int(max_points)),
                evaluation_chunk_size=int(self.config["oracle_chunk_size"]),
            ),
            source_derivative_backend="analytic",
        )

    def _register_export(self, exported: dict[str, Any]) -> None:
        self.registered_output_paths.update(
            {
                Path(exported["public_question"]),
                Path(exported["public_arrays"]),
            }
        )
        if exported["answer"] is not None:
            self.registered_output_paths.add(Path(exported["answer"]))

    def _public_row(
        self,
        case_id: str,
        *,
        exported: dict[str, Any],
        record: dict[str, Any],
        exact_macro_answer: Path | None,
        truth_times: np.ndarray | None,
    ) -> dict[str, Any]:
        return {
            "case_id": case_id,
            "root_case_id": str(record["root_case_id"]),
            "parent_case_id": record.get("parent_case_id"),
            "case_dir": _relative_public_output(
                exported["case_dir"], self.output_root
            ),
            "public_question": _relative_public_output(
                exported["public_question"], self.output_root
            ),
            "public_arrays": _relative_public_output(
                exported["public_arrays"], self.output_root
            ),
            "answer": (
                None
                if exported["answer"] is None
                else _relative_public_output(exported["answer"], self.output_root)
            ),
            "exact_macro_answer": (
                None
                if exact_macro_answer is None
                else _relative_public_output(exact_macro_answer, self.output_root)
            ),
            "truth_times": None if truth_times is None else truth_times.tolist(),
            "truth_time_semantics": (
                None
                if truth_times is None
                else "direct continuous hidden-function evaluation; no time interpolation"
            ),
            "resolution": dict(record["resolution"]),
            "study_id": str(record["study_id"]),
            "dimension": str(record["dimension"]),
            "level": int(record["level"]),
            "characteristic_spacing": float(record["characteristic_spacing"]),
            "policy_variant": str(record["variant"]),
            "condition_pool": str(record["condition_pool"]),
            "regime_id": str(self.regime["regime_id"]),
        }

    def finalize(self) -> dict[str, Any]:
        self._write_public_manifest()
        descriptor = self._write_compact_case_descriptors()
        audit = audit_online_output(
            self.output_root.resolve(strict=True),
            registered_files=sorted(
                path.resolve(strict=False)
                for path in self.registered_output_paths
            ),
            allow_missing_registered_files=True,
        )
        report = {
            "schema": "vpfp_answer_first_online_generator_service_report_v1",
            "target_solver_sha256": self.config["target_solver_sha256"],
            "solver_output_contract_version": MACRO_OUTPUT_CONTRACT,
            "decoder_checkpoint_sha256": self.decoder_checkpoint_sha256,
            "compact_case_descriptors": descriptor,
            "primary_truth": "continuous hidden answer",
            "source_origin": SOURCE_ORIGIN,
            "source_derivative_backend": "structured_fourier_hermite_jet_v1",
            "coarse_shape_seed_is_truth": False,
            "regime": self.regime,
            "guidance": self.guidance,
            "resolution_materialization": self.resolution_policy,
            "generation_config": self.generation_config.to_dict(),
            "equation_config": self.equation_config.to_dict(),
            "uptime_seconds": float(time.time() - self.started),
            "development_resource_budget": (
                self.resource_budget if self.resource_mode else None
            ),
            "actual_slurm_allocation": {
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "gpus": os.environ.get("SLURM_JOB_GPUS"),
                "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
                "memory_per_node": os.environ.get("SLURM_MEM_PER_NODE"),
            },
            "budget_usage": self._budget_report(),
            "registered_output_count": len(self.registered_output_paths),
            "output_audit": audit,
        }
        (self.private_root / "service_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return report

    def _assert_resources_available(self) -> None:
        if not self.resource_mode:
            return
        if time.monotonic() - self.started_monotonic >= float(
            self.resource_budget["active_wall_time_seconds"]
        ):
            raise RuntimeError("Agent development active-wall-time budget expired")
        self._refresh_storage_usage()
        if self.storage_bytes_used > int(self.resource_budget["scratch_bytes"]):
            raise RuntimeError("Agent development scratch budget exceeded")

    def _refresh_storage_usage(self) -> None:
        if not self.resource_mode:
            return
        started = time.perf_counter()
        roots: list[Path] = []
        for candidate in (
            self.workspace,
            self.private_root,
            self.output_root.resolve(strict=True),
        ):
            physical = Path(candidate).resolve(strict=True)
            if any(
                physical == root or physical.is_relative_to(root) for root in roots
            ):
                continue
            roots.append(physical)
        total = 0
        for root in roots:
            for directory, names, files in os.walk(root, followlinks=False):
                names[:] = [
                    name
                    for name in names
                    if not Path(directory, name).is_symlink()
                ]
                for name in files:
                    path = Path(directory, name)
                    if path.is_symlink():
                        continue
                    try:
                        total += path.stat().st_size
                    except FileNotFoundError:
                        continue
        self.storage_bytes_used = int(total)
        self.storage_scan_wall_time_seconds += time.perf_counter() - started

    def _aggregate_limit(self, name: str, *, resolution: bool = False) -> int | None:
        if self.resource_mode:
            return None
        source = self.resolution_policy if resolution else self.limits
        return int(source.get(name, 0))

    def _budget_report(self) -> dict[str, Any]:
        elapsed = float(time.monotonic() - self.started_monotonic)
        active_limit = (
            int(self.resource_budget["active_wall_time_seconds"])
            if self.resource_mode
            else None
        )
        return {
            "enforcement_basis": (
                "active_wall_time_allocation_scratch_and_per_request_safety"
                if self.resource_mode
                else "aggregate_count_limits"
            ),
            "aggregate_counts_mode": (
                AGGREGATE_LIMITS_ACCOUNTING_ONLY
                if self.resource_mode
                else "enforced"
            ),
            "sample_calls": int(self.sample_calls),
            "samples_used": int(self.sample_count),
            "samples_limit": self._aggregate_limit("max_samples"),
            "candidate_answers_used": int(self.candidate_answer_count),
            "candidate_answers_generated": int(
                self.candidate_answer_generated_count
            ),
            "prefetch_cache_hits": int(self.prefetch_cache_hits),
            "prefetch_generation_calls": int(self.prefetch_generation_calls),
            "prefetch_cached_answers": {
                name: len(cache)
                for name, cache in sorted(self.prefetch_cache.items())
            },
            "candidate_materializations": int(self.candidate_materialization_count),
            "private_materializations_evicted_after_export": int(
                self.private_materialization_eviction_count
            ),
            "materialization_integrity_audits": int(
                self.materialization_integrity_audit_count
            ),
            "selected_candidates_reused": int(self.selected_candidate_reuse_count),
            "guidance_scores_accepted": int(self.guidance_score_accepted_count),
            "guidance_scores_failed": int(self.guidance_score_failed_count),
            "candidate_answers_limit": self._aggregate_limit(
                "max_candidate_answers"
            ),
            "answer_grids_used": int(self.answer_count),
            "answer_grids_limit": self._aggregate_limit("max_answer_grids"),
            "query_calls": int(self.query_calls),
            "query_points_used": int(self.query_points),
            "query_points_limit": self._aggregate_limit("max_total_points"),
            "resolution_views_used": int(self.resolution_view_count),
            "resolution_views_limit": self._aggregate_limit(
                "max_views", resolution=True
            ),
            "resolution_target_elements_used": int(
                self.resolution_target_elements
            ),
            "resolution_target_elements_limit": self._aggregate_limit(
                "max_total_target_source_elements", resolution=True
            ),
            "materialization_generator_calls": int(
                self.materialization_generator_calls
            ),
            "raw_f_trajectory_bytes_discarded": int(
                self.raw_trajectory_bytes_discarded
            ),
            "max_batch_per_call": int(self.limits["max_batch"]),
            "max_points_per_call": int(self.limits["max_points_per_call"]),
            "active_wall_time_seconds_used": elapsed,
            "active_wall_time_seconds_limit": active_limit,
            "active_wall_time_expired": bool(
                active_limit is not None and elapsed >= active_limit
            ),
            "scratch_bytes_used": int(self.storage_bytes_used),
            "scratch_bytes_limit": (
                int(self.resource_budget["scratch_bytes"])
                if self.resource_mode
                else None
            ),
            "storage_scan_wall_time_seconds": float(
                self.storage_scan_wall_time_seconds
            ),
            "operation_usage": self.operation_usage,
            "declared_allocation": (
                dict(self.resource_budget["allocation"])
                if self.resource_mode
                else None
            ),
        }

    def _write_public_manifest(self) -> None:
        manifest = {
            "schema": "vpfp_online_generator_output_manifest_v1",
            "target_solver_sha256": self.config["target_solver_sha256"],
            "primary_object": "continuous hidden answer",
            "public_equation_is_derived_from_hidden_answer": True,
            "source_origin": SOURCE_ORIGIN,
            "coarse_shape_seed_is_truth": False,
            "solver_output_contract_version": MACRO_OUTPUT_CONTRACT,
            "regime": self.regime,
            "visible_files_are": "numerical access views only",
            "model_parameterization_exposed": False,
            "reward_tilt_selects_only_from_fresh_answer_pools": True,
            "resolution_materialization": self.resolution_policy,
            "cases": [
                {
                    "case_id": case_id,
                    "case_dir": f"cases/{case_id}",
                    "policy_variant": record["variant"],
                    "condition_pool": record["condition_pool"],
                    "numerical_answer_released": record["answers"] == "grid",
                    "root_case_id": record["root_case_id"],
                    "parent_case_id": record.get("parent_case_id"),
                    "resolution_view": bool(record["resolution_view"]),
                    "resolution": record["resolution"],
                    "study_id": record["study_id"],
                    "dimension": record["dimension"],
                    "level": int(record["level"]),
                    "characteristic_spacing": float(
                        record["characteristic_spacing"]
                    ),
                    "training_only": True,
                    "eligible_for_private_selection": False,
                    "eligible_for_final_evaluation": False,
                }
                for case_id, record in sorted(self.records.items())
            ],
            "budget_usage": self._budget_report(),
        }
        (self.output_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _write_compact_case_descriptors(self) -> dict[str, Any]:
        roots = [
            (case_id, record)
            for case_id, record in sorted(self.records.items())
            if not bool(record["resolution_view"])
        ]
        manifest_path = self.private_root / "case_descriptor_manifest.json"
        if not roots:
            manifest = {
                "schema": "vpfp_answer_first_compact_case_descriptors_v1",
                "case_count": 0,
                "descriptor_archive": None,
                "dense_grid_arrays_included": False,
                "decoder_checkpoint_sha256": self.decoder_checkpoint_sha256,
            }
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return manifest

        latent_rows = [
            record["latent"].detach().cpu().numpy()
            for _, record in roots
        ]
        condition_rows = [
            np.asarray(record["condition"])
            for _, record in roots
        ]
        latent_shape = tuple(latent_rows[0].shape)
        condition_shape = tuple(condition_rows[0].shape)
        if any(tuple(value.shape) != latent_shape for value in latent_rows):
            raise RuntimeError("online compact latent shapes changed within one run")
        if any(tuple(value.shape) != condition_shape for value in condition_rows):
            raise RuntimeError("online equation-condition shapes changed within one run")

        archive_path = self.private_root / "case_descriptors.npz"
        case_ids = [case_id for case_id, _ in roots]
        variants = [str(record["variant"]) for _, record in roots]
        pools = [str(record["condition_pool"]) for _, record in roots]
        np.savez_compressed(
            archive_path,
            case_ids=np.asarray(case_ids, dtype=f"<U{max(map(len, case_ids))}"),
            latents=np.stack(latent_rows),
            equation_conditions=np.stack(condition_rows),
            shape_seeds=np.asarray(
                [int(record["shape_seed"]) for _, record in roots],
                dtype=np.int64,
            ),
            policy_variants=np.asarray(
                variants, dtype=f"<U{max(map(len, variants))}"
            ),
            condition_pools=np.asarray(
                pools, dtype=f"<U{max(map(len, pools))}"
            ),
        )
        manifest = {
            "schema": "vpfp_answer_first_compact_case_descriptors_v1",
            "case_count": len(roots),
            "descriptor_archive": archive_path.name,
            "descriptor_archive_sha256": sha256_file(archive_path),
            "descriptor_archive_bytes": archive_path.stat().st_size,
            "latent_shape_per_case": list(latent_shape),
            "condition_shape_per_case": list(condition_shape),
            "dense_grid_arrays_included": False,
            "decoder_checkpoint_sha256": self.decoder_checkpoint_sha256,
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return manifest

    def _private_log(self, row: dict[str, Any]) -> None:
        with self.private_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _truth_times(value: Any, *, time_domain: tuple[float, float]) -> np.ndarray | None:
    if value is None:
        return None
    times = np.asarray(value, dtype=float)
    if times.ndim != 1 or times.size < 1 or not np.all(np.isfinite(times)):
        raise ValueError("truth_times must be a finite nonempty 1D array")
    if times.size > 4096:
        raise ValueError("truth_times exceed the per-view safety limit")
    t0, t1 = (float(time_domain[0]), float(time_domain[1]))
    tolerance = 1.0e-12 * max(1.0, abs(t1 - t0))
    if np.any(times < t0 - tolerance) or np.any(times > t1 + tolerance):
        raise ValueError("truth_times lie outside the hidden time domain")
    return times


def _resolution_dimension(base: VPFPResolution, target: VPFPResolution) -> str:
    changed = {
        "spatial": (base.nx, base.ny) != (target.nx, target.ny),
        "velocity": (base.nvx, base.nvy) != (target.nvx, target.nvy),
        "temporal": base.nt != target.nt,
    }
    active = [name for name, value in changed.items() if value]
    if not active:
        return "base"
    return active[0] if len(active) == 1 else "joint"


def _characteristic_spacing(
    base: VPFPResolution,
    target: VPFPResolution,
    *,
    dimension: str,
) -> float:
    ratios = []
    if dimension in {"spatial", "joint"}:
        ratios.extend((base.nx / target.nx, base.ny / target.ny))
    if dimension in {"velocity", "joint"}:
        ratios.extend(
            (
                (base.nvx - 1) / (target.nvx - 1),
                (base.nvy - 1) / (target.nvy - 1),
            )
        )
    if dimension in {"temporal", "joint"}:
        ratios.append((base.nt - 1) / (target.nt - 1))
    return float(max(ratios, default=1.0))


def _study_id(value: Any, *, default: str) -> str:
    result = str(default if value is None else value)
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
    if not 1 <= len(result) <= 64 or any(char not in allowed for char in result):
        raise ValueError("study_id must contain 1-64 safe characters")
    return result


def _materialization_level(value: Any, *, fallback: int) -> int:
    if value is None:
        return int(fallback)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("materialization level must be an integer")
    if not 0 <= int(value) <= 64:
        raise ValueError("materialization level must lie in [0,64]")
    return int(value)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def main() -> None:
    args = parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_TMPDIR"):
        raise SystemExit("answer-first online service requires a Slurm node")
    state = AnswerFirstOnlineGeneratorState(
        Path(args.config), device=str(args.device)
    )
    socket_path = Path(args.socket).resolve()
    if not any(
        _is_within(socket_path, root)
        for root in (state.workspace, Path(os.environ["SLURM_TMPDIR"]).resolve())
    ):
        raise SystemExit("online socket must stay in workspace or SLURM_TMPDIR")
    if socket_path.exists():
        raise SystemExit(f"refusing to replace socket: {socket_path}")
    server = UnixServer(str(socket_path), RequestHandler)
    server.state = state  # type: ignore[attr-defined]

    def stop(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if socket_path.exists():
            socket_path.unlink()
        print(json.dumps(state.finalize(), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
