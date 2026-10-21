"""Fresh paired answer-first development evaluation for VPFP solver Agents."""

from __future__ import annotations

import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from agent_verifier.storage_lifecycle import is_within_active_slurm_tmp

from .compute_budget import (
    COMPUTE_BUDGET_CONTRACT,
    SolverComputeBudget,
    coerce_budget_output,
)
from .continuous_hermite import VPFPContinuousHermiteConfig
from .continuous_macros import (
    MACRO_OUTPUT_KEYS,
    evaluate_continuous_hermite_macro_points,
)
from .continuous_v2 import STANDARD_VPFP_EQUATION_VERSION
from .continuous_cases import (
    materialize_continuous_case,
    verify_continuous_case_dir,
)
from .development_evaluation import (
    FINALIST_PURPOSE,
    VPFPRecoverableEvaluationError,
    VPFP_DEVELOPMENT_RESULT_SCHEMA,
    VPFP_FINALIST_COMPATIBILITY_SCHEMA,
    VPFPDevelopmentEvaluationRequest,
    deterministic_vpfp_candidate_evidence_seed,
    paired_mean_interval,
    tail_summary,
)
from .high_fidelity import VPFPHighFidelityConfig
from .online_generator_protocol import export_numerical_online_case
from .online_training import VPFPOnlineBatchGenerator, VPFPOnlineGenerationConfig
from .operators import VPFPGrid, VPFPTime
from .oracle_training import inspect_solver_bundle
from .public_vpfp_runtime import load_public_case
from .robust_perturbation import (
    VPFPRobustPerturbationConfig,
    build_deterministic_direct_model,
    calibrate_vpfp_robust_perturbation,
    expanded_vpfp_perturbation_config,
)
from .runtime_paths import is_persistent_run_path

VPFP_MACRO_MATCH_CONTRACT_ENV = "VPFP_MACRO_MATCH_CONTRACT"
VPFP_DEVELOPMENT_RUNTIME_SCHEMA = "vpfp_development_runtime_v2"
VPFP_PRIVATE_STAGE_SCHEMA = "vpfp_private_paired_development_stage_v2"
VPFP_BUDGETED_SOLVER_SUBPROCESS_SCHEMA = "vpfp_budgeted_solver_subprocess_v2"
FAILURE_SCORE = 1.0e6
TIME_BANDS = ("early", "middle", "late")


@dataclass(frozen=True)
class VPFPDevelopmentRuntimeConfig:
    regime_id: str
    collision_nu_range: tuple[float, float]
    temperature_range: tuple[float, float] = (0.8, 1.2)
    time_domain: tuple[float, float] = (0.0, 1.0)
    time_control_count: int = 33
    generation_batch_size: int = 8
    coarse_nx: int = 16
    coarse_ny: int = 16
    coarse_nv: int = 21
    coarse_velocity_max: float = 8.0
    internal_dt_max: float = 0.001
    materialized_nx: int = 33
    materialized_ny: int = 33
    materialized_nv: int = 9
    materialized_nt: int = 13
    materialized_velocity_max: float = 5.0
    materialization_chunk_size: int = 262_144
    evaluation_time_count: int = 17
    development_wall_seconds: tuple[float, ...] = (30.0, 180.0)
    formal_wall_seconds: tuple[float, ...] = (30.0, 180.0, 900.0)
    finalist_case_count: int = 1
    maximum_generation_attempt_factor: int = 16
    next_stage_maximum_mean_ratio: float = 1.5
    subprocess_grace_seconds: float = 60.0

    def validate(self) -> None:
        if not self.regime_id:
            raise ValueError("VPFP development runtime needs a regime id")
        if (
            len(self.collision_nu_range) != 2
            or self.collision_nu_range[0] < 0.0
            or self.collision_nu_range[1] <= self.collision_nu_range[0]
            or len(self.temperature_range) != 2
            or self.temperature_range[0] <= 0.0
            or self.temperature_range[1] <= self.temperature_range[0]
            or len(self.time_domain) != 2
            or self.time_domain[1] <= self.time_domain[0]
        ):
            raise ValueError("VPFP development physical regime is invalid")
        integers = (
            self.time_control_count,
            self.generation_batch_size,
            self.coarse_nx,
            self.coarse_ny,
            self.coarse_nv,
            self.materialized_nx,
            self.materialized_ny,
            self.materialized_nv,
            self.materialized_nt,
            self.materialization_chunk_size,
            self.evaluation_time_count,
            self.finalist_case_count,
            self.maximum_generation_attempt_factor,
        )
        if any(isinstance(value, bool) or int(value) < 1 for value in integers):
            raise ValueError("VPFP development integer settings must be positive")
        if min(self.time_control_count, self.materialized_nt) < 3:
            raise ValueError("VPFP development time grids need at least three points")
        if self.evaluation_time_count < 7:
            raise ValueError("VPFP development needs all three time bands")
        if (
            self.internal_dt_max <= 0.0
            or self.coarse_velocity_max <= 0.0
            or self.materialized_velocity_max <= 0.0
            or self.next_stage_maximum_mean_ratio < 1.0
            or self.subprocess_grace_seconds <= 0.0
        ):
            raise ValueError("VPFP development numerical settings are invalid")
        for schedule in (self.development_wall_seconds, self.formal_wall_seconds):
            if (
                len(schedule) < 2
                or any(not math.isfinite(value) or value <= 0.0 for value in schedule)
                or any(right <= left for left, right in zip(schedule, schedule[1:]))
            ):
                raise ValueError("VPFP development budget schedule is invalid")


class VPFPDevelopmentEvaluatorRuntime:
    """Controller-private fresh evaluator with cumulative 4/16/64 evidence."""

    def __init__(
        self,
        *,
        workspace: str | Path,
        private_root: str | Path,
        base_decoder_checkpoint: str | Path,
        expanded_decoder_checkpoint: str | Path,
        comparison_solver: str | Path,
        checkpoint_index: int,
        config: VPFPDevelopmentRuntimeConfig,
        perturbation_config: VPFPRobustPerturbationConfig | None = None,
        device: str = "cuda",
    ) -> None:
        config.validate()
        self.workspace = Path(workspace).resolve()
        self.private_root = Path(private_root).resolve()
        if (
            not is_persistent_run_path(self.workspace)
            or not (
                is_persistent_run_path(self.private_root)
                or is_within_active_slurm_tmp(self.private_root)
            )
            or self.workspace == self.private_root
            or self.workspace.is_relative_to(self.private_root)
            or self.private_root.is_relative_to(self.workspace)
        ):
            raise ValueError(
                "VPFP development workspace must be on /scratch and the private "
                "runtime must be on /scratch or active SLURM_TMPDIR"
            )
        if checkpoint_index < 0:
            raise ValueError("VPFP checkpoint index must be nonnegative")
        if not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("VPFP development evaluator requires a Slurm compute node")
        self.config = config
        self.checkpoint_index = int(checkpoint_index)
        self.device = str(device)
        self.perturbation_config = (
            perturbation_config or VPFPRobustPerturbationConfig()
        )
        self.private_root.mkdir(parents=True, exist_ok=True)
        self.base_decoder_checkpoint = Path(base_decoder_checkpoint).resolve()
        self.expanded_decoder_checkpoint = Path(expanded_decoder_checkpoint).resolve()
        for checkpoint in (
            self.base_decoder_checkpoint,
            self.expanded_decoder_checkpoint,
        ):
            if not checkpoint.is_file() or checkpoint.is_symlink():
                raise FileNotFoundError(checkpoint)

        import torch

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("VPFP development evaluator requested unavailable CUDA")
        base_payload = torch.load(
            self.base_decoder_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        expanded_payload = torch.load(
            self.expanded_decoder_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        self.base_config = VPFPContinuousHermiteConfig.from_dict(
            base_payload["decoder_config"]
        )
        self.expanded_config = VPFPContinuousHermiteConfig.from_dict(
            expanded_payload["decoder_config"]
        )
        for payload in (base_payload, expanded_payload):
            adapter = dict(payload.get("robust_answer_first_adapter") or {})
            if (
                adapter.get("unused_neural_parameters_zeroed") is not True
                or adapter.get("continuous_hidden_answer_is_primary") is not True
                or adapter.get("generator_parameterization_public") is not False
            ):
                raise ValueError("VPFP robust decoder checkpoint is not qualified")
        self._validate_checkpoint_configs()
        self.base_model = build_deterministic_direct_model(
            self.base_config,
            device=self.device,
        )
        generation = VPFPOnlineGenerationConfig(
            batch_size=int(config.generation_batch_size),
            nx=int(config.coarse_nx),
            ny=int(config.coarse_ny),
            velocity_grid_size=int(config.coarse_nv),
            nt=int(config.time_control_count),
            t1=float(config.time_domain[1]),
            velocity_max=float(config.coarse_velocity_max),
            internal_dt_max=float(config.internal_dt_max),
            cache_capacity_cases=max(64, int(config.generation_batch_size)),
            anchor_case_count=max(1, int(config.generation_batch_size)),
        )
        equation = VPFPHighFidelityConfig(
            internal_dt_max=float(config.internal_dt_max),
            collision_nu_range=tuple(config.collision_nu_range),
            temperature_range=tuple(config.temperature_range),
        )
        self.generator = VPFPOnlineBatchGenerator(
            decoder_config=self.base_config,
            generation_config=generation,
            equation_config=equation,
            device=self.device,
        )
        self.materialized_grid = VPFPGrid(
            nx=int(config.materialized_nx),
            ny=int(config.materialized_ny),
            nvx=int(config.materialized_nv),
            nvy=int(config.materialized_nv),
            x_domain=tuple(float(value) for value in self.base_config.equation.x_domain),
            v_bounds=(
                -float(config.materialized_velocity_max),
                float(config.materialized_velocity_max),
                -float(config.materialized_velocity_max),
                float(config.materialized_velocity_max),
            ),
        )
        self.materialized_time = VPFPTime(
            t0=float(config.time_domain[0]),
            t1=float(config.time_domain[1]),
            nt=int(config.materialized_nt),
        )
        self.materialized_grid.validate()
        self.materialized_time.validate()
        self.comparison_solver = Path(comparison_solver).resolve()
        self.comparison_sha256 = str(
            inspect_solver_bundle(self.comparison_solver)["aggregate_sha256"]
        )
        manifest = _read_json(self.workspace / "workspace_manifest.json")
        interface = dict(manifest["development_evaluation_interface"])
        self.evidence_case_counts = tuple(
            int(value) for value in interface["evidence_case_counts"]
        )
        self.worker_script = (
            Path(__file__).resolve().parents[3]
            / "scripts"
            / "run_vpfp_budgeted_solver_call_v2.py"
        )
        if not self.worker_script.is_file():
            raise FileNotFoundError(self.worker_script)
        self.runtime_manifest = {
            "schema": VPFP_DEVELOPMENT_RUNTIME_SCHEMA,
            "checkpoint_index": self.checkpoint_index,
            "regime_id": config.regime_id,
            "answer_first_exact_continuous_truth_primary": True,
            "base_and_structure_preserving_perturbed_required": True,
            "source_derivative_backend": "structured_fourier_hermite_jet_v1",
            "solver_process_is_fresh_per_case_budget": True,
            "solver_receives_public_only_case_directory": True,
            "comparison_sha256": self.comparison_sha256,
            "evidence_case_counts": list(self.evidence_case_counts),
            "development_budgets": [
                value.to_dict() for value in self._budgets(formal=False)
            ],
            "formal_budgets": [
                value.to_dict() for value in self._budgets(formal=True)
            ],
        }
        _write_json(self.private_root / "runtime_manifest.json", self.runtime_manifest)

    def _validate_checkpoint_configs(self) -> None:
        expected_expanded = expanded_vpfp_perturbation_config(
            self.base_config,
            self.perturbation_config,
        )
        if expected_expanded.to_dict() != self.expanded_config.to_dict():
            raise ValueError("VPFP expanded checkpoint does not match perturbation policy")
        if (
            tuple(float(value) for value in self.base_config.time_domain)
            != tuple(float(value) for value in self.config.time_domain)
            or int(self.base_config.structured_time_control_count)
            != int(self.config.time_control_count)
            or tuple(float(value) for value in self.base_config.collision_nu_range)
            != tuple(float(value) for value in self.config.collision_nu_range)
            or tuple(float(value) for value in self.base_config.temperature_range)
            != tuple(float(value) for value in self.config.temperature_range)
        ):
            raise ValueError("VPFP base checkpoint is bound to a different regime")

    def _budgets(self, *, formal: bool) -> tuple[SolverComputeBudget, ...]:
        wall = (
            self.config.formal_wall_seconds
            if formal
            else self.config.development_wall_seconds
        )
        cpus = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
        memory_bytes = 64 * 1024**3
        return tuple(
            SolverComputeBudget(
                tier=index,
                wall_time_seconds=float(seconds),
                cpus=cpus,
                gpus=1,
                memory_bytes=memory_bytes,
            )
            for index, seconds in enumerate(wall)
        )

    def handler(
        self,
        request: VPFPDevelopmentEvaluationRequest,
        candidate_path: Path,
        candidate_sha256: str,
        available_case_credits: int,
        previous_case_count: int,
    ) -> Mapping[str, Any]:
        if request.purpose == FINALIST_PURPOSE:
            seed = deterministic_vpfp_candidate_evidence_seed(
                checkpoint_index=self.checkpoint_index,
                agent_id=request.agent_id,
                candidate_sha256=candidate_sha256,
                evidence_case_count=1_000_001,
            )
            stage = self._run_fresh_stage(
                candidate_path=candidate_path,
                candidate_sha256=candidate_sha256,
                stage_label=f"finalist_{request.request_id}",
                stage_total_case_count=self.config.finalist_case_count,
                sample_count=self.config.finalist_case_count,
                seed=seed,
                budgets=self._budgets(formal=True),
            )
            result = summarize_vpfp_development_stage_reports(
                [stage],
                charged_case_credits=0,
            )
            result["cascade"] = {
                "next_stage_admissible": False,
                "official_promotion_result": False,
                "purpose": FINALIST_PURPOSE,
            }
            passed = bool(result["safety_passed"])
            result["finalist_compatibility"] = {
                "schema": VPFP_FINALIST_COMPATIBILITY_SCHEMA,
                "passed": passed,
                "candidate_sha256": candidate_sha256,
                "exact_final_bundle": True,
                "case_count": self.config.finalist_case_count,
                "budgets": [
                    value.to_dict() for value in self._budgets(formal=True)
                ],
                "all_solver_calls_safe": passed,
            }
            return result

        next_total = next(
            (
                value
                for value in self.evidence_case_counts
                if value > int(previous_case_count)
            ),
            None,
        )
        if (
            next_total is None
            or next_total > int(request.requested_evidence_case_count)
            or next_total - int(previous_case_count) > int(available_case_credits)
        ):
            raise ValueError("VPFP cumulative development stage is not admissible")
        stage_seed = deterministic_vpfp_candidate_evidence_seed(
            checkpoint_index=self.checkpoint_index,
            agent_id=request.agent_id,
            candidate_sha256=candidate_sha256,
            evidence_case_count=next_total,
        )
        self._run_fresh_stage(
            candidate_path=candidate_path,
            candidate_sha256=candidate_sha256,
            stage_label=f"evidence_{next_total:06d}",
            stage_total_case_count=next_total,
            sample_count=next_total - int(previous_case_count),
            seed=stage_seed,
            budgets=self._budgets(formal=False),
        )
        reports = [
            self._load_stage(candidate_sha256, total)
            for total in self.evidence_case_counts
            if total <= next_total
        ]
        if sum(int(row["sample_count"]) for row in reports) != next_total:
            raise RuntimeError("VPFP cumulative private evidence is incomplete")
        result = summarize_vpfp_development_stage_reports(
            reports,
            charged_case_credits=next_total - int(previous_case_count),
        )
        comparison_mean = float(result["comparison"]["high_budget_mean"])
        candidate_mean = float(result["candidate"]["high_budget_mean"])
        plausible = bool(
            result["safety_passed"]
            and candidate_mean
            <= max(
                comparison_mean * self.config.next_stage_maximum_mean_ratio,
                comparison_mean + 1.0e-2,
            )
        )
        following = next(
            (value for value in self.evidence_case_counts if value > next_total),
            None,
        )
        result["cascade"] = {
            "requested_evidence_case_count": int(
                request.requested_evidence_case_count
            ),
            "previous_evidence_case_count": int(previous_case_count),
            "completed_stage_total": int(next_total),
            "charged_incremental_case_credits": int(
                next_total - int(previous_case_count)
            ),
            "next_stage_case_count": following,
            "next_stage_admissible": bool(following is not None and plausible),
            "controller_plausibility_gate_passed": plausible,
            "same_hash_evidence_is_cumulative": True,
            "same_hash_fresh_reroll_forbidden": True,
            "official_promotion_result": False,
        }
        return result

    def _stage_root(self, candidate_sha256: str, stage_label: str) -> Path:
        return self.private_root / "candidates" / candidate_sha256 / stage_label

    def _load_stage(
        self, candidate_sha256: str, stage_total_case_count: int
    ) -> dict[str, Any]:
        return _read_json(
            self._stage_root(
                candidate_sha256,
                f"evidence_{stage_total_case_count:06d}",
            )
            / "report.json"
        )

    def _run_fresh_stage(
        self,
        *,
        candidate_path: Path,
        candidate_sha256: str,
        stage_label: str,
        stage_total_case_count: int,
        sample_count: int,
        seed: int,
        budgets: Sequence[SolverComputeBudget],
    ) -> dict[str, Any]:
        stage_root = self._stage_root(candidate_sha256, stage_label)
        report_path = stage_root / "report.json"
        if report_path.is_file():
            cached = _read_json(report_path)
            self._validate_cached_stage(
                cached,
                candidate_sha256=candidate_sha256,
                stage_label=stage_label,
                stage_total_case_count=stage_total_case_count,
                sample_count=sample_count,
                seed=seed,
                budgets=budgets,
            )
            return cached
        stage_root.mkdir(parents=True, exist_ok=True)
        attempt_root = stage_root / f"attempt_{time.time_ns()}"
        attempt_root.mkdir()
        started = time.perf_counter()
        rows: list[dict[str, Any]] = []
        rejection_count = 0
        attempted = 0
        generation_batch_size = int(self.config.generation_batch_size)
        maximum_attempts = max(
            int(sample_count * self.config.maximum_generation_attempt_factor),
            generation_batch_size,
        )
        generated_answer_count = 0
        generation_wall = 0.0
        while len(rows) < sample_count and attempted < maximum_attempts:
            take = min(
                generation_batch_size,
                maximum_attempts - attempted,
            )
            seeds = np.asarray(
                [int(seed) * 10_000 + attempted + index for index in range(take)],
                dtype=np.int64,
            )
            generation_started = time.perf_counter()
            batch = self.generator.generate(seeds)
            generated_answer_count += take
            generation_wall += time.perf_counter() - generation_started
            for index, equation_seed in enumerate(seeds):
                attempted += 1
                condition = (
                    batch["equation_conditions"][index]
                    .detach()
                    .cpu()
                    .numpy()
                )
                calibration = calibrate_vpfp_robust_perturbation(
                    self.base_model,
                    batch["latents"][index],
                    equation_condition=condition,
                    seed=int(equation_seed) + 50_000_000,
                    perturbation=self.perturbation_config,
                )
                if calibration.perturbed_latent is None:
                    rejection_count += 1
                    continue
                if (
                    calibration.model.config.to_dict()
                    != self.expanded_config.to_dict()
                ):
                    raise RuntimeError("VPFP perturbation model config changed")
                root_index = len(rows)
                query_times = deterministic_vpfp_query_times(
                    self.config.time_domain,
                    self.config.evaluation_time_count,
                    seed=int(equation_seed) + 104_729,
                )
                variants = {}
                for variant, latent in (
                    ("base", calibration.embedded_base_latent),
                    ("perturbed", calibration.perturbed_latent),
                ):
                    private_case = (
                        attempt_root
                        / "private_cases"
                        / f"case_{root_index:06d}_{variant}"
                    )
                    public_case = (
                        attempt_root
                        / "public_cases"
                        / f"case_{root_index:06d}_{variant}"
                    )
                    materialize_continuous_case(
                        private_case,
                        case_id=f"private_{root_index:06d}_{variant}",
                        model=calibration.model,
                        decoder_checkpoint=self.expanded_decoder_checkpoint,
                        latent=latent,
                        grid=self.materialized_grid,
                        time=self.materialized_time,
                        equation_condition=condition,
                        source_origin="generated_continuous_autodiff_residual",
                        source_derivative_backend="analytic",
                        solver_output_contract=(
                            "vpfp_continuous_macro_output_v1"
                        ),
                        generator_metadata={
                            "generator_family": "coarse_projection_answer_first",
                            "robust_variant": variant,
                            "base_generator_parameters_modified": False,
                        },
                        case_spec={
                            "cohort": "private_development",
                            "truth_primary": "continuous_hidden_answer",
                            "low_fidelity_equation_solve_is_truth": False,
                            "answer_first_perturbation_applied": (
                                variant == "perturbed"
                            ),
                        },
                        evaluation_chunk_size=int(
                            self.config.materialization_chunk_size
                        ),
                    )
                    audit = verify_continuous_case_dir(
                        private_case,
                        device=self.device,
                        truth_sample_count=4,
                        source_sample_count=4,
                    )
                    if not audit["accepted"]:
                        raise RuntimeError(
                            f"VPFP private materialization failed: {audit['failures']}"
                        )
                    _attach_compute_budget_contract(private_case)
                    export_numerical_online_case(
                        private_case,
                        public_case,
                        public_case_id=f"development_{root_index:06d}_{variant}",
                        include_answer=False,
                    )
                    truth = self._exact_macro_grid(
                        calibration.model,
                        latent,
                        condition,
                        query_times,
                    )
                    solver_rows = {}
                    for solver_id, solver_path, solver_sha256 in (
                        (
                            "candidate",
                            candidate_path,
                            candidate_sha256,
                        ),
                        (
                            "comparison",
                            self.comparison_solver,
                            self.comparison_sha256,
                        ),
                    ):
                        solver_rows[solver_id] = {
                            "solver_sha256": solver_sha256,
                            "budgets": [
                                self._run_solver_call(
                                    solver_path=solver_path,
                                    public_case_dir=public_case,
                                    query_times=query_times,
                                    truth=truth,
                                    budget=budget,
                                    output_root=(
                                        attempt_root
                                        / "solver_calls"
                                        / f"case_{root_index:06d}_{variant}"
                                        / solver_id
                                        / f"tier_{budget.tier}"
                                    ),
                                )
                                for budget in budgets
                            ],
                        }
                    variants[variant] = {"solvers": solver_rows}
                rows.append(
                    {
                        "variants": variants,
                        "perturbation_metrics": dict(
                            calibration.report["selected_attempt"]["metrics"]
                        ),
                    }
                )
                if len(rows) >= sample_count:
                    break
        if len(rows) != sample_count:
            raise VPFPRecoverableEvaluationError(
                "VPFP robust perturbation acceptance exhausted generation attempts"
            )
        report = {
            "schema": VPFP_PRIVATE_STAGE_SCHEMA,
            "stage_label": stage_label,
            "stage_total_case_count": int(stage_total_case_count),
            "sample_count": int(sample_count),
            "seed": int(seed),
            "candidate_sha256": candidate_sha256,
            "comparison_sha256": self.comparison_sha256,
            "budgets": [value.to_dict() for value in budgets],
            "root_case_rows": rows,
            "attempted_generation_count": int(attempted),
            "generated_answer_count": int(generated_answer_count),
            "perturbation_rejection_count": int(rejection_count),
            "generation_wall_time_seconds": float(generation_wall),
            "total_wall_time_seconds": float(time.perf_counter() - started),
            "private_case_ids_included": True,
            "public_result_path": None,
        }
        _write_json(report_path, report)
        return report

    def _validate_cached_stage(
        self,
        report: Mapping[str, Any],
        *,
        candidate_sha256: str,
        stage_label: str,
        stage_total_case_count: int,
        sample_count: int,
        seed: int,
        budgets: Sequence[SolverComputeBudget],
    ) -> None:
        if (
            report.get("schema") != VPFP_PRIVATE_STAGE_SCHEMA
            or report.get("stage_label") != stage_label
            or report.get("stage_total_case_count") != stage_total_case_count
            or report.get("sample_count") != sample_count
            or report.get("seed") != seed
            or report.get("candidate_sha256") != candidate_sha256
            or report.get("comparison_sha256") != self.comparison_sha256
            or report.get("budgets")
            != [value.to_dict() for value in budgets]
            or len(report.get("root_case_rows") or []) != sample_count
        ):
            raise ValueError("VPFP cached private development stage changed")

    def _exact_macro_grid(
        self,
        model: Any,
        latent: Any,
        condition: np.ndarray,
        query_times: np.ndarray,
    ) -> dict[str, np.ndarray]:
        x = np.linspace(
            0.0,
            float(self.materialized_grid.x_domain[0]),
            int(self.materialized_grid.nx),
            endpoint=False,
        )
        y = np.linspace(
            0.0,
            float(self.materialized_grid.x_domain[1]),
            int(self.materialized_grid.ny),
            endpoint=False,
        )
        mesh = np.meshgrid(query_times, x, y, indexing="ij")
        points = np.column_stack([value.reshape(-1) for value in mesh])
        values = evaluate_continuous_hermite_macro_points(
            model,
            latent,
            points,
            equation_condition=condition,
            chunk_size=8192,
        )
        scalar_shape = (len(query_times), len(x), len(y))
        return {
            "rho": np.asarray(values["rho"], dtype=np.float64).reshape(
                scalar_shape
            ),
            "momentum": np.asarray(
                values["momentum"], dtype=np.float64
            ).reshape(*scalar_shape, 2),
            "kinetic_energy": np.asarray(
                values["kinetic_energy"], dtype=np.float64
            ).reshape(scalar_shape),
        }

    def _run_solver_call(
        self,
        *,
        solver_path: Path,
        public_case_dir: Path,
        query_times: np.ndarray,
        truth: Mapping[str, np.ndarray],
        budget: SolverComputeBudget,
        output_root: Path,
    ) -> dict[str, Any]:
        output_root.mkdir(parents=True, exist_ok=True)
        query_path = output_root / "query_times.npy"
        budget_path = output_root / "budget.json"
        output_path = output_root / "prediction.npz"
        report_path = output_root / "worker_report.json"
        np.save(query_path, query_times)
        _write_json(budget_path, budget.to_dict())
        command = [
            sys.executable,
            str(self.worker_script),
            "--solver",
            str(solver_path),
            "--case",
            str(public_case_dir),
            "--query-times",
            str(query_path),
            "--budget",
            str(budget_path),
            "--output",
            str(output_path),
            "--report",
            str(report_path),
        ]
        stdout_path = output_root / "stdout.log"
        stderr_path = output_root / "stderr.log"
        timed_out = False
        returncode = None
        candidate_environment = os.environ.copy()
        candidate_environment.pop(VPFP_MACRO_MATCH_CONTRACT_ENV, None)
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                command,
                stdout=stdout,
                stderr=stderr,
                env=candidate_environment,
                start_new_session=True,
                cwd=str(Path(__file__).resolve().parents[3]),
            )
            try:
                returncode = process.wait(
                    timeout=(
                        float(budget.wall_time_seconds)
                        + float(self.config.subprocess_grace_seconds)
                    )
                )
            except subprocess.TimeoutExpired:
                timed_out = True
                _terminate_process_group(process)
        if (
            timed_out
            or returncode != 0
            or not report_path.is_file()
            or report_path.is_symlink()
        ):
            return _failed_solver_call(
                budget,
                failure=(
                    "controller_subprocess_timeout"
                    if timed_out
                    else f"worker_exit_{returncode}"
                ),
            )
        worker = _read_json(report_path)
        if (
            worker.get("schema") != VPFP_BUDGETED_SOLVER_SUBPROCESS_SCHEMA
            or worker.get("accepted") is not True
            or not output_path.is_file()
        ):
            return _failed_solver_call(
                budget,
                failure=str(
                    worker.get("exception_type")
                    or worker.get("status")
                    or "worker_failure"
                ),
                worker=worker,
            )
        public_case = load_public_case(public_case_dir)
        try:
            with np.load(output_path, allow_pickle=False) as payload:
                prediction = coerce_budget_output(
                    {name: np.asarray(payload[name]) for name in payload.files},
                    public_case,
                    query_times,
                )
            score = score_vpfp_macro_prediction(
                prediction,
                truth,
                query_times=query_times,
            )
        except Exception as exc:  # noqa: BLE001 - invalid prediction is evidence.
            return _failed_solver_call(
                budget,
                failure=f"parent_validation:{type(exc).__name__}",
                worker=worker,
            )
        requested_contract = os.environ.get(
            VPFP_MACRO_MATCH_CONTRACT_ENV,
            "",
        ).strip()
        if requested_contract:
            from pde_agent_verifier_solver.plugins.vpfp.legacy_macro_bridge import (
                evaluate_legacy_macro_arrays,
            )

            arrays = public_case["arrays"]
            unified_macro_match = evaluate_legacy_macro_arrays(
                case_id=str(public_case["case_id"]),
                query_times=query_times,
                x=arrays["x"],
                y=arrays["y"],
                prediction_rho=prediction["rho"],
                prediction_momentum=prediction["momentum"],
                prediction_kinetic_energy=prediction["kinetic_energy"],
                truth_rho=truth["rho"],
                truth_momentum=truth["momentum"],
                truth_kinetic_energy=truth["kinetic_energy"],
            )
            if unified_macro_match["contract"] != requested_contract:
                raise RuntimeError("VPFP unified macro-match contract changed")
            score["unified_macro_match"] = unified_macro_match
        wall = float(worker["wall_time_seconds"])
        result = {
            "budget": budget.to_dict(),
            "accepted": True,
            "wall_limit_passed": bool(
                math.isfinite(wall)
                and wall <= budget.wall_time_seconds + 1.0
            ),
            "wall_time_seconds": wall,
            "score_smaller_is_better": score["score_smaller_is_better"],
            "field_relative_l2": score["field_relative_l2"],
            "time_band_scores": score["time_band_scores"],
            "failure": None,
        }
        if "unified_macro_match" in score:
            result["unified_macro_match"] = score["unified_macro_match"]
        return result


def deterministic_vpfp_query_times(
    time_domain: tuple[float, float],
    count: int,
    *,
    seed: int,
) -> np.ndarray:
    t0, t1 = (float(value) for value in time_domain)
    if count < 3 or not t1 > t0:
        raise ValueError("VPFP query-time request is invalid")
    values = np.linspace(t0, t1, int(count), dtype=np.float64)
    if count > 2:
        rng = np.random.default_rng(int(seed))
        spacing = (t1 - t0) / float(count - 1)
        values[1:-1] += rng.uniform(-0.3, 0.3, size=count - 2) * spacing
        values[1:-1] = np.sort(values[1:-1])
    if not np.all(np.diff(values) > 0.0):
        raise RuntimeError("VPFP deterministic query times are not ordered")
    return values


def score_vpfp_macro_prediction(
    prediction: Mapping[str, Any],
    truth: Mapping[str, Any],
    *,
    query_times: Sequence[float],
) -> dict[str, Any]:
    times = np.asarray(query_times, dtype=np.float64)
    if times.ndim != 1 or times.size < 3 or not np.all(np.diff(times) > 0.0):
        raise ValueError("VPFP macro score needs ordered query times")
    fields = {}
    for name in MACRO_OUTPUT_KEYS:
        predicted = np.asarray(prediction[name], dtype=np.float64)
        reference = np.asarray(truth[name], dtype=np.float64)
        if (
            predicted.shape != reference.shape
            or predicted.shape[0] != times.size
            or not np.isfinite(predicted).all()
            or not np.isfinite(reference).all()
        ):
            raise ValueError("VPFP macro score field is invalid")
        fields[name] = (predicted, reference)

    def score(indices: np.ndarray) -> tuple[float, dict[str, float]]:
        errors = {
            name: _relative_l2(
                predicted[indices] - reference[indices],
                reference[indices],
            )
            for name, (predicted, reference) in fields.items()
        }
        rho = fields["rho"][0][indices]
        kinetic = fields["kinetic_energy"][0][indices]
        positivity = _relative_l2(np.minimum(rho, 0.0), rho) + _relative_l2(
            np.minimum(kinetic, 0.0),
            kinetic,
        )
        return float(np.mean(list(errors.values())) + positivity), errors

    all_indices = np.arange(times.size)
    aggregate, field_errors = score(all_indices)
    tau = (times - times[0]) / max(times[-1] - times[0], np.finfo(float).tiny)
    masks = {
        "early": tau <= 1.0 / 3.0,
        "middle": (tau > 1.0 / 3.0) & (tau <= 2.0 / 3.0),
        "late": tau > 2.0 / 3.0,
    }
    band_scores = {}
    for name, mask in masks.items():
        indices = np.flatnonzero(mask)
        if indices.size == 0:
            raise RuntimeError(f"VPFP query times missed the {name} band")
        band_scores[name] = score(indices)[0]
    return {
        "score_smaller_is_better": aggregate,
        "field_relative_l2": field_errors,
        "time_band_scores": band_scores,
    }


def summarize_vpfp_development_stage_reports(
    reports: Sequence[Mapping[str, Any]],
    *,
    charged_case_credits: int,
) -> dict[str, Any]:
    rows = [dict(value) for value in reports]
    if not rows:
        raise ValueError("VPFP development summary needs private stages")
    budget_payload = rows[0].get("budgets")
    if not isinstance(budget_payload, list) or len(budget_payload) < 2:
        raise ValueError("VPFP development stages need a budget frontier")
    for row in rows:
        if (
            row.get("schema") != VPFP_PRIVATE_STAGE_SCHEMA
            or row.get("budgets") != budget_payload
            or row.get("comparison_sha256") != rows[0].get("comparison_sha256")
            or not isinstance(row.get("root_case_rows"), list)
        ):
            raise ValueError("VPFP private development stage changed")
    case_count = sum(int(row["sample_count"]) for row in rows)
    if case_count <= 0 or sum(
        len(row["root_case_rows"]) for row in rows
    ) != case_count:
        raise ValueError("VPFP private development case count is invalid")

    summaries = {
        solver_id: _summarize_solver(rows, solver_id)
        for solver_id in ("candidate", "comparison")
    }
    candidate_high = summaries["candidate"].pop("_high_budget_values")
    comparison_high = summaries["comparison"].pop("_high_budget_values")
    robust = summaries["candidate"].pop("_robust")
    paired = paired_mean_interval(candidate_high, comparison_high)
    safety = bool(
        summaries["candidate"]["safety_passed"]
        and summaries["comparison"]["safety_passed"]
    )
    return {
        "schema": VPFP_DEVELOPMENT_RESULT_SCHEMA,
        "case_count": int(case_count),
        "charged_case_credits": int(charged_case_credits),
        "private_case_ids_included": False,
        "per_case_errors_included": False,
        "hidden_answers_included": False,
        "generator_parameters_included": False,
        "official_promotion_result": False,
        "safety_passed": safety,
        "robust": robust,
        "candidate": summaries["candidate"],
        "comparison": summaries["comparison"],
        "paired": {
            "high_budget_mean_difference": float(
                paired["mean_candidate_minus_comparison"]
            ),
            "normal_approximation_95_lower": float(
                paired["normal_approximation_95_lower"]
            ),
            "normal_approximation_95_upper": float(
                paired["normal_approximation_95_upper"]
            ),
            "candidate_better_fraction": float(
                paired["candidate_better_fraction"]
            ),
        },
        "cascade": {"next_stage_admissible": False},
        "answer_first_exact_continuous_truth_primary": True,
        "equation_first_reference_error_used": False,
    }


def _summarize_solver(
    reports: Sequence[Mapping[str, Any]],
    solver_id: str,
) -> dict[str, Any]:
    calls = []
    high_by_variant = {"base": [], "perturbed": []}
    bands = {
        variant: {band: [] for band in TIME_BANDS}
        for variant in ("base", "perturbed")
    }
    high_values = []
    for report in reports:
        for root_row in report["root_case_rows"]:
            variants = dict(root_row["variants"])
            if set(variants) != {"base", "perturbed"}:
                raise ValueError("VPFP robust stage is missing a variant")
            for variant, variant_row in variants.items():
                budget_rows = list(variant_row["solvers"][solver_id]["budgets"])
                if len(budget_rows) != len(report["budgets"]):
                    raise ValueError("VPFP solver budget evidence is incomplete")
                calls.extend(budget_rows)
                high = budget_rows[-1]
                value = float(high["score_smaller_is_better"])
                high_values.append(value)
                high_by_variant[variant].append(value)
                for band in TIME_BANDS:
                    bands[variant][band].append(
                        float(high["time_band_scores"][band])
                    )
    if not high_values:
        raise ValueError("VPFP solver summary is empty")
    budget_means = []
    for budget_index in range(len(reports[0]["budgets"])):
        values = []
        for report in reports:
            for root_row in report["root_case_rows"]:
                for variant in ("base", "perturbed"):
                    values.append(
                        float(
                            root_row["variants"][variant]["solvers"][solver_id][
                                "budgets"
                            ][budget_index]["score_smaller_is_better"]
                        )
                    )
        budget_means.append(float(np.mean(values)))
    high_tail = tail_summary(high_values)
    variant_tail = {
        variant: tail_summary(values)
        for variant, values in high_by_variant.items()
    }
    band_tail = {
        variant: {
            band: tail_summary(values)
            for band, values in by_band.items()
        }
        for variant, by_band in bands.items()
    }
    all_finite = all(
        math.isfinite(float(call["score_smaller_is_better"]))
        and float(call["score_smaller_is_better"]) >= 0.0
        for call in calls
    )
    safety = bool(
        calls
        and all(call.get("accepted") is True for call in calls)
        and all(call.get("wall_limit_passed") is True for call in calls)
        and all_finite
    )
    robust = {
        "worst_answer_first_p90": float(
            max(value["p90"] for value in variant_tail.values())
        ),
        "worst_time_band_p90": float(
            max(
                value["p90"]
                for variant in band_tail.values()
                for value in variant.values()
            )
        ),
        "base_high_budget_p90": float(variant_tail["base"]["p90"]),
        "perturbed_high_budget_p90": float(
            variant_tail["perturbed"]["p90"]
        ),
    }
    return {
        "safety_passed": safety,
        "valid_prediction_fraction": float(
            np.mean([call.get("accepted") is True for call in calls])
        ),
        "wall_limit_pass_fraction": float(
            np.mean([call.get("wall_limit_passed") is True for call in calls])
        ),
        "high_budget_mean": float(high_tail["mean"]),
        "high_budget_cvar90": float(high_tail["cvar90"]),
        "frontier_geometric_mean": float(
            math.exp(
                np.mean(np.log(np.maximum(budget_means, 1.0e-12)))
            )
        ),
        "budget_tier_mean_errors": budget_means,
        "mean_solver_wall_time_seconds": float(
            np.mean(
                [
                    float(call["wall_time_seconds"])
                    for call in calls
                    if call.get("wall_time_seconds") is not None
                    and math.isfinite(float(call["wall_time_seconds"]))
                ]
                or [0.0]
            )
        ),
        "_high_budget_values": high_values,
        "_robust": robust,
    }


def _failed_solver_call(
    budget: SolverComputeBudget,
    *,
    failure: str,
    worker: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    wall = None
    if worker is not None and worker.get("wall_time_seconds") is not None:
        try:
            wall = float(worker["wall_time_seconds"])
        except (TypeError, ValueError):
            wall = None
    return {
        "budget": budget.to_dict(),
        "accepted": False,
        "wall_limit_passed": False,
        "wall_time_seconds": wall,
        "score_smaller_is_better": FAILURE_SCORE,
        "field_relative_l2": {
            name: FAILURE_SCORE for name in MACRO_OUTPUT_KEYS
        },
        "time_band_scores": {name: FAILURE_SCORE for name in TIME_BANDS},
        "failure": str(failure)[:500],
    }


def _attach_compute_budget_contract(case_dir: Path) -> None:
    question_path = case_dir / "public_question.json"
    question = _read_json(question_path)
    question.update(
        {
            "equation_version": STANDARD_VPFP_EQUATION_VERSION,
            "solver_compute_budget_contract_version": COMPUTE_BUDGET_CONTRACT,
            "time_query_contract": {
                "query_times": (
                    "arbitrary finite 1D times in the public [t0, t1] interval"
                ),
                "output_time_axis": "exactly follows query_times",
                "internal_time_discretization": "candidate controlled",
            },
            "solver_contract": (
                "define solve_with_budget(public_case, query_times, budget) "
                "and return rho, momentum, and kinetic_energy"
            ),
        }
    )
    _write_json(question_path, question)


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10.0)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait(timeout=30.0)


def _relative_l2(value: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.asarray(value, dtype=np.float64))
        / (
            np.linalg.norm(np.asarray(reference, dtype=np.float64))
            + 1.0e-12
        )
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
