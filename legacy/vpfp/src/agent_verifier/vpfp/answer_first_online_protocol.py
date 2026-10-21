from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from agent_verifier.storage_lifecycle import is_within_active_slurm_tmp
from agent_verifier.vpfp.runtime_paths import is_persistent_run_path

from agent_verifier.vpfp.continuous_macros import MACRO_OUTPUT_CONTRACT
from agent_verifier.vpfp.guided_scoring import (
    normalize_guidance_solver_budget,
)
from agent_verifier.vpfp.high_fidelity import VPFPHighFidelityConfig
from agent_verifier.vpfp.online_training import VPFPOnlineGenerationConfig
from agent_verifier.vpfp.resolution_convergence import VPFPResolution


ANSWER_FIRST_BUILD_SCHEMA = (
    "vpfp_projected_continuous_answer_first_agent_round_case_build_v1"
)
ANSWER_FIRST_ONLINE_GENERATOR_CONFIG_SCHEMA = (
    "vpfp_projected_continuous_answer_first_online_controller_config_v1"
)
ANSWER_FIRST_RESOLUTION_MATERIALIZATION_SCHEMA = (
    "vpfp_answer_first_continuous_materialization_v1"
)
ANSWER_FIRST_DEVELOPMENT_RESOURCE_SCHEMA = (
    "vpfp_answer_first_agent_development_resource_v1"
)
AGGREGATE_LIMITS_ACCOUNTING_ONLY = "accounting_only"
RESOLUTION_KEYS = ("nx", "ny", "nvx", "nvy", "nt")


def validate_answer_first_controller_config(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    payload = dict(config)
    if payload.get("schema") != ANSWER_FIRST_ONLINE_GENERATOR_CONFIG_SCHEMA:
        raise ValueError("unexpected answer-first online controller schema")

    workspace = Path(str(payload.get("workspace") or "")).resolve()
    private_root = Path(str(payload.get("private_root") or "")).resolve()
    target_solver = Path(str(payload.get("target_solver") or "")).resolve()
    decoder_checkpoint = Path(
        str(payload.get("decoder_checkpoint") or "")
    ).resolve()
    if not workspace.is_dir() or workspace.is_symlink():
        raise FileNotFoundError(workspace)
    for path in (target_solver, decoder_checkpoint):
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
    private_is_runtime_ephemeral = is_within_active_slurm_tmp(private_root)
    if not is_persistent_run_path(workspace) or not (
        is_persistent_run_path(private_root) or private_is_runtime_ephemeral
    ):
        raise ValueError(
            "workspace must stay in configured run storage and private root "
            "must stay there or in the active SLURM_TMPDIR"
        )
    if _is_within(private_root, workspace) or _is_within(workspace, private_root):
        raise ValueError("controller-private root and Agent workspace must be disjoint")

    target_sha = str(payload.get("target_solver_sha256") or "")
    if len(target_sha) != 64 or any(
        character not in "0123456789abcdef" for character in target_sha
    ):
        raise ValueError("target solver SHA-256 is malformed")
    if payload.get("solver_output_contract_version") != MACRO_OUTPUT_CONTRACT:
        raise ValueError("answer-first online service requires macro outputs")

    generation = VPFPOnlineGenerationConfig(
        **dict(payload.get("generation_config") or {})
    )
    generation.validate()
    equation = VPFPHighFidelityConfig.from_dict(
        dict(payload.get("equation_config") or {})
    )
    if abs(float(equation.internal_dt_max) - float(generation.internal_dt_max)) > 1.0e-12:
        raise ValueError("coarse equation and online generation time steps disagree")

    materialization_grid = dict(payload.get("materialization_grid") or {})
    materialization_time = dict(payload.get("materialization_time") or {})
    for name in ("nx", "ny", "nvx", "nvy"):
        if int(materialization_grid.get(name, 0)) < 1:
            raise ValueError(f"materialization grid {name} must be positive")
    if len(materialization_grid.get("x_domain") or ()) != 2:
        raise ValueError("materialization x_domain must have two entries")
    if len(materialization_grid.get("v_bounds") or ()) != 4:
        raise ValueError("materialization v_bounds must have four entries")
    if int(materialization_time.get("nt", 0)) < 3:
        raise ValueError("materialization time grid needs at least three frames")
    if float(materialization_time.get("t1", 0.0)) <= float(
        materialization_time.get("t0", 0.0)
    ):
        raise ValueError("materialization time domain must be increasing")

    guidance = dict(payload.get("guidance") or {})
    if guidance.get("mode") != (
        "direct_frozen_solver_error_best_of_fresh_answer_pool"
    ):
        raise ValueError("unexpected answer-first online guidance mode")
    pool_size = int(guidance.get("candidate_pool_size", 0))
    if pool_size < 2:
        raise ValueError("candidate pool size must be at least two")
    guidance["solver_budget"] = normalize_guidance_solver_budget(
        guidance.get("solver_budget")
    ).to_dict()
    payload["guidance"] = guidance

    limits = dict(payload.get("limits") or {})
    resource_budget = payload.get("development_resource_budget")
    resource_mode = resource_budget is not None
    if resource_mode:
        payload["development_resource_budget"] = (
            validate_answer_first_development_resource_budget(resource_budget)
        )
    for name in ("max_batch", "max_points_per_call"):
        if int(limits.get(name, 0)) < 1:
            raise ValueError(f"online limit {name} must be positive")
    if resource_mode:
        if limits.get("aggregate_limits_mode") != AGGREGATE_LIMITS_ACCOUNTING_ONLY:
            raise ValueError("resource-mode aggregate counts must be accounting-only")
        forbidden = {
            "max_samples",
            "max_candidate_answers",
            "max_answer_grids",
            "max_total_points",
        }
        if forbidden.intersection(limits):
            raise ValueError("resource mode must not publish aggregate count ceilings")
    else:
        resolution_capacity = int(
            dict(payload.get("resolution_materialization") or {}).get(
                "max_views", 0
            )
        )
        for name in (
            "max_samples",
            "max_candidate_answers",
            "max_answer_grids",
            "max_total_points",
        ):
            if int(limits.get(name, 0)) < 1:
                raise ValueError(f"online limit {name} must be positive")
        if int(limits["max_batch"]) > int(limits["max_samples"]):
            raise ValueError("max_batch cannot exceed max_samples")
        if int(limits["max_answer_grids"]) > (
            int(limits["max_samples"]) + resolution_capacity
        ):
            raise ValueError("answer-grid budget exceeds sample plus view capacity")
        if int(limits["max_points_per_call"]) > int(limits["max_total_points"]):
            raise ValueError("point-call budget cannot exceed total point budget")
        if int(limits["max_candidate_answers"]) < (
            int(limits["max_samples"]) * pool_size
        ):
            raise ValueError("candidate answer budget is too small")

    compute_budget = payload.get("compute_budget_contract_enabled", False)
    if not isinstance(compute_budget, bool):
        raise ValueError("compute-budget contract flag must be boolean")
    payload["compute_budget_contract_enabled"] = compute_budget
    resolution_policy = payload.get("resolution_materialization")
    if compute_budget and resolution_policy is None:
        raise ValueError("compute-budget contract requires materialization")
    if resolution_policy is not None:
        payload["resolution_materialization"] = (
            validate_answer_first_materialization_config(resolution_policy)
        )

    seed_starts = dict(payload.get("seed_starts") or {})
    if set(seed_starts) != {"train", "validation"}:
        raise ValueError("online seed starts must define train and validation")
    if int(seed_starts["train"]) == int(seed_starts["validation"]):
        raise ValueError("online train and validation seed streams must differ")
    seed_step = int(payload.get("seed_step", 1))
    if seed_step < 1:
        raise ValueError("online seed step must be positive")
    if resource_mode:
        if seed_step < 2:
            raise ValueError("resource mode requires strided disjoint seed streams")
        residues = {
            int(seed_starts["train"]) % seed_step,
            int(seed_starts["validation"]) % seed_step,
        }
        if len(residues) != 2:
            raise ValueError("online seed streams overlap modulo seed_step")
    payload["seed_step"] = seed_step
    return payload


def validate_answer_first_materialization_config(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    payload = dict(config)
    if payload.get("schema") != ANSWER_FIRST_RESOLUTION_MATERIALIZATION_SCHEMA:
        raise ValueError("unexpected answer-first materialization schema")
    if payload.get("enabled") is not True:
        raise ValueError("answer-first materialization must be enabled")
    aggregate_mode = str(payload.get("aggregate_limits_mode") or "enforced")
    if aggregate_mode not in {"enforced", AGGREGATE_LIMITS_ACCOUNTING_ONLY}:
        raise ValueError("unexpected materialization aggregate-limit mode")
    per_view = payload.get("max_target_source_elements_per_view")
    if isinstance(per_view, bool) or not isinstance(per_view, int) or per_view < 1:
        raise ValueError("per-view source-element limit must be positive")
    if aggregate_mode == "enforced":
        for name in ("max_views", "max_total_target_source_elements"):
            value = payload.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"materialization limit {name} must be positive")
        if int(per_view) > int(payload["max_total_target_source_elements"]):
            raise ValueError("per-view source limit exceeds cumulative limit")
    else:
        if {"max_views", "max_total_target_source_elements"}.intersection(payload):
            raise ValueError("accounting-only mode must omit aggregate ceilings")
    payload["aggregate_limits_mode"] = aggregate_mode

    maximum = dict(payload.get("max_resolution") or {})
    if set(maximum) != set(RESOLUTION_KEYS):
        raise ValueError(f"max_resolution keys must be {sorted(RESOLUTION_KEYS)}")
    normalized = {}
    for name in RESOLUTION_KEYS:
        value = maximum[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"maximum {name} must be an integer")
        normalized[name] = int(value)
    VPFPResolution.from_dict(normalized)
    payload["max_resolution"] = normalized
    return payload


def validate_answer_first_materialization_resolution(
    value: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
) -> VPFPResolution:
    resolution = VPFPResolution.from_dict(
        {str(name): item for name, item in dict(value).items()}
    )
    validated = validate_answer_first_materialization_config(policy)
    maximum = VPFPResolution.from_dict(dict(validated["max_resolution"]))
    for name in RESOLUTION_KEYS:
        observed = int(getattr(resolution, name))
        allowed = int(getattr(maximum, name))
        if observed > allowed:
            raise ValueError(f"requested {name}={observed} exceeds {allowed}")
    if resolution.source_element_count > int(
        validated["max_target_source_elements_per_view"]
    ):
        raise ValueError("requested source view exceeds the per-view element limit")
    return resolution


def validate_answer_first_development_resource_budget(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    payload = dict(config)
    if payload.get("schema") != ANSWER_FIRST_DEVELOPMENT_RESOURCE_SCHEMA:
        raise ValueError("unexpected answer-first development-resource schema")
    if payload.get("aggregate_query_counts") != AGGREGATE_LIMITS_ACCOUNTING_ONLY:
        raise ValueError("aggregate query counts must be accounting-only")
    for name in ("active_wall_time_seconds", "scratch_bytes"):
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"development resource {name} must be positive")
    allocation = dict(payload.get("allocation") or {})
    for name in ("gpus", "cpus", "memory_bytes"):
        value = allocation.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"development allocation {name} must be positive")
    final_solver = dict(payload.get("final_solver") or {})
    soft = final_solver.get("soft_wall_time_seconds_per_case")
    hard = final_solver.get("hard_wall_time_seconds_per_case")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in (soft, hard)
    ) or not 0.0 < float(soft) <= float(hard):
        raise ValueError("final solver budgets must satisfy 0 < soft <= hard")
    payload["allocation"] = allocation
    payload["final_solver"] = final_solver
    return payload


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True
