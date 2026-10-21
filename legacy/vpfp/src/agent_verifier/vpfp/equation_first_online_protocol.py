from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from agent_verifier.vpfp.continuous_macros import MACRO_OUTPUT_CONTRACT
from agent_verifier.vpfp.equation_first_macro import (
    VPFPEquationFirstMacroConfig,
)
from agent_verifier.vpfp.resolution_convergence import VPFPResolution


EQUATION_FIRST_ONLINE_GENERATOR_CONFIG_SCHEMA = (
    "vpfp_equation_first_online_generator_controller_config_v1"
)
EQUATION_FIRST_RESOLUTION_MATERIALIZATION_SCHEMA = (
    "vpfp_equation_first_resolution_materialization_v1"
)
EQUATION_FIRST_DEVELOPMENT_RESOURCE_SCHEMA = (
    "vpfp_equation_first_agent_development_resource_v1"
)
AGGREGATE_LIMITS_ACCOUNTING_ONLY = "accounting_only"
RESOLUTION_KEYS = ("nx", "ny", "nvx", "nvy", "nt")


def validate_equation_first_controller_config(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    payload = dict(config)
    if payload.get("schema") != EQUATION_FIRST_ONLINE_GENERATOR_CONFIG_SCHEMA:
        raise ValueError("unexpected equation-first online controller schema")
    workspace = Path(str(payload.get("workspace") or "")).resolve()
    private_root = Path(str(payload.get("private_root") or "")).resolve()
    target_solver = Path(str(payload.get("target_solver") or "")).resolve()
    if not workspace.is_dir() or workspace.is_symlink():
        raise FileNotFoundError(workspace)
    if not target_solver.is_file() or target_solver.is_symlink():
        raise FileNotFoundError(target_solver)
    if not str(workspace).startswith("/scratch/") or not str(
        private_root
    ).startswith("/scratch/"):
        raise ValueError("workspace and private root must stay under /scratch")
    if _is_within(private_root, workspace) or _is_within(workspace, private_root):
        raise ValueError("controller-private root and Agent workspace must be disjoint")
    target_sha = str(payload.get("target_solver_sha256") or "")
    if len(target_sha) != 64 or any(
        character not in "0123456789abcdef" for character in target_sha
    ):
        raise ValueError("target solver SHA-256 is malformed")
    if payload.get("solver_output_contract_version") != MACRO_OUTPUT_CONTRACT:
        raise ValueError("equation-first online service requires macro outputs")
    VPFPEquationFirstMacroConfig.from_dict(
        dict(payload.get("generator_config") or {})
    )
    guidance = dict(payload.get("guidance") or {})
    if guidance.get("mode") != (
        "direct_frozen_solver_error_best_of_fresh_equation_pool"
    ):
        raise ValueError("unexpected equation-first online guidance mode")
    pool_size = int(guidance.get("candidate_pool_size", 0))
    if pool_size < 2:
        raise ValueError("candidate pool size must be at least two")
    limits = dict(payload.get("limits") or {})
    resource_budget = payload.get("development_resource_budget")
    resource_mode = resource_budget is not None
    if resource_mode:
        payload["development_resource_budget"] = (
            validate_development_resource_budget(resource_budget)
        )
    resolution_policy = payload.get("resolution_materialization")
    compute_budget_contract = payload.get(
        "compute_budget_contract_enabled", False
    )
    if not isinstance(compute_budget_contract, bool):
        raise ValueError("compute-budget contract flag must be boolean")
    payload["compute_budget_contract_enabled"] = compute_budget_contract
    if compute_budget_contract and resolution_policy is None:
        raise ValueError("compute-budget contract requires numerical materialization")
    if resolution_policy is not None:
        validated_resolution_policy = (
            validate_resolution_materialization_config(resolution_policy)
        )
        payload["resolution_materialization"] = validated_resolution_policy
    for name in ("max_batch", "max_points_per_call"):
        if int(limits.get(name, 0)) < 1:
            raise ValueError(f"online limit {name} must be positive")
    if resource_mode:
        if limits.get("aggregate_limits_mode") != AGGREGATE_LIMITS_ACCOUNTING_ONLY:
            raise ValueError(
                "resource-mode aggregate query counts must be accounting-only"
            )
        forbidden = {
            "max_samples",
            "max_candidate_equation_solves",
            "max_answer_grids",
            "max_total_points",
        }
        if forbidden.intersection(limits):
            raise ValueError(
                "resource mode must not publish aggregate count ceilings"
            )
    else:
        resolution_view_capacity = int(
            dict(payload.get("resolution_materialization") or {}).get(
                "max_views", 0
            )
        )
        for name in (
            "max_samples",
            "max_candidate_equation_solves",
            "max_answer_grids",
            "max_total_points",
        ):
            if int(limits.get(name, 0)) < 1:
                raise ValueError(f"online limit {name} must be positive")
        if int(limits["max_batch"]) > int(limits["max_samples"]):
            raise ValueError("max_batch cannot exceed max_samples")
        if int(limits["max_answer_grids"]) > (
            int(limits["max_samples"]) + resolution_view_capacity
        ):
            raise ValueError(
                "answer-grid budget exceeds sample plus resolution-view capacity"
            )
        if int(limits["max_points_per_call"]) > int(limits["max_total_points"]):
            raise ValueError("point-call budget cannot exceed total point budget")
        if int(limits["max_candidate_equation_solves"]) < (
            int(limits["max_samples"]) * pool_size
        ):
            raise ValueError("candidate equation-solve budget is too small")
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
    resolution_policy = payload.get("resolution_materialization")
    if resolution_policy is not None:
        payload["resolution_materialization"] = (
            validate_resolution_materialization_config(resolution_policy)
        )
    return payload


def validate_resolution_materialization_config(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    payload = dict(config)
    if payload.get("schema") != EQUATION_FIRST_RESOLUTION_MATERIALIZATION_SCHEMA:
        raise ValueError("unexpected resolution-materialization schema")
    if payload.get("enabled") is not True:
        raise ValueError("resolution materialization must be explicitly enabled")
    aggregate_mode = str(payload.get("aggregate_limits_mode") or "enforced")
    if aggregate_mode not in {"enforced", AGGREGATE_LIMITS_ACCOUNTING_ONLY}:
        raise ValueError("unexpected resolution aggregate-limit mode")
    for name in (
        "max_target_source_elements_per_view",
        "max_reference_state_elements_per_view",
    ):
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"resolution limit {name} must be a positive integer")
    if aggregate_mode == "enforced":
        for name in (
            "max_views",
            "max_total_target_source_elements",
            "max_total_reference_state_elements",
        ):
            value = payload.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(
                    f"resolution limit {name} must be a positive integer"
                )
        if int(payload["max_target_source_elements_per_view"]) > int(
            payload["max_total_target_source_elements"]
        ):
            raise ValueError(
                "per-view target limit exceeds the cumulative target limit"
            )
        if int(payload["max_reference_state_elements_per_view"]) > int(
            payload["max_total_reference_state_elements"]
        ):
            raise ValueError(
                "per-view reference limit exceeds the cumulative reference limit"
            )
    else:
        forbidden = {
            "max_views",
            "max_total_target_source_elements",
            "max_total_reference_state_elements",
        }
        if forbidden.intersection(payload):
            raise ValueError(
                "accounting-only resolution mode must not publish aggregate ceilings"
            )
    payload["aggregate_limits_mode"] = aggregate_mode
    maximum = dict(payload.get("max_resolution") or {})
    if set(maximum) != set(RESOLUTION_KEYS):
        raise ValueError(
            f"max_resolution keys must be {sorted(RESOLUTION_KEYS)}"
        )
    normalized_maximum = {}
    for name in RESOLUTION_KEYS:
        value = maximum[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"maximum {name} must be an integer")
        normalized_maximum[name] = int(value)
    VPFPResolution.from_dict(normalized_maximum)
    payload["max_resolution"] = normalized_maximum
    return payload


def validate_development_resource_budget(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    payload = dict(config)
    if payload.get("schema") != EQUATION_FIRST_DEVELOPMENT_RESOURCE_SCHEMA:
        raise ValueError("unexpected equation-first development-resource schema")
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
        raise ValueError(
            "final-solver per-case budgets must satisfy 0 < soft <= hard"
        )
    payload["allocation"] = allocation
    payload["final_solver"] = final_solver
    return payload


def validate_materialization_resolution(
    value: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
) -> VPFPResolution:
    normalized = {str(name): item for name, item in dict(value).items()}
    resolution = VPFPResolution.from_dict(normalized)
    validated_policy = validate_resolution_materialization_config(policy)
    maximum = VPFPResolution.from_dict(dict(validated_policy["max_resolution"]))
    for name in RESOLUTION_KEYS:
        observed = int(getattr(resolution, name))
        allowed = int(getattr(maximum, name))
        if observed > allowed:
            raise ValueError(
                f"requested {name}={observed} exceeds the public limit {allowed}"
            )
    if resolution.source_element_count > int(
        validated_policy["max_target_source_elements_per_view"]
    ):
        raise ValueError(
            "requested resolution exceeds the per-view source-element limit: "
            f"{resolution.source_element_count} > "
            f"{int(validated_policy['max_target_source_elements_per_view'])}"
        )
    reference = joint_reference_resolution(resolution)
    if reference.source_element_count > int(
        validated_policy["max_reference_state_elements_per_view"]
    ):
        raise ValueError(
            "requested resolution exceeds the per-view refined-reference limit: "
            f"{reference.source_element_count} > "
            f"{int(validated_policy['max_reference_state_elements_per_view'])}"
        )
    return resolution


def joint_reference_resolution(target: VPFPResolution) -> VPFPResolution:
    return VPFPResolution(
        nx=2 * int(target.nx),
        ny=2 * int(target.ny),
        nvx=2 * (int(target.nvx) - 1) + 1,
        nvy=2 * (int(target.nvy) - 1) + 1,
        nt=2 * (int(target.nt) - 1) + 1,
    )


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True
