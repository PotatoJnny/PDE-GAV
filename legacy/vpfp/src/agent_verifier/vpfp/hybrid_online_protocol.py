from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from agent_verifier.vpfp.continuous_macros import MACRO_OUTPUT_CONTRACT


HYBRID_ONLINE_GENERATOR_CONFIG_SCHEMA = (
    "vpfp_hybrid_online_generator_controller_config_v2"
)


def validate_hybrid_controller_config(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    payload = dict(config)
    if payload.get("schema") != HYBRID_ONLINE_GENERATOR_CONFIG_SCHEMA:
        raise ValueError("unexpected hybrid online-generator config schema")
    required_paths = (
        "workspace",
        "private_root",
        "hybrid_report",
        "guidance_report",
        "fresh_guidance_audit",
        "reward_checkpoint",
    )
    resolved: dict[str, Path] = {}
    for name in required_paths:
        path = Path(str(payload.get(name) or "")).resolve()
        if name == "workspace":
            if not path.is_dir() or path.is_symlink():
                raise FileNotFoundError(path)
        elif name == "private_root":
            if path.exists():
                raise FileExistsError(path)
        elif not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
        resolved[name] = path
    workspace = resolved["workspace"]
    private_root = resolved["private_root"]
    if _is_within(private_root, workspace) or _is_within(workspace, private_root):
        raise ValueError("hybrid controller private root and workspace must be disjoint")
    if not str(workspace).startswith("/scratch/") or not str(private_root).startswith(
        "/scratch/"
    ):
        raise ValueError("hybrid online workspace and private root must be under /scratch")
    if str(payload.get("solver_output_contract_version") or "") != (
        MACRO_OUTPUT_CONTRACT
    ):
        raise ValueError("hybrid online service requires the macro output contract")
    target_sha = str(payload.get("target_solver_sha256") or "")
    if len(target_sha) != 64 or any(
        char not in "0123456789abcdef" for char in target_sha
    ):
        raise ValueError("target_solver_sha256 must be a lowercase SHA-256 digest")
    limits = dict(payload.get("limits") or {})
    for name in (
        "max_batch",
        "max_samples",
        "max_candidate_macros",
        "max_answer_grids",
        "max_points_per_call",
        "max_total_points",
    ):
        if int(limits.get(name, 0)) < 1:
            raise ValueError(f"hybrid online limit {name} must be positive")
    if int(limits["max_batch"]) > int(limits["max_samples"]):
        raise ValueError("max_batch cannot exceed max_samples")
    if int(limits["max_answer_grids"]) > int(limits["max_samples"]):
        raise ValueError("max_answer_grids cannot exceed max_samples")
    if int(limits["max_candidate_macros"]) < int(limits["max_samples"]):
        raise ValueError("candidate macro budget cannot be below output sample budget")
    if int(limits["max_points_per_call"]) > int(limits["max_total_points"]):
        raise ValueError("max_points_per_call cannot exceed max_total_points")
    guidance = dict(payload.get("guidance") or {})
    if guidance.get("mode") != "absolute_log_reward_candidate_pool_selection":
        raise ValueError("unexpected hybrid macro guidance mode")
    pool_size = int(guidance.get("candidate_pool_size", 0))
    if pool_size < 2 or pool_size > 16:
        raise ValueError("hybrid macro candidate pool must lie in [2, 16]")
    if bool(guidance.get("micro_guidance_enabled", True)):
        raise ValueError("failed micro guidance must remain disabled")
    if int(limits["max_candidate_macros"]) < int(limits["max_samples"]) * pool_size:
        raise ValueError("candidate macro budget is insufficient for all-guided use")
    grid = dict(payload.get("grid") or {})
    time = dict(payload.get("time") or {})
    for name in ("nx", "ny", "nvx", "nvy"):
        if int(grid.get(name, 0)) < 1:
            raise ValueError(f"hybrid materialization grid {name} must be positive")
    if int(time.get("nt", 0)) < 1:
        raise ValueError("hybrid materialization nt must be positive")
    seed_starts = dict(payload.get("seed_starts") or {})
    if set(seed_starts) != {"train", "validation"}:
        raise ValueError("hybrid online service requires train/validation seed streams")
    if int(seed_starts["train"]) == int(seed_starts["validation"]):
        raise ValueError("hybrid online seed streams must be disjoint")
    return payload


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True
