from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from agent_verifier.vpfp.continuous_macros import MACRO_OUTPUT_CONTRACT


STRUCTURED_ONLINE_GENERATOR_CONFIG_SCHEMA = (
    "vpfp_structured_guided_online_generator_controller_config_v1"
)
STRUCTURED_ABSOLUTE_SELECTION_MODE = "absolute_candidate_selection_v1"
STRUCTURED_COMBINED_GUIDANCE_MODE = (
    "absolute_selection_plus_centered_local_guidance_v1"
)
STRUCTURED_GUIDANCE_MODE = STRUCTURED_ABSOLUTE_SELECTION_MODE
SUPPORTED_STRUCTURED_GUIDANCE_MODES = {
    STRUCTURED_ABSOLUTE_SELECTION_MODE,
    STRUCTURED_COMBINED_GUIDANCE_MODE,
}


def validate_structured_controller_config(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    payload = dict(config)
    if payload.get("schema") != STRUCTURED_ONLINE_GENERATOR_CONFIG_SCHEMA:
        raise ValueError("unexpected structured online-generator config schema")
    required_paths = (
        "workspace",
        "private_root",
        "guidance_report",
        "fresh_guidance_audit",
        "source_audit_report",
        "latent_artifact",
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
        raise ValueError(
            "structured controller private root and workspace must be disjoint"
        )
    if not str(workspace).startswith("/scratch/") or not str(
        private_root
    ).startswith("/scratch/"):
        raise ValueError(
            "structured online workspace and private root must be under /scratch"
        )
    if str(payload.get("solver_output_contract_version") or "") != (
        MACRO_OUTPUT_CONTRACT
    ):
        raise ValueError(
            "structured online service requires the macro output contract"
        )
    target_sha = str(payload.get("target_solver_sha256") or "")
    if len(target_sha) != 64 or any(
        character not in "0123456789abcdef" for character in target_sha
    ):
        raise ValueError("target_solver_sha256 must be a lowercase SHA-256 digest")

    limits = dict(payload.get("limits") or {})
    for name in (
        "max_batch",
        "max_samples",
        "max_candidate_trajectories",
        "max_answer_grids",
        "max_points_per_call",
        "max_total_points",
    ):
        if int(limits.get(name, 0)) < 1:
            raise ValueError(f"structured online limit {name} must be positive")
    if int(limits["max_batch"]) > int(limits["max_samples"]):
        raise ValueError("max_batch cannot exceed max_samples")
    if int(limits["max_answer_grids"]) > int(limits["max_samples"]):
        raise ValueError("max_answer_grids cannot exceed max_samples")
    if int(limits["max_points_per_call"]) > int(limits["max_total_points"]):
        raise ValueError("max_points_per_call cannot exceed max_total_points")

    guidance = dict(payload.get("guidance") or {})
    mode = str(guidance.get("mode") or "")
    if mode not in SUPPORTED_STRUCTURED_GUIDANCE_MODES:
        raise ValueError("unexpected structured guidance mode")
    local_enabled = bool(
        guidance.get(
            "centered_local_guidance_enabled",
            mode == STRUCTURED_COMBINED_GUIDANCE_MODE,
        )
    )
    if mode == STRUCTURED_ABSOLUTE_SELECTION_MODE and local_enabled:
        raise ValueError("absolute-selection mode cannot enable local guidance")
    if mode == STRUCTURED_COMBINED_GUIDANCE_MODE and not local_enabled:
        raise ValueError("combined mode requires centered local guidance")
    pool_size = int(guidance.get("candidate_pool_size", 0))
    if pool_size < 2 or pool_size > 16:
        raise ValueError("structured candidate pool must lie in [2, 16]")
    if int(limits["max_candidate_trajectories"]) < (
        int(limits["max_samples"]) * pool_size
    ):
        raise ValueError("candidate trajectory budget is insufficient")
    for name in (
        "maximum_gradient_norm",
        "maximum_local_rms_delta",
        "maximum_coordinate_delta",
    ):
        if float(guidance.get(name, 0.0)) <= 0.0:
            raise ValueError(f"structured guidance value {name} must be positive")
    centered_scale = float(guidance.get("centered_guidance_scale", 0.0))
    if local_enabled and centered_scale <= 0.0:
        raise ValueError(
            "structured centered guidance scale must be positive when enabled"
        )
    if not local_enabled and centered_scale != 0.0:
        raise ValueError(
            "structured centered guidance scale must be zero when disabled"
        )
    if not bool(guidance.get("freeze_rho", False)):
        raise ValueError("structured online guidance must freeze rho coordinates")

    ranges = dict(payload.get("condition_ranges") or {})
    if set(ranges) != {"train", "validation"}:
        raise ValueError(
            "structured online service requires train/validation condition ranges"
        )
    intervals = []
    full_count = int(payload.get("full_validation_condition_count", 0))
    for name in ("train", "validation"):
        current = dict(ranges[name])
        offset = int(current.get("offset", -1))
        count = int(current.get("count", 0))
        if offset < 0 or count < 1 or offset + count > full_count:
            raise ValueError(f"structured condition range {name} is invalid")
        intervals.append((offset, offset + count))
    if max(intervals[0][0], intervals[1][0]) < min(
        intervals[0][1], intervals[1][1]
    ):
        raise ValueError("structured train/validation condition ranges overlap")

    seeds = dict(payload.get("sampling_seed_starts") or {})
    if set(seeds) != {"train", "validation"}:
        raise ValueError("structured online service requires two sampling seeds")
    if int(seeds["train"]) == int(seeds["validation"]):
        raise ValueError("structured online sampling seeds must be disjoint")
    for section, names in (("grid", ("nx", "ny", "nvx", "nvy")), ("time", ("nt",))):
        values = dict(payload.get(section) or {})
        for name in names:
            if int(values.get(name, 0)) < 1:
                raise ValueError(f"structured materialization {section}.{name} is invalid")
    return payload


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True
