from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from agent_verifier.vpfp.continuous_v2 import _real_fourier_half_modes
from agent_verifier.vpfp.structured_diffusion import (
    VPFPStructuredDiffusionConfig,
)


RELIABILITY_SCHEMA = "vpfp_solution_distribution_reliability_v1"
RELIABILITY_ARTIFACT_SCHEMA = (
    "vpfp_solution_distribution_reliability_weights_v1"
)

UNRESOLVED = np.uint8(0)
TRANSITION = np.uint8(1)
TRUSTED = np.uint8(2)


@dataclass(frozen=True)
class VPFPSolutionReliabilityConfig:
    """Calibration policy for low-fidelity solution observations.

    The score measures whether a coordinate of the *solution representation*
    is repeatable under equation-first discretization refinement. It does not
    test whether a generated answer follows source-free VPFP dynamics.
    """

    trusted_minimum: float = 0.80
    transition_minimum: float = 0.30
    minimum_signal_variance: float = 1.0e-4
    nonconvergence_penalty_power: float = 1.0
    epsilon: float = 1.0e-12

    def validate(self) -> None:
        if not 0.0 <= float(self.transition_minimum) < float(
            self.trusted_minimum
        ) <= 1.0:
            raise ValueError("reliability thresholds must satisfy 0 <= transition < trusted <= 1")
        if float(self.minimum_signal_variance) < 0.0:
            raise ValueError("minimum signal variance cannot be negative")
        if float(self.nonconvergence_penalty_power) < 0.0:
            raise ValueError("nonconvergence penalty power cannot be negative")
        if float(self.epsilon) <= 0.0:
            raise ValueError("reliability epsilon must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VPFPSolutionReliabilityMap:
    latent_config: VPFPStructuredDiffusionConfig
    reliability: np.ndarray
    roles: np.ndarray
    groups: tuple[dict[str, Any], ...]
    calibration: dict[str, Any]

    def __post_init__(self) -> None:
        reliability = np.asarray(self.reliability, dtype=np.float32)
        roles = np.asarray(self.roles, dtype=np.uint8)
        expected = (int(self.latent_config.active_latent_dim),)
        if reliability.shape != expected or roles.shape != expected:
            raise ValueError("reliability vectors do not match the canonical latent")
        if not np.all(np.isfinite(reliability)):
            raise ValueError("reliability values must be finite")
        if np.min(reliability) < 0.0 or np.max(reliability) > 1.0:
            raise ValueError("reliability values must lie in [0,1]")
        if not np.all(np.isin(roles, (UNRESOLVED, TRANSITION, TRUSTED))):
            raise ValueError("unknown solution reliability role")

    def projection_weights(self, mode: str = "trusted") -> np.ndarray:
        if mode == "trusted":
            return (self.roles == TRUSTED).astype(np.float32)
        if mode == "trusted_and_transition":
            return (self.roles >= TRANSITION).astype(np.float32)
        if mode == "soft":
            return np.sqrt(np.asarray(self.reliability, dtype=np.float32))
        raise ValueError("projection mode must be trusted, trusted_and_transition, or soft")

    def apply(self, values, *, mode: str = "trusted"):
        """Project NumPy or torch rows without changing their layout."""

        expected = int(self.latent_config.active_latent_dim)
        if int(values.shape[-1]) != expected:
            raise ValueError("solution latent does not match reliability map")
        weights = self.projection_weights(mode)
        try:
            import torch
        except ImportError:  # pragma: no cover - NumPy-only environment
            torch = None
        if torch is not None and isinstance(values, torch.Tensor):
            tensor_weights = torch.as_tensor(
                weights, dtype=values.dtype, device=values.device
            )
            return values * tensor_weights
        return np.asarray(values) * weights

    def summary(self) -> dict[str, Any]:
        role_counts = {
            "unresolved": int(np.count_nonzero(self.roles == UNRESOLVED)),
            "transition": int(np.count_nonzero(self.roles == TRANSITION)),
            "trusted": int(np.count_nonzero(self.roles == TRUSTED)),
        }
        total = float(self.roles.size)
        return {
            "coordinate_count": int(self.roles.size),
            "role_counts": role_counts,
            "role_fractions": {
                key: float(value / total) for key, value in role_counts.items()
            },
            "reliability_mean": float(np.mean(self.reliability)),
            "reliability_median": float(np.median(self.reliability)),
        }


def fit_multiresolution_solution_reliability(
    levels: Sequence[np.ndarray],
    *,
    latent_config: VPFPStructuredDiffusionConfig,
    level_metadata: Sequence[Mapping[str, Any]] | None = None,
    rho_time_control_count: int | None = None,
    local_time_control_count: int | None = None,
    config: VPFPSolutionReliabilityConfig | None = None,
    normalization_sha256: str | None = None,
) -> VPFPSolutionReliabilityMap:
    """Fit coordinate reliability from paired coarse-to-fine observations.

    Every level must contain the same cases in the same order and use the same
    canonical normalization. Only the last three levels enter the score; any
    additional coarser levels are retained in provenance for auditing.
    """

    policy = config or VPFPSolutionReliabilityConfig()
    policy.validate()
    latent_config.validate()
    arrays = _validate_levels(levels, latent_config)
    if level_metadata is None:
        metadata = tuple({"level_index": index} for index in range(len(arrays)))
    else:
        if len(level_metadata) != len(arrays):
            raise ValueError("level metadata count does not match observations")
        metadata = tuple(dict(row) for row in level_metadata)

    coarse, medium, fine = arrays[-3:]
    descriptors, group_indices = _coordinate_groups(
        latent_config,
        rho_time_control_count=rho_time_control_count,
        local_time_control_count=local_time_control_count,
    )
    reliability = np.zeros(latent_config.active_latent_dim, dtype=np.float32)
    roles = np.zeros(latent_config.active_latent_dim, dtype=np.uint8)
    rows: list[dict[str, Any]] = []
    epsilon = float(policy.epsilon)
    for descriptor, indices in zip(descriptors, group_indices, strict=True):
        finest_values = fine[:, indices]
        signal = float(np.mean(np.var(finest_values, axis=0, ddof=0)))
        fine_discrepancy = float(
            np.mean(np.square(fine[:, indices] - medium[:, indices]))
        )
        coarse_discrepancy = float(
            np.mean(np.square(medium[:, indices] - coarse[:, indices]))
        )
        if signal < float(policy.minimum_signal_variance):
            score = 0.0
            reason = "below_signal_floor"
        else:
            score = signal / (signal + fine_discrepancy + epsilon)
            reason = "refinement_repeatability"
            if fine_discrepancy >= coarse_discrepancy:
                ratio = (coarse_discrepancy + epsilon) / (
                    fine_discrepancy + epsilon
                )
                score *= min(1.0, ratio) ** float(
                    policy.nonconvergence_penalty_power
                )
                reason = "nonconvergent_refinement_penalized"
        score = float(np.clip(score, 0.0, 1.0))
        role = _role_from_score(score, policy)
        reliability[indices] = np.float32(score)
        roles[indices] = role
        rows.append(
            {
                **descriptor,
                "coordinate_count": int(indices.size),
                "signal_variance": signal,
                "coarse_discrepancy_mse": coarse_discrepancy,
                "fine_discrepancy_mse": fine_discrepancy,
                "fine_to_coarse_discrepancy_ratio": float(
                    fine_discrepancy / (coarse_discrepancy + epsilon)
                ),
                "reliability": score,
                "role": _role_name(role),
                "reason": reason,
            }
        )

    calibration = {
        "schema": RELIABILITY_SCHEMA,
        "policy": policy.to_dict(),
        "level_count": int(len(arrays)),
        "paired_case_count": int(fine.shape[0]),
        "levels_coarse_to_fine": list(metadata),
        "normalization_sha256": normalization_sha256,
        "rho_time_control_count": rho_time_control_count,
        "local_time_control_count": local_time_control_count,
        "semantics": {
            "primary_object": "continuous_hidden_solution_function",
            "equation_first_role": "disposable_low_fidelity_solution_observation",
            "score_meaning": "solution_shape_repeatability_under_discretization_refinement",
            "not_enforced": [
                "source_free_dynamics",
                "phase_mixing_law",
                "collision_relaxation_law",
                "conservation_of_the_equation_first_case",
                "matching_the_equation_first_public_source",
            ],
            "answer_first_public_source": "derived_from_each_generated_hidden_solution",
        },
    }
    return VPFPSolutionReliabilityMap(
        latent_config=latent_config,
        reliability=reliability,
        roles=roles,
        groups=tuple(rows),
        calibration=calibration,
    )


def save_solution_reliability_map(
    output_directory: str | Path,
    reliability_map: VPFPSolutionReliabilityMap,
) -> dict[str, Any]:
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    weights_path = output / "solution_reliability_weights.npz"
    np.savez_compressed(
        weights_path,
        schema=np.asarray(RELIABILITY_ARTIFACT_SCHEMA),
        reliability=np.asarray(reliability_map.reliability, dtype=np.float32),
        roles=np.asarray(reliability_map.roles, dtype=np.uint8),
    )
    report = {
        **reliability_map.calibration,
        "latent_config": reliability_map.latent_config.to_dict(),
        "summary": reliability_map.summary(),
        "groups": list(reliability_map.groups),
        "weights_artifact": str(weights_path),
        "weights_artifact_sha256": _sha256(weights_path),
    }
    report_path = output / "solution_reliability_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"report": report_path, "weights": weights_path, "payload": report}


def load_solution_reliability_map(
    report_path: str | Path,
) -> VPFPSolutionReliabilityMap:
    path = Path(report_path).resolve()
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != RELIABILITY_SCHEMA:
        raise ValueError("unexpected solution reliability report schema")
    weights_path = Path(report["weights_artifact"]).resolve()
    with np.load(weights_path, allow_pickle=False) as payload:
        if str(payload["schema"].item()) != RELIABILITY_ARTIFACT_SCHEMA:
            raise ValueError("unexpected solution reliability artifact schema")
        reliability = np.asarray(payload["reliability"], dtype=np.float32)
        roles = np.asarray(payload["roles"], dtype=np.uint8)
    calibration = {
        key: value
        for key, value in report.items()
        if key
        not in {
            "latent_config",
            "summary",
            "groups",
            "weights_artifact",
            "weights_artifact_sha256",
        }
    }
    return VPFPSolutionReliabilityMap(
        latent_config=VPFPStructuredDiffusionConfig.from_dict(
            dict(report["latent_config"])
        ),
        reliability=reliability,
        roles=roles,
        groups=tuple(dict(row) for row in report["groups"]),
        calibration=calibration,
    )


def _validate_levels(
    levels: Sequence[np.ndarray],
    config: VPFPStructuredDiffusionConfig,
) -> tuple[np.ndarray, ...]:
    if len(levels) < 3:
        raise ValueError("solution reliability calibration needs at least three levels")
    arrays = tuple(np.asarray(values, dtype=np.float64) for values in levels)
    expected_tail = int(config.active_latent_dim)
    first_shape = arrays[0].shape
    if len(first_shape) != 2 or first_shape[0] < 2 or first_shape[1] != expected_tail:
        raise ValueError("reliability observations must have shape [case, active_latent]")
    for values in arrays:
        if values.shape != first_shape:
            raise ValueError("paired reliability levels must have identical shapes")
        if not np.all(np.isfinite(values)):
            raise ValueError("reliability observations must be finite")
    return arrays


def _coordinate_groups(
    config: VPFPStructuredDiffusionConfig,
    *,
    rho_time_control_count: int | None,
    local_time_control_count: int | None,
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    group_coordinates: dict[tuple[Any, ...], list[int]] = {}
    descriptors: dict[tuple[Any, ...], dict[str, Any]] = {}

    rho_modes = _real_fourier_half_modes(
        _rho_max_mode_from_count(config.rho_mode_count),
        _rho_max_mode_from_count(config.rho_mode_count),
    )
    if len(rho_modes) != int(config.rho_mode_count):
        raise ValueError("rho Fourier mode layout is not square")
    for time_index in range(int(config.rho_time_count)):
        time_kind, time_slot = _time_role(
            time_index,
            int(config.rho_time_count),
            rho_time_control_count,
        )
        for mode_index, (kx, ky) in enumerate(rho_modes):
            shell = max(abs(int(kx)), abs(int(ky)))
            key = ("rho", time_kind, time_slot, shell)
            descriptors.setdefault(
                key,
                {
                    "block": "rho",
                    "time_role": time_kind,
                    "time_slot": int(time_slot),
                    "spatial_mode_shell": int(shell),
                },
            )
            start = time_index * int(config.rho_feature_count) + 2 * mode_index
            group_coordinates.setdefault(key, []).extend((start, start + 1))

    local_offset = int(config.rho_latent_dim)
    local_modes = _real_fourier_half_modes(
        int(config.local_spatial_mode), int(config.local_spatial_mode)
    )
    degrees = _channel_degrees(int(config.local_channels))
    feature_count = int(config.local_spatial_feature_count)
    time_count = int(config.local_time_count)
    for channel, degree in enumerate(degrees):
        for time_index in range(time_count):
            time_kind, time_slot = _time_role(
                time_index, time_count, local_time_control_count
            )
            base = local_offset + (channel * time_count + time_index) * feature_count
            constant_key = ("micro", time_kind, time_slot, degree, 0)
            descriptors.setdefault(
                constant_key,
                {
                    "block": "micro",
                    "time_role": time_kind,
                    "time_slot": int(time_slot),
                    "velocity_degree": int(degree),
                    "spatial_mode_shell": 0,
                },
            )
            group_coordinates.setdefault(constant_key, []).append(base)
            for mode_index, (kx, ky) in enumerate(local_modes):
                shell = max(abs(int(kx)), abs(int(ky)))
                key = ("micro", time_kind, time_slot, degree, shell)
                descriptors.setdefault(
                    key,
                    {
                        "block": "micro",
                        "time_role": time_kind,
                        "time_slot": int(time_slot),
                        "velocity_degree": int(degree),
                        "spatial_mode_shell": int(shell),
                    },
                )
                start = base + 1 + 2 * mode_index
                group_coordinates.setdefault(key, []).extend((start, start + 1))

    keys = list(group_coordinates)
    covered = np.concatenate(
        [np.asarray(group_coordinates[key], dtype=np.int64) for key in keys]
    )
    if covered.size != int(config.active_latent_dim):
        raise RuntimeError("solution reliability grouping changed latent size")
    if not np.array_equal(np.sort(covered), np.arange(covered.size)):
        raise RuntimeError("solution reliability groups overlap or leave gaps")
    return (
        [descriptors[key] for key in keys],
        [np.asarray(group_coordinates[key], dtype=np.int64) for key in keys],
    )


def _rho_max_mode_from_count(mode_count: int) -> int:
    side_squared = 2 * int(mode_count) + 1
    side = int(round(np.sqrt(float(side_squared))))
    if side * side != side_squared or side % 2 != 1:
        raise ValueError("rho mode count is not a square real-Fourier layout")
    return (side - 1) // 2


def _channel_degrees(channel_count: int) -> tuple[int, ...]:
    degree = 0
    while (degree + 1) * (degree + 2) // 2 < int(channel_count):
        degree += 1
    if (degree + 1) * (degree + 2) // 2 == int(channel_count):
        return tuple(
            total
            for total in range(degree + 1)
            for _first in range(total + 1)
        )
    return tuple(range(int(channel_count)))


def _time_role(
    time_index: int,
    time_count: int,
    control_count: int | None,
) -> tuple[str, int]:
    if control_count is None:
        return "direct", int(time_index)
    count = int(control_count)
    if count < 1 or int(time_count) not in {count, 2 * count}:
        raise ValueError("time control count is inconsistent with latent layout")
    if int(time_count) == count:
        return "value", int(time_index)
    return (
        ("value", int(time_index))
        if int(time_index) < count
        else ("slope", int(time_index) - count)
    )


def _role_from_score(
    score: float, config: VPFPSolutionReliabilityConfig
) -> np.uint8:
    if score >= float(config.trusted_minimum):
        return TRUSTED
    if score >= float(config.transition_minimum):
        return TRANSITION
    return UNRESOLVED


def _role_name(role: np.uint8) -> str:
    return {
        int(UNRESOLVED): "unresolved",
        int(TRANSITION): "transition",
        int(TRUSTED): "trusted",
    }[int(role)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
