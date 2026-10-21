"""Immutable, equation-neutral scientific resources exposed to an Agent.

The controller may keep the manufactured-answer parameterization and
unreleased answers private.  The equation, numerical conventions, reference
operator code, and documented numerical-access clients belong here instead of
being reconstructed in equation-specific Agent prompts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from types import MappingProxyType
from typing import Mapping


PUBLIC_PROBLEM_PRIVATE_INFORMATION = (
    "generator_parameters",
    "latent_solution_parameterization",
    "unreleased_evaluation_answers",
)

TWO_PHASE_CONTINUOUS_SOLUTION_CONTRACT = MappingProxyType(
    {
        "abi": "prepare_solution_then_evaluate_points_v1",
        "prepare_solution": MappingProxyType(
            {
                "inputs": ("case", "solve_budget"),
                "solve_wall_limit": "controller_configured_finite_hard_limit",
                "scoring_coordinates_visible": False,
                "internal_grid_or_basis": "agent_selected",
            }
        ),
        "evaluate": MappingProxyType(
            {
                "input": "AnswerQuery(kind=points)",
                "primary_access": "physical_points",
                "points_capability_required": True,
                "requested_fields": "controller_selected",
                "projection_to_requested_physical_coordinates": "candidate_owned",
                "query_latency_accounted_separately": True,
            }
        ),
        "excluded_from_solve_and_query_time": (
            "controller_truth_materialization",
            "controller_macro_matching",
        ),
        "grid_queries_are_development_convenience": True,
        "applies_to": ("development", "leaderboard", "final"),
    }
)

# Backward-compatible name for callers that only inspected point capability.
CONTINUOUS_POINT_SOLUTION_CONTRACT = TWO_PHASE_CONTINUOUS_SOLUTION_CONTRACT


@dataclass(frozen=True)
class PublicResource:
    """One immutable file in an Agent-visible scientific resource bundle."""

    relative_path: str
    data: bytes
    media_type: str = "application/octet-stream"
    description: str = ""
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        path = _public_relative_path(self.relative_path)
        if not isinstance(self.data, bytes) or not self.data:
            raise ValueError("a public resource must contain nonempty bytes")
        if not isinstance(self.media_type, str) or not self.media_type.strip():
            raise ValueError("a public resource media type cannot be empty")
        if not isinstance(self.description, str):
            raise TypeError("a public resource description must be text")
        object.__setattr__(self, "relative_path", path.as_posix())
        object.__setattr__(self, "media_type", self.media_type.strip())
        object.__setattr__(self, "sha256", hashlib.sha256(self.data).hexdigest())


@dataclass(frozen=True)
class PublicOracleMethod:
    """One Agent-callable numerical equation oracle.

    The query coordinates are chosen by the Agent. A backend may publish
    domains and resource limits, but it must not substitute a fixed scoring
    panel for the development oracle.
    """

    name: str
    description: str
    input_schema: Mapping[str, object]
    output_schema: Mapping[str, object]
    cost_unit: str
    limits: Mapping[str, object] = field(default_factory=dict)
    coordinate_policy: str = "agent_selected"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name.isidentifier()
            or self.name.startswith("_")
        ):
            raise ValueError("public oracle method names must be public identifiers")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("public oracle method descriptions cannot be empty")
        if not isinstance(self.cost_unit, str) or not self.cost_unit.strip():
            raise ValueError("public oracle methods need a resource cost unit")
        if self.coordinate_policy != "agent_selected":
            raise ValueError(
                "public numerical-oracle coordinates must be Agent-selected"
            )
        object.__setattr__(self, "description", self.description.strip())
        object.__setattr__(self, "cost_unit", self.cost_unit.strip())
        object.__setattr__(
            self, "input_schema", _freeze_json_object(self.input_schema, "input")
        )
        object.__setattr__(
            self, "output_schema", _freeze_json_object(self.output_schema, "output")
        )
        object.__setattr__(
            self, "limits", _freeze_json_object(self.limits, "limits")
        )

    def manifest(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "coordinate_policy": self.coordinate_policy,
            "input_schema": _plain_json(self.input_schema),
            "output_schema": _plain_json(self.output_schema),
            "cost_unit": self.cost_unit,
            "limits": _plain_json(self.limits),
        }


@dataclass(frozen=True)
class FrozenGuidanceCapability:
    """Public capability advertisement for generator-distribution guidance.

    Snapshot identity, replay artifacts, stage transitions, and execution
    failures belong to :mod:`pde_agent_verifier_solver.sdk.guidance`; this
    bundle records only whether the backend advertises that protocol.
    """

    supported: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.supported, bool):
            raise TypeError("frozen guidance supported flag must be boolean")

    def manifest(self) -> dict[str, object]:
        return {
            "supported": self.supported,
            "mode": (
                "frozen_online_answer_first_v1"
                if self.supported
                else "unsupported"
            ),
            "unsupported_reason": (
                None
                if self.supported
                else "backend did not declare frozen online guidance support"
            ),
            "execution_contract": (
                "sdk.guidance.GuidanceCapability"
            ),
            "layer": "generator_distribution",
            "training": {
                "cadence": "once_at_round_boundary",
                "target": "previous_round_incumbent",
                "snapshot_policy": "train_once_then_freeze_for_next_round",
                "live_per_candidate_training": False,
            },
            "required_snapshot_bindings": [
                "base_generator_sha256",
                "target_solver_sha256",
                "reward_matcher_definition_sha256",
                "guidance_config_sha256",
                "training_seed",
                "training_budget",
                "guided_artifact.sha256",
                "snapshot_sha256",
            ],
            "next_round_development_access": {
                "kind": "online_frozen_guidance_oracle",
                "fresh_answer_first_sampling": True,
                "repeated_sampling": True,
                "fixed_sample_quota": False,
                "limited_only_by": [
                    "wall_time",
                    "concurrency",
                    "compute_capacity",
                ],
                "completed_solution_coordinates": "agent_selected",
                "guided_replay_bank": (
                    "optional_cache_or_reproducibility_artifact_not_a_quota"
                ),
            },
            "official_guided_evaluation": {
                "sampling": "controller_hidden_common_cohort",
                "same_frozen_snapshot": True,
                "candidate_cannot_resample_to_improve_score": True,
            },
            "metric_cohorts": {
                "base_average": "reported_separately",
                "guided_adversarial": "reported_separately",
            },
            "excluded_from_candidate_solve_and_query_time": [
                "guidance_training",
                "guided_case_generation",
            ],
            "candidate_promotion_mechanism": False,
        }


@dataclass(frozen=True)
class PublicProblemBundle:
    """Complete public scientific context supplied by one equation plugin.

    ``private_information`` is intentionally closed rather than extensible:
    an equation plugin cannot silently hide its operator, discretization, or
    numerical interface by adding another private category.
    """

    bundle_id: str
    equation_id: str
    description: str
    resources: tuple[PublicResource, ...]
    oracle_methods: tuple[PublicOracleMethod, ...] = ()
    development_solution_oracle_methods: tuple[PublicOracleMethod, ...] = ()
    guidance_capability: FrozenGuidanceCapability = FrozenGuidanceCapability()
    private_information: tuple[str, ...] = PUBLIC_PROBLEM_PRIVATE_INFORMATION
    bundle_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name, value in (
            ("bundle_id", self.bundle_id),
            ("equation_id", self.equation_id),
            ("description", self.description),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"public problem {name} cannot be empty")
        resources = tuple(self.resources)
        if not resources or any(
            not isinstance(resource, PublicResource) for resource in resources
        ):
            raise ValueError(
                "a public problem bundle needs at least one PublicResource"
            )
        paths = tuple(resource.relative_path for resource in resources)
        if len(paths) != len(set(paths)):
            raise ValueError("public problem resource paths must be unique")
        oracle_methods = tuple(self.oracle_methods)
        if any(
            not isinstance(method, PublicOracleMethod)
            for method in oracle_methods
        ):
            raise TypeError(
                "public problem oracle_methods must contain PublicOracleMethod values"
            )
        method_names = tuple(method.name for method in oracle_methods)
        if len(method_names) != len(set(method_names)):
            raise ValueError("public problem oracle method names must be unique")
        development_methods = tuple(self.development_solution_oracle_methods)
        if any(
            not isinstance(method, PublicOracleMethod)
            for method in development_methods
        ):
            raise TypeError(
                "development solution oracle methods must contain "
                "PublicOracleMethod values"
            )
        development_names = tuple(
            method.name for method in development_methods
        )
        if len(development_names) != len(set(development_names)):
            raise ValueError(
                "development solution oracle method names must be unique"
            )
        if set(method_names) & set(development_names):
            raise ValueError(
                "equation and completed-solution oracle methods must be disjoint"
            )
        guidance = self.guidance_capability
        if not isinstance(guidance, FrozenGuidanceCapability):
            raise TypeError(
                "guidance_capability must be a FrozenGuidanceCapability"
            )
        private = tuple(self.private_information)
        if private != PUBLIC_PROBLEM_PRIVATE_INFORMATION:
            raise ValueError(
                "only generator parameters, latent solution parameterization, "
                "and unreleased evaluation answers may be private"
            )
        object.__setattr__(self, "bundle_id", self.bundle_id.strip())
        object.__setattr__(self, "equation_id", self.equation_id.strip())
        object.__setattr__(self, "description", self.description.strip())
        object.__setattr__(self, "resources", resources)
        object.__setattr__(self, "oracle_methods", oracle_methods)
        object.__setattr__(
            self,
            "development_solution_oracle_methods",
            development_methods,
        )
        object.__setattr__(self, "private_information", private)
        digest_payload = {
            "bundle_id": self.bundle_id,
            "equation_id": self.equation_id,
            "description": self.description,
            "private_information": list(private),
            "guidance_capability": guidance.manifest(),
            "candidate_solution_access": _plain_json(
                CONTINUOUS_POINT_SOLUTION_CONTRACT
            ),
            "oracle_methods": [
                method.manifest()
                for method in sorted(oracle_methods, key=lambda item: item.name)
            ],
            "development_solution_oracle_methods": [
                method.manifest()
                for method in sorted(
                    development_methods, key=lambda item: item.name
                )
            ],
            "resources": [
                {
                    "path": resource.relative_path,
                    "sha256": resource.sha256,
                    "size_bytes": len(resource.data),
                    "media_type": resource.media_type,
                    "description": resource.description,
                }
                for resource in sorted(
                    resources, key=lambda item: item.relative_path
                )
            ],
        }
        encoded = json.dumps(
            digest_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        object.__setattr__(
            self, "bundle_sha256", hashlib.sha256(encoded).hexdigest()
        )

    def resource(self, relative_path: str) -> PublicResource:
        """Return one named resource or fail closed on an unknown path."""

        name = _public_relative_path(relative_path).as_posix()
        for resource in self.resources:
            if resource.relative_path == name:
                return resource
        raise KeyError(f"unknown public problem resource: {name}")

    def manifest(self) -> dict[str, object]:
        """Return the JSON-safe manifest staged beside the public files."""

        return {
            "schema": "pde-public-problem-bundle-v1",
            "bundle_id": self.bundle_id,
            "bundle_sha256": self.bundle_sha256,
            "equation_id": self.equation_id,
            "description": self.description,
            "private_information": list(self.private_information),
            "guidance_capability": self.guidance_capability.manifest(),
            "candidate_solution_access": _plain_json(
                CONTINUOUS_POINT_SOLUTION_CONTRACT
            ),
            "oracle_methods": [
                method.manifest()
                for method in sorted(
                    self.oracle_methods, key=lambda item: item.name
                )
            ],
            "development_solution_oracle_methods": [
                method.manifest()
                for method in sorted(
                    self.development_solution_oracle_methods,
                    key=lambda item: item.name,
                )
            ],
            "resources": [
                {
                    "path": resource.relative_path,
                    "sha256": resource.sha256,
                    "size_bytes": len(resource.data),
                    "media_type": resource.media_type,
                    "description": resource.description,
                }
                for resource in sorted(
                    self.resources, key=lambda item: item.relative_path
                )
            ],
        }

    def materialize(self, destination: str | Path) -> Path:
        """Atomically stage a fresh, read-only ``public_resources`` tree."""

        target = Path(destination).expanduser().resolve()
        if target == Path(target.anchor):
            raise ValueError("public resources need a specific destination")
        if target.exists() or target.is_symlink():
            raise FileExistsError(
                f"public resources destination must be fresh: {target}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{target.name}-staging-", dir=str(target.parent)
            )
        )
        try:
            for resource in self.resources:
                path = temporary / resource.relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as stream:
                    stream.write(resource.data)
                path.chmod(0o444)
            manifest_path = temporary / "manifest.json"
            with manifest_path.open("x", encoding="utf-8") as stream:
                json.dump(
                    self.manifest(),
                    stream,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                )
                stream.write("\n")
            manifest_path.chmod(0o444)
            os.replace(temporary, target)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return target


def _public_relative_path(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("public resource paths must be nonempty POSIX paths")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value:
        raise ValueError("public resource paths must be canonical and relative")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("public resource paths cannot escape their bundle")
    return path


def _freeze_json_object(
    value: Mapping[str, object], label: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"public oracle {label} schema must be a mapping")
    plain = _plain_json(value)
    if not isinstance(plain, dict):
        raise TypeError(f"public oracle {label} schema must be an object")
    encoded = json.dumps(
        plain, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return _freeze_json(json.loads(encoded))


def _freeze_json(value):
    if isinstance(value, dict):
        return MappingProxyType(
            {str(name): _freeze_json(item) for name, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _plain_json(value):
    if isinstance(value, Mapping):
        if any(not isinstance(name, str) for name in value):
            raise TypeError("public oracle schema keys must be text")
        return {name: _plain_json(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError("public oracle schemas must contain finite JSON values")


__all__ = [
    "CONTINUOUS_POINT_SOLUTION_CONTRACT",
    "TWO_PHASE_CONTINUOUS_SOLUTION_CONTRACT",
    "PUBLIC_PROBLEM_PRIVATE_INFORMATION",
    "FrozenGuidanceCapability",
    "PublicOracleMethod",
    "PublicProblemBundle",
    "PublicResource",
]
