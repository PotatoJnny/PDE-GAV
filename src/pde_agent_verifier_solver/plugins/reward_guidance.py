"""Frozen reward-surrogate selection over fresh answer-first proposals.

This is finite-pool importance resampling, not diffusion/flow score guidance.
The base sampler remains unchanged. Only public problem features and scalar
controller reward feedback train the guide; hidden answers never enter it.
"""

from __future__ import annotations

from copy import copy
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from math import isfinite
from random import Random
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import numpy as np

from ..sdk.models import Case, CaseRecord, GenerationKind, GenerationRequest, RoundFeedback
from ..sdk.guidance import GuidanceUnsupported
from .coefficient_flow import BackendGeneratorProgram


def _digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class RewardGuidanceConfig:
    pool_size: int = 4
    ridge: float = 1.0e-3
    strength: float = 2.0
    uniform_mix: float = 0.1
    minimum_feedback: int = 4
    maximum_standardized_feature: float = 8.0

    def __post_init__(self):
        for name in ("pool_size", "minimum_feedback"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f"{name} must be an integer at least two")
        for name in ("ridge", "strength", "maximum_standardized_feature"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not isfinite(self.uniform_mix) or not 0 < self.uniform_mix <= 1:
            raise ValueError("uniform_mix must be in (0, 1]")


@dataclass(frozen=True)
class FrozenRewardGuide:
    source_round_index: int
    target_round_index: int
    selected_artifact_id: str
    base_generator_sha256: str
    feature_id: str
    config_sha256: str
    feedback_sha256: str
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    reward_scale: float
    training_case_ids: tuple[str, ...] = ()
    method: str = "reward_surrogate_finite_pool_resampling"

    @property
    def sha256(self) -> str:
        return _digest(asdict(self))


class RewardGuidedBackend:
    """Add reproducible online guidance while preserving backend ownership.

    ``feature_extractor`` receives only a public Case. A PDE may close over
    its PUBLIC_PROBLEM numerical session to evaluate fixed public probes.
    It must not access truth, phase tags, Case IDs or private Generator state.
    ``feature_id`` versions that numerical feature definition.

    Guidance is fitted once by the existing Controller at a round boundary.
    Every call draws new base proposals, scores them without running the
    incumbent, and releases all discarded resident solutions. Retained Cases
    are returned by identity so their original perturbation anchors and
    numerical sessions remain valid. The sampler must own ``release_cases``.
    """

    supports_guidance = True

    def __init__(
        self,
        backend,
        *,
        feature_extractor: Callable[[Case], Sequence[float]],
        feature_id: str,
        base_generator_sha256: str,
        config: RewardGuidanceConfig | None = None,
    ):
        if not callable(feature_extractor):
            raise TypeError("feature_extractor must be callable")
        if not isinstance(feature_id, str) or not feature_id.strip():
            raise ValueError("feature_id is required")
        if len(base_generator_sha256) != 64 or any(c not in "0123456789abcdef" for c in base_generator_sha256):
            raise ValueError("base_generator_sha256 must be a SHA256")
        if not callable(getattr(backend, "release_cases", None)):
            raise TypeError("reward guidance needs backend resident-case release")
        self.backend = backend
        self.feature_extractor = feature_extractor
        self.feature_id = feature_id
        self.base_generator_sha256 = base_generator_sha256
        self.config = config or RewardGuidanceConfig()
        self.last_generation_report: Mapping[str, object] = MappingProxyType({})

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def _features(self, case: Case) -> np.ndarray:
        values = np.asarray(tuple(self.feature_extractor(case)), dtype=np.float64)
        if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)):
            raise ValueError("guidance features must be a nonempty finite vector")
        return values

    def fit_guidance(self, base_state, previous_guidance, feedback, settings, progress=None):
        del base_state, previous_guidance, settings
        if not isinstance(feedback, RoundFeedback):
            raise TypeError("guidance requires controller RoundFeedback")
        if len(feedback.cases) < self.config.minimum_feedback:
            raise GuidanceUnsupported("insufficient feedback to fit reward guidance")
        training_case_ids = tuple(row.case.case_id for row in feedback.cases)
        if len(set(training_case_ids)) != len(training_case_ids):
            raise ValueError("guidance training Cases must have unique identities")
        errors = np.asarray([row.primary_error for row in feedback.cases], dtype=np.float64)
        if not np.all(np.isfinite(errors)) or np.any(errors < 0):
            raise ValueError("guidance errors must be finite and nonnegative")
        features = np.stack([self._features(row.case) for row in feedback.cases])
        mean = features.mean(axis=0)
        scale = features.std(axis=0)
        scale = np.where(scale > 1.0e-12, scale, 1.0)
        design = np.clip((features - mean) / scale, -self.config.maximum_standardized_feature, self.config.maximum_standardized_feature)
        rewards = np.log1p(errors)
        intercept = float(rewards.mean())
        reward_scale = float(rewards.std())
        if reward_scale <= 1.0e-12 or not np.any(np.std(design, axis=0) > 1.0e-12):
            raise GuidanceUnsupported("feedback has no usable public-feature/reward variation")
        coefficients = np.linalg.solve(design.T @ design + self.config.ridge * np.eye(design.shape[1]), design.T @ (rewards - intercept))
        if not np.all(np.isfinite(coefficients)) or np.linalg.norm(coefficients) <= 1.0e-12:
            raise GuidanceUnsupported("public features provide no fitted reward signal")
        snapshot = FrozenRewardGuide(
            source_round_index=feedback.round_index,
            target_round_index=feedback.round_index + 1,
            selected_artifact_id=feedback.selected_artifact.artifact_id,
            base_generator_sha256=self.base_generator_sha256,
            feature_id=self.feature_id,
            config_sha256=_digest(asdict(self.config)),
            feedback_sha256=_digest({"features": features.tolist(), "errors": errors.tolist(), "case_ids": [row.case.case_id for row in feedback.cases]}),
            mean=tuple(mean), scale=tuple(scale), coefficients=tuple(coefficients),
            intercept=intercept, reward_scale=reward_scale,
            training_case_ids=training_case_ids,
        )
        if progress:
            progress("Frozen public-feature reward surrogate fitted", 1.0)
        return snapshot

    def _require_guide(self, guide, request):
        if not isinstance(guide, FrozenRewardGuide):
            raise GuidanceUnsupported("guided sampling needs a frozen reward surrogate")
        if (guide.base_generator_sha256 != self.base_generator_sha256 or guide.feature_id != self.feature_id or guide.config_sha256 != _digest(asdict(self.config))):
            raise ValueError("frozen guide backend, feature definition or config changed")
        # Terminal cohorts may deliberately replay an earlier round's guide.
        if request.round_index < guide.target_round_index:
            raise ValueError("guidance cannot be used before its target round")
        claimed = request.settings.get("fixed_generator_sha256")
        if claimed is not None and claimed != self.base_generator_sha256:
            raise ValueError("request base Generator identity differs from frozen guide")

    def generate(self, base_state, guidance_state, request, progress=None):
        if not isinstance(request, GenerationRequest):
            raise TypeError("guidance sampling needs GenerationRequest")
        if request.kind is not GenerationKind.GUIDED:
            return self.backend.generate(base_state, None, request, progress)
        self._require_guide(guidance_state, request)
        guide = guidance_state
        proposal_seed = int(_digest({"method": guide.method, "guide": guide.sha256, "seed": request.seed, "round": request.round_index, "purpose": request.purpose.value})[:15], 16)
        proposal_request = replace(request, kind=GenerationKind.BASE, count=request.count * self.config.pool_size, seed=proposal_seed)
        proposals = tuple(self.backend.generate(base_state, None, proposal_request, progress))
        release_ids = tuple(row.case.case_id for row in proposals if isinstance(row, CaseRecord))
        try:
            if len(proposals) != proposal_request.count or len(set(release_ids)) != len(proposals):
                raise ValueError("guided proposal pool is incomplete or has duplicate Case IDs")
            features = np.stack([self._features(row.case) for row in proposals])
            if features.shape[1] != len(guide.mean):
                raise ValueError("guidance feature dimension changed")
            design = np.clip((features - np.asarray(guide.mean)) / np.asarray(guide.scale), -self.config.maximum_standardized_feature, self.config.maximum_standardized_feature)
            scores = design @ np.asarray(guide.coefficients) / guide.reward_scale
            rng = Random(proposal_seed)
            selected = []
            effective_sample_sizes = []
            for offset in range(0, len(proposals), self.config.pool_size):
                group = scores[offset : offset + self.config.pool_size]
                logits = np.clip(self.config.strength * (group - group.max()), -700, 0)
                probabilities = np.exp(logits)
                probabilities /= probabilities.sum()
                probabilities = (1 - self.config.uniform_mix) * probabilities + self.config.uniform_mix / self.config.pool_size
                effective_sample_sizes.append(float(1.0 / np.sum(probabilities ** 2)))
                local = rng.choices(range(self.config.pool_size), weights=probabilities, k=1)[0]
                selected.append(offset + local)
            selected_set = set(selected)
            discard = tuple(row.case.case_id for index, row in enumerate(proposals) if index not in selected_set)
            self.backend.release_cases(discard)
        except BaseException:
            self.backend.release_cases(release_ids)
            raise
        self.last_generation_report = MappingProxyType({
            "method": guide.method,
            "snapshot_sha256": guide.sha256,
            "proposal_count": len(proposals),
            "returned_count": len(selected),
            "discarded_count": len(discard),
            "physical_guidance_qualification": "required",
            "is_diffusion_or_flow_guidance": False,
            "mean_pool_weight_ess": float(np.mean(effective_sample_sizes)),
            "minimum_pool_weight_ess": float(np.min(effective_sample_sizes)),
            "pool_size": self.config.pool_size,
        })
        return tuple(proposals[index] for index in selected)


class RewardGuidedGenerator(BackendGeneratorProgram):
    """Compose the wrapper with any existing backend-delegating PDE Program."""

    def __init__(self, program: BackendGeneratorProgram, **guidance_options):
        if not isinstance(program, BackendGeneratorProgram):
            raise TypeError("RewardGuidedGenerator needs a backend-delegating Generator")
        super().__init__(RewardGuidedBackend(program._backend(), **guidance_options))
        # Keep custom frozen-runtime load/sample guards (for example bound
        # provenance injection) while replacing only the numerical backend.
        # The caller's original Program remains unchanged.
        self.program = copy(program)
        self.program.backend = self.backend
        self.info = program.info
        self.capabilities = replace(program.capabilities, guidance=True)

    def setup_fields(self):
        return tuple(replace(field, default=True) if field.name == "guidance_enabled" else field for field in self.program.setup_fields())

    def train(self, dataset, settings, progress=None):
        return self.program.train(dataset, settings, progress)

    def load(self, checkpoint, settings, progress=None):
        return self.program.load(checkpoint, settings, progress)

    def update_guidance(self, state, feedback, settings, progress=None):
        return self.program.update_guidance(state, feedback, settings, progress)

    def sample(self, state, request, progress=None):
        return self.program.sample(state, request, progress)

    def open_case_session(self, state, case, access):
        return self.program.open_case_session(state, case, access)

    def case_session_methods(self, access):
        return self.program.case_session_methods(access)

    def release(self, state, case_ids):
        return self.program.release(state, case_ids)

    def preview(self, record):
        return self.program.preview(record)


__all__ = ["FrozenRewardGuide", "RewardGuidanceConfig", "RewardGuidedBackend", "RewardGuidedGenerator"]
