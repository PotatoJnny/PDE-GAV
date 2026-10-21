"""Held-out measurement of finite-pool guidance against a frozen incumbent.

This uses dedicated fresh roots, never the feedback used to fit the guide.
It is Controller-side scientific tooling, not an Agent capability or a new
campaign Controller. A real physical qualification requires a real scoring
callback plus independent physical checks; synthetic unit tests do not count.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from math import isfinite
from typing import Callable, Mapping, Sequence

import numpy as np

from ..sdk.models import CaseRecord, CaseVisibility, GenerationKind, GenerationPurpose, GenerationRequest
from ..sdk.sampling import RecoverableSamplingRejection
from .reward_guidance import FrozenRewardGuide, RewardGuidedBackend


@dataclass(frozen=True)
class QualificationScore:
    primary_error: float
    physical_valid: bool | None = None

    def __post_init__(self):
        if isinstance(self.primary_error, bool) or not isfinite(self.primary_error) or self.primary_error < 0:
            raise ValueError("qualification error must be finite and nonnegative")
        if self.physical_valid is not None and not isinstance(self.physical_valid, bool):
            raise TypeError("physical_valid must be boolean or unknown")


def _seed(value: int, source: str, snapshot: str) -> int:
    return int(sha256(f"guidance-qualification-v1:{snapshot}:{value}:{source}".encode()).hexdigest()[:15], 16)


def _cvar(values: np.ndarray, quantile: float) -> float:
    """Mean of the largest (1-q) fraction, including a fractional boundary."""
    ordered = np.sort(values)[::-1]
    mass = (1.0 - quantile) * len(ordered)
    whole = int(mass)
    numerator = float(ordered[:whole].sum())
    if whole < len(ordered):
        numerator += (mass - whole) * float(ordered[whole])
    return numerator / mass


def _summary(values, quantile):
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0, "mean": None, "cvar": None, "cvar_quantile": quantile}
    return {"count": len(array), "mean": float(array.mean()), "cvar": _cvar(array, quantile), "cvar_quantile": quantile}


def qualify_reward_guidance(
    backend: RewardGuidedBackend,
    *,
    base_state: object,
    guide: FrozenRewardGuide,
    request: GenerationRequest,
    seeds: Sequence[int],
    incumbent_artifact_id: str,
    matcher_sha256: str,
    score: Callable[[CaseRecord], QualificationScore],
    minimum_independent_seeds: int = 32,
    minimum_relative_mean_lift: float = 0.0,
    cvar_quantile: float = 0.9,
    confidence_level: float = 0.95,
    bootstrap_replicates: int = 1000,
    bootstrap_seed: int = 0,
    physical_counters: Callable[[], Mapping[str, int]] | None = None,
) -> dict[str, object]:
    """Compare independent BASE/GUIDED roots and bootstrap seed clusters.

    ``request.count`` is the number of samples per independent seed. The
    callback must use the same frozen incumbent and macro matcher throughout.
    It may access private truth inside this Controller-side function. A score
    with ``physical_valid=None`` permits measurements but prevents admission.

    Optional counters report cumulative ``physical_proposals`` and
    ``physical_rejections`` from the actual numerical runtime. Without them,
    only request-level recoverable rejections are measured. Finite-pool ESS
    describes normalized proposal selection weights, not independent draws
    or an exact importance-sampling estimate.
    """
    if not isinstance(backend, RewardGuidedBackend) or not isinstance(guide, FrozenRewardGuide):
        raise TypeError("qualification needs a reward backend and frozen guide")
    if incumbent_artifact_id != guide.selected_artifact_id:
        raise ValueError("qualification incumbent differs from the frozen guide target")
    if len(matcher_sha256) != 64 or any(c not in "0123456789abcdef" for c in matcher_sha256):
        raise ValueError("matcher_sha256 must be a SHA256")
    if not guide.training_case_ids:
        raise ValueError("training Case identities are required to audit held-out freshness")
    seeds = tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("qualification seeds must be nonempty, unique nonnegative integers")
    for value, label, minimum in ((minimum_independent_seeds, "minimum seeds", 2), (bootstrap_replicates, "bootstrap replicates", 100)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{label} must be at least {minimum}")
    if not 0 < cvar_quantile < 1 or not 0 < confidence_level < 1:
        raise ValueError("CVaR quantile and confidence level must be in (0,1)")
    if not isfinite(minimum_relative_mean_lift) or minimum_relative_mean_lift < 0:
        raise ValueError("minimum relative lift must be finite and nonnegative")
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int) or bootstrap_seed < 0:
        raise ValueError("bootstrap seed must be a nonnegative integer")
    training_ids = set(guide.training_case_ids)
    observed_ids: set[str] = set()
    errors = {"base": [], "guided": []}
    batches = {"base": {}, "guided": {}}
    rejected = {"base": [], "guided": []}
    physical = {"base": [], "guided": []}
    pool_ess = []
    counters_before = dict(physical_counters()) if physical_counters else None
    for seed in seeds:
        for source, kind in (("base", GenerationKind.BASE), ("guided", GenerationKind.GUIDED)):
            draw = replace(request, kind=kind, visibility=CaseVisibility.HIDDEN, purpose=GenerationPurpose.GUIDANCE_FEEDBACK, anchors=(), seed=_seed(seed, source, guide.sha256))
            records = ()
            try:
                records = tuple(backend.generate(base_state, guide, draw))
                if len(records) != draw.count:
                    raise ValueError("qualification received an incomplete sample batch")
                identifiers = tuple(record.case.case_id for record in records)
                if len(set(identifiers)) != len(identifiers) or (set(identifiers) & (training_ids | observed_ids)):
                    raise ValueError("qualification roots overlap training or previous evaluation roots")
                observed_ids.update(identifiers)
                results = tuple(score(record) for record in records)
                if any(not isinstance(result, QualificationScore) for result in results):
                    raise TypeError("score callback must return QualificationScore")
                values = [float(result.primary_error) for result in results]
                errors[source].extend(values)
                batches[source][seed] = values
                physical[source].extend(result.physical_valid for result in results)
                if source == "guided":
                    pool_ess.append(dict(backend.last_generation_report))
            except RecoverableSamplingRejection as error:
                rejected[source].append({"seed": seed, "reason": str(error), "audit": error.audit})
            finally:
                if records:
                    backend.release_cases(tuple(record.case.case_id for record in records))
    matched = [seed for seed in seeds if seed in batches["base"] and seed in batches["guided"]]
    intervals = None
    if len(matched) >= 2:
        rng = np.random.default_rng(bootstrap_seed)
        mean_lifts, cvar_lifts = [], []
        for _ in range(bootstrap_replicates):
            indices = rng.integers(0, len(matched), len(matched))
            base_values = np.concatenate([batches["base"][matched[index]] for index in indices])
            guided_values = np.concatenate([batches["guided"][matched[index]] for index in indices])
            mean_lifts.append(float(guided_values.mean() - base_values.mean()))
            cvar_lifts.append(_cvar(guided_values, cvar_quantile) - _cvar(base_values, cvar_quantile))
        tails = ((1 - confidence_level) / 2, (1 + confidence_level) / 2)
        intervals = {"confidence_level": confidence_level, "method": "percentile_bootstrap_by_controller_seed", "replicates": bootstrap_replicates, "independent_seed_pairs": len(matched), "mean_error_difference": np.quantile(mean_lifts, tails).tolist(), "cvar_error_difference": np.quantile(cvar_lifts, tails).tolist()}
    summaries = {source: _summary(values, cvar_quantile) for source, values in errors.items()}
    physical_report = {}
    for source, results in physical.items():
        physical_report[source] = {"valid": results.count(True), "invalid": results.count(False), "unknown": results.count(None)}
    numerical_rejection_rate = None
    if physical_counters:
        after = dict(physical_counters())
        keys = ("physical_proposals", "physical_rejections")
        if any(key not in counters_before or key not in after or isinstance(after[key], bool) or not isinstance(after[key], int) or not isinstance(counters_before[key], int) or after[key] < counters_before[key] for key in keys):
            raise ValueError("physical counters must be monotone integer proposals/rejections")
        proposals = after[keys[0]] - counters_before[keys[0]]
        rejections = after[keys[1]] - counters_before[keys[1]]
        if rejections > proposals:
            raise ValueError("physical rejections exceed proposals")
        numerical_rejection_rate = {"proposals": proposals, "rejections": rejections, "rate": rejections / proposals if proposals else None}
    complete_physics = all(results and all(result is True for result in results) for results in physical.values())
    sufficient = len(matched) >= minimum_independent_seeds and complete_physics and intervals is not None
    status = "insufficient_evidence"
    if sufficient:
        threshold = minimum_relative_mean_lift * summaries["base"]["mean"]
        status = "passed_clean_hardness" if intervals["mean_error_difference"][0] > threshold else "failed_clean_hardness"
    return {
        "schema": "reward_guidance_heldout_qualification_v1",
        "status": status,
        "method": guide.method,
        "snapshot_sha256": guide.sha256,
        "base_generator_sha256": guide.base_generator_sha256,
        "incumbent_artifact_id": incumbent_artifact_id,
        "matcher_sha256": matcher_sha256,
        "training_feedback_used_for_evaluation": False,
        "freshness": {"unique_evaluation_cases": len(observed_ids), "training_overlap": 0, "base_guided_overlap": 0, "independent_seeds": len(seeds), "completed_seed_pairs": len(matched)},
        "errors": summaries,
        "uncertainty": intervals,
        "physical_validation": physical_report,
        "request_rejections": {source: {"count": len(rows), "rate": len(rows) / len(seeds), "audit": rows} for source, rows in rejected.items()},
        "physical_sampling_rejection_rate": numerical_rejection_rate,
        "pool_weight_ess": {"meaning": "finite-proposal selection weight ESS, not independent sample count", "mean": float(np.mean([row["mean_pool_weight_ess"] for row in pool_ess])) if pool_ess else None, "minimum": min((row["minimum_pool_weight_ess"] for row in pool_ess), default=None), "pool_size": backend.config.pool_size},
        "minimum_independent_seeds": minimum_independent_seeds,
        "minimum_relative_mean_lift": minimum_relative_mean_lift,
        "perturbation_guardrail": "not_evaluated",
        "full_production_guidance_admission": False,
    }


__all__ = ["QualificationScore", "qualify_reward_guidance"]
