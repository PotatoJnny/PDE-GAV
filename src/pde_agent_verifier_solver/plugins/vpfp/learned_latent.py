"""Conditional probabilistic coefficient model for complete VPFP answers.

Fit a linear conditional mean and a low-rank-plus-diagonal Gaussian residual
to complete continuous-answer coefficients. Sampling never calls a PDE solver.
This is a transparent statistical baseline, not a qualified production model.
Its checkpoint explicitly preserves the numerical provenance and held-out
split; small test fixtures must never be presented as physical validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


SCHEMA = "vpfp_conditional_low_rank_answer_distribution_v1"


def _array(value, name, ndim):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != ndim or not result.size or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a nonempty finite {ndim}D array")
    return result


def _file_sha256(path):
    digest = sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class VPFPConditionalLatentDistribution:
    condition_center: np.ndarray
    condition_scale: np.ndarray
    latent_center: np.ndarray
    latent_scale: np.ndarray
    regression: np.ndarray
    factor: np.ndarray
    diagonal_std: np.ndarray
    metadata: Mapping[str, object]

    def __post_init__(self):
        dimension = len(_array(self.latent_center, "latent_center", 1))
        for name in ("latent_scale", "diagonal_std"):
            values = _array(getattr(self, name), name, 1)
            if values.shape != (dimension,) or np.any(values < 0):
                raise ValueError(f"invalid learned {name}")
        for name in ("condition_center", "condition_scale"):
            if _array(getattr(self, name), name, 1).shape != (2,):
                raise ValueError("VPFP condition dimension must be two")
        if np.any(self.condition_scale <= 0) or np.any(self.latent_scale <= 0):
            raise ValueError("learned normalization scales must be positive")
        if _array(self.regression, "regression", 2).shape != (3, dimension):
            raise ValueError("invalid conditional regression dimensions")
        if _array(self.factor, "factor", 2).shape[0] != dimension:
            raise ValueError("invalid learned covariance dimensions")
        if self.metadata.get("schema") != SCHEMA:
            raise ValueError("unexpected VPFP learned checkpoint schema")
        if self.metadata.get("latent_dimension") != dimension or self.metadata.get("rank") != self.factor.shape[1]:
            raise ValueError("learned metadata and array dimensions differ")
        bounds = np.asarray((self.metadata.get("collision_nu_range"),
                             self.metadata.get("temperature_range")), dtype=float)
        if bounds.shape != (2, 2) or not np.isfinite(bounds).all():
            raise ValueError("invalid learned checkpoint condition ranges")
        if (not np.array_equal(bounds.mean(axis=1), self.condition_center)
                or not np.array_equal(np.diff(bounds, axis=1).ravel() / 2., self.condition_scale)):
            raise ValueError("learned metadata and condition transform differ")

    @classmethod
    def fit(
        cls, latents, conditions, *, sample_ids: Sequence[int],
        base_decoder_sha256: str, collision_nu_range, temperature_range,
        provenance: Mapping[str, object], rank: int = 16,
        validation_fraction: float = 0.2, split_seed: int = 173,
        ridge: float = 1.e-4, variance_floor: float = 1.e-4,
    ):
        values = _array(latents, "latents", 2)
        cond = _array(conditions, "conditions", 2)
        if cond.shape != (len(values), 2) or len(values) < 12:
            raise ValueError("VPFP fitting requires >=12 complete answers with two conditions")
        if any(isinstance(value, (bool, np.bool_)) or int(value) != value
               for value in sample_ids):
            raise ValueError("sample identities must be integers")
        ids = tuple(int(value) for value in sample_ids)
        if len(ids) != len(values) or len(set(ids)) != len(ids):
            raise ValueError("VPFP corpus needs distinct sample identities")
        if len(base_decoder_sha256) != 64 or any(c not in "0123456789abcdef" for c in base_decoder_sha256):
            raise ValueError("VPFP corpus must bind the base decoder SHA256")
        if not 0 < validation_fraction < 0.5 or ridge <= 0 or variance_floor <= 0:
            raise ValueError("invalid learned fitting regularization/split")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ValueError("rank must be positive")
        origin = provenance.get("data_origin")
        if origin not in {"numerical_simulation", "experimental", "synthetic_test"}:
            raise ValueError("corpus provenance must declare its actual data_origin")
        if not provenance.get("source_artifact_sha256"):
            raise ValueError("corpus provenance must identify the source artifact")
        bounds = np.asarray((collision_nu_range, temperature_range), dtype=float)
        if bounds.shape != (2, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 1] <= bounds[:, 0]):
            raise ValueError("invalid learned condition ranges")
        if np.any(cond < bounds[:, 0]) or np.any(cond > bounds[:, 1]):
            raise ValueError("training conditions leave the declared regime")
        order = np.random.default_rng(split_seed).permutation(len(values))
        nvalidation = max(2, int(np.ceil(len(values) * validation_fraction)))
        validation, training = order[:nvalidation], order[nvalidation:]
        center = values[training].mean(axis=0)
        deviation = values[training].std(axis=0)
        active = deviation > 1.e-10
        scale = np.where(active, deviation, 1.)
        condition_center = bounds.mean(axis=1)
        condition_scale = 0.5 * (bounds[:, 1] - bounds[:, 0])
        design = np.column_stack((np.ones(len(values)), (cond - condition_center) / condition_scale))
        targets = (values - center) / scale
        train_design = design[training]
        penalty = np.diag((0., ridge, ridge))
        regression = np.linalg.solve(train_design.T @ train_design + penalty,
                                     train_design.T @ targets[training])
        residual = targets[training] - train_design @ regression
        # SVD is performed on the offline corpus; sampling stores only D x rank.
        _, singular, right = np.linalg.svd(residual, full_matrices=False)
        retained = min(rank, len(training) - 3, values.shape[1])
        factor = right[:retained].T * (singular[:retained] / np.sqrt(len(training)))
        represented = np.sum(factor * factor, axis=1)
        diagonal = np.maximum(np.mean(residual * residual, axis=0) - represented,
                              variance_floor)
        diagonal = np.where(active, diagonal, 0.)
        factor[~active] = 0.
        prediction = design[validation] @ regression
        held_out_mse = float(np.mean((targets[validation] - prediction) ** 2))
        meta = {
            "schema": SCHEMA, "base_decoder_sha256": base_decoder_sha256,
            "collision_nu_range": list(map(float, collision_nu_range)),
            "temperature_range": list(map(float, temperature_range)),
            "latent_dimension": values.shape[1], "rank": retained,
            "train_sample_ids": [ids[index] for index in training],
            "validation_sample_ids": [ids[index] for index in validation],
            "provenance": dict(provenance), "split_seed": split_seed,
            "validation_normalized_conditional_mean_mse": held_out_mse,
            "qualification": "unqualified_research_checkpoint",
            "learned_generation": True, "pde_solves_per_sample": 0,
            "training_data_origin": origin,
            "experimental_data_validation": False,
            "answer_first": {
                "primary": "sample_complete_continuous_answer_coefficients",
                "source": "derive_exact_analytic_residual_after_sampling",
                "initial_condition": "evaluate_sampled_answer_at_initial_time",
            },
        }
        return cls(condition_center, condition_scale, center, scale,
                   regression, factor, np.sqrt(diagonal), meta)

    def sample(self, seeds: Sequence[int]):
        if not len(seeds):
            raise ValueError("VPFP learned sampling requires seeds")
        latents, conditions = [], []
        for seed in seeds:
            if isinstance(seed, bool) or int(seed) != seed or int(seed) < 0:
                raise ValueError("sample seeds must be nonnegative integers")
            random = np.random.default_rng(int(seed))
            normalized = random.uniform(-1., 1., size=2)
            condition = self.condition_center + self.condition_scale * normalized
            mean = np.r_[1., normalized] @ self.regression
            fluctuation = (self.factor @ random.normal(size=self.factor.shape[1])
                           + self.diagonal_std * random.normal(size=len(self.latent_center)))
            latent = self.latent_center + self.latent_scale * (mean + fluctuation)
            if not np.isfinite(latent).all():
                raise FloatingPointError("learned VPFP draw is nonfinite")
            latents.append(latent)
            conditions.append(condition)
        return np.asarray(latents, dtype=np.float32), np.asarray(conditions, dtype=np.float32)

    def save(self, path: str | Path):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents silently replacing a qualified artifact.
        with destination.open("xb") as target:
            np.savez_compressed(target,
                condition_center=self.condition_center, condition_scale=self.condition_scale,
                latent_center=self.latent_center, latent_scale=self.latent_scale,
                regression=self.regression, factor=self.factor, diagonal_std=self.diagonal_std,
                metadata=np.asarray(json.dumps(dict(self.metadata), sort_keys=True)),
            )

    @classmethod
    def load(cls, path: str | Path, *, expected_sha256: str):
        source = Path(path)
        if source.is_symlink() or not source.is_file():
            raise FileNotFoundError(source)
        with np.load(source, allow_pickle=False) as stored:
            return cls(**{name: np.array(stored[name], copy=True) for name in (
                "condition_center", "condition_scale", "latent_center", "latent_scale",
                "regression", "factor", "diagonal_std",
            )}, metadata=json.loads(str(stored["metadata"])))


class VPFPLearnedBatchGenerator:
    """Drop-in private runtime sampler with zero online PDE calls."""

    def __init__(self, model: VPFPConditionalLatentDistribution, *, checkpoint_sha256: str):
        if (len(checkpoint_sha256) != 64
                or any(c not in "0123456789abcdef" for c in checkpoint_sha256)):
            raise ValueError("learned sampler needs a checkpoint SHA256")
        self.model = model
        self.identity = {**dict(model.metadata), "checkpoint_sha256": checkpoint_sha256,
                         "kind": "learned_conditional_low_rank_gaussian"}

    def generate(self, seeds):
        latents, conditions = self.model.sample(seeds)
        return {"schema": "vpfp_learned_answer_batch_v1", "latents": latents,
                "equation_conditions": conditions,
                "equation_seeds": np.asarray(seeds, dtype=np.int64),
                "diagnostics": {"pde_solves": 0, "learned_generation": True}}
