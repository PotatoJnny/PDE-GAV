"""Full-feature, trajectory-blocked VPFP distribution diagnostics.

This module does not certify a generator. A non-rejected two-sample null is not
equivalence, and synthetic/empirical-bank controls are not physical validation.
All numerical work is NumPy-only; large audits belong on compute nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "vpfp_full_feature_diffusion_audit_v1"
_ROLES = ("train", "validation", "calibration", "reference_a", "reference_b", "generated")
_STATISTICS = ("max_mean_deviation", "max_variance_deviation", "multiband_mmd_max", "sliced_wasserstein_max")


class AuditInputError(ValueError):
    """An input boundary failed; no statistical conclusion may be issued."""


@dataclass(frozen=True)
class AuditFeature:
    name: str
    group: str
    unit: str
    scale_floor: float = 1.0e-8


@dataclass(frozen=True)
class AuditSample:
    features: Any
    conditions: Any
    root_ids: Sequence[str]
    answer_sha256: Sequence[str] | None = None
    # Statistical-dependence ancestors, NOT shared decoder/config/checkpoint IDs.
    lineage_ids: Sequence[Sequence[str]] | None = None


@dataclass(frozen=True)
class DiffusionAuditConfig:
    permutations: int = 99
    bootstrap_repeats: int = 99
    projections: int = 32
    alpha: float = 0.05
    seed: int = 20260904
    minimum_per_stratum: int = 2
    bandwidth_multipliers: tuple[float, ...] = (0.5, 1.0, 2.0)

    def validate(self) -> None:
        for name in ("permutations", "bootstrap_repeats", "projections"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise AuditInputError(f"{name} must be a positive integer")
        if isinstance(self.minimum_per_stratum, bool) or int(self.minimum_per_stratum) != self.minimum_per_stratum or self.minimum_per_stratum < 2:
            raise AuditInputError("minimum_per_stratum must be at least two")
        if isinstance(self.seed, bool) or int(self.seed) != self.seed or self.seed < 0:
            raise AuditInputError("seed must be a nonnegative integer")
        if not np.isfinite(self.alpha) or not 0.0 < self.alpha < 1.0:
            raise AuditInputError("alpha must lie strictly between zero and one")
        if not self.bandwidth_multipliers or any(
            not np.isfinite(x) or x <= 0 for x in self.bandwidth_multipliers
        ):
            raise AuditInputError("kernel bandwidth multipliers must be positive")


def _feature_schema(features: Sequence[AuditFeature]) -> tuple[AuditFeature, ...]:
    result = tuple(features)
    if not result or len({f.name for f in result}) != len(result):
        raise AuditInputError("feature names must be nonempty and unique")
    for feature in result:
        if any(not isinstance(v, str) or not v.strip() for v in (feature.name, feature.group, feature.unit)):
            raise AuditInputError("each feature needs a name, group and unit")
        if not np.isfinite(feature.scale_floor) or feature.scale_floor <= 0:
            raise AuditInputError("feature scale floors must be finite and positive in that feature's units")
    return result


def _edges(nu_edges, temperature_edges) -> tuple[np.ndarray, np.ndarray]:
    result = []
    for name, values, count in (("nu", nu_edges, 5), ("temperature", temperature_edges, 3)):
        arr = np.asarray(values, dtype=np.float64)
        if arr.shape != (count,) or not np.isfinite(arr).all() or not np.all(np.diff(arr) > 0):
            raise AuditInputError(f"{name} edges need {count} finite increasing values")
        if arr[0] < 0 or (name == "temperature" and arr[0] <= 0):
            raise AuditInputError("nu must be nonnegative and temperature positive")
        result.append(arr)
    return result[0], result[1]


def _strata(conditions, edges) -> np.ndarray:
    raw = np.asarray(conditions)
    if raw.dtype.kind not in "fiu":
        raise AuditInputError("conditions must contain real numeric values")
    cond = np.asarray(raw, dtype=np.float64)
    if cond.ndim != 2 or cond.shape[1] != 2 or not np.isfinite(cond).all():
        raise AuditInputError("conditions must be finite [N,2] nu/temperature rows")
    bins = []
    for axis, bounds in enumerate(edges):
        # Accept ONLY the exact encoding of a physical endpoint in the input
        # floating dtype, not an epsilon band. E.g. float32(1.2) is just above
        # float64(1.2), while its next representable float must still fail.
        boundary_dtype = raw.dtype if raw.dtype.kind == "f" else np.dtype("float64")
        encoded = np.asarray(bounds[[0, -1]], dtype=boundary_dtype).astype(np.float64)
        values = cond[:, axis]
        below = (values < bounds[0]) & (values != encoded[0])
        above = (values > bounds[-1]) & (values != encoded[1])
        if np.any(below) or np.any(above):
            raise AuditInputError(f"condition axis {axis} is outside the frozen range")
        # Restore only encoded global endpoints before half-open bin lookup.
        values = np.where(values == encoded[0], bounds[0], values)
        values = np.where(values == encoded[1], bounds[-1], values)
        bins.append(np.minimum(np.searchsorted(bounds, values, side="right") - 1, len(bounds) - 2))
    return 2 * bins[0] + bins[1]


def _sample(sample: AuditSample, width: int, edges, role: str, minimum: int):
    matrix = np.asarray(sample.features, dtype=np.float64)
    cond = np.asarray(sample.conditions, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != width or not np.isfinite(matrix).all():
        raise AuditInputError(f"{role}: features must be finite [N,{width}] rows")
    if cond.shape != (len(matrix), 2):
        raise AuditInputError(f"{role}: conditions and features are misaligned")
    # Preserve the input dtype until endpoint validation has occurred.
    strata = _strata(sample.conditions, edges)
    counts = np.bincount(strata, minlength=8)
    if np.any(counts < minimum):
        raise AuditInputError(f"{role}: missing/undersampled joint stratum: {counts.tolist()}")
    if len(set(counts.tolist())) != 1:
        raise AuditInputError(f"{role}: frozen design requires equal counts in all eight joint strata")
    if len(sample.root_ids) != len(matrix) or any(not isinstance(v, str) or not v.strip() for v in sample.root_ids):
        raise AuditInputError(f"{role}: one nonempty root ID is required per complete trajectory")
    if sample.answer_sha256 is not None:
        if len(sample.answer_sha256) != len(matrix) or any(
            not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v)
            for v in sample.answer_sha256
        ):
            raise AuditInputError(f"{role}: answer hashes must be one lowercase SHA256 per trajectory")
    if sample.lineage_ids is not None:
        if len(sample.lineage_ids) != len(matrix) or any(
            isinstance(row, str) or any(not isinstance(v, str) or not v.strip() for v in row)
            for row in sample.lineage_ids
        ):
            raise AuditInputError(f"{role}: malformed statistical lineage IDs")
    return matrix, cond, strata


def _lineage_guard(samples: Mapping[str, AuditSample]) -> dict[str, Any]:
    identities: dict[str, tuple[str, int]] = {}
    hashes: dict[str, tuple[str, int]] = {}
    for role, sample in samples.items():
        for index, root in enumerate(sample.root_ids):
            ancestors = () if sample.lineage_ids is None else sample.lineage_ids[index]
            for identity in {root, *ancestors}:
                if identity in identities:
                    raise AuditInputError(f"duplicate/cross-split lineage {identity!r}: {identities[identity]} and {(role, index)}")
                identities[identity] = (role, index)
            if sample.answer_sha256 is not None:
                digest = sample.answer_sha256[index]
                if digest in hashes:
                    raise AuditInputError(f"duplicate/cross-split answer hash: {hashes[digest]} and {(role, index)}")
                hashes[digest] = (role, index)
    return {
        "root_and_supplied_ancestor_overlap": False,
        "answer_hashes_supplied_for_all_splits": all(s.answer_sha256 is not None for s in samples.values()),
        "identity_claims_verified_against_external_artifacts": False,
        "statistical_unit": "one_complete_trajectory_per_unique_root",
        "feature_equality_is_not_used_as_an_answer_identity_test": True,
    }


def _standardizer(calibration: np.ndarray, features):
    center = calibration.mean(axis=0)
    std = calibration.std(axis=0)
    floors = np.asarray([f.scale_floor for f in features])
    scale = np.maximum(std, floors)
    if not np.isfinite(center).all() or not np.isfinite(scale).all():
        raise AuditInputError("finite inputs overflowed calibration moments")
    return center, scale, std


def _standardize(matrix, center, scale):
    with np.errstate(over="ignore", invalid="ignore"):
        result = (matrix - center) / scale
    if not np.isfinite(result).all():
        raise AuditInputError("feature scaling overflowed; no clipping is allowed")
    return result


def _squared_distances(values):
    squared = np.sum(values * values, axis=1)
    result = np.maximum(squared[:, None] + squared[None, :] - 2.0 * values @ values.T, 0.0) / values.shape[1]
    if not np.isfinite(result).all():
        raise AuditInputError("full-feature distances overflowed")
    return result


def _bandwidth(calibration_z, calibration_strata):
    distances = _squared_distances(calibration_z)
    mask = (calibration_strata[:, None] == calibration_strata[None, :]) & (distances > 1.0e-20)
    return float(np.sqrt(np.median(distances[mask]))) if np.any(mask) else 1.0


class _PairEngine:
    def __init__(self, left, right, left_bins, right_bins, bandwidth, cfg, seed):
        self.values = np.concatenate((left, right))
        bins = np.concatenate((left_bins, right_bins))
        self.bin_indices = [np.flatnonzero(bins == index) for index in range(8)]
        self.left = [np.flatnonzero(left_bins == index) for index in range(8)]
        self.right = [len(left) + np.flatnonzero(right_bins == index) for index in range(8)]
        self.bandwidths = tuple(bandwidth * value for value in cfg.bandwidth_multipliers)
        distances = _squared_distances(self.values)
        self.kernels = [np.exp(-distances / (2.0 * value * value)) for value in self.bandwidths]
        random = np.random.default_rng(seed)
        directions = random.normal(size=(left.shape[1], cfg.projections))
        directions /= np.maximum(np.linalg.norm(directions, axis=0), 1.0e-30)
        self.projected = self.values @ directions

    def measure(self, left_indices, right_indices):
        means, variances, mmds, wasserstein = [], [], [], []
        for il, ir in zip(left_indices, right_indices, strict=True):
            left, right = self.values[il], self.values[ir]
            means.append(right.mean(axis=0) - left.mean(axis=0))
            variances.append(right.var(axis=0) - left.var(axis=0))
            mmds.append([
                max(float(kernel[np.ix_(il, il)].mean() + kernel[np.ix_(ir, ir)].mean() - 2.0 * kernel[np.ix_(il, ir)].mean()), 0.0)
                for kernel in self.kernels
            ])
            wasserstein.append(float(np.mean(np.abs(np.sort(self.projected[il], axis=0) - np.sort(self.projected[ir], axis=0)))))
        means, variances = np.asarray(means), np.asarray(variances)
        mmds, wasserstein = np.asarray(mmds), np.asarray(wasserstein)
        statistics = np.asarray([np.max(abs(means)), np.max(abs(variances)), np.max(mmds), np.max(wasserstein)])
        if not np.isfinite(statistics).all():
            raise AuditInputError("nonfinite statistic; no statistical conclusion is valid")
        return statistics, means, variances, mmds, wasserstein

    def permute(self, random):
        left, right = [], []
        for indices, original in zip(self.bin_indices, self.left, strict=True):
            order = random.permutation(indices)
            left.append(order[:len(original)])
            right.append(order[len(original):])
        return left, right

    def bootstrap(self, random):
        return (
            [random.choice(indices, size=len(indices), replace=True) for indices in self.left],
            [random.choice(indices, size=len(indices), replace=True) for indices in self.right],
        )


def _pair_report(left, right, left_bins, right_bins, *, calibration_z, calibration_bins, features, center, scale, cfg, seed):
    engine = _PairEngine(
        _standardize(left, center, scale), _standardize(right, center, scale),
        left_bins, right_bins, _bandwidth(calibration_z, calibration_bins), cfg, seed + 1,
    )
    observed, means, variances, mmd, sw = engine.measure(engine.left, engine.right)
    random = np.random.default_rng(seed + 2)
    null = np.asarray([engine.measure(*engine.permute(random))[0] for _ in range(cfg.permutations)])
    # +1 and >= include the observed allocation and ties: no zero Monte Carlo p.
    pvalues = (1.0 + np.sum(null >= observed[None, :] - 1.0e-12, axis=0)) / (cfg.permutations + 1.0)
    joint_pvalue = min(1.0, float(len(_STATISTICS) * pvalues.min()))
    random = np.random.default_rng(seed + 3)
    boot = [engine.measure(*engine.bootstrap(random)) for _ in range(cfg.bootstrap_repeats)]
    levels = (cfg.alpha / 2.0, 1.0 - cfg.alpha / 2.0)
    stat_interval = np.quantile(np.asarray([row[0] for row in boot]), levels, axis=0)
    mean_interval = np.quantile(np.asarray([row[1] for row in boot]), levels, axis=0)
    variance_interval = np.quantile(np.asarray([row[2] for row in boot]), levels, axis=0)
    feature_rows = []
    for index, feature in enumerate(features):
        feature_rows.append({
            "name": feature.name, "group": feature.group, "unit": feature.unit,
            "left_mean": float(left[:, index].mean()), "right_mean": float(right[:, index].mean()),
            "left_variance": float(left[:, index].var()), "right_variance": float(right[:, index].var()),
            "mean_difference": float(right[:, index].mean() - left[:, index].mean()),
            "max_stratum_standardized_mean_difference": float(np.max(abs(means[:, index]))),
            "max_stratum_standardized_variance_difference": float(np.max(abs(variances[:, index]))),
        })
    groups = {}
    for group in dict.fromkeys(feature.group for feature in features):
        indices = [i for i, feature in enumerate(features) if feature.group == group]
        groups[group] = {
            "feature_names": [features[i].name for i in indices], "input_dimension": len(indices), "retained_dimension": len(indices),
            "max_standardized_mean_deviation": float(np.max(abs(means[:, indices]))),
            "max_standardized_variance_deviation": float(np.max(abs(variances[:, indices]))),
        }
    strata = []
    for index in range(8):
        strata.append({
            "nu_bin": index // 2, "temperature_bin": index % 2,
            "left_count": len(engine.left[index]), "right_count": len(engine.right[index]),
            "mean_difference_physical_units": (means[index] * scale).tolist(),
            "mean_difference_ci_physical_units": (mean_interval[:, index] * scale).T.tolist(),
            "variance_difference_squared_physical_units": (variances[index] * scale**2).tolist(),
            "variance_difference_ci_squared_physical_units": (variance_interval[:, index] * scale**2).T.tolist(),
            "mmd2_biased_by_bandwidth": mmd[index].tolist(), "sliced_wasserstein": float(sw[index]),
        })
    return {
        "statistics": {name: {
            "effect": float(observed[i]), "bootstrap_percentile_interval": stat_interval[:, i].tolist(),
            "permutation_pvalue": float(pvalues[i]), "permutation_null_quantiles": np.quantile(null[:, i], (0.05, 0.5, 0.95)).tolist(),
        } for i, name in enumerate(_STATISTICS)},
        "joint_pvalue": joint_pvalue,
        "difference_detected": joint_pvalue <= cfg.alpha,
        "multiplicity": "Bonferroni across four predeclared statistics; each statistic already maximizes over its features/strata/bandwidths",
        "metric_definitions": {
            "mmd": "biased nonnegative RBF MMD squared, maximum over joint strata and reference-calibrated bandwidths",
            "kernel_distance": "squared Euclidean distance over ALL standardized features divided by feature dimension",
            "sliced_wasserstein": "mean projected W1 over fixed random unit directions, maximum over joint strata",
            "moments": "maximum absolute coordinatewise difference across ALL features and joint strata; variance uses ddof=0",
        },
        "permutation": {
            "unit": "complete trajectory", "labels_shuffled": "within_joint_nu_temperature_bin",
            "repeats": cfg.permutations, "plus_one_correction": True, "requires_exchangeability_within_bin": True,
            "minimum_attainable_bonferroni_pvalue": min(1.0, len(_STATISTICS)/(cfg.permutations+1.0)),
            "monte_carlo_resolution_can_reject": len(_STATISTICS)/(cfg.permutations+1.0) <= cfg.alpha,
            "label_allocations_per_stratum": [math.comb(len(a)+len(b), len(a)) for a, b in zip(engine.left, engine.right, strict=True)],
            "small_strata_power_warning": any(len(a) < 8 for a in engine.left),
        },
        "uncertainty": {"method": "independent trajectory bootstrap within each sample and joint bin", "repeats": cfg.bootstrap_repeats, "confidence_level": 1.0 - cfg.alpha, "intervals_are_pointwise_not_simultaneous": True, "finite_sample_coverage_validated": False},
        "bandwidths": list(engine.bandwidths), "features": feature_rows, "groups": groups, "joint_strata": strata,
    }


def audit_diffusion_distribution(
    splits: Mapping[str, AuditSample], *, features: Sequence[AuditFeature],
    nu_edges: Sequence[float], temperature_edges: Sequence[float],
    config: DiffusionAuditConfig | None = None,
) -> dict[str, Any]:
    """Compare independent generated trajectories to D0 under a fixed 4x2 design.

    All six roles are mandatory even though train/validation are used only for
    lineage and design checks. Physical-feature extraction and artifact hashing
    remain the caller's responsibility. Changing IDs cannot prove independence.
    """
    cfg = config or DiffusionAuditConfig()
    cfg.validate()
    schema = _feature_schema(features)
    edges = _edges(nu_edges, temperature_edges)
    if set(splits) != set(_ROLES):
        raise AuditInputError(f"exactly these split roles are required: {_ROLES}")
    validated = {role: _sample(splits[role], len(schema), edges, role, cfg.minimum_per_stratum) for role in _ROLES}
    lineage = _lineage_guard(splits)
    if len({len(validated[role][0]) for role in ("reference_a", "reference_b", "generated")}) != 1:
        raise AuditInputError("reference A/B/generated need identical counts per joint stratum")
    calibration, _, calibration_bins = validated["calibration"]
    center, scale, std = _standardizer(calibration, schema)
    calibration_z = _standardize(calibration, center, scale)
    comparisons = {}
    for offset, role in enumerate(("reference_b", "generated")):
        left, _, left_bins = validated["reference_a"]
        right, _, right_bins = validated[role]
        comparisons["reference_null" if role == "reference_b" else "generated"] = _pair_report(
            left, right, left_bins, right_bins, calibration_z=calibration_z, calibration_bins=calibration_bins,
            features=schema, center=center, scale=scale, cfg=cfg, seed=cfg.seed + 10000 * offset,
        )
    found = comparisons["generated"]["difference_detected"]
    return {
        "schema": SCHEMA, "status": "difference_detected" if found else "insufficient_evidence", "admitted": False,
        "equivalence": "not_established: no preregistered physical equivalence margins, independent power evidence or physical qualification supplied",
        "nonrejection_is_not_equivalence": True,
        "preprocessing": {
            "fit_split": "calibration_only", "generated_used_for_fit": False, "pca_or_learned_projection": False,
            "input_dimension": len(schema), "retained_dimension": len(schema), "clipping": False,
            "features": [{"name": f.name, "group": f.group, "unit": f.unit, "mean": float(center[i]), "standard_deviation": float(std[i]), "scale": float(scale[i]), "scale_floor": float(f.scale_floor), "floor_active": bool(std[i] <= f.scale_floor), "retained_when_calibration_constant": True} for i, f in enumerate(schema)],
        },
        "sampling_design": {"nu_edges": edges[0].tolist(), "temperature_edges": edges[1].tolist(), "strata_count": 8, "balanced_within_each_split": True, "split_counts": {role: len(validated[role][0]) for role in _ROLES}, "continuous_condition_distribution_within_bin_is_caller_contract": True},
        "lineage": lineage, "comparisons": comparisons,
        "limitations": [
            "reference-null result is one diagnostic draw, not global operating-characteristic calibration",
            "four Bonferroni-adjusted tests detect discrepancy; they cannot certify complete distribution equality",
            "within-bin label exchangeability requires matching continuous condition sampling laws, not merely matching counts",
            "physical feature adequacy, gauge correctness, quadrature convergence and reference fidelity require separate evidence",
            "bootstrap intervals have not been qualified for finite-sample coverage or equivalence admission",
        ],
    }


def _wilson(successes: int, trials: int) -> list[float]:
    z = 1.959963984540054
    p = successes / trials
    denominator = 1.0 + z*z/trials
    center = (p + z*z/(2*trials)) / denominator
    half = z * np.sqrt(p*(1-p)/trials + z*z/(4*trials*trials)) / denominator
    return [float(max(0.0, center-half)), float(min(1.0, center+half))]


def _synthetic_draw(random, count, edges):
    conditions, features = [], []
    for index in range(8):
        nu_bin, temperature_bin = index // 2, index % 2
        nu = random.uniform(edges[0][nu_bin], edges[0][nu_bin+1], count)
        temperature = random.uniform(edges[1][temperature_bin], edges[1][temperature_bin+1], count)
        interaction = (-1.0 if nu_bin < 2 else 1.0) * (-1.0 if temperature_bin == 0 else 1.0)
        x, y = random.normal(size=(2, count))
        features.append(np.column_stack((x, y, 0.6 + 0.1*x + 0.05*random.normal(size=count), interaction + 0.05*random.normal(size=count), 0.01*random.normal(size=count), 2 + 0.3*random.normal(size=count))))
        conditions.append(np.column_stack((nu, temperature)))
    return np.concatenate(features), np.concatenate(conditions)


def audit_selfcheck(
    *, config: DiffusionAuditConfig | None = None,
    features: Sequence[AuditFeature] | None = None,
    calibration_bank: AuditSample | None = None,
    nu_edges: Sequence[float] | None = None,
    temperature_edges: Sequence[float] | None = None,
    trials: int = 20, cases_per_stratum: int = 8,
) -> dict[str, Any]:
    """Measure rejection on a null and declared bad controls, never auto-qualify.

    Bank mode partitions donor roots into disjoint preprocessing/evaluation
    banks, then independently resamples evaluation donors with replacement.
    Its Wilson intervals measure Monte Carlo uncertainty CONDITIONAL ON that
    fixed bank, not uncertainty about the true PDE population.
    """
    cfg = config or DiffusionAuditConfig()
    cfg.validate()
    if isinstance(trials, bool) or int(trials) != trials or isinstance(cases_per_stratum, bool) or int(cases_per_stratum) != cases_per_stratum or trials < 1 or cases_per_stratum < 2:
        raise AuditInputError("selfcheck needs positive trials and at least two cases per stratum")
    edges = _edges(nu_edges if nu_edges is not None else (0., 1., 2., 3., 4.), temperature_edges if temperature_edges is not None else (1., 2., 3.))
    synthetic = calibration_bank is None
    if synthetic:
        if features is not None:
            raise AuditInputError("synthetic controls use their own declared six-feature schema")
        schema = tuple(AuditFeature(name, group, "synthetic_dimensionless", 1e-8) for name, group in (
            ("mode_1", "phase"), ("mode_2", "phase"), ("phase_coupling", "phase"),
            ("condition_interaction", "conditional"), ("small_variance_coordinate", "coordinate"), ("energy", "bulk"),
        ))
    else:
        if features is None:
            raise AuditInputError("empirical-bank controls require the physical feature schema")
        schema = _feature_schema(features)
    random = np.random.default_rng(cfg.seed + 400000)
    donor_counts = None
    if synthetic:
        fit, fit_conditions = _synthetic_draw(random, max(16, cases_per_stratum), edges)
        fit_bins = _strata(fit_conditions, edges)
        def draw():
            matrix, conditions = _synthetic_draw(random, cases_per_stratum, edges)
            return matrix, _strata(conditions, edges)
    else:
        bank, _, bank_bins = _sample(calibration_bank, len(schema), edges, "selfcheck_bank", 4)
        _lineage_guard({"selfcheck_bank": calibration_bank})
        fit_indices, evaluation_indices = [], []
        for index in range(8):
            indices = random.permutation(np.flatnonzero(bank_bins == index))
            split = len(indices) // 2
            fit_indices.extend(indices[:split].tolist())
            evaluation_indices.append(indices[split:])
        fit, fit_bins = bank[fit_indices], bank_bins[fit_indices]
        donor_counts = [len(indices) for indices in evaluation_indices]
        def draw():
            selected = np.concatenate([random.choice(indices, cases_per_stratum, replace=True) for indices in evaluation_indices])
            return bank[selected].copy(), bank_bins[selected]
    center, scale, _ = _standardizer(fit, schema)
    fit_z = _standardize(fit, center, scale)
    conditional_centers = np.asarray([fit[fit_bins == index].mean(axis=0) for index in range(8)])
    controls = {name: [] for name in ("null", "mean_collapse", "variance_shrink", "interaction_reversal", "off_pca_coordinate", "phase_or_coordinate_error")}
    for trial in range(int(trials)):
        left, left_bins = draw()
        candidate, candidate_bins = draw()
        conditional_center = conditional_centers[candidate_bins]
        variants = {
            "null": candidate.copy(),
            "mean_collapse": conditional_center.copy(),
            "variance_shrink": conditional_center + 0.1*(candidate-conditional_center),
        }
        interaction = candidate.copy()
        if synthetic:
            interaction[:, 3] *= -1.0
        else:
            # Flip the temperature-bin association while retaining each bin's
            # sample count. This is a conditional feature-law control, not a
            # claim of a physically executable transformation of f.
            for index in range(8):
                interaction[candidate_bins == index] = candidate[candidate_bins == (index ^ 1)]
        variants["interaction_reversal"] = interaction
        orthogonal = candidate.copy()
        coordinate = int(np.argmin(fit.var(axis=0)))
        orthogonal[:, coordinate] += max(float(scale[coordinate]) * 20, schema[coordinate].scale_floor * 20)
        variants["off_pca_coordinate"] = orthogonal
        phase = candidate.copy()
        phase_index = 2 if synthetic else 0
        phase[:, phase_index] = -phase[:, phase_index] + (0.0 if synthetic else 10.0*scale[phase_index])
        variants["phase_or_coordinate_error"] = phase
        for name, values in variants.items():
            report = _pair_report(left, values, left_bins, candidate_bins, calibration_z=fit_z, calibration_bins=fit_bins, features=schema, center=center, scale=scale, cfg=cfg, seed=cfg.seed + trial*1000 + 500000)
            controls[name].append(bool(report["difference_detected"]))
    # Identity guard is separate from statistical power: relabelling a copied
    # training trajectory is not a statistical goodness-of-fit issue.
    replay_sample = AuditSample(np.zeros((1, len(schema))), np.zeros((1, 2)), ("same_training_root",))
    replay_detected = False
    try:
        _lineage_guard({"train": replay_sample, "generated": replay_sample})
    except AuditInputError:
        replay_detected = True
    results = {name: {"detections": int(sum(rows)), "trials": int(trials), "rate": float(np.mean(rows)), "wilson_95_interval": _wilson(sum(rows), trials), "interpretation": "false_rejection_rate" if name == "null" else "control_detection_rate"} for name, rows in controls.items()}
    results["training_replay"] = {"detected": replay_detected, "mechanism": "root_lineage_guard", "statistical_power_claim": False}
    return {
        "schema": "vpfp_diffusion_audit_selfcheck_v1", "admitted": False, "status": "diagnostic_only",
        "population": "declared_synthetic_feature_distribution" if synthetic else "conditional_on_fixed_empirical_bank",
        "physical_model_validation": False, "independent_pde_population_trials": False,
        "preprocessing_fit_and_evaluation_donors_disjoint": True,
        "preprocessing_fixed_across_trials": True,
        "evaluation_donors_per_stratum": donor_counts,
        "bank_resampling_with_replacement": not synthetic,
        "interval_scope": "Monte Carlo rejection probability conditional on fixed preprocessing and, for bank mode, the fixed donor bank",
        "control_semantics": {
            "mean_collapse": "collapse to disjoint-fit-bank conditional-bin means",
            "variance_shrink": "scale residuals about disjoint-fit-bank conditional-bin means by 0.1",
            "interaction_reversal": "synthetic joint interaction sign reversal" if synthetic else "temperature-bin feature-law reassignment within nu bins",
            "off_pca_coordinate": "large error in the smallest-calibration-variance coordinate; no PCA is actually used",
            "phase_or_coordinate_error": "synthetic phase-coupling feature sign reversal" if synthetic else "declared feature sign/offset corruption, not asserted physical phase transformation",
        },
        "controls": results,
        "power_qualification": "not_established: controls lack externally justified physical effect margins and independent PDE confirmation",
        "seed": cfg.seed, "cases_per_stratum": cases_per_stratum,
    }
