"""VPFP regime measurements and reference-support coverage, never admission.

One row is one complete native answer. Physics is nondimensionalized using the
declared background density, thermal speed, horizon and domain length, never a
generated or tiny reference standard deviation. Reference-only PCA is a metric
view, not a physical-validity classifier. Fixed velocity probes are NOT a
full-velocity L2 quadrature. Failure and unresolved-numerics counts are kept.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import importlib
import json
import os
from typing import Mapping

import numpy as np
import torch

from .diffusion_audit import AuditSample, _edges, _lineage_guard, _strata
from .diffusion_features import _check_conditions
from .perturbation_backend import _legacy_api

SCHEMA = "vpfp_regime_fidelity_reference_coverage_v3"
BUDGET_AXES = ("source_fixed_probe_relative_to_collision", "net_momentum_injection_relative_to_collision",
               "net_energy_injection_relative_to_collision")
BUDGET_GRID = (0., *np.logspace(-4, 2, 25).tolist())


@dataclass(frozen=True)
class VPFPRegimeFeaturePlan:
    time_fractions: tuple = (0., .015625, .0625, .125, .25, .5, .75, 1.)
    grid_sizes: tuple = (17, 33)
    phase_probes_per_time: int = 16
    probe_seed: int = 2026090701
    chunk_size: int = 128
    independent_source_probe_count: int = 4
    numerical_absolute_budget: float | None = None

    def validate(self, cfg):
        times = np.asarray(self.time_fractions, dtype=float)
        if (times.ndim != 1 or len(times) < 3 or not np.isfinite(times).all()
                or times[0] != 0 or times[-1] != 1 or np.any(np.diff(times) <= 0)):
            raise ValueError("ordered physical time fractions must span 0 to 1")
        sizes = self.grid_sizes
        if (len(sizes) < 2 or tuple(sorted(set(sizes))) != sizes or any(
            isinstance(n, bool) or not isinstance(n, int) or n % 2 != 1 or n < 2*max(
                cfg.rho_max_mode_x, cfg.rho_max_mode_y, cfg.structured_local_spatial_mode)+1 for n in sizes)):
            raise ValueError("at least two increasing odd mode-resolving spatial grids required")
        for name in ("phase_probes_per_time", "chunk_size", "independent_source_probe_count"):
            if isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.probe_seed, bool) or not isinstance(self.probe_seed, int) or self.probe_seed < 0:
            raise ValueError("probe_seed must be a nonnegative integer")
        if self.numerical_absolute_budget is not None and (
            not np.isfinite(self.numerical_absolute_budget) or self.numerical_absolute_budget <= 0):
            raise ValueError("declared dimensionless numerical budget must be positive or absent")


@dataclass(frozen=True)
class RegimeAuditSample:
    features: object
    conditions: object
    root_ids: tuple
    answer_sha256: tuple
    valid_mask: object
    numerical_uncertain_mask: object
    budgets: object
    failures: tuple = ()
    lineage_ids: tuple | None = None
    budget_defined_mask: object | None = None


def _rms(value, axis=None):
    a = np.asarray(value, dtype=np.float64)
    maximum = np.max(np.abs(a), axis=axis, keepdims=True)
    scale = np.where(maximum > 0, maximum, 1.)
    value = maximum*np.sqrt(np.mean((a/scale)**2, axis=axis, keepdims=True))
    return np.squeeze(value, axis=axis) if axis is not None else float(value.reshape(-1)[0])


def _json_digest(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _distance(a, b):
    # Direct differences avoid negative cancellation near duplicates.
    return np.sqrt(np.sum((a[:, None]-b[None])**2, axis=-1))


def _frechet(a, b):
    if min(len(a), len(b)) < 2:
        return {"available": False, "reason": "fewer_than_two_valid_answers"}
    ma, mb = a.mean(0), b.mean(0)
    ca, cb = np.atleast_2d(np.cov(a, rowvar=False)), np.atleast_2d(np.cov(b, rowvar=False))
    # Covariance square roots amplify roundoff at rank-deficient strata. The
    # nuclear norm of centered rectangular sample factors gives the SAME Bures
    # cross term without an eigenvalue sqrt or a scientific regularizer.
    factor_product = (a-ma)@(b-mb).T/np.sqrt((len(a)-1)*(len(b)-1))
    cross = float(np.linalg.svd(factor_product, compute_uv=False).sum())
    value = float(np.sum((ma-mb)**2)+np.trace(ca+cb)-2*cross)
    budget = 128*max(len(a), len(b), len(ca))*np.finfo(float).eps*max(float(np.trace(ca+cb)), 1.)
    if value < -budget:
        raise ValueError("negative Fréchet distance beyond FP64 roundoff")
    return {"available": True, "squared_distance": max(value, 0.), "raw_squared_distance": value,
            "method": "centered_sample_factor_nuclear_norm_no_covariance_regularizer",
            "meaning": "Gaussian first_two_moment_distance_in_reference_fitted_physical_PCA_not_image_FID"}


def _support(reference, generated, *, k, attempted_count):
    if len(reference) <= k:
        return {"available": False, "reason": "reference_count_must_exceed_k", "k": k,
                "reference_count": len(reference), "valid_generated_count": len(generated), "attempted_count": attempted_count}
    rr = _distance(reference, reference)
    np.fill_diagonal(rr, np.inf)
    radii = np.partition(rr, k-1, axis=1)[:, k-1]
    cross = _distance(reference, generated)
    inside = np.any(cross <= radii[:, None], axis=0)
    covered = np.any(cross <= radii[:, None], axis=1)
    result = {"available": True, "k": k, "reference_count": len(reference),
        "valid_generated_count": len(generated), "attempted_count": attempted_count,
        "precision_all_attempts": float(inside.sum()/attempted_count) if attempted_count else None,
        "precision_valid_only": float(inside.mean()) if len(inside) else None,
        "reference_coverage": float(covered.mean()), "zero_reference_radius_count": int((radii == 0).sum()),
        "reference_radius_median": float(np.median(radii)),
        "support_not_physical_validity": True}
    if len(generated) > k:
        gg = _distance(generated, generated)
        np.fill_diagonal(gg, np.inf)
        gr = np.partition(gg, k-1, axis=1)[:, k-1]
        result["recall_generated_neighborhood"] = float(np.any(cross <= gr[None], axis=1).mean())
        result["zero_generated_radius_count"] = int((gr == 0).sum())
    else:
        result["recall_generated_neighborhood"] = None
    return result


def _validated(sample, width, edges):
    x, conditions = np.asarray(sample.features, dtype=float), np.asarray(sample.conditions)
    valid, uncertain, budgets = (np.asarray(sample.valid_mask), np.asarray(sample.numerical_uncertain_mask),
                                  np.asarray(sample.budgets, dtype=float))
    if (x.ndim != 2 or x.shape[1] != width or conditions.shape != (len(x), 2)
            or valid.dtype != bool or uncertain.dtype != bool or valid.shape != (len(x),)
            or uncertain.shape != valid.shape or budgets.shape != (len(x), 3)
            or not np.isfinite(x).all() or not np.isfinite(budgets).all() or np.any(budgets < 0)
            or len(sample.root_ids) != len(x) or len(sample.answer_sha256) != len(x)):
        raise ValueError("aligned finite features, conditions, masks, budgets and identities required")
    bins = _strata(conditions, edges)
    defined = np.ones((len(x), 3), dtype=bool) if sample.budget_defined_mask is None else np.asarray(sample.budget_defined_mask)
    if defined.shape != (len(x), 3) or defined.dtype != bool:
        raise ValueError("budget-defined mask must be aligned boolean [N,3]")
    for digest in sample.answer_sha256:
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("one SHA256 required per attempted native answer")
    return x, valid, uncertain, budgets, bins, defined


def audit_regime_fidelity_coverage(samples: Mapping[str, RegimeAuditSample], *, descriptors,
        nu_edges, temperature_edges, pca_dimension=16, neighborhood_sizes=(5, 3, 10),
        numerical_calibration_metadata=None, bootstrap_repeats=49, seed=2026090703):
    """Reference support diagnostic with failure denominators; no equivalence gate.

    calibration alone fits center/PCA. No variance division, whitening,
    histogram matching, PCA fitting to generated data, or science threshold.
    All attempts remain in denominators even when extraction failed. Uncertain
    but finite rows remain explicitly labeled, not silently dropped or admitted.
    """
    if set(samples) != {"calibration", "reference_a", "reference_b", "generated"}:
        raise ValueError("calibration/reference_a/reference_b/generated are required")
    names = [d["name"] for d in descriptors]
    if not names or len(set(names)) != len(names) or any(d.get("unit") != "dimensionless" for d in descriptors):
        raise ValueError("unique dimensionless physical descriptors required")
    if isinstance(pca_dimension, bool) or pca_dimension != 16 or tuple(neighborhood_sizes) != (5, 3, 10):
        raise ValueError("this frozen diagnostic uses PCA16 and k5 with k3/k10 sensitivity")
    if isinstance(bootstrap_repeats, bool) or not isinstance(bootstrap_repeats, int) or bootstrap_repeats < 0:
        raise ValueError("bootstrap_repeats must be a nonnegative integer")
    edges = _edges(nu_edges, temperature_edges)
    data = {role: _validated(sample, len(names), edges) for role, sample in samples.items()}
    # Repeated generated outputs are evidence of collapse, not a reason to
    # erase a run. Roots/ancestors must still be disjoint; reference leakage
    # must fail. Reference duplicates cannot define an independent null bank.
    lineage = _lineage_guard({role: AuditSample(sample.features, sample.conditions, sample.root_ids,
        None if role == "generated" else sample.answer_sha256, sample.lineage_ids) for role, sample in samples.items()})
    reference_hashes = {h for role, sample in samples.items() if role != "generated" for h in sample.answer_sha256}
    if reference_hashes.intersection(samples["generated"].answer_sha256):
        raise ValueError("generated answer duplicates a frozen reference/calibration answer")
    lineage.update(answer_hashes_supplied_for_all_splits=True,
        generated_repeated_answer_excess=len(samples["generated"].answer_sha256)-len(set(samples["generated"].answer_sha256)))
    calibration = data["calibration"][0][data["calibration"][1]]
    if len(calibration) <= pca_dimension or calibration.shape[1] < pca_dimension:
        raise ValueError("PCA16 needs >16 valid independent calibration answers and >=16 descriptors")
    transformed = np.arcsinh(calibration)
    center = transformed.mean(0)
    _, singular, axes = np.linalg.svd(transformed-center, full_matrices=False)
    axes = axes[:pca_dimension]
    z = {role: (np.arcsinh(values[0])-center)@axes.T for role, values in data.items()}
    total = float(np.square(singular).sum())
    fit_identity = {"root_ids": list(samples["calibration"].root_ids),
        "answer_sha256": list(samples["calibration"].answer_sha256), "valid_mask": data["calibration"][1].tolist()}
    report = {"schema": SCHEMA, "execution_passed": True, "admitted": False,
        "physical_regime_qualified": False, "distribution_equivalence_required": False,
        "status": "descriptive_fidelity_and_reference_support_only",
        "statistical_unit": "one_complete_trajectory_per_attempt", "lineage": lineage,
        "physical_thresholds": None, "confirmation_512_completed": False,
        "numerical_calibration": numerical_calibration_metadata or {"status": "not_qualified"},
        "geometry": {"fit_split": "calibration_reference_only", "generated_used_for_fit": False,
            "transform": "elementwise_asinh_after_physical_nondimensionalization_no_clipping",
            "standard_deviation_division": False, "whitening": False, "retained_dimension": pca_dimension,
            "input_dimension": len(names), "center": center.tolist(), "axes": axes.tolist(),
            "retained_variance_fraction": float(np.square(singular[:pca_dimension]).sum()/total) if total else None,
            "singular_values": singular.tolist(), "fit_identity_sha256": _json_digest(fit_identity)},
        "cohorts": {}, "comparisons": {}, "budget_curves": {}, "limitations": [
            "reference support is not physical invalidity outside the reference bank",
            "PCA16 discards directions: original physical descriptors and numerical errors remain necessary",
            "source fixed-probe RMS is not a full-space L2 norm; net injection uses macro balance identities",
            "8 strata and several k values are descriptive, not multiplicity-adjusted admission tests",
            "no equivalence margins or scientific acceptance rates are invented from generated results"]}
    for role, sample in samples.items():
        x, valid, uncertain, budgets, bins, defined = data[role]
        report["cohorts"][role] = {"attempted_count": len(x), "hard_valid_count": int(valid.sum()),
            "failed_count": int((~valid).sum()), "numerically_uncertain_count": int((uncertain & valid).sum()),
            "undefined_budget_count_by_axis": {name: int((~defined[:, i]).sum()) for i, name in enumerate(BUDGET_AXES)},
            "failures": list(sample.failures), "strata_attempted": np.bincount(bins, minlength=8).tolist(),
            "strata_valid": np.bincount(bins[valid], minlength=8).tolist(),
            "physical_descriptor_summary": [{"name": name,
                "mean": float(x[valid, i].mean()) if valid.any() else None,
                "q10_q50_q90": np.quantile(x[valid, i], [.1, .5, .9]).tolist() if valid.any() else None}
                for i, name in enumerate(names)]}
    def comparison(role, mask=None):
        reference = data["reference_a"]
        right = data[role]
        eligible = right[1] if mask is None else right[1] & mask
        rows = []
        for index in range(8):
            r = z["reference_a"][reference[1] & (reference[4] == index)]
            g = z[role][eligible & (right[4] == index)]
            attempted = int((right[4] == index).sum())
            rows.append({"stratum": index, "frechet": _frechet(r, g),
                "neighbors": {str(k): _support(r, g, k=k, attempted_count=attempted) for k in neighborhood_sizes}})
        return {"joint_strata": rows,
            "global_frechet_descriptive": _frechet(z["reference_a"][reference[1]], z[role][eligible])}
    report["comparisons"]["reference_null"] = comparison("reference_b")
    report["comparisons"]["generated"] = comparison("generated")
    gx, gv, gu, gb, bins, defined = data["generated"]
    for axis, name in enumerate(BUDGET_AXES):
        curve = []
        for budget in BUDGET_GRID:
            selected = gv & defined[:, axis] & (gb[:, axis] <= budget)
            coverage = [_support(z["reference_a"][data["reference_a"][1] & (data["reference_a"][4] == index)],
                z["generated"][selected & (bins == index)], k=5, attempted_count=int((bins == index).sum())).get("reference_coverage")
                for index in range(8)]
            curve.append({"maximum_dimensionless_budget": budget, "retained_count": int(selected.sum()),
                "retained_numerically_uncertain_count": int((selected & gu).sum()),
                "undefined_budget_count_all_attempts": int((~defined[:, axis]).sum()),
                "retained_is_not_physics_accepted": True,
                "retained_fraction_all_attempts": float(selected.sum()/len(gv)) if len(gv) else None,
                "k5_reference_coverage_by_stratum": coverage})
        report["budget_curves"][name] = {"grid": list(BUDGET_GRID), "curve": curve,
            "under_positive_grid_count": int((gv & defined[:, axis] & (gb[:, axis] > 0) & (gb[:, axis] < 1.e-4)).sum()),
            "zero_count": int((gv & defined[:, axis] & (gb[:, axis] == 0)).sum()),
            "undefined_budget_count_all_attempts": int((~defined[:, axis]).sum()),
            "overflow_count": int((gv & defined[:, axis] & (gb[:, axis] > 100)).sum()), "automatic_threshold_selected": False}
    # Each one-axis curve deliberately leaves the other axes unconstrained.
    # Also show a fixed intersection, without learning weights or a total score.
    joint_curve = []
    all_defined = defined.all(1)
    for budget in BUDGET_GRID:
        selected = gv & all_defined & (gb <= budget).all(1)
        coverage = [_support(z["reference_a"][data["reference_a"][1] & (data["reference_a"][4] == index)],
            z["generated"][selected & (bins == index)], k=5,
            attempted_count=int((bins == index).sum())).get("reference_coverage") for index in range(8)]
        joint_curve.append({"maximum_dimensionless_budget_on_each_axis": budget,
            "retained_count": int(selected.sum()),
            "retained_numerically_uncertain_count": int((selected & gu).sum()),
            "retained_fraction_all_attempts": float(selected.sum()/len(gv)) if len(gv) else None,
            "undefined_any_budget_count_all_attempts": int((~all_defined).sum()),
            "k5_reference_coverage_by_stratum": coverage,
            "retained_is_not_physics_accepted": True})
    report["joint_budget_curve"] = {"grid": list(BUDGET_GRID), "curve": joint_curve,
        "axes": list(BUDGET_AXES), "rule": "every_original_axis_at_most_same_fixed_budget",
        "axis_multipliers": [1., 1., 1.], "single_axis_curves_constrain_only_the_named_axis": True,
        "automatic_threshold_selected": False, "candidate_fitted_weights": False,
        "meaning": "fixed_dimensionless_box_intersection_not_a_total_quality_score_or_admission_rule"}
    report["morphology_occupancy"] = _morphology_occupancy(data, names)
    report["fine_structure_coverage"] = _fine_structure_coverage(data, names)
    report["semantic_near_duplicates"] = _near_duplicate_diagnostic(data, z)
    report["sample_count_curves"] = _sample_count_curves(data, z)
    report["trajectory_bootstrap"] = _trajectory_bootstrap(data, z, bootstrap_repeats, seed)
    controls = calibration_controls(calibration, data["reference_b"][0][data["reference_b"][1]])
    ref = z["reference_a"][data["reference_a"][1]]
    control_reports = {}
    for name, x in controls.pop("controls").items():
        coordinate = (np.arcsinh(x)-center)@axes.T
        control_reports[name] = {"global_frechet_descriptive": _frechet(ref, coordinate),
            "global_support_descriptive": _support(ref, coordinate, k=5, attempted_count=len(x))}
    report["metric_calibration_controls"] = {**controls, "results": control_reports,
        "global_control_is_not_condition_stratified": True}
    json.dumps(report, allow_nan=False)
    return report


def _morphology_occupancy(data, names):
    """Equal-cell presence, not matching the reference's frequency weights."""
    suffixes = ("j_coherent_norm", "thermal_energy_mean", "thermal_anisotropy",
                "j_spatial_std", "dt_K_rms")
    selected = {name: [i for i, full in enumerate(names) if full.rsplit(".", 1)[-1] == name] for name in suffixes}
    selected = {name: indices for name, indices in selected.items() if indices}
    if not selected:
        return {"available": False, "reason": "no_registered_physical_morphology_descriptors"}
    coordinates = {role: np.column_stack([values[0][:, indices].mean(1) for indices in selected.values()])
                   for role, values in data.items()}
    cal = coordinates["calibration"][data["calibration"][1]]
    # A binary coarse grid avoids 4**5 almost-empty cells with 16 references
    # per condition stratum. Constant calibration axes carry no boundary.
    edges = [np.asarray([np.median(cal[:, i])]) if np.ptp(cal[:, i]) > 0 else np.empty(0)
             for i in range(cal.shape[1])]
    cells = {role: [tuple(int(np.searchsorted(edge, row[i], side="right")) for i, edge in enumerate(edges))
                   for row in value] for role, value in coordinates.items()}
    result = {"available": True, "fit_split": "calibration_only", "axes": list(selected),
        "coordinate_reduction": "mean_of_registered_fixed_time_measurements",
        "cell_resolution": "one_calibration_median_boundary_per_nonconstant_axis",
        "edges": [e.tolist() for e in edges], "edges_are_not_physical_acceptance_thresholds": True,
        "reference_frequency_weighting": False, "rows": []}
    for stratum in range(8):
        ref = {cells["reference_a"][i] for i in np.flatnonzero(data["reference_a"][1] & (data["reference_a"][4] == stratum))}
        for role in ("reference_b", "generated"):
            occupied = {cells[role][i] for i in np.flatnonzero(data[role][1] & (data[role][4] == stratum))}
            result["rows"].append({"stratum": stratum, "cohort": role, "reference_cells": len(ref),
                "occupied_cells": len(occupied), "covered_reference_cells": len(ref & occupied),
                "missing_reference_cells": [list(c) for c in sorted(ref-occupied)],
                "new_cells": [list(c) for c in sorted(occupied-ref)],
                "new_cells_are_not_physics_invalid": True})
    return result


def _fine_structure_coverage(data, names):
    """Fine x structure and velocity-shape variation, without sparse joint bins.

    The existing coarse joint morphology table remains unchanged. These two
    additional marginal views do not claim complete micro-shape coverage.
    """
    suffixes = ("K_spectrum_high", "dv_f_fixed_probe_rms")
    result = {"fit_split": "calibration_only", "coordinate_reduction": "mean_of_registered_times",
        "reference_frequency_weighting": False, "joint_shape_coverage_claimed": False,
        "meaning": "separate_marginal_fine_spatial_energy_and_velocity_shape_diagnostics",
        "axes": {}}
    for name in suffixes:
        indices = [i for i, full in enumerate(names) if full.rsplit(".", 1)[-1] == name]
        if not indices:
            result["axes"][name] = {"available": False, "reason": "descriptor_absent"}
            continue
        values = {role: row[0][:, indices].mean(1) for role, row in data.items()}
        calibration = values["calibration"][data["calibration"][1]]
        edges = (np.unique(np.quantile(calibration, [.25, .5, .75]))
                 if np.ptp(calibration) > 0 else np.empty(0))
        cells = {role: np.searchsorted(edges, value, side="right") for role, value in values.items()}
        rows = []
        for stratum in range(8):
            mask = data["reference_a"][1] & (data["reference_a"][4] == stratum)
            reference = set(cells["reference_a"][mask].tolist())
            for role in ("reference_b", "generated"):
                mask = data[role][1] & (data[role][4] == stratum)
                occupied = set(cells[role][mask].tolist())
                rows.append({"stratum": stratum, "cohort": role,
                    "reference_bins": sorted(reference), "occupied_bins": sorted(occupied),
                    "covered_reference_bins": sorted(reference & occupied),
                    "missing_reference_bins": sorted(reference-occupied),
                    "new_bins": sorted(occupied-reference), "new_bins_are_not_physics_invalid": True})
        result["axes"][name] = {"available": True, "calibration_quartile_boundaries": edges.tolist(),
            "boundaries_are_not_physical_acceptance_thresholds": True, "rows": rows}
    return result


def _near_duplicate_diagnostic(data, z):
    rows = []
    for stratum in range(8):
        cal = z["calibration"][data["calibration"][1] & (data["calibration"][4] == stratum)]
        if len(cal) < 2:
            rows.append({"stratum": stratum, "available": False}); continue
        distances = _distance(cal, cal); np.fill_diagonal(distances, np.inf)
        resolutions = np.quantile(distances.min(1), [.1, .5, .9])
        row = {"stratum": stratum, "available": True, "calibration_NN_q10_q50_q90": resolutions.tolist(), "cohorts": {}}
        for role in ("reference_a", "reference_b", "generated"):
            valid = data[role][1] & (data[role][4] == stratum)
            x, features = z[role][valid], data[role][0][valid]
            distances = _distance(x, x); np.fill_diagonal(distances, np.inf)
            nearest = distances.min(1) if len(x) else np.empty(0)
            row["cohorts"][role] = {"valid_count": len(x),
                "exact_full_descriptor_duplicate_excess": len(features)-len(np.unique(features, axis=0)),
                "answers_with_PCA_neighbor_within_resolution": [int((nearest <= r).sum()) for r in resolutions]}
        rows.append(row)
    return {"rows": rows, "fit_split": "calibration_only", "physics_duplicate_threshold": None,
        "meaning": "semantic_resolution_sensitivity_in_fixed_PCA_not_bitwise_answer_identity"}


def _sample_count_curves(data, z):
    rows = []
    for count in (16, 32, 64, 128, 512):
        candidates = [np.flatnonzero(data["generated"][4] == s)[:count//8] for s in range(8)]
        if any(len(indices) != count//8 for indices in candidates):
            rows.append({"attempted_count": count, "available": False, "reason": "not_enough_frozen_attempts_no_resampling"}); continue
        strata = []
        for s, indices in enumerate(candidates):
            valid = indices[data["generated"][1][indices]]
            reference = z["reference_a"][data["reference_a"][1] & (data["reference_a"][4] == s)]
            point = _support(reference, z["generated"][valid], k=5, attempted_count=len(indices))
            point["frechet"] = _frechet(reference, z["generated"][valid])
            strata.append(point)
        selected = np.concatenate(candidates)
        valid = selected[data["generated"][1][selected]]
        rows.append({"attempted_count": count, "available": True, "strata": strata,
            "valid_count": len(valid), "failed_count": count-len(valid),
            "global_frechet_descriptive": _frechet(z["reference_a"][data["reference_a"][1]], z["generated"][valid])})
    return {"selection": "nested_original_order_prefix_with_equal_attempts_per_condition_stratum", "rows": rows,
        "synthetic_extra_answers": False, "independent_replicates": False,
        "finite_sample_model_dependent_Frechet_bias_removed": False,
        "reference_bank_and_geometry_fixed": True}


def _trajectory_bootstrap(data, z, repeats, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(8):
        r_indices = np.flatnonzero((data["reference_a"][4] == s) & data["reference_a"][1])
        g_indices = np.flatnonzero(data["generated"][4] == s)
        bank = z["reference_a"][r_indices]
        # Freeze the actual support being measured. Resampling reference rows
        # changes query weights, NOT the kNN radii or the union of reference
        # balls. Otherwise bootstrap duplicates spuriously shrink the support.
        if len(bank) > 5:
            distance = _distance(bank, bank)
            np.fill_diagonal(distance, np.inf)
            radii = np.partition(distance, 4, axis=1)[:, 4]
        else:
            radii = None
        collected = {"precision_all_attempts": [], "reference_coverage": [], "squared_frechet": []}
        for _ in range(repeats if len(r_indices) and len(g_indices) else 0):
            queries, gi = rng.choice(len(bank), len(bank)), rng.choice(g_indices, len(g_indices))
            r = bank[queries]
            g = z["generated"][gi[data["generated"][1][gi]]]
            if radii is not None:
                incidence = _distance(bank, g) <= radii[:, None]
                collected["precision_all_attempts"].append(float(np.any(incidence, axis=0).sum()/len(gi)))
                collected["reference_coverage"].append(float(np.any(incidence, axis=1)[queries].mean()))
            frechet = _frechet(r, g)
            if frechet.get("available"): collected["squared_frechet"].append(frechet["squared_distance"])
        rows.append({"stratum": s, "percentile_2_5_97_5": {key: np.quantile(values, [.025, .975]).tolist() if values else None
            for key, values in collected.items()}, "successful_replicates": {key: len(values) for key, values in collected.items()}})
    return {"repeats": repeats, "seed": seed, "resampling_unit": "whole_answer_with_failure_mask_within_condition_stratum",
        "calibration_axes_refitted": False, "reference_and_generated_resampled_independently": True,
        "neighborhoods_refitted_to_resampled_reference": False, "duplicates_can_change_neighborhood_radii": False,
        "reference_resampling_changes_query_weights_only": True,
        "support_intervals_condition_on_frozen_reference_bank_and_geometry": True,
        "unseen_reference_support_uncertainty_estimated": False,
        "intervals_are_exploratory_not_equivalence_or_multiplicity_adjusted": True, "rows": rows}


def calibration_controls(calibration, reference, *, seed=2026090702):
    """FEATURE-LEVEL calibration controls, never physical counterexamples.

    Useful for metric mechanics and collapse/omission sensitivity only. They do
    not qualify physical budgets or precision/recall operating characteristics.
    """
    x = np.asarray(reference, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("finite two-dimensional feature bank required")
    if len(x) < 4:
        return {"schema": "vpfp_feature_level_metric_controls_v1", "physical_calibration": False,
            "available": False, "reason": "fewer_than_four_valid_reference_b_answers",
            "valid_reference_count": len(x), "calibration_fit_unchanged": True, "controls": {}}
    rng = np.random.default_rng(seed)
    shifted = x.copy(); shifted[:, 0] += 1.
    return {"schema": "vpfp_feature_level_metric_controls_v1", "physical_calibration": False, "available": True,
        "calibration_fit_unchanged": True, "controls": {
            "same_bank_permutation": x[rng.permutation(len(x))].copy(),
            "collapsed_same_feature": np.repeat(x[:1], len(x), axis=0),
            "support_half_omission": x[rng.integers(0, len(x)//2, len(x))].copy(),
            "dimensionless_coordinate_shift": shifted},
        "note": "permutation is a same-bank invariance control, not an independent physical reference"}


_MACRO_NAMES = (
    "rho_fluctuation_rms", "jx_coherent", "jy_coherent", "j_coherent_norm",
    "j_spatial_std", "thermal_energy_mean", "thermal_energy_std", "electric_rms",
    "dt_j_rms", "dt_K_rms", "net_momentum_injection", "net_energy_injection",
    "rho_spectrum_low", "rho_spectrum_middle", "rho_spectrum_high",
    "j_spectrum_low", "j_spectrum_middle", "j_spectrum_high",
    "K_spectrum_low", "K_spectrum_middle", "K_spectrum_high", "thermal_anisotropy")
_PROBE_NAMES = (
    "f_fixed_probe_rms", "source_fixed_probe_rms", "dt_f_fixed_probe_rms",
    "transport_fixed_probe_rms", "electric_acceleration_fixed_probe_rms",
    "collision_fixed_probe_rms", "dx_f_fixed_probe_rms", "dv_f_fixed_probe_rms",
    "source_fixed_probe_abs_q99")


def _macro_values(jet, modes, temperature, floor):
    """Full-R2 moments and physical-time derivatives of native normalized c.

    N+1 extension is essential for squared velocity at the highest mode. The
    native normalization/fallback was already selected in the native dtype;
    FP64 moment arithmetic does not make a second fallback decision.
    """
    c, dc = jet["coefficients"].double(), jet["coefficient_derivatives"].double()
    rho, drho = jet["rho"].double(), jet["rho_derivatives"].double()
    degree = max(sum(pair) for pair in modes)
    extended = [(a, total-a) for total in range(degree+2) for a in range(total+1)]
    lookup = {pair: index for index, pair in enumerate(extended)}
    embed = c.new_zeros((len(modes), len(extended)))
    vx, vy = torch.zeros_like(embed), torch.zeros_like(embed)
    for index, pair in enumerate(modes):
        embed[index, lookup[tuple(pair)]] = 1.
        for axis, matrix in enumerate((vx, vy)):
            for step in (-1, 1):
                neighbor = list(pair); neighbor[axis] += step
                if neighbor[axis] >= 0:
                    matrix[index, lookup[tuple(neighbor)]] = np.sqrt(float(temperature)*(pair[axis]+(step == 1)))
    e, a, b = c@embed, c@vx, c@vy
    de, da, db = dc@embed, dc@vx, dc@vy
    norm = c.square().sum(-1)
    if bool((norm <= 0).any()):
        raise ValueError("native normalized amplitude has zero norm")
    dnorm = 2*(dc*c[None]).sum(-1)
    ux, uy = (e*a).sum(-1)/norm, (e*b).sum(-1)/norm
    kinetic = .5*(a.square()+b.square()).sum(-1)/norm
    dux = 2*(de*a[None]).sum(-1)/norm-ux*dnorm/norm
    duy = 2*(de*b[None]).sum(-1)/norm-uy*dnorm/norm
    dk = (da*a[None]+db*b[None]).sum(-1)/norm-kinetic*dnorm/norm
    ux, uy, dux, duy = [(1-floor)*v for v in (ux, uy, dux, duy)]
    kinetic, dk = (1-floor)*kinetic+floor*float(temperature), (1-floor)*dk
    second_x = (1-floor)*a.square().sum(-1)/norm+floor*float(temperature)
    second_y = (1-floor)*b.square().sum(-1)/norm+floor*float(temperature)
    cross = (1-floor)*(a*b).sum(-1)/norm
    anisotropy = torch.sqrt(((second_x-ux.square())-(second_y-uy.square())).square()
                           +4*(cross-ux*uy).square())/float(temperature)
    result = {"rho": rho, "jx": rho*ux, "jy": rho*uy, "K": rho*kinetic,
        "dt_jx": drho[0]*ux+rho*dux[0], "dt_jy": drho[0]*uy+rho*duy[0],
        "dt_K": drho[0]*kinetic+rho*dk[0], "electric": jet["electric"].double(),
        "thermal_energy": kinetic-.5*(ux.square()+uy.square()), "thermal_anisotropy": anisotropy, "coefficients": c}
    if any(not bool(torch.isfinite(value).all()) for value in result.values()) or bool((rho <= 0).any()):
        raise ValueError("nonfinite macro fields or nonpositive density")
    return {key: value.detach().cpu().numpy() for key, value in result.items()}


def _grid_measure(model, latent, condition, plan, size, analytic):
    cfg = model.config
    bg, temperature = float(cfg.equation.background_density), float(condition[1])
    horizon, length = float(cfg.time_domain[1]-cfg.time_domain[0]), np.sqrt(np.prod(cfg.equation.x_domain))
    times = cfg.time_domain[0]+np.asarray(plan.time_fractions)*horizon
    xy = [np.arange(size)*side/size for side in cfg.equation.x_domain]
    points = np.stack(np.meshgrid(times, *xy, indexing="ij"), axis=-1).reshape(-1, 3)
    collected = {}
    for start in range(0, len(points), plan.chunk_size):
        jet = analytic.analytic_structured_macro_jet(model, points[start:start+plan.chunk_size], latent)
        values = _macro_values(jet, model.hermite_modes, temperature, cfg.velocity_floor_mixture)
        for key, value in values.items():
            collected.setdefault(key, []).append(value)
    fields = {key: np.concatenate(chunks).reshape(len(times), size, size, *chunks[0].shape[1:])
              for key, chunks in collected.items()}
    frequency = np.fft.fftfreq(size)*size
    radius2 = frequency[:, None]**2+frequency[None]**2
    bands = ((radius2 > 0) & (radius2 <= 4), (radius2 > 4) & (radius2 <= 16), radius2 > 16)
    jscale, escale = bg*np.sqrt(temperature), bg*temperature
    rows, high = [], []
    for index in range(len(times)):
        v = {key: value[index] for key, value in fields.items()}
        rho, jx, jy, energy, electric = (v[key] for key in ("rho", "jx", "jy", "K", "electric"))
        source_j = np.asarray([np.mean(v["dt_jx"]+float(condition[0])*jx-rho*electric[..., 0]),
            np.mean(v["dt_jy"]+float(condition[0])*jy-rho*electric[..., 1])])*horizon/jscale
        source_k = np.mean(v["dt_K"]-electric[..., 0]*jx-electric[..., 1]*jy
            +2*float(condition[0])*(energy-temperature*rho))*horizon/escale
        row = [_rms(rho-bg)/bg, jx.mean()/jscale, jy.mean()/jscale,
            np.hypot(jx.mean(), jy.mean())/jscale, np.sqrt(jx.var()+jy.var())/jscale,
            v["thermal_energy"].mean()/temperature, v["thermal_energy"].std()/temperature,
            np.sqrt(np.mean(np.sum(electric**2, axis=-1)))*length/temperature,
            np.sqrt(np.mean(v["dt_jx"]**2+v["dt_jy"]**2))*horizon/jscale,
            _rms(v["dt_K"])*horizon/escale, np.linalg.norm(source_j), abs(source_k)]
        for arrays, scale in (((rho,), bg), ((jx, jy), jscale), ((energy,), escale)):
            power = sum(np.abs(np.fft.fft2(array)/(size*size*scale))**2 for array in arrays)
            row.extend(float(power[band].sum()) for band in bands)
        row.append(float(v["thermal_anisotropy"].mean()))
        rows.append(row)
        selected = np.asarray([sum(pair) >= 8 for pair in model.hermite_modes])
        high.append(float(np.mean(np.sum(v["coefficients"][..., selected]**2, axis=-1))))
    return np.asarray(rows), {"grid_size": size, "minimum_density": float(fields["rho"].min()),
        "minimum_thermal_energy": float(fields["thermal_energy"].min()),
        "native_amplitude_degree_ge8_energy_by_time": high,
        "amplitude_proxy_is_not_transformed_density_Hermite_energy": True,
        "net_injection_method": "periodic_spatial_mean_macro_balance_identity_not_independent_source_quadrature",
        "spectral_bands": "radial_integer_wave_number: (0,2],(2,4],(4,grid_limit]; different high-grid support"}


def _probe_measure(model, latent, condition, plan, analytic):
    cfg = model.config
    bg, temperature = float(cfg.equation.background_density), float(condition[1])
    horizon, length = float(cfg.time_domain[1]-cfg.time_domain[0]), np.sqrt(np.prod(cfg.equation.x_domain))
    rng = np.random.default_rng(plan.probe_seed)
    count = plan.phase_probes_per_time
    xy = rng.uniform(size=(count, 2))*np.asarray(cfg.equation.x_domain)
    velocity = rng.normal(size=(count, 2))*np.where(np.arange(count)[:, None] % 2, 3., 1.)*np.sqrt(temperature)
    times = cfg.time_domain[0]+np.asarray(plan.time_fractions)*horizon
    points = np.concatenate([np.column_stack((np.full(count, t), xy, velocity)) for t in times])
    rows, sources, algebraic = [], [], []
    for start in range(0, len(points), plan.chunk_size):
        p = points[start:start+plan.chunk_size]
        values = analytic.manufacture_standard_vpfp_source_analytic(model, p, latent, equation_condition=condition)
        f, source, gradient, electric, collision = [values[key].detach().double().cpu().numpy()
            for key in ("f", "source", "f_gradient", "electric", "collision")]
        dt, transport = gradient[:, 0], (p[:, 3:5]*gradient[:, 1:3]).sum(-1)
        acceleration, coll = (electric*gradient[:, 3:5]).sum(-1), float(condition[0])*collision
        row = np.column_stack((f*temperature/bg, source*temperature*horizon/bg,
            dt*temperature*horizon/bg, transport*temperature*horizon/bg,
            acceleration*temperature*horizon/bg, coll*temperature*horizon/bg,
            np.linalg.norm(gradient[:, 1:3], axis=-1)*length*temperature/bg,
            np.linalg.norm(gradient[:, 3:5], axis=-1)*temperature**1.5/bg))
        if not np.isfinite(row).all() or np.any(f < 0):
            raise ValueError("nonfinite source/derivatives or negative density on fixed probes")
        rows.append(row); sources.append(source)
        algebraic.extend(((dt+transport+acceleration-coll-source)*temperature*horizon/bg).tolist())
    per_probe = np.concatenate(rows).reshape(len(times), count, 8)
    feature = np.column_stack((_rms(per_probe, axis=1), np.quantile(np.abs(per_probe[:, :, 1]), .99, axis=1)))
    independent = {"execution_passed": False, "physical_qualification": False,
        "method": "separate_native_pointwise_autograd_vs_analytic_coordinate_jets",
        "independent_of_source_assembly_algebra_but_same_decoder": True}
    try:
        indices = np.unique(np.linspace(0, len(points)-1, min(plan.independent_source_probe_count, len(points))).astype(int))
        continuous = importlib.import_module("agent_verifier.vpfp.continuous_v2")
        with torch.enable_grad():
            actual = continuous.manufacture_standard_vpfp_source(model, points[indices], latent,
                equation_condition=condition, create_graph=False)["source"].detach().double().cpu().numpy()
        error = (actual-np.concatenate(sources)[indices])*temperature*horizon/bg
        if not np.isfinite(error).all():
            raise ValueError("nonfinite independent source comparison")
        independent.update(execution_passed=True, probe_indices=indices.tolist(),
            maximum_dimensionless_difference=float(np.abs(error).max()), rms_dimensionless_difference=_rms(error))
    except Exception as exc:
        independent["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    return feature, {"scope": "fixed_spatial_velocity_probes_not_full_space_L2_or_velocity_quadrature",
        "points_txyvv": points.tolist(), "algebraic_balance_maximum": float(np.max(np.abs(algebraic))),
        "algebraic_balance_is_independent_check": False, "independent_source_check": independent}


def _collision_budgets(probes, macros, *, nu, horizon, minimum_normal_f_rms):
    """Keep undefined collision ratios separate from finite JSON placeholders.

    Inputs already use horizon normalization. No epsilon is inserted in a
    physical ratio. A subnormal-range probe RMS is numerically unresolved,
    not evidence of small forcing and not a physical-invalidity verdict.
    """
    baseline = np.column_stack((probes[:, 1], macros[:, 10:12]))
    values = np.zeros_like(baseline)
    defined = np.zeros_like(baseline, dtype=bool)
    scale = float(nu)*float(horizon)
    if np.isfinite(scale) and scale > 0:
        defined[:, 1:] = True
        defined[:, 0] = probes[:, 0] >= minimum_normal_f_rms
        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            values[:, 1:] = baseline[:, 1:]/scale
            values[defined[:, 0], 0] = baseline[defined[:, 0], 0]/probes[defined[:, 0], 0]/scale
        defined &= np.isfinite(values)
        values[~defined] = 0.
    all_times_defined = defined.all(0)
    maximum = np.where(all_times_defined, values.max(0), 0.)
    return maximum, all_times_defined, {"horizon_normalized_source_and_injection_by_time": baseline.tolist(),
        "collision_normalized_budget_by_time": [[float(v) if ok else None for v, ok in zip(row, flags)]
            for row, flags in zip(values, defined)],
        "source_reference_f_fixed_probe_rms_by_time": probes[:, 0].tolist(),
        "nu_times_horizon": scale, "minimum_resolvable_normal_f_probe_rms": minimum_normal_f_rms,
        "zero_nu_or_unresolved_f_is_undefined_not_physical_failure": True,
        "source_denominator_is_nu_times_f_RMS_not_collision_operator_RMS": True}


def extract_vpfp_regime_features(model, latents, conditions, *, plan=None, progress=None):
    """Measure native direct answers; retain failures, never generate or solve.

    This entry does not accept a moving-frame m/z carrier: auditing z alone
    would measure a different physical answer. Its direct adapter must be used.
    """
    plan = plan or VPFPRegimeFeaturePlan()
    if isinstance(latents, Mapping) or hasattr(model, "macro_config"):
        raise ValueError("moving-frame carrier requires direct m/z physical audit adapter; z-only audit forbidden")
    cfg, parameter = model.config, next(model.parameters())
    plan.validate(cfg)
    if parameter.dtype != torch.float32:
        raise ValueError("this native measurement protocol requires the frozen FP32 decoder")
    if parameter.device.type == "cuda" and not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("GPU extraction requires a compute allocation")
    values, conditions = np.asarray(latents), np.asarray(conditions)
    if values.ndim != 2 or values.shape[1] != cfg.latent_dim or conditions.shape != (len(values), 2):
        raise ValueError("complete native latents and aligned conditions required")
    _legacy_api()
    analytic = importlib.import_module("agent_verifier.vpfp.analytic_source")
    analytic.require_analytic_structured_source(model)
    names = _MACRO_NAMES+_PROBE_NAMES
    descriptors = [{"name": f"time{index}.{name}", "unit": "dimensionless",
        "group": "operator_fixed_probes" if name in _PROBE_NAMES else "macro_spatial",
        "time_fraction": fraction} for index, fraction in enumerate(plan.time_fractions) for name in names]
    matrix = np.zeros((len(values), len(descriptors)), dtype=np.float64)
    cases, was_training = [], model.training
    model.eval()
    try:
        with torch.no_grad(), analytic.ieee_float32_coordinate_matmul(model):
            for index in range(len(values)):
                with np.errstate(over="ignore", invalid="ignore"):
                    latent = np.array(values[index], dtype="<f4", copy=True)
                    condition = np.array(conditions[index], dtype="<f4", copy=True)
                entry = {"row": index, "input_answer_sha256": sha256(latent.tobytes()+condition.tobytes()).hexdigest(),
                    "hard_valid": False, "numerically_uncertain": True, "budgets": [0., 0., 0.],
                    "budget_defined_mask": [False, False, False]}
                try:
                    _check_conditions(condition[None], cfg)
                    if not np.isfinite(latent).all():
                        raise ValueError("nonfinite native answer")
                    latent.flags.writeable = condition.flags.writeable = False
                    grids = [_grid_measure(model, latent, condition, plan, size, analytic) for size in plan.grid_sizes]
                    probes, probe_report = _probe_measure(model, latent, condition, plan, analytic)
                    row = np.column_stack((grids[-1][0], probes)).ravel()
                    if not np.isfinite(row).all():
                        raise ValueError("nonfinite dimensionless descriptors")
                    matrix[index] = row
                    sensitivities = [{"coarse": plan.grid_sizes[j], "fine": plan.grid_sizes[j+1],
                        "maximum_absolute_dimensionless_difference": float(np.abs(grids[j+1][0]-grids[j][0]).max()),
                        "difference_by_time_and_descriptor": (grids[j+1][0]-grids[j][0]).tolist(),
                        "is_convergence_certificate": False} for j in range(len(grids)-1)]
                    independent = probe_report["independent_source_check"]
                    uncertain = plan.numerical_absolute_budget is None or not independent["execution_passed"]
                    if not uncertain:
                        uncertain = max([s["maximum_absolute_dimensionless_difference"] for s in sensitivities]+
                            [independent["maximum_dimensionless_difference"]]) > plan.numerical_absolute_budget
                    horizon = float(cfg.time_domain[1]-cfg.time_domain[0])
                    budgets, defined, ledger = _collision_budgets(probes, grids[-1][0], nu=float(condition[0]), horizon=horizon,
                        minimum_normal_f_rms=np.finfo(np.float32).tiny*float(condition[1])/float(cfg.equation.background_density))
                    baseline = np.asarray(ledger["horizon_normalized_source_and_injection_by_time"])
                    fractions = np.asarray(plan.time_fractions)
                    entry.update(hard_valid=True, numerically_uncertain=bool(uncertain or not defined.all()), budgets=budgets.tolist(),
                        budget_defined_mask=defined.tolist(), budget_ledger=ledger,
                        grids=[g[1] for g in grids], numerical_sensitivity=sensitivities, source_probes=probe_report,
                        horizon_normalized_early_late_source_and_injection={label: baseline[mask].mean(0).tolist() for label, mask in
                            (("early_fraction_le_0_25", fractions <= .25), ("late_fraction_ge_0_5", fractions >= .5))})
                except Exception as exc:
                    matrix[index] = 0.
                    entry["failure"] = {"type": type(exc).__name__, "message": str(exc)}
                cases.append(entry)
                if progress is not None:
                    progress(index+1, len(values), entry)
    finally:
        model.train(was_training)
    report = {"schema": "vpfp_dimensionless_native_measurements_v2", "plan": asdict(plan),
        "plan_sha256": _json_digest(asdict(plan)), "decoder_config_sha256": _json_digest(cfg.to_dict()),
        "cases": cases, "complete": len(cases) == len(values), "admitted": False,
        "physical_regime_qualified": False, "full_velocity_source_quadrature_completed": False,
        "independent_physical_calibration_completed": False,
        "scales": "rho_background; sqrt(condition_temperature); horizon; sqrt(Lx*Ly)",
        "budget_axes": list(BUDGET_AXES), "budget_time_reduction": "maximum_of_all_registered_fixed_times_requires_all_defined",
        "budget_definitions": {BUDGET_AXES[0]: "fixed_probe_RMS(S)/(nu*fixed_probe_RMS(f)); not full-space L2",
            BUDGET_AXES[1]: "norm(spatial_mean(dt_j+nu*j-rho*E))/(nu*rho_background*sqrt(temperature))",
            BUDGET_AXES[2]: "abs(spatial_mean(dt_K-E_dot_j+2*nu*(K-temperature*rho)))/(nu*rho_background*temperature)"},
        "undefined_budget_placeholder": "zero_storage_only; budget_defined_mask_excludes_it_from_every_curve",
        "failed_row_placeholder": "zero_feature_row_excluded_by_hard_valid_mask_but_kept_in_attempt_denominators",
        "hard_valid_scope": "finite_selected_fields_positive_rho_nonnegative_f_not_full_physical_qualification",
        "hash_encoding": "little_endian_float32_complete_native_coefficients_then_nu_temperature",
        "evidence_role": "previously_opened_development_answers_not_blind_confirmation"}
    json.dumps(report, allow_nan=False)
    return matrix, descriptors, report


def sample_from_extraction(matrix, conditions, root_ids, report):
    cases = report["cases"]
    return RegimeAuditSample(matrix, conditions, tuple(root_ids),
        tuple(case["input_answer_sha256"] for case in cases),
        np.asarray([case["hard_valid"] for case in cases], dtype=bool),
        np.asarray([case["numerically_uncertain"] for case in cases], dtype=bool),
        np.asarray([case["budgets"] for case in cases], dtype=float).reshape(-1, 3),
        tuple({"row": case["row"], **case["failure"]} for case in cases if "failure" in case),
        budget_defined_mask=np.asarray([case["budget_defined_mask"] for case in cases], dtype=bool).reshape(-1, 3))
