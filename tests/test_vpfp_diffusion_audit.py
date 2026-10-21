from dataclasses import replace
import json

import numpy as np
import pytest

from pde_agent_verifier_solver.plugins.vpfp.diffusion_audit import (
    AuditFeature,
    AuditInputError,
    AuditSample,
    DiffusionAuditConfig,
    _PairEngine,
    _edges,
    _strata,
    audit_diffusion_distribution,
    audit_selfcheck,
)


NU_EDGES = (0.0, 1.0, 2.0, 3.0, 4.0)
TEMPERATURE_EDGES = (1.0, 2.0, 3.0)
FEATURES = (
    AuditFeature("density", "macro", "density_unit", 1e-6),
    AuditFeature("interaction", "conditional", "energy_unit", 1e-6),
    AuditFeature("constant_coordinate", "coordinate", "coefficient_unit", 1e-4),
    AuditFeature("phase_coupling", "phase", "density_cubed", 1e-6),
)
CFG = DiffusionAuditConfig(permutations=99, bootstrap_repeats=9, projections=4, seed=1209)


def _splits(count=8, seed=1500):
    random = np.random.default_rng(seed)
    result = {}
    for role in ("train", "validation", "calibration", "reference_a", "reference_b", "generated"):
        matrices, conditions = [], []
        for index in range(8):
            nu_bin, temperature_bin = index // 2, index % 2
            interaction = (-1 if nu_bin < 2 else 1) * (-1 if temperature_bin == 0 else 1)
            matrices.append(np.column_stack((
                random.normal(size=count),
                interaction + 0.005*random.normal(size=count),
                np.zeros(count),
                0.5 + 0.005*random.normal(size=count),
            )))
            conditions.append(np.column_stack((random.uniform(nu_bin, nu_bin+1, count), random.uniform(1+temperature_bin, 2+temperature_bin, count))))
        result[role] = AuditSample(np.concatenate(matrices), np.concatenate(conditions), tuple(f"{role}:{i}" for i in range(8*count)))
    return result


def _audit(splits, **kwargs):
    return audit_diffusion_distribution(splits, features=kwargs.pop("features", FEATURES), nu_edges=NU_EDGES, temperature_edges=TEMPERATURE_EDGES, config=kwargs.pop("config", CFG), **kwargs)


def test_full_features_and_constant_coordinate_are_retained():
    splits = _splits()
    generated = splits["generated"].features.copy()
    generated[:, 2] = 1.0
    splits["generated"] = replace(splits["generated"], features=generated)
    report = _audit(splits)
    assert report["status"] == "difference_detected"
    assert report["admitted"] is False
    assert report["preprocessing"]["input_dimension"] == report["preprocessing"]["retained_dimension"] == 4
    constant = report["preprocessing"]["features"][2]
    assert constant["floor_active"] and constant["retained_when_calibration_constant"]
    assert report["comparisons"]["generated"]["groups"]["coordinate"]["max_standardized_mean_deviation"] == pytest.approx(10000)
    assert report["comparisons"]["generated"]["joint_pvalue"] <= 0.05
    json.dumps(report, allow_nan=False)


def test_joint_interaction_reversal_is_detected_despite_matched_marginals():
    splits = _splits()
    generated = splits["generated"].features.copy()
    generated[:, 1] *= -1
    splits["generated"] = replace(splits["generated"], features=generated)
    report = _audit(splits)
    comparison = report["comparisons"]["generated"]
    assert comparison["difference_detected"]
    assert comparison["statistics"]["max_mean_deviation"]["permutation_pvalue"] <= 0.0125
    assert len(comparison["joint_strata"]) == 8
    assert min(abs(row["mean_difference_physical_units"][1]) for row in comparison["joint_strata"]) > 1.9


def test_phase_coordinate_errors_not_erased_by_invariant_projection():
    splits = _splits()
    generated = splits["generated"].features.copy()
    generated[:, 3] *= -1
    splits["generated"] = replace(splits["generated"], features=generated)
    assert _audit(splits)["comparisons"]["generated"]["difference_detected"]


def test_nonrejection_never_claims_equivalence_or_admits():
    splits = _splits()
    # Same feature summaries are allowed for different answer roots; they are
    # not proof of duplicate answers. Identical distributions give tied tests.
    splits["generated"] = replace(splits["generated"], features=splits["reference_a"].features.copy())
    report = _audit(splits)
    assert report["status"] == "insufficient_evidence"
    assert report["admitted"] is False
    assert report["nonrejection_is_not_equivalence"] is True
    assert "not_established" in report["equivalence"]


def test_calibration_only_fit_unchanged_by_candidate_error_and_reproducible():
    splits = _splits()
    first = _audit(splits)
    second = _audit(splits)
    assert first == second
    splits["generated"] = replace(splits["generated"], features=splits["generated"].features + 10)
    changed = _audit(splits)
    assert changed["preprocessing"] == first["preprocessing"]
    assert changed["comparisons"]["reference_null"] == first["comparisons"]["reference_null"]


@pytest.mark.parametrize("role", ["train", "validation", "calibration", "reference_a", "reference_b"])
def test_cross_split_root_replay_is_hard_rejected(role):
    splits = _splits()
    roots = list(splits["generated"].root_ids)
    roots[0] = splits[role].root_ids[0]
    splits["generated"] = replace(splits["generated"], root_ids=roots)
    with pytest.raises(AuditInputError, match="lineage"):
        _audit(splits)


def test_within_split_duplicate_and_hidden_shared_ancestor_rejected():
    splits = _splits()
    roots = list(splits["generated"].root_ids)
    roots[1] = roots[0]
    splits["generated"] = replace(splits["generated"], root_ids=roots)
    with pytest.raises(AuditInputError, match="lineage"):
        _audit(splits)
    splits = _splits()
    rows = [() for _ in splits["generated"].root_ids]
    rows[0] = (splits["train"].root_ids[0],)
    splits["generated"] = replace(splits["generated"], lineage_ids=rows)
    with pytest.raises(AuditInputError, match="lineage"):
        _audit(splits)


def test_duplicate_answer_hash_rejected_even_after_root_relabelling():
    splits = _splits()
    for offset, role in enumerate(splits):
        hashes = [f"{offset*10000 + index:064x}" for index in range(len(splits[role].root_ids))]
        splits[role] = replace(splits[role], answer_sha256=hashes)
    hashes = list(splits["generated"].answer_sha256)
    hashes[0] = splits["train"].answer_sha256[0]
    splits["generated"] = replace(splits["generated"], answer_sha256=hashes)
    with pytest.raises(AuditInputError, match="answer hash"):
        _audit(splits)


@pytest.mark.parametrize("violation", ["nan", "condition_nan", "range", "missing_stratum", "imbalance", "missing_split"])
def test_invalid_numeric_or_sampling_design_fails_closed(violation):
    splits = _splits()
    sample = splits["generated"]
    if violation == "nan":
        values = sample.features.copy()
        values[0, 0] = np.nan
        splits["generated"] = replace(sample, features=values)
    elif violation in ("condition_nan", "range", "missing_stratum", "imbalance"):
        conditions = sample.conditions.copy()
        if violation == "condition_nan":
            conditions[0, 0] = np.nan
        elif violation == "range":
            conditions[0, 0] = 5
        elif violation == "missing_stratum":
            conditions[:8, 1] = 2.5
        else:
            conditions[0, 1] = 2.5
        splits["generated"] = replace(sample, conditions=conditions)
    else:
        del splits["train"]
    with pytest.raises(AuditInputError):
        _audit(splits)


def test_bin_permutations_preserve_counts_and_do_not_cross_conditions():
    random = np.random.default_rng(45)
    left = random.normal(size=(32, 2))
    right = random.normal(size=(32, 2))
    bins = np.repeat(np.arange(8), 4)
    engine = _PairEngine(left, right, bins, bins, 1.0, CFG, 1)
    joined_bins = np.concatenate((bins, bins))
    il, ir = engine.permute(random)
    for index in range(8):
        assert len(il[index]) == len(ir[index]) == 4
        assert set(il[index]).isdisjoint(ir[index])
        assert set(il[index]) | set(ir[index]) == set(engine.bin_indices[index])
        assert np.all(joined_bins[il[index]] == index)
        assert np.all(joined_bins[ir[index]] == index)


def test_float32_global_endpoints_keep_their_intended_edge_strata():
    edges = _edges(np.linspace(.02, .08, 5), np.linspace(.8, 1.2, 3))
    conditions = np.array([[.02, .8], [.08, 1.2]], dtype=np.float32)
    assert _strata(conditions, edges).tolist() == [0, 7]
    # The same numerical overshoot is not authorized if its declared dtype is
    # float64: it is then not the exact dtype encoding of the physical bound.
    with pytest.raises(AuditInputError, match="outside"):
        _strata(conditions.astype(np.float64), edges)


@pytest.mark.parametrize("axis,endpoint,direction", [(0, .02, -np.inf), (0, .08, np.inf), (1, .8, -np.inf), (1, 1.2, np.inf)])
def test_next_float_beyond_encoded_endpoint_is_still_rejected(axis, endpoint, direction):
    edges = _edges(np.linspace(.02, .08, 5), np.linspace(.8, 1.2, 3))
    conditions = np.array([[.05, 1.]], dtype=np.float32)
    conditions[0, axis] = np.nextafter(np.float32(endpoint), np.float32(direction))
    with pytest.raises(AuditInputError, match="outside"):
        _strata(conditions, edges)


def test_public_audit_preserves_condition_dtype_for_endpoint_validation():
    splits = _splits()
    converted = {}
    for role, sample in splits.items():
        conditions = np.asarray(sample.conditions, dtype=np.float32).copy()
        conditions[:, 0] = .02 + conditions[:, 0]*.015
        conditions[:, 1] = .8 + (conditions[:, 1]-1.)*.2
        conditions[0] = np.array([.02, .8], dtype=np.float32)
        conditions[-1] = np.array([.08, 1.2], dtype=np.float32)
        converted[role] = replace(sample, conditions=conditions)
    report = audit_diffusion_distribution(converted, features=FEATURES,
        nu_edges=np.linspace(.02, .08, 5), temperature_edges=np.linspace(.8, 1.2, 3),
        config=replace(CFG, permutations=3, bootstrap_repeats=1))
    assert report["sampling_design"]["balanced_within_each_split"]


def test_selfcheck_reports_bad_controls_without_physical_admission():
    cfg = replace(CFG, permutations=79, bootstrap_repeats=2, projections=2)
    report = audit_selfcheck(config=cfg, trials=2, cases_per_stratum=8)
    assert report["population"] == "declared_synthetic_feature_distribution"
    assert report["physical_model_validation"] is False
    assert report["admitted"] is False
    assert report["controls"]["training_replay"]["detected"]
    for name in ("interaction_reversal", "off_pca_coordinate", "phase_or_coordinate_error"):
        assert report["controls"][name]["rate"] == 1
    for name, row in report["controls"].items():
        if name != "training_replay":
            lo, hi = row["wilson_95_interval"]
            assert 0 <= lo <= row["rate"] <= hi <= 1
    json.dumps(report, allow_nan=False)


def test_small_strata_can_be_underpowered_even_for_large_coordinate_errors():
    splits = _splits(count=2)
    generated = splits["generated"].features.copy()
    generated[:, 2] = 1.0
    splits["generated"] = replace(splits["generated"], features=generated)
    report = _audit(splits)
    assert report["status"] == "insufficient_evidence"
    assert report["admitted"] is False
    assert report["comparisons"]["generated"]["statistics"]["max_mean_deviation"]["effect"] > 999


def test_empirical_selfcheck_has_disjoint_fit_and_explicit_conditional_scope():
    bank = _splits(count=4)["calibration"]
    cfg = replace(CFG, permutations=3, bootstrap_repeats=1, projections=2)
    report = audit_selfcheck(config=cfg, features=FEATURES, calibration_bank=bank, nu_edges=NU_EDGES, temperature_edges=TEMPERATURE_EDGES, trials=1, cases_per_stratum=2)
    assert report["population"] == "conditional_on_fixed_empirical_bank"
    assert report["preprocessing_fit_and_evaluation_donors_disjoint"]
    assert report["evaluation_donors_per_stratum"] == [2]*8
    assert report["bank_resampling_with_replacement"]
    assert report["independent_pde_population_trials"] is False
    assert "not asserted physical" in report["control_semantics"]["phase_or_coordinate_error"]


def test_feature_unit_change_with_matching_floor_preserves_statistics():
    splits = _splits()
    before = _audit(splits)
    converted = {}
    for role, sample in splits.items():
        values = sample.features.copy()
        values[:, 0] *= 1000
        converted[role] = replace(sample, features=values)
    features = (replace(FEATURES[0], unit="milli_density_unit", scale_floor=FEATURES[0].scale_floor*1000), *FEATURES[1:])
    after = _audit(converted, features=features)
    for name, row in before["comparisons"]["generated"]["statistics"].items():
        assert row["effect"] == pytest.approx(after["comparisons"]["generated"]["statistics"][name]["effect"])
        assert row["permutation_pvalue"] == after["comparisons"]["generated"]["statistics"][name]["permutation_pvalue"]
