"""Tiny metric/decoder checks, not physical calibration or real-asset audit."""
from dataclasses import replace
from hashlib import sha256
import importlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pde_agent_verifier_solver.plugins.vpfp import regime_fidelity_coverage as audit


@pytest.fixture
def feature_bank():
    rng = np.random.default_rng(443)
    conditions = np.repeat(np.asarray([(nu, temp) for nu in (.125, .375, .625, .875)
        for temp in (.9, 1.1)], dtype=np.float32), 12, axis=0)
    samples = {}
    names = list(audit._MACRO_NAMES)+list(audit._PROBE_NAMES)
    descriptors = [{"name": "time0."+name, "unit": "dimensionless"} for name in names]
    for role in ("calibration", "reference_a", "reference_b", "generated"):
        n = len(conditions)
        samples[role] = audit.RegimeAuditSample(rng.normal(size=(n, len(names))), conditions,
            tuple(f"{role}:{i}" for i in range(n)),
            tuple(sha256(f"{role}:{i}".encode()).hexdigest() for i in range(n)),
            np.ones(n, dtype=bool), np.ones(n, dtype=bool), np.tile([.1, 1., 10.], (n, 1)))
    return samples, descriptors


def run_bank(bank, **kwargs):
    samples, descriptors = bank
    return audit.audit_regime_fidelity_coverage(samples, descriptors=descriptors,
        nu_edges=np.linspace(0, 1, 5), temperature_edges=(.8, 1., 1.2), bootstrap_repeats=2, **kwargs)


def test_reference_only_fit_and_no_admission(feature_bank):
    report = run_bank(feature_bank)
    assert not report["admitted"] and not report["physical_regime_qualified"]
    assert report["geometry"]["standard_deviation_division"] is False
    assert report["morphology_occupancy"]["reference_frequency_weighting"] is False
    assert report["trajectory_bootstrap"]["calibration_axes_refitted"] is False
    assert report["trajectory_bootstrap"]["neighborhoods_refitted_to_resampled_reference"] is False
    assert report["trajectory_bootstrap"]["support_intervals_condition_on_frozen_reference_bank_and_geometry"]
    assert report["sample_count_curves"]["rows"][-1]["available"] is False
    assert report["sample_count_curves"]["rows"][0]["global_frechet_descriptive"]["available"]
    assert report["sample_count_curves"]["rows"][0]["strata"][0]["frechet"]["available"]
    assert not report["sample_count_curves"]["finite_sample_model_dependent_Frechet_bias_removed"]
    assert report["confirmation_512_completed"] is False
    samples, descriptors = feature_bank
    changed = dict(samples)
    changed["generated"] = replace(samples["generated"], features=samples["generated"].features*100)
    second = run_bank((changed, descriptors))
    assert second["geometry"] == report["geometry"]
    json.dumps(report, allow_nan=False)


def test_failed_attempts_and_source_budget_denominators(feature_bank):
    samples, descriptors = feature_bank
    valid = samples["generated"].valid_mask.copy(); valid[0] = False
    budgets = samples["generated"].budgets.copy(); budgets[1, 0] = 1.e-7; budgets[2, 0] = 101.
    samples["generated"] = replace(samples["generated"], valid_mask=valid, budgets=budgets,
        failures=({"row": 0, "message": "fixture decoder failure"},))
    report = run_bank((samples, descriptors))
    assert report["cohorts"]["generated"]["attempted_count"] == 96
    assert report["cohorts"]["generated"]["failed_count"] == 1
    assert report["comparisons"]["generated"]["joint_strata"][0]["neighbors"]["5"]["attempted_count"] == 12
    curve = report["budget_curves"][audit.BUDGET_AXES[0]]
    assert curve["under_positive_grid_count"] == curve["overflow_count"] == 1
    assert curve["grid"] == list(audit.BUDGET_GRID)
    assert np.all(np.diff([row["retained_count"] for row in curve["curve"]]) >= 0)


def test_joint_budget_box_does_not_hide_the_other_injection_axes(feature_bank):
    samples, descriptors = feature_bank
    valid = samples["generated"].valid_mask.copy(); valid[0] = False
    defined = np.ones((96, 3), dtype=bool); defined[1, 2] = False
    samples["generated"] = replace(samples["generated"],valid_mask=valid,budget_defined_mask=defined)
    report = run_bank((samples,descriptors))
    joint = report["joint_budget_curve"]
    assert joint["axes"] == list(audit.BUDGET_AXES)
    assert not joint["candidate_fitted_weights"] and not joint["automatic_threshold_selected"]
    for point in joint["curve"]:
        budget = point["maximum_dimensionless_budget_on_each_axis"]
        expected = 94 if budget >= 10. else 0
        assert point["retained_count"] == point["retained_numerically_uncertain_count"] == expected
        assert point["retained_fraction_all_attempts"] == expected/96
        assert point["undefined_any_budget_count_all_attempts"] == 1
        assert point["retained_is_not_physics_accepted"]
        for axis in audit.BUDGET_AXES:
            single = next(p for p in report["budget_curves"][axis]["curve"] if p["maximum_dimensionless_budget"] == budget)
            assert point["retained_count"] <= single["retained_count"]


def test_cross_reference_hash_refused_but_generated_collapse_counted(feature_bank):
    samples, descriptors = feature_bank
    duplicate = (samples["generated"].answer_sha256[0],)*96
    samples["generated"] = replace(samples["generated"], answer_sha256=duplicate)
    assert run_bank((samples, descriptors))["lineage"]["generated_repeated_answer_excess"] == 95
    samples["generated"] = replace(samples["generated"], answer_sha256=(samples["reference_a"].answer_sha256[0],)*96)
    with pytest.raises(ValueError, match="duplicates a frozen"):
        run_bank((samples, descriptors))


def test_feature_controls_are_not_physical_calibration(feature_bank):
    samples, descriptors = feature_bank
    report = run_bank((samples, descriptors))
    control = report["metric_calibration_controls"]
    assert not control["physical_calibration"]
    assert "same_bank_permutation" in control["results"]
    assert "independent_reference_partition" not in control["results"]


def test_fine_structure_bins_are_calibration_only_and_not_frequency_scores(feature_bank):
    samples, descriptors = feature_bank
    report = run_bank(feature_bank)
    fine = report["fine_structure_coverage"]
    assert not fine["reference_frequency_weighting"] and not fine["joint_shape_coverage_claimed"]
    assert set(fine["axes"]) == {"K_spectrum_high", "dv_f_fixed_probe_rms"}
    changed = dict(samples)
    changed["generated"] = replace(samples["generated"], features=samples["generated"].features*100.)
    other = run_bank((changed,descriptors))["fine_structure_coverage"]
    for axis, row in fine["axes"].items():
        assert row["available"] and len(row["calibration_quartile_boundaries"]) == 3
        assert row["calibration_quartile_boundaries"] == other["axes"][axis]["calibration_quartile_boundaries"]
        assert row["boundaries_are_not_physical_acceptance_thresholds"]
        assert all(r["new_bins_are_not_physics_invalid"] for r in row["rows"])


@pytest.mark.parametrize("role", ["reference_b", "generated"])
def test_all_failed_cohort_preserves_report_and_attempts(feature_bank, role):
    samples, descriptors = feature_bank
    samples[role] = replace(samples[role], valid_mask=np.zeros(96, dtype=bool))
    report = run_bank((samples, descriptors))
    assert report["cohorts"][role]["attempted_count"] == report["cohorts"][role]["failed_count"] == 96
    if role == "reference_b":
        assert not report["metric_calibration_controls"]["available"]
        assert report["metric_calibration_controls"]["results"] == {}
    else:
        intervals = report["trajectory_bootstrap"]["rows"][0]["percentile_2_5_97_5"]
        assert intervals["precision_all_attempts"] == [0., 0.]
        assert intervals["reference_coverage"] == [0., 0.]
        assert intervals["squared_frechet"] is None


def test_frechet_rank_deficient_and_known_shift():
    x = np.tile(np.arange(6, dtype=float)[:, None], (1, 16))
    assert audit._frechet(x, x)["squared_distance"] < 1.e-10
    np.testing.assert_allclose(audit._frechet(x, x+2)["squared_distance"], 64., atol=1.e-10)
    assert not audit._frechet(x[:1], x)["available"]


def test_support_duplicate_radius_and_all_attempt_denominator():
    ref = np.zeros((7, 2)); gen = np.array([[0., 0.], [1., 1.]])
    result = audit._support(ref, gen, k=5, attempted_count=4)
    assert result["zero_reference_radius_count"] == 7
    assert result["precision_all_attempts"] == .25
    assert result["reference_coverage"] == 1.


def test_collision_budget_units_and_undefined_denominator():
    probes, macros = np.zeros((3, 9)), np.zeros((3, len(audit._MACRO_NAMES)))
    probes[:, 0], probes[:, 1] = 2., 6.
    macros[:, 10:12] = [8., 10.]
    values, defined, ledger = audit._collision_budgets(probes, macros, nu=.5, horizon=4., minimum_normal_f_rms=1.e-30)
    np.testing.assert_allclose(values, [1.5, 4., 5.])
    assert defined.all() and ledger["source_denominator_is_nu_times_f_RMS_not_collision_operator_RMS"]
    probes[1, 0] = 0.
    values, defined, ledger = audit._collision_budgets(probes, macros, nu=.5, horizon=4., minimum_normal_f_rms=1.e-30)
    assert defined.tolist() == [False, True, True]
    assert values[0] == 0. and ledger["collision_normalized_budget_by_time"][1][0] is None
    _, defined, _ = audit._collision_budgets(probes, macros, nu=0., horizon=4., minimum_normal_f_rms=1.e-30)
    assert not defined.any()


def test_undefined_budget_never_retained_and_uncertain_counts_kept(feature_bank):
    samples, descriptors = feature_bank
    defined = np.ones((96, 3), dtype=bool); defined[0, 0] = False
    budgets = np.zeros((96, 3))
    samples["generated"] = replace(samples["generated"], budgets=budgets, budget_defined_mask=defined)
    report = run_bank((samples, descriptors))
    for point in report["budget_curves"][audit.BUDGET_AXES[0]]["curve"]:
        assert point["retained_count"] == point["retained_numerically_uncertain_count"] == 95
        assert point["undefined_budget_count_all_attempts"] == 1


def test_full_R2_highest_Hermite_moment_and_temperature_units():
    modes = [(a, total-a) for total in range(15) for a in range(total+1)]
    c = torch.zeros((2, len(modes)), dtype=torch.float64)
    c[:, modes.index((14, 0))] = 1.
    jet = {"coefficients": c, "coefficient_derivatives": torch.zeros((3, *c.shape)),
        "rho": torch.tensor([2., 3.]), "rho_derivatives": torch.zeros((3, 2)), "electric": torch.zeros((2, 2))}
    result = audit._macro_values(jet, modes, 1.3, 0.)
    np.testing.assert_allclose(result["K"], np.array([2., 3.])*15*1.3, rtol=1.e-14)
    np.testing.assert_array_equal(result["jx"], 0.)
    np.testing.assert_allclose(result["thermal_anisotropy"], 28., rtol=1.e-14)


@pytest.fixture
def tiny_decoder():
    api = audit._legacy_api()
    hermite = importlib.import_module("agent_verifier.vpfp.continuous_hermite")
    cfg = replace(hermite.VPFPContinuousHermiteConfig(), latent_dim=1,
        micro_latent_dim=2, context_dim=4, hidden_width=4, hidden_layers=1,
        rho_max_mode_x=1, rho_max_mode_y=1, rho_positivity_gauge="phase_invariant_group_l1_v2",
        spatial_feature_modes=1, time_feature_modes=1, hermite_total_degree=1,
        coefficient_width=4, coefficient_blocks=1, structured_spectral_latent=True,
        structured_kinetic_global_dim=2, structured_local_channels=3,
        structured_local_spatial_mode=1, structured_time_basis="piecewise_cubic_hermite_v1",
        structured_time_control_count=3, structured_direct_coefficients=True,
        structured_residual_scale=0., time_domain=(0., 1.), dtype="float32")
    cfg = replace(cfg, latent_dim=cfg.structured_latent_layout()["total_latent_dim"])
    model = api.build_deterministic_direct_model(cfg, device="cpu")
    raw = np.zeros((1, cfg.latent_dim), dtype=np.float32)
    return SimpleNamespace(model=model, raw=raw, conditions=np.array([[.04, 1.]], dtype=np.float32),
        plan=audit.VPFPRegimeFeaturePlan(time_fractions=(0., .5, 1.), grid_sizes=(3, 5),
            phase_probes_per_time=2, independent_source_probe_count=2, chunk_size=32))


def test_tiny_native_maxwellian_full_extraction(tiny_decoder):
    fixture = tiny_decoder
    fixture.model.train()
    before = fixture.raw.copy()
    matrix, descriptors, report = audit.extract_vpfp_regime_features(fixture.model,
        fixture.raw, fixture.conditions, plan=fixture.plan)
    case = report["cases"][0]
    assert case["hard_valid"], case
    assert case["numerically_uncertain"]
    assert fixture.model.training
    np.testing.assert_array_equal(before, fixture.raw)
    assert case["source_probes"]["independent_source_check"]["execution_passed"], case
    assert matrix.shape == (1, len(descriptors))
    positions = {d["name"]: i for i, d in enumerate(descriptors)}
    assert abs(matrix[0, positions["time0.thermal_energy_mean"]]-1.) < 1.e-6
    assert matrix[0, positions["time0.j_coherent_norm"]] == 0.
    assert matrix[0, positions["time0.source_fixed_probe_rms"]] < 1.e-6
    assert report["full_velocity_source_quadrature_completed"] is False
    json.dumps(report, allow_nan=False)


def test_nonfinite_answer_preserves_failure_and_identity(tiny_decoder):
    fixture = tiny_decoder
    raw = fixture.raw.copy(); raw[0, 0] = np.nan
    matrix, _, report = audit.extract_vpfp_regime_features(fixture.model, raw, fixture.conditions, plan=fixture.plan)
    assert not report["cases"][0]["hard_valid"]
    assert report["cases"][0]["failure"]["type"] == "ValueError"
    assert report["cases"][0]["input_answer_sha256"]
    np.testing.assert_array_equal(matrix, 0.)


def test_moving_frame_z_only_is_refused(tiny_decoder):
    with pytest.raises(ValueError, match="moving-frame"):
        audit.extract_vpfp_regime_features(tiny_decoder.model, {"z": tiny_decoder.raw}, tiny_decoder.conditions)
