"""Synthetic tiny feature fixtures validate screening logic, not plasma models."""

from copy import deepcopy
import json

import numpy as np
import pytest

from pde_agent_verifier_solver.plugins.vpfp.direction_scorecard import (
    aggregate_vpfp_direction_scorecards, build_vpfp_direction_scorecard,
)


def fixture():
    physical_names = [f"kinetic_energy.time{index}.{statistic}" for index in range(3)
        for statistic in ("mean", "std", "spectrum_low", "spectrum_middle", "spectrum_high")]
    derivative_names = [f"{quantity}.probe{index}.{component}"
        for quantity in ("f", "rho", "kinetic_energy", "momentum_x", "momentum_y")
        for index in range(3) for component in ("value", "dt", "dx", "dy")]
    def schema(names):
        return [{"name": name, "group": "fixture", "unit": "fixture", "scale_floor": 1.e-6}
                for name in names]
    def sample(value, count):
        return {"physical": np.full((count, len(physical_names)), value),
                "derivative": np.full((count, len(derivative_names)), value)}
    def audit(generated, reference=.05):
        return {"comparisons": {role: {"statistics": {metric: {"effect": value}
            for metric in ("sliced_wasserstein_max", "multiband_mmd_max")}}
            for role, value in (("generated", generated), ("reference_null", reference))}}
    return {"regime": "weak", "seed": 1, "objective": "base", "baseline_arm": "full",
        "samples": {"reference_a": sample(1., 8), "reference_b": sample(1.02, 8),
                    "full": sample(1.8, 4), "value": sample(1.1, 4)},
        "physical_schema": schema(physical_names), "derivative_schema": schema(derivative_names),
        "joint_reports": {arm: {kind: audit(value) for kind in ("physical", "derivative")}
                          for arm, value in (("full", 1.), ("value", .6))},
        "physical_time_fractions": (0., .25, 1.),
        "derivative_time_fractions": (0., .25, 1.), "macro_probe_count": 1, "phase_probe_count": 1}


def test_improvement_uses_both_reference_sides_and_only_recommends_development_direction():
    args = fixture()
    original = deepcopy(args["samples"])
    report = build_vpfp_direction_scorecard(**args)
    candidate = report["arms"]["value"]
    assert candidate["direction_supported"]
    assert not report["recommend_scale_up"] and not report["recommend_fresh_confirmation"]
    assert not report["admitted"]
    assert not report["arms"]["full"]["direction_supported"]
    metric = candidate["metrics"]["physical.kinetic_energy.time2.std"]
    assert metric["reference_center"] == pytest.approx(1.01)
    assert metric["reference_mean_gap"] == pytest.approx(.02)
    assert metric["signed_mean_gap"] == pytest.approx(.09)
    assert metric["baseline_absolute_mean_gap"] == pytest.approx(.79)
    assert metric["time_fraction"] == 1.
    assert report["sample_counts"] == {"reference_a": 8, "reference_b": 8, "full": 4, "value": 4}
    json.dumps(report, allow_nan=False)
    for role in original:
        for kind in original[role]:
            np.testing.assert_array_equal(args["samples"][role][kind], original[role][kind])


def test_smaller_rms_can_be_a_regression_not_an_improvement():
    args = fixture()
    for kind in ("physical", "derivative"):
        args["samples"]["full"][kind][:] = 1.1
        args["samples"]["value"][kind][:] = .65
    arm = build_vpfp_direction_scorecard(**args)["arms"]["value"]
    assert not arm["direction_supported"]
    assert not arm["mean_gap_nonregression_pass"]
    assert not arm["momentum_collapse_veto"]
    assert arm["metrics"]["derivative.kinetic_energy.dt.late.rms"]["signed_mean_gap"] < 0


def test_momentum_collapse_veto_even_when_baseline_is_worse():
    args = fixture()
    for index, descriptor in enumerate(args["derivative_schema"]):
        if descriptor["name"].startswith("momentum_x.") and descriptor["name"].endswith(".value"):
            args["samples"]["value"]["derivative"][:, index] = .2
            args["samples"]["full"]["derivative"][:, index] = 4.
    arm = build_vpfp_direction_scorecard(**args)["arms"]["value"]
    assert arm["mean_gap_nonregression_pass"]
    assert arm["momentum_collapse_veto"] and not arm["direction_supported"]
    assert "derivative.momentum_x.value.late.rms" in arm["momentum_veto_metrics"]


def test_zero_reference_variance_has_physical_floor_and_no_division_by_zero():
    args = fixture()
    for role in args["samples"]:
        for kind in args["samples"][role]:
            args["samples"][role][kind][:] = 0.
    report = build_vpfp_direction_scorecard(**args)
    metric = report["arms"]["value"]["metrics"]["derivative.momentum_x.value.all.rms"]
    assert metric["reference_scale"] == 1.e-6
    assert metric["normalized_signed_mean_gap"] == 0.
    assert not report["arms"]["value"]["momentum_collapse_veto"]
    json.dumps(report, allow_nan=False)


def test_early_and_late_probe_statistics_are_separate_and_time_major():
    args = fixture()
    for index, descriptor in enumerate(args["derivative_schema"]):
        if descriptor["name"] == "f.probe2.dt":
            args["samples"]["value"]["derivative"][:, index] = 9.
    metrics = build_vpfp_direction_scorecard(**args)["arms"]["value"]["metrics"]
    assert metrics["derivative.f.dt.early.rms"]["candidate_mean"] == pytest.approx(1.1)
    assert metrics["derivative.f.dt.late.rms"]["candidate_mean"] == 9.


@pytest.mark.parametrize("kind", ["physical", "derivative"])
def test_both_joint_distributions_must_improve_by_twenty_percent(kind):
    args = fixture()
    args["joint_reports"]["value"][kind]["comparisons"]["generated"]["statistics"]["sliced_wasserstein_max"]["effect"] = .81
    arm = build_vpfp_direction_scorecard(**args)["arms"]["value"]
    assert not arm["direction_supported"] and not arm["joint_sw_reduction_pass"]


def test_baseline_at_reference_noise_cannot_establish_directional_reduction():
    args = fixture()
    for arm, value in (("full", .04), ("value", .01)):
        args["joint_reports"][arm]["physical"]["comparisons"]["generated"]["statistics"]["sliced_wasserstein_max"]["effect"] = value
    assert not build_vpfp_direction_scorecard(**args)["arms"]["value"]["direction_supported"]


def test_reference_noise_gap_is_included_in_nonregression_tolerance():
    args = fixture()
    for kind in ("physical", "derivative"):
        args["samples"]["reference_a"][kind][:] = .8
        args["samples"]["reference_b"][kind][:] = 1.2
        args["samples"]["full"][kind][:] = 1.
        args["samples"]["value"][kind][:] = 1.35
    report = build_vpfp_direction_scorecard(**args)
    assert report["arms"]["value"]["mean_gap_nonregression_pass"]


@pytest.mark.parametrize("corruption", ["nonfinite", "missing_field", "missing_role", "rows", "negative_effect", "null_mismatch", "duplicate_schema"])
def test_invalid_inputs_fail_closed(corruption):
    args = fixture()
    if corruption == "nonfinite":
        args["samples"]["value"]["derivative"][0, 0] = np.nan
    elif corruption == "missing_field":
        args["physical_schema"][0]["name"] = "unrecognized"
    elif corruption == "missing_role":
        del args["samples"]["reference_b"]
    elif corruption == "rows":
        args["samples"]["value"]["derivative"] = args["samples"]["value"]["derivative"][:2]
    elif corruption == "negative_effect":
        args["joint_reports"]["value"]["physical"]["comparisons"]["generated"]["statistics"]["sliced_wasserstein_max"]["effect"] = -1.
    elif corruption == "null_mismatch":
        args["joint_reports"]["value"]["physical"]["comparisons"]["reference_null"]["statistics"]["sliced_wasserstein_max"]["effect"] = .06
    else:
        args["physical_schema"][0]["name"] = args["physical_schema"][1]["name"]
    with pytest.raises(ValueError):
        build_vpfp_direction_scorecard(**args)


def replicas():
    result = []
    for regime in ("weak", "strong"):
        for seed in (1, 2):
            args = fixture()
            args.update(regime=regime, seed=seed)
            result.append(build_vpfp_direction_scorecard(**args))
    return result


def test_two_seeds_both_regimes_only_recommend_fresh_confirmation():
    result = aggregate_vpfp_direction_scorecards(replicas())
    assert result["candidates"]["value"]["recommend_fresh_confirmation"]
    assert not result["recommend_scale_up"] and not result["admitted"]
    json.dumps(result, allow_nan=False)


def test_missing_or_failed_strong_replica_cannot_be_masked_by_weak_success():
    reports = replicas()
    assert not aggregate_vpfp_direction_scorecards(reports[:3])["candidates"]["value"]["recommend_fresh_confirmation"]
    reports[-1]["arms"]["value"]["direction_supported"] = False
    result = aggregate_vpfp_direction_scorecards(reports)["candidates"]["value"]
    assert result["by_regime"]["weak"]["direction_supported"]
    assert not result["recommend_fresh_confirmation"]
    assert result["by_regime"]["strong"]["failed_seeds"] == [2]


@pytest.mark.parametrize("change", ["duplicate_seed", "objective", "baseline", "time_plan", "candidate_name"])
def test_aggregation_rejects_unmatched_or_duplicate_replicates(change):
    reports = replicas()
    if change == "duplicate_seed":
        reports[-1]["seed"] = 1
    elif change == "objective":
        reports[-1]["objective"] = "different_matching_objective"
    elif change == "baseline":
        reports[-1]["baseline_arm"] = "other"
    elif change == "time_plan":
        reports[-1]["feature_plan"]["derivative_time_fractions"][1] = .1
    else:
        reports[-1]["arms"]["other"] = reports[-1]["arms"].pop("value")
    with pytest.raises(ValueError):
        aggregate_vpfp_direction_scorecards(reports)
