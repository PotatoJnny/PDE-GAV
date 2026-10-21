"""Synthetic algebra/contract tests, not real-data calibration or validation."""

import copy
import json
import math

import numpy as np
import pytest

from pde_agent_verifier_solver.plugins.plasma_transport import (
    CALIBRATION_FAMILIES, FAILURE_POLICY, TRANSPORT_FIELDS, TRANSPORT_PROTOCOL_ID,
    TRANSPORT_WEIGHTS, VPFP_TRANSPORT_FIELDS, TransportCalibration,
    canonicalize_1d, canonicalize_2d, fit_calibration,
    load_transport_calibration, score_transport,
)
from pde_agent_verifier_solver.plugins.vp_1d1v.macro_matching import VP1DTransportMacroMatcher
from pde_agent_verifier_solver.plugins.vpfp.macro_matching import VPFPTransportMacroMatcher
from pde_agent_verifier_solver.sdk import (
    AnswerQuery, AnswerView, Case, CaseSource, CaseVisibility, MacroTarget,
)


UNITS = {"density": 1.0, "velocity": 1.0, "electric_field": 1.0}


def _raw1(count=2):
    # n=1, u=1, P=1, qh=1, E=1: every canonical reference norm is one.
    return {name: np.full(count, value) for name, value in
            {"M0": 1.0, "M1": 1.0, "M2": 2.0, "M3": 6.0, "E": 1.0}.items()}


def _raw2(count=2):
    result = {name: np.zeros(count) for name in VPFP_TRANSPORT_FIELDS}
    for name in ("rho", "electric_x", "momentum_x", "pressure_xx", "heat_flux_x"):
        result[name][:] = 1.0
    return result


def _rows(values=None):
    values = canonicalize_1d(_raw1()) if values is None else values
    return [{"family": family, "stratum": "test", "root_id": family, "values": values}
            for family in CALIBRATION_FAMILIES]


def _calibration(rows=None, units=None):
    return fit_calibration(
        _rows() if rows is None else rows, units=UNITS if units is None else units,
        provenance={"synthetic_unit_test": "not a production calibration asset"},
    )


def _view(fields):
    n = len(next(iter(fields.values())))
    points = tuple({"t": float(index)} for index in range(n))
    query = AnswerQuery.at_points(fields=tuple(fields), points=points)
    return query, AnswerView(query.kind, {name: tuple(values) for name, values in fields.items()},
                             points=points, shape=(n,))


def _case():
    return Case("test", CaseSource.GENERATED, CaseVisibility.DEVELOPMENT, {})


def test_fixed_weights_and_reference_rms_calibration():
    asset = _calibration()
    assert TRANSPORT_WEIGHTS == {"density": .125, "electric_field": .125, "particle_flux": .25,
                                  "pressure": .25, "heat_flux": .25}
    assert asset.to_dict()["floors"] == {name: 1.0 for name in TRANSPORT_FIELDS}
    assert asset.objective_identity["failure_policy"] == FAILURE_POLICY
    assert asset.contract_id == f"{TRANSPORT_PROTOCOL_ID}:{asset.objective_hash}"


def test_calibration_equal_weights_families_and_strata_not_sample_frequency():
    rows = _rows()
    # Strong has unequal stratum sizes; each stratum gets equal family weight.
    values3 = {key: array * 3 for key, array in canonicalize_1d(_raw1()).items()}
    rows.extend({"family": "strong", "stratum": "rare", "root_id": f"rare-{i}", "values": values3}
                for i in range(4))
    asset = _calibration(rows)
    # strong: (1 + 9)/2 = 5; weak=Artemis=1; overall=(5+1+1)/3.
    assert asset.to_dict()["floors"]["density"] == pytest.approx(math.sqrt(7 / 3))


def test_duplicate_roots_and_missing_families_fail_calibration():
    rows = _rows()
    with pytest.raises(ValueError, match="unique"):
        _calibration(rows + [rows[0]])
    with pytest.raises(ValueError, match="strong, weak, and Artemis"):
        _calibration(rows[:-1])


def test_zero_calibration_channel_never_gets_epsilon_fallback():
    values = canonicalize_1d(_raw1())
    values["heat_flux"][:] = 0
    with pytest.raises(ValueError, match="no epsilon fallback"):
        _calibration(_rows(values))


def test_asset_roundtrip_hash_and_immutable_returned_copies(tmp_path):
    asset = _calibration()
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(asset.to_dict()))
    loaded = load_transport_calibration(path)
    assert loaded.objective_hash == asset.objective_hash
    floors = loaded.physical_floors
    floors["density"] = 900
    assert loaded.physical_floors["density"] == 1
    identity = loaded.objective_identity
    identity["floors"]["density"] = 500
    assert loaded.objective_identity["floors"]["density"] == 1


@pytest.mark.parametrize("change", ["version", "floors", "hash", "provenance", "counts"])
def test_bad_asset_rejected(change):
    payload = _calibration().to_dict()
    if change == "version":
        payload["protocol_id"] = "old"
    elif change == "floors":
        payload["floors"]["heat_flux"] = 1e-30
        payload.pop("objective_hash")  # Even without the hash it must obey the declared rule.
    elif change == "hash":
        payload["objective_hash"] = "wrong"
    elif change == "provenance":
        payload["calibration"]["source_role"] = "candidate_scores"
    else:
        payload["calibration"]["families"]["strong"]["case_count"] = 10
    with pytest.raises(ValueError):
        TransportCalibration.from_dict(payload)


def test_global_rms_not_pointwise_relative_error():
    reference = canonicalize_2d(_raw2())
    prediction = copy.deepcopy(reference)
    reference["particle_flux"][:, 0] = (0, 10)
    prediction["particle_flux"][:, 0] = (2, 10)
    metrics = score_transport(prediction, reference, _calibration())
    expected = math.sqrt(2 / 51)
    assert metrics["particle_flux_normalized_rms"] == pytest.approx(expected)
    assert metrics[TRANSPORT_PROTOCOL_ID] == pytest.approx(.25 * expected)
    assert metrics["particle_flux_reference_rms"] == pytest.approx(math.sqrt(50))


def test_near_zero_flux_uses_frozen_floor_without_pointwise_division():
    reference = canonicalize_2d(_raw2())
    prediction = copy.deepcopy(reference)
    reference["heat_flux"][:] = 0
    prediction["heat_flux"][:] = (3, 4)
    metrics = score_transport(prediction, reference, _calibration())
    assert metrics["heat_flux_normalized_rms"] == 5
    assert metrics["heat_flux_floor_fraction"] == 1
    assert metrics[TRANSPORT_PROTOCOL_ID] == 1.25


def test_full_pressure_has_offdiagonal_multiplicity_two():
    raw = _raw2()
    reference = canonicalize_2d(raw)
    raw["pressure_xy"][:] = 2
    metrics = score_transport(canonicalize_2d(raw), reference, _calibration())
    assert metrics["pressure_normalized_rms"] == pytest.approx(math.sqrt(8 / 2))


def test_rotation_invariance_for_vector_and_full_tensor_groups():
    raw = _raw2(3)
    raw["pressure_xx"][:] = (1, 2, 4)
    raw["pressure_xy"][:] = (.1, -.4, .2)
    raw["pressure_yy"][:] = (2, 3, 1)
    reference = canonicalize_2d(raw)
    prediction = copy.deepcopy(reference)
    prediction["electric_field"] += [.2, -.7]
    prediction["particle_flux"] += [.3, .9]
    prediction["heat_flux"] += [-.2, .6]
    prediction["pressure"] += [.5, -.1, -.1, -.2]
    theta = .739
    rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])

    def rotate(values):
        result = copy.deepcopy(values)
        for name in ("electric_field", "particle_flux", "heat_flux"):
            result[name] = values[name] @ rotation.T
        pressure = np.einsum("ij,njk,lk->nil", rotation, values["pressure"].reshape(-1, 2, 2), rotation)
        result["pressure"] = pressure.reshape(-1, 4)
        return result

    original = score_transport(prediction, reference, _calibration())
    rotated = score_transport(rotate(prediction), rotate(reference), _calibration())
    for name in TRANSPORT_FIELDS:
        assert rotated[f"{name}_normalized_rms"] == pytest.approx(original[f"{name}_normalized_rms"], rel=1e-13)


def test_own_centering_drifting_gaussian_has_zero_heat_flow():
    n, u, temperature = 2., 3., 4.
    fields = {"M0": [n], "M1": [n*u], "M2": [n*(u*u+temperature)],
              "M3": [n*(u**3+3*u*temperature)], "E": [0.]}
    centered = canonicalize_1d(fields)
    assert centered["pressure"][0, 0] == n * temperature
    assert centered["heat_flux"][0, 0] == 0
    assert fields["M3"][0] != 0


def test_prediction_and_reference_each_use_own_velocity_for_centering():
    truth = {"M0": [1.], "M1": [0.], "M2": [1.], "M3": [0.], "E": [0.]}
    pred = {"M0": [1.], "M1": [2.], "M2": [5.], "M3": [14.], "E": [0.]}
    metrics = score_transport(canonicalize_1d(pred), canonicalize_1d(truth), _calibration())
    assert metrics["pressure_normalized_rms"] == 0
    assert metrics["heat_flux_normalized_rms"] == 0
    assert metrics["particle_flux_normalized_rms"] == 2


def test_unit_rescaling_keeps_dimensionless_objective():
    reference = canonicalize_2d(_raw2())
    prediction = {name: array * 1.2 for name, array in reference.items()}
    original = score_transport(prediction, reference, _calibration())
    n0, v0, e0 = 5., 3., 2.
    factors = dict(zip(TRANSPORT_FIELDS, (n0, e0, n0*v0, n0*v0**2, n0*v0**3)))
    scaled_ref = {name: array * factors[name] for name, array in reference.items()}
    scaled_pred = {name: array * factors[name] for name, array in prediction.items()}
    asset = _calibration(_rows(scaled_ref), {"density": n0, "velocity": v0, "electric_field": e0})
    scaled = score_transport(scaled_pred, scaled_ref, asset)
    assert asset.to_dict()["floors"] == _calibration().to_dict()["floors"]
    assert scaled[TRANSPORT_PROTOCOL_ID] == pytest.approx(original[TRANSPORT_PROTOCOL_ID])


@pytest.mark.parametrize("density", [0., -1., math.nan, math.inf])
def test_invalid_density_is_not_clamped(density):
    raw1, raw2 = _raw1(), _raw2()
    raw1["M0"][0] = density
    raw2["rho"][0] = density
    with pytest.raises(ValueError):
        canonicalize_1d(raw1)
    with pytest.raises(ValueError):
        canonicalize_2d(raw2)


@pytest.mark.parametrize("matcher_cls,raw", [(VP1DTransportMacroMatcher, _raw1), (VPFPTransportMacroMatcher, _raw2)])
def test_matcher_reports_five_errors_four_blocks_and_score(matcher_cls, raw):
    matcher = matcher_cls(_calibration())
    query, view = _view(raw())
    target = matcher.prepare_target(_case(), view, query)
    result = matcher.match(_case(), view, target)
    assert result.primary_error == 0
    assert result.passed is None
    card = matcher.result_card(result)
    assert len([name for name in card.values if name.endswith("_normalized_rms")]) == 5
    assert len([name for name in card.values if name.endswith("_block_error")]) == 4
    assert card.values["objective_hash"] == matcher.objective_hash


def test_different_calibration_cannot_reuse_target():
    matcher = VPFPTransportMacroMatcher(_calibration())
    query, view = _view(_raw2())
    target = matcher.prepare_target(_case(), view, query)
    altered_rows = _rows()
    altered_rows[0] = dict(altered_rows[0], values={name: array * 2 for name, array in altered_rows[0]["values"].items()})
    another = VPFPTransportMacroMatcher(_calibration(altered_rows))
    with pytest.raises(ValueError, match="objective version or calibration"):
        another.match(_case(), view, target)


def test_nonuniform_sample_weights_not_silently_ignored():
    matcher = VPFPTransportMacroMatcher(_calibration())
    query, view = _view(_raw2())
    target = MacroTarget(_case().case_id, matcher.contract_id, query, view, sample_weights=(1., 2.))
    with pytest.raises(ValueError, match="uniform all-point RMS"):
        matcher.match(_case(), view, target)


def test_coordinate_mismatch_is_invalid_not_success_subset():
    matcher = VPFPTransportMacroMatcher(_calibration())
    query, view = _view(_raw2())
    target = matcher.prepare_target(_case(), view, query)
    prediction = AnswerView(view.kind, view.fields, points=tuple(reversed(view.points)), shape=view.shape)
    with pytest.raises(ValueError, match="coordinates"):
        matcher.match(_case(), prediction, target)


def test_overflow_cannot_receive_small_or_nonfinite_score():
    reference = canonicalize_2d(_raw2())
    prediction = copy.deepcopy(reference)
    prediction["pressure"][:] = 1e300
    with pytest.raises(ValueError, match="overflowed"):
        score_transport(prediction, reference, _calibration())
