"""Tiny synthetic mechanics only; numerical execution belongs on compute nodes."""
import json

import numpy as np
import pytest
import torch

from tests.test_vpfp_moving_frame import tiny, constant_macro
from pde_agent_verifier_solver.plugins.vpfp import moving_frame as mf
from pde_agent_verifier_solver.plugins.vpfp import weak_dynamics_diagnostic as wd
from pde_agent_verifier_solver.plugins.vpfp.value_control_diffusion import lift_value_controls


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture(dtype="float64", *, drift=True):
    base, vc, mc, z, condition = tiny(degree=1, dtype=dtype)
    theta = float(condition[1])
    m = constant_macro(mc, u=(0., 0.), covariance=((theta, 0.), (0., theta))).to(z)
    if drift:
        u, local, chol = mf.split_macro_controls(m, mc)
        u[0, 0] = torch.tensor([.2, .35, .5], dtype=z.dtype)
        m = mf.pack_macro_controls(u, local, chol, mc)
    decoder = mf.MovingFrameDecoder(base, mc, value_config=vc)
    return decoder, {"m": m, "z": z}, condition, dict(output=None, root_id="fixture-root",
        provenance={"source_manifest_sha256": "a"*64, "split": "fixture", "data_origin": "synthetic_fixture"},
        design=wd.dynamics_design(control_count=3))


def test_default_registered_protocol_and_no_science_threshold():
    design = wd.dynamics_design()
    assert design["time_counts"] == [65, 129]
    assert design["spatial_probe_fractions"] == [[.173, .389], [.431, .713], [.719, .227], [.857, .563]]
    assert design["physical_velocity_probes"] == [[0., 0.], [1., 0.], [-1., 0.], [0., 1.]]
    assert "not_Cartesian_product" in design["phase_probe_pairing"]
    assert "not_full_space_qualification" in design["phase_probe_evidence"]
    assert design["deep_time_fractions"] == [.0703125, .7578125]
    assert design["deep_spatial_grids"] == [17, 33]
    assert design["relative_fd_step"] == 2.**-8
    assert not design["scientific_admission"] and not design["automatic_sample_reduction"]
    with pytest.raises(ValueError):
        wd.dynamics_design(control_count=True)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_actual_gaussian_global_translation_and_damping(dtype):
    decoder, carrier, condition, kwargs = fixture(dtype)
    frozen = {k: v.clone() for k, v in carrier.items()}
    was_training = decoder.training
    result = wd.measure_dynamics_carrier(decoder, carrier, condition, **kwargs)
    assert result["execution_passed"] and result["numerically_uncertain"]
    assert not result["physical_regime_qualified"] and not result["scientific_admission"]
    assert [g["count"] for g in result["timeline"]] == [5, 9]
    for grid in result["timeline"]:
        values = {k: np.asarray(v) for k, v in grid["values"].items()}
        count = grid["count"]
        physical = np.asarray(grid["physical_points_txyvxvy"])
        assert physical.shape == (count*4, 5) and grid["probe_count"] == 4
        np.testing.assert_array_equal(physical[:, 0], np.repeat(grid["times"], 4))
        np.testing.assert_array_equal(physical[:, 3:5], np.tile(kwargs["design"]["physical_velocity_probes"], (count, 1)))
        assert values["f"].shape == (count, 4)
        assert "not_Cartesian_product" in grid["probe_geometry"]
        np.testing.assert_allclose(values["dt_U"][..., 0], .1, rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(values["global_U_drive_proxy"], values["dt_U"]+float(condition[0])*values["U"], rtol=2e-14, atol=1e-14)
        np.testing.assert_allclose(values["U_comoving_dt_remainder"], 0., atol=2e-12)
        np.testing.assert_allclose(values["dt_f"], values["global_U_translation_dt_f"], atol=2e-12)
        np.testing.assert_allclose(values["S"], values["dt_f"]+values["transport"]+values["electric_acceleration"]-values["nu_collision"], atol=1e-14)
        assert not grid["remainder_is_pure_micro"] and not grid["fixed_w_trajectory_measured"]
        assert not grid["source_closure_is_independent_check"]
    for k in frozen:
        assert torch.equal(carrier[k], frozen[k])
    assert decoder.training == was_training
    json.dumps(result, allow_nan=False)


def test_zero_nu_ratios_undefined_not_epsilon_pass():
    decoder, carrier, condition, kwargs = fixture()
    condition[0] = 0.
    runtime, xy, vv = wd._prepare(decoder, carrier, condition, kwargs["design"])
    result = wd._phase_grid(runtime, xy, vv, 5, None)
    for region in result["early_late"].values():
        assert all(v is None for v in region["source_over_nu_f_by_time"])
        assert all(v is None for v in region["global_drive_over_nu_sqrtT_by_time"])


def test_signed_transport_cancellation_is_not_a_dt_penalty(monkeypatch):
    # Synthetic term ledger only: no manufactured fixture is claimed to solve
    # a transport equation, and this is not an independent residual test.
    decoder, carrier, condition, kwargs = fixture(drift=False)
    runtime, xy, vv = wd._prepare(decoder, carrier, condition, kwargs["design"])
    vv[:, 0] = 1.
    def synthetic_terms(decoder, query, answer, condition):
        fields = decoder.fields(query, answer, equation_condition=condition)
        gradient = torch.zeros_like(query)
        gradient[:, 0], gradient[:, 1] = -2., 2.
        return dict(fields, f_gradient=gradient, collision=torch.zeros_like(query[:, 0]),
                    source=torch.zeros_like(query[:, 0]))
    monkeypatch.setattr(wd, "manufacture_moving_frame_source", synthetic_terms)
    result = wd._phase_grid(runtime, xy, vv, 5, None)
    assert np.all(np.asarray(result["values"]["dt_f"]) == -2.)
    assert np.all(np.asarray(result["values"]["transport"]) == 2.)
    assert np.all(np.asarray(result["values"]["dt_plus_transport"]) == 0.)
    assert np.all(np.asarray(result["values"]["source_algebraic_closure"]) == 0.)
    assert not result["source_closure_is_independent_check"]


def test_exact_same_root_clean_gaussian_compression():
    decoder, carrier, condition, kwargs = fixture(drift=False)
    full = lift_value_controls(carrier["z"], decoder.value_config)
    kwargs["provenance"].update(data_origin="physical_reference", view="real_m_real_z")
    result = wd.measure_clean_compression(decoder, carrier, condition, reference_full=full, **kwargs)
    assert result["execution_passed"] and result["input_B_full_sha256"]
    for grid in result["grids"]:
        assert np.asarray(grid["physical_points_txyvxvy"]).shape == (grid["count"]*4, 5)
        assert grid["probe_count"] == 4
        for field in grid["fields"].values():
            assert field["RMS_difference"] < 2e-12
    assert result["numerically_uncertain"]


@pytest.mark.parametrize("bad", ["generated", "wrong_z"])
def test_clean_compression_cannot_relabel_generated_or_other_B(bad):
    decoder, carrier, condition, kwargs = fixture(drift=False)
    full = lift_value_controls(carrier["z"], decoder.value_config)
    kwargs["provenance"].update(data_origin="physical_reference", view="real_m_real_z")
    if bad == "generated":
        kwargs["provenance"]["data_origin"] = "generated_answer"
    else:
        carrier["z"] = carrier["z"]+.1
    result = wd.measure_clean_compression(decoder, carrier, condition, reference_full=full, **kwargs)
    assert not result["execution_passed"] and result["error"]["type"] == "ValueError"


def _mock_shallow(monkeypatch):
    monkeypatch.setattr(wd.controls, "temporal_frequency_diagnostics", lambda *a, **k: {"execution_passed": True, "fixture": True})
    monkeypatch.setattr(wd, "_phase_grid", lambda r, x, v, n, c: {"count": n, "fixture": True})


def test_deep_exact_points_and_failure_preserved(monkeypatch, tmp_path):
    decoder, carrier, condition, kwargs = fixture()
    _mock_shallow(monkeypatch)
    calls = []
    def deep(*args, **options):
        calls.append(options)
        return {"execution_passed": False, "error": "synthetic deep failure"}
    monkeypatch.setattr(wd, "audit_independent_physics", deep)
    kwargs["output"] = tmp_path/"diagnostic.json"
    result = wd.measure_dynamics_carrier(decoder, carrier, condition, deep=True, **kwargs)
    assert not result["execution_passed"] and result["stage"] == "measurement_failure"
    assert len(calls) == 1 and calls[0]["spatial_grids"] == (17, 33)
    assert calls[0]["points_txy"].shape == (8, 3)
    np.testing.assert_array_equal(calls[0]["points_txy"][:, 0], np.repeat(2+3*np.asarray([.0703125, .7578125]), 4))
    expected_xy = np.asarray(kwargs["design"]["spatial_probe_fractions"])*np.asarray(decoder.base_decoder.config.equation.x_domain)
    np.testing.assert_array_equal(calls[0]["points_txy"][:, 1:3], np.tile(expected_xy, (2, 1)))
    assert calls[0]["relative_fd_step"] == 2.**-8 and calls[0]["include_l2"]
    saved = json.loads(kwargs["output"].read_text())
    assert saved["independent_physics"]["error"] == "synthetic deep failure"
    assert len(saved["timeline"]) == 2


def test_deadline_keeps_completed_pass_no_automatic_sample_reduction(monkeypatch, tmp_path):
    decoder, carrier, condition, kwargs = fixture()
    _mock_shallow(monkeypatch)
    def phase(runtime, xy, vv, count, callback):
        if count == 9:
            raise TimeoutError("fixed synthetic wall budget")
        return {"count": count}
    monkeypatch.setattr(wd, "_phase_grid", phase)
    kwargs["output"] = tmp_path/"partial.json"
    result = wd.measure_dynamics_carrier(decoder, carrier, condition, **kwargs)
    assert not result["execution_passed"] and result["error"]["type"] == "TimeoutError"
    assert len(result["timeline"]) == 1 and result["design"]["time_counts"] == [5, 9]
    assert json.loads(kwargs["output"].read_text())["native_temporal_frequency"]["execution_passed"]


def test_fresh_output_and_registered_design_required(tmp_path):
    decoder, carrier, condition, kwargs = fixture()
    path = tmp_path/"existing.json"
    path.write_text("preserve")
    kwargs["output"] = path
    with pytest.raises(ValueError, match="fresh"):
        wd.measure_dynamics_carrier(decoder, carrier, condition, **kwargs)
    assert path.read_text() == "preserve"
    kwargs["output"] = None
    kwargs["design"]["deep_spatial_grids"] = [17]
    result = wd.measure_dynamics_carrier(decoder, carrier, condition, **kwargs)
    assert not result["execution_passed"] and "registered" in result["error"]["message"]


def test_phase_failure_does_not_change_decoder_mode(monkeypatch):
    decoder, carrier, condition, kwargs = fixture()
    _mock_shallow(monkeypatch)
    decoder.eval()
    monkeypatch.setattr(wd, "_phase_grid", lambda *a: (_ for _ in ()).throw(RuntimeError("synthetic")))
    result = wd.measure_dynamics_carrier(decoder, carrier, condition, **kwargs)
    assert not result["execution_passed"] and not decoder.training
