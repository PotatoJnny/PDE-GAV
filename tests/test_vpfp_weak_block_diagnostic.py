"""Synthetic full-carrier mechanisms; numerical execution only on compute."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from tests.test_vpfp_moving_frame import tiny, constant_macro
from tests.test_vpfp_weak_dynamics_diagnostic import fixture
from pde_agent_verifier_solver.plugins.vpfp import moving_frame as mf
from pde_agent_verifier_solver.plugins.vpfp import weak_block_diagnostic as block
from pde_agent_verifier_solver.plugins.vpfp.value_control_diffusion import lift_value_controls


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def pair(degree=1):
    base, value, macro, z, condition = tiny(degree=degree)
    m = constant_macro(macro).to(z)
    decoder = mf.MovingFrameDecoder(base, macro, value_config=value)
    real = {"m": m, "z": z}
    generated = {"m": m.clone(), "z": z.clone()}
    generated["z"][:, :value.rho_dimension] = .03
    generated["z"][:, value.rho_dimension:] += .07
    return decoder, real, generated, condition


def test_registered_complete_blocks_and_finite_deep_subset():
    design = block.block_design()
    assert design["time_count"] == 129
    assert design["phase_design"]["physical_velocity_probes"] == [[0., 0.], [1., 0.], [-1., 0.], [0., 1.]]
    assert design["deep_selection"] == {"replicas": [0], "strata": [0, 1], "arms": list(block.ARMS[1:3]), "carrier_count": 4}
    assert not design["scientific_admission"] and not design["local_macro_moments_fixed_claimed"]


@pytest.mark.parametrize("degree", [1, 14])
def test_complete_block_swap_keeps_highest_H_floor_and_inputs(degree):
    decoder, real, generated, condition = pair(degree)
    before = [block.tensor_sha(x[k]) for x in (real, generated) for k in ("m", "z")]
    carriers, receipts = block.make_block_carriers(decoder, real, generated, condition)
    cut = decoder.value_config.rho_dimension
    for arm, (r, k) in zip(block.ARMS, ((0, 0), (1, 0), (0, 1), (1, 1))):
        source = (real, generated)
        assert torch.equal(carriers[arm]["z"][:, :cut], source[r]["z"][:, :cut])
        assert torch.equal(carriers[arm]["z"][:, cut:], source[k]["z"][:, cut:])
        assert torch.equal(carriers[arm]["m"], real["m"])
        assert receipts[arm]["complete_floor_preserved"] and receipts[arm]["no_Hermite_reprojection"]
        assert receipts[arm]["kinetic_range"][-1] == decoder.value_config.latent_dimension
    assert before == [block.tensor_sha(x[k]) for x in (real, generated) for k in ("m", "z")]


def test_density_swap_changes_actual_rho_weighted_local_velocity_not_U_C():
    decoder, real, generated, condition = pair()
    u, local, chol = mf.split_macro_controls(real["m"], decoder.macro_config)
    local[:, 0, :, 0] = .2
    real["m"] = mf.pack_macro_controls(u, local, chol, decoder.macro_config)
    generated["m"] = real["m"].clone()
    generated["z"][:, :decoder.value_config.rho_dimension] = 0.
    generated["z"][:, :decoder.value_config.rho_dimension].reshape(1, 3, 8)[:, :, 0] = .4
    carriers, _ = block.make_block_carriers(decoder, real, generated, condition)
    query = torch.tensor([[2.3, .2, .4, 0., 0.], [4.1, .8, .6, .3, -.2]], dtype=torch.float64)
    a = decoder.fields(query, carriers[block.ARMS[0]], equation_condition=condition)
    b = decoder.fields(query, carriers[block.ARMS[1]], equation_condition=condition)
    assert not torch.allclose(a["rho"], b["rho"])
    assert not torch.allclose(a["velocity"], b["velocity"])
    torch.testing.assert_close(a["global_velocity"], b["global_velocity"], rtol=0, atol=0)
    torch.testing.assert_close(a["central_covariance"], b["central_covariance"], rtol=0, atol=0)


@pytest.mark.parametrize("bad", ["different_m", "nan_z", "truncated_z", "signed_zero_m"])
def test_bad_source_pair_is_rejected(bad):
    decoder, real, generated, condition = pair()
    if bad == "different_m": generated["m"][0, 0] += .1
    if bad == "nan_z": generated["z"][0, 0] = float("nan")
    if bad == "truncated_z": generated["z"] = generated["z"][:, :-1]
    if bad == "signed_zero_m":
        index = int(torch.nonzero(real["m"][0] == 0)[0])
        generated["m"][0, index] = -0.
    with pytest.raises((ValueError, FloatingPointError)):
        block.make_block_carriers(decoder, real, generated, condition)


def test_actual_single_timeline_has_signed_terms_macros_and_physical_units():
    decoder, carrier, condition, kwargs = fixture(drift=True)
    kwargs["design"] = block.block_design(control_count=3)
    result = block.measure_block_carrier(decoder, carrier, condition, **kwargs)
    assert result["execution_passed"] and result["timeline"]["count"] == 9
    t = result["timeline"]; v = {k: np.asarray(x) for k, x in t["values"].items()}
    np.testing.assert_allclose(v["dt_U"][..., 0], .1, rtol=1.e-10, atol=1.e-12)
    np.testing.assert_allclose(v["S"], v["dt_f"]+v["transport"]+v["electric_acceleration"]-v["nu_collision"], atol=1.e-14)
    assert all(k in v for k in ("rho", "u", "C", "U", "dt_u", "dt_C"))
    comparison = block.compare_timelines(t, t)
    assert all(x["difference_time_probe_RMS"] == 0 for x in comparison["fields"].values())
    assert comparison["comparison"] == "same_root_counterfactual_not_truth_error"
    bad = deepcopy(t); bad["physical_points_txyvxvy"][0][0] += .001
    with pytest.raises(ValueError): block.compare_timelines(t, bad)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_clean_gaussian_actual_A_and_B_source_derivatives_agree(dtype):
    decoder, carrier, condition, kwargs = fixture(dtype=dtype, drift=False)
    kwargs["design"] = block.block_design(control_count=3)
    kwargs["provenance"].update(data_origin="physical_reference", view="real_m_real_z")
    full = lift_value_controls(carrier["z"], decoder.value_config)
    result = block.measure_clean_derivatives(decoder, carrier, condition, reference_full=full, **kwargs)
    assert result["execution_passed"], result.get("error")
    assert result["derivative_order"] == ["t", "x", "y", "vx", "vy"]
    assert result["units"] == "physical_coordinates_no_horizon_rescaling"
    for field in result["fields"].values():
        for summary in field["by_time_kind"].values():
            assert summary["RMS_difference"] < (2.e-7 if dtype == "float32" else 3.e-12)
    counts = result["fields"]["f"]["by_time_kind"]
    assert [counts[k]["phase_query_count"] for k in ("all", "nodes", "midpoints", "quarters")] == [36, 12, 8, 16]
    assert result["numerically_uncertain"] and not result["independent_PDE_residual"]


def test_clean_wrong_B_and_generated_provenance_fail_closed():
    decoder, carrier, condition, kwargs = fixture(drift=False)
    kwargs["design"] = block.block_design(control_count=3)
    full = lift_value_controls(carrier["z"], decoder.value_config)
    result = block.measure_clean_derivatives(decoder, carrier, condition, reference_full=full, **kwargs)
    assert not result["execution_passed"]
    kwargs["provenance"].update(data_origin="physical_reference", view="real_m_real_z")
    other = full.clone(); other[0, 0] += .2
    result = block.measure_clean_derivatives(decoder, carrier, condition, reference_full=other, **kwargs)
    assert not result["execution_passed"]


def test_deep_failure_preserves_completed_timeline(tmp_path, monkeypatch):
    decoder, carrier, condition, kwargs = fixture(drift=False)
    kwargs.update(design=block.block_design(control_count=3), output=tmp_path/"result.json")
    calls = []
    def deep(*args, **kw):
        calls.append(kw)
        return {"execution_passed": False, "reason": "synthetic quadrature failure"}
    monkeypatch.setattr(block, "audit_independent_physics", deep)
    result = block.measure_block_carrier(decoder, carrier, condition, deep=True, **kwargs)
    assert not result["execution_passed"] and result["timeline"]["count"] == 9
    assert calls[0]["points_txy"].shape == (8, 3) and calls[0]["spatial_grids"] == (17, 33)
    assert json.loads(kwargs["output"].read_text())["independent_physics"]["reason"]
    with pytest.raises(ValueError, match="fresh"):
        block.measure_block_carrier(decoder, carrier, condition, **kwargs)


def test_deadline_records_error_and_restores_model_mode(tmp_path):
    decoder, carrier, condition, kwargs = fixture()
    kwargs.update(design=block.block_design(control_count=3), output=tmp_path/"result.json")
    def deadline(): raise TimeoutError("synthetic allocation deadline")
    decoder.eval()
    result = block.measure_block_carrier(decoder, carrier, condition, check_deadline=deadline, **kwargs)
    assert not result["execution_passed"] and result["error"]["type"] == "TimeoutError"
    assert not decoder.training and kwargs["output"].is_file()
