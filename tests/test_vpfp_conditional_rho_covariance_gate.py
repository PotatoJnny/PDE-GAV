"""Same affine arithmetic budget with nonidentity conditional rho constants."""
import importlib.util
from pathlib import Path

import pytest
import torch

from pde_agent_verifier_solver.plugins.vpfp import numerical_roundtrip as gate
from pde_agent_verifier_solver.plugins.vpfp.initial_conditioned_diffusion import VPFPInitialConditionedNormalizer
from pde_agent_verifier_solver.plugins.vpfp.temporal_covariance_diffusion import VPFPTemporalCovarianceNormalizer
from pde_agent_verifier_solver.plugins.vpfp.value_control_diffusion import extract_value_controls, lift_value_controls

loader = importlib.util.spec_from_file_location("rho_cov_gate_fixture", Path(__file__).with_name("test_vpfp_numerical_roundtrip.py"))
fixture = importlib.util.module_from_spec(loader)
loader.loader.exec_module(fixture)


@pytest.fixture(autouse=True)
def bounded_threads():
    count = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(count)


def make(count=33, correlated=True):
    cfg, raw, _ = fixture.case("conditional", count=count, correlated=correlated)
    normalizer = VPFPInitialConditionedNormalizer.fit(cfg, raw, sample_ids=list(range(8)),
        initial_prior=fixture.fixture.prior_spec(cfg), rho_policy="covariance", covariance_strength=1.,
        center_kinetic_dc=True, relative_floor=.05)
    return cfg, raw, normalizer


@pytest.mark.parametrize("count,correlated", [(5, False), (33, True)])
def test_rho_covariance_schur_gate_uses_actual_constants_and_original_tolerances(count, correlated):
    cfg, raw, normalizer = make(count, correlated)
    context, future = normalizer.encode(raw[:4])
    restored = normalizer.decode((context, future))
    report = gate.audit_normalizer_roundtrip(normalizer, raw[:4], restored)
    assert report["passed"], report
    assert report["budget"]["rho_policy"] == "covariance"
    assert report["value_tolerance"] == {"rtol": 2.e-4, "atol": 2.e-5}
    assert report["restored_native_lift_bitwise_passed"]
    assert normalizer.regression[0].abs().max() > .01
    loaded = VPFPInitialConditionedNormalizer.from_payload(normalizer.to_payload())
    torch.testing.assert_close(loaded.decode(loaded.encode(raw[:4])), restored, rtol=0, atol=0)
    changed = normalizer.decode((context, future + .2))
    changed_context, _ = normalizer.encode(changed)
    torch.testing.assert_close(changed_context, context, rtol=2.e-6, atol=2.e-6)


def test_rho_covariance_budget_is_source_only_and_rejects_value_and_slope_faults():
    cfg, raw, normalizer = make()
    budget, _ = gate.value_roundtrip_budget(normalizer, raw[:2])
    restored = normalizer.decode(normalizer.encode(raw[:2]))
    values = extract_value_controls(restored, cfg.value_config)
    values[0, 0] += 1.e-5
    fault = lift_value_controls(values, cfg.value_config)
    report = gate.audit_normalizer_roundtrip(normalizer, raw[:2], fault)
    assert report["groups"]["rho"]["value_old_allclose_passed"]
    assert not report["groups"]["rho"]["value_IEEE_budget_passed"] and not report["passed"]
    altered = restored.clone()
    altered.view(torch.int32)[0, cfg.value_config.rho_dimension] ^= 1
    assert not gate.audit_normalizer_roundtrip(normalizer, raw[:2], altered)["restored_native_lift_bitwise_passed"]
    assert torch.equal(budget, gate.value_roundtrip_budget(normalizer, raw[:2])[0])


def test_wrong_rho_inverse_cannot_be_absorbed_as_fixed_operator_defect():
    _, raw, normalizer = make()
    inverse = normalizer.inverse_square_root.clone()
    inverse[0] *= 1.00001
    normalizer.inverse_square_root = inverse
    with pytest.raises((ValueError, gate.UnsupportedRoundtripArithmetic), match="operator|canonical|buffer|frozen"):
        gate.value_roundtrip_budget(normalizer, raw[:2])


def test_unconditional_covariance_rho_stays_outside_this_narrow_scope():
    cfg, raw, _ = make()
    normalizer = VPFPTemporalCovarianceNormalizer.fit(cfg.value_config, raw,
        sample_ids=list(range(8)), rho_policy="covariance")
    with pytest.raises(gate.UnsupportedRoundtripArithmetic, match="recipe"):
        gate.value_roundtrip_budget(normalizer, raw[:2])
