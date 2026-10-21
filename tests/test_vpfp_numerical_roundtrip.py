"""Bounded synthetic arithmetic/fault tests; no real-data acceptance tuning."""
from fractions import Fraction
import ctypes
import importlib.util
import inspect
import json
from pathlib import Path

import pytest
import torch

from pde_agent_verifier_solver.plugins.vpfp import numerical_roundtrip as gate
from pde_agent_verifier_solver.plugins.vpfp.temporal_covariance_diffusion import VPFPTemporalCovarianceNormalizer
from pde_agent_verifier_solver.plugins.vpfp.initial_conditioned_diffusion import VPFPInitialConditionedNormalizer
from pde_agent_verifier_solver.plugins.vpfp.value_control_diffusion import lift_value_controls, extract_value_controls, _value_split

loader = importlib.util.spec_from_file_location("roundtrip_conditional_fixture", Path(__file__).with_name("test_vpfp_initial_conditioned_diffusion.py"))
fixture = importlib.util.module_from_spec(loader)
loader.loader.exec_module(fixture)


@pytest.fixture(autouse=True)
def bounded_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def case(arm, *, count=33, correlated=True):
    cfg = fixture.fixture_config(count)
    if not correlated:
        _, raw, _ = fixture.data(torch.float32, count=count)
    else:
        generator = torch.Generator().manual_seed(708)
        values = torch.zeros(8, cfg.value_config.latent_dimension, dtype=torch.float32)
        rho, kinetic = _value_split(values, cfg.value_config)
        time = torch.linspace(0., 1., count)
        rho[:] = .02*torch.randn(8, 1, cfg.rho_modes, generator=generator)*torch.exp(-3*time)[None, :, None]
        kinetic[:] = .2*torch.randn(8, cfg.hermite_channels, 1, cfg.kinetic_modes, generator=generator)*torch.exp(-14*time)[None, None, :, None]
        kinetic += 1.e-7*torch.randn(kinetic.shape, generator=generator)
        kinetic[:, 0, :, 0] += 9.
        raw = lift_value_controls(values, cfg.value_config)
    if arm == "cov":
        normalizer = VPFPTemporalCovarianceNormalizer.fit(cfg.value_config, raw, sample_ids=list(range(8)))
    else:
        normalizer = VPFPInitialConditionedNormalizer.fit(cfg, raw, sample_ids=list(range(8)), initial_prior=fixture.prior_spec(cfg))
    return cfg, raw, normalizer


@pytest.mark.parametrize("arm", ["cov", "conditional"])
@pytest.mark.parametrize("correlated,count", [(False, 5), (True, 33)])
def test_original_and_T33_large_DC_conditioned_roundtrips(arm, correlated, count):
    cfg, raw, normalizer = case(arm, count=count, correlated=correlated)
    restored = normalizer.decode(normalizer.encode(raw[:4]))
    report = gate.audit_normalizer_roundtrip(normalizer, raw[:4], restored)
    assert report["passed"], report
    assert report["source_original_8eps_support_passed"]
    assert report["restored_native_lift_bitwise_passed"] and report["inactive_positive_zero_bitwise_passed"]
    assert report["value_tolerance"] == {"rtol": 2.e-4, "atol": 2.e-5}
    assert report["budget"]["observed_residual_used_for_budget"] is False
    assert not report["physical_admission"] and not report["answer_modified"]
    if correlated:
        temporal = normalizer.temporal if arm == "conditional" else normalizer
        eig = temporal.temporal.regularized_eigenvalues
        condition = float((eig[:, -1]/eig[:, 0]).max())
        assert 10000 < condition <= count/.05**2*(1+1.e-10)
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("arm", ["cov", "conditional"])
def test_budget_has_no_restored_argument_and_rejects_small_consistent_value_fault(arm):
    cfg, raw, normalizer = case(arm)
    assert list(inspect.signature(gate.value_roundtrip_budget).parameters) == ["normalizer", "source_raw"]
    before, _ = gate.value_roundtrip_budget(normalizer, raw[:2])
    restored = normalizer.decode(normalizer.encode(raw[:2]))
    values = extract_value_controls(restored, cfg.value_config)
    values[0, 0] += 1.e-5  # Below old value atol, not derived from observed roundtrip error.
    altered = lift_value_controls(values, cfg.value_config)
    report = gate.audit_normalizer_roundtrip(normalizer, raw[:2], altered)
    assert report["groups"]["rho"]["value_old_allclose_passed"]
    assert not report["groups"]["rho"]["value_IEEE_budget_passed"]
    assert report["restored_native_lift_bitwise_passed"] and not report["passed"]
    after, _ = gate.value_roundtrip_budget(normalizer, raw[:2])
    assert torch.equal(before, after)


@pytest.mark.parametrize("arm", ["cov", "conditional"])
def test_single_bit_slope_fault_and_inactive_fault_rejected(arm):
    cfg, raw, normalizer = case(arm)
    restored = normalizer.decode(normalizer.encode(raw[:2]))
    altered = restored.clone()
    altered.view(torch.int32)[0, cfg.value_config.rho_dimension] ^= 1
    report = gate.audit_normalizer_roundtrip(normalizer, raw[:2], altered)
    assert not report["passed"] and not report["restored_native_lift_bitwise_passed"]
    assert report["restored_native_lift_bitwise_mismatches"] == 1
    altered = restored.clone()
    altered[0, cfg.full_config.rho_dimension] = 1.
    assert not gate.audit_normalizer_roundtrip(normalizer, raw[:2], altered)["passed"]


def test_source_support_is_not_relaxed():
    cfg, raw, normalizer = case("cov")
    source = raw[:2].clone()
    source[0, cfg.value_config.rho_dimension] += .001
    report = gate.audit_normalizer_roundtrip(normalizer, source, source)
    assert not report["passed"] and "slopes leave" in report["reason"]


@pytest.mark.parametrize("problem", ["subnormal", "float64", "AMP", "overflow"])
def test_unsupported_precision_ranges_fail_closed(problem):
    cfg, raw, normalizer = case("cov")
    source = raw[:2].clone()
    if problem == "AMP":
        with torch.autocast("cpu", dtype=torch.bfloat16):
            report = gate.audit_normalizer_roundtrip(normalizer, source, source)
    else:
        if problem == "subnormal":
            source[0, 0] = torch.finfo(torch.float32).tiny/2
        elif problem == "float64":
            source = source.double()
        else:
            values = extract_value_controls(source, cfg.value_config)
            rho, _ = _value_split(values, cfg.value_config)
            rho[:] = torch.finfo(torch.float32).max
            source = lift_value_controls(values, cfg.value_config)
        report = gate.audit_normalizer_roundtrip(normalizer, source, source)
    assert report["status"] == "unsupported" and not report["passed"], report


def test_positive_budget_arithmetic_is_outward_and_gamma_uses_unit_roundoff():
    for dtype, power in ((torch.float32, 24), (torch.float64, 53)):
        for count in (1, 2, 66):
            exact = Fraction(count, 2**power-count)
            assert Fraction.from_float(gate._gamma(count, dtype)) >= exact
    a = torch.tensor([.1, 1.e-200, 0.], dtype=torch.float64)
    b = torch.tensor([.3, 1.e-200, 0.], dtype=torch.float64)
    for function, operation in ((gate._mul, lambda x, y: x*y), (gate._add, lambda x, y: x+y)):
        result = function(a, b)
        for x, y, observed in zip(a.tolist(), b.tolist(), result.tolist()):
            assert Fraction.from_float(observed) >= operation(Fraction.from_float(x), Fraction.from_float(y))
    matrix = torch.tensor([[[.1, .3, .7], [1.e-200, 1.e-200, 1.e-200]]], dtype=torch.float64)
    values = torch.tensor([[[[.2], [.4], [.9]]]], dtype=torch.float64)
    observed = gate._dot(matrix, values)
    for row in range(2):
        exact = sum(Fraction.from_float(float(matrix[0, row, j]))*Fraction.from_float(float(values[0, 0, j, 0])) for j in range(3))
        assert Fraction.from_float(float(observed[0, 0, row, 0])) >= exact


def test_no_frozen_model_or_normalizer_changes():
    _, raw, normalizer = case("conditional")
    state = {name: value.clone() for name, value in normalizer.state_dict().items()}
    metadata = normalizer.to_payload()
    gate.value_roundtrip_budget(normalizer, raw[:2])
    assert normalizer.to_payload() == metadata
    assert all(torch.equal(value, state[name]) for name, value in normalizer.state_dict().items())


def test_DAZ_subnormal_bits_and_FTZ_mode_cannot_masquerade_as_exact_zero():
    _, raw, normalizer = case("cov")
    source = raw[:2].clone()
    source.view(torch.int32)[0, 0] = 1
    try:
        torch.set_flush_denormal(True)
        with pytest.raises(gate.UnsupportedRoundtripArithmetic, match="subnormal"):
            gate._normal(source, "source")
        result = gate.audit_normalizer_roundtrip(normalizer, raw[:2], raw[:2])
        assert result["status"] == "unsupported" and "FTZ/DAZ" in result["reason"]
    finally:
        torch.set_flush_denormal(False)


def test_local_rounding_uses_envelope_for_nonidentical_legal_paths():
    value = torch.tensor([[[[1.]]]], dtype=torch.float32)
    node = gate._Node(value, torch.full_like(value, .25, dtype=torch.float64))
    envelope = gate._envelope(node)
    assert float(envelope) >= 1.5
    result = gate._scale(node, torch.ones_like(value))
    required = .25+gate._gamma(1, torch.float32)*1.5
    assert float(result.error) >= required


@pytest.mark.parametrize("arm", ["cov", "conditional"])
def test_replaced_wrong_inverse_rejected_independently_of_affine_budget(arm):
    _, raw, normalizer = case(arm, count=5, correlated=False)
    target = normalizer.temporal if arm == "cov" else normalizer
    target.inverse_square_root = target.inverse_square_root*1.00001
    assert target.inverse_square_root._version == 0  # A version-only guard misses this replacement.
    restored = normalizer.decode(normalizer.encode(raw[:2]))
    report = gate.audit_normalizer_roundtrip(normalizer, raw[:2], restored)
    assert not report["passed"] and "operator" in report["reason"]
    assert "budget" not in report  # Never use larger RW-I allowance to admit the wrong inverse.


@pytest.mark.parametrize("mode", [0x400, 0x800, 0xc00])
def test_non_nearest_host_rounding_fails_closed_and_is_restored(mode):
    _, raw, normalizer = case("cov", count=5, correlated=False)
    libc = ctypes.CDLL(None)
    before = libc.fegetround()
    try:
        assert libc.fesetround(mode) == 0
        report = gate.audit_normalizer_roundtrip(normalizer, raw[:2], raw[:2])
        assert report["status"] == "unsupported" and "round-to-nearest" in report["reason"]
    finally:
        assert libc.fesetround(before) == 0


def test_source_support_includes_FP64_center_and_comparison_rounding():
    cfg = fixture.fixture_config(5).value_config
    values = torch.zeros(1, cfg.latent_dimension, dtype=torch.float32)
    rho, _ = _value_split(values, cfg)
    rho[0, 0, 0], rho[0, 1, 0] = 2.**-40, (2.**20-1)/4
    raw = lift_value_controls(values, cfg)
    source_rho, _ = gate._split(raw, cfg.full_config)
    source_rho[0, 1, 0, 0] = 2.**20+1
    extract_value_controls(raw, cfg)  # Original 8eps acceptance is unchanged.
    bound = gate._source_support_budget(source_rho[:, None, 0], source_rho[:, None, 1], cfg)
    exact_Dv = (Fraction.from_float(float(rho[0, 1, 0]))-Fraction.from_float(float(rho[0, 0, 0])))*4
    exact_defect = abs(Fraction(2**20+1)-exact_Dv)
    assert exact_defect > 2
    assert Fraction.from_float(float(bound[0, 0, 0, 0])) >= exact_defect
