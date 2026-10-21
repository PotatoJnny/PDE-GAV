"""Synthetic mechanics only, not evidence about the real VPFP distribution."""
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

from pde_agent_verifier_solver.plugins.vpfp.structured_diffusion import VPFPStructuredEDMConfig
from pde_agent_verifier_solver.plugins.vpfp.temporal_covariance_diffusion import VPFPTemporalCovarianceNormalizer
from pde_agent_verifier_solver.plugins.vpfp.value_control_diffusion import VPFPValueEDMConfig, lift_value_controls, _value_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("conditional_geometry_test", ROOT / "scripts/diagnose_vpfp_conditional_geometry.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture(autouse=True)
def bounded_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def data(dtype=torch.float64, recipe="mixture"):
    cfg = VPFPValueEDMConfig(VPFPStructuredEDMConfig(control_count=4, hermite_channels=2,
        rho_max_mode=1, kinetic_max_mode=1, inactive_dim=2, grid_size=3, width=8, depth=1))
    values = torch.randn(8, cfg.latent_dimension, dtype=dtype, generator=torch.Generator().manual_seed(14))
    _value_split(values, cfg)[1][..., 0] += torch.linspace(3, 1, cfg.control_count)
    raw = lift_value_controls(values, cfg)
    normalizer = VPFPTemporalCovarianceNormalizer.fit(cfg, raw, sample_ids=list(range(8)), covariance_recipe=recipe)
    return cfg, raw, module.ConditionalFutureCoordinates(normalizer)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("recipe", ["mixture", "trace_isotropic", "unit_trace_shape"])
def test_full_abi_roundtrip_and_schur_whitening(dtype, recipe):
    cfg, raw, codec = data(dtype, recipe)
    context, future = codec.encode(raw)
    restored = codec.decode(context, future)
    torch.testing.assert_close(restored, raw, rtol=3.e-6 if dtype == torch.float32 else 1.e-12,
                               atol=3.e-6 if dtype == torch.float32 else 1.e-12)
    assert context.shape[1] + future.shape[1] == cfg.latent_dimension
    assert codec.report()["whitening_max_abs_error"] < 1.e-12
    assert codec.report()["future_times"] == pytest.approx([1/3, 2/3, 1])


def test_no_future_or_slope_leak_into_context():
    cfg, raw, codec = data()
    context, future = codec.encode(raw)
    changed = codec.decode(context, future + 2)
    same_context, _ = codec.encode(changed)
    torch.testing.assert_close(same_context, context, rtol=0, atol=1.e-15)
    parent = codec.normalizer.parent.encode(raw).detach().requires_grad_(True)
    # Source lift is deliberately recomputed: endpoint slope changes with t1.
    full = codec.normalizer.parent.decode(parent)
    differentiable_context, _ = codec.encode(full)
    gradient, = torch.autograd.grad(differentiable_context.sum(), parent)
    rho_gradient, kinetic_gradient = _value_split(gradient, cfg)
    assert torch.count_nonzero(rho_gradient[:, 1:]) == 0
    assert torch.count_nonzero(kinetic_gradient[:, :, 1:]) == 0
    assert torch.count_nonzero(rho_gradient[:, 0]) > 0


def test_gradient_through_future_and_initial_condition():
    _, raw, codec = data()
    context, future = codec.encode(raw)
    context.requires_grad_(True)
    future.requires_grad_(True)
    codec.decode(context, future).square().mean().backward()
    assert torch.isfinite(context.grad).all() and context.grad.abs().sum() > 0
    assert torch.isfinite(future.grad).all() and future.grad.abs().sum() > 0


def test_reject_malformed_context_and_not_a_model_claim():
    _, raw, codec = data()
    context, future = codec.encode(raw)
    with pytest.raises(ValueError, match="aligned"):
        codec.decode(context[:, :-1], future)
    with pytest.raises(ValueError, match="aligned"):
        codec.decode(context, future * float("nan"))
    assert codec.report()["trained_conditional_model"] is False
    assert codec.report()["conditional_gaussian_target_assumed"] is False
