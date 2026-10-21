"""Mechanics fixtures, not real-data or regime qualification."""
from dataclasses import replace

import pytest
import torch

from pde_agent_verifier_solver.plugins.vpfp.structured_diffusion import VPFPStructuredEDMConfig
from pde_agent_verifier_solver.plugins.vpfp.value_control_diffusion import VPFPValueEDMConfig
from pde_agent_verifier_solver.plugins.vpfp.coherent_diffusion import (
    VPFPCoherentMacroConfig, VPFPCoherentMacroTransform, VPFPCoherentMacroNormalizer,
    VPFPCoherentMacroEDM, VPFPMacroConditionalEDM, sample_hierarchical,
)


def configs():
    full = VPFPStructuredEDMConfig(control_count=3, rho_max_mode=1, kinetic_max_mode=1,
        hermite_channels=3, inactive_dim=0, grid_size=5, width=16, depth=1,
        collision_nu_range=(.02,.08), temperature_range=(.8,1.2))
    return VPFPValueEDMConfig(full), VPFPCoherentMacroConfig(full, maximum_mode=2, coherent_mode=1)


def test_macro_transform_is_exact_in_declared_space():
    _, cfg = configs()
    transform = VPFPCoherentMacroTransform(cfg)
    x = torch.randn(2,cfg.latent_dimension,dtype=torch.float64)
    torch.testing.assert_close(transform.to_coefficients(transform.to_field(x)), x, atol=1e-13, rtol=1e-13)
    smooth = transform.to_coefficients(transform.to_field(x, coherent_only=True))
    assert not torch.allclose(smooth,x)
    torch.testing.assert_close(transform.to_coefficients(transform.to_field(smooth, coherent_only=True)), smooth)


def test_macro_normalizer_train_only_roundtrip_and_payload():
    _, cfg = configs()
    x = torch.randn(6,cfg.latent_dimension,dtype=torch.float64)
    normalizer = VPFPCoherentMacroNormalizer.fit(cfg,x,sample_ids=list(range(6)))
    rebuilt = VPFPCoherentMacroNormalizer.from_payload(normalizer.to_payload())
    torch.testing.assert_close(rebuilt.decode(rebuilt.encode(x)),x)
    with pytest.raises(ValueError,match="training"):
        VPFPCoherentMacroNormalizer.fit(cfg,x,sample_ids=list(range(6)),split="validation")
    with pytest.raises(ValueError,match="training"):
        VPFPCoherentMacroNormalizer.fit(cfg,x,sample_ids=[0]*6)


@pytest.mark.parametrize("conditional",[False,True])
def test_score_loss_finite_and_sampler_grouping_invariant(conditional):
    value_cfg, macro_cfg = configs()
    torch.manual_seed(19)
    model = VPFPMacroConditionalEDM(value_cfg,macro_cfg) if conditional else VPFPCoherentMacroEDM(macro_cfg)
    cfg = model.config
    conditions = torch.tensor([[.03,1.],[.07,1.1]])
    if conditional:
        conditions = torch.cat((conditions,torch.randn(2,macro_cfg.latent_dimension)),dim=1)
    x = torch.randn(2,cfg.latent_dimension)
    loss,_ = model.loss(x,conditions,generator=torch.Generator().manual_seed(9))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    batch = model.sample(conditions,[44,45],steps=2)
    single = torch.cat([model.sample(conditions[i:i+1],[44+i],steps=2) for i in range(2)])
    torch.testing.assert_close(batch,single,rtol=0,atol=0)


def test_conditional_input_required_and_high_sigma_skip_unchanged():
    value_cfg, macro_cfg = configs()
    model = VPFPMacroConditionalEDM(value_cfg,macro_cfg)
    x = torch.randn(2,value_cfg.latent_dimension)
    theta = torch.tensor([[.03,1.],[.07,1.1]])
    with pytest.raises(ValueError,match="explicit macro"):
        model.loss(x,theta)
    sigma = torch.tensor([1.,80.])
    conditions = torch.cat((theta,torch.randn(2,macro_cfg.latent_dimension)),dim=1)
    # Zero-initialized output is exactly the standard EDM skip at every sigma.
    expected = value_cfg.sigma_data**2/(sigma.square()+value_cfg.sigma_data**2)
    torch.testing.assert_close(model.denoise(x,sigma,conditions),x*expected[:,None])


def test_hierarchical_inference_generates_macro_and_uses_separate_streams():
    value_cfg, macro_cfg = configs()
    macro_model = VPFPCoherentMacroEDM(macro_cfg)
    macro_norm = VPFPCoherentMacroNormalizer.fit(macro_cfg,
        torch.randn(4,macro_cfg.latent_dimension),sample_ids=[1,2,3,4])
    micro_model = VPFPMacroConditionalEDM(value_cfg,macro_cfg)
    class Identity:
        def decode(self,x):
            return x
    out = sample_hierarchical(macro_model,macro_norm,micro_model,Identity(),
        [[.04,1.]],[909],steps=2)
    assert out["macro_coordinates"].shape == (1,macro_cfg.latent_dimension)
    assert out["full_coefficients"].shape == (1,value_cfg.latent_dimension)
    assert out["macro_noise_seeds"] != out["micro_noise_seeds"]
    assert out["scientific_qualification"] is False


def test_macro_config_serialization_and_alias_rejection():
    _, cfg = configs()
    assert VPFPCoherentMacroConfig.from_dict(cfg.to_dict()) == cfg
    with pytest.raises(ValueError,match="aliased"):
        replace(cfg,maximum_mode=3)
