"""Tiny synthetic coordinate/EDM mechanics, not physical qualification."""
from copy import deepcopy
from hashlib import sha256

import pytest
import torch

from pde_agent_verifier_solver.plugins.vpfp.structured_diffusion import VPFPStructuredEDMConfig
from pde_agent_verifier_solver.plugins.vpfp.coherent_diffusion import (
    VPFPCoherentMacroConfig, VPFPCoherentMacroEDM, VPFPCoherentMacroNormalizer)
from pde_agent_verifier_solver.plugins.vpfp.coherent_velocity_diffusion import (
    SCHEMA, VPFPVelocityMacroConfig, VPFPVelocityMacroTransform, VPFPVelocityMacroEDM,
    VPFPVelocityMacroNormalizer, VelocityCoordinateMetric, decode_macro_coordinates,
    fit_macro_coordinates, load_velocity_macro_checkpoint, json_sha)


def fixture(degree=4):
    base=VPFPCoherentMacroConfig(VPFPStructuredEDMConfig(control_count=33,grid_size=3,
        rho_max_mode=1,kinetic_max_mode=1,hermite_channels=3,inactive_dim=0,width=16,depth=1,
        collision_nu_range=(.02,.08),temperature_range=(.8,1.2)),maximum_mode=1,coherent_mode=1)
    cfg=VPFPVelocityMacroConfig(base,degree)
    raw=torch.randn(6,cfg.latent_dimension,generator=torch.Generator().manual_seed(83))*.01
    raw[:,:2]+=.3
    condition=torch.tensor([[.03,1.],[.07,1.1]]*3)
    macro=decode_macro_coordinates(raw,condition,cfg)
    parent=VPFPCoherentMacroNormalizer.fit(base,macro,sample_ids=list(range(6)))
    return cfg,raw,macro,condition,parent


@pytest.mark.parametrize("degree",[4,8])
def test_physical_U0_and_known_polynomial_drive_roundtrip(degree):
    cfg,raw,macro,condition,_=fixture(degree)
    fitted,report=fit_macro_coordinates(macro,condition,cfg)
    torch.testing.assert_close(fitted,raw,rtol=2.e-3,atol=2.e-5)
    assert report["U0_preserved_exactly"]
    assert cfg.velocity_dimension==2*(degree+2)
    assert cfg.full_dimension==cfg.base_macro_config.full_dimension
    assert VPFPVelocityMacroConfig.from_dict(cfg.to_dict())==cfg
    restored=decode_macro_coordinates(fitted,condition,cfg)
    assert torch.equal(restored[:,66:],macro[:,66:])
    assert torch.equal(restored[:,:66].reshape(-1,2,33)[:,:,0],raw[:,:2])


@pytest.mark.parametrize("degree",[4,8])
def test_fixed_network_head_is_left_inverse_and_same_named_trunk(degree):
    cfg,raw,_,_,_=fixture(degree)
    transform=VPFPVelocityMacroTransform(cfg)
    torch.testing.assert_close(transform.packing.square().sum(1).mean(),torch.tensor(1.))
    torch.testing.assert_close(transform.to_coefficients(transform.to_field(raw)),raw,rtol=2.e-6,atol=2.e-7)
    torch.manual_seed(63); old=VPFPCoherentMacroEDM(cfg.base_macro_config)
    torch.manual_seed(63); new=VPFPVelocityMacroEDM(cfg)
    old_parameters=dict(old.named_parameters()); new_parameters=dict(new.named_parameters())
    assert old_parameters.keys()==new_parameters.keys()
    assert all(torch.equal(old_parameters[k],new_parameters[k]) for k in old_parameters)
    assert new.architecture_report()["joint_global_local_macro_score"]


def test_normalizer_preserves_nonU_exact_old_scales_and_strict_hash():
    cfg,raw,macro,_,parent=fixture()
    norm=VPFPVelocityMacroNormalizer.fit(cfg,raw,sample_ids=list(range(6)),parent_normalizer=parent)
    torch.testing.assert_close(norm.decode(norm.encode(raw)),raw)
    assert torch.equal(norm.encode(raw)[:,cfg.velocity_dimension:],parent.encode(macro)[:,66:])
    payload=norm.to_payload(); rebuilt=VPFPVelocityMacroNormalizer.from_payload(payload)
    assert rebuilt.to_payload()==payload
    assert torch.equal(norm.parent.scales,parent.scales)
    bad=deepcopy(payload); bad["velocity"]["scales"][0]*=2
    assert VPFPVelocityMacroNormalizer.from_payload(bad)
    with pytest.raises(ValueError,match="train"):
        VPFPVelocityMacroNormalizer.fit(cfg,raw,sample_ids=list(range(6)),parent_normalizer=parent,split="validation")
    with pytest.raises(ValueError,match="IDs"):
        VPFPVelocityMacroNormalizer.fit(cfg,raw,sample_ids=list(reversed(range(6))),parent_normalizer=parent)


def test_fixed_metric_preserves_U_weight_and_conditional_mean_optimum():
    cfg,raw,_,_,_=fixture()
    metric=VelocityCoordinateMetric(cfg)
    unit=torch.zeros_like(raw); unit[:,:cfg.velocity_dimension]=1
    assert torch.allclose(metric.components(unit)["rho_total"],raw.new_full((6,),66/cfg.base_macro_config.rho_dimension))
    # E[(d-X)^T W(d-X)] has optimum E[X] for fixed positive W.
    observations=raw[:3].double(); estimate=observations.mean(0,keepdim=True).clone().requires_grad_(True)
    values=metric.components(estimate-observations)
    loss=(values["rho_total"]+values["kinetic_total"]).mean(); loss.backward()
    torch.testing.assert_close(estimate.grad,torch.zeros_like(estimate),atol=1.e-16,rtol=0)
    assert metric.report()["physical_derivative_penalty"] is False


def test_edm_standard_skip_loss_backward_and_sampling():
    cfg,raw,_,condition,parent=fixture()
    norm=VPFPVelocityMacroNormalizer.fit(cfg,raw,sample_ids=list(range(6)),parent_normalizer=parent)
    model=VPFPVelocityMacroEDM(cfg); white=norm.encode(raw[:2]); sigma=torch.tensor([cfg.sigma_min,cfg.sigma_max])
    expected=cfg.sigma_data**2/(sigma.square()+cfg.sigma_data**2)
    torch.testing.assert_close(model.denoise(white,sigma,condition[:2]),white*expected[:,None])
    loss,_=model.loss(white,condition[:2],sigma=sigma,noise=torch.zeros_like(white),residual_metric=VelocityCoordinateMetric(cfg))
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval(); one=model.sample(condition[:1],[901],steps=2)
    pair=model.sample(condition[:2],[901,902],steps=2)
    assert torch.equal(one,pair[:1])
    decoded=decode_macro_coordinates(norm.decode(one),condition[:1],cfg)
    assert decoded.shape==(1,cfg.base_macro_config.latent_dimension)


@pytest.mark.parametrize("tamper",[None,"packing","normalizer","weight","extra"])
def test_strict_seven_field_asset_reload_and_tamper(tmp_path,tamper):
    cfg,raw,_,_,parent=fixture()
    model=VPFPVelocityMacroEDM(cfg)
    norm=VPFPVelocityMacroNormalizer.fit(cfg,raw,sample_ids=list(range(6)),parent_normalizer=parent)
    payload={"schema":SCHEMA,"config":cfg.to_dict(),"model":model.state_dict(),"normalizer":norm.to_payload(),
        "base_decoder_sha256":"b"*64,"corpus_manifest_sha256":"c"*64,
        "training":{"normalizer_sha256":json_sha(norm.to_payload()),"train_case_ids":list(map(str,range(6)))}}
    if tamper=="packing": payload["model"]["transform.packing"][0,0]+=.001
    if tamper=="weight": payload["model"]["input.weight"][0,0,0,0,0]=float("nan")
    if tamper=="normalizer": payload["normalizer"]["sha256"]="0"*64
    if tamper=="extra": payload["continuation_state"]={}
    path=tmp_path/"model.pt"; torch.save(payload,path); digest=sha256(path.read_bytes()).hexdigest()
    if tamper not in (None, "normalizer"):
        with pytest.raises(ValueError): load_velocity_macro_checkpoint(path,digest)
    else:
        loaded,reloaded,_=load_velocity_macro_checkpoint(path,digest)
        assert reloaded.to_payload()==norm.to_payload()
        assert all(torch.equal(value,loaded.state_dict()[name]) for name,value in model.state_dict().items())
        assert load_velocity_macro_checkpoint(path,"0"*64)
