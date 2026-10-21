"""Small numerical fixtures: execute only in the allocated compute-node gate."""
from copy import deepcopy
import math

import pytest
import torch

from pde_agent_verifier_solver.plugins.vpfp.coherent_velocity_dynamics import (
    VPFPVelocityDynamicsConfig as Config,VPFPVelocityDynamicsNormalizer as Normalizer,
    evaluate_velocity_dynamics as evaluate,fit_velocity_dynamics as fit,
)


@pytest.mark.parametrize("nu",[0.,1.e-12,.03,20.,10000.])
def test_constant_nonzero_drive_exact_solution_and_initial(nu):
    cfg=Config(drive_degree=4,t0=2.,t1=3.)
    coordinates=torch.zeros((1,cfg.latent_dimension),dtype=torch.float64)
    coordinates[0,:2]=torch.tensor([1.2,-.4],dtype=torch.float64)
    coordinates[0,2]=.7; coordinates[0,7]=-.3
    t=torch.tensor([2.,2.001,2.25,2.75,3.],dtype=torch.float64)
    elapsed=t-2.; decay=torch.exp(-nu*elapsed)
    integral=elapsed if nu==0 else -torch.expm1(-nu*elapsed)/nu
    expected=coordinates[:,:2,None]*decay+torch.tensor([.7,-.3],dtype=torch.float64)[None,:,None]*integral
    result=evaluate(coordinates,[nu],t,cfg)
    torch.testing.assert_close(result["U"],expected,rtol=2.e-12,atol=2.e-13)
    assert torch.equal(result["U"][:,:,0],coordinates[:,:2])
    torch.testing.assert_close(result["dU_dt"],result["a"]-nu*result["U"],rtol=0,atol=0)
    assert torch.isfinite(result["U"]).all()


@pytest.mark.parametrize("degree",[4,8,16])
def test_in_basis_fit_recovers_initial_and_off_node_truth(degree):
    cfg=Config(drive_degree=degree)
    gen=torch.Generator().manual_seed(42)
    coordinates=torch.randn((3,cfg.latent_dimension),generator=gen,dtype=torch.float64)*.1
    nu=torch.tensor([0.,.1,50.],dtype=torch.float64)
    nodes=torch.linspace(0,1,33,dtype=torch.float64)
    trajectory=evaluate(coordinates,nu,nodes,cfg,quadrature_order=128)["U"]
    fitted=fit(trajectory,nu,cfg)
    torch.testing.assert_close(fitted.coordinates,coordinates,rtol=3.e-9,atol=3.e-10)
    offnodes=(torch.arange(32,dtype=torch.float64)+.25)/32
    truth=evaluate(coordinates,nu,offnodes,cfg,quadrature_order=128)["U"]
    restored=evaluate(fitted.coordinates,nu,offnodes,cfg)["U"]
    torch.testing.assert_close(restored,truth,rtol=3.e-9,atol=3.e-10)
    assert fitted.diagnostics["U0_preserved_exactly"]
    assert not fitted.diagnostics["off_node_reference_error_measured"]
    assert not fitted.diagnostics["full_support_in_original_33_node_space"]


@pytest.mark.parametrize("nu",[0.,1.e-10,.3,80.])
def test_ou_derivative_agrees_with_autograd_and_finite_difference(nu):
    cfg=Config(drive_degree=16)
    coefficients=torch.sin(torch.arange(cfg.latent_dimension,dtype=torch.float64))[None,:]*.1
    times=torch.tensor([.0703125,.2578125,.7578125],dtype=torch.float64,requires_grad=True)
    value=evaluate(coefficients,[nu],times,cfg)
    for component in range(2):
        derivative=torch.autograd.grad(value["U"][0,component].sum(),times,retain_graph=True)[0]
        torch.testing.assert_close(derivative,value["dU_dt"][0,component],rtol=2.e-9,atol=2.e-10)
    h=1.e-6
    fd=(evaluate(coefficients,[nu],times.detach()+h,cfg)["U"]-evaluate(coefficients,[nu],times.detach()-h,cfg)["U"])/(2*h)
    torch.testing.assert_close(fd,value["dU_dt"],rtol=2.e-6,atol=2.e-8)


def test_nonperiodic_nonzero_mean_drive_and_legendre_energy():
    cfg=Config(drive_degree=4)
    coefficients=torch.zeros((1,cfg.latent_dimension),dtype=torch.float64)
    coefficients[0,2:4]=torch.tensor([2.,3.],dtype=torch.float64)
    out=evaluate(coefficients,[0.],torch.linspace(0,1,129,dtype=torch.float64),cfg)
    assert out["a"][0,0,0]==-1 and out["a"][0,0,-1]==5
    assert out["drive_mean_square"].item()==pytest.approx(4+9/3)
    assert out["drive_derivative_mean_square"].item()==pytest.approx(36)
    assert out["U"][0,0,-1].item()==pytest.approx(2,abs=1.e-12)


def test_fixed_nu_linear_evaluation_and_coordinate_gradient():
    cfg=Config(drive_degree=8)
    x=torch.arange(cfg.latent_dimension,dtype=torch.float64)[None,:]/100
    y=torch.flip(x,[1]); t=torch.linspace(0,1,9,dtype=torch.float64)
    a=evaluate(2*x-y,[.4],t,cfg)
    b=evaluate(x,[.4],t,cfg); c=evaluate(y,[.4],t,cfg)
    torch.testing.assert_close(a["U"],2*b["U"]-c["U"],rtol=1.e-12,atol=1.e-13)
    x.requires_grad_(); evaluate(x,[.4],t,cfg)["U"].square().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum()>0


def test_out_of_basis_compression_is_reported_not_hidden():
    cfg=Config(drive_degree=4)
    t=torch.linspace(0,1,33,dtype=torch.float64)
    velocity=torch.stack((torch.sin(2*math.pi*11*t),torch.cos(2*math.pi*11*t)))[None,:]
    fitted=fit(velocity,[.02],cfg)
    assert fitted.diagnostics["node_rms_error"][0]>.1
    assert fitted.diagnostics["U0_preserved_exactly"]
    assert not fitted.diagnostics["scientific_admission"]


def test_train_only_positive_affine_normalization_and_strict_payload():
    cfg=Config(drive_degree=4)
    values=torch.arange(4*cfg.latent_dimension,dtype=torch.float64).reshape(4,-1)/37
    values[:,3]=0.
    normal=Normalizer.fit(cfg,values,sample_ids=[1,2,3,4])
    assert (normal.scales>0).all()
    torch.testing.assert_close(normal.decode(normal.encode(values)),values,rtol=1.e-14,atol=1.e-14)
    restored=Normalizer.from_payload(normal.to_payload())
    assert torch.equal(restored.center,normal.center) and torch.equal(restored.scales,normal.scales)
    payload=deepcopy(normal.to_payload()); payload["scales"][2]*=2
    assert Normalizer.from_payload(payload).scales[2] == payload["scales"][2]
    for ids,split in (([1,2,3,4],"validation"),([1,1,2,3],"train")):
        with pytest.raises(ValueError): Normalizer.fit(cfg,values,sample_ids=ids,split=split)


@pytest.mark.parametrize("kind",["negative_nu","nan","wrong_shape","outside_time","wrong_nodes"])
def test_unsupported_or_malformed_inputs_fail(kind):
    cfg=Config(); coordinates=torch.zeros((1,cfg.latent_dimension),dtype=torch.float64)
    if kind=="negative_nu":
        with pytest.raises(ValueError): evaluate(coordinates,[-.1],[0.,1.],cfg)
    elif kind=="nan":
        coordinates[0,0]=float("nan")
        with pytest.raises(ValueError): evaluate(coordinates,[.1],[0.,1.],cfg)
    elif kind=="wrong_shape":
        with pytest.raises(ValueError): evaluate(coordinates[:,:-1],[.1],[0.,1.],cfg)
    elif kind=="outside_time":
        with pytest.raises(ValueError): evaluate(coordinates,[.1],[-.1,1.],cfg)
    else:
        with pytest.raises(ValueError): fit(torch.zeros((1,2,32),dtype=torch.float64),[.1],cfg)
