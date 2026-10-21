"""Independent dense normal-equation and holdout checks for the joint fitter."""
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'legacy/ns/src'))
from agent_verifier.neural_generator.structured_ns_answer_first import StructuredNSProjectionConfig
from agent_verifier.neural_generator.structured_ns_spline_projection import JointSmoothNSCoefficientProjector
from pde_agent_verifier_solver.plugins.ns.completed_fields import completed_grid

@pytest.fixture
def problem():
    torch.set_num_threads(1)
    x,y=np.meshgrid(np.linspace(-1,2,12),np.linspace(.2,1.4,11),indexing='ij')
    geometry,_=completed_grid(x.ravel(),y.ravel(),{'obstacle_polygons':[],'outer_boundary':'manufactured_trace'})
    cfg=StructuredNSProjectionConfig(spatial_mode_x=5,spatial_mode_y=5,time_order=3,
        spatial_basis='bspline',temporal_basis='bspline',fitting_method='joint_smoothing')
    return JointSmoothNSCoefficientProjector(geometry=geometry,t=np.linspace(.3,.9,8),config=cfg)

def test_joint_matches_dense_normal_equations(problem):
    p=problem; rng=np.random.default_rng(918)
    y=torch.tensor(rng.normal(size=(2,8,2,132)),dtype=torch.float64)
    s=p.prepare_system(); ls,lt=1e-5,1e-4
    c,diag=p.solve(y,system=s,smoothing_spatial=ls,smoothing_temporal=lt,tolerance=1e-10)
    a=torch.kron(s['Gt'].contiguous(),s['Gv'].contiguous())+ls*torch.kron(p.time_mass.contiguous(),p.spatial_roughness.contiguous())+lt*torch.kron(p.time_roughness.contiguous(),p.velocity_mass.contiguous())
    a+=torch.kron(torch.eye(4,dtype=torch.float64),torch.outer(p.gauge,p.gauge))
    rhs=s['T'].T@y.reshape(2,8,-1)@s['V']/(8*264)
    rhs=p._gauge_fix(rhs)
    exact=torch.linalg.solve(a,rhs.reshape(2,-1).T).T.reshape(2,4,36)
    torch.testing.assert_close(c,exact,rtol=1e-7,atol=2e-9)
    assert max(diag['normal_relative_residual'])<1e-8

def test_unobserved_values_never_enter_fit(problem):
    p=problem; rng=np.random.default_rng(26)
    truth=p._gauge_fix(torch.tensor(rng.normal(size=(1,4,36)),dtype=torch.float64))
    target=p.predict(truth)
    tm=np.ones(8,bool);tm[3]=False
    sm=np.ones(132,bool);sm[::7]=False
    system=p.prepare_system(time_mask=tm,spatial_mask=sm)
    clean,_=p.solve(target,system=system)
    poisoned=target.clone();poisoned[:,~tm]=float('nan');poisoned[:,:,:,~sm]=float('nan')
    actual,_=p.solve(poisoned,system=system)
    torch.testing.assert_close(actual,clean,rtol=0,atol=0)
    torch.testing.assert_close(p.predict(actual),target,rtol=1e-6,atol=2e-7)
    assert float((actual@p.gauge).abs().max())<1e-12

def test_penalty_agrees_with_direct_velocity_hessian_quadrature(problem):
    from agent_verifier.neural_generator.structured_ns_spline_projection import axis_quadrature
    from agent_verifier.neural_generator.structured_ns_basis import axis_basis_derivatives
    p=problem; rng=np.random.default_rng(4)
    c=torch.tensor(rng.normal(size=(6,6)),dtype=torch.float64)
    x,wx=axis_quadrature(p.config,'x');y,wy=axis_quadrature(p.config,'y')
    bx=axis_basis_derivatives(x,config=p.config,axis='x',order=3)
    by=axis_basis_derivatives(y,config=p.config,axis='y',order=3)
    lx,ly=3.,1.2
    value=0.
    for dx,dy,weight in [(2,1,3),(1,2,3),(0,3,1),(3,0,1)]:
        d=(bx[dx]@c@by[dy].T)/(lx**dx*ly**dy)
        value+=weight*(wx[:,None]*wy[None,:]*d**2).sum()
    expected=.5*(lx*ly)**2*value
    torch.testing.assert_close(c.ravel()@p.spatial_roughness@c.ravel(),expected,rtol=1e-12,atol=1e-8)
