"""Scientific regression for internal polygon seams, not just derivative parity."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import sys

import numpy as np
import pytest

torch = pytest.importorskip('torch')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'legacy/ns/src'))
from agent_verifier.neural_generator import structured_ns_answer_first as mms
from agent_verifier.neural_generator.continuous_geometry import ContinuousGeometrySpec
from agent_verifier.neural_generator.structured_ns_analytic import analytic_quantities
from agent_verifier.neural_generator.structured_ns_geometry_jets import geometry_mask_derivatives

V5 = mms.STRUCTURED_NS_MMS_ABI_VERSION
V6 = mms.INTERIOR_C3_STRUCTURED_NS_MMS_ABI_VERSION
BOUNDS = (.0078125, .9921875, .0078125, .9921875)
POLYGONS = (((.66, .35), (.79, .43), (.69, .535)),
            ((.315, .535), (.438, .69), (.565, .515), (.405, .365)))


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def field(abi=V6, *, constant=False):
    config = mms.StructuredNSProjectionConfig(spatial_mode_x=3, spatial_mode_y=2, time_order=3)
    generator = torch.Generator().manual_seed(101)
    psi = torch.randn((4, 12), generator=generator, dtype=torch.float64) * 1e-4
    pressure = torch.randn((4, 12), generator=generator, dtype=torch.float64) * 3e-4
    if constant:
        psi.zero_(); pressure.zero_(); psi[0, 0] = 1e-3
    return mms.ContinuousStructuredNSMMS(
        mms.StructuredNSCaseCoefficients(psi, pressure), config=config,
        geometry_spec=ContinuousGeometrySpec(BOUNDS, POLYGONS), domain_bounds=BOUNDS,
        viscosity=.01, t0=0., t1=.12, continuous_mms_abi=abi, device='cpu',
    )


def seam(field, polygon=0, edge_index=0, location=1.1):
    vertices = torch.tensor(field.geometry_spec.normalized_obstacle_polygons()[polygon], dtype=torch.float64)
    a, b = vertices[edge_index], vertices[(edge_index + 1) % len(vertices)]
    edge = b - a
    area = (vertices[:, 0]*vertices.roll(-1, 0)[:, 1] - vertices[:, 1]*vertices.roll(-1, 0)[:, 0]).sum()
    normal = torch.sign(area) * torch.stack((edge[1], -edge[0])) / torch.linalg.vector_norm(edge)
    return a + location*edge, normal


def mask_crossing(field, anchor, normal, epsilon):
    offset = torch.tensor([-epsilon, epsilon], dtype=torch.float64, requires_grad=True)
    points = anchor[None, :] + offset[:, None]*normal[None, :]
    mask, fluid = field.geometry_masks(points)
    derivatives = [mask]
    for _ in range(3):
        derivatives.append(torch.autograd.grad(derivatives[-1].sum(), offset, create_graph=True)[0])
    assert fluid.tolist() == [1., 1.]
    return torch.stack(derivatives).detach().numpy()


@pytest.mark.parametrize('polygon,edge_index', [(p,e) for p,poly in enumerate(POLYGONS) for e in range(len(poly))])
@pytest.mark.parametrize('location', (-.1, 1.1))
def test_all_internal_edge_extensions_have_continuous_derivatives_through_three(polygon, edge_index, location):
    f = field()
    anchor, normal = seam(f, polygon, edge_index, location)
    coarse = mask_crossing(f, anchor, normal, 1e-8)
    fine = mask_crossing(f, anchor, normal, 1e-10)
    coarse_gap = np.abs(coarse[:, 1] - coarse[:, 0])
    fine_gap = np.abs(fine[:, 1] - fine[:, 0])
    assert np.all(fine_gap < .03*coarse_gap + 1e-7)
    assert np.all(fine_gap < 1e-4*(1 + np.abs(fine).max(axis=1)))


def test_original_counterexample_fails_v5_and_passes_v6():
    old, new = field(V5), field(V6)
    anchor, normal = seam(old)
    prior = mask_crossing(old, anchor, normal, 1e-8)
    fixed = mask_crossing(new, anchor, normal, 1e-8)
    assert abs(prior[2, 1] - prior[2, 0]) > 4000
    assert abs(fixed[2, 1] - fixed[2, 0]) < .01


@pytest.mark.parametrize('polygon,edge_index', [(p,e) for p,poly in enumerate(POLYGONS) for e in range(len(poly))])
def test_quadratic_flat_wall_trace_and_nonzero_shear_are_preserved(polygon, edge_index):
    old, new = field(V5), field(V6)
    anchor, normal = seam(new, polygon, edge_index, .5)
    offsets = torch.tensor([1e-8, 1e-6, 1e-4], dtype=torch.float64, requires_grad=True)
    points = anchor[None, :] + offsets[:, None]*normal[None, :]
    reference, indicator = old.geometry_masks(points)
    actual, actual_indicator = new.geometry_masks(points)
    torch.testing.assert_close(actual, reference, atol=3e-16, rtol=1e-8)
    torch.testing.assert_close(actual_indicator, indicator)
    first = torch.autograd.grad(actual.sum(), offsets, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), offsets)[0]
    assert float(second[0]) == pytest.approx(20/.05**2, rel=2e-6)
    # Velocity is curl(psi); its boundary trace is zero on every true face.
    xmin,xmax,ymin,ymax = new.domain_bounds
    xy = [xmin + anchor[0]*(xmax-xmin), ymin + anchor[1]*(ymax-ymin), .03]
    q = analytic_quantities(new, torch.tensor([xy], dtype=torch.float64))
    assert abs(float(q['u'][0])) < 1e-12
    assert abs(float(q['v'][0])) < 1e-12


def pillbox_balance(f, *, half_normal=1e-4, half_tangent=.002, order=16):
    """Integral conservative NS balance across the artificial interface.

    Boundary: integral [u(u.n) + p*n - nu*grad(u).n].
    Volume: integral (f - u_t). Split quadrature at the seam so a jump in the
    old pointwise forcing cannot be mistaken for integration error.
    """
    anchor, n = seam(f)
    xmin,xmax,ymin,ymax = f.domain_bounds
    anchor = np.array([xmin + anchor[0]*(xmax-xmin), ymin + anchor[1]*(ymax-ymin)])
    n = n.numpy(); tangent = np.array([-n[1], n[0]])
    nodes, weights = np.polynomial.legendre.leggauss(order)
    def quantities(xy):
        points = np.column_stack((xy, np.full(len(xy), .047)))
        return {k:v.detach().numpy() for k,v in analytic_quantities(f, points).items()}
    boundary = np.zeros(2)
    for normal, varying, fixed_half, varying_half in ((n,tangent,half_normal,half_tangent),
                                                      (tangent,n,half_tangent,half_normal)):
        for side in (-1.,1.):
            xy = anchor + side*fixed_half*normal + varying_half*nodes[:,None]*varying
            q = quantities(xy)
            outward = side*normal
            velocity = np.column_stack((q['u'],q['v']))
            gradient_n = np.column_stack((q['d_x_u']*outward[0]+q['d_y_u']*outward[1],
                                           q['d_x_v']*outward[0]+q['d_y_v']*outward[1]))
            flux = velocity*(velocity@outward)[:,None] + q['pressure'][:,None]*outward - f.viscosity*gradient_n
            boundary += varying_half*(weights[:,None]*flux).sum(axis=0)
    volume = np.zeros(2)
    for side in (-1.,1.):
        nn = side*half_normal*(nodes+1.)/2.
        tt = half_tangent*nodes
        ns,ts = np.meshgrid(nn,tt,indexing='ij')
        xy = anchor + ns.reshape(-1,1)*n + ts.reshape(-1,1)*tangent
        q = quantities(xy)
        density = np.column_stack((q['forcing_u']-q['d_t_u'],q['forcing_v']-q['d_t_v']))
        w = (weights[:,None]*weights[None,:]).reshape(-1)
        volume += half_normal/2*half_tangent*(w[:,None]*density).sum(axis=0)
    residual = boundary-volume
    return {'boundary':boundary.tolist(), 'volume':volume.tolist(), 'residual':residual.tolist(),
            'residual_l2':float(np.linalg.norm(residual)), 'half_normal':half_normal,
            'half_tangent':half_tangent, 'gauss_order':order}


@pytest.mark.parametrize('constant', (True,False))
def test_weak_momentum_balance_no_longer_omits_interface_force(constant):
    before = pillbox_balance(field(V5,constant=constant))
    after = pillbox_balance(field(V6,constant=constant))
    assert before['residual_l2'] > 1e-6
    assert after['residual_l2'] < 1e-10


@pytest.mark.parametrize('abi',(V5,V6))
def test_serialization_adapter_and_pressure_source_roundtrip(abi,tmp_path):
    from pde_agent_verifier_solver.plugins.ns.continuous_backend import StructuredNSMMSAnswer
    f = field(abi)
    mms.write_structured_continuous_case_state(tmp_path,field=f)
    loaded = mms.load_structured_continuous_case(tmp_path)
    assert loaded.continuous_mms_abi == abi
    manifest = json.loads((tmp_path/'hidden_continuous_manifest.json').read_text())
    assert manifest['continuous_mms_abi'] == abi
    points = np.array([[.803,.438,.03],[.8,.46,.09],[.1,.2,.07],[.7,.44,.05]])
    expected = analytic_quantities(f,points)
    actual = analytic_quantities(loaded,points)
    for key in expected:
        torch.testing.assert_close(actual[key],expected[key],atol=0,rtol=0)
    adapter = StructuredNSMMSAnswer(loaded)
    np.testing.assert_allclose(adapter.evaluate_forcing(points[:,[2,0,1]]),
                               torch.stack((actual['forcing_u'],actual['forcing_v']),dim=-1),rtol=1e-13,atol=1e-13)
    assert adapter._analytic_field


def test_v6_point_queries_are_finite_at_solid_corners_tiny_offsets_and_saturation():
    f = field()
    points = torch.tensor([[.3,.3,0.],[0.,0.,0.],[1.,1.,1.],[.5,.5,.5]],dtype=torch.float64)
    points = torch.cat((points, torch.tensor([[x,y,.3] for p in f.geometry_spec.normalized_obstacle_polygons() for x,y in p])))
    for delta in (-1e-12,0.,1e-12):
        q = (points + delta).requires_grad_()
        mask,_ = f.geometry_masks(q)
        for _ in range(3):
            assert torch.isfinite(mask).all()
            mask = torch.autograd.grad(mask.sum(),q,create_graph=True)[0].sum(-1)
        assert torch.isfinite(mask).all()
        jets,_ = geometry_mask_derivatives(f,q)
        assert all(torch.isfinite(value).all() for value in jets.values())
    jets,fluid = geometry_mask_derivatives(f,torch.empty((0,3),dtype=torch.float64))
    assert fluid.numel()==0 and all(v.numel()==0 for v in jets.values())


def test_projection_uses_the_requested_field_mask():
    f=field()
    x=np.linspace(BOUNDS[0],BOUNDS[1],25); y=x.copy()
    xx,yy=np.meshgrid(x,y,indexing='ij'); xy=np.column_stack((xx.ravel(),yy.ravel(),np.zeros(xx.size)))
    mask=f.geometry_masks(f._normalize_coordinates(torch.tensor(xy)))[1].numpy().reshape(xx.shape)>0
    geometry=SimpleNamespace(config=SimpleNamespace(nx=len(x),ny=len(y)),x=x,y=y,fluid_mask=mask,
                             to_json_dict=lambda:{'continuous_geometry':f.geometry_spec.to_dict()})
    projector=mms.StructuredNSCoefficientProjector(geometry=geometry,t=np.linspace(0,.12,6),config=f.config,
                                                  continuous_mms_abi=V6,device='cpu')
    # t=0 Chebyshev coefficients are (-1)^degree.
    coefficients=sum(f.coefficients.psi[i]*((-1)**i) for i in range(4))
    predicted=projector.spatial_operators['velocity_design']@coefficients
    q=analytic_quantities(f,xy)
    selected=torch.as_tensor(mask.ravel())
    actual=torch.cat((q['u'][selected],q['v'][selected]))
    torch.testing.assert_close(predicted,actual,atol=1e-12,rtol=1e-12)
    old=mms.StructuredNSCoefficientProjector(geometry=geometry,t=np.linspace(0,.12,6),config=f.config,device='cpu')
    assert torch.max(torch.abs(old.spatial_operators['velocity_design']-projector.spatial_operators['velocity_design']))>1


def artifact_pair(tmp_path,abi=V5,flow_abi=None):
    from agent_verifier.neural_generator.structured_ns_distribution import (
        StructuredCoefficientPCA,StructuredCoefficientFlowConfig,StructuredCoefficientRectifiedFlow,STRUCTURED_NS_FLOW_SCHEMA,
    )
    config=mms.StructuredNSProjectionConfig(spatial_mode_x=1,spatial_mode_y=1,time_order=1)
    signature={'projection_config':config.to_dict(),'continuous_mms_abi':abi}
    data=torch.randn((4,2,2,2,2),generator=torch.Generator().manual_seed(9))
    pca_path=tmp_path/'pca.pt'
    StructuredCoefficientPCA.fit(data,rank=2,source_signature=signature).save(pca_path)
    cfg=StructuredCoefficientFlowConfig(feature_dim=2,width=8,depth=1,time_width=4)
    model=StructuredCoefficientRectifiedFlow.build(cfg)
    payload={'schema':STRUCTURED_NS_FLOW_SCHEMA,'pca_sha256':hashlib.sha256(pca_path.read_bytes()).hexdigest(),
             'pca_rank':2,'model_config':cfg.to_dict(),'ema_state_dict':model.state_dict()}
    if flow_abi is not None: payload['continuous_mms_abi']=flow_abi
    flow_path=tmp_path/'flow.pt';torch.save(payload,flow_path)
    return {'pca_checkpoint':pca_path,'flow_checkpoint':flow_path,'time_interval':(0.,.12),'viscosity':.01,
            'device':'cpu','geometry_spec':ContinuousGeometrySpec(BOUNDS,POLYGONS).to_dict(),'integration_steps':1}


@pytest.mark.parametrize('abi',(V5,V6))
def test_hash_bound_artifact_contract_is_preserved_through_field_materialization(abi,tmp_path):
    from pde_agent_verifier_solver.plugins.ns.structured_flow_runtime import LegacyStructuredNSFlowSampler
    sampler=LegacyStructuredNSFlowSampler(**artifact_pair(tmp_path,abi))
    # Small synthetic checkpoint only; no production Generator/model request.
    coefficient_tensor=torch.zeros((2,2,2,2))
    f=sampler._field_from_coefficients(coefficient_tensor)
    assert f.continuous_mms_abi==abi==sampler.artifact_metadata['continuous_mms_abi']
    other=field(V5 if abi==V6 else V6)
    with pytest.raises(ValueError,match='ABIs differ'):
        sampler.encode_fields([other])


def test_v5_weights_cannot_be_silently_requested_as_v6(tmp_path):
    from pde_agent_verifier_solver.plugins.ns.structured_flow_runtime import LegacyStructuredNSFlowSampler
    with pytest.raises(ValueError,match='artifact contract'):
        LegacyStructuredNSFlowSampler(**artifact_pair(tmp_path,V5),continuous_mms_abi=V6)
    with pytest.raises(ValueError,match='ABIs differ'):
        LegacyStructuredNSFlowSampler(**artifact_pair(tmp_path,V6,flow_abi=V5))


def test_v5_and_v6_projection_caches_are_not_compatible():
    from agent_verifier.neural_generator.structured_ns_distribution import compatible_projection_signatures
    legacy={'projection_config':{},'coefficient_shape':[2,2,2,2],'split_manifest_sha256':'same'}
    v5={**legacy,'continuous_mms_abi':V5};v6={**legacy,'continuous_mms_abi':V6}
    assert compatible_projection_signatures(legacy,v5)
    assert compatible_projection_signatures(v6,v6)
    assert not compatible_projection_signatures(legacy,v6)
    assert not compatible_projection_signatures(v5,v6)
