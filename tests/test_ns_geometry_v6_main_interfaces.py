"""Keep the integrated release's existing basis and outer-trace capabilities."""
from dataclasses import replace
import numpy as np
import pytest
from .test_ns_geometry_v6 import field, V5, V6, mms, torch, single_thread
from agent_verifier.neural_generator.structured_ns_analytic import analytic_quantities
from agent_verifier.neural_generator.structured_ns_basis import open_uniform_knots

@pytest.mark.parametrize('abi',(V5,V6))
@pytest.mark.parametrize('outer_boundary',('no_slip','manufactured_trace'))
@pytest.mark.parametrize('basis',('cosine_extension','bspline'))
def test_integrated_basis_and_boundary_derivatives_survive_patch(abi,outer_boundary,basis,tmp_path):
    base=field(abi)
    if basis=='cosine_extension':
        config=replace(base.config,spatial_extension=.4)
    else:
        config=mms.StructuredNSProjectionConfig(
            spatial_mode_x=6,spatial_mode_y=7,time_order=4,
            spatial_basis='bspline',temporal_basis='bspline',spatial_degree=5,temporal_degree=3,
            spatial_knots_x=open_uniform_knots(7,5),spatial_knots_y=open_uniform_knots(8,5),
            temporal_knots=open_uniform_knots(5,3))
    generator=torch.Generator().manual_seed(101)
    shape=(config.time_coefficient_count,config.spatial_coefficient_count)
    coeff=mms.StructuredNSCaseCoefficients(torch.randn(shape,generator=generator,dtype=torch.float64)*1e-4,
                                         torch.randn(shape,generator=generator,dtype=torch.float64)*3e-4)
    f=mms.ContinuousStructuredNSMMS(coeff,config=config,
        geometry_spec=replace(base.geometry_spec,outer_boundary=outer_boundary),continuous_mms_abi=abi,
        domain_bounds=base.domain_bounds,viscosity=base.viscosity,t0=base.t0,t1=base.t1,device='cpu')
    x0,x1,y0,y1=f.domain_bounds
    points=np.array([[.803,.438,.03],[.8,.46,.09],[.7,.44,.05],[x0,.2,0.],[x1,.6,.12],
                     [.3,y0,.06],[.3,y1,.06],[.5,.5,.06],[.5,.4,.12]])
    expected=f.quantities(points,include_laplacian_components=True)
    actual=analytic_quantities(f,points,include_laplacian_components=True)
    for key in expected:
        torch.testing.assert_close(actual[key],expected[key],rtol=2e-10,atol=1e-9,msg=key)
    if outer_boundary=='manufactured_trace':
        assert actual['fluid_indicator'][3:7].tolist()==[1.]*4
        assert f.boundary_condition['access']=='query_boundary_velocity'
    mms.write_structured_continuous_case_state(tmp_path,field=f)
    loaded=mms.load_structured_continuous_case(tmp_path)
    assert loaded.config.to_dict()==f.config.to_dict()
    assert loaded.geometry_spec.to_dict()==f.geometry_spec.to_dict()
    assert loaded.continuous_mms_abi==abi
    for key,value in analytic_quantities(loaded,points,include_laplacian_components=True).items():
        torch.testing.assert_close(value,actual[key],rtol=0,atol=0,msg=key)
