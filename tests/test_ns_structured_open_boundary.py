"""The common structured NS representation supports declared measurement windows."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "legacy/ns/src"))
from agent_verifier.neural_generator.continuous_geometry import (
    ContinuousGeometrySpec, continuous_geometry_spec_from_contract,
)
from agent_verifier.neural_generator.structured_ns_answer_first import (
    ContinuousStructuredNSMMS, StructuredNSCaseCoefficients, StructuredNSCoefficientProjector,
    StructuredNSProjectionConfig, load_structured_continuous_case, write_structured_continuous_case_state,
)
from agent_verifier.neural_generator.structured_ns_analytic import analytic_quantities
from agent_verifier.neural_generator.structured_ns_robustness import make_perturbation, field_at_amplitude


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


_BOUNDS = (-2., 3., .3, 1.1)
_T0, _T1 = .2, .57


def _grid(spec, nx=13, ny=12):
    xmin, xmax, ymin, ymax = spec.domain_bounds
    return SimpleNamespace(x=np.linspace(xmin, xmax, nx), y=np.linspace(ymin, ymax, ny),
                           fluid_mask=np.ones((nx, ny), dtype=bool),
                           config=SimpleNamespace(nx=nx, ny=ny),
                           to_json_dict=lambda: {"continuous_geometry": spec.to_dict()})


def _known_solution(points, extension=.25):
    """Independent closed-form NS field, including nonzero outer velocities."""
    points = np.asarray(points)
    x, y, t = points.T
    xmin, xmax, ymin, ymax = _BOUNDS
    length_x = (xmax-xmin) * (1+2*extension)
    length_y = (ymax-ymin) * (1+2*extension)
    phase_x = np.pi * (x-xmin+extension*(xmax-xmin)) / length_x
    phase_y = np.pi * (y-ymin+extension*(ymax-ymin)) / length_y
    kx, ky = np.pi/length_x, np.pi/length_y
    tau = (t-_T0)/(_T1-_T0)
    amplitude = 1 + .4*tau + .1*tau*tau
    amplitude_dt = (.4+.2*tau)/(_T1-_T0)
    sx, cx, sy, cy = np.sin(phase_x), np.cos(phase_x), np.sin(phase_y), np.cos(phase_y)
    u, v = -amplitude*ky*cx*sy, amplitude*kx*sx*cy
    ut, vt = -amplitude_dt*ky*cx*sy, amplitude_dt*kx*sx*cy
    ux, uy = amplitude*kx*ky*sx*sy, -amplitude*ky*ky*cx*cy
    vx, vy = amplitude*kx*kx*cx*cy, -amplitude*kx*ky*sx*sy
    lapu, lapv = -(kx*kx+ky*ky)*u, -(kx*kx+ky*ky)*v
    p = .17*amplitude*np.cos(2*phase_x)
    px = -.34*amplitude*kx*np.sin(2*phase_x)
    fx, fy = ut+u*ux+v*uy+px-.037*lapu, vt+u*vx+v*vy-.037*lapv
    return np.column_stack((u,v)), p, np.column_stack((fx,fy))


def _project_known():
    geometry = _grid(ContinuousGeometrySpec(_BOUNDS, (), outer_boundary="manufactured_trace"))
    times = np.linspace(_T0, _T1, 6)
    tt, xx, yy = np.meshgrid(times, geometry.x, geometry.y, indexing="ij")
    velocity, pressure, _ = _known_solution(np.column_stack((xx.ravel(), yy.ravel(), tt.ravel())))
    velocity = velocity.reshape(len(times), len(geometry.x), len(geometry.y), 2).transpose(0,3,1,2)
    pressure = pressure.reshape(len(times), len(geometry.x), len(geometry.y))
    config = StructuredNSProjectionConfig(spatial_mode_x=3, spatial_mode_y=3, time_order=2,
                                         spatial_extension=.25, ridge_relative=0)
    projector = StructuredNSCoefficientProjector(geometry=geometry, t=times, config=config)
    coefficients, report = projector.project(velocity=velocity, pressure=pressure)
    field = ContinuousStructuredNSMMS(
        StructuredNSCaseCoefficients(coefficients[0,0].flatten(1), coefficients[0,1].flatten(1)),
        config=config, geometry_spec=continuous_geometry_spec_from_contract(geometry),
        domain_bounds=_BOUNDS, viscosity=.037, t0=_T0, t1=_T1)
    return field, report


def test_same_projector_fits_independent_nonzero_boundary_solution():
    field, report = _project_known()
    assert report["cases"][0]["velocity_relative_l2"] < 1e-11
    points = np.array([[-2., .52, .24], [3., .89, .5], [-1., .3, .3], [2., 1.1, .48], [.4, .72, .41]])
    velocity, _, forcing = _known_solution(points)
    actual = analytic_quantities(field, points)
    np.testing.assert_allclose(torch.stack((actual["u"],actual["v"]),dim=1), velocity, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(torch.stack((actual["forcing_u"],actual["forcing_v"]),dim=1), forcing, rtol=1e-9, atol=1e-9)
    assert np.all(np.linalg.norm(velocity[:4],axis=1) > .1)
    assert field.boundary_condition["kind"] == "manufactured_velocity_trace"
    # The expansion changes the basis; it does not enlarge the physical domain.
    assert field.domain_bounds == _BOUNDS


def test_spectral_complement_preserves_initial_velocity_and_first_derivative():
    field, _ = _project_known()
    perturbation = make_perturbation(field, family="spectral_complement", seed=123, spectral_max_mode=8)
    perturbed = field_at_amplitude(field, perturbation, amplitude=.003)
    assert perturbed.config.spatial_extension == .25
    assert perturbed.geometry_spec == field.geometry_spec
    assert perturbation.metadata["definition"] == "streamfunction_spatial_cosine_complement_with_tau_squared_envelope"
    assert perturbation.metadata["low_band_coefficient_leakage_max_abs"] == 0
    assert perturbation.metadata["cosine_expansion_extension"] == .25
    points = np.array([[-2., .52, _T0], [3., .89, _T0], [-1., .3, _T0], [2., 1.1, _T0], [.4,.72,_T0]])
    original, changed = analytic_quantities(field, points), analytic_quantities(perturbed, points)
    for key in ("u", "v", "d_t_u", "d_t_v"):
        torch.testing.assert_close(changed[key], original[key], rtol=1e-10, atol=1e-10)
    assert torch.max(torch.abs(changed["divergence"])) < 1e-12


@pytest.mark.parametrize("extension", [0., .25, .6])
def test_open_geometry_analytic_derivatives_match_autograd_with_obstacle(extension):
    spec = ContinuousGeometrySpec((0.,1.,0.,1.), (((.35,.35),(.65,.35),(.5,.65)),),
                                  outer_boundary="manufactured_trace")
    config = StructuredNSProjectionConfig(spatial_mode_x=4, spatial_mode_y=3, time_order=3,
                                         spatial_extension=extension, geometry_mask_transition=.08)
    generator = torch.Generator().manual_seed(711)
    field = ContinuousStructuredNSMMS(
        StructuredNSCaseCoefficients(torch.randn((4,20),generator=generator,dtype=torch.float64)*1e-3,
                                     torch.randn((4,20),generator=generator,dtype=torch.float64)*2e-3),
        config=config, geometry_spec=spec, domain_bounds=spec.domain_bounds, viscosity=.04, t0=0., t1=.5)
    points = np.array([[0.,.31,.1],[1.,.7,.3],[.19,0.,.2],[.81,1.,.4],
                       [.45,.35,.12],[.5,.5,.2],[.5,.33,.25],[.9,.8,.39]])
    actual, expected = analytic_quantities(field, points, include_laplacian_components=True), field.quantities(points, include_laplacian_components=True)
    for key in expected:
        if key in actual:
            torch.testing.assert_close(actual[key], expected[key], rtol=2e-10, atol=1e-9, msg=key)
    assert abs(float(actual["u"][4])) + abs(float(actual["v"][4])) < 1e-12
    assert abs(float(actual["u"][5])) + abs(float(actual["v"][5])) < 1e-12


def test_state_roundtrip_preserves_open_domain_and_extension(tmp_path):
    field, _ = _project_known()
    write_structured_continuous_case_state(tmp_path, field=field)
    restored = load_structured_continuous_case(tmp_path)
    assert restored.config == field.config
    assert restored.geometry_spec == field.geometry_spec
    assert restored.boundary_condition == field.boundary_condition
    points = np.array([[-2., .61, .2], [.1, .7, .41], [3.,.89,.57]])
    for key, value in analytic_quantities(field, points).items():
        torch.testing.assert_close(analytic_quantities(restored, points)[key], value, rtol=0, atol=0)
    manifest = json.loads((tmp_path / "hidden_continuous_manifest.json").read_text())
    assert "manufactured_outer_trace" in manifest["boundary_contract_version"]
    assert manifest["cosine_expansion_domain_role"] == "basis_only_physical_measurement_domain_unchanged"


def test_old_default_contract_hashes_and_numerical_values_are_unchanged():
    old_config = {"spatial_mode_x":3,"spatial_mode_y":2,"time_order":3,
                  "geometry_mask_transition":.05,"ridge_relative":1e-6,"evaluation_batch_points":4096}
    old_geometry = {"abi_version":"continuous_geometry_convex_polygons_v2",
                    "domain_bounds":list(_BOUNDS),"obstacle_polygons":[]}
    config = StructuredNSProjectionConfig.from_dict(old_config)
    spec = ContinuousGeometrySpec.from_dict(old_geometry)
    digest = lambda value: hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    assert digest(config.to_dict()) == digest(old_config)
    assert digest(spec.to_dict()) == digest(old_geometry)
    assert config.spatial_extension == 0 and spec.outer_boundary == "no_slip"
    values = torch.arange(48,dtype=torch.float64).reshape(4,12)
    field = ContinuousStructuredNSMMS(StructuredNSCaseCoefficients(torch.sin(values)*1e-3,torch.cos(values)*2e-3),
             config=config,geometry_spec=spec,domain_bounds=_BOUNDS,viscosity=.037,t0=_T0,t1=_T1)
    points = np.array([[-2.,.61,.2],[-1.95,.51,.27],[-.5,.64,.31],[1.9,.88,.52],[2.94,.64,.57]])
    # Recorded from the original implementation before adding either option.
    expected = {"u":[0.,-.0002627351960849818,-.0013382164555440518,.010128726172561883,-.011673724625762804],
                "v":[0.,-.0010173400689797675,-4.410009735106357e-05,.004737799410682754,-.1722807281261381],
                "forcing_u":[0.,.001715513711500865,-.03349286094070662,.3640193048957979,.06420946716889489],
                "forcing_v":[0.,-.009881110440992118,-.0010302146106443949,.05309840150483982,-5.963881613650361]}
    actual = field.quantities(points)
    for key, values in expected.items():
        np.testing.assert_allclose(actual[key].detach(),values,rtol=1e-14,atol=1e-14)


@pytest.mark.parametrize("extension", [True, -1, float("nan"), float("inf")])
def test_invalid_extension_rejected(extension):
    with pytest.raises(ValueError, match="spatial_extension"):
        StructuredNSProjectionConfig(spatial_extension=extension).validate()


def test_invalid_outer_boundary_rejected():
    with pytest.raises(ValueError, match="outer_boundary"):
        ContinuousGeometrySpec(_BOUNDS, (), outer_boundary="unknown").validate()


def test_uniform_through_flow_uses_same_projector_with_zero_manufactured_pressure():
    # Constant through-flow is independent of the represented finite cosine sum.
    spec = ContinuousGeometrySpec((0.,1.,0.,1.), (), outer_boundary="manufactured_trace")
    geometry = _grid(spec, nx=25, ny=23)
    times = np.linspace(0.,1.,5)
    velocity = np.empty((5,2,25,23))
    velocity[:,0], velocity[:,1] = 1., .25
    config = StructuredNSProjectionConfig(spatial_extension=.25, time_order=2, ridge_relative=1e-8)
    coefficients, report = StructuredNSCoefficientProjector(
        geometry=geometry, t=times, config=config).project(
            velocity=velocity, pressure=np.zeros((5,25,23)))
    assert report["cases"][0]["velocity_relative_l2"] < .006
    assert report["cases"][0]["pressure_gauge_relative_l2"] == 0
    assert torch.count_nonzero(coefficients[:,1]) == 0


def test_cuda_graph_signature_separates_outer_boundary_contracts_on_cpu():
    from pde_agent_verifier_solver.plugins.ns.analytic_cuda import _field_signature
    field, _ = _project_known()
    closed = ContinuousStructuredNSMMS(field.coefficients, config=field.config,
        geometry_spec=replace(field.geometry_spec, outer_boundary="no_slip"),
        domain_bounds=field.domain_bounds, viscosity=field.viscosity, t0=field.t0, t1=field.t1)
    assert field.coefficients.psi.device.type == "cpu"
    assert _field_signature(field) != _field_signature(closed)
    same = ContinuousStructuredNSMMS(field.coefficients, config=field.config,
        geometry_spec=field.geometry_spec, domain_bounds=field.domain_bounds,
        viscosity=field.viscosity, t0=field.t0, t1=field.t1)
    assert _field_signature(field) == _field_signature(same)
