"""Independent basis, polynomial MMS and legacy-configuration checks."""
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')
scipy_interpolate = pytest.importorskip('scipy.interpolate')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'legacy/ns/src'))
from agent_verifier.neural_generator.continuous_geometry import ContinuousGeometrySpec
from agent_verifier.neural_generator.structured_ns_answer_first import (
    ContinuousStructuredNSMMS, StructuredNSCaseCoefficients, StructuredNSCoefficientProjector,
    StructuredNSProjectionConfig, load_structured_continuous_case, write_structured_continuous_case_state,
)
from agent_verifier.neural_generator.structured_ns_basis import (
    axis_basis_derivatives, bspline_basis, open_uniform_knots,
)
from agent_verifier.neural_generator.structured_ns_analytic import analytic_quantities, analytic_velocity


@pytest.fixture(scope='module', autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


BOUNDS = (-2., 3., .3, 1.1)
T0, T1 = .2, .57


def config(space='bspline', time='bspline'):
    return StructuredNSProjectionConfig(
        spatial_basis=space, temporal_basis=time,
        spatial_mode_x=7 if space == 'bspline' else 3,
        spatial_mode_y=6 if space == 'bspline' else 3,
        time_order=5, spatial_extension=0. if space == 'bspline' else .25,
        ridge_relative=0.)


def field(values, cfg, *, geometry=None):
    return ContinuousStructuredNSMMS(
        StructuredNSCaseCoefficients(values[0].reshape(cfg.time_coefficient_count, -1),
                                     values[1].reshape(cfg.time_coefficient_count, -1)),
        config=cfg, geometry_spec=geometry or ContinuousGeometrySpec(BOUNDS, (), outer_boundary='manufactured_trace'),
        domain_bounds=BOUNDS, viscosity=.037, t0=T0, t1=T1)


def polynomial_coefficients(knots, degree, function):
    # Independent SciPy collocation at Greville points, only for test oracles.
    knots = np.asarray(knots)
    n = len(knots)-degree-1
    greville = np.array([knots[i+1:i+degree+1].mean() for i in range(n)])
    matrix = scipy_interpolate.BSpline(knots, np.eye(n), degree)(greville)
    return np.linalg.solve(matrix, function(greville))


@pytest.mark.parametrize('degree,count', [(3, 7), (5, 9)])
def test_bspline_values_derivatives_and_endpoint_autograd_match_scipy(degree, count):
    knots = open_uniform_knots(count, degree)
    inner = np.unique(knots)[1:-1]
    points = np.r_[0., 1., np.linspace(.031, .971, 17), inner, inner-1e-9, inner+1e-9]
    x = torch.tensor(points, dtype=torch.float64, requires_grad=True)
    oracle = scipy_interpolate.BSpline(knots, np.eye(count), degree)
    for order in range(4):
        value = bspline_basis(x, knots=knots, degree=degree, derivative_order=order)
        np.testing.assert_allclose(value.detach(), oracle(points, nu=order), rtol=2e-12, atol=2e-10)
        if order < 3:
            weights = torch.linspace(.2, 1.3, count, dtype=torch.float64)
            auto = torch.autograd.grad((value*weights).sum(), x, create_graph=True)[0]
            exact = (bspline_basis(x, knots=knots, degree=degree, derivative_order=order+1)*weights).sum(-1)
            torch.testing.assert_close(auto, exact, rtol=2e-11, atol=2e-9)
    np.testing.assert_allclose(bspline_basis(x, knots=knots, degree=degree).detach().sum(-1), 1., atol=1e-14)


def test_axis_chebyshev_second_derivative_matches_polynomial_reference():
    cfg = config('cosine', 'chebyshev')
    t = torch.tensor([0., .17, .5, 1.], dtype=torch.float64)
    result = axis_basis_derivatives(t, config=cfg, axis='t', order=2)
    for degree in range(cfg.time_coefficient_count):
        coefficients = np.eye(cfg.time_coefficient_count)[degree]
        for order in range(3):
            oracle = np.polynomial.chebyshev.chebval(2*t.numpy()-1,
                np.polynomial.chebyshev.chebder(coefficients, m=order))*2**order
            np.testing.assert_allclose(result[order][:, degree], oracle, atol=1e-12)


@pytest.mark.parametrize('temporal', ['chebyshev', 'bspline'])
def test_quintic_polynomial_mms_at_knots_boundaries_and_coeff_gradient(temporal):
    cfg = config(time=temporal)
    cx = polynomial_coefficients(cfg.spatial_knots_x, 5, lambda x: x*x)
    cy = polynomial_coefficients(cfg.spatial_knots_y, 5, lambda x: x*x)
    if temporal == 'bspline':
        ct = polynomial_coefficients(cfg.temporal_knots, 3, lambda t: 1+.4*t+.1*t*t)
    else:
        ct = np.zeros(cfg.time_coefficient_count); ct[:3] = (1.2375, .25, .0125)
    psi = ct[:, None, None]*cx[None, :, None]*cy[None, None, :]
    pressure = .17*ct[:, None, None]*cx[None, :, None]*np.ones_like(cy)[None, None, :]
    values = torch.tensor(np.stack((psi, pressure)), dtype=torch.float64, requires_grad=True)
    answer = field(values, cfg)
    xn = np.unique(np.r_[0., .17, .71, 1., cfg.spatial_knots_x[6:-6]])
    yn = np.unique(np.r_[0., .24, .83, 1., cfg.spatial_knots_y[6:-6]])
    tn = np.unique(np.r_[0., .13, .73, 1., cfg.temporal_knots[4:-4] if temporal == 'bspline' else []])
    x, y, t = np.stack(np.meshgrid(xn, yn, tn, indexing='ij'), -1).reshape(-1, 3).T
    lx, ly, lt = BOUNDS[1]-BOUNDS[0], BOUNDS[3]-BOUNDS[2], T1-T0
    points = np.column_stack((BOUNDS[0]+lx*x, BOUNDS[2]+ly*y, T0+lt*t))
    a, at = 1+.4*t+.1*t*t, (.4+.2*t)/lt
    u, v = a*2*x*x*y/ly, -a*2*x*y*y/lx
    ux, uy, vx, vy = a*4*x*y/(lx*ly), a*2*x*x/ly**2, -a*2*y*y/lx**2, -a*4*x*y/(lx*ly)
    ut, vt = at*2*x*x*y/ly, -at*2*x*y*y/lx
    lapu, lapv = a*4*y/(lx*lx*ly), -a*4*x/(lx*ly*ly)
    expected = {'u':u, 'v':v, 'd_x_u':ux, 'd_y_u':uy, 'd_x_v':vx, 'd_y_v':vy,
                'd_t_u':ut, 'd_t_v':vt, 'laplacian_u':lapu, 'laplacian_v':lapv,
                'forcing_u':ut+u*ux+v*uy+.34*a*x/lx-.037*lapu,
                'forcing_v':vt+u*vx+v*vy-.037*lapv}
    actual = analytic_quantities(answer, points)
    auto = answer.quantities(points)
    for name, value in expected.items():
        np.testing.assert_allclose(actual[name].detach(), value, rtol=2e-10, atol=2e-10)
        np.testing.assert_allclose(auto[name].detach(), value, rtol=2e-9, atol=2e-9)
    assert actual['divergence'].abs().max() == 0
    objective = (actual['forcing_u'].square()+actual['forcing_v'].square()).mean()
    gradient = torch.autograd.grad(objective, values)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().max() > 0


def test_constant_flow_reproduction_and_new_basis_cuda_cache_guard():
    cfg = config()
    gx = np.array([np.mean(cfg.spatial_knots_x[i+1:i+6]) for i in range(cfg.spatial_mode_x+1)])
    gy = np.array([np.mean(cfg.spatial_knots_y[i+1:i+6]) for i in range(cfg.spatial_mode_y+1)])
    psi = (BOUNDS[3]-BOUNDS[2])*.7*gy[None, :] + (BOUNDS[1]-BOUNDS[0])*.3*gx[:, None]
    coefficients = np.zeros((2, cfg.time_coefficient_count, len(gx), len(gy)))
    coefficients[0] = psi
    answer = field(coefficients, cfg)
    points = np.array([[-2., .3, T0], [3., 1.1, T1], [0., .7, .4]])
    q = analytic_quantities(answer, points)
    np.testing.assert_allclose(q['u'], .7, atol=1e-13)
    np.testing.assert_allclose(q['v'], -.3, atol=1e-13)
    np.testing.assert_allclose(q['forcing_u'], 0., atol=2e-11)
    np.testing.assert_allclose(q['forcing_v'], 0., atol=2e-11)
    from pde_agent_verifier_solver.plugins.ns.analytic_cuda import _field_signature
    with pytest.raises(NotImplementedError, match='cosine/Chebyshev'):
        _field_signature(answer)
    from pde_agent_verifier_solver.plugins.ns.continuous_backend import StructuredNSMMSAnswer
    wrapped = StructuredNSMMSAnswer(answer)
    class ForbiddenCache:
        def evaluate(self, *args):
            pytest.fail('B-spline fields must not enter the cosine CUDA graph fastpath')
    wrapped._analytic_cuda_cache = ForbiddenCache()
    np.testing.assert_allclose(wrapped.evaluate_velocity(points[:, (2, 0, 1)]), [[.7, -.3]]*3, atol=1e-13)
    np.testing.assert_allclose(wrapped.evaluate_forcing(points[:, (2, 0, 1)]), 0., atol=2e-11)


@pytest.mark.parametrize('space', ['cosine', 'bspline'])
@pytest.mark.parametrize('temporal', ['chebyshev', 'bspline'])
def test_projector_and_field_agree_for_all_four_basis_combinations(space, temporal):
    cfg = config(space, temporal)
    rng = np.random.default_rng(4)
    values = rng.normal(0, .002, (2, cfg.time_coefficient_count, cfg.spatial_mode_x+1, cfg.spatial_mode_y+1))
    answer = field(values, cfg)
    x, y, times = np.linspace(*BOUNDS[:2], 15), np.linspace(*BOUNDS[2:], 14), np.linspace(T0, T1, 9)
    grid = SimpleNamespace(x=x, y=y, fluid_mask=np.ones((len(x), len(y)), bool),
        config=SimpleNamespace(nx=len(x), ny=len(y)),
        to_json_dict=lambda: {'continuous_geometry': answer.geometry_spec.to_dict()})
    tt, xx, yy = np.meshgrid(times, x, y, indexing='ij')
    points = np.column_stack((xx.ravel(), yy.ravel(), tt.ravel()))
    u, v, p = answer.velocity_pressure(points)
    velocity = torch.stack((u, v), 1).reshape(len(times), len(x), len(y), 2).permute(0, 3, 1, 2)
    pressure = p.reshape(len(times), len(x), len(y))
    projector = StructuredNSCoefficientProjector(geometry=grid, t=times, config=cfg)
    recovered, report = projector.project(velocity=velocity, pressure=pressure)
    assert report['cases'][0]['velocity_relative_l2'] < 1e-10
    restored = field(recovered[0], cfg)
    for a, b in zip(analytic_velocity(restored, points), (u, v)):
        np.testing.assert_allclose(a.detach(), b.detach(), rtol=1e-8, atol=1e-10)


def test_new_state_roundtrip_and_old_configuration_serialization(tmp_path):
    old = {'spatial_mode_x':8, 'spatial_mode_y':8, 'time_order':8,
           'geometry_mask_transition':.05, 'ridge_relative':1e-6, 'evaluation_batch_points':4096}
    assert StructuredNSProjectionConfig().to_dict() == old
    assert StructuredNSProjectionConfig.from_dict(old).to_dict() == old
    cfg = replace(config(), fitting_method='joint_smoothing', smoothing_spatial=.01, smoothing_temporal=.02)
    encoded = json.loads(json.dumps(cfg.to_dict()))
    assert StructuredNSProjectionConfig.from_dict(encoded) == cfg
    coefficients = np.random.default_rng(6).normal(0, .002, (2, cfg.time_coefficient_count, cfg.spatial_mode_x+1, cfg.spatial_mode_y+1))
    answer = field(coefficients, cfg)
    write_structured_continuous_case_state(tmp_path, field=answer)
    restored = load_structured_continuous_case(tmp_path)
    assert restored.config == cfg
    points = np.array([[-2., .3, T0], [3., 1.1, T1], [0., .7, .4]])
    a, b = analytic_quantities(answer, points), analytic_quantities(restored, points)
    for name in ('u', 'v', 'forcing_u', 'forcing_v'):
        torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)


@pytest.mark.parametrize('change', [
    {'spatial_extension':.1}, {'spatial_degree':4}, {'temporal_degree':2},
    {'spatial_knots_x':(0.,)*6+(.5,.5)+(1.,)*6},
    {'temporal_knots':(0.,)*4+(.7,.3)+(1.,)*4},
])
def test_invalid_new_basis_contract_rejected(change):
    with pytest.raises(ValueError):
        replace(config(), **change).validate()


@pytest.mark.parametrize('outer_boundary', ['no_slip', 'manufactured_trace'])
def test_new_spline_masked_geometry_projection_and_mms(outer_boundary, record_property):
    """A bounded synthetic fit checks the original mask; no numerical campaign is rerun."""
    cfg = replace(config(), geometry_mask_transition=.1)
    polygon = ((.0, .58), (.7, .58), (.7, .78), (.0, .78))
    geometry = ContinuousGeometrySpec(BOUNDS, (polygon,), outer_boundary=outer_boundary)
    values = np.random.default_rng(8).normal(0, .0002,
        (2, cfg.time_coefficient_count, cfg.spatial_mode_x+1, cfg.spatial_mode_y+1))
    values[1] = 0.
    answer = field(values, cfg, geometry=geometry)
    x, y, times = np.linspace(*BOUNDS[:2], 23), np.linspace(*BOUNDS[2:], 22), np.linspace(T0, T1, 9)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    fluid = ~((xx >= .0) & (xx <= .7) & (yy >= .58) & (yy <= .78))
    grid = SimpleNamespace(x=x, y=y, fluid_mask=fluid,
        config=SimpleNamespace(nx=len(x), ny=len(y)),
        to_json_dict=lambda: {'continuous_geometry': geometry.to_dict()})
    tt, xx, yy = np.meshgrid(times, x, y, indexing='ij')
    points = np.column_stack((xx.ravel(), yy.ravel(), tt.ravel()))
    u, v, p = answer.velocity_pressure(points)
    velocity = torch.stack((u, v), 1).reshape(len(times), len(x), len(y), 2).permute(0, 3, 1, 2)
    projector = StructuredNSCoefficientProjector(geometry=grid, t=times, config=cfg)
    _, report = projector.project(velocity=velocity, pressure=p.reshape(len(times), len(x), len(y)))
    fit_error = report['cases'][0]['velocity_relative_l2']
    record_property('masked_velocity_relative_l2', fit_error)
    assert fit_error < 1e-10
    # Include obstacle edges and interior, ramp positions, outer edges and corners.
    queries = np.array([[-2., .3, T0], [3., 1.1, T1], [-1., .3, .4],
        [.0, .66, .4], [.7, .7, .4], [.3, .58, .4], [.3, .68, .4],
        [-.1, .66, .4], [.85, .7, .4], [.3, .56, .4], [1.3, .9, .4]])
    analytic, reference = analytic_quantities(answer, queries), answer.quantities(queries)
    max_error = 0.
    for name in ('u', 'v', 'd_x_u', 'd_y_u', 'd_x_v', 'd_y_v', 'd_t_u', 'd_t_v',
                 'laplacian_u', 'laplacian_v', 'forcing_u', 'forcing_v'):
        torch.testing.assert_close(analytic[name], reference[name], rtol=2e-9, atol=2e-9)
        max_error = max(max_error, float((analytic[name]-reference[name]).abs().max()))
    record_property('max_analytic_autograd_abs_error', max_error)
    boundary_indices = list(range(7)) if outer_boundary == 'no_slip' else [3, 4, 5, 6]
    boundary_max = max(float(analytic[name][boundary_indices].abs().max()) for name in ('u', 'v'))
    record_property('no_slip_max_velocity', boundary_max)
    assert boundary_max < 1e-14


def test_temporal_spline_basis_count_is_not_polynomial_degree():
    cfg = replace(config(), time_order=7, temporal_knots=(), fitting_method='joint_smoothing')
    x, y = np.linspace(*BOUNDS[:2], 10), np.linspace(*BOUNDS[2:], 9)
    geometry = ContinuousGeometrySpec(BOUNDS, (), outer_boundary='manufactured_trace')
    grid = SimpleNamespace(x=x, y=y, fluid_mask=np.ones((len(x), len(y)), bool),
        config=SimpleNamespace(nx=len(x), ny=len(y)),
        to_json_dict=lambda: {'continuous_geometry': geometry.to_dict()})
    projector = StructuredNSCoefficientProjector(geometry=grid, t=np.linspace(T0, T1, 4), config=cfg)
    assert projector.temporal_design.shape == (4, 8)
    with pytest.raises(ValueError, match='explicit joint'):
        projector.project(velocity=None, pressure=None)
    with pytest.raises(ValueError, match='at least two frames'):
        StructuredNSCoefficientProjector(geometry=grid, t=[T0], config=cfg)
