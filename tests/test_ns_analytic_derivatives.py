"""Independent autograd parity for the explicit NS derivative kernel.

The reference remains the existing physical-coordinate streamfunction graph.
No public Oracle adapter or derivative routing participates in this comparison.
Tolerances match the existing geometry-derivative qualification and are never
relaxed for high modes, wall transition endpoints, or perturbed fields.
"""
from pathlib import Path
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "legacy/ns/src"))
from agent_verifier.neural_generator import continuous_geometry as geometry
from agent_verifier.neural_generator import structured_ns_answer_first as mms
from agent_verifier.neural_generator import structured_ns_robustness as robustness
from agent_verifier.neural_generator.triplane_mms import navier_stokes_quantities_from_psi_pressure


_ABIS = (
    mms.LEGACY_C3_POLYGON_STRUCTURED_NS_MMS_ABI_VERSION,
    mms.LEGACY_HARD_PRESSURE_STRUCTURED_NS_MMS_ABI_VERSION,
    mms.LEGACY_SMOOTH_PRESSURE_STRUCTURED_NS_MMS_ABI_VERSION,
    mms.STRUCTURED_NS_MMS_ABI_VERSION,
    mms.INTERIOR_C3_STRUCTURED_NS_MMS_ABI_VERSION,
)
_TRIANGLE = ((.3, .3), (.6, .3), (.45, .6))
_SECOND = ((.68, .67), (.83, .64), (.86, .84), (.71, .87))
_RTOL, _ATOL = 2e-10, 1e-9


@pytest.fixture(scope="module", autouse=True)
def single_torch_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(params=("cpu", "cuda"))
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA derivative qualification requires an allocated GPU")
    return request.param


def _field(device="cpu", *, abi=mms.STRUCTURED_NS_MMS_ABI_VERSION,
           shape="ccw", modes=(3, 2, 3), seed=101):
    bounds = (-2.0, 3.0, .3, 1.1) if shape == "stretched" else (0., 1., 0., 1.)
    polygons = () if shape == "empty" else (_TRIANGLE,)
    if shape == "cw":
        polygons = (_TRIANGLE[::-1],)
    elif shape in ("multi", "stretched"):
        polygons = (_TRIANGLE, _SECOND[::-1])
    xmin, xmax, ymin, ymax = bounds
    physical_polygons = tuple(tuple((xmin + x * (xmax - xmin), ymin + y * (ymax - ymin))
                                   for x, y in polygon) for polygon in polygons)
    config = mms.StructuredNSProjectionConfig(
        spatial_mode_x=modes[0], spatial_mode_y=modes[1], time_order=modes[2],
        geometry_mask_transition=.05,
    )
    spec = geometry.ContinuousGeometrySpec(
        domain_bounds=bounds, obstacle_polygons=physical_polygons,
        abi_version=(geometry.LEGACY_CONTINUOUS_GEOMETRY_ABI_VERSION
                     if shape == "legacy_geo" else geometry.CONTINUOUS_GEOMETRY_ABI_VERSION),
    )
    rng = torch.Generator().manual_seed(seed)
    coefficient_shape = (modes[2] + 1, (modes[0] + 1) * (modes[1] + 1))
    coefficients = mms.StructuredNSCaseCoefficients(
        psi=torch.randn(coefficient_shape, generator=rng, dtype=torch.float64) * 1e-4,
        pressure=torch.randn(coefficient_shape, generator=rng, dtype=torch.float64) * 3e-4,
    )
    return mms.ContinuousStructuredNSMMS(
        coefficients, config=config, geometry_spec=spec, continuous_mms_abi=abi,
        domain_bounds=bounds, viscosity=.037, t0=.2, t1=.57, device=device,
    )


def _points(field):
    # Normalized endpoints, one-sided walls, both sides of ramp saturation, and
    # interior points. Times include both endpoints and an asymmetric interior.
    width = field.config.geometry_mask_transition
    epsilon = 1e-9
    xy = [(0., 0.), (0., 1.), (1., 0.), (1., 1.), (.5, .5)]
    offsets = (0., epsilon, width / 2, width - epsilon, width, width + epsilon)
    for distance in offsets:
        xy.extend(((distance, .13), (1. - distance, .13),
                   (.13, distance), (.13, 1. - distance)))
    for polygon in field.geometry_spec.normalized_obstacle_polygons():
        array = np.asarray(polygon)
        xy.extend(polygon)
        xy.append(tuple(array.mean(axis=0)))
        signed_area = sum(x * polygon[(i + 1) % len(polygon)][1]
                          - polygon[(i + 1) % len(polygon)][0] * y
                          for i, (x, y) in enumerate(polygon))
        orientation = 1. if signed_area > 0 else -1.
        for index, point in enumerate(array):
            edge = array[(index + 1) % len(array)] - point
            outward = orientation * np.array((edge[1], -edge[0])) / np.linalg.norm(edge)
            middle = point + .5 * edge
            for distance in (-epsilon, 0., epsilon, .5 * width, width - epsilon, width, width + epsilon):
                xy.append(tuple(middle + distance * outward))
    random = np.random.default_rng(794).uniform(.001, .999, (17, 2))
    xy.extend(map(tuple, random))
    normalized = np.array([(x, y, (0., .31, 1.)[i % 3]) for i, (x, y) in enumerate(xy)])
    xmin, xmax, ymin, ymax = field.domain_bounds
    normalized[:, 0] = xmin + normalized[:, 0] * (xmax - xmin)
    normalized[:, 1] = ymin + normalized[:, 1] * (ymax - ymin)
    normalized[:, 2] = field.t0 + normalized[:, 2] * (field.t1 - field.t0)
    return normalized


def _reference(field, points, *, components=True):
    # Calling the original differentiated psi/pressure graph explicitly keeps
    # this reference independent even if an adapter later routes to analytics.
    with torch.inference_mode(False), torch.enable_grad():
        coords = torch.as_tensor(points, dtype=torch.float64, device=field.device)
        if coords.ndim == 3:
            coords = coords[0]
        coords = coords.clone().detach().requires_grad_(True)
        psi, pressure, coords = field.psi_pressure(coords)
        result = navier_stokes_quantities_from_psi_pressure(
            psi, pressure, coords, viscosity=field.viscosity,
            include_laplacian_components=components,
        )
        if field.continuous_mms_abi in _ABIS[2:]:
            indicator = field.geometry_masks(field._normalize_coordinates(coords))[1]
            for name in ("forcing_u", "forcing_v", "residual_u", "residual_v"):
                result[name] = result[name] * indicator
            result["fluid_indicator"] = indicator
        return {key: value.detach().clone() for key, value in result.items()}


def _assert_quantities(actual, expected):
    assert actual.keys() == expected.keys()
    for name in expected:
        assert torch.isfinite(actual[name]).all(), name
        torch.testing.assert_close(actual[name], expected[name], rtol=_RTOL, atol=_ATOL, msg=name)


@pytest.mark.parametrize("abi", _ABIS, ids=("v2", "v3", "v4", "v5", "v6"))
@pytest.mark.parametrize("shape", ("empty", "ccw", "cw", "multi", "stretched", "legacy_geo"))
def test_every_analytic_derivative_matches_original_autograd(device, abi, shape):
    from agent_verifier.neural_generator.structured_ns_analytic import (
        analytic_quantities, analytic_velocity_pressure,
    )
    field = _field(device, abi=abi, shape=shape)
    points = _points(field)
    expected = _reference(field, points)
    actual = analytic_quantities(field, points, include_laplacian_components=True)
    _assert_quantities(actual, expected)
    # Includes psi_xy, psi_xt/psi_yt, psi_xxy and psi_xyy through velocity derivatives
    # and the separate Laplacian components, plus pressure gradients.
    velocity = analytic_velocity_pressure(field, points)
    for name, value in zip(("u", "v", "pressure"), velocity, strict=True):
        torch.testing.assert_close(value, expected[name], rtol=_RTOL, atol=_ATOL, msg=name)
    short = analytic_quantities(field, points)
    _assert_quantities(short, _reference(field, points, components=False))


@pytest.mark.parametrize("abi", (mms.STRUCTURED_NS_MMS_ABI_VERSION, mms.INTERIOR_C3_STRUCTURED_NS_MMS_ABI_VERSION))
@pytest.mark.parametrize("modes", ((8, 6, 8), (20, 20, 4)))
def test_high_spatial_and_time_orders_preserve_mixed_derivatives(device, modes, abi):
    from agent_verifier.neural_generator.structured_ns_analytic import analytic_quantities
    field = _field(device, shape="stretched", modes=modes, seed=918, abi=abi)
    points = _points(field)
    _assert_quantities(analytic_quantities(field, points, include_laplacian_components=True),
                       _reference(field, points))


@pytest.mark.parametrize("abi", (mms.STRUCTURED_NS_MMS_ABI_VERSION, mms.INTERIOR_C3_STRUCTURED_NS_MMS_ABI_VERSION))
def test_real_spectral_perturbation_expanded_band_and_initial_endpoint(device, abi):
    from agent_verifier.neural_generator.structured_ns_analytic import analytic_quantities
    base = _field(device, shape="multi", modes=(3, 2, 4), abi=abi)
    perturbation = robustness.make_perturbation(
        base, family="spectral_complement", seed=701, spectral_max_mode=20,
    )
    field = robustness.field_at_amplitude(base, perturbation, amplitude=3e-4, device=device)
    assert field.config.spatial_mode_x == field.config.spatial_mode_y == 20
    points = _points(field)
    _assert_quantities(analytic_quantities(field, points, include_laplacian_components=True),
                       _reference(field, points))
    initial = points.copy()
    initial[:, 2] = base.t0
    base_result = analytic_quantities(base, initial, include_laplacian_components=True)
    perturbed_result = analytic_quantities(field, initial, include_laplacian_components=True)
    for name in ("psi", "u", "v", "d_t_u", "d_t_v"):
        torch.testing.assert_close(perturbed_result[name], base_result[name],
                                   rtol=_RTOL, atol=_ATOL, msg=name)


@pytest.mark.parametrize("abi", _ABIS, ids=("v2", "v3", "v4", "v5", "v6"))
def test_pressure_sign_and_solid_restriction_are_preserved(device, abi):
    from agent_verifier.neural_generator.structured_ns_analytic import analytic_quantities
    field = _field(device, abi=abi)
    field.coefficients.psi.zero_()
    field.coefficients.pressure.zero_()
    field.coefficients.pressure[0, 2 * (field.config.spatial_mode_y + 1)] = 1.
    points = np.array([[.25, .8, field.t0], [.75, .8, field.t1],
                       [.45, .4, .31], [0., .8, .31]])
    actual = analytic_quantities(field, points, include_laplacian_components=True)
    _assert_quantities(actual, _reference(field, points))
    assert actual["forcing_u"][0] < 0 < actual["forcing_u"][1]
    if abi in _ABIS[2:]:
        assert actual["fluid_indicator"][2] == 0
        assert actual["forcing_u"][2] == actual["forcing_v"][2] == 0


@pytest.mark.parametrize("abi", (mms.STRUCTURED_NS_MMS_ABI_VERSION, mms.INTERIOR_C3_STRUCTURED_NS_MMS_ABI_VERSION))
def test_reordered_split_and_singleton_queries_have_no_cross_call_state(device, monkeypatch, abi):
    from agent_verifier.neural_generator.structured_ns_analytic import (
        analytic_quantities, analytic_velocity_pressure,
    )
    field = _field(device, shape="multi", abi=abi)
    points = torch.as_tensor(_points(field), device=device)
    original_points = points.clone()
    original_psi = field.coefficients.psi.clone()
    original_pressure = field.coefficients.pressure.clone()
    expected = _reference(field, points)
    def forbidden(*args, **kwargs):
        raise AssertionError("analytic inference called torch.autograd.grad")
    monkeypatch.setattr(torch.autograd, "grad", forbidden)
    actual = analytic_quantities(field, points, include_laplacian_components=True)
    reversed_result = analytic_quantities(field, points.flip(0), include_laplacian_components=True)
    _assert_quantities({name: value.flip(0) for name, value in reversed_result.items()}, expected)
    even = analytic_quantities(field, points[::2], include_laplacian_components=True)
    _assert_quantities(even, {name: value[::2] for name, value in expected.items()})
    singleton = analytic_velocity_pressure(field, points[:1].unsqueeze(0))
    for name, value in zip(("u", "v", "pressure"), singleton, strict=True):
        torch.testing.assert_close(value, expected[name][:1], rtol=_RTOL, atol=_ATOL)
    _assert_quantities(actual, expected)  # Later calls must not mutate earlier tensors.
    assert torch.equal(points, original_points)
    assert torch.equal(field.coefficients.psi, original_psi)
    assert torch.equal(field.coefficients.pressure, original_pressure)
    assert field.coefficients.psi.grad is field.coefficients.pressure.grad is None


@pytest.mark.parametrize("abi", _ABIS, ids=("v2", "v3", "v4", "v5", "v6"))
@pytest.mark.parametrize("times", ("initial", "final", "mixed"))
def test_uniform_time_and_narrow_outputs_match_independent_reference(device, abi, times, monkeypatch):
    from agent_verifier.neural_generator.structured_ns_analytic import (
        analytic_forcing, analytic_quantities, analytic_velocity, analytic_velocity_pressure,
    )
    field = _field(device, abi=abi, shape="stretched", modes=(3, 2, 8))
    points = _points(field)
    if times != "mixed":
        points[:, 2] = field.t0 if times == "initial" else field.t1
    expected = _reference(field, points)
    def forbidden(*args, **kwargs):
        raise AssertionError("narrow or homogeneous-time kernel called autograd")
    monkeypatch.setattr(torch.autograd, "grad", forbidden)
    # True is a trusted homogeneous-time hint, so never supply it to mixed
    # times. None's CPU inference and CUDA general path must both stay exact.
    for hint in ((None, False) if times == "mixed" else (None, True, False)):
        actual = analytic_quantities(field, points, include_laplacian_components=True, uniform_time=hint)
        _assert_quantities(actual, expected)
        for function, names in (
            (analytic_forcing, ("forcing_u", "forcing_v")),
            (analytic_velocity, ("u", "v")),
            (analytic_velocity_pressure, ("u", "v", "pressure")),
        ):
            outputs = function(field, points, uniform_time=hint)
            for name, value in zip(names, outputs, strict=True):
                torch.testing.assert_close(value, expected[name], rtol=_RTOL, atol=_ATOL,
                                           msg=f"{function.__name__}/{times}/{hint}/{name}")
