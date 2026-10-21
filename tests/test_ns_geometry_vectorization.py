"""Numerical parity of the NS geometry/oracle optimization, including derivatives."""
from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "legacy" / "ns" / "src"))
from agent_verifier.neural_generator import continuous_geometry as geometry
from agent_verifier.neural_generator import structured_ns_answer_first as mms


def _edge_loop_reference(coords, *, polygon):
    # Independent edge-by-edge signed distance, retained as a numerical reference.
    vertices = torch.as_tensor(polygon, dtype=coords.dtype, device=coords.device)
    area = sum(x * polygon[(i + 1) % len(polygon)][1] - polygon[(i + 1) % len(polygon)][0] * y
               for i, (x, y) in enumerate(polygon)) / 2
    orientation = 1.0 if area > 0 else -1.0
    values = []
    for i in range(len(polygon)):
        edge = vertices[(i + 1) % len(polygon)] - vertices[i]
        relative = coords[..., :2] - vertices[i]
        values.append(-orientation * (edge[0] * relative[..., 1] - edge[1] * relative[..., 0]) / edge.norm())
    return torch.stack(values, dim=-1)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("batched", [False, True])
def test_vectorized_distances_and_coordinate_gradients(reverse, batched):
    polygon = ((0.2, 0.3), (0.7, 0.2), (0.8, 0.7), (0.3, 0.8))
    if reverse:
        polygon = polygon[::-1]
    points = torch.tensor([[0.1, 0.2, 0.04], [0.5, 0.5, 0.03], [0.2, 0.3, 0.06], [1.0, 0.0, 0.12]], dtype=torch.float64)
    if batched:
        points = torch.stack((points, points + 0.02))
    reference_coords = points.clone().requires_grad_(True)
    actual_coords = points.clone().requires_grad_(True)
    reference = _edge_loop_reference(reference_coords, polygon=polygon)
    actual = geometry._convex_polygon_outward_distances(actual_coords, polygon=polygon)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    weights = torch.arange(1, 5, dtype=torch.float64)
    reference_grad = torch.autograd.grad((reference * weights).sum(), reference_coords)[0]
    actual_grad = torch.autograd.grad((actual * weights).sum(), actual_coords)[0]
    torch.testing.assert_close(actual_grad, reference_grad, rtol=2e-15, atol=2e-15)


@pytest.mark.parametrize("abi", [mms.LEGACY_C3_POLYGON_STRUCTURED_NS_MMS_ABI_VERSION,
                                  mms.LEGACY_HARD_PRESSURE_STRUCTURED_NS_MMS_ABI_VERSION,
                                  mms.LEGACY_SMOOTH_PRESSURE_STRUCTURED_NS_MMS_ABI_VERSION,
                                  mms.STRUCTURED_NS_MMS_ABI_VERSION])
def test_exact_mms_quantities_match_edge_reference_at_walls_and_interiors(monkeypatch, abi):
    config = mms.StructuredNSProjectionConfig(spatial_mode_x=2, spatial_mode_y=2, time_order=2)
    spec = geometry.ContinuousGeometrySpec(domain_bounds=(0., 1., 0., 1.),
        obstacle_polygons=(((.3, .3), (.6, .3), (.45, .6)),))
    generator = torch.Generator().manual_seed(101)
    coefficients = mms.StructuredNSCaseCoefficients(
        psi=torch.randn((3, 9), generator=generator, dtype=torch.float64) * 1e-4,
        pressure=torch.randn((3, 9), generator=generator, dtype=torch.float64) * 1e-4)
    field = mms.ContinuousStructuredNSMMS(coefficients, config=config, geometry_spec=spec,
        continuous_mms_abi=abi, domain_bounds=spec.domain_bounds, viscosity=.01, t0=0., t1=.12)
    points = torch.tensor([[0., .5, 0.], [1., .5, .12], [.5, 0., .06], [.5, 1., .06],
        [.3, .3, .06], [.6, .3, .06], [.45, .6, .06], [.45, .4, .06],
        [.15, .25, .03], [.8, .8, .09], [.28, .3, .07], [.45, .62, .07]], dtype=torch.float64)
    with monkeypatch.context() as context:
        context.setattr(geometry, "_convex_polygon_outward_distances", _edge_loop_reference)
        reference = {k: v.detach().clone() for k, v in field.quantities(points, include_laplacian_components=True).items()}
    actual = field.quantities(points, include_laplacian_components=True)
    assert actual.keys() == reference.keys()
    for key in reference:
        assert torch.isfinite(actual[key]).all(), key
        torch.testing.assert_close(actual[key], reference[key], rtol=2e-10, atol=1e-9, msg=key)
    # Repeated, different queries must not reuse a previous query's geometry or graph.
    field.quantities(points.flip(0), include_laplacian_components=True)


def test_forcing_reuses_only_current_query_masks(monkeypatch):
    config = mms.StructuredNSProjectionConfig(spatial_mode_x=1, spatial_mode_y=1, time_order=1)
    spec = geometry.ContinuousGeometrySpec(domain_bounds=(0., 1., 0., 1.), obstacle_polygons=())
    coefficients = mms.StructuredNSCaseCoefficients(psi=torch.ones((2, 4), dtype=torch.float64),
                                                  pressure=torch.ones((2, 4), dtype=torch.float64))
    field = mms.ContinuousStructuredNSMMS(coefficients, config=config, geometry_spec=spec,
        domain_bounds=spec.domain_bounds, viscosity=.01, t0=0., t1=.12)
    original = field.geometry_masks
    calls = []
    def counted(coords):
        calls.append(coords.detach().clone())
        return original(coords)
    monkeypatch.setattr(field, "geometry_masks", counted)
    field.quantities(torch.tensor([[.01, .2, .06]], dtype=torch.float64))
    field.quantities(torch.tensor([[.3, .01, .09]], dtype=torch.float64))
    assert len(calls) == 2
    assert not torch.equal(calls[0], calls[1])
