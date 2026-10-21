"""Analytic polygon jets match the existing mask's actual autograd semantics."""
from __future__ import annotations

from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _legacy_path(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "legacy/ns/src"))
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _field(*, mms=5, geometry_abi=2, polygons=(), bounds=(-2.0, 3.0, -4.0, 2.0), device="cpu"):
    from agent_verifier.neural_generator.continuous_geometry import ContinuousGeometrySpec
    from agent_verifier.neural_generator.structured_ns_answer_first import (
        ContinuousStructuredNSMMS, StructuredNSCaseCoefficients, StructuredNSProjectionConfig,
    )
    xmin, xmax, ymin, ymax = bounds
    physical_polygons = tuple(tuple((xmin + x * (xmax - xmin), ymin + y * (ymax - ymin))
                                   for x, y in polygon) for polygon in polygons)
    geometry = ContinuousGeometrySpec(
        bounds, physical_polygons, abi_version=f"continuous_geometry_convex_polygons_v{geometry_abi}",
    )
    coefficients = torch.zeros((2, 4), dtype=torch.float64)
    return ContinuousStructuredNSMMS(
        StructuredNSCaseCoefficients(coefficients, coefficients.clone()),
        config=StructuredNSProjectionConfig(spatial_mode_x=1, spatial_mode_y=1, time_order=1,
                                           geometry_mask_transition=0.1),
        geometry_spec=geometry, continuous_mms_abi=f"continuous_structured_ns_mms_v{mms}",
        domain_bounds=bounds, viscosity=0.01, t0=0.0, t1=0.12, device=device,
    )


def _autograd_jets(value, coordinates, order):
    result = {(0, 0): value}
    for degree in range(1, order + 1):
        for ny in range(degree + 1):
            nx = degree - ny
            axis = 0 if nx else 1
            parent = (nx - 1, ny) if nx else (nx, ny - 1)
            previous = result[parent]
            if not previous.requires_grad:
                result[(nx, ny)] = torch.zeros_like(value)
            else:
                gradient = torch.autograd.grad(previous.sum(), coordinates, create_graph=True,
                                               retain_graph=True)[0]
                result[(nx, ny)] = gradient[:, axis]
    return result


def _coordinates():
    # Exact wall and saturation endpoints, both sides of each clamp, obstacle
    # interiors/corners/transitions, and independent off-grid random queries.
    grid = torch.tensor([-0.05, 0., 1e-12, .03, .1 - 1e-12, .1, .1 + 1e-12,
                         .2, .3, .35, .4, .5, .7, .8, .9, 1., 1.05], dtype=torch.float64)
    xy = torch.cartesian_prod(grid, grid)
    generator = torch.Generator().manual_seed(117)
    xy = torch.cat((xy, torch.rand((70, 2), generator=generator, dtype=torch.float64)))
    return torch.cat((xy, torch.full((len(xy), 1), .37, dtype=torch.float64)), dim=1)


_POLYGONS = (
    (),
    (((.3, .3), (.5, .3), (.5, .5), (.3, .5)),),
    (((.3, .3), (.3, .5), (.5, .5), (.5, .3)),),
    (((.21, .28), (.46, .33), (.32, .57)), ((.66, .65), (.72, .88), (.9, .66))),
)


@pytest.mark.parametrize("mms", (2, 3, 4, 5, 6))
@pytest.mark.parametrize("geometry_abi", (1, 2))
@pytest.mark.parametrize("polygons", _POLYGONS)
def test_all_mixed_derivatives_match_existing_mask_autograd(mms, geometry_abi, polygons):
    from agent_verifier.neural_generator.structured_ns_geometry_jets import (
        geometry_mask_derivatives, pressure_mask_derivatives,
    )
    field = _field(mms=mms, geometry_abi=geometry_abi, polygons=polygons)
    coordinates = _coordinates().requires_grad_(True)
    reference_mask, reference_indicator = field.geometry_masks(coordinates)
    reference = _autograd_jets(reference_mask, coordinates, 3)
    actual, indicator = geometry_mask_derivatives(field, coordinates)
    assert set(actual) == set(reference)
    for key in actual:
        torch.testing.assert_close(actual[key], reference[key], rtol=2e-10, atol=1e-9,
                                   msg=lambda message: f"mask derivative {key}: {message}")
    torch.testing.assert_close(indicator, reference_indicator, rtol=2e-14, atol=2e-15)
    pressure = pressure_mask_derivatives(field, coordinates, order=1)
    reference_pressure = _autograd_jets(reference_indicator, coordinates, 1)
    for key in pressure:
        torch.testing.assert_close(pressure[key], reference_pressure[key], rtol=2e-10, atol=1e-9)


@pytest.mark.parametrize("order", (0, 1, 2, 3))
def test_requested_order_and_empty_batch(order):
    from agent_verifier.neural_generator.structured_ns_geometry_jets import geometry_mask_derivatives
    field = _field(polygons=_POLYGONS[-1])
    jets, fluid = geometry_mask_derivatives(field, torch.empty((0, 3), dtype=torch.float64), order=order)
    assert len(jets) == (order + 1) * (order + 2) // 2
    assert all(nx + ny <= order and value.shape == (0,) for (nx, ny), value in jets.items())
    assert fluid.shape == (0,)


def test_exact_modern_clamp_zero_uses_inclusive_pytorch_derivatives():
    from agent_verifier.neural_generator.structured_ns_geometry_jets import geometry_mask_derivatives
    field = _field(bounds=(0., 1., 0., 1.))
    coordinates = torch.tensor([[0., .5, .2], [-1e-12, .5, .2], [.1, .5, .2]], dtype=torch.float64)
    jets, fluid = geometry_mask_derivatives(field, coordinates)
    torch.testing.assert_close(jets[(2, 0)], torch.tensor([2000., 0., 0.], dtype=torch.float64), atol=1e-10, rtol=1e-12)
    torch.testing.assert_close(jets[(3, 0)], torch.tensor([-120000., 0., 0.], dtype=torch.float64), atol=1e-8, rtol=1e-12)
    assert fluid.tolist() == [0., 0., 1.]


def test_queries_reuse_prepared_geometry_without_validation_or_retained_coordinate_graph(monkeypatch):
    from agent_verifier.neural_generator.continuous_geometry import ContinuousGeometrySpec
    from agent_verifier.neural_generator.structured_ns_geometry_jets import geometry_mask_derivatives
    field = _field(polygons=_POLYGONS[-1])
    original_attributes = dict(vars(field))
    monkeypatch.setattr(ContinuousGeometrySpec, "validate", lambda self: pytest.fail("query revalidated geometry"))
    for _ in range(2):
        coordinates = _coordinates().requires_grad_(True)
        jets, _ = geometry_mask_derivatives(field, coordinates, order=1)
        assert jets[(0, 0)].requires_grad
    assert vars(field).keys() == original_attributes.keys()
    assert all(vars(field)[name] is value for name, value in original_attributes.items())


@pytest.mark.parametrize("order", (-1, 4, True, 1.5))
def test_rejects_unsupported_derivative_order(order):
    from agent_verifier.neural_generator.structured_ns_geometry_jets import geometry_mask_derivatives
    with pytest.raises(ValueError, match="derivative order"):
        geometry_mask_derivatives(_field(), _coordinates(), order=order)


def test_rejects_unknown_or_sampled_field():
    from types import SimpleNamespace
    from agent_verifier.neural_generator.structured_ns_geometry_jets import geometry_mask_derivatives
    with pytest.raises(ValueError, match="prepared continuous polygon"):
        geometry_mask_derivatives(SimpleNamespace(geometry_spec=None), _coordinates())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("mms", (2, 5, 6))
def test_cuda_mask_jets_match_original_autograd(mms):
    from agent_verifier.neural_generator.structured_ns_geometry_jets import geometry_mask_derivatives
    field = _field(mms=mms, polygons=_POLYGONS[-1], device="cuda")
    coordinates = _coordinates().to("cuda").requires_grad_(True)
    reference_mask, reference_indicator = field.geometry_masks(coordinates)
    reference = _autograd_jets(reference_mask, coordinates, 3)
    actual, indicator = geometry_mask_derivatives(field, coordinates)
    for key in actual:
        torch.testing.assert_close(actual[key], reference[key], rtol=2e-10, atol=1e-9)
    torch.testing.assert_close(indicator, reference_indicator, rtol=2e-14, atol=2e-15)
