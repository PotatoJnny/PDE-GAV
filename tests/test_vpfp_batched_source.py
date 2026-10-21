"""Small real-decoder numerical equivalence tests, not trained-model audits.

Tolerances are fixed before execution: FP64 (rtol=2e-10, atol=2e-11)
and FP32 (rtol=5e-5, atol=2e-6). The reference is the unchanged pinned
pointwise analytic source, not another call to the optimized implementation.
"""

from dataclasses import replace
import importlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from pde_agent_verifier_solver.plugins.vpfp import batched_source as batched
from tests.test_vpfp_derivative_features import tiny


@pytest.fixture(params=("float64", "float32"))
def decoder(request, tiny):
    model = tiny.model
    if request.param == "float32":
        model = tiny.api.build_deterministic_direct_model(
            replace(model.config, dtype="float32"), device="cpu",
        )
    return SimpleNamespace(
        model=model, api=tiny.api, latents=tiny.latents,
        conditions=tiny.conditions, precision=request.param,
        rtol=2e-10 if request.param == "float64" else 5e-5,
        atol=2e-11 if request.param == "float64" else 2e-6,
    )


def _numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _queries(kind):
    spacetime = np.asarray([
        [0., .113, .271], [.013, .317, .419], [.25, .523, .631],
        [.451, .739, .827], [.5, .919, 1.037],
        [.997, 1.151, 1.263], [1., 1.379, 1.487],
    ])
    velocity = np.asarray([
        [0., 0.], [.23, -.47], [1.3, -2.1], [6.25, -7.], [-5.8, 6.1],
    ])
    if kind == "cartesian":
        pairs = [(i, j) for i in range(len(spacetime)) for j in range(len(velocity))]
    else:
        pairs = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4),
                 (5, 0), (6, 1), (0, 2), (2, 4), (6, 0)]
        if kind == "sparse_shuffled_duplicates":
            pairs = [pairs[i] for i in (8, 3, 8, 0, 7, 1, 9, 2, 6, 5, 4, 3, 0)]
    return np.asarray([np.concatenate((spacetime[i], velocity[j])) for i, j in pairs])


def _reference(decoder, points, *, latent=None, condition=None):
    analytic = importlib.import_module("agent_verifier.vpfp.analytic_source")
    latent = decoder.latents[1] if latent is None else latent
    condition = decoder.conditions[1] if condition is None else condition
    rows = []
    with analytic.ieee_float32_coordinate_matmul(decoder.model), decoder.model.torch.no_grad():
        # This is the frozen runtime's original 512-point reference path.
        for start in range(0, len(points), 512):
            result = analytic.manufacture_standard_vpfp_source_analytic(
                decoder.model, points[start:start+512], latent,
                equation_condition=condition,
            )
            rows.append(_numpy(result["source"]))
    return np.concatenate(rows)


def _optimized(decoder, points, *, chunk_size=8192, latent=None, condition=None):
    latent = decoder.latents[1] if latent is None else latent
    condition = decoder.conditions[1] if condition is None else condition
    result, diagnostic = batched.evaluate_requested_source(
        decoder.model, points, latent, equation_condition=condition,
        chunk_size=chunk_size, return_diagnostics=True,
    )
    json.dumps(diagnostic, allow_nan=False)
    return _numpy(result), diagnostic


def _unique_counts(points, chunk_size):
    totals = {"unique_spacetime_points": 0, "unique_velocity_points": 0,
              "unique_requested_pairs": 0}
    for start in range(0, len(points), chunk_size):
        chunk = points[start:start+chunk_size]
        totals["unique_spacetime_points"] += len(np.unique(chunk[:, :3], axis=0))
        totals["unique_velocity_points"] += len(np.unique(chunk[:, 3:], axis=0))
        totals["unique_requested_pairs"] += len(np.unique(chunk, axis=0))
    return totals


@pytest.mark.parametrize("kind", ("cartesian", "sparse", "sparse_shuffled_duplicates"))
@pytest.mark.parametrize("chunk_size", (1, 7, 512))
def test_requested_values_match_pinned_source_for_arbitrary_order_and_batch_size(
    decoder, kind, chunk_size,
):
    points = _queries(kind)
    before = points.copy()
    expected = _reference(decoder, points)
    actual, diagnostic = _optimized(decoder, points, chunk_size=chunk_size)
    np.testing.assert_allclose(actual, expected, rtol=decoder.rtol, atol=decoder.atol)
    np.testing.assert_array_equal(points, before)
    assert actual.shape == (len(points),) and np.isfinite(actual).all()
    assert diagnostic["requested_points"] == len(points)
    assert diagnostic["evaluated_pairs"] == len(points)
    for name, expected_count in _unique_counts(points, chunk_size).items():
        assert diagnostic[name] == expected_count
    assert diagnostic["unique_count_scope"] == "sum_within_input_chunks"
    assert len(diagnostic["chunks"]) == (len(points)+chunk_size-1)//chunk_size


@pytest.mark.parametrize("kind", ("cartesian", "sparse", "sparse_shuffled_duplicates"))
def test_actual_kernels_reuse_only_requested_subcoordinates_without_cartesian_completion(
    decoder, monkeypatch, kind,
):
    points = _queries(kind)
    expected = _reference(decoder, points)
    analytic = importlib.import_module("agent_verifier.vpfp.analytic_source")
    original_macro = analytic.analytic_structured_macro_jet
    original_basis = analytic.hermite_basis_coordinate_jet
    original_assemble = analytic._assemble_pointwise
    observed = {"spacetime": [], "velocity": [], "assembled_points": []}

    def macro_spy(model, coordinates, latent):
        observed["spacetime"].append(_numpy(coordinates).copy())
        return original_macro(model, coordinates, latent)

    def basis_spy(torch, coordinates, **kwargs):
        observed["velocity"].append(_numpy(coordinates).copy())
        return original_basis(torch, coordinates, **kwargs)

    def assemble_spy(torch, **kwargs):
        observed["assembled_points"].append(len(kwargs["velocity"]))
        result = original_assemble(torch, **kwargs)
        # Check the actual returned source uses the same VPFP signs/factors,
        # including electric acceleration, bath temperature and collision nu.
        gradient = result["f_gradient"]
        velocity = kwargs["velocity"]
        condition = kwargs["condition"]
        electric = kwargs["macro"]["electric"]
        collision = (2*result["f"] + velocity[:, 0]*gradient[:, 3]
                     + velocity[:, 1]*gradient[:, 4]
                     + condition[:, 1]*result["f_velocity_laplacian"])
        reconstructed = (gradient[:, 0] + velocity[:, 0]*gradient[:, 1]
                         + velocity[:, 1]*gradient[:, 2]
                         + electric[:, 0]*gradient[:, 3]
                         + electric[:, 1]*gradient[:, 4] - condition[:, 0]*collision)
        np.testing.assert_allclose(_numpy(result["collision"]), _numpy(collision),
                                   rtol=decoder.rtol, atol=decoder.atol)
        np.testing.assert_allclose(_numpy(result["source"]), _numpy(reconstructed),
                                   rtol=decoder.rtol, atol=decoder.atol)
        return result

    monkeypatch.setattr(analytic, "analytic_structured_macro_jet", macro_spy)
    monkeypatch.setattr(analytic, "hermite_basis_coordinate_jet", basis_spy)
    monkeypatch.setattr(analytic, "_assemble_pointwise", assemble_spy)
    actual, diagnostic = _optimized(decoder, points)
    np.testing.assert_allclose(actual, expected, rtol=decoder.rtol, atol=decoder.atol)
    macro_points = np.concatenate(observed["spacetime"])
    velocity_points = np.concatenate(observed["velocity"])
    expected_macro = np.unique(points[:, :3], axis=0)
    expected_velocity = np.unique(points[:, 3:], axis=0)
    assert len(macro_points) == len(expected_macro) == diagnostic["unique_spacetime_points"]
    assert len(velocity_points) == len(expected_velocity) == diagnostic["unique_velocity_points"]
    # Conversion to model FP32 is identical to the original pointwise path.
    dtype = np.float64 if decoder.precision == "float64" else np.float32
    np.testing.assert_array_equal(np.unique(macro_points, axis=0), expected_macro.astype(dtype))
    np.testing.assert_array_equal(np.unique(velocity_points, axis=0), expected_velocity.astype(dtype))
    assert sum(observed["assembled_points"]) == len(points)
    if kind != "cartesian":
        assert len(points) < len(expected_macro)*len(expected_velocity)
        assert sum(observed["assembled_points"]) < len(expected_macro)*len(expected_velocity)


def test_condition_and_case_switches_do_not_reuse_stale_prepared_values(decoder):
    points = _queries("sparse")
    initial_latent = decoder.latents[1].copy()
    cases = [
        (initial_latent, np.asarray([.04, 1.], dtype=np.float32)),
        (decoder.latents[0], np.asarray([.04, 1.], dtype=np.float32)),
        (initial_latent, np.asarray([.06, 1.], dtype=np.float32)),
        (initial_latent, np.asarray([.06, 1.1], dtype=np.float32)),
        (initial_latent, np.asarray([.04, 1.], dtype=np.float32)),
    ]
    results = []
    for latent, condition in cases:
        expected = _reference(decoder, points, latent=latent, condition=condition)
        actual, _ = _optimized(decoder, points, latent=latent, condition=condition)
        np.testing.assert_allclose(actual, expected, rtol=decoder.rtol, atol=decoder.atol)
        results.append(actual)
    np.testing.assert_allclose(results[0], results[-1], rtol=decoder.rtol, atol=decoder.atol)
    # The synthetic cases really differ; a constant-output stub cannot pass.
    assert not np.allclose(results[0], results[1], rtol=decoder.rtol, atol=decoder.atol)
    assert not np.allclose(results[2], results[3], rtol=decoder.rtol, atol=decoder.atol)
    np.testing.assert_array_equal(initial_latent, decoder.latents[1])


def test_all_unique_sparse_coordinates_cross_internal_512_macro_boundary(decoder, monkeypatch):
    count = 513
    index = np.arange(count)
    points = np.column_stack((
        .01 + .98*index/(count-1), .13 + .001*index, .17 + .002*index,
        np.asarray([.2, 1.3, -6.1])[index % 3],
        np.asarray([-.3, .7, 6.3])[index % 3],
    ))
    expected = _reference(decoder, points)
    analytic = importlib.import_module("agent_verifier.vpfp.analytic_source")
    original = analytic.analytic_structured_macro_jet
    sizes = []

    def macro_spy(model, coordinates, latent):
        sizes.append(len(coordinates))
        return original(model, coordinates, latent)

    monkeypatch.setattr(analytic, "analytic_structured_macro_jet", macro_spy)
    actual, diagnostic = _optimized(decoder, points)
    np.testing.assert_allclose(actual, expected, rtol=decoder.rtol, atol=decoder.atol)
    assert sizes == [512, 1]
    assert diagnostic["unique_spacetime_points"] == count
    assert diagnostic["unique_velocity_points"] == 3
    assert diagnostic["evaluated_pairs"] == count
    assert diagnostic["cartesian_completion"] is False


def test_distinct_fp64_coordinates_are_not_approximately_deduplicated(decoder):
    points = np.asarray([[.451, .739, .827, .23, -.47],
                         [.451, .739, .827, .23, -.47]])
    # Both coordinates round to the same FP32 value, but the requested FP64
    # points are distinct and exact query accounting must retain that fact.
    points[1, 1] = np.nextafter(points[0, 1], np.inf)
    actual, diagnostic = _optimized(decoder, points)
    np.testing.assert_allclose(actual, _reference(decoder, points),
                               rtol=decoder.rtol, atol=decoder.atol)
    assert diagnostic["unique_spacetime_points"] == 2
    assert diagnostic["unique_velocity_points"] == 1
    assert diagnostic["unique_requested_pairs"] == 2
    assert diagnostic["evaluated_pairs"] == 2


def test_default_return_is_one_source_tensor_per_requested_point(decoder):
    points = _queries("sparse_shuffled_duplicates")
    actual = batched.evaluate_requested_source(
        decoder.model, points, decoder.latents[1],
        equation_condition=decoder.conditions[1],
    )
    assert decoder.model.torch.is_tensor(actual)
    assert tuple(actual.shape) == (len(points),)
    assert actual.dtype == next(decoder.model.parameters()).dtype
    np.testing.assert_allclose(_numpy(actual), _reference(decoder, points),
                               rtol=decoder.rtol, atol=decoder.atol)


@pytest.mark.parametrize("chunk_size", (0, -1, True, 2.5))
def test_invalid_batch_size_fails_before_any_source_kernel(decoder, monkeypatch, chunk_size):
    analytic = importlib.import_module("agent_verifier.vpfp.analytic_source")
    monkeypatch.setattr(analytic, "analytic_structured_macro_jet",
                        lambda *a, **k: pytest.fail("invalid batch size reached the decoder"))
    with pytest.raises((ValueError, TypeError)):
        _optimized(decoder, _queries("sparse"), chunk_size=chunk_size)


def test_nonconstant_per_point_condition_is_not_silently_shared(decoder):
    points = _queries("sparse")
    conditions = np.repeat(decoder.conditions[1:2], len(points), axis=0)
    conditions[1, 1] = 1.
    with pytest.raises(ValueError):
        _optimized(decoder, points, condition=conditions)
