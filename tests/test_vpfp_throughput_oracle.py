"""Exact coordinate-factor checks and tiny CPU kernel comparisons, not training."""
import numpy as np
import pytest

from pde_agent_verifier_solver.plugins.vpfp.structured_oracle import StructuredOracle
from pde_agent_verifier_solver.plugins.vpfp.throughput_oracle import ThroughputOracle, factor_requested
from tests.test_vpfp_derivative_features import tiny
from tests.test_vpfp_batched_source import decoder, _numpy, _queries


def ordered_query():
    space = np.array([[.1, .11, .19], [.1, .23, .19], [.1, .23, .41], [.5, .17, .31], [.8, .29, .17]])
    velocity = np.array([[-1.3, -.9], [-1.3, .3], [.17, -.9], [.17, .3], [.8, -.9], [.8, .3]])
    return np.concatenate((np.repeat(space, len(velocity), axis=0), np.tile(velocity, (len(space), 1))), axis=1)


def assert_unique_equivalence(query, factors):
    assert factors is not None
    space, si = np.unique(query[:, :3], axis=0, return_inverse=True)
    velocity, vi = np.unique(query[:, 3:], axis=0, return_inverse=True)
    for actual, expected in ((factors.spacetime, space), (factors.space_inverse, si),
                             (factors.velocity, velocity), (factors.velocity_inverse, vi)):
        np.testing.assert_array_equal(actual, expected)
    reconstructed = np.concatenate((factors.spacetime[factors.space_inverse],
                                    factors.velocity[factors.velocity_inverse]), axis=1)
    np.testing.assert_array_equal(reconstructed, query)


@pytest.mark.parametrize("first,last", [(0, 30), (1, 30), (0, 29), (2, 28), (7, 27), (0, 1), (3, 5)])
def test_ordered_factorization_matches_unique_including_partial_edges(first, last):
    query = ordered_query()[first:last]
    before = query.copy()
    assert_unique_equivalence(query, factor_requested(query))
    np.testing.assert_array_equal(query, before)


@pytest.mark.parametrize("offset", [0, 13, 262131])
def test_actual_0024_flattened_262144_query_shape_without_full_grid_allocation(offset):
    # 97x33x33x20x20, emitted in chunks; never allocate the full phase-space grid.
    axis = (np.linspace(0., 1., 97), np.linspace(0., 1., 33), np.linspace(0., 1., 33),
            np.polynomial.hermite.hermgauss(20)[0], np.polynomial.hermite.hermgauss(20)[0])
    indices = np.arange(offset, offset + 262144)
    columns = [None] * 5
    for dimension in range(4, -1, -1):
        columns[dimension] = axis[dimension][indices % len(axis[dimension])]
        indices //= len(axis[dimension])
    query = np.column_stack(columns)
    assert query.shape == (262144, 5)
    assert_unique_equivalence(query, factor_requested(query))


@pytest.mark.parametrize("kind", ["shuffled", "duplicates", "velocity-permuted", "interior-hole", "one-off-coordinate"])
def test_irregular_patterns_do_not_enter_ordered_fast_path(kind):
    query = ordered_query()
    if kind == "shuffled": query = query[np.random.default_rng(17).permutation(len(query))]
    elif kind == "duplicates": query = np.insert(query, 9, query[9], axis=0)
    elif kind == "velocity-permuted": query = query.reshape(5, 6, 5)[:, ::-1].reshape(-1, 5)
    elif kind == "interior-hole": query = np.delete(query, 9, axis=0)
    else: query[9, 4] = np.nextafter(query[9, 4], 1.)
    assert factor_requested(query) is None


def test_off_grid_and_adjacent_fp64_keys_remain_distinct():
    query = ordered_query()
    query[:, 0] *= .917
    query[:, 1:3] += .001371
    query[:, 3] = np.where(query[:, 3] < 0, .31, np.where(query[:, 3] < .5, np.nextafter(.31, 1.), .71))
    factors = factor_requested(query)
    assert_unique_equivalence(query, factors)
    assert factors.velocity.dtype == np.float64
    assert factors.velocity[2, 0] != factors.velocity[0, 0]
    assert np.float32(factors.velocity[2, 0]) == np.float32(factors.velocity[0, 0])


@pytest.mark.parametrize("points", [[], np.zeros((2, 4)), np.zeros((2, 6)), np.zeros(5)])
def test_invalid_query_shapes_are_rejected(points):
    with pytest.raises(ValueError, match="nonempty"):
        factor_requested(points)


@pytest.mark.parametrize("point_block", [2, 512, 1024])
@pytest.mark.parametrize("chunk_size", [1, 7, 262144])
def test_ordered_source_matches_unchanged_kernels_for_fp64_and_fp32(decoder, point_block, chunk_size):
    query = ordered_query()[2:-2]
    before = query.copy()
    baseline = StructuredOracle(decoder.model, decoder.latents[1], decoder.conditions[1])
    variant = ThroughputOracle(decoder.model, decoder.latents[1], decoder.conditions[1], point_block=point_block)
    with decoder.model.torch.no_grad():
        expected = _numpy(baseline.source(query, chunk_size=min(chunk_size, 8192)))
        actual = _numpy(variant.source(query, chunk_size=chunk_size))
    np.testing.assert_allclose(actual, expected, rtol=decoder.rtol, atol=decoder.atol)
    np.testing.assert_array_equal(query, before)
    info = variant.cache_info()
    assert actual.shape == (len(query),)
    assert info["requested_points"] == info["assembled_pairs"] == len(query)
    assert info["cartesian_pairs"] + info["sparse_pairs"] == len(query)
    assert info["macro_point_block"] == point_block
    assert info["cached_final_source_arrays"] is False
    if chunk_size == 262144:
        assert info["mixed_cartesian_blocks"] == 1
        assert info["sparse_pairs"] == 8
        assert info["grouping_fallback_blocks"] == 0


@pytest.mark.parametrize("kind", ["sparse", "sparse_shuffled_duplicates", "cartesian"])
def test_arbitrary_query_fallback_is_original_path_and_order(decoder, monkeypatch, kind):
    query = _queries(kind)
    baseline = StructuredOracle(decoder.model, decoder.latents[1], decoder.conditions[1])
    variant = ThroughputOracle(decoder.model, decoder.latents[1], decoder.conditions[1], point_block=1024)
    with decoder.model.torch.no_grad():
        expected = _numpy(baseline.source(query))
    blocks = []
    original = variant._macro
    def macro(points):
        blocks.append(variant.POINT_BLOCK)
        return original(points)
    monkeypatch.setattr(variant, "_macro", macro)
    with decoder.model.torch.no_grad():
        actual = _numpy(variant.source(query))
    np.testing.assert_array_equal(actual, expected)
    assert blocks == [512]
    assert variant.POINT_BLOCK == 1024
    assert variant.cache_info()["grouping_fallback_blocks"] == 1


def test_full_ordered_blocks_avoid_pair_scatter_and_incomplete_computation(decoder, monkeypatch):
    query = ordered_query()
    variant = ThroughputOracle(decoder.model, decoder.latents[1], decoder.conditions[1])
    def forbidden(*args, **kwargs):
        raise AssertionError("complete ordered query does not need sparse assembly")
    monkeypatch.setattr(variant, "_sparse_source", forbidden)
    with decoder.model.torch.no_grad():
        result = variant.source(query)
    assert result.shape == (len(query),)
    assert variant.cache_info()["cartesian_blocks"] == 1
    assert variant.cache_info()["assembled_pairs"] == len(query)


@pytest.mark.parametrize("field", ["rho", "g"])
def test_initial_api_remains_inherited_and_unchanged(decoder, field):
    query = ordered_query()
    query[:, 0] = 0.
    baseline = StructuredOracle(decoder.model, decoder.latents[1], decoder.conditions[1])
    variant = ThroughputOracle(decoder.model, decoder.latents[1], decoder.conditions[1])
    np.testing.assert_array_equal(_numpy(variant.initial(query, field)), _numpy(baseline.initial(query, field)))


@pytest.mark.parametrize("chunk", [0, -1, 262145, True, 1.5])
def test_block_bound_is_explicit(decoder, chunk):
    variant = ThroughputOracle(decoder.model, decoder.latents[1], decoder.conditions[1])
    with pytest.raises(ValueError, match="chunk_size"):
        variant.source(ordered_query(), chunk_size=chunk)


@pytest.mark.parametrize("block", [0, -1, 2049, True, 1.5])
def test_macro_block_bound_is_explicit(decoder, block):
    with pytest.raises(ValueError, match="point_block"):
        ThroughputOracle(decoder.model, decoder.latents[1], decoder.conditions[1], point_block=block)
