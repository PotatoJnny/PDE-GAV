"""Opt-in ordered-query grouping for the VPFP numerical Oracle.

No runtime registers this class automatically. It keeps the existing numerical
kernels and dtype, but amortizes CPU grouping and launches over a larger request
block. Nonconforming queries retain StructuredOracle's bounded fallback. No
unrequested Cartesian pairs or final-answer caches are introduced.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .structured_oracle import StructuredOracle
from .perturbation_backend import VPFPLegacyStructuredRuntime


def oracle_execution_options(*, oracle_variant="structured_reuse_v1",
                             source_request_block=8192, macro_point_block=512):
    """Validate fixed execution choices; never infer them from online outputs."""
    if oracle_variant not in ("structured_reuse_v1", "ordered_throughput_v1"):
        raise ValueError("unknown VPFP oracle_variant")
    for name, value, maximum in (("source_request_block", source_request_block, 262144),
                                 ("macro_point_block", macro_point_block, 2048)):
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError(f"{name} must be an integer in 1..{maximum}")
    if oracle_variant == "structured_reuse_v1" and (
            source_request_block != 8192 or macro_point_block != 512):
        raise ValueError("nondefault blocks require ordered_throughput_v1")
    return dict(oracle_variant=oracle_variant, source_request_block=source_request_block,
                macro_point_block=macro_point_block)


class ThroughputResidentRuntime(VPFPLegacyStructuredRuntime):
    """Private numerical replica with fixed blocks and the original public API.

    The parent retains the canonical runtime, carrier identity and hidden truth.
    This subclass changes only source assembly and its prepared-object factory;
    query validation, metering, initial fields and release rules stay inherited.
    """

    def _prepared_oracle_for(self, resident):
        from .perturbation_backend import _PREPARED_ORACLE_MAX_CASES
        cache, key = self._prepared_oracles, resident.case.case_id
        if key in cache:
            self._prepared_oracle_stats["hits"] += 1
            cache.move_to_end(key)
            return cache[key]
        self._prepared_oracle_stats["misses"] += 1
        if len(cache) >= _PREPARED_ORACLE_MAX_CASES:
            cache.popitem(last=False)
            self._prepared_oracle_stats["evictions"] += 1
        prepared = ThroughputOracle(self.expanded_model, resident.latent,
            resident.equation_condition, point_block=self.oracle_execution_options["macro_point_block"])
        cache[key] = prepared
        return prepared

    def evaluate_source(self, case_id, points):
        from .perturbation_backend import _validated_oracle_points, _trusted_oracle_array
        query = _validated_oracle_points(points, width=5, name="source")
        self._validate_query_times(query[:, 0])
        with self._runtime_lock:
            resident = self._resident_for_id(case_id)
            with self.api.torch.no_grad():
                values = self._prepared_oracle_for(resident).source(
                    query, chunk_size=self.oracle_execution_options["source_request_block"])
            return _trusted_oracle_array(values, (len(query),), "source")


@dataclass(frozen=True)
class RequestedFactors:
    spacetime: np.ndarray
    velocity: np.ndarray
    space_inverse: np.ndarray
    velocity_inverse: np.ndarray
    run_lengths: np.ndarray


def _strictly_increasing(rows):
    """Exact lexicographic order, with no row sorting or FP32 conversion."""
    previous, following = rows[:-1], rows[1:]
    less = np.zeros(len(previous), dtype=bool)
    equal = np.ones(len(previous), dtype=bool)
    for column in range(rows.shape[1]):
        less |= equal & (previous[:, column] < following[:, column])
        equal &= previous[:, column] == following[:, column]
    return bool(np.all(less))


def factor_requested(points):
    """Return exact factors for ordered velocity cycles, otherwise None.

    A flattened Cartesian slice may begin/end inside a velocity cycle. Every
    interior spacetime row must contain the same strictly ordered velocity set;
    only the first/last run may be a suffix/prefix. This recognizer checks actual
    FP64 coordinates, not expected grid sizes, spacing, or approximate hashes.
    """
    query = StructuredOracle._query(points)
    changes = np.any(query[1:, :3] != query[:-1, :3], axis=1)
    starts = np.r_[0, np.flatnonzero(changes) + 1]
    spacetime = query[starts, :3]
    if not _strictly_increasing(spacetime):
        return None
    lengths = np.diff(np.r_[starts, len(query)])
    longest = int(np.argmax(lengths))
    count = int(lengths[longest])
    velocity = query[starts[longest]:starts[longest] + count, 3:]
    if not _strictly_increasing(velocity) or np.any(lengths[1:-1] != count):
        return None
    offset = count - int(lengths[0])
    vi = (np.arange(len(query), dtype=np.int64) + offset) % count
    # Equality verifies BOTH partially requested edges and every full cycle.
    if not np.array_equal(query[:, 3:], velocity[vi]):
        return None
    si = np.repeat(np.arange(len(starts), dtype=np.int64), lengths)
    return RequestedFactors(spacetime, velocity, si, vi, lengths)


class ThroughputOracle(StructuredOracle):
    """Explicit experiment variant; default requests up to 262144 points/block."""

    MAX_REQUEST_BLOCK = 262144

    def __init__(self, model, latent, equation_condition, *, point_block=512):
        if type(point_block) is not int or not 1 <= point_block <= 2048:
            raise ValueError("point_block must be an integer in 1..2048")
        super().__init__(model, latent, equation_condition)
        self.POINT_BLOCK = point_block
        self._stats.update(ordered_factor_blocks=0, grouping_fallback_blocks=0)

    def source(self, points, chunk_size=262144):
        from .perturbation_backend import _legacy_api
        query = self._query(points)
        if type(chunk_size) is not int or not 1 <= chunk_size <= self.MAX_REQUEST_BLOCK:
            raise ValueError("throughput source chunk_size must be in 1..262144")
        outputs = []
        with _legacy_api().ieee_float32_coordinate_matmul(self.model):
            for start in range(0, len(query), chunk_size):
                requested = query[start:start + chunk_size]
                factors = factor_requested(requested)
                if factors is None:
                    self._stats["grouping_fallback_blocks"] += 1
                    # Arbitrary queries use the original exact grouping and
                    # 8192-point assembly limits, not Cartesian completion.
                    selected_block = self.POINT_BLOCK
                    try:
                        self.POINT_BLOCK = StructuredOracle.POINT_BLOCK
                        outputs.append(super().source(requested, chunk_size=min(chunk_size, 8192)))
                    finally:
                        self.POINT_BLOCK = selected_block
                    continue
                self._stats["ordered_factor_blocks"] += 1
                outputs.append(self._ordered_source(factors))
        return self.torch.cat(outputs)

    def _ordered_source(self, factors):
        spacetime, velocities = factors.spacetime, factors.velocity
        si, vi, lengths = factors.space_inverse, factors.velocity_inverse, factors.run_lengths
        count, requested_count = len(velocities), len(si)
        macro = self._macro(spacetime)
        velocity, basis = self._velocity_basis(velocities)
        complete = np.flatnonzero(lengths == count)
        self._stats["requested_points"] += requested_count
        self._stats["assembled_pairs"] += requested_count
        self._stats["cartesian_pairs"] += len(complete) * count
        if len(complete) == len(spacetime):
            self._stats["cartesian_blocks"] += 1
            return self._cartesian_source(macro, velocity, basis).flatten()
        self._stats["mixed_cartesian_blocks"] += 1
        compact = self._macro_subset(macro, self._tensor(complete, indices=True))
        full = self._cartesian_source(compact, velocity, basis).flatten()
        # The recognizer proved that only the first/last row can be partial.
        # Their requested points are evaluated pointwise and concatenated in
        # original order, without scatter kernels or computing missing pairs.
        first = int(lengths[0]) if lengths[0] < count else 0
        last = int(lengths[-1]) if lengths[-1] < count else 0
        positions = np.r_[np.arange(first), np.arange(requested_count - last, requested_count)]
        sparse = self._sparse_source(macro, velocity, basis, si[positions], vi[positions])
        self._stats["sparse_pairs"] += len(positions)
        return self.torch.cat((sparse[:first], full, sparse[first:]))

    def cache_info(self):
        return dict(super().cache_info(), variant="ordered_throughput_v1",
                    macro_point_block=self.POINT_BLOCK,
                    maximum_request_block=self.MAX_REQUEST_BLOCK)
