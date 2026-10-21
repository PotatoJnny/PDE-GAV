"""Versioned numerical execution options, separate from the frozen weak PDE."""
from __future__ import annotations

from collections.abc import Mapping


SOURCE_BATCHED_GEOMETRY_V2 = "batched_geometry_v2"
SOURCE_BATCHED_GEOMETRY_G0_V3 = "batched_geometry_g0_v3"


def validated_source_execution_options(options=None):
    """Versioned exact-coordinate grouping; v3 also batches initial g values.

    None and v2 preserve their historical g0 path. The v3 g0 path shares only
    case-local geometry, not sampled answers, and leaves macro truth unchanged.
    """
    if options is None:
        return None
    if not isinstance(options, Mapping):
        raise TypeError("weak Oracle execution options must be a mapping or None")
    expected = {"variant", "source_request_block", "geometry_batch_size", "geometry_cache_size"}
    variant = options.get("variant")
    if variant == SOURCE_BATCHED_GEOMETRY_G0_V3:
        expected.add("g0_request_block")
    if set(options) != expected or variant not in (SOURCE_BATCHED_GEOMETRY_V2, SOURCE_BATCHED_GEOMETRY_G0_V3):
        raise ValueError("unknown weak Oracle execution profile")
    limits = [("source_request_block", 8192), ("geometry_batch_size", 32),
              ("geometry_cache_size", 512)]
    if variant == SOURCE_BATCHED_GEOMETRY_G0_V3:
        limits.append(("g0_request_block", 8192))
    for name, limit in limits:
        value = options[name]
        if type(value) is not int or not 1 <= value <= limit:
            raise ValueError(f"{name} must be an integer in 1..{limit}")
    return dict(options)
