"""Share coordinate jets only within the points a VPFP Oracle was asked for.

The pinned legacy module still owns every physical formula.  This wrapper
changes only exact coordinate reuse and batching: it never fills a Cartesian
product, changes the decoder precision, or caches state across calls/cases.
"""

from __future__ import annotations

import numpy as np


_MACRO_POINT_DIMS = {
    "rho": 0,
    "rho_derivatives": 1,
    "phi": 0,
    "electric": 0,
    "rho_coefficients": 0,
    "coefficients": 0,
    "coefficient_derivatives": 1,
}
_MACRO_CHUNK_SIZE = 512


def _unique_macro_jet(analytic, model, spacetime, latent):
    """Bound all-unique sparse-query memory; each unique row is evaluated once."""

    rows = [
        analytic.analytic_structured_macro_jet(
            model, spacetime[start : start + _MACRO_CHUNK_SIZE], latent
        )
        for start in range(0, len(spacetime), _MACRO_CHUNK_SIZE)
    ]
    if len(rows) == 1:
        return rows[0]
    return {
        name: model.torch.cat(
            [row[name] for row in rows], dim=_MACRO_POINT_DIMS[name]
        )
        for name in rows[0]
    }


def evaluate_requested_source(
    model,
    points,
    latent,
    *,
    equation_condition=None,
    chunk_size=8192,
    return_diagnostics=False,
):
    """Return the source Tensor for ordered numerical ``[N,5]`` queries.

    This is a single-case operation: one full latent and one physical ``nu,T``
    condition are required.  Exact uniqueness is determined in the incoming
    FP64 coordinates, BEFORE conversion to the unchanged model dtype/device.
    Distinct FP64 points that happen to round to one FP32 value are not merged.
    The original pointwise assembly receives exactly N pairs, including
    duplicates, in the original order.  Coordinate inputs are numerical data;
    derivatives are the original analytic jets, not autograd through unique.

    With ``return_diagnostics=True``, return ``(source, diagnostics)``.  Unique
    counts are sums over actual input chunks, not a claim of global deduplication.
    No CPU output conversion or public-query accounting happens in this helper.
    Ordinary latent/condition autograd remains available to internal callers.
    """

    # Import lazily to avoid the runtime/module import cycle.  The loader checks
    # the immutable physical-module SHA bundle before exposing this module.
    from .perturbation_backend import _legacy_api, _validated_oracle_points

    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or not 1 <= chunk_size <= 262144
    ):
        raise ValueError("source chunk_size must be an integer in 1..262144")
    query = _validated_oracle_points(points, width=5, name="source")
    api = _legacy_api()
    analytic = getattr(api, "analytic_source", None)
    if analytic is None:
        raise RuntimeError(
            "batched VPFP source requires the SHA-checked analytic_source module; "
            "test runtimes must explicitly provide or replace the batched kernel"
        )
    analytic.require_analytic_structured_source(model)
    torch = model.torch
    parameter = next(model.parameters())
    # count=1 rejects point-varying conditions: a temperature-dependent basis
    # must never be shared across different physical cases/conditions.
    condition = analytic._condition_rows(model, equation_condition, count=1)
    if not bool(torch.isfinite(condition).all()):
        raise ValueError("source equation condition must be finite")
    outputs = []
    chunks = []
    with analytic.ieee_float32_coordinate_matmul(model):
        for start in range(0, len(query), chunk_size):
            requested = query[start : start + chunk_size]
            spacetime, space_inverse = np.unique(
                requested[:, :3], axis=0, return_inverse=True
            )
            velocity, velocity_inverse = np.unique(
                requested[:, 3:], axis=0, return_inverse=True
            )
            spacetime = torch.as_tensor(
                spacetime, dtype=parameter.dtype, device=parameter.device
            )
            velocity = torch.as_tensor(
                velocity, dtype=parameter.dtype, device=parameter.device
            )
            space_index = torch.as_tensor(
                space_inverse, dtype=torch.long, device=parameter.device
            )
            velocity_index = torch.as_tensor(
                velocity_inverse, dtype=torch.long, device=parameter.device
            )
            unique_macro = _unique_macro_jet(
                analytic, model, spacetime, latent
            )
            macro = {
                name: value.index_select(_MACRO_POINT_DIMS[name], space_index)
                for name, value in unique_macro.items()
            }
            unique_basis = analytic.hermite_basis_coordinate_jet(
                torch,
                velocity,
                modes=model.hermite_modes,
                temperature=condition[0, 1],
            )
            basis = tuple(
                value.index_select(0, velocity_index) for value in unique_basis
            )
            assembled = analytic._assemble_pointwise(
                torch,
                model=model,
                velocity=velocity.index_select(0, velocity_index),
                condition=condition.expand(len(requested), -1),
                macro=macro,
                basis=basis,
            )
            outputs.append(assembled["source"])
            if return_diagnostics:
                chunks.append({
                    "start": start,
                    "requested_points": len(requested),
                    "unique_spacetime_points": len(spacetime),
                    "unique_velocity_points": len(velocity),
                    "unique_requested_pairs": len(np.unique(requested, axis=0)),
                    "evaluated_pairs": len(requested),
                })
    source = torch.cat(outputs, dim=0)
    if not return_diagnostics:
        return source
    counts = {
        name: sum(chunk[name] for chunk in chunks)
        for name in (
            "requested_points", "unique_spacetime_points",
            "unique_velocity_points", "unique_requested_pairs", "evaluated_pairs",
        )
    }
    return source, {
        "schema": "vpfp_requested_coordinate_reuse_v1",
        **counts,
        "unique_count_scope": "sum_within_input_chunks",
        "chunk_size": chunk_size,
        "macro_chunk_size": _MACRO_CHUNK_SIZE,
        "chunks": chunks,
        "assembly": "original_pointwise_requested_pairs_only",
        "coordinate_derivative_backend": analytic.ANALYTIC_SOURCE_BACKEND,
        "coordinate_dtype": str(parameter.dtype),
        "coordinate_deduplication_dtype": "float64",
        "cross_call_cache": False,
        "cross_case_cache": False,
        "cartesian_completion": False,
    }
