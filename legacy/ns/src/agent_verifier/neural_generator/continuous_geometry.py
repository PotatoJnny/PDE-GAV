from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


LEGACY_CONTINUOUS_GEOMETRY_ABI_VERSION = "continuous_geometry_convex_polygons_v1"
CONTINUOUS_GEOMETRY_ABI_VERSION = "continuous_geometry_convex_polygons_v2"
SUPPORTED_CONTINUOUS_GEOMETRY_ABI_VERSIONS = {
    LEGACY_CONTINUOUS_GEOMETRY_ABI_VERSION,
    CONTINUOUS_GEOMETRY_ABI_VERSION,
}


@dataclass(frozen=True)
class ContinuousGeometrySpec:
    """Geometry data sufficient to evaluate smooth no-slip masks at any point."""

    domain_bounds: tuple[float, float, float, float]
    obstacle_polygons: tuple[tuple[tuple[float, float], ...], ...]
    abi_version: str = CONTINUOUS_GEOMETRY_ABI_VERSION
    outer_boundary: str = "no_slip"

    def validate(self) -> None:
        if self.outer_boundary not in ("no_slip", "manufactured_trace"):
            raise ValueError("outer_boundary must be no_slip or manufactured_trace")
        if self.outer_boundary == "manufactured_trace" and self.abi_version != CONTINUOUS_GEOMETRY_ABI_VERSION:
            raise ValueError("manufactured outer traces require the current continuous geometry ABI")
        if self.abi_version not in SUPPORTED_CONTINUOUS_GEOMETRY_ABI_VERSIONS:
            raise ValueError(f"unsupported continuous geometry ABI: {self.abi_version}")
        x_min, x_max, y_min, y_max = self.domain_bounds
        if not (x_max > x_min and y_max > y_min):
            raise ValueError("continuous geometry domain bounds are invalid")
        for polygon in self.obstacle_polygons:
            _validate_convex_polygon(polygon)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = {
            "abi_version": self.abi_version,
            "domain_bounds": [float(value) for value in self.domain_bounds],
            "obstacle_polygons": [
                [[float(x), float(y)] for x, y in polygon]
                for polygon in self.obstacle_polygons
            ],
        }
        # Preserve the exact serialized contract of existing numerical datasets.
        if self.outer_boundary != "no_slip":
            payload["outer_boundary"] = self.outer_boundary
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ContinuousGeometrySpec":
        spec = cls(
            abi_version=str(payload.get("abi_version", "")),
            outer_boundary=payload.get("outer_boundary", "no_slip"),
            domain_bounds=tuple(float(value) for value in payload["domain_bounds"]),
            obstacle_polygons=tuple(
                tuple((float(point[0]), float(point[1])) for point in polygon)
                for polygon in payload.get("obstacle_polygons", [])
            ),
        )
        spec.validate()
        return spec

    def normalized_obstacle_polygons(
        self,
    ) -> tuple[tuple[tuple[float, float], ...], ...]:
        x_min, x_max, y_min, y_max = self.domain_bounds
        width = x_max - x_min
        height = y_max - y_min
        return tuple(
            tuple(((x - x_min) / width, (y - y_min) / height) for x, y in polygon)
            for polygon in self.obstacle_polygons
        )


def continuous_geometry_spec_from_contract(geometry) -> ContinuousGeometrySpec:
    """Adapt a supported geometry provider to the minimal continuous ABI."""

    if isinstance(geometry, dict):
        payload = dict(geometry)
        config = dict(payload.get("config", {}))
    else:
        payload = dict(geometry.to_json_dict())
        config_object = getattr(geometry, "config", None)
        config = {
            name: getattr(config_object, name)
            for name in ("x_min", "x_max", "y_min", "y_max")
            if config_object is not None and hasattr(config_object, name)
        }
    direct = payload.get("continuous_geometry")
    if direct is not None:
        return ContinuousGeometrySpec.from_dict(dict(direct))
    bounds = _continuous_query_bounds(geometry, config=config)
    components = payload.get("obstacle_polygons")
    if components is None:
        vertices = payload.get("vertices")
        if not isinstance(vertices, dict):
            raise ValueError(
                "supported continuous geometry must provide continuous_geometry, "
                "obstacle_polygons, or polygon vertices"
            )
        components = [vertices[name] for name in sorted(vertices)]
    spec = ContinuousGeometrySpec(
        domain_bounds=bounds,
        outer_boundary=payload.get("outer_boundary", "no_slip"),
        obstacle_polygons=tuple(
            tuple((float(point[0]), float(point[1])) for point in polygon)
            for polygon in components
        ),
    )
    spec.validate()
    return spec


def continuous_no_slip_masks(
    normalized_coords,
    *,
    geometry: ContinuousGeometrySpec,
    transition: float,
    _prepared_polygons=None,
    _interior_c3=False,
):
    """Return the streamfunction mask and hard fluid-domain indicator.

    The default preserves v3-v5 exactly. MMS v6 uses the separate public
    ``interior_c3_no_slip_masks`` entry point.
    """

    if _prepared_polygons is None:
        geometry.validate()
    if float(transition) <= 0.0:
        raise ValueError("continuous geometry transition must be positive")
    x = normalized_coords[..., 0]
    y = normalized_coords[..., 1]
    outer = (
        no_slip_unit_ramp(x / float(transition))
        * no_slip_unit_ramp((1.0 - x) / float(transition))
        * no_slip_unit_ramp(y / float(transition))
        * no_slip_unit_ramp((1.0 - y) / float(transition))
    ) if geometry.outer_boundary == "no_slip" else normalized_coords.new_ones(x.shape)
    obstacle_clear = normalized_coords.new_ones(normalized_coords.shape[:-1])
    fluid_clear = normalized_coords.new_ones(normalized_coords.shape[:-1])
    if geometry.abi_version == CONTINUOUS_GEOMETRY_ABI_VERSION:
        fluid_clear = (
            ((x >= 0.0) & (x <= 1.0) & (y >= 0.0) & (y <= 1.0))
            if geometry.outer_boundary == "manufactured_trace" else
            ((x > 0.0) & (x < 1.0) & (y > 0.0) & (y < 1.0))
        ).to(dtype=normalized_coords.dtype)
    for outward_distances in _polygon_distances(normalized_coords, geometry, _prepared_polygons):
        if _interior_c3:
            smooth_clear = interior_c3_polygon_clear_mask(outward_distances / float(transition))
        else:
            edge_clear = no_slip_unit_ramp(outward_distances / float(transition))
            smooth_clear = 1.0 - (1.0 - edge_clear).prod(dim=-1)
        hard_clear = (outward_distances.amax(dim=-1) > 0.0).to(
            dtype=normalized_coords.dtype
        )
        obstacle_clear = obstacle_clear * smooth_clear
        fluid_clear = fluid_clear * hard_clear
    return outer * obstacle_clear, fluid_clear



def interior_c3_no_slip_masks(normalized_coords, *, geometry, transition, _prepared_polygons=None):
    """MMS v6: C3 inside the fluid, with the original quadratic wall trace.

    For each polygon use r(||positive(d / transition)||_4). Supporting-line
    extensions are C3 because positive(d)^4 is C3 and the norm is nonzero
    everywhere outside that polygon. On a single active face this is exactly
    the old quadratic ramp. This does NOT assert C3 through physical walls or
    polygon vertices; those are boundaries of the fluid, not internal seams.
    """
    return continuous_no_slip_masks(
        normalized_coords, geometry=geometry, transition=transition,
        _prepared_polygons=_prepared_polygons, _interior_c3=True,
    )


def _interior_c3_clearance(normalized_distances):
    """Stable positive L4 norm, with a safe all-solid branch.

    Clipping individual components at one changes only already-saturated
    mask values. Detached scaling is algebraically neutral: the norm is the
    same for every positive scale, including its coordinate derivatives.
    Scaling before the fourth power avoids tiny fourth powers at wall traces.
    """
    import torch

    positive = normalized_distances.clamp(min=0.0, max=1.0)
    scale = positive.amax(dim=-1).detach()
    safe_scale = torch.where(scale > 0.0, scale, torch.ones_like(scale))
    fourth_sum = (positive / safe_scale[..., None]).pow(4).sum(dim=-1)
    safe_sum = torch.where(scale > 0.0, fourth_sum, torch.ones_like(fourth_sum))
    radius = torch.where(scale > 0.0, safe_scale * safe_sum.pow(0.25), torch.zeros_like(scale))
    return positive, radius


def interior_c3_polygon_clear_mask(normalized_distances):
    import torch

    _, radius = _interior_c3_clearance(normalized_distances)
    # Exact single-face formula also avoids cancellation of norm derivatives
    # near a wall. At an internal seam its derivatives through order 3 agree
    # with the multi-face expression. Preserve clamp's wall endpoint convention.
    multi = (normalized_distances > 0.0).sum(dim=-1) > 1
    return torch.where(
        multi, no_slip_unit_ramp(radius),
        no_slip_unit_ramp(normalized_distances.amax(dim=-1)),
    )


def _continuous_query_bounds(geometry, *, config) -> tuple[float, float, float, float]:
    if not isinstance(geometry, dict):
        x = np.asarray(getattr(geometry, "x", ()), dtype=np.float64)
        y = np.asarray(getattr(geometry, "y", ()), dtype=np.float64)
        if x.ndim == 1 and y.ndim == 1 and x.size >= 2 and y.size >= 2:
            if np.all(np.diff(x) > 0.0) and np.all(np.diff(y) > 0.0):
                return float(x[0]), float(x[-1]), float(y[0]), float(y[-1])
    return (
        float(config["x_min"]),
        float(config["x_max"]),
        float(config["y_min"]),
        float(config["y_max"]),
    )


def legacy_c3_continuous_no_slip_masks(
    normalized_coords,
    *,
    geometry: ContinuousGeometrySpec,
    transition: float,
    _prepared_polygons=None,
):
    """Reproduce the archived v2 mask with C3 continuation through solids."""

    if _prepared_polygons is None:
        geometry.validate()
    x = normalized_coords[..., 0]
    y = normalized_coords[..., 1]
    outer = (
        c3_unit_ramp(x / float(transition))
        * c3_unit_ramp((1.0 - x) / float(transition))
        * c3_unit_ramp(y / float(transition))
        * c3_unit_ramp((1.0 - y) / float(transition))
    )
    obstacle_clear = normalized_coords.new_ones(normalized_coords.shape[:-1])
    for distances in _polygon_distances(normalized_coords, geometry, _prepared_polygons):
        edge_clear = c3_unit_ramp(distances / float(transition))
        obstacle_clear = obstacle_clear * (
            1.0 - (1.0 - edge_clear).prod(dim=-1)
        )
    return outer * obstacle_clear, obstacle_clear


def no_slip_unit_ramp(value):
    """One-sided wall ramp with psi=O(distance^2) and C3 saturation."""

    clipped = value.clamp(min=0.0, max=1.0)
    return clipped**2 * (
        10.0 - 20.0 * clipped + 15.0 * clipped**2 - 4.0 * clipped**3
    )


def c3_unit_ramp(value):
    clipped = value.clamp(min=0.0, max=1.0)
    return clipped**4 * (
        35.0 - 84.0 * clipped + 70.0 * clipped**2 - 20.0 * clipped**3
    )


def _convex_polygon_clear_masks(normalized_coords, *, polygon, transition: float):
    outward_distances = _convex_polygon_outward_distances(
        normalized_coords,
        polygon=polygon,
    )
    edge_clear = no_slip_unit_ramp(outward_distances / float(transition))
    inside_all_edge_transitions = (1.0 - edge_clear).prod(dim=-1)
    smooth_clear = 1.0 - inside_all_edge_transitions
    hard_clear = (outward_distances.amax(dim=-1) > 0.0).to(
        dtype=normalized_coords.dtype
    )
    return smooth_clear, hard_clear


def _convex_polygon_outward_distances(normalized_coords, *, polygon):
    prepared = _prepare_polygon(
        polygon, dtype=normalized_coords.dtype, device=normalized_coords.device,
    )
    return _prepared_polygon_outward_distances(normalized_coords, prepared)


def _prepare_continuous_polygons(geometry, *, dtype, device):
    """Build immutable geometry constants, with no query or autograd graph.

    Field construction calls this once. Standalone public mask functions keep
    validating supplied geometry on every call.
    """
    geometry.validate()
    return tuple(
        _prepare_polygon(polygon, dtype=dtype, device=device)
        for polygon in geometry.normalized_obstacle_polygons()
    )


def _polygon_distances(normalized_coords, geometry, prepared_polygons):
    if prepared_polygons is None:
        for polygon in geometry.normalized_obstacle_polygons():
            yield _convex_polygon_outward_distances(normalized_coords, polygon=polygon)
    else:
        for prepared in prepared_polygons:
            yield _prepared_polygon_outward_distances(normalized_coords, prepared)


def _prepare_polygon(polygon, *, dtype, device):
    import torch

    area = _signed_area(np.asarray(polygon, dtype=np.float64))
    orientation = 1.0 if area > 0.0 else -1.0
    # Later coordinate derivatives save these constants for backward. They must
    # be ordinary tensors even if construction occurs under inference mode.
    with torch.inference_mode(False), torch.no_grad():
        vertices = torch.as_tensor(polygon, dtype=dtype, device=device)
        edges = vertices.roll(-1, dims=0) - vertices
        lengths = torch.linalg.vector_norm(edges, dim=-1)
    return vertices, edges, lengths, orientation


def _prepared_polygon_outward_distances(normalized_coords, prepared):
    vertices, edges, lengths, orientation = prepared
    # Keep the original vectorized signed-distance arithmetic and its exact
    # coordinate derivatives. Only invariant tensor construction is hoisted.
    relative = normalized_coords[..., None, :2] - vertices
    cross = edges[:, 0] * relative[..., 1] - edges[:, 1] * relative[..., 0]
    return -orientation * cross / lengths


def _validate_convex_polygon(polygon) -> None:
    vertices = np.asarray(polygon, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or vertices.shape[0] < 3:
        raise ValueError("continuous obstacle polygons must contain at least three 2D vertices")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("continuous obstacle polygon vertices must be finite")
    area = _signed_area(vertices)
    if abs(area) <= 1.0e-12:
        raise ValueError("continuous obstacle polygon has zero area")
    orientation = 1.0 if area > 0.0 else -1.0
    edge_crosses = []
    for index in range(vertices.shape[0]):
        first = vertices[(index + 1) % vertices.shape[0]] - vertices[index]
        second = vertices[(index + 2) % vertices.shape[0]] - vertices[
            (index + 1) % vertices.shape[0]
        ]
        if np.linalg.norm(first) <= 1.0e-12:
            raise ValueError("continuous obstacle polygon contains a repeated vertex")
        edge_crosses.append(first[0] * second[1] - first[1] * second[0])
    if any(orientation * value < -1.0e-12 for value in edge_crosses):
        raise ValueError(
            "continuous convex-polygon geometry requires convex polygon components"
        )


def _signed_area(vertices: np.ndarray) -> float:
    array = np.asarray(vertices, dtype=np.float64)
    return 0.5 * float(
        np.sum(
            array[:, 0] * np.roll(array[:, 1], -1)
            - array[:, 1] * np.roll(array[:, 0], -1)
        )
    )
