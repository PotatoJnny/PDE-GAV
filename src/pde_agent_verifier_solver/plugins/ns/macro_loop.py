"""Thin NS binding for the shared continuous-solution Agent loop."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from random import Random
from types import MappingProxyType

from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    BoundMacroObjective,
    LoadedGenerator,
    MacroLoopPlugin,
    MacroQueryFactory,
    require_guidance_capability,
)

from .continuous_backend import StructuredNSManufacturedCaseSource
from .macro_matching import NS_VELOCITY_FIELDS, NSVelocityMatcher
from .public_problem import (
    NS_DEVELOPMENT_SOLUTION_SESSION_METHODS,
    NS_PUBLIC_SESSION_METHODS,
    ns_public_problem_bundle,
)


NS_VELOCITY_QUERY_PROFILES = frozenset(
    {"development", "selection", "promotion", "terminal"}
)
NS_VELOCITY_POINT_COUNTS = MappingProxyType(
    {
        "development": 32,
        "selection": 64,
        "promotion": 96,
        "terminal": 128,
    }
)


def _ns_domain(case):
    domain = case.public_data.get("domain")
    if not isinstance(domain, Mapping):
        raise ValueError("NS velocity query needs a public domain")
    interval = domain.get("time_interval")
    bounds = domain.get("bounds")
    if not isinstance(bounds, Mapping):
        raise ValueError("NS velocity query needs public spatial bounds")
    try:
        t0, t1 = (float(value) for value in interval)
        x0, x1 = (float(value) for value in bounds["x"])
        y0, y1 = (float(value) for value in bounds["y"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("NS velocity query public domain is invalid") from exc
    if not all(isfinite(value) for value in (t0, t1, x0, x1, y0, y1)):
        raise ValueError("NS velocity query public domain is non-finite")
    if t1 <= t0 or x1 <= x0 or y1 <= y0:
        raise ValueError("NS velocity query public domain is empty")
    return (
        (t0, t1),
        (x0, x1),
        (y0, y1),
        _obstacle_polygons(domain.get("geometry")),
    )



def _obstacle_polygons(geometry) -> tuple[tuple[tuple[float, float], ...], ...]:
    if not isinstance(geometry, Mapping):
        raise ValueError("NS velocity query needs public geometry")
    values = geometry.get("obstacle_polygons", geometry.get("obstacles", ()))
    if values is None:
        return ()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("NS public obstacle polygons are invalid")
    polygons = []
    for raw_polygon in values:
        if isinstance(raw_polygon, Mapping):
            raw_polygon = raw_polygon.get("vertices")
        if isinstance(raw_polygon, (str, bytes)) or not isinstance(
            raw_polygon, Sequence
        ):
            raise ValueError("NS public obstacle polygon is invalid")
        polygon = []
        for raw_point in raw_polygon:
            try:
                x, y = (float(value) for value in raw_point)
            except (TypeError, ValueError) as exc:
                raise ValueError("NS public obstacle vertex is invalid") from exc
            if not (isfinite(x) and isfinite(y)):
                raise ValueError("NS public obstacle vertex is non-finite")
            polygon.append((x, y))
        if len(polygon) < 3:
            raise ValueError("NS public obstacle polygon needs three vertices")
        polygons.append(tuple(polygon))
    return tuple(polygons)


def _inside_polygon(
    x: float,
    y: float,
    polygon: Sequence[tuple[float, float]],
) -> bool:
    """Return true for polygon interior or boundary (scoring rejects both)."""

    inside = False
    previous_x, previous_y = polygon[-1]
    for current_x, current_y in polygon:
        cross = (x - previous_x) * (current_y - previous_y) - (
            y - previous_y
        ) * (current_x - previous_x)
        if abs(cross) <= 1.0e-12 and (
            min(previous_x, current_x) - 1.0e-12
            <= x
            <= max(previous_x, current_x) + 1.0e-12
            and min(previous_y, current_y) - 1.0e-12
            <= y
            <= max(previous_y, current_y) + 1.0e-12
        ):
            return True
        if (current_y > y) != (previous_y > y):
            intersection = current_x + (y - current_y) * (
                previous_x - current_x
            ) / (previous_y - current_y)
            if x < intersection:
                inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside


def ns_velocity_point_query(
    case,
    profile: str,
    controller_seed: int = 0,
) -> AnswerQuery:
    """Sample a sealed set of physical (t,x,y) scoring points."""

    if profile not in NS_VELOCITY_QUERY_PROFILES:
        raise ValueError(f"unknown NS velocity query profile {profile!r}")
    if (
        isinstance(controller_seed, bool)
        or not isinstance(controller_seed, int)
        or controller_seed < 0
    ):
        raise ValueError("NS controller query seed must be nonnegative")
    interval, x_bounds, y_bounds, obstacles = _ns_domain(case)
    random = Random(controller_seed)
    points = []
    requested = NS_VELOCITY_POINT_COUNTS[profile]
    maximum_attempts = max(1024, 64 * requested)
    for _attempt in range(maximum_attempts):
        if len(points) == requested:
            break
        t = interval[0] + (interval[1] - interval[0]) * random.random()
        x = x_bounds[0] + (x_bounds[1] - x_bounds[0]) * random.random()
        y = y_bounds[0] + (y_bounds[1] - y_bounds[0]) * random.random()
        if any(_inside_polygon(x, y, polygon) for polygon in obstacles):
            continue
        points.append({"t": t, "x": x, "y": y})
    if len(points) != requested:
        raise ValueError("NS fluid-domain scoring rejection sampler exhausted")
    return AnswerQuery.at_points(fields=NS_VELOCITY_FIELDS, points=points)


def ns_native_velocity_query(case, profile: str) -> AnswerQuery:
    """Legacy development grid helper; official defaults use POINTS."""

    if profile not in NS_VELOCITY_QUERY_PROFILES:
        raise ValueError(f"unknown NS velocity query profile {profile!r}")
    counts = {
        "development": (3, 12, 12),
        "selection": (4, 16, 16),
        "promotion": (5, 20, 20),
        "terminal": (6, 24, 24),
    }[profile]
    return AnswerQuery.grid(
        fields=NS_VELOCITY_FIELDS,
        t=counts[0],
        x=counts[1],
        y=counts[2],
    )


def bind_ns_macro_loop(
    *,
    case_source: StructuredNSManufacturedCaseSource,
    query_factory: MacroQueryFactory | None = None,
    matcher: NSVelocityMatcher | None = None,
    guidance_backend: object | None = None,
) -> MacroLoopPlugin:
    """Bind fixed NS artifacts without introducing an NS-specific controller."""

    guidance_supported = False
    if guidance_backend is not None:
        require_guidance_capability(
            getattr(guidance_backend, "guidance_capability", None)
        )
        if getattr(guidance_backend, "case_source", None) is not case_source:
            raise ValueError("NS Guidance backend must bind the same case source")
        guidance_supported = True
    resolved_query_factory = query_factory or ns_velocity_point_query
    return MacroLoopPlugin(
        plugin_id="ns",
        public_result_kind="pde_solver",
        public_result_task="pde_solution",
        generator=LoadedGenerator.from_manufactured_source(
            case_source,
            guidance_backend=guidance_backend,
        ),
        objective=BoundMacroObjective(
            matcher=matcher or NSVelocityMatcher(),
            query_factory=resolved_query_factory,
            controller_query_factory=(
                ns_velocity_point_query if query_factory is None else None
            ),
            agent_visible_metrics=(
                "ns_velocity_field_match_v1_relative_l2",
                "velocity_x_relative_l2",
                "velocity_y_relative_l2",
            ),
            published_metric_names=(
                "velocity_x_relative_l2",
                "velocity_y_relative_l2",
            ),
        ),
        public_session_factory=case_source.public_session_factory,
        public_session_methods=NS_PUBLIC_SESSION_METHODS,
        development_solution_session_factory=(
            case_source.development_solution_session_factory
        ),
        development_solution_session_methods=(
            NS_DEVELOPMENT_SOLUTION_SESSION_METHODS
        ),
        public_problem_bundle=ns_public_problem_bundle(
            guidance_supported=guidance_supported
        ),
    )


__all__ = [
    "NS_VELOCITY_POINT_COUNTS",
    "bind_ns_macro_loop",
    "ns_native_velocity_query",
    "ns_velocity_point_query",
]
