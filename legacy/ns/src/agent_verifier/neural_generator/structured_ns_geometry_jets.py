"""Closed-form spatial derivatives of the existing polygon MMS masks.

The derivative keys refer to normalized x/y, not physical lengths.  Only
immutable polygon tensors prepared by ContinuousStructuredNSMMS are reused;
no coordinate-dependent tensor or autograd graph is retained on the field.
This module deliberately rejects the older sampled-SDF geometry path.
"""

from __future__ import annotations

from math import comb

from .continuous_geometry import (
    CONTINUOUS_GEOMETRY_ABI_VERSION,
    ContinuousGeometrySpec,
    _prepared_polygon_outward_distances,
    _interior_c3_clearance,
)


_LEGACY_C3_ABI = "continuous_structured_ns_mms_v2"
_POLYGON_ABIS = frozenset(
    f"continuous_structured_ns_mms_v{version}" for version in (2, 3, 4, 5, 6)
)
_KEYS = tuple((degree - dy, dy) for degree in range(4) for dy in range(degree + 1))
# Precompute only small integer algebra. Values and coordinate graphs are local
# to each call, including simultaneous CPU/CUDA queries on different fields.
_PRODUCT_TERMS = {
    key: tuple(
        ((ix, iy), (key[0] - ix, key[1] - iy), comb(key[0], ix) * comb(key[1], iy))
        for ix in range(key[0] + 1)
        for iy in range(key[1] + 1)
    )
    for key in _KEYS
}


def _validate(field, coordinates, order):
    import torch

    if isinstance(order, bool) or not isinstance(order, int) or not 0 <= order <= 3:
        raise ValueError("geometry derivative order must be an integer from zero to three")
    if not isinstance(coordinates, torch.Tensor) or coordinates.ndim != 2 or coordinates.shape[1] < 2:
        raise ValueError("normalized geometry coordinates must be a Tensor[N,2+]")
    geometry = getattr(field, "geometry_spec", None)
    if (not isinstance(geometry, ContinuousGeometrySpec)
            or getattr(field, "continuous_mms_abi", None) not in _POLYGON_ABIS
            or not isinstance(getattr(field, "_prepared_polygons", None), tuple)):
        raise ValueError("analytic geometry requires a prepared continuous polygon MMS field")
    # The field construction already validates the geometry and prepares these
    # constants in its precise dtype/device. Do not rescan polygons or silently
    # change their finite-precision construction during numerical queries.
    if coordinates.dtype != field.dtype or coordinates.device != field.coefficients.psi.device:
        raise ValueError("geometry coordinates must match the field dtype and device")
    return geometry, float(field.config.geometry_mask_transition), _KEYS[: (order + 1) * (order + 2) // 2]


def _unit_ramp_derivatives(value, order, *, legacy_c3):
    """Analytic derivatives with clamp's inclusive endpoint backward rule.

    Outside [0,1] all derivatives are zero. At an endpoint PyTorch differentiates
    the polynomial through clamp, whose backward is one there; in particular
    the modern ramp has second derivative 20 and third derivative -120 at zero.
    """
    clipped = value.clamp(min=0.0, max=1.0)
    if legacy_c3:
        result = [clipped**4 * (35.0 - 84.0 * clipped + 70.0 * clipped**2 - 20.0 * clipped**3)]
    else:
        result = [clipped**2 * (10.0 - 20.0 * clipped + 15.0 * clipped**2 - 4.0 * clipped**3)]
    if order == 0:
        return result
    active = ((value >= 0.0) & (value <= 1.0)).to(dtype=value.dtype)
    complement = 1.0 - clipped
    if legacy_c3:
        result.append(140.0 * clipped**3 * complement**3 * active)
        if order >= 2:
            result.append(420.0 * clipped**2 * complement**2 * (1.0 - 2.0 * clipped) * active)
        if order >= 3:
            result.append(840.0 * clipped * complement * (1.0 - 5.0 * clipped + 5.0 * clipped**2) * active)
    else:
        result.append(20.0 * clipped * complement**3 * active)
        if order >= 2:
            result.append(20.0 * complement**2 * (1.0 - 4.0 * clipped) * active)
        if order >= 3:
            result.append(-120.0 * complement * (1.0 - 2.0 * clipped) * active)
    return result


def _multiply_jets(left, right, keys):
    product = {}
    for key in keys:
        terms = _PRODUCT_TERMS[key]
        a, b, coefficient = terms[0]
        value = left[a] * right[b]
        if coefficient != 1:
            value = value * coefficient
        for a, b, coefficient in terms[1:]:
            term = left[a] * right[b]
            if coefficient != 1:
                term = term * coefficient
            value = value + term
        product[key] = value
    return product


def _wall_derivatives(coordinate, transition, order, *, legacy_c3):
    left = _unit_ramp_derivatives(coordinate / transition, order, legacy_c3=legacy_c3)
    right = _unit_ramp_derivatives((1.0 - coordinate) / transition, order, legacy_c3=legacy_c3)
    result = []
    for degree in range(order + 1):
        # Each right-ramp derivative contributes (-1)^k; the transition scale
        # is common to every term in the degree's Leibniz sum.
        value = left[0] * right[degree] * ((-1) ** degree)
        for index in range(1, degree + 1):
            coefficient = comb(degree, index) * (-1) ** (degree - index)
            value = value + left[index] * right[degree - index] * coefficient
        result.append(value / transition**degree if degree else value)
    return result



def _interior_c3_polygon_jets(distances, prepared, transition, keys, order):
    """Closed-form derivatives of r(||positive(d)||_4), normalized x/y.

    Use homogeneous norm derivatives, avoiding negative powers of sum(d^4).
    The explicit single-face branch preserves finite, nonzero wall shear.
    """
    import torch

    positive, radius = _interior_c3_clearance(distances)
    multi = (distances > 0.0).sum(dim=-1) > 1
    maximum = distances.amax(dim=-1)
    single_ramps = _unit_ramp_derivatives(maximum, order, legacy_c3=False)
    ramps = _unit_ramp_derivatives(radius, order, legacy_c3=False)
    result = {(0, 0): torch.where(multi, ramps[0], single_ramps[0])}
    if not order:
        return result
    _, edges, lengths, orientation = prepared
    slopes = (orientation * edges[:, 1] / lengths / transition,
              -orientation * edges[:, 0] / lengths / transition)
    ties = (distances == maximum[:, None]).to(dtype=distances.dtype)
    single_slopes = tuple((ties * v).sum(dim=-1) / ties.sum(dim=-1) for v in slopes)
    safe_radius = torch.where(multi, radius, torch.ones_like(radius))
    q = positive / safe_radius[:, None]
    first = tuple((q**3 * v).sum(dim=-1) for v in slopes)
    second, second_moment = {}, {}
    if order >= 2:
        for a in range(2):
            for b in range(a, 2):
                moment = (q**2 * slopes[a] * slopes[b]).sum(dim=-1)
                second_moment[a, b] = moment
                second[a, b] = 3.0 * (moment - first[a] * first[b]) / safe_radius
    for nx, ny in keys[1:]:
        axes = (0,) * nx + (1,) * ny
        degree = nx + ny
        a = axes[0]
        if degree == 1:
            value = ramps[1] * first[a]
        elif degree == 2:
            b = axes[1]
            value = ramps[2] * first[a] * first[b] + ramps[1] * second[a, b]
        else:
            b, c = axes[1:]
            third_moment = (q * slopes[a] * slopes[b] * slopes[c]).sum(dim=-1)
            norm_third = (
                6.0 * third_moment
                - 9.0 * (second_moment[a, b] * first[c]
                         + second_moment[a, c] * first[b]
                         + second_moment[b, c] * first[a])
                + 21.0 * first[a] * first[b] * first[c]
            ) / safe_radius**2
            value = (
                ramps[3] * first[a] * first[b] * first[c]
                + ramps[2] * (second[a, b] * first[c] + second[a, c] * first[b]
                              + second[b, c] * first[a])
                + ramps[1] * norm_third
            )
        single = single_ramps[degree] * single_slopes[0]**nx * single_slopes[1]**ny
        result[nx, ny] = torch.where(multi, value, single)
    return result


def _obstacle_derivatives(field, coordinates, transition, keys, order, *, legacy_c3):
    obstacle = None
    hard_fluid = coordinates.new_ones(coordinates.shape[0])
    for prepared in field._prepared_polygons:
        distances = _prepared_polygon_outward_distances(coordinates, prepared)
        if field.continuous_mms_abi == "continuous_structured_ns_mms_v6":
            clear = _interior_c3_polygon_jets(distances / transition, prepared, transition, keys, order)
            obstacle = clear if obstacle is None else _multiply_jets(obstacle, clear, keys)
            hard_fluid = hard_fluid * (distances.amax(dim=-1) > 0.0).to(dtype=coordinates.dtype)
            continue
        ramps = _unit_ramp_derivatives(distances / transition, order, legacy_c3=legacy_c3)
        _, edges, lengths, orientation = prepared
        if order:
            dx = orientation * edges[:, 1] / lengths / transition
            dy = -orientation * edges[:, 0] / lengths / transition
            dx_powers, dy_powers = [None, dx], [None, dy]
            for degree in range(2, order + 1):
                dx_powers.append(dx_powers[-1] * dx)
                dy_powers.append(dy_powers[-1] * dy)
        edge_complements = {(0, 0): 1.0 - ramps[0]}
        for nx, ny in keys[1:]:
            value = -ramps[nx + ny]
            if nx:
                value = value * dx_powers[nx]
            if ny:
                value = value * dy_powers[ny]
            edge_complements[(nx, ny)] = value
        # Product of the inside-edge transitions. A product rule stays finite
        # at exact zeros, unlike logarithmic derivatives or division by masks.
        inside = {key: edge_complements[key][:, 0] for key in keys}
        for edge in range(1, edges.shape[0]):
            inside = _multiply_jets(
                inside, {key: edge_complements[key][:, edge] for key in keys}, keys,
            )
        clear = {key: -inside[key] for key in keys}
        clear[(0, 0)] = 1.0 - inside[(0, 0)]
        obstacle = clear if obstacle is None else _multiply_jets(obstacle, clear, keys)
        if not legacy_c3:
            hard_fluid = hard_fluid * (distances.amax(dim=-1) > 0.0).to(dtype=coordinates.dtype)
    if obstacle is None:
        zero = coordinates.new_zeros(coordinates.shape[0])
        obstacle = {key: zero for key in keys}
        obstacle[(0, 0)] = coordinates.new_ones(coordinates.shape[0])
    return obstacle, hard_fluid


def geometry_mask_derivatives(field, normalized_coords, *, order=3):
    """Return ``({(dx,dy): d^(dx+dy)mask}, pressure_or_fluid_mask)``.

    The second value exactly matches ``field.geometry_masks``: MMS v2 returns
    its smooth obstacle-clear mask; v3+ return the hard fluid indicator. Outer
    walls enter that hard indicator only under geometry ABI v2, as before.
    Use ``pressure_mask_derivatives`` when differentiating v2 pressure.
    """
    geometry, transition, keys = _validate(field, normalized_coords, order)
    legacy_c3 = field.continuous_mms_abi == _LEGACY_C3_ABI
    x, y = normalized_coords[:, 0], normalized_coords[:, 1]
    if geometry.outer_boundary == "manufactured_trace":
        outer = {key: normalized_coords.new_zeros(x.shape) for key in keys}
        outer[(0, 0)] = normalized_coords.new_ones(x.shape)
    else:
        x_wall = _wall_derivatives(x, transition, order, legacy_c3=legacy_c3)
        y_wall = _wall_derivatives(y, transition, order, legacy_c3=legacy_c3)
        outer = {(nx, ny): x_wall[nx] * y_wall[ny] for nx, ny in keys}
    obstacle, fluid = _obstacle_derivatives(
        field, normalized_coords, transition, keys, order, legacy_c3=legacy_c3,
    )
    result = _multiply_jets(outer, obstacle, keys) if field._prepared_polygons else outer
    if legacy_c3:
        fluid = obstacle[(0, 0)]
    elif geometry.abi_version == CONTINUOUS_GEOMETRY_ABI_VERSION:
        outer_fluid = (((x >= 0.0) & (x <= 1.0) & (y >= 0.0) & (y <= 1.0))
                       if geometry.outer_boundary == "manufactured_trace" else
                       ((x > 0.0) & (x < 1.0) & (y > 0.0) & (y < 1.0)))
        fluid = fluid * outer_fluid.to(dtype=normalized_coords.dtype)
    return result, fluid


def pressure_mask_derivatives(field, normalized_coords, *, order=1):
    """Return v2's smooth pressure-mask jets, or constant hard-mask jets.

    This helper is for pressure ABIs v2/v3. MMS v4/v5 use unmasked pressure and
    their callers should evaluate the pressure basis directly.
    """
    geometry, transition, keys = _validate(field, normalized_coords, order)
    legacy_c3 = field.continuous_mms_abi == _LEGACY_C3_ABI
    obstacle, fluid = _obstacle_derivatives(
        field, normalized_coords, transition, keys, order, legacy_c3=legacy_c3,
    )
    if legacy_c3:
        return obstacle
    if geometry.abi_version == CONTINUOUS_GEOMETRY_ABI_VERSION:
        x, y = normalized_coords[:, 0], normalized_coords[:, 1]
        outer_fluid = (((x >= 0.0) & (x <= 1.0) & (y >= 0.0) & (y <= 1.0))
                       if geometry.outer_boundary == "manufactured_trace" else
                       ((x > 0.0) & (x < 1.0) & (y > 0.0) & (y < 1.0)))
        fluid = fluid * outer_fluid.to(dtype=normalized_coords.dtype)
    zero = normalized_coords.new_zeros(normalized_coords.shape[0])
    return {key: fluid if key == (0, 0) else zero for key in keys}


__all__ = ["geometry_mask_derivatives", "pressure_mask_derivatives"]
