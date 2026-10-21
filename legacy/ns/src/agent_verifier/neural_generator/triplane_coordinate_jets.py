"""Exact sparse coordinate derivatives for the continuous triplane NS decoder."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


COORDINATE_JET_BACKEND_ID = "analytic_sparse_coordinate_jets_v1"
DEFAULT_COORDINATE_JET_BATCH_POINTS = 8192


NS_ORDERS = (
    (0, 0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (2, 0, 0),
    (1, 1, 0),
    (0, 2, 0),
    (1, 0, 1),
    (0, 1, 1),
    (3, 0, 0),
    (2, 1, 0),
    (1, 2, 0),
    (0, 3, 0),
)
PRESSURE_ORDERS = ((0, 0, 0), (1, 0, 0), (0, 1, 0))


def _subtract(left, right):
    if left is None:
        return None
    result = tuple(a - b for a, b in zip(left, right))
    return result if all(value >= 0 for value in result) else None


@dataclass(frozen=True)
class JetSpec:
    orders: tuple[tuple[int, int, int], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "index", {order: index for index, order in enumerate(self.orders)})
        product_terms: list[tuple[tuple[int, int], ...]] = []
        nonzero_pairs: list[tuple[tuple[int, int], ...]] = []
        nonzero_triples: list[tuple[tuple[int, int, int], ...]] = []
        zero = (0, 0, 0)
        for alpha in self.orders:
            pairs = []
            second = []
            third = []
            for beta in self.orders:
                gamma = _subtract(alpha, beta)
                if gamma in self.index:
                    pairs.append((self.index[beta], self.index[gamma]))
                    if beta != zero and gamma != zero:
                        second.append((self.index[beta], self.index[gamma]))
            for beta in self.orders:
                if beta == zero:
                    continue
                remainder = _subtract(alpha, beta)
                if remainder is None:
                    continue
                for gamma in self.orders:
                    if gamma == zero:
                        continue
                    delta = _subtract(remainder, gamma)
                    if delta in self.index and delta != zero:
                        third.append((self.index[beta], self.index[gamma], self.index[delta]))
            product_terms.append(tuple(pairs))
            nonzero_pairs.append(tuple(second))
            nonzero_triples.append(tuple(third))
        object.__setattr__(self, "product_terms", tuple(product_terms))
        object.__setattr__(self, "nonzero_pairs", tuple(nonzero_pairs))
        object.__setattr__(self, "nonzero_triples", tuple(nonzero_triples))
        object.__setattr__(
            self,
            "factorials",
            tuple(math.prod(math.factorial(value) for value in order) for order in self.orders),
        )


NS_SPEC = JetSpec(NS_ORDERS)
PRESSURE_SPEC = JetSpec(PRESSURE_ORDERS)


class TriplaneNSCoordinateJetBackend:
    """Propagate only the coordinate Taylor coefficients needed by 2D NS."""

    def __init__(self, continuous) -> None:
        self.continuous = continuous
        self.torch = continuous.torch
        self.decoder = continuous.decoder
        if self.decoder.config.plane_interpolation != "quintic_bspline":
            raise ValueError("coordinate-jet backend requires quintic B-spline planes")
        if self.decoder.config.continuous_mms_abi != "continuous_triplane_mms_v1":
            raise ValueError("coordinate-jet backend requires the continuous MMS decoder")
        if self.decoder.config.field_stack_decode:
            raise ValueError("coordinate-jet backend does not support field-stack decoders")
        if not self.decoder.config.separate_output_heads:
            raise ValueError("coordinate-jet backend requires separate psi and pressure heads")
        if int(self.decoder.config.geometry_feature_dim) != 2:
            raise ValueError("coordinate-jet backend requires the two-channel canonical geometry ABI")
        if self.decoder.config.activation not in {"silu", "tanh"}:
            raise ValueError(f"coordinate-jet backend does not support {self.decoder.config.activation!r}")
        if str(self.continuous.dtype) != "torch.float64":
            raise ValueError("coordinate-jet backend requires float64 decoder evaluation")

    def materialize_cartesian(
        self,
        *,
        x,
        y,
        t,
        batch_points: int = DEFAULT_COORDINATE_JET_BATCH_POINTS,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        torch = self.torch
        x_values = torch.as_tensor(x, dtype=self.continuous.dtype, device=self.continuous.device)
        y_values = torch.as_tensor(y, dtype=self.continuous.dtype, device=self.continuous.device)
        t_values = torch.as_tensor(t, dtype=self.continuous.dtype, device=self.continuous.device)
        x_min, x_max, y_min, y_max = self.continuous.domain_bounds
        _validate_axis(torch, x_values, name="x", lower=x_min, upper=x_max)
        _validate_axis(torch, y_values, name="y", lower=y_min, upper=y_max)
        _validate_axis(torch, t_values, name="t", lower=self.continuous.t0, upper=self.continuous.t1)
        total = int(x_values.numel() * y_values.numel() * t_values.numel())
        names = (
            "u",
            "v",
            "d_x_u",
            "d_y_v",
            "pressure",
            "psi",
            "forcing_u",
            "forcing_v",
            "divergence",
            "residual_u",
            "residual_v",
        )
        parts: dict[str, list[Any]] = {name: [] for name in names}
        chunk = max(1, int(batch_points))
        with torch.no_grad():
            tables = self._build_cartesian_tables(x_values, y_values, t_values)
            for start in range(0, total, chunk):
                flat = torch.arange(start, min(start + chunk, total), device=self.continuous.device)
                quantities = self._chunk_quantities(
                    flat,
                    nx=int(x_values.numel()),
                    ny=int(y_values.numel()),
                    nt=int(t_values.numel()),
                    tables=tables,
                )
                for name in names:
                    parts[name].append(quantities[name])
                del quantities
            shape = (int(t_values.numel()), int(x_values.numel()), int(y_values.numel()))
            arrays = {
                name: torch.cat(values).reshape(shape).cpu().numpy()
                for name, values in parts.items()
            }
        diagnostics = {
            "backend_id": COORDINATE_JET_BACKEND_ID,
            "batch_points": int(chunk),
            "jet_orders": [list(order) for order in NS_SPEC.orders],
            "unique_xy_points": int(x_values.numel() * y_values.numel()),
            "unique_xt_points": int(x_values.numel() * t_values.numel()),
            "unique_yt_points": int(y_values.numel() * t_values.numel()),
            "materialized_points": int(total),
        }
        return arrays, diagnostics

    def _build_cartesian_tables(self, x, y, t) -> dict[str, Any]:
        torch = self.torch
        x_min, x_max, y_min, y_max = self.continuous.domain_bounds
        x_scale = 1.0 / (x_max - x_min)
        y_scale = 1.0 / (y_max - y_min)
        t_scale = 1.0 / (self.continuous.t1 - self.continuous.t0)
        xn = (x - x_min) * x_scale
        yn = (y - y_min) * y_scale
        tn = (t - self.continuous.t0) * t_scale
        xy_x, xy_y = torch.meshgrid(xn, yn, indexing="ij")
        xt_x, xt_t = torch.meshgrid(xn, tn, indexing="ij")
        yt_y, yt_t = torch.meshgrid(yn, tn, indexing="ij")
        p_xy, p_xt, p_yt = self.continuous.planes.as_tuple()
        xy_orders = sorted({(order[0], order[1]) for order in NS_SPEC.orders if order[2] == 0})
        xt_orders = sorted({(order[0], order[2]) for order in NS_SPEC.orders if order[1] == 0})
        yt_orders = sorted({(order[1], order[2]) for order in NS_SPEC.orders if order[0] == 0})
        geometry_orders = xy_orders
        return {
            "x": xn,
            "y": yn,
            "t": tn,
            "scales": (float(x_scale), float(y_scale), float(t_scale)),
            "xy": _quintic_plane_derivatives(
                torch,
                p_xy,
                xy_x.reshape(-1),
                xy_y.reshape(-1),
                first_scale=x_scale,
                second_scale=y_scale,
                orders=xy_orders,
            ),
            "xt": _quintic_plane_derivatives(
                torch,
                p_xt,
                xt_x.reshape(-1),
                xt_t.reshape(-1),
                first_scale=x_scale,
                second_scale=t_scale,
                orders=xt_orders,
            ),
            "yt": _quintic_plane_derivatives(
                torch,
                p_yt,
                yt_y.reshape(-1),
                yt_t.reshape(-1),
                first_scale=y_scale,
                second_scale=t_scale,
                orders=yt_orders,
            ),
            "geometry": _c3_plane_derivatives(
                torch,
                self.continuous.canonical_geometry[None, ...],
                xy_x.reshape(-1),
                xy_y.reshape(-1),
                first_scale=x_scale,
                second_scale=y_scale,
                orders=geometry_orders,
            ),
        }

    def _chunk_quantities(self, flat, *, nx: int, ny: int, nt: int, tables: dict[str, Any]):
        del nt
        iy = flat.remainder(ny)
        ix = flat.div(ny, rounding_mode="floor").remainder(nx)
        it = flat.div(nx * ny, rounding_mode="floor")
        xy_index = ix * ny + iy
        xt_index = ix * int(tables["t"].numel()) + it
        yt_index = iy * int(tables["t"].numel()) + it
        input_jets, geometry_jets, coordinate_jets = self._input_jets(
            ix,
            iy,
            it,
            xy_index,
            xt_index,
            yt_index,
            tables,
        )
        shared = _run_layers(self.torch, input_jets, self.decoder.layers, self.decoder.config.activation, NS_SPEC)
        psi_raw = _run_head(
            self.torch,
            shared,
            self.decoder.psi_head_layers,
            self.decoder.psi_output,
            self.decoder.config.activation,
            NS_SPEC,
        )
        pressure_shared = shared[[NS_SPEC.index[order] for order in PRESSURE_SPEC.orders]]
        pressure_raw = _run_head(
            self.torch,
            pressure_shared,
            self.decoder.pressure_head_layers,
            self.decoder.pressure_output,
            self.decoder.config.activation,
            PRESSURE_SPEC,
        )
        psi_raw = psi_raw * float(self.decoder.config.streamfunction_output_scale)
        pressure_raw = pressure_raw * float(self.decoder.config.pressure_output_scale)
        obstacle_mask, geometry_mask = _geometry_masks(
            self.torch,
            coordinate_jets,
            geometry_jets,
            transition=float(self.decoder.config.geometry_mask_transition),
        )
        psi = _jet_multiply(self.torch, geometry_mask, psi_raw, NS_SPEC)
        pressure = _jet_multiply(
            self.torch,
            obstacle_mask[[NS_SPEC.index[order] for order in PRESSURE_SPEC.orders]],
            pressure_raw,
            PRESSURE_SPEC,
        )
        return _assemble_navier_stokes(psi, pressure, viscosity=self.continuous.viscosity)

    def _input_jets(self, ix, iy, it, xy_index, xt_index, yt_index, tables):
        torch = self.torch
        count = int(ix.numel())
        channels = int(self.decoder.config.plane_channels)
        zero_plane = tables["xy"][(0, 0)].new_zeros((count, channels))
        zero_geometry = tables["geometry"][(0, 0)].new_zeros((count, 2))
        coordinate_jets = _coordinate_variable_jets(
            torch,
            tables["x"][ix],
            tables["y"][iy],
            tables["t"][it],
            scales=tables["scales"],
            spec=NS_SPEC,
        )
        rows = []
        geometry_rows = []
        for alpha in NS_SPEC.orders:
            ax, ay, at = alpha
            xy = tables["xy"].get((ax, ay))
            xt = tables["xt"].get((ax, at))
            yt = tables["yt"].get((ay, at))
            xy_value = xy[xy_index] / _multi_factorial(alpha) if at == 0 and xy is not None else zero_plane
            xt_value = xt[xt_index] / _multi_factorial(alpha) if ay == 0 and xt is not None else zero_plane
            yt_value = yt[yt_index] / _multi_factorial(alpha) if ax == 0 and yt is not None else zero_plane
            geometry = tables["geometry"].get((ax, ay))
            geometry_value = (
                geometry[xy_index] / _multi_factorial(alpha)
                if at == 0 and geometry is not None
                else zero_geometry
            )
            encoding = _coordinate_encoding_derivative(
                torch,
                coordinate_jets[NS_SPEC.index[alpha]],
                tables["x"][ix],
                tables["y"][iy],
                tables["t"][it],
                alpha=alpha,
                scales=tables["scales"],
                frequencies=int(self.decoder.config.coordinate_frequencies),
            )
            rows.append(torch.cat([xy_value, xt_value, yt_value, encoding, geometry_value], dim=-1))
            geometry_rows.append(geometry_value)
        return torch.stack(rows), torch.stack(geometry_rows), coordinate_jets


def _validate_axis(torch, values, *, name: str, lower: float, upper: float) -> None:
    if values.ndim != 1 or int(values.numel()) == 0:
        raise ValueError(f"coordinate-jet {name} axis must be a nonempty one-dimensional tensor")
    if not bool(torch.all(torch.isfinite(values)).item()):
        raise ValueError(f"coordinate-jet {name} axis contains NaN or Inf")
    tolerance = 1.0e-12 * max(1.0, abs(float(lower)), abs(float(upper)))
    if float(torch.min(values).item()) < float(lower) - tolerance or float(torch.max(values).item()) > float(upper) + tolerance:
        raise ValueError(
            f"coordinate-jet {name} axis lies outside [{float(lower)}, {float(upper)}]"
        )


def _run_layers(torch, jets, layers, activation: str, spec: JetSpec):
    for layer in layers:
        jets = _jet_activation(torch, _jet_linear(torch, jets, layer), activation, spec)
    return jets


def _run_head(torch, jets, layers, output, activation: str, spec: JetSpec):
    jets = _run_layers(torch, jets, layers, activation, spec)
    return _jet_linear(torch, jets, output)


def _jet_linear(torch, jets, layer):
    count = int(jets.shape[1])
    out = torch.matmul(jets.reshape(-1, jets.shape[-1]), layer.weight.transpose(0, 1))
    out = out.reshape(jets.shape[0], count, layer.weight.shape[0])
    if layer.bias is not None:
        out[0].add_(layer.bias)
    return out


def _jet_activation(torch, jets, activation: str, spec: JetSpec):
    value = jets[0]
    if activation == "silu":
        sigmoid = torch.sigmoid(value)
        a = sigmoid * (1.0 - sigmoid)
        b = 1.0 - 2.0 * sigmoid
        primal = value * sigmoid
        first = sigmoid + value * a
        second = 2.0 * a + value * a * b
        third = 3.0 * a * b + value * a * (b.square() - 2.0 * a)
    elif activation == "tanh":
        primal = torch.tanh(value)
        first = 1.0 - primal.square()
        second = -2.0 * primal * first
        third = -2.0 * first * (1.0 - 3.0 * primal.square())
    else:
        raise ValueError(f"unsupported jet activation: {activation}")
    return _compose_from_derivatives(torch, jets, primal, first, second, third, spec)


def _compose_from_derivatives(torch, jets, primal, first, second, third, spec: JetSpec):
    rows = [primal]
    for index, alpha in enumerate(spec.orders[1:], start=1):
        result = first * jets[index]
        if sum(alpha) >= 2:
            second_sum = torch.zeros_like(result)
            for left, right in spec.nonzero_pairs[index]:
                second_sum.add_(jets[left] * jets[right])
            result = result + 0.5 * second * second_sum
        if sum(alpha) >= 3:
            third_sum = torch.zeros_like(result)
            for first_index, second_index, third_index in spec.nonzero_triples[index]:
                third_sum.add_(jets[first_index] * jets[second_index] * jets[third_index])
            result = result + (1.0 / 6.0) * third * third_sum
        rows.append(result)
    return torch.stack(rows)


def _jet_multiply(torch, left, right, spec: JetSpec):
    rows = []
    for terms in spec.product_terms:
        value = None
        for left_index, right_index in terms:
            term = left[left_index] * right[right_index]
            value = term if value is None else value + term
        if value is None:
            value = left[0].new_zeros(left[0].shape)
        rows.append(value)
    return torch.stack(rows)


def _geometry_masks(torch, coordinate_jets, geometry_jets, *, transition: float):
    obstacle = _c3_ramp_jet(torch, geometry_jets[..., 0:1] / transition, NS_SPEC)
    wall_distance = _c3_ramp_jet(torch, geometry_jets[..., 1:2] / transition, NS_SPEC)
    x = coordinate_jets[..., 0:1]
    y = coordinate_jets[..., 1:2]
    one_minus_x = -x
    one_minus_x[0].add_(1.0)
    one_minus_y = -y
    one_minus_y[0].add_(1.0)
    factors = (
        _c3_ramp_jet(torch, x / transition, NS_SPEC),
        _c3_ramp_jet(torch, one_minus_x / transition, NS_SPEC),
        _c3_ramp_jet(torch, y / transition, NS_SPEC),
        _c3_ramp_jet(torch, one_minus_y / transition, NS_SPEC),
    )
    rectangular = factors[0]
    for factor in factors[1:]:
        rectangular = _jet_multiply(torch, rectangular, factor, NS_SPEC)
    return obstacle, _jet_multiply(
        torch,
        _jet_multiply(torch, obstacle, wall_distance, NS_SPEC),
        rectangular,
        NS_SPEC,
    )


def _c3_ramp_jet(torch, jets, spec: JetSpec):
    z = jets[0]
    s = z.clamp(0.0, 1.0)
    active = ((z > 0.0) & (z < 1.0)).to(dtype=z.dtype)
    primal = s**4 * (35.0 - 84.0 * s + 70.0 * s**2 - 20.0 * s**3)
    first = active * (140.0 * s**3 - 420.0 * s**4 + 420.0 * s**5 - 140.0 * s**6)
    second = active * (420.0 * s**2 - 1680.0 * s**3 + 2100.0 * s**4 - 840.0 * s**5)
    third = active * (840.0 * s - 5040.0 * s**2 + 8400.0 * s**3 - 4200.0 * s**4)
    return _compose_from_derivatives(torch, jets, primal, first, second, third, spec)


def _assemble_navier_stokes(psi, pressure, *, viscosity: float):
    raw_psi = {
        order: psi[NS_SPEC.index[order]] * NS_SPEC.factorials[NS_SPEC.index[order]]
        for order in NS_SPEC.orders
    }
    raw_p = {
        order: pressure[PRESSURE_SPEC.index[order]] * PRESSURE_SPEC.factorials[PRESSURE_SPEC.index[order]]
        for order in PRESSURE_SPEC.orders
    }
    u = raw_psi[(0, 1, 0)]
    v = -raw_psi[(1, 0, 0)]
    d_x_u = raw_psi[(1, 1, 0)]
    d_y_u = raw_psi[(0, 2, 0)]
    d_t_u = raw_psi[(0, 1, 1)]
    d_x_v = -raw_psi[(2, 0, 0)]
    d_y_v = -raw_psi[(1, 1, 0)]
    d_t_v = -raw_psi[(1, 0, 1)]
    laplacian_u = raw_psi[(2, 1, 0)] + raw_psi[(0, 3, 0)]
    laplacian_v = -(raw_psi[(3, 0, 0)] + raw_psi[(1, 2, 0)])
    advection_u = u * d_x_u + v * d_y_u
    advection_v = u * d_x_v + v * d_y_v
    forcing_u = d_t_u + advection_u + raw_p[(1, 0, 0)] - float(viscosity) * laplacian_u
    forcing_v = d_t_v + advection_v + raw_p[(0, 1, 0)] - float(viscosity) * laplacian_v
    divergence = d_x_u + d_y_v
    return {
        "psi": raw_psi[(0, 0, 0)],
        "pressure": raw_p[(0, 0, 0)],
        "u": u,
        "v": v,
        "d_x_u": d_x_u,
        "d_y_u": d_y_u,
        "d_t_u": d_t_u,
        "d_x_v": d_x_v,
        "d_y_v": d_y_v,
        "d_t_v": d_t_v,
        "laplacian_u": laplacian_u,
        "laplacian_v": laplacian_v,
        "d_x_p": raw_p[(1, 0, 0)],
        "d_y_p": raw_p[(0, 1, 0)],
        "advection_u": advection_u,
        "advection_v": advection_v,
        "forcing_u": forcing_u,
        "forcing_v": forcing_v,
        "divergence": divergence,
        "residual_u": forcing_u - forcing_u,
        "residual_v": forcing_v - forcing_v,
    }


def _coordinate_variable_jets(torch, x, y, t, *, scales, spec: JetSpec):
    rows = []
    values = torch.stack([x, y, t], dim=-1)
    for alpha in spec.orders:
        if alpha == (0, 0, 0):
            rows.append(values)
            continue
        row = torch.zeros_like(values)
        if alpha == (1, 0, 0):
            row[:, 0] = float(scales[0])
        elif alpha == (0, 1, 0):
            row[:, 1] = float(scales[1])
        elif alpha == (0, 0, 1):
            row[:, 2] = float(scales[2])
        rows.append(row)
    return torch.stack(rows)


def _coordinate_encoding_derivative(
    torch,
    coordinate_jet,
    x,
    y,
    t,
    *,
    alpha,
    scales,
    frequencies: int,
):
    coords = torch.stack([x, y, t], dim=-1)
    order = sum(alpha)
    if order == 0:
        pieces = [coords]
        for frequency in range(frequencies):
            wave = float(2**frequency) * np.pi
            pieces.extend([torch.sin(wave * coords), torch.cos(wave * coords)])
        return torch.cat(pieces, dim=-1)
    pieces = [coordinate_jet]
    active_axes = [axis for axis, value in enumerate(alpha) if value]
    axis = active_axes[0] if len(active_axes) == 1 else -1
    for frequency in range(frequencies):
        wave = float(2**frequency) * np.pi
        sin_value = torch.zeros_like(coords)
        cos_value = torch.zeros_like(coords)
        if axis >= 0:
            physical_wave = wave * float(scales[axis])
            sin_value[:, axis] = physical_wave**order * torch.sin(wave * coords[:, axis] + order * np.pi / 2.0)
            cos_value[:, axis] = physical_wave**order * torch.cos(wave * coords[:, axis] + order * np.pi / 2.0)
        pieces.extend([sin_value / math.factorial(order), cos_value / math.factorial(order)])
    return torch.cat(pieces, dim=-1)


def _quintic_plane_derivatives(
    torch,
    plane,
    first,
    second,
    *,
    first_scale: float,
    second_scale: float,
    orders,
):
    plane = plane if plane.ndim == 4 else plane[None, ...]
    if int(plane.shape[0]) != 1:
        raise ValueError("coordinate-jet backend supports one latent case at a time")
    _, channels, height, width = plane.shape
    first_pos = first * float(height) - 0.5
    second_pos = second * float(width) - 0.5
    first_base = torch.floor(first_pos).to(dtype=torch.long)
    second_base = torch.floor(second_pos).to(dtype=torch.long)
    max_first = max(order[0] for order in orders)
    max_second = max(order[1] for order in orders)
    first_terms = []
    second_terms = []
    for offset in range(-2, 4):
        first_index = first_base + offset
        second_index = second_base + offset
        first_terms.append(
            (
                first_index.clamp(0, max(height - 1, 0)),
                _quintic_weight_derivatives(
                    torch,
                    first_pos - first_index.to(first_pos.dtype),
                    max_order=max_first,
                    coordinate_scale=float(height) * float(first_scale),
                ),
            )
        )
        second_terms.append(
            (
                second_index.clamp(0, max(width - 1, 0)),
                _quintic_weight_derivatives(
                    torch,
                    second_pos - second_index.to(second_pos.dtype),
                    max_order=max_second,
                    coordinate_scale=float(width) * float(second_scale),
                ),
            )
        )
    flat = plane[0].reshape(channels, height * width)
    gathered = {}
    for first_offset, (first_index, _) in enumerate(first_terms):
        for second_offset, (second_index, _) in enumerate(second_terms):
            linear = (first_index * width + second_index).clamp(0, max(height * width - 1, 0))
            gathered[(first_offset, second_offset)] = flat.index_select(1, linear).transpose(0, 1)
    outputs = {}
    for first_order, second_order in orders:
        value = plane.new_zeros((first.numel(), channels))
        for first_offset, (_, first_weights) in enumerate(first_terms):
            for second_offset, (_, second_weights) in enumerate(second_terms):
                value.add_(
                    gathered[(first_offset, second_offset)]
                    * (first_weights[first_order] * second_weights[second_order])[:, None]
                )
        outputs[(first_order, second_order)] = value
    return outputs


def _quintic_weight_derivatives(torch, distance, *, max_order: int, coordinate_scale: float):
    work = distance.to(torch.float64)
    coefficients = (1.0, -6.0, 15.0, -20.0, 15.0, -6.0, 1.0)
    outputs = []
    for order in range(max_order + 1):
        power = 5 - order
        prefactor = math.factorial(5) / math.factorial(power) / 120.0
        value = torch.zeros_like(work)
        for offset, coefficient in enumerate(coefficients):
            value.add_(float(coefficient) * torch.relu(work + 3.0 - float(offset)) ** power)
        outputs.append((prefactor * float(coordinate_scale) ** order * value).to(distance.dtype))
    return tuple(outputs)


def _c3_plane_derivatives(
    torch,
    plane,
    first,
    second,
    *,
    first_scale: float,
    second_scale: float,
    orders,
):
    plane = plane if plane.ndim == 4 else plane[None, ...]
    _, channels, height, width = plane.shape
    first_raw = first * float(height) - 0.5
    second_raw = second * float(width) - 0.5
    first_pos = first_raw.clamp(0.0, float(max(height - 1, 0)))
    second_pos = second_raw.clamp(0.0, float(max(width - 1, 0)))
    first0 = torch.floor(first_pos).to(torch.long).clamp(0, max(height - 1, 0))
    second0 = torch.floor(second_pos).to(torch.long).clamp(0, max(width - 1, 0))
    first1 = (first0 + 1).clamp(max=max(height - 1, 0))
    second1 = (second0 + 1).clamp(max=max(width - 1, 0))
    max_first = max(order[0] for order in orders)
    max_second = max(order[1] for order in orders)
    first_weights = _c3_weight_derivatives(
        first_pos - first0.to(first_pos.dtype),
        max_order=max_first,
        coordinate_scale=float(height) * float(first_scale),
        active=(first_raw > 0.0) & (first_raw < float(max(height - 1, 0))),
    )
    second_weights = _c3_weight_derivatives(
        second_pos - second0.to(second_pos.dtype),
        max_order=max_second,
        coordinate_scale=float(width) * float(second_scale),
        active=(second_raw > 0.0) & (second_raw < float(max(width - 1, 0))),
    )
    flat = plane[0].reshape(channels, height * width)
    corners = {
        (0, 0): flat.index_select(1, first0 * width + second0).transpose(0, 1),
        (1, 0): flat.index_select(1, first1 * width + second0).transpose(0, 1),
        (0, 1): flat.index_select(1, first0 * width + second1).transpose(0, 1),
        (1, 1): flat.index_select(1, first1 * width + second1).transpose(0, 1),
    }
    outputs = {}
    for first_order, second_order in orders:
        value = plane.new_zeros((first.numel(), channels))
        for first_corner in (0, 1):
            wf = _corner_weight(first_weights, first_corner, first_order)
            for second_corner in (0, 1):
                ws = _corner_weight(second_weights, second_corner, second_order)
                value.add_(corners[(first_corner, second_corner)] * (wf * ws)[:, None])
        outputs[(first_order, second_order)] = value
    return outputs


def _c3_weight_derivatives(value, *, max_order: int, coordinate_scale: float, active):
    s = value.clamp(0.0, 1.0)
    active_value = active.to(dtype=s.dtype)
    base = s**4 * (35.0 - 84.0 * s + 70.0 * s**2 - 20.0 * s**3)
    derivatives = [base]
    if max_order >= 1:
        derivatives.append(
            active_value
            * float(coordinate_scale)
            * (140.0 * s**3 - 420.0 * s**4 + 420.0 * s**5 - 140.0 * s**6)
        )
    if max_order >= 2:
        derivatives.append(
            active_value
            * float(coordinate_scale) ** 2
            * (420.0 * s**2 - 1680.0 * s**3 + 2100.0 * s**4 - 840.0 * s**5)
        )
    if max_order >= 3:
        derivatives.append(
            active_value
            * float(coordinate_scale) ** 3
            * (840.0 * s - 5040.0 * s**2 + 8400.0 * s**3 - 4200.0 * s**4)
        )
    return tuple(derivatives)


def _corner_weight(weights, corner: int, order: int):
    if order == 0:
        return weights[0] if corner else 1.0 - weights[0]
    return weights[order] if corner else -weights[order]


def _multi_factorial(alpha) -> int:
    return math.prod(math.factorial(value) for value in alpha)
