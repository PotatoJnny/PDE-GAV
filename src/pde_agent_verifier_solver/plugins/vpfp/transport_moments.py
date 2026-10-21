"""FP64 full-space moments for the opt-in plasma transport score.

The legacy macro and manufactured-source paths deliberately do not import or
call these third-moment operations. All four symmetric third moments are kept
until after an affine pushforward; heat flux alone does not transform as a
vector under a general anisotropic change of velocity coordinates.
"""
from __future__ import annotations

from functools import lru_cache
import math

import numpy as np
import torch


TRANSPORT_MACRO_KEYS = (
    "rho", "momentum", "kinetic_energy", "raw_second_moment",
    "electric", "pressure", "heat_flux",
)


@lru_cache(maxsize=32)
def _recurrence_layout(modes, device):
    degree = max(sum(pair) for pair in modes) + 2
    extended = tuple((first, total-first) for total in range(degree+1)
                     for first in range(total+1))
    index = {pair: row for row, pair in enumerate(extended)}
    original = torch.tensor([index[pair] for pair in modes], device=device)
    operators = []
    for axis in (0, 1):
        source, destination, weight = [], [], []
        for row, pair in enumerate(extended):
            for shift, factor in ((1, math.sqrt(pair[axis]+1)),
                                  (-1, math.sqrt(pair[axis]))):
                neighbor = list(pair)
                neighbor[axis] += shift
                if tuple(neighbor) in index:
                    source.append(row)
                    destination.append(index[tuple(neighbor)])
                    weight.append(factor)
        operators.append((torch.tensor(source, device=device),
                          torch.tensor(destination, device=device),
                          torch.tensor(weight, device=device, dtype=torch.float64)))
    return len(extended), original, tuple(operators)


def squared_hermite_transport_moments(coefficients, *, modes, temperature,
                                     floor_mixture=0.0):
    """Normalized mean, raw second and full raw/central third moments.

    Sparse velocity recurrence is embedded through degree N+2. Thus the inner
    products <v_i c, v_j v_k c> include paths exiting and re-entering the
    original basis. Cubing a truncated multiplication matrix is incorrect.
    Packed second/third order is (xx,xy,yy)/(xxx,xxy,xyy,yyy).
    """
    c = torch.as_tensor(coefficients).double()
    pairs = tuple((int(a), int(b)) for a, b in modes)
    if (not pairs or len(set(pairs)) != len(pairs)
            or any(min(pair) < 0 for pair in pairs)
            or c.ndim < 1 or c.shape[-1] != len(pairs)):
        raise ValueError("transport Hermite coefficients/modes are malformed")
    shape = c.shape[:-1]
    flat = c.reshape(-1, len(pairs))
    theta = torch.broadcast_to(torch.as_tensor(temperature, dtype=torch.float64,
                                               device=c.device), shape).reshape(-1)
    floor = float(floor_mixture)
    if not math.isfinite(floor) or not 0 <= floor < 1:
        raise ValueError("floor_mixture must lie in [0,1)")
    norm = flat.square().sum(-1)
    if (not bool(torch.isfinite(flat).all()) or not bool(torch.isfinite(theta).all())
            or bool((theta <= 0).any()) or bool((norm <= 0).any())):
        raise ValueError("transport moments require finite coefficients, positive norm and temperature")
    width, original, operators = _recurrence_layout(pairs, str(c.device))
    embedded = flat.new_zeros((len(flat), width)).index_add(1, original, flat)

    def multiply(value, axis):
        source, destination, weight = operators[axis]
        return torch.zeros_like(value).index_add(
            1, destination,
            value.index_select(1, source)*weight[None]*theta.sqrt()[:, None])

    vx, vy = multiply(embedded, 0), multiply(embedded, 1)
    vxx, vyy = multiply(vx, 0), multiply(vy, 1)
    scale = (1-floor)/norm
    mean = torch.stack(((embedded*vx).sum(-1), (embedded*vy).sum(-1)), -1)*scale[:, None]
    second = torch.stack((vx.square().sum(-1), (vx*vy).sum(-1), vy.square().sum(-1)), -1)*scale[:, None]
    second = second + floor*theta[:, None]*second.new_tensor([1., 0., 1.])
    third = torch.stack(((vx*vxx).sum(-1), (vy*vxx).sum(-1),
                         (vx*vyy).sum(-1), (vy*vyy).sum(-1)), -1)*scale[:, None]
    ux, uy = mean.unbind(-1)
    xx, xy, yy = second.unbind(-1)
    central = third - torch.stack((3*ux*xx, uy*xx+2*ux*xy,
                                   ux*yy+2*uy*xy, 3*uy*yy), -1)
    central = central + 2*torch.stack((ux**3, ux.square()*uy,
                                       ux*uy.square(), uy**3), -1)
    covariance = second-torch.stack((ux.square(), ux*uy, uy.square()), -1)
    result = {"mean_velocity": mean.reshape(*shape, 2),
              "raw_second_moment": second.reshape(*shape, 3),
              "central_covariance": covariance.reshape(*shape, 3),
              "raw_third_moment": third.reshape(*shape, 4),
              "central_third_moment": central.reshape(*shape, 4)}
    if not all(bool(torch.isfinite(value).all()) for value in result.values()):
        raise FloatingPointError("nonfinite full-space transport moments")
    return result


def affine_central_third(packed, affine):
    """Push forward the complete symmetric third tensor by A tensor A tensor A."""
    value, a = packed.double(), affine.double()
    # Expanding only a 2x2x2 tensor is cheaper and clearer than a velocity grid.
    indices = torch.tensor([0, 1, 1, 2, 1, 2, 2, 3], device=value.device)
    tensor = value.index_select(-1, indices).reshape(*value.shape[:-1], 2, 2, 2)
    transformed = torch.einsum("...ia,...jb,...kc,...abc->...ijk", a, a, a, tensor)
    return torch.stack((transformed[..., 0, 0, 0], transformed[..., 0, 0, 1],
                        transformed[..., 0, 1, 1], transformed[..., 1, 1, 1]), -1)


def heat_flux_from_central_third(rho, packed):
    return .5*rho[..., None]*torch.stack((packed[..., 0]+packed[..., 2],
                                         packed[..., 1]+packed[..., 3]), -1)


def evaluate_hermite_transport_points(model, latent, points_txy, *,
                                      equation_condition, chunk_size=8192):
    """One decoder batch per block, with all v2 truth fields from that batch."""
    points = np.asarray(points_txy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError("transport query must be finite nonempty [N,3]")
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("transport chunk_size must be positive")
    rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(points), chunk_size):
            txy = points[start:start+chunk_size]
            fields = model.fields(np.column_stack((txy, np.zeros((len(txy), 2)))),
                                  latent, equation_condition=equation_condition)
            moments = squared_hermite_transport_moments(
                fields["hermite_coefficients"], modes=model.hermite_modes,
                temperature=fields["equation_condition"][:, 1],
                floor_mixture=model.config.velocity_floor_mixture)
            rho = fields["rho"].double()
            second = rho[:, None]*moments["raw_second_moment"]
            row = {"rho": rho, "momentum": rho[:, None]*moments["mean_velocity"],
                   "kinetic_energy": .5*(second[:, 0]+second[:, 2]),
                   "raw_second_moment": second, "electric": fields["electric"].double(),
                   "pressure": rho[:, None]*moments["central_covariance"],
                   "heat_flux": heat_flux_from_central_third(rho, moments["central_third_moment"])}
            rows.append({name: value.cpu().numpy() for name, value in row.items()})
    return {name: np.concatenate([row[name] for row in rows]) for name in TRANSPORT_MACRO_KEYS}
