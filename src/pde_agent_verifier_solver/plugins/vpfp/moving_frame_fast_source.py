"""Opt-in numerical A source: original first derivatives, analytic velocity Hessian.

No source law, carrier, floor, precision or query point is changed. The original
autograd evaluator remains the independent implementation comparison and the
create_graph=True path. This helper has no persistent/cross-answer cache.
"""
from __future__ import annotations

import importlib
import numpy as np  # Preserve frozen NumPy-before-Torch physics initialization.
import torch

from .moving_frame import MovingFrameDecoder, cholesky_from_log, log_cholesky
from .moving_frame_backend import _query, manufacture_moving_frame_source
from .perturbation_backend import _legacy_api


@torch.no_grad()
def _velocity_laplacian(decoder, query, fields, carrier):
    """rho det(A) tr(A^T Hessian_w(g) A), w=mu_z+A(v-u).

The complete squared-Hermite density AND Maxwellian floor are transformed.
Temperature here is the physical bath temperature, not the generated stress.
    The target factor is evaluated from its original log-Cholesky coordinates,
    never reconstructed from C (which could lose a small eigenvalue to roundoff).
"""
    _legacy_api()  # Verify the original immutable physical-module SHA bundle.
    analytic = importlib.import_module("agent_verifier.vpfp.analytic_source")
    dtype = torch.float64
    source_covariance = fields["source_central_covariance"].detach().to(dtype)
    target = decoder.macro_at(query.detach(), carrier["m"],
        {"rho_coefficients": fields["rho_coefficients"].detach()})["factor"]
    source = cholesky_from_log(log_cholesky(source_covariance))
    # A = L_source L_target^{-1}, retaining cross terms for anisotropic frames.
    affine = torch.linalg.solve_triangular(target.transpose(-1, -2),
        source.transpose(-1, -2), upper=True).transpose(-1, -2)
    centered = torch.linalg.solve_triangular(target,
        (query[:, 3:5].detach().to(dtype)-fields["velocity"].detach().to(dtype))[..., None],
        upper=False)[..., 0]
    velocity = fields["source_mean_velocity"].detach().to(dtype)+(source@centered[..., None])[..., 0]
    theta = fields["equation_condition"][:, 1].detach().to(dtype)
    coefficients = fields["source_hermite_coefficients"].detach().to(dtype)
    modes = decoder.base_decoder.hermite_modes
    maximum = max(max(int(a), int(b)) for a, b in modes)
    scale = torch.sqrt(2*theta)
    # This SHA-checked recurrence is also used by the strong source kernel.
    x = analytic._hermite_1d_jet(torch, velocity[:, 0], maximum, scale)
    y = analytic._hermite_1d_jet(torch, velocity[:, 1], maximum, scale)
    first = torch.tensor([a for a, _ in modes], device=query.device, dtype=torch.long)
    second = torch.tensor([b for _, b in modes], device=query.device, dtype=torch.long)
    x = tuple(value.index_select(1, first) for value in x)
    y = tuple(value.index_select(1, second) for value in y)
    def contract(dx, dy):
        return (coefficients*x[dx]*y[dy]).sum(-1)
    amplitude, ax, ay = contract(0, 0), contract(1, 0), contract(0, 1)
    axx, axy, ayy = contract(2, 0), contract(1, 1), contract(0, 2)
    norm = coefficients.square().sum(-1).clamp_min(torch.finfo(dtype).tiny)
    floor = float(decoder.base_decoder.config.velocity_floor_mixture)
    gaussian = torch.exp(-.5*velocity.square().sum(-1)/theta)/(2*torch.pi*theta)
    xx = (1-floor)*2*(ax.square()+amplitude*axx)/norm
    xy = (1-floor)*2*(ax*ay+amplitude*axy)/norm
    yy = (1-floor)*2*(ay.square()+amplitude*ayy)/norm
    xx += floor*gaussian*(velocity[:, 0].square()/theta.square()-1/theta)
    xy += floor*gaussian*velocity[:, 0]*velocity[:, 1]/theta.square()
    yy += floor*gaussian*(velocity[:, 1].square()/theta.square()-1/theta)
    metric = affine@affine.transpose(-1, -2)
    jacobian = source.diagonal(dim1=-2, dim2=-1).prod(-1)/target.diagonal(dim1=-2, dim2=-1).prod(-1)
    return fields["rho"].detach().to(dtype)*jacobian*(
        xx*metric[:, 0, 0]+2*xy*metric[:, 0, 1]+yy*metric[:, 1, 1])


@torch.enable_grad()
def manufacture_moving_frame_source_fast(decoder, points, carrier, condition, *, create_graph=False):
    """Numerical source with the old field/gradient ABI; never a qualification.

    Higher parameter/source derivatives requested by guidance deliberately use
    the original differentiable implementation. Numerical audit callers avoid
    the two velocity second-backward passes; coordinate first derivatives keep
    the original mixed native-FP32/affine-FP64 computation unchanged.
    """
    if type(decoder) is not MovingFrameDecoder:
        raise TypeError("fast source requires the actual MovingFrameDecoder")
    if create_graph:
        return manufacture_moving_frame_source(decoder, points, carrier, condition, create_graph=True)
    query = _query(decoder, points, 5)
    if not query.requires_grad:
        query = query.detach().clone().requires_grad_(True)
    values = decoder.fields(query, carrier, equation_condition=condition)
    laplacian = _velocity_laplacian(decoder, query, values, carrier)
    gradient = torch.autograd.grad(values["f"].sum(), query, create_graph=False, retain_graph=False)[0]
    physical = values["equation_condition"].detach().double()
    collision = 2*values["f"].detach()+(query[:, 3:5].detach()*gradient[:, 3:5]).sum(1)+physical[:, 1]*laplacian
    source = gradient[:, 0]+(query[:, 3:5].detach()*gradient[:, 1:3]).sum(1)
    source += (values["electric"].detach().double()*gradient[:, 3:5]).sum(1)-physical[:, 0]*collision
    result = {**values, "source": source, "f_gradient": gradient,
        "f_velocity_laplacian": laplacian, "collision": collision, "points": query}
    if any(not bool(torch.isfinite(value).all()) for value in result.values()):
        raise FloatingPointError("nonfinite fast moving-frame source; no clipping or dropped queries")
    return result
