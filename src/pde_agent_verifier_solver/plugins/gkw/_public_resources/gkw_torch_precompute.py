"""Compact PyTorch precompute for the pinned GKW dataset regime.

The public equation packet contains raw normalized grids, geometry arrays,
topology, species parameters, and numerical controls. This module turns that
packet into the compact coefficient mapping consumed by the spectral
operator. It follows the pinned Gyaradax formulas but has no runtime
dependency on JAX or Gyaradax.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch

from .gkw_equation_contract import (
    GKWEquationContract,
    PINNED_GYARADAX_AUDIT_COMMIT,
)
from .gkw_torch_operator import GKWTorchSpectralOperator

_GYARADAX_COEFFICIENT_EPS = 1.0e-30

_EPS = 1.0e-15
_GYARADAX_BESSEL_N_ITER = 50
_GYARADAX_BESSEL_ZERO_THRESHOLD = 1.0e-10


@dataclass(frozen=True)
class GKWCompactPrecomputeSummary:
    coefficient_bytes: int
    tensor_count: int
    reference_id: str
    parallel_coefficient_mode: str = "compact"
    gyroaverage_j0_backend: str = "jax_bessel_jn_compat_n_iter_50"


def _jax_bessel_j0_compat(
    argument: torch.Tensor,
    *,
    n_iter: int = _GYARADAX_BESSEL_N_ITER,
) -> torch.Tensor:
    """Match the pinned JAX ``bessel_jn(..., v=0)`` recurrence.

    Gyaradax uses JAX's fixed-iteration Zhang-Jin backward recurrence rather
    than a platform libm J0 implementation. Reproducing that numerical
    operator is required for an exact dataset-contract parity check.
    """

    if not argument.is_floating_point():
        raise TypeError("Bessel arguments must be floating point")
    if n_iter < 0:
        raise ValueError("n_iter must be nonnegative")

    output_dtype = argument.dtype
    working = argument.to(dtype=torch.float64)
    near_zero = working.abs() < _GYARADAX_BESSEL_ZERO_THRESHOLD
    safe_argument = torch.where(near_zero, torch.ones_like(working), working)
    f0 = torch.zeros_like(safe_argument)
    f1 = torch.full_like(safe_argument, 1.0e-16)
    normalizer = torch.zeros_like(safe_argument)
    last_value = torch.zeros_like(safe_argument)
    for k in range(n_iter, -1, -1):
        value = 2.0 * (k + 1.0) * f1 / safe_argument - f0
        if k % 2 == 0:
            normalizer = normalizer + 2.0 * value
        f0, f1 = f1, value
        if k == 0:
            last_value = value
    result = last_value / (normalizer - last_value)
    result = torch.where(near_zero, torch.ones_like(result), result)
    return result.to(dtype=output_dtype)


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _tensor(
    value: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.requires_grad:
            raise ValueError("public GKW coefficients must be static")
        return value.detach().to(device=device, dtype=dtype)
    return torch.as_tensor(
        np.asarray(value).copy(), device=device, dtype=dtype
    )


def _scalar(
    scalars: Mapping[str, Any],
    name: str,
) -> float:
    value = _as_numpy(scalars[name]).reshape(-1)
    if value.size != 1 or not np.isfinite(value).all():
        raise ValueError(f"invalid public GKW scalar: {name}")
    return float(value[0])


def _prime_factors_at_most(number: int, maximum: int) -> bool:
    candidate = int(number)
    divisor = 2
    while True:
        if candidate % divisor == 0:
            candidate //= divisor
        elif divisor == maximum:
            return candidate == 1
        else:
            divisor += 1


def _extended_first_fft_size(nky: int) -> tuple[int, int]:
    size = 3 * nky - 2
    if size % 2:
        size += 1
    while not _prime_factors_at_most(size, 7):
        size += 2
    for offset in range(1, 9):
        candidate = size + 2 * offset
        if _prime_factors_at_most(candidate, 2):
            size = candidate
            break
    return size, size // 2 + 1


def _extended_second_fft_size(nkx: int) -> int:
    size = int(math.ceil(1.5 * float(nkx + 1)) + 1)
    while not _prime_factors_at_most(size, 7):
        size += 1
    for offset in range(1, 9):
        candidate = size + offset
        if _prime_factors_at_most(candidate, 2):
            size = candidate
            break
    return size


def _validate_topology(
    arrays: Mapping[str, Any],
    *,
    contract: GKWEquationContract,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    grid = contract.grid
    topology = {
        "s_shift": _tensor(
            arrays["s_shift"],
            device=device,
            dtype=torch.long,
        ),
        "kx_shift": _tensor(
            arrays["kx_shift"],
            device=device,
            dtype=torch.long,
        ),
        "valid_shift": _tensor(
            arrays["valid_shift"],
            device=device,
            dtype=torch.bool,
        ),
        "pos_parallel_class": _tensor(
            arrays["pos_parallel_class"],
            device=device,
            dtype=torch.long,
        ),
    }
    valid = topology["valid_shift"]
    if valid.any():
        s_valid = topology["s_shift"][valid]
        kx_valid = topology["kx_shift"][valid]
        if int(s_valid.min()) < 0 or int(s_valid.max()) >= grid.ns:
            raise ValueError("public GKW s topology contains an invalid index")
        if int(kx_valid.min()) < 0 or int(kx_valid.max()) >= grid.nkx:
            raise ValueError("public GKW kx topology contains an invalid index")
    position = topology["pos_parallel_class"]
    if int(position.min()) < -2 or int(position.max()) > 2:
        raise ValueError("public GKW parallel position class is invalid")
    return topology


def build_compact_gkw_coefficients(
    arrays: Mapping[str, Any],
    scalars: Mapping[str, Any],
    *,
    contract: GKWEquationContract | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> tuple[dict[str, Any], GKWCompactPrecomputeSummary]:
    """Build compact operator coefficients from a complete public packet."""

    contract = contract or GKWEquationContract()
    audit = contract.audit_public_coefficients(
        {name: _as_numpy(value) for name, value in arrays.items()},
        {name: _as_numpy(value) for name, value in scalars.items()},
    )
    if not audit.passed:
        raise ValueError(
            "public GKW packet failed its contract: "
            + ", ".join(audit.failures)
        )
    if dtype not in (torch.float32, torch.float64):
        raise TypeError("GKW precompute requires float32 or float64")
    device = torch.device(device)

    idisp = _scalar(scalars, "idisp")
    if not np.isclose(idisp, 2.0):
        raise ValueError("the pinned GKW corpus requires idisp=2")
    mass = _scalar(scalars, "mass_i")
    charge = _scalar(scalars, "charge_i")
    temperature = _scalar(scalars, "temperature_i")
    density = _scalar(scalars, "density_i")
    vth_ratio = _scalar(scalars, "vth_ratio_i")
    dgrid = _scalar(scalars, "dgrid")
    dvp = _scalar(scalars, "dvp")
    ds = _scalar(scalars, "ds")
    positive = {
        "mass_i": mass,
        "temperature_i": temperature,
        "density_i": density,
        "vth_ratio_i": vth_ratio,
        "dgrid": dgrid,
        "dvp": dvp,
        "ds": ds,
    }
    invalid_positive = sorted(
        name for name, value in positive.items() if value <= 0.0
    )
    if invalid_positive or abs(charge) < _EPS:
        names = list(invalid_positive)
        if abs(charge) < _EPS:
            names.append("charge_i")
        raise ValueError(
            "invalid positive/nonzero GKW scalars: " + ", ".join(names)
        )

    vpar = _tensor(arrays["vpar"], device=device, dtype=dtype)
    intvp = _tensor(arrays["intvp"], device=device, dtype=dtype)
    mu = _tensor(arrays["mu"], device=device, dtype=dtype)
    intmu = _tensor(arrays["intmu"], device=device, dtype=dtype)
    ints = _tensor(arrays["ints"], device=device, dtype=dtype)
    magnetic_field = _tensor(arrays["B"], device=device, dtype=dtype)
    ffun = _tensor(arrays["F"], device=device, dtype=dtype)
    gfun = _tensor(arrays["G"], device=device, dtype=dtype)
    efun = _tensor(arrays["E_psi_zeta"], device=device, dtype=dtype)
    drift_tensor = _tensor(arrays["D"], device=device, dtype=dtype)
    metric = _tensor(arrays["metric_perp"], device=device, dtype=dtype)
    kx = _tensor(arrays["kx"], device=device, dtype=dtype)
    ky = _tensor(arrays["ky"], device=device, dtype=dtype)
    if torch.any(magnetic_field <= 0):
        raise ValueError("public GKW magnetic-field magnitude must be positive")
    if torch.any(intvp < 0) or torch.any(intmu < 0) or torch.any(ints < 0):
        raise ValueError("public GKW quadrature weights must be nonnegative")

    topology = _validate_topology(
        arrays,
        contract=contract,
        device=device,
    )
    grid = contract.grid
    ixzero = int(round(_scalar(scalars, "ixzero")))
    iyzero = int(round(_scalar(scalars, "iyzero")))
    if ixzero != int(torch.argmin(kx.abs())):
        raise ValueError("public GKW ixzero disagrees with the kx grid")
    if iyzero != int(torch.argmin(ky.abs())):
        raise ValueError("public GKW iyzero disagrees with the ky grid")

    vp = vpar.reshape(-1, 1, 1, 1, 1)
    vp2 = vp.square()
    mu_b = mu.reshape(1, -1, 1, 1, 1)
    field_b = magnetic_field.reshape(1, 1, -1, 1, 1)
    ffun_b = ffun.reshape(1, 1, -1, 1, 1)
    gfun_b = gfun.reshape(1, 1, -1, 1, 1)
    efun_b = efun.reshape(1, 1, -1, 1, 1)
    kx_b = kx.reshape(1, 1, 1, -1, 1)
    ky_b = ky.reshape(1, 1, 1, 1, -1)
    metric_b = metric.T.reshape(3, 1, 1, -1, 1, 1)

    kperp_squared = (
        ky_b.square() * metric_b[0]
        + 2.0 * ky_b * kx_b * metric_b[1]
        + kx_b.square() * metric_b[2]
    )
    if float(kperp_squared.min()) < -1.0e-12:
        raise ValueError("public GKW perpendicular metric is not positive")
    kperp = kperp_squared.clamp_min(_GYARADAX_COEFFICIENT_EPS).sqrt()
    bessel_argument = (
        mass
        * vth_ratio
        * kperp
        * (2.0 * mu_b / field_b).clamp_min(_GYARADAX_COEFFICIENT_EPS).sqrt()
        / charge
    )
    bessel = _jax_bessel_j0_compat(bessel_argument)

    fmaxwl = (
        (density / dgrid)
        * torch.exp(-(vp2 + 2.0 * field_b * mu_b))
        / math.pi**1.5
    )
    energy_offset = vp2 + 2.0 * field_b * mu_b - 1.5
    drive = (
        (
            _scalar(scalars, "rln_i")
            + _scalar(scalars, "rlt_i") * energy_offset
        )
        * fmaxwl
        * efun_b
        * ky_b
    )
    kinetic_energy = vp2 + field_b * mu_b
    drift_x = (
        temperature
        * kinetic_energy
        * drift_tensor[:, 0].reshape(1, 1, -1, 1, 1)
        / charge
    )
    drift_y = (
        temperature
        * kinetic_energy
        * drift_tensor[:, 1].reshape(1, 1, -1, 1, 1)
        / charge
    )
    upar = -ffun_b * vth_ratio * vp
    utrap = vth_ratio * mu_b * field_b * gfun_b
    vp_rms = torch.sqrt(torch.mean(vpar.square()))
    mu_rms = torch.sqrt(torch.mean(mu.square()))
    abs_parallel = torch.abs(ffun_b * vth_ratio * vp_rms)
    abs_vpar = torch.abs(vth_ratio * field_b * gfun_b * mu_rms)
    term7 = -charge * ffun_b * vth_ratio * vp * fmaxwl / temperature

    kxmax = kx.abs().max().clamp_min(_EPS)
    kymax = ky.abs().max().clamp_min(_EPS)
    disp_x = _scalar(scalars, "disp_x")
    disp_y = _scalar(scalars, "disp_y")
    exponent_x = 2.0 if disp_x < 0.0 else 4.0
    exponent_y = 2.0 if disp_y < 0.0 else 4.0
    hyper = -(
        abs(disp_y) * (ky_b / kymax).pow(exponent_y)
        + abs(disp_x) * (kx_b / kxmax).pow(exponent_x)
    )

    gamma_argument = (
        0.5
        * (
            mass
            * vth_ratio
            * kperp
            / (charge * field_b)
        ).square()
    ).clamp(0.0, 500.0)
    gamma = torch.special.i0e(gamma_argument)
    phi_weight = (
        charge
        * density
        * intvp.reshape(-1, 1, 1, 1, 1)
        * intmu.reshape(1, -1, 1, 1, 1)
        * bessel
        * field_b
    )
    phi_weight = torch.where(
        intvp.abs().reshape(-1, 1, 1, 1, 1) < 1.0e-9,
        torch.zeros((), device=device, dtype=dtype),
        phi_weight,
    )
    denominator = (
        charge * (gamma - 1.0) / temperature - 1.0 / temperature
    )
    denominator = torch.where(
        denominator.abs() < _EPS,
        torch.ones((), device=device, dtype=dtype),
        denominator,
    )
    phi_corr_weight = (
        -ints.reshape(1, 1, -1, 1, 1)
        / (charge * density * denominator)
    )
    ky_zonal = torch.arange(grid.nky, device=device) == iyzero
    has_zonal = float(abs(float(ky[iyzero])) < 1.0e-10)
    phi_corr_weight = (
        phi_corr_weight
        * ky_zonal.reshape(1, 1, 1, 1, -1)
        * has_zonal
    )

    mphi, mphiw3 = _extended_first_fft_size(grid.nky)
    mrad = _extended_second_fft_size(grid.nkx)
    kx_index = torch.arange(grid.nkx, device=device)
    jind = torch.where(
        kx_index >= ixzero,
        kx_index - ixzero,
        mrad + kx_index - ixzero,
    )

    coefficients: dict[str, Any] = {
        "bessel": bessel,
        "phi_weight": phi_weight,
        "phi_corr_weight": phi_corr_weight,
        "phi_gamma": gamma,
        "fmaxwl": fmaxwl,
        "dmaxwel_fm_ek": drive,
        "drift_x": drift_x,
        "drift_y": drift_y,
        "utrap": utrap,
        "abs_dum2_vp": abs_vpar,
        "upar": upar,
        "abs_dum2_par": abs_parallel,
        "term7_fac": term7,
        "pos_parallel_class": topology["pos_parallel_class"],
        "hyper": hyper,
        "kx": kx,
        "ky": ky,
        "nl_dum_s": -efun,
        "nl_jind": jind,
        "s_shift": topology["s_shift"],
        "kx_shift": topology["kx_shift"],
        "valid_shift": topology["valid_shift"],
        "dvp": dvp,
        "disp_vp": _scalar(scalars, "disp_vp"),
        "disp_par": _scalar(scalars, "disp_par"),
        "sgr_dist": ds,
        "phi_tmp": temperature,
        "phi_de": density,
        "phi_signz": charge,
        "phi_has_zonal": has_zonal,
        "phi_ixzero": ixzero,
        "phi_iyzero": iyzero,
        "ixzero": ixzero,
        "iyzero": iyzero,
        "nl_mrad": mrad,
        "nl_mphi": mphi,
        "nl_mphiw3": mphiw3,
        "nl_fft_scale": float(mrad * mphi),
    }
    coefficient_bytes = sum(
        value.numel() * value.element_size()
        for value in coefficients.values()
        if isinstance(value, torch.Tensor)
    )
    reference_id = (
        f"torch_reimplementation:gyaradax:{PINNED_GYARADAX_AUDIT_COMMIT}"
    )
    return coefficients, GKWCompactPrecomputeSummary(
        coefficient_bytes=coefficient_bytes,
        tensor_count=sum(
            isinstance(value, torch.Tensor)
            for value in coefficients.values()
        ),
        reference_id=reference_id,
    )


def build_dataset_exact_gkw_operator(
    arrays: Mapping[str, Any],
    scalars: Mapping[str, Any],
    *,
    contract: GKWEquationContract | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> tuple[GKWTorchSpectralOperator, GKWCompactPrecomputeSummary]:
    contract = contract or GKWEquationContract()
    coefficients, summary = build_compact_gkw_coefficients(
        arrays,
        scalars,
        contract=contract,
        device=device,
        dtype=dtype,
    )
    return (
        GKWTorchSpectralOperator(
            coefficients,
            contract=contract,
            reference_id=summary.reference_id,
        ),
        summary,
    )
