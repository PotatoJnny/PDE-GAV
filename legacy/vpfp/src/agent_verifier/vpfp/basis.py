from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from agent_verifier.vpfp.operators import EPS, VPFPGrid, make_axes, normalize_g, relative_l2

Array = np.ndarray


@dataclass(frozen=True)
class VPFPBasisConfig:
    thermal_speeds: tuple[float, ...] = (0.75, 1.0, 1.35)
    stream_speeds: tuple[float, ...] = (1.25, 1.85)
    stream_sigma_scale: float = 0.65
    include_y_streams: bool = True
    include_shifted_streams: bool = True
    dtype: str = "float64"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["thermal_speeds"] = [float(v) for v in self.thermal_speeds]
        data["stream_speeds"] = [float(v) for v in self.stream_speeds]
        data["include_shifted_streams"] = bool(self.include_shifted_streams)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "VPFPBasisConfig":
        if not data:
            return cls()
        return cls(
            thermal_speeds=tuple(float(v) for v in data.get("thermal_speeds", cls.thermal_speeds)),
            stream_speeds=tuple(float(v) for v in data.get("stream_speeds", cls.stream_speeds)),
            stream_sigma_scale=float(data.get("stream_sigma_scale", cls.stream_sigma_scale)),
            include_y_streams=bool(data.get("include_y_streams", cls.include_y_streams)),
            # Missing key means legacy 7-atom basis metadata. New generated cases write this key explicitly.
            include_shifted_streams=bool(data.get("include_shifted_streams", False)),
            dtype=str(data.get("dtype", cls.dtype)),
        )


def velocity_basis(grid: VPFPGrid, config: VPFPBasisConfig | dict[str, Any] | None = None) -> tuple[Array, dict[str, Any]]:
    """Return normalized fixed velocity atoms B_k(vx, vy).

    Each atom has discrete mass one under the repository's velocity quadrature.
    The basis is intentionally small and smooth: centered Maxwellians plus
    symmetric two-stream atoms along x and, optionally, y.
    """

    cfg = config if isinstance(config, VPFPBasisConfig) else VPFPBasisConfig.from_dict(config)
    axes = make_axes(grid)
    vx = axes["vx"][:, None]
    vy = axes["vy"][None, :]
    atoms: list[Array] = []
    names: list[str] = []
    params: list[dict[str, Any]] = []
    for sigma in cfg.thermal_speeds:
        atom = _maxwellian(vx, vy, 0.0, 0.0, float(sigma))
        atoms.append(normalize_g(atom, grid))
        names.append(f"maxwellian_center_sigma_{float(sigma):.3g}")
        params.append({"type": "maxwellian", "ux": 0.0, "uy": 0.0, "sigma": float(sigma)})
    for stream in cfg.stream_speeds:
        sigma = max(0.35, float(cfg.stream_sigma_scale) * float(stream))
        plus = _maxwellian(vx, vy, float(stream), 0.0, sigma)
        minus = _maxwellian(vx, vy, -float(stream), 0.0, sigma)
        atoms.append(normalize_g(0.5 * plus + 0.5 * minus, grid))
        names.append(f"bimodal_x_stream_{float(stream):.3g}")
        params.append({"type": "bimodal_x", "stream": float(stream), "sigma": float(sigma)})
        if bool(cfg.include_shifted_streams):
            atoms.append(normalize_g(plus, grid))
            names.append(f"shifted_x_plus_{float(stream):.3g}")
            params.append({"type": "shifted_x", "stream": float(stream), "sign": 1.0, "sigma": float(sigma)})
            atoms.append(normalize_g(minus, grid))
            names.append(f"shifted_x_minus_{float(stream):.3g}")
            params.append({"type": "shifted_x", "stream": float(stream), "sign": -1.0, "sigma": float(sigma)})
        if bool(cfg.include_y_streams):
            plus_y = _maxwellian(vx, vy, 0.0, float(stream), sigma)
            minus_y = _maxwellian(vx, vy, 0.0, -float(stream), sigma)
            atoms.append(normalize_g(0.5 * plus_y + 0.5 * minus_y, grid))
            names.append(f"bimodal_y_stream_{float(stream):.3g}")
            params.append({"type": "bimodal_y", "stream": float(stream), "sigma": float(sigma)})
            if bool(cfg.include_shifted_streams):
                atoms.append(normalize_g(plus_y, grid))
                names.append(f"shifted_y_plus_{float(stream):.3g}")
                params.append({"type": "shifted_y", "stream": float(stream), "sign": 1.0, "sigma": float(sigma)})
                atoms.append(normalize_g(minus_y, grid))
                names.append(f"shifted_y_minus_{float(stream):.3g}")
                params.append({"type": "shifted_y", "stream": float(stream), "sign": -1.0, "sigma": float(sigma)})
    arr = np.asarray(atoms, dtype=cfg.dtype)
    metadata = {
        "basis_type": "fixed_normalized_velocity_atoms_v1",
        "config": cfg.to_dict(),
        "atom_count": int(arr.shape[0]),
        "atom_names": names,
        "atoms": params,
        "normalization": "sum_v B_k(vx,vy) * dvx * dvy = 1",
    }
    return arr, metadata


def basis_config_from_metadata(metadata: dict[str, Any] | None) -> VPFPBasisConfig:
    if not metadata:
        return VPFPBasisConfig()
    if "config" in metadata:
        return VPFPBasisConfig.from_dict(dict(metadata["config"]))
    return VPFPBasisConfig.from_dict(metadata)


def decode_basis_coeffs(coeffs: Array, atoms: Array) -> Array:
    coeff_arr = simplex_project(coeffs)
    atom_arr = np.asarray(atoms, dtype=float)
    if coeff_arr.shape[-1] != atom_arr.shape[0]:
        raise ValueError(f"basis coefficient count {coeff_arr.shape[-1]} does not match atom count {atom_arr.shape[0]}")
    return np.einsum("...k,kvw->...vw", coeff_arr, atom_arr).astype(float)


def simplex_project(coeffs: Array) -> Array:
    arr = np.asarray(coeffs, dtype=float)
    out = np.maximum(arr, 0.0)
    total = np.sum(out, axis=-1, keepdims=True)
    if np.any(total <= EPS):
        out = out.copy()
        bad = np.squeeze(total <= EPS, axis=-1)
        out[bad, :] = 1.0 / float(out.shape[-1])
        total = np.sum(out, axis=-1, keepdims=True)
    return out / np.maximum(total, EPS)


def project_g_to_basis(g: Array, grid: VPFPGrid, config: VPFPBasisConfig | dict[str, Any] | None = None) -> tuple[Array, Array, dict[str, Any]]:
    atoms, metadata = velocity_basis(grid, config)
    g_arr = np.asarray(g, dtype=float)
    if g_arr.shape[-2:] != (grid.nvx, grid.nvy):
        raise ValueError(f"g trailing shape must be {(grid.nvx, grid.nvy)}, got {g_arr.shape}")
    atom_matrix = atoms.reshape(atoms.shape[0], -1).T
    weights = np.sqrt(float(grid.velocity_cell_area)) * np.ones(atom_matrix.shape[0], dtype=float)
    weighted_atoms = atom_matrix * weights[:, None]
    flat = g_arr.reshape(-1, atom_matrix.shape[0]) * weights[None, :]
    coeffs, *_ = np.linalg.lstsq(weighted_atoms, flat.T, rcond=None)
    coeffs = simplex_project(coeffs.T.reshape(*g_arr.shape[:-2], atoms.shape[0]))
    recon = decode_basis_coeffs(coeffs, atoms)
    metadata = {
        **metadata,
        "projection": "weighted_lstsq_then_nonnegative_simplex_projection",
        "projection_rel_l2": relative_l2(recon - g_arr, g_arr),
    }
    return coeffs.astype(float), recon.astype(float), metadata


def basis_projection_metrics(rho: Array, g: Array, grid: VPFPGrid, config: VPFPBasisConfig | dict[str, Any] | None = None) -> dict[str, float]:
    coeffs, recon_g, _metadata = project_g_to_basis(g, grid, config)
    rho_arr = np.asarray(rho, dtype=float)
    f_true = rho_arr[..., None, None] * np.asarray(g, dtype=float)
    f_recon = rho_arr[..., None, None] * recon_g
    mass = np.sum(recon_g, axis=(-2, -1)) * grid.velocity_cell_area
    return {
        "basis_coeff_min": float(np.min(coeffs)),
        "basis_coeff_simplex_max_abs_error": float(np.max(np.abs(np.sum(coeffs, axis=-1) - 1.0))),
        "basis_g_projection_rel_l2": relative_l2(recon_g - g, g),
        "basis_f_projection_rel_l2": relative_l2(f_recon - f_true, f_true),
        "basis_g_normalization_max_abs_error": float(np.max(np.abs(mass - 1.0))),
    }


def _maxwellian(vx: Array, vy: Array, ux: float, uy: float, sigma: float) -> Array:
    var = max(float(sigma) ** 2, EPS)
    return np.exp(-0.5 * (((vx - float(ux)) ** 2 + (vy - float(uy)) ** 2) / var)) / (2.0 * np.pi * var)
