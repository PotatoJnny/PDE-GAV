from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from agent_verifier.vpfp.basis import VPFPBasisConfig, decode_basis_coeffs, project_g_to_basis, velocity_basis
from agent_verifier.vpfp.io import load_case
from agent_verifier.vpfp.numpy_models import NumpyVPFPTriplaneAutoencoder, infer_json_checkpoint_backend, load_numpy_autoencoder_checkpoint
from agent_verifier.vpfp.operators import EPS, VPFPGrid, f_from_rho_g, relative_l2
from agent_verifier.vpfp.triplane import TriplaneBatch, plane_tuple

Array = np.ndarray
_TORCH_IMPORT_ERROR = None
try:
    import torch as _TORCH_MODULE  # type: ignore[no-redef]
except Exception as _exc:  # noqa: BLE001
    _TORCH_MODULE = None
    _TORCH_IMPORT_ERROR = _exc


@dataclass(frozen=True)
class VPFPTriplaneAutoencoderConfig:
    input_channels: int = 8
    plane_channels: int = 16
    hidden_width: int = 64
    basis: VPFPBasisConfig = VPFPBasisConfig()
    dtype: str = "float32"
    representation: str = "vpfp_velocity_basis_triplane_v1"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["basis"] = self.basis.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VPFPTriplaneAutoencoderConfig":
        out = dict(data)
        out["basis"] = VPFPBasisConfig.from_dict(out.get("basis"))
        return cls(**out)


class VPFPTriplaneAutoencoder:
    @staticmethod
    def build(config: VPFPTriplaneAutoencoderConfig | dict[str, Any]):
        torch = _torch()
        cfg = config if isinstance(config, VPFPTriplaneAutoencoderConfig) else VPFPTriplaneAutoencoderConfig.from_dict(dict(config))
        return _VPFPTriplaneAutoencoder(torch, cfg)


class _VPFPTriplaneAutoencoder((_TORCH_MODULE.nn.Module if _TORCH_MODULE is not None else object)):
    def __init__(self, torch_module, config: VPFPTriplaneAutoencoderConfig):
        super().__init__()
        self.torch = torch_module
        self.config = config
        torch = torch_module
        c_in = int(config.input_channels)
        c_plane = int(config.plane_channels)
        width = int(config.hidden_width)
        self.xy_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(c_in, width, kernel_size=3, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(width, c_plane, kernel_size=3, padding=1),
        )
        self.xt_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(c_in, width, kernel_size=3, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(width, c_plane, kernel_size=3, padding=1),
        )
        self.yt_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(c_in, width, kernel_size=3, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(width, c_plane, kernel_size=3, padding=1),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Conv3d(c_plane, width, kernel_size=1),
            torch.nn.SiLU(),
            torch.nn.Conv3d(width, c_in, kernel_size=1),
        )
        dtype = torch.float64 if config.dtype == "float64" else torch.float32
        self.to(dtype=dtype)

    def encode(self, features):
        # features: [B, C, T, X, Y]
        xy = self.xy_encoder(features.mean(dim=2))
        xt_in = features.mean(dim=4).permute(0, 1, 3, 2).contiguous()  # [B,C,X,T]
        yt_in = features.mean(dim=3).permute(0, 1, 3, 2).contiguous()  # [B,C,Y,T]
        return TriplaneBatch(
            p_xy=xy,
            p_xt=self.xt_encoder(xt_in),
            p_yt=self.yt_encoder(yt_in),
        )

    def decode_planes(self, planes: TriplaneBatch | tuple[Any, Any, Any], *, nt: int | None = None, nx: int | None = None, ny: int | None = None):
        p_xy, p_xt, p_yt = plane_tuple(planes)
        torch = self.torch
        batch = p_xy.shape[0]
        t_count = int(nt or p_xt.shape[-1])
        x_count = int(nx or p_xy.shape[-2])
        y_count = int(ny or p_xy.shape[-1])
        if p_xt.shape[-2] != x_count or p_xt.shape[-1] != t_count:
            p_xt = torch.nn.functional.interpolate(p_xt, size=(x_count, t_count), mode="bilinear", align_corners=False)
        if p_yt.shape[-2] != y_count or p_yt.shape[-1] != t_count:
            p_yt = torch.nn.functional.interpolate(p_yt, size=(y_count, t_count), mode="bilinear", align_corners=False)
        if p_xy.shape[-2:] != (x_count, y_count):
            p_xy = torch.nn.functional.interpolate(p_xy, size=(x_count, y_count), mode="bilinear", align_corners=False)
        xy = p_xy[:, :, None, :, :].expand(batch, -1, t_count, x_count, y_count)
        xt = p_xt.permute(0, 1, 3, 2)[:, :, :, :, None].expand(batch, -1, t_count, x_count, y_count)
        yt = p_yt.permute(0, 1, 3, 2)[:, :, :, None, :].expand(batch, -1, t_count, x_count, y_count)
        return self.decoder((xy + xt + yt) / 3.0)

    def decode_points(
        self,
        planes: TriplaneBatch | tuple[Any, Any, Any],
        points,
        *,
        x_domain: tuple[float, float],
        time_bounds: tuple[float, float],
    ):
        """Decode features at physical (t, x, y) query points.

        Spatial queries are periodic and time queries are nonperiodic. Bicubic
        interpolation keeps the decoder differentiable with respect to the
        coordinates, which is required when manufacturing continuous VPFP
        sources with autograd.
        """

        p_xy, p_xt, p_yt = plane_tuple(planes)
        torch = self.torch
        batch = int(p_xy.shape[0])
        if points.ndim == 2:
            points = points[None, ...]
        if points.ndim != 3 or int(points.shape[-1]) != 3:
            raise ValueError(f"points must have shape [B,N,3] or [N,3], got {tuple(points.shape)}")
        if int(points.shape[0]) == 1 and batch > 1:
            points = points.expand(batch, -1, -1)
        if int(points.shape[0]) != batch:
            raise ValueError(f"point batch {int(points.shape[0])} does not match plane batch {batch}")
        if any(int(size) < 2 for size in (*p_xy.shape[-2:], *p_xt.shape[-2:], *p_yt.shape[-2:])):
            raise ValueError("continuous triplane queries require every plane axis to have at least two samples")

        t0, t1 = (float(value) for value in time_bounds)
        lx, ly = (float(value) for value in x_domain)
        if t1 <= t0 or lx <= 0.0 or ly <= 0.0:
            raise ValueError("time_bounds and x_domain must have positive extent")
        t_fraction = (points[..., 0] - t0) / (t1 - t0)
        x_fraction = torch.remainder(points[..., 1], lx) / lx
        y_fraction = torch.remainder(points[..., 2], ly) / ly

        xy = self._sample_periodic_xy(p_xy, x_fraction, y_fraction)
        xt = self._sample_periodic_space_time(p_xt, x_fraction, t_fraction)
        yt = self._sample_periodic_space_time(p_yt, y_fraction, t_fraction)
        fused = ((xy + xt + yt) / 3.0)[:, :, :, None, None]
        return self.decoder(fused)[:, :, :, 0, 0]

    def _sample_periodic_xy(self, plane, first_fraction, second_fraction):
        torch = self.torch
        first_count, second_count = (int(value) for value in plane.shape[-2:])
        tiled = plane.repeat(1, 1, 3, 3)
        first_index = first_count + first_fraction * first_count
        second_index = second_count + second_fraction * second_count
        grid = torch.stack(
            (
                self._normalized_grid_index(second_index, 3 * second_count),
                self._normalized_grid_index(first_index, 3 * first_count),
            ),
            dim=-1,
        )[:, :, None, :]
        return torch.nn.functional.grid_sample(
            tiled,
            grid,
            mode="bicubic",
            padding_mode="border",
            align_corners=True,
        )[:, :, :, 0]

    def _sample_periodic_space_time(self, plane, space_fraction, time_fraction):
        torch = self.torch
        space_count, time_count = (int(value) for value in plane.shape[-2:])
        tiled = plane.repeat(1, 1, 3, 1)
        space_index = space_count + space_fraction * space_count
        time_index = time_fraction * (time_count - 1)
        grid = torch.stack(
            (
                self._normalized_grid_index(time_index, time_count),
                self._normalized_grid_index(space_index, 3 * space_count),
            ),
            dim=-1,
        )[:, :, None, :]
        return torch.nn.functional.grid_sample(
            tiled,
            grid,
            mode="bicubic",
            padding_mode="border",
            align_corners=True,
        )[:, :, :, 0]

    @staticmethod
    def _normalized_grid_index(index, count: int):
        return 2.0 * index / float(count - 1) - 1.0

    def forward(self, features):
        return self.decode_planes(self.encode(features), nt=features.shape[2], nx=features.shape[3], ny=features.shape[4])


def case_to_features(case_dir: str | Path, basis_config: VPFPBasisConfig | dict[str, Any] | None = None) -> tuple[Array, dict[str, Any]]:
    case = load_case(case_dir)
    grid = case["grid"]
    hidden = case["hidden_arrays"]
    rho = np.asarray(hidden["rho_true"], dtype=float)
    g = np.asarray(hidden["g_true"], dtype=float)
    basis_cfg = basis_config if basis_config is not None else case["public_question"].get("basis", {}).get("config")
    if "basis_coeffs_true" in hidden:
        coeffs = np.asarray(hidden["basis_coeffs_true"], dtype=float)
        atoms, basis_meta = velocity_basis(grid, basis_cfg)
    else:
        coeffs, _recon, basis_meta = project_g_to_basis(g, grid, basis_cfg)
    logrho = np.log(np.maximum(rho, EPS))[None, :, :, :]
    coeff_channels = np.moveaxis(coeffs, -1, 0)
    features = np.concatenate([logrho, coeff_channels], axis=0).astype(np.float32)
    return features, {"case": case, "basis": basis_meta}


def features_to_rho_g(
    features: Array,
    grid: VPFPGrid,
    basis_config: VPFPBasisConfig | dict[str, Any] | None = None,
    *,
    coeffs_are_logits: bool = True,
) -> tuple[Array, Array, Array]:
    arr = np.asarray(features, dtype=float)
    if arr.ndim != 4:
        raise ValueError(f"features must have shape [C,T,X,Y], got {arr.shape}")
    atoms, _metadata = velocity_basis(grid, basis_config)
    rho = np.exp(np.clip(arr[0], -20.0, 20.0))
    raw_coeffs = np.moveaxis(arr[1 : 1 + atoms.shape[0]], 0, -1)
    if coeffs_are_logits:
        logits = raw_coeffs - np.max(raw_coeffs, axis=-1, keepdims=True)
        coeffs = np.exp(logits)
    else:
        coeffs = np.maximum(raw_coeffs, 0.0)
        zero_mask = np.sum(coeffs, axis=-1, keepdims=True) <= EPS
        if np.any(zero_mask):
            coeffs = np.where(zero_mask, 1.0 / float(atoms.shape[0]), coeffs)
    coeffs = coeffs / np.maximum(np.sum(coeffs, axis=-1, keepdims=True), EPS)
    g = decode_basis_coeffs(coeffs, atoms)
    return rho.astype(float), g.astype(float), coeffs.astype(float)


def reconstruct_case(model, case_dir: str | Path, *, device="cpu") -> dict[str, Any]:
    features, meta = case_to_features(case_dir, model.config.basis)
    if isinstance(model, NumpyVPFPTriplaneAutoencoder):
        pred = model.reconstruct_features(features)
    else:
        torch = _torch()
        param = next(model.parameters())
        if str(device) == "auto":
            device = param.device
        tensor = torch.as_tensor(features[None], dtype=param.dtype, device=device)
        with torch.no_grad():
            pred = model(tensor)[0].detach().cpu().numpy()
    case = meta["case"]
    rho, g, coeffs = features_to_rho_g(pred, case["grid"], model.config.basis, coeffs_are_logits=not isinstance(model, NumpyVPFPTriplaneAutoencoder))
    hidden = case["hidden_arrays"]
    f_true = f_from_rho_g(hidden["rho_true"], hidden["g_true"])
    f_pred = f_from_rho_g(rho, g)
    return {
        "case_id": case["case_id"],
        "rho": rho,
        "g": g,
        "basis_coeffs": coeffs,
        "rho_rel_l2": relative_l2(rho - hidden["rho_true"], hidden["rho_true"]),
        "g_rel_l2": relative_l2(g - hidden["g_true"], hidden["g_true"]),
        "f_rel_l2": relative_l2(f_pred - f_true, f_true),
    }


def save_autoencoder_checkpoint(path: str | Path, model, optimizer, config: VPFPTriplaneAutoencoderConfig, train_config: dict[str, Any], step: int) -> None:
    torch = _torch()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": config.to_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "train_config": train_config,
            "step": int(step),
        },
        Path(path),
    )


def load_autoencoder_checkpoint(path: str | Path, *, device="cpu"):
    backend = infer_json_checkpoint_backend(path)
    if backend == NumpyVPFPTriplaneAutoencoder.backend:
        return load_numpy_autoencoder_checkpoint(path, config_loader=VPFPTriplaneAutoencoderConfig.from_dict)
    torch = _torch()
    if str(device) == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    cfg = VPFPTriplaneAutoencoderConfig.from_dict(payload["model_config"])
    model = VPFPTriplaneAutoencoder.build(cfg).to(device=device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def _torch():
    if _TORCH_MODULE is None:
        raise RuntimeError("torch is required for VPFP triplane autoencoder") from _TORCH_IMPORT_ERROR
    return _TORCH_MODULE
