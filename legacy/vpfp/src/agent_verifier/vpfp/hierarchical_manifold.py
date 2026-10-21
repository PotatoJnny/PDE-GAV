from __future__ import annotations

from typing import Any

from agent_verifier.vpfp.structured_bottleneck import fit_dual_pca
from agent_verifier.vpfp.structured_diffusion import (
    join_active_latent,
    split_active_latent,
)


_TORCH_IMPORT_ERROR = None
try:
    import torch as _TORCH_MODULE
except Exception as _exc:  # noqa: BLE001
    _TORCH_MODULE = None
    _TORCH_IMPORT_ERROR = _exc


HIERARCHICAL_MANIFOLD_SCHEMA = "vpfp_hierarchical_global_pca_v1"


def fit_hierarchical_global_pca(
    values,
    config,
    *,
    macro_max_rank: int,
    micro_max_rank: int,
) -> dict[str, Any]:
    """Fit independent case-level PCA maps for macro and micro trajectories."""

    torch = _torch()
    tensor = torch.as_tensor(values)
    _validate_values(tensor, config)
    rho, local = split_active_latent(tensor, config)
    macro = fit_dual_pca(
        rho.reshape(int(tensor.shape[0]), -1), max_rank=int(macro_max_rank)
    )
    micro = fit_dual_pca(
        local.reshape(int(tensor.shape[0]), -1), max_rank=int(micro_max_rank)
    )
    return {
        "schema": HIERARCHICAL_MANIFOLD_SCHEMA,
        "structured_config": config.to_dict(),
        "centering": "case_level_empirical_mean",
        "code_scaling": "principal_standard_deviation",
        "macro": macro,
        "micro": micro,
    }


def encode_hierarchical_macro_pca(
    values,
    basis: dict[str, Any],
    config,
    *,
    rank: int,
    whiten: bool = True,
):
    torch = _torch()
    tensor = torch.as_tensor(values)
    _validate_values(tensor, config)
    _validate_basis(basis, config)
    rho, _ = split_active_latent(tensor, config)
    return _encode_block(
        rho.reshape(int(tensor.shape[0]), -1),
        basis["macro"],
        rank=int(rank),
        whiten=bool(whiten),
    )


def decode_hierarchical_macro_pca(
    code,
    basis: dict[str, Any],
    config,
    *,
    whiten: bool = True,
):
    torch = _torch()
    tensor = torch.as_tensor(code)
    if tensor.ndim != 2:
        raise ValueError("hierarchical macro code must be a matrix")
    _validate_basis(basis, config)
    rho = _decode_block(tensor, basis["macro"], whiten=bool(whiten))
    return rho.reshape(
        int(tensor.shape[0]),
        config.rho_time_count,
        config.rho_feature_count,
    )



def encode_hierarchical_global_pca(
    values,
    basis: dict[str, Any],
    config,
    *,
    macro_rank: int,
    micro_rank: int,
    whiten: bool = True,
):
    """Encode structured active coordinates to separate macro/micro codes."""

    torch = _torch()
    tensor = torch.as_tensor(values)
    _validate_values(tensor, config)
    _validate_basis(basis, config)
    rho, local = split_active_latent(tensor, config)
    macro = _encode_block(
        rho.reshape(int(tensor.shape[0]), -1),
        basis["macro"],
        rank=int(macro_rank),
        whiten=bool(whiten),
    )
    micro = _encode_block(
        local.reshape(int(tensor.shape[0]), -1),
        basis["micro"],
        rank=int(micro_rank),
        whiten=bool(whiten),
    )
    return macro, micro


def decode_hierarchical_global_pca(
    macro_code,
    micro_code,
    basis: dict[str, Any],
    config,
    *,
    whiten: bool = True,
):
    """Decode separate case-level codes to structured active coordinates."""

    torch = _torch()
    macro = torch.as_tensor(macro_code)
    micro = torch.as_tensor(
        micro_code, dtype=macro.dtype, device=macro.device
    )
    if macro.ndim != 2 or micro.ndim != 2:
        raise ValueError("hierarchical manifold codes must be matrices")
    if int(macro.shape[0]) != int(micro.shape[0]):
        raise ValueError("macro and micro code batches must match")
    _validate_basis(basis, config)
    rho_flat = _decode_block(
        macro, basis["macro"], whiten=bool(whiten)
    )
    local_flat = _decode_block(
        micro, basis["micro"], whiten=bool(whiten)
    )
    batch = int(macro.shape[0])
    rho = rho_flat.reshape(
        batch, config.rho_time_count, config.rho_feature_count
    )
    local = local_flat.reshape(
        batch,
        config.local_channels,
        config.local_time_count,
        config.local_spatial_feature_count,
    )
    return join_active_latent(rho, local, config)


def hierarchical_manifold_basis_to_cpu(basis: dict[str, Any]) -> dict[str, Any]:
    """Detach tensor leaves so a fitted basis has a portable checkpoint."""

    torch = _torch()

    def convert(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, list):
            return [convert(item) for item in value]
        return value

    return convert(dict(basis))


def _encode_block(values, block, *, rank: int, whiten: bool):
    torch = _torch()
    tensor = torch.as_tensor(values)
    mean = torch.as_tensor(
        block["mean"], dtype=tensor.dtype, device=tensor.device
    )
    components = torch.as_tensor(
        block["components"], dtype=tensor.dtype, device=tensor.device
    )
    selected_rank = _selected_rank(rank, components)
    code = (tensor - mean) @ components[:, :selected_rank]
    if whiten:
        code = code / _code_scale(
            block, selected_rank, dtype=tensor.dtype, device=tensor.device
        )
    return code


def _decode_block(code, block, *, whiten: bool):
    torch = _torch()
    tensor = torch.as_tensor(code)
    mean = torch.as_tensor(
        block["mean"], dtype=tensor.dtype, device=tensor.device
    )
    components = torch.as_tensor(
        block["components"], dtype=tensor.dtype, device=tensor.device
    )
    selected_rank = _selected_rank(int(tensor.shape[1]), components)
    score = tensor
    if whiten:
        score = score * _code_scale(
            block, selected_rank, dtype=tensor.dtype, device=tensor.device
        )
    return mean + score @ components[:, :selected_rank].T


def _code_scale(block, rank: int, *, dtype, device):
    torch = _torch()
    variance = torch.as_tensor(
        block["explained_variance"], dtype=dtype, device=device
    )[: int(rank)]
    if not bool(torch.all(variance > 0.0).item()):
        raise ValueError("hierarchical PCA retained a nonpositive variance")
    return torch.sqrt(variance)


def _selected_rank(rank: int, components) -> int:
    selected = int(rank)
    if selected < 1 or selected > int(components.shape[1]):
        raise ValueError("hierarchical PCA rank is unavailable")
    return selected


def _validate_values(values, config) -> None:
    torch = _torch()
    if values.ndim != 2 or int(values.shape[1]) != int(
        config.active_latent_dim
    ):
        raise ValueError("values do not match the hierarchical structured layout")
    if not values.is_floating_point():
        raise ValueError("hierarchical manifold values must be floating point")
    if not bool(torch.all(torch.isfinite(values)).item()):
        raise ValueError("hierarchical manifold values must be finite")


def _validate_basis(basis: dict[str, Any], config) -> None:
    if basis.get("schema") != HIERARCHICAL_MANIFOLD_SCHEMA:
        raise ValueError("unexpected hierarchical manifold basis schema")
    if dict(basis.get("structured_config") or {}) != config.to_dict():
        raise ValueError("hierarchical manifold structured layout changed")
    for name in ("macro", "micro"):
        block = dict(basis.get(name) or {})
        if not {"mean", "components", "explained_variance"} <= set(block):
            raise ValueError(f"hierarchical manifold {name} basis is incomplete")


def _torch():
    if _TORCH_MODULE is None:
        raise RuntimeError("PyTorch is required") from _TORCH_IMPORT_ERROR
    return _TORCH_MODULE
