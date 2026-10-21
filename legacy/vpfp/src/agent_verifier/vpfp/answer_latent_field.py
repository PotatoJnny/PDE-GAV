from __future__ import annotations

from typing import Any, Mapping

import torch


ANSWER_RESIDUAL_CHANNEL_CODEC_SCHEMA = (
    "vpfp_answer_residual_block_channel_pca_v1"
)


def fit_answer_residual_channel_codec(
    residual_fields,
    *,
    macro_channels: int,
    macro_rank: int,
    micro_rank: int,
) -> dict[str, Any]:
    """Fit a translation-shared channel map to hidden-answer residuals."""

    values = torch.as_tensor(residual_fields).detach()
    if values.ndim != 4 or not values.is_floating_point():
        raise ValueError("answer residuals must have shape [case, channel, x, y]")
    if not bool(torch.all(torch.isfinite(values)).item()):
        raise ValueError("answer residuals must be finite")
    channel_count = int(values.shape[1])
    boundary = int(macro_channels)
    if not 0 < boundary < channel_count:
        raise ValueError("macro channel boundary must be strict")
    selected_macro = int(macro_rank)
    selected_micro = int(micro_rank)
    if not 0 < selected_macro <= boundary:
        raise ValueError("macro channel rank is unavailable")
    if not 0 < selected_micro <= channel_count - boundary:
        raise ValueError("micro channel rank is unavailable")

    macro = _fit_channel_block(values[:, :boundary], selected_macro)
    micro = _fit_channel_block(values[:, boundary:], selected_micro)
    return {
        "schema": ANSWER_RESIDUAL_CHANNEL_CODEC_SCHEMA,
        "field_shape": [int(value) for value in values.shape[1:]],
        "macro_channels": boundary,
        "macro_rank": selected_macro,
        "micro_rank": selected_micro,
        "latent_channels": selected_macro + selected_micro,
        "macro": _portable_block(macro),
        "micro": _portable_block(micro),
        "translation_shared": True,
        "learned": False,
        "differentiable": True,
        "fit_target": "hidden_answer_residual_only",
        "answer_first_contract": {
            "primary_object": "continuous_hidden_solution_function",
            "equation_first_source_used": False,
            "induced_source_used": False,
            "equation_dynamics_used": False,
        },
        "diagnostics": {
            "macro_captured_variance_fraction": float(
                macro["retained_variance"] / macro["total_variance"]
            ),
            "micro_captured_variance_fraction": float(
                micro["retained_variance"] / micro["total_variance"]
            ),
            "macro_orthogonality_max_abs_error": float(
                macro["orthogonality_error"]
            ),
            "micro_orthogonality_max_abs_error": float(
                micro["orthogonality_error"]
            ),
            "fit_sample_count_per_block": int(
                values.shape[0] * values.shape[2] * values.shape[3]
            ),
        },
    }


def encode_answer_residual_channel_latent(
    residual_fields,
    codec: Mapping[str, Any],
) -> torch.Tensor:
    values = torch.as_tensor(residual_fields)
    _validate_codec(codec, values=values)
    boundary = int(codec["macro_channels"])
    macro = _encode_block(values[:, :boundary], codec["macro"])
    micro = _encode_block(values[:, boundary:], codec["micro"])
    return torch.cat((macro, micro), dim=1)


def decode_answer_residual_channel_latent(
    latent_fields,
    codec: Mapping[str, Any],
) -> torch.Tensor:
    values = torch.as_tensor(latent_fields)
    if values.ndim != 4 or not values.is_floating_point():
        raise ValueError("answer residual latent must have shape [case, channel, x, y]")
    expected_channels = int(codec.get("latent_channels", -1))
    expected_shape = tuple(int(value) for value in codec["field_shape"])
    if (
        int(values.shape[1]) != expected_channels
        or tuple(values.shape[2:]) != expected_shape[1:]
    ):
        raise ValueError("answer residual latent layout changed")
    macro_rank = int(codec["macro_rank"])
    macro = _decode_block(values[:, :macro_rank], codec["macro"])
    micro = _decode_block(values[:, macro_rank:], codec["micro"])
    output = torch.cat((macro, micro), dim=1)
    _validate_codec(codec, values=output)
    return output


def encode_answer_residual_channel_tangent(
    residual_tangent_fields,
    codec: Mapping[str, Any],
) -> torch.Tensor:
    values = torch.as_tensor(residual_tangent_fields)
    _validate_codec(codec, values=values)
    boundary = int(codec["macro_channels"])
    macro = _encode_tangent_block(values[:, :boundary], codec["macro"])
    micro = _encode_tangent_block(values[:, boundary:], codec["micro"])
    return torch.cat((macro, micro), dim=1)


def decode_answer_residual_channel_tangent(
    latent_tangent_fields,
    codec: Mapping[str, Any],
) -> torch.Tensor:
    values = torch.as_tensor(latent_tangent_fields)
    if values.ndim != 4 or not values.is_floating_point():
        raise ValueError("answer residual tangent latent must be a field")
    expected_shape = tuple(int(value) for value in codec["field_shape"])
    if (
        int(values.shape[1]) != int(codec["latent_channels"])
        or tuple(values.shape[2:]) != expected_shape[1:]
    ):
        raise ValueError("answer residual tangent latent layout changed")
    macro_rank = int(codec["macro_rank"])
    macro = _decode_tangent_block(values[:, :macro_rank], codec["macro"])
    micro = _decode_tangent_block(values[:, macro_rank:], codec["micro"])
    output = torch.cat((macro, micro), dim=1)
    _validate_codec(codec, values=output)
    return output


def slice_answer_residual_channel_codec(
    codec: Mapping[str, Any],
    *,
    macro_rank: int,
    micro_rank: int,
) -> dict[str, Any]:
    _validate_codec(codec)
    selected_macro = int(macro_rank)
    selected_micro = int(micro_rank)
    maximum_macro = int(codec["macro_rank"])
    maximum_micro = int(codec["micro_rank"])
    if not 0 < selected_macro <= maximum_macro:
        raise ValueError("requested macro channel rank is unavailable")
    if not 0 < selected_micro <= maximum_micro:
        raise ValueError("requested micro channel rank is unavailable")
    result = dict(codec)
    result["macro_rank"] = selected_macro
    result["micro_rank"] = selected_micro
    result["latent_channels"] = selected_macro + selected_micro
    result["macro"] = _slice_block(codec["macro"], selected_macro)
    result["micro"] = _slice_block(codec["micro"], selected_micro)
    macro_fraction = float(
        torch.as_tensor(result["macro"]["eigenvalues"]).sum()
        / float(result["macro"]["total_variance"])
    )
    micro_fraction = float(
        torch.as_tensor(result["micro"]["eigenvalues"]).sum()
        / float(result["micro"]["total_variance"])
    )
    result["diagnostics"] = {
        **dict(codec.get("diagnostics") or {}),
        "macro_captured_variance_fraction": macro_fraction,
        "micro_captured_variance_fraction": micro_fraction,
        "sliced_from_macro_rank": maximum_macro,
        "sliced_from_micro_rank": maximum_micro,
    }
    return result


def answer_residual_channel_codec_to_device(
    codec: Mapping[str, Any],
    device,
) -> dict[str, Any]:
    _validate_codec(codec)
    result = dict(codec)
    for group in ("macro", "micro"):
        block = dict(codec[group])
        for name in ("mean", "components", "scale"):
            block[name] = torch.as_tensor(
                block[name], dtype=torch.float32, device=device
            )
        result[group] = block
    return result


def _fit_channel_block(values: torch.Tensor, rank: int) -> dict[str, torch.Tensor]:
    samples = values.permute(0, 2, 3, 1).reshape(-1, int(values.shape[1]))
    mean = samples.mean(dim=0)
    centered = samples - mean
    covariance = centered.T @ centered / float(max(1, int(samples.shape[0]) - 1))
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min(0.0)
    components = eigenvectors[:, order[: int(rank)]].T
    retained = eigenvalues[: int(rank)]
    floor = (
        torch.finfo(values.dtype).eps
        * eigenvalues[0].clamp_min(1.0)
    )
    scale = torch.sqrt(retained.clamp_min(floor))
    identity = torch.eye(
        int(rank), dtype=values.dtype, device=values.device
    )
    orthogonality = torch.max(
        torch.abs(components @ components.T - identity)
    )
    return {
        "mean": mean,
        "components": components,
        "scale": scale,
        "eigenvalues": retained,
        "retained_variance": retained.sum(),
        "total_variance": eigenvalues.sum().clamp_min(floor),
        "orthogonality_error": orthogonality,
    }


def _portable_block(block: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    result = {
        name: torch.as_tensor(block[name]).detach().cpu()
        for name in ("mean", "components", "scale", "eigenvalues")
    }
    result["total_variance"] = float(block["total_variance"])
    return result


def _slice_block(block: Mapping[str, Any], rank: int) -> dict[str, Any]:
    result = {
        name: torch.as_tensor(block[name])[: int(rank)].clone()
        if name != "mean"
        else torch.as_tensor(block[name]).clone()
        for name in ("mean", "components", "scale", "eigenvalues")
    }
    result["total_variance"] = float(block["total_variance"])
    return result


def _encode_tangent_block(
    values: torch.Tensor,
    block: Mapping[str, Any],
) -> torch.Tensor:
    _mean, components, scale = _block_tensors(values, block)
    score = torch.einsum("bchw,rc->brhw", values, components)
    return score / scale[None, :, None, None]


def _decode_tangent_block(
    values: torch.Tensor,
    block: Mapping[str, Any],
) -> torch.Tensor:
    _mean, components, scale = _block_tensors(values, block)
    score = values * scale[None, :, None, None]
    return torch.einsum("brhw,rc->bchw", score, components)


def _encode_block(values: torch.Tensor, block: Mapping[str, Any]) -> torch.Tensor:
    mean, components, scale = _block_tensors(values, block)
    centered = values - mean[None, :, None, None]
    score = torch.einsum("bchw,rc->brhw", centered, components)
    return score / scale[None, :, None, None]


def _decode_block(values: torch.Tensor, block: Mapping[str, Any]) -> torch.Tensor:
    mean, components, scale = _block_tensors(values, block)
    score = values * scale[None, :, None, None]
    decoded = torch.einsum("brhw,rc->bchw", score, components)
    return decoded + mean[None, :, None, None]


def _block_tensors(values: torch.Tensor, block: Mapping[str, Any]):
    mean = torch.as_tensor(
        block["mean"], dtype=values.dtype, device=values.device
    )
    components = torch.as_tensor(
        block["components"], dtype=values.dtype, device=values.device
    )
    scale = torch.as_tensor(
        block["scale"], dtype=values.dtype, device=values.device
    )
    if components.ndim != 2:
        raise ValueError("channel codec components must be a matrix")
    rank, channels = int(components.shape[0]), int(components.shape[1])
    if tuple(mean.shape) != (channels,) or tuple(scale.shape) != (rank,):
        raise ValueError("channel codec block shape changed")
    if int(values.shape[1]) not in {channels, rank}:
        raise ValueError("channel codec values have the wrong channel count")
    if bool(torch.any(scale <= 0.0).item()):
        raise ValueError("channel codec scale must be positive")
    return mean, components, scale


def _validate_codec(
    codec: Mapping[str, Any],
    *,
    values: torch.Tensor | None = None,
) -> None:
    if codec.get("schema") != ANSWER_RESIDUAL_CHANNEL_CODEC_SCHEMA:
        raise ValueError("unexpected answer residual channel codec schema")
    field_shape = tuple(int(value) for value in codec["field_shape"])
    if len(field_shape) != 3 or min(field_shape) < 1:
        raise ValueError("answer residual field shape is invalid")
    boundary = int(codec["macro_channels"])
    macro_rank = int(codec["macro_rank"])
    micro_rank = int(codec["micro_rank"])
    if not 0 < boundary < field_shape[0]:
        raise ValueError("answer residual macro boundary is invalid")
    if not 0 < macro_rank <= boundary:
        raise ValueError("answer residual macro rank is invalid")
    if not 0 < micro_rank <= field_shape[0] - boundary:
        raise ValueError("answer residual micro rank is invalid")
    if int(codec["latent_channels"]) != macro_rank + micro_rank:
        raise ValueError("answer residual latent channel count changed")
    expected = {
        "macro": (macro_rank, boundary),
        "micro": (micro_rank, field_shape[0] - boundary),
    }
    for name, (rank, channels) in expected.items():
        block = codec[name]
        if tuple(torch.as_tensor(block["mean"]).shape) != (channels,):
            raise ValueError("answer residual channel mean shape changed")
        if tuple(torch.as_tensor(block["components"]).shape) != (
            rank,
            channels,
        ):
            raise ValueError("answer residual component shape changed")
        if tuple(torch.as_tensor(block["scale"]).shape) != (rank,):
            raise ValueError("answer residual scale shape changed")
    if values is not None:
        if values.ndim != 4 or tuple(values.shape[1:]) != field_shape:
            raise ValueError("answer residual field layout changed")
        if not bool(torch.all(torch.isfinite(values)).item()):
            raise ValueError("answer residual field must be finite")
