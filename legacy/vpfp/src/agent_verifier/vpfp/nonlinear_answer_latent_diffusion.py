from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from agent_verifier.vpfp.nonlinear_answer_autoencoder import (
    NONLINEAR_ANSWER_LATENT_EDM_CHECKPOINT_SCHEMA,
)
from agent_verifier.vpfp.shared_global_latent_diffusion import (
    VPFPSharedGlobalLatentEDM,
)


def project_zero_mean_local_latent(local_latent) -> torch.Tensor:
    values = torch.as_tensor(local_latent)
    if values.ndim != 4:
        raise ValueError("local answer latent must have shape [B,C,X,Y]")
    return values - torch.mean(values, dim=(2, 3), keepdim=True)


def save_nonlinear_answer_latent_edm_checkpoint(
    path: str | Path,
    *,
    model: VPFPSharedGlobalLatentEDM,
    autoencoder_checkpoint: str | Path,
    autoencoder_checkpoint_sha256: str,
    latent_normalization: Mapping[str, Any],
    condition_normalization: Mapping[str, Any],
    provenance: Mapping[str, Any],
    train_config: Mapping[str, Any],
    step: int,
    optimizer_state: Mapping[str, Any] | None = None,
    ema_state_dict: Mapping[str, Any] | None = None,
) -> None:
    if int(model.config.sampled_global_dim) < 1:
        raise ValueError("nonlinear answer latent EDM must sample global codes")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": NONLINEAR_ANSWER_LATENT_EDM_CHECKPOINT_SCHEMA,
            "model_config": model.config.to_dict(),
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema_state_dict,
            "autoencoder_checkpoint": str(
                Path(autoencoder_checkpoint).resolve()
            ),
            "autoencoder_checkpoint_sha256": str(
                autoencoder_checkpoint_sha256
            ),
            "latent_normalization": dict(latent_normalization),
            "condition_normalization": dict(condition_normalization),
            "provenance": dict(provenance),
            "answer_first_contract": {
                "primary_object": "continuous_hidden_solution_function",
                "generated_object": "complete_answer_latent",
                "training_target": "frozen_autoencoder_answer_codes_only",
                "equation_first_source_used": False,
                "induced_source_used": False,
                "equation_dynamics_used": False,
                "batch_coverage_loss_used": False,
                "autoencoder_frozen": True,
                "public_source_stage": "after_complete_answer_sampling_only",
            },
            "train_config": dict(train_config),
            "step": int(step),
            "optimizer_state_dict": optimizer_state,
        },
        destination,
    )


def load_nonlinear_answer_latent_edm_checkpoint(
    path: str | Path,
    *,
    device: str = "cpu",
    use_ema: bool = True,
) -> tuple[VPFPSharedGlobalLatentEDM, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("schema") != NONLINEAR_ANSWER_LATENT_EDM_CHECKPOINT_SCHEMA:
        raise ValueError("unexpected nonlinear answer latent EDM schema")
    contract = dict(payload.get("answer_first_contract") or {})
    if not bool(
        contract.get("primary_object")
        == "continuous_hidden_solution_function"
        and contract.get("generated_object") == "complete_answer_latent"
        and contract.get("training_target")
        == "frozen_autoencoder_answer_codes_only"
        and not contract.get("equation_first_source_used")
        and not contract.get("induced_source_used")
        and not contract.get("equation_dynamics_used")
        and not contract.get("batch_coverage_loss_used")
        and contract.get("autoencoder_frozen")
    ):
        raise ValueError("nonlinear answer latent EDM violates contract")
    model = VPFPSharedGlobalLatentEDM.build(payload["model_config"]).to(
        device=device
    )
    state = (
        payload.get("ema_state_dict")
        if use_ema and payload.get("ema_state_dict") is not None
        else payload["model_state_dict"]
    )
    model.load_state_dict(state)
    model.eval()
    return model, payload
