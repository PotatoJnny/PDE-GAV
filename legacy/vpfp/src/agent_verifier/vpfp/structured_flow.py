from __future__ import annotations

import numpy as np

from agent_verifier.vpfp.structured_diffusion import split_active_latent


def minibatch_ot_noise_pairing(clean, noise, config):
    """Pair Gaussian rows to data rows with a block-balanced minibatch OT cost."""

    torch = _torch()
    clean_values = torch.as_tensor(clean)
    noise_values = torch.as_tensor(
        noise,
        dtype=clean_values.dtype,
        device=clean_values.device,
    )
    if clean_values.shape != noise_values.shape or clean_values.ndim != 2:
        raise ValueError("minibatch OT expects matching latent matrices")
    if int(clean_values.shape[1]) != int(config.active_latent_dim):
        raise ValueError("minibatch OT latent shape does not match config")
    batch = int(clean_values.shape[0])
    if batch < 1:
        raise ValueError("minibatch OT requires a nonempty batch")
    if batch == 1:
        return noise_values, torch.zeros(
            1, dtype=torch.long, device=clean_values.device
        )

    clean_rho, clean_local = split_active_latent(clean_values, config)
    noise_rho, noise_local = split_active_latent(noise_values, config)
    clean_rho = clean_rho.reshape(batch, -1) / np.sqrt(
        float(config.rho_latent_dim)
    )
    noise_rho = noise_rho.reshape(batch, -1) / np.sqrt(
        float(config.rho_latent_dim)
    )
    clean_local = clean_local.reshape(batch, -1) / np.sqrt(
        float(config.local_latent_dim)
    )
    noise_local = noise_local.reshape(batch, -1) / np.sqrt(
        float(config.local_latent_dim)
    )
    cost = 0.5 * torch.cdist(clean_rho, noise_rho).square()
    cost = cost + 0.5 * torch.cdist(clean_local, noise_local).square()

    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:
        raise RuntimeError("minibatch OT flow matching requires scipy") from exc
    rows, columns = linear_sum_assignment(
        cost.detach().cpu().numpy()
    )
    if not np.array_equal(rows, np.arange(batch)):
        raise RuntimeError("minibatch OT assignment did not cover every data row")
    permutation = torch.as_tensor(
        columns,
        dtype=torch.long,
        device=clean_values.device,
    )
    return noise_values.index_select(0, permutation), permutation


def structured_flow_matching_loss(
    model,
    clean_latent,
    conditions,
    *,
    path_times=None,
    noise=None,
    use_minibatch_ot: bool = True,
):
    """Regress the rectified-flow velocity from Gaussian noise to VPFP data."""

    torch = _torch()
    parameter = next(model.parameters())
    clean = torch.as_tensor(
        clean_latent,
        dtype=parameter.dtype,
        device=parameter.device,
    )
    condition = torch.as_tensor(
        conditions,
        dtype=parameter.dtype,
        device=parameter.device,
    )
    if clean.ndim != 2 or int(clean.shape[1]) != int(
        model.config.active_latent_dim
    ):
        raise ValueError("flow-matching clean latent shape is invalid")
    if condition.shape != (clean.shape[0], model.config.condition_dim):
        raise ValueError("flow-matching condition shape is invalid")
    source = (
        torch.randn_like(clean)
        if noise is None
        else torch.as_tensor(
            noise,
            dtype=parameter.dtype,
            device=parameter.device,
        )
    )
    if source.shape != clean.shape:
        raise ValueError("flow-matching noise shape is invalid")
    if use_minibatch_ot:
        source, permutation = minibatch_ot_noise_pairing(
            clean, source, model.config
        )
    else:
        permutation = torch.arange(
            int(clean.shape[0]),
            dtype=torch.long,
            device=parameter.device,
        )
    if path_times is None:
        times = torch.rand(
            int(clean.shape[0]),
            dtype=parameter.dtype,
            device=parameter.device,
        )
    else:
        times = torch.as_tensor(
            path_times,
            dtype=parameter.dtype,
            device=parameter.device,
        ).reshape(-1)
    if times.shape != (clean.shape[0],):
        raise ValueError("flow-matching path times must match the batch")
    if bool(torch.any((times < 0.0) | (times > 1.0)).item()):
        raise ValueError("flow-matching path times must lie in [0,1]")

    interpolation = (
        (1.0 - times[:, None]) * source
        + times[:, None] * clean
    )
    model_times = times * float(max(1, int(model.config.timesteps) - 1))
    prediction = model(interpolation, model_times, condition)
    target = clean - source
    predicted_rho, predicted_local = split_active_latent(
        prediction, model.config
    )
    target_rho, target_local = split_active_latent(target, model.config)
    rho_mse = torch.mean((predicted_rho - target_rho).square())
    local_mse = torch.mean((predicted_local - target_local).square())
    loss = 0.5 * (rho_mse + local_mse)
    return loss, {
        "loss": loss,
        "rho_velocity_mse": rho_mse,
        "local_velocity_mse": local_mse,
        "path_time_mean": times.mean(),
        "ot_permutation_fixed_fraction": (
            permutation
            == torch.arange(
                int(clean.shape[0]),
                dtype=torch.long,
                device=parameter.device,
            )
        ).to(dtype=parameter.dtype).mean(),
    }


def sample_structured_flow(
    model,
    conditions,
    *,
    initial_noise=None,
    sample_steps: int = 100,
    solver: str = "heun",
):
    """Integrate a learned VPFP rectified-flow ODE from noise to data."""

    torch = _torch()
    parameter = next(model.parameters())
    condition = torch.as_tensor(
        conditions,
        dtype=parameter.dtype,
        device=parameter.device,
    )
    if condition.ndim != 2 or int(condition.shape[1]) != int(
        model.config.condition_dim
    ):
        raise ValueError("flow sample conditions have an invalid shape")
    batch = int(condition.shape[0])
    if int(sample_steps) < 1:
        raise ValueError("flow sample_steps must be positive")
    if solver not in {"euler", "heun"}:
        raise ValueError("flow solver must be euler or heun")
    if initial_noise is None:
        current = torch.randn(
            (batch, int(model.config.active_latent_dim)),
            dtype=parameter.dtype,
            device=parameter.device,
        )
    else:
        current = torch.as_tensor(
            initial_noise,
            dtype=parameter.dtype,
            device=parameter.device,
        ).clone()
    expected = (batch, int(model.config.active_latent_dim))
    if current.shape != expected:
        raise ValueError("flow initial noise has an invalid shape")

    step_size = 1.0 / float(sample_steps)
    with torch.no_grad():
        for index in range(int(sample_steps)):
            time_value = float(index) * step_size
            model_time = torch.full(
                (batch,),
                time_value * float(max(1, int(model.config.timesteps) - 1)),
                dtype=parameter.dtype,
                device=parameter.device,
            )
            velocity = model(current, model_time, condition)
            if solver == "euler":
                current = current + step_size * velocity
                continue
            proposal = current + step_size * velocity
            next_time = torch.full(
                (batch,),
                float(index + 1)
                * step_size
                * float(max(1, int(model.config.timesteps) - 1)),
                dtype=parameter.dtype,
                device=parameter.device,
            )
            next_velocity = model(proposal, next_time, condition)
            current = current + 0.5 * step_size * (
                velocity + next_velocity
            )
    return current


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for structured flow matching") from exc
    return torch
