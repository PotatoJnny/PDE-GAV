from __future__ import annotations

import time as walltime
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np

from agent_verifier.vpfp.continuous_hermite import (
    VPFPContinuousHermiteConfig,
    VPFPContinuousHermiteField,
)
from agent_verifier.vpfp.continuous_v2 import (
    _real_fourier_half_modes,
    manufacture_standard_vpfp_source,
)
from agent_verifier.vpfp.continuous_flow_training import (
    prepare_structured_direct_projection_operators,
)
from agent_verifier.vpfp.hermite_density import total_degree_mode_pairs
from agent_verifier.vpfp.high_fidelity import (
    VPFPHighFidelityConfig,
    sample_continuous_equation_spec,
)
from agent_verifier.vpfp.operators import (
    VPFPGrid,
    VPFPTime,
    make_axes,
)
from agent_verifier.vpfp.torch_reference import (
    VPFPTorchSolverConfig,
    evaluate_torch_equation_parameter_batch,
    pack_torch_equation_specs,
    solve_torch_vpfp_batch,
)


@dataclass(frozen=True)
class VPFPOnlineGenerationConfig:
    batch_size: int = 64
    nx: int = 16
    ny: int = 16
    velocity_grid_size: int = 23
    nt: int = 9
    t1: float = 0.05
    velocity_max: float = 9.0
    internal_dt_max: float = 0.008
    dtype: str = "float32"
    cache_capacity_cases: int = 512
    high_fidelity_promotion_fraction: float = 0.02
    anchor_case_count: int = 64

    def validate(self) -> None:
        if min(
            self.batch_size,
            self.nx,
            self.ny,
            self.velocity_grid_size,
            self.nt,
            self.cache_capacity_cases,
            self.anchor_case_count,
        ) < 1:
            raise ValueError("online generation dimensions must be positive")
        if self.velocity_grid_size < 21:
            raise ValueError(
                "velocity grids below 21 failed the degree-14 Hermite audit"
            )
        if self.t1 <= 0.0 or self.velocity_max <= 0.0:
            raise ValueError("time and velocity domains must be positive")
        if self.internal_dt_max <= 0.0:
            raise ValueError("internal_dt_max must be positive")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")
        if not 0.0 <= self.high_fidelity_promotion_fraction <= 1.0:
            raise ValueError("promotion fraction must lie in [0,1]")

    def grid(self) -> VPFPGrid:
        self.validate()
        return VPFPGrid(
            nx=int(self.nx),
            ny=int(self.ny),
            nvx=int(self.velocity_grid_size),
            nvy=int(self.velocity_grid_size),
            v_bounds=(
                -float(self.velocity_max),
                float(self.velocity_max),
                -float(self.velocity_max),
                float(self.velocity_max),
            ),
        )

    def time(self) -> VPFPTime:
        self.validate()
        return VPFPTime(t0=0.0, t1=float(self.t1), nt=int(self.nt))

    def solver(self) -> VPFPTorchSolverConfig:
        self.validate()
        return VPFPTorchSolverConfig(
            internal_dt_max=float(self.internal_dt_max),
            dtype=str(self.dtype),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def equation_config_from_hybrid_report(
    report: dict[str, Any],
    *,
    generation_config: VPFPOnlineGenerationConfig,
) -> VPFPHighFidelityConfig:
    """Restore the equation distribution that causally produced a hybrid report."""

    payload = report.get("equation_config")
    if payload is None:
        config = VPFPHighFidelityConfig(
            internal_dt_max=float(generation_config.internal_dt_max)
        )
    else:
        config = VPFPHighFidelityConfig.from_dict(dict(payload))
    config.validate()
    if not np.isclose(
        float(config.internal_dt_max),
        float(generation_config.internal_dt_max),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(
            "hybrid equation and online-generation internal_dt_max disagree"
        )
    return config


class VPFPTorchStructuredProjector:
    """Project a transient grid solution directly to the canonical latent."""

    def __init__(
        self,
        decoder_config: VPFPContinuousHermiteConfig | dict[str, Any],
        *,
        grid: VPFPGrid,
        time: VPFPTime,
        device: str | Any,
        dtype: str = "float32",
        rho_feasible_fraction: float = 0.95,
    ):
        torch = _torch()
        self.torch = torch
        self.config = (
            decoder_config
            if isinstance(decoder_config, VPFPContinuousHermiteConfig)
            else VPFPContinuousHermiteConfig.from_dict(dict(decoder_config))
        )
        self.config.validate()
        grid.validate()
        time.validate()
        if not self.config.structured_spectral_latent:
            raise ValueError("online projection requires structured latents")
        if not self.config.structured_direct_coefficients:
            raise ValueError("online projection requires direct coefficients")
        if float(self.config.structured_residual_scale) != 0.0:
            raise ValueError("online projection requires zero residual scale")
        if not 0.0 < float(rho_feasible_fraction) < 1.0:
            raise ValueError("rho_feasible_fraction must lie in (0,1)")
        if dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")
        self.grid = grid
        self.time = time
        self.device = torch.device(device)
        self.dtype = torch.float32 if dtype == "float32" else torch.float64
        self.rho_feasible_fraction = float(rho_feasible_fraction)
        self.hermite_modes = total_degree_mode_pairs(
            self.config.hermite_total_degree
        )
        if len(self.hermite_modes) != int(
            self.config.structured_local_channels
        ):
            raise ValueError(
                "online direct projection needs one local channel per "
                "Hermite mode"
            )

        axes = make_axes(grid, time)
        proxy = SimpleNamespace(
            config=self.config,
            rho_half_modes=_real_fourier_half_modes(
                self.config.rho_max_mode_x,
                self.config.rho_max_mode_y,
            ),
            local_half_modes=_real_fourier_half_modes(
                self.config.structured_local_spatial_mode,
                self.config.structured_local_spatial_mode,
            ),
        )
        operator_case = {
            "t": np.asarray(axes["t"], dtype=np.float64),
            "x": np.asarray(axes["x"], dtype=np.float64),
            "y": np.asarray(axes["y"], dtype=np.float64),
        }
        operators = prepare_structured_direct_projection_operators(
            proxy, operator_case
        )
        self.cubic_time_controls = (
            self.config.structured_time_basis
            == "piecewise_cubic_hermite_v1"
        )
        self.local_pseudoinverse = self._tensor(
            operators["local_pseudoinverse"]
        )
        self.local_spatial_pseudoinverse = self._tensor(
            operators["local_spatial_pseudoinverse"]
        )
        self.rho_spatial_pseudoinverse = self._tensor(
            operators["rho_spatial_pseudoinverse"]
        )
        self.rho_time_pseudoinverse = self._tensor(
            operators["rho_time_pseudoinverse"]
        )
        self.tau = self._tensor(operators["tau"])
        self.operator_diagnostics = {
            key: float(operators[key])
            for key in (
                "local_design_condition_number",
                "rho_spatial_condition_number",
                "rho_time_condition_number",
            )
        }
        self.vx = self._tensor(axes["vx"])
        self.vy = self._tensor(axes["vy"])

    def project(
        self,
        solution: dict[str, Any],
        equation_conditions,
        *,
        include_projection_diagnostics: bool = False,
    ) -> dict[str, Any]:
        torch = self.torch
        started = walltime.perf_counter()
        rho = torch.as_tensor(
            solution["rho"], dtype=self.dtype, device=self.device
        )
        g = torch.as_tensor(
            solution["g"], dtype=self.dtype, device=self.device
        )
        conditions = torch.as_tensor(
            equation_conditions, dtype=self.dtype, device=self.device
        )
        expected_rho = (
            int(conditions.shape[0]),
            self.time.nt,
            self.grid.nx,
            self.grid.ny,
        )
        if tuple(rho.shape) != expected_rho:
            raise ValueError(
                f"rho shape {tuple(rho.shape)} does not match {expected_rho}"
            )
        if tuple(g.shape) != expected_rho + (
            self.grid.nvx,
            self.grid.nvy,
        ):
            raise ValueError("g shape does not match online projection grid")
        if tuple(conditions.shape) != (expected_rho[0], 2):
            raise ValueError("equation conditions must have shape [B,2]")

        coefficient_started = walltime.perf_counter()
        coefficients, design = self._project_hermite(
            g,
            conditions[:, 1],
        )
        self._synchronize()
        coefficient_seconds = walltime.perf_counter() - coefficient_started

        latent_started = walltime.perf_counter()
        latents, projection_report = self._project_structured_latent(
            rho, coefficients
        )
        self._synchronize()
        latent_seconds = walltime.perf_counter() - latent_started

        diagnostics: dict[str, Any] = {
            "backend": "torch_hermite_structured_direct_projection_v1",
            "coefficient_wall_time_seconds": float(coefficient_seconds),
            "structured_latent_wall_time_seconds": float(latent_seconds),
            "total_wall_time_seconds": float(
                walltime.perf_counter() - started
            ),
            "latent_dim": int(latents.shape[1]),
            "latent_bytes": int(latents.numel() * latents.element_size()),
            "hermite_coefficient_field_bytes": int(
                coefficients.numel() * coefficients.element_size()
            ),
            "operator_diagnostics": dict(self.operator_diagnostics),
            **projection_report,
        }
        if include_projection_diagnostics:
            singular_values = torch.linalg.svdvals(design)
            diagnostics["hermite_design_condition_number"] = (
                singular_values[:, 0] / singular_values[:, -1]
            ).detach().cpu().numpy()
            highest_degree = torch.as_tensor(
                [
                    first + second == int(self.config.hermite_total_degree)
                    for first, second in self.hermite_modes
                ],
                dtype=torch.bool,
                device=self.device,
            )
            diagnostics["highest_degree_energy_fraction"] = torch.sum(
                coefficients[..., highest_degree].square(), dim=-1
            ).detach().cpu().numpy()
        return {
            "schema": "vpfp_online_structured_latent_batch_v1",
            "latents": latents,
            "equation_conditions": conditions,
            "diagnostics": diagnostics,
        }

    def refine_time_slopes(
        self,
        latents,
        equation_conditions,
        *,
        source_points,
        target_source,
        steps: int = 4,
        learning_rate: float = 1.0e-2,
        regularization: float = 1.0e-3,
        maximum_hermite_degree: int = 4,
        maximum_spatial_mode: int = 1,
        maximum_update_ratio: float = 1.0,
        maximum_backtracking_steps: int = 8,
        point_chunk_size: int = 128,
    ) -> dict[str, Any]:
        """Fit only cubic-Hermite time slopes to the public source."""

        torch = self.torch
        if not self.cubic_time_controls:
            raise ValueError(
                "operator refinement requires piecewise cubic time controls"
            )
        if min(int(steps), int(point_chunk_size)) < 1:
            raise ValueError("refinement steps and chunk size must be positive")
        if int(maximum_backtracking_steps) < 0:
            raise ValueError("maximum_backtracking_steps must be nonnegative")
        if float(learning_rate) <= 0.0:
            raise ValueError("refinement learning_rate must be positive")
        if float(regularization) < 0.0:
            raise ValueError("refinement regularization must be nonnegative")
        if float(maximum_update_ratio) <= 0.0:
            raise ValueError("maximum_update_ratio must be positive")
        if not 0 <= int(maximum_hermite_degree) <= int(
            self.config.hermite_total_degree
        ):
            raise ValueError("maximum_hermite_degree is outside the model")
        if not 0 <= int(maximum_spatial_mode) <= int(
            self.config.structured_local_spatial_mode
        ):
            raise ValueError("maximum_spatial_mode is outside the model")

        base = torch.as_tensor(
            latents, dtype=self.dtype, device=self.device
        ).detach()
        conditions = torch.as_tensor(
            equation_conditions, dtype=self.dtype, device=self.device
        )
        points = torch.as_tensor(
            source_points, dtype=self.dtype, device=self.device
        )
        targets = torch.as_tensor(
            target_source, dtype=self.dtype, device=self.device
        )
        if base.ndim != 2 or int(base.shape[1]) != int(self.config.latent_dim):
            raise ValueError("refinement latents have the wrong shape")
        if tuple(conditions.shape) != (int(base.shape[0]), 2):
            raise ValueError("refinement conditions must have shape [B,2]")
        if points.ndim != 2 or tuple(points.shape[1:]) != (5,):
            raise ValueError("source_points must have shape [N,5]")
        if tuple(targets.shape) != (int(base.shape[0]), int(points.shape[0])):
            raise ValueError("target_source must have shape [B,N]")
        if not bool(torch.all(torch.isfinite(targets)).item()):
            raise ValueError("target_source must be finite")

        mask = self._time_slope_refinement_mask(
            maximum_hermite_degree=int(maximum_hermite_degree),
            maximum_spatial_mode=int(maximum_spatial_mode),
        )
        mask_values = mask.to(dtype=self.dtype)
        active_count = int(torch.sum(mask).item())
        if active_count < 1:
            raise RuntimeError("operator refinement has no active slopes")
        decoder = VPFPContinuousHermiteField.build(self.config).to(
            device=self.device
        ).eval()
        decoder.requires_grad_(False)
        refined_rows = []
        before_rows = []
        after_rows = []
        loss_traces = []
        update_ratios = []
        accepted_step_scales = []
        rejected_step_counts = []
        case_seconds = []
        started = walltime.perf_counter()
        for case_index in range(int(base.shape[0])):
            case_started = walltime.perf_counter()
            base_case = base[case_index]
            condition = conditions[case_index]
            target = targets[case_index]
            target_scale = torch.sqrt(torch.mean(target.square())).clamp_min(
                1.0e-4
            )
            before = self._manufactured_source_for_case(
                decoder,
                base_case,
                condition,
                points,
                point_chunk_size=int(point_chunk_size),
            )
            delta = torch.nn.Parameter(torch.zeros_like(base_case))
            optimizer = torch.optim.Adam(
                (delta,), lr=float(learning_rate)
            )
            reference_rms = torch.sqrt(
                torch.mean(base_case[mask].square())
            ).clamp_min(1.0e-3)
            initial_objective = float(
                torch.mean(((before - target) / target_scale).square()).cpu()
            )
            trace = [initial_objective]
            case_scales = []
            rejected_count = 0
            for _ in range(int(steps)):
                previous_delta = delta.detach().clone()
                current_objective = float(trace[-1])
                optimizer.zero_grad(set_to_none=True)
                point_count = int(points.shape[0])
                for chunk_start in range(
                    0, point_count, int(point_chunk_size)
                ):
                    chunk_stop = min(
                        chunk_start + int(point_chunk_size), point_count
                    )
                    candidate = base_case + mask_values * delta
                    values = manufacture_standard_vpfp_source(
                        decoder,
                        points[chunk_start:chunk_stop],
                        candidate,
                        equation_condition=condition,
                        create_graph=True,
                    )
                    normalized_residual = (
                        values["source"] - target[chunk_start:chunk_stop]
                    ) / target_scale
                    weight = float(chunk_stop - chunk_start) / float(
                        point_count
                    )
                    chunk_loss = weight * torch.mean(
                        normalized_residual.square()
                    )
                    chunk_loss.backward()
                regularity_loss = float(regularization) * torch.mean(
                    (delta[mask] / reference_rms).square()
                )
                regularity_loss.backward()
                optimizer.step()
                with torch.no_grad():
                    update_rms = torch.sqrt(
                        torch.mean(delta[mask].square())
                    )
                    maximum_rms = (
                        float(maximum_update_ratio) * reference_rms
                    )
                    if bool((update_rms > maximum_rms).item()):
                        delta.mul_(maximum_rms / update_rms)
                    proposed_delta = delta.detach().clone()

                accepted_scale = 1.0
                trial_objective = float("inf")
                for _backtrack in range(
                    int(maximum_backtracking_steps) + 1
                ):
                    trial_delta = previous_delta + accepted_scale * (
                        proposed_delta - previous_delta
                    )
                    with torch.no_grad():
                        delta.copy_(trial_delta)
                    trial_latent = base_case + mask_values * delta.detach()
                    trial_source = self._manufactured_source_for_case(
                        decoder,
                        trial_latent,
                        condition,
                        points,
                        point_chunk_size=int(point_chunk_size),
                    )
                    trial_data = torch.mean(
                        ((trial_source - target) / target_scale).square()
                    )
                    trial_regularity = float(regularization) * torch.mean(
                        (delta.detach()[mask] / reference_rms).square()
                    )
                    trial_objective = float(
                        (trial_data + trial_regularity).cpu()
                    )
                    if trial_objective <= current_objective:
                        break
                    accepted_scale *= 0.5
                else:
                    with torch.no_grad():
                        delta.copy_(previous_delta)
                    accepted_scale = 0.0
                    trial_objective = current_objective
                    rejected_count += 1
                trace.append(trial_objective)
                case_scales.append(accepted_scale)
            refined = (base_case + mask_values * delta).detach()
            after = self._manufactured_source_for_case(
                decoder,
                refined,
                condition,
                points,
                point_chunk_size=int(point_chunk_size),
            )
            update_ratio = torch.linalg.vector_norm(
                (refined - base_case)[mask]
            ) / torch.linalg.vector_norm(base_case[mask]).clamp_min(1.0e-6)
            refined_rows.append(refined)
            before_rows.append(before)
            after_rows.append(after)
            loss_traces.append(trace)
            update_ratios.append(float(update_ratio.detach().cpu()))
            accepted_step_scales.append(case_scales)
            rejected_step_counts.append(rejected_count)
            self._synchronize()
            case_seconds.append(walltime.perf_counter() - case_started)
        refined_latents = torch.stack(refined_rows)
        before_source = torch.stack(before_rows)
        after_source = torch.stack(after_rows)
        denominator = torch.linalg.vector_norm(targets, dim=1).clamp_min(
            torch.finfo(self.dtype).tiny
        )
        before_relative = torch.linalg.vector_norm(
            before_source - targets, dim=1
        ) / denominator
        after_relative = torch.linalg.vector_norm(
            after_source - targets, dim=1
        ) / denominator
        if not bool(torch.all(torch.isfinite(refined_latents)).item()):
            raise FloatingPointError("operator-refined latents are non-finite")
        return {
            "schema": "vpfp_operator_refined_time_slopes_v1",
            "latents": refined_latents,
            "diagnostics": {
                "active_slope_count": active_count,
                "maximum_hermite_degree": int(maximum_hermite_degree),
                "maximum_spatial_mode": int(maximum_spatial_mode),
                "steps": int(steps),
                "learning_rate": float(learning_rate),
                "regularization": float(regularization),
                "maximum_update_ratio": float(maximum_update_ratio),
                "maximum_backtracking_steps": int(
                    maximum_backtracking_steps
                ),
                "accepted_step_scale": np.asarray(
                    accepted_step_scales, dtype=np.float64
                ),
                "rejected_step_count": np.asarray(
                    rejected_step_counts, dtype=np.int64
                ),
                "source_relative_l2_before": (
                    before_relative.detach().cpu().numpy()
                ),
                "source_relative_l2_after": (
                    after_relative.detach().cpu().numpy()
                ),
                "slope_update_relative_l2": np.asarray(
                    update_ratios, dtype=np.float64
                ),
                "loss_trace": np.asarray(loss_traces, dtype=np.float64),
                "case_wall_time_seconds": np.asarray(
                    case_seconds, dtype=np.float64
                ),
                "total_wall_time_seconds": float(
                    walltime.perf_counter() - started
                ),
                "value_controls_changed": False,
            },
        }

    def _time_slope_refinement_mask(
        self,
        *,
        maximum_hermite_degree: int,
        maximum_spatial_mode: int,
    ):
        torch = self.torch
        layout = self.config.structured_latent_layout()
        control_count = int(self.config.structured_time_control_count)
        mask = torch.zeros(
            int(self.config.latent_dim),
            dtype=torch.bool,
            device=self.device,
        )
        rho_end = int(layout["rho_latent_dim"])
        rho_mask = mask[:rho_end].reshape(
            int(layout["rho_time_feature_count"]),
            int(layout["rho_mode_count"]),
            2,
        )
        active_rho_modes = torch.as_tensor(
            [
                max(abs(first), abs(second))
                <= int(maximum_spatial_mode)
                for first, second in _real_fourier_half_modes(
                    self.config.rho_max_mode_x,
                    self.config.rho_max_mode_y,
                )
            ],
            dtype=torch.bool,
            device=self.device,
        )
        rho_mask[control_count:] = active_rho_modes[None, :, None]
        local_start = rho_end + int(layout["kinetic_global_latent_dim"])
        local_mask = mask[local_start:].reshape(
            int(self.config.structured_local_channels),
            int(layout["local_time_feature_count"]),
            int(layout["local_spatial_feature_count"]),
        )
        active_channels = torch.as_tensor(
            [
                first + second <= int(maximum_hermite_degree)
                for first, second in self.hermite_modes
            ],
            dtype=torch.bool,
            device=self.device,
        )
        active_spatial = [True]
        for first, second in _real_fourier_half_modes(
            self.config.structured_local_spatial_mode,
            self.config.structured_local_spatial_mode,
        ):
            active = max(abs(first), abs(second)) <= int(
                maximum_spatial_mode
            )
            active_spatial.extend((active, active))
        active_spatial_tensor = torch.as_tensor(
            active_spatial,
            dtype=torch.bool,
            device=self.device,
        )
        local_mask[:, control_count:] = (
            active_channels[:, None, None]
            & active_spatial_tensor[None, None, :]
        )
        return mask

    def _manufactured_source_for_case(
        self,
        decoder,
        latent,
        condition,
        points,
        *,
        point_chunk_size: int,
    ):
        rows = []
        for start in range(0, int(points.shape[0]), int(point_chunk_size)):
            stop = min(start + int(point_chunk_size), int(points.shape[0]))
            values = manufacture_standard_vpfp_source(
                decoder,
                points[start:stop],
                latent,
                equation_condition=condition,
                create_graph=False,
            )
            rows.append(values["source"].detach())
        return self.torch.cat(rows)

    def _project_hermite(self, g, temperature):
        torch = self.torch
        scale = torch.sqrt(2.0 * temperature)
        x_functions = _batched_hermite_functions(
            torch,
            self.vx,
            max_degree=int(self.config.hermite_total_degree),
            scale=scale,
        )
        y_functions = _batched_hermite_functions(
            torch,
            self.vy,
            max_degree=int(self.config.hermite_total_degree),
            scale=scale,
        )
        basis = torch.stack(
            [
                x_functions[:, :, first, None]
                * y_functions[:, None, :, second]
                for first, second in self.hermite_modes
            ],
            dim=-1,
        ).reshape(g.shape[0], self.grid.nvx * self.grid.nvy, -1)
        sqrt_weight = np.sqrt(float(self.grid.velocity_cell_area))
        design = basis * sqrt_weight
        target = (
            torch.sqrt(torch.clamp(g, min=0.0))
            .reshape(g.shape[0], -1, self.grid.nvx * self.grid.nvy)
            * sqrt_weight
        )
        right_hand_side = torch.matmul(target, design)
        gram = torch.matmul(design.transpose(-1, -2), design)
        coefficients = torch.linalg.solve(
            gram,
            right_hand_side.transpose(-1, -2),
        ).transpose(-1, -2)
        norm = torch.linalg.vector_norm(
            coefficients, dim=-1, keepdim=True
        )
        fallback = torch.zeros_like(coefficients)
        fallback[..., 0] = 1.0
        coefficients = torch.where(
            norm > torch.finfo(coefficients.dtype).eps,
            coefficients / torch.clamp(
                norm, min=torch.finfo(coefficients.dtype).eps
            ),
            fallback,
        )
        return coefficients.reshape(
            g.shape[0],
            self.time.nt,
            self.grid.nx,
            self.grid.ny,
            len(self.hermite_modes),
        ), design

    def _project_structured_latent(self, rho, coefficients):
        torch = self.torch
        batch = int(rho.shape[0])
        layout = self.config.structured_latent_layout()
        if self.cubic_time_controls:
            coefficient_by_time = coefficients.reshape(
                batch,
                self.time.nt,
                self.grid.nx * self.grid.ny,
                len(self.hermite_modes),
            )
            local_solution_by_time = torch.einsum(
                "sn,btnc->btsc",
                self.local_spatial_pseudoinverse,
                coefficient_by_time,
            )
            local_value_controls = local_solution_by_time.permute(
                0, 3, 1, 2
            )
            local_slope_controls = _torch_normalized_time_slopes(
                torch,
                local_value_controls,
                self.tau,
                dim=2,
            )
            local_values = torch.cat(
                (local_value_controls, local_slope_controls), dim=2
            )
        else:
            coefficient_flat = coefficients.reshape(
                batch, -1, len(self.hermite_modes)
            )
            local_solution = torch.einsum(
                "pn,bnk->bpk",
                self.local_pseudoinverse,
                coefficient_flat,
            )
            local_values = local_solution.permute(0, 2, 1).reshape(
                batch,
                int(self.config.structured_local_channels),
                int(layout["local_time_feature_count"]),
                int(layout["local_spatial_feature_count"]),
            )

        background = float(self.config.equation.background_density)
        rho_anomaly = (rho - background).reshape(
            batch, self.time.nt, self.grid.nx * self.grid.ny
        )
        desired = torch.einsum(
            "qx,btx->btq",
            self.rho_spatial_pseudoinverse,
            rho_anomaly,
        )
        unconstrained_norm = _torch_rho_gauge_norm(
            torch,
            desired,
            gauge=str(self.config.rho_positivity_gauge),
        )
        coefficient_scale = float(self.config.rho_contrast) * background
        feasibility_scale = torch.clamp(
            (
                self.rho_feasible_fraction
                * coefficient_scale
                / torch.clamp(
                    unconstrained_norm,
                    min=torch.finfo(desired.dtype).tiny,
                )
            ),
            max=1.0,
        )
        desired = desired * feasibility_scale[..., None]
        coefficient_norm = _torch_rho_gauge_norm(
            torch,
            desired,
            gauge=str(self.config.rho_positivity_gauge),
        )
        raw_rho = desired / torch.clamp(
            coefficient_scale - coefficient_norm[..., None],
            min=torch.finfo(desired.dtype).eps,
        )
        if self.cubic_time_controls:
            rho_values = torch.cat(
                (
                    raw_rho,
                    _torch_normalized_time_slopes(
                        torch,
                        raw_rho,
                        self.tau,
                        dim=1,
                    ),
                ),
                dim=1,
            )
        else:
            rho_values = torch.einsum(
                "at,btq->baq",
                self.rho_time_pseudoinverse,
                raw_rho,
            )
        inactive_global = torch.zeros(
            (batch, int(layout["kinetic_global_latent_dim"])),
            dtype=self.dtype,
            device=self.device,
        )
        latents = torch.cat(
            (
                rho_values.reshape(batch, -1),
                inactive_global,
                local_values.reshape(batch, -1),
            ),
            dim=-1,
        ) / float(self.config.latent_input_scale)
        if int(latents.shape[1]) != int(self.config.latent_dim):
            raise RuntimeError("online structured latent size mismatch")
        return latents, {
            "rho_feasibility_scale_min": (
                torch.amin(feasibility_scale, dim=1)
                .detach()
                .cpu()
                .numpy()
            ),
            "rho_clipped_time_count": (
                torch.sum(feasibility_scale < 1.0, dim=1)
                .detach()
                .cpu()
                .numpy()
            ),
        }

    def _tensor(self, values):
        return self.torch.as_tensor(
            values, dtype=self.dtype, device=self.device
        )

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)


class VPFPOnlineBatchGenerator:
    """Generate replayable equation-first batches and retain only latents."""

    def __init__(
        self,
        *,
        decoder_config: VPFPContinuousHermiteConfig | dict[str, Any],
        generation_config: VPFPOnlineGenerationConfig | None = None,
        equation_config: VPFPHighFidelityConfig | None = None,
        device: str | Any = "cuda",
    ):
        self.config = generation_config or VPFPOnlineGenerationConfig()
        self.config.validate()
        self.equation_config = equation_config or VPFPHighFidelityConfig(
            internal_dt_max=float(self.config.internal_dt_max)
        )
        self.equation_config.validate()
        if not np.isclose(
            float(self.equation_config.internal_dt_max),
            float(self.config.internal_dt_max),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                "equation and generation internal_dt_max must agree"
            )
        self.grid = self.config.grid()
        self.time = self.config.time()
        self.solver_config = self.config.solver()
        self.device = _torch().device(device)
        self.projector = VPFPTorchStructuredProjector(
            decoder_config,
            grid=self.grid,
            time=self.time,
            device=self.device,
            dtype=self.config.dtype,
        )

    def generate(self, seeds: Sequence[int]) -> dict[str, Any]:
        requested_seeds = np.asarray(seeds, dtype=np.int64)
        if requested_seeds.ndim != 1 or requested_seeds.size < 1:
            raise ValueError("seeds must be a nonempty one-dimensional list")
        started = walltime.perf_counter()
        sampling_started = walltime.perf_counter()
        specs = [
            sample_continuous_equation_spec(
                seed=int(seed),
                config=self.equation_config,
                defer_rho_log_partition=True,
            )
            for seed in requested_seeds
        ]
        sampling_seconds = walltime.perf_counter() - sampling_started
        parameters = pack_torch_equation_specs(
            specs, dtype=self.config.dtype
        )
        prepared = evaluate_torch_equation_parameter_batch(
            parameters,
            grid=self.grid,
            time=self.time,
            device=self.device,
        )
        solution = solve_torch_vpfp_batch(
            prepared,
            grid=self.grid,
            time=self.time,
            config=self.solver_config,
            device=self.device,
        )
        torch = _torch()
        conditions = torch.stack(
            (
                torch.as_tensor(
                    prepared["collision_nu"],
                    dtype=self.projector.dtype,
                    device=self.device,
                ),
                torch.as_tensor(
                    prepared["temperature"],
                    dtype=self.projector.dtype,
                    device=self.device,
                ),
            ),
            dim=-1,
        )
        projected = self.projector.project(solution, conditions)
        raw_trajectory_bytes = int(
            solution["f"].numel() * solution["f"].element_size()
        )
        diagnostics = {
            "schema": "vpfp_online_generation_diagnostics_v1",
            "batch_size": int(requested_seeds.size),
            "equation_spec_sampling_wall_time_seconds": float(
                sampling_seconds
            ),
            "parameter_pack_wall_time_seconds": float(
                parameters["parameter_pack_wall_time_seconds"]
            ),
            "equation_evaluation_wall_time_seconds": float(
                prepared["equation_evaluation_wall_time_seconds"]
            ),
            "solver": solution["diagnostics"],
            "projection": projected["diagnostics"],
            "raw_f_trajectory_bytes_discarded": raw_trajectory_bytes,
            "retained_latent_bytes": int(
                projected["latents"].numel()
                * projected["latents"].element_size()
            ),
            "total_wall_time_seconds": float(
                walltime.perf_counter() - started
            ),
        }
        return {
            "schema": "vpfp_online_training_batch_v1",
            "latents": projected["latents"],
            "equation_conditions": projected["equation_conditions"],
            "equation_seeds": requested_seeds,
            "promotion_mask": _promotion_mask(
                requested_seeds,
                fraction=float(
                    self.config.high_fidelity_promotion_fraction
                ),
            ),
            "diagnostics": diagnostics,
        }


class VPFPCompactLatentRingBuffer:
    """Fixed-size CPU cache for online latents, conditions, and replay seeds."""

    def __init__(
        self,
        *,
        capacity_cases: int,
        latent_dim: int,
        condition_dim: int = 2,
        pin_memory: bool = False,
    ):
        torch = _torch()
        if min(capacity_cases, latent_dim, condition_dim) < 1:
            raise ValueError("ring-buffer dimensions must be positive")
        self.capacity = int(capacity_cases)
        self.latent_dim = int(latent_dim)
        self.condition_dim = int(condition_dim)
        self.latents = torch.empty(
            (self.capacity, self.latent_dim),
            dtype=torch.float32,
            pin_memory=bool(pin_memory),
        )
        self.conditions = torch.empty(
            (self.capacity, self.condition_dim),
            dtype=torch.float32,
            pin_memory=bool(pin_memory),
        )
        self.seeds = np.empty(self.capacity, dtype=np.int64)
        self.promoted = np.empty(self.capacity, dtype=bool)
        self.size = 0
        self.cursor = 0

    @property
    def retained_bytes(self) -> int:
        return int(
            self.latents.numel() * self.latents.element_size()
            + self.conditions.numel() * self.conditions.element_size()
            + self.seeds.nbytes
            + self.promoted.nbytes
        )

    def add(self, batch: dict[str, Any]) -> None:
        torch = _torch()
        latents = torch.as_tensor(
            batch["latents"], dtype=torch.float32
        ).detach().cpu()
        conditions = torch.as_tensor(
            batch["equation_conditions"], dtype=torch.float32
        ).detach().cpu()
        seeds = np.asarray(batch["equation_seeds"], dtype=np.int64)
        promoted = np.asarray(batch["promotion_mask"], dtype=bool)
        count = int(latents.shape[0])
        if tuple(latents.shape) != (count, self.latent_dim):
            raise ValueError("latent batch does not match ring-buffer shape")
        if tuple(conditions.shape) != (count, self.condition_dim):
            raise ValueError("condition batch does not match ring-buffer shape")
        if seeds.shape != (count,) or promoted.shape != (count,):
            raise ValueError("seed and promotion arrays must match batch")
        if count > self.capacity:
            latents = latents[-self.capacity :]
            conditions = conditions[-self.capacity :]
            seeds = seeds[-self.capacity :]
            promoted = promoted[-self.capacity :]
            count = self.capacity
        indices = (
            np.arange(count, dtype=np.int64) + int(self.cursor)
        ) % self.capacity
        index_tensor = torch.as_tensor(indices, dtype=torch.long)
        self.latents[index_tensor] = latents
        self.conditions[index_tensor] = conditions
        self.seeds[indices] = seeds
        self.promoted[indices] = promoted
        self.cursor = int((self.cursor + count) % self.capacity)
        self.size = min(self.capacity, self.size + count)

    def sample(
        self,
        batch_size: int,
        *,
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        if self.size < 1:
            raise ValueError("cannot sample an empty latent cache")
        if int(batch_size) < 1:
            raise ValueError("batch_size must be positive")
        indices = rng.integers(0, self.size, size=int(batch_size))
        return {
            "latents": self.latents[indices],
            "equation_conditions": self.conditions[indices],
            "equation_seeds": self.seeds[indices].copy(),
            "promotion_mask": self.promoted[indices].copy(),
        }



def fit_translation_invariant_structured_normalization(
    values,
    config,
    *,
    min_std: float = 1.0e-4,
) -> dict[str, Any]:
    """Fit normalization tied across torus-translation coefficient pairs."""

    array = np.asarray(
        _torch().as_tensor(values).detach().cpu(),
        dtype=np.float64,
    )
    if (
        array.ndim != 2
        or int(array.shape[1]) != int(config.active_latent_dim)
    ):
        raise ValueError("structured normalization shape mismatch")
    mean = np.mean(array, axis=0)
    std = np.std(array, axis=0)

    rho_values = array[:, : config.rho_latent_dim].reshape(
        array.shape[0],
        config.rho_time_count,
        config.rho_mode_count,
        2,
    )
    rho_mean = mean[: config.rho_latent_dim].reshape(
        config.rho_time_count, config.rho_mode_count, 2
    )
    rho_std = std[: config.rho_latent_dim].reshape(
        config.rho_time_count, config.rho_mode_count, 2
    )
    rho_mean.fill(0.0)
    rho_pair_std = np.sqrt(np.mean(rho_values**2, axis=(0, 3)))
    rho_std[...] = rho_pair_std[..., None]

    local_values = array[:, config.rho_latent_dim :].reshape(
        array.shape[0],
        config.local_channels,
        config.local_time_count,
        config.local_spatial_feature_count,
    )
    local_mean = mean[config.rho_latent_dim :].reshape(
        config.local_channels,
        config.local_time_count,
        config.local_spatial_feature_count,
    )
    local_std = std[config.rho_latent_dim :].reshape(
        config.local_channels,
        config.local_time_count,
        config.local_spatial_feature_count,
    )
    local_pairs = local_values[..., 1:].reshape(
        array.shape[0],
        config.local_channels,
        config.local_time_count,
        -1,
        2,
    )
    local_mean[..., 1:] = 0.0
    local_pair_std = np.sqrt(np.mean(local_pairs**2, axis=(0, 4)))
    local_std[..., 1:] = np.repeat(
        local_pair_std[..., None], 2, axis=-1
    ).reshape(local_std[..., 1:].shape)
    std = np.maximum(std, float(min_std))
    return {
        "schema": "vpfp_translation_invariant_latent_normalization_v1",
        "mean": mean.tolist(),
        "std": std.tolist(),
        "min_std": float(min_std),
        "group_action": "joint_torus_translation_phase_rotation",
        "sample_count": int(array.shape[0]),
    }


def _batched_hermite_functions(
    torch,
    values,
    *,
    max_degree: int,
    scale,
):
    points = torch.as_tensor(values)
    scales = torch.as_tensor(
        scale, dtype=points.dtype, device=points.device
    )
    xi = points[None, :] / scales[:, None]
    output = [
        torch.pi ** (-0.25)
        / torch.sqrt(scales[:, None])
        * torch.exp(-0.5 * xi.square())
    ]
    if int(max_degree) >= 1:
        output.append(np.sqrt(2.0) * xi * output[0])
    for order in range(1, int(max_degree)):
        output.append(
            np.sqrt(2.0 / float(order + 1))
            * xi
            * output[order]
            - np.sqrt(float(order) / float(order + 1))
            * output[order - 1]
        )
    return torch.stack(output, dim=-1)


def _torch_normalized_time_slopes(torch, values, tau, *, dim: int):
    controls = torch.movedim(values, dim, 0)
    coordinates = torch.as_tensor(
        tau, dtype=controls.dtype, device=controls.device
    )
    if int(controls.shape[0]) != int(coordinates.numel()):
        raise ValueError("time controls and normalized coordinates disagree")
    slopes = torch.empty_like(controls)
    slopes[0] = (controls[1] - controls[0]) / (
        coordinates[1] - coordinates[0]
    )
    slopes[-1] = (controls[-1] - controls[-2]) / (
        coordinates[-1] - coordinates[-2]
    )
    if int(coordinates.numel()) > 2:
        denominator = coordinates[2:] - coordinates[:-2]
        shape = (int(denominator.numel()),) + (1,) * (controls.ndim - 1)
        slopes[1:-1] = (controls[2:] - controls[:-2]) / denominator.reshape(
            shape
        )
    return torch.movedim(slopes, 0, dim)


def _torch_rho_gauge_norm(torch, coefficients, *, gauge: str):
    if int(coefficients.shape[-1]) % 2:
        raise ValueError("rho Fourier coefficients need cosine/sine pairs")
    if gauge == "coordinate_l1_v1":
        return torch.sum(torch.abs(coefficients), dim=-1)
    if gauge == "phase_invariant_group_l1_v2":
        pairs = coefficients.reshape(*coefficients.shape[:-1], -1, 2)
        return torch.sum(
            torch.linalg.vector_norm(pairs, dim=-1), dim=-1
        )
    raise ValueError(f"unsupported rho positivity gauge: {gauge!r}")


def _promotion_mask(seeds: np.ndarray, *, fraction: float) -> np.ndarray:
    if fraction <= 0.0:
        return np.zeros(seeds.shape, dtype=bool)
    if fraction >= 1.0:
        return np.ones(seeds.shape, dtype=bool)
    modulus = 1_000_003
    threshold = int(round(float(fraction) * modulus))
    hashed = (
        np.asarray(seeds, dtype=np.uint64) * np.uint64(1_103_515_245)
        + np.uint64(12_345)
    ) % np.uint64(modulus)
    return np.asarray(hashed < threshold, dtype=bool)


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for online VPFP generation") from exc
    return torch
