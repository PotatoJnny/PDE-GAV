from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from agent_verifier.vpfp.continuous_diffusion import (
    denormalize_vectors,
    diffusion_schedule,
    normalize_vectors,
    q_sample,
)
from agent_verifier.vpfp.continuous_flow_training import (
    load_hermite_autoencoder_checkpoint,
)
from agent_verifier.vpfp.structured_bottleneck import (
    decode_shared_local_time_channel,
    encode_shared_local_time_channel,
    load_shared_local_bottleneck_contract,
)
from agent_verifier.vpfp.structured_diffusion import (
    active_latent_from_full,
    full_latent_from_active,
    load_structured_diffusion_checkpoint,
    sample_structured_latents,
)
from agent_verifier.vpfp.structured_reward_guidance import (
    load_structured_reward_checkpoint,
    make_structured_reward_guidance,
    project_guided_local_trust_region,
)


POLICY_BASE = "base"
POLICY_ABSOLUTE = "absolute_selected"
POLICY_GUIDED = "guided"
SUPPORTED_POLICIES = {POLICY_BASE, POLICY_ABSOLUTE, POLICY_GUIDED}


@dataclass(frozen=True)
class VPFPStructuredGuidedGenerationConfig:
    candidate_pool_size: int = 8
    sample_steps: int = 21
    start_timestep: int = 20
    eta: float = 0.0
    batch_size: int = 32
    centered_local_guidance_enabled: bool = True
    centered_guidance_scale: float = 1.0
    absolute_local_guidance_scale: float = 0.0
    maximum_gradient_norm: float = 1.0
    maximum_local_rms_delta: float = 0.03
    maximum_coordinate_delta: float = 0.25
    freeze_rho: bool = True
    anchor_assignment: str = "unique_min_cost"
    representation: str = "vpfp_structured_guided_generator_v1"

    def validate(self) -> None:
        if not 2 <= int(self.candidate_pool_size) <= 16:
            raise ValueError("candidate pool size must lie in [2, 16]")
        if min(
            int(self.sample_steps),
            int(self.batch_size),
            int(self.start_timestep) + 1,
        ) < 1:
            raise ValueError("structured guided sampling counts must be positive")
        if float(self.eta) != 0.0:
            raise ValueError("the admitted structured sampler requires deterministic DDIM")
        for name in (
            "maximum_gradient_norm",
            "maximum_local_rms_delta",
            "maximum_coordinate_delta",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if bool(self.centered_local_guidance_enabled):
            if float(self.centered_guidance_scale) <= 0.0:
                raise ValueError(
                    "centered guidance scale must be positive when enabled"
                )
        elif float(self.centered_guidance_scale) != 0.0:
            raise ValueError(
                "centered guidance scale must be zero when guidance is disabled"
            )
        if not bool(self.freeze_rho):
            raise ValueError("the admitted structured policy freezes rho coordinates")
        if self.anchor_assignment != "unique_min_cost":
            raise ValueError("the admitted sampler requires unique anchor assignment")
        if self.representation != "vpfp_structured_guided_generator_v1":
            raise ValueError("unsupported structured guided representation")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VPFPStructuredGuidedGenerator:
    """Frozen diffusion plus solver-bound reward guidance.

    The object returns hidden decoder latents to controller-owned code. Public
    services must expose only materialized questions, numerical answers, or
    point-oracle values built from those latents.
    """

    def __init__(
        self,
        *,
        guidance_report: str | Path,
        fresh_guidance_audit: str | Path,
        latent_artifact: str | Path,
        config: VPFPStructuredGuidedGenerationConfig,
        device: str = "cuda",
    ) -> None:
        config.validate()
        self.config = config
        self.guidance_report_path = Path(guidance_report).resolve()
        self.fresh_audit_path = Path(fresh_guidance_audit).resolve()
        self.latent_artifact_path = Path(latent_artifact).resolve()
        self.guidance_report = _read_json(self.guidance_report_path)
        self.fresh_audit = _read_json(self.fresh_audit_path)
        if self.guidance_report.get("schema") != (
            "vpfp_structured_solver_reward_guidance_report_v1"
        ) or not bool(
            self.guidance_report.get("ready_for_fresh_actual_solver_audit")
        ):
            raise ValueError("structured reward regression gate did not pass")
        if self.fresh_audit.get("schema") != (
            "vpfp_structured_fresh_solver_guidance_audit_v1"
        ) or not bool(self.fresh_audit.get("ready_for_agent_iteration")):
            raise ValueError("fresh structured actual-solver gate did not pass")
        expected_route = (
            "absolute_candidate_selection_plus_centered_local_guidance"
            if bool(config.centered_local_guidance_enabled)
            else "absolute_candidate_selection_only"
        )
        if str(self.fresh_audit.get("admitted_policy_route")) != expected_route:
            raise ValueError("configured policy differs from the admitted fresh audit")
        if bool(config.centered_local_guidance_enabled) and not bool(
            self.fresh_audit.get("centered_local_guidance_gate_passed")
        ):
            raise ValueError("centered local guidance was not admitted")
        if not bool(config.centered_local_guidance_enabled) and not bool(
            self.fresh_audit.get("absolute_selection_gate_passed")
        ):
            raise ValueError("absolute candidate selection was not admitted")
        if Path(self.fresh_audit["guidance_report"]).resolve() != (
            self.guidance_report_path
        ):
            raise ValueError("fresh audit is bound to another guidance report")
        if str(self.fresh_audit.get("target_solver_sha256")) != str(
            self.guidance_report.get("target_solver_sha256")
        ):
            raise ValueError("structured reports target different solvers")

        torch = _torch()
        self.torch = torch
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("structured guided generation requires CUDA")
        torch.set_float32_matmul_precision("high")
        self.diffusion, self.diffusion_payload = (
            load_structured_diffusion_checkpoint(
                self.guidance_report["diffusion_checkpoint"],
                device=str(self.device),
                use_ema=True,
            )
        )
        self.reward_model, self.reward_payload = (
            load_structured_reward_checkpoint(
                self.guidance_report["checkpoint"], device=str(self.device)
            )
        )
        self.bottleneck = load_shared_local_bottleneck_contract(
            self.diffusion_payload
        )
        if self.bottleneck is None:
            raise ValueError("structured guided generation requires the PCA bottleneck")
        for model in (self.diffusion, self.reward_model):
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        reward_train = dict(self.reward_payload.get("train_config") or {})
        target_sha = str(self.guidance_report["target_solver_sha256"])
        if str(reward_train.get("target_solver_sha256")) != target_sha:
            raise ValueError("reward checkpoint targets another solver")
        self.target_solver_sha256 = target_sha

        fresh_sample_report = _read_json(
            Path(self.fresh_audit["fresh_sample_report"]).resolve()
        )
        initialization = dict(fresh_sample_report.get("initialization") or {})
        if not (
            initialization.get("mode") == "train_manifold_bank"
            and initialization.get("anchor_assignment") == "unique_min_cost"
            and bool(initialization.get("preserve_rho_coordinates"))
        ):
            raise ValueError("fresh audit did not use the admitted manifold policy")
        if int(initialization.get("start_timestep", -1)) != int(
            config.start_timestep
        ):
            raise ValueError("configured start timestep differs from the admitted audit")
        if int(fresh_sample_report.get("sample_steps", -1)) != int(
            config.sample_steps
        ) or float(fresh_sample_report.get("eta", -1.0)) != float(config.eta):
            raise ValueError("configured reverse sampler differs from the admitted audit")
        if Path(fresh_sample_report["latent_artifact"]).resolve() != (
            self.latent_artifact_path
        ):
            raise ValueError("latent artifact differs from the admitted fresh audit")

        with np.load(self.latent_artifact_path, allow_pickle=False) as payload:
            self.full_reference = np.asarray(
                payload["fitted_latents"], dtype=np.float32
            )
            self.conditions = np.asarray(
                payload["equation_conditions"], dtype=np.float32
            )
            self.train_indices = np.asarray(payload["train_indices"], dtype=np.int64)
            self.validation_indices = np.asarray(
                payload["validation_indices"], dtype=np.int64
            )
        self._validate_reference_arrays()
        if int(config.start_timestep) >= int(self.diffusion.config.timesteps):
            raise ValueError("guided start timestep exceeds diffusion schedule")
        self.schedule = diffusion_schedule(
            self.diffusion.config,
            device="cpu",
            dtype=next(self.diffusion.parameters()).dtype,
        )
        self.guidance = None
        if bool(config.centered_local_guidance_enabled):
            self.guidance = make_structured_reward_guidance(
                self.reward_model,
                absolute_scale=float(config.absolute_local_guidance_scale),
                centered_scale=float(config.centered_guidance_scale),
                rho_latent_dim=int(self.diffusion.config.rho_latent_dim),
                max_gradient_norm=float(config.maximum_gradient_norm),
                freeze_rho=True,
            )
        _encoder, self.decoder, _payload = load_hermite_autoencoder_checkpoint(
            self.diffusion_payload["decoder_checkpoint"],
            device=str(self.device),
        )
        self.decoder.eval()
        for parameter in self.decoder.parameters():
            parameter.requires_grad_(False)
        self.decoder_checkpoint = str(
            Path(self.diffusion_payload["decoder_checkpoint"]).resolve()
        )

    @property
    def validation_condition_count(self) -> int:
        return int(self.validation_indices.size)

    def generate(
        self,
        validation_offsets: Sequence[int],
        policy_sets: Sequence[Sequence[str]],
        *,
        seed: int,
    ) -> dict[str, Any]:
        offsets = np.asarray(validation_offsets, dtype=np.int64)
        if offsets.ndim != 1 or offsets.size < 1 or len(policy_sets) != offsets.size:
            raise ValueError("condition offsets and policy sets must align")
        if int(offsets.min()) < 0 or int(offsets.max()) >= self.validation_condition_count:
            raise ValueError("validation condition offset is outside the held-out split")
        normalized_policy_sets: list[tuple[str, ...]] = []
        candidate_counts = np.empty(offsets.size, dtype=np.int64)
        selection_pool_offsets = np.zeros(offsets.size, dtype=np.int64)
        for row, values in enumerate(policy_sets):
            policies = tuple(str(value) for value in values)
            if not policies or len(set(policies)) != len(policies):
                raise ValueError("each condition needs distinct output policies")
            if not set(policies) <= SUPPORTED_POLICIES:
                raise ValueError("unsupported structured output policy")
            normalized_policy_sets.append(policies)
            has_selection = bool(
                set(policies) & {POLICY_ABSOLUTE, POLICY_GUIDED}
            )
            has_base = POLICY_BASE in policies
            selection_pool_offsets[row] = int(has_selection and has_base)
            candidate_counts[row] = (
                int(self.config.candidate_pool_size) + selection_pool_offsets[row]
                if has_selection
                else 1
            )
        offsets_in_candidates = np.concatenate(
            (np.zeros(1, dtype=np.int64), np.cumsum(candidate_counts))
        )
        candidate_total = int(np.sum(candidate_counts))
        if candidate_total > int(self.train_indices.size):
            raise ValueError(
                "one generation call cannot exceed the unique train-anchor count"
            )
        condition_indices = self.validation_indices[offsets]
        source_indices = np.repeat(condition_indices, candidate_counts)
        candidate_conditions_np = self.conditions[source_indices]
        anchor_indices = self._select_unique_anchors(source_indices)
        anchor_rng = np.random.default_rng(int(seed) ^ 0x5A17C0DE)
        for group in range(offsets.size):
            start = int(offsets_in_candidates[group])
            stop = int(offsets_in_candidates[group + 1])
            anchor_indices[start:stop] = anchor_rng.permutation(
                anchor_indices[start:stop]
            )
        anchor_normalized = self._model_normalized_anchors(anchor_indices)
        dtype = next(self.diffusion.parameters()).dtype
        generator = self.torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        initial_noise = self.torch.randn(
            (candidate_total, int(self.diffusion.config.active_latent_dim)),
            dtype=dtype,
            generator=generator,
            device="cpu",
        )
        timesteps = self.torch.full(
            (candidate_total,),
            int(self.config.start_timestep),
            dtype=self.torch.long,
            device="cpu",
        )
        initial_state = q_sample(
            anchor_normalized, timesteps, initial_noise, self.schedule
        )
        candidate_conditions = self.torch.as_tensor(
            candidate_conditions_np, dtype=dtype, device=self.device
        )
        normalized_conditions = normalize_vectors(
            candidate_conditions, self.diffusion_payload["condition_normalization"]
        )
        clean = self._reverse_batches(initial_state, normalized_conditions)
        rho_end = int(self.diffusion.config.rho_latent_dim)
        clean[:, :rho_end] = anchor_normalized[:, :rho_end].to(self.device)
        zero = self.torch.zeros(
            candidate_total, dtype=self.torch.long, device=self.device
        )
        with self.torch.no_grad():
            prediction = self.reward_model(clean, zero, normalized_conditions)[
                "absolute"
            ]

        selected = np.empty(offsets.size, dtype=np.int64)
        for row in range(offsets.size):
            start = int(offsets_in_candidates[row])
            stop = int(offsets_in_candidates[row + 1])
            selection_start = start + int(selection_pool_offsets[row])
            selected[row] = selection_start + int(
                self.torch.argmax(prediction[selection_start:stop]).item()
            )
        selected_tensor = self.torch.as_tensor(
            selected, dtype=self.torch.long, device=self.device
        )
        guided_groups = [
            row
            for row, policies in enumerate(normalized_policy_sets)
            if POLICY_GUIDED in policies
        ]
        guided_by_group: dict[int, Any] = {}
        local_rms_by_group: dict[int, float] = {}
        local_max_by_group: dict[int, float] = {}
        if guided_groups:
            group_tensor = self.torch.as_tensor(
                guided_groups, dtype=self.torch.long, device=self.device
            )
            selected_for_guidance = selected_tensor.index_select(0, group_tensor)
            baseline = clean.index_select(0, selected_for_guidance)
            if bool(self.config.centered_local_guidance_enabled):
                guided = sample_structured_latents(
                    self.diffusion,
                    normalized_conditions.index_select(0, selected_for_guidance),
                    initial_noise=initial_state.to(self.device).index_select(
                        0, selected_for_guidance
                    ),
                    sample_steps=int(self.config.sample_steps),
                    start_timestep=int(self.config.start_timestep),
                    eta=float(self.config.eta),
                    guidance=self.guidance,
                )
                selected_anchors = anchor_normalized.to(self.device).index_select(
                    0, selected_for_guidance
                )
                guided[:, :rho_end] = selected_anchors[:, :rho_end]
                guided = project_guided_local_trust_region(
                    guided,
                    baseline,
                    rho_latent_dim=rho_end,
                    maximum_local_rms_delta=float(
                        self.config.maximum_local_rms_delta
                    ),
                    maximum_coordinate_delta=float(
                        self.config.maximum_coordinate_delta
                    ),
                )
            else:
                guided = baseline.clone()
            delta = guided[:, rho_end:] - baseline[:, rho_end:]
            rms = self.torch.sqrt(self.torch.mean(delta.square(), dim=1))
            maximum = self.torch.max(self.torch.abs(delta), dim=1).values
            for position, group in enumerate(guided_groups):
                guided_by_group[group] = guided[position]
                local_rms_by_group[group] = float(rms[position].detach().cpu())
                local_max_by_group[group] = float(maximum[position].detach().cpu())

        rows = []
        model_rows = []
        condition_rows = []
        for group, policies in enumerate(normalized_policy_sets):
            start = int(offsets_in_candidates[group])
            selected_row = int(selected[group])
            selection_pool_size = int(
                candidate_counts[group] - selection_pool_offsets[group]
            )
            for policy in policies:
                if policy == POLICY_BASE:
                    policy_implementation = "independent_base"
                elif bool(self.config.centered_local_guidance_enabled) and policy == POLICY_GUIDED:
                    policy_implementation = "absolute_selected_plus_centered_local_guidance"
                else:
                    policy_implementation = "absolute_selected"
                source_row = start if policy == POLICY_BASE else selected_row
                model_latent = (
                    guided_by_group[group]
                    if policy == POLICY_GUIDED
                    else clean[source_row]
                )
                model_rows.append(model_latent)
                condition_rows.append(candidate_conditions[source_row])
                rows.append(
                    {
                        "condition_group": int(group),
                        "validation_condition_offset": int(offsets[group]),
                        "validation_source_index": int(condition_indices[group]),
                        "policy": policy,
                        "candidate_pool_size": selection_pool_size,
                        "candidate_trajectory_count": int(candidate_counts[group]),
                        "independent_base_trajectory": bool(selection_pool_offsets[group]),
                        "policy_implementation": policy_implementation,
                        "candidate_source_row": int(source_row),
                        "selected_predicted_absolute_reward": float(
                            prediction[selected_row].detach().cpu()
                        ),
                        "guided_local_rms_delta": float(
                            local_rms_by_group.get(group, 0.0)
                        ),
                        "guided_local_coordinate_delta": float(
                            local_max_by_group.get(group, 0.0)
                        ),
                    }
                )
        model_output = self.torch.stack(model_rows, dim=0)
        full_latents = self._decode_model_latents(model_output)
        if not bool(self.torch.all(self.torch.isfinite(full_latents)).item()):
            raise FloatingPointError("structured guided decoder latents are non-finite")
        return {
            "full_latents": full_latents,
            "equation_conditions": self.torch.stack(condition_rows, dim=0),
            "rows": rows,
            "candidate_trajectory_count": candidate_total,
            "anchor_indices": anchor_indices,
        }

    def _reverse_batches(self, initial_state, normalized_conditions):
        rows = []
        with self.torch.no_grad():
            for start in range(0, int(initial_state.shape[0]), int(self.config.batch_size)):
                stop = min(
                    start + int(self.config.batch_size), int(initial_state.shape[0])
                )
                rows.append(
                    sample_structured_latents(
                        self.diffusion,
                        normalized_conditions[start:stop],
                        initial_noise=initial_state[start:stop].to(self.device),
                        sample_steps=int(self.config.sample_steps),
                        start_timestep=int(self.config.start_timestep),
                        eta=float(self.config.eta),
                    )
                )
        return self.torch.cat(rows, dim=0)

    def _select_unique_anchors(self, source_indices: np.ndarray) -> np.ndarray:
        mean = np.asarray(
            self.diffusion_payload["condition_normalization"]["mean"],
            dtype=np.float64,
        )
        std = np.asarray(
            self.diffusion_payload["condition_normalization"]["std"],
            dtype=np.float64,
        )
        normalized = (self.conditions.astype(np.float64) - mean[None]) / std[None]
        costs = np.sum(
            (
                normalized[source_indices, None, :]
                - normalized[self.train_indices][None, :, :]
            )
            ** 2,
            axis=-1,
        )
        rows, columns = linear_sum_assignment(costs)
        if rows.size != source_indices.size:
            raise RuntimeError("unique manifold anchor assignment is incomplete")
        selected = np.empty(source_indices.shape, dtype=np.int64)
        selected[rows] = self.train_indices[columns]
        if np.unique(selected).size != selected.size:
            raise RuntimeError("unique manifold anchor assignment repeated an anchor")
        return selected

    def _model_normalized_anchors(self, indices: np.ndarray):
        dtype = next(self.diffusion.parameters()).dtype
        full = self.torch.as_tensor(
            self.full_reference[indices], dtype=dtype, device="cpu"
        )
        canonical = active_latent_from_full(
            full, self.bottleneck["canonical_config"]
        )
        canonical_normalized = normalize_vectors(
            canonical, self.bottleneck["canonical_normalization"]
        )
        model_active = encode_shared_local_time_channel(
            canonical_normalized,
            self.bottleneck["basis"],
            self.bottleneck["canonical_config"],
            self.bottleneck["model_config"],
            rank=int(self.bottleneck["rank"]),
        )
        return normalize_vectors(
            model_active, self.diffusion_payload["active_latent_normalization"]
        ).contiguous()

    def _decode_model_latents(self, values):
        model_active = denormalize_vectors(
            values, self.diffusion_payload["active_latent_normalization"]
        )
        canonical_normalized = decode_shared_local_time_channel(
            model_active,
            self.bottleneck["basis"],
            self.bottleneck["model_config"],
            self.bottleneck["canonical_config"],
            rank=int(self.bottleneck["rank"]),
        )
        canonical = denormalize_vectors(
            canonical_normalized, self.bottleneck["canonical_normalization"]
        )
        return full_latent_from_active(
            canonical, self.bottleneck["canonical_config"]
        )

    def _validate_reference_arrays(self) -> None:
        if self.conditions.ndim != 2 or self.conditions.shape[0] != (
            self.full_reference.shape[0]
        ):
            raise ValueError("structured latent conditions are invalid")
        if self.full_reference.ndim != 2:
            raise ValueError("structured fitted latent bank is invalid")
        for name, indices in (
            ("train", self.train_indices),
            ("validation", self.validation_indices),
        ):
            if indices.ndim != 1 or indices.size < 1:
                raise ValueError(f"structured {name} split is empty")
            if int(indices.min()) < 0 or int(indices.max()) >= len(self.conditions):
                raise ValueError(f"structured {name} split index is invalid")
        if np.intersect1d(self.train_indices, self.validation_indices).size:
            raise ValueError("structured train and validation splits overlap")
        if not np.all(np.isfinite(self.full_reference)) or not np.all(
            np.isfinite(self.conditions)
        ):
            raise FloatingPointError("structured reference artifact is non-finite")


def _read_json(path: Path) -> dict[str, Any]:
    import json

    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _torch():
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("torch is required for structured guided generation") from exc
    return torch
