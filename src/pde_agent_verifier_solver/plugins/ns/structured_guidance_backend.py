"""Real frozen reward-Guidance adapter for structured continuous NS.

The shared controller owns cadence and policy.  This module only seals
controller-private coefficient/reward pairs, trains the legacy noisy-state
reward model once, and exposes frozen guided answer-first inference.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import isfinite
from pathlib import Path
from threading import Lock
from time import perf_counter
from types import MappingProxyType
from typing import Mapping, Sequence

from pde_agent_verifier_solver.sdk import (
    CaseVisibility,
    FrozenGuidanceSnapshot,
    GuidanceCapability,
    GuidanceTrainingRequest,
    GuidedAnswerFirstSample,
    GuidedSampleBatch,
    GuidedSampleRequest,
    GuidedSamplingPurpose,
)

from .continuous_backend import StructuredNSManufacturedCaseSource
from .public_problem import NS_DEVELOPMENT_SOLUTION_SESSION_METHODS
from .structured_flow_runtime import LegacyStructuredNSFlowSampler


NS_STRUCTURED_GUIDANCE_BACKEND_ID = "ns-structured-reward-gradient-v1"


@dataclass(frozen=True)
class StructuredNSGuidanceConfig:
    """Frozen training/inference choices hashed into the shared lifecycle."""

    training_steps: int = 5000
    batch_size: int = 256
    width: int = 512
    depth: int = 6
    time_width: int = 128
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-4
    maximum_gradient_norm: float = 1.0
    minimum_cases: int = 32
    guidance_strength: float = 0.10
    guidance_max_flow_ratio: float = 0.25

    def __post_init__(self) -> None:
        for name in (
            "training_steps",
            "batch_size",
            "width",
            "depth",
            "time_width",
            "minimum_cases",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"structured NS Guidance {name} must be positive")
        if self.time_width % 2:
            raise ValueError("structured NS Guidance time width must be even")
        for name in (
            "learning_rate",
            "maximum_gradient_norm",
            "guidance_max_flow_ratio",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"structured NS Guidance {name} must be positive")
        if not isfinite(float(self.weight_decay)) or self.weight_decay < 0.0:
            raise ValueError("structured NS Guidance weight decay must be nonnegative")
        if not isfinite(float(self.guidance_strength)) or self.guidance_strength < 0.0:
            raise ValueError("structured NS Guidance strength must be nonnegative")

    def identity_payload(self) -> dict[str, int | float]:
        return {
            name: getattr(self, name)
            for name in (
                "training_steps",
                "batch_size",
                "width",
                "depth",
                "time_width",
                "learning_rate",
                "weight_decay",
                "maximum_gradient_norm",
                "minimum_cases",
                "guidance_strength",
                "guidance_max_flow_ratio",
            )
        }

    @property
    def sha256(self) -> str:
        return _json_sha256(self.identity_payload())


class StructuredNSMacroGuidanceBackend:
    """Concrete ``MacroGuidanceBackend`` over the frozen structured flow."""

    def __init__(
        self,
        *,
        flow_sampler: LegacyStructuredNSFlowSampler,
        case_source: StructuredNSManufacturedCaseSource,
        artifact_root: str | Path,
        config: StructuredNSGuidanceConfig | None = None,
        symbols: object | None = None,
    ) -> None:
        if not isinstance(flow_sampler, LegacyStructuredNSFlowSampler) and not all(
            callable(getattr(flow_sampler, name, None))
            for name in ("encode_fields", "sample_many_with_guidance")
        ):
            raise TypeError("NS Guidance needs a structured flow sampler")
        if not isinstance(case_source, StructuredNSManufacturedCaseSource):
            raise TypeError("NS Guidance needs the bound NS case source")
        root = Path(artifact_root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        self.flow_sampler = flow_sampler
        self.case_source = case_source
        self.artifact_root = root
        self.config = config or StructuredNSGuidanceConfig()
        self._symbols = symbols or _legacy_guidance_symbols()
        self._capability = GuidanceCapability.online(
            NS_STRUCTURED_GUIDANCE_BACKEND_ID
        )
        self._attempted: set[str] = set()
        self._model_cache: dict[str, object] = {}
        self._feedback_sidecars: dict[str, tuple[Path, str]] = {}
        self._lock = Lock()

    @property
    def guidance_capability(self) -> GuidanceCapability:
        return self._capability

    @property
    def guidance_config_sha256(self) -> str:
        return self.config.sha256

    @property
    def official_sample_settings(self) -> Mapping[str, str | int | float]:
        metadata = self.flow_sampler.artifact_metadata
        flow_state = metadata.get("flow_state")
        integration_steps = metadata.get("integration_steps")
        if flow_state not in {"raw", "ema"}:
            raise ValueError("structured NS flow metadata has invalid flow_state")
        if isinstance(integration_steps, bool) or not isinstance(integration_steps, int) or integration_steps < 1:
            raise ValueError("structured NS flow metadata has invalid integration_steps")
        return MappingProxyType(
            {
                "sampling_policy": "solver_reward_guided_rectified_flow",
                "flow_state": flow_state,
                "integration_steps": integration_steps,
                "guidance_strength": float(self.config.guidance_strength),
                "guidance_max_flow_ratio": float(self.config.guidance_max_flow_ratio),
            }
        )

    def seal_reward_feedback(self, feedback):
        from pde_agent_verifier_solver.engine.guidance_controller import (
            GuidanceRewardArtifact,
            GuidanceRewardFeedback,
        )

        if not isinstance(feedback, GuidanceRewardFeedback):
            raise TypeError("NS Guidance feedback has the wrong type")
        if feedback.base_generator_sha256 != self.case_source.artifact_sha256:
            raise ValueError("NS Guidance feedback targets a different base Generator")
        if len(feedback.rows) < self.config.minimum_cases:
            raise ValueError("NS Guidance feedback has too few cases")
        case_ids = tuple(row.case.case_id for row in feedback.rows)
        raw_answers = self.case_source.resolve_guidance_feedback_answers(case_ids)
        scores = self.flow_sampler.encode_fields(raw_answers).detach().cpu().float()
        rewards = self._symbols.torch.as_tensor(
            [row.primary_error for row in feedback.rows],
            dtype=self._symbols.torch.float32,
        )
        path = self.artifact_root / (
            f"reward-feedback-r{feedback.source_round_index:04d}-"
            f"{feedback.target_solver_sha256[:16]}.json"
        )
        sidecar = path.with_suffix(".controller-private.pt")
        if path.exists() or sidecar.exists():
            raise FileExistsError(f"refusing to reuse NS Guidance feedback: {path}")
        sanitized = {
            "schema": "ns_unified_public_reward_feedback_v1",
            "source_round_index": feedback.source_round_index,
            "target_round_index": feedback.target_round_index,
            "target_solver_sha256": feedback.target_solver_sha256,
            "base_generator_sha256": feedback.base_generator_sha256,
            "reward_matcher_definition_sha256": (
                feedback.reward_matcher_definition_sha256
            ),
            "rows": [
                {
                    "case_id": row.case.case_id,
                    "source": row.case.source.value,
                    "visibility": row.case.visibility.value,
                    "round_index": row.case.round_index,
                    "public_data": _plain_json(row.case.public_data),
                    "primary_error": row.primary_error,
                    "status": row.status,
                    "metrics": dict(row.metrics),
                }
                for row in feedback.rows
            ],
        }
        path.write_text(json.dumps(sanitized, indent=2, sort_keys=True) + "\n")
        feedback_sha256 = _file_sha256(path)
        self._symbols.torch.save(
            {
                "schema": "ns_unified_guidance_private_state_v1",
                "reward_feedback_sha256": feedback_sha256,
                "target_solver_sha256": feedback.target_solver_sha256,
                "base_generator_sha256": feedback.base_generator_sha256,
                "reward_matcher_definition_sha256": (
                    feedback.reward_matcher_definition_sha256
                ),
                "case_ids": case_ids,
                "scores": scores,
                "rewards": rewards,
            },
            sidecar,
        )
        sidecar_sha256 = _file_sha256(sidecar)
        with self._lock:
            self._feedback_sidecars[feedback_sha256] = (sidecar, sidecar_sha256)
        return GuidanceRewardArtifact(str(path), feedback_sha256)

    def _resolve_private_artifact(self, uri):
        restored = getattr(self, "_resume_artifacts", {}).get(str(uri))
        return restored if restored is not None else _private_artifact_path(uri, root=self.artifact_root)

    def train_guidance(
        self,
        request: GuidanceTrainingRequest,
    ) -> FrozenGuidanceSnapshot:
        if not isinstance(request, GuidanceTrainingRequest):
            raise TypeError("NS Guidance training request has the wrong type")
        if request.base_generator_sha256 != self.case_source.artifact_sha256:
            raise ValueError("NS Guidance training targets a different base Generator")
        if request.guidance_config_sha256 != self.config.sha256:
            raise ValueError("NS Guidance training config digest differs")
        request_key = _json_sha256(request.identity_payload())
        with self._lock:
            if request_key in self._attempted:
                raise RuntimeError("NS Guidance training request was already attempted")
            self._attempted.add(request_key)
        feedback_path = self._resolve_private_artifact(request.reward_feedback_uri)
        sanitized = json.loads(feedback_path.read_text())
        if sanitized.get("schema") != "ns_unified_public_reward_feedback_v1":
            raise ValueError("unexpected NS Guidance public feedback schema")
        for key, expected in (
            ("target_solver_sha256", request.target_solver_sha256),
            ("base_generator_sha256", request.base_generator_sha256),
            (
                "reward_matcher_definition_sha256",
                request.reward_matcher_definition_sha256,
            ),
        ):
            if sanitized.get(key) != expected:
                raise ValueError(f"NS Guidance feedback changed {key}")
        with self._lock:
            sidecar_binding = self._feedback_sidecars.get(
                request.reward_feedback_sha256
            )
        if sidecar_binding is None:
            raise RuntimeError("NS Guidance private feedback state is unavailable")
        sidecar_path, sidecar_sha256 = sidecar_binding
        torch = self._symbols.torch
        payload = torch.load(sidecar_path, map_location="cpu", weights_only=False)
        if payload.get("schema") != "ns_unified_guidance_private_state_v1":
            raise ValueError("unexpected NS Guidance private state schema")
        if payload.get("reward_feedback_sha256") != request.reward_feedback_sha256:
            raise ValueError("NS Guidance private/public feedback binding differs")
        for key, expected in (
            ("target_solver_sha256", request.target_solver_sha256),
            ("base_generator_sha256", request.base_generator_sha256),
            (
                "reward_matcher_definition_sha256",
                request.reward_matcher_definition_sha256,
            ),
        ):
            if payload.get(key) != expected:
                raise ValueError(f"NS Guidance private state changed {key}")
        device = self.flow_sampler._device
        scores = torch.as_tensor(payload["scores"], device=device).float()
        rewards = torch.as_tensor(payload["rewards"], device=device).float()
        if scores.ndim != 2 or int(scores.shape[1]) != int(self.flow_sampler.pca.rank):
            raise ValueError("NS Guidance feedback PCA scores have the wrong shape")
        if rewards.shape != (scores.shape[0],):
            raise ValueError("NS Guidance feedback rewards have the wrong shape")
        if int(scores.shape[0]) < self.config.minimum_cases:
            raise ValueError("NS Guidance feedback has too few training rows")

        model_config = self._symbols.model_config_type(
            feature_dim=int(scores.shape[1]),
            width=self.config.width,
            depth=self.config.depth,
            time_width=self.config.time_width,
        )
        model = self._symbols.model_type.build(model_config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        reward_mean = rewards.mean()
        reward_std = rewards.std(unbiased=False).clamp_min(1.0e-8)
        targets = (rewards - reward_mean) / reward_std
        generator = torch.Generator(device=device).manual_seed(
            request.controller_training_seed
        )
        started = perf_counter()
        for _step in range(self.config.training_steps):
            if perf_counter() - started >= request.training_budget.maximum_wall_seconds:
                raise TimeoutError("NS Guidance training exhausted its wall budget")
            selected = torch.randint(
                int(scores.shape[0]),
                (self.config.batch_size,),
                device=device,
                generator=generator,
            )
            clean = scores[selected]
            noise = torch.randn(
                clean.shape,
                device=device,
                dtype=clean.dtype,
                generator=generator,
            )
            time = torch.rand(
                clean.shape[0],
                device=device,
                dtype=clean.dtype,
                generator=generator,
            )
            noisy = (1.0 - time[:, None]) * noise + time[:, None] * clean
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(noisy, time), targets[selected])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), self.config.maximum_gradient_norm
            )
            optimizer.step()
        # CUDA launches are asynchronous. Finish the queued optimizer work
        # before recording completion or checking the actual training wall.
        if getattr(device, "type", str(device).split(":", 1)[0]) == "cuda":
            torch.cuda.current_stream(device).synchronize()
        wall_seconds = perf_counter() - started
        if wall_seconds > request.training_budget.maximum_wall_seconds:
            raise TimeoutError("NS Guidance training exceeded its wall budget")

        output = self.artifact_root / (
            f"frozen-guidance-r{request.target_round_index:04d}-"
            f"{request.target_solver_sha256[:16]}.pt"
        )
        if output.exists():
            raise FileExistsError(f"refusing to reuse NS Guidance snapshot: {output}")
        checkpoint = {
            "schema": self._symbols.guidance_schema,
            "model_config": model_config.to_dict(),
            "model_state_dict": {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            },
            "reward_normalization": {
                "mean": float(reward_mean.detach().cpu()),
                "std": float(reward_std.detach().cpu()),
            },
            "reward_metric": "relative_l2",
            "reward_objective": "solver_error",
            "target_solver_sha256": request.target_solver_sha256,
            "base_generator_sha256": request.base_generator_sha256,
            "pca_sha256": self.flow_sampler.artifact_metadata["pca_sha256"],
            "pca_rank": int(self.flow_sampler.pca.rank),
            "guidance_config_sha256": self.config.sha256,
            "guidance_strength": self.config.guidance_strength,
            "guidance_max_flow_ratio": self.config.guidance_max_flow_ratio,
        }
        torch.save(checkpoint, output)
        artifact_sha256 = _file_sha256(output)
        snapshot = FrozenGuidanceSnapshot(
            training_request=request,
            backend_id=NS_STRUCTURED_GUIDANCE_BACKEND_ID,
            guided_artifact_uri=str(output),
            guided_artifact_sha256=artifact_sha256,
            campaign_guidance_training_wall_seconds=wall_seconds,
        )
        with self._lock:
            self._model_cache[snapshot.snapshot_sha256] = model.eval()
        return snapshot

    def open_guidance(
        self,
        snapshot: FrozenGuidanceSnapshot,
    ) -> "_FrozenStructuredNSGuidanceOracle":
        if not isinstance(snapshot, FrozenGuidanceSnapshot):
            raise TypeError("NS Guidance snapshot has the wrong type")
        if snapshot.backend_id != NS_STRUCTURED_GUIDANCE_BACKEND_ID:
            raise ValueError("NS Guidance snapshot targets another backend")
        if snapshot.training_request.base_generator_sha256 != self.case_source.artifact_sha256:
            raise ValueError("NS Guidance snapshot targets another base Generator")
        with self._lock:
            model = self._model_cache.get(snapshot.snapshot_sha256)
        if model is None:
            path = self._resolve_private_artifact(snapshot.guided_artifact_uri)
            payload = self._symbols.torch.load(
                path,
                map_location=self.flow_sampler._device,
                weights_only=False,
            )
            if payload.get("schema") != self._symbols.guidance_schema:
                raise ValueError("unexpected structured NS Guidance schema")
            if payload.get("base_generator_sha256") != self.case_source.artifact_sha256:
                raise ValueError("structured NS Guidance base binding differs")
            model_config = self._symbols.model_config_type.from_dict(
                dict(payload["model_config"])
            )
            model = self._symbols.model_type.build(model_config).to(
                self.flow_sampler._device
            )
            model.load_state_dict(payload["model_state_dict"])
            model.eval()
            with self._lock:
                self._model_cache[snapshot.snapshot_sha256] = model
        gradient = self._symbols.gradient_type(
            model,
            strength=self.config.guidance_strength,
            max_flow_ratio=self.config.guidance_max_flow_ratio,
        )
        return _FrozenStructuredNSGuidanceOracle(
            backend=self,
            snapshot=snapshot,
            gradient=gradient,
        )

    def release_guided_cases(self, case_ids: tuple[str, ...]) -> None:
        self.case_source.release_cases(case_ids)


class _FrozenStructuredNSGuidanceOracle:
    development_solution_methods = NS_DEVELOPMENT_SOLUTION_SESSION_METHODS

    def __init__(self, *, backend, snapshot, gradient) -> None:
        self._backend = backend
        self._snapshot = snapshot
        self._gradient = gradient
        self._issued_by_sample: dict[str, object] = {}
        self._closed = False
        self._lock = Lock()

    @property
    def guidance_capability(self) -> GuidanceCapability:
        return self._backend.guidance_capability

    @property
    def snapshot(self) -> FrozenGuidanceSnapshot:
        return self._snapshot

    def _validated_sample_settings(
        self,
        request: GuidedSampleRequest,
    ) -> Mapping[str, str | int | float]:
        expected = dict(self._backend.official_sample_settings)
        supplied = dict(request.spec.settings)
        if request.purpose is GuidedSamplingPurpose.OFFICIAL_COMMON_HIDDEN:
            if supplied != expected:
                raise ValueError(
                    "official NS guided requests require the frozen sample settings"
                )
        elif supplied and supplied != expected:
            raise ValueError("NS guided development settings are unsupported")
        return MappingProxyType(expected)

    def generate_answer_first(
        self,
        request: GuidedSampleRequest,
    ) -> Sequence[GuidedAnswerFirstSample]:
        if not isinstance(request, GuidedSampleRequest):
            raise TypeError("NS guided sample request has the wrong type")
        if request.snapshot_sha256 != self.snapshot.snapshot_sha256:
            raise ValueError("NS guided request targets another snapshot")
        effective_sample_settings = self._validated_sample_settings(request)
        with self._lock:
            if self._closed:
                raise RuntimeError("NS frozen Guidance oracle is closed")
        seeds = tuple(
            _seed(request.controller_seed, request.request_id, index)
            for index in range(request.spec.sample_count)
        )
        fields = self._backend.flow_sampler.sample_many_with_guidance(
            seeds,
            guidance=self._gradient,
            guidance_scale=1.0,
        )
        prefix = (
            f"ns-guided-r{self.snapshot.training_request.target_round_index}-"
            f"{_text_sha256(request.request_id)[:16]}"
        )
        records = self._backend.case_source.issue_answer_first_records(
            fields,
            round_index=self.snapshot.training_request.target_round_index,
            visibility=request.expected_visibility,
            case_id_prefix=prefix,
            controller_seed=request.controller_seed,
            distribution_lineage={
                "kind": "frozen_reward_guided",
                "snapshot_sha256": self.snapshot.snapshot_sha256,
                "base_generator_sha256": (
                    self.snapshot.training_request.base_generator_sha256
                ),
                "sampling_policy": effective_sample_settings["sampling_policy"],
                "sample_settings_sha256": _json_sha256(
                    dict(effective_sample_settings)
                ),
            },
        )
        samples = []
        with self._lock:
            for index, (seed, record) in enumerate(
                zip(seeds, records, strict=True)
            ):
                sample_id = f"{request.request_id}-{index:04d}"
                if sample_id in self._issued_by_sample:
                    raise ValueError("NS guided oracle reused a sample id")
                self._issued_by_sample[sample_id] = record.case
                reference_sha = _json_sha256(
                    {
                        "snapshot": self.snapshot.snapshot_sha256,
                        "request": request.request_id,
                        "seed": seed,
                    }
                )
                samples.append(
                    GuidedAnswerFirstSample(
                        sample_id=sample_id,
                        snapshot_sha256=self.snapshot.snapshot_sha256,
                        case=record.case,
                        reference_answer_sha256=reference_sha,
                        public_case_sha256=_text_sha256(
                            reference_sha + "\0ns-answer-to-public-case-v1"
                        ),
                        answer_to_case_derivation_sha256=_text_sha256(
                            "continuous-ns-answer-first-operator-v1"
                        ),
                        answer_access_locator=(
                            sample_id
                            if request.expected_visibility
                            is CaseVisibility.DEVELOPMENT
                            else None
                        ),
                    )
                )
        return tuple(samples)

    def resolve_case_records(self, batch: GuidedSampleBatch):
        if batch.request.snapshot_sha256 != self.snapshot.snapshot_sha256:
            raise ValueError("NS guided batch targets another snapshot")
        with self._lock:
            case_ids = tuple(
                self._issued_by_sample[sample.sample_id].case_id
                for sample in batch.samples
            )
        return self._backend.case_source.resolve_active_records(case_ids)

    def open_development_solution_session(self, sample_id: str):
        with self._lock:
            case = self._issued_by_sample.get(sample_id)
        if case is None:
            raise KeyError("NS guided sample is unknown")
        if case.visibility is not CaseVisibility.DEVELOPMENT:
            raise PermissionError("hidden NS guided truth is not Agent-visible")
        return self._backend.case_source.development_solution_session_factory(case)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            case_ids = tuple(case.case_id for case in self._issued_by_sample.values())
            self._issued_by_sample.clear()
        self._backend.case_source.release_cases(case_ids)


def _legacy_guidance_symbols():
    try:
        import torch
        from agent_verifier.neural_generator.structured_ns_guidance import (
            STRUCTURED_NS_REWARD_GUIDANCE_SCHEMA,
            StructuredRewardGradient,
            StructuredRewardModel,
            StructuredRewardModelConfig,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "real NS Guidance is unavailable; add legacy/ns/src to PYTHONPATH"
        ) from exc
    from types import SimpleNamespace

    return SimpleNamespace(
        torch=torch,
        guidance_schema=STRUCTURED_NS_REWARD_GUIDANCE_SCHEMA,
        model_config_type=StructuredRewardModelConfig,
        model_type=StructuredRewardModel,
        gradient_type=StructuredRewardGradient,
    )


def _plain_json(value):
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_json(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"NS public reward payload contains {type(value).__name__}")

def _private_artifact_path(uri: str, *, root: Path) -> Path:
    path = Path(uri).expanduser().resolve()
    if path != root and root not in path.parents:
        raise PermissionError("NS Guidance artifact escaped its controller-private root")
    if not path.is_file():
        raise FileNotFoundError(f"missing NS Guidance artifact: {path}")
    return path


def _seed(root_seed: int, request_id: str, index: int) -> int:
    return int.from_bytes(
        hashlib.sha256(
            f"ns-guided-seed-v1\0{root_seed}\0{request_id}\0{index}".encode()
        ).digest()[:8],
        "big",
    ) & ((1 << 63) - 1)


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "NS_STRUCTURED_GUIDANCE_BACKEND_ID",
    "StructuredNSGuidanceConfig",
    "StructuredNSMacroGuidanceBackend",
]
