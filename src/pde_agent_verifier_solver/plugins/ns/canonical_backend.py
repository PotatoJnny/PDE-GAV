"""Canonical CampaignController adapter for the frozen structured NS model.

The structured-flow sampler, smooth perturbation implementation, and reward
Guidance backend predate the product Controller.  This module reuses those
scientific components behind the normal ``GeneratorProgram`` lifecycle; it
does not bind or invoke the historical Macro runner.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
from typing import Mapping, Sequence

from pde_agent_verifier_solver.engine.guidance_controller import (
    FrozenGuidanceBoundary,
    GuidanceLifecycleController,
    GuidanceProtocolConfig,
    GuidanceRewardFeedback,
    GuidanceRewardRow,
)
from pde_agent_verifier_solver.sdk import (
    Artifact,
    Case,
    CaseRecord,
    CaseSource,
    CaseVisibility,
    GenerationKind,
    GenerationPurpose,
    GenerationRequest,
    GeneratorCheckpoint,
    GuidedSampleRequest,
    GuidedSampleSpec,
    GuidedSamplingPurpose,
    RoundFeedback,
    sample_frozen_guidance,
)

from .generator import NSStructuredGenerator
from .structured_flow_runtime import (
    StructuredNSArtifactBackend,
    build_structured_ns_backend,
)
from .structured_guidance_backend import (
    StructuredNSGuidanceConfig,
    StructuredNSMacroGuidanceBackend,
)


_NS_REWARD_MATCHER_DEFINITION_SHA256 = hashlib.sha256(
    b"ns-velocity-relative-l2-mean-v1"
).hexdigest()


@dataclass(frozen=True)
class StructuredNSBaseState:
    """Lightweight identity carried by a canonical Generator checkpoint."""

    generator_id: str
    artifact_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.generator_id, str) or not self.generator_id.strip():
            raise ValueError("structured NS base state needs a Generator id")
        _require_sha256(self.artifact_sha256, "structured NS Generator artifact")


@dataclass(frozen=True)
class CanonicalStructuredNSBundle:
    """Ready-to-compose NS Generator and its already-frozen checkpoint."""

    generator: NSStructuredGenerator
    checkpoint: GeneratorCheckpoint
    backend: "CanonicalStructuredNSBackend"
    components: StructuredNSArtifactBackend
    guidance_backend: StructuredNSMacroGuidanceBackend | None = None


class CanonicalStructuredNSBackend:
    """Expose the existing NS mathematics through the canonical Generator ABI."""

    # This adapter loads the already-trained D0 flow/PCA artifacts. A canonical
    # Campaign must therefore carry the explicit checkpoint returned by the
    # builder rather than trying to train again at Run start.
    trainable = False

    def __init__(
        self,
        components: StructuredNSArtifactBackend,
        *,
        guidance_backend: StructuredNSMacroGuidanceBackend | None = None,
        guidance_protocol: GuidanceProtocolConfig | None = None,
        development_guidance_enabled: bool = True,
        development_perturbations_enabled: bool = True,
    ) -> None:
        for value in (development_guidance_enabled, development_perturbations_enabled):
            if not isinstance(value, bool):
                raise TypeError("NS development permissions must be booleans")
        source = getattr(components, "case_source", None)
        if source is None or not callable(getattr(source, "sample", None)):
            raise TypeError("canonical NS backend needs a manufactured case source")
        if not callable(getattr(source, "release_cases", None)):
            raise TypeError("canonical NS case source needs release_cases")
        if (guidance_backend is None) != (guidance_protocol is None):
            raise ValueError(
                "NS guidance backend and GuidanceProtocolConfig must be supplied together"
            )
        if guidance_backend is not None:
            backend_config_sha256 = getattr(
                guidance_backend, "guidance_config_sha256", None
            )
            if backend_config_sha256 != guidance_protocol.guidance_config_sha256:
                raise ValueError("NS guidance backend and protocol config differ")
            if dict(guidance_backend.official_sample_settings) != dict(
                guidance_protocol.official_sample_settings
            ):
                raise ValueError("NS guidance backend and official sample settings differ")

        self.components = components
        self.case_source = source
        self.guidance_backend = guidance_backend
        self.guidance_protocol = guidance_protocol
        self.development_guidance_enabled = development_guidance_enabled
        self.development_perturbations_enabled = development_perturbations_enabled
        self.public_session_methods = tuple(
            getattr(source, "public_session_methods", ())
        )
        self.development_solution_session_methods = tuple(
            getattr(source, "development_solution_session_methods", ())
        )
        self._base_state = StructuredNSBaseState(
            generator_id=source.generator_id,
            artifact_sha256=source.artifact_sha256,
        )
        self._guidance_attempted: set[tuple[int, int]] = set()

    @property
    def base_state(self) -> StructuredNSBaseState:
        return self._base_state

    @property
    def supports_guidance(self) -> bool:
        return self.guidance_backend is not None

    @property
    def supports_perturbations(self) -> bool:
        return getattr(self.case_source, "supports_perturbations", False) is True

    def train_base(self, dataset, settings, progress=None):
        del dataset, settings, progress
        raise RuntimeError(
            "canonical structured NS runs require the already-frozen checkpoint"
        )

    def fit_guidance(
        self,
        base_state: object,
        previous_guidance: object | None,
        feedback: RoundFeedback,
        settings: Mapping[str, object],
        progress=None,
    ) -> FrozenGuidanceBoundary:
        self._require_base_state(base_state)
        if self.guidance_backend is None or self.guidance_protocol is None:
            raise RuntimeError("this canonical NS backend has no Guidance capability")
        if not isinstance(feedback, RoundFeedback):
            raise TypeError("canonical NS Guidance needs RoundFeedback")
        if not isinstance(feedback.selected_artifact, Artifact):
            raise TypeError("canonical NS Guidance needs the selected Artifact")
        if previous_guidance is not None:
            if not isinstance(previous_guidance, FrozenGuidanceBoundary):
                raise TypeError("previous NS Guidance has the wrong type")
            prior_target = (
                previous_guidance.snapshot.training_request.target_round_index
            )
            if prior_target != feedback.round_index:
                raise ValueError("previous NS Guidance does not belong to the source round")

        eligible = tuple(
            row for row in feedback.cases if _is_unguided_generator_root(row.case)
        )
        guidance_config = getattr(self.guidance_backend, "config", None)
        minimum = int(getattr(guidance_config, "minimum_cases", 1))
        if len(eligible) < minimum:
            raise ValueError(
                "NS Guidance needs at least "
                f"{minimum} unperturbed Base Generator feedback Cases; "
                f"received {len(eligible)}"
            )
        reward_feedback = GuidanceRewardFeedback(
            source_round_index=feedback.round_index,
            target_round_index=feedback.round_index + 1,
            target_solver_sha256=_candidate_backend_digest(
                feedback.selected_artifact
            ),
            base_generator_sha256=self._base_state.artifact_sha256,
            reward_matcher_definition_sha256=(
                _NS_REWARD_MATCHER_DEFINITION_SHA256
            ),
            rows=tuple(
                GuidanceRewardRow(
                    case=row.case,
                    primary_error=float(row.primary_error),
                    status="valid",
                    metrics={
                        str(name): float(value)
                        for name, value in row.metrics.items()
                    },
                )
                for row in eligible
            ),
        )
        controller_seed = settings.get("controller_seed")
        if (
            isinstance(controller_seed, bool)
            or not isinstance(controller_seed, int)
            or controller_seed < 0
        ):
            raise ValueError(
                "canonical NS Guidance needs a nonnegative controller_seed"
            )
        if progress:
            progress("Training frozen NS Guidance", 0.0)
        boundary_key = (feedback.round_index, feedback.round_index + 1)
        if boundary_key in self._guidance_attempted:
            raise RuntimeError("NS Guidance boundary training was already attempted")
        self._guidance_attempted.add(boundary_key)
        # Canonical Guidance consumes the previous round's one common Official
        # suite. Its clean Base stratum is user-sized; a legacy template count
        # must never cause the adapter to sample a second hidden cohort.
        effective_protocol = replace(
            self.guidance_protocol,
            reward_feedback_case_count=len(eligible),
        )
        boundary = GuidanceLifecycleController(
            config=effective_protocol,
            backend=self.guidance_backend,
        ).train_boundary(
            reward_feedback,
            controller_training_seed=controller_seed,
        )
        if progress:
            progress("Frozen NS Guidance is ready", 1.0)
        return boundary

    def generate(
        self,
        base_state: object,
        guidance_state: object | None,
        request: GenerationRequest,
        progress=None,
    ) -> tuple[CaseRecord, ...]:
        self._require_base_state(base_state)
        if not isinstance(request, GenerationRequest):
            raise TypeError("canonical NS generation needs a GenerationRequest")
        # Physical G/P support also serves hidden evaluation. It does not grant
        # the same scientific actions to the coding Agent in an ablation.
        if request.purpose is GenerationPurpose.DEVELOPMENT:
            if request.kind is GenerationKind.GUIDED and not self.development_guidance_enabled:
                raise PermissionError("NS development guidance is disabled")
            if request.kind is GenerationKind.PERTURBED and not self.development_perturbations_enabled:
                raise PermissionError("NS development perturbation is disabled")
        if request.kind is GenerationKind.GUIDED:
            records = self._generate_guided(guidance_state, request)
        else:
            records = self._generate_base_or_perturbed(request)
        if progress:
            progress(
                f"Generated {len(records)} structured NS Cases",
                1.0,
            )
        return records

    def public_session_factory(self, case: Case) -> object:
        return self.case_source.public_session_factory(case)

    def development_solution_session_factory(self, case: Case) -> object:
        return self.case_source.development_solution_session_factory(case)

    def release_cases(self, case_ids: Sequence[str]) -> None:
        self.case_source.release_cases(tuple(case_ids))

    def _generate_base_or_perturbed(
        self,
        request: GenerationRequest,
    ) -> tuple[CaseRecord, ...]:
        if request.kind not in (GenerationKind.BASE, GenerationKind.PERTURBED):
            raise ValueError("canonical NS source supports BASE and PERTURBED here")
        from pde_agent_verifier_solver.sdk.macro_loop import CohortKind, SampleRequest

        try:
            cohort = CohortKind(request.purpose.value)
        except ValueError as exc:
            raise ValueError(
                f"unsupported canonical NS sampling purpose {request.purpose.value!r}"
            ) from exc
        return tuple(
            self.case_source.sample(
                SampleRequest(
                    round_index=request.round_index,
                    cohort=cohort,
                    seed=request.seed,
                    count=request.count,
                    settings=request.settings,
                    kind=request.kind,
                    anchors=request.anchors,
                )
            )
        )

    def _generate_guided(
        self,
        guidance_state: object | None,
        request: GenerationRequest,
    ) -> tuple[CaseRecord, ...]:
        if self.guidance_backend is None:
            raise RuntimeError("this canonical NS backend has no Guidance capability")
        if not isinstance(guidance_state, FrozenGuidanceBoundary):
            raise ValueError("guided NS sampling needs a frozen round Guidance state")
        training = guidance_state.snapshot.training_request
        if training.target_round_index != request.round_index:
            raise ValueError("frozen NS Guidance does not belong to this sampling round")
        purpose = (
            GuidedSamplingPurpose.DEVELOPMENT_ONLINE
            if request.visibility is CaseVisibility.DEVELOPMENT
            else GuidedSamplingPurpose.OFFICIAL_COMMON_HIDDEN
        )
        settings = (
            {}
            if purpose is GuidedSamplingPurpose.DEVELOPMENT_ONLINE
            else guidance_state.config.official_sample_settings
        )
        nonce = (
            f"canonical-{request.purpose.value}-r{request.round_index}-"
            f"s{request.seed}"
        )
        guided_request = GuidedSampleRequest(
            spec=GuidedSampleSpec(request.count, settings),
            request_id=nonce,
            purpose=purpose,
            snapshot_sha256=guidance_state.snapshot.snapshot_sha256,
            controller_seed=request.seed,
            controller_nonce=nonce,
            inference_wall_budget_seconds=(
                guidance_state.config.inference_wall_budget_seconds
            ),
            compute_capacity=guidance_state.config.inference_compute_capacity,
        )
        oracle = guidance_state.backend.open_guidance(guidance_state.snapshot)
        close = getattr(oracle, "close", None)
        try:
            batch = sample_frozen_guidance(oracle, guided_request)
            resolver = getattr(oracle, "resolve_case_records", None)
            if not callable(resolver):
                raise TypeError(
                    "structured NS Guidance lacks controller record resolution"
                )
            records = tuple(resolver(batch))
            if len(records) != request.count:
                raise ValueError("structured NS Guidance returned the wrong Case count")
            if any(
                record.case.visibility is not request.visibility
                or record.case.source is CaseSource.PERTURBED
                or record.case.parent_case_id is not None
                for record in records
            ):
                raise ValueError("structured NS Guidance returned invalid clean Cases")
            return records
        except BaseException:
            if callable(close):
                close()
            raise

    def _require_base_state(self, value: object) -> None:
        if not isinstance(value, StructuredNSBaseState):
            raise TypeError("canonical NS Generator checkpoint payload has the wrong type")
        if value != self._base_state:
            raise ValueError("canonical NS Generator checkpoint targets another backend")


def build_canonical_structured_ns_generator(
    *,
    generator_id: str,
    generator_sha256: str,
    flow_checkpoint: str | Path,
    pca_checkpoint: str | Path,
    time_interval: tuple[float, float],
    viscosity: float,
    perturbation_amplitude: float,
    device: str = "cuda",
    integration_steps: int = 64,
    flow_state: str = "ema",
    integration_method: str = "heun",
    cuda_graph_sampling: bool = False,
    perturbation_family: str = "spectral_complement",
    spectral_max_mode: int = 20,
    geometry_spec: Mapping[str, object] | None = None,
    max_points_per_request: int = 4096,
    guidance_artifact_root: str | Path | None = None,
    guidance_protocol: GuidanceProtocolConfig | None = None,
    guidance_config: StructuredNSGuidanceConfig | None = None,
    development_guidance_enabled: bool = True,
    development_perturbations_enabled: bool = True,
    symbols: object | None = None,
    guidance_symbols: object | None = None,
) -> CanonicalStructuredNSBundle:
    """Load the real NS artifacts once and return canonical run components."""

    if guidance_protocol is not None and guidance_artifact_root is None:
        raise ValueError("enabled canonical NS Guidance needs an artifact root")
    if guidance_protocol is None and (
        guidance_artifact_root is not None or guidance_config is not None
    ):
        raise ValueError(
            "NS Guidance artifact/config settings require a GuidanceProtocolConfig"
        )
    components = build_structured_ns_backend(
        generator_id=generator_id,
        generator_sha256=generator_sha256,
        flow_checkpoint=flow_checkpoint,
        pca_checkpoint=pca_checkpoint,
        time_interval=time_interval,
        viscosity=viscosity,
        perturbation_amplitude=perturbation_amplitude,
        device=device,
        integration_steps=integration_steps,
        flow_state=flow_state,
        integration_method=integration_method,
        cuda_graph_sampling=cuda_graph_sampling,
        perturbation_family=perturbation_family,
        spectral_max_mode=spectral_max_mode,
        geometry_spec=geometry_spec,
        max_points_per_request=max_points_per_request,
        symbols=symbols,
    )
    guidance_backend = (
        components.build_guidance_backend(
            artifact_root=guidance_artifact_root,
            config=guidance_config,
            symbols=guidance_symbols,
        )
        if guidance_protocol is not None
        else None
    )
    backend = CanonicalStructuredNSBackend(
        components,
        guidance_backend=guidance_backend,
        guidance_protocol=guidance_protocol,
        development_guidance_enabled=development_guidance_enabled,
        development_perturbations_enabled=development_perturbations_enabled,
    )
    generator = NSStructuredGenerator(backend)
    checkpoint = GeneratorCheckpoint(
        generator_id=generator.info.program_id,
        payload=backend.base_state,
        location=str(Path(flow_checkpoint).expanduser().resolve()),
        report={
            "implementation": type(backend).__name__,
            "scientific_generator_id": generator_id,
            "scientific_artifact_sha256": generator_sha256,
            "guidance_enabled": backend.supports_guidance,
            "flow": dict(components.flow_sampler.artifact_metadata),
        },
    )
    return CanonicalStructuredNSBundle(
        generator=generator,
        checkpoint=checkpoint,
        backend=backend,
        components=components,
        guidance_backend=guidance_backend,
    )


def _is_unguided_generator_root(case: Case) -> bool:
    if (
        case.source is not CaseSource.GENERATED
        or case.parent_case_id is not None
    ):
        return False
    lineage = case.public_data.get("distribution_lineage")
    return not (
        isinstance(lineage, Mapping)
        and lineage.get("kind") == "frozen_reward_guided"
    )


def _candidate_backend_digest(artifact: Artifact) -> str:
    """Internal binding required by the legacy reward-model artifact format."""

    return hashlib.sha256(
        (
            "canonical-ns-candidate-v1\0"
            f"{artifact.artifact_id}\0{artifact.round_index}\0{artifact.agent_id}"
        ).encode("utf-8")
    ).hexdigest()


def _require_sha256(value: object, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} needs a lowercase sha256")


__all__ = [
    "CanonicalStructuredNSBundle",
    "CanonicalStructuredNSBackend",
    "StructuredNSBaseState",
    "build_canonical_structured_ns_generator",
]
