"""Reusable lifecycle for coefficient-space PDE Generators.

A PDE-specific subclass describes the fields shown to the scientist and supplies
one backend. The backend owns the mathematics (field-to-coefficient encoding,
latent model, guidance and answer materialization); this class supplies the
single product lifecycle used by the shared engine.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Protocol

from pde_agent_verifier_solver.sdk.models import (
    Artifact,
    Case,
    CaseAccessKind,
    CaseRecord,
    GenerationRequest,
    GeneratorCapabilities,
    GeneratorCheckpoint,
    GeneratorFeedback,
    GeneratorState,
    PreparedDataset,
    ProgramInfo,
    RoundFeedback,
)


class CoefficientFlowBackend(Protocol):
    """PDE-owned numerical operations behind a coefficient-flow Generator."""

    def train_base(
        self,
        dataset: PreparedDataset,
        settings: Mapping[str, object],
        progress=None,
    ) -> object: ...

    def fit_guidance(
        self,
        base_state: object,
        previous_guidance: object | None,
        feedback: RoundFeedback,
        settings: Mapping[str, object],
        progress=None,
    ) -> object: ...

    def generate(
        self,
        base_state: object,
        guidance_state: object | None,
        request: GenerationRequest,
        progress=None,
    ) -> tuple[CaseRecord, ...]: ...


class BackendGeneratorProgram:
    """Standard product implementation for basis/PCA/flow Generators.

    Subclasses provide program info, capabilities, setup fields and previews.
    They do not reimplement checkpoint loading, guided updates, sampling, or
    resource cleanup.
    """

    info: ProgramInfo
    capabilities: GeneratorCapabilities

    def __init__(self, backend: CoefficientFlowBackend | None = None) -> None:
        self.backend = backend

    def train(
        self,
        dataset: PreparedDataset,
        settings: Mapping[str, object],
        progress=None,
    ) -> GeneratorCheckpoint:
        base = self._backend().train_base(dataset, settings, progress)
        return GeneratorCheckpoint(
            generator_id=self.info.program_id,
            payload=base,
            report={"implementation": type(self).__name__},
        )

    def load(
        self,
        checkpoint: GeneratorCheckpoint,
        settings: Mapping[str, object],
        progress=None,
    ) -> GeneratorState:
        if checkpoint.generator_id != self.info.program_id:
            raise ValueError(
                f"checkpoint {checkpoint.generator_id!r} does not belong to "
                f"{self.info.program_id!r}"
            )
        if progress:
            progress("Generator checkpoint loaded", 1.0)
        return GeneratorState(checkpoint=checkpoint, base=checkpoint.payload)

    def update_guidance(
        self,
        state: GeneratorState,
        feedback: GeneratorFeedback,
        settings: Mapping[str, object],
        progress=None,
    ) -> GeneratorState:
        selected = feedback.selected_program
        if not isinstance(selected, Artifact):
            raise TypeError(
                "coefficient-flow guidance needs the selected Artifact program"
            )
        legacy_feedback = RoundFeedback(
            round_index=feedback.source_round_index,
            selected_artifact=selected,
            cases=feedback.cases,
        )
        guidance = self._backend().fit_guidance(
            state.base,
            state.guidance,
            legacy_feedback,
            settings,
            progress,
        )
        return replace(state, guidance=guidance)

    def sample(
        self,
        state: GeneratorState,
        request: GenerationRequest,
        progress=None,
    ) -> tuple[CaseRecord, ...]:
        return tuple(
            self._backend().generate(
                state.base,
                state.guidance,
                request,
                progress,
            )
        )

    def open_case_session(
        self,
        state: GeneratorState,
        case: Case,
        access: CaseAccessKind,
    ) -> object | None:
        method_name = {
            CaseAccessKind.PUBLIC_PROBLEM: "public_session_factory",
            CaseAccessKind.DEVELOPMENT_SOLUTION: (
                "development_solution_session_factory"
            ),
        }[access]
        factory = getattr(self._backend(), method_name, None)
        return factory(case) if callable(factory) else None

    def case_session_methods(
        self,
        access: CaseAccessKind,
    ) -> tuple[str, ...]:
        attribute = {
            CaseAccessKind.PUBLIC_PROBLEM: "public_session_methods",
            CaseAccessKind.DEVELOPMENT_SOLUTION: (
                "development_solution_session_methods"
            ),
        }[access]
        methods = getattr(self._backend(), attribute, ())
        return tuple(methods)

    def release(
        self,
        state: GeneratorState,
        case_ids: tuple[str, ...],
    ) -> None:
        release = getattr(self._backend(), "release_cases", None)
        if callable(release):
            release(tuple(case_ids))

    # Temporary source compatibility for old experiment scripts. Product and
    # Controller code use the lifecycle above.
    def train_base(self, dataset, settings, progress=None):
        return self.train(dataset, settings, progress).payload

    def fit_guidance(
        self,
        base_state,
        previous_guidance,
        feedback,
        settings,
        progress=None,
    ):
        return self._backend().fit_guidance(
            base_state,
            previous_guidance,
            feedback,
            settings,
            progress,
        )

    def generate(self, base_state, guidance_state, request, progress=None):
        checkpoint = GeneratorCheckpoint(self.info.program_id, base_state)
        return self.sample(
            GeneratorState(checkpoint, base_state, guidance_state),
            request,
            progress,
        )

    def release_cases(self, case_ids: tuple[str, ...]) -> None:
        checkpoint = GeneratorCheckpoint(self.info.program_id, None)
        self.release(GeneratorState(checkpoint, None), case_ids)

    def _backend(self) -> CoefficientFlowBackend:
        if self.backend is None:
            raise RuntimeError(self.missing_backend_message())
        return self.backend

    def missing_backend_message(self) -> str:
        return (
            f"{self.info.name} needs a concrete PDE backend before it can run."
        )


class CoefficientFlowGenerator(BackendGeneratorProgram):
    """Named base for basis/PCA/flow implementations such as NS."""


__all__ = [
    "BackendGeneratorProgram",
    "CoefficientFlowBackend",
    "CoefficientFlowGenerator",
]
