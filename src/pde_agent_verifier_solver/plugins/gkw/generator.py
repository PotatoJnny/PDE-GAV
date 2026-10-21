"""Product-facing boundary for the stable GKW answer-first Generator."""

from __future__ import annotations

from typing import Mapping, Protocol

from pde_agent_verifier_solver.sdk import (
    CaseRecord,
    FormField,
    GeneratorCapabilities,
    PanelCard,
    ProgramInfo,
)

from ..coefficient_flow import BackendGeneratorProgram, CoefficientFlowBackend
from .selected_torch_runtime import GKW_SOLUTION_SAMPLING_MODES


class GKWGeneratorBackend(CoefficientFlowBackend, Protocol):
    """Concrete GKW answer-first numerical backend."""

    public_session_methods: tuple[str, ...]
    development_solution_session_methods: tuple[str, ...]

    @property
    def supports_perturbations(self) -> bool: ...

    def release_cases(self, case_ids: tuple[str, ...]) -> None: ...

    def public_session_factory(self, case: object) -> object: ...

    def development_solution_session_factory(self, case: object) -> object: ...



class GKWAnswerFirstGenerator(BackendGeneratorProgram):
    """Answer-first GKW Generator with capabilities owned by its backend."""

    info = ProgramInfo(
        "gkw-continuous-answer-first",
        "GKW continuous answer-first Generator",
        "Generates a continuous ion-state answer, then derives the public GKW "
        "equation and retains only admissible completed cases.",
    )

    def __init__(self, backend: GKWGeneratorBackend | None = None) -> None:
        super().__init__(backend)

    @property
    def capabilities(self) -> GeneratorCapabilities:
        return GeneratorCapabilities(
            offline=True,
            online=True,
            guidance=(
                self.backend is not None
                and getattr(self.backend, "supports_guidance", False) is True
            ),
            perturbations=(
                self.backend is not None
                and getattr(self.backend, "supports_perturbations", False) is True
            ),
            trainable=False,
        )

    def setup_fields(self) -> tuple[FormField, ...]:
        return (
            FormField(
                "generator_sampling_mode",
                "Solution sampling mode",
                "choice",
                "local_edit",
                "Direct noise is experimental and uses separate compatibility authority.",
                choices=tuple(
                    (mode, mode.replace("_", " ").title())
                    for mode in GKW_SOLUTION_SAMPLING_MODES
                ),
            ),
            FormField(
                "interaction_protocol",
                "Interaction protocol",
                "text",
                ("reward_surrogate_finite_pool_resampling" if self.capabilities.guidance
                 else "unguided_base_diffusion_v1"),
            ),
            FormField(
                "decoder_precision",
                "Audited decoder precision",
                "text",
                "bfloat16",
            ),
            FormField("generation_batch_size", "Generation batch", "integer", 8),
            FormField(
                "perturbations_enabled",
                "Smooth physical perturbations",
                "boolean",
                self.capabilities.perturbations,
            ),
        )

    def release_cases(self, case_ids: tuple[str, ...]) -> None:
        """Release controller-private answers after their cohort is destroyed."""

        release = getattr(self._backend(), "release_cases", None)
        if callable(release):
            release(tuple(case_ids))

    def public_session_factory(self, case):
        """Open the equation-owned public oracle for an issued case."""

        factory = getattr(self._backend(), "public_session_factory", None)
        if not callable(factory):
            raise RuntimeError("the GKW backend has no public numerical oracle")
        return factory(case)

    def development_solution_session_factory(self, case):
        """Open a DEVELOPMENT-only completed-solution numerical session."""

        factory = getattr(
            self._backend(),
            "development_solution_session_factory",
            None,
        )
        if not callable(factory):
            raise RuntimeError(
                "the GKW backend has no completed-solution numerical oracle"
            )
        return factory(case)

    def preview(self, record: CaseRecord) -> PanelCard:
        macro_query = record.case.public_data.get("macro_query")
        output_contract = (
            macro_query.get("output_contract")
            if isinstance(macro_query, Mapping)
            else None
        )
        return PanelCard(
            title=f"Generated GKW case {record.case.case_id}",
            summary="Continuous hidden ion state with finite public macro views.",
            values={
                "source": record.case.source.value,
                "round": record.case.round_index,
                "output_contract": output_contract,
            },
        )

    def missing_backend_message(self) -> str:
        return (
            "GKW Generator execution requires an injected stable backend."
        )
