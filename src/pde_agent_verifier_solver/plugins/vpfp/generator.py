"""Product-facing VPFP continuous Generator.

The paper implementation is an online Fourier--Hermite answer-first service.
Long cluster runs use campaign.py; this class keeps the scientific interface
readable and permits a local backend to be injected for small experiments.
"""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence

from pde_agent_verifier_solver.sdk import (
    Case,
    CaseRecord,
    FormField,
    GeneratorCapabilities,
    PanelCard,
    ProgramInfo,
)

from ..coefficient_flow import BackendGeneratorProgram, CoefficientFlowBackend


class VPFPGeneratorBackend(CoefficientFlowBackend, Protocol):
    """Concrete Fourier-Hermite/latent-flow numerical backend."""

    supports_perturbations: bool
    supports_guidance: bool
    public_session_methods: tuple[str, ...]
    development_solution_session_methods: tuple[str, ...]

    def public_session_factory(self, case: Case) -> object: ...

    def development_solution_session_factory(self, case: Case) -> object: ...

    def release_cases(self, case_ids: Sequence[str]) -> None: ...


class VPFPContinuousGenerator(BackendGeneratorProgram):
    """Continuous positive Fourier--Hermite VPFP answer Generator."""

    info = ProgramInfo(
        "vpfp-continuous-answer-first",
        "VPFP continuous online Generator",
        "Generates continuous phase-space answers, derives the public equation, "
        "and can focus fresh samples using solver error.",
    )
    capabilities = GeneratorCapabilities(
        offline=True,
        online=True,
        guidance=True,
        perturbations=False,
        trainable=False,
    )

    def __init__(
        self,
        backend: VPFPGeneratorBackend | None = None,
    ) -> None:
        super().__init__(backend)
        self.capabilities = GeneratorCapabilities(
            offline=True,
            online=True,
            guidance=(
                getattr(backend, "supports_guidance", True) is True
            ),
            perturbations=(
                getattr(backend, "supports_perturbations", False) is True
            ),
            trainable=False,
        )

    def setup_fields(self) -> tuple[FormField, ...]:
        return (
            FormField(
                "representation",
                "Continuous representation",
                "text",
                "fourier_hermite_positive",
            ),
            FormField(
                "latent_rank",
                "Latent dimension",
                "integer",
                256,
            ),
            FormField(
                "generation_batch_size",
                "Online generation batch",
                "integer",
                16,
            ),
            FormField(
                "guidance_pool_size",
                "Guided candidate pool",
                "integer",
                4,
                "Fresh candidates per accepted guided answer.",
            ),
            FormField(
                "guidance_enabled",
                "Guided generation",
                "boolean",
                self.capabilities.guidance,
            ),
            FormField(
                "perturbations_enabled",
                "Structure-preserving perturbations",
                "boolean",
                self.capabilities.perturbations,
            ),
        )

    def preview(self, record: CaseRecord) -> PanelCard:
        question = record.case.public_data.get("question", {})
        return PanelCard(
            title=f"Generated VPFP case {record.case.case_id}",
            summary=(
                "Continuous kinetic answer; the panel shows finite "
                "distribution slices and exact moments."
            ),
            values={
                "source": record.case.source.value,
                "round": record.case.round_index,
                "collision_frequency": (
                    question.get("collision_nu")
                    if isinstance(question, Mapping)
                    else None
                ),
            },
        )

    def missing_backend_message(self) -> str:
        return (
            "The canonical VPFP product run needs a VPFPGeneratorBackend "
            "injected into CampaignController. Historical staged reproduction "
            "remains available explicitly through './pde legacy-run "
            "<vpfp-config>'."
        )
