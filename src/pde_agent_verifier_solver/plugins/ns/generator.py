"""Product-facing description of the structured NS Generator.

The mature implementation runs as long cluster stages in campaign.py. This
class supplies the shared Program surface used by forms and case previews. Its
methods delegate to a backend so a future local backend can use the same
interface without putting Slurm concepts into the scientific contract.
"""

from __future__ import annotations

from typing import Protocol, Sequence

from pde_agent_verifier_solver.sdk import (
    CaseRecord,
    FormField,
    GeneratorCapabilities,
    PanelCard,
    ProgramInfo,
)

from ..coefficient_flow import CoefficientFlowBackend, CoefficientFlowGenerator


class NSGeneratorBackend(CoefficientFlowBackend, Protocol):
    """Concrete NS projection/PCA/Rectified-Flow numerical backend."""

    supports_guidance: bool
    supports_perturbations: bool

    def release_cases(self, case_ids: Sequence[str]) -> None: ...


class NSStructuredGenerator(CoefficientFlowGenerator):
    """PCA-coefficient rectified flow used by the NS paper experiment."""

    info = ProgramInfo(
        "ns-structured-coefficient-flow",
        "NS coefficient Rectified Flow",
        "Projects velocity-pressure fields to a continuous basis, learns a "
        "latent flow, then supports solver-error guidance and perturbations.",
    )
    capabilities = GeneratorCapabilities(
        offline=True,
        online=False,
        guidance=True,
        perturbations=True,
    )

    def __init__(self, backend: NSGeneratorBackend | None = None) -> None:
        super().__init__(backend)
        # An unbound registry entry must not claim it can run online. Once a
        # user explicitly injects either the numerical or completed-PIV
        # backend, advertise exactly that backend's canonical capabilities.
        if backend is not None:
            declared = type(self).capabilities
            self.capabilities = GeneratorCapabilities(
                offline=True,
                online=True,
                guidance=(
                    getattr(backend, "supports_guidance", declared.guidance)
                    is True
                ),
                perturbations=(
                    getattr(
                        backend,
                        "supports_perturbations",
                        declared.perturbations,
                    )
                    is True
                ),
                trainable=(
                    getattr(backend, "trainable", declared.trainable) is True
                ),
            )

    def export_runtime_state(self, directory):
        from .resume import export_runtime_state
        return export_runtime_state(self, directory)

    def restore_runtime_state(self, payload, directory):
        from .resume import restore_runtime_state
        return restore_runtime_state(self, payload, directory)

    def export_generator_state(self, state, directory):
        from .resume import export_generator_state
        return export_generator_state(self, state, directory)

    def restore_generator_state(self, payload, directory):
        from .resume import restore_generator_state
        return restore_generator_state(self, payload, directory)

    def export_case_record(self, record, directory):
        from .resume import export_case_record
        return export_case_record(self, record, directory)

    def restore_case_record(self, payload, directory):
        from .resume import restore_case_record
        return restore_case_record(self, payload, directory)

    def setup_fields(self) -> tuple[FormField, ...]:
        return (
            FormField(
                "spatial_modes_x",
                "Spatial basis modes (x)",
                "integer",
                16,
            ),
            FormField(
                "spatial_modes_y",
                "Spatial basis modes (y)",
                "integer",
                16,
            ),
            FormField("time_order", "Time basis order", "integer", 12),
            FormField("pca_rank", "PCA latent dimension", "integer", 256),
            FormField(
                "flow_steps",
                "Base Generator training steps",
                "integer",
                30_000,
            ),
            FormField(
                "guidance_enabled",
                "Guided generation",
                "boolean",
                True,
            ),
            FormField(
                "perturbations_enabled",
                "Structured perturbations",
                "boolean",
                True,
            ),
        )

    def preview(self, record: CaseRecord) -> PanelCard:
        return PanelCard(
            title=f"Generated NS case {record.case.case_id}",
            summary=(
                "A continuous velocity-pressure answer and its derived forcing."
            ),
            values={
                "source": record.case.source.value,
                "round": record.case.round_index,
                "tags": record.case.tags,
            },
        )

    def missing_backend_message(self) -> str:
        return (
            "The canonical NS product run needs an NSGeneratorBackend injected "
            "into CampaignController. Historical staged reproduction remains "
            "available explicitly through './pde legacy-run <ns-config>'."
        )
