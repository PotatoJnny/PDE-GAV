"""Canonical Generator lifecycle around the admitted resident Artemis source."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from pde_agent_verifier_solver.plugins.coefficient_flow import BackendGeneratorProgram
from pde_agent_verifier_solver.sdk import (
    CaseVisibility, FormField, GenerationKind, GenerationRequest,
    GeneratorCapabilities, GeneratorCheckpoint, PanelCard, ProgramInfo,
)
from .public_problem import VP1D_PUBLIC_SESSION_METHODS, VP1D_DEVELOPMENT_SOLUTION_SESSION_METHODS
from .concurrency import LocalArtemisSession, artemis_ad_compute, serialized_artemis_compute


@dataclass(frozen=True)
class ArtemisCheckpoint:
    artifact_sha256: str
    physical_revision: str
    admission_sha256: str | None = None


class ArtemisBackend:
    """Translate requests only; the pinned source owns physics and admission."""

    public_session_methods = VP1D_PUBLIC_SESSION_METHODS
    development_solution_session_methods = VP1D_DEVELOPMENT_SOLUTION_SESSION_METHODS

    def __init__(self, source, *, physical_revision: str, admission_sha256=None, guidance_adapter=None):
        for method in ("sample", "public_session_factory", "development_solution_session_factory", "release_cases"):
            if not callable(getattr(source, method, None)):
                raise TypeError(f"Artemis source requires {method}")
        self.source = source
        self.checkpoint_payload = ArtemisCheckpoint(source.artifact_sha256, physical_revision, admission_sha256)
        self.supports_perturbations = source.supports_perturbations is True
        self.supports_guidance = guidance_adapter is not None
        self.guidance_adapter = guidance_adapter
        self._parallel_oracle_pool = None
        self.continuous_macro_truth = False
        if guidance_adapter is not None and any(not callable(getattr(guidance_adapter, name, None)) for name in ("fit_guidance", "generate")):
            raise TypeError("Artemis guidance adapter requires fit_guidance and generate")
        self.public_oracle_mode = source.public_oracle_mode

    def checkpoint(self):
        return GeneratorCheckpoint(
            ArtemisGenerator.info.program_id, self.checkpoint_payload,
            report={"artifact_sha256": self.source.artifact_sha256,
                    "physical_revision": self.checkpoint_payload.physical_revision,
                    "admission_sha256": self.checkpoint_payload.admission_sha256},
        )

    def train_base(self, dataset, settings, progress=None):
        raise ValueError("Artemis production Generator is frozen; load its admitted checkpoint")

    @serialized_artemis_compute
    def fit_guidance(self, base_state, previous_guidance, feedback, settings, progress=None):
        self._require_checkpoint(base_state)
        if self.guidance_adapter is None:
            raise ValueError("Artemis guidance requires an explicitly bound guidance adapter")
        return self.guidance_adapter.fit_guidance(base_state, previous_guidance, feedback, settings, progress)

    @serialized_artemis_compute
    def generate(self, base_state, guidance_state, request: GenerationRequest, progress=None):
        self._require_checkpoint(base_state)
        expected = CaseVisibility.DEVELOPMENT if request.purpose.value == "development" else CaseVisibility.HIDDEN
        # Development screens remain development records in the canonical engine.
        if request.purpose.value == "development_screen":
            expected = CaseVisibility.DEVELOPMENT
        if request.visibility is not expected:
            raise ValueError("Artemis request purpose and visibility disagree")
        if request.kind is GenerationKind.GUIDED:
            if self.guidance_adapter is None or guidance_state is None:
                raise ValueError("Artemis guided generation requires fitted guidance")
            return self._truth_records(self.guidance_adapter.generate(base_state, guidance_state, request, progress))
        # Only the explicitly bound old physical source needs these request
        # types. Plugin discovery must not import a legacy lifecycle surface.
        from pde_agent_verifier_solver.sdk.macro_loop import CohortKind, SampleRequest
        cohort = CohortKind(request.purpose.value)
        if request.visibility is CaseVisibility.DEVELOPMENT:
            cohort = CohortKind.DEVELOPMENT
        result = tuple(self.source.sample(SampleRequest(
            round_index=request.round_index, cohort=cohort, seed=request.seed,
            count=request.count, kind=request.kind, anchors=request.anchors,
            settings=dict(request.settings),
        )))
        capture = getattr(self.guidance_adapter, "retain_feedback_coordinates", None)
        if callable(capture):
            try:
                capture(result)
            except BaseException:
                self.release_cases(tuple(record.case.case_id for record in result))
                raise
        if progress:
            progress("Artemis cases generated", 1.0)
        return self._truth_records(result)

    def _truth_records(self, records):
        if not self.continuous_macro_truth:
            return tuple(records)
        from .continuous_truth import continuous_record
        return tuple(continuous_record(self, record) for record in records)

    def public_session_factory(self, case):
        if self._parallel_oracle_pool is not None:
            return self._parallel_oracle_pool.session(case)
        with artemis_ad_compute():
            return LocalArtemisSession(self.source.public_session_factory(case))

    @serialized_artemis_compute
    def development_solution_session_factory(self, case):
        return LocalArtemisSession(self.source.development_solution_session_factory(case))

    @serialized_artemis_compute
    def release_cases(self, case_ids):
        case_ids = tuple(case_ids)
        release = getattr(self.guidance_adapter, "release_feedback_coordinates", None)
        if callable(release):
            release(case_ids)
        self.source.release_cases(tuple(case_ids))

    def _require_checkpoint(self, value):
        if not isinstance(value, ArtemisCheckpoint) or value != self.checkpoint_payload:
            raise ValueError("Artemis checkpoint does not belong to this resident backend")


class ArtemisGenerator(BackendGeneratorProgram):
    info = ProgramInfo(
        "vp-1d1v-artemis-compact-v5-answer-first", "Artemis admitted continuous Generator",
        "Sample admitted kinetic answers and paired perturbations with exact public oracles.",
    )

    def __init__(self, backend: ArtemisBackend | None = None):
        super().__init__(backend)
        self.capabilities = GeneratorCapabilities(
            offline=False, online=True, trainable=False,
            guidance=bool(backend is not None and backend.supports_guidance),
            perturbations=bool(backend is not None and backend.supports_perturbations),
        )

    def setup_fields(self):
        return (
            FormField("guidance_enabled", "Guided generation", "boolean", self.capabilities.guidance),
            FormField("perturbations_enabled", "Paired perturbations", "boolean", self.capabilities.perturbations),
        )

    def public_session_operation_lock(self, state, case):
        """Only an active hidden-evaluation scope can replace the global lock."""
        pool = self._backend()._parallel_oracle_pool
        return None if pool is None else pool.operation_lock(case.case_id)

    def load(self, checkpoint, settings, progress=None):
        state = super().load(checkpoint, settings, progress)
        self._backend()._require_checkpoint(state.base)
        return state

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

    def preview(self, record):
        return PanelCard(title=f"Artemis case {record.case.case_id}",
                         summary="Continuous kinetic answer with M0--M3 and E.",
                         values={"source": record.case.source.value})

    def missing_backend_message(self):
        return "Bind the admitted Artemis runtime with load_artemis_backend before running this Generator."
