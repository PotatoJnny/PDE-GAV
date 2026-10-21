"""A real, fast 1D Poisson product slice.

The equation family is ``-u_xx = (k*pi)^2 a sin(k*pi*x)`` on ``[0, 1]`` with
zero boundary conditions. It is simple enough to run instantly while still
exercising coarse/noisy data, continuous generated answers, guidance,
perturbations, an Agent-facing Evaluator, hidden tests, and an evolving solver.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite, pi, sin, sqrt
from random import Random
from statistics import fmean
from typing import Mapping, Sequence

from pde_agent_verifier_solver.engine.evaluator import EvaluationRequest
from pde_agent_verifier_solver.engine.runtime import AgentContext, AgentOutcome
from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    AnswerView,
    Artifact,
    ArtifactEvaluation,
    Case,
    CaseAccessKind,
    CaseEvaluation,
    CaseRecord,
    CaseSource,
    CaseVisibility,
    DatasetInput,
    DatasetKind,
    EvaluationPhase,
    EvaluationPlan,
    FormField,
    GenerationKind,
    GenerationRequest,
    GeneratorCapabilities,
    GeneratorCheckpoint,
    GeneratorFeedback,
    GeneratorState,
    MultiplicativePerturbationSettings,
    PanelCard,
    PerturbationValidation,
    PreparedDataset,
    ProgramInfo,
    QueryKind,
    RelativeChangeRange,
    RoundFeedback,
    SampleCasesRequest,
    SampleSource,
    SamplingStrategy,
    TaskKind,
    aggregate_case_evaluations,
    sample_multiplicative_mms_perturbation,
)


@dataclass(frozen=True)
class SineTruth:
    amplitude: float
    frequency: int

    def answer(self, query: AnswerQuery) -> AnswerView:
        if set(query.fields) != {"u"}:
            raise ValueError("Toy Poisson exposes the field 'u'")
        if query.kind is QueryKind.GRID:
            count = query.resolution.get("x")
            if count is None:
                raise ValueError("Toy Poisson grid queries require an x resolution")
            xs = (0.5,) if count == 1 else tuple(i / (count - 1) for i in range(count))
            points: tuple[Mapping[str, float], ...] = ()
            axes = {"x": xs}
        else:
            xs = tuple(float(point["x"]) for point in query.points)
            points = query.points
            axes = {}
        values = tuple(self.amplitude * sin(self.frequency * pi * x) for x in xs)
        return AnswerView(
            kind=query.kind,
            fields={"u": values},
            axes=axes,
            points=points,
            shape=(len(xs),),
        )


class ToyPoissonProblem:
    info = ProgramInfo(
        "toy-poisson-1d",
        "Toy 1D Poisson",
        "A sine-forced Poisson family used to exercise the complete product flow.",
    )

    def setup_fields(self) -> tuple[FormField, ...]:
        return (
            FormField("domain", "Geometry", "text", "x in [0, 1]"),
            FormField("development_cases", "Initial development cases", "integer", 12),
            FormField("hidden_cases", "Initial hidden cases", "integer", 4),
        )

    def prepare_dataset(self, dataset: DatasetInput) -> PreparedDataset:
        development_count = int(dataset.settings.get("development_cases", 12))
        hidden_count = int(dataset.settings.get("hidden_cases", 4))
        if development_count <= 0 or hidden_count <= 0:
            raise ValueError("Toy dataset needs positive development and hidden case counts")

        development = tuple(
            self._initial_record(dataset.kind, index, hidden=False)
            for index in range(development_count)
        )
        hidden = tuple(
            self._initial_record(dataset.kind, index, hidden=True)
            for index in range(hidden_count)
        )
        return PreparedDataset(
            dataset=dataset,
            development_cases=development,
            hidden_cases=hidden,
            training_summary={
                "dataset_kind": dataset.kind.value,
                "development_cases": development_count,
                "hidden_cases": hidden_count,
                "representation": "sine coefficients",
            },
        )

    def case_card(self, case: Case) -> PanelCard:
        return PanelCard(
            title=f"Poisson case {case.case_id}",
            summary=f"Mode {case.public_data['frequency']} on x in [0, 1]",
            values={
                "source": case.source.value,
                "frequency": case.public_data["frequency"],
                "amplitude": case.public_data["amplitude"],
            },
        )

    def _initial_record(
        self,
        kind: DatasetKind,
        index: int,
        *,
        hidden: bool,
    ) -> CaseRecord:
        frequency = (5 + index) if hidden else (1 + index % 5)
        amplitude = 0.75 + 0.1 * (index % 5)
        truth = SineTruth(amplitude, frequency)
        observation_x = tuple(i / 8 for i in range(9))
        exact = tuple(amplitude * sin(frequency * pi * x) for x in observation_x)
        if kind is DatasetKind.NUMERICAL:
            observed = tuple(value * 0.985 for value in exact)
            observation_kind = "coarse_numerical_field"
            reference_kind = "high_accuracy_reference"
        else:
            observed = tuple(
                value + 0.015 * sin((index + 2) * (point_index + 1))
                for point_index, value in enumerate(exact)
            )
            observation_kind = "noisy_experimental_observation"
            reference_kind = "held_out_reference"

        prefix = "hidden" if hidden else "dev"
        case = self._case(
            case_id=f"toy-initial-{prefix}-{index + 1}",
            amplitude=amplitude,
            frequency=frequency,
            source=CaseSource.INITIAL,
            visibility=CaseVisibility.HIDDEN if hidden else CaseVisibility.DEVELOPMENT,
            round_index=None,
            parent_case_id=None,
            extra={
                "observation_kind": observation_kind,
                "observation_x": observation_x,
                "observed_u": observed,
                "reference_kind": reference_kind,
            },
        )
        return CaseRecord(case, truth)

    @staticmethod
    def _case(
        *,
        case_id: str,
        amplitude: float,
        frequency: int,
        source: CaseSource,
        visibility: CaseVisibility,
        round_index: int | None,
        parent_case_id: str | None,
        extra: Mapping[str, object] | None = None,
        tags: tuple[str, ...] = (),
    ) -> Case:
        public_data: dict[str, object] = {
            "equation": "-u_xx = f, u(0)=u(1)=0",
            "geometry": {"x": [0.0, 1.0]},
            "frequency": frequency,
            "amplitude": amplitude,
            "forcing_scale": (frequency * pi) ** 2 * amplitude,
        }
        public_data.update(extra or {})
        return Case(
            case_id=case_id,
            source=source,
            visibility=visibility,
            public_data=public_data,
            round_index=round_index,
            parent_case_id=parent_case_id,
            tags=tags,
        )


@dataclass(frozen=True)
class ToyBaseGeneratorState:
    max_frequency: int
    mean_amplitude: float
    basis_size: int


@dataclass(frozen=True)
class ToyGuidanceState:
    target_frequency: int
    target_amplitude: float


@dataclass(frozen=True)
class _ToyRawMultiplicativeSolution:
    """Pointwise multiplicative proposal before zero-boundary projection."""

    frequency: int
    sine_amplitude: float
    constant_shift: float


def _compose_toy_multiplicative_solution(
    parent: SineTruth,
    direction: float,
    alpha: float,
    epsilon_fraction: float,
) -> _ToyRawMultiplicativeSolution:
    """Apply the agreed formula analytically to the Toy sine solution.

    The Toy random field is the smooth constant field named direction.
    Applying the epsilon-shifted multiplicative formula creates a sine term
    plus a constant term. The following admissibility callback projects that
    proposal back to the Toy zero-boundary solution family before the helper
    measures its final physical change.
    """

    reference_scale = max(abs(parent.amplitude), 1.0)
    return _ToyRawMultiplicativeSolution(
        frequency=parent.frequency,
        sine_amplitude=parent.amplitude * (1.0 + alpha * direction),
        constant_shift=(
            alpha * epsilon_fraction * reference_scale * direction
        ),
    )


def _project_toy_zero_boundary(
    proposal: _ToyRawMultiplicativeSolution,
) -> SineTruth:
    """L2-project the raw proposal onto its original sine mode."""

    frequency = proposal.frequency
    constant_to_sine = (
        2.0 * (1.0 - (-1.0) ** frequency) / (frequency * pi)
    )
    return SineTruth(
        amplitude=(
            proposal.sine_amplitude
            + proposal.constant_shift * constant_to_sine
        ),
        frequency=frequency,
    )


def _toy_solution_relative_change(parent: SineTruth, child: SineTruth) -> float:
    return abs(child.amplitude - parent.amplitude) / max(
        abs(parent.amplitude),
        1.0e-12,
    )


def _manufacture_toy_problem(solution: SineTruth) -> Mapping[str, float]:
    return {
        "amplitude": solution.amplitude,
        "forcing_scale": (solution.frequency * pi) ** 2 * solution.amplitude,
    }


def _validate_toy_problem(
    parent: SineTruth,
    child: SineTruth,
    problem: Mapping[str, float],
) -> PerturbationValidation:
    parent_forcing = (parent.frequency * pi) ** 2 * parent.amplitude
    forcing = float(problem["forcing_scale"])
    forcing_change = abs(forcing - parent_forcing) / max(
        abs(parent_forcing),
        1.0e-12,
    )
    accepted = (
        child.frequency == parent.frequency
        and child.amplitude > 0.0
        and isfinite(child.amplitude)
        and isfinite(forcing)
    )
    return PerturbationValidation(
        accepted,
        "" if accepted else "Toy MMS produced an invalid sine problem",
        metrics={
            "boundary_error": 0.0,
            "forcing_relative_change": forcing_change,
        },
    )


class ToyPoissonGenerator:
    info = ProgramInfo(
        "toy-sine-generator",
        "Toy continuous sine Generator",
        "Learns a sine coefficient distribution and shifts toward solver weaknesses.",
    )
    capabilities = GeneratorCapabilities(
        offline=True,
        online=True,
        guidance=True,
        perturbations=True,
    )

    def setup_fields(self) -> tuple[FormField, ...]:
        return (
            FormField("basis_size", "Basis size", "integer", 8),
            FormField("guidance_enabled", "Guided generation", "boolean", True),
            FormField("perturbations_enabled", "Perturbed cases", "boolean", True),
            FormField(
                "perturbation_relative_change_min",
                "Minimum solution change",
                "number",
                0.05,
                help_text="Smallest accepted relative change from root to child.",
            ),
            FormField(
                "perturbation_relative_change_max",
                "Maximum solution change",
                "number",
                0.10,
                help_text="Largest accepted relative change from root to child.",
            ),
            FormField(
                "perturbation_epsilon_fraction",
                "Multiplicative epsilon",
                "number",
                0.001,
                help_text="Dimensionless offset fraction used near solution zeros.",
            ),
            FormField(
                "perturbation_maximum_alpha",
                "Maximum alpha",
                "number",
                10.0,
                help_text="Reject a random field if it needs a larger multiplier.",
            ),
        )

    def train(
        self,
        dataset: PreparedDataset,
        settings: Mapping[str, object],
        progress=None,
    ) -> GeneratorCheckpoint:
        """Train a fresh base Generator and return its reusable checkpoint."""

        training_cases = (
            dataset.generator_training_cases
            if dataset.generator_training_cases
            else dataset.development_cases
        )
        frequencies = [
            (
                record.generator_target.frequency
                if isinstance(record.generator_target, SineTruth)
                else int(record.case.public_data["frequency"])
            )
            for record in training_cases
        ]
        amplitudes = [
            (
                record.generator_target.amplitude
                if isinstance(record.generator_target, SineTruth)
                else float(record.case.public_data["amplitude"])
            )
            for record in training_cases
        ]
        if progress:
            progress("Mapped dataset cases into the sine coefficient space", 0.5)
            progress("Base coefficient distribution is ready", 1.0)
        base = ToyBaseGeneratorState(
            max_frequency=max(frequencies),
            mean_amplitude=fmean(amplitudes),
            basis_size=int(settings.get("basis_size", 8)),
        )
        return GeneratorCheckpoint(
            generator_id=self.info.program_id,
            payload=base,
            report={
                "training_cases": len(training_cases),
                "training_role": (
                    "generator_training_cases"
                    if dataset.generator_training_cases
                    else "development_cases_compatibility"
                ),
                "basis_size": base.basis_size,
            },
        )

    def load(
        self,
        checkpoint: GeneratorCheckpoint,
        settings: Mapping[str, object],
        progress=None,
    ) -> GeneratorState:
        """Load one checkpoint into the state sampled by the Controller."""

        if checkpoint.generator_id != self.info.program_id:
            raise ValueError("checkpoint belongs to a different Generator")
        base = _expect_base_state(checkpoint.payload)
        if progress:
            progress("Generator checkpoint loaded", 1.0)
        return GeneratorState(checkpoint=checkpoint, base=base)

    def update_guidance(
        self,
        state: GeneratorState,
        feedback: GeneratorFeedback,
        settings: Mapping[str, object],
        progress=None,
    ) -> GeneratorState:
        """Return the same base Generator with next-round guidance attached."""

        base = _expect_base_state(state.base)
        worst = max(feedback.cases, key=lambda row: row.primary_error)
        target_frequency = min(
            base.basis_size,
            int(worst.case.public_data["frequency"]) + 1,
        )
        if progress:
            progress(
                f"Guidance is focusing on mode {target_frequency}",
                1.0,
            )
        guidance = ToyGuidanceState(
            target_frequency=target_frequency,
            target_amplitude=float(worst.case.public_data["amplitude"]),
        )
        return replace(state, guidance=guidance)

    def sample(
        self,
        generator_state: GeneratorState,
        request: GenerationRequest,
        progress=None,
    ) -> tuple[CaseRecord, ...]:
        """Sample base, guided, or anchored perturbed Cases."""

        state = _expect_base_state(generator_state.base)
        guidance = (
            generator_state.guidance
            if isinstance(generator_state.guidance, ToyGuidanceState)
            else None
        )
        # Every Controller request receives its own deterministic stream.  The
        # seed must affect the physical Case, not only its identifier; otherwise
        # a "fresh" promotion or terminal batch can silently repeat the same
        # equation with new names.
        seed_phase = request.seed % max(1, state.max_frequency)
        seed_amplitude_shift = 0.01 * (request.seed / float(2**31) - 0.5)
        records: list[CaseRecord] = []
        for index in range(request.count):
            parent: Case | None = None
            if request.kind is GenerationKind.BASE:
                frequency = 1 + (
                    (request.round_index + seed_phase + index - 1)
                    % state.max_frequency
                )
                amplitude = (
                    state.mean_amplitude
                    + 0.04 * (((index + request.seed) % 3) - 1)
                    + seed_amplitude_shift
                )
                source = CaseSource.GENERATED
            elif request.kind is GenerationKind.GUIDED:
                if guidance is None:
                    raise ValueError("guided generation requires guidance from a previous round")
                frequency = max(
                    1,
                    guidance.target_frequency - ((index + request.seed) % 2),
                )
                amplitude = (
                    guidance.target_amplitude
                    + 0.03 * (((index + request.seed) % 3) - 1)
                    + seed_amplitude_shift
                )
                source = CaseSource.GENERATED
            elif request.kind is GenerationKind.PERTURBED:
                if len(request.anchors) != request.count:
                    raise ValueError(
                        "perturbed generation needs one clean anchor per child"
                    )
                parent = request.anchors[index]
                frequency = int(parent.public_data["frequency"])
                parent_solution = SineTruth(
                    amplitude=float(parent.public_data["amplitude"]),
                    frequency=frequency,
                )
                result = sample_multiplicative_mms_perturbation(
                    parent_solution,
                    settings=MultiplicativePerturbationSettings(
                        target_relative_change=RelativeChangeRange(
                            float(
                                request.settings.get(
                                    "perturbation_relative_change_min",
                                    0.05,
                                )
                            ),
                            float(
                                request.settings.get(
                                    "perturbation_relative_change_max",
                                    0.10,
                                )
                            ),
                        ),
                        epsilon_fraction=float(
                            request.settings.get(
                                "perturbation_epsilon_fraction",
                                0.001,
                            )
                        ),
                        maximum_alpha=float(
                            request.settings.get(
                                "perturbation_maximum_alpha",
                                10.0,
                            )
                        ),
                    ),
                    seed=request.seed + index,
                    draw_random_field=lambda field_seed: (
                        -1.0 if Random(field_seed).random() < 0.5 else 1.0
                    ),
                    compose_candidate=_compose_toy_multiplicative_solution,
                    make_admissible=_project_toy_zero_boundary,
                    measure_relative_change=_toy_solution_relative_change,
                    manufacture_problem=_manufacture_toy_problem,
                    validate=_validate_toy_problem,
                )
                amplitude = result.solution.amplitude
                source = CaseSource.PERTURBED
            else:
                raise ValueError(f"unsupported generation kind {request.kind}")

            case = ToyPoissonProblem._case(
                case_id=(
                    f"toy-{request.purpose.value}-r{request.round_index}-"
                    f"{request.kind.value}-{request.seed}-{index + 1}"
                ),
                amplitude=amplitude,
                frequency=frequency,
                source=source,
                visibility=request.visibility,
                round_index=request.round_index,
                parent_case_id=parent.case_id if parent else None,
                extra={"generator_mode": request.mode.value},
                tags=(request.kind.value, request.mode.value),
            )
            records.append(CaseRecord(case, SineTruth(amplitude, frequency)))
        if progress:
            progress(
                f"Generated {request.count} {request.kind.value} cases",
                1.0,
            )
        return tuple(records)

    def open_case_session(
        self,
        state: GeneratorState,
        case: Case,
        access: CaseAccessKind,
    ) -> object | None:
        # Toy Cases already contain their complete public problem statement.
        return None

    def case_session_methods(
        self,
        access: CaseAccessKind,
    ) -> tuple[str, ...]:
        return ()

    def release(self, state: GeneratorState, case_ids: tuple[str, ...]) -> None:
        # The toy Generator keeps no equation-private resident case state.
        return None

    # Compatibility aliases for old callers. New code uses the explicit
    # train -> load -> update_guidance/sample lifecycle above.
    def train_base(self, dataset, settings, progress=None):
        return self.train(dataset, settings, progress).payload

    def fit_guidance(
        self,
        base_state,
        previous_guidance,
        feedback: RoundFeedback,
        settings,
        progress=None,
    ):
        checkpoint = GeneratorCheckpoint(self.info.program_id, base_state)
        state = GeneratorState(checkpoint, base_state, previous_guidance)
        updated_state = self.update_guidance(
            state,
            GeneratorFeedback(
                source_round_index=feedback.round_index,
                target_round_index=feedback.round_index + 1,
                selected_program_id=feedback.selected_artifact.artifact_id,
                selected_program=feedback.selected_artifact,
                cases=feedback.cases,
            ),
            settings,
            progress,
        )
        return updated_state.guidance

    def generate(self, base_state, guidance_state, request, progress=None):
        checkpoint = GeneratorCheckpoint(self.info.program_id, base_state)
        return self.sample(
            GeneratorState(checkpoint, base_state, guidance_state),
            request,
            progress,
        )

    def preview(self, record: CaseRecord) -> PanelCard:
        return PanelCard(
            title=f"Generated mode {record.case.public_data['frequency']}",
            summary="A continuous sine solution; the panel displays only sampled views.",
            values={
                "amplitude": record.case.public_data["amplitude"],
                "source": record.case.source.value,
                "visibility": record.case.visibility.value,
            },
        )


@dataclass(frozen=True)
class ToySineSolver:
    max_resolved_mode: int

    def solve(
        self,
        case: Case,
        query: AnswerQuery,
        budget: float | None = None,
    ) -> AnswerView:
        frequency = int(case.public_data["frequency"])
        amplitude = float(case.public_data["amplitude"])
        if frequency <= self.max_resolved_mode:
            represented_amplitude = amplitude
        else:
            represented_amplitude = amplitude * self.max_resolved_mode / frequency
        return SineTruth(represented_amplitude, frequency).answer(query)


def _evaluate_toy_solver(
    artifact: Artifact,
    case: Case,
    truth: AnswerView,
    query: AnswerQuery,
) -> CaseEvaluation:
    """Compute the Toy Task's per-Case scientific metrics."""

    solver = artifact.program
    if not hasattr(solver, "solve"):
        raise TypeError("Toy Solver Task expects an object with solve(case, query, budget)")
    prediction = solver.solve(case, query, budget=None)
    expected = truth.fields["u"]
    actual = prediction.fields["u"]
    if len(expected) != len(actual):
        raise ValueError("solver prediction and truth use different sample counts")
    squared_error = sum((left - right) ** 2 for left, right in zip(actual, expected))
    squared_truth = sum(value**2 for value in expected)
    relative_l2 = sqrt(squared_error / max(squared_truth, 1e-15))
    max_abs = max(abs(left - right) for left, right in zip(actual, expected))
    return CaseEvaluation(
        case_id=case.case_id,
        primary_error=relative_l2,
        metrics={"relative_l2": relative_l2, "max_abs_error": max_abs},
        prediction=prediction,
    )


class ToySolverEvaluation:
    """Toy Task-owned query, metric, and aggregation policy."""

    program_id = "toy-full-state-solver-evaluation"

    def __init__(self, default_query: AnswerQuery) -> None:
        self._default_query = default_query

    def make_plan(
        self,
        case: Case,
        phase: EvaluationPhase,
        settings: Mapping[str, object],
        seed: int,
        request: AnswerQuery | None = None,
    ) -> EvaluationPlan:
        del case
        return EvaluationPlan(
            phase=phase,
            query=request or self._default_query,
            seed=seed,
            settings=settings,
        )

    def make_private_development_plan(
        self,
        case: Case,
        settings: Mapping[str, object],
        seed: int,
    ) -> EvaluationPlan:
        """Draw a fresh Controller-owned finite point set on ``x in [0, 1]``."""

        del case
        count = settings.get("development_point_count", 32)
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("Toy development_point_count must be positive")
        random = Random(seed)
        query = AnswerQuery.at_points(
            fields=self._default_query.fields,
            points=tuple({"x": random.random()} for _index in range(count)),
        )
        return EvaluationPlan(
            phase=EvaluationPhase.DEVELOPMENT,
            query=query,
            seed=seed,
            settings=settings,
        )

    def evaluate(
        self,
        artifact: Artifact,
        case: Case,
        truth: AnswerView,
        plan: EvaluationPlan,
    ) -> CaseEvaluation:
        return _evaluate_toy_solver(artifact, case, truth, plan.query)

    def aggregate(
        self,
        artifact: Artifact,
        case_results: Sequence[CaseEvaluation],
        plans: Sequence[EvaluationPlan],
    ) -> ArtifactEvaluation:
        return aggregate_case_evaluations(artifact, case_results, plans)


class ToySolverTask:
    info = ProgramInfo(
        "toy-full-state-solver",
        "Find a full-field Poisson solver",
        "Return u on any requested x grid or point set.",
    )
    kind = TaskKind.SOLVER

    def setup_fields(self) -> tuple[FormField, ...]:
        return (FormField("primary_metric", "Primary metric", "text", "relative_l2"),)

    def agent_brief(self, settings: Mapping[str, object]) -> str:
        return (
            "Build a solver with solve(case, query, budget=None) -> AnswerView. "
            "Use the Evaluator on development cases; hidden cases are unavailable."
        )

    def create_evaluation(
        self,
        settings: Mapping[str, object],
    ) -> ToySolverEvaluation:
        query = settings.get("default_query")
        if not isinstance(query, AnswerQuery):
            raise TypeError("Toy Solver Task needs an AnswerQuery default_query")
        return ToySolverEvaluation(query)

    def evaluate(
        self,
        artifact: Artifact,
        case: Case,
        truth: AnswerView,
        query: AnswerQuery,
    ) -> CaseEvaluation:
        """Compatibility entrypoint for callers not yet using EvaluationProgram."""

        return _evaluate_toy_solver(artifact, case, truth, query)

    def select(
        self,
        candidates: Sequence[ArtifactEvaluation],
        incumbent: ArtifactEvaluation | None,
    ) -> ArtifactEvaluation:
        choices = tuple(candidates) + ((incumbent,) if incumbent else ())
        return min(choices, key=lambda evaluation: evaluation.primary_error)

    def result_card(self, evaluation: ArtifactEvaluation) -> PanelCard:
        return PanelCard(
            title=evaluation.artifact.name,
            summary="Recommended full-field solver from the multi-round run.",
            values={
                "hidden_relative_l2": evaluation.primary_error,
                "resolved_modes": evaluation.artifact.metadata["max_resolved_mode"],
                "usage": evaluation.artifact.usage,
            },
        )


class ToyResearchAgent:
    """A local stand-in for Codex that genuinely uses the Evaluator contract."""

    def __init__(self, step_size: int = 1) -> None:
        self.step_size = step_size

    def run(self, context: AgentContext) -> AgentOutcome:
        incumbent_mode = (
            int(context.incumbent.metadata["max_resolved_mode"])
            if context.incumbent is not None
            else 0
        )
        policy = context.round_plan.development_sampling
        if (
            context.sampler is not None
            and policy is not None
            and policy.strategy is SamplingStrategy.AGENT_ADAPTIVE
        ):
            # This toy Agent chooses four probes per source even when the
            # environment imposes no cumulative quota.
            remaining = (4 if policy.max_cases_per_agent_per_round is None else
                         policy.max_cases_per_agent_per_round - len(context.evaluator.cases))
            if remaining and SampleSource.BASE in policy.allowed_sources:
                context.sampler.sample_cases(
                    SampleCasesRequest(SampleSource.BASE, min(4, remaining))
                )
            # This toy Agent chooses four probes per source even when the
            # environment imposes no cumulative quota.
            remaining = (4 if policy.max_cases_per_agent_per_round is None else
                         policy.max_cases_per_agent_per_round - len(context.evaluator.cases))
            if (
                remaining
                and context.round_plan.round_index > 1
                and SampleSource.GUIDED in policy.allowed_sources
            ):
                context.sampler.sample_cases(
                    SampleCasesRequest(SampleSource.GUIDED, min(4, remaining))
                )
        cases = context.evaluator.cases
        if not cases:
            raise RuntimeError("the Toy Agent needs at least one development Case")
        probe_mode = max(1, incumbent_mode)
        probe = self._artifact(context, probe_mode, suffix="probe")
        response = context.evaluator.evaluate(
            probe,
            EvaluationRequest(
                case_ids=tuple(case.case_id for case in cases),
            ),
        )
        worst_frequency = max(int(case.public_data["frequency"]) for case in cases)
        next_mode = min(worst_frequency, max(1, incumbent_mode + self.step_size))
        candidate = self._artifact(context, next_mode, suffix="candidate")
        return AgentOutcome(
            artifact=candidate,
            observations=(
                f"Development aggregate error was {response.primary_error:.6g}.",
                f"The acquired public Cases include frequency {worst_frequency}.",
                f"Expanded the solver from mode {incumbent_mode} to mode {next_mode}.",
            ),
            round_log=(
                f"# Round {context.round_plan.round_index} — "
                f"{context.agent.name}\n\n"
                "## Round goal\n\n"
                "Improve the range of forcing modes handled by the solver.\n\n"
                "## Experiments\n\n"
                f"- Probed mode {probe_mode} on {len(cases)} development Cases.\n"
                f"- Evaluator aggregate error: {response.primary_error:.6g}.\n"
                f"- Highest observed frequency: {worst_frequency}.\n\n"
                "## Final submission\n\n"
                f"- Expanded resolved modes from {incumbent_mode} to {next_mode}.\n\n"
                "## Handoff\n\n"
                "- Continue testing the first unresolved public frequency."
            ),
        )

    @staticmethod
    def _artifact(context: AgentContext, mode: int, *, suffix: str) -> Artifact:
        artifact_id = (
            f"toy-r{context.round_plan.round_index}-{context.agent.agent_id}-m{mode}-{suffix}"
        )
        return Artifact(
            artifact_id=artifact_id,
            kind=TaskKind.SOLVER,
            name=f"Sine solver (modes 1-{mode})",
            program=ToySineSolver(mode),
            summary=f"Resolves sine forcing modes through k={mode}.",
            usage="solver.solve(case, AnswerQuery.grid(fields=('u',), x=129))",
            round_index=context.round_plan.round_index,
            agent_id=context.agent.agent_id,
            parent_artifact_id=context.incumbent.artifact_id if context.incumbent else None,
            metadata={"max_resolved_mode": mode},
        )


def _expect_base_state(state: object) -> ToyBaseGeneratorState:
    if not isinstance(state, ToyBaseGeneratorState):
        raise TypeError("Toy Generator received an incompatible base state")
    return state


def build_toy_plugin():
    """Expose Toy Poisson through the registry used by real PDEs."""

    from pde_agent_verifier_solver.plugins.registry import PDEPlugin

    return PDEPlugin(
        plugin_id="toy",
        title="Toy Poisson",
        description="Fast local example of every shared product contract.",
        aliases=("poisson", "toy-poisson"),
        problem=ToyPoissonProblem(),
        generator=ToyPoissonGenerator(),
        tasks={"solver": ToySolverTask()},
        default_task="solver",
    )
