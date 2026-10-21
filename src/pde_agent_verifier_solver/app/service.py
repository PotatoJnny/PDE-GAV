"""Project and Run application service used by the panel."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from threading import Lock, Thread
from typing import TYPE_CHECKING, Mapping
from uuid import uuid4

from pde_agent_verifier_solver.engine.agent_execution import (
    AgentExecutionService,
    InProcessAgentExecutionService,
)
from pde_agent_verifier_solver.engine.campaign import (
    CampaignCompiler,
    CampaignRecipe,
    CompiledCampaign,
)
from pde_agent_verifier_solver.engine.candidate_execution import (
    CandidateExecutionService,
    InProcessCandidateExecutionService,
)
from pde_agent_verifier_solver.engine.runtime import (
    AgentProgram,
    CampaignResult,
    EventStream,
)
from pde_agent_verifier_solver.engine.unified_runtime import CampaignController
from pde_agent_verifier_solver.sdk import (
    DatasetInput,
    DatasetKind,
    GeneratorProgram,
    PreparedReconstructionSource,
    PreparedDataset,
    ProblemProgram,
    PublicProblemBundle,
    ReconstructionEvaluation,
    ReconstructionOutput,
    ReconstructionRecipe,
    ReconstructionReport,
    SamplingStrategy,
    TaskProgram,
    build_reconstruction_report,
)

if TYPE_CHECKING:
    from pde_agent_verifier_solver.plugins.registry import PDEPlugin


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class ProgramBundle:
    problem: ProblemProgram
    task: TaskProgram
    generator: GeneratorProgram
    agents: Mapping[str, AgentProgram]
    public_problem_bundle: PublicProblemBundle | None = None

    def __post_init__(self) -> None:
        if (
            self.public_problem_bundle is not None
            and not isinstance(self.public_problem_bundle, PublicProblemBundle)
        ):
            raise TypeError(
                "public_problem_bundle must be a PublicProblemBundle or None"
            )

    @classmethod
    def from_plugin(
        cls,
        plugin: PDEPlugin,
        agents: Mapping[str, AgentProgram],
        task_id: str | None = None,
    ) -> ProgramBundle:
        """Compose the canonical Program bundle from one PDE plugin."""

        from pde_agent_verifier_solver.plugins.registry import PDEPlugin

        if not isinstance(plugin, PDEPlugin):
            raise TypeError("ProgramBundle.from_plugin needs a PDEPlugin")
        if not isinstance(agents, Mapping) or not agents:
            raise TypeError("ProgramBundle.from_plugin needs a nonempty Agent mapping")
        if any(
            not isinstance(agent_id, str)
            or not agent_id.strip()
            or not callable(getattr(program, "run", None))
            for agent_id, program in agents.items()
        ):
            raise TypeError(
                "ProgramBundle Agent ids must be nonempty strings and values "
                "must implement run(context)"
            )
        return cls(
            problem=plugin.problem,
            task=plugin.task(task_id),
            generator=plugin.generator,
            agents=dict(agents),
            public_problem_bundle=plugin.public_problem_bundle,
        )


@dataclass(frozen=True)
class Project:
    project_id: str
    name: str
    dataset: PreparedDataset
    programs: ProgramBundle
    reconstruction_report: ReconstructionReport | None = None


@dataclass
class RunSession:
    run_id: str
    project_id: str
    campaign: CompiledCampaign
    events: EventStream
    status: RunStatus = RunStatus.QUEUED
    result: CampaignResult | None = None
    error: str | None = None
    thread: Thread | None = None


class ProductApplication:
    def __init__(
        self,
        agent_execution: AgentExecutionService | None = None,
        *,
        candidate_execution: CandidateExecutionService | None = None,
    ) -> None:
        if agent_execution is None:
            raise ValueError(
                "Formal runs require an explicit budget-enforcing Agent execution "
                "service, such as SubprocessAgentExecutionService. Use "
                "ProductApplication.local_development() only for trusted Toy or "
                "local test Agents."
            )
        if isinstance(agent_execution, InProcessAgentExecutionService):
            raise ValueError(
                "Cooperative in-process Agent execution is a local development "
                "mode. Construct it with ProductApplication.local_development()."
            )
        if candidate_execution is None:
            raise ValueError(
                "Formal runs require an explicit non-controller Candidate execution "
                "service, such as SubprocessCandidateExecutionService. Use "
                "ProductApplication.local_development() only for trusted Toy or "
                "local test Candidates."
            )
        if isinstance(candidate_execution, InProcessCandidateExecutionService):
            raise ValueError(
                "In-process Candidate execution is a local development mode. "
                "Construct it with ProductApplication.local_development()."
            )
        self._initialize(
            agent_execution,
            candidate_execution=candidate_execution,
            local_development=False,
        )

    @classmethod
    def local_development(
        cls,
        agent_execution: AgentExecutionService | None = None,
        *,
        candidate_execution: CandidateExecutionService | None = None,
    ) -> ProductApplication:
        """Build the trusted local/Toy application with cooperative execution.

        This named opt-in keeps the uninterruptible in-process adapter out of
        formal runs while retaining a compact path for the Toy panel and tests.
        """

        application = cls.__new__(cls)
        application._initialize(
            (
                agent_execution
                if agent_execution is not None
                else InProcessAgentExecutionService()
            ),
            candidate_execution=candidate_execution,
            local_development=True,
        )
        return application

    def _initialize(
        self,
        agent_execution: AgentExecutionService,
        *,
        candidate_execution: CandidateExecutionService | None,
        local_development: bool,
    ) -> None:
        if candidate_execution is not None and not callable(
            getattr(candidate_execution, "execute_cases", None)
        ):
            raise TypeError(
                "candidate_execution must implement execute_cases(...)"
            )
        if candidate_execution is not None:
            invocation_process = getattr(
                candidate_execution,
                "candidate_invocation_process",
                None,
            )
            if not isinstance(invocation_process, str) or not invocation_process:
                raise TypeError(
                    "candidate_execution must declare candidate_invocation_process"
                )
            if not local_development and invocation_process == "controller":
                raise ValueError(
                    "Controller-process Candidate execution is a local development "
                    "mode. Construct it with ProductApplication.local_development()."
                )
        self._projects: dict[str, Project] = {}
        self._runs: dict[str, RunSession] = {}
        # Keep a strong reference next to each identity-keyed lock. Projects
        # retain their Generators for the application's lifetime, and two
        # Projects may intentionally share the same ProgramBundle.
        self._generator_run_locks: dict[int, tuple[GeneratorProgram, Lock]] = {}
        self._lock = Lock()
        self._agent_execution = agent_execution
        self._candidate_execution = candidate_execution
        self._local_development = local_development

    def create_project(
        self,
        name: str,
        dataset: DatasetInput,
        programs: ProgramBundle,
        *,
        reconstruction: ReconstructionRecipe | None = None,
    ) -> Project:
        if not name.strip():
            raise ValueError("project name cannot be empty")
        report: ReconstructionReport | None = None
        if reconstruction is None:
            prepared = programs.problem.prepare_dataset(dataset)
        else:
            prepared, report = _prepare_reconstructed_project(
                dataset=dataset,
                problem=programs.problem,
                recipe=reconstruction,
            )
        return self.create_prepared_project(
            name, prepared, programs, reconstruction_report=report,
        )

    def create_prepared_project(
        self,
        name: str,
        dataset: PreparedDataset,
        programs: ProgramBundle,
        *,
        reconstruction_report: ReconstructionReport | None = None,
    ) -> Project:
        """Register already prepared data, including a resident online backend.

        This uses the same Run service as file-backed projects; model loading
        stays with the caller's PDE factory and never enters the shared engine.
        """
        if not name.strip():
            raise ValueError("project name cannot be empty")
        if not isinstance(dataset, PreparedDataset):
            raise TypeError("create_prepared_project requires a PreparedDataset")
        if not isinstance(programs, ProgramBundle):
            raise TypeError("create_prepared_project requires a ProgramBundle")
        project = Project(
            project_id=f"project-{uuid4().hex[:8]}",
            name=name.strip(),
            dataset=dataset,
            programs=programs,
            reconstruction_report=reconstruction_report,
        )
        with self._lock:
            self._projects[project.project_id] = project
            self._generator_lock_unlocked(programs.generator)
        return project

    def launch_run(self, project_id: str, recipe: CampaignRecipe) -> RunSession:
        project = self.get_project(project_id)
        compiled = CampaignCompiler().compile(
            recipe,
            project.programs.generator.capabilities,
        )
        _validate_case_inventory(compiled, project.dataset)
        session = RunSession(
            run_id=f"run-{uuid4().hex[:8]}",
            project_id=project_id,
            campaign=compiled,
            events=EventStream(),
        )
        thread = Thread(
            target=self._execute,
            args=(project, session),
            name=session.run_id,
            daemon=True,
        )
        session.thread = thread
        with self._lock:
            self._runs[session.run_id] = session
        thread.start()
        return session

    @property
    def candidate_execution_settings(self):
        snapshot = getattr(self._candidate_execution, "execution_settings_snapshot", None)
        return dict(snapshot()) if callable(snapshot) else {"mode": "local_in_process"}

    def get_project(self, project_id: str) -> Project:
        with self._lock:
            try:
                return self._projects[project_id]
            except KeyError as exc:
                raise KeyError(f"unknown project: {project_id}") from exc

    def get_run(self, run_id: str) -> RunSession:
        with self._lock:
            try:
                return self._runs[run_id]
            except KeyError as exc:
                raise KeyError(f"unknown run: {run_id}") from exc

    def wait(self, run_id: str, timeout: float | None = None) -> RunSession:
        session = self.get_run(run_id)
        if session.thread is not None:
            session.thread.join(timeout)
        return session

    def _execute(self, project: Project, session: RunSession) -> None:
        generator_lock = self._generator_lock(project.programs.generator)
        # A shared mutable Generator is one Campaign resource. A later Run
        # remains visibly QUEUED until the earlier Run has fully released it.
        with generator_lock:
            session.status = RunStatus.RUNNING
            try:
                runner = CampaignController(
                    campaign=session.campaign,
                    dataset=project.dataset,
                    problem=project.programs.problem,
                    task=project.programs.task,
                    generator=project.programs.generator,
                    agents=project.programs.agents,
                    events=session.events,
                    agent_execution=self._agent_execution,
                    public_problem_bundle=project.programs.public_problem_bundle,
                    candidate_execution=self._candidate_execution,
                )
                session.result = runner.run()
            except BaseException as exc:
                # This is a background-thread boundary: even process-control
                # exceptions must settle the public Run instead of leaving it
                # permanently RUNNING with an unhandled thread exception.
                session.error = f"{type(exc).__name__}: {exc}"
                session.status = RunStatus.FAILED
                session.events.emit("run_failed", session.error)
                return
            session.status = RunStatus.COMPLETED

    def _generator_lock(self, generator: GeneratorProgram) -> Lock:
        with self._lock:
            return self._generator_lock_unlocked(generator)

    def _generator_lock_unlocked(self, generator: GeneratorProgram) -> Lock:
        identity = id(generator)
        current = self._generator_run_locks.get(identity)
        if current is not None:
            retained, lock = current
            if retained is not generator:
                raise RuntimeError("Generator identity lock collision")
            return lock
        lock = Lock()
        self._generator_run_locks[identity] = (generator, lock)
        return lock


def _prepare_reconstructed_project(
    *,
    dataset: DatasetInput,
    problem: ProblemProgram,
    recipe: ReconstructionRecipe,
) -> tuple[PreparedDataset, ReconstructionReport]:
    """Run the optional reconstruction once, before a Project is registered."""

    if dataset.kind is not DatasetKind.EXPERIMENTAL:
        raise ValueError(
            "ReconstructionProgram may only be selected for an experimental dataset"
        )
    prepare_source = getattr(problem, "prepare_reconstruction_source", None)
    evaluate = getattr(problem, "evaluate_reconstruction", None)
    prepare_dataset = getattr(problem, "prepare_reconstructed_dataset", None)
    missing = [
        name
        for name, method in (
            ("prepare_reconstruction_source", prepare_source),
            ("evaluate_reconstruction", evaluate),
            ("prepare_reconstructed_dataset", prepare_dataset),
        )
        if not callable(method)
    ]
    if missing:
        raise TypeError(
            f"Problem {problem.info.program_id!r} does not support experimental "
            "reconstruction; missing " + ", ".join(missing)
        )

    source = prepare_source(dataset, recipe.data_settings, recipe.seed)
    if not isinstance(source, PreparedReconstructionSource):
        raise TypeError(
            "prepare_reconstruction_source() must return PreparedReconstructionSource"
        )
    output = recipe.program.reconstruct(
        source.program_input,
        recipe.method_settings,
        recipe.seed,
        None,
    )
    if not isinstance(output, ReconstructionOutput):
        raise TypeError("ReconstructionProgram must return ReconstructionOutput")
    # Materialize Generator data before private evaluation. This adapter call
    # receives only dataset/output, never an evaluation or report object; it is
    # a trusted-plugin contract rather than hostile in-process isolation.
    prepared = prepare_dataset(dataset, output)
    if not isinstance(prepared, PreparedDataset):
        raise TypeError(
            "prepare_reconstructed_dataset() must return PreparedDataset"
        )
    evaluation = evaluate(
        output,
        source.reconstruction_validation_reference,
        recipe.data_settings,
    )
    if not isinstance(evaluation, ReconstructionEvaluation):
        raise TypeError(
            "evaluate_reconstruction() must return ReconstructionEvaluation"
        )
    report = build_reconstruction_report(
        artifact_id=f"reconstruction-{uuid4().hex[:12]}",
        program=recipe.program,
        source=source.program_input,
        output=output,
        evaluation=evaluation,
        recipe=recipe,
    )
    return prepared, report


def _validate_case_inventory(
    campaign: CompiledCampaign,
    dataset: PreparedDataset,
) -> None:
    """Reject finite D0 recipes that would run out partway through a Run."""

    round_hidden_original = sum(
        plan.round_hidden.original
        for plan in campaign.rounds
        if plan.round_hidden is not None
    )
    terminal_original = sum(
        cohort.suite.original
        for cohort in (
            campaign.terminal_evaluation.cohorts
            if campaign.terminal_evaluation is not None
            else ()
        )
    )
    hidden_required = round_hidden_original + terminal_original
    if hidden_required > len(dataset.hidden_cases):
        raise ValueError(
            "Campaign needs "
            f"{hidden_required} fresh original hidden Cases, but the dataset "
            f"contains {len(dataset.hidden_cases)}"
        )

    fixed_original_required = sum(
        plan.original_count
        for plan in campaign.rounds
        if plan.development_sampling.strategy is SamplingStrategy.FIXED_RECIPE
    )
    if fixed_original_required > len(dataset.development_cases):
        raise ValueError(
            "fixed recipes need "
            f"{fixed_original_required} fresh original development Cases, but "
            f"the dataset contains {len(dataset.development_cases)}"
        )
