from __future__ import annotations

import unittest
from dataclasses import replace

from pde_agent_verifier_solver.engine.campaign import EvaluatorPolicy
from pde_agent_verifier_solver.engine.candidate_case import (
    CANDIDATE_CASE_ID_PREFIX,
    candidate_case_id,
)
from pde_agent_verifier_solver.engine.evaluator import (
    AgentEvaluator,
    EvaluationRequest,
    HiddenVerifier,
)
from pde_agent_verifier_solver.plugins.toy import (
    ToyPoissonProblem,
    ToyPoissonGenerator,
    ToyResearchAgent,
    ToySineSolver,
    ToySolverEvaluation,
)
from pde_agent_verifier_solver.panel.controller import build_toy_panel
from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    Artifact,
    CaseAccessKind,
    DatasetInput,
    DatasetKind,
    EvaluationPhase,
    GeneratorCheckpoint,
    GeneratorState,
    CaseVisibility,
    TaskKind,
    open_development_solution_session,
    open_public_case_session,
)


class _RecordingSession:
    def __init__(self, label: str) -> None:
        self.label = label
        self.close_calls = 0

    def read(self) -> str:
        return self.label

    def close(self) -> None:
        self.close_calls += 1


class _SessionGenerator:
    def __init__(self) -> None:
        self.sessions: list[tuple[CaseAccessKind, _RecordingSession]] = []

    def case_session_methods(
        self,
        access: CaseAccessKind,
    ) -> tuple[str, ...]:
        del access
        return ("read",)

    def open_case_session(self, state, case, access):
        del state
        session = _RecordingSession(f"{access.value}:{case.case_id}")
        self.sessions.append((access, session))
        return session


class _NoSessionGenerator:
    """Old plugins without optional numerical-session hooks remain valid."""


class _SessionToyGenerator(ToyPoissonGenerator):
    def __init__(self) -> None:
        super().__init__()
        self.sessions: list[tuple[CaseAccessKind, CaseVisibility, _RecordingSession]] = []
        self.release_calls: list[tuple[str, ...]] = []

    def case_session_methods(
        self,
        access: CaseAccessKind,
    ) -> tuple[str, ...]:
        del access
        return ("read",)

    def open_case_session(self, state, case, access):
        del state
        session = _RecordingSession(f"{access.value}:{case.case_id}")
        self.sessions.append((access, case.visibility, session))
        return session

    def release(self, state, case_ids):
        self.release_calls.append(tuple(case_ids))
        return super().release(state, case_ids)


class _StateBoundSessionGenerator(_SessionToyGenerator):
    """Fail if a resident Case is opened or released through another state."""

    def __init__(self) -> None:
        super().__init__()
        self.case_state_ids: dict[str, int] = {}
        self.state_mismatches: list[str] = []

    def sample(self, state, request, progress=None):
        records = super().sample(state, request, progress)
        for record in records:
            self.case_state_ids[record.case.case_id] = id(state)
        return records

    def open_case_session(self, state, case, access):
        if self.case_state_ids.get(case.case_id) != id(state):
            self.state_mismatches.append(case.case_id)
            raise RuntimeError("resident Case was opened through the wrong state")
        return super().open_case_session(state, case, access)

    def release(self, state, case_ids):
        wrong = [
            case_id
            for case_id in case_ids
            if self.case_state_ids.get(case_id) != id(state)
        ]
        if wrong:
            self.state_mismatches.extend(wrong)
            raise RuntimeError("resident Cases were released through the wrong state")
        return super().release(state, case_ids)


class _SessionAwareSolver:
    def __init__(
        self,
        *,
        request_development_solution: bool = True,
        catch_development_denial: bool = False,
        failure: BaseException | None = None,
    ) -> None:
        self.request_development_solution = request_development_solution
        self.catch_development_denial = catch_development_denial
        self.failure = failure
        self.public_method = None
        self.development_method = None
        self.development_denial: str | None = None
        self.public_values: list[str] = []
        self.development_values: list[str] = []
        self.delegate = ToySineSolver(100)

    def solve(self, case, query, budget=None):
        public = open_public_case_session(case)
        # Opening repeatedly within one candidate call must reuse one session.
        if open_public_case_session(case) is not public:
            raise AssertionError("public session was not memoized")
        self.public_method = public.read
        self.public_values.append(self.public_method())

        if self.request_development_solution:
            try:
                development = open_development_solution_session(case)
                if open_development_solution_session(case) is not development:
                    raise AssertionError("development session was not memoized")
                self.development_method = development.read
                self.development_values.append(self.development_method())
            except RuntimeError as exc:
                if not self.catch_development_denial:
                    raise
                self.development_denial = str(exc)

        if self.failure is not None:
            raise self.failure
        return self.delegate.solve(case, query, budget=budget)


class _SessionResearchAgent(ToyResearchAgent):
    def __init__(self) -> None:
        super().__init__()
        self.solvers: list[_SessionAwareSolver] = []

    def _artifact(self, context, mode, *, suffix):
        artifact = ToyResearchAgent._artifact(context, mode, suffix=suffix)
        solver = _SessionAwareSolver(catch_development_denial=True)
        self.solvers.append(solver)
        return replace(artifact, program=solver)


def _generator_state() -> GeneratorState:
    return GeneratorState(
        checkpoint=GeneratorCheckpoint("session-test", object()),
        base=object(),
    )


def _artifact(program: object, artifact_id: str = "session-solver") -> Artifact:
    return Artifact(
        artifact_id=artifact_id,
        kind=TaskKind.SOLVER,
        name="session-aware solver",
        program=program,
        summary="test",
        usage="test",
        round_index=1,
        agent_id="agent",
    )


class CanonicalSessionLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        dataset = ToyPoissonProblem().prepare_dataset(
            DatasetInput("dataset", DatasetKind.NUMERICAL)
        )
        self.development_record = dataset.development_cases[0]
        self.hidden_record = dataset.hidden_cases[0]
        self.query = AnswerQuery.grid(fields=("u",), x=9)
        self.evaluation = ToySolverEvaluation(self.query)

    def _development_evaluator(
        self,
        generator: object,
        *,
        allow_development_solution: bool,
    ) -> AgentEvaluator:
        return AgentEvaluator(
            (self.development_record,),
            self.evaluation,
            EvaluatorPolicy(
                self.query,
                allow_development_solution_session=(
                    allow_development_solution
                ),
            ),
            generator=generator,
            generator_state=_generator_state(),
        )

    def test_development_opens_memoizes_closes_and_revokes_both_sessions(
        self,
    ) -> None:
        generator = _SessionGenerator()
        solver = _SessionAwareSolver()
        evaluator = self._development_evaluator(
            generator,
            allow_development_solution=True,
        )

        evaluator.evaluate(
            _artifact(solver),
            EvaluationRequest(
                (candidate_case_id(self.development_record.case.case_id),)
            ),
        )

        self.assertEqual(
            [access for access, _session in generator.sessions],
            [
                CaseAccessKind.PUBLIC_PROBLEM,
                CaseAccessKind.DEVELOPMENT_SOLUTION,
            ],
        )
        self.assertTrue(
            all(session.close_calls == 1 for _access, session in generator.sessions)
        )
        assert solver.public_method is not None
        assert solver.development_method is not None
        with self.assertRaisesRegex(RuntimeError, "no longer active"):
            solver.public_method()
        with self.assertRaisesRegex(RuntimeError, "no longer active"):
            solver.development_method()

    def test_development_solution_requires_explicit_user_permission(self) -> None:
        generator = _SessionGenerator()
        solver = _SessionAwareSolver(catch_development_denial=True)
        evaluator = self._development_evaluator(
            generator,
            allow_development_solution=False,
        )

        evaluator.evaluate(
            _artifact(solver),
            EvaluationRequest(
                (candidate_case_id(self.development_record.case.case_id),)
            ),
        )

        self.assertIn("does not expose", solver.development_denial or "")
        self.assertEqual(
            [access for access, _session in generator.sessions],
            [CaseAccessKind.PUBLIC_PROBLEM],
        )

    def test_exception_and_timeout_close_and_revoke_open_sessions(self) -> None:
        for failure in (RuntimeError("failed"), TimeoutError("timed out")):
            with self.subTest(failure=type(failure).__name__):
                generator = _SessionGenerator()
                solver = _SessionAwareSolver(failure=failure)
                evaluator = self._development_evaluator(
                    generator,
                    allow_development_solution=True,
                )

                with self.assertRaises(type(failure)):
                    evaluator.evaluate(
                        _artifact(solver),
                        EvaluationRequest(
                            (
                                candidate_case_id(
                                    self.development_record.case.case_id
                                ),
                            )
                        ),
                    )

                self.assertEqual(len(generator.sessions), 2)
                self.assertTrue(
                    all(
                        session.close_calls == 1
                        for _access, session in generator.sessions
                    )
                )
                assert solver.public_method is not None
                assert solver.development_method is not None
                with self.assertRaisesRegex(RuntimeError, "no longer active"):
                    solver.public_method()
                with self.assertRaisesRegex(RuntimeError, "no longer active"):
                    solver.development_method()

    def test_official_and_terminal_expose_only_public_problem_session(self) -> None:
        for phase in (EvaluationPhase.OFFICIAL, EvaluationPhase.TERMINAL):
            with self.subTest(phase=phase.value):
                generator = _SessionGenerator()
                solver = _SessionAwareSolver(catch_development_denial=True)
                verifier = HiddenVerifier(
                    self.evaluation,
                    self.query,
                    phase=phase,
                    generator=generator,
                    generator_state=_generator_state(),
                )

                verifier.evaluate(_artifact(solver), (self.hidden_record,))

                self.assertEqual(
                    [access for access, _session in generator.sessions],
                    [CaseAccessKind.PUBLIC_PROBLEM],
                )
                self.assertIn("does not expose", solver.development_denial or "")
                self.assertEqual(generator.sessions[0][1].close_calls, 1)

    def test_plugin_without_session_hooks_remains_compatible(self) -> None:
        solver = ToySineSolver(100)
        evaluator = self._development_evaluator(
            _NoSessionGenerator(),
            allow_development_solution=False,
        )

        response = evaluator.evaluate(
            _artifact(solver, "ordinary-solver"),
            EvaluationRequest(
                (candidate_case_id(self.development_record.case.case_id),)
            ),
        )

        self.assertEqual(response.artifact_id, "ordinary-solver")

    def test_campaign_controller_wires_sessions_for_development_and_hidden_calls(
        self,
    ) -> None:
        panel = build_toy_panel()
        generator = _SessionToyGenerator()
        agent = _SessionResearchAgent()
        panel.programs = replace(panel.programs, generator=generator)
        panel.programs.agents["careful-agent"] = agent

        launched = panel.create_and_launch({
            "project_name": "Canonical session lifecycle",
            "sampling_strategy": "fixed_recipe",
            "round_count": 1,
            "agents": ["careful-agent"],
            "round_hidden": {"original": 1, "base": 1, "guided": 0},
            "terminal_hidden": {"original": 1, "base": 1},
        })
        panel.app.wait(launched["run_id"], timeout=5)
        page = panel.run_page(launched["run_id"])

        self.assertEqual(page["status"], "completed", page.get("error"))
        self.assertTrue(generator.sessions)
        self.assertEqual(
            {access for access, _visibility, _session in generator.sessions},
            {CaseAccessKind.PUBLIC_PROBLEM},
        )
        self.assertEqual(
            {visibility for _access, visibility, _session in generator.sessions},
            {CaseVisibility.DEVELOPMENT, CaseVisibility.HIDDEN},
        )
        self.assertTrue(
            all(
                session.close_calls == 1
                for _access, _visibility, session in generator.sessions
            )
        )
        released = [case_id for ids in generator.release_calls for case_id in ids]
        self.assertEqual(len(generator.release_calls), 3)
        self.assertEqual(len(released), len(set(released)))
        self.assertTrue(released)

    def test_fixed_recipe_blinding_keeps_resident_owner_identity_for_sessions(
        self,
    ) -> None:
        class RecordingAgent(_SessionResearchAgent):
            def __init__(self) -> None:
                super().__init__()
                self.visible_case_ids: tuple[str, ...] = ()

            def run(self, context):
                self.visible_case_ids = tuple(
                    case.case_id for case in context.evaluator.cases
                )
                return super().run(context)

        panel = build_toy_panel()
        generator = _StateBoundSessionGenerator()
        agent = RecordingAgent()
        panel.programs = replace(panel.programs, generator=generator)
        panel.programs.agents["careful-agent"] = agent

        launched = panel.create_and_launch({
            "project_name": "Fixed recipe resident identity",
            "sampling_strategy": "fixed_recipe",
            "round_count": 1,
            "agents": ["careful-agent"],
            "fixed_recipe": {"base": 1},
            "round_hidden": {"base": 1},
            "terminal_hidden": {"base": 1},
            "hide_case_types": True,
        })
        panel.app.wait(launched["run_id"], timeout=5)
        page = panel.run_page(launched["run_id"])

        self.assertEqual(page["status"], "completed", page.get("error"))
        self.assertEqual(generator.state_mismatches, [])
        self.assertTrue(agent.visible_case_ids)
        self.assertTrue(
            all(
                case_id.startswith(CANDIDATE_CASE_ID_PREFIX)
                for case_id in agent.visible_case_ids
            )
        )
        opened_case_ids = {
            session.label.split(":", 1)[1]
            for _access, _visibility, session in generator.sessions
        }
        self.assertTrue(opened_case_ids)
        self.assertTrue(opened_case_ids.issubset(generator.case_state_ids))
        self.assertTrue(opened_case_ids.isdisjoint(agent.visible_case_ids))

    def test_terminal_cohorts_open_sessions_with_their_own_generator_state(
        self,
    ) -> None:
        panel = build_toy_panel()
        generator = _StateBoundSessionGenerator()
        agent = _SessionResearchAgent()
        panel.programs = replace(panel.programs, generator=generator)
        panel.programs.agents["careful-agent"] = agent

        launched = panel.create_and_launch({
            "project_name": "Terminal cohort state ownership",
            "round_count": 1,
            "agents": ["careful-agent"],
            "starter_cases": {"base": 1},
            "generator_settings": {
                "guidance_enabled": True,
                "perturbations_enabled": False,
            },
            "round_hidden": {"base": 1},
            "terminal_evaluation": {
                "cohorts": [
                    {
                        "id": "base_model",
                        "guidance_source": "base",
                        "counts": {"base": 1},
                    },
                    {
                        "id": "final_model",
                        "guidance_source": "final_guided",
                        "counts": {"guided": 1},
                    },
                ],
                "ranking": {
                    "cohort_id": "base_model",
                    "distribution": "base",
                },
            },
        })
        panel.app.wait(launched["run_id"], timeout=5)
        page = panel.run_page(launched["run_id"])

        self.assertEqual(page["status"], "completed", page.get("error"))
        self.assertEqual(generator.state_mismatches, [])
        self.assertEqual(
            set(page["terminal_round_reports"][0]["distributions"]),
            {"base_model/base", "final_model/guided"},
        )


if __name__ == "__main__":
    unittest.main()
