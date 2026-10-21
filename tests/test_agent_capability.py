import json
import os
from pathlib import Path
from types import SimpleNamespace
import stat
import sys
import tempfile
from threading import Event, Lock, Thread
import time
import unittest

import numpy as np

from pde_agent_verifier_solver.app.service import ProductApplication, RunStatus
from pde_agent_verifier_solver.engine.agent_capability import (
    AgentCapabilityClientError,
    ControllerAgentCapabilityClient,
    ControllerAgentCapabilityFacade,
)
from pde_agent_verifier_solver.engine.agent_execution import (
    CommandAgentProgram,
    SubprocessAgentExecutionService,
)
from pde_agent_verifier_solver.engine.agent_wire import (
    AgentWireCodec,
    AgentWireCodecError,
)
from pde_agent_verifier_solver.engine.campaign import (
    AgentSpec,
    EvaluatorPolicy,
    DevelopmentSamplingPolicy,
    RoundPlan,
)
from pde_agent_verifier_solver.engine.evaluator import (
    AgentEvaluator,
    EvaluationCaseResult,
    EvaluationResponse,
)
from pde_agent_verifier_solver.engine.memory import CampaignMemorySnapshot
from pde_agent_verifier_solver.engine.runtime import AgentContext
from pde_agent_verifier_solver.panel.controller import build_toy_panel
from pde_agent_verifier_solver.plugins.toy import ToyResearchAgent, ToyPoissonProblem, ToySolverTask, ToySineSolver
from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    AnswerView,
    Artifact,
    Case,
    CaseSource,
    CaseVisibility,
    DevelopmentCoordinateMode,
    DatasetInput,
    DatasetKind,
    HiddenSuiteSpec,
    QueryKind,
    SampleBatch,
    SampleSource,
    TaskKind,
)


_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"


def _decode_toy_candidate(payload, context, workspace):
    del workspace
    return ToyResearchAgent._artifact(
        context,
        int(payload["mode"]),
        suffix=str(payload.get("suffix", "command")),
    )


def _command_program(script: Path, **kwargs) -> CommandAgentProgram:
    return CommandAgentProgram(
        (sys.executable, str(script)),
        _decode_toy_candidate,
        environment={"PYTHONPATH": str(_SOURCE_ROOT)},
        **kwargs,
    )


def _run_values(agent_id: str, *, allow_point_truth: bool) -> dict[str, object]:
    return {
        "project_name": f"Canonical capability {allow_point_truth}",
        "round_count": 1,
        "agents": [agent_id],
        "agent_budget_seconds": 8,
        "allowed_sources": ["base"],
        "starter_cases": {"base": 1},
        "max_sampled_cases_per_agent_per_round": 6,
        "generator_settings": {
            "guidance_enabled": False,
            "perturbations_enabled": True,
        },
        "round_hidden": {"base": 1},
        "terminal_hidden": {"base": 1},
        "allow_point_truth": allow_point_truth,
        "development_coordinate_mode": "private_random_points",
        "development_point_count": 8,
    }


class _ManifestEvaluator:
    fixed_query = None
    development_coordinate_mode = DevelopmentCoordinateMode.PRIVATE_RANDOM_POINTS
    development_point_count = 8
    allow_grid_truth = False
    allow_point_truth = False

    def __init__(self, cases=()):
        self.cases = tuple(cases)


class AgentCapabilityTests(unittest.TestCase):
    def test_manifest_discloses_fixed_query_point_count_and_truth_permissions(
        self,
    ) -> None:
        fixed_query = AnswerQuery.grid(fields=("u", "v"), x=17, y=9)

        class FixedEvaluator(_ManifestEvaluator):
            development_coordinate_mode = DevelopmentCoordinateMode.USER_FIXED_QUERY
            development_point_count = 153
            allow_grid_truth = True
            allow_point_truth = False

            @property
            def fixed_query(self):
                return fixed_query

        context = SimpleNamespace(
            evaluator=FixedEvaluator(),
            sampler=None,
            cases=(),
            public_problem_bundle=None,
        )
        with tempfile.TemporaryDirectory() as temporary:
            lease = ControllerAgentCapabilityFacade().open(
                context,
                Path(temporary),
                "execution-fixed-status",
                CommandAgentProgram(
                    (sys.executable, "-c", "pass"),
                    _decode_toy_candidate,
                ),
            )
            try:
                evaluation = lease.manifest["evaluation"]
                self.assertEqual(evaluation["coordinate_mode"], "user_fixed_query")
                self.assertEqual(evaluation["development_point_count"], 153)
                self.assertTrue(evaluation["allow_grid_truth"])
                self.assertFalse(evaluation["allow_point_truth"])
                self.assertEqual(
                    evaluation["fixed_query"],
                    {
                        "kind": "grid",
                        "fields": ["u", "v"],
                        "resolution": {"x": 17, "y": 9},
                        "points": [],
                    },
                )
            finally:
                lease.close()

    def test_default_subprocess_facade_runs_canonical_agent_adaptive_turn(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = root / "capability_worker.py"
            script.write_text(
                "\n".join(
                    (
                        "from pde_agent_verifier_solver.engine.agent_capability import (",
                        "    AgentCapabilityClientError,",
                        "    ControllerAgentCapabilityClient,",
                        ")",
                        "from pde_agent_verifier_solver.engine.agent_execution import (",
                        "    load_agent_worker_request,",
                        "    submit_agent_nomination,",
                        "    write_agent_worker_response,",
                        ")",
                        "from pde_agent_verifier_solver.sdk import AnswerQuery, SampleSource",
                        "request = load_agent_worker_request()",
                        "capability = request['capabilities']",
                        "assert capability['assurance']['loopback_only'] is True",
                        "assert capability['assurance']['per_turn_token'] is True",
                        "assert capability['assurance']['live_agent_context_shared'] is False",
                        "assert capability['assurance']['candidate_binding_process'] == 'controller'",
                        "assert capability['assurance']['candidate_invocation_process'] == 'controller'",
                        "assert capability['assurance']['candidate_scoring_process'] == 'controller'",
                        "assert capability['assurance']['hostile_filesystem_isolation'] is False",
                        "assert capability['assurance']['hostile_candidate_sandbox'] is False",
                        "client = ControllerAgentCapabilityClient.from_request(request)",
                        "assert len(client.current_cases()) == 1",
                        "base = client.sample_base(1)",
                        "paired = client.sample_paired_cases(SampleSource.BASE, 1)",
                        "assert len(paired.roots) == len(paired.children) == 1",
                        "assert paired.children[0].parent_case_id is None",
                        "perturbed = client.perturb_cases((base.cases[0].case_id,))",
                        "assert perturbed.cases[0].parent_case_id is None",
                        "cases = client.current_cases()",
                        "assert len(cases) == 5",
                        "case_ids = tuple(case.case_id for case in cases)",
                        "responses = client.evaluate_many(",
                        "    ({'mode': 1, 'suffix': 'probe-1'},",
                        "     {'mode': 2, 'suffix': 'probe-2'}),",
                        "    case_ids=case_ids,",
                        ")",
                        "assert len(responses) == 2",
                        "assert all(tuple(row.case_id for row in response.cases) == case_ids for response in responses)",
                        "assert all(row.truth is None and row.prediction is None for response in responses for row in response.cases)",
                        "try:",
                        "    client.evaluate(",
                        "        {'mode': 1, 'suffix': 'coordinate-probe'},",
                        "        case_ids=case_ids,",
                        "        query=AnswerQuery.at_points(",
                        "            fields=('u',), points=({'x': 0.5},),",
                        "        ),",
                        "    )",
                        "except AgentCapabilityClientError as exc:",
                        "    assert exc.code == 'access_denied'",
                        "else:",
                        "    raise AssertionError('private coordinates were not enforced')",
                        "truth_allowed = request['worker_settings']['truth_allowed']",
                        "if truth_allowed:",
                        "    disclosed = client.evaluate(",
                        "        {'mode': 2, 'suffix': 'truth-probe'},",
                        "        case_ids=case_ids,",
                        "        return_truth=True,",
                        "    )",
                        "    assert len(disclosed.cases) == len(case_ids)",
                        "    assert all(row.truth is not None for row in disclosed.cases)",
                        "    assert all(row.prediction is not None for row in disclosed.cases)",
                        "    assert all(",
                        "        len(row.truth.fields['u']) == 8 for row in disclosed.cases",
                        "    )",
                        "else:",
                        "    try:",
                        "        client.evaluate(",
                        "            {'mode': 2, 'suffix': 'denied-truth'},",
                        "            case_ids=case_ids,",
                        "            return_truth=True,",
                        "        )",
                        "    except AgentCapabilityClientError as exc:",
                        "        assert exc.code == 'access_denied'",
                        "    else:",
                        "        raise AssertionError('truth disclosure was not denied')",
                        "for sequence, mode in enumerate((2, 3), start=1):",
                        "    submit_agent_nomination(",
                        "        {'mode': mode, 'suffix': f'final-{mode}'},",
                        "        sequence=sequence,",
                        "    )",
                        "write_agent_worker_response(",
                        "    observations=('canonical capabilities exercised',),",
                        ")",
                    )
                ),
                encoding="utf-8",
            )
            service = SubprocessAgentExecutionService(
                root / "turns",
                hard_wall_seconds=15,
            )

            for allow_truth in (False, True):
                agent_id = f"command-agent-{str(allow_truth).lower()}"
                panel = build_toy_panel()
                panel.programs.agents[agent_id] = _command_program(
                    script,
                    worker_settings={"truth_allowed": allow_truth},
                )
                panel.app = ProductApplication.local_development(
                    agent_execution=service
                )

                launched = panel.create_and_launch(
                    _run_values(agent_id, allow_point_truth=allow_truth)
                )
                session = panel.app.wait(launched["run_id"], timeout=25)

                audit = service.audits[-1]
                stderr = Path(audit.stderr_path).read_text(encoding="utf-8")
                self.assertEqual(session.status, RunStatus.COMPLETED, (session.error, stderr))
                self.assertIsNotNone(session.result)
                assert session.result is not None
                self.assertEqual(
                    len(session.result.rounds[0].candidate_evaluations),
                    2,
                )
                self.assertEqual(audit.status, "completed", stderr)
                self.assertEqual(audit.nomination_count, 2)
                self.assertGreater(audit.controller_service_seconds, 0.0)
                self.assertLessEqual(audit.active_seconds, audit.wall_seconds)
                self.assertAlmostEqual(
                    audit.active_seconds,
                    audit.wall_seconds - audit.controller_service_seconds,
                    places=6,
                )
                self.assertTrue(audit.capability_revoked)
                self.assertTrue(audit.capability_drained)
                request = json.loads(Path(audit.request_path).read_text(encoding="utf-8"))
                assurance = request["capabilities"]["assurance"]
                self.assertEqual(assurance["candidate_decoder_process"], "controller")
                self.assertEqual(assurance["candidate_binding_process"], "controller")
                self.assertEqual(assurance["candidate_invocation_process"], "controller")
                self.assertEqual(assurance["candidate_scoring_process"], "controller")
                self.assertEqual(assurance["candidate_evaluation_process"], "controller")
                self.assertFalse(assurance["hostile_filesystem_isolation"])
                self.assertFalse(assurance["hostile_candidate_sandbox"])
                wire = request["capabilities"]["wire"]
                self.assertEqual(wire["codec"], "safe-json-npy-v1")
                self.assertGreater(wire["max_sidecar_bytes"], wire["inline_array_bytes"])
                evaluation = request["capabilities"]["evaluation"]
                self.assertEqual(evaluation["development_point_count"], 8)
                self.assertFalse(evaluation["allow_grid_truth"])
                self.assertEqual(evaluation["allow_point_truth"], allow_truth)

    def test_controller_rpc_time_does_not_consume_legacy_active_budget(self) -> None:
        self._exercise_controller_rpc_budget(legacy=True)

    def test_controller_rpc_cannot_pause_total_time_deadline(self) -> None:
        self._exercise_controller_rpc_budget(legacy=False)

    def _exercise_controller_rpc_budget(self, *, legacy: bool) -> None:
        alive_when_generation_finished = []
        class SlowSampler:
            def sample_cases(self, request):
                time.sleep(1.75)
                pid_file = next((root / "turns").rglob("worker.pid"))
                try:
                    os.kill(int(pid_file.read_text()), 0)
                except ProcessLookupError:
                    alive_when_generation_finished.append(False)
                else:
                    alive_when_generation_finished.append(True)
                return SampleBatch(
                    cases=(
                        Case(
                            case_id="slow-controller-case",
                            source=CaseSource.GENERATED,
                            visibility=CaseVisibility.DEVELOPMENT,
                            public_data={"source": request.source.value},
                            round_index=1,
                            tags=(request.source.value,),
                        ),
                    ),
                    remaining=0,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = root / "slow_capability_worker.py"
            script.write_text(
                "\n".join(
                    (
                        "from pde_agent_verifier_solver.engine.agent_capability import (",
                        "    ControllerAgentCapabilityClient,",
                        ")",
                        "from pde_agent_verifier_solver.engine.agent_execution import (",
                        "    load_agent_worker_request, write_agent_worker_response, submit_agent_nomination,",
                        ")",
                        "import os",
                        "from pathlib import Path",
                        "Path('worker.pid').write_text(str(os.getpid()))",
                        "request = load_agent_worker_request()",
                        "submit_agent_nomination({'mode': 1}, sequence=1)",
                        "client = ControllerAgentCapabilityClient.from_request(request)",
                        "assert client.sample_base(1).cases[0].case_id == 'slow-controller-case'",
                        "write_agent_worker_response(observations=('slow RPC completed',))",
                    )
                ),
                encoding="utf-8",
            )
            agent = AgentSpec(agent_id="slow-agent", name="Slow Agent")
            plan = RoundPlan(
                round_index=1,
                agents=(agent,),
                agent_budget_seconds=2,
                development_sampling=DevelopmentSamplingPolicy(
                    max_cases_per_agent_per_round=1,
                    allowed_sources=(SampleSource.BASE,),
                ),
                round_hidden=HiddenSuiteSpec(base=1),
            )
            evaluator = _ManifestEvaluator()
            context = AgentContext(
                round_plan=plan,
                agent=agent,
                prompt="Use the slow Controller capability.",
                evaluator=evaluator,
                incumbent=None,
                memory=CampaignMemorySnapshot(),
                seed=19,
                sampler=SlowSampler(),
            )
            service = SubprocessAgentExecutionService(
                root / "turns",
                hard_wall_seconds=4,
                enforce_active_budget=legacy,
                poll_interval_seconds=0.01,
            )

            outcome = service.execute(_command_program(script), context, 1.5)

            self.assertEqual(len(outcome.nominations), 1)
            audit = service.audits[-1]
            self.assertGreaterEqual(audit.controller_service_seconds, 1.7)
            self.assertTrue(audit.capability_revoked)
            self.assertTrue(audit.capability_drained)
            self.assertEqual(alive_when_generation_finished, [legacy])
            if legacy:
                self.assertEqual(outcome.observations, ("slow RPC completed",))
                self.assertEqual(audit.status, "completed")
                self.assertGreater(audit.wall_seconds, audit.active_budget_seconds)
                self.assertLess(audit.active_seconds, audit.active_budget_seconds)
            else:
                self.assertEqual(audit.status, "hard_wall_timeout")
                self.assertEqual(audit.time_budget_mode, "hard_wall")
                self.assertIsNone(audit.active_budget_seconds)
                self.assertEqual(audit.hard_wall_seconds, 1.5)
                self.assertFalse(Path(audit.response_path).exists())

    def test_manifest_reports_split_candidate_invocation_without_overclaiming(
        self,
    ) -> None:
        class SplitEvaluator(_ManifestEvaluator):
            candidate_invocation_process = "bounded_same_user_subprocess"

        context = SimpleNamespace(
            evaluator=SplitEvaluator(),
            sampler=None,
            cases=(),
            public_problem_bundle=None,
        )
        with tempfile.TemporaryDirectory() as temporary:
            lease = ControllerAgentCapabilityFacade().open(
                context,
                Path(temporary),
                "execution-split-assurance",
                CommandAgentProgram(
                    (sys.executable, "-c", "pass"),
                    _decode_toy_candidate,
                ),
            )
            try:
                assurance = lease.manifest["assurance"]
                self.assertEqual(assurance["candidate_decoder_process"], "controller")
                self.assertEqual(assurance["candidate_binding_process"], "controller")
                self.assertEqual(
                    assurance["candidate_invocation_process"],
                    "bounded_same_user_subprocess",
                )
                self.assertEqual(assurance["candidate_scoring_process"], "controller")
                self.assertEqual(
                    assurance["candidate_evaluation_process"],
                    "split_subprocess_invocation_controller_scoring",
                )
                self.assertFalse(assurance["hostile_filesystem_isolation"])
            finally:
                lease.close()

    def test_token_scope_revoke_and_drain_wait_for_an_accepted_call(self) -> None:
        entered = Event()
        release = Event()

        class BlockingEvaluator:
            fixed_query = None
            development_coordinate_mode = (
                DevelopmentCoordinateMode.PRIVATE_RANDOM_POINTS
            )
            development_point_count = 8
            allow_grid_truth = False
            allow_point_truth = False

            def __init__(self):
                self.call_count = 0
                self.call_lock = Lock()
                self.cancellation_requested = Event()

            def cancel_pending_evaluations(self):
                self.cancellation_requested.set()

            @property
            def cases(self):
                with self.call_lock:
                    self.call_count += 1
                entered.set()
                if not release.wait(3):
                    raise RuntimeError("test did not release the capability call")
                return ()

        evaluator = BlockingEvaluator()

        class BlockingContext:
            sampler = None

            def __init__(self):
                self.evaluator = evaluator

            @property
            def cases(self):
                return self.evaluator.cases

        context = BlockingContext()

        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            program = CommandAgentProgram(
                (sys.executable, "-c", "pass"),
                _decode_toy_candidate,
            )
            lease = ControllerAgentCapabilityFacade(
                drain_timeout_seconds=3,
            ).open(context, workspace, "execution-token-test", program)
            client = ControllerAgentCapabilityClient(
                lease.manifest,
                workspace=workspace,
                timeout_seconds=3,
            )
            wrong_manifest = dict(lease.manifest)
            wrong_manifest["token"] = "wrong-token"
            wrong_client = ControllerAgentCapabilityClient(
                wrong_manifest,
                workspace=workspace,
                timeout_seconds=3,
            )

            with self.assertRaises(AgentCapabilityClientError) as denied:
                wrong_client.current_cases()
            self.assertEqual(denied.exception.code, "permission_denied")

            result: dict[str, object] = {}
            queued_result: dict[str, object] = {}
            call_thread = Thread(
                target=lambda: result.setdefault("cases", client.current_cases())
            )
            call_thread.start()
            self.assertTrue(entered.wait(1))

            def queued_call() -> None:
                try:
                    client.current_cases()
                except BaseException as exc:
                    queued_result["error"] = exc

            queued_thread = Thread(target=queued_call)
            queued_thread.start()
            host = lease._revoke.__self__
            deadline = time.monotonic() + 1
            while host.state._active_calls != 2 and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(host.state._active_calls, 2)
            close_thread = Thread(target=lease.close)
            close_thread.start()
            time.sleep(0.05)
            self.assertTrue(close_thread.is_alive())
            self.assertTrue(lease.revoked)
            self.assertTrue(evaluator.cancellation_requested.is_set())
            release.set()
            call_thread.join(3)
            queued_thread.join(3)
            close_thread.join(3)

            self.assertFalse(call_thread.is_alive())
            self.assertFalse(queued_thread.is_alive())
            self.assertFalse(close_thread.is_alive())
            self.assertEqual(result["cases"], ())
            self.assertIsInstance(
                queued_result.get("error"),
                AgentCapabilityClientError,
            )
            self.assertEqual(queued_result["error"].code, "access_denied")
            self.assertEqual(evaluator.call_count, 1)
            self.assertTrue(lease.revoked)
            self.assertTrue(lease.drained)
            # Read-only status does not enter the scientific service clock.
            self.assertEqual(lease.controller_service_seconds, 0.0)
            with self.assertRaises(AgentCapabilityClientError) as expired:
                client.current_cases()
            self.assertEqual(expired.exception.code, "capability_unavailable")

    def test_context_mutation_is_serialized_across_worker_calls(self) -> None:
        class SerialSampler:
            def __init__(self):
                self.lock = Lock()
                self.active = 0
                self.maximum_active = 0
                self.calls = 0

            def sample_cases(self, request):
                with self.lock:
                    self.active += 1
                    self.maximum_active = max(self.maximum_active, self.active)
                    self.calls += 1
                    sequence = self.calls
                time.sleep(0.1)
                with self.lock:
                    self.active -= 1
                return SampleBatch(
                    cases=(
                        Case(
                            case_id=f"case-{sequence}",
                            source=CaseSource.GENERATED,
                            visibility=CaseVisibility.DEVELOPMENT,
                            public_data={"sequence": sequence},
                            round_index=1,
                            tags=(request.source.value,),
                        ),
                    ),
                    remaining=2 - sequence,
                )

        sampler = SerialSampler()
        evaluator = _ManifestEvaluator()
        context = SimpleNamespace(evaluator=evaluator, sampler=sampler, cases=())
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            lease = ControllerAgentCapabilityFacade().open(
                context,
                workspace,
                "execution-serialization-test",
                CommandAgentProgram(
                    (sys.executable, "-c", "pass"),
                    _decode_toy_candidate,
                ),
            )
            client = ControllerAgentCapabilityClient(
                lease.manifest,
                workspace=workspace,
            )
            batches = []
            errors = []

            def sample(source: SampleSource) -> None:
                try:
                    if source is SampleSource.BASE:
                        batches.append(client.sample_base(1))
                    else:
                        batches.append(client.sample_guided(1))
                except Exception as exc:
                    errors.append(exc)

            threads = (
                Thread(target=sample, args=(SampleSource.BASE,)),
                Thread(target=sample, args=(SampleSource.GUIDED,)),
            )
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(3)
            lease.close()

            self.assertEqual(errors, [])
            self.assertEqual(len(batches), 2)
            self.assertEqual(sampler.maximum_active, 1)
            self.assertEqual(
                {batch.cases[0].case_id for batch in batches},
                {"case-1", "case-2"},
            )
            self.assertEqual(
                {batch.cases[0].tags[0] for batch in batches},
                {"base", "guided"},
            )

    def test_controller_operation_error_does_not_cross_private_rpc_boundary(self) -> None:
        private_detail = "raw-case-controller-secret /private/truth/location"

        class FailingEvaluator(_ManifestEvaluator):
            def evaluate_many(self, artifacts, request):
                del artifacts, request
                raise ValueError(private_detail)

        def decode(payload, context, workspace):
            del context, workspace
            return Artifact(
                artifact_id=str(payload["artifact_id"]),
                kind=TaskKind.SOLVER,
                name="error-boundary solver",
                program=object(),
                summary="RPC error boundary test",
                usage="RPC error boundary test",
                round_index=1,
                agent_id="wire-agent",
            )

        evaluator = FailingEvaluator()
        context = SimpleNamespace(evaluator=evaluator, sampler=None, cases=())
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            lease = ControllerAgentCapabilityFacade().open(
                context,
                workspace,
                "execution-private-error-test",
                CommandAgentProgram((sys.executable, "-c", "pass"), decode),
            )
            client = ControllerAgentCapabilityClient(
                lease.manifest,
                workspace=workspace,
            )

            with self.assertRaises(AgentCapabilityClientError) as raised:
                client.evaluate(
                    {"artifact_id": "error-probe"},
                    case_ids=("opaque-case",),
                )
            lease.close()

            self.assertEqual(raised.exception.code, "operation_failed")
            self.assertEqual(
                str(raised.exception),
                "Controller capability operation failed",
            )
            self.assertNotIn(private_detail, str(raised.exception))

    def test_real_evaluator_case_scores_cross_rpc_with_truth_disabled(self) -> None:
        dataset = ToyPoissonProblem().prepare_dataset(DatasetInput("feedback-test", DatasetKind.NUMERICAL))
        evaluator = AgentEvaluator(
            dataset.development_cases[:2], ToySolverTask(),
            EvaluatorPolicy(AnswerQuery.grid(fields=("u",), x=17),
                            allow_grid_truth=False, allow_point_truth=False),
        )
        def decode(payload, context, workspace):
            return Artifact(
                artifact_id=payload["artifact_id"], kind=TaskKind.SOLVER,
                name="feedback probe", program=ToySineSolver(2),
                summary="probe", usage="probe", round_index=1, agent_id="probe",
            )
        context = SimpleNamespace(evaluator=evaluator, sampler=None, cases=evaluator.cases)
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            lease = ControllerAgentCapabilityFacade().open(
                context, workspace, "feedback-test",
                CommandAgentProgram((sys.executable, "-c", "pass"), decode),
            )
            try:
                client = ControllerAgentCapabilityClient(lease.manifest, workspace=workspace)
                case_ids = tuple(case.case_id for case in evaluator.cases)
                responses = client.evaluate_many(
                    ({"artifact_id": "first"}, {"artifact_id": "second"}),
                    case_ids=case_ids,
                )
                self.assertEqual(len(responses), 2)
                for response in responses:
                    self.assertEqual(tuple(row.case_id for row in response.cases), case_ids)
                    self.assertTrue(all(row.truth is None and row.prediction is None for row in response.cases))
                    self.assertTrue(all(row.metrics for row in response.cases))
                self.assertEqual(responses[0].cases, responses[1].cases)
                self.assertEqual(list((workspace / "capability-arrays").glob("*.npy")), [])
            finally:
                lease.close()

    def test_large_arrays_use_bounded_read_only_npy_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            sidecar_root = Path(temporary) / "arrays"
            codec = AgentWireCodec(
                sidecar_root,
                inline_array_bytes=1024,
                max_sidecar_bytes=4 * 1024 * 1024,
            )
            values = np.arange(262_144, dtype=np.float64)

            encoded = codec.encode({"field": values})

            descriptor = encoded["field"]
            self.assertEqual(descriptor["$pde_agent_wire_v1"], "ndarray_npy")
            self.assertLess(len(json.dumps(encoded)), 2048)
            sidecar = sidecar_root / descriptor["file"]
            self.assertTrue(sidecar.is_file())
            self.assertGreater(sidecar.stat().st_size, 2 * 1024 * 1024)
            self.assertEqual(stat.S_IMODE(sidecar.stat().st_mode), 0o400)
            decoded = codec.decode(encoded)
            np.testing.assert_array_equal(decoded["field"], values)

            escaped = dict(descriptor)
            escaped["file"] = "../../outside.npy"
            with self.assertRaises(AgentWireCodecError):
                codec.decode({"field": escaped})

            fake_digest = "f" * 64
            symlink = sidecar_root / f"{fake_digest}.npy"
            symlink.symlink_to(sidecar)
            linked = dict(descriptor)
            linked["file"] = symlink.name
            linked["sha256"] = fake_digest
            with self.assertRaises(AgentWireCodecError):
                codec.decode({"field": linked})

            with self.assertRaises(AgentWireCodecError):
                codec.encode(np.asarray([object()], dtype=object))

    def test_large_evaluation_view_crosses_rpc_by_sidecar(self) -> None:
        values = tuple(float(value) for value in range(262_144))
        view = AnswerView(
            kind=QueryKind.GRID,
            fields={"u": values},
            axes={"x": values},
            shape=(262_144,),
        )
        case = Case(
            case_id="large-view-case",
            source=CaseSource.GENERATED,
            visibility=CaseVisibility.DEVELOPMENT,
            public_data={"size": len(values)},
            round_index=1,
        )

        class LargeEvaluator(_ManifestEvaluator):
            def __init__(self):
                super().__init__((case,))

            def evaluate_many(self, artifacts, request):
                self.last_request = request
                return tuple(
                    EvaluationResponse(
                        artifact_id=artifact.artifact_id,
                        primary_error=0.25,
                        metrics={"relative_l2": 0.25},
                        cases=(
                            EvaluationCaseResult(
                                case_id=case.case_id,
                                primary_error=0.25,
                                metrics={"relative_l2": 0.25},
                                prediction=view,
                                truth=view,
                            ),
                        ),
                    )
                    for artifact in artifacts
                )

        def decode(payload, context, workspace):
            del context, workspace
            return Artifact(
                artifact_id=str(payload["artifact_id"]),
                kind=TaskKind.SOLVER,
                name="large-view solver",
                program=object(),
                summary="wire test",
                usage="wire test",
                round_index=1,
                agent_id="wire-agent",
            )

        evaluator = LargeEvaluator()
        context = SimpleNamespace(
            evaluator=evaluator,
            sampler=None,
            cases=evaluator.cases,
        )
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            lease = ControllerAgentCapabilityFacade(
                max_rpc_bytes=64 * 1024,
                inline_array_bytes=1024,
                max_sidecar_bytes=4 * 1024 * 1024,
            ).open(
                context,
                workspace,
                "execution-large-view-test",
                CommandAgentProgram((sys.executable, "-c", "pass"), decode),
            )
            client = ControllerAgentCapabilityClient(
                lease.manifest,
                workspace=workspace,
            )

            response = client.evaluate(
                {"artifact_id": "large-probe"},
                case_ids=(case.case_id,),
                return_truth=True,
            )
            lease.close()

            self.assertEqual(response.artifact_id, "large-probe")
            self.assertEqual(len(response.cases[0].truth.fields["u"]), 262_144)
            self.assertEqual(len(response.cases[0].prediction.axes["x"]), 262_144)
            self.assertTrue(evaluator.last_request.return_truth)
            sidecars = tuple((workspace / "capability-arrays").glob("*.npy"))
            self.assertEqual(len(sidecars), 1)
            self.assertLess(
                len(json.dumps(dict(lease.manifest))),
                64 * 1024,
            )


if __name__ == "__main__":
    unittest.main()
