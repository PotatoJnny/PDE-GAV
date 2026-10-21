from dataclasses import replace
from pathlib import Path
import sys
import tempfile
from threading import Barrier, Event, Lock, Thread
import time
import unittest
from unittest.mock import patch

from pde_agent_verifier_solver.app.service import ProductApplication, RunStatus
from pde_agent_verifier_solver.engine import agent_execution as agent_execution_module
from pde_agent_verifier_solver.engine.agent_execution import (
    AgentBatchReservation,
    AgentExecutionRequest,
    AgentExecutionResult,
    AgentGPUConflict,
    AgentInfrastructureError,
    CommandAgentProgram,
    InProcessAgentExecutionService,
    SubprocessAgentExecutionService,
    UnavailableAgentCapabilityFacade,
    execute_agent_batch,
)
from pde_agent_verifier_solver.engine.campaign import (
    AgentSpec,
    DevelopmentSamplingPolicy,
    RoundPlan,
)
from pde_agent_verifier_solver.engine.memory import CampaignMemorySnapshot
from pde_agent_verifier_solver.engine.runtime import AgentContext, AgentOutcome
from pde_agent_verifier_solver.engine.seeds import SeedLedger
from pde_agent_verifier_solver.engine.unified_runtime import CampaignController
from pde_agent_verifier_solver.panel.controller import build_toy_panel
from pde_agent_verifier_solver.plugins.toy import (
    ToyPoissonGenerator,
    ToyResearchAgent,
)
from pde_agent_verifier_solver.sdk import (
    HiddenSuiteSpec,
    SampleCasesRequest,
    SampleSource,
    TaskKind,
)


_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"


def _campaign_values(agent_ids, *, budget_seconds=2):
    return {
        "project_name": "Parallel canonical Agents",
        "round_count": 1,
        "agents": list(agent_ids),
        "agent_budget_seconds": budget_seconds,
        "allowed_sources": ["base"],
        "starter_cases": {"base": 0},
        "max_sampled_cases_per_agent_per_round": 1,
        "generator_settings": {
            "guidance_enabled": False,
            "perturbations_enabled": False,
        },
        "round_hidden": {"base": 1},
        "terminal_hidden": {"base": 1},
    }


def _agent_context(agent_id):
    agent = AgentSpec(agent_id=agent_id, name=agent_id.title())
    plan = RoundPlan(
        round_index=1,
        agents=(agent,),
        agent_budget_seconds=1,
        development_sampling=DevelopmentSamplingPolicy(
            max_cases_per_agent_per_round=1,
            allowed_sources=(SampleSource.BASE,),
        ),
        round_hidden=HiddenSuiteSpec(base=1),
    )
    return AgentContext(
        round_plan=plan,
        agent=agent,
        prompt="test",
        evaluator=type("Evaluator", (), {"cases": ()})(),
        incumbent=None,
        memory=CampaignMemorySnapshot(),
        seed=1,
    )


def _decode_nomination(payload, context, workspace):
    del workspace
    return ToyResearchAgent._artifact(
        context,
        int(payload["mode"]),
        suffix=str(payload.get("suffix", "command")),
    )


def _command_program(script, **kwargs):
    environment = {"PYTHONPATH": str(_SOURCE_ROOT)}
    environment.update(kwargs.pop("environment", {}))
    return CommandAgentProgram(
        (sys.executable, str(script)),
        _decode_nomination,
        environment=environment,
        **kwargs,
    )


class AgentParallelExecutionTests(unittest.TestCase):
    def test_explicit_batch_reservation_validates_once_and_releases_once(self):
        class StatefulBatchService:
            def __init__(self):
                self.validation_count = 0
                self.execution_count = 0
                self.release_count = 0
                self.reserved = False

            def validate_batch(self, requests):
                self.validation_count += 1
                if self.reserved:
                    raise RuntimeError("resource already reserved")
                self.reserved = True

            def reserve_batch(self, requests):
                requests = tuple(requests)
                self.validate_batch(requests)

                def release():
                    self.release_count += 1
                    self.reserved = False

                return AgentBatchReservation(self, requests, release=release)

            def execute_reserved_many(self, reservation):
                requests = reservation.requests_for(self)
                self.execution_count += 1
                return tuple(
                    AgentExecutionResult(request, outcome=request.program)
                    for request in requests
                )

            def execute_many(self, requests):
                return execute_agent_batch(self, requests)

            def execute(self, program, context, active_budget_seconds):
                del context, active_budget_seconds
                return program

        service = StatefulBatchService()
        request = AgentExecutionRequest("candidate", _agent_context("leased"), 1)

        results = execute_agent_batch(service, (request,))

        self.assertEqual(results[0].outcome, "candidate")
        self.assertEqual(service.validation_count, 1)
        self.assertEqual(service.execution_count, 1)
        self.assertEqual(service.release_count, 1)
        self.assertFalse(service.reserved)

    def test_controller_runs_agents_concurrently_but_assigns_ids_in_plan_order(self):
        barrier = Barrier(3)
        timing_lock = Lock()
        intervals = {}

        class SamplingAgent:
            def __init__(self, mode, delay):
                self.mode = mode
                self.delay = delay

            def run(self, context):
                started = time.monotonic()
                with timing_lock:
                    intervals[context.agent.agent_id] = [started, None]
                barrier.wait(timeout=2)
                context.sampler.sample_cases(
                    SampleCasesRequest(SampleSource.BASE, 1)
                )
                time.sleep(self.delay)
                with timing_lock:
                    intervals[context.agent.agent_id][1] = time.monotonic()
                return AgentOutcome(
                    ToyResearchAgent._artifact(
                        context,
                        self.mode,
                        suffix="parallel",
                    )
                )

        class ConcurrencyCheckingGenerator(ToyPoissonGenerator):
            def __init__(self):
                super().__init__()
                self._activity_lock = Lock()
                self.active_calls = 0
                self.maximum_active_calls = 0

            def sample(self, state, request, progress=None):
                with self._activity_lock:
                    self.active_calls += 1
                    self.maximum_active_calls = max(
                        self.maximum_active_calls,
                        self.active_calls,
                    )
                try:
                    time.sleep(0.03)
                    return super().sample(state, request, progress)
                finally:
                    with self._activity_lock:
                        self.active_calls -= 1

        class CountingExecutionService(InProcessAgentExecutionService):
            def __init__(self):
                self.validation_count = 0

            def validate_batch(self, requests):
                self.validation_count += 1
                super().validate_batch(requests)

        agent_ids = ("first", "second", "third")
        panel = build_toy_panel()
        generator = ConcurrencyCheckingGenerator()
        panel.programs = type(panel.programs)(
            problem=panel.programs.problem,
            task=panel.programs.task,
            generator=generator,
            agents={
                **panel.programs.agents,
                "first": SamplingAgent(1, 0.12),
                "second": SamplingAgent(2, 0.06),
                "third": SamplingAgent(3, 0.0),
            },
        )
        execution = CountingExecutionService()
        panel.app = ProductApplication.local_development(agent_execution=execution)

        launched = panel.create_and_launch(_campaign_values(agent_ids))
        session = panel.app.wait(launched["run_id"], timeout=5)

        self.assertEqual(session.status, RunStatus.COMPLETED, session.error)
        self.assertEqual(set(intervals), set(agent_ids))
        self.assertEqual(execution.validation_count, 1)
        self.assertLessEqual(
            max(interval[0] for interval in intervals.values()),
            min(interval[1] for interval in intervals.values()),
        )
        self.assertEqual(generator.maximum_active_calls, 1)

        evaluations = session.result.rounds[0].candidate_evaluations
        self.assertEqual(
            [row.artifact.artifact_id for row in evaluations],
            ["candidate-r1-0001", "candidate-r1-0002", "candidate-r1-0003"],
        )
        self.assertEqual(
            [row.artifact.metadata["submitted_artifact_id"] for row in evaluations],
            [
                "toy-r1-first-m1-parallel",
                "toy-r1-second-m2-parallel",
                "toy-r1-third-m3-parallel",
            ],
        )
        agent_events = [
            event
            for event in session.events.snapshot()
            if event.event_type in {"agent_started", "agent_finished"}
        ]
        self.assertEqual(
            [event.event_type for event in agent_events],
            ["agent_started"] * 3 + ["agent_finished"] * 3,
        )
        self.assertEqual(
            [event.payload["agent_id"] for event in agent_events[:3]],
            list(agent_ids),
        )
        self.assertEqual(
            [event.payload["agent_id"] for event in agent_events[3:]],
            list(agent_ids),
        )

    def test_one_agent_failure_does_not_cancel_its_peer(self):
        barrier = Barrier(2)
        completed = []

        class FailingAgent:
            def run(self, context):
                del context
                barrier.wait(timeout=2)
                raise RuntimeError("isolated failure")

        class SuccessfulAgent:
            def run(self, context):
                barrier.wait(timeout=2)
                time.sleep(0.05)
                completed.append(context.agent.agent_id)
                return AgentOutcome(
                    ToyResearchAgent._artifact(context, 2, suffix="survivor")
                )

        panel = build_toy_panel()
        panel.programs.agents["failing"] = FailingAgent()
        panel.programs.agents["successful"] = SuccessfulAgent()
        launched = panel.create_and_launch(
            _campaign_values(("failing", "successful"))
        )
        session = panel.app.wait(launched["run_id"], timeout=5)

        self.assertEqual(session.status, RunStatus.COMPLETED, session.error)
        self.assertEqual(completed, ["successful"])
        events = session.events.snapshot()
        failed = [event for event in events if event.event_type == "agent_failed"]
        finished = [event for event in events if event.event_type == "agent_finished"]
        self.assertEqual([event.payload["agent_id"] for event in failed], ["failing"])
        self.assertEqual(
            [event.payload["agent_id"] for event in finished],
            ["successful"],
        )

    def test_invalid_portfolio_does_not_consume_candidate_ids(self):
        class InvalidPortfolioAgent:
            def run(self, context):
                valid = ToyResearchAgent._artifact(
                    context,
                    1,
                    suffix="discarded-valid",
                )
                invalid = replace(
                    ToyResearchAgent._artifact(
                        context,
                        2,
                        suffix="invalid-kind",
                    ),
                    kind=TaskKind.STATISTICS,
                )
                return AgentOutcome(valid, artifacts=(invalid,))

        class SurvivorAgent:
            def run(self, context):
                return AgentOutcome(
                    ToyResearchAgent._artifact(context, 3, suffix="survivor")
                )

        panel = build_toy_panel()
        panel.programs.agents["invalid-portfolio"] = InvalidPortfolioAgent()
        panel.programs.agents["survivor"] = SurvivorAgent()
        launched = panel.create_and_launch(
            _campaign_values(("invalid-portfolio", "survivor"))
        )
        session = panel.app.wait(launched["run_id"], timeout=5)

        self.assertEqual(session.status, RunStatus.COMPLETED, session.error)
        evaluations = session.result.rounds[0].candidate_evaluations
        self.assertEqual(len(evaluations), 1)
        self.assertEqual(evaluations[0].artifact.artifact_id, "candidate-r1-0001")
        self.assertEqual(
            evaluations[0].artifact.metadata["submitted_artifact_id"],
            "toy-r1-survivor-m3-survivor",
        )

    def test_execute_only_service_keeps_sequential_compatibility_and_order(self):
        class LegacyService:
            def __init__(self):
                self.calls = []

            def execute(self, program, context, active_budget_seconds):
                del context, active_budget_seconds
                self.calls.append(program)
                if program == "bad":
                    raise RuntimeError("bad legacy turn")
                return program.upper()

        service = LegacyService()
        requests = tuple(
            AgentExecutionRequest(value, _agent_context(value), 1)
            for value in ("first", "bad", "third")
        )

        results = execute_agent_batch(service, requests)

        self.assertEqual(service.calls, ["first", "bad", "third"])
        self.assertEqual([result.request for result in results], list(requests))
        self.assertEqual(results[0].outcome, "FIRST")
        self.assertIsInstance(results[1].error, RuntimeError)
        self.assertEqual(results[2].outcome, "THIRD")

    def test_batch_re_raises_control_flow_after_all_peers_settle(self):
        class LegacyService:
            def __init__(self):
                self.calls = []

            def execute(self, program, context, active_budget_seconds):
                del context, active_budget_seconds
                self.calls.append(program)
                if program == "interrupt":
                    raise KeyboardInterrupt("stop batch")
                return program

        service = LegacyService()
        requests = tuple(
            AgentExecutionRequest(value, _agent_context(value), 1)
            for value in ("interrupt", "settled-peer")
        )

        with self.assertRaisesRegex(KeyboardInterrupt, "stop batch"):
            execute_agent_batch(service, requests)

        self.assertEqual(service.calls, ["interrupt", "settled-peer"])

    def test_product_run_settles_failed_on_process_control_exception(self):
        for error_type in (KeyboardInterrupt, SystemExit):
            with self.subTest(error_type=error_type.__name__):
                class InterruptingAgent:
                    def run(self, context):
                        del context
                        raise error_type("stop background Run")

                panel = build_toy_panel()
                panel.programs.agents["interrupt"] = InterruptingAgent()
                launched = panel.create_and_launch(
                    _campaign_values(("interrupt",))
                )
                session = panel.app.wait(launched["run_id"], timeout=5)

                self.assertEqual(session.status, RunStatus.FAILED)
                self.assertIn(error_type.__name__, session.error)
                self.assertIsNone(session.result)
                failed = [
                    event
                    for event in session.events.snapshot()
                    if event.event_type == "run_failed"
                ]
                self.assertEqual(len(failed), 1)

    def test_runs_sharing_one_generator_stay_queued_and_serialize(self):
        first_entered = Event()
        release_first = Event()
        second_entered = Event()
        activity_lock = Lock()
        calls = 0
        active = 0
        maximum_active = 0

        class BlockingAgent:
            def run(self, context):
                nonlocal calls, active, maximum_active
                with activity_lock:
                    calls += 1
                    ordinal = calls
                    active += 1
                    maximum_active = max(maximum_active, active)
                try:
                    if ordinal == 1:
                        first_entered.set()
                        if not release_first.wait(3):
                            raise RuntimeError("test did not release first Run")
                    else:
                        second_entered.set()
                    return AgentOutcome(
                        ToyResearchAgent._artifact(
                            context,
                            ordinal,
                            suffix=f"serialized-{ordinal}",
                        )
                    )
                finally:
                    with activity_lock:
                        active -= 1

        panel = build_toy_panel()
        panel.programs.agents["serialized"] = BlockingAgent()
        first = panel.create_and_launch(_campaign_values(("serialized",)))
        self.assertTrue(first_entered.wait(2))
        second = panel.create_and_launch(_campaign_values(("serialized",)))
        second_session = panel.app.get_run(second["run_id"])
        time.sleep(0.1)

        self.assertEqual(second_session.status, RunStatus.QUEUED)
        self.assertFalse(second_entered.is_set())
        release_first.set()
        first_session = panel.app.wait(first["run_id"], timeout=5)
        second_session = panel.app.wait(second["run_id"], timeout=5)

        self.assertEqual(first_session.status, RunStatus.COMPLETED, first_session.error)
        self.assertEqual(second_session.status, RunStatus.COMPLETED, second_session.error)
        self.assertTrue(second_entered.is_set())
        self.assertEqual(maximum_active, 1)

    def test_runs_with_distinct_generators_can_overlap(self):
        barrier = Barrier(2)
        completed = []

        class OverlapAgent:
            def run(self, context):
                barrier.wait(timeout=3)
                completed.append(context.agent.agent_id)
                return AgentOutcome(
                    ToyResearchAgent._artifact(context, 1, suffix="cross-run")
                )

        app = ProductApplication.local_development()
        panels = (build_toy_panel(), build_toy_panel())
        for panel in panels:
            panel.app = app
            panel.programs.agents["overlap"] = OverlapAgent()

        launched = tuple(
            panel.create_and_launch(_campaign_values(("overlap",)))
            for panel in panels
        )
        sessions = tuple(
            app.wait(item["run_id"], timeout=6)
            for item in launched
        )

        self.assertTrue(all(session.status is RunStatus.COMPLETED for session in sessions))
        self.assertEqual(completed, ["overlap", "overlap"])

    def test_gpu_conflict_fails_before_controller_emits_agent_started(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root / "worker-started"
            script = root / "worker.py"
            script.write_text(
                "import os\n"
                "from pathlib import Path\n"
                "Path(os.environ['MARKER']).write_text('started')\n",
                encoding="utf-8",
            )
            service = SubprocessAgentExecutionService(
                root / "turns",
                hard_wall_seconds=3,
                capability_facade=UnavailableAgentCapabilityFacade(),
            )
            panel = build_toy_panel()
            for agent_id in ("gpu-a", "gpu-b"):
                panel.programs.agents[agent_id] = CommandAgentProgram(
                    (sys.executable, str(script)),
                    _decode_nomination,
                    environment={
                        "PYTHONPATH": str(_SOURCE_ROOT),
                        "MARKER": str(marker),
                    },
                    gpu_devices=("0",),
                )
            panel.app = ProductApplication.local_development(
                agent_execution=service
            )

            launched = panel.create_and_launch(
                _campaign_values(("gpu-a", "gpu-b"), budget_seconds=2)
            )
            session = panel.app.wait(launched["run_id"], timeout=5)

            self.assertEqual(session.status, RunStatus.FAILED)
            self.assertIn(AgentGPUConflict.__name__, session.error)
            self.assertFalse(marker.exists())
            self.assertEqual(list((root / "turns").iterdir()), [])
            self.assertFalse(any(
                event.event_type == "agent_started"
                for event in session.events.snapshot()
            ))

    def test_subprocess_batch_has_private_gpu_environments(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = root / "worker.py"
            script.write_text(
                "import os\n"
                "from pde_agent_verifier_solver.engine.agent_execution import (\n"
                "    load_agent_worker_request, submit_agent_nomination,\n"
                "    write_agent_worker_response,\n"
                ")\n"
                "request = load_agent_worker_request()\n"
                "visible = os.environ['CUDA_VISIBLE_DEVICES']\n"
                "assert visible == request['worker_settings']['expected_gpu']\n"
                "submit_agent_nomination(\n"
                "    {'mode': request['worker_settings']['mode'], 'suffix': visible},\n"
                "    sequence=1,\n"
                ")\n"
                "write_agent_worker_response(observations=(visible,))\n",
                encoding="utf-8",
            )
            service = SubprocessAgentExecutionService(
                root / "turns",
                hard_wall_seconds=3,
                capability_facade=UnavailableAgentCapabilityFacade(),
                controller_gpu_devices=("7",),
                poll_interval_seconds=0.01,
            )
            requests = (
                AgentExecutionRequest(
                    _command_program(
                        script,
                        gpu_devices=("2",),
                        worker_settings={"expected_gpu": "2", "mode": 1},
                    ),
                    _agent_context("gpu-two"),
                    2,
                ),
                AgentExecutionRequest(
                    _command_program(
                        script,
                        gpu_devices=("5", "6"),
                        worker_settings={"expected_gpu": "5,6", "mode": 2},
                    ),
                    _agent_context("gpu-five-six"),
                    2,
                ),
            )

            results = service.execute_many(requests)

            self.assertTrue(all(result.succeeded for result in results))
            self.assertEqual(
                [result.outcome.observations for result in results],
                [("2",), ("5,6",)],
            )
            self.assertEqual(
                [result.request.context.agent.agent_id for result in results],
                ["gpu-two", "gpu-five-six"],
            )

    def test_gpu_reservation_is_service_wide_across_concurrent_batches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root / "first-worker-started"
            script = root / "worker.py"
            script.write_text(
                "import os\n"
                "import time\n"
                "from pathlib import Path\n"
                "from pde_agent_verifier_solver.engine.agent_execution import "
                "write_agent_worker_response\n"
                "Path(os.environ['MARKER']).write_text('started')\n"
                "time.sleep(0.4)\n"
                "write_agent_worker_response()\n",
                encoding="utf-8",
            )
            service = SubprocessAgentExecutionService(
                root / "turns",
                hard_wall_seconds=3,
                capability_facade=UnavailableAgentCapabilityFacade(),
                poll_interval_seconds=0.01,
            )

            def request(agent_id):
                return AgentExecutionRequest(
                    _command_program(
                        script,
                        gpu_devices=("0",),
                        environment={
                            "PYTHONPATH": str(_SOURCE_ROOT),
                            "MARKER": str(marker),
                        },
                    ),
                    _agent_context(agent_id),
                    2,
                )

            first_result = {}

            def run_first():
                try:
                    first_result["value"] = service.execute_many((request("first"),))
                except BaseException as exc:
                    first_result["error"] = exc

            thread = Thread(target=run_first)
            thread.start()
            deadline = time.monotonic() + 2
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(marker.exists())
            workspace_count = len(tuple((root / "turns").iterdir()))

            with self.assertRaisesRegex(AgentGPUConflict, "already reserved"):
                service.execute_many((request("conflict"),))

            self.assertEqual(len(tuple((root / "turns").iterdir())), workspace_count)
            thread.join(5)
            self.assertFalse(thread.is_alive())
            self.assertNotIn("error", first_result)
            self.assertTrue(first_result["value"][0].succeeded)
            retry = service.execute_many((request("after-release"),))
            self.assertTrue(retry[0].succeeded)

    def test_unconfirmed_cleanup_is_fatal_to_campaign_and_quarantines_gpu(self):
        class UnconfirmedCleanupService(SubprocessAgentExecutionService):
            @staticmethod
            def _broken(process):
                return any(str(value).endswith("broken.py") for value in process.args)

            def _terminate_process_group(self, process):
                confirmed = super()._terminate_process_group(process)
                return False if self._broken(process) else confirmed

            def _terminate_remaining_process_group(self, process):
                confirmed = super()._terminate_remaining_process_group(process)
                return False if self._broken(process) else confirmed

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            broken = root / "broken.py"
            healthy = root / "healthy.py"
            broken.write_text(
                "import subprocess\n"
                "import sys\n"
                "from pde_agent_verifier_solver.engine.agent_execution import (\n"
                "    submit_agent_nomination, write_agent_worker_response,\n"
                ")\n"
                "submit_agent_nomination({'mode': 1}, sequence=1)\n"
                "write_agent_worker_response()\n"
                "subprocess.Popen([sys.executable, '-c', "
                "'import time; time.sleep(60)'])\n",
                encoding="utf-8",
            )
            healthy.write_text(
                "from pde_agent_verifier_solver.engine.agent_execution import (\n"
                "    submit_agent_nomination, write_agent_worker_response,\n"
                ")\n"
                "submit_agent_nomination({'mode': 2}, sequence=1)\n"
                "write_agent_worker_response()\n",
                encoding="utf-8",
            )
            service = UnconfirmedCleanupService(
                root / "turns",
                hard_wall_seconds=3,
                capability_facade=UnavailableAgentCapabilityFacade(),
                termination_grace_seconds=0.1,
                poll_interval_seconds=0.01,
            )
            panel = build_toy_panel()
            panel.programs.agents["broken"] = _command_program(
                broken,
                gpu_devices=("4",),
            )
            panel.programs.agents["healthy"] = _command_program(
                healthy,
                gpu_devices=("5",),
            )
            panel.app = ProductApplication.local_development(
                agent_execution=service
            )

            launched = panel.create_and_launch(
                _campaign_values(("broken", "healthy"))
            )
            session = panel.app.wait(launched["run_id"], timeout=6)

            self.assertEqual(session.status, RunStatus.FAILED)
            self.assertIn(AgentInfrastructureError.__name__, session.error)
            self.assertIsNone(session.result)
            self.assertIn("4", service.quarantined_gpu_devices)
            healthy_audit = next(
                audit for audit in service.audits if audit.agent_id == "healthy"
            )
            self.assertEqual(healthy_audit.status, "completed")
            page = panel.run_page(launched["run_id"])
            self.assertEqual(len(page["candidate_history"]), 1)
            self.assertEqual(
                page["candidate_history"][0]["artifact_id"],
                "candidate-r1-0001",
            )
            self.assertEqual(page["candidate_history"][0]["status"], "not_evaluated")
            before_retry = len(tuple((root / "turns").iterdir()))
            quarantined_request = AgentExecutionRequest(
                _command_program(healthy, gpu_devices=("4",)),
                _agent_context("quarantined"),
                2,
            )
            with self.assertRaisesRegex(AgentGPUConflict, "quarantined"):
                service.execute_many((quarantined_request,))
            self.assertEqual(len(tuple((root / "turns").iterdir())), before_retry)

    def test_audit_failure_cannot_mask_cleanup_failure_or_gpu_quarantine(self):
        class UnconfirmedCleanupService(SubprocessAgentExecutionService):
            def _terminate_process_group(self, process):
                super()._terminate_process_group(process)
                return False

            def _terminate_remaining_process_group(self, process):
                super()._terminate_remaining_process_group(process)
                return False

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worker = root / "worker.py"
            worker.write_text(
                "import time\n"
                "time.sleep(60)\n",
                encoding="utf-8",
            )
            service = UnconfirmedCleanupService(
                root / "turns",
                hard_wall_seconds=1,
                capability_facade=UnavailableAgentCapabilityFacade(),
                termination_grace_seconds=0.05,
                poll_interval_seconds=0.01,
            )
            request = AgentExecutionRequest(
                _command_program(worker, gpu_devices=("7",)),
                _agent_context("audit-failure"),
                0.1,
            )
            original_write_json_new = agent_execution_module._write_json_new

            def fail_audit(path, payload):
                if Path(path).name == "audit.json":
                    raise OSError("synthetic audit write failure")
                return original_write_json_new(path, payload)

            with patch.object(
                agent_execution_module,
                "_write_json_new",
                side_effect=fail_audit,
            ):
                with self.assertRaises(AgentInfrastructureError) as caught:
                    service.execute_many((request,))

            self.assertIn("7", service.quarantined_gpu_devices)
            self.assertEqual(service.audits, ())
            self.assertTrue(
                any(
                    "audit persistence also failed" in note
                    and "synthetic audit write failure" in note
                    for note in getattr(caught.exception, "__notes__", ())
                )
            )

    def test_development_plan_seed_does_not_depend_on_plugin_case_id(self):
        controller = object.__new__(CampaignController)
        controller._seeds = SeedLedger(99)
        provider = controller._development_plan_seed_provider(2, "agent-a")

        first = provider(3, 4, "scheduler-dependent-id-a")
        second = provider(3, 4, "scheduler-dependent-id-b")

        self.assertEqual(first, second)

    def test_partial_timeout_nomination_does_not_cancel_completed_peer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            slow = root / "slow.py"
            fast = root / "fast.py"
            slow.write_text(
                "import time\n"
                "from pde_agent_verifier_solver.engine.agent_execution import "
                "submit_agent_nomination\n"
                "submit_agent_nomination({'mode': 1}, sequence=1)\n"
                "time.sleep(60)\n",
                encoding="utf-8",
            )
            fast.write_text(
                "import time\n"
                "from pde_agent_verifier_solver.engine.agent_execution import (\n"
                "    submit_agent_nomination, write_agent_worker_response,\n"
                ")\n"
                "time.sleep(0.05)\n"
                "submit_agent_nomination({'mode': 2}, sequence=1)\n"
                "write_agent_worker_response(observations=('completed peer',))\n",
                encoding="utf-8",
            )
            service = SubprocessAgentExecutionService(
                root / "turns",
                hard_wall_seconds=5,
                capability_facade=UnavailableAgentCapabilityFacade(),
                termination_grace_seconds=0.1,
                poll_interval_seconds=0.01,
            )
            # Only the deliberately sleeping worker should time out. Give the
            # fast peer enough budget for interpreter imports on a busy shared
            # machine so this test measures peer isolation, not host load.
            requests = (
                AgentExecutionRequest(
                    _command_program(slow),
                    _agent_context("slow"),
                    1.2,
                ),
                AgentExecutionRequest(
                    _command_program(fast),
                    _agent_context("fast"),
                    4.0,
                ),
            )

            results = service.execute_many(requests)

            self.assertTrue(all(result.succeeded for result in results))
            self.assertIn("preserved 1 committed nomination", results[0].outcome.observations[-1])
            self.assertEqual(results[1].outcome.observations, ("completed peer",))
            statuses = {
                audit.agent_id: audit.status
                for audit in service.audits
            }
            self.assertEqual(statuses, {"slow": "hard_wall_timeout", "fast": "completed"})

    def test_controller_gpu_assignment_is_reserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = SubprocessAgentExecutionService(
                root / "turns",
                hard_wall_seconds=2,
                capability_facade=UnavailableAgentCapabilityFacade(),
                controller_gpu_devices=("3",),
            )
            request = AgentExecutionRequest(
                CommandAgentProgram(
                    (sys.executable, "-c", "pass"),
                    _decode_nomination,
                    gpu_devices=("3",),
                ),
                _agent_context("conflict"),
                1,
            )

            with self.assertRaisesRegex(AgentGPUConflict, "Controller"):
                service.execute_many((request,))

            self.assertEqual(list((root / "turns").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
