from dataclasses import replace
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from pde_agent_verifier_solver.app.service import ProductApplication, RunStatus
from pde_agent_verifier_solver.engine.campaign import (
    AgentSpec,
    DevelopmentSamplingPolicy,
    RoundPlan,
)
from pde_agent_verifier_solver.engine.candidate_execution import (
    InProcessCandidateExecutionService,
)
from pde_agent_verifier_solver.engine.agent_execution import (
    AGENT_CAPABILITY_SCHEMA,
    AGENT_EXECUTION_ID_ENV,
    AGENT_PROTOCOL_MAX_BYTES_ENV,
    AGENT_ROUND_LOG_ENV,
    AGENT_ROUND_LOG_MAX_BYTES_ENV,
    AgentCapabilityLease,
    AgentExecutionError,
    AgentInfrastructureError,
    CommandAgentProgram,
    InProcessAgentExecutionService,
    SubprocessAgentExecutionService,
    UnavailableAgentCapabilityFacade,
    load_agent_worker_request,
    submit_agent_nomination,
    write_agent_round_log,
)
from pde_agent_verifier_solver.engine.memory import (
    CampaignMemorySnapshot,
    CandidateHistory,
)
from pde_agent_verifier_solver.engine.runtime import AgentContext, AgentOutcome
from pde_agent_verifier_solver.engine.unified_runtime import CampaignController
from pde_agent_verifier_solver.panel.controller import build_toy_panel
from pde_agent_verifier_solver.plugins.toy import ToyResearchAgent
from pde_agent_verifier_solver.sdk import (
    Case,
    CaseSource,
    CaseVisibility,
    HiddenSuiteSpec,
    PromotionMetricSummary,
    SampleSource,
)


_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"


def _run_values(agent_id: str, *, budget_seconds: int) -> dict[str, object]:
    return {
        "project_name": f"Execution service {agent_id}",
        "round_count": 1,
        "agents": [agent_id],
        "agent_budget_seconds": budget_seconds,
        "allowed_sources": ["base"],
        "starter_cases": {"base": 1},
        "max_sampled_cases_per_agent_per_round": 1,
        "generator_settings": {
            "guidance_enabled": False,
            "perturbations_enabled": False,
        },
        "round_hidden": {"base": 1},
        "terminal_hidden": {"base": 1},
    }


class _CasesOnlyEvaluator:
    def __init__(self, cases):
        self.cases = tuple(cases)


def _agent_context(agent_id: str = "command-agent") -> AgentContext:
    agent = AgentSpec(agent_id=agent_id, name="Command Agent")
    plan = RoundPlan(
        round_index=1,
        agents=(agent,),
        agent_budget_seconds=2,
        development_sampling=DevelopmentSamplingPolicy(
            max_cases_per_agent_per_round=3,
            allowed_sources=(SampleSource.BASE,),
        ),
        round_hidden=HiddenSuiteSpec(base=1),
    )
    case = Case(
        case_id="development-case-1",
        source=CaseSource.GENERATED,
        visibility=CaseVisibility.DEVELOPMENT,
        public_data={"frequency": 3},
        round_index=1,
        tags=("base",),
    )
    return AgentContext(
        round_plan=plan,
        agent=agent,
        prompt="Improve the public solver.",
        evaluator=_CasesOnlyEvaluator((case,)),
        incumbent=None,
        memory=CampaignMemorySnapshot(),
        seed=17,
    )


def _decode_toy_nomination(payload, context, workspace):
    if payload.get("workspace") is not None:
        assert Path(str(payload["workspace"])) == workspace
    return ToyResearchAgent._artifact(
        context,
        int(payload["mode"]),
        suffix=str(payload.get("suffix", "worker")),
    )


def _worker_program(script: Path, **kwargs) -> CommandAgentProgram:
    return CommandAgentProgram(
        (sys.executable, str(script)),
        _decode_toy_nomination,
        environment={"PYTHONPATH": str(_SOURCE_ROOT)},
        **kwargs,
    )


class AgentExecutionServiceTests(unittest.TestCase):
    def test_command_program_cannot_override_managed_python_cache_policy(self) -> None:
        for name in ("PYTHONDONTWRITEBYTECODE", "PYTHONPYCACHEPREFIX"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "managed values"):
                    CommandAgentProgram(
                        (sys.executable, "-c", "pass"),
                        _decode_toy_nomination,
                        environment={name: "caller-controlled"},
                    )

    def test_revoke_failure_still_drains_capability_and_cleans_worker(self) -> None:
        class BrokenRevokeFacade:
            def __init__(self):
                self.revoke_count = 0
                self.drain_count = 0

            def open(self, context, workspace, execution_id, program):
                del context, workspace, execution_id, program

                def revoke() -> None:
                    self.revoke_count += 1
                    raise RuntimeError("revoke callback failed")

                def drain() -> None:
                    self.drain_count += 1

                return AgentCapabilityLease(
                    {"schema": AGENT_CAPABILITY_SCHEMA},
                    revoke=revoke,
                    drain=drain,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            facade = BrokenRevokeFacade()
            service = SubprocessAgentExecutionService(
                root / "turns",
                hard_wall_seconds=3,
                capability_facade=facade,
            )
            program = CommandAgentProgram(
                (sys.executable, "-c", "pass"),
                _decode_toy_nomination,
            )

            with self.assertRaisesRegex(
                AgentInfrastructureError,
                "capability",
            ):
                service.execute(program, _agent_context(), 2)

            audit = service.audits[-1]
            self.assertEqual(audit.status, "infrastructure_error")
            self.assertFalse(audit.capability_revoked)
            self.assertTrue(audit.capability_drained)
            self.assertTrue(audit.process_group_cleaned)
            self.assertGreaterEqual(facade.revoke_count, 2)
            self.assertEqual(facade.drain_count, 1)

    def test_subprocess_protocol_preserves_multiple_nominations_and_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = root / "worker.py"
            script.write_text(
                "\n".join(
                    (
                        "import os",
                        "import sys",
                        "from pathlib import Path",
                        "from pde_agent_verifier_solver.engine.agent_execution import (",
                        "    load_agent_worker_request,",
                        "    submit_agent_nomination,",
                        "    write_agent_round_log,",
                        "    write_agent_worker_response,",
                        ")",
                        "request = load_agent_worker_request()",
                        "campaign_path = Path(os.environ['PDE_AGENT_CAMPAIGN_MEMORY'])",
                        "history_path = Path(os.environ['PDE_AGENT_CANDIDATE_HISTORY'])",
                        "round_log_path = Path(os.environ['PDE_AGENT_ROUND_LOG'])",
                        "assert request['campaign_memory']['completed_rounds'] == 0",
                        "assert request['campaign_memory']['path'] == 'CAMPAIGN_MEMORY.md'",
                        "assert campaign_path.read_text() == '# Campaign Memory Log'",
                        "assert request['candidate_history']['candidate_count'] == 0",
                        "assert request['candidate_history']['path'] == 'CANDIDATE_HISTORY.json'",
                        "history = __import__('json').loads(history_path.read_text())",
                        "assert history == {'schema': 'canonical-candidate-history-v1', 'candidates': []}",
                        "assert request['round_log']['path'] == 'ROUND_LOG.md'",
                        "assert request['round_log']['max_bytes'] == 8 * 1024 * 1024",
                        "assert os.environ['PDE_AGENT_PROTOCOL_MAX_BYTES'] == str(8 * 1024 * 1024)",
                        "assert round_log_path.is_file()",
                        "assert request['agent']['seed'] == 17",
                        "assert request['issued_cases'][0]['public_data']['frequency'] == 3",
                        "assert request['capabilities']['sampling']['available'] is False",
                        "assert request['capabilities']['evaluation']['available'] is False",
                        "assert os.environ['CUDA_VISIBLE_DEVICES'] == '2'",
                        "assert os.environ['PYTHONDONTWRITEBYTECODE'] == '1'",
                        "assert os.environ['PYTHONPYCACHEPREFIX'] == str(Path(os.environ['PDE_AGENT_WORKSPACE']) / 'pycache')",
                        "print('worker stdout', flush=True)",
                        "print('worker stderr', file=sys.stderr, flush=True)",
                        "write_agent_round_log('# Worker round log\\n\\nThree candidates tested.')",
                        "for sequence, mode in enumerate((1, 2, 3), start=1):",
                        "    submit_agent_nomination(",
                        "        {'mode': mode, 'suffix': f'worker-{mode}',",
                        "         'workspace': os.getcwd()},",
                        "        sequence=sequence,",
                        "    )",
                        "write_agent_worker_response(",
                        "    observations=('three committed candidates',),",
                        ")",
                    )
                ),
                encoding="utf-8",
            )
            service = SubprocessAgentExecutionService(
                root / "turns",
                hard_wall_seconds=3,
                capability_facade=UnavailableAgentCapabilityFacade(),
            )

            outcome = service.execute(
                _worker_program(script, gpu_devices=("2",)),
                _agent_context(),
                2,
            )

            self.assertEqual(len(outcome.nominations), 3)
            self.assertEqual(outcome.observations, ("three committed candidates",))
            self.assertEqual(
                outcome.round_log,
                "# Worker round log\n\nThree candidates tested.",
            )
            audit = service.audits[-1]
            self.assertEqual(audit.status, "completed")
            self.assertEqual(audit.nomination_count, 3)
            self.assertEqual(audit.rejected_nomination_count, 0)
            self.assertTrue(audit.capability_revoked)
            self.assertTrue(audit.capability_drained)
            self.assertTrue(audit.process_group_cleaned)
            self.assertTrue(audit.agent_process_isolated)
            self.assertFalse(audit.candidate_execution_isolated)
            self.assertFalse(audit.formal_private_scoring_claimed)
            workspace = Path(audit.workspace)
            request = json.loads((workspace / "request.json").read_text())
            self.assertEqual(request["schema"], "canonical-agent-worker-request-v2")
            self.assertIsNone(request["round"]["active_budget_seconds"])
            self.assertEqual(request["round"]["hard_wall_seconds"], 2.0)
            self.assertEqual(request["round"]["time_budget_mode"], "hard_wall")
            self.assertNotIn("truth", json.dumps(request).lower())
            self.assertEqual(
                (workspace / "CAMPAIGN_MEMORY.md").read_text(),
                "# Campaign Memory Log",
            )
            self.assertNotIn("markdown", request["campaign_memory"])
            self.assertNotIn("candidates", request["candidate_history"])
            self.assertEqual(
                stat.S_IMODE((workspace / "CAMPAIGN_MEMORY.md").stat().st_mode),
                0o400,
            )
            history_path = workspace / "CANDIDATE_HISTORY.json"
            self.assertEqual(
                json.loads(history_path.read_text()),
                {
                    "schema": "canonical-candidate-history-v1",
                    "candidates": [],
                },
            )
            self.assertEqual(stat.S_IMODE(history_path.stat().st_mode), 0o400)
            self.assertEqual(
                (workspace / "ROUND_LOG.md").read_text(),
                outcome.round_log,
            )
            self.assertIn("worker stdout", (workspace / "stdout.log").read_text())
            self.assertIn("worker stderr", (workspace / "stderr.log").read_text())
            self.assertTrue((workspace / "audit.json").is_file())
            self.assertEqual(
                stat.S_IMODE((workspace / "audit.json").stat().st_mode),
                0o400,
            )

    def test_worker_request_v1_is_explicitly_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            request = Path(temporary) / "request.json"
            request.write_text(
                json.dumps(
                    {
                        "schema": "canonical-agent-worker-request-v1",
                        "execution_id": "old-worker",
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(AgentExecutionError, "unknown schema"):
                load_agent_worker_request(request)

    def test_worker_request_loader_uses_the_controller_protocol_limit(self) -> None:
        expected = {
            "schema": "canonical-agent-worker-request-v2",
            "execution_id": "worker-limit",
        }
        request_path = Path("managed-request.json")
        with (
            patch.dict(
                os.environ,
                {
                    AGENT_EXECUTION_ID_ENV: "worker-limit",
                    AGENT_PROTOCOL_MAX_BYTES_ENV: "12345",
                },
                clear=False,
            ),
            patch(
                "pde_agent_verifier_solver.engine.agent_execution._read_json_object",
                return_value=expected,
            ) as reader,
        ):
            loaded = load_agent_worker_request(request_path)

        self.assertEqual(loaded, expected)
        reader.assert_called_once_with(request_path, 12345)

    def test_corrupted_round_log_is_rejected_without_exposing_target_data(self) -> None:
        variants = {
            "invalid-utf8": (
                "path.write_bytes(b'\\xff')",
                "invalid UTF-8",
            ),
            "oversized": (
                "path.write_bytes(b'x' * (int(os.environ["
                "'PDE_AGENT_ROUND_LOG_MAX_BYTES']) + 1))",
                "exceeds size limit",
            ),
            "symlink": (
                "path.unlink(); target = path.parent / 'private-target'; "
                "target.write_text('private-value'); path.symlink_to(target)",
                "cannot be opened",
            ),
        }
        for name, (mutation, expected_error) in variants.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                script = root / "corrupt_log.py"
                script.write_text(
                    "import os\n"
                    "from pathlib import Path\n"
                    "from pde_agent_verifier_solver.engine.agent_execution import "
                    "write_agent_worker_response\n"
                    "path = Path(os.environ['PDE_AGENT_ROUND_LOG'])\n"
                    f"{mutation}\n"
                    "write_agent_worker_response()\n",
                    encoding="utf-8",
                )
                service = SubprocessAgentExecutionService(
                    root / "turns",
                    hard_wall_seconds=3,
                    capability_facade=UnavailableAgentCapabilityFacade(),
                )

                outcome = service.execute(_worker_program(script), _agent_context(), 2)

                self.assertEqual(outcome.round_log, "")
                self.assertEqual(len(outcome.observations), 1)
                self.assertIn(expected_error, outcome.observations[0])
                self.assertNotIn("private-value", outcome.observations[0])

    def test_candidate_history_sidecar_serializes_complete_official_state(self) -> None:
        context = _agent_context()
        artifact = ToyResearchAgent._artifact(context, 2, suffix="history")
        history = CandidateHistory()
        history.add_candidate(artifact)
        history.record_official_result(
            artifact.artifact_id,
            {
                "guided": PromotionMetricSummary(
                    case_count=3,
                    mean=0.2,
                    cvar=0.4,
                    maximum=0.5,
                    metrics={"relative_l2": 0.2},
                )
            },
            ("perturbed:evaluation_failed",),
            technically_eligible=False,
        )
        document = SubprocessAgentExecutionService._candidate_history_document(
            replace(context, candidate_history=history.snapshot())
        )

        # A JSON round trip is the exact format consumed by an external worker.
        restored = json.loads(json.dumps(document))
        candidate = restored["candidates"][0]
        self.assertEqual(restored["schema"], "canonical-candidate-history-v1")
        self.assertEqual(candidate["artifact_id"], artifact.artifact_id)
        self.assertEqual(candidate["status"], "ineligible")
        self.assertEqual(candidate["failures"], ["perturbed:evaluation_failed"])
        self.assertEqual(candidate["official_distributions"]["guided"]["mean"], 0.2)
        self.assertNotIn("observations", candidate)

    def test_round_log_helper_is_bound_to_the_managed_path_and_size(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ROUND_LOG.md"
            environment = {
                AGENT_ROUND_LOG_ENV: str(path),
                AGENT_ROUND_LOG_MAX_BYTES_ENV: "4",
            }
            with patch.dict(os.environ, environment, clear=False):
                with self.assertRaisesRegex(ValueError, "cannot be empty"):
                    write_agent_round_log("  ")
                with self.assertRaisesRegex(AgentExecutionError, "exceeds size limit"):
                    write_agent_round_log("12345")
                written = write_agent_round_log("1234")

            self.assertEqual(written, path)
            self.assertEqual(path.read_text(encoding="utf-8"), "1234")

    def test_subprocess_allows_a_completed_zero_nomination_turn(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = root / "worker.py"
            script.write_text(
                "from pde_agent_verifier_solver.engine.agent_execution import "
                "write_agent_worker_response\n"
                "write_agent_worker_response(observations=('no candidate',))\n",
                encoding="utf-8",
            )
            service = SubprocessAgentExecutionService(
                root / "turns",
                hard_wall_seconds=3,
                capability_facade=UnavailableAgentCapabilityFacade(),
            )

            outcome = service.execute(_worker_program(script), _agent_context(), 2)

            self.assertEqual(outcome.nominations, ())
            self.assertEqual(outcome.observations, ("no candidate",))
            self.assertEqual(service.audits[-1].status, "completed")
            self.assertEqual(service.audits[-1].nomination_count, 0)

    def test_timeout_revokes_capability_kills_group_and_recovers_nomination(self) -> None:
        class RecordingFacade:
            def __init__(self):
                self.opened = []
                self.revoked = False

            def open(self, context, workspace, execution_id, program):
                self.opened.append((context, workspace, execution_id, program))
                return AgentCapabilityLease(
                    {
                        "schema": AGENT_CAPABILITY_SCHEMA,
                        "sampling": {"available": True, "transport": "test"},
                        "evaluation": {"available": True, "transport": "test"},
                    },
                    revoke=self._revoke,
                )

            def _revoke(self):
                self.revoked = True

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = root / "worker.py"
            script.write_text(
                "\n".join(
                    (
                        "import os",
                        "import subprocess",
                        "import sys",
                        "import time",
                        "from pathlib import Path",
                        "from pde_agent_verifier_solver.engine.agent_execution import (",
                        "    load_agent_worker_request, submit_agent_nomination,",
                        "    write_agent_round_log,",
                        ")",
                        "request = load_agent_worker_request()",
                        "assert request['capabilities']['sampling']['transport'] == 'test'",
                        "write_agent_round_log('# Partial timeout log\\n\\nNomination committed.')",
                        "submit_agent_nomination({'mode': 2}, sequence=1)",
                        "child = subprocess.Popen([sys.executable, '-c',",
                        "    'import time; time.sleep(60)'])",
                        "Path(os.environ['PDE_AGENT_WORKSPACE'], 'child.pid').write_text(",
                        "    str(child.pid), encoding='utf-8')",
                        "time.sleep(60)",
                    )
                ),
                encoding="utf-8",
            )
            facade = RecordingFacade()
            service = SubprocessAgentExecutionService(
                root / "turns",
                hard_wall_seconds=2,
                capability_facade=facade,
                termination_grace_seconds=0.2,
                poll_interval_seconds=0.01,
            )

            outcome = service.execute(_worker_program(script), _agent_context(), 1.5)

            self.assertEqual(len(outcome.nominations), 1)
            self.assertEqual(outcome.execution_status, "timed_out")
            self.assertIn("preserved 1 committed nomination", outcome.observations[-1])
            self.assertEqual(
                outcome.round_log,
                "# Partial timeout log\n\nNomination committed.",
            )
            audit = service.audits[-1]
            self.assertEqual(audit.status, "hard_wall_timeout")
            self.assertTrue(audit.capability_revoked)
            self.assertTrue(audit.capability_drained)
            self.assertTrue(facade.revoked)
            self.assertEqual(len(facade.opened), 1)
            self.assertTrue(audit.process_group_cleaned)
            with self.assertRaises(ProcessLookupError):
                os.killpg(audit.pid, 0)

    def test_worker_failure_without_nomination_is_audited_and_raises(self) -> None:
        class RecordingFacade:
            revoked = False

            def open(self, context, workspace, execution_id, program):
                del context, workspace, execution_id, program
                return AgentCapabilityLease(
                    {"schema": AGENT_CAPABILITY_SCHEMA},
                    revoke=lambda: setattr(self, "revoked", True),
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            facade = RecordingFacade()
            service = SubprocessAgentExecutionService(
                root / "turns",
                hard_wall_seconds=3,
                capability_facade=facade,
            )
            program = CommandAgentProgram(
                (sys.executable, "-c", "import sys; print('boom', file=sys.stderr); sys.exit(7)"),
                _decode_toy_nomination,
            )

            with self.assertRaisesRegex(AgentExecutionError, "status 7"):
                service.execute(program, _agent_context(), 2)

            audit = service.audits[-1]
            self.assertEqual(audit.status, "worker_failed")
            self.assertEqual(audit.return_code, 7)
            self.assertTrue(audit.capability_revoked)
            self.assertTrue(facade.revoked)
            self.assertIn("boom", Path(audit.stderr_path).read_text())

    def test_invalid_nomination_fails_closed_and_is_counted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = root / "worker.py"
            script.write_text(
                "from pde_agent_verifier_solver.engine.agent_execution import (\n"
                "    submit_agent_nomination, write_agent_worker_response,\n"
                ")\n"
                "submit_agent_nomination({'not_mode': 1}, sequence=1)\n"
                "write_agent_worker_response()\n",
                encoding="utf-8",
            )
            service = SubprocessAgentExecutionService(
                root / "turns",
                hard_wall_seconds=3,
                capability_facade=UnavailableAgentCapabilityFacade(),
            )

            with self.assertRaisesRegex(AgentExecutionError, "invalid nomination"):
                service.execute(_worker_program(script), _agent_context(), 2)

            audit = service.audits[-1]
            self.assertEqual(audit.status, "protocol_error")
            self.assertEqual(audit.nomination_count, 0)
            self.assertEqual(audit.rejected_nomination_count, 1)

    def test_mixed_valid_and_invalid_nominations_reject_the_entire_turn(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = root / "worker.py"
            script.write_text(
                "from pde_agent_verifier_solver.engine.agent_execution import (\n"
                "    submit_agent_nomination, write_agent_worker_response,\n"
                ")\n"
                "submit_agent_nomination({'mode': 1}, sequence=1)\n"
                "submit_agent_nomination({'not_mode': 2}, sequence=2)\n"
                "write_agent_worker_response()\n",
                encoding="utf-8",
            )
            service = SubprocessAgentExecutionService(
                root / "turns",
                hard_wall_seconds=3,
                capability_facade=UnavailableAgentCapabilityFacade(),
            )

            with self.assertRaisesRegex(AgentExecutionError, "invalid nomination"):
                service.execute(_worker_program(script), _agent_context(), 2)

            audit = service.audits[-1]
            self.assertEqual(audit.status, "protocol_error")
            self.assertEqual(audit.nomination_count, 1)
            self.assertEqual(audit.rejected_nomination_count, 1)

    def test_nomination_journal_sequence_must_be_contiguous(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = root / "worker.py"
            script.write_text(
                "import json\n"
                "import os\n"
                "from pathlib import Path\n"
                "from pde_agent_verifier_solver.engine.agent_execution import (\n"
                "    AGENT_NOMINATION_SCHEMA, AGENT_EXECUTION_ID_ENV,\n"
                "    AGENT_NOMINATIONS_ENV, write_agent_worker_response,\n"
                ")\n"
                "execution_id = os.environ[AGENT_EXECUTION_ID_ENV]\n"
                "path = Path(os.environ[AGENT_NOMINATIONS_ENV]) / "
                "'nomination-00000002.json'\n"
                "path.write_text(json.dumps({\n"
                "    'schema': AGENT_NOMINATION_SCHEMA,\n"
                "    'execution_id': execution_id,\n"
                "    'sequence': 2,\n"
                "    'payload': {'mode': 1},\n"
                "}))\n"
                "write_agent_worker_response()\n",
                encoding="utf-8",
            )
            service = SubprocessAgentExecutionService(
                root / "turns",
                hard_wall_seconds=3,
                capability_facade=UnavailableAgentCapabilityFacade(),
            )

            with self.assertRaisesRegex(AgentExecutionError, "invalid nomination"):
                service.execute(_worker_program(script), _agent_context(), 2)

            audit = service.audits[-1]
            self.assertEqual(audit.status, "protocol_error")
            self.assertEqual(audit.nomination_count, 0)
            self.assertEqual(audit.rejected_nomination_count, 1)

    def test_control_flow_exceptions_clean_up_and_are_re_raised(self) -> None:
        class InterruptingService(SubprocessAgentExecutionService):
            raised_error = None

            def _wait(self, process, lease, started, active_budget):
                del process, lease, started, active_budget
                raise self.raised_error

        for error_type in (KeyboardInterrupt, SystemExit):
            with self.subTest(error_type=error_type.__name__):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    service = InterruptingService(
                        root / "turns",
                        hard_wall_seconds=3,
                        capability_facade=UnavailableAgentCapabilityFacade(),
                        termination_grace_seconds=0.1,
                        poll_interval_seconds=0.01,
                    )
                    service.raised_error = error_type("controller interrupted")
                    program = CommandAgentProgram(
                        (sys.executable, "-c", "import time; time.sleep(60)"),
                        _decode_toy_nomination,
                    )

                    with self.assertRaises(error_type):
                        service.execute(program, _agent_context(), 2)

                    audit = service.audits[-1]
                    self.assertEqual(audit.status, "execution_error")
                    self.assertTrue(audit.capability_revoked)
                    self.assertTrue(audit.capability_drained)
                    self.assertTrue(audit.process_group_cleaned)
                    with self.assertRaises(ProcessLookupError):
                        os.killpg(audit.pid, 0)

    def test_unconfirmed_process_group_cleanup_never_decodes_nomination(self) -> None:
        decoded = []

        def decoder(payload, context, workspace):
            decoded.append((payload, context, workspace))
            return _decode_toy_nomination(payload, context, workspace)

        class UnconfirmedCleanupService(SubprocessAgentExecutionService):
            def _terminate_process_group(self, process):
                super()._terminate_process_group(process)
                return False

            def _terminate_remaining_process_group(self, process):
                super()._terminate_remaining_process_group(process)
                return False

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = root / "worker.py"
            script.write_text(
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
            service = UnconfirmedCleanupService(
                root / "turns",
                hard_wall_seconds=3,
                capability_facade=UnavailableAgentCapabilityFacade(),
                termination_grace_seconds=0.1,
                poll_interval_seconds=0.01,
            )
            program = CommandAgentProgram(
                (sys.executable, str(script)),
                decoder,
                environment={"PYTHONPATH": str(_SOURCE_ROOT)},
            )

            with self.assertRaisesRegex(
                AgentExecutionError,
                "cleanup could not be confirmed",
            ):
                service.execute(program, _agent_context(), 2)

            self.assertEqual(decoded, [])
            audit = service.audits[-1]
            self.assertFalse(audit.process_group_cleaned)
            self.assertEqual(audit.nomination_count, 0)
            self.assertIn("nomination recovery skipped", audit.error)
            with self.assertRaises(ProcessLookupError):
                os.killpg(audit.pid, 0)

    def test_active_budget_and_audit_exclude_setup_and_nomination_decode(self) -> None:
        class SlowSetupFacade(UnavailableAgentCapabilityFacade):
            def open(self, context, workspace, execution_id, program):
                time.sleep(0.25)
                return super().open(context, workspace, execution_id, program)

        def slow_decoder(payload, context, workspace):
            time.sleep(0.25)
            return _decode_toy_nomination(payload, context, workspace)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = root / "worker.py"
            script.write_text(
                "import time\n"
                "from pde_agent_verifier_solver.engine.agent_execution import (\n"
                "    submit_agent_nomination, write_agent_worker_response,\n"
                ")\n"
                "time.sleep(0.05)\n"
                "submit_agent_nomination({'mode': 1}, sequence=1)\n"
                "write_agent_worker_response()\n",
                encoding="utf-8",
            )
            service = SubprocessAgentExecutionService(
                root / "turns",
                hard_wall_seconds=5,
                capability_facade=SlowSetupFacade(),
                poll_interval_seconds=0.01,
            )
            program = CommandAgentProgram(
                (sys.executable, str(script)),
                slow_decoder,
                environment={"PYTHONPATH": str(_SOURCE_ROOT)},
            )

            external_started = time.monotonic()
            outcome = service.execute(program, _agent_context(), 3.0)
            external_seconds = time.monotonic() - external_started

            self.assertEqual(len(outcome.nominations), 1)
            audit = service.audits[-1]
            self.assertEqual(audit.status, "completed")
            self.assertLess(audit.active_seconds, 3.0)
            self.assertLess(audit.wall_seconds, 3.0)
            self.assertGreater(external_seconds, 0.5)
            self.assertGreater(external_seconds - audit.wall_seconds, 0.4)
            self.assertAlmostEqual(
                audit.finished_at_unix - audit.started_at_unix,
                audit.wall_seconds,
                delta=0.05,
            )

    def test_cleanup_time_is_separate_from_agent_deadline_observation(self) -> None:
        class SlowDrainFacade(UnavailableAgentCapabilityFacade):
            def open(self, context, workspace, execution_id, program):
                lease = super().open(context, workspace, execution_id, program)
                return AgentCapabilityLease(
                    lease.manifest, drain=lambda: time.sleep(0.25),
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = SubprocessAgentExecutionService(
                root / "turns", hard_wall_seconds=3,
                capability_facade=SlowDrainFacade(), poll_interval_seconds=0.01,
            )
            program = CommandAgentProgram(
                (sys.executable, "-c",
                 "from pde_agent_verifier_solver.engine.agent_execution import "
                 "write_agent_worker_response; write_agent_worker_response()"),
                _decode_toy_nomination,
                environment={"PYTHONPATH": str(_SOURCE_ROOT)},
            )
            service.execute(program, _agent_context(), 2)
            audit = service.audits[-1]
            self.assertEqual(audit.status, "completed")
            self.assertGreaterEqual(audit.cleanup_wall_seconds, 0.25)
            self.assertLess(audit.agent_execution_wall_seconds, 2)
            self.assertAlmostEqual(
                audit.wall_seconds,
                audit.agent_execution_wall_seconds + audit.cleanup_wall_seconds,
            )
            recorded = json.loads((Path(audit.workspace) / "audit.json").read_text())
            self.assertEqual(recorded["cleanup_wall_seconds"], audit.cleanup_wall_seconds)

    def test_nomination_journal_commit_is_atomic_and_non_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            committed = submit_agent_nomination(
                {"mode": 1},
                sequence=1,
                directory=directory,
                execution_id="execution-1",
            )
            with self.assertRaises(FileExistsError):
                submit_agent_nomination(
                    {"mode": 2},
                    sequence=1,
                    directory=directory,
                    execution_id="execution-1",
                )

            self.assertEqual(json.loads(committed.read_text())["payload"], {"mode": 1})
            self.assertEqual(list(directory.glob("*.partial")), [])

    def test_product_application_requires_formal_execution_by_default(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Formal runs require an explicit budget-enforcing Agent execution",
        ):
            ProductApplication()
        with self.assertRaisesRegex(ValueError, "local development mode"):
            ProductApplication(agent_execution=InProcessAgentExecutionService())
        with self.assertRaisesRegex(
            ValueError,
            "explicit non-controller Candidate execution",
        ):
            ProductApplication(agent_execution=object())
        with self.assertRaisesRegex(
            ValueError,
            "In-process Candidate execution is a local development mode",
        ):
            ProductApplication(
                agent_execution=object(),
                candidate_execution=InProcessCandidateExecutionService(),
            )

    def test_local_development_explicitly_uses_in_process_execution(self) -> None:
        class RecordingAgent(ToyResearchAgent):
            context = None

            def run(self, context):
                self.context = context
                return super().run(context)

        controllers = []

        class CapturingController(CampaignController):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                controllers.append(self)

        panel = build_toy_panel()
        agent = RecordingAgent()
        panel.programs.agents["default-agent"] = agent
        panel.app = ProductApplication.local_development()

        with patch(
            "pde_agent_verifier_solver.app.service.CampaignController",
            CapturingController,
        ):
            launched = panel.create_and_launch(
                _run_values("default-agent", budget_seconds=31)
            )
            session = panel.app.wait(launched["run_id"], timeout=5)

        self.assertEqual(session.status, RunStatus.COMPLETED, session.error)
        self.assertEqual(len(controllers), 1)
        self.assertIsInstance(
            controllers[0].agent_execution,
            InProcessAgentExecutionService,
        )
        self.assertIsNotNone(agent.context)
        self.assertEqual(agent.context.round_plan.agent_budget_seconds, 31)

    def test_product_application_accepts_candidate_execution_service(self) -> None:
        class CandidateExecution:
            candidate_invocation_process = "test-worker"

            def execute_cases(self, *args, **kwargs):
                del args, kwargs
                raise AssertionError("not invoked by this composition test")

        candidate_execution = CandidateExecution()
        application = ProductApplication.local_development(
            candidate_execution=candidate_execution,
        )

        self.assertIs(application._candidate_execution, candidate_execution)

        with tempfile.TemporaryDirectory() as temporary:
            agent_execution = SubprocessAgentExecutionService(
                Path(temporary) / "turns",
                hard_wall_seconds=3,
            )
            formal = ProductApplication(
                agent_execution=agent_execution,
                candidate_execution=candidate_execution,
            )
            self.assertIs(formal._candidate_execution, candidate_execution)

        with self.assertRaisesRegex(TypeError, "must implement execute_cases"):
            ProductApplication.local_development(candidate_execution=object())

    def test_formal_application_subprocess_stops_a_sleeping_agent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = SubprocessAgentExecutionService(
                root / "turns",
                hard_wall_seconds=3,
                capability_facade=UnavailableAgentCapabilityFacade(),
                termination_grace_seconds=0.1,
                poll_interval_seconds=0.01,
            )
            panel = build_toy_panel()
            panel.programs.agents["sleeping-agent"] = CommandAgentProgram(
                (
                    sys.executable,
                    "-c",
                    "import os, time; from pathlib import Path; "
                    "Path(os.environ['PDE_AGENT_ROUND_LOG']).write_text("
                    "'# Timeout notes\\n\\nCheckpointed.', encoding='utf-8'); "
                    "time.sleep(60)",
                ),
                _decode_toy_nomination,
                environment={"PYTHONPATH": str(_SOURCE_ROOT)},
            )
            panel.app = ProductApplication.local_development(
                agent_execution=service
            )

            started_at = time.monotonic()
            launched = panel.create_and_launch(
                _run_values("sleeping-agent", budget_seconds=2)
            )
            session = panel.app.wait(launched["run_id"], timeout=5)

            self.assertEqual(session.status, RunStatus.FAILED)
            self.assertLess(time.monotonic() - started_at, 4)
            self.assertEqual(service.audits[-1].status, "hard_wall_timeout")
            self.assertTrue(service.audits[-1].process_group_cleaned)
            failed = tuple(
                event
                for event in session.events.snapshot()
                if event.event_type == "agent_failed"
            )
            self.assertEqual(len(failed), 1)
            self.assertIn("agent_timeout", failed[0].message)
            self.assertEqual(failed[0].payload["reason_code"], "agent_timeout")
            page = panel.run_page(launched["run_id"])
            self.assertEqual(
                page["memory"]["rounds"][0]["agents"][0]["markdown"],
                "# Timeout notes\n\nCheckpointed.",
            )

    def test_custom_execution_receives_context_budget_and_uncapped_portfolio(self) -> None:
        class PortfolioAgent:
            context = None

            def run(self, context):
                self.context = context
                artifacts = tuple(
                    ToyResearchAgent._artifact(
                        context,
                        1 + index % 5,
                        suffix=f"portfolio-{index}",
                    )
                    for index in range(40)
                )
                return AgentOutcome(
                    artifacts[0],
                    ("Submitted an uncapped portfolio through the execution service.",),
                    artifacts=artifacts[1:],
                )

        class RecordingExecutionService:
            def __init__(self):
                self.calls = []

            def execute(self, program, context, active_budget_seconds):
                self.calls.append((program, context, active_budget_seconds))
                return program.run(context)

        controllers = []

        class CapturingController(CampaignController):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                controllers.append(self)

        panel = build_toy_panel()
        agent = PortfolioAgent()
        execution = RecordingExecutionService()
        panel.programs.agents["portfolio-agent"] = agent
        panel.app = ProductApplication.local_development(agent_execution=execution)

        with patch(
            "pde_agent_verifier_solver.app.service.CampaignController",
            CapturingController,
        ):
            launched = panel.create_and_launch(
                _run_values("portfolio-agent", budget_seconds=73)
            )
            session = panel.app.wait(launched["run_id"], timeout=5)

        self.assertEqual(session.status, RunStatus.COMPLETED, session.error)
        self.assertEqual(len(controllers), 1)
        self.assertIs(controllers[0].agent_execution, execution)
        self.assertEqual(len(execution.calls), 1)
        program, context, active_budget_seconds = execution.calls[0]
        self.assertIs(program, agent)
        self.assertIs(context, agent.context)
        self.assertEqual(active_budget_seconds, 73)
        self.assertEqual(context.round_plan.agent_budget_seconds, 73)
        self.assertEqual(len(session.result.rounds[0].candidate_evaluations), 40)
        finished = next(
            event
            for event in session.events.snapshot()
            if event.event_type == "agent_finished"
        )
        self.assertEqual(finished.payload["nomination_count"], 40)


if __name__ == "__main__":
    unittest.main()


def test_worker_receives_one_original_monotonic_deadline(tmp_path):
    from pde_agent_verifier_solver.engine.agent_execution import AGENT_DEADLINE_MONOTONIC_ENV
    script = tmp_path / "deadline_worker.py"
    script.write_text("\n".join([
        "import json, os, time",
        "from pathlib import Path",
        "from pde_agent_verifier_solver.engine.agent_execution import write_agent_worker_response",
        f"deadline = float(os.environ[{AGENT_DEADLINE_MONOTONIC_ENV!r}])",
        "first = deadline - time.monotonic()",
        "time.sleep(0.1)",
        f"assert float(os.environ[{AGENT_DEADLINE_MONOTONIC_ENV!r}]) == deadline",
        "Path('deadline-observation.json').write_text(json.dumps([first, deadline-time.monotonic()]))",
        "write_agent_worker_response()",
    ]))
    service = SubprocessAgentExecutionService(tmp_path / "turns", hard_wall_seconds=3,
        capability_facade=UnavailableAgentCapabilityFacade())
    outcome = service.execute(_worker_program(script), _agent_context(), 3)
    assert outcome.execution_status == "completed"
    first, later = json.loads((Path(service.audits[-1].workspace) / "deadline-observation.json").read_text())
    assert 0 < later < first <= 3
    assert first - later >= 0.09


def test_agent_program_cannot_replace_managed_deadline(tmp_path):
    import pytest
    from pde_agent_verifier_solver.engine.agent_execution import AGENT_DEADLINE_MONOTONIC_ENV
    with pytest.raises(ValueError, match="managed values"):
        CommandAgentProgram((sys.executable, "unused.py"), _decode_toy_nomination,
                            environment={AGENT_DEADLINE_MONOTONIC_ENV: "9999999999"})
