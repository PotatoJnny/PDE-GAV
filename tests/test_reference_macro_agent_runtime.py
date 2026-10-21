from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import time
import textwrap
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np

from pde_agent_verifier_solver.engine.candidate_case import candidate_facing_case
from pde_agent_verifier_solver.engine.development_session import (
    DevelopmentSession,
)
from pde_agent_verifier_solver.engine.guidance_controller import (
    DevelopmentGuidanceOracle,
)
from pde_agent_verifier_solver.engine.macro_closed_loop import (
    CompetitiveSearchPolicy,
    MacroAgentContext,
    MacroAgentSpec,
    SearchLeaderboardEntry,
)
from pde_agent_verifier_solver.engine.macro_evaluator import (
    DevelopmentMacroEvaluator,
    PreparedMacroCohort,
)
from pde_agent_verifier_solver.engine.reference_agent_runtime import (
    _ReferenceHTTPServer,
    _finalize_reference_agent_launch_audit,
    _launch_audit,
    _reference_agent_timing,
    _reference_candidate_submission,
    ReferenceAgentLaunchRequest,
    ReferenceControllerServiceClock,
    ReferenceDevelopmentEvaluatorRPC,
    ReferenceAgentLaunchError,
    ReferenceAgentRuntimeError,
    ReferenceEvaluatorClientError,
    ReferenceFilesystemBundleAttestor,
    ReferenceFilesystemMacroAgentProgram,
    ReferenceGuidanceOracleProxy,
    ReferenceGuidanceOracleRPC,
    SubprocessReferenceAgentLauncher,
    evaluate_reference_candidate,
    load_reference_completed_answer_release,
    screen_reference_candidate,
    submit_reference_candidate_screen,
    poll_reference_candidate_screen,
    load_python_macro_candidate,
    load_reference_public_workspace,
    python_bundle_sha256,
    query_reference_guided_completed_solution,
    query_reference_public_oracle,
    release_reference_completed_answers,
    sample_reference_guidance,
    write_reference_candidate_submission,
)
from pde_agent_verifier_solver.sdk import (
    AgentBundleSubmission,
    AnswerQuery,
    AnswerView,
    CandidateRef,
    CandidateSubmission,
    CandidateGuidanceSnapshot,
    CandidateGuidedSample,
    CandidateGuidedSampleBatch,
    Case,
    CaseRecord,
    CaseSource,
    CaseVisibility,
    FrozenGuidanceSnapshot,
    GuidanceCapability,
    GuidanceTrainingBudget,
    GuidanceTrainingRequest,
    GuidedAnswerFirstSample,
    GuidedSampleSpec,
    MacroMatchResult,
    MacroTarget,
    ProgramInfo,
    PublicOracleMethod,
    PublicProblemBundle,
    PublicResource,
    candidate_ref_from_attestation,
    measure_prepare_solution,
)
from pde_agent_verifier_solver.sdk.candidate_bundle import (
    write_candidate_bundle_manifest,
)


QUERY = AnswerQuery.grid(fields=("density",), x=1)
SECRET_TARGET = 937.125

CANDIDATE_SOURCE = textwrap.dedent(
    """
    from pde_agent_verifier_solver.sdk import AnswerView

    def predict_macros(case, query, budget=None):
        return AnswerView(
            kind=query.kind,
            fields={"density": (0.0,)},
            axes={"x": (0.0,)},
            shape=(1,),
        )
    """
).lstrip()


def _write_multifile_candidate(
    root: Path,
    value: float,
    *,
    reverse_manifest_order: bool = False,
) -> Path:
    root.mkdir(parents=True)
    (root / "assets").mkdir()
    (root / "helper.py").write_text(
        f"OFFSET = {value!r}\n",
        encoding="utf-8",
    )
    (root / "assets" / "offset.txt").write_text("0.25\n", encoding="utf-8")
    entrypoint = root / "submission.py"
    entrypoint.write_text(
        textwrap.dedent(
            """
            from pathlib import Path
            from .helper import OFFSET
            from pde_agent_verifier_solver.sdk import AnswerView

            ASSET_OFFSET = float(
                (Path(__file__).parent / "assets" / "offset.txt").read_text()
            )

            def predict_macros(case, query, budget=None):
                return AnswerView(
                    kind=query.kind,
                    fields={"density": (OFFSET + ASSET_OFFSET,)},
                    axes={"x": (0.0,)},
                    shape=(1,),
                )
            """
        ).lstrip(),
        encoding="utf-8",
    )
    files = ["submission.py", "helper.py", "assets/offset.txt"]
    write_candidate_bundle_manifest(
        root,
        files=reversed(files) if reverse_manifest_order else files,
    )
    return entrypoint


class _Truth:
    def answer(self, query):
        return AnswerView(
            kind=query.kind,
            fields={"density": (SECRET_TARGET,)},
            axes={"x": (0.0,)},
            shape=(1,),
        )


class _LargeTruth:
    def __init__(self, size):
        self.values = tuple(
            SECRET_TARGET + index / 7.0
            for index in range(size)
        )

    def answer(self, query):
        return AnswerView(
            kind=query.kind,
            fields={"density": self.values},
            axes={"x": (0.0,)},
            shape=(len(self.values),),
        )


class _FreshSessionFactory:
    def __init__(self, *, large_release_size=200_000, screen_error=None):
        self.large_release_size = large_release_size
        self.screen_error = screen_error
        self.requests = []
        self.root_ids = []

    def __call__(self, request):
        self.requests.append(request)
        if request.kind.value == "screen" and self.screen_error is not None:
            raise ValueError(self.screen_error)

        records = []
        for index in range(request.root_count):
            root_id = (
                f"{request.kind.value}-draw-{request.draw_index}-root-{index}"
            )
            self.root_ids.append(root_id)
            case = Case(
                case_id=root_id,
                source=CaseSource.GENERATED,
                visibility=CaseVisibility.DEVELOPMENT,
                public_data={"fresh_index": index},
                round_index=1,
                tags=("fresh-development",),
            )
            truth = (
                _LargeTruth(self.large_release_size)
                if request.kind.value == "release"
                else _Truth()
            )
            records.append(CaseRecord(case, truth))
        return PreparedMacroCohort.prepare(
            cohort_id=f"fresh-development-{request.draw_index}",
            records=records,
            matcher=_Matcher(),
            query=QUERY,
        )


class _Matcher:
    info = ProgramInfo("reference-test", "Reference test", "Reference test")
    contract_id = "reference_test_macro_v1"
    fields = ("density",)

    def prepare_target(self, case, truth, query):
        return MacroTarget(case.case_id, self.contract_id, query, truth)

    def match(self, case, prediction, target):
        actual = float(prediction.fields["density"][0])
        expected = float(target.values.fields["density"][0])
        error = abs(actual - expected) / abs(expected)
        return MacroMatchResult(
            case_id=case.case_id,
            contract_id=self.contract_id,
            primary_error=error,
            metrics={"private_detail": expected},
            passed=error < 0.1,
            failures=() if error < 0.1 else ("full private gate detail",),
        )

    def result_card(self, result):
        raise NotImplementedError


class _ConcurrentMatcher(_Matcher):
    def __init__(self):
        self._state_lock = threading.Lock()
        self._active = 0
        self.max_active = 0
        self._simultaneous = threading.Event()

    def match(self, case, prediction, target):
        with self._state_lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
            if self._active == 2:
                self._simultaneous.set()

        self._simultaneous.wait(timeout=0.25)
        try:
            return super().match(
                case,
                prediction,
                target,
            )
        finally:
            with self._state_lock:
                self._active -= 1


def _guidance_digest(label):
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _runtime_guidance_snapshot():
    request = GuidanceTrainingRequest(
        source_round_index=0,
        target_round_index=1,
        base_generator_sha256=_guidance_digest("base-generator"),
        target_solver_sha256=_guidance_digest("target-solver"),
        reward_matcher_definition_sha256=_guidance_digest("matcher"),
        reward_feedback_uri="sealed://reward-feedback",
        reward_feedback_sha256=_guidance_digest("reward-feedback"),
        guidance_config_sha256=_guidance_digest("guidance-config"),
        controller_training_seed=17,
        training_budget=GuidanceTrainingBudget(
            30.0,
            "test-compute",
            1,
        ),
    )
    return FrozenGuidanceSnapshot(
        training_request=request,
        backend_id="runtime-guidance-v1",
        guided_artifact_uri="sealed://private-guidance-artifact",
        guided_artifact_sha256=_guidance_digest("private-guidance-artifact"),
        campaign_guidance_training_wall_seconds=1.0,
    )


class _RuntimeFrozenGuidanceOracle:
    guidance_capability = GuidanceCapability.online("runtime-guidance-v1")
    development_solution_methods = ("query_values",)

    def __init__(self):
        self.snapshot = _runtime_guidance_snapshot()
        self.generated_requests = []
        self.opened_sample_ids = []
        self.sessions = []

    def generate_answer_first(self, request):
        self.generated_requests.append(request)
        samples = []
        for index in range(request.spec.sample_count):
            identity = (
                f"{request.request_id}-{request.controller_seed}-"
                f"{request.controller_nonce}-{index}"
            )
            samples.append(
                GuidedAnswerFirstSample(
                    sample_id=f"guided-{identity}",
                    snapshot_sha256=request.snapshot_sha256,
                    case=Case(
                        case_id=f"guided-case-{identity}",
                        source=CaseSource.GENERATED,
                        visibility=CaseVisibility.DEVELOPMENT,
                        public_data={
                            "setting": request.spec.settings.get("focus"),
                            "public_array": np.asarray(
                                [index, index + 1],
                                dtype=np.float64,
                            ),
                        },
                        round_index=1,
                        tags=("guided-development",),
                    ),
                    reference_answer_sha256=_guidance_digest(
                        f"private-answer-{identity}"
                    ),
                    public_case_sha256=_guidance_digest(
                        f"public-case-{identity}"
                    ),
                    answer_to_case_derivation_sha256=_guidance_digest(
                        "runtime-answer-to-case-v1"
                    ),
                    answer_access_locator=(
                        f"sealed://private-answer-locator/{identity}"
                    ),
                )
            )
        return tuple(samples)

    def open_development_solution_session(self, sample_id):
        self.opened_sample_ids.append(sample_id)
        owner = self

        class Session:
            def __init__(self):
                self.closed = False

            def query_values(self, coordinates, scale=1.0):
                values = np.asarray(coordinates, dtype=np.complex128)
                return {
                    "values": values * scale,
                    "marker": 2.0 + 3.0j,
                }

            def close(self):
                self.closed = True

        session = Session()
        owner.sessions.append(session)
        return session


class ReferenceAgentRuntimeTests(unittest.TestCase):
    def test_controller_service_clock_excludes_only_committed_service(self):
        clock = ReferenceControllerServiceClock()

        with patch(
            "pde_agent_verifier_solver.engine.reference_agent_runtime."
            "time.monotonic",
            return_value=10.0,
        ):
            credited = clock.begin_service()
        self.assertEqual(clock.creditable_seconds(now=14.0), 4.0)
        with patch(
            "pde_agent_verifier_solver.engine.reference_agent_runtime."
            "time.monotonic",
            return_value=16.0,
        ):
            clock.end_service(credited, credit=True)
        self.assertEqual(clock.creditable_seconds(now=13.0), 3.0)

        with patch(
            "pde_agent_verifier_solver.engine.reference_agent_runtime."
            "time.monotonic",
            return_value=20.0,
        ):
            rolled_back = clock.begin_service()
        self.assertEqual(clock.creditable_seconds(now=24.0), 10.0)
        with patch(
            "pde_agent_verifier_solver.engine.reference_agent_runtime."
            "time.monotonic",
            return_value=25.0,
        ):
            clock.end_service(rolled_back, credit=False)

        wall, service, provisional, active = _reference_agent_timing(
            started=0.0,
            current=20.0,
            service_clock=clock,
        )
        self.assertEqual(
            (wall, service, provisional, active),
            (20.0, 6.0, 0.0, 14.0),
        )

    def test_launch_audit_finalizes_inflight_rollback_after_drain(self):
        clock = ReferenceControllerServiceClock()
        with patch(
            "pde_agent_verifier_solver.engine.reference_agent_runtime."
            "time.monotonic",
            return_value=5.0,
        ):
            token = clock.begin_service()
        preliminary = _launch_audit(
            status="completed",
            returncode=None,
            started=0.0,
            current=10.0,
            active_timeout_seconds=8.0,
            hard_timeout_seconds=20.0,
            service_clock=clock,
        )
        self.assertEqual(preliminary.controller_service_seconds, 5.0)
        self.assertEqual(preliminary.controller_service_provisional_seconds, 5.0)

        with patch(
            "pde_agent_verifier_solver.engine.reference_agent_runtime."
            "time.monotonic",
            return_value=15.0,
        ):
            clock.end_service(token, credit=False)
        finalized = _finalize_reference_agent_launch_audit(preliminary, clock)

        self.assertEqual(finalized.controller_service_seconds, 0.0)
        self.assertEqual(finalized.controller_service_provisional_seconds, 0.0)
        self.assertEqual(finalized.agent_active_wall_seconds, 10.0)
        self.assertTrue(finalized.payload()["controller_service_accounting_finalized"])
        self.assertEqual(finalized.status, "active_time_timeout")

    def test_subprocess_hard_wall_is_absolute(self):
        process = Mock()
        process.pid = 123_456
        process.poll.return_value = 0
        service_clock = ReferenceControllerServiceClock()
        with patch(
            "pde_agent_verifier_solver.engine.reference_agent_runtime."
            "time.monotonic",
            return_value=0.0,
        ):
            service_token = service_clock.begin_service()
        with patch(
            "pde_agent_verifier_solver.engine.reference_agent_runtime."
            "time.monotonic",
            return_value=5.0,
        ):
            service_clock.end_service(service_token, credit=True)
        launcher = SubprocessReferenceAgentLauncher(
            ("reference-agent",),
            timeout_seconds=10.0,
            hard_timeout_seconds=12.0,
        )

        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            request = ReferenceAgentLaunchRequest(
                workspace=workspace,
                context_path=workspace / "context.json",
                prompt_path=workspace / "PROMPT.md",
                submission_path=workspace / "submission.json",
                controller_service_clock=service_clock,
            )
            with (
                patch(
                    "pde_agent_verifier_solver.engine.reference_agent_runtime."
                    "subprocess.Popen",
                    return_value=process,
                ),
                patch(
                    "pde_agent_verifier_solver.engine.reference_agent_runtime."
                    "time.monotonic",
                    side_effect=(0.0, 12.0),
                ),
                patch(
                    "pde_agent_verifier_solver.engine.reference_agent_runtime."
                    "_terminate_reference_agent_process_group",
                ) as terminate,
                self.assertRaisesRegex(
                    ReferenceAgentLaunchError,
                    "hard wall time budget",
                ) as raised,
            ):
                launcher.launch(request)

        terminate.assert_called_once_with(process)
        process.wait.assert_not_called()
        audit = raised.exception.audit
        self.assertIsNotNone(audit)
        self.assertEqual(audit.status, "hard_wall_timeout")
        self.assertEqual(audit.returncode, 0)
        self.assertEqual(audit.wall_time_seconds, 12.0)
        self.assertEqual(audit.controller_service_seconds, 5.0)
        self.assertEqual(audit.agent_active_wall_seconds, 7.0)
        self.assertEqual(audit.hard_timeout_seconds, 12.0)

    def test_subprocess_preserves_stdout_and_stderr_on_nonzero_exit(self):
        launcher = SubprocessReferenceAgentLauncher(
            (
                sys.executable,
                "-c",
                (
                    "import sys; print('agent-event'); "
                    "print('quota-detail', file=sys.stderr); sys.exit(1)"
                ),
            ),
            timeout_seconds=10.0,
        )

        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            request = ReferenceAgentLaunchRequest(
                workspace=workspace,
                context_path=workspace / "context.json",
                prompt_path=workspace / "PROMPT.md",
                submission_path=workspace / "submission.json",
            )
            with self.assertRaisesRegex(
                ReferenceAgentLaunchError,
                "exited with code 1",
            ):
                launcher.launch(request)

            stdout_path = workspace / "agent.stdout.log"
            stderr_path = workspace / "agent.stderr.log"
            self.assertEqual(
                stdout_path.read_text(encoding="utf-8"),
                "agent-event\n",
            )
            self.assertEqual(
                stderr_path.read_text(encoding="utf-8"),
                "quota-detail\n",
            )
            self.assertEqual(stat.S_IMODE(stdout_path.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(stderr_path.stat().st_mode), 0o600)

    def test_wait_completion_after_hard_wall_is_rejected_after_revocation(self):
        process = Mock()
        process.pid = 123_456
        process.poll.return_value = None
        process.wait.return_value = 0
        service_clock = ReferenceControllerServiceClock()
        with patch(
            "pde_agent_verifier_solver.engine.reference_agent_runtime."
            "time.monotonic",
            return_value=0.0,
        ):
            token = service_clock.begin_service()
        with patch(
            "pde_agent_verifier_solver.engine.reference_agent_runtime."
            "time.monotonic",
            return_value=5.0,
        ):
            service_clock.end_service(token, credit=True)
        launcher = SubprocessReferenceAgentLauncher(
            ("reference-agent",),
            timeout_seconds=10.0,
            hard_timeout_seconds=12.0,
        )
        events = []

        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            request = ReferenceAgentLaunchRequest(
                workspace=workspace,
                context_path=workspace / "context.json",
                prompt_path=workspace / "PROMPT.md",
                submission_path=workspace / "submission.json",
                controller_service_clock=service_clock,
                revoke_evaluator_capability=lambda: events.append("revoke"),
            )
            with (
                patch(
                    "pde_agent_verifier_solver.engine.reference_agent_runtime."
                    "subprocess.Popen",
                    return_value=process,
                ),
                patch(
                    "pde_agent_verifier_solver.engine.reference_agent_runtime."
                    "time.monotonic",
                    side_effect=(0.0, 11.0, 13.0),
                ),
                patch(
                    "pde_agent_verifier_solver.engine.reference_agent_runtime."
                    "_terminate_reference_agent_process_group",
                    side_effect=lambda _process: events.append("terminate"),
                ),
                self.assertRaisesRegex(
                    ReferenceAgentLaunchError,
                    "hard wall time budget",
                ) as raised,
            ):
                launcher.launch(request)

        self.assertEqual(events, ["revoke", "terminate"])
        self.assertEqual(raised.exception.audit.status, "hard_wall_timeout")
        self.assertEqual(raised.exception.audit.wall_time_seconds, 13.0)
        process.wait.assert_called_once_with(timeout=0.1)

    def test_credited_service_can_cross_active_deadline_before_hard_wall(self):
        process = Mock()
        process.pid = 123_456
        process.poll.return_value = None
        process.wait.return_value = 0
        service_clock = ReferenceControllerServiceClock()
        with patch(
            "pde_agent_verifier_solver.engine.reference_agent_runtime."
            "time.monotonic",
            return_value=0.0,
        ):
            token = service_clock.begin_service()
        with patch(
            "pde_agent_verifier_solver.engine.reference_agent_runtime."
            "time.monotonic",
            return_value=3.0,
        ):
            service_clock.end_service(token, credit=True)
        launcher = SubprocessReferenceAgentLauncher(
            ("reference-agent",),
            timeout_seconds=5.0,
            hard_timeout_seconds=10.0,
        )
        revoke = Mock()

        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            request = ReferenceAgentLaunchRequest(
                workspace=workspace,
                context_path=workspace / "context.json",
                prompt_path=workspace / "PROMPT.md",
                submission_path=workspace / "submission.json",
                controller_service_clock=service_clock,
                revoke_evaluator_capability=revoke,
            )
            with (
                patch(
                    "pde_agent_verifier_solver.engine.reference_agent_runtime."
                    "subprocess.Popen",
                    return_value=process,
                ),
                patch(
                    "pde_agent_verifier_solver.engine.reference_agent_runtime."
                    "time.monotonic",
                    side_effect=(0.0, 5.0, 6.0),
                ),
                patch(
                    "pde_agent_verifier_solver.engine.reference_agent_runtime."
                    "_reference_agent_process_group_exists",
                    return_value=False,
                ),
            ):
                audit = launcher.launch(request)

        revoke.assert_called_once_with()
        self.assertEqual(audit.status, "completed")
        self.assertEqual(audit.wall_time_seconds, 6.0)
        self.assertEqual(audit.controller_service_seconds, 3.0)
        self.assertEqual(audit.agent_active_wall_seconds, 3.0)
        process.wait.assert_called_once_with(timeout=0.1)

    def test_rpc_and_launcher_share_service_clock_for_wait_exclusion(self):
        original_screen = DevelopmentSession.screen

        def delayed_screen(session, *args, **kwargs):
            time.sleep(0.02)
            return original_screen(session, *args, **kwargs)

        class ClockProbeLauncher:
            def launch(self, request):
                self.clock = request.controller_service_clock
                self.asserted_clock_type = isinstance(
                    self.clock,
                    ReferenceControllerServiceClock,
                )
                candidate = request.workspace / "candidate.py"
                candidate.write_text(CANDIDATE_SOURCE, encoding="utf-8")
                started = time.monotonic()
                screen_reference_candidate(
                    request.context_path,
                    candidate_path=candidate,
                    candidate_id="clock-probe",
                    requested_root_count=4,
                )
                finished = time.monotonic()
                wall, service, provisional, active = _reference_agent_timing(
                    started=started,
                    current=finished,
                    service_clock=self.clock,
                )
                self.timing = (wall, service, active)
                self.provisional = provisional
                write_reference_candidate_submission(
                    request.context_path,
                    candidate_id="clock-probe",
                    bundle_path=candidate,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            incumbent_ref, _ = _incumbent(root)
            session, _ = _strong_session(
                incumbent_ref,
                screen_credits=4,
                large_release_size=1,
            )
            launcher = ClockProbeLauncher()
            program = ReferenceFilesystemMacroAgentProgram(
                run_root=root / "run",
                run_id="shared-service-clock",
                launcher=launcher,
            )
            with patch.object(DevelopmentSession, "screen", delayed_screen):
                program.run(_agent_context(session, incumbent_ref))

        wall, service, active = launcher.timing
        self.assertTrue(launcher.asserted_clock_type)
        self.assertEqual(launcher.provisional, 0.0)
        self.assertGreaterEqual(service, 0.015)
        self.assertAlmostEqual(active, wall - service, places=6)
        self.assertLess(active, wall)

    def test_subprocess_workspace_rpc_credits_sealing_and_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluator = _evaluator(case_credit_limit=1)
            incumbent_ref, incumbent_bytes = _incumbent(root)
            script = root / "agent_driver.py"
            script.write_text(
                textwrap.dedent(
                    f"""
                    import json
                    import os
                    from pathlib import Path
                    import subprocess

                    from pde_agent_verifier_solver.engine.reference_agent_runtime import (
                        write_reference_candidate_submission,
                    )

                    workspace = Path(os.environ["PDE_MACRO_WORKSPACE"])
                    context = Path(os.environ["PDE_MACRO_CONTEXT"])
                    if "REFERENCE_CONTROLLER_SECRET" in os.environ:
                        raise SystemExit(22)

                    candidate = workspace / "candidate.py"
                    candidate.write_text({CANDIDATE_SOURCE!r}, encoding="utf-8")
                    metadata = json.loads(context.read_text(encoding="utf-8"))
                    command = metadata["evaluator"]["cli"] + [
                        "--candidate", "candidate.py",
                        "--candidate-id", "probe",
                        "--case-id", "case-1",
                    ]
                    first = subprocess.run(
                        command,
                        cwd=workspace,
                        text=True,
                        capture_output=True,
                        check=False,
                    )
                    second = subprocess.run(
                        command,
                        cwd=workspace,
                        text=True,
                        capture_output=True,
                        check=False,
                    )
                    (workspace / "first-evaluation.json").write_text(
                        first.stdout,
                        encoding="utf-8",
                    )
                    (workspace / "second-evaluation.json").write_text(
                        second.stderr,
                        encoding="utf-8",
                    )
                    if first.returncode != 0 or second.returncode != 2:
                        raise SystemExit(21)
                    write_reference_candidate_submission(
                        context,
                        candidate_id="round-3-candidate",
                        bundle_path=candidate,
                        observations=("one sanitized development probe",),
                    )
                    """
                ),
                encoding="utf-8",
            )
            launcher = SubprocessReferenceAgentLauncher(
                (sys.executable, str(script)),
                environment={
                    "PYTHONPATH": os.pathsep.join(
                        (str(Path(__file__).parents[1] / "src"),
                         os.environ.get("PYTHONPATH", ""))
                    )
                },
                timeout_seconds=30.0,
            )
            program = ReferenceFilesystemMacroAgentProgram(
                run_root=root / "run",
                run_id="fresh-u0-reference",
                launcher=launcher,
            )
            context = MacroAgentContext(
                round_index=3,
                agent=MacroAgentSpec("agent-a", "Agent A"),
                prompt="Improve the public macro prediction.",
                cases=evaluator.public_cases,
                evaluator=evaluator,
                incumbent=incumbent_ref,
                prior_handoff=None,
            )

            with patch.dict(
                os.environ,
                {"REFERENCE_CONTROLLER_SECRET": "must-not-cross-boundary"},
            ):
                outcome = program.run(context)

            ref = outcome.candidate.ref
            self.assertEqual(ref.round_index, 3)
            self.assertEqual(ref.agent_id, "agent-a")
            self.assertEqual(ref.parent_sha256, incumbent_ref.bundle_sha256)
            self.assertEqual(
                ref.candidate_id,
                "agent-a:r3:round-3-candidate",
            )
            self.assertEqual(outcome.observations, ("one sanitized development probe",))
            self.assertFalse(program.formal_claims_allowed)
            self.assertTrue(program.limitations)

            workspace = next((root / "run" / "round-0003").iterdir())
            private_directories = (
                root / "run",
                root / "run" / "round-0003",
                workspace,
                workspace / "arrays",
                workspace / "incumbent",
            )
            for directory in private_directories:
                self.assertEqual(stat.S_IMODE(directory.stat().st_mode) & 0o077, 0)
            context_mode = stat.S_IMODE((workspace / "context.json").stat().st_mode)
            self.assertEqual(context_mode, 0o400)
            terminal_context = json.loads(
                (workspace / "context.json").read_text(encoding="utf-8")
            )
            self.assertNotIn("token", terminal_context["evaluator"])
            self.assertEqual(
                terminal_context["evaluator"]["capability_status"],
                "expired",
            )
            protected_files = (
                workspace / "PROMPT.md",
                workspace / "incumbent" / "candidate.py",
                *tuple((workspace / "arrays").iterdir()),
            )
            for protected_file in protected_files:
                mode = stat.S_IMODE(protected_file.stat().st_mode)
                self.assertEqual(mode, 0o400)
            submission_mode = stat.S_IMODE(
                (workspace / "submission.json").stat().st_mode
            )
            self.assertEqual(submission_mode, 0o600)
            self.assertFalse((workspace / "launcher.stdout.log").exists())
            self.assertFalse((workspace / "launcher.stderr.log").exists())
            public = load_reference_public_workspace(workspace / "context.json")
            self.assertEqual(public.round_index, 3)
            self.assertEqual(public.agent_id, "agent-a")
            self.assertEqual(public.queries, (QUERY,))
            self.assertIsInstance(
                public.cases[0].public_data["large_public_array"],
                np.memmap,
            )
            np.testing.assert_array_equal(
                public.cases[0].public_data["large_public_array"],
                np.arange(2048, dtype=np.float64),
            )
            self.assertEqual(public.incumbent_path.read_bytes(), incumbent_bytes)
            self.assertEqual(
                hashlib.sha256(public.incumbent_path.read_bytes()).hexdigest(),
                incumbent_ref.bundle_sha256,
            )

            first = json.loads(
                (workspace / "first-evaluation.json").read_text(encoding="utf-8")
            )
            second = json.loads(
                (workspace / "second-evaluation.json").read_text(encoding="utf-8")
            )
            self.assertEqual(first["charged_case_credits"], 1)
            self.assertEqual(first["remaining_case_credits"], 0)
            self.assertEqual(first["cases"][0]["score"]["metrics"], {})
            self.assertEqual(
                first["cases"][0]["score"]["failures"],
                ["macro_gate_rejected"],
            )
            self.assertEqual(second["error"]["code"], "case_credit_exceeded")
            disclosed = (
                (workspace / "context.json").read_text(encoding="utf-8")
                + json.dumps(first, sort_keys=True)
                + json.dumps(second, sort_keys=True)
            )
            self.assertNotIn(str(SECRET_TARGET), disclosed)
            self.assertNotIn("target_sha256", disclosed)
            self.assertNotIn("private_detail", disclosed)
            self.assertNotIn("full private gate detail", disclosed)
            self.assertNotIn("formal_isolated", json.dumps(first))

            mutable_candidate = workspace / "candidate.py"
            mutable_candidate.write_text(
                CANDIDATE_SOURCE.replace("(0.0,)", "(9.0,)"),
                encoding="utf-8",
            )
            sealed_path = Path(ref.bundle_uri)
            self.assertEqual(
                hashlib.sha256(sealed_path.read_bytes()).hexdigest(),
                ref.bundle_sha256,
            )
            self.assertEqual(
                stat.S_IMODE(sealed_path.stat().st_mode) & 0o222,
                0,
            )
            prediction = outcome.candidate.program.predict_macros(
                public.cases[0],
                QUERY,
            )
            self.assertEqual(prediction.fields["density"], (0.0,))

    def test_launcher_failure_expires_evaluator_capability_and_keeps_context(self):
        class FailingLauncher:
            active_token = None
            endpoint = None

            def launch(self, request):
                active = json.loads(
                    request.context_path.read_text(encoding="utf-8")
                )
                self.active_token = active["evaluator"]["token"]
                self.endpoint = active["evaluator"]["endpoint"]
                if active["evaluator"]["capability_status"] != "active":
                    raise AssertionError("evaluator capability was not active")
                os.chmod(request.context_path, 0o600)
                raise RuntimeError("injected launcher failure")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluator = _evaluator(case_credit_limit=1)
            incumbent_ref, _ = _incumbent(root)
            launcher = FailingLauncher()
            program = ReferenceFilesystemMacroAgentProgram(
                run_root=root / "run",
                run_id="failed-launcher-cleanup",
                launcher=launcher,
            )

            with self.assertRaisesRegex(RuntimeError, "injected launcher failure"):
                program.run(_agent_context(evaluator, incumbent_ref))

            workspace = next((root / "run" / "round-0001").iterdir())
            context_path = workspace / "context.json"
            terminal_text = context_path.read_text(encoding="utf-8")
            terminal = json.loads(terminal_text)
            self.assertIsNotNone(launcher.active_token)
            self.assertNotIn(launcher.active_token, terminal_text)
            self.assertNotIn("token", terminal["evaluator"])
            self.assertEqual(
                terminal["evaluator"]["capability_status"],
                "expired",
            )
            self.assertEqual(terminal["evaluator"]["endpoint"], launcher.endpoint)
            self.assertEqual(stat.S_IMODE(context_path.stat().st_mode), 0o400)
            public = load_reference_public_workspace(context_path)
            self.assertEqual(public.run_id, "failed-launcher-cleanup")
            self.assertEqual(
                tuple(case.case_id for case in public.cases),
                tuple(case.case_id for case in evaluator.public_cases),
            )
            self.assertNotEqual(public.cases[0].case_id, "case-1")
            audit_path = workspace / "agent_launch_audit.json"
            audit_text = audit_path.read_text(encoding="utf-8")
            audit = json.loads(audit_text)
            self.assertEqual(stat.S_IMODE(audit_path.stat().st_mode), 0o400)
            self.assertEqual(
                audit["schema"],
                "reference-agent-launch-audit-v1",
            )
            self.assertEqual(audit["status"], "launcher_failed_before_timing")
            self.assertEqual(audit["failure"], {"exception_type": "RuntimeError"})
            self.assertFalse(audit["in_process_evaluator_preemption_guaranteed"])
            self.assertNotIn("injected launcher failure", audit_text)

    def test_cleanup_failures_do_not_mask_original_launch_exception(self):
        class FailingLauncher:
            def launch(self, request):
                raise RuntimeError("original launch failure")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluator = _evaluator(case_credit_limit=1)
            incumbent_ref, _ = _incumbent(root)
            program = ReferenceFilesystemMacroAgentProgram(
                run_root=root / "run",
                run_id="cleanup-failure-preservation",
                launcher=FailingLauncher(),
            )
            with (
                patch(
                    "pde_agent_verifier_solver.engine.reference_agent_runtime."
                    "_expire_reference_evaluator_capability",
                    side_effect=OSError("expiration failed"),
                ),
                patch(
                    "pde_agent_verifier_solver.engine.reference_agent_runtime."
                    "_write_reference_agent_launch_audit",
                    side_effect=OSError("audit failed"),
                ) as write_audit,
                self.assertRaisesRegex(
                    RuntimeError,
                    "original launch failure",
                ) as raised,
            ):
                program.run(_agent_context(evaluator, incumbent_ref))

        write_audit.assert_called_once()
        notes = getattr(raised.exception, "__notes__", ())
        self.assertIn("evaluator capability expiration failed: OSError", notes)
        self.assertIn("launch audit write failed: OSError", notes)

    def test_subprocess_timeout_expires_evaluator_capability(self):
        observed = {}

        def start_then_time_out(*args, **kwargs):
            context_path = Path(kwargs["env"]["PDE_MACRO_CONTEXT"])
            active = json.loads(context_path.read_text(encoding="utf-8"))
            observed["token"] = active["evaluator"]["token"]
            observed["endpoint"] = active["evaluator"]["endpoint"]
            observed["context_path"] = context_path
            observed["start_new_session"] = kwargs["start_new_session"]
            self.assertEqual(
                active["evaluator"]["capability_status"],
                "active",
            )
            process = Mock()
            process.pid = 123_456
            process.poll.return_value = None

            def wait_then_expire(*, timeout):
                time.sleep(timeout)
                raise subprocess.TimeoutExpired(args[0], timeout)

            process.wait.side_effect = wait_then_expire
            observed["process"] = process
            return process

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluator = _evaluator(case_credit_limit=1)
            incumbent_ref, _ = _incumbent(root)
            launcher = SubprocessReferenceAgentLauncher(
                ("reference-agent",),
                timeout_seconds=0.25,
            )
            program = ReferenceFilesystemMacroAgentProgram(
                run_root=root / "run",
                run_id="timeout-cleanup",
                launcher=launcher,
            )

            def probe_during_process_cleanup(process):
                candidate = Path(observed["context_path"]).parent / "candidate.py"
                candidate.write_text(CANDIDATE_SOURCE, encoding="utf-8")
                try:
                    evaluate_reference_candidate(
                        observed["context_path"],
                        candidate_path=candidate,
                        candidate_id="post-timeout-probe",
                        case_ids=("case-1",),
                    )
                except ReferenceEvaluatorClientError as exc:
                    observed["cleanup_rpc_error"] = (exc.code, exc.message)
                else:
                    raise AssertionError("RPC remained active during TERM grace")

            with (
                patch(
                    "pde_agent_verifier_solver.engine.reference_agent_runtime."
                    "subprocess.Popen",
                    side_effect=start_then_time_out,
                ),
                patch(
                    "pde_agent_verifier_solver.engine.reference_agent_runtime."
                    "_terminate_reference_agent_process_group",
                    side_effect=probe_during_process_cleanup,
                ) as terminate,
                self.assertRaisesRegex(
                    ReferenceAgentLaunchError,
                    "exceeded its time budget",
                ),
            ):
                program.run(_agent_context(evaluator, incumbent_ref))

            self.assertTrue(observed["start_new_session"])
            terminate.assert_called_once_with(observed["process"])
            self.assertEqual(observed["cleanup_rpc_error"][0], "invalid_request")
            self.assertIn("capability is expired", observed["cleanup_rpc_error"][1])
            self.assertEqual(evaluator.charged_case_credits, 0)

            workspace = next((root / "run" / "round-0001").iterdir())
            context_path = workspace / "context.json"
            terminal_text = context_path.read_text(encoding="utf-8")
            terminal = json.loads(terminal_text)
            self.assertNotIn(observed["token"], terminal_text)
            self.assertNotIn("token", terminal["evaluator"])
            self.assertEqual(
                terminal["evaluator"]["capability_status"],
                "expired",
            )
            self.assertEqual(
                terminal["evaluator"]["endpoint"],
                observed["endpoint"],
            )
            self.assertEqual(stat.S_IMODE(context_path.stat().st_mode), 0o400)
            self.assertEqual(
                load_reference_public_workspace(context_path).run_id,
                "timeout-cleanup",
            )
            audit_path = workspace / "agent_launch_audit.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            self.assertEqual(stat.S_IMODE(audit_path.stat().st_mode), 0o400)
            self.assertEqual(audit["status"], "active_time_timeout")
            self.assertEqual(audit["failure"], {
                "exception_type": "ReferenceAgentLaunchError",
            })
            self.assertEqual(audit["active_timeout_seconds"], 0.25)
            self.assertEqual(audit["hard_timeout_seconds"], 0.25)
            self.assertGreaterEqual(audit["wall_time_seconds"], 0.25)
            self.assertEqual(audit["controller_service_seconds"], 0.0)
            self.assertFalse(
                audit["in_process_evaluator_preemption_guaranteed"]
            )

    def test_bundle_identity_is_exact_and_source_mutation_does_not_follow(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "candidate.py"
            original = CANDIDATE_SOURCE.encode("utf-8")
            source.write_bytes(original)
            attestor = ReferenceFilesystemBundleAttestor(root / "sealed")
            record = attestor.attest(
                AgentBundleSubmission("candidate", str(source)),
                round_index=7,
                agent_id="agent-b",
                parent_sha256="a" * 64,
            )

            source.write_text(
                CANDIDATE_SOURCE.replace("(0.0,)", "(5.0,)"),
                encoding="utf-8",
            )

            self.assertEqual(
                record.bundle_sha256,
                hashlib.sha256(original).hexdigest(),
            )
            self.assertEqual(Path(record.sealed_bundle_uri).read_bytes(), original)
            self.assertEqual(record.round_index, 7)
            self.assertEqual(record.agent_id, "agent-b")
            self.assertEqual(record.parent_sha256, "a" * 64)
            program = load_python_macro_candidate(
                record.sealed_bundle_uri,
                expected_sha256=record.bundle_sha256,
            )
            prediction = program.predict_macros(_case(), QUERY)
            self.assertEqual(prediction.fields["density"], (0.0,))

    def test_multifile_bundle_loads_assets_and_stages_incumbent_and_prior(self):
        class InspectingLauncher:
            def launch(self, request):
                metadata = json.loads(
                    request.context_path.read_text(encoding="utf-8")
                )
                self.incumbent = (
                    request.workspace / metadata["incumbent"]["path"]
                )
                self.prior = (
                    request.workspace / metadata["prior_candidates"][0]["path"]
                )
                self.incumbent_files = {
                    path.relative_to(self.incumbent.parent).as_posix()
                    for path in self.incumbent.parent.rglob("*")
                    if path.is_file()
                }
                self.prior_files = {
                    path.relative_to(self.prior.parent).as_posix()
                    for path in self.prior.parent.rglob("*")
                    if path.is_file()
                }
                public = load_reference_public_workspace(request.context_path)
                self.public_prior = public.prior_candidates[0]["path"]

                final_entrypoint = _write_multifile_candidate(
                    request.workspace / "final_bundle",
                    3.0,
                )
                self.evaluation = evaluate_reference_candidate(
                    request.context_path,
                    candidate_path=final_entrypoint.parent,
                    candidate_id="multifile-final",
                    case_ids=("case-1",),
                )
                write_reference_candidate_submission(
                    request.context_path,
                    candidate_id="multifile-final",
                    bundle_path=final_entrypoint.parent,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            incumbent_entrypoint = _write_multifile_candidate(
                root / "incumbent-source",
                1.0,
            )
            clone_entrypoint = _write_multifile_candidate(
                root / "incumbent-clone",
                1.0,
                reverse_manifest_order=True,
            )
            self.assertEqual(
                python_bundle_sha256(incumbent_entrypoint),
                python_bundle_sha256(clone_entrypoint.parent),
            )
            incumbent_attestor = ReferenceFilesystemBundleAttestor(
                root / "incumbent-sealed"
            )
            incumbent_ref = candidate_ref_from_attestation(
                incumbent_attestor.attest(
                    AgentBundleSubmission(
                        "multifile-incumbent",
                        str(incumbent_entrypoint.parent),
                    ),
                    round_index=0,
                    agent_id="cold-start",
                    parent_sha256=None,
                )
            )
            self.assertTrue(Path(incumbent_ref.bundle_uri).is_dir())
            (incumbent_entrypoint.parent / "helper.py").write_text(
                "OFFSET = 99.0\n",
                encoding="utf-8",
            )

            prior_entrypoint = _write_multifile_candidate(
                root / "prior-source",
                2.0,
            )
            prior_attestor = ReferenceFilesystemBundleAttestor(
                root / "prior-sealed"
            )
            prior_ref = candidate_ref_from_attestation(
                prior_attestor.attest(
                    AgentBundleSubmission(
                        "multifile-prior",
                        str(prior_entrypoint),
                    ),
                    round_index=1,
                    agent_id="prior-agent",
                    parent_sha256=incumbent_ref.bundle_sha256,
                )
            )
            evaluator = _evaluator(case_credit_limit=1)
            launcher = InspectingLauncher()
            context = MacroAgentContext(
                round_index=2,
                agent=MacroAgentSpec("multifile-agent", "Multifile Agent"),
                prompt="Exercise multi-file candidates.",
                cases=evaluator.public_cases,
                evaluator=evaluator,
                incumbent=incumbent_ref,
                prior_handoff=None,
                shared_candidates=(prior_ref,),
            )
            runtime = ReferenceFilesystemMacroAgentProgram(
                run_root=root / "run",
                run_id="multifile-runtime",
                launcher=launcher,
            )

            outcome = runtime.run(context)

            expected_files = {
                "assets/offset.txt",
                "candidate_bundle.json",
                "helper.py",
                "submission.py",
            }
            self.assertEqual(launcher.incumbent_files, expected_files)
            self.assertEqual(launcher.prior_files, expected_files)
            self.assertEqual(launcher.public_prior, launcher.prior)
            self.assertEqual(
                stat.S_IMODE(launcher.incumbent.stat().st_mode),
                0o400,
            )
            self.assertEqual(
                stat.S_IMODE(launcher.incumbent.parent.stat().st_mode),
                0o500,
            )
            self.assertTrue(Path(outcome.candidate.ref.bundle_uri).is_dir())
            self.assertEqual(
                python_bundle_sha256(outcome.candidate.ref.bundle_uri),
                outcome.candidate.ref.bundle_sha256,
            )
            prediction = outcome.candidate.program.predict_macros(_case(), QUERY)
            self.assertEqual(prediction.fields["density"], (3.25,))
            self.assertEqual(launcher.evaluation["candidate_id"], "multifile-final")

    def test_bundle_reads_reject_symlink_and_replacement_race(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.py"
            source.write_text(CANDIDATE_SOURCE, encoding="utf-8")
            symlink = root / "symlink.py"
            symlink.symlink_to(source.name)

            with self.assertRaisesRegex(
                ReferenceAgentRuntimeError,
                "regular, non-symlink",
            ):
                python_bundle_sha256(symlink)

            victim = root / "victim.py"
            replacement = root / "replacement.py"
            victim.write_text(CANDIDATE_SOURCE, encoding="utf-8")
            replacement.write_text(
                CANDIDATE_SOURCE.replace("(0.0,)", "(2.0,)"),
                encoding="utf-8",
            )
            real_open = os.open
            swapped = False

            def swapping_open(path, flags, *args, **kwargs):
                nonlocal swapped
                if Path(path) == victim and not swapped:
                    replacement.replace(victim)
                    swapped = True
                return real_open(path, flags, *args, **kwargs)

            with patch(
                "pde_agent_verifier_solver.engine.reference_agent_runtime.os.open",
                side_effect=swapping_open,
            ):
                with self.assertRaisesRegex(
                    ReferenceAgentRuntimeError,
                    "changed before it was opened",
                ):
                    python_bundle_sha256(victim)

    def test_candidate_import_failure_is_charged_before_module_execution(self):
        class ImportFailureLauncher:
            response = None

            def launch(self, request):
                failing = request.workspace / "failing.py"
                failing.write_text(
                    "raise RuntimeError('module import failed')\n",
                    encoding="utf-8",
                )
                self.response = evaluate_reference_candidate(
                    request.context_path,
                    candidate_path=failing,
                    candidate_id="import-failure",
                    case_ids=("case-1",),
                )
                candidate = request.workspace / "candidate.py"
                candidate.write_text(CANDIDATE_SOURCE, encoding="utf-8")
                write_reference_candidate_submission(
                    request.context_path,
                    candidate_id="final-after-import-failure",
                    bundle_path=candidate,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluator = _evaluator(case_credit_limit=1)
            incumbent_ref, _ = _incumbent(root)
            launcher = ImportFailureLauncher()
            program = ReferenceFilesystemMacroAgentProgram(
                run_root=root / "run",
                run_id="charged-import-failure",
                launcher=launcher,
            )
            program.run(_agent_context(evaluator, incumbent_ref))

            self.assertEqual(evaluator.charged_case_credits, 1)
            self.assertEqual(launcher.response["charged_case_credits"], 1)
            score = launcher.response["cases"][0]["score"]
            self.assertEqual(score["status"], "invalid")
            self.assertEqual(score["failures"], ["candidate_execution_failed"])

    def test_concurrent_rpc_requests_serialize_matcher_and_credits(self):
        class ConcurrentLauncher:
            responses = ()

            def launch(self, request):
                candidate = request.workspace / "candidate.py"
                candidate.write_text(CANDIDATE_SOURCE, encoding="utf-8")
                start = threading.Barrier(2)

                def evaluate(index):
                    start.wait(timeout=2.0)
                    return evaluate_reference_candidate(
                        request.context_path,
                        candidate_path=candidate,
                        candidate_id=f"concurrent-{index}",
                        case_ids=("case-1",),
                    )

                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures = tuple(
                        pool.submit(evaluate, index) for index in range(2)
                    )
                    self.responses = tuple(future.result() for future in futures)
                write_reference_candidate_submission(
                    request.context_path,
                    candidate_id="final-after-concurrency",
                    bundle_path=candidate,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            matcher = _ConcurrentMatcher()
            evaluator = _evaluator(case_credit_limit=2, matcher=matcher)
            incumbent_ref, _ = _incumbent(root)
            launcher = ConcurrentLauncher()
            program = ReferenceFilesystemMacroAgentProgram(
                run_root=root / "run",
                run_id="concurrent-credit-test",
                launcher=launcher,
            )
            program.run(_agent_context(evaluator, incumbent_ref))

            self.assertEqual(evaluator.charged_case_credits, 2)
            self.assertEqual(
                sorted(row["charged_case_credits"] for row in launcher.responses),
                [1, 2],
            )
            self.assertEqual(
                sorted(row["remaining_case_credits"] for row in launcher.responses),
                [0, 1],
            )
            self.assertEqual(matcher.max_active, 1)

    def test_rpc_close_drains_accepted_screen_before_ledger_write(self):
        self.assertFalse(_ReferenceHTTPServer.daemon_threads)
        self.assertTrue(_ReferenceHTTPServer.block_on_close)
        started = threading.Event()
        release = threading.Event()
        close_started = threading.Event()
        original_screen = DevelopmentSession.screen
        original_server_close = _ReferenceHTTPServer.server_close

        def blocked_screen(session, *args, **kwargs):
            started.set()
            if not release.wait(timeout=5.0):
                raise RuntimeError("timed out waiting to release screen")
            return original_screen(session, *args, **kwargs)

        def observed_server_close(server):
            close_started.set()
            return original_server_close(server)

        class InFlightLauncher:
            def __init__(self):
                self.returned = threading.Event()
                self.response = None
                self.error = None
                self.worker = None
                self.workspace = None

            def launch(self, request):
                self.workspace = request.workspace
                candidate = request.workspace / "candidate.py"
                candidate.write_text(CANDIDATE_SOURCE, encoding="utf-8")

                def screen():
                    try:
                        self.response = screen_reference_candidate(
                            request.context_path,
                            candidate_path=candidate,
                            candidate_id="accepted-in-flight",
                            requested_root_count=4,
                        )
                    except BaseException as exc:
                        self.error = exc

                self.worker = threading.Thread(target=screen, daemon=True)
                self.worker.start()
                if not started.wait(timeout=2.0):
                    raise RuntimeError("screen handler was not accepted")
                self.returned.set()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            incumbent_ref, _ = _incumbent(root)
            session, _ = _strong_session(
                incumbent_ref,
                screen_credits=4,
                large_release_size=1,
            )
            launcher = InFlightLauncher()
            program = ReferenceFilesystemMacroAgentProgram(
                run_root=root / "run",
                run_id="in-flight-drain",
                launcher=launcher,
                screen_timeout_seconds=10.0,
            )
            context = _agent_context(session, incumbent_ref)
            with (
                patch.object(DevelopmentSession, "screen", blocked_screen),
                patch.object(
                    _ReferenceHTTPServer,
                    "server_close",
                    observed_server_close,
                ),
                ThreadPoolExecutor(max_workers=1) as pool,
            ):
                future = pool.submit(program.run, context)
                self.assertTrue(launcher.returned.wait(timeout=2.0))
                self.assertTrue(close_started.wait(timeout=2.0))
                assert launcher.workspace is not None
                ledger_path = launcher.workspace / "development_ledger.json"
                try:
                    with self.assertRaises(TimeoutError):
                        future.result(timeout=0.1)
                    self.assertFalse(ledger_path.exists())
                finally:
                    release.set()
                with self.assertRaises(ReferenceAgentRuntimeError):
                    future.result(timeout=5.0)

            assert launcher.worker is not None
            launcher.worker.join(timeout=2.0)
            self.assertFalse(launcher.worker.is_alive())
            self.assertIsNone(launcher.error)
            self.assertIsNotNone(launcher.response)
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            self.assertEqual(len(ledger["screen_ledger"]), 1)
            self.assertEqual(
                ledger["screen_ledger"][0]["charged_root_credits"],
                4,
            )

    def test_strong_screen_completed_answers_cache_stages_and_ledger(self):
        class StrongLauncher:
            def launch(self, request):
                raw_context = request.context_path.read_bytes()
                metadata = json.loads(raw_context.decode("utf-8"))
                self.context_size = len(raw_context)
                self.evaluator_metadata = metadata["evaluator"]
                self.answer_node = metadata["public_tasks"][0][
                    "completed_answer"
                ]["fields"]["density"]
                public = load_reference_public_workspace(
                    request.context_path,
                    mmap_arrays=True,
                )
                values = public.completed_answers[0].fields["density"]
                self.answer_summary = (
                    len(values),
                    values[0],
                    values[-1],
                )
                self.public_credits = (
                    public.remaining_case_credits,
                    public.remaining_root_credits,
                )
                self.public_timeout = public.screen_timeout_seconds

                candidate = request.workspace / "candidate.py"
                candidate.write_text(CANDIDATE_SOURCE, encoding="utf-8")
                with patch(
                    "pde_agent_verifier_solver.engine.reference_agent_runtime."
                    "_call_reference_rpc",
                    return_value={},
                ) as rpc:
                    screen_reference_candidate(
                        request.context_path,
                        candidate_path=candidate,
                        candidate_id="timeout-probe",
                        requested_root_count=4,
                    )
                self.default_screen_timeout = (
                    rpc.call_args.kwargs["timeout_seconds"]
                )
                with patch(
                    "pde_agent_verifier_solver.engine.reference_agent_runtime."
                    "_call_reference_rpc",
                    return_value={},
                ) as rpc:
                    screen_reference_candidate(
                        request.context_path,
                        candidate_path=candidate,
                        candidate_id="explicit-timeout-probe",
                        requested_root_count=4,
                        timeout_seconds=45.0,
                    )
                self.explicit_screen_timeout = (
                    rpc.call_args.kwargs["timeout_seconds"]
                )

                legacy_metadata = json.loads(raw_context.decode("utf-8"))
                legacy_metadata["evaluator"].pop("screen_timeout_seconds")
                legacy_context = request.workspace / "legacy-context.json"
                legacy_context.write_text(
                    json.dumps(legacy_metadata),
                    encoding="utf-8",
                )
                with patch(
                    "pde_agent_verifier_solver.engine.reference_agent_runtime."
                    "_call_reference_rpc",
                    return_value={},
                ) as rpc:
                    screen_reference_candidate(
                        legacy_context,
                        candidate_path=candidate,
                        candidate_id="legacy-timeout-probe",
                        requested_root_count=4,
                    )
                self.legacy_screen_timeout = (
                    rpc.call_args.kwargs["timeout_seconds"]
                )

                try:
                    screen_reference_candidate(
                        request.context_path,
                        candidate_path=candidate,
                        candidate_id="shared-label",
                        requested_root_count=16,
                    )
                except ReferenceEvaluatorClientError as exc:
                    self.stage_error = (exc.code, exc.http_status)
                else:
                    raise AssertionError("direct deep screen unexpectedly passed")

                self.first = screen_reference_candidate(
                    request.context_path,
                    candidate_path=candidate,
                    candidate_id="shared-label",
                    requested_root_count=4,
                )
                self.cached = screen_reference_candidate(
                    request.context_path,
                    candidate_path=candidate,
                    candidate_id="different-label",
                    requested_root_count=4,
                )
                self.deeper = screen_reference_candidate(
                    request.context_path,
                    candidate_path=candidate,
                    candidate_id="shared-label",
                    requested_root_count=16,
                )
                other = request.workspace / "other.py"
                other.write_text(
                    CANDIDATE_SOURCE.replace("(0.0,)", "(2.0,)"),
                    encoding="utf-8",
                )
                try:
                    screen_reference_candidate(
                        request.context_path,
                        candidate_path=other,
                        candidate_id="other-label",
                        requested_root_count=4,
                    )
                except ReferenceEvaluatorClientError as exc:
                    self.credit_error = (exc.code, exc.http_status)
                else:
                    raise AssertionError("over-budget screen unexpectedly passed")

                write_reference_candidate_submission(
                    request.context_path,
                    candidate_id="shared-label",
                    bundle_path=candidate,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            incumbent_ref, _ = _incumbent(root)
            session, factory = _strong_session(incumbent_ref)
            launcher = StrongLauncher()
            program = ReferenceFilesystemMacroAgentProgram(
                run_root=root / "run",
                run_id="strong-screen-runtime",
                launcher=launcher,
                screen_timeout_seconds=1234.5,
            )
            context = MacroAgentContext(
                round_index=2,
                agent=MacroAgentSpec("strong-agent", "Strong Agent"),
                prompt="Use completed examples and fresh private screens.",
                cases=session.public_cases,
                evaluator=session,
                incumbent=incumbent_ref,
                prior_handoff=None,
            )

            outcome = program.run(context)

            self.assertLess(launcher.context_size, 8 * 1024 * 1024)
            self.assertEqual(launcher.answer_node["type"], "ndarray")
            self.assertEqual(launcher.answer_summary[0], 200_000)
            self.assertEqual(launcher.answer_summary[1], SECRET_TARGET)
            self.assertAlmostEqual(
                launcher.answer_summary[2],
                SECRET_TARGET + 199_999 / 7.0,
            )
            self.assertEqual(
                launcher.evaluator_metadata["credit_scope"],
                "per_agent_per_round",
            )
            self.assertEqual(
                launcher.evaluator_metadata["statistical_unit"],
                "independent_root",
            )
            self.assertNotIn(
                "remaining_case_credits",
                launcher.evaluator_metadata,
            )
            self.assertEqual(
                launcher.evaluator_metadata["remaining_root_credits"],
                16,
            )
            self.assertEqual(launcher.public_credits, (None, 16))
            self.assertEqual(
                launcher.evaluator_metadata["screen_timeout_seconds"],
                1234.5,
            )
            self.assertEqual(launcher.public_timeout, 1234.5)
            self.assertEqual(launcher.default_screen_timeout, 1234.5)
            self.assertEqual(launcher.explicit_screen_timeout, 45.0)
            self.assertEqual(launcher.legacy_screen_timeout, 300.0)
            self.assertEqual(
                launcher.evaluator_metadata["cli"][3],
                "screen",
            )
            self.assertEqual(
                launcher.stage_error,
                ("screen_rejected", 409),
            )
            self.assertEqual(
                launcher.credit_error,
                ("root_credit_exceeded", 409),
            )
            self.assertFalse(launcher.first["accounting"]["cached"])
            self.assertEqual(
                launcher.first["accounting"]["charged_root_credits"],
                4,
            )
            self.assertTrue(launcher.cached["accounting"]["cached"])
            self.assertEqual(
                launcher.cached["accounting"]["charged_root_credits"],
                0,
            )
            self.assertEqual(
                launcher.first["candidate_sha256"],
                launcher.cached["candidate_sha256"],
            )
            self.assertEqual(launcher.deeper["completed_root_count"], 16)
            self.assertEqual(
                launcher.deeper["accounting"]["charged_root_credits"],
                12,
            )
            for response in (
                launcher.first,
                launcher.cached,
                launcher.deeper,
            ):
                self.assertNotIn("cases", response)
                self.assertNotIn("case_ids", response)
                self.assertNotIn("target_sha256", response)
                self.assertFalse(response["finite_truth_included"])
                self.assertFalse(response["per_case_errors_included"])
                serialized = json.dumps(response, sort_keys=True)
                for root_id in factory.root_ids:
                    if root_id.startswith("screen-"):
                        self.assertNotIn(root_id, serialized)

            self.assertEqual(session.release_charged_root_credits, 1)
            self.assertIsNone(session.latest_release)
            self.assertEqual(session.screen_charged_root_credits, 16)
            self.assertEqual(
                [request.root_count for request in factory.requests],
                [1, 4, 12],
            )
            self.assertEqual(
                outcome.candidate.ref.candidate_id,
                "strong-agent:r2:shared-label",
            )
            self.assertEqual(
                outcome.candidate.ref.bundle_sha256,
                launcher.first["candidate_sha256"],
            )
            self.assertEqual(
                session.screened_candidates[0].ref.candidate_id,
                outcome.candidate.ref.candidate_id,
            )

            workspace = next((root / "run" / "round-0002").iterdir())
            array_path = workspace / launcher.answer_node["path"]
            self.assertTrue(array_path.is_file())
            self.assertEqual(stat.S_IMODE(array_path.stat().st_mode), 0o400)
            ledger_path = workspace / "development_ledger.json"
            self.assertEqual(stat.S_IMODE(ledger_path.stat().st_mode), 0o400)
            ledger_text = ledger_path.read_text(encoding="utf-8")
            ledger = json.loads(ledger_text)
            self.assertEqual(
                ledger["schema"],
                "reference-macro-development-ledger-v1",
            )
            self.assertEqual(
                ledger["credit_scope"],
                "per_agent_per_round",
            )
            self.assertEqual(
                ledger["statistical_unit"],
                "independent_root",
            )
            self.assertEqual(
                ledger["root_credits"],
                {
                    "release_charged": 1,
                    "release_remaining": 0,
                    "screen_charged": 16,
                    "screen_remaining": 0,
                },
            )
            self.assertEqual(len(ledger["release_ledger"]), 1)
            self.assertEqual(
                [row["new_root_count"] for row in ledger["screen_ledger"]],
                [4, 0, 12],
            )
            self.assertEqual(
                [row["cached"] for row in ledger["screen_ledger"]],
                [False, True, False],
            )
            self.assertIsNone(ledger["screen_ledger"][1]["draw"])
            self.assertEqual(
                ledger["screen_ledger"][0]["candidate_sha256"],
                launcher.first["candidate_sha256"],
            )
            self.assertNotIn("completed_answer", ledger_text)
            self.assertNotIn("fields", ledger_text)
            for root_id in factory.root_ids:
                self.assertNotIn(root_id, ledger_text)

    def test_screens_are_feedback_and_unscreened_nomination_is_valid(self):
        test_case = self

        class OpenSearchLauncher:
            def launch(self, request):
                self.metadata = json.loads(
                    request.context_path.read_text(encoding="utf-8")
                )
                self.public = load_reference_public_workspace(
                    request.context_path
                )
                candidate = request.workspace / "candidate.py"
                candidate.write_text(CANDIDATE_SOURCE, encoding="utf-8")
                other = request.workspace / "other.py"
                other.write_text(
                    CANDIDATE_SOURCE.replace("(0.0,)", "(2.0,)"),
                    encoding="utf-8",
                )
                nomination = request.workspace / "nomination.py"
                nomination.write_text(
                    CANDIDATE_SOURCE.replace("(0.0,)", "(3.0,)"),
                    encoding="utf-8",
                )
                self.screens = (
                    screen_reference_candidate(
                        request.context_path,
                        candidate_path=candidate,
                        candidate_id="development-one",
                        requested_root_count=4,
                    ),
                    screen_reference_candidate(
                        request.context_path,
                        candidate_path=other,
                        candidate_id="development-two",
                        requested_root_count=4,
                    ),
                )
                write_reference_candidate_submission(
                    request.context_path,
                    candidate_id="official-nomination",
                    bundle_path=nomination,
                )
                self.submission = json.loads(
                    request.submission_path.read_text(encoding="utf-8")
                )
                with test_case.assertRaises(FileExistsError):
                    write_reference_candidate_submission(
                        request.context_path,
                        candidate_id="second-official-nomination",
                        bundle_path=other,
                    )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            incumbent_ref, _ = _incumbent(root)
            session, _ = _strong_session(
                incumbent_ref,
                screen_credits=8,
                large_release_size=1,
            )
            launcher = OpenSearchLauncher()
            policy = CompetitiveSearchPolicy(
                min_distinct_screened_candidates=99,
                min_full_stage_candidates=99,
                min_released_roots=99,
                require_final_full_stage=True,
            )
            program = ReferenceFilesystemMacroAgentProgram(
                run_root=root / "run",
                run_id="open-search-nomination-runtime",
                launcher=launcher,
            )
            outcome = program.run(
                MacroAgentContext(
                    round_index=2,
                    agent=MacroAgentSpec("open-agent", "Open Agent"),
                    prompt="Use screens as optional development feedback.",
                    cases=session.public_cases,
                    evaluator=session,
                    incumbent=incumbent_ref,
                    prior_handoff=None,
                    search_policy=policy,
                )
            )

            self.assertIsNone(launcher.public.finish_endpoint)
            self.assertNotIn("finish_endpoint", launcher.metadata["evaluator"])
            self.assertNotIn(
                "search_finish_gate",
                launcher.metadata["evaluator"]["capabilities"],
            )
            for removed in (
                "finish_gate_available",
                "min_distinct_screened_candidates",
                "min_full_stage_candidates",
                "min_released_roots",
                "require_final_full_stage",
            ):
                self.assertNotIn(removed, launcher.public.search_contract)
            self.assertEqual(
                launcher.public.search_contract["official_nomination_limit"],
                1,
            )
            self.assertFalse(
                launcher.public.search_contract[
                    "development_screens_are_official_nominations"
                ]
            )
            self.assertEqual(len(session.screened_candidates), 2)
            screened_digests = {
                candidate.ref.bundle_sha256
                for candidate in session.screened_candidates
            }
            self.assertNotIn(
                outcome.candidate.ref.bundle_sha256,
                screened_digests,
            )
            self.assertEqual(
                launcher.submission["candidate_id"],
                "official-nomination",
            )

    def test_legacy_finish_metadata_loads_but_does_not_gate_nomination(self):
        class LegacyFinishContextLauncher:
            def launch(self, request):
                metadata = json.loads(
                    request.context_path.read_text(encoding="utf-8")
                )
                metadata["evaluator"] = dict(metadata["evaluator"])
                metadata["evaluator"]["finish_endpoint"] = (
                    "http://127.0.0.1:1/finish"
                )
                metadata["search_contract"] = {
                    "finish_gate_available": True,
                    "min_distinct_screened_candidates": 99,
                    "min_full_stage_candidates": 99,
                    "min_released_roots": 99,
                    "require_final_full_stage": True,
                    "leaderboard_limit": 8,
                    "cross_share_after_round": 1,
                    "counts_are_aggregate_only": True,
                }
                legacy_context = request.workspace / "legacy-finish-context.json"
                legacy_context.write_text(
                    json.dumps(metadata),
                    encoding="utf-8",
                )
                self.public = load_reference_public_workspace(legacy_context)
                candidate = request.workspace / "candidate.py"
                candidate.write_text(CANDIDATE_SOURCE, encoding="utf-8")
                write_reference_candidate_submission(
                    legacy_context,
                    candidate_id="legacy-unscreened",
                    bundle_path=candidate,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            incumbent_ref, _ = _incumbent(root)
            session, _ = _strong_session(
                incumbent_ref,
                screen_credits=4,
                large_release_size=1,
            )
            launcher = LegacyFinishContextLauncher()
            program = ReferenceFilesystemMacroAgentProgram(
                run_root=root / "run",
                run_id="legacy-finish-context-runtime",
                launcher=launcher,
            )
            context = MacroAgentContext(
                round_index=2,
                agent=MacroAgentSpec("legacy-agent", "Legacy Agent"),
                prompt="Read a historical context but nominate without a finish RPC.",
                cases=session.public_cases,
                evaluator=session,
                incumbent=incumbent_ref,
                prior_handoff=None,
            )

            outcome = program.run(context)

            self.assertEqual(
                launcher.public.finish_endpoint,
                "http://127.0.0.1:1/finish",
            )
            self.assertTrue(
                launcher.public.search_contract["finish_gate_available"]
            )
            self.assertEqual(len(session.screened_candidates), 0)
            self.assertEqual(
                outcome.candidate.ref.candidate_id,
                "legacy-agent:r2:legacy-unscreened",
            )

    def test_prior_candidate_staging_is_verified_read_only_and_sanitized(self):
        class InspectingLauncher:
            def launch(self, request):
                metadata = json.loads(
                    request.context_path.read_text(encoding="utf-8")
                )
                self.prior_candidates = metadata["prior_candidates"]
                self.prior_leaderboard = metadata["prior_leaderboard"]
                self.search_contract = metadata["search_contract"]
                self.public = load_reference_public_workspace(
                    request.context_path
                )
                legacy_metadata = dict(metadata)
                legacy_metadata.pop("search_contract")
                legacy_metadata.pop("prior_leaderboard")
                legacy_metadata.pop("prior_candidates")
                legacy_context = request.workspace / "legacy-search-context.json"
                legacy_context.write_text(
                    json.dumps(legacy_metadata),
                    encoding="utf-8",
                )
                self.legacy_public = load_reference_public_workspace(
                    legacy_context
                )
                staged = request.workspace / self.prior_candidates[0]["path"]
                self.staged_payload = staged.read_bytes()
                self.staged_mode = stat.S_IMODE(staged.stat().st_mode)
                candidate = request.workspace / "candidate.py"
                candidate.write_text(CANDIDATE_SOURCE, encoding="utf-8")
                write_reference_candidate_submission(
                    request.context_path,
                    candidate_id="fresh-candidate",
                    bundle_path=candidate,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            incumbent_ref, _ = _incumbent(root)
            prior_source = root / "prior.py"
            prior_payload = CANDIDATE_SOURCE.replace(
                "(0.0,)", "(3.0,)"
            ).encode("utf-8")
            prior_source.write_bytes(prior_payload)
            prior_attestor = ReferenceFilesystemBundleAttestor(
                root / "prior-sealed"
            )
            prior_ref = candidate_ref_from_attestation(
                prior_attestor.attest(
                    AgentBundleSubmission("prior-candidate", str(prior_source)),
                    round_index=1,
                    agent_id="prior-agent",
                    parent_sha256=incumbent_ref.bundle_sha256,
                )
            )
            leaderboard = SearchLeaderboardEntry(
                rank=1,
                candidate=prior_ref,
                agent_id="prior-agent",
                eligible=True,
                mean=0.2,
                cvar=0.3,
                maximum=0.4,
                generated_mean=0.15,
                perturbed_mean=0.25,
                guardrail_passed=True,
                roles=("screen_generated_mean",),
            )
            launcher = InspectingLauncher()
            evaluator = _evaluator(case_credit_limit=1)
            context = MacroAgentContext(
                round_index=2,
                agent=MacroAgentSpec("fresh-agent", "Fresh Agent"),
                prompt="Inspect aggregate prior candidates.",
                cases=evaluator.public_cases,
                evaluator=evaluator,
                incumbent=incumbent_ref,
                prior_handoff=None,
                prior_leaderboard=(leaderboard,),
                shared_candidates=(prior_ref,),
            )
            program = ReferenceFilesystemMacroAgentProgram(
                run_root=root / "run",
                run_id="prior-candidate-runtime",
                launcher=launcher,
            )

            program.run(context)

            self.assertEqual(launcher.staged_payload, prior_payload)
            self.assertEqual(launcher.staged_mode, 0o400)
            self.assertEqual(
                launcher.prior_candidates[0]["bundle_sha256"],
                prior_ref.bundle_sha256,
            )
            public_prior = json.dumps(
                {
                    "prior_candidates": launcher.prior_candidates,
                    "prior_leaderboard": launcher.prior_leaderboard,
                },
                sort_keys=True,
            )
            self.assertNotIn("bundle_uri", public_prior)
            self.assertNotIn("case_id", public_prior)
            self.assertNotIn("completed_answer", public_prior)
            self.assertTrue(
                launcher.search_contract["counts_are_aggregate_only"]
            )
            self.assertEqual(
                launcher.public.search_contract["leaderboard_limit"],
                context.search_policy.leaderboard_limit,
            )
            self.assertEqual(
                launcher.public.prior_leaderboard[0]["bundle_sha256"],
                prior_ref.bundle_sha256,
            )
            self.assertEqual(
                launcher.public.prior_candidates[0]["path"].read_bytes(),
                prior_payload,
            )
            self.assertIsNone(launcher.public.finish_endpoint)
            self.assertEqual(dict(launcher.legacy_public.search_contract), {})
            self.assertEqual(launcher.legacy_public.prior_leaderboard, ())
            self.assertEqual(launcher.legacy_public.prior_candidates, ())
            self.assertIsNone(launcher.legacy_public.finish_endpoint)

            mismatched_ref = CandidateRef(
                candidate_id=prior_ref.candidate_id,
                bundle_sha256="f" * 64,
                round_index=prior_ref.round_index,
                agent_id=prior_ref.agent_id,
                parent_sha256=prior_ref.parent_sha256,
                bundle_uri=prior_ref.bundle_uri,
            )
            bad_context = MacroAgentContext(
                round_index=3,
                agent=MacroAgentSpec("bad-agent", "Bad Agent"),
                prompt="Reject mismatched shared candidate bytes.",
                cases=evaluator.public_cases,
                evaluator=evaluator,
                incumbent=incumbent_ref,
                prior_handoff=None,
                shared_candidates=(mismatched_ref,),
            )
            bad_program = ReferenceFilesystemMacroAgentProgram(
                run_root=root / "bad-run",
                run_id="bad-prior-candidate-runtime",
                launcher=InspectingLauncher(),
            )
            bad_program.run(bad_context)
            self.assertEqual(bad_program.launcher.staged_payload, prior_payload)

    def test_candidate_ids_are_namespaced_across_agents_for_screen_and_final(self):
        class ScreenAndSubmitLauncher:
            def __init__(self, source):
                self.source = source
                self.response = None

            def launch(self, request):
                candidate = request.workspace / "candidate.py"
                candidate.write_text(self.source, encoding="utf-8")
                self.response = screen_reference_candidate(
                    request.context_path,
                    candidate_path=candidate,
                    candidate_id="gkw_candidate_v1",
                    requested_root_count=4,
                )
                write_reference_candidate_submission(
                    request.context_path,
                    candidate_id="gkw_candidate_v1",
                    bundle_path=candidate,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            incumbent_ref, _ = _incumbent(root)
            outcomes = []
            responses = []
            sessions = []
            for agent_id, offset in (
                ("agent-left", 0.0),
                ("agent-right", 1.0),
            ):
                session, _ = _strong_session(
                    incumbent_ref,
                    screen_credits=4,
                    large_release_size=1,
                )
                source = CANDIDATE_SOURCE.replace(
                    "(0.0,)",
                    f"({SECRET_TARGET + offset!r},)",
                )
                launcher = ScreenAndSubmitLauncher(source)
                program = ReferenceFilesystemMacroAgentProgram(
                    run_root=root / "run",
                    run_id="candidate-namespace-runtime",
                    launcher=launcher,
                )
                outcome = program.run(
                    MacroAgentContext(
                        round_index=3,
                        agent=MacroAgentSpec(agent_id, agent_id),
                        prompt="Submit one identically labelled candidate.",
                        cases=session.public_cases,
                        evaluator=session,
                        incumbent=incumbent_ref,
                        prior_handoff=None,
                    )
                )
                outcomes.append(outcome)
                responses.append(launcher.response)
                sessions.append(session)

            self.assertEqual(
                [outcome.candidate.ref.candidate_id for outcome in outcomes],
                [
                    "agent-left:r3:gkw_candidate_v1",
                    "agent-right:r3:gkw_candidate_v1",
                ],
            )
            self.assertNotEqual(
                outcomes[0].candidate.ref.bundle_sha256,
                outcomes[1].candidate.ref.bundle_sha256,
            )
            for outcome, response, session in zip(
                outcomes,
                responses,
                sessions,
            ):
                self.assertTrue(response["candidate"]["eligible"])
                self.assertEqual(
                    response["candidate_sha256"],
                    outcome.candidate.ref.bundle_sha256,
                )
                self.assertEqual(
                    session.screened_candidates[0].ref.candidate_id,
                    outcome.candidate.ref.candidate_id,
                )

    def test_official_nomination_need_not_match_a_screened_digest(self):
        class MutatingLauncher:
            def launch(self, request):
                candidate = request.workspace / "candidate.py"
                candidate.write_text(CANDIDATE_SOURCE, encoding="utf-8")
                self.screened = screen_reference_candidate(
                    request.context_path,
                    candidate_path=candidate,
                    candidate_id="screen-then-mutate",
                    requested_root_count=4,
                )
                candidate.write_text(
                    CANDIDATE_SOURCE.replace("(0.0,)", "(2.0,)"),
                    encoding="utf-8",
                )
                write_reference_candidate_submission(
                    request.context_path,
                    candidate_id="screen-then-mutate",
                    bundle_path=candidate,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            incumbent_ref, _ = _incumbent(root)
            session, _ = _strong_session(
                incumbent_ref,
                screen_credits=4,
                large_release_size=1,
            )
            launcher = MutatingLauncher()
            program = ReferenceFilesystemMacroAgentProgram(
                run_root=root / "run",
                run_id="screened-final-digest-runtime",
                launcher=launcher,
            )
            context = MacroAgentContext(
                round_index=4,
                agent=MacroAgentSpec("mutating-agent", "Mutating Agent"),
                prompt="Screen before final submission.",
                cases=session.public_cases,
                evaluator=session,
                incumbent=incumbent_ref,
                prior_handoff=None,
            )

            outcome = program.run(context)

            self.assertEqual(len(session.screened_candidates), 1)
            self.assertEqual(
                launcher.screened["candidate_sha256"],
                session.screened_candidates[0].ref.bundle_sha256,
            )
            self.assertNotEqual(
                outcome.candidate.ref.bundle_sha256,
                launcher.screened["candidate_sha256"],
            )
            prediction = outcome.candidate.program.predict_macros(
                _case(),
                QUERY,
            )
            self.assertEqual(prediction.fields["density"], (2.0,))

    def test_private_screen_backend_errors_do_not_leak_case_identity(self):
        secret = "PRIVATE-screen-root-case-secret-8842"

        class FailingScreenLauncher:
            def launch(self, request):
                candidate = request.workspace / "candidate.py"
                candidate.write_text(CANDIDATE_SOURCE, encoding="utf-8")
                try:
                    screen_reference_candidate(
                        request.context_path,
                        candidate_path=candidate,
                        candidate_id="private-backend-failure",
                        requested_root_count=4,
                    )
                except ReferenceEvaluatorClientError as exc:
                    self.error = exc
                    raise RuntimeError("stop after sanitized screen") from exc
                raise AssertionError("private screen unexpectedly passed")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            incumbent_ref, _ = _incumbent(root)
            session, _ = _strong_session(
                incumbent_ref,
                screen_credits=4,
                large_release_size=1,
                screen_error=secret,
            )
            launcher = FailingScreenLauncher()
            program = ReferenceFilesystemMacroAgentProgram(
                run_root=root / "run",
                run_id="private-screen-error-runtime",
                launcher=launcher,
            )
            context = MacroAgentContext(
                round_index=5,
                agent=MacroAgentSpec("error-agent", "Error Agent"),
                prompt="Screen without exposing private identities.",
                cases=session.public_cases,
                evaluator=session,
                incumbent=incumbent_ref,
                prior_handoff=None,
            )

            with self.assertRaisesRegex(
                RuntimeError,
                "stop after sanitized screen",
            ) as stopped:
                program.run(context)

            self.assertEqual(launcher.error.code, "screen_failed")
            self.assertEqual(
                launcher.error.message,
                "screen_failed:ValueError",
            )
            self.assertEqual(launcher.error.http_status, 500)
            self.assertNotIn(secret, str(launcher.error))
            self.assertNotIn(secret, str(stopped.exception))

            workspace = next((root / "run" / "round-0005").iterdir())
            ledger_text = (workspace / "development_ledger.json").read_text(
                encoding="utf-8"
            )
            self.assertNotIn(secret, ledger_text)
            self.assertEqual(
                json.loads(ledger_text)["screen_ledger"],
                [],
            )

    def test_submission_json_and_observation_limits_fail_closed(self):
        class OversizedObservationLauncher:
            def launch(self, request):
                candidate = request.workspace / "candidate.py"
                candidate.write_text(CANDIDATE_SOURCE, encoding="utf-8")
                write_reference_candidate_submission(
                    request.context_path,
                    candidate_id="oversized-observation",
                    bundle_path=candidate,
                    observations=("x" * 4097,),
                )

        class OversizedJSONLauncher:
            def launch(self, request):
                request.submission_path.write_bytes(b"{" + b" " * 65536)

        cases = (
            (OversizedObservationLauncher(), "observation is invalid or too long"),
            (OversizedJSONLauncher(), "exceeds 65536 bytes"),
        )
        for launcher, message in cases:
            with (
                self.subTest(message=message),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                evaluator = _evaluator(case_credit_limit=1)
                incumbent_ref, _ = _incumbent(root)
                program = ReferenceFilesystemMacroAgentProgram(
                    run_root=root / "run",
                    run_id="bounded-submission-test",
                    launcher=launcher,
                )
                with self.assertRaisesRegex(ReferenceAgentRuntimeError, message):
                    program.run(_agent_context(evaluator, incumbent_ref))

    def test_submission_cannot_override_controller_parent_or_round(self):
        class SpoofingLauncher:
            def launch(self, request):
                candidate = request.workspace / "candidate.py"
                candidate.write_text(CANDIDATE_SOURCE, encoding="utf-8")
                request.submission_path.write_text(
                    json.dumps(
                        {
                            "schema": "reference-macro-agent-submission-v1",
                            "candidate_id": "spoof",
                            "bundle_path": "candidate.py",
                            "observations": [],
                            "round_index": 99,
                            "parent_sha256": "f" * 64,
                        }
                    ),
                    encoding="utf-8",
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluator = _evaluator(case_credit_limit=1)
            incumbent_ref, _ = _incumbent(root)
            program = ReferenceFilesystemMacroAgentProgram(
                run_root=root / "run",
                run_id="identity-spoof-test",
                launcher=SpoofingLauncher(),
            )
            context = MacroAgentContext(
                round_index=2,
                agent=MacroAgentSpec("agent-c", "Agent C"),
                prompt="test",
                cases=evaluator.public_cases,
                evaluator=evaluator,
                incumbent=incumbent_ref,
                prior_handoff=None,
            )
            with self.assertRaisesRegex(
                ReferenceAgentRuntimeError,
                "one-candidate schema",
            ):
                program.run(context)


    def test_on_demand_release_and_async_submit_poll_are_workspace_scoped(self):
        class OracleLauncher:
            def launch(self, request):
                context_mode = stat.S_IMODE(request.context_path.stat().st_mode)
                context_bytes = request.context_path.read_bytes()
                metadata = json.loads(context_bytes.decode("utf-8"))
                evaluator = metadata["evaluator"]
                self.capabilities = evaluator["capabilities"]
                self.public_workspace = load_reference_public_workspace(
                    request.context_path,
                    mmap_arrays=False,
                )
                self.release_response = release_reference_completed_answers(
                    request.context_path,
                    root_count=2,
                )
                manifest_relative = self.release_response["manifest_path"]
                self.assert_relative = not Path(manifest_relative).is_absolute()
                manifest = request.workspace / manifest_relative
                self.release = load_reference_completed_answer_release(
                    manifest,
                    mmap_arrays=True,
                )
                self.manifest_text = manifest.read_text(encoding="utf-8")
                self.manifest_mode = stat.S_IMODE(manifest.stat().st_mode)
                self.array_modes = {
                    stat.S_IMODE(path.stat().st_mode)
                    for path in (manifest.parent / "arrays").glob("*.npy")
                }
                try:
                    release_reference_completed_answers(
                        request.context_path,
                        root_count=2,
                    )
                except ReferenceEvaluatorClientError as exc:
                    self.release_credit_error = (
                        exc.code,
                        exc.http_status,
                        exc.message,
                    )
                else:
                    raise AssertionError("over-budget release was accepted")
                try:
                    release_reference_completed_answers(
                        request.context_path,
                        root_count=33,
                    )
                except ValueError as exc:
                    self.release_bound_error = str(exc)
                else:
                    raise AssertionError("oversized release was accepted")

                candidate = request.workspace / "candidate.py"
                original = CANDIDATE_SOURCE.encode("utf-8")
                candidate.write_bytes(original)
                submitted = submit_reference_candidate_screen(
                    request.context_path,
                    candidate_path=candidate,
                    candidate_id="sealed-before-enqueue",
                    requested_root_count=4,
                )
                self.submitted = submitted
                candidate.write_bytes(
                    CANDIDATE_SOURCE.replace("(0.0,)", "(7.0,)").encode(
                        "utf-8"
                    )
                )
                for _ in range(200):
                    polled = poll_reference_candidate_screen(
                        request.context_path,
                        task_id=submitted["task_id"],
                    )
                    if polled["status"] in {"completed", "failed"}:
                        break
                    time.sleep(0.01)
                else:
                    raise AssertionError("async screen did not complete")
                self.polled = polled
                candidate.write_bytes(original)
                write_reference_candidate_submission(
                    request.context_path,
                    candidate_id="sealed-before-enqueue",
                    bundle_path=candidate,
                )
                self.context_mode_after = stat.S_IMODE(
                    request.context_path.stat().st_mode
                )
                self.context_mode_before = context_mode
                self.context_unchanged = (
                    request.context_path.read_bytes() == context_bytes
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            incumbent_ref, _ = _incumbent(root)
            session, factory = _strong_session(
                incumbent_ref,
                screen_credits=4,
                release_credits=4,
                initial_release_count=1,
                large_release_size=3,
            )
            launcher = OracleLauncher()
            program = ReferenceFilesystemMacroAgentProgram(
                run_root=root / "run",
                run_id="on-demand-oracle",
                launcher=launcher,
            )

            outcome = program.run(_agent_context(session, incumbent_ref))

            self.assertTrue(
                launcher.capabilities["on_demand_completed_answer_release"]
            )
            self.assertTrue(launcher.capabilities["async_screen_submit_poll"])
            self.assertEqual(
                launcher.public_workspace.remaining_release_root_credits,
                3,
            )
            self.assertEqual(
                launcher.public_workspace.max_release_roots_per_request,
                32,
            )
            self.assertEqual(
                launcher.public_workspace.max_async_screen_inflight,
                2,
            )
            self.assertTrue(launcher.assert_relative)
            self.assertEqual(launcher.release.root_count, 2)
            self.assertEqual(launcher.release.member_case_count, 2)
            self.assertEqual(len(launcher.release.answers), 2)
            self.assertEqual(
                tuple(launcher.release.answers[0].fields["density"]),
                tuple(
                    SECRET_TARGET + index / 7.0
                    for index in range(3)
                ),
            )
            self.assertFalse(
                json.loads(launcher.manifest_text)["private_seeds_included"]
            )
            self.assertEqual(launcher.manifest_mode, 0o400)
            self.assertEqual(launcher.array_modes, {0o400})
            self.assertEqual(
                launcher.release_response["accounting"][
                    "remaining_root_credits"
                ],
                1,
            )
            self.assertEqual(
                launcher.release_credit_error,
                (
                    "release_root_credit_exceeded",
                    409,
                    "completed-answer release root credit limit exceeded",
                ),
            )
            self.assertIn(
                "release root_count must be in [1, 32]",
                launcher.release_bound_error,
            )
            self.assertEqual(session.release_charged_root_credits, 3)
            self.assertIsNone(session.latest_release)
            self.assertEqual(session.screen_charged_root_credits, 4)
            self.assertEqual(launcher.polled["status"], "completed")
            self.assertEqual(
                launcher.submitted["candidate_sha256"],
                hashlib.sha256(CANDIDATE_SOURCE.encode("utf-8")).hexdigest(),
            )
            self.assertEqual(
                outcome.candidate.ref.bundle_sha256,
                launcher.submitted["candidate_sha256"],
            )
            self.assertEqual(
                launcher.context_mode_before,
                launcher.context_mode_after,
            )
            self.assertTrue(launcher.context_unchanged)
            self.assertEqual(launcher.context_mode_after, 0o400)
            self.assertEqual(
                [request.kind.value for request in factory.requests],
                ["release", "release", "screen"],
            )

    def test_async_screen_is_bounded_and_revoke_drains_accepted_tasks(self):
        started = threading.Event()
        release_worker = threading.Event()
        original_screen = DevelopmentSession.screen

        def blocked_screen(session, candidate, *, requested_root_count):
            started.set()
            if not release_worker.wait(timeout=5.0):
                raise RuntimeError("test screen worker was not released")
            return original_screen(
                session,
                candidate,
                requested_root_count=requested_root_count,
            )

        class AsyncLauncher:
            def launch(self, request):
                candidate = request.workspace / "candidate.py"
                candidate.write_text(CANDIDATE_SOURCE, encoding="utf-8")
                self.tasks = [
                    submit_reference_candidate_screen(
                        request.context_path,
                        candidate_path=candidate,
                        candidate_id=f"queued-{index}",
                        requested_root_count=4,
                    )
                    for index in range(2)
                ]
                if not started.wait(timeout=2.0):
                    raise RuntimeError("async screen worker did not start")
                try:
                    submit_reference_candidate_screen(
                        request.context_path,
                        candidate_path=candidate,
                        candidate_id="queue-overflow",
                        requested_root_count=4,
                    )
                except ReferenceEvaluatorClientError as exc:
                    self.queue_error = (
                        exc.code,
                        exc.http_status,
                        exc.message,
                    )
                else:
                    raise AssertionError("third inflight screen was accepted")

                assert request.revoke_evaluator_capability is not None
                request.revoke_evaluator_capability()
                try:
                    submit_reference_candidate_screen(
                        request.context_path,
                        candidate_path=candidate,
                        candidate_id="after-revoke",
                        requested_root_count=4,
                    )
                except ReferenceEvaluatorClientError as exc:
                    self.revoke_error = (
                        exc.code,
                        exc.http_status,
                        exc.message,
                    )
                else:
                    raise AssertionError("revoked screen submit was accepted")

                release_worker.set()
                write_reference_candidate_submission(
                    request.context_path,
                    candidate_id="queued-final",
                    bundle_path=candidate,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            incumbent_ref, _ = _incumbent(root)
            session, _ = _strong_session(
                incumbent_ref,
                screen_credits=4,
                large_release_size=1,
            )
            launcher = AsyncLauncher()
            program = ReferenceFilesystemMacroAgentProgram(
                run_root=root / "run",
                run_id="bounded-async-oracle",
                launcher=launcher,
            )
            with patch.object(DevelopmentSession, "screen", blocked_screen):
                outcome = program.run(_agent_context(session, incumbent_ref))

            self.assertEqual(
                launcher.queue_error,
                (
                    "screen_queue_full",
                    429,
                    "async screen inflight limit reached",
                ),
            )
            self.assertEqual(launcher.revoke_error[0:2], ("invalid_request", 400))
            self.assertIn("capability is expired", launcher.revoke_error[2])
            # RPC close drains all accepted work before the ledger is inspected.
            self.assertEqual(len(session.screen_ledger), 2)
            self.assertEqual(session.screen_charged_root_credits, 4)
            self.assertEqual(
                outcome.candidate.ref.bundle_sha256,
                launcher.tasks[0]["candidate_sha256"],
            )


    def test_completed_solution_oracle_rejects_hidden_case_before_factory(self):
        hidden = Case(
            case_id="hidden-oracle-case",
            source=CaseSource.GENERATED,
            visibility=CaseVisibility.HIDDEN,
            public_data={"public": True},
            round_index=1,
        )
        factory_calls = []

        def forbidden_factory(case):
            factory_calls.append(case.case_id)
            raise AssertionError("hidden completed answer was accessed")

        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            (workspace / "arrays").mkdir()
            rpc = ReferenceDevelopmentEvaluatorRPC(
                _evaluator(case_credit_limit=1),
                workspace=workspace,
                oracle_cases=(hidden,),
                development_solution_session_factory=forbidden_factory,
                development_solution_session_methods=("query_completed",),
            )
            payload = {
                "schema": "reference-numerical-oracle-request-v1",
                "namespace": "development_solution",
                "case_id": candidate_facing_case(hidden).case_id,
                "method": "query_completed",
                "args": {"type": "tuple", "items": []},
                "kwargs": {"type": "mapping", "items": {}},
            }
            with self.assertRaisesRegex(PermissionError, "DEVELOPMENT-only"):
                rpc._server.call_numerical_oracle(payload)
            rpc._server.server_close()
        self.assertEqual(factory_calls, [])

    def test_loader_accepts_formal_prepare_solution_without_legacy_adapter(self):
        source = textwrap.dedent(
            """
            class Prepared:
                def __init__(self, case_id, budget):
                    self.case_id = case_id
                    self.budget = budget

                def evaluate(self, query):
                    return (self.case_id, self.budget, query)

            def prepare_solution(case, solve_budget=None):
                return Prepared(case.case_id, solve_budget)
            """
        ).lstrip()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "formal.py"
            path.write_text(source, encoding="utf-8")
            formal = load_python_macro_candidate(path)
            prepare = getattr(formal, "prepare_solution", None)
            self.assertTrue(callable(prepare))
            handle = prepare(_case(), solve_budget=7.5)
            self.assertEqual(handle.evaluate("points-query"), (
                "case-1",
                7.5,
                "points-query",
            ))

            legacy_path = Path(temporary) / "legacy.py"
            legacy_path.write_text(CANDIDATE_SOURCE, encoding="utf-8")
            legacy = load_python_macro_candidate(legacy_path)
            self.assertIsNone(getattr(legacy, "prepare_solution", None))

    def test_multifile_fresh_runtime_isolates_formal_phase_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate_root = root / "counter-candidate"
            candidate_root.mkdir()
            (candidate_root / "helper.py").write_text(
                textwrap.dedent(
                    """
                    counter = 0

                    def next_value():
                        global counter
                        counter += 1
                        return float(counter)
                    """
                ).lstrip(),
                encoding="utf-8",
            )
            (candidate_root / "submission.py").write_text(
                textwrap.dedent(
                    """
                    import sys

                    from .helper import next_value
                    from pde_agent_verifier_solver.sdk import AnswerView

                    def predict_macros(case, query, budget=None):
                        leaked_modules = tuple(
                            name
                            for name in sys.modules
                            if name.startswith("_reference_macro_candidate_")
                        )
                        if leaked_modules:
                            raise RuntimeError(
                                f"formal runtime modules leaked: {leaked_modules!r}"
                            )
                        value = next_value()
                        return AnswerView(
                            kind=query.kind,
                            fields={"density": (value,)},
                            axes={"x": (0.0,)},
                            shape=(1,),
                        )
                    """
                ).lstrip(),
                encoding="utf-8",
            )
            write_candidate_bundle_manifest(
                candidate_root,
                files=("submission.py", "helper.py"),
            )
            attestor = ReferenceFilesystemBundleAttestor(root / "sealed")
            ref = candidate_ref_from_attestation(
                attestor.attest(
                    AgentBundleSubmission("counter-candidate", str(candidate_root)),
                    round_index=1,
                    agent_id="counter-agent",
                    parent_sha256="a" * 64,
                )
            )
            submission = _reference_candidate_submission(ref)

            selection_runtime = submission.fresh_program()
            selection_values = tuple(
                selection_runtime.predict_macros(
                    _case(),
                    QUERY,
                ).fields["density"][0]
                for _ in range(2)
            )
            self.assertFalse(
                any(
                    name.startswith("_reference_macro_candidate_")
                    for name in sys.modules
                )
            )
            terminal_runtime = submission.fresh_program()
            terminal_value = terminal_runtime.predict_macros(
                _case(),
                QUERY,
            ).fields["density"][0]
            self.assertFalse(
                any(
                    name.startswith("_reference_macro_candidate_")
                    for name in sys.modules
                )
            )

            self.assertIsNotNone(submission.program_factory)
            self.assertEqual(selection_values, (1.0, 2.0))
            self.assertEqual(terminal_value, 1.0)


    def test_deferred_loader_counts_module_execution_inside_prepare_clock(self):
        import builtins

        event_name = "_pde_candidate_deferred_timing_events"
        events = []
        setattr(builtins, event_name, events)
        source = textwrap.dedent(
            f"""
            import builtins
            builtins.{event_name}.append("module_load")

            class Prepared:
                def evaluate(self, query):
                    return query

            def prepare_solution(case, solve_budget=None):
                del case, solve_budget
                builtins.{event_name}.append("prepare")
                return Prepared()
            """
        ).lstrip()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "deferred.py"
                path.write_text(source, encoding="utf-8")
                program = load_python_macro_candidate(
                    path,
                    deferred=True,
                )
                self.assertEqual(events, [])
                clock_values = iter((4.0, 4.5))

                def clock():
                    events.append("clock")
                    return next(clock_values)

                prepared = measure_prepare_solution(
                    program,
                    _case(),
                    solve_budget=3.0,
                    clock=clock,
                )
                self.assertEqual(
                    events,
                    ["clock", "module_load", "prepare", "clock"],
                )
                self.assertAlmostEqual(
                    prepared.candidate_solve_wall_seconds,
                    0.5,
                )
        finally:
            delattr(builtins, event_name)

    def test_public_problem_and_both_oracles_share_revocable_runtime(self):
        equation_method = PublicOracleMethod(
            name="query_equation",
            description="Evaluate public equation data at Agent coordinates.",
            input_schema={"coordinates": "agent_selected"},
            output_schema={"dtype": "complex128"},
            cost_unit="coordinate",
            limits={"max_points_per_request": 128},
        )
        completed_method = PublicOracleMethod(
            name="query_completed",
            description="Evaluate a DEVELOPMENT completed solution.",
            input_schema={"coordinates": "agent_selected"},
            output_schema={"dtype": "complex128"},
            cost_unit="coordinate",
            limits={"max_points_per_request": 128},
        )
        bundle = PublicProblemBundle(
            bundle_id="reference-public-problem-v1",
            equation_id="reference-equation-v1",
            description="A complete public reference equation.",
            resources=(
                PublicResource(
                    relative_path="reference/operator.py",
                    data=b"PUBLIC_OPERATOR = 'reference'\n",
                    media_type="text/x-python",
                    description="Public reference operator.",
                ),
            ),
            oracle_methods=(equation_method,),
            development_solution_oracle_methods=(completed_method,),
        )
        equation_sessions = []
        completed_sessions = []

        class EquationSession:
            def __init__(self, case):
                self.case_id = case.case_id
                self.closed = False

            def query_equation(self, coordinates, bias=0.0):
                self.coordinates = np.asarray(coordinates).copy()
                return np.asarray(coordinates, dtype=np.complex128) + bias

            def close(self):
                self.closed = True

        class CompletedSession:
            def __init__(self, case):
                self.case_id = case.case_id
                self.closed = False

            def query_completed(self, coordinates):
                values = np.asarray(coordinates, dtype=np.complex128)
                return {
                    "values": values * (2.0 - 1.0j),
                    "marker": 5.0 + 6.0j,
                }

            def close(self):
                self.closed = True

        def equation_factory(case):
            session = EquationSession(case)
            equation_sessions.append(session)
            return session

        def completed_factory(case):
            session = CompletedSession(case)
            completed_sessions.append(session)
            return session

        class OracleLauncher:
            def launch(self, request):
                metadata = json.loads(
                    request.context_path.read_text(encoding="utf-8")
                )
                self.oracle_metadata = metadata["numerical_oracle"]
                self.public = load_reference_public_workspace(
                    request.context_path,
                    mmap_arrays=False,
                )
                initial_case_id = self.public.cases[0].case_id
                self.initial_case_id = initial_case_id
                self.initial_release_resident_after_context_write = (
                    session.latest_release is not None
                )
                self.equation_values = query_reference_public_oracle(
                    request.context_path,
                    namespace="equation",
                    case_id=initial_case_id,
                    method="query_equation",
                    args=(
                        np.asarray(
                            [1.0 + 2.0j, -1.0 + 0.5j],
                            dtype=np.complex128,
                        ),
                    ),
                    kwargs={"bias": 3.0 + 4.0j},
                )
                self.initial_completed = query_reference_public_oracle(
                    request.context_path,
                    namespace="development_solution",
                    case_id=initial_case_id,
                    method="query_completed",
                    args=(np.asarray([2.0, 3.0]),),
                )
                cli = subprocess.run(
                    self.oracle_metadata["cli"]
                    + [
                        "--namespace",
                        "equation",
                        "--case-id",
                        initial_case_id,
                        "--method",
                        "query_equation",
                        "--args-json",
                        "[[1.0, 2.0]]",
                        "--kwargs-json",
                        '{"bias": 0.5}',
                    ],
                    cwd=request.workspace,
                    env={
                        **os.environ,
                        "PYTHONPATH": os.pathsep.join(
                            (
                                str(Path(__file__).parents[1] / "src"),
                                os.environ.get("PYTHONPATH", ""),
                            )
                        ),
                    },
                    text=True,
                    capture_output=True,
                    check=False,
                )
                if cli.returncode != 0:
                    raise AssertionError(cli.stderr)
                cli_payload = json.loads(cli.stdout)
                cli_node = cli_payload["result"]
                self.cli_values = np.load(
                    request.workspace / cli_node["path"],
                    allow_pickle=False,
                )

                released_payload = release_reference_completed_answers(
                    request.context_path,
                    root_count=1,
                )
                released = load_reference_completed_answer_release(
                    request.workspace / released_payload["manifest_path"],
                    mmap_arrays=False,
                )
                self.released_case_id = released.cases[0].case_id
                try:
                    query_reference_public_oracle(
                        request.context_path,
                        namespace="development_solution",
                        case_id=initial_case_id,
                        method="query_completed",
                        args=(np.asarray([1.0]),),
                    )
                except ReferenceEvaluatorClientError as exc:
                    self.replaced_case_error = (exc.code, exc.http_status)
                else:
                    raise AssertionError("evicted Oracle case remained callable")
                self.released_completed = query_reference_public_oracle(
                    request.context_path,
                    namespace="development_solution",
                    case_id=self.released_case_id,
                    method="query_completed",
                    args=(np.asarray([5.0]),),
                )
                second_payload = release_reference_completed_answers(
                    request.context_path,
                    root_count=1,
                )
                second_release = load_reference_completed_answer_release(
                    request.workspace / second_payload["manifest_path"],
                    mmap_arrays=False,
                )
                self.second_released_case_id = second_release.cases[0].case_id
                try:
                    query_reference_public_oracle(
                        request.context_path,
                        namespace="development_solution",
                        case_id=self.released_case_id,
                        method="query_completed",
                        args=(np.asarray([1.0]),),
                    )
                except ReferenceEvaluatorClientError as exc:
                    self.second_replacement_error = (exc.code, exc.http_status)
                else:
                    raise AssertionError("previous fresh batch remained callable")
                self.second_released_completed = query_reference_public_oracle(
                    request.context_path,
                    namespace="development_solution",
                    case_id=self.second_released_case_id,
                    method="query_completed",
                    args=(np.asarray([7.0]),),
                )
                try:
                    query_reference_public_oracle(
                        request.context_path,
                        namespace="equation",
                        case_id=self.released_case_id,
                        method="close",
                    )
                except ReferenceAgentRuntimeError as exc:
                    self.allowlist_error = str(exc)
                else:
                    raise AssertionError("non-allowlisted method was accepted")

                resources = self.public.public_resources_path
                manifest = self.public.public_problem_manifest_path
                assert resources is not None and manifest is not None
                self.resource_modes = (
                    stat.S_IMODE(resources.stat().st_mode),
                    stat.S_IMODE(
                        (resources / "reference" / "operator.py").stat().st_mode
                    ),
                    stat.S_IMODE(manifest.stat().st_mode),
                )
                candidate = request.workspace / "candidate.py"
                candidate.write_text(CANDIDATE_SOURCE, encoding="utf-8")
                self.open_screens = tuple(
                    screen_reference_candidate(
                        request.context_path,
                        candidate_path=candidate,
                        candidate_id="repeatable-open-screen",
                        requested_root_count=3,
                    )
                    for _index in range(6)
                )
                write_reference_candidate_submission(
                    request.context_path,
                    candidate_id="public-oracle-final",
                    bundle_path=candidate,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            incumbent_ref, _ = _incumbent(root)
            session, _factory = _strong_session(
                incumbent_ref,
                release_credits=2,
                initial_release_count=1,
                large_release_size=2,
                open_search=True,
            )
            launcher = OracleLauncher()
            runtime = ReferenceFilesystemMacroAgentProgram(
                run_root=root / "run",
                run_id="public-problem-oracle-runtime",
                launcher=launcher,
            )
            context = MacroAgentContext(
                round_index=1,
                agent=MacroAgentSpec("oracle-agent", "Oracle Agent"),
                prompt="Use the public numerical Oracle.",
                cases=session.public_cases,
                evaluator=session,
                incumbent=incumbent_ref,
                prior_handoff=None,
                public_problem_bundle=bundle,
                public_session_factory=equation_factory,
                public_session_methods=("query_equation",),
                development_solution_session_factory=completed_factory,
                development_solution_session_methods=("query_completed",),
            )

            outcome = runtime.run(context)

            self.assertTrue(
                launcher.initial_release_resident_after_context_write
            )
            np.testing.assert_allclose(
                launcher.equation_values,
                np.asarray([4.0 + 6.0j, 2.0 + 4.5j]),
            )
            np.testing.assert_allclose(
                launcher.initial_completed["values"],
                np.asarray([4.0 - 2.0j, 6.0 - 3.0j]),
            )
            np.testing.assert_allclose(
                launcher.cli_values,
                np.asarray([1.5, 2.5]),
            )
            self.assertEqual(
                launcher.initial_completed["marker"],
                5.0 + 6.0j,
            )
            np.testing.assert_allclose(
                launcher.released_completed["values"],
                np.asarray([10.0 - 5.0j]),
            )
            np.testing.assert_allclose(
                launcher.second_released_completed["values"],
                np.asarray([14.0 - 7.0j]),
            )
            self.assertEqual(
                launcher.replaced_case_error,
                ("oracle_access_denied", 403),
            )
            self.assertEqual(
                launcher.second_replacement_error,
                ("oracle_access_denied", 403),
            )
            self.assertIn("not allowlisted", launcher.allowlist_error)
            self.assertEqual(launcher.resource_modes, (0o500, 0o400, 0o400))
            self.assertEqual(
                launcher.public.public_problem_manifest["bundle_sha256"],
                bundle.bundle_sha256,
            )
            self.assertEqual(
                launcher.public.equation_oracle_methods,
                ("query_equation",),
            )
            self.assertEqual(
                launcher.public.development_solution_oracle_methods,
                ("query_completed",),
            )
            self.assertFalse(
                launcher.oracle_metadata["fixed_lifetime_or_sample_quota"]
            )
            self.assertTrue(launcher.public.open_search)
            self.assertIsNone(launcher.public.remaining_root_credits)
            self.assertIsNone(launcher.public.remaining_release_root_credits)
            self.assertEqual(launcher.public.screen_root_request_capacity, 16)
            self.assertEqual(launcher.public.release_root_request_capacity, 2)
            self.assertEqual(
                launcher.public.observational_usage["release_sampled_roots"],
                1,
            )
            self.assertEqual(
                launcher.oracle_metadata["coordinate_policy"],
                "agent_selected",
            )
            self.assertEqual(len(equation_sessions), 2)
            self.assertEqual(len(completed_sessions), 3)
            self.assertEqual(len(launcher.open_screens), 6)
            self.assertTrue(
                all(not row["accounting"]["cached"] for row in launcher.open_screens)
            )
            self.assertEqual(session.screen_charged_root_credits, 18)
            self.assertEqual(session.release_charged_root_credits, 3)
            self.assertTrue(all(item.closed for item in equation_sessions))
            self.assertTrue(all(item.closed for item in completed_sessions))
            self.assertIsNotNone(session.latest_release)
            self.assertEqual(
                outcome.candidate.ref.candidate_id,
                "oracle-agent:r1:public-oracle-final",
            )
            workspace = next((root / "run" / "round-0001").iterdir())
            terminal = json.loads(
                (workspace / "context.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                terminal["numerical_oracle"]["capability_status"],
                "expired",
            )
            ledger = json.loads(
                (workspace / "development_ledger.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertFalse(ledger["fixed_lifetime_root_quota"])
            self.assertNotIn("root_credits", ledger)
            self.assertEqual(
                ledger["observational_usage"],
                {
                    "release_sampled_roots": 3,
                    "screen_evaluated_roots": 18,
                },
            )
            session.close()


    def test_parent_guidance_rpc_proxy_is_revocable_and_locator_free(self):
        frozen = _RuntimeFrozenGuidanceOracle()
        seeds = iter(range(800, 900))
        nonces = (f"worker-nonce-{index}" for index in range(100))
        guidance = DevelopmentGuidanceOracle(
            frozen,
            inference_wall_budget_seconds=5.0,
            compute_capacity="test-worker",
            request_namespace="worker-proxy",
            entropy64=lambda: next(seeds),
            nonce_factory=lambda: next(nonces),
        )
        with tempfile.TemporaryDirectory() as temporary:
            worker_root = Path(temporary) / "worker"
            worker_root.mkdir()
            with ReferenceGuidanceOracleRPC(
                guidance,
                workspace=worker_root / "guidance_transport",
            ) as rpc:
                capability = rpc.worker_capability()
                capability_text = json.dumps(
                    capability, sort_keys=True
                )
                proxy = (
                    ReferenceGuidanceOracleProxy.from_worker_capability(
                        capability,
                        worker_root=worker_root,
                    )
                )
                assert proxy is not None
                self.assertEqual(
                    (
                        proxy.call_count,
                        proxy.campaign_guided_generation_wall_seconds,
                        proxy.campaign_solution_query_wall_seconds,
                        proxy.closed,
                    ),
                    (0, 0.0, 0.0, False),
                )
                self.assertEqual(
                    proxy.snapshot.campaign_guidance_training_wall_seconds,
                    1.0,
                )
                self.assertIsInstance(proxy.snapshot, CandidateGuidanceSnapshot)
                batch = proxy.sample(
                    GuidedSampleSpec(2, {"focus": "worker-hard"})
                )
                solution = proxy.query_completed_solution(
                    batch.samples[0].sample_id,
                    "query_values",
                    np.asarray([1.0 + 2.0j]),
                    scale=2.0,
                )
                self.assertEqual(proxy.call_count, 1)
                self.assertEqual(
                    proxy.campaign_guided_generation_wall_seconds,
                    batch.campaign_guided_generation_wall_seconds,
                )
                self.assertGreater(
                    proxy.campaign_solution_query_wall_seconds, 0.0
                )
                self.assertNotIn("path", capability)
                self.assertNotIn("locator", capability_text)
                self.assertNotIn("controller_training_seed", capability_text)
                self.assertNotIn("guided_artifact", capability_text)
                self.assertNotIn("resolver", capability_text)
                self.assertIn(
                    "campaign_guidance_training_wall_seconds", capability
                )
                self.assertIsInstance(batch, CandidateGuidedSampleBatch)
                self.assertEqual(len(batch.samples), 2)
                self.assertTrue(
                    all(
                        isinstance(sample, CandidateGuidedSample)
                        and not hasattr(sample, "answer_access_locator")
                        for sample in batch.samples
                    )
                )
                np.testing.assert_allclose(
                    solution["values"],
                    np.asarray([2.0 + 4.0j]),
                )
            self.assertFalse(proxy.closed)
            with self.assertRaises(ReferenceEvaluatorClientError):
                proxy.sample(GuidedSampleSpec(1))
            proxy.close()
            self.assertTrue(proxy.closed)
            with self.assertRaises(RuntimeError):
                proxy.sample(GuidedSampleSpec(1))

    def test_guidance_rpc_is_fresh_inference_only_and_locator_free(self):
        frozen = _RuntimeFrozenGuidanceOracle()
        seeds = iter(range(100, 1000))
        nonces = (f"fresh-nonce-{index}" for index in range(100))
        guidance = DevelopmentGuidanceOracle(
            frozen,
            inference_wall_budget_seconds=5.0,
            compute_capacity="test-compute",
            request_namespace="runtime-agent",
            entropy64=lambda: next(seeds),
            nonce_factory=lambda: next(nonces),
        )

        class GuidanceLauncher:
            def launch(self, request):
                metadata = json.loads(
                    request.context_path.read_text(encoding="utf-8")
                )
                self.active_metadata = metadata["guidance_oracle"]
                self.context_text = json.dumps(metadata, sort_keys=True)
                self.public = load_reference_public_workspace(
                    request.context_path,
                    mmap_arrays=False,
                )
                spec = GuidedSampleSpec(
                    2,
                    {"focus": "hard-tail", "temperature": 0.75},
                )
                self.batches = tuple(
                    sample_reference_guidance(
                        request.context_path,
                        spec=spec,
                        mmap_arrays=False,
                    )
                    for _index in range(5)
                )
                cli = subprocess.run(
                    self.active_metadata["sample_cli"]
                    + [
                        "--sample-count",
                        "1",
                        "--settings-json",
                        '{"focus": "cli-fresh"}',
                    ],
                    cwd=request.workspace,
                    env={
                        **os.environ,
                        "PYTHONPATH": os.pathsep.join(
                            (
                                str(Path(__file__).parents[1] / "src"),
                                os.environ.get("PYTHONPATH", ""),
                            )
                        ),
                    },
                    text=True,
                    capture_output=True,
                    check=False,
                )
                if cli.returncode != 0:
                    raise AssertionError(cli.stderr)
                self.cli_text = cli.stdout
                self.cli_payload = json.loads(cli.stdout)
                first = self.batches[0].samples[0]
                self.solution = query_reference_guided_completed_solution(
                    request.context_path,
                    sample_id=first.sample_id,
                    method="query_values",
                    args=(
                        np.asarray(
                            [1.0 + 2.0j, -2.0 + 0.5j],
                            dtype=np.complex128,
                        ),
                    ),
                    kwargs={"scale": 2.0 - 1.0j},
                )
                try:
                    query_reference_guided_completed_solution(
                        request.context_path,
                        sample_id="never-issued",
                        method="query_values",
                        args=(np.asarray([1.0]),),
                    )
                except ReferenceEvaluatorClientError as exc:
                    self.unknown_error = (
                        exc.code,
                        exc.http_status,
                        exc.message,
                    )
                else:
                    raise AssertionError(
                        "unknown guided sample was resolved"
                    )
                try:
                    query_reference_guided_completed_solution(
                        request.context_path,
                        sample_id=first.sample_id,
                        method="open_development_solution_session",
                    )
                except ReferenceAgentRuntimeError as exc:
                    self.allowlist_error = str(exc)
                else:
                    raise AssertionError(
                        "non-allowlisted guided method was accepted"
                    )
                candidate = request.workspace / "candidate.py"
                candidate.write_text(CANDIDATE_SOURCE, encoding="utf-8")
                write_reference_candidate_submission(
                    request.context_path,
                    candidate_id="guidance-final",
                    bundle_path=candidate,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            incumbent_ref, _ = _incumbent(root)
            evaluator = _evaluator(case_credit_limit=1)
            launcher = GuidanceLauncher()
            runtime = ReferenceFilesystemMacroAgentProgram(
                run_root=root / "run",
                run_id="guidance-loopback-runtime",
                launcher=launcher,
            )
            context = MacroAgentContext(
                round_index=1,
                agent=MacroAgentSpec("guided-agent", "Guided Agent"),
                prompt="Use fresh inference-only guidance.",
                cases=evaluator.public_cases,
                evaluator=evaluator,
                incumbent=incumbent_ref,
                prior_handoff=None,
                guidance_oracle=guidance,
            )

            outcome = runtime.run(context)

            self.assertEqual(guidance.call_count, 6)
            self.assertEqual(len(frozen.generated_requests), 6)
            sample_ids = {
                sample.sample_id
                for batch in launcher.batches
                for sample in batch.samples
            }
            self.assertEqual(len(sample_ids), 10)
            self.assertTrue(
                all(
                    isinstance(sample, CandidateGuidedSample)
                    and sample.case.visibility is CaseVisibility.DEVELOPMENT
                    and not hasattr(sample, "answer_access_locator")
                    and not hasattr(sample, "reference_answer_sha256")
                    and not hasattr(sample, "public_case_sha256")
                    for batch in launcher.batches
                    for sample in batch.samples
                )
            )
            np.testing.assert_array_equal(
                launcher.batches[0].samples[0].case.public_data[
                    "public_array"
                ],
                np.asarray([0.0, 1.0]),
            )
            np.testing.assert_allclose(
                launcher.solution["values"],
                np.asarray([4.0 + 3.0j, -3.5 + 3.0j]),
            )
            self.assertEqual(launcher.solution["marker"], 2.0 + 3.0j)
            self.assertEqual(
                launcher.unknown_error[:2],
                ("guidance_access_denied", 403),
            )
            self.assertEqual(len(frozen.opened_sample_ids), 1)
            public_sample_id = launcher.batches[0].samples[0].sample_id
            self.assertNotEqual(frozen.opened_sample_ids[0], public_sample_id)
            self.assertNotIn(frozen.opened_sample_ids[0], launcher.context_text)
            self.assertNotIn(frozen.opened_sample_ids[0], launcher.cli_text)
            self.assertTrue(all(session.closed for session in frozen.sessions))
            self.assertIn("not allowlisted", launcher.allowlist_error)
            metadata = launcher.active_metadata
            self.assertTrue(metadata["unlimited_fresh_inference"])
            self.assertFalse(metadata["fixed_lifetime_or_sample_quota"])
            self.assertFalse(metadata["training_available"])
            self.assertFalse(metadata["hidden_resolver_available"])
            self.assertFalse(metadata["answer_locator_exposed"])
            self.assertNotIn(
                "sealed://private-guidance-artifact",
                launcher.context_text,
            )
            self.assertNotIn(
                "sealed://private-answer-locator",
                launcher.context_text,
            )
            self.assertEqual(launcher.cli_payload["action"], "sample")
            self.assertEqual(
                len(launcher.cli_payload["batch"]["samples"]),
                1,
            )
            self.assertNotIn("answer_access_locator", launcher.cli_text)
            self.assertNotIn(
                "sealed://private-answer-locator",
                launcher.cli_text,
            )
            self.assertTrue(
                launcher.public.guidance_unlimited_fresh_inference
            )
            self.assertEqual(
                launcher.public.guidance_solution_methods,
                ("query_values",),
            )
            self.assertEqual(
                launcher.public.guidance_snapshot_sha256,
                frozen.snapshot.snapshot_sha256,
            )
            self.assertEqual(
                outcome.candidate.ref.candidate_id,
                "guided-agent:r1:guidance-final",
            )
            workspace = next((root / "run" / "round-0001").iterdir())
            terminal = json.loads(
                (workspace / "context.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                terminal["guidance_oracle"]["capability_status"],
                "expired",
            )

    def test_unreleased_points_are_redacted_but_formal_evaluation_runs(self):
        points_query = AnswerQuery.at_points(
            fields=("density",),
            points=({"x": 0.125}, {"x": 0.875}),
        )

        class PointTruth:
            def answer(self, query):
                return AnswerView(
                    kind=query.kind,
                    fields={"density": (SECRET_TARGET, SECRET_TARGET)},
                    points=query.points,
                    shape=(len(query.points),),
                )

        cohort = PreparedMacroCohort.prepare(
            cohort_id="query-blind-development",
            records=(CaseRecord(_case(), PointTruth()),),
            matcher=_Matcher(),
            query=points_query,
        )
        evaluator = DevelopmentMacroEvaluator(
            cohort,
            case_credit_limit=1,
            allow_finite_truth=False,
            safe_metric_names=(),
        )
        formal_source = textwrap.dedent(
            """
            from pde_agent_verifier_solver.sdk import AnswerView

            class Prepared:
                def evaluate(self, query):
                    return AnswerView(
                        kind=query.kind,
                        fields={
                            "density": tuple(0.0 for _point in query.points)
                        },
                        points=query.points,
                        shape=(len(query.points),),
                    )

            def prepare_solution(case, solve_budget=None):
                return Prepared()
            """
        ).lstrip()

        class QueryBlindLauncher:
            def launch(self, request):
                metadata = json.loads(
                    request.context_path.read_text(encoding="utf-8")
                )
                public_query = metadata["public_tasks"][0]["query"]
                self.public_query = public_query
                self.context_text = json.dumps(metadata, sort_keys=True)
                public = load_reference_public_workspace(
                    request.context_path,
                    mmap_arrays=False,
                )
                self.loaded_points = public.queries[0].points
                candidate = request.workspace / "formal.py"
                candidate.write_text(formal_source, encoding="utf-8")
                self.evaluation = evaluate_reference_candidate(
                    request.context_path,
                    candidate_path=candidate,
                    candidate_id="query-blind-formal",
                )
                write_reference_candidate_submission(
                    request.context_path,
                    candidate_id="query-blind-formal",
                    bundle_path=candidate,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            incumbent_ref, _ = _incumbent(root)
            launcher = QueryBlindLauncher()
            runtime = ReferenceFilesystemMacroAgentProgram(
                run_root=root / "run",
                run_id="query-blind-runtime",
                launcher=launcher,
            )
            outcome = runtime.run(
                MacroAgentContext(
                    round_index=1,
                    agent=MacroAgentSpec(
                        "query-blind-agent",
                        "Query Blind Agent",
                    ),
                    prompt="Prepare before the controller reveals points.",
                    cases=evaluator.public_cases,
                    evaluator=evaluator,
                    incumbent=incumbent_ref,
                    prior_handoff=None,
                )
            )

            self.assertTrue(launcher.public_query["points_redacted"])
            self.assertEqual(launcher.public_query["point_count"], 2)
            self.assertEqual(launcher.public_query["points"], [])
            self.assertEqual(launcher.loaded_points, ({}, {}))
            self.assertNotIn('"x": 0.125', launcher.context_text)
            self.assertNotIn('"x": 0.875', launcher.context_text)
            self.assertEqual(
                launcher.evaluation["candidate_id"],
                "query-blind-formal",
            )
            self.assertEqual(
                outcome.candidate.ref.candidate_id,
                "query-blind-agent:r1:query-blind-formal",
            )


def _case():
    return Case(
        case_id="case-1",
        source=CaseSource.GENERATED,
        visibility=CaseVisibility.DEVELOPMENT,
        public_data={
            "large_public_array": np.arange(2048, dtype=np.float64),
            "metadata": {"name": "public-only", "shape": (2048,)},
        },
        round_index=1,
        tags=("development",),
    )


def _evaluator(*, case_credit_limit, matcher=None):
    selected_matcher = _Matcher() if matcher is None else matcher
    cohort = PreparedMacroCohort.prepare(
        cohort_id="reference-development",
        records=(CaseRecord(_case(), _Truth()),),
        matcher=selected_matcher,
        query=QUERY,
    )
    return DevelopmentMacroEvaluator(
        cohort,
        case_credit_limit=case_credit_limit,
        allow_finite_truth=False,
        safe_metric_names=(),
    )


def _strong_session(
    incumbent_ref,
    *,
    screen_credits=16,
    large_release_size=200_000,
    release_credits=1,
    initial_release_count=1,
    screen_error=None,
    open_search=False,
):
    incumbent_program = load_python_macro_candidate(
        incumbent_ref.bundle_uri,
        expected_sha256=incumbent_ref.bundle_sha256,
    )
    factory = _FreshSessionFactory(
        large_release_size=large_release_size,
        screen_error=screen_error,
    )
    session = DevelopmentSession(
        incumbent=CandidateSubmission(incumbent_ref, incumbent_program),
        cohort_factory=factory,
        screen_root_stages=(4, 16),
        screen_root_credit_limit=screen_credits,
        release_root_credit_limit=release_credits,
        safe_metric_names=(),
        open_search=open_search,
    )
    if initial_release_count:
        session.release(initial_release_count)
    return session, factory


def _agent_context(evaluator, incumbent_ref):
    return MacroAgentContext(
        round_index=1,
        agent=MacroAgentSpec("hardening-agent", "Hardening Agent"),
        prompt="Exercise the reference runtime boundary.",
        cases=evaluator.public_cases,
        evaluator=evaluator,
        incumbent=incumbent_ref,
        prior_handoff=None,
    )


def _incumbent(root):
    incumbent = root / "cold_start.py"
    payload = CANDIDATE_SOURCE.replace("(0.0,)", "(1.0,)").encode("utf-8")
    incumbent.write_bytes(payload)
    attestor = ReferenceFilesystemBundleAttestor(root / "cold-sealed")
    record = attestor.attest(
        AgentBundleSubmission("cold-start", str(incumbent)),
        round_index=0,
        agent_id="cold-start",
        parent_sha256=None,
    )
    return candidate_ref_from_attestation(record), payload


if __name__ == "__main__":
    unittest.main()
