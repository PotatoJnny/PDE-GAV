from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from pde_agent_verifier_solver.engine.guidance_controller import (
    DevelopmentGuidanceOracle,
)
from pde_agent_verifier_solver.engine.macro_closed_loop import (
    CompetitiveSearchPolicy,
    MacroAgentDevelopmentResult,
    MacroAgentSpec,
    MacroClosedLoopRunner,
    MacroLoopConfig,
    SearchCompletionAudit,
)
from pde_agent_verifier_solver.engine.reference_agent_iteration import (
    ReferenceIterationConfig,
    ReferenceProcessRoundDeveloper,
    ReferenceProcessWorkerConfig,
    _reference_agent_specs,
    _reference_loop_config,
)
from pde_agent_verifier_solver.engine.reference_agent_runtime import (
    ReferenceEvaluatorClientError,
    ReferenceFilesystemBundleAttestor,
    ReferenceGuidanceOracleProxy,
    ReferenceGuidanceSnapshotIdentity,
    ReferenceGuidedSample,
    ReferenceGuidedSampleBatch,
)
from pde_agent_verifier_solver.engine.selection import EvidenceStage
from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    AnswerView,
    Case,
    CaseRecord,
    CaseSource,
    CaseVisibility,
    GuidedSampleSpec,
    MacroMatchResult,
    MacroTarget,
    ProgramInfo,
    QueryKind,
)
from pde_agent_verifier_solver.sdk.macro_loop import (
    CandidateRef,
    CandidateSubmission,
    CohortKind,
    MacroLoopPlugin,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class _Truth:
    def answer(self, query: AnswerQuery) -> AnswerView:
        return AnswerView(
            kind=query.kind,
            fields={"macro": (1.0,)},
            axes={"sample": (0.0,)},
            shape=(1,),
        )


class _CaseSource:
    generator_id = "parallel-worker-test-generator"
    artifact_sha256 = "e" * 64

    def __init__(self) -> None:
        self.requests = []

    def sample(self, request):
        self.requests.append(request)
        visibility = (
            CaseVisibility.DEVELOPMENT
            if request.cohort
            in {CohortKind.DEVELOPMENT, CohortKind.DEVELOPMENT_SCREEN}
            else CaseVisibility.HIDDEN
        )
        return tuple(
            CaseRecord(
                Case(
                    case_id=(
                        f"controller-r{request.round_index}-"
                        f"{request.cohort.value}-{request.seed}-{index}"
                    ),
                    source=CaseSource.GENERATED,
                    visibility=visibility,
                    public_data={"input": float(index)},
                    round_index=request.round_index,
                ),
                _Truth(),
            )
            for index in range(request.count)
        )


class _Matcher:
    info = ProgramInfo("parallel-worker-test", "test matcher", "test")
    contract_id = "parallel_worker_macro_v1"
    fields = ("macro",)

    def prepare_target(self, case, truth, query):
        return MacroTarget(case.case_id, self.contract_id, query, truth)

    def match(self, case, prediction, target):
        error = abs(
            prediction.fields["macro"][0] - target.values.fields["macro"][0]
        )
        return MacroMatchResult(
            case.case_id,
            self.contract_id,
            error,
            metrics={"component": error},
        )


class _Objective:
    matcher = _Matcher()
    agent_visible_metrics = ()
    published_metric_names = ()

    def query(self, case, profile):
        return AnswerQuery.grid(fields=("macro",), sample=1)


class _ProcessGuidanceOracle(DevelopmentGuidanceOracle):
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self._snapshot = ReferenceGuidanceSnapshotIdentity(
            _digest(f"snapshot-{agent_id}"),
            0.0,
        )
        self._issued = set()
        self.sample_calls = 0
        self.solution_calls = 0
        self._fake_closed = False
        self.close_calls = 0

    @property
    def closed(self):
        return self._fake_closed

    @property
    def snapshot(self):
        return self._snapshot

    @property
    def available_solution_methods(self):
        return ("query_values",)

    def sample(self, spec):
        if self.closed:
            raise RuntimeError("process guidance oracle is closed")
        if not isinstance(spec, GuidedSampleSpec):
            raise TypeError("expected GuidedSampleSpec")
        self.sample_calls += 1
        rows = []
        for index in range(spec.sample_count):
            sample_id = (
                f"{self.agent_id}-sample-{self.sample_calls}-{index}"
            )
            self._issued.add(sample_id)
            rows.append(
                ReferenceGuidedSample(
                    sample_id=sample_id,
                    case=Case(
                        case_id=f"case-{sample_id}",
                        source=CaseSource.GENERATED,
                        visibility=CaseVisibility.DEVELOPMENT,
                        public_data={
                            "focus": spec.settings.get("focus"),
                            "index": index,
                            "distribution_lineage": {
                                "kind": "frozen_reward_guided",
                                "snapshot_sha256": self.snapshot.snapshot_sha256,
                                "base_generator_sha256": _digest("base-generator"),
                            },
                            "public_array": np.asarray(
                                [index, index + 1],
                                dtype=np.float64,
                            ),
                        },
                        round_index=1,
                    ),
                    answer_to_case_derivation_commitment_sha256=_digest(
                        "process-answer-to-case-v1"
                    ),
                )
            )
        return ReferenceGuidedSampleBatch(
            snapshot_sha256=self.snapshot.snapshot_sha256,
            sample_settings_commitment_sha256=spec.settings_sha256,
            samples=tuple(rows),
            campaign_guided_generation_wall_seconds=0.0,
        )

    def query_completed_solution(
        self,
        sample_id,
        method,
        /,
        value,
        *,
        scale=1.0,
    ):
        if self.closed:
            raise RuntimeError("process guidance oracle is closed")
        if sample_id not in self._issued:
            raise KeyError(sample_id)
        if method != "query_values":
            raise PermissionError(method)
        self.solution_calls += 1
        return {
            "values": np.asarray(value, dtype=np.float64)
            * float(scale)
        }

    def close(self):
        if self._fake_closed:
            return
        self._fake_closed = True
        self.close_calls += 1
        self._issued.clear()


class _Approximation:
    def __init__(self, error: float) -> None:
        self.error = error

    def predict_macros(self, case, query, budget=None):
        return AnswerView(
            kind=QueryKind.GRID,
            fields={"macro": (1.0 - self.error,)},
            axes={"sample": (0.0,)},
            shape=(1,),
        )


def _candidate(
    *,
    agent: MacroAgentSpec,
    round_index: int,
    incumbent: CandidateSubmission,
    error: float,
) -> CandidateSubmission:
    candidate_id = f"{agent.agent_id}-round-{round_index}"
    return CandidateSubmission(
        CandidateRef(
            candidate_id=candidate_id,
            bundle_sha256=_digest(f"{candidate_id}:{error}"),
            round_index=round_index,
            agent_id=agent.agent_id,
            parent_sha256=incumbent.ref.bundle_sha256,
        ),
        _Approximation(error),
    )


class _ReverseCompletionExecutor:
    """Return the second configured Agent first, as a real race may do."""

    def __init__(self) -> None:
        self.calls = []

    def run_agents(
        self,
        *,
        round_index,
        agents,
        incumbent,
        prior_handoff,
        prior_leaderboard,
        shared_candidates,
    ):
        self.calls.append(
            {
                "round_index": round_index,
                "agent_ids": tuple(agent.agent_id for agent in agents),
                "prior_rounds": tuple(
                    entry.candidate.round_index for entry in prior_leaderboard
                ),
                "shared_rounds": tuple(
                    candidate.round_index for candidate in shared_candidates
                ),
                "prior_handoff_round": (
                    None if prior_handoff is None else prior_handoff.round_index
                ),
            }
        )
        rows = []
        for index, agent in enumerate(agents):
            sampled_id = f"worker-{agent.agent_id}-r{round_index}-development"
            candidate = _candidate(
                agent=agent,
                round_index=round_index,
                incumbent=incumbent,
                error=0.20 + 0.05 * index,
            )
            rows.append(
                MacroAgentDevelopmentResult(
                    agent_id=agent.agent_id,
                    portfolio=((
                        candidate,
                        ("agent_final",),
                    ),),
                    development_case_ids=(sampled_id,),
                    sampled_case_ids=(sampled_id,),
                    sampling_attempts=(),
                    search_completion=SearchCompletionAudit(
                        agent_id=agent.agent_id,
                        distinct_screened_candidates=1,
                        full_stage_candidates=1,
                        released_roots=1,
                        final_candidate_sha256=candidate.ref.bundle_sha256,
                        final_completed_root_count=1,
                        sufficient=True,
                    ),
                )
            )
        return tuple(reversed(rows))


class _FixedResultExecutor:
    def __init__(self, results) -> None:
        self.results = tuple(results)

    def run_agents(self, **_kwargs):
        return self.results


class _CaptureAndFailAgent:
    def __init__(self) -> None:
        self.cases = ()

    def run(self, context):
        self.cases = tuple(context.cases)
        raise RuntimeError("stop after observing the candidate envelope")


def test_real_worker_keeps_controller_ids_for_external_merge_audit():
    agent = MacroAgentSpec("worker-agent", "Worker Agent")
    cold = CandidateSubmission(
        CandidateRef("cold", "c" * 64, 0, "baseline"),
        _Approximation(0.5),
    )
    config = MacroLoopConfig(
        name="external-dual-case-id-test",
        round_count=1,
        agents=(agent,),
        development_case_count=1,
        selection_case_count=2,
        evidence_stages=(
            EvidenceStage(
                2,
                confidence_z=0.0,
                min_win_rate=0.0,
                require_cvar_no_regression=False,
            ),
        ),
        terminal_case_count=2,
        agent_case_credits=1,
        seed_base=91_000,
        seed_stride=1_000,
        development_release_root_limit=1,
        development_screen_root_stages=(1,),
        development_screen_root_credit_limit=1,
        development_screen_confidence_z=0.0,
        terminal_confidence_z=0.0,
        terminal_min_win_rate=0.0,
        terminal_require_cvar_no_regression=False,
    )
    capture = _CaptureAndFailAgent()
    worker_source = _CaseSource()
    worker = MacroClosedLoopRunner(
        legacy_macro_reference=True,
        config=config,
        plugin=MacroLoopPlugin(
            "external-dual-case-id",
            worker_source,
            _Objective(),
        ),
        cold_start=cold,
        agents={agent.agent_id: capture},
    )

    worker_result = worker.run_agent_development(
        round_index=1,
        agent=agent,
        incumbent=cold,
    )

    assert len(capture.cases) == 1
    candidate_case = capture.cases[0]
    assert candidate_case.case_id.startswith("candidate-case-")
    assert candidate_case.source is CaseSource.GENERATED
    assert candidate_case.visibility is CaseVisibility.DEVELOPMENT
    assert candidate_case.round_index is None
    assert candidate_case.parent_case_id is None
    assert candidate_case.tags == ()
    assert set(worker_result.development_case_ids).issubset(
        set(worker_result.sampled_case_ids)
    )
    assert candidate_case.case_id not in worker_result.development_case_ids
    assert all(
        case_id.startswith("controller-r1-development-")
        for case_id in worker_result.development_case_ids
    )

    controller = MacroClosedLoopRunner(
        legacy_macro_reference=True,
        config=config,
        plugin=MacroLoopPlugin(
            "external-dual-case-id",
            _CaseSource(),
            _Objective(),
        ),
        cold_start=cold,
        agents={},
        development_executor=_FixedResultExecutor((worker_result,)),
    )
    result = controller.run()

    assert result.rounds[0].development_case_ids == (
        worker_result.development_case_ids
    )
    assert result.rounds[0].agent_failures == {
        agent.agent_id: "agent_failed:RuntimeError"
    }


def _runner(executor, *, round_count: int = 1):
    source = _CaseSource()
    agents = (
        MacroAgentSpec("z-agent", "Z Agent"),
        MacroAgentSpec("a-agent", "A Agent"),
    )
    cold = CandidateSubmission(
        CandidateRef("cold", "c" * 64, 0, "baseline"),
        _Approximation(0.5),
    )
    config = MacroLoopConfig(
        name="parallel-development-test",
        round_count=round_count,
        agents=agents,
        development_case_count=1,
        selection_case_count=2,
        evidence_stages=(EvidenceStage(
            2,
            confidence_z=0.0,
            min_win_rate=0.0,
            require_cvar_no_regression=False,
        ),),
        terminal_case_count=2,
        agent_case_credits=1,
        seed_base=90_000,
        seed_stride=1_000,
        development_release_root_limit=1,
        development_screen_root_stages=(1,),
        development_screen_root_credit_limit=1,
        development_screen_confidence_z=0.0,
        terminal_confidence_z=0.0,
        terminal_min_win_rate=0.0,
        terminal_require_cvar_no_regression=False,
        search_policy=CompetitiveSearchPolicy(
            leaderboard_limit=4,
            cross_share_after_round=1,
        ),
    )
    runner = MacroClosedLoopRunner(
        legacy_macro_reference=True,
        config=config,
        plugin=MacroLoopPlugin("parallel-development", source, _Objective()),
        cold_start=cold,
        agents={},
        development_executor=executor,
    )
    return runner, source, agents, cold


def _empty_result(agent_id: str, *, sampled_id: str | None = None):
    sampled = sampled_id or f"worker-{agent_id}-development"
    return MacroAgentDevelopmentResult(
        agent_id=agent_id,
        portfolio=(),
        development_case_ids=(sampled,),
        sampled_case_ids=(sampled,),
        sampling_attempts=(),
    )


def test_process_worker_gpu_tokens_are_singleton_unique_and_controller_distinct(
    tmp_path,
):
    config = ReferenceProcessWorkerConfig(
        command=("python", "worker.py"),
        gpu_devices=("GPU-worker-a", "GPU-worker-b"),
        controller_gpu_device="GPU-controller",
        node_local_root=tmp_path / "workers",
        cpus_per_worker=8,
        worker_timeout_seconds=120,
    )

    assert config.gpu_devices == ("GPU-worker-a", "GPU-worker-b")
    assert config.controller_gpu_device == "GPU-controller"
    assert config.node_local_root == (tmp_path / "workers").resolve()
    assert config.worker_timeout_seconds == 120.0

    invalid = (
        ({"gpu_devices": ()}, "worker GPU tokens"),
        ({"gpu_devices": ("0,1", "2")}, "singleton"),
        ({"gpu_devices": ("0", "0")}, "unique"),
        ({"controller_gpu_device": ""}, "controller GPU token"),
        ({"controller_gpu_device": "0,1"}, "singleton"),
        (
            {"gpu_devices": ("0", "1"), "controller_gpu_device": "1"},
            "distinct",
        ),
    )
    base = {
        "command": ("python", "worker.py"),
        "gpu_devices": ("0", "1"),
        "controller_gpu_device": "2",
        "node_local_root": tmp_path / "invalid-workers",
        "cpus_per_worker": 2,
        "worker_timeout_seconds": 30.0,
    }
    for overrides, message in invalid:
        with pytest.raises(ValueError, match=message):
            ReferenceProcessWorkerConfig(**{**base, **overrides})


def test_external_results_merge_in_config_order_and_share_only_prior_round_state():
    executor = _ReverseCompletionExecutor()
    runner, _source, agents, _cold = _runner(executor, round_count=2)

    result = runner.run()

    expected_agent_order = tuple(agent.agent_id for agent in agents)
    assert len(result.rounds) == 2
    for round_index, round_result in enumerate(result.rounds, start=1):
        assert tuple(
            candidate.agent_id for candidate in round_result.candidate_refs
        ) == expected_agent_order
        assert tuple(round_result.search_completion) == expected_agent_order
        assert tuple(round_result.development_case_ids) == tuple(
            f"worker-{agent_id}-r{round_index}-development"
            for agent_id in expected_agent_order
        )
        commitments = {
            (
                finalist.batch.cohort_sha256,
                finalist.batch.target_sha256,
                finalist.batch.case_ids,
            )
            for finalist in round_result.finalist_evaluations
        }
        assert len(commitments) == 1

    first, second = executor.calls
    assert first == {
        "round_index": 1,
        "agent_ids": expected_agent_order,
        "prior_rounds": (),
        "shared_rounds": (),
        "prior_handoff_round": None,
    }
    assert second["round_index"] == 2
    assert second["agent_ids"] == expected_agent_order
    assert second["prior_handoff_round"] == 1
    assert second["prior_rounds"]
    assert second["shared_rounds"]
    # The common leaderboard carries the prior incumbent (round 0 cold
    # start) alongside round-1 nominations; no same-round state is shared.
    assert set(second["prior_rounds"]) == {0, 1}
    assert set(second["shared_rounds"]) == {0, 1}
    assert result.terminal is not None


@pytest.mark.parametrize(
    ("results", "message"),
    (
        (
            (_empty_result("z-agent"),),
            "exactly one result per Agent",
        ),
        (
            (_empty_result("z-agent"), _empty_result("z-agent", sampled_id="two")),
            "exactly one result per Agent",
        ),
        (
            (_empty_result("z-agent"), _empty_result("unknown-agent")),
            "exactly one result per Agent",
        ),
        (
            (
                _empty_result("z-agent"),
                MacroAgentDevelopmentResult(
                    agent_id="a-agent",
                    portfolio=(),
                    development_case_ids=("missing-from-sample-audit",),
                    sampled_case_ids=("different-sampled-case",),
                    sampling_attempts=(),
                ),
            ),
            "initial cases are absent from its sample audit",
        ),
    ),
)
def test_missing_or_malformed_external_agent_results_fail_before_selection(
    results,
    message,
):
    runner, source, _agents, _cold = _runner(_FixedResultExecutor(results))

    with pytest.raises(RuntimeError, match=message):
        runner.run()

    assert source.requests == []


def _process_developer(
    tmp_path: Path,
    *,
    command,
    cohort_store_root: Path | None = None,
):
    cold_path = tmp_path / "cold.py"
    cold_payload = b"def predict_macros(case, query, budget=None):\n    return None\n"
    cold_path.write_bytes(cold_payload)
    launcher = tmp_path / "launcher"
    codex = tmp_path / "codex"
    for executable in (launcher, codex):
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o700)
    auth = tmp_path / "auth.json"
    auth.write_text("{}\n", encoding="utf-8")
    iteration_config = ReferenceIterationConfig(
        name="process-worker-test",
        run_root=tmp_path / "run",
        cold_start_candidate=cold_path,
        agent_launcher=launcher,
        codex_binary=codex,
        auth_file=auth,
        prompt="Exercise process workers.",
        model="test-model",
        reasoning_effort="high",
        agent_timeout_seconds=10.0,
        agent_hard_timeout_seconds=20.0,
        round_count=1,
        development_case_count=1,
        selection_case_count=2,
        evidence_stages=(EvidenceStage(
            2,
            confidence_z=0.0,
            min_win_rate=0.0,
            require_cvar_no_regression=False,
        ),),
        terminal_case_count=2,
        development_case_credits=1,
        seed_base=10_000,
        seed_stride=1_000,
        agent_count=2,
        development_release_root_limit=1,
        development_screen_root_stages=(1,),
        development_screen_root_credit_limit=1,
        development_screen_timeout_seconds=5.0,
        development_screen_confidence_z=0.0,
        terminal_confidence_z=0.0,
        terminal_min_win_rate=0.0,
        terminal_require_cvar_no_regression=False,
        paired_perturbations=False,
        cohort_store_root=cohort_store_root,
    )
    agents = _reference_agent_specs(iteration_config)
    loop_config = _reference_loop_config(iteration_config, agents)
    cold = CandidateSubmission(
        CandidateRef(
            "cold",
            hashlib.sha256(cold_payload).hexdigest(),
            0,
            "baseline",
            bundle_uri=str(cold_path),
        ),
        _Approximation(0.5),
    )
    execution = ReferenceProcessWorkerConfig(
        command=tuple(command),
        gpu_devices=("GPU-worker-a", "GPU-worker-b"),
        controller_gpu_device="GPU-controller",
        node_local_root=tmp_path / "node-local",
        cpus_per_worker=2,
        worker_timeout_seconds=10.0,
    )
    developer = ReferenceProcessRoundDeveloper(
        execution=execution,
        iteration_config=iteration_config,
        loop_config=loop_config,
        controller_root=tmp_path / "controller",
        central_attestor=ReferenceFilesystemBundleAttestor(
            tmp_path / "central-bundles"
        ),
        campaign_cold_start=cold,
        plugin_identity={
            "plugin_id": "test",
            "generator_id": "test-generator",
            "generator_sha256": "f" * 64,
        },
    )
    return developer, agents, cold


@pytest.mark.parametrize(
    ("mode", "expected_failure"),
    (
        ("missing", "worker_report_failed:FileNotFoundError"),
        ("malformed", "worker_report_failed:ReferenceIterationError"),
    ),
)
def test_process_executor_isolates_missing_or_malformed_worker_reports(
    tmp_path: Path,
    mode,
    expected_failure,
):
    if mode == "missing":
        command = ("/usr/bin/true",)
    else:
        writer = tmp_path / "malformed_worker.py"
        writer.write_text(
            "from pathlib import Path\n"
            "import sys\n"
            "target = Path(sys.argv[sys.argv.index('--agent-worker-result') + 1])\n"
            "target.write_text('{not-json', encoding='utf-8')\n",
            encoding="utf-8",
        )
        command = (sys.executable, str(writer))
    developer, agents, cold = _process_developer(tmp_path, command=command)

    results = developer.run_agents(
        round_index=1,
        agents=agents,
        incumbent=cold,
        prior_handoff=None,
        prior_leaderboard=(),
        shared_candidates=(),
    )

    assert tuple(result.agent_id for result in results) == tuple(
        agent.agent_id for agent in agents
    )
    assert all(result.portfolio == () for result in results)
    assert all(result.failure == expected_failure for result in results)
    audit = developer.audit_payload()["rounds"][0]
    assert audit["same_round_barrier"] is True
    assert audit["overlap_verified"] is False
    assert [worker["status"] for worker in audit["workers"]] == [
        "invalid_report",
        "invalid_report",
    ]
    assert len({worker["pid"] for worker in audit["workers"]}) == 2
    assert [worker["gpu_device"] for worker in audit["workers"]] == [
        "GPU-worker-a",
        "GPU-worker-b",
    ]

def _write_fake_process_worker(tmp_path: Path) -> Path:
    worker = tmp_path / "fake_reference_process_worker.py"
    worker.write_text(
        """from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from pde_agent_verifier_solver.engine.reference_agent_runtime import (
    ReferenceGuidanceOracleProxy,
)
from pde_agent_verifier_solver.sdk import GuidedSampleSpec


def _argument(name: str) -> Path:
    return Path(sys.argv[sys.argv.index(name) + 1]).resolve()


request_path = _argument("--agent-worker-request")
report_path = _argument("--agent-worker-result")
request_bytes = request_path.read_bytes()
request = json.loads(request_bytes)
agent_id = request["agent"]["agent_id"]
round_index = request["round_index"]
parent_sha256 = request["incumbent"]["bundle_sha256"]
token = os.environ.get("CUDA_VISIBLE_DEVICES", "")
assigned_token = os.environ.get("PDE_ASSIGNED_AGENT_GPU_TOKEN", "")
assigned_slot = int(os.environ["PDE_ASSIGNED_AGENT_GPU_SLOT"])
started_ns = time.monotonic_ns()
cohort_environment = os.environ.get("PLASMA_COHORT_STORAGE_ROOT")
(request_path.parent / "cohort-storage-env.txt").write_text(
    "<absent>" if cohort_environment is None else cohort_environment,
    encoding="utf-8",
)

guidance_capability = request.get("guidance_oracle")
if guidance_capability is not None:
    guidance = ReferenceGuidanceOracleProxy.from_worker_capability(
        guidance_capability,
        worker_root=request_path.parent,
    )
    if guidance is None:
        raise RuntimeError("worker guidance proxy was not constructed")
    guided_batch = guidance.sample(
        GuidedSampleSpec(1, {"focus": "process-worker"})
    )
    guided_solution = guidance.query_completed_solution(
        guided_batch.samples[0].sample_id,
        "query_values",
        np.asarray([3.0, 4.0], dtype=np.float64),
        scale=2.0,
    )
    (request_path.parent / "guidance-probe.json").write_text(
        json.dumps(
            {
                "sample_id": guided_batch.samples[0].sample_id,
                "case_id": guided_batch.samples[0].case.case_id,
                "public_keys": sorted(
                    guided_batch.samples[0].case.public_data
                ),
                "public_array": guided_batch.samples[0].case.public_data[
                    "public_array"
                ].tolist(),
                "solution": guided_solution["values"].tolist(),
                "capability_keys": sorted(guidance_capability),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

# A file barrier makes process overlap a property of the fixture instead of a
# scheduler-timing assumption. Both workers must be alive before either exits.
round_root = request_path.parent.parent
(round_root / f"{agent_id}.worker-ready").write_text("ready", encoding="utf-8")
deadline = time.monotonic() + 5.0
expected_workers = request["campaign"]["agent_count"]
while len(tuple(round_root.glob("*.worker-ready"))) < expected_workers:
    if time.monotonic() >= deadline:
        raise RuntimeError("fake worker barrier timed out")
    time.sleep(0.005)

# The first configured GPU deliberately finishes last. The controller must
# still return results in configured Agent order.
time.sleep(0.20 if token.endswith("-a") else 0.01)
failure_mode = os.environ.get("FAKE_PROCESS_WORKER_FAILURE_MODE", "")
failure_token = os.environ.get("FAKE_PROCESS_WORKER_FAILURE_TOKEN", "")
if failure_mode == "nonzero" and token == failure_token:
    raise SystemExit(9)

source = (
    "def predict_macros(case, query, budget=None):\\n"
    "    return None\\n"
    f"# generated by {agent_id}\\n"
)
candidate_path = request_path.parent / "candidate.py"
candidate_path.write_text(source, encoding="utf-8")
bundle_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
finished_ns = time.monotonic_ns()
visible_devices = token
if failure_mode == "bad_gpu" and token == failure_token:
    visible_devices = f"invalid-{token}"

report = {
    "schema": "reference-process-agent-worker-report-v1",
    "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
    "status": "complete",
    "agent_id": agent_id,
    "round_index": round_index,
    "plugin_identity": {
        "plugin_id": "test",
        "generator_id": "test-generator",
        "generator_sha256": "f" * 64,
    },
    "gpu": {
        "pid": os.getpid(),
        "available": True,
        "device_count": 1,
        "current_device": 0,
        "visible_devices": visible_devices,
        "assigned_token": assigned_token,
        "assigned_slot": assigned_slot,
        "started_monotonic_ns": started_ns,
        "finished_monotonic_ns": finished_ns,
        "device_name": "fake-singleton-gpu",
        "device_uuid": f"UUID-{token}",
    },
    "development": {
        "portfolio": [
            {
                "candidate": {
                    "candidate_id": f"fake-{agent_id}-r{round_index}",
                    "bundle_sha256": bundle_sha256,
                    "round_index": round_index,
                    "agent_id": agent_id,
                    "parent_sha256": parent_sha256,
                    "bundle_uri": str(candidate_path.resolve()),
                },
                "roles": ["agent_final"],
            }
        ],
        "development_case_ids": [f"{agent_id}-development"],
        "sampled_case_ids": [f"{agent_id}-development"],
        "sampling_attempts": [],
        "search_completion": None,
        "failure": None,
    },
}
report_path.write_text(
    json.dumps(report, sort_keys=True, separators=(",", ":")) + "\\n",
    encoding="utf-8",
)
""",
        encoding="utf-8",
    )
    return worker


def test_process_executor_overlaps_workers_orders_results_and_reattests(
    tmp_path: Path,
):
    worker = _write_fake_process_worker(tmp_path)
    developer, agents, cold = _process_developer(
        tmp_path,
        command=(sys.executable, str(worker)),
    )

    results = developer.run_agents(
        round_index=1,
        agents=agents,
        incumbent=cold,
        prior_handoff=None,
        prior_leaderboard=(),
        shared_candidates=(),
    )

    assert tuple(result.agent_id for result in results) == tuple(
        agent.agent_id for agent in agents
    )
    assert all(len(result.portfolio) == 1 for result in results)
    assert [result.portfolio[0][0].ref.agent_id for result in results] == [
        agent.agent_id for agent in agents
    ]

    audit = developer.audit_payload()["rounds"][0]
    workers = audit["workers"]
    assert audit["same_round_barrier"] is True
    assert audit["overlap_verified"] is True
    assert [row["status"] for row in workers] == ["complete", "complete"]
    assert len({row["pid"] for row in workers}) == 2
    assert [row["visible_devices"] for row in workers] == [
        "GPU-worker-a",
        "GPU-worker-b",
    ]
    assert all(row["cuda_device_count"] == 1 for row in workers)
    assert all(row["cuda_current_device"] == 0 for row in workers)
    assert (
        workers[1]["worker_finished_monotonic_ns"]
        < workers[0]["worker_finished_monotonic_ns"]
    )

    central_root = (tmp_path / "central-bundles").resolve()
    sealed_paths = [
        Path(result.portfolio[0][0].ref.bundle_uri).resolve()
        for result in results
    ]
    assert all(path.parent == central_root for path in sealed_paths)
    worker_paths = tuple(
        (tmp_path / "controller" / "agent_workers").rglob("candidate.py")
    )
    assert len(worker_paths) == 2
    assert set(sealed_paths).isdisjoint(path.resolve() for path in worker_paths)
    for result, path in zip(results, sealed_paths, strict=True):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == (
            result.portfolio[0][0].ref.bundle_sha256
        )


@pytest.mark.parametrize("explicit_store", (False, True))
def test_process_worker_cohort_storage_environment_is_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    explicit_store: bool,
):
    inherited = tmp_path / "inherited-legacy-cohorts"
    monkeypatch.setenv("PLASMA_COHORT_STORAGE_ROOT", str(inherited))
    worker = _write_fake_process_worker(tmp_path)
    configured = tmp_path / "offline-cohorts" if explicit_store else None
    developer, agents, cold = _process_developer(
        tmp_path,
        command=(sys.executable, str(worker)),
        cohort_store_root=configured,
    )

    results = developer.run_agents(
        round_index=1,
        agents=agents,
        incumbent=cold,
        prior_handoff=None,
        prior_leaderboard=(),
        shared_candidates=(),
    )

    assert all(result.failure is None for result in results)
    round_root = tmp_path / "controller" / "agent_workers" / "round-0001"
    observed = tuple(
        path.read_text(encoding="utf-8")
        for path in sorted(round_root.glob("agent-*/cohort-storage-env.txt"))
    )
    if explicit_store:
        assert observed == tuple(
            str(
                developer.execution.node_local_root
                / "cohorts"
                / "round-0001"
                / f"agent-{slot:02d}"
            )
            for slot in range(2)
        )
    else:
        assert observed == ("<absent>", "<absent>")


def test_process_workers_receive_distinct_revocable_guidance_proxies(
    tmp_path: Path,
):
    worker = _write_fake_process_worker(tmp_path)
    developer, agents, cold = _process_developer(
        tmp_path,
        command=(sys.executable, str(worker)),
    )
    oracles = {
        agent.agent_id: _ProcessGuidanceOracle(agent.agent_id)
        for agent in agents
    }

    results = developer.run_agents(
        round_index=1,
        agents=agents,
        incumbent=cold,
        prior_handoff=None,
        prior_leaderboard=(),
        shared_candidates=(),
        guidance_oracles=oracles,
    )

    assert tuple(result.agent_id for result in results) == tuple(
        agent.agent_id for agent in agents
    )
    assert all(oracle.sample_calls == 1 for oracle in oracles.values())
    assert all(oracle.solution_calls == 1 for oracle in oracles.values())
    assert all(oracle.closed for oracle in oracles.values())
    assert all(oracle.close_calls == 1 for oracle in oracles.values())
    round_root = (
        tmp_path
        / "controller"
        / "agent_workers"
        / "round-0001"
    )
    request_paths = sorted(round_root.glob("agent-*/request.json"))
    probe_paths = sorted(round_root.glob("agent-*/guidance-probe.json"))
    assert len(request_paths) == len(probe_paths) == 2
    capabilities = []
    for request_path, probe_path in zip(
        request_paths, probe_paths, strict=True
    ):
        request = json.loads(request_path.read_text(encoding="utf-8"))
        capability = request["guidance_oracle"]
        capabilities.append(capability)
        capability_text = json.dumps(capability, sort_keys=True)
        assert set(capability) == {
            "schema",
            "endpoint",
            "token",
            "snapshot_sha256",
            "campaign_guidance_training_wall_seconds",
            "development_solution_methods",
        }
        assert "path" not in capability
        assert "locator" not in capability_text
        assert "resolver" not in capability_text
        assert "controller_training_seed" not in capability_text
        assert "guided_artifact" not in capability_text
        probe = json.loads(probe_path.read_text(encoding="utf-8"))
        assert probe["public_array"] == [0.0, 1.0]
        assert "distribution_lineage" not in probe["public_keys"]
        assert probe["solution"] == [6.0, 8.0]
        assert probe["capability_keys"] == sorted(capability)
        arrays = tuple(
            (request_path.parent / "guidance_transport" / "arrays").glob(
                "*.npy"
            )
        )
        assert arrays
        proxy = ReferenceGuidanceOracleProxy.from_worker_capability(
            capability,
            worker_root=request_path.parent,
        )
        assert proxy is not None
        with pytest.raises(ReferenceEvaluatorClientError):
            proxy.sample(GuidedSampleSpec(1))
    assert len({row["endpoint"] for row in capabilities}) == 2
    assert len({row["token"] for row in capabilities}) == 2
    audit = developer.audit_payload()["rounds"][0]
    assert all(
        row["guidance_oracle_hosted"]
        for row in audit["workers"]
    )


def test_worker_failure_closes_guidance_oracles_and_revokes_capabilities(
    tmp_path: Path,
    monkeypatch,
):
    worker = _write_fake_process_worker(tmp_path)
    monkeypatch.setenv("FAKE_PROCESS_WORKER_FAILURE_MODE", "nonzero")
    monkeypatch.setenv(
        "FAKE_PROCESS_WORKER_FAILURE_TOKEN", "GPU-worker-b"
    )
    developer, agents, cold = _process_developer(
        tmp_path,
        command=(sys.executable, str(worker)),
    )
    oracles = {
        agent.agent_id: _ProcessGuidanceOracle(agent.agent_id)
        for agent in agents
    }

    results = developer.run_agents(
        round_index=1,
        agents=agents,
        incumbent=cold,
        prior_handoff=None,
        prior_leaderboard=(),
        shared_candidates=(),
        guidance_oracles=oracles,
    )

    assert any(result.failure is not None for result in results)
    assert all(oracle.closed for oracle in oracles.values())
    assert all(oracle.close_calls == 1 for oracle in oracles.values())
    request_paths = tuple(
        (
            tmp_path
            / "controller"
            / "agent_workers"
            / "round-0001"
        ).glob("agent-*/request.json")
    )
    assert len(request_paths) == 2
    for request_path in request_paths:
        capability = json.loads(
            request_path.read_text(encoding="utf-8")
        )["guidance_oracle"]
        proxy = ReferenceGuidanceOracleProxy.from_worker_capability(
            capability,
            worker_root=request_path.parent,
        )
        assert proxy is not None
        with pytest.raises(ReferenceEvaluatorClientError):
            proxy.sample(GuidedSampleSpec(1))


@pytest.mark.parametrize(
    ("failure_mode", "expected_status", "expected_failure"),
    (
        ("nonzero", "nonzero_exit", "worker_failed:NonzeroExit"),
        (
            "bad_gpu",
            "invalid_report",
            "worker_report_failed:ReferenceIterationError",
        ),
    ),
)
def test_process_executor_preserves_success_when_peer_worker_fails(
    tmp_path: Path,
    monkeypatch,
    failure_mode,
    expected_status,
    expected_failure,
):
    worker = _write_fake_process_worker(tmp_path)
    monkeypatch.setenv("FAKE_PROCESS_WORKER_FAILURE_MODE", failure_mode)
    monkeypatch.setenv("FAKE_PROCESS_WORKER_FAILURE_TOKEN", "GPU-worker-b")
    developer, agents, cold = _process_developer(
        tmp_path,
        command=(sys.executable, str(worker)),
    )

    results = developer.run_agents(
        round_index=1,
        agents=agents,
        incumbent=cold,
        prior_handoff=None,
        prior_leaderboard=(),
        shared_candidates=(),
    )

    assert tuple(result.agent_id for result in results) == tuple(
        agent.agent_id for agent in agents
    )
    assert len(results[0].portfolio) == 1
    assert results[0].failure is None
    assert results[1].portfolio == ()
    assert results[1].failure == expected_failure
    successful_uri = Path(results[0].portfolio[0][0].ref.bundle_uri).resolve()
    assert successful_uri.parent == (tmp_path / "central-bundles").resolve()

    audit = developer.audit_payload()["rounds"][0]
    assert [row["status"] for row in audit["workers"]] == [
        "complete",
        expected_status,
    ]
    assert len({row["pid"] for row in audit["workers"]}) == 2
    assert [row["gpu_device"] for row in audit["workers"]] == [
        "GPU-worker-a",
        "GPU-worker-b",
    ]
