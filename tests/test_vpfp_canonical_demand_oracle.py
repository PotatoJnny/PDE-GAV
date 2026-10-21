"""Real VPFP jets through canonical RPC and a two-round local Campaign.

The decoder/carriers are tiny synthetic fixtures, not learned assets. Neither
the source kernel nor the numerical session is replaced with a mock. Spies
forward into the pinned physical functions. The Agent is a local diagnostic,
not an LLM, and finite macro scores make no solver-quality claim.
The candidate exposes only prepare_solution and a POINTS evaluate handle, so
these tests cannot pass by falling back to predict_observables. This remains a
trusted local interface check, not OS isolation or a hard-budget guarantee.
"""

from contextlib import contextmanager
from dataclasses import replace
import importlib
import json
from pathlib import Path
from threading import RLock
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from pde_agent_verifier_solver.app.backend_run import BackendRunBinding
from pde_agent_verifier_solver.engine.agent_capability import (
    AgentCapabilityClientError, ControllerAgentCapabilityClient,
    ControllerAgentCapabilityFacade,
)
from pde_agent_verifier_solver.engine.agent_execution import CommandAgentProgram
from pde_agent_verifier_solver.engine.campaign import (
    AgentSpec, CampaignRecipe, CaseCounts, DevelopmentSamplingPolicy,
    EvaluatorPolicy, GeneratorRecipe, RoundSettings,
)
from pde_agent_verifier_solver.engine.evaluator import (
    AgentEvaluator, EvaluationRequest, HiddenVerifier,
)
from pde_agent_verifier_solver.engine.runtime import AgentOutcome
from pde_agent_verifier_solver.plugins.vpfp import (
    VPFPAnchoredPerturbationBackend, VPFPContinuousGenerator,
    VPFPLegacyStructuredRuntime, VPFPObservableTask,
    VPFP_DEVELOPMENT_SOLUTION_SESSION_METHODS, VPFP_PUBLIC_SESSION_METHODS,
    vpfp_public_problem_bundle,
)
from pde_agent_verifier_solver.plugins.vpfp.canonical_binding import (
    _BoundVPFPContinuousGenerator,
)
from pde_agent_verifier_solver.plugins.vpfp.macro_matching import VPFP_MACRO_FIELDS
from pde_agent_verifier_solver.plugins.vpfp.perturbation_backend import VPFPRuntimeCase
from pde_agent_verifier_solver.plugins.vpfp.product import build_vpfp_plugin
from pde_agent_verifier_solver.sdk import (
    AnswerQuery, AnswerView, Artifact, CaseVisibility, DatasetInput, DatasetKind,
    EvaluationPhase, GeneratorCheckpoint, GeneratorState, HiddenSuiteSpec,
    OfficialEvaluationSpec, PreparedDataset, QueryKind, SampleSource,
    SamplingStrategy, TaskKind, open_development_solution_session,
    open_public_case_session,
)
from tests.test_canonical_public_problem_capability import _context
from tests.test_vpfp_batched_source import decoder, _queries
from tests.test_vpfp_derivative_features import tiny


_MACRO_QUERY = AnswerQuery.at_points(
    fields=VPFP_MACRO_FIELDS,
    points=({"t": .253, "x": .137, "y": .419},
            {"t": .719, "x": .631, "y": .827}),
)


class _TinyResidentRuntime(VPFPLegacyStructuredRuntime):
    """Only sampling is synthetic; inherited session/source/truth are real."""

    def __init__(self, decoder, root):
        operators = importlib.import_module("agent_verifier.vpfp.operators")
        self.supports_perturbations = True
        self.materialize_cases = False
        self.public_session_methods = VPFP_PUBLIC_SESSION_METHODS
        self.development_solution_session_methods = VPFP_DEVELOPMENT_SOLUTION_SESSION_METHODS
        self.output_root = Path(root)
        self.output_root.mkdir()
        self._runtime_lock = RLock()
        self._resident_cases = {}
        self.chunk_size = 32
        self.expanded_model = decoder.model
        self.api = decoder.api
        self.grid = operators.VPFPGrid(nx=4, ny=4, nvx=3, nvy=3)
        self.time = operators.VPFPTime(t0=0., t1=1., nt=3)
        self.latents = decoder.latents.copy()
        self.conditions = decoder.conditions.copy()
        self.issued = []
        self.released = []
        self.sessions = []
        self.method_calls = {name: [] for name in ("source", "rho0", "g0")}

    def record(self, case_id, *, visibility=CaseVisibility.DEVELOPMENT,
               round_index=1, cohort="development", seed=1):
        record = self._resident_record(
            case_id=case_id, latent=self.latents[int(seed) % 2],
            condition=self.conditions[int(seed) % 2], visibility=visibility,
            round_index=round_index, cohort=cohort, parent_case_id=None,
            variant="base",
        )
        self.issued.append(record.case.case_id)
        return record

    def generate_base(self, *, seeds, case_ids, visibility, round_index, cohort):
        return tuple(VPFPRuntimeCase(
            self.record(case_id, visibility=visibility, round_index=round_index,
                        cohort=cohort, seed=seed),
            private_handle={"synthetic_carrier_seed": int(seed)},
        ) for seed, case_id in zip(seeds, case_ids))

    def public_session_factory(self, case):
        session = super().public_session_factory(case)
        self.sessions.append(session)
        return session

    def evaluate_source(self, case_id, points):
        self.method_calls["source"].append((case_id, np.asarray(points).copy()))
        return super().evaluate_source(case_id, points)

    def evaluate_rho0(self, case_id, points):
        self.method_calls["rho0"].append((case_id, np.asarray(points).copy()))
        return super().evaluate_rho0(case_id, points)

    def evaluate_g0(self, case_id, points):
        self.method_calls["g0"].append((case_id, np.asarray(points).copy()))
        return super().evaluate_g0(case_id, points)

    def release_cases(self, case_ids):
        self.released.extend(case_ids)
        return super().release_cases(case_ids)


@pytest.fixture
def physical(decoder, tmp_path):
    runtime = _TinyResidentRuntime(decoder, tmp_path / "physical")
    yield SimpleNamespace(decoder=decoder, runtime=runtime)
    runtime.release_cases(tuple(runtime._resident_cases))


def _generator(runtime, *, bound=False):
    backend = VPFPAnchoredPerturbationBackend(runtime)
    if bound:
        identity = {"binding_sha256": "a"*64,
                    "data_origin": "synthetic_tiny_continuous_carrier"}
        generator = _BoundVPFPContinuousGenerator(backend, identity)
        checkpoint = GeneratorCheckpoint(generator.info.program_id, identity)
    else:
        generator = VPFPContinuousGenerator(backend)
        checkpoint = GeneratorCheckpoint(generator.info.program_id, object())
    return generator, checkpoint


def _evaluator(physical, record):
    generator, checkpoint = _generator(physical.runtime)
    evaluator = AgentEvaluator(
        (record,), VPFPObservableTask(), EvaluatorPolicy(_MACRO_QUERY),
        generator=generator,
        generator_state=GeneratorState(checkpoint=checkpoint, base=checkpoint.payload),
    )
    return evaluator


@contextmanager
def _rpc(context, workspace):
    workspace.mkdir()
    lease = ControllerAgentCapabilityFacade().open(
        context, workspace, "real-vpfp-numerical-rpc",
        CommandAgentProgram((sys.executable, "-c", "pass"), lambda *_: None),
    )
    client = ControllerAgentCapabilityClient(lease.manifest, workspace=workspace)
    try:
        yield client, lease
    finally:
        lease.close()


def _assert_public_only(case):
    assert case.parent_case_id is None and case.tags == ()
    arrays = case.public_data["arrays"]
    assert "source" not in arrays and "g0" not in arrays
    # Small rho0/J0/K0 are public previews, not a prescribed source-query grid.
    assert arrays["rho0"].shape == (4, 4)
    text = repr(case.public_data)
    for forbidden in ("private_handle", "synthetic_carrier_seed", "latent",
                      "private-answer.pt", "checkpoint.pt"):
        assert forbidden not in text


def _assert_closed(sessions):
    for session in sessions:
        with pytest.raises(RuntimeError, match="closed"):
            session.source(np.zeros((1, 5)))


def test_canonical_rpc_zero_requests_do_not_evaluate_source_or_g0(physical, tmp_path):
    record = physical.runtime.record("private-zero-request-root")
    evaluator = _evaluator(physical, record)
    context = _context(evaluator, vpfp_public_problem_bundle())
    with _rpc(context, tmp_path / "zero-rpc") as (client, lease):
        cases = client.current_cases()
        assert len(cases) == 1
        _assert_public_only(cases[0])
        assert cases[0].case_id != record.case.case_id
        assert record.case.case_id not in json.dumps(dict(lease.manifest))
        assert not physical.runtime.sessions
        assert all(not calls for calls in physical.runtime.method_calls.values())
    with pytest.raises(AgentCapabilityClientError):
        client.invoke_public_problem(cases[0].case_id, "source", np.zeros((1, 5)))
    assert all(not calls for calls in physical.runtime.method_calls.values())


def test_canonical_rpc_hits_real_batched_kernel_for_two_arbitrary_sparse_requests(
    physical, tmp_path, monkeypatch,
):
    runtime, numerical = physical.runtime, physical.decoder
    record = runtime.record("private-real-source-root")
    evaluator = _evaluator(physical, record)
    context = _context(evaluator, vpfp_public_problem_bundle())
    all_points = _queries("sparse_shuffled_duplicates")
    batches = (all_points[:7], all_points[7:])
    expected = [runtime.evaluate_source_reference(record.case.case_id, points) for points in batches]
    analytic = importlib.import_module("agent_verifier.vpfp.analytic_source")
    original_macro = analytic.analytic_structured_macro_jet
    original_basis = analytic.hermite_basis_coordinate_jet
    original_assemble = analytic._assemble_pointwise
    counts = {"macro": [], "velocity": [], "assembled": []}

    def macro(model, points, latent):
        counts["macro"].append(len(points))
        return original_macro(model, points, latent)

    def basis(torch, points, **kwargs):
        counts["velocity"].append(len(points))
        return original_basis(torch, points, **kwargs)

    def assemble(torch, **kwargs):
        counts["assembled"].append(len(kwargs["velocity"]))
        return original_assemble(torch, **kwargs)

    monkeypatch.setattr(analytic, "analytic_structured_macro_jet", macro)
    monkeypatch.setattr(analytic, "hermite_basis_coordinate_jet", basis)
    monkeypatch.setattr(analytic, "_assemble_pointwise", assemble)
    with _rpc(context, tmp_path / "real-rpc") as (client, lease):
        public = client.current_cases()[0]
        _assert_public_only(public)
        for points, target in zip(batches, expected):
            result = client.invoke_public_problem(public.case_id, "source", points)
            assert result.shape == (len(points),)
            np.testing.assert_allclose(result, target, rtol=numerical.rtol, atol=numerical.atol)
        assert lease.controller_service_seconds > 0.
    assert len(runtime.method_calls["source"]) == 2
    for (_, observed), points in zip(runtime.method_calls["source"], batches):
        np.testing.assert_array_equal(observed, points)
    assert not runtime.method_calls["rho0"] and not runtime.method_calls["g0"]
    assert sum(counts["assembled"]) == len(all_points) == 13
    assert sum(counts["macro"]) == sum(len(np.unique(p[:, :3], axis=0)) for p in batches)
    assert sum(counts["velocity"]) == sum(len(np.unique(p[:, 3:], axis=0)) for p in batches)
    assert sum(counts["assembled"]) < sum(
        len(np.unique(p[:, :3], axis=0))*len(np.unique(p[:, 3:], axis=0)) for p in batches)
    _assert_closed(runtime.sessions)
    assert sum(session.usage_snapshot["evaluated_points"] for session in runtime.sessions) == 13
    with pytest.raises(AgentCapabilityClientError):
        client.invoke_public_problem(public.case_id, "source", all_points[:1])
    assert len(runtime.method_calls["source"]) == 2


def test_canonical_rpc_bad_query_is_counted_without_completed_points_or_truth(
    physical, tmp_path,
):
    record = physical.runtime.record("private-bad-source-root")
    evaluator = _evaluator(physical, record)
    context = _context(evaluator, vpfp_public_problem_bundle())
    with _rpc(context, tmp_path / "bad-rpc") as (client, _lease):
        opaque_id = client.current_cases()[0].case_id
        with pytest.raises(AgentCapabilityClientError) as invalid:
            client.invoke_public_problem(opaque_id, "source", np.ones((3, 4)))
        assert invalid.value.code == "operation_failed"
        assert record.case.case_id not in str(invalid.value)
        for method in ("released_macros", "f", "evaluate_source_reference"):
            with pytest.raises(AgentCapabilityClientError) as denied:
                client.invoke_public_problem(opaque_id, method, np.zeros((1, 5)))
            assert denied.value.code == "capability_unavailable"
        with pytest.raises(AgentCapabilityClientError) as denied:
            client.invoke_development_solution(opaque_id, "released_macros", np.zeros((1, 3)))
        assert denied.value.code == "capability_unavailable"
        with pytest.raises(AgentCapabilityClientError):
            client.invoke_public_problem(record.case.case_id, "source", np.zeros((1, 5)))
    assert len(physical.runtime.sessions) == 1
    usage = physical.runtime.sessions[0].usage_snapshot
    assert usage["attempted_calls"] == usage["failed_calls"] == 1
    assert usage["requested_points"] == 3
    assert usage["completed_points"] == usage["evaluated_points"] == usage["query_calls"] == 0
    _assert_closed(physical.runtime.sessions)


class _PreparedZeroMacros:
    def __init__(self, owner):
        self.owner = owner

    def evaluate(self, query):
        assert query.kind is QueryKind.POINTS
        assert set(query.fields) == set(VPFP_MACRO_FIELDS)
        self.owner.evaluation_calls += 1
        # No Oracle access occurs at these private scoring coordinates.
        return AnswerView(
            kind=query.kind, points=query.points, shape=(len(query.points),),
            fields={name: (0.,)*len(query.points) for name in query.fields},
        )


class _MacroOnlyCandidate:
    def __init__(self):
        self.public_queries = 0
        self.development_denials = 0
        self.prepare_calls = 0
        self.evaluation_calls = 0

    def prepare_solution(self, case, solve_budget=None):
        _assert_public_only(case)
        session = open_public_case_session(case)
        assert session.source(_queries("sparse")[:3]).shape == (3,)
        self.public_queries += 1
        self.prepare_calls += 1
        try:
            open_development_solution_session(case)
        except RuntimeError:
            self.development_denials += 1
        else:
            raise AssertionError("completed solution was exposed during macro-only evaluation")
        return _PreparedZeroMacros(self)

def _artifact(program, *, round_index=1, agent_id="diagnostic-agent"):
    return Artifact(
        f"macro-only-r{round_index}", TaskKind.OBSERVABLE_MODEL,
        "Local macro-only diagnostic", program, "Synthetic interface check",
        "prepare_solution(case)->handle.evaluate(POINTS)", round_index, agent_id,
        metadata={"full_solution_prediction_required": False,
                  "scientific_qualification": False},
    )


@pytest.mark.parametrize("phase", (EvaluationPhase.OFFICIAL, EvaluationPhase.TERMINAL))
def test_real_hidden_macro_evaluation_keeps_public_source_but_denies_completed_solution(
    physical, phase,
):
    record = physical.runtime.record("private-hidden-root", visibility=CaseVisibility.HIDDEN)
    generator, checkpoint = _generator(physical.runtime)
    candidate = _MacroOnlyCandidate()
    verifier = HiddenVerifier(
        VPFPObservableTask(), _MACRO_QUERY, phase=phase,
        generator=generator,
        generator_state=GeneratorState(checkpoint, checkpoint.payload),
    )
    result = verifier.evaluate(_artifact(candidate), (record,))
    assert np.isfinite(result.primary_error)
    assert candidate.public_queries == candidate.development_denials == 1
    assert candidate.prepare_calls == candidate.evaluation_calls == 1
    assert not hasattr(candidate, "predict_solution")
    assert not hasattr(candidate, "predict_observables")
    assert len(physical.runtime.method_calls["source"]) == 1
    _assert_closed(physical.runtime.sessions)
    with pytest.raises(PermissionError):
        physical.runtime.development_solution_session_factory(record.case)


class _CanonicalDiagnosticAgent:
    def __init__(self, root):
        self.root = root
        self.rounds = []
        self.clients = []
        self.programs = []

    def run(self, context):
        round_index = context.round_plan.round_index
        with _rpc(context, self.root / f"round-{round_index}") as (client, _lease):
            cases = client.current_cases()
            assert len(cases) == 1
            case = cases[0]
            _assert_public_only(case)
            first = _queries("sparse_shuffled_duplicates")[:7]
            values = client.invoke_public_problem(case.case_id, "source", first)
            assert values.shape == (7,) and np.isfinite(values).all()
            # The second numerical request depends on a returned public value.
            second = _queries("sparse")[4:7].copy()
            second[:, 0] = .2 + .1*np.tanh(abs(float(values[0])))
            assert np.isfinite(client.invoke_public_problem(case.case_id, "source", second)).all()
            with pytest.raises(AgentCapabilityClientError):
                client.invoke_development_solution(case.case_id, "released_macros", np.zeros((1, 3)))
            self.clients.append((client, case.case_id))
            # Revoking the formal capability now also cancels pending/new
            # evaluator work. Finish this trusted diagnostic's Candidate call
            # during its active lease; never disable cancellation to keep an
            # expired Agent context alive.
            program = _MacroOnlyCandidate()
            artifact = _artifact(program, round_index=round_index, agent_id=context.agent.agent_id)
            response = context.evaluator.evaluate(
                artifact, EvaluationRequest(tuple(case.case_id for case in context.evaluator.cases)),
            )
            assert np.isfinite(response.primary_error)
        self.rounds.append(round_index)
        self.programs.append(program)
        return AgentOutcome(artifact, observations=(
            "Local real-Oracle macro-only interface diagnostic; no learning claim.",
        ))


def test_two_round_canonical_backend_run_uses_real_oracles_macros_and_releases_cases(
    physical, tmp_path,
):
    runtime = physical.runtime
    generator, checkpoint = _generator(runtime, bound=True)
    binding = BackendRunBinding(
        plugin=replace(build_vpfp_plugin(), generator=generator), checkpoint=checkpoint,
        dataset=PreparedDataset(DatasetInput("tiny-canonical-vpfp", DatasetKind.NUMERICAL), (), ()),
        report={"data_origin": "synthetic_tiny_continuous_carrier",
                "trained_assets": False, "scientific_qualification": False},
    )
    agent = _CanonicalDiagnosticAgent(tmp_path)
    recipe = CampaignRecipe(
        name="two-round real VPFP Oracle interface diagnostic", round_count=2,
        agents=(AgentSpec("diagnostic-agent", "Trusted local diagnostic Agent"),),
        default_round=RoundSettings(
            agent_budget_seconds=30,
            development_sampling=DevelopmentSamplingPolicy(
                strategy=SamplingStrategy.AGENT_ADAPTIVE,
                allowed_sources=(SampleSource.BASE,), max_cases_per_agent_per_round=1,
                starter=CaseCounts(base=1),
            ),
            official_evaluation=OfficialEvaluationSpec(
                round=HiddenSuiteSpec(base=1), terminal=HiddenSuiteSpec(base=1),
            ),
        ),
        generator=GeneratorRecipe(guidance_enabled=False, perturbations_enabled=False),
        evaluator=EvaluatorPolicy(_MACRO_QUERY, development_point_count=1),
        task_settings={"official_point_count": 1}, master_seed=20260908,
    )
    result = binding.run(recipe, {"diagnostic-agent": agent}, local_development=True)
    assert len(result.rounds) == len(result.terminal_round_reports) == 2
    assert agent.rounds == [1, 2]
    assert np.isfinite(result.selected.primary_error)
    assert all(program.public_queries > 0 for program in agent.programs)
    assert all(program.public_queries == program.development_denials for program in agent.programs)
    assert all(program.prepare_calls == program.evaluation_calls == program.public_queries
               for program in agent.programs)
    assert not runtime._resident_cases
    assert set(runtime.issued) == set(runtime.released)
    assert len(runtime.issued) >= 5  # Two development, two Official, one Terminal.
    assert len(runtime.method_calls["source"]) >= 6
    assert not runtime.method_calls["rho0"] and not runtime.method_calls["g0"]
    assert all(len(points) in (3, 7) for _, points in runtime.method_calls["source"])
    _assert_closed(runtime.sessions)
    for client, case_id in agent.clients:
        with pytest.raises(AgentCapabilityClientError):
            client.invoke_public_problem(case_id, "source", np.zeros((1, 5)))
