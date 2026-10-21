"""Canonical Artemis file/RPC evidence using small, non-physical fixtures."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from pde_agent_verifier_solver.engine.agent_execution import CommandAgentProgram, SubprocessAgentExecutionService
from pde_agent_verifier_solver.engine.campaign import (
    AgentSpec, CampaignRecipe, CaseCounts, DevelopmentSamplingPolicy,
    EvaluatorPolicy, GeneratorRecipe, RoundSettings,
)
from pde_agent_verifier_solver.engine.python_candidate import PythonCandidateCodec, PythonCandidateProgram
from pde_agent_verifier_solver.sdk import (
    AnswerQuery, Case, CaseSource, CaseVisibility, EvaluationPhase,
    HiddenSuiteSpec, OfficialEvaluationSpec, SampleSource, TaskKind,
)
from pde_agent_verifier_solver.sdk.public_sessions import bind_public_session_factory

from tests.test_artemis_canonical import _binding, _query, _request


_ROOT = Path(__file__).resolve().parents[1]
_S0 = _ROOT / "configs/candidates/artemis_hold_initial_bundle"


def test_artemis_agent_brief_and_public_resource_require_formal_two_phase_abi():
    binding, _ = _binding()
    brief = binding.plugin.task().agent_brief({"require_prepared_solution": True})
    assert "export prepare_solution" in brief
    assert "not a file-submission entrypoint" in brief
    readme = next(resource.data.decode() for resource in binding.plugin.public_problem_bundle.resources
                  if resource.relative_path.endswith("README.md"))
    assert "file-submitted Candidates must export" in readme
    assert "require_prepared_solution: true" in readme
    assert "Development disclosure policy" in readme
    assert "allow_development_solution_session" in readme


def test_artemis_terminal_point_count_is_distinct_and_validated():
    binding, _ = _binding()
    generator = binding.plugin.generator
    state = generator.load(binding.checkpoint, {})
    case = generator.sample(state, _request())[0].case
    evaluation = binding.plugin.task().create_evaluation({"default_query": _query()})
    settings = {"official_point_count": 2, "terminal_point_count": 3}
    assert len(evaluation.make_plan(case, EvaluationPhase.OFFICIAL, settings, 17).query.points) == 2
    assert len(evaluation.make_plan(case, EvaluationPhase.TERMINAL, settings, 17).query.points) == 3
    assert len(evaluation.make_plan(case, EvaluationPhase.TERMINAL, {"official_point_count": 2}, 17).query.points) == 2
    for invalid in (0, True, 5, 1.5):
        with pytest.raises(ValueError, match="native nx"):
            evaluation.make_plan(case, EvaluationPhase.TERMINAL, {"terminal_point_count": invalid}, 17)


def test_real_s0_bundle_is_query_blind_oracle_only_reloadable_and_macro_only(tmp_path):
    codec = PythonCandidateCodec(task_kind=TaskKind.OBSERVABLE_MODEL)
    artifact = codec.bind_existing(_S0, storage_root=tmp_path / "sealed", artifact_id="s0",
        name="Hold initial", summary="weak public-data baseline", usage="prepare then evaluate")
    assert isinstance(artifact.program, PythonCandidateProgram)
    x = np.arange(32) * (2 * np.pi / 32)
    case = Case("public", CaseSource.GENERATED, CaseVisibility.DEVELOPMENT, {
        "arrays": {"x": x, "f0": np.full((32, 4), np.nan)},
        "equation": {
            "domain": {"x_interval": (0., 2 * np.pi), "time_interval": (0., .5),
                       "velocity_query_window": (-8., 8.)},
            "oracle_protocol": {"limits": {"max_points_per_request": 73}},
        },
    })
    calls, sessions = [], []
    class Session:
        def query_initial_distribution(self, points):
            assert len(points) <= 73
            calls.append(np.array(points, copy=True))
            rho = 1 + .1 * np.cos(points[:, 0])
            return rho * np.exp(-.5 * (points[:, 1] - .2)**2) / np.sqrt(2 * np.pi)
        def close(self):
            sessions.append("closed")
    with bind_public_session_factory(lambda _: Session(), case,
                                     allowed_methods=("query_initial_distribution",)):
        handle = deepcopy(artifact.program).prepare_solution(case, solve_budget=10)
    assert sessions == ["closed"]
    count_at_prepare = len(calls)
    # Reversed native rows and different horizons show preparation sees no query.
    query = AnswerQuery.at_points(fields=("M0", "M1", "M2", "M3", "E"), points=(
        {"t": .5, "x": x[8]}, {"t": .25, "x": x[0]}, {"t": .5, "x": x[8]},
    ))
    answer = handle.evaluate(query)
    assert len(calls) == count_at_prepare
    assert answer.points == query.points and answer.shape == (3,)
    rho = np.array([1., 1.1, 1.])
    expected = {"M0": rho, "M1": .2 * rho, "M2": 1.04 * rho,
                "M3": .608 * rho, "E": np.array([.1, 0., .1])}
    assert set(answer.fields) == set(expected)
    for name, values in expected.items():
        np.testing.assert_allclose(answer.fields[name], values, rtol=1e-10, atol=1e-11)
    handle.close()
    with pytest.raises(RuntimeError, match="closed"):
        handle.evaluate(query)
    turn = tmp_path / "next-turn"
    turn.mkdir()
    staged = codec.encode_incumbent(artifact, turn)
    assert staged["tree_sha256"] == artifact.program.tree_sha256


def test_two_round_artemis_uses_canonical_subprocess_rpc_file_candidates_and_incumbent(tmp_path):
    binding, source = _binding()
    # Authored deterministic fixture Agent, not an LLM or physical-quality claim.
    candidate = '''from pde_agent_verifier_solver.sdk import AnswerView, open_public_case_session, open_development_solution_session
import numpy as np
FACTOR = __FACTOR__
class Handle:
    def __init__(self, horizon): self.horizon = horizon
    def evaluate(self, query):
        assert all(point['t'] == self.horizon for point in query.points)
        return AnswerView(kind=query.kind, points=query.points, shape=(len(query.points),), fields={name: tuple(FACTOR * (index + 1) * (1 + point['t'] + point['x']) for point in query.points) for index, name in enumerate(('M0','M1','M2','M3','E'))})
def prepare_solution(case, solve_budget=None):
    horizon = case.public_data['equation']['domain']['time_interval'][1]
    assert case.public_data['arrays']['t'][-1] == horizon
    session = open_public_case_session(case)
    np.testing.assert_allclose(session.query_source(np.array([[horizon,.17,.13]])), [horizon+.3])
    try: open_development_solution_session(case)
    except RuntimeError: pass
    else: raise AssertionError('Candidate must not receive completed answers')
    return Handle(horizon)
'''
    worker = tmp_path / "worker.py"
    worker.write_text('''from pathlib import Path
import json
import numpy as np
from pde_agent_verifier_solver.engine.agent_execution import load_agent_worker_request, submit_agent_nomination, write_agent_round_log, write_agent_worker_response
from pde_agent_verifier_solver.engine.agent_capability import ControllerAgentCapabilityClient, AgentCapabilityClientError
request = load_agent_worker_request()
round_index = request['round']['round_index']
public = request['public_problem']
readme = Path(public['resources_path']) / 'vp_1d1v_reference/README.md'
assert 'file-submitted Candidates must export' in readme.read_text()
assert not request['capabilities']['assurance']['hostile_candidate_sandbox']
client = ControllerAgentCapabilityClient.from_request(request)
cases = client.current_cases()
assert len(cases) == 1
case = cases[0]
np.testing.assert_allclose(client.invoke_public_problem(case.case_id, 'query_initial_distribution', np.array([[.25,0.]])), [1.])
assert not request['capabilities']['public_problem']['development_solution_oracle']['available']
try: client.invoke_development_solution(case.case_id, 'query_completed_macros', np.array([[.5,.25]]))
except AgentCapabilityClientError: pass
else: raise AssertionError('formal run must deny completed answers')
if round_index == 2:
    assert request['incumbent']['program_available']
    incumbent = request['incumbent']['program']
    assert Path(incumbent['bundle_path'], 'candidate_bundle.json').is_file()
    assert 'FACTOR = 0.9' in Path(incumbent['bundle_path'], incumbent['entrypoint']).read_text()
    assert request['campaign_memory']['completed_rounds'] == 1
candidate = request['worker_settings']['candidate'].replace('__FACTOR__', repr(1 - .1 / round_index))
Path('submission.py').write_text(candidate)
result = client.evaluate({'bundle_path':'submission.py'}, case_ids=(case.case_id,))
assert abs(result.primary_error - .1 / round_index) < 1e-9
assert len(result.cases) == 1
assert result.cases[0].case_id == case.case_id
assert result.cases[0].truth is None and result.cases[0].prediction is None
submit_agent_nomination({'bundle_path':'submission.py', 'name':f'Macros r{round_index}'}, sequence=1)
write_agent_round_log(f'# Round {round_index}\\n\\nFile Candidate checked with macro-only public protocol.')
write_agent_worker_response(observations=('Artemis file protocol exercised',))
''', encoding="utf-8")
    codec = PythonCandidateCodec(task_kind=TaskKind.OBSERVABLE_MODEL)
    program = CommandAgentProgram((sys.executable, str(worker)), codec.decode_nomination,
        incumbent_encoder=codec.encode_incumbent,
        environment={"PYTHONPATH": str(_ROOT / "src")}, worker_settings={"candidate": candidate})
    service = SubprocessAgentExecutionService(tmp_path / "turns", hard_wall_seconds=20)
    recipe = CampaignRecipe(name="Artemis formal protocol fixture", round_count=2,
        agents=(AgentSpec("agent", "Fixture Agent"),),
        default_round=RoundSettings(agent_budget_seconds=10,
            development_sampling=DevelopmentSamplingPolicy(max_cases_per_agent_per_round=1,
                allowed_sources=(SampleSource.BASE,), starter=CaseCounts(base=1)),
            official_evaluation=OfficialEvaluationSpec(round=HiddenSuiteSpec(base=1, base_perturbed=1),
                terminal=HiddenSuiteSpec(base=1, base_perturbed=1))),
        generator=GeneratorRecipe(guidance_enabled=False, perturbations_enabled=True),
        evaluator=EvaluatorPolicy(_query(), allow_grid_truth=False, allow_point_truth=False,
            allow_development_solution_session=False, development_point_count=2),
        task_settings={"require_prepared_solution": True, "official_point_count": 2, "terminal_point_count": 3})
    # This small fixture deliberately executes Candidates in process; the
    # existing formal boundary requires an explicit local-development opt-in.
    result = binding.run(recipe, {"agent": program}, agent_execution=service,
                         local_development=True)
    assert len(result.rounds) == len(result.terminal_round_reports) == 2
    assert result.selected.primary_error == pytest.approx(.05)
    assert result.selected.artifact.round_index == 2
    assert isinstance(result.selected.artifact.program, PythonCandidateProgram)
    assert len(service.audits) == 2
    for audit in service.audits:
        assert audit.status == "completed", Path(audit.stderr_path).read_text()
        assert audit.nomination_count == 1
        assert audit.capability_revoked and audit.capability_drained and audit.process_group_cleaned
        assert audit.agent_process_isolated and not audit.candidate_execution_isolated
        assert not audit.formal_private_scoring_claimed
        request = json.loads(Path(audit.request_path).read_text())
        assert request["capabilities"]["public_problem"]["equation_oracle"]["available"]
    assert source.sessions and all(session.closed for session in source.sessions)
    assert not source.records
