"""Exercise staged NS guides with an ordinary worker, no LLM or GPU required."""
from dataclasses import replace
import json
from pathlib import Path
import re
import sys

from pde_agent_verifier_solver.engine import (
    CommandAgentProgram, PythonCandidateCodec, PythonTaskWorkerSpec,
    SubprocessAgentExecutionService, SubprocessCandidateExecutionService,
)
from pde_agent_verifier_solver.engine.campaign import CaseCounts
from pde_agent_verifier_solver.plugins.ns import NSSolverTask
from pde_agent_verifier_solver.plugins.ns.public_problem import ns_public_problem_bundle
from pde_agent_verifier_solver.sdk import (
    HiddenSuiteSpec, TaskKind,
)
from tests.test_ns_mixed_evaluation_preparation import cpu_field_factory, _controller


def test_ns_documented_inputs_output_bundle_and_commands_in_real_worker(tmp_path, cpu_field_factory):
    bundle = ns_public_problem_bundle()
    guide = next(row.data.decode() for row in bundle.resources
                 if row.relative_path == 'ns_reference/SOLVER_INTERFACE.md')
    adapter = next(block for block in re.findall(r'```python\n(.*?)```', guide, re.S)
                   if 'class PreparedVelocity:' in block)
    queries = next(block for block in re.findall(r'```python\n(.*?)```', guide, re.S)
                   if 'session = open_public_case_session(case)' in block)
    source = tmp_path / 'initial'
    source.mkdir()
    # A small constant-output fixture uses real public inputs and the documented
    # AnswerView adapter. This checks the interface, not numerical quality.
    (source / 'submission.py').write_text(adapter + '\n' + '''def prepare_solution(case, solve_budget=None):
    domain = case.public_data['domain']
    x = sum(domain['bounds']['x']) / 2
    y = sum(domain['bounds']['y']) / 2
    xy = np.array([[x, y]], dtype=np.float64)
    txy = np.array([[domain['time_interval'][0], x, y]], dtype=np.float64)
''' + '\n'.join('    ' + line for line in queries.strip().splitlines()) + '''
    assert initial_velocity.shape == forcing.shape == (1, 2)
    return PreparedVelocity(lambda points: np.repeat(initial_velocity, len(points), axis=0))
''')
    manifest = json.loads(next(block for block in re.findall(r'```json\n(.*?)```', guide, re.S)))
    manifest['files'].remove('numerics.py')
    (source / 'candidate_bundle.json').write_text(json.dumps(manifest))
    codec = PythonCandidateCodec(task_kind=TaskKind.SOLVER)
    initial = codec.bind_existing(source, storage_root=tmp_path / 'sealed', artifact_id='guide-s0',
                                  name='Interface fixture', summary='Test fixture', usage='two phase')
    script = tmp_path / 'read_guide.py'
    script.write_text('''from pathlib import Path
import json, re, subprocess
from pde_agent_verifier_solver.engine.agent_execution import write_agent_worker_response
reference = Path('public_resources/ns_reference')
assert (reference / 'README.md').is_file()
assert (reference / 'AGENT_TOOLS.txt').is_file()
assert Path('CAMPAIGN_MEMORY.md').is_file()
assert Path('CANDIDATE_HISTORY.json').is_file()
guide = (reference / 'SOLVER_INTERFACE.md').read_text()
blocks = re.findall(r'```bash\\n(.*?)```', guide, re.S)
assert len(blocks) == 3
subprocess.run(['bash', '-eu', '-c', blocks[0]], check=True)
feedback = json.loads(Path('feedback.json').read_text())
assert feedback['primary_error'] >= 0 and len(feedback['cases']) == 1
assert feedback['cases'][0]['truth'] is None
subprocess.run(['bash', '-eu', '-c', blocks[1]], check=True)
Path('notes.md').write_text('Checked the documented interface. Error: ' + str(feedback['primary_error']))
subprocess.run(['bash', '-eu', '-c', blocks[2]], check=True)
write_agent_worker_response(observations=('Public guide commands completed.',))
''')
    controller, _ = _controller(cpu_field_factory)
    plan = replace(controller.campaign.rounds[0], agent_budget_seconds=30,
                   development_sampling=replace(controller.campaign.rounds[0].development_sampling,
                                                max_cases_per_agent_per_round=None, starter=CaseCounts(base=1)),
                   round_hidden=HiddenSuiteSpec(base=1))
    controller.campaign = replace(controller.campaign, rounds=(plan,), initial_candidate=initial,
                                  generator=replace(controller.campaign.generator, guidance_enabled=False,
                                                    perturbations_enabled=False),
                                  evaluator=replace(controller.campaign.evaluator, development_point_count=4))
    controller.public_problem_bundle = bundle
    controller.agents = {plan.agents[0].agent_id: CommandAgentProgram(
        (sys.executable, str(script)), codec.decode_nomination, incumbent_encoder=codec.encode_incumbent,
        environment={'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'src')},
        worker_settings={'nomination_policy': 'latest'},
    )}
    controller.agent_execution = SubprocessAgentExecutionService(tmp_path / 'turns', hard_wall_seconds=30)
    settings = {**controller.campaign.task_settings,
                'default_query': controller.campaign.evaluator.default_query}
    service = SubprocessCandidateExecutionService(
        tmp_path / 'candidate_workers', task=PythonTaskWorkerSpec.from_task_type(NSSolverTask),
        evaluation_settings=settings, max_workers=1, case_hard_wall_seconds=20,
    )
    controller.candidate_execution = service
    try:
        result = controller.run()
        audit = controller.agent_execution.audits[-1]
        stderr = Path(audit.stderr_path).read_text()
        assert audit.status == 'completed', stderr
        assert audit.nomination_count == 1, stderr
        assert len(result.rounds[0].candidate_evaluations) == 1
        assert 'Checked the documented interface.' in result.memory.markdown
        assert service.audits and all(item.status == 'completed' for item in service.audits)
    finally:
        service._executor.shutdown(wait=True, cancel_futures=True)
