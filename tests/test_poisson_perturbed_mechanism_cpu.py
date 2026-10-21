"""CPU checks for the Gaussian-MMS counterexample and existing worker interface."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from pde_agent_verifier_solver.plugins.poisson_diagnostic import (
    CENTERS, PoissonField, PoissonSessionProvider, PoissonSolverTask,
    generate_records, gaussian_scale, poisson_public_problem_bundle,
)
from pde_agent_verifier_solver.sdk import CaseAccessKind, AnswerQuery, AnswerView, QueryKind
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('poisson_cpu_runner', ROOT/'scripts/run_poisson_perturbed_mechanism_cpu.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def test_gaussian_mms_and_fixed_ensemble_scale(tmp_path):
    result = m.construction_check(tmp_path)
    assert result['passed'] and len(result['cases']) == 16
    rows = generate_records(71, 3, paired=True)
    same = generate_records(71, 3, paired=False)
    assert rows[:3] == same
    assert rows == generate_records(71, 3, paired=True)
    # No per-realization normalization: twice the Gaussian coefficients doubles noise.
    child = rows[3].truth
    double = PoissonField(child.amplitude, tuple(2*z for z in child.gaussian_coefficients))
    x = np.linspace(0., 1., 71)
    base = rows[0].truth.solution(x)
    np.testing.assert_allclose(double.solution(x)-base, 2*(child.solution(x)-base), atol=1e-14)
    assert len(child.gaussian_coefficients) == len(CENTERS) == 16
    assert np.isfinite(gaussian_scale())


def test_public_forcing_only_and_branch_identity(tmp_path):
    m.prepare(tmp_path)
    common, _ = m.development(tmp_path, 'common')
    no, _ = m.development(tmp_path, 'without_p')
    yes, provider = m.development(tmp_path, 'with_p')
    assert (len(common), len(no), len(yes)) == (8, 24, 24)
    ids = lambda records: {r.case.case_id for r in records}
    assert len(ids(no)&ids(yes)) == 16
    terminal = m.bank(tmp_path, 'terminal')
    assert not ids(terminal.records)&(ids(no)|ids(yes))
    assert sum(r.case.parent_case_id is not None for r in yes) == 8
    for p, c in zip(terminal.records[:8], terminal.records[8:]):
        assert p.case.case_id == c.case.parent_case_id
        assert terminal.plans[p.case.case_id] == terminal.plans[c.case.case_id]
    public = json.dumps([r.case.public_data for r in yes])
    assert not any(word in public for word in ('amplitude', 'gaussian', 'coefficients', 'sin(', 'seed'))
    assert provider.case_session_methods(CaseAccessKind.DEVELOPMENT_SOLUTION) == ()
    with pytest.raises(PermissionError):
        provider.open_case_session(provider.state, yes[0].case, CaseAccessKind.DEVELOPMENT_SOLUTION)
    session = provider.open_case_session(provider.state, yes[-1].case, CaseAccessKind.PUBLIC_PROBLEM)
    np.testing.assert_allclose(session.query_forcing([[.2], [.7]])[:, 0], yes[-1].truth.forcing([.2, .7]))
    for invalid in ([.2, .7], [[1.01]], [[float('nan')]], []):
        with pytest.raises(ValueError): session.query_forcing(invalid)
    session.close()
    with pytest.raises(RuntimeError): session.query_forcing([[.2]])
    bundle = poisson_public_problem_bundle()
    assert not bundle.development_solution_oracle_methods
    assert tuple(v.name for v in bundle.oracle_methods) == ('query_forcing',)


def test_real_isolated_candidate_and_gaussian_mms_gate(tmp_path):
    m.prepare(tmp_path)
    m.qualify(tmp_path)
    result = json.loads((tmp_path/'qualification.json').read_text())
    assert result['passed']
    assert result['fine']['mean_error'] < result['coarse']['mean_error']
    assert len(result['fine']['cases']) == 4
    b = m.bank(tmp_path, 'control')
    zero = m.bind(ROOT/'configs/candidates/poisson_zero.py', tmp_path, 'zero')
    scored = m.evaluate(zero, b, tmp_path/'zero_test', records=b.records[:1])
    assert scored['mean_error'] == pytest.approx(1.)


def stub_codex(root):
    executable = root/'stub-codex.py'
    source = m.control_source(128)
    executable.write_text(f'''#!{sys.executable}
import json, os, time
from pathlib import Path
import sys
import numpy as np
from pde_agent_verifier_solver.engine.agent_execution import submit_agent_nomination, write_agent_round_log
from pde_agent_verifier_solver.engine.agent_capability import ControllerAgentCapabilityClient
print(json.dumps({{"type":"thread.started"}}), flush=True)
sys.stdin.read()
home = Path(os.environ["CODEX_HOME"])
for _ in range(2000):
    if not (home/"auth.json").exists() and (home/"auth-lifecycle.json").is_file(): break
    time.sleep(.001)
assert not (home/"auth.json").exists()
request = json.loads(Path(os.environ["PDE_AGENT_REQUEST"]).read_text())
client = ControllerAgentCapabilityClient.from_request(request)
cases = client.current_cases()
assert len(cases) in (8,24)
assert all(c.parent_case_id is None for c in cases)
f = client.invoke_public_problem(cases[0].case_id, "query_forcing", np.array([[.3],[.7]]))
assert f.shape == (2,1) and np.isfinite(f).all()
Path("submission.py").write_text({source!r})
report = client.evaluate({{"bundle_path":"submission.py"}}, case_ids=[cases[0].case_id])
assert report.primary_error < .001
submit_agent_nomination({{"bundle_path":"submission.py"}}, sequence=1)
write_agent_round_log("Local CPU wiring test; no model call.")
Path(sys.argv[sys.argv.index("--output-last-message")+1]).write_text("Local stub complete.")
Path({str(root)!r}, "entered-"+request["execution_id"]+".json").write_text(json.dumps({{
    "case_count":len(cases),"cuda":os.environ["CUDA_VISIBLE_DEVICES"],
    "auth_removed":not (home/"auth.json").exists(),"case_ids":[c.case_id for c in cases]
}}))
''')
    executable.chmod(0o700)
    auth = root/'dummy-auth.json'; auth.write_text('{"test_only":true}\n'); auth.chmod(0o600)
    return executable, auth


def test_common_and_both_branches_via_real_canonical_worker(tmp_path):
    m.prepare(tmp_path)
    m.save(tmp_path/'qualification.json', {'passed': True, 'test_only': 'wire test'})
    executable, auth = stub_codex(tmp_path)
    for stage in m.STAGES:
        m.run_agent(tmp_path, stage, codex_binary=executable, auth_file=auth)
        result = json.loads((tmp_path/stage/'result.json').read_text())
        assert result['status'] == 'completed'
        coverage = json.loads((tmp_path/stage/'development_coverage.json').read_text())
        assert coverage['completed'] == 1 and not coverage['all_cases_evaluated']
        audit = json.loads((tmp_path/stage/'agent_execution_audits.json').read_text())[0]
        request = json.loads(Path(audit['request_path']).read_text())
        entered = json.loads((tmp_path/('entered-'+request['execution_id']+'.json')).read_text())
        assert entered['auth_removed'] and entered['cuda'] == ''
        assert entered['case_count'] == (8 if stage == 'common' else 24)
        prompt = (Path(audit['workspace'])/'CANONICAL_AGENT_PROMPT.md').read_text()
        assert 'no GPU is assigned' in prompt
        assert '10 seconds' in prompt and '900 seconds' in prompt
        assert 'amplitude' not in prompt and 'sin(pi' not in prompt
    resources = []
    for stage in ('without_p', 'with_p'):
        audit = json.loads((tmp_path/stage/'agent_execution_audits.json').read_text())[0]
        resources.append((Path(audit['workspace'])/'public_resources/prior_research_record.json').read_bytes())
    assert resources[0] == resources[1]


def test_terminal_preserves_failure_and_completed_cache(tmp_path, monkeypatch):
    rows = tuple(SimpleNamespace(case=SimpleNamespace(case_id=str(i))) for i in range(3))
    b = SimpleNamespace(records=rows, strata={'0':'base', '1':'base', '2':'base_perturbed'})
    calls = []
    def run(artifact, bank, directory, *, records, workers):
        key = records[0].case.case_id; calls.append(key)
        if key == '0': raise m.CandidateExecutionError('solver failure')
        return {'cases':[{'case_id':key,'stratum':b.strata[key],'error':.01,'metrics':{}}], 'wall_seconds':.1}
    monkeypatch.setattr(m, 'evaluate', run)
    result = m.evaluate_terminal(None, b, tmp_path/'eval')
    assert result['mean_error'] is None and result['failure_rate'] == pytest.approx(1/3)
    assert result['by_source']['base_perturbed'] == pytest.approx(.01)
    m.evaluate_terminal(None, b, tmp_path/'eval')
    assert len(calls) == 3


def test_launcher_uses_zero_gpu_and_bounded_dependencies(tmp_path):
    spec = importlib.util.spec_from_file_location('poisson_submit', ROOT/'scripts/submit_poisson_perturbed_mechanism_cpu.py')
    launcher = importlib.util.module_from_spec(spec); spec.loader.exec_module(launcher)
    receipt = launcher.submit(tmp_path/'dry_run', dry_run=True)
    info = json.loads(receipt.read_text())
    assert info['resources']['gpus'] == 0
    assert list(info['jobs']) == ['qualify','common','without_p','with_p','terminal']
    for stage, job in info['jobs'].items():
        assert '--gres=gpu:0' in job['command']
        assert job['walltime'] in ('00:15:00','00:30:00')
    assert info['jobs']['terminal']['depends_on'] == ['without_p','with_p']
    assert (Path(info['source'])/'src/pde_agent_verifier_solver/plugins/poisson_diagnostic.py').is_file()
    assert not list(Path(info['source']).rglob('auth.json'))
    with pytest.raises(FileExistsError): launcher.submit(tmp_path/'dry_run', dry_run=True)
