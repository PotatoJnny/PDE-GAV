from hashlib import sha256
import json
import math
from types import SimpleNamespace

import pytest

from pde_agent_verifier_solver.plugins.ns.baselines.codepde import (
    CodePDEEvaluation, CodePDERepeatedSampleConfig, PROVENANCE_PATH,
    run_codepde_ns, run_repeated_sample,
)


def test_every_sample_is_cold_and_first_finite_score_stops_even_when_poor():
    requests = []
    def generate(request):
        requests.append(request)
        return f'sample-{request.sample_index}'
    errors = {'sample-0': 99., 'sample-1': 3., 'sample-2': 3.}
    result = run_repeated_sample(
        CodePDERepeatedSampleConfig(3, 5), generate=generate,
        evaluate=lambda key: CodePDEEvaluation(0, errors[key]),
    )
    assert len(requests) == 3
    assert all(r.parent_key is None and r.history == () and r.trial_index == 1 for r in requests)
    assert result.selected_key == 'sample-1'  # earlier finite tie, not last sample
    assert result.selection_error == 3.


def test_only_execution_or_nonfinite_results_debug_the_last_failed_program():
    requests = []
    def generate(request):
        requests.append(request)
        return f'{request.sample_index}-{request.trial_index}'
    feedback = {
        '0-1': CodePDEEvaluation(1, stderr='NameError'),
        '0-2': CodePDEEvaluation(0, math.nan, stdout='unstable'),
        '0-3': CodePDEEvaluation(0, 4.),
        '1-1': CodePDEEvaluation(0, 2.),
    }
    result = run_repeated_sample(CodePDERepeatedSampleConfig(2, 5),
                                 generate=generate, evaluate=feedback.__getitem__)
    assert [r.generation_mode for r in requests] == [
        'initial', 'debugging_execution_error', 'debugging_nan_inf', 'initial',
    ]
    assert [r.parent_key for r in requests] == [None, '0-1', '0-2', None]
    assert [[t.candidate_key for t in r.history] for r in requests] == [[], ['0-1'], ['0-1', '0-2'], []]
    assert result.selected_key == '1-1'


def test_trial_limit_includes_initial_and_all_failed_branches_remain():
    calls = []
    def generate(request):
        calls.append(request)
        return f'{request.sample_index}-{request.trial_index}'
    result = run_repeated_sample(CodePDERepeatedSampleConfig(2, 3), generate=generate,
                                 evaluate=lambda _: CodePDEEvaluation(-1, stderr='timeout'))
    assert len(calls) == 6
    assert result.selected_key is result.selection_error is None
    assert [s['trials'] for s in result.samples] == [3, 3]
    assert all(s['status'] == 'execution_failed' for s in result.samples)


@pytest.mark.parametrize('failure', ('none', 'parse_error'))
def test_generation_failure_aborts_sample_without_unbound_debug(failure):
    requests = []
    def generate(request):
        requests.append(request)
        if failure == 'none':
            return None
        raise ValueError('No relevant code block found')
    result = run_repeated_sample(CodePDERepeatedSampleConfig(2, 5), generate=generate,
                                 evaluate=lambda _: pytest.fail('No candidate should execute'))
    assert len(requests) == 2
    assert all(r.generation_mode == 'initial' and r.history == () for r in requests)
    assert all(t.status == 'generation_failed' for t in result.trials)
    assert all(t.exception for t in result.trials)


def test_result_parsing_exception_aborts_only_its_sample():
    def evaluate(key):
        if key == '0':
            raise ValueError('missing metric')
        return CodePDEEvaluation(0, 1.)
    result = run_repeated_sample(CodePDERepeatedSampleConfig(2, 5),
                                 generate=lambda request: str(request.sample_index), evaluate=evaluate)
    assert len(result.trials) == 2
    assert result.trials[0].status == 'evaluation_failed'
    assert result.selected_key == '1'


@pytest.mark.parametrize('error', (math.nan, math.inf, -math.inf))
def test_final_nonfinite_result_is_retained_but_never_selected(error, tmp_path):
    result = run_repeated_sample(CodePDERepeatedSampleConfig(1, 2),
                                 generate=lambda r: f'candidate-{r.trial_index}',
                                 evaluate=lambda _: CodePDEEvaluation(0, error),
                                 output_directory=tmp_path)
    assert result.samples[0]['status'] == 'nonfinite'
    assert result.samples[0]['candidate_key'] == 'candidate-2'
    assert result.selected_key is None
    report = json.loads((tmp_path / 'codepde_adaptation.json').read_text())
    assert report['samples'][0]['relative_error'] is None
    assert report['generation_calls'] == 2
    assert report['author_code_executed'] is False
    adapter_path = PROVENANCE_PATH.parent.parent / 'codepde.py'
    assert report['local_adapter_sha256'] == sha256(adapter_path.read_bytes()).hexdigest()
    assert report['upstream']['upstream_source_vendored'] is False
    assert report['upstream']['revision'] == '26938c5832ddd0a957426dff6f702b54649c14af'


def test_s0_rejected_before_any_generation_or_output(tmp_path):
    with pytest.raises(ValueError, match='cold start'):
        run_repeated_sample(CodePDERepeatedSampleConfig(), initial_key='s0',
                             generate=lambda _: pytest.fail('must not generate'),
                             evaluate=lambda _: pytest.fail('must not evaluate'),
                             output_directory=tmp_path / 'output')
    assert not (tmp_path / 'output').exists()


def test_ns_bridge_preserves_branch_local_feedback_and_failure_modes(tmp_path):
    config = SimpleNamespace(output_directory=tmp_path, codepde_num_repeated_samples=2,
                             codepde_num_debugging_trials_per_sample=4)
    calls = []
    keys = iter(['runtime-bad', 'numerically-bad', 'good', 'second-sample'])
    def propose(parent, history, phase):
        calls.append((parent, history, phase))
        return next(keys)
    scores = {'runtime-bad': math.inf, 'numerically-bad': math.inf, 'good': .3, 'second-sample': .7}
    failures = {'runtime-bad': 'RuntimeError: failed solve', 'numerically-bad': 'nonfinite_selection_error'}
    selected = run_codepde_ns(
        config, initial_key=None, propose=propose, score=scores.__getitem__,
        history_for=lambda key: ((f'Notes for {key}', None),), failure_for=failures.get,
    )
    assert selected == 'good'
    assert calls[0][0:2] == (None, ()) and calls[3][0:2] == (None, ())
    assert calls[1][0] == 'runtime-bad' and 'debugging_execution_error' in calls[1][2]
    assert calls[2][0] == 'numerically-bad' and 'debugging_nan_inf' in calls[2][2]
    assert len(calls[2][1]) == 2
    assert 'RuntimeError: failed solve' in calls[1][1][0][0]
    report = json.loads((tmp_path / 'codepde_adaptation.json').read_text())
    assert report['maximum_generation_calls'] == 8 and report['generation_calls'] == 4
    assert report['adaptation_details']['failure_classification'] == 'selection_failure_reason'


def test_ns_bridge_none_nomination_ends_sample_and_does_not_score(tmp_path):
    config = SimpleNamespace(output_directory=tmp_path, codepde_num_repeated_samples=2,
                             codepde_num_debugging_trials_per_sample=5)
    calls = []
    selected = run_codepde_ns(config, initial_key=None,
        propose=lambda *args: calls.append(args), score=lambda _: pytest.fail('no score'),
        history_for=lambda _: pytest.fail('no history'))
    assert selected is None and len(calls) == 2
    assert all(parent is None and history == () for parent, history, _ in calls)


def test_ns_bridge_infrastructure_failure_is_fatal_and_recorded(tmp_path):
    from pde_agent_verifier_solver.engine.agent_execution import AgentInfrastructureError
    config = SimpleNamespace(output_directory=tmp_path, codepde_num_repeated_samples=2,
                             codepde_num_debugging_trials_per_sample=5)
    def propose(*args):
        raise AgentInfrastructureError('missing worker binary')
    with pytest.raises(AgentInfrastructureError):
        run_codepde_ns(config, initial_key=None, propose=propose, score=lambda _: 1., history_for=lambda _: ())
    report = json.loads((tmp_path / 'codepde_adaptation.json').read_text())
    assert report['status'] == 'interrupted' and report['generation_calls'] == 1


@pytest.mark.parametrize('samples,trials', ((0, 1), (1, 0), (True, 1), (1, 1.5)))
def test_rejects_invalid_search_bounds(samples, trials):
    with pytest.raises(ValueError):
        CodePDERepeatedSampleConfig(samples, trials)


def test_callback_replacement_and_budget_defaults_are_explicit():
    config = CodePDERepeatedSampleConfig()
    assert config.maximum_generation_calls == 10
    provenance = json.loads(PROVENANCE_PATH.read_text())
    assert provenance['upstream_license'] is None
    assert len(provenance['source_files']) == 3
    assert all(len(row['sha256']) == 64 for row in provenance['source_files'])


def test_debug_bridge_truncates_execution_feedback_without_cross_sample_history(tmp_path):
    config = SimpleNamespace(output_directory=tmp_path, codepde_num_repeated_samples=1,
                             codepde_num_debugging_trials_per_sample=2)
    calls = []
    def propose(parent, history, phase):
        calls.append((parent, history, phase))
        return 'bad' if parent is None else 'good'
    error_text = 'SECRET_PREFIX' + 'x' * 5000
    run_codepde_ns(config, initial_key=None, propose=propose,
                   score=lambda key: math.inf if key == 'bad' else 1.,
                   history_for=lambda _: (),
                   failure_for=lambda key: error_text if key == 'bad' else None)
    assert 'SECRET_PREFIX' not in calls[1][1][0][0]
    assert 'x' * 5000 in calls[1][1][0][0]


def test_trace_output_refuses_to_overwrite_a_completed_search(tmp_path):
    kwargs = dict(generate=lambda _: 'candidate', evaluate=lambda _: CodePDEEvaluation(0, 1.),
                  output_directory=tmp_path)
    run_repeated_sample(CodePDERepeatedSampleConfig(1, 1), **kwargs)
    before = (tmp_path / 'codepde_adaptation.json').read_bytes()
    with pytest.raises(FileExistsError):
        run_repeated_sample(CodePDERepeatedSampleConfig(1, 1), **kwargs)
    assert (tmp_path / 'codepde_adaptation.json').read_bytes() == before
