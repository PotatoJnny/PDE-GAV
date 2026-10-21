from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pde_agent_verifier_solver.reporting.comparisons import (
    read_case_rows, summarize_comparison, write_comparison_report,
)
from pde_agent_verifier_solver.reporting.costs import (
    MeasuredGenerator, collect_run_costs, parse_codex_usage,
)


def case_rows():
    rows = []
    for method, factor in [('base', 1), ('full', .5)]:
        for seed in range(3):
            for case, error in [('root', 1), ('child', 3), ('independent', 8)]:
                rows.append(dict(task_id='ns', method=method, run_id=f'{method}-{seed}', seed=seed,
                                 artifact_id=f'{method}-{seed}', artifact_sha256=method * 4,
                                 cohort='target', dataset_id='dataset', case_id=case,
                                 parent_case_id='root' if case == 'child' else None,
                                 recording_id=None, metric='velocity_relative_l2',
                                 error=factor * (error + seed), status='ok',
                                 plan_sha256='plan-' + case, reference_sha256='truth-' + case,
                                 public_case_sha256='public-' + case))
    return rows


def test_paired_crossed_resampling_preserves_exact_program_difference():
    rows = case_rows()
    for row in rows:
        if row['method'] == 'full':
            row['error'] = next(r['error'] for r in rows if r['method'] == 'base'
                                and r['seed'] == row['seed'] and r['case_id'] == row['case_id']) - .5
    report = summarize_comparison(rows, baseline='base', bootstrap_replicates=100)
    group = report['groups'][0]
    assert group['n_test_clusters'] == 2  # root and its child remain together
    effect = group['paired_comparisons'][0]
    assert effect['error_reduction'] == pytest.approx(.5)
    assert effect['error_reduction_ci95'] == pytest.approx([.5, .5])


def test_failure_is_counted_and_never_silently_dropped_from_comparison():
    rows = case_rows()
    rows[-1].update(status='failed', error=None)
    report = summarize_comparison(rows, baseline='base', thresholds={'velocity_relative_l2': 4},
                                  bootstrap_replicates=100)
    group = report['groups'][0]
    full = next(row for row in group['methods'] if row['method'] == 'full')
    assert full['n_requested'] == 9 and full['n_failed'] == 1
    assert full['execution_rate'] == pytest.approx(8 / 9)
    assert full['accuracy_pass_rate'] <= 8 / 9
    assert full['mean_error_all_cases'] is None and full['mean_error_ci95'] is None
    assert group['paired_comparisons'][0]['error_reduction'] is None
    assert group['paired_comparisons'][0]['execution_rate_gain'] == pytest.approx(-1 / 9)


@pytest.mark.parametrize('mutation,message', [
    (lambda r: r.pop(), 'incomplete shared test'),
    (lambda r: r.append(deepcopy(r[0])), 'duplicate'),
    (lambda r: r[-1].update(plan_sha256='different'), 'exact query'),
    (lambda r: r[-1].update(reference_sha256='different'), 'reference'),
    (lambda r: r[-1].update(public_case_sha256='different'), 'public PDE'),
    (lambda r: r[-1].update(run_id='new-run'), 'multiple programs'),
])
def test_comparison_rejects_nonpaired_or_ambiguous_tables(mutation, message):
    rows = case_rows()
    mutation(rows)
    with pytest.raises(ValueError, match=message):
        summarize_comparison(rows, bootstrap_replicates=100)


def test_single_campaign_has_no_fake_search_uncertainty():
    report = summarize_comparison([r for r in case_rows() if r['seed'] == 0], baseline='base',
                                  bootstrap_replicates=100)
    assert all(r['execution_rate_ci95'] is None for r in report['groups'][0]['methods'])


def test_export_roundtrip_and_no_overwrite(tmp_path):
    path = tmp_path / 'cases.jsonl'
    path.write_text(''.join(json.dumps(row) + '\n' for row in case_rows()))
    report = summarize_comparison(read_case_rows([path]), baseline='base', bootstrap_replicates=100)
    write_comparison_report(report, tmp_path / 'report', plots=True)
    assert (tmp_path / 'report/comparison.pdf').read_bytes().startswith(b'%PDF')
    assert (tmp_path / 'report/paired_comparisons.csv').is_file()
    assert 'velocity\\_relative\\_l2' in (tmp_path / 'report/table.tex').read_text()
    with pytest.raises(FileExistsError):
        write_comparison_report(report, tmp_path / 'report')


def write_log(path, events):
    path.write_text('\n'.join(json.dumps(event) for event in events) + '\n')


def test_usage_ignores_cumulative_counters_and_keeps_cached_input_separate(tmp_path):
    path = tmp_path / 'stdout.log'
    write_log(path, [
        dict(type='turn.completed', usage=dict(input_tokens=100, cached_input_tokens=40, output_tokens=20)),
        dict(type='session.updated', usage=dict(input_tokens=100, output_tokens=20)),
        dict(type='turn.completed', usage=dict(input_tokens=30, cached_input_tokens=10, output_tokens=5)),
    ])
    usage = parse_codex_usage(path)
    assert usage['input_tokens'] == 130 and usage['output_tokens'] == 25
    assert usage['cached_input_tokens'] == 50 and usage['completed_turns'] == 2
    assert parse_codex_usage(tmp_path / 'missing.log')['input_tokens'] is None


def test_partial_usage_and_failed_executions_are_visible(tmp_path):
    path = tmp_path / 'stdout.log'
    write_log(path, [dict(type='turn.completed', usage=dict(input_tokens=100, cached_input_tokens=40, output_tokens=20))])
    with path.open('a') as stream:
        stream.write('{"type":')
    (tmp_path / 'execution_audits.json').write_text(json.dumps([
        dict(execution_id='x', status='failed', stdout_path='stdout.log', active_seconds=2,
             wall_seconds=3, controller_service_seconds=1, started_at_unix=10, finished_at_unix=13)]))
    (tmp_path / 'candidate_execution_audits.json').write_text(json.dumps([
        dict(status='failed', wall_seconds=1), dict(status='completed', wall_seconds=2)]))
    report = collect_run_costs(tmp_path, pricing=dict(currency='USD', model='test', as_of='2026-09-06',
                                                     input_per_million=2, cached_input_per_million=1,
                                                     output_per_million=4))
    assert report['usage_coverage'] == 'incomplete'
    assert report['candidate_wall_seconds_sum'] == 3
    assert report['agent_active_seconds_sum'] == 2
    assert report['observed_llm_cost'] == pytest.approx((60 * 2 + 40 + 20 * 4) / 1e6)
    assert report['observed_activity_span_seconds'] == 3
    assert report['prior_costs'] is None


def test_generator_measurement_preserves_results_sessions_and_failure(tmp_path):
    class Generator:
        capabilities = object()
        def sample(self, state, request, progress=None):
            if request.count == 2:
                raise RuntimeError('sampling rejected')
            return iter(['record'])
        def open_case_session(self, *args):
            return 'session'
    generator = Generator()
    measured = MeasuredGenerator(generator, tmp_path / 'generator_operations.jsonl')
    request = SimpleNamespace(count=1, kind='base', purpose='development')
    assert measured.capabilities is generator.capabilities
    assert measured.sample(None, request) == ('record',)
    assert measured.open_case_session(None) == 'session'
    request.count = 2
    with pytest.raises(RuntimeError, match='rejected'):
        measured.sample(None, request)
    report = collect_run_costs(tmp_path)
    assert report['generator_operations'][0]['calls'] == 2
    assert report['generator_operations'][0]['failed_calls'] == 1
    assert not report['generator_incomplete_operations']
    assert report['tokens']['input_tokens'] is None


def test_generator_interrupted_operation_is_not_zero_cost(tmp_path):
    row = dict(event='started', operation_id=1, operation='guidance_training', started_at_unix=1)
    (tmp_path / 'generator_operations.jsonl').write_text(json.dumps(row) + '\n')
    report = collect_run_costs(tmp_path)
    assert len(report['generator_incomplete_operations']) == 1
    assert not report['generator_operations']


def test_cost_join_uses_all_campaigns_and_does_not_price_missing_usage(tmp_path):
    from pde_agent_verifier_solver.reporting.comparisons import attach_cost_reports
    rows = case_rows()
    report = summarize_comparison(rows, baseline='base', bootstrap_replicates=100)
    manifest = []
    for method in ['base', 'full']:
        for seed in range(3):
            path = tmp_path / f'{method}-{seed}.json'
            path.write_text(json.dumps(dict(schema='run_resource_report_v1', run_id=f'{method}-{seed}',
                usage_coverage='completed_turns_observed' if seed else 'incomplete',
                tokens=dict(input_tokens=100, output_tokens=20),
                agent_active_seconds_sum=seed + 1, agent_wall_seconds_sum=seed + 5,
                observed_activity_span_seconds=4, observed_llm_cost=.01)))
            manifest.append(dict(method=method, run_id=f'{method}-{seed}', seed=seed, costs=path.name))
    manifest_path = tmp_path / 'costs.json'
    manifest_path.write_text(json.dumps(manifest))
    attach_cost_reports(report, rows, manifest_path)
    assert len(report['quality_cost_rows']) == 2
    assert all(row['input_plus_output_tokens'] is None for row in report['quality_cost_rows'])
    assert all(row['agent_active_seconds_sum'] == 2 for row in report['quality_cost_rows'])
    assert all(row['agent_wall_seconds_sum'] == 6 for row in report['quality_cost_rows'])
    write_comparison_report(report, tmp_path / 'cost-report', plots=True)
    assert (tmp_path / 'cost-report/quality_cost.csv').is_file()
    manifest.pop()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='every compared'):
        attach_cost_reports(report, rows, manifest_path)


def test_reference_failure_is_not_blame_assigned_to_agent(tmp_path):
    rows = case_rows()
    for row in rows:
        if row['case_id'] == 'root':
            row.update(status='failed', error=None, reference_sha256=None, error_code='OracleError')
    report = summarize_comparison(rows, baseline='base', bootstrap_replicates=100)
    group = report['groups'][0]
    assert group['status'] == 'invalid_evaluation'
    assert group['methods'] == [] and group['paired_comparisons'] == []
    assert group['evaluation_failures'][0]['case_id'] == 'root'
    write_comparison_report(report, tmp_path / 'invalid')
    assert (tmp_path / 'invalid/invalid_evaluations.json').exists()


def test_unbound_program_is_retained_as_failed_campaign():
    rows = case_rows()
    for row in rows:
        if row['method'] == 'full' and row['seed'] == 0:
            row.update(status='failed', error=None, artifact_sha256=None)
    report = summarize_comparison(rows, baseline='base', bootstrap_replicates=100)
    full = next(r for r in report['groups'][0]['methods'] if r['method'] == 'full')
    assert full['n_failed'] == 3 and full['execution_rate'] == pytest.approx(2 / 3)


@pytest.mark.parametrize('wrong_source', ['identity', 'reused_file'])
def test_cost_join_rejects_unrelated_or_reused_run_reports(tmp_path, wrong_source):
    from pde_agent_verifier_solver.reporting.comparisons import attach_cost_reports
    rows = case_rows()
    report = summarize_comparison(rows, bootstrap_replicates=100)
    first = tmp_path / 'base-0.json'
    first.write_text(json.dumps(dict(schema='run_resource_report_v1', run_id='base-0',
        usage_coverage='incomplete', tokens={})))
    entries = [dict(method='base', run_id='base-0', seed=0, costs=first.name)]
    if wrong_source == 'identity':
        entries[0]['run_id'] = 'unrelated-run'
        message = 'run_id differs'
    else:
        entries.append(dict(method='full', run_id='full-0', seed=0, costs=first.name))
        message = 'same cost report file'
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(entries))
    with pytest.raises(ValueError, match=message):
        attach_cost_reports(report, rows, path)
