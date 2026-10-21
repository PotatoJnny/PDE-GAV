"""Paired campaign × common-test analysis of frozen-program case tables."""
from __future__ import annotations

import csv
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


REQUIRED = ('task_id', 'method', 'run_id', 'seed', 'artifact_id', 'cohort',
            'dataset_id', 'case_id', 'metric', 'status', 'public_case_sha256')


def read_case_rows(paths: Sequence[str | Path]) -> list[dict]:
    result = []
    for raw_path in paths:
        path = Path(raw_path)
        with path.open(encoding='utf-8-sig', newline='') as stream:
            if path.suffix.lower() == '.csv':
                result.extend(dict(row) for row in csv.DictReader(stream))
            else:
                result.extend(json.loads(line) for line in stream if line.strip())
    if not result:
        raise ValueError('no case results supplied')
    return result


def _validated_rows(rows):
    result = []
    for source in rows:
        row = dict(source)
        for key in REQUIRED:
            value = row.get(key)
            if value is None or not str(value).strip():
                raise ValueError(f'case row needs {key}')
            row[key] = str(value)
        if row['status'] not in ('ok', 'failed'):
            raise ValueError('case status must be ok or failed')
        error = row.get('error')
        if row['status'] == 'ok':
            if not row.get('plan_sha256') or not row.get('reference_sha256'):
                raise ValueError('successful case needs query and reference identities')
            if not row.get('artifact_sha256'):
                raise ValueError('successful case needs frozen artifact identity')
            if isinstance(error, bool):
                raise ValueError('scientific error must be finite and nonnegative')
            try:
                error = float(error)
            except (TypeError, ValueError) as exc:
                raise ValueError('successful case needs a numeric error') from exc
            if not math.isfinite(error) or error < 0:
                raise ValueError('scientific error must be finite and nonnegative')
            row['error'] = error
        else:
            if error not in (None, ''):
                raise ValueError('failed case must have null error, not a made-up penalty')
            row['error'] = None
        result.append(row)
    return result


def _interval(values):
    if not values:
        return None
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def summarize_comparison(rows, *, baseline: str | None = None,
                         thresholds: Mapping[str, float] | None = None,
                         bootstrap_replicates: int = 2000, bootstrap_seed: int = 0) -> dict:
    """Require complete paired blocks; retain failed cases in success denominators.

    Error deltas/intervals are available only when both methods execute every
    requested case. With failures, report execution/accuracy rates and explicitly
    conditional error summaries instead of silently comparing survivor sets.
    Conditions (including different rounds/budgets) need distinct method labels.
    """
    if isinstance(bootstrap_replicates, bool) or bootstrap_replicates < 100:
        raise ValueError('use at least 100 bootstrap replicates')
    thresholds = dict(thresholds or {})
    for value in thresholds.values():
        if isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError('thresholds must be finite and nonnegative')
    groups = defaultdict(list)
    for row in _validated_rows(rows):
        groups[row['task_id'], row['cohort'], row['metric']].append(row)
    if not groups:
        raise ValueError('no case results supplied')
    rng = np.random.default_rng(bootstrap_seed)
    output = []
    for (task, cohort, metric), group in sorted(groups.items()):
        invalid = [r for r in group if not r.get('plan_sha256') or not r.get('reference_sha256')]
        if invalid:
            failures = sorted({(r['dataset_id'], r['case_id'], r.get('error_code') or 'missing_reference_or_plan')
                               for r in invalid})
            output.append({'task_id': task, 'cohort': cohort, 'metric': metric,
                           'status': 'invalid_evaluation',
                           'n_requested_rows': len(group),
                           'evaluation_failures': [dict(zip(('dataset_id', 'case_id', 'error_code'), item))
                                                   for item in failures],
                           'methods': [], 'paired_comparisons': []})
            continue
        methods = sorted({r['method'] for r in group})
        if baseline is not None and baseline not in methods:
            raise ValueError(f'baseline {baseline!r} missing from {task}/{cohort}/{metric}')
        seeds_by_method = {m: {r['seed'] for r in group if r['method'] == m} for m in methods}
        seeds = sorted(seeds_by_method[methods[0]])
        if any(value != set(seeds) for value in seeds_by_method.values()):
            raise ValueError('methods must have identical campaign seed blocks; do not drop unmatched runs')
        cells, identities, plans, clusters = {}, {}, {}, {}
        for row in group:
            condition = row['method'], row['seed']
            identity = row['run_id'], row['artifact_id'], row.get('artifact_sha256') or 'unbound'
            if condition in identities and identities[condition] != identity:
                raise ValueError('multiple programs in one method/seed: label each round or budget separately')
            identities[condition] = identity
            case = row['dataset_id'], row['case_id']
            key = (*condition, case)
            if key in cells:
                raise ValueError('duplicate method/seed/case/metric row')
            cells[key] = row
            plan = row['plan_sha256'], row['reference_sha256'], row['public_case_sha256']
            if case in plans and plans[case] != plan:
                raise ValueError('methods must share exact query plans, reference identities and public PDE inputs')
            plans[case] = plan
            # A clean root and its children form the same resampling cluster.
            recording = row.get('recording_id')
            parent = row.get('parent_case_id')
            cluster = (row['dataset_id'], 'recording', str(recording)) if recording else (
                row['dataset_id'], 'root', str(parent or row['case_id']))
            if case in clusters and clusters[case] != cluster:
                raise ValueError('inconsistent root/recording membership across programs')
            clusters[case] = cluster
        cases = sorted(plans)
        for method in methods:
            for seed in seeds:
                present = {key[2] for key in cells if key[:2] == (method, seed)}
                if present != set(cases):
                    raise ValueError('incomplete shared test matrix; add explicit failed rows, never intersect silently')
        matrices = {method: np.array([
            [cells[method, seed, case]['error'] if cells[method, seed, case]['status'] == 'ok'
             else np.nan for case in cases] for seed in seeds], dtype=float) for method in methods}
        cluster_indices = defaultdict(list)
        for index, case in enumerate(cases):
            cluster_indices[clusters[case]].append(index)
        cluster_members = list(cluster_indices.values())
        threshold = thresholds.get(f'{task}/{cohort}/{metric}', thresholds.get(metric))
        draws = {m: {'execution_rate': [], 'accuracy_pass_rate': [], 'mean_error': []} for m in methods}
        deltas = {m: {'execution_rate_gain': [], 'accuracy_pass_rate_gain': [], 'error_reduction': []}
                  for m in methods if m != baseline}
        # With one campaign, do not imply that test-case resampling measures search variability.
        if len(seeds) >= 2:
            for _ in range(bootstrap_replicates):
                seed_ix = rng.integers(0, len(seeds), len(seeds))
                root_ix = rng.integers(0, len(cluster_members), len(cluster_members))
                case_ix = [index for root in root_ix for index in cluster_members[root]]
                sample_stats = {}
                for method, matrix in matrices.items():
                    sampled = matrix[np.ix_(seed_ix, case_ix)]
                    values = {'execution_rate': float(np.isfinite(sampled).mean())}
                    if threshold is not None:
                        values['accuracy_pass_rate'] = float((np.isfinite(sampled) & (sampled <= threshold)).mean())
                    if np.isfinite(matrix).all():
                        values['mean_error'] = float(sampled.mean())
                    sample_stats[method] = values
                    for key, value in values.items():
                        draws[method][key].append(value)
                if baseline is not None:
                    reference = sample_stats[baseline]
                    for method in deltas:
                        for field, target, sign in [('execution_rate', 'execution_rate_gain', 1),
                                                     ('accuracy_pass_rate', 'accuracy_pass_rate_gain', 1),
                                                     ('mean_error', 'error_reduction', -1)]:
                            if field in reference and field in sample_stats[method]:
                                deltas[method][target].append(sign * (sample_stats[method][field] - reference[field]))
        summaries = []
        for method, matrix in matrices.items():
            finite = np.isfinite(matrix)
            successful = matrix[finite]
            tail_count = max(1, math.ceil(successful.size * .2))
            summaries.append({
                'method': method, 'campaigns': len(seeds), 'cases_per_campaign': len(cases),
                'n_requested': int(matrix.size), 'n_executed': int(finite.sum()),
                'n_failed': int((~finite).sum()), 'execution_rate': float(finite.mean()),
                'execution_rate_ci95': _interval(draws[method]['execution_rate']),
                'accuracy_threshold': threshold,
                'accuracy_pass_rate': (float((finite & (matrix <= threshold)).mean())
                                       if threshold is not None else None),
                'accuracy_pass_rate_ci95': _interval(draws[method]['accuracy_pass_rate']),
                'mean_error_successful_cases': float(successful.mean()) if successful.size else None,
                'cvar20_successful_cases': float(np.sort(successful)[-tail_count:].mean()) if successful.size else None,
                'mean_error_all_cases': float(matrix.mean()) if finite.all() else None,
                'mean_error_ci95': _interval(draws[method]['mean_error']),
            })
        by_method = {row['method']: row for row in summaries}
        comparisons = []
        if baseline is not None:
            reference = by_method[baseline]
            for method in methods:
                if method == baseline:
                    continue
                current = by_method[method]
                entry = {'method': method, 'baseline': baseline}
                for field, target, sign in [('execution_rate', 'execution_rate_gain', 1),
                                             ('accuracy_pass_rate', 'accuracy_pass_rate_gain', 1),
                                             ('mean_error_all_cases', 'error_reduction', -1)]:
                    left, right = current[field], reference[field]
                    entry[target] = sign * (left - right) if left is not None and right is not None else None
                    entry[target + '_ci95'] = _interval(deltas[method][target])
                comparisons.append(entry)
        output.append({'task_id': task, 'cohort': cohort, 'metric': metric, 'status': 'ok',
                       'n_independent_campaigns': len(seeds), 'n_test_clusters': len(cluster_members),
                       'methods': summaries, 'paired_comparisons': comparisons})
    return {'schema': 'paired_program_comparison_v1', 'bootstrap_seed': bootstrap_seed,
            'bootstrap_replicates': bootstrap_replicates, 'groups': output,
            'notes': [
                'Fixed task suite: campaign blocks and shared test roots/recordings are resampled together across methods.',
                'Reference/plan failures invalidate the corresponding group; they are not Agent failures or silently removed cases.',
                'No confidence interval is reported with fewer than two independent campaigns.',
                'Error summaries on successful cases are conditional; failures remain in execution and accuracy denominators.',
                'No raw error macro-average is taken across tasks or differently scaled metrics.',
                'Method labels must distinguish budgets and round snapshots; test does not choose the delivered program.',
            ]}


def write_comparison_report(report: Mapping, output: str | Path, *, plots=False):
    """Export reusable JSON/CSV/LaTeX, optionally a standard scientific PDF figure."""
    destination = Path(output)
    if plots:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    invalid_groups = [group for group in report['groups'] if group.get('status') == 'invalid_evaluation']
    if invalid_groups:
        (destination / 'invalid_evaluations.json').write_text(json.dumps(invalid_groups, indent=2) + '\n')
    rows = [{key: group[key] for key in ('task_id', 'cohort', 'metric')} | row
            for group in report['groups'] for row in group['methods']]
    if rows:
        with (destination / 'summary.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    paired = [{key: group[key] for key in ('task_id', 'cohort', 'metric')} | row
              for group in report['groups'] for row in group['paired_comparisons']]
    if paired:
        with (destination / 'paired_comparisons.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
            writer.writeheader()
            writer.writerows(paired)
    cost_rows = report.get('quality_cost_rows', [])
    if cost_rows:
        with (destination / 'quality_cost.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(cost_rows[0]))
            writer.writeheader()
            writer.writerows(cost_rows)
    def escape(value):
        replacements = {'\\': r'\textbackslash{}', '_': r'\_', '&': r'\&', '%': r'\%',
                        '#': r'\#', '$': r'\$', '{': r'\{', '}': r'\}'}
        return ''.join(replacements.get(char, char) for char in str(value))
    tex = [r'\begin{tabular}{llllrr}', r'Task & Cohort & Metric & Method & Execution & Error (successful) \\', r'\hline']
    for row in rows:
        error = row['mean_error_successful_cases']
        tex.append(' & '.join([escape(row[k]) for k in ('task_id', 'cohort', 'metric', 'method')] +
                              [f"{row['execution_rate']:.3f}", '--' if error is None else f'{error:.5g}']) + r' \\')
    tex.append(r'\end{tabular}')
    (destination / 'table.tex').write_text('\n'.join(tex) + '\n')
    if plots:
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(destination / 'comparison.pdf') as pdf:
            for group in report['groups']:
                if group.get('status') == 'invalid_evaluation':
                    figure, axis = plt.subplots(figsize=(8, 3))
                    axis.set_axis_off()
                    axis.text(.5, .5, f"{group['task_id']} / {group['cohort']}: invalid reference or plan.\nNo scientific comparison computed.", ha='center', va='center', transform=axis.transAxes)
                    pdf.savefig(figure)
                    plt.close(figure)
                    continue
                methods = group['methods']
                figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
                labels = [row['method'] for row in methods]
                axes[0].bar(labels, [row['execution_rate'] for row in methods])
                axes[0].set(ylabel='Execution rate (all requested cases)', ylim=(0, 1.05))
                if all(row['mean_error_all_cases'] is not None for row in methods):
                    means = np.array([row['mean_error_all_cases'] for row in methods])
                    if all(row['mean_error_ci95'] is not None for row in methods):
                        ci = np.array([row['mean_error_ci95'] for row in methods])
                        # Draw intervals independently: percentile intervals need not contain the point estimate.
                        x = np.arange(len(methods))
                        axes[1].vlines(x, ci[:, 0], ci[:, 1])
                        axes[1].plot(x, means, 'o')
                        axes[1].set_xticks(x, labels)
                    else:
                        axes[1].plot(labels, means, 'o')
                    axes[1].set_ylabel(group['metric'])
                else:
                    axes[1].text(.5, .5, 'Failures present; see execution/accuracy rates.\nConditional errors are in the table.',
                                 ha='center', va='center', transform=axes[1].transAxes)
                    axes[1].set_axis_off()
                for axis in axes:
                    axis.tick_params(axis='x', labelrotation=25)
                figure.suptitle(f"{group['task_id']} / {group['cohort']}")
                pdf.savefig(figure)
                plt.close(figure)
                cost_rows = [r for r in report.get('quality_cost_rows', [])
                             if all(r[k] == group[k] for k in ('task_id', 'cohort', 'metric'))]
                if cost_rows:
                    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
                    for column, field, label in zip(range(2), ('input_plus_output_tokens', 'agent_wall_seconds_sum'),
                                                   ('Observed input + output tokens', 'Agent elapsed seconds including tools (sum)')):
                        unavailable, failed = [], []
                        for row in cost_rows:
                            x, y = row[field], row['mean_error_all_cases']
                            if x is None:
                                unavailable.append(row['method'])
                                continue
                            axes[1, column].scatter([x], [row['execution_rate']])
                            axes[1, column].annotate(row['method'], (x, row['execution_rate']),
                                                     xytext=(4, 4), textcoords='offset points')
                            if y is None:
                                failed.append(row['method'])
                                continue
                            axes[0, column].scatter([x], [y])
                            axes[0, column].annotate(row['method'], (x, y), xytext=(4, 4), textcoords='offset points')
                        notices = []
                        if failed:
                            notices.append('Failures (rates below): ' + ', '.join(failed))
                        if unavailable:
                            notices.append('Cost unavailable: ' + ', '.join(unavailable))
                        if notices:
                            axes[0, column].text(.03, .97, '\n'.join(notices), va='top',
                                                  fontsize=8, transform=axes[0, column].transAxes)
                        axes[0, column].set(xlabel=label, ylabel=group['metric'])
                        axes[1, column].set(xlabel=label, ylabel='Execution rate (all requested cases)', ylim=(0, 1.05))
                    figure.suptitle(f"{group['task_id']} / {group['cohort']}: observed quality–cost points")
                    pdf.savefig(figure)
                    plt.close(figure)



def attach_cost_reports(report: dict, case_rows, manifest_path: str | Path):
    """Join one explicit observed cost report per method/run/seed.

    Manifest is a JSON list with method, run_id, seed, costs (relative filename).
    It is not a price table or a scheduler manifest. Missing usage stays unknown.
    """
    path = Path(manifest_path)
    entries = json.loads(path.read_text())
    if not isinstance(entries, list):
        raise ValueError('cost manifest must be a list')
    resources = {}
    source_paths = set()
    for entry in entries:
        key = str(entry['method']), str(entry['run_id']), str(entry['seed'])
        if key in resources:
            raise ValueError('duplicate method/run/seed cost report')
        cost_path = Path(entry['costs'])
        if not cost_path.is_absolute():
            cost_path = path.parent / cost_path
        cost_path = cost_path.resolve()
        if cost_path in source_paths:
            raise ValueError('different programs cannot reuse the same cost report file')
        source_paths.add(cost_path)
        value = json.loads(cost_path.read_text())
        if str(value.get('run_id')) != key[1]:
            raise ValueError('cost report run_id differs from comparison run_id')
        if value.get('schema') != 'run_resource_report_v1':
            raise ValueError('unsupported cost report schema')
        tokens = value.get('tokens', {})
        complete = value.get('usage_coverage') == 'completed_turns_observed'
        token_total = (tokens['input_tokens'] + tokens['output_tokens'] if complete
                       and tokens.get('input_tokens') is not None
                       and tokens.get('output_tokens') is not None else None)
        resources[key] = {
            'input_plus_output_tokens': token_total,
            'agent_active_seconds_sum': value.get('agent_active_seconds_sum'),
            'agent_wall_seconds_sum': value.get('agent_wall_seconds_sum'),
            'observed_activity_span_seconds': value.get('observed_activity_span_seconds'),
            'observed_llm_cost': value.get('observed_llm_cost') if complete else None,
        }
        for number in resources[key].values():
            if number is not None and (isinstance(number, bool) or not math.isfinite(number) or number < 0):
                raise ValueError('invalid resource value')
    programs = defaultdict(set)
    for row in _validated_rows(case_rows):
        group = row['task_id'], row['cohort'], row['metric'], row['method']
        programs[group].add((row['method'], row['run_id'], row['seed']))
    used = {program for group_programs in programs.values() for program in group_programs}
    if any(program not in resources for program in used):
        raise ValueError('cost manifest must cover every compared method/run/seed')
    cost_rows = []
    for group in report['groups']:
        for method in group['methods']:
            key = group['task_id'], group['cohort'], group['metric'], method['method']
            selected = programs[key]
            if any(program not in resources for program in selected):
                raise ValueError('cost manifest must cover every compared method/run/seed')
            used.update(selected)
            averages = {}
            for field in ('input_plus_output_tokens', 'agent_active_seconds_sum', 'agent_wall_seconds_sum',
                          'observed_activity_span_seconds', 'observed_llm_cost'):
                values = [resources[program][field] for program in selected]
                averages[field] = sum(values) / len(values) if all(v is not None for v in values) else None
            method['mean_resources_per_campaign'] = averages
            cost_rows.append({**dict(zip(('task_id', 'cohort', 'metric', 'method'), key)),
                              'campaigns': len(selected), **averages,
                              'mean_error_all_cases': method['mean_error_all_cases'],
                              'execution_rate': method['execution_rate'],
                              'accuracy_pass_rate': method['accuracy_pass_rate']})
    if set(resources) != used:
        raise ValueError('cost manifest contains unused programs; select the matching comparison')
    report['quality_cost_rows'] = cost_rows
    return report
