"""Observed resource use; elapsed service times are never advertised as FLOPs."""
from __future__ import annotations

from contextlib import contextmanager
from itertools import count
import json
import math
import os
from pathlib import Path
from threading import Lock
from time import perf_counter, time
from typing import Mapping


class MeasuredGenerator:
    """Record existing Generator calls without changing their scientific contract.

    Session factories, release and all other plugin capabilities are forwarded.
    Sampling is materialized inside the measured operation, including failures.
    Existing checkpoint training cost remains unknown unless supplied separately.
    """

    def __init__(self, generator, operations_path: str | Path, *, resume: bool = False):
        self._generator = generator
        self._path = Path(operations_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        maximum_id = 0
        if resume:
            from pde_agent_verifier_solver.engine.audit_journal import AuditJournal
            if not self._path.is_file():
                raise FileNotFoundError(self._path)
            # A killed allocation may leave an uncommitted final JSON fragment.
            # Preserve it for diagnosis, then append after the last valid row.
            rows = AuditJournal(self._path).rows
            maximum_id = max((row.get('operation_id', 0) for row in rows), default=0)
        else:
            with self._path.open('x', encoding='utf-8'):
                pass
        self._lock = Lock()
        self._ids = count(maximum_id + 1)
        self._dirty = False

    def __getattr__(self, name):
        return getattr(self._generator, name)

    def _write(self, row):
        with self._lock, self._path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False, sort_keys=True) + '\n')
            self._dirty = True

    def flush_operations(self):
        """Commit new measurements at a checkpoint, without per-request fsync."""
        with self._lock:
            if self._dirty:
                with self._path.open('ab') as stream:
                    os.fsync(stream.fileno())
                self._dirty = False

    @contextmanager
    def _operation(self, operation, **details):
        with self._lock:
            identifier = next(self._ids)
        row = dict(schema='generator_operation_v1', operation_id=identifier,
                   operation=operation, started_at_unix=time(), **details)
        self._write({**row, 'event': 'started'})
        started = perf_counter()
        outcome = {}
        try:
            yield outcome
        except BaseException as error:
            outcome.update(status='failed', error_type=type(error).__name__)
            raise
        else:
            outcome['status'] = 'ok'
        finally:
            self._write({**row, **outcome, 'event': 'finished',
                         'wall_seconds': perf_counter() - started,
                         'finished_at_unix': time()})

    def train(self, dataset, settings, progress=None):
        with self._operation('base_training'):
            return self._generator.train(dataset, settings, progress)

    def load(self, checkpoint, settings, progress=None):
        with self._operation('base_loading'):
            return self._generator.load(checkpoint, settings, progress)

    def update_guidance(self, state, feedback, settings, progress=None):
        with self._operation('guidance_training',
                             round_index=feedback.target_round_index,
                             feedback_cases=len(feedback.cases)):
            return self._generator.update_guidance(state, feedback, settings, progress)

    def sample(self, state, request, progress=None):
        enum = lambda value: getattr(value, 'value', value)
        with self._operation('sampling', source=enum(request.kind),
                             purpose=enum(request.purpose),
                             requested_cases=request.count) as outcome:
            records = tuple(self._generator.sample(state, request, progress))
            outcome['returned_cases'] = len(records)
            return records


def _number(value, *, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    if integer and int(value) != value:
        return None
    return int(value) if integer else float(value)


def parse_codex_usage(path: str | Path) -> dict:
    """Sum only per-turn completed usage, never cumulative/session snapshots.

    A missing/partial log is not zero usage. Cached input is a subset of input;
    it is retained separately and never added a second time to token totals.
    """
    path = Path(path)
    totals = dict(input_tokens=0, cached_input_tokens=0, output_tokens=0)
    valid = malformed = invalid_usage = 0
    if not path.is_file():
        return {'status': 'missing', 'completed_turns': 0,
                **dict.fromkeys(totals, None)}
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        try:
            event = json.loads(line)
        except (ValueError, TypeError):
            if line.strip().startswith('{'):
                malformed += 1
            continue
        if not isinstance(event, dict) or event.get('type') != 'turn.completed':
            continue
        usage = event.get('usage')
        if not isinstance(usage, dict):
            invalid_usage += 1
            continue
        values = {key: _number(usage.get(key), integer=True) for key in totals}
        # Some providers omit cache accounting; that field stays unknown.
        if values['input_tokens'] is None or values['output_tokens'] is None:
            invalid_usage += 1
            continue
        if (values['cached_input_tokens'] is not None and
                values['cached_input_tokens'] > values['input_tokens']):
            invalid_usage += 1
            continue
        valid += 1
        for key, value in values.items():
            totals[key] = (totals[key] + value
                           if totals[key] is not None and value is not None else None)
    if not valid:
        totals = dict.fromkeys(totals, None)
    return {'status': ('unavailable' if not valid else
                       'partial' if malformed or invalid_usage else 'observed'),
            'completed_turns': valid, 'malformed_json_lines': malformed,
            'invalid_usage_events': invalid_usage, **totals}


def _optional_list(path):
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    if not isinstance(data, list) or any(not isinstance(row, dict) for row in data):
        raise ValueError(f'{path.name} must contain a list of audit records')
    return data


def _sum_known(rows, field):
    if rows is None:
        return None
    values = [_number(row.get(field)) for row in rows]
    return sum(values) if all(value is not None for value in values) else None


def collect_run_costs(run_root: str | Path, *, pricing: Mapping | None = None,
                      prior_costs: Mapping | None = None) -> dict:
    """Collect full attempts, including failed executions; do not infer prices.

    Explicit pricing: currency/model/as_of plus input_per_million,
    cached_input_per_million and output_per_million. Dollar estimates exclude
    other tools and infrastructure. No prices are built into the repository.
    """
    root = Path(run_root).resolve()
    agents = _optional_list(root / 'execution_audits.json')
    candidates = _optional_list(root / 'candidate_execution_audits.json')
    usage_rows = []
    seen_logs = set()
    for audit in agents or []:
        raw = audit.get('stdout_path')
        if not raw:
            parsed = {'status': 'missing', 'completed_turns': 0,
                      'input_tokens': None, 'cached_input_tokens': None, 'output_tokens': None}
        else:
            log = Path(raw)
            log = log if log.is_absolute() else root / log
            if str(log.resolve()) in seen_logs:
                raise ValueError('duplicate Agent stdout path would double-count usage')
            seen_logs.add(str(log.resolve()))
            parsed = parse_codex_usage(log)
        usage_rows.append({'execution_id': audit.get('execution_id'),
                           'agent_status': audit.get('status'), **parsed})
    token_fields = ('input_tokens', 'cached_input_tokens', 'output_tokens')
    # Empty or partial completion data cannot establish complete billed usage.
    tokens = {key: (_sum_known(usage_rows, key) if usage_rows else None)
              for key in token_fields}
    completed_coverage = bool(usage_rows) and all(
        row['status'] == 'observed' and row['agent_status'] in ('completed', 'ok')
        for row in usage_rows)
    operation_path = root / 'generator_operations.jsonl'
    operation_rows, pending = [], {}
    malformed_operation_lines = 0
    if operation_path.is_file():
        for line in operation_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                malformed_operation_lines += 1
                continue
            identifier = row['operation_id']
            if row['event'] == 'started':
                if identifier in pending:
                    raise ValueError('duplicate Generator operation start')
                pending[identifier] = row
            elif row['event'] == 'finished':
                if identifier not in pending:
                    raise ValueError('Generator operation finish has no start')
                pending.pop(identifier)
                operation_rows.append(row)
            else:
                raise ValueError('unknown Generator operation event')
    operation_summary = []
    for operation in sorted({row['operation'] for row in operation_rows}):
        subset = [row for row in operation_rows if row['operation'] == operation]
        operation_summary.append({'operation': operation, 'calls': len(subset),
                                  'failed_calls': sum(r.get('status') != 'ok' for r in subset),
                                  'sum_service_wall_seconds': _sum_known(subset, 'wall_seconds')})
    started = [_number(row.get('started_at_unix')) for row in (agents or []) + operation_rows]
    finished = [_number(row.get('finished_at_unix')) for row in (agents or []) + operation_rows]
    valid_started = [v for v in started if v is not None]
    valid_finished = [v for v in finished if v is not None]
    span = (max(valid_finished) - min(valid_started)
            if valid_started and valid_finished else None)
    estimate = None
    if pricing is not None:
        for key in ('currency', 'model', 'as_of'):
            if not isinstance(pricing.get(key), str) or not pricing[key].strip():
                raise ValueError(f'pricing needs explicit {key}')
        price_fields = ('input_per_million', 'cached_input_per_million', 'output_per_million')
        prices = {key: _number(pricing.get(key)) for key in price_fields}
        if any(value is None for value in prices.values()):
            raise ValueError('pricing rates must be finite and nonnegative')
        if all(value is not None for value in tokens.values()):
            estimate = ((tokens['input_tokens'] - tokens['cached_input_tokens']) * prices['input_per_million']
                        + tokens['cached_input_tokens'] * prices['cached_input_per_million']
                        + tokens['output_tokens'] * prices['output_per_million']) / 1_000_000
    return {
        'schema': 'run_resource_report_v1', 'run_id': root.name,
        'usage_coverage': ('completed_turns_observed' if completed_coverage else 'incomplete'),
        'tokens': tokens, 'agent_usage': usage_rows,
        'agent_executions': None if agents is None else len(agents),
        'candidate_executions': None if candidates is None else len(candidates),
        'agent_active_seconds_sum': _sum_known(agents, 'active_seconds'),
        'agent_wall_seconds_sum': _sum_known(agents, 'wall_seconds'),
        'controller_service_seconds_sum': _sum_known(agents, 'controller_service_seconds'),
        'candidate_wall_seconds_sum': _sum_known(candidates, 'wall_seconds'),
        'observed_activity_span_seconds': span,
        'generator_operations': operation_summary,
        'generator_incomplete_operations': list(pending.values()),
        'generator_malformed_lines': malformed_operation_lines,
        'generator_measurement_available': operation_path.is_file(),
        'prior_costs': dict(prior_costs) if prior_costs is not None else None,
        'pricing': dict(pricing) if pricing is not None else None,
        'observed_llm_cost': estimate,
        'notes': [
            'Service wall sums overlap and must not be added as end-to-end wall or CPU/GPU compute.',
            'Activity span excludes uninstrumented preparation and idle scheduler time.',
            'Tokens cover logged completed turns; interrupted requests may incur unobserved charges.',
            'Existing checkpoint preparation cost is unknown unless explicitly supplied.',
            'Price estimate excludes tool charges and infrastructure; mixed models require separate reports.',
        ],
    }
