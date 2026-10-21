#!/usr/bin/env python3
"""Freeze reference-conditioned NS evaluation Cases using existing Controller operations.

This prepares private evaluation data only. It never executes a coding Agent or
reads a final solver. Counts and the reference bundle must be explicit.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import copy
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
from threading import Lock

from pde_agent_verifier_solver.engine import CampaignController
from pde_agent_verifier_solver.engine.common_evaluation import content_sha256, json_value
from pde_agent_verifier_solver.engine.seeds import SeedLedger
from pde_agent_verifier_solver.sdk import (
    CaseFeedback, CaseSource, EvaluationPhase, GenerationPurpose, GeneratorFeedback,
    HiddenSuiteSpec, TerminalCohortSpec, TerminalEvaluationSpec, TerminalGuidanceSource,
    TerminalMetricSpec,
)


def equal_suite(per_type: int) -> HiddenSuiteSpec:
    if isinstance(per_type, bool) or not isinstance(per_type, int) or per_type < 2 or per_type % 2:
        raise ValueError('each type needs a positive even count to split P equally across B/G parents')
    return HiddenSuiteSpec(base=per_type, guided=per_type,
                           base_perturbed=per_type // 2, guided_perturbed=per_type // 2)


class SamplingRecorder:
    def __init__(self, generator):
        self.generator = generator
        self.tag = None
        self.requests = {}

    def __getattr__(self, name):
        return getattr(self.generator, name)

    def sample(self, state, request, progress=None):
        records = tuple(self.generator.sample(state, request, progress))
        if self.tag is not None:
            self.requests.setdefault(self.tag, []).append({
                'purpose': request.purpose.value, 'round_index': request.round_index,
                'kind': request.kind.value, 'count': request.count, 'seed': request.seed,
                'anchors': [case.case_id for case in request.anchors],
                'case_ids': [record.case.case_id for record in records],
            })
        return records


def _probe(controller):
    probe = copy(controller)
    probe._seeds = SeedLedger(controller.campaign.master_seed)
    probe._case_samplers = []
    probe._pending_feedback_samplers = []
    probe._terminal_case_samplers = []
    for name in ('_terminal_hidden_case_ids', '_initial_guidance_hidden_case_ids',
                 '_development_case_ids', '_promotion_hidden_case_ids'):
        setattr(probe, name, set())
    probe._development_case_owners = {}
    probe._case_registry_lock = Lock()
    probe._generator_operation_lock = Lock()
    probe._loaded_generator_state = None
    return probe


def _write_bank(probe, path, bank_id, purpose, hidden, namespace, provenance):
    from pde_agent_verifier_solver.plugins.ns.case_bank import write_ns_continuous_case_bank
    phase = EvaluationPhase.OFFICIAL if purpose == 'promotion' else EvaluationPhase.TERMINAL
    seed_for = probe._hidden_plan_seed_provider(namespace, phase, hidden)
    settings = {**probe.campaign.task_settings, 'default_query': probe.campaign.evaluator.default_query}
    plans, strata, context, case_provenance = {}, {}, {}, {}
    records = []
    for distribution, rows in hidden.items():
        stratum = distribution.rsplit('/', 1)[-1]
        for index, record in enumerate(rows):
            case = record.case
            plans[case.case_id] = probe.evaluation.make_plan(case, phase, settings, seed_for(index, case.case_id))
            strata[case.case_id] = stratum
            context[case.case_id] = {'controller_distribution': distribution, 'case_position': index}
            detail = dict(context[case.case_id])
            if case.source is CaseSource.PERTURBED:
                detail['perturbation_audit'] = dict(probe.generator.backend.case_source.controller_perturbation_audit(case.case_id))
            case_provenance[case.case_id] = detail
            records.append(record)
    write_ns_continuous_case_bank(path, bank_id=bank_id, purpose=purpose, records=records,
                                 plans=plans, strata=strata, provenance=provenance,
                                 case_provenance=case_provenance)
    return context


def _write_comparison_bank(full_path, destination, bank_id):
    """Filter the same-directory manifest; share immutable continuous assets."""
    if full_path.parent != destination.parent:
        raise ValueError('comparison bank must share the original bank directory')
    payload = json.loads(full_path.read_text())
    payload.pop('content_sha256')
    payload['bank_id'] = bank_id
    payload['cases'] = [row for row in payload['cases'] if row['stratum'] in {'base', 'base_perturbed'}]
    payload['case_count'] = len(payload['cases'])
    payload['strata_counts'] = dict(Counter(row['stratum'] for row in payload['cases']))
    payload['provenance'] = {**payload['provenance'], 'purpose': 'comparison_promotion',
                             'shared_assets_manifest': full_path.name}
    payload['content_sha256'] = content_sha256(payload)
    with os.fdopen(os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return {row['case']['case_id']: {
                'controller_distribution': row['generation_provenance']['controller_distribution'],
                'case_position': row['generation_provenance']['case_position']}
            for row in payload['cases']}


def _reference_feedback(probe, reference, hidden, round_index):
    # The current NS Guidance consumes only unperturbed Base feedback. Evaluate
    # exactly that stratum; this avoids extra reference-solver evaluations on G/P.
    reports, evaluations = probe._official_reports(
        (reference,), {'base': hidden['base']}, seed_namespace=f'official_round:{round_index}',
    )
    if not reports[0].technically_eligible:
        raise RuntimeError('reference solver failed the Base feedback cohort')
    by_id = {record.case.case_id: record.case for record in hidden['base']}
    results = evaluations[reference.artifact_id]['base'].cases
    return GeneratorFeedback(source_round_index=round_index, target_round_index=round_index + 1,
                             selected_program_id=reference.artifact_id, selected_program=reference,
                             cases=tuple(CaseFeedback(by_id[row.case_id], row.primary_error, row.metrics)
                                         for row in results))


def _preparation_specs(controller, *, promotion_per_type=None, selection_per_type=None,
                       terminal_per_type=None, promotion_from_config=False,
                       selection_suite=None, terminal_from_config=False):
    """Resolve explicit counts without substituting a different paper protocol."""
    if promotion_from_config == (promotion_per_type is not None):
        raise ValueError('choose promotion_from_config or promotion_per_type')
    if terminal_from_config == (terminal_per_type is not None):
        raise ValueError('choose terminal_from_config or terminal_per_type')
    if (selection_suite is None) == (selection_per_type is None):
        raise ValueError('choose selection_suite or selection_per_type')
    rounds = tuple(controller.campaign.rounds)
    if not promotion_from_config:
        counts = equal_suite(promotion_per_type)
        rounds = tuple(replace(plan, round_hidden=counts) for plan in rounds)
    selection = selection_suite if selection_suite is not None else equal_suite(selection_per_type)
    if not isinstance(selection, HiddenSuiteSpec):
        raise TypeError('selection_suite must be HiddenSuiteSpec')
    for counts in (selection, *(plan.round_hidden for plan in rounds)):
        if counts.original or not counts.base:
            raise ValueError('continuous NS reference banks need Base and cannot include original array Cases')
    if terminal_from_config:
        terminal = controller.campaign.terminal_evaluation
        cohorts = {cohort.cohort_id: cohort for cohort in terminal.cohorts}
        if set(cohorts) != {'common_base', 'common_guided'}:
            raise ValueError('configured shared terminal needs common_base and common_guided cohorts')
        base_cohort, guided_cohort = cohorts['common_base'], cohorts['common_guided']
        if (base_cohort.guidance_source is not TerminalGuidanceSource.BASE
                or guided_cohort.guidance_source is not TerminalGuidanceSource.FINAL
                or base_cohort.suite.original or not base_cohort.suite.base):
            raise ValueError('configured shared terminal needs continuous Base and final Guided cohorts')
    else:
        counts = equal_suite(terminal_per_type)
        base_cohort = TerminalCohortSpec('common_base', HiddenSuiteSpec(
            base=counts.base, base_perturbed=counts.base_perturbed))
        guided_cohort = TerminalCohortSpec('common_guided', HiddenSuiteSpec(
            guided=counts.guided, guided_perturbed=counts.guided_perturbed), TerminalGuidanceSource.FINAL)
        terminal = TerminalEvaluationSpec((base_cohort, guided_cohort), TerminalMetricSpec('overall', 'overall'))
    return rounds, selection, terminal, base_cohort, guided_cohort


def prepare(controller, output, *, promotion_per_type=None, selection_per_type=None,
            terminal_per_type=None, promotion_from_config=False,
            selection_suite=None, terminal_from_config=False):
    """Freeze reference banks, an independent selection bank and terminal-only suite.

    The *_from_config options retain actual Campaign counts and terminal cohort
    identities. Legacy equal-per-type calls remain available explicitly. The
    full suite contains reference-conditioned Official banks for qualification;
    terminal_suite.json leaves each live run's Promotion generation unchanged.
    """
    rounds, selection_counts, terminal_spec, base_cohort, guided_cohort = _preparation_specs(
        controller, promotion_per_type=promotion_per_type, selection_per_type=selection_per_type,
        terminal_per_type=terminal_per_type, promotion_from_config=promotion_from_config,
        selection_suite=selection_suite, terminal_from_config=terminal_from_config)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    probe = _probe(controller)
    probe.campaign = replace(controller.campaign, rounds=rounds, terminal_evaluation=terminal_spec)
    reference = probe.campaign.initial_candidate
    if reference is None:
        raise ValueError('evaluation preparation needs an explicit frozen reference program')
    recorder = SamplingRecorder(probe.generator)
    probe.generator = recorder
    settings = {'basis_size': probe.campaign.generator.basis_size,
                **probe.campaign.generator.settings, 'master_seed': probe.campaign.master_seed,
                'controller_seed': probe._seeds.derive('generator', 'base')}
    task_settings = {**probe.campaign.task_settings, 'default_query': probe.campaign.evaluator.default_query}
    suite = {'schema': 'ns_shared_mixed_evaluation_v1', 'status': 'preparing',
             'master_seed': probe.campaign.master_seed,
             'task_settings_sha256': content_sha256(task_settings), 'official_rounds': [],
             'terminal_cohorts': json_value(terminal_spec.cohorts),
             'provenance': {'reference_artifact_id': reference.artifact_id,
                            'conditioning': 'same explicitly supplied frozen reference for every boundary',
                            'official_banks_role': 'reference qualification; live Promotion remains run-local with terminal-only suite',
                            'promotion_per_type': promotion_per_type, 'selection_per_type': selection_per_type,
                            'terminal_per_type': terminal_per_type,
                            'promotion_suites': [{'round_index': plan.round_index, **plan.round_hidden.as_dict()}
                                                for plan in probe.campaign.rounds],
                            'selection_suite': selection_counts.as_dict(),
                            'terminal_suites': {cohort.cohort_id: cohort.suite.as_dict() for cohort in terminal_spec.cohorts},
                            'terminal_guidance_reference_artifact_id': reference.artifact_id,
                            'terminal_guidance_source_round': probe.campaign.rounds[-1].round_index,
                            'terminal_guidance_target_round': probe.campaign.rounds[-1].round_index + 1,
                            'coding_agent_calls': 0}}
    comparison_rounds = []
    comparison_supported = all(plan.round_hidden.guided_perturbed == 0 for plan in probe.campaign.rounds)
    def provenance(state, purpose):
        boundary = state.guidance
        snapshot = getattr(boundary, 'snapshot', None)
        training_request = getattr(snapshot, 'training_request', None)
        return {**suite['provenance'], 'purpose': purpose,
                'guidance_source_round': getattr(training_request, 'source_round_index', None),
                'guidance_target_round': getattr(training_request, 'target_round_index', None),
                'guidance_target_solver_sha256': getattr(training_request, 'target_solver_sha256', None),
                'generator_id': probe.generator.info.program_id,
                'guidance_snapshot_sha256': getattr(snapshot, 'snapshot_sha256', None),
                'task_settings_sha256': suite['task_settings_sha256']}
    try:
        state = probe.generator.load(probe.campaign.generator.checkpoint, settings,
                                     probe._generator_progress(None, 'evaluation_bank_loading'))
        probe._loaded_generator_state = state
        first = probe.campaign.rounds[0]
        recorder.tag = 'terminal'
        terminal_base, _, _ = probe._hidden_suite(state, first, base_cohort.suite,
            GenerationPurpose.TERMINAL, 0, settings, seed_namespace='terminal:common_base', generation_round_index=0)
        recorder.tag = None
        feedback = probe._initial_guidance_feedback(reference, state, first, settings)
        previous_sampler = None
        for plan in probe.campaign.rounds:
            r = plan.round_index
            promotion_counts = plan.round_hidden
            state = probe.generator.update_guidance(state, feedback,
                {**settings, 'controller_seed': probe._seeds.derive('generator', 'guidance', r)},
                probe._generator_progress(r, 'reference_guidance_training'))
            probe._loaded_generator_state = state
            probe._close_case_samplers(probe._pending_feedback_samplers)
            probe._pending_feedback_samplers.clear()
            if previous_sampler is not None:
                previous_sampler.close()
            if r == 1:
                recorder.tag = None
                fixed, _, fixed_sampler = probe._hidden_suite(state, plan, selection_counts,
                    GenerationPurpose.PROMOTION, 0, settings, seed_namespace='fixed_validation')
                _write_bank(probe, output/'fixed_selection.json', 'fixed_selection', 'promotion', fixed,
                            'fixed_validation', provenance(state, 'fixed_selection'))
                suite['fixed_selection_manifest'] = 'fixed_selection.json'
                if fixed_sampler is not None:
                    fixed_sampler.close()
            tag = f'official:{r}'
            recorder.tag = tag
            hidden, _, previous_sampler = probe._hidden_suite(state, plan, promotion_counts,
                GenerationPurpose.PROMOTION, 0, settings)
            recorder.tag = None
            location = output/f'official_round_{r}.json'
            context = _write_bank(probe, location, f'official_round_{r}', 'promotion', hidden,
                                  f'official_round:{r}', provenance(state, 'promotion'))
            suite['official_rounds'].append({'round_index': r, 'manifest': location.name,
                'sampling_requests': recorder.requests[tag], 'evaluation_context': context})
            if comparison_supported:
                common_location = output/f'comparison_official_round_{r}.json'
                common_context = _write_comparison_bank(location, common_location, f'comparison_official_round_{r}')
                comparison_rounds.append({'round_index': r, 'manifest': common_location.name,
                    'sampling_requests': [request for request in recorder.requests[tag] if request['kind'] != 'guided'],
                    'evaluation_context': common_context, 'delegate_kinds': ['guided']})
            feedback = _reference_feedback(probe, reference, hidden, r)
            print(f'Frozen shared Official round {r}: {promotion_counts.case_count} Cases', flush=True)
        state = probe.generator.update_guidance(state, feedback,
            {**settings, 'controller_seed': probe._seeds.derive('generator', 'terminal_final_guidance')},
            probe._generator_progress(None, 'reference_terminal_guidance_training'))
        probe._loaded_generator_state = state
        recorder.tag = 'terminal'
        terminal_guided, _, _ = probe._hidden_suite(state, probe.campaign.rounds[-1], guided_cohort.suite,
            GenerationPurpose.TERMINAL, 0, settings, seed_namespace='terminal:common_guided',
            generation_round_index=probe.campaign.rounds[-1].round_index + 1)
        recorder.tag = None
        terminal = {probe._terminal_distribution_key(cohort.cohort_id, name): rows
                    for cohort, hidden in ((base_cohort, terminal_base), (guided_cohort, terminal_guided))
                    for name, rows in hidden.items()}
        context = _write_bank(probe, output/'terminal.json', 'shared_terminal', 'terminal', terminal,
                              'terminal_evaluation', provenance(state, 'terminal'))
        suite.update(status='frozen', terminal_manifest='terminal.json',
                     terminal_sampling_requests=recorder.requests['terminal'],
                     terminal_evaluation_context=context, seed_ledger=dict(probe._seeds.snapshot()))
        destination = output/'suite.json'
        terminal_suite = {key: value for key, value in suite.items() if key != 'official_rounds'}
        terminal_suite['provenance'] = {**suite['provenance'], 'shared_scope': 'terminal_only'}
        (output/'terminal_suite.json').write_text(json.dumps(json_value(terminal_suite), indent=2, allow_nan=False)+'\n')
        if comparison_supported:
            comparison_suite = {**suite, 'official_rounds': comparison_rounds,
                'provenance': {**suite['provenance'], 'shared_scope': 'official_base_base_perturbed_and_terminal',
                               'official_guided_source': 'run_local_incumbent'}}
            (output/'comparison_suite.json').write_text(json.dumps(json_value(comparison_suite), indent=2, allow_nan=False)+'\n')
        destination.write_text(json.dumps(json_value(suite), indent=2, allow_nan=False)+'\n')
        return destination
    finally:
        probe._close_case_samplers(probe._case_samplers)


def _selection_suite_json(value):
    try:
        return HiddenSuiteSpec(**json.loads(value))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(f'invalid selection suite: {error}') from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--reference-bundle', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    promotion = parser.add_mutually_exclusive_group(required=True)
    promotion.add_argument('--promotion-per-type', type=int)
    promotion.add_argument('--promotion-from-config', action='store_true')
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--selection-per-type', type=int)
    selection.add_argument('--selection-suite', type=_selection_suite_json,
                           help='explicit JSON counts, e.g. {"base":64,"guided":64,"base_perturbed":64}')
    terminal = parser.add_mutually_exclusive_group(required=True)
    terminal.add_argument('--terminal-per-type', type=int)
    terminal.add_argument('--terminal-from-config', action='store_true')
    parser.add_argument('--controller-gpu', required=True)
    parser.add_argument('--worker-gpus', required=True)
    args = parser.parse_args(argv)
    for count in (args.promotion_per_type, args.selection_per_type, args.terminal_per_type):
        if count is not None:
            equal_suite(count)
    from run_canonical_ns_campaign import _load_config, _validate_config, _compose
    config, path = _load_config(args.config)
    # Preparation creates these banks; it cannot consume the run's output paths.
    for key in ('shared_evaluation_suite_manifest', 'official_audit_manifest'):
        config.pop(key, None)
    config['candidate']['initial_bundle'] = str(args.reference_bundle.resolve())
    config['generator']['guidance_enabled'] = True
    config['generator']['perturbations_enabled'] = True
    _validate_config(config)
    run_root = args.output.resolve().with_name(args.output.name+'_reference_runtime')
    run_root.mkdir(parents=True, exist_ok=False)
    composition = _compose(config=config, config_path=path, run_root=run_root,
        codex_binary=Path(os.environ.get('CODEX_BINARY', 'codex')), auth_file=Path(os.environ.get('CODEX_AUTH_FILE', os.path.expanduser('~/.codex/auth.json'))),
        worker_gpus=tuple(args.worker_gpus.split(',')), controller_gpu=args.controller_gpu)
    try:
        controller = CampaignController(campaign=composition.campaign, dataset=composition.dataset,
            problem=composition.problem, task=composition.task, generator=composition.generator,
            agents=composition.agents, agent_execution=composition.execution,
            public_problem_bundle=composition.public_problem_bundle, candidate_execution=composition.candidate_execution)
        result = prepare(controller, args.output, promotion_per_type=args.promotion_per_type,
                         selection_per_type=args.selection_per_type, terminal_per_type=args.terminal_per_type,
                         promotion_from_config=args.promotion_from_config, selection_suite=args.selection_suite,
                         terminal_from_config=args.terminal_from_config)
        print(json.dumps({'suite': str(result), 'sha256': sha256(result.read_bytes()).hexdigest()}))
    finally:
        for service in (composition.execution, composition.candidate_execution):
            close = getattr(service, 'close', None)
            if callable(close):
                close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
