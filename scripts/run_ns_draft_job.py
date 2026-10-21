#!/usr/bin/env python3
"""Thin Slurm dispatch for the approved NS comparison; reuse existing runners."""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATHS = {
    name: f"configs/paper/ns_draft_v1/{name}_s101.json"
    for name in (
        "fixed", "base", "full", "bon", "aide", "full_mean_upfront",
        "full_tail_upfront", "full_tail_adaptive",
    )
}
CONFIG_PATHS.update({
    name: f"configs/paper/ns_gp_v2/{name}_s101.json"
    for name in ("base_guided", "base_perturbed")
})
BASELINE_GPUS = {"fixed": 2, "bon": 1, "aide": 1}
AUTHORIZED_METHODS = ("fixed", "base", "full", "base_guided", "base_perturbed")
PAPER_PROTOCOL = "iclr_ns_bgp_192_single_final_v5"
CONFIG_PATHS.update({name: f"configs/paper/ns_bgp_v3/{name}_s101.json"
                     for name in AUTHORIZED_METHODS})


def _baseline_payload(config_path, tokens):
    """Resolve the declared data paths and assign only the allocated Agent GPUs."""
    payload = json.loads(config_path.read_text())
    for field in ("manifest", "selection_manifest"):
        if payload["dataset"].get(field):
            payload["dataset"][field] = str((config_path.parent / payload["dataset"][field]).resolve())
    if payload.get("initial_candidate"):
        payload["initial_candidate"] = str((config_path.parent / payload["initial_candidate"]).resolve())
    roles = payload["agent"].get("roles")
    if roles is not None:
        for role, token in zip(roles, tokens, strict=True):
            role["gpu_devices"] = [token]
    else:
        if len(tokens) != 1:
            raise ValueError("a serial search job requires one allocated Agent GPU")
        payload["agent"]["gpu_devices"] = [tokens[0]]
    return payload


def _write(path, payload):
    from pde_agent_verifier_solver.engine.checkpoint_store import atomic_bytes
    data = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    atomic_bytes(Path(path), data.encode())


def verify_snapshot():
    manifest=json.loads((ROOT/'snapshot_manifest.json').read_text())
    for entry in manifest['files']:
        path=ROOT/entry['path']
        if not path.is_file():
            raise RuntimeError(f'frozen source changed: {entry["path"]}')


def main(argv=None, *, methods=AUTHORIZED_METHODS, preflight_fn=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('preflight', *methods),required=True)
    parser.add_argument('--batch-root',type=Path,required=True)
    args=parser.parse_args(argv)
    batch=args.batch_root.resolve()
    if ROOT != batch/'source':
        raise ValueError('run only from the frozen batch/source snapshot')
    verify_snapshot()
    tokens=tuple(v.strip() for v in os.environ['NS_JOB_GPU_TOKENS'].split(','))
    need=BASELINE_GPUS.get(args.mode, 3)
    if len(tokens)!=need or len(set(tokens))!=need or not all(tokens):
        raise ValueError(f'{args.mode} requires exactly {need} distinct allocated GPUs')
    if args.mode!='preflight':
        gate=json.loads((batch/'control'/'preflight_passed.json').read_text())
        if gate.get('status')!='passed' or gate.get('protocol_revision') != PAPER_PROTOCOL:
            raise RuntimeError('first-batch scientific/input preflight has not passed')
    job=os.environ['SLURM_JOB_ID']
    config_root=ROOT/'configs/paper/ns_bgp_v3'
    status=batch/'job_status'/f'{args.mode}-{job}.json'
    record=dict(mode=args.mode,job_id=job,status='running',started_at_unix=time.time(),
                source_snapshot=str(ROOT),gpu_tokens=tokens)
    _write(status,record)
    if args.mode == 'preflight':
        result = 1
        try:
            result = (preflight_fn or preflight)(config_root, batch, tokens, job)
            return result
        finally:
            record.update(status='completed' if result == 0 else 'failed', returncode=result,
                          finished_at_unix=time.time())
            _write(status, record)
    return run_segment(args.mode, batch, tokens, job, status, record)


def allocation_remaining_seconds(job):
    """Use the allocated job's actual end time, including wrapper/setup time."""
    import datetime
    raw = os.environ.get('SLURM_JOB_END_TIME')
    if raw and raw.isdigit():
        end = float(raw)
    else:
        reply = subprocess.run(['scontrol', 'show', 'job', str(job), '-o'],
                               check=True, capture_output=True, text=True, timeout=15)
        fields = dict(item.split('=', 1) for item in reply.stdout.split() if '=' in item)
        end = datetime.datetime.fromisoformat(fields['EndTime']).timestamp()
    seconds = end - time.time()
    if seconds <= 600:
        raise RuntimeError('allocation has insufficient time for safe setup and checkpoint flush')
    return seconds


def _checkpoint_progress(checkpoint, workflow_phase):
    """Compare completed scientific work, excluding saves, RNG and elapsed time."""
    payload = json.loads(checkpoint.read_text())
    if payload.get('schema') == 'common_evaluation_checkpoint_v1':
        progress = payload['completed']
    elif payload.get('schema') == 'campaign_checkpoint_v1' and 'progress' in payload:
        progress = payload['progress']
    else:
        raise ValueError('checkpoint has no scientific progress descriptor')
    return sha256(json.dumps({'phase': workflow_phase, 'progress': progress},
                             sort_keys=True, allow_nan=False).encode()).hexdigest()


def run_segment(mode, batch, tokens, job, status_path, record):
    from pde_agent_verifier_solver.engine.campaign_continuation import CampaignPaused
    from pde_agent_verifier_solver.engine.checkpoint_store import RunSegmentBudget
    from ns_slurm_continuation import (
        runner_lease, publish_outcome, submit_next, SubmissionUncertainError,
    )
    budget, output = None, None
    owns_run = False
    outcome = 'failed'
    checkpoint = None
    try:
        expected_parent = (batch / 'runs' / f'{mode}-s101').resolve()
        supplied = os.environ.get('NS_CONTINUATION_RUN_ROOT')
        output = (Path(supplied).resolve() if supplied else expected_parent / f'attempt-{job}')
        if output.parent != expected_parent:
            raise ValueError('continuation run root must belong to this frozen batch and method')
        index = int(os.environ.get('NS_CONTINUATION_SEGMENT_INDEX', '1'))
        maximum = int(os.environ.get('NS_CONTINUATION_MAX_SEGMENTS', '12'))
        scientific = output / 'scientific'
        workflow_path = output / 'workflow.json'
        record.update(run_root=str(output), scientific_run_root=str(scientific), segment_index=index)
        _write(status_path, record)
        with runner_lease(output, str(job), index, max_segments=maximum):
            owns_run = True
            # Record allocation/setup failures under the acquired run lease;
            # a rejected second writer must never overwrite the owner's status.
            budget = RunSegmentBudget(allocation_remaining_seconds(job))
            with budget.signals():
                workflow = (json.loads(workflow_path.read_text()) if workflow_path.is_file() else
                            {'mode': mode, 'phase': 'scientific', 'source_snapshot': str(ROOT)})
                if (workflow['mode'] != mode or workflow['source_snapshot'] != str(ROOT)
                        or workflow.get('phase') not in {'scientific', 'final_evaluation', 'completed'}):
                    raise ValueError('continuation workflow identity or phase differs')
                _write(workflow_path, workflow)
                try:
                    if workflow['phase'] == 'scientific':
                        if mode in BASELINE_GPUS:
                            from pde_agent_verifier_solver.plugins.ns.baselines.cli import run_from_config
                            config = ROOT / CONFIG_PATHS[mode]
                            effective = batch / 'control' / f'{mode}_effective_{job}.json'
                            _write(effective, _baseline_payload(config, tokens))
                            checkpoint = scientific / 'baseline_checkpoints' / 'checkpoint.json'
                            result = run_from_config(effective, scientific,
                                                     resume=checkpoint.is_file(), segment_budget=budget)
                            bundle = result.report['selected_bundle']
                        else:
                            from run_canonical_ns_campaign import main as canonical_main
                            checkpoint = scientific / 'checkpoints' / 'checkpoint.json'
                            argv = ['--config', str(ROOT / CONFIG_PATHS[mode]), '--run-root', str(scientific),
                                    '--worker-gpus', ','.join(tokens[:2]), '--controller-gpu', tokens[2]]
                            if checkpoint.is_file():
                                argv.append('--resume')
                            code = canonical_main(argv, segment_budget=budget)
                            if code == 75:
                                raise CampaignPaused(stage='scientific')
                            if code != 0:
                                raise RuntimeError(f'canonical runner failed: exit {code}')
                            bundle = str(scientific / 'selected_candidate')
                        workflow.update(phase='final_evaluation', selected_bundle=str(bundle))
                        _write(workflow_path, workflow)
                    if workflow['phase'] == 'final_evaluation':
                        device_map = {}
                        if mode == 'fixed':
                            from pde_agent_verifier_solver.plugins.ns.baselines.cli import allocated_evaluation_devices
                            device_map = allocated_evaluation_devices(tokens)
                        elif mode not in BASELINE_GPUS:
                            from pde_agent_verifier_solver.engine.controller_gpu_visibility import configure_controller_evaluation_devices
                            if os.environ.get('CUDA_VISIBLE_DEVICES'):
                                device_map = configure_controller_evaluation_devices(tokens[2], tokens[:2])
                        checkpoint = output / 'common_evaluation' / 'checkpoint.json'
                        final = evaluate_final_bundle(
                            mode, workflow['selected_bundle'], output, batch,
                            evaluation_devices=tuple(device_map.values()), segment_budget=budget,
                            resume=checkpoint.is_file(),
                        )
                        if final.get('status') == 'paused':
                            if Path(final['checkpoint_path']).resolve() != checkpoint.resolve():
                                raise ValueError('final evaluation returned an unexpected checkpoint path')
                            raise CampaignPaused(stage='final_evaluation')
                        if final.get('status') != 'completed':
                            raise RuntimeError('final evaluation did not produce a complete report')
                        workflow.update(phase='completed')
                        _write(workflow_path, workflow)
                    outcome = 'completed'
                    publish_outcome(output, status=outcome)
                except CampaignPaused as pause:
                    if not checkpoint or not checkpoint.is_file():
                        raise RuntimeError('safe pause has no committed scientific checkpoint') from pause
                    progress = _checkpoint_progress(checkpoint, workflow['phase'])
                    if progress == workflow.get('last_pause_progress'):
                        raise RuntimeError('continuation made no scientific progress; automatic continuation stopped') from pause
                    workflow['last_pause_progress'] = progress
                    _write(workflow_path, workflow)
                    outcome = 'paused'
                    publish_outcome(output, status=outcome, checkpoint_path=checkpoint)
                record.update(status=outcome, returncode=0, finished_at_unix=time.time(),
                              workflow_phase=workflow['phase'])
                _write(status_path, record)
                _write(output / 'status.json', record)
    except BaseException as error:
        record.update(status='failed', returncode=1, error_type=type(error).__name__,
                      error=str(error), finished_at_unix=time.time())
        _write(status_path, record)
        if owns_run:
            _write(output / 'status.json', record)
        raise
    finally:
        if budget is not None:
            budget.close()
    if outcome == 'paused':
        try:
            next_state = submit_next(
                output, slurm_script=ROOT / 'scripts/slurm_ns_draft_first_batch.sh',
                script_args=[mode, str(batch), sys.executable], gpu_count=len(tokens),
                max_segments=maximum,
            )
            record['continuation_status'] = next_state.get('status')
        except SubmissionUncertainError as error:
            # sbatch may have accepted a successor. Exit successfully so that
            # its afterok dependency can run; the durable intent prevents retry.
            record.update(continuation_status='submission_unknown', continuation_error=str(error))
        except Exception as error:
            record.update(continuation_status='stopped', continuation_error=str(error))
            _write(status_path, record)
            _write(output / 'status.json', record)
            raise
        _write(status_path, record)
        _write(output / 'status.json', record)
    return 0


def validate_prepared_banks(directory, round_count=5):
    """Check the actual scientific counts and disjoint frozen Case inventories."""
    from pde_agent_verifier_solver.engine.common_evaluation import content_sha256
    directory = Path(directory)
    suite = json.loads((directory/'comparison_suite.json').read_text())
    if suite.get('status') != 'frozen' or suite.get('master_seed') != 101:
        raise ValueError('shared evaluation suite is not frozen at seed101')
    if [row['round_index'] for row in suite['official_rounds']] != list(range(1, round_count+1)):
        raise ValueError('shared Base/P banks must cover every round')
    observed = set()
    checks = []
    def check(name, counts, purpose):
        path = directory/name
        data = json.loads(path.read_text())
        digest = data.pop('content_sha256')
        if data['purpose'] != purpose:
            raise ValueError('bank identity or evaluation purpose differs')
        if data['strata_counts'] != counts or data['case_count'] != sum(counts.values()):
            raise ValueError(f'bank does not have the approved Case counts: {name}')
        ids = {row['case']['case_id'] for row in data['cases']}
        if len(ids) != data['case_count'] or ids & observed:
            raise ValueError('evaluation banks contain repeated Cases')
        observed.update(ids)
        checks.append({'manifest': name, 'sha256': sha256(path.read_bytes()).hexdigest(),
                       'counts': counts, 'purpose': purpose})
    for row in suite['official_rounds']:
        if row.get('delegate_kinds') != ['guided']:
            raise ValueError('Official Guided must remain conditioned on each run incumbent')
        check(row['manifest'], {'base': 64, 'base_perturbed': 64}, 'promotion')
    check('fixed_selection.json', {'base':64, 'guided':64, 'base_perturbed':64}, 'promotion')
    check(suite['terminal_manifest'], {'base':256, 'guided':256,
                                      'base_perturbed':256}, 'terminal')
    return checks


def _require_finite_selection(evaluation, records):
    if tuple(row.case_id for row in evaluation.cases) != tuple(row.case.case_id for row in records):
        raise RuntimeError('S0 mixed selection returned incomplete or reordered Case results')
    if not math.isfinite(evaluation.primary_error) or any(
            not math.isfinite(row.primary_error) for row in evaluation.cases):
        raise RuntimeError('S0 mixed selection returned nonfinite errors')


def evaluate_final_bundle(method, bundle, run_root, batch, *, evaluation_devices=(),
                          resume=False, segment_budget=None):
    """Use the exact same hidden terminal bank for every delivered program."""
    from pde_agent_verifier_solver.plugins.ns.common_evaluation import run_ns_common_evaluation
    manifest = {
        'schema':'ns_common_evaluation_v1', 'task_id':'ns-full-field-solver',
        'datasets':[{'dataset_id':'ns-bgp-common-terminal',
                     'manifest':str(batch/'control/mixed_evaluation/terminal.json'),
                     'cohort':'bgp_common'}],
        'programs':[{'method':method, 'run_id':run_root.name, 'seed':101,
                     'artifact_id':f'{method}-final', 'role':'final', 'bundle':str(bundle)}],
        'query':{'point_count':128, 'include_pressure':False},
        'budget':{'max_workers':28, 'case_hard_wall_seconds':900}}
    path = run_root/'common_evaluation_manifest.json'
    _write(path, manifest)
    options = ({"resume": resume, "segment_budget": segment_budget}
               if resume or segment_budget is not None else {})
    if evaluation_devices:
        from pde_agent_verifier_solver.sdk.evaluation_devices import (
            EvaluationDevicePool, bind_evaluation_device_pool,
        )
        # Candidate workers remain CPU-only; the pool routes public Oracle
        # queries. All plugin allocations close before this function returns.
        with EvaluationDevicePool(evaluation_devices) as pool:
            with bind_evaluation_device_pool(pool):
                return run_ns_common_evaluation(path, run_root/'common_evaluation',
                                                **options)
    else:
        return run_ns_common_evaluation(path, run_root/'common_evaluation',
                                        **options)


def preflight(config_root,batch,tokens,job):
    from run_canonical_ns_campaign import _load_config,_validate_config,_compose
    from prepare_ns_mixed_evaluation import prepare
    from pde_agent_verifier_solver.engine import CampaignController
    from pde_agent_verifier_solver.engine.evaluator import HiddenVerifier
    from pde_agent_verifier_solver.plugins.ns.case_bank import load_ns_continuous_case_bank
    from pde_agent_verifier_solver.reporting.costs import MeasuredGenerator
    from pde_agent_verifier_solver.sdk import GenerationPurpose
    workspace=batch/'preflight'/f'attempt-{job}'
    workspace.mkdir(parents=True,exist_ok=False)
    composition=None
    try:
        for name in AUTHORIZED_METHODS:
            if name != 'fixed':
                config,path=_load_config(config_root/f'{name}_s101.json')
                _validate_config(config)
        config,path=_load_config(config_root/'full_s101.json')
        run=workspace/'reference_runtime'; run.mkdir()
        composition=_compose(config=config,config_path=path,run_root=run,
                             codex_binary=Path(os.environ.get('CODEX_BINARY', 'codex')),
                             auth_file=Path(os.environ.get('CODEX_AUTH_FILE', os.path.expanduser('~/.codex/auth.json'))),
                             worker_gpus=tokens[:2],controller_gpu=tokens[2])
        composition.generator=MeasuredGenerator(composition.generator, workspace/'generator_operations.jsonl')
        controller=CampaignController(campaign=composition.campaign,dataset=composition.dataset,
            problem=composition.problem,task=composition.task,generator=composition.generator,
            agents=composition.agents,agent_execution=composition.execution,
            public_problem_bundle=composition.public_problem_bundle,
            candidate_execution=composition.candidate_execution)
        prepared=workspace/'mixed_evaluation'
        prepare(controller,prepared,promotion_from_config=True,terminal_from_config=True,
                selection_suite=composition.campaign.rounds[0].round_hidden)
        checks=validate_prepared_banks(prepared)
        bank=load_ns_continuous_case_bank(prepared/'fixed_selection.json',
                                         expected_purpose=GenerationPurpose.PROMOTION,device='cpu')
        settings={**composition.campaign.task_settings,
                  'default_query':composition.campaign.evaluator.default_query}
        verifier=HiddenVerifier(bank.fixed_evaluation(composition.task.create_evaluation(settings)),
            composition.campaign.evaluator.default_query,settings,seed=101,
            generator=bank.generator,generator_state=bank.generator_state,
            candidate_execution=composition.candidate_execution)
        evaluation=verifier.evaluate(composition.campaign.initial_candidate,bank.records)
        _require_finite_selection(evaluation, bank.records)
        _write(workspace/'s0_mixed_selection.json', {
            'status':'passed', 'case_count':len(evaluation.cases),
            'primary_error':evaluation.primary_error,
            'cases':[{'case_id':row.case_id,'primary_error':row.primary_error} for row in evaluation.cases],
            'scope':'Initial reference on independent mixed selection bank; no final-test scores.'})
        legacy=config_root.parent/'ns_draft_v1'
        manifest={
            'schema':'ns_common_evaluation_v1','task_id':'ns-full-field-solver','require_initial':True,
            'datasets':[{'dataset_id':'ns-draft-calibration','manifest':str(legacy/'data/calibration_manifest.json'),
                         'split':'calibration','cohort':'calibration'}],
            'programs':[{'method':'S0','run_id':'reference-initial','seed':0,'artifact_id':'common-s0','role':'initial',
                         'bundle':str(legacy/'s0_hold_initial_bundle')}],
            'query':{'point_count':128,'seed':12345,'include_pressure':False},
            'budget':{'max_workers':28,'case_hard_wall_seconds':900}}
        manifest_path=workspace/'s0_calibration_manifest.json'; _write(manifest_path,manifest)
        result=subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_ns_programs.py'),
                               '--manifest',str(manifest_path),'--output',str(workspace/'s0_calibration')],check=False)
        if result.returncode:
            raise RuntimeError(f'S0 original-input calibration failed: exit{result.returncode}')
        calibration=json.loads((workspace/'s0_calibration/summary.json').read_text())
        if calibration.get('status') != 'completed' or calibration.get('n_failed_program_cases') != 0:
            raise RuntimeError('S0 original-input calibration contains failed Cases')
        published=batch/'control/mixed_evaluation'
        if published.exists():
            raise FileExistsError('a completed mixed evaluation bank already exists; use a new attempt')
        prepared.rename(published)
        validate_prepared_banks(published)
        _write(batch/'control/preflight_passed.json', {
            'job_id':job,'status':'passed','protocol_revision':PAPER_PROTOCOL,
            'workspace':str(workspace),'comparison_suite':str(published/'comparison_suite.json'),
            'bank_checks':checks,'completed_at_unix':time.time(),
            'scope':'Real Generator loading, S0-conditioned training and frozen Base/P plus terminal banks; full mixed-selection and original calibration S0 subprocess/RPC checks. No coding Agent and no final-test scores.'})
        return 0
    finally:
        if composition is not None:
            for service in (composition.execution,composition.candidate_execution):
                close=getattr(service,'close',None)
                if callable(close): close()


if __name__=='__main__': raise SystemExit(main())
