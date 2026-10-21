#!/usr/bin/env python3
"""Freeze the three approved NS seed-101 configurations; never submit jobs."""
from copy import deepcopy
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INPUT_ABI = """Deliver one solver supporting both public input representations. Generated
Cases expose domain and query_initial_velocity/query_forcing via the public
session. Original numerical Cases expose public_data['question'] and ['arrays']:
x[X], y[Y], t[T], initial_velocity[2,X,Y], forcing[T,2,X,Y], fluid_mask[X,Y].
Use these public inputs to solve the SAME declared NS equation; interpolate
finite arrays when necessary. The supplied S0 demonstrates both input paths.
Return a prepared handle evaluated at physical (t,x,y) points with velocity_x
and velocity_y. Preserve support for both input types when editing. No hidden
trajectory, reference files, selection data or test data may be inspected.
Selection minimizes untruncated mean u/v-component relative L2 on a complete,
finite validation cohort; execution/nonfinite failures make that program
ineligible, and exact ties retain the incumbent. Do not optimize pressure error.
"""


def main():
    directory = ROOT / 'configs/paper/ns_draft_v1'
    source = ROOT / 'configs/ns_canonical_structured_5round.json'
    original = json.loads(source.read_text())
    common = deepcopy(original)
    common['master_seed'] = 101
    # Count the entire turn, including Controller/Generator waits.
    common['execution'].pop('agent_active_budget_seconds', None)
    common['execution']['agent_hard_wall_seconds'] = 14400
    common['dataset']['manifest'] = 'data/development_manifest.json'
    common['dataset']['settings'].update(development_split='development', hidden_split='validation',
                                          development_cases=16, hidden_cases=8)
    common['generator']['artifacts'] = {
        k: str((source.parent / v).resolve()) for k, v in original['generator']['artifacts'].items()}
    common['candidate']['initial_bundle'] = 's0_hold_initial_bundle'
    common['candidate']['initial_summary'] = 'Common hold-initial velocity solver supporting finite original arrays and generated public oracles.'
    common['worker']['script'] = '../../../scripts/run_canonical_codex_agent.py'
    common['worker']['codex_config'] = '../../../scripts/canonical_codex.config.toml'
    common['official'] = dict(original=0, base=1024, guided=0, base_perturbed=0, guided_perturbed=0)
    common['terminal'] = {
        'cohorts': [{'cohort_id':'common_base', 'guidance_source':'base',
                     'suite':dict(original=0,base=256,guided=0,base_perturbed=0,guided_perturbed=0)}],
        'ranking':{'cohort_id':'common_base','distribution':'base','statistic':'mean'}}
    common['promotion'] = {'preset':'minimize_metric','settings':{'primary':{'distribution':'base','statistic':'mean'}}}
    # Relative to the frozen config; preflight fills this private evidence bank.
    common['official_audit_manifest'] = '../../../../control/official_s101.json'
    records=[]
    with (ROOT/'docs/paper/SUBMISSION_EXECUTION_RUNS.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    for method, label, enabled in [('base','base-new-cases',False),('full','full-framework',True)]:
        config=deepcopy(common)
        config['campaign_id']=f'ns-draft-hardwall-v2-{method}-s101'
        config['generator'].update(guidance_enabled=enabled, perturbations_enabled=enabled)
        config['development']['allowed_sources']=['base','guided'] if enabled else ['base']
        config['development']['starter']=dict(original=0,base=16 if enabled else 32,guided=16 if enabled else 0,base_perturbed=0,guided_perturbed=0)
        for agent in config['agents']:
            # Task remains the original role; only tool availability and goal differ.
            role='Explore physically distinct NS numerical methods.' if 'explorer' in agent['agent_id'] else 'Diagnose and refine numerical stability, boundary treatment and integration.'
            tools=('Base, Guided and Perturbed' if enabled else 'Base only; Guided and Perturbed are disabled')
            agent['prompt']=f'{role}\n{INPUT_ABI}\nDevelopment tools: {tools}. Each round starts with 32 Cases and allows at most 512 total new Cases per Agent, including starter and perturbed children. Choose your own timing, quantities and experiments. Use the previous selected program and cumulative Campaign Memory; submit all distinct runnable candidates worth common evaluation.'
        name=f'{method}_s101.json'
        (directory/name).write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
        row=next(r for r in rows if r['experiment']=='01' and r['task']=='NS' and r['method']==label and r['seed']=='101')
        records.append({'method':method,'kind':'canonical','config':name,'csv_run_id':row['run_id'], 'gpus':3})
    baseline={
        'schema':'ns_fixed_data_baseline_v1','method':'competitive_refinement','seed':101,
        'dataset':{'manifest':'data/development_manifest.json','development_split':'development','selection_split':'validation'},
        'initial_candidate':'s0_hold_initial_bundle','search':{'competition_rounds':5},
        'prompt':INPUT_ABI+'\nUse the fixed 512 development Cases and available evaluation tool; this run has no online case-generation or public oracle sessions. Both Agents compete on the same fixed 512 validation Cases at each round boundary. Retain all useful runnable nominations and cumulative logs.',
        'agent':{
            'command':['{python}','{repo}/scripts/run_canonical_codex_agent.py','--codex-binary','${CODEX_BINARY}','--auth-file','${CODEX_AUTH_FILE}','--runtime-root','{output}/agent_runtime','--config','{repo}/scripts/canonical_codex.config.toml'],
            'hard_wall_seconds':14400,'inline_array_bytes':256,
            'worker_settings':{'profile':'structured_numerical','model':'gpt-5.6-sol','reasoning_effort':'max','candidate_abi':'prepare_solution_then_evaluate_points_v1','candidate_execution':{'device':'cpu','cuda_available':False},'controller_rpc_timeout_seconds':7200},
            'environment':{'OMP_NUM_THREADS':'8','MKL_NUM_THREADS':'8'},
            'roles':[{'agent_id':a['agent_id'],'name':a['name'],'prompt':('Explore physically distinct NS numerical methods.' if 'explorer' in a['agent_id'] else 'Diagnose and refine numerical stability, boundary treatment and integration.'),'gpu_devices':[str(i)]} for i,a in enumerate(original['agents'])]},
        'evaluation':{'allow_point_truth':False,'allow_grid_truth':False,'development_coordinate_mode':'private_random_points','development_point_count':32,'selection_point_count':96,'candidate_case_workers':28,'candidate_case_hard_wall_seconds':900,'development_batch_hard_wall_seconds':6900}}
    (directory/'fixed_s101.json').write_text(json.dumps(baseline,ensure_ascii=False,indent=2)+'\n')
    row=next(r for r in rows if r['experiment']=='01' and r['task']=='NS' and r['method']=='fixed-data-iteration' and r['seed']=='101')
    records.insert(0, {'method':'fixed','kind':'baseline','config':'fixed_s101.json','csv_run_id':row['run_id'],'gpus':2})
    (directory/'first_batch.json').write_text(json.dumps({'schema':'ns_draft_first_batch_v1','protocol_revision':'iclr_hard_wall_4h_single_seed_v2','time_budget_mode':'hard_wall','agent_hard_wall_seconds':14400,'execution_status':'paused_not_submitted','seed':101,'entries':records,'failure_protocol':'complete_finite_selection_raw_external_v1','data_index':'data/bank_index.json'},ensure_ascii=False,indent=2)+'\n')
    print('Prepared fixed/Base/Full seed101 configs; no jobs submitted.')


if __name__=='__main__': main()
