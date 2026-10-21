#!/usr/bin/env python3
"""Prepare original-method measured NS inputs and clone the active Fixed/Full protocol."""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT/'src', ROOT/'legacy/ns/src'):
    sys.path.insert(0, str(folder))
ORIGINAL = ROOT/'project_runs/ns_measured_piv_generators_20260911_v1'
FINAL = ROOT/'project_runs/ns_measured_piv_final_20260911_v1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1<<20), b''): h.update(part)
    return h.hexdigest()


def write(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def choose_windows(rows, length, count=512, seed=101):
    """Only declared development recordings/ranges; no score-dependent selection."""
    intervals={}
    for row in rows:
        rec=row['source_recording'];lo,hi=row['source_frames']
        if rec in intervals: intervals[rec]=[min(lo,intervals[rec][0]),max(hi,intervals[rec][1])]
        else: intervals[rec]=[lo,hi]
    if len(intervals)!=1:
        raise ValueError('this frozen study expects one development recording per dataset')
    recording,(lo,hi)=next(iter(intervals.items()))
    starts=random.Random(seed).sample(list(range(lo,hi-length+1)),count)
    return recording, sorted(starts), intervals


def prepare_data(dataset, output):
    import numpy as np
    import torch
    from pde_agent_verifier_solver.plugins.ns.discrete_measured import write_discrete_measured_case
    from pde_agent_verifier_solver.plugins.ns.experimental_mms_project import ExperimentalMMSProblem, save_prepared_mms_dataset
    from pde_agent_verifier_solver.plugins.ns.reconstruction.real_piv import load_dense_piv_hdf5
    from pde_agent_verifier_solver.sdk import DatasetInput, DatasetKind
    torch.set_num_threads(16)
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    old=ORIGINAL/dataset;prepared=old/'prepared'
    contract=json.loads((prepared/'continuous_contract.json').read_text())
    plan=json.loads((old/'run_plan_cpu.json').read_text())
    source=Path(plan['domain']['source_manifest']);manifest=json.loads(source.read_text())
    rows=manifest['splits']['development']
    with np.load(prepared/'validation_reference.npz',allow_pickle=False) as f:
        x,y,times=(f[k].copy() for k in ('x','y','time'))
    length=len(times);recording,starts,intervals=choose_windows(rows,length)
    low,high=intervals[recording]
    # Check actual raw-frame overlap against all other scientific roles.
    for role,others in manifest['splits'].items():
        if role=='development':continue
        for row in others:
            if row['source_recording']==recording:
                a,b=row['source_frames']
                if max(a,low)<min(b,high):raise ValueError(f'development range overlaps {role}')
    print(json.dumps({'stage':'development_source','dataset':dataset,'range':intervals,'windows':len(starts)}),flush=True)
    if dataset=='cylinder':
        # Existing full-reference windows tile the complete development interval.
        velocity=np.empty((high-low,2,len(x)),dtype=np.float32);covered=np.zeros(high-low,dtype=bool)
        sources=[]
        for row in rows:
            path=Path(row['measurement_reference_file']);sources.append({'path':str(path),'sha256':sha(path)})
            with np.load(path,allow_pickle=False) as f:
                lookup={(float(a),float(b)):i for i,(a,b) in enumerate(zip(f['x'],f['y']))}
                ix=np.array([lookup[(float(a),float(b))] for a,b in zip(x,y)])
                values=f['velocity'][:,:,ix]
            a,b=row['source_frames'];velocity[a-low:b-low]=values;covered[a-low:b-low]=True
        if not covered.all():raise ValueError('source references do not cover cylinder development interval')
    elif dataset=='airfoil':
        path=Path('${PDE_GAV_RUN_ROOT}/datasets/realpde_airfoil/extracted/train_real')/(recording+'.h5')
        rec=load_dense_piv_hdf5(path,spatial_stride=2)
        lookup={(float(a),float(b)):i for i,(a,b) in enumerate(zip(rec.x,rec.y))}
        ix=np.array([lookup[(float(a),float(b))] for a,b in zip(x,y)])
        velocity=rec.velocity[low:high,:,ix].copy()
        sources=[{'path':str(path),'sha256':sha(path)}]
        # Check the exact velocity values against every existing development reference.
        for row in rows:
            with np.load(row['measurement_reference_file'],allow_pickle=False) as f:
                mapping={(float(a),float(b)):i for i,(a,b) in enumerate(zip(f['x'],f['y']))}
                j=np.array([mapping[(float(a),float(b))] for a,b in zip(x,y)])
                a,b=row['source_frames']
                np.testing.assert_array_equal(velocity[a-low:b-low],f['velocity'][:,:,j])
    else:raise ValueError(dataset)
    if velocity.shape!=(high-low,2,len(x)) or not np.isfinite(velocity).all():
        raise ValueError(f'invalid native velocity {velocity.shape}')
    entries=[];window_rows=[];diagnostics=[]
    for start in starts:
        case_id=f'{dataset}-fixed-discrete-{recording}-{start:05d}'
        case_dir=output/'fixed_development.cases'/case_id
        provenance={'case_id':case_id,'source_recording':recording,'source_frames':[start,start+length],
                    'source_manifest':str(source),'physical_settings':plan['mms']}
        diagnostics.append(write_discrete_measured_case(case_dir,case_id=case_id,
            velocity=velocity[start-low:start-low+length],point_x=x,point_y=y,times=times,
            viscosity=plan['mms']['viscosity'],provenance=provenance))
        entries.append({'case_id':case_id,'case_dir':str(case_dir),'family':'measured'})
        window_rows.append({k:provenance[k] for k in ('case_id','source_recording','source_frames')})
        if len(entries)%64==0:
            print(json.dumps({'stage':'fixed_discrete_arrays','complete':len(entries),'total':len(starts)}),flush=True)
    write(output/'fixed_development.json',{'schema':'ns_d0_v1',
        'dataset_id':f'{dataset}-fixed-discrete-development',
        'reference_kind':'measured_trajectory_derived_discrete_mms',
        'source_manifest':str(source),'splits':{'development':entries}})
    del entries,velocity
    # Generated-only comparison keeps original observation test splits archived.
    data=ExperimentalMMSProblem().prepare_dataset(DatasetInput(f'{dataset}-original-generator',
        DatasetKind.EXPERIMENTAL,location=str(prepared/'prepared_dataset.json')))
    keep={r.case.case_id for r in (*data.development_cases,*data.generator_training_cases,*data.generator_validation_cases)}
    summary=dict(data.training_summary)
    summary['split_counts']={**summary['split_counts'],'verifier':0,'final_test':0}
    summary['terminal_final_test_cases']=0;summary['official_hidden_cases']=0
    summary['source_provenance']={k:v for k,v in summary.get('source_provenance',{}).items() if k in keep}
    summary['comparison_scope']='Original generator training/validation; generated-only B/G/P comparison; raw verifier/final remain archived in source dataset.'
    runner=replace(data,hidden_cases=(),training_summary=summary)
    save_prepared_mms_dataset(runner,output/'runner_dataset.json')
    report={'dataset':dataset,'input_mode':'measured','source_manifest':str(source),
        'source_manifest_sha256':sha(source),'sources':sources,'development_range':intervals,
        'window_selection':'random.Random(101).sample(all valid starts in original development range,512), sorted',
        'development_case_count':512,'windows':window_rows,'overlapping_windows':True,
        'no_other_role_frames':True,'fixed_input_contract':'original_numerical_finite_arrays',
        'fixed_continuous_projection':False,'fixed_diagnostics':diagnostics,
        'generator_projection_config':contract['projection_config'],'original_generator_prepared':str(prepared),
        'pca_checkpoint_sha256':sha(prepared/'pca_checkpoint.pt'),
        'artifacts':{n:{'path':str(output/n),'sha256':sha(output/n)} for n in ('fixed_development.json','runner_dataset.json')}}
    write(output/'data_preparation.json',report)
    print(json.dumps({'status':'completed','report':str(output/'data_preparation.json')}),flush=True)


INPUT_NOTE='''NS task and public inputs
-------------------------
Build a solver for the declared incompressible Navier--Stokes initial-boundary
problem. All Cases expose domain and public query_initial_velocity(points_xy),
query_forcing(points_txy), and query_boundary_velocity(points_txy) Oracles.
The measurement window is an open rectangle: its boundary velocity can be
nonzero and time dependent. Use the boundary declared in each Case; do not
impose the original numerical container's zero wall condition.
Return the same prepare_solution handle evaluated at physical (t,x,y) points.
Measured trajectories and continuous coefficients are private preparation data,
not public solution inputs. Pressure is a manufactured convention and unscored.
'''


def adapt_prompt(value):
    start=value.index('NS task and public inputs')
    end=value.index('Selection objective',start)
    return value[:start]+INPUT_NOTE+'\n'+value[end:]


def discrete_fixed_config(numerical_config, manifest):
    """Reuse numerical Fixed, changing only dataset and the declared boundary."""
    fixed=json.loads(json.dumps(numerical_config))
    fixed['dataset'].pop('kind',None)
    fixed['dataset']['manifest']=str(Path(manifest).resolve())
    fixed['agent']['worker_settings']['profile']='structured_numerical'
    fixed['prompt']=fixed['prompt'].replace('Original numerical Cases expose',
        'Original measured finite-array development Cases expose').replace(
        'fixed 512 original finite-array development Cases',
        'fixed 512 original measured finite-array development Cases')
    fixed['prompt'] += """

Measured input boundary
-----------------------
Development Cases provide finite arrays only. In addition to x, y, t,
initial_velocity, forcing and fluid_mask, boundary_indices[Nb,2] identifies
perimeter grid nodes, boundary_points[Nb,2] gives their (x,y), and
boundary_velocity[T,2,Nb] supplies their prescribed velocity. The measurement
window has nonzero, time-dependent Dirichlet velocity; use these boundary
arrays, not zero walls. Raw measured velocity is retained privately for
evaluation, without continuous projection. Forcing uses the same numerical
D0 finite-difference MMS operator, with the explicitly declared manufactured
zero-pressure convention; pressure is unscored. No continuous development
session or new-case sampling is available.
Hidden generated selection/test Cases retain continuous public initial and
forcing oracles and also expose query_boundary_velocity(points_txy) for the
declared boundary trace. The solver must support both input representations.
"""
    return fixed


def prepare_batch(dataset, batch, reference_batch):
    batch=Path(batch).resolve();reference_batch=Path(reference_batch).resolve()
    batch.mkdir(parents=True,exist_ok=False)
    for name in ('control','job_status','logs','runs','preflight','monitoring'): (batch/name).mkdir()
    src=batch/'source';oldsrc=reference_batch/'source'
    shutil.copytree(oldsrc,src)
    # Keep the exact active shared engine; only NS data/ABI adapters are overlaid.
    overlay=[]
    for folder in ('src/pde_agent_verifier_solver/plugins/ns','legacy/ns/src/agent_verifier/neural_generator'):
        for p in (ROOT/folder).rglob('*.py'):
            rel=p.relative_to(ROOT);target=src/rel
            if not target.exists() or sha(target)!=sha(p):
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);overlay.append(str(rel))
    for name in ('run_canonical_ns_campaign.py','run_ns_draft_job.py','run_ns_measured_job.py',
                 'prepare_ns_measured_fixed_full.py','prepare_ns_mixed_evaluation.py'):
        rel=Path('scripts')/name;shutil.copy2(ROOT/rel,src/rel);overlay.append(str(rel))
    shared=[]
    for p in (oldsrc/'src/pde_agent_verifier_solver/engine').rglob('*.py'):
        rel=p.relative_to(oldsrc);assert sha(p)==sha(src/rel);shared.append(str(rel))
    data=FINAL/dataset/'adapter'
    p=ORIGINAL/dataset/'prepared'
    flow=(p/'base_generator/checkpoints/flow_step_0005000.pt' if dataset=='cylinder'
          else FINAL/'airfoil/base_generator/checkpoints/flow_step_0005000.pt')
    cfgdir=src/'configs/paper/ns_bgp_v3'
    original_full=json.loads((oldsrc/'configs/paper/ns_bgp_v3/full_s101.json').read_text())
    full=json.loads(json.dumps(original_full))
    full.update(campaign_id=f'ns-measured-original-{dataset}-full-s101',profile='experimental_mms')
    full['dataset']={'dataset_id':f'{dataset}-measured-original','manifest':str(data/'runner_dataset.json'),'settings':{}}
    full['generator']['generator_id']=f'{dataset}-measured-original-flow'
    full['generator']['generator_sha256']=sha(flow)
    full['generator']['artifacts']={'flow_checkpoint':str(flow),'pca_checkpoint':str(p/'pca_checkpoint.pt')}
    full['generator']['artifact_sha256']={k:sha(Path(v)) for k,v in full['generator']['artifacts'].items()}
    full['generator']['runtime'].pop('time_interval');full['generator']['runtime'].pop('viscosity')
    for agent in full['agents']: agent['prompt']=adapt_prompt(agent['prompt'])
    full['candidate']['initial_summary']='Same hold-initial S0 as numerical experiment01, using declared public NS inputs.'
    fixed=discrete_fixed_config(json.loads((oldsrc/'configs/paper/ns_bgp_v3/fixed_s101.json').read_text()),
                                data/'fixed_development.json')
    write(cfgdir/'full_s101.json',full);write(cfgdir/'fixed_s101.json',fixed)
    # Every controller/search budget and scientific evaluation setting stays identical.
    keys=('execution','round_count','master_seed','development','official','terminal','promotion','evaluator')
    for key in keys: assert full[key]==original_full[key],key
    assert full['generator']['guidance']==original_full['generator']['guidance']
    runtime_old={k:v for k,v in original_full['generator']['runtime'].items() if k not in ('time_interval','viscosity')}
    assert full['generator']['runtime']==runtime_old
    for key in ('search','evaluation','seed'):
        assert fixed[key]==json.loads((oldsrc/'configs/paper/ns_bgp_v3/fixed_s101.json').read_text())[key]
    wrapper=src/'scripts/slurm_ns_draft_first_batch.sh'
    wrapper.write_text(wrapper.read_text().replace('scripts/run_ns_draft_job.py --mode','scripts/run_ns_measured_job.py --mode'))
    registry=json.loads((FINAL/'frozen_protocol.json').read_text())
    write(batch/'control/measured_experiment.json',{'dataset':dataset,'reference_batch':str(reference_batch),
        'reference_protocol':'iclr_ns_bgp_192_single_final_v5','methods':['fixed','full'],
        'adapter_directory':str(data),'generator':registry['datasets'][dataset],
        'scope':'original measured Generator to Fixed versus Full,5 rounds; no new representation tuning',
        'shared_engine_files_identical':shared,'ns_adapter_overlay':overlay})
    # Delete copied metadata only; regenerate a truthful frozen-source inventory.
    inventory=[]
    for file in sorted(src.rglob('*')):
        if file.is_file() and file.name!='snapshot_manifest.json' and '__pycache__' not in file.parts and file.suffix!='.pyc':
            inventory.append({'path':str(file.relative_to(src)),'sha256':sha(file),'bytes':file.stat().st_size})
    write(src/'snapshot_manifest.json',{'schema':'ns_draft_source_snapshot_v1',
        'source_root':str(ROOT),'source_base_commit':json.loads((oldsrc/'snapshot_manifest.json').read_text())['source_base_commit'],
        'source_status':'Active numerical frozen release plus explicitly listed measured NS adapters; scientific method remains original.',
        'created_at_unix':time.time(),'reference_source_manifest_sha256':sha(oldsrc/'snapshot_manifest.json'),
        'files':inventory})
    write(batch/'control/configuration_verification.json',{'passed':True,'dataset':dataset,
        'matching_full_sections':list(keys)+['generator.guidance','generator.runtime except dataset time/viscosity'],
        'matching_fixed_sections':['search','evaluation','seed','agent budget/roles/model/nomination policy'],
        'shared_engine_file_count':len(shared),'shared_engine_identical':True,
        'source_snapshot_sha256':sha(src/'snapshot_manifest.json'),
        'configs':{name:sha(cfgdir/f'{name}_s101.json') for name in ('fixed','full')}})
    print(json.dumps({'batch_root':str(batch),'status':'prepared','source_files':len(inventory)}))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['data','batch'],required=True)
    p.add_argument('--dataset',choices=['cylinder','airfoil'],required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reference-batch',type=Path)
    a=p.parse_args()
    if a.stage=='data':prepare_data(a.dataset,a.out)
    else:
        if a.reference_batch is None:p.error('--reference-batch is required for batch stage')
        prepare_batch(a.dataset,a.out,a.reference_batch)
if __name__=='__main__':main()
