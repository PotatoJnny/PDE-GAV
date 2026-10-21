#!/usr/bin/env python3
"""Freeze, then calibrate one spectral-P epsilon on fixed paired training cases.

This reads three training coefficient rows per representation, never PCA/Flow or
validation fields. The common finite uniform query grid is a diagnostic, not an
exact physical integral. A single epsilon is selected and actually re-evaluated.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

INDICES = (0, 446, 892)
SEEDS = (101, 202, 303)
OLD_EPSILON = 2.861761125944777e-6


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def freeze(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    script = out / Path(__file__).name
    shutil.copyfile(__file__, script)
    origin = Path(args.workspace_root).resolve() / 'legacy/ns/src'
    snapshot = out / 'source_snapshot/legacy/ns/src'
    shutil.copytree(origin, snapshot, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    code = {str(path.relative_to(out)): sha(path) for path in sorted(snapshot.rglob('*')) if path.is_file()}
    expected_path = Path(args.expected_config).resolve()
    expected = json.loads(expected_path.read_text())
    if (expected['spatial_mode_x'], expected['spatial_mode_y'], expected['time_order']) != (26, 26, 20):
        raise ValueError('this frozen calibration is for the requested 27x27x21 representation')
    plan = {'schema':'ns_paired_spectral_p_calibration_v1',
        'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
        'script':str(script), 'script_sha256':sha(script), 'source_sha256':code,
        'old_prepared':str(Path(args.old).resolve()), 'new_prepared':str(Path(args.new).resolve()),
        'new_expected_config':expected, 'new_expected_config_file_sha256':sha(expected_path),
        'indices':list(INDICES), 'seeds':list(SEEDS), 'expected_training_cases':893,
        'old_epsilon':OLD_EPSILON, 'new_probe_epsilon':OLD_EPSILON,
        'old_spectral_max_mode':20, 'new_spectral_max_mode':28,
        'family':'spectral_complement', 'spatial_points':33, 'time_points':9,
        'grid':'uniform physical rectangle and normalized time endpoints, equal weights over fluid sites and both velocity components',
        'metric':'arithmetic mean over nine directions of delta_velocity_RMS / own_base_velocity_RMS',
        'epsilon_rule':'one global new_epsilon = new_probe_epsilon * old_mean_ratio / new_probe_mean_ratio; actually evaluate all nine again',
        'scope':'finite-grid offline calibration; not exact integration, not solver quality, not online per-case normalization',
        'exclusions':['PCA','Flow','generator_validation','development','verifier','final_test','reconstruction_validation'],
        'verification':{'relative_mean_tolerance':1e-9,'absolute_mean_tolerance':1e-12,
                        'initial_velocity_absolute_tolerance':1e-12,'initial_time_derivative_absolute_tolerance':1e-10}}
    write(out/'plan.json', plan)
    print(json.dumps({'event':'frozen_before_new_coefficients_read', 'plan':str(out/'plan.json'),
                      'script_sha256':plan['script_sha256']}), flush=True)


def selected_corpus(directory, plan):
    import numpy as np
    directory = Path(directory)
    manifest_path = directory/'coefficients/train/coefficient_manifest.json'
    preparation_path = directory/'preparation_config.json'
    manifest = json.loads(manifest_path.read_text())
    preparation = json.loads(preparation_path.read_text())
    if (manifest.get('schema') != 'ns2d_structured_coefficient_dataset_v2'
            or manifest.get('split') != 'train' or manifest['encoded_cases'] != plan['expected_training_cases']):
        raise ValueError('unexpected training corpus identity/count')
    if preparation.get('input_mode') != 'measured' or preparation.get('pressure_model') != 'zero':
        raise ValueError('calibration requires the requested direct measured p=0 corpus')
    signature = manifest['encoding_signature']
    if signature['projection_config'] != preparation['projection_config']:
        raise ValueError('preparation and training corpus configurations differ')
    rows, offset, files = {}, 0, {str(manifest_path):sha(manifest_path), str(preparation_path):sha(preparation_path)}
    for shard in manifest['shards']:
        requested = [i for i in plan['indices'] if offset <= i < offset+shard['count']]
        if requested:
            path = manifest_path.parent/shard['path']
            digest = sha(path)
            files[str(path)] = digest
            with np.load(path, allow_pickle=False) as arrays:
                coefficients, ids, families = arrays['coefficients'], arrays['case_ids'], arrays['families']
                if coefficients.shape != (shard['count'], *manifest['coefficient_shape']):
                    raise ValueError('coefficient shard shape differs')
                if len(ids) != shard['count'] or len(families) != shard['count']:
                    raise ValueError('coefficient row identities are misaligned')
                for index in requested:
                    local = index-offset
                    values = np.ascontiguousarray(coefficients[local], dtype=np.float32)
                    if not np.isfinite(values).all() or np.count_nonzero(values[1]):
                        raise ValueError('selected coefficient field is nonfinite or pressure is not zero')
                    rows[index] = {'index':index, 'case_id':str(ids[local]), 'family':str(families[local]),
                        'coefficient_sha256':hashlib.sha256(values.tobytes()).hexdigest(), 'values':values.copy(),
                        'shard':str(path), 'shard_row':local}
        offset += shard['count']
    if offset != manifest['encoded_cases'] or set(rows) != set(plan['indices']):
        raise ValueError('selected training indices are unavailable')
    return rows, signature, preparation, files


def run(args):
    import numpy as np
    import torch
    out = Path(args.out).resolve()
    plan = json.loads((out/'plan.json').read_text())
    if Path(__file__).resolve() != Path(plan['script']) or sha(__file__) != plan['script_sha256']:
        raise ValueError('run the exact frozen script recorded in plan.json')
    if (out/'calibration.json').exists():
        raise FileExistsError('refusing to overwrite completed calibration')
    for name, digest in plan['source_sha256'].items():
        if sha(out/name) != digest:
            raise ValueError(f'frozen source changed: {name}')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(out/'source_snapshot/legacy/ns/src'))
    from agent_verifier.neural_generator.structured_ns_answer_first import (
        ContinuousStructuredNSMMS, StructuredNSProjectionConfig, StructuredNSCaseCoefficients,
    )
    from agent_verifier.neural_generator.structured_ns_robustness import (
        make_perturbation, field_at_amplitude, structured_perturbation_velocity_metrics,
    )
    torch.set_num_threads(1)
    started = time.perf_counter()
    old, old_contract, old_preparation, old_files = selected_corpus(plan['old_prepared'], plan)
    new, new_contract, new_preparation, new_files = selected_corpus(plan['new_prepared'], plan)
    if new_contract['projection_config'] != plan['new_expected_config']:
        raise ValueError('new representation differs from the pre-frozen calibration plan')
    old_config = old_contract['projection_config']
    if (old_config['spatial_mode_x'], old_config['spatial_mode_y'], old_config['time_order'], old_config['spatial_extension']) != (18,18,12,.075):
        raise ValueError('old corpus is not the requested v2 representation')
    for key in ('geometry','time_interval','split_manifest_sha256'):
        if old_contract[key] != new_contract[key]:
            raise ValueError(f'paired corpora differ in physical/source contract: {key}')
    if old_preparation['viscosity'] != new_preparation['viscosity']:
        raise ValueError('paired corpora viscosities differ')
    for index in plan['indices']:
        if (old[index]['case_id'], old[index]['family']) != (new[index]['case_id'], new[index]['family']):
            raise ValueError(f'paired case identity differs at index {index}')
    def field(row, contract, prep):
        config = StructuredNSProjectionConfig.from_dict(contract['projection_config'])
        values = row['values']; nt = config.time_coefficient_count
        return ContinuousStructuredNSMMS(
            StructuredNSCaseCoefficients(values[0].reshape(nt,-1),values[1].reshape(nt,-1)),
            config=config, geometry_spec=contract['geometry'], domain_bounds=tuple(contract['geometry']['domain_bounds']),
            viscosity=prep['viscosity'], t0=contract['time_interval'][0], t1=contract['time_interval'][1], device='cpu')
    old_fields = {i:field(old[i],old_contract,old_preparation) for i in plan['indices']}
    new_fields = {i:field(new[i],new_contract,new_preparation) for i in plan['indices']}
    old_p, new_p, records = {}, {}, []
    def measure(f, p, epsilon):
        result = structured_perturbation_velocity_metrics(f,p,amplitude=epsilon,
            spatial_points=plan['spatial_points'],time_points=plan['time_points'])
        if result['base_velocity_rms'] <= 0 or result['delta_velocity_relative_rms'] <= 0:
            raise ValueError('calibration requires nonzero finite base/delta velocity RMS')
        if result['initial_velocity_delta_max_abs'] > plan['verification']['initial_velocity_absolute_tolerance']:
            raise AssertionError('perturbation changed initial velocity beyond float64 tolerance')
        return result
    for index in plan['indices']:
        for seed in plan['seeds']:
            key=(index,seed)
            old_p[key]=make_perturbation(old_fields[index],family=plan['family'],seed=seed,
                                          spectral_max_mode=plan['old_spectral_max_mode'])
            new_p[key]=make_perturbation(new_fields[index],family=plan['family'],seed=seed,
                                          spectral_max_mode=plan['new_spectral_max_mode'])
            records.append({'index':index,'seed':seed,'case_id':old[index]['case_id'],
                'old':measure(old_fields[index],old_p[key],plan['old_epsilon']),
                'new_probe':measure(new_fields[index],new_p[key],plan['new_probe_epsilon'])})
            print(json.dumps({'event':'paired_probe','index':index,'seed':seed}),flush=True)
    mean = lambda stage:float(np.mean([r[stage]['delta_velocity_relative_rms'] for r in records]))
    old_mean, probe_mean = mean('old'),mean('new_probe')
    epsilon = plan['new_probe_epsilon']*old_mean/probe_mean
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('invalid calibrated epsilon')
    for row in records:
        index,seed=row['index'],row['seed'];f=new_fields[index];p=new_p[index,seed]
        row['new_verified']=measure(f,p,epsilon)
        xmin,xmax,ymin,ymax=f.domain_bounds
        points=np.array([[xmin,(ymin+ymax)/2,f.t0],[(xmin+xmax)/2,(ymin+ymax)/2,f.t0],[xmax,(ymin+ymax)/2,f.t0]])
        a=f.quantities(points);b=field_at_amplitude(f,p,amplitude=epsilon).quantities(points)
        row['initial_dt_query_points_xyt']=points.tolist()
        row['new_initial_velocity_time_derivative_max_abs']=max(float((a[k]-b[k]).abs().max()) for k in ('d_t_u','d_t_v'))
        if row['new_initial_velocity_time_derivative_max_abs'] > plan['verification']['initial_time_derivative_absolute_tolerance']:
            raise AssertionError('perturbation changed initial velocity time derivative')
        print(json.dumps({'event':'verified_new_epsilon','index':index,'seed':seed}),flush=True)
    verified_mean=mean('new_verified')
    np.testing.assert_allclose(verified_mean,old_mean,rtol=plan['verification']['relative_mean_tolerance'],
                               atol=plan['verification']['absolute_mean_tolerance'])
    identities=lambda data:[{k:v for k,v in data[i].items() if k!='values'} for i in plan['indices']]
    report={'schema':'ns_paired_spectral_p_calibration_result_v1','success':True,
        'plan_sha256':sha(out/'plan.json'),'plan':plan,'old_contract':old_contract,'new_contract':new_contract,
        'viscosity':old_preparation['viscosity'],'input_sha256':{**old_files,**new_files},
        'old_selected_rows':identities(old),'new_selected_rows':identities(new),
        'old_epsilon':plan['old_epsilon'],'epsilon_new':epsilon,'epsilon_ratio_new_to_old':epsilon/plan['old_epsilon'],
        'old_mean_relative_velocity_rms':old_mean,'new_probe_mean_relative_velocity_rms':probe_mean,
        'new_verified_mean_relative_velocity_rms':verified_mean,
        'verified_mean_relative_mismatch':abs(verified_mean-old_mean)/old_mean,
        'directions':records,'direction_count':len(records),
        'initial_conditions':'velocity verified over the common grid; initial velocity time derivative additionally verified at three stated points, both unchanged within recorded float64 tolerances',
        'online_per_case_normalization':False,'exact_physical_integral':False,
        'environment':{'numpy':np.__version__,'torch':torch.__version__,'device':'cpu','torch_threads':1},
        'elapsed_seconds':time.perf_counter()-started}
    write(out/'calibration.json',report)
    write(out/'selected_perturbation.json',{'perturbation_family':plan['family'],
        'perturbation_amplitude':epsilon,'spectral_max_mode':plan['new_spectral_max_mode'],
        'calibration_report_sha256':sha(out/'calibration.json')})
    print(json.dumps({'event':'complete','epsilon_new':epsilon,'old_mean':old_mean,
        'new_verified_mean':verified_mean,'elapsed_seconds':report['elapsed_seconds']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('freeze','run'));parser.add_argument('--out',required=True)
    parser.add_argument('--old');parser.add_argument('--new');parser.add_argument('--expected-config')
    parser.add_argument('--workspace-root',default=str(Path(__file__).resolve().parents[1]))
    args=parser.parse_args()
    if args.mode=='freeze':
        if not all((args.old,args.new,args.expected_config)):parser.error('freeze requires --old, --new, --expected-config')
        freeze(args)
    else:run(args)
