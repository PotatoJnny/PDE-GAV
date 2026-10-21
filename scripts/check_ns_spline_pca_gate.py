#!/usr/bin/env python
"""Paired physical velocity fidelity at fixed PCA ranks, before Flow training."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'legacy/ns/src'),str(ROOT/'scripts')]
import numpy as np
import torch
from prepare_ns_reconstructed_generator import velocity_from_coefficients,relative_errors,summary,write_json
from agent_verifier.neural_generator.structured_ns_distribution import StructuredCoefficientPCA


def run(args):
    torch.set_num_threads(args.threads)
    result={};refs=[]
    for label,path in [('old',Path(args.old)),('new',Path(args.new))]:
        cfg=json.loads((path/'continuous_contract.json').read_text())
        with np.load(path/'validation_reference.npz',allow_pickle=False) as f:
            ref={key:f[key].copy() for key in ('x','y','time','coefficients','measured')}
        refs.append(ref)
        if label=='new':
            for key in ('x','y','time','measured'):
                if not np.array_equal(refs[0][key],ref[key]):raise ValueError('paired reference mismatch')
        pca=StructuredCoefficientPCA.load(path/'pca_checkpoint.pt')
        settings={'x':ref['x'],'y':ref['y'],'times':ref['time'],'geometry':cfg['geometry'],'projection_config':cfg['projection_config']}
        storage={key:[] for key in ('continuous_to_measured',*[f'pca{r}_to_measured' for r in (16,32,64)],*[f'pca{r}_to_continuous' for r in (16,32,64)])}
        for start in range(0,len(ref['coefficients']),32):
            c=ref['coefficients'][start:start+32];truth=ref['measured'][start:start+32]
            full=velocity_from_coefficients(c,**settings)
            storage['continuous_to_measured'].extend(relative_errors(full,truth))
            for rank in (16,32,64):
                values=pca.with_rank(rank).reconstruct(c).detach().numpy()
                pred=velocity_from_coefficients(values,**settings)
                storage[f'pca{rank}_to_measured'].extend(relative_errors(pred,truth))
                storage[f'pca{rank}_to_continuous'].extend(relative_errors(pred,full))
        result[label]={'prepared':str(path),'config':cfg['projection_config'],'errors':{key:summary(vals) for key,vals in storage.items()}}
    a,b=result['old']['errors'],result['new']['errors']
    result['pass']=b['continuous_to_measured']['mean']<a['continuous_to_measured']['mean'] and all(b[f'pca{r}_to_measured']['mean']<=a[f'pca{r}_to_measured']['mean'] for r in (32,64))
    result['rule']='new continuous fit and fixed PCA ranks 32/64 improve paired measured-velocity fidelity; own-continuous PCA errors also reported but different continuous targets are not a common truth'
    result['scope']='existing generator-validation diagnostic, previously used for selection; no verifier/final evaluation'
    write_json(args.out,result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--old',required=True);p.add_argument('--new',required=True);p.add_argument('--out',required=True);p.add_argument('--threads',type=int,default=8)
    run(p.parse_args())
