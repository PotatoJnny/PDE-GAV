"""Freeze numerical-comparison cohorts without generating or retraining Cases."""
import csv,json,shutil
from pathlib import Path
from collections import defaultdict
from dataclasses import replace
import numpy as np
import torch
from pde_agent_verifier_solver.plugins.ns.case_bank import write_ns_continuous_case_bank, _case
from pde_agent_verifier_solver.plugins.ns.continuous_backend import StructuredNSMMSAnswer, _NSVelocityTruth
from pde_agent_verifier_solver.sdk import CaseRecord,CaseVisibility,AnswerQuery,EvaluationPlan,EvaluationPhase
from agent_verifier.neural_generator.structured_ns_answer_first import ContinuousStructuredNSMMS,StructuredNSCaseCoefficients,StructuredNSProjectionConfig
REPO=Path(__file__).resolve().parents[2]
ROOT=REPO/'.paper_runs/ns_numerical_baselines_20260924'

def freeze():
    if (ROOT/'control/cohorts.json').exists():
        print('cohorts already frozen');return
    old=Path('/tmp/ns_v6_received_20260921')
    target=ROOT/'assets';target.mkdir(exist_ok=True)
    shutil.copy2(old/'control/promotion24.json',target/'promotion24.json')
    shutil.copytree(old/'control/promotion24.cases',target/'promotion24.cases',dirs_exist_ok=True)
    source=old/'controller_private/common_initial'
    residents=json.loads((source/'checkpoint.json').read_text())['generator_runtime']['residents']
    index=json.loads((old/'control/common_initial.json').read_text())['case_ids']
    resident={r['case']['case_id']:r for r in residents}
    records=[];plans={};strata={}
    for key in index[:4]:
        r=resident[key];raw=torch.load(source/'objects'/r['field']['asset'],map_location='cpu',weights_only=False)
        field=ContinuousStructuredNSMMS(StructuredNSCaseCoefficients(psi=raw['psi'],pressure=raw['pressure']),
            config=StructuredNSProjectionConfig.from_dict(raw['config']),geometry_spec=raw['geometry'],
            continuous_mms_abi=raw['abi'],domain_bounds=tuple(raw['bounds']),viscosity=raw['viscosity'],
            t0=raw['t0'],t1=raw['t1'],device='cpu')
        case=replace(_case(r['case']),visibility=CaseVisibility.HIDDEN)
        answer=StructuredNSMMSAnswer(field)
        record=CaseRecord(case,_NSVelocityTruth(answer));records.append(record)
        rng=np.random.default_rng(20260924)
        g=case.public_data['domain']['geometry'];x0,x1,y0,y1=g['domain_bounds'];points=[]
        while len(points)<128:
            x,y=rng.uniform([x0,y0],[x1,y1])
            inside=False
            for poly in g['obstacle_polygons']:
                z=np.asarray(poly);edge=np.roll(z,-1,axis=0)-z;v=np.array([x,y])-z
                cross=edge[:,0]*v[:,1]-edge[:,1]*v[:,0]
                inside=inside or bool((cross>=0).all() or (cross<=0).all())
            if not inside:points.append({'t':float(rng.uniform(raw['t0'],raw['t1'])),'x':float(x),'y':float(y)})
        query=AnswerQuery.at_points(fields=('velocity_x','velocity_y'),points=points)
        plans[key]=EvaluationPlan(EvaluationPhase.OFFICIAL,query,seed=20260924,settings={})
        strata[key]='base'
    write_ns_continuous_case_bank(target/'obstacle_development4.json',bank_id='numerical-calibration-common64-first4',
        purpose='promotion',records=records,plans=plans,strata=strata,
        provenance={'role':'calibration_only','source':str(source),'case_ids':index[:4],
          'changes':'Same stored continuous fields; hidden calibration views, no resampling of fields.'})
    piv=REPO/'.paper_runs/ns_measured_cylinder_fixed_full_20260911_r2/control/mixed_evaluation'
    rng=np.random.default_rng(20260924)
    cohorts={}
    for stage,filename,count in [('terminal','terminal.json',32),('validation','fixed_selection.json',8)]:
        bank=json.loads((piv/filename).read_text());groups=defaultdict(list)
        for r in bank['cases']:groups[r['stratum']].append(r['case']['case_id'])
        ids=[];remainder=[]
        for stratum in sorted(groups):
            order=rng.permutation(len(groups[stratum]))
            ids.extend(groups[stratum][i] for i in order[:count]);remainder.extend(groups[stratum][i] for i in order[count:])
        cohorts['piv_'+stage]={'manifest':str(piv/filename),'purpose':bank['purpose'],'case_ids':ids,'seed':20260924}
        if stage=='validation':cohorts['piv_calibration']={**cohorts['piv_validation'],'case_ids':remainder[:4]}
    for stage,path in [('calibration',target/'obstacle_development4.json'),('validation',target/'promotion24.json'),
        ('terminal',REPO/'.incoming/collaborator/independent96_20260921/data/controller_private/test_bank.json')]:
        bank=json.loads(path.read_text());cohorts['obstacle_'+stage]={'manifest':str(path),'purpose':bank['purpose'],'case_ids':[r['case']['case_id'] for r in bank['cases']]}
    for task in ('obstacle','piv'):
        ids=[set(cohorts[task+'_'+s]['case_ids']) for s in ('calibration','validation','terminal')]
        assert not (ids[0]&ids[1] or ids[0]&ids[2] or ids[1]&ids[2])
        assert [len(i) for i in ids]==[4,24,96]
    (ROOT/'control/cohorts.json').write_text(json.dumps(cohorts,indent=2)+'\n')
    # Recompute existing PIV results on exactly the frozen 96 IDs.
    rows=list(csv.DictReader((REPO/'docs/paper/iclr_results/consolidated_20260924/main_cases_768.csv').open(encoding='utf-8-sig')))
    selected=[r for r in rows if r['case_id'] in set(cohorts['piv_terminal']['case_ids'])]
    out=ROOT/'control/piv_existing_same96.csv'
    with out.open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(selected)
    groups=defaultdict(list)
    for r in selected:groups[r['method']].append(r)
    summaries=[]
    for name,rs in groups.items():
        assert len(rs)==96
        v=np.array([float(r['error_percent']) for r in rs if r['status']=='ok'])
        summaries.append({'method':name,'cases':len(rs),'success':len(v),'mean_percent':float(v.mean()),'tail20_percent':float(np.sort(v)[-int(np.ceil(len(v)*.2)):].mean())})
    assert len(summaries)==2
    (ROOT/'control/piv_existing_same96.json').write_text(json.dumps(summaries,indent=2)+'\n')
    print(json.dumps({'cohorts':{k:len(v['case_ids']) for k,v in cohorts.items()},'piv_existing':summaries}))
if __name__=='__main__':freeze()
