"""CPU-only before/after regularity and integral balance audit. No asset rewrite."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tests'),str(ROOT/'legacy/ns/src'),str(ROOT/'src')]
from test_ns_geometry_v6 import field,seam,mask_crossing,pillbox_balance,V5,V6,mms,torch,np
from agent_verifier.neural_generator.structured_ns_analytic import analytic_quantities


def audit(old):
    assert old.continuous_mms_abi==V5
    new=mms.ContinuousStructuredNSMMS(
        old.coefficients,config=old.config,geometry_spec=old.geometry_spec,continuous_mms_abi=V6,
        domain_bounds=old.domain_bounds,viscosity=old.viscosity,t0=old.t0,t1=old.t1,device='cpu')
    anchor,normal=seam(old)
    versions={}
    for f in (old,new):
        rows=[]
        for epsilon in (1e-4,1e-6,1e-8,1e-10):
            values=mask_crossing(f,anchor,normal,epsilon)
            xmin,xmax,ymin,ymax=f.domain_bounds
            xy=anchor.numpy()[None,:]+np.array([-epsilon,epsilon])[:,None]*normal.numpy()
            xy[:,0]=xmin+xy[:,0]*(xmax-xmin);xy[:,1]=ymin+xy[:,1]*(ymax-ymin)
            points=np.column_stack((xy,np.full(2,.047)))
            q=analytic_quantities(f,points)
            rows.append({'normalized_offset':epsilon,'mask_normal_derivatives_0_to_3':values.tolist(),
                         'mask_derivative_gaps':(values[:,1]-values[:,0]).tolist(),
                         'velocity':[q['u'].tolist(),q['v'].tolist()],
                         'forcing':[q['forcing_u'].tolist(),q['forcing_v'].tolist()],
                         'forcing_gap_l2':float(torch.hypot(q['forcing_u'][1]-q['forcing_u'][0],q['forcing_v'][1]-q['forcing_v'][0]))})
        versions[f.continuous_mms_abi]={'crossings':rows,'pillbox':[pillbox_balance(f,half_normal=h,order=24) for h in (1e-3,1e-4,1e-5)]}
    assert abs(versions[V5]['crossings'][-1]['mask_derivative_gaps'][2])>4000
    assert abs(versions[V6]['crossings'][-1]['mask_derivative_gaps'][2])<1e-5
    assert all(x['residual_l2']<1e-9 for x in versions[V6]['pillbox'])
    return versions


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-dir',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    report={'checked_at':datetime.now(timezone.utc).isoformat(),'cpu_only':True,
            'probe_scope':'old and fixed mask, full NS forcing, conservative momentum integral over an internal seam',
            'synthetic_constant_streamfunction':audit(field(V5,constant=True))}
    if args.case_dir:
        path=args.case_dir/'hidden_continuous_case.pt'
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        old=mms.load_structured_continuous_case(args.case_dir,device='cpu')
        report['actual_case']={'path':str(path.resolve()),'state_sha256':digest,
                               'same_coefficients_for_v6_probe':True,'results':audit(old)}
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(str(args.output.resolve()))
    for name,block in [('synthetic',report['synthetic_constant_streamfunction']),('actual_case',report.get('actual_case',{}).get('results'))]:
        if block:
            for abi,data in block.items():
                print(name,abi,'d2 gap at 1e-8:',data['crossings'][2]['mask_derivative_gaps'][2],
                      'pillbox residual at halfwidth 1e-4:',data['pillbox'][1]['residual_l2'])

if __name__=='__main__':main()
