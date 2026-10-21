import argparse,json,shutil
from pathlib import Path
from pde_agent_verifier_solver.sdk.candidate_bundle import write_candidate_bundle_manifest,inspect_candidate_bundle
SRC=Path(__file__).resolve().parent;ROOT=SRC.parents[1]/'.paper_runs/ns_numerical_baselines_20260924'
FILES=['submission.py','wire.py','backend.py','mac.py','fem.py','nektar.py','nektar_session.py','DOKKEN_LICENSE']
def freeze(method_filter=None):
    runtime=json.loads((ROOT/'control/runtime.json').read_text());path=ROOT/'control/profiles.json';rows=json.loads(path.read_text()) if path.exists() else {}
    for method,task,levels in [('mac','piv',[(64,160),(128,320),(256,640)]),
        ('fem','piv',[(48,160),(72,320),(108,640)]),('fem','obstacle',[(48,80),(72,160),(108,320)]),
        ('nektar','obstacle',[(24,240),(32,480)])]:
        if method_filter and method!=method_filter:continue
        if not (ROOT/'qualification'/f'{method}.json').exists():continue
        for level,(resolution,steps) in enumerate(levels):
            name=f'{method}_{task}_L{level}';bundle=ROOT/'bundles'/name;bundle.mkdir(parents=True,exist_ok=True)
            if (bundle/'candidate_bundle.json').exists():raise FileExistsError('profile already frozen: '+name)
            for f in FILES:shutil.copy2(SRC/f,bundle/f)
            config={**runtime,'method':method,'task':task,'resolution':resolution,'steps':steps,'quadrature_degree':6,
                    'modes':5+2*level,'quadrature_extra':2,'force_quadrature_factor':2}
            if method=='fem':config['python']=runtime['fenics_python']
            (bundle/'profile.json').write_text(json.dumps(config,indent=2)+'\n')
            write_candidate_bundle_manifest(bundle,files=FILES+['profile.json'])
            tree=inspect_candidate_bundle(bundle)
            rows[name]={'method':method,'task':task,'profile':config,'bundle':str(bundle),'tree_sha256':tree.tree_sha256}
    (ROOT/'control/profiles.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('frozen',len(rows),'profiles')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--method');a=p.parse_args();freeze(a.method)
