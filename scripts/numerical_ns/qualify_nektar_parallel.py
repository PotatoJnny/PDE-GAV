"""Four concurrent single-rank native solves, including post-solve evaluation."""
import argparse,json,math,os,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    root=parser.parse_args().root.resolve()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('CPU allocation required')
    out=root/'qualification'/'nektar-serial-mpi'
    out.mkdir(exist_ok=False)
    expected=json.loads((root/'qualification/nektar.json').read_text())['results']
    def check(index):
        worker=out/str(index);(worker/'control').mkdir(parents=True)
        (worker/'control/runtime.json').write_bytes((root/'control/runtime.json').read_bytes())
        with (worker/'output.log').open('w') as log:
            done=subprocess.run([sys.executable,str(Path(__file__).with_name('qualify.py')),
                '--method','nektar','--root',str(worker)],stdout=log,stderr=subprocess.STDOUT)
        if done.returncode:raise RuntimeError('worker '+str(index)+' failed; inspect '+str(worker/'output.log'))
        report=json.loads((worker/'qualification/nektar.json').read_text())
        for old,new in zip(expected,report['results'],strict=True):
            if not math.isclose(old['relative_l2'],new['relative_l2'],rel_tol=1e-8,abs_tol=1e-12):
                raise AssertionError('changed known-solution error')
        return {'worker':index,'results':report['results']}
    with ThreadPoolExecutor(max_workers=4) as pool:reports=list(pool.map(check,range(4)))
    data={'status':'passed','job_id':os.environ['SLURM_JOB_ID'],
        'independent_single_rank_workers':4,'known_solution_runs':8,'reports':reports}
    (out/'RESULT.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(data),flush=True)

if __name__=='__main__':main()
