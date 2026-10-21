"""Single Slurm writer; pause/retry decisions stay with the bounded stage manager."""
import argparse,fcntl,json,os,signal,subprocess,sys
from pathlib import Path
from evaluate import ROOT

def main():
    p=argparse.ArgumentParser();p.add_argument('--jobs',type=Path,required=True);p.add_argument('--index',type=int,required=True);p.add_argument('--segment',type=int,default=1);a=p.parse_args()
    job=json.loads(a.jobs.read_text())[a.index];out=Path(job['output']);out.mkdir(parents=True,exist_ok=True)
    with (out/'dispatch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=out/'checkpoint.json'
        if path.exists() and json.loads(path.read_text()).get('status')=='completed':return 0
        if not 1<=a.segment<=2:raise ValueError('at most two segments per shard')
        command=[sys.executable,str(Path(__file__).with_name('evaluate.py')),'--jobs',str(a.jobs),'--index',str(a.index)]
        child=subprocess.Popen(command)
        previous=signal.signal(signal.SIGUSR1,lambda *_:child.send_signal(signal.SIGUSR1) if child.poll() is None else None)
        try:rc=child.wait()
        finally:signal.signal(signal.SIGUSR1,previous)
        (out/f'segment{a.segment}.json').write_text(json.dumps({'slurm_job':os.environ['SLURM_JOB_ID'],'returncode':rc})+'\n')
        return rc
if __name__=='__main__':raise SystemExit(main())
