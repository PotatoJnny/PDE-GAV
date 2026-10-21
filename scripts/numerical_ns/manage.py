"""Finite calibration -> validation -> terminal chain; no test-driven tuning."""
import argparse,csv,fcntl,json,math,os,subprocess
from collections import defaultdict
from pathlib import Path
from evaluate import ROOT,atomic
PARTITIONS=','.join(f'gpubase_{g}_b{n}' for g in ('l40s','h100') for n in range(1,6))
ACCOUNT=[f"--account={os.environ['PDE_GAV_SLURM_ACCOUNT']}"] if os.environ.get('PDE_GAV_SLURM_ACCOUNT') else []
BASE=['sbatch','--parsable',*ACCOUNT,'--qos=normal','--partition='+PARTITIONS,'--gres=gpu:0','--ntasks=1']

def submit(command):
    result=subprocess.check_output(command,text=True).strip().split(';')[0]
    if not result.isdigit():raise ValueError('invalid scheduler response')
    return result

def schedule(family,stage,jobs,attempt=1,indices=None):
    path=ROOT/'control'/f'{family}_{stage}_jobs.json'
    if path.exists():
        if json.loads(path.read_text())!=jobs:raise ValueError('stage manifest changed')
    else:atomic(path,jobs)
    receipt=ROOT/'control'/f'{family}_{stage}_launch{attempt}.json'
    if receipt.exists():return json.loads(receipt.read_text())
    ix=list(range(len(jobs))) if indices is None else indices
    # Each family runs one 4-worker shard at a time: at most four families / 16
    # candidate workers in this entire campaign, and no GPUs.
    array=','.join(map(str,ix))+'%1'
    command=BASE+['--cpus-per-task=8','--mem=32G','--time=03:00:00','--signal=B:USR1@900',
        '--array='+array,'--job-name=num-'+family+'-'+stage,
        '--output='+str(ROOT/'logs'/f'{family}-{stage}-%A_%a.log'),
        str(ROOT/'evaluate.sh'),str(path),'0',str(attempt)]
    # Reserve the receipt before submission, so an uncertain response cannot
    # accidentally create duplicate writers. An interrupted submission is manual.
    atomic(receipt,{'status':'submitting','command':command})
    job=submit(command)
    follow=BASE+['--cpus-per-task=1','--mem=2G','--time=00:10:00','--dependency=afterany:'+job,
        '--job-name=num-advance-'+family,'--output='+str(ROOT/'logs'/f'{family}-{stage}-advance-%j.log'),
        str(ROOT/'manage.sh'),'advance',family,stage,str(attempt)]
    data={'status':'submitted','job_id':job,'command':command}
    atomic(receipt,data)
    continuation=submit(follow);data['coordinator_job_id']=continuation;atomic(receipt,data)
    print(json.dumps(data),flush=True);return data

def rows(jobs):
    records=[]
    for job in jobs:
        p=Path(job['output'])/'cases.csv'
        records.extend(r for r in csv.DictReader(p.open()) if r['metric']=='velocity_relative_l2')
    return records

def statistics(records):
    groups=defaultdict(list)
    for r in records:groups[r['method']].append(r)
    result=[]
    for name,rs in groups.items():
        if len({r['case_id'] for r in rs})!=len(rs):raise ValueError('duplicate case results')
        good=[r for r in rs if r['status']=='ok' and math.isfinite(float(r['error']))]
        errors=sorted(float(r['error']) for r in good);k=max(1,math.ceil(len(errors)*.2))
        result.append({'profile':name,'n':len(rs),'success':len(good),'failed':len(rs)-len(good),
            'mean':sum(errors)/len(errors) if errors else None,
            'tail20':sum(errors[-k:])/len(errors[-k:]) if errors else None,
            'mean_solve_seconds':sum(float(r['solve_wall_seconds']) for r in good)/len(good) if good else None})
    return result

def advance(family,stage,attempt):
    path=ROOT/'control'/f'{family}_{stage}_jobs.json';jobs=json.loads(path.read_text())
    incomplete=[]
    for i,j in enumerate(jobs):
        checkpoint=Path(j['output'])/'checkpoint.json'
        status=json.loads(checkpoint.read_text()).get('status') if checkpoint.exists() else None
        if status=='completed':continue
        if status=='paused' and attempt<2:incomplete.append(i)
        else:raise RuntimeError(f"{j['name']} stopped with checkpoint status {status}; inspect log, no further jobs submitted")
    if incomplete:return schedule(family,stage,jobs,attempt+1,incomplete)
    rs=rows(jobs);stats=statistics(rs);atomic(ROOT/'control'/f'{family}_{stage}_summary.json',stats)
    method,task=jobs[0]['method'],jobs[0]['task']
    if stage=='terminal':
        summary={'status':'completed','family':family,'results':stats,'settings':jobs[0]}
        atomic(ROOT/'results'/family/'RESULT.json',summary)
        print(json.dumps(summary),flush=True);return
    size=4 if stage=='calibration' else 24
    eligible=[r for r in stats if r['n']==size and r['failed']==0 and r['mean'] is not None]
    if not eligible:raise RuntimeError('No profile completed all '+str(size)+' cases; no final evaluation submitted')
    if stage=='calibration':
        profiles=[r['profile'] for r in eligible];next_stage='validation'
    else:
        best=min(eligible,key=lambda x:(x['mean'],x['mean_solve_seconds']))
        profiles=[best['profile']];next_stage='terminal'
        atomic(ROOT/'control'/f'{family}_selected.json',{'status':'frozen_before_test','selection':'minimum validation mean among 24/24 successful profiles','selected':best,'all_profiles':stats})
    cohort=json.loads((ROOT/'control/cohorts.json').read_text())[task+'_'+next_stage]
    new=[]
    for profile in profiles:
        for i in range(0,len(cohort['case_ids']),12):
            name=f'{family}-{next_stage}-{profile}-{i//12}'
            new.append({'name':name,'method':method,'task':task,'stage':next_stage,'profiles':[profile],
                'case_ids':cohort['case_ids'][i:i+12],'output':str(ROOT/'results'/family/next_stage/profile/f'shard{i//12}')})
    schedule(family,next_stage,new)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['launch','advance']);p.add_argument('family');p.add_argument('stage',nargs='?',default='calibration');p.add_argument('attempt',nargs='?',type=int,default=1);a=p.parse_args()
    with (ROOT/'control'/f'{a.family}_manager.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if a.action=='launch':schedule(a.family,'calibration',json.loads((ROOT/'control'/f'{a.family}_calibration_jobs.json').read_text()))
            else:advance(a.family,a.stage,a.attempt)
        except BaseException as e:
            atomic(ROOT/'control'/f'{a.family}_{a.stage}_attention.json',{'status':'needs_attention','error':str(e),'job_id':os.environ.get('SLURM_JOB_ID')});raise
if __name__=='__main__':main()
