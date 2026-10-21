#!/usr/bin/env python3
"""Freeze and submit the authorized, bounded CPU-only Poisson mechanism experiment."""
from __future__ import annotations
import os
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PARTITIONS = ','.join(f'gpubase_{gpu}_b{tier}' for gpu in ('l40s', 'h100') for tier in range(1, 6))
SOURCES = (
    'src/pde_agent_verifier_solver',
    'scripts/run_poisson_perturbed_mechanism_cpu.py',
    'scripts/submit_poisson_perturbed_mechanism_cpu.py',
    'scripts/run_canonical_codex_agent.py', 'scripts/run_codex_with_ephemeral_auth.py',
    'scripts/canonical_codex.config.toml', 'configs/candidates/poisson_zero.py',
)
JOBS = (
    ('qualify', '00:15:00', ()),
    ('common', '00:30:00', ('qualify',)),
    ('without_p', '00:30:00', ('common',)),
    ('with_p', '00:30:00', ('common',)),
    ('terminal', '00:15:00', ('without_p', 'with_p')),
)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def freeze(output):
    frozen = output/'source'
    for name in SOURCES:
        src = ROOT/name
        dst = frozen/name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        else:
            shutil.copy2(src, dst)
    return frozen


def wrapper(output, frozen, python):
    q = shlex.quote
    return f'''#!/usr/bin/env bash
set -euo pipefail
umask 077
if [[ ! "${{SLURM_JOB_ID:-}}" =~ ^[0-9]+$ || "$#" != 1 ]]; then
  echo "Requires a CPU Slurm allocation and exactly one stage." >&2
  exit 2
fi
if [[ -n "${{SLURM_JOB_GPUS:-}}" || -n "${{SLURM_STEP_GPUS:-}}" ]]; then
  echo "GPU allocation rejected for this CPU-only experiment." >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void
export PYTHONPATH={q(str(frozen/'src'))}
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cd {q(str(frozen))}
scontrol show job "${{SLURM_JOB_ID}}" > {q(str(output/'logs'))}/allocation-"${{SLURM_JOB_ID}}".txt
if [[ "$1" == qualify ]]; then
  {q(str(python))} scripts/run_poisson_perturbed_mechanism_cpu.py prepare --output {q(str(output))}
fi
exec {q(str(python))} scripts/run_poisson_perturbed_mechanism_cpu.py "$1" --output {q(str(output))}
'''


def submit(output, *, dry_run=False):
    output = output.resolve()
    # A new directory for one writer and one bounded dependency graph. Never
    # cancel/resubmit existing jobs or silently reuse a partially launched run.
    output.mkdir(parents=True, exist_ok=False)
    (output/'logs').mkdir()
    frozen = freeze(output)
    python = ROOT/'.venv/bin/python'
    launch = output/'run_stage.sh'
    launch.write_text(wrapper(output, frozen, python)); launch.chmod(0o700)
    subprocess.run(['bash', '-n', str(launch)], check=True)
    manifest = {'schema':'poisson_perturbed_cpu_launch_v1', 'created_utc':datetime.now(timezone.utc).isoformat(),
        'output':str(output), 'source':str(frozen), 'python':str(python),
        'resources':{'cpus':4,'memory':'8G','gpus':0,'partitions':PARTITIONS},
        'jobs':{}, 'status':'dry_run' if dry_run else 'submitting'}
    receipt = output/'LAUNCH.json'
    write_json(receipt, manifest)
    for stage, limit, prerequisites in JOBS:
        command = ['sbatch','--parsable',*([f"--account={os.environ['PDE_GAV_SLURM_ACCOUNT']}"] if os.environ.get('PDE_GAV_SLURM_ACCOUNT') else []),'--qos=normal',
            '--partition='+PARTITIONS,'--gres=gpu:0','--ntasks=1','--cpus-per-task=4','--mem=8G',
            '--time='+limit,'--job-name=poisson-p-'+stage,'--kill-on-invalid-dep=yes',
            '--chdir='+str(frozen),'--output='+str(output/'logs'/f'{stage}-%j.out'),
            '--error='+str(output/'logs'/f'{stage}-%j.err')]
        if prerequisites:
            command.append('--dependency=afterok:'+':'.join(manifest['jobs'][s]['job_id'] for s in prerequisites))
        command.extend([str(launch),stage])
        if dry_run:
            job_id = 'DRY_'+stage
        else:
            try:
                response = subprocess.run(command, check=True, text=True, capture_output=True)
                job_id = response.stdout.strip().split(';',1)[0]
                if not job_id.isdigit():
                    raise RuntimeError('Unexpected sbatch response: '+response.stdout)
            except Exception as exc:
                manifest['status']='submission_incomplete'
                manifest['error']=str(exc)
                write_json(receipt, manifest)
                raise
        manifest['jobs'][stage]={'job_id':job_id,'walltime':limit,'depends_on':prerequisites,'command':command}
        write_json(receipt, manifest)
        print(f'{stage}: {job_id}', flush=True)
    manifest['status']='dry_run' if dry_run else 'submitted'
    write_json(receipt, manifest)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dry-run',action='store_true',help='Freeze and print the graph without submitting any jobs')
    a=p.parse_args()
    print(submit(a.output,dry_run=a.dry_run))


if __name__=='__main__':
    main()
