#!/usr/bin/env python3
"""Run/resume one frozen terminal-evaluation lane inside a single-GPU allocation.

Reuses the normal NS shard runner and its per-lane writer lock. No Slurm
submission state is created; the operator may allocate the next segment on the
same account and invoke this same command. There are no Agent calls.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'legacy/ns/src'),str(ROOT/'scripts')]


def run_terminal(plan, lane, *, segment_seconds=10800, flush_seconds=600, runner=None):
    from pde_agent_verifier_solver.engine.checkpoint_store import RunSegmentBudget
    if runner is None:
        import torch
        if torch.cuda.device_count()!=1:
            raise ValueError('terminal lane requires exactly one allocated visible CUDA device')
        torch.set_num_threads(1)
        from ns_evaluation_shards import run_lane
        runner=run_lane
    budget=RunSegmentBudget(segment_seconds,flush_seconds=flush_seconds)
    try:
        with budget.signals():
            result=runner(plan,lane,budget=budget,evaluation_devices=('cuda:0',))
        print(json.dumps(result))
        if result['status'] not in ('completed','paused'):
            raise RuntimeError('terminal lane did not finish or safely pause')
        return 75 if result['status']=='paused' else 0
    finally:
        budget.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--lane',type=int,required=True)
    p.add_argument('--segment-seconds',type=int,default=10800)
    p.add_argument('--flush-seconds',type=int,default=600)
    a=p.parse_args()
    raise SystemExit(run_terminal(a.plan,a.lane,segment_seconds=a.segment_seconds,flush_seconds=a.flush_seconds))
