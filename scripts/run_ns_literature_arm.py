#!/usr/bin/env python3
"""Run one prepared PDE-Agent/environment arm using the existing resumable CLI."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'legacy/ns/src')]


def run_arm(plan_path, label, *, gpu_token, segment_seconds=10800, flush_seconds=600,
            resume=False, baseline_main=None):
    plan = json.loads(Path(plan_path).read_text())
    if plan.get('schema') != 'ns_literature_comparisons_v1' or plan.get('protocol') != 'completion_v2':
        raise ValueError('expected a prepared completion_v2 comparisons.json')
    rows = [row for row in plan['runs'] if row['label'] == label]
    if len(rows) != 1:
        raise ValueError('arm must identify exactly one prepared run')
    row = rows[0]
    config = Path(row['config'])
    if baseline_main is None:
        from pde_agent_verifier_solver.plugins.ns.baselines.cli import main as baseline_main
    args = ['--config', str(config), '--output', row['output'], '--gpu-token', gpu_token,
            '--segment-seconds', str(segment_seconds), '--flush-seconds', str(flush_seconds)]
    if resume:
        args.append('--resume')
    return baseline_main(args)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--arm', required=True)
    p.add_argument('--gpu-token', required=True)
    p.add_argument('--segment-seconds', type=int, default=10800)
    p.add_argument('--flush-seconds', type=int, default=600)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args(argv)
    return run_arm(args.plan, args.arm, gpu_token=args.gpu_token,
                   segment_seconds=args.segment_seconds, flush_seconds=args.flush_seconds,
                   resume=args.resume)


if __name__ == '__main__':
    raise SystemExit(main())
