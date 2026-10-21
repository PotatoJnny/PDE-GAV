#!/usr/bin/env python3
"""Check the corrected finite input artifacts, then use the numerical Fixed runner."""
from hashlib import sha256
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def verify_fixed_inputs(batch):
    check=json.loads((batch/'control/fixed_discrete_input_check.json').read_text())
    if check.get('status')!='passed' or check.get('development_cases')!=512:
        raise RuntimeError('corrected Fixed input verification has not passed')
    config=json.loads((ROOT/'configs/paper/ns_bgp_v3/fixed_s101.json').read_text())
    if (config['dataset'].get('kind') or config['agent']['worker_settings']['profile']!='structured_numerical'):
        raise RuntimeError('Fixed must use the original numerical finite-array loader')
    manifest=Path(config['dataset']['manifest']).resolve()
    if str(manifest) != check['manifest']:
        raise RuntimeError('corrected Fixed manifest changed after verification')
    calibration=json.loads(Path(check['s0_calibration']).read_text())
    if (calibration.get('status')!='completed' or calibration.get('n_cases')!=50
            or calibration.get('n_failed_program_cases')!=0):
        raise RuntimeError('corrected Fixed S0 calibration is incomplete')


if __name__=='__main__':
    import sys
    args=sys.argv[1:]
    if '--mode' not in args or args[args.index('--mode')+1]!='fixed':
        raise SystemExit('this correction entrypoint only starts Fixed')
    verify_fixed_inputs(ROOT.parent)
    from run_ns_draft_job import main
    raise SystemExit(main(methods=('fixed',)))
