#!/usr/bin/env python3
"""Summarize resource use already recorded by a run; never contact a model."""
import argparse
import json
from pathlib import Path
from pde_agent_verifier_solver.reporting.costs import collect_run_costs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pricing', type=Path)
    parser.add_argument('--prior-costs', type=Path)
    args = parser.parse_args(argv)
    load = lambda path: json.loads(path.read_text()) if path else None
    report = collect_run_costs(args.run_root, pricing=load(args.pricing),
                               prior_costs=load(args.prior_costs))
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
