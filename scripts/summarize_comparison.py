#!/usr/bin/env python3
"""Create paired statistics and manuscript tables from common evaluation output."""
import argparse
from pathlib import Path
from pde_agent_verifier_solver.reporting.comparisons import (
    attach_cost_reports, read_case_rows, summarize_comparison, write_comparison_report,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline')
    parser.add_argument('--methods', nargs='+', help='Explicit condition subset, e.g. final methods without S0/round diagnostics.')
    parser.add_argument('--cost-manifest', type=Path)
    parser.add_argument('--threshold', action='append', default=[], metavar='METRIC=VALUE')
    parser.add_argument('--bootstrap-replicates', type=int, default=2000)
    parser.add_argument('--bootstrap-seed', type=int, default=0)
    parser.add_argument('--plots', action='store_true', help='Write scientific PDF using matplotlib.')
    args = parser.parse_args(argv)
    thresholds = {}
    for item in args.threshold:
        name, value = item.rsplit('=', 1)
        if name in thresholds:
            parser.error(f'duplicate threshold {name}')
        thresholds[name] = float(value)
    rows = read_case_rows(args.cases)
    if args.methods:
        available = {row['method'] for row in rows}
        if not set(args.methods) <= available:
            parser.error('requested method absent from case table')
        rows = [row for row in rows if row['method'] in args.methods]
    report = summarize_comparison(rows, baseline=args.baseline,
                                  thresholds=thresholds, bootstrap_replicates=args.bootstrap_replicates,
                                  bootstrap_seed=args.bootstrap_seed)
    if args.cost_manifest:
        attach_cost_reports(report, rows, args.cost_manifest)
    write_comparison_report(report, args.output, plots=args.plots)
    return 1 if any(g.get('status') == 'invalid_evaluation' for g in report['groups']) else 0


if __name__ == '__main__':
    raise SystemExit(main())
