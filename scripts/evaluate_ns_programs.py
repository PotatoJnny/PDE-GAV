#!/usr/bin/env python3
"""Evaluate frozen S0/final/round NS bundles on a shared original numerical split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pde_agent_verifier_solver.plugins.ns.common_evaluation import run_ns_common_evaluation


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path,
                        help="ns_common_evaluation_v1 JSON manifest")
    parser.add_argument("--output", required=True, type=Path,
                        help="New output directory; existing directories are rejected")
    args = parser.parse_args(argv)
    try:
        summary = run_ns_common_evaluation(args.manifest, args.output)
    except (ValueError, TypeError, OSError, KeyError) as exc:
        parser.exit(2, f"common evaluation failed: {type(exc).__name__}: {exc}\n")
    print(json.dumps({"output": str(args.output.resolve()), "n_rows": summary["n_rows"],
                      "n_failed_program_cases": summary["n_failed_program_cases"]}))
    return 0 if summary["n_failed_program_cases"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
