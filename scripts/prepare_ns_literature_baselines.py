#!/usr/bin/env python3
"""Bind declared PDE-Agent environment pairs, or freeze their terminal programs."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "legacy/ns/src")]
from pde_agent_verifier_solver.plugins.ns.baselines.preparation import prepare_comparisons, export_terminal_manifest, export_starter_references, export_checkpoint_starter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--binding", type=Path, help="ns_literature_binding_v1 JSON with receiver asset paths")
    source.add_argument("--completed-plan", type=Path, help="comparisons.json after all declared runs complete")
    source.add_argument("--starter-references", type=Path, help="re-export an existing continuous development bank with finite reference arrays")
    source.add_argument("--starter-checkpoint", type=Path, help="CPU export of the original common64 fields using receiver_restore_config.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = (export_checkpoint_starter(args.starter_checkpoint, args.output) if args.starter_checkpoint else
              export_starter_references(args.starter_references, args.output) if args.starter_references else
              prepare_comparisons(args.binding, args.output) if args.binding else
              export_terminal_manifest(args.completed_plan, args.output))
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
