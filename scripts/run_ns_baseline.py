#!/usr/bin/env python3
"""Run NS fixed-data refinement, independent BoN + repair, or adapted ERA FUTS."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pde_agent_verifier_solver.plugins.ns.baselines.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
