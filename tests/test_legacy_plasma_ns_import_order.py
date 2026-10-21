"""Real vendored imports remain compatible in a mixed-Backend process."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("order", ["vpfp_first", "ns_first", "namespace_placeholder_first"])
def test_vendored_ns_and_vpfp_import_orders(order):
    source = """
import importlib
from pathlib import Path
import sys
root = Path.cwd()
order = sys.argv[1]
if order == "namespace_placeholder_first":
    # A bare namespace portion named like the legacy package, found before it on sys.path.
    import tempfile
    placeholder = Path(tempfile.mkdtemp()) / "agent_verifier" / "neural_generator"
    placeholder.mkdir(parents=True)
    sys.path.insert(0, str(placeholder.parent.parent))
    import agent_verifier.neural_generator
if order == "ns_first":
    sys.path.insert(0, str(root / "legacy/ns/src"))
    from agent_verifier.neural_generator import continuous_geometry
from pde_agent_verifier_solver.plugins.vpfp.perturbation_backend import _legacy_api
api = _legacy_api()
sys.path.insert(0, str(root / "legacy/ns/src"))
from agent_verifier.neural_generator import continuous_geometry
from agent_verifier.neural_generator import structured_ns_analytic
from agent_verifier.vpfp import analytic_source
assert Path(continuous_geometry.__file__).resolve().is_relative_to(root / "legacy/ns/src")
assert Path(structured_ns_analytic.__file__).resolve().is_relative_to(root / "legacy/ns/src")
assert Path(analytic_source.__file__).resolve().is_relative_to(root / "legacy/vpfp/src")
assert api.analytic_source is analytic_source
"""
    result = subprocess.run([sys.executable, "-c", source, order], cwd=ROOT,
        env={**os.environ, "PYTHONPATH": os.pathsep.join((str(ROOT), str(ROOT / "src"))),
             "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
