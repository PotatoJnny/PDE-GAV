"""Portable entrypoint for a solver plus prepared macro compatibility projection."""
from pathlib import Path
import importlib.util

from pde_agent_verifier_solver.sdk.solve_budget import candidate_solve_clock


def _load_local(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_solver = _load_local('_transport_base', 'base_solver.py')
_projection = _load_local('_transport_projection', 'charge_projection.py')


def prepare_solution(case, solve_budget=None):
    started = candidate_solve_clock()
    background = case.public_data['equation'].get('poisson', {}).get('neutralizing_background')
    remaining = (None if solve_budget is None else
                 max(0., float(solve_budget)-(candidate_solve_clock()-started)))
    prepared = _solver.prepare_solution(case, solve_budget=remaining)
    if background is None:
        return prepared
    return _projection.project_prepared(prepared, background)
