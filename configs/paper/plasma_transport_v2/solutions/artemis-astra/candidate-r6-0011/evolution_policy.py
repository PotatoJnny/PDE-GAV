"""Bound an optionally replaceable coarse solve and preserve completed output."""
from __future__ import annotations
import numpy as np

BASELINE_EVOLUTION_LIMIT = 45.
FAILED_BASELINE_RESERVE = 100.


def complete_solution(panel, times, backup, evolve, refine, began, allowance,
                      slice_costs, clock, synchronize):
    """Evolve a baseline, then optionally replace it with a complete refinement.

    A local baseline deadline is used only when a 65-node refinement is
    forecast to remain affordable after the whole deadline and rescue reserve.
    Failed attempts never replace the last completed trajectory.
    """
    synchronize()
    started = clock()
    costs = np.asarray(slice_costs, dtype=float)
    valid_costs = costs.size and np.isfinite(costs).all() and np.all(costs > 0)
    estimate = (max(float(np.mean(costs)), float(np.mean(costs[-4:])))
                if valid_costs else float('inf'))
    remaining = float(allowance)-(started-float(began))
    rescue_affordable = bool(len(times) == 33 and
        32.*estimate < remaining-BASELINE_EVOLUTION_LIMIT-FAILED_BASELINE_RESERVE)
    deadline = started+BASELINE_EVOLUTION_LIMIT if rescue_affordable else None
    baseline_diagnostic = {}
    baseline = evolve(panel, times, backup, deadline=deadline,
                      diagnostic=baseline_diagnostic)
    synchronize()
    baseline_cost = max(1.e-6, clock()-started)
    baseline_ready = baseline is not backup
    reserve = (max(30., 2.*baseline_cost+20.) if baseline_ready
               else FAILED_BASELINE_RESERVE)
    refined_panel, refined_times, info = refine(panel, times, reserve)
    info.update(baseline_ready=baseline_ready, baseline_source_nodes=len(times),
                baseline_evolution_seconds=baseline_cost,
                baseline_evolution_limit=(BASELINE_EVOLUTION_LIMIT if rescue_affordable else None),
                baseline_rescue_forecast=rescue_affordable,
                baseline_diagnostic=baseline_diagnostic,
                returned_source_nodes=int(baseline.effective_source_nodes),
                refined_evolution_succeeded=False,
                rescue_after_failed_baseline=not baseline_ready)
    baseline.source_refinement_info = info
    if not info.get('completed', False):
        return baseline
    refined_diagnostic = {}
    refined = evolve(refined_panel, refined_times, baseline,
                     diagnostic=refined_diagnostic)
    synchronize()
    info['refined_evolution_diagnostic'] = refined_diagnostic
    if refined is baseline:
        info['reason'] = 'refined_evolution_rollback'
        return baseline
    info['refined_evolution_succeeded'] = True
    info['returned_source_nodes'] = len(refined_times)
    refined.source_refinement_info = info
    return refined
