"""Source-audited literature policies, using the existing NS execution services.

These are independent NS task adaptations, not vendored author implementations.
The host owns execution, private validation, budget accounting and continuation.
Nothing in this module can read a terminal/test bank.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import random


LITERATURE_METHODS = ("codepde_refinement_ns_adaptation", "autonumerics_ns_adaptation",
                      "autopde_ns_adaptation", "pde_sharp_ns_adaptation")


class DevelopmentBudgetExhausted(Exception):
    """The declared development budget has been spent; deliver a validated result."""


@dataclass(frozen=True)
class LiteratureOptions:
    protocol: str = "bounded_v1"
    total_wall_seconds: int | None = 43200
    branch_wall_seconds: int | None = 7200
    planning_call_seconds: int | None = 600
    json_attempts: int = 2
    codepde_samples: int = 3
    codepde_trials: int = 5
    codepde_seed_pool_size: int = 3
    codepde_reference_counts: tuple[int, ...] = (2, 3, 3)
    autonumerics_plans: int = 10
    autonumerics_top_k: int = 5
    autonumerics_fresh_starts: int = 2
    autonumerics_coarse_repairs: int = 4
    # The pinned public runner uses two dense repairs (the v1 paper says six).
    autonumerics_fine_repairs: int = 2
    autonumerics_coarse_factor: int = 4
    autonumerics_probe_cases: int = 1
    autopde_attempts: int = 6
    autopde_review_seconds: int | None = 300
    sharp_samples: int = 3
    sharp_judges: int = 3
    sharp_cycles: int = 1
    sharp_hybrid_rounds: int = 1

    def __post_init__(self):
        if self.protocol not in {"bounded_v1", "completion_v2"}:
            raise ValueError("unknown literature protocol")
        for name, value in vars(self).items():
            if name == "protocol":
                continue
            if name in {"total_wall_seconds", "branch_wall_seconds", "planning_call_seconds"} and self.protocol == "completion_v2":
                if value is not None:
                    raise ValueError("completion_v2 requires null development time limits")
                continue
            if name == "autopde_review_seconds" and self.protocol == "completion_v2" and value is None:
                continue
            if name == "codepde_reference_counts":
                values = tuple(value)
                if not values or any(type(v) is not int or v < 1 for v in values):
                    raise ValueError("codepde_reference_counts must contain positive integers")
                object.__setattr__(self, name, values)
            elif type(value) is not int or value < (0 if name.endswith("repairs") else 1):
                raise ValueError(f"{name} must be a valid integer budget/count")
        if self.autonumerics_top_k > self.autonumerics_plans:
            raise ValueError("AutoNumerics top_k cannot exceed its number of plans")


def _valid(report):
    value = report.get("error")
    return (report.get("status") == "ok" and isinstance(value, (int, float))
            and not isinstance(value, bool) and math.isfinite(value) and value >= 0)


def run_codepde(host, options: LiteratureOptions):
    """Independent samples followed by multi-seed refinement and failure debugging.

    As upstream, a finite score ends debugging, even when accuracy is poor.
    The refinement pool is frozen before refinement; refinements do not reenter it.
    A supplied S0 is an explicitly recorded NS seed/delivery fallback.
    """
    valid = []
    if host.initial_key is not None and options.protocol == "bounded_v1":
        report = host.evaluate(host.initial_key, "validation", "initial")
        if _valid(report):
            valid.append((report["error"], host.initial_key))

    def sample(branch, references=()):
        parent, feedback = None, None
        for trial in range(options.codepde_trials):
            mode = "debug" if trial else ("refine" if references else "generate")
            key = host.program(
                f"{branch}/trial-{trial}", f"codepde_{mode}", branch,
                parent=parent, references=references,
                instructions=(
                    "Develop one reusable NS solver. "
                    + ("Study ALL supplied seed implementations and their validation errors; "
                       "combine useful ideas or develop a better numerical method. "
                       if references else "Explore a fresh numerical implementation. ")
                    + ("Repair the reported execution/nonfinite failure. " if trial else "")
                    + "Stop this attempt by nominating exactly one complete solver bundle."
                ),
                feedback=feedback,
            )
            # A missing/invalid program is not an executable debugging parent.
            if key is None:
                return
            report = host.evaluate(key, "validation", branch)
            if _valid(report):
                valid.append((report["error"], key))
                return
            parent, feedback = key, report

    for index in range(options.codepde_samples):
        sample(f"sample-{index + 1}")
    pool = tuple(dict.fromkeys(key for _, key in sorted(valid, key=lambda row: row[0])))
    if options.protocol == "bounded_v1":
        pool = pool[:options.codepde_seed_pool_size]
    host.record("codepde_seed_pool", {"keys": pool, "frozen_before_refinement": True})
    for index, count in enumerate(options.codepde_reference_counts):
        if not pool:
            break
        seed = int.from_bytes(sha256(f"{host.seed}:refine:{index}".encode()).digest()[:8], "big")
        references = (pool if options.protocol == "completion_v2" else
                      tuple(random.Random(seed).sample(list(pool), min(count, len(pool)))))
        sample(f"refinement-{index + 1}", references)
    if options.protocol == "completion_v2":
        chosen = min(valid, key=lambda row: row[0])[1] if valid else host.initial_key
        host.record("codepde_delivery", {"selected": chosen, "fallback_to_s0": not valid,
                                       "valid_outputs": len(valid)})
        return chosen
    return host.best_validated()


def validate_plans(payload, expected_count):
    plans = payload.get("plans")
    if not isinstance(plans, list) or len(plans) != expected_count:
        raise ValueError(f"Planner must return exactly {expected_count} plans")
    ids = set()
    for plan in plans:
        if not isinstance(plan, dict):
            raise ValueError("each plan must be an object")
        ident = plan.get("plan_id")
        if not isinstance(ident, str) or not ident.strip() or ident in ids:
            raise ValueError("plan ids must be nonempty and unique")
        ids.add(ident)
        if not isinstance(plan.get("description"), str) or not plan["description"].strip():
            raise ValueError("each plan needs a description")
        spatial = plan.get("spatial_discretization")
        if not isinstance(spatial, dict) or not isinstance(spatial.get("method"), str):
            raise ValueError("each plan needs a spatial discretization method")
        for axis in ("Nx", "Ny"):
            if type(spatial.get(axis)) is not int or spatial[axis] < 16:
                raise ValueError("NS plans must declare integer Nx and Ny >= 16")
        if not isinstance(plan.get("time_stepping"), dict):
            raise ValueError("each plan needs a time_stepping description")
    return plans


def rank_plans(plans, payload, top_k):
    scores = payload.get("scores")
    if not isinstance(scores, list) or len(scores) != len(plans):
        raise ValueError("Selector must score every proposed plan exactly once")
    indexed = {}
    for row in scores:
        if not isinstance(row, dict):
            raise ValueError("plan scores must be objects")
        key, score = row.get("plan_id"), row.get("score")
        if (key in indexed or key not in {p["plan_id"] for p in plans}
                or isinstance(score, bool) or not isinstance(score, (int, float))
                or not math.isfinite(score) or not 0 <= score <= 100
                or row.get("verdict") not in {"accept", "caution", "reject"}):
            raise ValueError("invalid Selector plan id, score or verdict")
        indexed[key] = row
    ordered = sorted(plans, key=lambda p: -indexed[p["plan_id"]]["score"])
    surviving = [p for p in ordered if indexed[p["plan_id"]]["verdict"] != "reject"]
    # Match the audited upstream all-rejected fallback, and retain it in the log.
    return (surviving or ordered)[:top_k], not surviving


def coarse_plan(plan, factor):
    result = deepcopy(plan)
    spatial = result["spatial_discretization"]
    for axis in ("Nx", "Ny", "Nz"):
        if axis in spatial:
            spatial[axis] = max(16, int(spatial[axis]) // factor)
    result["development_fidelity"] = "coarse"
    return result


def screen_plans(plans):
    """Deduplicate settings and check basic NS facts; not the author's PDE catalog."""
    unique, seen, checks, duplicates = [], {}, {}, []
    for plan in plans:
        numerical = {k: v for k, v in plan.items() if k not in {"plan_id", "description", "rationale"}}
        signature = json.dumps(numerical, sort_keys=True, allow_nan=False)
        if signature in seen:
            duplicates.append({"plan_id": plan["plan_id"], "duplicate_of": seen[signature]})
            continue
        seen[signature] = plan["plan_id"]
        unique.append(plan)
        dt = plan["time_stepping"].get("dt")
        if dt is not None and (isinstance(dt, bool) or not isinstance(dt, (int, float))
                               or not math.isfinite(dt) or dt <= 0):
            raise ValueError("plan dt must be positive and finite")
        checks[plan["plan_id"]] = {
            "spatial_cells": plan["spatial_discretization"]["Nx"] * plan["spatial_discretization"]["Ny"],
            "requires_boundary_and_obstacle_treatment": True,
            "requires_incompressibility_treatment": True,
            "stability_claim": "not_certified_by_static_checks",
        }
    return unique, checks, duplicates


NUMERICAL_CONTEXT = (
    "Numerical reference anchors (not prescribed choices): projection methods separate "
    "momentum and incompressibility solves (Chorin, 1968); mixed finite elements require "
    "compatible velocity/pressure spaces (Taylor and Hood, 1973); finite-volume fluxes "
    "should preserve conservation; spectral differentiation requires appropriate "
    "periodicity or boundary handling. Check advection/diffusion stability, boundary "
    "consistency, pressure gauge, and obstacle treatment for this particular Case ABI. "
    "These are lightweight method notes, not a retrieved literature database."
)


def validate_final_choice(payload, results):
    if payload.get("plan_id") not in {r["plan_id"] for r in results}:
        raise ValueError("final Selector must choose one successfully fine-evaluated plan")
    if not isinstance(payload.get("rationale"), str) or not payload["rationale"].strip():
        raise ValueError("final Selector must explain its choice")
    return payload


def run_autonumerics(host, options: LiteratureOptions):
    """Formulate, plan, rank, implement, coarse/fine repair and fresh restart.

    NS adaptations: a reusable Case solver ABI, shared private reference scoring,
    and either bounded_v1 timing or completion_v2 stages. Plan resolution is passed
    to the actual candidate process, never approximated by fewer scoring points.
    """
    spec = host.json_stage(
        "formulate", "autonumerics_formulator", "planning",
        "Describe the supplied NS task and numerical features in a JSON object. "
        "Include equation, boundary_conditions, input_interface, and numerical_features. "
        "The supplied physical problem is immutable; do not replace its geometry, "
        "forcing, time interval or initial data with a simpler problem. "
        "Put ONLY the JSON in your round log; do not nominate a solver.",
        required=("equation", "boundary_conditions", "input_interface", "numerical_features"),
    )
    plan_payload = host.json_stage(
        "plan", "autonumerics_planner", "planning",
        f"Propose {options.autonumerics_plans} distinct numerical plans for this task. "
        'Return {"plans":[{"plan_id":"...","description":"...",'
        '"spatial_discretization":{"method":"...","Nx":128,"Ny":128},'
        '"time_stepping":{"method":"...","dt":0.001}}]}. '
        "Values shown illustrate the schema, not prescribed numerical choices. "
        "Consider stability, consistency and computational cost. Grid sizes must be "
        "integers >=16. The same program must accept every supplied Case. "
        "Put only JSON in the round log; do not nominate a solver.\n"
        + json.dumps({"problem": spec, "reference_context": NUMERICAL_CONTEXT}
                     if options.protocol == "completion_v2" else spec, sort_keys=True),
        validator=lambda value: validate_plans(value, options.autonumerics_plans),
    )
    plans = validate_plans(plan_payload, options.autonumerics_plans)
    checks, duplicates = {}, []
    if options.protocol == "completion_v2":
        plans, checks, duplicates = screen_plans(plans)
    ranking = host.json_stage(
        "rank", "autonumerics_selector", "planning",
        "Assess each proposed plan's numerical compatibility, boundary treatment, "
        "stability and cost. Return scores for EVERY plan, with plan_id, score "
        '(finite, 0..100), verdict (accept/caution/reject), and rationale. '
        'Format: {"scores":[...]}. Put only JSON in the round log.\n'
        + json.dumps({"problem": spec, "plans": plans, **({"static_checks": checks}
                      if options.protocol == "completion_v2" else {})}, sort_keys=True),
        validator=lambda value: rank_plans(plans, value, options.autonumerics_top_k),
    )
    selected, fallback = rank_plans(plans, ranking, options.autonumerics_top_k)
    host.record("autonumerics_plans", {
        "problem": spec, "plans": plans, "ranking": ranking,
        "selected_plan_ids": [p["plan_id"] for p in selected],
        "all_rejected_fallback": fallback,
        "plan_screening": "NS_checks_and_selector_not_author_catalog_heuristics",
        "duplicates_removed": duplicates, "static_checks": checks,
    })
    successful = []
    for index, plan in enumerate(selected):
        branch = f"plan-{index + 1}"
        for restart in range(options.autonumerics_fresh_starts):
            prefix = f"{branch}/fresh-{restart + 1}"
            key = host.program(
                f"{prefix}/generate", "autonumerics_coder", branch,
                instructions=(
                    "Implement the supplied numerical plan from scratch as a reusable NS solver. "
                    "Export prepare_solution_for_plan(case, plan, solve_budget=None), "
                    "returning the standard handle with evaluate(query). Use the passed "
                    "plan's Nx/Ny and time-stepping settings so exactly the same code can "
                    "run on coarse and fine grids. Also export the ordinary "
                    "prepare_solution(case, solve_budget=None) for development tests. "
                    "Preserve all physical inputs and the full Case time interval. "
                    "Nominate exactly one bundle."
                ), plan=plan,
            )
            if key is None:
                continue
            succeeded = True
            for fidelity, limit in (("coarse", options.autonumerics_coarse_repairs),
                                    ("fine", options.autonumerics_fine_repairs)):
                active_plan = (coarse_plan(plan, options.autonumerics_coarse_factor)
                               if fidelity == "coarse" else plan)
                for attempt in range(limit + 1):
                    report = host.evaluate(key, fidelity, branch, plan=active_plan)
                    if _valid(report):
                        break
                    if attempt == limit:
                        succeeded = False
                        break
                    key = host.program(
                        f"{prefix}/{fidelity}/repair-{attempt + 1}",
                        f"autonumerics_critic_{fidelity}", branch, parent=key,
                        instructions=(
                            "Repair the supplied implementation using the trusted failure "
                            "diagnostic. Preserve the plan-parameterized solver ABI. "
                            + ("Check array shapes, indexing and logic on the coarse grid. "
                               if fidelity == "coarse" else
                               "Check stability and time stepping on the fine grid. ")
                            + "Any time-step adjustment must be implemented in the code "
                            "and still solve the complete declared time interval. "
                            "Nominate exactly one repaired bundle."
                        ), plan=active_plan, feedback=report,
                    )
                    if key is None:
                        succeeded = False
                        break
                if not succeeded:
                    break
            if succeeded:
                successful.append({"plan_id": plan["plan_id"], "plan": plan,
                                   "report": report, "tree_sha256": report["tree_sha256"]})
                break
    if options.protocol == "completion_v2":
        if not successful:
            host.record("autonumerics_final", {"fallback_to_s0": True, "selected": host.initial_key,
                                             "reason": "no_successful_plan"})
            return host.initial_key
        decision = host.json_stage(
            "final_select", "autonumerics_final_selection", "final_analysis",
            "Select ONE successful numerical plan using the actual trusted fine-grid reports. "
            "Prioritize accuracy and stability; use runtime to distinguish comparable accuracy. "
            "All reports use the same validation Cases. Do not invent measurements. "
            'Return {"plan_id":"...","rationale":"..."}; no solver nomination.\n'
            + json.dumps(successful, sort_keys=True),
            validator=lambda value: validate_final_choice(value, successful),
        )
        chosen = next(r for r in successful if r["plan_id"] == decision["plan_id"])
        explanation = host.json_stage(
            "reason", "autonumerics_reasoner", "final_analysis",
            "Explain the already selected numerical method, why it fits the original NS task, "
            "and its observed limitations using the supplied plan, code and measurements. "
            "Separate observed evidence from theoretical expectations. You cannot change the "
            "chosen program or selection, and have no terminal-test access. "
            'Return {"explanation":"...","limitations":[...]}; no solver nomination.\n'
            + json.dumps({"problem": spec, "chosen": chosen, "decision": decision}, sort_keys=True),
            required=("explanation", "limitations"), references=(chosen["tree_sha256"],),
        )
        host.record("autonumerics_final", {"selected": chosen["tree_sha256"],
            "fallback_to_s0": False, "successful_plans": successful,
            "selector": decision, "reasoner": explanation})
        return chosen["tree_sha256"]
    return host.best_validated()
