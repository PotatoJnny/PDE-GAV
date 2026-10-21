"""Paper-based AutoPDE and PDE-SHARP NS adaptations; no author code is vendored.

Source scope and budget reductions are recorded in the provenance JSON files.
Execution, case permissions, timing and checkpoints remain in LiteratureHost.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math


NS_METHOD_GUIDE = """NS numerical-method guide (independent NS adaptation).
Choose a formulation consistent with incompressible velocity-pressure coupling,
the declared forcing and initial data, obstacle geometry and boundary traces.
For mixed FEM, consider velocity/pressure compatibility (for example P2/P1),
pressure gauge and boundary enforcement. For projection methods, explain the
pressure solve and boundary treatment. Check convection stability and viscous
restrictions against the actual mesh and time step. Select an algebraic solver
appropriate to the matrix symmetry and definiteness; justify stabilization.
These are considerations, not prescribed algorithms. Public input queries obey
the supplied environment; never infer access to hidden reference values.
"""
PROFILING_GUIDE = """Pilot-test the solver on available development Cases before
submission. Record actual error and elapsed/charged time when provided. Adjust
mesh resolution, polynomial degree or time step when error is limiting; simplify
the solve or adjust tolerances when execution time is limiting; diagnose traces
when execution fails. Revisit the method card if numerical evidence contradicts
its assumptions. Preserve the full physical task and time interval. The objective
is lower validation error within the declared per-Case time limit; no invented
accuracy threshold is required. Missing measurements must be marked missing.
"""


def _text_fields(payload, required):
    if not isinstance(payload, dict) or any(
        not isinstance(payload.get(k), str) or not payload[k].strip() for k in required
    ):
        raise ValueError("method record requires nonempty text fields: " + ", ".join(required))
    return payload


def run_autopde(host, options):
    """Explicit strategy cards, autonomous pilot tuning, then local-only review.

    Six independent two-pass attempts are an explicit budget-matching NS
    extension. Cards, tuning and review share each attempt's Case/time budget.
    """
    if options.protocol == "completion_v2":
        return _run_autopde_completion(host)
    if host.initial_key is not None:
        host.evaluate(host.initial_key, "validation", "initial")
    for index in range(options.autopde_attempts):
        branch = f"autopde-attempt-{index + 1}"
        diagnosis_fields = ("equation_type", "coupling", "time_dependence", "boundary_conditions", "dominant_physics")
        diagnosis = host.json_stage(
            branch + "/diagnosis", "autopde_diagnosis", branch,
            "Before writing code, diagnose the supplied NS task. Return one JSON object with "
            + ", ".join(diagnosis_fields) + ". Each value must be explanatory text. "
            "Do not replace the physical task. Put JSON in the round log; nominate no solver.",
            validator=lambda value: _text_fields(value, diagnosis_fields),
        )
        method_fields = ("formulation", "spatial_method", "time_method", "algebraic_solver", "stabilization", "boundary_treatment")
        method = host.json_stage(
            branch + "/method", "autopde_method", branch,
            "Write a METHOD card justified by DIAGNOSIS. Return a JSON object with "
            + ", ".join(method_fields) + ". Each value must be explanatory text. "
            "Do not write solver code yet.\n" + NS_METHOD_GUIDE + "\nDIAGNOSIS:\n"
            + json.dumps(diagnosis, sort_keys=True),
            validator=lambda value: _text_fields(value, method_fields),
        )
        cards = {"DIAGNOSIS": diagnosis, "METHOD": method}
        host.record(branch + "/strategy", cards)
        key = host.program(
            branch + "/develop", "autopde_develop", branch,
            instructions="Implement this strategy as one reusable NS solver. Carry out autonomous "
            "development tests and numerical tuning. Keep DIAGNOSIS and METHOD alongside the "
            "program and update them if experiments change your numerical choices. Record the "
            "final cards and measured pilot results in the round log. Nominate one complete bundle.\n"
            + NS_METHOD_GUIDE + PROFILING_GUIDE,
            plan=cards, reserve_seconds=options.autopde_review_seconds,
        )
        if key is None:
            continue
        report = host.evaluate(key, "validation", branch)
        # The first-pass log carries any strategy revisions made while tuning.
        first_pass = host.state["actions"][branch + "/develop"]["round_log"]
        reviewed = host.program(
            branch + "/review", "autopde_review", branch, parent=key,
            instructions="Review the submitted program in a fresh context. Only make local "
            "repairs to imports, shapes, interface keys or small API mistakes. Preserve the "
            "numerical method, discretization and strategy chosen in the first pass; do not "
            "redesign it. Nominate the unchanged bundle when no repair is needed.\n"
            "First-pass strategy/tuning record:\n" + first_pass,
            plan=cards, feedback=report, call_limit_seconds=options.autopde_review_seconds,
        )
        if reviewed is not None:
            host.evaluate(reviewed, "validation", branch)
    return host.best_validated()


def validate_sharp_selection(payload, pool):
    if not isinstance(payload, dict):
        raise ValueError("judge selection must be a JSON object")
    selected = payload.get("selected", ())
    count = max(1, math.ceil(len(pool) / 2))
    if (not isinstance(selected, list) or len(selected) != count
            or any(not isinstance(k, str) for k in selected)
            or len(set(selected)) != len(selected) or not set(selected).issubset(pool)
            or payload.get("nominee") not in selected
            or not isinstance(payload.get("rationale"), str) or not payload["rationale"].strip()):
        raise ValueError(f"judge must choose {count} distinct available programs and nominate one of them")
    return payload


def run_pde_sharp(host, options):
    """Five analysis steps, unranked Genesis pool, judge nomination and hybrids.

    Only judge nominees and hybrids receive outer validation, preserving the
    distinction from evaluating and ranking every Genesis candidate beforehand.
    Judges see shared completed-round feedback, never another judge's new result
    from the same round. A later cycle resets judge history and expands the pool.
    """
    if host.initial_key is not None:
        host.evaluate(host.initial_key, "validation", "initial")
    if options.protocol == "completion_v2":
        checks = ("classification", "analytic", "transformation", "decomposition", "stability")
        analysis = host.json_stage(
            "sharp-analysis", "sharp_analysis", "sharp-analysis",
            "Analyze the given PDE in order: classification, analytical treatment, useful "
            "transformations, operator decomposition, then numerical stability. Choose what "
            "is useful for this task without changing it or assuming hidden truth access. "
            "Return one JSON object with explanatory text in these five fields: "
            + ", ".join(checks) + ". Put it in the round log; nominate no solver.",
            validator=lambda value: _text_fields(value, checks),
        )
    else:
        analysis = {}
        for stage, question in (
            ("classification", "Classify the actual PDE, coupling, domain and boundary conditions."),
            ("analytic", "Check whether an analytical treatment is justified for any part; do not assume hidden truth access."),
            ("transformation", "Assess useful mathematical transformations that preserve this problem."),
            ("decomposition", "Assess operator splitting or analytical-numerical decomposition."),
            ("stability", "State applicable stability constraints and suitable numerical schemes."),
        ):
            analysis[stage] = host.json_stage(
                "sharp-analysis/" + stage, "sharp_analysis_" + stage, "sharp-analysis",
                question + ' Return {"finding":"...","consequences":"..."} in the round log; '
                "nominate no solver. Previous analysis:\n" + json.dumps(analysis, sort_keys=True),
                validator=lambda value: _text_fields(value, ("finding", "consequences")),
            )
    host.record("sharp_analysis", analysis)
    pool = []
    for index in range(options.sharp_samples):
        branch = f"sharp-genesis-{index + 1}"
        key = host.program(branch, "sharp_genesis", branch,
            instructions="Generate a reusable NS solver guided by the mathematical analysis. "
            "Develop an independent implementation, test as useful, and nominate one complete bundle.",
            plan=analysis)
        if key is not None and key not in pool:
            pool.append(key)
    if options.protocol == "completion_v2":
        host.record("sharp_initial_pool", {"keys": list(pool), "unranked": True})
    feedback = []
    validated = []
    for cycle in range(options.sharp_cycles):
        frozen_pool = tuple(pool)
        if not frozen_pool:
            break
        nominees, histories = [], []
        for judge in range(options.sharp_judges):
            branch = f"sharp-cycle-{cycle + 1}-judge-{judge + 1}"
            choice = host.json_stage(
                branch + "/select", "sharp_judge", branch,
                "Inspect the complete candidate implementations and analysis. Select your top "
                f"{max(1, math.ceil(len(frozen_pool) / 2))} programs, and nominate one for execution. "
                'Return {"selected":["tree_sha256",...],"nominee":"tree_sha256","rationale":"..."}. '
                "Nominate no solver in this selection stage. Available hashes:\n"
                + json.dumps(frozen_pool), references=frozen_pool, feedback=feedback, plan=analysis,
                validator=lambda value: validate_sharp_selection(value, frozen_pool),
            )
            nominees.append(choice["nominee"])
            histories.append([choice])
            host.record(branch + "/selection", choice)
        # All judges choose before any new nominee evaluation is exposed.
        feedback = [host.evaluate(key, "validation", "sharp-nominees") for key in dict.fromkeys(nominees)]
        validated.extend(feedback)
        for round_index in range(options.sharp_hybrid_rounds):
            proposals = []
            for judge, parent in enumerate(nominees):
                branch = f"sharp-cycle-{cycle + 1}-judge-{judge + 1}"
                key = host.program(
                    branch + f"/hybrid-{round_index + 1}", "sharp_hybrid", branch, parent=parent,
                    references=tuple(histories[judge][0]["selected"]),
                    instructions="Use the shared performance feedback and your judgment history to "
                    "make an incremental improvement or combine useful ideas in the nominated solver. "
                    "Preserve working structure; describe the intended diff and numerical rationale "
                    "in the round log. Return a complete bundle containing those edits. "
                    "Keeping a working solver is allowed. Judge history:\n"
                    + json.dumps(histories[judge], sort_keys=True), feedback=feedback, plan=analysis,
                )
                proposals.append(key)
            new_feedback = []
            for judge, key in enumerate(proposals):
                if key is None:
                    histories[judge].append({"failure": "missing_candidate"})
                    continue
                report = host.evaluate(key, "validation", f"sharp-cycle-{cycle + 1}-judge-{judge + 1}")
                new_feedback.append(report)
                validated.append(report)
                histories[judge].append(report)
                nominees[judge] = key
                if key not in pool:
                    pool.append(key)
            feedback = feedback + new_feedback
            host.record(f"sharp-cycle-{cycle + 1}/round-{round_index + 1}",
                        deepcopy({"pool": pool, "feedback": feedback, "judge_histories": histories}))
    if options.protocol == "completion_v2":
        from .workflows import _valid
        valid = [report for report in validated if _valid(report)]
        selected = min(valid, key=lambda report: report["error"])["tree_sha256"] if valid else host.initial_key
        host.record("sharp_delivery", {"selected": selected, "fallback_to_s0": not valid,
                                      "valid_evaluated_versions": len({r["tree_sha256"] for r in valid})})
        return selected
    return host.best_validated()


def _run_autopde_completion(host):
    """One complete development session followed by one fresh local review.

    The reviewed bundle is the delivery, even if an earlier version scored better.
    Host finalization measures this exact version without feeding results back.
    """
    branch = "autopde"
    key = host.program(
        "autopde/develop", "autopde_develop", branch,
        instructions=(
            "Develop one reusable NS solver in this session. Before coding, write DIAGNOSIS "
            "and METHOD records covering the PDE, boundaries, numerical difficulties, "
            "spatial and temporal discretization, and solver choices. Then implement, "
            "run numerical experiments, and improve the method as needed. Choose your own "
            "Cases, grids and time steps; update the records when experiments change your "
            "choices. Use your available environment throughout. Submit one complete solver "
            "bundle with the final DIAGNOSIS and METHOD files, and record actual trial "
            "results and the final strategy in the round log. Finish when ready.\n"
            + NS_METHOD_GUIDE + PROFILING_GUIDE
        ),
    )
    if key is None:
        host.record("autopde_delivery", {"selected": None, "failure": "missing_development_submission"})
        return None
    first_pass = host.state["actions"]["autopde/develop"]["round_log"]
    host.record("autopde_strategy", {"tree_sha256": key, "round_log": first_pass})
    report = host.evaluate(key, "validation", branch)
    reviewed = host.program(
        "autopde/review", "autopde_review", branch, parent=key, references=(key,),
        instructions=(
            "Review the submitted solver and its DIAGNOSIS, METHOD and real validation "
            "report in this fresh context. Only repair imports, shapes, interface keys "
            "or small API mistakes; do not redesign the numerical method, discretization "
            "or strategy. Use your development environment to check local repairs. "
            "Submit the unchanged complete bundle if no repair is needed. Submit exactly "
            "one final solver and finish; no further independent searches follow. "
            "Preserve the method records in the bundle. First-pass trial/strategy log:\n"
            + first_pass
        ), feedback=report,
    )
    host.record("autopde_delivery", {"selected": reviewed, "first_pass": key,
        "selection": "single_review_output_not_validation_argmin",
        "failure": None if reviewed is not None else "missing_review_submission"})
    return reviewed
