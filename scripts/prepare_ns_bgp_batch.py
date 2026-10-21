#!/usr/bin/env python3
"""Prepare the five approved NS B/G/P evaluation arms; never submit jobs."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from pde_agent_verifier_solver.plugins.ns.structured_guidance_backend import StructuredNSGuidanceConfig

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_REVISION = "iclr_ns_bgp_192_single_final_v5"
BUDGET_REVISION = "iclr_hard_wall_4h_single_seed_v2"
DIRECTORY = "configs/paper/ns_bgp_v3"
CONTROL = "../../../../control/mixed_evaluation"
OFFICIAL_COUNTS = dict(original=0, base=64, guided=64, base_perturbed=64, guided_perturbed=0)
TERMINAL_COUNTS = dict(original=0, base=256, guided=256, base_perturbed=256, guided_perturbed=0)

NOMINATION_POLICY = "latest"
TRANSIENT_RESUME_MAX_ATTEMPTS = 3
FINAL_SUBMISSION = (
    "Develop and test as many versions as useful within your budget. Each submit "
    "replaces your own nominated representative; at the round boundary only your "
    "last committed nomination enters Promotion. Choose your strongest program "
    "using development evidence and submit it before finishing. To select an earlier "
    "version, submit it again last. Earlier versions and research logs remain archived."
)

RESEARCH_ROLES = {
    "ns-broad-explorer": (
        "Focus primarily on exploring new approaches to the task. Pursue the ideas "
        "you judge most promising for producing a stronger program. Use the current "
        "program and available evidence as context, and follow useful findings "
        "wherever they lead. You have full freedom to choose methods, representations, "
        "and experiments, including combining new ideas with improvements to existing "
        "approaches."
    ),
    "ns-critical-refiner": (
        "Focus primarily on improving the current approach. Use the current program "
        "and available evidence to identify where further work could produce a stronger "
        "program. You decide what to retain, change, or replace, and follow useful "
        "findings wherever they lead. You have full freedom to choose methods, "
        "representations, and experiments, including a different approach when it "
        "offers a better path forward."
    ),
}
SOURCE_ROLES = {
    "ns-broad-explorer": "Explore physically distinct NS numerical methods.",
    "ns-critical-refiner": "Diagnose and refine numerical stability, boundary treatment and integration.",
}


SOURCE_INPUT = """Deliver one solver supporting both public input representations. Generated
Cases expose domain and query_initial_velocity/query_forcing via the public
session. Original numerical Cases expose public_data['question'] and ['arrays']:
x[X], y[Y], t[T], initial_velocity[2,X,Y], forcing[T,2,X,Y], fluid_mask[X,Y].
Use these public inputs to solve the SAME declared NS equation; interpolate
finite arrays when necessary. The supplied S0 demonstrates both input paths.
Return a prepared handle evaluated at physical (t,x,y) points with velocity_x
and velocity_y. Preserve support for both input types when editing. No hidden
trajectory, reference files, selection data or test data may be inspected.
"""
PUBLIC_INPUTS = """NS task and public inputs
-------------------------
Build a solver for the supplied incompressible Navier--Stokes Cases, supporting
both public input representations. Generated Cases expose the domain and public
query_initial_velocity/query_forcing Oracles. Original numerical Cases expose
public_data['question'] and ['arrays']: x[X], y[Y], t[T],
initial_velocity[2,X,Y], forcing[T,2,X,Y], and fluid_mask[X,Y].
Both representations describe the declared NS equation. The supplied S0
illustrates support for both input paths.

Selection objective
-------------------
"""


OLD_SELECTION = """Selection minimizes untruncated mean u/v-component relative L2 on a complete,
finite validation cohort; execution/nonfinite failures make that program
ineligible, and exact ties retain the incumbent. Do not optimize pressure error."""
ONLINE_SELECTION = """At each round boundary, Promotion evaluates each Agent's final nominated
program and the entering incumbent on the same 192 hidden Cases within this run:
64 clean Base, 64 Guided, 64 Base-perturbed, and 0 Guided-perturbed.
Minimize overall.mean: the unweighted mean of the 192
per-Case u/v-component relative L2 errors, without truncation. Execution or
nonfinite failures make a program ineligible; exact ties retain the incumbent.
The Base and Base-perturbed Cases are frozen and shared across runs for the
corresponding round. Guided evaluation Cases use this run's entering incumbent
as their guidance reference and can differ across runs. All candidates within
the same run and round share the complete Official suite. Pressure is unscored."""
FIXED_SELECTION = """At each round boundary, each Agent's final nominated program and the
incumbent are evaluated on the same fixed 192 hidden selection Cases:
64 clean Base, 64 Guided, 64 Base-perturbed, and 0 Guided-perturbed.
Minimize the unweighted mean of all 192 per-Case u/v-component relative L2 errors, without truncation. Execution or
nonfinite failures make a program ineligible; exact ties retain the incumbent.
This selection bank was frozen using the common S0 reference. It stays fixed
throughout this baseline, independently of the Agent's current program, and is
separate from the final test bank. Pressure is unscored."""


def _read(path):
    return json.loads(path.read_text())


def _replace_prompt(prompt, old, new):
    if prompt.count(old) != 1:
        raise ValueError("source NS prompt changed; review the v3 derivation")
    return prompt.replace(old, new)


def _canonical(config, method):
    config = deepcopy(config)
    if (config["master_seed"] != 101 or config["round_count"] != 5
            or len(config["agents"]) != 2 or config["execution"]["agent_hard_wall_seconds"] != 14400
            or "agent_active_budget_seconds" in config["execution"]
            or config["development"]["max_cases_per_agent_per_round"] != 512):
        raise ValueError("source NS configuration differs from the approved time/development protocol")
    config["campaign_id"] = f"ns-bgp-v5-{method}-s101"
    config["execution"]["nomination_policy"] = NOMINATION_POLICY
    config["execution"]["transient_resume_max_attempts"] = TRANSIENT_RESUME_MAX_ATTEMPTS
    config["dataset"]["manifest"] = "../ns_draft_v1/data/development_manifest.json"
    config["candidate"]["initial_bundle"] = "../ns_draft_v1/s0_hold_initial_bundle"
    config["official"] = dict(OFFICIAL_COUNTS)
    config["promotion"] = {"preset": "minimize_metric", "settings": {
        "primary": {"distribution": "overall", "statistic": "mean"}}}
    config["terminal"] = {
        "cohorts": [
            {"cohort_id": "common_base", "guidance_source": "base", "suite": {
                "original": 0, "base": 256, "guided": 0, "base_perturbed": 256, "guided_perturbed": 0}},
            {"cohort_id": "common_guided", "guidance_source": "final_guided", "suite": {
                "original": 0, "base": 0, "guided": 256, "base_perturbed": 0, "guided_perturbed": 0}},
        ],
        "ranking": {"cohort_id": "overall", "distribution": "overall", "statistic": "mean"},
    }
    config.pop("official_audit_manifest", None)
    config["shared_evaluation_suite_manifest"] = f"{CONTROL}/comparison_suite.json"
    guidance = config["generator"]["guidance"]
    # The hash covers the scientific backend settings, not protocol case counts.
    # Call the actual runtime config implementation; do not reproduce its hashing.
    backend = StructuredNSGuidanceConfig(**guidance["backend"])
    if backend.training_steps != 5000:
        raise ValueError("v3 retains the approved 5000 Guidance training steps")
    guidance["protocol"].update(
        guidance_config_sha256=backend.sha256,
        reward_feedback_case_count=64,
        official_round_case_count=192,
        official_terminal_case_count=768,
    )
    for agent in config["agents"]:
        agent["prompt"] = _replace_prompt(agent["prompt"], SOURCE_INPUT, PUBLIC_INPUTS)
        agent["prompt"] = _replace_prompt(agent["prompt"], OLD_SELECTION, ONLINE_SELECTION)
        agent["prompt"] = _replace_prompt(
            agent["prompt"], SOURCE_ROLES[agent["agent_id"]], RESEARCH_ROLES[agent["agent_id"]] + "\n",
        )
        agent["prompt"] = _replace_prompt(
            agent["prompt"], "submit all distinct runnable candidates worth common evaluation.",
            FINAL_SUBMISSION,
        )
    return config


def _fixed(config):
    config = deepcopy(config)
    if (config["seed"] != 101 or config["search"]["competition_rounds"] != 5
            or len(config["agent"]["roles"]) != 2 or config["agent"]["hard_wall_seconds"] != 14400
            or "active_budget_seconds" in config["agent"]):
        raise ValueError("source fixed baseline differs from the approved time protocol")
    config["dataset"]["manifest"] = "../ns_draft_v1/data/development_manifest.json"
    config["dataset"].pop("selection_split", None)
    config["dataset"]["selection_manifest"] = f"{CONTROL}/fixed_selection.json"
    config["initial_candidate"] = "../ns_draft_v1/s0_hold_initial_bundle"
    config["evaluation"]["borrow_agent_gpus"] = True
    config["agent"]["nomination_policy"] = NOMINATION_POLICY
    config["agent"]["transient_resume_max_attempts"] = TRANSIENT_RESUME_MAX_ATTEMPTS
    config["prompt"] = _replace_prompt(config["prompt"], SOURCE_INPUT, PUBLIC_INPUTS)
    config["prompt"] = _replace_prompt(config["prompt"], OLD_SELECTION, FIXED_SELECTION)
    old = ("Use the fixed 512 development Cases and available evaluation tool; this run has no online "
           "case-generation or public oracle sessions. Both Agents compete on the same fixed 512 "
           "validation Cases at each round boundary. Retain all useful runnable nominations and cumulative logs.")
    new = ("Use the same fixed 512 original finite-array development Cases and the available evaluation "
           "tool throughout all five rounds. The Agents cannot generate new development Cases online. "
           "Submitted solvers must also support continuous generated Cases through their public initial-velocity "
           "and forcing oracles when evaluated on hidden selection/test Cases. " + FINAL_SUBMISSION)
    config["prompt"] = _replace_prompt(config["prompt"], old, new)
    for agent in config["agent"]["roles"]:
        agent["prompt"] = RESEARCH_ROLES[agent["agent_id"]]
    return config


def prepare(root: Path = ROOT) -> tuple[Path, ...]:
    root = Path(root).resolve()
    first = root / "configs/paper/ns_draft_v1"
    gp = root / "configs/paper/ns_gp_v2"
    old_entries = {
        entry["method"]: entry
        for manifest in (first / "first_batch.json", gp / "gp_batch.json")
        for entry in _read(manifest)["entries"]
    }
    methods = ("fixed", "base", "full", "base_guided", "base_perturbed")
    outputs, entries = {}, []
    destination = root / DIRECTORY
    for method in methods:
        source = (gp if method.startswith("base_") else first) / f"{method}_s101.json"
        config = _fixed(_read(source)) if method == "fixed" else _canonical(_read(source), method)
        name = f"{method}_s101.json"
        outputs[destination / name] = config
        entries.append({**old_entries[method], "mode": method, "config": name,
                        "source_config": str(source.relative_to(root)),
                        "evaluation_revision": PROTOCOL_REVISION})
    outputs[destination / "bgp_batch.json"] = {
        "schema": "ns_bgp_batch_v3", "protocol_revision": PROTOCOL_REVISION,
        "budget_revision": BUDGET_REVISION, "seed": 101,
        "time_budget_mode": "hard_wall", "agent_hard_wall_seconds": 14400,
        "agents": 2, "rounds": 5, "execution_status": "prepared_not_submitted",
        "submission_scope": "approved_first_and_second_batches_only", "entries": entries,
        "data_index": "../ns_draft_v1/data/bank_index.json",
        "reference_bundle": "../ns_draft_v1/s0_hold_initial_bundle",
        "agent_resume": {"maximum_attempts_including_initial": TRANSIENT_RESUME_MAX_ATTEMPTS,
                         "scope": "same_native_session_within_live_agent_turn",
                         "budget": "original_hard_deadline_including_reconnect_waits",
                         "policy_rejection_retry": False},
        "candidate_submission": {"policy": NOMINATION_POLICY, "per_agent_per_round": 1,
                                 "maximum_new_programs_per_round": 2, "include_incumbent": True,
                                 "earlier_submissions": "archived_not_evaluated"},
        "promotion": {"case_count": 192, "counts": dict(OFFICIAL_COUNTS), "metric": "overall.mean",
                      "guidance_reference": "each_run_entering_incumbent",
                      "common_within_run_and_round": True, "guided_cases_common_across_runs": False,
                      "base_and_base_perturbed_common_across_runs": True,
                      "shared_suite_manifest": f"{CONTROL}/comparison_suite.json",
                      "live_delegated_kinds": ["guided"]},
        "fixed_baseline": {"development_case_count": 512, "selection_case_count": 192,
                           "selection_counts": dict(OFFICIAL_COUNTS), "selection_metric": "mean_primary_error",
                           "selection_manifest": f"{CONTROL}/fixed_selection.json",
                           "guidance_reference": "frozen_common_s0", "selection_updates": "never",
                           "independent_of_terminal": True},
        "terminal": {"case_count": 768, "counts": dict(TERMINAL_COUNTS), "metric": "overall.mean",
                     "suite_manifest": f"{CONTROL}/terminal_suite.json",
                     "case_manifest": f"{CONTROL}/terminal.json",
                     "guidance_reference": "frozen_common_s0", "shared_across_all_five_methods": True,
                     "comparison_suite_manifest": f"{CONTROL}/comparison_suite.json"},
        "guidance": {"training_steps": 5000, "reward_feedback_case_count": 64,
                     "official_round_case_count": 192, "official_terminal_case_count": 768,
                     "config_hash_scope": "StructuredNSGuidanceConfig backend settings; excludes protocol case counts"},
        "failure_protocol": "complete_finite_selection_raw_external_v1",
    }
    destination.mkdir(parents=True, exist_ok=True)
    for path, payload in outputs.items():
        text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        path.write_text(text)
    return tuple(outputs)


if __name__ == "__main__":
    prepare()
    print("Prepared NS v3 fixed/Base/Full/Base+G/Base+P seed101 configurations; no jobs submitted.")
