#!/usr/bin/env python3
"""Prepare approved Full objective/interaction variants; never submit jobs."""
from copy import deepcopy
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "configs/paper/ns_draft_v1"
ARMS = {
    "full_mean_upfront": ("mean", "fixed_recipe", "full-framework_upfront-cases", "mean-error"),
    "full_tail_upfront": ("cvar", "fixed_recipe", "full-framework_upfront-cases", "worst-20pct-mean-error"),
    "full_tail_adaptive": ("cvar", "agent_adaptive", "full-framework_adaptive", "worst-20pct-mean-error"),
}
_MEAN_SENTENCE = """Selection minimizes untruncated mean u/v-component relative L2 on a complete,
finite validation cohort; execution/nonfinite failures make that program
ineligible, and exact ties retain the incumbent. Do not optimize pressure error."""


def build_configs(source):
    """Keep the current Full data, scientific backend, S0 and Official binding."""
    if source["master_seed"] != 101 or source["round_count"] != 5 or len(source["agents"]) != 2:
        raise ValueError("interaction batch requires the approved seed101 / 5 rounds / 2 Agents")
    if source["execution"]["agent_hard_wall_seconds"] != 14400:
        raise ValueError("interaction batch requires 4 hours total elapsed per Agent turn")
    if source["official"] != dict(original=0, base=1024, guided=0, base_perturbed=0, guided_perturbed=0):
        raise ValueError("interaction batch requires the common 1024 clean Base Official Cases")
    if not source["candidate"].get("initial_bundle"):
        raise ValueError("first-round Guided Cases require the common S0")
    result = {}
    for method, (statistic, strategy, _, _) in ARMS.items():
        config = deepcopy(source)
        config["campaign_id"] = f"ns-draft-hardwall-v2-{method.replace('_', '-')}-s101"
        config["execution"].pop("agent_active_budget_seconds", None)
        config["generator"].update(guidance_enabled=True, perturbations_enabled=True)
        config["promotion"] = {
            "preset": "minimize_metric",
            "settings": {"primary": {"distribution": "base", "statistic": statistic}},
        }
        config["terminal"]["ranking"]["statistic"] = statistic
        upfront = strategy == "fixed_recipe"
        config["development"] = {
            "strategy": strategy,
            "case_replay": False,
            # FIXED_RECIPE uses one Controller sampler and shares its whole
            # result with both Agents. This limit is NOT multiplied by Agents.
            "max_cases_per_agent_per_round": 1024 if upfront else 512,
            "allowed_sources": ["base", "guided"],
            "starter_only_sources": [],
            "starter": dict(original=0, base=0 if upfront else 16, guided=0 if upfront else 16,
                            base_perturbed=0, guided_perturbed=0),
        }
        if upfront:
            config["development"]["fixed_recipe"] = dict(
                original=0, base=256, guided=256, base_perturbed=256, guided_perturbed=256,
            )
        selection = (
            "Selection minimizes the mean of per-Case u/v-component relative L2 errors "
            "on the same complete 1024 clean Base Official Cases in each round."
            if statistic == "mean" else
            "Selection minimizes the mean of the largest 20% of per-Case u/v-component "
            "relative L2 errors on the same complete 1024 clean Base Official Cases in "
            "each round: sort each program's errors from largest to smallest and average "
            "the largest ceil(0.20 * 1024) = 205 errors (base.cvar). This is the actual "
            "Promotion rule; plan your development experiments to improve this tail objective."
        )
        selection += (
            " Execution/nonfinite failures make that program ineligible; exact ties retain "
            "the incumbent. The Official cohort contains no Guided/Perturbed Cases and is "
            "not the final external test. Do not optimize pressure error."
        )
        development = (
            "At the start of every round, both Agents receive the SAME complete batch of "
            "1024 development Cases: 256 Base, 256 Guided, 256 perturbed children of those "
            "Base parents, and 256 perturbed children of those Guided parents. Guided Cases "
            "use the entering selected program (S0 in round 1); the Guidance model is "
            "frozen during that round. No additional Cases or perturbations can be requested "
            "during the Agent turn. Choose how to evaluate and use this fixed batch."
            if upfront else
            "Development tools: Base, Guided and Perturbed. Each Agent starts every round "
            "with 16 Base and 16 Guided Cases, and may obtain at most 512 total new Cases "
            "including starter and perturbed children. Choose timing, source, quantities "
            "and perturbation parents to improve the stated selection objective. Guided "
            "Cases use the entering selected program (S0 in round 1); the Guidance model "
            "is frozen during that round."
        )
        for agent in config["agents"]:
            prefix, separator, _ = agent["prompt"].partition("\n\nDevelopment tools:")
            if not separator or _MEAN_SENTENCE not in prefix:
                raise ValueError("Full prompt changed; reconcile its selection and development instructions explicitly")
            agent["prompt"] = prefix.replace(_MEAN_SENTENCE, selection) + "\n\n" + development + (
                " Your turn lasts at most 4 hours total elapsed from Agent start, including "
                "all Generator, evaluator and other tool calls and waits. Use the previous "
                "selected program and cumulative Campaign Memory; submit all distinct "
                "runnable candidates worth common evaluation."
            )
        result[method] = config
    return result


def main():
    source = json.loads((DIRECTORY / "full_s101.json").read_text())
    configs = build_configs(source)
    with (ROOT / "docs/paper/SUBMISSION_EXECUTION_RUNS.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    entries = []
    for method, config in configs.items():
        _, _, label, metric = ARMS[method]
        row = next(row for row in rows if row["experiment"] == "03" and row["task"] == "NS"
                   and row["method"] == label and row["selection_metric"] == metric and row["seed"] == "101")
        name = f"{method}_s101.json"
        (DIRECTORY / name).write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
        entries.append({"method": method, "kind": "canonical", "config": name,
                        "csv_run_id": row["run_id"], "gpus": 3})
    reused = next(row for row in rows if row["experiment"] == "01" and row["task"] == "NS"
                  and row["method"] == "full-framework" and row["seed"] == "101")
    manifest = {
        "schema": "ns_draft_interaction_batch_v1",
        "protocol_revision": "iclr_hard_wall_4h_single_seed_v2",
        "time_budget_mode": "hard_wall", "agent_hard_wall_seconds": 14400,
        "execution_status": "prepared_not_submitted", "seed": 101,
        "entries": entries,
        "reused_entries": [{"method": "full_mean_adaptive", "config": "full_s101.json",
                            "csv_run_id": reused["run_id"], "submit": False}],
        "failure_protocol": "complete_finite_selection_raw_external_v1",
        "data_index": "data/bank_index.json",
        "comparison": {
            "purpose": "Full framework objective by overall interaction policy; this is not an equal-visible-case ablation.",
            "official": "All four cells use the same 1024 clean Base Official Cases per round and seed block.",
            "tail": "Canonical Controller base.cvar is the mean of the largest ceil(0.20*N) per-Case errors.",
            "upfront": "1024 fresh Cases generated once before each round and shared by both Agents; Base/Guided/Base-perturbed/Guided-perturbed each 256; no further draws.",
            "adaptive": "At most 1024 fresh team Cases per round, each Agent limited to 512 including its 16 Base + 16 Guided starter Cases and perturbed children.",
            "guidance": "Generated using the entering selected program (S0 in round 1); Guidance is frozen within each round.",
            "costs": "Report Agent elapsed time, total run elapsed time, Generator preparation/sampling and evaluator time separately; upfront preparation precedes the Agent clock.",
        },
    }
    (DIRECTORY / "interaction_batch.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print("Prepared 3 Full interaction/objective configs; mean adaptive reuses full_s101. No jobs submitted.")


if __name__ == "__main__":
    main()
