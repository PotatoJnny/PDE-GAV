#!/usr/bin/env python3
"""Prepare the approved NS BoN/AIDE seed-101 controls; never submit jobs."""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIRECTORY = ROOT / "configs/paper/ns_draft_v1"


def build_payloads():
    fixed = json.loads((CONFIG_DIRECTORY / "fixed_s101.json").read_text())
    if fixed["method"] != "competitive_refinement" or fixed["seed"] != 101:
        raise ValueError("search controls require the shared seed-101 fixed baseline")
    if fixed["agent"]["hard_wall_seconds"] != 14400:
        raise ValueError("search controls require the shared four-hour hard-wall protocol")
    manifest = json.loads((CONFIG_DIRECTORY / fixed["dataset"]["manifest"]).read_text())
    development = manifest["splits"][fixed["dataset"]["development_split"]]
    selection = manifest["splits"][fixed["dataset"]["selection_split"]]
    if len(development) != 512 or len(selection) != 512:
        raise ValueError("search controls require 512 development and 512 validation Cases")
    if {row["case_id"] for row in development} & {row["case_id"] for row in selection}:
        raise ValueError("development and validation must be disjoint")
    common_prompt, marker, _ = fixed["prompt"].partition("\nUse the fixed 512 development Cases")
    if not marker:
        raise ValueError("fixed baseline prompt no longer has the shared NS input contract")
    with (ROOT / "docs/paper/SUBMISSION_EXECUTION_RUNS.csv").open(
        encoding="utf-8-sig", newline="",
    ) as stream:
        rows = list(csv.DictReader(stream))
    controls = (
        ("bon", "BoN", "bon_repair",
         {"independent_candidates": 10, "repair_iterations": 0},
         "Each independent branch starts from the same supplied S0 with empty branch "
         "history. Branches share no improvements or notes, and there are no repair turns."),
        ("aide", "AIDE", "aide_ns_adaptation",
         {"iterations": 10, "aide_num_drafts": 5, "aide_debug_prob": 0.5,
          "aide_max_debug_depth": 3},
         "Follow the Controller-provided draft, debug or improve stage and selected "
         "parent. S0 is an existing draft node; a new draft may start from scratch. "
         "The five-draft target includes S0; the budget allows ten new Agent proposals."),
    )
    payloads, entries = {}, []
    for name, label, method, search, instructions in controls:
        config = deepcopy(fixed)
        config["method"] = method
        config["search"] = search
        config["agent"].pop("roles", None)
        # The Slurm launcher replaces this token with the actual allocation's
        # first GPU. These retained searches execute one Agent turn at a time.
        config["agent"]["gpu_devices"] = ["0"]
        config["prompt"] = (
            common_prompt + "\nUse the fixed 512 development Cases and available "
            "evaluation tool; this run has no online case-generation or public oracle "
            "sessions. All proposals are selected using the same fixed 512 validation "
            "Cases. Each Agent turn has 14400 seconds of total elapsed time, including "
            "all tool calls and waiting. " + instructions + " Nominate exactly one "
            "final runnable solver bundle in this turn."
        )
        filename = f"{name}_s101.json"
        payloads[filename] = config
        row, = [row for row in rows if row["experiment"] == "01" and row["task"] == "NS"
                and row["method"] == label and row["seed"] == "101"]
        entries.append({
            "method": name, "kind": "baseline", "config": filename,
            "csv_run_id": row["run_id"], "gpus": 1, "gpu_allocation_upper_bound": 2,
            "agent_concurrency": 1, "agent_gpu_allocation_indices": [0],
            "controller_gpu": False, "planned_agent_turns": 10,
            "agent_wall_hours_cap": 40,
        })
    payloads["search_batch.json"] = {
        "schema": "ns_draft_search_batch_v1",
        "protocol_revision": "iclr_hard_wall_4h_single_seed_v2",
        "time_budget_mode": "hard_wall", "agent_hard_wall_seconds": 14400,
        "execution_status": "configured_not_submitted", "seed": 101,
        "entries": entries, "failure_protocol": "complete_finite_selection_raw_external_v1",
        "data_index": "data/bank_index.json",
        "shared_fixed_config": "fixed_s101.json",
        "scheduling": "One Agent turn at a time per method; BoN and AIDE may run in separate one-GPU jobs.",
        "aide_s0_counts_as_existing_draft": True,
        "requires_preflight": True,
    }
    return payloads


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check the saved configs without rewriting them")
    args = parser.parse_args(argv)
    payloads = build_payloads()
    for name, payload in payloads.items():
        path = CONFIG_DIRECTORY / name
        if args.check:
            if not path.is_file() or json.loads(path.read_text()) != payload:
                raise ValueError(f"search config differs from the approved preparation: {path}")
        else:
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print("Checked" if args.check else "Prepared", "BoN/AIDE seed101 configs; no jobs submitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
