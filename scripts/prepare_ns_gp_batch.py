#!/usr/bin/env python3
"""Prepare the two online NS G/P ablation arms without changing the first batch."""
from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_REVISION = "iclr_hard_wall_4h_single_seed_v2"
ARMS = (
    ("base_guided", "base-plus-guided", "full", True, False),
    ("base_perturbed", "base-plus-perturbed", "base", False, True),
)


def prepare(root: Path = ROOT) -> tuple[Path, ...]:
    """Derive each arm from its matching first-batch starter and scientific inputs."""
    first = root / "configs/paper/ns_draft_v1"
    destination = root / "configs/paper/ns_gp_v2"
    with (root / "docs/paper/SUBMISSION_EXECUTION_RUNS.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    outputs = {}
    entries = []
    for method, label, source, guided, perturbed in ARMS:
        config = deepcopy(json.loads((first / f"{source}_s101.json").read_text()))
        # This arm differs only in enabled tools, their prompt, and run identity.
        assert config["master_seed"] == 101
        assert config["round_count"] == 5 and len(config["agents"]) == 2
        assert config["execution"]["agent_hard_wall_seconds"] == 14400
        assert "agent_active_budget_seconds" not in config["execution"]
        assert config["development"]["max_cases_per_agent_per_round"] == 512
        config["campaign_id"] = f"ns-gp-hardwall-v2-{method}-s101"
        config["dataset"]["manifest"] = "../ns_draft_v1/data/development_manifest.json"
        config["candidate"]["initial_bundle"] = "../ns_draft_v1/s0_hold_initial_bundle"
        config["generator"].update(
            guidance_enabled=guided,
            perturbations_enabled=perturbed,
        )
        config["development"]["allowed_sources"] = ["base", "guided"] if guided else ["base"]
        config["development"]["starter"] = dict(
            original=0, base=16 if guided else 32, guided=16 if guided else 0,
            base_perturbed=0, guided_perturbed=0,
        )
        previous_tools = (
            "Development tools: Base, Guided and Perturbed."
            if guided else
            "Development tools: Base only; Guided and Perturbed are disabled."
        )
        enabled_tools = (
            "Development tools: Base and Guided; Perturbed is disabled."
            if guided else
            "Development tools: Base and Perturbed; Guided is disabled."
        )
        for agent in config["agents"]:
            if agent["prompt"].count(previous_tools) != 1:
                raise ValueError("first-batch tool prompt changed; review the G/P derivation")
            agent["prompt"] = agent["prompt"].replace(previous_tools, enabled_tools)
        matches = [
            row for row in rows
            if row["experiment"] == "02" and row["task"] == "NS"
            and row["method"] == label and row["seed"] == "101"
        ]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one NS seed101 execution row for {label}")
        row = matches[0]
        if (
            row["budget_revision"] != PROTOCOL_REVISION
            or row["time_budget_mode"] != "hard_wall"
            or row["hard_wall_minutes_per_agent_per_unit"] != "240"
            or row["agents"] != "2" or row["units"] != "5"
        ):
            raise ValueError(f"NS execution row uses a different protocol: {label}")
        name = f"{method}_s101.json"
        outputs[destination / name] = config
        entries.append(dict(
            method=method, kind="canonical", config=name,
            csv_run_id=row["run_id"], gpus=3,
        ))
    outputs[destination / "gp_batch.json"] = dict(
        schema="ns_gp_batch_v1",
        protocol_revision=PROTOCOL_REVISION,
        time_budget_mode="hard_wall",
        agent_hard_wall_seconds=14400,
        execution_status="prepared_not_submitted",
        seed=101,
        entries=entries,
        reference_batch_manifest="../ns_draft_v1/first_batch.json",
        data_index="../ns_draft_v1/data/bank_index.json",
        official_audit_manifest="../../../../control/official_s101.json",
        failure_protocol="complete_finite_selection_raw_external_v1",
    )
    destination.mkdir(parents=True, exist_ok=True)
    for path, payload in outputs.items():
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return tuple(outputs)


if __name__ == "__main__":
    prepare()
    print("Prepared online NS Base+G and Base+P seed101; first batch unchanged; no jobs submitted.")
