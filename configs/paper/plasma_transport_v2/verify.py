#!/usr/bin/env python3
"""Verify this published experiment without executing candidates or loading weights."""

from __future__ import annotations

import ast
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO / "src"))

from pde_agent_verifier_solver.sdk.candidate_bundle import inspect_candidate_bundle
from pde_agent_verifier_solver.plugins.plasma_transport import load_transport_calibration


def read_json(relative):
    return json.loads((ROOT / relative).read_text())


def read_csv(relative):
    with (ROOT / relative).open(newline="") as stream:
        return list(csv.DictReader(stream))


def check(condition, message):
    if not condition:
        raise ValueError(message)


def close(left, right):
    return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-15)


def verify():
    seal = read_json("release_manifest.json")
    declared = {item["path"] for item in seal["files"]}
    files = {str(p.relative_to(ROOT)) for p in ROOT.rglob("*")
             if p.is_file() and "__pycache__" not in p.parts
             and p.name != "release_manifest.json"}
    check(files == declared, "release file list differs from its manifest")
    for item in seal["files"]:
        path = ROOT / item["path"]
        check(not path.is_symlink() and path.resolve().is_relative_to(ROOT),
              f"invalid release path: {item['path']}")
        raw = path.read_bytes()
        check(len(raw) == item["bytes"], f"size mismatch: {item['path']}")
        check(hashlib.sha256(raw).hexdigest() == item["sha256"],
              f"hash mismatch: {item['path']}")

    index = read_json("solutions/INDEX.json")
    entries = index["programs"]
    check(index["program_count"] == len(entries) == 77, "expected 77 archived bundles")
    lookup = {(p["arm"], p["artifact_id"]): p for p in entries}
    check(len(lookup) == len(entries), "duplicate program entry")
    check(sum(p["is_initial_program"] for p in entries) == 6, "expected six S0 entries")
    check(sum(p["is_final_program"] for p in entries) == 6, "expected six final programs")
    parsed = 0
    for entry in entries:
        bundle = inspect_candidate_bundle(ROOT / entry["bundle"])
        check(bundle.tree_sha256 == entry["tree_sha256"], f"bundle identity: {entry['bundle']}")
        check(bundle.total_bytes == entry["total_bytes"], f"bundle size: {entry['bundle']}")
        actual = [(m.path, m.byte_length, m.sha256) for m in bundle.members]
        expected = sorted((m["path"], m["bytes"], m["sha256"]) for m in entry["files"])
        check(actual == expected, f"bundle members: {entry['bundle']}")
        manifest = bundle.manifest_path.read_bytes()
        check(hashlib.sha256(manifest).hexdigest() == entry["manifest_sha256"],
              f"original manifest identity: {entry['bundle']}")
        for member in bundle.members:
            ast.parse((bundle.root / member.path).read_text(), filename=member.path)
            parsed += 1

    protocol = read_json("protocol.json")
    provenance = read_json("provenance.json")
    calibration_dict = read_json("metric_calibration.json")
    calibration = load_transport_calibration(calibration_dict)
    check(calibration.objective_hash == protocol["scoring"]["objective_hash"],
          "transport objective mismatch")
    check(dict(calibration.objective_identity["weights"]) == protocol["scoring"]["weights"],
          "transport weights mismatch")
    policies = read_json("results/selection_policies.json")
    for arm in protocol["arms"]:
        path = ROOT / f"recorded_configs/{arm}.json"
        source = provenance["sources"][f"config_{arm}"]
        check(hashlib.sha256(path.read_bytes()).hexdigest()
              == source.get("published_sha256", source["sha256"]), "config changed")
        config = json.loads(path.read_text())
        check(config["round_count"] == 6 and config["master_seed"] == 101, "run identity")
        check(len(config["agents"]) == 2, "Agent count")
        check(config["execution"]["agent_hard_wall_seconds"] == 10800, "Agent budget")
        check(config["execution"]["candidate_case_hard_wall_seconds"] == 900, "Case budget")
        check(config["task"]["metric_calibration"] == calibration_dict, "config calibration")
        check(policies[arm]["current_selection_policy"] == protocol["effective_selection_policy"],
              "effective selection policy mismatch")
        for agent in config["agents"]:
            path = ROOT / f"prompts/{config['equation']}/{agent['agent_id']}.md"
            check(path.read_bytes() == agent["prompt"].encode(), "prompt bytes changed")
            check(hashlib.sha256(path.read_bytes()).hexdigest()
                  == config["protocol"]["agent_prompt_sha256"][agent['agent_id']],
                  "prompt hash mismatch")

    selections = read_csv("results/selection.csv")
    decisions = read_json("results/selection_decisions.json")
    selected = {}
    exceptions = {(r["arm"], r["round"]): r for r in protocol["historical_selection_exceptions"]}
    check(len(selections) == 36, "expected 36 round selections")
    for row in selections:
        arm, rnd = row["arm"], int(row["round"])
        check((arm, rnd) not in selected, "duplicate round selection")
        entry = lookup[(arm, row["artifact_id"])]
        check(rnd in entry["selected_in_rounds"], "selection absent from program index")
        check(entry["tree_sha256"] == row["tree_sha256"], "selected program identity")
        expected_count = exceptions.get((arm, rnd), {}).get("case_count", 48)
        check(int(row["official_case_count"]) == expected_count, "historical selection count")
        native = next(r for r in decisions[arm] if r["round_index"] == rnd)
        check(native["promotion_decision"]["selected_digest"] == row["artifact_id"],
              "native selected program differs")
        contender = next(c for c in native["contenders"] if c["artifact_id"] == row["artifact_id"])
        check(contender["distributions"]["overall"]["case_count"] == expected_count,
              "native selection count differs")
        check(close(contender["distributions"]["overall"]["mean"], row["official_selection_mean"]),
              "native selection mean differs")
        selected[(arm, rnd)] = entry
    for arm in protocol["arms"]:
        check(all((arm, rnd) in selected for rnd in range(1, 7)), "missing round")
        check(selected[(arm, 6)]["is_final_program"], "final program is not S6")

    native_cases = read_csv("results/native_case_scores.csv")
    native_rounds = read_csv("results/native_round_scores.csv")
    grouped = defaultdict(list)
    for row in native_cases:
        grouped[(row["arm"], row["artifact_id"])].append(row)
    check(len(native_cases) == 2240 and len(grouped) == 35, "native terminal coverage")
    check(len(native_rounds) == 36, "native round coverage")
    for row in native_rounds:
        check(selected[(row["arm"], int(row["round_index"]))]["artifact_id"] == row["artifact_id"],
              "terminal program differs from selection")
        cases = grouped[(row["arm"], row["artifact_id"])]
        check(len(cases) == int(row["expected_cases"]) == 64, "terminal case count")
        check(close(math.fsum(float(c["observed_numeric_error"]) for c in cases) / 64,
                    row["observed_mean"]), "terminal scalar aggregation")

    for label in ("artemis", "vpfp_original", "vpfp_completed"):
        cases = read_csv(f"results/independent/{label}_per-case.csv")
        rounds = read_csv(f"results/independent/{label}_round-summary.csv")
        grouped = defaultdict(list)
        for row in cases:
            arm = f"artemis-{row['model']}" if label == "artemis" else row["arm"]
            rnd = int(row["round"])
            entry = lookup[(arm, "candidate-s0")] if rnd == 0 else selected[(arm, rnd)]
            tree_key = "candidate_tree_sha256" if label == "artemis" else "tree_sha256"
            check(row[tree_key] == entry["tree_sha256"], "independent program identity")
            grouped[(arm, rnd)].append(row)
        for row in rounds:
            arm = f"artemis-{row['model']}" if label == "artemis" else row["arm"]
            count = 6 if label == "artemis" else 8
            group = grouped[(arm, int(row["round"]))]
            check(len(group) == count, "independent case coverage")
            value_key = "mean_primary_error" if label == "artemis" else "primary_error"
            if row["status"] == "complete":
                check(all(r["status"] == "completed" for r in group), "incomplete independent group")
                check(close(math.fsum(float(r["primary_error"]) for r in group) / count,
                            row[value_key]), "independent scalar aggregation")
            else:
                check(row[value_key] == "", "partial-case mean published")
        if label == "vpfp_completed":
            supplements = Counter((r["arm"], r["round"]) for r in cases
                                  if r["accuracy_source"] == "declared_timeout_supplement")
            check(supplements == {("strong-sol", "6"): 8, ("strong-astra", "2"): 3},
                  "timeout-supplement scope differs")

    return {"status": "passed", "release_files": len(files), "bundles": len(entries),
            "python_files_parsed": parsed, "round_selections": len(selections),
            "native_terminal_cases": len(native_cases), "independent_round_rows": 42,
            "candidate_execution": False, "model_loading": False}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
