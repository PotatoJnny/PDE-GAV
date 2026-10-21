#!/usr/bin/env python3
"""Write Base, Base+G, Base+P and Base+G+P configurations for the NS runner.

Each arm uses the same number of fresh, shared Development Cases per round,
including perturbed children. This is a fixed acquisition schedule, not a
static/replayed dataset or an equal-cost claim. Official and Terminal use the
same clean Base specifications and seed; Promotion minimizes their Base mean.
The source recipe supplies the numerical data, checkpoints, Agents and budgets.
No data/checkpoint loading, GPU work or model calls occur here.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Mapping, Sequence

from run_canonical_ns_campaign import (
    _config_path,
    _integer,
    _load_config,
    _section,
    _text,
    _validate_config,
)


ARMS = {
    "base": (False, False),
    "base_g": (True, False),
    "base_p": (False, True),
    "base_g_p": (True, True),
}
COUNT_KEYS = ("original", "base", "guided", "base_perturbed", "guided_perturbed")


def _counts(total: int, guidance: bool, perturbations: bool) -> dict[str, int]:
    counts = dict.fromkeys(COUNT_KEYS, 0)
    guided_total = total // 2 if guidance else 0
    for source, count in (("base", total - guided_total), ("guided", guided_total)):
        children = count // 2 if perturbations else 0
        counts[source] = count - children
        counts[f"{source}_perturbed"] = children
    return counts


def _development(total: int, guidance: bool, perturbations: bool) -> dict[str, object]:
    return {
        "strategy": "fixed_recipe",
        "case_replay": False,
        "max_cases_per_agent_per_round": total,
        "allowed_sources": ["base", "guided"] if guidance else ["base"],
        "starter_only_sources": [],
        "starter": dict.fromkeys(COUNT_KEYS, 0),
        "fixed_recipe": _counts(total, guidance, perturbations),
        "hide_case_types": True,
    }


def build_configs(
    source: Mapping[str, object],
    source_path: Path,
    *,
    cases_per_round: int | None = None,
    official_cases: int | None = None,
    terminal_cases: int | None = None,
    without_initial_candidate: bool = False,
) -> dict[str, dict[str, object]]:
    if source.get("profile") != "structured_numerical":
        raise ValueError("G/P configuration generation currently supports structured numerical NS")
    if source.get("round_overrides"):
        raise ValueError("provide a source without round_overrides; the builder writes the cold-start override")
    if (cases_per_round is None
            and _section(source, "development").get("max_cases_per_agent_per_round") is None):
        raise ValueError("fixed-recipe ablations require an explicit --cases-per-round with an unlimited source")
    total = cases_per_round if cases_per_round is not None else _integer(
        _section(source, "development"), "max_cases_per_agent_per_round", minimum=4,
    )
    official = official_cases if official_cases is not None else _integer(
        _section(source, "official"), "base", minimum=1,
    )
    terminal = terminal_cases if terminal_cases is not None else sum(
        row["suite"]["base"] for row in source["terminal"]["cohorts"]
    )
    for label, value, minimum in (
        ("cases_per_round", total, 4), ("official_cases", official, 1), ("terminal_cases", terminal, 1),
    ):
        _integer({label: value}, label, minimum=minimum)
    common = deepcopy(dict(source))
    # Emitted recipes remain runnable when written outside the source directory.
    common["dataset"]["manifest"] = str(_config_path(source_path, common["dataset"]["manifest"]))
    common["generator"]["artifacts"] = {
        name: str(_config_path(source_path, raw))
        for name, raw in common["generator"]["artifacts"].items()
    }
    for key in ("script", "codex_config"):
        common["worker"][key] = str(_config_path(source_path, common["worker"][key]))
    if without_initial_candidate:
        common["candidate"]["initial_bundle"] = None
    elif common["candidate"].get("initial_bundle") is not None:
        common["candidate"]["initial_bundle"] = str(
            _config_path(source_path, common["candidate"]["initial_bundle"])
        )
    common["official"] = _counts(official, False, False)
    common["terminal"] = {
        "cohorts": [{
            "cohort_id": "common_base", "guidance_source": "base",
            "suite": _counts(terminal, False, False),
        }],
        "ranking": {"cohort_id": "common_base", "distribution": "base", "statistic": "mean"},
    }
    common["promotion"] = {
        "preset": "minimize_metric",
        "settings": {"primary": {"distribution": "base", "statistic": "mean"}},
    }
    # This compatibility identity records the actual clean Base feedback size.
    common["generator"]["guidance"]["protocol"]["reward_feedback_case_count"] = official
    result = {}
    for name, (guidance, perturbations) in ARMS.items():
        config = deepcopy(common)
        config["campaign_id"] = f"{_text(source, 'campaign_id')}-{name}"
        config["generator"]["guidance_enabled"] = guidance
        config["generator"]["perturbations_enabled"] = perturbations
        config["development"] = _development(total, guidance, perturbations)
        if guidance and config["candidate"].get("initial_bundle") is None:
            # An explicit first-round recipe preserves the total Case count
            # before an incumbent exists; later rounds can request Guidance.
            config["round_overrides"] = {
                "1": {"development": _development(total, False, perturbations)},
            }
        config["_config_path"] = str(source_path.resolve())
        _validate_config(config, inspect_initial_candidate=False)
        config.pop("_config_path", None)
        result[name] = config
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="A new directory; existing files are never overwritten.")
    parser.add_argument("--cases-per-round", type=int, help="Fresh shared Cases, including children; at least four.")
    parser.add_argument("--official-cases", type=int)
    parser.add_argument("--terminal-cases", type=int)
    parser.add_argument("--without-initial-candidate", action="store_true")
    args = parser.parse_args(argv)
    source, path = _load_config(args.config)
    configs = build_configs(
        source, path, cases_per_round=args.cases_per_round,
        official_cases=args.official_cases, terminal_cases=args.terminal_cases,
        without_initial_candidate=args.without_initial_candidate,
    )
    destination = args.output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    for name, config in configs.items():
        output = destination / f"{name}.json"
        output.write_text(json.dumps(config, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
