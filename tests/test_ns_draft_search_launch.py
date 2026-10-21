"""Check the search configs against the real fixed-data CLI and budget contract."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pde_agent_verifier_solver.plugins.ns.baselines.search import NSBaselineConfig

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "configs/paper/ns_draft_v1"


@pytest.mark.parametrize("name", ["bon", "aide"])
def test_search_config_matches_main_baseline_information_and_total_budget(name):
    fixed = json.loads((DIRECTORY / "fixed_s101.json").read_text())
    payload = json.loads((DIRECTORY / f"{name}_s101.json").read_text())
    config = NSBaselineConfig(
        method=payload["method"], output_directory="/tmp/not-created",
        seed=payload["seed"], agent_budget_seconds=payload["agent"]["hard_wall_seconds"],
        **payload["search"],
    )
    assert config.seed == 101
    assert config.turn_count == 10
    assert config.agent_budget_seconds * config.turn_count == 40 * 3600
    for key in ("dataset", "initial_candidate", "evaluation"):
        assert payload[key] == fixed[key]
    for key in ("command", "worker_settings", "hard_wall_seconds", "environment"):
        assert payload["agent"][key] == fixed["agent"][key]
    assert "roles" not in payload["agent"]
    assert payload["agent"]["gpu_devices"] == ["0"]
    assert "active_budget_seconds" not in payload["agent"]
    data = json.loads((DIRECTORY / payload["dataset"]["manifest"]).read_text())
    assert set(data["splits"]) == {"development", "validation"}
    assert len(data["splits"]["development"]) == len(data["splits"]["validation"]) == 512
    if name == "bon":
        assert config.independent_candidates == 10 and config.repair_iterations == 0
    else:
        assert (config.iterations, config.aide_num_drafts, config.aide_debug_prob,
                config.aide_max_debug_depth) == (10, 5, 0.5, 3)


def test_search_manifest_preserves_serial_turns_and_one_gpu_mapping():
    manifest = json.loads((DIRECTORY / "search_batch.json").read_text())
    assert manifest["time_budget_mode"] == "hard_wall"
    assert manifest["agent_hard_wall_seconds"] == 14400
    assert {entry["method"] for entry in manifest["entries"]} == {"bon", "aide"}
    for entry in manifest["entries"]:
        assert entry["gpus"] == entry["agent_concurrency"] == 1
        assert entry["agent_gpu_allocation_indices"] == [0]
        assert entry["gpu_allocation_upper_bound"] == 2
        assert not entry["controller_gpu"]
        assert entry["planned_agent_turns"] == 10
        assert entry["agent_wall_hours_cap"] == 40
    assert manifest["aide_s0_counts_as_existing_draft"]
