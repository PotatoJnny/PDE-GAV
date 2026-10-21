"""Small synthetic contracts only; not real Solver/Guidance validation."""
from collections import Counter
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("vpfp_campaign_paircheck_test", ROOT / "scripts/run_vpfp_campaign_guidance_paircheck.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


@pytest.fixture
def config():
    return {"schema": RUNNER.SCHEMA, "guide_round": 2, "root_count": 32, "point_count": 256,
        "condition_strata": deepcopy(RUNNER.STRATA), "root_master_seed": 10001,
        "bootstrap_seed": 10002, "random_seed": 10003, "bootstrap_replicates": 40,
        "candidate_workers": 4, "wall_seconds": 100., "guidance_sha256": "a"*64,
        "candidates": [{"id": "S1", "role": "guidance_target", "sha256": "b"*64},
                       {"id": "S2", "role": "final_delivery", "sha256": "c"*64}]}


def test_fixed_design_and_balanced_disjoint_roots(config):
    assert RUNNER.validate_config(config) == config
    roots = RUNNER.registered_roots(config, {0, 1})
    assert roots == RUNNER.registered_roots(config, {0, 1})
    assert len({row["root_seed"] for row in roots}) == 32
    assert len({row["query_seed"] for row in roots}) == 32
    assert Counter(row["stratum"] for row in roots) == Counter({i: 4 for i in range(8)})
    for row in roots:
        nu, temperature = row["conditions"]
        assert 10+5*(row["stratum"]//2) < nu < 15+5*(row["stratum"]//2)
        assert .8+.2*(row["stratum"]%2) < temperature < 1.+.2*(row["stratum"]%2)
    with pytest.raises(ValueError, match="overlaps history"):
        RUNNER.registered_roots(config, {roots[0]["root_seed"]})


@pytest.mark.parametrize("field,value", [("root_count", 16), ("point_count", 128),
    ("guide_round", 1), ("root_master_seed", True), ("candidate_workers", 0),
    ("wall_seconds", float("nan")), ("guidance_sha256", "wrong")])
def test_protocol_changes_rejected(config, field, value):
    config[field] = value
    with pytest.raises(ValueError):
        RUNNER.validate_config(config)


def test_history_includes_sampling_and_both_feedback_manifests(tmp_path):
    (tmp_path / "guidance_audits.json").write_text(json.dumps({"sampling": [{"root_seeds": [1, 2]}]}))
    for index, root in ((1, 3), (2, 4)):
        directory = tmp_path / "private_backend" / "guidance" / f"round-{index:03d}"
        directory.mkdir(parents=True)
        (directory / "feedback-manifest.json").write_text(json.dumps({"rows": [{"root_seed": root}]}))
    assert RUNNER.historical_roots(tmp_path) == {1, 2, 3, 4}


def test_arm_order_rotates_within_strata_and_preserves_all_arms():
    orders = [RUNNER.arm_order(index) for index in range(32)]
    assert all(set(order) == set(RUNNER.ARMS) for order in orders)
    assert all(len({orders[index] for index in range(stratum, 32, 8)}) == 3 for stratum in range(8))
    with pytest.raises(ValueError):
        RUNNER.arm_order(-1)


def test_common_private_query_is_checked_not_merely_seed_claimed():
    from pde_agent_verifier_solver.sdk import AnswerQuery
    points = tuple({"t": (i+.5)/256, "x": .1, "y": .2} for i in range(256))
    query = AnswerQuery.at_points(fields=("rho",), points=points)
    assert RUNNER.verify_same_queries([query, deepcopy(query)]) == RUNNER.query_sha(query)
    changed = AnswerQuery.at_points(fields=("rho",), points=tuple(reversed(points)))
    with pytest.raises(ValueError, match="changed private"):
        RUNNER.verify_same_queries([query, changed])
    short = AnswerQuery.at_points(fields=("rho",), points=points[:128])
    with pytest.raises(ValueError, match="256"):
        RUNNER.verify_same_queries([short])


def test_failure_and_pending_slots_are_not_removed_or_imputed(config):
    rows = RUNNER.pending_rows(RUNNER.registered_roots(config, set()), config)
    assert sum(len(scores) for row in rows for scores in row["scores"].values()) == 192
    for row in rows:
        for scores in row["scores"].values():
            for arm, score in scores.items():
                score.update(status="completed", valid=True, error=1. if arm == "base" else 1.2,
                             attempt_count=1)
    rows[0]["scores"]["S1"]["guided_relative"].update(status="failed", valid=False, error=None)
    result = RUNNER.summarize(rows, config)
    s1 = result["S1"]["guided_relative_vs_base"]
    assert s1["registered_roots"] == 32
    assert s1["paired"]["paired_valid_roots"] == 31
    assert s1["lost_valid_base_roots"] == 1
    assert s1["conclusion"] == "not_demonstrated"
    assert s1["failed_or_missing_errors_imputed"] is False
    assert s1["primary"] is True
    assert s1["secondary_transfer"] is False
    assert result["S2"]["guided_relative_vs_base"]["all_32_pairs_valid"] is True
    assert result["S2"]["guided_relative_vs_base"]["primary"] is False
    assert result["S2"]["guided_relative_vs_base"]["secondary_transfer"] is True
    random = result["S2"]["guided_relative_vs_random_relative"]
    assert random["paired"]["mean_delta"] == 0.
    assert random["arm_mapping"] == {"base": "random_relative", "guided": "guided_relative"}


def test_real_execution_cannot_run_on_login_node(config, monkeypatch, tmp_path):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_TMPDIR", raising=False)
    with pytest.raises(RuntimeError, match="Slurm compute"):
        RUNNER.run(config, tmp_path / "output", wall_seconds=10.)
    assert not (tmp_path / "output").exists()
