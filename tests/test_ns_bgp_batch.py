"""Validate the approved five-arm v3 plan without generating Cases or running Agents."""
from __future__ import annotations

import os

import importlib.util
import json
from pathlib import Path
import shutil

import pytest

from pde_agent_verifier_solver.engine import (
    AgentSpec, CampaignCompiler, CampaignRecipe, EvaluatorPolicy, GeneratorRecipe, RoundSettings,
)
from pde_agent_verifier_solver.engine.guidance_controller import GuidanceProtocolConfig
from pde_agent_verifier_solver.plugins.ns.baselines.search import NSBaselineConfig
from pde_agent_verifier_solver.plugins.ns.structured_guidance_backend import StructuredNSGuidanceConfig
from pde_agent_verifier_solver.sdk import (
    AnswerQuery, Artifact, GeneratorCapabilities, GeneratorCheckpoint, GuidanceTrainingBudget,
    OfficialEvaluationSpec, TaskKind,
)

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "configs/paper/ns_bgp_v3"
PROMOTION = dict(original=0, base=64, guided=64, base_perturbed=64, guided_perturbed=0)
TERMINAL = dict(original=0, base=256, guided=256, base_perturbed=256, guided_perturbed=0)


def _script(name):
    spec = importlib.util.spec_from_file_location("ns_bgp_test_" + name, ROOT / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("method,source,permissions", [
    ("base", "ns_draft_v1", (False, False)),
    ("full", "ns_draft_v1", (True, True)),
    ("base_guided", "ns_gp_v2", (True, False)),
    ("base_perturbed", "ns_gp_v2", (False, True)),
])
def test_v3_framework_configs_compile_exact_evaluation_and_preserve_development(method, source, permissions):
    runner = _script("run_canonical_ns_campaign.py")
    config, path = runner._load_config(DIRECTORY / f"{method}_s101.json")
    runner._validate_config(config)
    # The runner expands site placeholders from the environment; compare against the same expansion.
    previous = json.loads(os.path.expandvars((ROOT / f"configs/paper/{source}/{method}_s101.json").read_text()))
    for name in ("development", "worker", "evaluator"):
        assert config[name] == previous[name]
    assert config["execution"] == {**previous["execution"], "nomination_policy": "latest", "transient_resume_max_attempts": 3}
    assert runner._generator_switches(config) == permissions
    assert runner._effective_generator_switches(config) == (True, True)
    for key in ("runtime", "artifacts", "artifact_sha256"):
        assert config["generator"][key] == previous["generator"][key]
    assert config["generator"]["guidance"]["backend"] == previous["generator"]["guidance"]["backend"]
    assert config["official"] == PROMOTION and sum(PROMOTION.values()) == 192
    assert config["promotion"]["settings"]["primary"] == {"distribution": "overall", "statistic": "mean"}
    assert "official_audit_manifest" not in config
    assert config["shared_evaluation_suite_manifest"] == "../../../../control/mixed_evaluation/comparison_suite.json"
    assert runner._config_path(path, config["dataset"]["manifest"]) == ROOT / "configs/paper/ns_draft_v1/data/development_manifest.json"
    assert runner._config_path(path, config["candidate"]["initial_bundle"]) == ROOT / "configs/paper/ns_draft_v1/s0_hold_initial_bundle"
    for name in ("script", "codex_config"):
        assert runner._config_path(path, config["worker"][name]).is_file()
    for agent in config["agents"]:
        assert "same 192" in agent["prompt"] and "0 Guided-perturbed" in agent["prompt"]
        assert "overall.mean" in agent["prompt"] and "entering incumbent" in agent["prompt"]
        assert "can differ across runs" in agent["prompt"]
        assert "Base and Base-perturbed Cases are frozen and shared across runs" in agent["prompt"]
        old_prompt = next(row["prompt"] for row in previous["agents"] if row["agent_id"] == agent["agent_id"])
        assert agent["prompt"].split("Development tools:", 1)[1].split("Use the previous selected program", 1)[0] == old_prompt.split("Development tools:", 1)[1].split("Use the previous selected program", 1)[0]
        assert "only your last committed nomination enters Promotion" in agent["prompt"]
        assert "submit all distinct" not in agent["prompt"]
    guidance = config["generator"]["guidance"]
    backend = StructuredNSGuidanceConfig(**guidance["backend"])
    assert backend.training_steps == 5000
    assert guidance["protocol"]["guidance_config_sha256"] == backend.sha256
    assert backend.sha256 == previous["generator"]["guidance"]["protocol"]["guidance_config_sha256"]
    protocol = GuidanceProtocolConfig(**{**guidance["protocol"],
        "training_budget": GuidanceTrainingBudget(**guidance["protocol"]["training_budget"])})
    assert (protocol.reward_feedback_case_count, protocol.official_round_case_count,
            protocol.official_terminal_case_count) == (64, 192, 768)
    terminal = runner._terminal_evaluation(config["terminal"])
    assert {cohort.cohort_id for cohort in terminal.cohorts} == {"common_base", "common_guided"}
    totals = {name: sum(cohort.suite.as_dict()[name] for cohort in terminal.cohorts) for name in TERMINAL}
    assert totals == TERMINAL and sum(totals.values()) == 768
    assert terminal.ranking.cohort_id == terminal.ranking.distribution == "overall"
    recipe = CampaignRecipe(
        name=config["campaign_id"], round_count=config["round_count"], master_seed=config["master_seed"],
        agents=tuple(AgentSpec(row["agent_id"], row["name"], row["prompt"]) for row in config["agents"]),
        default_round=RoundSettings(agent_budget_seconds=config["execution"]["agent_hard_wall_seconds"],
            development_sampling=runner._development_policy(config["development"]),
            official_evaluation=OfficialEvaluationSpec(round=runner._hidden_suite(config["official"]))),
        generator=GeneratorRecipe(guidance_enabled=True, perturbations_enabled=True,
            checkpoint=GeneratorCheckpoint("ns-bgp-test", object())),
        evaluator=EvaluatorPolicy(AnswerQuery.at_points(fields=("velocity_x", "velocity_y"),
            points=({"t": .01, "x": .2, "y": .3},))),
        initial_candidate=Artifact("S0", TaskKind.SOLVER, "Shared S0", object(), "test", "test", 0, "user"),
        terminal_evaluation=terminal, promotion=config["promotion"], terminal_score_distribution="overall",
    )
    campaign = CampaignCompiler().compile(recipe, GeneratorCapabilities(
        online=True, guidance=True, perturbations=True, trainable=False))
    assert campaign.master_seed == 101 and len(campaign.rounds) == 5
    assert all(len(plan.agents) == 2 and plan.agent_budget_seconds == 14400 for plan in campaign.rounds)
    assert all(plan.round_hidden.as_dict() == PROMOTION for plan in campaign.rounds)


def test_v3_fixed_baseline_retains_512_development_and_uses_independent_fixed_192_selection():
    config = json.loads((DIRECTORY / "fixed_s101.json").read_text())
    previous = json.loads((ROOT / "configs/paper/ns_draft_v1/fixed_s101.json").read_text())
    assert config["search"] == previous["search"]
    assert config["agent"]["nomination_policy"] == "latest"
    assert config["agent"]["transient_resume_max_attempts"] == 3
    assert "only your last committed nomination enters Promotion" in config["prompt"]
    assert config["evaluation"]["borrow_agent_gpus"] is True
    assert {k: v for k, v in config["evaluation"].items() if k != "borrow_agent_gpus"} == previous["evaluation"]
    assert {key: value for key, value in config["agent"].items() if key not in {"roles", "nomination_policy", "transient_resume_max_attempts"}} == {
        key: value for key, value in previous["agent"].items() if key not in {"roles", "nomination_policy", "transient_resume_max_attempts"}
    }
    assert [{key: value for key, value in role.items() if key != "prompt"}
            for role in config["agent"]["roles"]] == [
        {key: value for key, value in role.items() if key != "prompt"}
        for role in previous["agent"]["roles"]
    ]
    assert config["dataset"]["development_split"] == "development"
    assert "selection_split" not in config["dataset"]
    assert config["dataset"]["selection_manifest"] == "../../../../control/mixed_evaluation/fixed_selection.json"
    data = (DIRECTORY / config["dataset"]["manifest"]).resolve()
    assert len(json.loads(data.read_text())["splits"]["development"]) == 512
    assert (DIRECTORY / config["initial_candidate"]).resolve() == ROOT / "configs/paper/ns_draft_v1/s0_hold_initial_bundle"
    assert "same fixed 192" in config["prompt"] and "0 Guided-perturbed" in config["prompt"]
    assert "common S0 reference" in config["prompt"] and "stays fixed" in config["prompt"]
    assert "same fixed 512 validation" not in config["prompt"]
    plan = NSBaselineConfig(method=config["method"], output_directory=Path("/tmp/not-created"),
        seed=config["seed"], agent_budget_seconds=config["agent"]["hard_wall_seconds"],
        competition_roles=tuple(AgentSpec(row["agent_id"], row["name"], row["prompt"]) for row in config["agent"]["roles"]),
        **config["search"])
    assert plan.turn_count == 10 and plan.agent_budget_seconds == 14400


def test_v3_batch_has_only_authorized_five_arms_and_resolves_frozen_control_paths(tmp_path):
    manifest = json.loads((DIRECTORY / "bgp_batch.json").read_text())
    assert manifest["protocol_revision"] == "iclr_ns_bgp_192_single_final_v5"
    assert [row["method"] for row in manifest["entries"]] == ["fixed", "base", "full", "base_guided", "base_perturbed"]
    assert all(row["mode"] == row["method"] and (DIRECTORY / row["config"]).is_file() for row in manifest["entries"])
    assert [row["gpus"] for row in manifest["entries"]] == [2, 3, 3, 3, 3]
    assert manifest["promotion"]["counts"] == manifest["fixed_baseline"]["selection_counts"] == PROMOTION
    assert manifest["promotion"]["guided_cases_common_across_runs"] is False
    assert manifest["promotion"]["base_and_base_perturbed_common_across_runs"] is True
    assert manifest["agent_resume"] == {
        "maximum_attempts_including_initial": 3,
        "scope": "same_native_session_within_live_agent_turn",
        "budget": "original_hard_deadline_including_reconnect_waits",
        "policy_rejection_retry": False,
    }
    assert manifest["candidate_submission"] == {
        "policy": "latest", "per_agent_per_round": 1,
        "maximum_new_programs_per_round": 2, "include_incumbent": True,
        "earlier_submissions": "archived_not_evaluated",
    }
    assert manifest["promotion"]["case_count"] == manifest["fixed_baseline"]["selection_case_count"] == 192
    assert manifest["terminal"]["counts"] == TERMINAL
    assert manifest["terminal"]["shared_across_all_five_methods"] is True
    batch = tmp_path / "batch"
    frozen_config_directory = batch / "source/configs/paper/ns_bgp_v3"
    assert (frozen_config_directory / manifest["promotion"]["shared_suite_manifest"]).resolve() == batch / "control/mixed_evaluation/comparison_suite.json"
    assert (frozen_config_directory / manifest["terminal"]["suite_manifest"]).resolve() == batch / "control/mixed_evaluation/terminal_suite.json"
    assert (frozen_config_directory / manifest["fixed_baseline"]["selection_manifest"]).resolve() == batch / "control/mixed_evaluation/fixed_selection.json"


def test_prepare_v3_is_reproducible_and_does_not_modify_old_configs(tmp_path):
    for name in ("ns_draft_v1", "ns_gp_v2"):
        origin = ROOT / "configs/paper" / name
        target = tmp_path / "configs/paper" / name
        target.mkdir(parents=True)
        for path in origin.glob("*.json"):
            shutil.copy2(path, target / path.name)
    before = {path: path.read_bytes() for path in (tmp_path / "configs/paper").rglob("*.json")}
    module = _script("prepare_ns_bgp_batch.py")
    outputs = module.prepare(tmp_path)
    assert len(outputs) == 6
    assert all(path.read_bytes() == (DIRECTORY / path.name).read_bytes() for path in outputs)
    assert all(path.read_bytes() == contents for path, contents in before.items())
    assert module.prepare(tmp_path) == outputs


def test_source_freeze_includes_every_v3_config(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import sys
    root = tmp_path / "repo"
    target = root / "configs/paper/ns_bgp_v3"
    shutil.copytree(DIRECTORY, target)
    module = _script("freeze_ns_draft_source.py")
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module.subprocess, "run", lambda args, **kwargs: SimpleNamespace(
        stdout="test-commit\n" if "rev-parse" in args else ""))
    batch = tmp_path / "frozen-batch"
    monkeypatch.setattr(sys, "argv", ["freeze_ns_draft_source.py", "--batch-root", str(batch)])
    module.main()
    manifest = json.loads((batch / "source/snapshot_manifest.json").read_text())
    paths = {entry["path"] for entry in manifest["files"]}
    expected = {"configs/paper/ns_bgp_v3/" + path.name for path in DIRECTORY.glob("*.json")}
    assert expected <= paths and len(expected) == 6
    assert all((batch / "source" / path).read_bytes() == (root / path).read_bytes() for path in expected)
