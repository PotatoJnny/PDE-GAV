from __future__ import annotations

import os

from copy import deepcopy
import csv
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from pde_agent_verifier_solver.engine import (
    AgentSpec, CampaignCompiler, CampaignRecipe, EvaluatorPolicy, GeneratorRecipe,
    RoundSettings,
)
from pde_agent_verifier_solver.engine.unified_runtime import _InteractiveSampler
from pde_agent_verifier_solver.sdk import (
    AnswerQuery, Artifact, Case, CaseSource, CaseVisibility, GeneratorCapabilities,
    GeneratorCheckpoint, OfficialEvaluationSpec, PerturbCasesRequest, SampleBatch,
    SampleCasesRequest, SampleSource, TaskKind,
)


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "configs/paper/ns_gp_v2"


def _load_script(name):
    spec = importlib.util.spec_from_file_location("ns_gp_test_" + name, ROOT / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "method,source,expected_flags,expected_starter",
    [
        ("base_guided", "full", (True, False), (16, 16)),
        ("base_perturbed", "base", (False, True), (32, 0)),
    ],
)
def test_gp_configs_preserve_scientific_protocol_and_compile(
    method, source, expected_flags, expected_starter,
):
    runner = _load_script("run_canonical_ns_campaign.py")
    config, path = runner._load_config(DIRECTORY / f"{method}_s101.json")
    runner._validate_config(config)
    # The runner expands site placeholders from the environment; compare against the same expansion.
    original = json.loads(os.path.expandvars((ROOT / "configs/paper/ns_draft_v1" / f"{source}_s101.json").read_text()))
    assert runner._generator_switches(config) == expected_flags
    for key in ("execution", "official", "terminal", "promotion", "evaluator", "worker"):
        assert config[key] == original[key]
    for key in ("runtime", "artifacts", "artifact_sha256", "guidance"):
        assert config["generator"][key] == original["generator"][key]
    for key, value in (
        ("manifest", config["dataset"]["manifest"]),
        ("initial_bundle", config["candidate"]["initial_bundle"]),
    ):
        resolved = runner._config_path(path, value)
        assert resolved.exists()
        parent = "dataset" if key == "manifest" else "candidate"
        expected = ROOT / "configs/paper/ns_draft_v1" / original[parent][key]
        assert resolved == expected.resolve()
    policy = runner._development_policy(config["development"])
    assert (policy.starter.base, policy.starter.guided) == expected_starter
    assert policy.starter.base_perturbed == policy.starter.guided_perturbed == 0
    assert policy.max_cases_per_agent_per_round == 512
    assert config["official_audit_manifest"] == original["official_audit_manifest"]
    recipe = CampaignRecipe(
        name=config["campaign_id"],
        round_count=config["round_count"],
        agents=tuple(AgentSpec(row["agent_id"], row["name"], row["prompt"]) for row in config["agents"]),
        default_round=RoundSettings(
            agent_budget_seconds=config["execution"]["agent_hard_wall_seconds"],
            development_sampling=policy,
            official_evaluation=OfficialEvaluationSpec(round=runner._hidden_suite(config["official"])),
        ),
        generator=GeneratorRecipe(
            guidance_enabled=expected_flags[0], perturbations_enabled=expected_flags[1],
            checkpoint=GeneratorCheckpoint("ns-gp-test", object()),
        ),
        evaluator=EvaluatorPolicy(AnswerQuery.at_points(
            fields=("velocity_x", "velocity_y"), points=({"t": 0.01, "x": 0.2, "y": 0.3},),
        )),
        initial_candidate=Artifact("common-s0", TaskKind.SOLVER, "S0", object(), "test", "test", 0, "user"),
        master_seed=config["master_seed"],
        terminal_evaluation=runner._terminal_evaluation(config["terminal"]),
        promotion=config["promotion"],
    )
    compiled = CampaignCompiler().compile(recipe, GeneratorCapabilities(
        online=True, guidance=True, perturbations=True, trainable=False,
    ))
    assert compiled.master_seed == 101 and len(compiled.rounds) == 5
    assert all(len(plan.agents) == 2 and plan.agent_budget_seconds == 14400 for plan in compiled.rounds)
    assert compiled.generator.guidance_enabled == expected_flags[0]
    assert compiled.generator.perturbations_enabled == expected_flags[1]


@pytest.mark.parametrize("method", ["base_guided", "base_perturbed"])
def test_gp_controller_tool_permissions_follow_actual_config(method):
    runner = _load_script("run_canonical_ns_campaign.py")
    config, _ = runner._load_config(DIRECTORY / f"{method}_s101.json")
    guidance, perturbations = runner._generator_switches(config)
    records = {}

    class Inner:
        def sample_cases(self, request):
            case = Case(request.source.value, CaseSource.GENERATED, CaseVisibility.DEVELOPMENT, {})
            records[case.case_id] = case
            return SampleBatch((case,), 511)

        def perturb_cases(self, request):
            case = Case("child", CaseSource.PERTURBED, CaseVisibility.DEVELOPMENT, {}, parent_case_id=request.parent_case_ids[0])
            records[case.case_id] = case
            return SampleBatch((case,), 510)

    evaluator = SimpleNamespace(
        _candidate_cases_for_controller_ids=lambda ids: tuple(records[case_id] for case_id in ids),
        _controller_case_ids_for_candidate_ids=lambda ids: tuple(ids),
    )
    sampler = _InteractiveSampler(
        Inner(), evaluator,
        runner._development_policy(config["development"]).allowed_sources,
        allow_guidance=guidance, allow_perturbations=perturbations,
    )
    sampler.seal_starter()
    assert sampler.sample_cases(SampleCasesRequest(SampleSource.BASE, 1)).cases[0].case_id == "base"
    if guidance:
        assert sampler.sample_cases(SampleCasesRequest(SampleSource.GUIDED, 1)).cases[0].case_id == "guided"
        with pytest.raises(PermissionError, match="perturbation is disabled"):
            sampler.perturb_cases(PerturbCasesRequest(("base",)))
    else:
        with pytest.raises(PermissionError, match="guided sampling is not enabled"):
            sampler.sample_cases(SampleCasesRequest(SampleSource.GUIDED, 1))
        child = sampler.perturb_cases(PerturbCasesRequest(("base",))).cases[0]
        assert child.parent_case_id == "base"
        assert sampler.counts["base_perturbed"] == 1


def test_prepare_gp_batch_reproduces_only_new_files_and_joins_execution_rows(tmp_path):
    preparer = _load_script("prepare_ns_gp_batch.py")
    first = tmp_path / "configs/paper/ns_draft_v1"
    first.mkdir(parents=True)
    for source in (ROOT / "configs/paper/ns_draft_v1").glob("*.json"):
        shutil.copyfile(source, first / source.name)
    plan = tmp_path / "docs/paper/SUBMISSION_EXECUTION_RUNS.csv"
    plan.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "docs/paper/SUBMISSION_EXECUTION_RUNS.csv", plan)
    before = {path: path.read_bytes() for path in first.iterdir()}
    outputs = preparer.prepare(tmp_path)
    assert len(outputs) == 3
    assert before == {path: path.read_bytes() for path in first.iterdir()}
    assert all(path.parent.name == "ns_gp_v2" for path in outputs)
    assert all(path.read_bytes() == (DIRECTORY / path.name).read_bytes() for path in outputs)
    payload = json.loads((DIRECTORY / "gp_batch.json").read_text())
    rows = list(csv.DictReader(plan.open(encoding="utf-8-sig")))
    for entry in payload["entries"]:
        matches = [row for row in rows if row["run_id"] == entry["csv_run_id"]]
        assert len(matches) == 1 and matches[0]["experiment"] == "02"
        assert entry["gpus"] == 3 and entry["kind"] == "canonical"
    assert payload["execution_status"] == "prepared_not_submitted"
