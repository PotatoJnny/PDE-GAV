from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from pde_agent_verifier_solver.engine import GeneratorRecipe
from pde_agent_verifier_solver.sdk import GeneratorCapabilities, SamplingStrategy


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "configs/ns_canonical_structured_5round.json"


@pytest.fixture
def builder(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("_ns_ablation_builder", ROOT / "scripts/build_ns_ablation_configs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runner(builder):
    import run_canonical_ns_campaign
    return run_canonical_ns_campaign


@pytest.mark.parametrize("total", (4, 5, 7, 512))
@pytest.mark.parametrize("cold_start", (False, True))
def test_four_arms_preserve_total_feedback_protocol_and_paths(builder, runner, total, cold_start):
    source, path = runner._load_config(SOURCE)
    before = deepcopy(source)
    configs = builder.build_configs(
        source, path, cases_per_round=total, official_cases=32, terminal_cases=6,
        without_initial_candidate=cold_start,
    )
    assert source == before
    assert set(configs) == {"base", "base_g", "base_p", "base_g_p"}
    for name, config in configs.items():
        guidance, perturbations = builder.ARMS[name]
        config["_config_path"] = str(SOURCE)
        runner._validate_config(config, inspect_initial_candidate=False)
        assert runner._generator_switches(config) == (guidance, perturbations)
        assert config["execution"] == source["execution"]
        assert config["agents"] == source["agents"]
        assert config["master_seed"] == source["master_seed"]
        assert config["official"] == builder._counts(32, False, False)
        assert config["terminal"] == configs["base"]["terminal"]
        assert config["promotion"] == configs["base"]["promotion"]
        assert config["terminal"]["cohorts"][0]["guidance_source"] == "base"
        assert Path(config["worker"]["script"]).is_absolute()
        assert all(Path(value).is_absolute() for value in config["generator"]["artifacts"].values())
        policy = runner._development_policy(config["development"])
        assert policy.strategy is SamplingStrategy.FIXED_RECIPE
        assert policy.fixed_recipe.total == total
        assert policy.max_cases_per_agent_per_round == total
        assert not policy.cumulative_fixed_cases
        assert policy.fixed_recipe.base_perturbed <= policy.fixed_recipe.base
        assert policy.fixed_recipe.guided_perturbed <= policy.fixed_recipe.guided
        overrides = runner._round_overrides(config)
        if cold_start and guidance:
            first = overrides[1].development_sampling.fixed_recipe
            assert first.total == total
            assert first.guided == first.guided_perturbed == 0
        else:
            assert not overrides


def test_builder_cli_writes_relocatable_configs_and_refuses_overwrite(builder, runner, tmp_path):
    output = tmp_path / "configs"
    args = ["--config", str(SOURCE), "--output-dir", str(output), "--cases-per-round", "8"]
    assert builder.main(args) == 0
    assert len(tuple(output.glob("*.json"))) == 4
    for path in output.glob("*.json"):
        config, _ = runner._load_config(path)
        runner._validate_config(config, inspect_initial_candidate=False)
        assert Path(config["candidate"]["initial_bundle"]).is_absolute()
    original = (output / "base.json").read_bytes()
    with pytest.raises(FileExistsError):
        builder.main(args)
    assert (output / "base.json").read_bytes() == original


@pytest.mark.parametrize("flag", ("guidance_enabled", "perturbations_enabled"))
def test_hidden_terminal_capabilities_do_not_enable_disabled_development(builder, runner, flag):
    source, path = runner._load_config(SOURCE)
    config = builder.build_configs(source, path, cases_per_round=8)["base"]
    config["_config_path"] = str(path)
    development = deepcopy(config["development"])
    cohort = config["terminal"]["cohorts"][0]
    if flag == "guidance_enabled":
        config["terminal"]["cohorts"].append({
            "cohort_id": "hidden_guided", "guidance_source": "final_guided",
            "suite": {**dict.fromkeys(builder.COUNT_KEYS, 0), "guided": 1},
        })
    else:
        cohort["suite"]["base_perturbed"] = 1
    runner._validate_config(config, inspect_initial_candidate=False)
    assert runner._effective_generator_switches(config) == (
        flag == "guidance_enabled", flag == "perturbations_enabled",
    )
    assert runner._generator_switches(config) == (False, False)
    assert config["development"] == development

    # Hidden support must not authorize the same capability in either the
    # default Development recipe or a later round's override.
    forbidden_development = deepcopy(development)
    forbidden_development["fixed_recipe"]["base"] -= 1
    if flag == "guidance_enabled":
        forbidden_development["allowed_sources"].append("guided")
        forbidden_development["fixed_recipe"]["guided"] = 1
        message = "guided development Cases.*guidance is disabled"
    else:
        forbidden_development["fixed_recipe"]["base_perturbed"] = 1
        message = "perturbed development Cases.*perturbation is disabled"
    for override in (False, True):
        invalid = deepcopy(config)
        if override:
            invalid["round_overrides"] = {"2": {"development": forbidden_development}}
        else:
            invalid["development"] = forbidden_development
        with pytest.raises(ValueError, match=message):
            runner._validate_config(invalid, inspect_initial_candidate=False)


@pytest.mark.parametrize("agent_count", (1, 3))
def test_composition_uses_actual_config_without_models_or_gpu(builder, runner, monkeypatch, tmp_path, agent_count):
    source, path = runner._load_config(SOURCE)
    config = builder.build_configs(source, path, cases_per_round=8, without_initial_candidate=True)["base_g_p"]
    config["_config_path"] = str(path)
    config["round_count"] = 3
    config["agents"] = [
        {**source["agents"][0], "agent_id": f"agent-{index}"} for index in range(agent_count)
    ]
    config["execution"].update({
        "agent_count": agent_count, "agent_active_budget_seconds": 60,
        "agent_hard_wall_seconds": 90, "model": "configured-backbone", "reasoning_effort": "low",
    })
    runner._validate_config(config, inspect_initial_candidate=False)
    generator = SimpleNamespace(capabilities=GeneratorCapabilities(online=True, guidance=True, perturbations=True))
    recipe = GeneratorRecipe(guidance_enabled=True, perturbations_enabled=True)
    monkeypatch.setattr(runner, "_compose_numerical", lambda *args: (None, None, generator, recipe, None, {}))
    monkeypatch.setattr(runner, "SubprocessAgentExecutionService", lambda root, **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(runner, "SubprocessCandidateExecutionService", lambda root, **kwargs: SimpleNamespace(**kwargs))
    composition = runner._compose(
        config=config, config_path=path, run_root=tmp_path,
        codex_binary=Path("/unused/fake-codex"), auth_file=Path("/unused/fake-auth"),
        worker_gpus=tuple(str(index) for index in range(agent_count)), controller_gpu=str(agent_count),
    )
    assert len(composition.campaign.rounds) == 3
    assert len(composition.agents) == agent_count
    assert composition.campaign.initial_candidate is None
    assert composition.execution.hard_wall_seconds == 90
    assert composition.provenance["initial_candidate"] is None
    assert composition.provenance["agent_protocol"]["parallel_agents"] == agent_count
    assert composition.provenance["agent_protocol"]["time_budget_mode"] == "hard_wall"
    assert composition.provenance["agent_protocol"]["hard_wall_seconds_per_agent_per_round"] == 90
    for program in composition.agents.values():
        assert program.worker_settings["model"] == "configured-backbone"
        assert program.worker_settings["reasoning_effort"] == "low"
    # A historical active-budget field must not shorten the total-time budget.
    assert [plan.agent_budget_seconds for plan in composition.campaign.rounds] == [90, 90, 90]
    assert [plan.development_sampling.fixed_recipe.total for plan in composition.campaign.rounds] == [8, 8, 8]
    assert composition.campaign.rounds[0].development_sampling.fixed_recipe.guided == 0
    assert composition.campaign.rounds[1].development_sampling.fixed_recipe.guided == 2


@pytest.mark.parametrize("guidance,perturbations", ((False, False), (True, False), (False, True), (True, True)))
@pytest.mark.parametrize("hidden_capabilities", (False, True))
def test_numerical_generator_separates_hidden_support_from_development_permissions(
    builder, runner, monkeypatch, tmp_path, guidance, perturbations, hidden_capabilities,
):
    config, path = runner._load_config(SOURCE)
    arm = next(name for name, flags in builder.ARMS.items() if flags == (guidance, perturbations))
    config = builder.build_configs(config, path, cases_per_round=8)[arm]
    if hidden_capabilities:
        config["official"].update(guided=1, base_perturbed=1)
    prepared = SimpleNamespace(training_summary={})
    monkeypatch.setattr(runner, "NSProblem", lambda: SimpleNamespace(prepare_dataset=lambda _: prepared))
    monkeypatch.setattr(runner, "_required_regular_file", lambda path, label: path)
    hashes = {
        runner._config_path(path, raw): config["generator"]["artifact_sha256"][name]
        for name, raw in config["generator"]["artifacts"].items()
    }
    monkeypatch.setattr(runner, "_file_sha256", lambda path: hashes[path])
    bundle = SimpleNamespace(
        generator=object(), checkpoint=None,
        components=SimpleNamespace(flow_sampler=SimpleNamespace(pca=SimpleNamespace(rank=8))),
    )
    build_arguments = {}

    def build_generator(**kwargs):
        build_arguments.update(kwargs)
        return bundle

    monkeypatch.setattr(runner, "build_canonical_structured_ns_generator", build_generator)
    result = runner._compose_numerical(config, path, tmp_path)
    assert result[3].guidance_enabled is (guidance or hidden_capabilities)
    assert result[3].perturbations_enabled is (perturbations or hidden_capabilities)
    assert build_arguments["development_guidance_enabled"] is guidance
    assert build_arguments["development_perturbations_enabled"] is perturbations
    assert runner._generator_switches(config) == (guidance, perturbations)


def test_round_override_cannot_enable_replay(builder, runner):
    source, path = runner._load_config(SOURCE)
    config = builder.build_configs(source, path, cases_per_round=8, without_initial_candidate=True)["base_g"]
    config["_config_path"] = str(path)
    config["round_overrides"]["1"]["development"]["case_replay"] = True
    with pytest.raises(ValueError, match="replay"):
        runner._validate_config(config, inspect_initial_candidate=False)


@pytest.mark.parametrize("rounds", (1, 3, 5))
def test_piv_composition_checks_available_splits_instead_of_fixed_rounds(runner, monkeypatch, rounds):
    config, path = runner._load_config(ROOT / "configs/ns_canonical_piv_5round.json")
    config["round_count"] = rounds
    assembly = SimpleNamespace(
        prepared_dataset=SimpleNamespace(development_cases=tuple(range(50)), training_summary={}),
        terminal_final_test_cases=25, official_hidden_cases=25,
        checkpoint=SimpleNamespace(report={}), project_reconstruction={},
    )
    monkeypatch.setattr(runner, "assemble_completed_piv", lambda **kwargs: assembly)
    monkeypatch.setattr(runner, "_file_sha256", lambda path: "0" * 64)
    assert runner._compose_piv(config, path)[0] is assembly
    config["round_count"] = 6
    with pytest.raises(ValueError, match="verifier split"):
        runner._compose_piv(config, path)
    config["official"]["original"] = 1
    with pytest.raises(ValueError, match="development split"):
        runner._compose_piv(config, path)


def test_builder_rejects_guidance_feedback_smaller_than_backend_minimum(builder, runner):
    source, path = runner._load_config(SOURCE)
    with pytest.raises(ValueError, match="minimum_cases"):
        builder.build_configs(source, path, cases_per_round=8, official_cases=31)


def test_unlimited_source_requires_explicit_fixed_case_count(builder, runner):
    source, path = runner._load_config(SOURCE)
    with pytest.raises(ValueError, match="explicit --cases-per-round"):
        builder.build_configs(source, path)
    # A finite historical source still supplies its original ablation count.
    source["development"]["max_cases_per_agent_per_round"] = 12
    configs = builder.build_configs(source, path)
    assert all(config["development"]["max_cases_per_agent_per_round"] == 12
               for config in configs.values())
