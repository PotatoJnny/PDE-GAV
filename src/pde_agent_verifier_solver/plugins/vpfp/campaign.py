"""Compile the established long-time VPFP experiment into one product Run.

This file is intentionally boring to read.  It translates the researcher-facing
JSON settings into stages that prepare two Agent workspaces, run the Agents,
compare their final models, audit the macro contract, and publish the winner.
The numerical algorithms remain in ``legacy/vpfp``; the shared engine only sees
ordinary stages and therefore contains no VPFP-specific branch.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Mapping

from pde_agent_verifier_solver.engine.staged import (
    RunStage,
    StageCommand,
    StageResources,
    StagedRunPlan,
)

from .macro_matching import VPFPMacroMatcher
from .unified_reference_campaign import (
    compile_unified_macro_reference,
    preflight_unified_macro_reference,
    validate_unified_macro_reference_settings,
)


PRODUCT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_IMPLEMENTATION_ROOT = PRODUCT_ROOT / "legacy" / "vpfp"
LEGACY_PROTOCOL = "legacy_long_time_v2"
UNIFIED_MACRO_REFERENCE_PROTOCOL = "unified_macro_reference_v1"
PREPARATION_SCHEMA = "vpfp_long_time_robust_two_max_preparation_v2"
RELEASE_BANK_SCHEMA = "vpfp_long_time_answer_first_release_bank_v2"
LEGACY_MACRO_OUTPUT_CONTRACT = "vpfp_continuous_macro_output_v1"
MACRO_MATCH_CONTRACT = VPFPMacroMatcher.contract_id
MACRO_MATCH_VALIDATION = "formal_dual_score_v1"
MACRO_MATCH_ENVIRONMENT_VARIABLE = "VPFP_MACRO_MATCH_CONTRACT"


@dataclass(frozen=True)
class VPFPCampaignConfig:
    """Resolved paths plus the original customer settings."""

    source_path: Path
    protocol: str
    name: str
    release_bank_report: Path
    base_decoder_checkpoint: Path
    expanded_decoder_checkpoint: Path | None
    incumbent_model: Path | None
    run_root: Path
    implementation_root: Path
    virtual_environment: Path
    backend: str
    values: Mapping[str, Any]

    def section(self, name: str) -> Mapping[str, Any]:
        value = self.values.get(name, {})
        if not isinstance(value, Mapping):
            raise ValueError(f"config section {name!r} must be an object")
        return value


def load_vpfp_campaign_config(
    path: Path | str,
    *,
    dataset_override: Path | str | None = None,
    run_root_override: Path | str | None = None,
    backend_override: str | None = None,
) -> VPFPCampaignConfig:
    """Read one VPFP JSON recipe and resolve paths relative to that file."""

    source = Path(path).resolve()
    payload = _read_object(source)
    problem = str(payload.get("problem", "")).strip().lower()
    if problem not in {
        "vpfp",
        "vlasov_poisson_fokker_planck",
        "vlasov-poisson-fokker-planck",
    }:
        raise ValueError("this config is not a VPFP campaign")
    name = str(payload.get("name", "")).strip()
    if not name:
        raise ValueError("campaign config needs a nonempty name")
    protocol = str(payload.get("protocol", LEGACY_PROTOCOL)).strip()
    if protocol not in {LEGACY_PROTOCOL, UNIFIED_MACRO_REFERENCE_PROTOCOL}:
        raise ValueError(
            "protocol must be legacy_long_time_v2 or "
            "unified_macro_reference_v1"
        )


    dataset = _section(payload, "dataset")
    generator = _section(payload, "generator")
    task = _section(payload, "task")
    execution = _section(payload, "execution")
    release_value = (
        dataset_override
        or dataset.get("release_bank_report")
        or dataset.get("manifest")
    )
    if not release_value:
        raise ValueError("provide the VPFP release-bank report")
    base_value = generator.get("base_decoder_checkpoint")
    expanded_value = generator.get("expanded_decoder_checkpoint")
    incumbent_value = task.get("incumbent_model") or task.get("champion_solver")
    if not base_value:
        raise ValueError("generator needs base_decoder_checkpoint")
    if protocol == LEGACY_PROTOCOL and not expanded_value:
        raise ValueError(
            "legacy generator needs expanded_decoder_checkpoint"
        )
    if protocol == LEGACY_PROTOCOL and not incumbent_value:
        raise ValueError("task needs incumbent_model")

    run_value = run_root_override or execution.get("run_root")
    if not run_value:
        run_value = PRODUCT_ROOT / "runs" / _slug(name)
    implementation_value = payload.get("implementation_root")
    implementation_root = (
        _resolve_path(implementation_value, source.parent)
        if implementation_value
        else (
            PRODUCT_ROOT if protocol == UNIFIED_MACRO_REFERENCE_PROTOCOL
            else DEFAULT_IMPLEMENTATION_ROOT
        )
    )
    environment_value = (
        execution.get("virtual_environment")
        or os.environ.get("VIRTUAL_ENV")
        or Path.home() / "environment" / "venv"
    )
    backend = str(backend_override or execution.get("backend", "slurm"))

    config = VPFPCampaignConfig(
        source_path=source,
        protocol=protocol,
        name=name,
        release_bank_report=_resolve_path(release_value, source.parent),
        base_decoder_checkpoint=_resolve_path(base_value, source.parent),
        expanded_decoder_checkpoint=(
            _resolve_path(expanded_value, source.parent)
            if expanded_value
            else None
        ),
        incumbent_model=(
            _resolve_path(incumbent_value, source.parent)
            if incumbent_value
            else None
        ),
        run_root=_resolve_path(run_value, source.parent),
        implementation_root=implementation_root.resolve(),
        virtual_environment=_resolve_path(environment_value, source.parent),
        backend=backend,
        values=payload,
    )
    _validate_settings(config)
    return config


def preflight_vpfp_campaign(
    config: VPFPCampaignConfig,
) -> Mapping[str, object]:
    """Check every expensive-run input before any Slurm job is submitted."""

    if config.protocol == UNIFIED_MACRO_REFERENCE_PROTOCOL:
        return preflight_unified_macro_reference(
            config,
            product_root=PRODUCT_ROOT,
            release_schema=RELEASE_BANK_SCHEMA,
            macro_contract=MACRO_MATCH_CONTRACT,
        )
    assert config.expanded_decoder_checkpoint is not None
    assert config.incumbent_model is not None
    if config.backend != "slurm":
        raise ValueError(
            "the historical VPFP online Generator and Evaluator require the "
            "Slurm backend"
        )
    storage_root = _storage_root(config)
    if not storage_root.is_dir():
        raise FileNotFoundError(f"VPFP run storage does not exist: {storage_root}")
    allowed_roots = (Path("/scratch").resolve(), storage_root)
    if not any(
        config.run_root.resolve().is_relative_to(root)
        for root in allowed_roots
    ):
        raise ValueError("VPFP run_root must be below execution.storage_root")
    release = _read_object(config.release_bank_report)
    if (
        release.get("schema") != RELEASE_BANK_SCHEMA
        or release.get("passed") is not True
    ):
        raise ValueError("VPFP release bank is not a passing v2 release bank")
    development = _path_list(release, "development_case_dirs")
    released = _path_list(release, "released_completed_case_dirs")
    if not development:
        raise ValueError("VPFP release bank has no development cases")
    if len(released) != 32:
        raise ValueError(
            "the reproduced VPFP protocol requires exactly 32 released cases"
        )
    missing_cases = [str(path) for path in (*development, *released) if not path.is_dir()]
    if missing_cases:
        preview = ", ".join(missing_cases[:4])
        raise FileNotFoundError(f"VPFP release-bank cases are missing: {preview}")

    required_inputs = (
        config.base_decoder_checkpoint,
        config.expanded_decoder_checkpoint,
        config.incumbent_model,
        config.virtual_environment / "bin" / "python",
        *_optional_round_inputs(config),
    )
    missing_inputs = [str(path) for path in required_inputs if not path.is_file()]
    if missing_inputs:
        raise FileNotFoundError(
            "VPFP campaign inputs are missing: " + ", ".join(missing_inputs)
        )
    python = config.virtual_environment / "bin" / "python"
    if not os.access(python, os.X_OK):
        raise PermissionError(
            f"VPFP virtual-environment Python is not executable: {python}"
        )

    missing_code = [
        str(path)
        for path in _required_implementation_files(config)
        if not path.is_file()
    ]
    if missing_code:
        raise FileNotFoundError(
            "the bundled VPFP implementation is incomplete: "
            + ", ".join(missing_code)
        )
    return {
        "dataset": str(config.release_bank_report),
        "dataset_format": RELEASE_BANK_SCHEMA,
        "development_cases": len(development),
        "released_completed_cases": len(released),
        "implementation_root": str(config.implementation_root),
        "virtual_environment": str(config.virtual_environment),
        "run_root": str(config.run_root),
        "backend": config.backend,
        "task": "observable_model",
        "agents": 2,
        "legacy_solver_output_contract": LEGACY_MACRO_OUTPUT_CONTRACT,
        "macro_match_contract": MACRO_MATCH_CONTRACT,
        "macro_match_validation": MACRO_MATCH_VALIDATION,
        "macro_match_used_for_winner_selection": False,
    }


class VPFPCampaignCompiler:
    """Compile historical VPFP staged lifecycles for explicit legacy runs."""

    def compile(
        self,
        config: VPFPCampaignConfig,
        *,
        legacy_staged_intent: bool = False,
    ) -> StagedRunPlan:
        if legacy_staged_intent is not True:
            raise PermissionError(
                "VPFPCampaignCompiler is a history-only staged compiler; pass "
                "legacy_staged_intent=True for an explicit legacy/control-arm run"
            )
        if config.protocol == UNIFIED_MACRO_REFERENCE_PROTOCOL:
            return compile_unified_macro_reference(
                config,
                product_root=PRODUCT_ROOT,
                macro_contract=MACRO_MATCH_CONTRACT,
                command_environment=_command_environment,
                resources=_resources,
            )
        assert config.expanded_decoder_checkpoint is not None
        assert config.incumbent_model is not None
        paths = _RunPaths(config.run_root)
        campaign = config.section("campaign")
        generator = config.section("generator")
        agent = config.section("agent")
        evaluator = config.section("evaluator")
        python = str(config.virtual_environment / "bin" / "python")
        environment = _command_environment(config)
        formal_environment = {
            **environment,
            MACRO_MATCH_ENVIRONMENT_VARIABLE: MACRO_MATCH_CONTRACT,
        }
        agent_budget = _integer(agent, "budget_seconds", 28_800)
        active_wall = _integer(agent, "active_wall_seconds", max(3600, agent_budget))

        prepare_arguments = [
            python,
            str(_script(config, "prepare_vpfp_long_time_robust_agent_campaign_v2.py")),
            "--release-bank-report",
            str(config.release_bank_report),
            "--out",
            str(paths.campaign),
            "--base-decoder-checkpoint",
            str(config.base_decoder_checkpoint),
            "--expanded-decoder-checkpoint",
            str(config.expanded_decoder_checkpoint),
            "--champion-solver",
            str(config.incumbent_model),
            "--cluster-notes",
            str(config.implementation_root / "docs" / "slurm_cluster_notes_for_codex.md"),
            "--round-index",
            str(_integer(campaign, "round_index", 0)),
            "--seed-base",
            str(_integer(campaign, "seed_base", 3_600_000)),
            "--active-wall-seconds",
            str(active_wall),
            "--development-case-credits",
            str(_integer(agent, "development_case_credits", 4096)),
            "--generation-batch-size",
            str(_integer(generator, "generation_batch_size", 16)),
            "--guidance-pool-size",
            str(_integer(generator, "guidance_pool_size", 4)),
            "--user-authorized-external-dispatch",
        ]
        _add_optional_path(
            prepare_arguments,
            "--agent-a-prior-solver",
            _agent_path(config, "prior_models", "agent_a"),
        )
        _add_optional_path(
            prepare_arguments,
            "--agent-b-prior-solver",
            _agent_path(config, "prior_models", "agent_b"),
        )
        _add_optional_path(
            prepare_arguments,
            "--agent-a-prior-feedback",
            _agent_path(config, "prior_feedback", "agent_a"),
        )
        _add_optional_path(
            prepare_arguments,
            "--agent-b-prior-feedback",
            _agent_path(config, "prior_feedback", "agent_b"),
        )
        prompt_file = _configured_path(config, agent.get("prompt_file"))
        _add_optional_path(
            prepare_arguments,
            "--agent-prompt-file",
            prompt_file,
        )

        worker_common = (
            "--campaign-root",
            str(paths.campaign),
            "--agent-timeout-seconds",
            str(agent_budget),
            "--finalist-timeout-seconds",
            str(_integer(agent, "finalist_timeout_seconds", 7200)),
        )
        mock_flag = (
            ("--mock-agent",)
            if str(agent.get("backend", "codex")) == "codex_mock"
            else ()
        )
        worker_script = str(
            _script(config, "run_vpfp_long_time_robust_agent_worker_v2.py")
        )
        agent_a = StageCommand(
            (python, worker_script, *worker_common, "--agent-index", "0", *mock_flag),
            "Run VPFP Agent A",
            environment,
        )
        agent_b = StageCommand(
            (python, worker_script, *worker_common, "--agent-index", "1", *mock_flag),
            "Run VPFP Agent B",
            environment,
        )

        formal_arguments = [
            python,
            str(_script(config, "run_vpfp_long_time_robust_final_evaluation_v2.py")),
            "--campaign-root",
            str(paths.campaign),
            "--out",
            str(paths.formal),
            "--broad-cases",
            str(_integer(evaluator, "broad_cases", 12)),
            "--frontier-cases",
            str(_integer(evaluator, "frontier_cases", 4)),
        ]
        if evaluator.get("seed_base") is not None:
            formal_arguments.extend(
                ("--seed-base", str(_integer(evaluator, "seed_base", 1)))
            )
        if _boolean(evaluator, "recover_incomplete_workers", False):
            formal_arguments.append("--recover-evaluator-failed-workers")
        macro_contract_audit = StageCommand(
            (
                python,
                "-m",
                "pde_agent_verifier_solver.plugins.vpfp.macro_contract_audit",
                "--formal-root",
                str(paths.formal),
                "--out",
                str(paths.macro_contract_audit),
            ),
            "Validate VPFP unified macro-match evidence",
            environment,
        )
        publish = StageCommand(
            (
                python,
                "-m",
                "pde_agent_verifier_solver.plugins.vpfp.publish",
                "--run-root",
                str(config.run_root),
                "--campaign-root",
                str(paths.campaign),
                "--formal-root",
                str(paths.formal),
                "--macro-contract-audit",
                str(paths.macro_contract_audit),
            ),
            "Publish the selected VPFP model",
            environment,
        )

        stages = (
            RunStage(
                "prepare_campaign",
                "Prepare the two VPFP Agent workspaces",
                (
                    StageCommand(
                        tuple(prepare_arguments),
                        "Prepare VPFP campaign",
                        environment,
                    ),
                ),
                resources=_resources(config, "prepare_campaign", gpu=False),
                expected_outputs=(paths.preparation_report,),
            ),
            RunStage(
                "agent_a",
                "Run VPFP Agent A with the online Generator and Evaluator",
                (agent_a,),
                depends_on=("prepare_campaign",),
                resources=_resources(config, "agent_a", gpu=True),
                expected_outputs=(paths.worker_report("vpfp_max_a"),),
            ),
            RunStage(
                "agent_b",
                "Run VPFP Agent B with the online Generator and Evaluator",
                (agent_b,),
                depends_on=("prepare_campaign",),
                resources=_resources(config, "agent_b", gpu=True),
                expected_outputs=(paths.worker_report("vpfp_max_b"),),
            ),
            RunStage(
                "final_evaluation",
                "Compare both Agent models on fresh VPFP cases",
                (
                    StageCommand(
                        tuple(formal_arguments),
                        "Run independent VPFP evaluation",
                        formal_environment,
                    ),
                ),
                depends_on=("agent_a", "agent_b"),
                resources=_resources(config, "final_evaluation", gpu=True),
                expected_outputs=(paths.formal_summary,),
            ),
            RunStage(
                "validate_macro_contract",
                "Audit unified VPFP macro matches without changing the winner",
                (macro_contract_audit,),
                depends_on=("final_evaluation",),
                resources=_resources(config, "validate_macro_contract", gpu=False),
                expected_outputs=(paths.macro_contract_audit,),
            ),
            RunStage(
                "publish_result",
                "Build the final VPFP result",
                (publish,),
                depends_on=("validate_macro_contract",),
                resources=_resources(config, "publish_result", gpu=False),
                expected_outputs=(paths.result_json, paths.final_model),
            ),
        )
        execution = config.section("execution")
        modules = execution.get(
            "modules",
            ["StdEnv/2023", "python/3.12", "scipy-stack"],
        )
        if not isinstance(modules, list) or not all(
            isinstance(item, str) for item in modules
        ):
            raise ValueError("execution.modules must be a list of module names")
        return StagedRunPlan(
            name=config.name,
            problem="vpfp",
            task=str(config.section("task").get("kind", "observable_model")),
            run_root=config.run_root,
            workdir=config.implementation_root,
            product_root=PRODUCT_ROOT,
            implementation_root=config.implementation_root,
            stages=stages,
            modules=tuple(modules),
            account=str(execution.get("account", os.environ.get("PDE_GAV_SLURM_ACCOUNT", ""))),
            qos=str(execution.get("qos", "normal")),
            result_path=paths.result_json,
        )


@dataclass(frozen=True)
class _RunPaths:
    root: Path

    @property
    def campaign(self) -> Path:
        return self.root / "campaign" / "vpfp_long_time"

    @property
    def preparation_report(self) -> Path:
        return self.campaign / "campaign_preparation.json"

    def worker_report(self, agent_id: str) -> Path:
        return (
            self.campaign
            / "controller_private"
            / agent_id
            / "agent_run"
            / "worker_report.json"
        )

    @property
    def formal(self) -> Path:
        return self.root / "artifacts" / "formal_evaluation"

    @property
    def formal_summary(self) -> Path:
        return self.formal / "formal_summary.json"

    @property
    def macro_contract_audit(self) -> Path:
        return self.formal / "macro_contract_validation.json"

    @property
    def result_json(self) -> Path:
        return self.root / "result" / "result.json"

    @property
    def final_model(self) -> Path:
        return self.root / "result" / "model" / "submission.py"


def _validate_settings(config: VPFPCampaignConfig) -> None:
    if config.backend not in {"local", "slurm"}:
        raise ValueError("execution.backend must be 'local' or 'slurm'")
    if config.protocol == UNIFIED_MACRO_REFERENCE_PROTOCOL:
        validate_unified_macro_reference_settings(
            config,
            macro_contract=MACRO_MATCH_CONTRACT,
        )
        return
    task = config.section("task")
    if str(task.get("kind", "observable_model")) not in {
        "observable",
        "observable_model",
    }:
        raise ValueError(
            "the historical VPFP campaign produces an observable_model; "
            "use the VPFP Task SDK for reduced_dynamics or statistics"
        )
    if str(task.get("macro_match_contract", MACRO_MATCH_CONTRACT)) != (
        MACRO_MATCH_CONTRACT
    ):
        raise ValueError(
            f"task.macro_match_contract must be {MACRO_MATCH_CONTRACT!r}"
        )
    agent = config.section("agent")
    agent_backend = str(agent.get("backend", "codex"))
    if agent_backend not in {"codex", "codex_mock"}:
        raise ValueError("agent.backend must be 'codex' or 'codex_mock'")
    if str(agent.get("model", "gpt-5.6-sol")) != "gpt-5.6-sol":
        raise ValueError("the reproduced VPFP v2 protocol uses gpt-5.6-sol")
    if str(agent.get("reasoning_effort", "max")) != "max":
        raise ValueError("the reproduced VPFP v2 protocol uses Max reasoning")
    budget = _integer(agent, "budget_seconds", 28_800)
    minimum_budget = 60 if agent_backend == "codex_mock" else 3600
    if budget < minimum_budget:
        raise ValueError(
            f"agent.budget_seconds must be at least {minimum_budget}"
        )
    active_wall = _integer(agent, "active_wall_seconds", max(3600, budget))
    if active_wall < 3600:
        raise ValueError("agent.active_wall_seconds must be at least 3600")
    if _integer(agent, "finalist_timeout_seconds", 7200) < 300:
        raise ValueError("agent.finalist_timeout_seconds must be at least 300")
    if _integer(agent, "development_case_credits", 4096) < 64:
        raise ValueError("agent.development_case_credits must be at least 64")

    generator = config.section("generator")
    if _integer(generator, "generation_batch_size", 16) != 16:
        raise ValueError("the VPFP online Generator batch size must be 16")
    if _integer(generator, "guidance_pool_size", 4) < 2:
        raise ValueError("generator.guidance_pool_size must be at least 2")
    campaign = config.section("campaign")
    if _integer(campaign, "round_index", 0) < 0:
        raise ValueError("campaign.round_index cannot be negative")
    if _integer(campaign, "seed_base", 3_600_000) % 4:
        raise ValueError("campaign.seed_base must be a multiple of four")
    evaluator = config.section("evaluator")
    if str(
        evaluator.get("macro_match_validation", MACRO_MATCH_VALIDATION)
    ) != MACRO_MATCH_VALIDATION:
        raise ValueError(
            "evaluator.macro_match_validation must be "
            f"{MACRO_MATCH_VALIDATION!r}"
        )
    if _integer(evaluator, "broad_cases", 12) < 4:
        raise ValueError("evaluator.broad_cases must be at least 4")
    if _integer(evaluator, "frontier_cases", 4) < 2:
        raise ValueError("evaluator.frontier_cases must be at least 2")


def _required_implementation_files(
    config: VPFPCampaignConfig,
) -> tuple[Path, ...]:
    script_names = (
        "prepare_vpfp_long_time_robust_agent_campaign_v2.py",
        "run_vpfp_long_time_robust_agent_worker_v2.py",
        "run_vpfp_long_time_robust_final_evaluation_v2.py",
        "run_vpfp_answer_first_online_generator_service.py",
        "run_vpfp_continuous_online_generator_service.py",
        "run_vpfp_development_evaluator_service_v2.py",
        "run_vpfp_budgeted_solver_call_v2.py",
        "run_vpfp_instruction_only_solver.sh",
        "run_codex_with_ephemeral_auth.py",
        "vpfp_instruction_only_codex.config.toml",
        "vpfp_solver_bundle_client.py",
        "vpfp_development_evaluation_client.py",
        "vpfp_instruction_only_solver_preflight.sh",
    )
    return tuple(_script(config, name) for name in script_names) + (
        PRODUCT_ROOT
        / "src"
        / "pde_agent_verifier_solver"
        / "plugins"
        / "vpfp"
        / "legacy_macro_bridge.py",
        PRODUCT_ROOT / "src" / "pde_agent_verifier_solver" / "plugins"
        / "vpfp" / "macro_contract_audit.py",
        config.implementation_root / "docs" / "slurm_cluster_notes_for_codex.md",
        config.implementation_root / "src" / "agent_verifier" / "vpfp" / "__init__.py",
        config.implementation_root / "src" / "agent_verifier" / "storage_lifecycle.py",
        config.implementation_root / "src" / "agent_verifier" / "vpfp" / "runtime_paths.py",
        config.implementation_root / "src" / "agent_verifier" / "core" / "types.py",
    )


def _optional_round_inputs(config: VPFPCampaignConfig) -> tuple[Path, ...]:
    paths = []
    for group in ("prior_models", "prior_feedback"):
        for agent in ("agent_a", "agent_b"):
            value = _agent_path(config, group, agent)
            if value is not None:
                paths.append(value)
    prompt = _configured_path(config, config.section("agent").get("prompt_file"))
    if prompt is not None:
        paths.append(prompt)
    return tuple(paths)


def _agent_path(
    config: VPFPCampaignConfig,
    group: str,
    agent: str,
) -> Path | None:
    raw_group = config.section("agent").get(group, {})
    if not isinstance(raw_group, Mapping):
        raise ValueError(f"agent.{group} must be an object")
    return _configured_path(config, raw_group.get(agent))


def _configured_path(
    config: VPFPCampaignConfig,
    value: object,
) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    return _resolve_path(value, config.source_path.parent)


def _add_optional_path(
    arguments: list[str],
    flag: str,
    value: Path | None,
) -> None:
    if value is not None:
        arguments.extend((flag, str(value)))


def _storage_root(config: VPFPCampaignConfig) -> Path:
    value = config.section("execution").get("storage_root", "/scratch")
    root = _resolve_path(value, config.source_path.parent)
    if root == Path(root.anchor):
        raise ValueError("execution.storage_root must be a specific directory")
    return root


def _command_environment(config: VPFPCampaignConfig) -> Mapping[str, str]:
    venv = str(config.virtual_environment)
    current_path = os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")
    return {
        "VIRTUAL_ENV": venv,
        "VPFP_RUN_STORAGE_ROOT": str(_storage_root(config)),
        "PATH": str(config.virtual_environment / "bin") + os.pathsep + current_path,
        "PYTHONUNBUFFERED": "1",
    }


def _resources(
    config: VPFPCampaignConfig,
    stage: str,
    *,
    gpu: bool,
) -> StageResources:
    defaults = {
        "macro_agent_round": (8, "32G", "08:00:00"),
        "prepare_campaign": (4, "24G", "01:00:00"),
        "agent_a": (16, "96G", "12:00:00"),
        "agent_b": (16, "96G", "12:00:00"),
        "final_evaluation": (16, "96G", "18:00:00"),
        "publish_result": (4, "16G", "01:00:00"),
        "validate_macro_contract": (2, "8G", "00:30:00"),
    }
    cpus, memory, time_limit = defaults[stage]
    execution = config.section("execution")
    resources = execution.get("resources", {})
    if not isinstance(resources, Mapping):
        raise ValueError("execution.resources must be an object")
    common = resources.get("default", {})
    override = resources.get(stage, {})
    if not isinstance(common, Mapping) or not isinstance(override, Mapping):
        raise ValueError("each execution resource entry must be an object")
    merged = {**common, **override}
    partition_key = "gpu_partition" if gpu else "preparation_partition"
    partition_default = (
        "gpubase_l40s_b3"
        if gpu or stage == "macro_agent_round"
        else "gpubase_l40s_b1"
    )
    return StageResources(
        cpus=int(merged.get("cpus", cpus)),
        memory=str(merged.get("memory", memory)),
        time_limit=str(merged.get("time", time_limit)),
        partition=str(
            merged.get(
                "partition",
                execution.get(partition_key, partition_default),
            )
        ),
        gres=str(
            merged.get(
                "gres",
                execution.get("gpu_gres", "gpu:l40s:1") if gpu else "",
            )
        ),
    )


def _path_list(payload: Mapping[str, Any], name: str) -> tuple[Path, ...]:
    values = payload.get(name)
    if not isinstance(values, list) or not all(
        isinstance(value, str) for value in values
    ):
        raise ValueError(f"VPFP release bank needs a string list named {name!r}")
    return tuple(Path(value).resolve() for value in values)


def _script(config: VPFPCampaignConfig, name: str) -> Path:
    return config.implementation_root / "scripts" / name


def _section(values: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = values.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"config section {name!r} must be an object")
    return value


def _integer(values: Mapping[str, Any], name: str, default: int) -> int:
    value = values.get(name, default)
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _boolean(values: Mapping[str, Any], name: str, default: bool) -> bool:
    value = values.get(name, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValueError(f"{name} must be true or false")


def _resolve_path(value: object, relative_to: Path) -> Path:
    expanded = os.path.expandvars(str(value))
    if "$" in expanded:
        raise ValueError(f"unresolved environment variable in path: {value}")
    path = Path(expanded).expanduser()
    return (relative_to / path).resolve() if not path.is_absolute() else path.resolve()


def _read_object(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _slug(value: str) -> str:
    result = "".join(
        character.lower() if character.isalnum() else "_"
        for character in value
    ).strip("_")
    return result or "vpfp_run"
