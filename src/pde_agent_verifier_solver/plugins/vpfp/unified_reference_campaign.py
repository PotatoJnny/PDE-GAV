"""Legacy staged adapter for the reference-only VPFP Macro loop.

VPFP owns the release bank, query and macro matcher.  This module only resolves
those inputs, performs cheap launch checks, and compiles one explicit
history/control-arm stage. It is not the canonical ``CampaignController``
pipeline and contains no additional Agent ranking, promotion, or iteration
policy.
"""

from __future__ import annotations

import json
from math import isfinite
import os
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Protocol

from pde_agent_verifier_solver.engine.staged import (
    RunStage,
    StageCommand,
    StageResources,
    StagedRunPlan,
)


_PATH_FIELDS = (
    ("task", "cold_start_model"),
    ("agent", "launcher"),
    ("agent", "codex_binary"),
    ("agent", "auth_file"),
)
_LOOP_FIELDS = (
    ("round_count", 1, 1),
    ("development_case_count", 4, 1),
    ("selection_case_count", 4, 1),
    ("promotion_case_count", 4, 1),
    ("terminal_case_count", 4, 1),
    ("seed_base", 8_240_000, 0),
    ("seed_stride", 10_000, 4),
    ("development_pool_multiplier", 1, 1),
)
_NUMBER_FIELDS = (
    ("promotion_confidence_z", 0.0, "nonnegative"),
    ("promotion_min_mean_improvement", 0.0, "nonnegative"),
    ("promotion_min_win_rate", 0.0, "unit"),
    ("terminal_confidence_z", 0.0, "nonnegative"),
    ("terminal_min_win_rate", 0.0, "unit"),
)
_BOOL_FIELDS = (
    "promotion_require_cvar_no_regression",
    "terminal_require_cvar_no_regression",
)
_REASONING_EFFORTS = {
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
    "ultra",
}


class _CampaignConfig(Protocol):
    source_path: Path
    name: str
    release_bank_report: Path
    base_decoder_checkpoint: Path
    run_root: Path
    virtual_environment: Path
    backend: str

    def section(self, name: str) -> Mapping[str, Any]: ...


def validate_unified_macro_reference_settings(
    config: _CampaignConfig,
    *,
    macro_contract: str,
) -> None:
    """Validate the small set of settings consumed by the reference run."""

    if config.backend != "slurm":
        raise ValueError(
            "unified_macro_reference_v1 currently requires the Slurm backend"
        )
    task = config.section("task")
    if str(task.get("kind", "observable_model")) not in {
        "observable",
        "observable_model",
    }:
        raise ValueError("unified VPFP macro matching produces an observable_model")
    if str(task.get("macro_match_contract", macro_contract)) != macro_contract:
        raise ValueError(
            f"task.macro_match_contract must be {macro_contract!r}"
        )

    agent = config.section("agent")
    if str(agent.get("backend", "codex")) != "codex":
        raise ValueError("unified macro reference runs require agent.backend='codex'")
    model = str(agent.get("model", "")).strip()
    if not model or re.fullmatch(r"[A-Za-z0-9._/-]+", model) is None:
        raise ValueError("agent.model is empty or contains unsupported characters")
    if str(agent.get("reasoning_effort", "")).strip() not in _REASONING_EFFORTS:
        raise ValueError("agent.reasoning_effort is unsupported")
    _integer(agent, "timeout_seconds", 21_600, minimum=60)
    credits = _integer(
        agent,
        "development_case_credits",
        16,
        minimum=1,
    )
    paths = _configured_paths(config)
    if len(paths) != len(_PATH_FIELDS):
        raise AssertionError("unreachable")

    loop = _loop_settings(config)
    if credits < loop["development_case_count"]:
        raise ValueError(
            "agent.development_case_credits must cover at least one "
            "development evaluation"
        )
    evidence = _evidence_settings(config)
    if (
        loop["terminal_case_count"] == 1
        and evidence["terminal_confidence_z"] > 0.0
    ):
        raise ValueError("positive-z terminal evidence requires at least two cases")
    if (
        loop["promotion_case_count"] == 1
        and evidence["promotion_confidence_z"] > 0.0
    ):
        raise ValueError("positive-z promotion evidence requires at least two cases")


def preflight_unified_macro_reference(
    config: _CampaignConfig,
    *,
    product_root: Path,
    release_schema: str,
    macro_contract: str,
) -> Mapping[str, object]:
    """Check launch inputs cheaply; byte hashing remains on the compute node."""

    storage_root = _storage_root(config)
    if not storage_root.is_dir():
        raise FileNotFoundError(f"VPFP run storage does not exist: {storage_root}")
    if not any(
        config.run_root.resolve().is_relative_to(root)
        for root in (Path("/scratch").resolve(), storage_root)
    ):
        raise ValueError("VPFP run_root must be below execution.storage_root")

    report = _regular_file(
        config.release_bank_report,
        "VPFP release-bank report",
    )
    release = _read_object(report)
    if (
        release.get("schema") != release_schema
        or release.get("passed") is not True
    ):
        raise ValueError("VPFP release bank is not a passing v2 release bank")
    answer_first = release.get("answer_first_contract")
    required_contract = (
        "continuous_hidden_answer_is_primary",
        "public_source_is_exact_jet_residual",
        "coarse_equation_first_solve_is_shape_seed_only",
        "grid_arrays_are_numerical_access_views",
    )
    if not isinstance(answer_first, Mapping) or not all(
        answer_first.get(name) is True for name in required_contract
    ):
        raise ValueError("VPFP release bank lacks the answer-first contract")
    case_dirs = _release_case_dirs(report, release)

    loop = _loop_settings(config)
    required_cases = (
        loop["round_count"]
        * (
            loop["development_case_count"]
            * loop["development_pool_multiplier"]
            + loop["selection_case_count"]
            + loop["promotion_case_count"]
        )
        + loop["terminal_case_count"]
    )
    if required_cases > len(case_dirs):
        raise ValueError(
            f"unified macro loop needs {required_cases} disjoint cases, "
            f"but the release bank has {len(case_dirs)}"
        )

    paths = _configured_paths(config)
    regular_inputs = {
        config.base_decoder_checkpoint: "VPFP decoder checkpoint",
        paths["cold_start_model"]: "VPFP cold-start candidate",
        paths["launcher"]: "reference Agent launcher",
        paths["auth_file"]: "Codex auth file",
        product_root
        / "src"
        / "pde_agent_verifier_solver"
        / "plugins"
        / "vpfp"
        / "fresh_u0.py": "VPFP fresh-U0 runner",
    }
    for path, label in regular_inputs.items():
        _regular_file(path, label)
    for path, label in (
        (config.virtual_environment / "bin" / "python", "virtual-environment Python"),
        (paths["codex_binary"], "Codex binary"),
        (paths["launcher"], "reference Agent launcher"),
    ):
        if not path.is_file() or not os.access(path, os.X_OK):
            raise PermissionError(f"{label} is not executable: {path}")

    return {
        "dataset": str(report),
        "dataset_format": release_schema,
        "release_bank_cases": len(case_dirs),
        "required_disjoint_cases": required_cases,
        "generator_checkpoint": str(config.base_decoder_checkpoint),
        "run_root": str(config.run_root),
        "backend": config.backend,
        "task": "observable_model",
        "agents": 1,
        "fresh_agent": True,
        "cold_start": str(paths["cold_start_model"]),
        "macro_match_contract": macro_contract,
        "macro_match_used_for_selection": True,
        "reference_only": True,
        "formal_claims_allowed": False,
    }


def compile_unified_macro_reference(
    config: _CampaignConfig,
    *,
    product_root: Path,
    macro_contract: str,
    command_environment: Callable[[_CampaignConfig], Mapping[str, str]],
    resources: Callable[..., StageResources],
) -> StagedRunPlan:
    """Compile one history/control-arm stage around the legacy Macro runner."""

    agent = config.section("agent")
    execution = config.section("execution")
    task = config.section("task")
    paths = _configured_paths(config)
    loop = _loop_settings(config)
    evidence = _evidence_settings(config)
    result_root = config.run_root / "result"
    arguments = [
        str(config.virtual_environment / "bin" / "python"),
        "-m",
        "pde_agent_verifier_solver.plugins.vpfp.fresh_u0",
        "--legacy-macro-reference",
        "--release-bank-report",
        str(config.release_bank_report),
        "--decoder-checkpoint",
        str(config.base_decoder_checkpoint),
        "--run-root",
        str(config.run_root),
        "--cold-start-candidate",
        str(paths["cold_start_model"]),
        "--agent-launcher",
        str(paths["launcher"]),
        "--codex-binary",
        str(paths["codex_binary"]),
        "--auth-file",
        str(paths["auth_file"]),
        "--model",
        str(agent["model"]),
        "--reasoning-effort",
        str(agent["reasoning_effort"]),
        "--agent-timeout-seconds",
        str(_integer(agent, "timeout_seconds", 21_600, minimum=60)),
        "--development-case-credits",
        str(
            _integer(
                agent,
                "development_case_credits",
                16,
                minimum=1,
            )
        ),
    ]
    _append_values(arguments, loop)
    _append_evidence(arguments, evidence)

    modules = execution.get(
        "modules",
        ["StdEnv/2023", "python/3.11", "scipy-stack/2024b"],
    )
    if not isinstance(modules, list) or not all(
        isinstance(item, str) for item in modules
    ):
        raise ValueError("execution.modules must be a list of module names")
    stage = RunStage(
        "macro_agent_round",
        "Improve and validate a fresh VPFP macroscopic model",
        (
            StageCommand(
                tuple(arguments),
                "Run the unified macro-only Agent protocol",
                {
                    **command_environment(config),
                    "PDE_MACRO_MATCH_CONTRACT": macro_contract,
                    "PDE_MACRO_REFERENCE_ONLY": "1",
                },
            ),
        ),
        resources=resources(config, "macro_agent_round", gpu=False),
        expected_outputs=(
            result_root / "result.json",
            result_root / "model" / "submission.py",
        ),
    )
    return StagedRunPlan(
        name=config.name,
        problem="vpfp",
        task=str(task.get("kind", "observable_model")),
        run_root=config.run_root,
        workdir=product_root,
        product_root=product_root,
        implementation_root=product_root,
        stages=(stage,),
        modules=tuple(modules),
        account=str(execution.get("account", os.environ.get("PDE_GAV_SLURM_ACCOUNT", ""))),
        qos=str(execution.get("qos", "normal")),
        result_path=result_root / "result.json",
    )


def _loop_settings(config: _CampaignConfig) -> dict[str, int]:
    campaign = config.section("campaign")
    return {
        name: _integer(campaign, name, default, minimum=minimum)
        for name, default, minimum in _LOOP_FIELDS
    }


def _evidence_settings(config: _CampaignConfig) -> dict[str, float | bool | None]:
    evaluator = config.section("evaluator")
    result: dict[str, float | bool | None] = {}
    for name, default, constraint in _NUMBER_FIELDS:
        value = _number(evaluator, name, default)
        if constraint == "nonnegative" and value < 0.0:
            raise ValueError(f"{name} must be nonnegative")
        if constraint == "unit" and not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1]")
        result[name] = value
    for name in _BOOL_FIELDS:
        value = evaluator.get(name, False)
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be true or false")
        result[name] = value
    budget = evaluator.get("candidate_compute_budget")
    if budget is not None and (
        isinstance(budget, bool)
        or not isinstance(budget, (int, float))
        or not isfinite(float(budget))
        or float(budget) <= 0.0
    ):
        raise ValueError(
            "evaluator.candidate_compute_budget must be finite and positive"
        )
    result["candidate_compute_budget"] = (
        None if budget is None else float(budget)
    )
    return result


def _append_values(arguments: list[str], values: Mapping[str, object]) -> None:
    for name, _default, _minimum in _LOOP_FIELDS:
        arguments.extend((f"--{name.replace('_', '-')}", str(values[name])))


def _append_evidence(
    arguments: list[str],
    values: Mapping[str, float | bool | None],
) -> None:
    for name, _default, _constraint in _NUMBER_FIELDS:
        arguments.extend((f"--{name.replace('_', '-')}", str(values[name])))
    for name in _BOOL_FIELDS:
        flag = name.replace("_", "-")
        arguments.append(f"--{flag}" if values[name] else f"--no-{flag}")
    budget = values["candidate_compute_budget"]
    if budget is not None:
        arguments.extend(("--candidate-compute-budget", str(budget)))


def _configured_paths(config: _CampaignConfig) -> dict[str, Path]:
    return {
        key: _configured_path(config, section_name, key)
        for section_name, key in _PATH_FIELDS
    }


def _configured_path(
    config: _CampaignConfig,
    section_name: str,
    key: str,
) -> Path:
    value = config.section(section_name).get(key)
    if value is None or not str(value).strip():
        raise ValueError(f"{section_name}.{key} is required")
    return _resolve_path(
        value,
        config.source_path.parent,
        label=f"{section_name}.{key}",
    )


def _storage_root(config: _CampaignConfig) -> Path:
    value = config.section("execution").get("storage_root", "/scratch")
    root = _resolve_path(
        value,
        config.source_path.parent,
        label="execution.storage_root",
    )
    if root == Path(root.anchor):
        raise ValueError("execution.storage_root must be a specific directory")
    return root


def _resolve_path(value: object, relative_to: Path, *, label: str) -> Path:
    expanded = os.path.expandvars(str(value))
    if "$" in expanded:
        raise ValueError(f"unresolved environment variable in {label}")
    path = Path(expanded).expanduser()
    return (relative_to / path).resolve() if not path.is_absolute() else path.resolve()


def _release_case_dirs(
    report: Path,
    payload: Mapping[str, Any],
) -> tuple[Path, ...]:
    result = []
    for field in ("development_case_dirs", "released_completed_case_dirs"):
        values = payload.get(field)
        if not isinstance(values, list) or not values:
            raise ValueError(f"VPFP release bank needs nonempty {field}")
        for raw in values:
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError(f"VPFP release bank has invalid {field}")
            path = Path(raw).expanduser()
            path = path if path.is_absolute() else report.parent / path
            if path.is_symlink() or not path.resolve().is_dir():
                raise FileNotFoundError(
                    f"VPFP release-bank case is missing: {path}"
                )
            result.append(path.resolve())
    if len(result) != len(set(result)):
        raise ValueError("VPFP release-bank case directories must be unique")
    return tuple(result)


def _integer(
    values: Mapping[str, Any],
    name: str,
    default: int,
    *,
    minimum: int,
) -> int:
    value = values.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(values: Mapping[str, Any], name: str, default: float) -> float:
    value = values.get(name, default)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
    ):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{label} is missing or not regular: {path}")
    return path.resolve()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value
