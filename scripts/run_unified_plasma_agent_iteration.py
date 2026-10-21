#!/usr/bin/env python3
"""Reproduce a legacy formal Macro campaign for history/control-arm use.

The historical filename and config schema are retained so existing VPFP/GKW
campaigns remain replayable. This is not the canonical product pipeline; new
runs use CampaignController and the AgentExecutionService boundary.
"""

from __future__ import annotations

import argparse
import errno
from dataclasses import asdict
import hashlib
import json
from math import isfinite
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import tempfile
from time import monotonic_ns, perf_counter
from typing import Mapping

from pde_agent_verifier_solver.engine.macro_closed_loop import (
    CompetitiveSearchPolicy,
)
from pde_agent_verifier_solver.engine.promotion import (
    DEFAULT_PROMOTION_PRESET,
    build_promotion_program,
    promotion_metric_references,
)
from pde_agent_verifier_solver.engine.reference_agent_iteration import (
    ReferenceIterationConfig,
    ReferenceMidRoundRecovery,
    ReferenceProcessWorkerConfig,
    load_reference_process_worker_request,
    run_reference_agent_iteration,
    run_reference_process_agent_worker,
    write_reference_process_worker_report,
)
from pde_agent_verifier_solver.engine.selection import (
    EvidenceStage,
    PairedObjectiveMode,
    PairedObjectivePolicy,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_SCHEMA = "unified_plasma_agent_iteration_v1"
CONTROLLER_POLICY_SCHEMA = "unified_macro_controller_policy_v1"
CAMPAIGN_SEED_SCHEMA = "unified_plasma_campaign_seed_v1"
RECOVERY_MANIFEST_SCHEMA = "reference-process-preselection-recovery-v1"
RECOVERY_PHASE = "round_0001_workers_complete_selection_uncommitted"
RECOVERY_SEED_SOURCE = "preselection_recovery_manifest"
AUTO_SEED_BITS = 62
MINIMUM_DURABLE_FREE_BYTES = 1 << 30
GKW_ARTIFACT_FIELDS = (
    "kinetic_edm",
    "kinetic_corpus",
    "potential_edm",
    "potential_corpus",
    "hierarchy_checkpoint",
    "parent_hierarchy_checkpoint",
    "geometry_sidecar",
    "velocity_metric",
    "correction_calibration",
    "time_registry",
    "encode_report",
    "latent_corpus",
    "equation_calibration",
)
GKW_QUALIFICATION_PATH_FIELDS = (
    "certificate",
    "qualification_report",
    "generator_manifest",
    "operator_manifest",
    "physical_source_manifest",
    "seed_plan",
)
GKW_QUALIFICATION_BINDING_FIELDS = (
    "qualification_report_sha256",
    "generator_manifest_sha256",
    "operator_manifest_sha256",
    "physical_source_manifest_sha256",
    "seed_plan_manifest_sha256",
)
NS_FORMAL_ARTIFACT_FIELDS = (
    "pca_checkpoint",
    "pca_selection",
    "pca_audit",
    "flow_checkpoint",
    "flow_selection",
    "flow_train_report",
    "flow_sample_audit",
    "answer_first_report",
)
NS_GENERATOR_MODES = frozenset(
    {
        "synthetic_cpu_chain_test_only",
        "trained_structured_coefficient_flow",
    }
)

# One root-based controller policy serves all equations and regimes. Canonical
# member-case counts include both members of every clean/perturbed root.
PILOT_POLICY = {
    "schema": CONTROLLER_POLICY_SCHEMA,
    "profile": "u0_reference",
    "agent_count": 1,
    "round_count": 1,
    "search_protocol": {
        "min_distinct_screened_candidates": 0,
        "min_full_stage_candidates": 0,
        "min_released_roots": 0,
        "require_final_full_stage": False,
        "leaderboard_limit": 0,
        "cross_share_after_round": 0,
    },
    "agent_roles": [],
    "development_root_count": 2,
    "development_pool_multiplier": 1,
    "development_release_root_limit": 0,
    "development_screen_root_stages": [],
    "development_screen_root_credit_limit": 0,
    "development_screen_confidence_z": 0.0,
    "development_screen_continuation_max_mean_ratio": 1.0,
    "selection_root_count": 2,
    "evidence_stages": [
        {
            "root_count": 2,
            "confidence_z": 0.0,
            "min_mean_improvement": 0.0,
            "min_win_rate": 0.0,
            "require_cvar_no_regression": False,
        }
    ],
    "terminal_root_count": 2,
    "development_case_credits": 4096,
    "seed_stride": 10_000,
    "paired_sampling_max_attempts": 16,
    "paired_objective": {
        "mode": "primary_with_perturbed_guardrail",
        "base_weight": 0.5,
        "perturbed_weight": 0.5,
        "guardrail_relative_margin": 0.05,
        "guardrail_confidence_z": 1.6448536269514722,
    },
    "candidate_compute_budget": 180.0,
    "candidate_query_seconds_per_point_limit": None,
    "agent_timeout_seconds": 28_800.0,
    "agent_hard_timeout_seconds": 28_800.0,
    "development_screen_timeout_seconds": 300.0,
    "model": "gpt-5.6-sol",
    "reasoning_effort": "max",
    "allow_development_truth": False,
    "cvar_tail_fraction": 0.1,
    "terminal_confidence_z": 0.0,
    "terminal_min_win_rate": 0.0,
    "terminal_require_cvar_no_regression": False,
}

_CONTROLLER_POLICY_KEYS = {
    "schema",
    "profile",
    "agent_count",
    "round_count",
    "search_protocol",
    "agent_roles",
    "development_root_count",
    "development_pool_multiplier",
    "development_release_root_limit",
    "development_screen_root_stages",
    "development_screen_root_credit_limit",
    "development_screen_confidence_z",
    "development_screen_continuation_max_mean_ratio",
    "selection_root_count",
    "evidence_stages",
    "terminal_root_count",
    "development_case_credits",
    "seed_stride",
    "paired_sampling_max_attempts",
    "paired_objective",
    "candidate_compute_budget",
    "candidate_query_seconds_per_point_limit",
    "agent_timeout_seconds",
    "agent_hard_timeout_seconds",
    "development_screen_timeout_seconds",
    "model",
    "reasoning_effort",
    "allow_development_truth",
    "cvar_tail_fraction",
    "terminal_confidence_z",
    "terminal_min_win_rate",
    "terminal_require_cvar_no_regression",
    "promotion",
}
_EVIDENCE_STAGE_KEYS = {
    "root_count",
    "confidence_z",
    "min_mean_improvement",
    "min_win_rate",
    "require_cvar_no_regression",
}
_PAIRED_OBJECTIVE_KEYS = {
    "mode",
    "base_weight",
    "perturbed_weight",
    "guardrail_relative_margin",
    "guardrail_confidence_z",
}
_SEARCH_PROTOCOL_KEYS = {
    "min_distinct_screened_candidates",
    "min_full_stage_candidates",
    "min_released_roots",
    "require_final_full_stage",
    "leaderboard_limit",
    "cross_share_after_round",
}


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce a legacy formal Macro history/control-arm campaign. "
            "Canonical product runs use CampaignController."
        )
    )
    parser.add_argument(
        "--legacy-macro-reference",
        action="store_true",
        help=(
            "Explicitly opt in to the history-only MacroClosedLoopRunner "
            "compatibility path"
        ),
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--agent-worker-request", type=Path)
    parser.add_argument("--agent-worker-result", type=Path)
    parser.add_argument("--recover-development-manifest", type=Path)
    parser.add_argument("--recover-development-manifest-sha256")
    arguments = parser.parse_args(argv)
    if not arguments.legacy_macro_reference:
        parser.error(
            "this history/control-arm runner is disabled by default; pass "
            "--legacy-macro-reference to opt in explicitly"
        )
    worker_values = (
        arguments.agent_worker_request,
        arguments.agent_worker_result,
    )
    if any(value is not None for value in worker_values) and not all(
        value is not None for value in worker_values
    ):
        parser.error("worker request and result paths must be supplied together")
    if arguments.preflight_only and arguments.agent_worker_request is not None:
        parser.error("preflight-only and worker mode are mutually exclusive")
    recovery_values = (
        arguments.recover_development_manifest,
        arguments.recover_development_manifest_sha256,
    )
    if any(value is not None for value in recovery_values) and not all(
        value is not None for value in recovery_values
    ):
        parser.error("recovery manifest and SHA-256 must be supplied together")
    if arguments.recover_development_manifest is not None and (
        arguments.preflight_only or arguments.agent_worker_request is not None
    ):
        parser.error("recovery, preflight-only, and worker mode are mutually exclusive")
    return arguments


def _load_config(path: Path) -> dict[str, object]:
    source = path.expanduser().resolve()
    raw = os.path.expandvars(source.read_text(encoding="utf-8"))
    if "${" in raw:
        raise ValueError("campaign config contains unresolved environment variables")
    payload = json.loads(raw)
    if not isinstance(payload, dict) or payload.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unexpected unified plasma campaign schema")
    if payload.get("physics") not in {"vpfp", "gkw", "ns"}:
        raise ValueError("physics must be vpfp, gkw, or ns")
    if not str(payload.get("campaign_id", "")).strip():
        raise ValueError("campaign_id cannot be empty")
    _configured_seed_base(payload)
    _controller_policy(payload)
    payload["_config_path"] = str(source)
    return payload


def _controller_policy(payload: Mapping[str, object]) -> dict[str, object]:
    """Return one canonical paired policy with root and member-case counts."""

    raw = payload.get("controller_policy", PILOT_POLICY)
    optional_fields = {
        "development_screen_timeout_seconds",
        "agent_hard_timeout_seconds",
        "paired_objective",
        "search_protocol",
        "agent_roles",
        "candidate_query_seconds_per_point_limit",
        "promotion",
    }
    required_fields = _CONTROLLER_POLICY_KEYS - optional_fields
    if (
        not isinstance(raw, Mapping)
        or not required_fields.issubset(raw)
        or not set(raw).issubset(_CONTROLLER_POLICY_KEYS)
    ):
        raise ValueError("controller_policy fields do not match its schema")
    raw = dict(raw)
    raw.setdefault("candidate_query_seconds_per_point_limit", None)
    raw.setdefault("development_screen_timeout_seconds", 300.0)
    raw.setdefault(
        "agent_hard_timeout_seconds", raw["agent_timeout_seconds"]
    )
    raw.setdefault(
        "paired_objective",
        {
            "mode": "primary_with_perturbed_guardrail",
            "base_weight": 0.5,
            "perturbed_weight": 0.5,
            "guardrail_relative_margin": 0.05,
            "guardrail_confidence_z": 1.6448536269514722,
        },
    )
    raw.setdefault(
        "search_protocol",
        {
            "min_distinct_screened_candidates": 0,
            "min_full_stage_candidates": 0,
            "min_released_roots": 0,
            "require_final_full_stage": False,
            "leaderboard_limit": 0,
            "cross_share_after_round": 0,
        },
    )
    raw.setdefault("agent_roles", [])
    raw.setdefault(
        "promotion",
        {"preset": DEFAULT_PROMOTION_PRESET, "settings": {}},
    )
    if raw.get("schema") != CONTROLLER_POLICY_SCHEMA:
        raise ValueError("unexpected controller_policy schema")
    profile = _policy_string(raw, "profile")
    model = _policy_string(raw, "model")
    reasoning_effort = _policy_string(raw, "reasoning_effort")
    agent_count = _policy_integer(raw, "agent_count", minimum=1)
    round_count = _policy_integer(raw, "round_count", minimum=1)
    search_raw = raw.get("search_protocol")
    if (
        not isinstance(search_raw, Mapping)
        or set(search_raw) != _SEARCH_PROTOCOL_KEYS
    ):
        raise ValueError("search_protocol fields do not match its schema")
    # Validate legacy milestone fields so old manifests remain readable, then
    # deliberately erase them. The open-search controller optimizes only
    # nomination quality; development effort is never a completion contract.
    _policy_integer(
        search_raw, "min_distinct_screened_candidates", minimum=0
    )
    _policy_integer(search_raw, "min_full_stage_candidates", minimum=0)
    _policy_integer(search_raw, "min_released_roots", minimum=0)
    _policy_bool(search_raw, "require_final_full_stage")
    search_protocol = {
        "min_distinct_screened_candidates": 0,
        "min_full_stage_candidates": 0,
        "min_released_roots": 0,
        "require_final_full_stage": False,
        "leaderboard_limit": _policy_integer(
            search_raw, "leaderboard_limit", minimum=0
        ),
        "cross_share_after_round": _policy_integer(
            search_raw, "cross_share_after_round", minimum=0
        ),
    }
    role_values = raw.get("agent_roles")
    if not isinstance(role_values, list):
        raise ValueError("controller policy agent_roles must be a list")
    agent_roles = []
    for index, value in enumerate(role_values):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"controller policy agent role {index} must be a nonempty string"
            )
        agent_roles.append(value.strip())
    if agent_roles and len(agent_roles) != agent_count:
        raise ValueError(
            "controller policy agent_roles must be empty or match agent_count"
        )
    development_roots = _policy_integer(
        raw, "development_root_count", minimum=1
    )
    development_pool_multiplier = _policy_integer(
        raw, "development_pool_multiplier", minimum=1
    )
    development_release_root_limit = _policy_integer(
        raw, "development_release_root_limit", minimum=0
    )
    screen_credit_limit = _policy_integer(
        raw, "development_screen_root_credit_limit", minimum=0
    )
    screen_confidence_z = _policy_float(
        raw, "development_screen_confidence_z", minimum=0.0
    )
    screen_continuation_ratio = _policy_float(
        raw,
        "development_screen_continuation_max_mean_ratio",
        minimum=0.0,
        strict=True,
    )
    screen_stage_values = raw.get("development_screen_root_stages")
    if not isinstance(screen_stage_values, list):
        raise ValueError(
            "controller policy development screen stages must be a list"
        )
    screen_stages = []
    previous_screen_roots = 0
    for roots_value in screen_stage_values:
        if (
            isinstance(roots_value, bool)
            or not isinstance(roots_value, int)
            or roots_value <= previous_screen_roots
        ):
            raise ValueError(
                "development screen root stages must strictly increase"
            )
        screen_stages.append(roots_value)
        previous_screen_roots = roots_value
    if screen_stages and screen_credit_limit < screen_stages[0]:
        raise ValueError(
            "development screen credits cannot fund the first stage"
        )
    if not screen_stages and (
        search_protocol["min_distinct_screened_candidates"]
        or search_protocol["min_full_stage_candidates"]
        or search_protocol["min_released_roots"]
        or search_protocol["require_final_full_stage"]
    ):
        raise ValueError(
            "competitive search milestones require development screen stages"
        )
    if (
        search_protocol["min_full_stage_candidates"]
        > search_protocol["min_distinct_screened_candidates"]
    ):
        raise ValueError(
            "full-stage candidate minimum cannot exceed distinct-screen minimum"
        )
    if (
        search_protocol["min_released_roots"]
        > development_release_root_limit
    ):
        raise ValueError("released-root milestone exceeds its credit limit")
    if search_protocol["cross_share_after_round"] > round_count:
        raise ValueError("cross-share round cannot exceed campaign rounds")
    if (
        search_protocol["leaderboard_limit"] == 0
        and search_protocol["cross_share_after_round"]
    ):
        raise ValueError("cross-share requires a nonempty leaderboard")
    selection_roots = _policy_integer(raw, "selection_root_count", minimum=1)
    terminal_roots = _policy_integer(raw, "terminal_root_count", minimum=1)
    development_case_credits = _policy_integer(
        raw, "development_case_credits", minimum=2 * development_roots
    )
    seed_stride = _policy_integer(raw, "seed_stride", minimum=1)
    sampling_attempts = _policy_integer(
        raw, "paired_sampling_max_attempts", minimum=1
    )
    candidate_budget = _policy_float(
        raw, "candidate_compute_budget", minimum=0.0, strict=True
    )
    query_limit_value = raw["candidate_query_seconds_per_point_limit"]
    candidate_query_limit = (
        None
        if query_limit_value is None
        else _policy_float(
            raw,
            "candidate_query_seconds_per_point_limit",
            minimum=0.0,
            strict=True,
        )
    )
    agent_timeout = _policy_float(
        raw, "agent_timeout_seconds", minimum=0.0, strict=True
    )
    agent_hard_timeout = _policy_float(
        raw, "agent_hard_timeout_seconds", minimum=0.0, strict=True
    )
    if agent_hard_timeout < agent_timeout:
        raise ValueError(
            "Agent hard timeout cannot be below its active-time timeout"
        )
    screen_timeout = _policy_float(
        raw, "development_screen_timeout_seconds", minimum=0.0, strict=True
    )
    if screen_stages and agent_hard_timeout <= agent_timeout:
        raise ValueError("development screens need hard-wall service allowance")
    if (
        screen_stages
        and agent_hard_timeout - agent_timeout < screen_timeout
    ):
        raise ValueError(
            "hard-wall service allowance must cover one screen timeout"
        )
    cvar_tail_fraction = _policy_float(
        raw, "cvar_tail_fraction", minimum=0.0, maximum=1.0, strict=True
    )
    terminal_z = _policy_float(raw, "terminal_confidence_z", minimum=0.0)
    terminal_win_rate = _policy_float(
        raw, "terminal_min_win_rate", minimum=0.0, maximum=1.0
    )
    allow_truth = _policy_bool(raw, "allow_development_truth")
    terminal_cvar = _policy_bool(
        raw, "terminal_require_cvar_no_regression"
    )
    promotion = _promotion_policy(raw.get("promotion"))
    paired_raw = raw.get("paired_objective")
    if (
        not isinstance(paired_raw, Mapping)
        or set(paired_raw) != _PAIRED_OBJECTIVE_KEYS
    ):
        raise ValueError("paired_objective fields do not match its schema")
    try:
        paired_objective = PairedObjectivePolicy(
            mode=PairedObjectiveMode(_policy_string(paired_raw, "mode")),
            base_weight=_policy_float(paired_raw, "base_weight", minimum=0.0),
            perturbed_weight=_policy_float(
                paired_raw, "perturbed_weight", minimum=0.0
            ),
            guardrail_relative_margin=_policy_float(
                paired_raw, "guardrail_relative_margin", minimum=0.0
            ),
            guardrail_confidence_z=_policy_float(
                paired_raw, "guardrail_confidence_z", minimum=0.0
            ),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("paired_objective is invalid") from exc

    if screen_stages:
        required_full_candidates = max(
            search_protocol["min_full_stage_candidates"],
            int(search_protocol["require_final_full_stage"]),
        )
        required_distinct_candidates = max(
            search_protocol["min_distinct_screened_candidates"],
            required_full_candidates,
        )
        minimum_screen_credits = (
            required_distinct_candidates * screen_stages[0]
            + required_full_candidates
            * (screen_stages[-1] - screen_stages[0])
        )
        if minimum_screen_credits > screen_credit_limit:
            raise ValueError(
                "competitive search milestones exceed screen credit limit"
            )

    stage_values = raw.get("evidence_stages")
    if not isinstance(stage_values, list) or not stage_values:
        raise ValueError("controller_policy evidence_stages must be nonempty")
    stages = []
    previous_roots = 0
    for index, value in enumerate(stage_values):
        if not isinstance(value, Mapping) or set(value) != _EVIDENCE_STAGE_KEYS:
            raise ValueError(f"evidence stage {index} fields do not match")
        roots = _policy_integer(value, "root_count", minimum=1)
        if roots <= previous_roots:
            raise ValueError("evidence stage root counts must strictly increase")
        previous_roots = roots
        stages.append(
            {
                "root_count": roots,
                "case_count": 2 * roots,
                "confidence_z": _policy_float(
                    value, "confidence_z", minimum=0.0
                ),
                "min_mean_improvement": _policy_float(
                    value, "min_mean_improvement", minimum=0.0
                ),
                "min_win_rate": _policy_float(
                    value, "min_win_rate", minimum=0.0, maximum=1.0
                ),
                "require_cvar_no_regression": _policy_bool(
                    value, "require_cvar_no_regression"
                ),
            }
        )
    if terminal_roots < previous_roots:
        raise ValueError("terminal root count cannot be below promotion evidence")

    return {
        "schema": CONTROLLER_POLICY_SCHEMA,
        "profile": profile,
        "agent_count": agent_count,
        "round_count": round_count,
        "search_protocol": search_protocol,
        "agent_roles": agent_roles,
        "development_root_count": development_roots,
        "development_case_count": 2 * development_roots,
        "development_pool_multiplier": development_pool_multiplier,
        "development_release_root_limit": development_release_root_limit,
        "development_screen_root_stages": screen_stages,
        "development_screen_root_credit_limit": screen_credit_limit,
        "development_screen_confidence_z": screen_confidence_z,
        "development_screen_continuation_max_mean_ratio": (
            screen_continuation_ratio
        ),
        "selection_root_count": selection_roots,
        "selection_case_count": 2 * selection_roots,
        "evidence_stages": stages,
        "terminal_root_count": terminal_roots,
        "terminal_case_count": 2 * terminal_roots,
        "development_case_credits": development_case_credits,
        "seed_stride": seed_stride,
        "paired_sampling_max_attempts": sampling_attempts,
        "paired_objective": {
            "mode": paired_objective.mode.value,
            "base_weight": paired_objective.base_weight,
            "perturbed_weight": paired_objective.perturbed_weight,
            "guardrail_relative_margin": paired_objective.guardrail_relative_margin,
            "guardrail_confidence_z": paired_objective.guardrail_confidence_z,
        },
        "candidate_compute_budget": candidate_budget,
        "candidate_query_seconds_per_point_limit": (
            candidate_query_limit
        ),
        "agent_timeout_seconds": agent_timeout,
        "agent_hard_timeout_seconds": agent_hard_timeout,
        "development_screen_timeout_seconds": screen_timeout,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "allow_development_truth": allow_truth,
        "cvar_tail_fraction": cvar_tail_fraction,
        "terminal_confidence_z": terminal_z,
        "terminal_min_win_rate": terminal_win_rate,
        "terminal_require_cvar_no_regression": terminal_cvar,
        "promotion": promotion,
    }


def _promotion_policy(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("controller promotion must be an object")
    spec = dict(value)
    settings = spec.get("settings", {})
    if not isinstance(settings, Mapping):
        raise ValueError("controller promotion settings must be an object")
    if "preset" in spec:
        if set(spec) - {"preset", "settings"}:
            raise ValueError("promotion preset has unexpected fields")
        normalized_settings = dict(settings)
        if (
            spec.get("preset") == DEFAULT_PROMOTION_PRESET
            and "guardrail_distributions" not in normalized_settings
        ):
            # Formal macro reports always contain Base and, when Guidance is
            # enabled, Guided. Perturbation safety is already represented by
            # the report's explicit base_perturbed_non_regression guardrail.
            normalized_settings["guardrail_distributions"] = ["guided"]
        normalized = {
            "preset": spec.get("preset"),
            "settings": normalized_settings,
        }
        # Validate through the shared builder so this launcher accepts every
        # framework preset and cannot drift onto its own smaller preset list.
        program = build_promotion_program(normalized)
        supported = {"base", "base_perturbed", "guided"}
        configured_distributions = {
            reference.distribution
            for reference in promotion_metric_references(normalized)
        }
        guardrail_distributions = getattr(program, "guardrail_distributions", ())
        configured_distributions.update(guardrail_distributions)
        unsupported = configured_distributions - supported
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise ValueError(
                "formal promotion reports do not provide distribution(s): "
                f"{names}; supported distributions are base, base_perturbed, "
                "and guided"
            )
        return normalized
    if set(spec) - {"program", "entrypoint", "settings"} or not {
        "program", "entrypoint"
    }.issubset(spec):
        raise ValueError("custom promotion needs program and entrypoint")
    for name in ("program", "entrypoint"):
        if not isinstance(spec[name], str) or not spec[name].strip():
            raise ValueError(f"custom promotion {name} must be a nonempty string")
    return {
        "program": spec["program"].strip(),
        "entrypoint": spec["entrypoint"].strip(),
        "settings": dict(settings),
    }


def _policy_string(payload: Mapping[str, object], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"controller policy {name} must be a nonempty string")
    return value.strip()


def _policy_integer(
    payload: Mapping[str, object],
    name: str,
    *,
    minimum: int,
) -> int:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"controller policy {name} is below its integer bound")
    return value


def _policy_float(
    payload: Mapping[str, object],
    name: str,
    *,
    minimum: float,
    maximum: float | None = None,
    strict: bool = False,
) -> float:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"controller policy {name} must be numeric")
    result = float(value)
    if not isfinite(result) or result < minimum or (strict and result == minimum):
        raise ValueError(f"controller policy {name} is below its numeric bound")
    if maximum is not None and result > maximum:
        raise ValueError(f"controller policy {name} exceeds its numeric bound")
    return result


def _policy_bool(payload: Mapping[str, object], name: str) -> bool:
    value = payload.get(name)
    if not isinstance(value, bool):
        raise ValueError(f"controller policy {name} must be bool")
    return value


def _configured_seed_base(payload: Mapping[str, object]) -> int | None:
    value = payload.get("seed_base")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("seed_base must be a nonnegative integer or omitted")
    return value


def _resolve_campaign_seed(
    payload: Mapping[str, object],
) -> tuple[dict[str, object], Mapping[str, object]]:
    configured = _configured_seed_base(payload)
    if configured is None:
        seed_base = secrets.randbits(AUTO_SEED_BITS)
        source = "system_entropy"
    else:
        seed_base = configured
        source = "explicit_config"
    resolved = dict(payload)
    resolved["seed_base"] = seed_base
    record = {
        "schema": CAMPAIGN_SEED_SCHEMA,
        "campaign_id": str(payload["campaign_id"]),
        "seed_base": seed_base,
        "source": source,
        "replay": {"config_field": "seed_base", "value": seed_base},
    }
    return resolved, record


def _persist_campaign_seed(
    run_root: Path,
    record: Mapping[str, object],
) -> Path:
    destination = run_root / "campaign_seed.json"
    serialized = (
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".campaign_seed.",
        suffix=".tmp",
        dir=run_root,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
    return destination


def _optional_path(
    payload: Mapping[str, object],
    name: str,
    default: Path,
) -> Path:
    if name not in payload:
        return default.expanduser().resolve()
    return _path(payload, name)


def _storage_roots(
    payload: Mapping[str, object],
    run_root: Path,
) -> dict[str, Path]:
    durable = _optional_path(payload, "storage_root", Path("/scratch"))
    if not durable.is_dir():
        raise FileNotFoundError(f"durable storage root is absent: {durable}")
    if run_root == durable or not run_root.is_relative_to(durable):
        raise ValueError("live plasma run_root must be below storage_root")
    return {"durable": durable}


def _require_free_bytes(root: Path, minimum: int, label: str) -> int:
    available = int(shutil.disk_usage(root).free)
    if available < minimum:
        raise OSError(
            errno.ENOSPC,
            f"{label} has {available} free bytes; at least {minimum} are required",
            str(root),
        )
    return available


def _gkw_qualification_settings(
    settings: Mapping[str, object],
) -> tuple[dict[str, Path], dict[str, str]]:
    qualification = _section(settings, "perturbation_qualification")
    expected = {
        *GKW_QUALIFICATION_PATH_FIELDS,
        *GKW_QUALIFICATION_BINDING_FIELDS,
    }
    if set(qualification) != expected:
        raise ValueError(
            "GKW perturbation qualification fields changed or are incomplete"
        )
    paths = {
        name: _input_path(qualification, name)
        for name in GKW_QUALIFICATION_PATH_FIELDS
    }
    bindings = {
        name: _required_sha256(qualification, name)
        for name in GKW_QUALIFICATION_BINDING_FIELDS
    }
    return paths, bindings


def _ns_generator_mode(settings: Mapping[str, object]) -> str:
    mode = settings.get("generator_mode")
    if not isinstance(mode, str) or mode not in NS_GENERATOR_MODES:
        raise ValueError(
            "NS generator_mode must be synthetic_cpu_chain_test_only or "
            "trained_structured_coefficient_flow"
        )
    return mode


def _ns_guidance_settings(
    settings: Mapping[str, object],
    *,
    mode: str,
):
    guidance = _section(settings, "guidance")
    if not isinstance(guidance.get("enabled"), bool):
        raise ValueError("NS guidance enabled must be boolean")
    enabled = bool(guidance["enabled"])
    if not enabled:
        if set(guidance) != {"enabled"}:
            raise ValueError("disabled NS Guidance accepts only enabled=false")
        return False, None, None, {"enabled": False}
    if mode != "trained_structured_coefficient_flow":
        raise ValueError(
            "NS Guidance requires the formal trained structured-Flow backend"
        )
    if set(guidance) != {"enabled", "protocol", "backend"}:
        raise ValueError("enabled NS Guidance fields are incomplete or changed")

    protocol = _section(guidance, "protocol")
    protocol_fields = {
        "guidance_config_sha256",
        "reward_feedback_case_count",
        "official_round_case_count",
        "official_terminal_case_count",
        "inference_wall_budget_seconds",
        "inference_compute_capacity",
        "training_budget",
        "official_sample_settings",
    }
    if set(protocol) != protocol_fields:
        raise ValueError("shared NS Guidance protocol fields are incomplete or changed")
    training = _section(protocol, "training_budget")
    if set(training) != {
        "maximum_wall_seconds",
        "compute_capacity",
        "maximum_concurrency",
    }:
        raise ValueError("NS Guidance training budget fields changed")
    inference_capacity = protocol.get("inference_compute_capacity")
    training_capacity = training.get("compute_capacity")
    if (
        not isinstance(inference_capacity, str)
        or not inference_capacity.strip()
        or not isinstance(training_capacity, str)
        or not training_capacity.strip()
    ):
        raise ValueError("NS Guidance compute capacity labels cannot be empty")
    sample_settings = protocol.get("official_sample_settings")
    if not isinstance(sample_settings, Mapping):
        raise ValueError("NS official Guidance sample settings must be an object")

    sample_fields = {
        "sampling_policy",
        "flow_state",
        "integration_steps",
        "guidance_strength",
        "guidance_max_flow_ratio",
    }
    if set(sample_settings) != sample_fields:
        raise ValueError("NS official Guidance sample settings fields changed")
    if sample_settings.get("sampling_policy") != "solver_reward_guided_rectified_flow":
        raise ValueError("NS official Guidance sampling policy changed")

    backend = _section(guidance, "backend")
    backend_fields = {
        "training_steps", "batch_size", "width", "depth", "time_width",
        "learning_rate", "weight_decay", "maximum_gradient_norm",
        "minimum_cases", "guidance_strength", "guidance_max_flow_ratio",
    }
    if set(backend) != backend_fields:
        raise ValueError("structured NS Guidance backend fields changed")
    from pde_agent_verifier_solver.plugins.ns import StructuredNSGuidanceConfig

    backend_config = StructuredNSGuidanceConfig(
        training_steps=_execution_integer(backend, "training_steps", minimum=1),
        batch_size=_execution_integer(backend, "batch_size", minimum=1),
        width=_execution_integer(backend, "width", minimum=1),
        depth=_execution_integer(backend, "depth", minimum=1),
        time_width=_execution_integer(backend, "time_width", minimum=1),
        learning_rate=_required_float(backend, "learning_rate", minimum=0.0, strict=True),
        weight_decay=_required_float(backend, "weight_decay", minimum=0.0),
        maximum_gradient_norm=_required_float(backend, "maximum_gradient_norm", minimum=0.0, strict=True),
        minimum_cases=_execution_integer(backend, "minimum_cases", minimum=1),
        guidance_strength=_required_float(backend, "guidance_strength", minimum=0.0),
        guidance_max_flow_ratio=_required_float(backend, "guidance_max_flow_ratio", minimum=0.0, strict=True),
    )
    runtime = _section(settings, "runtime")
    runtime_flow_state = runtime.get("flow_state")
    if not isinstance(runtime_flow_state, str) or not runtime_flow_state.strip():
        raise ValueError("NS runtime flow_state must be a nonempty string")
    official_sample_settings = {
        "sampling_policy": "solver_reward_guided_rectified_flow",
        "flow_state": sample_settings.get("flow_state"),
        "integration_steps": _execution_integer(sample_settings, "integration_steps", minimum=1),
        "guidance_strength": _required_float(sample_settings, "guidance_strength", minimum=0.0),
        "guidance_max_flow_ratio": _required_float(sample_settings, "guidance_max_flow_ratio", minimum=0.0, strict=True),
    }
    expected_sample_settings = {
        "sampling_policy": "solver_reward_guided_rectified_flow",
        "flow_state": runtime_flow_state,
        "integration_steps": _execution_integer(runtime, "integration_steps", minimum=1),
        "guidance_strength": backend_config.guidance_strength,
        "guidance_max_flow_ratio": backend_config.guidance_max_flow_ratio,
    }
    if official_sample_settings != expected_sample_settings:
        raise ValueError("NS official Guidance sample settings do not match runtime/backend")

    configured_sha256 = _required_sha256(protocol, "guidance_config_sha256")
    if configured_sha256 != backend_config.sha256:
        raise ValueError("NS Guidance protocol/backend config digest differs")

    from pde_agent_verifier_solver.engine.guidance_controller import GuidanceProtocolConfig
    from pde_agent_verifier_solver.sdk import GuidanceTrainingBudget

    protocol_config = GuidanceProtocolConfig(
        training_budget=GuidanceTrainingBudget(
            maximum_wall_seconds=_required_float(training, "maximum_wall_seconds", minimum=0.0, strict=True),
            compute_capacity=training_capacity.strip(),
            maximum_concurrency=_execution_integer(training, "maximum_concurrency", minimum=1),
        ),
        guidance_config_sha256=configured_sha256,
        reward_feedback_case_count=_execution_integer(protocol, "reward_feedback_case_count", minimum=1),
        official_round_case_count=_execution_integer(protocol, "official_round_case_count", minimum=1),
        official_terminal_case_count=_execution_integer(protocol, "official_terminal_case_count", minimum=1),
        inference_wall_budget_seconds=_required_float(protocol, "inference_wall_budget_seconds", minimum=0.0, strict=True),
        inference_compute_capacity=inference_capacity.strip(),
        official_sample_settings=official_sample_settings,
    )
    if protocol_config.reward_feedback_case_count < backend_config.minimum_cases:
        raise ValueError("NS Guidance feedback count is below its training minimum")
    audit = {
        "enabled": True,
        "backend_id": "ns-structured-reward-gradient-v1",
        "guidance_config_sha256": backend_config.sha256,
        "protocol": {
            "reward_feedback_case_count": protocol_config.reward_feedback_case_count,
            "official_round_case_count": protocol_config.official_round_case_count,
            "official_terminal_case_count": protocol_config.official_terminal_case_count,
            "training_budget": protocol_config.training_budget.identity_payload(),
            "inference_wall_budget_seconds": protocol_config.inference_wall_budget_seconds,
            "inference_compute_capacity": protocol_config.inference_compute_capacity,
            "unlimited_fresh_online_inference": True,
            "official_sample_settings": dict(official_sample_settings),
        },
        "backend": backend_config.identity_payload(),
    }
    return True, protocol_config, backend_config, audit


def _ns_runtime_settings(
    settings: Mapping[str, object],
    *,
    mode: str,
) -> dict[str, object]:
    runtime = _section(settings, "runtime")
    if mode == "synthetic_cpu_chain_test_only":
        if set(runtime) != {"max_points_per_request"}:
            raise ValueError(
                "synthetic NS runtime accepts only max_points_per_request"
            )
        return {
            "max_points_per_request": _execution_integer(
                runtime, "max_points_per_request", minimum=1
            )
        }

    expected = {
        "time_interval",
        "viscosity",
        "integration_steps",
        "flow_state",
        "perturbation_family",
        "perturbation_amplitude",
        "spectral_max_mode",
        "max_points_per_request",
        "device",
    }
    if set(runtime) != expected:
        raise ValueError("formal NS runtime fields are incomplete or changed")
    flow_state = runtime.get("flow_state")
    if flow_state not in {"raw", "ema"}:
        raise ValueError("NS flow_state must be raw or ema")
    perturbation_family = runtime.get("perturbation_family")
    if perturbation_family not in {
        "spectral_complement",
        "pca_tangent",
        "pca_normal",
    }:
        raise ValueError("NS perturbation_family is unsupported")
    device = runtime.get("device")
    if not isinstance(device, str) or not device.strip():
        raise ValueError("NS runtime device cannot be empty")
    return {
        "time_interval": _float_pair(runtime, "time_interval"),
        "viscosity": _required_float(
            runtime, "viscosity", minimum=0.0, strict=True
        ),
        "integration_steps": _execution_integer(
            runtime, "integration_steps", minimum=1
        ),
        "flow_state": str(flow_state),
        "perturbation_family": str(perturbation_family),
        "perturbation_amplitude": _required_float(
            runtime, "perturbation_amplitude", minimum=0.0, strict=True
        ),
        "spectral_max_mode": _execution_integer(
            runtime, "spectral_max_mode", minimum=1
        ),
        "max_points_per_request": _execution_integer(
            runtime, "max_points_per_request", minimum=1
        ),
        "device": device.strip(),
    }


def _ns_formal_artifacts(
    settings: Mapping[str, object],
) -> tuple[dict[str, Path], dict[str, str]]:
    artifacts = _section(settings, "artifacts")
    expected_sha256 = _section(settings, "artifact_sha256")
    if set(artifacts) != set(NS_FORMAL_ARTIFACT_FIELDS):
        raise ValueError("formal NS artifact paths are incomplete or changed")
    if set(expected_sha256) != set(NS_FORMAL_ARTIFACT_FIELDS):
        raise ValueError("formal NS artifact SHA fields are incomplete or changed")
    return (
        {name: _input_path(artifacts, name) for name in NS_FORMAL_ARTIFACT_FIELDS},
        {
            name: _required_sha256(expected_sha256, name)
            for name in NS_FORMAL_ARTIFACT_FIELDS
        },
    )


def _read_ns_json_report(
    path: Path,
    *,
    label: str,
    schema: str,
) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"formal NS {label} is not readable JSON") from exc
    if not isinstance(payload, Mapping) or payload.get("schema") != schema:
        raise ValueError(f"formal NS {label} has an unexpected schema")
    return payload


def _validate_ns_formal_reports(
    inputs: Mapping[str, Path],
    *,
    expected_sha256: Mapping[str, str],
    flow_state: str,
) -> Mapping[str, object]:
    pca_selection = _read_ns_json_report(
        inputs["pca_selection"],
        label="PCA selection",
        schema="ns_pca_selection_v1",
    )
    pca_audit = _read_ns_json_report(
        inputs["pca_audit"],
        label="PCA audit",
        schema="ns2d_structured_coefficient_pca_audit_v1",
    )
    flow_selection = _read_ns_json_report(
        inputs["flow_selection"],
        label="Flow selection",
        schema="ns_flow_selection_v1",
    )
    _read_ns_json_report(
        inputs["flow_train_report"],
        label="Flow train report",
        schema="ns2d_structured_coefficient_flow_train_report_v1",
    )
    flow_sample = _read_ns_json_report(
        inputs["flow_sample_audit"],
        label="Flow sample audit",
        schema="ns2d_structured_coefficient_flow_sample_audit_v1",
    )
    answer_first = _read_ns_json_report(
        inputs["answer_first_report"],
        label="answer-first report",
        schema="ns2d_structured_coefficient_flow_materialization_audit_v1",
    )
    selected_rank = pca_selection.get("selected_rank")
    selected_step = flow_selection.get("selected_step")
    if (
        isinstance(selected_rank, bool)
        or not isinstance(selected_rank, int)
        or selected_rank < 1
        or isinstance(selected_step, bool)
        or not isinstance(selected_step, int)
        or selected_step < 1
    ):
        raise ValueError("formal NS PCA/Flow selections are invalid")
    pca_rows = pca_audit.get("ranks")
    if not isinstance(pca_rows, list) or not any(
        isinstance(row, Mapping) and row.get("rank") == selected_rank
        for row in pca_rows
    ):
        raise ValueError("formal NS selected PCA rank is absent from its audit")
    candidates = flow_selection.get("candidates")
    selected_rows = (
        [
            row
            for row in candidates
            if isinstance(row, Mapping) and row.get("step") == selected_step
        ]
        if isinstance(candidates, list)
        else []
    )
    if (
        flow_selection.get("minimum_contract_pass_fraction") != 1.0
        or len(selected_rows) != 1
        or selected_rows[0].get("eligible") is not True
        or selected_rows[0].get("contract_pass_fraction") != 1.0
    ):
        raise ValueError("formal NS selected Flow did not pass the full contract gate")
    generated_cases = answer_first.get("generated_cases")
    if (
        flow_sample.get("flow_checkpoint_sha256")
        != expected_sha256["flow_checkpoint"]
        or flow_sample.get("flow_step") != selected_step
        or flow_sample.get("pca_rank") != selected_rank
        or flow_sample.get("flow_state") != flow_state
        or answer_first.get("flow_step") != selected_step
        or answer_first.get("pca_rank") != selected_rank
        or answer_first.get("flow_state") != flow_state
        or answer_first.get("sample_audit_sha256")
        != expected_sha256["flow_sample_audit"]
        or isinstance(generated_cases, bool)
        or not isinstance(generated_cases, int)
        or generated_cases < 1
        or answer_first.get("continuous_contract_passed") != generated_cases
    ):
        raise ValueError("formal NS selected artifact reports are not mutually bound")
    return {
        "formal_report_validation": {
            "selected_pca_rank": selected_rank,
            "selected_flow_step": selected_step,
            "continuous_contract_passed": generated_cases,
            "continuous_contract_tested": generated_cases,
            "minimum_contract_pass_fraction": 1.0,
            "flow_checkpoint_sha256": expected_sha256["flow_checkpoint"],
            "flow_sample_audit_sha256": expected_sha256["flow_sample_audit"],
            "flow_state": flow_state,
        }
    }


def _preflight(payload: Mapping[str, object]) -> Mapping[str, object]:
    run_root = _path(payload, "run_root")
    storage_roots = _storage_roots(payload, run_root)
    durable_free = _require_free_bytes(
        storage_roots["durable"],
        MINIMUM_DURABLE_FREE_BYTES,
        "durable experiment storage",
    )
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError(f"run_root is not empty: {run_root}")

    agent = _section(payload, "agent")
    agent_files = {
        "cold_start_candidate": _input_path(agent, "cold_start_candidate"),
        "agent_launcher": _input_path(agent, "agent_launcher"),
        "codex_binary": _input_path(agent, "codex_binary"),
        "auth_file": _input_path(agent, "auth_file"),
    }
    from pde_agent_verifier_solver.sdk import inspect_candidate_bundle

    cold_bundle = inspect_candidate_bundle(agent_files["cold_start_candidate"])
    for label, path in agent_files.items():
        if label == "cold_start_candidate":
            continue
        if not path.is_file():
            raise FileNotFoundError(f"{label} is absent: {path}")
    for label in ("agent_launcher", "codex_binary"):
        if not os.access(agent_files[label], os.X_OK):
            raise PermissionError(f"{label} is not executable")
    qualification_audit = None

    if payload["physics"] == "vpfp":
        physics = _section(payload, "vpfp")
        inputs = {
            "base_decoder_checkpoint": _input_path(
                physics, "base_decoder_checkpoint"
            ),
            "expanded_decoder_checkpoint": _input_path(
                physics, "expanded_decoder_checkpoint"
            ),
        }
        expected = {
            "base_decoder_checkpoint": _required_sha256(
                physics, "base_decoder_sha256"
            ),
            "expanded_decoder_checkpoint": _required_sha256(
                physics, "expanded_decoder_sha256"
            ),
        }
    elif payload["physics"] == "gkw":
        physics = _section(payload, "gkw")
        artifact_section = _section(physics, "artifacts")
        expected_section = _section(physics, "artifact_sha256")
        if set(expected_section) != set(GKW_ARTIFACT_FIELDS):
            raise ValueError("GKW artifact SHA fields do not match its inputs")
        inputs = {
            name: _input_path(artifact_section, name)
            for name in GKW_ARTIFACT_FIELDS
        }
        expected = {
            name: _required_sha256(expected_section, name)
            for name in GKW_ARTIFACT_FIELDS
        }
        qualification_paths, qualification_bindings = (
            _gkw_qualification_settings(physics)
        )
        for label, path in qualification_paths.items():
            if not path.is_file() or path.is_symlink():
                raise FileNotFoundError(
                    f"GKW production qualification {label} is absent"
                )
        qualification_audit = {
            "files": {
                name: str(path)
                for name, path in qualification_paths.items()
            },
            "bindings": qualification_bindings,
            "loaded_once_at_runtime_start": True,
            "per_case_hash_or_validation": False,
        }
        physical_root = _path(physics, "physical_root")
        external_root = _path(physics, "external_root")
        gyaradax_root = _path(physics, "gyaradax_root")
        for label, root in (
            ("physical_root", physical_root),
            ("external_root", external_root),
            ("gyaradax_root", gyaradax_root),
        ):
            if not root.is_dir():
                raise FileNotFoundError(f"{label} is absent: {root}")
        _require_git_commit(physical_root, str(physics["physical_commit"]))
        _require_git_commit(external_root, str(physics["external_commit"]))
        _require_git_commit(gyaradax_root, str(physics["gyaradax_commit"]))
    else:
        physics = _section(payload, "ns")
        mode = _ns_generator_mode(physics)
        guidance_enabled, _protocol, _backend, guidance_audit = (
            _ns_guidance_settings(physics, mode=mode)
        )
        runtime = _ns_runtime_settings(physics, mode=mode)
        if not cold_bundle.manifest_path.is_file():
            raise ValueError("NS cold start must be an explicit CandidateBundle")
        qualification_audit = {
            "generator_mode": mode,
            "formal_trained_generator": (
                mode == "trained_structured_coefficient_flow"
            ),
            "synthetic_cpu_chain_test_only": (
                mode == "synthetic_cpu_chain_test_only"
            ),
            "guidance_enabled": guidance_enabled,
            "guidance": guidance_audit,
            "online_audit_or_hashing": False,
            "cold_start_candidate_bundle": {
                "entrypoint": cold_bundle.entrypoint,
                "tree_sha256": cold_bundle.tree_sha256,
            },
            "runtime": runtime,
        }
        if mode == "synthetic_cpu_chain_test_only":
            if set(physics) != {"generator_mode", "runtime", "guidance"}:
                raise ValueError(
                    "synthetic NS mode cannot declare formal artifacts or claims"
                )
            inputs = {}
            expected = {}
        else:
            if set(physics) != {
                "generator_mode",
                "artifacts",
                "artifact_sha256",
                "runtime",
                "guidance",
            }:
                raise ValueError("formal NS fields are incomplete or changed")
            inputs, expected = _ns_formal_artifacts(physics)

    verified = {}
    for label, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label} is not a regular input: {path}")
        actual = expected[label]
        verified[label] = actual
    if payload["physics"] == "ns" and inputs:
        report_audit = _validate_ns_formal_reports(
            inputs,
            expected_sha256=expected,
            flow_state=str(
                _section(_section(payload, "ns"), "runtime")["flow_state"]
            ),
        )
        qualification_audit = {**dict(qualification_audit or {}), **report_audit}
    storage_audit = {
        "durable_root": str(storage_roots["durable"]),
        "durable_free_bytes": durable_free,
        "minimum_durable_free_bytes": MINIMUM_DURABLE_FREE_BYTES,
        "official_selection_terminal_mode": (
            "root_pair_streamed_memory_only"
        ),
        "cohort_disk_persistence": False,
    }
    return {
        "campaign_id": payload["campaign_id"],
        "physics": payload["physics"],
        "run_root": str(run_root),
        "input_count": len(inputs),
        "input_sha256": verified,
        "controller_policy": _controller_policy(payload),
        "storage": storage_audit,
        "perturbation_qualification": qualification_audit,
        "seed_policy": {
            "mode": (
                "system_entropy_at_campaign_start"
                if _configured_seed_base(payload) is None
                else "explicit_replay"
            ),
            "seed_base": _configured_seed_base(payload),
        },
    }


def _build_vpfp(payload: Mapping[str, object], run_root: Path):
    import torch

    from agent_verifier.vpfp.continuous_hermite import (
        VPFPContinuousHermiteConfig,
    )
    from agent_verifier.vpfp.high_fidelity import VPFPHighFidelityConfig
    from agent_verifier.vpfp.online_training import (
        VPFPOnlineBatchGenerator,
        VPFPOnlineGenerationConfig,
    )
    from agent_verifier.vpfp.operators import VPFPGrid, VPFPTime
    from agent_verifier.vpfp.robust_perturbation import (
        VPFPRobustPerturbationConfig,
        build_deterministic_direct_model,
    )
    from pde_agent_verifier_solver.plugins.vpfp import (
        VPFPAnchoredPerturbationBackend,
        VPFPContinuousGenerator,
        VPFPLegacyStructuredRuntime,
        bind_vpfp_macro_loop,
    )

    settings = _section(payload, "vpfp")
    base_checkpoint = _input_path(settings, "base_decoder_checkpoint")
    expanded_checkpoint = _input_path(settings, "expanded_decoder_checkpoint")
    base_payload = torch.load(base_checkpoint, map_location="cpu", weights_only=True)
    base_config = VPFPContinuousHermiteConfig.from_dict(
        base_payload["decoder_config"]
    )
    adapter = dict(base_payload.get("robust_answer_first_adapter") or {})
    if (
        adapter.get("unused_neural_parameters_zeroed") is not True
        or adapter.get("continuous_hidden_answer_is_primary") is not True
        or adapter.get("generator_parameterization_public") is not False
    ):
        raise ValueError("VPFP base checkpoint is not answer-first qualified")

    collision = _float_pair(settings, "collision_nu_range")
    temperature = _float_pair(settings, "temperature_range")
    time_domain = _float_pair(settings, "time_domain")
    if (
        tuple(float(value) for value in base_config.collision_nu_range)
        != collision
        or tuple(float(value) for value in base_config.temperature_range)
        != temperature
        or tuple(float(value) for value in base_config.time_domain)
        != time_domain
        or int(base_config.structured_time_control_count) != 33
    ):
        raise ValueError("VPFP checkpoint and requested legacy regime differ")

    generation = VPFPOnlineGenerationConfig(
        batch_size=16,
        nx=16,
        ny=16,
        velocity_grid_size=21,
        nt=33,
        t1=time_domain[1],
        velocity_max=8.0,
        internal_dt_max=0.001,
        dtype="float32",
        cache_capacity_cases=128,
        anchor_case_count=16,
    )
    equation = VPFPHighFidelityConfig(
        internal_dt_max=0.001,
        collision_nu_range=collision,
        temperature_range=temperature,
    )
    device = "cuda"
    online_generator = VPFPOnlineBatchGenerator(
        decoder_config=base_config,
        generation_config=generation,
        equation_config=equation,
        device=device,
    )
    base_model = build_deterministic_direct_model(base_config, device=device)
    materialized_grid = VPFPGrid(
        nx=33,
        ny=33,
        nvx=9,
        nvy=9,
        x_domain=tuple(float(value) for value in base_config.equation.x_domain),
        v_bounds=(-5.0, 5.0, -5.0, 5.0),
    )
    materialized_time = VPFPTime(t0=time_domain[0], t1=time_domain[1], nt=13)
    perturbation = VPFPRobustPerturbationConfig()
    runtime = VPFPLegacyStructuredRuntime(
        generator=online_generator,
        base_model=base_model,
        expanded_decoder_checkpoint=expanded_checkpoint,
        output_root=run_root / "physics" / "vpfp_cases",
        materialized_grid=materialized_grid,
        materialized_time=materialized_time,
        perturbation_config=perturbation,
        device=device,
        materialization_chunk_size=262_144,
    )
    identity = {
        "schema": "vpfp_unified_live_generator_identity_v1",
        "regime_id": str(settings["regime_id"]),
        "base_decoder_sha256": _required_sha256(
            settings, "base_decoder_sha256"
        ),
        "expanded_decoder_sha256": _required_sha256(
            settings, "expanded_decoder_sha256"
        ),
        "collision_nu_range": list(collision),
        "temperature_range": list(temperature),
        "time_domain": list(time_domain),
        "generation": generation.to_dict(),
        "equation": equation.to_dict(),
        "materialization": {
            "nt": 13,
            "nx": 33,
            "ny": 33,
            "nvx": 9,
            "nvy": 9,
            "velocity_bounds": [-5.0, 5.0, -5.0, 5.0],
        },
        "perturbation": asdict(perturbation),
        "unified_commit": _git_commit(ROOT),
    }
    generator_sha256 = _digest_json(identity)
    generator = VPFPContinuousGenerator(
        VPFPAnchoredPerturbationBackend(runtime)
    )
    plugin = bind_vpfp_macro_loop(
        generator=generator,
        base_state=None,
        generator_sha256=generator_sha256,
        generator_id=f"vpfp-live-{settings['regime_id']}",
        generator_settings={"regime_id": str(settings["regime_id"])},
    )
    return plugin, {"generator_identity": identity}


def _build_gkw(payload: Mapping[str, object], run_root: Path):
    from agent_verifier.fusion_simulation_first import (
        gkw_selected_service_factory as service_factory,
    )
    from agent_verifier.fusion_simulation_first import (
        gkw_perturbation_qualification as qualification_module,
    )
    from pde_agent_verifier_solver.plugins.gkw import (
        GKWAnswerFirstGenerator,
        GKWManufacturedGeneratorBackend,
        GKWSelectedTorchRuntime,
        bind_gkw_macro_loop,
    )

    settings = _section(payload, "gkw")
    storage_roots = _storage_roots(payload, run_root)
    physical_source = (_path(settings, "physical_root") / "src").resolve()
    for label, module in (
        ("selected service factory", service_factory),
        ("perturbation qualification", qualification_module),
    ):
        module_path = Path(str(module.__file__)).resolve()
        if not module_path.is_relative_to(physical_source):
            raise RuntimeError(
                f"GKW {label} was imported outside the pinned physical checkout"
            )
    artifact_section = _section(settings, "artifacts")
    artifact_sha256 = _section(settings, "artifact_sha256")
    artifacts = {
        name: _input_path(artifact_section, name)
        for name in GKW_ARTIFACT_FIELDS
    }
    qualification_paths, expected_qualification_bindings = (
        _gkw_qualification_settings(settings)
    )
    certificate = qualification_module.gkw_qualification_certificate_from_payload(
        json.loads(
            qualification_paths["certificate"].read_text(encoding="utf-8")
        )
    )
    actual_qualification_bindings = {
        name: getattr(certificate, name)
        for name in GKW_QUALIFICATION_BINDING_FIELDS
    }
    physical_commit = str(settings["physical_commit"])
    if (
        actual_qualification_bindings != expected_qualification_bindings
        or certificate.physical_git_commit != physical_commit
        or certificate.qualification_code_commit != physical_commit
        or certificate.qualification_sample_count < 2_700
    ):
        raise RuntimeError(
            "GKW perturbation certificate does not bind this production runtime"
        )
    generator_files = {
        name: artifacts[name]
        for name in qualification_module.GKW_QUALIFICATION_GENERATOR_ARTIFACT_KEYS
    }
    operator_files = {
        name: artifacts[name]
        for name in qualification_module.GKW_QUALIFICATION_OPERATOR_ARTIFACT_KEYS
    }
    qualification_validation_started = perf_counter()
    production_qualification = (
        qualification_module.validate_gkw_production_qualification_environment(
            certificate,
            qualification_report_path=(
                qualification_paths["qualification_report"]
            ),
            generator_manifest_path=qualification_paths["generator_manifest"],
            operator_manifest_path=qualification_paths["operator_manifest"],
            physical_source_manifest_path=(
                qualification_paths["physical_source_manifest"]
            ),
            seed_plan_path=qualification_paths["seed_plan"],
            generator_files=generator_files,
            operator_files=operator_files,
            physical_code_root=_path(settings, "physical_root"),
            external_commits={
                "neural_gyrokinetics": str(settings["external_commit"]),
                "gyaradax": str(settings["gyaradax_commit"]),
            },
        )
    )
    qualification_validation_wall_seconds = max(
        0.0, perf_counter() - qualification_validation_started
    )
    certificate_payload = (
        qualification_module.gkw_qualification_certificate_payload(certificate)
    )
    identity = {
        "schema": "gkw_unified_live_generator_identity_v2",
        "artifacts": {
            name: _required_sha256(artifact_sha256, name)
            for name in GKW_ARTIFACT_FIELDS
        },
        "physical_commit": str(settings["physical_commit"]),
        "external_commit": str(settings["external_commit"]),
        "gyaradax_commit": str(settings["gyaradax_commit"]),
        "unified_commit": _git_commit(ROOT),
        "service_policy": {
            "device": "cuda",
            "decoder_precision": "bfloat16",
            "reward_bootstrap_samples": 32,
            "reward_seed_namespace": "gkw-perturbation-qualification",
            "macro_time_count": 6,
            "macro_field_rows": 2048,
            "formal_macro_time_count": 4,
            "maximum_clean_sampling_attempts": 8,
            "oracle_max_total_rows": 20_000_000,
            "enforce_distribution_calibration": True,
        },
        "perturbation_qualification": certificate_payload,
    }
    generator_sha256 = _digest_json(identity)
    service = service_factory.build_gkw_selected_online_service(
        artifacts=service_factory.GKWSelectedServiceArtifacts(
            external_root=_path(settings, "external_root"),
            gyaradax_root=_path(settings, "gyaradax_root"),
            **artifacts,
        ),
        private_root=run_root / "physics" / "gkw_selected_service",
        private_storage_root=storage_roots["durable"],
        device="cuda",
        decoder_inference_precision="bfloat16",
        reward_bootstrap_samples=32,
        reward_seed_namespace="gkw-perturbation-qualification",
    )
    if str(service.decoder.config.neural_inference_precision) != "bfloat16":
        raise RuntimeError("GKW selected decoder precision changed")
    runtime = GKWSelectedTorchRuntime(
        service,
        implementation_sha256=generator_sha256,
        production_qualification=production_qualification,
        private_root=run_root / "physics" / "gkw_unified_runtime",
        private_storage_root=storage_roots["durable"],
        macro_time_count=6,
        macro_field_rows=2048,
        formal_macro_time_count=4,
        maximum_sampling_attempts=8,
        oracle_max_total_rows=20_000_000,
        enforce_distribution_calibration=True,
    )
    generator = GKWAnswerFirstGenerator(
        GKWManufacturedGeneratorBackend(runtime)
    )
    plugin = bind_gkw_macro_loop(
        generator=generator,
        base_state=None,
        generator_sha256=generator_sha256,
        generator_id="gkw-selected-live-answer-first",
    )
    return plugin, {
        "generator_identity": identity,
        "runtime_storage": {
            "durable_root": str(storage_roots["durable"]),
        },
        "qualification_startup": {
            "environment_bound": True,
            "wall_seconds": qualification_validation_wall_seconds,
            "included_in_per_case_generation_timing": False,
        },
    }


def _build_ns(payload: Mapping[str, object], run_root: Path):
    """Build exactly one declared NS scientific tier.

    The synthetic tier is only a process-chain smoke.  The formal tier loads
    the selected structured-coefficient PCA/Flow once at process construction;
    all report and transfer hashes were already checked by controller preflight.
    """

    settings = _section(payload, "ns")
    mode = _ns_generator_mode(settings)
    guidance_enabled, guidance_protocol, guidance_config, guidance_audit = (
        _ns_guidance_settings(settings, mode=mode)
    )
    runtime = _ns_runtime_settings(settings, mode=mode)
    if mode == "synthetic_cpu_chain_test_only":
        from ns_synthetic_smoke_support import build_synthetic_ns_smoke_plugin

        plugin, identity = build_synthetic_ns_smoke_plugin(
            max_points_per_request=int(runtime["max_points_per_request"]),
        )
        return plugin, {
            "generator_identity": identity,
            "artifact_startup": {
                "formal_artifacts_loaded": False,
                "synthetic_cpu_chain_test_only": True,
                "guidance_enabled": guidance_enabled,
            },
        }, None

    artifacts, expected_sha256 = _ns_formal_artifacts(settings)
    identity = {
        "schema": "ns_unified_structured_flow_generator_identity_v1",
        "scientific_tier": "trained_structured_coefficient_flow",
        "formal_training_distribution": True,
        "trained_generator": True,
        "generator_id": "ns-structured-coefficient-flow-formal",
        "artifacts": expected_sha256,
        "runtime": {**runtime, "integration_method": "heun"},
        "guidance_enabled": guidance_enabled,
        "online_audit_or_hashing": False,
        "unified_commit": _git_commit(ROOT),
    }
    generator_sha256 = _digest_json(identity)
    from pde_agent_verifier_solver.plugins.ns import build_structured_ns_backend

    components = build_structured_ns_backend(
        generator_id=str(identity["generator_id"]),
        generator_sha256=generator_sha256,
        flow_checkpoint=artifacts["flow_checkpoint"],
        pca_checkpoint=artifacts["pca_checkpoint"],
        time_interval=tuple(runtime["time_interval"]),
        viscosity=float(runtime["viscosity"]),
        perturbation_amplitude=float(runtime["perturbation_amplitude"]),
        device=str(runtime["device"]),
        integration_steps=int(runtime["integration_steps"]),
        flow_state=str(runtime["flow_state"]),
        integration_method="heun",
        perturbation_family=str(runtime["perturbation_family"]),
        spectral_max_mode=int(runtime["spectral_max_mode"]),
        max_points_per_request=int(runtime["max_points_per_request"]),
    )
    guidance_backend = (
        components.build_guidance_backend(
            artifact_root=run_root / "controller_private" / "ns_guidance",
            config=guidance_config,
        )
        if guidance_enabled
        else None
    )
    if guidance_backend is not None:
        if guidance_protocol is None:
            raise RuntimeError("enabled NS Guidance lacks its protocol")
        if dict(guidance_backend.official_sample_settings) != dict(
            guidance_protocol.official_sample_settings
        ):
            raise ValueError("NS Guidance backend did not consume official sample settings")
    plugin = components.bind(guidance_backend=guidance_backend)
    return plugin, {
        "generator_identity": identity,
        "artifact_startup": {
            "combined_generator_sha256": generator_sha256,
            "pca_flow_binding_validated_once_at_construction": True,
            "formal_reports_and_transfer_hashes_validated_in_preflight": True,
            "per_case_hash_or_validation": False,
            "guidance_enabled": guidance_enabled,
            "guidance": guidance_audit,
            "backend_metadata": dict(components.flow_sampler.artifact_metadata),
        },
    }, guidance_backend


def _iteration_config(
    payload: Mapping[str, object],
    *,
    run_root: Path,
) -> ReferenceIterationConfig:
    agent = _section(payload, "agent")
    policy = _controller_policy(payload)
    physics = str(payload["physics"])
    paired_objective = policy["paired_objective"]
    search_protocol = policy["search_protocol"]
    if not isinstance(search_protocol, Mapping):
        raise RuntimeError("canonical search protocol is unavailable")
    if not isinstance(paired_objective, Mapping):
        raise RuntimeError("canonical paired objective is unavailable")

    if physics == "vpfp":
        prompt = _vpfp_prompt(payload)
    elif physics == "gkw":
        prompt = _gkw_prompt(payload)
    else:
        prompt = _ns_prompt(payload)
    guidance_protocol = None
    if physics == "ns":
        settings = _section(payload, "ns")
        _enabled, guidance_protocol, _backend, _audit = _ns_guidance_settings(
            settings, mode=_ns_generator_mode(settings)
        )
    evidence_stages = tuple(
        EvidenceStage(
            case_count=int(stage["case_count"]),
            confidence_z=float(stage["confidence_z"]),
            min_mean_improvement=float(stage["min_mean_improvement"]),
            min_win_rate=float(stage["min_win_rate"]),
            require_cvar_no_regression=bool(
                stage["require_cvar_no_regression"]
            ),
        )
        for stage in policy["evidence_stages"]
    )
    return ReferenceIterationConfig(
        name=str(payload["campaign_id"]),
        run_root=run_root,
        cohort_store_root=None,
        cold_start_candidate=_path(agent, "cold_start_candidate"),
        agent_launcher=_path(agent, "agent_launcher"),
        codex_binary=_path(agent, "codex_binary"),
        auth_file=_path(agent, "auth_file"),
        prompt=prompt,
        model=str(policy["model"]),
        reasoning_effort=str(policy["reasoning_effort"]),
        agent_timeout_seconds=float(policy["agent_timeout_seconds"]),
        agent_hard_timeout_seconds=float(
            policy["agent_hard_timeout_seconds"]
        ),
        development_screen_timeout_seconds=float(
            policy["development_screen_timeout_seconds"]
        ),
        agent_count=int(policy["agent_count"]),
        search_policy=CompetitiveSearchPolicy(
            min_distinct_screened_candidates=int(
                search_protocol["min_distinct_screened_candidates"]
            ),
            min_full_stage_candidates=int(
                search_protocol["min_full_stage_candidates"]
            ),
            min_released_roots=int(search_protocol["min_released_roots"]),
            require_final_full_stage=bool(
                search_protocol["require_final_full_stage"]
            ),
            leaderboard_limit=int(search_protocol["leaderboard_limit"]),
            cross_share_after_round=int(search_protocol["cross_share_after_round"]),
        ),
        agent_roles=tuple(str(value) for value in policy["agent_roles"]),
        round_count=int(policy["round_count"]),
        development_case_count=int(policy["development_case_count"]),
        selection_case_count=int(policy["selection_case_count"]),
        evidence_stages=evidence_stages,
        terminal_case_count=int(policy["terminal_case_count"]),
        development_case_credits=int(policy["development_case_credits"]),
        seed_base=int(payload["seed_base"]),
        seed_stride=int(policy["seed_stride"]),
        development_pool_multiplier=int(
            policy["development_pool_multiplier"]
        ),
        development_release_root_limit=int(
            policy["development_release_root_limit"]
        ),
        development_screen_root_stages=tuple(
            int(value)
            for value in policy["development_screen_root_stages"]
        ),
        development_screen_root_credit_limit=int(
            policy["development_screen_root_credit_limit"]
        ),
        development_screen_confidence_z=float(
            policy["development_screen_confidence_z"]
        ),
        development_screen_continuation_max_mean_ratio=float(
            policy["development_screen_continuation_max_mean_ratio"]
        ),
        cvar_tail_fraction=float(policy["cvar_tail_fraction"]),
        allow_development_truth=bool(policy["allow_development_truth"]),
        terminal_confidence_z=float(policy["terminal_confidence_z"]),
        terminal_min_win_rate=float(policy["terminal_min_win_rate"]),
        terminal_require_cvar_no_regression=bool(
            policy["terminal_require_cvar_no_regression"]
        ),
        candidate_compute_budget=float(policy["candidate_compute_budget"]),
        candidate_query_seconds_per_point_limit=(
            None
            if policy["candidate_query_seconds_per_point_limit"] is None
            else float(
                policy["candidate_query_seconds_per_point_limit"]
            )
        ),
        paired_perturbations=True,
        paired_objective_policy=PairedObjectivePolicy(
            mode=PairedObjectiveMode(str(paired_objective["mode"])),
            base_weight=float(paired_objective["base_weight"]),
            perturbed_weight=float(paired_objective["perturbed_weight"]),
            guardrail_relative_margin=float(
                paired_objective["guardrail_relative_margin"]
            ),
            guardrail_confidence_z=float(paired_objective["guardrail_confidence_z"]),
        ),
        paired_sampling_max_attempts=int(
            policy["paired_sampling_max_attempts"]
        ),
        guidance_protocol=guidance_protocol,
        promotion_program=build_promotion_program(
            policy["promotion"],
            config_dir=(
                Path(str(payload["_config_path"])).parent
                if payload.get("_config_path") is not None
                else None
            ),
        ),
    )


def _portable_candidate_bundle_instruction() -> str:
    return (
        "Put every required learned weight, checkpoint, table, mesh, or other "
        "portable asset in the declared CandidateBundle. Resolve assets relative "
        "to the entrypoint's __file__; controller and worker paths intentionally "
        "differ. Do not depend on an absolute workspace, scratch, repository, or "
        "other external path. Declare every asset in candidate_bundle.json so the "
        "controller seals and transfers the complete solution."
    )


def _vpfp_prompt(payload: Mapping[str, object]) -> str:
    settings = _section(payload, "vpfp")
    policy = _controller_policy(payload)
    development_instruction = _development_instruction(policy)
    objective_instruction = _paired_objective_instruction(policy)
    runtime_helpers = _reference_runtime_helpers(policy)
    portable_assets = _portable_candidate_bundle_instruction()
    timing_instruction = _candidate_timing_instruction(policy)
    return f"""
You are one of {policy['agent_count']} independent Agents improving a VPFP
macroscopic ROM from a weak hold-initial cold start under controller profile
{policy['profile']}. The campaign has {policy['round_count']} scientific
round(s). Your only optimization objective is to nominate the strongest ROM you
can find under the published candidate wall-clock contract. There is no search
completion quota and no reward for merely finding a valid improvement.

The supplied incumbent, sanitized handoff, public resources, and development
services are campaign data. The live Generator uses regime
{settings['regime_id']}: t in {settings['time_domain']}, collision frequency in
{settings['collision_nu_range']}, and temperature in
{settings['temperature_range']}. Every clean answer-first root has a smooth,
equation-specific perturbed child. Treat each pair as one physical root and
generalize beyond the Generator representation.

The staged public-problem bundle contains the equation, conventions, and
documented numerical Oracle. A public case contains its physical domain,
velocity quadrature, source, and initial data. Match rho, momentum_x,
momentum_y, and kinetic_energy at controller-requested physical points. You
choose the ROM, internal grid or basis, resolution, adaptivity, and interpolation.
{objective_instruction}

Nominate one multi-file CandidateBundle. Its Python entrypoint should export
{portable_assets}
prepare_solution(case, solve_budget=None), returning a handle with
evaluate(points_query) -> AnswerView. Preparation does not receive hidden
scoring coordinates. The handle owns projection from your representation to
the later point query. The controller measures candidate preparation and point
query latency separately; Generator work, truth construction, matching, and
Guidance work are outside both clocks. {timing_instruction}

You may import NumPy and the public pde_agent_verifier_solver SDK. Use
{runtime_helpers} from pde_agent_verifier_solver.engine.reference_agent_runtime,
and write_candidate_bundle_manifest from the SDK.
{development_instruction}
You may search the web and public literature/code. Do not inspect unstaged
repository files, controller-private paths, scratch runs, Generator latents, or
unreleased answers. Use write_reference_candidate_submission(...) to hand off
one official nomination. Private experiments and development screens are not
additional official leaderboard tickets.
""".strip()


def _gkw_prompt(payload: Mapping[str, object]) -> str:
    policy = _controller_policy(payload)
    development_instruction = _development_instruction(policy)
    objective_instruction = _paired_objective_instruction(policy)
    runtime_helpers = _reference_runtime_helpers(policy)
    portable_assets = _portable_candidate_bundle_instruction()
    timing_instruction = _candidate_timing_instruction(policy)
    return f"""
You are one of {policy['agent_count']} independent Agents improving a GKW
electrostatic gyrokinetic macroscopic ROM from a weak initial-hold cold start
under controller profile {policy['profile']}. The campaign has
{policy['round_count']} scientific round(s). Your only optimization objective is
to nominate the strongest ROM you can find under the published candidate
wall-clock contract. There is no search completion quota and no reward for
merely finding a valid improvement.

The supplied incumbent, sanitized handoff, public resources, and development
services are campaign data. Every clean answer-first root has a smooth,
equation-specific perturbed child. Treat each pair as one physical root and
build a method that generalizes beyond the Generator representation.

Match the complex particle_density, parallel_momentum, kinetic_energy, and phi
fields at controller-requested physical points. The staged public-problem bundle
contains the full GKW operator, coefficient/grid conventions, reference
differentiable implementation, and the numerical source/initial-condition
Oracle. Oracle coordinates and batch dimensions are chosen by you; finer or
larger calls consume more backend time. You choose the ROM, internal grid or
basis, resolution, adaptivity, and interpolation.

Mean normalized macro error at the highest allowed compute budget is primary;
CVaR tail error is secondary; component regressions and a finite-sample maximum
are diagnostics, not generic gates or support suprema. Compare physical closures
and preserve useful incumbent structure rather than tuning one small example.
{objective_instruction}
{development_instruction}

Nominate one multi-file CandidateBundle. Its Python entrypoint should export
{portable_assets}
prepare_solution(case, solve_budget=None), returning a handle with
evaluate(points_query) -> AnswerView. Preparation does not receive hidden
scoring coordinates. The handle owns projection from your representation to
the later point query. The controller measures candidate preparation and point
query latency separately; Generator work, truth construction, matching, and
Guidance work are outside both clocks. {timing_instruction}

Use {runtime_helpers} from
pde_agent_verifier_solver.engine.reference_agent_runtime, and
write_candidate_bundle_manifest from the SDK. You may search the web and public
literature/code. Do not inspect unstaged repository files, controller-private
paths, scratch runs, Generator latents, or unreleased answers. Use
write_reference_candidate_submission(...) to hand off one official nomination.
Private experiments and development screens are not additional official
leaderboard tickets. Official ranking uses one common hidden cohort for all
nominations, followed by an untouched terminal cohort.
""".strip()


def _ns_prompt(payload: Mapping[str, object]) -> str:
    settings = _section(payload, "ns")
    policy = _controller_policy(payload)
    development_instruction = _development_instruction(policy)
    objective_instruction = _paired_objective_instruction(policy)
    runtime_helpers = _reference_runtime_helpers(policy)
    portable_assets = _portable_candidate_bundle_instruction()
    timing_instruction = _candidate_timing_instruction(policy)
    mode = _ns_generator_mode(settings)
    if mode == "synthetic_cpu_chain_test_only":
        tier_instruction = (
            "This launch is explicitly synthetic_cpu_chain_test_only. Its analytic "
            "source exercises controller, Oracle, CandidateBundle, timing, matching, "
            "perturbation, and 2-Agent+1-controller GPU wiring only. It is not a "
            "trained-Generator experiment and supports no scientific performance "
            "claim."
        )
    else:
        tier_instruction = (
            "The frozen base Generator is the selected trained structured-coefficient "
            "PCA/rectified-Flow distribution. Every runtime sample is a fresh hidden "
            "continuous field; the public NS forcing, initial condition, and boundary "
            "data are derived answer-first from that field."
        )
    guidance = _section(settings, "guidance")
    if guidance.get("enabled") is True:
        guidance_instruction = (
            "A frozen prior-solver Guidance model is enabled. It overloads two roles: "
            "targeted feedback about typical weaknesses of the previous-round (or "
            "cold-start) solver, and the robust/worst-case evaluation distribution. "
            "Call sample_reference_guidance(...) any number of times for fresh online "
            "Guidance samples, then query Agent-selected physical points with "
            "query_reference_guided_completed_solution(...). These are not promotion "
            "tickets or a fixed released collection. Guidance training, sampling, "
            "truth construction, and Oracle time are outside candidate preparation "
            "and evaluation clocks."
        )
    else:
        guidance_instruction = (
            "No frozen Guidance backend is enabled for this launch, so no guided "
            "distribution, robust/worst-case Guidance score, or Guidance Oracle is "
            "claimed."
        )
    return f"""
You are one of {policy['agent_count']} independent Agents solving the forced 2D
incompressible Navier--Stokes equation from a weak hold-initial cold start under
controller profile {policy['profile']}. The campaign has
{policy['round_count']} scientific round(s). Your only optimization objective is
to nominate the strongest PDE solution method you can find under the published
candidate wall-clock contract. There is no search completion quota and no reward
for merely finding a valid improvement.

{tier_instruction}
Every clean answer-first root has a smooth equation-specific perturbed child.
Treat each pair as one physical problem and generalize beyond the Generator
representation. {guidance_instruction}

The staged public-problem bundle exposes the full velocity-pressure NS operator,
incompressibility constraint, geometry, viscosity, no-slip boundary condition,
and numerical forcing/initial-velocity Oracle. Oracle coordinates and batch
dimensions are chosen by you; finer or larger requests cost more backend time.
Solve for the continuous velocity field and match velocity_x and velocity_y at
later controller-selected physical (t,x,y) points. You choose the numerical
method, internal mesh or basis, resolution, adaptivity, time stepping, and the
interpolation/projection from any discrete representation you construct.

Mean normalized velocity error at the highest allowed compute budget is primary;
CVaR tail error is secondary; component regressions and a finite-sample maximum
are diagnostics rather than generic gates or support suprema.
{objective_instruction}
{development_instruction}

Nominate one multi-file CandidateBundle. Its Python entrypoint must export
{portable_assets}
prepare_solution(case, solve_budget=None), returning a query-blind handle with
evaluate(points_query) -> AnswerView. Preparation never receives hidden scoring
coordinates. The handle owns
continuous access/interpolation at the later point
query. The controller measures solver preparation and point-query evaluation
latency separately. Generator sampling, public-question construction, truth,
matching, numerical Oracle, and Guidance work are outside both clocks. The
{timing_instruction}

Use {runtime_helpers} from
pde_agent_verifier_solver.engine.reference_agent_runtime, and
write_candidate_bundle_manifest from the SDK. You may search the web and public
literature/code. Do not inspect unstaged repository files, controller-private
paths, scratch runs, Generator latents, or unreleased answers. Use
write_reference_candidate_submission(...) to hand off one official nomination.
Private experiments and development screens are not official leaderboard
tickets. Official ranking uses one common hidden cohort for all nominations,
followed by an untouched terminal cohort.
""".strip()


def _paired_objective_instruction(policy: Mapping[str, object]) -> str:
    objective = policy.get("paired_objective")
    if not isinstance(objective, Mapping):
        raise ValueError("controller policy is missing paired_objective")
    if objective.get("mode") == "primary_with_perturbed_guardrail":
        margin_percent = 100.0 * float(objective["guardrail_relative_margin"])
        return (
            "The generated/base distribution is the optimization and ranking "
            "objective. The perturbed distribution is an independent robustness "
            f"guardrail with a {margin_percent:g}% practical non-regression margin, "
            "not a 1:1 term in the objective. Pursue aggressive generated-distribution "
            "improvements while rejecting candidates whose perturbed evidence shows "
            "significant degradation beyond that margin. Use the split generated, "
            "perturbed, and composite screen summaries rather than optimizing only "
            "the composite."
        )
    return (
        "The controller uses the configured generated/perturbed weighted composite "
        f"({float(objective['base_weight']):g}:"
        f"{float(objective['perturbed_weight']):g}) as the ranking objective. "
        "Development screens also expose the split distributions for diagnosis."
    )


def _candidate_timing_instruction(policy: Mapping[str, object]) -> str:
    solve_limit = float(policy["candidate_compute_budget"])
    query_limit = policy.get("candidate_query_seconds_per_point_limit")
    if query_limit is None:
        return (
            f"The per-case preparation limit is {solve_limit} seconds. "
            "This compatibility profile records query latency separately but "
            "does not make it an official eligibility gate; legacy one-shot "
            "predict_macros has no formal timing assurance."
        )
    return (
        f"The per-case query-blind preparation limit is {solve_limit} seconds. "
        "Official development screens, leaderboard ranking, and terminal "
        "evaluation require prepare_solution_then_evaluate_points_v1; the "
        "candidate query phase may only project/interpolate the prepared "
        "continuous solution. Its measured latency must not exceed "
        f"{float(query_limit):g} seconds per requested point. Legacy one-shot "
        "predict_macros and any candidate that defers its solve until evaluate "
        "are ineligible."
    )


def _reference_runtime_helpers(policy: Mapping[str, object]) -> str:
    if policy["development_screen_root_stages"]:
        evaluators = (
            "release_reference_completed_answers, "
            "load_reference_completed_answer_release, "
            "submit_reference_candidate_screen, "
            "poll_reference_candidate_screen, and "
            "screen_reference_candidate"
        )
    else:
        evaluators = "evaluate_reference_candidate"
    return (
        "load_reference_public_workspace, "
        "query_reference_public_oracle, "
        f"{evaluators}, and write_reference_candidate_submission"
    )


def _development_instruction(policy: Mapping[str, object]) -> str:
    screen_stages = tuple(policy["development_screen_root_stages"])
    if screen_stages:
        recommended = ", ".join(str(value) for value in screen_stages)
        return (
            "At launch, call load_reference_public_workspace(...). Inspect its "
            "public_problem_manifest and staged public_resources before choosing a "
            "method. Initial completed macro answers are aligned with the public "
            "development cases and queries. The public numerical Oracle accepts "
            "Agent-selected coordinates through query_reference_public_oracle(...); "
            "equation/source calls and DEVELOPMENT-only completed-solution calls are "
            "separate capabilities. Repeated calls have no protocol-level lifetime "
            "or sample quota: practical wall time, concurrency, backend capacity, and "
            "the documented per-request safety bounds apply. "
            "You may request fresh answer-first roots with "
            "release_reference_completed_answers(...), then load the returned "
            "manifest with load_reference_completed_answer_release(...). Fresh "
            "releases and screens are random; do not force deterministic seeds. "
            "Candidate and incumbent share matched roots only within a comparison. "
            "For optional aggregate feedback, use "
            "screen_reference_candidate(context_path, candidate_path=..., "
            "candidate_id=..., "
            f"requested_root_count={screen_stages[0]}). Request sizes "
            f"{recommended} are efficiency recommendations, not stages or promotion "
            "requirements. Any positive size up to the published per-request "
            "capacity is valid. Repeating a request for the same sealed candidate "
            "draws fresh randomness and accumulates evidence; there is no fixed "
            "lifetime root quota or continuation gate. The helper uses the "
            f"controller-published {policy['development_screen_timeout_seconds']}-"
            "second RPC deadline by default. Async submit/poll helpers may overlap "
            "independent development evaluations up to the published capacity. "
            "Screens return only aggregate loss/CVaR/component and paired confidence "
            "statistics, never hidden case identities, finite truth, or an official "
            "decision. They are private numerical experiments, not nominations. "
            "Candidate IDs and controller-computed bundle digests are opaque. "
            "Controller numerical-service time is excluded from the "
            f"{policy['agent_timeout_seconds']}-second Agent active-time budget, "
            "while the complete subprocess remains bounded by the "
            f"{policy['agent_hard_timeout_seconds']}-second hard wall. The official "
            "nomination can be any immutable bundle; prior screening is unnecessary."
        )
    if bool(policy["allow_development_truth"]):
        return (
            "Load the public workspace and staged public-problem resources first. "
            "Use query_reference_public_oracle(...) for Agent-selected public "
            "equation coordinates. Finite macro truth is available only through "
            "the explicit DEVELOPMENT capability; selection and terminal truths "
            "remain private. Optional candidate evaluation returns development "
            "feedback only and never creates an official nomination."
        )
    return (
        "Load the public workspace and staged public-problem resources first. "
        "Use query_reference_public_oracle(...) for Agent-selected public equation "
        "coordinates. Completed-solution access is unavailable for this runtime; "
        "use aggregate development feedback and public source/initial data."
    )


def _section(payload: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = payload.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"config section {name!r} must be an object")
    return value


def _path(payload: Mapping[str, object], name: str) -> Path:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"config path {name!r} is missing")
    return Path(value).expanduser().resolve()


def _input_path(payload: Mapping[str, object], name: str) -> Path:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"config path {name!r} is missing")
    source = Path(value).expanduser()
    if source.is_symlink():
        raise ValueError(f"config input {name!r} cannot be a symlink")
    return source.resolve()


def _required_sha256(payload: Mapping[str, object], name: str) -> str:
    value = payload.get(name)
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"config digest {name!r} must be lowercase SHA-256")
    return value


def _required_float(
    payload: Mapping[str, object],
    name: str,
    *,
    minimum: float,
    strict: bool = False,
) -> float:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"config value {name!r} must be numeric")
    try:
        resolved = float(value)
    except ValueError as exc:
        raise ValueError(f"config value {name!r} must be numeric") from exc
    if not isfinite(resolved) or (
        resolved <= minimum if strict else resolved < minimum
    ):
        qualifier = ">" if strict else ">="
        raise ValueError(
            f"config value {name!r} must be finite and {qualifier} {minimum}"
        )
    return resolved


def _float_pair(payload: Mapping[str, object], name: str) -> tuple[float, float]:
    value = payload.get(name)
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be a two-number list")
    pair = (float(value[0]), float(value[1]))
    if pair[1] <= pair[0]:
        raise ValueError(f"{name} must be strictly increasing")
    return pair


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def _expect_exact_fields(
    payload: Mapping[str, object],
    *,
    label: str,
    fields: frozenset[str],
) -> None:
    if set(payload) != fields:
        raise ValueError(f"{label} fields do not match the recovery schema")


def _required_git_sha1(payload: Mapping[str, object], name: str) -> str:
    value = payload.get(name)
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"recovery {name} must be a lowercase Git SHA-1")
    return value


def _normalize_recovery_controller_policy(value: object) -> dict[str, object]:
    """Parse a persisted historical policy with today's canonical defaults.

    Campaign seed records store the canonical policy, including derived member
    counts. Older records can legitimately lack optional fields introduced
    later (for example ``promotion``). Remove only those known derived values,
    run the remaining policy through the same parser used for a live campaign,
    and then verify that every value the historical record *did* contain was
    already canonical. An explicit historical choice is therefore never
    replaced by a new default.
    """

    if not isinstance(value, Mapping):
        raise ValueError("recovery source controller policy must be an object")
    stored = dict(value)
    derived_fields = {
        "development_case_count",
        "selection_case_count",
        "terminal_case_count",
    }
    unexpected = set(stored) - (_CONTROLLER_POLICY_KEYS | derived_fields)
    if unexpected:
        names = ", ".join(sorted(unexpected))
        raise ValueError(
            f"recovery source controller policy has unexpected fields: {names}"
        )

    raw = {
        name: field_value
        for name, field_value in stored.items()
        if name in _CONTROLLER_POLICY_KEYS
    }
    stages = raw.get("evidence_stages")
    if isinstance(stages, list):
        raw["evidence_stages"] = [
            {
                name: field_value
                for name, field_value in dict(stage).items()
                if name != "case_count"
            }
            if isinstance(stage, Mapping)
            else stage
            for stage in stages
        ]

    normalized = _controller_policy({"controller_policy": raw})
    for name, historical_value in stored.items():
        if normalized.get(name) != historical_value:
            raise ValueError(
                "recovery source controller policy contains a noncanonical "
                f"value for {name}"
            )
    return normalized


def _resolve_recovery_campaign_seed(
    *,
    payload: Mapping[str, object],
    manifest_argument: Path,
    expected_manifest_sha256: str,
    config_sha256: str,
    controller_policy: Mapping[str, object],
    unified_commit: str,
) -> tuple[dict[str, object], Mapping[str, object], Path]:
    expected_digest = _required_sha256(
        {"manifest_sha256": expected_manifest_sha256},
        "manifest_sha256",
    )
    manifest_input = manifest_argument.expanduser()
    if manifest_input.is_symlink():
        raise ValueError("recovery manifest cannot be a symlink")
    manifest_path = manifest_input.resolve(strict=True)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"recovery manifest is not a file: {manifest_path}")
    actual_digest = expected_digest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("recovery manifest must be a JSON object")
    _expect_exact_fields(
        manifest,
        label="recovery manifest",
        fields=frozenset(
            {
                "schema",
                "phase",
                "source",
                "target",
                "workers",
                "source_common_evidence_imported",
            }
        ),
    )
    if manifest["schema"] != RECOVERY_MANIFEST_SCHEMA:
        raise ValueError("unexpected recovery manifest schema")
    if manifest["phase"] != RECOVERY_PHASE:
        raise ValueError("recovery manifest is not at the pre-selection boundary")
    if manifest["source_common_evidence_imported"] is not False:
        raise ValueError("recovery cannot import source official evidence")
    workers = manifest["workers"]
    if not isinstance(workers, list) or not workers:
        raise ValueError("recovery manifest must contain completed workers")

    source = manifest["source"]
    target = manifest["target"]
    if not isinstance(source, Mapping) or not isinstance(target, Mapping):
        raise ValueError("recovery source and target must be JSON objects")
    _expect_exact_fields(
        source,
        label="recovery source",
        fields=frozenset(
            {
                "run_root",
                "campaign_seed_sha256",
                "config_sha256",
                "unified_commit",
                "campaign_id",
                "seed_base",
                "seed_stride",
                "plugin_identity",
                "cold_start_sha256",
            }
        ),
    )
    _expect_exact_fields(
        target,
        label="recovery target",
        fields=frozenset(
            {
                "unified_commit",
                "plugin_identity",
                "official_scoring_protocol",
                "fresh_official_evaluation",
            }
        ),
    )
    if target["fresh_official_evaluation"] is not True:
        raise ValueError("recovery must require fresh official evaluation")
    if not isinstance(source["plugin_identity"], Mapping) or not isinstance(
        target["plugin_identity"], Mapping
    ):
        raise ValueError("recovery plugin identities must be JSON objects")
    if not isinstance(target["official_scoring_protocol"], str) or not str(
        target["official_scoring_protocol"]
    ).strip():
        raise ValueError("recovery official scoring protocol must be nonempty")

    source_commit = _required_git_sha1(source, "unified_commit")
    target_commit = _required_git_sha1(target, "unified_commit")
    if target_commit != unified_commit:
        raise ValueError(
            f"recovery target commit is {target_commit}, expected {unified_commit}"
        )
    source_config_sha256 = _required_sha256(source, "config_sha256")
    if source_config_sha256 != config_sha256:
        raise ValueError("recovery source config does not match the current config")
    _required_sha256(source, "cold_start_sha256")
    campaign_id = str(payload["campaign_id"])
    if source["campaign_id"] != campaign_id:
        raise ValueError("recovery campaign id does not match the current campaign")
    source_seed_base = source["seed_base"]
    if (
        isinstance(source_seed_base, bool)
        or not isinstance(source_seed_base, int)
        or source_seed_base < 0
    ):
        raise ValueError("recovery source seed_base must be a nonnegative integer")
    configured_seed = _configured_seed_base(payload)
    if configured_seed is not None and configured_seed != source_seed_base:
        raise ValueError("configured seed_base conflicts with recovery source")
    source_seed_stride = source["seed_stride"]
    if (
        isinstance(source_seed_stride, bool)
        or not isinstance(source_seed_stride, int)
        or source_seed_stride != int(controller_policy["seed_stride"])
    ):
        raise ValueError("recovery seed_stride does not match controller policy")

    run_root_value = source["run_root"]
    if not isinstance(run_root_value, str) or not run_root_value.strip():
        raise ValueError("recovery source run_root must be a nonempty path")
    source_root_input = Path(run_root_value).expanduser()
    if not source_root_input.is_absolute() or source_root_input.is_symlink():
        raise ValueError("recovery source run_root must be absolute and not a symlink")
    source_root = source_root_input.resolve(strict=True)
    if not source_root.is_dir():
        raise FileNotFoundError(f"recovery source run_root is absent: {source_root}")
    source_seed_path = source_root / "campaign_seed.json"
    if source_seed_path.is_symlink() or not source_seed_path.is_file():
        raise FileNotFoundError(
            "recovery source campaign_seed.json is absent or is a symlink"
        )
    expected_seed_sha256 = _required_sha256(source, "campaign_seed_sha256")
    actual_seed_sha256 = expected_seed_sha256
    source_seed = json.loads(source_seed_path.read_text(encoding="utf-8"))
    if not isinstance(source_seed, Mapping):
        raise ValueError("recovery source campaign seed must be a JSON object")
    _expect_exact_fields(
        source_seed,
        label="source campaign seed",
        fields=frozenset(
            {
                "schema",
                "campaign_id",
                "seed_base",
                "source",
                "replay",
                "config_path",
                "config_sha256",
                "unified_commit",
                "controller_policy",
                "seed_derivation",
            }
        ),
    )
    source_controller_policy = _normalize_recovery_controller_policy(
        source_seed.get("controller_policy")
    )
    if (
        source_seed["schema"] != CAMPAIGN_SEED_SCHEMA
        or source_seed["campaign_id"] != campaign_id
        or source_seed["seed_base"] != source_seed_base
        or source_seed["config_sha256"] != config_sha256
        or source_seed["unified_commit"] != source_commit
        or source_controller_policy != dict(controller_policy)
        or source_seed["seed_derivation"] != "macro_loop_seed_schedule_v1"
        or source_seed["replay"]
        != {"config_field": "seed_base", "value": source_seed_base}
        or not isinstance(source_seed["source"], str)
        or not source_seed["source"].strip()
        or not isinstance(source_seed["config_path"], str)
        or not source_seed["config_path"].strip()
    ):
        raise ValueError("recovery source campaign seed record is inconsistent")

    resolved = dict(payload)
    resolved["seed_base"] = source_seed_base
    record = {
        "schema": CAMPAIGN_SEED_SCHEMA,
        "campaign_id": campaign_id,
        "seed_base": source_seed_base,
        "source": RECOVERY_SEED_SOURCE,
        "replay": {"config_field": "seed_base", "value": source_seed_base},
        "recovery": {
            "schema": RECOVERY_MANIFEST_SCHEMA,
            "phase": RECOVERY_PHASE,
            "manifest_path": str(manifest_path),
            "manifest_sha256": expected_digest,
            "source_run_root": str(source_root),
            "source_campaign_seed_sha256": expected_seed_sha256,
            "source_unified_commit": source_commit,
            "source_seed_origin": source_seed["source"],
        },
    }
    return resolved, record, manifest_path



def _digest_json(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_commit(root: Path) -> str:
    return subprocess.check_output(
        ("git", "-C", str(root), "rev-parse", "HEAD"),
        text=True,
    ).strip()


def _require_git_commit(root: Path, expected: str) -> None:
    actual = _git_commit(root)
    if actual != expected:
        raise ValueError(f"Git checkout {root} is {actual}, expected {expected}")
    dirty = subprocess.check_output(
        (
            "git",
            "-C",
            str(root),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ),
        text=True,
    )
    if dirty:
        raise ValueError(f"Git checkout is dirty: {root}")


def _process_worker_config(
    payload: Mapping[str, object],
) -> ReferenceProcessWorkerConfig | None:
    raw = payload.get("execution")
    if raw is None:
        return None
    expected_fields = {
        "mode",
        "worker_gpu_count",
        "controller_gpu_count",
        "cpus_per_worker",
        "timeout_seconds",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected_fields:
        raise ValueError("execution fields do not match the process-worker schema")
    if raw.get("mode") != "process_per_agent_with_central_gpu":
        raise ValueError("unsupported Agent worker execution mode")
    policy = _controller_policy(payload)
    worker_count = _execution_integer(raw, "worker_gpu_count", minimum=1)
    controller_count = _execution_integer(raw, "controller_gpu_count", minimum=1)
    cpus_per_worker = _execution_integer(raw, "cpus_per_worker", minimum=1)
    timeout_seconds = float(raw["timeout_seconds"])
    if not isfinite(timeout_seconds) or timeout_seconds <= 0.0:
        raise ValueError("execution timeout_seconds must be finite and positive")
    if worker_count != int(policy["agent_count"]):
        raise ValueError("execution requires exactly one GPU worker per Agent")
    if controller_count != 1:
        raise ValueError("execution requires exactly one central GPU")

    worker_environment = os.environ.get("PDE_AGENT_WORKER_GPU_DEVICES", "")
    worker_devices = tuple(worker_environment.split(","))
    controller_device = os.environ.get("PDE_CONTROLLER_GPU_DEVICE", "")
    visible_controller = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if (
        len(worker_devices) != worker_count
        or any(not value or value.strip() != value for value in worker_devices)
        or len(set(worker_devices)) != len(worker_devices)
        or not controller_device
        or controller_device.strip() != controller_device
        or controller_device in worker_devices
        or visible_controller != controller_device
    ):
        raise RuntimeError(
            "process-worker GPU environment is not a distinct 2+1 singleton layout"
        )
    temporary_root = os.environ.get("SLURM_TMPDIR")
    job_id = os.environ.get("SLURM_JOB_ID")
    if not temporary_root or not job_id:
        raise RuntimeError("process workers require a Slurm node-local directory")
    return ReferenceProcessWorkerConfig(
        command=(
            sys.executable,
            str(Path(__file__).resolve()),
            "--legacy-macro-reference",
            "--config",
            str(payload["_config_path"]),
        ),
        gpu_devices=worker_devices,
        controller_gpu_device=controller_device,
        node_local_root=(
            Path(temporary_root).resolve() / f"pde-agent-workers-{job_id}"
        ),
        cpus_per_worker=cpus_per_worker,
        worker_timeout_seconds=timeout_seconds,
    )


def _execution_integer(
    payload: Mapping[str, object],
    name: str,
    *,
    minimum: int,
) -> int:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"execution {name} must be an integer >= {minimum}")
    return value


def _cuda_worker_audit(torch, *, started_monotonic_ns: int) -> Mapping[str, object]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    assigned = os.environ.get("PDE_ASSIGNED_AGENT_GPU_TOKEN", "")
    slot_raw = os.environ.get("PDE_ASSIGNED_AGENT_GPU_SLOT", "")
    if (
        not visible
        or "," in visible
        or visible != assigned
        or not slot_raw.isdigit()
    ):
        raise RuntimeError("Agent worker did not inherit one assigned GPU token")
    available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count())
    current_device = int(torch.cuda.current_device()) if available else -1
    if not available or device_count != 1 or current_device != 0:
        raise RuntimeError("Agent worker CUDA runtime is not singleton-bound")
    properties = torch.cuda.get_device_properties(0)
    device_uuid = getattr(properties, "uuid", None)
    if isinstance(device_uuid, bytes):
        device_uuid = device_uuid.decode("ascii", errors="strict")
    elif device_uuid is not None:
        device_uuid = str(device_uuid)
    return {
        "pid": os.getpid(),
        "hostname": os.uname().nodename,
        "available": available,
        "visible_devices": visible,
        "assigned_token": assigned,
        "assigned_slot": int(slot_raw),
        "device_count": device_count,
        "current_device": current_device,
        "device_name": str(torch.cuda.get_device_name(0)),
        "device_uuid": device_uuid,
        "started_monotonic_ns": started_monotonic_ns,
    }


def _run_process_agent_worker(
    args: argparse.Namespace,
    payload: Mapping[str, object],
) -> int:
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_TMPDIR"):
        raise RuntimeError("Agent GPU worker requires a Slurm compute node")
    request_path = args.agent_worker_request.expanduser().resolve()
    result_path = args.agent_worker_result.expanduser().resolve()
    if request_path.parent != result_path.parent:
        raise ValueError("worker request and result must share one private directory")
    started_ns = monotonic_ns()
    request, request_sha256 = load_reference_process_worker_request(request_path)
    campaign = request.get("campaign")
    if not isinstance(campaign, Mapping) or not isinstance(
        campaign.get("seed_base"), int
    ):
        raise ValueError("worker request has no resolved campaign seed")
    resolved_payload = dict(payload)
    resolved_payload["seed_base"] = int(campaign["seed_base"])

    import torch

    gpu_audit = _cuda_worker_audit(torch, started_monotonic_ns=started_ns)
    worker_root = request_path.parent
    if resolved_payload["physics"] == "vpfp":
        plugin, _provenance = _build_vpfp(resolved_payload, worker_root)
    elif resolved_payload["physics"] == "gkw":
        plugin, _provenance = _build_gkw(resolved_payload, worker_root)
    else:
        plugin, _provenance, _guidance = _build_ns(resolved_payload, worker_root)
    report = run_reference_process_agent_worker(
        config=_iteration_config(
            resolved_payload,
            run_root=_path(resolved_payload, "run_root"),
        ),
        plugin=plugin,
        request=request,
        request_sha256=request_sha256,
        worker_root=worker_root,
        gpu_audit=gpu_audit,
        legacy_macro_reference=True,
    )
    write_reference_process_worker_report(result_path, report)
    print(
        json.dumps(
            {
                "worker_status": "complete",
                "agent_id": report["agent_id"],
                "round_index": report["round_index"],
                "gpu": report["gpu"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    payload = _load_config(args.config)
    if args.agent_worker_request is not None:
        return _run_process_agent_worker(args, payload)
    controller_policy = _controller_policy(payload)
    config_path = Path(str(payload["_config_path"]))
    config_sha256 = _sha256_file(config_path)
    unified_commit: str | None = None
    base_campaign_seed: Mapping[str, object] | None = None
    recovery: ReferenceMidRoundRecovery | None = None
    if args.recover_development_manifest is not None:
        unified_commit = _git_commit(ROOT)
        _require_git_commit(ROOT, unified_commit)
        payload, base_campaign_seed, manifest_path = (
            _resolve_recovery_campaign_seed(
                payload=payload,
                manifest_argument=args.recover_development_manifest,
                expected_manifest_sha256=(
                    args.recover_development_manifest_sha256
                ),
                config_sha256=config_sha256,
                controller_policy=controller_policy,
                unified_commit=unified_commit,
            )
        )
        recovery = ReferenceMidRoundRecovery(
            manifest_path,
            args.recover_development_manifest_sha256,
        )
    preflight = _preflight(payload)
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0
    print(
        json.dumps({"preflight": preflight}, sort_keys=True),
        flush=True,
    )
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_TMPDIR"):
        raise RuntimeError("live plasma Agent iteration requires a Slurm node")
    process_workers = _process_worker_config(payload)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("live plasma Agent iteration requires CUDA")
    if process_workers is not None and (
        torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0
    ):
        raise RuntimeError("central evaluator is not singleton-GPU bound")
    controller_properties = torch.cuda.get_device_properties(0)
    controller_device_uuid = getattr(controller_properties, "uuid", None)
    if isinstance(controller_device_uuid, bytes):
        controller_device_uuid = controller_device_uuid.decode(
            "ascii", errors="strict"
        )
    elif controller_device_uuid is not None:
        controller_device_uuid = str(controller_device_uuid)
    controller_cuda_audit = {
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_count": int(torch.cuda.device_count()),
        "current_device": int(torch.cuda.current_device()),
        "device_name": str(torch.cuda.get_device_name(0)),
        "device_uuid": controller_device_uuid,
    }
    if unified_commit is None:
        unified_commit = _git_commit(ROOT)
        _require_git_commit(ROOT, unified_commit)
    run_root = _path(payload, "run_root")
    run_root.mkdir(parents=True, exist_ok=True)
    if base_campaign_seed is None:
        payload, base_campaign_seed = _resolve_campaign_seed(payload)
    campaign_seed = {
        **dict(base_campaign_seed),
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "unified_commit": unified_commit,
        "controller_policy": dict(controller_policy),
        "seed_derivation": "macro_loop_seed_schedule_v1",
    }
    seed_path = _persist_campaign_seed(run_root, campaign_seed)
    print(
        json.dumps(
            {"campaign_seed": campaign_seed, "path": str(seed_path)},
            sort_keys=True,
        ),
        flush=True,
    )
    guidance_backend = None
    if payload["physics"] == "vpfp":
        plugin, provenance = _build_vpfp(payload, run_root)
    elif payload["physics"] == "gkw":
        plugin, provenance = _build_gkw(payload, run_root)
    else:
        plugin, provenance, guidance_backend = _build_ns(payload, run_root)
    provenance = {
        **provenance,
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "controller_policy": dict(controller_policy),
        "campaign_seed": dict(campaign_seed),
        "preflight": preflight,
        "unified_commit": unified_commit,
        "controller_cuda": controller_cuda_audit,
    }
    result = run_reference_agent_iteration(
        config=_iteration_config(payload, run_root=run_root),
        plugin=plugin,
        provenance=provenance,
        process_workers=process_workers,
        guidance_backend=guidance_backend,
        recovery=recovery,
        legacy_macro_reference=True,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
