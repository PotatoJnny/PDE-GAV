#!/usr/bin/env python
"""Prepare two authorized Max workspaces for one long-time VPFP regime."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from agent_verifier.vpfp.agent_workspace_v2 import (
    VPFPRobustAgentWorkspaceConfig,
    build_vpfp_robust_agent_workspace,
)
from agent_verifier.vpfp.answer_first_online_protocol import (
    AGGREGATE_LIMITS_ACCOUNTING_ONLY,
    ANSWER_FIRST_DEVELOPMENT_RESOURCE_SCHEMA,
    ANSWER_FIRST_ONLINE_GENERATOR_CONFIG_SCHEMA,
    ANSWER_FIRST_RESOLUTION_MATERIALIZATION_SCHEMA,
    validate_answer_first_controller_config,
)
from agent_verifier.vpfp.compute_budget import SolverComputeBudget
from agent_verifier.vpfp.continuous_hermite import VPFPContinuousHermiteConfig
from agent_verifier.vpfp.continuous_macros import MACRO_OUTPUT_CONTRACT
from agent_verifier.vpfp.development_runtime_v2 import (
    VPFPDevelopmentRuntimeConfig,
)
from agent_verifier.vpfp.external_launch_v2 import (
    build_vpfp_robust_external_launch_packet,
    write_vpfp_robust_external_launch_packet,
)
from agent_verifier.vpfp.high_fidelity import VPFPHighFidelityConfig
from agent_verifier.vpfp.online_training import VPFPOnlineGenerationConfig
from agent_verifier.vpfp.oracle_training import inspect_solver_bundle
from agent_verifier.vpfp.runtime_paths import is_persistent_run_path


REPORT_SCHEMA = "vpfp_long_time_robust_two_max_preparation_v2"
EVALUATOR_CONFIG_SCHEMA = "vpfp_development_evaluator_service_config_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-bank-report", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--base-decoder-checkpoint", required=True)
    parser.add_argument("--expanded-decoder-checkpoint", required=True)
    parser.add_argument("--champion-solver", required=True)
    parser.add_argument("--agent-a-prior-solver")
    parser.add_argument("--agent-b-prior-solver")
    parser.add_argument("--agent-a-prior-feedback")
    parser.add_argument("--agent-b-prior-feedback")
    parser.add_argument(
        "--agent-prompt-file",
        help=(
            "optional researcher-written prompt used for both Agents; "
            "the solver contract and scientific instructions remain separate"
        ),
    )
    parser.add_argument("--cluster-notes", required=True)
    parser.add_argument("--round-index", type=int, default=0)
    parser.add_argument("--seed-base", type=int, required=True)
    parser.add_argument("--active-wall-seconds", type=int, default=28800)
    parser.add_argument("--development-case-credits", type=int, default=4096)
    parser.add_argument("--generation-batch-size", type=int, default=16)
    parser.add_argument("--guidance-pool-size", type=int, default=4)
    parser.add_argument("--user-authorized-external-dispatch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.out).resolve()
    if output.exists() or not is_persistent_run_path(output):
        raise SystemExit("--out must be fresh and below the configured run storage root")
    if not args.user_authorized_external_dispatch:
        raise SystemExit("explicit user authorization is required")
    if (
        args.active_wall_seconds < 3600
        or args.development_case_credits < 64
        or args.generation_batch_size != 16
        or args.guidance_pool_size < 2
        or int(args.seed_base) % 4 != 0
        or int(args.round_index) < 0
    ):
        raise SystemExit("invalid VPFP main-experiment preparation settings")

    release_report = _read_json(Path(args.release_bank_report).resolve())
    if (
        release_report.get("schema")
        != "vpfp_long_time_answer_first_release_bank_v2"
        or release_report.get("passed") is not True
    ):
        raise SystemExit("release bank did not pass")
    regime = dict(release_report["regime"])
    development_cases = [
        Path(value).resolve() for value in release_report["development_case_dirs"]
    ]
    released_cases = [
        Path(value).resolve()
        for value in release_report["released_completed_case_dirs"]
    ]
    if len(released_cases) != 32:
        raise SystemExit("main preparation requires exactly 32 released solutions")

    base_checkpoint = Path(args.base_decoder_checkpoint).resolve()
    expanded_checkpoint = Path(args.expanded_decoder_checkpoint).resolve()
    champion = Path(args.champion_solver).resolve()
    notes = Path(args.cluster_notes).resolve()
    for path in (base_checkpoint, expanded_checkpoint, champion, notes):
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
    base_payload = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
    expanded_payload = torch.load(
        expanded_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    base_config = VPFPContinuousHermiteConfig.from_dict(
        base_payload["decoder_config"]
    )
    expanded_config = VPFPContinuousHermiteConfig.from_dict(
        expanded_payload["decoder_config"]
    )
    for payload in (base_payload, expanded_payload):
        adapter = dict(payload.get("robust_answer_first_adapter") or {})
        if not (
            adapter.get("continuous_hidden_answer_is_primary") is True
            and adapter.get("generator_parameterization_public") is False
        ):
            raise SystemExit("campaign checkpoint is not answer-first qualified")
    if (
        tuple(base_config.collision_nu_range)
        != tuple(float(value) for value in regime["collision_nu_range"])
        or tuple(base_config.temperature_range)
        != tuple(float(value) for value in regime["temperature_range"])
        or tuple(base_config.time_domain)
        != tuple(float(value) for value in regime["time_domain"])
    ):
        raise SystemExit("release bank and campaign checkpoint regimes differ")
    champion_bundle = inspect_solver_bundle(champion)
    agent_prior_solvers = {}
    agent_prior_bundles = {}
    for agent_id, raw_path in (
        ("vpfp_max_a", args.agent_a_prior_solver),
        ("vpfp_max_b", args.agent_b_prior_solver),
    ):
        prior = champion if raw_path is None else Path(raw_path).resolve()
        if not prior.is_file() or prior.is_symlink():
            raise FileNotFoundError(prior)
        agent_prior_solvers[agent_id] = prior
        agent_prior_bundles[agent_id] = inspect_solver_bundle(prior)
    agent_prior_feedback = {}
    for agent_id, raw_path in (
        ("vpfp_max_a", args.agent_a_prior_feedback),
        ("vpfp_max_b", args.agent_b_prior_feedback),
    ):
        if raw_path is None:
            agent_prior_feedback[agent_id] = None
            continue
        feedback = Path(raw_path).resolve()
        if not feedback.is_file() or feedback.is_symlink():
            raise FileNotFoundError(feedback)
        agent_prior_feedback[agent_id] = feedback
    custom_prompt = None
    if args.agent_prompt_file is not None:
        custom_prompt = Path(args.agent_prompt_file).resolve()
        if not custom_prompt.is_file() or custom_prompt.is_symlink():
            raise FileNotFoundError(custom_prompt)

    output.mkdir(parents=True)
    controller_root = output / "controller_private"
    controller_root.mkdir()
    workspaces_root = output / "workspaces"
    workspaces_root.mkdir()
    generation = VPFPOnlineGenerationConfig(
        batch_size=16,
        nx=16,
        ny=16,
        velocity_grid_size=21,
        nt=int(base_config.structured_time_control_count),
        t1=float(base_config.time_domain[1]),
        velocity_max=8.0,
        internal_dt_max=0.001,
        cache_capacity_cases=128,
        anchor_case_count=16,
    )
    equation = VPFPHighFidelityConfig(
        internal_dt_max=0.001,
        collision_nu_range=tuple(base_config.collision_nu_range),
        temperature_range=tuple(base_config.temperature_range),
    )
    resource_budget = {
        "schema": ANSWER_FIRST_DEVELOPMENT_RESOURCE_SCHEMA,
        "active_wall_time_seconds": int(args.active_wall_seconds),
        "scratch_bytes": 200 * 1024**3,
        "allocation": {
            "gpus": 1,
            "cpus": 16,
            "memory_bytes": 96 * 1024**3,
        },
        "final_solver": {
            "soft_wall_time_seconds_per_case": 900.0,
            "hard_wall_time_seconds_per_case": 1800.0,
        },
        "aggregate_query_counts": AGGREGATE_LIMITS_ACCOUNTING_ONLY,
    }
    resolution_policy = {
        "schema": ANSWER_FIRST_RESOLUTION_MATERIALIZATION_SCHEMA,
        "enabled": True,
        "aggregate_limits_mode": AGGREGATE_LIMITS_ACCOUNTING_ONLY,
        "max_target_source_elements_per_view": 200_000_000,
        "max_resolution": {
            "nx": 129,
            "ny": 129,
            "nvx": 33,
            "nvy": 33,
            "nt": 129,
        },
    }
    guidance_budget = SolverComputeBudget(
        tier=1,
        wall_time_seconds=180.0,
        cpus=16,
        gpus=1,
        memory_bytes=64 * 1024**3,
    ).to_dict()

    rows = []
    for index, agent_id in enumerate(("vpfp_max_a", "vpfp_max_b")):
        workspace = workspaces_root / agent_id
        private = controller_root / agent_id
        prior_solver = agent_prior_solvers[agent_id]
        prior_bundle = agent_prior_bundles[agent_id]
        workspace_report = build_vpfp_robust_agent_workspace(
            output_dir=workspace,
            controller_private_root=private,
            config=VPFPRobustAgentWorkspaceConfig(
                agent_id=agent_id,
                round_index=int(args.round_index),
                regime_id=str(regime["regime_id"]),
                collision_nu_range=tuple(regime["collision_nu_range"]),
                temperature_range=tuple(regime["temperature_range"]),
                time_domain=tuple(regime["time_domain"]),
                active_wall_time_seconds=int(args.active_wall_seconds),
                development_evaluator_case_credits=int(
                    args.development_case_credits
                ),
                generator_max_batch=16,
            ),
            development_case_dirs=development_cases,
            released_completed_case_dirs=released_cases,
            previous_solver_path=prior_solver,
            cluster_notes_path=notes,
            prior_feedback_path=agent_prior_feedback[agent_id],
        )
        if custom_prompt is not None:
            # The prompt states the research objective. The generated solver
            # contract still tells the Agent exactly how to call the online
            # Generator and Evaluator.
            shutil.copy2(custom_prompt, workspace / "prompt.md")

        online_config = {
            "schema": ANSWER_FIRST_ONLINE_GENERATOR_CONFIG_SCHEMA,
            "workspace": str(workspace),
            "private_root": str(private / "online_runtime"),
            "target_solver": str(prior_solver),
            "target_solver_sha256": prior_bundle["aggregate_sha256"],
            "solver_output_contract_version": MACRO_OUTPUT_CONTRACT,
            "compute_budget_contract_enabled": True,
            "decoder_checkpoint": str(base_checkpoint),
            "generation_config": generation.to_dict(),
            "equation_config": equation.to_dict(),
            "materialization_grid": {
                "nx": 33,
                "ny": 33,
                "nvx": 9,
                "nvy": 9,
                "x_domain": list(base_config.equation.x_domain),
                "v_bounds": [-5.0, 5.0, -5.0, 5.0],
            },
            "materialization_time": {
                "t0": float(base_config.time_domain[0]),
                "t1": float(base_config.time_domain[1]),
                "nt": 13,
            },
            "regime": regime,
            "guidance": {
                "mode": "direct_frozen_solver_error_best_of_fresh_answer_pool",
                "candidate_pool_size": int(args.guidance_pool_size),
                "solver_budget": guidance_budget,
            },
            "materialization_chunk_size": 262_144,
            "oracle_chunk_size": 262_144,
            "limits": {
                "max_batch": 16,
                "max_points_per_call": 2_000_000,
                "aggregate_limits_mode": AGGREGATE_LIMITS_ACCOUNTING_ONLY,
            },
            "seed_starts": {
                "train": int(args.seed_base) + 2 * index,
                "validation": int(args.seed_base) + 2 * index + 1,
            },
            "seed_step": 4,
            "development_resource_budget": resource_budget,
            "resolution_materialization": resolution_policy,
        }
        online_config = validate_answer_first_controller_config(online_config)
        online_config_path = private / "online_generator_controller.json"
        _write_json(online_config_path, online_config)

        runtime_config = VPFPDevelopmentRuntimeConfig(
            regime_id=str(regime["regime_id"]),
            collision_nu_range=tuple(regime["collision_nu_range"]),
            temperature_range=tuple(regime["temperature_range"]),
            time_domain=tuple(regime["time_domain"]),
            time_control_count=int(base_config.structured_time_control_count),
            generation_batch_size=16,
            finalist_case_count=1,
        )
        evaluator_config = {
            "schema": EVALUATOR_CONFIG_SCHEMA,
            "workspace": str(workspace),
            "private_root": str(private / "evaluator_runtime"),
            "base_decoder_checkpoint": str(base_checkpoint),
            "expanded_decoder_checkpoint": str(expanded_checkpoint),
            "comparison_solver": str(prior_solver),
            "checkpoint_index": int(args.round_index),
            "runtime_config": asdict(runtime_config),
            "device": "cuda",
        }
        evaluator_config_path = private / "development_evaluator_controller.json"
        _write_json(evaluator_config_path, evaluator_config)

        packet = build_vpfp_robust_external_launch_packet(
            workspace=workspace,
            user_authorized_external_dispatch=True,
        )
        packet_path = write_vpfp_robust_external_launch_packet(
            private / "external_launch_packet.json",
            packet,
        )
        rows.append(
            {
                "agent_id": agent_id,
                "development_incumbent_solver": str(prior_solver),
                "development_incumbent_sha256": prior_bundle["aggregate_sha256"],
                "prior_feedback": workspace_report["prior_feedback"],
                "workspace": str(workspace),
                "controller_private_root": str(private),
                "online_generator_config": str(online_config_path),
                "development_evaluator_config": str(evaluator_config_path),
                "launch_packet": str(packet_path),
                "packet_sha256": packet["packet_sha256"],
                "launchable": packet["launchable"],
                "workspace_audit": workspace_report["workspace_audit"],
            }
        )

    report = {
        "schema": REPORT_SCHEMA,
        "passed": bool(
            len(rows) == 2
            and all(row["launchable"] for row in rows)
            and all(row["workspace_audit"]["passed"] for row in rows)
        ),
        "user_authorized_external_dispatch": True,
        "external_agent_count": 2,
        "round_index": int(args.round_index),
        "model": "gpt-5.6-sol",
        "reasoning_effort": "max",
        "regime": regime,
        "release_bank_report": str(Path(args.release_bank_report).resolve()),
        "base_decoder_checkpoint": str(base_checkpoint),
        "expanded_decoder_checkpoint": str(expanded_checkpoint),
        "champion_solver": str(champion),
        "champion_sha256": champion_bundle["aggregate_sha256"],
        "generation_batch_size": 16,
        "agents": rows,
    }
    _write_json(output / "campaign_preparation.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not report["passed"]:
        raise SystemExit("VPFP two-Max campaign preparation failed")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    main()
