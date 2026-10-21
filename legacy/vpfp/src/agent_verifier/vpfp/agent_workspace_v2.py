"""Public workspace contract for long-time robust VPFP solver Agents."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .agent_loop import ORACLE_ASSISTED_CONTEXT_MODE, export_public_solver_workspace
from .continuous_macros import MACRO_OUTPUT_CONTRACT
from .io import assert_public_bundle_isolated
from .oracle_training import (
    assert_released_truth_bank,
    export_released_truth_bank,
    inspect_solver_bundle,
    stage_solver_bundle,
)
from .runtime_paths import is_persistent_run_path


VPFP_ROBUST_WORKSPACE_SCHEMA = "vpfp_robust_agent_workspace_v2"
VPFP_FORMAL_EVALUATION_SCHEMA = "vpfp_robust_formal_evaluation_contract_v2"
VPFP_EXTERNAL_AUTHORIZATION_SCHEMA = "vpfp_external_agent_authorization_v2"
VPFP_ROBUST_EXTERNAL_LAUNCH_SCHEMA = "vpfp_robust_external_agent_launch_v2"


@dataclass(frozen=True)
class VPFPRobustAgentWorkspaceConfig:
    agent_id: str
    round_index: int
    regime_id: str
    collision_nu_range: tuple[float, float]
    temperature_range: tuple[float, float]
    time_domain: tuple[float, float] = (0.0, 1.0)
    model: str = "gpt-5.6-sol"
    reasoning_effort: str = "max"
    released_completed_solution_limit: int = 32
    active_wall_time_seconds: int = 24 * 60 * 60
    scratch_bytes: int = 200 * 1024**3
    development_evaluator_case_credits: int = 4096
    development_evidence_case_counts: tuple[int, ...] = (4, 16, 64)
    development_default_evidence_case_count: int = 16
    development_request_lease_seconds: int = 2 * 60 * 60
    generator_max_batch: int = 16
    final_solver_soft_wall_seconds_per_case: float = 900.0
    final_solver_hard_wall_seconds_per_case: float = 1800.0

    def validate(self) -> None:
        if self.agent_id not in {"vpfp_max_a", "vpfp_max_b"}:
            raise ValueError("VPFP v2 uses exactly the two registered Max Agent IDs")
        if self.round_index < 0 or not self.regime_id:
            raise ValueError("VPFP Agent round identity is invalid")
        if self.model != "gpt-5.6-sol" or self.reasoning_effort != "max":
            raise ValueError("VPFP v2 requires gpt-5.6-sol Max")
        nu0, nu1 = (float(value) for value in self.collision_nu_range)
        theta0, theta1 = (float(value) for value in self.temperature_range)
        t0, t1 = (float(value) for value in self.time_domain)
        if not (0.0 <= nu0 < nu1 and 0.0 < theta0 < theta1 and t0 < t1):
            raise ValueError("VPFP physical regime intervals are invalid")
        if not 1 <= self.released_completed_solution_limit <= 32:
            raise ValueError("released completed-answer count must lie in [1,32]")
        evidence = tuple(int(value) for value in self.development_evidence_case_counts)
        if (
            not evidence
            or evidence != tuple(sorted(set(evidence)))
            or evidence[0] < 1
            or self.development_default_evidence_case_count not in evidence
            or evidence[-1] > self.development_evaluator_case_credits
        ):
            raise ValueError("development evidence cascade is invalid")
        if min(
            self.active_wall_time_seconds,
            self.scratch_bytes,
            self.development_evaluator_case_credits,
            self.generator_max_batch,
        ) < 1:
            raise ValueError("VPFP Agent resource limits must be positive")
        if not 60 <= self.development_request_lease_seconds <= 24 * 60 * 60:
            raise ValueError("VPFP evaluator lease is outside the safe range")
        if not (
            0.0
            < self.final_solver_soft_wall_seconds_per_case
            <= self.final_solver_hard_wall_seconds_per_case
        ):
            raise ValueError("final solver soft/hard wall limits are invalid")


def build_vpfp_robust_agent_workspace(
    *,
    output_dir: str | Path,
    controller_private_root: str | Path,
    config: VPFPRobustAgentWorkspaceConfig,
    development_case_dirs: Sequence[str | Path],
    released_completed_case_dirs: Sequence[str | Path],
    previous_solver_path: str | Path,
    cluster_notes_path: str | Path,
    historical_solver_paths: Mapping[str, str | Path] | None = None,
    prior_feedback_path: str | Path | None = None,
) -> dict[str, Any]:
    config.validate()
    root = Path(output_dir).resolve()
    private = Path(controller_private_root).resolve()
    if not is_persistent_run_path(root) or not is_persistent_run_path(private):
        raise ValueError(
            "VPFP Agent workspace and private state need configured run storage"
        )
    if root == private or root.is_relative_to(private) or private.is_relative_to(root):
        raise ValueError("VPFP Agent workspace and private state must be disjoint")
    if root.exists() or private.exists():
        raise FileExistsError("VPFP Agent workspace/private roots must be fresh")
    if not development_case_dirs:
        raise ValueError("VPFP Agent workspace needs public development cases")
    if not 1 <= len(released_completed_case_dirs) <= config.released_completed_solution_limit:
        raise ValueError("VPFP released completed-answer count violates authorization")

    previous = Path(previous_solver_path).resolve()
    previous_bundle = inspect_solver_bundle(previous)
    root.parent.mkdir(parents=True, exist_ok=True)
    workspace = export_public_solver_workspace(
        list(development_case_dirs),
        root,
        context_mode=ORACLE_ASSISTED_CONTEXT_MODE,
    )
    private.mkdir(parents=True)
    truth_audit = export_released_truth_bank(
        list(released_completed_case_dirs),
        workspace,
        private,
    )

    prior_root = workspace / "prior_solvers"
    champion = stage_solver_bundle(previous, prior_root / "champion")
    historical_rows = [
        {
            "solver_id": "champion",
            "entrypoint": "prior_solvers/champion/submission.py",
            "aggregate_sha256": previous_bundle["aggregate_sha256"],
        }
    ]
    for solver_id, source in sorted((historical_solver_paths or {}).items()):
        if solver_id == "champion":
            continue
        staged = stage_solver_bundle(
            Path(source).resolve(), prior_root / str(solver_id)
        )
        bundle = inspect_solver_bundle(staged)
        historical_rows.append(
            {
                "solver_id": str(solver_id),
                "entrypoint": f"prior_solvers/{solver_id}/submission.py",
                "aggregate_sha256": bundle["aggregate_sha256"],
            }
        )

    work = workspace / "work"
    work.mkdir()
    stage_solver_bundle(previous, work / "starter")
    submission_root = workspace / "submission"
    submission_root.mkdir()
    shutil.copytree(work / "starter", submission_root, dirs_exist_ok=True)
    inspect_solver_bundle(submission_root / "submission.py")
    for name in (
        "requests",
        "responses",
        "candidates",
        "claims",
        "claim_failures",
    ):
        (workspace / "evaluation_exchange" / name).mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[3]
    clients = {
        "online_generator.py": (
            repo_root / "src" / "agent_verifier" / "vpfp" / "public_online_generator_cli.py"
        ),
        "bundle_client.py": repo_root / "scripts" / "vpfp_solver_bundle_client.py",
        "vpfp_solver_bundle_client.py": (
            repo_root / "scripts" / "vpfp_solver_bundle_client.py"
        ),
        "evaluation_client.py": (
            repo_root / "scripts" / "vpfp_development_evaluation_client.py"
        ),
        "solver_instruction_preflight.sh": (
            repo_root / "scripts" / "vpfp_instruction_only_solver_preflight.sh"
        ),
    }
    for destination_name, source in clients.items():
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(source)
        destination = workspace / destination_name
        shutil.copy2(source, destination)
        destination.chmod(0o750)
    notes = Path(cluster_notes_path).resolve()
    if not notes.is_file() or notes.is_symlink():
        raise FileNotFoundError(notes)
    shutil.copy2(notes, workspace / "slurm_cluster_notes_for_codex.md")

    authorization = {
        "schema": VPFP_EXTERNAL_AUTHORIZATION_SCHEMA,
        "authorized": True,
        "agent_count": 2,
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "allowed": [
            "generated VPFP public numerical questions",
            "public numerical generator and equation oracle",
            "completed development answers and exact macro moments",
            "up to 32 explicitly released completed hidden solutions",
            "generated VPFP data and numerical generative-model interface",
            "base, guided, and mixed numerical samples returned by that interface",
            "aggregate historical reward and sanitized refinement summaries",
        ],
        "forbidden": [
            "hidden neural or procedural parameterization",
            "latent variables and projection coefficients",
            "private selection, promotion, or final holdout cases",
        ],
    }
    resource_budget = {
        "active_wall_time_seconds": config.active_wall_time_seconds,
        "scratch_bytes": config.scratch_bytes,
        "allocation": {"gpus": 1, "cpus": 16, "memory_bytes": 96 * 1024**3},
        "final_solver": {
            "soft_wall_time_seconds_per_case": (
                config.final_solver_soft_wall_seconds_per_case
            ),
            "hard_wall_time_seconds_per_case": (
                config.final_solver_hard_wall_seconds_per_case
            ),
        },
    }
    generator_interface = {
        "schema": "vpfp_long_time_online_generator_interface_v2",
        "transport": "controller_owned_unix_socket_via_public_client",
        "client": "online_generator.py",
        "operations": [
            "status",
            "sample",
            "materialize",
            "materialize_batch",
            "query",
            "query_batch",
        ],
        "sample_modes": ["base", "guided", "mixed"],
        "point_query_fields": ["source", "rho0", "g0", "truth", "macro_truth"],
        "maximum_cases_per_batch": config.generator_max_batch,
        "aggregate_query_counts": "accounting_only",
        "resource_budget": resource_budget,
        "returns_numerical_questions_and_optional_completed_answers": True,
        "neural_or_procedural_parameterization_visible": False,
        "latents_visible": False,
        "private_evaluation_cases_returned": False,
        "numerical_view_lifecycle": "job_scoped_ephemeral_regenerable",
        "durable_case_records_are_compact_descriptors": True,
        "runtime_cache_retention_guaranteed": False,
        "disk_use_is_leaderboard_objective": False,
    }
    development_interface = {
        "schema": "vpfp_development_evaluation_interface_v2",
        "transport": "controller_watched_filesystem_queue",
        "client": "evaluation_client.py",
        "bundle_client": "bundle_client.py",
        "asynchronous_submission_supported": True,
        "recommended_inflight_screen_batch": 8,
        "cascade": "fresh_paired_screen_confirm_v2",
        "evidence_case_counts": list(config.development_evidence_case_counts),
        "default_evidence_case_count": config.development_default_evidence_case_count,
        "agent_may_request_evidence_depth": True,
        "controller_controls_stage_admission": True,
        "same_hash_evidence_is_cumulative": True,
        "same_hash_fresh_reroll_forbidden": True,
        "case_credit_limit": config.development_evaluator_case_credits,
        "submission_count_limit": None,
        "same_fresh_cases_for_candidate_and_comparison": True,
        "candidate_bundles_archived_immutably": True,
        "queue_claim_protocol": "append_only_lease_v1",
        "request_lease_seconds": config.development_request_lease_seconds,
        "interrupted_requests_replayed_from_immutable_archive": True,
        "exact_final_bundle_compatibility_required": True,
        "promotion_requires_larger_independent_fresh_cohort": True,
        "official_promotion_result_returned": False,
        "private_case_ids_returned": False,
        "per_case_errors_returned": False,
        "hidden_answers_returned": False,
        "generator_parameters_returned": False,
        "predictions_and_evaluator_materializations_are_ephemeral": True,
        "durable_feedback_is_reduced_metrics_only": True,
    }
    formal = _formal_evaluation_contract(config)
    manifest = {
        "schema": VPFP_ROBUST_WORKSPACE_SCHEMA,
        "agent_id": config.agent_id,
        "round_index": config.round_index,
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "network_allowed": True,
        "regime": {
            "regime_id": config.regime_id,
            "collision_nu_range": list(config.collision_nu_range),
            "temperature_range": list(config.temperature_range),
            "time_domain": list(config.time_domain),
        },
        "solver_output_contract_version": MACRO_OUTPUT_CONTRACT,
        "previous_solver_sha256": previous_bundle["aggregate_sha256"],
        "historical_solvers": historical_rows,
        "public_development_case_count": len(development_case_dirs),
        "released_completed_solution_count": len(released_completed_case_dirs),
        "released_completed_solution_limit": config.released_completed_solution_limit,
        "generator_interface": generator_interface,
        "development_evaluation_interface": development_interface,
        "formal_evaluation_contract": "formal_evaluation_contract.json",
        "resource_budget": resource_budget,
        "external_authorization": authorization,
        "hidden_primary_object": "continuous rho(t,x,y) and normalized g(t,x,y,v)",
        "grid_arrays_are_numerical_access_views_not_primary_truth": True,
        "answer_first_source_derivation_preserved": True,
        "coarse_equation_first_solve_is_shape_seed_only": True,
        "generator_parameterization_visible": False,
        "private_cohorts_visible": False,
        "isolation_mechanism": "instructions_and_workspace_allowlist",
        "storage_lifecycle": {
            "schema": "agent_verifier_storage_lifecycle_v1",
            "dense_controller_runtime": "SLURM_TMPDIR",
            "online_numerical_views": "job_scoped_ephemeral",
            "durable_case_state": "compact_descriptors_and_hashes",
            "preserved_outputs": [
                "solver_bundle",
                "agent_process_logs",
                "leaderboard_rows",
                "compact_service_reports",
            ],
            "disk_use_is_leaderboard_objective": False,
        },
    }
    _write_json(workspace / "generator_interface.json", generator_interface)
    _write_json(
        workspace / "development_evaluation_interface.json",
        development_interface,
    )
    _write_json(workspace / "formal_evaluation_contract.json", formal)
    _write_json(workspace / "external_authorization.json", authorization)
    _write_json(workspace / "workspace_manifest.json", manifest)
    (workspace / "solver_contract.md").write_text(
        _solver_contract(), encoding="utf-8"
    )
    (workspace / "solver_agent_instructions.md").write_text(
        _agent_instructions(config), encoding="utf-8"
    )
    (workspace / "prompt.md").write_text(
        "Develop the strongest compute-budgeted numerical method for the public "
        "long-time VPFP macro-output task. Read solver_agent_instructions.md, "
        "solver_contract.md, formal_evaluation_contract.json, and the Slurm notes. "
        "Use numerical generator/oracle access, released completed answers, and "
        "asynchronous fresh paired evaluations. Place the final immutable bundle "
        "under submission/.\n",
        encoding="utf-8",
    )
    feedback_row = None
    if prior_feedback_path is not None:
        feedback = Path(prior_feedback_path).resolve()
        if not feedback.is_file() or feedback.is_symlink():
            raise FileNotFoundError(feedback)
        destination = workspace / "prior_round_feedback.json"
        shutil.copy2(feedback, destination)
        feedback_row = {"path": destination.name, "sha256": _sha256(destination)}

    audit = audit_vpfp_robust_agent_workspace(workspace)
    if not audit["passed"]:
        raise ValueError(f"VPFP robust Agent workspace audit failed: {audit['failures']}")
    return {
        "workspace": str(workspace),
        "controller_private_root": str(private),
        "manifest": manifest,
        "formal_evaluation_contract": formal,
        "external_authorization": authorization,
        "released_truth_audit": truth_audit,
        "workspace_audit": audit,
        "prior_feedback": feedback_row,
        "starter_submission": str(work / "starter" / "submission.py"),
        "champion_submission": str(champion),
    }


def audit_vpfp_robust_agent_workspace(path: str | Path) -> dict[str, Any]:
    root = Path(path).resolve()
    failures: list[str] = []
    if not root.is_dir() or root.is_symlink():
        return {"passed": False, "failures": ["workspace_missing"]}
    try:
        manifest = json.loads(
            (root / "workspace_manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"passed": False, "failures": [f"invalid_manifest:{type(exc).__name__}"]}
    if manifest.get("schema") != VPFP_ROBUST_WORKSPACE_SCHEMA:
        failures.append("wrong_workspace_schema")
    authorization = dict(manifest.get("external_authorization") or {})
    if not (
        authorization.get("authorized") is True
        and authorization.get("agent_count") == 2
        and authorization.get("model") == "gpt-5.6-sol"
        and authorization.get("reasoning_effort") == "max"
    ):
        failures.append("external_authorization_binding_failed")
    for item in root.rglob("*"):
        if item.is_symlink():
            failures.append(f"symlink_forbidden:{item.relative_to(root)}")
    for row in json.loads((root / "case_manifest.json").read_text())["cases"]:
        try:
            assert_public_bundle_isolated(root / str(row["case_dir"]))
        except Exception as exc:  # noqa: BLE001 - retained in aggregate audit.
            failures.append(f"public_case_invalid:{type(exc).__name__}")
    try:
        truth = assert_released_truth_bank(root / "released_truth")
    except Exception as exc:  # noqa: BLE001 - retained in aggregate audit.
        failures.append(f"released_truth_invalid:{type(exc).__name__}")
        truth = {"passed": False, "case_count": 0}
    for relative in (
        "online_generator.py",
        "bundle_client.py",
        "vpfp_solver_bundle_client.py",
        "evaluation_client.py",
        "solver_agent_instructions.md",
        "solver_contract.md",
        "formal_evaluation_contract.json",
        "slurm_cluster_notes_for_codex.md",
        "solver_instruction_preflight.sh",
    ):
        item = root / relative
        if not item.is_file() or item.is_symlink():
            failures.append(f"missing_public_interface:{relative}")
    for relative in (
        "online_generator.py",
        "bundle_client.py",
        "vpfp_solver_bundle_client.py",
        "evaluation_client.py",
        "solver_instruction_preflight.sh",
    ):
        item = root / relative
        if item.is_file() and item.stat().st_mode & 0o111 == 0:
            failures.append(f"nonexecutable_public_client:{relative}")
    try:
        champion = inspect_solver_bundle(root / "prior_solvers/champion/submission.py")
    except Exception as exc:  # noqa: BLE001 - retained in aggregate audit.
        failures.append(f"champion_bundle_invalid:{type(exc).__name__}")
        champion = None
    leaked_names = [
        str(item.relative_to(root))
        for item in root.rglob("*")
        if item.is_file()
        and item.name in {"hidden_truth_cache.npz", "decoder.pt", "latent.npy"}
    ]
    if leaked_names:
        failures.append(f"private_parameterization_files:{sorted(leaked_names)}")
    return {
        "schema": "vpfp_robust_agent_workspace_audit_v2",
        "passed": not failures,
        "failures": failures,
        "public_case_count": len(
            json.loads((root / "case_manifest.json").read_text())["cases"]
        ),
        "released_completed_solution_count": int(truth.get("case_count", 0)),
        "champion_bundle": champion,
        "hidden_parameterization_visible": False,
    }


def _formal_evaluation_contract(
    config: VPFPRobustAgentWorkspaceConfig,
) -> dict[str, Any]:
    return {
        "schema": VPFP_FORMAL_EVALUATION_SCHEMA,
        "objective": "efficient numerical approximation of requested VPFP macro observables",
        "required_entrypoint": "solve_with_budget(public_case, query_times, budget)",
        "required_outputs": ["rho", "momentum", "kinetic_energy"],
        "query_times": "arbitrary finite times in the public interval",
        "candidate_controls_internal_space_velocity_and_time_discretization": True,
        "compute_budgets": [
            {"tier": 0, "wall_time_seconds": 30.0},
            {"tier": 1, "wall_time_seconds": 180.0},
            {"tier": 2, "wall_time_seconds": 900.0},
        ],
        "hard_wall_time_seconds_per_case": config.final_solver_hard_wall_seconds_per_case,
        "primary_answer_first_cohorts": ["base", "structure_preserving_perturbed"],
        "both_answer_first_cohorts_are_required": True,
        "equation_first_self_convergence": {
            "role": "late auxiliary tie_break_only",
            "candidate_self_differences_only": True,
            "reference_error_used": False,
            "strict_monotonicity_required": False,
            "constant_resolution_independent_output_gets_credit": False,
        },
        "development_metrics": [
            "worst_answer_first_p90",
            "worst_time_band_p90",
            "high_budget_mean",
            "high_budget_cvar90",
            "paired_champion_difference",
            "compute_frontier_geometric_mean",
            "runtime_and_memory",
        ],
        "development_leaderboard_is_official_promotion": False,
        "official_promotion_requires_fresh_independent_paired_cohort": True,
        "no_error_monotonicity_gate": True,
        "grid_arrays_are_samples_not_primary_truth": True,
    }


def _solver_contract() -> str:
    return """# VPFP Solver Contract

Implement `solve_with_budget(public_case, query_times, budget)` in
`submission/submission.py`. Return finite NumPy-compatible arrays:

- `rho`: `[len(query_times), nx, ny]`
- `momentum`: `[len(query_times), nx, ny, 2]`
- `kinetic_energy`: `[len(query_times), nx, ny]`

`query_times` may be any finite points in the public time interval. Your method
owns its internal time, space, and velocity discretization and interpolation.
It must use the actual public case grids and honor the declared compute budget.
The full kinetic equation remains the governing equation; only the required
outputs are macroscopic. A compatibility `solve(public_case)` wrapper is useful
but does not replace `solve_with_budget`.
"""


def _agent_instructions(config: VPFPRobustAgentWorkspaceConfig) -> str:
    return f"""# Solver Agent Instructions

Develop the strongest efficient numerical approximation method for the public
VPFP task in regime `{config.regime_id}`. The objective is numerical quality
under compute budgets, not adherence to any named reduced-model family.

The hidden truth is a continuous answer-first function. Public grids are
numerical access views. The source is derived from that hidden function by the
declared VPFP operator. You may use the numerical online generator/oracle,
including batched requests and explicitly released completed answers. You may
not inspect model weights, procedural coefficients, latent variables,
controller-private files, or private evaluation cases. Do not infer permission
from filesystem visibility; follow the documented public interfaces.

Use `online_generator.py` for fresh base/guided samples, arbitrary-grid views,
and numerical source/initial/truth queries. Use `evaluation_client.py` for
fresh paired development feedback. Identical bundle hashes reuse cumulative
evidence and do not draw fresh cases. The development leaderboard is not the
official final evaluation.

You have an active development allocation rather than a small fixed query
quota. Use batch queries to avoid transport overhead. Your final solver may run
for minutes per case within the formal contract. Network access is allowed for
reasonable literature and software documentation searches.

Use the available development window for measured exploration. Preserve a
runnable incumbent while testing materially different numerical and learning
routes when local tuning saturates. This is an objective-level requirement, not
method guidance: select hypotheses from the numerical data and feedback.
If `prior_round_feedback.json` is present, read it first. It is compact recovery
context from an interrupted or completed earlier run, not fresh evaluation
evidence; independently validate any carried-forward hypothesis.

All substantial computation must run on Slurm compute nodes using the declared
venv. Never run heavy CPU or GPU work on a login node. Read
`slurm_cluster_notes_for_codex.md` before launching jobs. Keep large artifacts
under `/scratch`; keep only source code and compact summaries in the project.
Use disk reasonably, but keep your attention on the numerical method. Prefer
reproducible code and compact learned artifacts over retaining every generated
numerical case. Runtime oracle views and caches are job-scoped and may be
removed after development; the controller preserves your submitted solver and
process logs. Disk use is not a leaderboard objective.


Work under `work/`. Place the final immutable bundle under `submission/` and
bind all runtime assets with `bundle_client.py`. Preserve failures and exact
reasons; do not silently catch broad exceptions.
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
