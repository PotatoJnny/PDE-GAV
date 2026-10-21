from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


ACTIVE_POLICY_ROUTE = "continuous_reward_guidance"
DDPO_ROUTE_STATUS = "historical_reproducibility_only"
REWARD_METADATA_KEYS = (
    "pair_id",
    "condition_group_id",
    "condition_signature",
    "condition_group_index",
    "sample_index",
    "sample_index_within_group",
    "samples_per_condition",
    "sampling_seed",
    "sample_stream_id",
)


@dataclass(frozen=True)
class AgentIterationCohort:
    name: str
    cases_root: Path
    rollout_manifest: Path


def assert_agent_development_reward_disjoint(
    development_cohorts: Iterable[AgentIterationCohort],
    reward_cohorts: Iterable[AgentIterationCohort],
) -> dict[str, Any]:
    """Fail closed unless solver-development and reward cases are independently sampled."""

    development = _cohort_case_identity(list(development_cohorts), label="development")
    reward = _cohort_case_identity(list(reward_cohorts), label="reward")
    case_id_overlap = sorted(development["case_ids"] & reward["case_ids"])
    case_dir_overlap = sorted(development["case_dirs"] & reward["case_dirs"])
    seed_overlap = sorted(development["sampling_seeds"] & reward["sampling_seeds"])
    if case_id_overlap or case_dir_overlap or seed_overlap:
        raise ValueError(
            "solver-development and private reward cohorts are not independent: "
            f"case_ids={case_id_overlap[:5]}, case_dirs={case_dir_overlap[:5]}, "
            f"sampling_seeds={seed_overlap[:5]}"
        )
    return {
        "schema": "vpfp_guidance_agent_development_reward_isolation_v1",
        "passed": True,
        "development_case_count": len(development["case_ids"]),
        "reward_case_count": len(reward["case_ids"]),
        "case_ids_disjoint": True,
        "case_dirs_disjoint": True,
        "sampling_seeds_disjoint": True,
        "development_case_ids_sha256": _sha256_lines(development["case_ids"]),
        "reward_case_ids_sha256": _sha256_lines(reward["case_ids"]),
        "development_sampling_seeds": sorted(development["sampling_seeds"]),
        "reward_sampling_seeds": sorted(reward["sampling_seeds"]),
        "private_reward_cases_exposed_to_solver": False,
    }


def _cohort_case_identity(
    cohorts: list[AgentIterationCohort],
    *,
    label: str,
) -> dict[str, set[Any]]:
    if not cohorts:
        raise ValueError(f"at least one {label} cohort is required")
    case_ids: set[str] = set()
    case_dirs: set[str] = set()
    sampling_seeds: set[int] = set()
    for cohort in cohorts:
        for rollout in load_rollout_manifest(cohort.rollout_manifest):
            case_id = str(rollout.get("case_id") or "")
            if not case_id or case_id in case_ids:
                raise ValueError(f"invalid or duplicate {label} case_id: {case_id!r}")
            case_dir = str(_resolve_case_dir(rollout, cohort.rollout_manifest).resolve())
            if case_dir in case_dirs:
                raise ValueError(f"duplicate {label} case directory: {case_dir}")
            sampling_seed = rollout.get("sampling_seed")
            if not isinstance(sampling_seed, int):
                raise ValueError(f"{label} rollout has no integer sampling_seed: {case_id}")
            case_ids.add(case_id)
            case_dirs.add(case_dir)
            sampling_seeds.add(int(sampling_seed))
    return {"case_ids": case_ids, "case_dirs": case_dirs, "sampling_seeds": sampling_seeds}


def _sha256_lines(values: Iterable[Any]) -> str:
    payload = "\n".join(str(value) for value in sorted(values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def create_neutral_case_index(
    cohorts: Iterable[AgentIterationCohort],
    *,
    index_root: Path,
    private_mapping_path: Path,
    seed: int,
    purpose: str = "online_reward",
) -> dict[str, Any]:
    """Create a shuffled symlink index whose names reveal no policy cohort."""

    cohort_list = list(cohorts)
    if not cohort_list:
        raise ValueError("at least one agent-iteration cohort is required")
    allowed_purposes = {"online_reward", "solver_development", "solver_promotion", "final_solver_holdout"}
    if str(purpose) not in allowed_purposes:
        raise ValueError(f"unsupported neutral case index purpose: {purpose}")
    final_holdout = str(purpose) == "final_solver_holdout"
    if index_root.exists() and any(index_root.iterdir()):
        raise FileExistsError(f"neutral case index is not empty: {index_root}")
    index_root.mkdir(parents=True, exist_ok=True)
    private_mapping_path.parent.mkdir(parents=True, exist_ok=True)

    indexed: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    for cohort in cohort_list:
        if not final_holdout:
            _reject_holdout_path(cohort.cases_root)
            _reject_holdout_path(cohort.rollout_manifest)
        cases_root = cohort.cases_root.resolve()
        for rollout in load_rollout_manifest(cohort.rollout_manifest):
            case_id = str(rollout.get("case_id") or "")
            if not case_id:
                raise ValueError(f"rollout row has no case_id: {cohort.rollout_manifest}")
            if case_id in seen_case_ids:
                raise ValueError(f"duplicate case_id across agent cohorts: {case_id}")
            if rollout.get("status") != "ok" or not bool(rollout.get("hard_passed")):
                raise ValueError(f"agent cohort includes a rejected case: {case_id}")
            case_dir = _resolve_case_dir(rollout, cohort.rollout_manifest)
            if not case_dir.is_relative_to(cases_root):
                raise ValueError(f"case {case_id} is outside declared cohort root: {case_dir}")
            for name in ("public_question.json", "public_arrays.npz", "hidden_truth_cache.npz"):
                if not (case_dir / name).is_file():
                    raise FileNotFoundError(case_dir / name)
            indexed.append(
                {
                    "cohort": str(cohort.name),
                    "case_id": case_id,
                    "case_dir": str(case_dir),
                    **{key: rollout.get(key) for key in REWARD_METADATA_KEYS},
                }
            )
            seen_case_ids.add(case_id)

    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(indexed)).tolist()
    mapping_rows = []
    for neutral_index, source_index in enumerate(order):
        source = indexed[int(source_index)]
        neutral_id = f"case_{neutral_index:06d}"
        link = index_root / neutral_id
        link.symlink_to(Path(str(source["case_dir"])), target_is_directory=True)
        mapping_rows.append(
            {
                "neutral_id": neutral_id,
                **source,
            }
        )

    payload = {
        "schema": "vpfp_guidance_agent_private_case_index_v1",
        "seed": int(seed),
        "case_count": len(mapping_rows),
        "cohort_counts": {
            cohort.name: sum(row["cohort"] == cohort.name for row in mapping_rows)
            for cohort in cohort_list
        },
        "solver_visible_policy_labels": False,
        "mapping_is_controller_private": True,
        "purpose": str(purpose),
        "evaluation_only": final_holdout,
        "eligible_for_reward_model_training": not final_holdout,
        "rows": mapping_rows,
    }
    private_mapping_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def split_agent_reward_table(
    union_reward_table: Path,
    cohorts: Iterable[AgentIterationCohort],
    *,
    out_dir: Path,
    producer_metadata: Mapping[str, Any],
    require_all_ok: bool = True,
) -> dict[str, Any]:
    """Split one solver session's rewards back into redacted policy cohorts."""

    cohort_list = list(cohorts)
    expected: dict[str, tuple[str, dict[str, Any]]] = {}
    for cohort in cohort_list:
        for rollout in load_rollout_manifest(cohort.rollout_manifest):
            case_id = str(rollout.get("case_id") or "")
            if case_id in expected:
                raise ValueError(f"duplicate cohort case_id: {case_id}")
            expected[case_id] = (cohort.name, rollout)

    rewards: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(union_reward_table):
        case_id = str(row.get("case_id") or "")
        if case_id in rewards:
            raise ValueError(f"duplicate agent reward case_id: {case_id}")
        rewards[case_id] = row
    missing = sorted(set(expected) - set(rewards))
    unexpected = sorted(set(rewards) - set(expected))
    if missing or unexpected:
        raise ValueError(f"agent reward coverage mismatch: missing={missing}, unexpected={unexpected}")

    out_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, Any]]] = {cohort.name: [] for cohort in cohort_list}
    for case_id, (cohort_name, rollout) in expected.items():
        reward_row = rewards[case_id]
        status = str(reward_row.get("status") or "failed")
        reward = reward_row.get("reward")
        if not isinstance(reward, (int, float)) or not math.isfinite(float(reward)):
            raise ValueError(f"non-finite agent reward for {case_id}")
        if require_all_ok and status != "ok":
            raise ValueError(f"solver agent failed on {case_id}: {status}")
        enriched = {
            "case_id": case_id,
            "status": status,
            "reward": float(reward),
            **{key: rollout.get(key) for key in REWARD_METADATA_KEYS},
            **dict(producer_metadata),
            "agent_iteration_cohort": cohort_name,
        }
        grouped[cohort_name].append(enriched)

    cohort_summaries: dict[str, dict[str, Any]] = {}
    for cohort_name, rows in grouped.items():
        rows.sort(key=lambda row: str(row["case_id"]))
        path = out_dir / f"{cohort_name}_reward_table.jsonl"
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
        ok_rewards = [float(row["reward"]) for row in rows if row["status"] == "ok"]
        cohort_summaries[cohort_name] = {
            "reward_table": str(path),
            "reward_table_sha256": sha256_file(path),
            "case_count": len(rows),
            "ok_count": len(ok_rewards),
            "mean_reward": float(np.mean(ok_rewards)) if ok_rewards else None,
        }
    summary = {
        "schema": "vpfp_guidance_agent_reward_split_v1",
        "union_reward_table": str(union_reward_table.resolve()),
        "union_reward_table_sha256": sha256_file(union_reward_table),
        "producer_metadata": dict(producer_metadata),
        "cohorts": cohort_summaries,
    }
    (out_dir / "reward_split_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def diagnose_reward_signal(
    dataset_summary: Mapping[str, Any],
    *,
    min_reward_std: float = 1.0e-8,
    min_centered_reward_std: float = 1.0e-8,
) -> dict[str, Any]:
    """Choose a trainable reward objective or fail closed on numerical noise."""

    for name, value in (
        ("min_reward_std", min_reward_std),
        ("min_centered_reward_std", min_centered_reward_std),
    ):
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")

    has_train_reward = "train_reward_stats" in dataset_summary
    has_train_centered = "train_condition_centered_reward_stats" in dataset_summary
    if has_train_reward != has_train_centered:
        raise ValueError("reward dataset summary has incomplete train-split statistics")
    stats_scope = "train" if has_train_reward else "all"
    reward_key = "train_reward_stats" if has_train_reward else "reward_stats"
    centered_key = (
        "train_condition_centered_reward_stats"
        if has_train_centered
        else "condition_centered_reward_stats"
    )
    reward_stats = _finite_reward_stats(dataset_summary, reward_key)
    centered_stats = _finite_reward_stats(dataset_summary, centered_key)
    reward_std = float(reward_stats["std"])
    centered_std = float(centered_stats["std"])
    reward_trainable = reward_std >= float(min_reward_std)
    centered_trainable = centered_std >= float(min_centered_reward_std)

    if not reward_trainable:
        status = "skipped_degenerate_reward"
        objective = None
        reason = "overall agent-reward variation is below the configured numerical floor"
    elif centered_trainable:
        status = "trainable_combined"
        objective = "combined"
        reason = "absolute and within-condition reward signals both exceed their floors"
    else:
        status = "trainable_absolute_only"
        objective = "absolute"
        reason = "absolute reward varies, but within-condition centered reward is degenerate"

    normalization = dataset_summary.get("reward_normalization") or {}
    normalization_std = normalization.get("std")
    return {
        "schema": "vpfp_agent_reward_signal_diagnostics_v1",
        "status": status,
        "statistics_scope": stats_scope,
        "reason": reason,
        "trainable": reward_trainable,
        "centered_trainable": reward_trainable and centered_trainable,
        "recommended_objective": objective,
        "min_reward_std": float(min_reward_std),
        "min_centered_reward_std": float(min_centered_reward_std),
        "reward_count": int(reward_stats["count"]),
        "reward_mean": float(reward_stats["mean"]),
        "reward_std": reward_std,
        "reward_min": float(reward_stats["min"]),
        "reward_max": float(reward_stats["max"]),
        "reward_range": float(reward_stats["max"]) - float(reward_stats["min"]),
        "centered_reward_std": centered_std,
        "normalization_std": (
            float(normalization_std)
            if isinstance(normalization_std, (int, float)) and math.isfinite(float(normalization_std))
            else None
        ),
    }


def validate_agent_loop_artifacts(
    agent_out: Path,
    *,
    context_mode: str,
    require_actual_codex: bool,
    expected_cases: int | None = None,
    expected_development_cases: int | None = None,
    expected_reward_cases: int | None = None,
    require_disjoint_evaluation: bool = False,
    expected_public_previous_solver_sha256: str | None = None,
) -> dict[str, Any]:
    if expected_cases is not None:
        if expected_development_cases is None:
            expected_development_cases = int(expected_cases)
        if expected_reward_cases is None:
            expected_reward_cases = int(expected_cases)
    if expected_development_cases is None or expected_reward_cases is None:
        raise ValueError("expected development and reward case counts are required")
    expected_development_cases = int(expected_development_cases)
    expected_reward_cases = int(expected_reward_cases)
    if expected_development_cases <= 0 or expected_reward_cases <= 0:
        raise ValueError("expected development and reward case counts must be positive")

    round_root = agent_out / "rounds" / "round_000"
    final = _read_json(agent_out / "final_summary.json")
    isolation = _read_json(round_root / "solver_context_isolation.json")
    reward_table = round_root / "reward_table.jsonl"
    rewards = read_jsonl(reward_table)
    if int(final.get("rounds_completed", 0)) != 1:
        raise ValueError("solver agent did not complete exactly one controller round")
    if str(final.get("agent_context_mode")) != str(context_mode):
        raise ValueError("solver agent context mode does not match the requested mode")
    final_development_cases = int(final.get("cases_per_round", expected_development_cases))
    final_reward_cases = int(final.get("reward_cases_per_round", final_development_cases))
    if final_development_cases != expected_development_cases:
        raise ValueError(
            f"expected {expected_development_cases} solver-development cases, "
            f"agent reported {final_development_cases}"
        )
    if final_reward_cases != expected_reward_cases:
        raise ValueError(
            f"expected {expected_reward_cases} private reward cases, "
            f"agent reported {final_reward_cases}"
        )
    if len(rewards) != expected_reward_cases:
        raise ValueError(
            f"expected {expected_reward_cases} agent rewards, found {len(rewards)}"
        )
    if not bool(isolation.get("passed")):
        raise ValueError("solver-visible workspace isolation audit failed")
    if bool(isolation.get("previous_submission_exposed")) or bool(isolation.get("verifier_feedback_exposed")):
        raise ValueError("solver agent received forbidden cross-round private context")
    public_previous_exposed = bool(isolation.get("public_previous_solver_exposed"))
    if expected_public_previous_solver_sha256 is None:
        if public_previous_exposed:
            raise ValueError("solver agent unexpectedly received a public prior solver")
    else:
        if not public_previous_exposed:
            raise ValueError("solver refinement did not expose the declared public prior solver")
        observed_previous_sha = str(isolation.get("public_previous_solver_sha256") or "")

    workspace_manifest = round_root / "solver_workspace" / "case_manifest.json"
    if workspace_manifest.is_file():
        visible_cases = _read_json(workspace_manifest).get("cases", [])
        if len(visible_cases) != expected_development_cases:
            raise ValueError(
                f"solver workspace has {len(visible_cases)} cases; "
                f"expected {expected_development_cases}"
            )
    elif require_disjoint_evaluation:
        raise FileNotFoundError(workspace_manifest)

    if require_disjoint_evaluation:
        if not bool(final.get("separate_reward_evaluation")):
            raise ValueError("solver agent did not use a separate private reward cohort")
        if not bool(final.get("development_evaluation_disjoint")):
            raise ValueError("solver agent final summary reports development/reward overlap")
        if not bool(isolation.get("development_evaluation_disjoint")):
            raise ValueError("solver isolation report does not prove development/reward separation")
        if int(isolation.get("development_evaluation_overlap_count", -1)) != 0:
            raise ValueError("solver isolation report contains development/reward overlap")
        if bool(isolation.get("private_reward_cases_exposed_to_solver")):
            raise ValueError("private reward cases were exposed to the solver agent")
        if int(isolation.get("solver_visible_case_count", -1)) != expected_development_cases:
            raise ValueError("solver-visible development case count does not match")
        if int(isolation.get("private_reward_case_count", -1)) != expected_reward_cases:
            raise ValueError("private reward case count does not match")

    invocation_path = round_root / "codex_invocation.json"
    invocation = _read_json(invocation_path) if invocation_path.is_file() else None
    if require_actual_codex:
        if invocation is None:
            raise ValueError("missing actual Codex invocation record")
        if not bool(invocation.get("actual_codex_invoked")):
            raise ValueError("solver round did not invoke actual Codex")
        if int(invocation.get("returncode", -1)) != 0 or bool(invocation.get("timed_out")):
            raise ValueError("actual Codex invocation did not exit cleanly")

    candidate_submission = round_root / "solver_workspace" / "submission.py"
    if not candidate_submission.is_file() or candidate_submission.is_symlink():
        raise FileNotFoundError(f"missing regular solver artifact: {candidate_submission}")
    selected_submission = round_root / "selected_solver" / "submission.py"
    if not selected_submission.is_file():
        selected_submission = candidate_submission
    if selected_submission.is_symlink():
        raise FileNotFoundError(f"selected solver artifact is a symlink: {selected_submission}")
    improvement_path = round_root / "solver_improvement_report.json"
    solver_improvement = (
        _read_json(improvement_path) if improvement_path.is_file() else None
    )
    if expected_public_previous_solver_sha256 is not None and solver_improvement is None:
        raise FileNotFoundError("solver refinement is missing its improvement report")
    return {
        "agent_final_summary": final,
        "isolation": isolation,
        "invocation": invocation,
        "reward_table": str(reward_table),
        "reward_table_sha256": sha256_file(reward_table),
        "solver_artifact": str(selected_submission),
        "solver_artifact_sha256": sha256_file(selected_submission),
        "candidate_solver_artifact": str(candidate_submission),
        "candidate_solver_artifact_sha256": sha256_file(candidate_submission),
        "solver_improvement": solver_improvement,
        "solver_candidate_promoted": (
            None
            if solver_improvement is None
            else bool(solver_improvement.get("candidate_promoted"))
        ),
        "actual_codex_required": bool(require_actual_codex),
        "solver_development_case_count": expected_development_cases,
        "private_reward_case_count": expected_reward_cases,
        "development_reward_isolation": {
            "required": bool(require_disjoint_evaluation),
            "passed": bool(
                not require_disjoint_evaluation
                or not isolation.get("private_reward_cases_exposed_to_solver")
            ),
        },
    }


def artifact_snapshot(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    return {
        str(name): {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for name, path in paths.items()
    }


def assert_artifact_snapshot_unchanged(
    before: Mapping[str, Mapping[str, Any]],
    paths: Mapping[str, Path],
) -> dict[str, dict[str, Any]]:
    after = artifact_snapshot(paths)
    changes = {
        name: {"before": before.get(name), "after": metadata}
        for name, metadata in after.items()
        if before.get(name) != metadata
    }
    if changes:
        raise ValueError(f"frozen VPFP artifacts changed during agent iteration: {changes}")
    return after


def load_rollout_manifest(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"empty rollout manifest: {path}")
    return rows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_case_dir(row: Mapping[str, Any], manifest_path: Path) -> Path:
    raw = str(row.get("case_dir") or "")
    if not raw:
        raise ValueError(f"rollout row has no case_dir: {manifest_path}")
    value = Path(raw)
    if not value.is_absolute():
        value = manifest_path.parent / value
    return value.resolve()


def _reject_holdout_path(path: Path) -> None:
    if "final_holdout" in str(path).lower():
        raise ValueError(f"final holdout is forbidden in agent-iteration training: {path}")


def _finite_reward_stats(summary: Mapping[str, Any], key: str) -> dict[str, float | int]:
    raw = summary.get(key)
    if not isinstance(raw, Mapping):
        raise ValueError(f"reward dataset summary has no {key}")
    count = raw.get("count")
    if not isinstance(count, int) or count <= 0:
        raise ValueError(f"{key}.count must be a positive integer")
    values: dict[str, float | int] = {"count": count}
    for field in ("mean", "std", "min", "max"):
        value = raw.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{key}.{field} must be finite")
        values[field] = float(value)
    if float(values["std"]) < 0.0 or float(values["max"]) < float(values["min"]):
        raise ValueError(f"{key} contains inconsistent statistics")
    return values


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
