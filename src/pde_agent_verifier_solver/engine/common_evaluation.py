"""Offline common testing of frozen programs through the existing HiddenVerifier.

This utility exports evaluations; scientific plans and scores remain Task-owned.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from enum import Enum
import csv
import hashlib
import json
import logging
import os
import math
from pathlib import Path
from statistics import fmean
from threading import Event, Lock
import traceback
import time
from typing import Mapping, Sequence

from pde_agent_verifier_solver.sdk import AnswerView, Artifact, CaseRecord, EvaluationPhase, EvaluationPlan
from .evaluator import HiddenVerifier
from .candidate_execution import (
    CandidateExecutionCancelled, CandidateExecutionInfrastructureError, CandidateExecutionTimeout,
)
from .checkpoint_store import atomic_bytes, atomic_json

_FAILURE_LOG_LOCK = Lock()


def _is_infrastructure_failure(error: BaseException) -> bool:
    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, CandidateExecutionInfrastructureError):
            return True
        pending.extend(item for item in (current.__cause__, current.__context__) if item is not None)
        for attribute in ("_candidate_cleanup_errors", "_pde_session_cleanup_errors"):
            pending.extend(getattr(current, attribute, ()))
    return False


def record_common_evaluation_failure(
    output_directory: Path, error: BaseException, *, stage: str, **context,
) -> None:
    """Persist trusted exception details without adding them to result rows."""
    diagnostic = {
        "stage": stage, **context, "exception_type": type(error).__name__,
        "exception_message": str(error),
        "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
        "attached_cleanup_errors": [
            {"exception_type": type(related).__name__, "exception_message": str(related),
             "traceback": "".join(traceback.format_exception(type(related), related, related.__traceback__))}
            for attribute in ("_candidate_cleanup_errors", "_pde_session_cleanup_errors")
            for related in getattr(error, attribute, ())
        ],
    }
    logging.getLogger(__name__).error(
        "Common evaluation failure: stage=%s context=%s", stage, context,
        exc_info=(type(error), error, error.__traceback__),
    )
    private_root = output_directory / "controller_private"
    with _FAILURE_LOG_LOCK:
        private_root.mkdir(exist_ok=True, mode=0o700)
        descriptor = os.open(
            private_root / "common_evaluation_failures.jsonl",
            os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600,
        )
        with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(diagnostic, sort_keys=True) + "\n")
            handle.flush()


CASE_COLUMNS = (
    "task_id", "method", "run_id", "seed", "artifact_id", "artifact_sha256",
    "cohort", "dataset_id", "case_id", "parent_case_id", "recording_id",
    "metric", "error", "status", "error_code", "solve_wall_seconds",
    "query_wall_seconds", "solve_queue_wait_seconds", "query_queue_wait_seconds",
    "case_wall_seconds", "plan_sha256", "reference_sha256", "public_case_sha256",
)


def json_value(value):
    """Stable JSON representation of SDK values and numerical arrays."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return {key: json_value(getattr(value, key)) for key in value.__dataclass_fields__}
    if hasattr(value, "tolist"):
        return json_value(value.tolist())
    return value


def content_sha256(value) -> str:
    encoded = json.dumps(json_value(value), sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value) -> None:
    atomic_json(path, json_value(value))


@dataclass(frozen=True)
class CommonProgram:
    method: str
    run_id: str
    seed: int
    artifact_id: str
    artifact: Artifact | None
    artifact_sha256: str | None
    error_code: str | None = None


@dataclass(frozen=True)
class CommonCase:
    cohort: str
    dataset_id: str
    record: CaseRecord
    plan: EvaluationPlan
    recording_id: str | None = None


@dataclass(frozen=True)
class _FrozenTruth:
    view: AnswerView | None
    error: Exception | None

    def answer(self, query):
        del query
        if self.error is not None:
            raise self.error
        return self.view


class _FixedPlanEvaluation:
    def __init__(self, evaluation, plans):
        self._evaluation = evaluation
        self._plans = plans

    def __getattr__(self, name):
        return getattr(self._evaluation, name)

    def make_plan(self, case, phase, settings, seed, request=None):
        del settings, seed, request
        plan = self._plans[case.case_id]
        if plan.phase is not phase:
            raise ValueError("common plan phase differs from HiddenVerifier phase")
        return plan



class _CommonCheckpoint:
    """Atomic per-program/Case rows; only a fully closed segment is resumable."""
    def __init__(self, root, identity, *, resume):
        self.root = root
        self.path = root / "checkpoint.json"
        self.identity = identity
        self.references = {}
        self.rows = {}
        self.previous_wall = 0.0
        if resume:
            if self.path.is_symlink():
                raise ValueError("common checkpoint cannot be a symlink")
            saved = json.loads(self.path.read_text())
            if saved.get("schema") != "common_evaluation_checkpoint_v1" or saved.get("identity") != identity:
                raise ValueError("common checkpoint programs, cases, plans, references or settings differ")
            if saved.get("status") not in {"paused", "completed"}:
                raise ValueError("only safely paused or completed common evaluation can resume")
            self.previous_wall = float(saved["elapsed_wall_seconds"])
            if not math.isfinite(self.previous_wall) or self.previous_wall < 0:
                raise ValueError("invalid common checkpoint elapsed time")
            self.references = saved["completed"]
            for index, reference in self.references.items():
                if not index.isdigit() or str(int(index)) != index or int(index) >= identity["n_program_cases"]:
                    raise ValueError("invalid completed common program/Case index")
                relative = Path(reference["path"])
                expected = Path("completed_cases") / f"{int(index):08d}.json"
                path = root / relative
                if relative != expected or path.is_symlink():
                    raise ValueError("invalid common checkpoint row path")
                data = path.read_bytes()
                rows = json.loads(data)
                if not isinstance(rows, list) or [row.get("metric") for row in rows] != identity["metrics"]:
                    raise ValueError("common checkpoint has incomplete metric rows")
                self.rows[int(index)] = rows
        elif self.path.exists():
            raise FileExistsError(self.path)
        (root / "completed_cases").mkdir(exist_ok=True, mode=0o700)

    def add(self, index, rows):
        if index in self.rows:
            raise ValueError("common program/Case completed twice")
        relative = Path("completed_cases") / f"{index:08d}.json"
        data = (json.dumps(rows, sort_keys=True, allow_nan=False) + "\n").encode()
        atomic_bytes(self.root / relative, data)
        self.references[str(index)] = {"path": str(relative), "sha256": hashlib.sha256(data).hexdigest()}
        self.rows[index] = rows

    def save(self, status, elapsed):
        manifest = {"schema": "common_evaluation_checkpoint_v1", "identity": self.identity,
                    "status": status, "completed": self.references,
                    "elapsed_wall_seconds": self.previous_wall + elapsed}
        atomic_json(self.path, manifest)
        outcome = {"status": status, "checkpoint_path": str(self.path),
                   "n_completed_program_cases": len(self.rows),
                   "n_program_cases": self.identity["n_program_cases"],
                   "elapsed_wall_seconds": manifest["elapsed_wall_seconds"]}
        atomic_json(self.root / "status.json", outcome)
        return outcome

    def all_rows(self):
        return [row for index in sorted(self.rows) for row in self.rows[index]]


def evaluate_common_programs(
    *, task_id: str, programs: Sequence[CommonProgram], cases: Sequence[CommonCase],
    evaluation, evaluation_settings: Mapping[str, object], candidate_execution,
    metrics: Sequence[str], output_directory: Path, max_workers: int = 1,
    expected_plan_rows: Sequence[Mapping[str, object]] | None = None,
    generator=None, generator_state=None, resume: bool = False, segment_budget=None,
) -> dict:
    """Export every program/case/metric, retaining bundle and execution failures.

    The caller owns the output-directory writer lock and freezes bundles there.
    resume reuses only committed complete program/Case rows. Truth
    is materialized once before any Candidate call. All programs share finite
    queries and references. HiddenVerifier sends only public Cases and plans to
    the subprocess service. Error summaries explicitly condition on success.
    """
    if not cases or not programs or not metrics:
        raise ValueError("common testing requires cases, programs, and metrics")
    for filename in ("plans.json", "cases.jsonl", "cases.csv", "summary.json"):
        if not resume and (output_directory / filename).exists():
            raise FileExistsError(output_directory / filename)
    if len(set(metrics)) != len(metrics):
        raise ValueError("common metric names must be unique")
    keys = [(case.cohort, case.record.case.case_id) for case in cases]
    if len(set(keys)) != len(keys):
        raise ValueError("case ids must be unique within each common cohort")
    program_keys = [(p.method, p.run_id, p.seed, p.artifact_id) for p in programs]
    if len(set(program_keys)) != len(program_keys):
        raise ValueError("common program identities must be unique")
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
        raise ValueError("common max_workers must be a positive integer")
    started = time.perf_counter()
    prepared = []
    plan_rows = []
    reference_failure = None
    for item in cases:
        reference = None
        failure = None
        try:
            reference = item.record.truth.answer(item.plan.query)
            reference_hash = content_sha256(reference)
        except Exception as exc:
            failure = exc
            reference_failure = reference_failure or exc
            reference_hash = None
            record_common_evaluation_failure(
                output_directory, exc, stage="prepare_reference", phase="terminal",
                cohort=item.cohort, dataset_id=item.dataset_id, case_id=item.record.case.case_id,
            )
        frozen = CaseRecord(item.record.case, _FrozenTruth(reference, failure))
        plan_json = json_value(item.plan)
        plan_row = {
            "cohort": item.cohort, "dataset_id": item.dataset_id,
            "case_id": item.record.case.case_id, "plan": plan_json,
            "plan_sha256": content_sha256(plan_json),
            "public_case_sha256": content_sha256(item.record.case),
            "reference_sha256": reference_hash,
            "reference_status": "ok" if failure is None else "failed",
            "reference_error_code": None if failure is None else type(failure).__name__,
        }
        prepared.append((item, frozen, plan_row))
        plan_rows.append(plan_row)
    if reference_failure is not None:
        write_json(output_directory / "plans.json", {
            "schema": "common_evaluation_plans_v1", "cases": plan_rows,
        })
        write_json(output_directory / "failure.json", {
            "status": "interrupted", "error_code": "common_evaluation_infrastructure_failure",
            "stage": "prepare_reference", "n_completed_program_cases": 0,
        })
        raise CandidateExecutionInfrastructureError(
            "Common evaluation stopped because a Controller reference could not be prepared"
        ) from reference_failure
    if expected_plan_rows is not None:
        expected = {(row["cohort"], row["case_id"]): row for row in expected_plan_rows}
        if len(expected) != len(expected_plan_rows) or set(expected) != set(keys):
            raise ValueError("replayed plan cases differ from selected dataset cases")
        for row in plan_rows:
            original = expected[(row["cohort"], row["case_id"])]
            for key in ("dataset_id", "plan_sha256", "public_case_sha256", "reference_sha256"):
                if original.get(key) != row[key]:
                    raise ValueError(f"replayed {key} differs for {row['case_id']}")
    identity = {
        "task_id": task_id, "metrics": list(metrics),
        "programs": [{key: getattr(program, key) for key in
                      ("method", "run_id", "seed", "artifact_id", "artifact_sha256", "error_code")}
                     for program in programs],
        "plans_sha256": content_sha256(plan_rows),
        "settings_sha256": content_sha256(evaluation_settings),
        "evaluation_program_id": getattr(evaluation, "program_id", None),
        "evaluator_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "case_hard_wall_seconds": getattr(candidate_execution, "case_hard_wall_seconds", None),
        "max_workers": max_workers, "n_program_cases": len(programs) * len(cases),
    }
    checkpoint = _CommonCheckpoint(output_directory, identity, resume=resume)
    if resume:
        saved_plans = json.loads((output_directory / "plans.json").read_text())
        if saved_plans != {"schema": "common_evaluation_plans_v1", "cases": plan_rows}:
            raise ValueError("common saved plans or references changed")
    else:
        write_json(output_directory / "plans.json", {
            "schema": "common_evaluation_plans_v1", "cases": plan_rows,
        })
    checkpoint.save("running", time.perf_counter() - started)
    verifiers = {}
    for cohort in dict.fromkeys(item.cohort for item in cases):
        subset = [item for item in cases if item.cohort == cohort]
        verifiers[cohort] = HiddenVerifier(
            _FixedPlanEvaluation(evaluation, {
                item.record.case.case_id: item.plan for item in subset
            }), subset[0].plan.query, evaluation_settings,
            phase=EvaluationPhase.TERMINAL, candidate_execution=candidate_execution,
            generator=generator, generator_state=generator_state,
        )

    interrupted = Event()

    def evaluate_one(program, prepared_case):
        if interrupted.is_set():
            return [], None
        item, record, plan_row = prepared_case
        elapsed = time.perf_counter()
        measured = {}
        failure = program.error_code
        try:
            if failure is None:
                if program.artifact is None:
                    raise ValueError("missing frozen program")
                result = verifiers[item.cohort].evaluate(program.artifact, (record,))
                measured = dict(result.cases[0].metrics)
                for metric in metrics:
                    if metric not in measured or not math.isfinite(float(measured[metric])):
                        raise ValueError(f"missing or nonfinite metric {metric}")
        except Exception as exc:
            infrastructure_failure = _is_infrastructure_failure(exc)
            if (not infrastructure_failure and isinstance(exc, CandidateExecutionCancelled)
                    and segment_budget is not None and segment_budget.cancel.is_set()):
                # Allocation expiry does not say anything about solver quality.
                interrupted.set()
                return [], None
            record_common_evaluation_failure(
                output_directory, exc,
                stage=getattr(exc, "_hidden_evaluation_stage", "evaluate_candidate"),
                phase="terminal", method=program.method, run_id=program.run_id,
                candidate_id=program.artifact_id, cohort=item.cohort,
                case_id=record.case.case_id,
                available_execution_ids_for_candidate_phase=sorted({
                    audit.execution_id for audit in getattr(candidate_execution, "audits", ())
                    if audit.artifact_id == program.artifact_id and audit.phase == "terminal"
                }),
            )
            if infrastructure_failure:
                interrupted.set()
                return [], exc
            failure = ("candidate_case_timeout" if isinstance(exc, CandidateExecutionTimeout)
                       else "candidate_evaluation_failed")
        wall = time.perf_counter() - elapsed
        base = {
            "task_id": task_id, "method": program.method, "run_id": program.run_id,
            "seed": program.seed, "artifact_id": program.artifact_id,
            "artifact_sha256": program.artifact_sha256, "cohort": item.cohort,
            "dataset_id": item.dataset_id, "case_id": record.case.case_id,
            "parent_case_id": record.case.parent_case_id, "recording_id": item.recording_id,
            "status": "ok" if failure is None else "failed", "error_code": failure,
            "solve_wall_seconds": measured.get(
                "candidate_solve_active_seconds",
                measured.get("candidate_solve_wall_seconds"),
            ),
            "query_wall_seconds": measured.get("candidate_query_wall_seconds"),
            "solve_queue_wait_seconds": measured.get(
                "candidate_solve_queue_wait_seconds",
                measured.get("candidate_platform_wait_seconds"),
            ),
            "query_queue_wait_seconds": measured.get("candidate_query_queue_wait_seconds"),
            "case_wall_seconds": wall, "plan_sha256": plan_row["plan_sha256"],
            "reference_sha256": plan_row["reference_sha256"],
            "public_case_sha256": plan_row["public_case_sha256"],
        }
        return ([{**base, "metric": metric,
                  "error": float(measured[metric]) if failure is None else None}
                 for metric in metrics], None)

    infrastructure_failure = None
    with (output_directory / "cases.jsonl").open("w", encoding="utf-8") as jsonl, \
         (output_directory / "cases.csv").open("w", newline="", encoding="utf-8") as csvfile, \
         ThreadPoolExecutor(max_workers=max_workers) as pool:
        writer = csv.DictWriter(csvfile, fieldnames=CASE_COLUMNS)
        writer.writeheader()
        def export(rows):
            for row in rows:
                jsonl.write(json.dumps(row, allow_nan=False) + "\n")
                writer.writerow(row)
            jsonl.flush()
            csvfile.flush()
        export(checkpoint.all_rows())
        for program_index, program in enumerate(programs):
            remaining = iter((program_index * len(cases) + case_index, item)
                             for case_index, item in enumerate(prepared)
                             if program_index * len(cases) + case_index not in checkpoint.rows)
            pending = {}
            exhausted = False
            while pending or not exhausted:
                while len(pending) < max_workers and not exhausted and not interrupted.is_set():
                    if segment_budget is not None and segment_budget.should_pause("common_evaluation", 0):
                        interrupted.set()
                        break
                    job = next(remaining, None)
                    if job is None:
                        exhausted = True
                        break
                    index, item = job
                    pending[pool.submit(evaluate_one, program, item)] = index
                if not pending:
                    break
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    index = pending.pop(future)
                    rows, error = future.result()
                    if error is not None and infrastructure_failure is None:
                        infrastructure_failure = error
                    if rows:
                        checkpoint.add(index, rows)
                        export(rows)
            if interrupted.is_set():
                break
    all_rows = checkpoint.all_rows()
    # The live export can finish out of order. Publish a deterministic final
    # view only after all worker/session cleanup has completed.
    import io
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=CASE_COLUMNS)
    writer.writeheader()
    writer.writerows(all_rows)
    atomic_bytes(output_directory / "cases.csv", csv_buffer.getvalue().encode())
    atomic_bytes(output_directory / "cases.jsonl",
                 "".join(json.dumps(row, allow_nan=False) + "\n" for row in all_rows).encode())
    if infrastructure_failure is not None:
        checkpoint.save("failed", time.perf_counter() - started)
        write_json(output_directory / "failure.json", {
            "status": "interrupted", "error_code": "common_evaluation_infrastructure_failure",
            "stage": getattr(infrastructure_failure, "_hidden_evaluation_stage", "evaluate_candidate"),
            "n_completed_program_cases": len(all_rows) // len(metrics),
            "completed_rows_file": "cases.jsonl", "summary_available": False,
        })
        raise CandidateExecutionInfrastructureError(
            "Common evaluation stopped because the Controller evaluation service failed; "
            "completed Case rows were retained and no final comparison was produced"
        ) from infrastructure_failure
    if len(checkpoint.rows) < len(programs) * len(cases):
        if not interrupted.is_set() or segment_budget is None:
            raise RuntimeError("common evaluation stopped without a valid allocation pause")
        return checkpoint.save("paused", time.perf_counter() - started)
    groups = {}
    grouping = ("task_id", "method", "run_id", "seed", "artifact_id", "cohort", "metric")
    for row in all_rows:
        groups.setdefault(tuple(row[name] for name in grouping), []).append(row)
    summaries = []
    for key, rows in groups.items():
        values = sorted(row["error"] for row in rows if row["status"] == "ok")
        tail_count = max(1, math.ceil(0.2 * len(values)))
        summaries.append({
            **dict(zip(grouping, key)), "n_requested": len(rows),
            "n_success": len(values), "n_failed": len(rows) - len(values),
            "success_rate": len(values) / len(rows),
            "mean_success": fmean(values) if values else None,
            "cvar20_success": fmean(values[-tail_count:]) if values else None,
            "max_success": max(values) if values else None,
        })
    summary = {
        "schema": "common_evaluation_summary_v1", "status": "completed",
        "total_wall_seconds": checkpoint.previous_wall + time.perf_counter() - started,
        "n_programs": len(programs), "n_cases": len(cases), "n_rows": len(all_rows),
        "n_failed_program_cases": sum(row["status"] != "ok" for row in all_rows) // len(metrics),
        "metrics": list(metrics), "groups": summaries,
        "timing_definition": {
            "solve_wall_seconds": "Task-measured prepare_solution elapsed wall time excluding trusted controller Oracle queue wait; prefer explicit candidate_solve_active_seconds when present, otherwise candidate_solve_wall_seconds; actual Oracle computation remains charged; null if unavailable",
            "query_wall_seconds": "Task-measured finite query elapsed wall time excluding trusted controller Oracle queue wait; actual Oracle computation remains charged; null if unavailable",
            "solve_queue_wait_seconds": "Trusted controller Oracle queue wait excluded from solve_wall_seconds; use candidate_solve_queue_wait_seconds or compatible candidate_platform_wait_seconds; null if unavailable",
            "query_queue_wait_seconds": "Trusted controller Oracle queue wait excluded from query_wall_seconds; null if unavailable",
            "case_wall_seconds": "Raw HiddenVerifier elapsed wall time including queue wait, subprocess startup and scoring; excludes common reference preparation",
            "total_wall_seconds": "Sum of segment elapsed wall times including queue wait, common reference preparation and evaluation/export; excludes scheduler gaps, input loading and bundle freeze",
        },
        "aggregation_note": "Error summaries condition on successful cases; failures remain explicit. No seed-level inference is performed.",
    }
    write_json(output_directory / "summary.json", summary)
    checkpoint.save("completed", time.perf_counter() - started)
    return summary
