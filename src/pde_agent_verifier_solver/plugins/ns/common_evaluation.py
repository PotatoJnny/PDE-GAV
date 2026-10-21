"""Common evaluation on numerical NS splits or frozen continuous B/G/P banks.

Continuous banks replay their stored terminal plans and exact references while
retaining public initial/forcing oracles. No sampling or model training occurs.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Sequence

from pde_agent_verifier_solver.engine.common_evaluation import (
    CommonCase, CommonProgram, content_sha256, evaluate_common_programs, write_json,
    record_common_evaluation_failure,
)
from pde_agent_verifier_solver.engine.candidate_execution import (
    PythonTaskWorkerSpec, SubprocessCandidateExecutionService,
)
from pde_agent_verifier_solver.engine.python_candidate import PythonCandidateCodec
from pde_agent_verifier_solver.engine.checkpoint_store import run_writer_lock
from pde_agent_verifier_solver.sdk import (
    AnswerQuery, CaseRecord, CaseVisibility, EvaluationPhase, EvaluationPlan, GenerationPurpose, QueryKind, TaskKind,
)
from .case_bank import SCHEMA as CONTINUOUS_BANK_SCHEMA, NSBankSessionProvider, load_ns_continuous_case_bank
from .datasets import LegacyNSD0Adapter
from .tasks.solver import NSSolverTask

MANIFEST_SCHEMA = "ns_common_evaluation_v1"


def _object(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return data


def _path(value: str, base: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("manifest paths must be nonempty strings")
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_numerical_split(
    manifest_path: str | Path, *, split: str,
    visibility: CaseVisibility = CaseVisibility.HIDDEN,
    case_ids: Sequence[str] | None = None,
) -> tuple[CaseRecord, ...]:
    """Load exactly one original JSON/NPZ NS split without Generator preparation.

    Explicit case_ids preserve caller order; absent case_ids means the full
    split in manifest order. Both development and hidden visibility are usable.
    """
    path = Path(manifest_path).expanduser().resolve()
    manifest = _object(path)
    adapter = LegacyNSD0Adapter()
    if not adapter.accepts(manifest):
        raise ValueError("only original numerical NS ns_d0_v1 manifests are supported")
    if not isinstance(visibility, CaseVisibility):
        raise TypeError("visibility must be CaseVisibility")
    rows = manifest.get("splits", {}).get(split)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"missing or empty numerical NS split: {split}")
    indexed = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("case_id"), str) or not row["case_id"]:
            raise ValueError("NS split rows need nonempty case_id")
        if row["case_id"] in indexed:
            raise ValueError(f"duplicate NS case id: {row['case_id']}")
        indexed[row["case_id"]] = row
    selected = list(indexed) if case_ids is None else list(case_ids)
    if not selected or len(set(selected)) != len(selected) or any(key not in indexed for key in selected):
        raise ValueError("case_ids must be a nonempty, unique subset of the chosen split")
    records = []
    for key in selected:
        row = dict(indexed[key])
        # Resolve declared paths against the manifest before the legacy adapter's
        # cwd fallback, so a same-named local directory cannot change the bank.
        raw = row.get("case_dir")
        if raw:
            declared = _path(raw, path.parent)
            if not declared.is_dir():
                raise FileNotFoundError(f"declared NS case directory is missing: {declared}")
            row["case_dir"] = str(declared)
        record = adapter._record(row, path, visibility)
        parent = row.get("parent_case_id")
        if parent is not None:
            record = CaseRecord(replace(record.case, parent_case_id=str(parent)), record.truth)
        records.append(record)
    return tuple(records)


def _integer(value, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _identity(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def run_ns_common_evaluation(
    manifest_path: str | Path, output_directory: str | Path, *,
    resume: bool = False, segment_budget=None,
) -> dict:
    """Evaluate the complete declared program list on one shared external bank.

    Input schema has datasets, programs, query, budget, and optional task_id.
    Every program declares role=initial/final/round. require_initial=true can
    require S0; cold-start comparisons are otherwise valid. expected_tree_sha256
    is optional: absent values are pinned by
    the newly frozen copy and recorded in resolved_manifest.json. A safe paused
    result can resume in this same directory; completed Case rows are retained.
    Allocation cancellation returns status=paused after worker cleanup.
    """
    pipeline_started = time.perf_counter()
    source = Path(manifest_path).expanduser().resolve()
    spec = _object(source)
    if spec.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"manifest schema must be {MANIFEST_SCHEMA}")
    datasets = spec.get("datasets")
    programs = spec.get("programs")
    if not isinstance(datasets, list) or not datasets or not isinstance(programs, list) or not programs:
        raise ValueError("datasets and programs must be nonempty lists")
    identities = []
    for program in programs:
        if not isinstance(program, dict):
            raise ValueError("program entries must be objects")
        for key in ("method", "run_id", "artifact_id", "bundle"):
            _identity(program.get(key), key)
        _integer(program.get("seed"), "program seed")
        if program.get("role") not in {"initial", "final", "round"}:
            raise ValueError("program role must be initial, final, or round")
        identities.append(tuple(program[key] for key in ("method", "run_id", "seed", "artifact_id")))
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate program identity")
    require_initial = spec.get("require_initial", False)
    if not isinstance(require_initial, bool):
        raise ValueError("require_initial must be boolean")
    contains_initial = any(program["role"] == "initial" for program in programs)
    if require_initial and not contains_initial:
        raise ValueError("require_initial=true requires S0 (role=initial)")
    query_spec = spec.get("query", {})
    if not isinstance(query_spec, dict):
        raise ValueError("query must be an object")
    count = _integer(query_spec.get("point_count"), "query point_count", 1)
    query_seed = (None if query_spec.get("seed") is None
                  else _integer(query_spec["seed"], "query seed"))
    include_pressure = query_spec.get("include_pressure", False)
    if not isinstance(include_pressure, bool):
        raise ValueError("query include_pressure must be boolean")
    fields = ("velocity_x", "velocity_y") + (("pressure",) if include_pressure else ())
    default_query = AnswerQuery.at_points(fields=fields, points=({"t": 0.0, "x": 0.0, "y": 0.0},))
    settings = {
        "default_query": default_query, "primary_metric": "velocity_relative_l2",
        "include_pressure": include_pressure, "terminal_point_count": count,
    }
    task = NSSolverTask()
    evaluation = task.create_evaluation(settings)
    budget = spec.get("budget", {})
    if not isinstance(budget, dict):
        raise ValueError("budget must be an object")
    workers = _integer(budget.get("max_workers", 1), "budget max_workers", 1)
    raw_timeout = budget.get("case_hard_wall_seconds")
    if isinstance(raw_timeout, bool) or not isinstance(raw_timeout, (int, float)):
        raise ValueError("budget case_hard_wall_seconds must be numeric")
    timeout = float(raw_timeout)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("budget case_hard_wall_seconds must be finite and positive")
    expected = None
    replay = {}
    if query_spec.get("plans_file"):
        saved = _object(_path(query_spec["plans_file"], source.parent))
        if saved.get("schema") != "common_evaluation_plans_v1":
            raise ValueError("invalid common plans schema")
        expected = saved["cases"]
        replay = {(row["cohort"], row["case_id"]): row for row in expected}
    cases = []
    resolved_datasets = []
    continuous_banks = {}
    seen_cohorts = set()
    for dataset in datasets:
        if not isinstance(dataset, dict):
            raise ValueError("dataset entries must be objects")
        cohort = _identity(dataset.get("cohort"), "cohort")
        if cohort in seen_cohorts:
            raise ValueError("each dataset entry must have a distinct cohort")
        seen_cohorts.add(cohort)
        dataset_id = _identity(dataset.get("dataset_id"), "dataset_id")
        location = _path(dataset.get("manifest"), source.parent)
        data_manifest = _object(location)
        bank = None
        if data_manifest.get("schema") == CONTINUOUS_BANK_SCHEMA:
            requested_ids = dataset.get("case_ids")
            if requested_ids is not None and (not isinstance(requested_ids, list)
                    or any(not isinstance(key, str) for key in requested_ids)):
                raise ValueError("continuous dataset case_ids must be a list of strings")
            if location not in continuous_banks:
                # Cohorts may overlap within one bank. Load their union once,
                # so the public-session provider still owns each Case once.
                shared_ids = []
                for other in datasets:
                    if not isinstance(other, dict):
                        raise ValueError("dataset entries must be objects")
                    if _path(other.get("manifest"), source.parent) != location:
                        continue
                    subset = other.get("case_ids")
                    if subset is None:
                        shared_ids = None
                        break
                    if not isinstance(subset, list) or any(not isinstance(key, str) for key in subset):
                        raise ValueError("continuous dataset case_ids must be a list of strings")
                    shared_ids.extend(subset)
                if shared_ids is not None:
                    shared_ids = list(dict.fromkeys(shared_ids))
                continuous_banks[location] = load_ns_continuous_case_bank(
                    location, expected_purpose=GenerationPurpose.TERMINAL,
                    device=dataset.get("device", "cpu"),
                    **({"case_ids": shared_ids} if shared_ids is not None else {}),
                )
            bank = continuous_banks[location]
            records = bank.records
            if dataset.get("strata") is not None:
                from .case_bank import STRATA
                selected_strata = dataset["strata"]
                if (not isinstance(selected_strata, list) or not selected_strata
                        or set(selected_strata) - STRATA):
                    raise ValueError("continuous dataset strata must name B/G/P strata")
                allowed = set(selected_strata)
                if "perturbed" in allowed:
                    allowed.update(("base_perturbed", "guided_perturbed"))
                records = tuple(row for row in records if bank.strata[row.case.case_id] in allowed)
            indexed = {row.case.case_id: row for row in records}
            ids = dataset.get("case_ids", list(indexed))
            if (not isinstance(ids, list) or not ids or len(set(ids)) != len(ids)
                    or set(ids) - set(indexed)):
                raise ValueError("continuous dataset case_ids must be a nonempty unique subset")
            records = tuple(indexed[key] for key in ids)
            manifest_rows = {key: {} for key in indexed}
        else:
            split = _identity(dataset.get("split"), "split")
            _integer(query_seed, "query seed")
            records = load_numerical_split(location, split=split, case_ids=dataset.get("case_ids"))
            manifest_rows = {row["case_id"]: row for row in data_manifest["splits"][split]}
        for record in records:
            if bank is not None:
                # The bank's private query and seed are authoritative. A query
                # seed in the common manifest applies only to numerical splits.
                plan = bank.plans[record.case.case_id]
                seed = plan.seed
                if plan.query.fields != fields or len(plan.query.points) != count:
                    raise ValueError("continuous bank query differs from common evaluation settings")
            else:
                seed = query_seed + len(cases)
                plan = evaluation.make_plan(record.case, EvaluationPhase.TERMINAL, settings, seed)
            if expected is not None:
                saved_row = replay.get((cohort, record.case.case_id))
                if saved_row is None:
                    raise ValueError("replayed plan is missing a selected case")
                saved_plan = saved_row["plan"]
                raw_query = saved_plan["query"]
                saved_query = AnswerQuery(
                    QueryKind(raw_query["kind"]), tuple(raw_query["fields"]),
                    raw_query.get("resolution", {}), tuple(raw_query.get("points", ())),
                )
                if bank is not None:
                    if content_sha256(saved_plan) != content_sha256(plan):
                        raise ValueError("replayed plan differs from the frozen continuous bank")
                else:
                    if (saved_plan["phase"] != EvaluationPhase.TERMINAL.value
                            or saved_query.kind is not QueryKind.POINTS or saved_query.fields != fields
                            or len(saved_query.points) != count or saved_plan["seed"] != seed
                            or content_sha256(saved_plan["settings"]) != content_sha256(settings)):
                        raise ValueError("replayed query configuration differs from the requested plan")
                    plan = EvaluationPlan(EvaluationPhase(saved_plan["phase"]), saved_query,
                                          seed=seed, settings=settings)
            cases.append(CommonCase(cohort, dataset_id, record, plan,
                                    manifest_rows[record.case.case_id].get("recording_id")))
        resolved_datasets.append({
            **dataset, "manifest": str(location), "manifest_sha256": _file_sha256(location),
            "case_ids": [record.case.case_id for record in records],
            "reference_kind": ("frozen_continuous_ns_exact_reference" if bank is not None
                               else "original_numerical_hidden_truth_cache"),
            **({"bank_id": bank.bank_id, "purpose": bank.purpose.value,
                "strata": {row.case.case_id: bank.strata[row.case.case_id] for row in records},
                "generation_provenance": dict(bank.provenance)} if bank is not None else {}),
        })
    output = Path(output_directory).expanduser().resolve()
    if resume:
        if not output.is_dir():
            raise FileNotFoundError(output)
    else:
        output.mkdir(parents=True, exist_ok=False, mode=0o700)
    with run_writer_lock(output):
        return _evaluate_ns_common_locked(
            source=source, spec=spec, output=output, resolved_datasets=resolved_datasets,
            contains_initial=contains_initial, evaluation=evaluation, settings=settings,
            workers=workers, timeout=timeout, programs=programs, cases=cases,
            include_pressure=include_pressure, continuous_banks=continuous_banks,
            expected=expected, pipeline_started=pipeline_started, resume=resume,
            segment_budget=segment_budget,
        )


def _evaluate_ns_common_locked(
    *, source, spec, output, resolved_datasets, contains_initial, evaluation, settings,
    workers, timeout, programs, cases, include_pressure, continuous_banks, expected,
    pipeline_started, resume, segment_budget,
):
    previous_resolved = None
    previous_pipeline_wall = 0.0
    if resume:
        previous_pipeline_wall = float(_object(output / "checkpoint.json").get("pipeline_wall_seconds", 0.0))
        if not math.isfinite(previous_pipeline_wall) or previous_pipeline_wall < 0:
            raise ValueError("invalid saved common pipeline elapsed time")
    if resume:
        if _object(output / "input_manifest.json") != spec:
            raise ValueError("common resume input manifest differs")
        previous_resolved = _object(output / "resolved_manifest.json")
    else:
        write_json(output / "input_manifest.json", spec)
    resolved = {
        **spec, "manifest_path": str(source), "manifest_sha256": _file_sha256(source),
        "datasets": resolved_datasets, "programs": [],
        "contains_initial": contains_initial,
        "evaluation_program_id": evaluation.program_id,
        "evaluation_source_sha256": _file_sha256(Path(__file__).parent / "tasks" / "solver.py"),
        "budget": {"max_workers": workers, "case_hard_wall_seconds": timeout},
        "execution_policy": {
            "candidate_invocation": "bounded_same_user_subprocess",
            "controller_truth_shared": False, "hostile_filesystem_sandbox": False,
            "budget_scope": "per-case subprocess hard wall; no solver tolerance or GPU accounting",
        },
    }
    if resume:
        for key, value in resolved.items():
            if key != "programs" and previous_resolved.get(key) != value:
                raise ValueError(f"common resume resolved {key} differs")
        if len(previous_resolved["programs"]) != len(programs):
            raise ValueError("common resume frozen program count differs")
    codec = PythonCandidateCodec(task_kind=TaskKind.SOLVER)
    bound = []
    for index, program in enumerate(programs):
        artifact = None
        digest = None
        error_code = None
        bundle = _path(program["bundle"], source.parent)
        saved_program = previous_resolved["programs"][index] if resume else None
        if saved_program is not None:
            for key, value in {**program, "bundle": str(bundle)}.items():
                if saved_program.get(key) != value:
                    raise ValueError("common resume frozen program declaration differs")
            digest = saved_program["artifact_sha256"]
            error_code = saved_program["error_code"]
        try:
            if error_code is not None:
                # An invalid original bundle is a frozen experiment outcome;
                # resume must not silently replace it with newly repaired code.
                bound.append(CommonProgram(program["method"], program["run_id"], program["seed"],
                                           program["artifact_id"], None, digest, error_code))
                resolved["programs"].append(saved_program)
                continue
            artifact = codec.bind_existing(
                saved_program["frozen_bundle"] if resume else bundle,
                storage_root=output / "programs" / str(index),
                artifact_id=program["artifact_id"], name=program["artifact_id"],
                summary="Frozen program for external common evaluation", usage="prepare_solution",
            )
            digest = artifact.program.tree_sha256
        except Exception as exc:
            if resume:
                raise ValueError("common resume frozen bundle cannot be verified") from exc
            record_common_evaluation_failure(
                output, exc, stage="bind_frozen_program", phase="terminal",
                candidate_id=program["artifact_id"], method=program["method"],
                bundle=str(bundle),
            )
            artifact = None
            error_code = "bundle_" + type(exc).__name__
        bound.append(CommonProgram(
            program["method"], program["run_id"], program["seed"], program["artifact_id"],
            artifact, digest, error_code,
        ))
        resolved["programs"].append({
            **program, "bundle": str(bundle), "artifact_sha256": digest,
            "frozen_bundle": str(artifact.program.bundle_locator) if artifact else None,
            "status": "ok" if error_code is None else "failed", "error_code": error_code,
        })
    if resume:
        if previous_resolved != resolved:
            raise ValueError("common resume resolved manifest identity differs")
    else:
        write_json(output / "resolved_manifest.json", resolved)
    execution = SubprocessCandidateExecutionService(
        output / "execution", task=PythonTaskWorkerSpec.from_task_type(NSSolverTask),
        evaluation_settings=settings, max_workers=workers, case_hard_wall_seconds=timeout,
        oracle_max_workers=min(128, workers),
    )
    if segment_budget is not None:
        execution.allocation_cancellation_event = segment_budget.cancel
    metrics = ("velocity_relative_l2", "velocity_x_relative_l2", "velocity_y_relative_l2")
    if include_pressure:
        metrics += ("pressure_relative_l2",)
    bank_sessions = NSBankSessionProvider(tuple(continuous_banks.values())) if continuous_banks else None
    try:
        summary = evaluate_common_programs(
            task_id=spec.get("task_id", "ns-full-field-solver"), programs=bound, cases=cases,
            evaluation=evaluation, evaluation_settings=settings, candidate_execution=execution,
            metrics=metrics, output_directory=output, max_workers=workers,
            expected_plan_rows=expected,
            generator=bank_sessions,
            generator_state=bank_sessions.state if bank_sessions else None,
            resume=resume, segment_budget=segment_budget,
        )
        pipeline_wall = previous_pipeline_wall + time.perf_counter() - pipeline_started
        checkpoint = _object(output / "checkpoint.json")
        checkpoint["pipeline_wall_seconds"] = pipeline_wall
        write_json(output / "checkpoint.json", checkpoint)
        if summary["status"] == "paused":
            return summary
        summary["contains_initial"] = contains_initial
        summary["pipeline_wall_seconds"] = pipeline_wall
        summary["timing_definition"]["pipeline_wall_seconds"] = (
            "Sum of NS common evaluation segment calls including input loading, bundle freezing, "
            "reference preparation, candidate execution, scoring and export; excludes scheduler gaps "
            "and Python CLI startup"
        )
        write_json(output / "summary.json", summary)
        return summary
    except Exception as exc:
        write_json(output / "status.json", {
            "status": "failed", "error_code": "common_evaluation_pipeline_failed",
        })
        record_common_evaluation_failure(output, exc, stage="ns_common_pipeline", phase="terminal")
        # The shared evaluator records interruption and partial completion. Do
        # not overwrite that report with a generic pipeline failure.
        if not (output / "failure.json").exists():
            write_json(output / "failure.json", {
                "status": "failed", "error_code": "common_evaluation_pipeline_failed",
            })
        raise
