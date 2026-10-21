#!/usr/bin/env python3
"""Measure the real NS subprocess/Oracle path on unchanged frozen inputs.

Runs no Agent, API, sampling or training. Every source/worker/thread combination
is a fresh Controller process with the same diagnostic subset and candidate bytes.
The subset is a separate artifact, never a replacement for a scientific bank.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import subprocess
import sys
import time


THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
SOURCE_FILES = ("src/pde_agent_verifier_solver/engine/candidate_execution.py",
                "src/pde_agent_verifier_solver/engine/agent_wire.py",
                "src/pde_agent_verifier_solver/engine/evaluator.py",
                "src/pde_agent_verifier_solver/plugins/ns/continuous_backend.py",
                "src/pde_agent_verifier_solver/plugins/ns/evaluation_devices.py",
                "src/pde_agent_verifier_solver/plugins/ns/analytic_cuda.py",
                "legacy/ns/src/agent_verifier/neural_generator/structured_ns_analytic.py",
                "legacy/ns/src/agent_verifier/neural_generator/structured_ns_geometry_jets.py",
                "legacy/ns/src/agent_verifier/neural_generator/structured_ns_answer_first.py",
                "legacy/ns/src/agent_verifier/neural_generator/continuous_geometry.py")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def content_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                   allow_nan=False) + "\n")


def prepare_subset(bank_path, destination, strata, count):
    """Copy existing fields and plans verbatim; do not regenerate or reselect points."""
    bank_path, destination = Path(bank_path).resolve(), Path(destination).resolve()
    data = json.loads(bank_path.read_text())
    unsigned = {key: value for key, value in data.items() if key != "content_sha256"}
    selected, totals = [], Counter()
    for row in data["cases"]:
        stratum = row["stratum"]
        if stratum in strata and totals[stratum] < count:
            selected.append(row)
            totals[stratum] += 1
    if any(totals[key] != count for key in strata):
        raise ValueError(f"bank has insufficient cases for requested strata: {dict(totals)}")
    destination.parent.mkdir(parents=True, exist_ok=False)
    for index, row in enumerate(selected):
        source = (bank_path.parent / row["state_path"]).resolve()
        if not source.is_relative_to(bank_path.parent):
            raise ValueError('source field is outside bank')
        target = destination.parent / "fields" / f"{index:06d}" / source.name
        target.parent.mkdir(parents=True)
        shutil.copyfile(source, target)
        row["state_path"] = str(target.relative_to(destination.parent))
    data.update(bank_id="diagnostic-runtime-subset", cases=selected,
                case_count=len(selected), strata_counts=dict(totals))
    data["provenance"] = {**data.get("provenance", {}),
                          "runtime_benchmark_source_manifest": str(bank_path),
                          "runtime_benchmark_source_manifest_sha256": digest(bank_path),
                          "scope": "diagnostic subset; not scientific selection or final evaluation"}
    data.pop("content_sha256", None)
    data["content_sha256"] = content_digest(data)
    write_json(destination, data)
    return {"path": str(destination), "sha256": digest(destination),
            "source_sha256": digest(bank_path), "counts": dict(totals),
            "case_ids": [row["case"]["case_id"] for row in selected]}


def io_snapshot():
    try:
        return {key: int(value) for key, value in
                (line.split(":", 1) for line in Path("/proc/self/io").read_text().splitlines())}
    except OSError:
        return {}


def usage_snapshot():
    result = {}
    for label, target in (("controller", resource.RUSAGE_SELF), ("children", resource.RUSAGE_CHILDREN)):
        value = resource.getrusage(target)
        result[label] = {name: getattr(value, name) for name in
                         ("ru_utime", "ru_stime", "ru_inblock", "ru_oublock", "ru_nvcsw", "ru_nivcsw")}
    return result


def finite_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): finite_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_value(item) for item in value]
    return value


def run_child(spec):
    # Source-specific imports happen only after the parent fixes PYTHONPATH.
    import torch
    from pde_agent_verifier_solver.engine.candidate_execution import (
        PythonTaskWorkerSpec, SubprocessCandidateExecutionService,
    )
    from pde_agent_verifier_solver.engine.evaluator import HiddenVerifier
    from pde_agent_verifier_solver.engine.python_candidate import PythonCandidateCodec
    from pde_agent_verifier_solver.plugins.ns.case_bank import load_ns_continuous_case_bank
    from pde_agent_verifier_solver.plugins.ns.tasks import NSSolverTask
    from pde_agent_verifier_solver.sdk import TaskKind

    def interrupted(_signal, _frame):
        raise InterruptedError("benchmark Controller received termination; closing Candidate service")

    signal.signal(signal.SIGTERM, interrupted)
    torch.set_num_threads(spec["controller_threads"])
    torch.set_num_interop_threads(1)
    output = Path(spec["output"])
    output.mkdir(parents=True, exist_ok=False)
    root = Path(spec["work_root"])
    root.mkdir(parents=True, exist_ok=False)
    report = {**spec, "status": "running", "hostname": platform.node(),
              "python": sys.executable, "torch": torch.__version__,
              "controller_threads_actual": torch.get_num_threads(),
              "thread_environment": {key: os.environ.get(key) for key in THREAD_ENV},
              "model_invoked": False, "scientific_bank_modified": False,
              "benchmark_script_sha256": digest(Path(__file__)),
              "loaded_module_files": {name: str(Path(sys.modules[name].__file__).resolve()) for name in (
                  "pde_agent_verifier_solver.engine.candidate_execution",
                  "pde_agent_verifier_solver.engine.evaluator",
                  "pde_agent_verifier_solver.plugins.ns.continuous_backend")},
              "source_sha256": {name: digest(Path(spec["source_root"]) / name)
                                for name in SOURCE_FILES if (Path(spec["source_root"]) / name).is_file()}}
    sessions = []
    service = None
    evaluation_scope = ExitStack()
    pool = None
    started = time.perf_counter()
    try:
        purpose = json.loads(Path(spec["bank"]).read_text())["purpose"]
        bank = load_ns_continuous_case_bank(spec["bank"], expected_purpose=purpose, device=spec["device"])
        report["bank_load_wall_seconds"] = time.perf_counter() - started
        original_open = bank.generator.open_case_session

        def observed_open(*args, **kwargs):
            session = original_open(*args, **kwargs)
            sessions.append(session)
            return session

        bank.generator.open_case_session = observed_open
        settings = {"default_query": next(iter(bank.plans.values())).query,
                    "include_pressure": False}
        task = NSSolverTask()
        evaluation = bank.fixed_evaluation(task.create_evaluation(settings))
        artifact = PythonCandidateCodec(task_kind=TaskKind.SOLVER).bind_existing(
            spec["candidate"], storage_root=root / "candidate", artifact_id="runtime-benchmark",
            name=spec["candidate_label"], summary="Unmodified saved candidate", usage="Performance diagnostic")
        report["candidate_tree_sha256"] = artifact.program.tree_sha256
        execution_options = {}
        if "oracle_max_workers" in inspect.signature(SubprocessCandidateExecutionService).parameters:
            execution_options["oracle_max_workers"] = min(spec["workers"], 128)
        report["oracle_max_workers"] = execution_options.get("oracle_max_workers", 1)
        service = SubprocessCandidateExecutionService(
            root / "execution", task=PythonTaskWorkerSpec.from_task_type(NSSolverTask),
            evaluation_settings=settings, max_workers=spec["workers"],
            case_hard_wall_seconds=spec["case_wall_seconds"], python_executable=sys.executable,
            **execution_options)
        if spec.get("evaluation_devices"):
            from pde_agent_verifier_solver.sdk.evaluation_devices import (
                EvaluationDevicePool, bind_evaluation_device_pool,
            )

            pool = evaluation_scope.enter_context(EvaluationDevicePool(spec["evaluation_devices"]))
            evaluation_scope.enter_context(bind_evaluation_device_pool(pool))
        verifier = HiddenVerifier(evaluation, settings["default_query"], settings,
                                  seed=101, phase=next(iter(bank.plans.values())).phase,
                                  generator=bank.generator,
                                  generator_state=bank.generator_state, candidate_execution=service)
        before_io, before_usage = io_snapshot(), usage_snapshot()
        evaluation_started = time.perf_counter()
        result = verifier.evaluate(artifact, bank.records)
        if spec["device"].startswith("cuda"):
            torch.cuda.synchronize()
        report["evaluation_wall_seconds"] = time.perf_counter() - evaluation_started
        if spec.get("require_analytic_cuda"):
            caches = [getattr(resource, "_analytic_cuda_cache", None)
                      for resource in pool._resources.values()] if pool is not None else []
            if not any(cache is not None and cache.diagnostics["replays"] > 0 for cache in caches):
                raise RuntimeError("benchmark requested analytic CUDA but did not replay a production graph")
        after_io, after_usage = io_snapshot(), usage_snapshot()
        report["controller_io_delta"] = {key: after_io[key] - before_io.get(key, 0) for key in after_io}
        report["resource_usage_delta"] = {
            kind: {key: after_usage[kind][key] - before_usage[kind][key] for key in after_usage[kind]}
            for kind in after_usage}
        report["io_scope"] = ("proc io is Controller only; rusage children contains all completed child CPU/block IO. "
                              "Logical bytes and syscall counts do not measure time attributable to storage.")
        report["primary_error"] = result.primary_error
        report["cases"] = [{"case_id": row.case_id, "stratum": bank.strata[row.case_id],
                            "primary_error": row.primary_error, "metrics": dict(row.metrics)} for row in result.cases]
        complete = tuple(row.case_id for row in result.cases) == tuple(row.case.case_id for row in bank.records)
        report["complete_case_inventory"] = complete
        report["status"] = "passed" if complete and all(math.isfinite(row.primary_error) for row in result.cases) else "failed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        close = getattr(service, "close", None)
        if callable(close):
            close()
        caches = ([] if pool is None else [getattr(resource, "_analytic_cuda_cache", None)
                  for resource in pool._resources.values()])
        caches = [cache for cache in caches if cache is not None]
        report["analytic_cuda_before_cleanup"] = [cache.diagnostics for cache in caches]
        cleanup_started = time.perf_counter()
        evaluation_scope.close()
        report["evaluation_device_cleanup_seconds"] = time.perf_counter() - cleanup_started
        report["evaluation_device_cleanup_confirmed"] = None if pool is None else pool.cleanup_confirmed
        report["analytic_cuda_after_cleanup"] = [cache.diagnostics for cache in caches]
        report["oracle_sessions"] = [dict(session.usage_snapshot) for session in sessions]
        report["total_wall_seconds"] = time.perf_counter() - started
        audits = root / "execution" / "candidate_execution_audits.jsonl"
        if audits.is_file():
            shutil.copyfile(audits, output / "candidate_execution_audits.jsonl")
            rows = [json.loads(line) for line in audits.read_text().splitlines()]
            report["audit_status_counts"] = dict(Counter(row["status"] for row in rows))
            report["audit_timing_totals"] = {key: sum(float(row.get(key, 0) or 0) for row in rows)
                                              for key in ("wall_seconds", "charged_wall_seconds", "oracle_queue_wait_seconds")}
        report["oracle_usage_totals"] = {key: sum(float(row.get(key, 0)) for row in report["oracle_sessions"])
                                          for key in ("query_calls", "queried_points", "backend_wall_seconds")}
        write_json(output / "result.json", finite_value(report))
    return report["status"] == "passed"


def named_paths(values):
    result = []
    labels = set()
    for value in values:
        label, separator, path = value.partition("=")
        if not separator or not label or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in label):
            raise ValueError("named path must be safe-label=/absolute/path")
        if label in labels:
            raise ValueError("duplicate label")
        labels.add(label)
        result.append((label, str(Path(path).expanduser().resolve())))
    return result


def compare_results(reports, rtol=2e-10, atol=1e-9):
    reference, checks = {}, []
    for report in reports:
        key = report["candidate_label"], report["device"]
        if report["status"] != "passed":
            checks.append({"run": report["label"], "passed": False, "reason": "execution failed"})
            continue
        if key not in reference:
            reference[key] = report
        before = reference[key]
        same_ids = [row["case_id"] for row in before["cases"]] == [row["case_id"] for row in report["cases"]]
        same_bytes = before["candidate_tree_sha256"] == report["candidate_tree_sha256"]
        same_calls = before["oracle_usage_totals"]["query_calls"] == report["oracle_usage_totals"]["query_calls"]
        same_points = before["oracle_usage_totals"]["queried_points"] == report["oracle_usage_totals"]["queried_points"]
        errors_match = same_ids and all(a["primary_error"] is not None and b["primary_error"] is not None
                                      and math.isclose(a["primary_error"], b["primary_error"], rel_tol=rtol, abs_tol=atol)
                                      for a, b in zip(before["cases"], report["cases"]))
        checks.append({"run": report["label"], "reference": before["label"],
                       "passed": same_bytes and same_calls and same_points and errors_match,
                       "same_candidate": same_bytes, "same_cases": same_ids,
                       "same_query_calls": same_calls, "same_queried_points": same_points,
                       "errors_match": errors_match, "rtol": rtol, "atol": atol})
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child-spec", type=Path)
    parser.add_argument("--source-root", action="append", help="label=/path; repeat for baseline/current")
    parser.add_argument("--candidate", action="append", help="label=/path/to/saved/bundle")
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--work-root", type=Path, help="Optional node-local runtime root")
    parser.add_argument("--workers", default="1,4,8,28")
    parser.add_argument("--controller-threads", default="1")
    parser.add_argument("--cases-per-stratum", type=int, default=4)
    parser.add_argument("--strata", default="base,guided,base_perturbed")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--case-wall-seconds", type=float, default=900)
    parser.add_argument("--run-wall-seconds", type=float, default=1800)
    args = parser.parse_args()
    if args.child_spec:
        return 0 if run_child(json.loads(args.child_spec.read_text())) else 1
    if not all((args.source_root, args.candidate, args.bank, args.output)):
        parser.error("provide --source-root, --candidate, --bank and --output")
    workers = [int(value) for value in args.workers.split(",")]
    threads = [int(value) for value in args.controller_threads.split(",")]
    if (min(*workers, *threads, args.cases_per_stratum, args.repeats) < 1
            or not math.isfinite(args.case_wall_seconds) or args.case_wall_seconds <= 0
            or not math.isfinite(args.run_wall_seconds) or args.run_wall_seconds <= 0):
        parser.error("counts must be positive")
    sources, candidates = named_paths(args.source_root), named_paths(args.candidate)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    work_root = args.work_root.resolve() if args.work_root else output / "work"
    work_root.mkdir(parents=True, exist_ok=True)
    subset = prepare_subset(args.bank, output / "input" / "bank.json",
                            args.strata.split(","), args.cases_per_stratum)
    write_json(output / "inputs.json", subset)
    reports = []
    # Alternate source order each repeat to reduce systematic cold/warm bias.
    for repeat in range(args.repeats):
        for candidate_label, candidate in candidates:
            for worker_count in workers:
                for thread_count in threads:
                    for source_label, source in (sources if repeat % 2 == 0 else sources[::-1]):
                        label = f"{candidate_label}-{source_label}-w{worker_count}-t{thread_count}-r{repeat}"
                        spec = {"label": label, "source_root": source, "candidate": candidate,
                                "candidate_label": candidate_label, "bank": subset["path"],
                                "workers": worker_count, "controller_threads": thread_count,
                                "device": args.device, "case_wall_seconds": args.case_wall_seconds,
                                "output": str(output / label), "work_root": str(work_root / label)}
                        spec_path = output / f"{label}.json"
                        write_json(spec_path, spec)
                        environment = dict(os.environ)
                        environment.update({key: str(thread_count) for key in THREAD_ENV})
                        environment.update(PYTHONPATH=f"{source}/src:{source}/legacy/ns/src",
                                           PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
                        log = output / f"{label}.log"
                        print(json.dumps({"started": label}), flush=True)
                        process_started = time.perf_counter()
                        with log.open("w") as stream:
                            try:
                                child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                                          "--child-spec", str(spec_path)],
                                                         env=environment, cwd=source, stdout=stream,
                                                         stderr=subprocess.STDOUT, start_new_session=True)
                                returncode = child.wait(timeout=args.run_wall_seconds)
                            except subprocess.TimeoutExpired:
                                child.terminate()
                                try:
                                    child.wait(timeout=45)
                                except subprocess.TimeoutExpired:
                                    os.killpg(child.pid, signal.SIGKILL)
                                    child.wait()
                                returncode = 124
                        result_path = Path(spec["output"]) / "result.json"
                        report = json.loads(result_path.read_text()) if result_path.is_file() else {
                            **spec, "status": "failed", "error": f"child exited {returncode}; see {log}"}
                        report["process_wall_seconds"] = time.perf_counter() - process_started
                        if returncode:
                            report["status"] = "failed"
                        reports.append(report)
                        checks = compare_results(reports)
                        write_json(output / "summary.json", {"schema": "ns_evaluation_runtime_benchmark_v1",
                                   "inputs": subset, "runs": reports, "equivalence": checks,
                                   "all_passed": all(row["passed"] for row in checks),
                                   "scope": "Diagnostic unchanged-input runtime benchmark; no scientific campaign or model call"})
                        print(json.dumps({"finished": label, "status": report["status"],
                                          "wall_seconds": report.get("evaluation_wall_seconds")}), flush=True)
    return 0 if all(row["passed"] for row in compare_results(reports)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
