from __future__ import annotations

import hashlib
import json
import os
import resource
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from agent_verifier.vpfp.baselines import load_solver
from agent_verifier.vpfp.continuous_macros import (
    MACRO_OUTPUT_CONTRACT,
)
from agent_verifier.vpfp.io import assert_public_bundle_isolated
from agent_verifier.vpfp.resource_limits import wall_time_limit
from agent_verifier.vpfp.verifier import evaluate_prediction, write_feedback


BLIND_CONTEXT_MODE = "blind_public_only"
INSTRUCTION_ONLY_CONTEXT_MODE = "blind_instruction_only"
ORACLE_ASSISTED_CONTEXT_MODE = "oracle_assisted_training"
LEGACY_CONTEXT_MODE = "legacy_refinement"
PUBLIC_ONLY_CONTEXT_MODES = {BLIND_CONTEXT_MODE, INSTRUCTION_ONLY_CONTEXT_MODE}
REDACTED_CONTEXT_MODES = PUBLIC_ONLY_CONTEXT_MODES | {ORACLE_ASSISTED_CONTEXT_MODE}
SOLVER_PRIVATE_QUESTION_KEYS = {
    "source_origin",
    "source_family",
    "generator_backend",
    "generator_representation",
}
BLIND_FORBIDDEN_TEXT = {
    "generated_continuous_autodiff_residual",
    "generated_solution_residual",
    "randomized_public_equation",
    "generator_backend",
    "generator_representation",
    "source_origin",
    "reward_guided",
    "diffusion",
    "ddpo",
    "hidden_truth",
    "rho_true",
    "g_true",
    "selection_score",
    "reward_table",
    "previous hidden-verifier",
}
BLIND_ALLOWED_ROOT_FILES = {
    "README.md",
    "solver_artifact_bundle.example.json",
    "case_manifest.json",
    "generator_reservoir_manifest.json",
    "codex_submission_schema.json",
    "prompt.md",
    "public_eval.py",
    "generator_sample.py",
    "public_predict.py",
    "public_vpfp_runtime.py",
    "solver_agent_instructions.md",
    "solver_instruction_preflight.sh",
    "solver_tool_preflight.sh",
    "solver_contract.md",
    "starter_submission.py",
}
PUBLIC_RUNTIME_DIR = Path(__file__).resolve().parent


def evaluate_solver_on_case_dirs(
    case_dirs: list[str | Path],
    solver_path_or_name: str,
    out_dir: str | Path,
    *,
    max_wall_time_seconds_per_case: float | None = 600.0,
) -> dict[str, Any]:
    solver_root = _solver_bundle_root(solver_path_or_name)
    if solver_root is not None:
        _drop_conflicting_solver_modules(solver_root)
    try:
        evaluation_started = time.perf_counter()
        process_cpu_started = time.process_time()
        load_started = time.perf_counter()
        solver = load_solver(solver_path_or_name)
        solver_load_seconds = time.perf_counter() - load_started
        rows = []
        solver_times = []
        for case_dir in case_dirs:
            solve_started = time.perf_counter()
            try:
                with wall_time_limit(max_wall_time_seconds_per_case):
                    output = solver(Path(case_dir))
                solver_seconds = time.perf_counter() - solve_started
                row = evaluate_prediction(case_dir, output)
            except Exception as exc:  # noqa: BLE001 - verifier feedback.
                solver_seconds = time.perf_counter() - solve_started
                row = {
                    "case_id": Path(case_dir).name,
                    "agent_status": "failed",
                    "exception": repr(exc),
                    "score_smaller_is_better": float("inf"),
                }
            row["solver_wall_time_seconds"] = float(solver_seconds)
            rows.append(row)
            solver_times.append(float(solver_seconds))
        solver_call_total_seconds = float(sum(solver_times))
        process_cpu_seconds = time.process_time() - process_cpu_started
        pipeline_wall_seconds = time.perf_counter() - evaluation_started
        solver_end_to_end_seconds = (
            float(solver_load_seconds) + solver_call_total_seconds
        )
        feedback = write_feedback(
            rows,
            out_dir,
            summary_extra={
                "solver_load_wall_time_seconds": float(solver_load_seconds),
                "solver_call_total_wall_time_seconds": solver_call_total_seconds,
                "solver_call_mean_wall_time_seconds": float(
                    np.mean(solver_times) if solver_times else 0.0
                ),
                "solver_call_max_wall_time_seconds": float(
                    max(solver_times, default=0.0)
                ),
                "solver_evaluation_wall_time_seconds": solver_end_to_end_seconds,
                "evaluation_pipeline_wall_time_seconds": float(
                    pipeline_wall_seconds
                ),
                "solver_process_cpu_time_seconds": float(process_cpu_seconds),
                "solver_process_cpu_to_wall_ratio": float(
                    process_cpu_seconds / max(solver_end_to_end_seconds, 1.0e-12)
                ),
                "solver_hard_wall_time_seconds_per_case": (
                    max_wall_time_seconds_per_case
                ),
                "solver_peak_rss_mib": _peak_rss_mib(),
                "evaluation_allocation": _evaluation_allocation_metadata(),
            },
        )
    finally:
        if solver_root is not None:
            _drop_solver_modules_from_root(solver_root)
    return feedback


def _solver_bundle_root(path_or_name: str) -> Path | None:
    if path_or_name in {"truth", "hold", "hold_initial"}:
        return None
    path = Path(path_or_name).resolve()
    return path.parent if path.is_file() else None


def _drop_conflicting_solver_modules(root: Path) -> None:
    top_level_names = {path.stem for path in root.glob("*.py")}
    top_level_names.update(
        path.name for path in root.iterdir() if (path / "__init__.py").is_file()
    )
    for name in list(sys.modules):
        if name.split(".", 1)[0] in top_level_names:
            sys.modules.pop(name, None)


def _drop_solver_modules_from_root(root: Path) -> None:
    resolved_root = root.resolve()
    for name, module in list(sys.modules.items()):
        module_file = getattr(module, "__file__", None)
        if not module_file:
            continue
        try:
            Path(module_file).resolve().relative_to(resolved_root)
        except (OSError, ValueError):
            continue
        sys.modules.pop(name, None)


def _peak_rss_mib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return value / divisor


def _evaluation_allocation_metadata() -> dict[str, Any]:
    return {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "slurm_mem_per_node": os.environ.get("SLURM_MEM_PER_NODE"),
        "python_executable": sys.executable,
    }


def export_public_solver_workspace(
    case_dirs: list[str | Path],
    out_dir: str | Path,
    *,
    context_mode: str = LEGACY_CONTEXT_MODE,
    generator_case_dirs: list[str | Path] | None = None,
) -> Path:
    if context_mode not in {*REDACTED_CONTEXT_MODES, LEGACY_CONTEXT_MODE}:
        raise ValueError(f"unsupported solver context mode: {context_mode}")
    blind = context_mode in REDACTED_CONTEXT_MODES
    out = Path(out_dir)
    cases_root = out / "cases"
    cases_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    output_contracts: set[str] = set()
    for case_index, case_dir in enumerate(case_dirs):
        src = Path(case_dir)
        solver_case_id = f"case_{case_index:06d}" if blind else src.name
        dst = cases_root / solver_case_id
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)
        question = json.loads(
            (src / "public_question.json").read_text(encoding="utf-8")
        )
        output_contracts.add(
            str(question.get("solver_output_contract_version") or "full_state")
        )
        if blind:
            question = _solver_visible_question(question, solver_case_id)
            (dst / "public_question.json").write_text(json.dumps(question, indent=2, sort_keys=True) + "\n")
            shutil.copy2(src / "public_arrays.npz", dst / "public_arrays.npz")
        else:
            for name in ("public_question.json", "public_arrays.npz"):
                shutil.copy2(src / name, dst / name)
        assert_public_bundle_isolated(dst)
        manifest.append({"case_id": solver_case_id, "case_dir": f"cases/{solver_case_id}"})
    (out / "case_manifest.json").write_text(
        json.dumps({"context_mode": context_mode, "cases": manifest}, indent=2, sort_keys=True) + "\n"
    )
    generator_manifest = []
    if generator_case_dirs:
        reservoir_root = out / "generator_reservoir" / "cases"
        reservoir_root.mkdir(parents=True)
        for case_index, case_dir in enumerate(generator_case_dirs):
            src = Path(case_dir)
            solver_case_id = f"generator_case_{case_index:06d}"
            dst = reservoir_root / solver_case_id
            dst.mkdir()
            question = json.loads((src / "public_question.json").read_text())
            question = _solver_visible_question(question, solver_case_id)
            (dst / "public_question.json").write_text(
                json.dumps(question, indent=2, sort_keys=True) + "\n"
            )
            shutil.copy2(src / "public_arrays.npz", dst / "public_arrays.npz")
            assert_public_bundle_isolated(dst)
            generator_manifest.append(
                {
                    "case_id": solver_case_id,
                    "case_dir": f"generator_reservoir/cases/{solver_case_id}",
                }
            )
        (out / "generator_reservoir_manifest.json").write_text(
            json.dumps(
                {
                    "schema": "vpfp_public_generator_reservoir_v1",
                    "sampling_contract": "finite_frozen_public_reservoir",
                    "cases": generator_manifest,
                },
                indent=2,
                sort_keys=True,
            ) + "\n"
        )
    if blind:
        shutil.copy2(PUBLIC_RUNTIME_DIR / "public_vpfp_runtime.py", out / "public_vpfp_runtime.py")
        shutil.copy2(PUBLIC_RUNTIME_DIR / "public_eval_cli.py", out / "public_eval.py")
        shutil.copy2(PUBLIC_RUNTIME_DIR / "public_predict_cli.py", out / "public_predict.py")
    if len(output_contracts) != 1:
        raise ValueError(
            "public solver workspace requires one homogeneous output contract"
        )
    if output_contracts == {MACRO_OUTPUT_CONTRACT}:
        readme_contract = (
            "Implement a solver returning {'rho': rho_pred, 'momentum': "
            "momentum_pred, 'kinetic_energy': kinetic_energy_pred}. The full "
            "kinetic source and initial state are public, but full g(t) is not "
            "a required output.\n"
        )
    else:
        readme_contract = (
            "Implement a solver returning {'rho': rho_pred, 'g': g_pred}. "
            "The evaluator derives phi/E from rho and combines rho*g for "
            "kinetic distribution metrics.\n"
        )
    (out / "README.md").write_text(
        "# VPFP Public Solver Workspace\n\n"
        + readme_contract
    )
    return out


def audit_blind_solver_workspace(workspace: str | Path) -> dict[str, Any]:
    root = Path(workspace)
    leaks: list[str] = []
    files: list[dict[str, Any]] = []
    context_mode = BLIND_CONTEXT_MODE
    if not root.exists():
        return {
            "passed": False,
            "context_mode": context_mode,
            "files": [],
            "leaks": [f"missing workspace: {root}"],
        }
    manifest_path = root / "case_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        declared_mode = str(manifest.get("context_mode", BLIND_CONTEXT_MODE))
        if declared_mode in PUBLIC_ONLY_CONTEXT_MODES:
            context_mode = declared_mode
    for item in sorted(root.rglob("*")):
        if item.is_symlink():
            leaks.append(f"symlink is not allowed: {item.relative_to(root)}")
            continue
        if not item.is_file():
            continue
        relative = item.relative_to(root)
        if relative.parts[0] == "cases":
            allowed = len(relative.parts) == 3 and relative.name in {"public_question.json", "public_arrays.npz"}
        elif relative.parts[0] == "generator_reservoir":
            allowed = (
                len(relative.parts) == 4
                and relative.parts[1] == "cases"
                and relative.name in {"public_question.json", "public_arrays.npz"}
            )
        else:
            allowed = len(relative.parts) == 1 and relative.name in BLIND_ALLOWED_ROOT_FILES
        if not allowed:
            leaks.append(f"unexpected solver-visible file: {relative}")
        payload = item.read_bytes()
        files.append(
            {
                "path": str(relative),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
        if item.suffix == ".npz":
            with np.load(item) as arrays:
                unexpected = set(arrays.files) - {
                    "x",
                    "y",
                    "vx",
                    "vy",
                    "t",
                    "velocity_weights",
                    "rho0",
                    "g0",
                    "source",
                }
            if unexpected:
                leaks.append(f"unexpected public array keys in {relative}: {sorted(unexpected)}")
        elif item.suffix in {".json", ".md", ".py"}:
            lowered = payload.decode("utf-8", errors="replace").lower()
            for token in sorted(BLIND_FORBIDDEN_TEXT):
                if token in lowered:
                    leaks.append(f"forbidden generation/evaluation token {token!r} in {relative}")
    generator_manifest_path = root / "generator_reservoir_manifest.json"
    generator_manifest: dict[str, Any] = {}
    if generator_manifest_path.exists():
        try:
            if not generator_manifest_path.is_file() or generator_manifest_path.is_symlink():
                raise ValueError("generator reservoir manifest must be a regular file")
            generator_manifest = json.loads(generator_manifest_path.read_text())
            if generator_manifest.get("schema") != "vpfp_public_generator_reservoir_v1":
                raise ValueError("unexpected generator reservoir schema")
            declared_dirs = set()
            for row in generator_manifest.get("cases", []):
                neutral_id = str(row.get("case_id") or "")
                relative_case = Path(str(row.get("case_dir") or ""))
                case_dir = root / relative_case
                assert_public_bundle_isolated(case_dir)
                question = json.loads((case_dir / "public_question.json").read_text())
                if str(question.get("case_id")) != neutral_id:
                    leaks.append(f"generator reservoir case ID mismatch: {neutral_id}")
                leaked_keys = SOLVER_PRIVATE_QUESTION_KEYS.intersection(question)
                if leaked_keys:
                    leaks.append(
                        f"private keys in generator reservoir {neutral_id}: {sorted(leaked_keys)}"
                    )
                declared_dirs.add(relative_case)
            observed_dirs = {
                path.relative_to(root)
                for path in (root / "generator_reservoir" / "cases").iterdir()
                if path.is_dir() and not path.is_symlink()
            }
            if declared_dirs != observed_dirs:
                leaks.append("generator reservoir manifest/directory mismatch")
        except Exception as exc:  # noqa: BLE001 - retained in isolation report.
            leaks.append(f"invalid public generator reservoir: {exc!r}")
    return {
        "passed": not leaks,
        "context_mode": context_mode,
        "public_generator_case_count": len(generator_manifest.get("cases", [])),
        "files": files,
        "leaks": leaks,
    }


def assert_blind_solver_workspace_isolated(workspace: str | Path) -> dict[str, Any]:
    report = audit_blind_solver_workspace(workspace)
    if not report["passed"]:
        raise AssertionError("VPFP blind solver workspace leak detected:\n" + "\n".join(report["leaks"]))
    return report


def _solver_visible_question(question: dict[str, Any], solver_case_id: str) -> dict[str, Any]:
    visible = json.loads(json.dumps(question))
    visible["case_id"] = solver_case_id
    for key in SOLVER_PRIVATE_QUESTION_KEYS:
        visible.pop(key, None)
    return visible
