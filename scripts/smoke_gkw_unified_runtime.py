#!/usr/bin/env python3
"""Real selected-GKW smoke for the unified clean/perturbed Generator ABI."""

from __future__ import annotations

from contextlib import contextmanager
import argparse
from collections.abc import Mapping
import errno
import hashlib
import importlib
import importlib.metadata
import json
import platform
import resource
from time import perf_counter
from pathlib import Path
import shutil
import traceback

import numpy as np
import torch

from agent_verifier.fusion_simulation_first.gkw_selected_service_factory import (
    GKWSelectedServiceArtifacts,
    build_gkw_selected_online_service,
)
from agent_verifier.fusion_simulation_first import (
    gkw_perturbation_qualification as qualification_module,
)
from pde_agent_verifier_solver.engine import (
    ControllerMacroEvaluator,
    PreparedMacroCohort,
)
from pde_agent_verifier_solver.engine.macro_query_plan import (
    ControllerMacroQueryPlan,
)
from pde_agent_verifier_solver.plugins.gkw import (
    GKWAnswerFirstGenerator,
    GKWMacroMatcher,
    GKWManufacturedGeneratorBackend,
    GKWSelectedTorchRuntime,
    GKW_PUBLIC_SESSION_METHODS,
)
from pde_agent_verifier_solver.plugins.gkw.contract import (
    gkw_public_query_domain,
)
from pde_agent_verifier_solver.plugins.gkw.macro_loop import (
    gkw_macro_point_query,
)
from pde_agent_verifier_solver.sdk import (
    BoundMacroObjective,
    CohortKind,
    FixedGeneratorCaseSource,
    GenerationKind,
    QueryKind,
    SampleRequest,
    open_public_case_session,
)


_ARTIFACT_FIELDS = (
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
_MINIMUM_DURABLE_FREE_BYTES = 1 << 30
_PUBLIC_CONDITION_NAMES = ("rlt_i", "rln_i", "shat", "q")
_PHYSICAL_MODULES = (
    "gkw_continuous_state",
    "gkw_generated_pde_bridge",
    "gkw_macro_contract",
    "gkw_perturbed_macro_truth",
    "gkw_numerical_oracle",
    "gkw_online_bridge",
    "gkw_selected_generator_service",
    "gkw_selected_service_factory",
    "gkw_smooth_perturbation",
    "gkw_perturbation_qualification",
)
_RUNTIME_DISTRIBUTIONS = (
    "kappamodules",
    "einops",
    "omegaconf",
    "antlr4-python3-runtime",
    "PyYAML",
)
_SERVICE_RUNTIME_POLICY = {
    "device": "cuda",
    "decoder_precision": "bfloat16",
    "reward_bootstrap_samples": 32,
    "reward_seed_namespace": "gkw-perturbation-qualification",
    "macro_time_count": 3,
    "macro_field_rows": 16,
    "formal_macro_time_count": 4,
    "maximum_sampling_attempts": 8,
    "oracle_max_total_rows": 20_000_000,
    "enforce_distribution_calibration": True,
}
_SMOKE_COHORTS = ("development", "selection")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--gyaradax-root", type=Path, required=True)
    parser.add_argument("--physical-commit", required=True)
    parser.add_argument("--physical-root", type=Path, required=True)
    parser.add_argument("--unified-commit", required=True)
    parser.add_argument(
        "--qualification-certificate", type=Path, required=True
    )
    for name in (
        "qualification_report",
        "generator_manifest",
        "operator_manifest",
        "physical_source_manifest",
        "seed_plan",
    ):
        parser.add_argument(
            "--" + name.replace("_", "-"), type=Path, required=True
        )
    for name in _ARTIFACT_FIELDS:
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--lifecycle-root-count", type=_positive_int, default=1)
    parser.add_argument(
        "--cohort",
        choices=_SMOKE_COHORTS,
        default="development",
        help="manufacture and evaluate a development or formal selection cohort",
    )
    return parser.parse_args()


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _require_free_bytes(root: Path, minimum: int, label: str) -> int:
    available = int(shutil.disk_usage(root).free)
    if available < minimum:
        raise OSError(
            errno.ENOSPC,
            f"{label} has {available} free bytes; at least {minimum} are required",
            str(root),
        )
    return available


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_identity(args, certificate) -> tuple[str, dict[str, object]]:
    artifacts = {
        name: {
            "path": str(getattr(args, name).resolve()),
            "sha256": _sha256_path(getattr(args, name).resolve()),
        }
        for name in _ARTIFACT_FIELDS
    }
    sources = {}
    prefix = "agent_verifier.fusion_simulation_first."
    for name in _PHYSICAL_MODULES:
        module_path = Path(importlib.import_module(prefix + name).__file__).resolve()
        sources[name] = {
            "path": str(module_path),
            "sha256": _sha256_path(module_path),
        }
    runtime = {
        "python": platform.python_version(),
        "numpy": str(np.__version__),
        "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "installed_distributions": {
            name: importlib.metadata.version(name)
            for name in _RUNTIME_DISTRIBUTIONS
        },
    }
    identity = {
        "schema": "gkw_unified_selected_generator_identity_v2",
        "artifacts": {name: row["sha256"] for name, row in artifacts.items()},
        "physical_sources": {name: row["sha256"] for name, row in sources.items()},
        "physical_commit": args.physical_commit,
        "unified_commit": args.unified_commit,
        "external_commit": "568411b33be486262b8c5cfe1dfb3e0fe97f6419",
        "gyaradax_commit": "318e3d11d7326d99e96f2bd5a8998a462261dcd4",
        "decoder_precision": _SERVICE_RUNTIME_POLICY["decoder_precision"],
        "runtime": runtime,
        "service_runtime_policy": dict(_SERVICE_RUNTIME_POLICY),
        "perturbation_qualification": (
            qualification_module.gkw_qualification_certificate_payload(certificate)
        ),
    }
    manifest = {
        "schema": "gkw_unified_selected_generator_artifact_manifest_v1",
        "artifact_locations": artifacts,
        "physical_source_locations": sources,
        "identity": identity,
    }
    encoded = json.dumps(
        identity, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    generator_sha256 = hashlib.sha256(encoded).hexdigest()
    manifest["generator_sha256"] = generator_sha256
    return generator_sha256, manifest


def _equal_public_geometry(left, right) -> bool:
    for group in ("arrays", "scalars"):
        a = left.public_data[group]
        b = right.public_data[group]
        if set(a) != set(b):
            return False
        if any(
            np.asarray(a[name]).dtype != np.asarray(b[name]).dtype
            or not np.array_equal(np.asarray(a[name]), np.asarray(b[name]))
            for name in a
        ):
            return False
    if gkw_public_query_domain(left) != gkw_public_query_domain(right):
        return False
    return np.array_equal(
        np.asarray(left.public_data["metadata"]["time_nodes"]),
        np.asarray(right.public_data["metadata"]["time_nodes"]),
    )


def _public_condition(case) -> np.ndarray:
    scalars = case.public_data["scalars"]
    condition = np.asarray(
        [scalars[name] for name in _PUBLIC_CONDITION_NAMES],
        dtype=np.float64,
    )
    if condition.shape != (4,) or not bool(np.isfinite(condition).all()):
        raise RuntimeError("GKW public continuous condition is invalid")
    return condition


def _continuous_condition_admission(
    clean_records,
    child_records,
    archived_cases,
    conditioning_std,
) -> dict[str, object]:
    """Negative control against finite-condition lookup and child drift."""

    clean = np.stack([_public_condition(row.case) for row in clean_records])
    child = np.stack([_public_condition(row.case) for row in child_records])
    archive = np.asarray(
        [
            [getattr(case, name) for name in _PUBLIC_CONDITION_NAMES]
            for case in archived_cases.values()
        ],
        dtype=np.float64,
    )
    if (
        clean.shape != child.shape
        or clean.shape[1:] != (4,)
        or archive.ndim != 2
        or archive.shape[1:] != (4,)
        or archive.shape[0] == 0
        or not bool(np.isfinite(archive).all())
    ):
        raise RuntimeError("GKW continuous-condition admission inputs are invalid")
    scale = np.asarray(conditioning_std, dtype=np.float64)
    if (
        scale.shape != (4,)
        or bool(np.any(scale <= 0.0))
        or not bool(np.isfinite(scale).all())
    ):
        raise RuntimeError("GKW archive conditions cannot be standardized")
    exact = np.all(clean[:, None, :] == archive[None, :, :], axis=2)
    nearest = np.linalg.norm(
        (clean[:, None, :] - archive[None, :, :]) / scale[None, None, :],
        axis=2,
    ).min(axis=1)
    child_matches = np.all(clean == child, axis=1)
    hit_count = int(np.any(exact, axis=1).sum())
    match_count = int(child_matches.sum())
    passed = (
        hit_count == 0
        and match_count == clean.shape[0]
        and bool(np.isfinite(nearest).all())
        and bool(np.all(nearest > 0.0))
    )
    return {
        "schema": "gkw_continuous_condition_admission_v1",
        "condition_names": list(_PUBLIC_CONDITION_NAMES),
        "standardization": "generator_geometry_sidecar",
        "clean_condition_count": int(clean.shape[0]),
        "archive_condition_count": int(archive.shape[0]),
        "exact_archive_lookup_hit_count": hit_count,
        "minimum_nearest_standardized_distance": float(nearest.min()),
        "child_condition_match_count": match_count,
        "passed": passed,
    }


def _phase_rows(count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    extents = (32, 8, 16, 85, 32)
    return np.column_stack(
        [rng.integers(0, extent, size=count) for extent in extents]
    ).astype(np.int64)


def _field_rows(count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    extents = (16, 85, 32)
    return np.column_stack(
        [rng.integers(0, extent, size=count) for extent in extents]
    ).astype(np.int64)


def _cuda_memory() -> dict[str, int]:
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
    }


def _valid_cuda_memory(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "allocated_bytes",
        "reserved_bytes",
    }:
        return False
    return all(
        not isinstance(value[name], bool)
        and isinstance(value[name], int)
        and value[name] >= 0
        for name in ("allocated_bytes", "reserved_bytes")
    )


def _proc_key_values(path: Path) -> dict[str, int]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RuntimeError(f"cannot read required process telemetry: {path}") from exc
    values: dict[str, int] = {}
    for line in lines:
        name, separator, raw = line.partition(":")
        if not separator:
            continue
        fields = raw.strip().split()
        if not fields:
            continue
        try:
            value = int(fields[0])
        except ValueError:
            continue
        if len(fields) == 2 and fields[1] == "kB":
            value *= 1024
        values[name] = value
    return values


def _process_memory_snapshot() -> dict[str, int]:
    status = _proc_key_values(Path("/proc/self/status"))
    try:
        current_rss = status["VmRSS"]
        peak_rss = status["VmHWM"]
    except KeyError as exc:
        raise RuntimeError("/proc/self/status lacks RSS telemetry") from exc
    return {
        "current_rss_bytes": current_rss,
        "peak_rss_bytes": peak_rss,
        "ru_maxrss_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        ),
    }


def _process_io_snapshot() -> dict[str, int]:
    return _proc_key_values(Path("/proc/self/io"))


@contextmanager
def _measured_section(
    sections: dict[str, dict[str, object]],
    name: str,
):
    if not isinstance(name, str) or not name.strip() or name in sections:
        raise ValueError("smoke section names must be unique nonempty strings")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    memory_before = _process_memory_snapshot()
    io_before = _process_io_snapshot()
    cuda_before = _cuda_memory()
    started = perf_counter()
    completed = False
    try:
        yield
        completed = True
    finally:
        torch.cuda.synchronize()
        memory_after = _process_memory_snapshot()
        io_after = _process_io_snapshot()
        sections[name] = {
            "completed": completed,
            "wall_seconds": max(0.0, perf_counter() - started),
            "memory_before": memory_before,
            "memory_after": memory_after,
            "process_io_before": io_before,
            "process_io_after": io_after,
            "process_io_delta": {
                key: io_after.get(key, 0) - io_before.get(key, 0)
                for key in sorted(set(io_before) | set(io_after))
            },
            "cuda": {
                "allocated_before_bytes": cuda_before["allocated_bytes"],
                "reserved_before_bytes": cuda_before["reserved_bytes"],
                "allocated_after_bytes": int(torch.cuda.memory_allocated()),
                "reserved_after_bytes": int(torch.cuda.memory_reserved()),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            },
        }



def _record_private_exception(
    row: dict[str, object], stage: str, exc: BaseException
) -> None:
    row["controller_private_exceptions"].append(
        {
            "stage": stage,
            "type": type(exc).__name__,
            "message": str(exc),
        }
    )


class _AuditedPublicSession:
    """Controller-owned lifecycle probe around one real public session."""

    def __init__(self, target, row: dict[str, object]) -> None:
        self._target = target
        self._row = row
        self._close_attempted = False

    def query_initial_phi(self, field_indices):
        try:
            return self._target.query_initial_phi(field_indices)
        except BaseException as exc:
            _record_private_exception(self._row, "query_initial_phi", exc)
            raise
        finally:
            self._row["cuda_after_oracle_trigger"] = _cuda_memory()

    def close(self) -> None:
        if self._close_attempted:
            return
        self._close_attempted = True
        self._row["session_close_attempted"] = True
        target = self._target
        try:
            close = getattr(target, "close", None)
            if callable(close):
                close()
        except BaseException as exc:
            _record_private_exception(self._row, "session_close", exc)
            raise
        finally:
            self._target = None
            self._row["session_reference_released"] = True
            self._row["cuda_after_session_close"] = _cuda_memory()


class _ControllerLifecycleAudit:
    """Keep diagnostic exceptions and CUDA observations outside candidates."""

    def __init__(self, public_session_factory) -> None:
        self._public_session_factory = public_session_factory
        self._candidate_id: str | None = None
        self.rows: list[dict[str, object]] = []

    def begin_candidate(self, candidate_id: str) -> None:
        if self._candidate_id is not None:
            raise RuntimeError("a lifecycle candidate evaluation is already active")
        self._candidate_id = candidate_id

    def end_candidate(self, candidate_id: str) -> None:
        if self._candidate_id != candidate_id:
            raise RuntimeError("lifecycle candidate evaluation identity changed")
        self._candidate_id = None

    def public_session_factory(self, case):
        if self._candidate_id is None:
            raise RuntimeError("public session opened outside a lifecycle evaluation")
        row: dict[str, object] = {
            "candidate_id": self._candidate_id,
            "case_id": case.case_id,
            "cuda_before_session_open": _cuda_memory(),
            "cuda_after_session_factory": None,
            "cuda_after_oracle_trigger": None,
            "cuda_after_session_close": None,
            "session_close_attempted": False,
            "session_reference_released": False,
            "controller_private_exceptions": [],
        }
        self.rows.append(row)
        try:
            target = self._public_session_factory(case)
        except BaseException as exc:
            _record_private_exception(row, "session_factory", exc)
            row["cuda_after_session_factory"] = _cuda_memory()
            raise
        row["cuda_after_session_factory"] = _cuda_memory()
        return _AuditedPublicSession(target, row)


class _SnapshotSessionProgram:
    def __init__(self, answers, lifecycle_audit, candidate_id: str) -> None:
        self._answers = dict(answers)
        self._lifecycle_audit = lifecycle_audit
        self._candidate_id = candidate_id

    def predict_macros(self, case, query, budget=None):
        del query, budget
        self._lifecycle_audit.begin_candidate(self._candidate_id)
        try:
            session = open_public_case_session(case)
            session.query_initial_phi(_field_rows(1, 19))
            return self._answers[case.case_id]
        finally:
            self._lifecycle_audit.end_candidate(self._candidate_id)


def main() -> None:
    args = _arguments()
    cohort_kind = CohortKind(args.cohort)
    query_profile = args.cohort
    run_root = args.run_root.resolve()
    storage_root = args.storage_root.resolve()
    if not storage_root.is_dir():
        raise FileNotFoundError(f"smoke storage root is absent: {storage_root}")
    if run_root == storage_root or not run_root.is_relative_to(storage_root):
        raise ValueError("GKW unified smoke root must be below storage_root")
    _require_free_bytes(
        storage_root,
        _MINIMUM_DURABLE_FREE_BYTES,
        "GKW smoke durable storage",
    )
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError("GKW unified smoke root is not empty")
    run_root.mkdir(parents=True, exist_ok=True)
    for name in _ARTIFACT_FIELDS:
        path = getattr(args, name).resolve()
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)

    qualification_paths = {
        name: getattr(args, name).resolve()
        for name in (
            "qualification_certificate",
            "qualification_report",
            "generator_manifest",
            "operator_manifest",
            "physical_source_manifest",
            "seed_plan",
        )
    }
    for label, path in qualification_paths.items():
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(
                f"GKW production {label} is absent"
            )
    certificate_path = qualification_paths["qualification_certificate"]
    certificate = qualification_module.gkw_qualification_certificate_from_payload(
        json.loads(certificate_path.read_text(encoding="utf-8"))
    )
    if (
        certificate.physical_git_commit != args.physical_commit
        or certificate.qualification_code_commit != args.physical_commit
    ):
        raise RuntimeError(
            "GKW qualification certificate does not bind the physical commit"
        )
    qualification_validation_started = perf_counter()
    production_qualification = (
        qualification_module.validate_gkw_production_qualification_environment(
            certificate,
            qualification_report_path=qualification_paths[
                "qualification_report"
            ],
            generator_manifest_path=qualification_paths["generator_manifest"],
            operator_manifest_path=qualification_paths["operator_manifest"],
            physical_source_manifest_path=qualification_paths[
                "physical_source_manifest"
            ],
            seed_plan_path=qualification_paths["seed_plan"],
            generator_files={
                name: getattr(args, name).resolve()
                for name in qualification_module.GKW_QUALIFICATION_GENERATOR_ARTIFACT_KEYS
            },
            operator_files={
                name: getattr(args, name).resolve()
                for name in qualification_module.GKW_QUALIFICATION_OPERATOR_ARTIFACT_KEYS
            },
            physical_code_root=args.physical_root.resolve(),
            external_commits={
                "neural_gyrokinetics": (
                    "568411b33be486262b8c5cfe1dfb3e0fe97f6419"
                ),
                "gyaradax": (
                    "318e3d11d7326d99e96f2bd5a8998a462261dcd4"
                ),
            },
        )
    )
    qualification_validation_wall_seconds = max(
        0.0, perf_counter() - qualification_validation_started
    )
    service = build_gkw_selected_online_service(
        artifacts=GKWSelectedServiceArtifacts(
            external_root=args.external_root.resolve(),
            gyaradax_root=args.gyaradax_root.resolve(),
            **{name: getattr(args, name).resolve() for name in _ARTIFACT_FIELDS},
        ),
        private_root=run_root / "selected_service",
        private_storage_root=storage_root,
        device=_SERVICE_RUNTIME_POLICY["device"],
        decoder_inference_precision=_SERVICE_RUNTIME_POLICY["decoder_precision"],
        reward_bootstrap_samples=_SERVICE_RUNTIME_POLICY["reward_bootstrap_samples"],
        reward_seed_namespace="gkw-perturbation-qualification",
    )
    actual_decoder_precision = str(
        service.decoder.config.neural_inference_precision
    )
    if actual_decoder_precision != _SERVICE_RUNTIME_POLICY["decoder_precision"]:
        raise RuntimeError("selected GKW decoder precision changed")
    artifact_sha256, manifest = _artifact_identity(args, certificate)
    (run_root / "generator_artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sections: dict[str, dict[str, object]] = {}
    runtime = GKWSelectedTorchRuntime(
        service,
        implementation_sha256=artifact_sha256,
        production_qualification=production_qualification,
        private_root=run_root / "unified_runtime",
        private_storage_root=storage_root,
        macro_time_count=_SERVICE_RUNTIME_POLICY["macro_time_count"],
        macro_field_rows=_SERVICE_RUNTIME_POLICY["macro_field_rows"],
        formal_macro_time_count=_SERVICE_RUNTIME_POLICY[
            "formal_macro_time_count"
        ],
        maximum_sampling_attempts=_SERVICE_RUNTIME_POLICY["maximum_sampling_attempts"],
        oracle_max_total_rows=_SERVICE_RUNTIME_POLICY["oracle_max_total_rows"],
        enforce_distribution_calibration=_SERVICE_RUNTIME_POLICY[
            "enforce_distribution_calibration"
        ],
    )
    generator = GKWAnswerFirstGenerator(GKWManufacturedGeneratorBackend(runtime))
    source = FixedGeneratorCaseSource(
        generator_id=generator.info.program_id,
        artifact_sha256=artifact_sha256,
        generator=generator,
        base_state=None,
    )
    manufacturing_before = dict(runtime.private_manufacturing_telemetry)
    with _measured_section(sections, "clean_generation"):
        clean_records = source.sample(
            SampleRequest(
                1,
                cohort_kind,
                args.seed,
                args.lifecycle_root_count,
            )
        )
    with _measured_section(sections, "perturbed_generation"):
        child_records = source.sample(
            SampleRequest(
                1,
                cohort_kind,
                args.seed + 1,
                args.lifecycle_root_count,
                kind=GenerationKind.PERTURBED,
                anchors=tuple(row.case for row in clean_records),
            )
        )
    manufacturing_after_generation = dict(
        runtime.private_manufacturing_telemetry
    )
    continuous_condition_admission = _continuous_condition_admission(
        clean_records,
        child_records,
        service.archived_cases,
        service.decoder.geometry_sidecar["conditioning_std"],
    )
    zero_online_work_fields = (
        "perturbed_online_numerical_audits",
        "perturbed_online_parameter_hashes",
        "perturbed_failed_seed_replacements",
        "unified_serialization_calls",
        "unified_state_hash_calls",
        "unified_replay_calls",
    )
    lean_online_work_zero = all(
        manufacturing_after_generation[name] == 0
        for name in zero_online_work_fields
    )
    if not lean_online_work_zero:
        raise RuntimeError(
            "GKW qualified perturbation performed forbidden online audit work"
        )
    clean_decoder_rejections = manufacturing_after_generation[
        "clean_decoder_rejections_before_physical_audit"
    ]
    clean_audited_observations = manufacturing_after_generation[
        "clean_audited_observations"
    ]
    for clean_record, child_record in zip(clean_records, child_records):
        if child_record.case.parent_case_id != clean_record.case.case_id:
            raise RuntimeError("GKW unified child lineage changed")
        if not _equal_public_geometry(clean_record.case, child_record.case):
            raise RuntimeError("GKW clean/child public geometry changed")
        clean_handle = runtime._handles[clean_record.case.case_id]
        child_handle = runtime._handles[child_record.case.case_id]
        if child_handle.parent_answer_id != clean_handle.answer_id:
            raise RuntimeError("GKW resident child lineage changed")
        if (
            child_handle.resident.base_storage_identity
            != clean_handle.resident.base_storage_identity
        ):
            raise RuntimeError("GKW child did not share clean CPU base storage")
        if (
            clean_handle.resident.operator_resident
            or child_handle.resident.operator_resident
        ):
            raise RuntimeError("GKW resident answers unexpectedly retained operators")
        perturbation_provenance = (
            child_handle.resident.qualification_provenance.get("perturbation")
        )
        if (
            not isinstance(perturbation_provenance, Mapping)
            or perturbation_provenance.get("qualification_id")
            != certificate.qualification_id
            or perturbation_provenance.get("per_case_numerical_audit_performed")
            is not False
            or perturbation_provenance.get("parameter_hash_computed") is not False
        ):
            raise RuntimeError(
                "GKW resident child lost its offline qualification binding"
            )
    clean = clean_records[0]
    child = child_records[0]
    clean_manufactured = runtime._handles[clean.case.case_id]
    child_manufactured = runtime._handles[child.case.case_id]
    clean_answer_id = clean_manufactured.answer_id
    child_answer_id = child_manufactured.answer_id
    child_parent_answer_id = child_manufactured.parent_answer_id
    base_storage_shared = (
        clean_manufactured.resident.base_storage_identity
        == child_manufactured.resident.base_storage_identity
    )
    del clean_manufactured, child_manufactured, clean_handle, child_handle

    phase = _phase_rows(128, args.seed)
    field = _field_rows(32, args.seed)
    nodes = np.asarray(clean.case.public_data["metadata"]["time_nodes"])
    interior = float(0.5 * (nodes[1] + nodes[2]))
    with _measured_section(sections, "first_oracle_query"):
        clean_session = generator.public_session_factory(clean.case)
        try:
            clean_h0 = clean_session.query_initial_h(phase)
            clean_phi0 = clean_session.query_initial_phi(field)
            clean_source = clean_session.query_source(interior, phase)
            clean_session.query_source(interior, phase[:8])
        finally:
            close = getattr(clean_session, "close", None)
            if callable(close):
                close()

    backend_before_probe = dict(runtime.private_oracle_telemetry)
    with _measured_section(sections, "repeated_oracle_query"):
        clean_cached_session = generator.public_session_factory(clean.case)
        try:
            repeated_clean_source = clean_cached_session.query_source(
                interior,
                phase,
            )
        finally:
            close = getattr(clean_cached_session, "close", None)
            if callable(close):
                close()
    backend_after_probe = dict(runtime.private_oracle_telemetry)
    if (
        backend_after_probe["backend_loads"]
        != backend_before_probe["backend_loads"]
        or backend_after_probe["frame_hits"]
        <= backend_before_probe["frame_hits"]
    ):
        raise RuntimeError("GKW shared oracle backend cache probe missed")
    if not np.allclose(
        repeated_clean_source,
        clean_source,
        rtol=0.0,
        atol=0.0,
    ):
        raise RuntimeError("GKW shared oracle backend changed source values")
    clean_backend_cache_probe = {
        "before": backend_before_probe,
        "after": backend_after_probe,
        "backend_load_unchanged": True,
        "shared_frame_hit_increased": True,
    }

    with _measured_section(sections, "perturbed_oracle_query"):
        child_session = generator.public_session_factory(child.case)
        try:
            child_h0 = child_session.query_initial_h(phase)
            child_phi0 = child_session.query_initial_phi(field)
            child_source = child_session.query_source(interior, phase)
        finally:
            close = getattr(child_session, "close", None)
            if callable(close):
                close()
    if not np.allclose(clean_h0, child_h0, rtol=0.0, atol=1.0e-10):
        raise RuntimeError("GKW perturbation changed the initial kinetic state")
    if not np.allclose(clean_phi0, child_phi0, rtol=0.0, atol=1.0e-10):
        raise RuntimeError("GKW perturbation changed the initial field")
    source_delta = float(np.linalg.norm(child_source - clean_source))
    if not np.isfinite(source_delta) or source_delta <= 1.0e-12:
        raise RuntimeError("GKW perturbed source did not change")

    switch_before = dict(runtime.private_oracle_telemetry)
    with _measured_section(sections, "case_switch_oracle_query"):
        cached_session = generator.public_session_factory(clean.case)
        try:
            cached_session.query_initial_phi(field[:1])
        finally:
            close = getattr(cached_session, "close", None)
            if callable(close):
                close()
    switch_after = dict(runtime.private_oracle_telemetry)
    if (
        switch_after["backend_loads"]
        != switch_before["backend_loads"] + 1
        or switch_after["backend_evictions"]
        != switch_before["backend_evictions"] + 1
    ):
        raise RuntimeError("GKW capacity-one backend did not switch cases")
    backend_switch_probe = {
        "before": switch_before,
        "after": switch_after,
    }

    cohort_id = (
        f"gkw-unified-{args.cohort}-lifecycle-"
        f"{args.lifecycle_root_count:04d}-roots"
    )
    matcher = GKWMacroMatcher()
    objective = BoundMacroObjective(
        matcher=matcher,
        query_factory=gkw_macro_point_query,
        controller_query_factory=gkw_macro_point_query,
    )
    query_plan = ControllerMacroQueryPlan(
        cohort_id=cohort_id,
        profile=query_profile,
        controller_seed=args.seed + 2,
    )

    def scoring_query(case):
        return query_plan.build(objective, case)

    query = scoring_query(child.case)
    child_target = matcher.prepare_target(child.case, child.truth.answer(query), query)
    cross_error = matcher.match(
        child.case, clean.truth.answer(query), child_target
    ).primary_error
    if not np.isfinite(cross_error) or cross_error <= 0.0:
        raise RuntimeError("GKW perturbed macro truth did not change")

    root_queries = tuple(
        scoring_query(row.case) for row in clean_records
    )
    if any(query.kind is not QueryKind.POINTS for query in root_queries):
        raise RuntimeError("GKW controller smoke did not use POINTS queries")
    if any(
        scoring_query(clean_record.case)
        != scoring_query(child_record.case)
        for clean_record, child_record in zip(clean_records, child_records)
    ):
        raise RuntimeError("GKW clean/perturbed scoring queries drifted")
    if len({repr(query.points) for query in root_queries}) != len(root_queries):
        raise RuntimeError("GKW controller reused a scoring query across roots")
    query_point_counts = {len(query.points) for query in root_queries}
    if len(query_point_counts) != 1:
        raise RuntimeError("GKW controller query point counts changed by root")
    query_point_count = query_point_counts.pop()

    records = tuple(
        record
        for pair in zip(clean_records, child_records)
        for record in pair
    )
    cohort = PreparedMacroCohort.prepare(
        cohort_id=cohort_id,
        records=records,
        matcher=matcher,
        query=scoring_query,
        release_cases=source.release_cases,
    )
    answers = {
        row.case.case_id: row.truth.answer(scoring_query(row.case))
        for row in records
    }
    lifecycle_audit = _ControllerLifecycleAudit(
        generator.public_session_factory
    )
    evaluator = ControllerMacroEvaluator(
        cohort,
        public_session_factory=lifecycle_audit.public_session_factory,
        public_session_methods=GKW_PUBLIC_SESSION_METHODS,
    )
    candidate_ids = (
        "gkw-session-snapshot-a",
        "gkw-session-snapshot-b",
    )
    candidate_programs = tuple(
        (
            candidate_id,
            _SnapshotSessionProgram(
                answers,
                lifecycle_audit,
                candidate_id,
            ),
        )
        for candidate_id in candidate_ids
    )
    case_major_backend_before = dict(runtime.private_oracle_telemetry)
    try:
        with _measured_section(sections, "case_major_two_candidate"):
            evaluations = evaluator.evaluate_many(candidate_programs)
        case_major_backend_after = dict(runtime.private_oracle_telemetry)
    finally:
        with _measured_section(sections, "cohort_release"):
            cohort.discard()
    oracle_after_release = dict(runtime.private_oracle_telemetry)
    manufacturing_after_release = dict(
        runtime.private_manufacturing_telemetry
    )
    release_passed = (
        oracle_after_release["resident_answers"] == 0
        and oracle_after_release["backend_resident"] == 0
        and oracle_after_release["resident_operators"] == 0
        and manufacturing_after_release["resident_answers_released"]
        >= len(records)
    )
    if not release_passed:
        raise RuntimeError("GKW cohort discard retained resident oracle state")
    backend_snapshots = {
        "before_case_major": case_major_backend_before,
        "after_case_major_before_release": case_major_backend_after,
        "after_release": oracle_after_release,
    }

    expected_case_ids = tuple(row.case.case_id for row in records)
    expected_case_count = 2 * args.lifecycle_root_count
    expected_case_major_order = tuple(
        (case_id, candidate_id)
        for case_id in expected_case_ids
        for candidate_id in candidate_ids
    )
    observed_case_major_order = tuple(
        (row["case_id"], row["candidate_id"])
        for row in lifecycle_audit.rows
    )
    case_major_order_valid = (
        observed_case_major_order == expected_case_major_order
    )
    evaluation_reports = [
        {
            "candidate_id": evaluation.candidate_id,
            "accepted": evaluation.accepted,
            "case_count": len(evaluation.scores),
            "valid_case_count": sum(
                score.status == "valid" for score in evaluation.scores
            ),
            "zero_primary_error_case_count": sum(
                score.primary_error == 0.0 for score in evaluation.scores
            ),
            "case_order_valid": evaluation.case_ids == expected_case_ids,
            "statuses": [score.status for score in evaluation.scores],
            "primary_errors": [
                score.primary_error for score in evaluation.scores
            ],
            "sanitized_failures": [
                list(score.failures) for score in evaluation.scores
            ],
        }
        for evaluation in evaluations
    ]
    audit_by_candidate = {
        candidate_id: [
            row
            for row in lifecycle_audit.rows
            if row["candidate_id"] == candidate_id
        ]
        for candidate_id in candidate_ids
    }
    lifecycle_passed = (
        release_passed
        and case_major_order_valid
        and all(
            row["accepted"]
            and row["case_count"] == expected_case_count
            and row["valid_case_count"] == expected_case_count
            and row["zero_primary_error_case_count"] == expected_case_count
            and row["case_order_valid"]
            for row in evaluation_reports
        )
        and all(
            len(rows) == expected_case_count
            and tuple(row["case_id"] for row in rows) == expected_case_ids
            and all(row["session_close_attempted"] for row in rows)
            and all(row["session_reference_released"] for row in rows)
            and all(
                _valid_cuda_memory(row[name])
                for row in rows
                for name in (
                    "cuda_before_session_open",
                    "cuda_after_session_factory",
                    "cuda_after_oracle_trigger",
                    "cuda_after_session_close",
                )
            )
            and all(not row["controller_private_exceptions"] for row in rows)
            for rows in audit_by_candidate.values()
        )
    )

    section_peak_allocated = max(
        row["cuda"]["peak_allocated_bytes"] for row in sections.values()
    )
    section_peak_reserved = max(
        row["cuda"]["peak_reserved_bytes"] for row in sections.values()
    )
    smoke_passed = (
        lifecycle_passed
        and lean_online_work_zero
        and continuous_condition_admission["passed"] is True
    )
    report = {
        "schema": "gkw_unified_selected_runtime_smoke_v3",
        "passed": smoke_passed,
        "generator_sha256": artifact_sha256,
        "physical_commit": args.physical_commit,
        "unified_commit": args.unified_commit,
        "cohort": args.cohort,
        "profile": query_profile,
        "continuous_condition_admission": continuous_condition_admission,
        "query_plan": {
            "schema": "controller_private_root_paired_points_v1",
            "visibility": "controller-private",
            "kind": QueryKind.POINTS.value,
            "point_count": query_point_count,
            "root_count": len(root_queries),
            "clean_perturbed_shared": True,
            "distinct_between_roots": True,
        },
        "runtime_storage": {
            "durable_root": str(storage_root),
            "resident_answer_mode": "cpu_compact_capacity_one_gpu_backend",
            "minimum_durable_free_bytes": _MINIMUM_DURABLE_FREE_BYTES,
        },
        "cuda_peak_memory_allocated_bytes": section_peak_allocated,
        "cuda_peak_memory_reserved_bytes": section_peak_reserved,
        "online_generation": {
            "included": True,
            "visibility": "controller-private",
            "current_path": "offline_qualified_fixed_first_draw",
            "qualified_perturbation_phase_timing": {
                "case_count": manufacturing_after_generation[
                    "perturbed_generation_cases"
                ],
                "host_dispatch_without_cuda_synchronize": {
                    "sample_wall_seconds": (
                        manufacturing_after_generation[
                            "perturbed_sample_wall_seconds"
                        ]
                    ),
                    "operator_precompute_wall_seconds": (
                        manufacturing_after_generation[
                            "perturbed_operator_precompute_wall_seconds"
                        ]
                    ),
                    "macro_truth_wall_seconds": (
                        manufacturing_after_generation[
                            "perturbed_macro_truth_host_dispatch_wall_seconds"
                        ]
                    ),
                },
                "macro_truth_to_public_pairs_host_wall_seconds": (
                    manufacturing_after_generation[
                        "perturbed_macro_truth_to_public_pairs_wall_seconds"
                    ]
                ),
                "cuda_synchronized_perturbed_generation_wall_seconds": (
                    sections["perturbed_generation"]["wall_seconds"]
                ),
                "cuda_synchronized_oracle_wall_seconds": {
                    name: sections[name]["wall_seconds"]
                    for name in (
                        "first_oracle_query",
                        "repeated_oracle_query",
                        "perturbed_oracle_query",
                        "case_switch_oracle_query",
                        "case_major_two_candidate",
                    )
                },
                "stage_totals_are_not_additive": True,
                "startup_attestation_included": False,
            },
            "sections": {
                name: sections[name]
                for name in ("clean_generation", "perturbed_generation")
            },
            "manufacturing_telemetry_before": manufacturing_before,
            "manufacturing_telemetry_after_generation": (
                manufacturing_after_generation
            ),
            "manufacturing_telemetry_after_release": (
                manufacturing_after_release
            ),
            "clean_gate_accounting": {
                "strict_nonlinear_audits_derived": (
                    manufacturing_after_generation[
                        "clean_strict_nonlinear_audits_derived"
                    ]
                ),
                "operator_frames_derived": (
                    manufacturing_after_generation[
                        "clean_operator_frame_evaluations_derived"
                    ]
                ),
                "decoder_rejections_before_physical_audit": clean_decoder_rejections,
                "physical_audits": clean_audited_observations,
                "decoder_attempts_derived": (
                    clean_decoder_rejections + clean_audited_observations
                ),
            },
            "qualified_perturbation_online_work": {
                name: manufacturing_after_generation[name]
                for name in zero_online_work_fields
            },
            "lean_online_work_zero": lean_online_work_zero,
        },
        "required_oracle_query": {
            "included": True,
            "visibility": "controller-private",
            "sections": {
                name: sections[name]
                for name in (
                    "first_oracle_query",
                    "repeated_oracle_query",
                    "perturbed_oracle_query",
                    "case_switch_oracle_query",
                    "case_major_two_candidate",
                    "cohort_release",
                )
            },
            "clean_cache_probe": clean_backend_cache_probe,
            "case_switch_probe": backend_switch_probe,
            "case_major_backend": backend_snapshots,
            "release_passed": release_passed,
            "oracle_after_release": oracle_after_release,
        },
        "offline_qualification": {
            "included": True,
            "qualification_id": certificate.qualification_id,
            "qualification_report_sha256": (
                certificate.qualification_report_sha256
            ),
            "scale_rule_sha256": certificate.scale_rule_sha256,
            "qualification_sample_count": certificate.qualification_sample_count,
            "loaded_once_at_runtime_start": True,
            "environment_bound": True,
            "startup_validation_wall_seconds": (
                qualification_validation_wall_seconds
            ),
            "included_in_per_case_generation_timing": False,
            "per_case_file_hash_or_numerical_validation": False,
            "independent_reference_parity": {
                "included": False,
                "reason": (
                    "independent reference parity remains outside this "
                    "online runtime smoke"
                ),
            },
        },
        "clean_case_id": clean.case.case_id,
        "child_case_id": child.case.case_id,
        "clean_case_ids": [row.case.case_id for row in clean_records],
        "child_case_ids": [row.case.case_id for row in child_records],
        "clean_answer_id": clean_answer_id,
        "child_answer_id": child_answer_id,
        "child_parent_answer_id": child_parent_answer_id,
        "base_storage_shared": base_storage_shared,
        "source_delta_l2": source_delta,
        "macro_cross_relative_l2": float(cross_error),
        "oracle_max_total_rows": _SERVICE_RUNTIME_POLICY["oracle_max_total_rows"],
        "decoder_precision": actual_decoder_precision,
        "service_runtime_policy": dict(_SERVICE_RUNTIME_POLICY),
        "lifecycle_root_count": args.lifecycle_root_count,
        "lifecycle_case_count": expected_case_count,
        "lifecycle_two_candidate_passed": lifecycle_passed,
        "lifecycle_case_major_order_valid": case_major_order_valid,
        "lifecycle_case_major_order": [
            list(row) for row in observed_case_major_order
        ],
        "lifecycle_evaluations": evaluation_reports,
        "lifecycle_oracle_backend_snapshots": backend_snapshots,
        "controller_private_oracle_backend": {
            "visibility": "controller-private",
            "clean_cache_probe": clean_backend_cache_probe,
            "case_switch_probe": backend_switch_probe,
            "before_release": case_major_backend_after,
            "after_release": oracle_after_release,
        },
        "controller_private_lifecycle_audit": {
            "visibility": "controller-private",
            "cases": lifecycle_audit.rows,
        },
        "shared_snapshot_scores_valid": evaluations[0].accepted,
        "shared_snapshot_score_statuses": [
            score.status for score in evaluations[0].scores
        ],
        "shared_snapshot_primary_errors": [
            score.primary_error for score in evaluations[0].scores
        ],
    }
    (run_root / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))
    if not smoke_passed:
        raise RuntimeError(
            "GKW runtime smoke failed lifecycle, lean-online, or "
            "continuous-condition lookup admission"
        )


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        if exc.code not in (None, 0):
            traceback.print_exc()
        raise
