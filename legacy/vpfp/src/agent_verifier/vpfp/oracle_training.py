from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable

import numpy as np

from agent_verifier.vpfp.agent_loop import (
    SOLVER_PRIVATE_QUESTION_KEYS,
    _solver_visible_question,
)
from agent_verifier.vpfp.io import assert_public_bundle_isolated


RELEASED_TRUTH_SCHEMA = "vpfp_released_training_truth_v1"
RELEASED_TRUTH_ROLE = "released_historical_training_only"
RELEASED_TRUTH_DIRNAME = "released_truth"
MACRO_OUTPUT_CONTRACT = "vpfp_continuous_macro_output_v1"
RELEASED_TRUTH_KEYS = {
    "rho_true",
    "g_true",
    "phi_true",
    "E_true",
    "basis_coeffs_true",
    "momentum_true",
    "kinetic_energy_true",
    "raw_second_moment_true",
}
REQUIRED_RELEASED_TRUTH_KEYS = {"rho_true", "g_true", "phi_true", "E_true"}
SOLVER_BUNDLE_SCHEMA = "vpfp_solver_artifact_bundle_v1"
SOLVER_BUNDLE_MANIFEST = "solver_artifact_bundle.json"
SOLVER_BUNDLE_SUFFIXES = {
    ".json",
    ".npy",
    ".npz",
    ".pt",
    ".pth",
    ".py",
    ".safetensors",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
DEFAULT_SOLVER_BUNDLE_MAX_BYTES = 512 * 1024 * 1024
DEFAULT_RELEASED_TRUTH_MAX_BYTES = 16 * 1024 * 1024 * 1024
FINGERPRINT_KINDS = ("public_question", "hidden_solution")


def export_released_truth_bank(
    case_dirs: Iterable[str | Path],
    workspace: str | Path,
    controller_out: str | Path,
    *,
    max_total_bytes: int = DEFAULT_RELEASED_TRUTH_MAX_BYTES,
) -> dict[str, Any]:
    """Export explicitly released historical answers without source identities."""

    workspace_path = Path(workspace).resolve()
    controller_path = Path(controller_out).resolve()
    if _is_within(controller_path, workspace_path):
        raise ValueError("controller-private truth mapping must be outside the solver workspace")
    bank = workspace_path / RELEASED_TRUTH_DIRNAME
    if bank.exists():
        raise FileExistsError(bank)
    cases_root = bank / "cases"
    cases_root.mkdir(parents=True)
    controller_path.mkdir(parents=True, exist_ok=True)

    public_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    for index, raw_case_dir in enumerate(case_dirs):
        source = Path(raw_case_dir).resolve()
        assert_public_bundle_isolated(source)
        neutral_id = f"train_{index:06d}"
        destination = cases_root / neutral_id
        destination.mkdir()

        question = json.loads((source / "public_question.json").read_text(encoding="utf-8"))
        original_case_id = str(question.get("case_id") or source.name)
        visible_question = _solver_visible_question(question, neutral_id)
        (destination / "public_question.json").write_text(
            json.dumps(visible_question, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        shutil.copy2(source / "public_arrays.npz", destination / "public_arrays.npz")

        truth_source = source / "hidden_truth_cache.npz"
        if not truth_source.is_file() or truth_source.is_symlink():
            raise FileNotFoundError(f"missing regular hidden truth cache: {truth_source}")
        with np.load(truth_source, allow_pickle=False) as payload:
            unknown = set(payload.files) - RELEASED_TRUTH_KEYS
            missing = REQUIRED_RELEASED_TRUTH_KEYS - set(payload.files)
            if unknown or missing:
                raise ValueError(
                    f"invalid truth keys for {source}: unknown={sorted(unknown)}, "
                    f"missing={sorted(missing)}"
                )
            truth_arrays = {
                name: np.asarray(payload[name], dtype=np.float64)
                for name in payload.files
            }
        if not all(np.all(np.isfinite(value)) for value in truth_arrays.values()):
            raise ValueError(f"released truth contains non-finite values: {source}")
        _validate_truth_shapes(visible_question, truth_arrays)
        np.savez_compressed(destination / "released_truth.npz", **truth_arrays)

        public_fingerprint = case_fingerprint(source)
        hidden_fingerprint = hidden_solution_fingerprint(source)
        public_rows.append(
            {
                "case_id": neutral_id,
                "case_dir": f"cases/{neutral_id}",
                "public_question": f"cases/{neutral_id}/public_question.json",
                "public_arrays": f"cases/{neutral_id}/public_arrays.npz",
                "released_truth": f"cases/{neutral_id}/released_truth.npz",
                "truth_keys": sorted(truth_arrays),
                "training_role": RELEASED_TRUTH_ROLE,
                "eligible_for_selection": False,
                "eligible_for_promotion": False,
                "eligible_for_final_evaluation": False,
            }
        )
        private_rows.append(
            {
                "case_id": neutral_id,
                "source_case_id": original_case_id,
                "source_case_dir": str(source),
                "case_fingerprint": public_fingerprint,
                "hidden_solution_fingerprint": hidden_fingerprint,
            }
        )

    manifest = {
        "schema": RELEASED_TRUTH_SCHEMA,
        "role": RELEASED_TRUTH_ROLE,
        "case_count": len(public_rows),
        "identity_policy": "neutral_training_ids_no_generator_provenance",
        "oracle_policy": "completed answers are intentionally released for solver training",
        "selection_policy": "these cases are never used for candidate selection, promotion, or final evaluation",
        "cases": public_rows,
    }
    (bank / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (bank / "README.md").write_text(
        "# Released VPFP Training Truth\n\n"
        "These completed trajectories and answers are intentionally available for "
        "supervised solver development. They are historical training data only and "
        "are excluded from selection, promotion, and final evaluation cohorts.\n",
        encoding="utf-8",
    )
    private_mapping = {
        "schema": "vpfp_released_training_truth_private_mapping_v1",
        "solver_visible": False,
        "workspace": str(workspace_path),
        "cases": private_rows,
    }
    mapping_path = controller_path / "released_truth_private_mapping.json"
    mapping_path.write_text(
        json.dumps(private_mapping, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    audit = assert_released_truth_bank(bank, max_total_bytes=max_total_bytes)
    audit["private_mapping"] = str(mapping_path)
    audit["case_fingerprints"] = [row["case_fingerprint"] for row in private_rows]
    audit["hidden_solution_fingerprints"] = [
        row["hidden_solution_fingerprint"] for row in private_rows
    ]
    return audit


def audit_released_truth_bank(
    bank_root: str | Path,
    *,
    max_total_bytes: int = DEFAULT_RELEASED_TRUTH_MAX_BYTES,
) -> dict[str, Any]:
    root = Path(bank_root)
    failures: list[str] = []
    files: list[dict[str, Any]] = []
    total_bytes = 0
    if not root.is_dir() or root.is_symlink():
        return {
            "schema": "vpfp_released_training_truth_audit_v1",
            "passed": False,
            "failures": [f"missing regular released truth bank: {root}"],
            "files": [],
            "total_bytes": 0,
        }
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        failures.append("missing regular manifest.json")
        manifest: dict[str, Any] = {}
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != RELEASED_TRUTH_SCHEMA:
        failures.append("unexpected released truth schema")
    if manifest.get("role") != RELEASED_TRUTH_ROLE:
        failures.append("unexpected released truth role")

    allowed_top = {"README.md", "manifest.json"}
    declared = {str(row.get("case_id")): row for row in manifest.get("cases", [])}
    seen: set[str] = set()
    for item in sorted(root.rglob("*")):
        relative = item.relative_to(root)
        if item.is_symlink():
            failures.append(f"symlink is forbidden: {relative}")
            continue
        if not item.is_file():
            continue
        if len(relative.parts) == 1:
            allowed = relative.name in allowed_top
        else:
            allowed = (
                len(relative.parts) == 3
                and relative.parts[0] == "cases"
                and relative.name
                in {"public_question.json", "public_arrays.npz", "released_truth.npz"}
            )
        if not allowed:
            failures.append(f"unexpected released-truth file: {relative}")
        size = item.stat().st_size
        total_bytes += size
        files.append(
            {
                "path": str(relative),
                "bytes": int(size),
                "sha256": _sha256_file(item),
            }
        )
    if total_bytes > int(max_total_bytes):
        failures.append(
            f"released truth bank exceeds byte limit: {total_bytes} > {int(max_total_bytes)}"
        )

    for neutral_id, row in declared.items():
        if not neutral_id.startswith("train_") or "/" in neutral_id or "\\" in neutral_id:
            failures.append(f"invalid neutral training ID: {neutral_id!r}")
            continue
        if bool(row.get("eligible_for_selection")):
            failures.append(f"training case marked selection-eligible: {neutral_id}")
        if bool(row.get("eligible_for_promotion")):
            failures.append(f"training case marked promotion-eligible: {neutral_id}")
        if bool(row.get("eligible_for_final_evaluation")):
            failures.append(f"training case marked final-eligible: {neutral_id}")
        case_dir = root / "cases" / neutral_id
        try:
            assert_public_bundle_isolated(case_dir)
            question = json.loads((case_dir / "public_question.json").read_text(encoding="utf-8"))
            if str(question.get("case_id")) != neutral_id:
                failures.append(f"case ID mismatch for {neutral_id}")
            with np.load(case_dir / "released_truth.npz", allow_pickle=False) as truth:
                keys = set(truth.files)
                if not REQUIRED_RELEASED_TRUTH_KEYS <= keys <= RELEASED_TRUTH_KEYS:
                    failures.append(f"invalid released truth keys for {neutral_id}: {sorted(keys)}")
                arrays = {name: np.asarray(truth[name]) for name in truth.files}
            if not all(np.all(np.isfinite(value)) for value in arrays.values()):
                failures.append(f"non-finite released truth for {neutral_id}")
            _validate_truth_shapes(question, arrays)
        except Exception as exc:  # noqa: BLE001 - full audit report is more useful.
            failures.append(f"{neutral_id}: {exc!r}")
        seen.add(neutral_id)
    case_dirs = {
        path.name
        for path in (root / "cases").iterdir()
        if path.is_dir() and not path.is_symlink()
    } if (root / "cases").is_dir() else set()
    if case_dirs != seen:
        failures.append(
            f"manifest/case directory mismatch: declared={sorted(seen)}, observed={sorted(case_dirs)}"
        )
    if int(manifest.get("case_count", -1)) != len(declared):
        failures.append("released truth manifest case_count mismatch")
    return {
        "schema": "vpfp_released_training_truth_audit_v1",
        "passed": not failures,
        "case_count": len(declared),
        "total_bytes": int(total_bytes),
        "max_total_bytes": int(max_total_bytes),
        "files": files,
        "failures": failures,
    }


def assert_released_truth_bank(
    bank_root: str | Path,
    *,
    max_total_bytes: int = DEFAULT_RELEASED_TRUTH_MAX_BYTES,
) -> dict[str, Any]:
    report = audit_released_truth_bank(bank_root, max_total_bytes=max_total_bytes)
    if not report["passed"]:
        raise AssertionError(
            "VPFP released truth bank audit failed:\n" + "\n".join(report["failures"])
        )
    return report


def audit_oracle_training_workspace(
    workspace: str | Path,
    *,
    max_truth_bytes: int = DEFAULT_RELEASED_TRUTH_MAX_BYTES,
    max_solver_bytes: int = DEFAULT_SOLVER_BUNDLE_MAX_BYTES,
) -> dict[str, Any]:
    root = Path(workspace)
    failures: list[str] = []
    files: list[dict[str, Any]] = []
    allowed_root_files = {
        "README.md",
        "case_manifest.json",
        "compute_budget.py",
        "generator_reservoir_manifest.json",
        "codex_submission_schema.json",
        "prompt.md",
        "public_eval.py",
        "generator_sample.py",
        "historical_reward_summary.json",
        "online_generator.py",
        "online_generator_interface.json",
        "online_budget_feedback.py",
        "online_resolution_eval.py",
        "online_temporal_feedback.py",
        "public_predict.py",
        "temporal_feedback.json",
        "public_vpfp_runtime.py",
        "released_truth_eval.py",
        "resource_limits.py",
        "solver_agent_instructions.md",
        "solver_artifact_bundle.example.json",
        "solver_contract.md",
        "solver_instruction_preflight.sh",
        "starter_submission.py",
        "submission.py",
    }
    if not root.is_dir() or root.is_symlink():
        return {
            "schema": "vpfp_oracle_training_workspace_audit_v1",
            "passed": False,
            "failures": [f"missing regular oracle workspace: {root}"],
            "files": [],
        }
    manifest_path = root / "case_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("context_mode") != "oracle_assisted_training":
            failures.append("workspace context_mode is not oracle_assisted_training")
    except Exception as exc:  # noqa: BLE001 - retained in audit.
        failures.append(f"invalid public case manifest: {exc!r}")
        manifest = {}
    for item in sorted(root.rglob("*")):
        relative = item.relative_to(root)
        if relative.parts and relative.parts[0] in {
            RELEASED_TRUTH_DIRNAME, "champion_solver"
        }:
            continue
        if item.is_symlink():
            failures.append(f"symlink is forbidden: {relative}")
            continue
        if not item.is_file():
            continue
        if len(relative.parts) == 1:
            allowed = relative.name in allowed_root_files
        elif relative.parts[0] == "generator_reservoir":
            allowed = (
                len(relative.parts) == 4
                and relative.parts[1] == "cases"
                and relative.name in {"public_question.json", "public_arrays.npz"}
            )
        else:
            allowed = (
                len(relative.parts) == 3
                and relative.parts[0] == "cases"
                and relative.name in {"public_question.json", "public_arrays.npz"}
            )
        if not allowed:
            failures.append(f"unexpected oracle workspace file: {relative}")
        files.append(
            {
                "path": str(relative),
                "bytes": int(item.stat().st_size),
                "sha256": _sha256_file(item),
            }
        )
    for row in manifest.get("cases", []):
        neutral_id = str(row.get("case_id") or "")
        case_dir = root / str(row.get("case_dir") or "")
        try:
            assert_public_bundle_isolated(case_dir)
            question = json.loads(
                (case_dir / "public_question.json").read_text(encoding="utf-8")
            )
            if str(question.get("case_id")) != neutral_id:
                failures.append(f"public case ID mismatch: {neutral_id}")
            leaked_keys = SOLVER_PRIVATE_QUESTION_KEYS.intersection(question)
            if leaked_keys:
                failures.append(
                    f"private generation keys in public case {neutral_id}: {sorted(leaked_keys)}"
                )
        except Exception as exc:  # noqa: BLE001 - retained in audit.
            failures.append(f"invalid public development case {neutral_id}: {exc!r}")
    generator_manifest_path = root / "generator_reservoir_manifest.json"
    generator_manifest: dict[str, Any] = {}
    if generator_manifest_path.exists():
        try:
            if not generator_manifest_path.is_file() or generator_manifest_path.is_symlink():
                raise ValueError("generator reservoir manifest must be a regular file")
            generator_manifest = json.loads(
                generator_manifest_path.read_text(encoding="utf-8")
            )
            if generator_manifest.get("schema") != "vpfp_public_generator_reservoir_v1":
                raise ValueError("unexpected generator reservoir schema")
            for row in generator_manifest.get("cases", []):
                neutral_id = str(row.get("case_id") or "")
                case_dir = root / str(row.get("case_dir") or "")
                assert_public_bundle_isolated(case_dir)
                question = json.loads(
                    (case_dir / "public_question.json").read_text(encoding="utf-8")
                )
                if str(question.get("case_id")) != neutral_id:
                    failures.append(f"generator reservoir case ID mismatch: {neutral_id}")
                leaked_keys = SOLVER_PRIVATE_QUESTION_KEYS.intersection(question)
                if leaked_keys:
                    failures.append(
                        "private generation keys in generator reservoir case "
                        f"{neutral_id}: {sorted(leaked_keys)}"
                    )
        except Exception as exc:  # noqa: BLE001 - retained in audit.
            failures.append(f"invalid public generator reservoir: {exc!r}")
    truth_audit = audit_released_truth_bank(
        root / RELEASED_TRUTH_DIRNAME,
        max_total_bytes=max_truth_bytes,
    )
    failures.extend(f"released_truth: {value}" for value in truth_audit["failures"])
    champion_bundle = None
    champion_submission = root / "champion_solver" / "submission.py"
    if champion_submission.parent.exists():
        try:
            champion_bundle = inspect_solver_bundle(
                champion_submission,
                max_total_bytes=max_solver_bytes,
            )
        except Exception as exc:  # noqa: BLE001 - retained in audit.
            failures.append(f"invalid public champion bundle: {exc!r}")
    return {
        "schema": "vpfp_oracle_training_workspace_audit_v1",
        "passed": not failures,
        "public_case_count": len(manifest.get("cases", [])),
        "public_generator_case_count": len(generator_manifest.get("cases", [])),
        "released_truth_audit": truth_audit,
        "public_champion_bundle": champion_bundle,
        "files": files,
        "failures": failures,
    }


def assert_oracle_training_workspace(
    workspace: str | Path,
    *,
    max_truth_bytes: int = DEFAULT_RELEASED_TRUTH_MAX_BYTES,
    max_solver_bytes: int = DEFAULT_SOLVER_BUNDLE_MAX_BYTES,
) -> dict[str, Any]:
    report = audit_oracle_training_workspace(
        workspace,
        max_truth_bytes=max_truth_bytes,
        max_solver_bytes=max_solver_bytes,
    )
    if not report["passed"]:
        raise AssertionError(
            "VPFP oracle training workspace audit failed:\n"
            + "\n".join(report["failures"])
        )
    return report


def assert_training_private_disjoint(
    training_case_dirs: Iterable[str | Path],
    private_cohorts: dict[str, Iterable[str | Path]],
) -> dict[str, Any]:
    training_dirs = [Path(case_dir) for case_dir in training_case_dirs]
    private_dirs = {
        str(name): [Path(case_dir) for case_dir in case_dirs]
        for name, case_dirs in private_cohorts.items()
    }
    training = _cohort_fingerprint_maps(training_dirs)
    private = {
        name: _cohort_fingerprint_maps(case_dirs)
        for name, case_dirs in private_dirs.items()
    }
    overlaps: dict[str, dict[str, list[str]]] = {}
    for name, fingerprint_maps in private.items():
        by_kind: dict[str, list[str]] = {}
        for kind in FINGERPRINT_KINDS:
            common = sorted(set(training[kind]) & set(fingerprint_maps[kind]))
            if common:
                by_kind[kind] = common
        if by_kind:
            overlaps[name] = by_kind

    private_pairwise: dict[str, dict[str, list[str]]] = {}
    names = sorted(private)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            by_kind = {}
            for kind in FINGERPRINT_KINDS:
                common = sorted(set(private[left][kind]) & set(private[right][kind]))
                if common:
                    by_kind[kind] = common
            if by_kind:
                private_pairwise[f"{left}__{right}"] = by_kind
    passed = not overlaps and not private_pairwise
    report = {
        "schema": "vpfp_training_private_disjointness_v1",
        "passed": passed,
        "fingerprint_kinds": list(FINGERPRINT_KINDS),
        "training_case_count": len(training_dirs),
        "private_cohort_counts": {
            name: len(case_dirs) for name, case_dirs in private_dirs.items()
        },
        "overlaps": overlaps,
        "private_pairwise_overlaps": private_pairwise,
    }
    if not passed:
        raise ValueError(
            "released training truth overlaps private cohorts or private cohorts overlap: "
            f"training={overlaps}, private_pairwise={private_pairwise}"
        )
    return report


def stage_solver_bundle(
    submission: str | Path,
    out_dir: str | Path,
    *,
    max_total_bytes: int = DEFAULT_SOLVER_BUNDLE_MAX_BYTES,
) -> Path:
    raw_source = Path(submission)
    bundle = inspect_solver_bundle(raw_source, max_total_bytes=max_total_bytes)
    source = raw_source.resolve()
    destination_root = Path(out_dir)
    if destination_root.exists():
        raise FileExistsError(destination_root)
    destination_root.mkdir(parents=True)
    for row in bundle["files"]:
        relative = Path(row["path"])
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source.parent / relative, destination)
    normalized_manifest = {
        "schema": SOLVER_BUNDLE_SCHEMA,
        "entrypoint": "submission.py",
        "files": [row["path"] for row in bundle["files"]],
        "total_bytes": bundle["total_bytes"],
        "aggregate_sha256": bundle["aggregate_sha256"],
    }
    (destination_root / SOLVER_BUNDLE_MANIFEST).write_text(
        json.dumps(normalized_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    staged_submission = destination_root / "submission.py"
    staged_bundle = inspect_solver_bundle(
        staged_submission,
        max_total_bytes=max_total_bytes,
    )
    if (
        staged_bundle["aggregate_sha256"] != bundle["aggregate_sha256"]
        or staged_bundle["total_bytes"] != bundle["total_bytes"]
    ):
        raise RuntimeError(
            "solver bundle changed while it was being frozen"
        )
    return staged_submission


def inspect_solver_bundle(
    submission: str | Path,
    *,
    max_total_bytes: int = DEFAULT_SOLVER_BUNDLE_MAX_BYTES,
) -> dict[str, Any]:
    raw_source = Path(submission)
    if raw_source.is_symlink():
        raise FileNotFoundError(
            f"solver entrypoint must not be a symlink: {raw_source}"
        )
    source = raw_source.resolve()
    if not source.is_file() or source.name != "submission.py":
        raise FileNotFoundError(
            f"solver entrypoint must be a regular submission.py: {source}"
        )
    manifest_path = source.parent / SOLVER_BUNDLE_MANIFEST
    if manifest_path.exists():
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise ValueError(f"invalid solver bundle manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != SOLVER_BUNDLE_SCHEMA:
            raise ValueError("unexpected solver artifact bundle schema")
        if manifest.get("entrypoint", "submission.py") != "submission.py":
            raise ValueError("solver bundle entrypoint must be submission.py")
        declared_files = list(manifest.get("files") or [])
    else:
        declared_files = ["submission.py"]
    if "submission.py" not in declared_files:
        raise ValueError("solver bundle must declare submission.py")
    normalized: list[str] = []
    for raw in declared_files:
        relative = PurePosixPath(str(raw))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError(f"unsafe solver bundle path: {raw!r}")
        path = Path(*relative.parts)
        if path == Path(SOLVER_BUNDLE_MANIFEST):
            raise ValueError("solver bundle manifest must not declare itself")
        if path.suffix.lower() not in SOLVER_BUNDLE_SUFFIXES:
            raise ValueError(f"unsupported solver bundle suffix: {path}")
        if str(path) in normalized:
            raise ValueError(f"duplicate solver bundle path: {path}")
        item = source.parent / path
        if (
            not item.is_file()
            or item.is_symlink()
            or not _is_within(item.resolve(), source.parent)
        ):
            raise ValueError(
                f"solver bundle member is not a regular local file: {path}"
            )
        normalized.append(str(path))
    rows = []
    total_bytes = sum(
        (source.parent / relative).stat().st_size for relative in normalized
    )
    if total_bytes > int(max_total_bytes):
        raise ValueError(
            f"solver bundle exceeds byte limit: {total_bytes} > {max_total_bytes}"
        )
    digest = hashlib.sha256()
    for relative in sorted(normalized):
        item = source.parent / relative
        size = item.stat().st_size
        sha = _sha256_file(item)
        digest.update(
            relative.encode("utf-8") + b"\0" + sha.encode("ascii") + b"\0"
        )
        rows.append({"path": relative, "bytes": int(size), "sha256": sha})
    return {
        "schema": SOLVER_BUNDLE_SCHEMA,
        "entrypoint": str(source),
        "files": rows,
        "total_bytes": int(total_bytes),
        "max_total_bytes": int(max_total_bytes),
        "aggregate_sha256": digest.hexdigest(),
    }


def case_fingerprint(case_dir: str | Path) -> str:
    """Fingerprint the public mathematical question, independent of its answer."""

    root = Path(case_dir).resolve()
    question_path = root / "public_question.json"
    arrays_path = root / "public_arrays.npz"
    for path in (question_path, arrays_path):
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
    question = json.loads(question_path.read_text(encoding="utf-8"))
    visible_question = _solver_visible_question(question, "case")
    canonical_question = json.dumps(
        visible_question,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(b"public_question.json\0")
    digest.update(canonical_question + b"\0")
    digest.update(b"public_arrays.npz\0")
    digest.update(_canonical_npz_fingerprint(arrays_path).encode("ascii") + b"\0")
    return digest.hexdigest()


def hidden_solution_fingerprint(case_dir: str | Path) -> str:
    root = Path(case_dir).resolve()
    truth_path = root / "hidden_truth_cache.npz"
    return _canonical_npz_fingerprint(
        truth_path,
        required_keys=REQUIRED_RELEASED_TRUTH_KEYS,
    )


def _cohort_fingerprint_maps(
    case_dirs: Iterable[str | Path],
) -> dict[str, dict[str, str]]:
    paths = list(case_dirs)
    return {
        "public_question": _fingerprint_map(paths, case_fingerprint),
        "hidden_solution": _fingerprint_map(paths, hidden_solution_fingerprint),
    }


def _fingerprint_map(
    case_dirs: Iterable[str | Path],
    fingerprint_fn: Callable[[str | Path], str],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for case_dir in case_dirs:
        fingerprint = fingerprint_fn(case_dir)
        if fingerprint in result:
            raise ValueError(
                f"duplicate case content: {result[fingerprint]} and "
                f"{Path(case_dir).resolve()}"
            )
        result[fingerprint] = str(Path(case_dir).resolve())
    return result


def _canonical_npz_fingerprint(
    path: str | Path,
    *,
    required_keys: set[str] | None = None,
) -> str:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(source)
    digest = hashlib.sha256()
    with np.load(source, allow_pickle=False) as payload:
        available = set(payload.files)
        names = sorted(
            available if required_keys is None else set(required_keys)
        )
        missing = set(names) - available
        if missing:
            raise ValueError(f"fingerprint payload lacks keys: {sorted(missing)}")
        for name in names:
            value = np.asarray(payload[name])
            if value.dtype.kind not in {"b", "i", "u", "f"}:
                raise ValueError(
                    f"fingerprint array {name!r} has unsupported dtype {value.dtype}"
                )
            normalized = np.ascontiguousarray(
                value, dtype=np.dtype("<f8")
            )
            if not np.all(np.isfinite(normalized)):
                raise ValueError(f"fingerprint array {name!r} is non-finite")
            digest.update(name.encode("utf-8") + b"\0")
            digest.update(
                json.dumps(list(value.shape), separators=(",", ":")).encode("ascii")
                + b"\0"
            )
            digest.update(normalized.tobytes(order="C") + b"\0")
    return digest.hexdigest()


def _validate_truth_shapes(
    question: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> None:
    target_shapes = question["target_shapes"]
    rho_shape = tuple(int(value) for value in target_shapes["rho"])
    if "g" in target_shapes:
        g_shape = tuple(int(value) for value in target_shapes["g"])
    else:
        grid = question["grid"]
        g_shape = (
            *rho_shape,
            int(grid["nvx"]),
            int(grid["nvy"]),
        )
    if tuple(arrays["rho_true"].shape) != rho_shape:
        raise ValueError(
            f"rho truth shape mismatch: {arrays['rho_true'].shape} != {rho_shape}"
        )
    if tuple(arrays["g_true"].shape) != g_shape:
        raise ValueError(
            f"g truth shape mismatch: {arrays['g_true'].shape} != {g_shape}"
        )
    if tuple(arrays["phi_true"].shape) != rho_shape:
        raise ValueError(
            f"phi truth shape mismatch: {arrays['phi_true'].shape} != {rho_shape}"
        )
    expected_electric = (rho_shape[0], 2, *rho_shape[1:])
    if tuple(arrays["E_true"].shape) != expected_electric:
        raise ValueError(
            f"electric-field truth shape mismatch: "
            f"{arrays['E_true'].shape} != {expected_electric}"
        )
    macro_output = str(
        question.get("solver_output_contract_version") or ""
    ) == MACRO_OUTPUT_CONTRACT
    macro_shapes = {
        "momentum_true": (*rho_shape, 2),
        "kinetic_energy_true": rho_shape,
        "raw_second_moment_true": (*rho_shape, 3),
    }
    for name, expected_shape in macro_shapes.items():
        if macro_output and name not in arrays:
            raise ValueError(f"macro released truth is missing {name}")
        if name in arrays and tuple(arrays[name].shape) != expected_shape:
            raise ValueError(
                f"{name} shape mismatch: "
                f"{arrays[name].shape} != {expected_shape}"
            )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True
