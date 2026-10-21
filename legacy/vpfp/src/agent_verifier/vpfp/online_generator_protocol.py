from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from agent_verifier.vpfp.agent_loop import _solver_visible_question
from agent_verifier.vpfp.continuous_macros import is_continuous_macro_question
from agent_verifier.vpfp.io import assert_public_bundle_isolated


ONLINE_GENERATOR_CONFIG_SCHEMA = "vpfp_online_generator_controller_config_v1"
ONLINE_GENERATOR_INTERFACE_SCHEMA = "vpfp_online_generator_interface_v1"
ONLINE_GENERATOR_REQUEST_SCHEMA = "vpfp_online_generator_request_v1"
ONLINE_GENERATOR_RESPONSE_SCHEMA = "vpfp_online_generator_response_v1"
ONLINE_GENERATOR_OUTPUT_SCHEMA = "vpfp_online_generator_output_manifest_v1"

PUBLIC_ARRAY_KEYS = {
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
MACRO_PUBLIC_ARRAY_KEYS = PUBLIC_ARRAY_KEYS | {"momentum0", "kinetic_energy0"}
FULL_STATE_ANSWER_ARRAY_KEYS = {"rho", "g", "phi", "electric"}
MACRO_ANSWER_ARRAY_KEYS = FULL_STATE_ANSWER_ARRAY_KEYS | {
    "momentum",
    "kinetic_energy",
    "raw_second_moment",
}
EXACT_MACRO_ANSWER_ARRAY_KEYS = {
    "rho",
    "momentum",
    "kinetic_energy",
    "raw_second_moment",
}
QUERY_ARRAY_KEYS = {
    "values",
    "rho",
    "g",
    "f",
    "phi",
    "electric",
    "momentum",
    "kinetic_energy",
    "raw_second_moment",
    "points",
}
QUESTION_PRIVATE_KEYS = {
    "backend",
    "decoder_checkpoint",
    "decoder_checkpoint_sha256",
    "generator_backend",
    "generator_representation",
    "latent",
    "latent_variables",
    "model_checkpoint",
    "model_parameters",
    "parameterization",
    "representation",
    "source_family",
    "source_origin",
}
FORBIDDEN_VISIBLE_TEXT = (
    "/project/",
    "/scratch/",
    "continuous_private_state",
    "decoder_checkpoint",
    "latent_variables",
    "reward_guidance_final.pt",
)


def online_interface_manifest(
    *,
    max_batch: int,
    max_samples: int | None,
    max_answer_grids: int | None,
    max_points_per_call: int,
    max_total_points: int | None,
    include_macro_truth: bool = False,
    aggregate_limits_mode: str = "enforced",
    development_resource_budget: Mapping[str, Any] | None = None,
    supports_batch_requests: bool = False,
) -> dict[str, Any]:
    if aggregate_limits_mode not in {"enforced", "accounting_only"}:
        raise ValueError("unexpected aggregate-limit mode")
    limits = {
        "max_batch": int(max_batch),
        "max_points_per_call": int(max_points_per_call),
    }
    if aggregate_limits_mode == "enforced":
        if any(
            value is None
            for value in (max_samples, max_answer_grids, max_total_points)
        ):
            raise ValueError("enforced aggregate limits cannot be omitted")
        limits.update(
            {
                "max_samples": int(max_samples),
                "max_answer_grids": int(max_answer_grids),
                "max_total_points": int(max_total_points),
            }
        )
    manifest = {
        "schema": ONLINE_GENERATOR_INTERFACE_SCHEMA,
        "transport": "controller_owned_unix_socket_via_public_client",
        "client": "online_generator.py",
        "dynamic_output_root": "online_generator_outputs",
        "sample_modes": ["base", "guided", "mixed"],
        "answer_modes": ["none", "grid"],
        "operations": [
            "status",
            "sample",
            "query",
            *(
                ["materialize", "materialize_batch", "query_batch"]
                if supports_batch_requests
                else []
            ),
        ],
        "point_query_fields": [
            "source",
            "rho0",
            "g0",
            "truth",
            *(["macro_truth"] if include_macro_truth else []),
        ],
        "limits": limits,
        "aggregate_limits_mode": aggregate_limits_mode,
        "visibility_contract": {
            "returns": "numerical public questions and optional numerical development answers",
            "point_queries": "Agent-selected finite point arrays",
            "never_returns": [
                "decoder, diffusion, or reward-model parameters",
                "checkpoint paths",
                "latent variables or initial diffusion noise",
                "controller-private promotion or final cases",
            ],
            "generated_samples_are_training_only": True,
        },
    }
    if supports_batch_requests:
        manifest["batch_requests"] = {
            "maximum_requests_per_call": int(max_batch),
            "materialize": (
                "mixed roots and resolutions are accepted and grouped internally "
                "for GPU generation"
            ),
            "query": (
                "mixed cases and fields are accepted; total points in one batch "
                "must not exceed max_points_per_call"
            ),
        }
    if development_resource_budget is not None:
        manifest["development_resource_budget"] = dict(
            development_resource_budget
        )
    return manifest


def validate_controller_config(config: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(config)
    if payload.get("schema") != ONLINE_GENERATOR_CONFIG_SCHEMA:
        raise ValueError("unexpected online-generator controller config schema")
    required_paths = (
        "workspace",
        "private_root",
        "diffusion_checkpoint",
        "reward_checkpoint",
        "autoencoder_checkpoint",
        "encoded_latents",
    )
    resolved: dict[str, Path] = {}
    for name in required_paths:
        path = Path(str(payload.get(name) or "")).resolve()
        if name in {"workspace", "private_root"}:
            if name == "workspace" and not path.is_dir():
                raise FileNotFoundError(path)
        elif not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
        resolved[name] = path
    workspace = resolved["workspace"]
    private_root = resolved["private_root"]
    if _is_within(private_root, workspace) or _is_within(workspace, private_root):
        raise ValueError("online-generator private root and Agent workspace must be disjoint")
    if not str(workspace).startswith("/scratch/") or not str(private_root).startswith("/scratch/"):
        raise ValueError("online-generator workspace and private root must be under /scratch")
    target_sha = str(payload.get("target_solver_sha256") or "")
    if len(target_sha) != 64 or any(char not in "0123456789abcdef" for char in target_sha):
        raise ValueError("target_solver_sha256 must be a lowercase SHA-256 digest")
    limits = dict(payload.get("limits") or {})
    for name in (
        "max_batch",
        "max_samples",
        "max_answer_grids",
        "max_points_per_call",
        "max_total_points",
    ):
        if int(limits.get(name, 0)) < 1:
            raise ValueError(f"online-generator limit {name} must be positive")
    if int(limits["max_batch"]) > int(limits["max_samples"]):
        raise ValueError("max_batch cannot exceed max_samples")
    if int(limits["max_answer_grids"]) > int(limits["max_samples"]):
        raise ValueError("max_answer_grids cannot exceed max_samples")
    if int(limits["max_points_per_call"]) > int(limits["max_total_points"]):
        raise ValueError("max_points_per_call cannot exceed max_total_points")
    return payload


def assert_reward_checkpoint_target(
    reward_payload: Mapping[str, Any], expected_solver_sha256: str
) -> dict[str, Any]:
    train_config = dict(reward_payload.get("train_config") or {})
    observed = str(train_config.get("target_solver_sha256") or "")
    if observed != str(expected_solver_sha256):
        raise ValueError(
            "reward checkpoint is not bound to the frozen target solver: "
            f"expected={expected_solver_sha256}, observed={observed or '<missing>'}"
        )
    if train_config.get("reward") != "adversarial_score_larger_is_harder":
        raise ValueError("reward checkpoint uses an unexpected solver reward")
    return train_config


def export_numerical_online_case(
    private_case_dir: str | Path,
    public_case_dir: str | Path,
    *,
    public_case_id: str,
    include_answer: bool,
) -> dict[str, Any]:
    source = Path(private_case_dir).resolve()
    destination = Path(public_case_dir).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    question = json.loads((source / "public_question.json").read_text(encoding="utf-8"))
    visible = _drop_private_question_keys(
        _solver_visible_question(question, str(public_case_id))
    )
    encoded = json.dumps(visible, indent=2, sort_keys=True) + "\n"
    _assert_no_private_text(encoded, label="public_question.json")
    (destination / "public_question.json").write_text(encoded, encoding="utf-8")
    shutil.copy2(source / "public_arrays.npz", destination / "public_arrays.npz")
    assert_public_bundle_isolated(destination)
    expected_public_keys = (
        MACRO_PUBLIC_ARRAY_KEYS
        if is_continuous_macro_question(question)
        else PUBLIC_ARRAY_KEYS
    )
    with np.load(destination / "public_arrays.npz", allow_pickle=False) as public:
        if set(public.files) != expected_public_keys:
            raise ValueError(
                f"unexpected public online array keys: {sorted(public.files)}"
            )
        if not all(np.all(np.isfinite(public[name])) for name in public.files):
            raise ValueError("online public arrays contain non-finite values")
    answer_path = None
    if include_answer:
        with np.load(source / "hidden_truth_cache.npz", allow_pickle=False) as truth:
            answer = {
                "rho": np.asarray(truth["rho_true"], dtype=np.float64),
                "g": np.asarray(truth["g_true"], dtype=np.float64),
                "phi": np.asarray(truth["phi_true"], dtype=np.float64),
                "electric": np.asarray(truth["E_true"], dtype=np.float64),
            }
            if is_continuous_macro_question(question):
                required = {
                    "momentum_true",
                    "kinetic_energy_true",
                    "raw_second_moment_true",
                }
                missing = required - set(truth.files)
                if missing:
                    raise ValueError(
                        f"macro online answer lacks exact full-space fields: {sorted(missing)}"
                    )
                answer.update(
                    {
                        "momentum": np.asarray(
                            truth["momentum_true"], dtype=np.float64
                        ),
                        "kinetic_energy": np.asarray(
                            truth["kinetic_energy_true"], dtype=np.float64
                        ),
                        "raw_second_moment": np.asarray(
                            truth["raw_second_moment_true"], dtype=np.float64
                        ),
                    }
                )
        if not all(np.all(np.isfinite(value)) for value in answer.values()):
            raise ValueError("online numerical answer contains non-finite values")
        answer_path = destination / "answer.npz"
        np.savez_compressed(answer_path, **answer)
    return {
        "case_id": str(public_case_id),
        "case_dir": str(destination),
        "public_question": str(destination / "public_question.json"),
        "public_arrays": str(destination / "public_arrays.npz"),
        "answer": None if answer_path is None else str(answer_path),
    }


def audit_online_output(
    output_root: str | Path,
    *,
    registered_files: Iterable[str | Path] | None = None,
    allow_missing_registered_files: bool = False,
) -> dict[str, Any]:
    raw_root = Path(output_root)
    failures: list[str] = []
    files: list[dict[str, Any]] = []
    missing_registered: list[Path] = []
    if not raw_root.is_dir() or raw_root.is_symlink():
        return {
            "schema": "vpfp_online_generator_output_audit_v2",
            "passed": False,
            "failures": [f"missing regular online output root: {raw_root}"],
            "files": [],
        }
    root = raw_root.resolve()
    all_items = sorted(root.rglob("*"))
    if registered_files is None:
        audit_mode = "strict_entire_output_root"
        candidates = all_items
        ignored = []
    else:
        audit_mode = "registered_service_outputs"
        registered: set[Path] = set()
        for raw in registered_files:
            item = Path(raw)
            if not item.is_absolute():
                item = root / item
            try:
                relative = item.relative_to(root)
            except ValueError:
                failures.append(f"registered output escapes root: {item}")
                continue
            if ".." in relative.parts:
                failures.append(f"registered output escapes root: {item}")
                continue
            try:
                item.resolve(strict=False).relative_to(root)
            except ValueError:
                failures.append(f"registered output resolves outside root: {item}")
                continue
            cursor = root
            has_symlink_parent = False
            for part in relative.parts[:-1]:
                cursor /= part
                if cursor.is_symlink():
                    failures.append(f"symlink parent is forbidden: {relative}")
                    has_symlink_parent = True
                    break
            if has_symlink_parent:
                continue
            registered.add(item)
        candidates = sorted(registered)
        ignored = [
            item
            for item in all_items
            if (item.is_file() or item.is_symlink()) and item not in registered
        ]
    for item in candidates:
        relative = item.relative_to(root)
        if not item.exists():
            missing_registered.append(relative)
            if not allow_missing_registered_files:
                failures.append(f"missing registered online output: {relative}")
            continue
        if item.is_symlink():
            failures.append(f"symlink is forbidden: {relative}")
            continue
        if not item.is_file():
            if registered_files is not None:
                failures.append(f"registered output is not a file: {relative}")
            continue
        allowed = False
        if len(relative.parts) == 1:
            allowed = relative.name in {"manifest.json", "README.md"}
        elif relative.parts[0] == "cases":
            allowed = len(relative.parts) == 3 and relative.name in {
                "public_question.json",
                "public_arrays.npz",
                "answer.npz",
                "exact_macro_answer.npz",
            }
        elif relative.parts[0] == "queries":
            allowed = len(relative.parts) == 2 and relative.suffix == ".npz"
        if not allowed:
            failures.append(f"unexpected online output file: {relative}")
        payload = item.read_bytes()
        files.append(
            {
                "path": str(relative),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
        try:
            if item.name == "public_arrays.npz":
                _audit_npz_one_of(
                    item, (PUBLIC_ARRAY_KEYS, MACRO_PUBLIC_ARRAY_KEYS)
                )
            elif item.name == "answer.npz":
                _audit_npz_one_of(
                    item,
                    (FULL_STATE_ANSWER_ARRAY_KEYS, MACRO_ANSWER_ARRAY_KEYS),
                )
            elif item.name == "exact_macro_answer.npz":
                _audit_npz(item, EXACT_MACRO_ANSWER_ARRAY_KEYS)
            elif relative.parts[0] == "queries" and item.suffix == ".npz":
                _audit_npz(item, QUERY_ARRAY_KEYS, subset=True)
            elif item.suffix in {".json", ".md"}:
                _assert_no_private_text(
                    payload.decode("utf-8", errors="replace"), label=str(relative)
                )
        except Exception as exc:  # noqa: BLE001 - report every output violation.
            failures.append(f"{relative}: {exc!r}")
    return {
        "schema": "vpfp_online_generator_output_audit_v2",
        "passed": not failures,
        "audit_mode": audit_mode,
        "allow_missing_registered_files": bool(allow_missing_registered_files),
        "audited_file_count": len(files),
        "missing_registered_file_count": len(missing_registered),
        "missing_registered_files_sample": [
            str(item) for item in missing_registered[:32]
        ],
        "ignored_unregistered_file_count": len(ignored),
        "ignored_unregistered_files_sample": [
            str(item.relative_to(root)) for item in ignored[:32]
        ],
        "files": files,
        "failures": failures,
    }


def _drop_private_question_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _drop_private_question_keys(item)
            for key, item in value.items()
            if str(key).lower() not in QUESTION_PRIVATE_KEYS
        }
    if isinstance(value, list):
        return [_drop_private_question_keys(item) for item in value]
    return value


def _assert_no_private_text(text: str, *, label: str) -> None:
    lowered = text.lower()
    matches = [token for token in FORBIDDEN_VISIBLE_TEXT if token in lowered]
    if matches:
        raise ValueError(f"private controller text in {label}: {matches}")


def _audit_npz_one_of(path: Path, allowed_sets: Iterable[set[str]]) -> None:
    allowed = tuple(set(values) for values in allowed_sets)
    with np.load(path, allow_pickle=False) as payload:
        keys = set(payload.files)
        if keys not in allowed:
            raise ValueError(f"unexpected keys: {sorted(keys)}")
        for name in payload.files:
            value = np.asarray(payload[name])
            if value.dtype.kind not in {"b", "i", "u", "f"}:
                raise ValueError(f"unsupported dtype for {name}: {value.dtype}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"non-finite values for {name}")


def _audit_npz(path: Path, allowed: set[str], *, subset: bool = False) -> None:
    with np.load(path, allow_pickle=False) as payload:
        keys = set(payload.files)
        valid = bool(keys) and keys <= allowed if subset else keys == allowed
        if not valid:
            raise ValueError(f"unexpected keys: {sorted(keys)}")
        for name in payload.files:
            value = np.asarray(payload[name])
            if value.dtype.kind not in {"b", "i", "u", "f"}:
                raise ValueError(f"unsupported dtype for {name}: {value.dtype}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"non-finite values for {name}")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True
