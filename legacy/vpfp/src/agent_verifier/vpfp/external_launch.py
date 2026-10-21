from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from agent_verifier.vpfp.agent_loop import (
    BLIND_CONTEXT_MODE,
    INSTRUCTION_ONLY_CONTEXT_MODE,
    ORACLE_ASSISTED_CONTEXT_MODE,
    assert_blind_solver_workspace_isolated,
)
from agent_verifier.vpfp.oracle_training import assert_oracle_training_workspace


EXTERNAL_LAUNCH_SCHEMA = "vpfp_external_agent_launch_packet_v1"
SYNTHETIC_SOURCE_ORIGINS = {
    "fresh_answer_first_continuous_guidance",
    "generated_continuous_autodiff_residual",
    "generated_solution_residual",
    "randomized_public_equation",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _workspace_role(relative: Path) -> str:
    parts = relative.parts
    if parts[0] == "cases":
        return "public_question" if relative.name.endswith(".json") else "public_arrays"
    if parts[0] == "generator_reservoir":
        return "public_generator_question" if relative.name.endswith(".json") else "public_generator_arrays"
    if parts[0] == "released_truth":
        if relative.name == "released_truth.npz":
            return "intentionally_released_historical_answer"
        if relative.name == "public_arrays.npz":
            return "released_case_public_arrays"
        return "released_training_manifest_or_question"
    if parts[0] == "champion_solver":
        return "public_previous_solver_artifact"
    if relative.name == "historical_reward_summary.json":
        return "sanitized_historical_reward"
    if relative.name in {"online_generator.py", "online_generator_interface.json"}:
        return "public_bounded_online_generator_interface"
    return "public_solver_runtime_or_instruction"


def scan_external_workspace(workspace: str | Path) -> list[dict[str, Any]]:
    root = Path(workspace).resolve()
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"external workspace symlink is forbidden: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        rows.append(
            {
                "path": str(relative),
                "role": _workspace_role(relative),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return rows


def _audit_workspace(workspace: Path, context_mode: str) -> dict[str, Any]:
    if context_mode in {BLIND_CONTEXT_MODE, INSTRUCTION_ONLY_CONTEXT_MODE}:
        return assert_blind_solver_workspace_isolated(workspace)
    if context_mode == ORACLE_ASSISTED_CONTEXT_MODE:
        return assert_oracle_training_workspace(workspace)
    raise ValueError(f"unsupported external Agent context mode: {context_mode}")


def _packet_digest(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def build_external_launch_packet(
    workspace: str | Path,
    source_case_dirs: Iterable[str | Path],
    *,
    model: str,
    reasoning_effort: str,
    user_authorized_external_dispatch: bool,
    runtime_interfaces: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    root = Path(workspace).resolve()
    manifest = json.loads((root / "case_manifest.json").read_text())
    context_mode = str(manifest.get("context_mode"))
    audit = _audit_workspace(root, context_mode)
    source_rows = []
    for raw_case in source_case_dirs:
        case_dir = Path(raw_case).resolve()
        question_path = case_dir / "public_question.json"
        question = json.loads(question_path.read_text())
        source_rows.append(
            {
                "case_id": str(question.get("case_id") or case_dir.name),
                "equation_version": str(
                    question.get("equation_version") or question.get("equation")
                ),
                "source_origin": str(question.get("source_origin") or ""),
                "public_question_sha256": _sha256(question_path),
            }
        )
    synthetic_only = bool(source_rows) and all(
        row["source_origin"] in SYNTHETIC_SOURCE_ORIGINS for row in source_rows
    )
    allowed_files = scan_external_workspace(root)
    payload = {
        "schema": EXTERNAL_LAUNCH_SCHEMA,
        "workspace": str(root),
        "external_service": "OpenAI Codex solver Agent",
        "context_mode": context_mode,
        "model": str(model),
        "reasoning_effort": str(reasoning_effort),
        "synthetic_only": synthetic_only,
        "user_authorized_external_dispatch": bool(user_authorized_external_dispatch),
        "source_provenance": source_rows,
        "allowed_external_files": allowed_files,
        "allowed_external_file_count": len(allowed_files),
        "allowed_external_total_bytes": sum(row["bytes"] for row in allowed_files),
        "declared_runtime_interfaces": list(runtime_interfaces or []),
        "workspace_audit_schema": audit.get("schema"),
        "workspace_audit_passed": bool(audit.get("passed")),
        "explicit_non_upload_roots": [
            "controller-private selection and guard cases",
            "final holdout cases and truth",
            "generator decoder, diffusion, and reward checkpoints",
            "latent variables not intentionally released in this packet",
            "Codex auth and runtime home",
            "/project and /scratch outside the immutable workspace",
        ],
        "outer_launch_review": {
            "reviewed": False,
            "decision": "not_yet_submitted",
            "artifact": None,
        },
        "inner_solver_review": {
            "approval_policy": "never" if context_mode == BLIND_CONTEXT_MODE else "on-request",
            "approvals_reviewer": None if context_mode == BLIND_CONTEXT_MODE else "auto_review",
            "guardian_approval": context_mode != BLIND_CONTEXT_MODE,
            "runtime_codex_home": "ephemeral",
        },
    }
    if not payload["workspace_audit_passed"]:
        raise ValueError("external workspace audit did not pass")
    if not payload["synthetic_only"]:
        raise ValueError("external launch packet source provenance is not synthetic-only")
    if not payload["user_authorized_external_dispatch"]:
        raise ValueError("external dispatch requires explicit user authorization")
    for interface in payload["declared_runtime_interfaces"]:
        if interface.get("schema") != "vpfp_online_generator_interface_v1":
            raise ValueError("unsupported declared external runtime interface")
        if (
            interface.get("visibility_contract", {}).get(
                "generated_samples_are_training_only"
            )
            is not True
        ):
            raise ValueError("online generated samples must be declared training-only")
    payload["packet_sha256"] = _packet_digest(payload)
    return payload


def write_external_launch_packet(path: str | Path, packet: dict[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
    return destination


def validate_external_launch_packet(
    packet_path: str | Path,
    *,
    workspace: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(packet_path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"expected a regular launch packet: {path}")
    packet = json.loads(path.read_text())
    if packet.get("schema") != EXTERNAL_LAUNCH_SCHEMA:
        raise ValueError("unexpected external launch packet schema")
    packet_hash = str(packet.get("packet_sha256") or "")
    unsigned = dict(packet)
    unsigned.pop("packet_sha256", None)
    root = Path(workspace or packet["workspace"]).resolve()
    if root != Path(packet["workspace"]).resolve():
        raise ValueError("external launch workspace does not match packet")
    current_files = scan_external_workspace(root)
    if current_files != packet.get("allowed_external_files"):
        raise ValueError("external launch workspace changed after packet creation")
    if not packet.get("synthetic_only") or not packet.get(
        "user_authorized_external_dispatch"
    ):
        raise ValueError("external launch packet lacks synthetic provenance or consent")
    manifest = json.loads((root / "case_manifest.json").read_text())
    _audit_workspace(root, str(manifest.get("context_mode")))
    return packet
