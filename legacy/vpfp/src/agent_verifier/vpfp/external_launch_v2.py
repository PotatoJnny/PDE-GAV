"""Authorized external launch packets for robust long-time VPFP Agents."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .agent_workspace_v2 import (
    VPFP_ROBUST_EXTERNAL_LAUNCH_SCHEMA,
    VPFP_ROBUST_WORKSPACE_SCHEMA,
    audit_vpfp_robust_agent_workspace,
)
from .continuous_macros import MACRO_OUTPUT_CONTRACT


def build_vpfp_robust_external_launch_packet(
    *,
    workspace: str | Path,
    user_authorized_external_dispatch: bool,
) -> dict[str, Any]:
    """Freeze the exact solver-visible context before external dispatch."""

    root = Path(workspace).resolve()
    audit = audit_vpfp_robust_agent_workspace(root)
    if not audit["passed"]:
        raise ValueError(
            f"VPFP robust workspace failed launch audit: {audit['failures']}"
        )
    manifest = _read_json(root / "workspace_manifest.json")
    authorization = dict(manifest["external_authorization"])
    authorized = bool(
        user_authorized_external_dispatch
        and authorization.get("authorized") is True
        and authorization.get("agent_count") == 2
    )
    files = scan_vpfp_robust_external_workspace(root)
    payload = {
        "schema": VPFP_ROBUST_EXTERNAL_LAUNCH_SCHEMA,
        "workspace": str(root),
        "workspace_schema": VPFP_ROBUST_WORKSPACE_SCHEMA,
        "agent_id": manifest["agent_id"],
        "regime_id": manifest["regime"]["regime_id"],
        "model": "gpt-5.6-sol",
        "reasoning_effort": "max",
        "context_mode": "oracle_assisted_training",
        "approval_policy": "on-request",
        "approvals_reviewer": "auto_review",
        "sandbox_mode": "workspace-write",
        "network_allowed": True,
        "web_search": "live",
        "dangerous_sandbox_bypass": False,
        "instruction_and_workspace_allowlist_isolation": True,
        "user_authorized_external_dispatch": bool(
            user_authorized_external_dispatch
        ),
        "launchable": authorized,
        "allowed_external_files": files,
        "allowed_external_file_count": len(files),
        "allowed_external_scope": list(authorization["allowed"]),
        "explicit_non_upload_scope": list(authorization["forbidden"]),
        "solver_output_contract_version": MACRO_OUTPUT_CONTRACT,
        "generator_parameterization_visible": False,
        "private_evaluation_cohorts_visible": False,
    }
    payload["packet_sha256"] = _packet_digest(payload)
    return payload


def write_vpfp_robust_external_launch_packet(
    path: str | Path,
    packet: Mapping[str, Any],
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(packet), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def load_and_validate_vpfp_robust_external_launch_packet(
    packet_path: str | Path,
    *,
    workspace: str | Path,
    require_workspace_unchanged: bool = True,
) -> dict[str, Any]:
    path = Path(packet_path).resolve()
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    return validate_vpfp_robust_external_launch_packet(
        _read_json(path),
        workspace=workspace,
        require_workspace_unchanged=require_workspace_unchanged,
    )


def validate_vpfp_robust_external_launch_packet(
    packet: Mapping[str, Any],
    *,
    workspace: str | Path,
    require_workspace_unchanged: bool = True,
) -> dict[str, Any]:
    payload = dict(packet)
    supplied_digest = str(payload.pop("packet_sha256", ""))
    if payload.get("schema") != VPFP_ROBUST_EXTERNAL_LAUNCH_SCHEMA:
        raise ValueError("VPFP robust external launch packet is invalid")
    root = Path(workspace).resolve()
    if root != Path(str(payload.get("workspace") or "")).resolve():
        raise ValueError("VPFP robust launch workspace binding changed")
    if not (
        payload.get("launchable") is True
        and payload.get("user_authorized_external_dispatch") is True
        and payload.get("model") == "gpt-5.6-sol"
        and payload.get("reasoning_effort") == "max"
        and payload.get("approval_policy") == "on-request"
        and payload.get("approvals_reviewer") == "auto_review"
        and payload.get("sandbox_mode") == "workspace-write"
        and payload.get("network_allowed") is True
        and payload.get("dangerous_sandbox_bypass") is False
        and payload.get("generator_parameterization_visible") is False
        and payload.get("private_evaluation_cohorts_visible") is False
    ):
        raise ValueError("VPFP robust launch packet lacks authorization or safety")
    audit = audit_vpfp_robust_agent_workspace(root)
    if not audit["passed"]:
        raise ValueError("VPFP robust launch workspace no longer passes audit")
    if require_workspace_unchanged:
        current = scan_vpfp_robust_external_workspace(root)
        if current != payload.get("allowed_external_files"):
            raise ValueError("VPFP robust launch workspace changed after authorization")
    payload["packet_sha256"] = supplied_digest
    return payload


def scan_vpfp_robust_external_workspace(
    workspace: str | Path,
) -> list[dict[str, Any]]:
    root = Path(workspace).resolve()
    rows = []
    for item in sorted(root.rglob("*")):
        if item.is_symlink():
            raise ValueError(f"symlink is forbidden in external workspace: {item}")
        if not item.is_file():
            continue
        rows.append(
            {
                "path": item.relative_to(root).as_posix(),
                "bytes": int(item.stat().st_size),
                "sha256": _sha256(item),
            }
        )
    return rows


def _packet_digest(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("packet_sha256", None)
    canonical = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))
