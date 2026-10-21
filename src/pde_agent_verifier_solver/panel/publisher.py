"""Publish sanitized PDE Agent campaign metrics to the fixed web panel."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping
from urllib.request import Request, urlopen

from pde_agent_verifier_solver.panel.campaign_view import (
    build_hosted_dashboard_snapshot,
)


PANEL_SITE_ORIGIN = os.environ.get(
    "PDE_PANEL_SITE_ORIGIN", "https://pde-agent-panel.example.invalid"
)
PANEL_SNAPSHOT_ENDPOINT = PANEL_SITE_ORIGIN + "/api/snapshot"
_MAX_SNAPSHOT_BYTES = 7_500_000
_FORBIDDEN_KEYS = {
    "command",
    "commands",
    "dataset",
    "job_id",
    "log",
    "log_tail",
    "logs",
    "manifest",
    "run_root",
    "scheduler_id",
    "source",
}
_FORBIDDEN_TEXT = (
    "/home/",
    "/project/",
    "/scratch/",
    "sbatch ",
    "srun ",
    "ssh ",
)
_STATUS_ORDER = {"running": 0, "submitted": 1, "failed": 2, "completed": 3}
RunsRoots = Path | str | Iterable[Path | str]


class UnsafePanelSnapshot(ValueError):
    """Raised when a hosted payload contains server-only information."""


def validate_hosted_snapshot(snapshot: Mapping[str, object]) -> None:
    """Fail closed if private cluster material enters the hosted payload."""

    if snapshot.get("schema") != "pde_panel_snapshot_v1":
        raise UnsafePanelSnapshot("Unsupported panel snapshot schema")
    if not isinstance(snapshot.get("runs"), list):
        raise UnsafePanelSnapshot("Panel snapshot must contain campaign runs")

    def inspect(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key).lower() in _FORBIDDEN_KEYS:
                    raise UnsafePanelSnapshot(
                        "Hosted panel snapshot contains a server-only field"
                    )
                inspect(child)
        elif isinstance(value, list):
            for child in value:
                inspect(child)
        elif isinstance(value, str):
            lowered = value.lower()
            if any(fragment in lowered for fragment in _FORBIDDEN_TEXT):
                raise UnsafePanelSnapshot(
                    "Hosted panel snapshot contains a server-only value"
                )

    inspect(snapshot)


def build_combined_hosted_snapshot(runs_roots: RunsRoots) -> dict[str, object]:
    """Merge campaigns from every configured storage root into one panel."""

    runs_by_key: dict[str, dict[str, object]] = {}
    for root in _normalize_runs_roots(runs_roots):
        snapshot = build_hosted_dashboard_snapshot(root)
        for candidate in snapshot["runs"]:
            if not isinstance(candidate, dict):
                continue
            run_id = str(candidate.get("id") or "")
            if not run_id:
                continue
            key = f"{candidate.get('problem_id') or ''}:{run_id}"
            existing = runs_by_key.get(key)
            if existing is None or _freshness(candidate) > _freshness(existing):
                runs_by_key[key] = candidate

    runs = list(runs_by_key.values())
    runs.sort(key=lambda row: str(row.get("updated_at") or ""), reverse=True)
    runs.sort(key=lambda row: _STATUS_ORDER.get(str(row.get("status")), 4))
    counts = {
        status: sum(row.get("status") == status for row in runs)
        for status in ("running", "completed", "failed", "submitted")
    }
    return {
        "schema": "pde_panel_snapshot_v1",
        "synced_at": datetime.now(timezone.utc).isoformat(),
        "counts": counts,
        "runs": runs,
    }


def _normalize_runs_roots(runs_roots: RunsRoots) -> list[Path]:
    values: Iterable[Path | str]
    if isinstance(runs_roots, (str, Path)):
        values = (runs_roots,)
    else:
        values = runs_roots

    roots: list[Path] = []
    seen: set[Path] = set()
    for value in values:
        root = Path(value).expanduser().resolve()
        if root not in seen:
            seen.add(root)
            roots.append(root)
    if not roots:
        raise ValueError("At least one campaign storage root is required")
    return roots


def _freshness(run: Mapping[str, object]) -> tuple[str, int]:
    status_rank = {
        "submitted": 0,
        "failed": 1,
        "running": 2,
        "completed": 3,
    }
    return (
        str(run.get("updated_at") or run.get("created_at") or ""),
        status_rank.get(str(run.get("status")), -1),
    )


def load_sync_token(token_file: Path | str | None = None) -> str:
    """Read the write-only panel credential without putting it on the command line."""

    token = os.environ.get("PDE_PANEL_SYNC_TOKEN", "").strip()
    if not token and token_file:
        token = Path(token_file).read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError("PDE panel sync credential is unavailable")
    return token


def publish_hosted_snapshot(
    runs_roots: RunsRoots,
    token: str,
    *,
    opener: Callable[..., Any] = urlopen,
) -> dict[str, object]:
    """Publish one safe combined snapshot to the single fixed destination."""

    snapshot = build_combined_hosted_snapshot(runs_roots)
    validate_hosted_snapshot(snapshot)
    body = json.dumps(
        snapshot,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(body) > _MAX_SNAPSHOT_BYTES:
        raise UnsafePanelSnapshot("Hosted panel snapshot is too large")

    request = Request(
        PANEL_SNAPSHOT_ENDPOINT,
        data=body,
        method="POST",
        headers={
            "Accept": "application/json",
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json",
            "User-Agent": "PDE-Agent-Panel-Sync/1.0",
        },
    )
    with opener(request, timeout=30) as response:
        status = int(getattr(response, "status", 200))
        response_body = response.read()
    if status != 200:
        raise RuntimeError("PDE panel rejected the metric snapshot")
    try:
        acknowledgement = json.loads(response_body or b"{}")
    except json.JSONDecodeError as error:
        raise RuntimeError("PDE panel returned an invalid response") from error
    if acknowledgement.get("ok") is not True:
        raise RuntimeError("PDE panel did not acknowledge the metric snapshot")
    return snapshot


def run_panel_sync(
    runs_roots: RunsRoots,
    token: str,
    *,
    interval_seconds: float = 5.0,
    once: bool = False,
) -> None:
    """Keep the fixed panel current while campaigns run on the cluster."""

    interval = max(float(interval_seconds), 1.0)
    while True:
        try:
            snapshot = publish_hosted_snapshot(runs_roots, token)
        except UnsafePanelSnapshot:
            raise
        except Exception as error:
            if once:
                raise
            stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
            print(
                f"[{stamp}] Panel sync is retrying after {type(error).__name__}.",
                flush=True,
            )
        else:
            stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
            print(
                f"[{stamp}] Synced {len(snapshot['runs'])} PDE campaigns.",
                flush=True,
            )
            if once:
                return
        time.sleep(interval)
