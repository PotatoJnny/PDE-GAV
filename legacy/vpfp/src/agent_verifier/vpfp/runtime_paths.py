"""Paths accepted for persistent VPFP run files.

The original cluster scripts assumed that `/scratch` was always writable.
Clusters can temporarily fill that shared filesystem, so the product may set
`VPFP_RUN_STORAGE_ROOT` to one explicit project directory. Dense disposable
arrays still use `SLURM_TMPDIR`; this setting only chooses where completed run
files and Agent workspaces live.
"""

from __future__ import annotations

import os
from pathlib import Path


def persistent_run_roots() -> tuple[Path, ...]:
    roots = [Path("/scratch").resolve()]
    configured = os.environ.get("VPFP_RUN_STORAGE_ROOT", "").strip()
    if configured:
        root = Path(configured).expanduser().resolve()
        if root == Path(root.anchor):
            raise ValueError("VPFP_RUN_STORAGE_ROOT must be a specific directory")
        roots.append(root)
    return tuple(roots)


def is_persistent_run_path(value: str | Path) -> bool:
    path = Path(value).resolve()
    return any(path.is_relative_to(root) for root in persistent_run_roots())


def require_persistent_run_path(value: str | Path, label: str) -> Path:
    path = Path(value).resolve()
    if not is_persistent_run_path(path):
        allowed = ", ".join(str(root) for root in persistent_run_roots())
        raise ValueError(f"{label} must stay below one of: {allowed}")
    return path
