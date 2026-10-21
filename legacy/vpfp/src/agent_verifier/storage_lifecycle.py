"""Shared lifecycle primitives for dense scientific runtime artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


EPHEMERAL_SCOPE_SCHEMA = "agent_verifier_slurm_ephemeral_scope_v1"
_MARKER_NAME = ".agent_verifier_ephemeral_scope.json"
_SAFE_NAMESPACE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")


class StorageLifecycleError(RuntimeError):
    """Raised when runtime storage violates the declared lifecycle."""


def path_is_within(path: str | Path, root: str | Path) -> bool:
    """Return whether a resolved path is contained by a resolved root."""

    try:
        Path(path).expanduser().resolve(strict=False).relative_to(
            Path(root).expanduser().resolve(strict=False)
        )
    except ValueError:
        return False
    return True


def is_within_active_slurm_tmp(
    path: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Return whether a path is under the active job's node-local directory."""

    environment = os.environ if environ is None else environ
    if not environment.get("SLURM_JOB_ID") or not environment.get("SLURM_TMPDIR"):
        return False
    return path_is_within(path, environment["SLURM_TMPDIR"])


def sha256_file(path: str | Path) -> str:
    """Hash one regular file without loading it into memory."""

    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(source)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class SlurmEphemeralScope:
    """A marker-bound directory that is safe to remove after one Slurm job."""

    root: Path
    slurm_tmpdir: Path
    job_id: str
    namespace: str

    @classmethod
    def create(
        cls,
        namespace: str,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> "SlurmEphemeralScope":
        environment = os.environ if environ is None else environ
        job_id = str(environment.get("SLURM_JOB_ID") or "")
        raw_tmpdir = str(environment.get("SLURM_TMPDIR") or "")
        if not job_id or not raw_tmpdir:
            raise StorageLifecycleError(
                "dense runtime storage requires SLURM_JOB_ID and SLURM_TMPDIR"
            )
        if _SAFE_NAMESPACE.fullmatch(str(namespace)) is None:
            raise ValueError("ephemeral namespace must contain 1-96 safe characters")

        slurm_tmpdir = Path(raw_tmpdir).expanduser().resolve(strict=True)
        if not slurm_tmpdir.is_dir() or slurm_tmpdir.is_symlink():
            raise StorageLifecycleError("SLURM_TMPDIR must be a real directory")
        parent = slurm_tmpdir / "agent-verifier-runtime"
        parent.mkdir(mode=0o700, parents=False, exist_ok=True)
        if parent.is_symlink() or not path_is_within(parent, slurm_tmpdir):
            raise StorageLifecycleError("ephemeral runtime parent escaped SLURM_TMPDIR")

        leaf = f"{namespace}-{os.getpid()}-{uuid.uuid4().hex[:12]}"
        root = parent / leaf
        root.mkdir(mode=0o700, exist_ok=False)
        marker = {
            "schema": EPHEMERAL_SCOPE_SCHEMA,
            "job_id": job_id,
            "namespace": str(namespace),
            "root": str(root),
        }
        (root / _MARKER_NAME).write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return cls(
            root=root,
            slurm_tmpdir=slurm_tmpdir,
            job_id=job_id,
            namespace=str(namespace),
        )

    def contains(self, path: str | Path) -> bool:
        return path_is_within(path, self.root)

    def require_contains(self, path: str | Path) -> Path:
        resolved = Path(path).expanduser().resolve(strict=False)
        if not self.contains(resolved):
            raise StorageLifecycleError(
                f"runtime path must stay in ephemeral scope: {resolved}"
            )
        return resolved

    def cleanup(self) -> dict[str, object]:
        """Remove only this marker-bound scope and return compact accounting."""

        if not self.root.exists():
            return {
                "removed": False,
                "reason": "already_absent",
                "root": str(self.root),
            }
        if self.root.is_symlink() or not path_is_within(self.root, self.slurm_tmpdir):
            raise StorageLifecycleError("refusing to clean an unsafe ephemeral root")
        marker_path = self.root / _MARKER_NAME
        if not marker_path.is_file() or marker_path.is_symlink():
            raise StorageLifecycleError("ephemeral scope ownership marker is missing")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker != {
            "schema": EPHEMERAL_SCOPE_SCHEMA,
            "job_id": self.job_id,
            "namespace": self.namespace,
            "root": str(self.root),
        }:
            raise StorageLifecycleError("ephemeral scope ownership marker changed")
        shutil.rmtree(self.root)
        return {
            "removed": True,
            "reason": "job_scope_complete",
            "root": str(self.root),
        }
