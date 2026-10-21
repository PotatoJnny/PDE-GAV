#!/usr/bin/env python3
"""Bind a VPFP submission and local assets into one immutable bundle identity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


SCHEMA = "vpfp_solver_artifact_bundle_v1"
MANIFEST = "solver_artifact_bundle.json"
ALLOWED_SUFFIXES = {
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bind_solver_bundle(
    submission: str | Path,
    *,
    assets: list[str] | tuple[str, ...] = (),
    force: bool = False,
) -> dict | None:
    raw_submission = Path(submission)
    if raw_submission.is_symlink():
        raise ValueError("submission entrypoint cannot be a symlink")
    source = raw_submission.resolve()
    if not source.is_file() or source.suffix != ".py":
        raise ValueError("submission entrypoint must be a Python file")
    root = source.parent
    manifest_path = root / MANIFEST
    if manifest_path.exists():
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise ValueError("bundle manifest must be a regular file")
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            current.get("schema") != SCHEMA
            or current.get("entrypoint", "submission.py") != "submission.py"
        ):
            raise ValueError("bundle manifest schema or entrypoint is invalid")
        declared = list(current.get("files") or ())
    elif force or assets:
        declared = ["submission.py"]
    else:
        return None
    if source.name != "submission.py":
        raise ValueError("bundled entrypoint must be named submission.py")
    if assets:
        declared = ["submission.py", *[str(asset) for asset in assets]]
    if "submission.py" not in declared:
        raise ValueError("bundle must declare submission.py")

    files: dict[str, Path] = {}
    for raw in declared:
        relative = PurePosixPath(str(raw))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError(f"unsafe bundle path: {raw!r}")
        canonical = relative.as_posix()
        if canonical == MANIFEST:
            raise ValueError("bundle manifest cannot declare itself")
        if Path(canonical).suffix.lower() not in ALLOWED_SUFFIXES:
            raise ValueError(f"unsupported bundle asset: {canonical}")
        if canonical in files:
            raise ValueError(f"duplicate bundle asset: {canonical}")
        raw_item = root / Path(*relative.parts)
        item = raw_item.resolve()
        if (
            raw_item.is_symlink()
            or not item.is_relative_to(root)
            or not item.is_file()
        ):
            raise ValueError(f"bundle asset must be a regular local file: {canonical}")
        files[canonical] = item

    digest = hashlib.sha256()
    total_bytes = 0
    separator = bytes((0,))
    for relative in sorted(files):
        item = files[relative]
        total_bytes += item.stat().st_size
        digest.update(
            relative.encode("utf-8")
            + separator
            + _sha256_file(item).encode("ascii")
            + separator
        )
    payload = {
        "aggregate_sha256": digest.hexdigest(),
        "entrypoint": "submission.py",
        "files": sorted(files),
        "schema": SCHEMA,
        "total_bytes": total_bytes,
    }
    temporary = root / f".{MANIFEST}.tmp"
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", required=True)
    parser.add_argument("--asset", action="append", default=[])
    args = parser.parse_args()
    try:
        payload = bind_solver_bundle(
            args.submission,
            assets=args.asset,
            force=True,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
