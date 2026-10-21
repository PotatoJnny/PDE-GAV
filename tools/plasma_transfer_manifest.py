#!/usr/bin/env python3
"""Inventory explicitly allowed assets, then seal/verify their bytes; never transfer.

Planning only reads the small JSON spec and file metadata. Run large sealing or
verification jobs on a compute node, not a cluster login node. Payload files live
under ``fs/<absolute source without its leading slash>``. No credentials are read.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import socket
import stat
import sys
import tempfile

SCHEMA = "plasma_transfer_inventory_v1"
CACHES = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
SECRETS = {".ssh", ".aws", ".codex", ".gnupg", ".netrc", ".git-credentials",
           "auth.json", "credentials", "credentials.json", "credentials.yaml", "credentials.yml",
           "token", "token.json", "tokens.json", "secrets.json", "secrets.yaml", "secrets.yml",
           "id_rsa", "id_ed25519"}


def _path(value, *, absolute=True):
    if not isinstance(value, str) or not value or any(c in value for c in "\n\r\0"):
        raise ValueError("path must be nonempty and contain no newline or NUL")
    parts = value.split("/")
    if ".." in parts or any(p.lower() in SECRETS or p.lower().startswith((".env", ".secret")) for p in parts):
        raise ValueError(f"unsafe or secret path: {value}")
    if any(a == ".config" and b in {"gh", "gcloud"} for a, b in zip(parts, parts[1:])):
        raise ValueError(f"unsafe or secret path: {value}")
    path = Path(value)
    if path.is_absolute() != absolute or "." in parts:
        raise ValueError(f"path must be {'absolute' if absolute else 'relative'} and canonical: {value}")
    return path


def _checked(path):
    path = _path(str(path))
    for part in reversed((path, *path.parents)):
        result = part.lstat()
        if stat.S_ISLNK(result.st_mode):
            raise ValueError(f"symlink is not allowed: {part}")
    return result


def _signature(info):
    return {key: getattr(info, "st_" + key) for key in ("dev", "ino", "size", "mtime_ns", "ctime_ns")}


def _load(path):
    path = Path(path).absolute()
    if not stat.S_ISREG(_checked(path).st_mode):
        raise ValueError("JSON input must be a regular file")
    return json.loads(path.read_text())


def _write_new(path, text):
    path = Path(path).absolute()
    _path(str(path))
    _checked(path.parent)
    # A hard-link publishes a complete file atomically without replacing a target.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".transfer-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink()


def _json(value):
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _digest(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("sha256 must contain 64 lowercase hexadecimal characters")
    return value


def _destination(source):
    return "fs/" + str(source).lstrip("/")


def plan(spec, output):
    spec = _load(spec)
    if spec.get("schema") != "plasma_transfer_spec_v1":
        raise ValueError("unsupported transfer spec schema")
    repository = spec.get("code_repository")
    url = repository.get("url") if isinstance(repository, dict) else repository
    if not isinstance(url, str) or not url or not isinstance(spec.get("code_revision"), str) or not spec["code_revision"]:
        raise ValueError("code repository URL and fixed scientific revision are required")
    files, exclusions, targets = [], [], set()
    entries = spec.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("spec requires explicit nonempty entries")
    for entry in entries:
        source = _path(entry["source"])
        info = _checked(source)
        if entry.get("destination") != _destination(source):
            raise ValueError("destination must equal fs/<absolute source without leading slash>")
        role = entry.get("role")
        if not isinstance(role, str) or not role.strip():
            raise ValueError("each entry requires a role")
        pinned = entry.get("sha256")
        if pinned is not None:
            _digest(pinned)
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("pinned sha256 is supported only for file entries")
        if stat.S_ISDIR(info.st_mode) and (len(source.parts) <= 2 or
                (source.parts[1] in {"home", "scratch", "project"} and len(source.parts) <= 3)):
            raise ValueError(f"directory allowlist is too broad: {source}")
        pending = [source]
        while pending:
            path = pending.pop()
            info = _checked(path)
            if stat.S_ISDIR(info.st_mode):
                if path.name in CACHES:
                    exclusions.append({"source": str(path), "reason": "non-scientific cache or Git metadata"})
                else:
                    pending.extend(sorted(path.iterdir(), reverse=True))
                continue
            if not stat.S_ISREG(info.st_mode):
                raise ValueError(f"only regular files and directories are allowed: {path}")
            destination = _destination(path)
            if destination in targets:
                raise ValueError(f"duplicate destination: {destination}")
            targets.add(destination)
            row = {"source": str(path), "destination": destination, "role": role,
                   "size": info.st_size, "mode": stat.S_IMODE(info.st_mode), "snapshot": _signature(info)}
            if pinned is not None:
                row["pinned_sha256"] = pinned
            files.append(row)
    if not files:
        raise ValueError("allowlist contains no regular files")
    files.sort(key=lambda row: row["destination"])
    inventory = {"schema": SCHEMA, "status": "planned", "code_repository": repository,
                 "code_revision": spec["code_revision"], "entries": entries, "files": files,
                 "excluded": exclusions, "file_count": len(files), "total_bytes": sum(row["size"] for row in files)}
    output = Path(output).absolute()
    _path(str(output))
    _checked(output.parent)
    output.mkdir(mode=0o700)
    _write_new(output / "rsync-files.txt", "".join(row["source"].lstrip("/") + "\n" for row in files))
    _write_new(output / "inventory.json", _json(inventory))
    return inventory


def _inventory(path, status):
    result = _load(path)
    if result.get("schema") != SCHEMA or result.get("status") != status or not result.get("files"):
        raise ValueError(f"expected {status} transfer inventory")
    seen = set()
    for row in result["files"]:
        source = _path(row["source"])
        destination = str(_path(row["destination"], absolute=False))
        if destination != _destination(source) or destination in seen:
            raise ValueError("invalid or duplicate inventory destination")
        if type(row["size"]) is not int or row["size"] < 0:
            raise ValueError("invalid inventory size")
        if type(row.get("mode")) is not int or not 0 <= row["mode"] <= 0o7777:
            raise ValueError("invalid inventory file mode")
        snapshot = row.get("snapshot")
        if not isinstance(snapshot, dict) or set(snapshot) != {"dev", "ino", "size", "mtime_ns", "ctime_ns"} or any(
                type(value) is not int for value in snapshot.values()) or snapshot["size"] != row["size"]:
            raise ValueError("invalid inventory source snapshot")
        if row.get("pinned_sha256") is not None:
            _digest(row["pinned_sha256"])
        seen.add(destination)
        if status == "sealed":
            _digest(row["sha256"])
    if result.get("file_count") != len(seen) or result.get("total_bytes") != sum(row["size"] for row in result["files"]):
        raise ValueError("inventory totals are inconsistent")
    return result


def _hash_file(path, expected_snapshot=None):
    before = _checked(path)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"not a regular file: {path}")
    if expected_snapshot is not None and _signature(before) != expected_snapshot:
        raise ValueError(f"source changed since planning: {path}")
    digest = sha256()
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        if _signature(os.fstat(stream.fileno())) != _signature(before):
            raise ValueError(f"source changed before hashing: {path}")
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        after = os.fstat(stream.fileno())
    if _signature(before) != _signature(after) or _signature(before) != _signature(_checked(path)):
        raise ValueError(f"source changed while hashing: {path}")
    return digest.hexdigest(), before


def seal(manifest, output):
    result = _inventory(manifest, "planned")
    if "login" in socket.gethostname().split(".")[0].lower() and result["total_bytes"] > 256 * 1024 * 1024:
        raise ValueError("large seal must run on a data-transfer or allocated CPU compute node, not a login node")
    print("Hashing explicit assets: use a compute node for large inventories.", file=sys.stderr)
    for row in result["files"]:
        digest, info = _hash_file(Path(row["source"]), row["snapshot"])
        if info.st_size != row["size"] or (row.get("pinned_sha256") is not None and digest != row["pinned_sha256"]):
            raise ValueError(f"pinned SHA or size mismatch: {row['source']}")
        row["sha256"] = digest
    result["status"] = "sealed"
    _write_new(output, _json(result))
    return result


def verify(manifest, root):
    result = _inventory(manifest, "sealed")
    root = Path(root).absolute()
    if not stat.S_ISDIR(_checked(root).st_mode):
        raise ValueError("payload root must be a directory")
    for row in result["files"]:
        digest, info = _hash_file(root / row["destination"])
        if info.st_size != row["size"] or digest != row["sha256"]:
            raise ValueError(f"payload size/SHA mismatch: {row['destination']}")
        if stat.S_IMODE(info.st_mode) & 0o111 != row["mode"] & 0o111:
            raise ValueError(f"payload executable permissions differ: {row['destination']}")
    return {"status": "verified", "file_count": result["file_count"], "total_bytes": result["total_bytes"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "seal", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--spec" if name == "plan" else "--manifest", required=True)
        command.add_argument("--root" if name == "verify" else "--output", required=True)
    args = parser.parse_args()
    result = (plan(args.spec, args.output) if args.command == "plan" else
              seal(args.manifest, args.output) if args.command == "seal" else verify(args.manifest, args.root))
    print(_json({key: result[key] for key in ("status", "file_count", "total_bytes")}), end="")


if __name__ == "__main__":
    main()
