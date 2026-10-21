"""Offline exact-commit source transport without ancestor history.

Only a commit and its complete tree are packed. Restore retains the original
commit bytes and uses a genuine shallow boundary when parents are omitted.
This transports source evidence; it does not qualify a new numerical runtime.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess

SCHEMA = "plasma_exact_git_snapshot_v1"
MAX_LOGIN_OBJECT_BYTES = 64 * 1024 * 1024
_OID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_SECRET_COMPONENTS = {
    ".ssh", ".aws", ".gnupg", ".azure", ".kube", ".netrc", ".npmrc",
    ".pypirc", ".credentials", ".codex", ".git-credentials", "auth.json", "credentials.json",
    "credentials", "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519",
}


def _environment():
    # Do not allow shell Git settings to redirect the index/object database or
    # inject replacement objects, global filters, or another repository.
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
               GIT_NO_REPLACE_OBJECTS="1", GIT_TERMINAL_PROMPT="0")
    return env


def _git(repository, *arguments, input=None):
    result = subprocess.run(
        ["git", "-c", "core.hooksPath=" + os.devnull, "-C", str(repository), *arguments],
        input=input, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=_environment(), check=False)
    if result.returncode:
        raise ValueError("git command failed: " + " ".join(arguments[:2]) + ": "
                         + result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout


def _path(value, *, existing=False):
    path = Path(value).absolute()
    if ".." in path.parts or any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("normalized nonsymlink path required: " + str(path))
    if existing and not path.is_dir():
        raise ValueError("existing directory required: " + str(path))
    return path


def _fresh_directory(value):
    path = _path(value)
    if path.exists() or not path.parent.is_dir():
        raise ValueError("output must be a new directory with an existing parent: " + str(path))
    path.mkdir(mode=0o700)
    return path


def _digest(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _oid(value):
    if not isinstance(value, str) or not _OID.fullmatch(value):
        raise ValueError("full lowercase Git object ID required")
    return value


def _safe_tree_path(raw):
    try:
        value = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("source paths must be UTF-8") from error
    path = PurePosixPath(value)
    parts = value.split("/")
    if (not value or path.is_absolute() or any(part in ("", ".", "..") for part in parts)
            or any(ord(character) < 32 for character in value) or "\\" in value):
        raise ValueError("unsafe source path: " + repr(value))
    for index, part in enumerate(parts):
        lower = part.lower()
        if (lower in _SECRET_COMPONENTS or lower == ".git" or lower.startswith(".env")
                or (index and parts[index - 1].lower() == ".config" and lower in ("gh", "gcloud"))
                or lower.endswith((".pem", ".key", ".p12", ".pfx"))):
            raise ValueError("secret-like source path is not transferable: " + value)
    return value


def _inventory(repository, revision):
    revision = _oid(revision)
    commit = _git(repository, "cat-file", "commit", revision)
    headers = commit.split(b"\n\n", 1)[0].splitlines()
    roots = [line[5:].decode("ascii") for line in headers if line.startswith(b"tree ")]
    if len(roots) != 1:
        raise ValueError("one commit root tree required")
    root_tree = _oid(roots[0])
    parents = [_oid(line[7:].decode("ascii")) for line in headers if line.startswith(b"parent ")]
    object_ids = {revision, root_tree}
    paths = []
    raw = _git(repository, "ls-tree", "-r", "-t", "-z", "--full-tree", revision)
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        metadata, raw_path = entry.split(b"\t", 1)
        mode, kind, object_id = metadata.decode("ascii").split()
        path = _safe_tree_path(raw_path)
        if (mode, kind) not in (("040000", "tree"), ("100644", "blob"), ("100755", "blob")):
            raise ValueError("symlinks, gitlinks and nonregular source entries are forbidden: " + path)
        object_ids.add(_oid(object_id))
        if kind == "blob":
            paths.append(path)
    return {"revision": revision, "root_tree": root_tree, "parents": parents,
            "has_parents": bool(parents), "file_count": len(paths),
            "object_count": len(object_ids)}, sorted(object_ids)


def _source_identity(repository, revision):
    if Path(_git(repository, "rev-parse", "--show-toplevel").decode().strip()) != Path(repository).absolute():
        raise ValueError("repository must be the explicit checkout root, not a subdirectory")
    if _git(repository, "rev-parse", "HEAD").decode().strip() != revision:
        raise ValueError("source HEAD must equal the requested revision")
    if _git(repository, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("source checkout must be clean, including untracked files")


def _object_bytes(repository, object_ids):
    raw = _git(repository, "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)",
               input=("\n".join(object_ids) + "\n").encode())
    rows = [line.split() for line in raw.decode("ascii").splitlines()]
    if (len(rows) != len(object_ids) or any(len(row) != 3 or row[0] != expected
            or row[1] not in ("commit", "tree", "blob") or not row[2].isdigit()
            for row, expected in zip(rows, object_ids))):
        raise ValueError("selected Git object inventory is incomplete")
    return sum(int(row[2]) for row in rows)


def export_snapshot(repository, revision, output):
    repository = _path(repository, existing=True)
    revision = _oid(revision)
    if _path(output).is_relative_to(repository):
        raise ValueError("snapshot output must be outside the source checkout")
    _source_identity(repository, revision)
    inventory, object_ids = _inventory(repository, revision)
    object_bytes = _object_bytes(repository, object_ids)
    if object_bytes > MAX_LOGIN_OBJECT_BYTES and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("source export exceeds the 64 MiB login-node limit; use a compute allocation")
    output = _fresh_directory(output)
    temporary = output / "snapshot.pack.incomplete"
    with temporary.open("xb") as stream:
        result = subprocess.run(
            ["git", "-c", "pack.threads=1", "-C", str(repository), "pack-objects", "--stdout",
             "--compression=1", "--window=0", "--no-reuse-delta"],
            input=("\n".join(object_ids) + "\n").encode(), stdout=stream,
            stderr=subprocess.PIPE, env=_environment(), check=False)
    if result.returncode:
        raise ValueError("exact-object Git packing failed: "
                         + result.stderr.decode("utf-8", errors="replace"))
    _source_identity(repository, revision)
    pack = output / "snapshot.pack"
    temporary.rename(pack)
    receipt = {"schema": SCHEMA, **inventory,
               "object_format": _git(repository, "rev-parse", "--show-object-format").decode().strip(),
               "uncompressed_object_bytes": object_bytes, "pack_bytes": pack.stat().st_size,
               "pack_sha256": _digest(pack), "includes_ancestor_history": False,
               "source_repository": str(repository)}
    with (output / "receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return receipt


def restore_snapshot(snapshot, output):
    snapshot = _path(snapshot, existing=True)
    pack, receipt_path = snapshot / "snapshot.pack", snapshot / "receipt.json"
    if any(path.is_symlink() or not path.is_file() for path in (pack, receipt_path)):
        raise ValueError("regular nonsymlink snapshot files required")
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("schema") != SCHEMA or receipt.get("includes_ancestor_history") is not False
            or receipt.get("object_format") not in ("sha1", "sha256")
            or pack.stat().st_size != receipt.get("pack_bytes")
            or _digest(pack) != receipt.get("pack_sha256")):
        raise ValueError("snapshot receipt or pack digest differs")
    revision = _oid(receipt["revision"])
    output = _fresh_directory(output)
    _git(output, "init", "--quiet", "--template=", "--object-format=" + receipt["object_format"])
    with pack.open("rb") as stream:
        result = subprocess.run(["git", "-C", str(output), "index-pack", "--stdin"],
                                stdin=stream, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                env=_environment(), check=False)
    if result.returncode:
        raise ValueError("snapshot Git pack is invalid: "
                         + result.stderr.decode("utf-8", errors="replace"))
    inventory, expected_objects = _inventory(output, revision)
    if any(receipt.get(key) != value for key, value in inventory.items()):
        raise ValueError("restored commit/tree inventory differs from receipt")
    indices = list((output / ".git/objects/pack").glob("*.idx"))
    if len(indices) != 1:
        raise ValueError("one self-contained snapshot pack required")
    verified = _git(output, "verify-pack", "-v", str(indices[0])).decode("ascii")
    actual_objects = [line.split()[0] for line in verified.splitlines()
                      if line.split() and _OID.fullmatch(line.split()[0])]
    if sorted(actual_objects) != expected_objects:
        raise ValueError("snapshot pack contains missing or extra objects, including ancestor history")
    if _object_bytes(output, expected_objects) != receipt.get("uncompressed_object_bytes"):
        raise ValueError("restored object sizes differ from receipt")
    if inventory["has_parents"]:
        # Preserve the original commit, including parent headers/signatures.
        # This is Git's shallow-clone boundary, not a synthesized commit.
        (output / ".git/shallow").write_text(revision + "\n", encoding="ascii")
    _git(output, "update-ref", "refs/heads/frozen", revision)
    _git(output, "symbolic-ref", "HEAD", "refs/heads/frozen")
    _git(output, "-c", "core.autocrlf=false", "checkout", "--force", "frozen")
    _source_identity(output, revision)
    return {"schema": SCHEMA, "status": "restored", "revision": revision,
            "root_tree": inventory["root_tree"], "output": str(output),
            "clean": True, "shallow": inventory["has_parents"],
            "pack_sha256": receipt["pack_sha256"], "numerical_runtime_qualified": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="pack one clean HEAD and its complete tree")
    export.add_argument("--repository", required=True)
    export.add_argument("--revision", required=True)
    export.add_argument("--output", required=True)
    restore = commands.add_parser("restore", help="restore an offline exact-commit shallow checkout")
    restore.add_argument("--snapshot", required=True)
    restore.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = (export_snapshot(args.repository, args.revision, args.output)
              if args.command == "export" else restore_snapshot(args.snapshot, args.output))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
