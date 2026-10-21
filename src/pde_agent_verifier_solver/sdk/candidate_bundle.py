"""Backend-neutral identities for multi-file ROM candidate bundles.

An Agent declares only an entrypoint and the files that belong to its bundle.
The controller derives the identity from the bytes it reads; hashes or sizes in
an Agent-authored manifest are optional metadata, not loading gates. The tree
digest is independent of mtimes, modes, manifest formatting, and source root.

This module performs filesystem hygiene and best-effort immutable staging.  It
does not provide a security sandbox for executing untrusted candidate code.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from typing import Iterable, Mapping


CANDIDATE_BUNDLE_SCHEMA = "pde_agent_verifier.candidate_bundle.v1"
CANDIDATE_BUNDLE_MANIFEST = "candidate_bundle.json"
DEFAULT_MAX_BUNDLE_BYTES = 512 * 1024 * 1024
DEFAULT_MAX_BUNDLE_FILES = 4096
DEFAULT_MAX_MANIFEST_BYTES = 1024 * 1024
DEFAULT_MAX_PATH_BYTES = 1024
_DIGEST_DOMAIN = b"pde-agent-verifier-candidate-tree-v1\0"
_COPY_BLOCK_BYTES = 1024 * 1024


class CandidateBundleError(ValueError):
    """A candidate bundle is malformed, unsafe, too large, or changed."""


@dataclass(frozen=True)
class CandidateBundleLimits:
    """Controller-owned resource limits for one submitted artifact."""

    max_total_bytes: int = DEFAULT_MAX_BUNDLE_BYTES
    max_file_bytes: int = DEFAULT_MAX_BUNDLE_BYTES
    max_files: int = DEFAULT_MAX_BUNDLE_FILES
    max_manifest_bytes: int = DEFAULT_MAX_MANIFEST_BYTES
    max_path_bytes: int = DEFAULT_MAX_PATH_BYTES

    def __post_init__(self) -> None:
        for name in (
            "max_total_bytes",
            "max_file_bytes",
            "max_files",
            "max_manifest_bytes",
            "max_path_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, order=True)
class CandidateBundleMember:
    """One regular file captured in a candidate tree identity."""

    path: str
    byte_length: int
    sha256: str


@dataclass(frozen=True)
class CandidateBundle:
    """Immutable description of the exact bytes in one candidate tree."""

    root: Path
    entrypoint: str
    members: tuple[CandidateBundleMember, ...]
    total_bytes: int
    tree_sha256: str

    @property
    def entrypoint_path(self) -> Path:
        return self.root.joinpath(*PurePosixPath(self.entrypoint).parts)

    @property
    def manifest_path(self) -> Path:
        return self.root / CANDIDATE_BUNDLE_MANIFEST

    def normalized_manifest(self) -> dict[str, object]:
        """Return the canonical, controller-computed manifest payload."""

        return {
            "schema": CANDIDATE_BUNDLE_SCHEMA,
            "entrypoint": self.entrypoint,
            "files": [member.path for member in self.members],
            "total_bytes": self.total_bytes,
            "tree_sha256": self.tree_sha256,
        }


def write_candidate_bundle_manifest(
    root: Path | str,
    *,
    entrypoint: str = "submission.py",
    files: Iterable[str],
    overwrite: bool = False,
    limits: CandidateBundleLimits | None = None,
) -> Path:
    """Write a minimal manifest and check member paths, types and sizes.

    This is a convenience for candidate authors.  It intentionally does not
    record Agent-computed digests; :func:`inspect_candidate_bundle` computes
    those inside the controller boundary.
    """

    checked_limits = limits or CandidateBundleLimits()
    bundle_root = _regular_directory(root, "candidate bundle root")
    normalized_entrypoint = _normalize_relative_path(
        entrypoint,
        max_path_bytes=checked_limits.max_path_bytes,
    )
    normalized_files = _normalize_file_list(files, checked_limits)
    if normalized_entrypoint not in normalized_files:
        raise CandidateBundleError("candidate bundle files must include the entrypoint")
    manifest_path = bundle_root / CANDIDATE_BUNDLE_MANIFEST
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(manifest_path)
    if manifest_path.is_symlink():
        raise CandidateBundleError("candidate bundle manifest cannot be a symlink")
    payload = {
        "schema": CANDIDATE_BUNDLE_SCHEMA,
        "entrypoint": normalized_entrypoint,
        "files": list(normalized_files),
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if len(encoded) > checked_limits.max_manifest_bytes:
        raise CandidateBundleError("candidate bundle manifest exceeds its byte limit")
    flags = "wb" if overwrite else "xb"
    with manifest_path.open(flags) as handle:
        handle.write(encoded)
    try:
        inspect_candidate_bundle(manifest_path, limits=checked_limits, recorded_identity="manifest-only")
    except Exception:
        manifest_path.unlink(missing_ok=True)
        raise
    return manifest_path


def inspect_candidate_bundle(
    locator: Path | str,
    *,
    limits: CandidateBundleLimits | None = None,
    recorded_identity: str | None = None,
) -> CandidateBundle:
    """Inspect a manifest bundle or a backwards-compatible one-file candidate.

    A directory locator must contain ``candidate_bundle.json``.  A regular file
    locator uses the adjacent manifest when present; otherwise that file alone
    is treated as the entrypoint.  Manifest bundles are closed trees: every
    regular file below the root must be declared and symlinks are forbidden.
    """

    checked_limits = limits or CandidateBundleLimits()
    root, requested_entrypoint, manifest_path = _resolve_locator(
        locator,
        checked_limits,
    )
    if manifest_path is None:
        assert requested_entrypoint is not None
        entrypoint = requested_entrypoint
        file_paths = (entrypoint,)
    else:
        manifest = _read_manifest(manifest_path, checked_limits)
        entrypoint = _normalize_relative_path(
            _required_string(manifest, "entrypoint"),
            max_path_bytes=checked_limits.max_path_bytes,
        )
        if requested_entrypoint is not None and requested_entrypoint != entrypoint:
            raise CandidateBundleError(
                "submitted entrypoint differs from candidate bundle manifest"
            )
        raw_files = manifest.get("files")
        if not isinstance(raw_files, list):
            raise CandidateBundleError("candidate bundle manifest files must be a list")
        file_paths = _normalize_file_list(raw_files, checked_limits)
        if entrypoint not in file_paths:
            raise CandidateBundleError("candidate bundle files must include the entrypoint")
        # Historical checksums are metadata, not a condition for loading files.
        actual_paths = _enumerate_closed_tree(root, checked_limits)
        if set(actual_paths) != set(file_paths):
            missing = sorted(set(file_paths) - set(actual_paths))
            undeclared = sorted(set(actual_paths) - set(file_paths))
            raise CandidateBundleError(
                "candidate manifest does not match the closed file tree: "
                f"missing={missing}, undeclared={undeclared}"
            )

    members: list[CandidateBundleMember] = []
    total_bytes = 0
    for relative in file_paths:
        if recorded_identity is None:
            byte_length, digest = _hash_regular_member(root, relative, checked_limits)
        else:
            # Submitted programs already have an ID. Workers need filenames and
            # sizes, not another pass over checkpoint/model bytes.
            path = _member_path(root, relative)
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode):
                raise CandidateBundleError(f"candidate bundle member is not regular: {relative}")
            byte_length, digest = metadata.st_size, ""
            if byte_length > checked_limits.max_file_bytes:
                raise CandidateBundleError(f"candidate bundle member exceeds max_file_bytes: {relative}")
        total_bytes += byte_length
        if total_bytes > checked_limits.max_total_bytes:
            raise CandidateBundleError(
                "candidate bundle exceeds max_total_bytes: "
                f"{total_bytes} > {checked_limits.max_total_bytes}"
            )
        members.append(CandidateBundleMember(relative, byte_length, digest))
    members.sort(key=lambda member: member.path)
    if not members or total_bytes == 0:
        raise CandidateBundleError("candidate bundle cannot be empty")
    entrypoint_member = next(member for member in members if member.path == entrypoint)
    if entrypoint_member.byte_length == 0:
        raise CandidateBundleError("candidate bundle entrypoint cannot be empty")
    tree_sha256 = recorded_identity if recorded_identity is not None else _tree_digest(entrypoint, members)
    return CandidateBundle(
        root=root,
        entrypoint=entrypoint,
        members=tuple(members),
        total_bytes=total_bytes,
        tree_sha256=tree_sha256,
    )


def freeze_candidate_bundle(
    locator: Path | str | CandidateBundle,
    storage_root: Path | str,
    *,
    limits: CandidateBundleLimits | None = None,
) -> CandidateBundle:
    """Copy exact candidate bytes into a digest-named, read-only directory.

    The copy is atomically published. The ID is recorded at submission; workers
    reuse it without reading all model files to recompute checksums.
    Read-only modes are a best-effort immutability aid, not a sandbox boundary.
    """

    checked_limits = limits or CandidateBundleLimits()
    if isinstance(locator, CandidateBundle):
        source = locator
    else:
        source = inspect_candidate_bundle(locator, limits=checked_limits)
    raw_storage = Path(storage_root)
    if raw_storage.is_symlink():
        raise CandidateBundleError("candidate bundle storage root cannot be a symlink")
    raw_storage.mkdir(parents=True, exist_ok=True)
    storage = _regular_directory(raw_storage, "candidate bundle storage root")
    destination = storage / source.tree_sha256
    if destination.exists():
        return _existing_bundle(destination, source, checked_limits)

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{source.tree_sha256}.staging-", dir=storage)
    )
    published = False
    try:
        for member in source.members:
            target = temporary.joinpath(*PurePosixPath(member.path).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            _copy_member(source.root, member, target, checked_limits)
        manifest_bytes = (
            json.dumps(source.normalized_manifest(), indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        if len(manifest_bytes) > checked_limits.max_manifest_bytes:
            raise CandidateBundleError("normalized manifest exceeds its byte limit")
        (temporary / CANDIDATE_BUNDLE_MANIFEST).write_bytes(manifest_bytes)
        _make_tree_read_only(temporary)
        try:
            temporary.rename(destination)
            published = True
        except FileExistsError:
            return _existing_bundle(destination, source, checked_limits)
    finally:
        if not published and temporary.exists():
            _remove_staging_tree(temporary)
    return replace(source, root=destination)


def _resolve_locator(
    locator: Path | str,
    limits: CandidateBundleLimits,
) -> tuple[Path, str | None, Path | None]:
    raw = Path(locator)
    if raw.is_symlink():
        raise CandidateBundleError("candidate bundle locator cannot be a symlink")
    if raw.is_dir():
        root = raw.resolve()
        manifest = root / CANDIDATE_BUNDLE_MANIFEST
        if not manifest.exists():
            raise CandidateBundleError(
                f"candidate bundle directory lacks {CANDIDATE_BUNDLE_MANIFEST}"
            )
        return root, None, manifest
    if not raw.is_file():
        raise FileNotFoundError(raw)
    resolved = raw.resolve()
    root = resolved.parent
    if resolved.name == CANDIDATE_BUNDLE_MANIFEST:
        return root, None, resolved
    entrypoint = _normalize_relative_path(
        resolved.name,
        max_path_bytes=limits.max_path_bytes,
    )
    manifest = root / CANDIDATE_BUNDLE_MANIFEST
    return root, entrypoint, manifest if manifest.exists() else None


def _read_manifest(path: Path, limits: CandidateBundleLimits) -> Mapping[str, object]:
    if path.is_symlink() or not path.is_file():
        raise CandidateBundleError("candidate bundle manifest must be a regular file")
    payload = _read_regular_file(path, max_bytes=limits.max_manifest_bytes)
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CandidateBundleError("candidate bundle manifest is not valid UTF-8 JSON") from exc
    if not isinstance(decoded, dict):
        raise CandidateBundleError("candidate bundle manifest must be a JSON object")
    allowed = {"schema", "entrypoint", "files", "total_bytes", "tree_sha256"}
    unknown = sorted(set(decoded) - allowed)
    if unknown:
        raise CandidateBundleError(f"candidate bundle manifest has unknown keys: {unknown}")
    if decoded.get("schema") != CANDIDATE_BUNDLE_SCHEMA:
        raise CandidateBundleError("candidate bundle manifest schema is unsupported")
    return decoded


def _normalize_file_list(
    values: Iterable[object],
    limits: CandidateBundleLimits,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise CandidateBundleError("candidate bundle files must be an iterable of paths")
    normalized: list[str] = []
    portable: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise CandidateBundleError("candidate bundle file paths must be strings")
        path = _normalize_relative_path(value, max_path_bytes=limits.max_path_bytes)
        if path == CANDIDATE_BUNDLE_MANIFEST:
            raise CandidateBundleError("candidate bundle manifest cannot declare itself")
        folded = path.casefold()
        if path in normalized or folded in portable:
            raise CandidateBundleError(f"duplicate candidate bundle path: {path!r}")
        normalized.append(path)
        portable.add(folded)
        if len(normalized) > limits.max_files:
            raise CandidateBundleError(
                f"candidate bundle exceeds max_files: {len(normalized)} > {limits.max_files}"
            )
    if not normalized:
        raise CandidateBundleError("candidate bundle files cannot be empty")
    return tuple(normalized)


def _normalize_relative_path(value: str, *, max_path_bytes: int) -> str:
    if not isinstance(value, str) or not value:
        raise CandidateBundleError("candidate bundle path cannot be empty")
    if "\x00" in value or "\\" in value:
        raise CandidateBundleError(f"unsafe candidate bundle path: {value!r}")
    raw_parts = value.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise CandidateBundleError(f"unsafe candidate bundle path: {value!r}")
    relative = PurePosixPath(value)
    if relative.is_absolute() or relative.as_posix() != value:
        raise CandidateBundleError(f"unsafe candidate bundle path: {value!r}")
    encoded = value.encode("utf-8")
    if len(encoded) > max_path_bytes:
        raise CandidateBundleError("candidate bundle path exceeds max_path_bytes")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise CandidateBundleError(f"unsafe candidate bundle path: {value!r}")
    return value


def _enumerate_closed_tree(root: Path, limits: CandidateBundleLimits) -> tuple[str, ...]:
    found: list[str] = []

    def visit(directory: Path, relative_parts: tuple[str, ...]) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise CandidateBundleError("candidate bundle tree cannot be read") from exc
        for entry in entries:
            parts = (*relative_parts, entry.name)
            relative = "/".join(parts)
            _normalize_relative_path(relative, max_path_bytes=limits.max_path_bytes)
            if entry.is_symlink():
                raise CandidateBundleError(f"candidate bundle cannot contain symlink: {relative}")
            if entry.is_dir(follow_symlinks=False):
                visit(Path(entry.path), parts)
            elif entry.is_file(follow_symlinks=False):
                if relative != CANDIDATE_BUNDLE_MANIFEST:
                    found.append(relative)
                    if len(found) > limits.max_files:
                        raise CandidateBundleError("candidate bundle exceeds max_files")
            else:
                raise CandidateBundleError(
                    f"candidate bundle member is not a regular file: {relative}"
                )

    visit(root, ())
    return tuple(found)


def _hash_regular_member(
    root: Path,
    relative: str,
    limits: CandidateBundleLimits,
) -> tuple[int, str]:
    path = _member_path(root, relative)
    digest = hashlib.sha256()
    total = 0
    with _open_regular_member(root, relative) as handle:
        while True:
            block = handle.read(_COPY_BLOCK_BYTES)
            if not block:
                break
            total += len(block)
            if total > limits.max_file_bytes:
                raise CandidateBundleError(
                    f"candidate bundle member exceeds max_file_bytes: {relative}"
                )
            digest.update(block)
        _require_unchanged_open_file(handle, path, relative)
    return total, digest.hexdigest()


def _copy_member(
    root: Path,
    expected: CandidateBundleMember,
    destination: Path,
    limits: CandidateBundleLimits,
) -> None:
    total = 0
    source_path = _member_path(root, expected.path)
    with _open_regular_member(root, expected.path) as source, destination.open("xb") as target:
        while True:
            block = source.read(_COPY_BLOCK_BYTES)
            if not block:
                break
            total += len(block)
            if total > limits.max_file_bytes:
                raise CandidateBundleError(
                    f"candidate bundle member exceeds max_file_bytes: {expected.path}"
                )
            target.write(block)
        target.flush()
        os.fsync(target.fileno())
        _require_unchanged_open_file(source, source_path, expected.path)



def _open_regular_member(root: Path, relative: str):
    path = _member_path(root, relative)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise CandidateBundleError(f"candidate bundle member is not regular: {relative}")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CandidateBundleError(f"candidate bundle member cannot be opened: {relative}") from exc
    handle = os.fdopen(descriptor, "rb")
    opened = os.fstat(handle.fileno())
    if (
        not stat.S_ISREG(opened.st_mode)
        or opened.st_dev != before.st_dev
        or opened.st_ino != before.st_ino
    ):
        handle.close()
        raise CandidateBundleError(f"candidate bundle member changed before read: {relative}")
    return handle


def _member_path(root: Path, relative: str) -> Path:
    parts = PurePosixPath(relative).parts
    current = root
    for part in parts:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError as exc:
            raise CandidateBundleError(f"candidate bundle member is missing: {relative}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise CandidateBundleError(f"candidate bundle cannot contain symlink: {relative}")
    try:
        current.resolve().relative_to(root)
    except ValueError as exc:
        raise CandidateBundleError(f"candidate bundle path escapes its root: {relative}") from exc
    return current


def _require_unchanged_open_file(handle, path: Path, relative: str) -> None:
    opened = os.fstat(handle.fileno())
    try:
        current = path.lstat()
    except FileNotFoundError as exc:
        raise CandidateBundleError(f"candidate bundle member disappeared: {relative}") from exc
    if (
        not stat.S_ISREG(current.st_mode)
        or current.st_dev != opened.st_dev
        or current.st_ino != opened.st_ino
        or current.st_size != opened.st_size
        or current.st_mtime_ns != opened.st_mtime_ns
    ):
        raise CandidateBundleError(f"candidate bundle member changed while read: {relative}")


def _read_regular_file(path: Path, *, max_bytes: int) -> bytes:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise CandidateBundleError(f"file is not regular: {path.name}")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if opened.st_dev != before.st_dev or opened.st_ino != before.st_ino:
            raise CandidateBundleError(f"file changed before read: {path.name}")
        chunks: list[bytes] = []
        total = 0
        while True:
            block = os.read(descriptor, min(_COPY_BLOCK_BYTES, max_bytes + 1 - total))
            if not block:
                break
            total += len(block)
            if total > max_bytes:
                raise CandidateBundleError(f"file exceeds byte limit: {path.name}")
            chunks.append(block)
        current = path.lstat()
        after = os.fstat(descriptor)
        if (
            current.st_dev != after.st_dev
            or current.st_ino != after.st_ino
            or current.st_size != after.st_size
            or current.st_mtime_ns != after.st_mtime_ns
        ):
            raise CandidateBundleError(f"file changed while read: {path.name}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _tree_digest(entrypoint: str, members: Iterable[CandidateBundleMember]) -> str:
    digest = hashlib.sha256()
    digest.update(_DIGEST_DOMAIN)
    digest.update(entrypoint.encode("utf-8") + b"\0")
    ordered = tuple(sorted(members, key=lambda member: member.path))
    digest.update(len(ordered).to_bytes(8, "big"))
    for member in ordered:
        digest.update(member.path.encode("utf-8") + b"\0")
        digest.update(member.byte_length.to_bytes(8, "big"))
        digest.update(bytes.fromhex(member.sha256))
    return digest.hexdigest()


def _existing_bundle(
    destination: Path,
    source: CandidateBundle,
    limits: CandidateBundleLimits,
) -> CandidateBundle:
    inspect_candidate_bundle(destination, limits=limits,
                             recorded_identity=source.tree_sha256)
    return replace(source, root=destination)


def _make_tree_read_only(root: Path) -> None:
    directories = [root]
    for path in root.rglob("*"):
        if path.is_dir():
            directories.append(path)
        else:
            path.chmod(0o444)
    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        directory.chmod(0o555)


def _remove_staging_tree(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_dir():
            path.chmod(0o755)
        else:
            path.chmod(0o644)
    root.chmod(0o755)
    shutil.rmtree(root)


def _regular_directory(path: Path | str, label: str) -> Path:
    raw = Path(path)
    if raw.is_symlink() or not raw.is_dir():
        raise CandidateBundleError(f"{label} must be a regular directory")
    return raw.resolve()


def _required_string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise CandidateBundleError(f"candidate bundle manifest {key} must be a string")
    return value


__all__ = [
    "CANDIDATE_BUNDLE_MANIFEST",
    "CANDIDATE_BUNDLE_SCHEMA",
    "CandidateBundle",
    "CandidateBundleError",
    "CandidateBundleLimits",
    "CandidateBundleMember",
    "freeze_candidate_bundle",
    "inspect_candidate_bundle",
    "write_candidate_bundle_manifest",
]
