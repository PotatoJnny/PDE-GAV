"""Controller-private JSON/NPY storage for prepared macro cohorts.

The store deliberately accepts only tagged scalar/container values and finite
NumPy arrays.  Arrays are written as digest-named ``.npy`` files with pickle
disabled, then loaded read-only through ``mmap``.  A cohort is readable only
after its manifest has been written, so a partially prepared directory cannot
silently become evaluation evidence.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from math import isfinite
from numbers import Real
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
from typing import Mapping


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_ROW_JSON_BYTES = 64 * 1024 * 1024
_MAX_MANIFEST_JSON_BYTES = 64 * 1024 * 1024


class MacroCohortStoreError(RuntimeError):
    """A sealed cohort artifact is missing, malformed, or changed."""


class SafeNumpyValueCodec:
    """Tagged JSON values with content-addressed, non-pickle NPY arrays."""

    def __init__(self, arrays_root: Path) -> None:
        self.arrays_root = _ensure_private_directory(arrays_root)
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - exercised by callers
            raise MacroCohortStoreError(
                "numpy is required for a disk-backed macro cohort"
            ) from exc
        self.np = np
        self._verified_arrays: dict[Path, tuple[str, tuple[int, ...]]] = {}

    def encode(self, value: object) -> Mapping[str, object]:
        np = self.np
        if isinstance(value, np.ndarray):
            return self._encode_array(value)
        if isinstance(value, np.generic):
            node = dict(self._encode_array(np.asarray(value)))
            node["type"] = "numpy_scalar"
            return node
        if isinstance(value, Enum):
            return self.encode(value.value)
        if value is None or isinstance(value, (str, bool, int)):
            return {"type": "scalar", "value": value}
        if isinstance(value, Real):
            converted = float(value)
            if not isfinite(converted):
                raise ValueError("macro cohort scalar values must be finite")
            return {"type": "scalar", "value": converted}
        if isinstance(value, complex):
            real, imag = float(value.real), float(value.imag)
            if not isfinite(real) or not isfinite(imag):
                raise ValueError("macro cohort complex values must be finite")
            return {"type": "complex", "real": real, "imag": imag}
        if isinstance(value, Mapping):
            if any(
                not isinstance(key, str) or not key.strip() for key in value
            ):
                raise TypeError(
                    "macro cohort mapping keys must be nonempty strings"
                )
            return {
                "type": "mapping",
                "items": {
                    key: self.encode(value[key]) for key in sorted(value)
                },
            }
        if isinstance(value, tuple):
            return {
                "type": "tuple",
                "items": [self.encode(item) for item in value],
            }
        if isinstance(value, list):
            return {
                "type": "list",
                "items": [self.encode(item) for item in value],
            }
        raise TypeError(
            f"unsupported macro cohort value type: {type(value).__name__}"
        )


    def _encode_array(self, value: object) -> Mapping[str, object]:
        np = self.np
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError("macro cohort arrays cannot use object dtype")
        if array.dtype.kind in {"f", "c"} and not _array_is_finite(array, np):
            raise ValueError("macro cohort arrays must contain finite values")

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".array-",
            suffix=".npy",
            dir=self.arrays_root,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as output:
                np.save(output, array, allow_pickle=False)
                output.flush()
                os.fsync(output.fileno())
            digest = _sha256_file(temporary)
            destination = self.arrays_root / f"{digest}.npy"
            if destination.exists():
                if destination.is_symlink() or not destination.is_file():
                    raise MacroCohortStoreError(
                        "digest-named macro cohort array is not a regular file"
                    )
                temporary.unlink()
            else:
                os.chmod(temporary, 0o400)
                os.replace(temporary, destination)
            return {
                "type": "ndarray",
                "path": f"arrays/{digest}.npy",
                "sha256": digest,
                "dtype": str(array.dtype),
                "shape": list(array.shape),
            }
        except BaseException:
            try:
                temporary.unlink()
            except OSError:
                pass
            raise
    def decode(self, node: object) -> object:
        if not isinstance(node, Mapping):
            raise MacroCohortStoreError("macro cohort value node must be an object")
        kind = node.get("type")
        if kind == "scalar":
            if set(node) != {"type", "value"}:
                raise MacroCohortStoreError("invalid macro cohort scalar node")
            return node["value"]
        if kind == "complex":
            if set(node) != {"type", "real", "imag"}:
                raise MacroCohortStoreError("invalid macro cohort complex node")
            try:
                value = complex(float(node["real"]), float(node["imag"]))
            except (TypeError, ValueError) as exc:
                raise MacroCohortStoreError(
                    "invalid macro cohort complex value"
                ) from exc
            if not isfinite(value.real) or not isfinite(value.imag):
                raise MacroCohortStoreError(
                    "macro cohort complex value is not finite"
                )
            return value
        if kind in {"mapping", "tuple", "list"}:
            if set(node) != {"type", "items"}:
                raise MacroCohortStoreError("invalid macro cohort container node")
            items = node["items"]
            if kind == "mapping":
                if not isinstance(items, Mapping) or any(
                    not isinstance(key, str) or not key.strip() for key in items
                ):
                    raise MacroCohortStoreError(
                        "invalid macro cohort mapping items"
                    )
                return {key: self.decode(value) for key, value in items.items()}
            if not isinstance(items, list):
                raise MacroCohortStoreError(
                    "macro cohort sequence items must be a list"
                )
            values = [self.decode(item) for item in items]
            return tuple(values) if kind == "tuple" else values
        if kind in {"ndarray", "numpy_scalar"}:
            expected = {"type", "path", "sha256", "dtype", "shape"}
            if set(node) != expected:
                raise MacroCohortStoreError(
                    f"invalid macro cohort {kind} node"
                )
            relative = node.get("path")
            digest = node.get("sha256")
            shape = node.get("shape")
            dtype = node.get("dtype")
            if (
                not isinstance(relative, str)
                or not isinstance(digest, str)
                or not _SHA256.fullmatch(digest)
                or not isinstance(dtype, str)
                or not isinstance(shape, list)
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                    for value in shape
                )
            ):
                raise MacroCohortStoreError(
                    "invalid macro cohort ndarray metadata"
                )
            expected_relative = f"arrays/{digest}.npy"
            if relative != expected_relative:
                raise MacroCohortStoreError("macro cohort array path is not canonical")
            path = _resolve_within(self.arrays_root.parent, relative)
            if path.parent != self.arrays_root:
                raise MacroCohortStoreError("macro cohort array escaped its store")
            self._verify_array(path, digest)
            try:
                array = self.np.load(path, allow_pickle=False, mmap_mode="r")
            except (OSError, ValueError) as exc:
                raise MacroCohortStoreError(
                    "macro cohort array cannot be loaded safely"
                ) from exc
            if str(array.dtype) != dtype or list(array.shape) != shape:
                raise MacroCohortStoreError("macro cohort array metadata changed")
            if kind == "numpy_scalar":
                if array.shape != ():
                    raise MacroCohortStoreError(
                        "macro cohort NumPy scalar has non-scalar shape"
                    )
                return array[()]
            return array
        raise MacroCohortStoreError(
            f"unknown macro cohort value type: {kind!r}"
        )

    def _verify_array(self, path: Path, digest: str) -> None:
        if path.is_symlink():
            raise MacroCohortStoreError("macro cohort array cannot be a symlink")
        try:
            metadata = path.stat()
        except OSError as exc:
            raise MacroCohortStoreError("macro cohort array is unavailable") from exc
        if not stat.S_ISREG(metadata.st_mode):
            raise MacroCohortStoreError("macro cohort array is not a regular file")
        identity = (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )
        if self._verified_arrays.get(path) == (digest, identity):
            return
        self._verified_arrays[path] = (digest, identity)


class PrivateMacroCohortStore:
    """One private cohort directory, sealed by a final manifest."""

    def __init__(self, parent: Path, *, cohort_id: str) -> None:
        parent = _ensure_private_directory(parent)
        cohorts = _ensure_private_directory(parent / "cohorts")
        token = hashlib.sha256(cohort_id.encode("utf-8")).hexdigest()[:12]
        self.root = Path(
            tempfile.mkdtemp(prefix=f"cohort-{token}-", dir=cohorts)
        ).resolve()
        os.chmod(self.root, 0o700)
        self.rows_root = _ensure_private_directory(self.root / "rows")
        self.codec = SafeNumpyValueCodec(self.root / "arrays")
        self._sealed = False
        self._cohorts_root = cohorts
        self._discarded = False

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def write_row(self, index: int, payload: Mapping[str, object]) -> Mapping[str, object]:
        if self._sealed or self._discarded:
            raise MacroCohortStoreError("cannot append to a closed macro cohort")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("macro cohort row index must be nonnegative")
        encoded = _canonical_json_bytes(payload)
        if len(encoded) > _MAX_ROW_JSON_BYTES:
            raise MacroCohortStoreError("macro cohort row metadata is too large")
        relative = Path("rows") / f"{index:08d}.json"
        destination = self.root / relative
        _write_bytes_exclusive(destination, encoded, mode=0o400)
        return {
            "index": index,
            "path": relative.as_posix(),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "byte_length": len(encoded),
        }

    def seal(self, manifest: Mapping[str, object]) -> None:
        if self._sealed or self._discarded:
            raise MacroCohortStoreError("macro cohort store is already closed")
        encoded = _canonical_json_bytes(manifest)
        if len(encoded) > _MAX_MANIFEST_JSON_BYTES:
            raise MacroCohortStoreError("macro cohort manifest is too large")
        _write_bytes_exclusive(self.manifest_path, encoded, mode=0o400)
        self._sealed = True

    def read_row(self, reference: Mapping[str, object]) -> Mapping[str, object]:
        if not self._sealed or self._discarded:
            raise MacroCohortStoreError("macro cohort store is unavailable")
        expected = {"index", "path", "sha256", "byte_length"}
        if set(reference) != expected:
            raise MacroCohortStoreError("macro cohort row reference is invalid")
        relative = reference.get("path")
        digest = reference.get("sha256")
        byte_length = reference.get("byte_length")
        index = reference.get("index")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or relative != f"rows/{index:08d}.json"
            or not isinstance(digest, str)
            or not _SHA256.fullmatch(digest)
            or isinstance(byte_length, bool)
            or not isinstance(byte_length, int)
            or byte_length < 1
            or byte_length > _MAX_ROW_JSON_BYTES
        ):
            raise MacroCohortStoreError("macro cohort row reference changed")
        path = _resolve_within(self.root, str(relative))
        if path.parent != self.rows_root:
            raise MacroCohortStoreError("macro cohort row escaped its store")
        payload = _read_bounded_regular_file(
            path,
            max_bytes=byte_length,
            label="macro cohort row",
        )
        if len(payload) != byte_length:
            raise MacroCohortStoreError("macro cohort row byte length differs")
        try:
            decoded = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MacroCohortStoreError("macro cohort row is not valid JSON") from exc
        if not isinstance(decoded, Mapping):
            raise MacroCohortStoreError("macro cohort row must be a JSON object")
        return decoded

    def discard(self) -> None:
        if self._discarded:
            return
        root = self.root
        if (
            root == Path(root.anchor)
            or root.parent.resolve() != self._cohorts_root
            or not root.name.startswith("cohort-")
        ):
            raise MacroCohortStoreError("refusing to discard an unsafe cohort path")
        if root.exists():
            shutil.rmtree(root)
        self._discarded = True


def _ensure_private_directory(path: Path) -> Path:
    raw = Path(path).expanduser()
    if raw.exists() and raw.is_symlink():
        raise MacroCohortStoreError("macro cohort directory cannot be a symlink")
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    resolved = raw.resolve()
    if resolved == Path(resolved.anchor) or not resolved.is_dir():
        raise MacroCohortStoreError("macro cohort needs a specific directory")
    os.chmod(resolved, 0o700)
    return resolved


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MacroCohortStoreError(
            "macro cohort metadata is not canonical JSON"
        ) from exc


def _write_bytes_exclusive(path: Path, payload: bytes, *, mode: int) -> None:
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise MacroCohortStoreError("macro cohort output parent is unsafe")
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            mode,
        )
    except OSError as exc:
        raise MacroCohortStoreError(
            "macro cohort output already exists or cannot be created"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(path, mode)
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _read_bounded_regular_file(path: Path, *, max_bytes: int, label: str) -> bytes:
    if path.is_symlink():
        raise MacroCohortStoreError(f"{label} cannot be a symlink")
    try:
        metadata = path.stat()
    except OSError as exc:
        raise MacroCohortStoreError(f"{label} is unavailable") from exc
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > max_bytes:
        raise MacroCohortStoreError(f"{label} is not a bounded regular file")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise MacroCohortStoreError(f"{label} cannot be read") from exc



def _array_is_finite(array: object, np: object) -> bool:
    iterator = np.nditer(
        array,
        flags=["external_loop", "buffered", "zerosize_ok"],
        op_flags=["readonly"],
        order="C",
        buffersize=65_536,
    )
    return all(bool(np.all(np.isfinite(chunk))) for chunk in iterator)

def _sha256_file(path: Path) -> str:
    if path.is_symlink():
        raise MacroCohortStoreError("macro cohort array cannot be a symlink")
    try:
        metadata = path.stat()
    except OSError as exc:
        raise MacroCohortStoreError("macro cohort array is unavailable") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise MacroCohortStoreError("macro cohort array is not a regular file")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise MacroCohortStoreError("macro cohort array cannot be read") from exc
    return digest.hexdigest()


def _resolve_within(root: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise MacroCohortStoreError("macro cohort path must be relative")
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise MacroCohortStoreError("macro cohort path escaped its root")
    return candidate
