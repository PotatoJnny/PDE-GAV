"""Safe JSON value and numerical-array codec for the Agent boundary."""

from __future__ import annotations

import base64
from enum import Enum
from functools import lru_cache
import hashlib
import math
from numbers import Complex, Integral, Real
import os
from pathlib import Path
import re
import stat
from typing import Mapping
from uuid import uuid4


_WIRE_TAG = "$pde_agent_wire_v1"
_MAX_NESTING = 64
_ARRAY_FILE = re.compile(r"^[0-9a-f]{64}\.npy$")


class AgentWireCodecError(ValueError):
    """A value cannot be represented by the safe Agent wire protocol."""


class AgentWireCodec:
    """Tagged JSON with bounded NPY sidecars and strict request-safe defaults.

    ``validate_array_finiteness=False`` is for an explicitly trusted numerical
    result boundary only. It does not relax metadata, dtype or byte limits.
    """

    def __init__(
        self,
        sidecar_root: str | os.PathLike[str] | None = None,
        *,
        inline_array_bytes: int = 64 * 1024,
        max_sidecar_bytes: int = 2 * 1024 * 1024 * 1024,
        validate_encoded_array_finite: bool = True,
        validate_decoded_array_finite: bool = True,
        durable_sidecars: bool = True,
        validate_array_finiteness: bool | None = None,
    ) -> None:
        # Compatibility with the VPFP trusted-result transport. Prefer the
        # directional upstream switches when encode/decode policies differ.
        if validate_array_finiteness is not None:
            if type(validate_array_finiteness) is not bool:
                raise TypeError("validate_array_finiteness must be boolean")
            validate_encoded_array_finite = validate_array_finiteness
            validate_decoded_array_finite = validate_array_finiteness
        # Trusted numerical-service replies may skip array value scans. These
        # switches do not relax wire tags, metadata, byte limits or file access.
        for value, label in (
            (validate_encoded_array_finite, "validate_encoded_array_finite"),
            (validate_decoded_array_finite, "validate_decoded_array_finite"),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{label} must be a boolean")
        self.validate_encoded_array_finite = validate_encoded_array_finite
        self.validate_decoded_array_finite = validate_decoded_array_finite
        for value, label in (
            (inline_array_bytes, "inline_array_bytes"),
            (max_sidecar_bytes, "max_sidecar_bytes"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        if inline_array_bytes > max_sidecar_bytes:
            raise ValueError("inline_array_bytes cannot exceed max_sidecar_bytes")
        if not isinstance(durable_sidecars, bool):
            raise TypeError("durable_sidecars must be a boolean")
        self.durable_sidecars = durable_sidecars
        self.inline_array_bytes = inline_array_bytes
        self.max_sidecar_bytes = max_sidecar_bytes
        # Only trusted numerical RESULT transports opt out. Request codecs and
        # the general Agent boundary keep the default finite-array validation.
        self.validate_array_finiteness = (
            validate_encoded_array_finite and validate_decoded_array_finite
        )
        self.sidecar_root = None if sidecar_root is None else Path(sidecar_root).resolve()
        if self.sidecar_root is not None:
            self.sidecar_root.mkdir(parents=True, exist_ok=True, mode=0o700)
            if self.sidecar_root.is_symlink() or not self.sidecar_root.is_dir():
                raise AgentWireCodecError("Agent array sidecar root must be a directory")
            os.chmod(self.sidecar_root, 0o700)

    def encode(self, value: object) -> object:
        return self._encode(value, depth=0)

    def decode(self, node: object) -> object:
        return self._decode(node, depth=0)

    def _encode(self, value: object, *, depth: int) -> object:
        _check_depth(depth)
        numpy = _optional_numpy()
        if numpy is not None and isinstance(value, numpy.ndarray):
            return self._encode_array(value)
        if numpy is not None and isinstance(value, numpy.generic):
            return self._encode(value.item(), depth=depth + 1)
        if isinstance(value, Enum):
            return self._encode(value.value, depth=depth + 1)
        if value is None or isinstance(value, (str, bool)):
            return value
        if isinstance(value, Integral):
            return int(value)
        if isinstance(value, Real):
            converted = float(value)
            if not math.isfinite(converted):
                raise AgentWireCodecError("Agent numbers must be finite")
            return converted
        if isinstance(value, Complex):
            real = float(value.real)
            imaginary = float(value.imag)
            if not math.isfinite(real) or not math.isfinite(imaginary):
                raise AgentWireCodecError("Agent complex numbers must be finite")
            return {_WIRE_TAG: "complex", "real": real, "imaginary": imaginary}
        if isinstance(value, bytes):
            return {
                _WIRE_TAG: "bytes",
                "data": base64.b64encode(value).decode("ascii"),
            }
        if isinstance(value, tuple):
            return {
                _WIRE_TAG: "tuple",
                "items": [self._encode(item, depth=depth + 1) for item in value],
            }
        if isinstance(value, list):
            return [self._encode(item, depth=depth + 1) for item in value]
        if isinstance(value, Mapping):
            if any(not isinstance(key, str) for key in value):
                raise AgentWireCodecError("Agent mapping keys must be strings")
            encoded = {
                key: self._encode(item, depth=depth + 1)
                for key, item in value.items()
            }
            if _WIRE_TAG in encoded:
                return {_WIRE_TAG: "mapping", "items": encoded}
            return encoded
        raise AgentWireCodecError(
            f"Agent wire protocol does not support {type(value).__name__}"
        )

    def _encode_array(self, value: object) -> Mapping[str, object]:
        numpy = _require_numpy()
        if value.dtype.hasobject or value.dtype.fields is not None:
            raise AgentWireCodecError("Agent arrays cannot contain Python objects")
        contiguous = numpy.ascontiguousarray(value)
        if self.validate_encoded_array_finite and contiguous.dtype.kind in {"f", "c"} and not bool(
            numpy.all(numpy.isfinite(contiguous))
        ):
            raise AgentWireCodecError("Agent arrays must contain finite values")
        if contiguous.nbytes <= self.inline_array_bytes:
            return {
                _WIRE_TAG: "ndarray_inline",
                "dtype": contiguous.dtype.str,
                "shape": list(contiguous.shape),
                "data": base64.b64encode(contiguous.tobytes(order="C")).decode("ascii"),
            }
        if self.sidecar_root is None:
            raise AgentWireCodecError(
                "Agent array exceeds inline limit and no sidecar root is configured"
            )
        return self._publish_array(contiguous)

    def _publish_array(self, array) -> Mapping[str, object]:
        numpy = _require_numpy()
        assert self.sidecar_root is not None
        if array.nbytes > self.max_sidecar_bytes:
            raise AgentWireCodecError("Agent array sidecar exceeds the size limit")
        temporary = self.sidecar_root / f".array-{uuid4().hex}.partial"
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                numpy.save(handle, array, allow_pickle=False)
                handle.flush()
                if self.durable_sidecars:
                    os.fsync(handle.fileno())
            size = temporary.stat().st_size
            if size > self.max_sidecar_bytes:
                raise AgentWireCodecError("Agent array sidecar exceeds the size limit")
            digest = _sha256_regular_file(temporary, self.max_sidecar_bytes)
            destination = self.sidecar_root / f"{digest}.npy"
            try:
                os.link(temporary, destination)
            except FileExistsError:
                pass
            os.chmod(destination, 0o400)
            if self.durable_sidecars:
                _fsync_directory(self.sidecar_root)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        return {
            _WIRE_TAG: "ndarray_npy",
            "file": destination.name,
            "sha256": digest,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "bytes": size,
        }

    def _decode(self, node: object, *, depth: int) -> object:
        _check_depth(depth)
        if node is None or isinstance(node, (str, bool, int)):
            return node
        if isinstance(node, float):
            if not math.isfinite(node):
                raise AgentWireCodecError("Agent numbers must be finite")
            return node
        if isinstance(node, list):
            return [self._decode(item, depth=depth + 1) for item in node]
        if not isinstance(node, Mapping):
            raise AgentWireCodecError("Agent wire value is not valid JSON")
        kind = node.get(_WIRE_TAG)
        if kind is None:
            if any(not isinstance(key, str) for key in node):
                raise AgentWireCodecError("Agent mapping keys must be strings")
            return {
                key: self._decode(item, depth=depth + 1)
                for key, item in node.items()
            }
        if not isinstance(kind, str):
            raise AgentWireCodecError("Agent wire tag must be a string")
        if kind == "mapping":
            _require_keys(node, {_WIRE_TAG, "items"})
            items = node["items"]
            if not isinstance(items, Mapping) or any(
                not isinstance(key, str) for key in items
            ):
                raise AgentWireCodecError("Agent escaped mapping is invalid")
            return {
                key: self._decode(item, depth=depth + 1)
                for key, item in items.items()
            }
        if kind == "tuple":
            _require_keys(node, {_WIRE_TAG, "items"})
            items = node["items"]
            if not isinstance(items, list):
                raise AgentWireCodecError("Agent tuple payload is invalid")
            return tuple(self._decode(item, depth=depth + 1) for item in items)
        if kind == "complex":
            _require_keys(node, {_WIRE_TAG, "real", "imaginary"})
            real = _finite_number(node["real"], "complex real component")
            imaginary = _finite_number(
                node["imaginary"],
                "complex imaginary component",
            )
            return complex(real, imaginary)
        if kind == "bytes":
            _require_keys(node, {_WIRE_TAG, "data"})
            return _decode_base64(node["data"], "bytes")
        if kind == "ndarray_inline":
            _require_keys(node, {_WIRE_TAG, "dtype", "shape", "data"})
            return self._decode_inline_array(node)
        if kind == "ndarray_npy":
            _require_keys(
                {key: value for key, value in node.items() if key != "sha256"},
                {_WIRE_TAG, "file", "dtype", "shape", "bytes"},
            )
            return self._decode_sidecar_array(node)
        raise AgentWireCodecError(f"unknown Agent wire tag: {kind}")

    def _decode_inline_array(self, node: Mapping[str, object]):
        dtype, shape = _array_metadata(node)
        payload = _decode_base64(node["data"], "array")
        if len(payload) > self.inline_array_bytes:
            raise AgentWireCodecError("inline Agent array exceeds the size limit")
        if math.prod(shape) * dtype.itemsize != len(payload):
            raise AgentWireCodecError(
                "Agent array payload size does not match its shape"
            )
        numpy = _require_numpy()
        try:
            array = numpy.frombuffer(payload, dtype=dtype).reshape(shape).copy()
        except (TypeError, ValueError) as exc:
            raise AgentWireCodecError("Agent array payload is invalid") from exc
        if self.validate_decoded_array_finite:
            _validate_array(array)
        return array

    def _decode_sidecar_array(self, node: Mapping[str, object]):
        if self.sidecar_root is None:
            raise AgentWireCodecError("Agent array sidecars are not enabled")
        dtype, shape = _array_metadata(node)
        filename = node["file"]
        declared_bytes = node["bytes"]
        if not isinstance(filename, str) or _ARRAY_FILE.fullmatch(filename) is None:
            raise AgentWireCodecError("Agent array sidecar filename is invalid")
        if (
            isinstance(declared_bytes, bool)
            or not isinstance(declared_bytes, int)
            or declared_bytes < 1
            or declared_bytes > self.max_sidecar_bytes
        ):
            raise AgentWireCodecError("Agent array sidecar byte count is invalid")
        if math.prod(shape) * dtype.itemsize > declared_bytes:
            raise AgentWireCodecError("Agent array shape exceeds its sidecar size")
        path = self.sidecar_root / filename
        descriptor = _open_bounded_regular(path, self.max_sidecar_bytes)
        with os.fdopen(descriptor, "rb") as handle:
            metadata = os.fstat(handle.fileno())
            if metadata.st_size != declared_bytes:
                raise AgentWireCodecError(
                    "Agent array sidecar byte count changed"
                )
            try:
                array = _require_numpy().load(handle, allow_pickle=False)
            except (OSError, TypeError, ValueError) as exc:
                raise AgentWireCodecError("Agent array sidecar is invalid") from exc
        if array.dtype.hasobject or array.dtype.fields is not None:
            raise AgentWireCodecError("Agent arrays cannot contain Python objects")
        if array.dtype.str != dtype.str or tuple(array.shape) != shape:
            raise AgentWireCodecError("Agent array sidecar metadata changed")
        if self.validate_decoded_array_finite:
            _validate_array(array)
        return array


def encode_wire_value(value: object) -> object:
    """Encode a small in-band value; capability transports use ``AgentWireCodec``."""

    return AgentWireCodec(inline_array_bytes=64 * 1024 * 1024).encode(value)


def decode_wire_value(node: object) -> object:
    """Decode a small in-band value; capability transports use ``AgentWireCodec``."""

    return AgentWireCodec(inline_array_bytes=64 * 1024 * 1024).decode(node)


def _array_metadata(node: Mapping[str, object]):
    numpy = _require_numpy()
    dtype_value = node["dtype"]
    shape_value = node["shape"]
    if not isinstance(dtype_value, str) or not dtype_value:
        raise AgentWireCodecError("Agent array dtype is invalid")
    try:
        dtype = numpy.dtype(dtype_value)
    except (TypeError, ValueError) as exc:
        raise AgentWireCodecError("Agent array dtype is invalid") from exc
    if dtype.hasobject or dtype.fields is not None:
        raise AgentWireCodecError("Agent arrays cannot contain Python objects")
    if not isinstance(shape_value, list) or any(
        isinstance(size, bool) or not isinstance(size, int) or size < 0
        for size in shape_value
    ):
        raise AgentWireCodecError("Agent array shape is invalid")
    return dtype, tuple(shape_value)


def _validate_array(array) -> None:
    numpy = _require_numpy()
    if array.dtype.kind in {"f", "c"} and not bool(numpy.all(numpy.isfinite(array))):
        raise AgentWireCodecError("Agent arrays must contain finite values")


def _open_bounded_regular(path: Path, max_bytes: int) -> int:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise AgentWireCodecError("Agent array sidecar cannot be opened") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise AgentWireCodecError("Agent array sidecar is not a regular file")
        if metadata.st_size < 1 or metadata.st_size > max_bytes:
            raise AgentWireCodecError("Agent array sidecar exceeds the size limit")
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _sha256_regular_file(path: Path, max_bytes: int) -> str:
    descriptor = _open_bounded_regular(path, max_bytes)
    with os.fdopen(descriptor, "rb") as handle:
        return _sha256_handle(handle)


def _sha256_handle(handle) -> str:
    digest = hashlib.sha256()
    while True:
        chunk = handle.read(1024 * 1024)
        if not chunk:
            return digest.hexdigest()
        digest.update(chunk)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _decode_base64(value: object, label: str) -> bytes:
    if not isinstance(value, str):
        raise AgentWireCodecError(f"Agent {label} payload must be base64 text")
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        raise AgentWireCodecError(f"Agent {label} payload is invalid base64") from exc


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AgentWireCodecError(f"Agent {label} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise AgentWireCodecError(f"Agent {label} must be finite")
    return converted


def _require_keys(node: Mapping[str, object], expected: set[str]) -> None:
    if set(node) != expected:
        raise AgentWireCodecError("Agent tagged wire value has unexpected fields")


def _check_depth(depth: int) -> None:
    if depth > _MAX_NESTING:
        raise AgentWireCodecError("Agent wire value exceeds maximum nesting")


@lru_cache(maxsize=1)
def _optional_numpy():
    try:
        import numpy
    except ImportError:
        return None
    return numpy


def _require_numpy():
    numpy = _optional_numpy()
    if numpy is None:
        raise AgentWireCodecError("numpy is required for Agent array transport")
    return numpy


__all__ = [
    "AgentWireCodec",
    "AgentWireCodecError",
    "decode_wire_value",
    "encode_wire_value",
]
