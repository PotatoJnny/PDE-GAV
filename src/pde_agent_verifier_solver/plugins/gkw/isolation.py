"""Immutable snapshots separating GKW candidates from controller scoring state."""

from __future__ import annotations

from enum import Enum
import hashlib
from math import isfinite
from numbers import Integral, Real
import struct
from types import MappingProxyType
from typing import Any, Mapping

from pde_agent_verifier_solver.sdk import AnswerQuery, Case


_HASH_CHUNK_BYTES = 1 << 20


def snapshot_gkw_case(case: Case) -> Case:
    """Copy a Case onto immutable buffers before crossing a trust boundary."""

    return Case(
        case_id=case.case_id,
        source=case.source,
        visibility=case.visibility,
        public_data=_freeze_public(case.public_data),
        round_index=case.round_index,
        parent_case_id=case.parent_case_id,
        tags=tuple(case.tags),
    )


def snapshot_gkw_query(query: AnswerQuery) -> AnswerQuery:
    """Copy mutable resolution/point mappings into immutable mappings."""

    return AnswerQuery(
        kind=query.kind,
        fields=tuple(query.fields),
        resolution=MappingProxyType(dict(query.resolution)),
        points=tuple(MappingProxyType(dict(point)) for point in query.points),
    )


def gkw_public_case_sha256(case: Case) -> str:
    """Return a canonical, bounded-memory identity for one public GKW Case."""

    if not isinstance(case, Case):
        raise TypeError("GKW public identity requires a Case")
    digest = hashlib.sha256()
    digest.update(b"gkw-public-case-identity-v1\0")
    _update_public_identity(
        digest,
        (
            case.case_id,
            case.source,
            case.visibility,
            case.public_data,
            case.round_index,
            case.parent_case_id,
            tuple(case.tags),
        ),
    )
    return digest.hexdigest()


def _update_public_identity(digest: Any, value: Any) -> None:
    import numpy as np

    if value is None:
        digest.update(b"N")
        return
    if isinstance(value, Enum):
        digest.update(b"E")
        _update_public_identity(digest, value.value)
        return
    if isinstance(value, np.ndarray):
        _update_array_identity(digest, value)
        return
    if isinstance(value, np.generic):
        _update_array_identity(digest, np.asarray(value))
        return
    if isinstance(value, bool):
        digest.update(b"B1" if value else b"B0")
        return
    if isinstance(value, Integral):
        _update_sized_bytes(digest, b"I", str(int(value)).encode("ascii"))
        return
    if isinstance(value, Real):
        scalar = float(value)
        if not isfinite(scalar):
            raise ValueError("GKW public identity rejects non-finite scalars")
        if scalar == 0.0:
            scalar = 0.0
        _update_sized_bytes(digest, b"F", struct.pack(">d", scalar))
        return
    if isinstance(value, str):
        _update_sized_bytes(digest, b"S", value.encode("utf-8"))
        return
    if isinstance(value, bytes):
        _update_sized_bytes(digest, b"Y", value)
        return
    if isinstance(value, Mapping):
        keys = tuple(value)
        if any(not isinstance(key, str) for key in keys):
            raise TypeError("GKW public identity requires string mapping keys")
        digest.update(b"M")
        digest.update(struct.pack(">Q", len(keys)))
        for key in sorted(keys):
            _update_public_identity(digest, key)
            _update_public_identity(digest, value[key])
        return
    if isinstance(value, (tuple, list)):
        digest.update(b"L")
        digest.update(struct.pack(">Q", len(value)))
        for item in value:
            _update_public_identity(digest, item)
        return
    raise TypeError(
        f"unsupported GKW public identity value: {type(value).__name__}"
    )


def _update_array_identity(digest: Any, value: Any) -> None:
    import numpy as np

    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError("GKW public identity rejects object arrays")
    digest.update(b"A")
    _update_sized_bytes(digest, b"D", array.dtype.str.encode("ascii"))
    digest.update(struct.pack(">Q", array.ndim))
    for dimension in array.shape:
        digest.update(struct.pack(">Q", int(dimension)))
    digest.update(struct.pack(">Q", int(array.nbytes)))
    if array.flags.c_contiguous:
        raw = memoryview(array).cast("B")
        for start in range(0, len(raw), _HASH_CHUNK_BYTES):
            digest.update(raw[start : start + _HASH_CHUNK_BYTES])
        return
    iterator = np.nditer(
        array,
        flags=("external_loop", "buffered", "zerosize_ok"),
        op_flags=("readonly",),
        order="C",
        buffersize=max(1, _HASH_CHUNK_BYTES // max(1, array.dtype.itemsize)),
    )
    for chunk in iterator:
        digest.update(memoryview(np.asarray(chunk)).cast("B"))


def _update_sized_bytes(digest: Any, tag: bytes, value: bytes) -> None:
    digest.update(tag)
    digest.update(struct.pack(">Q", len(value)))
    digest.update(value)


def _freeze_public(value: Any) -> Any:
    import numpy as np

    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return np.frombuffer(
            array.tobytes(order="C"), dtype=array.dtype
        ).reshape(array.shape)
    if isinstance(value, Mapping):
        return MappingProxyType(
            {name: _freeze_public(item) for name, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_public(item) for item in value)
    return value
