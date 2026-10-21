"""Instance-scoped recovery of exact weak VPFP checkpoint serialization aliases.

The frozen numerical backend stays unchanged. Old checkpoint saves could emit
different Torch archive bytes for the same complete native Case. A differing
descriptor is accepted only after verifying its file and comparing every typed
field and every tensor bit with the already restored record. Restored objects
then seed the existing export cache, preventing further reserialization.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import struct
from types import MethodType

import numpy as np  # noqa: F401 -- preserve the frozen NumPy-before-Torch QR path
import torch

from pde_agent_verifier_solver.plugins.vpfp import weak_iteration_backend as native

SCHEMA = "vpfp_weak_checkpoint_exact_alias_compat_v1"


def assert_exact_record(left, right, *, path="record"):
    """Reject any typed content difference, without numerical tolerances.

    Dictionary insertion order and Python/Torch storage sharing are archive
    details, not record content. Tensor dtype, dimensions, strides, gradient
    flag and all value bits (including NaN payloads and signed zero) are checked.
    Unsupported value types fail closed instead of invoking custom equality.
    """
    def fail(reason):
        raise ValueError(f"weak checkpoint record differs at {path}: {reason}")

    if type(left) is not type(right):
        fail("type")
    if type(left) is torch.Tensor:
        if (left.layout != torch.strided or right.layout != torch.strided
                or left.is_quantized or right.is_quantized):
            fail("unsupported tensor layout")
        if (left.dtype != right.dtype or left.shape != right.shape
                or left.stride() != right.stride()
                or left.requires_grad != right.requires_grad
                or left.is_conj() != right.is_conj()
                or left.is_neg() != right.is_neg()):
            fail("tensor dtype/shape/stride/flags")
        def bits(value):
            return (value.detach().resolve_conj().resolve_neg().cpu().contiguous()
                    .reshape(-1).view(torch.uint8))
        if not torch.equal(bits(left), bits(right)):
            fail("tensor bits")
    elif type(left) is dict:
        if len(left) != len(right):
            fail("mapping length")
        remaining = list(right)
        for key, value in left.items():
            for index, candidate in enumerate(remaining):
                try:
                    assert_exact_record(key, candidate, path=path + ".<key>")
                except ValueError:
                    continue
                remaining.pop(index)
                assert_exact_record(value, right[candidate], path=f"{path}[{key!r}]")
                break
            else:
                fail("mapping key")
    elif type(left) in (tuple, list):
        if len(left) != len(right):
            fail("sequence length")
        for index, (one, two) in enumerate(zip(left, right, strict=True)):
            assert_exact_record(one, two, path=f"{path}[{index}]")
    elif type(left) is float:
        if struct.pack("!d", left) != struct.pack("!d", right):
            fail("float bits")
    elif type(left) is complex:
        if struct.pack("!dd", left.real, left.imag) != struct.pack("!dd", right.real, right.imag):
            fail("complex bits")
    elif left is None or type(left) in (bool, int, str, bytes):
        if left != right:
            fail("value")
    else:
        fail("unsupported type " + type(left).__name__)


def _object_fingerprint(directory, descriptor):
    # Match the native descriptor constraints before constructing a path.
    if (type(descriptor) is not dict or set(descriptor) != {"name", "sha256"}
            or not isinstance(descriptor["sha256"], str) or len(descriptor["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in descriptor["sha256"])
            or descriptor["name"] != descriptor["sha256"] + ".weak.pt"):
        raise ValueError("invalid weak continuation descriptor")
    directory = Path(directory)
    path = directory / descriptor["name"]
    if directory.is_symlink() or path.is_symlink():
        raise ValueError("weak continuation object symlink rejected")
    stat = path.stat()
    if not path.is_file():
        raise ValueError("weak continuation object must be a regular file")
    key = (str(directory.resolve()), descriptor["name"])
    return key, (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def install_restore_compat(generator):
    """Install only on this native generator; repeated installation is a no-op."""
    if not isinstance(generator, native.WeakIterationGenerator):
        raise TypeError("WeakIterationGenerator required")
    if hasattr(generator, "_weak_checkpoint_compat_state"):
        return generator
    original = generator.restore_case_record
    audit = dict(schema=SCHEMA, restore_calls=0, alias_verified=0,
                 cache_backfilled=0, verified_objects=0)
    verified = {}
    generator._weak_checkpoint_compat_state = audit

    def restore(self, payload, directory):
        if (payload.get("schema") != native.SCHEMA + "_case"
                or payload.get("binding_sha256") != self._identity["binding_sha256"]):
            raise ValueError("weak Case continuation binding differs")
        key, descriptor = payload["case_id"], payload["object"]
        if not isinstance(key, str) or not key:
            raise ValueError("invalid weak Case identity")
        with self.backend._lock:
            object_key, before = _object_fingerprint(directory, descriptor)
            prior = self.backend._record_refs.get(key)
            alias = prior is not None and prior != descriptor
            # Equal descriptors can be reused after successful file validation.
            # Replacements, symlinks, size/mtime/ctime changes invalidate it.
            if prior is not None and (alias or verified.get(object_key) != before):
                row = native._get(directory, descriptor)
                if row.get("case", {}).get("case_id") != key:
                    raise ValueError("restored weak Case identity/record changed")
                assert_exact_record(self.backend._restored_records[key], row)
                if _object_fingerprint(directory, descriptor) != (object_key, before):
                    raise ValueError("weak continuation object changed during load")
            # First-time restores are verified by the original _get. For an
            # alias, changing this pointer is safe only after exact comparison.
            if alias:
                self.backend._record_refs[key] = deepcopy(descriptor)
            try:
                record = original(payload, directory)
            except BaseException:
                if alias:
                    self.backend._record_refs[key] = prior
                raise
            if _object_fingerprint(directory, descriptor) != (object_key, before):
                raise ValueError("weak continuation object changed during restore")
            if object_key not in verified:
                audit["verified_objects"] += 1
            verified[object_key] = before
            export_key = (object_key[0], "case", key)
            cached = self.backend._export_cache.get(export_key)
            if cached is None or cached[1] != descriptor:
                self.backend._export_cache[export_key] = (
                    native._snapshot_vpfp_public_case(record.case), deepcopy(descriptor))
                audit["cache_backfilled"] += 1
            audit["restore_calls"] += 1
            audit["alias_verified"] += int(alias)
            return record

    def report(self):
        with self.backend._lock:
            return deepcopy(self._weak_checkpoint_compat_state)

    generator.restore_case_record = MethodType(restore, generator)
    generator.checkpoint_compatibility_audit = MethodType(report, generator)
    return generator


def checkpoint_compatibility_audit(generator):
    """Return detached evidence for runner status files, never hidden records."""
    if not hasattr(generator, "_weak_checkpoint_compat_state"):
        raise ValueError("weak checkpoint compatibility is not installed")
    return generator.checkpoint_compatibility_audit()
