"""Append only new execution audits before committing a scientific checkpoint."""
from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock
from uuid import uuid4


class AuditJournal:
    def __init__(self, path, *, prior_rows=(), encode=lambda row: row):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._encode = encode
        self._current_count = 0
        self._lock = Lock()
        self._rows = []
        self._failure = None
        if self.path.exists():
            if self.path.is_symlink():
                raise ValueError("audit journal cannot be a symlink")
            contents = self.path.read_bytes()
            lines = contents.splitlines(keepends=True)
            valid = 0
            for index, line in enumerate(lines):
                try:
                    row = json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    if index != len(lines) - 1:
                        raise ValueError("audit journal contains a corrupt committed row")
                    # A crash may interrupt the last append. Preserve its bytes
                    # separately; a partial row was never a completed audit.
                    tail = self.path.with_name(self.path.name + f".uncommitted-{uuid4().hex}")
                    with tail.open("xb") as stream:
                        os.chmod(tail, 0o600)
                        stream.write(contents[valid:])
                        stream.flush()
                        os.fsync(stream.fileno())
                    self._sync_parent()
                    break
                if not isinstance(row, dict):
                    raise ValueError("audit journal rows must be objects")
                self._rows.append(row)
                valid += len(line)
            if valid != len(contents) or (contents and not contents.endswith(b"\n")):
                with self.path.open("r+b") as stream:
                    stream.truncate(valid)
                    stream.seek(0, os.SEEK_END)
                    if valid and not contents[:valid].endswith(b"\n"):
                        stream.write(b"\n")
                    stream.flush()
                    os.fsync(stream.fileno())
        else:
            self._rows = [dict(row) for row in prior_rows]
            with self.path.open("xb") as stream:
                os.chmod(self.path, 0o600)
                stream.write(self._encode_rows(self._rows))
                stream.flush()
                os.fsync(stream.fileno())
            self._sync_parent()

    def _sync_parent(self):
        descriptor = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _encode_rows(rows):
        return b"".join((json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode() for row in rows)

    def append(self, audits):
        """Accept this process's cumulative service audits, retaining each once."""
        audits = tuple(audits)
        with self._lock:
            if self._failure is not None:
                raise RuntimeError("audit journal durability failed; refusing an ambiguous retry") from self._failure
            if len(audits) < self._current_count:
                raise ValueError("execution audit history moved backwards")
            new = [self._encode(row) for row in audits[self._current_count:]]
            if any(not isinstance(row, dict) for row in new):
                raise TypeError("encoded execution audits must be objects")
            if new:
                encoded = self._encode_rows(new)
                try:
                    with self.path.open("ab") as stream:
                        stream.write(encoded)
                        stream.flush()
                        os.fsync(stream.fileno())
                except BaseException as error:
                    self._failure = error
                    raise
                self._rows.extend(new)
            self._current_count = len(audits)

    @property
    def rows(self):
        with self._lock:
            return list(self._rows)
