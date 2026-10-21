"""Atomic, private continuation storage for an existing scientific Run.

Only Controller-created checkpoints are accepted. Immutable evaluation objects
and plugin assets are content addressed; an evaluation batch does not rewrite
all preceding Case fields or completed truth arrays.
"""
from __future__ import annotations

from contextlib import contextmanager
import copyreg
import fcntl
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path
import pickle
import random
import signal
import sys
from threading import Event, Thread
import time
import weakref
from types import MappingProxyType
from uuid import uuid4

from pde_agent_verifier_solver.sdk import (
    AnswerView, CaseEvaluation, CaseRecord, EvaluationPlan, GeneratorState,
)
from .python_candidate import PythonCandidateProgram


def _mapping_proxy(values):
    return MappingProxyType(values)


def _candidate_program(locator, digest):
    return PythonCandidateProgram(locator, expected_tree_sha256=digest)


def atomic_bytes(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            os.chmod(temporary, 0o600)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, data):
    atomic_bytes(path, (json.dumps(data, sort_keys=True, allow_nan=False) + "\n").encode())


@contextmanager
def run_writer_lock(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (root / ".checkpoint-writer.lock").open("a+b") as stream:
        os.chmod(stream.name, 0o600)
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another writer is already running this Run") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def capture_rng():
    state = {"python": random.getstate()}
    if "numpy" in sys.modules:
        import numpy as np
        state["numpy"] = np.random.get_state()
    if "torch" in sys.modules:
        import torch
        state["torch"] = torch.get_rng_state().numpy().copy()
        if torch.cuda.is_initialized():
            state["cuda"] = [value.cpu().numpy().copy() for value in torch.cuda.get_rng_state_all()]
    return state


def restore_rng(state):
    random.setstate(state["python"])
    if "numpy" in state:
        import numpy as np
        np.random.set_state(state["numpy"])
    if "torch" in state:
        import torch
        torch.set_rng_state(torch.from_numpy(state["torch"]))
        if "cuda" in state:
            if len(state["cuda"]) != torch.cuda.device_count():
                raise ValueError("resume requires the same CUDA device count")
            torch.cuda.set_rng_state_all([torch.from_numpy(value) for value in state["cuda"]])


_OBJECT_TYPES = (CaseRecord, GeneratorState, EvaluationPlan, AnswerView, CaseEvaluation)


class _Pickler(pickle.Pickler):
    def __init__(self, file, store, skip=None):
        super().__init__(file, protocol=5)
        self.store, self.skip = store, skip
        self.dispatch_table = copyreg.dispatch_table.copy()
        self.dispatch_table[type(MappingProxyType({}))] = lambda value: (_mapping_proxy, (dict(value),))
        self.dispatch_table[PythonCandidateProgram] = lambda value: (
            _candidate_program, (str(value.bundle_locator), value.tree_sha256),
        )

    def persistent_id(self, value):
        if value is self.skip or not isinstance(value, _OBJECT_TYPES):
            return None
        return self.store._save_object(value)


class _Unpickler(pickle.Unpickler):
    def __init__(self, file, store):
        super().__init__(file)
        self.store = store

    def persistent_load(self, token):
        return self.store._load_object(token)


def scientific_progress(value):
    """Small progress identity: excludes random state, logs and elapsed times."""
    state = getattr(value, "state", value)
    if not isinstance(state, dict):
        return None
    def completed(cache):
        rows = []
        def visit(node, path=()):
            if not isinstance(node, dict):
                return
            for key, item in node.items():
                label = path + (str(key),)
                if key == "cases" and isinstance(item, dict):
                    for artifact, cases in item.items():
                        rows.append((label, str(artifact), sorted(map(str, cases))))
                elif key == "failed_artifacts":
                    rows.append((label, sorted(map(str, item))))
                elif isinstance(item, dict):
                    visit(item, label)
        visit(cache)
        return sorted(rows, key=repr)
    competition = state.get("competition", {})
    summary = {
        "phase": state.get("phase"),
        "rounds": len(state.get("round_results", ())),
        "round_fields": sorted(state.get("round_data", {})),
        "terminal_cohorts": sorted(state.get("terminal_hidden", {})),
        "hidden": completed(state.get("hidden_progress", {})),
        "baseline_errors": sorted(state.get("errors", {})),
        "baseline_turns": len(state.get("turns", ())),
        "baseline_phase": competition.get("phase"),
        "baseline_next_round": competition.get("next_round"),
        "baseline_rounds": len(competition.get("rounds", ())),
        "baseline_active": competition.get("active") is not None,
    }
    return sha256(json.dumps(summary, sort_keys=True).encode()).hexdigest()


class CheckpointStore:
    def __init__(self, root, *, identity, generator=None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        self.objects = self.root / "objects"
        self.objects.mkdir(exist_ok=True, mode=0o700)
        self.path = self.root / "checkpoint.json"
        self.identity = identity
        self.generator = generator
        self._written = {}
        self._loaded = {}
        self.sequence = 0
        self.last_payload = None

    def _encode(self, value, skip=None):
        stream = io.BytesIO()
        _Pickler(stream, self, skip).dump(value)
        return stream.getvalue()

    def _decode(self, data):
        return _Unpickler(io.BytesIO(data), self).load()

    def _put(self, data, suffix):
        digest = sha256(data).hexdigest()
        path = self.objects / (digest + suffix)
        if not path.exists():
            atomic_bytes(path, data)
        return {"name": path.name, "sha256": digest}

    def _get(self, reference):
        name = reference["name"]
        if Path(name).name != name:
            raise ValueError("invalid checkpoint object path")
        path = self.objects / name
        if path.is_symlink():
            raise ValueError("checkpoint objects cannot be symlinks")
        data = path.read_bytes()
        return data

    def _save_object(self, value):
        cached = self._written.get(id(value))
        if cached is not None and cached[0]() is value:
            return cached[1]
        kind, data = "pickle", None
        hook = None
        if isinstance(value, CaseRecord):
            hook = getattr(self.generator, "export_case_record", None)
            kind = "case"
        elif isinstance(value, GeneratorState):
            hook = getattr(self.generator, "export_generator_state", None)
            kind = "generator"
        if callable(hook):
            try:
                data = json.dumps(hook(value, self.objects), sort_keys=True, allow_nan=False).encode()
            except NotImplementedError:
                pass
        if data is None:
            kind, data = "pickle", self._encode(value, skip=value)
        token = (kind, self._put(data, ".json" if kind != "pickle" else ".pkl"))
        self._remember(value, token)
        return token

    def _load_object(self, token):
        kind, reference = token
        key = (kind, reference["sha256"])
        existing = self._loaded.get(key)
        if existing is not None and existing() is not None:
            return existing()
        data = self._get(reference)
        if kind == "pickle":
            value = self._decode(data)
        elif kind == "case":
            value = self.generator.restore_case_record(json.loads(data), self.objects)
        elif kind == "generator":
            value = self.generator.restore_generator_state(json.loads(data), self.objects)
        else:
            raise ValueError("unknown checkpoint object kind")
        try:
            self._loaded[key] = weakref.ref(value, lambda ref: self._loaded.pop(key, None))
        except TypeError:
            pass
        self._remember(value, token)
        return value

    def _remember(self, value, token):
        identity = id(value)
        def discard(reference):
            cached = self._written.get(identity)
            if cached is not None and cached[0] is reference:
                self._written.pop(identity, None)
        try:
            self._written[identity] = (weakref.ref(value, discard), token)
        except TypeError:
            pass

    def save(self, value):
        rng = capture_rng()
        try:
            export = getattr(self.generator, "export_runtime_state", None)
            runtime = export(self.objects) if callable(export) else None
            payload = self._put(self._encode({"value": value, "rng": rng}), ".pkl")
            self.sequence += 1
            manifest = {"schema": "campaign_checkpoint_v1", "identity": self.identity,
                        "sequence": self.sequence, "payload": payload,
                        "generator_runtime": runtime, "progress": scientific_progress(value)}
            atomic_json(self.path, manifest)
            self.last_payload = payload["sha256"]
        finally:
            # Serialization must not consume the run's random streams.
            restore_rng(rng)

    def load(self):
        if self.path.is_symlink():
            raise ValueError("checkpoint manifest cannot be a symlink")
        manifest = json.loads(self.path.read_text())
        if manifest.get("schema") != "campaign_checkpoint_v1" or manifest.get("identity") != self.identity:
            raise ValueError("checkpoint scientific input/source identity differs")
        self.sequence = int(manifest["sequence"])
        runtime = manifest.get("generator_runtime")
        if runtime is not None:
            self.generator.restore_runtime_state(runtime, self.objects)
        payload = self._decode(self._get(manifest["payload"]))
        self.last_payload = manifest["payload"]["sha256"]
        restore_rng(payload["rng"])
        return payload["value"]


class RunSegmentBudget:
    """Allocation deadline, independent of each Agent's uninterrupted hard time."""
    def __init__(self, seconds, *, flush_seconds=600, monotonic=time.monotonic):
        if (not math.isfinite(seconds) or not math.isfinite(flush_seconds)
                or seconds <= flush_seconds or flush_seconds < 0):
            raise ValueError("segment must leave positive execution time before its flush margin")
        self._clock = monotonic
        self.deadline = monotonic() + seconds - flush_seconds
        self.cancel = Event()
        self._stop = Event()
        self._thread = Thread(target=self._watch, daemon=True, name="allocation-deadline")
        self._thread.start()

    def _watch(self):
        while not self._stop.wait(min(1.0, max(0.01, self.deadline - self._clock()))):
            if self._clock() >= self.deadline:
                self.cancel.set()
                return

    def should_pause(self, stage, required_seconds=0):
        del stage
        return self.cancel.is_set() or self._clock() + required_seconds >= self.deadline

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2)

    @contextmanager
    def signals(self):
        previous = signal.getsignal(signal.SIGUSR1)
        signal.signal(signal.SIGUSR1, lambda *_: self.cancel.set())
        try:
            yield
        finally:
            signal.signal(signal.SIGUSR1, previous)


def source_identity(repository):
    """Hash current executable source once per segment, including a frozen snapshot."""
    root = Path(repository)
    digest = sha256()
    for directory in (root / "src", root / "scripts", root / "legacy"):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix in {".py", ".sh", ".toml"}:
                digest.update(str(path.relative_to(root)).encode() + b"\0")
                digest.update(path.read_bytes())
    return digest.hexdigest()
