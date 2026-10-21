"""Debugging evidence from candidate execution, never from hidden scoring."""
from __future__ import annotations

from pathlib import Path
import traceback
from typing import Mapping


def _bounded(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[:limit] + "\n[diagnostic truncated]"


def candidate_exception_diagnostic(error: BaseException, *, bundle_root: Path) -> dict:
    """Only for a candidate worker or known candidate syntax check, not scoring.

    No frame locals, arrays, requests or environment are read.
    """
    root = bundle_root.resolve()

    def filename(value: str) -> str:
        path = Path(value)
        try:
            return str(path.relative_to(root))
        except ValueError:
            return path.name

    def text(value: str, limit: int) -> str:
        return _bounded(value.replace(str(root) + "/", ""), limit)

    trace = traceback.TracebackException.from_exception(error, capture_locals=False)
    frames = []
    pending = [trace]
    seen = set()
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        for frame in node.stack:
            source = frame.line or ""
            frame.filename = filename(frame.filename)
            frames.append({"file": frame.filename, "line": frame.lineno,
                           "function": frame.name, "source": text(source, 2048)})
        if node.exc_type and issubclass(node.exc_type, SyntaxError):
            node.filename = filename(node.filename or "<candidate>")
            frames.append({"file": node.filename, "line": int(node.lineno or 1),
                           "function": "<module>", "source": text(node.text or "", 2048)})
        pending.extend(x for x in (node.__cause__, node.__context__) if x is not None)
        pending.extend(getattr(node, "exceptions", None) or ())
    return {
        "origin": "candidate", "exception_type": type(error).__name__,
        "message": text(str(error), 4096),
        "traceback": text("".join(trace.format()), 32768),
        "frames": frames[-32:],
    }


def normalize_candidate_diagnostic(value: object) -> dict:
    """Copy only bounded debugging fields from a worker failure envelope."""
    if not isinstance(value, Mapping) or value.get("origin") != "candidate":
        return {}
    if any(not isinstance(value.get(key), str) for key in ("exception_type", "message", "traceback")):
        return {}
    result = {"origin": "candidate", "exception_type": _bounded(value["exception_type"], 128),
              "message": _bounded(value["message"], 4096),
              "traceback": _bounded(value["traceback"], 32768), "frames": []}
    frames = value.get("frames", [])
    if isinstance(frames, (list, tuple)):
        for frame in frames[-32:]:
            if not isinstance(frame, Mapping):
                continue
            line = frame.get("line")
            if not isinstance(line, int) or isinstance(line, bool) or line < 1:
                continue
            if any(not isinstance(frame.get(key), str) for key in ("file", "function", "source")):
                continue
            result["frames"].append({"file": _bounded(frame["file"], 1024), "line": line,
                                     "function": _bounded(frame["function"], 256),
                                     "source": _bounded(frame["source"], 2048)})
    if isinstance(value.get("case_id"), str):
        result["case_id"] = _bounded(value["case_id"], 256)
    return result


def candidate_diagnostic(error: BaseException) -> dict:
    return normalize_candidate_diagnostic(getattr(error, "candidate_diagnostic", None))
