"""Small command-line client available inside a canonical Agent workspace."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
from threading import current_thread, main_thread
import stat
import tempfile
from typing import Mapping, Sequence

from pde_agent_verifier_solver.sdk import (
    SampleSource,
    freeze_candidate_bundle,
    inspect_candidate_bundle,
    write_candidate_bundle_manifest,
)

from .agent_capability import (
    ControllerAgentCapabilityClient,
    AgentCapabilityClientError,
    _query_from_payload,
)
from .agent_execution import (
    AGENT_PROTOCOL_MAX_BYTES_ENV,
    AGENT_NOMINATIONS_ENV,
    AgentExecutionError,
    read_agent_primary_nomination,
    select_agent_primary_nomination,
    AGENT_WORKSPACE_ENV,
    load_agent_worker_request,
    submit_agent_nomination,
    write_agent_round_log,
)
from .agent_wire import AgentWireCodec


_DEFAULT_CONTROLLER_RPC_TIMEOUT_SECONDS = 300.0
_CONTROLLER_RPC_TIMEOUT_SETTING = "controller_rpc_timeout_seconds"
_DEFAULT_PROTOCOL_MAX_BYTES = 8 * 1024 * 1024
_EXPORTED_CASES_SCHEMA = "canonical-agent-public-cases-v1"


def main(argv: Sequence[str] | None = None) -> int:
    # SIGTERM should take the same request cleanup path as a user's Ctrl-C.
    if current_thread() is not main_thread():
        return _main_with_feedback(argv)
    previous = signal.getsignal(signal.SIGTERM)
    def interrupted(signum, frame):
        del signum, frame
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        return _main_with_feedback(argv)
    finally:
        signal.signal(signal.SIGTERM, previous)


def _main_with_feedback(argv: Sequence[str] | None = None) -> int:
    try:
        return _main(argv)
    except AgentCapabilityClientError as error:
        _print_json({
            "status": "failed", "request_id": error.request_id,
            "error": {"code": error.code, "message": str(error),
                      "http_status": error.http_status,
                      **({"diagnostic": error.diagnostic} if error.diagnostic else {})},
            "evaluation_progress": error.details,
        })
        return 1


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pde_agent_verifier_solver.engine.agent_tool",
        description="Use only this turn's canonical sampling/evaluation capability.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="Show the public round contract and issued Cases")
    cancel = commands.add_parser("cancel-evaluation", help="Cancel only one of this Agent's evaluations")
    cancel.add_argument("request_id")
    cancel.add_argument("--wait-seconds", type=float, default=30.0)

    sample = commands.add_parser("sample", help="Request fresh Development Cases")
    sample.add_argument("source", choices=tuple(value.value for value in SampleSource))
    sample.add_argument("count", type=int)

    paired = commands.add_parser(
        "sample-paired",
        help="Request fresh clean roots and one perturbation per root",
    )
    paired.add_argument("source", choices=("base", "guided"))
    paired.add_argument("count", type=int)
    paired.add_argument("--max-attempts-per-root", type=int, default=16)

    perturb = commands.add_parser(
        "perturb",
        help="Perturb already-issued clean Case IDs",
    )
    perturb.add_argument("case_ids", nargs="+")

    export_cases = commands.add_parser(
        "export-cases",
        help="Write full public data for specified already-issued Cases",
    )
    export_cases.add_argument("destination")
    export_cases.add_argument("case_ids", nargs="+")

    evaluate = commands.add_parser(
        "evaluate",
        help="Evaluate a workspace-relative Candidate bundle on Development Cases",
    )
    evaluate.add_argument("bundle_path")
    evaluate.add_argument("case_ids", nargs="*")
    evaluate.add_argument("--all", action="store_true", dest="all_cases")
    evaluate.add_argument("--query-json")
    evaluate.add_argument("--return-truth", action="store_true")

    evaluate_many = commands.add_parser(
        "evaluate-many",
        help="Compare Candidate bundles on one exactly shared Development plan",
    )
    evaluate_many.add_argument(
        "--bundle",
        action="append",
        required=True,
        dest="bundle_paths",
        help="Workspace-relative Candidate bundle; repeat for every Candidate",
    )
    evaluation_scope = evaluate_many.add_mutually_exclusive_group(required=True)
    evaluation_scope.add_argument("--all", action="store_true", dest="all_cases")
    evaluation_scope.add_argument(
        "--case-id",
        action="append",
        dest="case_ids",
        help="Issued Case ID; repeat to select multiple Cases",
    )
    evaluate_many.add_argument("--query-json")
    evaluate_many.add_argument("--return-truth", action="store_true")

    submit = commands.add_parser(
        "submit",
        help="Atomically nominate one workspace-relative Candidate bundle",
    )
    submit.add_argument("bundle_path")
    submit.add_argument("--sequence", required=True, type=int)
    submit.add_argument("--name", required=True)
    submit.add_argument("--summary", required=True)
    submit.add_argument("--usage", required=True)
    submit.add_argument("--primary", action="store_true", help="Explicitly select this saved version for round-end evaluation")
    primary = commands.add_parser("select-primary", help="Choose an earlier saved submission without copying or changing it")
    primary.add_argument("sequence", type=int)

    clone_incumbent = commands.add_parser(
        "clone-incumbent",
        help="Create an editable author-bundle copy of the read-only incumbent",
    )
    clone_incumbent.add_argument(
        "destination",
        help="New workspace-relative destination directory",
    )

    log = commands.add_parser("log", help="Atomically checkpoint the round notebook")
    log.add_argument("markdown_file", type=Path)

    args = parser.parse_args(argv)
    request = load_agent_worker_request()
    if args.command == "clone-incumbent":
        try:
            result = _clone_incumbent(request, args.destination)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        _print_json(result)
        return 0
    if args.command == "select-primary":
        try:
            _require_primary_policy(request)
            select_agent_primary_nomination(args.sequence)
        except (OSError, TypeError, ValueError, AgentExecutionError) as exc:
            parser.error(str(exc))
        _print_json({"primary_nomination_sequence": args.sequence})
        return 0
    if args.command == "submit":
        try:
            if args.primary:
                _require_primary_policy(request)
            path, snapshot = _commit_candidate_nomination(
                args.bundle_path,
                sequence=args.sequence,
                name=args.name,
                summary=args.summary,
                usage=args.usage,
            )
            if args.primary:
                select_agent_primary_nomination(args.sequence)
        except (OSError, TypeError, ValueError, AgentExecutionError) as exc:
            parser.error(str(exc))
        _print_json(
            {
                "submitted": str(path),
                "sequence": args.sequence,
                "bundle_snapshot": snapshot,
                **({"primary_nomination_sequence": args.sequence} if args.primary else {}),
            }
        )
        return 0
    if args.command == "log":
        path = args.markdown_file
        if not path.is_file() or path.is_symlink():
            parser.error("markdown_file must be a regular non-symlink file")
        destination = write_agent_round_log(path.read_text(encoding="utf-8"))
        _print_json({"checkpointed": str(destination)})
        return 0
    client = ControllerAgentCapabilityClient.from_request(
        request,
        timeout_seconds=_controller_rpc_timeout_seconds(request),
    )

    if args.command == "status":
        cases, evaluations = client.current_status()
        _print_json({**_status(request, cases), "evaluations": evaluations})
        return 0
    if args.command == "cancel-evaluation":
        result = client.cancel_evaluation(args.request_id, wait_seconds=args.wait_seconds)
        _print_json(result)
        return 0 if result.get("finished") and result.get("cleanup_confirmed") else 1
    if args.command == "sample":
        batch = client.sample_cases(SampleSource(args.source), args.count)
        _print_json(
            {
                "remaining": batch.remaining,
                "cases": [_case_summary(case) for case in batch.cases],
            }
        )
        return 0
    if args.command == "sample-paired":
        batch = client.sample_paired_cases(
            SampleSource(args.source),
            args.count,
            max_attempts_per_root=args.max_attempts_per_root,
        )
        _print_json(
            {
                "remaining": batch.remaining,
                "roots": [_case_summary(case) for case in batch.roots],
                "children": [_case_summary(case) for case in batch.children],
            }
        )
        return 0
    if args.command == "perturb":
        batch = client.perturb_cases(tuple(args.case_ids))
        _print_json(
            {
                "remaining": batch.remaining,
                "cases": [_case_summary(case) for case in batch.cases],
            }
        )
        return 0
    if args.command == "export-cases":
        try:
            result = _export_cases(
                request,
                client.current_cases(),
                tuple(args.case_ids),
                args.destination,
            )
        except (OSError, OverflowError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        _print_json(result)
        return 0
    if args.command == "evaluate":
        if args.all_cases and args.case_ids:
            parser.error("evaluate accepts either Case IDs or --all, not both")
        current = client.current_cases()
        case_ids = (
            tuple(case.case_id for case in current)
            if args.all_cases
            else tuple(args.case_ids)
        )
        if not case_ids:
            parser.error("evaluate needs Case IDs or --all")
        try:
            query = _load_query_json(args.query_json)
        except (OSError, OverflowError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        response = client.evaluate(
            _candidate_payload(args.bundle_path),
            case_ids=case_ids,
            query=query,
            return_truth=args.return_truth,
        )
        _print_json(_evaluation_payload(response, include_answers=args.return_truth))
        return 0
    if args.command == "evaluate-many":
        bundle_paths = tuple(args.bundle_paths)
        if len(bundle_paths) < 2:
            parser.error("evaluate-many needs at least two --bundle values")
        current = client.current_cases() if args.all_cases else ()
        case_ids = (
            tuple(case.case_id for case in current)
            if args.all_cases
            else tuple(args.case_ids or ())
        )
        if not case_ids:
            parser.error("evaluate-many needs --all or at least one --case-id")
        try:
            query = _load_query_json(args.query_json)
        except (OSError, OverflowError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        responses = client.evaluate_many(
            tuple(_candidate_payload(path) for path in bundle_paths),
            case_ids=case_ids,
            query=query,
            return_truth=args.return_truth,
        )
        _print_json(
            {
                "shared_plan": True,
                "evaluations": [
                    {
                        "bundle_path": bundle_path,
                        **_evaluation_payload(
                            response,
                            include_answers=args.return_truth,
                        ),
                    }
                    for bundle_path, response in zip(
                        bundle_paths,
                        responses,
                        strict=True,
                    )
                ],
            }
        )
        return 0
    raise AssertionError(args.command)


def _controller_rpc_timeout_seconds(request: Mapping[str, object]) -> float:
    """Read the optional Controller-authored RPC timeout without mutating it."""

    worker_settings = request.get("worker_settings", {})
    if not isinstance(worker_settings, Mapping):
        raise TypeError("Agent request worker_settings must be an object")
    value = worker_settings.get(
        _CONTROLLER_RPC_TIMEOUT_SETTING,
        _DEFAULT_CONTROLLER_RPC_TIMEOUT_SECONDS,
    )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            "worker_settings.controller_rpc_timeout_seconds must be a number"
        )
    timeout_seconds = float(value)
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError(
            "worker_settings.controller_rpc_timeout_seconds must be positive "
            "and finite"
        )
    return timeout_seconds


def _require_primary_policy(request):
    if request.get("worker_settings", {}).get("nomination_policy", "all") != "single_primary_v1":
        raise ValueError("explicit primary selection requires nomination_policy=single_primary_v1")


def _candidate_payload(bundle_path: str) -> dict[str, object]:
    return {"bundle_path": bundle_path}


def _commit_candidate_nomination(
    bundle_path: str,
    *,
    sequence: int,
    name: str,
    summary: str,
    usage: str,
) -> tuple[Path, str]:
    """Freeze exact Candidate bytes before publishing the journal entry."""

    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
        raise ValueError("nomination sequence must be a positive integer")
    workspace = _agent_workspace()
    source = _safe_workspace_path(workspace, bundle_path, label="Candidate bundle")
    inspected = inspect_candidate_bundle(source)
    snapshots = _safe_workspace_path(
        workspace,
        ".submitted-candidates",
        label="Candidate snapshot root",
    )
    if snapshots.is_symlink():
        raise ValueError("Candidate snapshot root cannot be a symlink")
    snapshots.mkdir(mode=0o700, exist_ok=True)
    if not snapshots.is_dir():
        raise ValueError("Candidate snapshot root must be a directory")
    snapshot_storage = _safe_workspace_path(
        workspace,
        f".submitted-candidates/nomination-{sequence:08d}",
        label="Candidate snapshot",
    )
    if snapshot_storage.exists() or snapshot_storage.is_symlink():
        raise FileExistsError("Candidate nomination sequence is already snapshotted")

    created = False
    try:
        snapshot_storage.mkdir(mode=0o700)
        created = True
        frozen = freeze_candidate_bundle(inspected, snapshot_storage)
        relative = frozen.root.relative_to(workspace).as_posix()
        journal = submit_agent_nomination(
            {
                "bundle_path": relative,
                "name": name,
                "summary": summary,
                "usage": usage,
            },
            sequence=sequence,
        )
    except BaseException:
        if created and snapshot_storage.exists():
            _remove_clone_tree(snapshot_storage)
        raise
    return journal, relative


def _load_query_json(relative_path: str | None):
    if relative_path is None:
        return None
    encoded = _read_workspace_bytes(relative_path, label="query JSON")
    try:
        payload = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("query JSON must be valid UTF-8 JSON") from exc
    return _query_from_payload(payload)


def _export_cases(
    request: Mapping[str, object],
    available_cases,
    case_ids: tuple[str, ...],
    destination_value: str,
) -> Mapping[str, object]:
    if not case_ids:
        raise ValueError("export-cases needs at least one --case-id")
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("export-cases Case IDs must be distinct")
    cases_by_id = {case.case_id: case for case in available_cases}
    unknown = tuple(case_id for case_id in case_ids if case_id not in cases_by_id)
    if unknown:
        raise ValueError("export-cases accepts only already-issued Case IDs")
    inline_array_bytes, max_sidecar_bytes, max_manifest_bytes = (
        _export_wire_limits(request)
    )
    workspace = _agent_workspace()
    destination = _safe_workspace_path(
        workspace,
        destination_value,
        label="export destination",
    )
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("export destination must be a new directory")
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}-partial-",
            dir=destination.parent,
        )
    )
    try:
        codec = AgentWireCodec(
            temporary / "arrays",
            inline_array_bytes=inline_array_bytes,
            max_sidecar_bytes=max_sidecar_bytes,
        )
        payload = {
            "schema": _EXPORTED_CASES_SCHEMA,
            "cases": [
                {
                    "case_id": case_id,
                    # These Case objects have already crossed the Controller's
                    # phase-blind Agent projection. Never add metadata here.
                    "public_data": codec.encode(
                        cases_by_id[case_id].public_data
                    ),
                }
                for case_id in case_ids
            ],
        }
        encoded = (
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        if len(encoded) > max_manifest_bytes:
            raise ValueError(
                "exported Case manifest exceeds the capability size limit; "
                "export fewer Cases at a time"
            )
        manifest = temporary / "cases.json"
        descriptor = os.open(
            manifest,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        manifest.chmod(0o400)
        os.rename(temporary, destination)
    except BaseException:
        if temporary.exists():
            _remove_clone_tree(temporary)
        raise
    arrays = tuple((destination / "arrays").glob("*.npy"))
    return {
        "exported": destination_value,
        "manifest": f"{destination_value}/cases.json",
        "case_ids": list(case_ids),
        "manifest_bytes": len(encoded),
        "array_sidecars": len(arrays),
        "decoder": (
            "from pde_agent_verifier_solver.engine.agent_wire import "
            "AgentWireCodec; "
            f"codec = AgentWireCodec({destination_value + '/arrays'!r}, "
            f"inline_array_bytes={inline_array_bytes}, "
            f"max_sidecar_bytes={max_sidecar_bytes}); "
            'public_data = codec.decode(case["public_data"])'
        ),
    }


def _export_wire_limits(
    request: Mapping[str, object],
) -> tuple[int, int, int]:
    capabilities = request.get("capabilities")
    if not isinstance(capabilities, Mapping):
        raise ValueError("Agent request has no capability wire declaration")
    wire = capabilities.get("wire")
    if not isinstance(wire, Mapping) or wire.get("codec") != "safe-json-npy-v1":
        raise ValueError("Agent request has no supported capability wire declaration")

    values = []
    for key in ("inline_array_bytes", "max_sidecar_bytes", "max_rpc_bytes"):
        value = wire.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"Agent capability wire {key} must be positive")
        values.append(value)
    inline_array_bytes, max_sidecar_bytes, max_rpc_bytes = values
    if inline_array_bytes > max_sidecar_bytes:
        raise ValueError(
            "Agent capability wire inline_array_bytes exceeds max_sidecar_bytes"
        )
    return inline_array_bytes, max_sidecar_bytes, max_rpc_bytes


def _read_workspace_bytes(relative_path: str, *, label: str) -> bytes:
    path = _safe_workspace_path(_agent_workspace(), relative_path, label=label)
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{label} must be a regular file")
        maximum = _protocol_max_bytes()
        if metadata.st_size > maximum:
            raise ValueError(f"{label} exceeds the Agent protocol size limit")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            encoded = handle.read(maximum + 1)
        if len(encoded) > maximum:
            raise ValueError(f"{label} exceeds the Agent protocol size limit")
        return encoded
    except FileNotFoundError as exc:
        raise ValueError(f"{label} does not exist") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _protocol_max_bytes() -> int:
    raw = os.environ.get(
        AGENT_PROTOCOL_MAX_BYTES_ENV,
        str(_DEFAULT_PROTOCOL_MAX_BYTES),
    )
    try:
        maximum = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Agent protocol size limit is invalid") from exc
    if maximum < 1:
        raise ValueError("Agent protocol size limit is invalid")
    return maximum


def _unique_json_object(pairs) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"query JSON contains duplicate field {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str):
    raise ValueError(f"query JSON contains non-finite number {value}")


def _finite_json_value(value: object, *, label: str) -> object:
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numerical products install numpy
        np = None
    if np is not None and isinstance(value, np.ndarray):
        if value.dtype.kind == "c":
            raise ValueError(f"{label} contains complex values")
        return _finite_json_value(value.tolist(), label=label)
    if np is not None and isinstance(value, np.generic):
        return _finite_json_value(value.item(), label=label)
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError(f"{label} contains a non-finite number")
        return converted
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"{label} contains a non-string object key")
        return {
            key: _finite_json_value(item, label=f"{label}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [
            _finite_json_value(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ValueError(f"{label} contains unsupported value {type(value).__name__}")


def _clone_incumbent(
    request: Mapping[str, object],
    destination_value: str,
) -> Mapping[str, object]:
    workspace = _agent_workspace()
    incumbent = request.get("incumbent")
    if (
        not isinstance(incumbent, Mapping)
        or incumbent.get("program_available") is not True
    ):
        raise ValueError("this turn has no clonable incumbent program")
    program = incumbent.get("program")
    if not isinstance(program, Mapping):
        raise ValueError("incumbent program metadata is missing")
    source_value = program.get("bundle_path")
    if not isinstance(source_value, str) or not source_value:
        raise ValueError("incumbent program bundle_path is missing")

    source = _safe_workspace_path(workspace, source_value, label="incumbent bundle")
    destination = _safe_workspace_path(
        workspace,
        destination_value,
        label="clone destination",
    )
    if source.is_symlink() or not source.is_dir():
        raise ValueError("incumbent bundle must be a regular directory")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("clone destination must be a new directory")
    if (
        source == destination
        or source in destination.parents
        or destination in source.parents
    ):
        raise ValueError("clone destination must be disjoint from the incumbent bundle")

    inspected = inspect_candidate_bundle(source, recorded_identity=program.get("tree_sha256"))
    expected_entrypoint = program.get("entrypoint")
    if expected_entrypoint != inspected.entrypoint:
        raise ValueError("incumbent program entrypoint does not match its bundle")

    created = False
    try:
        destination.mkdir(mode=0o700)
        created = True
        shutil.copytree(
            source,
            destination,
            symlinks=False,
            dirs_exist_ok=True,
        )
        _make_tree_user_writable(destination)
        write_candidate_bundle_manifest(
            destination,
            entrypoint=inspected.entrypoint,
            files=(member.path for member in inspected.members),
            overwrite=True,
        )
        cloned = inspect_candidate_bundle(destination, recorded_identity=inspected.tree_sha256)
    except BaseException:
        if created:
            _remove_clone_tree(destination)
        raise
    return {
        "cloned": destination.relative_to(workspace).as_posix(),
        "source": source.relative_to(workspace).as_posix(),
        "entrypoint": cloned.entrypoint,
        "files": [member.path for member in cloned.members],
    }


def _agent_workspace() -> Path:
    value = os.environ.get(AGENT_WORKSPACE_ENV)
    if not value:
        raise ValueError(f"{AGENT_WORKSPACE_ENV} is required")
    raw = Path(value)
    if raw.is_symlink() or not raw.is_dir():
        raise ValueError(f"{AGENT_WORKSPACE_ENV} must be a regular directory")
    return raw.resolve()


def _safe_workspace_path(workspace: Path, raw: str, *, label: str) -> Path:
    if (
        not raw
        or "\x00" in raw
        or "\\" in raw
        or any(ord(character) < 32 or ord(character) == 127 for character in raw)
    ):
        raise ValueError(f"{label} must be a safe relative path")
    raw_parts = raw.split("/")
    pure = PurePosixPath(raw)
    if (
        pure.is_absolute()
        or not pure.parts
        or pure.as_posix() != raw
        or any(part in {"", ".", ".."} for part in raw_parts)
    ):
        raise ValueError(f"{label} must be a safe relative path")
    current = workspace
    for index, part in enumerate(pure.parts):
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} cannot traverse symlinks")
        if index < len(pure.parts) - 1 and not current.is_dir():
            raise ValueError(f"{label} parent must be an existing directory")
    resolved = current.resolve()
    if resolved != workspace and workspace not in resolved.parents:
        raise ValueError(f"{label} escaped the Agent workspace")
    return resolved


def _make_tree_user_writable(root: Path) -> None:
    directories = [root]
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("incumbent bundle cannot contain symlinks")
        if path.is_dir():
            directories.append(path)
        elif path.is_file():
            executable = path.stat().st_mode & 0o100
            path.chmod(0o600 | executable)
        else:
            raise ValueError("incumbent bundle contains a non-file member")
    for directory in sorted(
        directories,
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        directory.chmod(0o700)


def _remove_clone_tree(root: Path) -> None:
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_dir():
            path.chmod(0o700)
        else:
            path.chmod(0o600)
    root.chmod(0o700)
    shutil.rmtree(root)


def _status(request: Mapping[str, object], cases) -> Mapping[str, object]:
    round_info = request.get("round")
    agent = request.get("agent")
    incumbent = request.get("incumbent")
    capabilities = request.get("capabilities")
    worker_settings = request.get("worker_settings")
    capability_summary: dict[str, object] = {}
    if isinstance(capabilities, Mapping):
        for name in ("sampling", "evaluation", "public_problem"):
            item = capabilities.get(name)
            if isinstance(item, Mapping):
                capability_summary[name] = {
                    key: value
                    for key, value in item.items()
                    if key not in {"endpoint", "token"}
                }
    candidate_execution = None
    if isinstance(worker_settings, Mapping):
        configured_execution = worker_settings.get("candidate_execution")
        if isinstance(configured_execution, Mapping):
            device = configured_execution.get("device")
            cuda_available = configured_execution.get("cuda_available")
            if isinstance(device, str) and isinstance(cuda_available, bool):
                candidate_execution = {
                    "device": device,
                    "cuda_available": cuda_available,
                }
                for key in ("development_solver_placement", "solver_batch_size", "gpu_slots_per_device",
                            "oracle_concurrency_per_device"):
                    if key in configured_execution:
                        candidate_execution[key] = configured_execution[key]
                if configured_execution.get("protocol") == "shared_gpu_v1":
                    candidate_execution.update({
                        "protocol": "shared_gpu_v1",
                        "gpu_policy": configured_execution.get("gpu_policy"),
                        "cuda_probe_required": True,
                    })
    agent_gpu_execution = None
    if isinstance(worker_settings, Mapping):
        configured_agent_gpu = worker_settings.get("agent_gpu_execution")
        if (isinstance(configured_agent_gpu, Mapping)
                and configured_agent_gpu.get("protocol") == "shared_gpu_v1"
                and configured_agent_gpu.get("required") is True):
            agent_gpu_execution = {
                "protocol": "shared_gpu_v1",
                "qualification_required": True,
                "sandbox_backend": configured_agent_gpu.get("sandbox_backend"),
                "readiness_receipt": "AGENT_GPU_READINESS.json",
            }
    execution_limits = request.get("candidate_execution")
    if isinstance(execution_limits, Mapping):
        candidate_execution = dict(candidate_execution or {})
        for key in (
            "case_charged_wall_seconds", "budget_scope", "excluded_time",
            "oracle_numerical_execution_is_charged",
        ):
            if key in execution_limits:
                candidate_execution[key] = execution_limits[key]
    return {
        "round": round_info,
        "agent": agent,
        "incumbent": incumbent,
        "capabilities": capability_summary,
        "candidate_execution": candidate_execution,
        **({"agent_gpu_execution": agent_gpu_execution} if agent_gpu_execution is not None else {}),
        "nomination_policy": (worker_settings or {}).get("nomination_policy", "all"),
        "primary_nomination_sequence": (
            read_agent_primary_nomination(Path(os.environ[AGENT_NOMINATIONS_ENV]), request["execution_id"])
            if isinstance(worker_settings, Mapping)
            and worker_settings.get("nomination_policy") == "single_primary_v1"
            and AGENT_NOMINATIONS_ENV in os.environ else None
        ),
        "source_counts": dict(Counter(case.source.value for case in cases)),
        "cases": [_case_summary(case) for case in cases],
    }


def _case_summary(case) -> Mapping[str, object]:
    return {
        "case_id": case.case_id,
        "source": case.source.value,
        "round_index": case.round_index,
        "parent_case_id": case.parent_case_id,
        "tags": list(case.tags),
        "public_data": _value_summary(case.public_data),
    }


def _value_summary(value: object, *, depth: int = 0) -> object:
    if depth >= 5:
        return {"type": type(value).__name__, "truncated": True}
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numerical products install numpy
        np = None
    if np is not None and isinstance(value, np.ndarray):
        summary: dict[str, object] = {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        if value.size and value.dtype.kind in {"b", "i", "u", "f", "c"}:
            finite = np.asarray(value)
            if finite.dtype.kind == "c":
                finite = np.abs(finite)
            summary.update(
                minimum=float(np.nanmin(finite)),
                maximum=float(np.nanmax(finite)),
            )
        return summary
    if isinstance(value, Mapping):
        return {
            str(key): _value_summary(item, depth=depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        if len(value) > 24:
            return {
                "type": type(value).__name__,
                "length": len(value),
                "head": [
                    _value_summary(item, depth=depth + 1) for item in value[:4]
                ],
            }
        return [_value_summary(item, depth=depth + 1) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"type": type(value).__name__}


def _evaluation_payload(response, *, include_answers: bool) -> Mapping[str, object]:
    payload: dict[str, object] = {
        "artifact_id": response.artifact_id,
        "primary_error": _finite_json_value(
            response.primary_error,
            label="evaluation primary_error",
        ),
        "metrics": _finite_json_value(
            response.metrics,
            label="evaluation metrics",
        ),
        "cases": [],
    }
    payload["cases"] = [
        {
            "case_id": case.case_id,
            "primary_error": _finite_json_value(
                case.primary_error,
                label=f"Case {case.case_id} primary_error",
            ),
            "metrics": _finite_json_value(
                case.metrics,
                label=f"Case {case.case_id} metrics",
            ),
            "prediction": _answer_payload(
                case.prediction,
                label=f"Case {case.case_id} prediction",
            ) if include_answers else None,
            "truth": _answer_payload(
                case.truth,
                label=f"Case {case.case_id} truth",
            ) if include_answers else None,
        }
        for case in response.cases
    ]
    return payload


def _answer_payload(answer, *, label: str) -> object:
    if answer is None:
        return None
    shape = []
    for index, value in enumerate(answer.shape):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f"{label}.shape[{index}] must be a nonnegative integer")
        shape.append(int(value))
    return {
        "kind": answer.kind.value,
        "fields": _finite_json_value(answer.fields, label=f"{label}.fields"),
        "axes": _finite_json_value(answer.axes, label=f"{label}.axes"),
        "points": _finite_json_value(answer.points, label=f"{label}.points"),
        "shape": shape,
    }


def _print_json(payload: Mapping[str, object]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    raise SystemExit(main())
