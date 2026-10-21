from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path
from typing import Any

ONLINE_GENERATOR_REQUEST_SCHEMA = "vpfp_online_generator_request_v1"
ONLINE_GENERATOR_RESPONSE_SCHEMA = "vpfp_online_generator_response_v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use the bounded controller-owned VPFP numerical generator/oracle."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status")

    sample = subparsers.add_parser("sample")
    sample.add_argument("--count", type=int, default=8)
    sample.add_argument("--mode", choices=("base", "guided", "mixed"), default="mixed")
    sample.add_argument("--answers", choices=("none", "grid"), default="grid")
    sample.add_argument("--condition-pool", choices=("train", "validation"), default="train")

    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--case-id", required=True)
    for name in ("nx", "ny", "nvx", "nvy", "nt"):
        materialize.add_argument(f"--{name}", type=int, required=True)
    materialize.add_argument(
        "--answers", choices=("none", "grid"), default="grid"
    )
    materialize.add_argument("--study-id", default=None)
    materialize.add_argument("--level", type=int, default=None)

    materialize_batch = subparsers.add_parser("materialize-batch")
    materialize_batch.add_argument(
        "--requests",
        required=True,
        help="JSON list (or {'requests': [...]}) of materialization requests",
    )

    query = subparsers.add_parser("query")
    query.add_argument("--case-id", required=True)
    query.add_argument(
        "--field",
        choices=("source", "rho0", "g0", "truth", "macro_truth"),
        required=True,
    )
    query.add_argument("--points", required=True, help="Local .npy or .npz file containing points")
    query.add_argument("--out", default=None, help="Optional output .npz path under online_generator_outputs/queries")
    query_batch = subparsers.add_parser("query-batch")
    query_batch.add_argument(
        "--requests",
        required=True,
        help="JSON list (or {'requests': [...]}) of point-query requests",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    request: dict[str, Any] = {
        "schema": ONLINE_GENERATOR_REQUEST_SCHEMA,
        "operation": str(args.command),
    }
    if args.command == "sample":
        request.update(
            {
                "count": int(args.count),
                "mode": str(args.mode),
                "answers": str(args.answers),
                "condition_pool": str(args.condition_pool),
            }
        )
    elif args.command == "materialize":
        request.update(
            {
                "case_id": str(args.case_id),
                "answers": str(args.answers),
                "resolution": {
                    name: int(getattr(args, name))
                    for name in ("nx", "ny", "nvx", "nvy", "nt")
                },
                "study_id": (
                    None if args.study_id is None else str(args.study_id)
                ),
                "level": None if args.level is None else int(args.level),
            }
        )
    elif args.command == "query":
        points = Path(args.points).resolve()
        if not points.is_file() or points.is_symlink():
            raise SystemExit(f"expected a regular points file: {points}")
        request.update(
            {
                "case_id": str(args.case_id),
                "field": str(args.field),
                "points_file": str(points),
                "out": None if args.out is None else str(Path(args.out).resolve()),
            }
        )
    elif args.command == "materialize-batch":
        request.update(
            {
                "operation": "materialize_batch",
                "requests": _load_batch_requests(
                    args.requests, operation="materialize"
                ),
            }
        )
    elif args.command == "query-batch":
        request.update(
            {
                "operation": "query_batch",
                "requests": _load_batch_requests(args.requests, operation="query"),
            }
        )
    response = send_request(request)
    print(json.dumps(response, indent=2, sort_keys=True))
    if not response.get("ok"):
        raise SystemExit(str(response.get("error") or "online generator request failed"))


def send_request(request: dict[str, Any]) -> dict[str, Any]:
    socket_path = os.environ.get("VPFP_ONLINE_GENERATOR_SOCKET") or str(
        Path(__file__).resolve().with_name(".g.sock")
    )
    endpoint = Path(socket_path)
    if not endpoint.is_socket():
        raise RuntimeError(
            "the authorized online generator endpoint is unavailable; "
            "use this client inside the prepared Agent session"
        )
    encoded = json.dumps(request, separators=(",", ":")).encode("utf-8") + b"\n"
    if len(encoded) > 1024 * 1024:
        raise ValueError("online generator request exceeds 1 MiB")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(1800.0)
        client.connect(socket_path)
        client.sendall(encoded)
        chunks = []
        total = 0
        while True:
            chunk = client.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > 8 * 1024 * 1024:
                raise RuntimeError("online generator response exceeds 8 MiB")
            if b"\n" in chunk:
                break
    response = json.loads(b"".join(chunks).splitlines()[0])
    if response.get("schema") != ONLINE_GENERATOR_RESPONSE_SCHEMA:
        raise RuntimeError("unexpected online generator response schema")
    return response


def _load_batch_requests(path: str | Path, *, operation: str) -> list[dict[str, Any]]:
    source = Path(path).resolve()
    if not source.is_file() or source.is_symlink():
        raise SystemExit(f"expected a regular batch-request JSON file: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    rows = payload.get("requests") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise SystemExit("batch-request JSON must contain a nonempty request list")
    normalized = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, dict):
            raise SystemExit(f"batch request {index} must be an object")
        row = dict(raw)
        if operation == "materialize":
            resolution = dict(row.get("resolution") or {})
            if set(resolution) != {"nx", "ny", "nvx", "nvy", "nt"}:
                raise SystemExit(
                    f"materialization request {index} has malformed resolution"
                )
            row = {
                "case_id": str(row.get("case_id") or ""),
                "answers": str(row.get("answers") or "grid"),
                "resolution": {
                    name: int(resolution[name])
                    for name in ("nx", "ny", "nvx", "nvy", "nt")
                },
                "study_id": row.get("study_id"),
                "level": row.get("level"),
            }
        elif operation == "query":
            points = Path(
                str(row.get("points") or row.get("points_file") or "")
            ).resolve()
            if not points.is_file() or points.is_symlink():
                raise SystemExit(
                    f"query request {index} has no regular points file: {points}"
                )
            output = row.get("out")
            row = {
                "case_id": str(row.get("case_id") or ""),
                "field": str(row.get("field") or ""),
                "points_file": str(points),
                "out": (
                    None if output is None else str(Path(str(output)).resolve())
                ),
            }
        else:
            raise ValueError(f"unsupported batch operation: {operation}")
        normalized.append(row)
    return normalized


if __name__ == "__main__":
    main()
