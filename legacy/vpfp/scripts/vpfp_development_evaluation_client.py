#!/usr/bin/env python3
"""Submit or resume a controller-owned VPFP development evaluation."""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from pathlib import Path

from vpfp_solver_bundle_client import bind_solver_bundle


REQUEST_SCHEMA = "vpfp_development_evaluation_request_v2"
PENDING_SCHEMA = "vpfp_development_evaluation_pending_v2"
CASCADE = "fresh_paired_screen_confirm_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission")
    parser.add_argument("--asset", action="append", default=[])
    parser.add_argument("--evidence-cases", type=int)
    parser.add_argument("--resume-request")
    parser.add_argument("--submit-only", action="store_true")
    parser.add_argument("--wait-seconds", type=int, default=1800)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = Path.cwd().resolve()
    exchange = Path(
        os.environ.get("VPFP_EVALUATION_EXCHANGE", "evaluation_exchange")
    ).resolve()
    if not exchange.is_relative_to(workspace):
        raise SystemExit("evaluation exchange must remain in the workspace")
    manifest = json.loads(
        (workspace / "workspace_manifest.json").read_text(encoding="utf-8")
    )
    interface = dict(manifest["development_evaluation_interface"])
    if interface.get("cascade") != CASCADE:
        raise SystemExit("workspace evaluator cascade is incompatible")
    requests = exchange / "requests"
    responses = exchange / "responses"

    if args.resume_request:
        if (
            args.submission is not None
            or args.asset
            or args.submit_only
            or args.evidence_cases is not None
        ):
            raise SystemExit("resume cannot be combined with a submission or assets")
        request_id = args.resume_request.lower()
        if len(request_id) != 32 or any(
            value not in "0123456789abcdef" for value in request_id
        ):
            raise SystemExit("invalid resumed request ID")
    else:
        evidence_cases = (
            int(args.evidence_cases)
            if args.evidence_cases is not None
            else int(interface["default_evidence_case_count"])
        )
        if evidence_cases not in {
            int(value) for value in interface["evidence_case_counts"]
        }:
            raise SystemExit("requested evidence depth violates the interface")
        if args.submission is None:
            raise SystemExit("--submission is required")
        raw_candidate = Path(args.submission)
        candidate = raw_candidate.resolve()
        if (
            not candidate.is_relative_to(workspace)
            or not candidate.is_file()
            or raw_candidate.is_symlink()
            or candidate.name != "submission.py"
        ):
            raise SystemExit(
                "candidate must be a regular workspace file named submission.py"
            )
        relative = candidate.relative_to(workspace)
        if relative.parts[0] in {
            "cases",
            "released_completed",
            "released_truth",
            "prior_solvers",
            "generator_exchange",
            "oracle_exchange",
            "evaluation_exchange",
            "online_generator_outputs",
        }:
            raise SystemExit("candidate submission must be under writable work")
        try:
            bind_solver_bundle(
                candidate,
                assets=args.asset,
                force=bool(args.asset),
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(str(exc)) from exc
        request_id = uuid.uuid4().hex
        payload = {
            "schema": REQUEST_SCHEMA,
            "request_id": request_id,
            "agent_id": manifest["agent_id"],
            "candidate_entrypoint": relative.as_posix(),
            "purpose": "development_candidate",
            "requested_evidence_case_count": evidence_cases,
            "cascade": interface["cascade"],
        }
        temporary = requests / f".{request_id}.tmp"
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(requests / f"{request_id}.json")
        if args.submit_only:
            print(json.dumps(_pending(request_id), indent=2, sort_keys=True))
            return

    deadline = time.monotonic() + max(0, int(args.wait_seconds))
    response = responses / f"{request_id}.json"
    while time.monotonic() < deadline:
        if response.is_file():
            print(response.read_text(encoding="utf-8"), end="")
            return
        time.sleep(1.0)
    print(json.dumps(_pending(request_id), indent=2, sort_keys=True))


def _pending(request_id: str) -> dict:
    return {
        "schema": PENDING_SCHEMA,
        "request_id": request_id,
        "status": "pending",
        "asynchronous_submission": True,
        "resume_command": (
            "python evaluation_client.py --resume-request "
            f"{request_id} --wait-seconds 1800"
        ),
    }


if __name__ == "__main__":
    main()
