#!/usr/bin/env python3
"""Deterministic Agent used only by the real-GPU process-worker chain smoke.

The launcher deliberately performs no model inference.  It consumes the same
public workspace and evaluator RPC as a live Agent, screens one immutable
candidate on one clean/perturbed root, and submits that exact screened bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from pde_agent_verifier_solver.engine.reference_agent_runtime import (
    load_reference_public_workspace,
    screen_reference_candidate,
    write_reference_candidate_submission,
)
from pde_agent_verifier_solver.sdk import CaseSource


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # Keep the standard launcher ABI so the normal reference runtime can inject
    # this smoke launcher without a special controller path.
    parser.add_argument("--model", required=True)
    parser.add_argument("--reasoning-effort", required=True)
    parser.add_argument("--codex-binary", required=True)
    parser.add_argument("--auth-file", required=True)
    return parser.parse_args()


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(f"required smoke Agent environment is absent: {name}")
    return value


def _gpu_audit() -> dict[str, object]:
    visible = _required_environment("CUDA_VISIBLE_DEVICES")
    if "," in visible:
        raise RuntimeError("smoke Agent must see exactly one CUDA token")

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("smoke Agent did not receive exactly one CUDA device")
    torch.cuda.set_device(0)
    probe = torch.ones(1, dtype=torch.float32, device="cuda")
    if float(probe.item()) != 1.0:
        raise RuntimeError("smoke Agent CUDA probe changed value")
    torch.cuda.synchronize()
    properties = torch.cuda.get_device_properties(0)
    device_uuid = getattr(properties, "uuid", None)
    return {
        "cuda_visible_devices": visible,
        "torch_device_count": int(torch.cuda.device_count()),
        "torch_current_device": int(torch.cuda.current_device()),
        "torch_device_name": str(properties.name),
        "torch_device_uuid": (
            None if device_uuid is None else str(device_uuid)
        ),
        "cuda_memory_allocated_bytes": int(torch.cuda.memory_allocated()),
        "cuda_memory_reserved_bytes": int(torch.cuda.memory_reserved()),
    }


def _paired_release_audit(public) -> dict[str, object]:
    if not (
        len(public.cases)
        == len(public.queries)
        == len(public.completed_answers)
        == 2
    ):
        raise RuntimeError("smoke Agent expected one two-member paired root")
    clean, perturbed = public.cases
    if (
        clean.source is CaseSource.PERTURBED
        or clean.parent_case_id is not None
        or perturbed.source is not CaseSource.PERTURBED
        or perturbed.parent_case_id != clean.case_id
        or any(answer is None for answer in public.completed_answers)
    ):
        raise RuntimeError("smoke Agent received an invalid clean/perturbed pair")
    return {
        "root_count": 1,
        "member_case_count": 2,
        "completed_answer_count": 2,
        "parent_linkage_valid": True,
    }


def main() -> int:
    # Parse, but intentionally do not use, the standard model/auth arguments.
    _arguments()
    workspace = Path(_required_environment("PDE_MACRO_WORKSPACE")).resolve()
    context_path = Path(_required_environment("PDE_MACRO_CONTEXT")).resolve()
    if context_path.parent != workspace:
        raise RuntimeError("smoke Agent context escaped its workspace")

    gpu = _gpu_audit()
    public = load_reference_public_workspace(context_path)
    release = _paired_release_audit(public)

    candidate_id = f"{public.agent_id}-gpu-chain-smoke"
    candidate_path = workspace / "gpu_chain_smoke_candidate.py"
    incumbent = public.incumbent_path.read_bytes()
    marker = (
        "\n# Distinct immutable scripted smoke candidate for "
        f"{public.agent_id}.\n"
    ).encode("utf-8")
    candidate_payload = incumbent + marker
    candidate_path.write_bytes(candidate_payload)
    candidate_sha256 = hashlib.sha256(candidate_payload).hexdigest()

    screen = screen_reference_candidate(
        context_path,
        candidate_path=candidate_path,
        candidate_id=candidate_id,
        requested_root_count=1,
    )
    candidate_summary = screen.get("candidate")
    if (
        screen.get("schema") != "macro-development-screen-response-v2"
        or screen.get("candidate_sha256") != candidate_sha256
        or screen.get("completed_root_count") != 1
        or screen.get("member_case_count") != 2
        or screen.get("completed_stage_root_counts") != [1]
        or not isinstance(candidate_summary, dict)
        or candidate_summary.get("eligible") is not True
        or screen.get("private_case_ids_included") is not False
        or screen.get("per_case_errors_included") is not False
        or screen.get("finite_truth_included") is not False
    ):
        raise RuntimeError("smoke Agent one-root screen was not valid and complete")

    submission_path = write_reference_candidate_submission(
        context_path,
        candidate_id=candidate_id,
        bundle_path=candidate_path,
        observations=(
            "Scripted process-worker smoke completed one paired-root screen.",
        ),
    )
    audit = {
        "schema": "reference-macro-scripted-gpu-smoke-agent-v1",
        "agent_id": public.agent_id,
        "round_index": public.round_index,
        "pid": os.getpid(),
        "gpu": gpu,
        "initial_release": release,
        "screen": {
            "requested_root_count": 1,
            "completed_root_count": 1,
            "member_case_count": 2,
            "candidate_sha256": candidate_sha256,
            "eligible": True,
        },
        "submission_written": submission_path.is_file(),
    }
    audit_path = workspace / "scripted-gpu-smoke-agent-audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
