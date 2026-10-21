#!/usr/bin/env python
"""Run one Codex turn behind the canonical AgentExecutionService wire.

The Controller owns deadlines, capabilities, nomination decoding, and the
final response path. This worker supplies Codex with the public canonical
workspace, supervises ephemeral authentication, and always publishes a
canonical response after Codex exits. Nominations and ``ROUND_LOG.md`` are
append/checkpoint artifacts written during the turn and are never rolled back.
"""

from __future__ import annotations

import argparse
import json
from math import isfinite
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from uuid import UUID
from typing import Mapping, Sequence

from pde_agent_verifier_solver.engine.agent_execution import (
    AGENT_CAMPAIGN_MEMORY_ENV,
    AGENT_CANDIDATE_HISTORY_ENV,
    AGENT_EXECUTION_ID_ENV,
    AGENT_NOMINATIONS_ENV,
    AGENT_PROTOCOL_MAX_BYTES_ENV,
    AGENT_REQUEST_ENV,
    AGENT_RESPONSE_ENV,
    AGENT_ROUND_LOG_ENV,
    AGENT_ROUND_LOG_MAX_BYTES_ENV,
    AGENT_WORKSPACE_ENV,
    load_agent_worker_request,
    write_agent_worker_response,
)


MODEL = "gpt-5.6-sol"
REASONING_EFFORT = "max"
_SINGLE_GPU = re.compile(
    r"^(?:[0-9]+|GPU-[0-9A-Fa-f-]+|MIG-[A-Za-z0-9._:/-]+)$"
)
_CANONICAL_AGENT_PATH_ENVIRONMENT = (
    AGENT_WORKSPACE_ENV,
    AGENT_REQUEST_ENV,
    AGENT_NOMINATIONS_ENV,
    AGENT_CAMPAIGN_MEMORY_ENV,
    AGENT_CANDIDATE_HISTORY_ENV,
    AGENT_ROUND_LOG_ENV,
)
_CANONICAL_AGENT_ENVIRONMENT = (
    *_CANONICAL_AGENT_PATH_ENVIRONMENT,
    AGENT_EXECUTION_ID_ENV,
    AGENT_ROUND_LOG_MAX_BYTES_ENV,
    AGENT_PROTOCOL_MAX_BYTES_ENV,
)
_OPTIONAL_ENVIRONMENT = (
    "VIRTUAL_ENV",
    "LC_ALL",
    "LC_CTYPE",
    "TERM",
    "USER",
    "LD_LIBRARY_PATH",
    "LIBRARY_PATH",
    "CPATH",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "CUDA_DEVICE_ORDER",
    "SLURM_JOB_ID",
    "SLURM_JOB_NAME",
    "SLURM_TMPDIR",
    "SLURM_CPUS_PER_TASK",
    "SLURM_MEM_PER_NODE",
    "SLURM_JOB_GPUS",
    "SLURM_GPUS",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a canonical scientific Agent with the configured model and reasoning effort."
    )
    parser.add_argument("--codex-binary", required=True, type=Path)
    parser.add_argument("--auth-file", required=True, type=Path)
    parser.add_argument("--cpu-only", action="store_true",
                        help="Run without a GPU; requires an empty managed CUDA device assignment.")
    parser.add_argument(
        "--runtime-root",
        required=True,
        type=Path,
        help="Trusted temporary root outside the Agent workspace.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("canonical_codex.config.toml"),
    )
    args = parser.parse_args(argv)

    response_published = False
    try:
        request = load_agent_worker_request()
        workspace = _canonical_workspace()
        _canonical_paths(workspace)
        codex_binary = _executable(args.codex_binary, "Codex binary")
        auth_source = _trusted_file(args.auth_file, "Codex auth source")
        config_source = _trusted_file(args.config, "Codex config")
        runtime_root = _trusted_runtime_root(args.runtime_root, workspace)
        if _inside(workspace, auth_source):
            raise ValueError("Codex auth source must stay outside the Agent workspace")
        if _inside(workspace, config_source):
            raise ValueError("Codex config must stay outside the Agent workspace")

        gpu_sandbox_backend = None
        if args.cpu_only and request.get("worker_settings", {}).get("agent_gpu_execution") is not None:
            raise ValueError("--cpu-only conflicts with a required Agent GPU execution profile")
        if request.get("worker_settings", {}).get("agent_gpu_execution") is not None:
            from shared_agent_gpu import qualify_agent_gpu
            gpu_sandbox_backend = qualify_agent_gpu(
                request, codex_binary=codex_binary, config_source=config_source,
                runtime_root=runtime_root, workspace=workspace,
            )
        model, reasoning_effort = _model_settings(request)
        resume_max_attempts = _resume_attempt_limit(request)
        native_response_timeout_seconds = _native_response_timeout(request)
        resume_until_deadline = _resume_until_deadline(request)
        native_session_persistence = _native_session_persistence(request)
        prompt = _research_prompt(request)
        if args.cpu_only:
            prompt = prompt.replace(
                "The Agent turn may use its assigned GPU for research, but every\nsubmitted bundle must execute correctly without CUDA.",
                "This Agent turn and every submitted solver use CPU only; no GPU is assigned.",
            )
        if args.cpu_only:
            prompt = prompt.replace(
                "The Agent turn may also use its assigned GPU for research.",
                "This Agent turn has no assigned research GPU; Candidate evaluation resources are separate.",
            )
        prompt_path = workspace / "CANONICAL_AGENT_PROMPT.md"
        _write_new(prompt_path, prompt + "\n")
        final_message = workspace / "codex-last-message.txt"
        auth_audit_copy = workspace / "codex-auth-lifecycle.json"
        for output in (final_message, auth_audit_copy):
            if output.exists() or output.is_symlink():
                raise FileExistsError(f"refusing to overwrite {output.name}")

        return_code, native_failure = _run_codex(
            workspace=workspace,
            prompt=prompt,
            codex_binary=codex_binary,
            auth_source=auth_source,
            config_source=config_source,
            runtime_root=runtime_root,
            final_message=final_message,
            auth_audit_copy=auth_audit_copy,
            model=model,
            reasoning_effort=reasoning_effort,
            resume_max_attempts=resume_max_attempts,
            native_response_timeout_seconds=native_response_timeout_seconds,
            resume_until_deadline=resume_until_deadline,
            completion_driven=request.get("round", {}).get("time_budget_mode") == "completion",
            cpu_only=args.cpu_only,
            **({"native_session_persistence": True} if native_session_persistence else {}),
            **({"gpu_sandbox_backend": gpu_sandbox_backend}
               if gpu_sandbox_backend is not None else {}),
        )
        if return_code == 0 and native_failure is None:
            write_agent_worker_response(
                observations=(
                    "Codex completed the canonical research turn; committed "
                    "nominations and the last complete round log were preserved.",
                )
            )
        else:
            write_agent_worker_response(
                status="failed",
                observations=(
                    "Codex ended before normal completion; any committed "
                    "nominations and the last complete round log were preserved.",
                ),
                error=(f"Codex exited with status {return_code}"
                       + (f"; {native_failure}" if native_failure else "")),
            )
        response_published = True
        # A worker-reported failure uses a valid canonical response. Returning
        # zero lets AgentExecutionService classify it from that response while
        # retaining every already committed nomination.
        return 0
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        message = f"canonical Codex worker failed: {type(exc).__name__}: {exc}"
        print(message, file=sys.stderr, flush=True)
        try:
            write_agent_worker_response(status="failed", error=message)
        except Exception:
            return 1
        response_published = True
        return 0
    finally:
        # Kept as an explicit invariant for future refactors: this worker must
        # never silently report success without publishing response.json.
        if response_published:
            sys.stderr.flush()


def _canonical_workspace() -> Path:
    raw = os.environ.get(AGENT_WORKSPACE_ENV)
    if not raw:
        raise ValueError(f"{AGENT_WORKSPACE_ENV} is required")
    workspace = Path(raw)
    if not workspace.is_dir() or workspace.is_symlink():
        raise ValueError(f"{AGENT_WORKSPACE_ENV} must be a real directory")
    workspace = workspace.resolve()
    if Path.cwd().resolve() != workspace:
        raise ValueError("canonical Agent worker must start in its workspace")
    return workspace


def _canonical_paths(workspace: Path) -> Mapping[str, Path]:
    paths: dict[str, Path] = {}
    for name in _CANONICAL_AGENT_PATH_ENVIRONMENT:
        value = os.environ.get(name)
        if not value:
            raise ValueError(f"{name} is required")
        path = Path(value).resolve()
        if not _inside(workspace, path):
            raise ValueError(f"{name} must stay inside the Agent workspace")
        paths[name] = path
    if not paths[AGENT_REQUEST_ENV].is_file():
        raise ValueError("canonical Agent request is missing")
    if not paths[AGENT_NOMINATIONS_ENV].is_dir():
        raise ValueError("canonical Agent nominations directory is missing")
    for name in (
        AGENT_CAMPAIGN_MEMORY_ENV,
        AGENT_CANDIDATE_HISTORY_ENV,
        AGENT_ROUND_LOG_ENV,
    ):
        if not paths[name].is_file() or paths[name].is_symlink():
            raise ValueError(f"{name} must name a regular workspace file")
    response = Path(os.environ[AGENT_RESPONSE_ENV]).resolve()
    if not _inside(workspace, response):
        raise ValueError(f"{AGENT_RESPONSE_ENV} must stay inside the Agent workspace")
    if response.exists() or response.is_symlink():
        raise FileExistsError("canonical Agent response already exists")
    paths[AGENT_RESPONSE_ENV] = response
    if not os.environ.get(AGENT_EXECUTION_ID_ENV):
        raise ValueError(f"{AGENT_EXECUTION_ID_ENV} is required")
    for name in (AGENT_ROUND_LOG_MAX_BYTES_ENV, AGENT_PROTOCOL_MAX_BYTES_ENV):
        try:
            value = int(os.environ.get(name, ""))
        except ValueError as exc:
            raise ValueError(f"{name} must be a positive integer") from exc
        if value < 1:
            raise ValueError(f"{name} must be a positive integer")
    return paths


def _research_prompt(request: Mapping[str, object]) -> str:
    round_node = _mapping(request, "round")
    agent_node = _mapping(request, "agent")
    prompt = agent_node.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("canonical Agent request has no scientific prompt")
    settings = request.get("worker_settings", {})
    if not isinstance(settings, Mapping):
        raise TypeError("worker_settings must be an object")
    candidate_device = settings.get("candidate_execution", {})
    if not isinstance(candidate_device, Mapping):
        raise TypeError("worker_settings.candidate_execution must be an object")
    gpu_candidate = candidate_device.get("protocol") == "shared_gpu_v1"
    if gpu_candidate and (candidate_device.get("device") != "cuda"
                          or candidate_device.get("cuda_available") is not True):
        raise ValueError("shared_gpu_v1 requires an explicitly configured CUDA Candidate")
    if candidate_device.get("device") == "cuda" and not gpu_candidate:
        raise ValueError("CUDA Candidates require the versioned shared_gpu_v1 protocol")
    nomination_policy = settings.get("nomination_policy", "all")
    if nomination_policy not in ("all", "latest", "single_primary_v1"):
        raise ValueError("worker_settings.nomination_policy must be all, latest or single_primary_v1")
    if nomination_policy == "single_primary_v1":
        nomination_instructions = (
            "Explicitly choose one primary Candidate to represent your work this round. "
            "Ordinary agent_tool submit saves a frozen version without changing that choice. "
            "Use agent_tool submit BUNDLE --sequence N --name NAME --summary SUMMARY "
            "--usage USAGE --primary to save and select a new version. To choose any already "
            "saved version, use agent_tool select-primary N. Later ordinary submissions do "
            "not replace the primary; status reports primary_nomination_sequence. "
            "You can change your explicit choice during development. Only the selected "
            "snapshot enters Official evaluation, alongside the Controller's incumbent. "
            "Submission sequences remain contiguous starting at 1. Preserve a runnable "
            "primary before the deadline: timeout or failure retains the last completed "
            "explicit selection. Missing or invalid selection never falls back to the "
            "latest or an automatically chosen earlier program."
        )
    elif nomination_policy == "latest":
        nomination_instructions = (
            "Choose exactly one final Candidate to represent your work this round. "
            "You may freely develop and test as many versions as you judge useful "
            "within the existing time and Case budgets. Each `agent_tool submit` "
            "atomically replaces your previously nominated representative. Only "
            "your last committed submission enters Official evaluation; previous "
            "submissions remain saved as research history. To choose an earlier "
            "version as your final result, explicitly submit that version again. "
            "Nomination sequences must be contiguous starting at 1. Keep a runnable "
            "representative committed before the deadline: if your turn times out "
            "or fails, its last committed submission is retained. An invalid final "
            "submission is not replaced by an earlier version."
        )
    else:
        nomination_instructions = (
            "You may submit zero, one, or multiple distinct Candidates. There is no "
            "nomination-count cap. Nomination sequences must be contiguous starting "
            "at 1. Each `agent_tool submit` is an immutable atomic commitment and "
            "remains eligible for recovery if the turn later times out or fails."
        )
    round_index = round_node.get("round_index")
    active_budget = round_node.get("active_budget_seconds")
    hard_wall = round_node.get("hard_wall_seconds")
    if round_node.get("time_budget_mode") == "completion":
        timing = ("This stage ends when you finish the requested task. There is no Agent "
                  "development time cap. Scheduler pauses preserve this same stage, conversation, "
                  "workspace and acquired Cases; a pause is not a final submission.")
        nomination_instructions = nomination_instructions.replace(
            "within the existing time and Case budgets", "using the advertised environment capabilities"
        ).replace("before the deadline", "before completing the stage")
    elif active_budget is None or round_node.get("time_budget_mode") == "hard_wall":
        timing = (
            f"This turn has {hard_wall} seconds of total elapsed time, starting "
            "when this Agent starts. All time spent calling the Generator, "
            "generating Cases, evaluating Candidates, and waiting for tools "
            "counts toward this same deadline; the timer never pauses."
        )
    else:
        timing = (
            f"This legacy turn has {active_budget} seconds of Agent-active time "
            f"and a {hard_wall} second real-time hard wall. Controller capability "
            "service is excluded only from the active-time clock."
        )
    candidate_limits = request.get("candidate_execution")
    candidate_limit = (
        candidate_limits.get("case_charged_wall_seconds")
        if isinstance(candidate_limits, Mapping) else None
    )
    if candidate_limit is None:
        candidate_timing = (
            "This runner has not published a numeric per-Case execution limit. "
            "Do not interpret solve_budget=None as permission for an unlimited "
            "solve; consult the public Task contract for its own preparation budget."
        )
    else:
        if (isinstance(candidate_limit, bool)
                or not isinstance(candidate_limit, (int, float))
                or not isfinite(candidate_limit) or candidate_limit <= 0):
            raise ValueError("candidate per-Case execution limit must be finite and positive")
        candidate_timing = (
            f"Each Candidate has a {candidate_limit:g}-second per-Case charged-wall "
            "limit covering the entire Candidate worker, including startup, "
            "prepare_solution, handle.evaluate, and returning its result. "
            "Controller-metered queue and shared-lock waits are excluded from "
            "this Candidate limit; actual Oracle numerical computation, your "
            "own computation, sleep, and unmetered waits are charged. "
            "This per-Case limit applies during Development and Official "
            "evaluation. Exceeding it makes that Case evaluation fail. "
            "Use any numeric solve_budget supplied by the Task, reserve time "
            "for evaluate and returning the result. "
            "A passed Development sample does not guarantee "
            "that every new Official Case will finish in time. "
            "The Agent's round deadline is separate: ALL of these waits still "
            "count toward the Agent's elapsed-time limit."
        )
    device_settings = settings.get("candidate_execution", {})
    cuda_available = isinstance(device_settings, Mapping) and device_settings.get("cuda_available") is True
    if cuda_available:
        candidate_resources = (
            "Candidate evaluations support CPU and CUDA. Each worker sees one assigned "
            "GPU as cuda:0. Use GPU acceleration when it improves accuracy or performance; "
            "CPU, GPU, and combined implementations are welcome. Return ordinary host "
            "values through AnswerView. GPU work and transfers count toward the same "
            "per-Case time budget. The Agent turn may also use its assigned GPU for research."
        )
    else:
        candidate_resources = (
            "Formal Candidate evaluations run in CPU-only subprocesses with CUDA\n"
            "disabled. The Agent turn may use its assigned GPU for research, but every\n"
            "submitted bundle must execute correctly without CUDA."
        )
    if device_settings.get("development_solver_placement") == "agent_gpu":
        candidate_resources += (
            " Development evaluate runs on your own assigned research GPU, sharing its "
            "resources with your local experiments. The other Agent uses its own GPU. "
            "Public initial/forcing queries run on the separate Controller Oracle GPU; "
            "Official and Terminal solvers use the dedicated evaluation GPU."
        )
    if gpu_candidate:
        candidate_resources += """
This run uses shared_gpu_v1. Each worker sees one assigned Candidate GPU;
concurrency on that device follows the published execution settings.
The worker checks CUDA availability and synchronizes pending work before returning.
A Candidate GPU does not prove that the Agent research GPU is available. Do not
change sandbox settings; use the assigned research device when it is available.
You may include trained weights and auxiliary files in candidate_bundle.json;
resolve them relative to __file__ so the submitted program can run independently.
Training and evaluation use the published time budgets. No additional preparation
budget applies in this profile. Use only authorized Development information.
"""
    solver_batch_size = candidate_device.get("solver_batch_size", 1)
    if solver_batch_size > 1:
        candidate_resources += (
            f"\nOptional batch solving is enabled for up to {solver_batch_size} Cases per worker. "
            "Keep prepare_solution(case, solve_budget=None) and optionally add "
            "prepare_solutions(cases, solve_budget=None), returning {case_id: handle}. "
            "The Controller calls the batch function once, then evaluates each handle separately. "
            "Scoring coordinates are not passed to preparation. "
            "A batch shares a charged-wall budget equal to the per-Case limit times its actual "
            "Case count; this is a shared limit, not a separate timer for each member. "
            "Only intervals when every member is waiting on the Controller are refunded once. "
            "Use sdk.public_sessions.query_public_cases for concurrent public input queries. "
            "Single-Case solvers remain supported. "
            f"GPU worker slots/device: {candidate_device.get('gpu_slots_per_device', 1)}; "
            f"Oracle concurrency/device: {candidate_device.get('oracle_concurrency_per_device', 1)}."
        )
    if settings.get("agent_gpu_execution") is not None:
        from shared_agent_gpu import agent_gpu_backend
        agent_gpu_backend({"worker_settings": settings})
        candidate_resources += """
The Agent research GPU is checked before this turn starts. AGENT_GPU_READINESS.json
records the result, including CUDA computation and checkpoint save/reload.
It must report status=passed; a device token alone is not that evidence.
A passed receipt permits research on the assigned cuda:0 within this turn's budget.
"""
    sampling = round_node.get("development_sampling")
    if not isinstance(sampling, Mapping):
        raise ValueError("canonical Agent request has no sampling policy")
    max_cases = sampling.get("max_cases")
    case_allowance = (
        "There is no cumulative Development Case-count limit. Choose how many "
        "Cases to acquire and evaluate within this turn's time budget."
        if max_cases is None else f"The Development Case cap is {max_cases}."
    )
    allowed_sources = sampling.get("allowed_sources")
    if not isinstance(allowed_sources, list) or any(
        not isinstance(value, str) for value in allowed_sources
    ):
        raise ValueError("canonical Agent request has invalid allowed sources")

    public_resources = ""
    public_problem = request.get("public_problem")
    if isinstance(public_problem, Mapping):
        resources_path = public_problem.get("resources_path")
        manifest_path = public_problem.get("manifest_path")
        if isinstance(resources_path, str) and resources_path:
            public_resources = f"Public Task resources are in `{resources_path}`."
            if isinstance(manifest_path, str) and manifest_path:
                public_resources += f" Their manifest is `{manifest_path}`."
            manifest = public_problem.get("manifest")
            resources = (
                manifest.get("resources", []) if isinstance(manifest, Mapping) else []
            )
            if isinstance(resources, list):
                readmes = [
                    str(Path(resources_path) / item["path"])
                    for item in resources
                    if isinstance(item, Mapping)
                    and isinstance(item.get("path"), str)
                    and Path(item["path"]).name.lower() == "readme.md"
                ]
                if readmes:
                    public_resources += " Public Task documentation: " + ", ".join(
                        f"`{path}`" for path in readmes
                    ) + "."
            public_resources += "\n\n"

    project_dependency = ""
    issued_cases = request.get("issued_cases", [])
    if isinstance(issued_cases, list) and any(
        isinstance(case, Mapping)
        and isinstance(case.get("public_data"), Mapping)
        and isinstance(case["public_data"].get("project_reconstruction"), Mapping)
        for case in issued_cases
    ):
        project_dependency = (
            "\n\nThe `public_data.project_reconstruction` descriptor names a Project-owned, "
            "content-identified scientific data dependency. Read it through its `locator`; "
            "do not copy, vendor, rename, or modify the shared artifact inside a Candidate "
            "bundle. The Project descriptor and frozen Candidate bundle are recorded "
            "together for replay."
        )

    return f"""
You are an independent scientific research Agent in round {round_index} of a
Generator-Agent-Verifier campaign. Seek the strongest valid scientific program
you can produce under the published budgets. Choose your methods, discretization, model structure and experiments freely.
There is no checklist, solver-family preference, or reward for merely producing
more Candidates. Use evidence and spend resources where they are most useful.

Take initiative in pursuing substantial improvements throughout the available
research time. A first runnable program or an initial improvement is a milestone;
keep developing the strongest result you can achieve. Carry promising ideas
through to implementation and evaluation, and let the evidence guide further
work. When an experiment disappoints, use what you learn to improve the approach
or explore another direction. Preserve promising runnable progress through atomic
submissions while continuing to investigate useful opportunities within the
remaining budget.

Learning from public research
-----------------------------
Relevant papers, numerical-method references, library documentation and public
implementations may help you find stronger approaches. Web search and network
access are available; use them freely when useful. Your choice of sources,
methods and use of research time is up to you. Keep useful source links and
experimental findings in your round notebook for later rounds.

Controller objective and role
-----------------------------
{prompt.strip()}

Canonical workspace protocol
----------------------------
`CAMPAIGN_MEMORY.md` and `CANDIDATE_HISTORY.json` contain prior-round research
notes and results. The `status` command reports this turn's
available Cases, capabilities, and evaluation request IDs with their cleanup status:

    python -m pde_agent_verifier_solver.engine.agent_tool status

{timing}
{case_allowance} Allowed sources are {", ".join(allowed_sources) or "none"}.
{public_resources}Task data comes from staged public resources and this turn's canonical
capability. Public research sources inform methods; evaluation data still comes
from the authorized Task resources and capability. Do not inspect Controller-private
files, hidden/Official truth, Generator latents, credentials, unrelated runs, or
unstaged repository data.

These commands provide Campaign sampling, evaluation, submission, and logging:

    python -m pde_agent_verifier_solver.engine.agent_tool sample SOURCE COUNT
    python -m pde_agent_verifier_solver.engine.agent_tool sample-paired SOURCE COUNT
    python -m pde_agent_verifier_solver.engine.agent_tool perturb CASE_ID...
    python -m pde_agent_verifier_solver.engine.agent_tool export-cases DEST \
        CASE_ID [CASE_ID ...]
    python -m pde_agent_verifier_solver.engine.agent_tool clone-incumbent DEST
    python -m pde_agent_verifier_solver.engine.agent_tool evaluate BUNDLE CASE_ID... \
        [--query-json QUERY.json] [--return-truth]
    python -m pde_agent_verifier_solver.engine.agent_tool evaluate BUNDLE --all \
        [--query-json QUERY.json] [--return-truth]
    python -m pde_agent_verifier_solver.engine.agent_tool evaluate-many \
        --bundle A --bundle B --case-id CASE_ID [--case-id CASE_ID ...] \
        [--query-json QUERY.json] [--return-truth]
    python -m pde_agent_verifier_solver.engine.agent_tool evaluate-many \
        --bundle A --bundle B --all [--query-json QUERY.json] [--return-truth]
    python -m pde_agent_verifier_solver.engine.agent_tool submit BUNDLE --sequence N \
        --name NAME --summary SUMMARY --usage USAGE
    python -m pde_agent_verifier_solver.engine.agent_tool log NOTES.md
    python -m pde_agent_verifier_solver.engine.agent_tool cancel-evaluation REQUEST_ID

An interrupted evaluation requests cancellation. Check `status` for confirmed
cleanup before continuing; unfinished feedback is not a completed report.

Sampling and evaluation are Development-only experiments. Official evaluation
is performed later by the Controller on one configured common hidden cohort; never
claim Development feedback is Official evidence.

`status`, `sample`, `sample-paired`, and `perturb` deliberately summarize
potentially large public values. Use `export-cases` when a Candidate needs the complete
Agent-visible public_data for specified, already-issued Case IDs. It creates a
new DEST directory with `cases.json` and safe non-pickle `arrays/*.npy`
sidecars. The manifest contains no truth, private provenance, or raw lineage;
use the printed AgentWireCodec decoder hint to recover values losslessly. In a
`sample-paired` response, `roots[i]` corresponds to `children[i]`. A `perturb`
response preserves request order, so child N corresponds to requested parent N
even though the phase-blind public projection intentionally hides raw parent
IDs, source labels, and phase.

`status` reports `capabilities.evaluation.coordinate_mode`,
`development_point_count`, `allow_grid_truth`, and `allow_point_truth`; these
describe the user's protocol, never private coordinates or seeds. In
`private_random_points` mode, do not pass `--query-json`: every evaluation call
draws a fresh private coordinate plan. In `user_fixed_query` mode, do not pass
an override: the user's fixed plan applies. Only `agent_selected_query` accepts
`--query-json`, whose file must be the exact AnswerQuery wire object with `kind`,
`fields`, `resolution`, and `points`. Use the fields and physical coordinate
names declared by the public Task contract.

`--return-truth` requests full finite prediction/truth values and coordinates,
but succeeds only when the user-authorized disclosure policy permits it. Every
evaluation returns aggregate metrics and per-Case IDs, errors, and metrics;
without that flag, prediction/truth fields and scoring coordinates remain
hidden. `evaluate-many` compares Candidates in one batch with a shared coordinate
plan, preserves bundle order, and reports
`shared_plan: true`. Sequential `evaluate` calls in `private_random_points` mode
use independent plans.

On evaluation failure, inspect `error.diagnostic` in the CLI JSON or the Python
client exception's `diagnostic` attribute. Candidate errors include the original
exception, traceback, source file, line number and source line; use them to fix
the program. Completed case feedback remains available in `evaluation_progress`
and request status. An incomplete candidate has no overall score.

Candidate ABI and nominations
-----------------------------
{candidate_timing}

Each Candidate is a workspace-relative Python bundle accepted by the public
candidate-bundle format. Its entrypoint must implement the two-phase ABI:

    prepare_solution(case, solve_budget=None) -> handle
    handle.evaluate(points_query) -> AnswerView

`prepare_solution` must be an undecorated, synchronous module-level function
with exactly those two parameters, and the entrypoint must not rebind it later.
It must build the candidate-owned continuous solution without seeing later
scoring coordinates. `handle.evaluate` may only query/project that already
prepared representation; it must not defer the solve until the scoring query
arrives. {candidate_resources} Keep the bundle portable
and declare every required algorithm file.{project_dependency}

When status reports an incumbent program, its normalized bundle is a
Controller-owned read-only snapshot. Never edit that directory in place.
To modify the incumbent, use agent_tool clone-incumbent DEST to create a
writable author bundle, whose manifest contains only schema, entrypoint, and
files. You may also create an independent new Candidate bundle.

{nomination_instructions}

Round notebook and finish
-------------------------
`agent_tool log` preserves a Markdown research record for subsequent rounds.
Before finishing, save a concise final research record and atomically submit
any chosen nominations. The wrapper publishes the canonical worker response
when Codex exits; do not create or edit `response.json` yourself.
""".strip()


def _model_settings(request: Mapping[str, object]) -> tuple[str, str]:
    settings = request.get("worker_settings", {})
    if not isinstance(settings, Mapping):
        raise TypeError("worker_settings must be an object")
    values = []
    for key, default in (("model", MODEL), ("reasoning_effort", REASONING_EFFORT)):
        value = settings.get(key, default)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"worker_settings.{key} must be nonempty text")
        values.append(value.strip())
    return values[0], values[1]


def _resume_attempt_limit(request: Mapping[str, object]) -> int:
    settings = request.get("worker_settings", {})
    if not isinstance(settings, Mapping):
        raise TypeError("worker_settings must be an object")
    attempts = settings.get("transient_resume_max_attempts", 1)
    if isinstance(attempts, bool) or not isinstance(attempts, int) or not 1 <= attempts <= 3:
        raise ValueError("transient_resume_max_attempts must be an integer from 1 to 3")
    return attempts


def _native_response_timeout(request: Mapping[str, object]) -> float:
    value = request.get("worker_settings", {}).get("native_response_timeout_seconds", 900.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value <= 0:
        raise ValueError("native_response_timeout_seconds must be finite and positive")
    return float(value)


def _resume_backoff_seconds(attempt: int, failure: str | None) -> float:
    # Reconcile the existing capacity backoff from the migrated weak runtime.
    # Every retry/cleanup/wait remains charged to the original Agent deadline.
    if "capacity" in (failure or "").lower():
        return 30.0 if attempt == 1 else 60.0
    return 5.0 if attempt == 1 else min(60.0, 15.0 * 2 ** min(attempt - 2, 2))


def _resume_until_deadline(request: Mapping[str, object]) -> bool:
    settings = request.get("worker_settings", {})
    if not isinstance(settings, Mapping):
        raise TypeError("worker_settings must be an object")
    # Completion-driven literature stages have no elapsed deadline and retain
    # their finite attempt limit. Managed rounds recover within their budget.
    value = settings.get("transient_resume_until_deadline",
                         request.get("round", {}).get("time_budget_mode") != "completion")
    if type(value) is not bool:
        raise ValueError("transient_resume_until_deadline must be a boolean")
    return value


def _native_session_persistence(request: Mapping[str, object]) -> bool:
    settings = request.get("worker_settings", {})
    if not isinstance(settings, Mapping):
        raise TypeError("worker_settings must be an object")
    value = settings.get("native_session_persistence", False)
    if type(value) is not bool:
        raise ValueError("native_session_persistence must be a boolean")
    return value


_RESUME_PROMPT = (
    "Continue the interrupted research turn using the saved conversation and current "
    "workspace. First inspect agent_tool status for Cases already acquired and "
    "evaluation cleanup; do not replay previous sampling or perturbation requests. "
    "Inspect existing files and committed submission sequences before continuing. "
    "The original research task and original Agent deadline still apply."
)


def _recorded_session(codex_home: Path, thread_id: str) -> bool:
    """Confirm this private runtime owns a persisted rollout for the exact UUID."""
    try:
        if str(UUID(thread_id)) != thread_id:
            return False
        root = codex_home / "sessions"
        if not root.is_dir() or root.is_symlink():
            return False
        for path in root.rglob(f"*{thread_id}*.jsonl"):
            if path.is_symlink() or not path.is_file() or not _inside(root.resolve(), path.resolve()):
                continue
            with path.open("rb") as handle:
                first = handle.readline(64 * 1024 + 1)
            if len(first) > 64 * 1024:
                continue
            row = json.loads(first)
            if (isinstance(row, dict) and row.get("type") == "session_meta"
                    and isinstance(row.get("payload"), dict)
                    and row["payload"].get("id") == thread_id):
                return True
    except (OSError, ValueError, UnicodeDecodeError):
        return False
    return False


def _settle_capabilities(workspace: Path, deadline: float) -> dict[str, object]:
    """Cancel old evaluations and confirm cleanup before another native turn."""
    from pde_agent_verifier_solver.engine.agent_capability import ControllerAgentCapabilityClient

    started = time.monotonic()
    limit = started + 60.0 if deadline is None else min(deadline, started + 60.0)
    cancelled = set()
    receipt = {"confirmed": False, "evaluations_seen": 0, "cancelled_count": 0,
               "reason": "deadline", "wall_seconds": 0.0}
    try:
        request = load_agent_worker_request()
        capabilities = request.get("capabilities", {})
        if not capabilities.get("evaluation", {}).get("available", False):
            receipt.update(confirmed=True, reason="evaluation_unavailable")
            return receipt
        client = ControllerAgentCapabilityClient.from_request(
            request, workspace=workspace, timeout_seconds=min(2.0, max(0.01, limit - started)),
        )
        while time.monotonic() < limit:
            _cases, evaluations = client.current_status()
            if not isinstance(evaluations, list) or any(not isinstance(row, Mapping) for row in evaluations):
                receipt["reason"] = "invalid_status"
                return receipt
            receipt["evaluations_seen"] = len(evaluations)
            if any(row.get("finished") is True and row.get("cleanup_confirmed") is not True for row in evaluations):
                receipt["reason"] = "cleanup_failed"
                return receipt
            pending = [row for row in evaluations if row.get("finished") is not True]
            if not pending:
                receipt.update(confirmed=True, reason="settled")
                return receipt
            for row in pending:
                request_id = row.get("request_id")
                if not isinstance(request_id, str) or not request_id:
                    receipt["reason"] = "invalid_status"
                    return receipt
                if request_id not in cancelled and time.monotonic() < limit:
                    client.cancel_evaluation(request_id, wait_seconds=0.0)
                    cancelled.add(request_id)
            receipt["cancelled_count"] = len(cancelled)
            time.sleep(min(0.2, max(0.0, limit - time.monotonic())))
    except Exception:
        receipt["reason"] = "status_unavailable"
    finally:
        receipt["wall_seconds"] = max(0.0, time.monotonic() - started)
    return receipt


def _write_recovery_audit(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _run_codex(
    *,
    workspace: Path,
    prompt: str,
    codex_binary: Path,
    auth_source: Path,
    config_source: Path,
    runtime_root: Path,
    final_message: Path,
    auth_audit_copy: Path,
    model: str = MODEL,
    reasoning_effort: str = REASONING_EFFORT,
    resume_max_attempts: int = 1,
    resume_until_deadline: bool = False,
    native_response_timeout_seconds: float = 900.0,
    completion_driven: bool = False,
    cpu_only: bool = False,
    native_session_persistence: bool = False,
    gpu_sandbox_backend=None,
) -> tuple[int, str | None]:
    gpu_native_launch = None
    gpu_cli_overrides = ()
    if gpu_sandbox_backend is not None:
        from shared_agent_gpu import AgentGpuNativeLaunch, validate_agent_gpu_native_launch
        if isinstance(gpu_sandbox_backend, AgentGpuNativeLaunch):
            validate_agent_gpu_native_launch(gpu_sandbox_backend, codex_binary=codex_binary)
            gpu_native_launch = gpu_sandbox_backend
            config_source = gpu_native_launch.config_source
            gpu_cli_overrides = gpu_native_launch.cli_overrides
        elif gpu_sandbox_backend == "legacy-landlock":
            gpu_cli_overrides = ("--enable", "use_legacy_landlock")
        else:
            raise ValueError("unqualified Agent GPU sandbox backend")
    if type(native_session_persistence) is not bool:
        raise ValueError("native_session_persistence must be a boolean")
    if type(resume_until_deadline) is not bool:
        raise ValueError("transient_resume_until_deadline must be a boolean")
    recovery_enabled = resume_until_deadline or resume_max_attempts > 1 or native_session_persistence
    # Preserve one conversation across model-service interruptions.
    persist_session = recovery_enabled or completion_driven
    repository_root = Path(__file__).resolve().parents[1]
    supervisor = _trusted_file(
        Path(__file__).with_name("run_codex_with_ephemeral_auth.py"),
        "ephemeral-auth supervisor",
    )
    raw_deadline = os.environ.get("PDE_AGENT_DEADLINE_MONOTONIC")
    deadline = float(raw_deadline) if raw_deadline is not None else None
    if deadline is not None and (not isfinite(deadline) or deadline <= 0):
        raise ValueError("managed Agent deadline must be finite and positive")
    if resume_until_deadline and deadline is None:
        raise ValueError("deadline-based continuation requires the managed Agent deadline")
    if resume_max_attempts > 1 and deadline is None and not completion_driven:
        raise ValueError("automatic same-session continuation requires the managed Agent deadline")
    if native_session_persistence and deadline is None:
        raise ValueError("persistent native sessions require the managed Agent deadline")
    recovery_path = workspace / "codex-recovery.json"
    recovery = {"schema": "canonical_codex_same_session_recovery_v1",
                "max_attempts": None if resume_until_deadline else resume_max_attempts,
                "until_deadline": resume_until_deadline, "deadline_monotonic": deadline,
                "attempts": [], "status": "running"}
    if native_session_persistence:
        recovery["native_session_persistence"] = True
    if recovery_enabled:
        _write_new(recovery_path, json.dumps(recovery, indent=2) + "\n")
    marker = workspace / "codex-continuation.json"
    prior_thread = None
    if completion_driven and marker.exists():
        if marker.is_symlink():
            raise ValueError("continuation marker must be a regular file")
        saved = json.loads(marker.read_text())
        runtime = Path(saved["runtime"])
        if (runtime.parent.resolve() != runtime_root.resolve() or runtime.is_symlink()
                or not runtime.is_dir() or saved.get("execution_id") != os.environ.get(AGENT_EXECUTION_ID_ENV)
                or saved.get("workspace") != str(workspace) or saved["model"] != model
                or saved["reasoning_effort"] != reasoning_effort):
            raise ValueError("private session runtime or model changed")
        sessions = set()
        for record in (runtime / "codex-home" / "sessions").rglob("*.jsonl"):
            if record.is_symlink():
                raise ValueError("session file cannot be a symlink")
            with record.open() as handle:
                row = json.loads(handle.readline(65536))
            if row.get("type") == "session_meta":
                sessions.add(row["payload"]["id"])
        if len(sessions) != 1:
            raise RuntimeError("allocation resume requires exactly one recorded native session")
        prior_thread = sessions.pop()
        if not _recorded_session(runtime / "codex-home", prior_thread):
            raise RuntimeError("saved native session could not be verified")
    else:
        runtime = Path(tempfile.mkdtemp(prefix="canonical-codex-", dir=runtime_root))
        if completion_driven:
            _write_new(marker, json.dumps({"runtime": str(runtime), "model": model,
                "reasoning_effort": reasoning_effort, "workspace": str(workspace),
                "execution_id": os.environ.get(AGENT_EXECUTION_ID_ENV)}) + "\n")
    try:
        codex_home = runtime / "codex-home"
        agent_home = runtime / "agent-home"
        codex_home.mkdir(mode=0o700, exist_ok=completion_driven)
        codex_home.chmod(0o700)
        agent_home.mkdir(mode=0o700, exist_ok=completion_driven)
        auth = codex_home / "auth.json"
        # Every attempt uses the same credential bytes loaded for this turn;
        # a concurrently changed source file cannot switch accounts on resume.
        auth_bytes = auth_source.read_bytes()
        config = codex_home / "config.toml"
        shutil.copyfile(config_source, config)
        os.chmod(config, 0o600)
        sqlite_home = runtime / "sqlite"
        if persist_session:
            # Only session/state subdirectories stay writable. Auth/config's
            # parent still becomes read-only before model tools can execute.
            (codex_home / "sessions").mkdir(mode=0o700, exist_ok=completion_driven)
            sqlite_home.mkdir(mode=0o700, exist_ok=completion_driven)

        environment = _codex_environment(
            repository_root=repository_root,
            workspace=workspace,
            codex_home=codex_home,
            agent_home=agent_home,
            cpu_only=cpu_only,
        )
        if gpu_native_launch is not None and gpu_native_launch.gpu_uuid is not None:
            # Use the same identity as the successful sandbox CUDA probe.
            environment['CUDA_VISIBLE_DEVICES'] = gpu_native_launch.gpu_uuid
        native_command = (
            str(codex_binary),
            "exec",
            "--model",
            model,
            "--config",
            f"model_reasoning_effort={json.dumps(reasoning_effort)}",
            *(("--ephemeral",) if not persist_session else (
                "--config", f"sqlite_home={json.dumps(str(sqlite_home))}",
            )),
            "--ignore-rules",
            "--strict-config",
            "--enable",
            "shell_tool",
            *gpu_cli_overrides,
            "--disable",
            "apps",
            "--disable",
            "plugins",
            "--disable",
            "browser_use",
            "--disable",
            "browser_use_external",
            "--disable",
            "browser_use_full_cdp_access",
            "--disable",
            "computer_use",
            "--disable",
            "image_generation",
            "--disable",
            "in_app_browser",
            "--disable",
            "memories",
            "--disable",
            "goals",
            "--disable",
            "workspace_dependencies",
            "--disable",
            "skill_mcp_dependency_install",
            "--config",
            'history.persistence="none"',
            "--json",
            "--color",
            "never",
            "--output-last-message",
            str(final_message),
            "--skip-git-repo-check",
            "--cd",
            str(workspace),
        )
        thread_id = prior_thread
        attempt = 0
        while True:
            attempt += 1
            if gpu_native_launch is not None:
                validate_agent_gpu_native_launch(gpu_native_launch, codex_binary=codex_binary)
                if config.read_bytes() != gpu_native_launch.config_source.read_bytes():
                    raise ValueError("private native GPU config changed before launch")
            if deadline is not None and time.monotonic() >= deadline:
                recovery["status"] = "deadline"
                if recovery_enabled:
                    _write_recovery_audit(recovery_path, recovery)
                return 1, "original Agent deadline reached"
            if attempt > 1:
                for previous in (final_message, auth_audit_copy):
                    if previous.is_symlink():
                        raise ValueError("native attempt output cannot be a symlink")
                    if previous.exists():
                        previous.rename(previous.with_name(f"attempt-{attempt - 1:02d}-{previous.name}"))
            # The preceding attempt's process group has been confirmed stopped
            # before credentials can be copied back into this same private home.
            codex_home.chmod(0o700)
            (codex_home / "auth-lifecycle.json").unlink(missing_ok=True)
            auth.write_bytes(auth_bytes)
            os.chmod(auth, 0o600)
            result_path = runtime / f"codex-result-{time.time_ns()}-{attempt}.json"
            recovery.update(status="running", current_attempt=attempt, thread_id=thread_id)
            if recovery_enabled:
                _write_recovery_audit(recovery_path, recovery)
            command = (
                sys.executable, str(supervisor), "--auth", str(auth),
                "--audit", str(codex_home / "auth-lifecycle.json"),
                "--audit-copy", str(auth_audit_copy), "--result", str(result_path),
                *(("--shutdown-receipt", str(Path(os.environ[AGENT_REQUEST_ENV]).parent / "native_shutdown.json"))
                  if completion_driven else ()),
                *(("--deadline-monotonic", str(deadline)) if deadline is not None else ()),
                *(("--expected-thread-id", thread_id) if thread_id is not None else ()),
                "--response-timeout-seconds", str(native_response_timeout_seconds),
                *(("--require-descendant-cleanup",) if persist_session else ()),
                "--", *native_command,
                *(("resume", thread_id, "-") if thread_id is not None else ("-",)),
            )
            started = time.monotonic()
            completed = subprocess.run(
                command,
                input=((
                    "Continue the SAME interrupted stage and saved workspace. The Controller "
                    "has refreshed request/capability files and restored acquired Cases. "
                    "Reload them using agent_tool; do not reuse cached RPC tokens or replay "
                    "sampling requests. Read your existing code, ROUND_LOG.md and nominations. "
                    "There is no Agent deadline. Finish the original stage.\n\n" + prompt
                    if completion_driven and thread_id is not None else
                    prompt if attempt == 1 else _RESUME_PROMPT
                )).encode("utf-8"),
                cwd=workspace, env=environment, check=False,
            )
            result = _read_codex_result(result_path)
            failure = result["failure"]
            safety_valid = True
            try:
                _validate_auth_audit(auth_audit_copy)
            except (RuntimeError, OSError):
                safety_valid = False
                safety_error = "auth safety check incomplete (ephemeral-auth audit unavailable or invalid)"
                failure = f"{failure}; {safety_error}" if failure else safety_error
            if not result["process_group_cleaned"]:
                safety_valid = False
                cleanup_error = "native process-group cleanup unconfirmed"
                failure = f"{failure}; {cleanup_error}" if failure else cleanup_error
            if completed.returncode == 0 and failure is None:
                if not final_message.is_file() or final_message.is_symlink():
                    raise RuntimeError("Codex did not publish its final message")
                stop_reason = "completed"
            elif not safety_valid:
                stop_reason = "cleanup_or_auth_unconfirmed"
            elif not result["retryable"]:
                stop_reason = "not_retryable"
            elif not resume_until_deadline and attempt >= resume_max_attempts:
                stop_reason = "attempt_limit"
            elif result["thread_id"] is None or not _recorded_session(codex_home, result["thread_id"]):
                stop_reason = "no_persisted_session"
            elif thread_id is not None and result["thread_id"] != thread_id:
                stop_reason = "session_mismatch"
            else:
                stop_reason = "retry"
            capability_cleanup = None
            if stop_reason == "retry":
                capability_cleanup = _settle_capabilities(workspace, deadline)
                if not capability_cleanup["confirmed"]:
                    stop_reason = "controller_cleanup_unconfirmed"
            delay = _resume_backoff_seconds(attempt, failure) if stop_reason == "retry" else 0.0
            if stop_reason == "retry" and deadline is not None:
                remaining = max(0.0, deadline - time.monotonic())
                if resume_until_deadline:
                    # Wait only within the original budget, including the final
                    # partial backoff; the Controller can still cancel us.
                    delay = min(delay, remaining)
                    if remaining <= 0:
                        stop_reason = "deadline"
                elif delay >= remaining:
                    stop_reason, delay = "deadline", 0.0
            recovery["attempts"].append({
                "attempt": attempt, "return_code": completed.returncode,
                "failure": failure, "thread_id": result["thread_id"],
                "process_group_cleaned": result["process_group_cleaned"],
                "wall_seconds": max(0.0, time.monotonic() - started),
                "remaining_seconds": None if deadline is None else max(0.0, deadline - time.monotonic()),
                "decision": stop_reason, "backoff_seconds": delay,
                "controller_cleanup": capability_cleanup,
            })
            recovery["status"] = stop_reason
            if recovery_enabled:
                _write_recovery_audit(recovery_path, recovery)
            if stop_reason != "retry":
                return completed.returncode, failure
            thread_id = result["thread_id"]
            remaining_text = ("no elapsed deadline" if deadline is None else
                              f"{max(0.0, deadline - time.monotonic()):.1f}s remaining")
            print(f"[Agent recovery] {failure}; resuming the same session in "
                  f"{delay:.1f}s ({remaining_text}).", file=sys.stderr, flush=True)
            time.sleep(delay)
    finally:
        if not completion_driven:
            _remove_runtime(runtime)
        else:
            # Retain conversation/state but never a credential for the next job.
            home = runtime / "codex-home"
            if home.is_dir():
                home.chmod(0o700)
                (home / "auth.json").unlink(missing_ok=True)
                home.chmod(0o500)


def _read_codex_result(path: Path) -> dict[str, object]:
    # Fixed labels and a session UUID only; never persist raw provider responses
    # or authentication data in the public recovery audit.
    invalid = {"failure": "native exit summary unavailable or invalid", "thread_id": None,
               "retryable": False, "process_group_cleaned": False}
    try:
        if path.is_symlink() or path.stat().st_size > 2048:
            raise ValueError("invalid exit summary")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or set(payload) != set(invalid):
            raise ValueError("invalid exit summary")
        failure = payload["failure"]
        if failure is not None and (not isinstance(failure, str) or len(failure) > 512):
            raise ValueError("invalid exit summary")
        thread_id = payload["thread_id"]
        if thread_id is not None and (not isinstance(thread_id, str) or str(UUID(thread_id)) != thread_id):
            raise ValueError("invalid thread id")
        if not isinstance(payload["retryable"], bool) or not isinstance(payload["process_group_cleaned"], bool):
            raise ValueError("invalid cleanup/retry flags")
        return payload
    except (OSError, ValueError, UnicodeDecodeError):
        return invalid


def _read_codex_failure(path: Path) -> str | None:
    return _read_codex_result(path)["failure"]


def _codex_environment(
    *,
    repository_root: Path,
    workspace: Path,
    codex_home: Path,
    agent_home: Path,
    cpu_only: bool = False,
) -> dict[str, str]:
    if cpu_only and os.environ.get("CUDA_VISIBLE_DEVICES", ""):
        raise ValueError("CPU-only Agent cannot have assigned CUDA devices")
    inherited_pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath = str(repository_root / "src")
    if inherited_pythonpath:
        pythonpath += os.pathsep + inherited_pythonpath
    python_bin = str(Path(sys.executable).parent)
    inherited_path = os.environ.get("PATH", os.defpath)
    path = python_bin
    if inherited_path:
        path += os.pathsep + inherited_path
    no_proxy = "127.0.0.1,localhost"
    if os.environ.get("NO_PROXY"):
        no_proxy += "," + os.environ["NO_PROXY"]
    environment = {
        "HOME": str(agent_home),
        "CODEX_HOME": str(codex_home),
        "PATH": path,
        "PYTHONPATH": pythonpath,
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "TMPDIR": str(workspace / "tmp"),
        "TMP": str(workspace / "tmp"),
        "TEMP": str(workspace / "tmp"),
        "PYTHONDONTWRITEBYTECODE": "1",
        # ``py_compile`` ignores PYTHONDONTWRITEBYTECODE.  Keep every bytecode
        # artifact in the execution-owned cache so a local syntax/import check
        # cannot add an undeclared file to a closed Candidate bundle.
        "PYTHONPYCACHEPREFIX": str(workspace / "pycache"),
        "PYTHONUNBUFFERED": "1",
        "PYTHONNOUSERSITE": "1",
        "CUDA_VISIBLE_DEVICES": "" if cpu_only else _single_gpu(),
        "NO_PROXY": no_proxy,
    }
    for name in _OPTIONAL_ENVIRONMENT:
        value = os.environ.get(name)
        if value:
            environment[name] = value
    # The command Agent has its own declared CPU allowance (eight threads in
    # formal NS), independent of the Controller's single-thread Oracle kernels.
    # NumPy/SciPy may use OpenBLAS or BLIS instead of the OMP/MKL defaults.
    for name in ("OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                 "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        environment[name] = environment.get("OMP_NUM_THREADS", "1")
    if sys.prefix != getattr(sys, "base_prefix", sys.prefix):
        environment["VIRTUAL_ENV"] = sys.prefix
    for name in _CANONICAL_AGENT_ENVIRONMENT:
        environment[name] = os.environ[name]
    # PDE_AGENT_RESPONSE is deliberately worker-owned. Codex uses agent_tool
    # for nominations/logs; only this wrapper publishes the final response.
    return environment


def _validate_auth_audit(path: Path) -> None:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("ephemeral-auth supervisor did not publish its audit")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("ephemeral-auth audit is invalid") from exc
    if not isinstance(payload, dict) or payload.get("auth_removed_before_first_event_forwarded") is not True:
        raise RuntimeError("Codex auth was not removed before tool execution")


def _single_gpu() -> str:
    value = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not _SINGLE_GPU.fullmatch(value):
        raise ValueError("CUDA_VISIBLE_DEVICES must name exactly one GPU or MIG device")
    order = os.environ.get("CUDA_DEVICE_ORDER")
    if order not in {None, "", "FASTEST_FIRST", "PCI_BUS_ID"}:
        raise ValueError("CUDA_DEVICE_ORDER must be FASTEST_FIRST or PCI_BUS_ID")
    return value


def _mapping(payload: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = payload.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"canonical Agent request has no {name} object")
    return value


def _executable(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ValueError(f"{label} must be an executable regular file")
    return resolved


def _trusted_file(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{label} cannot be a symlink")
    resolved = path.expanduser().resolve()
    try:
        mode = resolved.stat().st_mode
    except FileNotFoundError as exc:
        raise ValueError(f"{label} is missing") from exc
    if not stat.S_ISREG(mode):
        raise ValueError(f"{label} must be a regular file")
    return resolved


def _trusted_runtime_root(path: Path, workspace: Path) -> Path:
    if path.is_symlink():
        raise ValueError("runtime root cannot be a symlink")
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError("runtime root must be an existing directory")
    if _inside(workspace, resolved) or _inside(resolved, workspace):
        raise ValueError("runtime root and Agent workspace must be disjoint")
    return resolved


def _inside(root: Path, path: Path) -> bool:
    return path == root or root in path.parents


def _write_new(path: Path, text: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _remove_runtime(path: Path) -> None:
    # Runtime caches can contain links to the shared Codex executable.  Never
    # chmod a link target, including when the runtime root itself is a link.
    if path.is_symlink():
        path.unlink()
        return
    try:
        path.chmod(0o700, follow_symlinks=False)
    except FileNotFoundError:
        return
    # Unlinking even a read-only file needs a writable parent, not chmod on
    # the file.  Only repair real directories, before descending into them.
    for directory, names, _files in os.walk(path, topdown=True, followlinks=False):
        for name in names:
            member = Path(directory) / name
            if member.is_symlink():
                continue
            try:
                member.chmod(0o700, follow_symlinks=False)
            except FileNotFoundError:
                pass
    shutil.rmtree(path)


if __name__ == "__main__":
    raise SystemExit(main())
