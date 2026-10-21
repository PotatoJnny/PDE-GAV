"""Reproduce the legacy VPFP fresh-U0 macro experiment.

This module is the historical composition root for the reference U0
experiment. It binds a verified historical release bank and the VPFP macro
objective, while all Agent interaction, candidate selection, paired promotion,
and terminal evaluation remain in :class:`MacroClosedLoopRunner`; canonical
product runs use :class:`CampaignController` instead.

The runtime is intentionally ``reference_in_process``.  Replaying a release
bank is not a fresh Generator draw, and the same-user Agent/candidate execution
path is not a security boundary.  Published results state those limitations
and never contain truth arrays or private target commitments.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from math import isfinite
import os
from pathlib import Path
import stat
from statistics import fmean
import tempfile
import time
from typing import Mapping, Sequence

from pde_agent_verifier_solver.engine.macro_closed_loop import (
    MacroAgentSpec,
    MacroClosedLoopRunner,
    MacroLoopConfig,
    MacroLoopResult,
)
from pde_agent_verifier_solver.engine.reference_agent_runtime import (
    ReferenceFilesystemBundleAttestor,
    ReferenceFilesystemMacroAgentProgram,
    SubprocessReferenceAgentLauncher,
    load_python_macro_candidate,
)
from pde_agent_verifier_solver.engine.selection import EvidenceStage
from pde_agent_verifier_solver.sdk import (
    AgentBundleSubmission,
    CandidateRef,
    CandidateSubmission,
    ExecutionAssurance,
    candidate_ref_from_attestation,
)

from .macro_loop import bind_vpfp_release_bank_reference


RESULT_SCHEMA = "pde-agent-verifier-result-v1"
PROTOCOL_ID = "unified_macro_reference_v1"
_RUN_NAME = "vpfp-unified-fresh-u0-reference"
_COLD_CANDIDATE_ID = "vpfp-hold-initial-u0"
_COLD_AGENT_ID = "controller-cold-start"
_FRESH_AGENT_ID = "fresh-observable-agent"
_MAX_CANDIDATE_BYTES = 2 * 1024 * 1024

# The reference launcher itself inherits only PATH/locale/TMPDIR.  These are
# the additional non-secret execution settings needed by the scheduled Agent.
_EXPLICIT_AGENT_ENVIRONMENT = (
    "VIRTUAL_ENV",
    "PYTHONPATH",
    "SLURM_JOB_ID",
    "SLURM_TMPDIR",
    "SLURM_CPUS_PER_TASK",
    "USER",
    "LD_LIBRARY_PATH",
    "LIBRARY_PATH",
    "CPATH",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
)


class FreshU0RunError(RuntimeError):
    """The reference U0 run cannot produce a verified product artifact."""


@dataclass(frozen=True)
class FreshU0Config:
    release_bank_report: Path
    decoder_checkpoint: Path
    run_root: Path
    cold_start_candidate: Path
    agent_launcher: Path
    codex_binary: Path
    auth_file: Path
    model: str
    reasoning_effort: str
    agent_timeout_seconds: float
    development_case_credits: int
    round_count: int
    development_case_count: int
    selection_case_count: int
    promotion_case_count: int
    terminal_case_count: int
    seed_base: int
    seed_stride: int
    development_pool_multiplier: int
    promotion_confidence_z: float
    promotion_min_mean_improvement: float
    promotion_min_win_rate: float
    promotion_require_cvar_no_regression: bool
    terminal_confidence_z: float
    terminal_min_win_rate: float
    terminal_require_cvar_no_regression: bool
    candidate_compute_budget: float | None = None

    def __post_init__(self) -> None:
        input_paths = {
            "release-bank report": self.release_bank_report,
            "decoder checkpoint": self.decoder_checkpoint,
            "cold-start candidate": self.cold_start_candidate,
            "Agent launcher": self.agent_launcher,
            "Codex binary": self.codex_binary,
            "auth file": self.auth_file,
        }
        resolved: dict[str, Path] = {}
        for label, raw in input_paths.items():
            path = Path(raw).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"{label} is not a regular file: {path}")
            resolved[label] = path
        for label in ("Agent launcher", "Codex binary"):
            if not os.access(resolved[label], os.X_OK):
                raise PermissionError(f"{label} is not executable: {resolved[label]}")

        object.__setattr__(self, "release_bank_report", resolved["release-bank report"])
        object.__setattr__(self, "decoder_checkpoint", resolved["decoder checkpoint"])
        object.__setattr__(self, "cold_start_candidate", resolved["cold-start candidate"])
        object.__setattr__(self, "agent_launcher", resolved["Agent launcher"])
        object.__setattr__(self, "codex_binary", resolved["Codex binary"])
        object.__setattr__(self, "auth_file", resolved["auth file"])
        run_root = Path(self.run_root).expanduser().resolve()
        if run_root.exists() and not run_root.is_dir():
            raise NotADirectoryError(f"run root is not a directory: {run_root}")
        object.__setattr__(self, "run_root", run_root)

        if not self.model.strip() or not self.reasoning_effort.strip():
            raise ValueError("model and reasoning effort cannot be empty")
        _require_positive_number(
            self.agent_timeout_seconds,
            "agent_timeout_seconds",
        )
        for label, value in (
            ("development_case_credits", self.development_case_credits),
            ("round_count", self.round_count),
            ("development_case_count", self.development_case_count),
            ("selection_case_count", self.selection_case_count),
            ("promotion_case_count", self.promotion_case_count),
            ("terminal_case_count", self.terminal_case_count),
            ("development_pool_multiplier", self.development_pool_multiplier),
        ):
            _require_integer(value, label, minimum=1)
        _require_integer(self.seed_base, "seed_base", minimum=0)
        _require_integer(self.seed_stride, "seed_stride", minimum=4)
        for label, value in (
            ("promotion_confidence_z", self.promotion_confidence_z),
            (
                "promotion_min_mean_improvement",
                self.promotion_min_mean_improvement,
            ),
            ("terminal_confidence_z", self.terminal_confidence_z),
        ):
            _require_nonnegative_number(value, label)
        for label, value in (
            ("promotion_min_win_rate", self.promotion_min_win_rate),
            ("terminal_min_win_rate", self.terminal_min_win_rate),
        ):
            if not isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{label} must lie in [0, 1]")
        for label, value in (
            (
                "promotion_require_cvar_no_regression",
                self.promotion_require_cvar_no_regression,
            ),
            (
                "terminal_require_cvar_no_regression",
                self.terminal_require_cvar_no_regression,
            ),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{label} must be bool")
        if self.candidate_compute_budget is not None:
            _require_positive_number(
                self.candidate_compute_budget,
                "candidate_compute_budget",
            )


def run_fresh_u0(
    config: FreshU0Config,
    *,
    legacy_macro_reference: bool = False,
) -> Mapping[str, object]:
    """Execute and publish one reference-only fresh-Agent VPFP run."""

    if legacy_macro_reference is not True:
        raise PermissionError(
            "legacy VPFP fresh-U0 execution is disabled by default; pass "
            "legacy_macro_reference=True for an explicit history/control-arm run"
        )

    started = time.monotonic()
    config.run_root.mkdir(parents=True, exist_ok=True)
    plugin = bind_vpfp_release_bank_reference(
        report_path=config.release_bank_report,
        decoder_checkpoint=config.decoder_checkpoint,
    )
    release_bank = plugin.generator.state.base
    report_sha256 = _verified_component_sha256(
        release_bank,
        "report_sha256",
        "release-bank report",
    )
    checkpoint_sha256 = _verified_component_sha256(
        release_bank,
        "decoder_checkpoint_sha256",
        "decoder checkpoint",
    )

    controller_root = config.run_root / "_controller"
    attestor = ReferenceFilesystemBundleAttestor(
        controller_root / "sealed_python_bundles"
    )
    cold_attestation = attestor.attest(
        AgentBundleSubmission(
            candidate_id=_COLD_CANDIDATE_ID,
            bundle_uri=str(config.cold_start_candidate),
        ),
        round_index=0,
        agent_id=_COLD_AGENT_ID,
        parent_sha256=None,
    )
    cold_ref = candidate_ref_from_attestation(cold_attestation)
    cold_bundle_uri = cold_ref.bundle_uri
    if cold_bundle_uri is None:
        raise FreshU0RunError("attested cold start has no sealed bundle URI")

    def fresh_cold_program():
        return load_python_macro_candidate(
            cold_bundle_uri,
            expected_sha256=cold_ref.bundle_sha256,
            deferred=True,
        )

    cold_start = CandidateSubmission(
        cold_ref,
        fresh_cold_program(),
        program_factory=fresh_cold_program,
    )

    launcher = SubprocessReferenceAgentLauncher(
        command=(
            str(config.agent_launcher),
            "--model",
            config.model,
            "--reasoning-effort",
            config.reasoning_effort,
            "--codex-binary",
            str(config.codex_binary),
            "--auth-file",
            str(config.auth_file),
        ),
        environment=_agent_environment(os.environ),
        timeout_seconds=config.agent_timeout_seconds,
    )
    agent_program = ReferenceFilesystemMacroAgentProgram(
        run_root=controller_root / "agent_runtime",
        run_id=f"{_RUN_NAME}:{config.run_root.name}",
        launcher=launcher,
        bundle_attestor=attestor,
    )
    agent_spec = MacroAgentSpec(
        agent_id=_FRESH_AGENT_ID,
        name="Fresh VPFP observable-model Agent",
        prompt=_fresh_agent_prompt(config),
    )
    loop_config = MacroLoopConfig(
        name=_RUN_NAME,
        round_count=config.round_count,
        agents=(agent_spec,),
        development_case_count=config.development_case_count,
        selection_case_count=config.selection_case_count,
        evidence_stages=(
            EvidenceStage(
                case_count=config.promotion_case_count,
                confidence_z=config.promotion_confidence_z,
                min_mean_improvement=config.promotion_min_mean_improvement,
                min_win_rate=config.promotion_min_win_rate,
                require_cvar_no_regression=(
                    config.promotion_require_cvar_no_regression
                ),
            ),
        ),
        terminal_case_count=config.terminal_case_count,
        agent_case_credits=config.development_case_credits,
        seed_base=config.seed_base,
        seed_stride=config.seed_stride,
        development_pool_multiplier=config.development_pool_multiplier,
        terminal_confidence_z=config.terminal_confidence_z,
        terminal_min_win_rate=config.terminal_min_win_rate,
        terminal_require_cvar_no_regression=(
            config.terminal_require_cvar_no_regression
        ),
        allow_development_truth=False,
        candidate_compute_budget=config.candidate_compute_budget,
        execution_assurance=ExecutionAssurance.REFERENCE_IN_PROCESS,
        paired_perturbations=False,
    )
    result = MacroClosedLoopRunner(
        legacy_macro_reference=True,
        config=loop_config,
        plugin=plugin,
        cold_start=cold_start,
        agents={agent_spec.agent_id: agent_program},
    ).run()
    wall_time_seconds = max(0.0, time.monotonic() - started)
    return _publish_result(
        config,
        result,
        report_sha256=report_sha256,
        checkpoint_sha256=checkpoint_sha256,
        wall_time_seconds=wall_time_seconds,
    )


def _fresh_agent_prompt(config: FreshU0Config) -> str:
    return f"""
You are a fresh, independent Agent improving a VPFP observable model from a
weak hold-initial cold start.  There is no prior Agent trajectory or tuned
champion in this run.  Work only with the files staged in PDE_MACRO_WORKSPACE
and the budgeted evaluator described by PDE_MACRO_CONTEXT.  Do not inspect the
repository, controller directories, scratch storage, the web, or any hidden
truth/answer artifact.  Do not request finite truth from the evaluator.

The public cases describe 2D2V Vlasov--Poisson--Fokker--Planck problems.  Their
public_data contains the public question plus arrays such as t, x, y, vx, vy,
velocity_weights, source, g0, rho0, momentum0, and kinetic_energy0.  The query
asks for rho, momentum_x, momentum_y, and kinetic_energy on the native t/x/y
grid.  Your implementation may use any numerical or reduced model justified
by those public inputs; the shared controller only compares the four resulting
macroscopic fields.

The only candidate ABI is one Python source file (at most 2 MiB) exporting
predict_macros(case, query, budget=None) -> AnswerView.  Match query.kind,
query.fields, the public t/x/y axes, and shape (nt, nx, ny).  The candidate may
import NumPy and the public SDK, but it cannot depend on companion files.

Use load_reference_public_workspace and evaluate_reference_candidate from
pde_agent_verifier_solver.engine.reference_agent_runtime to inspect the public
tasks and spend at most {config.development_case_credits} case credits in this
round.  Evaluate deliberately: each requested case consumes one credit.  When
finished, call write_reference_candidate_submission with PDE_MACRO_CONTEXT,
one unique candidate_id, your candidate .py path inside the workspace, and
short observations.  Submit exactly once before exiting.
""".strip()


def _publish_result(
    config: FreshU0Config,
    result: MacroLoopResult,
    *,
    report_sha256: str,
    checkpoint_sha256: str,
    wall_time_seconds: float,
) -> Mapping[str, object]:
    if result.execution_assurance is not ExecutionAssurance.REFERENCE_IN_PROCESS:
        raise FreshU0RunError("fresh U0 must remain reference_in_process")
    if result.formal_claims_allowed:
        raise FreshU0RunError("reference U0 cannot publish formal claims")
    selected_payload = _verified_candidate_bytes(result.selected)
    model_path = config.run_root / "result" / "model" / "submission.py"
    _atomic_write(model_path, selected_payload, mode=0o444)

    terminal = result.terminal
    cold_mean = fmean(terminal.cold_start_score.errors)
    selected_mean = fmean(terminal.selected_score.errors)
    payload: dict[str, object] = {
        "schema": RESULT_SCHEMA,
        "protocol": PROTOCOL_ID,
        "kind": "observable_model",
        "problem": "Vlasov--Poisson--Fokker--Planck",
        "name": "Selected VPFP macroscopic observable model",
        "model": str(model_path),
        "usage": (
            "from submission import predict_macros; "
            "predict_macros(public_case, macro_query, budget=None)"
        ),
        "winner": result.selected.candidate_id,
        "rounds_completed": len(result.rounds),
        "primary_error": selected_mean,
        "primary_metric": "vpfp_macro_match_v1_relative_l2",
        "overall_improvement": _relative_improvement(cold_mean, selected_mean),
        "evaluated_cases": _evaluated_case_count(config),
        "wall_time_seconds": float(wall_time_seconds),
        "execution_assurance": result.execution_assurance.value,
        "formal_claims_allowed": False,
        "reference_only": True,
        "fresh_agent_start": True,
        "fresh_generator_draws": False,
        "evidence_scope": "verified_fixed_release_bank_replay",
        "limitations": [
            "The verified release bank is replayed rather than generated afresh.",
            "Agent and candidate execution share the controller user and node.",
            "This result cannot support a formal isolation claim.",
        ],
        "task": {
            "outputs": [
                "rho",
                "momentum_x",
                "momentum_y",
                "kinetic_energy",
            ],
            "macro_contract_id": result.macro_contract_id,
            "published_metric_names": list(result.published_metric_names),
            "candidate_abi": "predict_macros(case, query, budget=None)",
            "internal_method_is_unconstrained": True,
        },
        "generator": {
            "generator_id": result.generator_id,
            "generator_sha256": result.generator_sha256,
            "release_bank_report_sha256": report_sha256,
            "decoder_checkpoint_sha256": checkpoint_sha256,
            "fresh_draws": False,
            "paired_perturbations": result.paired_perturbations,
        },
        "artifact": {
            "path": str(model_path),
            "sha256": result.selected.bundle_sha256,
            "single_file": True,
        },
        "candidates": {
            "cold_start": _candidate_ref_payload(result.cold_start),
            "selected": _candidate_ref_payload(result.selected),
        },
        "rounds": [_round_payload(row) for row in result.rounds],
        "terminal": _terminal_payload(result),
        "resource_usage": {
            "agent_rounds": config.round_count,
            "agents_per_round": 1,
            "development_case_credit_limit_per_round": (
                config.development_case_credits
            ),
            "maximum_development_case_credits": (
                config.development_case_credits * config.round_count
            ),
            "candidate_compute_budget": config.candidate_compute_budget,
        },
    }
    output = config.run_root / "result" / "result.json"
    _atomic_write(
        output,
        (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        ),
        mode=0o644,
    )
    return payload


def _round_payload(row) -> Mapping[str, object]:
    return {
        "round_index": row.round_index,
        "development_case_count": len(row.development_case_ids),
        "submitted_candidates": [
            _candidate_ref_payload(candidate) for candidate in row.candidate_refs
        ],
        "agent_failures": dict(row.agent_failures),
        "challenger": _candidate_ref_payload(row.challenger),
        "selected": _candidate_ref_payload(row.selected),
        "selection": [
            {
                "candidate": _candidate_ref_payload(finalist.candidate),
                "eligible": finalist.score.eligible,
                "errors": list(finalist.score.errors),
                "aggregate": _aggregate_payload(finalist.aggregate),
                "metric_aggregates": _published_metric_summary_payload(
                    finalist.published_metrics
                ),
            }
            for finalist in row.finalist_evaluations
        ],
        "promotion_evidence": _promotion_payload(
            row.promotion,
            row.promotion_published_metrics,
        ),
        "handoff": {
            "sha256": row.handoff.handoff_sha256,
            "previous_sha256": row.handoff.previous_handoff_sha256,
            "promoted": row.handoff.promoted,
            "decision_reason": row.handoff.decision_reason,
            "evidence_case_count": row.handoff.feedback.evidence_case_count,
            "confidence_group_size": (
                row.handoff.feedback.confidence_group_size
            ),
            "paired_mean_improvement": (
                row.handoff.feedback.paired_mean_improvement
            ),
            "paired_confidence_lower": (
                row.handoff.feedback.paired_confidence_lower
            ),
            "paired_win_rate": row.handoff.feedback.paired_win_rate,
        },
    }


def _promotion_payload(
    promotion,
    published_metrics=(),
) -> Mapping[str, object] | None:
    if promotion is None:
        return None
    if published_metrics and len(published_metrics) != len(promotion.comparisons):
        raise FreshU0RunError("promotion metric summaries do not match comparisons")
    paired_metrics = (
        tuple(published_metrics)
        if published_metrics
        else (None,) * len(promotion.comparisons)
    )
    return {
        "incumbent_candidate_id": promotion.incumbent_artifact_id,
        "incumbent_sha256": promotion.incumbent_digest,
        "selected_candidate_id": promotion.selected_artifact_id,
        "selected_sha256": promotion.selected_digest,
        "promoted": promotion.promoted,
        "reason": promotion.reason,
        "comparisons": [
            _comparison_payload(row, metric_summary)
            for row, metric_summary in zip(
                promotion.comparisons,
                paired_metrics,
            )
        ],
    }


def _terminal_payload(result: MacroLoopResult) -> Mapping[str, object]:
    terminal = result.terminal
    return {
        "cohort_id": terminal.cohort_id,
        "cohort_sha256": terminal.cohort_sha256,
        "case_count": len(terminal.selected_score.errors),
        "cold_start": {
            "candidate": _candidate_ref_payload(result.cold_start),
            "errors": list(terminal.cold_start_score.errors),
            "mean_error": fmean(terminal.cold_start_score.errors),
            "metric_aggregates": _published_metric_summary_payload(
                terminal.cold_start_published_metrics
            ),
            "statuses": [row.status for row in terminal.cold_start.scores],
            "failures": [list(row.failures) for row in terminal.cold_start.scores],
        },
        "selected": {
            "candidate": _candidate_ref_payload(result.selected),
            "errors": list(terminal.selected_score.errors),
            "mean_error": fmean(terminal.selected_score.errors),
            "metric_aggregates": _published_metric_summary_payload(
                terminal.selected_published_metrics
            ),
            "statuses": [row.status for row in terminal.selected.scores],
            "failures": [list(row.failures) for row in terminal.selected.scores],
        },
        "paired_evidence": (
            _comparison_payload(
                terminal.comparison,
                terminal.paired_published_metrics,
            )
            if terminal.comparison is not None
            else None
        ),
        "supports_selected_over_cold_start": (
            terminal.supports_selected_over_cold_start
        ),
    }


def _comparison_payload(
    comparison,
    published_metrics=None,
) -> Mapping[str, object]:
    return {
        "candidate_id": comparison.candidate_artifact_id,
        "candidate_sha256": comparison.candidate_digest,
        "incumbent_id": comparison.incumbent_artifact_id,
        "incumbent_sha256": comparison.incumbent_digest,
        "case_count": comparison.stage.case_count,
        "confidence_group_size": comparison.confidence_group_size,
        "confidence_sample_count": comparison.confidence_sample_count,
        "policy": {
            "confidence_z": comparison.stage.confidence_z,
            "min_mean_improvement": comparison.stage.min_mean_improvement,
            "min_win_rate": comparison.stage.min_win_rate,
            "require_cvar_no_regression": (
                comparison.stage.require_cvar_no_regression
            ),
        },
        "candidate_aggregate": _aggregate_payload(
            comparison.candidate_aggregate
        ),
        "incumbent_aggregate": _aggregate_payload(
            comparison.incumbent_aggregate
        ),
        "mean_improvement": comparison.mean_improvement,
        "confidence_lower": comparison.confidence_lower,
        "confidence_upper": comparison.confidence_upper,
        "win_rate": comparison.win_rate,
        "passed": comparison.passed,
        "failures": list(comparison.failures),
        "paired_metric_aggregates": (
            _paired_published_metric_summary_payload(published_metrics)
        ),
    }


def _aggregate_payload(value) -> Mapping[str, float] | None:
    if value is None:
        return None
    return {
        "mean": float(value.mean),
        "cvar": float(value.cvar),
        "max": float(value.max),
    }


def _published_metric_summary_payload(value) -> Mapping[str, object] | None:
    if value is None:
        return None
    return {
        "case_count": int(value.case_count),
        "metrics": {
            name: _aggregate_payload(aggregate_value)
            for name, aggregate_value in value.aggregates.items()
        },
    }


def _paired_published_metric_summary_payload(
    value,
) -> Mapping[str, object] | None:
    if value is None:
        return None
    return {
        "case_count": int(value.case_count),
        "candidate": _published_metric_summary_payload(value.candidate),
        "incumbent": _published_metric_summary_payload(value.incumbent),
    }


def _candidate_ref_payload(candidate: CandidateRef | None) -> Mapping[str, object] | None:
    if candidate is None:
        return None
    return {
        "candidate_id": candidate.candidate_id,
        "bundle_sha256": candidate.bundle_sha256,
        "round_index": candidate.round_index,
        "agent_id": candidate.agent_id,
        "parent_sha256": candidate.parent_sha256,
    }


def _verified_candidate_bytes(candidate: CandidateRef) -> bytes:
    if candidate.bundle_uri is None:
        raise FreshU0RunError("selected candidate has no sealed bundle URI")
    path = Path(candidate.bundle_uri)
    if path.suffix != ".py":
        raise FreshU0RunError("selected candidate bundle is not one Python file")
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise FreshU0RunError("selected candidate bundle cannot be opened") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise FreshU0RunError("selected candidate bundle is not a regular file")
        if metadata.st_size <= 0 or metadata.st_size > _MAX_CANDIDATE_BYTES:
            raise FreshU0RunError("selected candidate bundle has an invalid size")
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise FreshU0RunError("selected candidate bundle changed while read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise FreshU0RunError("selected candidate bundle grew while read")
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    return payload


def _agent_environment(environment: Mapping[str, str]) -> Mapping[str, str]:
    return {
        name: environment[name]
        for name in _EXPLICIT_AGENT_ENVIRONMENT
        if name in environment
    }


def _evaluated_case_count(config: FreshU0Config) -> int:
    per_round = (
        config.development_case_count * config.development_pool_multiplier
        + config.selection_case_count
        + config.promotion_case_count
    )
    return config.round_count * per_round + config.terminal_case_count


def _relative_improvement(cold_error: float, selected_error: float) -> float | None:
    if cold_error > 0.0:
        return (cold_error - selected_error) / cold_error
    return 0.0 if selected_error == cold_error else None


def _verified_component_sha256(source: object, attribute: str, label: str) -> str:
    value = getattr(source, attribute, None)
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FreshU0RunError(f"verified {label} digest is unavailable")
    return value


def _atomic_write(path: Path, payload: bytes, *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pde_agent_verifier_solver.plugins.vpfp.fresh_u0",
        description=(
            "Reproduce the legacy VPFP fresh-U0 Macro history/control-arm "
            "experiment; canonical product runs use CampaignController."
        ),
    )
    parser.add_argument(
        "--legacy-macro-reference",
        action="store_true",
        help=(
            "Explicitly opt in to the history-only MacroClosedLoopRunner "
            "compatibility path"
        ),
    )
    parser.add_argument("--release-bank-report", type=Path, required=True)
    parser.add_argument("--decoder-checkpoint", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--cold-start-candidate", type=Path, required=True)
    parser.add_argument("--agent-launcher", type=Path, required=True)
    parser.add_argument("--codex-binary", type=Path, required=True)
    parser.add_argument("--auth-file", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reasoning-effort", required=True)
    parser.add_argument(
        "--agent-timeout-seconds",
        type=_positive_float,
        required=True,
    )
    parser.add_argument(
        "--development-case-credits",
        type=_positive_int,
        required=True,
    )
    parser.add_argument("--round-count", type=_positive_int, required=True)
    parser.add_argument(
        "--development-case-count",
        type=_positive_int,
        required=True,
    )
    parser.add_argument(
        "--selection-case-count",
        type=_positive_int,
        required=True,
    )
    parser.add_argument(
        "--promotion-case-count",
        type=_positive_int,
        required=True,
    )
    parser.add_argument(
        "--terminal-case-count",
        type=_positive_int,
        required=True,
    )
    parser.add_argument("--seed-base", type=_nonnegative_int, required=True)
    parser.add_argument("--seed-stride", type=_positive_int, required=True)
    parser.add_argument(
        "--development-pool-multiplier",
        type=_positive_int,
        required=True,
    )
    parser.add_argument(
        "--promotion-confidence-z",
        type=_nonnegative_float,
        required=True,
    )
    parser.add_argument(
        "--promotion-min-mean-improvement",
        type=_nonnegative_float,
        required=True,
    )
    parser.add_argument(
        "--promotion-min-win-rate",
        type=_unit_float,
        required=True,
    )
    parser.add_argument(
        "--promotion-require-cvar-no-regression",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--terminal-confidence-z",
        type=_nonnegative_float,
        required=True,
    )
    parser.add_argument(
        "--terminal-min-win-rate",
        type=_unit_float,
        required=True,
    )
    parser.add_argument(
        "--terminal-require-cvar-no-regression",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--candidate-compute-budget",
        type=_positive_float,
        default=None,
    )
    return parser


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if not arguments.legacy_macro_reference:
        parser.error(
            "this history/control-arm runner is disabled by default; pass "
            "--legacy-macro-reference to opt in explicitly"
        )
    return arguments


def _config_from_args(args: argparse.Namespace) -> FreshU0Config:
    return FreshU0Config(
        release_bank_report=args.release_bank_report,
        decoder_checkpoint=args.decoder_checkpoint,
        run_root=args.run_root,
        cold_start_candidate=args.cold_start_candidate,
        agent_launcher=args.agent_launcher,
        codex_binary=args.codex_binary,
        auth_file=args.auth_file,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        agent_timeout_seconds=args.agent_timeout_seconds,
        development_case_credits=args.development_case_credits,
        round_count=args.round_count,
        development_case_count=args.development_case_count,
        selection_case_count=args.selection_case_count,
        promotion_case_count=args.promotion_case_count,
        terminal_case_count=args.terminal_case_count,
        seed_base=args.seed_base,
        seed_stride=args.seed_stride,
        development_pool_multiplier=args.development_pool_multiplier,
        promotion_confidence_z=args.promotion_confidence_z,
        promotion_min_mean_improvement=args.promotion_min_mean_improvement,
        promotion_min_win_rate=args.promotion_min_win_rate,
        promotion_require_cvar_no_regression=(
            args.promotion_require_cvar_no_regression
        ),
        terminal_confidence_z=args.terminal_confidence_z,
        terminal_min_win_rate=args.terminal_min_win_rate,
        terminal_require_cvar_no_regression=(
            args.terminal_require_cvar_no_regression
        ),
        candidate_compute_budget=args.candidate_compute_budget,
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a nonnegative integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be finite and nonnegative")
    return parsed


def _unit_float(value: str) -> float:
    parsed = float(value)
    if not isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must lie in [0, 1]")
    return parsed


def _require_integer(value: object, label: str, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")


def _require_positive_number(value: object, label: str) -> None:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and positive")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite and positive") from exc
    if not isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{label} must be finite and positive")


def _require_nonnegative_number(value: object, label: str) -> None:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and nonnegative")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite and nonnegative") from exc
    if not isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    config = _config_from_args(arguments)
    payload = run_fresh_u0(
        config,
        legacy_macro_reference=arguments.legacy_macro_reference,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main tests
    raise SystemExit(main())
