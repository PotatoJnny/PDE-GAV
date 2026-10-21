"""Asynchronous, replayable development evaluation for VPFP solver Agents."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .oracle_training import inspect_solver_bundle, stage_solver_bundle
from .runtime_paths import is_persistent_run_path


VPFP_DEVELOPMENT_INTERFACE_SCHEMA = "vpfp_development_evaluation_interface_v2"
VPFP_DEVELOPMENT_REQUEST_SCHEMA = "vpfp_development_evaluation_request_v2"
VPFP_DEVELOPMENT_RESPONSE_SCHEMA = "vpfp_development_evaluation_response_v2"
VPFP_DEVELOPMENT_RESULT_SCHEMA = "vpfp_development_evaluation_result_v2"
VPFP_DEVELOPMENT_ACCOUNTING_SCHEMA = "vpfp_development_evaluation_accounting_v2"
VPFP_DEVELOPMENT_LEADERBOARD_SCHEMA = "vpfp_development_leaderboard_v2"
VPFP_DEVELOPMENT_CLAIM_SCHEMA = "vpfp_development_evaluation_claim_v2"
VPFP_FINALIST_COMPATIBILITY_SCHEMA = "vpfp_finalist_compatibility_v2"
VPFP_SOLVER_SOURCE_AUDIT_SCHEMA = "vpfp_solver_source_audit_v2"
VPFP_DEVELOPMENT_CASCADE = "fresh_paired_screen_confirm_v2"
DEVELOPMENT_PURPOSE = "development_candidate"
FINALIST_PURPOSE = "finalist_compatibility"


class VPFPRecoverableEvaluationError(RuntimeError):
    """A candidate-specific private cohort could not be constructed."""


@dataclass(frozen=True)
class VPFPDevelopmentEvaluationRequest:
    request_id: str
    agent_id: str
    candidate_entrypoint: str
    requested_evidence_case_count: int
    purpose: str = DEVELOPMENT_PURPOSE
    cascade: str = VPFP_DEVELOPMENT_CASCADE

    def __post_init__(self) -> None:
        if (
            len(self.request_id) != 32
            or any(value not in "0123456789abcdef" for value in self.request_id)
            or not self.agent_id
            or not self.candidate_entrypoint
            or isinstance(self.requested_evidence_case_count, bool)
            or self.requested_evidence_case_count <= 0
            or self.purpose not in {DEVELOPMENT_PURPOSE, FINALIST_PURPOSE}
            or self.cascade != VPFP_DEVELOPMENT_CASCADE
        ):
            raise ValueError("invalid VPFP development evaluation request")


VPFPDevelopmentEvaluationHandler = Callable[
    [VPFPDevelopmentEvaluationRequest, Path, str, int, int], Mapping[str, Any]
]


def audit_vpfp_solver_source(path: str | Path) -> dict[str, Any]:
    """Reject silent broad exception handlers while allowing observable fallback."""

    source = Path(path).resolve()
    if not source.is_file() or source.is_symlink() or source.suffix != ".py":
        raise FileNotFoundError("VPFP solver source audit input is invalid")
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    failures = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        broad = node.type is None or (
            isinstance(node.type, ast.Name)
            and node.type.id in {"Exception", "BaseException"}
        )
        silent = bool(node.body) and all(
            isinstance(statement, ast.Pass) for statement in node.body
        )
        if broad and silent:
            failures.append(
                {"line": int(node.lineno), "reason": "silent_broad_exception_handler"}
            )
    return {
        "schema": VPFP_SOLVER_SOURCE_AUDIT_SCHEMA,
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "broad_exception_fallback_must_be_observable": True,
        "source_path_included": False,
    }


def tail_summary(values: Sequence[float], *, alpha: float = 0.9) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("VPFP tail summary needs finite values")
    tail_count = max(1, int(math.ceil((1.0 - alpha) * array.size)))
    ordered = np.sort(array)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, alpha)),
        "cvar90": float(np.mean(ordered[-tail_count:])),
        "maximum": float(np.max(array)),
        "cvar90_tail_count": tail_count,
    }


def paired_mean_interval(candidate: Sequence[float], comparison: Sequence[float]) -> dict[str, Any]:
    left = np.asarray(candidate, dtype=np.float64)
    right = np.asarray(comparison, dtype=np.float64)
    if (
        left.ndim != 1
        or left.size == 0
        or left.shape != right.shape
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
    ):
        raise ValueError("VPFP paired interval needs aligned finite values")
    differences = left - right
    mean = float(np.mean(differences))
    if differences.size > 1:
        standard_error = float(np.std(differences, ddof=1) / math.sqrt(differences.size))
        radius = 1.96 * standard_error
    else:
        standard_error = None
        radius = 0.0
    return {
        "paired_count": int(differences.size),
        "mean_candidate_minus_comparison": mean,
        "standard_error": standard_error,
        "normal_approximation_95_lower": mean - radius,
        "normal_approximation_95_upper": mean + radius,
        "mean_candidate_to_comparison_ratio": float(
            np.mean(left / np.maximum(right, 1.0e-30))
        ),
        "candidate_better_fraction": float(np.mean(differences < 0.0)),
        "interval_is_for_development_only": True,
        "official_promotion_test": False,
    }


class VPFPDevelopmentEvaluationExchangeController:
    """Freeze and evaluate arbitrary candidates under cumulative case credits."""

    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace).resolve()
        if not is_persistent_run_path(self.workspace):
            raise ValueError("VPFP development evaluator needs configured run storage")
        manifest = _read_json(self.workspace / "workspace_manifest.json")
        interface = manifest.get("development_evaluation_interface")
        if (
            not isinstance(interface, Mapping)
            or interface.get("schema") != VPFP_DEVELOPMENT_INTERFACE_SCHEMA
            or interface.get("cascade") != VPFP_DEVELOPMENT_CASCADE
        ):
            raise ValueError("VPFP development evaluation interface is invalid")
        self.agent_id = str(manifest["agent_id"])
        self.case_credit_limit = int(interface["case_credit_limit"])
        self.evidence_case_counts = tuple(
            int(value) for value in interface["evidence_case_counts"]
        )
        self.default_evidence_case_count = int(
            interface.get("default_evidence_case_count", self.evidence_case_counts[-1])
        )
        self.request_lease_seconds = int(
            interface.get("request_lease_seconds", 2 * 60 * 60)
        )
        if (
            not self.evidence_case_counts
            or tuple(sorted(set(self.evidence_case_counts))) != self.evidence_case_counts
            or self.evidence_case_counts[0] <= 0
            or self.evidence_case_counts[-1] > self.case_credit_limit
            or self.default_evidence_case_count not in self.evidence_case_counts
            or not 60 <= self.request_lease_seconds <= 24 * 60 * 60
        ):
            raise ValueError("VPFP development evaluator credits are invalid")
        exchange = self.workspace / "evaluation_exchange"
        self.requests_root = exchange / "requests"
        self.responses_root = exchange / "responses"
        self.candidates_root = exchange / "candidates"
        self.claims_root = exchange / "claims"
        self.failures_root = exchange / "claim_failures"
        for path in (
            self.requests_root,
            self.responses_root,
            self.candidates_root,
            self.claims_root,
            self.failures_root,
        ):
            path.mkdir(parents=True, exist_ok=True)
        self.worker_id = uuid.uuid4().hex
        self.responses_by_digest: dict[str, dict[str, Any]] = {}
        self.finalist_by_digest: dict[str, dict[str, Any]] = {}
        self.request_count = 0
        self.duplicate_count = 0
        self.rejected_count = 0
        self.charged_case_credits = 0
        self.service_wall_seconds = 0.0
        self._restore()
        self._publish_state()

    @property
    def remaining_case_credits(self) -> int:
        return max(0, self.case_credit_limit - self.charged_case_credits)

    def _restore(self) -> None:
        for path in sorted(self.responses_root.glob("*.json")):
            payload = _read_json(path)
            if payload.get("schema") != VPFP_DEVELOPMENT_RESPONSE_SCHEMA:
                continue
            self.request_count += 1
            accounting = payload.get("accounting", {})
            self.charged_case_credits += int(accounting.get("charged_case_credits", 0))
            self.service_wall_seconds += float(
                accounting.get("evaluator_service_wall_seconds", 0.0)
            )
            if payload.get("status") == "duplicate":
                self.duplicate_count += 1
            elif payload.get("status") in {
                "rejected",
                "evaluation_failed",
            }:
                self.rejected_count += 1
            elif payload.get("status") == "completed":
                digest = str(payload.get("candidate_sha256") or "")
                target = (
                    self.finalist_by_digest
                    if payload.get("purpose") == FINALIST_PURPOSE
                    else self.responses_by_digest
                )
                previous = target.get(digest)
                previous_count = -1 if previous is None else int(previous["result"]["case_count"])
                current_count = int(payload["result"]["case_count"])
                if current_count > previous_count:
                    target[digest] = payload

    def accounting(self) -> dict[str, Any]:
        return {
            "schema": VPFP_DEVELOPMENT_ACCOUNTING_SCHEMA,
            "request_count": self.request_count,
            "distinct_candidate_count": len(self.responses_by_digest),
            "finalist_compatibility_count": len(self.finalist_by_digest),
            "duplicate_candidate_count": self.duplicate_count,
            "rejected_request_count": self.rejected_count,
            "charged_case_credits": self.charged_case_credits,
            "case_credit_limit": self.case_credit_limit,
            "remaining_case_credits": self.remaining_case_credits,
            "submission_count_limit": None,
            "evaluator_service_wall_seconds": self.service_wall_seconds,
            "queue_claim_protocol": "append_only_lease_v1",
            "request_lease_seconds": self.request_lease_seconds,
            "evidence_case_counts": list(self.evidence_case_counts),
            "default_evidence_case_count": self.default_evidence_case_count,
            "worker_id": self.worker_id,
        }

    def process_once(
        self,
        handler: VPFPDevelopmentEvaluationHandler,
        *,
        only_request_id: str | None = None,
    ) -> int:
        paths = (
            [self.requests_root / f"{only_request_id}.json"]
            if only_request_id is not None
            else sorted(self.requests_root.glob("*.json"))
        )
        processed = 0
        for path in paths:
            if not path.is_file() or (self.responses_root / path.name).exists():
                continue
            raw_text = path.read_text(encoding="utf-8")
            raw = json.loads(raw_text)
            request = self._parse_request(path, raw)
            claim = self._claim(path.stem, raw_text)
            if claim is None:
                continue
            response_path = self.responses_root / path.name
            archive = self.candidates_root / request.request_id
            try:
                if archive.exists():
                    staged = archive / "submission.py"
                else:
                    staged = stage_solver_bundle(
                        self._candidate_path(request.candidate_entrypoint), archive
                    )
                bundle = inspect_solver_bundle(staged)
                source_audits = [
                    audit_vpfp_solver_source(archive / str(row["path"]))
                    for row in bundle["files"]
                    if Path(str(row["path"])).suffix == ".py"
                ]
                if not source_audits or not all(
                    audit["passed"] is True for audit in source_audits
                ):
                    raise ValueError("candidate source audit failed")
            except (FileNotFoundError, OSError, SyntaxError, TypeError, ValueError) as exc:
                self.request_count += 1
                self.rejected_count += 1
                self._write_response(
                    response_path,
                    {
                        "schema": VPFP_DEVELOPMENT_RESPONSE_SCHEMA,
                        "request_id": request.request_id,
                        "purpose": request.purpose,
                        "claim_attempt": claim["attempt"],
                        "status": "rejected",
                        "candidate_sha256": None,
                        "candidate_archive": None,
                        "reason": f"candidate_bundle_validation_failed:{type(exc).__name__}",
                        "accounting": self._request_accounting(0, 0.0),
                    },
                )
                self._publish_state()
                processed += 1
                continue
            digest = str(bundle["aggregate_sha256"])
            cache = (
                self.finalist_by_digest
                if request.purpose == FINALIST_PURPOSE
                else self.responses_by_digest
            )
            previous = cache.get(digest)
            previous_count = 0 if previous is None else int(previous["result"]["case_count"])
            if previous is not None and (
                request.purpose == FINALIST_PURPOSE
                or request.requested_evidence_case_count <= previous_count
                or previous["result"].get("cascade", {}).get("next_stage_admissible") is False
            ):
                self.request_count += 1
                self.duplicate_count += 1
                payload = {
                    "schema": VPFP_DEVELOPMENT_RESPONSE_SCHEMA,
                    "request_id": request.request_id,
                    "purpose": request.purpose,
                    "claim_attempt": claim["attempt"],
                    "status": "duplicate",
                    "candidate_sha256": digest,
                    "candidate_archive": archive.relative_to(self.workspace).as_posix(),
                    "duplicate_of_request_id": previous["request_id"],
                    "result": previous["result"],
                    "accounting": self._request_accounting(0, 0.0),
                }
                self._write_response(response_path, payload)
                self._publish_state()
                processed += 1
                continue
            next_count = next(
                (
                    value
                    for value in self.evidence_case_counts
                    if previous_count < value <= request.requested_evidence_case_count
                ),
                None,
            )
            needed = 0 if next_count is None else next_count - previous_count
            if request.purpose == DEVELOPMENT_PURPOSE and (
                next_count is None or needed > self.remaining_case_credits
            ):
                self.request_count += 1
                self.rejected_count += 1
                payload = {
                    "schema": VPFP_DEVELOPMENT_RESPONSE_SCHEMA,
                    "request_id": request.request_id,
                    "purpose": request.purpose,
                    "claim_attempt": claim["attempt"],
                    "status": "rejected",
                    "candidate_sha256": digest,
                    "candidate_archive": archive.relative_to(self.workspace).as_posix(),
                    "reason": "insufficient_case_credits_for_evidence_stage",
                    "accounting": self._request_accounting(0, 0.0),
                }
                self._write_response(response_path, payload)
                self._publish_state()
                processed += 1
                continue
            started = time.monotonic()
            try:
                result = dict(
                    handler(
                        request,
                        staged,
                        digest,
                        self.remaining_case_credits,
                        previous_count,
                    )
                )
                charged = self._validate_result(
                    request, result, digest=digest, previous_count=previous_count
                )
            except VPFPRecoverableEvaluationError as exc:
                elapsed = float(time.monotonic() - started)
                self.request_count += 1
                self.rejected_count += 1
                self.service_wall_seconds += elapsed
                self._record_failure(
                    claim,
                    exc,
                    request_replay_allowed=False,
                )
                self._write_response(
                    response_path,
                    {
                        "schema": VPFP_DEVELOPMENT_RESPONSE_SCHEMA,
                        "request_id": request.request_id,
                        "purpose": request.purpose,
                        "claim_attempt": claim["attempt"],
                        "status": "evaluation_failed",
                        "candidate_sha256": digest,
                        "candidate_archive": archive.relative_to(
                            self.workspace
                        ).as_posix(),
                        "reason": "private_answer_first_cohort_unavailable",
                        "accounting": self._request_accounting(0, elapsed),
                    },
                )
                self._publish_state()
                processed += 1
                continue
            except BaseException as exc:
                self._record_failure(claim, exc)
                raise
            elapsed = float(time.monotonic() - started)
            self.request_count += 1
            self.charged_case_credits += charged
            self.service_wall_seconds += elapsed
            payload = {
                "schema": VPFP_DEVELOPMENT_RESPONSE_SCHEMA,
                "request_id": request.request_id,
                "purpose": request.purpose,
                "claim_attempt": claim["attempt"],
                "status": "completed",
                "candidate_sha256": digest,
                "candidate_archive": archive.relative_to(self.workspace).as_posix(),
                "result": result,
                "accounting": self._request_accounting(charged, elapsed),
            }
            cache[digest] = payload
            self._write_response(response_path, payload)
            self._publish_state()
            processed += 1
        return processed

    def _parse_request(self, path: Path, raw: Mapping[str, Any]) -> VPFPDevelopmentEvaluationRequest:
        if raw.get("schema") != VPFP_DEVELOPMENT_REQUEST_SCHEMA:
            raise ValueError("VPFP development request schema changed")
        request = VPFPDevelopmentEvaluationRequest(
            request_id=str(raw.get("request_id") or ""),
            agent_id=str(raw.get("agent_id") or ""),
            candidate_entrypoint=str(raw.get("candidate_entrypoint") or ""),
            requested_evidence_case_count=int(
                raw.get("requested_evidence_case_count", self.default_evidence_case_count)
            ),
            purpose=str(raw.get("purpose") or DEVELOPMENT_PURPOSE),
            cascade=str(raw.get("cascade") or ""),
        )
        if request.request_id != path.stem or request.agent_id != self.agent_id:
            raise ValueError("VPFP development request identity binding changed")
        if (
            request.purpose == DEVELOPMENT_PURPOSE
            and request.requested_evidence_case_count not in self.evidence_case_counts
        ):
            raise ValueError("VPFP requested evidence count is not registered")
        return request

    def _candidate_path(self, relative: str) -> Path:
        source = (self.workspace / relative).resolve()
        if (
            not source.is_relative_to(self.workspace)
            or not source.is_file()
            or source.is_symlink()
            or source.suffix != ".py"
        ):
            raise ValueError("VPFP candidate entrypoint is invalid")
        forbidden = {
            "cases",
            "released_completed",
            "prior_solvers",
            "generator_exchange",
            "oracle_exchange",
            "evaluation_exchange",
        }
        if source.relative_to(self.workspace).parts[0] in forbidden:
            raise ValueError("VPFP candidate entrypoint is outside writable work")
        return source

    def _validate_result(
        self,
        request: VPFPDevelopmentEvaluationRequest,
        result: Mapping[str, Any],
        *,
        digest: str,
        previous_count: int,
    ) -> int:
        case_count = int(result.get("case_count", -1))
        charged = int(result.get("charged_case_credits", -1))
        forbidden = {
            "case_ids",
            "root_case_ids",
            "rows",
            "solver_evaluation_rows",
            "hidden_answers",
            "generator_parameters",
        }
        invalid = bool(
            result.get("schema") != VPFP_DEVELOPMENT_RESULT_SCHEMA
            or result.get("private_case_ids_included") is not False
            or result.get("per_case_errors_included") is not False
            or result.get("hidden_answers_included") is not False
            or result.get("generator_parameters_included") is not False
            or result.get("official_promotion_result") is not False
            or forbidden.intersection(result)
            or case_count <= 0
        )
        if request.purpose == DEVELOPMENT_PURPOSE:
            invalid = bool(
                invalid
                or case_count not in self.evidence_case_counts
                or case_count > request.requested_evidence_case_count
                or case_count != previous_count + charged
                or charged <= 0
                or charged > self.remaining_case_credits
            )
        else:
            compatibility = result.get("finalist_compatibility")
            invalid = bool(
                invalid
                or charged != 0
                or not isinstance(compatibility, Mapping)
                or compatibility.get("schema") != VPFP_FINALIST_COMPATIBILITY_SCHEMA
                or compatibility.get("candidate_sha256") != digest
                or compatibility.get("exact_final_bundle") is not True
                or not isinstance(compatibility.get("passed"), bool)
            )
        robust = result.get("robust")
        candidate = result.get("candidate")
        paired = result.get("paired")
        cascade = result.get("cascade")
        invalid = bool(
            invalid
            or not isinstance(result.get("safety_passed"), bool)
            or not isinstance(robust, Mapping)
            or not _finite_nonnegative(robust.get("worst_answer_first_p90"))
            or not _finite_nonnegative(robust.get("worst_time_band_p90"))
            or not isinstance(candidate, Mapping)
            or not _finite_nonnegative(candidate.get("high_budget_mean"))
            or not _finite_nonnegative(candidate.get("high_budget_cvar90"))
            or not _finite_nonnegative(candidate.get("frontier_geometric_mean"))
            or not isinstance(paired, Mapping)
            or not _finite(paired.get("high_budget_mean_difference"))
            or not isinstance(cascade, Mapping)
            or not isinstance(cascade.get("next_stage_admissible"), bool)
        )
        if invalid:
            raise ValueError("VPFP development result violates public contract")
        return charged

    def _claim(self, request_id: str, raw_text: str) -> dict[str, Any] | None:
        now = time.time_ns()
        request_sha = hashlib.sha256(raw_text.encode()).hexdigest()
        existing = []
        for path in sorted(self.claims_root.glob(f"{request_id}.*.json")):
            row = _read_json(path)
            if (
                row.get("schema") != VPFP_DEVELOPMENT_CLAIM_SCHEMA
                or row.get("request_id") != request_id
                or row.get("request_sha256") != request_sha
            ):
                raise ValueError("VPFP evaluator claim record changed")
            existing.append(row)
        attempt = 0
        if existing:
            latest = existing[-1]
            attempt = int(latest["attempt"])
            failed = (self.failures_root / f"{request_id}.{attempt:06d}.json").is_file()
            if not failed and int(latest["lease_expires_at_unix_ns"]) > now:
                return None
        payload = {
            "schema": VPFP_DEVELOPMENT_CLAIM_SCHEMA,
            "request_id": request_id,
            "request_sha256": request_sha,
            "attempt": attempt + 1,
            "worker_id": self.worker_id,
            "claimed_at_unix_ns": now,
            "lease_expires_at_unix_ns": now + self.request_lease_seconds * 1_000_000_000,
            "append_only": True,
        }
        path = self.claims_root / f"{request_id}.{attempt + 1:06d}.json"
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return None
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return payload

    def _record_failure(
        self,
        claim: Mapping[str, Any],
        exc: BaseException,
        *,
        request_replay_allowed: bool = True,
    ) -> None:
        path = self.failures_root / f"{claim['request_id']}.{int(claim['attempt']):06d}.json"
        payload = {
            "schema": "vpfp_development_evaluation_claim_failure_v2",
            "request_id": claim["request_id"],
            "attempt": claim["attempt"],
            "worker_id": claim["worker_id"],
            "failed_at_unix_ns": time.time_ns(),
            "exception_type": type(exc).__name__,
            "exception_message": str(exc)[:500],
            "request_replay_allowed": bool(request_replay_allowed),
            "append_only": True,
        }
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def _request_accounting(self, charged: int, elapsed: float) -> dict[str, Any]:
        return {
            "charged_case_credits": int(charged),
            "remaining_case_credits": self.remaining_case_credits,
            "submission_count_limit": None,
            "evaluator_service_wall_seconds": float(elapsed),
        }

    def _write_response(self, path: Path, payload: dict[str, Any]) -> None:
        payload["development_leaderboard"] = (
            "evaluation_exchange/development_leaderboard.json"
        )
        _atomic_json(path, payload)

    def _publish_state(self) -> None:
        _atomic_json(
            self.workspace / "development_evaluation_accounting.json",
            self.accounting(),
        )
        rows = []
        for payload in self.responses_by_digest.values():
            result = payload["result"]
            robust = result.get("robust", {})
            candidate = result.get("candidate", {})
            paired = result.get("paired", {})
            rows.append(
                {
                    "candidate_sha256": payload["candidate_sha256"],
                    "candidate_archive": payload["candidate_archive"],
                    "case_count": result["case_count"],
                    "safety_passed": result.get("safety_passed") is True,
                    "worst_answer_first_p90": robust.get("worst_answer_first_p90"),
                    "worst_time_band_p90": robust.get("worst_time_band_p90"),
                    "high_budget_mean": candidate.get("high_budget_mean"),
                    "high_budget_cvar90": candidate.get("high_budget_cvar90"),
                    "paired_high_budget_mean_difference": paired.get(
                        "high_budget_mean_difference"
                    ),
                    "frontier_geometric_mean": candidate.get(
                        "frontier_geometric_mean"
                    ),
                    "official_promotion_result": False,
                }
            )
        rows.sort(key=_development_ranking_key)
        for index, row in enumerate(rows):
            row["development_rank"] = index + 1
        valid = [row for row in rows if row["safety_passed"]]
        selectors = {
            "best_robust_tail": "worst_answer_first_p90",
            "best_paired_mean": "paired_high_budget_mean_difference",
            "best_high_budget_mean": "high_budget_mean",
            "best_high_budget_cvar90": "high_budget_cvar90",
            "best_compute_frontier": "frontier_geometric_mean",
        }
        representatives = {}
        for name, field in selectors.items():
            predicate = (
                _finite
                if field == "paired_high_budget_mean_difference"
                else _finite_nonnegative
            )
            if valid and all(predicate(row.get(field)) for row in valid):
                representatives[name] = min(
                    valid, key=lambda row: float(row[field])
                )["candidate_sha256"]
        _atomic_json(
            self.workspace / "evaluation_exchange" / "development_leaderboard.json",
            {
                "schema": VPFP_DEVELOPMENT_LEADERBOARD_SCHEMA,
                "ranking_is_development_only": True,
                "official_promotion_requires_fresh_paired_cohort": True,
                "rows": rows,
                "archive_representatives": representatives,
                "accounting": self.accounting(),
            },
        )


def deterministic_vpfp_candidate_evidence_seed(
    *,
    checkpoint_index: int,
    agent_id: str,
    candidate_sha256: str,
    evidence_case_count: int,
) -> int:
    if (
        checkpoint_index < 0
        or not agent_id
        or len(candidate_sha256) != 64
        or any(value not in "0123456789abcdef" for value in candidate_sha256)
        or isinstance(evidence_case_count, bool)
        or evidence_case_count <= 0
    ):
        raise ValueError("invalid VPFP candidate evidence seed inputs")
    payload = (
        f"{checkpoint_index}:{agent_id}:{candidate_sha256}:{evidence_case_count}"
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def _development_ranking_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    if row.get("safety_passed") is not True:
        return (math.inf, math.inf, math.inf, math.inf, str(row.get("candidate_sha256")))
    return (
        float(row["worst_answer_first_p90"]),
        float(row["worst_time_band_p90"]),
        float(row["paired_high_budget_mean_difference"]),
        float(row["high_budget_cvar90"]),
        str(row["candidate_sha256"]),
    )


def _finite_nonnegative(value: Any) -> bool:
    try:
        return bool(math.isfinite(float(value)) and float(value) >= 0.0)
    except (TypeError, ValueError):
        return False


def _finite(value: Any) -> bool:
    try:
        return bool(math.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
