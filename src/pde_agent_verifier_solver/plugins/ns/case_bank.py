"""Frozen continuous NS evaluation cases, retaining the original public oracle ABI.

These Controller-private banks hold actual B/G/P fields, never a discretized D0
substitute. Only the public Case and initial/forcing session reach a Candidate.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from pde_agent_verifier_solver.engine.common_evaluation import content_sha256, json_value
from pde_agent_verifier_solver.sdk import (
    AnswerQuery, AnswerView, Case, CaseAccessKind, CaseRecord, CaseSource, CaseVisibility,
    EvaluationPhase, EvaluationPlan, GenerationPurpose, GeneratorCheckpoint, GeneratorState,
    QueryKind,
)
from .continuous_backend import StructuredNSMMSAnswer, _NSPublicSession

SCHEMA = "ns_continuous_case_bank_v1"
STRATA = frozenset(("base", "guided", "perturbed", "base_perturbed", "guided_perturbed"))


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _purpose(value):
    purpose = GenerationPurpose(value)
    if purpose not in (GenerationPurpose.PROMOTION, GenerationPurpose.TERMINAL):
        raise ValueError("continuous evaluation bank purpose must be promotion or terminal")
    return purpose


def _query(node):
    return AnswerQuery(QueryKind(node["kind"]), tuple(node["fields"]),
                       dict(node.get("resolution", {})), tuple(node.get("points", ())))


def _plan(node):
    settings = dict(node.get("settings", {}))
    if isinstance(settings.get("default_query"), Mapping):
        settings["default_query"] = _query(settings["default_query"])
    return EvaluationPlan(EvaluationPhase(node["phase"]), _query(node["query"]),
                          seed=node["seed"], settings=settings)


def _case(node):
    return Case(node["case_id"], CaseSource(node["source"]), CaseVisibility(node["visibility"]),
                node["public_data"], round_index=node.get("round_index"),
                parent_case_id=node.get("parent_case_id"), tags=tuple(node.get("tags", ())))


def _reference(node):
    return AnswerView(QueryKind(node["kind"]),
                      {key: tuple(values) for key, values in node["fields"].items()},
                      {key: tuple(values) for key, values in node.get("axes", {}).items()},
                      tuple(node.get("points", ())), tuple(node.get("shape", ())))


def _validate(case, plan, stratum, purpose, reference=None):
    expected_phase = (EvaluationPhase.OFFICIAL if purpose is GenerationPurpose.PROMOTION
                      else EvaluationPhase.TERMINAL)
    if (not case.case_id or case.visibility is not CaseVisibility.HIDDEN
            or stratum not in STRATA or "arrays" in case.public_data):
        raise ValueError("continuous bank requires hidden B/G/P Cases with the continuous public ABI")
    perturbed = stratum in {"perturbed", "base_perturbed", "guided_perturbed"}
    if ((case.source is CaseSource.PERTURBED) != perturbed
            or (perturbed and not case.parent_case_id)
            or (not perturbed and (case.source is not CaseSource.GENERATED or case.parent_case_id))):
        raise ValueError("continuous Case source/parent differs from its declared stratum")
    if (plan.phase is not expected_phase or plan.query.kind is not QueryKind.POINTS
            or set(plan.query.fields) != {"velocity_x", "velocity_y"}):
        raise ValueError("bank plan phase/fields differ from its declared purpose")
    if reference is not None:
        if (reference.kind is not plan.query.kind
                or set(reference.fields) != set(plan.query.fields)
                or json_value(reference.points) != json_value(plan.query.points)
                or reference.shape != (len(plan.query.points),)
                or any(len(values) != len(plan.query.points) or not np.isfinite(values).all()
                       for values in reference.fields.values())):
            raise ValueError("bank reference must be complete, finite and match the exact query")


def write_ns_continuous_case_bank(
    manifest_path, *, bank_id: str, purpose, records: Sequence[CaseRecord],
    plans: Mapping[str, EvaluationPlan], strata: Mapping[str, str],
    provenance: Mapping[str, object], case_provenance: Mapping[str, Mapping] | None = None,
) -> Path:
    """Freeze actual fields and fixed finite views; no sampling or Agent call.

    Each row retains its original round index. Use separate manifests/purposes
    for promotion rounds and independent terminal cases. Counts, anchors and
    Guidance identities belong to the caller's explicit provenance.
    """
    from agent_verifier.neural_generator.structured_ns_answer_first import (
        write_structured_continuous_case_state,
    )

    path = Path(manifest_path).expanduser().resolve()
    purpose = _purpose(purpose)
    records = tuple(records)
    ids = tuple(row.case.case_id for row in records)
    if not isinstance(bank_id, str) or not bank_id.strip() or not records:
        raise ValueError("continuous bank needs a nonempty id and Cases")
    if len(set(ids)) != len(ids) or set(ids) != set(plans) or set(ids) != set(strata):
        raise ValueError("continuous bank records/plans/strata must have identical unique Case ids")
    if not isinstance(provenance, Mapping):
        raise TypeError("continuous bank provenance must be a mapping")
    if case_provenance is not None and set(case_provenance) - set(ids):
        raise ValueError("case provenance contains unknown Case ids")
    for record in records:
        _validate(record.case, plans[record.case.case_id], strata[record.case.case_id], purpose)
        if not isinstance(getattr(record.truth, "_answer", None), StructuredNSMMSAnswer):
            raise TypeError("bank writer requires original structured continuous NS truth")
    assets = path.with_suffix(".cases")
    if path.exists() or assets.exists():
        raise FileExistsError(path if path.exists() else assets)
    path.parent.mkdir(parents=True, exist_ok=True)
    assets.mkdir(mode=0o700)
    entries = []
    for index, record in enumerate(records):
        case, plan = record.case, plans[record.case.case_id]
        answer = record.truth._answer
        reference = record.truth.answer(plan.query)
        _validate(case, plan, strata[case.case_id], purpose, reference)
        directory = assets / f"{index:06d}"
        state_path = write_structured_continuous_case_state(directory, field=answer.raw_answer)
        for saved in directory.iterdir():
            saved.chmod(0o600)
        entries.append({
            "case": json_value(case), "stratum": strata[case.case_id],
            "state_path": str(state_path.relative_to(path.parent)),
            "state_sha256": _digest(state_path), "plan": json_value(plan),
            "plan_sha256": content_sha256(plan), "reference": json_value(reference),
            "reference_sha256": content_sha256(reference),
            "public_case_sha256": content_sha256(case),
            "generation_provenance": json_value((case_provenance or {}).get(case.case_id, {})),
        })
    payload = {
        "schema": SCHEMA, "status": "frozen", "bank_id": bank_id,
        "purpose": purpose.value, "case_count": len(entries),
        "strata_counts": dict(Counter(strata.values())), "provenance": json_value(provenance),
        "public_input_contract": "continuous_initial_velocity_and_forcing_oracles",
        "reference_policy": "stored_exact_view_of_original_continuous_truth_at_frozen_query",
        "cases": entries,
    }
    payload["content_sha256"] = content_sha256(payload)
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return path


@dataclass(frozen=True)
class _BankTruth:
    plan: EvaluationPlan
    reference: AnswerView
    _answer: StructuredNSMMSAnswer

    def answer(self, query):
        if json_value(query) != json_value(self.plan.query):
            raise ValueError("continuous bank truth may only answer its frozen evaluation query")
        return self.reference


class FixedNSBankEvaluation:
    def __init__(self, evaluation, plans):
        self._evaluation, self._plans = evaluation, plans

    def __getattr__(self, name):
        return getattr(self._evaluation, name)

    def make_plan(self, case, phase, settings, seed, request=None):
        del settings, seed
        plan = self._plans[case.case_id]
        if phase is not plan.phase or request is not None:
            raise ValueError("continuous bank evaluation cannot override its purpose or fixed query")
        return plan


class NSBankSessionProvider:
    """Only public problem sessions; no sampling or completed-answer capability."""
    def __init__(self, banks):
        self._cases, self._answers = {}, {}
        identities = []
        for bank in banks:
            identities.append(bank.manifest_sha256)
            for record in bank.records:
                key = record.case.case_id
                if key in self._cases:
                    raise ValueError("continuous evaluation banks contain duplicate Case ids")
                self._cases[key] = record.case
                self._answers[key] = bank._answers[key]
        identity = content_sha256(identities)
        checkpoint = GeneratorCheckpoint("ns-frozen-evaluation-bank", identity)
        self.state = GeneratorState(checkpoint, base=identity)

    def case_session_methods(self, access):
        if access is not CaseAccessKind.PUBLIC_PROBLEM:
            return ()
        methods = ("query_initial_velocity", "query_forcing")
        if any(case.public_data.get("boundary_condition", {}).get("kind") ==
               "manufactured_velocity_trace" for case in self._cases.values()):
            methods += ("query_boundary_velocity",)
        return methods

    def open_case_session(self, state, case, access):
        if state != self.state or access is not CaseAccessKind.PUBLIC_PROBLEM:
            raise PermissionError("continuous bank exposes only its public problem session")
        expected = self._cases.get(case.case_id)
        if expected is None or json_value(expected) != json_value(case):
            raise ValueError("continuous bank public Case identity changed")
        answer = self._answers[case.case_id]
        limits = case.public_data.get("oracle_protocol", {}).get("limits", {})
        return _NSPublicSession(case, answer.evaluate_forcing, answer.evaluate_velocity,
                                answer.time_interval[0], answer.domain_bounds, answer.time_interval,
                                int(limits.get("max_points_per_request", 4096)),
                                boundary_geometry=(answer.public_geometry if
                                    case.public_data.get("boundary_condition", {}).get("kind") ==
                                    "manufactured_velocity_trace" else None))


@dataclass
class NSContinuousCaseBank:
    manifest_path: Path
    bank_id: str
    purpose: GenerationPurpose
    records: tuple[CaseRecord, ...]
    plans: Mapping[str, EvaluationPlan]
    strata: Mapping[str, str]
    provenance: Mapping[str, object]
    manifest_sha256: str
    _answers: Mapping[str, StructuredNSMMSAnswer]
    manifest: Mapping[str, object]

    def __post_init__(self):
        self.generator = NSBankSessionProvider((self,))
        self.generator_state = self.generator.state

    def fixed_evaluation(self, evaluation):
        return FixedNSBankEvaluation(evaluation, self.plans)


def load_ns_continuous_case_bank(
    manifest_path, *, expected_purpose, device: str = "cpu", case_ids=None,
) -> NSContinuousCaseBank:
    """Load a frozen bank without Flow/Guidance training or GPU allocation.

    Stored references are replayed exactly, avoiding device-dependent changes
    to evaluation targets. Original continuous fields still answer arbitrary
    public initial-condition and forcing queries through the same NS adapter.
    """
    from agent_verifier.neural_generator.structured_ns_answer_first import (
        load_structured_continuous_case,
    )

    path = Path(manifest_path).expanduser().resolve()
    payload = json.loads(path.read_text())
    if payload.get("schema") != SCHEMA or payload.get("status") != "frozen":
        raise ValueError("not a complete frozen continuous NS case bank")
    purpose = _purpose(payload["purpose"])
    if purpose is not _purpose(expected_purpose):
        raise ValueError("continuous bank purpose differs from the requested evaluation")
    rows = payload.get("cases", ())
    if not rows or len(rows) != payload.get("case_count"):
        raise ValueError("continuous bank has missing or empty Case rows")
    # Check the complete manifest before selecting a shard. Only selected
    # field assets need to be loaded; the full manifest still supplies cohort IDs.
    indexed = {row["case"]["case_id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("continuous bank contains duplicate Case ids")
    if dict(Counter(row["stratum"] for row in rows)) != payload.get("strata_counts"):
        raise ValueError("continuous bank stratum counts changed")
    if case_ids is not None:
        if (isinstance(case_ids, (str, bytes)) or not case_ids
                or len(set(case_ids)) != len(case_ids) or set(case_ids) - set(indexed)):
            raise ValueError("continuous bank case_ids must be a nonempty unique subset")
        rows = [indexed[key] for key in case_ids]
    records, plans, strata, answers = [], {}, {}, {}
    for row in rows:
        case, plan, reference = _case(row["case"]), _plan(row["plan"]), _reference(row["reference"])
        _validate(case, plan, row["stratum"], purpose, reference)
        key = case.case_id
        if key in plans:
            raise ValueError("continuous bank contains duplicate Case ids")
        state_path = (path.parent / row["state_path"]).resolve()
        if not state_path.is_relative_to(path.parent) or state_path.name != "hidden_continuous_case.pt":
            raise ValueError("continuous Case state must remain inside its bank directory")
        field = load_structured_continuous_case(state_path.parent, device=device)
        answer = StructuredNSMMSAnswer(field)
        domain = case.public_data.get("domain", {})
        bounds = domain.get("bounds", {})
        if (list(answer.time_interval) != list(domain.get("time_interval", ()))
                or list(answer.domain_bounds) != [*bounds.get("x", ()), *bounds.get("y", ())]
                or answer.viscosity != case.public_data.get("parameters", {}).get("viscosity")
                or json_value(answer.public_geometry) != json_value(domain.get("geometry"))):
            raise ValueError("continuous bank public problem and hidden field disagree")
        records.append(CaseRecord(case, _BankTruth(plan, reference, answer)))
        plans[key], strata[key], answers[key] = plan, row["stratum"], answer
    return NSContinuousCaseBank(
        path, payload["bank_id"], purpose, tuple(records), MappingProxyType(plans),
        MappingProxyType(strata), MappingProxyType(payload.get("provenance", {})), _digest(path),
        MappingProxyType(answers), MappingProxyType(payload),
    )
