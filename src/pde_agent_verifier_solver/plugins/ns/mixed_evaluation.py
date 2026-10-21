"""Optional replay of explicitly frozen NS Official and Terminal B/G/P banks.

The normal Generator still owns all development and Guidance feedback work.
This adapter only replaces declared hidden draws, preserving each original
continuous Case and its public initial-condition/forcing session.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from threading import RLock
from typing import Mapping

from pde_agent_verifier_solver.engine.common_evaluation import content_sha256
from pde_agent_verifier_solver.engine.seeds import SeedLedger
from pde_agent_verifier_solver.sdk import CaseVisibility, EvaluationPhase, GenerationKind, GenerationPurpose
from pde_agent_verifier_solver.sdk.macro_loop import CohortKind
from .case_bank import load_ns_continuous_case_bank
from .continuous_backend import StructuredNSMMSAnswer
from .tasks.solver import NSSolverTask

SCHEMA = "ns_shared_mixed_evaluation_v1"
_STRATA = {"base", "guided", "base_perturbed", "guided_perturbed"}


def _request_identity(request):
    return {
        "purpose": request.purpose.value, "round_index": request.round_index,
        "kind": request.kind.value, "count": request.count, "seed": request.seed,
        "anchors": [case.case_id for case in request.anchors],
    }


def attach_ns_continuous_bank_records(source, records, *, purpose, case_provenance):
    """Register frozen Cases for the existing NS public/feedback resolvers.

    The caller returns the original bank records, whose truth enforces its
    frozen query. Residents retain the original continuous field for public
    initial/forcing queries and permitted next-round Guidance feedback.
    """
    cohort = CohortKind(purpose.value)
    staged = []
    for record in records:
        answer = getattr(record.truth, "_answer", None)
        if not isinstance(answer, StructuredNSMMSAnswer):
            raise TypeError("mixed NS evaluation requires structured continuous truth")
        staged.append(source._resident(
            record.case, answer.raw_answer, answer, cohort,
            perturbation_audit=case_provenance.get(record.case.case_id, {}).get("perturbation_audit", {}),
        ))
    source._commit(staged)


@dataclass
class _Replay:
    label: str
    purpose: GenerationPurpose
    manifest_path: Path
    requests: tuple[dict, ...]
    contexts: Mapping[str, Mapping]
    round_index: int | None = None
    delegate_kinds: tuple[str, ...] = ()
    next_request: int = 0
    bank: object | None = None
    records: dict = field(default_factory=dict)
    sources: dict = field(default_factory=dict)
    children: dict = field(default_factory=dict)
    offsets: dict = field(default_factory=lambda: {"base": 0, "guided": 0})
    active: set = field(default_factory=set)
    issued: set = field(default_factory=set)
    counts: Counter = field(default_factory=Counter)
    strata: dict = field(default_factory=dict)
    case_provenance: dict = field(default_factory=dict)


class SharedNSMixedEvaluationGenerator:
    """Serve the exact bank/request/plan identities chosen by a suite author.

    An omitted Official or Terminal bank delegates that purpose unchanged.
    Evaluation banks load on CPU; stored finite references remain bitwise fixed,
    and arbitrary public queries use the original continuous field adapter.
    """

    def __init__(self, generator, suite_manifest_path, audit_path, master_seed, task_settings):
        self._generator = generator
        self._source = generator.backend.case_source
        self._seed = master_seed
        self._ledger = SeedLedger(master_seed)
        self._settings = dict(task_settings)
        self._evaluation = NSSolverTask().create_evaluation(self._settings)
        self._lock = RLock()
        suite_path = Path(suite_manifest_path).expanduser().resolve()
        suite = json.loads(suite_path.read_text())
        if suite.get("schema") != SCHEMA or suite.get("status") != "frozen":
            raise ValueError("mixed NS evaluation suite is not frozen")
        if suite.get("master_seed") != master_seed:
            raise ValueError("mixed NS suite master_seed differs from the run")
        if suite.get("task_settings_sha256") != content_sha256(self._settings):
            raise ValueError("mixed NS suite Task evaluation settings differ")
        self._official = {}
        for entry in suite.get("official_rounds", ()):
            index = entry["round_index"]
            if isinstance(index, bool) or not isinstance(index, int) or index < 1 or index in self._official:
                raise ValueError("mixed NS suite needs unique positive Official round indices")
            self._official[index] = self._descriptor(
                suite_path, f"official:{index}", GenerationPurpose.PROMOTION,
                entry["manifest"], entry["sampling_requests"], entry["evaluation_context"], index,
                entry.get("delegate_kinds", ()),
            )
        self._terminal = None
        if suite.get("terminal_manifest") is not None:
            self._terminal = self._descriptor(
                suite_path, "terminal", GenerationPurpose.TERMINAL,
                suite["terminal_manifest"], suite["terminal_sampling_requests"],
                suite["terminal_evaluation_context"], None,
            )
        if not self._official and self._terminal is None:
            raise ValueError("mixed NS suite must declare an Official or Terminal bank")
        self._all = (*self._official.values(), *((self._terminal,) if self._terminal else ()))
        self._bank_case_ids = set()
        self._audit_path = Path(audit_path)
        self._audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self._audit_path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "schema": SCHEMA, "event": "suite_bound", "suite_manifest": str(suite_path),
                "suite_sha256": hashlib.sha256(suite_path.read_bytes()).hexdigest(),
                "master_seed": master_seed, "bank_load_device": "cpu",
                "official_rounds": sorted(self._official), "terminal": self._terminal is not None,
            }, sort_keys=True, allow_nan=False) + "\n")

    @staticmethod
    def _descriptor(path, label, purpose, manifest, requests, contexts, round_index, delegate_kinds=()):
        delegate_kinds = tuple(delegate_kinds)
        if delegate_kinds and (purpose is not GenerationPurpose.PROMOTION or delegate_kinds != ('guided',)):
            raise ValueError('only Official Guided requests may be explicitly delegated')
        if not isinstance(requests, list) or not requests or not isinstance(contexts, Mapping):
            raise ValueError("mixed bank needs an explicit sampling schedule and evaluation context")
        identifiers = []
        for request in requests:
            if request.get("purpose") != purpose.value or (round_index is not None and request.get("round_index") != round_index):
                raise ValueError("mixed sampling schedule has the wrong purpose or round")
            if request.get("kind") not in {"base", "guided", "perturbed"}:
                raise ValueError("mixed sampling schedule has an unsupported kind")
            if request['kind'] in delegate_kinds:
                raise ValueError('delegated kinds cannot occur in the frozen sampling schedule')
            ids = request.get("case_ids", [])
            if (not isinstance(ids, list) or not ids or request.get("count") != len(ids)
                    or not isinstance(request.get("anchors"), list)):
                raise ValueError("mixed sampling schedule Case count or anchors are invalid")
            if request["kind"] == "perturbed":
                if len(request["anchors"]) != len(ids):
                    raise ValueError("each frozen perturbed Case needs its exact parent")
            elif request["anchors"]:
                raise ValueError("clean frozen draws cannot have anchors")
            identifiers.extend(ids)
        if len(identifiers) != len(set(identifiers)) or set(identifiers) != set(contexts):
            raise ValueError("mixed sampling schedule and evaluation contexts need identical unique Cases")
        return _Replay(label, purpose, (path.parent / manifest).resolve(), tuple(requests), contexts, round_index, delegate_kinds)

    def export_runtime_state(self, directory):
        from .resume import export_mixed_runtime
        return export_mixed_runtime(self, directory)

    def restore_runtime_state(self, payload, directory):
        from .resume import restore_mixed_runtime
        return restore_mixed_runtime(self, payload, directory)

    def __getattr__(self, name):
        return getattr(self._generator, name)

    def _audit(self, event, **payload):
        with self._audit_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"schema": SCHEMA, "event": event, **payload},
                                    sort_keys=True, allow_nan=False) + "\n")

    def _load(self, replay):
        if replay.bank is not None:
            return
        bank = load_ns_continuous_case_bank(replay.manifest_path, expected_purpose=replay.purpose, device="cpu")
        records = {record.case.case_id: record for record in bank.records}
        if set(records) != set(replay.contexts) or set(records) & self._bank_case_ids:
            raise ValueError("mixed bank Cases differ from the schedule or overlap another bank")
        strata = dict(bank.strata)
        if replay.delegate_kinds and any(kind in {'guided', 'guided_perturbed'} for kind in strata.values()):
            raise ValueError('a delegated Guided suite can freeze only Base and Base-perturbed Cases')
        children = {}
        for key, record in records.items():
            if strata[key] in {"perturbed", "base_perturbed", "guided_perturbed"}:
                parent = record.case.parent_case_id
                if parent not in records or strata[parent] not in {"base", "guided"} or parent in children:
                    raise ValueError("mixed bank needs one frozen child per declared clean parent")
                expected = strata[parent] + "_perturbed"
                if strata[key] != "perturbed" and strata[key] != expected:
                    raise ValueError("mixed bank perturbation stratum differs from its parent")
                strata[key] = expected
                children[parent] = key
            if strata[key] not in _STRATA:
                raise ValueError("mixed bank has an unsupported evaluation stratum")
        phase = EvaluationPhase.OFFICIAL if replay.purpose is GenerationPurpose.PROMOTION else EvaluationPhase.TERMINAL
        namespace = f"official_round:{replay.round_index}" if replay.purpose is GenerationPurpose.PROMOTION else "terminal_evaluation"
        positions = {}
        manifest_rows = {row["case"]["case_id"]: row for row in bank.manifest["cases"]}
        for key, record in records.items():
            context = replay.contexts[key]
            distribution = context["controller_distribution"]
            position = context["case_position"]
            if (not isinstance(distribution, str) or not distribution
                    or isinstance(position, bool) or not isinstance(position, int) or position < 0):
                raise ValueError("mixed bank evaluation context is invalid")
            if (distribution if phase is EvaluationPhase.OFFICIAL else distribution.rsplit("/", 1)[-1]) != strata[key]:
                raise ValueError("mixed bank Controller distribution differs from its frozen stratum")
            seed = self._ledger.derive("evaluation_plan", namespace, phase.value,
                                       "distribution", distribution, "case_position", position)
            actual = self._evaluation.make_plan(record.case, phase, self._settings, seed)
            if content_sha256(actual) != content_sha256(bank.plans[key]):
                raise ValueError("mixed bank frozen query or plan seed differs from the Controller")
            expected_row = manifest_rows[key]
        # Filtering the actual request stream by distribution gives exactly
        # the order assembled by Controller._hidden_suite for that source.
        for request in replay.requests:
            for key in request["case_ids"]:
                context = replay.contexts[key]
                positions.setdefault(context["controller_distribution"], []).append(context["case_position"])
        if any(values != list(range(len(values))) for values in positions.values()):
            raise ValueError("mixed bank Controller positions must follow the request order within each distribution")
        replay.bank, replay.records, replay.children, replay.strata = bank, records, children, strata
        replay.sources = {name: tuple(key for key in records if strata[key] == name) for name in ("base", "guided")}
        replay.case_provenance = {row["case"]["case_id"]: row.get("generation_provenance", {}) for row in bank.manifest["cases"]}
        self._bank_case_ids.update(records)
        self._audit("bank_loaded", bank=replay.label, manifest_sha256=bank.manifest_sha256,
                    counts=dict(Counter(strata.values())))

    def sample(self, state, request, progress=None):
        if request.visibility is not CaseVisibility.HIDDEN:
            return self._generator.sample(state, request, progress)
        if request.purpose is GenerationPurpose.PROMOTION and self._official:
            replay = self._official.get(request.round_index)
            if replay is None:
                raise ValueError("mixed NS suite has no bank for this Official round")
        elif request.purpose is GenerationPurpose.TERMINAL and self._terminal is not None:
            replay = self._terminal
        else:
            return self._generator.sample(state, request, progress)
        if request.kind.value in replay.delegate_kinds:
            return self._generator.sample(state, request, progress)
        with self._lock:
            identity = _request_identity(request)
            attached = ()
            try:
                if replay.next_request >= len(replay.requests):
                    raise ValueError("mixed bank sampling schedule is already complete")
                expected = replay.requests[replay.next_request]
                if identity != {name: expected[name] for name in identity}:
                    raise ValueError("mixed bank sampling seed, count, kind, round or parents changed")
                self._load(replay)
                source = request.kind.value
                if request.kind is GenerationKind.PERTURBED:
                    for anchor in request.anchors:
                        parent = replay.records.get(anchor.case_id)
                        if (parent is None or anchor.case_id not in replay.active
                                or content_sha256(anchor) != content_sha256(parent.case)):
                            raise ValueError("mixed bank perturbation parent is not the active frozen Case")
                    ids = tuple(replay.children[anchor.case_id] for anchor in request.anchors)
                else:
                    start = replay.offsets[source]
                    ids = replay.sources[source][start:start + request.count]
                if list(ids) != expected["case_ids"] or set(ids) & replay.issued:
                    raise ValueError("mixed bank source order or frozen child identity differs")
                records = tuple(replay.records[key] for key in ids)
                if any(row.case.round_index != request.round_index for row in records):
                    raise ValueError("mixed bank Case round differs from the sampling request")
                attach_ns_continuous_bank_records(self._source, records, purpose=replay.purpose,
                                                  case_provenance=replay.case_provenance)
                attached = ids
                counts = replay.counts + Counter(replay.strata[key] for key in ids)
                if progress:
                    progress(f"Loaded {len(records)} frozen NS evaluation Cases", 1.0)
                self._audit("verified", bank=replay.label, **identity, case_ids=list(ids),
                            counts=dict(counts))
                replay.next_request += 1
                replay.issued.update(ids)
                replay.active.update(ids)
                replay.counts = counts
                if request.kind is not GenerationKind.PERTURBED:
                    replay.offsets[source] += len(ids)
                return records
            except BaseException as error:
                if attached:
                    self._generator.release(state, attached)
                self._audit("verification_failed", bank=replay.label, **identity,
                            error_type=type(error).__name__)
                raise

    def release(self, state, case_ids):
        identifiers = tuple(case_ids)
        with self._lock:
            self._generator.release(state, identifiers)
            for replay in self._all:
                replay.active.difference_update(identifiers)
                if replay.bank is not None and not replay.active and replay.next_request == len(replay.requests):
                    # Official fields survive until Guidance seals its feedback;
                    # Terminal fields survive until the final common evaluation.
                    replay.bank = None
                    replay.records.clear()
                    replay.sources.clear()
                    replay.children.clear()
                    replay.case_provenance.clear()
                    self._audit("bank_released", bank=replay.label, counts=dict(replay.counts))

    def assert_complete(self):
        with self._lock:
            missing = [replay.label for replay in self._all if replay.next_request != len(replay.requests)]
            if missing:
                raise ValueError(f"mixed NS evaluation is missing completed banks: {missing}")
            return {replay.label: dict(replay.counts) for replay in self._all}


__all__ = ["SharedNSMixedEvaluationGenerator", "attach_ns_continuous_bank_records"]
