"""Verify shared clean NS Official cases through the existing Generator boundary.

The frozen bank contains identities, not a second sampler or evaluator. Actual
runtime fields and Official queries must match exactly before any scoring.
"""
from __future__ import annotations

from copy import copy
import hashlib
import inspect
import json
from pathlib import Path
from threading import Lock

import numpy as np

from pde_agent_verifier_solver.engine.common_evaluation import content_sha256, json_value
from pde_agent_verifier_solver.engine.seeds import SeedLedger
from pde_agent_verifier_solver.sdk import (
    CaseSource, CaseVisibility, EvaluationPhase, GenerationKind, GenerationPurpose,
)
from .continuous_backend import StructuredNSMMSAnswer
from .tasks.solver import NSSolverTask


SCHEMA = "ns_shared_official_fingerprints_v1"


def _file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _signature(generator, master_seed, task_settings):
    SeedLedger(master_seed)  # Same validation as the Controller.
    if "default_query" not in task_settings:
        raise ValueError("Official audit requires the Controller default_query")
    source = generator.backend.case_source
    return {
        "generator_id": generator.info.program_id,
        "source_generator_id": source.generator_id,
        "source_artifact_sha256": source.artifact_sha256,
        "backend_type": type(generator.backend).__module__ + "." + type(generator.backend).__qualname__,
        "master_seed": master_seed,
        "task_settings_sha256": content_sha256(task_settings),
        "evaluation_program_id": NSSolverTask().create_evaluation(task_settings).program_id,
        "task_source_sha256": _file_digest(inspect.getfile(NSSolverTask)),
        "answer_adapter_source_sha256": _file_digest(inspect.getfile(StructuredNSMMSAnswer)),
    }


def _coefficient_identity(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype="<f8", order="C")
    if not array.size or not np.isfinite(array).all():
        raise ValueError("Official coefficients must be nonempty and finite")
    return {"shape": list(array.shape), "dtype": "<f8",
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest()}


def fingerprint_ns_official_batch(records, request, *, master_seed, task_settings):
    """Fingerprint full structured field definitions and the actual Task query.

    Opaque case IDs are excluded from comparisons. Coefficients plus basis,
    geometry, physical parameters and implementation identify the continuous
    initial condition, forcing and solution beyond the finite scoring points.
    """
    if (request.purpose is not GenerationPurpose.PROMOTION
            or request.kind is not GenerationKind.BASE
            or request.visibility is not CaseVisibility.HIDDEN
            or request.anchors):
        raise ValueError("Official audit supports one clean Base batch per round")
    records = tuple(records)
    if len(records) != request.count or len({r.case.case_id for r in records}) != len(records):
        raise ValueError("Official batch count or identities are invalid")
    ledger = SeedLedger(master_seed)
    evaluation = NSSolverTask().create_evaluation(task_settings)
    rows = []
    implementation_hashes = {}
    for index, record in enumerate(records):
        case = record.case
        if (case.source is not CaseSource.GENERATED or case.parent_case_id is not None
                or case.visibility is not CaseVisibility.HIDDEN
                or case.round_index != request.round_index):
            raise ValueError("Official record must be a hidden clean root in this round")
        answer = getattr(record.truth, "_answer", None)
        if not isinstance(answer, StructuredNSMMSAnswer):
            raise TypeError("Official fingerprint needs structured continuous NS truth")
        field = answer.raw_answer
        if field.geometry_spec is None:
            raise ValueError("Official fingerprint requires continuous polygon geometry")
        field_type = type(field)
        if field_type not in implementation_hashes:
            implementation_hashes[field_type] = _file_digest(inspect.getfile(field_type))
        definition = {
            "implementation": type(field).__module__ + "." + type(field).__qualname__,
            "implementation_source_sha256": implementation_hashes[field_type],
            "continuous_mms_abi": field.continuous_mms_abi,
            "config": field.config.to_dict(),
            "geometry": field.geometry_spec.to_dict(),
            "domain_bounds": list(field.domain_bounds),
            "time_interval": [field.t0, field.t1], "viscosity": field.viscosity,
            "psi": _coefficient_identity(field.coefficients.psi),
            "pressure": _coefficient_identity(field.coefficients.pressure),
        }
        seed = ledger.derive("evaluation_plan", f"official_round:{request.round_index}",
                             "official", "distribution", "base", "case_position", index)
        plan = evaluation.make_plan(case, EvaluationPhase.OFFICIAL, task_settings, seed)
        reference = record.truth.answer(plan.query)
        if set(reference.fields) != set(plan.query.fields):
            raise ValueError("Official reference fields differ from the query")
        for values in reference.fields.values():
            if len(values) != len(plan.query.points) or not np.isfinite(values).all():
                raise ValueError("Official reference must be complete and finite")
        rows.append({
            "position": index, "case_id": case.case_id,
            "physics_sha256": content_sha256(definition),
            "public_data_sha256": content_sha256(case.public_data),
            "plan_seed": seed, "plan": json_value(plan),
            "plan_sha256": content_sha256(plan),
            "reference_sha256": content_sha256(reference), "reference_status": "ok",
        })
    return {
        "round_index": request.round_index, "count": request.count,
        "generation_seed": request.seed, "kind": "base", "purpose": "promotion",
        "cases": rows,
    }


def _comparison_content(batch):
    return {**batch, "cases": [{key: value for key, value in row.items() if key != "case_id"}
                               for row in batch["cases"]]}


class OfficialAuditGenerator:
    """Forward all Generator operations; verify actual Official sampling only."""

    def __init__(self, generator, expected_bank_path, audit_path, master_seed, task_settings):
        self._generator = generator
        self._seed = master_seed
        self._settings = dict(task_settings)
        expected_path = Path(expected_bank_path)
        self._bank_sha256 = _file_digest(expected_path)
        bank = json.loads(expected_path.read_text(encoding="utf-8"))
        if bank.get("schema") != SCHEMA or bank.get("status") != "frozen":
            raise ValueError("Official bank is not a complete frozen NS bank")
        if bank.get("signature") != _signature(generator, master_seed, task_settings):
            raise ValueError("Official bank Generator, evaluation or seed signature differs")
        batches = bank.get("rounds", [])
        indices = [batch["round_index"] for batch in batches]
        if not indices or indices != list(range(1, len(indices) + 1)):
            raise ValueError("Official bank must cover consecutive rounds from one")
        self._expected = {batch["round_index"]: batch for batch in batches}
        self._seen = set()
        self._lock = Lock()
        self._path = Path(audit_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps({"schema": SCHEMA, "event": "bank_bound",
                                     "expected_bank_path": str(expected_path.resolve()),
                                     "expected_bank_sha256": _file_digest(expected_path)},
                                    sort_keys=True, allow_nan=False) + "\n")

    def export_runtime_state(self, directory):
        with self._lock:
            return {"schema": "ns_official_audit_resume_v1",
                    "bank_sha256": self._bank_sha256,
                    "seen": sorted(self._seen),
                    "generator": self._generator.export_runtime_state(directory)}

    def restore_runtime_state(self, payload, directory):
        if (payload.get("schema") != "ns_official_audit_resume_v1"
                or payload.get("bank_sha256") != self._bank_sha256):
            raise ValueError("Official resume bank identity changed")
        seen = payload.get("seen")
        if (not isinstance(seen, list) or any(type(value) is not int for value in seen)
                or len(seen) != len(set(seen)) or not set(seen) <= set(self._expected)):
            raise ValueError("Official resume completed rounds are invalid")
        with self._lock:
            self._generator.restore_runtime_state(payload["generator"], directory)
            self._seen = set(seen)

    def __getattr__(self, name):
        return getattr(self._generator, name)

    def sample(self, state, request, progress=None):
        if request.purpose is not GenerationPurpose.PROMOTION:
            return self._generator.sample(state, request, progress)
        with self._lock:
            records = ()
            batch = None
            try:
                expected = self._expected.get(request.round_index)
                if expected is None or request.round_index in self._seen:
                    raise ValueError("Official round is absent from bank or already sampled")
                records = tuple(self._generator.sample(state, request, progress))
                batch = fingerprint_ns_official_batch(records, request, master_seed=self._seed,
                                                       task_settings=self._settings)
                if _comparison_content(batch) != _comparison_content(expected):
                    raise ValueError("Official physical fields, query or reference differ from frozen bank")
            except BaseException as error:
                with self._path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"schema": SCHEMA, "event": "verification_failed",
                                             "round_index": request.round_index,
                                             "error_type": type(error).__name__, "actual_batch": batch},
                                            allow_nan=False, sort_keys=True) + "\n")
                if records:
                    self._generator.release(state, tuple(r.case.case_id for r in records))
                raise
            self._seen.add(request.round_index)
            with self._path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"schema": SCHEMA, "event": "verified", **batch},
                                        allow_nan=False, sort_keys=True) + "\n")
            return records

    def assert_complete(self):
        """Call after a completed campaign before claiming all rounds were checked."""
        if self._seen != set(self._expected):
            raise ValueError("Official audit is missing completed rounds")


def freeze_ns_official_bank(controller, expected_bank_path):
    """Use a fresh composed Controller to freeze its actual clean Official recipe.

    No Agent, candidate execution, terminal test or Guidance call is made. This
    loads only an explicit frozen checkpoint and invokes the existing Controller
    sampling method on a shallow probe; caller's Controller state is untouched.
    """
    destination = Path(expected_bank_path)
    if destination.exists():
        raise FileExistsError(destination)
    campaign = controller.campaign
    if campaign.generator.checkpoint is None:
        raise ValueError("Official preparation needs an existing frozen checkpoint")
    for plan in campaign.rounds:
        suite = plan.round_hidden
        if not suite.base or suite.case_count != suite.base:
            raise ValueError("Shared NS Official preparation requires only clean Base")
    settings = {**campaign.task_settings, "default_query": campaign.evaluator.default_query}
    signature = _signature(controller.generator, campaign.master_seed, settings)
    probe = copy(controller)
    probe._seeds = SeedLedger(campaign.master_seed)
    probe._case_samplers = []
    probe._generator_operation_lock = Lock()
    batches = []
    generator = controller.generator

    class Recorder:
        def __getattr__(self, name):
            return getattr(generator, name)

        def sample(self, state, request, progress=None):
            records = tuple(generator.sample(state, request, progress))
            try:
                batches.append(fingerprint_ns_official_batch(
                    records, request, master_seed=campaign.master_seed, task_settings=settings))
            except BaseException:
                generator.release(state, tuple(r.case.case_id for r in records))
                raise
            return records

    probe.generator = Recorder()
    generator_settings = {"basis_size": campaign.generator.basis_size,
                          **campaign.generator.settings, "master_seed": campaign.master_seed,
                          "controller_seed": probe._seeds.derive("generator", "base")}
    state = probe.generator.load(campaign.generator.checkpoint, generator_settings,
                                 probe._generator_progress(None, "official_preparation_loading"))
    try:
        for plan in campaign.rounds:
            _hidden, _offset, sampler = probe._hidden_suite(
                state, plan, plan.round_hidden, GenerationPurpose.PROMOTION, 0, generator_settings)
            if sampler is not None:
                sampler.close()
    finally:
        probe._close_case_samplers(probe._case_samplers)
    if len(batches) != len(campaign.rounds):
        raise ValueError("Official preparation did not produce one batch per round")
    bank = {"schema": SCHEMA, "status": "frozen", "signature": signature,
            "rounds": batches, "seed_ledger": dict(probe._seeds.snapshot())}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(json_value(bank), indent=2, allow_nan=False) + "\n")
    return bank


__all__ = ["OfficialAuditGenerator", "fingerprint_ns_official_batch", "freeze_ns_official_bank"]
