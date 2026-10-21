"""Tiny synthetic terminal export/replay tests; no checkpoint/model execution."""
from hashlib import sha256
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import pickle
import torch

from pde_agent_verifier_solver.engine.runtime import EventStream
from pde_agent_verifier_solver.engine.seeds import SeedLedger
from pde_agent_verifier_solver.engine.unified_runtime import CampaignController
from pde_agent_verifier_solver.plugins.vpfp.formal_evaluation_bank import (
    export_terminal_bank, rehydrate_terminal_bank,
)
from pde_agent_verifier_solver.plugins.vpfp.problem import VPFPContinuousMacroTruth
from pde_agent_verifier_solver.plugins.vpfp.tasks.observable import VPFPObservableTask
from pde_agent_verifier_solver.sdk import (
    AnswerQuery, Case, CaseRecord, CaseSource, CaseVisibility, SampleSource,
)


FIELDS = ("rho", "momentum_x", "momentum_y", "kinetic_energy")


class Runtime:
    materialize_cases = False

    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.expanded_model = SimpleNamespace(config=SimpleNamespace(to_dict=lambda: {"test": "exact-expanded"}))
        self._resident_cases = {}
        self.calls = []

    def _register_resident(self, record, latent, condition):
        self._resident_cases[record.case.case_id] = SimpleNamespace(
            case=record.case, latent=latent, equation_condition=condition)

    def _active_resident(self, case):
        return self._resident_cases[case.case_id]

    def _preflight_resident_ids(self, identifiers):
        if set(identifiers).intersection(self._resident_cases):
            raise FileExistsError("resident already exists")

    def release_cases(self, identifiers):
        for identifier in identifiers:
            self._resident_cases.pop(identifier, None)

    def evaluate_macros(self, identifier, points):
        assert points.shape == (256, 3)
        self.calls.append((identifier, points.copy()))
        value = float(self._resident_cases[identifier].latent[0])
        rho = value + points[:, 0]
        return {"rho": rho, "momentum": np.column_stack((rho * 2, rho * 3)),
                "kinetic_energy": rho * 4}


@pytest.fixture
def fixture(tmp_path):
    checkpoint = tmp_path / "decoder.pt"
    checkpoint.write_bytes(b"synthetic decoder identity; not physical validation")
    guide = tmp_path / "guide.pt"
    guide.write_bytes(b"synthetic guide identity")
    runtime = Runtime(checkpoint)
    samplers, all_records, draws = [], [], {}
    events = EventStream()
    for prefix, source in (("common_base", SampleSource.BASE), ("common_guided", SampleSource.GUIDED)):
        records = []
        clean_sources = {}
        # The actual sampler inserts paired roots/children interleaved, then
        # unpaired roots; exporter must reproduce Controller distribution order.
        for suffix, parent in (("r0", None), ("p0", "r0"), ("r1", None), ("p1", "r1"), ("r2", None)):
            identifier = prefix + "-" + suffix
            parent_id = None if parent is None else prefix + "-" + parent
            case = Case(identifier, CaseSource.GENERATED if parent is None else CaseSource.PERTURBED,
                        CaseVisibility.HIDDEN, {"arrays": {name: np.array([0., 1.]) for name in ("t", "x", "y")}},
                        parent_case_id=parent_id, tags=("private",))
            record = CaseRecord(case, VPFPContinuousMacroTruth(identifier, (0., 1.), (0., 1.), (0., 1.), runtime.evaluate_macros))
            runtime._register_resident(record, torch.tensor([len(all_records) + 1., 7., 9.]), np.array([20., 1.]))
            records.append(record)
            all_records.append(record)
            if parent is None:
                clean_sources[identifier] = source
                draws[identifier] = {"source": source.value, "root_seed": len(all_records),
                                     "draw": {"full": torch.tensor([[1., 2.]]), "root_seeds": [len(all_records)]}}
        by_id = {record.case.case_id: record for record in records}
        sampler = SimpleNamespace(closed=False, _seed_namespace=f"terminal:{prefix}",
                                  issued_cases=tuple(record.case for record in records), record_for=by_id.__getitem__,
                                  _clean_source_by_case_id=clean_sources)
        samplers.append(sampler)
        events.emit("terminal_cohort_sealed", "sealed", payload={"cohort_id": prefix})
    query = AnswerQuery.at_points(fields=FIELDS, points=({"t": .5, "x": .5, "y": .5},))
    settings = {"terminal_point_count": 256, "development_point_count": 128, "default_query": query}
    cohorts = tuple(SimpleNamespace(cohort_id=name, suite=SimpleNamespace(case_count=5))
                    for name in ("common_base", "common_guided"))
    campaign = SimpleNamespace(terminal_evaluation=SimpleNamespace(cohorts=cohorts),
                               evaluator=SimpleNamespace(default_query=query), task_settings=settings)
    controller = SimpleNamespace(campaign=campaign, events=events, _terminal_case_samplers=samplers,
                                 _terminal_hidden_case_ids={record.case.case_id for record in all_records},
                                 _development_case_ids=set(), _initial_guidance_hidden_case_ids=set(),
                                 _promotion_hidden_case_ids=set(), _seeds=SeedLedger(101),
                                 evaluation=VPFPObservableTask().create_evaluation(settings))
    controller._hidden_plan_seed_provider = MethodType(CampaignController._hidden_plan_seed_provider, controller)
    controller._terminal_distribution_key = MethodType(CampaignController._terminal_distribution_key, controller)
    audit = {"target_round_index": 1, "source_round_index": 0, "target_solver_sha256": "a" * 64,
             "checkpoint_path": str(guide), "snapshot_sha256": sha256(guide.read_bytes()).hexdigest()}
    backend = SimpleNamespace(runtime=runtime, training_audits=[audit], _draws=draws)
    binding = SimpleNamespace(plugin=SimpleNamespace(generator=SimpleNamespace(backend=backend)),
                              report={"binding_sha256": "b" * 64})
    return controller, binding, runtime, tmp_path / "bank"


def export(fixture):
    controller, binding, _, path = fixture
    return export_terminal_bank(controller, binding, path, initial_solver_sha256="a" * 64)


@pytest.mark.parametrize("kernel", ["basic_vectorized_v1", "structured_batch_v1", "structured_reuse_v1"])
def test_bank_identity_records_opt_in_kernel(fixture, kernel):
    from pde_agent_verifier_solver.plugins.vpfp.formal_evaluation_bank import _runtime_identity
    _, _, runtime, _ = fixture
    before = _runtime_identity(runtime)
    assert "oracle_kernel" not in before  # Default replay identity stays stable.
    runtime.oracle_kernel = kernel
    after = _runtime_identity(runtime)
    assert after["oracle_kernel"] == kernel
    assert "basic_oracle.py" in after["implementation_sha256"]
    assert ("structured_oracle.py" in after["implementation_sha256"]) == kernel.startswith("structured_")


def test_exact_export_rehydrate_and_same_parent_queries(fixture):
    _, _, original, path = fixture
    manifest = export(fixture)
    assert manifest["case_count"] == 10 and manifest["root_count"] == 6
    assert len(original.calls) == 10
    assert manifest["precomputed_source_grid"] is False
    rows = {row["case_id"]: row for row in manifest["cases"]}
    for row in rows.values():
        if row["parent_case_id"]:
            assert row["query_sha256"] == rows[row["parent_case_id"]]["query_sha256"]
            assert row["query_seed"] == rows[row["parent_case_id"]]["query_seed"]
    restored = Runtime(original.checkpoint)
    hidden, queries, loaded = rehydrate_terminal_bank(restored, path,
        expected_manifest_sha256=manifest["manifest_sha256"])
    assert loaded == manifest
    assert not restored.calls  # rehydration itself does not evaluate any grid
    for records in hidden.values():
        for record in records:
            identifier = record.case.case_id
            assert torch.equal(restored._resident_cases[identifier].latent, original._resident_cases[identifier].latent)
            assert isinstance(record.case.public_data["arrays"]["t"], np.ndarray)
            assert record.case.tags == ("private",)
            answer = record.truth.answer(queries[identifier])
            payload = torch.load(path / rows[identifier]["file"], weights_only=True)
            assert dict(answer.fields) == payload["reference"]["fields"]
    with pytest.raises(FileExistsError, match="immutable"):
        export(fixture)


@pytest.mark.parametrize("change,match", [
    ("early", "both terminal"), ("agent", "before the first Agent"),
    ("wrong_s0", "prescribed S0"), ("wrong_points", "256"),
    ("overlap", "overlaps"), ("missing_draw", "captured draw"),
])
def test_export_fails_closed(fixture, change, match):
    controller, binding, _, path = fixture
    if change == "early":
        controller.events._events.pop()
    elif change == "agent":
        controller.events.emit("agent_started", "already too late")
    elif change == "wrong_s0":
        binding.plugin.generator.backend.training_audits[0]["target_solver_sha256"] = "c" * 64
    elif change == "wrong_points":
        controller.campaign.task_settings["terminal_point_count"] = 128
    elif change == "overlap":
        controller._development_case_ids.add("common_base-r0")
    elif change == "missing_draw":
        binding.plugin.generator.backend._draws.pop("common_base-r0")
    with pytest.raises((ValueError, RuntimeError), match=match):
        export(fixture)
    assert not (path / "manifest.json").exists()


def test_restore_requires_external_manifest_digest_and_valid_case_hash(fixture):
    _, _, original, path = fixture
    manifest = export(fixture)
    rehydrate_terminal_bank(Runtime(original.checkpoint), path, expected_manifest_sha256="0" * 64)
    case_path = path / manifest["cases"][0]["file"]
    case_path.chmod(0o600)
    case_path.write_bytes(b"corrupted")
    restored = Runtime(original.checkpoint)
    with pytest.raises(pickle.UnpicklingError):
        rehydrate_terminal_bank(restored, path, expected_manifest_sha256=manifest["manifest_sha256"])
    assert not restored._resident_cases


def test_restore_refuses_decoder_change_or_resident_collision(fixture):
    _, _, original, path = fixture
    manifest = export(fixture)
    with pytest.raises(FileExistsError, match="resident"):
        rehydrate_terminal_bank(original, path, expected_manifest_sha256=manifest["manifest_sha256"])
    original.checkpoint.write_bytes(b"changed decoder")
    with pytest.raises(ValueError, match="decoder differs"):
        rehydrate_terminal_bank(Runtime(original.checkpoint), path, expected_manifest_sha256=manifest["manifest_sha256"])
