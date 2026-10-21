"""Mock-only canonical VPFP round integration; no physical assets or Solver."""
from copy import deepcopy
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch

from pde_agent_verifier_solver.plugins.vpfp import iteration_backend as integration
from pde_agent_verifier_solver.plugins.vpfp import perturbation_backend as physical
from pde_agent_verifier_solver.plugins.vpfp.round_solver_guidance import RoundGuidanceConfig
from pde_agent_verifier_solver.sdk import (Artifact, Case, CaseFeedback, CaseRecord, CaseSource,
    CaseVisibility, GenerationKind, GenerationPurpose, GenerationRequest, GeneratorMode,
    RoundFeedback, TaskKind)


def _request(kind=GenerationKind.BASE, *, count=2, seed=41, round_index=3, anchors=()):
    return GenerationRequest(round_index=round_index, count=count, mode=GeneratorMode.ONLINE,
        kind=kind, visibility=CaseVisibility.HIDDEN, seed=seed,
        purpose=GenerationPurpose.SELECTION, anchors=anchors)


def _record(identifier, request, *, source=CaseSource.GENERATED, parent=None):
    return CaseRecord(Case(case_id=identifier, source=source, visibility=request.visibility,
        public_data={"question": {"case_id": identifier}}, round_index=request.round_index,
        parent_case_id=parent), object())


def _draw(seeds, offset=0.):
    n = len(seeds)
    full = torch.tensor([[float(seed % 71) + offset, .25] for seed in seeds])
    return {"full": full, "future_white": full.clone(), "initial_context": torch.ones(n, 2),
        "equation_conditions": torch.tensor([[20., 1.]]).repeat(n, 1),
        "root_seeds": list(seeds), "known_prior_seeds": [int(seed) + 101 for seed in seeds],
        "future_noise_seeds": [int(seed) + 202 for seed in seeds]}


class _Sampler:
    def __init__(self, base):
        self.base = base
        self.last_report = {"nonzero_gradient_calls": 1}
        self.calls, self.freeze_checks = [], 0
        self.frozen = True

    def assert_frozen(self):
        self.freeze_checks += 1
        if not self.frozen:
            raise ValueError("frozen guide/config/identity changed")

    def draw(self, seeds):
        self.assert_frozen()
        self.calls.append(tuple(seeds))
        return _draw(seeds, offset=10.)


def _snapshot(base, *, target_round=3, tag="c", config=None):
    return SimpleNamespace(guide=object(), sampler=_Sampler(base), checkpoint_path=f"/private/{tag}.pt",
        checkpoint_sha256=tag * 64, manifest_sha256="d" * 64,
        report={"source_round_index": target_round - 1, "target_round_index": target_round,
                "target_solver_sha256": "a" * 64, "execution_passed": True,
                "config": asdict(config or RoundGuidanceConfig()), "diagnostic_only": False})


@pytest.fixture
def backend(monkeypatch, tmp_path):
    base = SimpleNamespace(identity={"checkpoint_sha256": "e" * 64})
    capture = integration._CapturedSampler(base)
    native_calls, serialized, trained = [], [], []
    monkeypatch.setattr(physical, "_seeds", lambda request: tuple(request.seed + i for i in range(request.count)))
    def native(sampler, seeds):
        assert sampler is base
        native_calls.append(tuple(seeds))
        return _draw(seeds)
    monkeypatch.setattr(integration, "draw_frozen_ic40", native)
    def serialize(draw, *, guidance_report=None):
        serialized.append({"draw": draw, "guidance": guidance_report})
        return {"latents": draw["full"], "equation_seeds": tuple(draw["root_seeds"])}
    monkeypatch.setattr(integration, "native_batch_from_draw", serialize)

    class Inner:
        runtime = object()
        public_session_methods = ("evaluate_source",)
        development_solution_session_methods = ("evaluate_macros",)
        def __init__(self):
            self.calls, self.released, self.anchors = [], [], {}
            self.fail_after_capture = False
            self.return_short_batch = False
        def generate(self, base_state, guidance_state, request, progress=None):
            self.calls.append((base_state, guidance_state, request))
            assert guidance_state is None
            if request.kind is GenerationKind.PERTURBED:
                return tuple(_record("child-" + anchor.case_id, request, source=CaseSource.PERTURBED,
                    parent=anchor.case_id) for anchor in request.anchors)
            native_batch = capture.generate(physical._seeds(request))
            if self.fail_after_capture:
                raise RuntimeError("mock resident registration failed")
            rows = tuple(_record(f"native-{seed}", request) for seed in native_batch["equation_seeds"])
            for row in rows:
                self.anchors[row.case.case_id] = row
            return rows[:-1] if self.return_short_batch else rows
        def release_cases(self, identifiers):
            self.released.extend(identifiers)
            for identifier in identifiers:
                self.anchors.pop(identifier, None)
        def public_session_factory(self, case):
            return ("public", case.case_id)
        def development_solution_session_factory(self, case):
            return ("solution", case.case_id)
    inner = Inner()
    config = RoundGuidanceConfig()
    def trainer(**kwargs):
        trained.append(kwargs)
        return _snapshot(base, target_round=kwargs["target_round_index"], tag="f", config=config)
    wrapper = integration.VPFPTrainingGuidanceBackend(inner, capture,
        output_root=tmp_path / "guidance", config=config, trainer=trainer)
    return SimpleNamespace(wrapper=wrapper, base=base, capture=capture, inner=inner,
        native_calls=native_calls, serialized=serialized, trained=trained, config=config)


def _feedback(records, *, round_index=2, solver_sha="a" * 64):
    artifact = Artifact(artifact_id="selected-solver", kind=TaskKind.OBSERVABLE_MODEL,
        name="frozen Solver", program=SimpleNamespace(tree_sha256=solver_sha), summary="mock",
        usage="mock", round_index=round_index, agent_id="agent")
    return RoundFeedback(round_index, artifact,
        tuple(CaseFeedback(row.case, .02 + i * .001, {}) for i, row in enumerate(records)))


def test_guided_requires_round_snapshot_before_any_native_draw(backend):
    with pytest.raises(ValueError, match="snapshot"):
        backend.wrapper.generate({}, None, _request(GenerationKind.GUIDED))
    assert not backend.native_calls and not backend.inner.calls


def test_feedback_uses_exact_actual_draw_without_sampling_again(backend):
    records = backend.wrapper.generate({}, None, _request(round_index=2))
    assert len(backend.native_calls) == 1 and len(backend.serialized) == 1
    for index, record in enumerate(records):
        cached = backend.wrapper._draws[record.case.case_id]
        assert torch.equal(cached["draw"]["full"], backend.serialized[0]["draw"]["full"][index:index+1])
    trained = backend.wrapper.fit_guidance({}, None, _feedback(records), {"controller_seed": 99})
    assert len(backend.native_calls) == 1 and len(backend.trained) == 1
    rows = backend.trained[0]["rows"]
    assert [r["case_id"] for r in rows] == [record.case.case_id for record in records]
    assert [r["root_seed"] for r in rows] == [41, 42]
    assert all(row["source"] == "base" and row["target_solver_sha256"] == "a" * 64 for row in rows)
    assert trained.report["target_round_index"] == 3
    assert not backend.capture.pending and backend.capture.guide is None


def test_guided_draws_keep_same_inner_records_and_cannot_train_base_objective(backend):
    base_records = backend.wrapper.generate({}, None, _request(seed=11, round_index=2))
    snapshot = _snapshot(backend.base, target_round=2)
    guided_records = backend.wrapper.generate({}, snapshot, _request(GenerationKind.GUIDED, seed=22, round_index=2))
    assert len(backend.native_calls) == 1 and len(snapshot.sampler.calls) == 1
    assert all(record is backend.inner.anchors[record.case.case_id] for record in guided_records)
    assert all(backend.wrapper._draws[record.case.case_id]["source"] == "guided" for record in guided_records)
    backend.wrapper.fit_guidance({}, snapshot, _feedback(base_records + guided_records), {"controller_seed": 99})
    assert len(backend.trained[0]["rows"]) == len(base_records)
    assert len(backend.wrapper.training_audits[0]["excluded"]) == len(guided_records)


def test_release_removes_private_training_coordinates_and_delegates_oracles(backend):
    records = backend.wrapper.generate({}, None, _request())
    identifiers = tuple(row.case.case_id for row in records)
    assert backend.wrapper.public_session_factory(records[0].case) == ("public", identifiers[0])
    assert backend.wrapper.development_solution_session_factory(records[0].case) == ("solution", identifiers[0])
    backend.wrapper.release_cases(identifiers)
    assert not backend.wrapper._draws and backend.inner.released == list(identifiers)
    with pytest.raises(ValueError, match="still-resident"):
        backend.wrapper.fit_guidance({}, None, _feedback(records), {"controller_seed": 99})


def test_perturbation_preserves_inner_anchor_and_is_excluded_from_fit(backend):
    records = backend.wrapper.generate({}, None, _request(round_index=2))
    anchors = tuple(record.case for record in records)
    request = _request(GenerationKind.PERTURBED, round_index=2, anchors=anchors)
    children = backend.wrapper.generate({}, None, request)
    assert backend.inner.calls[-1][2] is request
    assert all(child.case.parent_case_id == anchor.case_id for child, anchor in zip(children, anchors))
    assert len(backend.native_calls) == 1
    backend.wrapper.fit_guidance({}, None, _feedback(records + children), {"controller_seed": 99})
    assert len(backend.trained[0]["rows"]) == len(records)
    assert all(row["reason"] == "not_clean_base" for row in backend.wrapper.training_audits[-1]["excluded"])


def test_partial_capture_failure_restores_transaction_for_next_request(backend):
    backend.inner.fail_after_capture = True
    with pytest.raises(RuntimeError, match="registration failed"):
        backend.wrapper.generate({}, _snapshot(backend.base), _request(GenerationKind.GUIDED))
    assert not backend.capture.pending and backend.capture.guide is None and not backend.capture.diagnostics
    assert not backend.wrapper._draws
    backend.inner.fail_after_capture = False
    records = backend.wrapper.generate({}, None, _request(seed=82))
    assert len(records) == 2 and len(backend.native_calls) == 1


def test_incomplete_return_batch_releases_inserted_draws_and_clears_capture(backend):
    backend.inner.return_short_batch = True
    with pytest.raises(ValueError):
        backend.wrapper.generate({}, None, _request())
    assert not backend.wrapper._draws and not backend.capture.pending and backend.capture.guide is None
    assert backend.inner.released == ["native-41"]


def test_previous_guide_object_cannot_be_returned_as_fresh_training(backend):
    records = backend.wrapper.generate({}, None, _request(round_index=2))
    previous = _snapshot(backend.base)
    backend.wrapper.trainer = lambda **kwargs: previous
    with pytest.raises(RuntimeError, match="reused previous"):
        backend.wrapper.fit_guidance({}, previous, _feedback(records), {"controller_seed": 99})
    assert not backend.wrapper.training_audits


def test_invalid_labels_are_not_imputed_as_large_regression_rewards(backend):
    records = backend.wrapper.generate({}, None, _request(round_index=2))
    feedback = _feedback(records)
    feedback = replace(feedback, cases=(replace(feedback.cases[0], primary_error=float("inf")), feedback.cases[1]))
    backend.wrapper.fit_guidance({}, None, feedback, {"controller_seed": 99})
    rows = backend.trained[0]["rows"]
    assert rows[0]["valid"] is False and rows[0]["error"] is None
    assert rows[1]["valid"] is True and rows[1]["error"] == feedback.cases[1].primary_error


def test_guide_model_or_runtime_mutation_is_rejected(backend):
    snapshot = _snapshot(backend.base)
    snapshot.sampler.frozen = False
    with pytest.raises(ValueError, match="frozen"):
        backend.wrapper.generate({}, snapshot, _request(GenerationKind.GUIDED))
    assert not backend.capture.pending and backend.capture.guide is None


def test_new_training_must_not_mutate_previous_frozen_guide(backend):
    records = backend.wrapper.generate({}, None, _request(round_index=2))
    previous = _snapshot(backend.base)
    def malicious_trainer(**kwargs):
        previous.sampler.frozen = False
        return _snapshot(backend.base, tag="f")
    backend.wrapper.trainer = malicious_trainer
    with pytest.raises(ValueError, match="frozen"):
        backend.wrapper.fit_guidance({}, previous, _feedback(records), {"controller_seed": 99})
    assert not backend.wrapper.training_audits


def test_another_base_sampler_snapshot_cannot_be_used(backend):
    snapshot = _snapshot(SimpleNamespace(identity={"checkpoint_sha256": "9" * 64}))
    with pytest.raises(ValueError, match="[Bb]ase"):
        backend.wrapper.generate({}, snapshot, _request(GenerationKind.GUIDED))
    assert not backend.inner.calls


def test_future_round_snapshot_cannot_be_used_early(backend):
    snapshot = _snapshot(backend.base, target_round=4)
    with pytest.raises(ValueError, match="round"):
        backend.wrapper.generate({}, snapshot, _request(GenerationKind.GUIDED, round_index=3))
    assert not backend.inner.calls


def test_bound_config_cannot_change_after_binding(backend):
    backend.wrapper.config = RoundGuidanceConfig(steps=2)
    with pytest.raises(ValueError, match="recipe changed"):
        backend.wrapper.generate({}, None, _request())
    with pytest.raises(ValueError, match="recipe changed"):
        backend.wrapper.fit_guidance({}, None, _feedback(()), {"controller_seed": 99})
    assert not backend.native_calls and not backend.trained


def test_old_round_snapshot_remains_frozen_during_new_training(backend):
    records = backend.wrapper.generate({}, None, _request(round_index=2))
    old = _snapshot(backend.base, target_round=2)
    before_report = deepcopy(old.report)
    new = backend.wrapper.fit_guidance({}, old, _feedback(records), {"controller_seed": 99})
    assert new.guide is not old.guide
    assert old.report == before_report and old.sampler.frozen
    assert old.sampler.freeze_checks >= 2
    assert backend.wrapper.training_audits[-1]["previous_snapshot_sha256"] == old.checkpoint_sha256
