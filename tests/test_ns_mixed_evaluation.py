"""Frozen mixed-bank request identity, original oracles and feedback isolation."""
from dataclasses import replace
import json

import numpy as np
import pytest

from pde_agent_verifier_solver.plugins.ns.case_bank import load_ns_continuous_case_bank
from pde_agent_verifier_solver.plugins.ns.mixed_evaluation import SharedNSMixedEvaluationGenerator
from pde_agent_verifier_solver.sdk import (
    CaseAccessKind, CaseVisibility, GenerationKind, GenerationPurpose,
    GenerationRequest, GeneratorMode,
)
from tests.test_ns_mixed_evaluation_preparation import (
    _controller, cpu_field_factory, prepared_suite,
)


def _wrapper(prepared_suite, tmp_path, *, suite=None):
    controller, _ = _controller(prepared_suite["field_factory"])
    settings = {**controller.campaign.task_settings, "default_query": controller.campaign.evaluator.default_query}
    path = prepared_suite["path"]
    if suite is not None:
        path = path.with_name("altered_suite.json")
        path.write_text(json.dumps(suite))
    wrapper = SharedNSMixedEvaluationGenerator(controller.generator, path,
        tmp_path / "mixed_audit.jsonl", controller.campaign.master_seed, settings)
    state = wrapper.load(controller.campaign.generator.checkpoint, {})
    return wrapper, state, controller.generator.backend.case_source


def _request(row, issued):
    return GenerationRequest(
        round_index=row["round_index"], count=row["count"], mode=GeneratorMode.ONLINE,
        kind=GenerationKind(row["kind"]), visibility=CaseVisibility.HIDDEN,
        seed=row["seed"], purpose=GenerationPurpose(row["purpose"]),
        anchors=tuple(issued[key].case for key in row["anchors"]),
    )


def _consume(wrapper, state, schedule, issued):
    for row in schedule:
        records = wrapper.sample(state, _request(row, issued))
        assert [record.case.case_id for record in records] == row["case_ids"]
        issued.update((record.case.case_id, record) for record in records)


def test_bank_records_keep_original_oracles_feedback_and_terminal_isolation(prepared_suite, tmp_path):
    wrapper, state, source = _wrapper(prepared_suite, tmp_path)
    entry = prepared_suite["suite"]["official_rounds"][0]
    bank = load_ns_continuous_case_bank(prepared_suite["path"].parent / entry["manifest"],
                                       expected_purpose="promotion")
    issued = {}
    _consume(wrapper, state, entry["sampling_requests"], issued)
    for record in bank.records:
        actual = issued[record.case.case_id]
        assert actual.case == record.case
        assert actual.truth.answer(bank.plans[record.case.case_id].query) == record.truth.answer(bank.plans[record.case.case_id].query)
        session = wrapper.open_case_session(state, actual.case, CaseAccessKind.PUBLIC_PROBLEM)
        points = np.array([[.2, .3], [.6, .7]])
        expected = record.truth._answer.evaluate_velocity(np.column_stack((np.zeros(2), points)))
        np.testing.assert_array_equal(session.query_initial_velocity(points), expected)
        session.close()
    base_ids = [key for key in issued if bank.strata[key] == "base"]
    feedback_fields = source.resolve_guidance_feedback_answers(base_ids)
    for key, field in zip(base_ids, feedback_fields):
        assert field is issued[key].truth._answer.raw_answer
    # Both frozen parent families remain identifiable; no Case/public_data rewrite.
    child_ids = [key for key in issued if bank.strata[key].endswith("_perturbed")]
    assert {bank.strata[issued[key].case.parent_case_id] for key in child_ids} == {"base", "guided"}
    for key in child_ids:
        row = next(row for row in bank.manifest["cases"] if row["case"]["case_id"] == key)
        assert dict(source.controller_perturbation_audit(key)) == row["generation_provenance"]["perturbation_audit"]
    wrapper.release(state, tuple(issued))
    assert not source._issued
    assert wrapper._official[1].bank is None
    assert not wrapper._official[1].records
    with pytest.raises(ValueError, match="already complete"):
        wrapper.sample(state, _request(entry["sampling_requests"][0], {}))
    with pytest.raises(ValueError, match="missing completed banks"):
        wrapper.assert_complete()

    terminal = {}
    _consume(wrapper, state, prepared_suite["suite"]["terminal_sampling_requests"], terminal)
    with pytest.raises(ValueError, match="only feedback or Official"):
        source.resolve_guidance_feedback_answers(tuple(terminal))
    wrapper.release(state, tuple(terminal))
    assert wrapper._terminal.bank is None and not source._issued


@pytest.mark.parametrize("change", ["seed", "count", "context", "source_order"])
def test_changed_request_or_controller_plan_blocks_before_resident_attach(prepared_suite, tmp_path, change):
    suite = prepared_suite["suite"]
    entry = suite["official_rounds"][0]
    if change == "context":
        key = entry["sampling_requests"][0]["case_ids"][0]
        entry["evaluation_context"][key]["case_position"] += 1
    elif change == "source_order":
        base_requests = [row for row in entry["sampling_requests"] if row["kind"] == "base"]
        first, second = base_requests[:2]
        first["case_ids"][0], second["case_ids"][0] = second["case_ids"][0], first["case_ids"][0]
    wrapper, state, source = _wrapper(prepared_suite, tmp_path, suite=suite)
    request = _request(entry["sampling_requests"][0], {})
    if change == "seed":
        request = replace(request, seed=request.seed + 1)
    elif change == "count":
        request = replace(request, count=request.count + 1)
    with pytest.raises(ValueError, match="changed|query or plan seed|positions"):
        wrapper.sample(state, request)
    assert not source._issued
    assert wrapper._official[1].next_request == 0
    audit = [json.loads(line) for line in (tmp_path / "mixed_audit.jsonl").read_text().splitlines()]
    assert audit[-1]["event"] == "verification_failed"


def test_changed_parent_snapshot_cannot_select_frozen_child(prepared_suite, tmp_path):
    wrapper, state, source = _wrapper(prepared_suite, tmp_path)
    schedule = prepared_suite["suite"]["official_rounds"][0]["sampling_requests"]
    assert schedule[0]["kind"] == "base" and schedule[1]["kind"] == "perturbed"
    root = wrapper.sample(state, _request(schedule[0], {}))[0]
    request = _request(schedule[1], {root.case.case_id: root})
    changed = replace(root.case, public_data={**root.case.public_data, "changed": True})
    with pytest.raises(ValueError, match="active frozen Case"):
        wrapper.sample(state, replace(request, anchors=(changed,)))
    assert set(source._issued) == {root.case.case_id}
    child = wrapper.sample(state, request)[0]
    assert child.case.parent_case_id == root.case.case_id
    wrapper.release(state, (root.case.case_id, child.case.case_id))


def test_development_and_initial_guidance_are_forwarded(prepared_suite, tmp_path, monkeypatch):
    wrapper, state, source = _wrapper(prepared_suite, tmp_path)
    calls = []
    original = wrapper._generator.sample
    def sample(state, request, progress=None):
        calls.append(request)
        return original(state, request, progress)
    monkeypatch.setattr(wrapper._generator, "sample", sample)
    for purpose, visibility in ((GenerationPurpose.DEVELOPMENT, CaseVisibility.DEVELOPMENT),
                                (GenerationPurpose.GUIDANCE_FEEDBACK, CaseVisibility.HIDDEN)):
        request = GenerationRequest(0, 1, GeneratorMode.ONLINE, GenerationKind.BASE,
                                    visibility, seed=23, purpose=purpose)
        records = wrapper.sample(state, request)
        assert calls[-1] is request
        assert records[0].case.visibility is visibility
        wrapper.release(state, tuple(record.case.case_id for record in records))
    assert not source._issued and not wrapper._bank_case_ids


def test_progress_failure_releases_attached_records_and_does_not_consume_request(prepared_suite, tmp_path):
    wrapper, state, source = _wrapper(prepared_suite, tmp_path)
    request = _request(prepared_suite["suite"]["official_rounds"][0]["sampling_requests"][0], {})
    def broken_progress(*args):
        raise RuntimeError("progress writer failed")
    with pytest.raises(RuntimeError, match="progress writer"):
        wrapper.sample(state, request, broken_progress)
    assert not source._issued and wrapper._official[1].next_request == 0
    records = wrapper.sample(state, request)
    assert len(records) == request.count
    wrapper.release(state, tuple(record.case.case_id for record in records))


@pytest.mark.parametrize("per_type", [2, 4])
def test_bank_size_is_configured_and_all_frozen_children_are_counted(cpu_field_factory, tmp_path, per_type):
    from tests.test_ns_mixed_evaluation_preparation import _preparation
    controller, _ = _controller(cpu_field_factory)
    path = _preparation().prepare(controller, tmp_path / "sized_banks",
        promotion_per_type=per_type, selection_per_type=per_type, terminal_per_type=per_type)
    prepared = {"path": path, "suite": json.loads(path.read_text()), "field_factory": cpu_field_factory}
    wrapper, state, source = _wrapper(prepared, tmp_path)
    for entry in prepared["suite"]["official_rounds"]:
        issued = {}
        _consume(wrapper, state, entry["sampling_requests"], issued)
        assert len(issued) == 3 * per_type
        wrapper.release(state, tuple(issued))
    issued = {}
    _consume(wrapper, state, prepared["suite"]["terminal_sampling_requests"], issued)
    assert len(issued) == 3 * per_type
    wrapper.release(state, tuple(issued))
    expected = {"base": per_type, "guided": per_type,
                "base_perturbed": per_type // 2, "guided_perturbed": per_type // 2}
    assert wrapper.assert_complete() == {"official:1": expected, "official:2": expected, "terminal": expected}
    assert not source._issued
