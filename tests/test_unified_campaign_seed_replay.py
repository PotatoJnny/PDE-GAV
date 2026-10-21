from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import random
from unittest.mock import patch


from pde_agent_verifier_solver.engine.macro_closed_loop import (
    MacroAgentOutcome,
    MacroAgentSpec,
    MacroClosedLoopRunner,
    MacroLoopConfig,
)
from pde_agent_verifier_solver.engine.selection import EvidenceStage
from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    AnswerView,
    Case,
    CaseRecord,
    CaseSource,
    CaseVisibility,
    GenerationKind,
    MacroMatchResult,
    MacroTarget,
    PanelCard,
    ProgramInfo,
)
from pde_agent_verifier_solver.sdk.macro_loop import (
    CandidateRef,
    CandidateSubmission,
    CohortKind,
    MacroLoopPlugin,
)


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_unified_plasma_agent_iteration.py"
)
_SPEC = importlib.util.spec_from_file_location("unified_plasma_iteration", SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
UNIFIED = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(UNIFIED)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class _Truth:
    def __init__(self, value: float):
        self.value = value

    def answer(self, query):
        return AnswerView(
            kind=query.kind,
            fields={"macro": (self.value,)},
            axes={"sample": (0.0,)},
            shape=(1,),
        )


class _Matcher:
    info = ProgramInfo("seed-test", "Seed test matcher", "test")
    contract_id = "seed_test_macro_v1"
    fields = ("macro",)

    def prepare_target(self, case, truth, query):
        return MacroTarget(case.case_id, self.contract_id, query, truth)

    def match(self, case, prediction, target):
        error = abs(
            float(prediction.fields["macro"][0])
            - float(target.values.fields["macro"][0])
        )
        return MacroMatchResult(case.case_id, self.contract_id, error)

    def result_card(self, result):
        return PanelCard("seed-test", "test", {"error": result.primary_error})


class _Objective:
    matcher = _Matcher()
    agent_visible_metrics = ()
    published_metric_names = ()

    def query(self, case, profile):
        return AnswerQuery.grid(fields=("macro",), sample=1)


class _SeededPairedSource:
    generator_id = "seeded-paired-source"
    artifact_sha256 = "7" * 64
    supports_perturbations = True

    def __init__(self):
        self.requests = []
        self.responses = []

    def sample(self, request):
        self.requests.append(request)
        rng = random.Random(request.seed)
        visibility = (
            CaseVisibility.DEVELOPMENT
            if request.cohort is CohortKind.DEVELOPMENT
            else CaseVisibility.HIDDEN
        )
        records = []
        if request.kind is GenerationKind.BASE:
            anchors = (None,) * request.count
        else:
            assert request.kind is GenerationKind.PERTURBED
            anchors = request.anchors
        for index, anchor in enumerate(anchors):
            draw = rng.randrange(2**32)
            if anchor is None:
                case_id = (
                    f"r{request.round_index}-{request.cohort.value}-"
                    f"s{request.seed}-{index}-{draw:08x}"
                )
                source = CaseSource.GENERATED
                parent_case_id = None
            else:
                case_id = f"{anchor.case_id}-pert-{draw:08x}"
                source = CaseSource.PERTURBED
                parent_case_id = anchor.case_id
            value = float(draw) / float(2**32)
            records.append(
                CaseRecord(
                    Case(
                        case_id=case_id,
                        source=source,
                        visibility=visibility,
                        public_data={
                            "cohort": request.cohort.value,
                            "sample": value,
                        },
                        round_index=request.round_index,
                        parent_case_id=parent_case_id,
                        tags=(f"cohort:{request.cohort.value}",),
                    ),
                    _Truth(value),
                )
            )
        response = tuple(records)
        self.responses.append(response)
        return response


class _ConstantCandidate:
    def __init__(self, value: float):
        self.value = value

    def predict_macros(self, case, query, budget=None):
        return AnswerView(
            kind=query.kind,
            fields={"macro": (self.value,)},
            axes={"sample": (0.0,)},
            shape=(1,),
        )


class _CapturingAgent:
    def __init__(self, agent_id: str, value: float):
        self.agent_id = agent_id
        self.value = value
        self.contexts = []

    def run(self, context):
        self.contexts.append(context)
        candidate_id = f"{self.agent_id}-round-{context.round_index}"
        ref = CandidateRef(
            candidate_id=candidate_id,
            bundle_sha256=_digest(f"{candidate_id}:{self.value}"),
            round_index=context.round_index,
            agent_id=context.agent.agent_id,
            parent_sha256=context.incumbent.bundle_sha256,
        )
        return MacroAgentOutcome(
            CandidateSubmission(ref, _ConstantCandidate(self.value))
        )


def _run(seed_base: int):
    source = _SeededPairedSource()
    agents = {
        "agent-a": _CapturingAgent("agent-a", 0.25),
        "agent-b": _CapturingAgent("agent-b", 0.75),
    }
    cold = CandidateSubmission(
        CandidateRef("cold", "c" * 64, 0, "baseline"),
        _ConstantCandidate(0.5),
    )
    result = MacroClosedLoopRunner(
        legacy_macro_reference=True,
        config=MacroLoopConfig(
            name="campaign-seed-contract",
            round_count=1,
            agents=(
                MacroAgentSpec("agent-a", "Agent A"),
                MacroAgentSpec("agent-b", "Agent B"),
            ),
            development_case_count=2,
            selection_case_count=2,
            evidence_stages=(EvidenceStage(2, confidence_z=0.0),),
            terminal_case_count=2,
            agent_case_credits=2,
            seed_base=seed_base,
            paired_perturbations=True,
            terminal_confidence_z=0.0,
            terminal_require_cvar_no_regression=False,
        ),
        plugin=MacroLoopPlugin("seed-test", source, _Objective()),
        cold_start=cold,
        agents=agents,
    ).run()
    return result, source, agents


def _request_signature(source):
    return tuple(
        (
            request.round_index,
            request.cohort,
            request.kind,
            request.seed,
            tuple(anchor.case_id for anchor in request.anchors),
        )
        for request in source.requests
    )


def test_omitted_seed_uses_fresh_entropy_and_changes_samples():
    payload = {"campaign_id": "fresh-campaign"}
    with patch.object(UNIFIED.secrets, "randbits", side_effect=(101, 202)) as entropy:
        first, first_record = UNIFIED._resolve_campaign_seed(payload)
        second, second_record = UNIFIED._resolve_campaign_seed(payload)

    assert "seed_base" not in payload
    assert (first["seed_base"], second["seed_base"]) == (101, 202)
    assert first_record["source"] == second_record["source"] == "system_entropy"
    assert entropy.call_args_list[0].args == (UNIFIED.AUTO_SEED_BITS,)
    assert entropy.call_args_list[1].args == (UNIFIED.AUTO_SEED_BITS,)

    first_result, _first_source, first_agents = _run(first["seed_base"])
    second_result, _second_source, second_agents = _run(second["seed_base"])
    first_ids = tuple(
        case.case_id for case in first_agents["agent-a"].contexts[0].cases
    )
    second_ids = tuple(
        case.case_id for case in second_agents["agent-a"].contexts[0].cases
    )
    assert first_ids != second_ids
    assert first_result.rounds[0].finalist_evaluations[0].batch.cohort_sha256 != (
        second_result.rounds[0].finalist_evaluations[0].batch.cohort_sha256
    )


def test_explicit_replay_seed_reproduces_and_is_persisted(tmp_path):
    payload = {"campaign_id": "replay-campaign", "seed_base": 314159}
    with patch.object(
        UNIFIED.secrets,
        "randbits",
        side_effect=AssertionError("explicit replay must not draw entropy"),
    ):
        first, record = UNIFIED._resolve_campaign_seed(payload)
        second, second_record = UNIFIED._resolve_campaign_seed(payload)

    assert first["seed_base"] == second["seed_base"] == 314159
    assert record == second_record
    assert record["source"] == "explicit_config"
    assert record["replay"] == {"config_field": "seed_base", "value": 314159}

    seed_path = UNIFIED._persist_campaign_seed(tmp_path, record)
    assert json.loads(seed_path.read_text(encoding="utf-8")) == record
    assert seed_path.stat().st_mode & 0o777 == 0o600
    assert list(tmp_path.glob(".campaign_seed.*.tmp")) == []
    try:
        UNIFIED._persist_campaign_seed(tmp_path, record)
    except FileExistsError:
        pass
    else:
        raise AssertionError("seed audit must not be overwritten")

    assert list(tmp_path.glob(".campaign_seed.*.tmp")) == []
    first_result, first_source, first_agents = _run(first["seed_base"])
    second_result, second_source, second_agents = _run(second["seed_base"])
    assert _request_signature(first_source) == _request_signature(second_source)
    assert tuple(
        case.case_id for case in first_agents["agent-a"].contexts[0].cases
    ) == tuple(
        case.case_id for case in second_agents["agent-a"].contexts[0].cases
    )
    assert first_result.terminal.cohort_sha256 == second_result.terminal.cohort_sha256
    assert first_result.terminal.cold_start.target_sha256 == (
        second_result.terminal.cold_start.target_sha256
    )
    assert first_result.terminal.selected.target_sha256 == (
        second_result.terminal.selected.target_sha256
    )


def test_controller_pair_lineage_and_candidate_comparisons_share_one_frozen_cohort():
    result, source, agents = _run(271828)

    assert len(source.requests) == 6
    assert tuple(request.cohort for request in source.requests[::2]) == (
        CohortKind.DEVELOPMENT,
        CohortKind.SELECTION,
        CohortKind.TERMINAL,
    )
    assert all(
        request.cohort is not CohortKind.PROMOTION
        for request in source.requests
    )
    for pair_index, (base, perturbed) in enumerate(
        zip(source.requests[::2], source.requests[1::2])
    ):
        assert base.kind is GenerationKind.BASE
        assert perturbed.kind is GenerationKind.PERTURBED
        assert perturbed.seed == base.seed + 4
        roots = source.responses[2 * pair_index]
        children = source.responses[2 * pair_index + 1]
        assert tuple(anchor.case_id for anchor in perturbed.anchors) == tuple(
            root.case.case_id for root in roots
        )
        assert tuple(child.case.parent_case_id for child in children) == tuple(
            root.case.case_id for root in roots
        )

    contexts = [agents[name].contexts[0] for name in ("agent-a", "agent-b")]
    context_ids = [
        tuple(case.case_id for case in context.cases) for context in contexts
    ]
    assert context_ids[0] == context_ids[1]
    # Pair lineage remains available to the trusted controller above, but it
    # is deliberately removed from the Agent-facing envelope. Otherwise an
    # Agent could branch on the clean/perturbed label instead of solving the
    # public PDE question. Both Agents must receive the same opaque view.
    assert all(
        case.case_id.startswith("candidate-case-")
        and case.source is CaseSource.GENERATED
        and case.visibility is CaseVisibility.DEVELOPMENT
        and case.round_index is None
        and case.parent_case_id is None
        and case.tags == ()
        for case in contexts[0].cases
    )

    finalists = result.rounds[0].finalist_evaluations
    assert len(finalists) == 2
    batches = [finalist.batch for finalist in finalists]
    assert batches[0].case_ids == batches[1].case_ids
    assert batches[0].cohort_sha256 == batches[1].cohort_sha256
    assert batches[0].target_sha256 == batches[1].target_sha256
    assert contexts[0].evaluator.cohort_sha256 == contexts[1].evaluator.cohort_sha256
