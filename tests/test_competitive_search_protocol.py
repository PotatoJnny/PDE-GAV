from __future__ import annotations

from dataclasses import asdict
import hashlib
import json

from pde_agent_verifier_solver.engine.development_session import DevelopmentSession
from pde_agent_verifier_solver.engine.macro_closed_loop import (
    CompetitiveSearchPolicy,
    MacroAgentContext,
    MacroAgentOutcome,
    MacroAgentSpec,
    MacroClosedLoopRunner,
    MacroLoopConfig,
    SearchLeaderboardEntry,
    evaluate_search_completion,
)
from pde_agent_verifier_solver.engine.macro_evaluator import PreparedMacroCohort
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
    ProgramInfo,
    QueryKind,
)
from pde_agent_verifier_solver.sdk.macro_loop import (
    CandidateRef,
    CandidateSubmission,
    CohortKind,
    MacroLoopPlugin,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class _Truth:
    def answer(self, query: AnswerQuery) -> AnswerView:
        return AnswerView(
            kind=query.kind,
            fields={"macro": (1.0,)},
            axes={"sample": (0.0,)},
            shape=(1,),
        )


class _FreshCaseSource:
    generator_id = "competitive-search-toy-generator"
    artifact_sha256 = "a" * 64
    supports_perturbations = True

    def __init__(self) -> None:
        self.requests = []

    def sample(self, request):
        self.requests.append(request)
        visibility = (
            CaseVisibility.DEVELOPMENT
            if request.cohort is CohortKind.DEVELOPMENT
            else CaseVisibility.HIDDEN
        )
        if request.kind is GenerationKind.BASE:
            return tuple(
                CaseRecord(
                    Case(
                        case_id=(
                            f"r{request.round_index}-{request.cohort.value}-"
                            f"s{request.seed}-root-{index}"
                        ),
                        source=CaseSource.GENERATED,
                        visibility=visibility,
                        public_data={"input": float(index)},
                        round_index=request.round_index,
                    ),
                    _Truth(),
                )
                for index in range(request.count)
            )
        if request.kind is not GenerationKind.PERTURBED:
            raise AssertionError(f"unexpected generation kind: {request.kind}")
        return tuple(
            CaseRecord(
                Case(
                    case_id=f"{anchor.case_id}-perturbed-s{request.seed}",
                    source=CaseSource.PERTURBED,
                    visibility=visibility,
                    public_data={"input": anchor.public_data["input"]},
                    round_index=request.round_index,
                    parent_case_id=anchor.case_id,
                ),
                _Truth(),
            )
            for anchor in request.anchors
        )


class _PowerMatcher:
    fields = ("macro",)

    def __init__(self, power: int) -> None:
        self.power = power
        self.contract_id = f"toy_power_{power}_macro_v1"
        self.info = ProgramInfo(
            f"power-{power}",
            f"Power-{power} toy objective",
            "test",
        )

    def prepare_target(self, case, truth, query):
        return MacroTarget(case.case_id, self.contract_id, query, truth)

    def match(self, case, prediction, target):
        error = abs(
            prediction.fields["macro"][0]
            - target.values.fields["macro"][0]
        ) ** self.power
        return MacroMatchResult(
            case.case_id,
            self.contract_id,
            error,
            metrics={"equation_owned_component": error},
        )


class _Objective:
    agent_visible_metrics = ("equation_owned_component",)
    published_metric_names = ("equation_owned_component",)

    def __init__(self, power: int = 1) -> None:
        self.matcher = _PowerMatcher(power)

    def query(self, case, profile):
        return AnswerQuery.grid(fields=("macro",), sample=1)


class _Approximation:
    def __init__(self, error: float) -> None:
        self.error = error

    def predict_macros(self, case, query, budget=None):
        return AnswerView(
            kind=QueryKind.GRID,
            fields={"macro": (1.0 - self.error,)},
            axes={"sample": (0.0,)},
            shape=(1,),
        )


class _ScriptedSearchAgent:
    """Screen deterministic candidates while recording generic Agent context."""

    def __init__(
        self,
        branch_id: str,
        errors_by_round_attempt: dict[tuple[int, int], tuple[float, ...]],
        *,
        full_stage_on: set[tuple[int, int, int]] | None = None,
    ) -> None:
        self.branch_id = branch_id
        self.errors_by_round_attempt = errors_by_round_attempt
        self.full_stage_on = full_stage_on or set()
        self.contexts = []
        self.candidates = []

    def run(self, context):
        self.contexts.append(context)
        errors = self.errors_by_round_attempt.get(
            (context.round_index, 1),
            (0.45,),
        )
        current = None
        for index, error in enumerate(errors, start=1):
            label = (
                f"{self.branch_id}-r{context.round_index}-"
                f"a1-c{index}"
            )
            current = CandidateSubmission(
                CandidateRef(
                    candidate_id=label,
                    bundle_sha256=_digest(f"{label}:{error}"),
                    round_index=context.round_index,
                    agent_id=context.agent.agent_id,
                    parent_sha256=context.incumbent.bundle_sha256,
                ),
                _Approximation(error),
            )
            context.evaluator.screen(current, requested_root_count=1)
            if (
                context.round_index,
                1,
                index,
            ) in self.full_stage_on:
                context.evaluator.screen(current, requested_root_count=2)
            self.candidates.append(current)
        assert current is not None
        return MacroAgentOutcome(current)


class _UnscreenedFinalAgent(_ScriptedSearchAgent):
    """Return a candidate that was not submitted to the screen oracle."""

    def run(self, context):
        super().run(context)
        label = f"{self.branch_id}-r{context.round_index}-unscreened-final"
        final = CandidateSubmission(
            CandidateRef(
                candidate_id=label,
                bundle_sha256=_digest(label),
                round_index=context.round_index,
                agent_id=context.agent.agent_id,
                parent_sha256=context.incumbent.bundle_sha256,
            ),
            _Approximation(0.01),
        )
        return MacroAgentOutcome(final)


class _DevelopmentFactory:
    def __init__(self) -> None:
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        records = tuple(
            CaseRecord(
                Case(
                    case_id=(
                        f"{request.kind.value}-draw-{request.draw_index}-"
                        f"root-{index}"
                    ),
                    source=CaseSource.GENERATED,
                    visibility=CaseVisibility.DEVELOPMENT,
                    public_data={"input": float(index)},
                ),
                _Truth(),
            )
            for index in range(request.root_count)
        )
        return PreparedMacroCohort.prepare(
            cohort_id=f"development-{request.kind.value}-{request.draw_index}",
            records=records,
            matcher=_PowerMatcher(1),
            query=AnswerQuery.grid(fields=("macro",), sample=1),
        )


def _milestone_session():
    factory = _DevelopmentFactory()
    incumbent = CandidateSubmission(
        CandidateRef("milestone-cold", "d" * 64, 0, "baseline"),
        _Approximation(0.5),
    )
    session = DevelopmentSession(
        incumbent=incumbent,
        cohort_factory=factory,
        screen_root_stages=(1, 2),
        screen_root_credit_limit=8,
        release_root_credit_limit=2,
        confidence_z=0.0,
        continuation_max_mean_ratio=None,
    )
    return session, incumbent


def _candidate(label: str, error: float, incumbent: CandidateSubmission):
    return CandidateSubmission(
        CandidateRef(
            label,
            _digest(label),
            1,
            "agent-a",
            parent_sha256=incumbent.ref.bundle_sha256,
        ),
        _Approximation(error),
    )


def _config(
    agents: tuple[MacroAgentSpec, ...],
    search_policy: CompetitiveSearchPolicy,
    *,
    round_count: int = 1,
) -> MacroLoopConfig:
    return MacroLoopConfig(
        name="competitive-search-toy",
        round_count=round_count,
        agents=agents,
        development_case_count=2,
        selection_case_count=4,
        evidence_stages=(
            EvidenceStage(
                4,
                confidence_z=0.0,
                min_win_rate=0.0,
                require_cvar_no_regression=False,
            ),
        ),
        terminal_case_count=4,
        agent_case_credits=2,
        seed_base=10_000,
        seed_stride=1_000,
        development_release_root_limit=1,
        development_screen_root_stages=(1, 2),
        development_screen_root_credit_limit=8,
        development_screen_confidence_z=0.0,
        development_screen_continuation_max_mean_ratio=None,
        terminal_confidence_z=0.0,
        terminal_min_win_rate=0.0,
        terminal_require_cvar_no_regression=False,
        paired_perturbations=True,
        search_policy=search_policy,
    )


def _run(
    programs: dict[str, _ScriptedSearchAgent],
    policy: CompetitiveSearchPolicy,
    *,
    round_count: int = 1,
    power: int = 1,
    backend_label: str = "generic",
):
    source = _FreshCaseSource()
    specs = tuple(
        MacroAgentSpec(agent_id, f"Agent {agent_id}")
        for agent_id in programs
    )
    cold = CandidateSubmission(
        CandidateRef("cold", "c" * 64, 0, "baseline"),
        _Approximation(0.5),
    )
    result = MacroClosedLoopRunner(
        legacy_macro_reference=True,
        config=_config(specs, policy, round_count=round_count),
        plugin=MacroLoopPlugin(
            f"{backend_label}-competitive-search-power-{power}",
            source,
            _Objective(power),
        ),
        cold_start=cold,
        agents=programs,
    ).run()
    return result, source


def test_default_search_policy_opens_leaderboard_and_cross_round_sharing():
    policy = CompetitiveSearchPolicy()

    assert asdict(policy) == {
        "min_distinct_screened_candidates": 0,
        "min_full_stage_candidates": 0,
        "min_released_roots": 0,
        "require_final_full_stage": False,
        "leaderboard_limit": 8,
        "cross_share_after_round": 1,
    }
    session, _incumbent = _milestone_session()
    audit = evaluate_search_completion(
        session,
        final_candidate_sha256=None,
        agent_id="agent-a",
        policy=policy,
    )

    assert audit.agent_id == "agent-a"
    assert audit.distinct_screened_candidates == 0
    assert audit.full_stage_candidates == 0
    assert audit.released_roots == 0
    assert audit.final_candidate_sha256 is None
    assert audit.final_completed_root_count == 0
    assert audit.sufficient
    assert audit.failures == ()

    config = _config(
        (MacroAgentSpec("agent-a", "Agent A"),),
        policy,
    )
    assert config.search_policy == policy


def test_search_accounting_ignores_legacy_minima_and_deduplicates_digests():
    policy = CompetitiveSearchPolicy(
        min_distinct_screened_candidates=2,
        min_full_stage_candidates=2,
        min_released_roots=2,
        require_final_full_stage=True,
        leaderboard_limit=2,
        cross_share_after_round=1,
    )
    session, incumbent = _milestone_session()
    first = _candidate("candidate-a", 0.30, incumbent)
    relabeled_first = CandidateSubmission(
        CandidateRef(
            "candidate-a-relabeled",
            first.ref.bundle_sha256,
            1,
            "agent-a",
            parent_sha256=incumbent.ref.bundle_sha256,
        ),
        _Approximation(0.90),
    )
    session.release(1)
    session.screen(first, requested_root_count=1)
    cached = session.screen(relabeled_first, requested_root_count=1)
    assert cached.cached

    incomplete = evaluate_search_completion(
        session,
        final_candidate_sha256=first.ref.bundle_sha256,
        agent_id="agent-a",
        policy=policy,
    )

    assert incomplete.distinct_screened_candidates == 1
    assert incomplete.full_stage_candidates == 0
    assert incomplete.released_roots == 1
    assert incomplete.final_completed_root_count == 1
    assert incomplete.sufficient
    assert incomplete.failures == ()

    session.release(1)
    session.screen(first, requested_root_count=2)
    second = _candidate("candidate-b", 0.20, incumbent)
    session.screen(second, requested_root_count=1)
    session.screen(second, requested_root_count=2)

    complete = evaluate_search_completion(
        session,
        final_candidate_sha256=second.ref.bundle_sha256,
        agent_id="agent-a",
        policy=policy,
    )

    assert complete.distinct_screened_candidates == 2
    assert complete.full_stage_candidates == 2
    assert complete.released_roots == 2
    assert complete.final_candidate_sha256 == second.ref.bundle_sha256
    assert complete.final_completed_root_count == 2
    assert complete.sufficient
    assert complete.failures == ()


def test_agent_context_carries_only_sanitized_prior_search_state():
    session, incumbent = _milestone_session()
    shared = _candidate("shared", 0.10, incumbent).ref
    entry = SearchLeaderboardEntry(
        rank=1,
        candidate=shared,
        agent_id="agent-b",
        eligible=True,
        mean=0.10,
        cvar=0.12,
        maximum=0.15,
        generated_mean=0.09,
        perturbed_mean=0.11,
        guardrail_passed=True,
        roles=("screen_mean",),
    )
    context = MacroAgentContext(
        round_index=2,
        agent=MacroAgentSpec("agent-a", "Agent A"),
        prompt="Continue from aggregate prior evidence.",
        cases=(),
        evaluator=session,
        incumbent=incumbent.ref,
        prior_handoff=None,
        prior_leaderboard=(entry,),
        shared_candidates=(shared,),
    )

    assert context.prior_leaderboard == (entry,)
    assert context.shared_candidates == (shared,)
    public = json.dumps(asdict(entry), sort_keys=True).lower()
    for forbidden in (
        "case_ids",
        "per_case",
        "finite_truth",
        "target_values",
    ):
        assert forbidden not in public


def test_round_leaderboard_is_identical_across_backend_labels():
    signatures = []
    for backend_label in ("ns", "vpfp", "gkw"):
        policy = CompetitiveSearchPolicy(
            leaderboard_limit=2,
            cross_share_after_round=1,
        )
        left = _ScriptedSearchAgent("left", {(1, 1): (0.10,)})
        right = _ScriptedSearchAgent("right", {(1, 1): (0.20,)})
        result, source = _run(
            {"left": left, "right": right},
            policy,
            backend_label=backend_label,
        )
        round_result = result.rounds[0]
        commitments = {
            (
                finalist.batch.cohort_sha256,
                finalist.batch.target_sha256,
                finalist.batch.case_ids,
            )
            for finalist in round_result.finalist_evaluations
        }
        assert len(commitments) == 1
        assert len(round_result.leaderboard) == 2
        assert all(
            isinstance(entry, SearchLeaderboardEntry)
            for entry in round_result.leaderboard
        )
        assert [entry.rank for entry in round_result.leaderboard] == [1, 2]
        assert {entry.agent_id for entry in round_result.leaderboard} == {
            "left",
            "right",
        }
        assert result.selected.candidate_id.startswith("left-")
        assert (
            round_result.promotion.reason
            == "base_improved_with_robustness_guardrails"
        )
        assert all(
            request.cohort is not CohortKind.PROMOTION
            for request in source.requests
        )
        serialized = json.dumps(
            [asdict(entry) for entry in round_result.leaderboard],
            sort_keys=True,
        ).lower()
        assert "case_ids" not in serialized
        assert "per_case" not in serialized
        signatures.append(
            (
                tuple(
                    (
                        request.round_index,
                        request.cohort,
                        request.kind,
                        request.seed,
                        request.count,
                    )
                    for request in source.requests
                ),
                tuple(
                    (
                        entry.rank,
                        entry.candidate.bundle_sha256,
                        entry.agent_id,
                        entry.eligible,
                        entry.mean,
                        entry.cvar,
                        entry.maximum,
                        entry.roles,
                    )
                    for entry in round_result.leaderboard
                ),
            )
        )

    assert signatures[0] == signatures[1] == signatures[2]


def test_same_digest_nominations_receive_only_one_official_vote():
    shared_digest = _digest("shared-official-bundle")

    class SameDigestAgent:
        def __init__(self, error):
            self.error = error

        def run(self, context):
            candidate_id = f"{context.agent.agent_id}-same-digest"
            return MacroAgentOutcome(
                CandidateSubmission(
                    CandidateRef(
                        candidate_id,
                        shared_digest,
                        context.round_index,
                        context.agent.agent_id,
                        parent_sha256=context.incumbent.bundle_sha256,
                    ),
                    _Approximation(self.error),
                )
            )

    result, _source = _run(
        {
            "left": SameDigestAgent(0.10),
            "right": SameDigestAgent(0.20),
        },
        CompetitiveSearchPolicy(),
    )
    round_result = result.rounds[0]
    nominations = tuple(
        row
        for row in round_result.leaderboard
        if "incumbent" not in row.roles
    )

    assert len(round_result.candidate_refs) == 1
    assert len(round_result.finalist_evaluations) == 1
    assert len(nominations) == 1
    assert nominations[0].candidate.bundle_sha256 == shared_digest
    assert nominations[0].roles == ("agent_nomination",)


def test_legacy_investment_minimum_cannot_block_a_strong_checkpoint():
    policy = CompetitiveSearchPolicy(
        min_distinct_screened_candidates=2,
    )
    agent = _ScriptedSearchAgent(
        "underinvested",
        {(1, 1): (0.10,)},
    )

    result, _source = _run({"agent-a": agent}, policy)
    round_result = result.rounds[0]
    audit = round_result.search_completion["agent-a"]

    assert audit.sufficient
    assert audit.failures == ()
    assert round_result.agent_failures == {}
    assert round_result.candidate_refs == (agent.candidates[0].ref,)
    assert tuple(
        row.candidate for row in round_result.finalist_evaluations
    ) == (agent.candidates[0].ref,)
    assert result.selected == agent.candidates[0].ref
    prompt = agent.contexts[0].prompt
    assert "there is no minimum usage or finish milestone" in prompt
    assert (
        "dimensions, coordinates, basis, and interpolation are not fixed"
        in prompt
    )
    assert "prepare_solution(case, solve_budget)" in prompt
    assert (
        round_result.promotion.reason
        == "base_improved_with_robustness_guardrails"
    )
    assert "one official entry for the round" in prompt


def test_common_cohort_keeps_the_better_incumbent():
    agent = _ScriptedSearchAgent(
        "worse",
        {(1, 1): (0.80,)},
    )

    result, source = _run(
        {"agent-a": agent},
        CompetitiveSearchPolicy(),
    )
    round_result = result.rounds[0]

    assert result.selected.candidate_id == "cold"
    assert round_result.leaderboard[0].candidate.candidate_id == "cold"
    assert round_result.leaderboard[0].roles == ("incumbent",)
    assert not round_result.promotion.promoted
    assert (
        round_result.promotion.reason
        == "incumbent_kept_after_robustness_guardrails"
    )
    assert all(
        request.cohort is not CohortKind.PROMOTION
        for request in source.requests
    )


def test_private_archive_representatives_do_not_gain_official_votes():
    policy = CompetitiveSearchPolicy(
        min_distinct_screened_candidates=2,
        min_full_stage_candidates=1,
        min_released_roots=1,
        require_final_full_stage=True,
    )
    agents = (
        _ScriptedSearchAgent(
            "partial-final",
            {(1, 1): (0.20, 0.05)},
            full_stage_on={(1, 1, 1)},
        ),
        _UnscreenedFinalAgent(
            "unscreened-final",
            {(1, 1): (0.20, 0.05)},
            full_stage_on={(1, 1, 1)},
        ),
    )

    for agent in agents:
        result, _source = _run({"agent-a": agent}, policy)
        round_result = result.rounds[0]
        audit = round_result.search_completion["agent-a"]
        screened = {candidate.ref for candidate in agent.candidates}

        assert audit.distinct_screened_candidates == 2
        assert audit.full_stage_candidates == 1
        assert audit.released_roots == 1
        assert audit.sufficient
        assert audit.failures == ()
        assert round_result.agent_failures == {}
        assert len(screened) == 2
        assert len(round_result.candidate_refs) == 1
        assert len(round_result.finalist_evaluations) == 1
        official = round_result.candidate_refs[0]
        assert round_result.finalist_evaluations[0].candidate == official
        assert round_result.candidate_portfolio_roles == {
            official.candidate_id: ("agent_nomination",),
        }
        assert (
            agent.contexts[0].search_policy.min_distinct_screened_candidates
            == 0
        )
        if isinstance(agent, _UnscreenedFinalAgent):
            assert official not in screened
            assert official.candidate_id.endswith("unscreened-final")
        else:
            assert official == agent.candidates[1].ref
        assert result.selected == official


def test_unscreened_checkpoint_is_still_evaluated_as_the_nomination():
    agent = _UnscreenedFinalAgent(
        "open",
        {(1, 1): (0.10,)},
    )

    result, _source = _run(
        {"agent-a": agent},
        CompetitiveSearchPolicy(),
    )
    round_result = result.rounds[0]
    audit = round_result.search_completion["agent-a"]

    assert audit.sufficient
    assert audit.failures == ()
    assert result.selected.candidate_id.endswith("unscreened-final")
    assert result.selected in round_result.candidate_refs
    assert result.selected in tuple(
        row.candidate for row in round_result.finalist_evaluations
    )
    assert (
        round_result.candidate_portfolio_roles[result.selected.candidate_id]
        == ("agent_nomination",)
    )
