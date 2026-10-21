import hashlib
import json

import pytest

from pde_agent_verifier_solver.engine.development_session import (
    DevelopmentCreditExceeded,
    DevelopmentDrawKind,
    DevelopmentDrawRequest,
    DevelopmentSession,
    DevelopmentSessionError,
)
from pde_agent_verifier_solver.engine.candidate_case import candidate_case_id
from pde_agent_verifier_solver.engine.macro_evaluator import PreparedMacroCohort
from pde_agent_verifier_solver.engine.selection import (
    PairedObjectiveMode,
    PairedObjectivePolicy,
)
from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    AnswerView,
    CandidateRef,
    CandidateSubmission,
    Case,
    CaseRecord,
    CaseSource,
    CaseVisibility,
    MacroMatchResult,
    MacroTarget,
    ProgramInfo,
    QueryKind,
)


QUERY = AnswerQuery.grid(fields=("macro",), sample=1)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class _Truth:
    def __init__(self, value: float = 1.0) -> None:
        self.value = value

    def answer(self, query):
        return AnswerView(
            kind=query.kind,
            fields={"macro": (self.value,)},
            axes={"sample": (0.0,)},
            shape=(1,),
        )


class _Matcher:
    info = ProgramInfo("development-test", "Development test", "test")
    contract_id = "development_session_macro_v1"
    fields = ("macro",)

    def prepare_target(self, case, truth, query):
        return MacroTarget(case.case_id, self.contract_id, query, truth)

    def match(self, case, prediction, target):
        error = abs(
            float(prediction.fields["macro"][0])
            - float(target.values.fields["macro"][0])
        )
        return MacroMatchResult(
            case_id=case.case_id,
            contract_id=self.contract_id,
            primary_error=error,
            metrics={"component_error": error, "private_metric": 99.0},
        )


class _Program:
    def __init__(self, error: float) -> None:
        self.error = error
        self.case_ids = []

    def predict_macros(self, case, query, budget=None):
        self.case_ids.append(case.case_id)
        return AnswerView(
            kind=QueryKind.GRID,
            fields={"macro": (1.0 - self.error,)},
            axes={"sample": (0.0,)},
            shape=(1,),
        )


class _OrderedProgram(_Program):
    def __init__(self, error: float, label: str, events: list) -> None:
        super().__init__(error)
        self.label = label
        self.events = events

    def predict_macros(self, case, query, budget=None):
        self.events.append((case.case_id, self.label))
        return super().predict_macros(case, query, budget=budget)


class _SplitProgram:
    def __init__(self, generated_error: float, perturbed_error: float) -> None:
        self.generated_error = generated_error
        self.perturbed_error = perturbed_error

    def predict_macros(self, case, query, budget=None):
        error = (
            self.perturbed_error
            if float(case.public_data["input"]) % 1.0 != 0.0
            else self.generated_error
        )
        return AnswerView(
            kind=QueryKind.GRID,
            fields={"macro": (1.0 - error,)},
            axes={"sample": (0.0,)},
            shape=(1,),
        )


class _ScriptedReleaseOrderRng:
    def __init__(self, *orders) -> None:
        self.orders = list(orders)

    def shuffle(self, values) -> None:
        if not self.orders:
            raise AssertionError("test exhausted scripted release orders")
        order = self.orders.pop(0)
        if sorted(order) != list(range(len(values))):
            raise AssertionError("test supplied an invalid release order")
        values[:] = order


class _FreshPairedFactory:
    def __init__(self, private_store_root=None, *, vary_truth=False) -> None:
        self.requests = []
        self.vary_truth = vary_truth
        self.returned_root_ids = []
        self.private_store_root = private_store_root
        self.cohorts = []
        self.store_roots = []

    def __call__(self, request):
        self.requests.append(request)
        records = []
        roots = []
        for index in range(request.root_count):
            root_id = (
                f"{request.kind.value}-draw-{request.draw_index}-root-{index}"
            )
            roots.append(root_id)
            root = Case(
                case_id=root_id,
                source=CaseSource.GENERATED,
                visibility=CaseVisibility.DEVELOPMENT,
                public_data={"input": float(index)},
            )
            child = Case(
                case_id=f"{root_id}-perturbed",
                source=CaseSource.PERTURBED,
                visibility=CaseVisibility.DEVELOPMENT,
                public_data={"input": float(index) + 0.25},
                parent_case_id=root_id,
            )
            records.extend(
                (
                    CaseRecord(root, _Truth(float(index) if self.vary_truth else 1.0)),
                    CaseRecord(
                        child, _Truth(float(index) + 0.25 if self.vary_truth else 1.0)
                    ),
                )
            )
        self.returned_root_ids.extend(roots)
        cohort = PreparedMacroCohort.prepare(
            cohort_id=f"development-{request.draw_index}",
            records=records,
            matcher=_Matcher(),
            query=QUERY,
            private_store_root=self.private_store_root,
        )
        self.cohorts.append(cohort)
        if cohort.private_manifest_path is not None:
            self.store_roots.append(cohort.private_manifest_path.parent)
        return cohort


def _submission(
    label: str,
    program: _Program,
    *,
    parent_sha256: str | None,
    round_index: int = 1,
    digest: str | None = None,
) -> CandidateSubmission:
    return CandidateSubmission(
        CandidateRef(
            candidate_id=label,
            bundle_sha256=digest or _digest(label),
            round_index=round_index,
            agent_id="agent-a",
            parent_sha256=parent_sha256,
        ),
        program,
    )


def _session(
    *,
    screen_credits: int = 32,
    release_credits: int = 4,
    private_store_root=None,
    open_search: bool = False,
    release_order_rng=None,
    vary_truth: bool = False,
):
    factory = _FreshPairedFactory(private_store_root, vary_truth=vary_truth)
    incumbent_program = _Program(0.5)
    incumbent = _submission(
        "incumbent",
        incumbent_program,
        parent_sha256=None,
        round_index=0,
    )
    session = DevelopmentSession(
        incumbent=incumbent,
        cohort_factory=factory,
        screen_root_stages=(4, 16),
        screen_root_credit_limit=screen_credits,
        release_root_credit_limit=release_credits,
        safe_metric_names=("component_error",),
        confidence_z=1.96,
        open_search=open_search,
        release_order_rng=release_order_rng,
    )
    return session, factory, incumbent, incumbent_program


def test_screen_adds_fresh_roots_and_pairs_candidate_with_incumbent():
    session, factory, incumbent, incumbent_program = _session()
    candidate_program = _Program(0.25)
    candidate = _submission(
        "candidate",
        candidate_program,
        parent_sha256=incumbent.ref.bundle_sha256,
    )

    screen = session.screen(candidate, requested_root_count=4)

    assert [request.root_count for request in factory.requests] == [4]
    assert candidate_program.case_ids == incumbent_program.case_ids
    assert len(candidate_program.case_ids) == 8
    assert screen.completed_root_count == 4
    assert screen.member_case_count == 8
    assert screen.charged_root_credits == 4
    public = screen.public_payload()
    assert public["candidate"]["loss"]["mean"] == pytest.approx(0.25)
    assert public["incumbent"]["loss"]["mean"] == pytest.approx(0.5)
    assert public["candidate"]["metrics"]["component_error"]["mean"] == pytest.approx(0.25)
    assert "private_metric" not in public["candidate"]["metrics"]
    assert public["paired"]["mean_improvement"] == pytest.approx(0.25)
    assert public["paired"]["win_rate"] == 1.0

    deeper = session.screen(candidate, requested_root_count=16)

    assert [request.root_count for request in factory.requests] == [4, 12]
    assert [request.cumulative_screen_root_count for request in factory.requests] == [
        4,
        16,
    ]
    assert candidate_program.case_ids == incumbent_program.case_ids
    assert len(candidate_program.case_ids) == 32
    assert deeper.completed_stage_root_counts == (4, 16)
    assert deeper.completed_root_count == 16
    assert deeper.member_case_count == 32
    assert deeper.charged_root_credits == 12
    assert session.screen_charged_root_credits == 16
    assert [row.new_root_count for row in session.screen_ledger] == [4, 12]
    assert len(factory.returned_root_ids) == len(set(factory.returned_root_ids))


def test_screen_evaluates_candidate_and_incumbent_case_major():
    events = []
    factory = _FreshPairedFactory()
    incumbent_program = _OrderedProgram(0.5, "incumbent", events)
    incumbent = _submission(
        "case-major-incumbent",
        incumbent_program,
        parent_sha256=None,
        round_index=0,
    )
    session = DevelopmentSession(
        incumbent=incumbent,
        cohort_factory=factory,
        screen_root_stages=(2,),
        screen_root_credit_limit=2,
        release_root_credit_limit=0,
        confidence_z=0.0,
    )
    candidate_program = _OrderedProgram(0.25, "candidate", events)
    candidate = _submission(
        "case-major-candidate",
        candidate_program,
        parent_sha256=incumbent.ref.bundle_sha256,
    )

    session.screen(candidate, requested_root_count=2)

    assert candidate_program.case_ids == incumbent_program.case_ids
    assert events == [
        event
        for case_id in candidate_program.case_ids
        for event in ((case_id, "candidate"), (case_id, "incumbent"))
    ]


def test_generated_primary_uses_perturbed_distribution_as_guardrail():
    factory = _FreshPairedFactory()
    incumbent = _submission(
        "incumbent-split",
        _SplitProgram(0.5, 0.5),
        parent_sha256=None,
        round_index=0,
    )
    policy = PairedObjectivePolicy(
        mode=PairedObjectiveMode.PRIMARY_WITH_PERTURBED_GUARDRAIL,
        guardrail_relative_margin=0.05,
        guardrail_confidence_z=1.6448536269514722,
    )
    session = DevelopmentSession(
        incumbent=incumbent,
        cohort_factory=factory,
        screen_root_stages=(4, 16),
        screen_root_credit_limit=32,
        release_root_credit_limit=4,
        safe_metric_names=("component_error",),
        confidence_z=1.6448536269514722,
        paired_objective_policy=policy,
    )
    aggressive = _submission(
        "aggressive",
        _SplitProgram(0.1, 0.6),
        parent_sha256=incumbent.ref.bundle_sha256,
    )

    public = session.screen(aggressive, requested_root_count=4).public_payload()

    assert public["candidate"]["loss"]["mean"] == pytest.approx(0.1)
    distributions = public["candidate"]["distributions"]
    assert distributions["generated"]["loss"]["mean"] == pytest.approx(0.1)
    assert distributions["perturbed"]["loss"]["mean"] == pytest.approx(0.6)
    assert distributions["composite"]["loss"]["mean"] == pytest.approx(0.35)
    assert public["paired"]["mean_improvement"] == pytest.approx(0.4)
    assert public["perturbed_guardrail"]["passed"] is False
    assert "perturbed_regression_beyond_margin" in (
        public["perturbed_guardrail"]["failures"]
    )
    assert session.screened_candidate_summaries[0].eligible is True

    acceptable = _submission(
        "acceptable",
        _SplitProgram(0.1, 0.52),
        parent_sha256=incumbent.ref.bundle_sha256,
    )
    accepted = session.screen(acceptable, requested_root_count=4).public_payload()

    assert accepted["candidate"]["loss"]["mean"] == pytest.approx(0.1)
    assert accepted["perturbed_guardrail"]["passed"] is True
    summaries = {
        row.candidate.ref.candidate_id: row
        for row in session.screened_candidate_summaries
    }
    assert summaries["acceptable"].eligible is True
    assert summaries["acceptable"].generated_loss_mean == pytest.approx(0.1)
    assert summaries["acceptable"].perturbed_loss_mean == pytest.approx(0.52)

    assert summaries["aggressive"].eligible is True

    session.screen(aggressive, requested_root_count=16)
    session.screen(acceptable, requested_root_count=16)
    full_stage = {
        row.candidate.ref.candidate_id: row
        for row in session.screened_candidate_summaries
    }
    assert full_stage["aggressive"].eligible is False
    assert full_stage["acceptable"].eligible is True

def test_digest_cache_prevents_relabel_reroll_and_uses_original_program():
    session, factory, incumbent, _ = _session()
    digest = _digest("same-sealed-bytes")
    original_program = _Program(0.2)
    original = _submission(
        "original-label",
        original_program,
        parent_sha256=incumbent.ref.bundle_sha256,
        digest=digest,
    )
    relabel_program = _Program(0.9)
    relabel = _submission(
        "different-label",
        relabel_program,
        parent_sha256=incumbent.ref.bundle_sha256,
        digest=digest,
    )

    first = session.screen(original, requested_root_count=4)
    cached = session.screen(relabel, requested_root_count=4)

    assert not first.cached
    assert cached.cached
    assert cached.charged_root_credits == 0
    assert cached.public_payload()["candidate"]["loss"]["mean"] == pytest.approx(
        0.2
    )
    assert len(factory.requests) == 1
    assert relabel_program.case_ids == []

    session.screen(relabel, requested_root_count=16)

    assert [request.root_count for request in factory.requests] == [4, 12]
    assert len(original_program.case_ids) == 32
    assert relabel_program.case_ids == []
    assert session.screen_ledger[1].cached
    assert session.screen_ledger[1].draw is None
    assert len(session.screened_candidates) == 1
    assert session.screened_candidates[0].ref.candidate_id == "original-label"


def test_screen_public_payload_contains_only_aggregate_feedback():
    session, factory, incumbent, _ = _session()
    candidate = _submission(
        "candidate",
        _Program(0.25),
        parent_sha256=incumbent.ref.bundle_sha256,
    )

    response = session.screen(candidate, requested_root_count=4)
    public = response.public_payload()
    serialized = json.dumps(public, sort_keys=True)

    assert public["private_case_ids_included"] is False
    assert public["per_case_errors_included"] is False
    assert public["finite_truth_included"] is False
    assert public["official_promotion_result"] is False
    assert "cases" not in public
    assert "cohort_id" not in public
    assert "cohort_sha256" not in public
    assert "target_sha256" not in public
    for root_id in factory.returned_root_ids:
        assert root_id not in serialized
    assert "private_metric" not in serialized


def test_open_search_uses_fresh_arbitrary_draws_without_lifetime_quota():
    session, factory, incumbent, _ = _session(
        screen_credits=5,
        release_credits=3,
        open_search=True,
    )
    candidate = _submission(
        "open-candidate",
        _Program(0.25),
        parent_sha256=incumbent.ref.bundle_sha256,
    )

    first = session.screen(candidate, requested_root_count=3)
    second = session.screen(candidate, requested_root_count=3)

    assert [request.root_count for request in factory.requests] == [3, 3]
    assert first.completed_root_count == 3
    assert second.completed_root_count == 6
    assert second.completed_stage_root_counts == (3, 6)
    assert not first.cached and not second.cached
    assert session.screen_charged_root_credits == 6
    assert session.screen_remaining_root_credits == 5
    accounting = second.public_payload()["accounting"]
    assert accounting["protocol_lifetime_limit"] is False
    assert accounting["per_request_root_limit"] == 5

    first_release = session.release(2)
    second_release = session.release(2)

    assert first_release.release_index == 1
    assert second_release.release_index == 2
    assert session.release_charged_root_credits == 4
    assert session.release_remaining_root_credits == 3
    assert [request.root_count for request in factory.requests] == [3, 3, 2, 2]
    before = len(factory.requests)
    with pytest.raises(DevelopmentCreditExceeded, match="per-request capacity"):
        session.screen(candidate, requested_root_count=6)
    with pytest.raises(DevelopmentCreditExceeded, match="per-request capacity"):
        session.release(4)
    assert len(factory.requests) == before


def test_completed_answer_release_has_independent_credit_and_ledger():
    session, factory, incumbent, _ = _session(
        screen_credits=4,
        release_credits=3,
    )

    released = session.release(2)

    assert released.root_count == 2
    assert released.member_case_count == 4
    assert len(released.cases) == len(released.queries) == len(released.answers) == 4
    assert all(answer.fields["macro"] == (1.0,) for answer in released.answers)
    assert session.release_charged_root_credits == 2
    assert session.screen_charged_root_credits == 0
    assert len(session.release_ledger) == 1
    assert session.screen_ledger == ()
    assert session.release_ledger[0].draw.kind is DevelopmentDrawKind.RELEASE
    assert session.latest_release == released
    assert session.public_cases == released.cases
    assert session.public_queries == released.queries
    assert session.public_answers == released.answers
    assert session.cohort_id == released.cohort_id
    assert session.cohort_sha256 == released.cohort_sha256
    assert session.contract_id == released.contract_id

    candidate = _submission(
        "candidate",
        _Program(0.25),
        parent_sha256=incumbent.ref.bundle_sha256,
    )
    session.screen(candidate, requested_root_count=4)

    assert session.release_charged_root_credits == 2
    assert session.screen_charged_root_credits == 4
    assert session.release_ledger[0].draw.cohort_sha256 != (
        session.screen_ledger[0].draw.cohort_sha256
    )
    assert {request.kind for request in factory.requests} == {
        DevelopmentDrawKind.RELEASE,
        DevelopmentDrawKind.SCREEN,
    }
    with pytest.raises(DevelopmentCreditExceeded, match="only 1 remain"):
        session.release(2)
    assert session.release_charged_root_credits == 2
    assert session.screen_charged_root_credits == 4


def test_release_uses_one_private_order_for_public_and_controller_rows():
    order = (2, 0, 3, 1)
    session, _factory, _incumbent, _program = _session(
        release_credits=2,
        release_order_rng=_ScriptedReleaseOrderRng(order),
        vary_truth=True,
    )

    released = session.release(2)
    raw_cases = session._controller_release_cases()
    inputs = [case.public_data["input"] for case in released.cases]
    assert inputs == [1.0, 0.0, 1.25, 0.25]
    assert (
        tuple(case.case_id for case in released.cases)
        == tuple(candidate_case_id(case.case_id) for case in raw_cases)
    )
    assert [answer.fields["macro"][0] for answer in released.answers] == inputs
    assert released.queries == (QUERY,) * len(order)
    assert [case.parent_case_id for case in raw_cases] == [
        None, None, raw_cases[0].case_id, raw_cases[1].case_id
    ]
    assert all(case.parent_case_id is None for case in released.cases)

    released.cases[0].public_data["input"] = -999.0
    assert session._controller_release_cases()[0].public_data["input"] == 1.0


def test_disk_backed_release_and_screen_discard_temporary_cohorts(tmp_path):
    session, factory, incumbent, _ = _session(
        screen_credits=4,
        release_credits=2,
        private_store_root=tmp_path / "cohorts",
    )

    released = session.release(2)

    assert released.answers
    assert session.public_answers == released.answers
    assert len(factory.cohorts) == len(factory.store_roots) == 1
    assert factory.store_roots[0].exists()
    assert factory.cohorts[0].public_cases()

    session.discard_latest_release()
    assert session.latest_release is None
    assert not factory.store_roots[0].exists()
    with pytest.raises(RuntimeError, match="discarded"):
        factory.cohorts[0].public_cases()
    assert released.answers
    with pytest.raises(DevelopmentSessionError, match="call release"):
        _ = session.public_answers

    candidate = _submission(
        "candidate",
        _Program(0.25),
        parent_sha256=incumbent.ref.bundle_sha256,
    )
    session.screen(candidate, requested_root_count=4)

    assert len(factory.cohorts) == len(factory.store_roots) == 2
    assert all(not root.exists() for root in factory.store_roots)
    with pytest.raises(RuntimeError, match="discarded"):
        factory.cohorts[1].public_cases()


def test_screen_stages_and_credits_fail_closed_before_a_new_draw():
    session, factory, incumbent, _ = _session(screen_credits=4)
    candidate = _submission(
        "candidate",
        _Program(0.25),
        parent_sha256=incumbent.ref.bundle_sha256,
    )

    with pytest.raises(DevelopmentSessionError, match="exactly one"):
        session.screen(candidate, requested_root_count=16)
    assert factory.requests == []

    session.screen(candidate, requested_root_count=4)
    with pytest.raises(DevelopmentCreditExceeded, match="only 0 remain"):
        session.screen(candidate, requested_root_count=16)
    assert [request.root_count for request in factory.requests] == [4]


def test_bad_candidate_cannot_continue_to_fresh_stage_or_spend_credit():
    session, factory, incumbent, _ = _session()
    candidate = _submission(
        "bad-candidate",
        _Program(0.8),
        parent_sha256=incumbent.ref.bundle_sha256,
    )

    session.screen(candidate, requested_root_count=4)

    with pytest.raises(
        DevelopmentSessionError,
        match="continuation mean-ratio gate",
    ):
        session.screen(candidate, requested_root_count=16)

    assert [request.root_count for request in factory.requests] == [4]
    assert session.screen_charged_root_credits == 4
    assert session.screen_remaining_root_credits == 28
    assert len(session.screen_ledger) == 1


def test_candidate_must_be_attested_to_the_session_incumbent():
    session, factory, _, _ = _session()
    candidate = _submission(
        "wrong-parent",
        _Program(0.25),
        parent_sha256=_digest("other-parent"),
    )

    with pytest.raises(DevelopmentSessionError, match="session incumbent"):
        session.screen(candidate, requested_root_count=4)
    assert factory.requests == []


def test_close_drains_other_cohorts_and_retries_failed_release():
    session, factory, _, _ = _session()
    first = factory(DevelopmentDrawRequest(DevelopmentDrawKind.RELEASE, 101, 1))
    second = factory(DevelopmentDrawRequest(DevelopmentDrawKind.RELEASE, 102, 1))
    session._active_cohorts.update((first, second))
    calls = {"first": 0, "second": 0}

    def fail_once(_case_ids):
        calls["first"] += 1
        if calls["first"] == 1:
            raise RuntimeError("transient backend close")

    def succeed(_case_ids):
        calls["second"] += 1

    first._release_cases = fail_once
    second._release_cases = succeed
    with pytest.raises(RuntimeError, match="transient backend close"):
        session.close()

    assert first in session._active_cohorts
    assert second not in session._active_cohorts
    assert calls == {"first": 1, "second": 1}

    session.close()
    assert session._active_cohorts == set()
    assert calls == {"first": 2, "second": 1}
