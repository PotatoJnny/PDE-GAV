import hashlib
import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from pde_agent_verifier_solver.engine.candidate_case import candidate_case_id

from pde_agent_verifier_solver.engine.macro_closed_loop import (
    MacroAgentDevelopmentResult,
    MacroAgentOutcome,
    MacroAgentSpec,
    MacroClosedLoopRunner,
    MacroLoopConfig,
    verify_round_handoff,
)
from pde_agent_verifier_solver.engine.macro_evaluator import (
    ControllerMacroEvaluator,
    MacroBatchStreamAccumulator,
    PreparedMacroCohort,
)
from pde_agent_verifier_solver.engine.selection import (
    EvidenceStage,
    PairedObjectiveMode,
    PairedObjectivePolicy,
)
from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    AnswerView,
    Case,
    ExecutionAssurance,
    FormalExecutionUnavailable,
    CaseRecord,
    CaseSource,
    CaseVisibility,
    GenerationKind,
    GeneratorCapabilities,
    MacroMatchResult,
    MacroTarget,
    PanelCard,
    ProgramInfo,
    QueryKind,
    RecoverableSamplingRejection,
)
from pde_agent_verifier_solver.sdk.macro_loop import (
    BoundMacroObjective,
    CandidateRef,
    CandidateSubmission,
    CohortKind,
    FixedGeneratorCaseSource,
    MacroLoopPlugin,
    SampleRequest,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class _ScalarTruth:
    calls = 0

    def answer(self, query):
        type(self).calls += 1
        return AnswerView(
            kind=query.kind,
            fields={"macro": (1.0,)},
            axes={"sample": (0.0,)},
            shape=(1,),
        )


class _FixedCaseSource:
    generator_id = "already-trained-generator"
    artifact_sha256 = "a" * 64

    def __init__(self):
        self.requests = []

    def sample(self, request):
        self.requests.append(request)
        root_offset = int(
            request.settings.get("logical_root_offset", 0)
        )
        return tuple(
            CaseRecord(
                Case(
                    case_id=(
                        f"{request.round_index}-{request.cohort.value}-"
                        f"{request.seed}-{index}"
                    ),
                    source=CaseSource.GENERATED,
                    visibility=(
                        CaseVisibility.DEVELOPMENT
                        if request.cohort is CohortKind.DEVELOPMENT
                        else CaseVisibility.HIDDEN
                    ),
                    public_data={"input": float(root_offset + index)},
                    round_index=request.round_index,
                ),
                _ScalarTruth(),
            )
            for index in range(request.count)
        )


class _AbsoluteMacroMatcher:
    info = ProgramInfo("absolute", "Absolute macro objective", "test")
    contract_id = "equation_a_macro_v1"
    fields = ("macro",)

    def prepare_target(self, case, truth, query):
        return MacroTarget(case.case_id, self.contract_id, query, truth)

    def match(self, case, prediction, target):
        error = abs(prediction.fields["macro"][0] - target.values.fields["macro"][0])
        return MacroMatchResult(
            case.case_id,
            self.contract_id,
            error,
            metrics={"equation_specific_error": error},
        )

    def result_card(self, result):
        return PanelCard("absolute", "test", {"error": result.primary_error})


class _SquaredMacroMatcher(_AbsoluteMacroMatcher):
    info = ProgramInfo("squared", "Squared macro objective", "test")
    contract_id = "equation_b_macro_v1"

    def match(self, case, prediction, target):
        difference = prediction.fields["macro"][0] - target.values.fields["macro"][0]
        error = difference * difference
        return MacroMatchResult(
            case.case_id,
            self.contract_id,
            error,
            metrics={"different_equation_specific_error": error},
        )


class _PublishedMetricMatcher(_AbsoluteMacroMatcher):
    contract_id = "published_component_macro_v1"

    def match(self, case, prediction, target):
        error = abs(
            prediction.fields["macro"][0] - target.values.fields["macro"][0]
        )
        factor = int(case.public_data["input"]) + 1
        metrics = {
            "public_component": error * factor,
            "private_component": 10_000.0 + factor,
        }
        if factor < 4:
            metrics["incomplete_component"] = error + factor
        return MacroMatchResult(
            case.case_id,
            self.contract_id,
            error,
            metrics=metrics,
        )


class _Objective:
    def __init__(
        self,
        matcher,
        *,
        agent_visible_metrics=(),
        published_metric_names=(),
    ):
        self.matcher = matcher
        self.agent_visible_metrics = tuple(agent_visible_metrics)
        self.published_metric_names = tuple(published_metric_names)

    def query(self, case, profile):
        return AnswerQuery.grid(fields=("macro",), sample=1)


class _ProfileRecordingObjective(_Objective):
    def __init__(self, matcher):
        super().__init__(matcher)
        self.query_profiles = []

    def query(self, case, profile):
        self.query_profiles.append((case.case_id, profile))
        return super().query(case, profile)


class _Approximation:
    def __init__(self, error):
        self.error = error

    def predict_macros(self, case, query, budget=None):
        return AnswerView(
            kind=QueryKind.GRID,
            fields={"macro": (1.0 - self.error,)},
            axes={"sample": (0.0,)},
            shape=(1,),
        )


class _ImprovingAgent:
    def __init__(self):
        self.terminal_ids_seen = []
        self.contexts = []

    def run(self, context):
        self.contexts.append(context)
        self.terminal_ids_seen.extend(
            case.case_id for case in context.cases if "terminal" in case.case_id
        )
        error = max(0.05, 0.5 - 0.15 * context.round_index)
        candidate_id = f"candidate-r{context.round_index}"
        ref = CandidateRef(
            candidate_id=candidate_id,
            bundle_sha256=_digest(f"{candidate_id}:{error}"),
            round_index=context.round_index,
            agent_id=context.agent.agent_id,
            parent_sha256=context.incumbent.bundle_sha256,
        )
        candidate = CandidateSubmission(ref, _Approximation(error))
        response = context.evaluator.evaluate(
            candidate_id,
            candidate.program,
            case_ids=tuple(case.case_id for case in context.cases),
        )
        if any(row.score.status != "valid" for row in response.cases):
            raise AssertionError("development candidate unexpectedly failed")
        return MacroAgentOutcome(candidate)


class _RecordingApproximation(_Approximation):
    def __init__(self, error):
        super().__init__(error)
        self.case_ids = []

    def predict_macros(self, case, query, budget=None):
        self.case_ids.append(case.case_id)
        return super().predict_macros(case, query, budget=budget)


class _StrongScreenAgent:
    def __init__(self, error):
        self.error = error
        self.release_case_ids = ()
        self.release_answers = ()
        self.screen_payload = None
        self.screen_case_ids = ()
        self.program = None

    def run(self, context):
        self.release_case_ids = tuple(case.case_id for case in context.cases)
        self.release_answers = context.evaluator.public_answers
        candidate_id = f"{context.agent.agent_id}-r{context.round_index}"
        self.program = _RecordingApproximation(self.error)
        candidate = CandidateSubmission(
            CandidateRef(
                candidate_id=candidate_id,
                bundle_sha256=_digest(f"{candidate_id}:{self.error}"),
                round_index=context.round_index,
                agent_id=context.agent.agent_id,
                parent_sha256=context.incumbent.bundle_sha256,
            ),
            self.program,
        )
        screen = context.evaluator.screen(
            candidate,
            requested_root_count=2,
        )
        self.screen_payload = screen.public_payload()
        self.screen_case_ids = tuple(self.program.case_ids)
        return MacroAgentOutcome(candidate)


class _PortfolioScreenAgent:
    def __init__(self, errors, *, fail=False, digests=()):
        self.errors = tuple(errors)
        self.fail = fail
        self.digests = tuple(digests)
        self.candidates = []

    def run(self, context):
        for index, error in enumerate(self.errors):
            candidate_id = (
                f"{context.agent.agent_id}-portfolio-{index}"
            )
            digest = (
                self.digests[index]
                if index < len(self.digests)
                else _digest(f"{candidate_id}:{error}")
            )
            candidate = CandidateSubmission(
                CandidateRef(
                    candidate_id,
                    digest,
                    context.round_index,
                    context.agent.agent_id,
                    parent_sha256=context.incumbent.bundle_sha256,
                ),
                _Approximation(error),
            )
            context.evaluator.screen(
                candidate,
                requested_root_count=2,
            )
            self.candidates.append(candidate)
        if self.fail:
            raise TimeoutError("synthetic Agent timeout after screening")
        return MacroAgentOutcome(self.candidates[-2])


class _StatefulTruth:
    def __init__(self, source):
        self.source = source

    def answer(self, query):
        self.source.truth_answer_calls += 1
        return AnswerView(
            kind=query.kind,
            fields={"macro": (1.0,)},
            axes={"sample": (0.0,)},
            shape=(1,),
        )


class _StatefulCaseSource:
    generator_id = "stateful-fixed-generator"
    artifact_sha256 = "f" * 64

    def __init__(self):
        self.requests = []
        self.sample_call_count = 0
        self.truth_answer_calls = 0
        self.generated_ids = []

    def sample(self, request):
        self.requests.append(request)
        self.sample_call_count += 1
        call_index = self.sample_call_count
        records = []
        for index in range(request.count):
            case_id = (
                f"call-{call_index}-r{request.round_index}-"
                f"{request.cohort.value}-{index}"
            )
            self.generated_ids.append(case_id)
            records.append(
                CaseRecord(
                    Case(
                        case_id=case_id,
                        source=CaseSource.GENERATED,
                        visibility=(
                            CaseVisibility.DEVELOPMENT
                            if request.cohort is CohortKind.DEVELOPMENT
                            else CaseVisibility.HIDDEN
                        ),
                        public_data={
                            "input": float(index),
                            "cohort": request.cohort.value,
                        },
                        round_index=request.round_index,
                        tags=(f"cohort:{request.cohort.value}",),
                    ),
                    _StatefulTruth(self),
                )
            )
        return tuple(records)


class _LegacyMixedCaseSource(_StatefulCaseSource):
    """Pre-existing source that returns composed perturbations to BASE calls."""

    def sample(self, request):
        records = super().sample(request)
        return tuple(
            CaseRecord(
                replace(
                    record.case,
                    source=CaseSource.PERTURBED,
                    parent_case_id=f"{record.case.case_id}-legacy-root",
                ),
                record.truth,
            )
            for record in records
        )


class _PairedCaseSource:
    generator_id = "paired-fixed-generator"
    artifact_sha256 = "0" * 64
    supports_perturbations = True

    def __init__(self, *, invalid_child=None, pair_difficulties=()):
        self.invalid_child = invalid_child
        self.pair_difficulties = tuple(pair_difficulties)
        self.requests = []

    def sample(self, request):
        self.requests.append(request)
        visibility = (
            CaseVisibility.DEVELOPMENT
            if request.cohort is CohortKind.DEVELOPMENT
            else CaseVisibility.HIDDEN
        )
        if request.kind is GenerationKind.BASE:
            root_offset = int(
                request.settings.get("logical_root_offset", 0)
            )
            return tuple(
                CaseRecord(
                    Case(
                        case_id=(
                            f"r{request.round_index}-{request.cohort.value}-"
                            f"root-{root_offset + index}"
                        ),
                        source=CaseSource.GENERATED,
                        visibility=visibility,
                        public_data={
                            "cohort": request.cohort.value,
                            "difficulty": self._difficulty(
                                root_offset + index,
                                child=False,
                            ),
                            "logical_root_index": root_offset + index,
                        },
                        round_index=request.round_index,
                        tags=(f"cohort:{request.cohort.value}",),
                    ),
                    _ScalarTruth(),
                )
                for index in range(request.count)
            )
        if request.kind is not GenerationKind.PERTURBED:
            raise AssertionError(f"unexpected generation kind: {request.kind}")
        records = []
        for index, anchor in enumerate(request.anchors):
            root_index = int(anchor.public_data["logical_root_index"])
            parent_case_id = anchor.case_id
            source = CaseSource.PERTURBED
            if index == 0 and self.invalid_child == "parent":
                parent_case_id = request.anchors[-1].case_id
            if index == 0 and self.invalid_child == "source":
                source = CaseSource.GENERATED
            records.append(
                CaseRecord(
                    Case(
                        case_id=f"{anchor.case_id}-perturbed",
                        source=source,
                        visibility=visibility,
                        public_data={
                            "cohort": request.cohort.value,
                            "difficulty": self._difficulty(
                                root_index,
                                child=True,
                            ),
                            "logical_root_index": root_index,
                        },
                        round_index=request.round_index,
                        parent_case_id=parent_case_id,
                        tags=(f"cohort:{request.cohort.value}",),
                    ),
                    _ScalarTruth(),
                )
            )
        return tuple(records)

    def _difficulty(self, index, *, child):
        if index >= len(self.pair_difficulties):
            return float(index + (0.5 if child else 0.0))
        root, perturbed = self.pair_difficulties[index]
        return float(perturbed if child else root)


class _ReleasingPairedCaseSource(_PairedCaseSource):
    def __init__(self):
        super().__init__()
        self.live_case_ids = set()
        self.maximum_live_case_count = 0
        self.release_batches = []

    def sample(self, request):
        records = super().sample(request)
        case_ids = {record.case.case_id for record in records}
        if case_ids & self.live_case_ids:
            raise AssertionError("test source repeated a live case")
        self.live_case_ids.update(case_ids)
        self.maximum_live_case_count = max(
            self.maximum_live_case_count,
            len(self.live_case_ids),
        )
        return records

    def release_cases(self, case_ids):
        released = tuple(case_ids)
        if not set(released) <= self.live_case_ids:
            raise AssertionError("test source released an unknown case")
        self.release_batches.append(released)
        self.live_case_ids.difference_update(released)


class _RecoverablyRejectingPairedCaseSource(_PairedCaseSource):
    def __init__(self, *, development_rejections: int):
        super().__init__()
        self.development_rejections = development_rejections
        self.rejected_root_ids = []

    def sample(self, request):
        records = super().sample(request)
        if request.kind is GenerationKind.BASE:
            return tuple(
                CaseRecord(
                    replace(
                        record.case,
                        case_id=f"{record.case.case_id}-seed-{request.seed}",
                    ),
                    record.truth,
                )
                for record in records
            )
        if (
            request.cohort is CohortKind.DEVELOPMENT
            and self.development_rejections > 0
        ):
            self.development_rejections -= 1
            self.rejected_root_ids.append(
                tuple(anchor.case_id for anchor in request.anchors)
            )
            raise RecoverableSamplingRejection(
                "synthetic equation acceptance rejection",
                audit={
                    "schema": "synthetic_sampling_rejection_v1",
                    "perturbed_seed": request.seed,
                    "failed_gates": ["synthetic_gate"],
                },
            )
        return records


class _PrecomputeObservingAgent(_ImprovingAgent):
    def __init__(self, source, *, fail=False):
        super().__init__()
        self.source = source
        self.fail = fail
        self.pre_run_snapshots = []

    def run(self, context):
        self.pre_run_snapshots.append(
            (
                tuple(
                    (request.round_index, request.cohort, request.seed, request.count)
                    for request in self.source.requests
                ),
                self.source.truth_answer_calls,
            )
        )
        if self.fail:
            raise RuntimeError("intentional Agent failure")
        return super().run(context)


class _FrozenGenerator:
    capabilities = GeneratorCapabilities(
        offline=True,
        perturbations=True,
    )

    def __init__(self):
        self.generate_calls = []

    def train_base(self, *args, **kwargs):
        raise AssertionError("the closed loop must not train the Generator")

    def fit_guidance(self, *args, **kwargs):
        raise AssertionError("the closed loop must not update Generator guidance")

    def generate(self, base_state, guidance_state, request, progress=None):
        self.generate_calls.append((base_state, guidance_state, request))
        return tuple(
            CaseRecord(
                Case(
                    case_id=f"fixed-{request.settings['seed']}-{index}",
                    source=(
                        CaseSource.PERTURBED
                        if request.kind is GenerationKind.PERTURBED
                        else CaseSource.GENERATED
                    ),
                    visibility=request.visibility,
                    public_data={"input": index},
                    round_index=request.round_index,
                    parent_case_id=(
                        request.anchors[index].case_id
                        if request.kind is GenerationKind.PERTURBED
                        else None
                    ),
                ),
                _ScalarTruth(),
            )
            for index in range(request.count)
        )

    def preview(self, record):
        return PanelCard("fixed", "fixed", {})


class MacroClosedLoopTests(unittest.TestCase):
    def _run(self, matcher, *, published_metric_names=()):
        source = _FixedCaseSource()
        plugin = MacroLoopPlugin(
            plugin_id=matcher.contract_id,
            cases=source,
            objective=_Objective(
                matcher,
                published_metric_names=published_metric_names,
            ),
        )
        cold_ref = CandidateRef(
            candidate_id="cold-start",
            bundle_sha256="c" * 64,
            round_index=0,
            agent_id="frozen-baseline",
        )
        agent = _ImprovingAgent()
        config = MacroLoopConfig(
            name="shared-protocol-test",
            round_count=2,
            agents=(MacroAgentSpec("agent", "Agent"),),
            development_case_count=4,
            selection_case_count=4,
            evidence_stages=(
                EvidenceStage(
                    2,
                    confidence_z=0.0,
                    min_win_rate=0.5,
                    require_cvar_no_regression=True,
                ),
                EvidenceStage(
                    4,
                    confidence_z=0.0,
                    min_win_rate=0.5,
                    require_cvar_no_regression=True,
                ),
            ),
            terminal_case_count=4,
            agent_case_credits=4,
            seed_base=100,
        )
        result = MacroClosedLoopRunner(
            legacy_macro_reference=True,
            config=config,
            plugin=plugin,
            cold_start=CandidateSubmission(cold_ref, _Approximation(0.5)),
            agents={"agent": agent},
        ).run()
        return result, source, agent

    def _stateful_runner(self, *, fail, source=None):
        source = source or _StatefulCaseSource()
        agent = _PrecomputeObservingAgent(source, fail=fail)
        matcher = _AbsoluteMacroMatcher()
        cold_ref = CandidateRef(
            "cold-start",
            "9" * 64,
            0,
            "baseline",
        )
        config = MacroLoopConfig(
            name="stateful-precompute",
            round_count=2,
            agents=(MacroAgentSpec("agent", "Agent"),),
            development_case_count=2,
            selection_case_count=2,
            evidence_stages=(EvidenceStage(2, confidence_z=0.0),),
            terminal_case_count=2,
            agent_case_credits=2,
            seed_base=50,
            development_pool_multiplier=2,
        )
        runner = MacroClosedLoopRunner(
            legacy_macro_reference=True,
            config=config,
            plugin=MacroLoopPlugin("stateful", source, _Objective(matcher)),
            cold_start=CandidateSubmission(cold_ref, _Approximation(0.5)),
            agents={"agent": agent},
        )
        return runner, source, agent

    def test_strong_development_uses_fresh_agent_and_candidate_roots(self):
        source = _RecoverablyRejectingPairedCaseSource(
            development_rejections=0
        )
        objective = _ProfileRecordingObjective(_AbsoluteMacroMatcher())
        plugin = MacroLoopPlugin(
            "strong-development",
            source,
            objective,
        )
        cold_ref = CandidateRef(
            "strong-cold",
            "8" * 64,
            0,
            "baseline",
        )
        agents = {
            "strong-a": _StrongScreenAgent(0.3),
            "strong-b": _StrongScreenAgent(0.2),
        }
        config = MacroLoopConfig(
            name="strong-development",
            round_count=1,
            agents=(
                MacroAgentSpec("strong-a", "Strong A"),
                MacroAgentSpec("strong-b", "Strong B"),
            ),
            development_case_count=4,
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
            agent_case_credits=4,
            seed_base=100,
            seed_stride=100,
            development_release_root_limit=2,
            development_screen_root_stages=(2, 4),
            development_screen_root_credit_limit=8,
            development_screen_confidence_z=0.0,
            terminal_confidence_z=0.0,
            terminal_min_win_rate=0.0,
            terminal_require_cvar_no_regression=False,
            paired_perturbations=True,
        )

        result = MacroClosedLoopRunner(
            legacy_macro_reference=True,
            config=config,
            plugin=plugin,
            cold_start=CandidateSubmission(
                cold_ref,
                _Approximation(0.5),
            ),
            agents=agents,
        ).run()

        left = agents["strong-a"]
        right = agents["strong-b"]
        self.assertEqual(len(left.release_case_ids), 4)
        self.assertEqual(len(right.release_case_ids), 4)
        self.assertTrue(
            set(left.release_case_ids).isdisjoint(right.release_case_ids)
        )
        self.assertEqual(len(left.release_answers), 4)
        self.assertTrue(
            all(answer.fields["macro"] == (1.0,) for answer in left.release_answers)
        )
        self.assertEqual(len(left.screen_case_ids), 4)
        self.assertEqual(len(right.screen_case_ids), 4)
        self.assertTrue(
            set(left.screen_case_ids).isdisjoint(right.screen_case_ids)
        )
        self.assertTrue(
            set(left.screen_case_ids).isdisjoint(left.release_case_ids)
        )
        for agent in (left, right):
            payload = agent.screen_payload
            self.assertFalse(payload["private_case_ids_included"])
            self.assertFalse(payload["per_case_errors_included"])
            self.assertFalse(payload["finite_truth_included"])
            serialized = json.dumps(payload, sort_keys=True)
            self.assertTrue(
                all(case_id not in serialized for case_id in agent.screen_case_ids)
            )
        self.assertEqual(len(result.rounds[0].development_case_ids), 8)
        self.assertEqual(result.selected.agent_id, "strong-b")
        base_requests = tuple(
            request
            for request in source.requests
            if request.kind is GenerationKind.BASE
        )
        self.assertEqual(
            sum(
                request.cohort is CohortKind.DEVELOPMENT
                for request in base_requests
            ),
            2,
        )
        self.assertEqual(
            sum(
                request.cohort is CohortKind.DEVELOPMENT_SCREEN
                for request in base_requests
            ),
            2,
        )
        self.assertEqual(
            len({request.seed for request in base_requests}),
            len(base_requests),
        )
        profiles_by_cohort = {
            cohort: {
                profile
                for case_id, profile in objective.query_profiles
                if f"-{cohort.value}-" in case_id
            }
            for cohort in (
                CohortKind.DEVELOPMENT,
                CohortKind.DEVELOPMENT_SCREEN,
                CohortKind.SELECTION,
                CohortKind.TERMINAL,
            )
        }
        self.assertEqual(
            profiles_by_cohort[CohortKind.DEVELOPMENT],
            {"development"},
        )
        for cohort in tuple(profiles_by_cohort)[1:]:
            self.assertEqual(profiles_by_cohort[cohort], {"terminal"})

    def test_one_official_nomination_per_agent_with_timeout_recovery(self):
        source = _RecoverablyRejectingPairedCaseSource(
            development_rejections=0
        )
        shared_digest = _digest("shared-screened-bundle")
        left = _PortfolioScreenAgent(
            (0.1, 0.2, 0.3, 0.9, 0.6),
            digests=(shared_digest,),
        )
        right = _PortfolioScreenAgent(
            (0.1, 0.15),
            fail=True,
            digests=(shared_digest,),
        )
        agents = {"portfolio-a": left, "portfolio-b": right}
        cold_ref = CandidateRef(
            "portfolio-cold",
            "7" * 64,
            0,
            "baseline",
        )
        config = MacroLoopConfig(
            name="strong-screened-portfolio",
            round_count=1,
            agents=(
                MacroAgentSpec("portfolio-a", "Portfolio A"),
                MacroAgentSpec("portfolio-b", "Portfolio B"),
            ),
            development_case_count=4,
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
            agent_case_credits=4,
            seed_base=910,
            seed_stride=100,
            development_release_root_limit=2,
            development_screen_root_stages=(2,),
            development_screen_root_credit_limit=10,
            development_screen_confidence_z=0.0,
            development_screen_representative_limit=3,
            terminal_confidence_z=0.0,
            terminal_min_win_rate=0.0,
            terminal_require_cvar_no_regression=False,
            paired_perturbations=True,
        )

        result = MacroClosedLoopRunner(
            legacy_macro_reference=True,
            config=config,
            plugin=MacroLoopPlugin(
                "strong-screened-portfolio",
                source,
                _Objective(_AbsoluteMacroMatcher()),
            ),
            cold_start=CandidateSubmission(
                cold_ref,
                _Approximation(0.5),
            ),
            agents=agents,
        ).run()

        round_result = result.rounds[0]
        self.assertEqual(
            round_result.agent_failures,
            {"portfolio-b": "agent_failed:TimeoutError"},
        )
        self.assertEqual(len(round_result.candidate_refs), 2)
        self.assertEqual(
            len({row.bundle_sha256 for row in round_result.candidate_refs}),
            2,
        )
        explicit = left.candidates[-2].ref
        recovered = right.candidates[0].ref
        self.assertEqual(
            {row.candidate_id for row in round_result.candidate_refs},
            {explicit.candidate_id, recovered.candidate_id},
        )
        self.assertTrue(
            all(
                candidate.ref.candidate_id
                not in {
                    row.candidate_id
                    for row in round_result.candidate_refs
                }
                for candidate in left.candidates
                if candidate.ref != explicit
            )
        )
        self.assertEqual(
            round_result.candidate_portfolio_roles,
            {
                explicit.candidate_id: ("agent_nomination",),
                recovered.candidate_id: ("recovered_checkpoint",),
            },
        )
        self.assertEqual(result.selected, recovered)
        selection_commitments = {
            (row.batch.cohort_sha256, row.batch.target_sha256, row.batch.case_ids)
            for row in round_result.finalist_evaluations
        }
        self.assertEqual(len(selection_commitments), 1)

    def test_development_transport_rejects_multiple_official_nominations(self):
        parent = "6" * 64

        def nomination(label, error):
            return CandidateSubmission(
                CandidateRef(
                    label,
                    _digest(label),
                    1,
                    "agent",
                    parent_sha256=parent,
                ),
                _Approximation(error),
            )

        with self.assertRaisesRegex(
            ValueError,
            "at most one official nomination",
        ):
            MacroAgentDevelopmentResult(
                agent_id="agent",
                portfolio=(
                    (nomination("first", 0.1), ("agent_nomination",)),
                    (nomination("second", 0.2), ("agent_nomination",)),
                ),
                development_case_ids=(),
                sampled_case_ids=(),
                sampling_attempts=(),
            )

    def test_publication_allowlist_is_independent_and_validated(self):
        query_factory = lambda case, profile: AnswerQuery.grid(
            fields=("macro",),
            sample=1,
        )
        objective = BoundMacroObjective(
            matcher=_AbsoluteMacroMatcher(),
            query_factory=query_factory,
            agent_visible_metrics=("agent_only",),
            published_metric_names=("aggregate_only",),
        )
        self.assertEqual(objective.agent_visible_metrics, ("agent_only",))
        self.assertEqual(
            objective.published_metric_names,
            ("aggregate_only",),
        )
        with self.assertRaisesRegex(ValueError, "published.*unique"):
            BoundMacroObjective(
                matcher=_AbsoluteMacroMatcher(),
                query_factory=query_factory,
                published_metric_names=("duplicate", "duplicate"),
            )
        with self.assertRaisesRegex(ValueError, "published metrics must be a sequence"):
            BoundMacroObjective(
                matcher=_AbsoluteMacroMatcher(),
                query_factory=query_factory,
                published_metric_names="not-a-sequence",
            )

    def test_allowlisted_component_aggregates_cover_each_evidence_boundary(self):
        result, _source, _agent = self._run(
            _PublishedMetricMatcher(),
            published_metric_names=(
                "public_component",
                "incomplete_component",
            ),
        )

        self.assertEqual(
            result.published_metric_names,
            ("public_component", "incomplete_component"),
        )
        selection = result.rounds[0].finalist_evaluations[0].published_metrics
        self.assertEqual(selection.case_count, 4)
        self.assertEqual(set(selection.aggregates), {"public_component"})
        self.assertAlmostEqual(selection.aggregates["public_component"].mean, 0.875)
        self.assertAlmostEqual(selection.aggregates["public_component"].cvar, 1.4)
        self.assertAlmostEqual(selection.aggregates["public_component"].max, 1.4)

        common_cohort = result.rounds[0].promotion_published_metrics
        self.assertEqual(
            tuple(row.case_count for row in common_cohort),
            (4,),
        )
        self.assertEqual(
            set(common_cohort[0].candidate.aggregates),
            {"public_component"},
        )
        self.assertEqual(
            set(common_cohort[0].candidate.aggregates),
            set(common_cohort[0].incumbent.aggregates),
        )

        terminal = result.terminal
        self.assertEqual(
            set(terminal.cold_start_published_metrics.aggregates),
            {"public_component"},
        )
        self.assertEqual(
            set(terminal.selected_published_metrics.aggregates),
            {"public_component"},
        )
        self.assertAlmostEqual(
            terminal.cold_start_published_metrics.aggregates[
                "public_component"
            ].mean,
            1.25,
        )
        self.assertAlmostEqual(
            terminal.selected_published_metrics.aggregates[
                "public_component"
            ].mean,
            0.5,
        )
        self.assertEqual(terminal.paired_published_metrics.case_count, 4)
        self.assertEqual(
            set(terminal.paired_published_metrics.candidate.aggregates),
            {"public_component"},
        )
        for summary in (
            selection,
            *(row.candidate for row in common_cohort),
            *(row.incumbent for row in common_cohort),
            terminal.cold_start_published_metrics,
            terminal.selected_published_metrics,
            terminal.paired_published_metrics.candidate,
            terminal.paired_published_metrics.incumbent,
        ):
            self.assertNotIn("private_component", summary.aggregates)

        closed, _source, _agent = self._run(_PublishedMetricMatcher())
        self.assertEqual(closed.published_metric_names, ())
        self.assertEqual(
            closed.terminal.selected_published_metrics.aggregates,
            {},
        )

    def test_same_protocol_runs_different_equation_specific_macro_math(self):
        absolute, absolute_source, absolute_agent = self._run(
            _AbsoluteMacroMatcher()
        )
        squared, squared_source, squared_agent = self._run(
            _SquaredMacroMatcher()
        )

        for result in (absolute, squared):
            self.assertEqual(len(result.rounds), 2)
            self.assertTrue(all(row.promotion.promoted for row in result.rounds))
            self.assertEqual(result.selected.candidate_id, "candidate-r2")
            self.assertTrue(result.terminal.comparison.passed)
            self.assertEqual(
                [row.handoff.selected.candidate_id for row in result.rounds],
                ["candidate-r1", "candidate-r2"],
            )
        self.assertNotEqual(
            absolute.macro_contract_id,
            squared.macro_contract_id,
        )
        self.assertEqual(
            [(row.cohort, row.seed, row.count) for row in absolute_source.requests],
            [(row.cohort, row.seed, row.count) for row in squared_source.requests],
        )
        self.assertEqual(absolute_agent.terminal_ids_seen, [])
        self.assertEqual(squared_agent.terminal_ids_seen, [])
        self.assertIsNone(absolute_agent.contexts[0].prior_handoff)
        self.assertEqual(
            absolute_agent.contexts[1].prior_handoff.selected.candidate_id,
            "candidate-r1",
        )

    def test_fixed_generator_binding_has_no_training_or_guidance_operation(self):
        generator = _FrozenGenerator()
        source = FixedGeneratorCaseSource(
            generator_id="fixed-release",
            artifact_sha256="d" * 64,
            generator=generator,
            base_state={"checkpoint": "release"},
        )

        records = source.sample(
            SampleRequest(
                round_index=1,
                cohort=CohortKind.PROMOTION,
                seed=1234,
                count=3,
            )
        )

        self.assertEqual(len(records), 3)
        state, guidance, request = generator.generate_calls[0]
        self.assertEqual(state, {"checkpoint": "release"})
        self.assertIsNone(guidance)
        self.assertEqual(request.settings["fixed_generator_sha256"], "d" * 64)
        self.assertEqual(request.settings["seed"], 1234)
        self.assertTrue(
            all(record.case.visibility is CaseVisibility.HIDDEN for record in records)
        )
        self.assertIs(request.kind, GenerationKind.BASE)
        self.assertEqual(request.anchors, ())

        anchors = tuple(record.case for record in records)
        perturbed_records = source.sample(
            SampleRequest(
                round_index=1,
                cohort=CohortKind.PROMOTION,
                seed=1238,
                count=3,
                kind=GenerationKind.PERTURBED,
                anchors=anchors,
            )
        )
        _state, _guidance, perturbed_request = generator.generate_calls[1]
        self.assertIs(perturbed_request.kind, GenerationKind.PERTURBED)
        self.assertEqual(perturbed_request.anchors, anchors)
        self.assertEqual(perturbed_request.settings["seed"], 1238)
        self.assertEqual(
            tuple(record.case.source for record in perturbed_records),
            (CaseSource.PERTURBED,) * 3,
        )
        self.assertEqual(
            tuple(record.case.parent_case_id for record in perturbed_records),
            tuple(anchor.case_id for anchor in anchors),
        )

    def test_fixed_generator_rejects_unsupported_perturbation_before_generate(self):
        generator = _FrozenGenerator()
        generator.capabilities = GeneratorCapabilities(
            offline=True,
            perturbations=False,
        )
        source = FixedGeneratorCaseSource(
            generator_id="base-only-release",
            artifact_sha256="e" * 64,
            generator=generator,
            base_state={"checkpoint": "base-only"},
        )
        anchor = Case(
            case_id="clean-anchor",
            source=CaseSource.GENERATED,
            visibility=CaseVisibility.HIDDEN,
            public_data={"input": 0},
            round_index=1,
        )

        with self.assertRaisesRegex(
            ValueError,
            "does not support perturbations",
        ):
            source.sample(
                SampleRequest(
                    round_index=1,
                    cohort=CohortKind.PROMOTION,
                    seed=1238,
                    count=1,
                    kind=GenerationKind.PERTURBED,
                    anchors=(anchor,),
                )
            )

        self.assertEqual(generator.generate_calls, [])

    def test_perturbed_sample_request_requires_clean_matching_anchors(self):
        anchor = Case(
            case_id="clean-anchor",
            source=CaseSource.GENERATED,
            visibility=CaseVisibility.HIDDEN,
            public_data={"cohort": CohortKind.PROMOTION.value},
            round_index=1,
            tags=(f"cohort:{CohortKind.PROMOTION.value}",),
        )
        request = SampleRequest(
            round_index=1,
            cohort=CohortKind.PROMOTION,
            seed=1238,
            count=1,
            kind=GenerationKind.PERTURBED,
            anchors=(anchor,),
        )
        self.assertEqual(request.anchors, (anchor,))

        with self.assertRaisesRegex(ValueError, "BASE or PERTURBED"):
            SampleRequest(
                round_index=1,
                cohort=CohortKind.PROMOTION,
                seed=1238,
                count=1,
                kind=GenerationKind.GUIDED,
            )

        bad_anchors = (
            (
                replace(anchor, source=CaseSource.PERTURBED),
                "clean root",
            ),
            (
                replace(anchor, parent_case_id="existing-parent"),
                "clean root",
            ),
            (
                replace(anchor, visibility=CaseVisibility.DEVELOPMENT),
                "visibility",
            ),
            (
                replace(anchor, round_index=2),
                "round",
            ),
        )
        for bad_anchor, message in bad_anchors:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    SampleRequest(
                        round_index=1,
                        cohort=CohortKind.PROMOTION,
                        seed=1238,
                        count=1,
                        kind=GenerationKind.PERTURBED,
                        anchors=(bad_anchor,),
                    )

    def test_agent_failure_keeps_the_incumbent_and_still_runs_terminal(self):
        class FailingAgent:
            def run(self, context):
                raise RuntimeError("private details must not enter the result")

        source = _FixedCaseSource()
        matcher = _AbsoluteMacroMatcher()
        cold_ref = CandidateRef(
            "cold-start",
            "e" * 64,
            0,
            "baseline",
        )
        result = MacroClosedLoopRunner(
            legacy_macro_reference=True,
            config=MacroLoopConfig(
                name="failure",
                round_count=1,
                agents=(MacroAgentSpec("agent", "Agent"),),
                development_case_count=2,
                selection_case_count=2,
                evidence_stages=(EvidenceStage(2, confidence_z=0.0),),
                terminal_case_count=2,
                agent_case_credits=2,
            ),
            plugin=MacroLoopPlugin("equation", source, _Objective(matcher)),
            cold_start=CandidateSubmission(cold_ref, _Approximation(0.5)),
            agents={"agent": FailingAgent()},
        ).run()

        self.assertEqual(result.selected, cold_ref)
        self.assertEqual(
            result.rounds[0].agent_failures,
            {"agent": "agent_failed:RuntimeError"},
        )
        self.assertEqual(result.rounds[0].handoff.decision_reason, "no_eligible_finalist")
        self.assertIsNotNone(result.terminal)
        self.assertFalse(result.terminal.supports_selected_over_cold_start)
        development, *terminal = source.requests
        self.assertEqual(
            (
                development.round_index,
                development.cohort,
                development.seed,
                development.count,
            ),
            (1, CohortKind.DEVELOPMENT, 10_000, 2),
        )
        self.assertEqual(len(terminal), 2)
        self.assertEqual(
            tuple(
                (
                    row.round_index,
                    row.cohort,
                    row.count,
                    row.settings["logical_root_offset"],
                    row.settings["logical_root_count"],
                )
                for row in terminal
            ),
            (
                (2, CohortKind.TERMINAL, 1, 0, 2),
                (2, CohortKind.TERMINAL, 1, 1, 2),
            ),
        )
        self.assertEqual(len({row.seed for row in terminal}), 2)


    def test_lazy_sampling_uses_only_cohorts_reached_by_each_outcome(self):
        success_runner, success_source, success_agent = self._stateful_runner(
            fail=False
        )
        failure_runner, failure_source, failure_agent = self._stateful_runner(
            fail=True
        )

        success = success_runner.run()
        failure = failure_runner.run()
        def request_shape(request):
            return (
                request.round_index,
                request.cohort,
                request.count,
                request.settings.get("logical_root_offset"),
            )

        success_expected = (
            (1, CohortKind.DEVELOPMENT, 4, None),
            (1, CohortKind.SELECTION, 1, 0),
            (1, CohortKind.SELECTION, 1, 1),
            (2, CohortKind.DEVELOPMENT, 4, None),
            (2, CohortKind.SELECTION, 1, 0),
            (2, CohortKind.SELECTION, 1, 1),
            (3, CohortKind.TERMINAL, 1, 0),
            (3, CohortKind.TERMINAL, 1, 1),
        )
        failure_expected = (
            (1, CohortKind.DEVELOPMENT, 4, None),
            (2, CohortKind.DEVELOPMENT, 4, None),
            (3, CohortKind.TERMINAL, 1, 0),
            (3, CohortKind.TERMINAL, 1, 1),
        )
        self.assertEqual(
            tuple(map(request_shape, success_source.requests)),
            success_expected,
        )
        self.assertEqual(
            tuple(map(request_shape, failure_source.requests)),
            failure_expected,
        )
        self.assertEqual(success_source.truth_answer_calls, 14)
        self.assertEqual(failure_source.truth_answer_calls, 10)
        success_request_rows = tuple(
            (row.round_index, row.cohort, row.seed, row.count)
            for row in success_source.requests
        )
        failure_request_rows = tuple(
            (row.round_index, row.cohort, row.seed, row.count)
            for row in failure_source.requests
        )
        self.assertEqual(
            success_agent.pre_run_snapshots,
            [(success_request_rows[:1], 4), (success_request_rows[:4], 10)],
        )
        self.assertEqual(
            failure_agent.pre_run_snapshots,
            [(failure_request_rows[:1], 4), (failure_request_rows[:2], 8)],
        )
        self.assertNotEqual(
            success.terminal.cohort_sha256,
            failure.terminal.cohort_sha256,
        )
        self.assertTrue(success.terminal.supports_selected_over_cold_start)
        self.assertFalse(failure.terminal.supports_selected_over_cold_start)
        self.assertTrue(
            all(
                row.agent_failures == {"agent": "agent_failed:RuntimeError"}
                for row in failure.rounds
            )
        )

    def test_default_loop_preserves_legacy_mixed_case_sources(self):
        runner, source, agent = self._stateful_runner(
            fail=False,
            source=_LegacyMixedCaseSource(),
        )

        result = runner.run()

        self.assertFalse(result.paired_perturbations)
        self.assertTrue(
            all(request.kind is GenerationKind.BASE for request in source.requests)
        )
        self.assertTrue(
            all(
                case.source is CaseSource.GENERATED
                and case.visibility is CaseVisibility.DEVELOPMENT
                and case.round_index is None
                and case.parent_case_id is None
                and case.tags == ()
                and "cohort" not in case.public_data
                for context in agent.contexts
                for case in context.cases
            )
        )

    def test_runner_is_single_use_and_does_not_resample(self):
        runner, source, _agent = self._stateful_runner(fail=False)
        runner.run()
        request_count = len(source.requests)

        with self.assertRaisesRegex(RuntimeError, "single-use"):
            runner.run()

        self.assertEqual(len(source.requests), request_count)

    def test_config_requires_strict_integer_count_round_and_seed_fields(self):
        base = {
            "name": "strict-integers",
            "round_count": 1,
            "agents": (MacroAgentSpec("agent", "Agent"),),
            "development_case_count": 2,
            "selection_case_count": 2,
            "evidence_stages": (EvidenceStage(2, confidence_z=0.0),),
            "terminal_case_count": 2,
            "agent_case_credits": 2,
            "seed_base": 0,
            "seed_stride": 10_000,
            "development_pool_multiplier": 1,
        }
        invalid_values = {
            "round_count": 1.0,
            "development_case_count": True,
            "selection_case_count": 2.0,
            "terminal_case_count": False,
            "agent_case_credits": 2.0,
            "seed_base": 0.0,
            "seed_stride": 10_000.0,
            "development_pool_multiplier": 1.0,
        }
        for field_name, value in invalid_values.items():
            with self.subTest(field_name=field_name):
                kwargs = dict(base)
                kwargs[field_name] = value
                with self.assertRaisesRegex(ValueError, field_name):
                    MacroLoopConfig(**kwargs)

        one_case = dict(base)
        one_case["terminal_case_count"] = 1
        with self.assertRaisesRegex(ValueError, "at least two"):
            MacroLoopConfig(**one_case)
        one_case["terminal_confidence_z"] = 0.0
        self.assertEqual(MacroLoopConfig(**one_case).terminal_case_count, 1)
        default_terminal_policy = MacroLoopConfig(**base)
        self.assertTrue(
            default_terminal_policy.terminal_require_cvar_no_regression
        )
        cvar_optional = dict(base)
        cvar_optional["terminal_require_cvar_no_regression"] = False
        self.assertFalse(
            MacroLoopConfig(**cvar_optional).terminal_require_cvar_no_regression
        )
        cvar_optional["terminal_require_cvar_no_regression"] = "false"
        with self.assertRaisesRegex(TypeError, "terminal_require_cvar"):
            MacroLoopConfig(**cvar_optional)

    def test_duplicate_batch_does_not_mutate_seen_ids_and_consumes_runner(self):
        class DuplicateSource:
            generator_id = "duplicate-source"
            artifact_sha256 = "8" * 64

            def sample(self, request):
                return tuple(
                    CaseRecord(
                        Case(
                            case_id="duplicate",
                            source=CaseSource.GENERATED,
                            visibility=CaseVisibility.DEVELOPMENT,
                            public_data={"cohort": request.cohort.value},
                            round_index=request.round_index,
                        ),
                        _ScalarTruth(),
                    )
                    for _index in range(request.count)
                )

        source = DuplicateSource()
        agent = _ImprovingAgent()
        cold_ref = CandidateRef(
            "cold",
            "7" * 64,
            0,
            "baseline",
        )
        runner = MacroClosedLoopRunner(
            legacy_macro_reference=True,
            config=MacroLoopConfig(
                name="duplicate",
                round_count=1,
                agents=(MacroAgentSpec("agent", "Agent"),),
                development_case_count=2,
                selection_case_count=2,
                evidence_stages=(EvidenceStage(2, confidence_z=0.0),),
                terminal_case_count=2,
                agent_case_credits=2,
            ),
            plugin=MacroLoopPlugin(
                "duplicate", source, _Objective(_AbsoluteMacroMatcher())
            ),
            cold_start=CandidateSubmission(cold_ref, _Approximation(0.5)),
            agents={"agent": agent},
        )

        with self.assertRaisesRegex(ValueError, "unique within"):
            runner.run()
        self.assertEqual(runner._seen_case_ids, set())
        self.assertEqual(agent.contexts, [])
        with self.assertRaisesRegex(RuntimeError, "single-use"):
            runner.run()

    def test_source_response_metadata_must_match_the_controller_request(self):
        class InvalidMetadataSource:
            generator_id = "invalid-metadata-source"
            artifact_sha256 = "6" * 64

            def __init__(self, mode):
                self.mode = mode

            def sample(self, request):
                visibility = (
                    CaseVisibility.DEVELOPMENT
                    if request.cohort is CohortKind.DEVELOPMENT
                    else CaseVisibility.HIDDEN
                )
                source = CaseSource.GENERATED
                round_index = request.round_index
                public_data = {"cohort": request.cohort.value}
                if self.mode == "visibility":
                    visibility = CaseVisibility.HIDDEN
                elif self.mode == "source":
                    source = "generated"
                elif self.mode == "round":
                    round_index = True
                elif self.mode == "cohort":
                    public_data = {"cohort": CohortKind.SELECTION.value}
                return tuple(
                    CaseRecord(
                        Case(
                            case_id=f"{self.mode}-{index}",
                            source=source,
                            visibility=visibility,
                            public_data=public_data,
                            round_index=round_index,
                        ),
                        _ScalarTruth(),
                    )
                    for index in range(request.count)
                )

        expected_messages = {
            "visibility": "visibility",
            "source": "source",
            "round": "round_index",
            "cohort": "cohort",
        }
        for mode, message in expected_messages.items():
            with self.subTest(mode=mode):
                source = InvalidMetadataSource(mode)
                agent = _ImprovingAgent()
                cold_ref = CandidateRef(
                    "cold",
                    "5" * 64,
                    0,
                    "baseline",
                )
                runner = MacroClosedLoopRunner(
                    legacy_macro_reference=True,
                    config=MacroLoopConfig(
                        name=f"metadata-{mode}",
                        round_count=1,
                        agents=(MacroAgentSpec("agent", "Agent"),),
                        development_case_count=2,
                        selection_case_count=2,
                        evidence_stages=(EvidenceStage(2, confidence_z=0.0),),
                        terminal_case_count=2,
                        agent_case_credits=2,
                    ),
                    plugin=MacroLoopPlugin(
                        "metadata", source, _Objective(_AbsoluteMacroMatcher())
                    ),
                    cold_start=CandidateSubmission(
                        cold_ref, _Approximation(0.5)
                    ),
                    agents={"agent": agent},
                )
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    runner.run()
                self.assertEqual(agent.contexts, [])


    def test_formal_assurance_fails_before_sampling_or_agent_execution(self):
        source = _FixedCaseSource()
        agent = _ImprovingAgent()
        runner = MacroClosedLoopRunner(
            legacy_macro_reference=True,
            config=MacroLoopConfig(
                name="formal-must-fail-closed",
                round_count=1,
                agents=(MacroAgentSpec("agent", "Agent"),),
                development_case_count=2,
                selection_case_count=2,
                evidence_stages=(EvidenceStage(2, confidence_z=0.0),),
                terminal_case_count=2,
                agent_case_credits=2,
                execution_assurance=ExecutionAssurance.FORMAL_ISOLATED,
            ),
            plugin=MacroLoopPlugin(
                "formal",
                source,
                _Objective(_AbsoluteMacroMatcher()),
            ),
            cold_start=CandidateSubmission(
                CandidateRef("cold", "4" * 64, 0, "baseline"),
                _Approximation(0.5),
            ),
            agents={"agent": agent},
        )

        with self.assertRaises(FormalExecutionUnavailable):
            runner.run()

        self.assertEqual(source.requests, [])
        self.assertEqual(agent.contexts, [])

    def test_terminal_does_not_compare_cold_start_with_itself(self):
        class CountingApproximation(_Approximation):
            def __init__(self):
                super().__init__(0.5)
                self.calls = 0

            def predict_macros(self, case, query, budget=None):
                self.calls += 1
                self.error = 0.1 if self.calls <= 2 else 0.9
                return super().predict_macros(case, query, budget=budget)

        class FailingAgent:
            def run(self, context):
                raise RuntimeError("no candidate")

        source = _FixedCaseSource()
        cold_program = CountingApproximation()
        cold_ref = CandidateRef("cold", "3" * 64, 0, "baseline")
        result = MacroClosedLoopRunner(
            legacy_macro_reference=True,
            config=MacroLoopConfig(
                name="terminal-self-comparison",
                round_count=1,
                agents=(MacroAgentSpec("agent", "Agent"),),
                development_case_count=2,
                selection_case_count=2,
                evidence_stages=(EvidenceStage(2, confidence_z=0.0),),
                terminal_case_count=2,
                agent_case_credits=2,
            ),
            plugin=MacroLoopPlugin(
                "terminal-self",
                source,
                _Objective(_AbsoluteMacroMatcher()),
            ),
            cold_start=CandidateSubmission(cold_ref, cold_program),
            agents={"agent": FailingAgent()},
        ).run()

        self.assertEqual(cold_program.calls, 2)
        self.assertIs(result.terminal.cold_start, result.terminal.selected)
        self.assertIs(result.terminal.cold_start_score, result.terminal.selected_score)
        self.assertIsNone(result.terminal.comparison)

    def test_candidate_bundle_digest_is_unique_across_ids(self):
        class DuplicateBundleAgent:
            def run(self, context):
                return MacroAgentOutcome(
                    CandidateSubmission(
                        CandidateRef(
                            "renamed-incumbent",
                            context.incumbent.bundle_sha256,
                            context.round_index,
                            context.agent.agent_id,
                            parent_sha256=context.incumbent.bundle_sha256,
                        ),
                        _Approximation(0.1),
                    )
                )

        source = _FixedCaseSource()
        cold_ref = CandidateRef("cold", "2" * 64, 0, "baseline")
        result = MacroClosedLoopRunner(
            legacy_macro_reference=True,
            config=MacroLoopConfig(
                name="duplicate-bundle",
                round_count=1,
                agents=(MacroAgentSpec("agent", "Agent"),),
                development_case_count=2,
                selection_case_count=2,
                evidence_stages=(EvidenceStage(2, confidence_z=0.0),),
                terminal_case_count=2,
                agent_case_credits=2,
            ),
            plugin=MacroLoopPlugin(
                "duplicate-bundle",
                source,
                _Objective(_AbsoluteMacroMatcher()),
            ),
            cold_start=CandidateSubmission(cold_ref, _Approximation(0.5)),
            agents={"agent": DuplicateBundleAgent()},
        ).run()

        self.assertEqual(result.selected, cold_ref)
        self.assertEqual(result.rounds[0].candidate_refs, ())
        self.assertEqual(
            result.rounds[0].agent_failures,
            {"agent": "agent_failed:ValueError"},
        )

    def test_hard_mining_prioritizes_equation_owned_gate_rejections(self):
        class GateMatcher(_AbsoluteMacroMatcher):
            contract_id = "gate_aware_macro_v1"

            def match(self, case, prediction, target):
                index = int(case.public_data["input"])
                rejected = index == 0
                return MacroMatchResult(
                    case.case_id,
                    self.contract_id,
                    float(index),
                    passed=not rejected,
                    failures=("equation_gate",) if rejected else (),
                )

        class CapturingAgent:
            def __init__(self):
                self.inputs = ()

            def run(self, context):
                self.inputs = tuple(
                    float(case.public_data["input"]) for case in context.cases
                )
                raise RuntimeError("stop after observing development cases")

        source = _FixedCaseSource()
        agent = CapturingAgent()
        result = MacroClosedLoopRunner(
            legacy_macro_reference=True,
            config=MacroLoopConfig(
                name="gate-aware-mining",
                round_count=1,
                agents=(MacroAgentSpec("agent", "Agent"),),
                development_case_count=2,
                selection_case_count=2,
                evidence_stages=(EvidenceStage(2, confidence_z=0.0),),
                terminal_case_count=2,
                agent_case_credits=2,
                development_pool_multiplier=2,
            ),
            plugin=MacroLoopPlugin(
                "gate-aware",
                source,
                _Objective(GateMatcher()),
            ),
            cold_start=CandidateSubmission(
                CandidateRef("cold", "1" * 64, 0, "baseline"),
                _Approximation(0.5),
            ),
            agents={"agent": agent},
        ).run()

        self.assertEqual(agent.inputs, (0.0, 3.0))
        self.assertEqual(result.selected.candidate_id, "cold")

    def test_recoverable_paired_sampling_fresh_resamples_and_audits(self):
        def make_runner(source, agent, *, maximum_attempts):
            return MacroClosedLoopRunner(
                legacy_macro_reference=True,
                config=MacroLoopConfig(
                    name="paired-retry",
                    round_count=1,
                    agents=(MacroAgentSpec("agent", "Agent"),),
                    development_case_count=4,
                    selection_case_count=4,
                    evidence_stages=(
                        EvidenceStage(4, confidence_z=0.0),
                    ),
                    terminal_case_count=4,
                    agent_case_credits=4,
                    seed_base=100,
                    paired_perturbations=True,
                    paired_sampling_max_attempts=maximum_attempts,
                    terminal_confidence_z=0.0,
                ),
                plugin=MacroLoopPlugin(
                    "paired-retry",
                    source,
                    _Objective(_AbsoluteMacroMatcher()),
                ),
                cold_start=CandidateSubmission(
                    CandidateRef("cold", "9" * 64, 0, "baseline"),
                    _Approximation(0.5),
                ),
                agents={"agent": agent},
            )

        source = _RecoverablyRejectingPairedCaseSource(
            development_rejections=1
        )
        result = make_runner(
            source,
            _ImprovingAgent(),
            maximum_attempts=3,
        ).run()
        development_attempts = tuple(
            row
            for row in result.sampling_attempts
            if row.cohort is CohortKind.DEVELOPMENT
        )
        self.assertEqual(len(development_attempts), 2)
        rejected, accepted = development_attempts
        self.assertFalse(rejected.accepted)
        self.assertTrue(accepted.accepted)
        self.assertEqual(
            rejected.rejection_audit["failed_gates"],
            ["synthetic_gate"],
        )
        self.assertEqual(rejected.base_seed, 10_100)
        self.assertNotEqual(accepted.base_seed, rejected.base_seed)
        self.assertEqual(
            rejected.root_case_ids,
            source.rejected_root_ids[0],
        )
        self.assertTrue(
            set(rejected.root_case_ids).isdisjoint(
                result.rounds[0].development_case_ids
            )
        )

        replay_source = _RecoverablyRejectingPairedCaseSource(
            development_rejections=1
        )
        replay = make_runner(
            replay_source,
            _ImprovingAgent(),
            maximum_attempts=3,
        ).run()
        self.assertEqual(
            tuple(
                (row.base_seed, row.perturbed_seed, row.accepted)
                for row in replay.sampling_attempts
            ),
            tuple(
                (row.base_seed, row.perturbed_seed, row.accepted)
                for row in result.sampling_attempts
            ),
        )

        exhausted_source = _RecoverablyRejectingPairedCaseSource(
            development_rejections=3
        )
        exhausted_agent = _ImprovingAgent()
        with self.assertRaises(RecoverableSamplingRejection) as caught:
            make_runner(
                exhausted_source,
                exhausted_agent,
                maximum_attempts=2,
            ).run()
        self.assertEqual(
            caught.exception.audit["schema"],
            "paired_sampling_exhaustion_v1",
        )
        self.assertEqual(len(caught.exception.audit["attempts"]), 2)
        self.assertEqual(exhausted_agent.contexts, [])

    def test_paired_perturbations_cover_all_cohorts_with_adjacent_lineage(self):
        source = _PairedCaseSource()
        agent = _ImprovingAgent()
        result = MacroClosedLoopRunner(
            legacy_macro_reference=True,
            config=MacroLoopConfig(
                name="paired-all-cohorts",
                round_count=1,
                agents=(MacroAgentSpec("agent", "Agent"),),
                development_case_count=4,
                selection_case_count=4,
                evidence_stages=(
                    EvidenceStage(2, confidence_z=0.0),
                    EvidenceStage(4, confidence_z=1.0),
                ),
                terminal_case_count=4,
                agent_case_credits=4,
                seed_base=100,
                paired_perturbations=True,
                terminal_confidence_z=1.0,
            ),
            plugin=MacroLoopPlugin(
                "paired",
                source,
                _Objective(_AbsoluteMacroMatcher()),
            ),
            cold_start=CandidateSubmission(
                CandidateRef("cold", "0" * 63 + "1", 0, "baseline"),
                _Approximation(0.5),
            ),
            agents={"agent": agent},
        ).run()
        self.assertTrue(result.paired_perturbations)

        expected_streams = (
            (1, CohortKind.DEVELOPMENT, 2, None, 10_100, 10_104),
            (1, CohortKind.SELECTION, 1, 0, None, None),
            (1, CohortKind.SELECTION, 1, 1, None, None),
            (2, CohortKind.TERMINAL, 1, 0, None, None),
            (2, CohortKind.TERMINAL, 1, 1, None, None),
        )
        self.assertEqual(len(source.requests), 2 * len(expected_streams))
        for (base, perturbed), (
            round_index,
            cohort,
            count,
            root_offset,
            base_seed,
            perturbed_seed,
        ) in zip(
            zip(source.requests[::2], source.requests[1::2]),
            expected_streams,
        ):
            with self.subTest(cohort=cohort, root_offset=root_offset):
                self.assertEqual(
                    (base.round_index, base.cohort, base.kind, base.count),
                    (
                        round_index,
                        cohort,
                        GenerationKind.BASE,
                        count,
                    ),
                )
                self.assertEqual(base.anchors, ())
                self.assertEqual(
                    (
                        perturbed.round_index,
                        perturbed.cohort,
                        perturbed.kind,
                        perturbed.count,
                    ),
                    (
                        round_index,
                        cohort,
                        GenerationKind.PERTURBED,
                        count,
                    ),
                )
                self.assertEqual(
                    base.settings.get("logical_root_offset"),
                    root_offset,
                )
                self.assertEqual(
                    perturbed.settings.get("logical_root_offset"),
                    root_offset,
                )
                if base_seed is not None:
                    self.assertEqual(base.seed, base_seed)
                    self.assertEqual(perturbed.seed, perturbed_seed)
                first_root = 0 if root_offset is None else root_offset
                self.assertEqual(
                    tuple(anchor.case_id for anchor in perturbed.anchors),
                    tuple(
                        f"r{round_index}-{cohort.value}-root-{index}"
                        for index in range(first_root, first_root + count)
                    ),
                )
        official_seeds = tuple(
            request.seed
            for request in source.requests
            if "logical_root_offset" in request.settings
        )
        self.assertEqual(len(official_seeds), len(set(official_seeds)))

        development_cases = agent.contexts[0].cases
        self.assertEqual(
            tuple(case.case_id for case in development_cases),
            tuple(
                candidate_case_id(case_id)
                for case_id in result.rounds[0].development_case_ids
            ),
        )
        self.assertTrue(all(
            case.source is CaseSource.GENERATED
            and case.visibility is CaseVisibility.DEVELOPMENT
            and case.round_index is None
            and case.parent_case_id is None
            and case.tags == ()
            for case in development_cases
        ))
        self.assertNotIn(
            "adjacent clean/perturbed pairs",
            agent.contexts[0].prompt,
        )
        self.assertIn(
            "evaluation order are controller-private",
            agent.contexts[0].prompt,
        )
        self.assertEqual(
            result.rounds[0].handoff.feedback.confidence_group_size,
            1,
        )

        selection_ids = result.rounds[0].finalist_evaluations[0].batch.case_ids
        promotion = result.rounds[0].promotion
        promotion_ids = promotion.comparisons[-1].case_ids
        terminal_ids = result.terminal.selected_score.case_ids
        for case_ids in (
            result.rounds[0].development_case_ids,
            selection_ids,
        ):
            self.assertEqual(len(case_ids), 4)
            for root_id, child_id in zip(case_ids[::2], case_ids[1::2]):
                self.assertEqual(child_id, f"{root_id}-perturbed")
        self.assertEqual(len(promotion_ids), 2)
        self.assertEqual(len(terminal_ids), 2)
        self.assertEqual(
            promotion.comparisons[0].case_ids,
            promotion_ids,
        )
        self.assertEqual(
            tuple(
                comparison.confidence_group_size
                for comparison in promotion.comparisons
            ),
            (1,),
        )
        self.assertEqual(
            tuple(
                comparison.confidence_sample_count
                for comparison in promotion.comparisons
            ),
            (2,),
        )
        self.assertIsNotNone(result.terminal.comparison)
        self.assertEqual(result.terminal.comparison.confidence_group_size, 1)
        self.assertEqual(result.terminal.comparison.confidence_sample_count, 2)

    def test_streamed_batch_commitment_matches_monolithic_cohort(self):
        matcher = _AbsoluteMacroMatcher()
        query = AnswerQuery.grid(fields=("macro",), sample=1)
        records = tuple(
            CaseRecord(
                Case(
                    case_id=f"commitment-case-{index}",
                    source=CaseSource.GENERATED,
                    visibility=CaseVisibility.HIDDEN,
                    public_data={"input": float(index)},
                ),
                _ScalarTruth(),
            )
            for index in range(4)
        )
        entries = (
            ("commitment-left", _Approximation(0.5)),
            ("commitment-right", _Approximation(0.25)),
        )
        full = PreparedMacroCohort.prepare(
            cohort_id="commitment-common",
            records=records,
            matcher=matcher,
            query=query,
        )
        try:
            expected = ControllerMacroEvaluator(full).evaluate_many(entries)
        finally:
            full.discard()

        accumulator = MacroBatchStreamAccumulator(
            cohort_id="commitment-common",
            contract_id=matcher.contract_id,
            candidate_ids=tuple(row[0] for row in entries),
        )
        for start in (0, 2):
            chunk = PreparedMacroCohort.prepare(
                cohort_id="commitment-common",
                records=records[start : start + 2],
                matcher=matcher,
                query=query,
            )
            try:
                accumulator.consume(
                    chunk,
                    ControllerMacroEvaluator(chunk).evaluate_many(entries),
                )
            finally:
                chunk.discard()
        streamed = accumulator.finalize()

        for actual, reference in zip(streamed, expected):
            self.assertEqual(actual.candidate_id, reference.candidate_id)
            self.assertEqual(actual.cohort_id, reference.cohort_id)
            self.assertEqual(actual.cohort_sha256, reference.cohort_sha256)
            self.assertEqual(actual.target_sha256, reference.target_sha256)
            self.assertEqual(actual.contract_id, reference.contract_id)
            self.assertEqual(actual.scores, reference.scores)
            self.assertEqual(actual.root_ids, reference.root_ids)
            self.assertEqual(
                tuple(row.case_id for row in actual.case_timings),
                tuple(row.case_id for row in reference.case_timings),
            )

    def test_stream_accumulator_rejects_disk_backed_chunk(self):
        matcher = _AbsoluteMacroMatcher()
        record = CaseRecord(
            Case(
                case_id="disk-stream-case",
                source=CaseSource.GENERATED,
                visibility=CaseVisibility.HIDDEN,
                public_data={"input": 0.0},
            ),
            _ScalarTruth(),
        )
        entry = ("disk-stream-candidate", _Approximation(0.5))
        with TemporaryDirectory() as temporary:
            cohort = PreparedMacroCohort.prepare(
                cohort_id="disk-stream-common",
                records=(record,),
                matcher=matcher,
                query=AnswerQuery.grid(fields=("macro",), sample=1),
                private_store_root=Path(temporary) / "private",
            )
            stored_root = cohort.private_manifest_path.parent
            try:
                batch = ControllerMacroEvaluator(cohort).evaluate(*entry)
                accumulator = MacroBatchStreamAccumulator(
                    cohort_id=cohort.cohort_id,
                    contract_id=cohort.contract_id,
                    candidate_ids=(entry[0],),
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "must not use private disk storage",
                ):
                    accumulator.consume(cohort, (batch,))
            finally:
                cohort.discard()
            self.assertFalse(stored_root.exists())

    def test_official_common_cohort_streams_pairs_without_disk_or_growth(self):
        source = _ReleasingPairedCaseSource()
        left = _RecordingApproximation(0.5)
        right = _RecordingApproximation(0.25)
        cold = CandidateSubmission(
            CandidateRef("stream-cold", "1" * 64, 0, "baseline"),
            left,
        )
        challenger = CandidateSubmission(
            CandidateRef(
                "stream-challenger",
                "2" * 64,
                1,
                "agent",
                parent_sha256=cold.ref.bundle_sha256,
            ),
            right,
        )
        config = MacroLoopConfig(
            name="bounded-official-stream",
            round_count=1,
            agents=(MacroAgentSpec("agent", "Agent"),),
            development_case_count=4,
            selection_case_count=8,
            evidence_stages=(EvidenceStage(8, confidence_z=0.0),),
            terminal_case_count=4,
            agent_case_credits=4,
            paired_perturbations=True,
            terminal_confidence_z=0.0,
        )

        with TemporaryDirectory() as temporary:
            private_store = Path(temporary) / "private-cohorts"
            runner = MacroClosedLoopRunner(
                legacy_macro_reference=True,
                config=config,
                plugin=MacroLoopPlugin(
                    "bounded-official-stream",
                    source,
                    _Objective(_AbsoluteMacroMatcher()),
                ),
                cold_start=cold,
                agents={"agent": _ImprovingAgent()},
                cohort_store_root=private_store,
            )
            runner._precompute_cohorts()
            batches = runner._evaluate_official_common_cohort(
                round_index=1,
                cohort_kind=CohortKind.SELECTION,
                submissions=(cold, challenger),
            )

            self.assertFalse(private_store.exists())

        expected_case_ids = tuple(
            case_id
            for root_index in range(4)
            for case_id in (
                f"r1-selection-root-{root_index}",
                f"r1-selection-root-{root_index}-perturbed",
            )
        )
        self.assertEqual(tuple(batch.case_ids for batch in batches), (
            expected_case_ids,
            expected_case_ids,
        ))
        self.assertEqual(left.case_ids, right.case_ids)
        self.assertEqual(
            set(left.case_ids),
            {candidate_case_id(case_id) for case_id in expected_case_ids},
        )
        self.assertEqual(batches[0].execution_order, batches[1].execution_order)
        self.assertEqual(set(batches[0].execution_order), set(expected_case_ids))
        self.assertEqual(source.maximum_live_case_count, 2)
        self.assertEqual(source.live_case_ids, set())
        self.assertEqual(
            tuple(len(case_ids) for case_ids in source.release_batches),
            (2, 2, 2, 2),
        )
        self.assertEqual(
            tuple(
                request.settings["logical_root_offset"]
                for request in source.requests[::2]
            ),
            (0, 1, 2, 3),
        )
        self.assertTrue(
            all(request.count == 1 for request in source.requests)
        )
        for request in source.requests:
            self.assertEqual(
                request.settings["logical_cohort_id"],
                "round-1-selection",
            )
            self.assertEqual(request.settings["logical_root_count"], 4)
        self.assertEqual(
            len(
                {
                    (batch.cohort_sha256, batch.target_sha256)
                    for batch in batches
                }
            ),
            1,
        )
        self.assertEqual(
            batches[0].root_ids,
            tuple(
                root_id
                for root_index in range(4)
                for root_id in (
                    f"r1-selection-root-{root_index}",
                    f"r1-selection-root-{root_index}",
                )
            ),
        )

        class InterruptingApproximation:
            def predict_macros(self, case, query, budget=None):
                raise KeyboardInterrupt("synthetic official evaluation stop")

        failing_source = _ReleasingPairedCaseSource()
        failing_cold = CandidateSubmission(
            CandidateRef("interrupt-cold", "3" * 64, 0, "baseline"),
            InterruptingApproximation(),
        )
        with TemporaryDirectory() as temporary:
            failing_store = Path(temporary) / "private-cohorts"
            failing_runner = MacroClosedLoopRunner(
                legacy_macro_reference=True,
                config=config,
                plugin=MacroLoopPlugin(
                    "bounded-official-stream",
                    failing_source,
                    _Objective(_AbsoluteMacroMatcher()),
                ),
                cold_start=failing_cold,
                agents={"agent": _ImprovingAgent()},
                cohort_store_root=failing_store,
            )
            failing_runner._precompute_cohorts()
            with self.assertRaises(KeyboardInterrupt):
                failing_runner._evaluate_official_common_cohort(
                    round_index=1,
                    cohort_kind=CohortKind.SELECTION,
                    submissions=(failing_cold,),
                )
            self.assertFalse(failing_store.exists())

        self.assertEqual(failing_source.live_case_ids, set())
        self.assertEqual(
            tuple(len(case_ids) for case_ids in failing_source.release_batches),
            (2,),
        )

    def test_fresh_formal_runtime_resets_between_phases_and_reuses_stream(self):
        source = _FixedCaseSource()
        runtimes = []

        class CounterProgram:
            def __init__(self, calls):
                self.calls = calls

            def predict_macros(self, case, query, budget=None):
                del case, budget
                self.calls.append(len(self.calls) + 1)
                return AnswerView(
                    kind=query.kind,
                    fields={"macro": (1.0,)},
                    axes={"sample": (0.0,)},
                    shape=(1,),
                )

        def factory():
            calls = []
            runtimes.append(calls)
            return CounterProgram(calls)

        submission = CandidateSubmission(
            CandidateRef("phase-isolated", "6" * 64, 0, "baseline"),
            _Approximation(0.75),
            program_factory=factory,
        )
        config = MacroLoopConfig(
            name="phase-isolated-formal-runtime",
            round_count=1,
            agents=(MacroAgentSpec("agent", "Agent"),),
            development_case_count=2,
            selection_case_count=3,
            evidence_stages=(EvidenceStage(3, confidence_z=0.0),),
            terminal_case_count=2,
            agent_case_credits=2,
            terminal_confidence_z=0.0,
        )
        runner = MacroClosedLoopRunner(
            legacy_macro_reference=True,
            config=config,
            plugin=MacroLoopPlugin(
                "phase-isolated-formal-runtime",
                source,
                _Objective(_AbsoluteMacroMatcher()),
            ),
            cold_start=submission,
            agents={"agent": _ImprovingAgent()},
        )
        runner._precompute_cohorts()

        (selection,) = runner._evaluate_official_common_cohort(
            round_index=1,
            cohort_kind=CohortKind.SELECTION,
            submissions=(submission,),
        )
        (terminal,) = runner._evaluate_official_common_cohort(
            round_index=2,
            cohort_kind=CohortKind.TERMINAL,
            submissions=(submission,),
        )

        self.assertEqual(runtimes, [[1, 2, 3], [1, 2]])
        self.assertEqual(len(selection.scores), 3)
        self.assertEqual(len(terminal.scores), 2)
        self.assertTrue(
            all(score.status == "valid" for score in selection.scores)
        )


    def test_terminal_evaluation_streams_pairs_without_disk_or_growth(self):
        source = _ReleasingPairedCaseSource()
        cold = CandidateSubmission(
            CandidateRef("terminal-stream-cold", "4" * 64, 0, "baseline"),
            _Approximation(0.5),
        )
        selected = CandidateSubmission(
            CandidateRef(
                "terminal-stream-selected",
                "5" * 64,
                1,
                "agent",
                parent_sha256=cold.ref.bundle_sha256,
            ),
            _Approximation(0.25),
        )
        config = MacroLoopConfig(
            name="bounded-terminal-stream",
            round_count=1,
            agents=(MacroAgentSpec("agent", "Agent"),),
            development_case_count=2,
            selection_case_count=2,
            evidence_stages=(EvidenceStage(2, confidence_z=0.0),),
            terminal_case_count=6,
            agent_case_credits=2,
            paired_perturbations=True,
            terminal_confidence_z=0.0,
        )

        with TemporaryDirectory() as temporary:
            private_store = Path(temporary) / "private-cohorts"
            runner = MacroClosedLoopRunner(
                legacy_macro_reference=True,
                config=config,
                plugin=MacroLoopPlugin(
                    "bounded-terminal-stream",
                    source,
                    _Objective(_AbsoluteMacroMatcher()),
                ),
                cold_start=cold,
                agents={"agent": _ImprovingAgent()},
                cohort_store_root=private_store,
            )
            runner._precompute_cohorts()
            terminal = runner._terminal_evaluation(selected)
            self.assertFalse(private_store.exists())

        expected_case_ids = tuple(
            case_id
            for root_index in range(3)
            for case_id in (
                f"r2-terminal-root-{root_index}",
                f"r2-terminal-root-{root_index}-perturbed",
            )
        )
        self.assertEqual(terminal.cold_start.case_ids, expected_case_ids)
        self.assertEqual(terminal.selected.case_ids, expected_case_ids)
        self.assertEqual(source.maximum_live_case_count, 2)
        self.assertEqual(source.live_case_ids, set())
        self.assertEqual(
            tuple(len(case_ids) for case_ids in source.release_batches),
            (2, 2, 2),
        )
        self.assertTrue(all(request.count == 1 for request in source.requests))
        for request in source.requests:
            self.assertEqual(
                request.settings["logical_cohort_id"],
                "terminal-holdout",
            )
            self.assertEqual(request.settings["logical_root_count"], 3)

    def test_paired_perturbation_counts_must_end_on_pair_boundaries(self):
        base = {
            "name": "paired-even-counts",
            "round_count": 1,
            "agents": (MacroAgentSpec("agent", "Agent"),),
            "development_case_count": 4,
            "selection_case_count": 4,
            "evidence_stages": (
                EvidenceStage(2, confidence_z=0.0),
                EvidenceStage(4, confidence_z=0.0),
            ),
            "terminal_case_count": 4,
            "agent_case_credits": 4,
            "paired_perturbations": True,
        }
        self.assertTrue(MacroLoopConfig(**base).paired_perturbations)
        invalid = (
            {"development_case_count": 3},
            {"selection_case_count": 3},
            {
                "evidence_stages": (
                    EvidenceStage(3, confidence_z=0.0),
                    EvidenceStage(4, confidence_z=0.0),
                )
            },
            {"terminal_case_count": 3},
        )
        for override in invalid:
            with self.subTest(override=override):
                values = dict(base)
                values.update(override)
                with self.assertRaises(ValueError):
                    MacroLoopConfig(**values)

        invalid_type = dict(base)
        invalid_type["paired_perturbations"] = 1
        with self.assertRaises(TypeError):
            MacroLoopConfig(**invalid_type)

        colliding_seeds = dict(base)
        colliding_seeds["seed_stride"] = 7
        with self.assertRaisesRegex(ValueError, "seed_stride"):
            MacroLoopConfig(**colliding_seeds)

        one_pair_stage = dict(base)
        one_pair_stage["evidence_stages"] = (
            EvidenceStage(2, confidence_z=1.0),
            EvidenceStage(4, confidence_z=0.0),
        )
        with self.assertRaisesRegex(ValueError, "at least two pairs"):
            MacroLoopConfig(**one_pair_stage)

        one_pair_terminal = dict(base)
        one_pair_terminal.update(
            {
                "terminal_case_count": 2,
                "terminal_confidence_z": 1.0,
            }
        )
        with self.assertRaisesRegex(ValueError, "at least two pairs"):
            MacroLoopConfig(**one_pair_terminal)

    def test_paired_source_parent_and_source_fail_before_agent_execution(self):
        for invalid_child in ("parent", "source"):
            with self.subTest(invalid_child=invalid_child):
                source = _PairedCaseSource(invalid_child=invalid_child)
                agent = _ImprovingAgent()
                runner = MacroClosedLoopRunner(
                    legacy_macro_reference=True,
                    config=MacroLoopConfig(
                        name=f"invalid-paired-{invalid_child}",
                        round_count=1,
                        agents=(MacroAgentSpec("agent", "Agent"),),
                        development_case_count=4,
                        selection_case_count=4,
                        evidence_stages=(EvidenceStage(4, confidence_z=0.0),),
                        terminal_case_count=4,
                        agent_case_credits=4,
                        paired_perturbations=True,
                    ),
                    plugin=MacroLoopPlugin(
                        "invalid-paired",
                        source,
                        _Objective(_AbsoluteMacroMatcher()),
                    ),
                    cold_start=CandidateSubmission(
                        CandidateRef("cold", "1" * 64, 0, "baseline"),
                        _Approximation(0.5),
                    ),
                    agents={"agent": agent},
                )

                with self.assertRaises(ValueError):
                    runner.run()

                self.assertEqual(
                    tuple(request.kind for request in source.requests),
                    (GenerationKind.BASE, GenerationKind.PERTURBED),
                )
                self.assertEqual(agent.contexts, [])
                self.assertEqual(runner._seen_case_ids, set())

    def test_published_component_metrics_give_each_paired_root_equal_weight(self):
        class RootMetricMatcher(_AbsoluteMacroMatcher):
            contract_id = "paired_published_metric_v1"

            def match(self, case, prediction, target):
                error = abs(
                    prediction.fields["macro"][0]
                    - target.values.fields["macro"][0]
                )
                return MacroMatchResult(
                    case.case_id,
                    self.contract_id,
                    error,
                    metrics={
                        "root_component": float(case.public_data["difficulty"])
                    },
                )

        result = MacroClosedLoopRunner(
            legacy_macro_reference=True,
            config=MacroLoopConfig(
                name="paired-published-metric",
                round_count=1,
                agents=(MacroAgentSpec("agent", "Agent"),),
                development_case_count=4,
                selection_case_count=4,
                evidence_stages=(EvidenceStage(4, confidence_z=0.0),),
                terminal_case_count=4,
                agent_case_credits=4,
                paired_perturbations=True,
                terminal_confidence_z=0.0,
            ),
            plugin=MacroLoopPlugin(
                "paired-published-metric",
                _PairedCaseSource(
                    pair_difficulties=((0.0, 10.0), (4.0, 4.0))
                ),
                _Objective(
                    RootMetricMatcher(),
                    published_metric_names=("root_component",),
                ),
            ),
            cold_start=CandidateSubmission(
                CandidateRef("cold", "2" * 64, 0, "baseline"),
                _Approximation(0.5),
            ),
            agents={"agent": _ImprovingAgent()},
        ).run()

        summary = result.rounds[0].finalist_evaluations[0].published_metrics
        aggregate = summary.aggregates["root_component"]
        self.assertAlmostEqual(aggregate.mean, 4.5)
        self.assertAlmostEqual(aggregate.cvar, 5.0)
        self.assertAlmostEqual(aggregate.max, 5.0)

    def test_paired_hard_mining_keeps_root_and_perturbed_child_together(self):
        class DifficultyMatcher(_AbsoluteMacroMatcher):
            contract_id = "paired_hardness_v1"

            def match(self, case, prediction, target):
                error = float(case.public_data["difficulty"])
                return MacroMatchResult(
                    case.case_id,
                    self.contract_id,
                    error,
                )

        class CapturingAgent:
            def __init__(self):
                self.cases = ()

            def run(self, context):
                self.cases = context.cases
                raise RuntimeError("stop after paired hard mining")

        source = _PairedCaseSource(
            pair_difficulties=((0.1, 9.0), (8.0, 0.2))
        )
        agent = CapturingAgent()
        result = MacroClosedLoopRunner(
            legacy_macro_reference=True,
            config=MacroLoopConfig(
                name="paired-hard-mining",
                round_count=1,
                agents=(MacroAgentSpec("agent", "Agent"),),
                development_case_count=2,
                selection_case_count=2,
                evidence_stages=(EvidenceStage(2, confidence_z=0.0),),
                terminal_case_count=2,
                agent_case_credits=2,
                development_pool_multiplier=2,
                paired_perturbations=True,
                terminal_confidence_z=0.0,
            ),
            plugin=MacroLoopPlugin(
                "paired-hardness",
                source,
                _Objective(DifficultyMatcher()),
            ),
            cold_start=CandidateSubmission(
                CandidateRef("cold", "2" * 64, 0, "baseline"),
                _Approximation(0.5),
            ),
            agents={"agent": agent},
        ).run()

        expected = (
            "r1-development-root-0",
            "r1-development-root-0-perturbed",
        )
        self.assertEqual(
            tuple(case.case_id for case in agent.cases),
            tuple(candidate_case_id(case_id) for case_id in expected),
        )
        self.assertEqual(result.rounds[0].development_case_ids, expected)
        self.assertTrue(all(
            case.source is CaseSource.GENERATED
            and case.visibility is CaseVisibility.DEVELOPMENT
            and case.round_index is None
            and case.parent_case_id is None
            and case.tags == ()
            for case in agent.cases
        ))

    def test_handoff_digest_binds_candidate_provenance_and_private_evidence(self):
        result, _source, _agent = self._run(_AbsoluteMacroMatcher())
        incumbent = result.cold_start

        for round_result in result.rounds:
            comparison = round_result.promotion.comparisons[-1]
            verification = {
                "run_name": result.name,
                "plugin_id": result.plugin_id,
                "generator_id": result.generator_id,
                "generator_sha256": result.generator_sha256,
                "macro_contract_id": result.macro_contract_id,
                "incumbent": incumbent,
                "evidence_target_sha256": comparison.target_sha256,
                "evidence_contract_id": comparison.contract_id,
            }
            self.assertTrue(
                verify_round_handoff(round_result.handoff, **verification)
            )
            tampered = replace(
                round_result.handoff,
                selected=replace(
                    round_result.handoff.selected,
                    agent_id="tampered-agent",
                ),
            )
            self.assertFalse(verify_round_handoff(tampered, **verification))
            grouping_tampered = replace(
                round_result.handoff,
                feedback=replace(
                    round_result.handoff.feedback,
                    confidence_group_size=2,
                ),
            )
            self.assertFalse(
                verify_round_handoff(grouping_tampered, **verification)
            )
            incumbent = round_result.selected



    def test_generated_primary_uses_root_counts_and_terminal_guardrail(self):
        class TerminalShiftedPairedCaseSource(_PairedCaseSource):
            def sample(self, request):
                records = super().sample(request)
                if request.cohort is not CohortKind.TERMINAL:
                    return records
                return tuple(
                    CaseRecord(
                        replace(
                            record.case,
                            public_data={
                                **record.case.public_data,
                                "difficulty": (
                                    float(record.case.public_data["difficulty"])
                                    + 10.0
                                ),
                            },
                        ),
                        record.truth,
                    )
                    for record in records
                )

        class CohortAwareApproximation:
            def predict_macros(self, case, query, budget=None):
                difficulty = float(case.public_data["difficulty"])
                perturbed = not difficulty.is_integer()
                if perturbed:
                    error = 0.8 if difficulty >= 10.0 else 0.4
                else:
                    error = 0.1
                return AnswerView(
                    kind=query.kind,
                    fields={"macro": (1.0 - error,)},
                    axes={"sample": (0.0,)},
                    shape=(1,),
                )

        class PrimaryAgent:
            def run(self, context):
                candidate_id = "generated-primary-candidate"
                return MacroAgentOutcome(CandidateSubmission(
                    CandidateRef(
                        candidate_id,
                        _digest(candidate_id),
                        context.round_index,
                        context.agent.agent_id,
                        parent_sha256=context.incumbent.bundle_sha256,
                    ),
                    CohortAwareApproximation(),
                ))

        result = MacroClosedLoopRunner(
            legacy_macro_reference=True,
            config=MacroLoopConfig(
                name="generated-primary-terminal-guardrail",
                round_count=1,
                agents=(MacroAgentSpec("agent", "Agent"),),
                development_case_count=4,
                selection_case_count=4,
                evidence_stages=(
                    EvidenceStage(
                        2,
                        confidence_z=0.0,
                        min_win_rate=0.0,
                        require_cvar_no_regression=False,
                    ),
                    EvidenceStage(
                        4,
                        confidence_z=0.0,
                        min_win_rate=0.0,
                        require_cvar_no_regression=False,
                    ),
                ),
                terminal_case_count=4,
                agent_case_credits=4,
                paired_perturbations=True,
                paired_objective_policy=PairedObjectivePolicy(
                    mode=(
                        PairedObjectiveMode.PRIMARY_WITH_PERTURBED_GUARDRAIL
                    ),
                    guardrail_relative_margin=0.05,
                    guardrail_confidence_z=0.0,
                ),
                terminal_confidence_z=0.0,
                terminal_min_win_rate=0.0,
                terminal_require_cvar_no_regression=False,
            ),
            plugin=MacroLoopPlugin(
                "generated-primary",
                TerminalShiftedPairedCaseSource(),
                _Objective(_AbsoluteMacroMatcher()),
            ),
            cold_start=CandidateSubmission(
                CandidateRef("cold", "4" * 64, 0, "baseline"),
                _Approximation(0.5),
            ),
            agents={"agent": PrimaryAgent()},
        ).run()

        round_result = result.rounds[0]
        finalist = round_result.finalist_evaluations[0]
        self.assertEqual(len(finalist.score.case_ids), 2)
        self.assertAlmostEqual(finalist.aggregate.mean, 0.1)
        self.assertAlmostEqual(
            finalist.objective_aggregates.generated.mean,
            0.1,
        )
        self.assertAlmostEqual(
            finalist.objective_aggregates.perturbed.mean,
            0.4,
        )
        self.assertAlmostEqual(
            finalist.objective_aggregates.composite.mean,
            0.25,
        )
        self.assertEqual(
            tuple(
                row.stage.case_count
                for row in round_result.promotion.comparisons
            ),
            (2,),
        )
        self.assertEqual(
            tuple(
                row.confidence_group_size
                for row in round_result.promotion.comparisons
            ),
            (1,),
        )
        self.assertEqual(
            tuple(
                row.case_count
                for row in round_result.promotion_published_metrics
            ),
            (4,),
        )
        self.assertTrue(round_result.promotion.promoted)
        self.assertTrue(round_result.promotion_perturbed_guardrail.passed)
        self.assertEqual(round_result.handoff.feedback.confidence_group_size, 1)
        self.assertTrue(
            round_result.handoff.feedback.perturbed_guardrail_passed
        )

        terminal = result.terminal
        self.assertTrue(terminal.comparison.passed)
        self.assertEqual(terminal.comparison.stage.case_count, 2)
        self.assertEqual(len(terminal.selected_score.case_ids), 2)
        self.assertFalse(terminal.perturbed_guardrail.passed)
        self.assertIn(
            "perturbed_regression_beyond_margin",
            terminal.perturbed_guardrail.failures,
        )
        self.assertFalse(terminal.supports_selected_over_cold_start)
        self.assertFalse(hasattr(terminal.perturbed_guardrail, "case_ids"))
        self.assertFalse(hasattr(terminal.perturbed_guardrail, "root_ids"))

    def test_generated_primary_common_leaderboard_requires_perturbed_guardrail(self):
        class PromotionRegressingApproximation:
            def predict_macros(self, case, query, budget=None):
                difficulty = float(case.public_data["difficulty"])
                error = 0.9 if not difficulty.is_integer() else 0.1
                return AnswerView(
                    kind=query.kind,
                    fields={"macro": (1.0 - error,)},
                    axes={"sample": (0.0,)},
                    shape=(1,),
                )

        class AggressiveAgent:
            def run(self, context):
                candidate_id = "guardrail-rejected-candidate"
                return MacroAgentOutcome(CandidateSubmission(
                    CandidateRef(
                        candidate_id,
                        _digest(candidate_id),
                        context.round_index,
                        context.agent.agent_id,
                        parent_sha256=context.incumbent.bundle_sha256,
                    ),
                    PromotionRegressingApproximation(),
                ))

        result = MacroClosedLoopRunner(
            legacy_macro_reference=True,
            config=MacroLoopConfig(
                name="generated-primary-promotion-guardrail",
                round_count=1,
                agents=(MacroAgentSpec("agent", "Agent"),),
                development_case_count=4,
                selection_case_count=4,
                evidence_stages=(EvidenceStage(
                    4,
                    confidence_z=0.0,
                    min_win_rate=0.0,
                    require_cvar_no_regression=False,
                ),),
                terminal_case_count=4,
                agent_case_credits=4,
                paired_perturbations=True,
                paired_objective_policy=PairedObjectivePolicy(
                    mode=(
                        PairedObjectiveMode.PRIMARY_WITH_PERTURBED_GUARDRAIL
                    ),
                    guardrail_confidence_z=0.0,
                ),
                terminal_confidence_z=0.0,
            ),
            plugin=MacroLoopPlugin(
                "generated-primary",
                _PairedCaseSource(),
                _Objective(_AbsoluteMacroMatcher()),
            ),
            cold_start=CandidateSubmission(
                CandidateRef("cold", "5" * 64, 0, "baseline"),
                _Approximation(0.5),
            ),
            agents={"agent": AggressiveAgent()},
        ).run()

        round_result = result.rounds[0]
        finalist = round_result.finalist_evaluations[0]
        self.assertFalse(finalist.perturbed_guardrail.passed)
        standing = next(
            row
            for row in round_result.leaderboard
            if row.candidate.candidate_id == finalist.candidate.candidate_id
        )
        self.assertFalse(standing.eligible)
        self.assertIsNone(round_result.promotion)
        self.assertIsNone(round_result.promotion_perturbed_guardrail)
        self.assertEqual(result.selected.candidate_id, "cold")

    def test_generated_primary_skips_guardrail_failed_best_finalist(self):
        class DistributionApproximation:
            def __init__(self, base_error, perturbed_error):
                self.base_error = base_error
                self.perturbed_error = perturbed_error

            def predict_macros(self, case, query, budget=None):
                difficulty = float(case.public_data["difficulty"])
                error = (
                    self.perturbed_error
                    if not difficulty.is_integer()
                    else self.base_error
                )
                return AnswerView(
                    kind=query.kind,
                    fields={"macro": (1.0 - error,)},
                    axes={"sample": (0.0,)},
                    shape=(1,),
                )

        class FixedAgent:
            def __init__(self, base_error, perturbed_error):
                self.program = DistributionApproximation(
                    base_error, perturbed_error
                )

            def run(self, context):
                candidate_id = f"candidate-{context.agent.agent_id}"
                return MacroAgentOutcome(CandidateSubmission(
                    CandidateRef(
                        candidate_id,
                        _digest(candidate_id),
                        context.round_index,
                        context.agent.agent_id,
                        parent_sha256=context.incumbent.bundle_sha256,
                    ),
                    self.program,
                ))

        config = MacroLoopConfig(
            name="generated-primary-finalist-guardrail",
            round_count=1,
            agents=(
                MacroAgentSpec("aggressive", "Aggressive"),
                MacroAgentSpec("robust", "Robust"),
            ),
            development_case_count=4,
            selection_case_count=4,
            evidence_stages=(EvidenceStage(
                4,
                confidence_z=0.0,
                min_win_rate=0.0,
                require_cvar_no_regression=False,
            ),),
            terminal_case_count=4,
            agent_case_credits=4,
            paired_perturbations=True,
            paired_objective_policy=PairedObjectivePolicy(
                mode=PairedObjectiveMode.PRIMARY_WITH_PERTURBED_GUARDRAIL,
                guardrail_confidence_z=0.0,
            ),
            terminal_confidence_z=0.0,
            terminal_min_win_rate=0.0,
            terminal_require_cvar_no_regression=False,
        )
        result = MacroClosedLoopRunner(
            legacy_macro_reference=True,
            config=config,
            plugin=MacroLoopPlugin(
                "generated-primary",
                _PairedCaseSource(),
                _Objective(_AbsoluteMacroMatcher()),
            ),
            cold_start=CandidateSubmission(
                CandidateRef("cold", "6" * 64, 0, "baseline"),
                _Approximation(0.5),
            ),
            agents={
                "aggressive": FixedAgent(0.05, 0.9),
                "robust": FixedAgent(0.2, 0.4),
            },
        ).run()

        finalists = {
            row.candidate.agent_id: row
            for row in result.rounds[0].finalist_evaluations
        }
        self.assertFalse(finalists["aggressive"].perturbed_guardrail.passed)
        self.assertTrue(finalists["robust"].perturbed_guardrail.passed)
        self.assertEqual(result.rounds[0].challenger.agent_id, "robust")
        self.assertEqual(result.selected.agent_id, "robust")


    def test_discard_prepared_cohort_retains_failed_owner_for_retry(self):
        class FailOnce:
            def __init__(self):
                self.calls = 0

            def discard(self):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("transient release")

        runner = object.__new__(MacroClosedLoopRunner)
        key = (1, CohortKind.SELECTION)
        cohort = FailOnce()
        runner._prepared_cohorts = {key: cohort}

        with self.assertRaisesRegex(RuntimeError, "transient release"):
            runner._discard_prepared_cohort(*key)
        self.assertIs(runner._prepared_cohorts[key], cohort)

        runner._discard_prepared_cohort(*key)
        self.assertEqual(runner._prepared_cohorts, {})
        self.assertEqual(cohort.calls, 2)

    def test_drain_prepared_cohorts_continues_after_failure(self):
        class Cohort:
            def __init__(self, fail_once=False):
                self.fail_once = fail_once
                self.calls = 0

            def discard(self):
                self.calls += 1
                if self.fail_once and self.calls == 1:
                    raise RuntimeError("transient first release")

        runner = object.__new__(MacroClosedLoopRunner)
        failed_key = (1, CohortKind.SELECTION)
        success_key = (1, CohortKind.PROMOTION)
        failed = Cohort(fail_once=True)
        succeeded = Cohort()
        runner._prepared_cohorts = {
            failed_key: failed,
            success_key: succeeded,
        }

        with self.assertRaisesRegex(RuntimeError, "transient first release"):
            runner._drain_prepared_cohorts()
        self.assertEqual(runner._prepared_cohorts, {failed_key: failed})
        self.assertEqual((failed.calls, succeeded.calls), (1, 1))

        runner._drain_prepared_cohorts()
        self.assertEqual(runner._prepared_cohorts, {})
        self.assertEqual((failed.calls, succeeded.calls), (2, 1))

    def test_drain_development_sessions_continues_and_retries_failed_close(self):
        class Session:
            def __init__(self, fail_once=False):
                self.fail_once = fail_once
                self.calls = 0

            def close(self):
                self.calls += 1
                if self.fail_once and self.calls == 1:
                    raise RuntimeError("transient development-session close")

        runner = object.__new__(MacroClosedLoopRunner)
        failed = Session(fail_once=True)
        succeeded = Session()
        runner._active_development_sessions = {failed, succeeded}

        with self.assertRaisesRegex(
            RuntimeError, "transient development-session close"
        ):
            runner._drain_development_sessions()
        self.assertEqual(runner._active_development_sessions, {failed})
        self.assertEqual((failed.calls, succeeded.calls), (1, 1))

        runner._drain_development_sessions()
        self.assertEqual(runner._active_development_sessions, set())
        self.assertEqual((failed.calls, succeeded.calls), (2, 1))

    def test_run_cleanup_retains_failed_session_and_drains_other_resources(self):
        class Session:
            def __init__(self):
                self.calls = 0

            def close(self):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("transient owned-session close")

        class Cohort:
            def __init__(self):
                self.calls = 0

            def discard(self):
                self.calls += 1

        runner = object.__new__(MacroClosedLoopRunner)
        session = Session()
        cohort = Cohort()
        key = (1, CohortKind.SELECTION)
        runner._active_development_sessions = {session}
        runner._prepared_cohorts = {key: cohort}
        primary = RuntimeError("injected agent or release failure")

        def fail_once():
            raise primary

        runner._run_once = fail_once
        with self.assertRaisesRegex(
            RuntimeError, "injected agent or release failure"
        ) as caught:
            runner.run()

        self.assertIs(caught.exception, primary)
        self.assertEqual(cohort.calls, 1)
        self.assertEqual(runner._prepared_cohorts, {})
        self.assertEqual(runner._active_development_sessions, {session})
        self.assertEqual(session.calls, 1)
        self.assertTrue(
            any(
                "macro-loop resource cleanup also failed" in note
                and "transient owned-session close" in note
                for note in getattr(caught.exception, "__notes__", ())
            )
        )

        runner._drain_owned_resources()
        self.assertEqual(runner._active_development_sessions, set())
        self.assertEqual(session.calls, 2)



if __name__ == "__main__":
    unittest.main()
