import hashlib
import json
import tempfile
from pathlib import Path
from types import MappingProxyType
import unittest

from pde_agent_verifier_solver.engine import (
    EvidenceStage,
    MacroAgentOutcome,
    MacroAgentSpec,
    MacroClosedLoopRunner,
    MacroLoopConfig,
    ControllerMacroEvaluator,
    PreparedMacroCohort,
)
from pde_agent_verifier_solver.plugins.gkw import (
    GKWAnswerFirstGenerator,
    GKWMacroMatcher,
    GKWProblem,
    GKW_LEGACY_MACRO_OUTPUT_CONTRACT,
    GKW_MACRO_FIELDS,
    bind_gkw_macro_loop,
    gkw_native_macro_query,
)
from pde_agent_verifier_solver.plugins.gkw.contract import (
    gkw_answer_view_from_pairs,
    gkw_macro_query_panel,
)
from pde_agent_verifier_solver.sdk import (
    CandidateRef,
    CandidateSubmission,
    Case,
    CaseRecord,
    CaseSource,
    CaseVisibility,
    DatasetInput,
    DatasetKind,
)

from tests.gkw_release_fixture import write_gkw_release


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _case(case_id, visibility, round_index):
    import numpy as np

    return Case(
        case_id=case_id,
        source=CaseSource.GENERATED,
        visibility=visibility,
        public_data={
            "arrays": {
                "s": np.asarray([0.0, 1.0]),
                "kx": np.asarray([-1.0, 1.0]),
                "ky": np.asarray([0.0, 2.0]),
                "B": np.asarray([2.0, 4.0]),
                "ints": np.asarray([1.0, 0.5]),
                "parseval_ky": np.asarray([1.0, 2.0]),
            },
            "macro_query": {
                "times": np.asarray([0.5, 1.0]),
                "field_indices": np.asarray([[0, 0, 0], [1, 1, 1]]),
                "output_contract": GKW_LEGACY_MACRO_OUTPUT_CONTRACT,
            },
        },
        round_index=round_index,
        tags=("gkw", "generated"),
    )


def _view(case, query, scale):
    import numpy as np

    times, indices = gkw_macro_query_panel(case)
    values = np.full((len(times), len(indices), 2), float(scale))
    return gkw_answer_view_from_pairs(
        query,
        {field: values.copy() for field in GKW_MACRO_FIELDS},
        times=times,
        field_indices=indices,
    )


class _Truth:
    def __init__(self, case):
        self.case = case

    def answer(self, query):
        return _view(self.case, query, 1.0)


class _Backend:
    def __init__(self):
        self.serial = 0
        self.generate_calls = []

    def train_base(self, *args, **kwargs):
        raise AssertionError("shared GKW loop must not train the frozen Generator")

    def fit_guidance(self, *args, **kwargs):
        raise AssertionError("shared GKW loop must not update Generator guidance")

    def generate(self, base_state, guidance_state, request, progress=None):
        self.generate_calls.append((base_state, guidance_state, request))
        records = []
        for _index in range(request.count):
            self.serial += 1
            case = _case(
                f"gkw-fixed-{self.serial}",
                request.visibility,
                request.round_index,
            )
            records.append(CaseRecord(case, _Truth(case)))
        return tuple(records)


class _MacroProgram:
    def __init__(self, scale):
        self.scale = float(scale)

    def predict_macros(self, case, query, budget=None):
        if budget is not None:
            raise AssertionError("test uses the prebound reference budget")
        return _view(case, query, self.scale)


class _Agent:
    def run(self, context):
        candidate_id = f"gkw-r{context.round_index}"
        candidate = CandidateSubmission(
            CandidateRef(
                candidate_id=candidate_id,
                bundle_sha256=_digest(candidate_id),
                round_index=context.round_index,
                agent_id=context.agent.agent_id,
                parent_sha256=context.incumbent.bundle_sha256,
            ),
            _MacroProgram(0.75),
        )
        response = context.evaluator.evaluate(
            candidate_id,
            candidate.program,
            case_ids=tuple(case.case_id for case in context.cases),
        )
        self.metric_names = tuple(response.cases[0].score.metrics)
        return MacroAgentOutcome(candidate)


class GKWMacroLoopBindingTests(unittest.TestCase):
    def test_fixed_gkw_generator_and_matcher_use_shared_controller(self):
        backend = _Backend()
        base_state = {"checkpoint": "trained-gkw-release"}
        plugin = bind_gkw_macro_loop(
            generator=GKWAnswerFirstGenerator(backend),
            base_state=base_state,
            generator_sha256="a" * 64,
            query_factory=gkw_native_macro_query,
        )
        cold_start = CandidateSubmission(
            CandidateRef(
                candidate_id="gkw-cold",
                bundle_sha256="b" * 64,
                round_index=0,
                agent_id="baseline",
            ),
            _MacroProgram(0.5),
        )
        agent = _Agent()

        result = MacroClosedLoopRunner(
            legacy_macro_reference=True,
            config=MacroLoopConfig(
                name="gkw-thin-binding",
                round_count=1,
                agents=(MacroAgentSpec("agent", "Agent"),),
                development_case_count=1,
                selection_case_count=1,
                evidence_stages=(
                    EvidenceStage(
                        1,
                        confidence_z=0.0,
                        min_win_rate=1.0,
                        require_cvar_no_regression=True,
                    ),
                ),
                terminal_case_count=1,
                agent_case_credits=1,
                terminal_confidence_z=0.0,
                terminal_min_win_rate=1.0,
            ),
            plugin=plugin,
            cold_start=cold_start,
            agents={"agent": agent},
        ).run()

        self.assertIsInstance(plugin.objective.matcher, GKWMacroMatcher)
        self.assertEqual(result.macro_contract_id, GKWMacroMatcher.contract_id)
        expected_published = {
            "particle_density_relative_l2",
            "parallel_momentum_relative_l2",
            "kinetic_energy_relative_l2",
            "phi_relative_l2",
        }
        self.assertEqual(set(result.published_metric_names), expected_published)
        self.assertEqual(
            set(result.terminal.selected_published_metrics.aggregates),
            expected_published,
        )
        self.assertTrue(result.rounds[0].promotion.promoted)
        self.assertEqual(result.selected.candidate_id, "gkw-r1")
        self.assertTrue(result.terminal.comparison.passed)
        self.assertIn("gkw_macro_match_v1_relative_l2", agent.metric_names)
        self.assertTrue(backend.generate_calls)
        self.assertTrue(
            all(state is base_state for state, _guidance, _request in backend.generate_calls)
        )
        self.assertTrue(
            all(guidance is None for _state, guidance, _request in backend.generate_calls)
        )
        self.assertTrue(
            all(
                request.settings["fixed_generator_sha256"] == "a" * 64
                for _state, _guidance, request in backend.generate_calls
            )
        )

    def test_paired_loop_rejects_unsupported_gkw_perturbation_before_backend_call(self):
        backend = _Backend()
        plugin = bind_gkw_macro_loop(
            generator=GKWAnswerFirstGenerator(backend),
            base_state={"checkpoint": "trained-gkw-release"},
            generator_sha256="a" * 64,
        )
        cold_start = CandidateSubmission(
            CandidateRef(
                candidate_id="gkw-cold",
                bundle_sha256="b" * 64,
                round_index=0,
                agent_id="baseline",
            ),
            _MacroProgram(0.5),
        )

        with self.assertRaisesRegex(
            ValueError,
            "case source does not support paired perturbations",
        ):
            MacroClosedLoopRunner(
                legacy_macro_reference=True,
                config=MacroLoopConfig(
                    name="gkw-paired-capability-gate",
                    round_count=1,
                    agents=(MacroAgentSpec("agent", "Agent"),),
                    development_case_count=2,
                    selection_case_count=2,
                    evidence_stages=(
                        EvidenceStage(
                            2,
                            confidence_z=0.0,
                            min_win_rate=1.0,
                            require_cvar_no_regression=True,
                        ),
                    ),
                    terminal_case_count=2,
                    agent_case_credits=2,
                    paired_perturbations=True,
                    terminal_confidence_z=0.0,
                    terminal_min_win_rate=1.0,
                ),
                plugin=plugin,
                cold_start=cold_start,
                agents={"agent": _Agent()},
            ).run()

        self.assertEqual(backend.generate_calls, [])

    def test_native_query_uses_the_released_public_panel(self):
        case = _case("query", CaseVisibility.DEVELOPMENT, 1)
        query = gkw_native_macro_query(case, "development")
        self.assertEqual(query.fields, GKW_MACRO_FIELDS)
        self.assertEqual(query.resolution, {"t": 2, "field_row": 2})
        with self.assertRaisesRegex(ValueError, "unknown GKW"):
            gkw_native_macro_query(case, "historical-private-panel")


    def test_real_release_mappingproxy_enters_shared_evaluator(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            development_dir = root / "development"
            hidden_dir = root / "hidden"
            write_gkw_release(development_dir, "gkw-real-development", 1.0)
            write_gkw_release(hidden_dir, "gkw-real-hidden", 2.0)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema": "gkw_test_release_bank_v1",
                        "cohorts": {
                            "development": [
                                {
                                    "case_id": "gkw-real-development",
                                    "case_dir": str(development_dir),
                                    "source": "generated",
                                }
                            ],
                            "released_training": [
                                {
                                    "case_id": "gkw-real-hidden",
                                    "case_dir": str(hidden_dir),
                                    "source": "generated",
                                }
                            ],
                        },
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            dataset = GKWProblem().prepare_dataset(
                DatasetInput(
                    dataset_id="gkw-real-release",
                    kind=DatasetKind.NUMERICAL,
                    location=str(manifest_path),
                    settings={
                        "development_cases": 1,
                        "hidden_cases": 1,
                    },
                )
            )
            record = dataset.development_cases[0]
            self.assertIsInstance(record.case.public_data, MappingProxyType)
            self.assertFalse(record.case.public_data["arrays"]["B"].flags.writeable)

            cohort = PreparedMacroCohort.prepare(
                cohort_id="gkw-real-release-cohort",
                records=(record,),
                matcher=GKWMacroMatcher(),
                query=lambda case: gkw_native_macro_query(case, "development"),
            )

            class ExactReleasedPanelProgram:
                def predict_macros(self, case, query, budget=None):
                    import numpy as np

                    self.case = case
                    times, indices = gkw_macro_query_panel(case)
                    pairs = np.zeros((len(times), len(indices), 2))
                    pairs[..., 0] = 1.0
                    return gkw_answer_view_from_pairs(
                        query,
                        {
                            field: pairs.copy()
                            for field in GKW_MACRO_FIELDS
                        },
                        times=times,
                        field_indices=indices,
                    )

            program = ExactReleasedPanelProgram()
            result = ControllerMacroEvaluator(cohort).evaluate("exact", program)
            self.assertTrue(result.accepted)
            self.assertAlmostEqual(result.scores[0].primary_error, 0.0)
            self.assertIsInstance(program.case.public_data, dict)
            self.assertTrue(program.case.public_data["arrays"]["B"].flags.writeable)

            exposed = cohort.public_cases()[0]
            exposed.public_data["arrays"]["B"][0] = -99.0
            fresh = cohort.public_cases()[0]
            self.assertEqual(float(fresh.public_data["arrays"]["B"][0]), 2.0)
            self.assertEqual(
                float(record.case.public_data["arrays"]["B"][0]),
                2.0,
            )

if __name__ == "__main__":
    unittest.main()
