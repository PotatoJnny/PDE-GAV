import json
import tempfile
import unittest
from pathlib import Path

from pde_agent_verifier_solver.plugins.gkw import (
    GKWObservableTask,
    GKWProblem,
    GKW_MACRO_FIELDS,
)
from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    Artifact,
    DatasetInput,
    DatasetKind,
    EvaluationPhase,
    TaskKind,
    resolve_evaluation_program,
)

from tests.gkw_release_fixture import write_gkw_release

class _ExactObservable:
    def __init__(self, answer):
        self.answer = answer

    def predict_observables(self, case, query, budget=None):
        return self.answer




class GKWReferencePluginTests(unittest.TestCase):
    def test_problem_keeps_truth_private_and_task_scores_exact_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            development = root / "development"
            hidden = root / "hidden"
            write_gkw_release(development, "gkw-development", 1.0)
            write_gkw_release(hidden, "gkw-hidden", 2.0)
            manifest = root / "release_bank.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema": "fusion_gkw_macro_release_bank_v1",
                        "development_case_dirs": [str(development)],
                        "released_completed_case_dirs": [str(hidden)],
                    }
                ),
                encoding="utf-8",
            )

            prepared = GKWProblem().prepare_dataset(
                DatasetInput(
                    "gkw-test",
                    DatasetKind.NUMERICAL,
                    str(manifest),
                    settings={"development_cases": 1, "hidden_cases": 1},
                )
            )
            record = prepared.development_cases[0]
            self.assertNotIn("case_dir", record.case.public_data)
            self.assertNotIn("released_answer", record.case.public_data)
            self.assertNotIn("particle_density_values", record.case.public_data)
            self.assertEqual(
                record.case.public_data["macro_query"]["field_indices"].tolist(),
                [[0, 0, 0], [1, 1, 1]],
            )

            query = AnswerQuery.grid(
                fields=GKW_MACRO_FIELDS, t=2, field_row=2
            )
            truth = record.truth.answer(query)
            artifact = Artifact(
                "gkw-exact",
                TaskKind.OBSERVABLE_MODEL,
                "exact GKW macros",
                _ExactObservable(truth),
                "test",
                "predict_observables(case, query)",
                0,
                "test-agent",
            )
            task = GKWObservableTask()
            direct = task.evaluate(artifact, record.case, truth, query)
            program = resolve_evaluation_program(
                task,
                {"default_query": query},
            )
            plan = program.make_plan(
                record.case,
                EvaluationPhase.OFFICIAL,
                {"default_query": query},
                seed=23,
            )
            native = program.evaluate(artifact, record.case, truth, plan)
            self.assertEqual(program.program_id, "gkw-observable-model-evaluation-v1")
            self.assertEqual(native, direct)
            self.assertEqual(native.primary_error, 0.0)
            aggregate = program.aggregate(artifact, (native,), (plan,))
            self.assertEqual(aggregate.primary_error, direct.primary_error)
            self.assertEqual(aggregate.metrics, direct.metrics)

            point_query = AnswerQuery.at_points(
                fields=GKW_MACRO_FIELDS,
                points=(
                    {
                        "t": 1.0,
                        "s_index": 1,
                        "kx_index": 1,
                        "ky_index": 1,
                    },
                ),
            )
            point_truth = record.truth.answer(point_query)
            self.assertEqual(point_truth.shape, (1, 2))
            self.assertEqual(point_truth.fields["phi"], (1.0, 0.0))

    def test_release_hidden_state_policy_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            development = root / "development"
            hidden = root / "hidden"
            write_gkw_release(development, "gkw-development", 1.0)
            write_gkw_release(hidden, "gkw-hidden", 2.0)
            manifest_path = development / "manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["full_hidden_state_included"] = True
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            release = root / "release.json"
            release.write_text(
                json.dumps(
                    {
                        "development_case_dirs": [str(development)],
                        "released_completed_case_dirs": [str(hidden)],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "completion/privacy policy"):
                GKWProblem().prepare_dataset(
                    DatasetInput(
                        "gkw-test",
                        DatasetKind.NUMERICAL,
                        str(release),
                        settings={"development_cases": 1, "hidden_cases": 1},
                    )
                )


if __name__ == "__main__":
    unittest.main()
