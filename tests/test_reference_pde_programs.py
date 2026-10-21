import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from pde_agent_verifier_solver.plugins.gkw import (
    GKW_LEGACY_MACRO_OUTPUT_CONTRACT,
    GKW_MACRO_FIELDS,
    GKWObservableTask,
)
from pde_agent_verifier_solver.plugins.ns import NSProblem, NSSolverTask
from pde_agent_verifier_solver.plugins.vpfp import (
    VPFPObservableTask,
    VPFPProblem,
    VPFPReducedDynamicsTask,
    VPFPStatisticsTask,
)
from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    Artifact,
    Case,
    CaseSource,
    CaseVisibility,
    DatasetInput,
    DatasetKind,
    EvaluationPhase,
    QueryKind,
    TaskKind,
    resolve_evaluation_program,
)


class _ExactNSSolver:
    def __init__(self, answer):
        self.answer = answer

    def solve(self, case, query, budget=None):
        return self.answer


class _ExactObservableModel:
    def __init__(self, answer):
        self.answer = answer

    def predict_observables(self, case, query, budget=None):
        return self.answer


class _ExactReducedModel:
    def __init__(self, answer):
        self.answer = answer

    def initialize(self, case):
        return {"case": case.case_id}

    def advance(self, state, query_times, budget=None):
        return {"state": state, "times": tuple(query_times)}

    def observe(self, reduced_trajectory, query):
        return self.answer


class _ExactStatisticsModel:
    def __init__(self, values):
        self.values = values

    def statistics(self, case, requested, settings):
        return {name: self.values[name] for name in requested}


class ReferencePDEProgramTests(unittest.TestCase):
    def test_formal_product_tasks_resolve_native_evaluation_programs(self) -> None:
        default_query = AnswerQuery.grid(fields=("field",), x=2)
        requested_query = AnswerQuery.at_points(
            fields=("field",),
            points=({"x": 0.5},),
        )
        case = Case(
            "evaluation-plan-case",
            CaseSource.INITIAL,
            CaseVisibility.DEVELOPMENT,
            public_data={},
        )
        expected_ids = {
            NSSolverTask: "ns-full-field-solver-evaluation-v1",
            VPFPObservableTask: "vpfp-observable-model-evaluation-v1",
            VPFPReducedDynamicsTask: "vpfp-reduced-dynamics-evaluation-v1",
            VPFPStatisticsTask: "vpfp-statistics-evaluation-v1",
            GKWObservableTask: "gkw-observable-model-evaluation-v1",
        }

        for task_type, expected_id in expected_ids.items():
            with self.subTest(task=task_type.__name__):
                program = resolve_evaluation_program(
                    task_type(),
                    {"default_query": default_query},
                )
                self.assertEqual(program.program_id, expected_id)
                self.assertNotIn("legacy", program.program_id)
                plan = program.make_plan(
                    case,
                    EvaluationPhase.DEVELOPMENT,
                    {"comparison": "shared"},
                    seed=17,
                    request=requested_query,
                )
                self.assertEqual(plan.query, requested_query)
                self.assertIsNot(plan.query, requested_query)
                self.assertEqual(plan.seed, 17)
                self.assertEqual(plan.settings["comparison"], "shared")
                with self.assertRaises(FrozenInstanceError):
                    plan.seed = 18
                with self.assertRaises(TypeError):
                    plan.settings["comparison"] = "changed"
                with self.assertRaises(TypeError):
                    plan.query.points[0]["x"] = 0.75

    def test_ns_problem_and_solver_task_use_the_shared_contracts(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case = root / "ns_case"
            case.mkdir()
            (case / "public_question.json").write_text(
                json.dumps(
                    {
                        "viscosity": 0.01,
                        "geometry": {"geometry_id": "test"},
                        "time": {"t0": 0.0, "t1": 1.0},
                    }
                )
            )
            (case / "case_spec.json").write_text(
                json.dumps({"family": "numerical", "parameters": {"drive": 1.0}})
            )
            np.savez(
                case / "public_arrays.npz",
                x=[0.0, 1.0],
                y=[0.0, 1.0],
                fluid_mask=np.ones((2, 2), dtype=bool),
                initial_velocity=np.zeros((2, 2, 2)),
                forcing=np.zeros((2, 2, 2, 2)),
            )
            velocity = np.arange(16, dtype=float).reshape(2, 2, 2, 2)
            np.savez(
                case / "hidden_truth_cache.npz",
                u_true=velocity,
                pressure=np.ones((2, 2, 2)),
                t=[0.0, 1.0],
            )
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "splits": {
                            "validation": [{"case_id": "ns-1", "case_dir": str(case)}],
                            "d0_distribution_test": [
                                {"case_id": "ns-1", "case_dir": str(case)}
                            ],
                        }
                    }
                )
            )
            prepared = NSProblem().prepare_dataset(
                DatasetInput(
                    "ns-test",
                    DatasetKind.NUMERICAL,
                    str(manifest),
                    settings={"development_cases": 1, "hidden_cases": 1},
                )
            )
            record = prepared.development_cases[0]
            query = AnswerQuery.grid(
                fields=("velocity_x", "velocity_y", "pressure"),
                t=2,
                x=2,
                y=2,
            )
            truth = record.truth.answer(query)
            artifact = Artifact(
                "ns-artifact",
                TaskKind.SOLVER,
                "exact NS",
                _ExactNSSolver(truth),
                "test",
                "solve(case, query)",
                0,
                "test-agent",
            )
            task = NSSolverTask()
            direct = task.evaluate(artifact, record.case, truth, query)
            program = resolve_evaluation_program(
                task,
                {"default_query": query},
            )
            plan = program.make_plan(
                record.case,
                EvaluationPhase.OFFICIAL,
                {"default_query": query},
                seed=5,
            )
            native = program.evaluate(artifact, record.case, truth, plan)
            self.assertEqual(native, direct)
            self.assertEqual(native.primary_error, 0.0)
            aggregate = program.aggregate(artifact, (native,), (plan,))
            self.assertEqual(aggregate.primary_error, direct.primary_error)
            self.assertEqual(aggregate.metrics, direct.metrics)
            private_settings = {
                "default_query": query,
                "development_point_count": 5,
            }
            first = program.make_private_development_plan(
                record.case,
                private_settings,
                seed=37,
            )
            repeated = program.make_private_development_plan(
                record.case,
                private_settings,
                seed=37,
            )
            changed = program.make_private_development_plan(
                record.case,
                private_settings,
                seed=38,
            )
            self.assertIs(first.phase, EvaluationPhase.DEVELOPMENT)
            self.assertIs(first.query.kind, QueryKind.POINTS)
            self.assertEqual(first.query, repeated.query)
            self.assertNotEqual(first.query, changed.query)
            self.assertEqual(len(first.query.points), 5)
            for point in first.query.points:
                self.assertLessEqual(0.0, point["t"])
                self.assertLessEqual(point["t"], 1.0)
                self.assertLessEqual(0.0, point["x"])
                self.assertLessEqual(point["x"], 1.0)
                self.assertLessEqual(0.0, point["y"])
                self.assertLessEqual(point["y"], 1.0)
            self.assertNotIn("case_dir", record.case.public_data)
            self.assertIn("forcing", record.case.public_data["arrays"])
            self.assertNotIn("u_true", record.case.public_data["arrays"])

    def test_ns_private_development_requires_public_time_bounds(self) -> None:
        query = AnswerQuery.grid(fields=("velocity_x", "velocity_y"), t=2, x=2, y=2)
        program = resolve_evaluation_program(
            NSSolverTask(),
            {"default_query": query},
        )
        case = Case(
            "ns-missing-public-time",
            CaseSource.INITIAL,
            CaseVisibility.DEVELOPMENT,
            public_data={
                "question": {},
                "arrays": {
                    "x": (0.0, 1.0),
                    "y": (0.0, 1.0),
                    "fluid_mask": ((True, True), (True, True)),
                },
            },
        )

        with self.assertRaisesRegex(ValueError, "question.time t0/t1"):
            program.make_private_development_plan(
                case,
                {"default_query": query, "development_point_count": 3},
                seed=1,
            )

    def test_ns_private_development_excludes_the_public_obstacle_mask(self) -> None:
        query = AnswerQuery.grid(fields=("velocity_x", "velocity_y"), t=2, x=3, y=3)
        settings = {"default_query": query, "development_point_count": 64}
        program = resolve_evaluation_program(NSSolverTask(), settings)
        case = Case(
            "ns-public-obstacle-mask",
            CaseSource.INITIAL,
            CaseVisibility.DEVELOPMENT,
            public_data={
                "question": {"time": {"t0": 0.0, "t1": 2.0}},
                "arrays": {
                    "x": (0.0, 0.5, 1.0),
                    "y": (0.0, 0.5, 1.0),
                    "fluid_mask": (
                        (True, True, True),
                        (True, False, True),
                        (True, True, True),
                    ),
                },
            },
        )

        plan = program.make_private_development_plan(case, settings, seed=19)

        self.assertEqual(len(plan.query.points), 64)
        self.assertTrue(
            all(
                (float(point["x"]), float(point["y"])) != (0.5, 0.5)
                for point in plan.query.points
            )
        )
        self.assertTrue(
            all(0.0 <= float(point["t"]) <= 2.0 for point in plan.query.points)
        )

    def test_one_vpfp_problem_supports_three_independent_tasks(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            development = root / "development"
            released = root / "released"
            for index, case in enumerate((development, released)):
                case.mkdir()
                (case / "public_question.json").write_text(
                    json.dumps(
                        {
                            "case_id": f"vpfp-{index}",
                            "collision_nu": 12.0,
                            "temperature": 1.0,
                        }
                    )
                )
                np.savez(
                    case / "public_arrays.npz",
                    t=[0.0, 1.0],
                    x=[0.0, 1.0],
                    y=[0.0, 1.0],
                    rho0=np.ones((2, 2)),
                    source=np.zeros((2, 2, 2, 1, 1)),
                )
                rho = np.ones((2, 2, 2)) * (index + 1)
                momentum = np.stack((rho * 0.5, rho * -0.25), axis=-1)
                np.savez(
                    case / "hidden_truth_cache.npz",
                    rho_true=rho,
                    momentum_true=momentum,
                    kinetic_energy_true=rho * 2.0,
                )
            manifest = root / "release.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema": "vpfp_long_time_answer_first_release_bank_v2",
                        "development_case_dirs": [str(development)],
                        "released_completed_case_dirs": [str(released)],
                    }
                )
            )
            prepared = VPFPProblem().prepare_dataset(
                DatasetInput(
                    "vpfp-test",
                    DatasetKind.NUMERICAL,
                    str(manifest),
                    settings={"development_cases": 1, "hidden_cases": 1},
                )
            )
            record = prepared.development_cases[0]
            self.assertIs(record.case.source, CaseSource.INITIAL)
            self.assertIn("source", record.case.public_data["arrays"])
            self.assertNotIn("rho_true", record.case.public_data["arrays"])
            self.assertIs(
                prepared.hidden_cases[0].case.source,
                CaseSource.INITIAL,
            )
            query = AnswerQuery.grid(
                fields=("rho", "momentum_x", "momentum_y", "kinetic_energy"),
                t=2,
                x=2,
                y=2,
            )
            truth = record.truth.answer(query)

            observable = Artifact(
                "observable",
                TaskKind.OBSERVABLE_MODEL,
                "exact observable",
                _ExactObservableModel(truth),
                "test",
                "predict_observables(case, query)",
                0,
                "agent",
            )
            observable_task = VPFPObservableTask()
            observable_direct = observable_task.evaluate(
                observable, record.case, truth, query
            )

            reduced = Artifact(
                "reduced",
                TaskKind.REDUCED_DYNAMICS,
                "exact reduced",
                _ExactReducedModel(truth),
                "test",
                "initialize/advance/observe",
                0,
                "agent",
                metadata={"state_dimension": 1},
            )
            reduced_task = VPFPReducedDynamicsTask()
            reduced_direct = reduced_task.evaluate(
                reduced, record.case, truth, query
            )

            statistics_values = {"rho.mean": 1.0, "kinetic_energy.rms": 2.0}
            statistics = Artifact(
                "statistics",
                TaskKind.STATISTICS,
                "exact statistics",
                _ExactStatisticsModel(statistics_values),
                "test",
                "statistics(case, requested, settings)",
                0,
                "agent",
                metadata={"requested_statistics": tuple(statistics_values)},
            )
            statistics_task = VPFPStatisticsTask()
            statistics_direct = statistics_task.evaluate(
                statistics, record.case, truth, query
            )

            for task, artifact, direct in (
                (observable_task, observable, observable_direct),
                (reduced_task, reduced, reduced_direct),
                (statistics_task, statistics, statistics_direct),
            ):
                with self.subTest(task=task.info.program_id):
                    program = resolve_evaluation_program(
                        task,
                        {"default_query": query},
                    )
                    plan = program.make_plan(
                        record.case,
                        EvaluationPhase.OFFICIAL,
                        {"default_query": query},
                        seed=11,
                    )
                    native = program.evaluate(
                        artifact,
                        record.case,
                        truth,
                        plan,
                    )
                    self.assertEqual(native, direct)
                    self.assertEqual(native.primary_error, 0.0)
                    aggregate = program.aggregate(
                        artifact,
                        (native,),
                        (plan,),
                    )
                    self.assertEqual(
                        aggregate.primary_error,
                        direct.primary_error,
                    )
                    self.assertEqual(aggregate.metrics, direct.metrics)

            private_settings = {
                "default_query": query,
                "development_point_count": 6,
            }
            for task in (observable_task, reduced_task):
                with self.subTest(private_task=task.info.program_id):
                    program = resolve_evaluation_program(task, private_settings)
                    first = program.make_private_development_plan(
                        record.case,
                        private_settings,
                        seed=71,
                    )
                    repeated = program.make_private_development_plan(
                        record.case,
                        private_settings,
                        seed=71,
                    )
                    changed = program.make_private_development_plan(
                        record.case,
                        private_settings,
                        seed=72,
                    )
                    self.assertIs(first.query.kind, QueryKind.POINTS)
                    self.assertEqual(first.query, repeated.query)
                    self.assertNotEqual(first.query, changed.query)
                    self.assertEqual(len(first.query.points), 6)
                    for point in first.query.points:
                        self.assertEqual(set(point), {"t", "x", "y"})
                        self.assertTrue(
                            all(0.0 <= float(value) <= 1.0 for value in point.values())
                        )

            statistics_program = resolve_evaluation_program(
                statistics_task,
                private_settings,
            )
            with self.assertRaisesRegex(ValueError, "does not support"):
                statistics_program.make_private_development_plan(
                    record.case,
                    private_settings,
                    seed=71,
                )

    def test_gkw_private_development_samples_the_released_public_panel(self) -> None:
        import numpy as np

        times = (0.0, 0.5, 1.0)
        field_indices = ((0, 0, 0), (1, 1, 0))
        case = Case(
            "gkw-private-development",
            CaseSource.INITIAL,
            CaseVisibility.DEVELOPMENT,
            public_data={
                "arrays": {
                    "s": np.asarray((0.0, 1.0)),
                    "kx": np.asarray((0.0, 1.0)),
                    "ky": np.asarray((0.0,)),
                    "B": np.ones(2),
                    "ints": np.ones(2),
                    "parseval_ky": np.ones(1),
                },
                "macro_query": {
                    "times": times,
                    "field_indices": field_indices,
                    "output_contract": GKW_LEGACY_MACRO_OUTPUT_CONTRACT,
                },
            },
        )
        query = AnswerQuery.grid(
            fields=GKW_MACRO_FIELDS,
            t=len(times),
            field_row=len(field_indices),
        )
        settings = {"default_query": query, "development_point_count": 4}
        program = resolve_evaluation_program(GKWObservableTask(), settings)

        first = program.make_private_development_plan(case, settings, seed=101)
        repeated = program.make_private_development_plan(case, settings, seed=101)
        changed = program.make_private_development_plan(case, settings, seed=102)

        self.assertIs(first.query.kind, QueryKind.POINTS)
        self.assertEqual(first.query, repeated.query)
        self.assertNotEqual(first.query, changed.query)
        self.assertEqual(len(first.query.points), 4)
        allowed = {
            (float(time), tuple(int(value) for value in row))
            for time in times
            for row in field_indices
        }
        selected = {
            (
                float(point["t"]),
                (
                    int(point["s_index"]),
                    int(point["kx_index"]),
                    int(point["ky_index"]),
                ),
            )
            for point in first.query.points
        }
        self.assertEqual(len(selected), 4)
        self.assertTrue(selected <= allowed)

        official = program.make_plan(
            case,
            EvaluationPhase.OFFICIAL,
            settings,
            seed=101,
        )
        self.assertEqual(official.query, query)

        continuous_case = Case(
            "gkw-generated-private-development",
            CaseSource.GENERATED,
            CaseVisibility.DEVELOPMENT,
            public_data={
                "arrays": case.public_data["arrays"],
                "oracle_protocol": {
                    "time_interval_closed": (2.0, 3.0),
                    "field_grid_shape": (2, 2, 1),
                },
            },
        )
        continuous = program.make_private_development_plan(
            continuous_case,
            settings,
            seed=103,
        )
        self.assertEqual(len(continuous.query.points), 4)
        for point in continuous.query.points:
            self.assertLessEqual(2.0, float(point["t"]))
            self.assertLessEqual(float(point["t"]), 3.0)
            self.assertIn(int(point["s_index"]), (0, 1))
            self.assertIn(int(point["kx_index"]), (0, 1))
            self.assertEqual(int(point["ky_index"]), 0)


if __name__ == "__main__":
    unittest.main()
