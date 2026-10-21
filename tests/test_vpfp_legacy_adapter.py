import copy
from pathlib import Path
import sys
import unittest

from pde_agent_verifier_solver.plugins.vpfp.legacy_adapter import (
    LEGACY_COMPUTE_BUDGET_CONTRACT,
    LEGACY_SOLVER_OUTPUT_CONTRACT,
    VPFPLegacyObservableAdapter,
)
from pde_agent_verifier_solver.plugins.vpfp.macro_matching import (
    VPFP_MACRO_FIELDS,
)
from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    Case,
    CaseSource,
    CaseVisibility,
    QueryKind,
)


ROOT = Path(__file__).resolve().parents[1]
LEGACY_SRC = ROOT / "legacy" / "vpfp" / "src"
PACKAGE_PREFIX = "agent_verifier"


def _load_real_legacy_contracts():
    preserved = {
        name: module
        for name, module in sys.modules.items()
        if name == PACKAGE_PREFIX or name.startswith(PACKAGE_PREFIX + ".")
    }
    for name in preserved:
        sys.modules.pop(name)
    sys.path.insert(0, str(LEGACY_SRC))
    try:
        from agent_verifier.vpfp.compute_budget import (
            COMPUTE_BUDGET_CONTRACT as real_compute_budget_contract,
            run_budgeted_submission as real_runner,
        )
        from agent_verifier.vpfp.continuous_macros import (
            MACRO_OUTPUT_CONTRACT as real_solver_output_contract,
        )
        loaded = (
            real_compute_budget_contract,
            real_solver_output_contract,
            real_runner,
        )
    finally:
        sys.path.pop(0)
        for name in tuple(sys.modules):
            if name == PACKAGE_PREFIX or name.startswith(PACKAGE_PREFIX + "."):
                sys.modules.pop(name)
        sys.modules.update(preserved)
    return loaded


(
    REAL_COMPUTE_BUDGET_CONTRACT,
    REAL_SOLVER_OUTPUT_CONTRACT,
    run_budgeted_submission,
) = _load_real_legacy_contracts()


def _budget(**overrides):
    value = {
        "schema": "vpfp_solver_compute_budget_v1",
        "tier": 1,
        "wall_time_seconds": 30.0,
        "cpus": 2,
        "gpus": 0,
        "memory_bytes": 1024,
    }
    value.update(overrides)
    return value


def _case() -> Case:
    import numpy as np

    nx, ny, nv, nt = 2, 3, 3, 3
    arrays = {
        "x": np.asarray([0.0, 0.5], dtype=np.float32),
        "y": np.asarray([0.0, 1.0, 2.0]),
        "vx": np.asarray([-1.0, 0.0, 1.0]),
        "vy": np.asarray([-1.0, 0.0, 1.0]),
        "t": np.asarray([0.0, 0.5, 1.0]),
        "velocity_weights": np.ones((nv, nv)),
        "rho0": np.ones((nx, ny)),
        "g0": np.ones((nx, ny, nv, nv)),
        "source": np.ones((nt, nx, ny, nv, nv)),
        "momentum0": np.ones((nx, ny, 2)),
        "kinetic_energy0": np.ones((nx, ny)),
    }
    return Case(
        case_id="legacy-case",
        source=CaseSource.INITIAL,
        visibility=CaseVisibility.DEVELOPMENT,
        public_data={
            "equation": "2D2V VPFP",
            "question": {
                "case_id": "legacy-case",
                "solver_output_contract_version": REAL_SOLVER_OUTPUT_CONTRACT,
                "grid": {
                    "nx": nx,
                    "ny": ny,
                    "nvx": nv,
                    "nvy": nv,
                    "x_domain": [1.0, 3.0],
                    "v_bounds": [-1.0, 1.0, -1.0, 1.0],
                    "dtype": "float64",
                },
                "time": {"t0": 0.0, "t1": 1.0, "nt": nt},
                "collision_nu": 0.25,
                "nested": {"unchanged": True},
            },
            "arrays": arrays,
        },
    )


class _Submission:
    def solve_with_budget(self, public_case, query_times, budget):
        raise AssertionError("the injected test runner owns invocation")


class _CapturingRunner:
    def __init__(self, *, aliases=False):
        self.calls = []
        self.aliases = aliases

    def __call__(self, submission, public_case, query_times, budget):
        import numpy as np

        self.calls.append(
            (submission, public_case, query_times.copy(), dict(budget))
        )
        nt = len(query_times)
        nx = public_case["grid"].nx
        ny = public_case["grid"].ny
        rho = np.arange(nt * nx * ny, dtype=float).reshape(nt, nx, ny)
        momentum = np.stack((rho + 100.0, rho + 200.0), axis=-1)
        energy = rho + 300.0
        output = (
            {"rho_pred": rho, "j": momentum, "K": energy}
            if self.aliases
            else {
                "rho": rho,
                "momentum": momentum,
                "kinetic_energy": energy,
            }
        )
        return output, 0.125


class VPFPLegacyObservableAdapterTests(unittest.TestCase):
    def test_adapts_structured_legacy_call_to_grid_answer_view(self) -> None:
        import numpy as np

        case = _case()
        original_question = copy.deepcopy(case.public_data["question"])
        original_x = case.public_data["arrays"]["x"].copy()
        runner = _CapturingRunner()
        legacy_submission = _Submission()
        adapter = VPFPLegacyObservableAdapter(
            legacy_submission,
            _budget(),
            runner,
        )
        query = AnswerQuery.grid(
            fields=VPFP_MACRO_FIELDS,
            t=5,
            x=2,
            y=3,
        )

        answer = adapter.predict_macros(case, query)

        self.assertEqual(len(runner.calls), 1)
        submission, public_case, times, budget = runner.calls[0]
        self.assertIs(submission.__wrapped__, legacy_submission)
        np.testing.assert_array_equal(times, np.linspace(0.0, 1.0, 5))
        self.assertEqual(budget, _budget())
        self.assertEqual(public_case["case_id"], case.case_id)
        self.assertEqual(public_case["grid"].x_domain, (1.0, 3.0))
        self.assertEqual(public_case["time"].dt, 0.5)
        self.assertEqual(public_case["collision_nu"], 0.25)
        self.assertIsNot(public_case["question"], case.public_data["question"])
        self.assertEqual(
            public_case["question"][
                "solver_compute_budget_contract_version"
            ],
            REAL_COMPUTE_BUDGET_CONTRACT,
        )
        self.assertNotIn(
            "solver_compute_budget_contract_version",
            case.public_data["question"],
        )
        self.assertIsNot(
            public_case["arrays"]["x"], case.public_data["arrays"]["x"]
        )
        self.assertEqual(
            public_case["arrays"]["x"].dtype,
            case.public_data["arrays"]["x"].dtype,
        )
        self.assertEqual(case.public_data["question"], original_question)
        np.testing.assert_array_equal(case.public_data["arrays"]["x"], original_x)

        self.assertIs(answer.kind, QueryKind.GRID)
        self.assertEqual(answer.shape, (5, 2, 3))
        self.assertEqual(tuple(answer.axes), ("t", "x", "y"))
        self.assertEqual(answer.axes["t"], tuple(np.linspace(0.0, 1.0, 5)))
        self.assertEqual(answer.axes["x"], (0.0, 0.5))
        self.assertEqual(answer.axes["y"], (0.0, 1.0, 2.0))
        rho = np.arange(30, dtype=float).reshape(5, 2, 3)
        self.assertEqual(answer.fields["rho"], tuple(rho.reshape(-1)))
        self.assertEqual(
            answer.fields["momentum_x"], tuple((rho + 100.0).reshape(-1))
        )
        self.assertEqual(
            answer.fields["momentum_y"], tuple((rho + 200.0).reshape(-1))
        )

    def test_native_and_midpoint_time_rules_and_legacy_aliases(self) -> None:
        runner = _CapturingRunner(aliases=True)
        adapter = VPFPLegacyObservableAdapter(
            _Submission(), _budget(), runner
        )

        native = adapter.predict_observables(
            _case(),
            AnswerQuery.grid(fields=VPFP_MACRO_FIELDS, x=2, y=3),
        )
        midpoint = adapter.predict_observables(
            _case(),
            AnswerQuery.grid(fields=VPFP_MACRO_FIELDS, t=1),
        )

        self.assertEqual(native.axes["t"], (0.0, 0.5, 1.0))
        self.assertEqual(midpoint.axes["t"], (0.5,))
        self.assertEqual(midpoint.shape, (1, 2, 3))
        self.assertEqual(set(midpoint.fields), set(VPFP_MACRO_FIELDS))

    def test_rejects_queries_outside_the_historical_grid_contract(self) -> None:
        adapter = VPFPLegacyObservableAdapter(
            _Submission(), _budget(), _CapturingRunner()
        )
        point_query = AnswerQuery.at_points(
            fields=VPFP_MACRO_FIELDS,
            points=({"t": 0.0, "x": 0.0, "y": 0.0},),
        )
        cases = (
            (point_query, "GRID"),
            (AnswerQuery.grid(fields=("rho",), t=3), "all registered"),
            (
                AnswerQuery.grid(fields=VPFP_MACRO_FIELDS, x=4),
                "native x-resolution",
            ),
            (
                AnswerQuery.grid(fields=VPFP_MACRO_FIELDS, vx=3),
                "unsupported axes",
            ),
        )
        for query, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    adapter.predict_observables(_case(), query)
        with self.assertRaisesRegex(ValueError, "float budget override"):
            adapter.predict_observables(
                _case(),
                AnswerQuery.grid(fields=VPFP_MACRO_FIELDS, t=3),
                budget=30.0,
            )

    def test_validates_structured_budget_without_mutating_the_caller(self) -> None:
        budget = _budget()
        adapter = VPFPLegacyObservableAdapter(
            _Submission(), budget, _CapturingRunner()
        )
        budget["tier"] = 99
        self.assertEqual(adapter.compute_budget["tier"], 1)

        invalid = (
            (_budget(schema="wrong"), "schema"),
            (_budget(tier=True), "tier"),
            (_budget(wall_time_seconds=0.0), "wall time"),
            (_budget(gpus=-1), "gpus"),
        )
        for value, message in invalid:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    VPFPLegacyObservableAdapter(
                        _Submission(), value, _CapturingRunner()
                    )

    def test_validates_public_contract_versions_before_runner(self) -> None:
        self.assertEqual(
            LEGACY_SOLVER_OUTPUT_CONTRACT, REAL_SOLVER_OUTPUT_CONTRACT
        )
        self.assertEqual(
            LEGACY_COMPUTE_BUDGET_CONTRACT, REAL_COMPUTE_BUDGET_CONTRACT
        )
        output_key = "solver_output_contract_version"
        budget_key = "solver_compute_budget_contract_version"
        query = AnswerQuery.grid(fields=VPFP_MACRO_FIELDS, t=3)

        marked_case = _case()
        marked_case.public_data["question"][budget_key] = (
            REAL_COMPUTE_BUDGET_CONTRACT
        )
        marked_runner = _CapturingRunner()
        marked_adapter = VPFPLegacyObservableAdapter(
            _Submission(), _budget(), marked_runner
        )
        marked_adapter.predict_observables(marked_case, query)
        self.assertEqual(len(marked_runner.calls), 1)

        mutations = (
            (output_key, None, True, "solver output contract"),
            (output_key, "full_state", False, "solver output contract"),
            (budget_key, None, False, "compute-budget contract"),
            (budget_key, "wrong-contract", False, "compute-budget contract"),
        )
        for key, value, remove, message in mutations:
            with self.subTest(key=key, value=value):
                case = _case()
                if remove:
                    del case.public_data["question"][key]
                else:
                    case.public_data["question"][key] = value
                runner = _CapturingRunner()
                adapter = VPFPLegacyObservableAdapter(
                    _Submission(), _budget(), runner
                )
                with self.assertRaisesRegex(ValueError, message):
                    adapter.predict_observables(
                        case,
                        query,
                    )
                self.assertEqual(runner.calls, [])

    def test_rejects_invalid_submission_runner_and_public_case(self) -> None:
        with self.assertRaisesRegex(TypeError, "solve_with_budget"):
            VPFPLegacyObservableAdapter(object(), _budget(), _CapturingRunner())
        with self.assertRaisesRegex(TypeError, "runner"):
            VPFPLegacyObservableAdapter(_Submission(), _budget(), object())

        case = _case()
        case.public_data["question"]["case_id"] = "another-case"
        adapter = VPFPLegacyObservableAdapter(
            _Submission(), _budget(), _CapturingRunner()
        )
        with self.assertRaisesRegex(ValueError, "case id"):
            adapter.predict_observables(
                case,
                AnswerQuery.grid(fields=VPFP_MACRO_FIELDS, t=3),
            )

    def test_rejects_malformed_runner_results_and_predictions(self) -> None:
        import numpy as np

        query = AnswerQuery.grid(fields=VPFP_MACRO_FIELDS, t=3)

        def malformed(*_args):
            return {}

        with self.assertRaisesRegex(TypeError, "prediction, wall_seconds"):
            VPFPLegacyObservableAdapter(
                _Submission(), _budget(), malformed
            ).predict_observables(_case(), query)

        def negative_wall(*_args):
            shape = (3, 2, 3)
            return {
                "rho": np.ones(shape),
                "momentum": np.ones((*shape, 2)),
                "kinetic_energy": np.ones(shape),
            }, -1.0

        with self.assertRaisesRegex(ValueError, "wall time"):
            VPFPLegacyObservableAdapter(
                _Submission(), _budget(), negative_wall
            ).predict_observables(_case(), query)

        def wrong_shape(*_args):
            return {
                "rho": np.ones((2, 2, 3)),
                "momentum": np.ones((3, 2, 3, 2)),
                "kinetic_energy": np.ones((3, 2, 3)),
            }, 0.0

        with self.assertRaisesRegex(ValueError, "rho has shape"):
            VPFPLegacyObservableAdapter(
                _Submission(), _budget(), wrong_shape
            ).predict_observables(_case(), query)

        def nonfinite(*_args):
            shape = (3, 2, 3)
            rho = np.ones(shape)
            rho[0, 0, 0] = np.nan
            return {
                "rho": rho,
                "momentum": np.ones((*shape, 2)),
                "kinetic_energy": np.ones(shape),
            }, 0.0

        with self.assertRaisesRegex(ValueError, "non-finite"):
            VPFPLegacyObservableAdapter(
                _Submission(), _budget(), nonfinite
            ).predict_observables(_case(), query)

    def test_real_runner_rejects_complex_before_legacy_float_coercion(self) -> None:
        import numpy as np

        class _RealSubmission:
            def __init__(self, *, complex_rho: bool) -> None:
                self.complex_rho = complex_rho
                self.calls = 0

            def solve_with_budget(self, public_case, query_times, budget):
                del budget
                self.calls += 1
                shape = (
                    len(query_times),
                    int(public_case["grid"].nx),
                    int(public_case["grid"].ny),
                )
                dtype = (
                    np.complex128
                    if self.complex_rho
                    else np.float64
                )
                rho = np.ones(shape, dtype=dtype)
                if self.complex_rho:
                    rho[0, 0, 0] = 1.0 + 2.0j
                return {
                    "rho": rho,
                    "momentum": np.ones((*shape, 2)),
                    "kinetic_energy": np.ones(shape),
                }

        query = AnswerQuery.grid(fields=VPFP_MACRO_FIELDS, t=3)
        valid_submission = _RealSubmission(complex_rho=False)
        answer = VPFPLegacyObservableAdapter(
            valid_submission,
            _budget(),
            run_budgeted_submission,
        ).predict_observables(_case(), query)
        self.assertEqual(answer.shape, (3, 2, 3))
        self.assertEqual(valid_submission.calls, 1)

        complex_submission = _RealSubmission(complex_rho=True)
        with self.assertRaisesRegex(ValueError, "rho contains complex values"):
            VPFPLegacyObservableAdapter(
                complex_submission,
                _budget(),
                run_budgeted_submission,
            ).predict_observables(_case(), query)
        self.assertEqual(complex_submission.calls, 1)

    def test_rejects_complex_prediction_before_float_coercion(self) -> None:
        import numpy as np

        query = AnswerQuery.grid(fields=VPFP_MACRO_FIELDS, t=3)

        def complex_prediction(*_args):
            shape = (3, 2, 3)
            rho = np.ones(shape, dtype=np.complex128)
            rho[0, 0, 0] = 1.0 + 2.0j
            return {
                "rho": rho,
                "momentum": np.ones((*shape, 2)),
                "kinetic_energy": np.ones(shape),
            }, 0.0

        with self.assertRaisesRegex(ValueError, "rho contains complex values"):
            VPFPLegacyObservableAdapter(
                _Submission(), _budget(), complex_prediction
            ).predict_observables(_case(), query)


if __name__ == "__main__":
    unittest.main()
