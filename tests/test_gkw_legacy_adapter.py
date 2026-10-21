import unittest

from pde_agent_verifier_solver.plugins.gkw import (
    GKWLegacyObservableAdapter,
    GKWLegacyRuntimeRunner,
    GKW_LEGACY_COMPUTE_BUDGET_SCHEMA,
    GKW_LEGACY_MACRO_OUTPUT_CONTRACT,
    GKW_MACRO_FIELDS,
)
from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    Case,
    CaseSource,
    CaseVisibility,
    QueryKind,
)


def _arrays():
    import numpy as np

    return {
        "s": np.asarray([0.0, 1.0]),
        "kx": np.asarray([-1.0, 1.0]),
        "ky": np.asarray([0.0, 2.0]),
        "B": np.asarray([2.0, 4.0]),
        "ints": np.asarray([1.0, 0.5]),
        "parseval_ky": np.asarray([1.0, 2.0]),
    }


def _case() -> Case:
    import numpy as np

    return Case(
        "gkw-adapter-case",
        CaseSource.GENERATED,
        CaseVisibility.DEVELOPMENT,
        {
            "arrays": _arrays(),
            "scalars": {"beta": 1.0},
            "metadata": {
                "time_nodes": [0.0, 0.5, 1.0],
                "solver_output_contract_version": GKW_LEGACY_MACRO_OUTPUT_CONTRACT,
            },
            "oracle_protocol": {
                "time_interval_closed": [0.0, 1.0],
                "field_grid_shape": [2, 2, 2],
            },
            "macro_query": {
                "times": np.asarray([0.5, 1.0]),
                "field_indices": np.asarray([[0, 0, 0], [1, 1, 1]]),
                "output_contract": GKW_LEGACY_MACRO_OUTPUT_CONTRACT,
            },
        },
    )


def _budget(**overrides):
    value = {
        "schema": GKW_LEGACY_COMPUTE_BUDGET_SCHEMA,
        "tier": 1,
        "wall_time_seconds": 30.0,
        "cpus": 2,
        "gpus": 0,
        "memory_bytes": 1024,
    }
    value.update(overrides)
    return value


class _Session:
    def __init__(self, case_id="gkw-adapter-case"):
        self.case_id = case_id
        self.arrays = _arrays()
        self.scalars = {"beta": 1.0}
        self.metadata = {
            "time_nodes": [0.0, 0.5, 1.0],
            "solver_output_contract_version": GKW_LEGACY_MACRO_OUTPUT_CONTRACT
        }
        self.oracle_protocol = {
            "time_interval_closed": [0.0, 1.0],
            "field_grid_shape": [2, 2, 2],
        }

    def query_source(self, times, indices):
        raise NotImplementedError

    def query_initial_h(self, indices):
        raise NotImplementedError

    def query_initial_phi(self, indices):
        raise NotImplementedError



class _Submission:
    def solve_macros_with_budget(self, public_case, times, indices, budget):
        import numpy as np

        base = np.arange(len(times) * len(indices), dtype=float).reshape(
            len(times), len(indices), 1
        )
        pairs = np.concatenate((base, -base), axis=-1)
        return {name: pairs + offset for offset, name in enumerate(GKW_MACRO_FIELDS)}


class _Runner:
    def __init__(self):
        self.calls = []

    def __call__(self, solver, session, times, indices, budget):
        self.calls.append((solver, session, times.copy(), indices.copy(), budget))
        return solver(session, times, indices, budget), 0.125


class GKWLegacyObservableAdapterTests(unittest.TestCase):
    def test_adapts_native_grid_and_preserves_structured_budget(self) -> None:
        runner = _Runner()
        submission = _Submission()
        adapter = GKWLegacyObservableAdapter(
            submission,
            _budget(),
            runner,
            lambda case: _Session(case.case_id),
        )
        query = AnswerQuery.grid(fields=GKW_MACRO_FIELDS, t=2, field_row=2)
        answer = adapter.predict_observables(_case(), query)

        self.assertEqual(len(runner.calls), 1)
        solver, _session, times, indices, budget = runner.calls[0]
        self.assertIs(solver.__wrapped__.__self__, submission)
        self.assertEqual(budget, _budget())
        self.assertEqual(tuple(times), (0.5, 1.0))
        self.assertEqual(indices.tolist(), [[0, 0, 0], [1, 1, 1]])
        self.assertIs(answer.kind, QueryKind.GRID)
        self.assertEqual(answer.shape, (2, 2, 2))
        self.assertEqual(tuple(answer.axes), ("t", "field_row", "complex_component"))
        self.assertEqual(answer.axes["complex_component"], (0.0, 1.0))

        macro_answer = adapter.predict_macros(_case(), query)
        self.assertEqual(macro_answer, answer)
        self.assertEqual(len(runner.calls), 2)

    def test_point_adapter_evaluates_one_cross_product_and_gathers_pairs(self) -> None:
        runner = _Runner()
        adapter = GKWLegacyObservableAdapter(
            _Submission(), _budget(), runner, lambda case: _Session(case.case_id)
        )
        query = AnswerQuery.at_points(
            fields=GKW_MACRO_FIELDS,
            points=(
                {"t": 1.0, "s_index": 1, "kx_index": 1, "ky_index": 1},
                {"t": 0.5, "s_index": 0, "kx_index": 0, "ky_index": 0},
                {"t": 1.0, "s_index": 1, "kx_index": 1, "ky_index": 1},
            ),
        )
        answer = adapter.predict_observables(_case(), query)
        self.assertEqual(len(runner.calls), 1)
        self.assertEqual(len(runner.calls[0][2]), 2)
        self.assertEqual(len(runner.calls[0][3]), 2)
        self.assertEqual(answer.shape, (3, 2))
        self.assertEqual(answer.points, query.points)
        self.assertEqual(answer.fields["particle_density"][0:2], (0.0, 0.0))
        self.assertEqual(answer.fields["particle_density"][2:4], (3.0, -3.0))
        self.assertEqual(answer.fields["particle_density"][4:6], (0.0, 0.0))

    def test_runtime_runner_converts_the_stable_budget_object(self) -> None:
        converted = []

        def budget_from_dict(value):
            converted.append(value)
            return ("structured", value["tier"])

        def run_submission(solver, session, times, indices, budget):
            del solver, session, times, indices
            return {"budget": budget}, 0.25

        runner = GKWLegacyRuntimeRunner(
            run_submission, budget_from_dict
        )
        output, wall = runner(object(), object(), (), (), _budget())
        self.assertEqual(output, {"budget": ("structured", 1)})
        self.assertEqual(wall, 0.25)
        self.assertEqual(converted, [_budget()])

    def test_fails_closed_on_budget_session_query_and_raw_dtype(self) -> None:
        with self.assertRaisesRegex(ValueError, "schema"):
            GKWLegacyObservableAdapter(
                _Submission(),
                _budget(schema="wrong"),
                _Runner(),
                lambda case: _Session(),
            )
        adapter = GKWLegacyObservableAdapter(
            _Submission(), _budget(), _Runner(), lambda case: _Session("wrong")
        )
        query = AnswerQuery.grid(fields=GKW_MACRO_FIELDS, t=2, field_row=2)
        with self.assertRaisesRegex(ValueError, "case id"):
            adapter.predict_observables(_case(), query)

        adapter = GKWLegacyObservableAdapter(
            _Submission(), _budget(), _Runner(), lambda case: _Session(case.case_id)
        )
        with self.assertRaisesRegex(ValueError, "float budget override"):
            adapter.predict_observables(_case(), query, budget=30.0)
        with self.assertRaisesRegex(ValueError, "native t panel"):
            adapter.predict_observables(
                _case(),
                AnswerQuery.grid(fields=GKW_MACRO_FIELDS, t=3, field_row=2),
            )

        import numpy as np

        def unsafe(_session, times, indices, budget):
            del budget
            value = np.empty((len(times), len(indices), 2), dtype=object)
            value[:] = 1.0
            return {name: value.copy() for name in GKW_MACRO_FIELDS}

        unsafe_adapter = GKWLegacyObservableAdapter(
            unsafe, _budget(), _Runner(), lambda case: _Session(case.case_id)
        )
        with self.assertRaisesRegex(TypeError, "real numeric dtype"):
            unsafe_adapter.predict_observables(_case(), query)


if __name__ == "__main__":
    unittest.main()
