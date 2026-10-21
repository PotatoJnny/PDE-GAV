import math
import unittest

from pde_agent_verifier_solver.plugins.gkw import (
    GKWMacroMatcher,
    GKW_MACRO_FIELDS,
)
from pde_agent_verifier_solver.plugins.gkw.contract import (
    GKW_LEGACY_MACRO_OUTPUT_CONTRACT,
    gkw_answer_view_from_pairs,
)
from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    AnswerView,
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
        "gkw-case",
        CaseSource.GENERATED,
        CaseVisibility.DEVELOPMENT,
        {
            "arrays": _arrays(),
            "metadata": {"time_nodes": [0.0, 0.5, 1.0]},
            "oracle_protocol": {
                "time_interval_closed": [0.0, 1.0],
                "field_grid_shape": [2, 2, 2],
            },
            "macro_query": {
                "times": np.asarray([0.5, 1.0]),
                "field_indices": np.asarray([[0, 0, 0], [1, 1, 1]]),
                "output_contract": GKW_LEGACY_MACRO_OUTPUT_CONTRACT,
            },
            "observations": {"particle_density": [999.0]},
        },
    )


def _values():
    import numpy as np

    value = np.zeros((2, 2, 2), dtype=np.float64)
    value[..., 0] = 1.0
    return {name: value.copy() for name in GKW_MACRO_FIELDS}


def _query() -> AnswerQuery:
    return AnswerQuery.grid(fields=GKW_MACRO_FIELDS, t=2, field_row=2)


def _view(values=None) -> AnswerView:
    import numpy as np

    case = _case()
    times = np.asarray(case.public_data["macro_query"]["times"])
    indices = np.asarray(case.public_data["macro_query"]["field_indices"])
    return gkw_answer_view_from_pairs(
        _query(), values or _values(), times=times, field_indices=indices
    )


class GKWMacroMatcherTests(unittest.TestCase):
    def test_exact_prediction_and_stable_weighted_metric(self) -> None:
        matcher = GKWMacroMatcher()
        truth = _view()
        target = matcher.prepare_target(_case(), truth, _query())

        exact = matcher.match(_case(), truth, target)
        self.assertEqual(exact.primary_error, 0.0)
        self.assertIsNone(exact.passed)
        self.assertEqual(len(target.sample_weights), 8)

        changed = _values()
        changed["particle_density"][0, 0, 0] = 2.0
        result = matcher.match(_case(), _view(changed), target)
        expected_density = math.sqrt(2.0 / 12.0)
        self.assertAlmostEqual(
            result.metrics["particle_density_relative_l2"],
            expected_density,
        )
        self.assertAlmostEqual(result.primary_error, expected_density / 4.0)
        self.assertEqual(
            result.metrics["legacy_fusion_gkw_aggregate_relative_l2"],
            result.primary_error,
        )
        self.assertNotIn("observations", result.metrics)

    def test_explicit_gate_and_coordinate_binding(self) -> None:
        matcher = GKWMacroMatcher(maximum_relative_l2=0.01)
        target = matcher.prepare_target(_case(), _view(), _query())
        changed = _values()
        changed["phi"][:] = 2.0
        failed = matcher.match(_case(), _view(changed), target)
        self.assertIs(failed.passed, False)
        self.assertTrue(failed.failures)

        wrong_axes = AnswerView(
            QueryKind.GRID,
            _view().fields,
            axes={
                "t": (0.5, 1.1),
                "field_row": (0.0, 1.0),
                "complex_component": (0.0, 1.0),
            },
            shape=(2, 2, 2),
        )
        with self.assertRaisesRegex(ValueError, "axes"):
            matcher.match(_case(), wrong_axes, target)

    def test_point_query_uses_public_field_quadrature(self) -> None:
        import numpy as np

        query = AnswerQuery.at_points(
            fields=GKW_MACRO_FIELDS,
            points=(
                {"t": 0.5, "s_index": 0, "kx_index": 0, "ky_index": 0},
                {"t": 1.0, "s_index": 1, "kx_index": 1, "ky_index": 1},
            ),
        )
        values = {
            name: np.asarray([[1.0, 0.0], [1.0, 0.0]])
            for name in GKW_MACRO_FIELDS
        }
        view = gkw_answer_view_from_pairs(
            query,
            values,
            times=np.asarray([0.5, 1.0]),
            field_indices=np.asarray([[0, 0, 0], [1, 1, 1]]),
        )
        target = GKWMacroMatcher().prepare_target(_case(), view, query)
        self.assertEqual(target.sample_weights, (2.0, 2.0, 4.0, 4.0))
        self.assertEqual(
            GKWMacroMatcher().match(_case(), view, target).primary_error,
            0.0,
        )

    def test_rejects_non_native_grid_resolution_and_missing_fields(self) -> None:
        matcher = GKWMacroMatcher()
        with self.assertRaisesRegex(ValueError, "all registered"):
            matcher.prepare_target(
                _case(),
                _view(),
                AnswerQuery.grid(fields=("phi",), t=2),
            )
        query = AnswerQuery.grid(fields=GKW_MACRO_FIELDS, t=3, field_row=2)
        with self.assertRaisesRegex(ValueError, "native t panel"):
            matcher.prepare_target(_case(), _view(), query)


if __name__ == "__main__":
    unittest.main()
