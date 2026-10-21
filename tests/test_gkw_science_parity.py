import math
import unittest

from pde_agent_verifier_solver.plugins.gkw import GKWMacroMatcher, GKW_MACRO_FIELDS
from pde_agent_verifier_solver.plugins.gkw.contract import (
    GKW_LEGACY_MACRO_OUTPUT_CONTRACT,
    gkw_answer_view_from_pairs,
    gkw_point_query_coordinates,
    validate_gkw_macro_query,
)
from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    Case,
    CaseSource,
    CaseVisibility,
    QueryKind,
)


def _case():
    import numpy as np

    return Case(
        "parity-case",
        CaseSource.GENERATED,
        CaseVisibility.DEVELOPMENT,
        {
            "arrays": {
                "s": np.asarray([0.0, 1.0]),
                "kx": np.asarray([-1.0, 1.0]),
                "ky": np.asarray([0.0, 2.0]),
                "B": np.asarray([2.0, 4.0]),
                "ints": np.asarray([1.0, 0.5]),
                "parseval_ky": np.asarray([1.0, 2.0]),
            },
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
        },
    )


class GKWScienceParityTests(unittest.TestCase):
    def test_random_complex_arrays_match_stable_metric_formula(self) -> None:
        import numpy as np

        rng = np.random.default_rng(20260823)
        case = _case()
        query = AnswerQuery.grid(fields=GKW_MACRO_FIELDS, t=2, field_row=2)
        truth = {
            name: rng.normal(size=(2, 2, 2)) for name in GKW_MACRO_FIELDS
        }
        prediction = {
            name: truth[name] + 0.1 * rng.normal(size=(2, 2, 2))
            for name in GKW_MACRO_FIELDS
        }
        times = case.public_data["macro_query"]["times"]
        indices = case.public_data["macro_query"]["field_indices"]
        truth_view = gkw_answer_view_from_pairs(
            query, truth, times=times, field_indices=indices
        )
        prediction_view = gkw_answer_view_from_pairs(
            query, prediction, times=times, field_indices=indices
        )
        matcher = GKWMacroMatcher()
        result = matcher.match(
            case,
            prediction_view,
            matcher.prepare_target(case, truth_view, query),
        )

        weights = np.asarray([2.0, 4.0])
        stable_fields = []
        for name in GKW_MACRO_FIELDS:
            error_square = np.sum((prediction[name] - truth[name]) ** 2, axis=-1)
            truth_square = np.sum(truth[name] ** 2, axis=-1)
            stable_fields.append(
                math.sqrt(
                    float(np.sum(weights[None, :] * error_square))
                    / max(float(np.sum(weights[None, :] * truth_square)), 1.0e-30)
                )
            )
        self.assertAlmostEqual(result.primary_error, sum(stable_fields) / 4.0)

    def test_grid_and_released_point_domains_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "axes must be exactly"):
            validate_gkw_macro_query(
                _case(),
                AnswerQuery(
                    QueryKind.GRID,
                    GKW_MACRO_FIELDS,
                    resolution={"t": 2},
                ),
            )
        for point, message in (
            (
                {"t": 0.5, "s_index": 0.0, "kx_index": 0, "ky_index": 0},
                "leaves the public grid",
            ),
            (
                {
                    "t": 0.5,
                    "s_index": 0,
                    "kx_index": 0,
                    "ky_index": 0,
                    "extra": 1,
                },
                "require",
            ),
            (
                {"t": 0.5, "s_index": 0, "kx_index": 1, "ky_index": 0},
                "not on the released panel",
            ),
        ):
            with self.assertRaisesRegex(ValueError, message):
                gkw_point_query_coordinates(
                    _case(),
                    AnswerQuery.at_points(fields=GKW_MACRO_FIELDS, points=(point,)),
                )


if __name__ == "__main__":
    unittest.main()
