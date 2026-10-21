from __future__ import annotations

from pathlib import Path
from types import MappingProxyType, SimpleNamespace
import unittest

import numpy as np

from pde_agent_verifier_solver.plugins.gkw import (
    GKW_PUBLIC_SESSION_METHODS,
    GKW_LEGACY_MACRO_OUTPUT_CONTRACT,
    GKW_MACRO_FIELDS,
    gkw_native_macro_query,
)
from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    Case,
    CaseSource,
    CaseVisibility,
)
from pde_agent_verifier_solver.sdk.public_sessions import (
    bind_public_session_factory,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = (
    REPOSITORY_ROOT
    / "configs"
    / "candidates"
    / "gkw_hold_initial_macro_candidate.py"
)


class GKWHoldInitialCandidateTests(unittest.TestCase):
    def test_two_phase_zero_baseline_is_query_blind_and_accepts_off_panel_points(self) -> None:
        namespace = _load_candidate(CANDIDATE)
        prepare = namespace["prepare_solution"]
        predict = namespace["predict_macros"]
        case = _case()

        with bind_public_session_factory(
            _forbidden_session_factory,
            case,
            allowed_methods=GKW_PUBLIC_SESSION_METHODS,
        ):
            handle = prepare(case, solve_budget=1.0)
        point_query = AnswerQuery.at_points(
            fields=GKW_MACRO_FIELDS,
            points=(
                {
                    "t": 0.73,
                    "s_index": 1,
                    "kx_index": 0,
                    "ky_index": 1,
                },
                {
                    "t": 0.5,
                    "s_index": 0,
                    "kx_index": 0,
                    "ky_index": 0,
                },
            ),
        )
        point_prediction = handle.evaluate(point_query)

        grid_query = gkw_native_macro_query(case, "development")
        with bind_public_session_factory(
            _forbidden_session_factory,
            case,
            allowed_methods=GKW_PUBLIC_SESSION_METHODS,
        ):
            grid_prediction = predict(case, grid_query, budget=1.0)

        self.assertEqual(point_prediction.points, point_query.points)
        self.assertEqual(point_prediction.shape, (2, 2))
        for field in GKW_MACRO_FIELDS:
            actual = np.asarray(point_prediction.fields[field]).reshape(2, 2)
            np.testing.assert_array_equal(actual, np.zeros((2, 2)))
            self.assertFalse(any(grid_prediction.fields[field]))

    def test_prepare_rejects_invalid_budget_before_oracle_access(self) -> None:
        prepare = _load_candidate(CANDIDATE)["prepare_solution"]
        with self.assertRaisesRegex(ValueError, "budget"):
            prepare(_case(), solve_budget=0.0)


def _load_candidate(path: Path) -> dict[str, object]:
    namespace: dict[str, object] = {
        "__file__": str(path),
        "__name__": "_gkw_hold_initial_candidate_test",
        "__package__": None,
    }
    exec(compile(path.read_bytes(), str(path), "exec"), namespace)
    return namespace


def _readonly(values, *, dtype=np.float64):
    array = np.asarray(values, dtype=dtype)
    array.setflags(write=False)
    return array


def _case() -> Case:
    arrays = MappingProxyType(
        {
            "s": _readonly([0.0, 1.0]),
            "kx": _readonly([-1.0, 1.0]),
            "ky": _readonly([0.0, 2.0]),
            "B": _readonly([2.0, 4.0]),
            "ints": _readonly([1.0, 0.5]),
            "parseval_ky": _readonly([1.0, 2.0]),
            "vpar": _readonly([-1.0, 1.0]),
            "mu": _readonly([0.5, 1.5]),
            "intvp": _readonly([0.25, 0.75]),
            "intmu": _readonly([0.4, 0.6]),
        }
    )
    return Case(
        case_id="gkw-hold-initial-points-test",
        source=CaseSource.GENERATED,
        visibility=CaseVisibility.DEVELOPMENT,
        public_data=MappingProxyType(
            {
                "arrays": arrays,
                "macro_query": MappingProxyType(
                    {
                        "times": _readonly([0.5, 1.0]),
                        "field_indices": _readonly(
                            [[0, 0, 0], [1, 1, 1]],
                            dtype=np.int64,
                        ),
                        "output_contract": GKW_LEGACY_MACRO_OUTPUT_CONTRACT,
                    }
                ),
                "oracle_protocol": MappingProxyType(
                    {
                        "time_interval_closed": (0.0, 1.0),
                        "field_grid_shape": (2, 2, 2),
                        "limits": MappingProxyType(
                            {"max_rows_per_request": 3}
                        )
                    }
                ),
            }
        ),
        round_index=0,
        tags=("gkw", "cold-start"),
    )


def _forbidden_session_factory(_case: Case):
    raise AssertionError("strict two-phase zero baseline must not open an oracle")


def _session_factory(case: Case):
    def query_initial_h(indices):
        rows = np.asarray(indices, dtype=np.int64)
        values = np.zeros((len(rows), 2), dtype=np.float64)
        values[:, 0] = 1.0 + 0.1 * rows[:, 0] + 0.01 * rows[:, 1]
        values[:, 1] = -0.2 * values[:, 0]
        return values

    def query_initial_phi(indices):
        rows = np.asarray(indices, dtype=np.int64)
        values = np.zeros((len(rows), 2), dtype=np.float64)
        values[:, 0] = 2.0 + rows[:, 0]
        values[:, 1] = -0.5
        return values

    return SimpleNamespace(
        case_id=case.case_id,
        query_initial_h=query_initial_h,
        query_initial_phi=query_initial_phi,
    )


if __name__ == "__main__":
    unittest.main()
