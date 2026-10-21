from __future__ import annotations

from tempfile import TemporaryDirectory
from pathlib import Path
import unittest

import numpy as np

from pde_agent_verifier_solver.engine.macro_evaluator import (
    ControllerMacroEvaluator,
    PreparedMacroCohort,
)
from pde_agent_verifier_solver.engine.macro_query_plan import (
    ControllerMacroQueryPlan,
    controller_query_plan_seed,
)
from pde_agent_verifier_solver.plugins._regular_grid import (
    answer_from_regular_grid,
)
from pde_agent_verifier_solver.plugins.gkw.contract import (
    GKW_LEGACY_MACRO_OUTPUT_CONTRACT,
    GKW_MACRO_FIELDS,
    gkw_point_query_coordinates,
)
from pde_agent_verifier_solver.plugins.gkw.generator import (
    GKWAnswerFirstGenerator,
)
from pde_agent_verifier_solver.plugins.gkw.macro_loop import (
    GKW_MACRO_POINT_COUNTS,
    bind_gkw_macro_loop,
    gkw_macro_point_query,
    gkw_native_macro_query,
)
from pde_agent_verifier_solver.plugins.gkw.problem import (
    GKWReleasedMacroTruth,
)
from pde_agent_verifier_solver.plugins.gkw.release_io import (
    freeze_gkw_public_data,
    load_gkw_release,
)
from pde_agent_verifier_solver.plugins.vpfp.generator import (
    VPFPContinuousGenerator,
)
from pde_agent_verifier_solver.plugins.vpfp.macro_loop import (
    VPFP_MACRO_POINT_COUNTS,
    bind_vpfp_macro_loop,
    vpfp_macro_point_query,
    vpfp_native_macro_query,
)
from pde_agent_verifier_solver.plugins.vpfp.macro_matching import (
    VPFP_MACRO_FIELDS,
)
from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    AnswerView,
    Case,
    CaseRecord,
    CaseSource,
    CaseVisibility,
    QueryKind,
)
from tests.gkw_release_fixture import (
    canonical_gkw_public_components,
    write_gkw_release,
)


def _gkw_case(case_id: str = "gkw-query-plan") -> Case:
    arrays, scalars, metadata = canonical_gkw_public_components(case_id)
    indices = np.asarray(
        [
            (s_index, kx_index, ky_index)
            for s_index in range(4)
            for kx_index in range(5)
            for ky_index in range(2)
        ],
        dtype=np.int64,
    )
    return Case(
        case_id=case_id,
        source=CaseSource.GENERATED,
        visibility=CaseVisibility.HIDDEN,
        public_data={
            "arrays": arrays,
            "scalars": scalars,
            "metadata": metadata,
            "oracle_protocol": {
                "time_interval_closed": [0.0, 1.0],
                "field_grid_shape": [
                    len(arrays["s"]),
                    len(arrays["kx"]),
                    len(arrays["ky"]),
                ],
            },
            "macro_query": {
                "times": np.linspace(0.5, 1.0, 4, dtype=np.float64),
                "field_indices": indices,
                "output_contract": GKW_LEGACY_MACRO_OUTPUT_CONTRACT,
            },
        },
        tags=("gkw",),
    )


def _vpfp_case(
    case_id: str = "vpfp-query-plan",
    *,
    parent_case_id: str | None = None,
) -> Case:
    return Case(
        case_id=case_id,
        source=(
            CaseSource.PERTURBED
            if parent_case_id is not None
            else CaseSource.GENERATED
        ),
        visibility=CaseVisibility.HIDDEN,
        public_data={
            "arrays": {
                "t": np.linspace(0.0, 2.0, 5, dtype=np.float64),
                "x": np.linspace(-1.0, 1.0, 6, dtype=np.float64),
                "y": np.linspace(3.0, 5.0, 7, dtype=np.float64),
            },
        },
        parent_case_id=parent_case_id,
        tags=("vpfp",),
    )


def _point_answer(query: AnswerQuery) -> AnswerView:
    values = tuple(1.0 for _point in query.points)
    return AnswerView(
        kind=QueryKind.POINTS,
        fields={name: values for name in VPFP_MACRO_FIELDS},
        points=query.points,
        shape=(len(query.points),),
    )


class _PointTruth:
    def answer(self, query: AnswerQuery) -> AnswerView:
        return _point_answer(query)


class _PointHandle:
    def __init__(self, owner) -> None:
        self.owner = owner

    def evaluate(self, query: AnswerQuery) -> AnswerView:
        self.owner.revealed_queries.append(query)
        return _point_answer(query)


class _TwoPhaseCandidate:
    def __init__(self) -> None:
        self.prepared_cases = []
        self.revealed_queries = []

    def prepare_solution(self, case: Case, solve_budget=None):
        self.prepared_cases.append((case, solve_budget))
        if "scoring_query" in case.public_data:
            raise AssertionError("prepare case leaked the scoring query")
        return _PointHandle(self)


class ControllerPointQueryPlanTests(unittest.TestCase):
    def test_profile_counts_and_coordinates_are_bounded(self):
        gkw_case = _gkw_case()
        for profile, expected in GKW_MACRO_POINT_COUNTS.items():
            query = gkw_macro_point_query(gkw_case, profile, 700 + expected)
            self.assertIs(query.kind, QueryKind.POINTS)
            self.assertEqual(len(query.points), expected)
            times, indices = gkw_point_query_coordinates(
                gkw_case, query, require_native_panel=False
            )
            self.assertEqual(times.shape, (expected,))
            self.assertEqual(indices.shape, (expected, 3))
            self.assertTrue(np.all(times >= 0.0))
            self.assertTrue(np.all(times <= 1.0))
            panel_times = np.asarray(gkw_case.public_data["macro_query"]["times"])
            panel_indices = np.asarray(
                gkw_case.public_data["macro_query"]["field_indices"]
            )
            self.assertTrue(
                any(
                    not np.any(np.isclose(panel_times, point["t"]))
                    or not np.any(
                        np.all(panel_indices == np.asarray(index)[None, :], axis=1)
                    )
                    for point, index in zip(query.points, indices)
                )
            )
        self.assertIs(
            gkw_native_macro_query(gkw_case, "development").kind,
            QueryKind.GRID,
        )

        vpfp_case = _vpfp_case()
        axes = vpfp_case.public_data["arrays"]
        for profile, expected in VPFP_MACRO_POINT_COUNTS.items():
            query = vpfp_macro_point_query(vpfp_case, profile, 900 + expected)
            self.assertIs(query.kind, QueryKind.POINTS)
            self.assertEqual(len(query.points), expected)
            for point in query.points:
                self.assertEqual(set(point), {"t", "x", "y"})
                for name in ("t", "x", "y"):
                    self.assertGreaterEqual(point[name], float(axes[name][0]))
                    self.assertLessEqual(point[name], float(axes[name][-1]))

            shape = tuple(len(axes[name]) for name in ("t", "x", "y"))
            values = np.ones(shape, dtype=np.float64)
            truth = answer_from_regular_grid(
                query,
                axis_order=("t", "x", "y"),
                native_axes=axes,
                fields={name: values for name in VPFP_MACRO_FIELDS},
            )
            self.assertEqual(truth.shape, (expected,))
        self.assertIs(
            vpfp_native_macro_query(vpfp_case, "development").kind,
            QueryKind.GRID,
        )

    def test_clean_and_perturbed_cases_share_one_root_draw(self):
        plugin = bind_vpfp_macro_loop(
            generator=VPFPContinuousGenerator(),
            base_state=None,
            generator_sha256="a" * 64,
        )
        clean = _vpfp_case("clean-root")
        child = _vpfp_case("perturbed-child", parent_case_id=clean.case_id)
        unrelated = _vpfp_case("unrelated-root")
        plan = ControllerMacroQueryPlan(
            cohort_id="paired-selection",
            profile="selection",
            controller_seed=12345,
        )

        clean_query = plan.build(plugin.objective, clean)
        child_query = plan.build(plugin.objective, child)
        unrelated_query = plan.build(plugin.objective, unrelated)
        self.assertEqual(plan.seed_for(clean), plan.seed_for(child))
        self.assertEqual(clean_query, child_query)
        self.assertNotEqual(clean_query, unrelated_query)

    def test_public_campaign_state_does_not_determine_official_query_seed(self):
        public_arguments = {
            "seed_base": 1234,
            "cohort_id": "selection-round-1",
            "profile": "selection",
        }

        first = controller_query_plan_seed(
            **public_arguments,
            controller_entropy=b"a" * 32,
        )
        second = controller_query_plan_seed(
            **public_arguments,
            controller_entropy=b"b" * 32,
        )

        self.assertNotEqual(first, second)
        self.assertEqual(
            first,
            controller_query_plan_seed(
                **public_arguments,
                controller_entropy=b"a" * 32,
            ),
        )

    def test_gkw_released_truth_accepts_a_scoring_subset(self):
        with TemporaryDirectory() as temporary:
            case_dir = Path(temporary) / "case"
            write_gkw_release(case_dir, "released-subset", 2.5)
            package = load_gkw_release(
                case_dir,
                expected_case_id="released-subset",
            )
            case = Case(
                case_id=package.case_id,
                source=CaseSource.INITIAL,
                visibility=CaseVisibility.HIDDEN,
                public_data=freeze_gkw_public_data(package),
                tags=("gkw",),
            )
            panel = case.public_data["macro_query"]
            panel_times = np.asarray(panel["times"], dtype=np.float64)
            panel_indices = np.asarray(panel["field_indices"], dtype=np.int64)
            subset = AnswerQuery.at_points(
                fields=GKW_MACRO_FIELDS,
                points=tuple(
                    {
                        "t": float(panel_times[position]),
                        "s_index": int(panel_indices[position, 0]),
                        "kx_index": int(panel_indices[position, 1]),
                        "ky_index": int(panel_indices[position, 2]),
                    }
                    for position in range(2)
                ),
            )
            answer = GKWReleasedMacroTruth(
                case_dir,
                package.case_id,
            ).answer(subset)

            self.assertEqual(answer.points, subset.points)
            self.assertEqual(answer.shape, (2, 2))
            for values in answer.fields.values():
                self.assertEqual(values, (2.5, 0.0, 2.5, 0.0))

    def test_defaults_materialize_one_query_then_use_formal_two_phase(self):
        vpfp_plugin = bind_vpfp_macro_loop(
            generator=VPFPContinuousGenerator(),
            base_state=None,
            generator_sha256="b" * 64,
        )
        case = _vpfp_case()
        plan = ControllerMacroQueryPlan(
            cohort_id="shared-official",
            profile="selection",
            controller_seed=456,
        )
        cohort = PreparedMacroCohort.prepare(
            cohort_id="shared-official",
            records=(CaseRecord(case, _PointTruth()),),
            matcher=vpfp_plugin.objective.matcher,
            query=lambda public_case: plan.build(
                vpfp_plugin.objective,
                public_case,
            ),
        )

        first = _TwoPhaseCandidate()
        second = _TwoPhaseCandidate()
        batches = ControllerMacroEvaluator(cohort).evaluate_many(
            (("first", first), ("second", second))
        )

        self.assertTrue(all(batch.accepted for batch in batches))
        self.assertTrue(
            all(batch.case_timings[0].formal_timing_eligible for batch in batches)
        )
        self.assertEqual(len(first.prepared_cases), 1)
        self.assertEqual(len(second.prepared_cases), 1)
        self.assertEqual(first.prepared_cases[0][1], None)
        self.assertEqual(second.prepared_cases[0][1], None)
        self.assertEqual(first.revealed_queries, second.revealed_queries)
        self.assertEqual(first.revealed_queries, [cohort.public_queries()[0]])
        self.assertIs(first.revealed_queries[0].kind, QueryKind.POINTS)

        gkw_plugin = bind_gkw_macro_loop(
            generator=GKWAnswerFirstGenerator(),
            base_state=None,
            generator_sha256="c" * 64,
        )
        gkw_query = ControllerMacroQueryPlan(
            cohort_id="gkw-official",
            profile="terminal",
            controller_seed=789,
        ).build(gkw_plugin.objective, _gkw_case())
        self.assertIs(gkw_query.kind, QueryKind.POINTS)
        self.assertEqual(
            len(gkw_query.points),
            GKW_MACRO_POINT_COUNTS["terminal"],
        )


if __name__ == "__main__":
    unittest.main()
