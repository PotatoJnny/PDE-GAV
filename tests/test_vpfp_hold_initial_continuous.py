from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest

from pde_agent_verifier_solver.plugins.vpfp import (
    VPFPHoldInitialMacroProgram,
)
from pde_agent_verifier_solver.plugins.vpfp.macro_matching import (
    VPFP_MACRO_FIELDS,
)
from pde_agent_verifier_solver.plugins.vpfp.tasks._shared import (
    private_vpfp_macro_query,
)
from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    Case,
    CaseSource,
    CaseVisibility,
)


def _case():
    x = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    y = np.asarray([0.0, 2.0, 4.0], dtype=np.float64)
    base = 10.0 * np.arange(len(x))[:, None] + np.arange(len(y))[None, :]
    arrays = {
        "t": np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
        "x": x,
        "y": y,
        "rho0": base.copy(),
        "momentum0": np.stack((2.0 * base, -base), axis=-1),
        "kinetic_energy0": 100.0 + base,
    }
    case = Case(
        case_id="vpfp-hold-initial-continuous-test",
        source=CaseSource.GENERATED,
        visibility=CaseVisibility.DEVELOPMENT,
        public_data=MappingProxyType(
            {
                "question": MappingProxyType(
                    {
                        "domain": MappingProxyType(
                            {"time": (0.0, 1.0)}
                        ),
                        "grid": MappingProxyType(
                            {"x_domain": (4.0, 6.0)}
                        ),
                    }
                ),
                "arrays": MappingProxyType(arrays),
            }
        ),
        round_index=0,
        tags=("vpfp", "cold-start"),
    )
    return case, arrays


def test_query_blind_hold_initial_periodically_interpolates_points():
    case, arrays = _case()
    program = VPFPHoldInitialMacroProgram()
    prepared = program.prepare_solution(case, solve_budget=1.0)
    arrays["rho0"][:] = -999.0
    query = AnswerQuery.at_points(
        fields=VPFP_MACRO_FIELDS,
        points=(
            {"t": 0.25, "x": 1.0, "y": 2.0},
            {"t": 0.75, "x": 1.5, "y": 3.0},
            {"t": 0.50, "x": 3.5, "y": 5.0},
        ),
    )

    answer = prepared.evaluate(query)

    assert answer.points == query.points
    np.testing.assert_allclose(answer.fields["rho"], (11.0, 16.5, 16.0))
    np.testing.assert_allclose(
        answer.fields["momentum_x"], (22.0, 33.0, 32.0)
    )
    np.testing.assert_allclose(
        answer.fields["momentum_y"], (-11.0, -16.5, -16.0)
    )
    np.testing.assert_allclose(
        answer.fields["kinetic_energy"], (111.0, 116.5, 116.0)
    )


def test_hold_initial_preserves_grid_and_canonical_observable_entry_points():
    case, _arrays = _case()
    program = VPFPHoldInitialMacroProgram()
    grid = AnswerQuery.grid(fields=VPFP_MACRO_FIELDS, t=3, x=4, y=3)
    grid_answer = program.predict_observables(case, grid, budget=1.0)
    rho = np.asarray(grid_answer.fields["rho"]).reshape(grid_answer.shape)
    np.testing.assert_array_equal(rho[0], rho[1])
    np.testing.assert_array_equal(rho[1], rho[2])

    points = AnswerQuery.at_points(
        fields=VPFP_MACRO_FIELDS,
        points=({"t": 0.4, "x": 1.0, "y": 2.0},),
    )
    assert program.predict_observables(case, points).fields["rho"] == (11.0,)
    with pytest.raises(ValueError, match="budget"):
        program.prepare_solution(case, solve_budget=0.0)


def test_private_vpfp_query_samples_full_endpoint_free_torus():
    case, _arrays = _case()
    query = private_vpfp_macro_query(
        case,
        {
            "default_query": AnswerQuery.grid(
                fields=VPFP_MACRO_FIELDS, t=3, x=4, y=3
            ),
            "development_point_count": 128,
        },
        seed=19,
    )

    assert all(0.0 <= point["x"] < 4.0 for point in query.points)
    assert all(0.0 <= point["y"] < 6.0 for point in query.points)
    assert any(point["x"] > 3.0 for point in query.points)
    assert any(point["y"] > 4.0 for point in query.points)
