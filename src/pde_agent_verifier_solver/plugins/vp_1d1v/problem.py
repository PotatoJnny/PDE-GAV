"""Canonical generated-data Problem and causal public view for Artemis."""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
from typing import Mapping

from pde_agent_verifier_solver.sdk import (
    AnswerQuery, Case, DatasetInput, FormField, PanelCard, PreparedDataset,
    ProgramInfo, QueryKind,
)
from .contract import VP_1D1V_EQUATION_VERSION, VP_1D1V_MACRO_FIELDS

VP1D_EQUATION_VERSION = VP_1D1V_EQUATION_VERSION
VP1D_MACRO_FIELDS = VP_1D1V_MACRO_FIELDS
VP1D_CACHED_ARRAY_ORACLE_MODE = "cached_array"
VP1D_EXACT_COORDINATE_ORACLE_MODE = "exact_coordinate"
VP1D_PUBLIC_ORACLE_MODES = frozenset({
    VP1D_CACHED_ARRAY_ORACLE_MODE, VP1D_EXACT_COORDINATE_ORACLE_MODE,
})


class VP1DProblem:
    info = ProgramInfo(
        "vp-1d1v-generated-answer-first", "Artemis / 1D1V Vlasov--Poisson",
        "Use admitted continuous kinetic answers and their exactly derived source.",
    )

    def setup_fields(self):
        return (FormField("online_only", "Generate cases online", "boolean", True),)

    def prepare_dataset(self, dataset: DatasetInput) -> PreparedDataset:
        if dataset.location:
            raise ValueError("Artemis uses an admitted Generator checkpoint; dataset location must be empty")
        return PreparedDataset(
            dataset=dataset, development_cases=(), hidden_cases=(),
            training_summary={
                "dataset_role": "optional_generator_prior_only",
                "truth_source": "admitted_continuous_generated_answer",
                "fields": VP1D_MACRO_FIELDS, "online_only": True,
            },
        )

    def case_card(self, case: Case) -> PanelCard:
        arrays = case.public_data.get("arrays", {})
        return PanelCard(
            title=f"Artemis case {case.case_id}",
            summary="Forced periodic 1D1V VP with public initial data and source.",
            values={name: len(arrays.get(name, ())) for name in ("t", "x", "v")},
        )


def causal_horizon_case(case: Case, query: AnswerQuery, *, scoring_coordinate_mode="native_x_v1") -> Case:
    """Detach the source prefix at one non-initial native scoring horizon."""
    import numpy as np

    if query.kind is not QueryKind.POINTS or set(query.fields) != set(VP1D_MACRO_FIELDS):
        raise ValueError("Artemis causal scoring requires M0--M3,E POINTS")
    arrays = case.public_data["arrays"]
    time = np.asarray(arrays["t"], dtype=np.float64)
    space = np.asarray(arrays["x"], dtype=np.float64)
    rows = np.asarray([(p["t"], p["x"]) for p in query.points], dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 2 or not len(rows) or not np.isfinite(rows).all():
        raise ValueError("Artemis causal points must be nonempty and finite")
    horizons = np.unique(rows[:, 0])
    matches = np.flatnonzero(time == horizons[0])
    if len(horizons) != 1 or len(matches) != 1 or matches[0] == 0:
        raise ValueError("Artemis causal points must share one non-initial native horizon")
    if scoring_coordinate_mode == "native_x_v1":
        if not np.isin(rows[:, 1], space).all():
            raise ValueError("Artemis admitted scoring uses native x coordinates")
    elif scoring_coordinate_mode == "continuous_x_v1":
        left, right = case.public_data["equation"]["domain"]["x_interval"]
        if np.any(rows[:, 1] < left) or np.any(rows[:, 1] >= right):
            raise ValueError("Artemis scoring points leave the periodic spatial domain")
    else:
        raise ValueError("unknown Artemis scoring_coordinate_mode")
    stop = int(matches[0]) + 1
    projected = {}
    for name, value in arrays.items():
        copy = np.array(value[:stop] if name in ("t", "source") else value, copy=True)
        copy.setflags(write=False)
        projected[name] = copy
    equation = _mutable(case.public_data["equation"])
    equation["domain"]["time_interval"] = (float(time[0]), float(horizons[0]))
    return replace(case, public_data=MappingProxyType({
        "equation": _freeze(equation), "arrays": MappingProxyType(projected),
    }))


def _mutable(value):
    if isinstance(value, Mapping):
        return {key: _mutable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_mutable(item) for item in value)
    return value


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    return value
