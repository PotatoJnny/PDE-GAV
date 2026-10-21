"""VPFP ProblemProgram for continuous answer-first kinetic cases.

This is the second reference Problem implementation. Unlike NS, the public
question lives in phase space while the default task asks only for exact
full-velocity moments on a time-space grid.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    AnswerView,
    Case,
    CaseRecord,
    CaseSource,
    CaseVisibility,
    DatasetInput,
    DatasetKind,
    FormField,
    PanelCard,
    PreparedDataset,
    ProgramInfo,
    QueryKind,
)

from .._regular_grid import answer_from_regular_grid


_VPFP_MACRO_FIELDS = (
    "rho",
    "momentum_x",
    "momentum_y",
    "kinetic_energy",
)
_VPFP_TRANSPORT_FIELDS = (
    "rho", "electric_x", "electric_y", "momentum_x", "momentum_y",
    "pressure_xx", "pressure_xy", "pressure_yy", "heat_flux_x", "heat_flux_y",
)
VPFPMacroPointEvaluator = Callable[[str, object], Mapping[str, object]]


@dataclass(frozen=True)
class VPFPMacroTruth:
    """Controller-only exact full-velocity moments for one VPFP case."""

    case_dir: Path

    def answer(self, query: AnswerQuery) -> AnswerView:
        import numpy as np

        with np.load(self.case_dir / "public_arrays.npz") as public:
            axes = {
                "t": np.asarray(public["t"]),
                "x": np.asarray(public["x"]),
                "y": np.asarray(public["y"]),
            }
        with np.load(self.case_dir / "hidden_truth_cache.npz") as hidden:
            rho = np.asarray(hidden["rho_true"])
            momentum = _momentum_last(hidden["momentum_true"], rho.shape)
            energy = np.asarray(hidden["kinetic_energy_true"])
        return answer_from_regular_grid(
            query,
            axis_order=("t", "x", "y"),
            native_axes=axes,
            fields={
                "rho": rho,
                "momentum_x": momentum[..., 0],
                "momentum_y": momentum[..., 1],
                "kinetic_energy": energy,
            },
        )


@dataclass(frozen=True)
class VPFPContinuousMacroTruth:
    """Controller-private continuous macros for one active resident answer."""

    case_id: str
    t_axis: tuple[float, ...]
    x_axis: tuple[float, ...]
    y_axis: tuple[float, ...]
    evaluate_points: VPFPMacroPointEvaluator
    evaluate_transport_points: VPFPMacroPointEvaluator | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise ValueError("continuous VPFP truth case id cannot be empty")
        if not callable(self.evaluate_points):
            raise TypeError("continuous VPFP truth evaluator must be callable")
        if self.evaluate_transport_points is not None and not callable(self.evaluate_transport_points):
            raise TypeError("continuous VPFP transport evaluator must be callable")
        for name, values in (
            ("t", self.t_axis),
            ("x", self.x_axis),
            ("y", self.y_axis),
        ):
            _validated_continuous_axis(values, name)

    def answer(self, query: AnswerQuery) -> AnswerView:
        import numpy as np

        if not isinstance(query, AnswerQuery):
            raise TypeError("continuous VPFP truth requires an AnswerQuery")
        allowed = set(_VPFP_MACRO_FIELDS)
        if self.evaluate_transport_points is not None:
            allowed.update(_VPFP_TRANSPORT_FIELDS)
        if len(query.fields) != len(set(query.fields)) or not set(query.fields).issubset(allowed):
            raise ValueError("continuous VPFP truth received unknown macro fields")
        transport = bool(set(query.fields)-set(_VPFP_MACRO_FIELDS))
        evaluate = self.evaluate_transport_points if transport else self.evaluate_points
        if query.kind is QueryKind.POINTS:
            points = _continuous_query_points(query)
            values = _continuous_macro_values(
                evaluate(self.case_id, points), len(points), transport=transport
            )
            return AnswerView(
                kind=query.kind,
                fields={
                    name: tuple(float(value) for value in values[name])
                    for name in query.fields
                },
                points=query.points,
                shape=(len(query.points),),
            )
        if query.kind is not QueryKind.GRID:
            raise ValueError("continuous VPFP truth received an unknown query kind")
        native = {
            "t": np.asarray(self.t_axis, dtype=np.float64),
            "x": np.asarray(self.x_axis, dtype=np.float64),
            "y": np.asarray(self.y_axis, dtype=np.float64),
        }
        axes = {
            name: _continuous_requested_axis(axis, query.resolution.get(name))
            for name, axis in native.items()
        }
        mesh = np.meshgrid(axes["t"], axes["x"], axes["y"], indexing="ij")
        points = np.column_stack([value.reshape(-1) for value in mesh])
        values = _continuous_macro_values(
            evaluate(self.case_id, points), len(points), transport=transport
        )
        shape = tuple(len(axes[name]) for name in ("t", "x", "y"))
        return AnswerView(
            kind=query.kind,
            fields={
                name: tuple(float(value) for value in values[name])
                for name in query.fields
            },
            axes={
                name: tuple(float(value) for value in axes[name])
                for name in ("t", "x", "y")
            },
            shape=shape,
        )


def load_vpfp_case_record(
    case_dir: Path | str,
    *,
    visibility: CaseVisibility,
    source: CaseSource = CaseSource.INITIAL,
    round_index: int | None = None,
    parent_case_id: str | None = None,
    case_id: str | None = None,
    tags: tuple[str, ...] = ("vpfp",),
    continuous_macro_evaluator: VPFPMacroPointEvaluator | None = None,
) -> CaseRecord:
    """Load one materialized VPFP case without exposing its private path.

    This is the equation-owned bridge used by both dataset preparation and
    replayable manufactured-case sources. The caller owns provenance fields;
    the public payload never contains ``case_dir`` because that directory also
    contains controller-only truth.
    """

    if not isinstance(visibility, CaseVisibility):
        raise TypeError("VPFP case visibility must be a CaseVisibility")
    if not isinstance(source, CaseSource):
        raise TypeError("VPFP case source must be a CaseSource")
    if round_index is not None and (
        isinstance(round_index, bool)
        or not isinstance(round_index, int)
        or round_index < 0
    ):
        raise ValueError("VPFP case round_index must be a nonnegative integer")
    resolved = Path(case_dir).expanduser().resolve()
    question = _read_object(resolved / "public_question.json")
    public_arrays = _read_public_arrays(resolved / "public_arrays.npz")
    resolved_case_id = str(case_id or question.get("case_id") or resolved.name)
    if not resolved_case_id.strip():
        raise ValueError("VPFP case id cannot be empty")
    question_case_id = str(question.get("case_id") or "")
    if question_case_id and question_case_id != resolved_case_id:
        raise ValueError("VPFP public question case id does not match the record")
    case = Case(
        case_id=resolved_case_id,
        source=source,
        visibility=visibility,
        public_data={
            "equation": "2D2V Vlasov--Poisson--Fokker--Planck",
            "question": question,
            "arrays": public_arrays,
        },
        round_index=round_index,
        parent_case_id=parent_case_id,
        tags=tuple(tags),
    )
    truth = (
        VPFPMacroTruth(resolved)
        if continuous_macro_evaluator is None
        else VPFPContinuousMacroTruth(
            case_id=resolved_case_id,
            t_axis=_axis_tuple(public_arrays, "t"),
            x_axis=_axis_tuple(public_arrays, "x"),
            y_axis=_axis_tuple(public_arrays, "y"),
            evaluate_points=continuous_macro_evaluator,
        )
    )
    return CaseRecord(case, truth)


class VPFPProblem:
    """Two-dimensional Vlasov--Poisson--Fokker--Planck family."""

    info = ProgramInfo(
        "vpfp-2d2v-answer-first",
        "2D2V Vlasov--Poisson--Fokker--Planck",
        "Continuous phase-space answers with exact full-velocity moments.",
    )

    def setup_fields(self) -> tuple[FormField, ...]:
        return (
            FormField(
                "regime",
                "Collision regime",
                "choice",
                "long_time_strong_collision_v2",
                "The current paper templates provide weak and strong regimes.",
                choices=(
                    (
                        "long_time_strong_collision_v2",
                        "Strong collision",
                    ),
                    (
                        "long_time_weak_collision_v2",
                        "Weak collision",
                    ),
                ),
            ),
            FormField(
                "development_cohort",
                "Agent-visible cohort",
                "text",
                "development",
            ),
            FormField(
                "hidden_cohort",
                "Initial hidden cohort",
                "text",
                "released_training",
            ),
            FormField(
                "development_cases",
                "Initial development cases",
                "integer",
                16,
            ),
            FormField("hidden_cases", "Initial hidden cases", "integer", 8),
        )

    def prepare_dataset(self, dataset: DatasetInput) -> PreparedDataset:
        if dataset.kind is not DatasetKind.NUMERICAL:
            raise ValueError(
                "the current VPFP plugin accepts numerical answer-first cases; "
                "an observation adapter can be added in this Problem class"
            )
        if not dataset.location:
            raise ValueError(
                "VPFP needs a split manifest or long-time release-bank report"
            )
        manifest_path = Path(dataset.location).expanduser().resolve()
        payload = _read_object(manifest_path)
        cohorts = _cohorts(payload)

        development_name = str(
            dataset.settings.get("development_cohort", "development")
        )
        hidden_name = str(
            dataset.settings.get("hidden_cohort", "released_training")
        )
        development_count = int(
            dataset.settings.get("development_cases", 16)
        )
        hidden_count = int(dataset.settings.get("hidden_cases", 8))
        development_rows = _select_rows(
            cohorts,
            development_name,
            development_count,
        )
        hidden_rows = _select_rows(cohorts, hidden_name, hidden_count)
        development = tuple(
            self._record(row, manifest_path, CaseVisibility.DEVELOPMENT)
            for row in development_rows
        )
        hidden = tuple(
            self._record(row, manifest_path, CaseVisibility.HIDDEN)
            for row in hidden_rows
        )
        return PreparedDataset(
            dataset=dataset,
            development_cases=development,
            hidden_cases=hidden,
            training_summary={
                "dataset_format": str(
                    payload.get("schema", "vpfp_answer_first")
                ),
                "manifest": str(manifest_path),
                "cohort_counts": {
                    name: len(rows) for name, rows in cohorts.items()
                },
                "fields": (
                    "rho",
                    "momentum_x",
                    "momentum_y",
                    "kinetic_energy",
                ),
            },
        )

    def case_card(self, case: Case) -> PanelCard:
        question = _mapping(case.public_data.get("question"))
        return PanelCard(
            title=f"VPFP case {case.case_id}",
            summary=(
                "Kinetic equation with requested density, momentum and "
                "kinetic-energy fields."
            ),
            values={
                "source": case.source.value,
                "collision_frequency": question.get("collision_nu"),
                "temperature": question.get("temperature"),
                "time_interval": question.get("time_domain"),
            },
        )

    def _record(
        self,
        row: Mapping[str, object],
        manifest_path: Path,
        visibility: CaseVisibility,
    ) -> CaseRecord:
        case_dir = _resolve_case_dir(row, manifest_path)
        question = _read_object(case_dir / "public_question.json")
        case_id = str(
            row.get("case_id") or question.get("case_id") or case_dir.name
        )
        variant = str(row.get("policy_variant", "initial"))
        return load_vpfp_case_record(
            case_dir,
            visibility=visibility,
            source=_case_source(row),
            case_id=case_id,
            tags=("vpfp", variant),
        )


def _case_source(row: Mapping[str, object]) -> CaseSource:
    """Classify where this Case enters the current product Campaign.

    Rows read from the user supplied dataset are initial Cases by default, even
    when that dataset was itself produced by an earlier Generator experiment.
    A custom manifest may explicitly set source to generated or perturbed.
    """

    value = str(row.get("source", CaseSource.INITIAL.value)).strip().lower()
    try:
        return CaseSource(value)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in CaseSource)
        raise ValueError(
            f"unknown VPFP case source {value!r}; expected one of {allowed}"
        ) from exc


def _cohorts(
    payload: Mapping[str, object],
) -> dict[str, tuple[Mapping[str, object], ...]]:
    raw = payload.get("cohorts")
    if isinstance(raw, Mapping):
        return {
            str(name): _normalize_rows(rows)
            for name, rows in raw.items()
        }

    # Long-time release banks predate the shared split-manifest name. Treat
    # their two visible lists as ordinary cohorts.
    mapping = {
        "development": payload.get("development_case_dirs"),
        "released_training": payload.get(
            "released_completed_case_dirs"
        ),
        "final_holdout": payload.get("final_holdout_case_dirs"),
    }
    cohorts = {
        name: _normalize_rows(rows)
        for name, rows in mapping.items()
        if isinstance(rows, Sequence) and not isinstance(rows, str)
    }
    if not cohorts:
        raise ValueError(
            "VPFP dataset needs a 'cohorts' object or release-bank case lists"
        )
    return cohorts


def _normalize_rows(
    rows: object,
) -> tuple[Mapping[str, object], ...]:
    if not isinstance(rows, Sequence) or isinstance(rows, str):
        raise ValueError("VPFP cohort must be a list")
    normalized = []
    for row in rows:
        if isinstance(row, Mapping):
            normalized.append(dict(row))
        else:
            normalized.append({"case_dir": str(row)})
    return tuple(normalized)


def _select_rows(
    cohorts: Mapping[str, tuple[Mapping[str, object], ...]],
    name: str,
    count: int,
) -> tuple[Mapping[str, object], ...]:
    rows = cohorts.get(name)
    if not rows:
        raise ValueError(f"VPFP cohort {name!r} is missing or empty")
    if count <= 0 or count > len(rows):
        raise ValueError(
            f"requested {count} cases from VPFP cohort {name!r}, "
            f"which contains {len(rows)}"
        )
    return rows[:count]


def _resolve_case_dir(
    row: Mapping[str, object],
    manifest_path: Path,
) -> Path:
    raw = Path(str(row.get("case_dir", "")))
    candidates = (raw, manifest_path.parent / raw)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(f"cannot resolve VPFP case directory: {raw}")


def _momentum_last(value, rho_shape):
    import numpy as np

    momentum = np.asarray(value)
    if momentum.shape == tuple(rho_shape) + (2,):
        return momentum
    if momentum.shape == (
        rho_shape[0],
        2,
        rho_shape[1],
        rho_shape[2],
    ):
        return np.moveaxis(momentum, 1, -1)
    raise ValueError(
        f"unexpected VPFP momentum shape {momentum.shape} "
        f"for density shape {rho_shape}"
    )


def _read_public_arrays(path: Path) -> Mapping[str, object]:
    """Load public initial data, coordinates, and forcing without truth files."""

    import numpy as np

    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as arrays:
        return {name: np.asarray(arrays[name]).copy() for name in arrays.files}


def _read_object(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _axis_tuple(
    arrays: Mapping[str, object], name: str
) -> tuple[float, ...]:
    return tuple(
        float(value)
        for value in _validated_continuous_axis(arrays.get(name), name)
    )


def _validated_continuous_axis(values: object, name: str):
    import numpy as np

    try:
        axis = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"continuous VPFP {name} axis must be numeric") from exc
    if axis.ndim != 1 or axis.size < 1 or not np.all(np.isfinite(axis)):
        raise ValueError(
            f"continuous VPFP {name} axis must be finite nonempty 1D"
        )
    if axis.size > 1 and not np.all(np.diff(axis) > 0.0):
        raise ValueError(
            f"continuous VPFP {name} axis must be strictly increasing"
        )
    return axis


def _continuous_query_points(query: AnswerQuery):
    import numpy as np

    try:
        points = np.asarray(
            [
                [float(point[name]) for name in ("t", "x", "y")]
                for point in query.points
            ],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "continuous VPFP points must use numeric (t,x,y)"
        ) from exc
    if points.shape != (len(query.points), 3) or not np.all(np.isfinite(points)):
        raise ValueError(
            "continuous VPFP points must be finite physical coordinates"
        )
    return points


def _continuous_requested_axis(native, count: int | None):
    import numpy as np

    if count is None or count == len(native):
        return native.copy()
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError(
            "continuous VPFP grid resolutions must be positive integers"
        )
    if count == 1:
        return np.asarray(
            [(float(native[0]) + float(native[-1])) / 2.0],
            dtype=np.float64,
        )
    return np.linspace(
        float(native[0]), float(native[-1]), count, dtype=np.float64
    )


def _continuous_macro_values(payload: object, count: int, *, transport=False):
    import numpy as np

    if not isinstance(payload, Mapping):
        raise TypeError("continuous VPFP evaluator must return a mapping")
    rho = np.asarray(payload.get("rho"), dtype=np.float64)
    momentum = np.asarray(payload.get("momentum"), dtype=np.float64)
    energy = np.asarray(payload.get("kinetic_energy"), dtype=np.float64)
    if (
        rho.shape != (count,)
        or momentum.shape != (count, 2)
        or energy.shape != (count,)
    ):
        raise ValueError(
            "continuous VPFP evaluator returned malformed macro shapes"
        )
    if not all(
        np.all(np.isfinite(value)) for value in (rho, momentum, energy)
    ):
        raise FloatingPointError(
            "continuous VPFP evaluator returned non-finite macros"
        )
    result = {
        "rho": rho,
        "momentum_x": momentum[:, 0],
        "momentum_y": momentum[:, 1],
        "kinetic_energy": energy,
    }
    if transport:
        for name, labels in (("electric", ("electric_x", "electric_y")),
                             ("pressure", ("pressure_xx", "pressure_xy", "pressure_yy")),
                             ("heat_flux", ("heat_flux_x", "heat_flux_y"))):
            value = np.asarray(payload.get(name), dtype=np.float64)
            if value.shape != (count, len(labels)):
                raise ValueError(f"continuous VPFP evaluator returned malformed {name} shape")
            if not np.isfinite(value).all():
                raise FloatingPointError(f"continuous VPFP evaluator returned non-finite {name}")
            result.update({label: value[:, axis] for axis, label in enumerate(labels)})
    return result
