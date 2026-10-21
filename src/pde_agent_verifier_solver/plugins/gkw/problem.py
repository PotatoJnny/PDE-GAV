"""GKW ProblemProgram for released completed macroscopic answers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

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

from .contract import (
    GKW_MACRO_FIELDS,
    gkw_answer_view_from_pairs,
    gkw_point_query_coordinates,
    validate_gkw_macro_query,
)
from .release_io import (
    GKWReleasePackage,
    freeze_gkw_public_data,
    load_gkw_release,
)

@dataclass(frozen=True)
class GKWReleasedMacroTruth:
    """Controller-only numerical macro values from one completed GKW case."""

    case_dir: Path
    case_id: str

    def answer(self, query: AnswerQuery) -> AnswerView:
        import numpy as np

        package = load_gkw_release(
            self.case_dir, expected_case_id=self.case_id
        )
        times, indices, values = package.times, package.field_indices, package.values
        case = _case_for_truth_coordinates(package)
        if query.kind is QueryKind.GRID:
            query_times, query_indices = validate_gkw_macro_query(case, query)
            if not np.array_equal(query_times, times) or not np.array_equal(
                query_indices, indices
            ):
                raise ValueError("GKW released GRID query changed its native panel")
            return gkw_answer_view_from_pairs(
                query,
                values,
                times=times,
                field_indices=indices,
            )

        point_times, point_indices = gkw_point_query_coordinates(case, query)
        selected = {name: [] for name in GKW_MACRO_FIELDS}
        for time_value, index_row in zip(point_times, point_indices):
            time_matches = np.flatnonzero(
                np.isclose(times, time_value, rtol=0.0, atol=1.0e-12)
            )
            field_matches = np.flatnonzero(
                np.all(indices == index_row[None, :], axis=1)
            )
            if len(time_matches) != 1 or len(field_matches) != 1:
                raise ValueError(
                    "GKW released POINTS query must use released coordinates"
                )
            for name in GKW_MACRO_FIELDS:
                selected[name].append(
                    values[name][time_matches[0], field_matches[0], :]
                )
        selected_arrays = {
            name: np.asarray(rows, dtype=np.float64)
            for name, rows in selected.items()
        }
        return gkw_answer_view_from_pairs(
            query,
            selected_arrays,
            times=point_times,
            field_indices=point_indices,
        )


class GKWProblem:
    """Electrostatic GKW family with continuous hidden ion-state answers."""

    info = ProgramInfo(
        "gkw-continuous-answer-first",
        "GKW electrostatic gyrokinetics",
        "Continuous five-dimensional ion-state answers scored through four "
        "velocity-contracted complex macroscopic fields.",
    )

    def setup_fields(self) -> tuple[FormField, ...]:
        return (
            FormField(
                "development_cohort",
                "Agent-visible cohort",
                "text",
                "development",
            ),
            FormField(
                "hidden_cohort",
                "Hidden evaluation cohort",
                "text",
                "released_training",
            ),
            FormField("development_cases", "Development cases", "integer", 4),
            FormField("hidden_cases", "Hidden cases", "integer", 4),
        )

    def prepare_dataset(self, dataset: DatasetInput) -> PreparedDataset:
        if dataset.kind is not DatasetKind.NUMERICAL:
            raise ValueError(
                "the current GKW plugin accepts numerical answer-first releases"
            )
        if not dataset.location:
            raise ValueError("GKW needs a macro release-bank manifest")
        manifest_path = Path(dataset.location).expanduser().resolve()
        payload = _read_object(manifest_path)
        cohorts = _cohorts(payload)
        development_name = str(
            dataset.settings.get("development_cohort", "development")
        )
        hidden_name = str(
            dataset.settings.get("hidden_cohort", "released_training")
        )
        development_count = int(dataset.settings.get("development_cases", 4))
        hidden_count = int(dataset.settings.get("hidden_cases", 4))
        development = tuple(
            self._record(row, manifest_path, CaseVisibility.DEVELOPMENT)
            for row in _select_rows(
                cohorts, development_name, development_count
            )
        )
        hidden = tuple(
            self._record(row, manifest_path, CaseVisibility.HIDDEN)
            for row in _select_rows(cohorts, hidden_name, hidden_count)
        )
        _validate_disjoint_cohorts(development, hidden)
        return PreparedDataset(
            dataset=dataset,
            development_cases=development,
            hidden_cases=hidden,
            training_summary={
                "dataset_format": str(payload.get("schema", "gkw_macro_release")),
                "manifest": str(manifest_path),
                "cohort_counts": {
                    name: len(rows) for name, rows in cohorts.items()
                },
                "fields": GKW_MACRO_FIELDS,
                "hidden_primary_object": (
                    "continuous delta_f_i(t,vpar,mu,s,kx,ky)"
                ),
            },
        )

    def case_card(self, case: Case) -> PanelCard:
        query = case.public_data.get("macro_query")
        metadata = case.public_data.get("metadata")
        return PanelCard(
            title=f"GKW case {case.case_id}",
            summary=(
                "Electrostatic gyrokinetic question with complex macros on a "
                "public (time, field-index) panel."
            ),
            values={
                "source": case.source.value,
                "time_count": (
                    len(query.get("times", ()))
                    if isinstance(query, Mapping)
                    else None
                ),
                "field_row_count": (
                    len(query.get("field_indices", ()))
                    if isinstance(query, Mapping)
                    else None
                ),
                "operator_id": (
                    metadata.get("operator_id")
                    if isinstance(metadata, Mapping)
                    else None
                ),
            },
        )

    def _record(
        self,
        row: Mapping[str, object],
        manifest_path: Path,
        visibility: CaseVisibility,
    ) -> CaseRecord:
        case_dir = _resolve_case_dir(row, manifest_path)
        declared_id = row.get("case_id")
        if declared_id is not None and (
            not isinstance(declared_id, str) or not declared_id.strip()
        ):
            raise ValueError("GKW release-bank case_id must be nonempty text")
        package = load_gkw_release(
            case_dir,
            expected_case_id=declared_id if isinstance(declared_id, str) else None,
        )
        case = Case(
            case_id=package.case_id,
            source=_case_source(row),
            visibility=visibility,
            public_data=freeze_gkw_public_data(package),
            tags=("gkw", "macro", "completed-answer-first"),
        )
        return CaseRecord(
            case,
            GKWReleasedMacroTruth(package.case_dir, package.case_id),
        )


def _case_for_truth_coordinates(package: GKWReleasePackage) -> Case:
    return Case(
        case_id=package.case_id,
        source=CaseSource.INITIAL,
        visibility=CaseVisibility.HIDDEN,
        public_data=freeze_gkw_public_data(package),
        tags=("gkw", "macro", "completed-answer-first"),
    )


def _validate_disjoint_cohorts(
    development: Sequence[CaseRecord], hidden: Sequence[CaseRecord]
) -> None:
    records = tuple(development) + tuple(hidden)
    if not all(isinstance(record.truth, GKWReleasedMacroTruth) for record in records):
        raise TypeError("GKW release cohort contains another truth provider")
    identifiers = [record.case.case_id for record in records]
    directories = [record.truth.case_dir.resolve() for record in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("GKW development and hidden case ids must be disjoint")
    if len(set(directories)) != len(directories):
        raise ValueError("GKW development and hidden case directories must be disjoint")


def _cohorts(
    payload: Mapping[str, object],
) -> dict[str, tuple[Mapping[str, object], ...]]:
    raw = payload.get("cohorts")
    if isinstance(raw, Mapping):
        return {
            str(name): _normalize_rows(rows) for name, rows in raw.items()
        }
    mapping = {
        "development": payload.get("development_case_dirs"),
        "released_training": payload.get("released_completed_case_dirs"),
        "final_holdout": payload.get("final_holdout_case_dirs"),
    }
    result = {
        name: _normalize_rows(rows)
        for name, rows in mapping.items()
        if isinstance(rows, Sequence) and not isinstance(rows, str)
    }
    if not result:
        raise ValueError("GKW dataset needs cohorts or release-bank case lists")
    return result


def _normalize_rows(rows: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(rows, Sequence) or isinstance(rows, str):
        raise ValueError("GKW cohort must be a list")
    return tuple(
        dict(row) if isinstance(row, Mapping) else {"case_dir": str(row)}
        for row in rows
    )


def _select_rows(
    cohorts: Mapping[str, tuple[Mapping[str, object], ...]],
    name: str,
    count: int,
) -> tuple[Mapping[str, object], ...]:
    rows = cohorts.get(name)
    if not rows:
        raise ValueError(f"GKW cohort {name!r} is missing or empty")
    if count <= 0 or count > len(rows):
        raise ValueError(
            f"requested {count} GKW cases from {name!r}, which has {len(rows)}"
        )
    return rows[:count]


def _resolve_case_dir(row: Mapping[str, object], manifest_path: Path) -> Path:
    raw = Path(str(row.get("case_dir", "")))
    for candidate in (raw, manifest_path.parent / raw):
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(f"cannot resolve GKW case directory: {raw}")


def _case_source(row: Mapping[str, object]) -> CaseSource:
    value = str(row.get("source", CaseSource.INITIAL.value)).strip().lower()
    try:
        return CaseSource(value)
    except ValueError as exc:
        raise ValueError(f"unknown GKW case source {value!r}") from exc


def _read_object(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value
