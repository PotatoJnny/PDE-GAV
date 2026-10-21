"""Dataset adapters used by the Navier--Stokes ProblemProgram.

There is intentionally no geometry name in this module.  An adapter is chosen
from the manifest schema and reads the coordinates and fields supplied by that
dataset.  The legacy numerical adapter and the completed experimental-field
adapter therefore expose the same public ``CaseRecord`` contract.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path
from threading import Lock
from typing import Mapping, Protocol

from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    AnswerView,
    Case,
    CaseRecord,
    CaseSource,
    CaseVisibility,
    DatasetInput,
    PreparedDataset,
    QueryKind,
)

from .._regular_grid import answer_from_regular_grid
from .geometry import NSGeometry, geometry_from_dataset


class NSDatasetAdapter(Protocol):
    """Translate one documented manifest schema into NS Cases."""

    format_id: str

    def accepts(self, manifest: Mapping[str, object]) -> bool: ...

    def prepare(
        self,
        dataset: DatasetInput,
        manifest_path: Path,
        manifest: Mapping[str, object],
    ) -> PreparedDataset: ...


# Cache only Controller-private raw arrays, never query plans or finite answers.
# A bounded shared working set serves repeated development queries from either
# Agent without retaining the entire original-data bank in every evaluator.
_NS_TRUTH_CACHE_MAX_BYTES = 128 * 1024 * 1024
_NS_TRUTH_CACHE_BYTES = 0
_NS_TRUTH_ARRAY_CACHE: OrderedDict[tuple, object] = OrderedDict()
_NS_TRUTH_CACHE_LOCK = Lock()


def _ns_truth_file_identity(case_dir: Path) -> tuple:
    hidden = (case_dir / "hidden_truth_cache.npz").stat()
    public = (case_dir / "public_arrays.npz").stat()
    return (str(case_dir), hidden.st_size, hidden.st_mtime_ns,
            public.st_size, public.st_mtime_ns)


def _ns_truth_arrays(case_dir: Path, query_fields: tuple[str, ...]) -> Mapping[str, object]:
    """Read only requested field members; keep immutable arrays in a byte LRU."""
    import numpy as np

    global _NS_TRUTH_CACHE_BYTES
    missing_fields = set(query_fields) - {"velocity_x", "velocity_y", "pressure"}
    if missing_fields:
        raise ValueError(f"truth does not provide requested fields: {sorted(missing_fields)}")
    names = ("t", "x", "y")
    if set(query_fields) & {"velocity_x", "velocity_y"}:
        names += ("u_true",)
    if "pressure" in query_fields:
        names += ("pressure",)
    for _attempt in range(3):
        # Stat before every lookup: deletion or a new file version cannot return
        # an old cached answer. Frozen datasets do not require per-query hashing.
        identity = _ns_truth_file_identity(case_dir)
        values = {}
        with _NS_TRUTH_CACHE_LOCK:
            for name in names:
                key = (identity, name)
                if key in _NS_TRUTH_ARRAY_CACHE:
                    values[name] = _NS_TRUTH_ARRAY_CACHE[key]
                    _NS_TRUTH_ARRAY_CACHE.move_to_end(key)
        missing = tuple(name for name in names if name not in values)
        # File reads and decompression stay outside the shared cache lock so
        # unrelated Cases and simultaneous Agent queries can make progress.
        for filename, available in (("hidden_truth_cache.npz", ("t", "u_true", "pressure")),
                                    ("public_arrays.npz", ("x", "y"))):
            needed = tuple(name for name in missing if name in available)
            if needed:
                with np.load(case_dir / filename, allow_pickle=False) as arrays:
                    for name in needed:
                        array = np.asarray(arrays[name])
                        array.setflags(write=False)
                        values[name] = array
        if missing and _ns_truth_file_identity(case_dir) != identity:
            # A writer changed a source while it was read. Do not publish a
            # mixture of versions; retry from both newly observed identities.
            continue
        with _NS_TRUTH_CACHE_LOCK:
            for name in names:
                key = (identity, name)
                existing = _NS_TRUTH_ARRAY_CACHE.get(key)
                if existing is not None:
                    values[name] = existing
                    _NS_TRUTH_ARRAY_CACHE.move_to_end(key)
                    continue
                array = values[name]
                if array.nbytes > _NS_TRUTH_CACHE_MAX_BYTES:
                    continue
                _NS_TRUTH_ARRAY_CACHE[key] = array
                _NS_TRUTH_CACHE_BYTES += array.nbytes
                while _NS_TRUTH_CACHE_BYTES > _NS_TRUTH_CACHE_MAX_BYTES:
                    _key, evicted = _NS_TRUTH_ARRAY_CACHE.popitem(last=False)
                    _NS_TRUTH_CACHE_BYTES -= evicted.nbytes
        return values
    raise RuntimeError("NS truth files changed repeatedly while being read")


@dataclass(frozen=True)
class NSFieldTruth:
    """Controller-only velocity and pressure arrays for a regular-grid case."""

    case_dir: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_dir", Path(self.case_dir).expanduser().resolve())

    def answer(self, query: AnswerQuery) -> AnswerView:
        arrays = _ns_truth_arrays(self.case_dir, query.fields)
        fields = {}
        for name, component in (("velocity_x", 0), ("velocity_y", 1)):
            if name in query.fields:
                fields[name] = arrays["u_true"][:, component]
        if "pressure" in query.fields:
            fields["pressure"] = arrays["pressure"]
        return answer_from_regular_grid(
            query,
            axis_order=("t", "x", "y"),
            native_axes={name: arrays[name] for name in ("t", "x", "y")},
            fields=fields,
        )


@dataclass(frozen=True)
class NSPointCloudTruth:
    """Controller-only velocity on the native points supplied by a dataset."""

    case_file: Path

    def answer(self, query: AnswerQuery) -> AnswerView:
        import numpy as np

        with np.load(self.case_file, allow_pickle=False) as arrays:
            velocity = np.asarray(arrays["velocity"], dtype=np.float64)
            x = np.asarray(arrays["x"], dtype=np.float64)
            y = np.asarray(arrays["y"], dtype=np.float64)
            times = np.asarray(arrays["time"], dtype=np.float64)
        _validate_point_cloud_arrays(velocity, x, y, times, self.case_file)
        unknown = set(query.fields) - {"velocity_x", "velocity_y"}
        if unknown:
            raise ValueError(
                "this completed NS dataset contains only velocity_x and "
                f"velocity_y; unavailable fields: {sorted(unknown)}"
            )
        if query.kind is QueryKind.POINTS:
            return _answer_point_queries(query, velocity, x, y, times)
        return _answer_native_point_cloud(query, velocity, x, y, times)


@dataclass(frozen=True)
class NSPointCloudArrayTruth:
    """In-memory truth for a Generator-created point-cloud velocity Case."""

    velocity: object
    x: object
    y: object
    times: object

    def answer(self, query: AnswerQuery) -> AnswerView:
        import numpy as np

        velocity = np.asarray(self.velocity, dtype=np.float64)
        x = np.asarray(self.x, dtype=np.float64)
        y = np.asarray(self.y, dtype=np.float64)
        times = np.asarray(self.times, dtype=np.float64)
        _validate_point_cloud_arrays(velocity, x, y, times, Path("generated Case"))
        unknown = set(query.fields) - {"velocity_x", "velocity_y"}
        if unknown:
            raise ValueError(
                "this generated NS Case contains only velocity_x and velocity_y; "
                f"unavailable fields: {sorted(unknown)}"
            )
        if query.kind is QueryKind.POINTS:
            return _answer_point_queries(query, velocity, x, y, times)
        return _answer_native_point_cloud(query, velocity, x, y, times)


class LegacyNSD0Adapter:
    """Read the original regular-grid numerical NS dataset."""

    format_id = "ns_d0_v1"

    def accepts(self, manifest: Mapping[str, object]) -> bool:
        schema = str(manifest.get("schema", ""))
        if schema:
            return schema == self.format_id
        splits = manifest.get("splits")
        if not isinstance(splits, Mapping):
            return False
        return any(
            isinstance(rows, list)
            and rows
            and isinstance(rows[0], Mapping)
            and "case_dir" in rows[0]
            for rows in splits.values()
        )

    def prepare(
        self,
        dataset: DatasetInput,
        manifest_path: Path,
        manifest: Mapping[str, object],
    ) -> PreparedDataset:
        splits = _splits(manifest)
        development_split = str(
            dataset.settings.get("development_split", "validation")
        )
        hidden_split = str(
            dataset.settings.get("hidden_split", "d0_distribution_test")
        )
        development_rows = _selected_rows(
            splits,
            development_split,
            dataset.settings.get("development_cases", 16),
        )
        hidden_rows = _selected_rows(
            splits,
            hidden_split,
            dataset.settings.get("hidden_cases", 8),
        )
        development = tuple(
            self._record(row, manifest_path, CaseVisibility.DEVELOPMENT)
            for row in development_rows
        )
        hidden = tuple(
            self._record(row, manifest_path, CaseVisibility.HIDDEN)
            for row in hidden_rows
        )
        first_geometry = _case_geometry(development[0].case)
        return PreparedDataset(
            dataset=dataset,
            development_cases=development,
            hidden_cases=hidden,
            training_summary={
                "dataset_format": self.format_id,
                "manifest": str(manifest_path),
                "split_counts": _split_counts(splits),
                "fields": ("velocity_x", "velocity_y", "pressure"),
                "geometry": first_geometry,
                "topology": "regular_grid",
            },
        )

    @staticmethod
    def _record(
        row: Mapping[str, object],
        manifest_path: Path,
        visibility: CaseVisibility,
    ) -> CaseRecord:
        case_id = _case_id(row)
        case_dir = _resolve_case_dir(row, manifest_path, case_id)
        question = _read_object(case_dir / "public_question.json")
        specification = _read_object(case_dir / "case_spec.json")
        public_arrays = _read_public_arrays(case_dir / "public_arrays.npz")
        question_geometry = _mapping(question.get("geometry"))
        geometry = geometry_from_dataset(
            {},
            question_geometry,
            default_topology="regular_grid",
            coordinate_names=("x", "y"),
        )
        case = Case(
            case_id=case_id,
            source=CaseSource.INITIAL,
            visibility=visibility,
            public_data={
                "equation": "2D incompressible Navier--Stokes",
                "geometry": geometry.public_description(),
                "question": question,
                "case_parameters": specification.get("parameters", {}),
                "arrays": public_arrays,
                # The source directory also contains truth and is never public.
            },
            tags=("ns", str(specification.get("family", "numerical"))),
        )
        return CaseRecord(case, NSFieldTruth(case_dir))


class CompletedNSFieldAdapter:
    """Read completed velocity Cases on dataset-provided spatial samples."""

    format_id = "pde_completed_case_dataset_v1"

    def accepts(self, manifest: Mapping[str, object]) -> bool:
        return (
            str(manifest.get("schema", "")) == self.format_id
            and str(manifest.get("problem", "")).lower() in {"ns", "navier_stokes"}
        )

    def prepare(
        self,
        dataset: DatasetInput,
        manifest_path: Path,
        manifest: Mapping[str, object],
    ) -> PreparedDataset:
        fields = _normalized_fields(manifest.get("fields"))
        if not {"velocity_x", "velocity_y"}.issubset(fields):
            raise ValueError(
                "completed NS field datasets need both velocity components"
            )
        splits = _splits(manifest)
        evaluation_reference = str(
            dataset.settings.get("evaluation_reference", "auto")
        )
        if evaluation_reference not in {"auto", "measurement", "completed"}:
            raise ValueError(
                "dataset.settings.evaluation_reference must be auto, measurement, "
                "or completed"
            )
        default_development = "development" if "development" in splits else "train"
        default_hidden = "verifier" if "verifier" in splits else "validation"
        development_split = str(
            dataset.settings.get("development_split", default_development)
        )
        hidden_split = str(dataset.settings.get("hidden_split", default_hidden))
        generator_training_split = str(
            dataset.settings.get(
                "generator_training_split",
                "generator_training"
                if "generator_training" in splits
                else development_split,
            )
        )
        generator_validation_split = str(
            dataset.settings.get(
                "generator_validation_split",
                "generator_validation"
                if "generator_validation" in splits
                else hidden_split,
            )
        )
        final_test_split = str(
            dataset.settings.get(
                "final_test_split",
                "final_test" if "final_test" in splits else hidden_split,
            )
        )
        development_rows = _selected_rows(
            splits,
            development_split,
            dataset.settings.get("development_cases", 0),
        )
        hidden_rows = _selected_rows(
            splits,
            hidden_split,
            dataset.settings.get("hidden_cases", 0),
        )
        generator_training_rows = _selected_rows(
            splits,
            generator_training_split,
            dataset.settings.get("generator_training_cases", 0),
        )
        generator_validation_rows = _selected_rows(
            splits,
            generator_validation_split,
            dataset.settings.get("generator_validation_cases", 0),
        )
        # The canonical Controller owns one immutable hidden inventory.  It
        # seals BASE Terminal cohorts before it allocates any per-round
        # Official Cases, so a distinct final-test split must occupy the first
        # segment of that inventory.  When an older completed manifest has no
        # distinct final-test split, keep the single hidden cohort instead of
        # duplicating the same Case ids.
        final_test_rows = (
            _selected_rows(
                splits,
                final_test_split,
                dataset.settings.get("final_test_cases", 0),
            )
            if final_test_split != hidden_split
            else ()
        )
        geometry = geometry_from_dataset(
            dataset.geometry,
            _mapping(manifest.get("geometry")),
            default_topology="point_cloud",
            coordinate_names=("x", "y"),
        )
        if geometry.topology != "point_cloud":
            raise ValueError(
                "the current completed-field adapter expects point_cloud storage; "
                "regular-grid datasets use the numerical adapter and an "
                "unstructured mesh can add its own adapter"
            )
        def records(rows, visibility):
            return tuple(
                self._record(
                    row,
                    manifest_path,
                    manifest,
                    geometry,
                    visibility,
                    evaluation_reference,
                )
                for row in rows
            )

        development = records(development_rows, CaseVisibility.DEVELOPMENT)
        official_hidden = records(hidden_rows, CaseVisibility.HIDDEN)
        generator_training = records(
            generator_training_rows,
            CaseVisibility.HIDDEN,
        )
        generator_validation = records(
            generator_validation_rows,
            CaseVisibility.HIDDEN,
        )
        final_test = records(final_test_rows, CaseVisibility.HIDDEN)
        _ensure_distinct_case_ids(final_test, official_hidden)
        hidden = final_test + official_hidden
        representation = _representation_path(manifest, manifest_path)
        return PreparedDataset(
            dataset=dataset,
            development_cases=development,
            hidden_cases=hidden,
            training_summary={
                "dataset_format": self.format_id,
                "manifest": str(manifest_path),
                "split_counts": _split_counts(splits),
                "fields": tuple(sorted(fields)),
                "geometry": geometry.public_description(),
                "topology": geometry.topology,
                "representation": str(representation) if representation else None,
                "source_kind": manifest.get("source_kind"),
                "evaluation_reference": evaluation_reference,
                "reconstruction_method": manifest.get("reconstruction_method"),
                "hidden_case_order": (
                    ("terminal_final_test", "official")
                    if final_test
                    else ("official",)
                ),
                "terminal_final_test_cases": len(final_test),
                "official_hidden_cases": len(official_hidden),
            },
            generator_training_cases=generator_training,
            generator_validation_cases=generator_validation,
        )

    @staticmethod
    def _record(
        row: Mapping[str, object],
        manifest_path: Path,
        manifest: Mapping[str, object],
        geometry: NSGeometry,
        visibility: CaseVisibility,
        evaluation_reference: str,
    ) -> CaseRecord:
        import numpy as np

        case_id = _case_id(row)
        case_file = _resolve_case_file(row, manifest_path, case_id)
        measurement_file = None
        reference = row.get("measurement_reference_file")
        if reference:
            candidate = _resolve_path(str(reference), manifest_path.parent)
            if not candidate.is_file():
                raise FileNotFoundError(
                    "PIV measurement reference does not exist: "
                    f"{candidate}"
                )
            measurement_file = candidate
        if evaluation_reference == "measurement" and measurement_file is None:
            raise ValueError(
                f"Case {case_id} has no measurement_reference_file, but the Run "
                "requires measurement evaluation"
            )
        use_measurement = (
            measurement_file is not None and evaluation_reference != "completed"
        )
        reference_kind = (
            "withheld_measurements" if use_measurement else "completed_reconstruction"
        )
        with np.load(case_file, allow_pickle=False) as arrays:
            velocity = np.asarray(arrays["velocity"])
            x = np.asarray(arrays["x"], dtype=np.float64)
            y = np.asarray(arrays["y"], dtype=np.float64)
            times = np.asarray(arrays["time"], dtype=np.float64)
            public_values = np.asarray(arrays["public_values"], dtype=np.float64)
            sensor_indices = np.asarray(arrays["sensor_indices"], dtype=np.int64)
            observed_frames = np.asarray(arrays["observed_frames"], dtype=np.int64)
        _validate_point_cloud_arrays(velocity, x, y, times, case_file)
        _validate_sparse_observations(
            public_values,
            sensor_indices,
            observed_frames,
            velocity,
            case_file,
        )
        source_dataset = _mapping(manifest.get("source_dataset"))
        observation_layout = _mapping(manifest.get("observation_layout"))
        problem = {
            "equation": "2D incompressible Navier--Stokes",
            **dict(_mapping(source_dataset.get("problem"))),
        }
        case = Case(
            case_id=case_id,
            source=CaseSource.INITIAL,
            visibility=visibility,
            public_data={
                "equation": problem["equation"],
                "geometry": geometry.public_description(),
                "coordinates": {
                    "x": tuple(float(value) for value in x),
                    "y": tuple(float(value) for value in y),
                    "time": tuple(float(value) for value in times),
                },
                "observations": {
                    "fields": ("velocity_x", "velocity_y"),
                    "values": public_values.tolist(),
                    "sensor_indices": tuple(int(value) for value in sensor_indices),
                    "observed_frames": tuple(int(value) for value in observed_frames),
                    "noise_standard_deviation": tuple(
                        float(value)
                        for value in observation_layout.get(
                            "noise_standard_deviation", (1.0, 1.0)
                        )
                    ),
                },
                "source_recording": row.get("source_recording"),
                "source_metadata": dict(_mapping(row.get("source_metadata"))),
                "source_frames": tuple(int(value) for value in row.get("source_frames", ())),
                "evaluation_reference": reference_kind,
                "reconstruction_method": manifest.get("reconstruction_method"),
            },
            tags=("ns", "completed_field", geometry.topology, reference_kind),
        )
        # The completed field teaches the Generator what a full Case looks like.
        # The withheld measurements answer a different product question: did the
        # solver recover what the experiment actually measured?
        completed_truth = NSPointCloudTruth(case_file)
        evaluation_truth = NSPointCloudTruth(
            measurement_file if use_measurement else case_file
        )
        return CaseRecord(
            case,
            evaluation_truth,
            generator_target=completed_truth,
        )


def default_ns_dataset_adapters() -> tuple[NSDatasetAdapter, ...]:
    """Adapters bundled with the first NS product examples."""

    return (CompletedNSFieldAdapter(), LegacyNSD0Adapter())


def _answer_native_point_cloud(query, velocity, x, y, times) -> AnswerView:
    import numpy as np

    requested_t = int(query.resolution.get("t", len(times)))
    requested_points = int(query.resolution.get("point", len(x)))
    if set(query.resolution) - {"t", "point"}:
        raise ValueError(
            "point-cloud NS grid queries use 't' and 'point' resolutions; "
            "use an explicit POINTS query for arbitrary x/y coordinates"
        )
    time_indices = _even_indices(len(times), requested_t)
    point_indices = _even_indices(len(x), requested_points)
    selected = velocity[np.ix_(time_indices, (0, 1), point_indices)]
    fields = {}
    for name in query.fields:
        component = 0 if name == "velocity_x" else 1
        fields[name] = tuple(float(value) for value in selected[:, component].ravel())
    return AnswerView(
        kind=query.kind,
        fields=fields,
        axes={
            "t": tuple(float(value) for value in times[time_indices]),
            "point": tuple(float(value) for value in point_indices),
            "x": tuple(float(value) for value in x[point_indices]),
            "y": tuple(float(value) for value in y[point_indices]),
        },
        shape=(len(time_indices), len(point_indices)),
    )


def _answer_point_queries(query, velocity, x, y, times) -> AnswerView:
    import numpy as np

    values = {name: [] for name in query.fields}
    for point in query.points:
        missing = {"t", "x", "y"} - set(point)
        if missing:
            raise ValueError(f"NS point query is missing coordinates: {sorted(missing)}")
        time_index = int(np.argmin(np.abs(times - float(point["t"]))))
        distances = np.square(x - float(point["x"])) + np.square(
            y - float(point["y"])
        )
        point_index = int(np.argmin(distances))
        for name in query.fields:
            component = 0 if name == "velocity_x" else 1
            values[name].append(float(velocity[time_index, component, point_index]))
    return AnswerView(
        kind=query.kind,
        fields={name: tuple(rows) for name, rows in values.items()},
        points=query.points,
        shape=(len(query.points),),
    )


def _even_indices(native_count: int, requested_count: int):
    import numpy as np

    if requested_count < 1:
        raise ValueError("requested point-cloud resolution must be positive")
    if requested_count > native_count:
        raise ValueError(
            f"requested {requested_count} samples from {native_count} native values"
        )
    if requested_count == native_count:
        return np.arange(native_count, dtype=np.int64)
    return np.rint(
        np.linspace(0, native_count - 1, requested_count)
    ).astype(np.int64)


def _validate_point_cloud_arrays(velocity, x, y, times, path: Path) -> None:
    if velocity.ndim != 3 or velocity.shape[1] != 2:
        raise ValueError(f"{path} velocity must have shape [time,2,point]")
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError(f"{path} x/y coordinates must be matching vectors")
    if velocity.shape[-1] != len(x) or velocity.shape[0] != len(times):
        raise ValueError(f"{path} coordinates do not match velocity")


def _validate_sparse_observations(
    values,
    sensor_indices,
    observed_frames,
    velocity,
    path: Path,
) -> None:
    expected = (len(observed_frames), 2, len(sensor_indices))
    if values.shape != expected:
        raise ValueError(f"{path} public_values must have shape {expected}")
    if len(sensor_indices) == 0 or len(observed_frames) == 0:
        raise ValueError(f"{path} contains no public observations")
    if sensor_indices.min() < 0 or sensor_indices.max() >= velocity.shape[-1]:
        raise ValueError(f"{path} sensor index lies outside the point cloud")
    if observed_frames.min() < 0 or observed_frames.max() >= velocity.shape[0]:
        raise ValueError(f"{path} observed frame lies outside the time window")


def _normalized_fields(value: object) -> set[str]:
    if not isinstance(value, list):
        raise ValueError("completed dataset manifest needs a fields list")
    aliases = {"u": "velocity_x", "v": "velocity_y", "p": "pressure"}
    return {aliases.get(str(name), str(name)) for name in value}


def _representation_path(
    manifest: Mapping[str, object], manifest_path: Path
) -> Path | None:
    artifacts = _mapping(manifest.get("artifacts"))
    value = artifacts.get("representation")
    if value:
        return _resolve_path(str(value), manifest_path.parent)
    fallback = manifest_path.parent / "artifacts" / "trajectory_pod.npz"
    return fallback.resolve() if fallback.is_file() else None


def _selected_rows(
    splits: Mapping[str, object],
    name: str,
    raw_count: object,
) -> tuple[Mapping[str, object], ...]:
    rows = splits.get(name)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"NS dataset split {name!r} is missing or empty")
    count = int(raw_count)
    if count == 0:
        count = len(rows)
    if count < 1 or count > len(rows):
        raise ValueError(
            f"requested {count} cases from NS split {name!r}, which contains {len(rows)}"
        )
    selected = rows[:count]
    if not all(isinstance(row, Mapping) for row in selected):
        raise ValueError(f"NS split {name!r} contains an invalid row")
    return tuple(selected)


def _splits(manifest: Mapping[str, object]) -> Mapping[str, object]:
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("NS dataset manifest needs a 'splits' object")
    return splits


def _split_counts(splits: Mapping[str, object]) -> dict[str, int]:
    return {
        str(name): len(rows)
        for name, rows in splits.items()
        if isinstance(rows, list)
    }


def _case_id(row: Mapping[str, object]) -> str:
    case_id = str(row.get("case_id", "")).strip()
    if not case_id:
        raise ValueError("NS manifest row has no case_id")
    return case_id


def _resolve_case_file(
    row: Mapping[str, object], manifest_path: Path, case_id: str
) -> Path:
    raw = str(row.get("case_file", ""))
    if not raw:
        raise ValueError(f"completed NS case {case_id!r} has no case_file")
    path = _resolve_path(raw, manifest_path.parent)
    if not path.is_file():
        raise FileNotFoundError(f"cannot resolve completed NS case {case_id!r}: {path}")
    return path


def _resolve_case_dir(
    row: Mapping[str, object], manifest_path: Path, case_id: str
) -> Path:
    raw = Path(str(row.get("case_dir", "")))
    candidates = (
        raw,
        manifest_path.parent / raw,
        manifest_path.parent / "cases" / case_id,
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(f"cannot resolve NS case directory for {case_id!r}")


def _resolve_path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _read_public_arrays(path: Path) -> Mapping[str, object]:
    """Load only arrays that a numerical question declares public."""

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


def _case_geometry(case: Case) -> Mapping[str, object]:
    return _mapping(case.public_data.get("geometry"))


def _ensure_distinct_case_ids(
    first: tuple[CaseRecord, ...],
    second: tuple[CaseRecord, ...],
) -> None:
    overlap = {
        record.case.case_id for record in first
    }.intersection(record.case.case_id for record in second)
    if overlap:
        raise ValueError(
            "NS final-test and Official hidden splits must contain distinct "
            f"Case ids; overlap: {sorted(overlap)}"
        )
