"""Represent and reconstruct dense experimental velocity observations.

The NS reconstruction plugin normalizes laboratory-specific MATLAB or HDF5
files into geometry-neutral PIV recordings. It then supplies windowing,
trajectory POD, sparse observation, Gappy-POD, and ES-MDA utilities. Dataset
coordinates and time values are preserved; no cylinder, airfoil, grid size, or
pressure field is assumed by the common representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class RealPIVRecording:
    """One dense PIV movie on points measured in every retained frame.

    The class is geometry-neutral. A cylinder MATLAB file, an airfoil HDF5
    file, or a customer's loader all become the same velocity, coordinates and
    time arrays before reconstruction sees them.
    """

    velocity: np.ndarray  # [frame, component, valid_point]
    x: np.ndarray  # [valid_point]
    y: np.ndarray  # [valid_point]
    sampling_hz: float
    original_grid_shape: tuple[int, int]
    spatial_stride: int
    time: np.ndarray | None = None  # [frame], physical time when supplied
    recording_id: str = "recording"
    metadata: Mapping[str, object] | None = None

    @property
    def frame_count(self) -> int:
        return int(self.velocity.shape[0])

    @property
    def point_count(self) -> int:
        return int(self.velocity.shape[-1])

    def window(self, start: int, length: int) -> np.ndarray:
        stop = int(start) + int(length)
        if start < 0 or stop > self.frame_count:
            raise IndexError("PIV window lies outside the recording")
        return np.asarray(self.velocity[start:stop], dtype=np.float32)

    def window_time(self, start: int, length: int) -> np.ndarray:
        """Return physical time for a window, falling back to sampling_hz."""

        stop = int(start) + int(length)
        if start < 0 or stop > self.frame_count:
            raise IndexError("PIV time window lies outside the recording")
        if self.time is not None:
            return np.asarray(self.time[start:stop], dtype=np.float64)
        return np.arange(int(start), stop, dtype=np.float64) / self.sampling_hz


def load_shang_tu_cylinder_piv(
    path: str | Path,
    *,
    sampling_hz: float = 20.0,
    spatial_stride: int = 2,
) -> RealPIVRecording:
    """Load the Shang--Tu experimental cylinder wake distributed on Zenodo.

    The MATLAB arrays have shape [x_index, y_index, frame].  Exact zero pairs
    mark regions where no velocity was reported.  For a fixed observation
    domain we retain only points measured and finite in every frame.
    """

    from scipy.io import loadmat

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if spatial_stride < 1:
        raise ValueError("spatial_stride must be positive")
    raw = loadmat(source, variable_names=("u", "v", "x", "y"))
    required = {"u", "v", "x", "y"}
    if not required.issubset(raw):
        missing = sorted(required.difference(raw))
        raise ValueError(f"real PIV file is missing arrays: {missing}")

    u_raw = np.asarray(raw["u"])
    v_raw = np.asarray(raw["v"])
    x_raw = np.asarray(raw["x"])
    y_raw = np.asarray(raw["y"])
    if u_raw.ndim != 3 or u_raw.shape != v_raw.shape:
        raise ValueError("u and v must share a three-dimensional PIV grid")
    if x_raw.shape != u_raw.shape[:2] or y_raw.shape != u_raw.shape[:2]:
        raise ValueError("coordinate arrays do not match the PIV spatial grid")

    stride = int(spatial_stride)
    u = np.asarray(u_raw[::stride, ::stride, :], dtype=np.float32).transpose(2, 0, 1)
    v = np.asarray(v_raw[::stride, ::stride, :], dtype=np.float32).transpose(2, 0, 1)
    x_grid = np.asarray(x_raw[::stride, ::stride], dtype=np.float64)
    y_grid = np.asarray(y_raw[::stride, ::stride], dtype=np.float64)
    valid = np.all(np.isfinite(u) & np.isfinite(v), axis=0)
    valid &= np.all((u != 0.0) | (v != 0.0), axis=0)
    if int(np.count_nonzero(valid)) < 100:
        raise ValueError("too few points are measured throughout the recording")

    velocity = np.stack((u[:, valid], v[:, valid]), axis=1)
    return RealPIVRecording(
        velocity=np.asarray(velocity, dtype=np.float32),
        x=np.asarray(x_grid[valid], dtype=np.float64),
        y=np.asarray(y_grid[valid], dtype=np.float64),
        sampling_hz=float(sampling_hz),
        original_grid_shape=tuple(int(value) for value in u_raw.shape[:2]),
        spatial_stride=stride,
        time=np.arange(len(velocity), dtype=np.float64) / float(sampling_hz),
        recording_id=source.stem,
        metadata={"format": "shang_tu_mat"},
    )


def load_piv_recordings(
    path: str | Path,
    *,
    settings: Mapping[str, object] | None = None,
) -> tuple[RealPIVRecording, ...]:
    """Load one or more dense PIV recordings through a data adapter.

    ``settings.format`` may be ``shang_tu_mat``, ``dense_piv_hdf5``, or
    ``auto``. Directories are expanded with ``settings.file_glob``. The
    reconstruction algorithm only receives ``RealPIVRecording``, so a new
    laboratory format does not add geometry branches to the Generator.
    """

    values = dict(settings or {})
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    file_glob = str(values.get("file_glob", "*.h5"))
    if source.is_dir():
        sources = sorted(
            candidate for candidate in source.glob(file_glob) if candidate.is_file()
        )
    else:
        sources = [source]
    excluded = {str(value) for value in values.get("exclude_files", ())}
    sources = [candidate for candidate in sources if candidate.name not in excluded]
    max_recordings = int(values.get("max_recordings", 0))
    if max_recordings > 0:
        sources = sources[:max_recordings]
    if not sources:
        raise ValueError(f"no PIV recordings matched {source} with {file_glob!r}")

    format_id = str(values.get("format", "auto"))
    recordings = []
    for candidate in sources:
        selected_format = format_id
        if selected_format == "auto":
            selected_format = (
                "dense_piv_hdf5"
                if candidate.suffix.lower() in {".h5", ".hdf5"}
                else "shang_tu_mat"
                if candidate.suffix.lower() == ".mat"
                else ""
            )
        if selected_format == "shang_tu_mat":
            recordings.append(
                load_shang_tu_cylinder_piv(
                    candidate,
                    sampling_hz=float(values.get("sampling_hz", 20.0)),
                    spatial_stride=int(values.get("spatial_stride", 1)),
                )
            )
        elif selected_format == "dense_piv_hdf5":
            recordings.append(
                load_dense_piv_hdf5(
                    candidate,
                    sampling_hz=(
                        float(values["sampling_hz"])
                        if values.get("sampling_hz") is not None
                        else None
                    ),
                    spatial_stride=int(values.get("spatial_stride", 1)),
                    u_path=_optional_text(values.get("u_path")),
                    v_path=_optional_text(values.get("v_path")),
                    x_path=_optional_text(values.get("x_path")),
                    y_path=_optional_text(values.get("y_path")),
                    time_path=_optional_text(values.get("time_path")),
                    time_axis=(
                        int(values["time_axis"])
                        if values.get("time_axis") is not None
                        else None
                    ),
                    zero_pair_is_missing=bool(
                        values.get("zero_pair_is_missing", False)
                    ),
                )
            )
        else:
            raise ValueError(
                f"unsupported dense PIV format {selected_format!r} for {candidate}"
            )
    coordinate_rtol = float(values.get("coordinate_rtol", 1.0e-5))
    coordinate_atol = float(values.get("coordinate_atol", 1.0e-7))
    _validate_shared_observation_grid(
        recordings,
        rtol=coordinate_rtol,
        atol=coordinate_atol,
    )
    shared_support = values.get("shared_nonzero_min_fraction")
    if shared_support is not None:
        if bool(values.get("zero_pair_is_missing", False)):
            raise ValueError(
                "use either zero_pair_is_missing or shared_nonzero_min_fraction"
            )
        recordings = list(
            apply_shared_measurement_support(
                recordings,
                minimum_fraction=float(shared_support),
            )
        )
    return tuple(recordings)


def load_dense_piv_hdf5(
    path: str | Path,
    *,
    sampling_hz: float | None = None,
    spatial_stride: int = 1,
    u_path: str | None = None,
    v_path: str | None = None,
    x_path: str | None = None,
    y_path: str | None = None,
    time_path: str | None = None,
    time_axis: int | None = None,
    zero_pair_is_missing: bool = False,
) -> RealPIVRecording:
    """Load a conventional dense PIV HDF5 file without assuming geometry.

    Dataset paths are configurable. When omitted, the loader recognizes the
    common top-level ``u/v/x/y/t`` layout and the RealPDE
    ``measured_data/u``/``measured_data/v`` layout.
    """

    import h5py

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if spatial_stride < 1:
        raise ValueError("spatial_stride must be positive")
    with h5py.File(source, "r") as handle:
        u_raw = _hdf5_array(handle, u_path, ("u", "measured_data/u"), "u")
        v_raw = _hdf5_array(handle, v_path, ("v", "measured_data/v"), "v")
        x_raw = _hdf5_array(handle, x_path, ("x", "measured_data/x"), "x")
        y_raw = _hdf5_array(handle, y_path, ("y", "measured_data/y"), "y")
        raw_time = _hdf5_array(
            handle,
            time_path,
            ("t", "time", "measured_data/t", "measured_data/time"),
            "time",
            required=False,
        )
        metadata = {"format": "dense_piv_hdf5"}
        for name in ("aoa", "re", "reynolds_number"):
            if name in handle and getattr(handle[name], "shape", None) == ():
                metadata[name] = np.asarray(handle[name]).item()

    u_values = np.asarray(u_raw)
    v_values = np.asarray(v_raw)
    if u_values.ndim != 3 or u_values.shape != v_values.shape:
        raise ValueError(f"{source} u and v must have one time and two spatial axes")
    x_grid, y_grid = _coordinate_grids(x_raw, y_raw)
    resolved_axis = _resolve_time_axis(u_values.shape, x_grid.shape, time_axis)
    u_values = np.moveaxis(u_values, resolved_axis, 0)
    v_values = np.moveaxis(v_values, resolved_axis, 0)
    if tuple(u_values.shape[1:]) != tuple(x_grid.shape):
        raise ValueError(f"{source} velocity spatial axes do not match x/y coordinates")

    stride = int(spatial_stride)
    u = np.asarray(u_values[:, ::stride, ::stride], dtype=np.float32)
    v = np.asarray(v_values[:, ::stride, ::stride], dtype=np.float32)
    x = np.asarray(x_grid[::stride, ::stride], dtype=np.float64)
    y = np.asarray(y_grid[::stride, ::stride], dtype=np.float64)
    valid = np.all(np.isfinite(u) & np.isfinite(v), axis=0)
    valid &= np.isfinite(x) & np.isfinite(y)
    if zero_pair_is_missing:
        valid &= np.all((u != 0.0) | (v != 0.0), axis=0)
    if int(np.count_nonzero(valid)) < 16:
        raise ValueError(f"{source} has too few positions measured in every frame")

    frame_count = int(u.shape[0])
    times = None
    if raw_time is not None:
        times = np.asarray(raw_time, dtype=np.float64).reshape(-1)
        if len(times) != frame_count:
            raise ValueError(f"{source} time coordinate does not match its frames")
        if len(times) > 1 and np.any(np.diff(times) <= 0.0):
            raise ValueError(f"{source} time coordinate must be strictly increasing")
    if sampling_hz is None:
        if times is None or len(times) < 2:
            raise ValueError(
                f"{source} needs either a time coordinate or dataset.sampling_hz"
            )
        sampling_hz = 1.0 / float(np.median(np.diff(times)))
    if times is None:
        times = np.arange(frame_count, dtype=np.float64) / float(sampling_hz)

    velocity = np.stack((u[:, valid], v[:, valid]), axis=1)
    return RealPIVRecording(
        velocity=np.asarray(velocity, dtype=np.float32),
        x=np.asarray(x[valid], dtype=np.float64),
        y=np.asarray(y[valid], dtype=np.float64),
        sampling_hz=float(sampling_hz),
        original_grid_shape=tuple(int(value) for value in x_grid.shape),
        spatial_stride=stride,
        time=times,
        recording_id=source.stem,
        metadata=metadata,
    )


def _hdf5_array(handle, explicit, candidates, label, *, required=True):
    paths = (explicit,) if explicit else candidates
    for name in paths:
        if name and name in handle:
            return np.asarray(handle[name])
    if required:
        raise ValueError(
            f"HDF5 PIV file has no {label!r} dataset; tried "
            + ", ".join(repr(value) for value in paths if value)
        )
    return None


def _coordinate_grids(x_values, y_values) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    if x.ndim == 1 and y.ndim == 1:
        x_grid, y_grid = np.meshgrid(x, y, indexing="xy")
        return x_grid, y_grid
    if x.ndim != 2 or x.shape != y.shape:
        raise ValueError("PIV x and y must be matching grids or coordinate vectors")
    return x, y


def _resolve_time_axis(
    velocity_shape: tuple[int, ...],
    spatial_shape: tuple[int, int],
    explicit: int | None,
) -> int:
    if explicit is not None:
        axis = int(explicit) % 3
        remaining = tuple(
            velocity_shape[index] for index in range(3) if index != axis
        )
        if remaining != tuple(spatial_shape):
            raise ValueError("configured time_axis leaves spatial axes unlike x/y")
        return axis
    matches = [
        axis
        for axis in range(3)
        if tuple(velocity_shape[index] for index in range(3) if index != axis)
        == tuple(spatial_shape)
    ]
    if len(matches) != 1:
        raise ValueError(
            "cannot infer the PIV time axis; set dataset.time_axis explicitly"
        )
    return matches[0]


def apply_shared_measurement_support(
    recordings: Sequence[RealPIVRecording],
    *,
    minimum_fraction: float,
    reference_windows: Sequence[tuple[RealPIVRecording, int]] | None = None,
    window_length: int | None = None,
    reference_label: str = "all_recordings",
) -> tuple[RealPIVRecording, ...]:
    """Retain one common measured support learned from selected measurements.

    Some PIV files encode solid or persistently unmeasured pixels as exact
    zero pairs. A shared threshold removes those pixels without requiring a
    geometry-specific mask and keeps every recording on the same point set.
    ``reference_windows`` lets a campaign learn that mask from its
    Generator-training role without inspecting validation or hidden values.
    """

    if not recordings:
        raise ValueError("shared PIV measurement support needs recordings")
    if not 0.0 <= minimum_fraction <= 1.0:
        raise ValueError("shared_nonzero_min_fraction must lie in [0, 1]")
    if reference_windows is None:
        reference_rows = tuple(
            (recording, np.arange(recording.frame_count, dtype=np.int64))
            for recording in recordings
        )
    else:
        if window_length is None or int(window_length) < 1:
            raise ValueError("window_length is required for reference windows")
        frames_by_recording: dict[str, set[int]] = {}
        known = {recording.recording_id for recording in recordings}
        for recording, start in reference_windows:
            if recording.recording_id not in known:
                raise ValueError(
                    "measurement-support reference names an unknown recording"
                )
            stop = int(start) + int(window_length)
            if int(start) < 0 or stop > recording.frame_count:
                raise IndexError(
                    "measurement-support window lies outside the recording"
                )
            frames_by_recording.setdefault(recording.recording_id, set()).update(
                range(int(start), stop)
            )
        reference_rows = tuple(
            (
                recording,
                np.asarray(
                    sorted(frames_by_recording.get(recording.recording_id, ())),
                    dtype=np.int64,
                ),
            )
            for recording in recordings
            if frames_by_recording.get(recording.recording_id)
        )
        if not reference_rows:
            raise ValueError("measurement-support reference windows are empty")
    measured_count = np.zeros(recordings[0].point_count, dtype=np.int64)
    frame_count = 0
    for recording, frame_indices in reference_rows:
        values = np.asarray(recording.velocity)[frame_indices]
        measured_count += np.count_nonzero(
            np.isfinite(values[:, 0])
            & np.isfinite(values[:, 1])
            & ((values[:, 0] != 0.0) | (values[:, 1] != 0.0)),
            axis=0,
        )
        frame_count += len(frame_indices)
    support = measured_count / max(frame_count, 1) >= minimum_fraction
    if int(np.count_nonzero(support)) < 16:
        raise ValueError("shared PIV measurement support retains too few points")

    selected = []
    for recording in recordings:
        metadata = dict(recording.metadata or {})
        metadata["shared_nonzero_min_fraction"] = float(minimum_fraction)
        metadata["shared_measurement_support_source"] = str(reference_label)
        selected.append(
            RealPIVRecording(
                velocity=np.asarray(recording.velocity[:, :, support], dtype=np.float32),
                x=np.asarray(recording.x[support], dtype=np.float64),
                y=np.asarray(recording.y[support], dtype=np.float64),
                sampling_hz=recording.sampling_hz,
                original_grid_shape=recording.original_grid_shape,
                spatial_stride=recording.spatial_stride,
                time=recording.time,
                recording_id=recording.recording_id,
                metadata=metadata,
            )
        )
    return tuple(selected)


def _validate_shared_observation_grid(
    recordings: Sequence[RealPIVRecording],
    *,
    rtol: float,
    atol: float,
) -> None:
    if rtol < 0.0 or atol < 0.0:
        raise ValueError("coordinate_rtol and coordinate_atol cannot be negative")
    first = recordings[0]
    for recording in recordings[1:]:
        if recording.point_count != first.point_count:
            raise ValueError(
                "all recordings in one Generator run must share a point count"
            )
        if not (
            np.allclose(recording.x, first.x, rtol=rtol, atol=atol)
            and np.allclose(recording.y, first.y, rtol=rtol, atol=atol)
        ):
            raise ValueError(
                "all recordings in one Generator run must share coordinates; "
                "run different geometries as separate Projects"
            )


def _optional_text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def window_starts(
    start: int,
    stop: int,
    *,
    window_length: int,
    window_stride: int | None = None,
) -> np.ndarray:
    """Return every complete window at an independently configurable stride."""

    stride = int(window_length if window_stride is None else window_stride)
    if window_length < 1 or stride < 1:
        raise ValueError("window_length and window_stride must be positive")
    if stop <= start:
        raise ValueError("split stop must exceed split start")
    return np.arange(
        int(start),
        int(stop) - int(window_length) + 1,
        stride,
        dtype=np.int64,
    )


def nonoverlapping_window_starts(
    start: int,
    stop: int,
    *,
    window_length: int,
) -> np.ndarray:
    """Backward-compatible shorthand for windows separated by their length."""

    return window_starts(
        start,
        stop,
        window_length=window_length,
        window_stride=window_length,
    )


def collect_windows(
    recording: RealPIVRecording,
    starts: Sequence[int],
    *,
    window_length: int,
) -> np.ndarray:
    """Copy selected windows into [case, time, component, point] order."""

    return np.stack(
        [recording.window(int(start), int(window_length)) for start in starts],
        axis=0,
    ).astype(np.float32, copy=False)


@dataclass(frozen=True)
class TrajectoryPODPrior:
    """Training-only trajectory POD with standard-normal modal coefficients."""

    mean: np.ndarray  # [time, component, point]
    basis: np.ndarray  # [rank, time, component, point]
    component_scale: np.ndarray  # [component]
    explained_variance_fraction: np.ndarray  # [rank]

    @property
    def rank(self) -> int:
        return int(self.basis.shape[0])

    @property
    def window_length(self) -> int:
        return int(self.mean.shape[0])

    @property
    def point_count(self) -> int:
        return int(self.mean.shape[-1])

    def encode(self, velocity: np.ndarray) -> np.ndarray:
        """Map complete velocity windows back to modal coefficients.

        The POD was fitted after balancing the two velocity components.  We
        apply the same balance here, then solve the diagonal modal projection.
        Both one window ``[time,2,point]`` and a batch of windows are accepted.
        """

        values = np.asarray(velocity, dtype=np.float32)
        single = values.ndim == 3
        if single:
            values = values[None, ...]
        if values.ndim != 4 or tuple(values.shape[1:]) != tuple(self.mean.shape):
            raise ValueError(
                "velocity must have shape [case,time,2,point] matching the POD"
            )
        scale = self.component_scale[None, None, :, None]
        difference = (values - self.mean[None, ...]) / scale
        balanced_basis = self.basis / self.component_scale[None, None, :, None]
        flattened_basis = balanced_basis.reshape(self.rank, -1)
        modal_energy = np.sum(np.square(flattened_basis), axis=1)
        coefficients = (
            difference.reshape(len(values), -1) @ flattened_basis.T
        ) / np.maximum(modal_energy[None, :], 1.0e-12)
        return coefficients[0] if single else coefficients

    def with_rank(self, rank: int) -> "TrajectoryPODPrior":
        selected = int(rank)
        if selected < 1 or selected > self.rank:
            raise ValueError(f"rank must lie in [1, {self.rank}]")
        return TrajectoryPODPrior(
            mean=self.mean,
            basis=self.basis[:selected],
            component_scale=self.component_scale,
            explained_variance_fraction=self.explained_variance_fraction[:selected],
        )

    def decode(self, latent: np.ndarray) -> np.ndarray:
        values = np.asarray(latent)
        if values.shape[-1] != self.rank:
            raise ValueError("latent dimension does not match trajectory POD rank")
        return self.mean + np.einsum(
            "...r,rtcf->...tcf",
            values,
            self.basis,
            optimize=True,
        )


def fit_trajectory_pod(
    training_windows: np.ndarray,
    *,
    max_rank: int,
) -> tuple[TrajectoryPODPrior, dict[str, Any]]:
    """Fit a component-balanced trajectory POD by the method of snapshots.

    Each velocity component is divided by its training RMS before fitting so
    the larger streamwise component cannot hide the cross-stream wake motion.
    The returned basis already contains each modal standard deviation.  Thus a
    standard-normal latent vector has the empirical training covariance.
    """

    values = np.asarray(training_windows, dtype=np.float32)
    if values.ndim != 4 or values.shape[2] != 2:
        raise ValueError("training_windows must have shape [case,time,2,point]")
    sample_count = int(values.shape[0])
    rank = min(int(max_rank), sample_count - 1)
    if rank < 1:
        raise ValueError("at least two training windows are required")

    component_scale = np.sqrt(np.mean(np.square(values), axis=(0, 1, 3)))
    component_scale = np.maximum(component_scale, 1.0e-8).astype(np.float32)
    standardized = values / component_scale[None, None, :, None]
    matrix = standardized.reshape(sample_count, -1)
    mean_flat = matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
    centered = np.asarray(matrix - mean_flat[None, :], dtype=np.float32)

    gram = (centered @ centered.T).astype(np.float64)
    gram /= float(sample_count - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    positive = eigenvalues > max(float(eigenvalues[0]) * 1.0e-10, 1.0e-12)
    rank = min(rank, int(np.count_nonzero(positive)))
    if rank < 1:
        raise ValueError("training trajectories have no nonzero POD modes")

    # If C = X^T X/(n-1), then sqrt(lambda) times its unit eigenvector
    # equals X^T u/sqrt(n-1).  Storing this scaled mode whitens the latent.
    scaled_modes = (
        eigenvectors[:, :rank].T.astype(np.float32) @ centered
    ) / np.sqrt(float(sample_count - 1))
    shape = tuple(int(value) for value in values.shape[1:])
    mean = mean_flat.reshape(shape) * component_scale[None, :, None]
    basis = scaled_modes.reshape((rank,) + shape)
    basis = basis * component_scale[None, None, :, None]
    total_variance = max(float(np.sum(eigenvalues)), 1.0e-30)
    cumulative = np.cumsum(eigenvalues[:rank]) / total_variance
    prior = TrajectoryPODPrior(
        mean=np.asarray(mean, dtype=np.float32),
        basis=np.asarray(basis, dtype=np.float32),
        component_scale=component_scale,
        explained_variance_fraction=np.asarray(cumulative, dtype=np.float64),
    )
    report = {
        "training_window_count": sample_count,
        "window_length": int(values.shape[1]),
        "point_count": int(values.shape[-1]),
        "max_rank": rank,
        "component_rms_m_per_s": [float(value) for value in component_scale],
        "explained_variance_fraction": {
            str(candidate): float(cumulative[candidate - 1])
            for candidate in sorted(
                {
                    min(rank, value)
                    for value in (3, 8, 16, 32, 48, 64, 96, rank)
                    if min(rank, value) >= 1
                }
            )
        },
    }
    return prior, report


def save_trajectory_pod(path: str | Path, prior: TrajectoryPODPrior) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        mean=prior.mean,
        basis=prior.basis,
        component_scale=prior.component_scale,
        explained_variance_fraction=prior.explained_variance_fraction,
    )


def load_trajectory_pod(path: str | Path) -> TrajectoryPODPrior:
    with np.load(Path(path), allow_pickle=False) as data:
        return TrajectoryPODPrior(
            mean=np.asarray(data["mean"], dtype=np.float32),
            basis=np.asarray(data["basis"], dtype=np.float32),
            component_scale=np.asarray(data["component_scale"], dtype=np.float32),
            explained_variance_fraction=np.asarray(
                data["explained_variance_fraction"],
                dtype=np.float64,
            ),
        )


@dataclass(frozen=True)
class SparsePIVLayout:
    """Velocity samples visible to every reconstruction method."""

    sensor_indices: np.ndarray
    observed_frames: np.ndarray
    window_length: int
    point_count: int

    @property
    def observation_count(self) -> int:
        return int(len(self.sensor_indices) * len(self.observed_frames) * 2)

    @property
    def visible_fraction(self) -> float:
        return float(self.observation_count) / float(
            self.window_length * self.point_count * 2
        )

    def observe(self, velocity: np.ndarray) -> np.ndarray:
        values = np.asarray(velocity)
        if values.shape[-3:] != (self.window_length, 2, self.point_count):
            raise ValueError("velocity does not match the sparse PIV layout")
        selected = values[..., self.observed_frames, :, :]
        return selected[..., self.sensor_indices]

    def public_mask(self) -> np.ndarray:
        mask = np.zeros((self.window_length, self.point_count), dtype=bool)
        mask[np.ix_(self.observed_frames, self.sensor_indices)] = True
        return mask

@dataclass(frozen=True)
class SparsePIVObservations:
    """Only the velocity values a reconstruction program is allowed to see."""

    values: np.ndarray  # [observed_time, component, sensor]
    layout: SparsePIVLayout
    noise_standard_deviation: np.ndarray  # [component]

    def __post_init__(self) -> None:
        expected = (
            len(self.layout.observed_frames),
            2,
            len(self.layout.sensor_indices),
        )
        if np.asarray(self.values).shape != expected:
            raise ValueError(
                "sparse PIV values must have shape "
                "[observed_time,component,sensor]"
            )
        noise = np.asarray(self.noise_standard_deviation)
        if noise.shape != (2,) or np.any(noise <= 0.0):
            raise ValueError("PIV noise standard deviation must contain two positive values")

    @classmethod
    def from_window(
        cls,
        window: np.ndarray,
        layout: SparsePIVLayout,
        noise_standard_deviation: np.ndarray,
    ) -> "SparsePIVObservations":
        """Create a benchmark input by masking one fully measured PIV window."""

        return cls(
            values=np.asarray(layout.observe(window), dtype=np.float32),
            layout=layout,
            noise_standard_deviation=np.asarray(
                noise_standard_deviation,
                dtype=np.float64,
            ),
        )



def farthest_point_sensors(
    x: np.ndarray,
    y: np.ndarray,
    *,
    sensor_count: int,
) -> np.ndarray:
    """Choose deterministic, well-spread measurement positions."""

    coordinates = np.column_stack((np.asarray(x), np.asarray(y))).astype(np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("x and y must be one-dimensional coordinate arrays")
    count = int(sensor_count)
    if count < 1 or count > coordinates.shape[0]:
        raise ValueError("sensor_count lies outside the available PIV points")
    span = np.ptp(coordinates, axis=0)
    normalized = (coordinates - coordinates.min(axis=0)) / np.maximum(span, 1.0e-12)
    center = normalized.mean(axis=0)
    first = int(np.argmax(np.sum(np.square(normalized - center), axis=1)))
    chosen = [first]
    minimum_distance = np.sum(np.square(normalized - normalized[first]), axis=1)
    minimum_distance[first] = -1.0
    for _ in range(1, count):
        candidate = int(np.argmax(minimum_distance))
        chosen.append(candidate)
        distance = np.sum(np.square(normalized - normalized[candidate]), axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
        minimum_distance[np.asarray(chosen, dtype=np.int64)] = -1.0
    return np.asarray(chosen, dtype=np.int64)


def estimate_piv_noise(
    training_velocity: np.ndarray,
) -> np.ndarray:
    """Estimate component noise from training frames only.

    For locally smooth signals, the median absolute deviation of the temporal
    second difference divided by sqrt(6) estimates independent frame noise.
    Wake dynamics can still contribute, so this is also a conservative model
    discrepancy scale rather than a claim about camera noise alone.
    """

    values = np.asarray(training_velocity, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != 2:
        raise ValueError("training_velocity must have shape [frame,2,point]")
    second_difference = values[2:] - 2.0 * values[1:-1] + values[:-2]
    median = np.median(second_difference, axis=(0, 2), keepdims=True)
    mad = np.median(np.abs(second_difference - median), axis=(0, 2))
    robust_sigma = 1.4826 * mad / np.sqrt(6.0)
    return np.maximum(robust_sigma, 1.0e-8)


def estimate_piv_noise_from_windows(
    reference_windows: Sequence[tuple[RealPIVRecording, int]],
    *,
    window_length: int,
) -> np.ndarray:
    """Estimate PIV noise from the unique frames of one scientific role.

    Overlapping training windows must not count the same camera frame many
    times. Unique frames are collected per recording, split at temporal gaps,
    and only within-segment second differences enter the robust estimate.
    """

    length = int(window_length)
    if length < 3:
        raise ValueError("noise estimation windows need at least three frames")
    recordings: dict[str, RealPIVRecording] = {}
    frames_by_recording: dict[str, set[int]] = {}
    for recording, start in reference_windows:
        stop = int(start) + length
        if int(start) < 0 or stop > recording.frame_count:
            raise IndexError("noise-estimation window lies outside the recording")
        recordings[recording.recording_id] = recording
        frames_by_recording.setdefault(recording.recording_id, set()).update(
            range(int(start), stop)
        )
    component_differences: tuple[list[np.ndarray], list[np.ndarray]] = ([], [])
    for recording_id, frame_values in frames_by_recording.items():
        indices = np.asarray(sorted(frame_values), dtype=np.int64)
        segments = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
        recording = recordings[recording_id]
        for segment in segments:
            if len(segment) < 3:
                continue
            velocity = np.asarray(recording.velocity[segment], dtype=np.float64)
            second_difference = (
                velocity[2:] - 2.0 * velocity[1:-1] + velocity[:-2]
            )
            for component in range(2):
                values = second_difference[:, component].reshape(-1)
                component_differences[component].append(values[np.isfinite(values)])
    noise = []
    for chunks in component_differences:
        if not chunks or not any(len(chunk) for chunk in chunks):
            raise ValueError("noise-estimation role has too few contiguous frames")
        values = np.concatenate(chunks)
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        noise.append(max(float(1.4826 * mad / np.sqrt(6.0)), 1.0e-8))
    return np.asarray(noise, dtype=np.float64)


@dataclass(frozen=True)
class RealPIVEstimate:
    """A complete velocity-window estimate and optional posterior spread."""

    method: str
    velocity: np.ndarray
    velocity_standard_deviation: np.ndarray | None
    seconds: float
    details: dict[str, Any]


def prior_mean_estimate(prior: TrajectoryPODPrior) -> RealPIVEstimate:
    return RealPIVEstimate(
        method="prior_mean",
        velocity=np.asarray(prior.mean),
        velocity_standard_deviation=None,
        seconds=0.0,
        details={},
    )


def nearest_training_estimate(
    training_windows: np.ndarray,
    truth_window: np.ndarray,
    layout: SparsePIVLayout,
    noise_standard_deviation: np.ndarray,
) -> RealPIVEstimate:
    """Return the training trajectory closest on the visible measurements."""

    started = time.perf_counter()
    observed_training = layout.observe(training_windows)
    observed_target = layout.observe(truth_window)
    sigma = np.asarray(noise_standard_deviation)[None, None, :, None]
    distance = np.mean(
        np.square((observed_training - observed_target[None, ...]) / sigma),
        axis=(1, 2, 3),
    )
    selected = int(np.argmin(distance))
    return RealPIVEstimate(
        method="nearest_training_window",
        velocity=np.asarray(training_windows[selected]),
        velocity_standard_deviation=None,
        seconds=time.perf_counter() - started,
        details={
            "selected_training_window": selected,
            "whitened_public_mse": float(distance[selected]),
        },
    )


@dataclass(frozen=True)
class GappyPODPosterior:
    """Exact linear-Gaussian posterior in the trajectory POD coordinates."""

    rank: int
    prior_precision: float = 1.0
    noise_scale: float = 1.0
    method_id: str = "gappy_pod_map"

    def reconstruct(
        self,
        prior: TrajectoryPODPrior,
        truth_window: np.ndarray,
        layout: SparsePIVLayout,
        noise_standard_deviation: np.ndarray,
        *,
        include_uncertainty: bool = True,
    ) -> RealPIVEstimate:
        return self.reconstruct_observations(
            prior,
            SparsePIVObservations.from_window(
                truth_window,
                layout,
                noise_standard_deviation,
            ),
            include_uncertainty=include_uncertainty,
        )

    def reconstruct_observations(
        self,
        prior: TrajectoryPODPrior,
        observations: SparsePIVObservations,
        *,
        include_uncertainty: bool = True,
    ) -> RealPIVEstimate:
        return self.reconstruct_many_observations(
            prior,
            (observations,),
            include_uncertainty=include_uncertainty,
        )[0]

    def reconstruct_many_observations(
        self,
        prior: TrajectoryPODPrior,
        observations: Sequence[SparsePIVObservations],
        *,
        include_uncertainty: bool = True,
    ) -> tuple[RealPIVEstimate, ...]:
        """Reconstruct Cases sharing one sensor layout in one matrix solve.

        The Gappy-POD design matrix depends on the basis, sensor locations and
        noise weights, not on the measured values. Reusing it across Cases is
        mathematically identical to repeated scalar calls and makes validation
        over several rank/regularization candidates much faster.
        """

        if not observations:
            raise ValueError("Gappy POD needs at least one observation Case")
        started = time.perf_counter()
        selected = prior.with_rank(self.rank)
        first = observations[0]
        layout = first.layout
        sigma = (
            np.asarray(first.noise_standard_deviation, dtype=np.float64)
            * float(self.noise_scale)
        )
        for observation in observations[1:]:
            other = observation.layout
            same_layout = (
                other.window_length == layout.window_length
                and other.point_count == layout.point_count
                and np.array_equal(other.sensor_indices, layout.sensor_indices)
                and np.array_equal(other.observed_frames, layout.observed_frames)
            )
            same_noise = np.allclose(
                observation.noise_standard_deviation,
                first.noise_standard_deviation,
            )
            if not same_layout or not same_noise:
                raise ValueError(
                    "batched Gappy POD Cases must share one layout and noise scale"
                )

        mean_observed = layout.observe(selected.mean)
        basis_observed = layout.observe(selected.basis)
        observed_values = np.stack(
            [np.asarray(item.values, dtype=np.float64) for item in observations],
            axis=0,
        )
        targets = (
            (
                observed_values
                - mean_observed[None, ...]
            )
            / sigma[None, None, :, None]
        ).reshape(len(observations), -1)
        design = (
            basis_observed / sigma[None, None, :, None]
        ).reshape(selected.rank, -1).T
        normal = design.T @ design
        normal += float(self.prior_precision) * np.eye(selected.rank)
        covariance = _symmetric_inverse(normal)
        latent = (targets @ design) @ covariance
        velocity = selected.decode(latent)
        standard_deviation: np.ndarray | None = None
        if include_uncertainty:
            covariance_root = _covariance_root(covariance)
            transformed_basis = np.einsum(
                "rtcf,rs->stcf",
                selected.basis,
                covariance_root,
                optimize=True,
            )
            standard_deviation = np.sqrt(
                np.sum(np.square(transformed_basis), axis=0)
            ).astype(np.float32)
        seconds_per_case = (time.perf_counter() - started) / len(observations)
        condition = _condition_number(normal)
        return tuple(
            RealPIVEstimate(
                method=self.method_id,
                velocity=np.asarray(velocity[index], dtype=np.float32),
                velocity_standard_deviation=standard_deviation,
                seconds=seconds_per_case,
                details={
                    "rank": selected.rank,
                    "prior_precision": float(self.prior_precision),
                    "noise_scale": float(self.noise_scale),
                    "latent_norm": float(np.linalg.norm(latent[index])),
                    "design_condition": condition,
                },
            )
            for index in range(len(observations))
        )


@dataclass(frozen=True)
class TrajectoryESMDA:
    """ES-MDA applied to the trajectory POD coefficients.

    Source: Emerick and Reynolds, Computers and Geosciences 55 (2013), 3-15,
    https://doi.org/10.1016/j.cageo.2012.03.011
    """

    rank: int
    ensemble_size: int = 64
    assimilation_steps: int = 4
    noise_scale: float = 1.0
    seed: int = 0
    latent_clip: float = 10.0
    method_id: str = "es_mda"

    def reconstruct(
        self,
        prior: TrajectoryPODPrior,
        truth_window: np.ndarray,
        layout: SparsePIVLayout,
        noise_standard_deviation: np.ndarray,
        *,
        case_seed_offset: int = 0,
    ) -> RealPIVEstimate:
        if self.ensemble_size < 3:
            raise ValueError("ES-MDA needs at least three ensemble members")
        if self.assimilation_steps < 1:
            raise ValueError("assimilation_steps must be positive")
        started = time.perf_counter()
        selected = prior.with_rank(self.rank)
        rng = np.random.default_rng(int(self.seed) + int(case_seed_offset))
        latent = rng.normal(size=(self.ensemble_size, selected.rank))
        sigma = (
            np.asarray(noise_standard_deviation, dtype=np.float64)
            * float(self.noise_scale)
        )
        mean_observed = layout.observe(selected.mean)
        basis_observed = layout.observe(selected.basis)
        observed = (
            (layout.observe(truth_window) - mean_observed)
            / sigma[None, :, None]
        ).reshape(-1)
        design = (
            basis_observed / sigma[None, None, :, None]
        ).reshape(selected.rank, -1).T
        alpha = float(self.assimilation_steps)
        residual_history: list[float] = []
        identity = np.eye(self.ensemble_size)
        for _step in range(self.assimilation_steps):
            predicted = latent @ design.T
            residual_history.append(
                float(np.sqrt(np.mean(np.square(predicted.mean(axis=0) - observed))))
            )
            latent_anomaly = latent - latent.mean(axis=0, keepdims=True)
            predicted_anomaly = predicted - predicted.mean(axis=0, keepdims=True)
            perturbed = observed[None, :] + np.sqrt(alpha) * rng.normal(
                size=predicted.shape
            )
            innovation = perturbed - predicted
            system = predicted_anomaly @ predicted_anomaly.T
            system += float(self.ensemble_size - 1) * alpha * identity
            weights = _symmetric_solve(
                system,
                predicted_anomaly @ innovation.T,
            )
            latent += (latent_anomaly.T @ weights).T
            np.clip(
                latent,
                -float(self.latent_clip),
                float(self.latent_clip),
                out=latent,
            )

        decoded = selected.decode(latent.astype(np.float32))
        velocity = decoded.mean(axis=0)
        standard_deviation = decoded.std(axis=0, ddof=1)
        final_public = (
            (layout.observe(velocity) - mean_observed)
            / sigma[None, :, None]
        ).reshape(-1)
        residual_history.append(
            float(np.sqrt(np.mean(np.square(final_public - observed))))
        )
        return RealPIVEstimate(
            method=self.method_id,
            velocity=np.asarray(velocity, dtype=np.float32),
            velocity_standard_deviation=np.asarray(
                standard_deviation,
                dtype=np.float32,
            ),
            seconds=time.perf_counter() - started,
            details={
                "rank": selected.rank,
                "ensemble_size": int(self.ensemble_size),
                "assimilation_steps": int(self.assimilation_steps),
                "noise_scale": float(self.noise_scale),
                "residual_history": residual_history,
                "posterior_latent_mean_norm": float(
                    np.linalg.norm(latent.mean(axis=0))
                ),
                "posterior_latent_std_mean": float(
                    latent.std(axis=0, ddof=1).mean()
                ),
            },
        )


def real_piv_metrics(
    truth: np.ndarray,
    estimate: RealPIVEstimate,
    layout: SparsePIVLayout,
    noise_standard_deviation: np.ndarray,
) -> dict[str, float]:
    """Score visible fit, hidden measurements, dynamics and uncertainty."""

    expected = (layout.window_length, 2, layout.point_count)
    truth_values = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(estimate.velocity, dtype=np.float64)
    if truth_values.shape != expected or prediction.shape != expected:
        raise ValueError("truth and estimate must match the sparse PIV window")
    public = layout.public_mask()
    hidden = ~public
    sensor_mask = np.zeros(layout.point_count, dtype=bool)
    sensor_mask[layout.sensor_indices] = True
    unobserved_frames = np.ones(layout.window_length, dtype=bool)
    unobserved_frames[layout.observed_frames] = False

    error = prediction - truth_values
    hidden_error = np.concatenate(
        [error[:, component, :][hidden] for component in range(2)]
    )
    hidden_truth = np.concatenate(
        [truth_values[:, component, :][hidden] for component in range(2)]
    )
    public_error = np.concatenate(
        [error[:, component, :][public] for component in range(2)]
    )
    public_truth = np.concatenate(
        [truth_values[:, component, :][public] for component in range(2)]
    )
    metrics = {
        "hidden_velocity_relative_l2": _relative_l2(
            hidden_error,
            hidden_truth,
        ),
        "hidden_u_relative_l2": _relative_l2(
            error[:, 0, :][hidden],
            truth_values[:, 0, :][hidden],
        ),
        "hidden_v_relative_l2": _relative_l2(
            error[:, 1, :][hidden],
            truth_values[:, 1, :][hidden],
        ),
        "spatial_holdout_relative_l2": _relative_l2(
            error[:, :, ~sensor_mask].reshape(-1),
            truth_values[:, :, ~sensor_mask].reshape(-1),
        ),
        "temporal_holdout_relative_l2": _relative_l2(
            error[unobserved_frames].reshape(-1),
            truth_values[unobserved_frames].reshape(-1),
        ),
        "public_velocity_relative_l2": _relative_l2(
            public_error,
            public_truth,
        ),
        "kinetic_energy_relative_l2": _kinetic_energy_error(
            prediction,
            truth_values,
        ),
        "seconds": float(estimate.seconds),
    }
    observed_error = layout.observe(error)
    sigma = np.asarray(noise_standard_deviation)[None, :, None]
    metrics["public_whitened_rmse"] = float(
        np.sqrt(np.mean(np.square(observed_error / sigma)))
    )
    standard_deviation = estimate.velocity_standard_deviation
    if standard_deviation is not None:
        spread = np.maximum(np.asarray(standard_deviation, dtype=np.float64), 1.0e-12)
        hidden_spread = np.concatenate(
            [spread[:, component, :][hidden] for component in range(2)]
        )
        metrics["hidden_95pct_coverage"] = float(
            np.mean(np.abs(hidden_error) <= 1.959964 * hidden_spread)
        )
        metrics["hidden_standardized_rmse"] = float(
            np.sqrt(np.mean(np.square(hidden_error / hidden_spread)))
        )
        metrics["hidden_uncertainty_rms"] = float(
            np.sqrt(np.mean(np.square(hidden_spread)))
        )
    return metrics


def summarize_case_metrics(rows: Sequence[dict[str, float]]) -> dict[str, Any]:
    """Return medians and 10--90 percent intervals over complete test windows."""

    if not rows:
        raise ValueError("cannot summarize an empty metric table")
    names = sorted(set.intersection(*(set(row) for row in rows)))
    summary: dict[str, Any] = {}
    for name in names:
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        summary[name] = {
            "median": float(np.median(values)),
            "mean": float(np.mean(values)),
            "q10": float(np.quantile(values, 0.1)),
            "q90": float(np.quantile(values, 0.9)),
        }
    return summary


def dominant_frequency(
    signal: np.ndarray,
    *,
    sampling_hz: float,
    minimum_hz: float = 0.2,
    maximum_hz: float = 2.0,
) -> float:
    """Dominant Welch-spectrum frequency inside the wake-shedding band."""

    from scipy.signal import welch

    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    values = values - values.mean()
    nperseg = min(512, values.size)
    frequencies, power = welch(
        values,
        fs=float(sampling_hz),
        nperseg=nperseg,
        noverlap=min(nperseg // 2, nperseg - 1),
    )
    band = (frequencies >= float(minimum_hz)) & (
        frequencies <= float(maximum_hz)
    )
    if not np.any(band):
        return float("nan")
    return float(frequencies[band][np.argmax(power[band])])


def choose_frequency_probe(
    truth_windows: np.ndarray,
    layout: SparsePIVLayout,
) -> int:
    """Select a hidden point with strong cross-stream fluctuation."""

    values = np.asarray(truth_windows)
    variance = np.var(values[:, :, 1, :], axis=(0, 1))
    variance[np.asarray(layout.sensor_indices, dtype=np.int64)] = -1.0
    return int(np.argmax(variance))


def _relative_l2(error: np.ndarray, truth: np.ndarray) -> float:
    numerator = float(np.linalg.norm(np.asarray(error, dtype=np.float64)))
    denominator = float(np.linalg.norm(np.asarray(truth, dtype=np.float64)))
    return numerator / max(denominator, 1.0e-12)


def _kinetic_energy_error(prediction: np.ndarray, truth: np.ndarray) -> float:
    predicted_energy = 0.5 * np.mean(np.square(prediction), axis=-1).sum(axis=1)
    truth_energy = 0.5 * np.mean(np.square(truth), axis=-1).sum(axis=1)
    return _relative_l2(predicted_energy - truth_energy, truth_energy)


def _symmetric_solve(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    target = np.asarray(rhs, dtype=np.float64)
    try:
        return np.linalg.solve(values, target)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(values, target, rcond=1.0e-10)[0]


def _symmetric_inverse(matrix: np.ndarray) -> np.ndarray:
    identity = np.eye(int(np.asarray(matrix).shape[0]))
    return _symmetric_solve(matrix, identity)


def _covariance_root(covariance: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(
        0.5 * (np.asarray(covariance) + np.asarray(covariance).T)
    )
    values = np.maximum(values, 0.0)
    return vectors @ np.diag(np.sqrt(values))


def _condition_number(matrix: np.ndarray) -> float:
    try:
        return float(np.linalg.cond(np.asarray(matrix, dtype=np.float64)))
    except np.linalg.LinAlgError:
        return float("inf")
