"""Gappy-POD S0 using the Project-owned frozen trajectory representation.

The large POD basis is deliberately not part of this Candidate bundle. Every
PIV Case carries a content-identified ``project_reconstruction`` descriptor;
the Candidate is free to read that public artifact and implement any
reconstruction algorithm without vendoring a new 152 MiB copy per edit,
evaluation, or nomination snapshot.
"""

from __future__ import annotations

import hashlib
import json
from math import isfinite
import os
from pathlib import Path
import stat
from typing import Mapping

import numpy as np

from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    AnswerView,
    Case,
    QueryKind,
)


_VELOCITY_FIELDS = ("velocity_x", "velocity_y")
_ARTIFACT_SCHEMA = "pde_project_reconstruction_artifact_v1"
_MODEL_CACHE = {}


class _GappyPODVelocity:
    """Prepared full velocity that answers only subsequently revealed points."""

    def __init__(self, *, velocity, x, y, time) -> None:
        self._velocity = np.asarray(velocity, dtype=np.float32)
        self._x = np.asarray(x, dtype=np.float64)
        self._y = np.asarray(y, dtype=np.float64)
        self._time = np.asarray(time, dtype=np.float64)

    def evaluate(self, query: AnswerQuery) -> AnswerView:
        if not isinstance(query, AnswerQuery):
            raise TypeError("PIV Gappy-POD needs an AnswerQuery")
        if query.kind is not QueryKind.POINTS:
            raise ValueError("PIV Gappy-POD formal evaluation supports POINTS only")
        if set(query.fields) != set(_VELOCITY_FIELDS):
            raise ValueError("PIV Gappy-POD needs both velocity components")

        fields = {name: [] for name in query.fields}
        for point in query.points:
            if not isinstance(point, Mapping) or {"t", "x", "y"} - set(point):
                raise ValueError("PIV point query needs finite t, x, and y")
            coordinates = tuple(float(point[name]) for name in ("t", "x", "y"))
            if not all(isfinite(value) for value in coordinates):
                raise ValueError("PIV point query coordinates must be finite")
            time_index = int(np.argmin(np.abs(self._time - coordinates[0])))
            distances = np.square(self._x - coordinates[1]) + np.square(
                self._y - coordinates[2]
            )
            point_index = int(np.argmin(distances))
            for name in query.fields:
                component = 0 if name == "velocity_x" else 1
                fields[name].append(
                    float(self._velocity[time_index, component, point_index])
                )
        return AnswerView(
            kind=QueryKind.POINTS,
            fields={name: tuple(values) for name, values in fields.items()},
            points=query.points,
            shape=(len(query.points),),
        )


def prepare_solution(
    case: Case,
    solve_budget: float | None = None,
) -> _GappyPODVelocity:
    """Reconstruct a complete velocity before private points are revealed."""

    if not isinstance(case, Case):
        raise TypeError("PIV Gappy-POD needs a Case")
    _validate_budget(solve_budget)
    public = case.public_data
    coordinates = _mapping(public.get("coordinates"), "coordinates")
    observations = _mapping(public.get("observations"), "observations")
    artifact = _project_artifact(public.get("project_reconstruction"))
    x = _finite_vector(coordinates.get("x"), "coordinates.x")
    y = _finite_vector(coordinates.get("y"), "coordinates.y")
    time = _finite_vector(coordinates.get("time"), "coordinates.time")
    if len(x) != len(y):
        raise ValueError("PIV x/y coordinates must be paired")
    if len(time) > 1 and not np.all(np.diff(time) > 0.0):
        raise ValueError("PIV time coordinates must be strictly increasing")

    values = np.asarray(observations.get("values"), dtype=np.float64)
    sensors = _index_vector(observations.get("sensor_indices"), "sensor_indices")
    frames = _index_vector(observations.get("observed_frames"), "observed_frames")
    noise = _finite_vector(
        observations.get("noise_standard_deviation"),
        "noise_standard_deviation",
    )
    if values.shape != (len(frames), 2, len(sensors)):
        raise ValueError("public PIV observation values have an unexpected shape")
    if len(noise) != 2 or np.any(noise <= 0.0):
        raise ValueError("PIV needs one positive noise scale per velocity component")

    method = _load_method(
        Path(__file__).resolve().parent / "assets/reconstruction_settings.json"
    )
    mean, basis = _load_model(artifact, requested_rank=int(method["rank"]))
    if mean.shape != (len(time), 2, len(x)):
        raise ValueError("trajectory POD mean does not match the public PIV geometry")
    if basis.ndim != 4 or basis.shape[1:] != mean.shape:
        raise ValueError("trajectory POD basis does not match its mean")
    if np.any(sensors >= len(x)) or np.any(frames >= len(time)):
        raise ValueError("public PIV observation index leaves the native window")

    rank = len(basis)
    if rank < 1:
        raise ValueError("PIV Gappy-POD rank must be positive")
    observed_mean = mean[np.ix_(frames, (0, 1), sensors)]
    observed_basis = basis[np.ix_(np.arange(rank), frames, (0, 1), sensors)]
    design = observed_basis.reshape(rank, -1).T.astype(np.float64, copy=False)
    target = (values - observed_mean).reshape(-1)
    observation_scale = np.broadcast_to(
        noise[None, :, None], values.shape
    ).reshape(-1)
    weighted_design = design / observation_scale[:, None]
    weighted_target = target / observation_scale
    normal = weighted_design.T @ weighted_design
    normal.flat[:: rank + 1] += float(method["prior_precision"])
    coefficients = np.linalg.solve(normal, weighted_design.T @ weighted_target)
    velocity = mean + np.einsum(
        "r,rtcp->tcp",
        coefficients.astype(np.float32),
        basis,
        optimize=True,
    )
    if not np.all(np.isfinite(velocity)):
        raise ValueError("PIV Gappy-POD produced a non-finite velocity")
    return _GappyPODVelocity(velocity=velocity, x=x, y=y, time=time)


def _project_artifact(value) -> Mapping[str, object]:
    artifact = _mapping(value, "project_reconstruction")
    required = {
        "schema",
        "artifact_id",
        "artifact_kind",
        "format",
        "locator",
        "sha256",
        "size_bytes",
        "selected_method",
    }
    if set(artifact) != required:
        raise ValueError("PIV Project reconstruction descriptor is malformed")
    if artifact["schema"] != _ARTIFACT_SCHEMA:
        raise ValueError("PIV Project reconstruction schema is unsupported")
    if artifact["artifact_kind"] != "trajectory_pod":
        raise ValueError("PIV Project reconstruction artifact has the wrong kind")
    if artifact["format"] != "numpy_npz":
        raise ValueError("PIV Project reconstruction format is unsupported")
    digest = artifact["sha256"]
    size = artifact["size_bytes"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 1
    ):
        raise ValueError("PIV Project reconstruction identity is invalid")
    return artifact


def _load_model(artifact: Mapping[str, object], *, requested_rank: int):
    locator = artifact["locator"]
    if not isinstance(locator, str) or not locator:
        raise ValueError("PIV Project reconstruction locator is invalid")
    path = Path(locator)
    if not path.is_absolute() or path.is_symlink() or path.resolve() != path:
        raise ValueError("PIV Project reconstruction locator must be canonical")
    key = (locator, artifact["sha256"], artifact["size_bytes"], requested_rank)
    cached = _MODEL_CACHE.get(key)
    if cached is None:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("PIV Project reconstruction must be a regular file")
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            opened_identity = (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
                opened.st_mtime_ns,
            )
            if opened_identity != before_identity:
                raise ValueError("PIV Project reconstruction identity changed")
            if opened.st_size != artifact["size_bytes"]:
                raise ValueError("PIV Project reconstruction size changed")
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                digest = hashlib.sha256()
                bytes_read = 0
                while block := handle.read(1024 * 1024):
                    digest.update(block)
                    bytes_read += len(block)
                if bytes_read != opened.st_size:
                    raise ValueError("PIV Project reconstruction changed while reading")
                if digest.hexdigest() != artifact["sha256"]:
                    raise ValueError("PIV Project reconstruction digest changed")
                handle.seek(0)
                with np.load(handle, allow_pickle=False) as model:
                    available_basis = np.asarray(model["basis"], dtype=np.float32)
                    rank = min(requested_rank, len(available_basis))
                    cached = (
                        np.array(model["mean"], dtype=np.float32, copy=True),
                        np.array(available_basis[:rank], dtype=np.float32, copy=True),
                    )
            after = os.fstat(descriptor)
            current = path.lstat()
            after_identity = (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            )
            current_identity = (
                current.st_dev,
                current.st_ino,
                current.st_size,
                current.st_mtime_ns,
            )
            if after_identity != opened_identity or current_identity != opened_identity:
                raise ValueError("PIV Project reconstruction changed while loading")
        finally:
            os.close(descriptor)
        _MODEL_CACHE[key] = cached
    return cached


def _load_method(path: Path) -> Mapping[str, float | int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    method = _mapping(payload, "reconstruction settings")
    rank = method.get("rank")
    precision = method.get("prior_precision")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError("PIV reconstruction rank must be a positive integer")
    if (
        isinstance(precision, bool)
        or not isinstance(precision, (int, float))
        or not isfinite(float(precision))
        or float(precision) < 0.0
    ):
        raise ValueError("PIV prior_precision must be finite and nonnegative")
    return {"rank": rank, "prior_precision": float(precision)}


def _mapping(value, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"PIV public Case needs {name}")
    return value


def _finite_vector(value, name: str):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or len(array) < 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"PIV {name} must be a nonempty finite vector")
    return array


def _index_vector(value, name: str):
    array = np.asarray(value)
    if (
        array.ndim != 1
        or len(array) < 1
        or array.dtype.kind not in "iu"
        or np.any(array < 0)
        or len(np.unique(array)) != len(array)
    ):
        raise ValueError(f"PIV {name} must contain unique nonnegative integers")
    return array.astype(np.int64, copy=False)


def _validate_budget(value: float | None) -> None:
    if value is not None and (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError("candidate solve budget must be finite and positive")


__all__ = ["prepare_solution"]
