"""Bounded raw-array reuse retains original NS queries and file freshness."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Lock
import os

import numpy as np
import pytest

from pde_agent_verifier_solver.plugins.ns import datasets
from pde_agent_verifier_solver.plugins._regular_grid import answer_from_regular_grid
from pde_agent_verifier_solver.sdk import AnswerQuery


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch):
    monkeypatch.setattr(datasets, "_NS_TRUTH_ARRAY_CACHE", OrderedDict())
    monkeypatch.setattr(datasets, "_NS_TRUTH_CACHE_BYTES", 0)
    monkeypatch.setattr(datasets, "_NS_TRUTH_CACHE_LOCK", Lock())


def make_case(root, name="case", offset=0):
    directory = root / name
    directory.mkdir(exist_ok=True)
    axes = {"t": np.linspace(0., .12, 3), "x": np.linspace(0., 1., 4), "y": np.linspace(0., 1., 5)}
    velocity = np.arange(3 * 2 * 4 * 5, dtype=np.float32).reshape(3, 2, 4, 5) / 7 + offset
    pressure = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5) / 9 - offset
    np.savez_compressed(directory / "hidden_truth_cache.npz", u_true=velocity,
                        pressure=pressure, t=axes["t"], unused=np.ones(10))
    np.savez_compressed(directory / "public_arrays.npz", x=axes["x"], y=axes["y"],
                        forcing=np.ones((3, 2, 4, 5)))
    return directory, axes, {"velocity_x": velocity[:, 0], "velocity_y": velocity[:, 1], "pressure": pressure}


def points(fields=("velocity_x", "velocity_y"), time=.04):
    return AnswerQuery.at_points(fields=fields, points=(
        {"t": time, "x": .23, "y": .67}, {"t": .12, "x": 1., "y": 0.},
    ))


def reference(query, axes, fields):
    # Unchanged reference helper with the complete old NSFieldTruth inputs.
    return answer_from_regular_grid(query, axis_order=("t", "x", "y"),
                                    native_axes=axes, fields=fields)


@pytest.mark.parametrize("fields", (("velocity_x", "velocity_y"), ("pressure",),
                                    ("velocity_x", "velocity_y", "pressure")))
@pytest.mark.parametrize("kind", ("points", "grid"))
def test_cached_views_are_exact_for_all_supported_queries(tmp_path, fields, kind):
    directory, axes, arrays = make_case(tmp_path)
    query = points(fields) if kind == "points" else AnswerQuery.grid(fields=fields, t=4, x=3, y=6)
    truth = datasets.NSFieldTruth(directory)
    assert truth.answer(query) == reference(query, axes, arrays)
    assert truth.answer(query) == reference(query, axes, arrays)
    assert all(not value.flags.writeable for value in datasets._NS_TRUTH_ARRAY_CACHE.values())


def test_new_query_reuses_raw_data_and_loads_pressure_only_when_requested(tmp_path, monkeypatch):
    directory, axes, arrays = make_case(tmp_path)
    original_load, original_getitem = np.load, np.lib.npyio.NpzFile.__getitem__
    loads, members = [], []

    def load(path, *args, **kwargs):
        loads.append(Path(path).name)
        return original_load(path, *args, **kwargs)

    def getitem(self, name):
        members.append(name)
        return original_getitem(self, name)

    monkeypatch.setattr(np, "load", load)
    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", getitem)
    truth = datasets.NSFieldTruth(directory)
    first = truth.answer(points(time=.02))
    assert len(loads) == 2 and "pressure" not in members and "forcing" not in members
    second = truth.answer(points(time=.09))
    assert len(loads) == 2 and first != second
    assert second == reference(points(time=.09), axes, arrays)
    pressure_query = points(("pressure",))
    assert truth.answer(pressure_query) == reference(pressure_query, axes, arrays)
    assert loads == ["hidden_truth_cache.npz", "public_arrays.npz", "hidden_truth_cache.npz"]
    assert members.count("u_true") == members.count("pressure") == 1
    assert members.count("x") == members.count("y") == members.count("t") == 1


def test_pressure_only_does_not_decompress_velocity(tmp_path, monkeypatch):
    directory, axes, arrays = make_case(tmp_path)
    original = np.lib.npyio.NpzFile.__getitem__

    def getitem(self, name):
        assert name != "u_true"
        return original(self, name)

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", getitem)
    query = points(("pressure",))
    assert datasets.NSFieldTruth(directory).answer(query) == reference(query, axes, arrays)


@pytest.mark.parametrize("filename", ("hidden_truth_cache.npz", "public_arrays.npz"))
def test_changed_file_replaces_cached_arrays(tmp_path, filename):
    directory, axes, arrays = make_case(tmp_path)
    truth = datasets.NSFieldTruth(directory)
    query = points()
    before = truth.answer(query)
    path = directory / filename
    stat = path.stat()
    if filename == "hidden_truth_cache.npz":
        velocity = np.stack((arrays["velocity_x"], arrays["velocity_y"]), axis=1) + 100
        arrays["velocity_x"], arrays["velocity_y"] = velocity[:, 0], velocity[:, 1]
        np.savez_compressed(path, u_true=velocity, pressure=arrays["pressure"], t=axes["t"])
    else:
        axes["x"] = axes["x"] * 1.1
        np.savez_compressed(path, x=axes["x"], y=axes["y"])
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    after = truth.answer(query)
    assert after != before
    assert after == reference(query, axes, arrays)


@pytest.mark.parametrize("filename", ("hidden_truth_cache.npz", "public_arrays.npz"))
def test_deleted_file_cannot_return_stale_cached_truth(tmp_path, filename):
    directory, _, _ = make_case(tmp_path)
    truth = datasets.NSFieldTruth(directory)
    truth.answer(points())
    (directory / filename).unlink()
    with pytest.raises(FileNotFoundError):
        truth.answer(points())


def test_cache_evicts_by_array_bytes_and_oversized_fields_are_not_retained(tmp_path, monkeypatch):
    first, axes, arrays = make_case(tmp_path, "first")
    second, _, _ = make_case(tmp_path, "second", offset=1)
    # One 480-byte velocity plus 96-byte axes fits; two trajectories do not.
    monkeypatch.setattr(datasets, "_NS_TRUTH_CACHE_MAX_BYTES", 600)
    loads = []
    original = np.load

    def load(path, *args, **kwargs):
        loads.append(Path(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", load)
    first_truth, second_truth = datasets.NSFieldTruth(first), datasets.NSFieldTruth(second)
    first_truth.answer(points())
    second_truth.answer(points())
    assert datasets._NS_TRUTH_CACHE_BYTES <= 600
    assert datasets._NS_TRUTH_CACHE_BYTES == sum(array.nbytes for array in datasets._NS_TRUTH_ARRAY_CACHE.values())
    assert all(key[0][0] == str(second) for key in datasets._NS_TRUTH_ARRAY_CACHE)
    previous = len(loads)
    second_truth.answer(points(time=.08))
    assert len(loads) == previous
    assert first_truth.answer(points()) == reference(points(), axes, arrays)
    assert len(loads) > previous and datasets._NS_TRUTH_CACHE_BYTES <= 600
    monkeypatch.setattr(datasets, "_NS_TRUTH_CACHE_MAX_BYTES", 64)
    # A newly read field larger than the budget stays invocation-local.
    third, _, _ = make_case(tmp_path, "third")
    datasets.NSFieldTruth(third).answer(points())
    assert datasets._NS_TRUTH_CACHE_BYTES <= 64
    assert not any(key[0][0] == str(third) and key[1] == "u_true" for key in datasets._NS_TRUTH_ARRAY_CACHE)


@pytest.mark.parametrize("same_case", (False, True))
def test_parallel_loads_do_not_hold_global_lock_or_double_count_cache(tmp_path, monkeypatch, same_case):
    first, axes, fields = make_case(tmp_path, "first")
    second, other_axes, other_fields = (first, axes, fields) if same_case else make_case(tmp_path, "second", offset=2)
    barrier = Barrier(2)
    original = np.load

    def load(path, *args, **kwargs):
        if Path(path).name == "hidden_truth_cache.npz":
            # Both loaders must reach disk concurrently, even for one Case.
            barrier.wait(timeout=5)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", load)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(datasets.NSFieldTruth(path).answer, points()) for path in (first, second)]
        outputs = [future.result(timeout=10) for future in futures]
    assert outputs == [reference(points(), axes, fields), reference(points(), other_axes, other_fields)]
    assert datasets._NS_TRUTH_CACHE_BYTES == sum(array.nbytes for array in datasets._NS_TRUTH_ARRAY_CACHE.values())
    assert len(datasets._NS_TRUTH_ARRAY_CACHE) == (4 if same_case else 8)
