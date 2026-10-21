"""Project preparation uses public PIV observations and sealed validation truth."""
from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from pde_agent_verifier_solver.app import ProductApplication, ProgramBundle
from pde_agent_verifier_solver.plugins.ns.experimental_mms import (
    ExperimentalMMSDatasetRegistry,
)
from pde_agent_verifier_solver.plugins.ns.experimental_mms_project import (
    ContinuousPIVReconstruction, ExperimentalMMSProblem,
    save_continuous_mms_dataset, save_prepared_mms_dataset, project_reconstructed_output,
)
from pde_agent_verifier_solver.plugins.ns.tasks import NSSolverTask
from pde_agent_verifier_solver.plugins.ns.generator import NSStructuredGenerator
from pde_agent_verifier_solver.plugins.ns.completed_fields import field_to_state, numerical_symbols
from pde_agent_verifier_solver.sdk import DatasetInput, DatasetKind, ReconstructionRecipe


_METHOD = {"rank": 3, "prior_precision": 1e-5, "noise_scale": 1.0}
_PROJECTION = {"spatial_mode_x": 2, "spatial_mode_y": 2, "time_order": 2,
               "ridge_relative": 1e-8, "spatial_extension": .25}


def _fixture(tmp_path):
    x, y = np.meshgrid(np.linspace(-0.7, 1.2, 5), np.linspace(-0.4, 0.8, 5), indexing="ij")
    x, y = x.ravel(), y.ravel()
    time = np.linspace(0, 0.3, 4)
    sensors = np.array([0, 4, 12, 20, 24])
    frames = np.array([0, 3])
    roles = {"generator_training": 4, "generator_validation": 1, "development": 1,
             "verifier": 1, "final_test": 1, "reconstruction_validation": 1}
    manifest = {
        "schema": "pde_completed_case_dataset_v1", "problem": "ns",
        "geometry": {"label": "test measurement domain", "obstacle_polygons": []},
        "observation_layout": {"noise_standard_deviation": [0.002, 0.003]}, "splits": {},
    }
    serial = 0
    reference = None
    for role, count in roles.items():
        rows = []
        for index in range(count):
            # Independent polynomial parameters span calibration POD modes.
            a, b, c = 0.6 + 0.12 * serial, -0.3 + 0.03 * serial**2, 0.02 * serial**3
            factor = a + b * time[:, None]
            velocity = np.stack((factor * (x**2 + 0.4 * x * y) + c,
                                 -factor * (2 * x * y + 0.2 * y**2)), axis=1)
            case_id = f"{role}_case_{index}"
            path = tmp_path / f"{case_id}.npz"
            public = velocity[np.ix_(frames, np.arange(2), sensors)]
            # Outside calibration, the completed array must not become method input.
            stored = velocity if role == "generator_training" else np.full_like(velocity, 9.9e6)
            np.savez(path, velocity=stored, x=x, y=y, time=time, public_values=public,
                     sensor_indices=sensors, observed_frames=frames)
            row = {"case_id": case_id, "case_file": path.name,
                   "generator_target_kind": "measured_full_field" if role == "generator_training" else "reconstruction_mean",
                   "source_recording": "recording_a", "source_frames": [10 * serial, 10 * serial + 4]}
            if role == "reconstruction_validation":
                reference = tmp_path / "sealed_validation_reference.npz"
                np.savez(reference, velocity=velocity, x=x, y=y, time=time)
                row["measurement_reference_file"] = reference.name
            rows.append(row)
            serial += 1
        manifest["splits"][role] = rows
    path = tmp_path / "completed_manifest.json"
    path.write_text(json.dumps(manifest))
    dataset = DatasetInput("experimental-test", DatasetKind.EXPERIMENTAL, str(path),
                           settings={"viscosity": 0.02, "coordinate_units": "nondimensional",
                                     "mms_geometry": {"label": "declared PIV window", "obstacle_polygons": [],
                                                      "outer_boundary": "manufactured_trace"},
                                     "projection_config": dict(_PROJECTION)})
    return dataset, manifest, reference


def _reconstruct(dataset, problem=None):
    problem = problem or ExperimentalMMSProblem()
    source = problem.prepare_reconstruction_source(dataset, {}, 0)
    output = ContinuousPIVReconstruction().reconstruct(source.program_input, _METHOD, 0)
    return problem, source, output


def _completed_velocities(output):
    data = output.completed_data
    result = {row["case_id"]: row["velocity"].copy()
              for rows in data["split_fields"].values() for row in rows}
    result.update({key: row["velocity"].copy() for key, row in data["validation"].items()})
    return result


def test_project_reconstruction_produces_original_solver_cases(tmp_path):
    dataset, _, _ = _fixture(tmp_path)
    backend = ExperimentalMMSDatasetRegistry()
    problem = ExperimentalMMSProblem(backend)
    programs = ProgramBundle(problem, NSSolverTask(), NSStructuredGenerator(), {})
    project = ProductApplication.local_development().create_project(
        "PIV to solver", dataset, programs,
        reconstruction=ReconstructionRecipe(ContinuousPIVReconstruction(), method_settings=_METHOD),
    )
    prepared = project.dataset
    assert len(prepared.generator_training_cases) == 4
    assert len(prepared.generator_validation_cases) == 1
    assert len(prepared.development_cases) == 1
    assert len(prepared.hidden_cases) == 2
    assert prepared.training_summary["task"] == "initial_boundary_value_solver"
    assert project.reconstruction_report.aggregate_metrics["gappy_withheld_relative_l2"] < 0.01
    for record in (*prepared.development_cases, *prepared.hidden_cases):
        public = record.case.public_data
        assert "observations" not in public
        assert "project_reconstruction" not in public
        assert public["forcing"]["access"] == "query_forcing"
        assert public["initial_condition"]["access"] == "query_initial_velocity"
        assert public["boundary_condition"]["access"] == "query_boundary_velocity"
        assert record.case.source.value == "initial"


def test_private_reference_changes_only_evaluation_not_reconstruction(tmp_path):
    dataset, _, reference = _fixture(tmp_path)
    problem, source, output = _reconstruct(dataset)
    original = _completed_velocities(output)
    first = problem.evaluate_reconstruction(output, source.reconstruction_validation_reference, {})
    with np.load(reference, allow_pickle=False) as arrays:
        changed = {key: arrays[key].copy() for key in arrays.files}
    changed["velocity"] = changed["velocity"] + 100.0
    np.savez(reference, **changed)
    _, changed_source, changed_output = _reconstruct(dataset)
    for key, coefficients in _completed_velocities(changed_output).items():
        np.testing.assert_array_equal(coefficients, original[key])
    second = problem.evaluate_reconstruction(changed_output, changed_source.reconstruction_validation_reference, {})
    assert second.aggregate_metrics["gappy_withheld_relative_l2"] > first.aggregate_metrics["gappy_withheld_relative_l2"] + 0.5
    assert "references" not in source.program_input.fit_data
    assert "reconstruction_validation_reference" not in source.program_input.metadata


def test_saved_dataset_excludes_sealed_validation_and_restores_answers(tmp_path):
    dataset, _, _ = _fixture(tmp_path)
    problem, _, output = _reconstruct(dataset)
    saved = save_continuous_mms_dataset(output, tmp_path / "continuous_dataset.json")
    text = saved.read_text()
    payload = json.loads(text)
    assert "reconstruction_validation" not in payload["splits"]
    assert "sealed_validation_reference" not in text
    assert "completed_velocity" not in text
    assert "references" not in text
    assert sum(len(rows) for rows in payload["splits"].values()) == 8
    restored = problem.prepare_dataset(replace(dataset, location=str(saved)))
    actual = restored.development_cases[0].truth._answer
    expected = problem.prepare_reconstructed_dataset(dataset, output).development_cases[0].truth._answer
    points = np.array([[0.12, 0.15, 0.2], [0.2, 0.9, 0.4]])
    np.testing.assert_array_equal(actual.evaluate_velocity(points), expected.evaluate_velocity(points))
    np.testing.assert_array_equal(actual.evaluate_forcing(points), expected.evaluate_forcing(points))
    with pytest.raises(FileExistsError):
        save_continuous_mms_dataset(output, saved)


def test_rejects_overlapping_recording_frames_across_splits(tmp_path):
    dataset, manifest, _ = _fixture(tmp_path)
    manifest["splits"]["generator_validation"][0]["source_frames"] = [1, 5]
    from pathlib import Path
    Path(dataset.location).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="overlap"):
        ExperimentalMMSProblem().prepare_reconstruction_source(dataset, {}, 0)


@pytest.mark.parametrize("missing", ["coordinate_units"])
def test_requires_declared_coordinate_units(tmp_path, missing):
    dataset, _, _ = _fixture(tmp_path)
    settings = dict(dataset.settings)
    settings.pop(missing)
    with pytest.raises(ValueError, match="viscosity|coordinate_units"):
        ExperimentalMMSProblem().prepare_reconstruction_source(replace(dataset, settings=settings), {}, 0)


def test_rejects_reconstructed_fields_as_dense_pod_calibration(tmp_path):
    dataset, manifest, _ = _fixture(tmp_path)
    manifest["splits"]["generator_training"][0]["generator_target_kind"] = "reconstruction_mean"
    from pathlib import Path
    Path(dataset.location).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="measured_full_field"):
        ExperimentalMMSProblem().prepare_reconstruction_source(dataset, {}, 0)


@pytest.mark.parametrize("geometry", [None, {"label": "unspecified obstacles"}])
def test_requires_explicit_manufactured_geometry(tmp_path, geometry):
    dataset, _, _ = _fixture(tmp_path)
    settings = dict(dataset.settings)
    settings.pop("mms_geometry")
    if geometry is not None:
        settings["mms_geometry"] = geometry
    # The old manifest's point-cloud label is not a physical-domain declaration.
    with pytest.raises(ValueError, match="geometry|obstacle_polygons"):
        ExperimentalMMSProblem().prepare_reconstruction_source(replace(dataset, settings=settings), {}, 0)


def test_accepts_explicit_dataset_geometry_and_data_settings_override(tmp_path):
    dataset, _, _ = _fixture(tmp_path)
    settings = dict(dataset.settings)
    declared = settings.pop("mms_geometry")
    dataset = replace(dataset, settings=settings, geometry=declared)
    problem = ExperimentalMMSProblem()
    source = problem.prepare_reconstruction_source(dataset, {}, 0)
    assert source.program_input.geometry["label"] == "declared PIV window"
    override = {"label": "explicit override", "obstacle_polygons": []}
    source = problem.prepare_reconstruction_source(dataset, {"mms_geometry": override}, 0)
    assert source.program_input.geometry["label"] == "explicit override"


def _prepared_groups(prepared):
    final_count = prepared.training_summary["terminal_final_test_cases"]
    return {"generator_training": prepared.generator_training_cases,
            "generator_validation": prepared.generator_validation_cases,
            "development": prepared.development_cases,
            "final_test": prepared.hidden_cases[:final_count],
            "verifier": prepared.hidden_cases[final_count:]}


def test_frozen_prepared_reload_preserves_cases_provenance_and_hash(tmp_path):
    import hashlib
    from pde_agent_verifier_solver.plugins.ns.experimental_mms import _dataset_digest
    dataset, _, _ = _fixture(tmp_path)
    backend = ExperimentalMMSDatasetRegistry()
    problem, _, output = _reconstruct(dataset, ExperimentalMMSProblem(backend))
    prepared = problem.prepare_reconstructed_dataset(dataset, output)
    source_provenance = prepared.training_summary["source_provenance"]
    for role, records in _prepared_groups(prepared).items():
        for index, record in enumerate(records):
            assert source_provenance[record.case.case_id] == {
                "case_id": f"{role}_case_{index}", "source_recording": "recording_a",
            }
    first = save_prepared_mms_dataset(prepared, tmp_path / "prepared_1.json")
    restored_backend = ExperimentalMMSDatasetRegistry()
    restored_problem = ExperimentalMMSProblem(restored_backend)
    restored_dataset = replace(dataset, location=str(first))
    restored = restored_problem.prepare_dataset(restored_dataset)
    for role, records in _prepared_groups(prepared).items():
        reloaded = _prepared_groups(restored)[role]
        assert tuple(record.case for record in reloaded) == tuple(record.case for record in records)
        for old, new in zip(records, reloaded):
            assert field_to_state(new.truth._answer) == field_to_state(old.truth._answer)
    assert restored.training_summary == prepared.training_summary
    assert restored.training_summary["source_provenance"] == source_provenance
    assert _dataset_digest(restored) == _dataset_digest(prepared)
    # The backend serves these restored identities, rather than freshly issued aliases.
    case = restored.development_cases[0].case
    session = restored_backend.public_session_factory(case)
    try:
        points = np.array([[0.15, -0.7, 0.2]])
        np.testing.assert_allclose(session.query_boundary_velocity(points),
                                   restored.development_cases[0].truth._answer.evaluate_velocity(points),
                                   rtol=1e-14, atol=1e-14)
    finally:
        session.close()
    second = save_prepared_mms_dataset(restored, tmp_path / "prepared_2.json")
    assert hashlib.sha256(first.read_bytes()).hexdigest() == hashlib.sha256(second.read_bytes()).hexdigest()
    # Re-reading into an already bound backend also preserves active answer identity.
    reread = restored_problem.prepare_dataset(restored_dataset)
    assert tuple(record.case for record in reread.hidden_cases) == tuple(record.case for record in restored.hidden_cases)
    third = save_prepared_mms_dataset(reread, tmp_path / "prepared_3.json")
    assert third.read_bytes() == first.read_bytes()


@pytest.mark.parametrize("corruption", ["public_forcing", "answer", "case_id", "role", "unknown_provenance"])
def test_frozen_prepared_reload_rejects_inconsistent_snapshots(tmp_path, corruption):
    from pathlib import Path
    dataset, _, _ = _fixture(tmp_path)
    problem, _, output = _reconstruct(dataset)
    prepared = problem.prepare_reconstructed_dataset(dataset, output)
    saved = save_prepared_mms_dataset(prepared, tmp_path / "frozen.json")
    payload = json.loads(saved.read_text())
    row = payload["splits"]["development"][0]
    if corruption == "public_forcing":
        row["case"]["public_data"]["forcing"]["access"] = "query_secret_solution"
    elif corruption == "answer":
        row["answer"]["psi"][0][1] += 1.0
    elif corruption == "case_id":
        row["case_id"] = "another-id"
    elif corruption == "role":
        payload["splits"]["final_test"], payload["splits"]["verifier"] = (
            payload["splits"]["verifier"], payload["splits"]["final_test"])
    else:
        payload["training_summary"]["source_provenance"]["unknown-case"] = {"case_id": "private-source"}
    Path(saved).write_text(json.dumps(payload))
    if corruption == "answer":
        # A valid edited field loads; its archived checksum is not a gate.
        assert ExperimentalMMSProblem().prepare_dataset(replace(dataset, location=str(saved)))
    else:
        with pytest.raises(ValueError, match="frozen|manufactured answer contract"):
            ExperimentalMMSProblem().prepare_dataset(replace(dataset, location=str(saved)))


def test_explicit_noise_settings_override_manifest_and_are_not_silent(tmp_path):
    dataset, _, _ = _fixture(tmp_path)
    problem = ExperimentalMMSProblem()
    from_dataset = replace(dataset, settings={**dataset.settings, "noise_standard_deviation": [0.04, 0.05]})
    source = problem.prepare_reconstruction_source(from_dataset, {}, 0)
    assert source.program_input.noise_model["standard_deviation"] == [0.04, 0.05]
    override = [0.11, 0.12]
    source = problem.prepare_reconstruction_source(from_dataset, {"noise_standard_deviation": override}, 0)
    assert source.program_input.noise_model["standard_deviation"] == override
    for rows in source.program_input.target_observations.values():
        for row in rows:
            np.testing.assert_array_equal(row["observations"].noise_standard_deviation, override)


def _replace_npz(path, **changes):
    with np.load(path, allow_pickle=False) as arrays:
        payload = {key: arrays[key].copy() for key in arrays.files}
    payload.update(changes)
    np.savez(path, **payload)


def _float32_timestamp_fixture(tmp_path, *, declared_rate=True, keep_float32=False):
    """Match legacy completed NPZ: float32 acquisition -> float64 -> local time."""
    from pathlib import Path
    dataset, manifest, _ = _fixture(tmp_path)
    if declared_rate:
        # The actual airfoil manifest estimates its period from float32 times.
        manifest["source_dataset"] = {"sampling_hz": 50.000047683761295}
        manifest["time"] = {"sampling_hz": 50.000047683761295, "samples_per_case": 4}
    serial = 0
    for rows in manifest["splits"].values():
        for row in rows:
            start = 8 + serial * 100
            row["source_frames"] = [start, start + 4]
            raw = ((np.arange(start, start + 4) + 1) / 50.0).astype(np.float32)
            time = raw if keep_float32 else raw.astype(np.float64) - float(raw[0])
            _replace_npz(tmp_path / row["case_file"], time=time)
            if "measurement_reference_file" in row:
                _replace_npz(tmp_path / row["measurement_reference_file"], time=time)
            serial += 1
    Path(dataset.location).write_text(json.dumps(manifest))
    return dataset, manifest


@pytest.mark.parametrize("keep_float32", [False, True])
def test_declared_clock_accepts_legacy_float32_timestamps(tmp_path, keep_float32):
    dataset, _ = _float32_timestamp_fixture(tmp_path, keep_float32=keep_float32)
    source = ExperimentalMMSProblem().prepare_reconstruction_source(dataset, {}, 0)
    expected = np.arange(4) / 50.000047683761295
    np.testing.assert_array_equal(source.program_input.fit_data["time"], expected)
    np.testing.assert_array_equal(source.reconstruction_validation_reference["coordinates"][2], expected)
    clock = source.program_input.metadata["frame_clock"]
    assert clock["policy"] == "declared_sampling_hz"
    assert clock["sampling_hz"] == 50.000047683761295
    assert 1e-8 < clock["maximum_timestamp_adjustment"] < 1e-5


def test_undeclared_clock_accepts_only_float32_roundoff(tmp_path):
    dataset, _ = _float32_timestamp_fixture(tmp_path, declared_rate=False)
    source = ExperimentalMMSProblem().prepare_reconstruction_source(dataset, {}, 0)
    clock = source.program_input.metadata["frame_clock"]
    assert clock["policy"] == "first_window"
    assert clock["sampling_hz"] is None
    assert 1e-8 < clock["maximum_timestamp_adjustment"] < 1e-5


@pytest.mark.parametrize("declared_rate", [False, True])
def test_rejects_genuine_sampling_rate_change(tmp_path, declared_rate):
    dataset, manifest = _float32_timestamp_fixture(tmp_path, declared_rate=declared_rate)
    changed = manifest["splits"]["generator_training"][1]
    start = changed["source_frames"][0]
    raw = ((np.arange(start, start + 4) + 1) / 49.5).astype(np.float32).astype(np.float64)
    _replace_npz(tmp_path / changed["case_file"], time=raw - raw[0])
    with pytest.raises(ValueError, match="sampling_hz and sampling intervals"):
        ExperimentalMMSProblem().prepare_reconstruction_source(dataset, {}, 0)


def test_undeclared_float64_clock_does_not_inherit_float32_tolerance(tmp_path):
    dataset, manifest, _ = _fixture(tmp_path)
    changed = manifest["splits"]["generator_training"][1]
    _replace_npz(tmp_path / changed["case_file"], time=np.linspace(0, 0.3, 4) * (1 + 1e-8))
    with pytest.raises(ValueError, match="timestamp rounding"):
        ExperimentalMMSProblem().prepare_reconstruction_source(dataset, {}, 0)


def test_declared_clock_rejects_changed_window_length(tmp_path):
    dataset, manifest = _float32_timestamp_fixture(tmp_path)
    changed = manifest["splits"]["generator_training"][1]
    _replace_npz(tmp_path / changed["case_file"], time=np.arange(5) / 50.0)
    with pytest.raises(ValueError, match="window length"):
        ExperimentalMMSProblem().prepare_reconstruction_source(dataset, {}, 0)


@pytest.mark.parametrize("target", ["case_file", "measurement_reference_file"])
def test_clock_roundoff_does_not_relax_spatial_coordinates(tmp_path, target):
    dataset, manifest = _float32_timestamp_fixture(tmp_path)
    changed = manifest["splits"]["reconstruction_validation"][0]
    path = tmp_path / changed[target]
    with np.load(path, allow_pickle=False) as arrays:
        x = arrays["x"].copy()
    x[0] += 1e-6
    _replace_npz(path, x=x)
    with pytest.raises(ValueError, match="coordinates"):
        ExperimentalMMSProblem().prepare_reconstruction_source(dataset, {}, 0)


def test_rejects_conflicting_declared_sampling_rates(tmp_path):
    from pathlib import Path
    dataset, manifest = _float32_timestamp_fixture(tmp_path)
    manifest["time"]["sampling_hz"] = 49.5
    Path(dataset.location).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="sampling_hz declarations disagree"):
        ExperimentalMMSProblem().prepare_reconstruction_source(dataset, {}, 0)


@pytest.mark.parametrize("missing", [("source_recording",), ("source_frames",),
                                      ("source_recording", "source_frames")])
def test_raw_completed_import_requires_split_provenance(tmp_path, missing):
    from pathlib import Path
    dataset, manifest, _ = _fixture(tmp_path)
    row = manifest["splits"]["generator_validation"][0]
    for key in missing:
        row.pop(key)
    Path(dataset.location).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="requires source_recording and source_frames"):
        ExperimentalMMSProblem().prepare_reconstruction_source(dataset, {}, 0)


def test_overlapping_window_cannot_bypass_split_check_by_omitting_recording(tmp_path):
    from pathlib import Path
    dataset, manifest, _ = _fixture(tmp_path)
    row = manifest["splits"]["generator_validation"][0]
    row["source_frames"] = [1, 5]
    row.pop("source_recording")
    Path(dataset.location).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="verify split isolation"):
        ExperimentalMMSProblem().prepare_reconstruction_source(dataset, {}, 0)


def test_reconstruction_stops_at_complete_arrays_without_continuous_field_or_generator_pca(tmp_path, monkeypatch):
    dataset, _, _ = _fixture(tmp_path)
    module = numerical_symbols()
    from agent_verifier.neural_generator.structured_ns_distribution import StructuredCoefficientPCA
    def forbidden(*args, **kwargs):
        raise AssertionError("continuous projection and Generator PCA belong after Reconstruction")
    monkeypatch.setattr(module, "StructuredNSCoefficientProjector", forbidden)
    monkeypatch.setattr(module, "ContinuousStructuredNSMMS", forbidden)
    monkeypatch.setattr(StructuredCoefficientPCA, "fit", forbidden)
    _, source, output = _reconstruct(dataset)
    data = output.completed_data
    assert "split_answers" not in data
    assert "coefficients" not in data and "pca" not in data
    assert output.diagnostics["output_kind"] == "complete_discrete_velocity"
    assert data["physical_settings"]["pressure_model"] == "zero"
    assert output.public_settings == {"rank": output.selected_rank, "prior_precision": 1e-5, "noise_scale": 1.0}
    for key in ("x", "y", "time"):
        np.testing.assert_array_equal(data[key], source.program_input.fit_data[key])
    for rows in data["split_fields"].values():
        for row in rows:
            assert set(row) == {"case_id", "source_recording", "velocity"}
            assert isinstance(row["velocity"], np.ndarray)
            assert row["velocity"].shape == (4, 2, 25)


def test_viscosity_is_required_for_mms_but_not_reconstruction(tmp_path):
    dataset, _, _ = _fixture(tmp_path)
    settings = dict(dataset.settings)
    settings.pop("viscosity")
    dataset = replace(dataset, settings=settings)
    problem, _, output = _reconstruct(dataset)
    assert output.completed_data["physical_settings"]["viscosity"] is None
    assert output.completed_data["split_fields"]["development"]
    with pytest.raises(ValueError, match="MMS viscosity"):
        problem.prepare_reconstructed_dataset(dataset, output)


@pytest.mark.parametrize("pressure", [None, "measured"])
def test_mms_projection_requires_explicit_zero_pressure_convention(tmp_path, pressure):
    dataset, _, _ = _fixture(tmp_path)
    _, _, output = _reconstruct(dataset)
    data = dict(output.completed_data)
    physical = dict(data["physical_settings"])
    if pressure is None:
        physical.pop("pressure_model")
    else:
        physical["pressure_model"] = pressure
    data["physical_settings"] = physical
    with pytest.raises(ValueError, match="explicit manufactured pressure_model"):
        project_reconstructed_output(replace(output, completed_data=data))


def test_saved_default_no_slip_geometry_restores_with_explicit_default(tmp_path):
    dataset, _, _ = _fixture(tmp_path)
    settings = dict(dataset.settings)
    settings["mms_geometry"] = {"obstacle_polygons": [], "outer_boundary": "no_slip"}
    dataset = replace(dataset, settings=settings)
    problem, _, output = _reconstruct(dataset)
    saved = save_continuous_mms_dataset(output, tmp_path / "no_slip.json")
    restored = problem.prepare_dataset(replace(dataset, location=str(saved)))
    assert restored.development_cases[0].case.public_data["boundary_condition"]["kind"] == "no_slip_velocity"


def test_saved_projection_rejects_actual_basis_change(tmp_path):
    dataset, _, _ = _fixture(tmp_path)
    problem, _, output = _reconstruct(dataset)
    saved = save_continuous_mms_dataset(output, tmp_path / "saved.json")
    settings = dict(dataset.settings)
    settings["projection_config"] = {**_PROJECTION, "spatial_extension": .5}
    with pytest.raises(ValueError, match="frozen projection_config"):
        problem.prepare_dataset(replace(dataset, location=str(saved), settings=settings))
