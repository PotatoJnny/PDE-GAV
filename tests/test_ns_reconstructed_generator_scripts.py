"""The PIV preparation CLI hands actual numerical artifacts to the original trainer."""
from __future__ import annotations

from dataclasses import replace
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _scripts(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return (importlib.import_module("prepare_ns_reconstructed_generator"),
            importlib.import_module("evaluate_ns_reconstructed_generator"))



def _measurement_references(tmp_path, manifest):
    """Keep real references separate and make every legacy case unusable."""
    expected = {}
    serial = 0
    for role, rows in manifest["splits"].items():
        expected[role] = []
        for row in rows:
            with np.load(tmp_path / row["case_file"]) as arrays:
                x, y, time = (arrays[name].copy() for name in ("x", "y", "time"))
                original = arrays["velocity"].copy()
            if role == "generator_training":
                velocity = original
            else:
                velocity = np.stack(((1 + .15*serial) * np.sin(x)[None] * (1 + time[:, None]),
                                     (.2 + .03*serial) * np.cos(y)[None] * (1 - 2*time[:, None])), axis=1)
            reference = tmp_path / f"measured_{row['case_id']}.npz"
            np.savez(reference, velocity=velocity, x=x, y=y, time=time)
            row["measurement_reference_file"] = reference.name
            expected[role].append(velocity)
            # Direct measured preparation cannot obtain coordinates, observations,
            # or a full field by accidentally opening this old Case.
            np.savez(tmp_path / row["case_file"], velocity=np.array([np.nan]))
            serial += 1
    return {role: np.stack(values) for role, values in expected.items()}


def test_coefficient_export_is_readable_by_original_numerical_loader(tmp_path, monkeypatch):
    prepare, _ = _scripts(monkeypatch)
    from agent_verifier.neural_generator.structured_ns_distribution import load_structured_coefficient_corpus
    values = np.arange(3*2*2*3*3, dtype=np.float64).reshape(3, 2, 2, 3, 3)
    path = prepare.export_coefficient_corpus(values, case_ids=["a", "b", "c"], families=["real"]*3,
        signature={"projection_config": {}, "coefficient_shape": list(values.shape[1:])},
        output=tmp_path / "corpus", split="train", shard_cases=2)
    corpus = load_structured_coefficient_corpus(path)
    np.testing.assert_array_equal(corpus.coefficients.numpy(), values.astype(np.float32))
    assert corpus.case_ids == ("a", "b", "c")
    assert prepare.coefficient_sha256(values) == prepare.coefficient_sha256(corpus.coefficients.numpy())


@pytest.mark.parametrize("input_mode", ["reconstructed", "measured"])
def test_scripts_project_freeze_train_and_evaluate_numerical_generator(tmp_path, monkeypatch, input_mode):
    torch = pytest.importorskip("torch")
    from tests.test_ns_experimental_mms_project import _fixture
    prepare, evaluate = _scripts(monkeypatch)
    dataset, manifest, _ = _fixture(tmp_path)
    original = manifest["splits"]["generator_validation"][0]
    for index in range(1, 8):
        with np.load(tmp_path / original["case_file"]) as a:
            values = {name: a[name].copy() for name in a.files}
        values["public_values"] *= 1 + .025*index
        path = tmp_path / f"extra_validation_{index}.npz"
        np.savez(path, **values)
        manifest["splits"]["generator_validation"].append({**original,
            "case_id": f"extra_validation_{index}", "case_file": path.name,
            "source_recording": f"extra_validation_recording_{index}"})
    Path(dataset.location).write_text(json.dumps(manifest))
    with np.load(tmp_path / manifest["splits"]["generator_training"][0]["case_file"]) as a:
        x, y = a["x"], a["y"]
        left = float(np.unique(x)[1])
    geometry = tmp_path / "geometry.json"
    geometry.write_text(json.dumps({"outer_boundary": "manufactured_trace", "obstacle_polygons": [],
        "domain_bounds": [left, float(x.max()), float(y.min()), float(y.max())]}))
    if input_mode == "measured":
        expected = _measurement_references(tmp_path, manifest)
        Path(dataset.location).write_text(json.dumps(manifest))
        def forbidden(*args, **kwargs):
            raise AssertionError("measured mode must not use sparse reconstruction or POD")
        for name in ("prepare_piv_reconstruction_source", "reconstruct_selected", "fit_trajectory_pod", "GappyPODPosterior"):
            monkeypatch.setattr(prepare, name, forbidden)
    out = tmp_path / "prepared"
    args = prepare.parser().parse_args(["--manifest", dataset.location, "--out", str(out),
        "--geometry-json", str(geometry), "--viscosity", ".02", "--pod-rank", "3",
        "--spatial-mode-x", "2", "--spatial-mode-y", "2", "--time-order", "1", "--pca-ranks", "1,2,3",
        "--device", "cpu", "--train-flow", "--flow-steps", "2", "--flow-width", "8",
        "--flow-depth", "1", "--flow-time-width", "8", "--flow-batch-size", "4", "--checkpoint-every", "1"])
    args.input_mode = input_mode
    if input_mode == "measured":
        args.pod_rank = 1  # Must not cap direct coefficient PCA at this unused parameter.
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    try:
        report = prepare.prepare(args)
        assert report["training_cases"] == 4 and report["validation_cases"] == 8
        assert report["fitted_point_count"] == 20 and report["input_point_count"] == 25
        contract = json.loads((out / "continuous_contract.json").read_text())
        from pde_agent_verifier_solver.plugins.ns.experimental_mms_project import ExperimentalMMSProblem
        from pde_agent_verifier_solver.plugins.ns.experimental_mms import _answer, _coefficient_batch
        restored = ExperimentalMMSProblem().prepare_dataset(replace(dataset, location=str(out / "prepared_dataset.json"), settings={}))
        for role, records in (("training", restored.generator_training_cases), ("validation", restored.generator_validation_cases)):
            assert prepare.coefficient_sha256(_coefficient_batch([_answer(record) for record in records])) == contract[f"{role}_coefficients_sha256"]
        assert len(restored.development_cases) == 1 and len(restored.hidden_cases) == 2
        explicit = ExperimentalMMSProblem().prepare_dataset(replace(dataset, location=str(out / "prepared_dataset.json"),
            settings={"projection_config": contract["projection_config"]}))
        assert len(explicit.generator_training_cases) == 4
        checkpoint = out / "base_generator/checkpoints/flow_step_0000002.pt"
        assert checkpoint.is_file()
        from pde_agent_verifier_solver.plugins.ns.experimental_mms import assemble_experimental_mms
        from pde_agent_verifier_solver.plugins.ns.canonical_backend import CanonicalStructuredNSBackend
        assembly = assemble_experimental_mms(restored, flow_checkpoint=checkpoint, pca_checkpoint=out / "pca_checkpoint.pt",
            generator_settings={"device": "cpu", "integration_steps": 2, "flow_state": "raw", "perturbation_amplitude": 1e-7})
        assert isinstance(assembly.bundle.backend, CanonicalStructuredNSBackend)
        if input_mode == "measured":
            selection = json.loads((out / "pca_selection.json").read_text())
            assert selection["candidate_rank_cap"] == 3
            assert not (out / "reconstruction_report.json").exists()
            assert "reconstruction_validation" not in report
            with np.load(out / "validation_reference.npz") as arrays:
                np.testing.assert_array_equal(arrays["measured"], expected["generator_validation"][..., x >= left])
                assert not any("reconstruct" in name for name in arrays.files)
            projection = json.loads((out / "projection_report.json").read_text())
            for role in prepare.ROLES:
                assert len(projection[role]["cases"]) == len(expected[role])
            assert restored.training_summary["input_mode"] == "measured"
            assert all(record.case.public_data["distribution_lineage"]["origin"] == "experimental_measurement"
                       for record in (*restored.generator_training_cases, *restored.generator_validation_cases,
                                      *restored.development_cases, *restored.hidden_cases))
        quality = evaluate.evaluate(evaluate.parser().parse_args(["--prepared", str(out),
            "--samples", "8", "--integration-steps", "2", "--baseline-repeats", "2", "--device", "cpu"]))
        assert quality["generated_values_finite"]
        assert quality["generator_sample_count"] == 8
        assert quality[f"continuous_projection_to_{input_mode}_validation_relative_l2"]["count"] == 8
        if input_mode == "measured":
            assert "continuous_reconstruction_withheld_measured_relative_l2" not in quality
            assert "continuous_projection_to_reconstructed_validation_relative_l2" not in quality
            wrong = evaluate.parser().parse_args(["--prepared", str(out), "--input-mode", "reconstructed"])
            with pytest.raises(ValueError, match="input_mode differs"):
                evaluate.evaluate(wrong)
        assert (out / "evaluation_flow_step_0000002_raw/velocity_fields.png").is_file()
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("problem,match", [
    ("missing_reference", "PIV file reference"),
    ("legacy_reconstruction", "reconstructed case"),
    ("overlap", "overlap across scientific splits"),
    ("missing_provenance", "source_recording and source_frames"),
    ("duplicate_reference", "same measurement reference"),
])
def test_measured_loader_rejects_ambiguous_truth_and_split_leakage(tmp_path, monkeypatch, problem, match):
    from tests.test_ns_experimental_mms_project import _fixture
    prepare, _ = _scripts(monkeypatch)
    dataset, manifest, _ = _fixture(tmp_path)
    _measurement_references(tmp_path, manifest)
    row = manifest["splits"]["development"][0]
    if problem == "missing_reference":
        row.pop("measurement_reference_file")
    elif problem == "legacy_reconstruction":
        row["measurement_reference_file"] = row["case_file"]
    elif problem == "overlap":
        row["source_frames"] = manifest["splits"]["generator_training"][0]["source_frames"]
    elif problem == "missing_provenance":
        row.pop("source_recording")
    else:
        row["measurement_reference_file"] = manifest["splits"]["generator_training"][0]["measurement_reference_file"]
    Path(dataset.location).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=match):
        prepare.load_measured_selected(dataset)


def test_measured_loader_returns_truth_in_every_role_without_sparse_layouts(tmp_path, monkeypatch):
    from tests.test_ns_experimental_mms_project import _fixture
    prepare, _ = _scripts(monkeypatch)
    dataset, manifest, _ = _fixture(tmp_path)
    expected = _measurement_references(tmp_path, manifest)
    manifest.pop("observation_layout")
    Path(dataset.location).write_text(json.dumps(manifest))
    actual, records, coordinates, metadata = prepare.load_measured_selected(dataset)
    for role in prepare.ROLES:
        np.testing.assert_array_equal(actual[role], expected[role])
        assert [row["case_id"] for row in records[role]] == [row["case_id"] for row in manifest["splits"][role]]
    assert metadata["split_counts"]["generator_training"] == 4
    assert coordinates[2].shape == (4,)



def test_heldout_measurements_do_not_change_pca_or_rank_selection(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from tests.test_ns_experimental_mms_project import _fixture
    prepare, _ = _scripts(monkeypatch)
    dataset, manifest, _ = _fixture(tmp_path)
    _measurement_references(tmp_path, manifest)
    Path(dataset.location).write_text(json.dumps(manifest))
    geometry = tmp_path / "geometry.json"
    geometry.write_text(json.dumps({"outer_boundary": "manufactured_trace", "obstacle_polygons": []}))
    args = prepare.parser().parse_args(["--manifest", dataset.location, "--input-mode", "measured",
        "--out", str(tmp_path / "first"), "--geometry-json", str(geometry),
        "--spatial-mode-x", "2", "--spatial-mode-y", "2", "--time-order", "1", "--pca-ranks", "1,2,3",
        "--device", "cpu"])
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        prepare.prepare(args)
        for role in ("development", "verifier", "final_test", "reconstruction_validation"):
            for row in manifest["splits"][role]:
                path = tmp_path / row["measurement_reference_file"]
                with np.load(path) as arrays:
                    values = {name: arrays[name].copy() for name in arrays.files}
                values["velocity"] *= 1000.0
                np.savez(path, **values)
        args.out = str(tmp_path / "second")
        prepare.prepare(args)
        cls = prepare.StructuredCoefficientPCA
        first = cls.load(tmp_path / "first/pca_checkpoint.pt")
        second = cls.load(tmp_path / "second/pca_checkpoint.pt")
        assert first.fit_case_count == second.fit_case_count == 4
        for name in ("coordinate_mean", "field_scale", "components", "score_scale", "singular_values"):
            torch.testing.assert_close(getattr(first, name), getattr(second, name), rtol=0, atol=0)
        assert json.loads((tmp_path / "first/pca_selection.json").read_text()) == json.loads(
            (tmp_path / "second/pca_selection.json").read_text())
        for split in ("train", "validation"):
            a = json.loads((tmp_path / f"first/coefficients/{split}/coefficient_manifest.json").read_text())
            b = json.loads((tmp_path / f"second/coefficients/{split}/coefficient_manifest.json").read_text())
            assert a["encoding_signature"] == b["encoding_signature"]
    finally:
        torch.set_num_threads(previous_threads)
