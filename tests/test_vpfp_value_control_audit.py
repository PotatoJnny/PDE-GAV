"""Tiny synthetic encoding fixtures only; no PDE solve or real-data claim."""

from hashlib import sha256
import json

import numpy as np
import pytest

from pde_agent_verifier_solver.plugins.vpfp import diffusion_data as data
from pde_agent_verifier_solver.plugins.vpfp import value_control_audit as audit
from pde_agent_verifier_solver.plugins.vpfp.canonical_binding import VPFPOnlineModelSpec


def source_slopes(values):
    """Independent finite-difference fixture using NumPy's first-order endpoints."""
    return np.gradient(values, np.linspace(0., 1., values.shape[-2]), axis=-2,
                       edge_order=1).astype(np.float32)


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "unit-test-no-compute")
    monkeypatch.setattr(data, "_SCRATCH_ROOT", tmp_path)
    base = tmp_path / "base.pt"
    base.write_bytes(b"synthetic never-loaded decoder")
    digest = sha256(base.read_bytes()).hexdigest()
    spec = VPFPOnlineModelSpec("audit_fixture", str(base), digest, str(base), digest, (.02, .08))
    config = {"structured_time_basis": "piecewise_cubic_hermite_v1",
              "structured_direct_coefficients": True, "structured_spectral_latent": True,
              "structured_time_control_count": 5, "rho_max_mode_x": 1,
              "rho_max_mode_y": 1, "structured_local_spatial_mode": 0,
              "hermite_total_degree": 1, "structured_local_channels": 3,
              "structured_kinetic_global_dim": 2, "latent_dim": 112,
              "time_domain": [0., 1.], "dtype": "float32"}
    layout = {"rho_mode_count": 4, "rho_time_feature_count": 10,
              "rho_latent_dim": 80, "kinetic_global_latent_dim": 2,
              "local_spatial_feature_count": 1, "local_time_feature_count": 10,
              "local_latent_dim": 30, "hermite_mode_count": 3, "total_latent_dim": 112}

    class Sampler:
        def condition(self, seed):
            return {"seed": seed, "collision_nu": .05, "temperature": 1.}

        def generate(self, seeds):
            rows = []
            for seed in seeds:
                rng = np.random.default_rng(seed)
                rho = rng.normal(size=(5, 8)).astype(np.float32)
                micro = rng.normal(size=(3, 5, 1)).astype(np.float32)
                rows.append(np.concatenate((np.stack((rho, source_slopes(rho))).ravel(),
                    np.zeros(2, dtype=np.float32), np.stack((micro, source_slopes(micro)), axis=1).ravel())))
            return {"latents": np.asarray(rows, dtype=np.float32),
                    "equation_conditions": np.asarray([[.05, 1.]] * len(seeds), dtype=np.float32),
                    "equation_seeds": np.asarray(seeds, dtype=np.int64),
                    "diagnostics": {"synthetic_encoding_fixture": True}}

    sampler = Sampler()
    runtime = data._CollectionRuntime(sampler, sampler.condition, 112, {
        "decoder_config": config, "structured_latent_layout": layout,
        "generation_config": {"nt": 5, "t1": 1., "dtype": "float32"},
        "equation_config": {"synthetic": True}})
    monkeypatch.setattr(data, "_collection_runtime", lambda *_: runtime)
    root = tmp_path / "corpus"
    data.collect_corpus(spec, root, {name: 10 for name in data.SPLIT_NAMES}, 9371,
                        strata=(1, 1), batch_size=1, device="cpu")
    return root


def repin_manifest(root, mutate):
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    mutate(manifest)
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = sha256(data._canonical_bytes(manifest)).hexdigest()
    path.write_text(json.dumps(manifest))


def change_coefficients(root, split, change):
    path = root / split / "coefficients.npy"
    values = np.load(path, mmap_mode="r+")
    change(values)
    values.flush()
    del values
    repin_manifest(root, lambda manifest: manifest["splits"][split]["files"]["coefficients"].update(
        sha256=data._file_sha256(path)))


def test_complete_five_split_chunked_readonly_gate(corpus, monkeypatch):
    before = {path.relative_to(corpus): (data._file_sha256(path), path.stat().st_mtime_ns)
              for path in corpus.rglob("*") if path.is_file()}
    original = audit._metrics
    seen = []

    def tracked(values, slopes, dt):
        seen.append(len(values))
        assert len(values) <= 3
        return original(values, slopes, dt)

    monkeypatch.setattr(audit, "_metrics", tracked)
    report = audit.audit_value_control_corpus(corpus, batch_size=3)
    assert audit.require_value_control_audit(report) is report
    assert report["passed"] is True
    assert report["trajectory_count"] == 50
    assert report["chunks_scanned"] == 20
    assert len(seen) == 40
    assert set(report["splits"]) == set(data.SPLIT_NAMES)
    assert report["all_five_splits_fully_scanned"] is True
    assert report["sampled_rows_only"] is False
    assert report["physical_fidelity_verified"] is False
    assert report["target_projection_performed"] is False
    assert len(report["source_files"]) == 26
    assert report["layout"]["delta_tau"] == .25
    assert report["layout"]["structured_latent_layout"]["total_latent_dim"] == 112
    assert report["source_manifest_sha256"] == data.read_manifest(corpus)["manifest_sha256"]
    for role in data.SPLIT_NAMES:
        split = data.load_split(corpus, role)
        expected_hashes = [sha256(np.asarray(row, dtype="<f4").tobytes() + np.asarray(condition, dtype="<f4").tobytes()).hexdigest()
                           for row, condition in zip(split.latents, split.conditions)]
        assert report["splits"][role]["input_answer_sha256"] == expected_hashes
    for split in report["splits"].values():
        assert set(split["groups"]) == {"rho", "kinetic_channel_000", "kinetic_channel_001", "kinetic_channel_002"}
        assert split["trajectory_count"] == 10
        assert split["failed_case_ids"] == []
        for group in split["groups"].values():
            assert group["metrics"]["scaled_rms_error"]["max"] < 2e-7
    assert before == {path.relative_to(corpus): (data._file_sha256(path), path.stat().st_mtime_ns)
                      for path in corpus.rglob("*") if path.is_file()}
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("index,group", [(40, "rho"), (77, "rho"), (87, "kinetic_channel_000"), (109, "kinetic_channel_002")])
def test_bad_endpoint_interior_and_last_case_are_never_silently_projected(corpus, index, group):
    def corrupt(values):
        values[-1, index] += .01
    change_coefficients(corpus, "reference_b", corrupt)
    before = data._file_sha256(corpus / "reference_b" / "coefficients.npy")
    report = audit.audit_value_control_corpus(corpus, batch_size=3)
    assert report["status"] == "rejected"
    failed_id = str(np.load(corpus / "reference_b" / "case_ids.npy")[-1])
    assert report["splits"]["reference_b"]["failed_case_ids"] == [failed_id]
    assert report["splits"]["reference_b"]["groups"][group]["failed_case_ids"] == [failed_id]
    with pytest.raises(ValueError, match="training rejected"):
        audit.require_value_control_audit(report)
    assert data._file_sha256(corpus / "reference_b" / "coefficients.npy") == before


def test_inactive_compatibility_coefficients_must_be_zero(corpus):
    change_coefficients(corpus, "train", lambda values: values.__setitem__((0, 80), .0001))
    report = audit.audit_value_control_corpus(corpus)
    assert report["passed"] is False
    assert len(report["splits"]["train"]["inactive_nonzero_case_ids"]) == 1


def test_constant_and_zero_channels_have_finite_statistics(corpus):
    def constant(values):
        values[:, :] = 0
        values[:, :40] = 3.5
        values[:, 82:87] = 1e-24
    for split in data.SPLIT_NAMES:
        change_coefficients(corpus, split, constant)
    report = audit.audit_value_control_corpus(corpus)
    assert report["passed"] is True
    json.dumps(report, allow_nan=False)


def test_tiny_kinetic_channel_is_not_hidden_by_large_other_channels(corpus):
    def tiny_bad_channel(values):
        values[:, 102:112] *= 1e-12
        values[0, 111] += 1e-13
    change_coefficients(corpus, "validation", tiny_bad_channel)
    report = audit.audit_value_control_corpus(corpus)
    assert report["passed"] is False
    assert report["splits"]["validation"]["groups"]["kinetic_channel_002"]["failed_case_count"] == 1


def test_float32_roundoff_is_separated_from_structural_error(corpus):
    report = audit.audit_value_control_corpus(corpus)
    metrics = report["splits"]["train"]["groups"]["rho"]["metrics"]
    assert 0 < metrics["scaled_rms_error"]["max"] < 1e-6
    assert metrics["float32_reconstruction_scaled_rms_error"]["max"] < 1e-6


@pytest.mark.parametrize("field,value", [
    ("structured_time_basis", "global_chebyshev_v1"),
    ("structured_time_control_count", 33),
    ("structured_time_control_count", True),
    ("structured_direct_coefficients", False),
    ("structured_spectral_latent", False),
    ("hermite_total_degree", 2),
    ("latent_dim", 113),
    ("dtype", "float64"),
    ("time_domain", [0., 2.]),
])
def test_unknown_or_inconsistent_source_configuration_rejected(corpus, field, value):
    repin_manifest(corpus, lambda manifest: manifest["provenance"]["decoder_config"].update({field: value}))
    with pytest.raises(ValueError):
        audit.audit_value_control_corpus(corpus)


@pytest.mark.parametrize("section,field,value", [
    ("generation_config", "nt", 6), ("generation_config", "t1", 2.),
    ("generation_config", "dtype", "float64"),
    ("structured_latent_layout", "rho_latent_dim", 81),
    ("structured_latent_layout", "local_time_feature_count", 8),
])
def test_layout_and_generation_must_match_decoder_exactly(corpus, section, field, value):
    repin_manifest(corpus, lambda manifest: manifest["provenance"][section].update({field: value}))
    with pytest.raises(ValueError):
        audit.audit_value_control_corpus(corpus)


@pytest.mark.parametrize("batch_size", [0, 9, True, 2.5])
def test_invalid_chunk_size(corpus, batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        audit.audit_value_control_corpus(corpus, batch_size=batch_size)


@pytest.mark.parametrize("tolerance", [0, -.1, .01, np.nan, np.inf, True])
def test_invalid_tolerance(corpus, tolerance):
    with pytest.raises(ValueError, match="tolerance"):
        audit.audit_value_control_corpus(corpus, tolerance=tolerance)


def test_corrupt_file_is_rejected_before_scan(corpus, monkeypatch):
    values = np.load(corpus / "train" / "coefficients.npy", mmap_mode="r+")
    values[0, 0] += 1
    values.flush()
    monkeypatch.setattr(audit, "_metrics", lambda *_: pytest.fail("must hash before scanning"))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        audit.audit_value_control_corpus(corpus)


def test_wrong_complete_coefficient_shape_is_rejected(corpus):
    path = corpus / "train" / "coefficients.npy"
    values = np.load(path)[:, :-1].copy()
    np.save(path, values)
    repin_manifest(corpus, lambda manifest: manifest["splits"]["train"]["files"]["coefficients"].update(
        sha256=data._file_sha256(path), size_bytes=path.stat().st_size, shape=list(values.shape)))
    with pytest.raises(ValueError, match="shape/dtype mismatch"):
        audit.audit_value_control_corpus(corpus)


def test_mutation_during_scan_cannot_publish_passing_report(corpus, monkeypatch):
    original = audit._metrics
    mutated = False

    def mutate(values, slopes, dt):
        nonlocal mutated
        if not mutated:
            path = corpus / "train" / "coefficients.npy"
            writable = np.load(path, mmap_mode="r+")
            writable[0, 0] += 1
            writable.flush()
            mutated = True
        return original(values, slopes, dt)

    monkeypatch.setattr(audit, "_metrics", mutate)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        audit.audit_value_control_corpus(corpus)


def test_repinning_manifest_during_scan_is_still_detected(corpus, monkeypatch):
    original = audit._metrics
    mutated = False

    def mutate(values, slopes, dt):
        nonlocal mutated
        if not mutated:
            repin_manifest(corpus, lambda manifest: manifest.update(audit_time_mutation="unexpected"))
            mutated = True
        return original(values, slopes, dt)

    monkeypatch.setattr(audit, "_metrics", mutate)
    with pytest.raises(ValueError, match="changed during"):
        audit.audit_value_control_corpus(corpus)


def test_nonfinite_corpus_cannot_pass(corpus):
    change_coefficients(corpus, "calibration", lambda values: values.__setitem__((0, 0), np.nan))
    with pytest.raises(ValueError, match="nonfinite coefficients"):
        audit.audit_value_control_corpus(corpus)


@pytest.mark.parametrize("change", [
    lambda report: report.update(passed=False),
    lambda report: report.update(all_five_splits_fully_scanned=False),
    lambda report: report.update(full_file_hashes_verified_before_and_after=False),
    lambda report: report["splits"].pop("reference_b"),
    lambda report: report["splits"]["train"].update(passed=False),
])
def test_gate_rejects_failed_or_incomplete_reports(corpus, change):
    report = audit.audit_value_control_corpus(corpus)
    change(report)
    with pytest.raises(ValueError, match="training rejected"):
        audit.require_value_control_audit(report)


def test_two_controls_use_one_sided_same_slope():
    values = np.asarray([[[1.], [3.]]], dtype=np.float32)
    slopes = np.asarray([[[2.], [2.]]], dtype=np.float32)
    metrics = audit._metrics(values, slopes, 1.)
    assert metrics["scaled_rms_error"].tolist() == [0.]


def test_physical_horizon_does_not_change_normalized_slope_units(corpus):
    def change(manifest):
        manifest["time_domain"] = [0., 3.]
        manifest["provenance"]["decoder_config"]["time_domain"] = [0., 3.]
        manifest["provenance"]["generation_config"]["t1"] = 3.
    repin_manifest(corpus, change)
    report = audit.audit_value_control_corpus(corpus)
    assert report["passed"] is True
    assert report["layout"]["delta_tau"] == .25
    assert report["layout"]["delta_physical_time"] == .75
