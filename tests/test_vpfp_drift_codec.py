"""Tiny analytic/local codec tests; no real trajectory collection or training."""
from copy import deepcopy
from dataclasses import replace
import importlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("drift_codec_test", ROOT / "scripts/diagnose_vpfp_drift_codec.py")
codec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(codec)


@pytest.fixture(autouse=True)
def threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def state(degree=4):
    a = np.zeros(len(codec.mode_pairs(degree)))
    pairs = codec.mode_pairs(degree)
    a[pairs.index((degree, 0))] = .7
    a[pairs.index((degree-1, 0))] = .6
    a[0] = .4
    return a / np.linalg.norm(a)


@pytest.mark.parametrize("degree", (0, 1, 4, 14))
def test_zero_shift_complete_abi_and_expanded_intermediate_are_identity(degree):
    for middle in (degree, degree+4):
        overlap = codec.translation_overlap(degree, middle, [0., 0.], .9, middle+2)
        expected = np.zeros_like(overlap)
        expected[:len(codec.mode_pairs(degree))] = np.eye(len(codec.mode_pairs(degree)))
        np.testing.assert_allclose(overlap, expected, atol=8.e-14, rtol=8.e-14)
    rng = np.random.default_rng(5)
    a = rng.normal(size=len(codec.mode_pairs(degree)))
    report = codec.roundtrip(a, .9, rho=1.7, floor_mixture=.01, zero_shift=True)
    assert report["raw_amplitude_relative_l2_error"] < 2.e-13
    assert report["returned_norm_retained_fraction"] == pytest.approx(1., abs=2.e-13)
    assert np.max(np.abs(report["momentum_error"])) < 2.e-12
    assert abs(report["kinetic_energy_error"]) < 2.e-12


def test_translation_ground_state_exact_series_sign_and_norm_tail():
    # phi_0(v+d) has c_n=e^-lambda/2 (-d/(2sqrt(T)))^n/sqrt(n!).
    theta, displacement, degree = .8, .7, 8
    overlap = codec.translation_overlap(0, degree, [displacement, 0.], theta, degree+2)[:, 0]
    lam = displacement**2/(4*theta)
    expected = np.array([math.exp(-lam/2)*(-displacement/(2*math.sqrt(theta)))**nx/math.sqrt(math.factorial(nx))
        if ny == 0 else 0. for nx, ny in codec.mode_pairs(degree)])
    np.testing.assert_allclose(overlap, expected, atol=2.e-15, rtol=2.e-12)
    retained = math.exp(-lam)*sum(lam**n/math.factorial(n) for n in range(degree+1))
    assert overlap @ overlap == pytest.approx(retained, abs=2.e-14)


def test_finite_projection_is_not_reversible_and_higher_order_recovers_without_dropping_original():
    a = state()
    results = [codec.roundtrip(a, 1., intermediate_degree=degree) for degree in (4, 8, 12)]
    assert results[0]["raw_amplitude_relative_l2_error"] > .01
    assert results[0]["returned_norm_retained_fraction"] < .99
    assert results[0]["original"]["highest_degree_energy_fraction"] > .4
    assert results[2]["raw_amplitude_relative_l2_error"] < results[1]["raw_amplitude_relative_l2_error"] < results[0]["raw_amplitude_relative_l2_error"]
    assert results[1]["centered_energy_above_original_degree_fraction"] > 0
    with pytest.raises(ValueError, match="may not truncate"):
        codec.roundtrip(a, 1., intermediate_degree=3)


def test_quadrature_convergence_is_separate_from_truncation_error():
    a = state()
    low = codec.roundtrip(a, 1.1, rho=.7, intermediate_degree=4, quadrature_order=6)
    high = codec.roundtrip(a, 1.1, rho=.7, intermediate_degree=4, quadrature_order=14)
    assert low["raw_amplitude_relative_l2_error"] > .01
    assert low["raw_amplitude_relative_l2_error"] == pytest.approx(high["raw_amplitude_relative_l2_error"], abs=3.e-14)
    assert abs(low["velocity_quadrature_mass_error"]) < 2.e-14
    assert abs(low["velocity_quadrature_energy_error"]) < 1.e-13
    with pytest.raises(ValueError, match="does not integrate"):
        codec.translation_overlap(4, 4, [.1, .2], 1., 3)


def test_analytic_moments_include_top_mode_excursions_and_floor():
    # For pure |N,0>, <vx^2>=T(2N+1), including raising OUTSIDE degree N.
    degree, theta, floor, rho = 5, .8, .2, 1.3
    a = np.zeros(len(codec.mode_pairs(degree)))
    a[codec.mode_pairs(degree).index((degree, 0))] = 1.
    result = codec.amplitude_moments(a, theta, floor_mixture=floor, rho=rho)
    assert result["raw_second_moment"][0][0] == pytest.approx((1-floor)*theta*(2*degree+1)+floor*theta)
    assert result["raw_second_moment"][1][1] == pytest.approx(theta)
    quad = codec.quadrature_moments(a, theta, degree+2, rho=rho, floor_mixture=floor)
    assert quad["kinetic_energy"] == pytest.approx(result["kinetic_energy"], abs=2.e-13)
    np.testing.assert_allclose(result["momentum"], [0., 0.], atol=1.e-15)


def test_floor_residual_and_shift_are_derived_from_same_state():
    a = np.array([1., .1, .3])
    result = codec.roundtrip(a, .9, floor_mixture=.2, intermediate_degree=12)
    u = np.array(result["original"]["velocity"])
    np.testing.assert_allclose(result["shift_from_same_state_physical_velocity"], u)
    np.testing.assert_allclose(result["ideal_center_velocity_with_unchanged_floor"], .2*u, atol=1.e-16)
    np.testing.assert_allclose(result["centered"]["velocity"], .2*u, atol=1.e-11)
    assert np.linalg.norm(u) > .1


@pytest.mark.parametrize("amplitude", (np.zeros(3), np.array([1., np.nan, 0.]), np.ones(4)))
def test_invalid_or_zero_norm_state_is_not_silently_repaired(amplitude):
    with pytest.raises(ValueError):
        codec.roundtrip(amplitude, 1.)


def test_exact_moments_match_existing_decoder_recurrence():
    from pde_agent_verifier_solver.plugins.vpfp.perturbation_backend import _legacy_api
    _legacy_api()
    macros = importlib.import_module("agent_verifier.vpfp.continuous_macros")
    a = np.random.default_rng(1).normal(size=len(codec.mode_pairs(4)))
    result = codec.amplitude_moments(a, .9, rho=1.2, floor_mixture=.07)
    actual = macros.torch_squared_hermite_moments(torch, torch.tensor(a), modes=codec.mode_pairs(4),
        temperature=.9, floor_mixture=.07)
    np.testing.assert_allclose(result["velocity"], actual["mean_velocity"].numpy(), rtol=2.e-14, atol=2.e-14)
    assert result["kinetic_energy"] == pytest.approx(1.2*actual["kinetic_energy"].item(), rel=2.e-14)


def test_local_amplitudes_use_real_decoder_scales_time_domain_and_fallback():
    from pde_agent_verifier_solver.plugins.vpfp.perturbation_backend import _legacy_api
    api = _legacy_api()
    hermite = importlib.import_module("agent_verifier.vpfp.continuous_hermite")
    cfg = replace(hermite.VPFPContinuousHermiteConfig(), latent_dim=1, latent_input_scale=3.7,
        structured_local_scale=2.4, micro_latent_dim=2, context_dim=4, hidden_width=4, hidden_layers=1,
        rho_max_mode_x=1, rho_max_mode_y=1, rho_positivity_gauge="phase_invariant_group_l1_v2",
        spatial_feature_modes=1, time_feature_modes=1, hermite_total_degree=1, coefficient_width=4,
        coefficient_blocks=1, structured_spectral_latent=True, structured_kinetic_global_dim=2,
        structured_local_channels=3, structured_local_spatial_mode=1,
        structured_time_basis="piecewise_cubic_hermite_v1", structured_time_control_count=3,
        structured_direct_coefficients=True, structured_residual_scale=0., time_domain=(2.,5.), dtype="float64")
    layout = cfg.structured_latent_layout()
    cfg = replace(cfg, latent_dim=layout["total_latent_dim"])
    decoder = api.build_deterministic_direct_model(cfg, device="cpu")
    raw = np.zeros(cfg.latent_dim, dtype=np.float32)
    controls = raw[layout["rho_latent_dim"]+cfg.structured_kinetic_global_dim:].reshape(3,6,9)
    controls[:, :3, 0] = np.array([1.2, .3, -.4])[:, None]
    points, amplitudes, density, digest = codec.local_amplitudes(decoder, raw, [.03, .9])
    expected = np.array([1.2, .3, -.4], dtype=np.float32).astype(float)
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(amplitudes, np.broadcast_to(expected, amplitudes.shape), rtol=2.e-14, atol=2.e-14)
    assert min(points[:,0]) == 2. and max(points[:,0]) == 5.
    assert len(digest) == 64 and np.all(density > 0)
    _, fallback, _, _ = codec.local_amplitudes(decoder, np.zeros_like(raw), [.03,.9])
    np.testing.assert_array_equal(fallback, np.broadcast_to([1.,0.,0.], fallback.shape))


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    root, output, config = tmp_path / "corpus", tmp_path / "output", tmp_path / "config.json"
    root.mkdir(); config.write_text("{}")
    counts = {role: 8 for role in ("train","validation","calibration","reference_a","reference_b")}
    edges = [np.linspace(.02,.08,5).tolist(), [.8,1.,1.2]]
    plan = {"regime_id": "weak_fixture", "bin_edges": edges, "split_counts": counts, "splits": {}}
    manifest = {"schema": "vpfp_stratified_coefficient_corpus_v1", "status": "complete", "regime_id": "weak_fixture",
        "latent_dimension": 4, "split_counts": counts, "bin_edges": edges, "splits": {}}
    for number, role in enumerate(("train", "validation")):
        directory = root / role; directory.mkdir()
        ids = np.arange(number*10,number*10+8,dtype=np.int64)
        conditions = np.array([[nu,theta] for nu in (.025,.04,.055,.07) for theta in (.9,1.1)],dtype=np.float32)
        case_ids = np.array([f"weak_fixture:{seed}" for seed in ids])
        arrays = {"coefficients": np.ones((8,4),dtype=np.float32), "conditions": conditions,
            "equation_seeds": ids, "stratum_ids": np.arange(8,dtype=np.int64), "case_ids": case_ids}
        files = {}
        for key, value in arrays.items():
            path = directory / f"{key}.npy"; np.save(path,value,allow_pickle=False)
            files[key] = {"path": str(path.relative_to(root)), "sha256": codec._file_sha(path),
                "shape": list(value.shape), "dtype": value.dtype.str}
        manifest["splits"][role] = {"files": files, "status":"complete", "rows_written":8, "count":8}
        plan["splits"][role] = [{"equation_seed":int(ids[i]),"conditions":conditions[i].tolist(),
            "stratum_id":i,"case_id":str(case_ids[i])} for i in range(8)]
    # The diagnostic must never try to open any of these unavailable arrays.
    for role in ("calibration","reference_a","reference_b"):
        manifest["splits"][role] = {"files":{"coefficients":{"path":"DO-NOT-OPEN"}}}
    codec._write(root / "seed-plan.json", plan)
    manifest["seed_plan"] = {"path":"seed-plan.json", "sha256":codec._file_sha(root / "seed-plan.json")}
    manifest["manifest_sha256"] = codec.sha256(codec._json_bytes(manifest)).hexdigest()
    codec._write(root / "manifest.json",manifest)
    monkeypatch.setenv("SLURM_JOB_ID","synthetic-only")
    monkeypatch.setattr(codec,"DEGREE_OFFSETS",(0,2))
    fake = SimpleNamespace(config=SimpleNamespace(hermite_total_degree=1,velocity_floor_mixture=.01))
    monkeypatch.setattr(codec,"build_decoder",lambda *a:fake)
    item = SimpleNamespace(root=root,output=output,config=config,fail=False,calls=0)
    def local(*args):
        item.calls += 1
        a = np.array([[1.,.1,.3]])
        if item.fail and item.calls == 1:a*=0
        return np.array([[0.,.1,.2,0.,0.]]),a,np.array([1.]),"f"*64
    monkeypatch.setattr(codec,"local_amplitudes",local)
    return item


def test_development_loader_never_opens_reference_arrays(corpus):
    manifest,splits,frozen = codec.load_development(corpus.root)
    assert set(splits)=={"train","validation"}
    assert len(frozen)==12 and not any("reference" in p or "calibration" in p for p in frozen)
    assert codec.select_rows(splits["train"],manifest["bin_edges"])==list(range(8))


def test_run_records_complete_local_scope_and_no_codec_qualification(corpus):
    result=codec.run(corpus.root,corpus.config,corpus.output,"cpu")
    assert result["execution_passed"] is True and result["selected_case_count"]==16 and result["local_point_count"]==16
    assert not result["codec_qualified"] and not result["spatial_or_time_codec_implemented"]
    assert result["new_pde_solves"]==0 and not result["physical_reference_arrays_read"]
    assert len(result["cases"][0]["rows"][0]["roundtrips"])==6
    assert (corpus.output/"frozen-plan.json").exists() and (corpus.output/"partial-cases.json").exists()


def test_invalid_case_is_preserved_and_other_cases_continue(corpus):
    corpus.fail=True
    result=codec.run(corpus.root,corpus.config,corpus.output,"cpu")
    assert result["execution_passed"] is False and len(result["failures"])==1
    assert len(result["cases"])==16 and result["cases"][0]["execution_passed"] is False
    assert result["cases"][-1]["execution_passed"] is True
    assert result["partial_artifacts_preserved"]


def test_manifest_or_selected_file_drift_is_rejected(corpus):
    path=corpus.root/"train/conditions.npy"
    np.save(path,np.zeros((8,2),dtype=np.float32),allow_pickle=False)
    with pytest.raises(ValueError,match="SHA"):
        codec.load_development(corpus.root)


def test_no_slurm_and_existing_or_inside_source_output_are_rejected(corpus,monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID")
    with pytest.raises(RuntimeError,match="Slurm"):
        codec.run(corpus.root,corpus.config,corpus.output,"cpu")
    assert not corpus.output.exists()
    monkeypatch.setenv("SLURM_JOB_ID","synthetic")
    with pytest.raises(ValueError,match="fresh"):
        codec.run(corpus.root,corpus.config,corpus.root/"nested","cpu")
    corpus.output.symlink_to(corpus.root/"missing")
    with pytest.raises(ValueError,match="fresh"):
        codec.run(corpus.root,corpus.config,corpus.output,"cpu")
