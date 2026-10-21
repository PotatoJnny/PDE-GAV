"""Bounded synthetic tests only; no distribution qualification or PDE solve."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pde_agent_verifier_solver.plugins.vpfp.structured_diffusion import VPFPStructuredEDMConfig
from pde_agent_verifier_solver.plugins.vpfp.temporal_covariance_diffusion import (
    CHECKPOINT_SCHEMA, VPFPTemporalCovarianceNormalizer, VPFPTemporalCovarianceEDM)
from pde_agent_verifier_solver.plugins.vpfp.value_control_diffusion import (
    VPFPValueEDMConfig, lift_value_controls, _value_split)

ROOT = Path(__file__).resolve().parents[1]
loader = importlib.util.spec_from_file_location("weak_structure_diagnostic_test", ROOT / "scripts/diagnose_vpfp_weak_structure_prior.py")
diagnostic = importlib.util.module_from_spec(loader)
loader.loader.exec_module(diagnostic)


@pytest.fixture(autouse=True)
def threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def config():
    return VPFPValueEDMConfig.from_full_config(VPFPStructuredEDMConfig(
        control_count=3, hermite_channels=3, rho_max_mode=3, kinetic_max_mode=3,
        inactive_dim=2, grid_size=7, width=8, depth=1))


def conditions():
    return np.asarray([[nu, theta] for nu in (.025, .04, .055, .07) for theta in (.9, 1.1)], dtype=np.float32)


def test_validation_selection_is_one_per_joint_stratum_and_id_stable():
    cond = np.repeat(conditions(), 2, axis=0)
    ids = np.arange(32, 48, dtype=np.int64)[::-1].copy()
    edges = (np.linspace(.02, .08, 5), np.linspace(.8, 1.2, 3))
    chosen = diagnostic.select_validation_rows(cond, ids, edges)
    assert chosen.tolist() == list(range(1, 16, 2))
    perm = np.random.default_rng(4).permutation(16)
    assert ids[chosen].tolist() == ids[perm][diagnostic.select_validation_rows(cond[perm], ids[perm], edges)].tolist()
    with pytest.raises(ValueError, match="no case"):
        diagnostic.select_validation_rows(cond[:14], ids[:14], edges)
    with pytest.raises(ValueError, match="unique"):
        diagnostic.select_validation_rows(cond, np.zeros(16, dtype=np.int64), edges)
    with pytest.raises(ValueError, match="exactly eight"):
        diagnostic.select_validation_rows(cond, ids, edges, sample_count=16)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_exact_disjoint_box_bands_and_original_units(dtype):
    cfg = config()
    target = torch.zeros(2, cfg.latent_dimension, dtype=dtype)
    prediction = target.clone()
    rho, kinetic = _value_split(prediction, cfg)
    masks = diagnostic._mode_bands(3, constant=True)
    assert [int(mask.sum()) for mask in masks.values()] == [1, 24, 24]
    assert torch.stack(tuple(masks.values())).sum(dim=0).eq(1).all()
    kinetic[..., masks["dc"]] = 2.
    kinetic[..., masks["low_box_non_dc"]] = 3.
    kinetic[..., masks["rest"]] = 4.
    rho.fill_(5.)
    report = diagnostic.value_band_diagnostics(prediction, target, cfg)
    assert report["rho"]["dc"] == {"available": False, "fourier_value_count": 0}
    assert [report["kinetic"][k]["rmse"] for k in masks] == [2., 3., 4.]
    assert report["rho"]["rest"]["rmse"] == 5.
    assert report["kinetic"]["dc"]["per_control_time_rmse"] == [2.] * 3
    assert report["kinetic"]["dc"]["per_case_rmse"] == [2.] * 2
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_neighbor_coupling_has_correct_pair_order_sign_and_gauge_invariance(dtype):
    cfg = config()
    # Total-degree ordering is (0,0), (0,1), (1,0), not swapped x/y.
    target = torch.zeros(1, cfg.latent_dimension, dtype=dtype)
    _, a = _value_split(target, cfg)
    a[:, 0, :, 0] = 1.
    a[:, 2, :, 0] = 1.
    report = diagnostic.amplitude_neighbor_diagnostics(-7 * target, target, cfg, [[.05, 1.]])
    assert report["definition"].endswith("NOT_physical_j")
    assert report["axes"]["x"]["target_coherent_mean_rms"] == pytest.approx(1.)
    assert report["axes"]["y"]["target_coherent_mean_rms"] == 0.
    assert report["axes"]["x"]["coherent_mean_rmse"] == pytest.approx(0., abs=1.e-12)
    prediction = target.clone()
    _value_split(prediction, cfg)[1][:, 2] *= -1
    report = diagnostic.amplitude_neighbor_diagnostics(prediction, target, cfg, [[.05, 1.]])
    assert report["axes"]["x"]["pointwise_rmse"] == pytest.approx(2.)
    empty = diagnostic.amplitude_neighbor_diagnostics(target * 0, target * 0, cfg, [[.05, 1.]])
    assert empty["target_zero_norm_points"] == 3 * 7 * 7
    assert empty["axes"]["x"]["pointwise_rmse"] == 0.
    json.dumps(empty, allow_nan=False)


def test_common_noise_is_per_id_order_independent_without_global_rng_side_effect():
    before = torch.random.get_rng_state().clone()
    noise, seeds = diagnostic._noise_for_ids([21, 7], 30, 500)
    reordered, other = diagnostic._noise_for_ids([7, 21], 30, 500)
    assert torch.equal(noise.flip(0), reordered) and seeds[::-1] == other
    assert torch.equal(before, torch.random.get_rng_state())
    assert not torch.equal(noise[0], noise[1])


@pytest.fixture
def native(tmp_path, monkeypatch):
    cfg = config()
    train_values = torch.randn(8, cfg.latent_dimension, generator=torch.Generator().manual_seed(1))
    val_values = torch.randn(8, cfg.latent_dimension, generator=torch.Generator().manual_seed(2))
    _value_split(train_values, cfg)[1][:, 0, :, 0] += 4.
    _value_split(val_values, cfg)[1][:, 0, :, 0] += 4.
    train_raw, raw = lift_value_controls(train_values, cfg), lift_value_controls(val_values, cfg)
    train_ids, ids = np.arange(100, 108), np.arange(200, 208)
    normalizer = VPFPTemporalCovarianceNormalizer.fit(cfg, train_raw, sample_ids=train_ids.tolist(), covariance_strength=.5)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    files = {}
    for name, value in (("coefficients", raw.numpy()), ("conditions", conditions()), ("equation_seeds", ids)):
        path = corpus / (name + ".npy")
        np.save(path, value, allow_pickle=False)
        files[name] = {"path": path.name, "sha256": diagnostic._file_sha(path)}
    manifest = {"manifest_sha256": "b"*64, "base_decoder_sha256": "a"*64,
        "collision_nu_range": [.02, .08], "temperature_range": [.8, 1.2], "splits": {"validation": {"files": files}}}
    manifest["provenance"] = {"decoder_config": {"hermite_total_degree": 1, "structured_spectral_latent": True,
        "structured_direct_coefficients": True, "structured_residual_scale": 0.,
        "structured_time_basis": "piecewise_cubic_hermite_v1", "structured_time_control_count": 3,
        "structured_local_channels": 3, "structured_local_spatial_mode": 3, "dtype": "float32",
        "latent_input_scale": 1., "structured_local_scale": 1., "velocity_floor_mixture": 1.e-5}}
    (corpus / "manifest.json").write_text(json.dumps(manifest))
    splits = {"validation": SimpleNamespace(latents=raw.numpy(), conditions=conditions(), sample_ids=ids),
        "train": SimpleNamespace(latents=train_raw.numpy(), sample_ids=train_ids)}
    monkeypatch.setattr(diagnostic, "read_manifest", lambda *a, **k: deepcopy(manifest))
    monkeypatch.setattr(diagnostic, "load_split", lambda root, name, **kwargs: splits[name])
    monkeypatch.setattr(VPFPTemporalCovarianceEDM, "sample", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no generative sampling allowed")))
    entries = {}
    for label in ("baseline", "architecture"):
        model = VPFPTemporalCovarianceEDM(cfg)
        payload = {"schema": CHECKPOINT_SCHEMA, "config": cfg.to_dict(), "model": model.state_dict(),
            "normalizer": normalizer.to_payload(), "base_decoder_sha256": "a"*64,
            "corpus_manifest_sha256": "b"*64, "training": {"step": 1, "synthetic_fixture": True}}
        path = tmp_path / (label + ".pt")
        torch.save(payload, path)
        entries[label] = {"checkpoint": str(path), "sha256": diagnostic._file_sha(path), "snapshot_kind": "best"}
    return SimpleNamespace(config=cfg, corpus=corpus, output=tmp_path / "diagnostic", entries=entries, manifest=manifest)


def test_tiny_native_models_share_raw_inputs_noise_geometry_and_leave_source_unchanged(native):
    frozen = {str(p): diagnostic._file_sha(p) for p in native.corpus.iterdir()}
    report = diagnostic.run_structure_denoising_diagnostic(native.entries, native.corpus, native.output,
        seed=10, device="cpu", expected_corpus_sha256="b"*64)
    assert report["execution_passed"], report.get("error")
    assert report["sigmas"] == [.03, .1, .3]
    assert report["validation_ids"] == list(range(200, 208))
    assert report["generated_samples"] == report["new_pde_solves"] == 0
    assert not report["blind_test"] and not report["used_for_checkpoint_selection"] and not report["admitted"]
    first, second = report["models"].values()
    assert first == second  # Native zero-initialized output: identical denoiser.
    assert set(first["sigmas"]) == {"0.03", "0.1", "0.3"}
    assert report["all_inputs_revalidated"] and (native.output / "result.json").is_file()
    diagnostic._assert_hashes(frozen)
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("failure", ("geometry", "nan", "source_mutation", "checkpoint_mutation"))
def test_failure_preserves_completed_model_and_partial_report(native, monkeypatch, failure):
    original = diagnostic.load_vpfp_diffusion_sampler
    calls = []
    def load(*args, **kwargs):
        sampler = original(*args, **kwargs)
        calls.append(args[0])
        if len(calls) == 2:
            if failure == "geometry":
                sampler.normalizer.covariance_strength = 1.
            elif failure == "nan":
                sampler.model.denoise = lambda noisy, *a: noisy * float("nan")
            else:
                target = native.corpus / "conditions.npy" if failure == "source_mutation" else Path(args[0])
                target.write_bytes(b"synthetic input corruption")
        return sampler
    monkeypatch.setattr(diagnostic, "load_vpfp_diffusion_sampler", load)
    report = diagnostic.run_structure_denoising_diagnostic(native.entries, native.corpus, native.output, seed=10, device="cpu")
    assert not report["execution_passed"] and report["partial_artifacts_preserved"]
    assert "baseline" in report["models"] and len(report["models"]["baseline"]["sigmas"]) == 3
    assert (native.output / "result.partial.json").is_file() and (native.output / "result.json").is_file()
    assert ("frozen diagnostic input changed" in report["error"]) if failure.endswith("mutation") else bool(report["error"])


def test_refuses_nonhalf_geometry_and_wrong_corpus_before_comparison(native):
    report = diagnostic.run_structure_denoising_diagnostic(native.entries, native.corpus, native.output,
        seed=10, device="cpu", expected_corpus_sha256="c"*64)
    assert not report["execution_passed"] and not report["models"]
    assert "corpus differs" in report["error"]
    with pytest.raises(ValueError, match="fresh"):
        diagnostic.run_structure_denoising_diagnostic(native.entries, native.corpus, native.output, seed=10, device="cpu")
    with pytest.raises(ValueError, match="eight cases"):
        diagnostic.run_structure_denoising_diagnostic(native.entries, native.corpus, native.output.parent / "new", seed=10, device="cpu", sample_count=16)


@pytest.mark.parametrize("branch", ("lowmode_coherent", "branch_identity_control"))
def test_actual_structure_native_checkpoint_joins_same_paired_diagnostic(native, branch):
    from pde_agent_verifier_solver.plugins.vpfp.structure_prior_diffusion import make_structure_prior_model
    original = torch.load(native.entries["baseline"]["checkpoint"], map_location="cpu", weights_only=True)
    model = make_structure_prior_model(native.config, branch)
    payload = dict(original, schema=model.checkpoint_schema, model=model.state_dict())
    path = native.output.parent / (branch + ".pt")
    torch.save(payload, path)
    native.entries[branch] = {"checkpoint": str(path), "sha256": diagnostic._file_sha(path)}
    report = diagnostic.run_structure_denoising_diagnostic(native.entries, native.corpus, native.output, seed=10, device="cpu")
    assert report["execution_passed"], report.get("error")
    assert len(report["models"]) == 3
    # All native models begin with zero final outputs; added branch is zero.
    assert report["models"][branch] == report["models"]["baseline"]


def test_valid_but_different_train_geometry_is_not_silently_compared(native):
    training = diagnostic.load_split(native.corpus, "train")
    normalizer = VPFPTemporalCovarianceNormalizer.fit(native.config,
        torch.from_numpy(training.latents.copy()) * 2, sample_ids=training.sample_ids.tolist(), covariance_strength=.5)
    path = Path(native.entries["architecture"]["checkpoint"])
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["normalizer"] = normalizer.to_payload()
    torch.save(payload, path)
    native.entries["architecture"]["sha256"] = diagnostic._file_sha(path)
    report = diagnostic.run_structure_denoising_diagnostic(native.entries, native.corpus, native.output, seed=10, device="cpu")
    assert not report["execution_passed"]
    assert "identical normalizer payloads" in report["error"]
    assert "baseline" in report["models"] and "architecture" not in report["models"]


@pytest.mark.parametrize("change", ({"structured_direct_coefficients": False}, {"structured_residual_scale": .01},
    {"structured_local_channels": 4}, {"structured_time_basis": "global_chebyshev_v1"}))
def test_proxy_rejects_decoder_with_unmodeled_base_or_wrong_layout(native, change):
    native.manifest["provenance"]["decoder_config"].update(change)
    with pytest.raises(ValueError, match="qualified direct"):
        diagnostic._decoder_amplitude_contract(native.manifest, native.config)


def test_real_legacy_direct_decoder_moments_match_proxy_at_control_knots():
    # Actual small resident decoder, not just an algebraic surrogate. No solve.
    from pde_agent_verifier_solver.plugins.vpfp.perturbation_backend import _legacy_api
    _legacy_api()
    from agent_verifier.vpfp.continuous_hermite import VPFPContinuousHermiteConfig, VPFPContinuousHermiteField
    from agent_verifier.vpfp.continuous_macros import torch_squared_hermite_moments
    cfg = config()
    legacy = VPFPContinuousHermiteConfig(latent_dim=cfg.full_dimension, latent_input_scale=1.7,
        context_dim=8, hidden_width=8, hidden_layers=1, coefficient_width=8, coefficient_blocks=1,
        rho_max_mode_x=3, rho_max_mode_y=3, hermite_total_degree=1, structured_spectral_latent=True,
        structured_kinetic_global_dim=2, structured_local_channels=3, structured_local_spatial_mode=3,
        structured_time_basis="piecewise_cubic_hermite_v1", structured_time_control_count=3,
        structured_local_scale=.6, structured_direct_coefficients=True, structured_residual_scale=0.,
        time_domain=cfg.time_domain, dtype="float64")
    model = VPFPContinuousHermiteField.build(legacy)
    raw = torch.randn(1, cfg.latent_dimension, dtype=torch.float64, generator=torch.Generator().manual_seed(25))
    _, kinetic = _value_split(raw, cfg)
    kinetic[:, 0, :, 0] += 3.
    full = lift_value_controls(raw, cfg)
    grid = 7
    times = torch.linspace(*cfg.time_domain, 3, dtype=torch.float64)
    x = torch.arange(grid, dtype=torch.float64) * legacy.equation.x_domain[0] / grid
    y = torch.arange(grid, dtype=torch.float64) * legacy.equation.x_domain[1] / grid
    tt, xx, yy = torch.meshgrid(times, x, y, indexing="ij")
    points = torch.stack((tt.flatten(), xx.flatten(), yy.flatten(), torch.zeros(tt.numel()), torch.zeros(tt.numel())), dim=1)
    with torch.no_grad():
        fields = model.fields(points, full[0], equation_condition=[.05, 1.1])
        moments = torch_squared_hermite_moments(torch, fields["hermite_coefficients"],
            modes=model.hermite_modes, temperature=torch.full((len(points),), 1.1, dtype=torch.float64), floor_mixture=0.)
    proxy = diagnostic.amplitude_neighbor_diagnostics(raw, raw, cfg, [[.05, 1.1]], decoder_amplitude_scale=1.7*.6/7)
    for axis, name in enumerate(("x", "y")):
        actual = moments["mean_velocity"][:, axis].reshape(3, grid, grid)
        expected = proxy["axes"][name]
        assert expected["target_pointwise_rms"] == pytest.approx(float(diagnostic._rms(actual)), rel=2.e-12)
        assert expected["target_coherent_mean_rms"] == pytest.approx(float(diagnostic._rms(actual.mean(dim=(-2,-1)))), rel=2.e-12)
    tiny = diagnostic.amplitude_neighbor_diagnostics(raw * 1.e-30, raw, cfg, [[.05, 1.1]])
    assert tiny["prediction_decoder_ground_state_fallback_points"] == 3*7*7
    assert tiny["axes"]["x"]["prediction_pointwise_rms"] == 0.
