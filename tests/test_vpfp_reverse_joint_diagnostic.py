"""Synthetic native-sampler invariance and bounded orchestration tests only."""
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
    CHECKPOINT_SCHEMA, VPFPTemporalCovarianceEDM, VPFPTemporalCovarianceNormalizer,
)
from pde_agent_verifier_solver.plugins.vpfp.value_control_diffusion import (
    VPFPValueEDMConfig, _value_split, lift_value_controls,
)

ROOT = Path(__file__).resolve().parents[1]
loader = importlib.util.spec_from_file_location("reverse_joint_diagnostic_test", ROOT / "scripts/diagnose_vpfp_reverse_joint.py")
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
        control_count=3, hermite_channels=3, rho_max_mode=1, kinetic_max_mode=1,
        inactive_dim=2, grid_size=3, width=8, depth=1, time_domain=(2., 6.)))


def conditions():
    return np.asarray([[nu, theta] for nu in (.025, .04, .055, .07) for theta in (.9, 1.1)], dtype=np.float32)


@pytest.mark.parametrize("steps", (64, 128, 256))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_observer_is_bitwise_native_with_no_extra_evaluations_or_rng_changes(steps, dtype):
    model = VPFPTemporalCovarianceEDM(config()).to(dtype=dtype).eval()
    # Nonzero learned branch, not merely the initial Gaussian skip baseline.
    with torch.no_grad():
        model.output.weight.fill_(.001)
        model.output.bias.fill_(.002)
    original = model.denoise
    counts = []
    def counted(*args):
        counts.append(1)
        return original(*args)
    model.denoise = counted
    rng = torch.random.get_rng_state().clone()
    with torch.inference_mode():
        native = model.sample(conditions()[:1], [19], steps=steps, churn=0.)
    assert len(counts) == 2 * steps - 1
    counts.clear()
    observed, trace = diagnostic.observe_native_sample(model, conditions()[:1], [19], steps=steps)
    assert torch.equal(observed, native)
    assert model.denoise is counted
    assert len(counts) == trace["denoiser_evaluations"] == 2 * steps - 1
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(row["xhat0"].shape == (1, model.config.latent_dimension) for row in trace["stages"])
    assert trace["stages"][0]["native_step"] == 0
    assert trace["stages"][-1]["native_step"] == steps - 1
    assert trace["stages"][0]["actual_sigma"] == trace["sigma_grid"][0]


def test_common_initial_noise_is_step_independent_and_per_case_order_invariant():
    model = VPFPTemporalCovarianceEDM(config()).eval()
    seeds = diagnostic._noise_seeds([200, 100], 9)
    first, a = diagnostic.observe_native_sample(model, conditions()[:2], seeds, steps=4)
    _, b = diagnostic.observe_native_sample(model, conditions()[:2], seeds, steps=7)
    reordered, c = diagnostic.observe_native_sample(model, conditions()[:2][::-1].copy(), seeds[::-1], steps=4)
    assert a["initial_noisy_states_sha256"] == b["initial_noisy_states_sha256"]
    assert a["initial_noisy_state_per_case_sha256"][::-1] == c["initial_noisy_state_per_case_sha256"]
    assert torch.equal(first.flip(0), reordered)
    assert "denoise" not in model.__dict__
    assert diagnostic._noise_seeds([100, 200], 9) == seeds[::-1]
    assert not set(seeds).intersection(diagnostic._noise_seeds([100, 200], 10))


def test_observer_restores_instance_method_after_failure():
    model = VPFPTemporalCovarianceEDM(config()).eval()
    def failure(*args):
        raise RuntimeError("synthetic denoise failure")
    model.denoise = failure
    with pytest.raises(RuntimeError, match="synthetic"):
        diagnostic.observe_native_sample(model, conditions()[:1], [2], steps=3)
    assert model.denoise is failure
    model.train()
    with pytest.raises(ValueError, match="eval-mode"):
        diagnostic.observe_native_sample(model, conditions()[:1], [2], steps=3)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_coefficient_derivative_uses_physical_time_not_tau_and_removes_no_band(dtype):
    cfg = config()
    values = torch.zeros(2, cfg.latent_dimension, dtype=dtype)
    rho, kinetic = _value_split(values, cfg)
    rho[:] = torch.tensor([0., 4., 8.], dtype=dtype)[None, :, None]
    kinetic[..., 0] = torch.tensor([0., 6., 12.], dtype=dtype)[None, None, :]
    report = diagnostic.coefficient_time_statistics(values, cfg)
    assert report["physical_time"] == [2., 4., 6.]
    assert report["groups"]["rho"]["low_box_non_dc"]["dt_rms"] == 2.
    assert report["groups"]["kinetic"]["dc"]["dt_rms"] == 3.
    assert report["groups"]["kinetic"]["dc"]["early_dt_rms"] == 3.
    assert report["groups"]["kinetic"]["dc"]["late_dt_rms"] == 3.
    assert report["groups"]["rho"]["dc"] == {"available": False}
    assert "NOT_physical" in report["definition"]
    json.dumps(report, allow_nan=False)


def test_proxy_summary_retains_fallback_without_fake_reconstruction_error():
    cfg = config()
    values = torch.zeros(1, cfg.latent_dimension)
    report = diagnostic._neighbor_summary(values, cfg, conditions()[:1], 1 / 3)
    assert report["zero_norm_points"] == report["decoder_ground_state_fallback_points"] == 27
    assert report["axes"]["x"] == {"coherent_mean_rms": 0., "pointwise_rms": 0.}
    assert "target" not in json.dumps(report) and "rmse" not in json.dumps(report)


@pytest.fixture
def native(tmp_path, monkeypatch):
    cfg = config()
    train_values = torch.randn(8, cfg.latent_dimension, generator=torch.Generator().manual_seed(1))
    val_values = torch.randn(8, cfg.latent_dimension, generator=torch.Generator().manual_seed(2))
    train_raw, val_raw = lift_value_controls(train_values, cfg), lift_value_controls(val_values, cfg)
    train_ids, val_ids = np.arange(100, 108, dtype=np.int64), np.arange(200, 208, dtype=np.int64)
    norm = VPFPTemporalCovarianceNormalizer.fit(cfg, train_raw, sample_ids=train_ids.tolist(), covariance_strength=.5)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    manifest = {"manifest_sha256": "b"*64, "base_decoder_sha256": "a"*64,
                "collision_nu_range": [.02, .08], "temperature_range": [.8, 1.2], "splits": {}}
    splits = {}
    for split, raw, ids in (("train", train_raw, train_ids), ("validation", val_raw, val_ids)):
        files = {}
        for name, array in (("coefficients", raw.numpy()), ("conditions", conditions()), ("equation_seeds", ids)):
            path = corpus / f"{split}-{name}.npy"
            np.save(path, array, allow_pickle=False)
            files[name] = {"path": path.name, "sha256": diagnostic._file_sha(path)}
        manifest["splits"][split] = {"files": files}
        splits[split] = SimpleNamespace(latents=raw.numpy(), conditions=conditions(), sample_ids=ids)
    manifest["provenance"] = {"decoder_config": {"hermite_total_degree": 1, "structured_spectral_latent": True,
        "structured_direct_coefficients": True, "structured_residual_scale": 0.,
        "structured_time_basis": "piecewise_cubic_hermite_v1", "structured_time_control_count": 3,
        "structured_local_channels": 3, "structured_local_spatial_mode": 1, "dtype": "float32",
        "latent_input_scale": 1., "structured_local_scale": 1., "velocity_floor_mixture": 1.e-5}}
    (corpus / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(diagnostic, "read_manifest", lambda *a, **k: deepcopy(manifest))
    monkeypatch.setattr(diagnostic, "load_split", lambda root, name, **k: splits[name])
    entries = {}
    for label in ("seed-0", "seed-1"):
        model = VPFPTemporalCovarianceEDM(cfg)
        payload = {"schema": CHECKPOINT_SCHEMA, "config": cfg.to_dict(), "model": model.state_dict(),
                   "normalizer": norm.to_payload(), "base_decoder_sha256": "a"*64,
                   "corpus_manifest_sha256": "b"*64, "training": {"step": 1, "synthetic_fixture": True}}
        path = tmp_path / (label + ".pt")
        torch.save(payload, path)
        entries[label] = {"checkpoint": str(path), "sha256": diagnostic._file_sha(path)}
    # The complete orchestration tests use the actual native four-step path;
    # the separate invariance tests above cover real 64/128/256 Heun grids.
    real_observe = diagnostic.observe_native_sample
    def quick(model, condition, seeds, *, steps):
        return real_observe(model, condition, seeds, steps=4)
    monkeypatch.setattr(diagnostic, "observe_native_sample", quick)
    return SimpleNamespace(config=cfg, corpus=corpus, entries=entries, output=tmp_path / "output", splits=splits)


def test_orchestration_binds_noise_geometry_sources_and_does_not_compare_generation_to_val_truth(native):
    frozen = {str(p): diagnostic._file_sha(p) for p in native.corpus.iterdir()}
    report = diagnostic.run_reverse_joint_diagnostic(native.entries, native.corpus, native.output,
        seed=10, device="cpu", expected_corpus_sha256="b"*64)
    assert report["execution_passed"], report.get("error")
    assert report["all_inputs_revalidated"]
    assert report["generated_answers"] == 48 and report["new_pde_solves"] == 0
    assert not report["blind_test"] and not report["new_physical_audit_performed"]
    assert report["validation_ids"] == list(range(200, 208))
    for model in report["models"].values():
        assert list(model["by_steps"]) == ["64", "128", "256"]
        for trace in model["by_steps"].values():
            assert trace["initial_noisy_states_sha256"] == report["initial_noisy_states_sha256"]
            assert "NOT_generated_answers" in trace["stages"][0]["role"]
            assert "rmse" not in json.dumps(trace["final"]["summary"]["amplitude_neighbor_proxy"])
            artifact = trace["final"]["raw_reduced_values"]
            assert diagnostic._file_sha(artifact["path"]) == artifact["sha256"]
        assert all(not pair["right_is_truth"] for pair in model["paired_numerical_differences"].values())
    diagnostic._assert_hashes(frozen)
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("failure", ("checkpoint_mutation", "validation_mutation", "geometry", "noise"))
def test_failure_preserves_first_model_and_exact_stage(native, monkeypatch, failure):
    original = diagnostic.load_vpfp_diffusion_sampler
    count = 0
    def load(*args, **kwargs):
        nonlocal count
        sampler = original(*args, **kwargs)
        count += 1
        if count == 2:
            if failure == "geometry":
                sampler.normalizer.covariance_strength = 1.
            elif failure == "noise":
                old_noise = sampler.model._noise
                sampler.model._noise = lambda *a, **k: old_noise(*a, **k) * 2
            else:
                path = Path(args[0]) if failure == "checkpoint_mutation" else native.corpus / "validation-conditions.npy"
                path.write_bytes(b"synthetic mutation")
        return sampler
    monkeypatch.setattr(diagnostic, "load_vpfp_diffusion_sampler", load)
    report = diagnostic.run_reverse_joint_diagnostic(native.entries, native.corpus, native.output, seed=10, device="cpu")
    assert not report["execution_passed"] and report["partial_artifacts_preserved"]
    assert len(report["models"]["seed-0"]["by_steps"]) == 3
    assert report["failed_stage"] and report["error"]
    assert (native.output / "result.partial.json").exists() and (native.output / "result.json").exists()


def test_wrong_corpus_and_unsafe_or_overwriting_output_fail_closed(native):
    report = diagnostic.run_reverse_joint_diagnostic(native.entries, native.corpus, native.output,
        seed=10, device="cpu", expected_corpus_sha256="c"*64)
    assert not report["execution_passed"] and not report["models"]
    with pytest.raises(ValueError, match="fresh"):
        diagnostic.run_reverse_joint_diagnostic(native.entries, native.corpus, native.output, seed=10, device="cpu")
    with pytest.raises(ValueError, match="64/128/256"):
        diagnostic.run_reverse_joint_diagnostic(native.entries, native.corpus, native.output.parent / "wrong",
            seed=10, device="cpu", steps=(64, 128))
    bad = {"../unsafe": next(iter(native.entries.values()))}
    with pytest.raises(ValueError, match="safe labels"):
        diagnostic.run_reverse_joint_diagnostic(bad, native.corpus, native.output.parent / "other", seed=10, device="cpu")
