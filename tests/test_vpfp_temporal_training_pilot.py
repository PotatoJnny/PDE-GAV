"""Tiny synthetic pipeline mechanics, not VPFP distribution qualification."""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pde_agent_verifier_solver.plugins.vpfp.structured_diffusion import VPFPStructuredEDMConfig, load_vpfp_diffusion_sampler
from pde_agent_verifier_solver.plugins.vpfp.value_control_diffusion import VPFPValueEDMConfig, lift_value_controls

ROOT = Path(__file__).resolve().parents[1]
loader = importlib.util.spec_from_file_location("temporal_pilot_test", ROOT / "scripts/run_vpfp_temporal_training_pilot.py")
cli = importlib.util.module_from_spec(loader)
loader.loader.exec_module(cli)


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def training(monkeypatch):
    cfg = VPFPStructuredEDMConfig(control_count=3, hermite_channels=2, rho_max_mode=1,
        kinetic_max_mode=1, inactive_dim=3, grid_size=3, width=8, depth=1)
    value_cfg = VPFPValueEDMConfig.from_full_config(cfg)
    random = torch.Generator().manual_seed(117)
    class Split:
        def __init__(self, seed):
            self.latents = lift_value_controls(torch.randn(8, value_cfg.latent_dimension, generator=random), value_cfg).numpy()
            self.conditions = np.tile([.05, 1.], (8, 1)).astype("float32")
            self.sample_ids = np.arange(seed, seed + 8)
            self.case_ids = list(map(str, self.sample_ids))
        def __len__(self): return len(self.sample_ids)
    splits = {"train": Split(1), "validation": Split(21)}
    manifest = {"latent_dimension": cfg.latent_dimension, "base_decoder_sha256": "a"*64, "manifest_sha256": "b"*64}
    monkeypatch.setattr(cli.shared, "read_manifest", lambda *a, **k: manifest)
    monkeypatch.setattr(cli.shared, "load_split", lambda path, role, **k: splits[role])
    monkeypatch.setattr(cli.previous, "load_split", lambda path, role, **k: splits[role])
    monkeypatch.setattr(cli.shared, "model_config", lambda *a: cfg)
    monkeypatch.setenv("SLURM_JOB_ID", "fixture_not_real_job")
    args = SimpleNamespace(seed=31, device="cpu", covariance_floor=.05, learning_rate=2.e-4,
        batch_size=2, steps=2, train_seconds=30, validate_every=1, source_revision="fixture")
    return args, SimpleNamespace(base_decoder_sha256="a"*64), cfg, splits


def test_four_arm_training_same_network_reloadable_full_abi(training, tmp_path):
    args, spec, cfg, splits = training
    results, samplers = {}, {}
    for arm in cli.ARMS:
        config, make, fit = cli.factories(args, spec, arm)
        directory = tmp_path / arm
        directory.mkdir()
        checkpoint, result = cli.shared.train(args, spec, tmp_path / "corpus", directory,
            model_config_factory=lambda *a, cfg=config: cfg, model_factory=make, normalizer_factory=fit)
        results[arm] = result
        sampler = load_vpfp_diffusion_sampler(checkpoint, result["checkpoint_sha256"], device="cpu", steps=2)
        samplers[arm] = sampler
        assert sampler.generate([100])["latents"].shape == (1, cfg.latent_dimension)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert payload["training"]["residual_metric"] is None
        assert payload["training"]["training_noise_distribution"] == result["training_noise_distribution"]
        if arm.endswith("_low"):
            assert result["training_noise_distribution"]["low_min"] == .01
            assert result["training_noise_distribution"]["low_max"] == .3
        assert payload["normalizer"]["fit_metadata"]["sample_ids"] == list(range(1, 9))
    assert len({r["initial_model_sha256"] for r in results.values()}) == 1
    for prefix in ("value", "cov"):
        assert results[prefix+"_base"]["normalizer_sha256"] == results[prefix+"_low"]["normalizer_sha256"]
    clean = torch.from_numpy(splits["validation"].latents)
    old, new = (samplers[arm].normalizer.encode(clean) for arm in ("value_base", "cov_base"))
    dim = samplers["value_base"].model.config.rho_dimension
    assert torch.equal(old[:, :dim], new[:, :dim])
    assert not torch.allclose(old[:, dim:], new[:, dim:])


def test_preflight_real_tiny_full_roundtrip_and_gradients(training, tmp_path):
    args, spec, cfg, _ = training
    report = cli.preflight(args, spec, tmp_path)
    assert set(report) == set(cli.ARMS)
    assert len({r["parameter_count"] for r in report.values()}) == 1
    assert all(r["execution_passed"] and not r["scientific_validation"] for r in report.values())
    assert all(r["full_dimension"] == cfg.latent_dimension for r in report.values())


def arguments(tmp_path):
    return ["--config", str(tmp_path/"missing.json"), "--cache", str(tmp_path/"cache"),
        "--diagnosis", str(tmp_path/"diagnosis"), "--output", str(tmp_path/"new"), "--regime", "weak", "--seed", "1"]


def test_requires_slurm_before_creating_output(monkeypatch, tmp_path):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(SystemExit, match="2"):
        cli.main(arguments(tmp_path))
    assert not (tmp_path/"new").exists()


@pytest.mark.parametrize("option,value", [("--replicas","1"),("--batch-size","9"),("--train-seconds","0"),
    ("--generated-answers","31"),("--generated-answers","33"),("--covariance-floor","nan"),
    ("--covariance-floor","0"),("--sample-steps","1")])
def test_bad_budget_rejected_before_source_reads(monkeypatch, tmp_path, option, value):
    monkeypatch.setenv("SLURM_JOB_ID", "fixture_not_real_job")
    with pytest.raises(SystemExit, match="2"):
        cli.main(arguments(tmp_path)+[option,value])
    assert not (tmp_path/"new").exists()


def test_existing_output_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("SLURM_JOB_ID", "fixture_not_real_job")
    (tmp_path/"new").mkdir()
    with pytest.raises(SystemExit, match="2"):
        cli.main(arguments(tmp_path))
