"""Tiny synthetic pipeline mechanics, not a distribution or plasma validation."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pde_agent_verifier_solver.plugins.vpfp.structured_diffusion import VPFPStructuredEDMConfig

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/train_vpfp_structured_diffusion.py"
specification = importlib.util.spec_from_file_location("vpfp_training_cli_under_test", SCRIPT)
cli = importlib.util.module_from_spec(specification)
specification.loader.exec_module(cli)


@pytest.fixture(autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_audit_conditions_independent_balanced_and_deterministic():
    config = VPFPStructuredEDMConfig()
    first = cli.balanced_conditions(config, 128, 31)
    assert np.array_equal(first, cli.balanced_conditions(config, 128, 31))
    assert not np.array_equal(first, cli.balanced_conditions(config, 128, 32))
    nu = np.searchsorted(np.linspace(*config.collision_nu_range, 5)[1:-1], first[:, 0])
    temperature = np.searchsorted(np.linspace(*config.temperature_range, 3)[1:-1], first[:, 1])
    np.testing.assert_array_equal(np.bincount(2 * nu + temperature), [16] * 8)
    assert len(np.unique(first, axis=0)) == 128
    with pytest.raises(ValueError):
        cli.balanced_conditions(config, 15, 31)


def test_real_pipeline_refuses_login_node(monkeypatch, tmp_path):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(SystemExit, match="2"):
        cli.main(["--config", str(tmp_path / "missing.json"), "--output", str(tmp_path / "out"), "--seed", "1"])
    assert not (tmp_path / "out").exists()


def test_checkpoint_corpus_binding_rejects_other_manifest_and_training_split():
    sampler = SimpleNamespace(identity={"corpus_manifest_sha256": "a" * 64, "base_decoder_sha256": "b" * 64},
                              normalizer=SimpleNamespace(fit_metadata={"sample_ids": [1, 2]}))
    manifest = {"manifest_sha256": "a" * 64, "base_decoder_sha256": "b" * 64}
    cli.assert_checkpoint_corpus(sampler, manifest, [1, 2])
    with pytest.raises(ValueError, match="manifest"):
        cli.assert_checkpoint_corpus(sampler, {**manifest, "manifest_sha256": "c" * 64}, [1, 2])
    with pytest.raises(ValueError, match="training split"):
        cli.assert_checkpoint_corpus(sampler, manifest, [2, 1])
    with pytest.raises(ValueError, match="decoder"):
        cli.assert_checkpoint_corpus(sampler, {**manifest, "base_decoder_sha256": "c" * 64}, [1, 2])


def test_allocation_preflight_is_explicitly_synthetic():
    config = VPFPStructuredEDMConfig(control_count=3, hermite_channels=2, rho_max_mode=1,
        kinetic_max_mode=1, inactive_dim=3, grid_size=3, width=8, depth=1)
    report = cli.allocation_preflight(config, SimpleNamespace(device="cpu", seed=14, batch_size=2))
    assert report["execution_passed"] is True
    assert report["scientific_validation"] is False
    assert report["data_origin"] == "synthetic_mechanics_fixture"


def test_tiny_training_selects_only_validation_and_saves_reloadable_checkpoint(monkeypatch, tmp_path):
    config = VPFPStructuredEDMConfig(control_count=3, hermite_channels=2, rho_max_mode=1,
        kinetic_max_mode=1, inactive_dim=3, grid_size=3, width=8, depth=1)
    random = np.random.default_rng(17)
    def make_split(seed):
        values = random.normal(size=(8, config.latent_dimension)).astype("float32")
        values[:, config.rho_dimension:config.rho_dimension + config.inactive_dim] = 0
        class Split:
            latents = values
            conditions = np.tile([.05, 1.], (8, 1)).astype("float32")
            sample_ids = np.arange(seed, seed + 8)
            case_ids = [str(value) for value in sample_ids]
            def __len__(self):
                return 8
        return Split()
    splits = {"train": make_split(10), "validation": make_split(30)}
    opened = []
    def load(path, role, verify_hashes):
        opened.append(role)
        return splits[role]
    monkeypatch.setattr(cli, "load_split", load)
    monkeypatch.setattr(cli, "read_manifest", lambda *a, **k: {
        "latent_dimension": config.latent_dimension, "base_decoder_sha256": "a" * 64,
        "manifest_sha256": "b" * 64})
    monkeypatch.setattr(cli, "VPFPStructuredEDMConfig", lambda **kwargs: config)
    monkeypatch.setenv("SLURM_JOB_ID", "fixture-not-real-job")
    arguments = SimpleNamespace(width=8, depth=1, seed=14, device="cpu", learning_rate=2e-4,
        batch_size=2, steps=2, train_seconds=30, validate_every=1, source_revision="fixture")
    model_spec = SimpleNamespace(collision_nu_range=config.collision_nu_range,
        temperature_range=config.temperature_range, time_domain=config.time_domain, base_decoder_sha256="a" * 64)
    checkpoint, result = cli.train(arguments, model_spec, tmp_path / "corpus", tmp_path)
    assert opened == ["train", "validation"]
    assert result["training_presentations"] == 4
    assert result["unique_training_answers"] == 8
    assert result["audit_used_for_selection"] is False
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["normalizer"]["fit_metadata"]["sample_ids"] == list(range(10, 18))
    assert payload["training"]["step"] in (1, 2)
    assert payload["training"]["validation"]["counts_by_sigma"] == [8] * 4
    from pde_agent_verifier_solver.plugins.vpfp.structured_diffusion import load_vpfp_diffusion_sampler
    sampler = load_vpfp_diffusion_sampler(checkpoint, expected_sha256=result["checkpoint_sha256"], device="cpu", steps=2)
    assert sampler.generate([500])["latents"].shape == (1, config.latent_dimension)
    assert json.loads((tmp_path / "training-summary.json").read_text())["final_step"] == 2
