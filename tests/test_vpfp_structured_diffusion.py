"""Tiny synthetic mathematical/API checks; never VPFP scientific validation."""

from dataclasses import replace
from hashlib import sha256
import importlib.util
import io
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from pde_agent_verifier_solver.plugins.vpfp.structured_diffusion import (
    VPFPCoefficientNormalizer, VPFPDiffusionBatchSampler,
    VPFPFourierCoefficientTransform, VPFPStructuredEDM, VPFPStructuredEDMConfig,
    _SpatiotemporalBlock, load_vpfp_diffusion_sampler,
)


@pytest.fixture(scope="module", autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _config(**changes):
    return VPFPStructuredEDMConfig(control_count=3, hermite_channels=2,
        rho_max_mode=1, kinetic_max_mode=1, inactive_dim=3, grid_size=3,
        width=8, depth=1, **changes)


def _answers(config, count=4, dtype=torch.float32):
    values = torch.randn((count, config.latent_dimension), dtype=dtype,
                         generator=torch.Generator().manual_seed(100))
    values[:, config.rho_dimension:config.rho_dimension + config.inactive_dim] = 0.
    return values


def _conditions(config, count):
    return torch.tensor([[sum(config.collision_nu_range) / 2,
                          sum(config.temperature_range) / 2]] * count)


def test_production_config_matches_complete_current_coefficient_layout():
    config = VPFPStructuredEDMConfig()
    assert config.rho_dimension == 11088
    assert config.kinetic_dimension == 641520
    assert config.latent_dimension == 652672
    assert config.field_channels == 242
    assert config.control_count == 33
    assert config.slope_interval == 1 / 32
    assert config.temporal_dilation_cycle == (1, 2, 4, 9)
    assert config.convolutional_time_receptive_field == 33
    assert VPFPStructuredEDMConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError, match="rank"):
        replace(config, width=128)
    with pytest.raises(ValueError, match="aliases"):
        replace(config, grid_size=9)
    with pytest.raises(ValueError, match="interval|distribution"):
        replace(config, sigma_min=30.)


@pytest.mark.parametrize("dtype,tolerance", [(torch.float32, 2.e-6), (torch.float64, 2.e-14)])
def test_fourier_roundtrip_preserves_all_active_coefficients_and_gradients(dtype, tolerance):
    config = _config()
    transform = VPFPFourierCoefficientTransform(config)
    values = _answers(config, dtype=dtype).requires_grad_(True)
    field = transform.to_field(values)
    assert field.shape == (4, 6, 3, 3, 3)
    restored = transform.to_coefficients(field)
    torch.testing.assert_close(restored, values, rtol=tolerance, atol=tolerance)
    restored.square().sum().backward()
    expected = 2 * values.detach()
    torch.testing.assert_close(values.grad, expected, rtol=tolerance * 2, atol=tolerance * 2)


def test_fourier_conventions_match_native_rho_and_kinetic_order():
    config = _config()
    transform = VPFPFourierCoefficientTransform(config)
    values = torch.zeros((1, config.latent_dimension), dtype=torch.float64)
    values[0, 0] = 1.  # rho control0, cos(kx=0,ky=1)
    local_start = config.rho_dimension + config.inactive_dim
    values[0, local_start] = 2.  # Hermite0 control0 spatial DC
    values[0, local_start + 1] = 1.  # sqrt(2)*cos(ky=1)
    field = transform.to_field(values)
    cosine = torch.cos(torch.arange(3, dtype=torch.float64) * 2 * math.pi / 3)
    torch.testing.assert_close(field[0, 0, 0], cosine[None].expand(3, -1))
    torch.testing.assert_close(field[0, 2, 0], 2 + math.sqrt(2) * cosine[None].expand(3, -1))
    # A DC rho field is outside the declared coefficient image, not noise.
    outside = torch.zeros_like(field)
    outside[:, :2] = 1.
    assert transform.to_coefficients(outside).abs().max() < 1.e-14


def test_actual_spatial_modes_and_33_value_slope_controls_roundtrip_in_fp32():
    # Real 168/81 Fourier supports and 33 controls, but only one synthetic
    # Hermite channel: a bounded CPU transform check, not real model sampling.
    config = replace(VPFPStructuredEDMConfig(), hermite_channels=1, width=4)
    values = _answers(config, count=2)
    transform = VPFPFourierCoefficientTransform(config)
    normalizer = VPFPCoefficientNormalizer.fit(config, values, sample_ids=(41, 42))
    normalized = normalizer.encode(values)
    field = transform.to_field(normalized)
    restored = normalizer.decode(transform.to_coefficients(field))
    torch.testing.assert_close(restored, values, rtol=3.e-6, atol=3.e-6)
    assert field.dtype == restored.dtype == torch.float32
    assert field.shape == (2, 4, 33, 13, 13)


def test_normalizer_is_invertible_train_only_and_preserves_inactive_zero():
    config = _config()
    values = _answers(config, dtype=torch.float64)
    normalizer = VPFPCoefficientNormalizer.fit(config, values, sample_ids=(10, 11, 12, 13))
    torch.testing.assert_close(normalizer.decode(normalizer.encode(values)), values)
    reloaded = VPFPCoefficientNormalizer.from_payload(normalizer.to_payload())
    torch.testing.assert_close(reloaded.encode(values), normalizer.encode(values))
    assert reloaded.fit_metadata["centering"] == "none"
    assert reloaded.fit_metadata["slope_transform"] == "d_d_tau_times_delta_tau"
    assert reloaded.fit_metadata["physical_answer_fidelity_certified"] is False
    with pytest.raises(ValueError, match="train split"):
        VPFPCoefficientNormalizer.fit(config, values, sample_ids=(1, 2, 3, 4), split="validation")
    with pytest.raises(ValueError, match="distinct"):
        VPFPCoefficientNormalizer.fit(config, values, sample_ids=(1, 1, 2, 3))
    bad = values.clone()
    bad[0, config.rho_dimension] = 1.e-12
    with pytest.raises(ValueError, match="exactly zero"):
        normalizer.encode(bad)
    with pytest.raises(ValueError, match="exactly zero"):
        VPFPCoefficientNormalizer.fit(config, bad, sample_ids=(1, 2, 3, 4))


def test_normalizer_slopes_are_interval_scaled_and_rare_groups_have_floor():
    config = _config()
    values = torch.ones((3, config.latent_dimension), dtype=torch.float64)
    values[:, config.rho_dimension:config.rho_dimension + config.inactive_dim] = 0.
    local_start = config.rho_dimension + config.inactive_dim
    values[:, local_start + 2 * config.control_count * config.kinetic_modes:] *= 1.e-12
    normalizer = VPFPCoefficientNormalizer.fit(config, values, sample_ids=(1, 2, 3))
    torch.testing.assert_close(normalizer.group_scales[0], torch.tensor([1., .5], dtype=torch.float64))
    assert torch.all(normalizer.group_scales[2] >= normalizer.fit_metadata["kinetic_floor"])
    torch.testing.assert_close(normalizer.decode(normalizer.encode(values)), values)


def test_edm_preconditioning_loss_is_coefficient_space_and_reports_sigma_bins():
    config = _config()
    model = VPFPStructuredEDM(config)
    clean = _answers(config)
    conditions = _conditions(config, 4)
    noise = _answers(config).roll(1, dims=0)
    sigma = torch.tensor([.003, .03, .3, 3.])
    noisy = clean + sigma[:, None] * noise
    expected = noisy / (1 + sigma[:, None].square())
    torch.testing.assert_close(model.denoise(noisy, sigma, conditions), expected)
    loss, metrics = model.loss(clean, conditions, sigma=sigma, noise=noise)
    error = (expected - clean).square()
    weight = (1 + sigma.square()) / sigma.square()
    expected_loss = .5 * (error[:, :config.rho_dimension].mean(1)
        + error[:, config.rho_dimension + config.inactive_dim:].mean(1)) * weight
    torch.testing.assert_close(loss, expected_loss.mean())
    torch.testing.assert_close(metrics["loss_per_case"], expected_loss)
    torch.testing.assert_close(metrics["zero_mean_isotropic_baseline_per_case"], expected_loss)
    assert metrics["sigma_bin_counts"].tolist() == [1, 1, 1, 1]
    torch.testing.assert_close(metrics["sigma_bin_loss_sums"], expected_loss)
    loss.backward()
    assert model.output.weight.grad is not None
    assert torch.isfinite(model.output.weight.grad).all()
    assert model.output.weight.grad.abs().sum() > 0
    bad_noise = noise.clone()
    bad_noise[0, config.rho_dimension] = 1.
    with pytest.raises(ValueError, match="inactive"):
        model.loss(clean, conditions, sigma=sigma, noise=bad_noise)


def test_joint_network_has_physical_time_coordinates_and_condition_dependence():
    config = _config()
    torch.manual_seed(91)
    model = VPFPStructuredEDM(config)
    with torch.no_grad():
        model.output.weight.normal_(std=.03)
    noisy = _answers(config, count=1)
    condition = _conditions(config, 1)
    first = model.denoise(noisy, torch.ones(1), condition)
    changed = model.denoise(noisy, torch.ones(1), condition + torch.tensor([[.01, .05]]))
    assert not torch.allclose(first, changed)
    tau = torch.tensor([0., .5, 1.])
    position = model.time_position(torch.stack((tau, (math.pi * tau).sin(), (math.pi * tau).cos()), dim=1))
    assert not torch.allclose(position[0], position[-1])
    assert model.input.in_channels == model.output.out_channels == config.field_channels
    assert model.output.in_channels >= config.field_channels


@pytest.mark.parametrize("dilation", [1, 2, 9])
def test_block_padding_wraps_only_space_not_time(dilation):
    block = _SpatiotemporalBlock(8, temporal_dilation=dilation)
    captured = []
    hook = block.depthwise.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
    result = block(torch.randn(1, 8, 3, 3, 3), torch.randn(1, 8))
    hook.remove()
    padded = captured[0]
    assert result.shape == (1, 8, 3, 3, 3)
    assert padded.shape == (1, 8, 3 + 2 * dilation, 5, 5)
    assert torch.count_nonzero(padded[:, :, :dilation]) == torch.count_nonzero(padded[:, :, -dilation:]) == 0
    torch.testing.assert_close(padded[:, :, dilation:-dilation, 0, 1:-1], padded[:, :, dilation:-dilation, -2, 1:-1])
    torch.testing.assert_close(padded[:, :, dilation:-dilation, 1:-1, 0], padded[:, :, dilation:-dilation, 1:-1, -2])


@pytest.mark.parametrize("churn", [0., 2.])
def test_sampling_replays_per_seed_independently_of_batch_and_order(churn):
    config = _config()
    model = VPFPStructuredEDM(config)
    conditions = _conditions(config, 2)
    together = model.sample(conditions, (17, 18), steps=3, churn=churn)
    separately = torch.cat([model.sample(conditions[i:i + 1], (seed,), steps=3, churn=churn)
                            for i, seed in enumerate((17, 18))])
    reverse = model.sample(conditions.flip(0), (18, 17), steps=3, churn=churn)
    torch.testing.assert_close(together, separately, rtol=0, atol=0)
    torch.testing.assert_close(together, reverse.flip(0), rtol=0, atol=0)
    assert torch.isfinite(together).all()
    assert not torch.allclose(together[0], together[1])
    assert torch.count_nonzero(together[:, config.rho_dimension:config.rho_dimension + config.inactive_dim]) == 0
    with pytest.raises(ValueError, match="regime"):
        model.sample(torch.tensor([[1., 1.]]), (1,), steps=3)
    with pytest.raises(ValueError, match="seeds"):
        model.sample(conditions[:1], (True,), steps=3)


def test_checkpoint_state_roundtrip_and_batch_sampler_abi_without_legacy_imports():
    config = _config()
    model = VPFPStructuredEDM(config)
    normalizer = VPFPCoefficientNormalizer.fit(config, _answers(config), sample_ids=(1, 2, 3, 4))
    buffer = io.BytesIO()
    torch.save({"config": config.to_dict(), "model": model.state_dict(), "normalizer": normalizer.to_payload()}, buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, weights_only=True)
    restored = VPFPStructuredEDM(VPFPStructuredEDMConfig.from_dict(checkpoint["config"]))
    restored.load_state_dict(checkpoint["model"], strict=True)
    restored_normalizer = VPFPCoefficientNormalizer.from_payload(checkpoint["normalizer"])
    sampler = VPFPDiffusionBatchSampler(restored, restored_normalizer,
        base_decoder_sha256="a" * 64, checkpoint_sha256="b" * 64, steps=3)
    batch = sampler.generate((101, 102))
    replay = sampler.generate((102, 101))
    np.testing.assert_array_equal(batch["latents"], replay["latents"][::-1])
    np.testing.assert_array_equal(batch["equation_conditions"], replay["equation_conditions"][::-1])
    assert batch["latents"].shape == (2, config.latent_dimension)
    assert batch["equation_seeds"].tolist() == [101, 102]
    assert batch["diagnostics"]["pde_solves"] == 0
    assert sampler.identity["qualification"] == "unqualified_research_checkpoint"
    assert sampler.identity["latent_dimension"] == config.latent_dimension
    assert sampler.identity["base_decoder_sha256"] == "a" * 64
    assert sampler.identity["coefficient_fidelity_is_not_physical_fidelity"] is True
    with pytest.raises(ValueError, match="SHA256"):
        VPFPDiffusionBatchSampler(restored, restored_normalizer,
            base_decoder_sha256="invalid", checkpoint_sha256="b" * 64)


@pytest.mark.parametrize("bad_sigma", [torch.tensor([0.]), torch.tensor([float("nan")]), torch.tensor([21.])])
def test_loss_rejects_out_of_law_or_nonfinite_noise(bad_sigma):
    config = _config()
    model = VPFPStructuredEDM(config)
    with pytest.raises(ValueError, match="sigma"):
        model.loss(_answers(config, count=1), _conditions(config, 1), sigma=bad_sigma)


def _checkpoint_payload():
    config = _config()
    model = VPFPStructuredEDM(config)
    normalizer = VPFPCoefficientNormalizer.fit(config, _answers(config), sample_ids=(1, 2, 3, 4))
    return {"schema": "vpfp_structured_diffusion_checkpoint_v1", "config": config.to_dict(),
            "model": model.state_dict(), "normalizer": normalizer.to_payload(),
            "base_decoder_sha256": "a" * 64, "corpus_manifest_sha256": "c" * 64,
            "training": {"data_origin": "synthetic_test", "step": 0}}


def _save_checkpoint(tmp_path, payload):
    checkpoint = tmp_path / "fixture.pt"
    torch.save(payload, checkpoint)
    return checkpoint, sha256(checkpoint.read_bytes()).hexdigest()


def test_strict_checkpoint_loader_preserves_provenance_and_freezes_weights(tmp_path):
    checkpoint, digest = _save_checkpoint(tmp_path, _checkpoint_payload())
    sampler = load_vpfp_diffusion_sampler(checkpoint, digest, "cpu", steps=3)
    assert sampler.identity["corpus_manifest_sha256"] == "c" * 64
    assert sampler.identity["training"]["data_origin"] == "synthetic_test"
    assert sampler.identity["checkpoint_sha256"] == digest
    assert not sampler.model.training
    assert all(not parameter.requires_grad for parameter in sampler.model.parameters())
    assert sampler.generate((91,))["latents"].shape == (1, sampler.model.config.latent_dimension)
    loaded = load_vpfp_diffusion_sampler(checkpoint, "0" * 64, steps=3)
    np.testing.assert_array_equal(sampler.generate((91,))["latents"], loaded.generate((91,))["latents"])
    link = tmp_path / "alias.pt"
    link.symlink_to(checkpoint)
    with pytest.raises(ValueError, match="non-symlink"):
        load_vpfp_diffusion_sampler(link, digest)


@pytest.mark.parametrize("mutation,error", [
    ("schema", "schema/fields"), ("missing_tensor", "keys"),
    ("nonfinite_tensor", "tensor"), ("wrong_dtype", "tensor"),
    ("fixed_basis", "fixed Fourier"), ("normalizer_config", "configurations"),
    ("corpus_hash", "corpus manifest"), ("base_hash", "base decoder"),
    ("training", "training provenance"), ("nonfinite_provenance", "finite JSON"),
    ("normalizer_ids", "provenance/layout"),
])
def test_strict_checkpoint_loader_rejects_malformed_payloads(tmp_path, mutation, error):
    payload = _checkpoint_payload()
    if mutation == "schema":
        payload["schema"] = "legacy_checkpoint"
    elif mutation == "missing_tensor":
        payload["model"].pop("input.weight")
    elif mutation == "nonfinite_tensor":
        payload["model"]["input.weight"].flatten()[0] = float("nan")
    elif mutation == "wrong_dtype":
        payload["model"]["input.weight"] = payload["model"]["input.weight"].double()
    elif mutation == "fixed_basis":
        payload["model"]["transform.rho_basis"].flatten()[0] += .001
    elif mutation == "normalizer_config":
        payload["normalizer"]["config"]["depth"] = 2
    elif mutation == "corpus_hash":
        payload["corpus_manifest_sha256"] = "invalid"
    elif mutation == "base_hash":
        payload["base_decoder_sha256"] = "invalid"
    elif mutation == "training":
        payload["training"] = {}
    elif mutation == "nonfinite_provenance":
        payload["training"]["loss"] = float("inf")
    elif mutation == "normalizer_ids":
        payload["normalizer"]["fit_metadata"]["sample_ids"] = [1, 1]
    checkpoint, digest = _save_checkpoint(tmp_path, payload)
    with pytest.raises(ValueError, match=error):
        load_vpfp_diffusion_sampler(checkpoint, digest)


def _vpfp_smoke_binding():
    """Tiny VPFP-only fixture; no dependency on other PDE smoke tests."""
    from pde_agent_verifier_solver.app.backend_run import BackendRunBinding
    from pde_agent_verifier_solver.plugins.vpfp.canonical_binding import _BoundVPFPContinuousGenerator
    from pde_agent_verifier_solver.plugins.vpfp.macro_programs import VPFPHoldInitialMacroProgram
    from pde_agent_verifier_solver.plugins.vpfp.product import build_vpfp_plugin
    from pde_agent_verifier_solver.sdk import (
        Case, CaseRecord, CaseSource, CaseVisibility, DatasetInput, DatasetKind,
        GeneratorCheckpoint, PreparedDataset,
    )

    class Truth:
        def __init__(self, case):
            self.case = case

        def answer(self, query):
            return VPFPHoldInitialMacroProgram().predict_observables(self.case, query)

    class Backend:
        supports_perturbations = True
        supports_guidance = False

        def __init__(self):
            self.active = set()

        def generate(self, base, guidance, request, progress=None):
            records = []
            for index in range(request.count):
                parent = request.anchors[index] if request.anchors else None
                template = Case("private-id-is-not-a-feature", CaseSource.GENERATED,
                    CaseVisibility.DEVELOPMENT, public_data={
                        "question": {"collision_nu": 0.03, "temperature": 1.0},
                        "arrays": {"t": np.array([0., 1.]), "x": np.array([0., 0.5]),
                                   "y": np.array([0., 0.5]), "rho0": np.ones((2, 2)),
                                   "momentum0": np.zeros((2, 2, 2)),
                                   "kinetic_energy0": np.ones((2, 2))},
                    })
                case = replace(template,
                    case_id=f"{request.purpose.value}-{request.kind.value}-{request.seed}-{index}",
                    visibility=request.visibility, round_index=request.round_index,
                    source=CaseSource.PERTURBED if parent else CaseSource.GENERATED,
                    parent_case_id=parent.case_id if parent else None)
                records.append(CaseRecord(case, Truth(case)))
                self.active.add(case.case_id)
            return tuple(records)

        def release_cases(self, case_ids):
            self.active.difference_update(case_ids)

    backend = Backend()
    identity = {"binding_sha256": "a" * 64}
    generator = _BoundVPFPContinuousGenerator(backend, identity)
    return BackendRunBinding(
        replace(build_vpfp_plugin(), generator=generator),
        GeneratorCheckpoint(generator.info.program_id, identity),
        PreparedDataset(DatasetInput("smoke-fixture", DatasetKind.NUMERICAL), (), ()),
        report={"data_origin": "synthetic_interface_fixture"},
    ), lambda: not backend.active


def _online_generation_fixture(monkeypatch):
    import sys
    from types import SimpleNamespace

    actual_calls = []

    class OnlineGenerator:
        def generate(self, *args, **kwargs):
            actual_calls.append(True)

    module = SimpleNamespace(VPFPOnlineBatchGenerator=OnlineGenerator)
    monkeypatch.setitem(sys.modules, "agent_verifier.vpfp.online_training", module)
    return module, OnlineGenerator.generate, actual_calls


@pytest.fixture
def diffusion_smoke_module():
    path = Path(__file__).resolve().parents[1] / "scripts/run_plasma_canonical_smoke.py"
    spec = importlib.util.spec_from_file_location("vpfp_diffusion_smoke_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("physics,options,error", [
    ("vpfp", ["--diffusion-checkpoint", "test.pt"], "supplied together"),
    ("vpfp", ["--diffusion-checkpoint-sha256", "a" * 64], "supplied together"),
    ("gkw", ["--diffusion-checkpoint", "test.pt", "--diffusion-checkpoint-sha256", "a" * 64], "only for VPFP"),
    ("vpfp", ["--diffusion-checkpoint", "test.pt", "--diffusion-checkpoint-sha256", "short"], "SHA256"),
    ("vpfp", ["--learned-checkpoint", "test.npz", "--learned-checkpoint-sha256", "a" * 64,
              "--diffusion-checkpoint", "test.pt", "--diffusion-checkpoint-sha256", "a" * 64], "mutually exclusive"),
])
def test_diffusion_smoke_options_reject_invalid_combinations(tmp_path, capsys, diffusion_smoke_module, physics, options, error):
    with pytest.raises(SystemExit) as failure:
        diffusion_smoke_module.main(["--physics", physics, "--config", str(tmp_path / "unused.json"),
                                     "--output", str(tmp_path / "output"), *options])
    assert failure.value.code == 2
    assert error in capsys.readouterr().err
    assert not (tmp_path / "output").exists()


def test_diffusion_smoke_loader_dispatch_and_entire_campaign_guard(tmp_path, monkeypatch, diffusion_smoke_module):
    from pde_agent_verifier_solver.plugins.vpfp import canonical_binding, structured_diffusion
    binding, released = _vpfp_smoke_binding()
    sampler = object()
    calls = []

    def load(path, expected_sha256, device):
        calls.append((path, expected_sha256, device))
        return sampler

    def bind(spec, **settings):
        assert settings["latent_sampler"] is sampler
        return binding

    monkeypatch.setattr(structured_diffusion, "load_vpfp_diffusion_sampler", load)
    monkeypatch.setattr(canonical_binding, "build_vpfp_online_binding", bind)
    online, original, actual_calls = _online_generation_fixture(monkeypatch)
    backend = binding.plugin.generator.backend
    original_generate = backend.generate
    guarded = []

    def generate(*args, **kwargs):
        assert online.VPFPOnlineBatchGenerator.generate is not original
        guarded.append(True)
        return original_generate(*args, **kwargs)

    monkeypatch.setattr(backend, "generate", generate)
    monkeypatch.setenv("SLURM_JOB_ID", "unit-test-no-real-job")
    config = tmp_path / "config.json"
    config.write_text('{"physics":"vpfp","vpfp":{}}')
    checkpoint = tmp_path / "model.pt"
    output = tmp_path / "output"
    result = diffusion_smoke_module.main(["--physics", "vpfp", "--config", str(config), "--output", str(output),
        "--device", "cpu", "--diffusion-checkpoint", str(checkpoint), "--diffusion-checkpoint-sha256", "b" * 64])
    report = json.loads((output / "report.json").read_text())
    assert result == 0, report.get("traceback")
    assert calls == [(checkpoint, "b" * 64, "cpu")]
    assert report["generation_mode"] == "diffusion_vpfp_checkpoint"
    assert report["diffusion_checkpoint_sha256"] == "b" * 64
    assert report["learned_checkpoint"] is None
    assert report["online_pde_calls_during_learned_sampling"] == 0
    assert report["online_pde_guard_restored"] is True
    assert report["model_training_performed"] is False
    assert report["scientific_qualification"] == "not_established_by_interface_smoke"
    assert len(guarded) >= 5 and not actual_calls
    assert online.VPFPOnlineBatchGenerator.generate is original
    assert released()
