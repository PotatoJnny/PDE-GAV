"""Tiny conditional mechanics; not real-data training or qualification."""
from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pde_agent_verifier_solver.plugins.vpfp import initial_conditioned_diffusion as joint
from pde_agent_verifier_solver.plugins.vpfp import known_initial_prior as prior
from pde_agent_verifier_solver.plugins.vpfp.canonical_binding import VPFPOnlineModelSpec
from pde_agent_verifier_solver.plugins.vpfp.structured_diffusion import VPFPStructuredEDMConfig
from pde_agent_verifier_solver.plugins.vpfp.value_control_diffusion import VPFPValueControlEDM, lift_value_controls, extract_value_controls, _value_split


@pytest.fixture(autouse=True)
def bounded_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def fixture_config(count=5):
    return joint.VPFPInitialConditionedEDMConfig(VPFPStructuredEDMConfig(control_count=count,
        hermite_channels=3, rho_max_mode=1, kinetic_max_mode=1, inactive_dim=2,
        grid_size=3, width=8, depth=1, collision_nu_range=(10., 30.), sigma_max=2.))


def prior_spec(config, *, model_spec=None, decoder=None, source_hashes=None):
    full = config.full_config
    decoder = decoder or {"structured_time_basis": "piecewise_cubic_hermite_v1", "structured_direct_coefficients": True,
        "structured_spectral_latent": True, "structured_time_control_count": full.control_count,
        "rho_max_mode_x": full.rho_max_mode, "rho_max_mode_y": full.rho_max_mode,
        "structured_local_spatial_mode": full.kinetic_max_mode, "hermite_total_degree": 1,
        "structured_local_channels": full.hermite_channels, "structured_kinetic_global_dim": full.inactive_dim,
        "latent_dim": full.latent_dimension, "time_domain": list(full.time_domain), "dtype": "float32",
        "structured_residual_scale": 0., "collision_nu_range": list(full.collision_nu_range),
        "temperature_range": list(full.temperature_range)}
    k, r, m, h = full.control_count, full.rho_modes, full.kinetic_modes, full.hermite_channels
    layout = {"rho_mode_count": r//2, "rho_time_feature_count": 2*k, "rho_latent_dim": 2*k*r,
        "kinetic_global_latent_dim": full.inactive_dim, "local_spatial_feature_count": m,
        "local_time_feature_count": 2*k, "local_latent_dim": h*2*k*m,
        "hermite_mode_count": h, "total_latent_dim": full.latent_dimension}
    if model_spec is None:
        model_spec = VPFPOnlineModelSpec("strong_fixture", "/never-read/base.pt", "a"*64,
            "/never-read/expanded.pt", "b"*64, full.collision_nu_range, full.temperature_range, full.time_domain)
    manifest = {"model_spec": asdict(model_spec), "manifest_sha256": "c"*64, "provenance": {
        "decoder_config": decoder, "generation_config": {"batch_size": 1, "nx": 4, "ny": 4,
            "velocity_grid_size": 21, "nt": k, "t1": full.time_domain[1], "dtype": "float32"},
        "equation_config": {"equation_family": prior.FAMILY, "collision_nu_range": list(full.collision_nu_range),
                            "temperature_range": list(full.temperature_range)},
        "structured_latent_layout": layout, "physical_module_sha256": source_hashes or {name: "d"*64 for name in prior.MODULES}}}
    return prior.KnownInitialPriorSpec.from_manifest(manifest)


def data(dtype=torch.float64, count=5):
    config = fixture_config(count)
    values = torch.randn(8, config.value_config.latent_dimension, dtype=dtype, generator=torch.Generator().manual_seed(821))
    _value_split(values, config.value_config)[1][..., 0] += torch.linspace(3., .5, count)
    raw = lift_value_controls(values, config.value_config)
    normalizer = joint.VPFPInitialConditionedNormalizer.fit(config, raw,
        sample_ids=list(range(8)), initial_prior=prior_spec(config))
    return config, raw, normalizer


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_schur_roundtrip_full_abi_and_clean_context(dtype):
    config, raw, normalizer = data(dtype)
    context, future = normalizer.encode(raw)
    assert context.shape == (8, config.context_dimension)
    assert future.shape == (8, config.latent_dimension)
    restored = normalizer.decode((context, future))
    torch.testing.assert_close(restored, raw, rtol=3.e-6 if dtype == torch.float32 else 1.e-12,
                               atol=3.e-6 if dtype == torch.float32 else 1.e-12)
    changed = normalizer.decode((context, future + 1.))
    changed_context, _ = normalizer.encode(changed)
    torch.testing.assert_close(changed_context, context, rtol=2.e-6 if dtype == torch.float32 else 1.e-12,
                               atol=2.e-6 if dtype == torch.float32 else 1.e-12)
    identity = normalizer.inverse_square_root @ normalizer.schur @ normalizer.inverse_square_root.transpose(-1, -2)
    torch.testing.assert_close(identity, torch.eye(config.control_count, dtype=torch.float64).expand_as(identity), rtol=1.e-12, atol=1.e-12)
    assert normalizer.fit_metadata["future_times"] == pytest.approx([.25, .5, .75, 1.])


def test_context_uses_only_unwhitened_t0_values_and_encoder_preserves_all_modes():
    config, raw, normalizer = data()
    context, future = normalizer.encode(raw)
    values = joint._value_split(normalizer.temporal.parent.encode(raw), config.value_config)
    expected = torch.cat((values[0][:, 0], values[1][:, :, 0].flatten(1)), dim=1)
    torch.testing.assert_close(context, expected, rtol=0, atol=0)
    full_values = extract_value_controls(raw, config.value_config)
    r, k = _value_split(full_values, config.value_config)
    raw_initial = torch.cat((r[:, 0], k[:, :, 0].flatten(1)), dim=1)
    torch.testing.assert_close(normalizer.encode_initial(raw_initial), context, rtol=1.e-15, atol=1.e-15)
    model = joint.VPFPInitialConditionedEDM(config).double()
    changed = context.clone()
    changed[:, 1] += 1.
    delta = model.initial_field(changed) - model.initial_field(context)
    assert delta[:, 0].std() > .1
    assert torch.count_nonzero(delta[:, 1:]) == 0


def test_exact_original_trunk_initialization_and_parameter_budget():
    config = joint.VPFPInitialConditionedEDMConfig(VPFPStructuredEDMConfig(collision_nu_range=(10., 30.)))
    torch.manual_seed(25)
    old = VPFPValueControlEDM(config.value_config)
    torch.manual_seed(25)
    new = joint.VPFPInitialConditionedEDM(config)
    for name, parameter in old.named_parameters():
        torch.testing.assert_close(dict(new.named_parameters())[name], parameter, rtol=0, atol=0)
    added = sum(p.numel() for p in new.parameters()) - sum(p.numel() for p in old.parameters())
    assert added == (config.field_channels+1)*config.width == 31232
    assert config.latent_dimension == 316416 and config.context_dimension == 9888 and config.full_dimension == 652672


def test_conditional_loss_original_objective_future_only_and_explicit_rng(monkeypatch):
    config, raw, normalizer = data(torch.float32)
    model = joint.VPFPInitialConditionedEDM(config)
    context, future = normalizer.encode(raw[:2])
    conditions = torch.tensor([[15., 1.], [25., 1.1]])
    sigma, noise = torch.tensor([.1, .3]), torch.ones_like(future)
    seen = []
    original = model.denoise
    def capture(noisy, sigma, theta, *, initial_context):
        seen.append(initial_context.clone())
        return original(noisy, sigma, theta, initial_context=initial_context)
    monkeypatch.setattr(model, "denoise", capture)
    generator = torch.Generator().manual_seed(19)
    state = generator.get_state().clone()
    loss, metrics = model.loss((context, future), conditions, sigma=sigma, noise=noise, generator=generator)
    torch.testing.assert_close(generator.get_state(), state, rtol=0, atol=0)
    torch.testing.assert_close(seen[0], context, rtol=0, atol=0)
    assert metrics["sigma_bin_counts"].sum() == 2
    assert metrics["loss_per_case"].shape == (2,)
    # Zero-initialized residual has exactly the original Gaussian skip loss;
    # context does not add any target term to this objective.
    torch.testing.assert_close(metrics["loss_per_case"], metrics["zero_mean_isotropic_baseline_per_case"], rtol=0, atol=0)
    loss.backward()
    assert model.output.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="requires"):
        model.loss(future, conditions)
    with pytest.raises(ValueError, match="not conditional"):
        model.loss((context, future), conditions, residual_metric=object())


def test_clean_initial_encoder_receives_gradient_after_output_starts():
    config, raw, normalizer = data(torch.float32)
    model = joint.VPFPInitialConditionedEDM(config)
    with torch.no_grad():
        model.output.weight.normal_(std=.03)
    loss, _ = model.loss(normalizer.encode(raw[:2]), [[15., 1.], [25., 1.]], sigma=torch.tensor([.1, .3]))
    loss.backward()
    assert torch.isfinite(model.initial_input.weight.grad).all() and model.initial_input.weight.grad.abs().sum() > 0


def test_future_time_positions_are_not_renormalized_to_zero_one():
    config, raw, normalizer = data(torch.float32)
    model = joint.VPFPInitialConditionedEDM(config)
    seen = []
    handle = model.time_position.register_forward_pre_hook(lambda module, args: seen.append(args[0].clone()))
    model.loss(normalizer.encode(raw[:1]), [[20., 1.]], sigma=torch.tensor([.1]))
    handle.remove()
    torch.testing.assert_close(seen[0][:, 0], torch.tensor([.25, .5, .75, 1.]))


def test_heun_only_future_states_fixed_context_and_grouping(monkeypatch):
    config, raw, normalizer = data(torch.float32)
    model = joint.VPFPInitialConditionedEDM(config)
    context, _ = normalizer.encode(raw[:2])
    seen = []
    original = model.denoise
    def capture(noisy, sigma, theta, *, initial_context):
        seen.append((noisy.shape, initial_context.clone()))
        return original(noisy, sigma, theta, initial_context=initial_context)
    monkeypatch.setattr(model, "denoise", capture)
    conditions = [[15., 1.], [25., 1.]]
    paired = model.sample(context, conditions, [21, 22], steps=3)
    separate = torch.cat([model.sample(context[i:i+1], conditions[i:i+1], [21+i], steps=3) for i in range(2)])
    torch.testing.assert_close(paired, separate, rtol=0, atol=0)
    assert all(shape == (1, config.latent_dimension) for shape, _ in seen)
    for _, value in seen[:5]:
        torch.testing.assert_close(value, context[:1], rtol=0, atol=0)


def checkpoint_payload():
    config, raw, normalizer = data(torch.float32)
    model = joint.VPFPInitialConditionedEDM(config)
    payload = {"schema": joint.CHECKPOINT_SCHEMA, "config": config.to_dict(), "model": model.state_dict(),
        "normalizer": normalizer.to_payload(), "base_decoder_sha256": "a"*64, "corpus_manifest_sha256": "c"*64,
        "training": {"seed": 0, "normalizer_sha256": joint._json_sha(normalizer.to_payload()),
            "train_case_ids": [f"strong_fixture:{i}" for i in range(8)]}}
    return config, raw, normalizer, model, payload


def test_native_safe_payload_reload_and_sampler_full_dimension(monkeypatch):
    config, raw, normalizer, model, payload = checkpoint_payload()
    sampler = joint._sampler_from_initial_conditioned_payload(payload, "e"*64, "cpu", steps=3, churn=0.)
    context, future = normalizer.encode(raw[:2])
    theta = [[15., 1.], [25., 1.]]
    torch.testing.assert_close(sampler.model.denoise(future, [.1, .3], theta, initial_context=context),
        model.denoise(future, [.1, .3], theta, initial_context=context), rtol=0, atol=0)
    assert sampler.identity["latent_dimension"] == config.full_dimension
    assert sampler.identity["random_dimension"] == config.latent_dimension
    assert sampler.identity["sampling"]["seed_is_network_condition"] is False
    calls = []
    def fake_prior(seeds, conditions=None):
        calls.append((tuple(seeds), conditions))
        return {"initial_controls": torch.ones(len(seeds), config.context_dimension),
                "equation_conditions": np.array(theta if conditions is None else conditions, dtype=np.float32)}
    monkeypatch.setattr(sampler.prior, "sample", fake_prior)
    full = sampler.sample_conditions(theta, [901, 902])
    assert full.shape == (2, config.full_dimension) and torch.isfinite(full).all()
    new_context, _ = normalizer.encode(full)
    torch.testing.assert_close(new_context, normalizer.encode_initial(torch.ones(2, config.context_dimension)), rtol=2.e-6, atol=2.e-6)
    output = sampler.generate([901, 902])
    assert output["latents"].shape == (2, config.full_dimension)
    assert output["diagnostics"]["pde_solves"] == 0
    assert output["diagnostics"]["known_prior_seeds"] != output["diagnostics"]["future_noise_seeds"]
    assert calls[0][0] == calls[1][0] and calls[1][1] is None


@pytest.mark.parametrize("mutate,match", [
    (lambda p: p.update(extra=True), "schema/fields"),
    (lambda p: p["model"].pop("initial_input.weight"), "tensor keys"),
    (lambda p: p["model"]["initial_input.weight"].fill_(float("nan")), "invalid conditional"),
    (lambda p: p["model"]["transform.rho_basis"].add_(.1), "immutable"),
    (lambda p: p["normalizer"]["fit_metadata"].update(context_slopes_or_future_values=True), "provenance"),
    (lambda p: p["normalizer"]["initial_prior"]["equation_config"].update(equation_family="fake"), "SHA256"),
    (lambda p: p.update(corpus_manifest_sha256="f"*64), "corpus identity"),
    (lambda p: p["training"].update(train_case_ids=["wrong"]), "case IDs"),
    (lambda p: p["training"].update(normalizer_sha256="f"*64), "normalizer SHA256"),
])
def test_strict_loader_rejects_tampering(mutate, match):
    *_, payload = checkpoint_payload()
    payload = deepcopy(payload)
    mutate(payload)
    with pytest.raises(ValueError, match=match):
        joint._sampler_from_initial_conditioned_payload(payload, "e"*64, "cpu", steps=2, churn=0.)


def test_normalizer_freeze_device_roundtrip_and_reject_downcast():
    config, raw, normalizer = data(torch.float32)
    restored = joint.VPFPInitialConditionedNormalizer.from_payload(normalizer.to_payload()).to("cpu")
    for a, b in zip(restored.encode(raw), normalizer.encode(raw)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    with torch.no_grad():
        restored.square_root.add_(.1)
    with pytest.raises(ValueError, match="buffers changed"):
        restored.encode(raw)
    normalizer.float()
    with pytest.raises(ValueError, match="float64|dtype"):
        normalizer.encode(raw)


def test_train_only_fit_and_invalid_config_or_prior():
    config, raw, normalizer = data()
    with pytest.raises(ValueError, match="train"):
        joint.VPFPInitialConditionedNormalizer.fit(config, raw, sample_ids=list(range(8)),
            initial_prior=prior_spec(config), split="validation")
    with pytest.raises(ValueError, match="config"):
        joint.VPFPInitialConditionedEDMConfig.from_dict({"unknown": True})
    changed = prior_spec(config).to_payload()
    changed["sha256"] = "0"*64
    with pytest.raises(ValueError, match="SHA256"):
        prior.KnownInitialPriorSpec.from_payload(changed)


def test_actual_tiny_known_prior_law_projection_no_pde(tmp_path, monkeypatch):
    from pde_agent_verifier_solver.plugins.vpfp.perturbation_backend import _legacy_api
    api = _legacy_api()
    modules = {name: importlib.import_module(f"agent_verifier.vpfp.{name}") for name in prior.MODULES}
    for name in ("online_training", "torch_reference"):
        monkeypatch.setattr(modules[name], "solve_torch_vpfp_batch", lambda *a, **k: pytest.fail("PDE solve forbidden"))
    config = fixture_config(3)
    decoder = replace(modules["continuous_hermite"].VPFPContinuousHermiteConfig(), latent_dim=config.full_dimension,
        micro_latent_dim=2, context_dim=4, hidden_width=4, hidden_layers=1, rho_max_mode_x=1, rho_max_mode_y=1,
        rho_positivity_gauge="phase_invariant_group_l1_v2", spatial_feature_modes=1, time_feature_modes=1,
        hermite_total_degree=1, coefficient_width=4, coefficient_blocks=1, structured_spectral_latent=True,
        structured_kinetic_global_dim=2, structured_local_channels=3, structured_local_spatial_mode=1,
        structured_time_basis="piecewise_cubic_hermite_v1", structured_time_control_count=3,
        structured_direct_coefficients=True, structured_residual_scale=0., time_domain=(0., 1.),
        collision_nu_range=(10., 30.), dtype="float32")
    checkpoint = tmp_path / "decoder.pt"
    torch.save({"decoder_config": decoder.to_dict(), "robust_answer_first_adapter": {
        "unused_neural_parameters_zeroed": True, "continuous_hidden_answer_is_primary": True,
        "generator_parameterization_public": False}}, checkpoint)
    digest = prior._file_sha(checkpoint)
    spec = VPFPOnlineModelSpec("tiny_strong", str(checkpoint), digest, str(checkpoint), digest, (10., 30.))
    binding = prior_spec(config, model_spec=spec, decoder=decoder.to_dict(),
        source_hashes={name: prior._file_sha(module.__file__) for name, module in modules.items()})
    # Use the complete normalized equation config exactly as real corpora do.
    payload = binding.to_payload()
    payload["equation_config"] = modules["high_fidelity"].VPFPHighFidelityConfig(collision_nu_range=(10., 30.)).to_dict()
    payload["sha256"] = prior._json_sha({k: v for k, v in payload.items() if k != "sha256"})
    binding = prior.KnownInitialPriorSpec(payload)
    sampler = prior.KnownInitialPrior(binding)
    first = sampler.sample([711], [[20., 1.]])
    again = sampler.sample([711], [[20., 1.]])
    torch.testing.assert_close(first["initial_controls"], again["initial_controls"], rtol=0, atol=0)
    assert first["initial_controls"].shape == (1, config.context_dimension)
    np.testing.assert_array_equal(first["equation_conditions"], [[20., 1.]])
    natural = sampler.sample([711])
    equation = modules["high_fidelity"].sample_continuous_equation_spec(seed=711,
        config=sampler._runtime["equation"], defer_rho_log_partition=True)
    np.testing.assert_array_equal(natural["equation_conditions"],
        np.asarray([[equation["collision_nu"], equation["temperature"]]], dtype=np.float32))
    with pytest.raises(ValueError, match="regime"):
        sampler.sample([711], [[40., 1.]])
    for endpoint in ([10., .8], [30., 1.2]):
        boundary = sampler.sample([711], [endpoint])
        np.testing.assert_array_equal(boundary["equation_conditions"], np.asarray([endpoint], dtype=np.float32))


def test_public_file_sha_loader_dispatch(tmp_path):
    from pde_agent_verifier_solver.plugins.vpfp.structured_diffusion import load_vpfp_diffusion_sampler
    *_, payload = checkpoint_payload()
    path = tmp_path / "native.pt"
    torch.save(payload, path)
    digest = sha256(path.read_bytes()).hexdigest()
    sampler = load_vpfp_diffusion_sampler(path, digest, steps=2)
    assert isinstance(sampler, joint.VPFPInitialConditionedBatchSampler)
    assert sampler.identity["checkpoint_sha256"] == digest
    with pytest.raises(ValueError, match="SHA256"):
        load_vpfp_diffusion_sampler(path, "0"*64, steps=2)
    symbolic = tmp_path / "symbolic.pt"
    symbolic.symlink_to(path)
    with pytest.raises(ValueError, match="non-symlink"):
        load_vpfp_diffusion_sampler(symbolic, digest, steps=2)


def test_real_shared_train_two_updates_and_safe_ema_reload(tmp_path, monkeypatch):
    from pde_agent_verifier_solver.plugins.vpfp.structured_diffusion import load_vpfp_diffusion_sampler
    root = Path(__file__).resolve().parents[1]
    loader = importlib.util.spec_from_file_location("conditional_shared_tiny", root / "scripts/train_vpfp_structured_diffusion.py")
    shared = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(shared)
    config, raw, normalizer = data(torch.float32)
    split = SimpleNamespace(latents=raw[:4].numpy(), conditions=np.array([[15., 1.], [20., 1.1], [25., .9], [28., 1.]], dtype=np.float32),
        sample_ids=np.arange(4), case_ids=np.array([f"strong_fixture:{i}" for i in range(4)]))
    class Split(SimpleNamespace):
        def __len__(self):
            return len(self.sample_ids)
    train = Split(**vars(split))
    validation = Split(**{**vars(split), "sample_ids": np.arange(4, 8), "case_ids": np.array([f"strong_fixture:{i}" for i in range(4, 8)])})
    manifest = {"latent_dimension": config.full_dimension, "base_decoder_sha256": "a"*64, "manifest_sha256": "c"*64}
    monkeypatch.setenv("SLURM_JOB_ID", "synthetic-mechanics")
    monkeypatch.setattr(shared, "read_manifest", lambda *a, **k: manifest)
    monkeypatch.setattr(shared, "load_split", lambda root, role, **k: train if role == "train" else validation)
    args = SimpleNamespace(checkpoint_seconds=(), seed=91, device="cpu", learning_rate=2.e-4,
        train_seconds=30., steps=2, batch_size=2, validate_every=1, source_revision="synthetic-test")
    output = tmp_path / "train"
    output.mkdir()
    path, report = shared.train(args, SimpleNamespace(base_decoder_sha256="a"*64), tmp_path / "unused-corpus", output,
        model_config_factory=lambda *a: config, model_factory=joint.VPFPInitialConditionedEDM,
        normalizer_factory=lambda cfg, values, sample_ids: joint.VPFPInitialConditionedNormalizer.fit(
            cfg, values, sample_ids=sample_ids, initial_prior=prior_spec(cfg)))
    assert report["final_step"] == 2 and report["stop_reason"] == "step_limit"
    sampler = load_vpfp_diffusion_sampler(path, report["checkpoint_sha256"], steps=2)
    assert sampler.identity["training"]["unique_training_answers"] == 4
    assert sampler.normalizer.fit_metadata["sample_ids"] == [0, 1, 2, 3]
    monkeypatch.setattr(sampler.prior, "sample", lambda seeds, conditions: {
        "initial_controls": torch.ones(len(seeds), config.context_dimension), "equation_conditions": np.asarray(conditions, dtype=np.float32)})
    assert sampler.sample_conditions([[20., 1.]], [101]).shape == (1, config.full_dimension)
