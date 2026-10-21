"""Tiny synthetic mechanics only; never execute real projection on login."""
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("known_initial_test", ROOT / "scripts/diagnose_vpfp_known_initial.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def layout():
    return {"control_count": 3, "rho_modes": 2, "hermite_channels": 3,
            "kinetic_modes": 2, "inactive_dimension": 2}


def controls(value=1.):
    return {"rho": np.array([value, value + .2], dtype=np.float32),
            "kinetic": np.arange(6, dtype=np.float32).reshape(3, 2) + value}


def test_t0_extraction_ignores_poisoned_future_slopes_and_inactive(layout):
    expected = controls()
    raw = module.initial_carrier(expected, layout)[None]
    rho = raw[:, :12].reshape(1, 2, 3, 2)
    kinetic = raw[:, 14:].reshape(1, 3, 2, 3, 2)
    rho[:, 1] = np.nan
    rho[:, 0, 1:] = np.nan
    raw[:, 12:14] = np.nan
    kinetic[:, :, 1] = np.nan
    kinetic[:, :, 0, 1:] = np.nan
    restored = module.initial_controls(raw, 0, layout)
    for key in expected:
        np.testing.assert_array_equal(restored[key], expected[key])
    raw[0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite t0"):
        module.initial_controls(raw, 0, layout)


def test_constant_carrier_zero_slopes_and_inactive(layout):
    raw = module.initial_carrier(controls(), layout)
    assert raw.dtype == np.float32
    rho, kinetic = raw[:12].reshape(2, 3, 2), raw[14:].reshape(3, 2, 3, 2)
    assert not np.count_nonzero(rho[1]) and not np.count_nonzero(kinetic[:, 1])
    assert not np.count_nonzero(raw[12:14])
    np.testing.assert_array_equal(rho[0, 0], rho[0, -1])
    np.testing.assert_array_equal(kinetic[:, 0, 0], kinetic[:, 0, -1])
    with pytest.raises(ValueError, match="dimension"):
        module.initial_controls(raw[None, :-1], 0, layout)
    with pytest.raises(ValueError, match="malformed"):
        module.initial_carrier({"rho": np.ones(3), "kinetic": controls()["kinetic"]}, layout)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_comparison_rejects_nonfinite(bad):
    with pytest.raises(ValueError, match="finite"):
        module.comparison([bad], [1.])


def test_original_unit_error_is_not_equivalence_or_normalized_pointwise():
    metrics = module.comparison([2., 0.], [1., 0.])
    assert metrics["rmse"] == pytest.approx(2**-.5)
    assert metrics["relative_l2"] == pytest.approx(1.)
    assert metrics["prediction_std"] == 1.
    assert metrics["float32_replay_allclose"] is False
    assert np.isfinite(module.comparison([1.], [0.])["relative_l2"])


def fake_physics():
    component = {"base_logit": 0., "base_mean": [2., -1.], "base_log_std": [0., np.log(2.)],
        "correlation": .5, **{name: [] for name in
            ("weight_modes", "mean_x_modes", "mean_y_modes", "log_std_x_modes", "log_std_y_modes")}}
    initial = {"family": module.FAMILY, "g0_components": [component], "collision_nu": 20.,
               "temperature": 1., "rho_log_partition": None, "seed": 9}
    physical = SimpleNamespace(_spatial_mode_field=lambda *args, **kwargs: np.zeros((2, 2)),
        _evaluate_rho0=lambda *args: np.full((2, 2), 3.))
    grid = SimpleNamespace(nx=2, ny=2, nvx=2, nvy=2, dvx=1., dvy=1.)
    axes = {"x": np.arange(2), "y": np.arange(2), "vx": np.array([-1., 1.]), "vy": np.array([-1., 1.])}
    return initial, physical, grid, axes


def test_fullspace_gaussian_moments_do_not_use_velocity_truncation():
    initial, physical, grid, axes = fake_physics()
    actual = module.analytic_initial_macros(initial, grid, physical, axes)
    for key, expected in {"rho": 3., "jx": 6., "jy": -3., "K": 15.}.items():
        np.testing.assert_allclose(actual[key], expected)
    # A second component with equal weight changes the analytic moment,
    # independently of correlation and of the very small velocity probe box.
    initial["g0_components"].append({**initial["g0_components"][0], "base_mean": [-2., 1.]})
    actual = module.analytic_initial_macros(initial, grid, physical, axes)
    np.testing.assert_allclose(actual["jx"], 0.)
    np.testing.assert_allclose(actual["K"], 15.)
    initial["family"] = "single_maxwellian"
    with pytest.raises(ValueError, match="frozen rich"):
        module.analytic_initial_macros(initial, grid, physical, axes)


def test_discrete_moments_use_uniform_not_trapezoidal():
    _, _, grid, axes = fake_physics()
    actual = module.discrete_initial_macros(np.ones((2, 2, 2, 2)), grid, axes)
    np.testing.assert_array_equal(actual["rho"], np.full((2, 2), 4.))
    np.testing.assert_array_equal(actual["K"], np.full((2, 2), 4.))
    np.testing.assert_array_equal(actual["jx"], np.zeros((2, 2)))


@dataclass
class Time:
    t0: float = 0.
    t1: float = 1.
    nt: int = 3


def test_reconstruction_uses_original_helpers_no_solver_or_future(layout):
    initial, physical, grid, axes = fake_physics()
    seen = {}
    def sample(**kwargs):
        seen["sample"] = kwargs
        return initial
    physical.sample_continuous_equation_spec = sample
    def pack(specs, dtype):
        assert specs == [initial] and dtype == "float32"
        return "packed"
    def prepare(parameters, **kwargs):
        assert parameters == "packed" and kwargs["time"].nt == 3
        return {"f0": torch.ones(1, 2, 2, 2, 2), "source_frames": object(),
                "collision_nu": torch.tensor([20.]), "temperature": torch.tensor([1.])}
    def uniform(torch_module, f0, **kwargs):
        seen["uniform"] = kwargs
        return f0.sum((-2, -1)), f0 / 4
    def project(carrier, conditions):
        assert carrier["rho"].shape == (1, 3, 2, 2)
        assert carrier["g"].shape == (1, 3, 2, 2, 2, 2)
        torch.testing.assert_close(carrier["g"][:, 0], carrier["g"][:, -1])
        torch.testing.assert_close(conditions, torch.tensor([[20., 1.]]))
        return {"latents": torch.from_numpy(module.initial_carrier(controls(), layout)[None])}
    reference = SimpleNamespace(pack_torch_equation_specs=pack, evaluate_torch_equation_parameter_batch=prepare,
        _uniform_rho_g=uniform, solve_torch_vpfp_batch=lambda *args, **kwargs: pytest.fail("no PDE solve allowed"))
    runtime = SimpleNamespace(physical=physical, grid=grid, axes=axes, reference=reference, equation=object(),
        time=Time(), device="cpu", model=object(), projector=SimpleNamespace(project=project),
        api=SimpleNamespace(ieee_float32_coordinate_matmul=lambda _: nullcontext()),
        generation=SimpleNamespace(solver=lambda: SimpleNamespace(rho_floor=1.e-9)))
    actual = module.reconstruct_initial(runtime, 9, [20., 1.], layout)
    assert actual["rho_log_partition_deferred"]
    assert seen["sample"] == {"seed": 9, "config": runtime.equation, "defer_rho_log_partition": True}
    assert seen["uniform"]["rho_floor"] == 1.e-9
    for key in controls():
        np.testing.assert_array_equal(actual["controls"][key], controls()[key])
    with pytest.raises(ValueError, match="seed.*condition"):
        module.reconstruct_initial(runtime, 9, [21., 1.], layout)


def test_native_decoder_queries_only_exact_t0(layout):
    _, _, grid, axes = fake_physics()
    def evaluate(model, latent, points, **kwargs):
        assert np.all(points[:, 0] == 0.)
        assert kwargs["chunk_size"] == 256
        np.testing.assert_array_equal(module.initial_controls(latent[None], 0, layout)["rho"], controls()["rho"])
        return {"rho": np.ones(4), "momentum": np.arange(8).reshape(4, 2), "kinetic_energy": np.ones(4) * 3}
    api = SimpleNamespace(ieee_float32_coordinate_matmul=lambda _: nullcontext(),
        evaluate_continuous_hermite_macro_points=evaluate)
    actual = module.native_initial_macros(SimpleNamespace(api=api, model=None, grid=grid, axes=axes, time=Time()),
        controls(), [20., 1.], layout)
    assert actual["rho"].shape == (2, 2)
    np.testing.assert_array_equal(actual["jy"].ravel(), [1, 3, 5, 7])


@pytest.fixture
def orchestration(tmp_path, monkeypatch, layout):
    monkeypatch.setenv("SLURM_JOB_ID", "1234")
    monkeypatch.setattr(module, "_SCRATCH_ROOT", tmp_path)
    decoder = tmp_path / "decoder.pt"
    decoder.write_bytes(b"synthetic decoder never loaded")
    digest = module._file_sha(decoder)
    model_spec = module.VPFPOnlineModelSpec("synthetic_strong", str(decoder), digest,
        str(decoder), digest, (10., 30.))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"vpfp": asdict(model_spec)}))
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "manifest.json").write_text("{}")
    manifest = {"model_spec": asdict(model_spec), "base_decoder_sha256": digest,
                "manifest_sha256": "b" * 64, "provenance": {"synthetic": True}}
    conditions = np.array([[12.5 + 5 * i, .9 + .2 * j] for i in range(4) for j in range(2)], dtype=np.float32)
    def split(offset):
        initial = module.initial_carrier(controls(), layout)
        raw = np.tile(initial, (16, 1))
        # Earlier rows have higher seeds. Selection must be metadata-only.
        return SimpleNamespace(sample_ids=np.r_[np.arange(8) + offset + 100, np.arange(8) + offset],
            conditions=np.tile(conditions, (2, 1)), latents=raw)
    splits = {"train": split(1000), "validation": split(2000)}
    reads, reconstructs = [], []
    def read(root, verify_hashes):
        reads.append(verify_hashes)
        return manifest
    monkeypatch.setattr(module, "read_manifest", read)
    def load(root, name, verify_hashes):
        assert name in ("train", "validation") and verify_hashes is False
        return splits[name]
    monkeypatch.setattr(module, "load_split", load)
    monkeypatch.setattr(module, "_layout", lambda _: layout)
    monkeypatch.setattr(module, "_runtime", lambda *args: SimpleNamespace(frozen_files={}, grid=SimpleNamespace(nx=2, ny=2)))
    macros = {key: np.ones((2, 2)) for key in ("rho", "jx", "jy", "K")}
    def reconstruct(runtime, seed, condition, supplied_layout):
        reconstructs.append(seed)
        assert supplied_layout == layout
        return {"controls": controls(), "analytic": {key: value * 2 for key, value in macros.items()},
                "discrete": macros, "equation_spec_sha256": "f" * 64, "component_count": 4,
                "condition": condition.tolist(), "rho_log_partition_deferred": True}
    monkeypatch.setattr(module, "reconstruct_initial", reconstruct)
    monkeypatch.setattr(module, "native_initial_macros", lambda *args: macros)
    return SimpleNamespace(config=config, corpus=corpus, output=tmp_path / "output", manifest=manifest,
        splits=splits, reads=reads, reconstructs=reconstructs, reconstruct=reconstruct)


def test_run_freezes_stratified_train_validation_and_separates_projection_error(orchestration):
    fixture = orchestration
    result = module.run(fixture.config, fixture.corpus, fixture.output, device="cpu")
    assert result["execution_passed"] and result["mechanical_replay_passed"]
    assert result["scientific_distribution_validation"] is False
    assert result["physical_projection_qualified"] is False
    assert result["new_pde_solves"] == 0 and result["training_run"] is False
    assert len(result["cases"]) == 16
    assert fixture.reconstructs == list(range(1000, 1008)) + list(range(2000, 2008))
    assert fixture.reads == [True, True]
    assert result["cases"][0]["macros"]["projected_vs_analytic"]["K"]["relative_l2"] == .5
    plan = json.loads((fixture.output / "plan.json").read_text())
    assert plan["published_future_trajectory"] is False
    assert plan["macro_equivalence_margins"] is None
    assert result["plan_sha256"] == module._file_sha(fixture.output / "plan.json")
    assert json.loads((fixture.output / "result.json").read_text()) == result


def test_missing_slurm_fails_before_output(orchestration, monkeypatch):
    fixture = orchestration
    monkeypatch.delenv("SLURM_JOB_ID")
    with pytest.raises(RuntimeError, match="Slurm"):
        module.run(fixture.config, fixture.corpus, fixture.output)
    assert not fixture.output.exists()


def test_existing_output_is_preserved(orchestration):
    fixture = orchestration
    fixture.output.mkdir()
    (fixture.output / "keep").write_text("preserve")
    with pytest.raises(ValueError, match="fresh"):
        module.run(fixture.config, fixture.corpus, fixture.output)
    assert (fixture.output / "keep").read_text() == "preserve"


def test_changed_decoder_rejected_before_projection(orchestration):
    fixture = orchestration
    decoder = Path(fixture.manifest["model_spec"]["base_decoder_checkpoint"])
    decoder.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA256"):
        module.run(fixture.config, fixture.corpus, fixture.output)
    assert not fixture.reconstructs
    assert "SHA256" in json.loads((fixture.output / "result.json").read_text())["failure"]["message"]


def test_wrong_frozen_spec_fails_before_projection(orchestration):
    fixture = orchestration
    fixture.manifest["model_spec"]["collision_nu_range"] = [11., 30.]
    with pytest.raises(ValueError, match="spec differs"):
        module.run(fixture.config, fixture.corpus, fixture.output)
    assert not fixture.reconstructs


def test_missing_stratum_fails_before_runtime(orchestration):
    fixture = orchestration
    fixture.splits["train"].conditions[:] = [20., 1.]
    with pytest.raises(ValueError, match="no case"):
        module.run(fixture.config, fixture.corpus, fixture.output)
    assert not fixture.reconstructs


def test_failure_keeps_preceding_cases_and_reason(orchestration, monkeypatch):
    fixture = orchestration
    def fail_after_one(*args):
        if fixture.reconstructs:
            raise ValueError("synthetic projection failure")
        return fixture.reconstruct(*args)
    monkeypatch.setattr(module, "reconstruct_initial", fail_after_one)
    with pytest.raises(ValueError, match="synthetic projection"):
        module.run(fixture.config, fixture.corpus, fixture.output)
    result = json.loads((fixture.output / "result.json").read_text())
    assert result["execution_passed"] is False and len(result["cases"]) == 1
    assert result["failure"]["message"] == "synthetic projection failure"
    assert (fixture.output / "plan.json").is_file()


def test_replay_mismatch_is_failed_gate_not_silent_projection(orchestration):
    fixture = orchestration
    fixture.splits["train"].latents[:, 0] += 1
    with pytest.raises(ValueError, match="does not replay"):
        module.run(fixture.config, fixture.corpus, fixture.output)
    result = json.loads((fixture.output / "result.json").read_text())
    assert not result["mechanical_replay_passed"] and len(result["cases"]) == 16


def test_final_hash_revalidation_rejects_mutation(orchestration, monkeypatch):
    fixture = orchestration
    def change_file(*args):
        result = fixture.reconstruct(*args)
        fixture.config.write_text("changed during diagnostic")
        return result
    monkeypatch.setattr(module, "reconstruct_initial", change_file)
    with pytest.raises(ValueError, match="frozen initial-audit input changed"):
        module.run(fixture.config, fixture.corpus, fixture.output)
    assert json.loads((fixture.output / "result.json").read_text())["execution_passed"] is False


def test_source_hash_mismatch_fails_before_loading_decoder(tmp_path, monkeypatch):
    fake_source = tmp_path / "source.py"
    fake_source.write_text("synthetic")
    monkeypatch.setattr(module, "_legacy_api", lambda: None)
    monkeypatch.setattr(module.importlib, "import_module", lambda name: SimpleNamespace(__file__=str(fake_source)))
    monkeypatch.setattr(module.torch, "load", lambda *args, **kwargs: pytest.fail("must check source before loading"))
    manifest = {"provenance": {"physical_module_sha256": {name: "0" * 64 for name in module.PHYSICAL_MODULES}}}
    with pytest.raises(ValueError, match="frozen initial-audit input changed"):
        module._runtime(manifest, None, "cpu")


def test_tiny_actual_law_materializer_projector_and_native_t0(monkeypatch):
    """4x4x3x3, degree one; real interfaces, not a real-data fidelity test."""
    api = module._legacy_api()
    physical = importlib.import_module("agent_verifier.vpfp.high_fidelity")
    reference = importlib.import_module("agent_verifier.vpfp.torch_reference")
    hermite = importlib.import_module("agent_verifier.vpfp.continuous_hermite")
    online = importlib.import_module("agent_verifier.vpfp.online_training")
    operators = importlib.import_module("agent_verifier.vpfp.operators")
    monkeypatch.setattr(reference, "solve_torch_vpfp_batch", lambda *a, **k: pytest.fail("PDE solve forbidden"))
    monkeypatch.setattr(online, "solve_torch_vpfp_batch", lambda *a, **k: pytest.fail("PDE solve forbidden"))
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        cfg = replace(hermite.VPFPContinuousHermiteConfig(), latent_dim=1,
            micro_latent_dim=2, context_dim=4, hidden_width=4, hidden_layers=1,
            rho_max_mode_x=1, rho_max_mode_y=1, rho_positivity_gauge="phase_invariant_group_l1_v2",
            spatial_feature_modes=1, time_feature_modes=1, hermite_total_degree=1,
            coefficient_width=4, coefficient_blocks=1, structured_spectral_latent=True,
            structured_kinetic_global_dim=2, structured_local_channels=3,
            structured_local_spatial_mode=1, structured_time_basis="piecewise_cubic_hermite_v1",
            structured_time_control_count=5, structured_direct_coefficients=True,
            structured_residual_scale=0., time_domain=(0., 1.), collision_nu_range=(10., 30.), dtype="float32")
        cfg = replace(cfg, latent_dim=cfg.structured_latent_layout()["total_latent_dim"])
        layout = {"control_count": 5, "rho_modes": 8, "hermite_channels": 3,
                  "kinetic_modes": 9, "inactive_dimension": 2}
        grid = operators.VPFPGrid(nx=4, ny=4, nvx=3, nvy=3, v_bounds=(-2., 2., -2., 2.))
        time = operators.VPFPTime(t0=0., t1=1., nt=5)
        equation = physical.VPFPHighFidelityConfig(collision_nu_range=(10., 30.))
        initial = physical.sample_continuous_equation_spec(seed=7248, config=equation, defer_rho_log_partition=True)
        assert initial["family"] == module.FAMILY and "equation_family" not in initial
        condition = np.asarray([initial["collision_nu"], initial["temperature"]], dtype=np.float32)
        model = api.build_deterministic_direct_model(cfg, device="cpu")
        projector = online.VPFPTorchStructuredProjector(cfg, grid=grid, time=time, device="cpu")
        runtime = SimpleNamespace(api=api, physical=physical, reference=reference, equation=equation,
            grid=grid, time=time, model=model, projector=projector, axes=api.make_axes(grid), device="cpu",
            generation=online.VPFPOnlineGenerationConfig())
        reconstructed = module.reconstruct_initial(runtime, 7248, condition, layout)
        macros = module.native_initial_macros(runtime, reconstructed["controls"], condition, layout)
        assert all(value.shape == (4, 4) and np.isfinite(value).all() for value in macros.values())
        assert reconstructed["rho_log_partition_deferred"] is True
        assert 4 <= reconstructed["component_count"] <= 10
        # Independently prepare on the ORIGINAL time grid, then change all
        # future density/velocity fields. Original t0 must still replay.
        packed = reference.pack_torch_equation_specs([initial], dtype="float32")
        prepared = reference.evaluate_torch_equation_parameter_batch(packed, grid=grid, time=time, device="cpu")
        rho, g = reference._uniform_rho_g(torch, prepared["f0"], grid=grid, rho_floor=runtime.generation.solver().rho_floor)
        carrier = {"rho": rho[:, None].repeat(1, 5, 1, 1), "g": g[:, None].repeat(1, 5, 1, 1, 1, 1)}
        carrier["rho"][:, 1:] *= 1.1
        carrier["g"][:, 1:] = torch.flip(carrier["g"][:, 1:], dims=(-1,))
        projected = projector.project(carrier, torch.from_numpy(condition[None]))
        other = module.initial_controls(projected["latents"].numpy(), 0, layout)
        for key in other:
            np.testing.assert_allclose(other[key], reconstructed["controls"][key], rtol=2.e-6, atol=2.e-6)
    finally:
        torch.set_num_threads(old_threads)
