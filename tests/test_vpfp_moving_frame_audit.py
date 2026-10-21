"""Small synthetic/direct-decoder fixtures; scheduled tests, not physical calibration."""
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pde_agent_verifier_solver.plugins.vpfp import moving_frame as mf
from pde_agent_verifier_solver.plugins.vpfp import moving_frame_audit as audit
from pde_agent_verifier_solver.plugins.vpfp import regime_fidelity_coverage as core


@pytest.fixture
def tiny():
    spec = importlib.util.spec_from_file_location("moving_audit_tiny_helpers", Path(__file__).with_name("test_vpfp_moving_frame.py"))
    helpers = importlib.util.module_from_spec(spec); spec.loader.exec_module(helpers)
    base, value, config, z, theta = helpers.tiny(dtype="float32")
    model = mf.MovingFrameDecoder(base, config, value_config=value)
    m = helpers.constant_macro(config).float()
    raw = mf.lift_value_controls(z, value).detach().numpy()
    plan = core.VPFPRegimeFeaturePlan(time_fractions=(0., .5, 1.), grid_sizes=(3, 5),
        phase_probes_per_time=2, independent_source_probe_count=2, chunk_size=32)
    return SimpleNamespace(model=model, base=base, value=value, config=config, z=z, m=m,
        raw=raw, theta=theta.numpy()[None], plan=plan, helpers=helpers)


def extract(fixture, **overrides):
    return audit.extract_moving_frame_regime_features(fixture.model, overrides.get("raw", fixture.raw),
        overrides.get("macro", fixture.m.numpy()), fixture.theta, plan=fixture.plan)


def test_A_direct_macro_units_and_schema_not_B_substitution(tiny, monkeypatch):
    def forbidden(*args, **kwargs): raise AssertionError("old B moment/source path is forbidden for A audit")
    monkeypatch.setattr(core, "_macro_values", forbidden)
    analytic = __import__("agent_verifier.vpfp.analytic_source", fromlist=["unused"])
    monkeypatch.setattr(analytic, "manufacture_standard_vpfp_source_analytic", forbidden)
    matrix, descriptors, report = extract(tiny)
    assert report["cases"][0]["hard_valid"], report["cases"][0]
    assert report["cases"][0]["numerically_uncertain"]
    positions = {d["name"]: i for i, d in enumerate(descriptors)}
    temperature = float(tiny.theta[0, 1])
    np.testing.assert_allclose(matrix[0, positions["time0.jx_coherent"]], .4/np.sqrt(temperature), rtol=2.e-6)
    np.testing.assert_allclose(matrix[0, positions["time0.jy_coherent"]], -.3/np.sqrt(temperature), rtol=2.e-6)
    np.testing.assert_allclose(matrix[0, positions["time0.thermal_energy_mean"]], 1./temperature, rtol=2.e-6)
    np.testing.assert_allclose(matrix[0, positions["time0.thermal_anisotropy"]], np.sqrt(.6**2+4*.2**2)/temperature, rtol=2.e-6)
    assert not report["B_analytic_source_substituted_for_A"]
    assert not report["full_velocity_source_quadrature_completed"]
    independent = report["cases"][0]["source_probes"]["independent_source_check"]
    assert independent["execution_passed"], independent
    assert not independent["is_convergence_certificate"]
    json.dumps(report, allow_nan=False)


def test_A_and_B_share_fixed_points_units_and_frozen_draw_identity(tiny):
    a, descriptors, ar = extract(tiny)
    b, old_descriptors, br = core.extract_vpfp_regime_features(tiny.base, tiny.raw, tiny.theta, plan=tiny.plan)
    assert descriptors == old_descriptors
    assert ar["cases"][0]["source_probes"]["points_txyvv"] == br["cases"][0]["source_probes"]["points_txyvv"]
    paired = audit.paired_decoder_comparison(a, ar, b, br, descriptors)
    assert paired["same_native_draw_and_condition_verified"]
    assert paired["both_valid_count"] == 1
    assert not paired["paired_truth_claimed"] and not paired["A_or_B_is_truth"]
    assert any(abs(row["signed_mean_A_minus_B"]) > .1 for row in paired["rows"])
    bad = {**br, "cases": [{**br["cases"][0], "input_answer_sha256": "0"*64}]}
    with pytest.raises(ValueError, match="same frozen draw"):
        audit.paired_decoder_comparison(a, ar, b, bad, descriptors)


def test_A_macro_time_derivative_is_physical_time_not_tau(tiny):
    u, local, chol = mf.split_macro_controls(tiny.m.clone(), tiny.config)
    u[0, 0] = torch.tensor([.2, .35, .5]); u[0, 1] = 0.
    carrier = {"m": mf.pack_macro_controls(u, local, chol, tiny.config), "z": tiny.z}
    q = np.asarray([[2.2, .1, .3], [4.8, .1, .3]])
    values = audit._physical_macro_chunk(tiny.model, q, carrier, torch.as_tensor(tiny.theta[0]))
    bg = tiny.base.config.equation.background_density
    np.testing.assert_allclose(values["dt_jx"], .1*bg, rtol=2.e-6, atol=1.e-8)
    np.testing.assert_allclose(values["dt_K"], .01*q[:, 0]*bg, rtol=3.e-6, atol=1.e-8)


def test_thermal_measurement_uses_direct_C_at_large_drift(tiny, monkeypatch):
    backend = audit._backend()
    def large_drift(decoder, query, carrier, condition):
        rho = 1+query[:, 0]*0
        u = query.new_tensor([1.e12, -1.e12])
        covariance = query.new_tensor([[1., .2], [.2, 2.]])[None].expand(len(query), 2, 2)
        return {"rho": rho, "momentum": rho[:, None]*u, "kinetic_energy": rho*1.e24,
            "raw_second_moment": rho[:, None]*query.new_tensor([1.e24, -1.e24, 1.e24]),
            "electric": query[:, :2]*0, "central_covariance": covariance}
    monkeypatch.setattr(backend, "evaluate_moving_frame_macros", large_drift)
    values = audit._physical_macro_chunk(tiny.model, np.asarray([[3., .2, .4]]), {}, torch.as_tensor(tiny.theta[0]))
    np.testing.assert_allclose(values["thermal_energy"], 1.5, rtol=0., atol=0.)


def test_full_lift_malformed_A_carrier_is_not_projected(tiny):
    raw = tiny.raw.copy()
    # First rho slope is finite but deliberately inconsistent with values.
    raw[0, tiny.value.rho_dimension] = .1
    matrix, _, report = extract(tiny, raw=raw)
    assert not report["cases"][0]["hard_valid"]
    assert "slopes" in report["cases"][0]["failure"]["message"]
    assert report["cases"][0]["budget_defined_mask"] == [False]*3
    np.testing.assert_array_equal(matrix, 0.)


def test_failed_macro_keeps_attempt_and_B_identity(tiny):
    m = tiny.m.numpy().copy(); m[0, 0] = np.nan
    matrix, descriptors, report = extract(tiny, macro=m)
    case = report["cases"][0]
    assert not case["hard_valid"] and case["numerically_uncertain"]
    assert case["source_B_answer_sha256"] == sha256(tiny.raw[0].tobytes()+tiny.theta[0].tobytes()).hexdigest()
    assert len(report["cases"]) == 1 and matrix.shape == (1, len(descriptors))
    json.dumps(report, allow_nan=False)


def test_carrier_hash_binds_m_config_and_condition(tiny):
    identity = tiny.model.identity()
    initial = audit.carrier_input_sha256(tiny.raw[0], tiny.m[0].numpy(), tiny.theta[0], identity)
    m = tiny.m[0].numpy().copy(); m[0] += .1
    assert initial != audit.carrier_input_sha256(tiny.raw[0], m, tiny.theta[0], identity)
    assert initial != audit.carrier_input_sha256(tiny.raw[0], tiny.m[0].numpy(), tiny.theta[0], {**identity, "floor": "tampered"})
    with pytest.raises(ValueError, match="identity mismatch"):
        audit.extract_moving_frame_regime_features(tiny.model, tiny.raw, tiny.m.numpy(), tiny.theta,
            plan=tiny.plan, expected_identity={})


def test_independent_finite_differences_handle_physical_time_endpoints():
    class Toy:
        base_decoder = SimpleNamespace(config=SimpleNamespace(time_domain=(2., 5.), equation=SimpleNamespace(x_domain=(6., 7.))))
        def fields(self, points, carrier, *, equation_condition):
            t, vx, vy = points[:, 0], points[:, 3], points[:, 4]
            p = 1+.1*vx+.3*vy+.2*vx**2+.4*vy**2
            return {"f": (1+t)*p, "electric": points.new_tensor([.1, -.2])[None].expand(len(points), 2)}
    q = torch.tensor([[2., 0., 0., .2, -.4], [5., 5.99, 6.99, -.3, .5]], dtype=torch.float64)
    theta = torch.tensor([.3, 1.2], dtype=torch.float64)
    actual = audit._finite_difference_source(Toy(), q, {}, theta, np.full(5, 1.e-3))
    t, vx, vy = q[:, 0], q[:, 3], q[:, 4]
    p = 1+.1*vx+.3*vy+.2*vx**2+.4*vy**2
    f, gx, gy = (1+t)*p, (1+t)*(.1+.4*vx), (1+t)*(.3+.8*vy)
    expected = p+.1*gx-.2*gy-theta[0]*(2*f+vx*gx+vy*gy+theta[1]*1.2*(1+t))
    torch.testing.assert_close(actual, expected, rtol=1.e-8, atol=1.e-8)


def test_A_autodiff_source_matches_independent_drifting_Gaussian_formula(tiny):
    from pde_agent_verifier_solver.plugins.vpfp.moving_frame_backend import manufacture_moving_frame_source
    macro = tiny.helpers.constant_macro(tiny.config, covariance=((1., 0.), (0., 1.))).float()
    carrier = {"m": macro, "z": tiny.z}
    q = tiny.helpers.points()
    theta = torch.as_tensor(tiny.theta[0])
    actual = manufacture_moving_frame_source(tiny.model, q, carrier, theta)
    u = macro[0, :2*tiny.config.control_count].reshape(2, -1)[:, 0].double()
    centered = q[:, 3:5]-u
    rho = tiny.base.config.equation.background_density
    f = rho/(2*torch.pi)*torch.exp(-.5*centered.square().sum(-1))
    collision = f*(2-(q[:, 3:5]*centered).sum(-1)+theta[1].double()*(centered.square().sum(-1)-2))
    torch.testing.assert_close(actual["f"], f, rtol=2.e-12, atol=2.e-13)
    torch.testing.assert_close(actual["source"], -theta[0].double()*collision, rtol=2.e-10, atol=2.e-12)


def test_complete_pair_receipt_freezes_A_m_and_B_z_without_sampling(tiny, tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[1]/"scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location("moving_audit_driver_test", scripts/"run_vpfp_regime_fidelity_coverage.py")
    driver = importlib.util.module_from_spec(spec); spec.loader.exec_module(driver)
    from pde_agent_verifier_solver.plugins.vpfp.coherent_diffusion import VPFPCoherentMacroConfig
    from pde_agent_verifier_solver.plugins.vpfp import moving_frame_binding as binding
    macro_cfg = VPFPCoherentMacroConfig(tiny.value.full_config, maximum_mode=1, coherent_mode=1)
    monkeypatch.setattr(binding, "validate_hierarchical_payloads", lambda *args, **kwargs: (macro_cfg, tiny.value))
    directory = tmp_path/"generated"; directory.mkdir()
    manifest_sha, decoder_sha = "a"*64, "b"*64
    entries = {}
    for role in ("macro", "conditional"):
        path = tmp_path/(role+".pt")
        torch.save({"corpus_manifest_sha256": manifest_sha}, path)
        entries[role] = {"checkpoint": str(path), "checkpoint_sha256": driver.file_sha(path)}
    conditions = driver.balanced_conditions(tiny.value, 128, 12)
    raw = np.repeat(tiny.raw, 128, axis=0)
    np.save(directory/"generated-coefficients.npy", raw)
    np.save(directory/"generated-conditions.npy", conditions)
    np.save(directory/"generated-macro.npy", np.repeat(tiny.m.numpy(), 128, axis=0))
    receipt = {"schema": "vpfp_coherent_generated_native_pair_v1", "seeds": list(range(128)),
        "checkpoint_bindings": entries, "macro_config": macro_cfg.to_dict(), "micro_config": tiny.value.to_dict(),
        "coefficients_sha256": driver.file_sha(directory/"generated-coefficients.npy"),
        "conditions_sha256": driver.file_sha(directory/"generated-conditions.npy"),
        "macro_sha256": driver.file_sha(directory/"generated-macro.npy"),
        "decoder_views": ["B_old_full_decoder", "A_continuous_moving_frame"], "full_coefficients_are_A_answer": False}
    (directory/"generated-provenance.json").write_text(json.dumps(receipt))
    spec = SimpleNamespace(base_decoder_sha256=decoder_sha)
    manifest = {"latent_dimension": tiny.value.full_dimension, "manifest_sha256": manifest_sha}
    edges = driver._edges(np.linspace(*tiny.value.collision_nu_range, 5), np.linspace(*tiny.value.temperature_range, 3))
    result = driver.freeze_generated(directory, manifest=manifest, spec=spec, count=128, edges=edges)
    assert result["status"] == "ready_paired_A_B"
    assert len(result["selected_same_draw_noise_seeds"]) == 128
    assert result["checkpoint_sha256"] is None and len(result["checkpoint_pair_sha256"]) == 64
    driver.check_frozen(result["files"])
    receipt["micro_config"]["schema"] = "tampered"
    (directory/"generated-provenance.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="config/shape"):
        driver.freeze_generated(directory, manifest=manifest, spec=spec, count=128, edges=edges)
