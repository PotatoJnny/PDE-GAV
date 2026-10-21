"""Tiny CPU diagnostic mechanics; no scientific plasma validation or jobs."""

from dataclasses import replace
import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture
def cli():
    path = Path(__file__).resolve().parents[1] / "scripts/diagnose_vpfp_noise_geometry.py"
    spec = importlib.util.spec_from_file_location("vpfp_geometry_cli_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_jet_summary_preserves_time_windows_and_case_before_ensemble_rms(cli):
    plan = cli.VPFPDerivativeFeaturePlan(time_fractions=(0., .25, .5, 1.),
        phase_probe_count=3, macro_probe_count=2)
    descriptors, columns = [], []
    for quantity in ("f", "rho", "momentum_x", "momentum_y", "kinetic_energy"):
        count = plan.phase_probe_count if quantity == "f" else plan.macro_probe_count
        for index in range(4 * count):
            for component in ("value", "dt", "dx", "dy"):
                descriptors.append({"name": f"{quantity}.probe{index}.{component}"})
                columns.append([1., 3.] if index // count < 2 else [10., 20.])
    matrix = np.asarray(columns).T
    report = cli.jet_summary(matrix, descriptors, plan)
    for quantity in ("f", "rho", "momentum_x", "momentum_y", "kinetic_energy"):
        for component in ("value", "dt", "dx", "dy"):
            prefix = f"{quantity}.{component}."
            assert report[prefix + "early"]["mean_rms"] == 2.
            assert report[prefix + "late"]["mean_rms"] == 15.
            per_case = np.sqrt(np.array([50.5, 204.5]))
            assert report[prefix + "all"]["mean_rms"] == pytest.approx(per_case.mean())
            assert report[prefix + "all"]["median_rms"] == pytest.approx(np.median(per_case))
            assert report[prefix + "all"]["p90_rms"] == pytest.approx(np.quantile(per_case, .9))
            assert report[prefix + "all"]["maximum_rms"] == per_case.max()
            assert report[prefix + "early"]["mean_rms"] != pytest.approx(np.sqrt(5.))


@pytest.fixture
def direct():
    from pde_agent_verifier_solver.plugins.vpfp.perturbation_backend import _legacy_api
    api = _legacy_api()
    hermite = importlib.import_module("agent_verifier.vpfp.continuous_hermite")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    config = replace(hermite.VPFPContinuousHermiteConfig(), latent_dim=1,
        latent_input_scale=3.7, structured_local_scale=2.4,
        micro_latent_dim=2, context_dim=4, hidden_width=4, hidden_layers=1,
        rho_max_mode_x=1, rho_max_mode_y=1, rho_positivity_gauge="phase_invariant_group_l1_v2",
        spatial_feature_modes=1, time_feature_modes=1, hermite_total_degree=1,
        coefficient_width=4, coefficient_blocks=1, structured_spectral_latent=True,
        structured_kinetic_global_dim=2, structured_local_channels=3,
        structured_local_spatial_mode=1, structured_time_basis="piecewise_cubic_hermite_v1",
        structured_time_control_count=3, structured_direct_coefficients=True,
        structured_residual_scale=0., time_domain=(2., 5.), dtype="float64")
    layout = config.structured_latent_layout()
    config = replace(config, latent_dim=layout["total_latent_dim"])
    model = api.build_deterministic_direct_model(config, device="cpu")
    latent = np.zeros(config.latent_dim)
    start = layout["rho_latent_dim"] + config.structured_kinetic_global_dim
    controls = latent[start:].reshape(3, 6, 9)
    tau = np.array([0., .5, 1.])
    controls[0, :3, 0] = 1.2
    controls[1, :3, 0] = .2 + .6 * tau
    controls[1, 3:, 0] = .6
    controls[2, :3, 0] = .3 - .1 * tau
    controls[2, 3:, 0] = -.1
    points = np.array([[2.3, .1, .3, .2, -.4], [3.2, 2., .8, 3., 1.], [4.7, 4., 5., -2., .1]])
    yield SimpleNamespace(model=model, latent=latent, points=points, api=api)
    torch.set_num_threads(threads)


def test_raw_replay_matches_analytic_scale_and_direct_decoder_autodiff(cli, direct):
    norm, angular = cli.raw_amplitude_jets(direct.model, direct.points, direct.latent)
    cfg = direct.model.config
    tau = (direct.points[:, 0] - 2.) / 3.
    factor = cfg.latent_input_scale * cfg.structured_local_scale / 3.
    amplitude = factor * np.column_stack((np.full(len(tau), 1.2), .2 + .6*tau, .3 - .1*tau))
    expected_norm = np.linalg.norm(amplitude, axis=1)
    np.testing.assert_allclose(norm, expected_norm, rtol=1.e-12, atol=1.e-12)
    query = torch.tensor(direct.points, dtype=torch.float64, requires_grad=True)
    coefficients = direct.model.fields(query, direct.latent, equation_condition=[.03, .9])["hermite_coefficients"]
    derivative = torch.stack([torch.autograd.grad(coefficients[:, i].sum(), query, retain_graph=True)[0][:, 0]
                              for i in range(3)], dim=-1)
    np.testing.assert_allclose(angular, torch.linalg.vector_norm(derivative, dim=-1).detach().numpy(),
                               rtol=1.e-11, atol=1.e-12)
    unit = amplitude / expected_norm[:, None]
    physical_dt = np.broadcast_to(factor * np.array([0., .6/3., -.1/3.]), amplitude.shape)
    tangent = physical_dt - unit * np.sum(unit*physical_dt, axis=1)[:, None]
    np.testing.assert_allclose(angular, np.linalg.norm(tangent, axis=1)/expected_norm, rtol=1.e-12, atol=1.e-12)
    other = direct.model.fields(query, direct.latent, equation_condition=[.07, 1.15])["hermite_coefficients"]
    torch.testing.assert_close(coefficients, other, rtol=0., atol=0.)
    assert all(parameter.grad is None for parameter in direct.model.parameters())


def test_raw_replay_rejects_decoder_fallback_and_nonzero_residual(cli, direct):
    with pytest.raises(ValueError, match="fallback"):
        cli.raw_amplitude_jets(direct.model, direct.points, np.zeros_like(direct.latent))
    direct.model.config = replace(direct.model.config, structured_residual_scale=.1)
    with pytest.raises(ValueError, match="zero-residual"):
        cli.raw_amplitude_jets(direct.model, direct.points, direct.latent)


def arguments(tmp_path):
    return ["--cache", str(tmp_path / "cache"), "--pilot", str(tmp_path / "pilot"),
            "--output", str(tmp_path / "output"), "--regime", "weak", "--device", "cpu"]


def test_main_requires_slurm_before_any_run_or_output(cli, monkeypatch, tmp_path):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(cli, "run", lambda _: pytest.fail("guard must run before real diagnostics"))
    with pytest.raises(SystemExit) as error:
        cli.main(arguments(tmp_path))
    assert error.value.code == 2
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("target", ["existing", "broken_symlink", "inside_cache", "inside_pilot"])
def test_main_requires_fresh_output_outside_sources(cli, monkeypatch, tmp_path, target):
    monkeypatch.setenv("SLURM_JOB_ID", "tiny-guard-test")
    monkeypatch.setattr(cli, "run", lambda _: pytest.fail("fresh-output guard must prevent diagnostics"))
    args = arguments(tmp_path)
    output = tmp_path / "output"
    if target == "existing":
        output.mkdir()
    elif target == "broken_symlink":
        output.symlink_to(tmp_path / "missing-target")
    else:
        output = tmp_path / ("cache" if target == "inside_cache" else "pilot") / "new"
    args[args.index("--output") + 1] = str(output)
    with pytest.raises(SystemExit) as error:
        cli.main(args)
    assert error.value.code == 2


@pytest.mark.parametrize("flag,value", [("--answers", "0"), ("--answers", "33"), ("--seed", "-1")])
def test_main_bounds_real_diagnostic_size(cli, monkeypatch, tmp_path, flag, value):
    monkeypatch.setenv("SLURM_JOB_ID", "tiny-guard-test")
    monkeypatch.setattr(cli, "run", lambda _: pytest.fail("bounds must prevent diagnostics"))
    with pytest.raises(SystemExit) as error:
        cli.main(arguments(tmp_path) + [flag, value])
    assert error.value.code == 2
