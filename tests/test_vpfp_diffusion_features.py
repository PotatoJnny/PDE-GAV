"""Physical feature contract checks, not trained-asset or high-fidelity evidence."""

from dataclasses import replace
from hashlib import sha256
import importlib

import numpy as np
import pytest

from pde_agent_verifier_solver.plugins.vpfp.diffusion_features import (
    VPFPFeaturePlan, extract_vpfp_features, summarize_continuous_fields,
)


def _summary_inputs(phase=0.):
    plan = VPFPFeaturePlan(grid_size=17, time_fractions=(0., .5, 1.), phase_probe_count=4)
    x = np.arange(plan.grid_size)[:, None] * 2. * np.pi / plan.grid_size
    shape = (len(plan.time_fractions), plan.grid_size, plan.grid_size)
    rho = np.broadcast_to(1. + .2 * np.cos(x) + .1 * np.cos(2. * x + phase), shape).copy()
    fields = {"rho": rho, "momentum_x": np.zeros(shape), "momentum_y": np.zeros(shape),
              "kinetic_energy": rho.copy(), "electric_x": np.zeros(shape), "electric_y": np.zeros(shape)}
    amplitude = np.ones((*shape, 1))
    return fields, amplitude, ((0, 0),), plan


def _feature_map(row, descriptors):
    return {entry["name"]: value for value, entry in zip(row, descriptors, strict=True)}


def test_bispectrum_detects_relative_phase_beyond_identical_power_spectra():
    inputs = _summary_inputs()
    first, description = summarize_continuous_fields(*inputs)
    second, second_description = summarize_continuous_fields(*_summary_inputs(np.pi / 2.))
    assert description == second_description
    left, right = _feature_map(first, description), _feature_map(second, description)
    for key in left:
        if key.startswith("rho.") and ".spectrum_" in key:
            assert left[key] == pytest.approx(right[key], abs=1.e-15)
    key = "rho.time0.bispectrum_x112_real"
    imag = "rho.time0.bispectrum_x112_imag"
    assert left[key] == pytest.approx(.1 * .1 * .05)
    assert abs(right[key]) < 1.e-15
    assert right[imag] == pytest.approx(-.1 * .1 * .05)
    assert not np.allclose(first, second)


def test_bispectrum_respects_common_physical_translation():
    fields, amplitude, modes, plan = _summary_inputs(.67)
    first, description = summarize_continuous_fields(fields, amplitude, modes, plan)
    translated = {key: np.roll(value, shift=(3, 5), axis=(1, 2)) for key, value in fields.items()}
    second, second_description = summarize_continuous_fields(translated, amplitude, modes, plan)
    assert description == second_description
    np.testing.assert_allclose(first, second, atol=1.e-14, rtol=1.e-12)


@pytest.mark.parametrize("plan", [
    VPFPFeaturePlan(grid_size=12), VPFPFeaturePlan(grid_size=17.5),
    VPFPFeaturePlan(grid_size=True), VPFPFeaturePlan(chunk_size=0),
    VPFPFeaturePlan(probe_seed=-1), VPFPFeaturePlan(phase_probe_count=3),
    VPFPFeaturePlan(phase_probe_count=4.5), VPFPFeaturePlan(time_fractions=(1., 0.)),
    VPFPFeaturePlan(time_fractions=(0., .5, .5, 1.)),
    VPFPFeaturePlan(time_fractions=(0., np.nan, 1.)),
    VPFPFeaturePlan(time_fractions=(0., 1.01)), VPFPFeaturePlan(time_fractions=()),
])
def test_invalid_or_aliased_plan_rejected(plan):
    with pytest.raises(ValueError):
        plan.validate()


def test_time_order_is_rejected_not_silently_sorted():
    fields, amplitude, modes, plan = _summary_inputs()
    with pytest.raises(ValueError, match="strictly increasing"):
        summarize_continuous_fields(fields, amplitude, modes,
                                    replace(plan, time_fractions=(1., .5, 0.)))


@pytest.mark.parametrize("bad", ["density_zero", "density_negative", "energy_negative", "nan", "complex", "shape", "missing", "amplitude", "modes"])
def test_summary_rejects_invalid_physical_fields(bad):
    fields, amplitude, modes, plan = _summary_inputs()
    if bad == "density_zero":
        fields["rho"][0, 0, 0] = 0.
    elif bad == "density_negative":
        fields["rho"][0, 0, 0] = -.1
    elif bad == "energy_negative":
        fields["kinetic_energy"][0, 0, 0] = -.1
    elif bad == "nan":
        fields["electric_x"][0, 0, 0] = np.nan
    elif bad == "complex":
        fields["rho"] = fields["rho"].astype(np.complex128) + .1j
    elif bad == "shape":
        fields["rho"] = fields["rho"][0]
    elif bad == "missing":
        del fields["electric_y"]
    elif bad == "amplitude":
        amplitude[0, 0, 0, 0] = np.inf
    elif bad == "modes":
        modes = ((-1, 0),)
    with pytest.raises(ValueError):
        summarize_continuous_fields(fields, amplitude, modes, plan)


@pytest.fixture
def tiny_decoder():
    """Actual pinned decoder/source on a tiny synthetic coefficient fixture.

    There is no checkpoint load, PDE solve, fitted model or production asset
    validation in this fixture. One trajectory has only 142 coefficients.
    """
    from pde_agent_verifier_solver.plugins.vpfp.perturbation_backend import _legacy_api
    api = _legacy_api()
    hermite = importlib.import_module("agent_verifier.vpfp.continuous_hermite")
    config = replace(hermite.VPFPContinuousHermiteConfig(), latent_dim=1,
        micro_latent_dim=2, context_dim=4, hidden_width=4, hidden_layers=1,
        rho_max_mode_x=1, rho_max_mode_y=1, rho_positivity_gauge="phase_invariant_group_l1_v2",
        spatial_feature_modes=1, time_feature_modes=1, hermite_total_degree=1,
        coefficient_width=4, coefficient_blocks=1, structured_spectral_latent=True,
        structured_kinetic_global_dim=2, structured_local_channels=3,
        structured_local_spatial_mode=1, structured_time_basis="piecewise_cubic_hermite_v1",
        structured_time_control_count=2, structured_direct_coefficients=True,
        structured_residual_scale=0., time_domain=(0., 1.), dtype="float64")
    config = replace(config, latent_dim=config.structured_latent_layout()["total_latent_dim"])
    config.validate()
    model = api.build_deterministic_direct_model(config, device="cpu")
    plan = VPFPFeaturePlan(grid_size=3, time_fractions=(0., .5, 1.), phase_probe_count=4, chunk_size=11)
    latents = np.zeros((2, config.latent_dim), dtype=np.float32)
    latents[1] = np.random.default_rng(1729).normal(scale=.005, size=config.latent_dim)
    conditions = np.array([[.04, 1.], [.06, 1.1]], dtype=np.float32)
    return model, latents, conditions, plan


def test_actual_decoder_macros_and_source_match_exact_interfaces(tiny_decoder):
    model, latents, conditions, plan = tiny_decoder
    progress = []
    matrix, descriptors, report = extract_vpfp_features(model, latents, conditions,
        plan=plan, progress=lambda completed, total: progress.append((completed, total)))
    assert matrix.shape == (2, len(descriptors))
    assert np.isfinite(matrix).all()
    assert progress == [(1, 2), (2, 2)]
    assert report["sampled_density_positivity_passed"] is True
    assert report["sampled_f_nonnegativity_passed"] is True
    assert report["high_fidelity_validation"] is False
    assert report["spatial_quadrature_convergence_qualified"] is False
    analytic = importlib.import_module("agent_verifier.vpfp.analytic_source")
    macros = importlib.import_module("agent_verifier.vpfp.continuous_macros")
    points = np.asarray(report["phase_points"])
    names = [entry["name"] for entry in descriptors]
    for index in range(2):
        source = analytic.manufacture_standard_vpfp_source_analytic(
            model, points, latents[index], equation_condition=conditions[index])
        direct = model.fields(points, latents[index], equation_condition=conditions[index])
        np.testing.assert_allclose(source["f"].detach().numpy(), direct["f"].detach().numpy(),
                                   atol=1.e-13, rtol=1.e-12)
        # These are the actual keys, not a fixture's renamed schema.
        for key, prefix in (("source", "induced_source"), ("f", "f")):
            columns = [names.index(f"{prefix}.probe{i}") for i in range(plan.phase_probe_count)]
            np.testing.assert_allclose(matrix[index, columns], source[key].detach().numpy(), atol=1.e-13)
    grid = np.stack(np.meshgrid(np.array(plan.time_fractions),
        np.arange(3) * model.config.equation.x_domain[0] / 3,
        np.arange(3) * model.config.equation.x_domain[1] / 3, indexing="ij"), axis=-1).reshape(-1, 3)
    exact = macros.evaluate_continuous_hermite_macro_points(
        model, latents[1], grid, equation_condition=conditions[1], chunk_size=7)
    actual = _feature_map(matrix[1], descriptors)
    for name, key, axis in (("rho", "rho", None), ("momentum_x", "momentum", 0),
                            ("momentum_y", "momentum", 1), ("kinetic_energy", "kinetic_energy", None)):
        values = exact[key] if axis is None else exact[key][:, axis]
        values = values.reshape(3, 3, 3)
        for time_index, frame in enumerate(values):
            assert actual[f"{name}.time{time_index}.mean"] == pytest.approx(frame.mean(), abs=1.e-13)
    ground = _feature_map(matrix[0], descriptors)
    assert ground["rho.time0.mean"] == pytest.approx(1.)
    assert ground["kinetic_energy.time0.mean"] == pytest.approx(1.)
    assert ground["momentum_x.time0.rms"] == pytest.approx(0.)
    assert report["per_case"][0]["source_maximum_absolute"] < 1.e-14


@pytest.mark.parametrize("bad", ["nan_latent", "nan_condition", "negative_nu", "zero_temperature", "out_of_regime", "width", "empty"])
def test_extraction_rejects_bad_input(tiny_decoder, bad):
    model, latents, conditions, plan = tiny_decoder
    if bad == "nan_latent":
        latents[0, 0] = np.nan
    elif bad == "nan_condition":
        conditions[0, 0] = np.nan
    elif bad == "negative_nu":
        conditions[0, 0] = -.1
    elif bad == "zero_temperature":
        conditions[0, 1] = 0.
    elif bad == "out_of_regime":
        conditions[0, 0] = .09
    elif bad == "width":
        latents = latents[:, :-1]
    elif bad == "empty":
        latents, conditions = latents[:0], conditions[:0]
    with pytest.raises(ValueError):
        extract_vpfp_features(model, latents, conditions, plan=plan)


def test_actual_decoder_grid_alias_is_rejected_before_eval(tiny_decoder):
    model, latents, conditions, plan = tiny_decoder
    with pytest.raises(ValueError, match="aliases"):
        extract_vpfp_features(model, latents, conditions, plan=replace(plan, grid_size=2))


@pytest.mark.parametrize("bad", ["negative_f", "nan_source", "source_shape"])
def test_analytic_source_positivity_and_shape_hard_gates(tiny_decoder, monkeypatch, bad):
    model, latents, conditions, plan = tiny_decoder
    analytic = importlib.import_module("agent_verifier.vpfp.analytic_source")
    original = analytic.manufacture_standard_vpfp_source_analytic
    def invalid(*args, **kwargs):
        result = original(*args, **kwargs)
        if bad == "negative_f":
            result["f"] = result["f"].clone()
            result["f"][0] = -.1
        elif bad == "nan_source":
            result["source"] = result["source"].clone()
            result["source"][0] = float("nan")
        elif bad == "source_shape":
            result["source"] = result["source"][:-1]
        return result
    monkeypatch.setattr(analytic, "manufacture_standard_vpfp_source_analytic", invalid)
    with pytest.raises(ValueError):
        extract_vpfp_features(model, latents, conditions, plan=plan)


def test_float32_regime_endpoint_encoding_is_valid(tiny_decoder):
    model, latents, conditions, plan = tiny_decoder
    conditions = np.array([[.02, .8]], dtype=np.float32)
    matrix, _, _ = extract_vpfp_features(model, latents[:1], conditions, plan=plan)
    assert np.isfinite(matrix).all()


def test_integer_condition_cannot_truncate_regime_bounds(tiny_decoder):
    model, latents, _, plan = tiny_decoder
    with pytest.raises(ValueError, match="outside"):
        extract_vpfp_features(model, latents[:1], np.array([[0, 1]], dtype=np.int64), plan=plan)


def test_live_input_mutation_cannot_change_current_snapshot_features_or_hash(tiny_decoder, monkeypatch):
    model, latents, conditions, plan = tiny_decoder
    live_latents, live_conditions = latents[1:].copy(), conditions[1:].copy()
    expected, descriptors, expected_report = extract_vpfp_features(
        model, live_latents, live_conditions, plan=plan)
    expected_hash = sha256(live_latents[0].tobytes() + live_conditions[0].tobytes()).hexdigest()
    original_fields = model.fields
    observed_snapshots = []
    def fields(points, frozen_latent, *, equation_condition):
        assert not frozen_latent.flags.writeable
        assert not equation_condition.flags.writeable
        observed_snapshots.append((id(frozen_latent), id(equation_condition)))
        # Simulate the mmap owner changing its buffer during chunked feature
        # evaluation. The extractor may never reread it for the current case.
        live_latents[0] += .125
        live_conditions[0, 0] = .03
        return original_fields(points, frozen_latent, equation_condition=equation_condition)
    monkeypatch.setattr(model, "fields", fields)
    actual, actual_descriptors, report = extract_vpfp_features(model, live_latents, live_conditions, plan=plan)
    np.testing.assert_array_equal(actual, expected)
    assert actual_descriptors == descriptors
    assert len(observed_snapshots) > 1
    assert len(set(observed_snapshots)) == 1
    assert report["per_case"][0]["input_answer_sha256"] == expected_hash
    assert report["per_case"][0]["input_equation_condition"] == expected_report["per_case"][0]["input_equation_condition"]
    assert expected_hash != sha256(live_latents[0].tobytes() + live_conditions[0].tobytes()).hexdigest()


def test_snapshot_hash_and_computation_share_canonical_float32_precision(tiny_decoder):
    model, latents, conditions, plan = tiny_decoder
    # A perturbation below float32 precision must not affect only the physical
    # computation while disappearing from the stored content identity.
    high_precision = latents[1:].astype(np.float64)
    high_precision += np.spacing(latents[1:]).astype(np.float64) * .125
    high_conditions = conditions[1:].astype(np.float64)
    canonical_latents = high_precision.astype("<f4")
    canonical_conditions = high_conditions.astype("<f4")
    first, _, report = extract_vpfp_features(model, high_precision, high_conditions, plan=plan)
    second, _, _ = extract_vpfp_features(model, canonical_latents, canonical_conditions, plan=plan)
    np.testing.assert_array_equal(first, second)
    assert report["per_case"][0]["input_answer_sha256"] == sha256(
        canonical_latents[0].tobytes() + canonical_conditions[0].tobytes()).hexdigest()
    assert "canonical_float32" in report["input_answer_snapshot_precision"]


def test_float32_conversion_overflow_is_rejected_before_fields(tiny_decoder, monkeypatch):
    model, latents, conditions, plan = tiny_decoder
    values = latents[:1].astype(np.float64)
    values[0, 0] = 1.e100
    monkeypatch.setattr(model, "fields", lambda *args, **kwargs: pytest.fail("invalid snapshot reached fields"))
    with pytest.raises(ValueError, match="nonfinite"):
        extract_vpfp_features(model, values, conditions[:1], plan=plan)
