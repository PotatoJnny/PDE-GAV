"""Explicit spline P uses exact knot refinement, separate from cosine spectral P."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "legacy/ns/src"))
from agent_verifier.neural_generator.continuous_geometry import ContinuousGeometrySpec
from agent_verifier.neural_generator.structured_ns_answer_first import (
    StructuredNSProjectionConfig, StructuredNSCaseCoefficients, ContinuousStructuredNSMMS,
)
from agent_verifier.neural_generator.structured_ns_robustness import (
    _midpoint_spline_refinement, _tau_squared_coefficients,
    coefficient_tensor_from_field, make_perturbation, field_at_amplitude,
    spline_perturbation_velocity_metrics,
)
from pde_agent_verifier_solver.plugins.ns.completed_fields import field_to_state, field_from_state


@pytest.fixture(scope="module", autouse=True)
def small_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _field(seed=71, *, temporal_basis="bspline"):
    config = StructuredNSProjectionConfig(
        spatial_basis="bspline", temporal_basis=temporal_basis,
        spatial_mode_x=6, spatial_mode_y=5, time_order=4,
        spatial_extension=0, ridge_relative=1e-6,
    )
    values = np.zeros((2, 5, 7, 6))
    values[0] = np.random.default_rng(seed).normal(size=values[0].shape) * .002
    geometry = ContinuousGeometrySpec((-1., 2., .2, 1.4), (), outer_boundary="manufactured_trace")
    return ContinuousStructuredNSMMS(
        StructuredNSCaseCoefficients(values[0].reshape(5, -1), values[1].reshape(5, -1)),
        config=config, geometry_spec=geometry, domain_bounds=geometry.domain_bounds,
        viscosity=.01, t0=.3, t1=.9,
    )


def _mixed_field(seed=71):
    field = _field(seed)
    config = replace(field.config, spatial_basis="cosine", spatial_knots_x=(),
                     spatial_knots_y=(), spatial_extension=.075)
    return ContinuousStructuredNSMMS(field.coefficients, config=config,
        geometry_spec=field.geometry_spec, domain_bounds=field.domain_bounds,
        viscosity=field.viscosity, t0=field.t0, t1=field.t1)


def _points(field, *, initial=False):
    return np.array([[-1., .7, field.t0 if initial else .4],
                     [2., .8, field.t0 if initial else .9],
                     [.3, .2, field.t0 if initial else .6],
                     [.5, 1.4, field.t0 if initial else .7],
                     [.2, .55, field.t0 if initial else .5]])


def test_exact_knot_insertion_and_coefficient_complement():
    from scipy.interpolate import BSpline
    field = _field()
    perturbation = make_perturbation(field, family="spline_refinement", seed=812)
    fine = field_at_amplitude(field, perturbation, amplitude=0.)
    coarse = coefficient_tensor_from_field(field).numpy()
    refined = coefficient_tensor_from_field(fine).numpy()
    matrices = []
    for axis in ("x", "y"):
        knots = getattr(field.config, f"spatial_knots_{axis}")
        new_knots, matrix = _midpoint_spline_refinement(knots, field.config.spatial_degree)
        positions = np.linspace(0., 1., 23)
        old_basis = BSpline(knots, np.eye(matrix.shape[1]), 5)(positions)
        new_basis = BSpline(new_knots, np.eye(matrix.shape[0]), 5)(positions)
        np.testing.assert_allclose(new_basis @ matrix, old_basis, rtol=1e-12, atol=1e-13)
        matrices.append(matrix)
    rx, ry = matrices
    np.testing.assert_allclose(refined, np.einsum("ik,ctkl,jl->ctij", rx, coarse, ry), rtol=0, atol=1e-17)
    direction = perturbation.direction_coefficients.numpy()
    assert np.count_nonzero(direction[1]) == 0
    assert np.linalg.norm(direction) == pytest.approx(1.)
    for temporal_slice in direction[0]:
        np.testing.assert_allclose(rx.T @ temporal_slice @ ry, 0., atol=1e-14)
    for key, value in field.quantities(_points(field)).items():
        torch.testing.assert_close(fine.quantities(_points(field))[key], value, rtol=2e-10, atol=1e-10)
    replay = make_perturbation(field, family="spline_refinement", seed=812)
    torch.testing.assert_close(perturbation.direction_coefficients, replay.direction_coefficients, rtol=0, atol=0)
    assert perturbation.metadata["amplitude_equivalent_to_cosine"] is False


@pytest.mark.parametrize("temporal_basis", ["bspline", "chebyshev"])
def test_tau_squared_reproduction_and_initial_velocity_derivative(temporal_basis):
    from scipy.interpolate import BSpline
    field = _field(temporal_basis=temporal_basis)
    coefficients = _tau_squared_coefficients(field.config)
    tau = np.array([0., .00001, .125, .5, .73, 1.])
    if temporal_basis == "bspline":
        spline = BSpline(field.config.temporal_knots, coefficients, 3)
        np.testing.assert_allclose(spline(tau), tau**2, atol=2e-16)
        np.testing.assert_allclose(spline(tau, nu=1), 2*tau, atol=2e-15)
    else:
        np.testing.assert_allclose(np.polynomial.chebyshev.chebval(2*tau-1, coefficients), tau**2, atol=2e-16)
    perturbation = make_perturbation(field, family="spline_refinement", seed=81)
    child = field_at_amplitude(field, perturbation, amplitude=.001)
    original, changed = field.quantities(_points(field, initial=True)), child.quantities(_points(field, initial=True))
    for key in ("u", "v", "d_t_u", "d_t_v"):
        torch.testing.assert_close(changed[key], original[key], rtol=2e-10, atol=1e-11)
    assert torch.max(torch.abs(changed["divergence"])) < 1e-12
    middle_original, middle_child = field.quantities(_points(field)), child.quantities(_points(field))
    assert torch.max(torch.abs(middle_child["u"] - middle_original["u"])) > 1e-8
    assert torch.max(torch.abs(middle_child["forcing_u"] - middle_original["forcing_u"])) > 1e-8
    from agent_verifier.neural_generator.structured_ns_analytic import analytic_quantities
    analytic = analytic_quantities(child, _points(field))
    for key in ("u", "v", "d_t_u", "d_t_v", "forcing_u", "forcing_v"):
        torch.testing.assert_close(analytic[key], middle_child[key], rtol=1e-10, atol=1e-10)


def test_offline_velocity_statistics_and_refined_state_restore():
    field = _field()
    perturbation = make_perturbation(field, family="spline_refinement", seed=712)
    diagnostics = spline_perturbation_velocity_metrics(field, perturbation, amplitude=.001, spatial_points=7, time_points=3)
    doubled = spline_perturbation_velocity_metrics(field, perturbation, amplitude=.002, spatial_points=7, time_points=3)
    assert diagnostics["delta_velocity_rms"] > 0
    assert doubled["delta_velocity_rms"] == pytest.approx(2*diagnostics["delta_velocity_rms"], rel=1e-10)
    assert diagnostics["initial_velocity_delta_max_abs"] < 1e-11
    child = field_at_amplitude(field, perturbation, amplitude=.001)
    restored = field_from_state(field_to_state(child))
    assert field_to_state(restored) == field_to_state(child)
    for key, value in child.quantities(_points(field)).items():
        torch.testing.assert_close(restored.quantities(_points(field))[key], value, rtol=0, atol=0)


def test_wrong_family_never_silently_reinterprets_cosine_perturbation():
    field = _field()
    with pytest.raises(ValueError, match="cosine space"):
        make_perturbation(field, family="spectral_complement", seed=1)
    cosine = replace(field.config, spatial_basis="cosine", temporal_basis="chebyshev",
                     spatial_knots_x=(), spatial_knots_y=(), temporal_knots=())
    old = ContinuousStructuredNSMMS(field.coefficients, config=cosine,
        geometry_spec=field.geometry_spec, domain_bounds=field.domain_bounds,
        viscosity=field.viscosity, t0=field.t0, t1=field.t1)
    with pytest.raises(ValueError, match="spatial B-spline"):
        make_perturbation(old, family="spline_refinement", seed=1)


def _prepared(field_factory=_field):
    from pde_agent_verifier_solver.plugins.ns.experimental_mms import prepare_experimental_mms_answers
    from pde_agent_verifier_solver.sdk import DatasetInput, DatasetKind
    split = {"generator_training": tuple(field_factory(i) for i in range(8)),
             "generator_validation": (field_factory(8), field_factory(9)), "development": (field_factory(10),),
             "verifier": (field_factory(11),), "final_test": (field_factory(12),)}
    return prepare_experimental_mms_answers(DatasetInput("spline-fixture", DatasetKind.EXPERIMENTAL), split)


def test_actual_pca_flow_refinement_and_case_restore(tmp_path):
    # Reuse a tiny frozen numerical-format model fixture; no RF optimization.
    from tests.test_ns_experimental_mms_backend import _artifacts, _request, raw_answer
    from pde_agent_verifier_solver.plugins.ns.experimental_mms import assemble_experimental_mms
    from pde_agent_verifier_solver.sdk import GenerationKind
    dataset = _prepared()
    flow, pca = _artifacts(tmp_path / "artifacts", dataset)
    settings = {"device": "cpu", "integration_steps": 2, "flow_state": "ema",
                "integration_method": "heun", "perturbation_amplitude": 1e-4,
                "perturbation_family": "spline_refinement"}
    def assembly():
        return assemble_experimental_mms(dataset, flow_checkpoint=flow, pca_checkpoint=pca,
                                         generator_settings=settings)
    first = assembly()
    state = first.generator.load(first.checkpoint, {})
    parents = first.generator.sample(state, _request())
    request = _request(GenerationKind.PERTURBED, seed=89, anchors=tuple(row.case for row in parents))
    children = first.generator.sample(state, request)
    for parent, child in zip(parents, children):
        metadata = child.case.public_data["perturbation"]
        assert metadata["family"] == "spline_refinement"
        assert metadata["coefficient_support_complement"] is True
        assert metadata["spectral_support_complement"] is False
        assert metadata["amplitude_equivalent_to_cosine"] is False
        assert "refinement_details" not in metadata and "seed" not in metadata
        assert child.case.parent_case_id == parent.case.case_id
        assert raw_answer(child).config.spatial_basis == "bspline"
    objects = tmp_path / "objects"
    runtime = first.generator.export_runtime_state(objects)
    saved_state = first.generator.export_generator_state(state, objects)
    saved_cases = [first.generator.export_case_record(row, objects) for row in (*parents, *children)]
    second = assembly()
    second.generator.restore_runtime_state(runtime, objects)
    restored_state = second.generator.restore_generator_state(saved_state, objects)
    for original, saved in zip((*parents, *children), saved_cases):
        restored = second.generator.restore_case_record(saved, objects)
        assert restored.case == original.case
        assert field_to_state(raw_answer(restored)) == field_to_state(raw_answer(original))
    next_request = _request(GenerationKind.PERTURBED, seed=90, anchors=tuple(row.case for row in parents))
    expected = first.generator.sample(state, next_request)
    actual = second.generator.sample(restored_state, next_request)
    for left, right in zip(expected, actual):
        assert left.case == right.case
        np.testing.assert_array_equal(raw_answer(left).coefficients.psi, raw_answer(right).coefficients.psi)
    settings["perturbation_family"] = "spectral_complement"
    with pytest.raises(ValueError, match="explicitly choose"):
        assembly()


def test_spline_scores_use_the_original_neural_guidance(tmp_path):
    from types import SimpleNamespace
    from tests.test_ns_experimental_mms_backend import _artifacts, _request, _guidance, raw_answer
    from pde_agent_verifier_solver.plugins.ns.experimental_mms import assemble_experimental_mms
    from pde_agent_verifier_solver.plugins.ns.structured_guidance_backend import StructuredNSMacroGuidanceBackend
    from pde_agent_verifier_solver.sdk import Artifact, CaseFeedback, GenerationKind, RoundFeedback, TaskKind
    dataset = _prepared()
    flow, pca = _artifacts(tmp_path / "artifacts", dataset)
    config, protocol = _guidance()
    assembly = assemble_experimental_mms(dataset, flow_checkpoint=flow, pca_checkpoint=pca,
        generator_settings={"device": "cpu", "integration_steps": 2, "flow_state": "ema",
                            "integration_method": "heun", "perturbation_family": "spline_refinement",
                            "perturbation_amplitude": 1e-4},
        guidance_artifact_root=tmp_path / "guidance", guidance_config=config, guidance_protocol=protocol)
    state = assembly.generator.load(assembly.checkpoint, {})
    parents = assembly.generator.sample(state, _request())
    solver = Artifact("synthetic-feedback", TaskKind.SOLVER, "Interface test", SimpleNamespace(), "", "", 0, "local")
    feedback = RoundFeedback(0, solver, tuple(CaseFeedback(row.case, .1 + index*.2, {})
                                             for index, row in enumerate(parents)))
    guidance = assembly.generator.backend.fit_guidance(state.base, None, feedback, {"controller_seed": 191})
    assert type(assembly.bundle.guidance_backend) is StructuredNSMacroGuidanceBackend
    guided = assembly.generator.sample(replace(state, guidance=guidance), _request(GenerationKind.GUIDED, index=1))
    for record in guided:
        assert raw_answer(record).config.spatial_basis == "bspline"
        assert raw_answer(record).config.temporal_basis == "bspline"
        assert record.case.public_data["distribution_lineage"]["kind"] == "frozen_reward_guided"


def test_harness_requires_explicit_spline_epsilon_before_file_io(monkeypatch, tmp_path):
    import importlib
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    harness = importlib.import_module("check_ns_measured_generator_integration")
    original = dict(harness.RUNTIME)
    monkeypatch.setattr(sys, "argv", ["check", "--prepared", str(tmp_path / "missing"),
                        "--out", str(tmp_path / "output"), "--perturbation-family", "spline_refinement"])
    with pytest.raises(SystemExit) as error:
        harness.main()
    assert error.value.code == 2
    assert harness.RUNTIME == original
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("spatial_basis", ["bspline", "cosine"])
def test_full_harness_spline_fixture_base_guidance_perturbation_panel(monkeypatch, tmp_path, spatial_basis):
    import importlib
    import json
    import shutil
    from tests.test_ns_experimental_mms_backend import _artifacts
    from pde_agent_verifier_solver.plugins.ns.experimental_mms_project import save_prepared_mms_dataset
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    harness = importlib.import_module("check_ns_measured_generator_integration")
    dataset = _prepared(_field if spatial_basis == "bspline" else _mixed_field)
    family = "spline_refinement" if spatial_basis == "bspline" else "spectral_complement"
    prepared = tmp_path / "prepared"
    flow, pca = _artifacts(prepared / "artifacts", dataset)
    (prepared / "base_generator").mkdir()
    (prepared / "base_generator/train_report.json").write_text(json.dumps({"final_checkpoint": str(flow)}))
    shutil.copyfile(pca, prepared / "pca_checkpoint.pt")
    save_prepared_mms_dataset(dataset, prepared / "prepared_dataset.json")
    out = tmp_path / "integration"
    original_runtime = dict(harness.RUNTIME)
    monkeypatch.setattr(sys, "argv", ["check", "--prepared", str(prepared), "--out", str(out),
                        "--perturbation-family", family, "--perturbation-amplitude", "0.0001",
                        "--spectral-max-mode", "28"])
    harness.main()
    report = json.loads((out / "integration_report.json").read_text())
    assert report["success"] and report["panel"]["success"] and report["resume"]["success"]
    assert report["runtime_settings"].get("perturbation_family", "spectral_complement") == family
    assert report["runtime_settings"]["spectral_max_mode"] == 28
    assert report["runtime_settings"]["perturbation_amplitude"] == 1e-4
    assert report["runtime_settings"]["integration_steps"] == 256
    assert report["panel"]["scientific_result"] is False
    assert harness.RUNTIME == original_runtime


def test_cosine_shell_with_spline_time_preserves_initial_derivative_and_low_band():
    from agent_verifier.neural_generator.structured_ns_robustness import structured_perturbation_velocity_metrics
    field = _mixed_field()
    p = make_perturbation(field, family="spectral_complement", seed=341, spectral_max_mode=8)
    assert p.metadata["temporal_basis"] == "bspline"
    assert p.metadata["amplitude_equivalent_to_cosine"] is False
    assert p.metadata["low_band_coefficient_leakage_max_abs"] == 0
    child = field_at_amplitude(field, p, amplitude=.0001)
    np.testing.assert_array_equal(coefficient_tensor_from_field(child)[:, :, :7, :6], coefficient_tensor_from_field(field))
    before, after = field.quantities(_points(field, initial=True)), child.quantities(_points(field, initial=True))
    for key in ("u", "v", "d_t_u", "d_t_v"):
        torch.testing.assert_close(before[key], after[key], rtol=1e-12, atol=1e-12)
    metrics = structured_perturbation_velocity_metrics(field, p, amplitude=.0001, spatial_points=7, time_points=3)
    assert metrics["delta_velocity_rms"] > 0
    assert metrics["initial_velocity_delta_max_abs"] < 1e-12
    with pytest.raises(ValueError, match="must exceed"):
        make_perturbation(field, family="spectral_complement", seed=341, spectral_max_mode=6)


def test_harness_changed_spectral_limit_requires_new_explicit_epsilon(monkeypatch, tmp_path):
    import importlib
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    harness = importlib.import_module("check_ns_measured_generator_integration")
    monkeypatch.setattr(sys, "argv", ["check", "--prepared", str(tmp_path / "missing"),
                        "--out", str(tmp_path / "output"), "--spectral-max-mode", "28"])
    with pytest.raises(SystemExit) as error:
        harness.main()
    assert error.value.code == 2
    assert not (tmp_path / "output").exists()


def test_offline_diagnostic_batches_do_not_change_physical_statistics():
    from agent_verifier.neural_generator.structured_ns_robustness import structured_perturbation_velocity_metrics
    field = _mixed_field()
    def metrics(f):
        p = make_perturbation(f, family="spectral_complement", seed=43, spectral_max_mode=8)
        return structured_perturbation_velocity_metrics(f, p, amplitude=.0001, spatial_points=7, time_points=3)
    original = metrics(field)
    chunked = ContinuousStructuredNSMMS(field.coefficients,
        config=replace(field.config, evaluation_batch_points=32), geometry_spec=field.geometry_spec,
        domain_bounds=field.domain_bounds, viscosity=field.viscosity, t0=field.t0, t1=field.t1)
    actual = metrics(chunked)
    for key in ("base_velocity_rms", "delta_velocity_rms", "delta_velocity_relative_rms"):
        assert actual[key] == pytest.approx(original[key], rel=1e-12, abs=1e-14)
