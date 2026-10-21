"""Independent full-space quadrature for transport v2, using tiny CPU fixtures."""
import importlib
import math

import numpy as np
import pytest
import torch

from pde_agent_verifier_solver.plugins.vpfp.transport_moments import (
    affine_central_third, evaluate_hermite_transport_points,
    heat_flux_from_central_third, squared_hermite_transport_moments,
)
from pde_agent_verifier_solver.plugins.vpfp.moving_frame import MovingFrameDecoder, source_local_moments
from pde_agent_verifier_solver.plugins.vpfp.moving_frame_fast_macros import evaluate_moving_frame_transport_macros
from pde_agent_verifier_solver.plugins.vpfp.moving_frame_batched_source import MovingFrameSourceEvaluator
from tests.test_vpfp_moving_frame import tiny, constant_macro, points


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def quadrature(degree, theta):
    nodes, weights = np.polynomial.hermite.hermgauss(degree+4)
    mesh = np.stack(np.meshgrid(nodes, nodes, indexing="ij"), -1).reshape(-1, 2)
    velocity = mesh*math.sqrt(2*theta)
    measure = np.outer(weights, weights).ravel()*np.exp(np.square(mesh).sum(-1))*2*theta
    return torch.tensor(velocity, dtype=torch.float64), torch.tensor(measure, dtype=torch.float64)


@pytest.mark.parametrize("degree", [1, 4, 14])
@pytest.mark.parametrize("floor", [0., .27])
def test_all_moments_match_independent_fullspace_quadrature(degree, floor):
    base, _, _, _, _ = tiny(degree=degree, floor=floor)
    density = importlib.import_module("agent_verifier.vpfp.hermite_density")
    generator = torch.Generator().manual_seed(872)
    coeff = torch.randn(len(base.hermite_modes), generator=generator, dtype=torch.float64)
    theta = .73
    moment = squared_hermite_transport_moments(coeff, modes=base.hermite_modes,
        temperature=theta, floor_mixture=floor)
    velocity, measure = quadrature(degree, theta)
    g = density.torch_squared_hermite_density(torch, velocity, coeff,
        modes=base.hermite_modes, temperature=theta, floor_mixture=floor)
    weight = g*measure
    mean = (weight[:, None]*velocity).sum(0)
    second = torch.einsum("n,ni,nj->ij", weight, velocity, velocity)
    third = torch.einsum("n,ni,nj,nk->ijk", weight, velocity, velocity, velocity)
    centered = velocity-mean
    central = torch.einsum("n,ni,nj,nk->ijk", weight, centered, centered, centered)
    def pack(t):
        return torch.stack((t[0, 0, 0], t[0, 0, 1], t[0, 1, 1], t[1, 1, 1]))
    torch.testing.assert_close(weight.sum(), torch.tensor(1., dtype=torch.float64), rtol=3e-13, atol=3e-13)
    torch.testing.assert_close(moment["mean_velocity"], mean, rtol=3e-12, atol=3e-12)
    torch.testing.assert_close(moment["raw_second_moment"], torch.stack((second[0, 0], second[0, 1], second[1, 1])), rtol=3e-12, atol=3e-12)
    torch.testing.assert_close(moment["raw_third_moment"], pack(third), rtol=3e-12, atol=3e-12)
    torch.testing.assert_close(moment["central_third_moment"], pack(central), rtol=3e-12, atol=3e-12)


def test_top_mode_paths_are_not_truncated_and_input_float32_is_reduced_in_float64():
    # <x^3> for c=(H0+H1)/sqrt(2) is 3*theta^(3/2), not theta^(3/2).
    result = squared_hermite_transport_moments(torch.tensor([1., 1.]),
        modes=((0, 0), (1, 0)), temperature=.7)
    assert result["raw_third_moment"].dtype == torch.float64
    torch.testing.assert_close(result["raw_third_moment"][0],
        torch.tensor(3*.7**1.5, dtype=torch.float64), rtol=2e-15, atol=2e-15)


def test_anisotropic_affine_pushforward_needs_mixed_third_moments():
    packed = torch.tensor([0., 2., -1., 0.], dtype=torch.float64)
    affine = torch.tensor([[2., .3], [-.4, .7]], dtype=torch.float64)
    transformed = affine_central_third(packed, affine)
    expected = torch.zeros((2, 2, 2), dtype=torch.float64)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for a in range(2):
                    for b in range(2):
                        for c in range(2):
                            expected[i, j, k] += affine[i, a]*affine[j, b]*affine[k, c]*packed[a+b+c]
    torch.testing.assert_close(transformed, torch.stack((expected[0, 0, 0], expected[0, 0, 1], expected[0, 1, 1], expected[1, 1, 1])))
    rho = torch.tensor(1., dtype=torch.float64)
    assert not torch.allclose(heat_flux_from_central_third(rho, transformed),
                              affine@heat_flux_from_central_third(rho, packed))


@pytest.mark.parametrize("degree", [1, 4, 14])
def test_weak_transport_uses_complete_affine_floor_and_matches_density_integral(degree):
    base, value, cfg, z, condition = tiny(degree=degree, floor=.27)
    amplitude = z[:, value.rho_dimension:].reshape(1, value.hermite_channels, 3, 9)
    amplitude[:, -1, :, 0], amplitude[:, -2, :, 0] = .7, .3
    amplitude[:, 1, :, 0] = -.2
    decoder = MovingFrameDecoder(base, cfg, value_config=value)
    carrier = {"m": constant_macro(cfg), "z": z}
    p = points()[1:2]
    native = decoder.source_fields(p, z, condition)
    source = source_local_moments(native, base)
    target = decoder.macro_at(p, carrier["m"], native)
    velocity, measure = quadrature(degree, float(condition[1]))
    standardized = torch.linalg.solve_triangular(source["factor"][0],
        (velocity-source["mean"][0]).T, upper=False).T
    transformed = target["mean"]+standardized@target["factor"][0].T
    phase = p.expand(len(transformed), -1).clone()
    phase[:, 3:5] = transformed
    f = decoder.fields(phase, carrier, equation_condition=condition)["f"]
    jac = torch.linalg.det(target["factor"])/torch.linalg.det(source["factor"])
    centered = transformed-target["mean"]
    expected_heat = (.5*f[:, None]*measure[:, None]*jac*centered.square().sum(-1)[:, None]*centered).sum(0)
    expected_pressure = torch.einsum("n,ni,nj->ij", f*measure*jac, centered, centered)
    actual = evaluate_moving_frame_transport_macros(decoder, p[:, :3], carrier, condition)
    torch.testing.assert_close(actual["heat_flux"][0], expected_heat, rtol=4e-11, atol=4e-12)
    torch.testing.assert_close(actual["pressure"][0], torch.stack((expected_pressure[0, 0], expected_pressure[0, 1], expected_pressure[1, 1])), rtol=4e-11, atol=4e-12)


def test_drifting_gaussian_heat_zero_and_cached_transport_never_differentiates_third_moments():
    base, value, cfg, z, condition = tiny()
    decoder = MovingFrameDecoder(base, cfg, value_config=value)
    carrier = {"m": constant_macro(cfg), "z": z}
    evaluator = MovingFrameSourceEvaluator(decoder, carrier, condition)
    p = points()
    first = evaluator.transport_macros(p[:, :3])
    assert torch.count_nonzero(first["heat_flux"]) == 0
    assert first["momentum"].abs().max() > .1
    assert evaluator.cache_info()["geometry_jet_batches"] == 0
    before = evaluator.cache_info()["geometry_value_batches"]
    shuffled = evaluator.transport_macros(p[[3, 1, 1, 0], :3])
    assert evaluator.cache_info()["geometry_value_batches"] == before
    for key in first:
        torch.testing.assert_close(shuffled[key], first[key][[3, 1, 1, 0]])
    source = evaluator.source(p)
    fresh = MovingFrameSourceEvaluator(decoder, carrier, condition)
    expected = fresh.source(p)
    torch.testing.assert_close(source["source"], expected["source"], rtol=2e-12, atol=2e-12)
    assert evaluator.cache_info()["geometry_jet_batches"] == fresh.cache_info()["geometry_jet_batches"]
    assert evaluator.cache_info()["assembled_phase_points"] == len(p)
    assert "heat_flux" not in source


def test_strong_single_batch_transport_and_chunking_agree_with_legacy_moments():
    base, value, _, z, condition = tiny(degree=4)
    from pde_agent_verifier_solver.plugins.vpfp.value_control_diffusion import lift_value_controls
    from pde_agent_verifier_solver.plugins.vpfp.perturbation_backend import _legacy_api
    amplitude = z[:, value.rho_dimension:].reshape(1, value.hermite_channels, 3, 9)
    amplitude[:, -1, :, 0], amplitude[:, 1, :, 0] = .35, -.21
    latent = lift_value_controls(z, value)
    p = points()[:, :3].numpy()
    actual = evaluate_hermite_transport_points(base, latent, p, equation_condition=condition)
    single = evaluate_hermite_transport_points(base, latent, p, equation_condition=condition, chunk_size=1)
    old = _legacy_api().evaluate_continuous_hermite_macro_points(base, latent, p, equation_condition=condition)
    for name in actual:
        np.testing.assert_allclose(actual[name], single[name], rtol=3e-13, atol=3e-13)
    for name in old:
        np.testing.assert_allclose(actual[name], old[name], rtol=3e-13, atol=3e-13)


def test_truth_selects_v2_only_when_transport_fields_requested():
    from pde_agent_verifier_solver.plugins.vpfp.problem import VPFPContinuousMacroTruth
    from pde_agent_verifier_solver.sdk import AnswerQuery, QueryKind
    calls = []
    def evaluate(identifier, points):
        calls.append("v1")
        n = len(points)
        return {"rho": np.ones(n), "momentum": np.zeros((n, 2)), "kinetic_energy": np.ones(n)}
    def transport(identifier, points):
        result = evaluate(identifier, points)
        calls[-1] = "v2"
        n = len(points)
        return {**result, "electric": np.zeros((n, 2)), "pressure": np.ones((n, 3)), "heat_flux": np.zeros((n, 2))}
    truth = VPFPContinuousMacroTruth("a", (0., 1.), (0., 1.), (0., 1.), evaluate,
                                    evaluate_transport_points=transport)
    q = AnswerQuery(kind=QueryKind.POINTS, fields=("rho",), points=({"t": .2, "x": .3, "y": .4},))
    truth.answer(q)
    q = AnswerQuery(kind=QueryKind.POINTS, fields=("rho", "heat_flux_y", "pressure_xy"), points=({"t": .2, "x": .3, "y": .4},))
    answer = truth.answer(q)
    assert calls == ["v1", "v2"]
    assert answer.fields["pressure_xy"] == (1.,)


@pytest.mark.parametrize("accelerated", [False, True])
def test_resident_transport_truth_and_development_session_are_single_batch(tmp_path, accelerated):
    from tests.test_vpfp_moving_frame_backend import FixedSampler, request
    from pde_agent_verifier_solver.plugins.vpfp.moving_frame_backend import MovingFrameBackend
    from pde_agent_verifier_solver.sdk import AnswerQuery
    base, value, cfg, z, condition = tiny()
    decoder = MovingFrameDecoder(base, cfg, value_config=value)
    carrier = {"m": constant_macro(cfg), "z": z}
    runtime = MovingFrameBackend(decoder, FixedSampler(carrier, condition),
        output_root=tmp_path/"private", public_grid=3, public_times=3,
        field_chunk=2, adaptive_source_only=accelerated)
    record, = runtime.generate(None, None, request())
    session = runtime.development_solution_session_factory(record.case)
    result = session.released_transport_macros(points()[:, :3].numpy())
    assert result["heat_flux"].shape == (4, 2)
    assert session.usage_snapshot["completed_points"] == 4
    assert session.usage_snapshot["query_calls"] == 1
    before = runtime.oracle_calls.copy()
    query = AnswerQuery.at_points(
        fields=("rho", "electric_x", "electric_y", "momentum_x", "momentum_y",
                "pressure_xx", "pressure_xy", "pressure_yy", "heat_flux_x", "heat_flux_y"),
        points=({"t": 2., "x": .1, "y": .3},))
    answer = record.truth.answer(query)
    assert len(answer.fields) == 10
    assert runtime.oracle_calls["transport_macros"] == before["transport_macros"]+1
    assert runtime.oracle_calls["source"] == 0
    assert runtime.oracle_calls["macros"] == before["macros"]
    if accelerated:
        assert runtime._source_evaluators[record.case.case_id].cache_info()["geometry_jet_batches"] == 0
