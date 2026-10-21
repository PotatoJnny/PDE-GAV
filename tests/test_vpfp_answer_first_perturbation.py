from __future__ import annotations

import hashlib
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_LEGACY_SRC = Path(__file__).resolve().parents[1] / "legacy" / "vpfp" / "src"
sys.path.insert(0, str(_LEGACY_SRC))
import agent_verifier as _agent_verifier

# The full suite may already have imported the modern agent_verifier package.
# Extending its search path keeps this legacy VPFP test independent of
# collection order.
_LEGACY_NAMESPACE = str(_LEGACY_SRC / "agent_verifier")
if _LEGACY_NAMESPACE not in _agent_verifier.__path__:
    _agent_verifier.__path__.append(_LEGACY_NAMESPACE)

from agent_verifier.vpfp import analytic_source, robust_perturbation
from agent_verifier.vpfp.continuous_cases import (
    materialize_continuous_case,
    verify_continuous_case_dir,
)
from agent_verifier.vpfp.continuous_flow_training import (
    VPFPFlowObservationEncoder,
    VPFPFlowObservationEncoderConfig,
    save_hermite_autoencoder_checkpoint,
)
from agent_verifier.vpfp.continuous_hermite import VPFPContinuousHermiteConfig
from agent_verifier.vpfp.continuous_macros import MACRO_OUTPUT_CONTRACT
from agent_verifier.vpfp.hermite_density import total_degree_mode_pairs
from agent_verifier.vpfp.operators import VPFPGrid, VPFPTime


def _direct_config() -> VPFPContinuousHermiteConfig:
    hermite_degree = 1
    provisional = replace(
        VPFPContinuousHermiteConfig(),
        latent_dim=1,
        micro_latent_dim=2,
        context_dim=8,
        hidden_width=8,
        hidden_layers=1,
        rho_max_mode_x=1,
        rho_max_mode_y=1,
        rho_positivity_gauge="phase_invariant_group_l1_v2",
        spatial_feature_modes=1,
        time_feature_modes=1,
        hermite_total_degree=hermite_degree,
        coefficient_width=8,
        coefficient_blocks=1,
        structured_spectral_latent=True,
        structured_kinetic_global_dim=2,
        structured_local_channels=len(total_degree_mode_pairs(hermite_degree)),
        structured_local_spatial_mode=1,
        structured_time_basis="piecewise_cubic_hermite_v1",
        structured_time_control_count=5,
        structured_direct_coefficients=True,
        structured_residual_scale=0.0,
        dtype="float64",
    )
    config = replace(
        provisional,
        latent_dim=int(provisional.structured_latent_layout()["total_latent_dim"]),
    )
    config.validate()
    return config


def _perturbation_config() -> robust_perturbation.VPFPRobustPerturbationConfig:
    return robust_perturbation.VPFPRobustPerturbationConfig(
        base_hermite_total_degree=1,
        expanded_rho_max_mode_x=2,
        expanded_rho_max_mode_y=2,
        expanded_local_spatial_mode=2,
        expanded_hermite_total_degree=2,
        density_rank=3,
        kinetic_rank=4,
        maximum_temporal_frequency=3,
    )


def _sample(seed: int):
    base = _direct_config()
    policy = _perturbation_config()
    expanded = robust_perturbation.expanded_vpfp_perturbation_config(base, policy)
    latent = robust_perturbation.sample_vpfp_private_perturbation_latent(
        base,
        expanded,
        seed=seed,
        perturbation=policy,
        dtype=torch.float64,
        device="cpu",
    )
    return base, expanded, policy, latent


def _hermite_value_and_derivative(
    values: np.ndarray, slopes: np.ndarray, tau: float
) -> tuple[np.ndarray, np.ndarray]:
    count = int(values.shape[0])
    unit = float(tau) * float(count - 1)
    lower = min(int(np.floor(unit)), count - 2)
    local = unit - float(lower)
    local2 = local * local
    local3 = local2 * local
    h00 = 2.0 * local3 - 3.0 * local2 + 1.0
    h10 = local3 - 2.0 * local2 + local
    h01 = -2.0 * local3 + 3.0 * local2
    h11 = local3 - local2
    dh00 = 6.0 * local2 - 6.0 * local
    dh10 = 3.0 * local2 - 4.0 * local + 1.0
    dh01 = -6.0 * local2 + 6.0 * local
    dh11 = 3.0 * local2 - 2.0 * local
    interval = 1.0 / float(count - 1)
    value = (
        h00 * values[lower]
        + h01 * values[lower + 1]
        + interval * h10 * slopes[lower]
        + interval * h11 * slopes[lower + 1]
    )
    derivative = (
        float(count - 1)
        * (dh00 * values[lower] + dh01 * values[lower + 1])
        + dh10 * slopes[lower]
        + dh11 * slopes[lower + 1]
    )
    return value, derivative


def _gauged_value_and_derivative(
    values: np.ndarray, slopes: np.ndarray, tau: float
) -> tuple[np.ndarray, np.ndarray]:
    raw, raw_d = _hermite_value_and_derivative(values, slopes, tau)
    norm = float(np.linalg.norm(raw))
    norm_d = float(np.dot(raw, raw_d) / norm)
    denominator = 1.0 + norm
    value = raw / denominator
    derivative = raw_d / denominator - raw * norm_d / denominator**2
    return value, derivative


def test_density_paths_are_seeded_random_internal_nonzero_and_c1():
    base, expanded, policy, latent = _sample(401)
    _base2, _expanded2, _policy2, replay = _sample(401)
    _base3, _expanded3, _policy3, other = _sample(402)

    assert torch.equal(latent, replay)
    assert not torch.equal(latent, other)

    layout = expanded.structured_latent_layout()
    count = int(expanded.structured_time_control_count)
    rho = (
        latent[: layout["rho_latent_dim"]]
        .reshape(count * 2, int(layout["rho_mode_count"]), 2)
        .detach()
        .cpu()
        .numpy()
    )
    active = np.flatnonzero(
        np.max(np.linalg.norm(rho, axis=-1), axis=0) > 0.0
    )
    assert active.size == policy.density_rank

    dense_tau = np.linspace(1.0e-6, 1.0, 2001)
    for mode_index in active:
        values = rho[:count, mode_index]
        slopes = rho[count:, mode_index]
        assert np.array_equal(values[0], np.zeros(2))
        assert np.array_equal(slopes[0], np.zeros(2))
        norms = np.asarray(
            [
                np.linalg.norm(
                    _hermite_value_and_derivative(values, slopes, tau)[0]
                )
                for tau in dense_tau
            ]
        )
        assert np.all(norms > 0.0)
        for knot in np.linspace(0.0, 1.0, count)[1:-1]:
            left_value, left_d = _gauged_value_and_derivative(
                values, slopes, knot - 1.0e-8
            )
            right_value, right_d = _gauged_value_and_derivative(
                values, slopes, knot + 1.0e-8
            )
            np.testing.assert_allclose(left_value, right_value, atol=2.0e-7)
            np.testing.assert_allclose(left_d, right_d, atol=2.0e-6)

    assert base.rho_max_mode_x < expanded.rho_max_mode_x


def test_analytic_density_derivative_matches_autograd_and_rejects_a_cusp():
    _base, expanded, _policy, latent = _sample(173)
    model = robust_perturbation.build_deterministic_direct_model(
        expanded, device="cpu"
    )
    condition = torch.tensor([0.05, 1.0], dtype=torch.float64)
    query = torch.tensor(
        [
            [0.037, 0.3, 0.8, -0.5, 0.25],
            [0.119, 1.2, 0.4, 0.0, -0.25],
            [0.231, 2.1, 1.7, 0.5, 0.75],
            [0.319, 4.2, 5.1, -0.75, 0.0],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    rho = model.fields(
        query, latent, equation_condition=condition
    )["rho"]
    autodiff = torch.autograd.grad(rho.sum(), query)[0][:, 0]
    analytic = analytic_source.analytic_structured_macro_jet(
        model, query[:, :3].detach(), latent
    )["rho_derivatives"][0]
    torch.testing.assert_close(analytic, autodiff, rtol=2.0e-10, atol=2.0e-11)

    mode_count = len(model.rho_half_modes)
    raw = torch.zeros((1, 2 * mode_count), dtype=torch.float64)
    raw_t = torch.zeros_like(raw)
    raw_t[0, 0] = 1.0
    with pytest.raises(ValueError, match="group norm crosses zero"):
        analytic_source._rho_coordinate_jet(
            model,
            torch.zeros((1, 3), dtype=torch.float64),
            raw,
            raw_t,
        )


def _write_checkpoint(path: Path, model) -> None:
    encoder = VPFPFlowObservationEncoder.build(
        VPFPFlowObservationEncoderConfig(
            latent_dim=4,
            width=4,
            point_layers=1,
            attention_heads=1,
            query_tokens=1,
            self_attention_layers=1,
            time_feature_modes=1,
            spatial_feature_modes=1,
            time_domain=model.config.time_domain,
            x_domain=model.config.equation.x_domain,
            collision_nu_range=model.config.collision_nu_range,
            temperature_range=model.config.temperature_range,
            dtype="float64",
        )
    )
    save_hermite_autoencoder_checkpoint(
        path,
        decoder=model,
        encoder=encoder,
        feature_normalization={},
        latent_normalization={},
        case_latents=torch.empty((0, int(model.config.latent_dim))),
        case_ids=(),
        train_indices=(),
        validation_indices=(),
        train_config={"role": "answer_first_initial_gate_test"},
        step=0,
    )


def _materialize_perturbed_case(root: Path) -> Path:
    _base, expanded, _policy, latent = _sample(991)
    model = robust_perturbation.build_deterministic_direct_model(
        expanded, device="cpu"
    )
    checkpoint = root / "decoder.pt"
    _write_checkpoint(checkpoint, model)
    case_dir = root / "case"
    grid = VPFPGrid(
        nx=9,
        ny=9,
        nvx=3,
        nvy=3,
        x_domain=expanded.equation.x_domain,
        dtype="float64",
    )
    time = VPFPTime(
        t0=expanded.time_domain[0],
        t1=expanded.time_domain[1],
        nt=3,
    )
    materialize_continuous_case(
        case_dir,
        case_id="vpfp-answer-first-perturbed",
        model=model,
        decoder_checkpoint=checkpoint,
        latent=latent,
        grid=grid,
        time=time,
        equation_condition=np.asarray([0.05, 1.0]),
        source_derivative_backend="analytic",
        solver_output_contract=MACRO_OUTPUT_CONTRACT,
        case_spec={
            "truth_primary": "continuous_hidden_answer",
            "answer_first_perturbation_applied": True,
        },
        evaluation_chunk_size=grid.nx * grid.ny * grid.nvx * grid.nvy,
    )
    return case_dir


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tamper_public_initial_array(case_dir: Path, name: str) -> None:
    public_path = case_dir / "public_arrays.npz"
    with np.load(public_path) as payload:
        arrays = {key: np.asarray(payload[key]) for key in payload.files}
    arrays[name] = arrays[name].copy()
    arrays[name].reshape(-1)[0] += 0.125
    np.savez_compressed(public_path, **arrays)

    manifest_path = case_dir / "hidden_truth_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["continuous_primary"]["public_arrays_sha256"] = _sha256(public_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_public_initial_data_is_rederived_from_primary_hidden_answer(tmp_path):
    case_dir = _materialize_perturbed_case(tmp_path)
    report = verify_continuous_case_dir(
        case_dir,
        device="cpu",
        truth_sample_count=16,
        source_sample_count=8,
    )
    assert report["accepted"] is True
    assert report["metrics"]["answer_first_initial_data_rederived_from_primary"]
    assert report["metrics"]["rho0_materialized_cache_exact"]
    assert report["metrics"]["g0_materialized_cache_exact"]

    for name in ("rho0", "g0"):
        tampered = tmp_path / f"tampered-{name}"
        shutil.copytree(case_dir, tampered)
        _tamper_public_initial_array(tampered, name)
        rejected = verify_continuous_case_dir(
            tampered,
            device="cpu",
            truth_sample_count=16,
            source_sample_count=8,
        )
        assert rejected["accepted"] is False
        assert f"{name}_materialized_cache_mismatch" in rejected["failures"]
        assert f"{name}_primary_answer_mismatch" in rejected["failures"]
        assert "momentum0_mismatch" not in rejected["failures"]
        assert "kinetic_energy0_mismatch" not in rejected["failures"]
