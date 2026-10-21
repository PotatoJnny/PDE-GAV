from __future__ import annotations

import json

import numpy as np
import pytest

from pde_agent_verifier_solver.reconstruction import (
    GAPPY_POD_LS,
    POD_LINEAR_GAUSSIAN_MAP,
    GappyPODLSProgram,
    PODBasis,
    PODCompletedData,
    PODFitData,
    PODLinearGaussianMAPProgram,
    PODNoiseModel,
    PODObservation,
    ReconstructionNumericsError,
    fit_weighted_pod,
    gappy_pod_ls,
    get_reconstruction_program,
    pod_linear_gaussian_map,
    pretrained_pod_basis,
    reconstruction_programs,
)
from pde_agent_verifier_solver.sdk import ReconstructionInput


def _basis() -> PODBasis:
    return pretrained_pod_basis(
        mean=np.array([1.0, -2.0, 0.5, 3.0]),
        modes=np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        ),
        modal_variances=np.array([2.0, 0.5]),
    )


def test_weighted_pod_modes_are_q_orthonormal() -> None:
    snapshots = np.array(
        [
            [0.0, 1.0, -1.0, 2.0],
            [1.0, 0.0, 2.0, -1.0],
            [2.0, 3.0, 0.0, 1.0],
            [-1.0, 2.0, 1.0, 0.0],
            [3.0, -2.0, 1.0, 4.0],
        ]
    )
    quadrature = np.array([0.25, 2.0, 1.5, 4.0])

    basis = fit_weighted_pod(
        snapshots,
        rank=3,
        quadrature_weights=quadrature,
    )

    gram = basis.modes.T @ (quadrature[:, None] * basis.modes)
    np.testing.assert_allclose(gram, np.eye(3), rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(basis.mean, snapshots.mean(axis=0))
    assert basis.calibration_count == 5


def test_gappy_pod_ls_recovers_exact_field() -> None:
    basis = _basis()
    expected_coefficients = np.array([0.75, -1.25])
    expected = basis.mean + basis.modes @ expected_coefficients
    indices = np.array([0, 1, 2])

    result = gappy_pod_ls(basis, expected[indices], indices)

    assert result.method_id == GAPPY_POD_LS
    np.testing.assert_allclose(result.coefficients, expected_coefficients, atol=1.0e-13)
    np.testing.assert_allclose(result.completed, expected, atol=1.0e-13)
    assert result.observation_rank == 2
    assert result.whitened_residual_norm < 1.0e-13
    assert "whitened_solve_design_condition" in result.diagnostics()


@pytest.mark.parametrize(
    "noise_covariance",
    (
        np.array([0.2, 2.0, 5.0]),
        np.array(
            [
                [0.5, 0.1, 0.0],
                [0.1, 1.5, 0.2],
                [0.0, 0.2, 2.0],
            ]
        ),
    ),
)
def test_gappy_pod_ls_matches_heteroscedastic_and_correlated_gls(
    noise_covariance: np.ndarray,
) -> None:
    basis = pretrained_pod_basis(
        mean=np.zeros(3),
        modes=np.ones((3, 1)) / np.sqrt(3.0),
    )
    operator = np.eye(3)
    observations = np.array([0.8, 1.4, 0.1])
    covariance = (
        np.diag(noise_covariance)
        if noise_covariance.ndim == 1
        else noise_covariance
    )
    design = operator @ basis.modes
    solved_design = np.linalg.solve(covariance, design)
    solved_target = np.linalg.solve(covariance, observations)
    expected = np.linalg.solve(
        design.T @ solved_design,
        design.T @ solved_target,
    )

    result = gappy_pod_ls(
        basis,
        observations,
        operator,
        noise_covariance=noise_covariance,
    )

    np.testing.assert_allclose(result.coefficients, expected, rtol=1.0e-12, atol=1.0e-12)


def test_pod_map_matches_closed_form_with_correlated_noise_and_prior() -> None:
    basis = _basis()
    operator = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )
    observations = np.array([1.8, -1.0, -2.4])
    noise = np.array(
        [
            [0.4, 0.1, 0.0],
            [0.1, 0.8, 0.05],
            [0.0, 0.05, 0.3],
        ]
    )
    prior_mean = np.array([0.2, -0.1])
    prior_covariance = np.array([[1.5, 0.25], [0.25, 0.7]])
    design = operator @ basis.modes
    target = observations - operator @ basis.mean
    noise_solved_design = np.linalg.solve(noise, design)
    noise_solved_target = np.linalg.solve(noise, target)
    prior_precision = np.linalg.solve(prior_covariance, np.eye(2))
    expected = np.linalg.solve(
        design.T @ noise_solved_design + prior_precision,
        design.T @ noise_solved_target + prior_precision @ prior_mean,
    )

    result = pod_linear_gaussian_map(
        basis,
        observations,
        operator,
        noise_covariance=noise,
        coefficient_prior_covariance=prior_covariance,
        coefficient_prior_mean=prior_mean,
    )

    assert result.method_id == POD_LINEAR_GAUSSIAN_MAP
    np.testing.assert_allclose(result.coefficients, expected, rtol=1.0e-12, atol=1.0e-12)


def test_ls_rank_failure_is_closed_but_map_reports_prior_reliance() -> None:
    basis = _basis()
    indices = np.array([0])
    observations = np.array([1.5])

    with pytest.raises(ReconstructionNumericsError, match="observations for 2 coefficients"):
        gappy_pod_ls(basis, observations, indices)

    result = pod_linear_gaussian_map(
        basis,
        observations,
        indices,
        noise_covariance=1.0e6,
        coefficient_prior_precision=1.0e3,
    )
    assert result.observation_rank == 1
    assert result.data_precision_trace_fraction is not None
    assert result.data_precision_trace_fraction < 0.1
    assert any("rank deficient" in warning for warning in result.warnings)
    assert any("prior dominates" in warning for warning in result.warnings)


def test_ls_rejects_singular_duplicate_sensor_layout() -> None:
    basis = _basis()
    with pytest.raises(ReconstructionNumericsError, match="rank deficient"):
        gappy_pod_ls(
            basis,
            observations=np.array([1.5, 1.5]),
            observation_operator=np.array([0, 0]),
        )


def test_ls_program_rejects_full_rank_over_conditioned_observation_design() -> None:
    basis = pretrained_pod_basis(mean=np.zeros(2), modes=np.eye(2))
    operator = np.array([[1.0, 0.0], [1.0, 1.0e-8]])
    target = np.array([1.0, -0.5])
    source = ReconstructionInput(
        public_split_id="ill-conditioned-design",
        fit_data=PODFitData(None, np.ones(2)),
        target_observations=(
            PODObservation(
                "target",
                None,
                operator @ target,
                observation_operator=operator,
            ),
        ),
        geometry={},
        pretrained_basis=basis,
        metadata={"fields": ["velocity"]},
    )

    with pytest.raises(ReconstructionNumericsError, match="too ill-conditioned"):
        GappyPODLSProgram().reconstruct(
            source,
            {"rank": 2, "max_condition": 10.0},
            seed=7,
        )


def test_map_regularizes_a_noisy_estimate_toward_declared_prior() -> None:
    basis = pretrained_pod_basis(
        mean=np.zeros(3),
        modes=np.ones((3, 1)) / np.sqrt(3.0),
    )
    observations = np.array([3.0, 2.5, 4.0])
    operator = np.arange(3)
    least_squares = gappy_pod_ls(basis, observations, operator)

    mapped = pod_linear_gaussian_map(
        basis,
        observations,
        operator,
        noise_covariance=1.0,
        coefficient_prior_precision=20.0,
        coefficient_prior_mean=np.array([0.0]),
    )

    assert abs(mapped.coefficients[0]) < abs(least_squares.coefficients[0])


def test_map_program_uses_basis_modal_variances_as_default_prior() -> None:
    basis = _basis()
    target = basis.mean + basis.modes @ np.array([0.5, -0.25])
    source = ReconstructionInput(
        public_split_id="basis-prior",
        fit_data=PODFitData(None, np.ones(4)),
        target_observations=(
            PODObservation(
                "target-1",
                np.array([0, 1]),
                target[[0, 1]],
                noise_covariance=0.2,
            ),
        ),
        geometry={},
        pretrained_basis=basis,
        metadata={"fields": ["velocity"]},
    )

    output = PODLinearGaussianMAPProgram().reconstruct(
        source,
        {"rank": 2},
        seed=7,
    )

    assert output.public_settings["coefficient_prior_source"] == (
        "basis_modal_variances"
    )
    assert output.public_settings["coefficient_prior_kind"] == "diagonal"
    assert output.public_settings["coefficient_prior_mean"] == "zero"
    assert output.regularization is None


def test_pretrained_basis_program_path_does_not_require_snapshots() -> None:
    basis = _basis()
    target = basis.mean + basis.modes @ np.array([0.5, -0.25])
    source = ReconstructionInput(
        public_split_id="frozen-split",
        fit_data=PODFitData(None, np.ones(4)),
        target_observations=(
            PODObservation("target-1", np.array([0, 1]), target[[0, 1]]),
        ),
        geometry={"layout": "plugin-owned"},
        pretrained_basis=basis,
        metadata={"fields": ["velocity"]},
    )

    output = GappyPODLSProgram().reconstruct(source, {"rank": 2}, seed=7)

    assert output.method_id == GAPPY_POD_LS
    assert output.fields == ("velocity",)
    assert output.diagnostics["basis_source"] == "pretrained_basis"
    assert isinstance(output.completed_data, PODCompletedData)
    np.testing.assert_allclose(output.completed_data.values[0], target)


def test_map_program_and_registry_expose_distinct_public_methods() -> None:
    basis = _basis()
    target = basis.mean + basis.modes @ np.array([0.5, -0.25])
    source = ReconstructionInput(
        public_split_id="frozen-split",
        fit_data=PODFitData(None, np.ones(4)),
        target_observations=(
            PODObservation(
                "target-1",
                np.array([0]),
                target[[0]],
                noise_covariance=0.2,
            ),
        ),
        geometry={},
        pretrained_basis=basis,
        metadata={"fields": ("velocity",)},
    )

    output = PODLinearGaussianMAPProgram().reconstruct(
        source,
        {"rank": 2, "prior_precision": 2.0},
        seed=7,
    )

    assert output.method_id == POD_LINEAR_GAUSSIAN_MAP
    assert output.regularization == 2.0
    assert set(reconstruction_programs()) == {
        GAPPY_POD_LS,
        POD_LINEAR_GAUSSIAN_MAP,
    }
    assert isinstance(get_reconstruction_program(GAPPY_POD_LS), GappyPODLSProgram)
    assert isinstance(
        get_reconstruction_program(POD_LINEAR_GAUSSIAN_MAP),
        PODLinearGaussianMAPProgram,
    )


def test_program_noise_model_supports_explicit_global_and_target_routing() -> None:
    basis = _basis()
    target = basis.mean + basis.modes @ np.array([0.5, -0.25])
    observation = PODObservation(
        "target-1",
        np.array([0, 1]),
        target[[0, 1]],
    )
    common = dict(
        public_split_id="noise-routing",
        fit_data=PODFitData(None, np.ones(4)),
        target_observations=(observation,),
        geometry={},
        pretrained_basis=basis,
        metadata={"fields": ["velocity"]},
    )

    global_output = GappyPODLSProgram().reconstruct(
        ReconstructionInput(**common, noise_model=0.25),
        {"rank": 2},
        seed=7,
    )
    target_output = PODLinearGaussianMAPProgram().reconstruct(
        ReconstructionInput(
            **common,
            noise_model=PODNoiseModel(by_target={"target-1": 0.25}),
        ),
        {"rank": 2, "prior_precision": 1.0},
        seed=7,
    )

    assert global_output.method_id == GAPPY_POD_LS
    assert target_output.method_id == POD_LINEAR_GAUSSIAN_MAP
    assert target_output.public_settings["coefficient_prior_source"] == (
        "recipe_precision"
    )
    assert target_output.public_settings["noise_covariance_kinds"] == ("scalar",)


@pytest.mark.parametrize("program", (GappyPODLSProgram(), PODLinearGaussianMAPProgram()))
def test_program_rejects_ambiguous_or_incomplete_mapping_noise_model(program) -> None:
    basis = _basis()
    source = ReconstructionInput(
        public_split_id="bad-noise-routing",
        fit_data=PODFitData(None, np.ones(4)),
        target_observations=(
            PODObservation("target-1", np.array([0, 1]), np.array([1.5, -2.25])),
        ),
        geometry={},
        noise_model={"variance": 0.25},
        pretrained_basis=basis,
        metadata={"fields": ["velocity"]},
    )

    with pytest.raises(ValueError, match="must contain every POD target ID"):
        program.reconstruct(
            source,
            {"rank": 2},
            seed=7,
        )


def test_typed_noise_model_failure_does_not_disclose_private_target_id() -> None:
    private_id = "/sealed/piv/frame-001.npy"
    source = ReconstructionInput(
        public_split_id="private-noise-routing",
        fit_data=PODFitData(None, np.ones(4)),
        target_observations=(
            PODObservation(private_id, np.array([0, 1]), np.array([1.5, -2.25])),
        ),
        geometry={},
        noise_model=PODNoiseModel(by_target={"another-target": 0.25}),
        pretrained_basis=_basis(),
        metadata={"fields": ["velocity"]},
    )

    with pytest.raises(ValueError) as caught:
        GappyPODLSProgram().reconstruct(source, {"rank": 2}, seed=7)
    assert private_id not in str(caught.value)


def test_program_rejects_invalid_seed_and_pretrained_weight_shape() -> None:
    basis = _basis()
    source = ReconstructionInput(
        public_split_id="invalid-seed",
        fit_data=PODFitData(None, np.ones(4)),
        target_observations=(
            PODObservation("target", np.array([0, 1]), np.array([1.5, -2.25])),
        ),
        geometry={},
        pretrained_basis=basis,
        metadata={"fields": ["velocity"]},
    )
    with pytest.raises(TypeError, match="seed must be an integer"):
        GappyPODLSProgram().reconstruct(source, {"rank": 2}, seed=True)
    with pytest.raises(ValueError, match="seed must be nonnegative"):
        GappyPODLSProgram().reconstruct(source, {"rank": 2}, seed=-1)

    mismatched = ReconstructionInput(
        public_split_id="wrong-weight-shape",
        fit_data=PODFitData(None, np.ones(1)),
        target_observations=source.target_observations,
        geometry={},
        pretrained_basis=basis,
        metadata={"fields": ["velocity"]},
    )
    with pytest.raises(ValueError, match="different quadrature weights"):
        GappyPODLSProgram().reconstruct(mismatched, {"rank": 2}, seed=7)


def test_program_preserves_multi_target_shapes_and_ids() -> None:
    basis = _basis()
    first = basis.mean + basis.modes @ np.array([0.5, -0.25])
    second = basis.mean + basis.modes @ np.array([-0.75, 1.25])
    source = ReconstructionInput(
        public_split_id="multi-target",
        fit_data=PODFitData(None, np.ones(4)),
        target_observations=(
            PODObservation("first", np.array([0, 1]), first[[0, 1]]),
            PODObservation("second", np.array([0, 1]), second[[0, 1]]),
        ),
        geometry={},
        pretrained_basis=basis,
        metadata={"fields": ["velocity_x", "velocity_y"]},
    )

    output = GappyPODLSProgram().reconstruct(source, {"rank": 2}, seed=7)

    assert isinstance(output.completed_data, PODCompletedData)
    assert output.completed_data.target_ids == ("first", "second")
    assert output.completed_data.values.shape == (2, 4)
    assert output.completed_data.coefficients.shape == (2, 2)
    np.testing.assert_allclose(output.completed_data.values, np.stack((first, second)))


def test_completed_data_detaches_mutable_target_id_list() -> None:
    target_ids = ["target"]
    completed = PODCompletedData(
        target_ids=target_ids,  # type: ignore[arg-type]
        values=np.zeros((1, 4)),
        coefficients=np.zeros((1, 2)),
        basis=_basis(),
    )

    target_ids.append("mutated")
    assert completed.target_ids == ("target",)


def test_program_supports_plugin_owned_dense_observation_operator() -> None:
    basis = _basis()
    target = basis.mean + basis.modes @ np.array([0.4, -0.7])
    operator = np.array(
        [
            [0.75, 0.25, 0.0, 0.0],
            [0.15, 0.85, 0.0, 0.0],
            [0.50, 0.20, 0.20, 0.10],
        ]
    )
    source = ReconstructionInput(
        public_split_id="off-grid",
        fit_data=PODFitData(None, np.ones(4)),
        target_observations=(
            PODObservation(
                "private-target-id",
                None,
                operator @ target,
                observation_operator=operator,
            ),
        ),
        geometry={"operator_owner": "problem-plugin"},
        pretrained_basis=basis,
        metadata={"fields": ["velocity"]},
    )

    output = GappyPODLSProgram().reconstruct(source, {"rank": 2}, seed=7)

    assert isinstance(output.completed_data, PODCompletedData)
    np.testing.assert_allclose(output.completed_data.values[0], target, atol=1.0e-12)


def test_observation_requires_exactly_one_operator_representation() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        PODObservation("missing", None, np.array([1.0]))
    with pytest.raises(ValueError, match="exactly one"):
        PODObservation(
            "duplicate",
            np.array([0]),
            np.array([1.0]),
            observation_operator=np.ones((1, 2)),
        )


def test_private_target_id_is_not_copied_to_public_diagnostics() -> None:
    private_id = "/private/raw/piv/frame001.npy"
    basis = _basis()
    source = ReconstructionInput(
        public_split_id="privacy",
        fit_data=PODFitData(None, np.ones(4)),
        target_observations=(
            PODObservation(private_id, np.array([0]), np.array([1.5]), 1.0e6),
        ),
        geometry={},
        pretrained_basis=basis,
        metadata={"fields": ["velocity"]},
    )

    output = PODLinearGaussianMAPProgram().reconstruct(
        source,
        {"rank": 2, "prior_precision": 1.0e3},
        seed=7,
    )

    public_text = json.dumps(
        {"diagnostics": output.diagnostics, "warnings": output.warnings}
    )
    assert private_id not in public_text
    assert "target_1" in public_text
    assert output.completed_data.target_ids == (private_id,)


def test_invalid_covariance_and_method_settings_fail_explicitly() -> None:
    basis = _basis()
    with pytest.raises(ValueError, match="explicit observation noise"):
        pod_linear_gaussian_map(
            basis,
            np.array([1.0, 2.0]),
            np.array([0, 1]),
            noise_covariance=None,
            coefficient_prior_precision=1.0,
        )
    with pytest.raises(ReconstructionNumericsError, match="positive definite"):
        pod_linear_gaussian_map(
            basis,
            np.array([1.0, 2.0]),
            np.array([0, 1]),
            noise_covariance=np.array([[1.0, 2.0], [2.0, 1.0]]),
            coefficient_prior_precision=1.0,
        )
    with pytest.raises(ValueError, match="choose coefficient prior"):
        pod_linear_gaussian_map(
            basis,
            np.array([1.0, 2.0]),
            np.array([0, 1]),
            noise_covariance=1.0,
            coefficient_prior_covariance=1.0,
            coefficient_prior_precision=1.0,
        )
    with pytest.raises(TypeError, match="rank setting"):
        GappyPODLSProgram().reconstruct(
            ReconstructionInput(
                public_split_id="invalid-settings",
                fit_data=PODFitData(None, np.ones(4)),
                target_observations=(
                    PODObservation(
                        "target",
                        np.array([0, 1]),
                        np.array([1.0, 2.0]),
                    ),
                ),
                geometry={},
                pretrained_basis=basis,
                metadata={"fields": ["velocity"]},
            ),
            {"rank": True},
            seed=7,
        )
    with pytest.raises(ValueError, match="unknown POD method settings"):
        GappyPODLSProgram().reconstruct(
            ReconstructionInput(
                public_split_id="unknown-setting",
                fit_data=PODFitData(None, np.ones(4)),
                target_observations=(
                    PODObservation(
                        "target",
                        np.array([0, 1]),
                        np.array([1.0, 2.0]),
                    ),
                ),
                geometry={},
                pretrained_basis=basis,
                metadata={"fields": ["velocity"]},
            ),
            {"rank": 2, "prior_dominance_threshold": 0.2},
            seed=7,
        )


def test_pretrained_basis_rejects_wrong_quadrature_semantics() -> None:
    with pytest.raises(ValueError, match="not orthonormal"):
        pretrained_pod_basis(
            mean=np.zeros(2),
            modes=np.eye(2),
            quadrature_weights=np.array([1.0, 2.0]),
        )


def test_program_rejects_ambiguous_calibration_and_pretrained_basis() -> None:
    basis = _basis()
    source = ReconstructionInput(
        public_split_id="ambiguous-basis",
        fit_data=PODFitData(np.zeros((2, 4)), np.ones(4)),
        target_observations=(
            PODObservation("target", np.array([0, 1]), np.array([1.5, -2.25])),
        ),
        geometry={},
        pretrained_basis=basis,
        metadata={"fields": ["velocity"]},
    )

    with pytest.raises(ValueError, match="not both"):
        GappyPODLSProgram().reconstruct(source, {"rank": 2}, seed=7)


def test_real_valued_builtins_reject_complex_data_without_truncation() -> None:
    with pytest.raises(TypeError, match="must be real-valued"):
        pretrained_pod_basis(
            mean=np.array([1.0 + 7.0j, 0.0]),
            modes=np.eye(2),
        )

    with pytest.raises(TypeError, match="must be real-valued"):
        pod_linear_gaussian_map(
            _basis(),
            np.array([1.0, 2.0]),
            np.array([0, 1]),
            noise_covariance=np.array(
                [[1.0 + 0.0j, 0.25j], [-0.25j, 1.0 + 0.0j]]
            ),
            coefficient_prior_precision=1.0,
        )


def test_numeric_arrays_reject_boolean_values() -> None:
    with pytest.raises(TypeError, match="not boolean"):
        pretrained_pod_basis(
            mean=np.array([False, False]),
            modes=np.eye(2),
        )
    with pytest.raises(TypeError, match="not boolean"):
        pretrained_pod_basis(
            mean=[True, 1.0],
            modes=np.eye(2),
        )
