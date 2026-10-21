"""PDE-neutral, real-valued Gappy POD reconstruction methods.

The module keeps the two inner products in a Gappy POD calculation explicit:

* ``quadrature_weights`` define the spatial inner product used to learn POD
  modes from calibration snapshots;
* ``noise_covariance`` defines the observation likelihood used to infer modal
  coefficients for a target.

They are deliberately never combined into a generic ``weights`` argument.
``gappy_pod_ls`` implements the classical (generalized) least-squares fit,
while ``pod_linear_gaussian_map`` adds an explicitly named Gaussian
coefficient prior.  Both solvers use whitening and ``numpy.linalg.lstsq``;
neither forms or inverts the normal equations, nor unconditionally falls back
after a rank failure.

The fixed-basis formulation follows Willcox, *Computers & Fluids* 35 (2006),
208--226, https://doi.org/10.1016/j.compfluid.2004.11.006. The MAP extension
uses the standard finite-dimensional linear-Gaussian inverse problem and is
named separately so it is not presented as part of the original LS method.
Complex-valued PDE data require a plugin-owned method that implements the
appropriate Hermitian inner products; the built-ins fail closed rather than
silently discarding imaginary components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ..sdk.models import FormField, ProgramInfo
from ..sdk.reconstruction import (
    ReconstructionInput,
    ReconstructionOutput,
    ReconstructionProgress,
)


GAPPY_POD_LS = "gappy_pod_ls"
POD_LINEAR_GAUSSIAN_MAP = "pod_linear_gaussian_map"


class ReconstructionNumericsError(ValueError):
    """Raised when a requested reconstruction is not numerically identified."""


@dataclass(frozen=True)
class PODNoiseModel:
    """Explicit observation-covariance routing for POD reconstruction.

    A single scalar, vector, or matrix can be supplied directly as
    ``ReconstructionInput.noise_model``.  Use this wrapper only when covariance
    differs by target or when a default with target-specific overrides is
    needed.  Missing target entries fail closed when no default is declared.
    """

    default_covariance: object | None = None
    by_target: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = dict(self.by_target)
        if not values and self.default_covariance is None:
            raise ValueError(
                "PODNoiseModel requires a default covariance or target overrides"
            )
        if any(
            not isinstance(target_id, str) or not target_id.strip()
            for target_id in values
        ):
            raise ValueError("PODNoiseModel target IDs must be nonempty strings")
        if any(covariance is None for covariance in values.values()):
            raise ValueError("PODNoiseModel target covariances cannot be None")
        object.__setattr__(self, "by_target", MappingProxyType(values))

    def covariance_for(self, target_id: str) -> object:
        if target_id in self.by_target:
            return self.by_target[target_id]
        if self.default_covariance is not None:
            return self.default_covariance
        raise ValueError(
            "PODNoiseModel has no covariance for one or more reconstruction targets"
        )


@dataclass(frozen=True)
class PODBasis:
    """Mean and modes that are orthonormal in a quadrature inner product.

    Arrays use a flattened, plugin-owned feature order.  If ``Q`` is the
    diagonal matrix represented by ``quadrature_weights``, construction checks
    ``modes.T @ Q @ modes == I``.  ``modal_variances`` are optional because a
    pretrained basis need not carry a statistical prior.
    """

    mean: np.ndarray
    modes: np.ndarray
    quadrature_weights: np.ndarray
    modal_variances: np.ndarray | None = None
    calibration_count: int | None = None

    def __post_init__(self) -> None:
        mean = _finite_vector(self.mean, "POD mean")
        modes = _finite_matrix(self.modes, "POD modes")
        if modes.shape[0] != mean.size:
            raise ValueError("POD modes and mean have different feature counts")
        if modes.shape[1] < 1:
            raise ValueError("POD basis must contain at least one mode")
        weights = _quadrature_weights(self.quadrature_weights, mean.size)
        gram = modes.T @ (weights[:, None] * modes)
        identity = np.eye(modes.shape[1], dtype=np.float64)
        if not np.allclose(gram, identity, rtol=1.0e-7, atol=1.0e-9):
            deviation = float(np.linalg.norm(gram - identity, ord=np.inf))
            raise ValueError(
                "POD modes are not orthonormal in the supplied quadrature "
                f"inner product (infinity-norm deviation {deviation:.3e})"
            )

        variances = None
        if self.modal_variances is not None:
            variances = _finite_vector(
                self.modal_variances,
                "POD modal variances",
            )
            if variances.size != modes.shape[1]:
                raise ValueError("POD modal variances must match the mode count")
            if np.any(variances <= 0.0):
                raise ValueError("POD modal variances must be strictly positive")
        if self.calibration_count is not None:
            if (
                isinstance(self.calibration_count, bool)
                or not isinstance(self.calibration_count, int)
                or self.calibration_count < 1
            ):
                raise ValueError("POD calibration_count must be a positive integer")

        object.__setattr__(self, "mean", _readonly(mean))
        object.__setattr__(self, "modes", _readonly(modes))
        object.__setattr__(self, "quadrature_weights", _readonly(weights))
        object.__setattr__(
            self,
            "modal_variances",
            None if variances is None else _readonly(variances),
        )

    @property
    def feature_count(self) -> int:
        return int(self.mean.size)

    @property
    def rank(self) -> int:
        return int(self.modes.shape[1])

    def truncated(self, rank: int) -> "PODBasis":
        """Return the leading ``rank`` modes without changing their meaning."""

        selected = _positive_rank(rank, self.rank)
        if selected == self.rank:
            return self
        return PODBasis(
            mean=self.mean,
            modes=self.modes[:, :selected],
            quadrature_weights=self.quadrature_weights,
            modal_variances=(
                None
                if self.modal_variances is None
                else self.modal_variances[:selected]
            ),
            calibration_count=self.calibration_count,
        )


@dataclass(frozen=True)
class GappyPODResult:
    """One reconstructed flattened field and auditable solver diagnostics."""

    method_id: str
    completed: np.ndarray
    coefficients: np.ndarray
    observation_residual_norm: float
    whitened_residual_norm: float
    observation_rank: int
    whitened_solve_design_condition: float
    data_precision_trace_fraction: float | None = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        completed = _finite_vector(self.completed, "completed reconstruction")
        coefficients = _finite_vector(self.coefficients, "POD coefficients")
        for value, name in (
            (self.observation_residual_norm, "observation residual norm"),
            (self.whitened_residual_norm, "whitened residual norm"),
            (
                self.whitened_solve_design_condition,
                "whitened solve-design condition",
            ),
        ):
            if not isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.observation_rank < 0:
            raise ValueError("observation_rank must be nonnegative")
        if self.data_precision_trace_fraction is not None and not (
            0.0 <= float(self.data_precision_trace_fraction) <= 1.0
        ):
            raise ValueError("data_precision_trace_fraction must lie in [0, 1]")
        object.__setattr__(self, "completed", _readonly(completed))
        object.__setattr__(self, "coefficients", _readonly(coefficients))

    def diagnostics(self) -> dict[str, object]:
        values: dict[str, object] = {
            "observation_residual_norm": float(self.observation_residual_norm),
            "whitened_residual_norm": float(self.whitened_residual_norm),
            "observation_rank": int(self.observation_rank),
            "whitened_solve_design_condition": float(
                self.whitened_solve_design_condition
            ),
        }
        if self.data_precision_trace_fraction is not None:
            values["data_precision_trace_fraction"] = float(
                self.data_precision_trace_fraction
            )
        return values


@dataclass(frozen=True)
class PODFitData:
    """Calibration snapshots and their geometry-owned quadrature weights.

    ``snapshots`` may be ``None`` only when ``ReconstructionInput`` supplies a
    validated :class:`PODBasis` as ``pretrained_basis``.
    """

    snapshots: np.ndarray | None
    quadrature_weights: np.ndarray

    def __post_init__(self) -> None:
        weights = _finite_vector(self.quadrature_weights, "quadrature weights")
        if np.any(weights <= 0.0):
            raise ValueError("quadrature weights must be strictly positive")
        snapshots = None
        if self.snapshots is not None:
            snapshots = _finite_matrix(self.snapshots, "calibration snapshots")
            if snapshots.shape[1] != weights.size:
                raise ValueError(
                    "calibration snapshots and quadrature weights have "
                    "different feature counts"
                )
        object.__setattr__(self, "snapshots", None if snapshots is None else _readonly(snapshots))
        object.__setattr__(self, "quadrature_weights", _readonly(weights))


@dataclass(frozen=True)
class PODObservation:
    """Sparse observations of one target in the plugin's flattened order.

    Direct samples use ``indices``. Interpolated, averaged, or off-grid
    measurements instead set ``indices=None`` and supply a dense linear
    ``observation_operator``. Exactly one representation is required.
    ``target_id`` remains inside trusted materialization data and is never
    copied into public method diagnostics.
    """

    target_id: str
    indices: np.ndarray | None
    values: np.ndarray
    noise_covariance: object | None = None
    observation_operator: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.target_id, str) or not self.target_id.strip():
            raise ValueError("POD observation target_id must be nonempty")
        values = _finite_vector(self.values, "POD observation values")
        if (self.indices is None) == (self.observation_operator is None):
            raise ValueError(
                "POD observation requires exactly one of indices or "
                "observation_operator"
            )
        if self.indices is not None:
            indices = np.asarray(self.indices)
            if indices.ndim != 1 or indices.size < 1:
                raise ValueError("POD observation indices must be a nonempty vector")
            if not np.issubdtype(indices.dtype, np.integer):
                raise TypeError("POD observation indices must be integers")
            indices = np.asarray(indices, dtype=np.int64)
            if np.any(indices < 0):
                raise ValueError("POD observation indices must be nonnegative")
            if values.size != indices.size:
                raise ValueError("POD observation indices and values differ in length")
            object.__setattr__(self, "indices", _readonly(indices))
        else:
            operator = _finite_matrix(
                self.observation_operator,
                "POD observation operator",
            )
            if operator.shape[0] != values.size or operator.shape[1] < 1:
                raise ValueError(
                    "POD observation operator rows must match values and it "
                    "must contain at least one feature column"
                )
            object.__setattr__(self, "observation_operator", _readonly(operator))
        object.__setattr__(self, "values", _readonly(values))

    @property
    def operator(self) -> np.ndarray:
        """Return the validated selection indices or dense linear operator."""

        if self.indices is not None:
            return self.indices
        assert self.observation_operator is not None
        return self.observation_operator


@dataclass(frozen=True)
class PODCompletedData:
    """Opaque completed-data payload returned to a Problem adapter."""

    target_ids: tuple[str, ...]
    values: np.ndarray
    coefficients: np.ndarray
    basis: PODBasis

    def __post_init__(self) -> None:
        if not isinstance(self.target_ids, (tuple, list)):
            raise TypeError("completed POD target IDs must be a list or tuple")
        target_ids = tuple(self.target_ids)
        if not target_ids or any(
            not isinstance(value, str) or not value.strip()
            for value in target_ids
        ):
            raise ValueError("completed POD data needs nonempty target IDs")
        if len(set(target_ids)) != len(target_ids):
            raise ValueError("completed POD target IDs must be unique")
        values = _finite_matrix(self.values, "completed POD values")
        coefficients = _finite_matrix(self.coefficients, "completed POD coefficients")
        if values.shape != (len(target_ids), self.basis.feature_count):
            raise ValueError("completed POD values have an inconsistent shape")
        if coefficients.shape != (len(target_ids), self.basis.rank):
            raise ValueError("completed POD coefficients have an inconsistent shape")
        object.__setattr__(self, "target_ids", target_ids)
        object.__setattr__(self, "values", _readonly(values))
        object.__setattr__(self, "coefficients", _readonly(coefficients))


def fit_weighted_pod(
    snapshots: object,
    *,
    rank: int,
    quadrature_weights: object | None = None,
) -> PODBasis:
    """Fit a fixed POD basis from complete calibration snapshots.

    The input has shape ``[calibration_snapshot, flattened_feature]``.  SVD is
    applied to ``(X - mean) Q**1/2`` and the returned physical-space modes are
    therefore orthonormal under ``Q``.
    """

    values = _finite_matrix(snapshots, "calibration snapshots")
    if values.shape[0] < 2:
        raise ValueError("weighted POD needs at least two calibration snapshots")
    weights = _quadrature_weights(quadrature_weights, values.shape[1])
    requested_rank = _positive_rank(rank, min(values.shape))
    mean = np.mean(values, axis=0, dtype=np.float64)
    centered = values - mean[None, :]
    weighted = centered * np.sqrt(weights)[None, :]
    try:
        _left, singular_values, right_transpose = np.linalg.svd(
            weighted,
            full_matrices=False,
        )
    except np.linalg.LinAlgError as exc:
        raise ReconstructionNumericsError("weighted POD SVD did not converge") from exc
    numerical_rank, _condition = _svd_rank_and_condition(singular_values, weighted.shape, None)
    if requested_rank > numerical_rank:
        raise ReconstructionNumericsError(
            f"requested POD rank {requested_rank} exceeds calibration numerical "
            f"rank {numerical_rank}"
        )
    modes = right_transpose[:requested_rank].T / np.sqrt(weights)[:, None]
    variances = (
        singular_values[:requested_rank] ** 2 / float(values.shape[0] - 1)
    )
    return PODBasis(
        mean=mean,
        modes=modes,
        quadrature_weights=weights,
        modal_variances=variances,
        calibration_count=int(values.shape[0]),
    )


def pretrained_pod_basis(
    *,
    mean: object,
    modes: object,
    quadrature_weights: object | None = None,
    modal_variances: object | None = None,
) -> PODBasis:
    """Validate a pretrained fixed basis without silently rotating its modes.

    A checkpoint must already use the declared quadrature inner product.  A
    silent QR correction would rotate coefficient priors and make a stored
    diagonal covariance scientifically ambiguous, so a mismatched checkpoint
    is rejected instead.
    """

    mean_array = _finite_vector(mean, "pretrained POD mean")
    return PODBasis(
        mean=mean_array,
        modes=_finite_matrix(modes, "pretrained POD modes"),
        quadrature_weights=_quadrature_weights(
            quadrature_weights,
            mean_array.size,
        ),
        modal_variances=(
            None
            if modal_variances is None
            else _finite_vector(modal_variances, "pretrained POD modal variances")
        ),
    )


def gappy_pod_ls(
    basis: PODBasis,
    observations: object,
    observation_operator: object,
    *,
    noise_covariance: object | None = None,
    rank: int | None = None,
    rcond: float | None = None,
    max_condition: float = 1.0e12,
) -> GappyPODResult:
    """Classical Gappy POD coefficient fit with optional GLS whitening.

    ``observation_operator`` may be a dense linear map ``[observation,
    feature]`` or an integer vector selecting flattened features.  LS fails
    closed if the observed basis is underdetermined, rank deficient, or more
    ill-conditioned than ``max_condition``.
    """

    selected = basis.truncated(basis.rank if rank is None else rank)
    values = _finite_vector(observations, "observations")
    observed_mean, observed_modes = _observe_basis(
        selected,
        observation_operator,
        values.size,
    )
    target = values - observed_mean
    design, whitened_target = _whiten_observations(
        observed_modes,
        target,
        noise_covariance,
    )
    condition_limit = _condition_limit(max_condition)
    singular_values = _singular_values(design, "Gappy POD observation design")
    observed_rank, condition = _svd_rank_and_condition(
        singular_values,
        design.shape,
        rcond,
    )
    if design.shape[0] < selected.rank:
        raise ReconstructionNumericsError(
            f"Gappy POD LS has {design.shape[0]} observations for "
            f"{selected.rank} coefficients"
        )
    if observed_rank < selected.rank:
        raise ReconstructionNumericsError(
            "Gappy POD LS observation design is rank deficient: "
            f"rank {observed_rank} for {selected.rank} coefficients"
        )
    if condition > condition_limit:
        raise ReconstructionNumericsError(
            "Gappy POD LS observation design is too ill-conditioned: "
            f"condition {condition:.3e} exceeds {condition_limit:.3e}"
        )
    coefficients = _least_squares(design, whitened_target, rcond)
    completed = selected.mean + selected.modes @ coefficients
    predicted = _apply_observation_operator(completed, observation_operator)
    residual = values - predicted
    _design_again, whitened_residual = _whiten_observations(
        observed_modes,
        residual,
        noise_covariance,
    )
    return GappyPODResult(
        method_id=GAPPY_POD_LS,
        completed=completed,
        coefficients=coefficients,
        observation_residual_norm=float(np.linalg.norm(residual)),
        whitened_residual_norm=float(np.linalg.norm(whitened_residual)),
        observation_rank=observed_rank,
        whitened_solve_design_condition=condition,
    )


def pod_linear_gaussian_map(
    basis: PODBasis,
    observations: object,
    observation_operator: object,
    *,
    noise_covariance: object,
    coefficient_prior_covariance: object | None = None,
    coefficient_prior_precision: object | None = None,
    coefficient_prior_mean: object | None = None,
    rank: int | None = None,
    rcond: float | None = None,
    max_condition: float = 1.0e12,
    minimum_data_precision_trace_fraction: float = 0.1,
) -> GappyPODResult:
    """Linear-Gaussian MAP estimate in POD coefficient space.

    The MAP problem is solved as one stacked, whitened least-squares system.
    At most one of covariance and precision may be supplied.  If neither is
    supplied, variances learned with the POD basis are used.
    """

    if noise_covariance is None:
        raise ValueError(
            "POD linear-Gaussian MAP requires an explicit observation noise covariance"
        )
    selected = basis.truncated(basis.rank if rank is None else rank)
    values = _finite_vector(observations, "observations")
    observed_mean, observed_modes = _observe_basis(
        selected,
        observation_operator,
        values.size,
    )
    target = values - observed_mean
    design, whitened_target = _whiten_observations(
        observed_modes,
        target,
        noise_covariance,
    )
    prior_mean = (
        np.zeros(selected.rank, dtype=np.float64)
        if coefficient_prior_mean is None
        else _finite_vector(coefficient_prior_mean, "coefficient prior mean")
    )
    if prior_mean.size != selected.rank:
        raise ValueError("coefficient prior mean must match the selected POD rank")
    prior_whitener = _prior_whitener(
        selected,
        covariance=coefficient_prior_covariance,
        precision=coefficient_prior_precision,
    )
    stacked_design = np.vstack((design, prior_whitener))
    stacked_target = np.concatenate((whitened_target, prior_whitener @ prior_mean))
    condition_limit = _condition_limit(max_condition)
    singular_values = _singular_values(stacked_design, "POD MAP design")
    solve_rank, condition = _svd_rank_and_condition(
        singular_values,
        stacked_design.shape,
        rcond,
    )
    if solve_rank < selected.rank:
        raise ReconstructionNumericsError(
            "POD MAP system is rank deficient even after applying the prior"
        )
    if condition > condition_limit:
        raise ReconstructionNumericsError(
            "POD MAP system is too ill-conditioned: "
            f"condition {condition:.3e} exceeds {condition_limit:.3e}"
        )
    coefficients = _least_squares(stacked_design, stacked_target, rcond)
    completed = selected.mean + selected.modes @ coefficients
    predicted = _apply_observation_operator(completed, observation_operator)
    residual = values - predicted
    _design_again, whitened_residual = _whiten_observations(
        observed_modes,
        residual,
        noise_covariance,
    )
    observation_singular_values = _singular_values(
        design,
        "POD MAP observation design",
    )
    observed_rank, _observed_condition = _svd_rank_and_condition(
        observation_singular_values,
        design.shape,
        rcond,
    )
    data_precision_trace = float(np.linalg.norm(design, ord="fro") ** 2)
    prior_precision_trace = float(np.linalg.norm(prior_whitener, ord="fro") ** 2)
    denominator = data_precision_trace + prior_precision_trace
    trace_fraction = data_precision_trace / denominator if denominator else 0.0
    threshold = _fraction_threshold(minimum_data_precision_trace_fraction)
    warnings: list[str] = []
    if observed_rank < selected.rank:
        warnings.append(
            "observation design is rank deficient; MAP reconstruction relies "
            "on the coefficient prior"
        )
    if trace_fraction < threshold:
        warnings.append(
            "coefficient prior dominates the configured whitened precision-trace "
            "diagnostic"
        )
    return GappyPODResult(
        method_id=POD_LINEAR_GAUSSIAN_MAP,
        completed=completed,
        coefficients=coefficients,
        observation_residual_norm=float(np.linalg.norm(residual)),
        whitened_residual_norm=float(np.linalg.norm(whitened_residual)),
        observation_rank=observed_rank,
        whitened_solve_design_condition=condition,
        data_precision_trace_fraction=trace_fraction,
        warnings=tuple(warnings),
    )


class _BaseGappyPODProgram:
    method_id: str
    info: ProgramInfo

    def setup_fields(self) -> tuple[FormField, ...]:
        return (
            FormField(
                "rank",
                "POD rank",
                "integer",
                8,
                "Number of fixed POD modes used for reconstruction.",
            ),
            FormField(
                "max_condition",
                "Maximum solve condition",
                "number",
                1.0e12,
                "Reject a numerically unstable whitened solve.",
            ),
        )

    def _inputs(
        self,
        source: ReconstructionInput,
        settings: Mapping[str, object],
    ) -> tuple[PODBasis, tuple[PODObservation, ...], tuple[str, ...], int]:
        if not isinstance(source.fit_data, PODFitData):
            raise TypeError("Gappy POD requires ReconstructionInput.fit_data=PODFitData")
        if not isinstance(source.target_observations, tuple) or any(
            not isinstance(value, PODObservation)
            for value in source.target_observations
        ):
            raise TypeError(
                "Gappy POD requires target_observations=tuple[PODObservation, ...]"
            )
        observations = source.target_observations
        if not observations:
            raise ValueError("Gappy POD requires at least one target observation")
        if len({value.target_id for value in observations}) != len(observations):
            raise ValueError("Gappy POD target IDs must be unique")
        rank = _setting_rank(settings)
        if source.pretrained_basis is None:
            if source.fit_data.snapshots is None:
                raise ValueError(
                    "Gappy POD needs complete calibration snapshots or a "
                    "pretrained PODBasis"
                )
            basis = fit_weighted_pod(
                source.fit_data.snapshots,
                rank=rank,
                quadrature_weights=source.fit_data.quadrature_weights,
            )
        else:
            if not isinstance(source.pretrained_basis, PODBasis):
                raise TypeError("pretrained_basis must be a validated PODBasis")
            if source.fit_data.snapshots is not None:
                raise ValueError(
                    "choose calibration snapshots or a pretrained POD basis, not both"
                )
            if (
                source.fit_data.quadrature_weights.shape
                != source.pretrained_basis.quadrature_weights.shape
                or not np.allclose(
                source.fit_data.quadrature_weights,
                source.pretrained_basis.quadrature_weights,
                rtol=1.0e-12,
                atol=1.0e-14,
                )
            ):
                raise ValueError(
                    "pretrained basis and fit data declare different "
                    "quadrature weights"
                )
            basis = source.pretrained_basis.truncated(rank)
        fields = _metadata_fields(source.metadata)
        return basis, observations, fields, rank

    def _output(
        self,
        *,
        basis: PODBasis,
        observations: Sequence[PODObservation],
        fields: tuple[str, ...],
        results: Sequence[GappyPODResult],
        basis_source: str,
        regularization: float | None,
        public_settings: Mapping[str, object],
    ) -> ReconstructionOutput:
        warnings = tuple(
            f"target_{target_index}: {warning}"
            for target_index, result in enumerate(results, start=1)
            for warning in result.warnings
        )
        completed = PODCompletedData(
            target_ids=tuple(value.target_id for value in observations),
            values=np.stack([value.completed for value in results]),
            coefficients=np.stack([value.coefficients for value in results]),
            basis=basis,
        )
        return ReconstructionOutput(
            method_id=self.method_id,
            completed_data=completed,
            fields=fields,
            case_counts={
                "calibration_snapshots": int(basis.calibration_count or 0),
                "targets": len(observations),
                "visible_observations": sum(len(value.values) for value in observations),
            },
            selected_rank=basis.rank,
            regularization=regularization,
            public_settings=dict(public_settings),
            diagnostics={
                "basis_source": basis_source,
                "quadrature_weight_min": float(np.min(basis.quadrature_weights)),
                "quadrature_weight_max": float(np.max(basis.quadrature_weights)),
                "targets": {
                    f"target_{target_index}": result.diagnostics()
                    for target_index, result in enumerate(results, start=1)
                },
            },
            warnings=warnings,
        )


class GappyPODLSProgram(_BaseGappyPODProgram):
    """Built-in classical Gappy POD least-squares ReconstructionProgram."""

    method_id = GAPPY_POD_LS
    info = ProgramInfo(
        GAPPY_POD_LS,
        "Gappy POD least squares",
        "Fixed-basis Gappy POD with explicit observation-noise whitening.",
    )

    def reconstruct(
        self,
        source: ReconstructionInput,
        settings: Mapping[str, object],
        seed: int,
        progress: ReconstructionProgress | None = None,
    ) -> ReconstructionOutput:
        _validate_program_settings(
            settings,
            {"rank", "max_condition", "rcond"},
        )
        _reconstruction_seed(seed)
        basis, observations, fields, _rank = self._inputs(source, settings)
        max_condition = _condition_limit(settings.get("max_condition", 1.0e12))
        rcond = _optional_float(settings.get("rcond"), "rcond")
        noise_covariances = tuple(
            _observation_noise(source, observation)
            for observation in observations
        )
        if progress is not None:
            progress("Fitted or loaded the fixed POD basis", 0.25)
        results = tuple(
            gappy_pod_ls(
                basis,
                observation.values,
                observation.operator,
                noise_covariance=noise_covariance,
                max_condition=max_condition,
                rcond=rcond,
            )
            for observation, noise_covariance in zip(
                observations,
                noise_covariances,
                strict=True,
            )
        )
        if progress is not None:
            progress("Reconstructed all visible targets", 1.0)
        return self._output(
            basis=basis,
            observations=observations,
            fields=fields,
            results=results,
            basis_source=(
                "pretrained_basis"
                if source.pretrained_basis is not None
                else "calibration_snapshots"
            ),
            regularization=None,
            public_settings={
                "solver": "whitened_least_squares",
                "max_condition": max_condition,
                "rcond": rcond,
                "noise_covariance_kinds": _covariance_kinds(noise_covariances),
            },
        )


class PODLinearGaussianMAPProgram(_BaseGappyPODProgram):
    """Built-in noise-aware linear-Gaussian POD MAP ReconstructionProgram."""

    method_id = POD_LINEAR_GAUSSIAN_MAP
    info = ProgramInfo(
        POD_LINEAR_GAUSSIAN_MAP,
        "POD linear-Gaussian MAP",
        "Fixed-basis POD reconstruction with explicit noise and coefficient priors.",
    )

    def setup_fields(self) -> tuple[FormField, ...]:
        return (
            *super().setup_fields(),
            FormField(
                "prior_precision",
                "Coefficient prior precision",
                "number",
                None,
                "Optional isotropic precision; otherwise modal variances stored "
                "with the basis are used.",
            ),
        )

    def reconstruct(
        self,
        source: ReconstructionInput,
        settings: Mapping[str, object],
        seed: int,
        progress: ReconstructionProgress | None = None,
    ) -> ReconstructionOutput:
        _validate_program_settings(
            settings,
            {
                "rank",
                "max_condition",
                "rcond",
                "prior_covariance",
                "prior_precision",
                "prior_mean",
                "minimum_data_precision_trace_fraction",
            },
        )
        _reconstruction_seed(seed)
        basis, observations, fields, _rank = self._inputs(source, settings)
        covariance = settings.get("prior_covariance")
        precision = settings.get("prior_precision")
        if covariance is not None and precision is not None:
            raise ValueError("choose prior_covariance or prior_precision, not both")
        max_condition = _condition_limit(settings.get("max_condition", 1.0e12))
        rcond = _optional_float(settings.get("rcond"), "rcond")
        minimum_trace_fraction = _fraction_threshold(
            settings.get("minimum_data_precision_trace_fraction", 0.1)
        )
        noise_covariances = tuple(
            _required_observation_noise(source, observation)
            for observation in observations
        )
        prior_source, prior_kind = _prior_provenance(
            basis,
            covariance=covariance,
            precision=precision,
        )
        if progress is not None:
            progress("Fitted or loaded the fixed POD basis", 0.25)
        results = tuple(
            pod_linear_gaussian_map(
                basis,
                observation.values,
                observation.operator,
                noise_covariance=noise_covariance,
                coefficient_prior_covariance=covariance,
                coefficient_prior_precision=precision,
                coefficient_prior_mean=settings.get("prior_mean"),
                max_condition=max_condition,
                rcond=rcond,
                minimum_data_precision_trace_fraction=minimum_trace_fraction,
            )
            for observation, noise_covariance in zip(
                observations,
                noise_covariances,
                strict=True,
            )
        )
        if progress is not None:
            progress("Reconstructed all visible targets", 1.0)
        regularization = (
            float(precision)
            if isinstance(precision, (int, float)) and not isinstance(precision, bool)
            else None
        )
        return self._output(
            basis=basis,
            observations=observations,
            fields=fields,
            results=results,
            basis_source=(
                "pretrained_basis"
                if source.pretrained_basis is not None
                else "calibration_snapshots"
            ),
            regularization=regularization,
            public_settings={
                "solver": "stacked_whitened_least_squares",
                "max_condition": max_condition,
                "rcond": rcond,
                "noise_covariance_kinds": _covariance_kinds(noise_covariances),
                "coefficient_prior_source": prior_source,
                "coefficient_prior_kind": prior_kind,
                "coefficient_prior_mean": (
                    "zero" if settings.get("prior_mean") is None else "recipe"
                ),
                "minimum_data_precision_trace_fraction": minimum_trace_fraction,
            },
        )


def _setting_rank(settings: Mapping[str, object]) -> int:
    raw = settings.get("rank", 8)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise TypeError("POD rank setting must be an integer")
    return _positive_rank(raw, None)


def _validate_program_settings(
    settings: object,
    allowed: set[str],
) -> None:
    if not isinstance(settings, Mapping):
        raise TypeError("POD method settings must be a mapping")
    if any(not isinstance(key, str) for key in settings):
        raise TypeError("POD method setting names must be strings")
    unknown = sorted(set(settings) - allowed)
    if unknown:
        raise ValueError("unknown POD method settings: " + ", ".join(unknown))


def _reconstruction_seed(seed: object) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("reconstruction seed must be an integer")
    if seed < 0:
        raise ValueError("reconstruction seed must be nonnegative")
    return seed


def _metadata_fields(metadata: Mapping[str, object]) -> tuple[str, ...]:
    raw = metadata.get("fields")
    if not isinstance(raw, (tuple, list)):
        raise TypeError("ReconstructionInput.metadata['fields'] must be a list or tuple")
    fields = tuple(raw)
    if not fields or any(not isinstance(value, str) or not value.strip() for value in fields):
        raise ValueError("ReconstructionInput.metadata['fields'] must be nonempty names")
    return fields


def _observation_noise(
    source: ReconstructionInput,
    observation: PODObservation,
) -> object | None:
    if observation.noise_covariance is not None:
        return observation.noise_covariance
    if isinstance(source.noise_model, PODNoiseModel):
        return source.noise_model.covariance_for(observation.target_id)
    if isinstance(source.noise_model, Mapping):
        if observation.target_id not in source.noise_model:
            raise ValueError(
                "mapping noise_model must contain every POD target ID; use "
                "PODNoiseModel(default_covariance=...) for a shared default"
            )
        covariance = source.noise_model[observation.target_id]
        if covariance is None:
            raise ValueError("mapping noise_model target covariance cannot be None")
        return covariance
    return source.noise_model


def _required_observation_noise(
    source: ReconstructionInput,
    observation: PODObservation,
) -> object:
    noise = _observation_noise(source, observation)
    if noise is None:
        raise ValueError(
            "POD linear-Gaussian MAP requires an explicit observation noise covariance"
        )
    return noise


def _fraction_threshold(value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError("minimum_data_precision_trace_fraction must be numeric")
    threshold = float(value)
    if not isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "minimum_data_precision_trace_fraction must lie in [0, 1]"
        )
    return threshold


def _covariance_kind(value: object | None) -> str:
    if value is None:
        return "identity_unweighted"
    array = _real_array(value, "covariance provenance")
    if array.ndim == 0:
        return "scalar"
    if array.ndim == 1:
        return "diagonal"
    if array.ndim == 2:
        return "full"
    raise ValueError("covariance must be scalar, diagonal, or full")


def _covariance_kinds(values: Sequence[object | None]) -> tuple[str, ...]:
    return tuple(sorted({_covariance_kind(value) for value in values}))


def _prior_provenance(
    basis: PODBasis,
    *,
    covariance: object | None,
    precision: object | None,
) -> tuple[str, str]:
    if covariance is not None:
        return "recipe_covariance", _covariance_kind(covariance)
    if precision is not None:
        return "recipe_precision", _covariance_kind(precision)
    if basis.modal_variances is None:
        raise ValueError(
            "POD MAP needs coefficient prior covariance/precision or "
            "modal variances stored with the basis"
        )
    return "basis_modal_variances", "diagonal"


def _optional_float(value: object, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _positive_rank(value: object, upper: int | None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("POD rank must be a positive integer")
    if upper is not None and value > upper:
        raise ValueError(f"POD rank {value} exceeds available rank {upper}")
    return int(value)


def _quadrature_weights(value: object | None, feature_count: int) -> np.ndarray:
    if value is None:
        return np.ones(feature_count, dtype=np.float64)
    weights = _finite_vector(value, "quadrature weights")
    if weights.size != feature_count:
        raise ValueError("quadrature weights must match the feature count")
    if np.any(weights <= 0.0):
        raise ValueError("quadrature weights must be strictly positive")
    return weights


def _observe_basis(
    basis: PODBasis,
    operator: object,
    observation_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(operator)
    if values.ndim == 1:
        if not np.issubdtype(values.dtype, np.integer):
            raise TypeError("one-dimensional observation operator must contain indices")
        indices = np.asarray(values, dtype=np.int64)
        if len(indices) != observation_count:
            raise ValueError("observation indices and values differ in length")
        if np.any(indices < 0) or np.any(indices >= basis.feature_count):
            raise ValueError("observation index lies outside the POD feature domain")
        return basis.mean[indices], basis.modes[indices]
    matrix = _finite_matrix(values, "observation operator")
    if matrix.shape != (observation_count, basis.feature_count):
        raise ValueError(
            "observation operator must have shape "
            f"({observation_count}, {basis.feature_count})"
        )
    return matrix @ basis.mean, matrix @ basis.modes


def _apply_observation_operator(field: np.ndarray, operator: object) -> np.ndarray:
    values = np.asarray(operator)
    if values.ndim == 1:
        return field[np.asarray(values, dtype=np.int64)]
    return np.asarray(values, dtype=np.float64) @ field


def _whiten_observations(
    design: np.ndarray,
    target: np.ndarray,
    covariance: object | None,
) -> tuple[np.ndarray, np.ndarray]:
    count = target.size
    if covariance is None:
        return np.asarray(design, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if isinstance(covariance, (bool, np.bool_)):
        raise TypeError("noise covariance must be numeric, not boolean")
    values = _real_array(covariance, "noise covariance")
    if not np.all(np.isfinite(values)):
        raise ValueError("noise covariance contains non-finite values")
    if values.ndim == 0:
        variance = float(values)
        if variance <= 0.0:
            raise ValueError("noise variance must be strictly positive")
        scale = np.sqrt(variance)
        return design / scale, target / scale
    if values.ndim == 1:
        if values.size != count:
            raise ValueError("noise variances must match the observation count")
        if np.any(values <= 0.0):
            raise ValueError("noise variances must be strictly positive")
        scale = np.sqrt(values)
        return design / scale[:, None], target / scale
    if values.ndim != 2 or values.shape != (count, count):
        raise ValueError(
            f"noise covariance must be scalar, length {count}, or shape ({count}, {count})"
        )
    if not np.allclose(values, values.T, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("noise covariance must be symmetric")
    try:
        factor = np.linalg.cholesky(values)
        return np.linalg.solve(factor, design), np.linalg.solve(factor, target)
    except np.linalg.LinAlgError as exc:
        raise ReconstructionNumericsError(
            "noise covariance must be positive definite"
        ) from exc


def _prior_whitener(
    basis: PODBasis,
    *,
    covariance: object | None,
    precision: object | None,
) -> np.ndarray:
    if covariance is not None and precision is not None:
        raise ValueError("choose coefficient prior covariance or precision, not both")
    if covariance is None and precision is None:
        if basis.modal_variances is None:
            raise ValueError(
                "POD MAP needs coefficient prior covariance/precision or "
                "modal variances stored with the basis"
            )
        covariance = basis.modal_variances
    if covariance is not None:
        return _covariance_whitener(covariance, basis.rank, "coefficient prior")
    return _precision_whitener(precision, basis.rank, "coefficient prior")


def _covariance_whitener(value: object, size: int, name: str) -> np.ndarray:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} covariance must be numeric, not boolean")
    covariance = _real_array(value, f"{name} covariance")
    if not np.all(np.isfinite(covariance)):
        raise ValueError(f"{name} covariance contains non-finite values")
    if covariance.ndim == 0:
        variance = float(covariance)
        if variance <= 0.0:
            raise ValueError(f"{name} covariance must be positive definite")
        return np.eye(size, dtype=np.float64) / np.sqrt(variance)
    if covariance.ndim == 1:
        if covariance.size != size or np.any(covariance <= 0.0):
            raise ValueError(
                f"{name} diagonal covariance must have {size} positive entries"
            )
        return np.diag(1.0 / np.sqrt(covariance))
    if covariance.ndim != 2 or covariance.shape != (size, size):
        raise ValueError(f"{name} covariance must have shape ({size}, {size})")
    if not np.allclose(covariance, covariance.T, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError(f"{name} covariance must be symmetric")
    try:
        factor = np.linalg.cholesky(covariance)
        return np.linalg.solve(factor, np.eye(size, dtype=np.float64))
    except np.linalg.LinAlgError as exc:
        raise ReconstructionNumericsError(
            f"{name} covariance must be positive definite"
        ) from exc


def _precision_whitener(value: object, size: int, name: str) -> np.ndarray:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} precision must be numeric, not boolean")
    precision = _real_array(value, f"{name} precision")
    if not np.all(np.isfinite(precision)):
        raise ValueError(f"{name} precision contains non-finite values")
    if precision.ndim == 0:
        scalar = float(precision)
        if scalar <= 0.0:
            raise ValueError(f"{name} precision must be positive definite")
        return np.sqrt(scalar) * np.eye(size, dtype=np.float64)
    if precision.ndim == 1:
        if precision.size != size or np.any(precision <= 0.0):
            raise ValueError(
                f"{name} diagonal precision must have {size} positive entries"
            )
        return np.diag(np.sqrt(precision))
    if precision.ndim != 2 or precision.shape != (size, size):
        raise ValueError(f"{name} precision must have shape ({size}, {size})")
    if not np.allclose(precision, precision.T, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError(f"{name} precision must be symmetric")
    try:
        factor = np.linalg.cholesky(precision)
        return factor.T
    except np.linalg.LinAlgError as exc:
        raise ReconstructionNumericsError(
            f"{name} precision must be positive definite"
        ) from exc


def _least_squares(design: np.ndarray, target: np.ndarray, rcond: float | None) -> np.ndarray:
    try:
        solution, _residuals, _rank, _singular_values = np.linalg.lstsq(
            design,
            target,
            rcond=rcond,
        )
    except np.linalg.LinAlgError as exc:
        raise ReconstructionNumericsError("least-squares solve did not converge") from exc
    if not np.all(np.isfinite(solution)):
        raise ReconstructionNumericsError("least-squares solve returned non-finite values")
    return np.asarray(solution, dtype=np.float64)


def _singular_values(matrix: np.ndarray, name: str) -> np.ndarray:
    try:
        values = np.linalg.svd(matrix, compute_uv=False)
    except np.linalg.LinAlgError as exc:
        raise ReconstructionNumericsError(f"{name} SVD did not converge") from exc
    if not np.all(np.isfinite(values)):
        raise ReconstructionNumericsError(f"{name} has non-finite singular values")
    return values


def _svd_rank_and_condition(
    singular_values: np.ndarray,
    shape: tuple[int, ...],
    rcond: float | None,
) -> tuple[int, float]:
    if singular_values.size == 0 or singular_values[0] <= 0.0:
        return 0, float("inf")
    if rcond is None:
        tolerance = (
            np.finfo(np.float64).eps * max(shape) * float(singular_values[0])
        )
    else:
        if isinstance(rcond, (bool, np.bool_)) or not isinstance(
            rcond,
            (int, float, np.integer, np.floating),
        ):
            raise TypeError("rcond must be numeric")
        cutoff = float(rcond)
        if not isfinite(cutoff) or cutoff < 0.0:
            raise ValueError("rcond must be finite and nonnegative")
        tolerance = cutoff * float(singular_values[0])
    rank = int(np.count_nonzero(singular_values > tolerance))
    if rank < min(shape):
        condition = float("inf")
    else:
        condition = float(singular_values[0] / singular_values[-1])
    return rank, condition


def _condition_limit(value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError("max_condition must be numeric")
    limit = float(value)
    if not isfinite(limit) or limit < 1.0:
        raise ValueError("max_condition must be finite and at least one")
    return limit


def _finite_vector(value: object, name: str) -> np.ndarray:
    result = _real_array(value, name)
    if result.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return np.array(result, dtype=np.float64, copy=True)


def _finite_matrix(value: object, name: str) -> np.ndarray:
    result = _real_array(value, name)
    if result.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return np.array(result, dtype=np.float64, copy=True)


def _real_array(value: object, name: str) -> np.ndarray:
    if _contains_boolean(value):
        raise TypeError(f"{name} must be numerical, not boolean")
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise TypeError(
            f"{name} must be real-valued; the built-in POD methods do not "
            "silently discard complex components"
        )
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain real numerical values") from exc


def _contains_boolean(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.bool_):
            return True
        if value.dtype == object:
            return any(_contains_boolean(item) for item in value.flat)
        return False
    if isinstance(value, (tuple, list)):
        return any(_contains_boolean(item) for item in value)
    return False


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result
