"""Fail-closed equation contract for the PINC/GyroSwin GKW corpus.

The public contract is coefficient-first. A solver receives all normalized
grids, quadrature weights, geometry tensors, topology maps, species scalars,
and numerical dissipation controls needed to evaluate the declared GKW
branch. It never has to infer a normalization convention from a hidden neural
representation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np


PINNED_NEURAL_GYROKINETICS_COMMIT = (
    "568411b33be486262b8c5cfe1dfb3e0fe97f6419"
)
PINNED_GYARADAX_AUDIT_COMMIT = (
    "318e3d11d7326d99e96f2bd5a8998a462261dcd4"
)


@dataclass(frozen=True)
class GKWGridContract:
    """Discrete operator view used by the source and field oracles."""

    nvpar: int = 32
    nmu: int = 8
    ns: int = 16
    nkx: int = 85
    nky: int = 32
    nperiod: int = 1
    vpar_max: float = 3.0

    def __post_init__(self) -> None:
        counts = (self.nvpar, self.nmu, self.ns, self.nkx, self.nky)
        if min(counts) <= 0 or self.nperiod <= 0 or self.vpar_max <= 0.0:
            raise ValueError("GKW grid counts and extents must be positive")
        if self.nkx % 2 != 1:
            raise ValueError("the pinned GKW mode box uses an odd kx count")

    @property
    def state_shape(self) -> tuple[int, int, int, int, int]:
        return (self.nvpar, self.nmu, self.ns, self.nkx, self.nky)


@dataclass(frozen=True)
class GKWRegimeContract:
    """Physics and numerical branch shared by all 301 archived inputs."""

    dataset_id: str = "pinc_gyroswin_gkw_301_collisionless_es_adiabatic_ion"
    operator_id: str = "gkw_delta_f_es_adiabatic_ion_answer_first_v1"
    state_variable: str = "delta_f_i"
    api_state_alias: str = "h"
    geometry_model: str = "circ"
    parallel_boundary: str = "open_twist_and_shift_mode_topology"
    perpendicular_boundary: str = "flux_tube_fourier_mode_box"
    velocity_boundary: str = "truncated_fast_decay_zero_ghost_stencil"
    nonlinear: bool = True
    electrostatic: bool = True
    adiabatic_electrons: bool = True
    zonal_adiabatic: bool = True
    collisions: bool = False
    kinetic_species: int = 1
    beta: float = 0.0
    nlapar: bool = False
    nlbpar: bool = False
    vcor: float = 0.0
    uprim: float = 0.0
    eps: float = 0.19
    order_of_scheme: str = "fourth_order"
    time_method: str = "EXP_meth2"
    dtim: float = 0.01
    disp_par: float = 1.0
    disp_vp: float = 0.2
    disp_x: float = 0.1
    disp_y: float = 0.1

    def __post_init__(self) -> None:
        numeric = (
            self.beta,
            self.vcor,
            self.uprim,
            self.eps,
            self.dtim,
            self.disp_par,
            self.disp_vp,
            self.disp_x,
            self.disp_y,
        )
        if not all(np.isfinite(value) for value in numeric):
            raise ValueError("GKW regime scalars must be finite")
        if self.kinetic_species != 1:
            raise ValueError("the pinned corpus evolves one kinetic ion species")
        if not self.nonlinear or not self.electrostatic:
            raise ValueError("the pinned corpus is nonlinear and electrostatic")
        if not self.adiabatic_electrons or not self.zonal_adiabatic:
            raise ValueError("the pinned corpus uses zonal adiabatic electrons")
        if self.collisions or self.beta != 0.0 or self.nlapar or self.nlbpar:
            raise ValueError("electromagnetic or collisional controls are out of scope")
        if self.vcor != 0.0 or self.uprim != 0.0:
            raise ValueError("the pinned corpus has no equilibrium rotation")


@dataclass(frozen=True)
class GKWPublicContractAudit:
    passed: bool
    failures: tuple[str, ...]
    checked_arrays: tuple[str, ...]
    checked_scalars: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class GKWEquationContract:
    """Dataset-exact public equation declaration and schema validator.

    The equation is d_t delta_f_i = R_GKW(delta_f_i, phi; C) + S_balance,
    together with adiabatic-electron quasineutrality
    Q_GKW(phi; C) = M_GKW(delta_f_i; C).

    C is the public coefficient payload validated here. The answer-first
    source is S_balance = d_t delta_f_i - R_GKW and may be physically
    substantial.
    """

    _VECTOR_SHAPES = {
        "vpar": "nvpar",
        "intvp": "nvpar",
        "mu": "nmu",
        "intmu": "nmu",
        "s": "ns",
        "ints": "ns",
        "B": "ns",
        "F": "ns",
        "G": "ns",
        "E_psi_zeta": "ns",
        "R": "ns",
        "B_t_fraction": "ns",
        "kx": "nkx",
        "ky": "nky",
        "parseval_ky": "nky",
    }
    _MATRIX_SHAPES = {
        "D": ("ns", 3),
        "metric_perp": ("ns", 3),
        "mode_label": ("nkx", "nky"),
        "ixplus": ("nkx", "nky"),
        "ixminus": ("nkx", "nky"),
        "pos_parallel_class": ("ns", "nkx", "nky"),
    }
    _TOPOLOGY_SHAPES = {
        "s_shift": (9, "ns", "nkx", "nky"),
        "kx_shift": (9, "ns", "nkx", "nky"),
        "valid_shift": (9, "ns", "nkx", "nky"),
    }
    _REQUIRED_SCALARS = (
        "q",
        "shat",
        "eps",
        "rlt_i",
        "rln_i",
        "ikxspace",
        "krhomax",
        "kthnorm",
        "dvp",
        "ds",
        "idisp",
        "dgrid",
        "mass_i",
        "charge_i",
        "temperature_i",
        "density_i",
        "vth_ratio_i",
        "d2X",
        "sign_B",
        "ixzero",
        "iyzero",
        "disp_par",
        "disp_vp",
        "disp_x",
        "disp_y",
    )

    def __init__(
        self,
        grid: GKWGridContract | None = None,
        regime: GKWRegimeContract | None = None,
    ) -> None:
        self.grid = grid or GKWGridContract()
        self.regime = regime or GKWRegimeContract()

    @property
    def active_terms(self) -> tuple[str, ...]:
        return (
            "I_parallel_streaming",
            "II_magnetic_curvature_gradB_drift",
            "III_nonlinear_gyroaveraged_ExB_bracket",
            "IV_mirror_trapping_vpar_advection",
            "V_density_temperature_gradient_drive",
            "VII_parallel_electrostatic_field_drive",
            "VIII_magnetic_drift_electrostatic_field_drive",
            "parallel_fourth_order_numerical_dissipation",
            "vpar_fourth_order_numerical_dissipation",
            "kx_ky_hyper_dissipation",
        )

    @property
    def closed_terms(self) -> tuple[str, ...]:
        return (
            "collisions",
            "kinetic_electron_evolution",
            "A_parallel",
            "delta_B_parallel",
            "equilibrium_rotation_and_flow_shear",
            "neoclassical_background_correction",
        )

    def public_metadata(self) -> dict[str, Any]:
        required_arrays = (
            set(self._VECTOR_SHAPES)
            | set(self._MATRIX_SHAPES)
            | set(self._TOPOLOGY_SHAPES)
        )
        return {
            "schema": "fusion_gkw_public_equation_contract_v1",
            "dataset_id": self.regime.dataset_id,
            "operator_id": self.regime.operator_id,
            "equation": (
                "d_t delta_f_i = R_GKW[delta_f_i, phi; "
                "public_coefficients] + S_balance"
            ),
            "field_closure": (
                "GKW electrostatic gyrokinetic quasineutrality with "
                "zonal adiabatic-electron correction"
            ),
            "state_variable": self.regime.state_variable,
            "api_state_alias": self.regime.api_state_alias,
            "api_alias_definition": "h == delta_f_i; A_parallel is disabled",
            "state_is_complex_half_ky_spectrum": True,
            "continuous_hidden_state_is_primary": True,
            "operator_grid_is_an_oracle_quadrature_view": True,
            "grid": asdict(self.grid),
            "semidiscrete_problem": {
                "class": "continuous_time_method_of_lines",
                "state_axis_order": [
                    "v_parallel",
                    "mu",
                    "s",
                    "kx",
                    "ky_nonnegative_half_spectrum",
                ],
                "state_shape": list(self.grid.state_shape),
                "field_axis_order": ["s", "kx", "ky_nonnegative_half_spectrum"],
                "phase_operator": (
                    "fixed public GKW quadrature, finite-difference, "
                    "twist-and-shift, and dealiased Fourier operator"
                ),
                "exactness_statement": (
                    "the hidden continuous-time state satisfies this declared "
                    "float64 semidiscrete equation with the numerical source "
                    "oracle, up to float64 evaluation roundoff"
                ),
                "truth_precision": "float64",
                "float32_status": (
                    "diagnostic throughput mode only; not the truth contract"
                ),
            },
            "public_solver_reference": {
                "reference_is_part_of_public_question": True,
                "modules": [
                    "gkw_equation_contract.py",
                    "gkw_public_packet.py",
                    "gkw_torch_precompute.py",
                    "gkw_torch_operator.py",
                ],
                "coefficient_rule": (
                    "all derived operator coefficients are deterministic "
                    "functions of required_public_arrays and "
                    "required_public_scalars"
                ),
                "hidden_representation_included": False,
                "generator_parameters_included": False,
            },
            "regime": asdict(self.regime),
            "active_terms": list(self.active_terms),
            "closed_terms": list(self.closed_terms),
            "source_policy": {
                "origin": "generated_solution_residual",
                "definition": "S_balance = d_t delta_f_i - R_GKW",
                "absolute_norm_must_be_small": False,
                "physical_admissibility_required": True,
                "public_access": "metered numerical oracle",
                "neural_parameterization_public": False,
            },
            "array_conventions": {
                "E_psi_zeta": (
                    "signed GKW efun coefficient, equal to -E^{psi,zeta} "
                    "in the pinned geometry assembly"
                ),
                "D": "GKW drift tensor columns used directly by drift_x/y",
                "metric_perp": (
                    "ordered as (g_yy, g_xy, g_xx) in k_perp^2"
                ),
                "ky": "nonnegative real-to-complex half spectrum",
                "parseval_ky": (
                    "physical half-spectrum weights: 1 for ky=0 and 2 "
                    "for every retained ky>0 mode"
                ),
                "kx": "centered signed spectrum",
            },
            "required_public_scalars": list(self._REQUIRED_SCALARS),
            "required_public_arrays": sorted(required_arrays),
            "provenance": {
                "neural_gyrokinetics_commit": (
                    PINNED_NEURAL_GYROKINETICS_COMMIT
                ),
                "gyaradax_audit_commit": PINNED_GYARADAX_AUDIT_COMMIT,
                "gkw_reference_doi": "10.1016/j.cpc.2009.07.001",
            },
        }

    def audit_public_coefficients(
        self,
        arrays: Mapping[str, Any],
        scalars: Mapping[str, Any],
    ) -> GKWPublicContractAudit:
        failures: list[str] = []
        checked_arrays: list[str] = []
        checked_scalars: list[str] = []
        dimensions = asdict(self.grid)

        for name, axis in self._VECTOR_SHAPES.items():
            expected = (int(dimensions[axis]),)
            self._check_array(name, arrays, expected, failures)
            checked_arrays.append(name)
        for name, shape_spec in self._MATRIX_SHAPES.items():
            expected = tuple(
                int(dimensions[value]) if isinstance(value, str) else value
                for value in shape_spec
            )
            self._check_array(name, arrays, expected, failures)
            checked_arrays.append(name)
        for name, shape_spec in self._TOPOLOGY_SHAPES.items():
            expected = tuple(
                int(dimensions[value]) if isinstance(value, str) else value
                for value in shape_spec
            )
            self._check_array(name, arrays, expected, failures)
            checked_arrays.append(name)

        for name in self._REQUIRED_SCALARS:
            if name not in scalars:
                failures.append(f"missing_scalar:{name}")
                continue
            value = np.asarray(scalars[name])
            if value.size != 1 or not np.isfinite(value).all():
                failures.append(f"invalid_scalar:{name}")
            checked_scalars.append(name)

        self._check_regime_scalars(scalars, failures)
        self._check_half_spectrum_weights(arrays, failures)
        return GKWPublicContractAudit(
            passed=not failures,
            failures=tuple(failures),
            checked_arrays=tuple(checked_arrays),
            checked_scalars=tuple(checked_scalars),
        )

    @staticmethod
    def _check_half_spectrum_weights(
        arrays: Mapping[str, Any],
        failures: list[str],
    ) -> None:
        if "ky" not in arrays or "parseval_ky" not in arrays:
            return
        ky = np.asarray(arrays["ky"], dtype=np.float64)
        weights = np.asarray(arrays["parseval_ky"], dtype=np.float64)
        if ky.shape != weights.shape or ky.ndim != 1:
            return
        if np.any(ky < -1.0e-12):
            failures.append("half_spectrum_has_negative_ky")
            return
        expected = np.where(np.abs(ky) < 1.0e-12, 1.0, 2.0)
        if not np.array_equal(weights, expected):
            failures.append("invalid_physical_parseval_ky_weights")

    @staticmethod
    def _check_array(
        name: str,
        arrays: Mapping[str, Any],
        expected_shape: tuple[int, ...],
        failures: list[str],
    ) -> None:
        if name not in arrays:
            failures.append(f"missing_array:{name}")
            return
        value = np.asarray(arrays[name])
        if value.shape != expected_shape:
            failures.append(
                f"shape:{name}:expected={expected_shape}:actual={value.shape}"
            )
            return
        if value.size == 0 or not np.isfinite(value).all():
            failures.append(f"nonfinite_array:{name}")

    def _check_regime_scalars(
        self,
        scalars: Mapping[str, Any],
        failures: list[str],
    ) -> None:
        expected = {
            "eps": self.regime.eps,
            "mass_i": 1.0,
            "charge_i": 1.0,
            "temperature_i": 1.0,
            "density_i": 1.0,
            "vth_ratio_i": 1.0,
            "idisp": 2.0,
            "dgrid": 1.0,
            "disp_par": self.regime.disp_par,
            "disp_vp": self.regime.disp_vp,
            "disp_x": self.regime.disp_x,
            "disp_y": self.regime.disp_y,
        }
        for name, target in expected.items():
            if name not in scalars:
                continue
            value = np.asarray(scalars[name]).reshape(-1)
            if value.size == 1 and not np.isclose(
                float(value[0]),
                float(target),
                rtol=1.0e-10,
                atol=1.0e-12,
            ):
                failures.append(
                    f"regime_mismatch:{name}:expected={target}:"
                    f"actual={value[0]}"
                )


def dataset_exact_gkw_contract() -> GKWEquationContract:
    """Return the immutable contract inferred from all 301 archived inputs."""

    return GKWEquationContract()
