"""Differentiable PyTorch operator for the pinned electrostatic GKW branch.

This module re-expresses the documented GKW term formulas using a public
precomputed-coefficient payload. It is not declared production-ready until a
term-by-term comparison against the pinned GKW-compatible reference passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from .gkw_equation_contract import GKWEquationContract


_VPAR_D1 = (1.0, -8.0, 0.0, 8.0, -1.0)
_VPAR_D4 = (-1.0, 4.0, -6.0, 4.0, -1.0)
_VPAR_SCALE = 1.0 / 12.0

# GKW's fourth-order parallel stencils. Rows are open-boundary position
# classes (-2, -1, 0, 1, 2); columns are shifts (-4, ..., 4).
_PARALLEL_D1_POS = (
    (0.0, 0.0, 0.0, 0.0, -18.0, 24.0, -6.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, -4.0, -6.0, 12.0, -2.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, -8.0, 0.0, 8.0, -1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, -8.0, 0.0, 8.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, -6.0, 0.0, 0.0, 0.0, 0.0, 0.0),
)
_PARALLEL_D1_NEG = (
    (0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, -8.0, 0.0, 8.0, -1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, -8.0, 0.0, 8.0, -1.0, 0.0, 0.0),
    (0.0, 0.0, 2.0, -12.0, 6.0, 4.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 6.0, -24.0, 18.0, 0.0, 0.0, 0.0, 0.0),
)
_PARALLEL_D4_POS = (
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, -1.0, 4.0, -6.0, 4.0, -1.0, 0.0, 0.0),
    (0.0, 0.0, -1.0, 4.0, -6.0, 4.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 12.0, -24.0, 0.0, 0.0, 0.0, 0.0),
)
_PARALLEL_D4_NEG = (
    (0.0, 0.0, 0.0, 0.0, -24.0, 12.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 4.0, -6.0, 4.0, -1.0, 0.0, 0.0),
    (0.0, 0.0, -1.0, 4.0, -6.0, 4.0, -1.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
)


@dataclass
class GKWSpectralOperatorEvaluation:
    rhs: torch.Tensor
    phi: torch.Tensor
    terms: dict[str, torch.Tensor]

    def __post_init__(self) -> None:
        if not self.rhs.is_complex() or not self.phi.is_complex():
            raise TypeError("GKW RHS and electrostatic potential must be complex")
        if not torch.isfinite(torch.view_as_real(self.phi)).all():
            raise FloatingPointError("GKW electrostatic potential is non-finite")
        for name, value in self.terms.items():
            if value.shape != self.rhs.shape:
                raise ValueError(f"GKW term {name} has an invalid shape")
            if not torch.isfinite(torch.view_as_real(value)).all():
                raise FloatingPointError(f"GKW term {name} is non-finite")
        if not torch.isfinite(torch.view_as_real(self.rhs)).all():
            raise FloatingPointError("GKW RHS sum is non-finite")


class GKWTorchSpectralOperator:
    """Dataset-aligned nonlinear electrostatic adiabatic-ion GKW RHS."""

    _BASE_ARRAY_KEYS = (
        "bessel",
        "phi_weight",
        "phi_corr_weight",
        "phi_gamma",
        "fmaxwl",
        "dmaxwel_fm_ek",
        "drift_x",
        "drift_y",
        "utrap",
        "abs_dum2_vp",
        "hyper",
        "kx",
        "ky",
        "nl_dum_s",
        "nl_jind",
        "s_shift",
        "kx_shift",
        "valid_shift",
    )
    _FUSED_PARALLEL_ARRAY_KEYS = (
        "s_total_upar",
        "s_total_t7",
    )
    _COMPACT_PARALLEL_ARRAY_KEYS = (
        "upar",
        "abs_dum2_par",
        "term7_fac",
        "pos_parallel_class",
    )
    _BASE_SCALAR_KEYS = (
        "dvp",
        "disp_vp",
        "phi_tmp",
        "phi_de",
        "phi_signz",
        "phi_has_zonal",
        "phi_ixzero",
        "phi_iyzero",
        "ixzero",
        "iyzero",
        "nl_mrad",
        "nl_mphi",
        "nl_mphiw3",
        "nl_fft_scale",
    )
    _COMPACT_PARALLEL_SCALAR_KEYS = (
        "disp_par",
        "sgr_dist",
    )

    def __init__(
        self,
        coefficients: Mapping[str, Any],
        *,
        contract: GKWEquationContract | None = None,
        reference_id: str,
    ) -> None:
        if not reference_id:
            raise ValueError("GKW operator coefficients need provenance")
        provided = set(coefficients)
        fused = set(self._FUSED_PARALLEL_ARRAY_KEYS)
        compact = set(self._COMPACT_PARALLEL_ARRAY_KEYS) | set(
            self._COMPACT_PARALLEL_SCALAR_KEYS
        )
        has_fused = fused <= provided
        has_compact = compact <= provided
        if has_fused == has_compact:
            raise ValueError(
                "GKW coefficients must provide exactly one parallel stencil "
                "mode: fused or compact"
            )
        self.parallel_coefficient_mode = "fused" if has_fused else "compact"
        array_keys = list(self._BASE_ARRAY_KEYS)
        scalar_keys = list(self._BASE_SCALAR_KEYS)
        if has_fused:
            array_keys.extend(self._FUSED_PARALLEL_ARRAY_KEYS)
        else:
            array_keys.extend(self._COMPACT_PARALLEL_ARRAY_KEYS)
            scalar_keys.extend(self._COMPACT_PARALLEL_SCALAR_KEYS)
        missing = sorted((set(array_keys) | set(scalar_keys)) - provided)
        if missing:
            raise KeyError(f"GKW operator coefficients are incomplete: {missing}")
        self.contract = contract or GKWEquationContract()
        self.reference_id = reference_id
        self._array_keys = tuple(array_keys)
        self._scalar_keys = tuple(scalar_keys)
        self._arrays = {
            name: self._constant_tensor(coefficients[name])
            for name in self._array_keys
        }
        self._scalars = {
            name: self._scalar(coefficients[name], name)
            for name in self._scalar_keys
        }
        self._validate_static_schema()

    @staticmethod
    def _scalar(value: Any, name: str) -> float | int:
        if isinstance(value, torch.Tensor):
            if value.requires_grad:
                raise ValueError(
                    f"GKW scalar coefficient {name} must be static"
                )
            array = value.detach().cpu().numpy()
        else:
            array = np.asarray(value)
        if array.size != 1 or not np.isfinite(array).all():
            raise ValueError(f"GKW scalar coefficient {name} is invalid")
        scalar = array.reshape(-1)[0].item()
        if name in {
            "phi_ixzero",
            "phi_iyzero",
            "ixzero",
            "iyzero",
            "nl_mrad",
            "nl_mphi",
            "nl_mphiw3",
        }:
            return int(scalar)
        return float(scalar)

    @staticmethod
    def _constant_tensor(value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            if value.requires_grad:
                raise ValueError("GKW operator coefficients must be static")
            return value.detach().clone()
        return torch.as_tensor(np.asarray(value).copy())

    def _validate_static_schema(self) -> None:
        grid = self.contract.grid
        for name, value in self._arrays.items():
            if (
                (value.is_floating_point() or value.is_complex())
                and not torch.isfinite(value).all()
            ):
                raise ValueError(
                    f"GKW coefficient {name} contains non-finite values"
                )
        state_shape = grid.state_shape
        broadcast_keys = (
            "bessel",
            "phi_weight",
            "fmaxwl",
            "dmaxwel_fm_ek",
            "drift_x",
            "drift_y",
            "utrap",
            "abs_dum2_vp",
            "hyper",
        )
        for name in broadcast_keys:
            try:
                torch.broadcast_shapes(
                    tuple(self._arrays[name].shape),
                    state_shape,
                )
            except RuntimeError as exc:
                raise ValueError(
                    f"GKW coefficient {name} cannot broadcast to {state_shape}"
                ) from exc
        if self.parallel_coefficient_mode == "fused":
            for name in self._FUSED_PARALLEL_ARRAY_KEYS:
                expected = (9, *state_shape)
                try:
                    torch.broadcast_shapes(
                        tuple(self._arrays[name].shape),
                        expected,
                    )
                except RuntimeError as exc:
                    raise ValueError(
                        f"GKW stencil coefficient {name} cannot broadcast to "
                        f"{expected}"
                    ) from exc
        else:
            for name in ("upar", "abs_dum2_par", "term7_fac"):
                try:
                    torch.broadcast_shapes(
                        tuple(self._arrays[name].shape),
                        state_shape,
                    )
                except RuntimeError as exc:
                    raise ValueError(
                        f"GKW compact coefficient {name} cannot broadcast to "
                        f"{state_shape}"
                    ) from exc
            position = self._arrays["pos_parallel_class"]
            expected_position = (grid.ns, grid.nkx, grid.nky)
            if tuple(position.shape) != expected_position:
                raise ValueError(
                    "GKW parallel position classes must have shape "
                    f"{expected_position}"
                )
            if position.is_floating_point() or position.is_complex():
                raise TypeError("GKW parallel position classes must be integers")
            if position.numel() and (
                int(position.min()) < -2 or int(position.max()) > 2
            ):
                raise ValueError("GKW parallel position classes must lie in [-2, 2]")
            if float(self._scalars["sgr_dist"]) <= 0.0:
                raise ValueError("GKW parallel grid spacing must be positive")

        topology_shape = (9, grid.ns, grid.nkx, grid.nky)
        for name in ("s_shift", "kx_shift", "valid_shift"):
            if tuple(self._arrays[name].shape) != topology_shape:
                raise ValueError(
                    f"GKW topology {name} must have shape {topology_shape}"
                )
        if tuple(self._arrays["kx"].shape) != (grid.nkx,):
            raise ValueError("GKW kx coefficient has an invalid shape")
        if tuple(self._arrays["ky"].shape) != (grid.nky,):
            raise ValueError("GKW ky coefficient has an invalid shape")
        if tuple(self._arrays["nl_dum_s"].shape) != (grid.ns,):
            raise ValueError("GKW nonlinear geometry factor has an invalid shape")
        if tuple(self._arrays["nl_jind"].shape) != (grid.nkx,):
            raise ValueError("GKW nonlinear kx packing map has an invalid shape")
        if int(self._scalars["nl_mphiw3"]) != (
            int(self._scalars["nl_mphi"]) // 2 + 1
        ):
            raise ValueError("GKW nonlinear real-FFT dimensions are inconsistent")
        for name in ("ixzero", "phi_ixzero"):
            if not 0 <= int(self._scalars[name]) < grid.nkx:
                raise ValueError(f"GKW scalar {name} lies outside kx")
        for name in ("iyzero", "phi_iyzero"):
            if not 0 <= int(self._scalars[name]) < grid.nky:
                raise ValueError(f"GKW scalar {name} lies outside ky")

    @property
    def device(self) -> torch.device:
        return next(iter(self._arrays.values())).device

    def public_metadata(self) -> dict[str, Any]:
        metadata = self.contract.public_metadata()
        metadata["implementation"] = {
            "backend": "pytorch_complex_spectral",
            "reference_id": self.reference_id,
            "term_level_reference_parity_required": True,
            "production_eligible": False,
            "parallel_coefficient_mode": self.parallel_coefficient_mode,
            "public_precomputed_coefficients": list(self._array_keys),
            "public_precomputed_scalars": list(self._scalar_keys),
        }
        return metadata

    def evaluate(
        self,
        delta_f_i: torch.Tensor,
        phi: torch.Tensor | None = None,
    ) -> GKWSpectralOperatorEvaluation:
        self._validate_state(delta_f_i)
        return self._evaluate_impl(delta_f_i, phi)

    def evaluate_batch(
        self,
        delta_f_i: torch.Tensor,
        phi: torch.Tensor | None = None,
    ) -> GKWSpectralOperatorEvaluation:
        """Evaluate independent leading batch dimensions algebraically."""

        self._validate_batched_state(delta_f_i)
        return self._evaluate_impl(delta_f_i, phi)

    def _evaluate_impl(
        self,
        delta_f_i: torch.Tensor,
        phi: torch.Tensor | None,
    ) -> GKWSpectralOperatorEvaluation:
        if phi is None:
            phi = self._solve_phi_impl(delta_f_i)
        else:
            expected = (
                *delta_f_i.shape[:-5],
                self.contract.grid.ns,
                self.contract.grid.nkx,
                self.contract.grid.nky,
            )
            if tuple(phi.shape) != expected or not phi.is_complex():
                raise ValueError(f"GKW phi must be complex with shape {expected}")
            phi = phi.to(device=delta_f_i.device, dtype=delta_f_i.dtype)

        coefficients = {
            name: self._coefficient(name, delta_f_i)
            for name in self._array_keys
        }
        gyro_phi = coefficients["bessel"] * phi.unsqueeze(-4).unsqueeze(-4)

        if self.parallel_coefficient_mode == "fused":
            parallel = self._apply_parallel(
                delta_f_i,
                coefficients["s_total_upar"],
                coefficients,
            )
            landau = self._apply_parallel(
                gyro_phi,
                coefficients["s_total_t7"],
                coefficients,
            )
        else:
            parallel = self._apply_parallel_compact(
                delta_f_i,
                coefficients["upar"],
                coefficients,
                dissipation=coefficients["abs_dum2_par"],
                term7=False,
            )
            landau = self._apply_parallel_compact(
                gyro_phi,
                coefficients["term7_fac"],
                coefficients,
                dissipation=None,
                term7=True,
            )
        velocity_d1, velocity_d4 = self._apply_vpar_dual(delta_f_i)

        kx = coefficients["kx"].reshape(1, 1, 1, -1, 1)
        ky = coefficients["ky"].reshape(1, 1, 1, 1, -1)
        k_dot_drift = (
            coefficients["drift_x"] * kx
            + coefficients["drift_y"] * ky
        )
        drift_df = -1j * k_dot_drift * delta_f_i
        trapping = (
            coefficients["utrap"]
            * velocity_d1
            / float(self._scalars["dvp"])
        )
        vpar_dissipation = (
            float(self._scalars["disp_vp"])
            * coefficients["abs_dum2_vp"]
            * velocity_d4
            / float(self._scalars["dvp"])
        )
        gradient_drive = (
            1j * coefficients["dmaxwel_fm_ek"] * gyro_phi
        )
        curvature_field_drive = (
            -1j
            * float(self._scalars["phi_signz"])
            * k_dot_drift
            * (
                coefficients["fmaxwl"]
                / max(float(self._scalars["phi_tmp"]), 1.0e-15)
            )
            * gyro_phi
        )
        nonlinear = self._nonlinear_term(
            delta_f_i,
            phi,
            coefficients,
        )
        hyper_dissipation = coefficients["hyper"] * delta_f_i

        terms = {
            "I_parallel_streaming_plus_dissipation": parallel,
            "II_magnetic_drift_delta_f": drift_df,
            "III_nonlinear_ExB": nonlinear,
            "IV_mirror_trapping": trapping,
            "IV_vpar_dissipation": vpar_dissipation,
            "V_equilibrium_gradient_drive": gradient_drive,
            "VII_parallel_field_drive": landau,
            "VIII_magnetic_drift_field_drive": curvature_field_drive,
            "kx_ky_hyper_dissipation": hyper_dissipation,
        }
        rhs = torch.zeros_like(delta_f_i)
        for term in terms.values():
            rhs = rhs + term
        return GKWSpectralOperatorEvaluation(rhs=rhs, phi=phi, terms=terms)

    def solve_phi(self, delta_f_i: torch.Tensor) -> torch.Tensor:
        """Apply GKW adiabatic-electron quasineutrality."""

        self._validate_state(delta_f_i)
        return self._solve_phi_impl(delta_f_i)

    def solve_phi_batch(self, delta_f_i: torch.Tensor) -> torch.Tensor:
        """Apply quasineutrality to independent leading batch dimensions."""

        self._validate_batched_state(delta_f_i)
        return self._solve_phi_impl(delta_f_i)

    def solve_phi_truncated_batch(
        self,
        delta_f_i: torch.Tensor,
        *,
        kx_start: int,
    ) -> torch.Tensor:
        """Apply quasineutrality without padding a truncated 5D state.

        The neural state stores a contiguous kx band and the leading
        nonnegative ky band. Quasineutrality is linear in the state, so its
        velocity-weighted charge can be contracted on that compact band; only
        the resulting three-dimensional field is padded.
        """

        grid = self.contract.grid
        if (
            delta_f_i.ndim < 6
            or tuple(delta_f_i.shape[-5:-2])
            != (grid.nvpar, grid.nmu, grid.ns)
            or not delta_f_i.is_complex()
            or not torch.isfinite(torch.view_as_real(delta_f_i)).all()
        ):
            raise ValueError("truncated batched GKW state is invalid")
        nkx = int(delta_f_i.shape[-2])
        nky = int(delta_f_i.shape[-1])
        if (
            nkx <= 0
            or nky <= 0
            or kx_start < 0
            or kx_start + nkx > grid.nkx
            or nky > grid.nky
        ):
            raise ValueError("truncated GKW Fourier band leaves the full grid")
        phi_weight = torch.broadcast_to(
            self._coefficient("phi_weight", delta_f_i),
            grid.state_shape,
        )
        selected_weight = phi_weight[
            ...,
            kx_start : kx_start + nkx,
            :nky,
        ]
        compact_raw = torch.sum(
            selected_weight * delta_f_i,
            dim=(-5, -4),
        )
        phi_raw = F.pad(
            compact_raw,
            (
                0,
                grid.nky - nky,
                kx_start,
                grid.nkx - kx_start - nkx,
            ),
        )
        return self._solve_phi_raw_impl(phi_raw, delta_f_i)

    def _solve_phi_impl(self, delta_f_i: torch.Tensor) -> torch.Tensor:
        phi_weight = self._coefficient("phi_weight", delta_f_i)
        phi_raw = torch.sum(
            phi_weight * delta_f_i,
            dim=(-5, -4),
        )
        return self._solve_phi_raw_impl(phi_raw, delta_f_i)

    def _solve_phi_raw_impl(
        self,
        phi_raw: torch.Tensor,
        like: torch.Tensor,
    ) -> torch.Tensor:
        grid = self.contract.grid
        field_shape = (grid.ns, grid.nkx, grid.nky)
        if (
            tuple(phi_raw.shape[-3:]) != field_shape
            or not phi_raw.is_complex()
            or not torch.isfinite(torch.view_as_real(phi_raw)).all()
        ):
            raise ValueError("GKW weighted charge field is invalid")

        def field_coefficient(name: str) -> torch.Tensor:
            value = self._coefficient(name, like)
            target = (1, 1, *field_shape)
            try:
                expanded = torch.broadcast_to(value, target)
            except RuntimeError as exc:
                raise ValueError(
                    f"GKW field coefficient {name} is not velocity independent"
                ) from exc
            return expanded[0, 0]

        phi_corr_weight = field_coefficient("phi_corr_weight")
        gamma = field_coefficient("phi_gamma")
        buffer_phi = torch.sum(
            phi_corr_weight * phi_raw,
            dim=-3,
            keepdim=True,
        )

        tmp = float(self._scalars["phi_tmp"])
        density = float(self._scalars["phi_de"])
        charge = float(self._scalars["phi_signz"])
        has_zonal = float(self._scalars["phi_has_zonal"])
        ixzero = int(self._scalars["phi_ixzero"])
        iyzero = int(self._scalars["phi_iyzero"])
        real_dtype = phi_raw.real.dtype
        device = phi_raw.device

        maty = tmp / density + torch.sum(
            -phi_corr_weight,
            dim=-3,
            keepdim=True,
        )
        x_mask = torch.zeros(
            (1, grid.nkx, 1),
            device=device,
            dtype=real_dtype,
        )
        x_mask[..., ixzero, :] = 1.0
        y_mask = torch.zeros(
            (1, 1, grid.nky),
            device=device,
            dtype=real_dtype,
        )
        y_mask[..., iyzero] = has_zonal

        one = torch.ones((), device=device, dtype=phi_raw.dtype)
        safe_maty = torch.where(x_mask > 0, one, maty)
        safe_maty = torch.where(
            safe_maty.abs() < 1.0e-15,
            one,
            safe_maty,
        )
        corrected = phi_raw + buffer_phi * y_mask / safe_maty

        poisson_diagonal = charge**2 * density * (gamma - 1.0) / tmp
        zonal_mask = x_mask * y_mask
        pdiag = (
            poisson_diagonal * (1.0 - zonal_mask)
            - charge * density / tmp
        )
        fallback = torch.full_like(pdiag, -1.0)
        pdiag = torch.where(pdiag.abs() < 1.0e-15, fallback, pdiag)
        return corrected * (-1.0 / pdiag)

    def _validate_state(self, delta_f_i: torch.Tensor) -> None:
        if tuple(delta_f_i.shape) != self.contract.grid.state_shape:
            raise ValueError(
                "GKW delta_f_i must have shape "
                f"{self.contract.grid.state_shape}"
            )
        if not delta_f_i.is_complex():
            raise TypeError("GKW delta_f_i must be complex")
        if not torch.isfinite(torch.view_as_real(delta_f_i)).all():
            raise FloatingPointError("GKW delta_f_i is non-finite")

    def _validate_batched_state(self, delta_f_i: torch.Tensor) -> None:
        expected = self.contract.grid.state_shape
        if (
            delta_f_i.ndim <= len(expected)
            or tuple(delta_f_i.shape[-5:]) != expected
        ):
            raise ValueError(
                "batched GKW delta_f_i must have one or more leading batch "
                f"dimensions followed by {expected}"
            )
        if not delta_f_i.is_complex():
            raise TypeError("batched GKW delta_f_i must be complex")
        if not torch.isfinite(torch.view_as_real(delta_f_i)).all():
            raise FloatingPointError("batched GKW delta_f_i is non-finite")

    def _coefficient(
        self,
        name: str,
        like: torch.Tensor,
    ) -> torch.Tensor:
        value = self._arrays[name].to(device=like.device)
        if name in {
            "nl_jind",
            "s_shift",
            "kx_shift",
            "pos_parallel_class",
        }:
            return value.to(dtype=torch.long)
        if name == "valid_shift":
            return value.to(dtype=torch.bool)
        return value.to(dtype=like.real.dtype)

    def _apply_parallel(
        self,
        field: torch.Tensor,
        coefficients: torch.Tensor,
        arrays: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        output = torch.zeros_like(field)
        nky = self.contract.grid.nky
        ky_index = torch.arange(
            nky,
            device=field.device,
            dtype=torch.long,
        ).reshape(1, 1, nky)
        for offset in range(9):
            s_map = arrays["s_shift"][offset]
            kx_map = arrays["kx_shift"][offset]
            valid = arrays["valid_shift"][offset]
            shifted = field[..., s_map, kx_map, ky_index]
            shifted = torch.where(
                valid,
                shifted,
                torch.zeros((), device=field.device, dtype=field.dtype),
            )
            output = output + coefficients[offset] * shifted
        return output

    def _apply_parallel_compact(
        self,
        field: torch.Tensor,
        characteristic: torch.Tensor,
        arrays: Mapping[str, torch.Tensor],
        *,
        dissipation: torch.Tensor | None,
        term7: bool,
    ) -> torch.Tensor:
        real_dtype = field.real.dtype
        device = field.device
        tables = tuple(
            torch.as_tensor(value, device=device, dtype=real_dtype)
            * _VPAR_SCALE
            for value in (
                _PARALLEL_D1_POS,
                _PARALLEL_D1_NEG,
                _PARALLEL_D4_POS,
                _PARALLEL_D4_NEG,
            )
        )
        d1_positive, d1_negative, d4_positive, d4_negative = tables
        position = arrays["pos_parallel_class"] + 2
        spacing = float(self._scalars["sgr_dist"])
        parallel_dissipation = float(self._scalars["disp_par"])
        output = torch.zeros_like(field)

        for offset in range(9):
            d1p = d1_positive[position, offset][None, None]
            d1n = d1_negative[position, offset][None, None]
            if term7:
                d1 = torch.where(characteristic < 0, d1p, d1n)
            else:
                d1 = torch.where(characteristic > 0, d1p, d1n)
            coefficient = characteristic * d1
            if dissipation is not None:
                d4p = d4_positive[position, offset][None, None]
                d4n = d4_negative[position, offset][None, None]
                d4 = torch.where(characteristic > 0, d4p, d4n)
                coefficient = (
                    coefficient
                    + parallel_dissipation * dissipation * d4
                )
            shifted = self._parallel_shift(field, arrays, offset)
            output = output + (coefficient / spacing) * shifted
        return output

    def _parallel_shift(
        self,
        field: torch.Tensor,
        arrays: Mapping[str, torch.Tensor],
        offset: int,
    ) -> torch.Tensor:
        nky = self.contract.grid.nky
        ky_index = torch.arange(
            nky,
            device=field.device,
            dtype=torch.long,
        ).reshape(1, 1, nky)
        s_map = arrays["s_shift"][offset]
        kx_map = arrays["kx_shift"][offset]
        valid = arrays["valid_shift"][offset]
        shifted = field[..., s_map, kx_map, ky_index]
        return torch.where(
            valid,
            shifted,
            torch.zeros((), device=field.device, dtype=field.dtype),
        )

    @staticmethod
    def _apply_vpar_dual(
        field: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nvpar = field.shape[-5]
        index_base = torch.arange(
            nvpar,
            device=field.device,
            dtype=torch.long,
        )
        first = torch.zeros_like(field)
        fourth = torch.zeros_like(field)
        for first_weight, fourth_weight, shift in zip(
            _VPAR_D1,
            _VPAR_D4,
            (-2, -1, 0, 1, 2),
        ):
            unbounded = index_base + shift
            valid = (unbounded >= 0) & (unbounded < nvpar)
            index = unbounded.clamp(0, nvpar - 1)
            shifted = field.index_select(-5, index)
            mask = valid.reshape(-1, 1, 1, 1, 1)
            shifted = torch.where(
                mask,
                shifted,
                torch.zeros((), device=field.device, dtype=field.dtype),
            )
            first = first + (_VPAR_SCALE * first_weight) * shifted
            fourth = fourth + (_VPAR_SCALE * fourth_weight) * shifted
        return first, fourth

    def _nonlinear_term(
        self,
        delta_f_i: torch.Tensor,
        phi: torch.Tensor,
        arrays: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        mrad = int(self._scalars["nl_mrad"])
        mphi = int(self._scalars["nl_mphi"])
        mphiw3 = int(self._scalars["nl_mphiw3"])
        nky = self.contract.grid.nky
        jind = arrays["nl_jind"]

        gyro_phi = arrays["bessel"] * phi.unsqueeze(-4).unsqueeze(-4)
        packed_df = self._pack_half_spectrum(
            delta_f_i,
            jind,
            mrad,
            mphiw3,
        )
        packed_phi = self._pack_half_spectrum(
            gyro_phi,
            jind,
            mrad,
            mphiw3,
        )
        kx2d = arrays["kx"].reshape(-1, 1).expand(
            -1,
            self.contract.grid.nky,
        )
        ky2d = arrays["ky"].reshape(1, -1).expand(
            self.contract.grid.nkx,
            -1,
        )
        packed_ikx = 1j * self._pack_half_spectrum(
            kx2d,
            jind,
            mrad,
            mphiw3,
        )
        packed_iky = 1j * self._pack_half_spectrum(
            ky2d,
            jind,
            mrad,
            mphiw3,
        )

        grad_phi_y = torch.fft.irfft2(
            packed_iky * packed_phi,
            s=(mrad, mphi),
            dim=(-2, -1),
            norm="backward",
        )
        grad_phi_x = torch.fft.irfft2(
            packed_ikx * packed_phi,
            s=(mrad, mphi),
            dim=(-2, -1),
            norm="backward",
        )
        grad_f_x = torch.fft.irfft2(
            packed_ikx * packed_df,
            s=(mrad, mphi),
            dim=(-2, -1),
            norm="backward",
        )
        grad_f_y = torch.fft.irfft2(
            packed_iky * packed_df,
            s=(mrad, mphi),
            dim=(-2, -1),
            norm="backward",
        )
        geometry_factor = arrays["nl_dum_s"].reshape(
            1,
            1,
            -1,
            1,
            1,
        )
        nonlinear_real = geometry_factor * (
            grad_phi_y * grad_f_x
            - grad_phi_x * grad_f_y
        )
        nonlinear_half = torch.fft.rfft2(
            nonlinear_real,
            s=(mrad, mphi),
            dim=(-2, -1),
            norm="backward",
        )
        nonlinear_half = (
            float(self._scalars["nl_fft_scale"])
            * nonlinear_half
        )
        nonlinear = nonlinear_half[..., jind, :nky]
        nonlinear = nonlinear.clone()
        nonlinear[
            ...,
            int(self._scalars["ixzero"]),
            int(self._scalars["iyzero"]),
        ] = 0.0
        return nonlinear.to(delta_f_i.dtype)

    @staticmethod
    def _pack_half_spectrum(
        spectrum: torch.Tensor,
        jind: torch.Tensor,
        mrad: int,
        mphiw3: int,
    ) -> torch.Tensor:
        if spectrum.shape[-2] != jind.numel():
            raise ValueError("GKW packed spectrum and jind disagree")
        nky = spectrum.shape[-1]
        if nky > mphiw3:
            raise ValueError("GKW ky spectrum exceeds the dealiased grid")
        dtype = (
            spectrum.dtype
            if spectrum.is_complex()
            else (
                torch.complex64
                if spectrum.dtype == torch.float32
                else torch.complex128
            )
        )
        output = torch.zeros(
            (*spectrum.shape[:-2], mrad, mphiw3),
            device=spectrum.device,
            dtype=dtype,
        )
        output[..., jind, :nky] = spectrum.to(dtype)
        return output
