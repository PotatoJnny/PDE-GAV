"""Canonical public equation packets for dataset-aligned GKW cases."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .gkw_equation_contract import (
    GKWEquationContract,
    PINNED_GYARADAX_AUDIT_COMMIT,
)


_ARRAY_MAP = {
    "vpar": "vpgr",
    "intvp": "intvp",
    "mu": "mugr",
    "intmu": "intmu",
    "s": "sgrid",
    "ints": "ints",
    "B": "bn",
    "F": "ffun",
    "G": "gfun",
    "E_psi_zeta": "efun",
    "R": "rfun",
    "B_t_fraction": "bt_frac",
    "kx": "kxrh",
    "ky": "krho",
    "parseval_ky": "parseval",
    "D": "dfun",
    "metric_perp": "little_g",
    "mode_label": "mode_label",
    "ixplus": "ixplus",
    "ixminus": "ixminus",
    "pos_parallel_class": "pos_par_grid_class",
    "s_shift": "s_shift",
    "kx_shift": "kx_shift",
    "valid_shift": "valid_shift",
}
_INTEGER_ARRAYS = {
    "mode_label",
    "ixplus",
    "ixminus",
    "pos_parallel_class",
    "s_shift",
    "kx_shift",
}
_PUBLIC_ARRAY_NAMES = frozenset(_ARRAY_MAP)
_PUBLIC_SCALAR_NAMES = frozenset(GKWEquationContract._REQUIRED_SCALARS)


def _numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    elif not isinstance(value, np.ndarray) and not np.isscalar(value):
        try:
            import jax

            value = jax.device_get(value)
        except (ImportError, TypeError):
            pass
    return np.asarray(value).copy()


def _scalar(value: Any, name: str) -> float:
    array = _numpy(value).reshape(-1)
    if array.size != 1 or not np.isfinite(array).all():
        raise ValueError(f"invalid GKW packet scalar: {name}")
    return float(array[0])


def _parameter(parameters: Any, name: str) -> Any:
    if isinstance(parameters, Mapping):
        if name not in parameters:
            raise KeyError(f"missing GKW runtime parameter: {name}")
        return parameters[name]
    if not hasattr(parameters, name):
        raise AttributeError(f"missing GKW runtime parameter: {name}")
    return getattr(parameters, name)


@dataclass(frozen=True)
class GKWPublicEquationPacket:
    arrays: dict[str, np.ndarray]
    scalars: dict[str, float]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        array_names = set(self.arrays)
        scalar_names = set(self.scalars)
        if array_names != _PUBLIC_ARRAY_NAMES:
            raise ValueError(
                "GKW public packet array keys are not canonical: "
                f"missing={sorted(_PUBLIC_ARRAY_NAMES - array_names)}, "
                f"unexpected={sorted(array_names - _PUBLIC_ARRAY_NAMES)}"
            )
        if scalar_names != _PUBLIC_SCALAR_NAMES:
            raise ValueError(
                "GKW public packet scalar keys are not canonical: "
                f"missing={sorted(_PUBLIC_SCALAR_NAMES - scalar_names)}, "
                f"unexpected={sorted(scalar_names - _PUBLIC_SCALAR_NAMES)}"
            )
        if self.metadata.get("hidden_parameterization_public") is not False:
            raise ValueError("GKW public packet must exclude hidden parameters")
        if self.metadata.get("source_access") != "metered_numerical_oracle":
            raise ValueError("GKW public packet must declare source access")

    @property
    def byte_count(self) -> int:
        return sum(value.nbytes for value in self.arrays.values())


def build_public_packet_from_gyaradax(
    geometry: Mapping[str, Any],
    parameters: Any,
    *,
    case_parameters: Mapping[str, Any],
    case_id: str,
    contract: GKWEquationContract | None = None,
    geometry_reference: str = "gyaradax_compute_geometry",
) -> GKWPublicEquationPacket:
    """Map a pinned Gyaradax geometry dictionary to the public contract."""

    if not case_id:
        raise ValueError("GKW public packet requires a case id")
    contract = contract or GKWEquationContract()
    missing_geometry = sorted(set(_ARRAY_MAP.values()) - set(geometry))
    if missing_geometry:
        raise KeyError(
            f"Gyaradax geometry is incomplete: {missing_geometry}"
        )
    for required in ("ikxspace", "krhomax"):
        if required not in case_parameters:
            raise KeyError(f"missing GKW case parameter: {required}")

    arrays = {
        public_name: _numpy(geometry[geometry_name])
        for public_name, geometry_name in _ARRAY_MAP.items()
    }
    arrays["parseval_ky"] = np.where(
        np.abs(arrays["ky"]) < 1.0e-12,
        1.0,
        2.0,
    ).astype(arrays["ky"].dtype, copy=False)
    for name in _INTEGER_ARRAYS:
        value = arrays[name]
        if not np.issubdtype(value.dtype, np.integer):
            if not np.equal(value, np.rint(value)).all():
                raise ValueError(f"GKW topology array {name} is not integral")
            arrays[name] = np.rint(value).astype(np.int64)
    arrays["valid_shift"] = arrays["valid_shift"].astype(np.bool_)

    scalars = {
        "q": _scalar(geometry["q"], "q"),
        "shat": _scalar(geometry["shat"], "shat"),
        "eps": _scalar(geometry["eps"], "eps"),
        "rlt_i": _scalar(_parameter(parameters, "rlt"), "rlt_i"),
        "rln_i": _scalar(_parameter(parameters, "rln"), "rln_i"),
        "ikxspace": _scalar(case_parameters["ikxspace"], "ikxspace"),
        "krhomax": _scalar(case_parameters["krhomax"], "krhomax"),
        "kthnorm": _scalar(geometry["kthnorm"], "kthnorm"),
        "dvp": _scalar(geometry["dvp"], "dvp"),
        "ds": _scalar(geometry["sgr_dist"], "ds"),
        "idisp": _scalar(_parameter(parameters, "idisp"), "idisp"),
        "dgrid": _scalar(_parameter(parameters, "dgrid"), "dgrid"),
        "mass_i": _scalar(_parameter(parameters, "mas"), "mass_i"),
        "charge_i": _scalar(
            _parameter(parameters, "signz"), "charge_i"
        ),
        "temperature_i": _scalar(
            _parameter(parameters, "tmp"), "temperature_i"
        ),
        "density_i": _scalar(
            _parameter(parameters, "de"), "density_i"
        ),
        "vth_ratio_i": _scalar(
            _parameter(parameters, "vthrat"), "vth_ratio_i"
        ),
        "d2X": _scalar(geometry["d2X"], "d2X"),
        "sign_B": _scalar(geometry["signB"], "sign_B"),
        "ixzero": _scalar(geometry["ixzero"], "ixzero"),
        "iyzero": _scalar(geometry["iyzero"], "iyzero"),
        "disp_par": _scalar(
            _parameter(parameters, "disp_par"), "disp_par"
        ),
        "disp_vp": _scalar(
            _parameter(parameters, "disp_vp"), "disp_vp"
        ),
        "disp_x": _scalar(
            _parameter(parameters, "disp_x"), "disp_x"
        ),
        "disp_y": _scalar(
            _parameter(parameters, "disp_y"), "disp_y"
        ),
    }
    audit = contract.audit_public_coefficients(arrays, scalars)
    if not audit.passed:
        raise ValueError(
            "canonical GKW public packet failed: "
            + ", ".join(audit.failures)
        )

    equation_metadata = contract.public_metadata()
    metadata = {
        "schema": "fusion_gkw_public_equation_packet_v1",
        "case_id": case_id,
        "operator_id": equation_metadata["operator_id"],
        "dataset_id": equation_metadata["dataset_id"],
        "geometry_reference": geometry_reference,
        "gyaradax_commit": PINNED_GYARADAX_AUDIT_COMMIT,
        "source_access": "metered_numerical_oracle",
        "hidden_parameterization_public": False,
        "continuous_hidden_state_is_primary": True,
        "state_access": "initial_state_and_completed_released_cases_only",
        "array_conventions": equation_metadata["array_conventions"],
        "equation_contract": equation_metadata,
        "truth_numeric_precision": "float64",
        "public_solver_reference_included": True,
        "parseval_ky_origin": "physical_nonnegative_ky_half_spectrum",
        "legacy_neural_gk_parseval_is_public": False,
        "byte_count": sum(value.nbytes for value in arrays.values()),
    }
    return GKWPublicEquationPacket(
        arrays=arrays,
        scalars=scalars,
        metadata=metadata,
    )



def load_gkw_public_packet(
    path: str | Path,
) -> GKWPublicEquationPacket:
    """Load a canonical public packet without accepting private payloads."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as payload:
        expected_keys = {
            *(f"array__{name}" for name in _PUBLIC_ARRAY_NAMES),
            *(f"scalar__{name}" for name in _PUBLIC_SCALAR_NAMES),
            "metadata_json",
        }
        if set(payload.files) != expected_keys:
            raise ValueError("serialized GKW public packet keys are invalid")
        metadata_value = np.asarray(payload["metadata_json"])
        if metadata_value.size != 1:
            raise ValueError("serialized GKW metadata must be scalar JSON")
        metadata = json.loads(str(metadata_value.reshape(-1)[0]))
        arrays = {
            name: np.asarray(payload[f"array__{name}"]).copy()
            for name in _PUBLIC_ARRAY_NAMES
        }
        scalars = {
            name: _scalar(payload[f"scalar__{name}"], name)
            for name in _PUBLIC_SCALAR_NAMES
        }
    return GKWPublicEquationPacket(
        arrays=arrays,
        scalars=scalars,
        metadata=metadata,
    )
