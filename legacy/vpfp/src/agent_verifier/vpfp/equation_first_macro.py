from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np

from agent_verifier.vpfp.high_fidelity import (
    VPFPHighFidelityConfig,
    evaluate_continuous_equation,
    make_high_fidelity_case,
    sample_continuous_equation_spec,
)
from agent_verifier.vpfp.io import VPFPCase
from agent_verifier.vpfp.operators import VPFPGrid, VPFPTime
from agent_verifier.vpfp.torch_reference import (
    VPFPTorchSolverConfig,
    evaluate_torch_equation_parameter_batch,
    pack_torch_equation_specs,
    solve_torch_vpfp_batch,
)


@dataclass(frozen=True)
class VPFPEquationFirstMacroConfig:
    target_grid: VPFPGrid
    target_time: VPFPTime
    reference_grid: VPFPGrid
    reference_time: VPFPTime
    equation: VPFPHighFidelityConfig
    solver: VPFPTorchSolverConfig

    def validate(self) -> None:
        self.target_grid.validate()
        self.target_time.validate()
        self.reference_grid.validate()
        self.reference_time.validate()
        self.equation.validate()
        self.solver.validate()
        if self.reference_grid.x_domain != self.target_grid.x_domain:
            raise ValueError("target and reference spatial domains must match")
        if self.reference_grid.v_bounds != self.target_grid.v_bounds:
            raise ValueError("target and reference velocity domains must match")
        if (
            self.reference_time.t0 != self.target_time.t0
            or self.reference_time.t1 != self.target_time.t1
        ):
            raise ValueError("target and reference time domains must match")
        if self.reference_grid.nx % self.target_grid.nx != 0:
            raise ValueError("reference nx must be a nested periodic refinement")
        if self.reference_grid.ny % self.target_grid.ny != 0:
            raise ValueError("reference ny must be a nested periodic refinement")
        for coarse, fine, name in (
            (self.target_grid.nvx, self.reference_grid.nvx, "nvx"),
            (self.target_grid.nvy, self.reference_grid.nvy, "nvy"),
            (self.target_time.nt, self.reference_time.nt, "nt"),
        ):
            if (fine - 1) % (coarse - 1) != 0:
                raise ValueError(f"reference {name} must be an endpoint refinement")
        if self.solver.dtype not in {"float32", "float64"}:
            raise ValueError("unsupported Torch reference dtype")

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_grid": self.target_grid.to_dict(),
            "target_time": self.target_time.to_dict(),
            "reference_grid": self.reference_grid.to_dict(),
            "reference_time": self.reference_time.to_dict(),
            "equation": self.equation.to_dict(),
            "solver": self.solver.to_dict(),
        }


    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "VPFPEquationFirstMacroConfig":
        payload = dict(data)

        def grid(name: str) -> VPFPGrid:
            values = dict(payload[name])
            values["x_domain"] = tuple(float(item) for item in values["x_domain"])
            values["v_bounds"] = tuple(float(item) for item in values["v_bounds"])
            return VPFPGrid(**values)

        config = cls(
            target_grid=grid("target_grid"),
            target_time=VPFPTime(**dict(payload["target_time"])),
            reference_grid=grid("reference_grid"),
            reference_time=VPFPTime(**dict(payload["reference_time"])),
            equation=VPFPHighFidelityConfig.from_dict(payload["equation"]),
            solver=VPFPTorchSolverConfig(**dict(payload["solver"])),
        )
        config.validate()
        return config


class VPFPEquationFirstMacroGenerator:
    """Generate macro-answer cases from batched refined equation solves."""

    def __init__(
        self,
        config: VPFPEquationFirstMacroConfig,
        *,
        device: str | Any = "cuda",
    ) -> None:
        config.validate()
        self.config = config
        self.device = device

    def generate(
        self,
        seeds: Sequence[int],
        *,
        case_ids: Sequence[str],
        cohort: str,
        policy_variant: str,
        candidate_pool_size: int,
    ) -> dict[str, Any]:
        seed_array = np.asarray(seeds, dtype=np.int64)
        if seed_array.ndim != 1 or seed_array.size < 1:
            raise ValueError("seeds must be a nonempty one-dimensional sequence")
        if len(case_ids) != int(seed_array.size):
            raise ValueError("case_ids must match seeds")
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("case_ids must be unique")
        started = time.perf_counter()
        specs = [
            sample_continuous_equation_spec(
                seed=int(seed),
                config=self.config.equation,
                defer_rho_log_partition=False,
            )
            for seed in seed_array
        ]
        parameters = pack_torch_equation_specs(
            specs,
            dtype=self.config.solver.dtype,
        )
        prepared = evaluate_torch_equation_parameter_batch(
            parameters,
            grid=self.config.reference_grid,
            time=self.config.reference_time,
            device=self.device,
        )
        solution = solve_torch_vpfp_batch(
            prepared,
            grid=self.config.reference_grid,
            time=self.config.reference_time,
            config=self.config.solver,
            device=self.device,
        )
        f_batch = solution["f"].detach().cpu().numpy()
        diagnostics = dict(solution["diagnostics"])
        cases: list[VPFPCase] = []
        rows: list[dict[str, Any]] = []
        for index, (seed, case_id, spec) in enumerate(
            zip(seed_array, case_ids, specs, strict=True)
        ):
            reference_f = np.asarray(f_batch[index], dtype=np.float64).copy()
            reference_f[0] = evaluate_continuous_equation(
                spec,
                grid=self.config.reference_grid,
                time=self.config.reference_time,
            )["f0_continuous"]
            case_diagnostics = _slice_batch_diagnostics(
                diagnostics,
                index=index,
                batch_size=len(specs),
            )
            reference_solution = {
                "f": reference_f,
                "diagnostics": case_diagnostics,
            }
            convergence = {
                "schema": "vpfp_equation_first_macro_reference_record_v1",
                "audit_mode": "refined_reference_pending_campaign_convergence_audit",
                "studies": {},
                "velocity_domain_sensitivity": {"not_run_per_case": True},
            }
            case = make_high_fidelity_case(
                case_id=str(case_id),
                spec=spec,
                target_grid=self.config.target_grid,
                target_time=self.config.target_time,
                reference_grid=self.config.reference_grid,
                reference_time=self.config.reference_time,
                reference_solution=reference_solution,
                convergence=convergence,
                config=self.config.equation,
                macro_output=True,
            )
            case = replace(
                case,
                diagnostics={
                    **case.diagnostics,
                    "reference_solver": case_diagnostics,
                    "equation_first_macro_generator": {
                        "schema": "vpfp_torch_equation_first_macro_generation_v1",
                        "equation_seed": int(seed),
                        "cohort": str(cohort),
                        "policy_variant": str(policy_variant),
                        "candidate_pool_size": int(candidate_pool_size),
                    },
                },
                case_spec={
                    **case.case_spec,
                    "cohort": str(cohort),
                    "policy_variant": str(policy_variant),
                    "candidate_pool_size": int(candidate_pool_size),
                },
                generator_metadata={
                    **case.generator_metadata,
                    "backend": str(case_diagnostics["backend"]),
                    "generator_route": "equation_first_refined_torch_macro_v1",
                    "policy_variant": str(policy_variant),
                    "candidate_pool_size": int(candidate_pool_size),
                },
            )
            cases.append(case)
            rows.append(
                {
                    "case_id": str(case_id),
                    "equation_seed": int(seed),
                    "collision_nu": float(spec["collision_nu"]),
                    "temperature": float(spec["temperature"]),
                    "positivity_repair_mass": float(
                        case_diagnostics["repair_mass"]
                    ),
                    "positivity_repair_steps": int(
                        case_diagnostics["repair_steps"]
                    ),
                }
            )
        return {
            "schema": "vpfp_torch_equation_first_macro_batch_v1",
            "cases": cases,
            "rows": rows,
            "diagnostics": {
                "batch_size": len(cases),
                "backend": str(diagnostics["backend"]),
                "solver": _json_safe(diagnostics),
                "wall_time_seconds": float(time.perf_counter() - started),
                "raw_reference_f_bytes": int(f_batch.nbytes),
            },
        }


def _slice_batch_diagnostics(
    diagnostics: dict[str, Any],
    *,
    index: int,
    batch_size: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in diagnostics.items():
        array = np.asarray(value) if isinstance(value, np.ndarray) else None
        if array is not None and array.ndim >= 1 and array.shape[0] == batch_size:
            result[name] = _json_safe(array[index])
        else:
            result[name] = _json_safe(value)
    result["batch_index"] = int(index)
    result["source_batch_size"] = int(batch_size)
    return result


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
