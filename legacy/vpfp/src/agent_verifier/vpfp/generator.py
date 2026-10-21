from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from agent_verifier.vpfp.calibration import build_coarse_forward_seed_summary
from agent_verifier.vpfp.io import write_case
from agent_verifier.vpfp.mms import VPFPManufactureConfig, make_mms_case
from agent_verifier.vpfp.operators import VPFPGrid, VPFPTime


class VPFPGeneratorProtocol(Protocol):
    """Backend-stable VPFP generator contract.

    CPU MMS, GPU triplane, and diffusion backends should all write the same
    case directory schema and return a JSON-serializable manifest.
    """

    def generate(self, *, case_count: int, seed: int, out_dir: str | Path) -> dict[str, Any]:
        ...


@dataclass
class CPUMMSGenerator:
    config: dict[str, Any] | None = None

    def generate(self, *, case_count: int, seed: int, out_dir: str | Path) -> dict[str, Any]:
        cfg = dict(self.config or {})
        grid = _grid_from_config(cfg.get("grid", {}))
        time = _time_from_config(cfg.get("time", {}))
        pde_cfg = dict(cfg.get("pde", {}))
        collision_nu = float(pde_cfg.get("collision_nu", 0.05))
        manufacture_cfg = dict(cfg.get("manufacture", {}))
        family = str(manufacture_cfg.get("family", "mixed"))
        mms_cfg = VPFPManufactureConfig(
            family=family,
            amplitude_range=tuple(float(x) for x in manufacture_cfg.get("amplitude_range", [0.03, 0.12])),
            drift_range=tuple(float(x) for x in manufacture_cfg.get("drift_range", [0.02, 0.18])),
            thermal_speed_range=tuple(float(x) for x in manufacture_cfg.get("thermal_speed_range", [0.75, 1.25])),
            stream_speed_range=tuple(float(x) for x in manufacture_cfg.get("stream_speed_range", [1.0, 2.0])),
            temporal_frequency_range=tuple(float(x) for x in manufacture_cfg.get("temporal_frequency_range", [0.6, 2.0])),
            damping_range=tuple(float(x) for x in manufacture_cfg.get("damping_range", [0.1, 0.6])),
            min_rho=float(manufacture_cfg.get("min_rho", 0.05)),
        )
        root = Path(out_dir)
        cases_root = root / "cases"
        cases_root.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(int(seed))
        rows = []
        calibration = build_coarse_forward_seed_summary(
            grid=grid,
            time=time,
            collision_nu=collision_nu,
            samples=int(cfg.get("calibration_samples", 4)),
            seed=int(seed) + 9000,
        )
        for idx in range(int(case_count)):
            sample_seed = int(rng.integers(0, 2**31 - 1))
            case_rng = np.random.default_rng(sample_seed)
            case_id = f"vpfp_mms_{idx:06d}"
            case = make_mms_case(
                case_id=case_id,
                grid=grid,
                time=time,
                collision_nu=collision_nu,
                config=mms_cfg,
                rng=case_rng,
                generator_metadata={
                    "sample_seed": sample_seed,
                    "latent_id": f"cpu_mms_seed_{sample_seed}",
                    "calibration_summary": "coarse_forward_distribution_calibration_not_validation",
                },
            )
            case_dir = cases_root / case_id
            write_case(case_dir, case)
            rows.append(
                {
                    "case_id": case_id,
                    "case_dir": str(case_dir),
                    "family": case.case_spec["family"],
                    "status": "accepted",
                    "truth_vpfp_residual_rel": case.diagnostics["truth_vpfp_residual_rel"],
                    "repeat_initial_difficulty": case.diagnostics["repeat_initial_difficulty"],
                }
            )
        manifest = {
            "backend": "cpu_mms_factorized",
            "case_count": int(case_count),
            "seed": int(seed),
            "equation": "electrostatic_vpfp_factorized",
            "case_schema": "vpfp_factorized_public_hidden_v1",
            "supports_gpu_generator_extension": True,
            "grid": grid.to_dict(),
            "time": time.to_dict(),
            "collision_nu": collision_nu,
            "calibration": calibration,
            "cases": rows,
        }
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return manifest


def _grid_from_config(data: dict[str, Any]) -> VPFPGrid:
    return VPFPGrid(
        nx=int(data.get("nx", data.get("shape", [8, 8, 9, 9])[0] if "shape" in data else 8)),
        ny=int(data.get("ny", data.get("shape", [8, 8, 9, 9])[1] if "shape" in data else 8)),
        nvx=int(data.get("nvx", data.get("shape", [8, 8, 9, 9])[2] if "shape" in data else 9)),
        nvy=int(data.get("nvy", data.get("shape", [8, 8, 9, 9])[3] if "shape" in data else 9)),
        x_domain=tuple(float(x) for x in data.get("x_domain", [2.0 * np.pi, 2.0 * np.pi])),
        v_bounds=tuple(float(x) for x in data.get("v_bounds", [-5.0, 5.0, -5.0, 5.0])),
        dtype=str(data.get("dtype", "float64")),
    )


def _time_from_config(data: dict[str, Any]) -> VPFPTime:
    return VPFPTime(
        t0=float(data.get("t0", 0.0)),
        t1=float(data.get("t1", 0.5)),
        nt=int(data.get("nt", 5)),
    )
