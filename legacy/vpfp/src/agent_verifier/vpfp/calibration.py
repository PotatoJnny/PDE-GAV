from __future__ import annotations

from typing import Any

import numpy as np

from agent_verifier.vpfp.mms import VPFPManufactureConfig, sample_factorized_answer
from agent_verifier.vpfp.operators import VPFPGrid, VPFPTime, compute_source, f_from_rho_g


def build_coarse_forward_seed_summary(
    *,
    grid: VPFPGrid,
    time: VPFPTime,
    collision_nu: float,
    families: tuple[str, ...] = ("landau", "two_stream"),
    samples: int = 8,
    seed: int = 0,
) -> dict[str, Any]:
    """Small CPU calibration summary for locating reasonable VPFP scales.

    This intentionally does not provide validation evidence.  It samples the
    same Landau/two-stream factorized family used by the MMS backend and records
    broad bands that a later neural generator can use as D0-style diagnostics.
    """

    rng = np.random.default_rng(int(seed))
    rows = []
    for idx in range(int(samples)):
        family = families[idx % len(families)]
        cfg = VPFPManufactureConfig(family=family)
        rho, g, spec = sample_factorized_answer(grid=grid, time=time, config=cfg, rng=rng)
        source, _phi, electric = compute_source(rho, g, grid=grid, time=time, collision_nu=collision_nu)
        f = f_from_rho_g(rho, g)
        rows.append(
            {
                "family": spec["family"],
                "f_rms": float(np.sqrt(np.mean(f**2))),
                "rho_rms": float(np.sqrt(np.mean(rho**2))),
                "source_rms": float(np.sqrt(np.mean(source**2))),
                "electric_rms": float(np.sqrt(np.mean(electric**2))),
                "rho_min": float(np.min(rho)),
                "g_max": float(np.max(g)),
            }
        )
    metric_names = sorted(k for k in rows[0] if k != "family") if rows else []
    bands = {}
    for name in metric_names:
        values = np.asarray([row[name] for row in rows], dtype=float)
        bands[name] = {
            "count": int(values.size),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "q05": float(np.quantile(values, 0.05)),
            "q95": float(np.quantile(values, 0.95)),
        }
    return {
        "purpose": "coarse_forward_distribution_calibration_not_validation",
        "equation": "electrostatic_vpfp_factorized",
        "samples": int(samples),
        "seed": int(seed),
        "families": list(families),
        "bands": bands,
        "rows": rows,
    }
