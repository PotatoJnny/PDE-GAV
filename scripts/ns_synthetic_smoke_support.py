"""Analytic NS source used only by the 2+1 GPU controller-chain smoke.

This is deliberately not a trained Generator and must never back a formal
scientific run.  It provides a cheap smooth answer-first field so the shared
process-worker, public Oracle, perturbation, CandidateBundle, and evaluator
chain can be exercised before sealed structured-Flow artifacts are available.
"""

from __future__ import annotations

import hashlib
from math import pi
from pathlib import Path

import numpy as np


class _SyntheticSmokeAnswer:
    domain_bounds = (0.0, 1.0, 0.0, 1.0)
    time_interval = (0.0, 1.0)
    viscosity = 0.05
    public_geometry = {
        "schema": "synthetic_unit_square_smoke_geometry_v1",
        "outer_boundary": "unit_square",
        "obstacles": [],
    }
    public_metadata = {
        "role": "synthetic_cpu_chain_test_only",
        "formal_training_distribution": False,
        "trained_generator": False,
    }

    def __init__(self, amplitude: float) -> None:
        self.amplitude = float(amplitude)

    def evaluate_velocity(self, points):
        values = np.asarray(points, dtype=np.float64)
        t, x, y = values.T
        scale = self.amplitude * (1.0 + t)
        velocity_x = scale * pi * np.sin(pi * x) ** 2 * np.sin(2.0 * pi * y)
        velocity_y = -scale * pi * np.sin(2.0 * pi * x) * np.sin(pi * y) ** 2
        return np.column_stack((velocity_x, velocity_y))

    def evaluate_forcing(self, points):
        values = np.asarray(points, dtype=np.float64)
        t, x, y = values.T
        amplitude = self.amplitude
        scale = amplitude * (1.0 + t)
        sin_x = np.sin(pi * x)
        sin_y = np.sin(pi * y)
        sin_2x = np.sin(2.0 * pi * x)
        sin_2y = np.sin(2.0 * pi * y)
        cos_2x = np.cos(2.0 * pi * x)
        cos_2y = np.cos(2.0 * pi * y)
        velocity_x, velocity_y = self.evaluate_velocity(values).T
        d_t_u = amplitude * pi * sin_x**2 * sin_2y
        d_t_v = -amplitude * pi * sin_2x * sin_y**2
        d_x_u = scale * pi**2 * sin_2x * sin_2y
        d_y_u = 2.0 * scale * pi**2 * sin_x**2 * cos_2y
        d_x_v = -2.0 * scale * pi**2 * cos_2x * sin_y**2
        d_y_v = -scale * pi**2 * sin_2x * sin_2y
        laplacian_u = (
            scale * pi**3 * (2.0 * cos_2x - 4.0 * sin_x**2) * sin_2y
        )
        laplacian_v = (
            scale * pi**3 * sin_2x * (4.0 * sin_y**2 - 2.0 * cos_2y)
        )
        return np.column_stack(
            (
                d_t_u
                + velocity_x * d_x_u
                + velocity_y * d_y_u
                - self.viscosity * laplacian_u,
                d_t_v
                + velocity_x * d_x_v
                + velocity_y * d_y_v
                - self.viscosity * laplacian_v,
            )
        )


def build_synthetic_ns_smoke_plugin(*, max_points_per_request: int):
    from pde_agent_verifier_solver.plugins.ns import (
        StructuredNSManufacturedCaseSource,
        bind_ns_macro_loop,
    )

    implementation_sha256 = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()

    def clean_sampler(seed: int):
        return _SyntheticSmokeAnswer(0.35 + (int(seed) % 101) / 1000.0)

    def perturbation_sampler(parent: _SyntheticSmokeAnswer, seed: int):
        direction = 1.0 if int(seed) % 2 else -1.0
        return _SyntheticSmokeAnswer(parent.amplitude + direction * 0.025)

    source = StructuredNSManufacturedCaseSource(
        generator_id="ns-synthetic-cpu-chain-test-only",
        artifact_sha256=implementation_sha256,
        clean_sampler=clean_sampler,
        perturbation_sampler=perturbation_sampler,
        max_points_per_request=int(max_points_per_request),
    )
    identity = {
        "schema": "ns_synthetic_cpu_chain_test_identity_v1",
        "scientific_tier": "synthetic_cpu_chain_test_only",
        "formal_training_distribution": False,
        "trained_generator": False,
        "guidance_enabled": False,
        "implementation_sha256": implementation_sha256,
    }
    return bind_ns_macro_loop(case_source=source), identity


__all__ = ["build_synthetic_ns_smoke_plugin"]
