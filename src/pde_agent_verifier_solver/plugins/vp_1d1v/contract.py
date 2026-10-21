"""Mathematical contract for manufactured one-dimensional Vlasov--Poisson.

The electric field in this module is the nondimensional acceleration field:

``d_t f + v d_x f + E d_v f = S`` and ``d_x E = n - n_b``.

The charge sign is therefore already absorbed into ``E``.  A fixed neutralizing
background and a zero-mean electric-field gauge make the periodic Poisson
problem unique.  The velocity interval below is only a public query window;
the smooth answers themselves live on the full real velocity line.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi
from typing import Final


VP_1D1V_EQUATION_VERSION: Final = "vp_1d1v_forced_periodic_v1"
VP_1D1V_COORDINATE_ORDER: Final = ("t", "x", "v")
VP_1D1V_MACRO_FIELDS: Final = ("M0", "M1", "M2", "M3", "E")


@dataclass(frozen=True)
class VP1D1VContract:
    """Frozen domain, normalization, and gauge for one VP experiment."""

    x_length: float = 2.0 * pi
    time_interval: tuple[float, float] = (0.0, 1.0)
    background_density: float = 1.0
    velocity_query_window: tuple[float, float] = (-6.0, 6.0)
    minimum_density_fraction: float = 0.15

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not isfinite(float(self.x_length)) or float(self.x_length) <= 0.0:
            raise ValueError("VP x_length must be finite and positive")
        if len(self.time_interval) != 2:
            raise ValueError("VP time_interval must contain two endpoints")
        t0, t1 = (float(value) for value in self.time_interval)
        if not isfinite(t0) or not isfinite(t1) or t1 <= t0:
            raise ValueError("VP time_interval must be finite and increasing")
        if not isfinite(float(self.background_density)) or (
            float(self.background_density) <= 0.0
        ):
            raise ValueError("VP background_density must be finite and positive")
        if len(self.velocity_query_window) != 2:
            raise ValueError("VP velocity_query_window must contain two endpoints")
        v0, v1 = (float(value) for value in self.velocity_query_window)
        if not isfinite(v0) or not isfinite(v1) or v1 <= v0:
            raise ValueError(
                "VP velocity_query_window must be finite and increasing"
            )
        fraction = float(self.minimum_density_fraction)
        if not isfinite(fraction) or not 0.0 < fraction < 1.0:
            raise ValueError("VP minimum_density_fraction must lie in (0,1)")

    @property
    def t0(self) -> float:
        return float(self.time_interval[0])

    @property
    def t1(self) -> float:
        return float(self.time_interval[1])

    @property
    def time_span(self) -> float:
        return self.t1 - self.t0

    @property
    def fundamental_wavenumber(self) -> float:
        return 2.0 * pi / float(self.x_length)

    def validate_density_amplitudes(self, amplitudes) -> None:
        """Conservatively prove positivity of ``1 + sum a_k cos(theta_k)``."""

        values = tuple(float(value) for value in amplitudes)
        if any(not isfinite(value) for value in values):
            raise ValueError("VP density amplitudes must be finite")
        if sum(abs(value) for value in values) > (
            1.0 - float(self.minimum_density_fraction)
        ):
            raise ValueError(
                "VP density modes do not prove the configured positive lower bound"
            )
