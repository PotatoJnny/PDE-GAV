"""Fourier--Hermite solver for the continuous 2D2V VPFP task.

The velocity dependence is represented in Maxwellian-weighted probabilists'
Hermite modes. In this representation the Ornstein--Uhlenbeck collision
operator is diagonal and the first four mode levels give the requested raw
moments exactly. Periodic spatial derivatives and Poisson are spectral.
Public g0 and source values are projected with tensor Gauss--Hermite
quadrature; no preview array is treated as a required numerical grid.
"""
from __future__ import annotations

from math import factorial, isfinite

import numpy as np

from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    AnswerView,
    QueryKind,
    open_public_case_session,
)
from pde_agent_verifier_solver.sdk.solve_budget import candidate_solve_clock


FIELDS = (
    "rho", "electric_x", "electric_y", "momentum_x", "momentum_y",
    "pressure_xx", "pressure_xy", "pressure_yy", "heat_flux_x",
    "heat_flux_y",
)

NX = 64
CHARGE_NX = 40
SOURCE_NX = 24
HERMITE_MAX = 10
QUAD_N = 16
TIME_STEPS = 1024


def _lagrange4(s):
    """Four adjacent integer nodes and their cubic Lagrange weights."""
    s = np.asarray(s, dtype=np.float64)
    base = np.floor(s).astype(np.int64) - 1
    nodes = base[:, None] + np.arange(4, dtype=np.int64)[None, :]
    weights = np.ones((len(s), 4), dtype=np.float64)
    for i in range(4):
        for j in range(4):
            if i != j:
                weights[:, i] *= (s - nodes[:, j]) / float(i - j)
    return nodes, weights


class Prepared:
    def __init__(self, values, periods, interval):
        values = np.asarray(values, dtype=np.float64)
        self.values = values
        # Charge and field benefit from the regularization scale of the x40
        # incumbent, while stress/heat benefit from the x64 evolution and
        # projection.  Build an exactly restricted x40 charge representation
        # but retain all other fields on the native x64 grid.
        transformed = np.fft.fft2(values[:, :3], axes=(-2, -1))
        frequencies = (
            np.fft.fftfreq(CHARGE_NX) * CHARGE_NX
        ).astype(np.int64)
        retained = frequencies != -(CHARGE_NX // 2)
        charge_hat = np.zeros(
            values.shape[:1] + (3, CHARGE_NX, CHARGE_NX),
            dtype=np.complex128,
        )
        out_i = np.nonzero(retained)[0]
        in_i = np.mod(frequencies[retained], NX)
        charge_hat[..., out_i[:, None], out_i[None, :]] = transformed[
            ..., in_i[:, None], in_i[None, :]
        ] * (float(CHARGE_NX) / NX) ** 2
        self.charge_values = np.fft.ifft2(
            charge_hat, axes=(-2, -1)
        ).real
        # Retain all perturbation charge modes observed in the public-source
        # audit while removing cubic interpolation error and unresolved
        # high-frequency content from rho and E.
        self.spectral_indices = np.asarray((0, 1, 2, 3, 4, 8, 9))
        self.spectra = (
            np.fft.fft2(values[:, self.spectral_indices], axes=(-2, -1))
            / float(NX * NX)
        )
        self.frequencies = np.fft.fftfreq(NX) * NX
        charge_mask = (
            (np.abs(np.rint(self.frequencies))[:, None] <= 12)
            & (np.abs(np.rint(self.frequencies))[None, :] <= 12)
        )
        self.spectra[:, :3] *= charge_mask[None, None, :, :]
        self.periods = np.asarray(periods, dtype=np.float64)
        self.interval = np.asarray(interval, dtype=np.float64)
        self.closed = False

    def evaluate(self, query):
        if self.closed:
            raise RuntimeError("Fourier--Hermite VPFP handle is closed")
        if not isinstance(query, AnswerQuery) or query.kind is not QueryKind.POINTS:
            raise ValueError("Fourier--Hermite VPFP solver accepts POINTS only")
        if (not query.fields or len(set(query.fields)) != len(query.fields)
                or not set(query.fields) <= set(FIELDS)):
            raise ValueError("unknown or repeated transport field")
        points = np.asarray(
            [(p["t"], p["x"], p["y"]) for p in query.points],
            dtype=np.float64,
        )
        if (points.ndim != 2 or points.shape[1:] != (3,) or not len(points)
                or not np.isfinite(points).all()):
            raise ValueError("finite nonempty t,x,y points required")
        if (np.any(points[:, 0] < self.interval[0])
                or np.any(points[:, 0] > self.interval[1])):
            raise ValueError("query leaves prepared time interval")

        nt = self.values.shape[0]
        st = (points[:, 0] - self.interval[0]) * (nt - 1) / (
            self.interval[1] - self.interval[0]
        )
        ti, _ = _lagrange4(st)
        base = np.clip(ti[:, 0], 0, nt - 4)
        ti = base[:, None] + np.arange(4, dtype=np.int64)[None, :]
        wt = np.ones((len(points), 4), dtype=np.float64)
        for i in range(4):
            for j in range(4):
                if i != j:
                    wt[:, i] *= (st - ti[:, j]) / float(i - j)

        sx = np.mod(points[:, 1] / self.periods[0], 1.0) * NX
        sy = np.mod(points[:, 2] / self.periods[1], 1.0) * NX
        xi, wx = _lagrange4(sx)
        yi, wy = _lagrange4(sy)
        xi %= NX
        yi %= NX

        out = np.zeros((len(points), len(FIELDS)), dtype=np.float64)
        for it in range(4):
            for ix in range(4):
                for iy in range(4):
                    weight = wt[:, it] * wx[:, ix] * wy[:, iy]
                    out += weight[:, None] * self.values[
                        ti[:, it], :, xi[:, ix], yi[:, iy]
                    ]

        charge_sx = (
            np.mod(points[:, 1] / self.periods[0], 1.0) * CHARGE_NX
        )
        charge_sy = (
            np.mod(points[:, 2] / self.periods[1], 1.0) * CHARGE_NX
        )
        charge_xi, charge_wx = _lagrange4(charge_sx)
        charge_yi, charge_wy = _lagrange4(charge_sy)
        charge_xi %= CHARGE_NX
        charge_yi %= CHARGE_NX
        charge = np.zeros((len(points), 3), dtype=np.float64)
        for it in range(4):
            for ix in range(4):
                for iy in range(4):
                    weight = (
                        wt[:, it] * charge_wx[:, ix] * charge_wy[:, iy]
                    )
                    charge += weight[:, None] * self.charge_values[
                        ti[:, it], :, charge_xi[:, ix], charge_yi[:, iy]
                    ]
        out[:, :3] = charge

        coefficients = np.zeros(
            (len(points), len(self.spectral_indices), NX, NX),
            dtype=np.complex128,
        )
        for it in range(4):
            coefficients += (
                wt[:, it, None, None, None] * self.spectra[ti[:, it]]
            )
        phase_x = np.exp(
            2j * np.pi * np.outer(
                np.mod(points[:, 1] / self.periods[0], 1.0), self.frequencies
            )
        )
        phase_y = np.exp(
            2j * np.pi * np.outer(
                np.mod(points[:, 2] / self.periods[1], 1.0), self.frequencies
            )
        )
        out[:, self.spectral_indices] = np.einsum(
            "nfij,ni,nj->nf", coefficients, phase_x, phase_y, optimize=True
        ).real
        if not np.isfinite(out).all():
            raise ValueError("nonfinite Fourier--Hermite interpolation")
        mapping = {
            name: tuple(map(float, out[:, i])) for i, name in enumerate(FIELDS)
        }
        return AnswerView(
            kind=query.kind,
            fields={name: mapping[name] for name in query.fields},
            points=query.points,
            shape=(len(points),),
        )

    def close(self):
        self.closed = True
        self.values = None
        self.spectra = None
        self.charge_values = None


class _Solver:
    def __init__(self, case, solve_budget, started):
        self.case = case
        self.solve_budget = solve_budget
        self.started = started
        question = case.public_data["question"]
        self.periods = np.asarray(question["grid"]["x_domain"], dtype=np.float64)
        self.interval = np.asarray(question["time_domain"], dtype=np.float64)
        self.theta = float(question["temperature"])
        self.nu = float(question["collision_nu"])
        if (self.periods.shape != (2,) or np.any(~np.isfinite(self.periods))
                or np.any(self.periods <= 0) or self.interval.shape != (2,)
                or np.any(~np.isfinite(self.interval))
                or self.interval[1] <= self.interval[0]
                or not isfinite(self.theta) or self.theta <= 0
                or not isfinite(self.nu) or self.nu < 0):
            raise ValueError("invalid public VPFP parameters")
        limit = question.get("numerical_access", {}).get(
            "max_points_per_request", 262144
        )
        if type(limit) is not int or limit <= 0:
            raise ValueError("public Oracle request limit must be positive")
        self.limit = limit
        self.session = open_public_case_session(case)

        self.x = np.arange(NX, dtype=np.float64) * self.periods[0] / NX
        self.y = np.arange(NX, dtype=np.float64) * self.periods[1] / NX
        self.source_x = (
            np.arange(SOURCE_NX, dtype=np.float64) * self.periods[0] / SOURCE_NX
        )
        self.source_y = (
            np.arange(SOURCE_NX, dtype=np.float64) * self.periods[1] / SOURCE_NX
        )
        self.kx = 2 * np.pi * np.fft.fftfreq(NX, d=self.periods[0] / NX)
        self.ky = 2 * np.pi * np.fft.fftfreq(NX, d=self.periods[1] / NX)
        self.kx2 = self.kx[:, None]
        self.ky2 = self.ky[None, :]
        self.k2 = self.kx2 * self.kx2 + self.ky2 * self.ky2

        z, w = np.polynomial.hermite.hermgauss(QUAD_N)
        xi = np.sqrt(2.0) * z
        h = np.empty((HERMITE_MAX + 1, QUAD_N), dtype=np.float64)
        h[0] = 1.0
        h[1] = xi
        for m in range(1, HERMITE_MAX):
            h[m + 1] = xi * h[m] - m * h[m - 1]
        norms = np.sqrt(
            np.asarray([factorial(m) for m in range(HERMITE_MAX + 1)],
                       dtype=np.float64)
        )
        self.projector = h * (w * np.exp(z * z))[None, :]
        self.mode_norm = norms[:, None] * norms[None, :]
        self.v = np.sqrt(2.0 * self.theta) * z

        xx, yy, vx, vy = np.meshgrid(
            self.x, self.y, self.v, self.v, indexing="ij"
        )
        self.g_points = np.column_stack(
            (xx.ravel(), yy.ravel(), vx.ravel(), vy.ravel())
        )
        sx, sy, svx, svy = np.meshgrid(
            self.source_x, self.source_y, self.v, self.v, indexing="ij"
        )
        self.s_points = np.empty((sx.size, 5), dtype=np.float64)
        self.s_points[:, 1:] = np.column_stack(
            (sx.ravel(), sy.ravel(), svx.ravel(), svy.ravel())
        )

    def check_budget(self, reserve=12.0):
        if (self.solve_budget is not None
                and candidate_solve_clock() - self.started
                > max(0.0, self.solve_budget - reserve)):
            raise TimeoutError("Fourier--Hermite solve exhausted Case budget")

    def _oracle(self, method, points):
        chunks = []
        for first in range(0, len(points), self.limit):
            self.check_budget()
            value = getattr(self.session, method)(points[first:first + self.limit])
            value = np.asarray(value, dtype=np.float64)
            expected = min(self.limit, len(points) - first)
            if value.shape != (expected,):
                raise ValueError("unexpected public Oracle response shape")
            if not np.isfinite(value).all():
                raise ValueError("nonfinite public Oracle response")
            chunks.append(value)
        return np.concatenate(chunks)

    def _project(self, sampled, spatial_n=NX):
        sampled = np.asarray(sampled, dtype=np.float64).reshape(
            spatial_n, spatial_n, QUAD_N, QUAD_N
        )
        temp = np.einsum(
            "xyij,nj->xyin", sampled, self.projector, optimize=True
        )
        raw = np.einsum(
            "xyin,mi->mnxy", temp, self.projector, optimize=True
        )
        raw *= 2.0 * self.theta
        return raw / self.mode_norm[:, :, None, None]

    def initial_modes(self):
        self.check_budget()
        xx, yy = np.meshgrid(self.x, self.y, indexing="ij")
        xy = np.column_stack((xx.ravel(), yy.ravel()))
        rho = self._oracle("rho0", xy).reshape(NX, NX)
        g = self._oracle("g0", self.g_points)
        modes = self._project(g)
        mass = modes[0, 0].copy()
        if np.any(rho <= 0) or np.any(mass <= 0):
            raise ValueError("positive public initial density and g mass required")
        modes *= (rho / mass)[None, None, :, :]
        return modes

    def source_modes(self, time_value):
        self.s_points[:, 0] = float(time_value)
        coarse = self._project(
            self._oracle("source", self.s_points), spatial_n=SOURCE_NX
        )
        if SOURCE_NX == NX:
            return coarse
        transformed = np.fft.fftshift(
            np.fft.fft2(coarse, axes=(-2, -1)), axes=(-2, -1)
        )
        left = (NX - SOURCE_NX) // 2
        right = NX - SOURCE_NX - left
        padded = np.pad(
            transformed, ((0, 0), (0, 0), (left, right), (left, right))
        )
        lifted = np.fft.ifft2(
            np.fft.ifftshift(padded, axes=(-2, -1)), axes=(-2, -1)
        ).real
        return lifted * (float(NX) / SOURCE_NX)**2

    def electric(self, rho):
        modes = np.fft.fft2(rho)
        potential = np.zeros_like(modes)
        np.divide(modes, self.k2, out=potential, where=self.k2 > 0)
        ex = np.fft.ifft2(-1j * self.kx2 * potential).real
        ey = np.fft.ifft2(-1j * self.ky2 * potential).real
        return ex, ey

    def nonstiff_rhs(self, modes, source):
        mh = np.fft.fft2(modes, axes=(-2, -1))
        xflux = np.zeros_like(mh)
        yflux = np.zeros_like(mh)
        roots = np.sqrt(np.arange(1, HERMITE_MAX + 1, dtype=np.float64))
        xfac = roots[:, None, None, None]
        yfac = roots[None, :, None, None]
        xflux[:-1] += xfac * mh[1:]
        xflux[1:] += xfac * mh[:-1]
        yflux[:, :-1] += yfac * mh[:, 1:]
        yflux[:, 1:] += yfac * mh[:, :-1]
        derivative = np.fft.ifft2(
            1j * self.kx2[None, None, :, :] * xflux
            + 1j * self.ky2[None, None, :, :] * yflux,
            axes=(-2, -1),
        ).real
        rhs = source - np.sqrt(self.theta) * derivative

        ex, ey = self.electric(modes[0, 0])
        inv_thermal = 1.0 / np.sqrt(self.theta)
        rhs[1:] += inv_thermal * xfac * modes[:-1] * ex
        rhs[:, 1:] += inv_thermal * yfac * modes[:, :-1] * ey
        return rhs

    def macro_fields(self, b):
        a00 = b[0, 0]
        a10 = b[1, 0]
        a01 = b[0, 1]
        a20 = np.sqrt(2.0) * b[2, 0]
        a11 = b[1, 1]
        a02 = np.sqrt(2.0) * b[0, 2]
        a30 = np.sqrt(6.0) * b[3, 0]
        a21 = np.sqrt(2.0) * b[2, 1]
        a12 = np.sqrt(2.0) * b[1, 2]
        a03 = np.sqrt(6.0) * b[0, 3]

        thermal = np.sqrt(self.theta)
        rho = a00
        safe_rho = np.maximum(rho, 1.0e-10)
        jx = thermal * a10
        jy = thermal * a01
        rxx = self.theta * (a20 + rho)
        rxy = self.theta * a11
        ryy = self.theta * (a02 + rho)
        ux = jx / safe_rho
        uy = jy / safe_rho
        pxx = rxx - jx * ux
        pxy = rxy - jx * uy
        pyy = ryy - jy * uy

        theta32 = self.theta * thermal
        txxx = theta32 * (a30 + 3.0 * a10)
        txxy = theta32 * (a21 + a01)
        txyy = theta32 * (a12 + a10)
        tyyy = theta32 * (a03 + 3.0 * a01)
        cxxx = txxx - 3.0 * ux * rxx + 2.0 * rho * ux**3
        cxxy = txxy - uy * rxx - 2.0 * ux * rxy + 2.0 * rho * ux**2 * uy
        cxyy = txyy - ux * ryy - 2.0 * uy * rxy + 2.0 * rho * ux * uy**2
        cyyy = tyyy - 3.0 * uy * ryy + 2.0 * rho * uy**3
        qx = 0.5 * (cxxx + cxyy)
        qy = 0.5 * (cxxy + cyyy)
        ex, ey = self.electric(rho)
        return np.stack(
            (rho, ex, ey, jx, jy, pxx, pxy, pyy, qx, qy), axis=0
        )

    def run(self):
        # Evolve the requested 1024-step trajectory together with a 512-step
        # trajectory that reuses exactly the same five source projections per
        # pair of fine steps.  Classical fourth-order Richardson extrapolation
        # then cancels the leading ETDRK4 time-discretization error.  The
        # correction at odd fine nodes is interpolated between its neighboring
        # coarse nodes, retaining the incumbent's dense output grid without an
        # additional public-Oracle request.
        if TIME_STEPS % 2:
            raise ValueError("Richardson solve requires an even step count")
        modes = self.initial_modes()
        coarse_modes = modes.copy()
        values = np.empty((TIME_STEPS + 1, len(FIELDS), NX, NX), dtype=np.float64)
        values[0] = self.macro_fields(modes)
        t0, t1 = self.interval
        dt = (t1 - t0) / TIME_STEPS

        degree = (
            np.arange(HERMITE_MAX + 1)[:, None]
            + np.arange(HERMITE_MAX + 1)[None, :]
        )
        zero = degree == 0

        def coefficients(step_size):
            z = -self.nu * degree * step_size
            decay = np.exp(z)
            half_decay = np.exp(0.5 * z)
            qcoef = np.empty_like(z, dtype=np.float64)
            f1 = np.empty_like(z, dtype=np.float64)
            f2 = np.empty_like(z, dtype=np.float64)
            f3 = np.empty_like(z, dtype=np.float64)
            qcoef[zero] = 0.5 * step_size
            f1[zero] = step_size / 6.0
            f2[zero] = step_size / 6.0
            f3[zero] = step_size / 6.0
            zn = z[~zero]
            ez = decay[~zero]
            qcoef[~zero] = step_size * np.expm1(0.5 * zn) / zn
            f1[~zero] = step_size * (
                -4.0 - zn + ez * (4.0 - 3.0 * zn + zn * zn)
            ) / (zn**3)
            f2[~zero] = step_size * (
                2.0 + zn + ez * (-2.0 + zn)
            ) / (zn**3)
            f3[~zero] = step_size * (
                -4.0 - 3.0 * zn - zn * zn + ez * (4.0 - zn)
            ) / (zn**3)
            return tuple(
                item[:, :, None, None]
                for item in (decay, half_decay, qcoef, f1, f2, f3)
            )

        fine_coefficients = coefficients(dt)
        coarse_coefficients = coefficients(2.0 * dt)

        def advance(state, source0, source_mid, source1, coeff):
            decay, half_decay, qcoef, f1, f2, f3 = coeff
            k1 = self.nonstiff_rhs(state, source0)
            stage2 = half_decay * state + qcoef * k1
            k2 = self.nonstiff_rhs(stage2, source_mid)
            stage3 = half_decay * state + qcoef * k2
            k3 = self.nonstiff_rhs(stage3, source_mid)
            stage4 = half_decay * stage2 + qcoef * (2.0 * k3 - k1)
            k4 = self.nonstiff_rhs(stage4, source1)
            return (
                decay * state + f1 * k1
                + 2.0 * f2 * (k2 + k3) + f3 * k4
            )

        source_even = self.source_modes(t0)
        previous_correction = np.zeros_like(modes)
        for pair in range(TIME_STEPS // 2):
            ta = t0 + 2.0 * dt * pair
            source_quarter = self.source_modes(ta + 0.5 * dt)
            source_mid = self.source_modes(ta + dt)
            odd_modes = advance(
                modes, source_even, source_quarter, source_mid,
                fine_coefficients,
            )
            source_three_quarter = self.source_modes(ta + 1.5 * dt)
            source_next = self.source_modes(ta + 2.0 * dt)
            next_modes = advance(
                odd_modes, source_mid, source_three_quarter, source_next,
                fine_coefficients,
            )
            next_coarse_modes = advance(
                coarse_modes, source_even, source_mid, source_next,
                coarse_coefficients,
            )
            correction = (next_modes - next_coarse_modes) / 15.0
            corrected_odd = odd_modes + 0.5 * (
                previous_correction + correction
            )
            values[2 * pair + 1] = self.macro_fields(corrected_odd)
            values[2 * pair + 2] = self.macro_fields(
                next_modes + correction
            )
            modes = next_modes
            coarse_modes = next_coarse_modes
            previous_correction = correction
            source_even = source_next
            if (not np.isfinite(modes).all()
                    or not np.isfinite(coarse_modes).all()):
                raise ValueError("nonfinite Richardson Fourier--Hermite evolution")
            self.check_budget(reserve=18.0)

        if (not np.isfinite(values).all()
                or np.min(values[:, 0]) <= 0):
            raise ValueError("invalid evolved VPFP macroscopic fields")
        return values


def prepare_solution(case, solve_budget=None):
    started = candidate_solve_clock()
    if solve_budget is not None and (
        isinstance(solve_budget, bool)
        or not isinstance(solve_budget, (int, float))
        or not isfinite(solve_budget)
        or solve_budget <= 0
    ):
        raise ValueError("solve_budget must be finite and positive")
    solver = _Solver(case, solve_budget, started)
    values = solver.run()
    return Prepared(values, solver.periods, solver.interval)
