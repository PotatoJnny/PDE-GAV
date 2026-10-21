"""Small forced Hermite hierarchy and features for VPFP macro inference."""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline


INDICES = tuple((n - b, b) for n in range(4) for b in range(n + 1))
POSITION = {degree: index for index, degree in enumerate(INDICES)}


def _spatial_operators(nx, ny, lengths):
    lx, ly = map(float, lengths)
    kx = 2*np.pi*np.fft.fftfreq(nx, d=lx/nx)
    ky = 2*np.pi*np.fft.fftfreq(ny, d=ly/ny)

    def dx(a):
        shape = (1,)*(a.ndim-2) + (nx, 1)
        return np.fft.ifft(1j*kx.reshape(shape)*np.fft.fft(a, axis=-2),
                           axis=-2).real

    def dy(a):
        shape = (1,)*(a.ndim-1) + (ny,)
        return np.fft.ifft(1j*ky.reshape(shape)*np.fft.fft(a, axis=-1),
                           axis=-1).real

    den = kx[:, None]**2 + ky[None, :]**2
    den[0, 0] = 1.0

    def electric(rho):
        rh = np.fft.fft2(rho-rho.mean())
        return (np.fft.ifft2(-1j*kx[:, None]*rh/den).real,
                np.fft.ifft2(-1j*ky[None, :]*rh/den).real)
    return dx, dy, electric


def _velocity_data(case, log_refine=True):
    a = case["arrays"]
    rho = np.asarray(a["rho0"], float)
    momentum = np.asarray(a["momentum0"], float)
    kinetic = np.asarray(a["kinetic_energy0"], float)
    g0 = np.asarray(a["g0"], float)
    source = np.asarray(a["source"], float)
    vxv = np.asarray(a["vx"], float)
    vyv = np.asarray(a["vy"], float)
    vx, vy = vxv[:, None], vyv[None, :]
    weights = np.asarray(a["velocity_weights"], float)
    f0 = rho[..., None, None]*g0

    def integral(z):
        return np.sum(z*weights, axis=(-2, -1))

    initial = np.stack(
        [integral(f0*vx**ax*vy**ay) for ax, ay in INDICES], axis=0)
    forcing = np.stack(
        [integral(source*vx**ax*vy**ay) for ax, ay in INDICES], axis=1)
    initial[POSITION[0, 0]] = rho
    initial[POSITION[1, 0]] = momentum[..., 0]
    initial[POSITION[0, 1]] = momentum[..., 1]
    trace_error = (2*kinetic - initial[POSITION[2, 0]]
                   - initial[POSITION[0, 2]])
    initial[POSITION[2, 0]] += .5*trace_error
    initial[POSITION[0, 2]] += .5*trace_error

    if log_refine and len(vxv) >= 4 and len(vyv) >= 4:
        n = 29
        tx = np.linspace(vxv[0], vxv[-1], n)
        ty = np.linspace(vyv[0], vyv[-1], n)
        logg = CubicSpline(vxv, np.log(np.maximum(g0, 1e-35)),
                           axis=-2)(tx)
        logg = CubicSpline(vyv, logg, axis=-1)(ty)
        gr = np.exp(np.clip(logg, -700, 20))
        weight = (tx[1]-tx[0])*(ty[1]-ty[0])
        gr /= np.maximum(gr.sum(axis=(-2, -1))*weight, 1e-30)[..., None, None]
        mean_x = np.sum(gr*tx[:, None], axis=(-2, -1))*weight
        mean_y = np.sum(gr*ty[None, :], axis=(-2, -1))*weight
        ux = momentum[..., 0]/np.maximum(rho, .2)
        uy = momentum[..., 1]/np.maximum(rho, .2)
        thermal = np.sum(
            gr*((tx[:, None]-mean_x[..., None, None])**2
                +(ty[None, :]-mean_y[..., None, None])**2),
            axis=(-2, -1))*weight
        thermal_target = (2*kinetic/np.maximum(rho, .2)-ux*ux-uy*uy)
        scale = np.sqrt(np.maximum(thermal_target, 1e-8)
                        / np.maximum(thermal, 1e-8))
        mxgrid = (ux[..., None, None]
                  + scale[..., None, None]*(tx[:, None]-mean_x[..., None, None]))
        mygrid = (uy[..., None, None]
                  + scale[..., None, None]*(ty[None, :]-mean_y[..., None, None]))
        refined = {
            degree: rho*np.sum(gr*mxgrid**degree[0]*mygrid**degree[1],
                               axis=(-2, -1))*weight
            for degree in INDICES if sum(degree) >= 2
        }
        for degree in INDICES:
            total = sum(degree)
            if total == 2:
                blend = 1.437 if degree != (1, 1) else 1.111
            elif total == 3:
                blend = 1.0
            else:
                continue
            j = POSITION[degree]
            initial[j] += blend*(refined[degree]-initial[j])
        # Authoritative public energy always wins over reconstruction.
        trace_error = (2*kinetic-initial[POSITION[2, 0]]
                       - initial[POSITION[0, 2]])
        initial[POSITION[2, 0]] += .5*trace_error
        initial[POSITION[0, 2]] += .5*trace_error
    return initial, forcing


def solve_hierarchy(case, query_times, maximum_step=.002, return_state=False, kurtosis=1.0, source_scales=None):
    """Evolve degrees zero through three with a Gaussian degree-four edge."""
    q = np.asarray(query_times, float)
    if q.ndim != 1 or q.size == 0 or not np.all(np.isfinite(q)):
        raise ValueError("query_times must be a finite nonempty 1D array")
    arrays, question = case["arrays"], case["question"]
    source_times = np.asarray(arrays["t"], float)
    t0 = float(question.get("time", {}).get("t0", source_times[0]))
    t1 = float(question.get("time", {}).get("t1", source_times[-1]))
    if np.any(q < t0-1e-12) or np.any(q > t1+1e-12):
        raise ValueError("query time outside public interval")
    initial, source_moments = _velocity_data(case)
    if source_scales is not None:
        scales = np.asarray(source_scales, float).ravel()
        source_moments = source_moments*np.asarray(
            [scales[min(sum(d), len(scales)-1)] for d in INDICES]
        )[None, :, None, None]
    state = initial.copy()
    nx, ny = state.shape[-2:]
    dx, dy, electric = _spatial_operators(
        nx, ny, question["grid"]["x_domain"])
    nu = float(case["collision_nu"])
    temperature = float(question["temperature"])
    if len(source_times) > 1:
        source_spline = CubicSpline(source_times, source_moments, axis=0)
        source_at = source_spline
    else:
        source_spline = None
        source_at = lambda _t: source_moments[0]

    def gaussian_boundary(y):
        rho = y[POSITION[0, 0]]
        safe = np.maximum(rho, .2)
        ux, uy = y[POSITION[1, 0]]/safe, y[POSITION[0, 1]]/safe
        cxx = y[POSITION[2, 0]]/safe-ux*ux
        cxy = y[POSITION[1, 1]]/safe-ux*uy
        cyy = y[POSITION[0, 2]]/safe-uy*uy
        g = {(0, 0): rho}
        for total in range(1, 5):
            for by in range(total+1):
                ax = total-by
                if ax:
                    value = ux*g[ax-1, by]
                    if ax >= 2:
                        value += (ax-1)*cxx*g[ax-2, by]
                    if by:
                        value += by*cxy*g[ax-1, by-1]
                else:
                    value = uy*g[0, by-1]
                    if by >= 2:
                        value += (by-1)*cyy*g[0, by-2]
                g[ax, by] = value
        excess = float(kurtosis)-1.0
        if excess:
            g[4, 0] += rho*excess*3*cxx*cxx
            g[3, 1] += rho*excess*3*cxx*cxy
            g[2, 2] += rho*excess*(cxx*cyy+2*cxy*cxy)
            g[1, 3] += rho*excess*3*cyy*cxy
            g[0, 4] += rho*excess*3*cyy*cyy
        return g

    def rhs(time, y):
        forcing = source_at(time)
        ex, ey = electric(y[POSITION[0, 0]])
        edge = gaussian_boundary(y)

        def moment(degree):
            return y[POSITION[degree]] if degree in POSITION else edge[degree]

        out = []
        for j, (ax, ay) in enumerate(INDICES):
            value = (forcing[j]-dx(moment((ax+1, ay)))
                     -dy(moment((ax, ay+1))))
            if ax:
                value += ax*ex*moment((ax-1, ay))
            if ay:
                value += ay*ey*moment((ax, ay-1))
            degree = ax+ay
            if degree:
                value -= nu*degree*moment((ax, ay))
                if ax >= 2:
                    value += nu*temperature*ax*(ax-1)*moment((ax-2, ay))
                if ay >= 2:
                    value += nu*temperature*ay*(ay-1)*moment((ax, ay-2))
            if degree == 0:
                # The continuous normalization fixes total mass exactly.  A
                # coarse velocity quadrature otherwise creates a small drift.
                value -= value.mean()
            out.append(value)
        return np.stack(out)

    unique, inverse = np.unique(np.clip(q, t0, t1), return_inverse=True)
    states = np.empty((len(unique),)+state.shape)
    time = t0
    for output_index, target in enumerate(unique):
        target = float(target)
        while time < target-2e-15:
            h = min(float(maximum_step), target-time)
            k1 = rhs(time, state)
            k2 = rhs(time+.5*h, state+.5*h*k1)
            k3 = rhs(time+.5*h, state+.5*h*k2)
            k4 = rhs(time+h, state+h*k3)
            state = state+h/6*(k1+2*k2+2*k3+k4)
            time += h
        states[output_index] = state
    states = states[inverse]
    result = {
        "rho": states[:, POSITION[0, 0]],
        "momentum": np.stack((states[:, POSITION[1, 0]],
                              states[:, POSITION[0, 1]]), axis=-1),
        "kinetic_energy": .5*(states[:, POSITION[2, 0]]
                              + states[:, POSITION[0, 2]]),
    }
    if return_state:
        result["_state"] = states
        result["_initial_state"] = initial
        result["_source_moments"] = source_moments
        result["_source_spline"] = source_spline
    return result


def _canonical_velocity(case, count=9):
    a = case["arrays"]
    vx, vy = np.asarray(a["vx"], float), np.asarray(a["vy"], float)
    tx = np.linspace(vx[0], vx[-1], count)
    ty = np.linspace(vy[0], vy[-1], count)

    def project(value):
        z = np.asarray(value, float)
        if len(vx) != count or not np.allclose(vx, tx):
            z = CubicSpline(vx, z, axis=-2)(tx)
        if len(vy) != count or not np.allclose(vy, ty):
            z = CubicSpline(vy, z, axis=-1)(ty)
        return z
    return project(a["g0"]), project(a["source"])


def build_features(case, query_times, hierarchy_result):
    """Return canonical local feature images, shape (time, channel, x, y)."""
    q = np.asarray(query_times, float)
    state = np.asarray(hierarchy_result["_state"])
    initial = np.asarray(hierarchy_result["_initial_state"])
    source_times = np.asarray(case["arrays"]["t"], float)
    source_moments = np.asarray(hierarchy_result["_source_moments"])
    if len(source_times) > 1:
        spline = CubicSpline(source_times, source_moments, axis=0)
        sm = spline(q)
        sd = spline(q, 1)
    else:
        sm = np.repeat(source_moments[0:1], len(q), axis=0)
        sd = np.zeros_like(sm)
    g0, source = _canonical_velocity(case)
    if len(source_times) > 1:
        sq = CubicSpline(source_times, source, axis=0)(q)
    else:
        sq = np.repeat(source[0:1], len(q), axis=0)
    # Move the two velocity axes into a channel axis.
    nt, nx, ny = len(q), state.shape[-2], state.shape[-1]
    source_channels = np.moveaxis(sq, (-2, -1), (1, 2)).reshape(nt, 81, nx, ny)
    g_channels = np.moveaxis(g0, (-2, -1), (0, 1)).reshape(81, nx, ny)
    logg_channels = np.moveaxis(np.log(np.maximum(g0, 1e-35)),
                                (-2, -1), (0, 1)).reshape(81, nx, ny)
    repeated_initial = np.repeat(initial[None], nt, axis=0)
    repeated_g = np.repeat(g_channels[None], nt, axis=0)
    repeated_logg = np.repeat(logg_channels[None], nt, axis=0)
    temperature = float(case["question"]["temperature"])
    nu = float(case["collision_nu"])
    tau = q/0.2
    scalar = np.stack((tau, np.full(nt, (nu-20)/10),
                       np.full(nt, (temperature-1)/.2), np.exp(-nu*q),
                       np.full(nt, 20/nu)), axis=1)
    scalar = np.broadcast_to(scalar[:, :, None, None], (nt, 5, nx, ny))
    return np.concatenate((state, sm, sd, repeated_initial, source_channels,
                           repeated_g, repeated_logg, scalar), axis=1)
