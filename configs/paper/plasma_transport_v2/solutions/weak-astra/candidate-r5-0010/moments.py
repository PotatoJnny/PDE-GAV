"""Initial moment hierarchy in a fixed velocity frame.

Pure numerical helpers; there is no Oracle access in this module.
"""
from __future__ import annotations
import numpy as np

INDICES = tuple((a, degree-a) for degree in range(5) for a in range(degree, -1, -1))
INDEX = {pair: i for i, pair in enumerate(INDICES)}


def project(values):
    n = values.shape[-1]
    modes = np.fft.fft2(values, axes=(-2, -1))/n**2
    if n % 2 == 0:
        modes[..., n//2, :] = 0
        modes[..., :, n//2] = 0
    return modes


def electric_modes(rho_modes, periods):
    n = rho_modes.shape[-1]
    kx, ky = np.meshgrid(*(2*np.pi*np.fft.fftfreq(n, d=l/n) for l in periods), indexing='ij')
    k2 = kx*kx+ky*ky
    inv = np.divide(1., k2, out=np.zeros_like(k2), where=k2 > 0)
    return np.stack((-1j*kx*inv*rho_modes, -1j*ky*inv*rho_modes))


def evaluate_grid(modes, n):
    """Evaluate a normalized Fourier polynomial on a larger uniform grid."""
    m = modes.shape[-1]
    if n < m:
        raise ValueError('oversampling must retain every initial Fourier mode')
    frequencies = np.rint(np.fft.fftfreq(m)*m).astype(int)
    out = np.zeros(modes.shape[:-2]+(n, n), complex)
    out[..., frequencies[:, None] % n, frequencies[None, :] % n] = modes
    return np.fft.ifft2(out, axes=(-2, -1)).real*n*n


def resample_modes(modes, n):
    m = modes.shape[-1]
    freqs = np.rint(np.fft.fftfreq(n)*n).astype(int)
    out = modes[..., freqs[:, None] % m, freqs[None, :] % m].copy()
    if n % 2 == 0:
        out[..., n//2, :] = 0
        out[..., :, n//2] = 0
    return out


def hierarchy(initial_modes, source_modes, periods, center, nu, theta):
    """Return derivatives of all raw relative moments through total degree3.

    d M_ab = -div(M_(a+1,b)+c_x M_ab, M_(a,b+1)+c_y M_ab)
             + a (E_x-nu*c_x) M_(a-1,b) + b (E_y-nu*c_y) M_(a,b-1)
             -nu*(a+b) M_ab + nu*theta*(a*(a-1) M_(a-2,b)+...)+S_ab.
    Products use 2x spatial oversampling to avoid quadratic aliasing.
    """
    n = initial_modes.shape[-1]
    if initial_modes.shape != (15, n, n) or source_modes.shape != (10, n, n):
        raise ValueError('degree4 initial and degree3 source Fourier arrays required')
    kx, ky = np.meshgrid(*(2*np.pi*np.fft.fftfreq(n, d=l/n) for l in periods), indexing='ij')
    initial = evaluate_grid(initial_modes, 2*n)
    field = evaluate_grid(electric_modes(initial_modes[0], periods), 2*n)
    force = field-nu*np.asarray(center)[:, None, None]
    derivative = source_modes.copy()
    for i, (a, b) in enumerate(INDICES[:10]):
        m = initial_modes[i]
        derivative[i] -= 1j*kx*(initial_modes[INDEX[a+1, b]]+center[0]*m)
        derivative[i] -= 1j*ky*(initial_modes[INDEX[a, b+1]]+center[1]*m)
        derivative[i] -= nu*(a+b)*m
        product = np.zeros((2*n, 2*n))
        if a:
            product += a*force[0]*initial[INDEX[a-1, b]]
        if b:
            product += b*force[1]*initial[INDEX[a, b-1]]
        derivative[i] += resample_modes(project(product), n)
        if a > 1:
            derivative[i] += nu*theta*a*(a-1)*initial_modes[INDEX[a-2, b]]
        if b > 1:
            derivative[i] += nu*theta*b*(b-1)*initial_modes[INDEX[a, b-2]]
    derivative[..., n//2, :] = 0
    derivative[..., :, n//2] = 0
    return derivative


def central_values(raw, center):
    """rho, laboratory momentum, central pressure, central heat, no field."""
    r, x, y, xx, xy, yy, xxx, xxy, xyy, yyy = raw[:10]
    ux, uy = x/r, y/r
    speed = ux*ux+uy*uy
    return np.stack((r, x+r*center[0], y+r*center[1],
        xx-x*ux, xy-x*uy, yy-y*uy,
        .5*(xxx+xyy)-ux*(1.5*xx+.5*yy)-uy*xy+r*ux*speed,
        .5*(xxy+yyy)-uy*(.5*xx+1.5*yy)-ux*xy+r*uy*speed))


def central_tangent(raw, slope, center):
    """Exact Jacobian of own-velocity centralization, including mixed thirds."""
    r, x, y, xx, xy, yy, xxx, xxy, xyy, yyy = raw[:10]
    dr, dx, dy, dxx, dxy, dyy, dxxx, dxxy, dxyy, dyyy = slope[:10]
    ux, uy = x/r, y/r
    dux, duy = (dx-ux*dr)/r, (dy-uy*dr)/r
    speed, dspeed = ux*ux+uy*uy, 2*(ux*dux+uy*duy)
    dqx = (.5*(dxxx+dxyy)-dux*(1.5*xx+.5*yy)-ux*(1.5*dxx+.5*dyy)
           -duy*xy-uy*dxy+dr*ux*speed+r*dux*speed+r*ux*dspeed)
    dqy = (.5*(dxxy+dyyy)-duy*(.5*xx+1.5*yy)-uy*(.5*dxx+1.5*dyy)
           -dux*xy-ux*dxy+dr*uy*speed+r*duy*speed+r*uy*dspeed)
    return np.stack((dr, dx+center[0]*dr, dy+center[1]*dr,
        dxx-dx*ux-x*dux, dxy-dx*uy-x*duy, dyy-dy*uy-y*duy, dqx, dqy))
