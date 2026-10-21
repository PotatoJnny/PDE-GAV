"""Pure local initial-data quadrature, with no Oracle or query access.

The seed is a numerical g0 acquisition on a coarse periodic mesh.  Its
interpolated first two moments choose quadrature coordinates only: the fine
moments always come from fresh g0 values at those coordinates.
"""
from __future__ import annotations
import math
import numpy as np

POWERS = tuple((degree-b, b) for degree in range(6) for b in range(degree+1))


def periodic_values(values, xy, periods):
    """Continuous real Fourier interpolation, omitting even-grid Nyquist."""
    values = np.asarray(values, float)
    xy = np.asarray(xy, float)
    periods = np.asarray(periods, float)
    if (values.ndim != 3 or values.shape[-1] != values.shape[-2]
            or xy.ndim != 2 or xy.shape[1] != 2 or periods.shape != (2,)
            or np.any(periods <= 0) or not np.isfinite(values).all()
            or not np.isfinite(xy).all() or not np.isfinite(periods).all()):
        raise ValueError('finite square periodic moment grid and points required')
    n = values.shape[-1]
    modes = np.fft.fft2(values)/n**2
    if n % 2 == 0:
        modes[..., n//2, :] = 0
        modes[..., :, n//2] = 0
    wave = [2*np.pi*np.fft.fftfreq(n, d=length/n) for length in periods]
    bx = np.exp(1j*xy[:, 0, None]*wave[0])
    by = np.exp(1j*xy[:, 1, None]*wave[1])
    return np.einsum('mij,ri,rj->mr', modes, bx, by).real


def _bilinear(values, xy, periods):
    n = values.shape[-1]
    position = np.mod(xy/periods, 1.)*n
    lower = np.floor(position).astype(int)
    fraction = position-lower
    result = np.zeros((len(values), len(xy)))
    for a in [0, 1]:
        for b in [0, 1]:
            weight = (fraction[:, 0] if a else 1-fraction[:, 0])*(fraction[:, 1] if b else 1-fraction[:, 1])
            result += values[:, (lower[:, 0]+a)%n, (lower[:, 1]+b)%n]*weight
    return result


def _mean_covariance(raw):
    mean = raw[1:3].T/raw[0, :, None]
    covariance = np.empty((raw.shape[1], 2, 2))
    covariance[:, 0, 0] = raw[3]/raw[0]-mean[:, 0]**2
    covariance[:, 0, 1] = covariance[:, 1, 0] = raw[4]/raw[0]-mean[:, 0]*mean[:, 1]
    covariance[:, 1, 1] = raw[5]/raw[0]-mean[:, 1]**2
    return mean, covariance


def plan_initial_quadrature(seed_raw, xy, periods, order=32, scale=.65):
    """Return per-site relative velocities and integration weights.

    Fourier interpolation supplies smooth local mean/covariance.  If it loses
    positivity at a site, a convex periodic bilinear interpolation of the raw
    first-two-moment matrix supplies that site's quadrature coordinates.
    This fallback does not replace any requested fine g0 moment.
    """
    seed_raw = np.asarray(seed_raw, float)
    xy = np.asarray(xy, float)
    periods = np.asarray(periods, float)
    if (seed_raw.ndim != 3 or seed_raw.shape[0] < 6 or np.any(seed_raw[0] <= 0)
            or isinstance(order, bool) or int(order) != order or order < 2
            or not np.isfinite(scale) or scale <= 0):
        raise ValueError('positive seed mass, integer order>=2, positive scale required')
    raw = periodic_values(seed_raw[:6], xy, periods)
    bad_mass = raw[0] <= 0
    if np.any(bad_mass):
        raw[:, bad_mass] = _bilinear(seed_raw[:6], xy[bad_mass], periods)
    mean, covariance = _mean_covariance(raw)
    eigen, vectors = np.linalg.eigh(covariance)
    bad = np.any(eigen <= 0, axis=1) | bad_mass
    if np.any(bad):
        raw[:, bad] = _bilinear(seed_raw[:6], xy[bad], periods)
        mean, covariance = _mean_covariance(raw)
        eigen, vectors = np.linalg.eigh(covariance)
    if np.any(eigen <= 0) or not np.isfinite(eigen).all():
        raise ValueError('seed interpolation must have positive velocity covariance')
    transforms = vectors*np.sqrt(2*eigen)[:, None, :]*float(scale)
    nodes, weights = np.polynomial.hermite.hermgauss(int(order))
    normal = np.stack(np.meshgrid(nodes, nodes, indexing='ij'), -1).reshape(-1, 2)
    unweighted = weights*np.exp(nodes**2)
    velocity = mean[:, None, :]+np.einsum('rij,nj->rni', transforms, normal)
    integration = abs(np.linalg.det(transforms))[:, None]*np.outer(unweighted, unweighted).ravel()
    return velocity, integration, {
        'order': int(order), 'scale': float(scale), 'rotated': True,
        'sites': len(xy), 'nodes_per_site': int(order)**2,
        'seed_nx': seed_raw.shape[-1], 'bilinear_planning_fallback_sites': int(bad.sum()),
        'minimum_planning_variance': float(eigen.min()),
        'maximum_planning_variance': float(eigen.max()),
    }


def integrate_initial(values, velocity, integration, rho=None, degree=5):
    """Integrate sampled g0 through degree5 in the fixed relative frame.

    Return moments[components,sites] and the unnormalized sampled mass.
    Public rho0, when supplied, fixes the local normalization exactly.
    """
    values = np.asarray(values, float)
    velocity = np.asarray(velocity, float)
    integration = np.asarray(integration, float)
    if (values.ndim != 2 or velocity.shape != values.shape+(2,)
            or integration.shape != values.shape or degree not in range(6)
            or not np.isfinite(values).all() or not np.isfinite(velocity).all()
            or not np.isfinite(integration).all() or np.any(integration <= 0)):
        raise ValueError('finite compatible values, velocity, integration required')
    weighted = values*integration
    vx, vy = velocity[..., 0], velocity[..., 1]
    moments = np.stack([np.sum(weighted*vx**a*vy**b, axis=1)
                        for a, b in POWERS[:(degree+1)*(degree+2)//2]])
    mass = moments[0].copy()
    if np.any(mass <= 0) or not np.isfinite(moments).all():
        raise ValueError('positive finite initial quadrature mass required')
    if rho is not None:
        rho = np.asarray(rho, float)
        if rho.shape != mass.shape or np.any(rho <= 0) or not np.isfinite(rho).all():
            raise ValueError('positive finite rho0 at every quadrature site required')
        moments *= rho[None]/mass[None]
        moments[0] = rho
    return moments, mass


def _central(raw):
    degree = int((np.sqrt(8*len(raw)+1)-3)/2)
    pairs = POWERS[:len(raw)]
    ids = {pair: i for i, pair in enumerate(pairs)}
    u, v = raw[1]/raw[0], raw[2]/raw[0]
    answer = np.zeros_like(raw)
    for row, (a, b) in enumerate(pairs):
        for i in range(a+1):
            for j in range(b+1):
                answer[row] += (math.comb(a, i)*math.comb(b, j)
                                *(-u)**(a-i)*(-v)**(b-j)*raw[ids[i, j]])
    return answer


def initial_error_metrics(reference, approximation, theta=1., center=None):
    """Degree-group errors plus the actual central transport observables.

    Both raw and central degree4/5 comparisons matter for source cancellation;
    a small initial pressure/heat error alone is not a degree5 certificate.
    The supplied reference is a numerical rule, never asserted exact.
    """
    reference = np.asarray(reference, float)
    approximation = np.asarray(approximation, float)
    center = np.zeros(2) if center is None else np.asarray(center, float)
    if (reference.shape != approximation.shape or reference.ndim != 2
            or reference.shape[0] != 21 or np.any(reference[0] <= 0)
            or np.any(approximation[0] <= 0) or center.shape != (2,)
            or not np.isfinite(reference).all() or not np.isfinite(approximation).all()
            or not np.isfinite(theta) or theta <= 0):
        raise ValueError('finite degree5 moment matrices and positive theta required')
    refcentral, appcentral = _central(reference), _central(approximation)
    result = {'raw_degree_relative_rms': [], 'central_degree_relative_rms': []}
    for name, ref, app in [('raw', reference, approximation), ('central', refcentral, appcentral)]:
        for degree in range(6):
            selected = slice(degree*(degree+1)//2, (degree+1)*(degree+2)//2)
            weight = np.asarray([math.comb(degree, b) for b in range(degree+1)])[:, None]
            error = np.sqrt(np.mean(np.sum(weight*(app[selected]-ref[selected])**2, axis=0)))
            denominator = np.sqrt(np.mean(np.sum(weight*ref[selected]**2, axis=0)))
            floor = 1e-5*float(np.sqrt(np.mean(reference[0]**2)))*theta**(degree/2)
            result[f'{name}_degree_relative_rms'].append(float(error/max(denominator, floor)))
    transports = [
        ('current', reference[1:3]+reference[0]*center[:, None],
         approximation[1:3]+approximation[0]*center[:, None], np.ones((2, 1))),
        ('pressure', refcentral[3:6], appcentral[3:6], np.asarray([1, 2, 1])[:, None]),
        ('heat', .5*np.stack((refcentral[6]+refcentral[8], refcentral[7]+refcentral[9])),
         .5*np.stack((appcentral[6]+appcentral[8], appcentral[7]+appcentral[9])), np.ones((2, 1)))]
    for name, ref, app, weight in transports:
        error = np.sqrt(np.mean(np.sum(weight*(app-ref)**2, axis=0)))
        denominator = np.sqrt(np.mean(np.sum(weight*ref**2, axis=0)))
        result[name] = {'physical_rms_error': float(error), 'reference_rms': float(denominator),
                        'relative_rms_error': float(error/max(denominator, .001))}
    return result
