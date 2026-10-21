"""Drop-in helpers for public-limit-safe initial sampling and mass repair."""
from __future__ import annotations
import numpy as np
from initial_quadrature import integrate_initial


def local_initial(sites, nodes, weights, rho, *, session, center, limit, check):
    """Sample every requested local velocity, splitting even one large rule.

    Return raw relative moments through total degree five and unnormalized g0
    masses. Request slicing never changes the site-major quadrature ordering.
    """
    if type(limit) is not int or limit <= 0:
        raise ValueError('positive public request limit required')
    count = nodes.shape[1]
    block = max(1, limit//count)
    parts = []
    masses = []
    for offset in range(0, len(sites), block):
        check()
        stop = min(offset+block, len(sites))
        local = nodes[offset:stop]
        points = np.column_stack((np.repeat(sites[offset:stop], count, axis=0),
                                  (local+center).reshape(-1, 2)))
        responses = []
        for start in range(0, len(points), limit):
            check()
            batch = points[start:start+limit]
            response = np.asarray(session.g0(batch), float)
            if response.shape != (len(batch),) or not np.isfinite(response).all():
                raise ValueError('finite scalar g0 value per requested point required')
            responses.append(response)
        values = np.concatenate(responses).reshape(stop-offset, count)
        raw, mass = integrate_initial(values, local, weights[offset:stop],
                                      rho[offset:stop], degree=5)
        parts.append(raw)
        masses.append(mass)
    check()
    return np.concatenate(parts, axis=1), np.concatenate(masses)


def repair_initial_mass(initial, mass, sites, rho, *, uniform_initial, check,
                        tolerance=.003):
    """Replace moments at production cells violating the validation mass gate.

    `initial` is updated in place, and only the offending cells are queried.
    A valid integral of g0 is one, independent of local public density rho.
    This inexpensive diagnostic does not certify accuracy of higher moments.
    """
    mass = np.asarray(mass, float)
    if mass.shape != (len(sites),) or not np.isfinite(mass).all():
        raise ValueError('finite initial quadrature mass at every site required')
    bad = np.abs(mass-1.) > tolerance
    count = int(np.sum(bad))
    maximum_error = float(np.max(np.abs(mass-1.)))
    if count:
        check()
        repaired = uniform_initial(sites[bad], rho[bad])
        if repaired.shape != (initial.shape[0], count) or not np.isfinite(repaired).all():
            raise ValueError('finite compatible repaired initial moments required')
        initial[:, bad] = repaired
        check()
    return {'initial_mass_repaired_cells': count,
            'initial_production_maximum_mass_error': maximum_error}
