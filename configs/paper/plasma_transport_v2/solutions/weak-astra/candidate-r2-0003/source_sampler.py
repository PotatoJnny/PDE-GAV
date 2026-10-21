"""Continuous source quadrature confined exactly to the public initial time."""
import numpy as np
from rectangle_rule import rectangle_rule

SOURCE_NX = 24
PILOT_NV = 96
PILOT_VMAX = 16.
PILOT_SITES = 16


def query_blocks(session, st, velocity, limit, check_budget):
    block = max(1, limit//len(velocity))
    for start in range(0, len(st), block):
        cells = st[start:start+block]
        points = np.concatenate([np.repeat(cells, len(velocity), axis=0),
                                 np.tile(velocity, (len(cells), 1))], axis=1)
        parts = []
        for offset in range(0, len(points), limit):
            check_budget()
            batch = points[offset:offset+limit]
            values = np.asarray(session.source(batch), dtype=float)
            if values.shape != (len(batch),) or not np.isfinite(values).all():
                raise ValueError("source must return one finite scalar per point")
            parts.append(values)
            check_budget()
        yield start, np.concatenate(parts).reshape(len(cells), len(velocity))


def sample_source_at_initial(case, session, initial, check_budget):
    question = case.public_data["question"]
    limit = min(262144, question.get("numerical_access", {}).get("max_points_per_request", 262144))
    if type(limit) is not int or limit <= 0:
        raise ValueError("positive request limit required")
    periods, center = initial["periods"], initial["center"]
    t0 = float(initial["interval"][0])
    rng = np.random.default_rng(1729)
    sites = rng.random((PILOT_SITES, 2))*periods
    st = np.column_stack([np.full(PILOT_SITES, t0), sites])
    v = (np.arange(PILOT_NV)-PILOT_NV//2)*2*PILOT_VMAX/PILOT_NV
    fine_v = np.column_stack([a.ravel() for a in np.meshgrid(v, v, indexing="ij")])
    pilot = np.empty((PILOT_SITES, len(fine_v)))
    for start, values in query_blocks(session, st, fine_v+center, limit, check_budget):
        pilot[start:start+len(values)] = values
    rule = rectangle_rule(pilot.reshape(PILOT_SITES, PILOT_NV, PILOT_NV), v, degree=3)
    velocity, weights = np.asarray(rule["velocities"]), np.asarray(rule["weights"])
    pairs = [(a, d-a) for d in range(4) for a in range(d, -1, -1)]
    moment_weights = np.asarray([velocity[:, 0]**a*velocity[:, 1]**b
                                 for a, b in pairs])*weights
    axes = [np.arange(SOURCE_NX)*length/SOURCE_NX for length in periods]
    xy = np.column_stack([a.ravel() for a in np.meshgrid(*axes, indexing="ij")])
    st = np.column_stack([np.full(len(xy), t0), xy])
    result = np.empty((len(xy), 10))
    for start, values in query_blocks(session, st, velocity+center, limit, check_budget):
        result[start:start+len(values)] = values@moment_weights.T
    result = result.T.reshape(10, SOURCE_NX, SOURCE_NX)
    mean_removed = 0.
    if "rho_spatial_mean" in question.get("normalization", {}):
        mean_removed = float(np.mean(result[0]))
        result[0] -= mean_removed
    return result, {"source_time": t0, "source_nx": SOURCE_NX,
                    "source_points": int(len(st)*len(velocity)+pilot.size),
                    "velocity_nodes": len(velocity), "pilot_points": int(pilot.size),
                    "removed_density_source_mean": mean_removed,
                    "rule": rule.get("diagnostics", {})}
