"""Condition-guarded local quintic output projection, with cubic fallback.

This helper consumes only prepared low-moment state snapshots. It performs
no PDE solve or Oracle calls. Independent corner pieces prevent stencils
from crossing a known source-time derivative discontinuity.
"""
from __future__ import annotations

import numpy as np


class GuardedOutputInterpolator:
    def __init__(self, times, values, break_indices, cubic_builder,
                 amplification_limit=128., node_count=8):
        self.times = np.asarray(times, dtype=float)
        self.values = np.asarray(values)
        if (self.times.ndim != 1 or len(self.times) < 2
                or not np.isfinite(self.times).all()
                or np.any(np.diff(self.times) <= 0)
                or len(self.values) != len(self.times)):
            raise ValueError('Ordered times and matching snapshot values required')
        self.cubic = cubic_builder(self.times, self.values, break_indices)
        self.node_count = int(node_count)
        if self.node_count not in (6, 8):
            raise ValueError('Local output interpolation requires six or eight nodes')
        node_count = self.node_count
        bounds = np.unique(np.r_[0, np.asarray(break_indices, dtype=int),
                                  len(self.times)-1])
        self.starts = np.full(len(self.times)-1, -1, dtype=int)
        self.local_counts = np.zeros(len(self.times)-1, dtype=int)
        self.scales = np.ones(len(self.times)-1)
        self.nodes = np.zeros((len(self.times)-1, node_count))
        self.weights = np.zeros((len(self.times)-1, node_count))
        self.amplification_bounds = np.full(len(self.times)-1, np.inf)
        for left, right in zip(bounds[:-1], bounds[1:]):
            if right-left+1 < 6:
                continue
            for interval in range(left, right):
                for count in ([8,6] if node_count == 8 else [6]):
                    if right-left+1 < count:
                        continue
                    start = int(np.clip(interval-(count//2-1), left, right-count+1))
                    selected = self.times[start:start+count]
                    origin = self.times[interval]
                    scale = selected[-1]-selected[0]
                    nodes = (selected-origin)/scale
                    width = (self.times[interval+1]-origin)/scale
                    differences = nodes[:, None]-nodes[None, :]
                    np.fill_diagonal(differences, 1.)
                    # On the complete target interval, each Lagrange basis is
                    # bounded by this product of endpoint-max numerator factors.
                    # Their sum is a rigorous upper bound on the Lebesgue
                    # function (in exact arithmetic), independent of values.
                    numerator = np.maximum(abs(nodes), abs(width-nodes))
                    factors = numerator[None, :]/abs(differences)
                    np.fill_diagonal(factors, 1.)
                    bound = float(np.sum(np.prod(factors, axis=1)))
                    self.amplification_bounds[interval] = bound
                    if not np.isfinite(bound) or bound > amplification_limit:
                        continue
                    weights = 1./np.prod(differences, axis=1)
                    weights /= np.max(abs(weights))
                    self.starts[interval] = start
                    self.local_counts[interval] = count
                    self.scales[interval] = scale
                    self.nodes[interval,:count] = nodes
                    self.weights[interval,:count] = weights
                    break

    def __call__(self, query):
        query = np.asarray(query, dtype=float)
        flat = query.ravel()
        if (not np.isfinite(flat).all()
                or np.any(flat < self.times[0])
                or np.any(flat > self.times[-1])):
            raise ValueError('Output query outside prepared interval')
        output = np.asarray(self.cubic(flat))
        slots = np.clip(np.searchsorted(self.times, flat, side='right')-1,
                        0, len(self.times)-2)
        for slot in np.unique(slots):
            start = self.starts[slot]
            if start < 0:
                continue
            rows = np.flatnonzero(slots == slot)
            q = flat[rows]
            count = self.local_counts[slot]
            nodes = self.nodes[slot,:count]
            z = (q-self.times[slot])/self.scales[slot]
            differences = z[:, None]-nodes[None, :]
            exact = ((q[:, None] == self.times[start:start+count][None, :])
                     | (differences == 0.))
            has_exact = np.any(exact, axis=1)
            if np.any(has_exact):
                output[rows[has_exact]] = self.values[
                    start + np.argmax(exact[has_exact], axis=1)]
            nonexact = ~has_exact
            if np.any(nonexact):
                raw = self.weights[slot,:count][None, :]/differences[nonexact]
                barycentric = raw/np.sum(raw, axis=1, keepdims=True)
                output[rows[nonexact]] = np.einsum('qs,s...->q...', barycentric,
                    self.values[start:start+count], optimize=False)
        # Retain every supplied snapshot exactly, including short pieces
        # and intervals that used the conditioning fallback.
        exact_indices = np.minimum(np.searchsorted(self.times, flat), len(self.times)-1)
        exact_rows = self.times[exact_indices] == flat
        output[exact_rows] = self.values[exact_indices[exact_rows]]
        return output.reshape(query.shape+self.values.shape[1:])

    def diagnostics(self):
        accepted = self.starts >= 0
        return {'degree': self.node_count-1,
                'high_order_intervals': int(np.sum(accepted)),
                'degree7_intervals': int(np.sum(self.local_counts == 8)),
                'degree5_intervals': int(np.sum(self.local_counts == 6)),
                'cubic_intervals': int(np.sum(~accepted)),
                'max_accepted_amplification_bound': (
                    float(np.max(self.amplification_bounds[accepted]))
                    if np.any(accepted) else None)}
