"""Fixed engineering dispatch between two existing numerical source kernels.

Threshold 16 is an explicitly recorded workload heuristic, not a physical
threshold or a claim of optimality. No query points or answers are cached,
filtered, rounded, completed into grids, or changed by this dispatcher.
"""
from __future__ import annotations

import numpy as np
import torch

from .moving_frame_batched_source import MovingFrameSourceEvaluator, _finite_fields
from .moving_frame_fast_source import manufacture_moving_frame_source_fast


SOURCE_POLICY = 'sparse_cold_fast_v1_else_cached_or_repeated_txy_batched_v2'
REPEATED_TXY_THRESHOLD = 16
FAST_SOURCE_CHUNK_SIZE = 32
_ROUTES = ('warm_v2', 'repeated_v2', 'cold_fast_v1')


class MovingFrameAdaptiveSourceEvaluator(MovingFrameSourceEvaluator):
    """One cloned immutable case; same flat ABI and fallback as the parent.

    Dispatch is per exact FP64 t/x/y group within each bounded phase block.
    A cached value WITHOUT first jets is cold. Warm groups run before any new
    geometry so LRU insertions cannot evict their selected entries. Source
    exceptions propagate: there is no retry, alternate answer, or dropped row.
    """

    SOURCE_POLICY = SOURCE_POLICY
    REPEATED_TXY_THRESHOLD = REPEATED_TXY_THRESHOLD
    FAST_SOURCE_CHUNK_SIZE = FAST_SOURCE_CHUNK_SIZE

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._adaptive_stats = {'adaptive_source_calls':0, 'adaptive_source_failures':0,
                                'adaptive_coordinate_graph_calls':0}
        self._adaptive_stats.update({f'{route}_{stage}_points':0 for route in _ROUTES
                                     for stage in ('attempted', 'completed')})

    def _routing(self, points):
        exact = np.ascontiguousarray(points[:,:3].detach().cpu().numpy(), dtype=np.float64)
        unique, inverse, counts = np.unique(exact, axis=0, return_inverse=True, return_counts=True)
        warm = np.asarray([((entry := self._cache.get(row.tobytes())) is not None
                             and entry[1] is not None) for row in unique], dtype=bool)
        repeated = ~warm & (counts >= REPEATED_TXY_THRESHOLD)
        # flatnonzero preserves request order within each partition, including
        # duplicate phase points. Every row belongs to exactly one partition.
        return (np.flatnonzero(warm[inverse]), np.flatnonzero(repeated[inverse]),
                np.flatnonzero((~warm & ~repeated)[inverse]))

    @staticmethod
    def _scatter(output, values, indices, count):
        _finite_fields(values)
        if not output:
            output.update({key: value.new_empty((count, *value.shape[1:])) for key, value in values.items()})
        if set(values) != set(output):
            raise RuntimeError('source kernels returned different field sets')
        for key, value in values.items():
            if (value.shape != (len(indices), *output[key].shape[1:])
                    or value.dtype != output[key].dtype or value.device != output[key].device):
                raise RuntimeError(f'source kernel field ABI differs: {key}')
            output[key].index_copy_(0, indices, value.detach())

    def source(self, points, *, create_graph=False):
        if create_graph:
            self._adaptive_stats['adaptive_coordinate_graph_calls'] += 1
            return super().source(points, create_graph=True)
        self._adaptive_stats['adaptive_source_calls'] += 1
        try:
            with torch.no_grad():
                query = self._points(points, 5).detach()
                self._stats['requested_phase_points'] += len(query)
                outputs = []
                for part in query.split(self.phase_chunk_size):
                    output = {}
                    for route, positions in zip(_ROUTES, self._routing(part), strict=True):
                        step = FAST_SOURCE_CHUNK_SIZE if route == 'cold_fast_v1' else self.phase_chunk_size
                        for start in range(0, len(positions), step):
                            indices = torch.as_tensor(positions[start:start+step], device=part.device, dtype=torch.long)
                            selected = part.index_select(0, indices)
                            self._adaptive_stats[f'{route}_attempted_points'] += len(selected)
                            if route == 'cold_fast_v1':
                                values = manufacture_moving_frame_source_fast(self.decoder, selected,
                                    self._carrier, self._condition, create_graph=False)
                            else:
                                values = self._velocity(selected, *self._lookup(selected[:,:3], True))
                            self._scatter(output, values, indices, len(part))
                            self._adaptive_stats[f'{route}_completed_points'] += len(selected)
                            self._stats['assembled_phase_points'] += len(selected)
                    outputs.append(output)
                return _finite_fields({key:torch.cat([row[key] for row in outputs]) for key in outputs[0]})
        except Exception:
            self._adaptive_stats['adaptive_source_failures'] += 1
            raise

    def cache_info(self):
        return {**super().cache_info(), **self._adaptive_stats, 'source_policy':SOURCE_POLICY,
                'repeated_txy_threshold':REPEATED_TXY_THRESHOLD, 'fast_source_chunk_size':FAST_SOURCE_CHUNK_SIZE,
                'routing_scope':'exact_txy_groups_within_each_phase_chunk; warm_jets_before_new_geometry',
                'threshold_role':'fixed_engineering_workload_heuristic_not_physics_or_optimality'}
