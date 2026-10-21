"""Controller-private exact off-grid macros for the aligned Artemis task.

Native arrays and source generation stay unchanged. New scoring coordinates
use the same field, phase, sensor map and tail-safe quadrature as production.
The shared evaluator caches each finite hidden target across Candidates.
"""
from dataclasses import dataclass, replace
import importlib

from pde_agent_verifier_solver.sdk import AnswerView, QueryKind
from .problem import VP1D_MACRO_FIELDS
from .concurrency import serialized_artemis_compute


@dataclass(frozen=True)
class ArtemisContinuousMacroTruth:
    native: object
    runtime: object
    manufactured: object

    def answer(self, query):
        if set(query.fields) != set(VP1D_MACRO_FIELDS):
            raise ValueError("Artemis truth requires M0,M1,M2,M3,E")
        if query.kind is QueryKind.GRID or all(
            point["t"] in self.native.t and point["x"] in self.native.x
            for point in query.points
        ):
            return self.native.answer(query)
        return self._continuous_answer(query)

    @serialized_artemis_compute
    def _continuous_answer(self, query):
        import numpy as np
        import torch

        rows = np.asarray([(p["t"], p["x"]) for p in query.points], dtype=np.float64)
        if rows.shape != (len(query.points), 2) or not len(rows) or not np.isfinite(rows).all():
            raise ValueError("Artemis truth coordinates must be nonempty finite t,x rows")
        runtime = self.runtime
        handle = runtime._exact_query_handle(self.manufactured)
        field = handle.field
        contract = runtime.contract
        if (np.any(rows[:, 0] < contract.t0) or np.any(rows[:, 0] > contract.t1)
                or np.any(rows[:, 1] < 0.0) or np.any(rows[:, 1] >= contract.x_length)):
            raise ValueError("Artemis truth query leaves its physical domain")
        module = importlib.import_module(type(runtime).__module__)
        velocity = runtime._quadrature_v_tensor
        mapping = None if handle.curvatures is None else runtime._sensor_map(velocity, handle.curvatures)
        chunks = {name: [] for name in VP1D_MACRO_FIELDS}
        # Bounded GPU batches: never materialize a larger public t*x*v array.
        for start in range(0, len(rows), 16):
            coordinates = torch.as_tensor(rows[start:start + 16], dtype=field.dtype, device=field.device)
            jet = runtime._factorized_tx_jet(
                field, coordinates[:, 0], coordinates[:, 1], handle.spatial_phase)
            moments = jet.moments
            if mapping is not None:
                parent = field.distribution_from_tx_jet(jet, velocity)
                child = field.distribution_from_tx_jet(
                    jet, mapping.transformed_velocity) * mapping.derivative
                moments = module.delta_corrected_child_moments_from_views(
                    jet, child, parent, velocity, runtime._quadrature_weight_tensor)
            for name in VP1D_MACRO_FIELDS:
                chunks[name].extend(getattr(moments, name).detach().cpu().reshape(-1).tolist())
        return AnswerView(kind=query.kind, points=query.points, shape=(len(rows),),
                          fields={name: tuple(chunks[name]) for name in query.fields})


def continuous_record(backend, record, *, manufactured=None):
    """Wrap only this new profile; never expose the truth through public sessions."""
    if not getattr(backend, "continuous_macro_truth", False):
        return record
    if isinstance(record.truth, ArtemisContinuousMacroTruth):
        return record
    if manufactured is None:
        manufactured = backend.source._issued_records[record.case.case_id].manufactured
    return replace(record, truth=ArtemisContinuousMacroTruth(
        record.truth, backend.source.runtime, manufactured))
