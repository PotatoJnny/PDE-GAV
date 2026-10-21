"""Replay an explicitly frozen terminal bank at the normal Generator boundary.

This is a handoff adapter, not fresh sampling. It returns the original accepted
answers in cohort order, including their original children, and never repeats
rejected perturbation attempts. All non-Terminal generation remains unchanged.
No source grid or reference prediction is recomputed to restore the bank.
"""
from collections import Counter

from pde_agent_verifier_solver.sdk import CaseVisibility, GenerationKind, GenerationPurpose, EvaluationPhase


class SavedTerminalReplay:
    def __init__(self, backend, hidden, queries, manifest):
        self.backend, self.hidden, self.queries, self.manifest = backend, hidden, queries, manifest
        self._original = backend.generate
        self._installed = False
        self._issued = set()
        self._offsets = {(0, GenerationKind.BASE): 0, (1, GenerationKind.GUIDED): 0}
        self._roots = {
            (0, GenerationKind.BASE): tuple(hidden["common_base/base"]),
            (1, GenerationKind.GUIDED): tuple(hidden["common_guided/guided"]),
        }
        self._records = {record.case.case_id: record for rows in hidden.values() for record in rows}
        self._children = {}
        for record in self._records.values():
            parent = record.case.parent_case_id
            if parent is not None:
                if parent in self._children or parent not in self._records:
                    raise ValueError("terminal replay requires one saved child per saved parent")
                self._children[parent] = record
        if (len(self._records) != manifest["case_count"] or set(queries) != set(self._records)
                or any(record.case.visibility is not CaseVisibility.HIDDEN for record in self._records.values())):
            raise ValueError("terminal replay inventory is not the original hidden bank")
        for (round_index, kind), rows in self._roots.items():
            if any(row.case.round_index != round_index or row.case.parent_case_id is not None for row in rows):
                raise ValueError("terminal root round/lineage differs from its saved cohort")
        self.calls = 0

    def install(self):
        if self._installed:
            raise RuntimeError("terminal replay is already installed")
        self.backend.generate = self.generate
        self._installed = True
        return self

    def close(self):
        if self._installed:
            self.backend.generate = self._original
            self._installed = False

    def generate(self, base_state, guidance_state, request, progress=None):
        if request.purpose is not GenerationPurpose.TERMINAL:
            return self._original(base_state, guidance_state, request, progress)
        if request.visibility is not CaseVisibility.HIDDEN:
            raise PermissionError("saved terminal answers cannot be issued as Development cases")
        if request.kind is GenerationKind.PERTURBED:
            if len(request.anchors) != request.count:
                raise ValueError("terminal replay needs exactly one anchor per saved child")
            rows = []
            for anchor in request.anchors:
                if anchor.case_id not in self._issued or anchor.round_index != request.round_index:
                    raise ValueError("saved child requested before its original parent")
                try:
                    rows.append(self._children[anchor.case_id])
                except KeyError as error:
                    raise ValueError("terminal parent has no saved child; never recalibrate") from error
            records = tuple(rows)
        else:
            key = (request.round_index, request.kind)
            if key not in self._roots or request.anchors:
                raise ValueError("terminal replay is restricted to the original two cohorts")
            if request.kind is GenerationKind.GUIDED and (
                guidance_state is None or guidance_state.checkpoint_sha256 != self.manifest["first_guidance_sha256"]
            ):
                raise ValueError("saved guided terminal cohort requires the original G1")
            start = self._offsets[key]
            records = self._roots[key][start:start + request.count]
            if len(records) != request.count:
                raise ValueError("saved terminal root inventory exhausted; never resample")
        ids = tuple(record.case.case_id for record in records)
        if len(set(ids)) != len(ids) or self._issued.intersection(ids):
            raise ValueError("saved terminal answer requested more than once")
        if request.kind is not GenerationKind.PERTURBED:
            self._offsets[key] += len(records)
        self._issued.update(ids)
        self.calls += 1
        if progress:
            progress("Restored original accepted terminal answers; no new sampling", 1.)
        return records

    def verify_sealed(self, controller):
        """One handoff-boundary metadata/plan check, not an online Oracle check."""
        from .formal_evaluation_bank import _terminal_records
        current = _terminal_records(controller, self.manifest["cohort_ids"])
        if self._issued != set(self._records) or set(current) != set(self.hidden):
            raise ValueError("the Controller did not seal exactly the saved terminal bank")
        for distribution, rows in self.hidden.items():
            if tuple(row.case.case_id for row in current[distribution]) != tuple(row.case.case_id for row in rows):
                raise ValueError("terminal distribution order changed during handoff")
        seeds = controller._hidden_plan_seed_provider("terminal_evaluation", EvaluationPhase.TERMINAL, current)
        settings = {**controller.campaign.task_settings, "default_query": controller.campaign.evaluator.default_query}
        for rows in current.values():
            for index, record in enumerate(rows):
                query = controller.evaluation.make_plan(record.case, EvaluationPhase.TERMINAL, settings,
                    seed=seeds(index, record.case.case_id)).query
                if query != self.queries[record.case.case_id]:
                    raise ValueError("saved terminal scoring points changed during handoff")
        return {"status": "passed", "source_manifest_sha256": self.manifest["manifest_sha256"],
                "case_count": len(self._records), "distribution_counts": {key: len(rows) for key, rows in current.items()},
                "exact_saved_cases_and_queries": True, "native_draws_recomputed": 0,
                "source_grids_materialized": 0, "reference_values_recomputed": 0,
                "original_rejected_sampling_attempts_replayed": False,
                "issuance_mode": "saved_accepted_terminal_bank_not_fresh_sampling"}


def verify_official_aliases(hidden, expected_batches):
    """Compare the full old/new ordered public cohort identities, not scores."""
    from pde_agent_verifier_solver.engine.candidate_case import candidate_case_id
    actual = tuple(tuple(candidate_case_id(row.case.case_id) for row in rows) for rows in hidden.values())
    expected = tuple(tuple(rows) for rows in expected_batches)
    if Counter(actual) != Counter(expected):
        raise ValueError("recovered Official case identities/order differ from the original cohort")
    return {"status": "passed", "case_count": sum(map(len, actual)), "ordered_batches": len(actual),
            "original_official_alias_order_preserved": True, "old_scores_imported": False,
            "old_full_numerical_outputs_available": False,
            "private_plans": "same_task_config_seed_and_cohort_positions"}
