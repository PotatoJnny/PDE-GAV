import unittest
from collections.abc import Mapping

from pde_agent_verifier_solver.plugins.gkw import (
    GKWLegacyObservableAdapter,
    GKW_MACRO_FIELDS,
)
from pde_agent_verifier_solver.sdk import AnswerQuery

from tests.test_gkw_legacy_adapter import _Session, _budget, _case


class _Runner:
    def __call__(self, solver, session, times, indices, budget):
        return solver(session, times, indices, budget), 0.1


class _TwoReadMapping(Mapping):
    def __init__(self, value):
        self.value = value
        self.reads = {name: 0 for name in GKW_MACRO_FIELDS}

    def __iter__(self):
        return iter(GKW_MACRO_FIELDS)

    def __len__(self):
        return len(GKW_MACRO_FIELDS)

    def __getitem__(self, name):
        import numpy as np

        self.reads[name] += 1
        if self.reads[name] == 1:
            return self.value.copy()
        unsafe = np.empty(self.value.shape, dtype=object)
        unsafe[:] = "second-read payload"
        return unsafe


class GKWAdapterAdversarialTests(unittest.TestCase):
    def test_dtype_guard_returns_first_read_plain_numeric_copies(self) -> None:
        import numpy as np

        returned = []

        def submission(session, times, indices, budget):
            del session, budget
            mapping = _TwoReadMapping(
                np.ones((len(times), len(indices), 2), dtype=np.float64)
            )
            returned.append(mapping)
            return mapping

        adapter = GKWLegacyObservableAdapter(
            submission,
            _budget(),
            _Runner(),
            lambda case: _Session(case.case_id),
        )
        answer = adapter.predict_observables(
            _case(), AnswerQuery.grid(fields=GKW_MACRO_FIELDS, t=2, field_row=2)
        )
        self.assertEqual(answer.shape, (2, 2, 2))
        self.assertEqual(set(returned[0].reads.values()), {1})

    def test_full_scalar_and_oracle_protocol_binding_fails_closed(self) -> None:
        query = AnswerQuery.grid(fields=GKW_MACRO_FIELDS, t=2, field_row=2)

        def wrong_scalar(case):
            session = _Session(case.case_id)
            session.scalars["beta"] = 2.0
            return session

        adapter = GKWLegacyObservableAdapter(
            lambda session, times, indices, budget: {},
            _budget(),
            _Runner(),
            wrong_scalar,
        )
        with self.assertRaisesRegex(ValueError, "scalar beta"):
            adapter.predict_observables(_case(), query)

        def wrong_oracle(case):
            session = _Session(case.case_id)
            session.oracle_protocol = {"time_interval_closed": [0.0, 2.0]}
            return session

        adapter = GKWLegacyObservableAdapter(
            lambda session, times, indices, budget: {},
            _budget(),
            _Runner(),
            wrong_oracle,
        )
        with self.assertRaisesRegex(ValueError, "oracle protocol"):
            adapter.predict_observables(_case(), query)


if __name__ == "__main__":
    unittest.main()
