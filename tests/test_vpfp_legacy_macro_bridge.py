import json
import math
import unittest

import numpy as np

from pde_agent_verifier_solver.plugins.vpfp.legacy_macro_bridge import (
    LEGACY_MACRO_BRIDGE_SCHEMA,
    evaluate_legacy_macro_arrays,
    legacy_macro_arrays_to_answer_view,
)


class VPFPLegacyMacroBridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.query_times = np.array([0.0, 0.2, 1.1])
        self.x = np.array([-2.0, -0.25])
        self.y = np.array([0.0, 0.4, 2.0, 5.5])
        self.shape = (3, 2, 4)
        self.rho = np.ones(self.shape)
        self.momentum = np.ones(self.shape + (2,))
        self.energy = np.ones(self.shape)

    def _evaluate(self, **changes):
        arguments = {
            "case_id": "legacy-vpfp-case",
            "query_times": self.query_times,
            "x": self.x,
            "y": self.y,
            "prediction_rho": self.rho,
            "prediction_momentum": self.momentum,
            "prediction_kinetic_energy": self.energy,
            "truth_rho": self.rho,
            "truth_momentum": self.momentum,
            "truth_kinetic_energy": self.energy,
        }
        arguments.update(changes)
        return evaluate_legacy_macro_arrays(**arguments)

    def test_exact_match_returns_versioned_json_safe_diagnostic(self) -> None:
        payload = self._evaluate()

        self.assertEqual(payload["schema"], LEGACY_MACRO_BRIDGE_SCHEMA)
        self.assertEqual(payload["version"], 1)
        self.assertEqual(
            payload["contract"],
            "vpfp_full_velocity_macro_match_v1",
        )
        self.assertEqual(payload["axis_order"], ["t", "x", "y"])
        self.assertEqual(payload["shape"], [3, 2, 4])
        self.assertRegex(payload["coordinate_digest"], r"^sha256:[0-9a-f]{64}$")
        self.assertEqual(payload["primary_error"], 0.0)
        self.assertEqual(
            payload["metrics"]["vpfp_macro_match_v1_relative_l2"],
            0.0,
        )
        self.assertIsNone(payload["passed"])
        self.assertEqual(payload["failures"], [])
        self.assertIs(payload["used_for_winner_selection"], False)
        json.dumps(payload, allow_nan=False)

    def test_vector_momentum_uses_one_matcher_physical_group(self) -> None:
        prediction = self.momentum.copy()
        prediction[..., 0] = 2.0

        payload = self._evaluate(prediction_momentum=prediction)

        expected_momentum = 1.0 / math.sqrt(2.0)
        self.assertAlmostEqual(
            payload["metrics"]["momentum_relative_l2"],
            expected_momentum,
        )
        self.assertAlmostEqual(
            payload["primary_error"],
            expected_momentum / 3.0,
        )
        self.assertAlmostEqual(
            payload["metrics"]["legacy_four_scalar_macro_relative_l2"],
            0.25,
        )

    def test_irregular_coordinates_are_preserved_and_digest_bound(self) -> None:
        view = legacy_macro_arrays_to_answer_view(
            rho=self.rho,
            momentum=self.momentum,
            kinetic_energy=self.energy,
            query_times=self.query_times,
            x=self.x,
            y=self.y,
        )
        original = self._evaluate()
        changed = self._evaluate(
            query_times=np.array([0.0, 0.3, 1.1]),
        )

        self.assertEqual(view.axes["t"], (0.0, 0.2, 1.1))
        self.assertEqual(view.axes["x"], (-2.0, -0.25))
        self.assertEqual(view.axes["y"], (0.0, 0.4, 2.0, 5.5))
        self.assertEqual(view.shape, self.shape)
        self.assertNotEqual(
            original["coordinate_digest"],
            changed["coordinate_digest"],
        )

    def test_bad_shapes_nonfinite_values_and_unordered_axes_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "momentum shape"):
            self._evaluate(
                prediction_momentum=np.ones(self.shape + (3,)),
            )
        with self.assertRaisesRegex(ValueError, "rho shape"):
            self._evaluate(prediction_rho=np.ones((3, 2, 3)))

        nonfinite = self.energy.copy()
        nonfinite[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "only finite"):
            self._evaluate(prediction_kinetic_energy=nonfinite)

        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            self._evaluate(query_times=np.array([0.0, 0.5, 0.5]))
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            self._evaluate(x=np.array([-0.25, -2.0]))
        with self.assertRaisesRegex(ValueError, "axis y must be finite"):
            self._evaluate(y=np.array([0.0, 0.4, np.inf, 5.5]))

        with self.assertRaisesRegex(ValueError, "nonempty case_id"):
            self._evaluate(case_id=" ")


if __name__ == "__main__":
    unittest.main()
