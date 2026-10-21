import copy
import math
import unittest

from pde_agent_verifier_solver.plugins.gkw import (
    GKW_LEGACY_COMPUTE_BUDGET_SCHEMA,
    GKW_MACRO_FIELDS,
    evaluate_gkw_legacy_macro_match,
    validate_gkw_legacy_macro_evidence,
)

from tests.gkw_release_fixture import canonical_gkw_public_components


def _audit_kwargs(case_id: str):
    arrays, scalars, metadata = canonical_gkw_public_components(case_id)
    return {
        "public_arrays": arrays,
        "public_scalars": scalars,
        "public_metadata": metadata,
        "budget": {
            "schema": GKW_LEGACY_COMPUTE_BUDGET_SCHEMA,
            "tier": 1,
            "wall_time_seconds": 30.0,
            "cpus": 2,
            "gpus": 0,
            "memory_bytes": 1024,
        },
        "solver_bundle_sha256": "a" * 64,
        "legacy_evaluation_sha256": "b" * 64,
        "accepted_call_id": "accepted-tier-1",
    }


class GKWLegacyMacroBridgeTests(unittest.TestCase):
    def test_bridge_matches_stable_score_and_persists_only_digests(self) -> None:
        import numpy as np

        truth_value = np.zeros((2, 2, 2), dtype=np.float64)
        truth_value[..., 0] = 1.0
        truth = {name: truth_value.copy() for name in GKW_MACRO_FIELDS}
        prediction = {name: truth_value.copy() for name in GKW_MACRO_FIELDS}
        prediction["particle_density"][0, 0, 0] = 2.0
        legacy = math.sqrt(2.0 / 12.0) / 4.0

        evidence = evaluate_gkw_legacy_macro_match(
            case_id="bridge-case",
            prediction=prediction,
            truth=truth,
            query_times=np.asarray([0.5, 1.0]),
            field_indices=np.asarray([[0, 0, 0], [1, 1, 1]]),
            legacy_aggregate_relative_l2=legacy,
            **_audit_kwargs("bridge-case"),
        )

        self.assertAlmostEqual(evidence["primary_error"], legacy)
        self.assertIs(evidence["used_for_winner_selection"], False)
        self.assertIs(evidence["private_truth_persisted"], False)
        self.assertEqual(
            evidence["binding_authority"],
            "controller_verified_artifacts_required",
        )
        self.assertNotIn("truth", evidence)
        self.assertNotIn("prediction", evidence)
        self.assertEqual(len(evidence["query"]["query_times_sha256"]), 64)
        self.assertEqual(
            len(evidence["commitments"]["accepted_call_sha256"]), 64
        )
        validate_gkw_legacy_macro_evidence(evidence)

        tampered = copy.deepcopy(evidence)
        tampered["metrics"]["phi_relative_l2"] = 1.0
        with self.assertRaisesRegex(ValueError, "incoherent"):
            validate_gkw_legacy_macro_evidence(tampered)

        tampered = copy.deepcopy(evidence)
        particle = tampered["metrics"]["particle_density_relative_l2"]
        tampered["metrics"]["particle_density_relative_l2"] = 0.0
        tampered["metrics"]["phi_relative_l2"] = particle
        with self.assertRaisesRegex(ValueError, "accepted-call commitment"):
            validate_gkw_legacy_macro_evidence(tampered)

        tampered = copy.deepcopy(evidence)
        tampered["passed"] = False
        tampered["failures"] = ["forged failure"]
        with self.assertRaisesRegex(ValueError, "outcome"):
            validate_gkw_legacy_macro_evidence(tampered)

        tampered = copy.deepcopy(evidence)
        tampered["used_for_winner_selection"] = 0
        with self.assertRaisesRegex(ValueError, "contract changed"):
            validate_gkw_legacy_macro_evidence(tampered)

        tampered = copy.deepcopy(evidence)
        tampered["budget"]["tier"] = 2
        with self.assertRaisesRegex(ValueError, "budget commitment"):
            validate_gkw_legacy_macro_evidence(tampered)

    def test_bridge_rejects_incomplete_public_packet(self) -> None:
        import numpy as np

        value = np.asarray([[[1.0, 0.0]]])
        fields = {name: value.copy() for name in GKW_MACRO_FIELDS}
        kwargs = _audit_kwargs("incomplete-case")
        kwargs["public_arrays"] = {
            name: kwargs["public_arrays"][name]
            for name in ("s", "kx", "ky", "B", "ints", "parseval_ky")
        }
        with self.assertRaisesRegex(ValueError, "array keys"):
            evaluate_gkw_legacy_macro_match(
                case_id="incomplete-case",
                prediction=fields,
                truth=fields,
                query_times=np.asarray([1.0]),
                field_indices=np.asarray([[0, 0, 0]]),
                legacy_aggregate_relative_l2=0.0,
                **kwargs,
            )

    def test_bridge_rejects_out_of_interval_query_times(self) -> None:
        import numpy as np

        value = np.ones((2, 1, 2), dtype=np.float64)
        fields = {name: value.copy() for name in GKW_MACRO_FIELDS}
        with self.assertRaisesRegex(ValueError, "public interval"):
            evaluate_gkw_legacy_macro_match(
                case_id="interval-case",
                prediction=fields,
                truth=fields,
                query_times=np.asarray([10.0, 11.0]),
                field_indices=np.asarray([[0, 0, 0]]),
                legacy_aggregate_relative_l2=0.0,
                **_audit_kwargs("interval-case"),
            )

    def test_bridge_rejects_legacy_score_mismatch(self) -> None:
        import numpy as np

        value = np.asarray([[[1.0, 0.0]]])
        fields = {name: value.copy() for name in GKW_MACRO_FIELDS}
        with self.assertRaisesRegex(ValueError, "scores differ"):
            evaluate_gkw_legacy_macro_match(
                case_id="score-case",
                prediction=fields,
                truth=fields,
                query_times=np.asarray([1.0]),
                field_indices=np.asarray([[0, 0, 0]]),
                legacy_aggregate_relative_l2=1.0,
                **_audit_kwargs("score-case"),
            )


if __name__ == "__main__":
    unittest.main()
