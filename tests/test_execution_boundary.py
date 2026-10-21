import hashlib
import inspect
import unittest
from dataclasses import fields

from pde_agent_verifier_solver.sdk import (
    AgentBundleSubmission,
    ControllerOwnedBundleAttestor,
    ExecutionAssurance,
    FormalExecutionCapabilities,
    FormalExecutionUnavailable,
    GeneratorSnapshot,
    candidate_ref_from_attestation,
    require_execution_assurance,
)


class _MemoryAttestor(ControllerOwnedBundleAttestor):
    def __init__(self, payloads):
        self.payloads = dict(payloads)
        self.sealed = {}

    @property
    def attestor_id(self):
        return "controller-memory-store-v1"

    def _read_untrusted_bundle(self, bundle_uri):
        return self.payloads[bundle_uri]

    def _seal_verified_bundle(self, bundle_sha256, payload):
        uri = f"controller://sha256/{bundle_sha256}"
        self.sealed[uri] = payload
        return uri


class ExecutionBoundaryTests(unittest.TestCase):
    def test_reference_assurance_is_explicitly_not_formal(self):
        boundary = require_execution_assurance(
            ExecutionAssurance.REFERENCE_IN_PROCESS
        )

        self.assertEqual(
            boundary.assurance,
            ExecutionAssurance.REFERENCE_IN_PROCESS,
        )
        self.assertFalse(boundary.formal_claims_allowed)
        self.assertTrue(boundary.limitations)

    def test_formal_request_fails_closed_and_has_no_marker_escape_hatch(self):
        self.assertEqual(
            tuple(inspect.signature(require_execution_assurance).parameters),
            ("requested",),
        )
        with self.assertRaisesRegex(
            FormalExecutionUnavailable,
            "formal_isolated execution is not wired",
        ):
            require_execution_assurance(ExecutionAssurance.FORMAL_ISOLATED)

        # A capability-shaped record is a design contract, not authorization.
        attestor = _MemoryAttestor({"agent://candidate": b"candidate bytes"})
        capabilities = FormalExecutionCapabilities(
            bundle_attestor=attestor,
            agent_executor=object(),
            candidate_executor=object(),
            generator_replay=object(),
        )
        self.assertIs(capabilities.bundle_attestor, attestor)
        with self.assertRaises(FormalExecutionUnavailable):
            require_execution_assurance(ExecutionAssurance.FORMAL_ISOLATED)

    def test_assurance_does_not_accept_string_downgrades(self):
        with self.assertRaises(TypeError):
            require_execution_assurance("reference_in_process")

    def test_agent_submission_cannot_declare_a_bundle_digest(self):
        self.assertEqual(
            tuple(field.name for field in fields(AgentBundleSubmission)),
            ("candidate_id", "bundle_uri"),
        )

    def test_controller_computes_and_seals_candidate_identity(self):
        payload = b"immutable candidate bundle\x00v1"
        expected = hashlib.sha256(payload).hexdigest()
        attestor = _MemoryAttestor({"agent://candidate": payload})

        record = attestor.attest(
            AgentBundleSubmission("candidate-1", "agent://candidate"),
            round_index=2,
            agent_id="agent-a",
            parent_sha256="a" * 64,
        )

        self.assertEqual(record.bundle_sha256, expected)
        self.assertEqual(record.byte_length, len(payload))
        self.assertEqual(
            attestor.sealed[record.sealed_bundle_uri],
            payload,
        )
        ref = candidate_ref_from_attestation(record)
        self.assertEqual(ref.bundle_sha256, expected)
        self.assertEqual(ref.bundle_uri, record.sealed_bundle_uri)
        self.assertEqual(record.attestor_id, attestor.attestor_id)

    def test_attestor_rejects_empty_or_nonbyte_bundle(self):
        empty = _MemoryAttestor({"agent://empty": b""})
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            empty.attest(
                AgentBundleSubmission("empty", "agent://empty"),
                round_index=1,
                agent_id="agent-a",
                parent_sha256=None,
            )

        wrong_type = _MemoryAttestor({"agent://text": "not bytes"})
        with self.assertRaisesRegex(TypeError, "must return bytes"):
            wrong_type.attest(
                AgentBundleSubmission("text", "agent://text"),
                round_index=1,
                agent_id="agent-a",
                parent_sha256=None,
            )

    def test_attestation_algorithm_cannot_be_overridden_by_adapter(self):
        with self.assertRaisesRegex(TypeError, "cannot be overridden"):

            class _UnsafeAttestor(ControllerOwnedBundleAttestor):
                def attest(self, *args, **kwargs):
                    return None

    def test_generator_snapshot_requires_artifact_and_plan_digests(self):
        snapshot = GeneratorSnapshot(
            snapshot_id="snapshot-1",
            generator_id="generator-1",
            sealed_artifact_uri="controller://generator/1",
            artifact_sha256="b" * 64,
            cohort_plan_sha256="c" * 64,
        )
        self.assertEqual(snapshot.artifact_sha256, "b" * 64)

        with self.assertRaisesRegex(ValueError, "cohort plan sha256"):
            GeneratorSnapshot(
                snapshot_id="snapshot-1",
                generator_id="generator-1",
                sealed_artifact_uri="controller://generator/1",
                artifact_sha256="b" * 64,
                cohort_plan_sha256="agent-declared",
            )


if __name__ == "__main__":
    unittest.main()
