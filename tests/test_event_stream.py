import json
import unittest

from pde_agent_verifier_solver.engine.runtime import EventStream


class EventStreamIntegrityTests(unittest.TestCase):
    def test_emit_input_and_return_value_cannot_mutate_stored_payload(self) -> None:
        source = {
            "candidate": {"artifact_id": "candidate-1", "scores": [1.0, 2.0]},
        }

        emitted = EventStream()
        returned = emitted.emit("candidate_ready", "Candidate ready", payload=source)

        source["candidate"]["artifact_id"] = "changed-through-input"
        source["candidate"]["scores"].append(3.0)
        returned.payload["candidate"]["artifact_id"] = "changed-through-return"
        returned.payload["candidate"]["scores"].clear()

        stored = emitted.snapshot()[0]
        self.assertEqual(stored.payload["candidate"]["artifact_id"], "candidate-1")
        self.assertEqual(stored.payload["candidate"]["scores"], [1.0, 2.0])

    def test_each_subscriber_receives_an_isolated_payload(self) -> None:
        stream = EventStream()
        second_subscriber_payloads = []

        def mutate_first(event) -> None:
            event.payload["round"]["candidate_ids"].append("injected")
            event.payload["round"]["status"] = "mutated"

        def record_second(event) -> None:
            second_subscriber_payloads.append(event.payload)

        stream.subscribe(mutate_first)
        stream.subscribe(record_second)

        returned = stream.emit(
            "round_finished",
            "Round finished",
            payload={
                "round": {
                    "candidate_ids": ["candidate-1"],
                    "status": "completed",
                }
            },
        )

        expected = {
            "round": {
                "candidate_ids": ["candidate-1"],
                "status": "completed",
            }
        }
        self.assertEqual(second_subscriber_payloads, [expected])
        self.assertEqual(returned.payload, expected)
        self.assertEqual(stream.snapshot()[0].payload, expected)

    def test_snapshot_payload_is_detached_and_json_serializable(self) -> None:
        stream = EventStream()
        stream.emit(
            "campaign_memory_updated",
            "Memory updated",
            payload={"memory": {"rounds": [{"round_index": 1}]}},
        )

        first_snapshot = stream.snapshot()
        self.assertIsInstance(first_snapshot[0].payload, dict)
        json.dumps(first_snapshot[0].payload)
        first_snapshot[0].payload["memory"]["rounds"][0]["round_index"] = 99
        first_snapshot[0].payload["memory"]["rounds"].append({"round_index": 2})

        second_snapshot = stream.snapshot()
        self.assertEqual(
            second_snapshot[0].payload,
            {"memory": {"rounds": [{"round_index": 1}]}},
        )


if __name__ == "__main__":
    unittest.main()
