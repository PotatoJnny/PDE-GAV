import json
import unittest
from unittest.mock import patch

from pde_agent_verifier_solver.panel.publisher import (
    PANEL_SNAPSHOT_ENDPOINT,
    UnsafePanelSnapshot,
    build_combined_hosted_snapshot,
    publish_hosted_snapshot,
    validate_hosted_snapshot,
)


class _Response:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return b'{"ok":true}'


class PanelPublisherTests(unittest.TestCase):
    def test_publishes_only_to_the_fixed_panel_with_bearer_auth(self) -> None:
        snapshot = {
            "schema": "pde_panel_snapshot_v1",
            "synced_at": "2026-08-20T00:00:00+00:00",
            "counts": {"running": 1, "completed": 0, "failed": 0, "submitted": 0},
            "runs": [
                {
                    "id": "ns-live",
                    "name": "NS live",
                    "status": "running",
                    "progress_percent": 42,
                    "summary": {"hidden_error": 0.2},
                }
            ],
        }
        captured = {}

        def opener(request, timeout):
            captured["url"] = request.full_url
            captured["authorization"] = request.get_header("Authorization")
            captured["user_agent"] = request.get_header("User-agent")
            captured["body"] = json.loads(request.data)
            captured["timeout"] = timeout
            return _Response()

        with patch(
            "pde_agent_verifier_solver.panel.publisher.build_hosted_dashboard_snapshot",
            return_value=snapshot,
        ):
            published = publish_hosted_snapshot("/unused", "panel-secret", opener=opener)

        self.assertEqual(captured["url"], PANEL_SNAPSHOT_ENDPOINT)
        self.assertEqual(captured["user_agent"], "PDE-Agent-Panel-Sync/1.0")
        self.assertEqual(captured["authorization"], "Bearer panel-secret")
        self.assertEqual(captured["body"]["runs"], snapshot["runs"])
        self.assertEqual(captured["body"]["counts"], snapshot["counts"])
        self.assertEqual(published, captured["body"])
        self.assertEqual(captured["timeout"], 30)

    def test_combines_storage_roots_and_keeps_the_freshest_duplicate(self) -> None:
        snapshots = {
            "project-runs": {
                "schema": "pde_panel_snapshot_v1",
                "runs": [
                    {
                        "id": "formal",
                        "problem_id": "ns",
                        "status": "running",
                        "updated_at": "2026-08-21T02:00:00+00:00",
                    },
                    {
                        "id": "smoke",
                        "problem_id": "ns",
                        "status": "completed",
                        "updated_at": "2026-08-20T01:00:00+00:00",
                    },
                ],
            },
            "unified-runs": {
                "schema": "pde_panel_snapshot_v1",
                "runs": [
                    {
                        "id": "formal",
                        "problem_id": "ns",
                        "status": "submitted",
                        "updated_at": "2026-08-21T01:00:00+00:00",
                    },
                    {
                        "id": "smoke",
                        "problem_id": "ns",
                        "status": "completed",
                        "updated_at": "2026-08-20T03:00:00+00:00",
                    },
                ],
            },
        }

        def snapshot_for(root):
            return snapshots[root.name]

        with patch(
            "pde_agent_verifier_solver.panel.publisher.build_hosted_dashboard_snapshot",
            side_effect=snapshot_for,
        ):
            combined = build_combined_hosted_snapshot(
                ["project-runs", "unified-runs", "project-runs"]
            )

        self.assertEqual([row["id"] for row in combined["runs"]], ["formal", "smoke"])
        self.assertEqual(combined["runs"][0]["status"], "running")
        self.assertEqual(
            combined["runs"][1]["updated_at"],
            "2026-08-20T03:00:00+00:00",
        )
        self.assertEqual(
            combined["counts"],
            {"running": 1, "completed": 1, "failed": 0, "submitted": 0},
        )

    def test_rejects_server_paths_and_solver_source_before_network(self) -> None:
        base = {
            "schema": "pde_panel_snapshot_v1",
            "runs": [],
        }
        validate_hosted_snapshot(base)

        for private_value in (
            {"schema": "pde_panel_snapshot_v1", "runs": [], "source": "code"},
            {
                "schema": "pde_panel_snapshot_v1",
                "runs": [{"name": "/scratch/private/run"}],
            },
            {
                "schema": "pde_panel_snapshot_v1",
                "runs": [{"job_id": "123"}],
            },
        ):
            with self.subTest(private_value=private_value):
                with self.assertRaises(UnsafePanelSnapshot):
                    validate_hosted_snapshot(private_value)


if __name__ == "__main__":
    unittest.main()
