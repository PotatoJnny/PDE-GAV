import json
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LEGACY_SRC = ROOT / "legacy" / "ns" / "src"
if str(LEGACY_SRC) not in sys.path:
    sys.path.insert(0, str(LEGACY_SRC))

from agent_verifier.neural_generator import triplane_dataset as DATASET  # noqa: E402
from agent_verifier.neural_generator.triplane_agent_solver import (  # noqa: E402
    WorkspaceEvaluatorBroker,
    evaluate_solver_file,
)


class NSParallelEvaluatorTests(unittest.TestCase):
    def test_parallel_evaluation_matches_serial_and_keeps_case_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            public_root = root / "public"
            private_root = root / "private"
            public_root.mkdir()
            private_root.mkdir()

            case_rows = []
            for index in range(4):
                case_id = f"case_{index:03d}"
                public_case = public_root / case_id
                private_case = private_root / case_id
                public_case.mkdir()
                private_case.mkdir()
                t = np.linspace(0.0, 1.0, 3)
                x = np.linspace(0.0, 1.0, 3)
                y = np.linspace(0.0, 1.0, 4)
                np.savez_compressed(
                    public_case / "public_arrays.npz",
                    t=t,
                    x=x,
                    y=y,
                    fluid_mask=np.ones((len(x), len(y)), dtype=bool),
                )
                (public_case / "public_question.json").write_text(
                    json.dumps({"case_id": case_id})
                )
                np.savez_compressed(
                    private_case / "hidden_truth_cache.npz",
                    u_true=np.ones((len(t), 2, len(x), len(y)), dtype=np.float32),
                )
                case_rows.append(
                    {"case_id": case_id, "private_case_dir": str(private_case)}
                )

            manifest = root / "private_manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "reward_spec": {"metric": "relative_l2"},
                        "public_cases_root": str(public_root),
                        "cases": case_rows,
                    }
                )
            )
            solver = root / "submission.py"
            solver.write_text(
                "from pathlib import Path\n"
                "import numpy as np\n\n"
                "_PREPARED = False\n\n"
                "def prepare_solver(public_cases_root):\n"
                "    global _PREPARED\n"
                "    _PREPARED = Path(public_cases_root).is_dir()\n\n"
                "def solve_case(case_dir, query_grid=None):\n"
                "    if not _PREPARED:\n"
                "        raise RuntimeError('prepare_solver was not called')\n"
                "    t = np.asarray(query_grid['t'])\n"
                "    x = np.asarray(query_grid['x'])\n"
                "    y = np.asarray(query_grid['y'])\n"
                "    return {'velocity': np.ones((len(t), 2, len(x), len(y)))}\n"
            )

            serial = evaluate_solver_file(
                solver_path=solver,
                public_cases_root=public_root,
                private_manifest_path=manifest,
                max_workers=1,
            )
            checkpoint = root / "parallel_checkpoint.json"
            parallel = evaluate_solver_file(
                solver_path=solver,
                public_cases_root=public_root,
                private_manifest_path=manifest,
                checkpoint_path=checkpoint,
                checkpoint_interval=1,
                max_workers=2,
            )

            expected_ids = [row["case_id"] for row in case_rows]
            self.assertEqual(
                [row["case_id"] for row in parallel["results"]], expected_ids
            )
            self.assertEqual(
                [row["relative_l2"] for row in parallel["results"]],
                [row["relative_l2"] for row in serial["results"]],
            )
            self.assertEqual(parallel["summary"]["ok"], 4)
            self.assertEqual(parallel["worker_count"], 2)
            self.assertTrue(json.loads(checkpoint.read_text())["complete"])

            audit = root / "broker" / "evaluator_broker_audit.jsonl"
            with WorkspaceEvaluatorBroker(
                workspace=root,
                private_manifest=manifest,
                audit_path=audit,
                max_workers=1,
            ) as broker:
                responses = []
                for _ in range(2):
                    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                        client.connect(str(broker.socket_path))
                        request = {
                            "token": broker.token,
                            "solver": "submission.py",
                            "cases": "public",
                        }
                        client.sendall((json.dumps(request) + "\n").encode("utf-8"))
                        received = b""
                        while not received.endswith(b"\n"):
                            received += client.recv(65536)
                    responses.append(json.loads(received.decode("utf-8"))["payload"])

            self.assertFalse(responses[0]["cache_hit"])
            self.assertTrue(responses[1]["cache_hit"])
            self.assertEqual(
                len(list((audit.parent / "evaluator_requests").glob("*.json"))),
                2,
            )
            audit_rows = [json.loads(line) for line in audit.read_text().splitlines()]
            self.assertEqual([row["cache_hit"] for row in audit_rows], [False, True])

    def test_case_writer_computes_diagnostic_fields_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            geometry = DATASET.build_kite_baffle_geometry()
            t = np.linspace(0.0, 1.0, 3)
            velocity = np.zeros((len(t), 2, len(geometry.x), len(geometry.y)))
            pressure = np.zeros((len(t), len(geometry.x), len(geometry.y)))
            forcing = np.zeros_like(velocity)
            original = DATASET._case_diagnostic_arrays

            with mock.patch.object(
                DATASET,
                "_case_diagnostic_arrays",
                wraps=original,
            ) as diagnostic_fields:
                metrics = DATASET.write_case(
                    Path(temporary) / "case_000",
                    case_id="case_000",
                    geometry=geometry,
                    t=t,
                    velocity=velocity,
                    pressure=pressure,
                    forcing=forcing,
                    viscosity=0.01,
                    case_spec={"case_id": "case_000"},
                )

            self.assertEqual(diagnostic_fields.call_count, 1)
            saved = json.loads(
                (Path(temporary) / "case_000" / "diagnostics.json").read_text()
            )
            self.assertEqual(saved["velocity_rms"], metrics["velocity_rms"])


if __name__ == "__main__":
    unittest.main()
