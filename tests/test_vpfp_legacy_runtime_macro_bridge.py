import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LEGACY_SRC = ROOT / "legacy" / "vpfp" / "src"
PACKAGE_PREFIX = "agent_verifier"


def _load_legacy_runtime():
    preserved = {
        name: module
        for name, module in sys.modules.items()
        if name == PACKAGE_PREFIX or name.startswith(PACKAGE_PREFIX + ".")
    }
    for name in preserved:
        sys.modules.pop(name)
    sys.path.insert(0, str(LEGACY_SRC))
    try:
        from agent_verifier.vpfp import development_runtime_v2
        from agent_verifier.vpfp.compute_budget import SolverComputeBudget
        loaded = development_runtime_v2, SolverComputeBudget
    finally:
        sys.path.pop(0)
        for name in tuple(sys.modules):
            if name == PACKAGE_PREFIX or name.startswith(PACKAGE_PREFIX + "."):
                sys.modules.pop(name)
        sys.modules.update(preserved)
    return loaded


runtime, SolverComputeBudget = _load_legacy_runtime()


EXACT_MACRO_CONTRACT = "vpfp_full_velocity_macro_match_v1"


class VPFPLegacyRuntimeMacroBridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.child_environments = []

        self.query_times = np.array(
            [0.0, 0.05, 0.2, 0.45, 0.62, 0.8, 1.0],
            dtype=np.float64,
        )
        self.x = np.array([-1.0, 0.25], dtype=np.float64)
        self.y = np.array([0.0, 0.7, 2.0], dtype=np.float64)
        shape = (len(self.query_times), len(self.x), len(self.y))
        self.truth = {
            "rho": np.ones(shape, dtype=np.float64),
            "momentum": np.ones(shape + (2,), dtype=np.float64),
            "kinetic_energy": np.ones(shape, dtype=np.float64),
        }
        self.prediction = {
            name: np.array(value, copy=True)
            for name, value in self.truth.items()
        }
        self.prediction["momentum"][..., 0] = 1.25
        self.public_case = {
            "case_id": "legacy-runtime-case",
            "arrays": {"x": self.x, "y": self.y},
            "grid": SimpleNamespace(nx=len(self.x), ny=len(self.y)),
            "time": SimpleNamespace(t0=0.0, t1=1.0),
        }
        self.budget = SolverComputeBudget(
            tier=0,
            wall_time_seconds=30.0,
            cpus=1,
            gpus=0,
            memory_bytes=1024,
        )
        self.evaluator = object.__new__(runtime.VPFPDevelopmentEvaluatorRuntime)
        self.evaluator.worker_script = self.root / "worker.py"
        self.evaluator.config = SimpleNamespace(subprocess_grace_seconds=1.0)

    def _successful_popen(self, command, **_kwargs):
        child_environment = _kwargs.get("env")
        self.assertIsInstance(child_environment, dict)
        self.child_environments.append(dict(child_environment))
        output_path = Path(command[command.index("--output") + 1])
        report_path = Path(command[command.index("--report") + 1])
        np.savez(output_path, **self.prediction)
        report_path.write_text(
            json.dumps(
                {
                    "schema": runtime.VPFP_BUDGETED_SOLVER_SUBPROCESS_SCHEMA,
                    "accepted": True,
                    "wall_time_seconds": 0.01,
                }
            ),
            encoding="utf-8",
        )

        class _CompletedProcess:
            @staticmethod
            def wait(timeout):
                del timeout
                return 0

        return _CompletedProcess()

    def _run(self, label: str, contract: str | None):
        environment = {"VPFP_TEST_VISIBLE": label}
        if contract is not None:
            environment[runtime.VPFP_MACRO_MATCH_CONTRACT_ENV] = contract
        with (
            mock.patch.dict(os.environ, environment, clear=True),
            mock.patch.object(runtime, "load_public_case", return_value=self.public_case),
            mock.patch.object(runtime.subprocess, "Popen", self._successful_popen),
        ):
            return self.evaluator._run_solver_call(
                solver_path=self.root / "solver.py",
                public_case_dir=self.root / "public-case",
                query_times=self.query_times,
                truth=self.truth,
                budget=self.budget,
                output_root=self.root / label,
            )

    def test_guard_off_omits_unified_macro_diagnostic(self) -> None:
        result = self._run("guard-off", None)

        self.assertIs(result["accepted"], True)
        self.assertNotIn("unified_macro_match", result)

    def test_exact_contract_adds_diagnostic_without_changing_legacy_score(self) -> None:
        guard_off = self._run("guard-off-for-parity", None)
        guarded = self._run("guard-on", EXACT_MACRO_CONTRACT)

        self.assertEqual(len(self.child_environments), 2)
        for child_environment in self.child_environments:
            self.assertNotIn(
                runtime.VPFP_MACRO_MATCH_CONTRACT_ENV,
                child_environment,
            )
        self.assertEqual(
            self.child_environments[-1]["VPFP_TEST_VISIBLE"],
            "guard-on",
        )
        self.assertIs(guarded["accepted"], True)
        diagnostic = guarded["unified_macro_match"]
        self.assertEqual(diagnostic["contract"], EXACT_MACRO_CONTRACT)
        self.assertEqual(
            diagnostic["schema"],
            "vpfp_legacy_macro_match_payload_v1",
        )
        self.assertIs(diagnostic["used_for_winner_selection"], False)
        self.assertRegex(
            diagnostic["coordinate_digest"],
            r"^sha256:[0-9a-f]{64}$",
        )
        self.assertEqual(
            guarded["score_smaller_is_better"],
            guard_off["score_smaller_is_better"],
        )
        self.assertEqual(
            guarded["field_relative_l2"],
            guard_off["field_relative_l2"],
        )
        self.assertEqual(
            guarded["time_band_scores"],
            guard_off["time_band_scores"],
        )

    def test_wrong_contract_fails_closed(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError,
            "unified macro-match contract changed",
        ):
            self._run("wrong-contract", "vpfp_wrong_contract_v9")


if __name__ == "__main__":
    unittest.main()
