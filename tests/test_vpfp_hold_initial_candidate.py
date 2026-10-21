from __future__ import annotations

import ast
from pathlib import Path
from types import MappingProxyType
import unittest

import numpy as np

from pde_agent_verifier_solver.plugins.vpfp import vpfp_native_macro_query
from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    AnswerView,
    Case,
    CaseSource,
    CaseVisibility,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = (
    REPOSITORY_ROOT
    / "configs"
    / "candidates"
    / "vpfp_hold_initial_macro_candidate.py"
)


class VPFPHoldInitialOneFileCandidateTests(unittest.TestCase):
    def test_source_uses_only_numpy_stdlib_and_public_sdk(self) -> None:
        source = CANDIDATE.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(CANDIDATE))
        allowed_roots = {
            "__future__",
            "collections",
            "math",
            "numpy",
            "pde_agent_verifier_solver",
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertIn(alias.name.split(".", 1)[0], allowed_roots)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                self.assertIn(module.split(".", 1)[0], allowed_roots)
                if module.startswith("pde_agent_verifier_solver"):
                    self.assertEqual(module, "pde_agent_verifier_solver.sdk")

        self.assertNotIn("VPFPHoldInitialMacroProgram", source)
        self.assertNotIn("pde_agent_verifier_solver.plugins", source)
        namespace = _load_like_reference_runtime(CANDIDATE)
        self.assertTrue(callable(namespace.get("predict_macros")))

    def test_loaded_candidate_repeats_public_initial_macros(self) -> None:
        predict = _load_like_reference_runtime(CANDIDATE)["predict_macros"]
        case = _case()
        query = AnswerQuery.grid(
            fields=("rho", "momentum_x", "momentum_y", "kinetic_energy"),
            t=4,
            x=2,
            y=3,
        )

        prediction = predict(case, query, budget=1.0)

        self.assertIsInstance(prediction, AnswerView)
        self.assertEqual(prediction.shape, (4, 2, 3))
        self.assertEqual(prediction.axes["t"], (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0))
        self.assertEqual(prediction.axes["x"], (0.0, 0.5))
        self.assertEqual(prediction.axes["y"], (-1.0, 0.0, 1.0))
        expected_rho = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        expected_momentum_x = tuple(0.25 * value for value in expected_rho)
        expected_momentum_y = tuple(-0.5 * value for value in expected_rho)
        expected_energy = tuple(1.0 + value for value in expected_rho)
        for name, expected in (
            ("rho", expected_rho),
            ("momentum_x", expected_momentum_x),
            ("momentum_y", expected_momentum_y),
            ("kinetic_energy", expected_energy),
        ):
            frames = tuple(
                prediction.fields[name][offset : offset + 6]
                for offset in range(0, 24, 6)
            )
            self.assertEqual(frames, (expected,) * 4)

    def test_candidate_accepts_the_equation_owned_native_query(self) -> None:
        predict = _load_like_reference_runtime(CANDIDATE)["predict_macros"]
        case = _case()
        query = vpfp_native_macro_query(case, "development")

        prediction = predict(case, query)

        self.assertEqual(prediction.shape, (3, 2, 3))
        self.assertEqual(tuple(prediction.fields), tuple(query.fields))

    def test_invalid_budget_and_spatial_resampling_are_rejected(self) -> None:
        predict = _load_like_reference_runtime(CANDIDATE)["predict_macros"]
        case = _case()
        native = vpfp_native_macro_query(case, "development")
        with self.assertRaisesRegex(ValueError, "budget"):
            predict(case, native, budget=0.0)

        incompatible = AnswerQuery.grid(
            fields=native.fields,
            t=3,
            x=4,
            y=3,
        )
        with self.assertRaisesRegex(ValueError, "native x-resolution"):
            predict(case, incompatible)


def _load_like_reference_runtime(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    namespace: dict[str, object] = {
        "__file__": str(path),
        "__name__": "_one_file_candidate_test",
        "__package__": None,
    }
    exec(compile(payload, str(path), "exec"), namespace)
    return namespace


def _readonly(values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    array.setflags(write=False)
    return array


def _case() -> Case:
    rho0 = _readonly([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    momentum0 = np.stack((0.25 * rho0, -0.5 * rho0), axis=-1)
    momentum0.setflags(write=False)
    kinetic_energy0 = _readonly(1.0 + rho0)
    arrays = MappingProxyType(
        {
            "t": _readonly([0.0, 0.5, 1.0]),
            "x": _readonly([0.0, 0.5]),
            "y": _readonly([-1.0, 0.0, 1.0]),
            "rho0": rho0,
            "momentum0": momentum0,
            "kinetic_energy0": kinetic_energy0,
        }
    )
    return Case(
        case_id="standalone-hold-initial-test",
        source=CaseSource.GENERATED,
        visibility=CaseVisibility.DEVELOPMENT,
        public_data=MappingProxyType({"arrays": arrays}),
        round_index=0,
        tags=("vpfp", "cold-start"),
    )


if __name__ == "__main__":
    unittest.main()
