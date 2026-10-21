import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from pde_agent_verifier_solver.plugins.gkw import (
    GKWObservableTask,
    GKWProblem,
    GKW_MACRO_FIELDS,
)
from pde_agent_verifier_solver.plugins.gkw.release_io import load_gkw_release
from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    AnswerView,
    Artifact,
    DatasetInput,
    DatasetKind,
    TaskKind,
)

from tests.gkw_release_fixture import write_gkw_release


def _refresh_payload(case_dir: Path, name: str) -> None:
    manifest_path = case_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data = (case_dir / name).read_bytes()
    row = next(item for item in manifest["files"] if item["path"] == name)
    row["bytes"] = len(data)
    row["sha256"] = hashlib.sha256(data).hexdigest()
    manifest["total_bytes"] = sum(item["bytes"] for item in manifest["files"])
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")


def _write_bank(root: Path, development: Path, hidden: Path) -> Path:
    path = root / "release_bank.json"
    path.write_text(
        json.dumps(
            {
                "schema": "fusion_gkw_macro_release_bank_v1",
                "development_case_dirs": [str(development)],
                "released_completed_case_dirs": [str(hidden)],
            }
        ),
        encoding="utf-8",
    )
    return path


def _prepare(root: Path, development: Path, hidden: Path):
    return GKWProblem().prepare_dataset(
        DatasetInput(
            "gkw-security",
            DatasetKind.NUMERICAL,
            str(_write_bank(root, development, hidden)),
            settings={"development_cases": 1, "hidden_cases": 1},
        )
    )


class GKWSecurityBoundaryTests(unittest.TestCase):
    def test_manifest_digest_is_verified_before_payload_use(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            case_dir = Path(temporary) / "case"
            write_gkw_release(case_dir, "digest-case", 1.0)
            (case_dir / "README.md").write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "payload integrity"):
                load_gkw_release(case_dir)

    def test_extra_public_array_and_fractional_answer_indices_fail_closed(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            extra = root / "extra"
            write_gkw_release(extra, "extra-case", 1.0)
            public_path = extra / "case_public.npz"
            with np.load(public_path, allow_pickle=False) as payload:
                copied = {name: np.asarray(payload[name]).copy() for name in payload.files}
            copied["array__hidden_controls"] = np.asarray([123.0])
            np.savez(public_path, **copied)
            _refresh_payload(extra, "case_public.npz")
            with self.assertRaisesRegex(ValueError, "packet keys"):
                load_gkw_release(extra)

            fractional = root / "fractional"
            write_gkw_release(fractional, "fractional-case", 1.0)
            answer_path = fractional / "released_answer.npz"
            with np.load(answer_path, allow_pickle=False) as payload:
                copied = {name: np.asarray(payload[name]).copy() for name in payload.files}
            copied["field_indices"] = copied["field_indices"].astype(float)
            copied["field_indices"][0, 0] = 0.5
            np.savez(answer_path, **copied)
            _refresh_payload(fractional, "released_answer.npz")
            with self.assertRaisesRegex(TypeError, "must be int64"):
                load_gkw_release(fractional)

    def test_public_shape_and_nested_metadata_fail_closed(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shaped = root / "shaped"
            write_gkw_release(shaped, "shaped-case", 1.0)
            public_path = shaped / "case_public.npz"
            with np.load(public_path, allow_pickle=False) as payload:
                copied = {
                    name: np.asarray(payload[name]).copy()
                    for name in payload.files
                }
            original = copied["array__F"]
            copied["array__F"] = np.column_stack((original, original))
            np.savez(public_path, **copied)
            _refresh_payload(shaped, "case_public.npz")
            with self.assertRaisesRegex(TypeError, "public array F"):
                load_gkw_release(shaped)

            nested = root / "nested"
            write_gkw_release(nested, "nested-case", 1.0)
            public_path = nested / "case_public.npz"
            with np.load(public_path, allow_pickle=False) as payload:
                copied = {
                    name: np.asarray(payload[name]).copy()
                    for name in payload.files
                }
            metadata = json.loads(str(copied["metadata_json"].reshape(-1)[0]))
            metadata["equation_contract"]["undeclared_payload"] = "forbidden"
            copied["metadata_json"] = np.asarray(
                json.dumps(metadata, sort_keys=True)
            )
            np.savez(public_path, **copied)
            _refresh_payload(nested, "case_public.npz")
            with self.assertRaisesRegex(ValueError, "recursively changed"):
                load_gkw_release(nested)

    def test_json_scalar_type_substitutions_fail_closed(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            public_case = root / "public-type"
            write_gkw_release(public_case, "public-type-case", 1.0)
            public_path = public_case / "case_public.npz"
            with np.load(public_path, allow_pickle=False) as payload:
                copied = {
                    name: np.asarray(payload[name]).copy()
                    for name in payload.files
                }
            metadata = json.loads(str(copied["metadata_json"].reshape(-1)[0]))
            metadata["equation_contract"]["grid"]["nperiod"] = True
            copied["metadata_json"] = np.asarray(
                json.dumps(metadata, sort_keys=True)
            )
            np.savez(public_path, **copied)
            _refresh_payload(public_case, "case_public.npz")
            with self.assertRaisesRegex(ValueError, "recursively changed"):
                load_gkw_release(public_case)

            oracle_case = root / "oracle-type"
            write_gkw_release(oracle_case, "oracle-type-case", 1.0)
            oracle_path = oracle_case / "oracle_protocol.json"
            oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
            oracle["arbitrary_time_queries_supported"] = 1
            oracle_path.write_text(
                json.dumps(oracle, sort_keys=True), encoding="utf-8"
            )
            _refresh_payload(oracle_case, "oracle_protocol.json")
            with self.assertRaisesRegex(ValueError, "pinned policy"):
                load_gkw_release(oracle_case)

            manifest_case = root / "manifest-type"
            write_gkw_release(manifest_case, "manifest-type-case", 1.0)
            manifest_path = manifest_case / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["phase_row_count"] = False
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "privacy policy"):
                load_gkw_release(manifest_case)

            answer_case = root / "answer-type"
            write_gkw_release(answer_case, "answer-type-case", 1.0)
            answer_path = answer_case / "released_answer.npz"
            with np.load(answer_path, allow_pickle=False) as payload:
                copied = {
                    name: np.asarray(payload[name]).copy()
                    for name in payload.files
                }
            metadata = json.loads(str(copied["metadata_json"].reshape(-1)[0]))
            metadata["macro_truth_construction"]["time_control_count"] = 4.0
            copied["metadata_json"] = np.asarray(
                json.dumps(metadata, sort_keys=True)
            )
            np.savez(answer_path, **copied)
            _refresh_payload(answer_case, "released_answer.npz")
            with self.assertRaisesRegex(ValueError, "pinned contract"):
                load_gkw_release(answer_case)

    def test_case_identity_and_cohort_disjointness_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case_dir = root / "case"
            write_gkw_release(case_dir, "bound-case", 1.0)
            with self.assertRaisesRegex(ValueError, "requested case"):
                load_gkw_release(case_dir, expected_case_id="another-case")
            with self.assertRaisesRegex(ValueError, "must be disjoint"):
                _prepare(root, case_dir, case_dir)

    def test_candidate_cannot_mutate_controller_weights_or_query(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            development = root / "development"
            hidden = root / "hidden"
            write_gkw_release(development, "mutation-development", 1.0)
            write_gkw_release(hidden, "mutation-hidden", 2.0)
            record = _prepare(root, development, hidden).development_cases[0]
            query = AnswerQuery.grid(fields=GKW_MACRO_FIELDS, t=2, field_row=2)
            truth = record.truth.answer(query)
            fields = dict(truth.fields)
            changed = list(fields["particle_density"])
            changed[0] += 1.0
            fields["particle_density"] = tuple(changed)
            prediction = AnswerView(
                kind=truth.kind,
                fields=fields,
                axes=truth.axes,
                points=truth.points,
                shape=truth.shape,
            )

            class MutatingModel:
                def predict_observables(self, case, query, budget=None):
                    arrays = dict(case.public_data["arrays"])
                    arrays["B"] = np.zeros_like(arrays["B"])
                    public = dict(case.public_data)
                    public["arrays"] = arrays
                    object.__setattr__(case, "public_data", public)
                    object.__setattr__(query, "resolution", {"t": 1, "field_row": 1})
                    return prediction

            artifact = Artifact(
                "mutator",
                TaskKind.OBSERVABLE_MODEL,
                "mutating candidate",
                MutatingModel(),
                "adversarial test",
                "predict_observables",
                0,
                "test-agent",
            )
            evaluation = GKWObservableTask().evaluate(
                artifact, record.case, truth, query
            )
            self.assertGreater(evaluation.primary_error, 0.0)
            self.assertEqual(
                record.case.public_data["arrays"]["B"][:2].tolist(),
                [2.0, 4.0],
            )
            self.assertEqual(dict(query.resolution), {"t": 2, "field_row": 2})
            with self.assertRaises(ValueError):
                record.case.public_data["arrays"]["B"][0] = 99.0


if __name__ == "__main__":
    unittest.main()
