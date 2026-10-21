from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pde_agent_verifier_solver.plugins.vpfp import (
    VPFPHoldInitialMacroProgram,
    VPFPReleaseBankCaseSource,
    bind_vpfp_release_bank_reference,
    make_vpfp_hold_initial_cold_start,
    vpfp_native_macro_query,
)
from pde_agent_verifier_solver.plugins.vpfp.macro_matching import (
    VPFP_MACRO_FIELDS,
    VPFPMacroMatcher,
)
from pde_agent_verifier_solver.sdk import (
    Case,
    CaseSource,
    CaseVisibility,
    CohortKind,
    GenerationKind,
    QueryKind,
    SampleRequest,
)


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_case(
    root: Path,
    *,
    index: int,
    checkpoint_sha256: str,
) -> Path:
    import numpy as np

    case_id = f"bank-case-{index:02d}"
    case_dir = root / "cases" / case_id
    case_dir.mkdir(parents=True)
    t = np.asarray([0.0, 0.5, 1.0])
    x = np.asarray([0.0, 0.5])
    y = np.asarray([0.0, 1.0, 2.0])
    shape = (len(t), len(x), len(y))
    base = float(index + 1)
    time_increment = np.arange(len(t), dtype=float)[:, None, None]
    spatial = np.arange(len(x) * len(y), dtype=float).reshape(len(x), len(y))
    rho = base + 0.1 * time_increment + 0.01 * spatial
    momentum = np.stack((0.2 * rho, -0.3 * rho), axis=-1)
    kinetic_energy = 0.5 * rho + 1.0
    public_arrays = case_dir / "public_arrays.npz"
    hidden_truth = case_dir / "hidden_truth_cache.npz"
    np.savez_compressed(
        public_arrays,
        t=t,
        x=x,
        y=y,
        rho0=rho[0],
        momentum0=momentum[0],
        kinetic_energy0=kinetic_energy[0],
    )
    np.savez_compressed(
        hidden_truth,
        rho_true=rho,
        momentum_true=momentum,
        kinetic_energy_true=kinetic_energy,
    )
    _write_json(
        case_dir / "public_question.json",
        {
            "case_id": case_id,
            "solver_output_contract_version": (
                "vpfp_continuous_macro_output_v1"
            ),
            "target_shapes": {
                "rho": list(shape),
                "momentum": [*shape, 2],
                "kinetic_energy": list(shape),
            },
        },
    )
    _write_json(
        case_dir / "case_spec.json",
        {
            "schema": "vpfp_continuous_materialized_case_v1",
            "decoder_checkpoint_sha256": checkpoint_sha256,
            "solver_output_contract_version": (
                "vpfp_continuous_macro_output_v1"
            ),
        },
    )
    _write_json(
        case_dir / "hidden_truth_manifest.json",
        {
            "case_id": case_id,
            "continuous_primary": {
                "decoder_checkpoint_sha256": checkpoint_sha256,
                "public_arrays_sha256": _sha256_file(public_arrays),
                "materialized_cache_sha256": _sha256_file(hidden_truth),
            },
        },
    )
    return case_dir


def _write_bank(root: Path, *, case_count: int = 8):
    checkpoint = root / "decoder.pt"
    checkpoint.write_bytes(b"frozen-vpfp-decoder-checkpoint-v1\n")
    checkpoint_sha = _sha256_file(checkpoint)
    case_dirs = tuple(
        _write_case(root, index=index, checkpoint_sha256=checkpoint_sha)
        for index in range(case_count)
    )
    report_payload = {
        "schema": "vpfp_long_time_answer_first_release_bank_v2",
        "passed": True,
        "decoder_checkpoint": checkpoint.name,
        "answer_first_contract": {
            "continuous_hidden_answer_is_primary": True,
            "public_source_is_exact_jet_residual": True,
            "coarse_equation_first_solve_is_shape_seed_only": True,
            "grid_arrays_are_numerical_access_views": True,
        },
        "development_case_dirs": [
            str(path.relative_to(root))
            for path in case_dirs[: case_count // 2]
        ],
        "released_completed_case_dirs": [
            str(path.relative_to(root))
            for path in case_dirs[case_count // 2 :]
        ],
    }
    report = root / "release_bank_report.json"
    _write_json(report, report_payload)
    return report, checkpoint, report_payload


class VPFPReleaseBankReferenceTests(unittest.TestCase):
    def test_byte_identity_and_deterministic_global_nonoverlap(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            report, checkpoint, _payload = _write_bank(root)
            source = VPFPReleaseBankCaseSource(
                report,
                decoder_checkpoint=checkpoint,
            )
            replay = VPFPReleaseBankCaseSource(report)

            self.assertTrue(source.reference_only)
            self.assertFalse(source.formal_claims_allowed)
            self.assertEqual(source.case_count, 8)
            self.assertEqual(source.remaining_case_count, 8)
            self.assertEqual(
                source.decoder_checkpoint_sha256,
                _sha256_file(checkpoint),
            )
            self.assertEqual(len(source.artifact_sha256), 64)
            self.assertEqual(
                source.artifact_sha256,
                replay.artifact_sha256,
            )
            perturbed_request = SampleRequest(
                round_index=1,
                cohort=CohortKind.DEVELOPMENT,
                seed=9_999,
                count=1,
                kind=GenerationKind.PERTURBED,
                anchors=(
                    Case(
                        case_id="vpfp-anchor",
                        source=CaseSource.GENERATED,
                        visibility=CaseVisibility.DEVELOPMENT,
                        public_data={},
                        round_index=1,
                    ),
                ),
            )
            with self.assertRaisesRegex(ValueError, "only clean BASE"):
                source.sample(perturbed_request)
            self.assertEqual(source.remaining_case_count, 8)

            development_request = SampleRequest(
                round_index=1,
                cohort=CohortKind.DEVELOPMENT,
                seed=10_000,
                count=3,
            )
            selection_request = SampleRequest(
                round_index=1,
                cohort=CohortKind.SELECTION,
                seed=10_001,
                count=2,
            )
            development = source.sample(development_request)
            selection = source.sample(selection_request)
            replay_development = replay.sample(development_request)
            replay_selection = replay.sample(selection_request)

            development_ids = tuple(row.case.case_id for row in development)
            selection_ids = tuple(row.case.case_id for row in selection)
            self.assertEqual(
                development_ids,
                tuple(row.case.case_id for row in replay_development),
            )
            self.assertEqual(
                selection_ids,
                tuple(row.case.case_id for row in replay_selection),
            )
            self.assertFalse(set(development_ids) & set(selection_ids))
            self.assertEqual(source.remaining_case_count, 3)
            for row in development:
                self.assertIs(row.case.source, CaseSource.GENERATED)
                self.assertIs(
                    row.case.visibility,
                    CaseVisibility.DEVELOPMENT,
                )
                self.assertEqual(row.case.round_index, 1)
                self.assertNotIn("case_dir", row.case.public_data)
            for row in selection:
                self.assertIs(row.case.source, CaseSource.GENERATED)
                self.assertIs(row.case.visibility, CaseVisibility.HIDDEN)
                self.assertEqual(row.case.round_index, 1)

            source.sample(
                SampleRequest(
                    round_index=2,
                    cohort=CohortKind.TERMINAL,
                    seed=20_003,
                    count=3,
                )
            )
            with self.assertRaisesRegex(ValueError, "0 unused cases"):
                source.sample(
                    SampleRequest(
                        round_index=2,
                        cohort=CohortKind.PROMOTION,
                        seed=20_002,
                        count=1,
                    )
                )

    def test_identity_binds_report_bytes_and_rejects_checkpoint_mismatch(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            report, checkpoint, payload = _write_bank(root)
            original = VPFPReleaseBankCaseSource(report)

            alternate_report = root / "alternate_report.json"
            _write_json(alternate_report, {**payload, "audit_note": "same bank"})
            alternate = VPFPReleaseBankCaseSource(alternate_report)
            self.assertNotEqual(
                original.artifact_sha256,
                alternate.artifact_sha256,
            )

            other_checkpoint = root / "other.pt"
            other_checkpoint.write_bytes(b"different-checkpoint-bytes\n")
            with self.assertRaisesRegex(ValueError, "bytes differ"):
                VPFPReleaseBankCaseSource(
                    report,
                    decoder_checkpoint=other_checkpoint,
                )

            copied_checkpoint = root / "copied.pt"
            copied_checkpoint.write_bytes(checkpoint.read_bytes())
            copied = VPFPReleaseBankCaseSource(
                report,
                decoder_checkpoint=copied_checkpoint,
            )
            self.assertEqual(
                copied.artifact_sha256,
                original.artifact_sha256,
            )

            failed_report = root / "failed_report.json"
            _write_json(failed_report, {**payload, "passed": False})
            with self.assertRaisesRegex(ValueError, "not passing"):
                VPFPReleaseBankCaseSource(failed_report)

            request = SampleRequest(
                round_index=0,
                cohort=CohortKind.DEVELOPMENT,
                seed=41_000,
                count=1,
            )
            probe_record = VPFPReleaseBankCaseSource(report).sample(request)[0]
            question = (
                root
                / "cases"
                / probe_record.case.case_id
                / "public_question.json"
            )
            question.write_bytes(question.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "changed after"):
                original.sample(request)

    def test_native_query_hold_initial_and_thin_binding(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            report, checkpoint, _payload = _write_bank(root)
            plugin = bind_vpfp_release_bank_reference(
                report_path=report,
                decoder_checkpoint=checkpoint,
                query_factory=vpfp_native_macro_query,
            )
            record = plugin.generator.sample(
                SampleRequest(
                    round_index=1,
                    cohort=CohortKind.DEVELOPMENT,
                    seed=30_000,
                    count=1,
                )
            )[0]

            queries = tuple(
                vpfp_native_macro_query(record.case, profile)
                for profile in (
                    "development",
                    "selection",
                    "promotion",
                    "terminal",
                )
            )
            self.assertTrue(all(query == queries[0] for query in queries))
            query = queries[0]
            self.assertIs(query.kind, QueryKind.GRID)
            self.assertEqual(tuple(query.fields), VPFP_MACRO_FIELDS)
            self.assertEqual(query.resolution, {"t": 3, "x": 2, "y": 3})
            self.assertEqual(plugin.objective.query(record.case, "terminal"), query)
            with self.assertRaisesRegex(ValueError, "unknown"):
                vpfp_native_macro_query(record.case, "not-a-profile")

            cold_start = make_vpfp_hold_initial_cold_start()
            self.assertEqual(cold_start.ref.round_index, 0)
            self.assertIsNone(cold_start.ref.parent_sha256)
            self.assertIsInstance(
                cold_start.program,
                VPFPHoldInitialMacroProgram,
            )
            selection_runtime = cold_start.fresh_program()
            terminal_runtime = cold_start.fresh_program()
            self.assertIsNot(selection_runtime, terminal_runtime)
            self.assertIsInstance(selection_runtime, VPFPHoldInitialMacroProgram)
            self.assertIsInstance(terminal_runtime, VPFPHoldInitialMacroProgram)
            source_path = Path(cold_start.ref.bundle_uri)
            self.assertEqual(
                cold_start.ref.bundle_sha256,
                _sha256_file(source_path),
            )
            prediction = cold_start.program.predict_macros(
                record.case,
                query,
                budget=1.0,
            )
            self.assertEqual(prediction.shape, (3, 2, 3))
            rho_frames = tuple(
                prediction.fields["rho"][offset : offset + 6]
                for offset in range(0, 18, 6)
            )
            self.assertEqual(rho_frames[0], rho_frames[1])
            self.assertEqual(rho_frames[1], rho_frames[2])

            matcher = VPFPMacroMatcher()
            truth = record.truth.answer(query)
            target = matcher.prepare_target(record.case, truth, query)
            result = matcher.match(record.case, prediction, target)
            self.assertGreater(result.primary_error, 0.0)
            self.assertEqual(result.contract_id, matcher.contract_id)


if __name__ == "__main__":
    unittest.main()
