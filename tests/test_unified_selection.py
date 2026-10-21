import unittest

from pde_agent_verifier_solver.engine.selection import (
    EvidenceStage,
    MacroCandidateScore,
    PairedObjectiveMode,
    PairedObjectivePolicy,
    aggregate,
    aggregate_score,
    compare_perturbed_non_regression,
    paired_compare,
    paired_objective_score,
    paired_weighted_composite,
    select_or_keep,
    split_paired_score,
)


CASE_IDS = tuple(f"case-{index}" for index in range(8))


def _score(
    label: str,
    role: str,
    errors,
    *,
    digest: str | None = None,
    valid: bool = True,
    gate_failures=(),
    root_ids=None,
    target_sha256: str = "target-sha",
    contract_id: str | None = None,
) -> MacroCandidateScore:
    return MacroCandidateScore(
        artifact_id=f"{label}:{role}",
        artifact_digest=digest or f"digest-{label}-{role}",
        target_sha256=target_sha256,
        contract_id=contract_id or label,
        case_ids=CASE_IDS,
        errors=tuple(errors),
        valid=valid,
        gate_failures=tuple(gate_failures),
        root_ids=None if root_ids is None else tuple(root_ids),
    )


def _stages() -> tuple[EvidenceStage, ...]:
    return (
        EvidenceStage(
            case_count=4,
            confidence_z=1.0,
            min_mean_improvement=0.1,
            min_win_rate=0.75,
        ),
        EvidenceStage(
            case_count=8,
            confidence_z=1.96,
            min_mean_improvement=0.1,
            min_win_rate=0.75,
        ),
    )


class UnifiedSelectionTests(unittest.TestCase):
    def test_aggregate_reports_mean_upper_tail_and_max(self) -> None:
        result = aggregate((1.0, 2.0, 3.0, 4.0), cvar_tail_fraction=0.5)

        self.assertEqual(result.mean, 2.5)
        self.assertEqual(result.cvar, 3.5)
        self.assertEqual(result.max, 4.0)

    def test_confidence_groups_avoid_clean_child_pseudoreplication(self) -> None:
        incumbent = _score("macro", "incumbent", (2.0,) * 8)
        candidate = _score(
            "macro",
            "candidate",
            (1.0, 1.0, 2.2, 2.2, 1.0, 1.0, 1.0, 1.0),
        )
        stage = EvidenceStage(
            4,
            confidence_z=1.0,
            min_win_rate=0.5,
            require_cvar_no_regression=False,
        )

        independent = paired_compare(candidate, incumbent, stage)
        grouped = paired_compare(
            candidate,
            incumbent,
            stage,
            confidence_group_size=2,
        )

        self.assertTrue(independent.passed)
        self.assertFalse(grouped.passed)
        self.assertAlmostEqual(independent.mean_improvement, 0.4)
        self.assertAlmostEqual(independent.confidence_lower, 0.0535898385)
        self.assertAlmostEqual(grouped.confidence_lower, -0.2)
        self.assertIn(
            "mean_improvement_not_supported",
            grouped.failures,
        )
        self.assertEqual(independent.confidence_group_size, 1)
        self.assertEqual(independent.confidence_sample_count, 4)
        self.assertEqual(grouped.confidence_group_size, 2)
        self.assertEqual(grouped.confidence_sample_count, 2)

    def test_confidence_group_contract_fails_closed(self) -> None:
        incumbent = _score("macro", "incumbent", (2.0,) * 8)
        candidate = _score("macro", "candidate", (1.0,) * 8)
        stage = EvidenceStage(
            4,
            confidence_z=1.0,
            require_cvar_no_regression=False,
        )

        for group_size, message in (
            (0, "positive integer"),
            (True, "positive integer"),
            (3, "divisible"),
            (4, "at least two independent"),
        ):
            with self.subTest(group_size=group_size):
                with self.assertRaisesRegex(ValueError, message):
                    paired_compare(
                        candidate,
                        incumbent,
                        stage,
                        confidence_group_size=group_size,
                    )

        one_group = paired_compare(
            candidate,
            incumbent,
            EvidenceStage(4, confidence_z=0.0),
            confidence_group_size=4,
        )
        self.assertEqual(one_group.confidence_sample_count, 1)

    def test_explicit_roots_drive_all_robust_statistics(self) -> None:
        roots = tuple(f"root-{index // 2}" for index in range(8))
        incumbent = _score(
            "macro",
            "incumbent",
            (5.0,) * 8,
            root_ids=roots,
        )
        candidate = _score(
            "macro",
            "candidate",
            (2.0, 6.0, 2.0, 6.0, 2.0, 10.0, 4.0, 4.0),
            root_ids=roots,
        )
        stage = EvidenceStage(
            8,
            confidence_z=0.5,
            min_win_rate=0.7,
            require_cvar_no_regression=False,
        )

        grouped = paired_compare(
            candidate,
            incumbent,
            stage,
            cvar_tail_fraction=0.5,
            confidence_group_size=2,
        )
        legacy_incumbent = _score(
            "macro-legacy",
            "incumbent",
            (5.0,) * 8,
        )
        legacy_candidate = _score(
            "macro-legacy",
            "candidate",
            (2.0, 6.0, 2.0, 6.0, 2.0, 10.0, 4.0, 4.0),
        )
        legacy = paired_compare(
            legacy_candidate,
            legacy_incumbent,
            stage,
            cvar_tail_fraction=0.5,
        )
        root_promotion = select_or_keep(
            incumbent,
            candidate,
            (stage,),
            cvar_tail_fraction=0.5,
            confidence_group_size=2,
        )
        legacy_promotion = select_or_keep(
            legacy_incumbent,
            legacy_candidate,
            (stage,),
            cvar_tail_fraction=0.5,
        )
        root_full = aggregate_score(
            candidate,
            cvar_tail_fraction=0.5,
        )
        root_prefix = aggregate_score(
            candidate,
            4,
            cvar_tail_fraction=0.5,
        )
        legacy_full = aggregate_score(
            legacy_candidate,
            cvar_tail_fraction=0.5,
        )

        self.assertEqual(grouped.root_ids, tuple(dict.fromkeys(roots)))
        self.assertEqual(grouped.confidence_sample_count, 4)
        self.assertAlmostEqual(grouped.candidate_aggregate.mean, 4.5)
        self.assertAlmostEqual(grouped.candidate_aggregate.cvar, 5.0)
        self.assertAlmostEqual(grouped.candidate_aggregate.max, 6.0)
        self.assertAlmostEqual(grouped.win_rate, 0.75)
        self.assertAlmostEqual(grouped.confidence_lower, 0.25)
        self.assertTrue(grouped.passed)
        self.assertTrue(root_promotion.promoted)
        self.assertEqual(root_full, grouped.candidate_aggregate)
        self.assertEqual(root_prefix.mean, 4.0)
        self.assertEqual(root_prefix.cvar, 4.0)
        self.assertEqual(root_prefix.max, 4.0)
        self.assertEqual(legacy_full, legacy.candidate_aggregate)
        self.assertEqual(legacy.root_ids, ())
        self.assertEqual(legacy.confidence_sample_count, 8)
        self.assertAlmostEqual(legacy.candidate_aggregate.mean, 4.5)
        self.assertAlmostEqual(legacy.candidate_aggregate.cvar, 6.5)
        self.assertAlmostEqual(legacy.candidate_aggregate.max, 10.0)
        self.assertAlmostEqual(legacy.win_rate, 0.625)
        self.assertFalse(legacy.passed)
        self.assertFalse(legacy_promotion.promoted)

    def test_explicit_root_contract_fails_closed(self) -> None:
        roots = tuple(f"root-{index // 2}" for index in range(8))
        legacy = MacroCandidateScore(
            "legacy",
            "legacy-digest",
            "target-sha",
            "macro",
            CASE_IDS,
            (1.0,) * 8,
            False,
            ("legacy-gate",),
        )
        self.assertIsNone(legacy.root_ids)
        self.assertEqual(legacy.gate_failures, ("legacy-gate",))

        with self.assertRaisesRegex(ValueError, "align one-for-one"):
            _score(
                "macro",
                "short-lineage",
                (1.0,) * 8,
                root_ids=roots[:-1],
            )

        empty_root_ids = list(roots)
        empty_root_ids[0] = ""
        with self.assertRaisesRegex(ValueError, "nonempty strings"):
            _score(
                "macro",
                "empty-root",
                (1.0,) * 8,
                root_ids=empty_root_ids,
            )

        with self.assertRaisesRegex(ValueError, "contiguous"):
            _score(
                "macro",
                "noncontiguous",
                (1.0,) * 8,
                root_ids=(
                    "root-0",
                    "root-1",
                    "root-0",
                    "root-1",
                    "root-2",
                    "root-2",
                    "root-3",
                    "root-3",
                ),
            )

        incumbent = _score(
            "macro",
            "incumbent",
            (2.0,) * 8,
            root_ids=roots,
        )
        candidate = _score(
            "macro",
            "candidate",
            (1.0,) * 8,
            root_ids=roots,
        )
        with self.assertRaisesRegex(ValueError, "split a root group"):
            aggregate_score(candidate, 3)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            aggregate_score(candidate, 0)
        with self.assertRaisesRegex(ValueError, "requested aggregate prefix"):
            aggregate_score(candidate, 9)
        with self.assertRaisesRegex(TypeError, "MacroCandidateScore"):
            aggregate_score((1.0, 2.0))
        with self.assertRaisesRegex(ValueError, "split a root group"):
            paired_compare(
                candidate,
                incumbent,
                EvidenceStage(3, confidence_z=0.0),
            )
        with self.assertRaisesRegex(ValueError, "two independent roots"):
            paired_compare(
                candidate,
                incumbent,
                EvidenceStage(2, confidence_z=1.0),
            )

        mismatched = _score(
            "macro",
            "mismatched-lineage",
            (1.0,) * 8,
            root_ids=("other", "other", *roots[2:]),
        )
        with self.assertRaisesRegex(ValueError, "root lineages"):
            paired_compare(mismatched, incumbent, EvidenceStage(8))

    def test_contract_labels_do_not_change_numeric_decision_or_cascade(self) -> None:
        signatures = []
        for contract_label in ("vpfp_full_velocity_v1", "gkw_four_field_v1"):
            incumbent = _score(contract_label, "incumbent", (1.0,) * 8)
            candidate = _score(
                contract_label,
                "candidate",
                (0.70, 0.80, 0.75, 0.80, 0.70, 0.75, 0.80, 0.70),
            )

            result = select_or_keep(incumbent, candidate, _stages())

            self.assertTrue(result.promoted)
            self.assertEqual(
                [comparison.stage.case_count for comparison in result.comparisons],
                [4, 8],
            )
            signatures.append(
                tuple(
                    (
                        comparison.mean_improvement,
                        comparison.confidence_lower,
                        comparison.confidence_upper,
                        comparison.win_rate,
                        comparison.passed,
                    )
                    for comparison in result.comparisons
                )
            )

        self.assertEqual(signatures[0], signatures[1])

    def test_cascade_stops_after_first_failed_cumulative_stage(self) -> None:
        incumbent = _score("macro", "incumbent", (1.0,) * 8)
        candidate = _score(
            "macro",
            "candidate",
            (0.70, 0.70, 0.70, 0.70, 1.50, 1.50, 1.50, 1.50),
        )

        result = select_or_keep(incumbent, candidate, _stages())

        self.assertFalse(result.promoted)
        self.assertEqual(result.selected_artifact_id, incumbent.artifact_id)
        self.assertEqual(
            [comparison.passed for comparison in result.comparisons],
            [True, False],
        )

    def test_invalid_or_gate_failed_candidate_cannot_promote(self) -> None:
        incumbent = _score("macro", "incumbent", (1.0,) * 8)
        candidate = _score(
            "macro",
            "candidate",
            (0.1,) * 8,
            valid=False,
            gate_failures=("positivity",),
        )

        result = select_or_keep(incumbent, candidate, _stages())

        self.assertFalse(result.promoted)
        self.assertEqual(result.selected_digest, incumbent.artifact_digest)
        self.assertIn("candidate_invalid", result.comparisons[0].failures)
        self.assertIn("candidate_gate:positivity", result.comparisons[0].failures)

    def test_pairing_and_stage_order_fail_closed(self) -> None:
        incumbent = _score("macro", "incumbent", (1.0,) * 8)
        candidate = MacroCandidateScore(
            artifact_id="macro:candidate",
            artifact_digest="candidate-digest",
            target_sha256="target-sha",
            contract_id="macro",
            case_ids=tuple(reversed(CASE_IDS)),
            errors=(0.5,) * 8,
        )

        with self.assertRaisesRegex(ValueError, "case order"):
            select_or_keep(incumbent, candidate, _stages())
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            select_or_keep(
                incumbent,
                _score("macro", "paired", (0.5,) * 8),
                (EvidenceStage(4), EvidenceStage(4)),
            )
        with self.assertRaisesRegex(ValueError, "positive integer"):
            EvidenceStage(4.0)
        with self.assertRaisesRegex(ValueError, "at least two"):
            EvidenceStage(1, confidence_z=1.0)

    def test_exact_tie_retains_incumbent(self) -> None:
        incumbent = _score("macro", "incumbent", (1.0,) * 8)
        candidate = _score("macro", "candidate", (1.0,) * 8)

        result = select_or_keep(
            incumbent,
            candidate,
            (EvidenceStage(8, confidence_z=0.0, min_win_rate=0.0),),
        )

        self.assertFalse(result.promoted)
        self.assertEqual(result.selected_artifact_id, incumbent.artifact_id)
        self.assertIn(
            "mean_improvement_not_supported",
            result.comparisons[0].failures,
        )

    def test_promotion_accepts_exactly_one_preselected_challenger(self) -> None:
        incumbent = _score("macro", "incumbent", (1.0,) * 8)
        candidate_a = _score(
            "macro",
            "candidate-a",
            (0.5,) * 8,
            digest="digest-a",
        )
        candidate_z = _score(
            "macro",
            "candidate-z",
            (0.5,) * 8,
            digest="digest-z",
        )
        stage = (EvidenceStage(8, confidence_z=1.96, min_win_rate=1.0),)

        self.assertTrue(select_or_keep(incumbent, candidate_a, stage).promoted)
        with self.assertRaisesRegex(TypeError, "exactly one"):
            select_or_keep(incumbent, (candidate_a, candidate_z), stage)

    def test_pairing_binds_exact_cohort_and_macro_contract(self) -> None:
        incumbent = _score("macro", "incumbent", (1.0,) * 8)
        other_cohort = _score(
            "macro",
            "other-cohort",
            (0.5,) * 8,
            target_sha256="other-target-sha",
        )
        other_contract = _score(
            "macro-v2",
            "other-contract",
            (0.5,) * 8,
            contract_id="macro-v2",
        )

        with self.assertRaisesRegex(ValueError, "target digests"):
            select_or_keep(incumbent, other_cohort, _stages())
        with self.assertRaisesRegex(ValueError, "macro contracts"):
            select_or_keep(incumbent, other_contract, _stages())

    def test_cvar_no_regression_is_an_explicit_stage_gate(self) -> None:
        incumbent = _score("macro", "incumbent", (1.0,) * 8)
        candidate = _score(
            "macro",
            "candidate",
            (0.1, 0.1, 0.1, 2.0, 0.1, 0.1, 0.1, 2.0),
        )
        strict = EvidenceStage(
            8,
            confidence_z=0.0,
            min_mean_improvement=0.0,
            min_win_rate=0.5,
            require_cvar_no_regression=True,
        )
        relaxed = EvidenceStage(
            8,
            confidence_z=0.0,
            min_mean_improvement=0.0,
            min_win_rate=0.5,
            require_cvar_no_regression=False,
        )

        strict_result = select_or_keep(incumbent, candidate, (strict,))
        relaxed_result = select_or_keep(incumbent, candidate, (relaxed,))

        self.assertFalse(strict_result.promoted)
        self.assertIn("cvar_regression", strict_result.comparisons[0].failures)
        self.assertTrue(relaxed_result.promoted)

    def test_same_bundle_digest_cannot_compete_under_a_new_id(self) -> None:
        incumbent = _score(
            "macro",
            "incumbent",
            (1.0,) * 8,
            digest="same-content",
        )
        candidate = _score(
            "macro",
            "new-id",
            (0.1,) * 8,
            digest="same-content",
        )

        with self.assertRaisesRegex(ValueError, "artifact digest"):
            select_or_keep(incumbent, candidate, _stages())



    def test_paired_objective_split_and_weighted_legacy_compatibility(self) -> None:
        roots = tuple(f"root-{index // 2}" for index in range(8))
        score = _score(
            "macro",
            "candidate",
            (1.0, 3.0, 2.0, 6.0, 4.0, 8.0, 6.0, 10.0),
            root_ids=roots,
        )

        split = split_paired_score(score)
        legacy = paired_weighted_composite(score)
        weighted = paired_weighted_composite(
            score,
            base_weight=0.75,
            perturbed_weight=0.25,
        )
        primary = paired_objective_score(
            score,
            PairedObjectivePolicy(
                mode="primary_with_perturbed_guardrail",
            ),
        )

        self.assertEqual(split.base.errors, (1.0, 2.0, 4.0, 6.0))
        self.assertEqual(split.perturbed.errors, (3.0, 6.0, 8.0, 10.0))
        self.assertEqual(legacy.errors, (2.0, 4.0, 6.0, 8.0))
        self.assertEqual(aggregate_score(legacy), aggregate_score(score))
        self.assertEqual(weighted.errors, (1.5, 3.0, 5.0, 7.0))
        self.assertEqual(primary.errors, split.base.errors)
        self.assertIs(
            PairedObjectivePolicy().mode,
            PairedObjectiveMode.WEIGHTED_COMPOSITE,
        )
        self.assertAlmostEqual(
            PairedObjectivePolicy().guardrail_confidence_z,
            1.6448536269514722,
        )

        with self.assertRaisesRegex(ValueError, "explicit root lineage"):
            split_paired_score(_score("macro", "legacy", (1.0,) * 8))
        with self.assertRaisesRegex(ValueError, "exactly base and perturbed"):
            split_paired_score(
                _score(
                    "macro",
                    "bad-pairs",
                    (1.0,) * 8,
                    root_ids=("root-0",) * 3 + ("root-1",) * 2 + ("root-2",) * 3,
                )
            )

    def test_perturbed_guardrail_fails_only_supported_practical_regression(self) -> None:
        roots = tuple(f"root-{index // 2}" for index in range(8))
        incumbent = _score(
            "macro",
            "incumbent",
            (1.0, 1.0) * 4,
            root_ids=roots,
        )
        clear_regression = _score(
            "macro",
            "clear-regression",
            (0.5, 1.10) * 4,
            root_ids=roots,
        )
        noisy_regression = _score(
            "macro",
            "noisy-regression",
            (0.5, 1.20, 0.5, 0.90, 0.5, 1.20, 0.5, 1.00),
            root_ids=roots,
        )
        invalid = _score(
            "macro",
            "invalid",
            (0.5, 0.5) * 4,
            valid=False,
            root_ids=roots,
        )

        clear = compare_perturbed_non_regression(clear_regression, incumbent)
        noisy = compare_perturbed_non_regression(noisy_regression, incumbent)
        invalid_result = compare_perturbed_non_regression(invalid, incumbent)

        self.assertFalse(clear.passed)
        self.assertGreater(clear.confidence_lower, 0.0)
        self.assertIn("perturbed_regression_beyond_margin", clear.failures)
        self.assertTrue(noisy.passed)
        self.assertGreater(noisy.mean_excess, 0.0)
        self.assertLess(noisy.confidence_lower, 0.0)
        self.assertFalse(invalid_result.passed)
        self.assertIn("candidate_invalid", invalid_result.failures)


if __name__ == "__main__":
    unittest.main()
