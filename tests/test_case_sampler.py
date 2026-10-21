import unittest

from pde_agent_verifier_solver.engine.case_sampler import RoundCaseSampler
from pde_agent_verifier_solver.sdk.models import (
    Case,
    CaseRecord,
    CaseSource,
    CaseVisibility,
    DatasetInput,
    DatasetKind,
    GenerationKind,
    GeneratorCapabilities,
    GeneratorCheckpoint,
    GeneratorState,
    PreparedDataset,
)
from pde_agent_verifier_solver.sdk.sampling import (
    PairedSampleRequest,
    PerturbCasesRequest,
    RecoverableSamplingRejection,
    SampleCasesRequest,
    SampleSource,
    SamplingStrategy,
)


class _Truth:
    def answer(self, query):  # pragma: no cover - sampling never reads truth
        raise AssertionError("truth must stay behind the sampler")


def _record(
    case_id: str,
    *,
    source: CaseSource = CaseSource.INITIAL,
    parent_case_id: str | None = None,
    round_index: int | None = None,
    visibility: CaseVisibility = CaseVisibility.DEVELOPMENT,
) -> CaseRecord:
    return CaseRecord(
        case=Case(
            case_id=case_id,
            source=source,
            visibility=visibility,
            public_data={"case_id": case_id},
            round_index=round_index,
            parent_case_id=parent_case_id,
        ),
        truth=_Truth(),
    )


def _dataset(count: int = 4) -> PreparedDataset:
    return PreparedDataset(
        dataset=DatasetInput("test", DatasetKind.NUMERICAL),
        development_cases=tuple(_record(f"original-{index}") for index in range(count)),
        hidden_cases=(),
    )


class _Generator:
    capabilities = GeneratorCapabilities(
        offline=True,
        online=True,
        guidance=True,
        perturbations=True,
    )

    def __init__(self) -> None:
        self.requests = []
        self.release_calls = []
        self.live_case_ids: set[str] = set()
        self.return_wrong_count = False
        self.reject_perturbed_calls: set[int] = set()
        self.duplicate_perturbed_call: int | None = None
        self.wrong_parent = False
        self._serial = 0
        self._perturbed_calls = 0

    def sample(self, state, request, progress=None):
        self.requests.append(request)
        if request.kind is GenerationKind.PERTURBED:
            self._perturbed_calls += 1
            if self._perturbed_calls in self.reject_perturbed_calls:
                raise RecoverableSamplingRejection(
                    "retry this physical draw",
                    audit={"gate": "test", "call": self._perturbed_calls},
                )
        count = request.count - 1 if self.return_wrong_count else request.count
        records = []
        for index in range(count):
            self._serial += 1
            parent = (
                request.anchors[index]
                if request.kind is GenerationKind.PERTURBED
                else None
            )
            records.append(
                _record(
                    (
                        "generated-2"
                        if request.kind is GenerationKind.PERTURBED
                        and self._perturbed_calls == self.duplicate_perturbed_call
                        else f"generated-{self._serial}"
                    ),
                    source=(
                        CaseSource.PERTURBED
                        if parent is not None
                        else CaseSource.GENERATED
                    ),
                    parent_case_id=(
                        "wrong-parent"
                        if parent is not None and self.wrong_parent
                        else (parent.case_id if parent is not None else None)
                    ),
                    round_index=request.round_index,
                    visibility=request.visibility,
                )
            )
        result = tuple(records)
        self.live_case_ids.update(record.case.case_id for record in result)
        return result

    def release(self, state, case_ids):
        self.release_calls.append(tuple(case_ids))
        self.live_case_ids.difference_update(case_ids)


def _state(*, guided: bool = True) -> GeneratorState:
    checkpoint = GeneratorCheckpoint("test-generator", payload={})
    return GeneratorState(
        checkpoint=checkpoint,
        base=object(),
        guidance=object() if guided else None,
    )


class RoundCaseSamplerTests(unittest.TestCase):
    def sampler(
        self,
        *,
        max_cases: int = 6,
        generator: _Generator | None = None,
        guided: bool = True,
        strategy: SamplingStrategy = SamplingStrategy.AGENT_ADAPTIVE,
        generation_settings=None,
        record_reserver=None,
    ) -> RoundCaseSampler:
        return RoundCaseSampler(
            agent_id="agent-a",
            round_index=2,
            max_cases=max_cases,
            strategy=strategy,
            dataset=_dataset(),
            generator=generator,
            generator_state=_state(guided=guided) if generator is not None else None,
            seed=41,
            generation_settings=generation_settings,
            record_reserver=record_reserver,
        )

    def test_registry_failure_rolls_back_generated_batch_before_local_commit(self) -> None:
        generator = _Generator()

        def reject(records):
            self.assertEqual(len(records), 1)
            raise ValueError("global Case id collision")

        sampler = self.sampler(
            max_cases=1,
            generator=generator,
            record_reserver=reject,
        )

        with self.assertRaisesRegex(ValueError, "global Case id collision"):
            sampler.sample_cases(SampleCasesRequest(SampleSource.BASE, 1))

        self.assertEqual(sampler.issued_count, 0)
        self.assertEqual(sampler.remaining, 1)
        self.assertEqual(sampler.generated_case_ids, ())
        self.assertEqual(generator.live_case_ids, set())
        self.assertEqual(generator.release_calls, [("generated-1",)])

    def test_original_and_generated_cases_share_one_limit(self) -> None:
        generator = _Generator()
        sampler = self.sampler(max_cases=4, generator=generator)

        original = sampler.sample_cases(
            SampleCasesRequest(SampleSource.ORIGINAL, 2)
        )
        base = sampler.sample_cases(SampleCasesRequest(SampleSource.BASE, 2))

        self.assertEqual(original.remaining, 2)
        self.assertEqual(base.remaining, 0)
        self.assertEqual(sampler.issued_count, 4)
        with self.assertRaisesRegex(ValueError, "only 0 remain"):
            sampler.sample_cases(SampleCasesRequest(SampleSource.ORIGINAL, 1))
        self.assertEqual(len(generator.requests), 1)

    def test_close_releases_once_with_owning_state_and_preserves_audit(self) -> None:
        generator = _Generator()
        state = _state()
        sampler = RoundCaseSampler(
            agent_id="agent-a",
            round_index=2,
            max_cases=2,
            strategy=SamplingStrategy.AGENT_ADAPTIVE,
            dataset=_dataset(),
            generator=generator,
            generator_state=state,
            seed=41,
        )
        sampler.sample_paired_cases(PairedSampleRequest(SampleSource.BASE, 1))
        attempts = sampler.sampling_attempts

        sampler.close()
        sampler.close()

        self.assertTrue(sampler.closed)
        self.assertEqual(sampler.generated_case_ids, ())
        self.assertEqual(sampler.sampling_attempts, attempts)
        self.assertEqual(len(generator.release_calls), 1)
        self.assertEqual(set(generator.release_calls[0]), {"generated-1", "generated-2"})
        with self.assertRaisesRegex(RuntimeError, "cohort is closed"):
            sampler.sample_cases(SampleCasesRequest(SampleSource.BASE, 1))

    def test_guided_request_uses_round_frozen_state_and_online_mode(self) -> None:
        generator = _Generator()
        sampler = self.sampler(generator=generator)

        batch = sampler.sample_cases(SampleCasesRequest(SampleSource.GUIDED, 2))

        request = generator.requests[0]
        self.assertEqual(batch.count, 2)
        self.assertIs(request.kind, GenerationKind.GUIDED)
        self.assertEqual(request.round_index, 2)
        self.assertEqual(request.seed, 41)
        self.assertEqual(request.mode.value, "online")

    def test_guided_request_fails_without_frozen_guidance_and_spends_nothing(self) -> None:
        generator = _Generator()
        sampler = self.sampler(generator=generator, guided=False)

        with self.assertRaisesRegex(ValueError, "guidance frozen"):
            sampler.sample_cases(SampleCasesRequest(SampleSource.GUIDED, 2))

        self.assertEqual(sampler.issued_count, 0)
        self.assertEqual(sampler.remaining, 6)
        self.assertEqual(sampler.generated_case_ids, ())
        self.assertEqual(generator.requests, [])

    def test_invalid_generator_batch_spends_nothing(self) -> None:
        generator = _Generator()
        generator.return_wrong_count = True
        sampler = self.sampler(generator=generator)

        with self.assertRaisesRegex(ValueError, "requested 2 Cases but received 1"):
            sampler.sample_cases(SampleCasesRequest(SampleSource.BASE, 2))

        self.assertEqual(sampler.issued_count, 0)
        self.assertEqual(sampler.remaining, 6)
        self.assertEqual(sampler.generated_case_ids, ())
        self.assertEqual(generator.release_calls, [("generated-1",)])

    def test_paired_sampling_retries_only_the_failed_logical_root(self) -> None:
        generator = _Generator()
        generator.reject_perturbed_calls = {2}
        sampler = self.sampler(max_cases=6, generator=generator)

        batch = sampler.sample_paired_cases(
            PairedSampleRequest(SampleSource.BASE, 3, max_attempts_per_root=3)
        )

        self.assertEqual(
            [request.kind for request in generator.requests],
            [
                GenerationKind.BASE,
                GenerationKind.PERTURBED,
                GenerationKind.BASE,
                GenerationKind.PERTURBED,
                GenerationKind.BASE,
                GenerationKind.PERTURBED,
                GenerationKind.BASE,
                GenerationKind.PERTURBED,
            ],
        )
        self.assertEqual(batch.roots[0].case_id, "generated-1")
        self.assertEqual(batch.children[0].case_id, "generated-2")
        self.assertEqual(generator.release_calls, [("generated-3",)])
        self.assertEqual(
            [row.root_index for row in batch.attempts],
            [0, 1, 1, 2],
        )
        self.assertEqual(
            [row.accepted for row in batch.attempts],
            [True, False, True, True],
        )
        self.assertEqual(
            tuple(child.parent_case_id for child in batch.children),
            tuple(root.case_id for root in batch.roots),
        )
        self.assertEqual(sampler.issued_count, 6)
        self.assertEqual(batch.remaining, 0)

    def test_paired_sampling_seed_schedule_replays_exactly(self) -> None:
        signatures = []
        batches = []
        for _ in range(2):
            generator = _Generator()
            generator.reject_perturbed_calls = {1}
            sampler = self.sampler(max_cases=4, generator=generator)
            batches.append(
                sampler.sample_paired_cases(
                    PairedSampleRequest(
                        SampleSource.GUIDED,
                        2,
                        max_attempts_per_root=2,
                    )
                )
            )
            signatures.append(
                tuple(
                    (
                        request.kind,
                        request.seed,
                        tuple(anchor.case_id for anchor in request.anchors),
                    )
                    for request in generator.requests
                )
            )

        self.assertEqual(signatures[0], signatures[1])
        self.assertEqual(
            tuple(case.case_id for case in batches[0].cases),
            tuple(case.case_id for case in batches[1].cases),
        )
        seeds = [row[1] for row in signatures[0]]
        self.assertEqual(len(seeds), len(set(seeds)))

    def test_logical_root_offsets_follow_roots_across_operations(self) -> None:
        generator = _Generator()
        sampler = self.sampler(
            max_cases=6,
            generator=generator,
            generation_settings={"logical_root_offset": 7},
        )

        paired = sampler.sample_paired_cases(
            PairedSampleRequest(SampleSource.BASE, 2)
        )
        sampler.sample_cases(SampleCasesRequest(SampleSource.BASE, 1))
        sampler.perturb_cases(
            PerturbCasesRequest((paired.roots[1].case_id,))
        )

        self.assertEqual(
            [request.settings["logical_root_offset"] for request in generator.requests],
            [7, 7, 8, 8, 9, 8],
        )

    def test_paired_sampling_exhaustion_releases_every_owned_record(self) -> None:
        generator = _Generator()
        generator.reject_perturbed_calls = {2, 3}
        sampler = self.sampler(max_cases=4, generator=generator)

        with self.assertRaises(RecoverableSamplingRejection) as caught:
            sampler.sample_paired_cases(
                PairedSampleRequest(
                    SampleSource.BASE,
                    2,
                    max_attempts_per_root=2,
                )
            )

        self.assertEqual(caught.exception.audit["schema"], "paired_sampling_exhaustion_v1")
        self.assertEqual(caught.exception.audit["root_index"], 1)
        self.assertEqual(sampler.issued_count, 0)
        self.assertEqual(sampler.remaining, 4)
        self.assertEqual(sampler.generated_case_ids, ())
        self.assertEqual(generator.live_case_ids, set())
        self.assertEqual(
            {case_id for call in generator.release_calls for case_id in call},
            {"generated-1", "generated-2", "generated-3", "generated-4"},
        )
        released = [
            case_id for call in generator.release_calls for case_id in call
        ]
        self.assertEqual(len(released), len(set(released)))

    def test_exhausted_operation_advances_the_next_seed_namespace(self) -> None:
        generator = _Generator()
        generator.reject_perturbed_calls = {1, 2, 3, 4}
        sampler = self.sampler(max_cases=2, generator=generator)

        attempt_seeds = []
        for _ in range(2):
            start = len(generator.requests)
            with self.assertRaises(RecoverableSamplingRejection):
                sampler.sample_paired_cases(
                    PairedSampleRequest(
                        SampleSource.BASE,
                        1,
                        max_attempts_per_root=2,
                    )
                )
            attempt_seeds.append(
                tuple(request.seed for request in generator.requests[start:])
            )

        self.assertNotEqual(attempt_seeds[0], attempt_seeds[1])
        self.assertEqual(sampler.issued_count, 0)
        self.assertEqual(generator.live_case_ids, set())

    def test_invalid_child_lineage_releases_root_and_child(self) -> None:
        generator = _Generator()
        generator.wrong_parent = True
        sampler = self.sampler(max_cases=2, generator=generator)

        with self.assertRaisesRegex(ValueError, "exactly one child"):
            sampler.sample_paired_cases(
                PairedSampleRequest(SampleSource.BASE, 1)
            )

        self.assertEqual(sampler.issued_count, 0)
        self.assertEqual(sampler.generated_case_ids, ())
        self.assertEqual(generator.live_case_ids, set())
        released = [
            case_id for call in generator.release_calls for case_id in call
        ]
        self.assertEqual(released.count("generated-1"), 1)
        self.assertEqual(released.count("generated-2"), 1)

    def test_duplicate_child_id_is_released_once_with_atomic_rollback(self) -> None:
        generator = _Generator()
        generator.duplicate_perturbed_call = 2
        sampler = self.sampler(max_cases=4, generator=generator)

        with self.assertRaisesRegex(ValueError, "reused an active Case id"):
            sampler.sample_paired_cases(
                PairedSampleRequest(SampleSource.BASE, 2)
            )

        self.assertEqual(sampler.issued_count, 0)
        self.assertEqual(sampler.generated_case_ids, ())
        self.assertEqual(generator.live_case_ids, set())
        released = [
            case_id for call in generator.release_calls for case_id in call
        ]
        self.assertEqual(released.count("generated-2"), 1)
        self.assertEqual(set(released), {"generated-1", "generated-2", "generated-3"})

    def test_perturbation_routes_issued_parents_without_physics_settings(self) -> None:
        generator = _Generator()
        sampler = self.sampler(max_cases=5, generator=generator)
        roots = sampler.sample_cases(SampleCasesRequest(SampleSource.BASE, 2))

        children = sampler.perturb_cases(
            PerturbCasesRequest(tuple(case.case_id for case in roots.cases))
        )

        request = generator.requests[1]
        self.assertIs(request.kind, GenerationKind.PERTURBED)
        self.assertEqual(request.settings, {})
        self.assertEqual(
            {case.parent_case_id for case in children.cases},
            {case.case_id for case in roots.cases},
        )
        self.assertEqual(children.remaining, 1)

    def test_perturbation_rejects_unknown_or_already_perturbed_parent(self) -> None:
        generator = _Generator()
        sampler = self.sampler(max_cases=6, generator=generator)

        with self.assertRaisesRegex(ValueError, "was not issued"):
            sampler.perturb_cases(PerturbCasesRequest(("missing",)))

        root = sampler.sample_cases(SampleCasesRequest(SampleSource.BASE, 1))
        child = sampler.perturb_cases(
            PerturbCasesRequest((root.cases[0].case_id,))
        )
        with self.assertRaisesRegex(ValueError, "clean base or guided Cases"):
            sampler.perturb_cases(
                PerturbCasesRequest((child.cases[0].case_id,))
            )
        self.assertEqual(sampler.issued_count, 2)

    def test_perturbation_rejects_original_dataset_parent(self) -> None:
        generator = _Generator()
        sampler = self.sampler(max_cases=3, generator=generator)
        original = sampler.sample_cases(
            SampleCasesRequest(SampleSource.ORIGINAL, 1)
        )

        with self.assertRaisesRegex(
            ValueError,
            "clean base or guided Cases",
        ):
            sampler.perturb_cases(
                PerturbCasesRequest((original.cases[0].case_id,))
            )

    def test_fixed_recipe_prefers_supported_online_generator_adapter(self) -> None:
        generator = _Generator()
        sampler = self.sampler(
            generator=generator,
            strategy=SamplingStrategy.FIXED_RECIPE,
        )

        sampler.sample_cases(SampleCasesRequest(SampleSource.BASE, 1))

        self.assertEqual(generator.requests[0].mode.value, "online")

    def test_both_strategies_fall_back_to_an_offline_only_generator(self) -> None:
        for strategy in (
            SamplingStrategy.AGENT_ADAPTIVE,
            SamplingStrategy.FIXED_RECIPE,
        ):
            with self.subTest(strategy=strategy.value):
                generator = _Generator()
                generator.capabilities = GeneratorCapabilities(
                    offline=True,
                    online=False,
                    guidance=True,
                    perturbations=True,
                )
                sampler = self.sampler(generator=generator, strategy=strategy)

                sampler.sample_cases(SampleCasesRequest(SampleSource.BASE, 1))

                self.assertEqual(generator.requests[0].mode.value, "offline")

    def test_sampling_rejects_a_generator_with_no_compatible_mode(self) -> None:
        generator = _Generator()
        generator.capabilities = GeneratorCapabilities(
            offline=False,
            online=False,
            guidance=True,
            perturbations=True,
        )
        sampler = self.sampler(generator=generator)

        with self.assertRaisesRegex(ValueError, "neither online nor offline"):
            sampler.sample_cases(SampleCasesRequest(SampleSource.BASE, 1))

        self.assertEqual(generator.requests, [])
        self.assertEqual(sampler.remaining, 6)


class SamplingContractTests(unittest.TestCase):
    def test_requests_reject_empty_or_duplicate_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            SampleCasesRequest(SampleSource.BASE, 0)
        with self.assertRaisesRegex(ValueError, "at least one"):
            PerturbCasesRequest(())
        with self.assertRaisesRegex(ValueError, "unique"):
            PerturbCasesRequest(("case-a", "case-a"))


if __name__ == "__main__":
    unittest.main()
