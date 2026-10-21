import unittest

from pde_agent_verifier_solver.plugins.toy import (
    ToyPoissonGenerator,
    ToyPoissonProblem,
)
from pde_agent_verifier_solver.sdk import (
    CaseFeedback,
    CaseSource,
    CaseVisibility,
    DatasetInput,
    DatasetKind,
    GenerationKind,
    GenerationPurpose,
    GenerationRequest,
    GeneratorFeedback,
    GeneratorMode,
)


class UnifiedGeneratorProgramTests(unittest.TestCase):
    def setUp(self) -> None:
        self.problem = ToyPoissonProblem()
        self.dataset = self.problem.prepare_dataset(
            DatasetInput(
                dataset_id="toy-generator-contract",
                kind=DatasetKind.NUMERICAL,
                location="memory://toy",
                settings={"development_cases": 12, "hidden_cases": 4},
            )
        )
        self.generator = ToyPoissonGenerator()

    def test_training_and_loading_are_explicit_separate_operations(self) -> None:
        checkpoint = self.generator.train(self.dataset, {"basis_size": 8})
        state = self.generator.load(checkpoint, {})

        self.assertEqual(checkpoint.generator_id, self.generator.info.program_id)
        self.assertIs(state.checkpoint, checkpoint)
        self.assertIsNone(state.guidance)
        self.assertFalse(hasattr(self.generator, "prepare"))

    def test_guided_clean_cases_can_be_perturbation_anchors(self) -> None:
        checkpoint = self.generator.train(self.dataset, {"basis_size": 8})
        state = self.generator.load(checkpoint, {})
        feedback_case = self.dataset.hidden_cases[0].case
        state = self.generator.update_guidance(
            state,
            GeneratorFeedback(
                source_round_index=1,
                target_round_index=2,
                selected_program_id="solver-round-1",
                cases=(
                    CaseFeedback(
                        case=feedback_case,
                        primary_error=0.5,
                        metrics={"relative_l2": 0.5},
                    ),
                ),
            ),
            {},
        )
        guided = self.generator.sample(
            state,
            GenerationRequest(
                round_index=2,
                count=2,
                mode=GeneratorMode.OFFLINE,
                kind=GenerationKind.GUIDED,
                visibility=CaseVisibility.DEVELOPMENT,
                seed=20,
                purpose=GenerationPurpose.DEVELOPMENT,
            ),
        )
        children = self.generator.sample(
            state,
            GenerationRequest(
                round_index=2,
                count=2,
                mode=GeneratorMode.OFFLINE,
                kind=GenerationKind.PERTURBED,
                visibility=CaseVisibility.DEVELOPMENT,
                seed=21,
                purpose=GenerationPurpose.DEVELOPMENT,
                anchors=tuple(record.case for record in guided),
            ),
        )

        self.assertTrue(
            all(record.case.source is CaseSource.GENERATED for record in guided)
        )
        self.assertEqual(
            [record.case.parent_case_id for record in children],
            [record.case.case_id for record in guided],
        )
        self.assertTrue(
            all(record.case.source is CaseSource.PERTURBED for record in children)
        )


if __name__ == "__main__":
    unittest.main()
