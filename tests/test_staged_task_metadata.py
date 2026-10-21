import tempfile
import unittest
from pathlib import Path

from pde_agent_verifier_solver.engine.staged import (
    LocalStageRunner,
    RunStage,
    StageCommand,
    StagedRunPlan,
)


class StagedTaskMetadataTests(unittest.TestCase):
    def test_task_is_visible_in_plan_and_completed_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "done.txt"
            plan = StagedRunPlan(
                name="one-task run",
                problem="example_pde",
                task="statistics",
                run_root=root / "run",
                workdir=root,
                product_root=root,
                implementation_root=root,
                stages=(
                    RunStage(
                        "finish",
                        "Finish",
                        (
                            StageCommand(
                                (
                                    "python",
                                    "-c",
                                    (
                                        "from pathlib import Path; "
                                        f"Path({str(output)!r}).write_text('done')"
                                    ),
                                )
                            ),
                        ),
                        expected_outputs=(output,),
                    ),
                ),
            )

            self.assertEqual(plan.summary()["task"], "statistics")
            completed = LocalStageRunner().run(
                plan,
                legacy_staged_intent=True,
            )
            self.assertEqual(completed["task"], "statistics")


if __name__ == "__main__":
    unittest.main()
