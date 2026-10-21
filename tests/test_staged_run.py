import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pde_agent_verifier_solver.engine.staged import (
    LocalStageRunner,
    RunStage,
    SlurmStageLauncher,
    StageCommand,
    StagedRunPlan,
    _render_slurm_script,
    read_staged_run,
)


class StagedRunTests(unittest.TestCase):
    def test_runners_reject_execution_without_explicit_legacy_intent(self) -> None:
        for action in (
            lambda: LocalStageRunner().run(None),
            lambda: SlurmStageLauncher().launch(None),
            lambda: SlurmStageLauncher().resume(None),
        ):
            with self.subTest(action=action), self.assertRaisesRegex(
                PermissionError,
                "legacy_staged_intent=True",
            ):
                action()

    def test_slurm_script_reloads_scientific_modules(self) -> None:
        root = Path("/tmp/ns-stage-test")
        stage = RunStage(
            "project_train",
            "Project",
            (StageCommand(("python", "project.py")),),
        )
        plan = StagedRunPlan(
            name="NS",
            problem="ns",
            run_root=root / "run",
            workdir=root,
            product_root=root / "product",
            implementation_root=root / "implementation",
            stages=(stage,),
            modules=("scipy-stack", "hdf5"),
        )

        script = _render_slurm_script(plan, stage)

        self.assertLess(script.index("source /etc/profile"), script.index("module -q purge"))
        self.assertLess(script.index("module -q purge"), script.index("module load scipy-stack hdf5"))
        self.assertLess(script.index("module load scipy-stack hdf5"), script.index("python project.py"))

    def test_local_pipeline_persists_stage_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_root = root / "run"
            first = root / "first.txt"
            second = root / "second.txt"
            plan = StagedRunPlan(
                name="tiny staged run",
                problem="test",
                run_root=run_root,
                workdir=root,
                product_root=root,
                implementation_root=root,
                stages=(
                    RunStage(
                        "prepare",
                        "Prepare",
                        (
                            StageCommand(
                                (
                                    sys.executable,
                                    "-c",
                                    f"from pathlib import Path; Path({str(first)!r}).write_text('ready')",
                                )
                            ),
                        ),
                        expected_outputs=(first,),
                    ),
                    RunStage(
                        "finish",
                        "Finish",
                        (
                            StageCommand(
                                (
                                    sys.executable,
                                    "-c",
                                    f"from pathlib import Path; Path({str(second)!r}).write_text(Path({str(first)!r}).read_text())",
                                )
                            ),
                        ),
                        depends_on=("prepare",),
                        expected_outputs=(second,),
                    ),
                ),
            )
            result = LocalStageRunner().run(plan, legacy_staged_intent=True)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(second.read_text(), "ready")
            refreshed = read_staged_run(run_root)
            self.assertEqual(
                [stage["status"] for stage in refreshed["stages"]],
                ["completed", "completed"],
            )

    def test_slurm_resume_reuses_outputs_and_submits_only_missing_stages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_root = root / "run"
            for directory in (run_root, run_root / "runtime", run_root / "logs"):
                directory.mkdir(exist_ok=True)
            (run_root / "run.json").write_text(
                json.dumps(
                    {
                        "name": "formal NS",
                        "problem": "ns",
                        "backend": "slurm",
                        "created_at": "2026-08-20T10:00:00+00:00",
                        "stages": [],
                    }
                ),
                encoding="utf-8",
            )
            (run_root / "jobs.json").write_text(
                json.dumps({"jobs": {}}), encoding="utf-8"
            )
            prepared = root / "prepared.txt"
            finished = root / "finished.txt"
            prepared.write_text("ready", encoding="utf-8")
            plan = StagedRunPlan(
                name="formal NS",
                problem="ns",
                run_root=run_root,
                workdir=root,
                product_root=root,
                implementation_root=root,
                stages=(
                    RunStage(
                        "prepare",
                        "Prepare",
                        (StageCommand(("python", "prepare.py")),),
                        expected_outputs=(prepared,),
                    ),
                    RunStage(
                        "finish",
                        "Finish",
                        (StageCommand(("python", "finish.py")),),
                        depends_on=("prepare",),
                        expected_outputs=(finished,),
                    ),
                ),
            )
            submitted = subprocess.CompletedProcess(
                args=["sbatch"], returncode=0, stdout="9001\n", stderr=""
            )

            with patch(
                "pde_agent_verifier_solver.engine.staged.shutil.which",
                return_value="/usr/bin/sbatch",
            ), patch(
                "pde_agent_verifier_solver.engine.staged.subprocess.run",
                return_value=submitted,
            ) as run_command:
                result = SlurmStageLauncher().resume(
                    plan,
                    legacy_staged_intent=True,
                )

            self.assertEqual(result["reused_stages"], ["prepare"])
            self.assertEqual(result["jobs"], {"finish": "9001"})
            self.assertEqual(run_command.call_count, 1)
            updated = json.loads((run_root / "run.json").read_text())
            self.assertEqual(
                [stage["id"] for stage in updated["stages"]],
                ["prepare", "finish"],
            )
            prepared_runtime = json.loads(
                (run_root / "runtime" / "prepare.json").read_text()
            )
            self.assertEqual(prepared_runtime["status"], "completed")
            self.assertTrue((run_root / "stages" / "finish.sh").is_file())


if __name__ == "__main__":
    unittest.main()
