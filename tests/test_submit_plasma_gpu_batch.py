"""Submission tests never contact Slurm or read authentication contents."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import submit_plasma_gpu_batch as batch
from scripts import ns_slurm_continuation as continuation


class PlasmaBatchTests(unittest.TestCase):
    def test_resources_and_shell_syntax(self):
        for equation in ("strong", "weak", "artemis"):
            resources = batch.resources(equation)
            self.assertEqual(resources["gpu_count"], 4)
            self.assertEqual(resources["cpus"], 24)
            site = dict(resources, source_root="/tmp/source", run_root="/tmp/run",
                        python="/tmp/venv/bin/python", account="${PDE_GAV_SLURM_ACCOUNT}",
                        partition="gpubase_l40s_b2", qos="normal", arm_name=equation+"-sol")
            script = batch.slurm_text(site, Path("/tmp/site.json"))
            subprocess.run(["bash", "-n"], input=script, text=True, check=True)
            self.assertIn("--gres=gpu:l40s:4", script)
            self.assertIn("PDE_AGENT_WORKER_GPU_DEVICES=0,1", script)
            self.assertNotIn("scancel", script)

    def test_new_partition_selection_matches_exact_successor_resources(self):
        self.assertEqual(batch.PLASMA_L40S_PARTITIONS, "gpubase_l40s_b2,gpubase_l40s_b3")
        with tempfile.TemporaryDirectory() as temporary:
            wrapper = Path(temporary) / "run.slurm"
            for equation in ("strong", "weak", "artemis"):
                site = dict(batch.resources(equation), source_root="/tmp/source",
                            run_root="/tmp/run", python="/tmp/python", account="${PDE_GAV_SLURM_ACCOUNT}",
                            partition=batch.PLASMA_L40S_PARTITIONS, qos="normal",
                            arm_name=equation + "-sol")
                script = batch.slurm_text(site, Path("/tmp/site.json"))
                subprocess.run(["bash", "-n"], input=script, text=True, check=True)
                wrapper.write_text(script)
                profile = "plasma_gpu_6h_b2_b3" if equation == "artemis" else "plasma_gpu_12h_b2_b3"
                spec = continuation._submission_spec(wrapper, (), 4, profile)
                for flag in continuation.submission_resource_flags(spec):
                    self.assertIn("#SBATCH " + flag + "\n", script)
                self.assertIn("NS_CONTINUATION_SLURM_PARTITION=" + batch.PLASMA_L40S_PARTITIONS, script)

    def test_prepare_defaults_all_six_sites_to_combined_partitions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, configs = root / "source", root / "configs"
            source.mkdir()
            configs.mkdir()
            (source / "source.py").write_text("# frozen fixture\n")
            runtime, auth = root / "codex", root / "auth.json"
            runtime.write_text("#!/bin/sh\nexit 0\n")
            runtime.chmod(0o700)
            auth.write_text("{}")
            auth.chmod(0o600)
            arms = []
            for name in batch.NAMES:
                config = configs / (name + ".json")
                config.write_text(json.dumps({"equation": name.split("-")[0], "campaign_id": name}))
                arms.append({"name": name, "config": str(config), "sha256": batch.digest(config)})
            (configs / "migration.json").write_text(json.dumps({"arms": arms}))
            with patch.object(batch, "ROOT", source), \
                 patch.object(batch, "clean_source", return_value="f" * 40), \
                 patch.object(batch, "RUNTIME_SHA256", batch.digest(runtime)), \
                 patch.object(batch.subprocess, "check_output", return_value=b"source.py\0"), \
                 patch("plasma_gpu_campaign.load_config", side_effect=batch.read), \
                 patch("plasma_gpu_campaign.static_check"):
                plan = batch.prepare(configs, root / "deployment", run_parent=root / "runs",
                                     python=sys.executable, codex_binary=runtime, auth_file=auth)
            self.assertEqual(len(plan["entries"]), 6)
            for row in plan["entries"]:
                site = batch.read(row["site"])
                self.assertEqual(site["partition"], batch.PLASMA_L40S_PARTITIONS)
                self.assertIn("#SBATCH --partition=" + batch.PLASMA_L40S_PARTITIONS,
                              Path(row["script"]).read_text())

    def make_plan(self, root):
        entries = []
        for name in batch.NAMES:
            site, script = root / (name + ".json"), root / (name + ".slurm")
            site.write_text("{}")
            script.write_text("#!/bin/bash\n")
            entries.append(dict(name=name, site=str(site), script=str(script),
                                site_sha256=batch.digest(site), script_sha256=batch.digest(script)))
        path = root / "plan.json"
        path.write_text(json.dumps(dict(schema="pde_gav_six_gpu_submission_v1", status="prepared",
                            source_root=str(batch.ROOT), source_revision="f"*40, entries=entries)))
        return path

    def test_all_six_acknowledged_and_no_second_submission(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = self.make_plan(root)
            replies = [subprocess.CompletedProcess([], 0, str(8000000+i)+";cluster\n", "")
                       for i in range(6)]
            with patch.object(batch, "clean_source", return_value="f"*40), \
                 patch("run_plasma_gpu_segment.verify_site"), \
                 patch.object(batch.subprocess, "run", side_effect=replies) as run:
                result = batch.submit(plan)
                self.assertEqual(result["status"], "all_six_submitted")
                self.assertEqual(len(result["jobs"]), 6)
                self.assertFalse(result["old_jobs_cancelled"])
                self.assertEqual([call.args[0][:2] for call in run.call_args_list],
                                 [["sbatch", "--parsable"]]*6)
                with self.assertRaises(FileExistsError):
                    batch.submit(plan)
                self.assertEqual(run.call_count, 6)

    def test_uncertain_outcome_stops_with_durable_intent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = self.make_plan(root)
            with patch.object(batch, "clean_source", return_value="f"*40), \
                 patch("run_plasma_gpu_segment.verify_site"), \
                 patch.object(batch.subprocess, "run", side_effect=[
                     subprocess.CompletedProcess([], 0, "8000000\n", ""),
                     subprocess.TimeoutExpired("sbatch", 45)]) as run:
                with self.assertRaises(subprocess.TimeoutExpired):
                    batch.submit(plan)
                self.assertEqual(run.call_count, 2)
                receipt = json.loads((root/"submissions.json").read_text())
                self.assertEqual(receipt["status"], "submission_unknown")
                self.assertEqual(receipt["pending_arm"], "strong-astra")
                self.assertEqual(len(receipt["jobs"]), 1)
                with self.assertRaises(FileExistsError):
                    batch.submit(plan)
                self.assertEqual(run.call_count, 2)

    def test_changed_wrapper_fails_before_any_submission(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = self.make_plan(root)
            (root/"artemis-astra.slurm").write_text("changed\n")
            with patch.object(batch, "clean_source", return_value="f"*40), \
                 patch("run_plasma_gpu_segment.verify_site"), \
                 patch.object(batch.subprocess, "run") as run:
                with self.assertRaises(ValueError):
                    batch.submit(plan)
                run.assert_not_called()
                self.assertFalse((root/"submissions.json").exists())


if __name__ == "__main__":
    unittest.main()
