"""Small synthetic provenance tests, not real-data numerical validation."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from pde_agent_verifier_solver.engine.agent_wire import AgentWireCodec
from pde_agent_verifier_solver.engine.candidate_case import candidate_facing_case
from pde_agent_verifier_solver.engine.candidate_execution import _case_to_payload, _plan_to_payload
from pde_agent_verifier_solver.sdk import (
    AnswerQuery, Case, CaseSource, CaseVisibility, EvaluationPhase, EvaluationPlan,
    QueryKind, freeze_candidate_bundle,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("artemis_round_recovery", ROOT / "scripts/artemis_round_recovery.py")
recovery = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = recovery
SPEC.loader.exec_module(recovery)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


class RoundOneRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.commit, self.backend = "a" * 40, "b" * 64
        self.agent_ids = ("explorer", "refiner")
        self.config = {
            "backend": {"sha256": self.backend}, "official": {"base": 128},
            "agents": [{"agent_id": name} for name in self.agent_ids],
        }
        write_json(self.root / "resolved_config.json", self.config)
        write_json(self.root / "source_state.json", {"commit": self.commit, "status_porcelain": ""})
        write_json(self.root / "input_provenance.json", {
            "backend_config_sha256": self.backend, "initial_bundle_tree_sha256": "c" * 64,
        })
        self.history = [{"artifact_id": "candidate-s0", "round": 0}]
        self.events = []
        self.bundles = {}
        for agent_id, candidate_indices in (("explorer", [1]), ("refiner", [2, 3, 4])):
            execution_id = agent_id + "123"
            workspace = self.root / "agent_turns" / f"round-0001-{agent_id}-{execution_id}"
            ids = []
            for sequence, index in enumerate(candidate_indices, 1):
                artifact_id = f"candidate-r1-{index:04d}"
                ids.append(artifact_id)
                source = self.root / f"source-{index}.py"
                source.write_text(f"VALUE = {index}\ndef prepare_solution(case, solve_budget=None):\n    return None\n")
                submitted = freeze_candidate_bundle(source, workspace / ".submitted-candidates" / f"nomination-{sequence:08d}")
                frozen = freeze_candidate_bundle(submitted, workspace / ".controller-candidates")
                self.bundles[artifact_id] = frozen
                body = {"bundle_path": str(submitted.root.relative_to(workspace)),
                        "name": f"Candidate {index}", "summary": f"Summary {index}", "usage": "prepare then evaluate"}
                write_json(workspace / "nominations" / f"nomination-{sequence:08d}.json", {
                    "schema": "canonical-agent-nomination-v1", "sequence": sequence,
                    "execution_id": execution_id, "payload": body,
                })
                self.history.append({"artifact_id": artifact_id, "round": 1, "agent_id": agent_id,
                                     "title": body["name"], "summary": body["summary"]})
            self.events.append({"event_type": "agent_finished", "round_index": 1,
                                "payload": {"agent_id": agent_id, "artifact_ids": ids, "nomination_count": len(ids)}})
        self.events.extend([
            {"event_type": "campaign_memory_updated", "round_index": 1, "payload": {
                "candidate_history": self.history,
                "round_memory": {"round": 1, "status": "completed", "agents": [
                    {"agent_id": name, "agent_name": name, "markdown": f"Original {name} R1 notes", "status": "completed"}
                    for name in self.agent_ids], "markdown": "OLD OFFICIAL CONCLUSION MUST NOT BE IMPORTED"},
            }},
            {"event_type": "round_finished", "round_index": 1,
             "payload": {"agent_case_counts": {name: {"base": 16} for name in self.agent_ids}}},
            # This must never be parsed or included in any returned memory.
            {"event_type": "campaign_memory_updated", "round_index": 2,
             "payload": {"round_memory": {"agents": [{"markdown": "R2 FUTURE KNOWLEDGE"}]}}},
        ])
        self.write_events()
        self.cases = tuple(Case(case_id=f"official-r1-{index}", source=CaseSource.GENERATED,
                                visibility=CaseVisibility.HIDDEN, public_data={
                                    "arrays": {"f0": np.array([index, index + 0.25])},
                                    "equation": {"case_id": f"official-r1-{index}", "cohort": "base", "coefficient": 1.0},
                                }, round_index=1) for index in range(128))
        self.plans = tuple(EvaluationPlan(
            phase=EvaluationPhase.OFFICIAL, query=AnswerQuery(kind=QueryKind.POINTS,
                fields=("M0",), points=({"t": 0.5, "x": 0.25},)), seed=index, settings={"budget": 600},
        ) for index in range(128))
        self.audits = [{"phase": "official", "artifact_id": "candidate-r1-0002", "case_index": index,
                        "execution_id": "success", "status": "completed",
                        "case_id": candidate_facing_case(case).case_id,
                        "workspace": str(self.root / "candidate_execution" / "deleted-success" / str(index))}
                       for index, case in enumerate(self.cases)]
        self.add_retained("candidate-r1-0001", 0)
        self.add_retained("candidate-r1-0003", 0)
        self.add_retained("candidate-r1-0004", 55)
        # An R2 path would fail JSON decoding if the reader inspected it.
        future = self.root / "candidate_execution" / "r2" / "case"
        future.mkdir(parents=True)
        (future / "request.json").write_text("DO NOT READ R2")
        self.audits.append({"phase": "official", "artifact_id": "candidate-r2-0005", "workspace": str(future)})
        self.write_audits()

    def write_events(self):
        (self.root / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in self.events))

    def write_audits(self):
        (self.root / "candidate_execution/candidate_execution_audits.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in self.audits))

    def add_retained(self, artifact_id, index):
        workspace = self.root / "candidate_execution" / artifact_id / f"case-{index}"
        codec = AgentWireCodec(workspace / "arrays", inline_array_bytes=1)
        case = candidate_facing_case(self.cases[index])
        write_json(workspace / "request.json", {
            "execution_id": artifact_id, "case_index": index, "sidecar_root": str(codec.sidecar_root),
            "artifact": {"artifact_id": artifact_id, "round_index": 1,
                         "agent_id": "explorer" if artifact_id.endswith("0001") else "refiner",
                         "bundle_tree_sha256": self.bundles[artifact_id].tree_sha256},
            "case": _case_to_payload(case, codec), "plan": _plan_to_payload(self.plans[index], codec),
            "public_oracle": {"token": "NEVER_EXPORT_THIS_CAPABILITY", "endpoint": "http://source-oracle.invalid"},
        })
        self.audits.append({"phase": "official", "artifact_id": artifact_id, "case_index": index,
                            "execution_id": artifact_id, "case_id": case.case_id,
                            "status": "failed", "workspace": str(workspace)})

    def load(self, **overrides):
        kwargs = {"expected_source_head": self.commit, "expected_backend_config_sha256": self.backend,
                  "expected_config_sha256": hashlib.sha256((self.root / "resolved_config.json").read_bytes()).hexdigest()}
        kwargs.update(overrides)
        return recovery.load_frozen_round_one(self.root, **kwargs)

    def test_freezes_only_r1_with_honest_partial_coverage_and_no_writes(self):
        before = {str(path): (path.stat().st_mode, path.stat().st_mtime_ns, path.stat().st_ctime_ns)
                  for path in self.root.rglob("*")}
        frozen = self.load()
        after = {str(path): (path.stat().st_mode, path.stat().st_mtime_ns, path.stat().st_ctime_ns)
                 for path in self.root.rglob("*")}
        self.assertEqual(before, after)
        self.assertEqual([a.artifact_id for a in frozen.artifacts], [f"candidate-r1-{i:04d}" for i in range(1, 5)])
        self.assertEqual(len(frozen.original_case_ids), 128)
        self.assertEqual(frozen.manifest["numerical_reference_count"], 2)
        self.assertFalse(frozen.manifest["full_numerical_reference_saved"])
        exported = repr(frozen.manifest) + repr(frozen.agent_logs)
        for prohibited in ("NEVER_EXPORT_THIS_CAPABILITY", "R2 FUTURE KNOWLEDGE", "OLD OFFICIAL CONCLUSION"):
            self.assertNotIn(prohibited, exported)
        report = recovery.verify_official_replay(frozen, self.cases, self.plans)
        self.assertEqual(report["identity_order_verified_count"], 128)
        self.assertEqual(report["private_plan_verified_count"], 2)
        self.assertFalse(report["hidden_answer_equality_claimed"])

    def test_source_revision_and_hash_fail_closed(self):
        with self.assertRaises(ValueError):
            self.load(expected_source_head="wrong")
        self.load(expected_config_sha256="0" * 64,
                  expected_backend_config_sha256="0" * 64)

    def test_later_round_history_rejected(self):
        self.history.append({"artifact_id": "candidate-r2-0005", "round": 2})
        self.write_events()
        with self.assertRaisesRegex(ValueError, "contaminated"):
            self.load()

    def test_missing_identity_and_duplicate_batch_rejected(self):
        self.audits.pop(40)
        self.write_audits()
        with self.assertRaisesRegex(ValueError, "identity audit"):
            self.load()

    def test_shared_case_disagreement_rejected(self):
        path = self.root / "candidate_execution/candidate-r1-0003/case-0/request.json"
        data = json.loads(path.read_text())
        data["plan"]["seed"] = 777
        write_json(path, data)
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            self.load()

    def test_changed_identity_order_public_or_private_values_rejected(self):
        frozen = self.load()
        changed_cases = list(self.cases)
        changed_cases[0], changed_cases[1] = changed_cases[1], changed_cases[0]
        with self.assertRaisesRegex(ValueError, "identity/order"):
            recovery.verify_official_replay(frozen, changed_cases, self.plans)
        changed_cases = list(self.cases)
        changed_cases[0] = replace(changed_cases[0], public_data={"arrays": {"f0": np.array([999.0])}})
        with self.assertRaisesRegex(ValueError, "numerical content"):
            recovery.verify_official_replay(frozen, changed_cases, self.plans)
        changed_plans = list(self.plans)
        changed_plans[55] = replace(changed_plans[55], seed=999)
        with self.assertRaisesRegex(ValueError, "private plan"):
            recovery.verify_official_replay(frozen, self.cases, changed_plans)

    def test_retained_source_path_escape_rejected(self):
        path = self.root / "candidate_execution/candidate-r1-0003/case-0/request.json"
        data = json.loads(path.read_text())
        data["sidecar_root"] = str(self.root)
        write_json(path, data)
        with self.assertRaisesRegex(ValueError, "escaped"):
            self.load()

    def test_later_round_updates_do_not_change_frozen_manifest(self):
        original = self.load().source_manifest_sha256
        self.events[-1]["payload"] = {"arbitrary": "changed R2 state"}
        self.write_events()
        self.assertEqual(self.load().source_manifest_sha256, original)

    def test_later_official_incumbent_batch_does_not_enter_r1_recovery(self):
        original = self.load().source_manifest_sha256
        self.audits.append({"phase": "official", "artifact_id": "candidate-r1-0002", "execution_id": "r2-incumbent",
                            "workspace": str(self.root / "candidate_execution/r2/case"),
                            "case_id": "not-r1", "case_index": 0, "status": "completed"})
        self.write_audits()
        self.assertEqual(self.load().source_manifest_sha256, original)

    def test_staging_preserves_exact_bundles_and_source_manifest(self):
        frozen = self.load()
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "r1-frozen"
            staged = recovery.stage_frozen_round_one(frozen, destination)
            self.assertEqual(staged.source_manifest_sha256, frozen.source_manifest_sha256)
            self.assertEqual(staged.manifest, frozen.manifest)
            self.assertEqual(len(tuple(destination.iterdir())), 4)
            for original, copied in zip(frozen.artifacts, staged.artifacts):
                self.assertEqual(original.program.tree_sha256, copied.program.tree_sha256)
                self.assertNotEqual(original.program.bundle_locator, copied.program.bundle_locator)
                self.assertTrue(copied.program.bundle_locator.is_relative_to(destination))
                self.assertEqual(original.program.bundle_locator.read_bytes(), copied.program.bundle_locator.read_bytes())
            with self.assertRaises(FileExistsError):
                recovery.stage_frozen_round_one(frozen, destination)

    def test_smoke_metric_accounting_gate(self):
        from check_artemis_budget_recovery import validate_case_metrics
        metrics = {"candidate_solve_wall_seconds": 10.0, "candidate_solve_active_seconds": 4.0,
                   "candidate_platform_wait_seconds": 6.0, "candidate_oracle_queue_wait_seconds": 5.0,
                   "candidate_oracle_lock_wait_seconds": 1.0, "candidate_oracle_timed_requests": 4.0}
        validate_case_metrics(metrics)
        for changed in ({"candidate_solve_active_seconds": 10.0}, {"candidate_oracle_queue_wait_seconds": 4.0},
                        {"candidate_solve_wall_seconds": float("nan")}, {"candidate_platform_wait_seconds": -1.0}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_case_metrics({**metrics, **changed})


if __name__ == "__main__":
    unittest.main()
