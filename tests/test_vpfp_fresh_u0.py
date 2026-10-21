import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pde_agent_verifier_solver.engine.macro_closed_loop import (
    FinalistEvaluation,
    MacroLoopResult,
    MacroRoundResult,
    PairedPublishedMetricSummary,
    PublishedMetricAggregate,
    PublishedMetricSummary,
    RoundFeedback,
    RoundHandoff,
    TerminalEvaluation,
)
from pde_agent_verifier_solver.engine.macro_evaluator import (
    MacroBatchEvaluation,
    SanitizedMacroCaseScore,
)
from pde_agent_verifier_solver.engine.selection import (
    EvidenceStage,
    MacroCandidateScore,
    PromotionResult,
    aggregate,
    paired_compare,
)
from pde_agent_verifier_solver.plugins.vpfp import fresh_u0
from pde_agent_verifier_solver.sdk import CandidateRef, ExecutionAssurance


PRIVATE_METRIC_VALUE = 937.125
CONTRACT_ID = "vpfp_macro_test_v1"
COHORT_SHA = "c" * 64
PRIVATE_TARGET_SHA = "d" * 64


class _CaptureConfig:
    captured = None

    def __init__(self, **values):
        type(self).captured = values


class VPFPU0RunnerTests(unittest.TestCase):
    def test_cli_requires_explicit_legacy_macro_opt_in(self):
        with tempfile.TemporaryDirectory() as temporary:
            arguments = _arguments(_inputs(Path(temporary)))
            arguments.remove("--legacy-macro-reference")
            stderr = io.StringIO()

            with contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as error:
                    fresh_u0.main(arguments)

            self.assertEqual(error.exception.code, 2)
            self.assertIn("history/control-arm", stderr.getvalue())
            self.assertIn("--legacy-macro-reference", stderr.getvalue())
            self.assertIn("legacy VPFP", fresh_u0._parser().format_help())

    def test_cli_composes_shared_loop_and_publishes_sanitized_product_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = _inputs(root)
            paths["cold"].write_text(
                "calls = 0\n"
                "def predict_macros(case, query, budget=None):\n"
                "    global calls\n"
                "    calls += 1\n"
                "    return calls\n",
                encoding="utf-8",
            )
            selected_source = (
                "# selected fresh Agent program\n"
                "def predict_macros(case, query, budget=None):\n"
                "    return None\n"
            ).encode("utf-8")
            paths["selected"].write_bytes(selected_source)
            selected_sha = hashlib.sha256(selected_source).hexdigest()
            captured = {}

            class _Runner:
                def __init__(self, **values):
                    captured["runner"] = values

                def run(self):
                    cold_start = captured["runner"]["cold_start"]
                    cold_ref = cold_start.ref
                    selection = cold_start.fresh_program()
                    terminal = cold_start.fresh_program()
                    captured["formal_runtime_calls"] = (
                        selection.predict_macros(None, None),
                        selection.predict_macros(None, None),
                        terminal.predict_macros(None, None),
                    )
                    return _loop_result(cold_ref, paths["selected"], selected_sha)

            plugin = SimpleNamespace(
                generator=SimpleNamespace(
                    state=SimpleNamespace(
                        base=SimpleNamespace(
                            report_sha256=hashlib.sha256(
                                paths["report"].read_bytes()
                            ).hexdigest(),
                            decoder_checkpoint_sha256=hashlib.sha256(
                                paths["checkpoint"].read_bytes()
                            ).hexdigest(),
                        )
                    )
                )
            )
            environment = {
                "PYTHONPATH": "/public/product/src",
                "VIRTUAL_ENV": "/public/venv",
                "SLURM_JOB_ID": "12345",
                "USER": "researcher",
                "OPENAI_API_KEY": "must-not-leak",
                "CONTROLLER_SECRET": "must-not-leak",
                "HOME": "/controller/home",
            }
            with (
                patch.object(
                    fresh_u0,
                    "bind_vpfp_release_bank_reference",
                    return_value=plugin,
                ) as bind,
                patch.object(fresh_u0, "MacroLoopConfig", _CaptureConfig),
                patch.object(fresh_u0, "MacroClosedLoopRunner", _Runner),
                patch.dict(os.environ, environment, clear=True),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                exit_code = fresh_u0.main(_arguments(paths))

            self.assertEqual(exit_code, 0)
            bind.assert_called_once_with(
                report_path=paths["report"].resolve(),
                decoder_checkpoint=paths["checkpoint"].resolve(),
            )
            loop_values = _CaptureConfig.captured
            self.assertEqual(loop_values["round_count"], 1)
            self.assertEqual(loop_values["agent_case_credits"], 5)
            self.assertFalse(loop_values["allow_development_truth"])
            self.assertEqual(
                loop_values["execution_assurance"],
                ExecutionAssurance.REFERENCE_IN_PROCESS,
            )
            self.assertTrue(
                loop_values["evidence_stages"][0].require_cvar_no_regression
            )
            self.assertFalse(loop_values["terminal_require_cvar_no_regression"])
            self.assertEqual(loop_values["candidate_compute_budget"], 9.5)

            runner_values = captured["runner"]
            self.assertIsNotNone(
                runner_values["cold_start"].program_factory
            )
            self.assertEqual(
                captured["formal_runtime_calls"], (1, 2, 1)
            )
            self.assertIs(runner_values["plugin"], plugin)
            self.assertEqual(tuple(runner_values["agents"]), ("fresh-observable-agent",))
            agent_program = runner_values["agents"]["fresh-observable-agent"]
            launcher = agent_program.launcher
            self.assertEqual(
                launcher.command,
                (
                    str(paths["launcher"].resolve()),
                    "--model",
                    "gpt-test",
                    "--reasoning-effort",
                    "max",
                    "--codex-binary",
                    str(paths["codex"].resolve()),
                    "--auth-file",
                    str(paths["auth"].resolve()),
                ),
            )
            self.assertEqual(
                dict(launcher.environment),
                {
                    "PYTHONPATH": "/public/product/src",
                    "VIRTUAL_ENV": "/public/venv",
                    "SLURM_JOB_ID": "12345",
                    "USER": "researcher",
                },
            )
            prompt = loop_values["agents"][0].prompt
            self.assertIn("fresh, independent Agent", prompt)
            self.assertIn("at most 5 case credits", prompt)
            self.assertIn("predict_macros(case, query, budget=None)", prompt)
            self.assertIn("Do not inspect the", prompt)

            model_path = paths["run_root"] / "result" / "model" / "submission.py"
            result_path = paths["run_root"] / "result" / "result.json"
            self.assertEqual(model_path.read_bytes(), selected_source)
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], fresh_u0.RESULT_SCHEMA)
            self.assertEqual(payload["protocol"], fresh_u0.PROTOCOL_ID)
            self.assertEqual(payload["execution_assurance"], "reference_in_process")
            self.assertFalse(payload["formal_claims_allowed"])
            self.assertTrue(payload["fresh_agent_start"])
            self.assertFalse(payload["fresh_generator_draws"])
            self.assertFalse(payload["generator"]["paired_perturbations"])
            self.assertEqual(payload["primary_error"], 0.45)
            self.assertAlmostEqual(payload["overall_improvement"], 0.5)
            self.assertEqual(payload["evaluated_cases"], 8)
            self.assertEqual(payload["artifact"]["sha256"], selected_sha)
            self.assertEqual(
                payload["generator"]["release_bank_report_sha256"],
                hashlib.sha256(paths["report"].read_bytes()).hexdigest(),
            )
            self.assertEqual(
                payload["generator"]["decoder_checkpoint_sha256"],
                hashlib.sha256(paths["checkpoint"].read_bytes()).hexdigest(),
            )
            self.assertEqual(payload["terminal"]["selected"]["errors"], [0.4, 0.5])
            self.assertTrue(payload["terminal"]["paired_evidence"]["passed"])
            self.assertEqual(
                payload["terminal"]["paired_evidence"]["confidence_group_size"],
                1,
            )
            self.assertEqual(
                payload["terminal"]["paired_evidence"]["confidence_sample_count"],
                2,
            )
            self.assertEqual(
                payload["rounds"][0]["handoff"]["confidence_group_size"],
                1,
            )
            self.assertTrue(payload["rounds"][0]["promotion_evidence"]["promoted"])
            expected_metrics = {
                "rho_relative_l2",
                "momentum_relative_l2",
                "kinetic_energy_relative_l2",
            }
            self.assertEqual(
                set(payload["task"]["published_metric_names"]),
                expected_metrics,
            )
            selection_metrics = payload["rounds"][0]["selection"][0][
                "metric_aggregates"
            ]
            self.assertEqual(selection_metrics["case_count"], 2)
            self.assertEqual(set(selection_metrics["metrics"]), expected_metrics)
            self.assertAlmostEqual(
                selection_metrics["metrics"]["rho_relative_l2"]["mean"],
                0.2,
            )
            promotion_metrics = payload["rounds"][0]["promotion_evidence"][
                "comparisons"
            ][0]["paired_metric_aggregates"]
            self.assertEqual(promotion_metrics["case_count"], 2)
            self.assertAlmostEqual(
                promotion_metrics["candidate"]["metrics"][
                    "momentum_relative_l2"
                ]["mean"],
                0.4,
            )
            self.assertAlmostEqual(
                promotion_metrics["incumbent"]["metrics"][
                    "momentum_relative_l2"
                ]["mean"],
                1.2,
            )
            terminal_metrics = payload["terminal"]
            self.assertAlmostEqual(
                terminal_metrics["selected"]["metric_aggregates"]["metrics"][
                    "kinetic_energy_relative_l2"
                ]["mean"],
                0.6,
            )
            self.assertAlmostEqual(
                terminal_metrics["cold_start"]["metric_aggregates"]["metrics"][
                    "kinetic_energy_relative_l2"
                ]["mean"],
                1.8,
            )
            self.assertIsNotNone(
                terminal_metrics["paired_evidence"]["paired_metric_aggregates"]
            )

            serialized = result_path.read_text(encoding="utf-8")
            self.assertNotIn("target_sha256", serialized)
            self.assertNotIn("private_secret_metric", serialized)
            self.assertNotIn(str(PRIVATE_METRIC_VALUE), serialized)
            self.assertNotIn("hidden_truth", serialized)
            self.assertNotIn(str(paths["selected"].resolve()), serialized)
            self.assertNotIn(str(paths["cold"].resolve()), serialized)
            self.assertNotIn("must-not-leak", serialized)

    def test_digest_mismatch_fails_before_result_or_model_is_published(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = _inputs(root)
            paths["selected"].write_text(
                "def predict_macros(case, query, budget=None):\n    return None\n",
                encoding="utf-8",
            )
            captured = {}

            class _Runner:
                def __init__(self, **values):
                    captured.update(values)

                def run(self):
                    return _loop_result(
                        captured["cold_start"].ref,
                        paths["selected"],
                        "0" * 64,
                    )

            with (
                patch.object(
                    fresh_u0,
                    "bind_vpfp_release_bank_reference",
                    return_value=SimpleNamespace(
                        generator=SimpleNamespace(
                            state=SimpleNamespace(
                                base=SimpleNamespace(
                                    report_sha256=hashlib.sha256(
                                        paths["report"].read_bytes()
                                    ).hexdigest(),
                                    decoder_checkpoint_sha256=hashlib.sha256(
                                        paths["checkpoint"].read_bytes()
                                    ).hexdigest(),
                                )
                            )
                        )
                    ),
                ),
                patch.object(fresh_u0, "MacroLoopConfig", _CaptureConfig),
                patch.object(fresh_u0, "MacroClosedLoopRunner", _Runner),
            ):
                with self.assertRaisesRegex(
                    fresh_u0.FreshU0RunError,
                    "differ from its attestation",
                ):
                    fresh_u0.main(_arguments(paths))

            self.assertFalse((paths["run_root"] / "result" / "result.json").exists())
            self.assertFalse(
                (paths["run_root"] / "result" / "model" / "submission.py").exists()
            )

    def test_cli_rejects_out_of_range_protocol_values(self):
        parser = fresh_u0._parser()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--promotion-min-win-rate", "1.01"])


def _inputs(root):
    paths = {
        "report": root / "release.json",
        "checkpoint": root / "decoder.pt",
        "run_root": root / "run",
        "cold": root / "cold.py",
        "launcher": root / "launch-agent",
        "codex": root / "codex",
        "auth": root / "auth.json",
        "selected": root / "selected.py",
    }
    paths["report"].write_text('{"schema":"test-release"}\n', encoding="utf-8")
    paths["checkpoint"].write_bytes(b"decoder-checkpoint")
    paths["cold"].write_text(
        "def predict_macros(case, query, budget=None):\n    return None\n",
        encoding="utf-8",
    )
    paths["auth"].write_text("{}\n", encoding="utf-8")
    for name in ("launcher", "codex"):
        paths[name].write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        paths[name].chmod(0o700)
    return paths


def _arguments(paths):
    return [
        "--legacy-macro-reference",
        "--release-bank-report",
        str(paths["report"]),
        "--decoder-checkpoint",
        str(paths["checkpoint"]),
        "--run-root",
        str(paths["run_root"]),
        "--cold-start-candidate",
        str(paths["cold"]),
        "--agent-launcher",
        str(paths["launcher"]),
        "--codex-binary",
        str(paths["codex"]),
        "--auth-file",
        str(paths["auth"]),
        "--model",
        "gpt-test",
        "--reasoning-effort",
        "max",
        "--agent-timeout-seconds",
        "30",
        "--development-case-credits",
        "5",
        "--round-count",
        "1",
        "--development-case-count",
        "2",
        "--selection-case-count",
        "2",
        "--promotion-case-count",
        "2",
        "--terminal-case-count",
        "2",
        "--seed-base",
        "100",
        "--seed-stride",
        "1000",
        "--development-pool-multiplier",
        "1",
        "--promotion-confidence-z",
        "0",
        "--promotion-min-mean-improvement",
        "0",
        "--promotion-min-win-rate",
        "0",
        "--promotion-require-cvar-no-regression",
        "--terminal-confidence-z",
        "0",
        "--terminal-min-win-rate",
        "0",
        "--no-terminal-require-cvar-no-regression",
        "--candidate-compute-budget",
        "9.5",
    ]


def _loop_result(cold_ref, selected_path, selected_sha):
    selected_ref = CandidateRef(
        candidate_id="fresh-u0-selected",
        bundle_sha256=selected_sha,
        round_index=1,
        agent_id="fresh-observable-agent",
        parent_sha256=cold_ref.bundle_sha256,
        bundle_uri=str(selected_path.resolve()),
    )
    cold_score = _candidate_score(cold_ref, (0.8, 1.0))
    selected_score = _candidate_score(selected_ref, (0.4, 0.5))
    stage = EvidenceStage(
        2,
        confidence_z=0.0,
        min_mean_improvement=0.0,
        min_win_rate=0.0,
        require_cvar_no_regression=False,
    )
    comparison = paired_compare(selected_score, cold_score, stage)
    promotion = PromotionResult(
        incumbent_artifact_id=cold_ref.candidate_id,
        incumbent_digest=cold_ref.bundle_sha256,
        selected_artifact_id=selected_ref.candidate_id,
        selected_digest=selected_ref.bundle_sha256,
        promoted=True,
        reason="all_evidence_stages_passed",
        comparisons=(comparison,),
    )
    feedback = RoundFeedback(
        previous_incumbent_sha256=cold_ref.bundle_sha256,
        selected_sha256=selected_ref.bundle_sha256,
        promoted=True,
        decision_reason=promotion.reason,
        evidence_case_count=2,
        paired_mean_improvement=comparison.mean_improvement,
        paired_confidence_lower=comparison.confidence_lower,
        paired_win_rate=comparison.win_rate,
    )
    handoff = RoundHandoff(
        round_index=1,
        selected=selected_ref,
        promoted=True,
        challenger=selected_ref,
        decision_reason=promotion.reason,
        feedback=feedback,
        previous_handoff_sha256=None,
        handoff_sha256="e" * 64,
    )
    selected_batch = _batch(selected_ref, (0.4, 0.5))
    cold_batch = _batch(cold_ref, (0.8, 1.0))
    selected_published = _published_summary(0.2)
    cold_published = _published_summary(0.6)
    paired_published = PairedPublishedMetricSummary(
        case_count=2,
        candidate=selected_published,
        incumbent=cold_published,
    )
    finalist = FinalistEvaluation(
        candidate=selected_ref,
        batch=selected_batch,
        score=selected_score,
        aggregate=aggregate(selected_score.errors),
        published_metrics=selected_published,
    )
    round_result = MacroRoundResult(
        round_index=1,
        development_case_ids=("development-a", "development-b"),
        candidate_refs=(selected_ref,),
        agent_failures={},
        finalist_evaluations=(finalist,),
        challenger=selected_ref,
        promotion=promotion,
        selected=selected_ref,
        handoff=handoff,
        promotion_published_metrics=(paired_published,),
    )
    terminal = TerminalEvaluation(
        cohort_id="terminal-holdout",
        cohort_sha256=COHORT_SHA,
        cold_start=cold_batch,
        selected=selected_batch,
        cold_start_score=cold_score,
        selected_score=selected_score,
        comparison=comparison,
        cold_start_published_metrics=cold_published,
        selected_published_metrics=selected_published,
        paired_published_metrics=paired_published,
    )
    return MacroLoopResult(
        name="test-fresh-u0",
        plugin_id="vpfp",
        generator_id="verified-release-bank",
        generator_sha256="f" * 64,
        macro_contract_id=CONTRACT_ID,
        execution_assurance=ExecutionAssurance.REFERENCE_IN_PROCESS,
        formal_claims_allowed=False,
        cold_start=cold_ref,
        selected=selected_ref,
        rounds=(round_result,),
        terminal=terminal,
        published_metric_names=(
            "rho_relative_l2",
            "momentum_relative_l2",
            "kinetic_energy_relative_l2",
        ),
    )


def _published_summary(scale):
    return PublishedMetricSummary(
        case_count=2,
        aggregates={
            name: PublishedMetricAggregate(
                mean=scale * multiplier,
                cvar=scale * multiplier + 0.1,
                max=scale * multiplier + 0.1,
            )
            for name, multiplier in (
                ("rho_relative_l2", 1.0),
                ("momentum_relative_l2", 2.0),
                ("kinetic_energy_relative_l2", 3.0),
            )
        },
    )


def _candidate_score(candidate, errors):
    return MacroCandidateScore(
        artifact_id=candidate.candidate_id,
        artifact_digest=candidate.bundle_sha256,
        target_sha256=PRIVATE_TARGET_SHA,
        contract_id=CONTRACT_ID,
        case_ids=("private-terminal-a", "private-terminal-b"),
        errors=errors,
    )


def _batch(candidate, errors):
    return MacroBatchEvaluation(
        candidate_id=candidate.candidate_id,
        cohort_id="terminal-holdout",
        cohort_sha256=COHORT_SHA,
        target_sha256=PRIVATE_TARGET_SHA,
        contract_id=CONTRACT_ID,
        scores=tuple(
            SanitizedMacroCaseScore(
                case_id=case_id,
                contract_id=CONTRACT_ID,
                status="valid",
                primary_error=error,
                metrics={"private_secret_metric": PRIVATE_METRIC_VALUE},
            )
            for case_id, error in zip(
                ("private-terminal-a", "private-terminal-b"),
                errors,
            )
        ),
    )


if __name__ == "__main__":
    unittest.main()
