#!/usr/bin/env python3
"""Compare frozen campaign Solvers on fresh, paired BASE/G2/random problems.

This is a controller-private evaluation companion, not another Campaign loop.
Completed native draws are replayed into the production continuous resident
runtime. Its public demand-only Oracle and isolated Candidate service are used
unchanged. No Solver/Guide training, candidate selection, or asset replacement.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

SCHEMA = "vpfp_campaign_guidance_paircheck_v1"
ARMS = ("base", "guided_relative", "random_relative")
STRATA = {"collision_nu_edges": [10, 15, 20, 25, 30],
          "temperature_edges": [.8, 1., 1.2]}


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def seed_for(master, label):
    return int.from_bytes(sha256(f"{SCHEMA}:{master}:{label}".encode()).digest()[:8], "big") % 2**63


def validate_config(config):
    fixed = {"schema": SCHEMA, "guide_round": 2, "root_count": 32,
             "point_count": 256, "condition_strata": STRATA}
    for name, value in fixed.items():
        if config.get(name) != value:
            raise ValueError("fixed paired experiment differs: " + name)
    for name in ("root_master_seed", "bootstrap_seed", "random_seed"):
        if type(config.get(name)) is not int or not 0 <= config[name] < 2**63:
            raise ValueError("nonnegative signed-64-bit seed required: " + name)
    for name in ("bootstrap_replicates", "candidate_workers"):
        if type(config.get(name)) is not int or config[name] <= 0:
            raise ValueError("positive integer required: " + name)
    if not math.isfinite(config["wall_seconds"]) or config["wall_seconds"] <= 0:
        raise ValueError("finite positive wall budget required")
    rows = config["candidates"]
    if (len(rows) != 2 or len({row["id"] for row in rows}) != 2 or
            {row["role"] for row in rows} != {"guidance_target", "final_delivery"}):
        raise ValueError("one guidance-target and one final-delivery Solver required")
    for value in [config["guidance_sha256"], *(row["sha256"] for row in rows)]:
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("explicit frozen SHA256 required")
    return config


def registered_roots(config, previous_roots):
    """Prespecified 4x2 physical condition strata, four roots each; no redraw."""
    import numpy as np
    roots = []
    edges = config["condition_strata"]
    for index in range(config["root_count"]):
        stratum = index % 8
        root = seed_for(config["root_master_seed"], f"root:{index}")
        if root in previous_roots or root in {row["root_seed"] for row in roots}:
            raise ValueError("new registered root overlaps history; no silent replacement")
        rng = np.random.default_rng(seed_for(config["root_master_seed"], f"condition:{index}"))
        condition = []
        for name, cell in (("collision_nu_edges", stratum // 2), ("temperature_edges", stratum % 2)):
            lo, hi = edges[name][cell:cell+2]
            value = float(np.float32(lo + (.01 + .98 * rng.random()) * (hi-lo)))
            if not lo < value < hi:
                raise ValueError("FP32 condition escaped its open stratum")
            condition.append(value)
        roots.append({"index": index, "stratum": stratum, "root_seed": root,
                      "conditions": condition, "query_seed": seed_for(root, "private-points")})
    return roots


def historical_roots(campaign):
    audit = json.loads((campaign / "guidance_audits.json").read_text())
    roots = {root for row in audit["sampling"] for root in row["root_seeds"]}
    paths = sorted((campaign / "private_backend" / "guidance").glob("round-*/feedback-manifest.json"))
    if len(paths) != 2:
        raise ValueError("both completed-round feedback manifests required for root isolation")
    for path in paths:
        roots.update(row["root_seed"] for row in json.loads(path.read_text())["rows"])
    return roots


def arm_order(index):
    if type(index) is not int or index < 0:
        raise ValueError("nonnegative root index required")
    offset = (index % 8 + index // 8) % len(ARMS)
    return ARMS[offset:] + ARMS[:offset]


def summarize(rows, config):
    from vpfp_guidance_effect_summary import summarize_pairs
    results = {}
    for candidate in config["candidates"]:
        key = candidate["id"]
        results[key] = {}
        for left, right in (("base", "guided_relative"), ("random_relative", "guided_relative"),
                            ("base", "random_relative")):
            paired = [{"root_seed": row["root_seed"], "stratum": row["stratum"],
                       "base": row["scores"][key][left], "guided": row["scores"][key][right]}
                      for row in rows]
            report = summarize_pairs(paired, bootstrap_seed=config["bootstrap_seed"],
                                     bootstrap_replicates=config["bootstrap_replicates"])
            report.update(arm_mapping={"base": left, "guided": right},
                          primary=candidate["role"] == "guidance_target" and right == "guided_relative",
                          solver_role=candidate["role"],
                          secondary_transfer=candidate["role"] == "final_delivery",
                          intervals_are_pointwise_not_simultaneous=True,
                          automatic_candidate_admission=False)
            results[key][right + "_vs_" + left] = report
    return results


def pending_rows(roots, config):
    return [{**row, "generation": {}, "scores": {candidate["id"]: {
        arm: {"status": "pending", "valid": False, "error": None, "attempt_count": 0}
        for arm in ARMS} for candidate in config["candidates"]}} for row in roots]


def query_sha(query):
    from run_canonical_vpfp_campaign import jsonable
    return digest(jsonable(query))


def verify_same_queries(queries):
    fingerprints = {query_sha(query) for query in queries}
    if len(fingerprints) != 1:
        raise ValueError("paired arms changed private evaluation points")
    for query in queries:
        if len(query.points) != 256:
            raise ValueError("paired evaluation must retain all 256 private points")
    return next(iter(fingerprints))


def run(config, output, *, wall_seconds):
    from run_canonical_vpfp_campaign import FIELDS, Writer, file_sha, jsonable, load_config
    if not os.environ.get("SLURM_JOB_ID", "").isdigit() or not os.environ.get("SLURM_TMPDIR"):
        raise RuntimeError("real paired sampling and Solver evaluation require a Slurm compute allocation")
    if not math.isfinite(wall_seconds) or wall_seconds <= 0:
        raise ValueError("finite positive wall allocation required")
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    writer, started = Writer(output), time.monotonic()
    report = {"schema": SCHEMA, "status": "running", "guidance_training_performed": False,
              "solver_training_performed": False, "online_assets_replaced": False,
              "regression_admission_gate": False, "scientific_admission": False,
              "paired_draw_adapter": "evaluation_only_completed_native_draw_replay",
              "canonical_guided_seed_rehash_used": False,
              "public_source_labels": "all replay cases are GENERATED; true sampling arm is controller-private",
              "expected_roots": 32, "expected_generation_attempts": 96,
              "expected_candidate_case_slots": 192, "failures": []}
    writer.write("config.json", config)
    writer.write("status.json", report)
    binding = candidates = replay = None
    rows, initial_runtime_generator = [], None
    guards = {}

    def check():
        if time.monotonic() - started >= wall_seconds:
            raise TimeoutError("paired test wall allocation exhausted; pending cases are retained, never dropped")

    def persist():
        report["wall_seconds"] = time.monotonic() - started
        report["completed_root_slots"] = sum(all(score["status"] in ("completed", "failed")
            for scores in row["scores"].values() for score in scores.values()) for row in rows)
        writer.write("paired-results.json", rows)
        writer.write("summary.json", {**report, "comparisons": summarize(rows, config)})
        writer.write("status.json", report)

    try:
        revision = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        if subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all"], text=True).strip():
            raise ValueError("paired experiment requires clean committed source")
        report["source_revision"] = revision
        import numpy as np
        import torch
        from pde_agent_verifier_solver.engine import PythonCandidateCodec, PythonTaskWorkerSpec, SubprocessCandidateExecutionService
        from pde_agent_verifier_solver.engine.evaluator import HiddenVerifier
        from pde_agent_verifier_solver.sdk import AnswerQuery, CaseVisibility, EvaluationPhase, TaskKind, inspect_candidate_bundle
        from pde_agent_verifier_solver.plugins.vpfp.iteration_backend import build_vpfp_strong_iteration_binding
        from pde_agent_verifier_solver.plugins.vpfp.solver_error_guidance import draw_frozen_ic40, _module_sha, _json_sha
        from pde_agent_verifier_solver.plugins.vpfp.standardized_solver_error_guidance import regressor_from_payload
        from pde_agent_verifier_solver.plugins.vpfp.rescaled_solver_error_guidance import RescaledIC40Sampler, RescalingPolicy
        from pde_agent_verifier_solver.plugins.vpfp.round_solver_guidance import _tensor_sha
        from run_plasma_canonical_smoke import _forbid_online_vpfp_generation
        from vpfp_strong_guidance_rollouts import CompletedDrawReplay

        campaign = Path(config["campaign_root"]).resolve()
        previous = historical_roots(campaign)
        roots = registered_roots(config, previous)
        rows = pending_rows(roots, config)
        writer.write("root-manifest.json", {"roots": roots, "historical_root_count": len(previous),
            "historical_roots_sha256": digest(sorted(previous)), "overlap_count": 0})
        campaign_config = load_config(Path(config["campaign_config"]).resolve())
        actual_campaign_config = json.loads((campaign / "resolved_config.json").read_text())
        if any(campaign_config[key] != actual_campaign_config[key] for key in ("backend", "task")):
            raise ValueError("paired Base or Task settings differ from the actual completed campaign")
        guide_path = Path(config["guidance_checkpoint"])
        bundle_infos = {item["id"]: inspect_candidate_bundle(Path(item["path"])) for item in config["candidates"]}
        payload = torch.load(guide_path, map_location="cpu", weights_only=True)
        training = payload["training"]
        target = next(row for row in config["candidates"] if row["role"] == "guidance_target")
        if (training["target_solver_sha256"] != target["sha256"] or
                (training["source_round_index"], training["target_round_index"], training["step"]) != (1, 2, 2000) or
                training["label_source"] != "actual_public_solver_macro_error"):
            raise ValueError("G2 does not target the declared frozen first-round Solver")
        if set(training["real_root_seeds"]) & {row["root_seed"] for row in rows}:
            raise ValueError("paired roots overlap the actual G2 feedback roots")
        binding = build_vpfp_strong_iteration_binding(**campaign_config["backend"], output_root=output / "private_backend")
        generator, backend = binding.plugin.generator, binding.plugin.generator.backend
        base = backend.capture.base
        base_before = (_module_sha(base.model), _json_sha(base.normalizer.to_payload()))
        guide = regressor_from_payload(payload, device=next(base.model.parameters()).device)
        samplers = {arm: RescaledIC40Sampler(base, guide, RescalingPolicy(mode="relative",
            target_ratio=.1, max_flow_ratio=.25, random_direction=arm == "random_relative",
            random_seed=config["random_seed"]), max_gradient_norm=1., sigma_min=.002, sigma_max=2.)
            for arm in ARMS if arm != "base"}
        state = generator.load(binding.checkpoint, {})
        initial_runtime_generator = backend.runtime.generator
        replay = CompletedDrawReplay(base.identity)
        backend.runtime.generator = replay
        codec = PythonCandidateCodec(task_kind=TaskKind.OBSERVABLE_MODEL)
        artifacts = {item["id"]: codec.bind_existing(bundle_infos[item["id"]].root,
            storage_root=output / "controller_candidates", artifact_id=item["id"], name=item["id"],
            round_index=1 if item["role"] == "guidance_target" else 2, agent_id="vpfp-agent",
            summary="Frozen Solver from completed canonical VPFP campaign", usage="prepare then evaluate POINTS")
            for item in config["candidates"]}
        task = binding.plugin.task("observable_model")
        query = AnswerQuery.at_points(fields=FIELDS, points=({"t": .5, "x": 0., "y": 0.},))
        task_settings = {**campaign_config["task"], "require_prepared_solution": True,
                         "official_point_count": config["point_count"], "terminal_point_count": config["point_count"]}
        candidates = SubprocessCandidateExecutionService(output / "candidate_execution",
            task=PythonTaskWorkerSpec.from_task_type(type(task)),
            evaluation_settings={**task_settings, "default_query": query}, max_workers=config["candidate_workers"],
            case_hard_wall_seconds=campaign_config["execution"]["candidate_case_hard_wall_seconds"],
            development_batch_hard_wall_seconds=campaign_config["execution"]["development_candidate_batch_hard_wall_seconds"],
            python_executable=sys.executable)
        report.update(guidance_target_solver=target["id"], guidance_sha256=config["guidance_sha256"],
            base_identity=base.identity, input_candidate_sha256={item["id"]: item["sha256"] for item in config["candidates"]},
            source_revision=subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
            historical_backend_and_task_exact=True,
            sampling_policies={arm: sampler.policy.to_dict() for arm, sampler in samplers.items()},
            random_control="same relative drift-RMS policy at each arm's own evolving state; not equal absolute trajectory displacement",
            prepared_query_blind=True, private_scoring_points=256, source_prematerialized=False,
            query_contract="same root shares bitwise-identical 256 POINTS across all arms and both Solvers")
        persist()
        with _forbid_online_vpfp_generation(guards):
            for row in rows:
                check()
                records, draws = {}, {}
                case_ids = []
                verifier = HiddenVerifier(task, query, settings=task_settings, phase=EvaluationPhase.TERMINAL,
                    plan_seed_provider=lambda index, case_id, fixed=row["query_seed"]: fixed,
                    generator=generator, generator_state=state, candidate_execution=candidates)
                try:
                    for arm in arm_order(row["index"]):
                        check()
                        began = time.monotonic()
                        generation = row["generation"][arm] = {"status": "running", "attempt_count": 1}
                        persist()
                        try:
                            conditions = np.asarray([row["conditions"]], dtype=np.float32)
                            draw = (draw_frozen_ic40(base, [row["root_seed"]], conditions) if arm == "base"
                                    else samplers[arm].draw([row["root_seed"]], conditions))
                            for name in ("full", "initial_context", "future_white", "equation_conditions"):
                                if not torch.isfinite(draw[name]).all():
                                    raise FloatingPointError("nonfinite completed native " + name)
                            if draw["root_seeds"] != [row["root_seed"]]:
                                raise ValueError("sampler changed paired root")
                            if draws:
                                other = next(iter(draws.values()))
                                for name in ("initial_context", "equation_conditions"):
                                    if not torch.equal(draw[name], other[name]):
                                        raise ValueError("paired arms changed clean IC or physical condition")
                                for name in ("known_prior_seeds", "future_noise_seeds"):
                                    if draw[name] != other[name]:
                                        raise ValueError("paired arms changed native RNG stream")
                            diagnostics = None if arm == "base" else deepcopy(samplers[arm].last_report)
                            draw_path = output / f"draw-{row['index']:03d}-{arm}.controller-private.pt"
                            with draw_path.open("xb") as handle:
                                torch.save({"schema": SCHEMA + "_actual_draw", "arm": arm,
                                    "root": deepcopy({key: row[key] for key in ("index", "stratum", "root_seed", "conditions", "query_seed")}),
                                    "guidance_sha256": None if arm == "base" else config["guidance_sha256"],
                                    "draw": {key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else deepcopy(value)
                                             for key, value in draw.items()}}, handle)
                            draw_path.chmod(0o600)
                            generation["actual_draw"] = {"path": str(draw_path), "sha256": file_sha(draw_path)}
                            replay.set_draw(draw, guidance_report=diagnostics)
                            case_id = f"vpfp-paircheck-{row['root_seed']}-{arm}"
                            case_ids.append(case_id)
                            emitted = backend.runtime.generate_base(seeds=(row["root_seed"],), case_ids=(case_id,),
                                visibility=CaseVisibility.HIDDEN, round_index=2, cohort="paired-guidance-evaluation")
                            if len(emitted) != 1 or replay._pending is not None:
                                raise RuntimeError("native completed-draw replay did not emit exactly one Case")
                            if emitted[0].transaction is not None:
                                emitted[0].transaction.commit()
                            records[arm], draws[arm] = emitted[0].record, draw
                            generation.update(status="completed", case_id=case_id,
                                wall_seconds=time.monotonic()-began,
                                tensor_sha256={name: _tensor_sha(draw[name]) for name in
                                    ("full", "future_white", "initial_context", "equation_conditions")},
                                known_prior_seeds=draw["known_prior_seeds"], future_noise_seeds=draw["future_noise_seeds"])
                            if diagnostics is not None:
                                writer.write(f"sampling-{row['index']:03d}-{arm}.json", diagnostics)
                        except Exception as error:
                            replay.clear()
                            generation.update(status="failed", error_type=type(error).__name__, error=str(error),
                                              wall_seconds=time.monotonic()-began)
                            for scores in row["scores"].values():
                                scores[arm].update(status="failed", attempt_count=1,
                                    candidate_execution_attempted=False, failure_stage="generation",
                                    error_type=type(error).__name__, message=str(error))
                        persist()
                    plans = [verifier._plan_for(record, 0).query for record in records.values()]
                    if plans:
                        row["private_query_sha256"] = verify_same_queries(plans)
                        row["same_points_all_arms_and_solvers"] = True
                        writer.write(f"query-{row['index']:03d}.controller-private.json", jsonable(plans[0]))
                    if "base" in draws:
                        for arm, draw in draws.items():
                            delta = draw["future_white"].double()-draws["base"]["future_white"].double()
                            row["generation"][arm]["future_delta_l2_vs_base"] = float(delta.norm())
                            row["generation"][arm]["future_bitwise_equal_base"] = torch.equal(draw["future_white"], draws["base"]["future_white"])
                    for candidate in config["candidates"]:
                        if not records:
                            continue
                        check()
                        key = candidate["id"]
                        for arm in records:
                            row["scores"][key][arm].update(status="running", attempt_count=1,
                                                         candidate_execution_attempted=True)
                        persist()
                        try:
                            evaluated = verifier.evaluate(artifacts[key], tuple(records.values()))
                            by_id = {item.case_id: item for item in evaluated.cases}
                            if set(by_id) != {record.case.case_id for record in records.values()}:
                                raise RuntimeError("isolated evaluation returned the wrong Case inventory")
                            for arm, record in records.items():
                                result = by_id[record.case.case_id]
                                error = float(result.primary_error)
                                valid = math.isfinite(error) and error >= 0
                                row["scores"][key][arm].update(status="completed", valid=valid,
                                    error=error if valid else None, metrics=jsonable(dict(result.metrics)),
                                    query_sha256=row["private_query_sha256"])
                        except Exception as error:
                            for arm in records:
                                row["scores"][key][arm].update(status="failed", valid=False, error=None,
                                    failure_stage="candidate_evaluation", error_type=type(error).__name__, message=str(error))
                        persist()
                finally:
                    backend.runtime.release_cases(case_ids)
                    if replay is not None:
                        replay.clear()
                print(json.dumps({"event": "paired_root_finished", "index": row["index"],
                    "completed_roots": report["completed_root_slots"], "root_seed": row["root_seed"]}), flush=True)
            for sampler in samplers.values():
                sampler.assert_frozen()
            for item in config["candidates"]:
                if inspect_candidate_bundle(Path(item["path"])).tree_sha256 != item["sha256"]:
                    raise ValueError("historical frozen Solver asset changed")
            report.update(status="completed", frozen_assets_final_verified=True)
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        report["failures"].append({"type": type(error).__name__, "message": str(error)})
        (output / "failure-traceback.txt").write_text(traceback.format_exc())
    finally:
        if binding is not None:
            backend = binding.plugin.generator.backend
            backend.runtime.release_cases(tuple(backend.runtime._resident_cases))
            if initial_runtime_generator is not None:
                backend.runtime.generator = initial_runtime_generator
            report["retention"] = {"resident_cases": len(backend.runtime._resident_cases),
                "draws": len(backend._draws), "anchors": len(backend.backend._anchors),
                "pending_draws": len(backend.capture.pending)}
            if backend.training_audits or any(report["retention"].values()):
                report.update(status="failed", cleanup_or_training_contract_failed=True)
        audits = [] if candidates is None else jsonable(candidates.audits)
        writer.write("candidate-execution-audits.json", audits)
        report["candidate_execution_status_counts"] = dict(Counter(item["status"] for item in audits))
        report["candidate_execution_attempts"] = len(audits)
        report["generation_attempts"] = sum(len(row["generation"]) for row in rows)
        report["online_coarse_solver_guard"] = guards
        report["source_final_verified"] = (
            report.get("source_revision") == subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
            and not subprocess.check_output(
                ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all"], text=True).strip())
        if not report["source_final_verified"]:
            report.update(status="failed", source_mutation_detected=True)
        if any(item["candidate_process_isolated"] is not True or item["controller_truth_shared"] is not False
               or item["process_group_cleaned"] is not True for item in audits):
            report.update(status="failed", candidate_isolation_or_cleanup_failed=True)
        report["execution_passed"] = report["status"] == "completed" and len(audits) == 192 and all(
            score["status"] == "completed" and score["valid"] for row in rows
            for scores in row["scores"].values() for score in scores.values())
        persist()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--wall-seconds", type=float)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    config = validate_config(json.loads(args.config.read_text()))
    if args.validate_only:
        roots = registered_roots(config, historical_roots(Path(config["campaign_root"])))
        print(json.dumps({"status": "static_configuration_validated", "registered_roots": len(roots),
                          "generation_attempts": 96, "candidate_case_slots": 192, "models_loaded": False}))
        return 0
    if args.output is None:
        parser.error("execution requires --output")
    report = run(config, args.output.resolve(), wall_seconds=(config["wall_seconds"]
        if args.wall_seconds is None else args.wall_seconds))
    print(json.dumps({key: report.get(key) for key in ("status", "execution_passed", "wall_seconds")}), flush=True)
    return 0 if report["execution_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
