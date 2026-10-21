#!/usr/bin/env python3
"""Real CUDA/Public-SDK acceptance for one Plasma transport-v2 family.

Run one family per process, under Slurm. This is an engineering acceptance,
not a distribution audit, benchmark, Guide fit, Agent call or solver-quality
claim. One fresh BASE and its perturbation are scored through the same hidden
verifier and bounded candidate subprocess API used by an actual Campaign.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
SCHEMA = "plasma_macro_transport_v2_native_acceptance_v1"
DEFAULT_SEED = 202610040917


def build_transport_config(config, calibration):
    """Adapt a copy of an old deployment; never edit frozen config files."""
    from scripts.plasma_gpu_campaign import SCHEMA as OLD_SCHEMA, TRANSPORT_SCHEMA
    from pde_agent_verifier_solver.plugins.plasma_transport import (
        TRANSPORT_PROTOCOL_ID, load_transport_calibration,
    )
    if config.get("schema") not in (OLD_SCHEMA, TRANSPORT_SCHEMA):
        raise ValueError("acceptance expects a supported Plasma deployment config")
    if config.get("equation") not in ("strong", "weak", "artemis"):
        raise ValueError("acceptance supports strong, weak, or artemis")
    asset = load_transport_calibration(calibration)
    adapted = deepcopy(config)
    adapted["schema"] = TRANSPORT_SCHEMA
    adapted.setdefault("task", {}).update(
        metric_profile=TRANSPORT_PROTOCOL_ID, metric_calibration=asset.to_dict(),
    )
    adapted.setdefault("generator", {}).setdefault("settings", {}).update(
        metric_profile=TRANSPORT_PROTOCOL_ID, metric_calibration=asset.to_dict(),
    )
    protocol = adapted.setdefault("protocol", {})
    protocol.update(metric_profile=TRANSPORT_PROTOCOL_ID, scoring_objective_hash=asset.objective_hash)
    if protocol.get("parallel_evaluation") != "service_global_oracle_v1":
        execution, equation = adapted["execution"], adapted["equation"]
        per_device = (adapted.get("migration", {}).get("oracle_execution_options", {}).get("workers_per_gpu", 4)
                      if equation == "strong" else execution["workers_per_gpu"] if equation == "weak" else 1)
        limit = min(execution["candidate_case_workers"], 3 * per_device)
        if equation == "weak":
            limit = max(limit, execution["development_workers_per_gpu"])
        execution.update(solver_batch_size=1, gpu_slots_per_device=1,
                         oracle_concurrency_per_device=1, oracle_max_workers=limit)
        protocol["parallel_evaluation"] = "service_global_oracle_v1"
    from scripts.plasma_gpu_campaign import validate_config
    return validate_config(adapted)


def _backend(binding):
    generator = binding.plugin.generator
    backend = getattr(generator, "backend", None)
    return generator._backend() if backend is None else backend


def check_objective_bindings(binding, calibration):
    """Validate the real matcher and Guide recipe without training a Guide."""
    matcher = binding.plugin.macro_matcher
    task = binding.plugin.task("observable_model")
    backend = _backend(binding)
    guide = getattr(backend, "guidance_adapter", None) or backend
    objects = {"plugin_matcher": matcher, "task_matcher": task.macro_matcher,
               "backend": backend, "guidance_recipe": guide}
    hashes = {}
    for name, owner in objects.items():
        asset = getattr(owner, "calibration", None) or getattr(owner, "score_calibration", None)
        if asset is None or asset.objective_hash != calibration.objective_hash:
            raise ValueError(f"{name} did not bind the same frozen transport objective")
        hashes[name] = asset.objective_hash
    return hashes


def verify_case_score(result, reference, prediction, calibration, family):
    """Independently recompute the declared formula from returned physical arrays."""
    import numpy as np
    from pde_agent_verifier_solver.plugins.plasma_transport import (
        TRANSPORT_PROTOCOL_ID, TRANSPORT_FIELDS, TRANSPORT_WEIGHTS,
        canonicalize_1d, canonicalize_2d,
    )
    canonicalize = canonicalize_1d if family == "artemis" else canonicalize_2d
    expected, actual = canonicalize(reference), canonicalize(prediction)
    floors = calibration.physical_floors
    recomputed = {}
    for name in TRANSPORT_FIELDS:
        difference = actual[name] - expected[name]
        numerator = np.sum(difference * difference) / difference.shape[0]
        denominator = np.sum(expected[name] * expected[name]) / expected[name].shape[0] + floors[name] ** 2
        recomputed[name] = math.sqrt(float(numerator / denominator))
        if not math.isclose(recomputed[name], float(result.metrics[f"{name}_normalized_rms"]), rel_tol=1e-11, abs_tol=1e-13):
            raise ValueError(f"returned {name} error differs from the physical all-point RMS formula")
    total = sum(TRANSPORT_WEIGHTS[name] * recomputed[name] for name in TRANSPORT_FIELDS)
    if not math.isfinite(total) or not math.isclose(total, result.primary_error, rel_tol=1e-11, abs_tol=1e-13):
        raise ValueError("primary score differs from the frozen four-block formula")
    if not math.isclose(total, result.metrics[TRANSPORT_PROTOCOL_ID], rel_tol=1e-11, abs_tol=1e-13):
        raise ValueError("transport Guidance label differs from the promotion score")
    if result.macro_match is None or result.macro_match.contract_id != calibration.contract_id:
        raise ValueError("case score lost its calibrated scientific objective identity")
    return {"primary_error": total, "physical_errors": recomputed,
            "truth_density_min": float(expected["density"].min()),
            "prediction_density_min": float(actual["density"].min()),
            "point_count": expected["density"].shape[0],
            "comparison_rtol": 1e-11, "comparison_atol": 1e-13,
            "bitwise_equality_required": False}


def run_acceptance(config_path, calibration_path, output_root, *, seed=DEFAULT_SEED, point_count=32):
    if not os.environ.get("SLURM_JOB_ID", "").isdigit():
        raise RuntimeError("native transport acceptance requires a Slurm compute allocation")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("native transport acceptance requires CUDA; no CPU fallback")
    if isinstance(point_count, bool) or not isinstance(point_count, int) or point_count < 2:
        raise ValueError("acceptance point_count must be an integer >=2")
    from scripts.plasma_gpu_campaign import _binding
    from scripts.check_plasma_gpu_integration import check_binding
    from pde_agent_verifier_solver.engine import PythonCandidateCodec, PythonTaskWorkerSpec, SubprocessCandidateExecutionService
    from pde_agent_verifier_solver.engine.checkpoint_store import atomic_json
    from pde_agent_verifier_solver.engine.evaluator import HiddenVerifier
    from pde_agent_verifier_solver.plugins.plasma_transport import load_transport_calibration
    from pde_agent_verifier_solver.plugins.plasma_transport_settings import check_feedback_objective
    from pde_agent_verifier_solver.sdk import (
        AnswerQuery, CaseFeedback, CaseVisibility, GenerationKind, GenerationPurpose,
        GenerationRequest, GeneratorMode, RoundFeedback, TaskKind,
    )
    config_path, calibration_path, output_root = map(lambda p: Path(p).resolve(), (config_path, calibration_path, output_root))
    calibration = load_transport_calibration(calibration_path)
    original = json.loads(config_path.read_text())
    config = build_transport_config(original, calibration)
    family = config["equation"]
    requested_device = (config["backend"]["factory_settings"].get("device", "cpu")
                        if family == "artemis" else config["backend"].get("device", "cuda"))
    if not str(requested_device).startswith("cuda"):
        raise ValueError("native acceptance config must load its physical model on CUDA")
    output_root.mkdir(mode=0o700, parents=True, exist_ok=False)
    started = perf_counter()
    report = {
        "schema": SCHEMA, "status": "running", "family": family, "seed": seed,
        "config_path": str(config_path), "config_sha256": sha256(config_path.read_bytes()).hexdigest(),
        "calibration_path": str(calibration_path), "calibration_file_sha256": sha256(calibration_path.read_bytes()).hexdigest(),
        "objective_identity": calibration.objective_identity,
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "cuda_device": torch.cuda.get_device_name(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "engineering_point_count": point_count, "formal_case_count": 0,
        "agent_called": False, "guidance_trained": False, "new_pde_residual_audit": False,
        "solver_quality_claim": False, "candidate_accelerator": "cpu_numpy_public_baseline",
        "cases": [], "candidate_audits": [], "private_cases_released": False,
    }
    generator = state = service = None
    owned = []
    try:
        atomic_json(output_root / "report.json", report)
        binding, binding_receipt = _binding(config, output_root)
        report["objective_bindings"] = check_objective_bindings(binding, calibration)
        # One existing readiness check, including its existing source checks;
        # do not add a new residual sweep or repeated kinetic solve here.
        report["source_readiness"] = check_binding(family, binding,
            output_path=output_root / "source-readiness.json", seed=seed + 1)
        report["binding_receipt"] = binding_receipt
        generator = binding.plugin.generator
        # Exercise the same configured settings handoff as CampaignController,
        # including the frozen scoring objective, before qualifying sampling.
        from scripts.plasma_gpu_campaign import recipe
        from pde_agent_verifier_solver.engine.seeds import SeedLedger
        plan = recipe(config, None)
        generator_settings = {"basis_size": plan.generator.basis_size,
            **plan.generator.settings, "master_seed": plan.master_seed,
            "controller_seed": SeedLedger(plan.master_seed).derive("generator", "base")}
        state = generator.load(binding.checkpoint, generator_settings)
        report["generator_settings_qualified"] = sorted(generator_settings)
        records = []
        for index, kind in enumerate((GenerationKind.BASE, GenerationKind.PERTURBED)):
            request = GenerationRequest(round_index=1, count=1, mode=GeneratorMode.ONLINE,
                kind=kind, visibility=CaseVisibility.HIDDEN, seed=seed + 11 + index,
                purpose=GenerationPurpose.PROMOTION, settings=generator_settings,
                anchors=() if not records else (records[0].case,))
            generated = tuple(generator.sample(state, request))
            owned.extend(record.case.case_id for record in generated)
            if len(generated) != 1 or generated[0].case.visibility is not CaseVisibility.HIDDEN:
                raise ValueError("native acceptance requires one fresh hidden case per generation request")
            records.append(generated[0])
        if records[0].case.parent_case_id is not None or records[1].case.parent_case_id != records[0].case.case_id:
            raise ValueError("native acceptance BASE/perturbation lineage is inconsistent")
        task = binding.plugin.task("observable_model")
        fields = task.macro_matcher.fields
        if family == "artemis":
            default_point = {"t": float(records[0].case.public_data["arrays"]["t"][1]),
                             "x": float(records[0].case.public_data["arrays"]["x"][0])}
            bundle = ROOT / "configs/candidates/artemis_hold_initial_bundle"
        else:
            interval = records[0].case.public_data["question"]["time_domain"]
            default_point = {"t": float(sum(interval) / 2), "x": 0.0, "y": 0.0}
            bundle = ROOT / "configs/candidates/vpfp_transport_simple_bundle"
        query = AnswerQuery.at_points(fields=fields, points=(default_point,))
        settings = {**config["task"], "default_query": query, "official_point_count": point_count,
                    "require_prepared_solution": True, "candidate_budget_mode": "ns_remaining_case_v1"}
        settings.pop("candidate_compute_budget", None)
        codec = PythonCandidateCodec(task_kind=TaskKind.OBSERVABLE_MODEL)
        artifact = codec.bind_existing(bundle, storage_root=output_root / "frozen_s0",
            artifact_id=f"{family}-transport-v2-acceptance-s0", name="Simple transport-v2 S0",
            summary="Public initial-data baseline, not a qualified solver", usage="prepare_solution then evaluate")
        report["candidate_bundle_sha256"] = artifact.program.tree_sha256
        service = SubprocessCandidateExecutionService(output_root / "candidate_execution",
            task=PythonTaskWorkerSpec.from_task_type(type(task)), evaluation_settings=settings,
            max_workers=1, oracle_max_workers=1, case_hard_wall_seconds=900,
            oracle_drain_timeout_seconds=30, python_executable=sys.executable)
        verifier = HiddenVerifier(task, query, settings=settings, seed=seed + 31,
            generator=generator, generator_state=state, candidate_execution=service)
        plans = [verifier._plan_for(record, index) for index, record in enumerate(records)]
        truths = [verifier._truth_for(record, plan) for record, plan in zip(records, plans)]
        result = verifier.evaluate(artifact, records)
        if len(result.cases) != 2:
            raise ValueError("acceptance did not score both BASE and perturbation")
        feedback = []
        for record, truth, item in zip(records, truths, result.cases):
            if item.prediction is None:
                raise ValueError("acceptance scorer did not retain the finite candidate output")
            row = verify_case_score(item, truth, item.prediction, calibration, family)
            report["cases"].append({"case_id": record.case.case_id,
                "parent_case_id": record.case.parent_case_id, "distribution": record.case.source.value,
                "metrics": dict(item.metrics), **row})
            feedback.append(CaseFeedback(record.case, item.primary_error, item.metrics))
        check_feedback_objective(calibration, RoundFeedback(1, artifact, tuple(feedback)))
        report["guidance_feedback_matches_promotion"] = True
        report["candidate_audits"] = [asdict(item) for item in service.audits]
        if len(service.audits) != 2 or any(item.status != "completed" or not item.process_group_cleaned for item in service.audits):
            raise ValueError("native public-SDK candidate subprocesses did not complete cleanly")
        torch.cuda.synchronize()
        report["status"] = "passed"
        return report
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
        raise
    finally:
        try:
            try:
                if service is not None:
                    report["candidate_audits"] = [asdict(item) for item in service.audits]
                    service._executor.shutdown(wait=True)
                    if service._processes or service._unhealthy_reason is not None:
                        raise RuntimeError("acceptance candidate service did not finish cleanly")
            finally:
                if generator is not None and owned:
                    generator.release(state, tuple(owned))
                report["private_cases_released"] = True
        except BaseException as error:
            report.update(status="failed", cleanup_error=str(error))
            raise
        finally:
            report["wall_seconds"] = perf_counter() - started
            atomic_json(output_root / "report.json", report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--point-count", type=int, default=32)
    args = parser.parse_args(argv)
    report = run_acceptance(args.config, args.calibration, args.output_root,
                            seed=args.seed, point_count=args.point_count)
    print(json.dumps({"status": report["status"], "family": report["family"],
                      "report": str(args.output_root / "report.json")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
