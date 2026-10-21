#!/usr/bin/env python3
"""Prepare six fresh transport-v2 experiments; never submit/cancel/resume jobs.

Consumes the previous six frozen configurations and a real, frozen independent
reference calibration. Historical scores, Guidance and Campaign Memory are not
imported. Model assets and experimental budgets remain unchanged.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

from scripts import plasma_gpu_campaign as campaign
from pde_agent_verifier_solver.plugins.plasma_transport import (
    TRANSPORT_PROTOCOL_ID, load_transport_calibration,
)
from pde_agent_verifier_solver.plugins.plasma_transport_settings import transport_brief
from pde_agent_verifier_solver.sdk import inspect_candidate_bundle

FAMILIES = ("strong", "weak", "artemis")
QUALIFICATION_SCHEMA = "plasma_transport_v2_qualification_v1"
VALIDATION_STAGES = tuple("collect-" + family for family in FAMILIES) + ("fit",) + tuple(
    "acceptance-" + family for family in FAMILIES)


def _regular_json(path):
    path = Path(path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("qualification evidence must be an absolute regular file")
    return json.loads(path.read_text()), campaign.file_sha(path)


def _science_hash(config):
    return sha256(json.dumps({key: config.get(key) for key in ("equation", "backend", "asset_bindings")},
                             sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validation_qualification(validation_root, calibration, *, reference_config_hashes=None):
    """Require all three real-family acceptances from this exact source tree."""
    from pde_agent_verifier_solver.engine.checkpoint_store import source_identity
    root = Path(validation_root)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError("validation_root must be an absolute regular directory")
    root = root.resolve()
    report, report_sha = _regular_json(root / "validation.json")
    frozen, calibration_sha = _regular_json(root / "calibration.json")
    checked = load_transport_calibration(frozen)
    stages = report.get("stages", ())
    if (report.get("schema") != "plasma_transport_v2_validation_pipeline_v1"
            or report.get("status") != "passed"
            or tuple(row.get("name") for row in stages) != VALIDATION_STAGES
            or any(row.get("status") != "passed" or row.get("exit_code") != 0 for row in stages)):
        raise ValueError("all seven calibration and native acceptance stages must pass")
    if (report.get("source_identity") != source_identity(ROOT)
            or report.get("calibration_sha256") != calibration_sha
            or report.get("objective_hash") != checked.objective_hash
            or checked.objective_hash != calibration.objective_hash):
        raise ValueError("validation source, calibration bytes, or objective changed")
    references = report.get("reference_config_hashes", {})
    if set(references) != set(FAMILIES) or (reference_config_hashes is not None and references != reference_config_hashes):
        raise ValueError("validation reference configurations differ from the migration inputs")
    acceptance_hashes, bundles, science = {}, {}, {}
    for family in FAMILIES:
        receipt, receipt_sha = _regular_json(root / f"acceptance-{family}" / "report.json")
        if (receipt.get("schema") != "plasma_macro_transport_v2_native_acceptance_v1"
                or receipt.get("family") != family or receipt.get("status") != "passed"
                or receipt.get("private_cases_released") is not True
                or receipt.get("config_sha256") != references[family]
                or receipt.get("calibration_file_sha256") != calibration_sha
                or receipt.get("objective_identity", {}).get("objective_hash") != calibration.objective_hash):
            raise ValueError(f"{family} acceptance did not pass with the same calibration and clean up")
        audits = receipt.get("candidate_audits", ())
        if len(audits) != 2 or any(row.get("status") != "completed" or row.get("process_group_cleaned") is not True for row in audits):
            raise ValueError(f"{family} acceptance candidate processes did not clean up")
        source, source_sha = _regular_json(receipt.get("config_path", ""))
        if source_sha != references[family] or source.get("equation") != family:
            raise ValueError(f"{family} accepted reference configuration changed")
        bundle_sha = receipt.get("candidate_bundle_sha256", "")
        if not isinstance(bundle_sha, str) or re.fullmatch(r"[0-9a-f]{64}", bundle_sha) is None:
            raise ValueError(f"{family} acceptance omitted its immutable S0 bundle")
        acceptance_hashes[family], bundles[family], science[family] = receipt_sha, bundle_sha, _science_hash(source)
    return {"schema": QUALIFICATION_SCHEMA, "validation_root": str(root),
            "validation_sha256": report_sha, "calibration_sha256": calibration_sha,
            "objective_hash": calibration.objective_hash, "source_identity": report["source_identity"],
            "reference_config_hashes": references, "acceptance_sha256": acceptance_hashes,
            "candidate_bundle_sha256": bundles, "science_config_sha256": science}


def check_transport_qualification(config):
    """Deployment gate; synthetic validate_config alone is never admission."""
    qualification = config.get("transport_qualification")
    if not isinstance(qualification, dict) or qualification.get("schema") != QUALIFICATION_SCHEMA:
        raise ValueError("transport deployment requires passed native validation evidence")
    calibration = load_transport_calibration(config["task"]["metric_calibration"])
    current = validation_qualification(qualification.get("validation_root", ""), calibration)
    if current != qualification:
        raise ValueError("transport qualification receipt bytes or identities changed")
    family = config["equation"]
    if _science_hash(config) != qualification["science_config_sha256"][family]:
        raise ValueError("formal model assets differ from the accepted family configuration")
    bundle = inspect_candidate_bundle(config["candidate"]["initial_bundle"])
    if bundle.tree_sha256 != qualification["candidate_bundle_sha256"][family]:
        raise ValueError("formal S0 differs from the native-accepted bundle")
    return qualification


def agent_prompt(equation, index, calibration):
    role = ("Explore independent numerical, reduced or learned methods; no solver family is prescribed."
            if index == 0 else
            "Independently diagnose the incumbent and test improvements in stability, quadrature, conservation and generalization.")
    abi = ("Return M0, M1, M2, M3 and E. These are raw velocity moments; the evaluator converts "
           "each answer's own moments into central pressure and heat flux. M3 itself is NOT the scored heat flux."
           if equation == "artemis" else
           "Return rho, electric_x, electric_y, momentum_x, momentum_y, pressure_xx, pressure_xy, "
           "pressure_yy, heat_flux_x and heat_flux_y. Pressure and heat flux are CENTRAL moments; "
           "2D heat flux includes mixed third moments. kinetic_energy is not an independent primary target.")
    return f"""{equation} / plasma_macro_transport_v2: {role}

This is a fresh seed-101 five-round experiment with a deliberately simple S0 and empty Campaign Memory.
Do not import previous workspaces, candidates, scores, hidden cases or Guidance. Only use staged public
resources, the incumbent, Campaign Memory and approved tools. Predict macros, not the full distribution.
{abi}
{transport_brief(calibration)}

Your Agent has 14400 seconds TOTAL elapsed time per round, including tool calls and Controller waits.
The adaptive allowance is 32 fresh Development Cases INCLUDING four initial BASE cases and every
perturbation child. Base/Guided use is optional; no Development mix is forced. Hidden truth is unavailable.
Development scores use 128 private points; Official/Terminal use 256. Respect each case's public time
domain and causal source access. Scoring coordinates are not available during preparation.

Export prepare_solution(case, solve_budget=None), returning a handle with evaluate(points_query)
-> AnswerView. A Case has ONE shared 900 charged-second budget for startup/imports, preparation,
evaluation and serialization; NO separate preparation cap. solve_budget is remaining time, not a fresh
allowance. Only Controller-metered Oracle queue/shared-lock waiting is excluded. Oracle numerical work,
transfer, and your solver work are charged. Use the SDK solve-budget clock and leave time to return outputs.

You have a dedicated GPU and CPU resources. GPU-native computation is permitted, not mandatory.
Candidate evaluation uses its assigned GPU. Follow the staged SDK public Oracle interface; do not assume
Oracle tensors or CUDA contexts share your process. You may train models and include checkpoints/data
needed at inference in your candidate bundle manifest; keep imports and asset loading within the Case
budget. Never include credentials, hidden answers or dependencies on files outside the submitted bundle.

Every Official contender sees the SAME 256 fresh cases: 86 BASE, 85 GUIDED, 85 BASE-perturbed children
of distinct BASE parents. Promotion minimizes the case-weighted overall mean of the transport score.
Failures cannot be removed to improve that mean; successful-subset diagnostics do not determine promotion.
The incumbent is evaluated separately. Only ONE explicitly selected primary from each Agent enters Official.
Archive alternatives if useful, but do not submit the incumbent just to retain it.

Use python -m pde_agent_verifier_solver.engine.agent_tool submit candidate_bundle --sequence 1
--name NAME --summary EVIDENCE --usage 'prepare_solution then evaluate' --primary (one shell command).
All submissions share contiguous sequence numbers starting at 1. A successful --primary updates your
immutable primary; plain submit only archives. Re-select an older alternative by a new primary submission.
A rejected pre-commit update leaves the old primary intact: fix and retry the same next sequence.
Verify receipts; edits or chat statements do not change a frozen submission. No primary means no contender.
Record evidence, selected primary and known limitations using agent_tool log NOTES.md before finishing.

Guidance uses this SAME transport objective and the incumbent's first 80 clean BASE feedback rows in order;
the fitted snapshot is shared/frozen for the round and the Base generator stays fixed. Terminal uses common
S0/G1 cohorts: 256 BASE+128 BASE-perturbed, 256 G1-GUIDED+128 G1-GUIDED-perturbed. Deliver S5;
Terminal is diagnostic, not a mechanism to select a solver retrospectively.
"""


def transform(original, *, name, source_path, calibration, run_tag):
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", run_tag):
        raise ValueError("run_tag must be a simple new identifier")
    equation, arm = name.split("-")
    if equation not in ("strong", "weak", "artemis") or arm not in ("sol", "astra"):
        raise ValueError("unknown experiment arm")
    if original.get("schema") != campaign.SCHEMA or original.get("equation") != equation:
        raise ValueError("expected a frozen v1 configuration for this equation")
    expected = ("gpt-5.6-sol", "max") if arm == "sol" else ("gpt-6-astra", "ultra")
    if (original["execution"]["model"], original["execution"]["reasoning_effort"]) != expected:
        raise ValueError("model arm identity differs")
    config = deepcopy(original)
    config.update(schema=campaign.TRANSPORT_SCHEMA,
                  campaign_id=f"pde-gav-{name}-transport-v2-s101-{run_tag}")
    config["source_experiment"] = {"config": str(source_path),
        "sha256": campaign.file_sha(source_path), "campaign_id": original["campaign_id"],
        "historical_campaign_imported": False, "previous_scores_imported": False,
        "previous_guidance_imported": False}
    metric = {"metric_profile": TRANSPORT_PROTOCOL_ID, "metric_calibration": calibration.to_dict()}
    config["task"].update(deepcopy(metric))
    config.setdefault("generator", {}).setdefault("settings", {}).update(deepcopy(metric))
    config["protocol"].update(metric_profile=TRANSPORT_PROTOCOL_ID,
        scoring_objective_hash=calibration.objective_hash,
        parallel_evaluation="service_global_oracle_v1")
    execution = config["execution"]
    if original["protocol"].get("parallel_evaluation") != "service_global_oracle_v1":
        per_device = (config.get("migration", {}).get("oracle_execution_options", {}).get("workers_per_gpu", 4)
                      if equation == "strong" else execution["workers_per_gpu"] if equation == "weak" else 1)
        limit = min(execution["candidate_case_workers"], 3 * per_device)
        if equation == "weak":
            limit = max(limit, execution["development_workers_per_gpu"])
        execution.update(solver_batch_size=1, gpu_slots_per_device=1,
                         oracle_concurrency_per_device=1, oracle_max_workers=limit)
    config["worker"] = {"script": str(ROOT / "scripts/run_canonical_codex_agent.py"),
                        "codex_config": str(ROOT / "scripts/canonical_codex.config.toml")}
    bundle = ROOT / "configs/candidates" / ("artemis_hold_initial_bundle" if equation == "artemis"
                                             else "vpfp_transport_simple_bundle")
    inspected = inspect_candidate_bundle(bundle)
    config["candidate"].update(initial_bundle=str(bundle), initial_bundle_sha256=inspected.tree_sha256,
        initial_name="Simple stable transport S0",
        initial_summary=("Hold public initial macros." if equation == "artemis" else
                         "Public rho0 and periodic Poisson; static Maxwellian closure j=qh=0, P=rho*T*I."))
    for index, agent in enumerate(config["agents"]):
        agent["prompt"] = agent_prompt(equation, index, calibration)
    config.setdefault("migration", {})["transport_v2"] = {
        "objective_hash": calibration.objective_hash, "new_experiment_required": True,
        "simple_s0_not_accuracy_qualified": True, "base_model_assets_unchanged": True}
    return campaign.validate_config(config)


def prepare(source_directory, calibration_path, output, *, run_tag, validation_root):
    calibration = load_transport_calibration(calibration_path)
    source_directory, output = Path(source_directory).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("choose a fresh configuration output directory")
    reference_hashes = {family: campaign.file_sha(source_directory / f"{family}-sol.json") for family in FAMILIES}
    qualification = validation_qualification(Path(validation_root).absolute(), calibration,
                                              reference_config_hashes=reference_hashes)
    if campaign.file_sha(calibration_path) != qualification["calibration_sha256"]:
        raise ValueError("migration calibration file differs from the validated calibration bytes")
    rows = []
    for equation in ("strong", "weak", "artemis"):
        for arm in ("sol", "astra"):
            name = f"{equation}-{arm}"
            source = source_directory / f"{name}.json"
            config = transform(json.loads(source.read_text()), name=name, source_path=source,
                               calibration=calibration, run_tag=run_tag)
            config["transport_qualification"] = deepcopy(qualification)
            rows.append((name, config, campaign.static_check(config)))
    output.mkdir(parents=True, mode=0o700)
    receipt = {"schema": "pde_gav_plasma_transport_migration_v2", "status": "static_checks_passed",
        "objective_hash": calibration.objective_hash, "new_source_root": str(ROOT), "arms": [],
        "formal_jobs_submitted": False, "old_jobs_modified": False, "cuda_qualified": True,
        "transport_qualification": qualification}
    for name, config, checks in rows:
        target = output / f"{name}.json"
        target.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        receipt["arms"].append({"name": name, "config": str(target),
                                "sha256": campaign.file_sha(target), "checks": checks})
    (output / "migration.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    args = parser.parse_args()
    receipt = prepare(args.source_directory, args.calibration, args.output, run_tag=args.run_tag,
                      validation_root=args.validation_root)
    print(json.dumps({"status": receipt["status"], "arms": len(receipt["arms"]),
                      "jobs_submitted": False, "objective_hash": receipt["objective_hash"]}))


if __name__ == "__main__":
    main()
