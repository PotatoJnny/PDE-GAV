#!/usr/bin/env python3
"""Import six frozen science configurations, without submitting or resuming jobs.

Weights and evidence remain external. Generated configurations select the new
PDE-GAV composition; no Python source is loaded from the previous checkout.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]
SCHEMA = "pde_gav_plasma_gpu_v1"
ARMS = {"sol": ("gpt-5.6-sol", "max"), "astra": ("gpt-6-astra", "ultra")}


def file_sha(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def checked_file(path, digest):
    path = Path(path)
    if not path.is_file() or path.is_symlink() or file_sha(path) != digest:
        raise ValueError(f"migration input changed: {path}")
    return path.resolve()


def checked_source(path, revision):
    root = Path(path).resolve(strict=True)
    def git(*arguments):
        return subprocess.check_output(["git", "-C", str(root), *arguments], text=True).strip()
    if git("rev-parse", "HEAD") != revision or git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError(f"historical source is not its frozen clean revision: {root}")
    return root


def transform_config(original, *, equation, arm, source_root, config_path):
    """Only rebind paths/entrypoints; retain the science and experimental budget."""
    from pde_agent_verifier_solver.sdk import inspect_candidate_bundle

    config = deepcopy(original)
    expected_model = ARMS[arm]
    execution = config["execution"]
    if (execution["model"], execution["reasoning_effort"]) != expected_model:
        raise ValueError("source model/effort does not match the named arm")
    config["source_experiment"] = {
        "config": str(config_path), "sha256": file_sha(config_path),
        "source_root": str(source_root), "schema": original["schema"],
        "campaign_id": original["campaign_id"],
        "historical_campaign_imported": False,
    }
    config.update(schema=SCHEMA, equation=equation,
                  campaign_id=f"pde-gav-{equation}-{arm}-s101-20261002")
    config["worker"] = {
        "script": str(ROOT / "scripts/run_canonical_codex_agent.py"),
        "codex_config": str(ROOT / "scripts/canonical_codex.config.toml"),
    }
    execution["native_session_persistence"] = True
    # Preserve recorded attempt bounds for explicitly selecting the old policy.
    # New runs recover the same session within the original Agent deadline.
    execution["transient_resume_max_attempts"] = 3 if equation == "weak" else 1
    execution["transient_resume_until_deadline"] = True
    config.setdefault("protocol", {})["official_nomination_policy"] = {
        "mode": "single_primary_v1", "start_round": 1,
    }
    # Upstream now limits all Oracle hosts together; the historical value was
    # per host. Translate at this new-run boundary, never by mutating a live service.
    previous_oracle_limit = execution.get("oracle_max_workers", 1)
    if equation == "strong":
        per_device = config.get("migration", {}).get("oracle_execution_options", {}).get("workers_per_gpu", 4)
    elif equation == "weak":
        per_device = execution["workers_per_gpu"]
    else:
        per_device = 1
    global_oracle_limit = min(execution["candidate_case_workers"], 3 * per_device)
    if equation == "weak":
        global_oracle_limit = max(global_oracle_limit, execution["development_workers_per_gpu"])
    execution.update(solver_batch_size=1, gpu_slots_per_device=1,
                     oracle_concurrency_per_device=1, oracle_max_workers=global_oracle_limit)
    config["protocol"]["parallel_evaluation"] = "service_global_oracle_v1"
    config.setdefault("migration", {})["parallel_evaluation"] = {
        "previous_oracle_limit": previous_oracle_limit,
        "previous_scope": "per_oracle_host", "new_scope": "whole_candidate_service",
        "oracle_max_workers": global_oracle_limit,
        "solver_batch_size": 1, "gpu_slots_per_device": 1,
        "gpu_concurrency_not_automatically_increased": True,
    }
    config["official"] = {key: value for key, value in config["official"].items() if value}
    candidate = config["candidate"]
    old_bundle = Path(candidate["initial_bundle"])
    if not old_bundle.is_absolute():
        old_bundle = config_path.parent / old_bundle
    bundle = inspect_candidate_bundle(old_bundle)
    if candidate.get("initial_bundle_sha256", bundle.tree_sha256) != bundle.tree_sha256:
        raise ValueError("initial program no longer matches source configuration")
    local = ROOT / "configs/candidates" / old_bundle.name
    if local.is_dir() and inspect_candidate_bundle(local).tree_sha256 == bundle.tree_sha256:
        candidate["initial_bundle"] = str(local)
    else:
        candidate["initial_bundle"] = str(old_bundle.resolve())
    candidate["initial_bundle_sha256"] = bundle.tree_sha256
    if equation == "strong":
        # Relocates embedded decoder paths without changing checkpoint bytes.
        checkpoint = Path(config["backend"]["base_checkpoint"])
        config["asset_bindings"] = {"strong_asset_root": str(checkpoint.parents[2])}
    elif equation == "artemis":
        execution_path = source_root / "configs/plasma_artemis_d5_guided.json"
        backend = read(execution_path)
        config["backend"] = {
            "factory_settings": backend["factory_settings"],
            "source_config": {"path": str(execution_path), "sha256": file_sha(execution_path)},
        }
        # In the frozen deployment these instructions replaced inline prompts.
        suffix = "astra_ultra_" if arm == "astra" else ""
        package_path = source_root / f"configs/artemis_shared_gpu_{suffix}single_primary.json"
        package = read(package_path)
        checked_file(config_path, package["base_config_sha256"])
        for agent in config["agents"]:
            item = package["agent_instructions"][agent["agent_id"]]
            prompt = checked_file(package_path.parent / item["path"], item["sha256"])
            agent["prompt"] = prompt.read_text()
        config["source_experiment"]["primary_package_sha256"] = file_sha(package_path)
        for name in ("development_point_count", "official_point_count", "terminal_point_count"):
            config["task"][name] = config["evaluator"][name]
    return config


def prepare(plan_path, output, *, weak_assets):
    from plasma_gpu_campaign import validate_config, static_check
    if set(weak_assets) != {"portable_base", "historical_runtime_root", "historical_path_map"}:
        raise ValueError("weak migration only accepts the three external asset/evidence bindings")
    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    plan = read(plan_path)
    rows = plan["entries"]
    expected = {f"{equation}-{arm}" for equation in ("strong", "weak", "artemis") for arm in ARMS}
    if {row["name"] for row in rows} != expected or len(rows) != 6:
        raise ValueError("expected exactly the six authorized experiment arms")
    if output.exists():
        raise FileExistsError("preserve existing migration artifacts; choose a new output directory")
    prepared = []
    for row in rows:
        source_root = checked_source(row["source"], row["source_revision"])
        for item in row["files"]:
            checked_file(item["path"], item["sha256"])
        config_path = Path(row["config"])
        descriptor = next(item for item in row["files"] if item["path"] == str(config_path))
        checked_file(config_path, descriptor["sha256"])
        equation, arm = row["name"].split("-")
        config = transform_config(read(config_path), equation=equation, arm=arm,
                                  source_root=source_root, config_path=config_path)
        if equation == "weak":
            if set(weak_assets).intersection(config["backend"]):
                raise ValueError("migration asset bindings may not overwrite scientific backend settings")
            config["backend"].update(deepcopy(weak_assets))
        validate_config(config)
        report = static_check(config)
        prepared.append((row, config, report))
    # Validate the complete batch before creating any output.
    output.mkdir(parents=True, mode=0o700)
    receipt = {"schema": "pde_gav_plasma_migration_v1", "status": "static_checks_passed",
               "source_plan": {"path": str(plan_path), "sha256": file_sha(plan_path)},
               "new_source_root": str(ROOT), "formal_jobs_submitted": False,
               "old_jobs_modified": False, "cuda_qualified": False, "arms": []}
    for row, config, report in prepared:
        target = output / f"{row['name']}.json"
        target.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        receipt["arms"].append({"name": row["name"], "config": str(target),
                                "sha256": file_sha(target), "checks": report,
                                "previous_source_revision": row["source_revision"]})
    (output / "migration.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--weak-assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = prepare(args.source_plan, args.output, weak_assets=read(args.weak_assets))
    print(json.dumps({"status": receipt["status"], "arms": len(receipt["arms"]),
                      "output": str(args.output), "cuda_qualified": False,
                      "jobs_submitted": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
