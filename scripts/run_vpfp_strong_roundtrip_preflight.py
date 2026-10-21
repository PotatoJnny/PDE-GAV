"""Allocation-only strong codec preflight; no formal training or distribution audit.

Reuse the actual pilot's per-arm mechanics, and separately inspect the old
2048-train frozen normalizer on first-eight train/validation rows. This entry
never calls pilot.main, shared.train, or the distribution reference pipeline.
"""
from __future__ import annotations

import argparse
from copy import copy
from dataclasses import asdict
import gc
import importlib.util
import os
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
loader = importlib.util.spec_from_file_location("strong_gate_preflight_reuse", ROOT / "scripts/run_vpfp_strong_initial_pilot.py")
pilot = importlib.util.module_from_spec(loader)
loader.loader.exec_module(pilot)
shared, previous = pilot.shared, pilot.previous
from pde_agent_verifier_solver.plugins.vpfp.temporal_covariance_diffusion import VPFPTemporalCovarianceNormalizer, _json_sha

SOURCE = Path("${PDE_GAV_RUN_ROOT}/vpfp-strong-main-5244223")
CONFIG = ROOT / "configs/vpfp_unified_live_strong_collision.json"
SCRATCH = Path("${PDE_GAV_RUN_ROOT}")
SCHEMA = "vpfp_strong_roundtrip_preflight_only_v1"
SEED = 1620260906
PHYSICAL_MODULES = ("high_fidelity", "online_training", "torch_reference", "continuous_flow_training", "continuous_hermite")
PINS = {
    "corpus/manifest.json": "51440da877a10ddb5637b2ba3721d1f25e43ca8be4c1f311b371451d54687167",
    "frozen-plan.json": "5fb5ae34d9696ea775ed7dd2d63a2042b5e660c808d38872334c109a36d3d7e0",
    "result.json": "0c4bd41dfca5564b6c654437d4bc7745a1ad968c2b9e368b937e75e87505f9bf",
}


def freeze_inputs(arguments):
    hashes = {str(arguments.source / name): digest for name, digest in PINS.items()}
    previous.assert_hashes(hashes)
    # False skips coefficient-array contents for calibration/reference splits;
    # the existing loader still checks all corpus headers/identity metadata.
    manifest = previous.read_manifest(arguments.source / "corpus", verify_hashes=False)
    spec = previous.VPFPOnlineModelSpec.from_mapping(pilot.read_json(arguments.config)["vpfp"])
    if (manifest["split_counts"] != pilot.COUNTS or "strong" not in spec.regime_id
            or spec != previous.VPFPOnlineModelSpec.from_mapping(manifest["model_spec"])):
        raise ValueError("preflight requires the original strong-main 2048/128x4 corpus and decoder")
    models = spec.preflight()
    pilot.check_baseline_recipe(arguments.source, manifest)
    pilot.check_completed_source(arguments.source)
    for item in models["checkpoints"].values():
        hashes[item["path"]] = item["sha256"]
    physical = manifest["provenance"]["physical_module_sha256"]
    if set(physical) != set(PHYSICAL_MODULES):
        raise ValueError("original physical module inventory differs")
    for name, digest in physical.items():
        path = ROOT / "legacy/vpfp/src/agent_verifier/vpfp" / (name + ".py")
        hashes[str(path)] = digest
    hashes[str(arguments.config)] = shared.file_sha(arguments.config)
    hashes[str(arguments.source / "corpus" / manifest["seed_plan"]["path"])] = manifest["seed_plan"]["sha256"]
    # Hash only the two splits whose values will actually be read. This is a
    # streaming integrity check, not an in-memory 2048-row fit or evaluation.
    for role in ("train", "validation"):
        for entry in manifest["splits"][role]["files"].values():
            path = arguments.source / "corpus" / entry["path"]
            if path.is_symlink() or not path.resolve().is_relative_to((arguments.source / "corpus" / role).resolve()):
                raise ValueError("preflight split file escaped its fixed cohort")
            hashes[str(path)] = entry["sha256"]
    source_result = pilot.read_json(arguments.source / "result.json")
    summary = source_result["training"]["seed-0/cov_base"]
    checkpoint = arguments.source / "seed-0/cov_base/checkpoints/best.pt"
    if Path(summary["checkpoint"]).resolve() != checkpoint.resolve():
        raise ValueError("old frozen normalizer checkpoint path differs")
    hashes[str(checkpoint)] = summary["checkpoint_sha256"]
    previous.assert_hashes(hashes)
    return spec, manifest, summary, hashes


def frozen_normalizer(arguments, spec, manifest, summary):
    """Load only old checkpoint data; do not instantiate/sample the old model."""
    payload = torch.load(summary["checkpoint"], map_location="cpu", weights_only=True)
    if (set(payload) != {"schema", "config", "model", "normalizer", "base_decoder_sha256", "corpus_manifest_sha256", "training"}
            or payload["schema"] != "vpfp_temporal_covariance_diffusion_checkpoint_v1"
            or payload["corpus_manifest_sha256"] != manifest["manifest_sha256"]
            or payload["base_decoder_sha256"] != spec.base_decoder_sha256
            or _json_sha(payload["normalizer"]) != summary["normalizer_sha256"]):
        raise ValueError("old normalizer native schema/corpus/decoder/SHA differs")
    temporal = VPFPTemporalCovarianceNormalizer.from_payload(payload["normalizer"])
    train = previous.load_split(arguments.source / "corpus", "train", verify_hashes=False)
    if (temporal.fit_metadata["sample_ids"] != list(map(int, train.sample_ids))
            or temporal.config.full_config.to_dict() != shared.model_config(arguments, spec).to_dict()):
        raise ValueError("old 2048-train normalizer identities or full layout differ")
    return temporal


def old_normalizer_gate(arguments, manifest, temporal, arm, role):
    if arm == "cov_base":
        normalizer = temporal.to(arguments.device)
    elif arm == "initial_conditioned":
        config = pilot.VPFPInitialConditionedEDMConfig(temporal.config.full_config)
        normalizer = pilot.VPFPInitialConditionedNormalizer(config, temporal,
            initial_prior=pilot.KnownInitialPriorSpec.from_manifest(manifest)).to(arguments.device)
    else:
        raise ValueError("unknown preflight arm")
    split = previous.load_split(arguments.source / "corpus", role, verify_hashes=False)
    raw = torch.from_numpy(np.array(split.latents[:8], copy=True)).to(arguments.device)
    restored = normalizer.decode(normalizer.encode(raw))
    gate = pilot.audit_normalizer_roundtrip(normalizer, raw, restored)
    return {"execution_passed": bool(gate["passed"]), "arm": arm, "source_split": role,
        "sample_ids": list(map(int, split.sample_ids[:8])), "count": len(raw),
        "normalizer_fit_rows": len(temporal.fit_metadata["sample_ids"]), "normalizer_refitted": False,
        "conditional_operator_derived_from_old_temporal_normalizer": arm == "initial_conditioned",
        "normalization_gate": gate, "new_pde_solves": 0, "formal_training_updates": 0,
        "scientific_validation": False}


def execute(arguments):
    """Run independent bounded stages after the CLI's allocation/fresh guards."""
    arguments.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    code = shared.source_fingerprints()
    for name in (Path(__file__).name, "run_vpfp_strong_roundtrip_preflight.slurm", "run_vpfp_strong_initial_pilot.py",
                 "run_vpfp_temporal_training_pilot.py", "run_vpfp_value_control_pilot.py"):
        code["scripts/" + name] = shared.file_sha(ROOT / "scripts" / name)
    code = {str(ROOT / path): digest for path, digest in code.items()}
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    plan = {"schema": SCHEMA, "source": str(arguments.source), "config": str(arguments.config), "seed": arguments.seed,
        "source_revision": revision, "frozen_code_sha256": code, "pinned_source_sha256": PINS,
        "arms": list(pilot.ARMS), "mechanics_fit_train_rows": 8, "mechanics_batch_rows": 4,
        "mechanics_optimizer_updates_per_arm": 1, "mechanics_sample_steps": 2,
        "old_normalizer_fit_train_rows": 2048, "old_normalizer_gate_rows_per_split": 8,
        "old_normalizer_gate_splits": ["train", "validation"], "formal_training_updates": 0,
        "training_checkpoint_written": False, "new_pde_solves": 0, "distribution_audit_performed": False,
        "distribution_reference_features_opened": False, "reference_coefficient_values_evaluated": False,
        "corpus_all_split_header_and_lineage_checks": True,
        "physical_admission": False, "automatic_training_or_submission": False}
    shared.write_json(arguments.output / "frozen-plan.json", plan)
    frozen = {**code, str(arguments.output / "frozen-plan.json"): shared.file_sha(arguments.output / "frozen-plan.json")}
    reports = {}
    def stage(name, operation):
        shared.write_json(arguments.output / "stage.json", {"stage": name, "status": "running"})
        try:
            previous.assert_hashes(code)
            report = operation()
        except Exception as error:
            report = {"execution_passed": False, "error": repr(error), "traceback": traceback.format_exc(),
                "formal_training_updates": 0, "new_pde_solves": 0, "partial_artifacts_preserved": True}
        reports[name] = report
        shared.write_json(arguments.output / (name + ".json"), report)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return report
    try:
        spec, manifest, summary, inputs = freeze_inputs(arguments)
        frozen.update(inputs)
        shared.write_json(arguments.output / "frozen-inputs.json", {"sha256": inputs,
            "manifest_sha256": manifest["manifest_sha256"], "model_spec": asdict(spec),
            "old_normalizer_checkpoint_sha256": summary["checkpoint_sha256"],
            "old_normalizer_sha256": summary["normalizer_sha256"]})
        frozen[str(arguments.output / "frozen-inputs.json")] = shared.file_sha(arguments.output / "frozen-inputs.json")
        for arm in pilot.ARMS:
            run = copy(arguments)
            stage("mechanics-" + arm, lambda arm=arm, run=run: pilot.preflight(run, spec, manifest, arms=(arm,))[arm])
        successes = [reports["mechanics-" + arm] for arm in pilot.ARMS if reports["mechanics-" + arm]["execution_passed"]]
        paired = len(successes) == 2 and len({row["common_trunk_sha256"] for row in successes}) == 1
        stage("paired-initialization", lambda: {"execution_passed": paired,
            "both_common_trunk_initializations_identical": paired, "formal_training_updates": 0, "new_pde_solves": 0})
        # The old parent is reused without fitting; each stage uses the same
        # immutable payload, not a normalization selected on these rows.
        for arm in pilot.ARMS:
            for role in ("train", "validation"):
                stage(f"old2048-{arm}-{role}", lambda arm=arm, role=role:
                    old_normalizer_gate(arguments, manifest, frozen_normalizer(arguments, spec, manifest, summary), arm, role))
        previous.assert_hashes(frozen)
        passed = all(row["execution_passed"] for row in reports.values())
        result = {"execution_passed": bool(passed), "reports": reports}
    except Exception as error:
        result = {"execution_passed": False, "reports": reports, "error": repr(error), "traceback": traceback.format_exc(),
            "partial_artifacts_preserved": True}
    result.update(schema=SCHEMA, formal_training_updates=0, new_pde_solves=0, distribution_audit_performed=False,
        scientific_validation=False, physical_admission=False, automatic_training_or_submission=False,
        wall_seconds=time.monotonic()-started)
    shared.write_json(arguments.output / "result.json", result)
    shared.write_json(arguments.output / "stage.json", {"stage": "finished", "status": "complete" if result["execution_passed"] else "failed"})
    hashes = {str(path.relative_to(arguments.output)): shared.file_sha(path)
        for path in sorted(arguments.output.rglob("*.json"))}
    shared.write_json(arguments.output / "frozen-outputs.json", {"sha256": hashes, "self_excluded": True})
    return result


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--source", type=Path, default=SOURCE)
    cli.add_argument("--config", type=Path, default=CONFIG)
    cli.set_defaults(seed=SEED, device="cuda", width=256, depth=4, learning_rate=2.e-4, covariance_floor=.05)
    return cli


def main(argv=None):
    cli = parser()
    arguments = cli.parse_args(argv)
    job = os.environ.get("SLURM_JOB_ID", "")
    if not job.isdigit() or not os.environ.get("SLURM_TMPDIR"):
        cli.error("preflight-only real data requires a Slurm allocation")
    if (arguments.source.resolve() != SOURCE.resolve() or arguments.config.resolve() != CONFIG.resolve()
            or arguments.output.exists() or arguments.output.is_symlink()
            or arguments.output.resolve() != SCRATCH / f"vpfp-strong-roundtrip-preflight-{job}"):
        cli.error("fixed original source/config and fresh exact job output are required")
    if not torch.cuda.is_available():
        cli.error("preflight-only allocation requires usable CUDA")
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all", "--",
        "src", "scripts", "configs", "legacy/vpfp"], cwd=ROOT, text=True)
    if dirty.strip():
        cli.error("commit/freeze relevant source before preflight")
    arguments.source, arguments.config, arguments.output = arguments.source.resolve(), arguments.config.resolve(), arguments.output.resolve()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    return 0 if execute(arguments)["execution_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
