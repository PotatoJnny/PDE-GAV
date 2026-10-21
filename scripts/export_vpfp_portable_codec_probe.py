"""One-shot conversion of an explicitly bound completed replay into a NEW codec.

CPU tests precede all conversion. This does not sample, solve a PDE, recover
the old e5 operator, enable guidance, or use a replay marker as a physics gate.
Run only inside the explicit zero-GPU CPU allocation with SHA-bound inputs.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

TREE = Path(__file__).resolve().parents[1]
SCRATCH = Path("${PDE_GAV_RUN_ROOT}")
PROBE_SOURCE = Path("${PDE_GAV_SOURCE_ROOT}/experiments/vpfp-strong-asset-probe-5297554")
OLD_TREE = Path("${PDE_GAV_WORKTREES_ROOT}/vpfp-strong-solver-guidance-20260908")
OLD_REVISION = "39177a47e9d3817f4e68129e7f2ad33d7627e82e"
OLD_ROOT = SCRATCH/"vpfp-strong-regime-coherent-5268586"
CHECKPOINT = OLD_ROOT/"seed-0/initial_conditioned/checkpoints/fixed-step-040000.pt"
CHECKPOINT_SHA = "255b8f77e418568b4d48b3669b95aa706cd83d94cee76d291806ae4e55794f62"
ARCHIVE = OLD_ROOT/"seed-0/initial_conditioned/step-040000/evaluation"
PRIMARY = {
    str(CHECKPOINT): CHECKPOINT_SHA,
    str(ARCHIVE/"generated-coefficients.npy"): "6e41a8c64e41f6989e4b4f2ed7bb30fead42905d7442ef89aa3a5538dd1f8d53",
    str(ARCHIVE/"generated-conditions.npy"): "2d24653a02bbb20c960bb23706088e18aa07a307478a6be1e1241424a0ccca8f",
    str(ARCHIVE/"generated-provenance.json"): "97e4d4c174f3bf413dd575ff842c6635a40c6f131e38abacd8f3f3ffc1340154",
    str(SCRATCH/"vpfp-strong-regime-followup-5286256/followup/frozen-plan.json"):
        "c0e9e9135229447f0017ef55f915401c5dadff273b8e83fc8567ec13ab8c1ec7",
}
SCHEMA = "vpfp_explicit_portable_codec_conversion_probe_v1"


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha_text(value):
    require(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value), "lowercase SHA256 required")
    return value


def file_sha(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"regular input required: {path}")
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path, digest):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"regular JSON input required: {path}")
    raw = path.read_bytes()
    return json.loads(raw)


def write_json(path, value):
    temporary = path.with_name(path.name+".partial")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def clean_revision(expected=None):
    revision = subprocess.check_output(["git", "rev-parse", "--verify", "HEAD^{commit}"], cwd=TREE, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=TREE, text=True).strip()
    require(not dirty and len(revision) == 40 and (expected is None or revision == expected), "source must remain fully committed and clean")
    return revision


def validate_report(report, plan, runtime, *, probe_root):
    probe_job = probe_root.name.removeprefix("vpfp-strong-portable-replay-probe-")
    require(report.get("schema") == "vpfp_diagnostic_new_normalizer_paired_replay_v1"
        and report.get("job_id") == probe_job and report.get("source_revision") == OLD_REVISION,
        "wrong explicit replay/source identity")
    require(report.get("execution_passed") is True and report.get("stage") == "complete"
        and report.get("planned_attempt_count") == 128 and report.get("completed_attempt_count") == 128,
        "only a fully executed 128-attempt replay can be converted")
    for key in ("inputs_final_verified", "network_final_bitwise_verified", "candidate_final_buffers_bitwise_verified"):
        require(report.get(key) is True, f"probe final check missing: {key}")
    for key in ("old_checkpoint_restored", "scientific_admission", "guidance_enabled", "production_assets_replaced", "base_weights_modified"):
        require(report.get(key) is False, f"unexpected historical/restoration claim: {key}")
    require(not report.get("failures"), "completed probe contains failures")
    require(plan.get("schema") == "vpfp_diagnostic_new_normalizer_paired_replay_plan_v1"
        and plan.get("source_revision") == OLD_REVISION and plan.get("sampler_steps") == 64
        and plan.get("churn") == 0. and plan.get("new_noise_or_condition_draws") is False
        and plan.get("allowed_normalizer_metadata_changes") == ["schur_operator_sha256"], "probe plan changed")
    for path, digest in PRIMARY.items():
        require(plan["input_sha256"].get(path) == digest and runtime["input_sha256"].get(path) == digest,
            f"original checkpoint/archive differs: {path}")
    for name in ("portable_replay_probe.py", "portable_replay_probe.slurm"):
        path = str(PROBE_SOURCE/name)
        require(path in plan["input_sha256"] and runtime["input_sha256"].get(path) == plan["input_sha256"][path],
            f"probe source fingerprint missing: {path}")
    candidate = report["candidate"]
    require(candidate.get("path") == str(probe_root/"candidate-normalizer-full-fp64-state.pt")
        and candidate.get("buffer_count") == 15 and runtime.get("candidate_normalizer_sha256") == candidate["sha256"],
        "wrong frozen fifteen-buffer candidate")
    rows, registered = report.get("rows"), plan.get("registered_rows")
    require(isinstance(rows, list) and isinstance(registered, list) and len(rows) == len(registered) == 128, "incomplete row registry")
    keys = ("row", "root_seed", "known_initial_seed", "future_noise_seed", "conditions", "stratum", "priority_eight")
    for index, (row, planned) in enumerate(zip(rows, registered)):
        require(row.get("row") == index and row.get("status") == "completed" and all(row.get(k) == planned.get(k) for k in keys),
            f"probe row does not match its frozen schedule: {index}")
        require(row["root_seed"] == 1920260906+index and row["stratum"] == index//16, "historical 128-root schedule differs")
    markers = {"completed": 128,
        "full_bitwise_equal_rows": sum(r["markers"]["all_full_coefficients_bitwise_equal"] is True for r in rows),
        "every_value_within_one_ulp_rows": sum(r["markers"]["every_value_within_one_ulp"] is True for r in rows),
        "maximum_value_ulp": max(r["paired"]["values"]["maximum_ulp"] for r in rows)}
    require(all(report["summary"].get(k) == value for k, value in markers.items()), "probe marker summary differs from per-row data")
    markers.update(all128_bitwise_equal=markers["full_bitwise_equal_rows"] == 128,
        all128_every_value_within_one_ulp=markers["every_value_within_one_ulp_rows"] == 128,
        comparison_unit="each FP32 reduced value, not mean error", marker_failure_blocks_export=False,
        physics_threshold=False, old_checkpoint_restored=False, guidance_ready=False)
    return markers


def run(args):
    job = os.environ.get("SLURM_JOB_ID", "")
    expected = SCRATCH/("vpfp-portable-codec-export-"+job)/"conversion"
    output = args.output.absolute()
    require(job.isdigit() and os.environ.get("SLURM_TMPDIR") and output == expected
        and output.parent.resolve() == expected.parent and not output.exists() and not output.is_symlink(),
        "Slurm and exact fresh conversion output required")
    probe_root = args.probe_result.absolute().parent
    probe_job = probe_root.name.removeprefix("vpfp-strong-portable-replay-probe-")
    require(args.probe_result.is_absolute() and args.probe_result.name == "result.json"
        and probe_root.parent == SCRATCH and probe_root.resolve() == probe_root
        and probe_root.name == "vpfp-strong-portable-replay-probe-"+probe_job and probe_job.isdigit(),
        "explicit SHA-bound replay job directory required; no latest-job discovery")
    require(math.isfinite(args.wall_seconds) and 60 < args.wall_seconds <= 1740, "conversion wall budget must fit the 30-minute CPU job")
    output.mkdir()
    started = time.monotonic()
    deadline = started+args.wall_seconds
    inputs, outputs = {}, {}
    revision = None
    report = {"schema": SCHEMA, "job_id": job, "hostname": socket.gethostname(), "requested_gpus": 0,
        "execution_passed": False, "stage": "source_binding", "failures": [],
        "old_checkpoint_restored": False, "guidance_ready": False, "guidance_enabled": False,
        "scientific_admission": False, "base_weights_modified": False, "new_samples": 0, "pde_solves": 0}

    def save():
        report["wall_seconds"] = time.monotonic()-started
        write_json(output/"result.json", report)

    def remaining():
        seconds = deadline-time.monotonic()
        if seconds <= 20:
            raise TimeoutError("conversion deadline reached; preserve partial artifacts")
        return seconds-15

    def verify(mapping):
        for path, digest in mapping.items():
            remaining()
            require(Path(path).is_file(), f"required file missing: {path}")

    save()
    try:
        revision = clean_revision()
        report["source_revision"] = revision
        for name in ("scripts/export_vpfp_portable_codec_probe.py", "scripts/run_vpfp_portable_codec_export_cpu.slurm",
                "src/pde_agent_verifier_solver/plugins/vpfp/portable_initial_conditioned.py", "tests/test_vpfp_portable_initial_conditioned.py"):
            inputs[str(TREE/name)] = file_sha(TREE/name)
        inputs[str(args.probe_result)] = sha_text(args.probe_result_sha256)
        probe = read_json(args.probe_result, args.probe_result_sha256)
        require(probe.get("execution_passed") is True and probe.get("completed_attempt_count") == 128,
            "failed or partial replay is not eligible for conversion")
        for name, key in (("frozen-plan.json", "frozen_plan_sha256"), ("frozen-runtime.json", "frozen_runtime_sha256")):
            inputs[str(probe_root/name)] = sha_text(probe[key])
        plan = read_json(probe_root/"frozen-plan.json", probe["frozen_plan_sha256"])
        runtime = read_json(probe_root/"frozen-runtime.json", probe["frozen_runtime_sha256"])
        report["replay_markers"] = validate_report(probe, plan, runtime, probe_root=probe_root)
        for path, digest in runtime["input_sha256"].items():
            require(Path(path).is_absolute(), "absolute frozen source path required")
            if path in inputs:
                require(inputs[path] == digest, "conflicting frozen input hashes")
            inputs[path] = sha_text(digest)
        candidate_path = Path(probe["candidate"]["path"])
        inputs[str(candidate_path)] = sha_text(probe["candidate"]["sha256"])
        for name in ("portable_replay_probe.py", "portable_replay_probe.slurm"):
            inputs[str(probe_root/"source-snapshots"/name)] = plan["input_sha256"][str(PROBE_SOURCE/name)]
        verify(inputs)
        frozen = {"schema": SCHEMA, "source_revision": revision, "input_sha256": inputs,
            "probe_result_sha256": args.probe_result_sha256, "source_checkpoint_sha256": CHECKPOINT_SHA,
            "no_sampling_or_guidance": True, "replay_markers_are_physics_thresholds": False}
        write_json(output/"frozen-plan.json", frozen)
        outputs[str(output/"frozen-plan.json")] = file_sha(output/"frozen-plan.json")
        report["frozen_plan_sha256"] = outputs[str(output/"frozen-plan.json")]
        report["stage"] = "portable_cpu_tests"
        save()
        command = [sys.executable, "-m", "pytest", "-o", "addopts=", "-q", "-p", "no:cacheprovider",
            "--basetemp", str(Path(os.environ["SLURM_TMPDIR"])/"vpfp-portable-export-tests"),
            "tests/test_vpfp_portable_initial_conditioned.py"]
        log_path = output/"mechanics-tests.log"
        report["tests"] = {"command": command, "log": str(log_path), "returncode": None}
        save()
        try:
            with log_path.open("x") as stream:
                completed = subprocess.run(command, cwd=TREE, stdout=stream, stderr=subprocess.STDOUT,
                    timeout=min(900., remaining()), check=False)
            report["tests"]["returncode"] = completed.returncode
            require(completed.returncode == 0, "portable CPU tests failed; conversion not attempted")
        finally:
            if log_path.is_file():
                report["tests"]["log_sha256"] = file_sha(log_path)
                outputs[str(log_path)] = report["tests"]["log_sha256"]
        report["stage"] = "candidate_conversion"
        save()
        remaining()
        import torch
        from pde_agent_verifier_solver.plugins.vpfp import portable_initial_conditioned as portable
        torch.set_num_threads(8)
        require(not torch.cuda.is_available(), "zero-GPU conversion must not see a GPU")
        diagnostic = portable._read_weights_only(candidate_path, probe["candidate"]["sha256"])
        checkpoint = portable._read_weights_only(CHECKPOINT, CHECKPOINT_SHA)
        require(diagnostic.get("schema") == "vpfp_explicit_diagnostic_candidate_normalizer_state_v1"
            and diagnostic.get("source_revision") == OLD_REVISION and diagnostic.get("old_checkpoint_sha256") == CHECKPOINT_SHA,
            "wrong diagnostic candidate payload")
        for key in ("base_weights_included", "original_e5_operator_recovered", "strict_original_checkpoint_loaded",
                "portable_loader_or_production_asset", "scientific_admission"):
            require(diagnostic.get(key) is False, f"unexpected diagnostic claim: {key}")
        require(diagnostic.get("candidate_constructed_without_old_fit_metadata") is True, "candidate construction provenance missing")
        require(diagnostic["original_normalizer_payload"] == checkpoint["normalizer"], "diagnostic original payload differs from checkpoint")
        for key, value, receipt_key in (("old_normalizer_payload_sha256", diagnostic["original_normalizer_payload"], "old_normalizer_payload_sha256"),
                ("candidate_normalizer_payload_sha256", diagnostic["candidate_normalizer_payload"], "new_normalizer_payload_sha256")):
            require(diagnostic[key] == json_sha(value) == probe["candidate"][receipt_key], "diagnostic normalization SHA differs")
        weights = portable.tensor_state_manifest(checkpoint["model"])
        raw_weight_hashes = {k: value["sha256"] for k, value in weights.items()}
        require(raw_weight_hashes == diagnostic["network_state_raw_bytes_sha256"] == probe["candidate"]["network_state_raw_bytes_sha256"],
            "diagnostic network bytes differ from original weights")
        manifest = portable.tensor_state_manifest(diagnostic["state_dict"])
        require(len(manifest) == 15 and all(v["dtype"] == "torch.float64" for v in manifest.values())
            and manifest == portable.tensor_state_manifest(diagnostic["named_buffers"]), "diagnostic state and named buffers differ")
        converted = portable.export_frozen_normalizer(diagnostic["original_normalizer_payload"], diagnostic["candidate_normalizer_payload"],
            diagnostic["state_dict"], source_checkpoint_sha256=CHECKPOINT_SHA,
            model_state_sha256=portable.tensor_state_sha256(checkpoint["model"]))
        require(converted["buffer_manifest"] == manifest, "export changed diagnostic buffer bytes")
        target = output/"portable-codec.pt"
        with target.open("xb") as stream:
            torch.save(converted, stream)
        target.chmod(0o444)
        target_sha = file_sha(target)
        outputs[str(target)] = target_sha
        report["candidate"] = {"path": str(target), "sha256": target_sha, "content_sha256": converted["sha256"],
            "identity": converted["identity"], "read_only_file": True, "new_candidate_not_restored_old_asset": True}
        report["stage"] = "no_eigh_native_sampler_load"
        save()
        remaining()
        def forbidden(*args, **kwargs):
            raise AssertionError("frozen-buffer loader attempted eigh/eigvalsh")
        with patch.object(torch.linalg, "eigh", forbidden), patch.object(torch.linalg, "eigvalsh", forbidden):
            sampler = portable.load_portable_initial_conditioned_sampler(CHECKPOINT, checkpoint_sha256=CHECKPOINT_SHA,
                portable_bundle_path=target, portable_bundle_sha256=target_sha, device="cpu", sample_steps=64, sample_churn=0.)
            sampler.normalizer.assert_frozen(full=True)
            require(portable.tensor_state_manifest(sampler.normalizer.state_dict()) == manifest, "loaded codec buffers changed")
            require(portable.tensor_state_manifest(sampler.model.state_dict()) == weights, "loaded original model bytes changed")
            compact = sampler.normalizer.to_payload()
            json.dumps(compact, allow_nan=False)
            require(compact["schema"] == portable.SCHEMA and compact["sha256"] == converted["sha256"], "compact codec identity differs")
            require(sampler.identity["checkpoint_sha256"] == CHECKPOINT_SHA
                and sampler.identity["checkpoint_schema"] == portable.SAMPLER_SCHEMA
                and sampler.identity["portable_bundle_sha256"] == target_sha, "sampler portable/source binding differs")
        report["load_validation"] = {"execution_passed": True, "eigh_eigvalsh_disabled": True,
            "loaded_buffer_count": len(manifest), "all_buffer_bytes_unchanged": True,
            "all_original_model_bytes_unchanged": True, "sampler_identity": sampler.identity,
            "compact_normalizer_identity": compact, "compact_normalizer_sha256": json_sha(compact),
            "samples_generated": 0, "prior_runtime_loaded": sampler.prior._runtime is not None}
        require(sampler.prior._runtime is None, "conversion unexpectedly loaded the physical prior runtime")
        verify(inputs)
        verify(outputs)
        clean_revision(revision)
        report.update(inputs_final_verified=True, outputs_final_verified=True, source_final_verified=True,
            execution_passed=True, stage="complete")
    except Exception as error:
        report["failures"].append({"stage": report["stage"], "type": type(error).__name__, "reason": str(error), "traceback": traceback.format_exc()})
        report["execution_passed"] = False
    finally:
        if revision is not None:
            try:
                clean_revision(revision)
            except Exception as error:
                report["execution_passed"] = False
                report["failures"].append({"stage": "final_source_check", "type": type(error).__name__, "reason": str(error)})
        report["input_sha256"], report["output_sha256"] = inputs, outputs
        save()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-result", type=Path, required=True)
    parser.add_argument("--probe-result-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=float, default=1700.)
    args = parser.parse_args(argv)
    def interrupted(signum, frame):
        raise TimeoutError(f"conversion interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    return 0 if run(args)["execution_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
