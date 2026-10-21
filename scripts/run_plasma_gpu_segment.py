#!/usr/bin/env python3
"""Run one frozen plasma allocation and continue only a clean, saved boundary.

The common Campaign owns scientific state and its CheckpointStore. This wrapper
verifies deployment identities and the four allocated GPU roles, then reuses the
existing exactly-once Slurm continuation record. Failures never receive a fresh
Agent budget. No old campaign, authentication contents, or environment dump is
copied into the new run.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

from pde_agent_verifier_solver.engine.checkpoint_store import atomic_json
from pde_agent_verifier_solver.engine.controller_gpu_visibility import (
    configure_controller_evaluation_devices,
)
from scripts import ns_slurm_continuation as continuation
from scripts.run_ns_draft_job import allocation_remaining_seconds, _checkpoint_progress

# Site-specific Slurm account; see docs/SITE_CONFIGURATION.md.
SLURM_ACCOUNT = os.environ.get("PDE_GAV_SLURM_ACCOUNT", "${PDE_GAV_SLURM_ACCOUNT}")

SCHEMA = "pde_gav_plasma_gpu_site_v1"
SAFE_STAGES = frozenset((
    "terminal_cohort_sealed", "bootstrap_completed", "before_guidance",
    "guidance_ready", "before_development", "development_completed",
    "official_sampled", "official_completed", "round_finished",
    "before_terminal_guidance", "terminal_guidance_ready", "terminal_sampled",
    "terminal_completed", "before_hidden_batch", "hidden_evaluation",
))
_SHA = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")


def file_sha(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checked_file(path, expected=None):
    path = Path(path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("deployment inputs must be absolute regular files")
    actual = file_sha(path)
    if expected is not None and (not isinstance(expected, str)
                                or _SHA.fullmatch(expected) is None or actual != expected):
        raise ValueError(f"deployment file identity changed: {path}")
    return path, actual


def _git(*arguments):
    return subprocess.run(["git", "-C", str(ROOT), *arguments], check=True,
                          capture_output=True, text=True, timeout=30).stdout.strip()


def verify_site(site_path):
    """Read-only, allocation-independent verification; never load model weights."""
    site_path, site_sha = _checked_file(site_path)
    site = json.loads(site_path.read_text())
    if site.get("schema") != SCHEMA or Path(site.get("source_root", "")).resolve() != ROOT:
        raise ValueError("site belongs to another source checkout")
    revision = site.get("source_revision")
    if (not isinstance(revision, str) or _REVISION.fullmatch(revision) is None
            or _git("rev-parse", "HEAD") != revision
            or _git("status", "--porcelain", "--untracked-files=all")):
        raise ValueError("plasma launch requires its clean frozen source revision")
    manifest_path, manifest_sha = _checked_file(
        site["source_manifest"], site["source_manifest_sha256"])
    manifest = json.loads(manifest_path.read_text())
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("frozen source manifest must contain files")
    names = set()
    for item in entries:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts or str(relative) in names:
            raise ValueError("source manifest contains an escaping or repeated path")
        names.add(str(relative))
        _checked_file(ROOT / relative, item["sha256"])
    required = {"scripts/run_plasma_gpu_segment.py", "scripts/plasma_gpu_campaign.py",
                "scripts/ns_slurm_continuation.py"}
    if not required.issubset(names):
        raise ValueError("source manifest omits a plasma execution entrypoint")
    config_path, config_sha = _checked_file(site["config_path"], site["config_sha256"])
    config = json.loads(config_path.read_text())
    equation = config.get("equation")
    if equation not in {"strong", "weak", "artemis"}:
        raise ValueError("site requires one of the three explicit plasma equations")
    resource_profile = "plasma_gpu_6h" if equation == "artemis" else "plasma_gpu_12h"
    partition = site.get("partition")
    if partition == continuation.PLASMA_L40S_PARTITIONS:
        resource_profile += "_b2_b3"
    elif partition != "gpubase_l40s_b2":
        raise ValueError("site resources require the approved b2 or b2,b3 partition selection")
    expected = ("64G", "06:00:00", 600) if equation == "artemis" else ("128G", "12:00:00", 2400)
    if ((site.get("account"), site.get("qos"), site.get("cpus"),
         site.get("gpu_count"), site.get("max_segments")) !=
            (SLURM_ACCOUNT, "normal", 24, 4, 12)
            or (site.get("memory"), site.get("time"), site.get("shutdown_margin_seconds"))
            != expected):
        raise ValueError("site resources differ from the approved four-GPU equation profile")
    root = Path(site["run_root"])
    if not root.is_absolute() or root.is_symlink() or root.resolve().is_relative_to(ROOT):
        raise ValueError("run root must be a distinct absolute persistent directory")
    root = root.resolve()
    python = Path(site["python"])
    if not python.is_absolute() or not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError("configured Python must be an existing absolute executable")
    codex, runtime_sha = _checked_file(site["codex_binary"], site["codex_sha256"])
    if not os.access(codex, os.X_OK):
        raise ValueError("qualified Codex runtime is not executable")
    auth = Path(site["auth_file"])
    if (not auth.is_absolute() or auth.is_symlink() or not auth.is_file()
            or stat.S_IMODE(auth.stat().st_mode) != 0o600):
        raise ValueError("authentication locator must name a regular mode-600 file")
    wrapper, wrapper_sha = _checked_file(site["slurm_script"], site.get("slurm_script_sha256"))
    # Checking the common helper also binds its exact explicit continuation resources.
    spec = continuation._submission_spec(wrapper, (), 4, resource_profile)
    identity = {"schema": SCHEMA, "site_path": str(site_path), "site_sha256": site_sha,
                "source_root": str(ROOT), "source_revision": revision,
                "source_manifest_sha256": manifest_sha, "config_sha256": config_sha,
                "codex_sha256": runtime_sha, "slurm_script_sha256": wrapper_sha,
                "run_root": str(root), "resource_profile": resource_profile,
                "resources": spec}
    return site, config, identity


def _binding_check(binding, config, campaign_root):
    from scripts.check_plasma_gpu_integration import check_binding
    return check_binding(config["equation"], binding, seed=config["master_seed"],
                         output_path=Path(campaign_root) / "qualification/binding.json")


def _validated_outcome(result, campaign, previous):
    if not isinstance(result, dict) or result.get("cleanup_complete") is not True:
        raise ValueError("campaign did not confirm complete execution-service cleanup")
    if result.get("status") == "completed":
        return {"status": "completed", "cleanup_complete": True}
    if (result.get("status") != "paused" or result.get("retry_safe") is not True
            or result.get("stage") not in SAFE_STAGES):
        raise ValueError("only a supported safe Campaign pause may automatically continue")
    checkpoint = campaign / "checkpoints/checkpoint.json"
    if Path(result.get("checkpoint_path", "")).resolve() != checkpoint.resolve():
        raise ValueError("paused campaign named a different checkpoint")
    _, digest = _checked_file(checkpoint)
    progress = _checkpoint_progress(checkpoint, "scientific")
    if previous is not None and progress == previous.get("pause_progress"):
        raise ValueError("no scientific progress; automatic continuation stopped")
    return {"status": "paused", "stage": result["stage"], "retry_safe": True,
            "cleanup_complete": True, "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": digest, "pause_progress": progress}


def run(site_path, *, campaign_run=None, role_probe=None,
        verify=verify_site, remaining_seconds=allocation_remaining_seconds,
        submitter=None):
    """Execute one allocation, preserving the previous lease/checkpoint identity."""
    _require_compute(os.environ)
    if (os.environ.get("SLURM_CPUS_PER_TASK") != "24"
            or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"):
        raise ValueError("plasma allocation requires 24 CPUs and CUDA_DEVICE_ORDER=PCI_BUS_ID")
    roles = allocation_roles(os.environ)
    if tuple(roles.values()) != ("0", "1", "2", "3"):
        raise ValueError("frozen profile requires Agent 0/1, Oracle 2 and Candidate 3")
    site_path = Path(site_path).resolve(strict=True)
    site, config, identity = verify(site_path)
    root = Path(site["run_root"]).resolve()
    campaign = root / "campaign"
    supplied = os.environ.get("NS_CONTINUATION_RUN_ROOT")
    if supplied and Path(supplied).resolve() != root:
        raise ValueError("successor environment belongs to another run")
    index = int(os.environ.get("NS_CONTINUATION_SEGMENT_INDEX", "1"))
    maximum = int(os.environ.get("NS_CONTINUATION_MAX_SEGMENTS", "12"))
    if not 1 <= index <= 12 or maximum != 12:
        raise ValueError("segment index/cap differs from the approved twelve-allocation limit")
    state_path = root / "segment-status.json"
    previous = None
    if index == 1:
        # Slurm may already have created this directory for stdout/stderr.
        if any((root / name).exists() for name in
               ("campaign", "segment-status.json", "slurm_continuation.json", "workflow.json")):
            raise FileExistsError("fresh launch refuses existing scientific or continuation state")
    else:
        previous = json.loads(state_path.read_text())
        if (previous.get("status") != "paused" or previous.get("identity") != identity
                or previous.get("cleanup_complete") is not True):
            raise ValueError("successor does not match a clean paused frozen deployment")
        _checked_file(previous["checkpoint_path"], previous["checkpoint_sha256"])
        if (os.environ.get("NS_CONTINUATION_CHECKPOINT_PATH") != previous["checkpoint_path"]
                or os.environ.get("NS_CONTINUATION_CHECKPOINT_SHA256") != previous["checkpoint_sha256"]):
            raise ValueError("successor checkpoint environment differs from the committed pause")
    record = {"schema": "pde_gav_plasma_gpu_segment_v1", "identity": identity,
              "job_id": os.environ["SLURM_JOB_ID"], "segment_index": index,
              "status": "running", "started_at_unix": time.time()}
    owned = False
    with continuation.runner_lease(root, record["job_id"], index, max_segments=12):
        owned = True
        atomic_json(state_path, record)
        try:
            probe = role_probe or probe_allocated_roles
            receipt = probe(root / "gpu_allocation" / record["job_id"],
                            python_executable=site["python"])
            if receipt.get("status") != "passed":
                raise ValueError("allocation GPU role probe did not pass")
            record["gpu_probe"] = str(root / "gpu_allocation" / record["job_id"] / "result.json")
            original_visible = os.environ["CUDA_VISIBLE_DEVICES"]
            try:
                mapping = configure_controller_evaluation_devices("2", ("0", "1"))
                if campaign_run is None:
                    from scripts.plasma_gpu_campaign import run as campaign_run
                result = campaign_run(
                    Path(site["config_path"]), campaign,
                    codex_binary=Path(site["codex_binary"]), auth_file=Path(site["auth_file"]),
                    worker_gpus=("0", "1"), controller_gpu="2",
                    evaluation_device_map=mapping,
                    allocation_seconds=remaining_seconds(record["job_id"]),
                    shutdown_margin_seconds=site["shutdown_margin_seconds"],
                    resume=index > 1, binding_check=_binding_check,
                )
            finally:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_visible
            record.update(_validated_outcome(result, campaign, previous))
            if verify(site_path)[2] != identity:
                raise ValueError("frozen deployment changed during allocation")
            record["finished_at_unix"] = time.time()
            atomic_json(state_path, record)
            continuation.publish_outcome(root, status=record["status"],
                checkpoint_path=record.get("checkpoint_path"),
                checkpoint_sha256=record.get("checkpoint_sha256"))
        except BaseException as error:
            if owned:
                record.update(status="failed", error_type=type(error).__name__,
                              error=str(error), finished_at_unix=time.time())
                atomic_json(state_path, record)
            raise
    if record["status"] == "paused":
        try:
            next_state = (submitter or continuation.submit_next)(
                root, slurm_script=site["slurm_script"], script_args=(),
                gpu_count=4, max_segments=12, resource_profile=identity["resource_profile"])
            record["continuation_status"] = next_state["status"]
        except continuation.SubmissionUncertainError as error:
            # Its durable submission intent prevents retries; allow an accepted
            # afterok job to run even if the parent's receipt was lost.
            record.update(continuation_status="submission_unknown", continuation_error=str(error))
        except Exception as error:
            record.update(continuation_status="stopped", continuation_error=str(error))
            atomic_json(state_path, record)
            raise
        atomic_json(state_path, record)
    return record


ROLE_NAMES = ("agent_broad", "agent_refiner", "controller_oracle", "candidate")
_TOKEN = re.compile(r"(?:[0-9]+|GPU-[0-9a-fA-F-]+)")


def allocation_roles(environment: Mapping[str, str]) -> dict[str, str]:
    """Use inherited visible order, rejecting ambiguous site-specific aliases."""
    visible = tuple(environment.get("CUDA_VISIBLE_DEVICES", "").split(","))
    allocated = tuple(environment.get("SLURM_JOB_GPUS", "").split(","))
    if (len(visible) != 4 or len(set(visible)) != 4
            or any(_TOKEN.fullmatch(value) is None for value in visible)
            or len(allocated) != 4 or set(allocated) != set(visible)):
        raise ValueError("four distinct visible GPU tokens must match the explicit Slurm allocation")
    return dict(zip(ROLE_NAMES, visible, strict=True))


def normalize_uuid(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("a concrete physical GPU UUID is required")
    normalized = value.lower().removeprefix("gpu-").replace("-", "")
    if re.fullmatch(r"[0-9a-f]{32}", normalized) is None:
        raise ValueError("a concrete physical GPU UUID is required")
    return normalized


def validate_role_reports(roles: Mapping[str, str], reports: Mapping) -> dict[str, str]:
    """Reject numeric/UUID aliases and CUDA versus NVIDIA numbering mismatch."""
    from pde_agent_verifier_solver.engine.gpu_execution import validate_cuda_metadata

    if (set(roles) != set(ROLE_NAMES) or len(set(roles.values())) != 4
            or set(reports) != set(ROLE_NAMES)):
        raise ValueError("all four distinct assigned roles must be probed")
    identities = {}
    for role, token in roles.items():
        row = reports[role]
        if not isinstance(row, Mapping) or row.get("status") != "passed" or row.get("token") != token:
            raise ValueError("a role did not pass on its assigned token")
        metadata = validate_cuda_metadata(row.get("cuda_metadata"), token)
        actual = normalize_uuid(metadata["uuid"])
        if normalize_uuid(row.get("nvidia_uuid")) != actual:
            raise ValueError("CUDA and NVIDIA allocation UUIDs differ; do not guess device aliases")
        if token.startswith("GPU-") and normalize_uuid(token) != actual:
            raise ValueError("observed GPU differs from its explicit UUID token")
        identities[token] = actual
    if len(set(identities.values())) != 4:
        raise ValueError("assigned role tokens alias a physical GPU UUID")
    return identities


def _require_compute(environment: Mapping[str, str]) -> None:
    if (not environment.get("SLURM_JOB_ID", "").isdigit()
            or "login" in socket.gethostname().lower()):
        raise RuntimeError("GPU startup probes require a Slurm compute allocation")


def _probe_token(token: str) -> dict:
    """Internal child entry: one token, no model, no numerical Oracle assets."""
    _require_compute(os.environ)
    if (_TOKEN.fullmatch(token) is None
            or os.environ.get("CUDA_VISIBLE_DEVICES") != token
            or token not in os.environ.get("SLURM_JOB_GPUS", "").split(",")):
        raise ValueError("child GPU token must exactly match its inherited allocation")
    discovery = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader", "-i", token],
        capture_output=True, text=True, timeout=10, check=True,
    )
    rows = discovery.stdout.strip().splitlines()
    if len(rows) != 1:
        raise ValueError("NVIDIA query must identify exactly one allocated GPU")
    normalize_uuid(rows[0])
    sys.path.insert(0, str(ROOT / "src"))
    import torch
    from pde_agent_verifier_solver.engine.gpu_execution import (
        initialize_cuda_device, synchronize_cuda_device,
    )

    torch.set_num_threads(1)
    metadata = initialize_cuda_device(token)
    synchronize_cuda_device(metadata)
    if normalize_uuid(metadata["uuid"]) != normalize_uuid(rows[0]):
        raise ValueError("CUDA and NVIDIA allocation UUIDs differ; do not guess device aliases")
    return {"status": "passed", "token": token, "nvidia_uuid": rows[0],
            "cuda_metadata": metadata}


def _write_report(path: Path, report: dict) -> None:
    atomic_json(path, report)


def probe_allocated_roles(output_dir: Path, *, environment: Mapping[str, str] | None = None,
                          python_executable: str = sys.executable,
                          timeout_seconds: float = 60) -> dict:
    """Record distinct physical roles without changing any caller CUDA state.

    Returns a passed receipt or raises after preserving partial failure evidence.
    The output directory must be new. Its receipt is allocation-specific, not a
    reusable scientific admission or a replacement for the real Agent gate.
    """
    env = dict(os.environ if environment is None else environment)
    _require_compute(env)
    roles = allocation_roles(env)
    if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 120):
        raise ValueError("startup child timeout must be finite and within 120 seconds")
    output = Path(output_dir)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    report = {"schema": "shared_gpu_launch_roles_v1", "status": "running",
        "slurm_job_id": env["SLURM_JOB_ID"], "hostname": socket.gethostname(),
        "initial_cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
        "slurm_job_gpus": env["SLURM_JOB_GPUS"], "roles": roles,
        "role_probes": {}, "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "native_agent_gate": "required_at_real_agent_launch",
        "performance_benchmark": False, "scientific_validation": False,
        "model_invoked": False, "caller_environment_changed": False}
    started = time.monotonic()
    _write_report(output / "result.json", report)
    try:
        for role, token in roles.items():
            report["stage"] = role
            _write_report(output / "result.json", report)
            child_env = dict(env, CUDA_VISIBLE_DEVICES=token, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            with (output / f"{role}.stdout.log").open("w") as stdout, \
                    (output / f"{role}.stderr.log").open("w") as stderr:
                completed = subprocess.run(
                    [str(python_executable), str(Path(__file__).resolve()), "--probe-token", token],
                    cwd=ROOT, env=child_env, stdout=stdout, stderr=stderr,
                    timeout=timeout_seconds, check=False,
                )
            if completed.returncode != 0:
                raise RuntimeError(f"{role} CUDA startup failed; inspect retained role logs")
            rows = (output / f"{role}.stdout.log").read_text().strip().splitlines()
            if not rows:
                raise ValueError(f"{role} CUDA startup returned no readiness metadata")
            report["role_probes"][role] = json.loads(rows[-1])
        report["physical_uuid_by_token"] = validate_role_reports(roles, report["role_probes"])
        report.update(status="passed", stage="complete")
        return report
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - started
        _write_report(output / "result.json", report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--site", type=Path)
    mode.add_argument("--probe-token", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.probe_token is not None:
        print(json.dumps(_probe_token(args.probe_token), allow_nan=False))
    else:
        record = run(args.site)
        print(json.dumps({"status": record["status"], "job_id": record["job_id"],
                          "segment_index": record["segment_index"],
                          "continuation_status": record.get("continuation_status")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
