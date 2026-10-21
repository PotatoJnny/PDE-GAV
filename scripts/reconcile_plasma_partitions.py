#!/usr/bin/env python3
"""One sweep extending only verified, pending plasma jobs from L40S b2 to b2+b3.

Dry-run by default. This is not a daemon or a submitter: frozen deployments and
scientific state remain read-only, and receipts must live outside both. Old
frozen continuations still submit b2; call this again when inspecting successors.
"""
from __future__ import annotations

import argparse
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

# Site-specific Slurm account; see docs/SITE_CONFIGURATION.md.
SLURM_ACCOUNT = os.environ.get("PDE_GAV_SLURM_ACCOUNT", "${PDE_GAV_SLURM_ACCOUNT}")

BASE = "gpubase_l40s_b2"
TARGET = BASE + ",gpubase_l40s_b3"
ARMS = {f"{equation}-{model}" for equation in ("strong", "weak", "artemis")
        for model in ("sol", "astra")}
TIMESTAMPS = ("SubmitTime", "EligibleTime", "AccrueTime")
UNKNOWN = {None, "Unknown", "N/A", "(null)"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    path = Path(path)
    require(path.is_absolute() and not path.is_symlink() and path.is_file(),
            f"expected absolute regular file: {path}")
    return json.loads(path.read_text())


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def job_id(value):
    require(isinstance(value, str) and re.fullmatch(r"[1-9][0-9]*", value),
            "invalid accepted job ID")
    return value


def atomic_json(path, value):
    descriptor, name = tempfile.mkstemp(prefix=".receipt-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class Scheduler:
    """Only two commands are possible; arbitrary resource changes are not exposed."""

    def show(self, identifier):
        result = subprocess.run(["scontrol", "-o", "show", "job", job_id(identifier)],
                                capture_output=True, text=True, timeout=30)
        if result.returncode:
            if "Invalid job id specified" in result.stderr:
                return None
            raise RuntimeError(result.stderr.strip() or "scontrol show failed")
        raw = result.stdout.strip()
        require("\n" not in raw and raw, "expected exactly one scheduler job")
        fields = re.split(r"(?:^|\s)([A-Za-z][A-Za-z0-9_:/]*)=", raw)
        job = {fields[index]: fields[index + 1].strip()
               for index in range(1, len(fields), 2)}
        job["_raw"] = raw
        return job

    def expand(self, identifier):
        subprocess.run(["scontrol", "update", f"JobId={job_id(identifier)}",
                        f"Partition={TARGET}"], check=True, capture_output=True,
                       text=True, timeout=30)


def load_plan(path):
    plan = read_json(path)
    submitted = read_json(path.parent / "submissions.json")
    require(plan.get("schema") == "pde_gav_six_gpu_submission_v1", "wrong plan schema")
    require(submitted.get("schema") == "pde_gav_six_gpu_submission_receipt_v1"
            and submitted.get("status") == "all_six_submitted"
            and submitted.get("plan_sha256") == digest(path), "unbound submission receipt")
    entries, jobs = plan.get("entries", []), submitted.get("jobs", [])
    require(len(entries) == len(jobs) == 6
            and {item["name"] for item in entries} == ARMS
            and {item["name"] for item in jobs} == ARMS, "expected the six plasma arms")
    require(len({job_id(item["job_id"]) for item in jobs}) == 6,
            "duplicate initial job IDs")
    return plan, {item["name"]: item for item in jobs}


def run_targets(plan, entry, submitted):
    name = entry["name"]
    require(submitted["script"] == entry["script"] and submitted["site"] == entry["site"],
            "accepted wrapper/site differs from plan")
    site = read_json(entry["site"])
    require(digest(entry["site"]) == entry["site_sha256"], "site digest changed")
    wrapper = Path(entry["script"])
    require(wrapper.is_absolute() and not wrapper.is_symlink() and wrapper.is_file()
            and digest(wrapper) == entry["script_sha256"] == site["slurm_script_sha256"],
            "wrapper digest changed")
    short = name.startswith("artemis-")
    memory, hours = ("64G", 6) if short else ("128G", 12)
    require(site.get("schema") == "pde_gav_plasma_gpu_site_v1"
            and site.get("arm_name") == name and site.get("source_root") == plan["source_root"]
            and site.get("source_revision") == plan["source_revision"]
            and site.get("slurm_script") == str(wrapper), "site identity mismatch")
    require((site.get("account"), site.get("qos"), site.get("cpus"),
             site.get("gpu_count"), site.get("memory"), site.get("time"),
             site.get("max_segments")) ==
            (SLURM_ACCOUNT, "normal", 24, 4, memory, f"{hours:02d}:00:00", 12)
            and site.get("partition") in {BASE, TARGET}, "unapproved site resources")
    root = Path(site["run_root"])
    require(root.is_absolute() and not root.is_symlink(), "invalid run root")
    targets = [{"job_id": job_id(submitted["job_id"]), "comment": None}]
    state_path = root / "slurm_continuation.json"
    if state_path.exists():
        state = read_json(state_path)
        require(state.get("schema") == "ns_slurm_continuation_v1"
                and state.get("run_root") == str(root)
                and state.get("max_segments") == 12
                and re.fullmatch(r"[0-9a-f]{32}", state.get("run_id", "")),
                "continuation run identity mismatch")
        require(state.get("segments") and state["segments"][0].get("job_id") == targets[0]["job_id"]
                and state["segments"][0].get("segment_index") == 1,
                "continuation does not descend from accepted initial job")
        submissions = state.get("submissions")
        require(isinstance(submissions, list) and len(submissions) <= 11,
                "invalid bounded continuation list")
        expected = {"slurm_script": str(wrapper), "script_sha256": entry["script_sha256"],
                    "script_args": [], "gpu_count": 4, "account": SLURM_ACCOUNT, "qos": "normal",
                    "partition": site["partition"], "wall_seconds": hours * 3600,
                    "cpus": 24, "memory": memory}
        require(not submissions or state.get("submission_spec") == expected,
                "continuation submission resources/wrapper changed")
        for item in submissions:
            if item.get("status") not in {"submitted", "running"}:
                continue  # An uncertain intent is not an accepted job.
            intent = item.get("intent_id", "")
            require(re.fullmatch(r"[0-9a-f]{32}", intent)
                    and item.get("comment") == f"ns-continuation:{state['run_id']}:{intent}"
                    and isinstance(item.get("segment_index"), int)
                    and 2 <= item["segment_index"] <= 12,
                    "invalid accepted successor identity")
            targets.append({"job_id": job_id(item["job_id"]), "comment": item["comment"]})
    require(len({item["job_id"] for item in targets}) == len(targets), "duplicate accepted job IDs")
    return site, targets


def memory_mib(value):
    match = re.fullmatch(r"([0-9]+)([KMGT]?)", str(value))
    require(match is not None, "unrecognized memory request")
    return int(match[1]) * {"": 1, "K": 1 / 1024, "M": 1, "G": 1024, "T": 1024**2}[match[2]]


def validate_job(job, target, site, uid):
    require(job.get("JobId") == target["job_id"], "scheduler returned unrelated job")
    require(re.fullmatch(r"[^()]+\(" + str(uid) + r"\)", job.get("UserId", "")),
            "job belongs to another user")
    require(job.get("Command") == site["slurm_script"]
            and job.get("WorkDir") == site["source_root"], "live wrapper/source mismatch")
    require(target["comment"] is None or job.get("Comment") == target["comment"],
            "live continuation comment mismatch")
    require(job.get("Account") == SLURM_ACCOUNT and job.get("QOS") == "normal"
            and job.get("NumCPUs") == "24" and job.get("CPUs/Task") == "24"
            and job.get("NumNodes") in {"1", "1-1"} and job.get("NumTasks") == "1"
            and job.get("TimeLimit") == site["time"]
            and job.get("TresPerNode") == "gres/gpu:l40s:4", "live resource mismatch")
    tres = dict(item.split("=", 1) for item in job.get("ReqTRES", "").split(",") if "=" in item)
    require(tres.get("cpu") == "24" and tres.get("node") == "1"
            and tres.get("gres/gpu") == "4" and tres.get("gres/gpu:l40s") == "4"
            and memory_mib(tres.get("mem")) == memory_mib(site["memory"]), "live TRES mismatch")
    allowed_partitions = {BASE, TARGET}
    if job.get("JobState") != "PENDING":
        allowed_partitions.add("gpubase_l40s_b3")  # Slurm records the selected partition after start.
    require(job.get("Partition") in allowed_partitions, "unexpected live partition")


def verify_times(job, baseline):
    deferred = []
    for name in TIMESTAMPS:
        if job.get(name) == baseline.get(name):
            continue
        if name == "AccrueTime" and job.get(name) in UNKNOWN:
            deferred.append(name)
            continue
        raise ValueError(f"queue timestamp changed: {name}")
    return deferred


def reconcile_job(scheduler, target, site, receipt_dir, apply, uid):
    result = {"job_id": target["job_id"]}
    try:
        before = scheduler.show(target["job_id"])
        result["before"] = before
        if before is None:
            return {**result, "status": "not_in_scheduler"}
        validate_job(before, target, site, uid)
        baseline_path = receipt_dir / f"job-{target['job_id']}-baseline.json"
        baseline = read_json(baseline_path) if baseline_path.exists() else None
        if baseline:
            require(baseline.get("job_id") == target["job_id"]
                    and baseline.get("site") == site["slurm_script"], "baseline identity mismatch")
            result["deferred_timestamp_checks"] = verify_times(before, baseline["before"])
        if before.get("JobState") != "PENDING":
            return {**result, "status": "skipped_not_pending"}
        if before["Partition"] == TARGET:
            return {**result, "status": "already_expanded"}
        if not apply:
            return {**result, "status": "would_expand"}
        require(all(before.get(name) not in UNKNOWN for name in TIMESTAMPS),
                "queue timestamps unavailable; defer mutation")
        if baseline is None:
            baseline = {"job_id": target["job_id"], "site": site["slurm_script"], "before": before}
            atomic_json(baseline_path, baseline)  # Durable original timestamps before mutation.
        try:
            scheduler.expand(target["job_id"])
        except Exception as error:
            result["update_error"] = str(error)
            after = scheduler.show(target["job_id"])
            result["after"] = after
            if after is not None:
                validate_job(after, target, site, uid)
                if after.get("JobState") != "PENDING":
                    return {**result, "status": "race_no_retry"}
            raise
        after = scheduler.show(target["job_id"])
        result["after"] = after
        require(after is not None, "job left scheduler before verification; inspect receipt")
        validate_job(after, target, site, uid)
        result["deferred_timestamp_checks"] = verify_times(after, baseline["before"])
        if after.get("JobState") != "PENDING":
            return {**result, "status": "expanded_then_started"}
        require(after["Partition"] == TARGET, "partition update not verified")
        return {**result, "status": "expanded"}
    except Exception as error:
        return {**result, "status": "error", "error": f"{type(error).__name__}: {error}"}


def sweep(plan_path, receipt_dir, *, apply=False, scheduler=None, uid=None):
    plan_path, receipt_dir = Path(plan_path).absolute(), Path(receipt_dir).absolute()
    plan, submitted = load_plan(plan_path)
    require(not receipt_dir.is_symlink(), "receipt directory must not be a symlink")
    prepared, errors = [], []
    for entry in plan["entries"]:
        try:
            site, targets = run_targets(plan, entry, submitted[entry["name"]])
            prepared.append((entry["name"], site, targets))
        except Exception as error:
            errors.append({"arm": entry["name"], "status": "error", "error": str(error)})
    protected = [plan_path.parent.parent, Path(plan["source_root"])]
    protected.extend(Path(site["run_root"]) for _, site, _ in prepared)
    require(not any(receipt_dir.resolve().is_relative_to(path.resolve()) for path in protected),
            "receipts must be outside frozen deployment and scientific run roots")
    owners = {}
    for name, _, targets in prepared:
        for target in targets:
            owners.setdefault(target["job_id"], set()).add(name)
    receipt_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(receipt_dir / ".reconcile.lock", os.O_CREAT | os.O_RDWR
                         | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = list(errors)
        scheduler, uid = scheduler or Scheduler(), os.getuid() if uid is None else uid
        for name, site, targets in prepared:
            if any(len(owners[target["job_id"]]) != 1 for target in targets):
                rows.append({"arm": name, "status": "error", "error": "cross-run duplicate job ID"})
                continue
            for target in targets:
                rows.append({"arm": name, **reconcile_job(scheduler, target, site, receipt_dir, apply, uid)})
        receipt = {"schema": "plasma_partition_reconciliation_v1", "plan": str(plan_path),
                   "plan_sha256": digest(plan_path), "apply": apply, "target_partition": TARGET,
                   "created_at_unix": time.time(), "jobs": rows}
        atomic_json(receipt_dir / f"sweep-{time.time_ns()}.json", receipt)
        return receipt
    finally:
        os.close(descriptor)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--receipt-dir", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="Otherwise scheduler reads only")
    args = parser.parse_args(argv)
    result = sweep(args.plan, args.receipt_dir, apply=args.apply)
    print(json.dumps({"apply": result["apply"], "jobs": [
        {key: value for key, value in row.items() if key not in {"before", "after"}}
        for row in result["jobs"]]}, sort_keys=True))
    return int(any(row["status"] == "error" for row in result["jobs"]))


if __name__ == "__main__":
    raise SystemExit(main())
