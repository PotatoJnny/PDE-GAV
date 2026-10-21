"""No real scheduler operations: frozen data and a six-arm fake Slurm service."""
from copy import deepcopy
import json

import pytest

from scripts import reconcile_plasma_partitions as reconcile


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path


class FakeScheduler:
    def __init__(self, jobs):
        self.jobs, self.updates = jobs, []
        self.race = False
        self.transient_accrue = False
        self.start_after_expansion = False

    def show(self, identifier):
        return deepcopy(self.jobs.get(identifier))

    def expand(self, identifier):
        self.updates.append(identifier)
        if self.race:
            self.jobs[identifier]["JobState"] = "RUNNING"
            raise RuntimeError("job is no longer pending")
        self.jobs[identifier]["Partition"] = reconcile.TARGET
        if self.transient_accrue:
            self.jobs[identifier]["AccrueTime"] = "Unknown"
        if self.start_after_expansion:
            self.jobs[identifier].update(JobState="RUNNING", Partition="gpubase_l40s_b3")


@pytest.fixture
def deployment(tmp_path):
    source = tmp_path / "frozen/source"
    source.mkdir(parents=True)
    directory = tmp_path / "frozen/deployment"
    entries, submitted, sites, jobs = [], [], {}, {}
    for index, arm in enumerate(sorted(reconcile.ARMS)):
        short = arm.startswith("artemis")
        identifier = str(100 + index)
        wrapper = directory / arm / "run.slurm"
        wrapper.parent.mkdir(parents=True)
        wrapper.write_text("#!/bin/bash\nexit 0\n")
        site = {"schema": "pde_gav_plasma_gpu_site_v1", "arm_name": arm,
                "source_root": str(source), "source_revision": "a" * 40,
                "slurm_script": str(wrapper), "slurm_script_sha256": reconcile.digest(wrapper),
                "run_root": str(tmp_path / "runs" / arm), "account": "${PDE_GAV_SLURM_ACCOUNT}", "qos": "normal",
                "cpus": 24, "gpu_count": 4, "memory": "64G" if short else "128G",
                "time": "06:00:00" if short else "12:00:00", "max_segments": 12,
                "partition": reconcile.BASE}
        site_path = write_json(directory / arm / "site.json", site)
        entry = {"name": arm, "script": str(wrapper), "site": str(site_path),
                 "script_sha256": reconcile.digest(wrapper), "site_sha256": reconcile.digest(site_path)}
        entries.append(entry)
        submitted.append({"name": arm, "script": str(wrapper), "site": str(site_path), "job_id": identifier})
        sites[arm] = site
        jobs[identifier] = {"JobId": identifier, "UserId": "owner(1234)", "Command": str(wrapper),
                            "WorkDir": str(source), "Account": "${PDE_GAV_SLURM_ACCOUNT}", "QOS": "normal",
                            "NumCPUs": "24", "CPUs/Task": "24", "NumNodes": "1-1", "NumTasks": "1",
                            "TimeLimit": site["time"], "TresPerNode": "gres/gpu:l40s:4",
                            "ReqTRES": f"cpu=24,mem={site['memory']},node=1,gres/gpu=4,gres/gpu:l40s=4",
                            "JobState": "PENDING", "Partition": reconcile.TARGET,
                            **{name: "2026-10-02T21:04:51" for name in reconcile.TIMESTAMPS}}
    plan = write_json(directory / "plan.json", {"schema": "pde_gav_six_gpu_submission_v1",
                      "source_root": str(source), "source_revision": "a" * 40, "entries": entries})
    write_json(directory / "submissions.json", {"schema": "pde_gav_six_gpu_submission_receipt_v1",
               "status": "all_six_submitted", "plan_sha256": reconcile.digest(plan), "jobs": submitted})
    return plan, tmp_path / "scheduling", FakeScheduler(jobs), sites


def sweep(deployment, **kwargs):
    plan, receipts, scheduler, _ = deployment
    return reconcile.sweep(plan, receipts, scheduler=scheduler, uid=1234, **kwargs)


def add_successor(deployment):
    plan, _, scheduler, sites = deployment
    site = sites["weak-sol"]
    entry = next(row for row in json.loads(plan.read_text())["entries"] if row["name"] == "weak-sol")
    initial = next(identifier for identifier, job in scheduler.jobs.items()
                   if job["Command"] == site["slurm_script"])
    comment = f"ns-continuation:{'b' * 32}:{'c' * 32}"
    spec = {"slurm_script": site["slurm_script"], "script_sha256": entry["script_sha256"],
            "script_args": [], "gpu_count": 4, "account": "${PDE_GAV_SLURM_ACCOUNT}", "qos": "normal",
            "partition": reconcile.BASE, "wall_seconds": 43200, "cpus": 24, "memory": "128G"}
    state = {"schema": "ns_slurm_continuation_v1", "run_root": site["run_root"], "run_id": "b" * 32,
             "max_segments": 12, "segments": [{"job_id": initial, "segment_index": 1}],
             "submission_spec": spec, "submissions": [{"job_id": "200", "status": "submitted",
                    "segment_index": 2, "intent_id": "c" * 32, "comment": comment}]}
    path = write_json(reconcile.Path(site["run_root"]) / "slurm_continuation.json", state)
    scheduler.jobs["200"] = {**scheduler.jobs[initial], "JobId": "200", "Comment": comment,
                              "Partition": reconcile.BASE}
    return path, state, initial


def test_default_dry_run_and_idempotent_apply(deployment):
    _, receipts, scheduler, _ = deployment
    scheduler.jobs["100"]["Partition"] = reconcile.BASE
    result = sweep(deployment)
    assert result["jobs"][0]["status"] == "would_expand"
    assert scheduler.updates == []
    result = sweep(deployment, apply=True)
    assert result["jobs"][0]["status"] == "expanded"
    assert scheduler.updates == ["100"]
    assert result["jobs"][0]["before"]["SubmitTime"] == result["jobs"][0]["after"]["SubmitTime"]
    assert (receipts / "job-100-baseline.json").exists()
    assert all(row["status"] == "already_expanded" for row in sweep(deployment, apply=True)["jobs"])
    assert scheduler.updates == ["100"]


def test_accepted_successor_only_and_scientific_state_unchanged(deployment):
    path, state, initial = add_successor(deployment)
    scheduler = deployment[2]
    scheduler.jobs[initial]["JobState"] = "RUNNING"
    scheduler.jobs["999"] = {**scheduler.jobs["200"], "JobId": "999"}
    original = path.read_bytes()
    result = sweep(deployment, apply=True)
    assert scheduler.updates == ["200"]
    assert path.read_bytes() == original
    assert next(row for row in result["jobs"] if row.get("job_id") == initial)["status"] == "skipped_not_pending"
    assert "999" not in {row.get("job_id") for row in result["jobs"]}


@pytest.mark.parametrize("field,value", [
    ("UserId", "other(5678)"), ("Account", "other"), ("QOS", "interac"),
    ("NumCPUs", "16"), ("CPUs/Task", "16"), ("NumNodes", "2"), ("NumTasks", "2"),
    ("TimeLimit", "03:00:00"), ("TresPerNode", "gres/gpu:h100:4"),
    ("ReqTRES", "cpu=24,mem=32G,node=1,gres/gpu=4,gres/gpu:l40s=4"),
    ("JobId", "999"), ("Command", "/other/run.slurm"), ("WorkDir", "/other/source"),
    ("Partition", "gpubase_l40s_b3"),
])
def test_wrong_live_metadata_fails_closed_per_run(deployment, field, value):
    scheduler = deployment[2]
    scheduler.jobs["100"]["Partition"] = reconcile.BASE
    scheduler.jobs["100"][field] = value
    scheduler.jobs["101"]["Partition"] = reconcile.BASE
    result = sweep(deployment, apply=True)
    assert result["jobs"][0]["status"] == "error"
    assert scheduler.updates == ["101"]


@pytest.mark.parametrize("change", ["comment", "run_root", "spec", "initial", "duplicate"])
def test_bad_successor_identity_never_mutates(deployment, change):
    path, state, initial = add_successor(deployment)
    if change == "comment":
        deployment[2].jobs["200"]["Comment"] = "unrelated"
    elif change == "run_root":
        state["run_root"] += "-other"
    elif change == "spec":
        state["submission_spec"]["cpus"] = 16
    elif change == "initial":
        state["segments"][0]["job_id"] = "999"
    else:
        state["submissions"].append(deepcopy(state["submissions"][0]))
    write_json(path, state)
    result = sweep(deployment, apply=True)
    assert any(row["status"] == "error" for row in result["jobs"])
    assert deployment[2].updates == []


def test_unknown_submission_is_not_accepted(deployment):
    path, state, _ = add_successor(deployment)
    state["submissions"][0].update(status="submission_unknown", job_id=None)
    write_json(path, state)
    assert all(row["status"] == "already_expanded" for row in sweep(deployment, apply=True)["jobs"])
    assert deployment[2].updates == []


def test_transient_accrue_is_deferred_then_checked(deployment):
    scheduler = deployment[2]
    scheduler.jobs["100"]["Partition"] = reconcile.BASE
    scheduler.transient_accrue = True
    result = sweep(deployment, apply=True)
    assert result["jobs"][0]["deferred_timestamp_checks"] == ["AccrueTime"]
    result = sweep(deployment, apply=True)
    assert result["jobs"][0]["status"] == "already_expanded"
    scheduler.jobs["100"]["AccrueTime"] = "2026-10-02T21:04:51"
    assert sweep(deployment)["jobs"][0]["deferred_timestamp_checks"] == []
    scheduler.jobs["100"]["SubmitTime"] = "2026-10-03T12:00:00"
    assert sweep(deployment)["jobs"][0]["status"] == "error"
    assert scheduler.updates == ["100"]


def test_race_does_not_retry_or_requeue(deployment):
    scheduler = deployment[2]
    scheduler.jobs["100"]["Partition"] = reconcile.BASE
    scheduler.race = True
    result = sweep(deployment, apply=True)
    assert result["jobs"][0]["status"] == "race_no_retry"
    assert scheduler.updates == ["100"]


def test_historical_job_missing_is_harmless(deployment):
    del deployment[2].jobs["100"]
    assert sweep(deployment)["jobs"][0]["status"] == "not_in_scheduler"


def test_already_running_on_selected_b3_is_skipped(deployment):
    deployment[2].jobs["100"].update(JobState="RUNNING", Partition="gpubase_l40s_b3")
    assert sweep(deployment, apply=True)["jobs"][0]["status"] == "skipped_not_pending"
    assert deployment[2].updates == []


def test_successful_update_racing_start_on_b3_is_not_an_error(deployment):
    scheduler = deployment[2]
    scheduler.jobs["100"]["Partition"] = reconcile.BASE
    scheduler.start_after_expansion = True
    assert sweep(deployment, apply=True)["jobs"][0]["status"] == "expanded_then_started"
    assert scheduler.updates == ["100"]


def test_scontrol_parser_preserves_embedded_tres_and_slash_fields(monkeypatch):
    from types import SimpleNamespace
    raw = ("JobId=100 UserId=user(1234) AllocNode:Sid=login01:123 "
           "NumCPUs=24 CPUs/Task=24 ReqTRES=cpu=24,mem=128G,node=1,gres/gpu=4,gres/gpu:l40s=4 "
           "Command=/test/run.slurm TresPerNode=gres/gpu:l40s:4")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=raw + "\n", stderr="")

    monkeypatch.setattr(reconcile.subprocess, "run", run)
    result = reconcile.Scheduler().show("100")
    assert result["CPUs/Task"] == "24"
    assert result["AllocNode:Sid"] == "login01:123"
    assert result["ReqTRES"] == "cpu=24,mem=128G,node=1,gres/gpu=4,gres/gpu:l40s=4"
    assert result["_raw"] == raw
    assert calls == [["scontrol", "-o", "show", "job", "100"]]


def test_receipts_cannot_modify_frozen_or_scientific_state(deployment):
    plan, _, scheduler, sites = deployment
    for path in (plan.parent / "receipts", reconcile.Path(sites["weak-sol"]["run_root"]) / "receipts"):
        with pytest.raises(ValueError, match="outside"):
            reconcile.sweep(plan, path, scheduler=scheduler, uid=1234, apply=True)
    assert scheduler.updates == []


def test_duplicate_initial_jobs_rejected_before_scheduler(deployment):
    plan = deployment[0]
    path = plan.parent / "submissions.json"
    submitted = json.loads(path.read_text())
    submitted["jobs"][1]["job_id"] = submitted["jobs"][0]["job_id"]
    write_json(path, submitted)
    with pytest.raises(ValueError, match="duplicate"):
        sweep(deployment, apply=True)
    assert deployment[2].updates == []
