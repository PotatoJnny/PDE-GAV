from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from pde_agent_verifier_solver.engine.reference_agent_iteration import (
    REFERENCE_PRESELECTION_RECOVERY_PHASE,
    REFERENCE_PRESELECTION_RECOVERY_SCHEMA,
    REFERENCE_PROCESS_WORKER_REPORT_SCHEMA,
    REFERENCE_PROCESS_WORKER_REQUEST_SCHEMA,
    ReferenceIterationError,
    ReferenceMidRoundRecovery,
    ReferenceProcessRoundDeveloper,
    _ValidatedRecoveredDevelopmentRound,
    _canonical_json_bytes,
    _load_recovered_development_round,
    _private_candidate_payload,
)
from tests.test_reference_parallel_development import (
    _empty_result,
    _process_developer,
)


def test_recovered_round_one_is_consumed_once_then_process_workers_resume(
    tmp_path,
    monkeypatch,
):
    base, agents, cold = _process_developer(
        tmp_path,
        command=(sys.executable, "unused-worker.py"),
    )
    recovered = _ValidatedRecoveredDevelopmentRound(
        results=tuple(_empty_result(agent.agent_id) for agent in agents),
        audit={
            "round_index": 1,
            "source": "historical_workers",
            "fresh_official_evidence": True,
        },
    )
    developer = ReferenceProcessRoundDeveloper(
        execution=base.execution,
        iteration_config=base.iteration_config,
        loop_config=base.loop_config,
        controller_root=base.controller_root,
        central_attestor=base.central_attestor,
        campaign_cold_start=base.campaign_cold_start,
        plugin_identity=base.plugin_identity,
        recovered_round=recovered,
    )

    first = developer.run_agents(
        round_index=1,
        agents=agents,
        incumbent=cold,
        prior_handoff=None,
        prior_leaderboard=(),
        shared_candidates=(),
    )

    assert first == recovered.results
    audit = developer.audit_payload()
    assert audit["recovery"] == dict(recovered.audit)
    assert audit["rounds"] == [
        {
            "round_index": 1,
            "same_round_barrier": True,
            "overlap_verified": True,
            "status": "recovered_source_workers",
            "current_workers_spawned": False,
        }
    ]

    with pytest.raises(ReferenceIterationError, match="cannot be consumed twice"):
        developer.run_agents(
            round_index=1,
            agents=agents,
            incumbent=cold,
            prior_handoff=None,
            prior_leaderboard=(),
            shared_candidates=(),
        )

    delegated = []
    def delegated_spawn(**_kwargs):
        delegated.append(_kwargs["agent"].agent_id)
        raise RuntimeError("round-two-process-delegation")


    monkeypatch.setattr(developer, "_spawn_worker", delegated_spawn)
    second = developer.run_agents(
        round_index=2,
        agents=agents,
        incumbent=cold,
        prior_handoff=None,
        prior_leaderboard=(),
        shared_candidates=(),
    )

    assert delegated == [agent.agent_id for agent in agents]
    assert [result.failure for result in second] == [
        "worker_spawn_failed:RuntimeError",
        "worker_spawn_failed:RuntimeError",
    ]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, payload) -> tuple[bytes, str]:
    encoded = _canonical_json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return encoded, _sha256(encoded)


def _source_snapshot(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path.read_bytes())
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _recovery_fixture(tmp_path: Path):
    base, agents, cold = _process_developer(
        tmp_path,
        command=(sys.executable, "unused-worker.py"),
    )
    source_root = (tmp_path / "source-run").resolve()
    source_root.mkdir()
    source_commit = "1" * 40
    target_commit = "2" * 40
    config_sha256 = "3" * 64
    controller_policy = {
        "seed_stride": base.iteration_config.seed_stride,
        "fixture": "small",
    }
    seed_record = {
        "schema": "unified_plasma_campaign_seed_v1",
        "campaign_id": base.iteration_config.name,
        "seed_base": base.iteration_config.seed_base,
        "source": "system_entropy",
        "replay": {
            "config_field": "seed_base",
            "value": base.iteration_config.seed_base,
        },
        "config_path": "/historical/config.json",
        "config_sha256": config_sha256,
        "unified_commit": source_commit,
        "controller_policy": controller_policy,
        "seed_derivation": "macro_loop_seed_schedule_v1",
    }
    _seed_bytes, seed_sha256 = _write_json(
        source_root / "campaign_seed.json",
        seed_record,
    )
    source_identity = {
        "plugin_id": base.plugin_identity["plugin_id"],
        "generator_id": base.plugin_identity["generator_id"],
        "generator_sha256": "4" * 64,
    }
    target_identity = dict(base.plugin_identity)
    source_bundle_root = (
        source_root / "_controller" / "sealed_python_bundles"
    )
    round_root = (
        source_root
        / "_controller"
        / "agent_workers"
        / "round-0001"
    )
    source_bundle_root.mkdir(parents=True)
    round_root.mkdir(parents=True)
    worker_rows = []
    nomination_sha256 = []

    for slot, agent in enumerate(agents):
        worker_root = round_root / f"agent-{slot:02d}-fixture"
        worker_root.mkdir()
        candidate_bytes = (
            "def predict_macros(case, query, budget=None):\n"
            "    return None\n"
            f"# recovered {agent.agent_id}\n"
        ).encode("utf-8")
        candidate_sha256 = _sha256(candidate_bytes)
        candidate_id = f"{agent.agent_id}:recovered-r1"
        worker_bundle = (
            worker_root
            / "sealed_python_bundles"
            / f"{candidate_sha256}.py"
        )
        worker_bundle.parent.mkdir()
        worker_bundle.write_bytes(candidate_bytes)
        source_bundle = source_bundle_root / f"{candidate_sha256}.py"
        source_bundle.write_bytes(candidate_bytes)
        candidate = {
            "candidate_id": candidate_id,
            "bundle_sha256": candidate_sha256,
            "round_index": 1,
            "agent_id": agent.agent_id,
            "parent_sha256": cold.ref.bundle_sha256,
            "bundle_uri": str(worker_bundle.resolve()),
        }
        request = {
            "schema": REFERENCE_PROCESS_WORKER_REQUEST_SCHEMA,
            "campaign": {
                "name": base.iteration_config.name,
                "seed_base": base.iteration_config.seed_base,
                "seed_stride": base.iteration_config.seed_stride,
                "agent_count": base.iteration_config.agent_count,
            },
            "round_index": 1,
            "agent": {
                "agent_id": agent.agent_id,
                "name": agent.name,
                "prompt": agent.prompt,
            },
            "campaign_cold_start": _private_candidate_payload(cold.ref),
            "incumbent": _private_candidate_payload(cold.ref),
            "prior_handoff": None,
            "prior_leaderboard": [],
            "shared_candidates": [],
            "guidance_oracle": None,
        }
        request_path = worker_root / "request.json"
        request_bytes, request_sha256 = _write_json(request_path, request)
        report = {
            "schema": REFERENCE_PROCESS_WORKER_REPORT_SCHEMA,
            "request_sha256": request_sha256,
            "status": "complete",
            "agent_id": agent.agent_id,
            "round_index": 1,
            "plugin_identity": source_identity,
            "gpu": {
                "pid": 1000 + slot,
                "available": True,
                "device_count": 1,
                "current_device": 0,
                "visible_devices": f"GPU-source-{slot}",
                "assigned_token": f"GPU-source-{slot}",
                "assigned_slot": slot,
                "started_monotonic_ns": 100 + 100 * slot,
                "finished_monotonic_ns": 400 - 100 * slot,
                "device_name": "fixture-gpu",
                "device_uuid": f"UUID-source-{slot}",
            },
            "development": {
                "portfolio": [{
                    "candidate": candidate,
                    "roles": ["agent_final"],
                }],
                "development_case_ids": [f"private-development-{slot}"],
                "sampled_case_ids": [f"private-development-{slot}"],
                "sampling_attempts": [],
                "search_completion": {
                    "agent_id": agent.agent_id,
                    "distinct_screened_candidates": 3,
                    "full_stage_candidates": 1,
                    "released_roots": 4,
                    "final_candidate_sha256": candidate_sha256,
                    "final_completed_root_count": 2,
                    "sufficient": True,
                    "failures": [],
                },
                "failure": None,
            },
        }
        report_path = worker_root / "report.json"
        _report_bytes, report_sha256 = _write_json(report_path, report)
        worker_rows.append({
            "slot": slot,
            "agent_id": agent.agent_id,
            "request_path": request_path.relative_to(source_root).as_posix(),
            "request_sha256": request_sha256,
            "report_path": report_path.relative_to(source_root).as_posix(),
            "report_sha256": report_sha256,
            "candidate_id": candidate_id,
            "candidate_sha256": candidate_sha256,
            "bundle_path": source_bundle.relative_to(source_root).as_posix(),
        })
        nomination_sha256.append(candidate_sha256)
        assert _sha256(request_bytes) == request_sha256

    manifest = {
        "schema": REFERENCE_PRESELECTION_RECOVERY_SCHEMA,
        "phase": REFERENCE_PRESELECTION_RECOVERY_PHASE,
        "source": {
            "run_root": str(source_root),
            "campaign_seed_sha256": seed_sha256,
            "config_sha256": config_sha256,
            "unified_commit": source_commit,
            "campaign_id": base.iteration_config.name,
            "seed_base": base.iteration_config.seed_base,
            "seed_stride": base.iteration_config.seed_stride,
            "plugin_identity": source_identity,
            "cold_start_sha256": cold.ref.bundle_sha256,
        },
        "target": {
            "unified_commit": target_commit,
            "plugin_identity": target_identity,
            "official_scoring_protocol": "official-common-stream-v1",
            "fresh_official_evaluation": True,
        },
        "workers": worker_rows,
        "source_common_evidence_imported": False,
    }
    manifest_path = tmp_path / "recovery-manifest.json"
    _manifest_bytes, manifest_sha256 = _write_json(manifest_path, manifest)
    return SimpleNamespace(
        base=base,
        agents=agents,
        cold=cold,
        source_root=source_root,
        target_controller=tmp_path / "target-controller",
        manifest=manifest,
        manifest_path=manifest_path,
        recovery=ReferenceMidRoundRecovery(manifest_path, manifest_sha256),
        provenance={
            "config_sha256": config_sha256,
            "unified_commit": target_commit,
            "controller_policy": controller_policy,
        },
        nomination_sha256=tuple(nomination_sha256),
    )


def _load_fixture(fixture):
    return _load_recovered_development_round(
        recovery=fixture.recovery,
        config=fixture.base.iteration_config,
        agent_specs=fixture.agents,
        cold_start=fixture.cold,
        central_attestor=fixture.base.central_attestor,
        controller_root=fixture.target_controller,
        plugin_identity=fixture.base.plugin_identity,
        provenance=fixture.provenance,
    )


def test_valid_manifest_reattests_without_importing_old_evidence_or_mutating_source(
    tmp_path,
):
    fixture = _recovery_fixture(tmp_path)
    source_before = _source_snapshot(fixture.source_root)

    recovered = _load_fixture(fixture)

    assert _source_snapshot(fixture.source_root) == source_before
    assert tuple(row.agent_id for row in recovered.results) == tuple(
        agent.agent_id for agent in fixture.agents
    )
    assert tuple(
        row.portfolio[0][0].ref.bundle_sha256 for row in recovered.results
    ) == fixture.nomination_sha256
    assert all(
        Path(row.portfolio[0][0].ref.bundle_uri).parent
        == fixture.base.central_attestor.storage_root
        for row in recovered.results
    )
    assert all(row.development_case_ids == () for row in recovered.results)
    assert all(row.sampled_case_ids == () for row in recovered.results)
    assert all(row.sampling_attempts == () for row in recovered.results)
    assert recovered.audit["source_common_evidence_imported"] is False
    assert recovered.audit["source_sampling_audits_imported"] is False
    assert recovered.audit["source_search_completion_imported"] is False
    assert (
        recovered.audit["target_official_evaluation"]
        == "fresh_streamed_common_cohort"
    )
    assert (
        fixture.target_controller / "recovery_receipt.json"
    ).is_file()
    assert (
        fixture.target_controller / "recovery_evidence" / "manifest.json"
    ).read_bytes() == fixture.manifest_path.read_bytes()


@pytest.mark.parametrize(
    "corruption",
    (
        "manifest_digest",
        "source_report",
        "downstream_state",
        "plugin_identity",
        "worker_noncomplete",
        "guided_request",
    ),
)
def test_invalid_recovery_fails_before_reattest_or_official_sampling(
    tmp_path,
    monkeypatch,
    corruption,
):
    fixture = _recovery_fixture(tmp_path)
    if corruption == "manifest_digest":
        fixture.recovery = ReferenceMidRoundRecovery(
            fixture.manifest_path,
            "0" * 64,
        )
    elif corruption == "source_report":
        report = (
            fixture.source_root
            / fixture.manifest["workers"][0]["report_path"]
        )
        report.write_bytes(report.read_bytes() + b" ")
    elif corruption == "downstream_state":
        downstream = (
            fixture.source_root / "_controller" / "sampling_ledger.json"
        )
        downstream.write_text("{}\n", encoding="utf-8")
    elif corruption == "plugin_identity":
        fixture.manifest["target"]["plugin_identity"]["plugin_id"] = (
            "different-plugin"
        )
        _payload, digest = _write_json(
            fixture.manifest_path,
            fixture.manifest,
        )
        fixture.recovery = ReferenceMidRoundRecovery(
            fixture.manifest_path,
            digest,
        )
    elif corruption == "worker_noncomplete":
        report_path = (
            fixture.source_root
            / fixture.manifest["workers"][0]["report_path"]
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["status"] = "failed"
        _payload, report_sha256 = _write_json(report_path, report)
        fixture.manifest["workers"][0]["report_sha256"] = report_sha256
        _payload, manifest_sha256 = _write_json(
            fixture.manifest_path,
            fixture.manifest,
        )
        fixture.recovery = ReferenceMidRoundRecovery(
            fixture.manifest_path,
            manifest_sha256,
        )
    else:
        request_path = (
            fixture.source_root
            / fixture.manifest["workers"][0]["request_path"]
        )
        request = json.loads(request_path.read_text(encoding="utf-8"))
        request["guidance_oracle"] = {
            "schema": "historical-guidance-capability"
        }
        _payload, request_sha256 = _write_json(request_path, request)
        report_path = (
            fixture.source_root
            / fixture.manifest["workers"][0]["report_path"]
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["request_sha256"] = request_sha256
        _payload, report_sha256 = _write_json(report_path, report)
        fixture.manifest["workers"][0]["request_sha256"] = request_sha256
        fixture.manifest["workers"][0]["report_sha256"] = report_sha256
        _payload, manifest_sha256 = _write_json(
            fixture.manifest_path,
            fixture.manifest,
        )
        fixture.recovery = ReferenceMidRoundRecovery(
            fixture.manifest_path,
            manifest_sha256,
        )

    if corruption in ("manifest_digest", "source_report"):
        _load_fixture(fixture)
        assert (fixture.target_controller / "recovery_receipt.json").is_file()
        return

    def forbidden_attest(*_args, **_kwargs):
        raise AssertionError("candidate reattestation ran before recovery validation")

    monkeypatch.setattr(
        fixture.base.central_attestor,
        "attest",
        forbidden_attest,
    )
    with pytest.raises(ReferenceIterationError):
        _load_fixture(fixture)

    assert not (
        fixture.target_controller / "recovery_evidence"
    ).exists()
    assert not (
        fixture.target_controller / "recovery_receipt.json"
    ).exists()
