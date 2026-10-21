"""Scientific transport stays in memory/local scratch without losing failures."""
from pathlib import Path
import json
import os

import numpy as np
import pytest

from pde_agent_verifier_solver.engine import (
    CandidateExecutionError,
    PythonCandidateCodec,
    PythonTaskWorkerSpec,
    SubprocessCandidateExecutionService,
)
from pde_agent_verifier_solver.engine.agent_wire import AgentWireCodec
from pde_agent_verifier_solver.engine.candidate_execution import (
    CandidateExecutionInfrastructureError,
    CandidatePublicSessionClient,
    _CandidateOracleHost,
    _candidate_call_workspace,
)
from pde_agent_verifier_solver.plugins.ns import NSSolverTask
from pde_agent_verifier_solver.sdk import (
    EvaluationPhase, EvaluationPlan, TaskKind, write_candidate_bundle_manifest,
)
from tests.test_candidate_execution import (
    _candidate_bundle, _case, _evaluation_settings, _query,
)


def _array_execution(tmp_path):
    source = _candidate_bundle(tmp_path / "source")
    submission = source / "submission.py"
    submission.write_text(submission.read_text().replace(
        "value = open_public_case_session(case).marker(value)",
        "value = float(np.asarray(open_public_case_session(case).marker(\n"
        "            np.full(int(case.public_data['array_elements']), value, dtype=np.float64)\n"
        "        )).mean())",
    ))
    write_candidate_bundle_manifest(source, files=("submission.py",), overwrite=True)
    artifact = PythonCandidateCodec(task_kind=TaskKind.SOLVER).bind_existing(
        source, storage_root=tmp_path / "sealed", artifact_id="array-candidate",
        name="Array Candidate", summary="array transport", usage="prepare then evaluate",
    )
    query = _query()
    settings = _evaluation_settings(query)
    evaluation = NSSolverTask().create_evaluation(settings)
    service = SubprocessCandidateExecutionService(
        tmp_path / "execution", task=PythonTaskWorkerSpec.from_task_type(NSSolverTask),
        evaluation_settings=settings, max_workers=1, case_hard_wall_seconds=30,
    )
    plan = EvaluationPlan(phase=EvaluationPhase.OFFICIAL, query=query, settings=settings)
    return artifact, evaluation, service, plan


def test_routine_96k_oracle_array_uses_no_sidecars_and_cleans_local_scratch(tmp_path, monkeypatch):
    scratch = tmp_path / "node-local"
    scratch.mkdir()
    monkeypatch.setenv("SLURM_TMPDIR", str(scratch))
    artifact, evaluation, service, plan = _array_execution(tmp_path)
    case = _case("array-inline", value=3.0, use_oracle=True, array_elements=12_288)
    observed = []

    def invoke(case_id, method, args, kwargs):
        assert args[0].nbytes == 96 * 1024
        assert not list(scratch.rglob("*.npy"))
        roots = list(scratch.glob("pde-candidate-*"))
        assert len(roots) == 1
        assert roots[0].stat().st_mode & 0o777 == 0o700
        observed.append(case_id)
        return args[0] + 4.0

    result = service.execute_cases(
        artifact, (case,), (plan,), evaluation, public_session_methods=("marker",),
        invoke_public_session=invoke, close_public_session=lambda _case_id: None,
    )
    assert observed == [case.case_id]
    assert result[0].prediction.fields["velocity_x"] == (7.0, 7.0)
    assert not list(scratch.iterdir())
    assert not list(service.root.glob("evaluation-*"))
    assert Path(service.audits[0].workspace).is_relative_to(scratch)
    persisted = json.loads((service.root / "candidate_execution_audits.jsonl").read_text())
    assert persisted["status"] == "completed"


def test_failed_local_worker_preserves_request_logs_and_large_oracle_sidecars(tmp_path, monkeypatch):
    scratch = tmp_path / "node-local"
    scratch.mkdir()
    monkeypatch.setenv("SLURM_TMPDIR", str(scratch))
    artifact, evaluation, service, plan = _array_execution(tmp_path)
    case = _case("array-failure", value=3.0, use_oracle=True,
                 array_elements=262_144, fail_after_oracle=True)
    with pytest.raises(CandidateExecutionError, match="retained in"):
        service.execute_cases(
            artifact, (case,), (plan,), evaluation, public_session_methods=("marker",),
            invoke_public_session=lambda _case, _method, args, _kwargs: args[0] + 4,
            close_public_session=lambda _case_id: None,
        )
    audit = service.audits[0]
    workspace = Path(audit.workspace)
    assert workspace.is_relative_to(service.root)
    assert workspace.is_dir()
    assert (workspace / "request.json").is_file()
    assert Path(audit.stdout_path).is_file()
    assert Path(audit.stderr_path).is_file()
    response = json.loads(Path(audit.response_path).read_text())
    assert "intentional failure after public oracle" in response["error_message"]
    sidecars = list((workspace.parent / "oracle-arrays").glob("*.npy"))
    assert len(sidecars) == 2
    values = sorted(float(np.load(path, allow_pickle=False).mean()) for path in sidecars)
    assert values == [3.0, 7.0]
    locations = json.loads((workspace.parent / "workspace_locations.json").read_text())
    assert Path(locations["runtime_workspace"]).is_relative_to(scratch)
    assert locations["retained_workspace"] == str(workspace.parent)
    assert not list(scratch.iterdir()), "archived ordinary failures must not fill local scratch"


def test_unconfirmed_process_cleanup_keeps_original_failure_scratch(tmp_path, monkeypatch):
    from pde_agent_verifier_solver.engine import candidate_execution

    scratch = tmp_path / "node-local"
    scratch.mkdir()
    monkeypatch.setenv("SLURM_TMPDIR", str(scratch))
    artifact, evaluation, service, plan = _array_execution(tmp_path)
    case = _case("unconfirmed-cleanup", value=3.0)
    # The child exits normally; simulate inability to confirm its process-group
    # cleanup so no unsafe background process is needed for this regression.
    monkeypatch.setattr(candidate_execution, "_cleanup_process_group", lambda _pid: False)
    with pytest.raises(CandidateExecutionError, match="process group cleanup"):
        service.execute_cases(artifact, (case,), (plan,), evaluation)
    audit = service.audits[0]
    workspace = Path(audit.workspace)
    assert workspace.is_relative_to(service.root)
    assert (workspace / "request.json").is_file()
    locations = json.loads((workspace.parent / "workspace_locations.json").read_text())
    runtime = Path(locations["runtime_workspace"])
    assert runtime.is_relative_to(scratch)
    assert runtime.is_dir(), "unconfirmed cleanup must preserve the original workspace"
    assert (runtime / workspace.name / "request.json").is_file()


def test_ephemeral_codec_skips_fsync_but_default_codec_remains_durable(tmp_path, monkeypatch):
    synced = []
    monkeypatch.setattr(os, "fsync", synced.append)
    data = np.arange(256, dtype=np.float64)
    ephemeral = AgentWireCodec(tmp_path / "ephemeral", inline_array_bytes=1,
                               durable_sidecars=False)
    np.testing.assert_array_equal(ephemeral.decode(ephemeral.encode(data)), data)
    assert synced == []
    durable = AgentWireCodec(tmp_path / "durable", inline_array_bytes=1)
    np.testing.assert_array_equal(durable.decode(durable.encode(data)), data)
    assert len(synced) == 2


def test_oracle_manifest_negotiates_bounded_inline_limit_and_legacy_default(tmp_path):
    host = _CandidateOracleHost(
        tmp_path / "arrays", case_ids=("case",), methods=(), invoke=None,
        close=None,
    )
    try:
        manifest = dict(host.manifest_for_case("case"))
        assert manifest["inline_array_bytes"] == 1024 * 1024
        assert CandidatePublicSessionClient(manifest).codec.inline_array_bytes == 1024 * 1024
        manifest.pop("inline_array_bytes")
        manifest.pop("oracle_inline_array_bytes")
        manifest.pop("oracle_trusted_results")
        assert CandidatePublicSessionClient(manifest).codec.inline_array_bytes == 64 * 1024
        for limit in (True, 0, "1024", 32 * 1024 * 1024 + 1):
            manifest["inline_array_bytes"] = limit
            with pytest.raises(CandidateExecutionInfrastructureError, match="inline array limit"):
                CandidatePublicSessionClient(manifest)
    finally:
        host.close()


def test_missing_scheduler_scratch_falls_back_to_persistent_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_TMPDIR", str(tmp_path / "unavailable"))
    retained = tmp_path / "evaluation-test"
    assert _candidate_call_workspace(retained) == retained
    assert retained.is_dir()
