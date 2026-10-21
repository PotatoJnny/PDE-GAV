"""Synthetic loopback tests for bounded, opt-in numerical Oracle transport."""
from contextlib import contextmanager
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

import numpy as np
import pytest

from pde_agent_verifier_solver.engine import candidate_execution as execution
from pde_agent_verifier_solver.engine import agent_wire as wire
from pde_agent_verifier_solver.sdk import (
    AnswerQuery, AnswerView, CandidateEvaluationOutput, Case, CaseSource,
    CaseVisibility, EvaluationPhase, EvaluationPlan, QueryKind,
)


FAST = {"oracle_inline_array_bytes": 8 * 1024 * 1024, "oracle_trusted_results": True}


@contextmanager
def running_host(tmp_path, invoke, **options):
    host = execution._CandidateOracleHost(
        tmp_path / "oracle-arrays", case_ids=("case-a", "case-b"),
        methods=("source",), invoke=invoke, close=lambda case_id: None, **options)
    host.start()
    try:
        client = execution.CandidatePublicSessionClient(host.manifest_for_case("case-a"))
        yield host, client
    finally:
        host.close()


def test_typical_points_and_result_are_bitwise_identical_without_file_io(tmp_path, monkeypatch):
    points = np.arange(8192 * 5, dtype=np.float64).reshape(8192, 5) / 17
    points[0, 0] = -0.0
    expected = points[:, 0].copy()
    expected[1] = np.nextafter(0., 1.)
    seen = []

    def invoke(case, method, args, kwargs):
        assert case == "case-a" and method == "source" and kwargs == {}
        seen.append(args[0].tobytes())
        return expected

    with running_host(tmp_path, invoke, **FAST) as (host, client):
        manifest = host.manifest_for_case("case-a")
        assert manifest["oracle_inline_array_bytes"] == 8 * 1024 * 1024
        assert host.codec.validate_array_finiteness is True
        assert client.codec.validate_array_finiteness is True
        assert host.result_codec.validate_array_finiteness is False
        assert client.result_codec.validate_array_finiteness is False
        with monkeypatch.context() as patch:
            def forbidden(*args, **kwargs):
                raise AssertionError("typical Oracle request must not touch sidecar files")
            patch.setattr(wire.AgentWireCodec, "_publish_array", forbidden)
            patch.setattr(wire.os, "open", forbidden)
            patch.setattr(wire.os, "fsync", forbidden)
            patch.setattr(np, "save", forbidden)
            actual = client.invoke("case-a", "source", (points,), {})
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert actual.tobytes() == expected.tobytes()
        assert seen == [points.tobytes()]
        assert not tuple((tmp_path / "oracle-arrays").iterdir())


@pytest.mark.parametrize("trusted,expected_scans", [(False, 4), (True, 2)])
def test_only_result_array_finite_scans_are_skipped(tmp_path, monkeypatch, trusted, expected_scans):
    points = np.zeros((8192, 5), dtype=np.float64)
    result = np.arange(8192, dtype=np.float64)
    with running_host(tmp_path, lambda *args: result,
                      oracle_inline_array_bytes=8 * 1024 * 1024,
                      oracle_trusted_results=trusted) as (_, client):
        scans = []
        original = np.isfinite
        with monkeypatch.context() as patch:
            def counted(value, *args, **kwargs):
                scans.append(np.shape(value))
                return original(value, *args, **kwargs)
            patch.setattr(np, "isfinite", counted)
            actual = client.invoke("case-a", "source", (points,), {})
        assert len(scans) == expected_scans
        assert scans.count((8192, 5)) == 2
        assert actual.tobytes() == result.tobytes()


def test_explicit_legacy_transport_uses_sidecar_and_rejects_nonfinite_results(tmp_path):
    points = np.zeros((8192, 5), dtype=np.float64)
    with running_host(tmp_path, lambda *args: np.ones(8192),
                      oracle_inline_array_bytes=64 * 1024,
                      oracle_trusted_results=False) as (host, client):
        assert host.codec.inline_array_bytes == 64 * 1024
        assert host.result_codec is host.codec and client.result_codec is client.codec
        client.invoke("case-a", "source", (points,), {})
        assert len(tuple((tmp_path / "oracle-arrays").glob("*.npy"))) == 1
        # Old manifests lacking the new optional policy retain old defaults.
        manifest = dict(host.manifest_for_case("case-a"))
        del manifest["oracle_inline_array_bytes"], manifest["oracle_trusted_results"]
        del manifest["inline_array_bytes"]
        legacy_client = execution.CandidatePublicSessionClient(manifest)
        assert legacy_client.codec.inline_array_bytes == 64 * 1024
        host.state.invoke = lambda *args: np.array([np.nan])
        with pytest.raises(execution.CandidateExecutionError):
            client.invoke("case-a", "source", (points,), {})


def _params(host, points):
    return {"case_id": "case-a", "method": "source", "args": host.codec.encode((points,)),
            "kwargs": host.codec.encode({})}


def test_trusted_result_does_not_weaken_auth_method_request_or_policy_checks(tmp_path):
    calls = []
    with running_host(tmp_path, lambda *args: calls.append(args) or np.array([np.nan]), **FAST) as (host, client):
        params = _params(host, np.zeros((2, 5)))
        token = host.manifest_for_case("case-a")["token"]
        with pytest.raises(PermissionError):
            host.state.submit("not-a-token", "bad-token", params)
        with pytest.raises(PermissionError):
            host.state.submit(token, "wrong-case", {**params, "case_id": "case-b"})
        with pytest.raises(PermissionError):
            host.state.submit(token, "hidden-method", {**params, "method": "hidden_truth"})
        with pytest.raises(ValueError, match="fields"):
            host.state.submit(token, "extra-policy", {**params, "oracle_trusted_results": True})
        # A hostile client can encode unchecked arrays; host request codec
        # still validates them before invoking the trusted numerical function.
        unchecked = host.result_codec.encode((np.array([[np.nan] * 5]),))
        with pytest.raises(wire.AgentWireCodecError, match="finite"):
            host.state.submit(token, "nonfinite-args", {**params, "args": unchecked})
        with pytest.raises(wire.AgentWireCodecError, match="finite"):
            client.invoke("case-a", "source", (np.array([[np.nan] * 5]),), {})
        assert calls == []
        # Transport deliberately trusts server numerical result arrays; their
        # scientific validation remains owned by the numerical implementation.
        value = client.invoke("case-a", "source", (np.zeros((2, 5)),), {})
        assert np.isnan(value[0]) and len(calls) == 1


def test_trusted_codec_retains_shape_dtype_and_array_size_checks():
    codec = wire.AgentWireCodec(inline_array_bytes=16, max_sidecar_bytes=32,
                                validate_array_finiteness=False)
    encoded = codec.encode(np.array([1., 2.]))
    with pytest.raises(wire.AgentWireCodecError, match="shape"):
        codec.decode({**encoded, "shape": [3]})
    with pytest.raises(wire.AgentWireCodecError, match="objects"):
        codec.decode({**encoded, "dtype": "O"})
    with pytest.raises(wire.AgentWireCodecError, match="objects"):
        codec.encode(np.array([object()]))
    large = wire.AgentWireCodec().encode(np.arange(3, dtype=np.float64))
    with pytest.raises(wire.AgentWireCodecError, match="size limit"):
        codec.decode(large)
    with pytest.raises(wire.AgentWireCodecError, match="inline limit"):
        codec.encode(np.arange(3, dtype=np.float64))


def test_rpc_total_request_and_response_limits_remain_enforced(tmp_path, monkeypatch):
    with running_host(tmp_path, lambda *args: "x" * 4096, **FAST) as (host, client):
        # Lower only the envelope in this test; no large allocation is needed.
        monkeypatch.setattr(execution, "_MAX_RPC_BYTES", 2048)
        with pytest.raises(execution.CandidateExecutionError, match="request is too large"):
            client.invoke("case-a", "source", ("x" * 4096,), {})
        manifest = host.manifest_for_case("case-a")
        request = Request(str(manifest["endpoint"]), data=b"{}", method="POST",
            headers={"Content-Length": "2049", execution._TOKEN_HEADER: str(manifest["token"])})
        with pytest.raises(HTTPError) as caught:
            build_opener(ProxyHandler({})).open(request, timeout=5.)
        assert caught.value.code == 413
        with pytest.raises(execution.CandidateExecutionError):
            client.invoke("case-a", "source", (), {})


@pytest.mark.parametrize("options", [
    {"oracle_inline_array_bytes": True}, {"oracle_inline_array_bytes": 0},
    {"oracle_inline_array_bytes": 64 * 1024 * 1024}, {"oracle_trusted_results": 1},
])
def test_service_rejects_invalid_transport_options(tmp_path, options):
    with pytest.raises((TypeError, ValueError), match="oracle_"):
        execution.SubprocessCandidateExecutionService(tmp_path / "service",
            task=execution.PythonTaskWorkerSpec("test", "Task"), evaluation_settings={},
            max_workers=1, case_hard_wall_seconds=10., **options)


def test_service_propagates_opt_in_to_actual_host(tmp_path, monkeypatch):
    service = execution.SubprocessCandidateExecutionService(tmp_path / "service",
        task=execution.PythonTaskWorkerSpec("test", "Task"), evaluation_settings={},
        max_workers=1, case_hard_wall_seconds=10., **FAST)
    query = AnswerQuery.at_points(fields=("u",), points=({"x": 0.},))
    case = Case("case-a", CaseSource.GENERATED, CaseVisibility.HIDDEN, {})
    plan = EvaluationPlan(EvaluationPhase.OFFICIAL, query)
    artifact = SimpleNamespace(artifact_id="candidate", program=object.__new__(execution.PythonCandidateProgram))
    evaluator = SimpleNamespace(supports_split_candidate_execution=True, program_id="test")
    seen = []

    def execute_one(*args, **kwargs):
        host = args[6]
        manifest = host.manifest_for_case(case.case_id)
        seen.append((manifest["oracle_inline_array_bytes"], manifest["oracle_trusted_results"]))
        client = execution.CandidatePublicSessionClient(manifest)
        value = client.invoke(case.case_id, "source", (np.zeros((8192, 5)),), {})
        return CandidateEvaluationOutput(AnswerView(QueryKind.POINTS, {"u": (float(value[0]),)}, points=query.points, shape=(1,)))

    monkeypatch.setattr(service, "_execute_one", execute_one)
    try:
        results = service.execute_cases(artifact, (case,), (plan,), evaluator,
            public_session_methods=("source",), invoke_public_session=lambda *args: np.array([2.]),
            close_public_session=lambda identifier: None)
        assert results[0].prediction.fields["u"] == (2.,)
        assert seen == [(8 * 1024 * 1024, True)]
    finally:
        service._executor.shutdown(wait=True)
