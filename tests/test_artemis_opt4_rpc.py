"""Real loopback transport tests with synthetic values, not physical validation."""

from contextlib import contextmanager
import base64
import importlib.util
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener
from uuid import uuid4

import numpy as np
import pytest

from pde_agent_verifier_solver.engine import agent_wire as wire
from pde_agent_verifier_solver.engine.candidate_execution import (
    CANDIDATE_ORACLE_REQUEST_SCHEMA, CandidateExecutionInfrastructureError,
    CandidatePublicSessionClient, _CandidateOracleHost, _ORACLE_INLINE_ARRAY_BYTES, _TOKEN_HEADER,
)


@contextmanager
def oracle(tmp_path, invoke):
    host = _CandidateOracleHost(tmp_path / "arrays", case_ids=("one", "two"),
        methods=("query_source", "query_initial_distribution", "echo"),
        invoke=invoke, close=lambda _: None)
    host.start()
    try:
        yield host, CandidatePublicSessionClient(host.manifest_for_case("one"))
    finally:
        host.close()


def forbid_sidecars(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("bounded numerical RPC touched a sidecar or numeric file hash")
    for name in ("_publish_array", "_decode_sidecar_array"):
        monkeypatch.setattr(wire.AgentWireCodec, name, forbidden)
    for name in ("_sha256_regular_file", "_sha256_handle", "_open_bounded_regular", "_fsync_directory"):
        monkeypatch.setattr(wire, name, forbidden)


@pytest.mark.parametrize("method,width", [("query_source", 3), ("query_initial_distribution", 2)])
def test_complete_http_source_and_initial_requests_never_use_sidecars(tmp_path, monkeypatch, method, width):
    received = []
    def invoke(case_id, called, args, kwargs):
        assert case_id == "one" and called == method and kwargs == {}
        points, = args
        received.append(points.copy())
        return points[:, -1].copy()
    with oracle(tmp_path, invoke) as (host, client):
        assert host.codec.inline_array_bytes == client.codec.inline_array_bytes == _ORACLE_INLINE_ARRAY_BYTES
        forbid_sidecars(monkeypatch)
        for iteration in range(3):
            points = np.arange(4096 * width, dtype=np.float64).reshape(4096, width)
            points[:, 0] += iteration / 8
            points = points[::-1]  # Preserve order, including a non-contiguous input.
            result = client.invoke("one", method, (points,), {})
            assert result.shape == (4096,) and result.dtype == np.float64
            assert result.tobytes() == points[:, -1].tobytes()
            assert received[-1].tobytes() == points.tobytes()
        assert len(received) == 3
        assert list((tmp_path / "arrays").iterdir()) == []


def test_larger_http_arrays_keep_existing_sidecar_roundtrip(tmp_path, monkeypatch):
    counts = {"publish": 0, "decode": 0}
    publish, decode = wire.AgentWireCodec._publish_array, wire.AgentWireCodec._decode_sidecar_array
    def counted_publish(self, value):
        counts["publish"] += 1
        return publish(self, value)
    def counted_decode(self, value):
        counts["decode"] += 1
        return decode(self, value)
    monkeypatch.setattr(wire.AgentWireCodec, "_publish_array", counted_publish)
    monkeypatch.setattr(wire.AgentWireCodec, "_decode_sidecar_array", counted_decode)
    rows = _ORACLE_INLINE_ARRAY_BYTES // (3 * np.dtype(np.float64).itemsize) + 1
    points = np.arange(rows * 3, dtype=np.float64).reshape(rows, 3)
    with oracle(tmp_path, lambda case, method, args, kwargs: args[0]) as (_, client):
        result = client.invoke("one", "echo", (points,), {})
    assert result.dtype == points.dtype and result.shape == points.shape
    assert result.tobytes() == points.tobytes()
    assert counts == {"publish": 2, "decode": 2}
    assert len(list((tmp_path / "arrays").glob("*.npy"))) == 1


def test_trusted_reply_skips_finite_scans_but_candidate_input_does_not(tmp_path):
    values = np.array([np.nan, np.inf, -np.inf], dtype=np.float64)
    calls = []
    def invoke(*args):
        calls.append(args)
        return values
    with oracle(tmp_path, invoke) as (host, client):
        assert not host.codec.validate_encoded_array_finite
        assert host.codec.validate_decoded_array_finite
        assert client.codec.validate_encoded_array_finite
        assert not client.codec.validate_decoded_array_finite
        result = client.invoke("one", "echo", (np.zeros((1, 3)),), {})
        assert result.tobytes() == values.tobytes()  # Intentional trusted-output policy.
        with pytest.raises(wire.AgentWireCodecError, match="finite"):
            client.invoke("one", "echo", (values,), {})
        assert len(calls) == 1


def raw_request(manifest, params, *, token=None):
    body = json.dumps({"schema": CANDIDATE_ORACLE_REQUEST_SCHEMA,
        "request_id": uuid4().hex, "params": params}).encode()
    request = Request(manifest["endpoint"], data=body, method="POST", headers={
        "Content-Type": "application/json", "Content-Length": str(len(body)),
        _TOKEN_HEADER: manifest["token"] if token is None else token})
    return build_opener(ProxyHandler({})).open(request, timeout=3)


@pytest.mark.parametrize("mutation", ["token", "case", "method", "shape", "dtype", "size", "nonfinite", "fields"])
def test_http_external_input_authorization_and_array_boundaries_still_reject(tmp_path, mutation):
    calls = []
    with oracle(tmp_path, lambda *args: calls.append(args)) as (host, client):
        manifest = dict(host.manifest_for_case("one"))
        args = client.codec.encode((np.zeros((4096, 3), dtype=np.float64),))
        node = args["items"][0]
        params = {"case_id": "one", "method": "query_source", "args": args, "kwargs": {}}
        if mutation == "case": params["case_id"] = "two"
        if mutation == "method": params["method"] = "not_exposed"
        if mutation == "shape": node["shape"] = [4096, 4]
        if mutation == "dtype": node["dtype"] = "|O"
        if mutation == "size":
            count = host.codec.inline_array_bytes // np.dtype(np.float64).itemsize + 1
            values = np.zeros(count, dtype=np.float64)
            node.update(shape=[count], data=base64.b64encode(values.tobytes()).decode())
        if mutation == "nonfinite":
            values = np.zeros((4096, 3), dtype=np.float64)
            values[0, 0] = np.nan
            node["data"] = base64.b64encode(values.tobytes()).decode()
        if mutation == "fields": params["extra"] = True
        with pytest.raises(HTTPError) as caught:
            raw_request(manifest, params, token="invalid" if mutation == "token" else None)
        assert caught.value.code == (403 if mutation in ("token", "case", "method") else 500)
        caught.value.close()
        assert calls == []


def test_client_loopback_case_and_method_checks_unchanged(tmp_path):
    with oracle(tmp_path, lambda *args: np.ones(1)) as (host, client):
        with pytest.raises(PermissionError, match="another Case"):
            client.invoke("two", "echo", (), {})
        with pytest.raises(AttributeError):
            client.invoke("one", "not_exposed", (), {})
        manifest = dict(host.manifest_for_case("one"))
        manifest["endpoint"] = "http://example.com/v1/candidate-public-oracle"
        with pytest.raises(CandidateExecutionInfrastructureError, match="not loopback"):
            CandidatePublicSessionClient(manifest)


@pytest.mark.parametrize("scan_option", ["validate_encoded_array_finite", "validate_decoded_array_finite"])
def test_new_codec_options_are_boolean_and_strict_by_default(scan_option):
    codec = wire.AgentWireCodec()
    assert codec.inline_array_bytes == 64 * 1024
    with pytest.raises(TypeError, match="boolean"):
        wire.AgentWireCodec(**{scan_option: "false"})
    values = np.array([np.nan], dtype=np.float64)
    with pytest.raises(wire.AgentWireCodecError, match="finite"):
        codec.encode(values)
    trusted = wire.AgentWireCodec(validate_encoded_array_finite=False)
    with pytest.raises(wire.AgentWireCodecError, match="finite"):
        codec.decode(trusted.encode(values))


def test_disabling_reply_scan_does_not_disable_metadata_validation():
    codec = wire.AgentWireCodec(validate_decoded_array_finite=False)
    node = codec.encode(np.zeros(3, dtype=np.float64))
    node["shape"] = [4]
    with pytest.raises(wire.AgentWireCodecError, match="shape"):
        codec.decode(node)


def test_default_asset_codec_still_uses_64k_sidecars(tmp_path):
    codec = wire.AgentWireCodec(tmp_path / "assets")
    values = np.arange(4096 * 3, dtype=np.float64).reshape(4096, 3)
    encoded = codec.encode(values)
    assert encoded[wire._WIRE_TAG] == "ndarray_npy"
    assert codec.decode(encoded).tobytes() == values.tobytes()


def test_loopback_microbenchmark_smoke_is_explicit_and_records_real_http(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/benchmark_artemis_opt4_rpc.py"
    spec = importlib.util.spec_from_file_location("_opt4_rpc_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    output = tmp_path / "rpc.json"
    args = ["--scratch-parent", str(tmp_path), "--output", str(output), "--iterations", "1"]
    with pytest.raises(RuntimeError, match="Slurm"):
        module.main(args)
    assert not output.exists()
    assert module.main([*args, "--allow-local-smoke"]) == 0
    report = json.loads(output.read_text())
    assert report["status"] == "passed" and report["model_execution_performed"] is False
    variants = report["filesystems"]["scratch"]["variants"]
    assert variants["baseline_64k_scanned"]["measured_sidecar_count"] == 2
    assert variants["opt4_128k_trusted"]["measured_sidecar_count"] == 0
    assert all(row["source_calls"] == 2 and row["large_array_compatibility_passed"] for row in variants.values())
    with pytest.raises(FileExistsError):
        module.main([*args, "--allow-local-smoke"])
