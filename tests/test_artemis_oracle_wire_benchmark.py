"""Small CPU smoke tests for the offline, unchanged-codec comparison."""

import json

import numpy as np
import pytest

from scripts import benchmark_artemis_oracle_wire as benchmark


def test_payload_matches_source_rpc_shape_and_changes_each_iteration():
    first, first_reply = benchmark.make_payload(0)
    second, second_reply = benchmark.make_payload(1)
    assert first.shape == (4096, 3) and first_reply.shape == (4096,)
    assert first.dtype == first_reply.dtype == np.dtype("float64")
    assert first.nbytes == 96 * 1024 and first_reply.nbytes == 32 * 1024
    assert np.all(first[:, 0] != second[:, 0])
    assert np.array_equal(first[:, 1:], second[:, 1:])
    assert not np.array_equal(first_reply, second_reply)


def test_numeric_equality_check_rejects_reordering_or_dtype_change():
    points, _ = benchmark.make_payload(1)
    benchmark._require_exact(points.copy(), points)
    with pytest.raises(AssertionError, match="row order"):
        benchmark._require_exact(points[::-1], points)
    with pytest.raises(AssertionError, match="dtype"):
        benchmark._require_exact(points.astype(np.float32), points)


def test_two_thresholds_use_identical_payloads_and_separate_warmup_sidecars(tmp_path):
    report = benchmark.benchmark_filesystem(tmp_path, iterations=2, warmup=1)
    assert report["status"] == "passed", report
    small, large = (report["variants"][str(value)] for value in benchmark.THRESHOLDS)
    assert small["measured_sidecar_count"] == 2
    assert small["measured_sidecar_bytes"] > 2 * 96 * 1024
    assert large["measured_sidecar_count"] == large["measured_sidecar_bytes"] == 0
    for variant in (small, large):
        assert all(variant["safety"].values())
        assert variant["warmup_and_safety_files_excluded"] is True
        assert len(variant["samples"]) == 2
        for field in ("encode_seconds", "decode_seconds", "full_roundtrip_seconds"):
            summary = variant["timing_seconds"][field]
            assert summary["p95"] >= summary["median"] > 0.0
        assert all(row["bitwise_equal"] for row in variant["samples"])
    for a, b in zip(small["samples"], large["samples"]):
        assert a["request_payload_sha256"] == b["request_payload_sha256"]
        assert a["reply_payload_sha256"] == b["reply_payload_sha256"]
        assert a["request_json_bytes"] < 2048
        assert b["request_json_bytes"] > 128 * 1024
        assert a["response_json_bytes"] == b["response_json_bytes"]
    assert small["samples"][0]["request_payload_sha256"] != small["samples"][1]["request_payload_sha256"]
    assert report["median_codec_roundtrip_ratio_64_over_128"] > 0.0


def test_failed_safety_probe_preserves_reason_and_omits_speed_ratio(tmp_path, monkeypatch):
    def reject(*args):
        raise AssertionError("intentional safety failure")
    monkeypatch.setattr(benchmark, "verify_safety", reject)
    report = benchmark.benchmark_filesystem(tmp_path, iterations=1, warmup=0)
    assert report["status"] == "failed"
    assert report["error"] == "intentional safety failure"
    assert "median_codec_roundtrip_ratio_64_over_128" not in report
    assert "workspace" in report


def test_cli_rejects_login_benchmark_before_writing(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(SystemExit) as caught:
        benchmark.main(["--scratch-parent", str(tmp_path / "scratch"),
                        "--output", str(tmp_path / "report.json")])
    assert caught.value.code == 2
    assert list(tmp_path.iterdir()) == []


def test_cli_small_local_smoke_marks_no_http_or_model_and_missing_local(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_TMPDIR", raising=False)
    output = tmp_path / "report.json"
    args = ["--scratch-parent", str(tmp_path / "scratch"), "--output", str(output),
            "--iterations", "1", "--warmup", "0", "--allow-local-smoke"]
    assert benchmark.main(args) == 0
    report = json.loads(output.read_text())
    assert report["status"] == "passed"
    assert report["benchmark_scope"] == "codec_microbenchmark_not_http_or_end_to_end"
    assert report["model_execution_performed"] is report["deployment_changed"] is False
    assert report["synthetic_payloads"] is True
    assert report["filesystems"]["local"]["status"] == "skipped"
    assert "SLURM_TMPDIR unset" in report["filesystems"]["local"]["reason"]
    with pytest.raises(SystemExit) as caught:
        benchmark.main(args)
    assert caught.value.code == 2  # Existing evidence cannot be overwritten.


def test_cli_uses_slurm_local_directory_when_available(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "unit-test-only")
    monkeypatch.setenv("SLURM_TMPDIR", str(tmp_path / "local"))
    output = tmp_path / "report.json"
    assert benchmark.main(["--scratch-parent", str(tmp_path / "scratch"),
                           "--output", str(output), "--iterations", "1", "--warmup", "0"]) == 0
    report = json.loads(output.read_text())
    assert report["filesystems"]["scratch"]["status"] == "passed"
    assert report["filesystems"]["local"]["status"] == "passed"
    assert report["filesystems"]["local"]["parent"] == str(tmp_path / "local")
    assert report["scratch_and_local_same_device"] is True


def test_percentile_summary_is_explicit_and_reproducible():
    summary = benchmark.summarize([{"encode_seconds": float(i)} for i in (1, 2, 3)])
    assert summary["encode_seconds"]["median"] == 2.0
    assert summary["encode_seconds"]["p95"] == pytest.approx(2.9)
