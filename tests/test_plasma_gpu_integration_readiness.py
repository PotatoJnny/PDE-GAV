"""Synthetic readiness orchestration; no GPU/model qualification claims."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import check_plasma_gpu_integration as api


def test_comparison_accepts_small_floating_point_differences():
    result = api.compare(np.array([.1, -2.]), np.array([.100001, -2.00001]), name="synthetic")
    assert result["passed"] is True
    assert result["bitwise_equality_required"] is False


@pytest.mark.parametrize("value", [np.array([np.nan]), np.array([np.inf]), np.array([])])
def test_nonfinite_or_empty_arrays_never_pass(value):
    with pytest.raises(ValueError):
        api.compare(np.array([1.]), value, name="synthetic")


def test_large_error_or_shape_difference_never_passes():
    with pytest.raises(ValueError, match="numerical readiness mismatch"):
        api.compare(np.array([1.]), np.array([1.1]), name="synthetic")
    with pytest.raises(ValueError, match="shape changed"):
        api.compare(np.array([1.]), np.array([[1.]]), name="synthetic")


def test_real_readiness_requires_compute_allocation(monkeypatch, tmp_path):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="compute allocation"):
        api.check_binding("weak", None, output_path=tmp_path / "result.json")


def _binding(nonfinite=False):
    class Session:
        closed = False
        def source(self, points):
            return np.full(len(points), np.nan if nonfinite else .5)
        def rho0(self, points):
            return np.ones(len(points))
        def g0(self, points):
            return np.full(len(points), .2)
        def close(self):
            self.closed = True
    session = Session()
    class Generator:
        released = None
        request = None
        backend = SimpleNamespace(public_session_factory=lambda case: session,
            runtime=SimpleNamespace(oracle_kernel="synthetic", expanded_model=SimpleNamespace(
                parameters=lambda: iter([SimpleNamespace(device=SimpleNamespace(type="cuda"))]))))
        def load(self, checkpoint, settings):
            return "synthetic-state"
        def sample(self, state, request):
            self.request = request
            case = SimpleNamespace(case_id="synthetic-readiness", public_data={"question": {
                "time_domain": [0., 1.], "grid": {"x_domain": [2., 2.]}}})
            truth = SimpleNamespace(answer=lambda q: SimpleNamespace(
                fields={name: np.ones(len(q.points)) for name in q.fields}))
            return (SimpleNamespace(case=case, truth=truth),)
        def release(self, state, case_ids):
            self.released = (state, case_ids)
    generator = Generator()
    return SimpleNamespace(plugin=SimpleNamespace(generator=generator), checkpoint="synthetic"), session


@pytest.fixture
def simulated_cuda(monkeypatch):
    import torch
    monkeypatch.setenv("SLURM_JOB_ID", "synthetic-test-only")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "synthetic-no-gpu")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)


def test_one_case_receipt_cleanup_and_no_formal_budget(simulated_cuda, tmp_path):
    binding, session = _binding()
    path = tmp_path / "receipt.json"
    report = api.check_binding("strong", binding, output_path=path)
    saved = json.loads(path.read_text())
    assert saved["status"] == report["status"] == "passed"
    assert saved["formal_case_count"] == 0
    assert saved["case_count"] == 1
    assert saved["private_probe_released"] is True
    assert saved["performance_benchmark"] is False
    assert binding.plugin.generator.released == ("synthetic-state", ("synthetic-readiness",))
    assert session.closed
    assert binding.plugin.generator.request.count == 1
    with pytest.raises(FileExistsError):
        api.check_binding("strong", binding, output_path=path)


def test_failure_receipt_and_probe_release_are_preserved(simulated_cuda, tmp_path):
    binding, session = _binding(nonfinite=True)
    path = tmp_path / "receipt.json"
    with pytest.raises(ValueError, match="public source"):
        api.check_binding("strong", binding, output_path=path)
    saved = json.loads(path.read_text())
    assert saved["status"] == "failed"
    assert saved["private_probe_released"] is True
    assert binding.plugin.generator.released is not None
    assert session.closed


def test_weak_probe_includes_sparse_and_repeated_phase_geometry():
    binding, _ = _binding()
    record = binding.plugin.generator.sample(None, SimpleNamespace())[0]
    points, start = api._vpfp_points(record)
    assert points.shape == (19, 5) and start == 0.
    unique, counts = np.unique(points[:, :3], axis=0, return_counts=True)
    assert len(unique) == 2 and max(counts) >= 16
    assert np.unique(points[-16:, 3:], axis=0).shape[0] == 16
