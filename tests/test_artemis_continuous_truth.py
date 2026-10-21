"""Exact off-grid routing and compact checkpoint tests; no real model needed."""
from dataclasses import replace
from types import SimpleNamespace
import sys
import numpy as np
import pytest
import torch

from pde_agent_verifier_solver.plugins.vp_1d1v.continuous_truth import (
    ArtemisContinuousMacroTruth, continuous_record,
)
from pde_agent_verifier_solver.plugins.vp_1d1v import resume
from pde_agent_verifier_solver.sdk import AnswerQuery, AnswerView
from tests.test_artemis_resume import _fake_generator, _fake_records, fake_module


FIELDS = ("M0", "M1", "M2", "M3", "E")


def test_continuous_record_and_checkpoint_preserve_field_without_resampling(tmp_path, fake_module):
    original = _fake_generator()
    original.backend.continuous_macro_truth = True
    record = continuous_record(original.backend, _fake_records(original))
    descriptor = resume.export_case_record(original, record, tmp_path)
    runtime = resume.export_runtime_state(original, tmp_path)
    assert runtime["identity"]["continuous_macro_truth"]
    assert "continuous_manufactured" in descriptor
    restored = _fake_generator()
    restored.backend.continuous_macro_truth = True
    resume.restore_runtime_state(restored, runtime, tmp_path)
    decoded = resume.restore_case_record(restored, descriptor, tmp_path)
    assert isinstance(decoded.truth, ArtemisContinuousMacroTruth)
    assert decoded.truth.native == record.truth.native
    handle = decoded.truth.manufactured.hidden_handle
    assert torch.equal(handle.field._condition_scores,
                       record.truth.manufactured.hidden_handle.field._condition_scores)
    assert restored.backend.source.runtime.materializations == 0
    # A saved released Case also restores from its compact descriptor.
    restored.backend.source.release_cases((record.case.case_id,))
    decoded = resume.restore_case_record(restored, descriptor, tmp_path)
    assert isinstance(decoded.truth, ArtemisContinuousMacroTruth)
    assert restored.backend.source.runtime.materializations == 0


def test_legacy_truth_record_unchanged():
    original = _fake_generator()
    record = _fake_records(original)
    assert continuous_record(original.backend, record) is record


class _Native:
    t, x = (0., .5, 1.), (0., .5)
    def answer(self, query):
        return "native-cache"


class _Field:
    dtype, device = torch.float64, torch.device("cpu")
    def __init__(self):
        self.distribution_calls = 0
    def distribution_from_tx_jet(self, jet, velocity):
        self.distribution_calls += 1
        return torch.ones((jet.moments.M0.numel(), velocity.numel()), dtype=self.dtype)


class _Runtime:
    def __init__(self, child):
        self.field = _Field()
        self.handle = SimpleNamespace(field=self.field, spatial_phase=object(),
                                      curvatures=object() if child else None)
        self.contract = SimpleNamespace(t0=0., t1=1., x_length=1.)
        self._quadrature_v_tensor = torch.tensor([-1., 0., 1.], dtype=torch.float64)
        self._quadrature_weight_tensor = torch.tensor([.5, 1., .5], dtype=torch.float64)
        self.batch_sizes = []
        self.map_calls = 0
    def _exact_query_handle(self, manufactured):
        return self.handle
    def _sensor_map(self, velocity, curvatures):
        self.map_calls += 1
        return SimpleNamespace(transformed_velocity=velocity, derivative=torch.ones_like(velocity))
    def _factorized_tx_jet(self, field, time, position, phase):
        self.batch_sizes.append(time.numel())
        return SimpleNamespace(moments=SimpleNamespace(**{
            name: time + position + index for index, name in enumerate(FIELDS)
        }))


def delta_corrected_child_moments_from_views(jet, child, parent, velocity, weights):
    return SimpleNamespace(**{
        name: getattr(jet.moments, name) + (0 if name in ("M0", "E") else 1)
        for name in FIELDS
    })


@pytest.mark.parametrize("child", [False, True])
def test_exact_point_routing_batches_without_native_grid_growth(child):
    runtime = _Runtime(child)
    truth = ArtemisContinuousMacroTruth(_Native(), runtime, object())
    query = AnswerQuery.at_points(fields=FIELDS, points=tuple(
        {"t": .5, "x": (index + .25) / 256} for index in range(256)))
    result = truth.answer(query)
    assert result.shape == (256,)
    assert result.points == query.points
    expected = np.array([p["t"] + p["x"] for p in query.points])
    np.testing.assert_allclose(result.fields["M0"], expected)
    np.testing.assert_allclose(result.fields["M1"], expected + 1 + child)
    np.testing.assert_allclose(result.fields["E"], expected + 4)
    assert runtime.batch_sizes == [16] * 16
    assert runtime.map_calls == int(child)
    assert runtime.field.distribution_calls == (32 if child else 0)
    native_query = AnswerQuery.at_points(fields=FIELDS, points=({"t": .5, "x": 0.},))
    assert truth.answer(native_query) == "native-cache"
    assert runtime.batch_sizes == [16] * 16


def test_exact_point_routing_rejects_out_of_domain():
    truth = ArtemisContinuousMacroTruth(_Native(), _Runtime(False), object())
    query = AnswerQuery.at_points(fields=FIELDS, points=({"t": .5, "x": 1.01},))
    with pytest.raises(ValueError, match="physical domain"):
        truth.answer(query)


def test_backend_smoke_requires_compute_before_writing(tmp_path, monkeypatch):
    from scripts.check_artemis_vpfp_aligned_backend import main
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(SystemExit):
        main(["--backend-config", "unused", "--output-root", str(tmp_path / "smoke")])
    assert not (tmp_path / "smoke").exists()
