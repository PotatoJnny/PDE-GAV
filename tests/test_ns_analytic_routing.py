"""The production public NS boundary selects only the exact audited kernel."""
from pathlib import Path
import sys

import numpy as np
import pytest

torch = pytest.importorskip('torch')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'legacy/ns/src'))
from agent_verifier.neural_generator import structured_ns_answer_first as mms
from agent_verifier.neural_generator.continuous_geometry import ContinuousGeometrySpec
from pde_agent_verifier_solver.plugins.ns.continuous_backend import StructuredNSMMSAnswer


def field(*, trainable=False, sampled=False):
    config = mms.StructuredNSProjectionConfig(spatial_mode_x=2, spatial_mode_y=2, time_order=2)
    rng = torch.Generator().manual_seed(101)
    coefficients = mms.StructuredNSCaseCoefficients(
        psi=torch.randn((3, 9), dtype=torch.float64, generator=rng) * 1e-4,
        pressure=torch.randn((3, 9), dtype=torch.float64, generator=rng) * 1e-4)
    coefficients.psi.requires_grad_(trainable)
    kwargs = ({'canonical_geometry': torch.ones((2, 5, 5), dtype=torch.float64)} if sampled else
              {'geometry_spec': ContinuousGeometrySpec((0., 1., 0., 1.), ())})
    return mms.ContinuousStructuredNSMMS(coefficients, config=config, domain_bounds=(0., 1., 0., 1.),
                                       viscosity=.01, t0=0., t1=.12, **kwargs)


def test_public_answer_uses_no_autograd_and_preserves_reference(monkeypatch):
    raw = field()
    points = np.array([[.03, .2, .3], [.11, .03, .5], [0., 0., .5], [.12, .9, .99]])
    q = raw.quantities(points[:, (1, 2, 0)])
    expected_f = torch.stack((q['forcing_u'], q['forcing_v']), dim=-1).detach().numpy()
    u, v, _ = raw.velocity_pressure(points[:, (1, 2, 0)])
    expected_u = torch.stack((u, v), dim=-1).detach().numpy()
    answer = StructuredNSMMSAnswer(raw)
    assert answer._analytic_field
    def unexpected(*args, **kwargs):
        raise AssertionError('public optimized call entered coordinate autograd')
    monkeypatch.setattr(torch.autograd, 'grad', unexpected)
    np.testing.assert_allclose(answer.evaluate_forcing(points), expected_f, rtol=2e-10, atol=1e-9)
    np.testing.assert_allclose(answer.evaluate_velocity(points), expected_u, rtol=2e-10, atol=1e-9)


def test_trainable_coefficients_keep_autograd_path(monkeypatch):
    raw = field(trainable=True)
    answer = StructuredNSMMSAnswer(raw)
    assert not answer._analytic_field
    calls = []
    original = raw.quantities
    def observed(points):
        calls.append(True)
        return original(points)
    monkeypatch.setattr(raw, 'quantities', observed)
    actual = answer.evaluate_forcing(np.array([[.06, .4, .6]]))
    assert actual.shape == (1, 2) and np.isfinite(actual).all()
    assert calls == [True]


def test_overridden_field_methods_are_not_bypassed(monkeypatch):
    raw = field()
    monkeypatch.setattr(raw, 'velocity_pressure', lambda points: (
        np.full(len(points), 3.), np.full(len(points), 4.), np.zeros(len(points))))
    answer = StructuredNSMMSAnswer(raw)
    assert not answer._analytic_field
    np.testing.assert_array_equal(answer.evaluate_velocity([[.06, .2, .3]]), [[3., 4.]])


def test_sampled_legacy_geometry_preserves_reference_backend():
    raw = field(sampled=True)
    answer = StructuredNSMMSAnswer(raw, public_geometry={'legacy_sampled': True})
    assert not answer._analytic_field


@pytest.mark.parametrize('points', [[[float('nan'), .2, .3]], [[.2, .2, .3]], [[.06, 1.1, .3]], [[.1, .2]]])
def test_analytic_boundary_still_rejects_invalid_public_queries(points):
    with pytest.raises(ValueError):
        StructuredNSMMSAnswer(field()).evaluate_forcing(points)
