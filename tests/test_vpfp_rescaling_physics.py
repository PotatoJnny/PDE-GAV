"""CPU-only synthetic wiring tests; no physical or solver validation claims."""
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_PATH = Path(__file__).resolve().parents[1]/"scripts/vpfp_rescaling_physics.py"
_SPEC = importlib.util.spec_from_file_location("vpfp_rescaling_physics_contract_test", _PATH)
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)


def _finite(value):
    if isinstance(value, dict):
        return {key: _finite(v) for key, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_finite(v) for v in value]
    if isinstance(value, np.ndarray):
        return _finite(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _fixture(monkeypatch):
    model = torch.nn.Linear(1, 1).float()
    model.config = SimpleNamespace(latent_dim=3, hermite_total_degree=14,
        time_domain=(2., 6.), equation=SimpleNamespace(x_domain=(10., 20.), background_density=2.))
    draw = {"full": torch.tensor([[1., 2., 3.]], dtype=torch.float32),
        "equation_conditions": torch.tensor([[2., 4.]], dtype=torch.float32)}
    state = {"probe_calls": 0, "local_calls": 0, "ieee_active": False}
    plan = SimpleNamespace(time_fractions=(0., .015625, .0625, .125, .25, .5, .75, 1.),
        phase_probes_per_time=16, independent_source_probe_count=4,
        probe_seed=2026090701, chunk_size=128)

    @contextmanager
    def ieee(selected):
        assert selected is model and not selected.training
        assert not state["ieee_active"]
        state["ieee_active"] = True
        try:
            yield
        finally:
            state["ieee_active"] = False

    def probe(selected, full, theta, received_plan, analytic):
        state["probe_calls"] += 1
        assert selected is model and not selected.training
        assert not torch.is_grad_enabled() and state["ieee_active"]
        assert np.array_equal(full, draw["full"].numpy())
        assert np.array_equal(theta, draw["equation_conditions"].numpy()[0])
        assert received_plan is plan
        # The real helper internally re-enables AD for its four source checks.
        with torch.enable_grad():
            assert torch.is_grad_enabled()
        features = np.tile(np.arange(1., 10.), (8, 1))
        detail = {"points_txyvv": np.zeros((128, 5)), "algebraic_balance_maximum": 0.,
            "independent_source_check": {"execution_passed": True,
                "probe_indices": [0, 42, 84, 127], "maximum_dimensionless_difference": .001}}
        return features, detail

    def local(selected, full, condition, **kwargs):
        state["local_calls"] += 1
        state["local_kwargs"] = kwargs
        assert selected is model and not selected.training
        assert full is draw["full"] and condition is draw["equation_conditions"]
        assert kwargs["spatial_times"] == kwargs["spatial_grids"] == ()
        assert kwargs["relative_fd_step"] == 1/512 and kwargs["chunk_size"] == 512
        assert kwargs["include_l2"] is True
        points = []
        for index, txy in enumerate(kwargs["points_txy"]):
            checks = [{"velocity_order": order, "balance_residual": [value, 0., 0., 0.]}
                for order, value in ((17, .1), (17, .05), (19, .2), (19, .25))]
            points.append({"index": index, "txy": txy, "execution_passed": True,
                "source_L2_relative_to_nu_f_L2": .5+index,
                "source_moments_collision_normalized": [0., .1, .2, .3],
                "source_moment_fine_minus_coarse": [0., .01, 0., 0.],
                "L2_squared_fine_minus_coarse": {"f_l2_squared": .01, "source_l2_squared": .02},
                "finite_difference_checks": checks})
        return {"execution_passed": True, "numerically_uncertain": True,
            "attempted_point_count": 4, "completed_point_count": 4, "points": points,
            "spatial_mass": [], "pending": ["a_priori_calibrated_numerical_error_budget"]}

    components = SimpleNamespace(np=np, torch=torch, Plan=lambda: plan, probe=probe,
        local=local, finite=_finite, analytic=SimpleNamespace(ieee_float32_coordinate_matmul=ieee))
    monkeypatch.setattr(audit, "_components", lambda: components)
    # Synthetic contract checks run on CPU; they do not claim a real allocation.
    monkeypatch.setattr(audit, "_require_allocation", lambda: None)
    return model, draw, state, components


def test_light_exact_inventory_native_units_and_context_restoration(monkeypatch):
    model, draw, state, _ = _fixture(monkeypatch)
    original = {key: value.clone() for key, value in draw.items()}
    grad_enabled = torch.is_grad_enabled()
    report = audit.audit_draw(draw, model, full_local_audit=False)
    json.dumps(report, allow_nan=False)
    assert report["execution_passed"] and report["numerically_undetermined"]
    assert not report["scientific_admission"] and report["solver_validity_label"] is None
    assert state["probe_calls"] == 1 and state["local_calls"] == 0
    assert report["plan"]["light_phase_points"] == 128 and report["plan"]["independent_AD_points"] == 4
    row = report["summary"]["fixed_probes"]["by_time"][0]
    assert row["f_fixed_probe_rms"] == .5
    assert row["source_fixed_probe_rms"] == .25
    assert row["source_relative_to_nu_f_fixed_probe_rms"] == .25
    assert report["local"] is None and len(report["input"]["native_answer_sha256"]) == 64
    assert model.training and torch.is_grad_enabled() == grad_enabled and not state["ieee_active"]
    assert all(torch.equal(draw[key], value) for key, value in original.items())


def test_four_local_points_use_config_units_no_grid_and_no_convergence_gate(monkeypatch):
    model, draw, state, _ = _fixture(monkeypatch)
    report = audit.audit_draw(draw, model, full_local_audit=True)
    assert report["execution_passed"] and state["local_calls"] == 1
    assert np.allclose(state["local_kwargs"]["points_txy"],
        [[2.8, 2.5, 5.], [2.8, 7.5, 15.], [5.2, 2.5, 5.], [5.2, 7.5, 15.]])
    local = report["summary"]["local"]
    assert local["maximum_local_source_L2_relative_to_nu_f_L2"] == 3.5
    assert [x["fine_residual_not_larger"] for x in local["points"][0]["fd_refinement"]] == [True, False]
    assert report["numerically_undetermined"] and report["plan"]["numeric_tolerance"] is None
    assert report["local"]["pending"] and not local["global_spatial_mass_measured"]
    json.dumps(report, allow_nan=False)


def test_light_failure_keeps_local_attempt_and_does_not_become_solver_failure(monkeypatch):
    model, draw, state, components = _fixture(monkeypatch)
    def broken(*args):
        raise ArithmeticError("original probe reason")
    components.probe = broken
    model.eval()
    report = audit.audit_draw(draw, model, full_local_audit=True)
    assert not report["execution_passed"] and state["local_calls"] == 1
    assert report["failures"][0]["message"] == "original probe reason"
    assert report["local"]["execution_passed"] and report["solver_validity_label"] is None
    assert not model.training and not state["ieee_active"]


def test_nested_independent_failure_is_retained(monkeypatch):
    model, draw, _, components = _fixture(monkeypatch)
    original = components.probe
    def failed_probe(*args):
        features, detail = original(*args)
        detail["independent_source_check"] = {"execution_passed": False,
            "failure": {"type": "RuntimeError", "message": "independent AD unavailable"}}
        return features, detail
    components.probe = failed_probe
    report = audit.audit_draw(draw, model, full_local_audit=False)
    assert not report["execution_passed"] and not report["light"]["execution_passed"]
    assert report["light"]["detail"]["independent_source_check"]["failure"]["message"] == "independent AD unavailable"


@pytest.mark.parametrize("damage", ["nan", "dtype", "shape", "domain", "flag"])
def test_input_errors_fail_closed_before_measurement(monkeypatch, damage):
    model, draw, state, _ = _fixture(monkeypatch)
    local = False
    if damage == "nan": draw["full"][0, 0] = float("nan")
    elif damage == "dtype": draw["full"] = draw["full"].double()
    elif damage == "shape": draw["equation_conditions"] = draw["equation_conditions"][0]
    elif damage == "domain": model.config.time_domain = (2., 2.)
    else: local = 1
    report = audit.audit_draw(draw, model, full_local_audit=local)
    assert not report["execution_passed"] and report["numerically_undetermined"]
    assert state["probe_calls"] == state["local_calls"] == 0 and report["failures"]
    json.dumps(report, allow_nan=False)


def test_missing_allocation_blocks_before_optional_numerical_imports(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    def unexpected():
        pytest.fail("numerical components imported before allocation guard")
    monkeypatch.setattr(audit, "_components", unexpected)
    report = audit.audit_draw({}, None, full_local_audit=False)
    assert not report["execution_passed"]
    assert "compute allocation" in report["failures"][0]["message"]
