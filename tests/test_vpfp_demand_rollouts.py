"""Synthetic orchestration only, not physical or guidance validation."""
import importlib.util
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pde_agent_verifier_solver.sdk import AnswerQuery, AnswerView, Case, CaseSource, CaseVisibility
from pde_agent_verifier_solver.plugins.vpfp.macro_matching import VPFP_MACRO_FIELDS

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("demand_rollouts_under_test", ROOT/"scripts/vpfp_strong_guidance_rollouts.py")
roll = importlib.util.module_from_spec(spec)
spec.loader.exec_module(roll)


@pytest.mark.parametrize("mode", ["demand_points_v1", "demand_legacy_arrays_v1"])
def test_demand_rollout_does_not_materialize_and_scores_only_after_solver(monkeypatch, tmp_path, mode):
    cfg = json.loads((ROOT/"configs/vpfp_strong_solver_guidance_demand_20260908.json").read_text())
    row = roll.registered_roots(cfg)["fit"][0]
    events = []
    query = AnswerQuery.grid(fields=VPFP_MACRO_FIELDS, t=2, x=1, y=1)
    public = Case("anonymous", CaseSource.PUBLIC, CaseVisibility.HIDDEN,
        {"question": {"background_density": 1., "temperature": 1.},
         "arrays": {"t": np.asarray([0., 1.])}})
    def view(value):
        return AnswerView(query.kind, {name: (value, value) for name in query.fields},
            axes={"t": (0., 1.), "x": (0.,), "y": (0.,)}, shape=(2, 1, 1))
    class Session:
        def close(self): events.append("close")
    class Truth:
        def answer(self, selected):
            assert "solve" in events
            events.append("truth")
            return view(0.)
    replay = roll.CompletedDrawReplay({})
    class Runtime:
        def generate_base(self, **kwargs):
            replay.generate(kwargs["seeds"])
            events.append("register")
            case = Case(kwargs["case_ids"][0], CaseSource.GENERATED, CaseVisibility.HIDDEN, {})
            return [SimpleNamespace(record=SimpleNamespace(case=case, truth=Truth()), transaction=None)]
        def public_session_factory(self, case): return Session()
        def release_cases(self, case_ids): events.append("release")
    def forbidden(*args, **kwargs): raise AssertionError("eager full grid evaluation forbidden")
    def descriptor(case, *, legacy_array_adapter):
        assert legacy_array_adapter == (mode == "demand_legacy_arrays_v1")
        events.append("descriptor")
        return public
    def solve(asset, case, selected, **kwargs):
        assert kwargs["legacy_array_adapter"] == (mode == "demand_legacy_arrays_v1")
        assert kwargs["public_session"] is not None
        assert not (tmp_path/"draw"/"public-case.pt").exists()
        events.append("solve")
        return view(1.), {"execution_passed": True, "stage": "solver_completed",
            "prematerialized_points": 0,
            "oracle_usage": {"source": {"calls": 1, "points": 3}},
            "oracle_requests": [{"method": "source", "points": 3, "status": "completed"}]}
    monkeypatch.setattr(roll, "materialize_public_case", forbidden)
    monkeypatch.setattr(roll, "run_frozen_solver", forbidden)
    monkeypatch.setattr(roll, "build_demand_public_case", descriptor)
    monkeypatch.setattr(roll, "native_macro_query", lambda case: query)
    monkeypatch.setattr(roll, "run_demand_solver", solve)
    seed = row["root_seed"]
    draw = {"full": torch.zeros(1, 6), "initial_context": torch.ones(1, 2),
        "future_white": torch.ones(1, 4), "equation_conditions": torch.tensor([row["conditions"]]),
        "root_seeds": [seed], "known_prior_seeds": [roll._seed(f"vpfp-initial-conditioned-prior:{seed}")],
        "future_noise_seeds": [roll._seed(f"vpfp-initial-conditioned-future:{seed}")]}
    result = roll.evaluate_draw(draw, row, runtime=Runtime(), replay=replay,
        solver=SimpleNamespace(sha256="a"*64), output=tmp_path/"draw", budget_seconds=10.,
        check=lambda: None, public_access_mode=mode)
    assert result["valid"] and result["prematerialized_oracle_points"] == 0
    assert "public_materialization" not in result
    assert "public_materialization_seconds" not in result["stage_wall_seconds"]
    assert result["public_oracle_usage"]["source"]["points"] == 3
    assert events == ["register", "descriptor", "solve", "truth", "close", "release"]
    assert not (tmp_path/"draw"/"public-materialization.json").exists()
    assert (tmp_path/"draw"/"public-oracle-requests.json").is_file()


def test_demand_configuration_distinguishes_solver_grid_from_eager_materialization():
    cfg = json.loads((ROOT/"configs/vpfp_strong_solver_guidance_demand_20260908.json").read_text())
    assert cfg["public_access_mode"] == "demand_legacy_arrays_v1"
    assert "public_materialization" not in cfg
    assert cfg["legacy_solver_array_grid"] == roll.PUBLIC_DESIGN
    roll._config_scope(cfg)
    cfg["public_access_mode"] = "misspelled"
    with pytest.raises(ValueError, match="unknown public access"):
        roll._config_scope(cfg)


def test_legacy_prediction_scalars_load_with_scoped_safe_types_and_verified_hash(tmp_path):
    spec = importlib.util.spec_from_file_location("demand_probe_loader_test", ROOT/"scripts/probe_vpfp_demand_oracle.py")
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    path = tmp_path/"prediction.pt"
    payload = {"fields": {"rho": [np.float64(1.25)]},
        "axes": {"t": [np.float64(.5)]}, "shape": [1]}
    torch.save(payload, path)
    expected = sha256(path.read_bytes()).hexdigest()
    before = list(torch.serialization.get_safe_globals())
    actual = probe.load_saved_prediction(path, expected)
    assert actual == payload
    assert list(torch.serialization.get_safe_globals()) == before
    with pytest.raises(ValueError, match="SHA"):
        probe.load_saved_prediction(path, "0"*64)
