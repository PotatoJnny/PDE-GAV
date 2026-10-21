"""Synthetic persistence/ordering checks, not real guidance qualification."""
from copy import deepcopy
from functools import partial
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pde_agent_verifier_solver.sdk import AnswerQuery, AnswerView, Case, CaseSource, CaseVisibility
from pde_agent_verifier_solver.plugins.vpfp.macro_matching import VPFP_MACRO_FIELDS

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("guidance_rollouts_test_module", ROOT/"scripts/vpfp_strong_guidance_rollouts.py")
roll = importlib.util.module_from_spec(spec)
spec.loader.exec_module(roll)


@pytest.fixture(autouse=True)
def historical_materialized_rollout(monkeypatch):
    """These retained v1 tests explicitly exercise the historical adapter."""
    monkeypatch.setattr(roll, "evaluate_draw", partial(roll.evaluate_draw,
        public_access_mode="materialized_legacy_v1"))


def _config():
    return json.loads((ROOT/"configs/vpfp_strong_solver_guidance_20260908.json").read_text())


def test_roots_are_fixed_balanced_and_disjoint_without_drawing_a_model():
    config = _config()
    registered = roll.registered_roots(config)
    assert registered == roll.registered_roots(config)
    used = set()
    for split, rows in registered.items():
        assert len(rows) == config["counts"][split]
        assert len({row["slot_id"] for row in rows}) == len(rows)
        for s in range(8):
            group = [r for r in rows if r["stratum"] == s]
            assert len(group) == len(rows)//8
            for row in group:
                assert row["root_seed"] not in used
                used.add(row["root_seed"])
                nu, temp = row["conditions"]
                assert 10+5*(s//2) < nu < 15+5*(s//2)
                assert .8+.2*(s%2) < temp < 1.+.2*(s%2)
                assert np.array_equal(np.asarray([nu, temp]), np.asarray([nu, temp], dtype=np.float32))
    assert len(used) == 128


@pytest.mark.parametrize("change", ["count", "seed", "grid", "trainable"])
def test_changed_design_fails_without_silent_subsampling(change):
    config = _config()
    if change == "count": config["counts"]["fit"] = 8
    elif change == "seed": config["seeds"]["validation"] = config["seeds"]["fit"]
    elif change == "grid": config["public_materialization"]["nx"] = 17
    elif change == "trainable": config["base_weights_trainable"] = True
    with pytest.raises(ValueError): roll.registered_roots(config)


def _fixture_assets(tmp_path):
    config = _config()
    for key in ("base_checkpoint", "retained_control_checkpoint", "solver"):
        source = tmp_path/(key+".py" if key == "solver" else key+".pt")
        source.write_text("def predict_macros(*a,**k): pass\n" if key == "solver" else key)
        config[key], config[key+"_sha256"] = str(source), roll.file_sha(source)
    native = {"regime_id": "long_time_strong_collision_v2", "collision_nu_range": [10., 30.],
              "temperature_range": [.8, 1.2], "time_domain": [0., 1.]}
    for key in ("base_decoder", "expanded_decoder"):
        source = tmp_path/(key+".pt")
        source.write_bytes(key.encode())
        native[key+"_checkpoint"], native[key+"_sha256"] = str(source), roll.file_sha(source)
    native_config = tmp_path/"native.json"
    native_config.write_text(json.dumps({"vpfp": native}))
    config["native_config"] = str(native_config)
    return config


def test_freezing_checks_actual_bytes_copies_readonly_and_never_changes_originals(tmp_path):
    config = _fixture_assets(tmp_path)
    original_paths = [Path(config[k]) for k in ("base_checkpoint", "retained_control_checkpoint", "solver")]
    before = {str(p): (p.read_bytes(), p.stat().st_mode) for p in original_paths}
    manifest, native = roll.freeze_round_assets(config, tmp_path/"out")
    assert Path(manifest["manifest_path"]).is_file()
    for role, item in manifest["assets"].items():
        assert Path(item["path"]) != Path(item["source_path"])
        assert roll.file_sha(item["path"]) == item["sha256"]
        assert Path(item["path"]).stat().st_mode & 0o222 == 0
    assert native.base_decoder_checkpoint == manifest["assets"]["base_decoder"]["path"]
    assert {str(p): (p.read_bytes(), p.stat().st_mode) for p in original_paths} == before
    second, other = roll.freeze_round_assets(config, tmp_path/"out")
    assert second == manifest and native == other
    assert manifest["native_asset_structural_preflight_completed"] is False


def test_frozen_asset_mutation_is_not_repaired_or_overwritten(tmp_path):
    config = _fixture_assets(tmp_path)
    manifest, _ = roll.freeze_round_assets(config, tmp_path/"out")
    frozen = Path(manifest["assets"]["solver"]["path"])
    frozen.chmod(0o600)
    frozen.write_text("# corrupted\n")
    with pytest.raises(ValueError, match="SHA"):
        roll.freeze_round_assets(config, tmp_path/"out")
    assert frozen.read_text() == "# corrupted\n"


def _draw(row):
    seed = row["root_seed"]
    return {"full": torch.zeros(1, 6), "initial_context": torch.ones(1, 2),
        "future_white": torch.ones(1, 4)*2, "equation_conditions": torch.tensor([row["conditions"]], dtype=torch.float32),
        "root_seeds": [seed], "known_prior_seeds": [roll._seed(f"vpfp-initial-conditioned-prior:{seed}")],
        "future_noise_seeds": [roll._seed(f"vpfp-initial-conditioned-future:{seed}")]}


def test_replay_consumes_completed_draw_once_without_second_sampling():
    row = roll.registered_roots(_config())["fit"][0]
    draw = _draw(row)
    identity = {"base_decoder_sha256": "a"*64, "kind": "learned"}
    replay = roll.CompletedDrawReplay(identity)
    replay.set_draw(draw, guidance_report={"strength": .5})
    with pytest.raises(ValueError, match="overwrite"): replay.set_draw(draw)
    with pytest.raises(ValueError, match="changed"): replay.generate((row["root_seed"]+1,))
    batch = replay.generate((row["root_seed"],))
    np.testing.assert_array_equal(batch["latents"], draw["full"].numpy())
    assert batch["diagnostics"]["guidance"] == {"strength": .5}
    assert replay.consumed_count == 1
    with pytest.raises(ValueError, match="resample"): replay.generate((row["root_seed"],))
    assert "completed_draw_replay" not in identity


def test_binding_uses_no_shared_backend_guide_and_replay_identity(monkeypatch, tmp_path):
    captured = {}
    def bind(spec, **kwargs):
        captured.update(kwargs)
        return "binding"
    monkeypatch.setattr(roll, "build_vpfp_online_binding", bind)
    binding, replay = roll.build_rollout_runtime("native", SimpleNamespace(identity={"base": "sha"}), tmp_path, "cuda")
    assert binding == "binding" and captured["latent_sampler"] is replay
    assert captured["guidance"] is False
    assert replay.identity["shared_backend_guidance_enabled"] is False


def _orchestration(monkeypatch):
    events = []
    query = AnswerQuery.grid(fields=VPFP_MACRO_FIELDS, t=2, x=1, y=1)
    def view(value):
        return AnswerView(query.kind, {k: (value, value) for k in query.fields},
                          axes={"t": (0., 1.), "x": (0.,), "y": (0.,)}, shape=(2, 1, 1))
    public = Case("anonymous", CaseSource.PUBLIC, CaseVisibility.HIDDEN,
                  {"question": {"background_density": 1., "temperature": 1.}, "arrays": {"t": np.asarray([0., 1.])}})
    class Session:
        def close(self): events.append("close")
    class Truth:
        def answer(self, q):
            assert events.index("solve") < len(events)
            events.append("truth")
            return view(0.)
    class Runtime:
        def generate_base(self, **kwargs):
            self.replay.generate(kwargs["seeds"])
            events.append("register")
            case = Case(kwargs["case_ids"][0], CaseSource.GENERATED, CaseVisibility.HIDDEN, {})
            return [SimpleNamespace(record=SimpleNamespace(case=case, truth=Truth()),
                transaction=SimpleNamespace(commit=lambda: events.append("commit")))]
        def public_session_factory(self, case):
            events.append("session")
            return Session()
        def release_cases(self, ids): events.append("release")
    runtime = Runtime()
    replay = roll.CompletedDrawReplay({})
    runtime.replay = replay
    def materialize(*args):
        events.append("materialize")
        return public, {"execution_passed": True, "oracle_methods": ["source", "rho0", "g0"]}
    def solver(*args, **kwargs):
        events.append("solve")
        return view(1.), {"execution_passed": True, "stage": "solver_completed"}
    monkeypatch.setattr(roll, "materialize_public_case", materialize)
    monkeypatch.setattr(roll, "native_macro_query", lambda case: query)
    monkeypatch.setattr(roll, "run_frozen_solver", solver)
    return runtime, replay, events


def test_real_reward_scoring_occurs_only_after_solver_and_preserves_private_draw(monkeypatch, tmp_path):
    runtime, replay, events = _orchestration(monkeypatch)
    row = roll.registered_roots(_config())["fit"][0]
    result = roll.evaluate_draw(_draw(row), row, runtime=runtime, replay=replay,
        solver=SimpleNamespace(sha256="a"*64), output=tmp_path/"draw", budget_seconds=10, check=lambda: None)
    assert result["valid"] and result["execution_passed"]
    assert result["validity_label_available"] and result["label_kind"] == "valid_solver_error"
    assert result["label_unavailable_reason"] is None
    assert result["error"] == 1.0
    assert result["error"] == result["scoring"]["physical_scaled_rmse"]
    assert result["reward"] == pytest.approx(np.log(2)/(1+np.log(2)))
    assert events.index("materialize") < events.index("solve") < events.index("truth") < events.index("release")
    assert events[-2:] == ["close", "release"]
    private = torch.load(result["private_draw"]["path"], weights_only=True)
    assert torch.equal(private["future_white"], _draw(row)["future_white"])
    assert private["row"] == row
    assert roll.file_sha(result["result_path"]) == result["result_sha256"]
    saved = json.loads(Path(result["result_path"]).read_text())
    assert saved["valid"] and saved["resident_released"]
    assert Path(result["private_draw"]["path"]).stat().st_mode & 0o077 == 0
    with pytest.raises(FileExistsError):
        roll.evaluate_draw(_draw(row), row, runtime=runtime, replay=replay,
            solver=SimpleNamespace(sha256="a"*64), output=tmp_path/"draw", budget_seconds=10, check=lambda: None)


@pytest.mark.parametrize("failure", ["solver", "materialize", "deadline", "root", "condition", "stream", "nan", "dtype"])
def test_failures_retain_original_slot_no_reward_and_release(monkeypatch, tmp_path, failure):
    runtime, replay, events = _orchestration(monkeypatch)
    row = roll.registered_roots(_config())["fit"][0]
    draw = _draw(row)
    checks = [0]
    def check():
        checks[0] += 1
        if failure == "deadline" and checks[0] == 3: raise TimeoutError("allocation deadline")
    if failure == "solver":
        monkeypatch.setattr(roll, "run_frozen_solver", lambda *a, **k: (None,
            {"execution_passed": False, "stage": "solver_timeout", "solver_prediction_attempted": True,
             "validity_label_available": True, "error": {"type": "TimeoutError", "message": "own solver timeout"}}))
    elif failure == "materialize":
        def fail(*args): raise ValueError("public source not finite")
        monkeypatch.setattr(roll, "materialize_public_case", fail)
    elif failure == "root": draw["root_seeds"] = [1]
    elif failure == "condition": draw["equation_conditions"][0, 0] += 1
    elif failure == "stream": draw["future_noise_seeds"] = [1]
    elif failure == "nan": draw["full"][0, 0] = float("nan")
    elif failure == "dtype": draw["full"] = draw["full"].double()
    def invoke():
        return roll.evaluate_draw(draw, row, runtime=runtime, replay=replay,
            solver=SimpleNamespace(sha256="a"*64), output=tmp_path/"draw", budget_seconds=10, check=check)
    if failure == "deadline":
        with pytest.raises(TimeoutError, match="allocation deadline") as caught:
            invoke()
        result = caught.value.rollout_result
        assert result["stage"] == "allocation_interrupted"
    else:
        result = invoke()
    assert not result["valid"] and result["reward"] is None
    assert result["row"] == row and result["resident_released"]
    assert "truth" not in events
    assert "release" in events and replay._pending is None
    assert Path(result["result_path"]).is_file()
    assert Path(result["private_draw"]["path"]).is_file()
    assert result["validity_label_available"] is (failure == "solver")
    assert result["label_kind"] == ("numerical_solver_failure" if failure == "solver" else "unavailable")
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("checkpoint", [1, 4, 5])
def test_allocation_interruption_persists_and_keeps_only_completed_real_score(monkeypatch, tmp_path, checkpoint):
    runtime, replay, events = _orchestration(monkeypatch)
    row = roll.registered_roots(_config())["fit"][0]
    checks = 0
    interruption = TimeoutError("allocation ended, not solver budget")
    def check():
        nonlocal checks
        checks += 1
        if checks == checkpoint: raise interruption
    with pytest.raises(TimeoutError) as caught:
        roll.evaluate_draw(_draw(row), row, runtime=runtime, replay=replay,
            solver=SimpleNamespace(sha256="a"*64), output=tmp_path/"draw", budget_seconds=10, check=check)
    assert caught.value is interruption
    result = caught.value.rollout_result
    saved = json.loads(Path(result["result_path"]).read_text())
    assert roll.file_sha(result["result_path"]) == result["result_sha256"]
    assert not saved["execution_passed"] and saved["stage"] == "allocation_interrupted"
    assert saved["resident_released"] and "release" in events and replay._pending is None
    assert saved["validity_label_available"] is (checkpoint == 5)
    if checkpoint == 5:
        assert saved["valid"] and saved["label_kind"] == "valid_solver_error"
        assert saved["error"] == saved["scoring"]["physical_scaled_rmse"] == 1.
        assert saved["reward"] > 0 and "truth" in events
    else:
        assert not saved["valid"] and saved["label_kind"] == "unavailable"
        assert saved["reward"] is None and "truth" not in events


@pytest.mark.parametrize("failure", ["asset", "public_timeout", "cleanup"])
def test_infrastructure_failures_are_not_numeric_solver_labels(monkeypatch, tmp_path, failure):
    runtime, replay, events = _orchestration(monkeypatch)
    row = roll.registered_roots(_config())["fit"][0]
    if failure == "asset":
        monkeypatch.setattr(roll, "run_frozen_solver", lambda *a, **k: (None, {
            "execution_passed": False, "stage": "solver_failed", "solver_prediction_attempted": False,
            "validity_label_available": False, "failure_origin": "asset_changed",
            "error": {"type": "ValueError", "message": "asset SHA mismatch"}}))
    elif failure == "public_timeout":
        def fail(*args): raise TimeoutError("public API deadline, not allocation")
        monkeypatch.setattr(roll, "materialize_public_case", fail)
    else:
        def cleanup(ids):
            events.append("release")
            raise ValueError("resident cleanup failed")
        runtime.release_cases = cleanup
    result = roll.evaluate_draw(_draw(row), row, runtime=runtime, replay=replay,
        solver=SimpleNamespace(sha256="a"*64), output=tmp_path/"draw", budget_seconds=10, check=lambda: None)
    assert not result["validity_label_available"] and result["label_kind"] == "unavailable"
    assert result["label_unavailable_reason"] and not result["valid"] and result["reward"] is None
    assert Path(result["result_path"]).is_file() and "allocation_interruption" not in result
    assert replay._pending is None and "release" in events
    if failure == "cleanup":
        assert result["scoring"]["reward_eligible"] and result["label_unavailable_reason"] == "cleanup_failed"
    json.dumps(result, allow_nan=False)
