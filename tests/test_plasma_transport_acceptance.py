"""CPU-only acceptance orchestration/algebra tests; no physical GPU claims."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from scripts import check_plasma_transport as api
from scripts import plasma_gpu_campaign as campaign
from tests.test_plasma_gpu_campaign import config
from tests.test_plasma_transport_scoring import _calibration, _raw2, _view, _case
from pde_agent_verifier_solver.plugins.vpfp.macro_matching import VPFPTransportMacroMatcher
from pde_agent_verifier_solver.sdk import evaluate_macro_prediction


@pytest.mark.parametrize("family", ["strong", "weak", "artemis"])
def test_config_adaptation_is_in_memory_and_preserves_resources(family):
    old = config(family)
    before = deepcopy(old)
    asset = _calibration()
    new = api.build_transport_config(old, asset)
    assert old == before
    assert new["schema"] == campaign.TRANSPORT_SCHEMA
    assert new["task"]["metric_profile"] == "plasma_macro_transport_v2"
    assert new["task"]["metric_calibration"]["objective_hash"] == asset.objective_hash
    assert new["generator"]["settings"]["metric_calibration"] == new["task"]["metric_calibration"]
    assert new["protocol"]["scoring_objective_hash"] == asset.objective_hash
    for name in ("backend", "execution", "official", "terminal", "master_seed"):
        assert new[name] == before[name]


def test_acceptance_refuses_login_node_before_reading_assets(monkeypatch, tmp_path):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="compute allocation"):
        api.run_acceptance(tmp_path / "missing-config", tmp_path / "missing-calibration", tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_acceptance_refuses_cpu_fallback(monkeypatch, tmp_path):
    import torch
    monkeypatch.setenv("SLURM_JOB_ID", "123456789")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        api.run_acceptance(tmp_path / "missing-config", tmp_path / "missing-calibration", tmp_path / "output")


def _binding(asset):
    matcher = VPFPTransportMacroMatcher(asset)
    task = SimpleNamespace(macro_matcher=matcher)
    guide = SimpleNamespace(score_calibration=asset)
    backend = SimpleNamespace(score_calibration=asset, guidance_adapter=guide)
    return SimpleNamespace(plugin=SimpleNamespace(macro_matcher=matcher,
        task=lambda _: task, generator=SimpleNamespace(backend=backend)))


def test_objective_identity_covers_actual_task_backend_and_guide():
    asset = _calibration()
    binding = _binding(asset)
    hashes = api.check_objective_bindings(binding, asset)
    assert set(hashes) == {"plugin_matcher", "task_matcher", "backend", "guidance_recipe"}
    assert set(hashes.values()) == {asset.objective_hash}
    binding.plugin.generator.backend.guidance_adapter.score_calibration = None
    with pytest.raises(ValueError, match="guidance_recipe"):
        api.check_objective_bindings(binding, asset)


def test_independent_formula_check_uses_retained_candidate_values():
    asset = _calibration()
    query, truth = _view(_raw2())
    predicted = _raw2()
    predicted["heat_flux_y"][:] = 0.3
    _, prediction = _view(predicted)
    result = evaluate_macro_prediction(matcher=VPFPTransportMacroMatcher(asset),
        case=_case(), prediction=prediction, truth=truth, query=query)
    row = api.verify_case_score(result, truth, prediction, asset, "weak")
    assert row["point_count"] == 2
    assert row["primary_error"] == pytest.approx(result.primary_error)
    assert row["physical_errors"]["heat_flux"] > 0
    assert row["prediction_density_min"] > 0
    assert row["bitwise_equality_required"] is False
    with pytest.raises(ValueError, match="four-block"):
        api.verify_case_score(replace(result, primary_error=0.), truth, prediction, asset, "weak")


def test_acceptance_keeps_one_existing_source_readiness_and_real_candidate_path():
    import inspect
    source = inspect.getsource(api.run_acceptance)
    assert source.count("check_binding(family,") == 1
    for required in ("PythonCandidateCodec(", "SubprocessCandidateExecutionService(",
                     "HiddenVerifier(", "verifier.evaluate(artifact, records)",
                     "GenerationKind.PERTURBED", "check_feedback_objective("):
        assert required in source
    assert "case_hard_wall_seconds=900" in source
    assert "fit_guidance(" not in source
    assert "sbatch" not in source and "scancel" not in source


@pytest.mark.parametrize("bad_public_density", [False, True])
def test_full_cpu_orchestration_uses_real_candidate_subprocess_and_public_rpc(
    monkeypatch, tmp_path, bad_public_density,
):
    """Fake physics only: the real SDK, worker, RPC, scoring and receipts run."""
    import json
    import numpy as np
    import torch
    from scripts import check_plasma_gpu_integration as readiness
    from pde_agent_verifier_solver.plugins.vpfp.product import build_vpfp_plugin
    from pde_agent_verifier_solver.sdk import (
        AnswerView, Case, CaseAccessKind, CaseRecord, CaseSource,
        CaseVisibility, GenerationKind,
    )
    asset = _calibration()
    public_calls, released, requests, readiness_calls = [], [], [], []

    class Session:
        def rho0(self, points):
            points = np.asarray(points)
            public_calls.append(points.shape)
            return np.full(len(points), np.nan if bad_public_density else 1.)

        def close(self):
            pass

    class Truth:
        def answer(self, query):
            values = _raw2(len(query.points))
            return AnswerView(query.kind, {key: tuple(values[key]) for key in query.fields},
                              points=query.points, shape=(len(query.points),))

    class Generator:
        backend = SimpleNamespace(score_calibration=asset)

        def load(self, checkpoint, settings):
            assert settings["metric_profile"] == "plasma_macro_transport_v2"
            assert settings["metric_calibration"] == asset.to_dict()
            assert settings["master_seed"] == 101
            assert isinstance(settings["controller_seed"], int)
            self.settings = deepcopy(settings)
            return "cpu-synthetic-state"

        def sample(self, state, request):
            assert request.settings == self.settings
            requests.append(request)
            perturbed = request.kind is GenerationKind.PERTURBED
            case = Case(
                "synthetic-child" if perturbed else "synthetic-base",
                CaseSource.PERTURBED if perturbed else CaseSource.GENERATED,
                CaseVisibility.HIDDEN,
                {"arrays": {"t": (0., 1.), "x": (0., np.pi), "y": (0., np.pi)},
                 "question": {"grid": {"x_domain": (2*np.pi, 2*np.pi)},
                              "time_domain": (0., 1.), "domain": {"time": (0., 1.)},
                              "temperature": 1., "collision_nu": .1,
                              "numerical_access": {"max_points_per_request": 1024}}},
                parent_case_id=request.anchors[0].case_id if perturbed else None,
            )
            return (CaseRecord(case, Truth()),)

        def release(self, state, case_ids):
            released.extend(case_ids)

        def case_session_methods(self, access):
            return ("rho0",) if access is CaseAccessKind.PUBLIC_PROBLEM else ()

        def open_case_session(self, state, case, access):
            assert access is CaseAccessKind.PUBLIC_PROBLEM
            return Session()

    generator = Generator()
    plugin = build_vpfp_plugin(metric_profile="plasma_macro_transport_v2", metric_calibration=asset)
    binding = SimpleNamespace(plugin=replace(plugin, generator=generator), checkpoint="synthetic")
    monkeypatch.setenv("SLURM_JOB_ID", "123456789")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "synthetic-test-no-real-GPU")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(campaign, "_binding", lambda configuration, root: (binding, {"synthetic": True}))

    def source_check(family, current_binding, *, output_path, seed):
        readiness_calls.append((family, seed))
        return {"status": "passed", "synthetic": True}

    monkeypatch.setattr(readiness, "check_binding", source_check)
    config_path, calibration_path = tmp_path / "config.json", tmp_path / "calibration.json"
    config_path.write_text(json.dumps(config("weak")))
    calibration_path.write_text(json.dumps(asset.to_dict()))
    output = tmp_path / "acceptance"
    if bad_public_density:
        with pytest.raises(Exception, match="Candidate"):
            api.run_acceptance(config_path, calibration_path, output, point_count=4)
    else:
        result = api.run_acceptance(config_path, calibration_path, output, point_count=4)
        assert result["status"] == "passed"
        assert result["guidance_feedback_matches_promotion"]
        assert len(result["cases"]) == len(result["candidate_audits"]) == 2
        assert all(item["status"] == "completed" for item in result["candidate_audits"])
        assert all(item["pid"] is not None and item["process_group_cleaned"]
                   for item in result["candidate_audits"])
    saved = json.loads((output / "report.json").read_text())
    assert saved["status"] == ("failed" if bad_public_density else "passed")
    assert saved["private_cases_released"]
    assert released == ["synthetic-base", "synthetic-child"]
    assert len(readiness_calls) == 1
    assert [request.kind for request in requests] == [GenerationKind.BASE, GenerationKind.PERTURBED]
    assert public_calls and all(shape == (1024, 2) for shape in public_calls)
