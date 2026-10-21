"""Smoke launcher checks with tiny interface fixtures, never real GPU assets."""

from dataclasses import replace
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from pde_agent_verifier_solver.app.backend_run import BackendRunBinding
from pde_agent_verifier_solver.plugins.gkw.canonical_binding import bind_gkw_runtime
from pde_agent_verifier_solver.plugins.vpfp.canonical_binding import _BoundVPFPContinuousGenerator
from pde_agent_verifier_solver.plugins.vpfp.macro_programs import VPFPHoldInitialMacroProgram
from pde_agent_verifier_solver.plugins.vpfp.product import build_vpfp_plugin
from pde_agent_verifier_solver.sdk import (
    CaseRecord, CaseSource, DatasetInput, DatasetKind, GeneratorCheckpoint, PreparedDataset,
)
from tests.test_artemis_canonical import _binding as artemis_binding
from tests.test_gkw_manufactured_backend import _PhysicalRuntime, _StateTruth
from tests.test_vpfp_canonical_binding import _public_case


_SPEC = importlib.util.spec_from_file_location("plasma_canonical_smoke_fixture", Path(__file__).resolve().parents[1] / "scripts/run_plasma_canonical_smoke.py")
SMOKE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = SMOKE
_SPEC.loader.exec_module(SMOKE)


class _VPFPTruth:
    def __init__(self, case):
        self.case = case

    def answer(self, query):
        return VPFPHoldInitialMacroProgram().predict_observables(self.case, query)


class _VPFPBackend:
    supports_perturbations = True
    supports_guidance = False

    def __init__(self):
        self.active = set()

    def generate(self, base, guidance, request, progress=None):
        records = []
        for index in range(request.count):
            parent = request.anchors[index] if request.anchors else None
            case = replace(_public_case(), case_id=f"{request.purpose.value}-{request.kind.value}-{request.seed}-{index}",
                visibility=request.visibility, round_index=request.round_index,
                source=CaseSource.PERTURBED if parent else CaseSource.GENERATED,
                parent_case_id=parent.case_id if parent else None)
            records.append(CaseRecord(case, _VPFPTruth(case)))
            self.active.add(case.case_id)
        return tuple(records)

    def release_cases(self, case_ids):
        self.active.difference_update(case_ids)


class _GKWNativeDomainSmokeRuntime(_PhysicalRuntime):
    """Expose the production smoke cell inside this otherwise tiny fixture.

    The actual selected runtime permits its complete public field grid. The
    old fixture advertises a two-cell release panel, so include the legal
    production smoke cell (0,42,0) in that panel without changing the script.
    """

    def _case(self, *args, **kwargs):
        item = super()._case(*args, **kwargs)
        if kwargs.get("parent_public") is not None:
            return item  # Preserve the trusted parent's exact panel identity.
        public = dict(item.public_data)
        indices = np.asarray([[0, 0, 0], [1, 1, 1], [0, 42, 0]], dtype=np.int64)
        indices.setflags(write=False)
        public["macro_query"] = MappingProxyType({**public["macro_query"], "field_indices": indices})
        public = MappingProxyType(public)
        return replace(item, public_data=public, truth=_StateTruth(item.hidden_handle, public))


def _binding(physics):
    if physics == "artemis":
        binding, source = artemis_binding()
        return binding, lambda: not source.records
    if physics == "gkw":
        runtime = _GKWNativeDomainSmokeRuntime()
        return bind_gkw_runtime(runtime), lambda: not runtime.committed_case_ids
    backend = _VPFPBackend()
    identity = {"binding_sha256": "a" * 64}
    generator = _BoundVPFPContinuousGenerator(backend, identity)
    return BackendRunBinding(
        replace(build_vpfp_plugin(), generator=generator),
        GeneratorCheckpoint(generator.info.program_id, identity),
        PreparedDataset(DatasetInput("smoke-fixture", DatasetKind.NUMERICAL), (), ()),
        report={"data_origin": "synthetic_interface_fixture"},
    ), lambda: not backend.active


@pytest.mark.parametrize("physics", ["artemis", "gkw", "vpfp"])
def test_smoke_main_writes_actual_canonical_result_with_zero_macro_fixture(tmp_path, monkeypatch, physics):
    binding, released = _binding(physics)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"physics": physics}))
    monkeypatch.setenv("SLURM_JOB_ID", "unit-test-no-real-job")
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    monkeypatch.setattr(SMOKE, "load_binding", lambda *args: binding)
    output = tmp_path / "report"
    exit_code = SMOKE.main(["--physics", physics, "--config", str(config), "--output", str(output), "--device", "cpu"])
    report = json.loads((output / "report.json").read_text())
    assert exit_code == 0, report.get("traceback")
    assert report["status"] == "passed"
    assert report["completed_rounds"] == report["terminal_round_reports"] == 1
    assert report["promotion_performed"] is True
    assert report["official_cohort_counts"][0]["total"] == 2
    assert report["official_cohort_counts"][0]["agent_case_counts"] == {"zero-macro-agent": {"original": 0, "base": 1, "guided": 0, "base_perturbed": 0, "guided_perturbed": 0}}
    assert math.isfinite(report["final_macro_score"])
    assert len(report["development_macro_scores"]) == 1
    assert {"base", "base_perturbed"} <= set(report["terminal_distribution_metrics"])
    assert report["full_solution_prediction_required"] is False
    assert report["model_training_performed"] is False
    assert report["scientific_qualification"] == "not_established_by_interface_smoke"
    assert report["online_pde_calls_during_learned_sampling"] is None
    assert report["online_pde_call_check"] == "not_instrumented"
    assert report["process_environment"] == {
        "JAX_PLATFORMS": "cpu", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    assert released()


@pytest.mark.parametrize("physics,module_name,factory_name", [
    ("artemis", "pde_agent_verifier_solver.plugins.vp_1d1v.product", "load_artemis_backend"),
    ("gkw", "pde_agent_verifier_solver.plugins.gkw.canonical_binding", "load_gkw_binding"),
    ("vpfp", "pde_agent_verifier_solver.plugins.vpfp.canonical_binding", "build_vpfp_online_binding"),
])
def test_factory_dispatch_overrides_device_once(tmp_path, monkeypatch, physics, module_name, factory_name):
    module = __import__(module_name, fromlist=[factory_name])
    calls = []
    sentinel = object()
    def factory(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel
    monkeypatch.setattr(module, factory_name, factory)
    settings = {"device": "cuda"} if physics == "artemis" else {"some_model_setting": "value"}
    payload = {"physics": physics, "factory_settings": settings}
    assert SMOKE.load_binding(physics, payload, tmp_path, "cpu") is sentinel
    assert len(calls) == 1 and calls[0][1]["device"] == "cpu"
    assert payload["factory_settings"] == settings


def test_smoke_retains_factory_failure_reason(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text('{"physics":"artemis"}')
    monkeypatch.setenv("SLURM_JOB_ID", "unit-test-no-real-job")
    def fail(*args):
        raise ValueError("test asset binding mismatch")
    monkeypatch.setattr(SMOKE, "load_binding", fail)
    output = tmp_path / "failed"
    assert SMOKE.main(["--physics", "artemis", "--config", str(config), "--output", str(output)]) == 1
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "failed"
    assert "test asset binding mismatch" in report["error"]
    assert "traceback" in report


@pytest.mark.parametrize("physics,options,error", [
    ("vpfp", ["--learned-checkpoint", "learned.npz"], "supplied together"),
    ("vpfp", ["--learned-checkpoint-sha256", "a" * 64], "supplied together"),
    ("artemis", ["--learned-checkpoint", "learned.npz", "--learned-checkpoint-sha256", "a" * 64], "only for VPFP"),
    ("gkw", ["--learned-checkpoint", "learned.npz", "--learned-checkpoint-sha256", "a" * 64], "only for VPFP"),
    ("vpfp", ["--learned-checkpoint", "learned.npz", "--learned-checkpoint-sha256", "short"], "64-character SHA256"),
    ("vpfp", ["--learned-checkpoint", "learned.npz", "--learned-checkpoint-sha256", "A" * 64], "lowercase"),
])
def test_learned_options_fail_before_loading_assets(tmp_path, monkeypatch, capsys, physics, options, error):
    monkeypatch.setenv("SLURM_JOB_ID", "unit-test-no-real-job")
    monkeypatch.setattr(SMOKE, "load_binding", lambda *args, **kwargs: pytest.fail("must reject before binding"))
    output = tmp_path / "not-created"
    with pytest.raises(SystemExit) as failure:
        SMOKE.main(["--physics", physics, "--config", str(tmp_path / "unused.json"),
                    "--output", str(output), *options])
    assert failure.value.code == 2
    assert error in capsys.readouterr().err
    assert not output.exists()


def test_learned_factory_dispatch_loads_hash_bound_sampler(tmp_path, monkeypatch):
    from pde_agent_verifier_solver.plugins.vpfp import canonical_binding, learned_latent
    model, sampler, binding = object(), object(), object()
    calls = []
    checkpoint = tmp_path / "learned.npz"
    checkpoint_sha256 = "a" * 64

    def load(path, *, expected_sha256):
        calls.append(("load", path, expected_sha256))
        return model

    def wrap(loaded_model, *, checkpoint_sha256):
        assert loaded_model is model
        calls.append(("sampler", checkpoint_sha256))
        return sampler

    def bind(spec, **settings):
        assert spec == {"regime_id": "fixture"}
        assert settings == {"output_root": tmp_path / "physical_runtime", "device": "cpu",
                            "guidance": False, "latent_sampler": sampler}
        calls.append(("binding",))
        return binding

    monkeypatch.setattr(learned_latent.VPFPConditionalLatentDistribution, "load", load)
    monkeypatch.setattr(learned_latent, "VPFPLearnedBatchGenerator", wrap)
    monkeypatch.setattr(canonical_binding, "build_vpfp_online_binding", bind)
    result = SMOKE.load_binding("vpfp", {"vpfp": {"regime_id": "fixture"}}, tmp_path, "cpu",
        learned_checkpoint=checkpoint, learned_checkpoint_sha256=checkpoint_sha256)
    assert result is binding
    assert calls == [("load", checkpoint, checkpoint_sha256), ("sampler", checkpoint_sha256), ("binding",)]


def _online_generation_fixture(monkeypatch):
    actual_calls = []

    class OnlineGenerator:
        def generate(self, *args, **kwargs):
            actual_calls.append(True)

    module = SimpleNamespace(VPFPOnlineBatchGenerator=OnlineGenerator)
    monkeypatch.setitem(sys.modules, "agent_verifier.vpfp.online_training", module)
    return module, OnlineGenerator.generate, actual_calls


def _learned_smoke_arguments(tmp_path):
    config = tmp_path / "config.json"
    config.write_text('{"physics":"vpfp"}')
    return ["--physics", "vpfp", "--config", str(config), "--output", str(tmp_path / "report"),
            "--device", "cpu", "--learned-checkpoint", str(tmp_path / "fixture.npz"),
            "--learned-checkpoint-sha256", "a" * 64]


def test_learned_smoke_guards_complete_canonical_campaign_and_restores(tmp_path, monkeypatch):
    binding, released = _binding("vpfp")
    online, original_generate, actual_calls = _online_generation_fixture(monkeypatch)
    monkeypatch.setenv("SLURM_JOB_ID", "unit-test-no-real-job")
    calls = []
    guarded_generations = []
    backend = binding.plugin.generator.backend
    backend_generate = backend.generate

    def generate(*args, **kwargs):
        assert online.VPFPOnlineBatchGenerator.generate is not original_generate
        guarded_generations.append(True)
        return backend_generate(*args, **kwargs)

    monkeypatch.setattr(backend, "generate", generate)

    def load(*args, **kwargs):
        calls.append(kwargs)
        return binding

    monkeypatch.setattr(SMOKE, "load_binding", load)
    assert SMOKE.main(_learned_smoke_arguments(tmp_path)) == 0
    report = json.loads((tmp_path / "report/report.json").read_text())
    assert calls == [{"learned_checkpoint": tmp_path / "fixture.npz", "learned_checkpoint_sha256": "a" * 64}]
    assert report["completed_rounds"] == report["terminal_round_reports"] == 1
    assert report["generation_mode"] == "learned_vpfp_checkpoint"
    assert report["online_pde_call_check"] == "guarded_entire_learned_campaign"
    assert report["online_pde_calls_during_learned_sampling"] == 0
    assert report["online_pde_guard_restored"] is True
    assert report["model_training_performed"] is False
    assert report["scientific_performance_validation"] is False
    assert report["scientific_qualification"] == "not_established_by_interface_smoke"
    assert online.VPFPOnlineBatchGenerator.generate is original_generate
    assert not actual_calls
    assert len(guarded_generations) >= 5  # Dev, Official base/pair, Terminal base/pair.
    assert released()


@pytest.mark.parametrize("failure_mode,attempts,error", [
    ("forbidden_online", 1, "attempted an online VPFP coarse PDE solve"),
    ("campaign_error", 0, "fixture Campaign failure"),
    ("suppressed_online", 1, "suppressed a forbidden online VPFP generation attempt"),
])
def test_learned_guard_counts_attempts_and_restores_on_campaign_failure(
        tmp_path, monkeypatch, failure_mode, attempts, error):
    online, original_generate, actual_calls = _online_generation_fixture(monkeypatch)
    monkeypatch.setenv("SLURM_JOB_ID", "unit-test-no-real-job")

    def run(*args, **kwargs):
        assert online.VPFPOnlineBatchGenerator.generate is not original_generate
        if failure_mode == "campaign_error":
            raise ValueError("fixture Campaign failure")
        if failure_mode == "forbidden_online":
            online.VPFPOnlineBatchGenerator().generate((1,))
        else:
            try:
                online.VPFPOnlineBatchGenerator().generate((1,))
            except RuntimeError:
                pass

    binding = SimpleNamespace(run=run, report={"data_origin": "synthetic_interface_fixture"})
    monkeypatch.setattr(SMOKE, "load_binding", lambda *args, **kwargs: binding)
    assert SMOKE.main(_learned_smoke_arguments(tmp_path)) == 1
    report = json.loads((tmp_path / "report/report.json").read_text())
    assert report["status"] == "failed"
    assert error in report["error"]
    assert "traceback" in report
    assert report["online_pde_calls_during_learned_sampling"] == attempts
    assert report["online_pde_guard_restored"] is True
    assert report["model_training_performed"] is False
    assert online.VPFPOnlineBatchGenerator.generate is original_generate
    assert not actual_calls


@pytest.mark.parametrize("script", ["slurm_gkw_canonical_smoke.sh", "slurm_plasma_canonical_smoke.sh"])
def test_gkw_launchers_preserve_qualified_jax_cpu_torch_cuda_split(script):
    source = (Path(__file__).resolve().parents[1] / "scripts" / script).read_text()
    setting = "export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false"
    assert setting in source
    assert source.index(setting) < source.index("--device cuda")
