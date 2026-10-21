"""Exercise formal Controller startup with both real VPFP generator classes.

Only the numerical backend is synthetic; no physical/GPU qualification is claimed.
"""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from pde_agent_verifier_solver.app.backend_run import BackendRunBinding
from pde_agent_verifier_solver.plugins.vpfp.canonical_binding import _BoundVPFPContinuousGenerator
from pde_agent_verifier_solver.plugins.vpfp.macro_programs import VPFPHoldInitialMacroProgram
from pde_agent_verifier_solver.plugins.vpfp.product import build_vpfp_plugin
from pde_agent_verifier_solver.plugins.vpfp.weak_iteration_backend import WeakIterationGenerator
from pde_agent_verifier_solver.sdk import (
    Artifact, CaseVisibility, DatasetInput, DatasetKind, GenerationKind,
    GenerationPurpose, GenerationRequest, GeneratorCheckpoint, GeneratorMode,
    PreparedDataset, TaskKind,
)
from scripts import plasma_gpu_campaign as campaign
from tests.test_plasma_transport_campaign import upgraded
from tests.test_plasma_transport_task_wiring import calibration


def binding_for(equation, *, transport=True):
    asset = calibration()
    identity = {"binding_sha256": "a" * 64, "sampling": {"model_sha256": "b" * 64}}
    metric = {}
    if transport:
        identity["scoring_objective"] = asset.objective_identity
        metric = {"metric_profile": "plasma_macro_transport_v2", "metric_calibration": asset.to_dict()}
    backend = SimpleNamespace(supports_guidance=True, supports_perturbations=True,
                              initial_guidance=None, requests=[])

    def generate(base, guidance, request, progress=None):
        backend.requests.append(request)
        return ()

    backend.generate = generate
    generator_class = _BoundVPFPContinuousGenerator if equation == "strong" else WeakIterationGenerator
    generator = generator_class(backend, identity)
    binding = BackendRunBinding(
        replace(build_vpfp_plugin(**metric), generator=generator),
        GeneratorCheckpoint(generator.info.program_id, identity),
        PreparedDataset(DatasetInput("synthetic-load", DatasetKind.NUMERICAL), (), ()),
        report=identity,
    )
    return binding, metric


@pytest.mark.parametrize("equation", ["strong", "weak"])
@pytest.mark.parametrize("arm", ["sol", "astra"])
def test_formal_controller_loads_both_vpfp_generators_with_configured_objective(
    tmp_path, monkeypatch, equation, arm,
):
    binding, metric = binding_for(equation)
    _, config = upgraded(tmp_path, equation, arm)
    config["generator"]["settings"].update(metric)
    config["task"].update(metric)
    initial = Artifact("S0", TaskKind.OBSERVABLE_MODEL, "Synthetic incumbent",
        VPFPHoldInitialMacroProgram(), "load-boundary fixture", "unused", 0, "fixture")
    plan = campaign.recipe(config, initial)
    controller = binding.controller(plan, {row["agent_id"]: object() for row in config["agents"]},
                                    local_development=True, task_id="observable_model")

    class Loaded(Exception):
        pass

    def stop_after_load():
        assert controller._continuation_state["phase"] == "bootstrap"
        raise Loaded

    monkeypatch.setattr(controller, "_save_continuation", stop_after_load)
    with pytest.raises(Loaded):
        controller.run()
    state = controller._continuation_state
    settings = state["generator_settings"]
    assert settings["metric_calibration"] == metric["metric_calibration"]
    assert settings["metric_profile"] == metric["metric_profile"]
    assert settings["master_seed"] == 101 and isinstance(settings["controller_seed"], int)
    assert state["generator_state"].base == binding.checkpoint.payload
    request = GenerationRequest(1, 1, GeneratorMode.ONLINE, GenerationKind.BASE,
        CaseVisibility.HIDDEN, 102, GenerationPurpose.SELECTION, settings=settings)
    binding.plugin.generator.sample(state["generator_state"], request)
    assert binding.plugin.generator.backend.requests[0].settings["metric_calibration"] == metric["metric_calibration"]
    assert binding.plugin.generator.backend.requests[0].settings["fixed_generator_sha256"] == "a" * 64


@pytest.mark.parametrize("equation", ["strong", "weak"])
@pytest.mark.parametrize("mutation", ["objective", "floor", "remove", "profile", "physical"])
def test_bound_generator_rejects_scoring_or_physical_mutation_on_load_and_sample(equation, mutation):
    binding, metric = binding_for(equation)
    generator = binding.plugin.generator
    state = generator.load(binding.checkpoint, metric)
    changed = deepcopy(metric)
    if mutation == "objective":
        changed["metric_calibration"] = calibration(2.).to_dict()
    elif mutation == "floor":
        changed["metric_calibration"]["floors"]["density"] *= 2
    elif mutation == "remove":
        changed = {"metric_profile": None, "metric_calibration": None}
    elif mutation == "profile":
        changed["metric_profile"] = "unrecognized-objective"
    else:
        changed["latent_rank"] = 256
    with pytest.raises(ValueError):
        generator.load(binding.checkpoint, changed)
    request = GenerationRequest(1, 1, GeneratorMode.ONLINE, GenerationKind.BASE,
        CaseVisibility.HIDDEN, 102, GenerationPurpose.SELECTION, settings=changed)
    with pytest.raises(ValueError):
        generator.sample(state, request)
    assert generator.backend.requests == []


@pytest.mark.parametrize("equation", ["strong", "weak"])
def test_transport_cannot_be_injected_into_a_legacy_binding_or_mutate_checkpoint(equation):
    legacy, _ = binding_for(equation, transport=False)
    _, metric = binding_for(equation)
    legacy.plugin.generator.load(legacy.checkpoint, {})
    with pytest.raises(ValueError, match="bound scoring objective"):
        legacy.plugin.generator.load(legacy.checkpoint, metric)
    binding, metric = binding_for(equation)
    binding.checkpoint.payload["scoring_objective"]["objective_hash"] = "c" * 64
    with pytest.raises(ValueError, match="does not match"):
        binding.plugin.generator.load(binding.checkpoint, metric)
