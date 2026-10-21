from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pytest

from pde_agent_verifier_solver.plugins.vpfp.canonical_binding import (
    LegacyMacroObservableAdapter, VPFPOnlineModelSpec, vpfp_public_guidance_features,
)
from pde_agent_verifier_solver.plugins.vpfp.learned_latent import (
    VPFPConditionalLatentDistribution, VPFPLearnedBatchGenerator,
)
from pde_agent_verifier_solver.plugins.vpfp.macro_programs import VPFPHoldInitialMacroProgram
from pde_agent_verifier_solver.plugins.vpfp.product import build_vpfp_plugin
from pde_agent_verifier_solver.sdk import (
    AnswerQuery, Artifact, Case, CaseSource, CaseVisibility, TaskKind,
)


@pytest.fixture
def mocked_factory_binding(tmp_path, monkeypatch):
    """Exercise the complete factory identity/wrapper path without Torch/PDE work."""
    import pde_agent_verifier_solver.plugins.vpfp.canonical_binding as module
    from pde_agent_verifier_solver.plugins.reward_guidance import RewardGuidanceConfig
    from pde_agent_verifier_solver.plugins.vpfp.perturbation_backend import VPFPRuntimeCase
    from pde_agent_verifier_solver.sdk import CaseRecord

    base, expanded = tmp_path / "base.pt", tmp_path / "expanded.pt"
    base.write_bytes(b"qualified-decoder-test-double")
    expanded.write_bytes(b"expanded-decoder-test-double")
    spec = VPFPOnlineModelSpec("weak", str(base), sha256(base.read_bytes()).hexdigest(),
        str(expanded), sha256(expanded.read_bytes()).hexdigest(), (.02, .08))
    decoder = SimpleNamespace(collision_nu_range=spec.collision_nu_range,
        temperature_range=spec.temperature_range, time_domain=spec.time_domain,
        structured_time_control_count=33, latent_dim=4,
        equation=SimpleNamespace(x_domain=(0., 1.)))

    class Config:
        def __init__(self, **values):
            self.values = values

        def to_dict(self):
            return dict(self.values)

    class Truth:
        def answer(self, query):
            raise AssertionError("guidance must not inspect physical truth")

    class Runtime:
        supports_perturbations = True

        def __init__(self, **_kwargs):
            self.calls, self.released, self.live = [], [], {}

        def generate_base(self, **kwargs):
            self.calls.append(kwargs)
            result = []
            for seed, case_id in zip(kwargs["seeds"], kwargs["case_ids"]):
                template = _public_case()
                public = {**template.public_data, "question": {
                    "collision_nu": .02 + (seed % 100000) / 100000 * .06,
                    "temperature": 1.0,
                }}
                case = replace(template, case_id=case_id, public_data=public,
                    visibility=kwargs["visibility"], round_index=kwargs["round_index"])
                self.live[case_id] = case
                result.append(VPFPRuntimeCase(CaseRecord(case, Truth()), private_handle={"seed": seed}))
            return tuple(result)

        def release_cases(self, case_ids):
            self.released.extend(case_ids)
            for case_id in case_ids:
                self.live.pop(case_id, None)

    original_import = module.importlib.import_module
    modules = {
        "agent_verifier.vpfp.continuous_hermite": SimpleNamespace(__file__=module.__file__,
            VPFPContinuousHermiteConfig=SimpleNamespace(from_dict=lambda _: decoder)),
        "agent_verifier.vpfp.online_training": SimpleNamespace(__file__=module.__file__,
            VPFPOnlineGenerationConfig=Config, VPFPOnlineBatchGenerator=lambda **kwargs: SimpleNamespace(**kwargs)),
        "agent_verifier.vpfp.high_fidelity": SimpleNamespace(VPFPHighFidelityConfig=Config),
        "agent_verifier.vpfp.operators": SimpleNamespace(VPFPGrid=Config, VPFPTime=Config),
    }
    monkeypatch.setattr(module.importlib, "import_module", lambda name, *args, **kwargs: modules[name] if name in modules else original_import(name, *args, **kwargs))
    payload = {"robust_answer_first_adapter": {
        "unused_neural_parameters_zeroed": True,
        "continuous_hidden_answer_is_primary": True,
        "generator_parameterization_public": False,
    }, "decoder_config": {}}
    monkeypatch.setattr(module, "_legacy_api", lambda: SimpleNamespace(
        torch=SimpleNamespace(load=lambda *args, **kwargs: payload),
        build_deterministic_direct_model=lambda *args, **kwargs: object()))
    monkeypatch.setattr(module, "VPFPLegacyStructuredRuntime", Runtime)
    binding = module.build_vpfp_online_binding(spec, output_root=tmp_path / "runtime",
        device="cpu", guidance=True, guidance_config=RewardGuidanceConfig(pool_size=3))
    return binding, binding.plugin.generator.backend.backend.runtime


def test_factory_binds_demand_oracle_implementation_identity(mocked_factory_binding):
    from pathlib import Path
    import pde_agent_verifier_solver.plugins.vpfp.canonical_binding as module

    binding, _runtime = mocked_factory_binding
    identity = binding.report["public_oracle_evaluation"]
    assert identity["mode"] == "continuous_on_demand"
    assert identity["cartesian_completion"] is False
    assert identity["cross_case_cache"] is False
    assert identity["source_chunk_size"] == 8192
    for name in ("batched_source.py", "perturbation_backend.py"):
        expected = sha256(Path(module.__file__).with_name(name).read_bytes()).hexdigest()
        assert identity["implementation_sha256"][name] == expected
    assert binding.checkpoint.payload["public_oracle_evaluation"] == identity


def test_factory_guided_sampling_uses_exact_finalized_binding_identity(mocked_factory_binding):
    from pde_agent_verifier_solver.sdk import (
        CaseFeedback, GenerationKind, GenerationPurpose, GenerationRequest,
        GeneratorFeedback, GeneratorMode,
    )
    binding, runtime = mocked_factory_binding
    generator = binding.plugin.generator
    state = generator.load(binding.checkpoint, {})
    request = GenerationRequest(0, 4, GeneratorMode.ONLINE, GenerationKind.BASE,
        CaseVisibility.HIDDEN, 101, GenerationPurpose.GUIDANCE_FEEDBACK)
    feedback_records = generator.sample(state, request)
    artifact = Artifact("S0", TaskKind.OBSERVABLE_MODEL, "Frozen incumbent",
        object(), "test double", "test double", 0, "test")
    feedback = GeneratorFeedback(0, 1, artifact.artifact_id,
        tuple(CaseFeedback(row.case, row.case.public_data["question"]["collision_nu"] * 10, {}) for row in feedback_records),
        selected_program=artifact)
    guided_state = generator.update_guidance(state, feedback, {})
    identity = binding.checkpoint.payload["binding_sha256"]
    assert generator.backend.base_generator_sha256 == identity
    assert guided_state.guidance.base_generator_sha256 == identity
    assert binding.report["guidance_config"]["pool_size"] == 3
    guided = generator.sample(guided_state, replace(request,
        round_index=1, count=2, kind=GenerationKind.GUIDED, seed=202,
        purpose=GenerationPurpose.SELECTION))
    assert len(guided) == 2
    assert len(runtime.calls[-1]["seeds"]) == 6
    assert len(runtime.released) == 4  # rejected proposal roots, not outputs
    assert all(runtime.live[row.case.case_id] is row.case for row in guided)
    generator.release(guided_state, tuple(row.case.case_id for row in feedback_records + guided))
    assert not runtime.live


@pytest.mark.parametrize("settings", [
    {"representation": "fourier_hermite_positive"},
    {"latent_rank": 256},
    {"generation_batch_size": 16},
    {"guidance_pool_size": 4},
    {"perturbation_relative_change_min": .01},
])
def test_bound_vpfp_rejects_unused_run_model_settings(mocked_factory_binding, settings):
    from pde_agent_verifier_solver.sdk import GenerationKind, GenerationPurpose, GenerationRequest, GeneratorMode
    binding, runtime = mocked_factory_binding
    generator = binding.plugin.generator
    assert {field.name for field in generator.setup_fields()} == {"guidance_enabled", "perturbations_enabled"}
    state = generator.load(binding.checkpoint, {})
    with pytest.raises(ValueError, match="frozen at backend binding"):
        generator.load(binding.checkpoint, settings)
    request = GenerationRequest(1, 1, GeneratorMode.ONLINE, GenerationKind.BASE,
        CaseVisibility.HIDDEN, 202, GenerationPurpose.SELECTION, settings=settings)
    with pytest.raises(ValueError, match="frozen at backend binding"):
        generator.sample(state, request)
    assert not runtime.calls


def test_bound_identity_is_not_mutated_by_checkpoint_metadata_aliases(mocked_factory_binding):
    binding, _runtime = mocked_factory_binding
    generator = binding.plugin.generator
    binding.checkpoint.payload["sampling"]["kind"] = "silently-replaced-sampler"
    with pytest.raises(ValueError, match="does not match"):
        generator.load(binding.checkpoint, {})


def test_online_assets_fail_closed_and_legacy_config_is_understood(tmp_path):
    base, expanded = tmp_path / "base.pt", tmp_path / "expanded.pt"
    base.write_bytes(b"base-checkpoint")
    expanded.write_bytes(b"expanded-checkpoint")
    spec = VPFPOnlineModelSpec.from_mapping({
        "regime_id": "weak", "base_decoder_checkpoint": str(base),
        "expanded_decoder_checkpoint": str(expanded),
        "base_decoder_sha256": sha256(base.read_bytes()).hexdigest(),
        "expanded_decoder_sha256": sha256(expanded.read_bytes()).hexdigest(),
        "collision_nu_range": [0.02, 0.08],
    })
    assert spec.preflight()["regime_id"] == "weak"
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        replace(spec, base_decoder_sha256="0" * 64).preflight()
    with pytest.raises(ValueError, match="unknown VPFP"):
        VPFPOnlineModelSpec.from_mapping({"quietly_skip_hash": True})


def _public_case():
    return Case("private-id-is-not-a-feature", CaseSource.GENERATED,
                CaseVisibility.DEVELOPMENT, public_data={
        "question": {"collision_nu": 0.03, "temperature": 1.0},
        "arrays": {"t": np.array([0., 1.]), "x": np.array([0., 0.5]),
                   "y": np.array([0., 0.5]), "rho0": np.ones((2, 2)),
                   "momentum0": np.zeros((2, 2, 2)),
                   "kinetic_energy0": np.ones((2, 2))},
    })


def test_native_task_accepts_macros_only_and_never_requests_phase_space():
    case = _public_case()
    old = VPFPHoldInitialMacroProgram()
    assert not hasattr(old, "predict_solution")
    candidate = LegacyMacroObservableAdapter(old)
    query = AnswerQuery.at_points(
        fields=("rho", "momentum_x", "momentum_y", "kinetic_energy"),
        points=({"t": 0.4, "x": 0.12, "y": 0.37},),
    )
    truth = old.predict_macros(case, query)
    artifact = Artifact("macro-only", TaskKind.OBSERVABLE_MODEL, "macro-only",
                        candidate, "no f prediction", "predict_observables", 1, "test")
    result = build_vpfp_plugin().task().evaluate(artifact, case, truth, query)
    assert result.primary_error == pytest.approx(0.)
    assert set(truth.fields) == {"rho", "momentum_x", "momentum_y", "kinetic_energy"}
    assert vpfp_public_guidance_features(case) == vpfp_public_guidance_features(
        replace(case, case_id="different-id", visibility=CaseVisibility.HIDDEN,
                round_index=42, tags=("terminal",)))


def _trained_fixture():
    # Synthetic arrays exercise statistics only; this is not physical validation.
    random = np.random.default_rng(71)
    conditions = random.uniform([0.02, .8], [.08, 1.2], size=(64, 2))
    noise = random.normal(size=(64, 2))
    latents = np.column_stack((10 * conditions[:, 0] + noise[:, 0],
                              2 * noise[:, 0] + .03 * noise[:, 1],
                              conditions[:, 1], np.zeros(64)))
    return VPFPConditionalLatentDistribution.fit(
        latents, conditions, sample_ids=range(64), base_decoder_sha256="a" * 64,
        collision_nu_range=(.02, .08), temperature_range=(.8, 1.2), rank=2,
        provenance={"data_origin": "synthetic_test", "source_artifact_sha256": "b" * 64},
    )


def test_learned_distribution_replays_joint_samples_and_keeps_inactive_coefficients(tmp_path):
    model = _trained_fixture()
    latents, conditions = model.sample(range(200))
    assert np.array_equal(model.sample([31])[0][0], latents[31])
    assert np.corrcoef(latents[:, :2].T)[0, 1] > .95
    assert np.all(latents[:, 3] == 0)
    assert np.all((conditions[:, 0] >= .02) & (conditions[:, 0] <= .08))
    assert not set(model.metadata["train_sample_ids"]) & set(model.metadata["validation_sample_ids"])
    assert model.metadata["qualification"] == "unqualified_research_checkpoint"
    assert model.metadata["experimental_data_validation"] is False
    checkpoint = tmp_path / "learned.npz"
    model.save(checkpoint)
    digest = sha256(checkpoint.read_bytes()).hexdigest()
    restored = VPFPConditionalLatentDistribution.load(checkpoint, expected_sha256=digest)
    generator = VPFPLearnedBatchGenerator(restored, checkpoint_sha256=digest)
    batch = generator.generate([31, 5])
    assert batch["diagnostics"]["pde_solves"] == 0
    assert np.array_equal(batch["latents"][0], latents[31])
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        VPFPConditionalLatentDistribution.load(checkpoint, expected_sha256="c" * 64)
    with pytest.raises(FileExistsError):
        model.save(checkpoint)


def test_macro_only_candidate_completes_canonical_promotion_and_terminal():
    from pde_agent_verifier_solver.app.backend_run import BackendRunBinding
    from pde_agent_verifier_solver.engine.campaign import (
        AgentSpec, CampaignRecipe, CaseCounts, DevelopmentSamplingPolicy,
        EvaluatorPolicy, GeneratorRecipe, RoundSettings,
    )
    from pde_agent_verifier_solver.engine.runtime import AgentOutcome
    from pde_agent_verifier_solver.plugins.vpfp.canonical_binding import _BoundVPFPContinuousGenerator
    from pde_agent_verifier_solver.sdk import (
        CaseRecord, DatasetInput, DatasetKind, GeneratorCheckpoint,
        HiddenSuiteSpec, OfficialEvaluationSpec, PreparedDataset, SampleSource,
        SamplingStrategy,
    )

    class Truth:
        def __init__(self, case):
            self.case = case

        def answer(self, query):
            return VPFPHoldInitialMacroProgram().predict_observables(self.case, query)

    class Backend:
        supports_guidance = supports_perturbations = False

        def __init__(self):
            self.requests, self.released = [], []

        def generate(self, base, guidance, request, progress=None):
            assert request.settings["fixed_generator_sha256"] == "a" * 64
            self.requests.append(request)
            rows = []
            for index in range(request.count):
                case = replace(_public_case(), case_id=f"case-{request.seed}-{index}",
                               visibility=request.visibility, round_index=request.round_index)
                rows.append(CaseRecord(case, Truth(case)))
            return tuple(rows)

        def release_cases(self, case_ids):
            self.released.extend(case_ids)

    class Agent:
        def run(self, context):
            return AgentOutcome(Artifact(
                f"macro-{context.round_plan.round_index}", TaskKind.OBSERVABLE_MODEL,
                "macro only", VPFPHoldInitialMacroProgram(), "no full phase-space prediction",
                "predict_observables", context.round_plan.round_index, context.agent.agent_id,
            ))

    backend = Backend()
    identity = {"binding_sha256": "a" * 64}
    generator = _BoundVPFPContinuousGenerator(backend, identity)
    checkpoint = GeneratorCheckpoint(generator.info.program_id, identity)
    binding = BackendRunBinding(
        replace(build_vpfp_plugin(), generator=generator), checkpoint,
        PreparedDataset(DatasetInput("unit-online", DatasetKind.NUMERICAL), (), ()),
    )
    recipe = CampaignRecipe(
        name="VPFP macro-only contract test", round_count=2,
        agents=(AgentSpec("macro-agent", "macro-agent"),),
        default_round=RoundSettings(
            development_sampling=DevelopmentSamplingPolicy(
                strategy=SamplingStrategy.FIXED_RECIPE, allowed_sources=(SampleSource.BASE,),
                max_cases_per_agent_per_round=1, fixed_recipe=CaseCounts(base=1)),
            official_evaluation=OfficialEvaluationSpec(
                round=HiddenSuiteSpec(base=2), terminal=HiddenSuiteSpec(base=2)),
        ),
        generator=GeneratorRecipe(guidance_enabled=False, perturbations_enabled=False),
        evaluator=EvaluatorPolicy(AnswerQuery.grid(
            fields=("rho", "momentum_x", "momentum_y", "kinetic_energy"), t=2, x=2, y=2)),
    )
    result = binding.run(recipe, {"macro-agent": Agent()}, local_development=True)
    assert len(result.rounds) == len(result.terminal_round_reports) == 2
    assert all(row.promotion_decision is not None for row in result.rounds)
    assert result.selected.primary_error == pytest.approx(0.)
    assert backend.released
    assert not hasattr(result.selected.artifact.program, "predict_solution")
    with pytest.raises(ValueError, match="does not match"):
        generator.load(GeneratorCheckpoint(generator.info.program_id, {"binding_sha256": "b" * 64}), {})
