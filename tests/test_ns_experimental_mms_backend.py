"""Reconstructed fields use the actual numerical NS backend and artifact formats."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pde_agent_verifier_solver.plugins.ns.experimental_mms import (
    ExperimentalMMSDatasetRegistry, assemble_experimental_mms,
    prepare_experimental_mms_answers, coefficient_sha256,
    _answer as raw_answer, _coefficient_batch,
)
from pde_agent_verifier_solver.plugins.ns.completed_fields import (
    field_from_coefficients, field_to_state, field_from_state, numerical_symbols,
)
from pde_agent_verifier_solver.sdk import (
    Artifact, CaseFeedback, CaseSource, CaseVisibility, DatasetInput, DatasetKind,
    GenerationKind, GenerationPurpose, GenerationRequest, GeneratorMode, RoundFeedback, TaskKind,
)


@pytest.fixture(scope="module", autouse=True)
def _small_cpu_work():
    torch = pytest.importorskip("torch")
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def _answer(index, *, viscosity=.01):
    numerical_symbols()
    from agent_verifier.neural_generator.continuous_geometry import ContinuousGeometrySpec
    rng = np.random.default_rng(index)
    values = np.zeros((2, 3, 3, 3))
    values[0] = rng.normal(size=(3, 3, 3)) * .03
    values[0, :, 0, 0] = 0
    return field_from_coefficients(values,
        config={"spatial_mode_x": 2, "spatial_mode_y": 2, "time_order": 2, "spatial_extension": .25},
        geometry=ContinuousGeometrySpec((0., 2., -1., 1.), (), outer_boundary="manufactured_trace").to_dict(),
        time_interval=(0., .5), viscosity=viscosity)


def _prepared(backend=None, *, viscosity=.01):
    split = {
        "generator_training": tuple(_answer(i, viscosity=viscosity) for i in range(8)),
        "generator_validation": tuple(_answer(i, viscosity=viscosity) for i in range(8, 10)),
        "development": (_answer(10, viscosity=viscosity), _answer(11, viscosity=viscosity)),
        "verifier": (_answer(12, viscosity=viscosity),), "final_test": (_answer(13, viscosity=viscosity),),
    }
    return prepare_experimental_mms_answers(DatasetInput("synthetic-experimental-mms", DatasetKind.EXPERIMENTAL),
                                             split, backend=backend)


def _settings():
    return {"device": "cpu", "integration_steps": 2, "flow_state": "ema",
            "integration_method": "heun", "perturbation_amplitude": 1e-4,
            "spectral_max_mode": 3}


def _artifacts(directory, dataset):
    """Tiny frozen numerical-format test weights, without RF optimization."""
    torch = pytest.importorskip("torch")
    numerical_symbols()
    from agent_verifier.neural_generator.structured_ns_distribution import (
        STRUCTURED_NS_FLOW_SCHEMA, StructuredCoefficientPCA, StructuredCoefficientFlowConfig,
        StructuredCoefficientRectifiedFlow,
    )
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    train = _coefficient_batch([raw_answer(record) for record in dataset.generator_training_cases])
    validation = _coefficient_batch([raw_answer(record) for record in dataset.generator_validation_cases])
    template = raw_answer(dataset.generator_training_cases[0])
    signature = {"projection_config": template.config.to_dict(), "coefficient_shape": list(train.shape[1:]),
        "training_coefficients_sha256": coefficient_sha256(train),
        "validation_coefficients_sha256": coefficient_sha256(validation),
        "geometry": template.geometry_spec.to_dict(), "time_interval": [template.t0, template.t1]}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        pca = StructuredCoefficientPCA.fit(train, rank=3, source_signature=signature, seed=19)
        pca.save(directory / "pca.pt")
        config = StructuredCoefficientFlowConfig(3, width=8, depth=1, time_width=4)
        model = StructuredCoefficientRectifiedFlow.build(config)
    torch.save({"schema": STRUCTURED_NS_FLOW_SCHEMA, "pca_rank": 3,
        "pca_sha256": hashlib.sha256((directory / "pca.pt").read_bytes()).hexdigest(),
        "model_config": config.to_dict(), "model_state_dict": model.state_dict(),
        "ema_state_dict": model.state_dict()}, directory / "flow.pt")
    return directory / "flow.pt", directory / "pca.pt"


def _guidance():
    from pde_agent_verifier_solver.plugins.ns.structured_guidance_backend import StructuredNSGuidanceConfig
    from pde_agent_verifier_solver.engine.guidance_controller import GuidanceProtocolConfig
    from pde_agent_verifier_solver.sdk import GuidanceTrainingBudget
    config = StructuredNSGuidanceConfig(training_steps=2, batch_size=2, width=8,
                                       depth=1, time_width=4, minimum_cases=2)
    protocol = GuidanceProtocolConfig(GuidanceTrainingBudget(30., "cpu", 1),
        config.sha256, 2, 2, 2, 30., "cpu", {
            "sampling_policy": "solver_reward_guided_rectified_flow", "flow_state": "ema",
            "integration_steps": 2, "guidance_strength": config.guidance_strength,
            "guidance_max_flow_ratio": config.guidance_max_flow_ratio})
    return config, protocol


def _assembly(directory, dataset=None, *, artifacts=None, guidance=True):
    dataset = dataset or _prepared()
    directory = Path(directory)
    flow, pca = artifacts or _artifacts(directory / "artifacts", dataset)
    config, protocol = _guidance() if guidance else (None, None)
    return assemble_experimental_mms(dataset, flow_checkpoint=flow, pca_checkpoint=pca,
        generator_settings=_settings(), guidance_artifact_root=directory / "guidance" if guidance else None,
        guidance_config=config, guidance_protocol=protocol)


def _request(kind=GenerationKind.BASE, *, seed=73, index=0, anchors=()):
    return GenerationRequest(round_index=index, count=2, mode=GeneratorMode.ONLINE,
        kind=kind, visibility=CaseVisibility.HIDDEN, purpose=GenerationPurpose.PROMOTION,
        seed=seed, anchors=anchors)


def test_original_cases_use_shared_answer_and_keep_coefficients_private():
    registry = ExperimentalMMSDatasetRegistry()
    dataset = _prepared(registry)
    assert not hasattr(registry, "train_base") and not hasattr(registry, "generate")
    for record in (*dataset.development_cases, *dataset.hidden_cases, *dataset.generator_training_cases):
        public = record.case.public_data
        assert record.case.source is CaseSource.INITIAL
        assert "observations" not in public and "coefficients" not in json.dumps(public)
        assert public["boundary_condition"]["kind"] == "manufactured_velocity_trace"
        session = registry.public_session_factory(record.case)
        points = [[0., 0., 0.], [.25, 2., .3]]
        np.testing.assert_allclose(session.query_boundary_velocity(points),
                                   record.truth._answer.evaluate_velocity(points))
        assert np.linalg.norm(session.query_boundary_velocity(points)) > 0


def test_assembly_uses_actual_numerical_backend_model_and_sampler(tmp_path):
    from pde_agent_verifier_solver.plugins.ns.canonical_backend import CanonicalStructuredNSBackend, StructuredNSBaseState
    from pde_agent_verifier_solver.plugins.ns.generator import NSStructuredGenerator
    from pde_agent_verifier_solver.plugins.ns.structured_flow_runtime import LegacyStructuredNSFlowSampler
    assembly = _assembly(tmp_path)
    assert type(assembly.generator) is NSStructuredGenerator
    assert type(assembly.generator.backend) is CanonicalStructuredNSBackend
    assert type(assembly.checkpoint.payload) is StructuredNSBaseState
    sampler = assembly.bundle.components.flow_sampler
    assert type(sampler) is LegacyStructuredNSFlowSampler
    assert type(sampler._model).__name__ == "_StructuredCoefficientRectifiedFlow"
    assert sampler.artifact_metadata["flow_state"] == "ema"
    assert sampler.artifact_metadata["integration_method"] == "heun"
    assert assembly.bundle.components.perturbation_sampler.family == "spectral_complement"
    with pytest.raises(RuntimeError, match="already-frozen checkpoint"):
        assembly.generator.train(assembly.prepared_dataset, {})
    state = assembly.generator.load(assembly.checkpoint, {})
    first = assembly.generator.sample(state, _request())
    assembly.generator.release(state, tuple(row.case.case_id for row in first))
    second = assembly.generator.sample(state, _request())
    for left, right in zip(first, second):
        np.testing.assert_array_equal(raw_answer(left).coefficients.psi, raw_answer(right).coefficients.psi)


def test_original_neural_guidance_is_frozen_and_samples_solver_cases(tmp_path):
    from pde_agent_verifier_solver.engine.guidance_controller import FrozenGuidanceBoundary
    from pde_agent_verifier_solver.plugins.ns.structured_guidance_backend import StructuredNSMacroGuidanceBackend
    assembly = _assembly(tmp_path)
    generator = assembly.generator
    state = generator.load(assembly.checkpoint, {})
    parents = generator.sample(state, _request())
    solver = Artifact("probe", TaskKind.SOLVER, "Probe", SimpleNamespace(), "", "", 0, "local")
    feedback = RoundFeedback(0, solver, tuple(CaseFeedback(row.case, .1+index*.2, {})
                                             for index, row in enumerate(parents)))
    guidance = generator.backend.fit_guidance(state.base, None, feedback, {"controller_seed": 191})
    assert type(assembly.bundle.guidance_backend) is StructuredNSMacroGuidanceBackend
    assert isinstance(guidance, FrozenGuidanceBoundary)
    assert guidance.snapshot.training_request.target_round_index == 1
    guided = generator.sample(replace(state, guidance=guidance), _request(GenerationKind.GUIDED, index=1))
    assert len(guided) == 2
    assert all(row.case.public_data["distribution_lineage"]["kind"] == "frozen_reward_guided" for row in guided)
    assert all(row.case.public_data["boundary_condition"]["access"] == "query_boundary_velocity" for row in guided)
    with pytest.raises(RuntimeError, match="already attempted"):
        generator.backend.fit_guidance(state.base, None, feedback, {"controller_seed": 191})


def test_original_spectral_perturbation_preserves_initial_data_and_low_band(tmp_path):
    assembly = _assembly(tmp_path, guidance=False)
    generator = assembly.generator
    state = generator.load(assembly.checkpoint, {})
    parents = generator.sample(state, _request())
    children = generator.sample(state, _request(GenerationKind.PERTURBED, seed=89,
                                               anchors=tuple(row.case for row in parents)))
    for parent, child in zip(parents, children):
        a, b = raw_answer(parent), raw_answer(child)
        assert b.config.spatial_mode_x == 3 and a.config.spatial_mode_x == 2
        assert b.config.spatial_extension == a.config.spatial_extension == .25
        assert b.geometry_spec == a.geometry_spec
        original, changed = _coefficient_batch([a])[0], _coefficient_batch([b])[0]
        np.testing.assert_array_equal(changed[:, :, :3, :3], original)
        points = [[0., 0., 0.], [0., .4, .3], [0., 2., -.1]]
        np.testing.assert_allclose(child.truth._answer.evaluate_velocity(points),
                                   parent.truth._answer.evaluate_velocity(points), atol=1e-13, rtol=1e-12)
        assert child.case.public_data["perturbation"]["family"] == "spectral_complement"
        assert child.case.parent_case_id == parent.case.case_id
        np.testing.assert_allclose(generator.backend.public_session_factory(child.case).query_forcing([[.2, .4, .3]]),
                                   child.truth._answer.evaluate_forcing([[.2, .4, .3]]))


def test_shared_resume_preserves_prepared_originals_and_future_generation(tmp_path):
    first = _assembly(tmp_path / "first", guidance=False)
    generator = first.generator
    state = generator.load(first.checkpoint, {})
    generator.sample(state, _request())
    objects = tmp_path / "objects"
    runtime = generator.export_runtime_state(objects)
    saved_state = generator.export_generator_state(state, objects)
    original = first.prepared_dataset.development_cases[0]
    saved_case = generator.export_case_record(original, objects)
    artifacts = (first.bundle.components.flow_sampler.flow_checkpoint, first.bundle.components.flow_sampler.pca_checkpoint)
    second = _assembly(tmp_path / "second", first.prepared_dataset, artifacts=artifacts, guidance=False)
    second.generator.restore_runtime_state(runtime, objects)
    restored_state = second.generator.restore_generator_state(saved_state, objects)
    replay = second.generator.restore_case_record(saved_case, objects)
    assert replay.case == original.case
    assert field_to_state(raw_answer(replay)) == field_to_state(raw_answer(original))
    expected = generator.sample(state, _request(seed=103))
    actual = second.generator.sample(restored_state, _request(seed=103))
    for left, right in zip(expected, actual):
        assert left.case == right.case
        np.testing.assert_array_equal(raw_answer(left).coefficients.psi, raw_answer(right).coefficients.psi)
    # A new generated Case is still forbidden before restoring a snapshot.
    with pytest.raises(ValueError, match="before creating any new Cases"):
        second.generator.restore_runtime_state(runtime, objects)


def test_shared_resume_rejects_changed_prepared_original(tmp_path):
    first = _assembly(tmp_path / "first", guidance=False)
    runtime = first.generator.export_runtime_state(tmp_path / "objects")
    artifacts = (first.bundle.components.flow_sampler.flow_checkpoint, first.bundle.components.flow_sampler.pca_checkpoint)
    second = _assembly(tmp_path / "second", first.prepared_dataset, artifacts=artifacts, guidance=False)
    key = first.prepared_dataset.development_cases[0].case.case_id
    raw = second.generator.backend.case_source._issued[key].raw_answer
    raw.coefficients.psi[0, 0] += 1
    with pytest.raises(ValueError, match="Original differs"):
        second.generator.restore_runtime_state(runtime, tmp_path / "objects")


@pytest.mark.parametrize("settings", [{"flow_step": 2}, {"guidance_ridge": 1}, {"perturbation_scale": .15}])
def test_obsolete_or_unknown_settings_fail_before_checkpoint_io(tmp_path, settings):
    with pytest.raises(ValueError, match="unsupported numerical NS runtime"):
        assemble_experimental_mms(_prepared(), flow_checkpoint=tmp_path / "missing-flow.pt",
                                 pca_checkpoint=tmp_path / "missing-pca.pt", generator_settings=settings)


def test_training_coefficient_binding_rejects_wrong_corpus_but_not_viscosity(tmp_path):
    dataset = _prepared()
    artifacts = _artifacts(tmp_path / "artifacts", dataset)
    wrong = replace(dataset, generator_training_cases=dataset.generator_training_cases[:-1])
    _assembly(tmp_path / "wrong", wrong, artifacts=artifacts, guidance=False)
    changed_physics = _prepared(viscosity=.03)
    assembled = _assembly(tmp_path / "viscosity", changed_physics, artifacts=artifacts, guidance=False)
    assert assembled.bundle.components.flow_sampler._viscosity == .03


def test_old_experimental_only_checkpoint_is_explicitly_rejected(tmp_path):
    torch = pytest.importorskip("torch")
    dataset = _prepared()
    flow, pca = _artifacts(tmp_path / "artifacts", dataset)
    torch.save({"schema": "ns_experimental_mms_coefficient_flow_v1"}, flow)
    with pytest.raises(ValueError, match="unexpected structured NS flow checkpoint schema"):
        _assembly(tmp_path / "old", dataset, artifacts=(flow, pca), guidance=False)


def test_shared_field_state_keeps_basis_boundary_and_pressure(tmp_path):
    original = _answer(19)
    payload = json.loads(json.dumps(field_to_state(original)))
    restored = field_from_state(payload)
    assert field_to_state(restored) == payload
    assert restored.config.spatial_extension == .25
    assert restored.geometry_spec.outer_boundary == "manufactured_trace"
    assert not np.any(restored.coefficients.pressure.numpy())
