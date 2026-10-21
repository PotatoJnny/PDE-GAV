"""Real CPU NS numerical checkpoint parity, including a fresh Python process."""
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "legacy/ns/src"))
from pde_agent_verifier_solver.engine.common_evaluation import json_value
from pde_agent_verifier_solver.engine.guidance_controller import (
    GuidanceLifecycleController, GuidanceProtocolConfig, GuidanceRewardFeedback, GuidanceRewardRow,
)
from pde_agent_verifier_solver.plugins.ns.canonical_backend import build_canonical_structured_ns_generator
from pde_agent_verifier_solver.plugins.ns.structured_guidance_backend import StructuredNSGuidanceConfig
from pde_agent_verifier_solver.sdk import (
    CaseVisibility, GenerationKind, GenerationPurpose, GenerationRequest, GeneratorMode,
    GuidanceTrainingBudget,
)


def _artifacts(directory):
    from agent_verifier.neural_generator.structured_ns_answer_first import StructuredNSProjectionConfig
    from agent_verifier.neural_generator.structured_ns_distribution import (
        STRUCTURED_NS_FLOW_SCHEMA, StructuredCoefficientPCA, StructuredCoefficientFlowConfig,
        StructuredCoefficientRectifiedFlow,
    )
    directory.mkdir()
    projection = StructuredNSProjectionConfig(spatial_mode_x=1, spatial_mode_y=1, time_order=2)
    shape = (2, 3, 2, 2)
    torch.manual_seed(513)
    pca = StructuredCoefficientPCA(
        coordinate_mean=torch.randn(shape) * .0001, field_scale=torch.ones(shape),
        components=torch.randn((2, 24)) * .0001, score_scale=torch.ones(2),
        singular_values=torch.ones(2), total_centered_energy=torch.tensor(2.),
        coefficient_shape=shape, fit_case_count=3,
        source_signature={"projection_config": projection.to_dict()},
    )
    pca.save(directory / "pca.pt")
    config = StructuredCoefficientFlowConfig(feature_dim=2, width=8, depth=1, time_width=4)
    model = StructuredCoefficientRectifiedFlow.build(config)
    torch.save({"schema": STRUCTURED_NS_FLOW_SCHEMA, "pca_rank": 2,
                "pca_sha256": sha256((directory / "pca.pt").read_bytes()).hexdigest(),
                "model_config": config.to_dict(), "model_state_dict": model.state_dict()}, directory / "flow.pt")


def _bundle(artifacts, guidance_root):
    config = StructuredNSGuidanceConfig(training_steps=2, batch_size=2, width=8,
                                       depth=1, time_width=4, minimum_cases=2)
    protocol = GuidanceProtocolConfig(
        GuidanceTrainingBudget(30., "cpu", 1), config.sha256, 2, 2, 2, 30., "cpu",
        {"sampling_policy": "solver_reward_guided_rectified_flow", "flow_state": "raw",
         "integration_steps": 2, "guidance_strength": config.guidance_strength,
         "guidance_max_flow_ratio": config.guidance_max_flow_ratio},
    )
    return build_canonical_structured_ns_generator(
        generator_id="resume-cpu", generator_sha256="a" * 64,
        flow_checkpoint=artifacts / "flow.pt", pca_checkpoint=artifacts / "pca.pt",
        time_interval=(0., .12), viscosity=.01, perturbation_amplitude=1e-5,
        device="cpu", integration_steps=2, flow_state="raw", spectral_max_mode=2,
        guidance_artifact_root=guidance_root, guidance_protocol=protocol, guidance_config=config,
    )


def _request(kind=GenerationKind.BASE, *, index=1, seed=101, anchors=(), count=2):
    return GenerationRequest(index, count, GeneratorMode.ONLINE, kind, CaseVisibility.HIDDEN,
                             seed, anchors=anchors, purpose=GenerationPurpose.PROMOTION)


def _coefficients(records):
    return [(row.case.case_id, row.truth._answer.raw_answer.coefficients.psi.tolist(),
             row.truth._answer.raw_answer.coefficients.pressure.tolist()) for row in records]


def _train(bundle, records, source=0):
    feedback = GuidanceRewardFeedback(source, source + 1, "b" * 64, "a" * 64, "c" * 64,
                  tuple(GuidanceRewardRow(row.case, float(index + 1), "valid", {})
                        for index, row in enumerate(records)))
    return GuidanceLifecycleController(config=bundle.backend.guidance_protocol,
                backend=bundle.guidance_backend).train_boundary(feedback, controller_training_seed=203 + source)


def _forbid(*args, **kwargs):
    raise AssertionError("checkpoint attempted numerical generation or truth evaluation")


def _fresh_process_restore(root):
    # Invoked by a new Python interpreter; nothing live is inherited.
    torch.set_num_threads(1)
    root = Path(root)
    bundle = _bundle(root / "artifacts", root / "guide-restored")
    saved = json.loads((root / "checkpoint.json").read_text())
    generator = bundle.generator
    generator.restore_runtime_state(saved["runtime"], root / "objects")
    state = generator.restore_generator_state(saved["state"], root / "objects")
    records = tuple(generator.restore_case_record(row, root / "objects") for row in saved["records"])
    assert _coefficients(records) == [tuple(row) for row in saved["resident_coefficients"]]
    assert state.guidance.snapshot.snapshot_sha256 == saved["guidance_sha"]
    with pytest.raises(RuntimeError, match="already attempted"):
        bundle.guidance_backend.train_guidance(state.guidance.snapshot.training_request)
    # The runner restores RNG only after all Base/Guide loading. Saving RNG is
    # intentionally outside the NS codec so it has a single owner.
    torch.set_rng_state(torch.tensor(saved["rng"], dtype=torch.uint8))
    assert torch.rand(4).tolist() == saved["next_rng_values"]
    base = generator.sample(state, _request(seed=201))
    guided = generator.sample(state, _request(GenerationKind.GUIDED, seed=301))
    perturbed = generator.sample(state, _request(GenerationKind.PERTURBED, seed=401,
                                                anchors=tuple(row.case for row in base)))
    actual = {"base": _coefficients(base), "guided": _coefficients(guided), "perturbed": _coefficients(perturbed)}
    assert json_value(actual) == saved["future"]
    # Previously pending feedback remains usable for the next completed
    # Guidance boundary after its original process has disappeared.
    guide2 = _train(bundle, records, source=1)
    assert guide2.snapshot.training_request.target_round_index == 2
    model = bundle.guidance_backend._model_cache[guide2.snapshot.snapshot_sha256]
    assert {name: value.tolist() for name, value in model.state_dict().items()} == saved["next_guidance_weights"]


def test_real_ns_cpu_checkpoint_continues_in_new_process(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    _artifacts(tmp_path / "artifacts")
    bundle = _bundle(tmp_path / "artifacts", tmp_path / "guide-original")
    generator = bundle.generator
    state = generator.load(bundle.checkpoint, {})
    records = generator.sample(state, _request(index=0))
    guide = _train(bundle, records)
    state = replace(state, guidance=guide)
    # Populate the cache in both uninterrupted and resumed runs before the
    # post-load RNG boundary. It must not change sampling math or Case IDs.
    bundle.guidance_backend.open_guidance(guide.snapshot)
    objects = tmp_path / "objects"
    with monkeypatch.context() as patch:
        patch.setattr(bundle.components.flow_sampler, "sample_many_with_guidance", _forbid)
        from pde_agent_verifier_solver.plugins.ns.continuous_backend import _NSVelocityTruth
        patch.setattr(_NSVelocityTruth, "answer", _forbid)
        runtime = generator.export_runtime_state(objects)
        state_payload = generator.export_generator_state(state, objects)
        case_payloads = [generator.export_case_record(row, objects) for row in records]
        before = {path.name: path.stat().st_mtime_ns for path in objects.iterdir()}
        # Subsequent progress checkpoints reuse frozen tensors without CPU
        # conversion, tensor serialization, or re-reading Guidance file bytes.
        import pde_agent_verifier_solver.plugins.ns.resume as resume
        patch.setattr(resume, "_field_payload", _forbid)
        patch.setattr(resume, "_put_tensors", _forbid)
        assert runtime == generator.export_runtime_state(objects)
        assert case_payloads == [generator.export_case_record(row, objects) for row in records]
        assert before == {path.name: path.stat().st_mtime_ns for path in objects.iterdir()}
    rng = torch.get_rng_state().tolist()
    next_rng_values = torch.rand(4).tolist()
    base = generator.sample(state, _request(seed=201))
    guided = generator.sample(state, _request(GenerationKind.GUIDED, seed=301))
    perturbed = generator.sample(state, _request(GenerationKind.PERTURBED, seed=401,
                                                anchors=tuple(row.case for row in base)))
    payload = {"runtime": runtime, "state": state_payload, "records": case_payloads,
               "resident_coefficients": _coefficients(records), "guidance_sha": guide.snapshot.snapshot_sha256,
               "rng": rng, "next_rng_values": next_rng_values,
               "future": {"base": _coefficients(base), "guided": _coefficients(guided),
                          "perturbed": _coefficients(perturbed)}}
    guide2 = _train(bundle, records, source=1)
    model2 = bundle.guidance_backend._model_cache[guide2.snapshot.snapshot_sha256]
    payload["next_guidance_weights"] = {name: value.tolist() for name, value in model2.state_dict().items()}
    (tmp_path / "checkpoint.json").write_text(json.dumps(payload))
    # The checkpoint must preserve frozen Guidance even when the original
    # per-job artifact directory no longer exists at its URI.
    import shutil
    shutil.rmtree(tmp_path / "guide-original")
    repo = Path(__file__).resolve().parents[1]
    environment = {**os.environ, "PYTHONPATH": f"{repo}:{repo / 'src'}:{repo / 'legacy/ns/src'}",
                   "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    completed = subprocess.run([sys.executable, "-c", "from tests.test_ns_resume import _fresh_process_restore; import sys; _fresh_process_restore(sys.argv[1])", str(tmp_path)],
                               env=environment, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_resume_rejects_changed_sampling_contract_and_asset(tmp_path):
    torch.set_num_threads(1)
    _artifacts(tmp_path / "artifacts")
    bundle = _bundle(tmp_path / "artifacts", tmp_path / "guide-a")
    payload = bundle.generator.export_runtime_state(tmp_path / "objects")
    other = _bundle(tmp_path / "artifacts", tmp_path / "guide-b")
    altered = {**payload, "maximum_points": 7}
    with pytest.raises(ValueError, match="query limits"):
        other.generator.restore_runtime_state(altered, tmp_path / "objects")
    changed_device = {**payload, "flow_identity": {**payload["flow_identity"], "device_type": "cuda"}}
    with pytest.raises(ValueError, match="sampling settings"):
        other.generator.restore_runtime_state(changed_device, tmp_path / "objects")
    asset = tmp_path / "objects" / payload["numerical"]["asset"]
    asset.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash differs"):
        other.generator.restore_runtime_state(payload, tmp_path / "objects")

# These fixtures prepare a tiny physically real immutable B/G/P bank. The
# resumed Generator below is the real RF/PCA backend, not the fixture backend.
from tests.test_ns_mixed_evaluation_preparation import cpu_field_factory, prepared_suite, _controller


def test_mixed_bank_resume_preserves_cursor_and_finite_truth_without_recompute(prepared_suite, tmp_path, monkeypatch):
    from pde_agent_verifier_solver.plugins.ns.mixed_evaluation import SharedNSMixedEvaluationGenerator
    from pde_agent_verifier_solver.plugins.ns.continuous_backend import StructuredNSMMSAnswer
    from tests.test_ns_mixed_evaluation import _request as bank_request
    _artifacts(tmp_path / "artifacts")
    controller, _ = _controller(prepared_suite["field_factory"])
    settings = {**controller.campaign.task_settings, "default_query": controller.campaign.evaluator.default_query}
    bundle = _bundle(tmp_path / "artifacts", tmp_path / "guide-mixed-a")
    wrapper = SharedNSMixedEvaluationGenerator(bundle.generator, prepared_suite["path"],
                 tmp_path / "audit-a.jsonl", controller.campaign.master_seed, settings)
    state = wrapper.load(bundle.checkpoint, {})
    entry = prepared_suite["suite"]["official_rounds"][0]
    schedule = entry["sampling_requests"]
    first = wrapper.sample(state, bank_request(schedule[0], {}))
    objects = tmp_path / "objects"
    state_payload = wrapper.export_generator_state(state, objects)
    saved = wrapper.export_runtime_state(objects)
    records_payload = [wrapper.export_case_record(record, objects) for record in first]
    reference = [json_value(record.truth.reference) for record in first]
    restored_bundle = _bundle(tmp_path / "artifacts", tmp_path / "guide-mixed-b")
    restored = SharedNSMixedEvaluationGenerator(restored_bundle.generator, prepared_suite["path"],
                 tmp_path / "audit-b.jsonl", controller.campaign.master_seed, settings)
    with monkeypatch.context() as patch:
        patch.setattr(StructuredNSMMSAnswer, "evaluate_velocity", _forbid)
        patch.setattr(StructuredNSMMSAnswer, "evaluate_forcing", _forbid)
        patch.setattr(restored_bundle.components.flow_sampler, "sample_many_with_guidance", _forbid)
        restored.restore_runtime_state(saved, objects)
        resumed_state = restored.restore_generator_state(state_payload, objects)
        rows = [restored.restore_case_record(record, objects) for record in records_payload]
        assert [json_value(record.truth.reference) for record in rows] == reference
        issued = {record.case.case_id: record for record in rows}
        for request in schedule[1:]:
            records = restored.sample(resumed_state, bank_request(request, issued))
            assert [record.case.case_id for record in records] == request["case_ids"]
            issued.update({record.case.case_id: record for record in records})
    assert restored._official[1].next_request == len(schedule)
    restored.release(resumed_state, tuple(issued))
    assert not restored._source._issued
    # Released/completed cursors must survive too; they cannot restart at zero.
    released = restored.export_runtime_state(objects)
    final_bundle = _bundle(tmp_path / "artifacts", tmp_path / "guide-mixed-c")
    final = SharedNSMixedEvaluationGenerator(final_bundle.generator, prepared_suite["path"],
                tmp_path / "audit-c.jsonl", controller.campaign.master_seed, settings)
    final.restore_runtime_state(released, objects)
    assert final._official[1].bank is None
    assert final._official[1].next_request == len(schedule)
    assert final._official[1].issued == set(issued)
