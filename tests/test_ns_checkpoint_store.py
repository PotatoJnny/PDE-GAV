"""Actual NS continuation through the Controller store, across processes/devices."""
from dataclasses import replace
import gc
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
from types import MappingProxyType
from unittest.mock import patch
import weakref

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from pde_agent_verifier_solver.engine.campaign_continuation import CampaignContinuation
from pde_agent_verifier_solver.engine.checkpoint_store import CheckpointStore, capture_rng, restore_rng
from pde_agent_verifier_solver.engine.common_evaluation import json_value
from pde_agent_verifier_solver.engine.python_candidate import PythonCandidateProgram
from pde_agent_verifier_solver.plugins.ns.case_bank import _query
from pde_agent_verifier_solver.plugins.ns.mixed_evaluation import SharedNSMixedEvaluationGenerator
from pde_agent_verifier_solver.sdk import (
    AnswerQuery, AnswerView, Artifact, ArtifactEvaluation, CaseEvaluation, CaseRecord,
    EvaluationPlan, GenerationKind, GeneratorState, TaskKind, inspect_candidate_bundle,
)
from tests.test_ns_resume import _artifacts, _bundle, _coefficients, _forbid, _request, _train
from tests.test_ns_mixed_evaluation_preparation import cpu_field_factory, prepared_suite
from tests.test_ns_mixed_evaluation import _request as bank_request
from tests.test_python_candidate import _candidate


def _device_bundle(artifacts, guide_root, device):
    # Reuse the same small actual RF/PCA/Guidance fixture, changing only its
    # physical allocation. The checkpoint's scientific configuration is fixed.
    import tests.test_ns_resume as helpers
    build = helpers.build_canonical_structured_ns_generator
    def on_device(**kwargs):
        return build(**{**kwargs, "device": device})
    with patch.object(helpers, "build_canonical_structured_ns_generator", on_device):
        return _bundle(artifacts, guide_root)


def _draw_global_rng():
    result = {"python": [random.random() for _ in range(4)],
              "numpy": np.random.random(4).tolist(), "torch": torch.rand(4).tolist()}
    if torch.cuda.is_initialized():
        result["cuda"] = [torch.rand(4, device=f"cuda:{index}").cpu().tolist()
                          for index in range(torch.cuda.device_count())]
    return result


def _future(bundle, state):
    generator = bundle.generator
    base = generator.sample(state, _request(index=1, seed=201))
    guided = generator.sample(state, _request(GenerationKind.GUIDED, index=1, seed=301))
    perturbed = generator.sample(state, _request(GenerationKind.PERTURBED, index=1, seed=401,
                                             anchors=tuple(row.case for row in base)))
    return json_value({"base": _coefficients(base), "guided": _coefficients(guided),
                       "perturbed": _coefficients(perturbed)})


def _settings(prepared):
    campaign = prepared["controller"].campaign
    return {**campaign.task_settings, "default_query": campaign.evaluator.default_query}


def _restore_complete_store(root):
    """Child interpreter entry point: load real backend plus the whole typed tree."""
    root = Path(root)
    torch.set_num_threads(1)
    expected = json.loads((root / "expected.json").read_text())
    device = expected["restore_device"]
    if device.startswith("cuda"):
        torch.cuda.set_device(device)
    bundle = _device_bundle(root / "artifacts", root / "guide-restored", device)
    settings = expected["settings"]
    settings["default_query"] = _query(settings["default_query"])
    wrapper = SharedNSMixedEvaluationGenerator(bundle.generator, expected["suite"],
               root / "audit-restored.jsonl", 101, settings)
    store = CheckpointStore(root / "store", identity=expected["identity"], generator=wrapper)
    from pde_agent_verifier_solver.plugins.ns.continuous_backend import (
        StructuredNSMMSAnswer, _NSVelocityTruth,
    )
    # Loading saved work and consuming the rest of the frozen bank cannot
    # invoke a sampler or evaluate the continuous reference again.
    with patch.object(bundle.components.flow_sampler, "sample_many_with_guidance", _forbid), \
         patch.object(_NSVelocityTruth, "answer", _forbid), \
         patch.object(StructuredNSMMSAnswer, "evaluate_velocity", _forbid), \
         patch.object(StructuredNSMMSAnswer, "evaluate_forcing", _forbid):
        restored = store.load()
        assert _draw_global_rng() == expected["rng_draws"]
        assert isinstance(restored, CampaignContinuation)
        state, controller = restored.state, restored.controller_state
        assert isinstance(state["generator_state"], GeneratorState)
        assert state["generator_state"] is controller["generator_alias"]
        assert state["prior_generator_state"].guidance is None
        assert state["generator_state"].guidance.snapshot.snapshot_sha256 == expected["guide_sha"]
        assert isinstance(state["pending_cases"][0], CaseRecord)
        assert state["pending_cases"][0] is controller["pending_alias"]
        assert state["frozen_cases"][0] is controller["frozen_alias"]
        assert isinstance(state["plan"], EvaluationPlan)
        assert state["plan"] is controller["plan_alias"]
        assert isinstance(state["truth"], AnswerView)
        assert state["truth"] is controller["truth_alias"]
        evaluation = state["evaluation"]
        assert isinstance(evaluation, ArtifactEvaluation)
        assert isinstance(evaluation.cases[0], CaseEvaluation)
        assert evaluation.cases[0] is controller["score_alias"]
        assert evaluation.cases[0].prediction is state["truth"]
        assert evaluation.cases[0].primary_error == 0.125
        assert evaluation.artifact.program is state["program_alias"]
        assert isinstance(evaluation.artifact.program, PythonCandidateProgram)
        assert json_value(state["plan"]) == expected["plan"]
        assert json_value(state["truth"]) == expected["truth"]
        assert json_value(state["frozen_cases"][0].truth.reference) == expected["truth"]
        assert controller["registry"] == {row.case.case_id for row in state["pending_cases"]}
        assert isinstance(controller["memory"], MappingProxyType)
        assert json_value(_coefficients(state["pending_cases"])) == expected["pending_coefficients"]
        # Live CUDA fields relocate to the new allocation's logical GPU;
        # frozen bank fields deliberately retain their original CPU device.
        for row in state["pending_cases"]:
            assert str(row.truth._answer.raw_answer.coefficients.psi.device) == device
        assert state["frozen_cases"][0].truth._answer.raw_answer.coefficients.psi.device.type == "cpu"
        issued = {row.case.case_id: row for row in state["frozen_cases"]}
        for request in expected["schedule"][1:]:
            rows = wrapper.sample(state["generator_state"], bank_request(request, issued))
            assert [row.case.case_id for row in rows] == request["case_ids"]
            issued.update((row.case.case_id, row) for row in rows)
        assert wrapper._official[1].next_request == len(expected["schedule"])
    query = AnswerQuery.at_points(fields=("u",), points=({"x": .25},))
    answer = evaluation.artifact.program.prepare_solution(state["pending_cases"][0].case, None).evaluate(query)
    assert answer.fields["u"] == (2.,)
    assert _future(bundle, state["generator_state"]) == expected["future"]
    with pytest.raises(RuntimeError, match="already attempted"):
        bundle.guidance_backend.train_guidance(state["generator_state"].guidance.snapshot.training_request)
    guide2 = _train(bundle, state["pending_cases"], source=1)
    model2 = bundle.guidance_backend._model_cache[guide2.snapshot.snapshot_sha256]
    assert {key: value.tolist() for key, value in model2.state_dict().items()} == expected["next_guide_weights"]


@pytest.mark.parametrize("save_device,restore_device", [
    pytest.param("cpu", "cpu", id="cpu"),
    pytest.param("cuda:0", "cuda:0", id="cuda-same", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="requires CUDA")),
    pytest.param("cuda:0", "cuda:1", id="cuda-relocated", marks=pytest.mark.skipif(
        torch.cuda.device_count() < 2, reason="requires two CUDA devices")),
])
def test_real_ns_checkpoint_store_new_process(prepared_suite, tmp_path, monkeypatch,
                                               save_device, restore_device):
    torch.set_num_threads(1)
    if save_device.startswith("cuda"):
        torch.cuda.set_device(save_device)
    _artifacts(tmp_path / "artifacts")
    bundle = _device_bundle(tmp_path / "artifacts", tmp_path / "guide-original", save_device)
    settings = _settings(prepared_suite)
    wrapper = SharedNSMixedEvaluationGenerator(bundle.generator, prepared_suite["path"],
                 tmp_path / "audit-original.jsonl", 101, settings)
    prior = wrapper.load(bundle.checkpoint, {})
    pending = bundle.generator.sample(prior, _request(index=0))
    guide = _train(bundle, pending)
    current = replace(prior, guidance=guide)
    bundle.guidance_backend.open_guidance(guide.snapshot)
    schedule = prepared_suite["suite"]["official_rounds"][0]["sampling_requests"]
    frozen = wrapper.sample(current, bank_request(schedule[0], {}))
    plan, truth = frozen[0].truth.plan, frozen[0].truth.reference
    candidate_path = _candidate(tmp_path / "solver")
    program = PythonCandidateProgram(candidate_path,
                  expected_tree_sha256=inspect_candidate_bundle(candidate_path).tree_sha256)
    artifact = Artifact("solver-r1", TaskKind.SOLVER, "Checkpoint solver", program,
                        "frozen result", "prepare_solution", 1, "agent-1")
    score = CaseEvaluation(frozen[0].case.case_id, .125, {"relative_l2": .125}, truth)
    evaluation = ArtifactEvaluation(artifact, .125, {"relative_l2": .125}, (score,))
    continuation = CampaignContinuation(
        {"generator_state": current, "prior_generator_state": prior,
         "pending_cases": tuple(pending), "frozen_cases": list(frozen),
         "plan": plan, "truth": truth, "evaluation": evaluation, "program_alias": program},
        {"generator_alias": current, "pending_alias": pending[0], "frozen_alias": frozen[0],
         "plan_alias": plan, "truth_alias": truth, "score_alias": score,
         "registry": {row.case.case_id for row in pending},
         "memory": MappingProxyType({"round": 1, "completed_batches": 1})})
    identity = {"fixture": "real-ns-cross-process-v1", "seed": 101}
    store = CheckpointStore(tmp_path / "store", identity=identity, generator=wrapper)
    random.seed(611)
    np.random.seed(612)
    torch.manual_seed(613)
    boundary_rng = capture_rng()
    expected_rng = _draw_global_rng()
    restore_rng(boundary_rng)
    store.save(continuation)
    assert _draw_global_rng() == expected_rng
    restore_rng(boundary_rng)
    before = {path.name: (path.stat().st_mtime_ns, path.stat().st_size)
              for path in store.objects.iterdir()}
    import pde_agent_verifier_solver.plugins.ns.resume as resume
    with monkeypatch.context() as scoped:
        scoped.setattr(resume, "_field_payload", _forbid)
        scoped.setattr(resume, "_put_tensors", _forbid)
        store.save(continuation)
    assert before == {path.name: (path.stat().st_mtime_ns, path.stat().st_size)
                      for path in store.objects.iterdir()}
    assert store.sequence == 2
    assert _draw_global_rng() == expected_rng
    expected = {"identity": identity, "settings": json_value(settings), "suite": str(prepared_suite["path"]),
                "schedule": schedule, "restore_device": restore_device, "rng_draws": expected_rng,
                "guide_sha": guide.snapshot.snapshot_sha256, "plan": json_value(plan),
                "truth": json_value(truth), "pending_coefficients": json_value(_coefficients(pending)),
                "future": _future(bundle, current)}
    guide2 = _train(bundle, pending, source=1)
    model2 = bundle.guidance_backend._model_cache[guide2.snapshot.snapshot_sha256]
    expected["next_guide_weights"] = {key: value.tolist() for key, value in model2.state_dict().items()}
    (tmp_path / "expected.json").write_text(json.dumps(expected))
    shutil.rmtree(tmp_path / "guide-original")
    repo = Path(__file__).resolve().parents[1]
    environment = {**os.environ, "PYTHONPATH": f"{repo}:{repo / 'src'}:{repo / 'legacy/ns/src'}",
                   "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    code = "from tests.test_ns_checkpoint_store import _restore_complete_store; import sys; _restore_complete_store(sys.argv[1])"
    completed = subprocess.run([sys.executable, "-c", code, str(tmp_path)], env=environment,
                               capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_checkpoint_store_caches_do_not_retain_released_answers(tmp_path):
    query = AnswerQuery.at_points(fields=("u",), points=({"x": .2},))
    answer = AnswerView(query.kind, {"u": (1.,)}, points=query.points, shape=(1,))
    reference = weakref.ref(answer)
    store = CheckpointStore(tmp_path, identity={"fixture": "weak-caches"})
    store.save({"answer": answer, "same": answer})
    del answer
    gc.collect()
    assert reference() is None
    assert not store._written
    value = store.load()
    assert value["answer"] is value["same"]
    restored_reference = weakref.ref(value["answer"])
    del value
    gc.collect()
    assert restored_reference() is None
    assert not store._written
    assert not store._loaded
