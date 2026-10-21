"""Behavior checks for the optional request-local NS CUDA graph sampler."""
from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from pde_agent_verifier_solver.plugins.ns.flow_cuda_graph import (
    _FlowStageGraph,
    sample_flow_cuda_graph,
)


torch = pytest.importorskip("torch")


@pytest.fixture
def symbols(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / "legacy/ns/src")
    )
    return (
        importlib.import_module("agent_verifier.neural_generator.structured_ns_distribution"),
        importlib.import_module("agent_verifier.neural_generator.structured_ns_guidance"),
    )


def _models(symbols, device="cpu", diagonal=False):
    flow, reward = symbols
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        model = flow.StructuredCoefficientRectifiedFlow.build(
            flow.StructuredCoefficientFlowConfig(
                feature_dim=7, width=24, depth=2, time_width=8,
                full_rank_diagonal_velocity=diagonal,
            )
        )
        reward_model = reward.StructuredRewardModel.build(
            reward.StructuredRewardModelConfig(feature_dim=7, width=24, depth=2, time_width=8)
        )
        with torch.no_grad():
            for current in (model, reward_model):
                for parameter in current.parameters():
                    parameter.uniform_(-0.15, 0.15)
    return model.to(device).eval(), reward_model.to(device).eval()


def _gradient(symbols, reward_model):
    return symbols[1].StructuredRewardGradient(
        reward_model, strength=0.5, max_flow_ratio=0.25,
    )


def test_cpu_fallback_preserves_input_rng_and_custom_model():
    class NeverCalled:
        def __call__(self, *_):
            raise AssertionError("unsupported graph path must not execute the model")

    values = torch.ones((3, 7))
    rng = torch.get_rng_state().clone()
    with torch.no_grad():
        assert sample_flow_cuda_graph(
            NeverCalled(), values, steps=5, method="heun",
            guidance=SimpleNamespace(), guidance_scale=1.0,
        ) is None
    assert torch.equal(values, torch.ones((3, 7)))
    assert torch.equal(rng, torch.get_rng_state())


def test_guidance_tensor_helper_does_not_commit_warmup_diagnostics(symbols):
    _, model = _models(symbols)
    reference = _gradient(symbols, model)
    split = _gradient(symbols, model)
    values = torch.arange(14, dtype=torch.float32).reshape(2, 7) / 7
    time = torch.tensor([0.25, 0.5])
    base = torch.ones_like(values)
    before = split.diagnostics()
    with torch.no_grad():
        expected = reference(values, time, base)
        for _ in range(3):
            tensors = split._tensor_update(values, time, base)
        assert split.diagnostics() == before
        split._commit_diagnostics(2, tensors)
    assert torch.equal(tensors[0], expected)
    assert split.diagnostics() == reference.diagnostics()


@pytest.fixture
def cuda_device():
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    return "cuda"


@pytest.mark.parametrize("method", ["euler", "heun"])
@pytest.mark.parametrize("count", [1, 3])
@pytest.mark.parametrize("diagonal", [False, True])
@pytest.mark.parametrize("guided", [False, True])
def test_cuda_graph_preserves_sample_diagnostics_and_rng(
    symbols, cuda_device, method, count, diagonal, guided,
):
    model, reward_model = _models(symbols, cuda_device, diagonal)
    reference_guidance = _gradient(symbols, reward_model) if guided else None
    graph_guidance = _gradient(symbols, reward_model) if guided else None
    initial = torch.arange(count * 7, device=cuda_device, dtype=torch.float32).reshape(count, 7) / 13
    initial_copy = initial.clone()
    rng = torch.cuda.get_rng_state().clone()
    cpu_rng = torch.get_rng_state().clone()
    options = dict(steps=9, method=method, guidance_scale=0.7 if guided else 0.0)
    with torch.no_grad():
        expected = model.sample(
            count, device=cuda_device, initial_values=initial,
            guidance=reference_guidance, **options,
        )
        actual = sample_flow_cuda_graph(model, initial, guidance=graph_guidance, **options)
    assert actual is not None
    assert torch.equal(actual, expected)
    assert torch.equal(initial, initial_copy)
    assert torch.equal(rng, torch.cuda.get_rng_state())
    assert torch.equal(cpu_rng, torch.get_rng_state())
    if guided:
        assert graph_guidance.diagnostics() == reference_guidance.diagnostics()
        assert graph_guidance.diagnostics()["calls"] == (9 if method == "euler" else 17)


def test_cuda_graph_stage_outputs_and_diagnostics_survive_next_replay(symbols, cuda_device):
    model, reward_model = _models(symbols, cuda_device)
    reference = _gradient(symbols, reward_model)
    guided = _gradient(symbols, reward_model)
    values = torch.ones((2, 7), device=cuda_device)
    first_time = torch.full((2,), 0.2, device=cuda_device)
    second_time = torch.full((2,), 0.7, device=cuda_device)
    with torch.no_grad():
        stage = _FlowStageGraph(torch, model, values, guided, 0.7)
        output = stage(values, first_time)
        snapshot = output.clone()
        stage(values * 3, second_time)
        for state, time in ((values, first_time), (values * 3, second_time)):
            velocity = model(state, time)
            reference(state, time, velocity)
        torch.cuda.synchronize()
    assert torch.equal(output, snapshot)
    assert guided.diagnostics() == reference.diagnostics()


def test_cuda_graph_preserves_immediate_nonfinite_failure(symbols, cuda_device):
    model, reward_model = _models(symbols, cuda_device)
    guided = _gradient(symbols, reward_model)
    values = torch.ones((1, 7), device=cuda_device)
    time = torch.zeros(1, device=cuda_device)
    with torch.no_grad():
        stage = _FlowStageGraph(torch, model, values, guided, 1.0)
        before = guided.diagnostics()
        with pytest.raises(FloatingPointError, match="became non-finite"):
            stage(torch.full_like(values, float("inf")), time)
        torch.cuda.synchronize()
    assert guided.diagnostics() == before


@pytest.mark.parametrize("restriction", ["training", "hook", "grad", "autocast", "inference", "dtype", "layout"])
def test_cuda_graph_unsupported_execution_keeps_eager_path(symbols, cuda_device, restriction):
    model, _ = _models(symbols, cuda_device)
    values = torch.ones((2, 7), device=cuda_device)
    hook = None
    if restriction == "training":
        model.train()
    elif restriction == "hook":
        hook = model.register_forward_hook(lambda *_: None)
    elif restriction == "dtype":
        values = values.double()
    elif restriction == "layout":
        values = torch.ones((7, 2), device=cuda_device).t()
    try:
        with torch.set_grad_enabled(restriction == "grad"):
            with torch.autocast("cuda", enabled=restriction == "autocast"):
                with torch.inference_mode(restriction == "inference"):
                    assert sample_flow_cuda_graph(model, values, steps=5, method="heun") is None
    finally:
        if hook is not None:
            hook.remove()


def test_cuda_graph_new_call_uses_reloaded_weights_and_new_batch(symbols, cuda_device):
    model, reward_model = _models(symbols, cuda_device, diagonal=True)
    with torch.no_grad():
        for count, reload_weights in ((2, False), (2, True), (3, False)):
            if reload_weights:
                for current in (model, reward_model):
                    checkpoint = {name: value.clone() for name, value in current.state_dict().items()}
                    checkpoint["time_mlp.0.weight"] += 0.07
                    current.load_state_dict(checkpoint)
            initial = torch.arange(count * 7, device=cuda_device, dtype=torch.float32).reshape(count, 7) / 11
            reference = _gradient(symbols, reward_model)
            guided = _gradient(symbols, reward_model)
            expected = model.sample(
                count, steps=5, device=cuda_device, initial_values=initial,
                method="heun", guidance=reference, guidance_scale=0.7,
            )
            actual = sample_flow_cuda_graph(
                model, initial, steps=5, method="heun", guidance=guided, guidance_scale=0.7,
            )
            assert actual is not None and torch.equal(actual, expected)
            assert guided.diagnostics() == reference.diagnostics()


# Truth materialization bypasses the Controller's Generator session lock. These
# tests exercise real CaseRecord truth adapters, including the Perturbed path.
_TEST_GEOMETRY = {
    "schema": "tiny_unit_square_test_geometry_v1",
    "outer_boundary": "unit_square",
    "obstacles": [],
}
_TEST_POINTS = ((0.2, 0.3, 0.4), (0.7, 0.6, 0.5))


def _truth_cases(field_factory):
    from pde_agent_verifier_solver.plugins.ns.continuous_backend import (
        StructuredNSManufacturedCaseSource, StructuredNSMMSAnswer,
    )
    from pde_agent_verifier_solver.sdk import CaseVisibility, GenerationKind
    from pde_agent_verifier_solver.sdk.macro_loop import CohortKind, SampleRequest

    adapters = []

    def adapt(raw):
        answer = StructuredNSMMSAnswer(raw, public_geometry=_TEST_GEOMETRY)
        adapters.append(answer)
        return answer

    source = StructuredNSManufacturedCaseSource(
        generator_id="capture-concurrency-test", artifact_sha256="a" * 64,
        clean_sampler=lambda seed: field_factory(1.0),
        perturbation_sampler=lambda parent, seed: field_factory(parent.amplitude + 0.2),
        answer_adapter=adapt,
    )
    base = source.sample(SampleRequest(
        round_index=1, cohort=CohortKind.DEVELOPMENT, seed=11, count=1,
    ))[0]
    perturbed = source.sample(SampleRequest(
        round_index=1, cohort=CohortKind.DEVELOPMENT, seed=12, count=1,
        kind=GenerationKind.PERTURBED, anchors=(base.case,),
    ))[0]
    guided = source.issue_answer_first_records(
        (field_factory(1.4),), round_index=1,
        visibility=CaseVisibility.DEVELOPMENT, case_id_prefix="guided",
        controller_seed=13, distribution_lineage={"kind": "guidance-test"},
    )[0]
    return (base, perturbed, guided), adapters


def _truth_query():
    from pde_agent_verifier_solver.sdk import AnswerQuery

    return AnswerQuery.at_points(
        fields=("velocity_x", "velocity_y"),
        points=tuple(dict(zip(("t", "x", "y"), point)) for point in _TEST_POINTS),
    )


class _ProbeField:
    domain_bounds = (0.0, 1.0, 0.0, 1.0)
    t0 = 0.0
    t1 = 1.0
    viscosity = 0.01

    def __init__(self, amplitude, evaluate):
        self.amplitude = amplitude
        self.evaluate = evaluate

    def velocity_pressure(self, coordinates):
        value = self.evaluate(coordinates)
        return value, value, value

    def quantities(self, coordinates):
        value = self.evaluate(coordinates)
        return {"forcing_u": value, "forcing_v": value}


def _cpu_graph_torch(capture_context):
    from contextlib import nullcontext

    stream = SimpleNamespace(wait_stream=lambda _: None)
    cuda = SimpleNamespace(
        CUDAGraph=object,
        Stream=lambda **_: stream,
        current_stream=lambda _: stream,
        stream=lambda _: nullcontext(),
        graph=lambda *_, **__: capture_context(),
    )
    return SimpleNamespace(zeros=torch.zeros, cuda=cuda)


@pytest.mark.parametrize("endpoint", ["truth", "forcing"])
def test_capture_excludes_direct_truth_and_forcing(endpoint):
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import contextmanager
    from threading import Event

    capture_entered, release_capture = Event(), Event()
    query_started, field_entered = Event(), Event()

    @contextmanager
    def capture():
        capture_entered.set()
        assert release_capture.wait(5), "capture test release timed out"
        yield

    def evaluate(coordinates):
        field_entered.set()
        return torch.ones(len(coordinates))

    records, answers = _truth_cases(lambda amplitude: _ProbeField(amplitude, evaluate))
    query = _truth_query()

    def ask():
        query_started.set()
        return records[0].truth.answer(query) if endpoint == "truth" else answers[0].evaluate_forcing(_TEST_POINTS)

    with ThreadPoolExecutor(max_workers=2) as pool:
        graph = pool.submit(
            _FlowStageGraph, _cpu_graph_torch(capture), lambda values, time: values * 2,
            torch.ones((2, 7)), None, 0.0,
        )
        try:
            assert capture_entered.wait(5)
            answer = pool.submit(ask)
            assert query_started.wait(5)
            assert not field_entered.wait(0.1), "truth kernels entered while capture held the NS lock"
        finally:
            release_capture.set()
        graph.result(timeout=5)
        answer.result(timeout=5)
    assert field_entered.is_set()


@pytest.mark.parametrize("endpoint", ["truth", "forcing"])
def test_capture_waits_for_truth_host_copy_completion(endpoint):
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import nullcontext
    from threading import Event

    copy_started, release_copy = Event(), Event()
    graph_started, warmup_started = Event(), Event()

    class BlockingHostCopy:
        def __init__(self, count):
            self.count = count

        def detach(self):
            return self

        def cpu(self):
            copy_started.set()
            assert release_copy.wait(5), "host copy test release timed out"
            return self

        def numpy(self):
            return torch.ones(self.count).numpy()

    records, answers = _truth_cases(lambda amplitude: _ProbeField(
        amplitude, lambda coordinates: BlockingHostCopy(len(coordinates)),
    ))
    query = _truth_query()

    def compute(values, time):
        warmup_started.set()
        return values * 2

    def build_graph():
        graph_started.set()
        return _FlowStageGraph(
            _cpu_graph_torch(nullcontext), compute, torch.ones((2, 7)), None, 0.0,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        answer = pool.submit(
            records[0].truth.answer, query,
        ) if endpoint == "truth" else pool.submit(answers[0].evaluate_forcing, _TEST_POINTS)
        try:
            assert copy_started.wait(5)
            graph = pool.submit(build_graph)
            assert graph_started.wait(5)
            assert not warmup_started.wait(0.1), "capture started before truth host copy completed"
        finally:
            release_copy.set()
        answer.result(timeout=5)
        graph.result(timeout=5)
    assert warmup_started.is_set()


def test_cuda_graph_concurrent_base_guided_perturbed_truth_matches_serial(
    symbols, cuda_device, monkeypatch,
):
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import contextmanager
    from threading import Event

    capture_active = Event()

    def make_field(amplitude):
        def evaluate(coordinates):
            assert not capture_active.is_set(), "truth CUDA work overlapped graph capture"
            points = torch.as_tensor(coordinates, device=cuda_device, dtype=torch.float32)
            return amplitude * (points[:, 0] + points[:, 1] * points[:, 2])
        return _ProbeField(amplitude, evaluate)

    records, answers = _truth_cases(make_field)
    query = _truth_query()
    expected_truths = tuple(record.truth.answer(query) for record in records)
    expected_forcing = tuple(answer.evaluate_forcing(_TEST_POINTS) for answer in answers)
    model, reward_model = _models(symbols, cuda_device, diagonal=True)
    initial = torch.arange(14, device=cuda_device, dtype=torch.float32).reshape(2, 7) / 11
    reference_guidance = _gradient(symbols, reward_model)
    with torch.no_grad():
        expected = model.sample(
            2, steps=5, device=cuda_device, initial_values=initial,
            method="heun", guidance=reference_guidance, guidance_scale=0.7,
        )
    original_graph_context = torch.cuda.graph
    for _ in range(3):
        attempted = Event()

        @contextmanager
        def capture(*args, **kwargs):
            try:
                with original_graph_context(*args, **kwargs):
                    capture_active.set()
                    assert attempted.wait(10), "truth worker did not attempt concurrent query"
                    yield
            finally:
                capture_active.clear()

        monkeypatch.setattr(torch.cuda, "graph", capture)

        def concurrent_truth():
            assert capture_active.wait(10), "graph capture did not start"
            attempted.set()
            actual_truths = tuple(record.truth.answer(query) for record in records)
            actual_forcing = tuple(answer.evaluate_forcing(_TEST_POINTS) for answer in answers)
            return actual_truths, actual_forcing

        guided = _gradient(symbols, reward_model)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(concurrent_truth)
            with torch.no_grad():
                actual = sample_flow_cuda_graph(
                    model, initial, steps=5, method="heun", guidance=guided, guidance_scale=0.7,
                )
            actual_truths, actual_forcing = future.result(timeout=10)
        assert actual is not None and torch.equal(actual, expected)
        assert guided.diagnostics() == reference_guidance.diagnostics()
        assert actual_truths == expected_truths
        for concurrent, serial in zip(actual_forcing, expected_forcing, strict=True):
            assert (concurrent == serial).all()
