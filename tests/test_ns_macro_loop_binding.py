from __future__ import annotations

import hashlib
import json
from math import pi
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pde_agent_verifier_solver.engine import (
    ControllerMacroEvaluator,
    PreparedMacroCohort,
)
import pde_agent_verifier_solver.plugins.ns.structured_guidance_backend as ns_guidance_module
from pde_agent_verifier_solver.engine.guidance_controller import (
    GuidanceRewardFeedback,
    GuidanceRewardRow,
)
from pde_agent_verifier_solver.engine.macro_query_plan import (
    ControllerMacroQueryPlan,
)
from pde_agent_verifier_solver.engine.reference_agent_runtime import (
    load_python_macro_candidate,
)
from pde_agent_verifier_solver.plugins.ns import (
    NS_DEVELOPMENT_SOLUTION_SESSION_METHODS,
    NS_PUBLIC_SESSION_METHODS,
    NS_VELOCITY_FIELDS,
    NS_VELOCITY_POINT_COUNTS,
    LegacyStructuredNSFlowSampler,
    LegacyStructuredNSSmoothPerturber,
    StructuredNSManufacturedCaseSource,
    StructuredNSGuidanceConfig,
    StructuredNSMacroGuidanceBackend,
    StructuredNSPerturbationResult,
    bind_ns_macro_loop,
    ns_velocity_point_query,
)
from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    AnswerView,
    Case,
    CaseSource,
    CaseVisibility,
    CohortKind,
    GenerationKind,
    GuidanceTrainingBudget,
    GuidanceTrainingRequest,
    GuidedSampleRequest,
    GuidedSampleSpec,
    GuidedSamplingPurpose,
    QueryKind,
    SampleRequest,
    sample_frozen_guidance,
    inspect_candidate_bundle,
)


class _TinyAnswer:
    """Deterministic smooth manufactured answer used only for CPU tests."""

    domain_bounds = (0.0, 1.0, 0.0, 1.0)
    time_interval = (0.0, 1.0)
    viscosity = 0.05
    public_geometry = {
        "schema": "tiny_unit_square_test_geometry_v1",
        "outer_boundary": "unit_square",
        "obstacles": [],
    }
    public_metadata = {
        "role": "synthetic_cpu_chain_test_only",
        "formal_training_distribution": False,
    }

    def __init__(self, amplitude: float) -> None:
        self.amplitude = float(amplitude)

    def evaluate_velocity(self, points):
        values = np.asarray(points, dtype=np.float64)
        t, x, y = values.T
        scale = self.amplitude * (1.0 + t)
        velocity_x = scale * pi * np.sin(pi * x) ** 2 * np.sin(2.0 * pi * y)
        velocity_y = -scale * pi * np.sin(2.0 * pi * x) * np.sin(pi * y) ** 2
        return np.column_stack((velocity_x, velocity_y))

    def evaluate_forcing(self, points):
        values = np.asarray(points, dtype=np.float64)
        t, x, y = values.T
        a = self.amplitude
        scale = a * (1.0 + t)
        sin_x = np.sin(pi * x)
        sin_y = np.sin(pi * y)
        sin_2x = np.sin(2.0 * pi * x)
        sin_2y = np.sin(2.0 * pi * y)
        cos_2x = np.cos(2.0 * pi * x)
        cos_2y = np.cos(2.0 * pi * y)
        u, v = self.evaluate_velocity(values).T
        d_t_u = a * pi * sin_x**2 * sin_2y
        d_t_v = -a * pi * sin_2x * sin_y**2
        d_x_u = scale * pi**2 * sin_2x * sin_2y
        d_y_u = 2.0 * scale * pi**2 * sin_x**2 * cos_2y
        d_x_v = -2.0 * scale * pi**2 * cos_2x * sin_y**2
        d_y_v = -scale * pi**2 * sin_2x * sin_2y
        laplacian_u = (
            scale * pi**3 * (2.0 * cos_2x - 4.0 * sin_x**2) * sin_2y
        )
        laplacian_v = (
            scale * pi**3 * sin_2x * (4.0 * sin_y**2 - 2.0 * cos_2y)
        )
        return np.column_stack(
            (
                d_t_u + u * d_x_u + v * d_y_u - self.viscosity * laplacian_u,
                d_t_v + u * d_x_v + v * d_y_v - self.viscosity * laplacian_v,
            )
        )


def _source() -> StructuredNSManufacturedCaseSource:
    return StructuredNSManufacturedCaseSource(
        generator_id="tiny-structured-ns-test-sampler",
        artifact_sha256=hashlib.sha256(b"tiny-ns-test").hexdigest(),
        clean_sampler=lambda seed: _TinyAnswer(
            0.4 + (int(seed) % 101) / 1000.0
        ),
        perturbation_sampler=lambda parent, seed: _TinyAnswer(
            parent.amplitude + (0.03 if int(seed) % 2 else -0.03)
        ),
        max_points_per_request=128,
    )


def _development_request(
    *,
    seed: int,
    count: int,
    kind: GenerationKind,
    anchors=(),
) -> SampleRequest:
    return SampleRequest(
        round_index=1,
        cohort=CohortKind.DEVELOPMENT,
        seed=seed,
        count=count,
        kind=kind,
        anchors=tuple(anchors),
    )


def test_guidance_feedback_resolution_accepts_the_canonical_official_cohort():
    source = _source()
    official = source.sample(
        SampleRequest(
            round_index=1,
            cohort=CohortKind.PROMOTION,
            seed=9,
            count=2,
            kind=GenerationKind.BASE,
        )
    )
    development = source.sample(
        _development_request(seed=10, count=1, kind=GenerationKind.BASE)
    )

    answers = source.resolve_guidance_feedback_answers(
        tuple(record.case.case_id for record in official)
    )

    assert len(answers) == 2
    with pytest.raises(ValueError, match="Official promotion cases"):
        source.resolve_guidance_feedback_answers(
            (development[0].case.case_id,)
        )
    source.release_cases(
        tuple(record.case.case_id for record in (*official, *development))
    )


def test_answer_first_clean_and_smooth_perturbed_sessions():
    source = _source()
    clean = source.sample(
        _development_request(seed=11, count=2, kind=GenerationKind.BASE)
    )
    perturbed = source.sample(
        _development_request(
            seed=12,
            count=2,
            kind=GenerationKind.PERTURBED,
            anchors=tuple(record.case for record in clean),
        )
    )

    assert all(record.case.source is CaseSource.GENERATED for record in clean)
    assert all(
        record.case.source is CaseSource.PERTURBED for record in perturbed
    )
    assert [record.case.parent_case_id for record in perturbed] == [
        record.case.case_id for record in clean
    ]
    assert all(
        record.case.case_id.startswith("ns-case-")
        and len(record.case.case_id) == len("ns-case-") + 64
        for record in (*clean, *perturbed)
    )
    replayed = _source().sample(
        _development_request(seed=11, count=2, kind=GenerationKind.BASE)
    )
    assert replayed[0].case.case_id != clean[0].case.case_id
    for record in perturbed:
        public_perturbation = record.case.public_data["perturbation"]
        assert "seed" not in public_perturbation
        assert "signed_calibrated_amplitude" not in public_perturbation
        assert public_perturbation["realization_seed_exposed"] is False
        assert public_perturbation["signed_amplitude_exposed"] is False
        audit = source.controller_perturbation_audit(record.case.case_id)
        assert isinstance(audit["seed"], int)
        assert f"{audit['seed']:016x}" not in record.case.case_id

    for record in (*clean, *perturbed):
        contract = record.case.public_data["answer_first"]
        assert contract["solution_chosen_before_question"] is True
        assert contract["forcing_rederived_after_perturbation"] is True
        assert record.case.public_data["operator"]["operator_id"].startswith(
            "incompressible_navier_stokes"
        )

    points = np.asarray(
        [[0.0, 0.2, 0.3], [0.7, 0.6, 0.8]], dtype=np.float64
    )
    public = source.public_session_factory(perturbed[0].case)
    assert source.public_session_methods == NS_PUBLIC_SESSION_METHODS
    assert (
        source.development_solution_session_methods
        == NS_DEVELOPMENT_SOLUTION_SESSION_METHODS
    )
    assert public.query_forcing(points).shape == (2, 2)
    assert public.query_initial_velocity(points[:, 1:]).shape == (2, 2)
    assert public.usage_snapshot["queried_points"] == 4
    public.close()
    with pytest.raises(RuntimeError, match="closed"):
        public.query_forcing(points)

    solution = source.development_solution_session_factory(perturbed[0].case)
    assert solution.query_completed_velocity(points).shape == (2, 2)
    solution.close()

    hidden = source.sample(
        SampleRequest(
            round_index=1,
            cohort=CohortKind.SELECTION,
            seed=13,
            count=1,
            kind=GenerationKind.BASE,
        )
    )[0]
    assert hidden.case.visibility is CaseVisibility.HIDDEN
    with pytest.raises(PermissionError, match="DEVELOPMENT-only"):
        source.development_solution_session_factory(hidden.case)


class _ExactPointHandle:
    def __init__(self, answer: _TinyAnswer) -> None:
        self.answer = answer

    def evaluate(self, query: AnswerQuery) -> AnswerView:
        values = np.asarray(
            [
                [float(point[name]) for name in ("t", "x", "y")]
                for point in query.points
            ]
        )
        velocity = self.answer.evaluate_velocity(values)
        return AnswerView(
            kind=QueryKind.POINTS,
            fields={
                "velocity_x": tuple(velocity[:, 0]),
                "velocity_y": tuple(velocity[:, 1]),
            },
            points=query.points,
            shape=(len(query.points),),
        )


class _ExactTwoPhaseCandidate:
    def __init__(self, answer: _TinyAnswer) -> None:
        self.answer = answer
        self.prepared_without_query = False

    def prepare_solution(self, case, solve_budget=None):
        assert "scoring_query" not in case.public_data
        self.prepared_without_query = True
        return _ExactPointHandle(self.answer)


def test_ns_default_points_and_shared_two_phase_controller_path():
    source = _source()
    plugin = bind_ns_macro_loop(case_source=source)
    record = source.sample(
        _development_request(seed=21, count=1, kind=GenerationKind.BASE)
    )[0]
    query = ControllerMacroQueryPlan(
        cohort_id="ns-common-selection",
        profile="selection",
        controller_seed=777,
    ).build(plugin.objective, record.case)
    assert query.kind is QueryKind.POINTS
    assert query.fields == NS_VELOCITY_FIELDS
    assert len(query.points) == NS_VELOCITY_POINT_COUNTS["selection"]
    assert all(set(point) == {"t", "x", "y"} for point in query.points)

    # This test-only candidate gets the same analytic object solely to verify
    # the shared evaluator and matcher end to end.
    answer = record.truth._answer
    cohort = PreparedMacroCohort.prepare(
        cohort_id="ns-common-selection",
        records=(record,),
        matcher=plugin.objective.matcher,
        query=query,
    )
    candidate = _ExactTwoPhaseCandidate(answer)
    evaluation = ControllerMacroEvaluator(cohort).evaluate(
        "ns-exact-test-candidate", candidate
    )
    assert evaluation.accepted
    assert evaluation.scores[0].primary_error == pytest.approx(0.0)
    assert evaluation.case_timings[0].formal_timing_eligible
    assert candidate.prepared_without_query is True


def test_public_problem_declares_agent_selected_oracles_without_guidance_claim():
    plugin = bind_ns_macro_loop(case_source=_source())
    manifest = plugin.public_problem_bundle.manifest()
    assert {
        method["name"] for method in manifest["oracle_methods"]
    } == {"query_forcing", "query_initial_velocity"}
    assert {
        method["name"]
        for method in manifest["development_solution_oracle_methods"]
    } == {"query_completed_velocity"}
    assert manifest["guidance_capability"]["supported"] is False
    assert all(
        method["coordinate_policy"] == "agent_selected"
        for method in (
            *manifest["oracle_methods"],
            *manifest["development_solution_oracle_methods"],
        )
    )


def test_legacy_perturber_is_random_smooth_and_does_no_online_audit():
    calls = []

    def make(parent, **kwargs):
        calls.append(("make", parent, kwargs))
        return {"direction": kwargs["seed"]}

    def materialize(parent, perturbation, *, amplitude):
        calls.append(("materialize", parent, perturbation, amplitude))
        return _TinyAnswer(parent.amplitude + amplitude)

    backend = LegacyStructuredNSSmoothPerturber(
        amplitude=0.02,
        family="spectral_complement",
        spectral_max_mode=24,
        make_perturbation=make,
        field_at_amplitude=materialize,
    )
    first = backend(_TinyAnswer(0.5), 100)
    second = backend(_TinyAnswer(0.5), 101)
    assert first.metadata["online_calibration_or_audit"] is False
    assert first.metadata["smooth_continuous_answer"] is True
    assert first.metadata["spectral_support_complement"] is True
    assert first.metadata["seed"] != second.metadata["seed"]
    assert calls[0][2]["spectral_max_mode"] == 24
    assert isinstance(first, StructuredNSPerturbationResult)
    assert first.metadata["amplitude_magnitude"] == pytest.approx(0.02)
    source = StructuredNSManufacturedCaseSource(
        generator_id="tiny-structured-ns-perturbation-audit-test",
        artifact_sha256=hashlib.sha256(b"tiny-ns-private-perturbation").hexdigest(),
        clean_sampler=lambda _seed: _TinyAnswer(0.5),
        perturbation_sampler=backend,
        max_points_per_request=128,
    )
    clean = source.sample(
        _development_request(seed=200, count=1, kind=GenerationKind.BASE)
    )[0]
    child = source.sample(
        _development_request(
            seed=201,
            count=1,
            kind=GenerationKind.PERTURBED,
            anchors=(clean.case,),
        )
    )[0]
    public = child.case.public_data["perturbation"]
    assert public["amplitude_magnitude"] == pytest.approx(0.02)
    assert "seed" not in public
    assert "signed_calibrated_amplitude" not in public
    private = source.controller_perturbation_audit(child.case.case_id)
    assert isinstance(private["seed"], int)
    assert abs(private["signed_calibrated_amplitude"]) == pytest.approx(0.02)


def test_query_blind_cold_start_bundle_runs_through_shared_evaluator():
    bundle_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "candidates"
        / "ns_hold_initial_bundle"
    )
    bundle = inspect_candidate_bundle(bundle_path)
    assert bundle.entrypoint == "submission.py"
    candidate = load_python_macro_candidate(bundle_path)

    source = _source()
    plugin = bind_ns_macro_loop(case_source=source)
    record = source.sample(
        _development_request(seed=31, count=1, kind=GenerationKind.BASE)
    )[0]
    query = ControllerMacroQueryPlan(
        cohort_id="ns-cold-start-chain",
        profile="development",
        controller_seed=991,
    ).build(plugin.objective, record.case)
    cohort = PreparedMacroCohort.prepare(
        cohort_id="ns-cold-start-chain",
        records=(record,),
        matcher=plugin.objective.matcher,
        query=query,
    )
    result = ControllerMacroEvaluator(
        cohort,
        public_session_factory=source.public_session_factory,
        public_session_methods=plugin.public_session_methods,
    ).evaluate("ns-hold-initial-cold-start", candidate)

    assert result.accepted
    assert result.case_timings[0].formal_timing_eligible
    assert result.scores[0].primary_error is not None
    assert np.isfinite(result.scores[0].primary_error)


def test_clean_source_uses_one_batch_call_and_guidance_manifest_is_truthful():
    class BatchOnlySampler:
        def __init__(self):
            self.calls = []

        def __call__(self, seed):
            raise AssertionError("per-case clean sampling must not run")

        def sample_many(self, seeds):
            self.calls.append(tuple(seeds))
            return tuple(
                _TinyAnswer(0.5 + index / 100.0)
                for index in range(len(seeds))
            )

    sampler = BatchOnlySampler()
    source = StructuredNSManufacturedCaseSource(
        generator_id="tiny-batched-flow-test",
        artifact_sha256=hashlib.sha256(b"tiny-batched-flow-test").hexdigest(),
        clean_sampler=sampler,
        perturbation_sampler=None,
    )
    rows = source.sample(
        _development_request(seed=41, count=4, kind=GenerationKind.BASE)
    )
    assert len(rows) == 4
    assert len(sampler.calls) == 1
    assert len(sampler.calls[0]) == 4
    assert bind_ns_macro_loop(
        case_source=source,
    ).public_problem_bundle.manifest()["guidance_capability"]["supported"] is False


@pytest.mark.parametrize("cuda_graph_sampling", [False, True])
def test_artifact_flow_sampler_batches_one_integration_call(tmp_path, cuda_graph_sampling):
    torch = pytest.importorskip("torch")

    class FakePCA:
        rank = 2
        coefficient_shape = (2, 1, 1, 1)
        source_signature = {
            "projection_config": {
                "time_order": 0,
                "spatial_mode_x": 0,
                "spatial_mode_y": 0,
            }
        }

        @classmethod
        def load(cls, path, *, map_location):
            return cls()

        def with_rank(self, rank):
            assert rank == self.rank
            return self

        def decode(self, scores):
            values = torch.zeros(
                (scores.shape[0], *self.coefficient_shape),
                device=scores.device,
            )
            values[:, 0, 0, 0, 0] = scores[:, 0]
            values[:, 1, 0, 0, 0] = scores[:, 1]
            return values

    class FakeProjection:
        time_coefficient_count = 1
        spatial_mode_x = 0
        spatial_mode_y = 0

        @classmethod
        def from_dict(cls, payload):
            assert payload["time_order"] == 0
            return cls()

        def validate(self):
            return None

    class FakeFlowConfig:
        feature_dim = 2

        @classmethod
        def from_dict(cls, payload):
            assert payload["feature_dim"] == 2
            return cls()

    class FakeFlow:
        calls = 0

        def load_state_dict(self, state):
            assert state == {}

        def to(self, device):
            return self

        def eval(self):
            return self

        def sample(self, count, **kwargs):
            type(self).calls += 1
            assert kwargs["initial_values"].shape == (count, 2)
            return kwargs["initial_values"]

    class FakeFlowType:
        @staticmethod
        def build(config):
            return FakeFlow()

    class FakeGeometry:
        domain_bounds = (0.0, 1.0, 0.0, 1.0)

        @classmethod
        def from_dict(cls, payload):
            assert payload["domain_bounds"] == [0.0, 1.0, 0.0, 1.0]
            return cls()

        def validate(self):
            return None

    class FakeCoefficients:
        def __init__(self, *, psi, pressure):
            self.psi = psi
            self.pressure = pressure

    class FakeField:
        def __init__(self, coefficients, **kwargs):
            self.coefficients = coefficients
            self.kwargs = kwargs

    pca_path = tmp_path / "formal-selected-pca.pt"
    pca_path.write_bytes(b"synthetic-test-pca-only")
    pca_sha = hashlib.sha256(pca_path.read_bytes()).hexdigest()
    flow_path = tmp_path / "formal-selected-flow.pt"
    torch.save(
        {
            "schema": "fake-structured-flow",
            "pca_sha256": pca_sha,
            "pca_rank": 2,
            "model_config": {"feature_dim": 2},
            "ema_state_dict": {},
            "model_state_dict": {},
        },
        flow_path,
    )
    symbols = SimpleNamespace(
        torch=torch,
        flow_schema="fake-structured-flow",
        pca_type=FakePCA,
        flow_config_type=FakeFlowConfig,
        flow_type=FakeFlowType,
        projection_config_type=FakeProjection,
        field_type=FakeField,
        case_coefficients_type=FakeCoefficients,
        continuous_mms_abi="synthetic-test-abi",
        geometry_spec_type=FakeGeometry,
        build_default_geometry=lambda: None,
        geometry_from_contract=lambda geometry: FakeGeometry(),
        coefficient_tensor_from_field=lambda field: None,
    )
    sampler = LegacyStructuredNSFlowSampler(
        flow_checkpoint=flow_path,
        pca_checkpoint=pca_path,
        time_interval=(0.0, 2.0),
        viscosity=0.01,
        device="cpu",
        integration_steps=3,
        cuda_graph_sampling=cuda_graph_sampling,
        geometry_spec={"domain_bounds": [0.0, 1.0, 0.0, 1.0]},
        symbols=symbols,
    )
    sampled = sampler.sample_many((7, 8, 9, 10))
    assert len(sampled) == 4
    assert FakeFlow.calls == 1
    assert sampler.artifact_metadata["cuda_graph_sampling"] is cuda_graph_sampling
    repeated = sampler.sample_many((7, 8, 9, 10))
    for before, after in zip(sampled, repeated):
        torch.testing.assert_close(before.coefficients.psi, after.coefficients.psi, rtol=0, atol=0)
        torch.testing.assert_close(before.coefficients.pressure, after.coefficients.pressure, rtol=0, atol=0)
    assert FakeFlow.calls == 2
    assert all(field.kwargs["t1"] == 2.0 for field in sampled)


def test_velocity_scoring_points_reject_public_obstacle_interior():
    source = _source()
    original = source.sample(
        _development_request(seed=51, count=1, kind=GenerationKind.BASE)
    )[0].case
    obstacle = [
        [0.25, 0.25],
        [0.75, 0.25],
        [0.75, 0.75],
        [0.25, 0.75],
    ]
    public_data = dict(original.public_data)
    domain = dict(public_data["domain"])
    domain["geometry"] = {
        "abi_version": "synthetic_obstacle_test_v1",
        "domain_bounds": [0.0, 1.0, 0.0, 1.0],
        "obstacle_polygons": [obstacle],
    }
    public_data["domain"] = domain
    case = Case(
        case_id="synthetic-obstacle-scoring-test",
        source=original.source,
        visibility=original.visibility,
        public_data=public_data,
        round_index=original.round_index,
        tags=original.tags,
    )
    first = ns_velocity_point_query(case, "terminal", controller_seed=1001)
    second = ns_velocity_point_query(case, "terminal", controller_seed=1001)
    assert first == second
    assert all(
        not (0.25 <= point["x"] <= 0.75 and 0.25 <= point["y"] <= 0.75)
        for point in first.points
    )

    blocked_domain = dict(domain)
    blocked_domain["geometry"] = {
        "abi_version": "synthetic_blocked_test_v1",
        "domain_bounds": [0.0, 1.0, 0.0, 1.0],
        "obstacle_polygons": [
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        ],
    }
    blocked_public = dict(public_data)
    blocked_public["domain"] = blocked_domain
    blocked = Case(
        case_id="synthetic-fully-blocked-scoring-test",
        source=original.source,
        visibility=original.visibility,
        public_data=blocked_public,
        round_index=original.round_index,
        tags=original.tags,
    )
    with pytest.raises(ValueError, match="rejection sampler exhausted"):
        ns_velocity_point_query(blocked, "development", controller_seed=1001)


def test_real_guidance_lifecycle_is_sanitized_frozen_and_releases_residents(
    tmp_path, monkeypatch
):
    torch = pytest.importorskip("torch")

    class TinyGuidanceFlow:
        def __init__(self):
            self._device = torch.device("cpu")
            self.pca = SimpleNamespace(rank=2)
            self.artifact_metadata = {
                "pca_sha256": "a" * 64,
                "flow_state": "raw",
                "integration_steps": 8,
            }

        def encode_fields(self, fields):
            return torch.as_tensor(
                [[field.amplitude, field.amplitude**2] for field in fields],
                dtype=torch.float32,
            )

        def sample_many_with_guidance(self, seeds, *, guidance, guidance_scale=1.0):
            assert guidance is not None
            assert guidance_scale == 1.0
            return tuple(
                _TinyAnswer(0.45 + (int(seed) % 17) / 100.0)
                for seed in seeds
            )

    class TinyRewardConfig:
        def __init__(self, *, feature_dim, width, depth, time_width):
            self.feature_dim = feature_dim
            self.width = width
            self.depth = depth
            self.time_width = time_width

        def to_dict(self):
            return {
                "feature_dim": self.feature_dim,
                "width": self.width,
                "depth": self.depth,
                "time_width": self.time_width,
            }

        @classmethod
        def from_dict(cls, payload):
            return cls(**payload)

    class TinyRewardModel(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.linear = torch.nn.Linear(config.feature_dim, 1)

        def forward(self, values, time):
            return self.linear(values)[:, 0] + 0.0 * time

    class TinyRewardModelType:
        @staticmethod
        def build(config):
            return TinyRewardModel(config)

    class TinyGradient:
        def __init__(self, model, *, strength, max_flow_ratio):
            self.model = model
            self.strength = strength
            self.max_flow_ratio = max_flow_ratio

    symbols = SimpleNamespace(
        torch=torch,
        guidance_schema="synthetic-guidance-checkpoint-test-v1",
        model_config_type=TinyRewardConfig,
        model_type=TinyRewardModelType,
        gradient_type=TinyGradient,
    )
    flow = TinyGuidanceFlow()
    source = _source()
    config = StructuredNSGuidanceConfig(
        training_steps=2,
        batch_size=2,
        width=4,
        depth=1,
        time_width=2,
        minimum_cases=2,
        guidance_strength=0.1,
        guidance_max_flow_ratio=0.25,
    )
    backend = StructuredNSMacroGuidanceBackend(
        flow_sampler=flow,
        case_source=source,
        artifact_root=tmp_path / "controller-private-guidance",
        config=config,
        symbols=symbols,
    )
    assert dict(backend.official_sample_settings) == {
        "sampling_policy": "solver_reward_guided_rectified_flow",
        "flow_state": "raw",
        "integration_steps": 8,
        "guidance_strength": 0.1,
        "guidance_max_flow_ratio": 0.25,
    }
    plugin = bind_ns_macro_loop(case_source=source, guidance_backend=backend)
    manifest = plugin.public_problem_bundle.manifest()
    assert manifest["guidance_capability"]["supported"] is True
    assert plugin.public_result_kind == "pde_solver"
    assert plugin.public_result_task == "pde_solution"

    failed_records = source.sample(
        SampleRequest(
            round_index=0,
            cohort=CohortKind.GUIDANCE_FEEDBACK,
            seed=60,
            count=1,
            kind=GenerationKind.BASE,
        )
    )
    failed_feedback = GuidanceRewardFeedback(
        source_round_index=0,
        target_round_index=1,
        target_solver_sha256=hashlib.sha256(b"failed-target").hexdigest(),
        base_generator_sha256=source.artifact_sha256,
        reward_matcher_definition_sha256=hashlib.sha256(b"failed-match").hexdigest(),
        rows=(GuidanceRewardRow(failed_records[0].case, 0.4, "ok"),),
    )
    try:
        with pytest.raises(ValueError, match="too few cases"):
            backend.seal_reward_feedback(failed_feedback)
    finally:
        source.release_cases((failed_records[0].case.case_id,))
    with pytest.raises(KeyError, match="no longer active"):
        source.resolve_active_records((failed_records[0].case.case_id,))

    records = source.sample(
        SampleRequest(
            round_index=0,
            cohort=CohortKind.GUIDANCE_FEEDBACK,
            seed=61,
            count=2,
            kind=GenerationKind.BASE,
        )
    )
    generator_sha = source.artifact_sha256
    solver_sha = hashlib.sha256(b"tiny-guidance-target-solver").hexdigest()
    matcher_sha = hashlib.sha256(b"tiny-guidance-matcher").hexdigest()
    feedback = GuidanceRewardFeedback(
        source_round_index=0,
        target_round_index=1,
        target_solver_sha256=solver_sha,
        base_generator_sha256=generator_sha,
        reward_matcher_definition_sha256=matcher_sha,
        rows=tuple(
            GuidanceRewardRow(
                record.case,
                0.2 + 0.1 * index,
                "ok",
                {"relative_l2": 0.2 + 0.1 * index},
            )
            for index, record in enumerate(records)
        ),
    )

    # The shared controller seals while feedback records are still active,
    # then its finally block discards them.
    artifact = backend.seal_reward_feedback(feedback)
    sanitized = json.loads(Path(artifact.uri).read_text())
    serialized = json.dumps(sanitized, sort_keys=True)
    assert sanitized["schema"] == "ns_unified_public_reward_feedback_v1"
    assert '"scores"' not in serialized
    assert "coefficients" not in serialized
    assert "answer_access_locator" not in serialized
    source.release_cases(tuple(record.case.case_id for record in records))
    with pytest.raises(KeyError, match="no longer active"):
        source.resolve_active_records(tuple(record.case.case_id for record in records))

    request = GuidanceTrainingRequest(
        source_round_index=0,
        target_round_index=1,
        base_generator_sha256=generator_sha,
        target_solver_sha256=solver_sha,
        reward_matcher_definition_sha256=matcher_sha,
        reward_feedback_uri=artifact.uri,
        reward_feedback_sha256=artifact.sha256,
        guidance_config_sha256=config.sha256,
        controller_training_seed=71,
        training_budget=GuidanceTrainingBudget(10.0, "synthetic_cpu_test"),
    )
    snapshot = backend.train_guidance(request)
    assert snapshot.training_request == request

    original_hash = ns_guidance_module._file_sha256
    snapshot_hash_calls = 0

    def counted_hash(path):
        nonlocal snapshot_hash_calls
        if Path(path).resolve() == Path(snapshot.guided_artifact_uri).resolve():
            snapshot_hash_calls += 1
        return original_hash(path)

    monkeypatch.setattr(ns_guidance_module, "_file_sha256", counted_hash)
    # A freshly constructed backend cold-verifies once; subsequent opens use
    # its frozen model cache and perform no online artifact audit.
    reopened = StructuredNSMacroGuidanceBackend(
        flow_sampler=flow,
        case_source=source,
        artifact_root=tmp_path / "controller-private-guidance",
        config=config,
        symbols=symbols,
    )
    oracle = reopened.open_guidance(snapshot)
    reopened.open_guidance(snapshot)
    assert snapshot_hash_calls == 1

    guided_request = GuidedSampleRequest(
        spec=GuidedSampleSpec(2),
        request_id="synthetic-guidance-development-request",
        purpose=GuidedSamplingPurpose.DEVELOPMENT_ONLINE,
        snapshot_sha256=snapshot.snapshot_sha256,
        controller_seed=81,
        controller_nonce="synthetic-guidance-nonce",
        inference_wall_budget_seconds=10.0,
        compute_capacity="synthetic_cpu_test",
    )
    batch = sample_frozen_guidance(oracle, guided_request)
    assert len(batch.samples) == 2
    session = oracle.open_development_solution_session(batch.samples[0].sample_id)
    velocity = session.query_completed_velocity([[0.2, 0.3, 0.4]])
    assert velocity.shape == (1, 2)
    session.close()
    issued_ids = tuple(sample.case.case_id for sample in batch.samples)
    assert len(source.resolve_active_records(issued_ids)) == 2
    oracle.close()
    with pytest.raises(KeyError, match="no longer active"):
        source.resolve_active_records(issued_ids)
