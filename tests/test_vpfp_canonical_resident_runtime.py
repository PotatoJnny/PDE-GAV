from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from threading import RLock
from types import SimpleNamespace

import numpy as np
import pytest

import pde_agent_verifier_solver.plugins.vpfp.perturbation_backend as backend_module
from pde_agent_verifier_solver.engine.campaign import EvaluatorPolicy
from pde_agent_verifier_solver.engine.candidate_case import candidate_case_id
from pde_agent_verifier_solver.engine.evaluator import (
    AgentEvaluator,
    EvaluationRequest,
    HiddenVerifier,
)
from pde_agent_verifier_solver.plugins.vpfp import (
    VPFPAnchoredPerturbationBackend,
    VPFPContinuousGenerator,
    VPFPLegacyStructuredRuntime,
    VPFPObservableTask,
    VPFPRuntimeCase,
    VPFP_DEVELOPMENT_SOLUTION_SESSION_METHODS,
    VPFP_PUBLIC_SESSION_METHODS,
    vpfp_public_problem_bundle,
)
from pde_agent_verifier_solver.plugins.vpfp.macro_matching import (
    VPFP_MACRO_FIELDS,
)
from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    AnswerView,
    Artifact,
    CaseAccessKind,
    CaseVisibility,
    DevelopmentCoordinateMode,
    EvaluationPhase,
    GenerationKind,
    GenerationPurpose,
    GenerationRequest,
    GeneratorCheckpoint,
    GeneratorMode,
    GeneratorState,
    TaskKind,
    open_development_solution_session,
    open_public_case_session,
)


class _TorchShim:
    @staticmethod
    def as_tensor(value, *, dtype, device):
        del device
        return np.asarray(value, dtype=dtype)

    @staticmethod
    def no_grad():
        return nullcontext()


class _ExpandedModel:
    torch = _TorchShim()

    def __init__(self) -> None:
        self.config = SimpleNamespace(time_domain=(0.0, 1.0))

    def parameters(self):
        yield SimpleNamespace(dtype=np.float64, device="cpu")

    def fields(self, points, latent, *, equation_condition):
        del latent, equation_condition
        points = np.asarray(points)
        return {
            "rho": 1.0 + points[:, 1],
            "g": 2.0 + points[:, 3],
        }

    def public_equation_metadata(self):
        return {
            "equation_version": "vpfp-test-v1",
            "x_domain": [1.0, 1.0],
            "domain": {
                "space": "two_dimensional_torus",
                "velocity": "R^2",
                "time": [0.0, 1.0],
            },
        }


class _API:
    torch = _TorchShim()
    MACRO_OUTPUT_CONTRACT = "vpfp_continuous_macro_output_v1"

    def __init__(self) -> None:
        self.source_chunk_sizes: list[int] = []
        self.macro_calls: list[np.ndarray] = []

    @staticmethod
    def ieee_float32_coordinate_matmul(model):
        del model
        return nullcontext()

    def manufacture_standard_vpfp_source_analytic(
        self, model, points, latent, *, equation_condition
    ):
        del model, latent, equation_condition
        points = np.asarray(points)
        self.source_chunk_sizes.append(len(points))
        return {"source": np.sum(points, axis=1)}

    def evaluate_continuous_hermite_macro_points(
        self,
        model,
        latent,
        points,
        *,
        equation_condition,
        chunk_size,
    ):
        del model, latent, equation_condition, chunk_size
        points = np.asarray(points)
        self.macro_calls.append(points.copy())
        return _macro_values(points)

    @staticmethod
    def make_axes(grid, time):
        del grid, time
        return {
            "t": np.asarray((0.0, 0.5, 1.0)),
            "x": np.asarray((0.0, 0.5)),
            "y": np.asarray((0.0, 0.5)),
            "vx": np.asarray((-1.0, 1.0)),
            "vy": np.asarray((-1.0, 1.0)),
        }

    @staticmethod
    def trapezoidal_velocity_weights(grid):
        del grid
        return np.ones((2, 2), dtype=np.float64)


@pytest.fixture
def resident_runtime(tmp_path, monkeypatch):
    runtime = object.__new__(VPFPLegacyStructuredRuntime)
    runtime.supports_perturbations = True
    runtime.materialize_cases = False
    runtime.public_session_methods = VPFP_PUBLIC_SESSION_METHODS
    runtime.development_solution_session_methods = (
        VPFP_DEVELOPMENT_SOLUTION_SESSION_METHODS
    )
    runtime.output_root = (tmp_path / "resident-runtime").resolve()
    runtime.output_root.mkdir()
    runtime._resident_cases = {}
    runtime._runtime_lock = RLock()
    runtime.chunk_size = 32
    runtime.expanded_model = _ExpandedModel()
    runtime.api = _API()
    runtime.api.requested_source_sizes = []

    def requested_source(model, points, latent, *, equation_condition=None,
                         chunk_size=8192, return_diagnostics=False):
        del model, latent, equation_condition, chunk_size
        assert not return_diagnostics
        points = np.asarray(points)
        runtime.api.requested_source_sizes.append(len(points))
        return np.sum(points, axis=1)

    # This fixture checks session/engine wiring only. The real analytic kernel
    # and its unique-coordinate counts are tested in test_vpfp_batched_source.
    monkeypatch.setattr(backend_module, "evaluate_requested_source", requested_source)
    runtime.grid = SimpleNamespace(
        to_dict=lambda: {"x_domain": [1.0, 1.0]}
    )
    runtime.time = SimpleNamespace(
        to_dict=lambda: {"t0": 0.0, "t1": 1.0, "nt": 3}
    )
    return runtime


def _macro_values(points):
    points = np.asarray(points, dtype=np.float64)
    return {
        "rho": 1.0 + points[:, 0],
        "momentum": points[:, 1:3].copy(),
        "kinetic_energy": 2.0 + np.sum(points[:, 1:3] ** 2, axis=1),
        "raw_second_moment": np.column_stack(
            (
                points[:, 1] ** 2,
                points[:, 1] * points[:, 2],
                points[:, 2] ** 2,
            )
        ),
    }


def _resident_record(
    runtime,
    case_id="resident",
    visibility=CaseVisibility.DEVELOPMENT,
):
    return runtime._resident_record(
        case_id=case_id,
        latent=np.asarray((1.0, 2.0)),
        condition=np.asarray((0.1, 1.0)),
        visibility=visibility,
        round_index=1,
        cohort="development",
        parent_case_id=None,
        variant="base",
    )


def _state(generator):
    checkpoint = GeneratorCheckpoint(generator.info.program_id, object())
    return GeneratorState(checkpoint=checkpoint, base=object())


def _artifact(program):
    return Artifact(
        artifact_id="resident-candidate",
        kind=TaskKind.OBSERVABLE_MODEL,
        name="resident candidate",
        program=program,
        summary="test",
        usage="test",
        round_index=1,
        agent_id="agent",
    )


class _DevelopmentOracleCandidate:
    def __init__(self) -> None:
        self.public_method = None
        self.development_method = None

    def predict_observables(self, case, query, budget=None):
        del budget
        public = open_public_case_session(case)
        self.public_method = public.rho0
        assert self.public_method(np.asarray(((0.2, 0.3),))).shape == (1,)
        solution = open_development_solution_session(case)
        self.development_method = solution.released_macros
        points = np.asarray(
            [
                [point[name] for name in ("t", "x", "y")]
                for point in query.points
            ]
        )
        values = self.development_method(points)
        return AnswerView(
            kind=query.kind,
            fields={
                "rho": tuple(values["rho"]),
                "momentum_x": tuple(values["momentum"][:, 0]),
                "momentum_y": tuple(values["momentum"][:, 1]),
                "kinetic_energy": tuple(values["kinetic_energy"]),
            },
            points=query.points,
            shape=(len(query.points),),
        )


class _OfficialCandidate:
    def __init__(self) -> None:
        self.development_denied = False

    def predict_observables(self, case, query, budget=None):
        del budget
        public = open_public_case_session(case)
        assert public.source(np.zeros((1, 5))).shape == (1,)
        try:
            open_development_solution_session(case)
        except RuntimeError:
            self.development_denied = True
        points = np.asarray(
            [
                [point[name] for name in ("t", "x", "y")]
                for point in query.points
            ]
        )
        values = _macro_values(points)
        return AnswerView(
            kind=query.kind,
            fields={
                "rho": tuple(values["rho"]),
                "momentum_x": tuple(values["momentum"][:, 0]),
                "momentum_y": tuple(values["momentum"][:, 1]),
                "kinetic_energy": tuple(values["kinetic_energy"]),
            },
            points=query.points,
            shape=(len(query.points),),
        )


def test_resident_truth_is_continuous_and_creates_no_case_files(
    resident_runtime,
):
    record = _resident_record(resident_runtime)
    query = AnswerQuery.at_points(
        fields=VPFP_MACRO_FIELDS,
        points=(
            {"t": 0.25, "x": 0.1, "y": 0.2},
            {"t": 0.75, "x": 0.3, "y": 0.4},
        ),
    )

    first = record.truth.answer(query)
    second = record.truth.answer(query)

    assert first == second
    assert first.fields["rho"] == (1.25, 1.75)
    assert len(resident_runtime.api.macro_calls) == 3
    assert tuple(resident_runtime.output_root.iterdir()) == ()


def test_sessions_validate_snapshots_chunk_source_close_and_release(
    resident_runtime,
):
    record = _resident_record(resident_runtime)
    copied = replace(record.case, public_data=deepcopy(record.case.public_data))
    public = resident_runtime.public_session_factory(copied)
    assert public.rho0(np.zeros((3, 2))).shape == (3,)
    assert public.g0(np.zeros((4, 4))).shape == (4,)
    assert public.source(np.zeros((1025, 5))).shape == (1025,)
    assert resident_runtime.api.requested_source_sizes == [1025]
    assert resident_runtime.api.source_chunk_sizes == []
    assert public.usage_snapshot["evaluated_points"] == 1032
    assert public.usage_snapshot["completed_points"] == 1032
    assert public.usage_snapshot["attempted_calls"] == 3
    assert public.usage_snapshot["requested_points"] == 1032
    assert public.usage_snapshot["failed_calls"] == 0
    public.close()
    with pytest.raises(RuntimeError, match="closed"):
        public.rho0(np.zeros((1, 2)))
    assert public.usage_snapshot["attempted_calls"] == 4
    assert public.usage_snapshot["query_calls"] == 3
    assert public.usage_snapshot["requested_points"] == 1033
    assert public.usage_snapshot["failed_calls"] == 1
    assert public.usage_snapshot["by_method"]["rho0"]["failed_calls"] == 1

    tampered = replace(copied, public_data=deepcopy(copied.public_data))
    tampered.public_data["arrays"]["t"][0] = -1.0
    with pytest.raises(ValueError, match="snapshot changed"):
        resident_runtime.public_session_factory(tampered)

    active = resident_runtime.development_solution_session_factory(record.case)
    resident_runtime.release_cases((record.case.case_id,))
    resident_runtime.release_cases((record.case.case_id,))
    with pytest.raises(ValueError, match="no longer active"):
        active.released_macros(np.zeros((1, 3)))
    assert active.usage_snapshot["attempted_calls"] == 1
    assert active.usage_snapshot["requested_points"] == 1
    assert active.usage_snapshot["completed_points"] == 0
    assert active.usage_snapshot["failed_calls"] == 1


def test_explicit_reference_source_preserves_old_512_point_chunks(resident_runtime):
    record = _resident_record(resident_runtime)
    points = np.zeros((1025, 5))
    actual = resident_runtime.evaluate_source_reference(record.case.case_id, points)
    assert actual.shape == (1025,)
    assert resident_runtime.api.source_chunk_sizes == [512, 512, 1]
    assert resident_runtime.api.requested_source_sizes == []


def test_public_session_has_per_call_cap_without_cumulative_quota(
    resident_runtime,
    monkeypatch,
):
    record = _resident_record(resident_runtime)
    session = resident_runtime.public_session_factory(record.case)
    monkeypatch.setattr(
        backend_module, "VPFP_MAX_ORACLE_POINTS_PER_CALL", 3
    )

    for _index in range(20):
        assert session.source(np.zeros((3, 5))).shape == (3,)
    assert session.usage_snapshot["evaluated_points"] == 60
    with pytest.raises(ValueError, match="1..3 points"):
        session.source(np.zeros((4, 5)))
    assert session.usage_snapshot["attempted_calls"] == 21
    assert session.usage_snapshot["requested_points"] == 64
    assert session.usage_snapshot["completed_points"] == 60
    assert session.usage_snapshot["failed_calls"] == 1


def test_session_usage_keeps_validation_and_backend_failures_in_attempts(
    resident_runtime, monkeypatch,
):
    record = _resident_record(resident_runtime)
    session = resident_runtime.public_session_factory(record.case)
    assert set(session.usage_snapshot["by_method"]) == {"source", "rho0", "g0"}
    session.source(np.zeros((2, 5)))
    with pytest.raises(ValueError, match="shape"):
        session.source(np.zeros((3, 4)))
    session.g0(np.zeros((4, 4)))
    with pytest.raises(ValueError, match="finite"):
        session.rho0(np.full((5, 2), np.nan))
    original = ArithmeticError("original source backend failure")

    def fail(*_args):
        raise original

    monkeypatch.setattr(resident_runtime, "evaluate_source", fail)
    with pytest.raises(ArithmeticError) as caught:
        session.source(np.zeros((7, 5)))
    assert caught.value is original
    snapshot = session.usage_snapshot
    assert snapshot["attempted_calls"] == 5
    assert snapshot["requested_points"] == 21
    assert snapshot["completed_points"] == snapshot["evaluated_points"] == 6
    assert snapshot["query_calls"] == 2
    assert snapshot["failed_calls"] == 3
    assert snapshot["backend_wall_seconds"] >= 0.0
    assert "not points entering the math kernel" in snapshot["evaluated_points_semantics"]
    assert snapshot["by_method"]["source"]["attempted_calls"] == 3
    assert snapshot["by_method"]["source"]["requested_points"] == 12
    assert snapshot["by_method"]["source"]["completed_points"] == 2
    assert snapshot["by_method"]["source"]["failed_calls"] == 2
    assert snapshot["by_method"]["rho0"]["failed_calls"] == 1
    assert snapshot["by_method"]["g0"]["completed_points"] == 4
    # Detached nested snapshots cannot mutate the resident's internal counts.
    snapshot["by_method"]["source"]["attempted_calls"] = 900
    assert session.usage_snapshot["by_method"]["source"]["attempted_calls"] == 3


def test_usage_len_failure_does_not_hide_original_backend_exception(
    resident_runtime, monkeypatch,
):
    record = _resident_record(resident_runtime)
    session = resident_runtime.public_session_factory(record.case)

    class Uncountable:
        def __len__(self):
            raise RuntimeError("diagnostic len failure")

    original = ArithmeticError("the actual backend failure")

    def fail(*_args):
        raise original

    monkeypatch.setattr(resident_runtime, "evaluate_source", fail)
    with pytest.raises(ArithmeticError) as caught:
        session.source(Uncountable())
    assert caught.value is original
    assert session.usage_snapshot["attempted_calls"] == 1
    assert session.usage_snapshot["requested_points"] == 0
    assert session.usage_snapshot["unknown_request_size_calls"] == 1
    assert session.usage_snapshot["failed_calls"] == 1
    assert session.usage_snapshot["completed_points"] == 0


def test_successful_array_input_without_len_counts_returned_points(resident_runtime):
    record = _resident_record(resident_runtime)
    session = resident_runtime.public_session_factory(record.case)

    class ArrayWithoutLen:
        def __len__(self):
            raise TypeError("not a sized input")

        def __array__(self, dtype=None, copy=None):
            return np.zeros((3, 5), dtype=dtype or np.float64)

    assert session.source(ArrayWithoutLen()).shape == (3,)
    assert session.usage_snapshot["query_calls"] == 1
    assert session.usage_snapshot["requested_points"] == 0
    assert session.usage_snapshot["unknown_request_size_calls"] == 1
    assert session.usage_snapshot["completed_points"] == 3
    assert session.usage_snapshot["evaluated_points"] == 3
    assert session.usage_snapshot["failed_calls"] == 0


def test_development_macro_usage_counts_points_not_number_of_fields(resident_runtime):
    record = _resident_record(resident_runtime)
    session = resident_runtime.development_solution_session_factory(record.case)
    result = session.released_macros(np.zeros((6, 3)))
    assert result["rho"].shape == (6,)
    assert session.usage_snapshot["completed_points"] == 6
    assert session.usage_snapshot["requested_points"] == 6
    assert session.usage_snapshot["query_calls"] == 1
    assert session.usage_snapshot["by_method"]["released_macros"]["completed_points"] == 6


def test_canonical_evaluator_uses_vpfp_allowlists_and_revokes_sessions(
    resident_runtime,
):
    record = _resident_record(resident_runtime)
    backend = VPFPAnchoredPerturbationBackend(resident_runtime)
    generator = VPFPContinuousGenerator(backend)
    assert generator.case_session_methods(CaseAccessKind.PUBLIC_PROBLEM) == (
        "source",
        "rho0",
        "g0",
    )
    assert generator.case_session_methods(
        CaseAccessKind.DEVELOPMENT_SOLUTION
    ) == ("released_macros", "released_transport_macros")
    query = AnswerQuery.at_points(
        fields=VPFP_MACRO_FIELDS,
        points=({"t": 0.25, "x": 0.1, "y": 0.2},),
    )
    candidate = _DevelopmentOracleCandidate()
    evaluator = AgentEvaluator(
        (record,),
        VPFPObservableTask(),
        EvaluatorPolicy(
            query,
            allow_development_solution_session=True,
            development_coordinate_mode=(
                DevelopmentCoordinateMode.USER_FIXED_QUERY
            ),
        ),
        generator=generator,
        generator_state=_state(generator),
    )

    result = evaluator.evaluate(
        _artifact(candidate),
        EvaluationRequest((candidate_case_id(record.case.case_id),)),
    )

    assert result.primary_error == pytest.approx(0.0)
    with pytest.raises(RuntimeError, match="no longer active"):
        candidate.public_method(np.zeros((1, 2)))
    with pytest.raises(RuntimeError, match="no longer active"):
        candidate.development_method(np.zeros((1, 3)))


def test_official_keeps_public_oracle_but_denies_development_solution(
    resident_runtime,
):
    record = _resident_record(
        resident_runtime,
        case_id="hidden-resident",
        visibility=CaseVisibility.HIDDEN,
    )
    backend = VPFPAnchoredPerturbationBackend(resident_runtime)
    generator = VPFPContinuousGenerator(backend)
    query = AnswerQuery.at_points(
        fields=VPFP_MACRO_FIELDS,
        points=({"t": 0.5, "x": 0.2, "y": 0.3},),
    )
    candidate = _OfficialCandidate()
    verifier = HiddenVerifier(
        VPFPObservableTask(),
        query,
        phase=EvaluationPhase.OFFICIAL,
        generator=generator,
        generator_state=_state(generator),
    )

    result = verifier.evaluate(_artifact(candidate), (record,))

    assert result.primary_error == pytest.approx(0.0)
    assert candidate.development_denied is True


def test_resident_transaction_rollback_and_duplicate_preflight(
    resident_runtime,
    monkeypatch,
):
    resident_runtime.generator = SimpleNamespace(
        config=SimpleNamespace(batch_size=2)
    )

    def fail_after_first(**kwargs):
        record = _resident_record(
            resident_runtime, case_id=kwargs["case_ids"][0]
        )
        kwargs["registered_case_ids"].append(record.case.case_id)
        raise RuntimeError("forced second-case failure")

    monkeypatch.setattr(
        resident_runtime, "_generate_base_batch", fail_after_first
    )
    with pytest.raises(RuntimeError, match="forced second-case failure"):
        resident_runtime.generate_base(
            seeds=(1, 2),
            case_ids=("batch-a", "batch-b"),
            visibility=CaseVisibility.DEVELOPMENT,
            round_index=3,
            cohort="development",
        )
    assert resident_runtime._resident_cases == {}

    existing = _resident_record(resident_runtime, case_id="already-active")
    with pytest.raises(FileExistsError, match="active resident answer"):
        resident_runtime.generate_base(
            seeds=(1,),
            case_ids=(existing.case.case_id,),
            visibility=CaseVisibility.DEVELOPMENT,
            round_index=3,
            cohort="development",
        )
    assert resident_runtime._resident_cases[existing.case.case_id].case is (
        existing.case
    )


def test_backend_validation_rolls_back_resident_transaction(
    resident_runtime,
    monkeypatch,
):
    resident_runtime.generator = SimpleNamespace(
        config=SimpleNamespace(batch_size=1)
    )

    def wrong_visibility(**kwargs):
        record = resident_runtime._resident_record(
            case_id=kwargs["case_ids"][0],
            latent=np.asarray((1.0, 2.0)),
            condition=np.asarray((0.1, 1.0)),
            visibility=CaseVisibility.HIDDEN,
            round_index=kwargs["round_index"],
            cohort=kwargs["cohort"],
            parent_case_id=None,
            variant="base",
        )
        kwargs["registered_case_ids"].append(record.case.case_id)
        return (
            VPFPRuntimeCase(record, private_handle={"latent": 1}),
        )

    monkeypatch.setattr(
        resident_runtime, "_generate_base_batch", wrong_visibility
    )
    backend = VPFPAnchoredPerturbationBackend(resident_runtime)
    request = GenerationRequest(
        round_index=4,
        count=1,
        mode=GeneratorMode.OFFLINE,
        kind=GenerationKind.BASE,
        visibility=CaseVisibility.DEVELOPMENT,
        seed=17,
        purpose=GenerationPurpose.DEVELOPMENT,
        settings={
            "fixed_generator_id": "vpfp-resident-test",
            "fixed_generator_sha256": "a" * 64,
        },
    )

    with pytest.raises(ValueError, match="changed requested envelope"):
        backend.generate(None, None, request)
    assert resident_runtime._resident_cases == {}


def test_public_problem_resource_is_owned_by_vpfp_plugin():
    bundle = vpfp_public_problem_bundle()
    runtime = next(
        resource
        for resource in bundle.resources
        if resource.relative_path.endswith("public_vpfp_runtime.py")
    )
    resource_path = (
        Path(backend_module.__file__).with_name("_public_resources")
        / "public_vpfp_runtime.py"
    )
    assert resource_path.is_file()
    assert runtime.data == resource_path.read_bytes()
    assert "legacy" not in str(resource_path.relative_to(Path.cwd()))
