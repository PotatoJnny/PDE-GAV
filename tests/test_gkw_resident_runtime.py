from __future__ import annotations

import gc
import weakref
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Event, Lock, RLock, Thread
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

import pde_agent_verifier_solver.plugins.gkw.selected_torch_runtime as runtime_module
from pde_agent_verifier_solver.plugins.gkw.contract import GKW_MACRO_FIELDS
from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    Case,
    CaseSource,
    CaseVisibility,
    GenerationKind,
    GenerationPurpose,
    GenerationRequest,
    GeneratorMode,
)


_IMPLEMENTATION_SHA256 = "a" * 64
_PUBLIC_CONDITION = (5.25, 2.75, 0.875, 1.5)
_ANCHOR_KEYS = (
    MappingProxyType(
        {"trajectory": 0, "start_position": 1, "end_position": 2}
    ),
)


class _ValidatedQualification:
    def __init__(self) -> None:
        self.certificate = SimpleNamespace(
            qualification_id="offline-qualified-test-rule"
        )


class _Samples:
    anchor_keys = _ANCHOR_KEYS
    solution_sampling_mode = "local_edit"
    sampled_public_conditions = np.asarray(
        [_PUBLIC_CONDITION], dtype=np.float32
    )

    def save_private(self, *_args, **_kwargs):
        raise AssertionError("production generation must not serialize samples")

    @classmethod
    def load_private(cls, *_args, **_kwargs):
        raise AssertionError("production generation must not replay samples")


class _ForbiddenDecoder:
    config = SimpleNamespace(neural_inference_precision="bfloat16")

    def decode_and_project(self, *_args, **_kwargs):
        raise AssertionError("a resident answer must not replay the decoder")


class _Sampler:
    device = "cpu"
    config = SimpleNamespace(sampling_mode="local_edit")

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def sample(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return _Samples()


class _Trackable:
    def axis_interpolation_matrix(self, axis, times):
        import torch

        assert axis == "t"
        query = torch.as_tensor(times).reshape(-1)
        return torch.ones((query.numel(), 1), dtype=torch.float64)

    def time_values(self, times):
        import torch

        query = torch.as_tensor(times).reshape(-1)
        return torch.ones((query.numel(), 1), dtype=torch.float64)


class _Resident:
    def __init__(
        self,
        *,
        state: object | None = None,
        parent: "_Resident | None" = None,
        qualified: bool = True,
        perturbed: bool = False,
        solution_sampling_mode: str = "local_edit",
    ) -> None:
        self.state = state if state is not None else _Trackable()
        self.base_state = parent.base_state if parent is not None else self.state
        self.parent = parent
        self.solution_sampling_mode = solution_sampling_mode
        self.qualification_provenance = {
            "schema": "test-resident-qualification-v1",
            "passed": qualified,
        }
        if perturbed:
            if solution_sampling_mode == "direct_noise":
                self.qualification_provenance["perturbation"] = {
                    "schema": "experimental-direct-test-rule",
                    "compatibility_id": "experimental-direct-test",
                    "eligibility_basis": (
                        "experimental_direct_noise_compatibility_only"
                    ),
                    "clean_distribution_qualified": False,
                    "production_certificate_eligible": False,
                    "old_local_edit_authority_inherited": False,
                    "passed": qualified,
                }
            else:
                self.qualification_provenance["perturbation"] = {
                    "schema": "offline-qualified-test-rule",
                    "qualification_id": "offline-qualified-test-rule",
                    "passed": qualified,
                }


@dataclass(frozen=True)
class _PublicPacket:
    arrays: object
    scalars: object
    metadata: dict[str, object]


@dataclass(frozen=True)
class _BridgeCase:
    case_id: str
    public_packet: _PublicPacket
    time_nodes: tuple[float, ...]


class _OracleSession:
    def __init__(self, protocol) -> None:
        self.protocol = protocol
        self.calls: list[str] = []
        self.close_calls = 0

    def query_source(self, times, phase_indices):
        self.calls.append("source")
        return np.zeros((len(np.asarray(phase_indices)), 2), dtype=np.float64)

    def query_initial_h(self, phase_indices):
        self.calls.append("initial_h")
        return np.zeros((len(np.asarray(phase_indices)), 2), dtype=np.float64)

    def query_initial_phi(self, field_indices):
        self.calls.append("initial_phi")
        return np.zeros((len(np.asarray(field_indices)), 2), dtype=np.float64)

    def close(self) -> None:
        self.close_calls += 1


class _OracleBackend:
    def __init__(self, protocol) -> None:
        self.protocol = protocol
        self.sessions: list[_OracleSession] = []
        self.close_calls = 0

    @property
    def private_performance_snapshot(self):
        return {
            "source_frame_evaluations": 0,
            "source_frame_cache_hits": 0,
        }

    def open_session(self, *, limits):
        del limits
        session = _OracleSession(self.protocol)
        self.sessions.append(session)
        return session

    def close(self) -> None:
        self.close_calls += 1


def _grid():
    return SimpleNamespace(
        state_shape=(1, 1, 1, 1, 1),
        ns=1,
        nkx=1,
        nky=1,
    )


def _observation(case_id: str, *, perturbed: bool = False):
    grid = _grid()
    state = _Trackable()
    state.contract = SimpleNamespace(grid=grid)
    macro_perturbation = None
    if perturbed:
        state.perturbation = _Trackable()
        macro_perturbation = _Trackable()
        state.perturbation_audit = {
            "parameter_sha256": "b" * 64,
            "adaptive_scale_calibration": {
                "strict_nonlinear_audits": 2,
                "base_operator_window_evaluations": 3,
                "perturbed_operator_window_evaluations": 3,
                "base_operator_time_evaluations": 7,
                "perturbed_operator_time_evaluations": 7,
            },
        }
    packet = _PublicPacket(
        arrays={
            "s": np.asarray([0.0], dtype=np.float64),
            "kx": np.asarray([0.0], dtype=np.float64),
            "ky": np.asarray([0.0], dtype=np.float64),
            "B": np.asarray([1.0], dtype=np.float64),
            "ints": np.asarray([1.0], dtype=np.float64),
            "parseval_ky": np.asarray([1.0], dtype=np.float64),
        },
        scalars={
            "test_scalar": np.float64(1.0),
            **dict(
                zip(
                    runtime_module._PUBLIC_CONDITION_NAMES,
                    _PUBLIC_CONDITION,
                )
            ),
        },
        metadata={"case_id": case_id, "schema": "test-public-packet"},
    )
    return SimpleNamespace(
        passed=True,
        case=_BridgeCase(
            case_id=case_id,
            public_packet=packet,
            time_nodes=(0.0, 1.0),
        ),
        state=state,
        operator=_Trackable(),
        macro_perturbation=macro_perturbation,
    )


def _protocol(case_id: str):
    return {
        "case_id": case_id,
        "time_interval_closed": [0.0, 1.0],
        "phase_grid_shape": [1, 1, 1, 1, 1],
        "field_grid_shape": [1, 1, 1],
        "limits": {
            "max_rows_per_request": 2,
            "max_total_rows": 3,
            "max_unique_source_times": 2,
        },
    }


def _macro_values(value: float, times, field_indices):
    shape = (len(times), len(field_indices), 2)
    return {
        name: np.full(shape, value, dtype=np.float64)
        for name in GKW_MACRO_FIELDS
    }


def _request(
    kind: GenerationKind,
    *,
    purpose: GenerationPurpose = GenerationPurpose.DEVELOPMENT,
    sampling_mode: str = "local_edit",
):
    return GenerationRequest(
        round_index=0,
        count=1,
        mode=GeneratorMode.OFFLINE,
        kind=kind,
        visibility=CaseVisibility.DEVELOPMENT,
        seed=0,
        purpose=purpose,
        settings={
            "fixed_generator_sha256": _IMPLEMENTATION_SHA256,
            "generator_sampling_mode": sampling_mode,
        },
    )


def _public_case(item) -> Case:
    return Case(
        case_id=item.case_id,
        source=CaseSource.GENERATED,
        visibility=CaseVisibility.DEVELOPMENT,
        public_data=item.public_data,
    )


def _runtime(tmp_path: Path):
    runtime = object.__new__(runtime_module.GKWSelectedTorchRuntime)
    runtime.service = SimpleNamespace(
        sampler=_Sampler(),
        decoder=_ForbiddenDecoder(),
        config=SimpleNamespace(maximum_representation_relative_l2=1.0),
        calibrated=SimpleNamespace(
            source=object(), hidden_support=object(), off_grid=object()
        ),
        archived_cases={},
        time_registry={},
        selected_positions=(0,),
        window_length=1,
    )
    runtime.implementation_sha256 = _IMPLEMENTATION_SHA256
    runtime.production_qualification = _ValidatedQualification()
    runtime.experimental_direct_noise_compatibility = None
    runtime.expected_solution_sampling_mode = "local_edit"
    runtime.decoder_precision = "bfloat16"
    runtime.private_root = tmp_path
    runtime.private_storage_root = tmp_path
    runtime.transient_storage_root = tmp_path
    runtime._transaction_parent = tmp_path
    runtime.macro_time_count = 2
    runtime.macro_field_rows = 1
    runtime.formal_macro_time_count = 1
    runtime.maximum_sampling_attempts = 1
    runtime.oracle_max_total_rows = 3
    runtime.enforce_distribution_calibration = False
    runtime._oracle_limits = {
        "max_rows_per_request": 2,
        "max_total_rows": 3,
        "max_unique_source_times": 2,
    }
    runtime._handles = {}
    runtime._pending_resident_handles = {}
    runtime._oracle_backend = None
    runtime._oracle_backend_case_id = None
    runtime._oracle_backend_resident = None
    runtime._oracle_backend_cache_capacity = 8
    runtime._oracle_backend_loads = 0
    runtime._oracle_backend_hits = 0
    runtime._oracle_backend_evictions = 0
    runtime._peak_resident_answers = 0
    runtime._answer_counter = 0
    runtime._case_id_salt = b"test-gkw-case-id-salt".ljust(32, b"x")
    runtime._manufacturing_telemetry = (
        runtime_module._empty_manufacturing_telemetry()
    )
    runtime._lock = Lock()
    runtime._oracle_backend_lock = RLock()

    calls = SimpleNamespace(
        capture=[],
        full_macro=0,
        perturb_macro=0,
        perturb_macro_inputs=[],
        perturb_build=[],
        backend_build=[],
        backends=[],
        persist=0,
        load=0,
    )

    def build_macro_truth(state, operator, arrays):
        import torch

        del operator, arrays
        calls.full_macro += 1
        return SimpleNamespace(
            state=state,
            control_fields={
                name: torch.full(
                    (1, 1, 1, 1),
                    2.0 + 2.0j,
                    dtype=torch.complex128,
                )
                for name in GKW_MACRO_FIELDS
            },
            construction_audit={"schema": "test-clean-macro-truth"},
        )

    def build_perturbation_macro_truth(
        perturbation, operator, arrays, *, private_stage_timings=None
    ):
        if private_stage_timings is not None:
            private_stage_timings["macro_truth_wall_seconds"] = 0.3
        calls.perturb_macro += 1
        calls.perturb_macro_inputs.append((perturbation, operator, arrays))
        del perturbation, operator, arrays
        import torch

        return SimpleNamespace(
            rank_fields={
                name: torch.full(
                    (1, 1, 1, 1),
                    0.25 + 0.0j,
                    dtype=torch.complex128,
                )
                for name in GKW_MACRO_FIELDS
            },
            construction_audit={"schema": "test-perturbation-macro-truth"},
        )

    def capture(observation, parent=None, solution_sampling_mode=None):
        mode = (
            solution_sampling_mode
            if solution_sampling_mode is not None
            else parent.solution_sampling_mode
        )
        calls.capture.append((observation, parent, mode))
        resident = _Resident(
            state=observation.state,
            parent=parent,
            perturbed=parent is not None,
            solution_sampling_mode=mode,
        )
        resident.case_id = observation.case.case_id
        resident.case = observation.case
        return resident

    def perturb(parent, *, case, **kwargs):
        calls.perturb_build.append((parent, case, kwargs))
        timings = kwargs["private_stage_timings"]
        timings["perturbation_sample_wall_seconds"] = 0.1
        timings["operator_precompute_wall_seconds"] = 0.2
        observation = _observation(case.case_id, perturbed=True)
        observation.case = case
        return observation

    def build_backend(resident, **kwargs):
        calls.backend_build.append((id(resident), kwargs))
        backend = _OracleBackend(_protocol(resident.case_id))
        calls.backends.append(backend)
        return backend
    def forbidden_persist(*_args, **_kwargs):
        calls.persist += 1
        raise AssertionError("production must not persist oracle artifacts")

    def forbidden_load(*_args, **_kwargs):
        calls.load += 1
        raise AssertionError("production must not load oracle artifacts")

    runtime._symbols = SimpleNamespace(
        prepare_cases=lambda *_args, **_kwargs: (),
        build_macro_truth=build_macro_truth,
        build_perturbation_macro_truth=build_perturbation_macro_truth,
        oracle_protocol=lambda *, case_id, **_kwargs: _protocol(case_id),
        capture_resident_answer=capture,
        build_qualified_perturbed_observation=perturb,
        build_oracle_backend_from_resident=build_backend,
        persist_oracle_artifacts=forbidden_persist,
        load_oracle_backend=forbidden_load,
    )
    return runtime, calls


def _manufacture_from_observation(
    runtime,
    observation,
    *,
    seed: int,
    parent_handle=None,
    panel=None,
    sampling_mode="local_edit",
):
    return runtime._from_observation(
        observation,
        case_prefix=observation.case.case_id,
        expected_case_id=observation.case.case_id,
        anchor_keys=_ANCHOR_KEYS,
        sampled_public_condition=_PUBLIC_CONDITION,
        seed=seed,
        perturbed=parent_handle is not None,
        parent_handle=parent_handle,
        panel=panel,
        cohort="development",
        panel_profile=runtime_module._DEVELOPMENT_PANEL_PROFILE,
        solution_sampling_mode=sampling_mode,
    )


def test_clean_capture_has_no_online_persistence_or_replay(monkeypatch, tmp_path):
    import torch

    monkeypatch.setattr(
        torch,
        "save",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("production generation must not call torch.save")
        ),
    )
    monkeypatch.setattr(
        torch,
        "load",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("production generation must not call torch.load")
        ),
    )
    runtime, calls = _runtime(tmp_path)
    observation = _observation("clean-resident")
    item = _manufacture_from_observation(runtime, observation, seed=7)
    runtime._track_pending(item)
    runtime.commit_manufactured((item,))

    assert len(calls.capture) == 1
    assert calls.capture[0] == (observation, None, "local_edit")
    assert calls.full_macro == 1
    assert calls.perturb_macro == 0
    assert isinstance(item.hidden_handle, runtime_module._ResidentHandle)
    assert item.hidden_handle.resident is not observation.state
    assert item.audit.answer_first is True
    assert item.audit.passed is True

    # Opening from the completed answer must promote that resident directly;
    # the forbidden decoder/sample persistence hooks remain untouched.
    runtime._open_oracle(_public_case(item)).close()
    assert len(calls.backend_build) == 1
    assert calls.backend_build[0][0] == id(item.hidden_handle.resident)
    assert calls.backend_build[0][1]["device"] == "cpu"
    assert calls.backend_build[0][1]["source_frame_cache_entries"] >= 8
    assert calls.persist == 0
    assert calls.load == 0


def test_perturb_uses_parent_resident_and_adds_only_incremental_macro_truth(
    monkeypatch, tmp_path
):
    runtime, calls = _runtime(tmp_path)
    clean_observation = _observation("clean-parent")
    parent = _manufacture_from_observation(runtime, clean_observation, seed=11)
    runtime._track_pending(parent)
    runtime.commit_manufactured((parent,))

    monkeypatch.setattr(
        runtime,
        "_prepare_case_from_anchor_keys",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("perturbation must reuse the exact parent PDE")
        ),
    )
    monkeypatch.setattr(
        runtime.service.sampler,
        "sample",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("perturbation must not resample the generator")
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_run_bridge",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("perturbation must not replay the decoder bridge")
        ),
    )
    import torch

    monkeypatch.setattr(
        torch,
        "isfinite",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("qualified online macro path must not finite-scan")
        ),
    )

    child = runtime.manufacture_perturbed(
        None,
        parent,
        _request(GenerationKind.PERTURBED),
        seed=13,
    )

    assert len(calls.perturb_build) == 1
    built_parent, built_case, built_kwargs = calls.perturb_build[0]
    assert built_parent is parent.hidden_handle.resident
    assert built_case.case_id.startswith("gkw-u-pert-")
    assert built_case.case_id.endswith("-0000")
    assert f"{13:016x}" not in built_case.case_id
    assert parent.case_id not in built_case.case_id
    assert built_case is not clean_observation.case
    assert built_case.public_packet is not clean_observation.case.public_packet
    assert (
        built_case.public_packet.arrays
        is clean_observation.case.public_packet.arrays
    )
    assert (
        built_case.public_packet.scalars
        is clean_observation.case.public_packet.scalars
    )
    assert built_case.public_packet.metadata["case_id"] == built_case.case_id
    assert (
        clean_observation.case.public_packet.metadata["case_id"]
        == clean_observation.case.case_id
    )
    assert built_kwargs["qualification"] is (
        runtime.production_qualification
    )
    assert built_kwargs["seed"] == 13
    assert built_kwargs["device"] == "cpu"
    stage_timings = built_kwargs["private_stage_timings"]
    assert set(stage_timings) == {
        "perturbation_sample_wall_seconds",
        "operator_precompute_wall_seconds",
        "macro_truth_wall_seconds",
        "macro_truth_to_public_pairs_wall_seconds",
    }
    assert stage_timings["perturbation_sample_wall_seconds"] == 0.1
    assert stage_timings["operator_precompute_wall_seconds"] == 0.2
    assert stage_timings["macro_truth_wall_seconds"] == 0.3
    assert stage_timings[
        "macro_truth_to_public_pairs_wall_seconds"
    ] >= 0.0
    assert len(calls.capture) == 2
    assert calls.capture[1][1] is parent.hidden_handle.resident
    assert child.hidden_handle.resident.base_state is parent.hidden_handle.resident.base_state
    assert child.hidden_handle.parent_answer_id == parent.hidden_handle.answer_id
    assert (
        child.hidden_handle.sampled_public_condition
        == parent.hidden_handle.sampled_public_condition
        == _PUBLIC_CONDITION
    )
    assert child.audit.perturbation_qualification_id == "offline-qualified-test-rule"
    assert child.audit.perturbation_qualification_seed == 13
    assert child.public_data["arrays"] is parent.public_data["arrays"]
    assert child.public_data["scalars"] is parent.public_data["scalars"]
    assert "macro_query" not in child.public_data
    assert calls.full_macro == 1
    assert calls.perturb_macro == 1
    macro_perturbation, _operator, _arrays = calls.perturb_macro_inputs[0]
    assert macro_perturbation is calls.capture[1][0].macro_perturbation
    assert macro_perturbation is not calls.capture[1][0].state.perturbation
    telemetry = runtime.private_manufacturing_telemetry
    assert telemetry["perturbed_generation_calls"] == 1
    assert telemetry["perturbed_generation_cases"] == 1
    assert telemetry["perturbed_generation_wall_seconds"] >= 0.0
    assert telemetry["perturbed_sample_wall_seconds"] == pytest.approx(0.1)
    assert telemetry["perturbed_operator_precompute_wall_seconds"] == (
        pytest.approx(0.2)
    )
    assert telemetry[
        "perturbed_macro_truth_host_dispatch_wall_seconds"
    ] == pytest.approx(0.3)
    assert telemetry[
        "perturbed_macro_truth_to_public_pairs_wall_seconds"
    ] >= 0.0
    assert telemetry["perturbed_online_numerical_audits"] == 0
    assert telemetry["perturbed_online_parameter_hashes"] == 0
    assert telemetry["perturbed_failed_seed_replacements"] == 0
    off_panel_query = AnswerQuery.at_points(
        fields=GKW_MACRO_FIELDS,
        points=(
            {
                "t": 0.375,
                "s_index": 0,
                "kx_index": 0,
                "ky_index": 0,
            },
        ),
    )
    parent_view = parent.truth.answer(off_panel_query)
    child_view = child.truth.answer(off_panel_query)
    for name in GKW_MACRO_FIELDS:
        expected = np.asarray(parent_view.fields[name]).reshape(1, 2)
        expected[:, 0] += 0.25
        np.testing.assert_array_equal(
            np.asarray(child_view.fields[name]).reshape(1, 2), expected
        )


def test_qualified_perturbation_telemetry_reports_zero_online_audits_and_hashes(
    tmp_path,
):
    runtime, _calls = _runtime(tmp_path)
    clean_observation = _observation("telemetry-clean")

    runtime._record_generation(
        "clean",
        wall_seconds=0.25,
        accepted_cases=1,
    )
    runtime._record_clean_audit(clean_observation)

    telemetry = runtime.private_manufacturing_telemetry
    assert telemetry["clean_generation_calls"] == 1
    assert telemetry["clean_generation_cases"] == 1
    assert telemetry["clean_generation_wall_seconds"] == pytest.approx(0.25)
    assert telemetry["clean_strict_nonlinear_audits_derived"] == 1
    assert telemetry["clean_operator_frame_evaluations_derived"] == 2
    for name in (
        "perturbed_online_numerical_audits",
        "perturbed_online_parameter_hashes",
        "perturbed_failed_seed_replacements",
        "unified_serialization_calls",
        "unified_state_hash_calls",
        "unified_replay_calls",
    ):
        assert telemetry[name] == 0


def test_direct_noise_inherits_exact_parent_pde_and_private_provenance(tmp_path):
    runtime, calls = _runtime(tmp_path)
    compatibility = SimpleNamespace(compatibility_id="experimental-direct-test")
    runtime.production_qualification = None
    runtime.experimental_direct_noise_compatibility = compatibility
    runtime.expected_solution_sampling_mode = "direct_noise"
    runtime.service.sampler.config = SimpleNamespace(sampling_mode="direct_noise")
    runtime._symbols.build_direct_noise_perturbed_observation = (
        runtime._symbols.build_qualified_perturbed_observation
    )
    runtime._symbols.build_qualified_perturbed_observation = (
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("direct noise must not use local-edit authority")
        )
    )

    clean_observation = _observation("direct-clean-parent")
    parent = _manufacture_from_observation(
        runtime,
        clean_observation,
        seed=73,
        sampling_mode="direct_noise",
    )
    runtime._track_pending(parent)
    runtime.commit_manufactured((parent,))
    child = runtime.manufacture_perturbed(
        None,
        parent,
        _request(
            GenerationKind.PERTURBED,
            sampling_mode="direct_noise",
        ),
        seed=79,
    )

    assert len(calls.perturb_build) == 1
    built_parent, built_case, built_kwargs = calls.perturb_build[0]
    assert built_parent is parent.hidden_handle.resident
    assert built_kwargs["compatibility"] is compatibility
    assert "qualification" not in built_kwargs
    assert built_case is not clean_observation.case
    assert built_case.public_packet is not clean_observation.case.public_packet
    assert (
        built_case.public_packet.arrays
        is clean_observation.case.public_packet.arrays
    )
    assert (
        built_case.public_packet.scalars
        is clean_observation.case.public_packet.scalars
    )
    assert built_case.public_packet.metadata["case_id"] == child.case_id
    assert clean_observation.case.public_packet.metadata == {
        "case_id": parent.case_id,
        "schema": "test-public-packet",
    }
    assert parent.hidden_handle.solution_sampling_mode == "direct_noise"
    assert dict(parent.hidden_handle.sampling_provenance) == {
        "schema": "gkw_unified_solution_sampling_provenance_v1",
        "solution_sampling_mode": "direct_noise",
        "inherited_from_clean_parent": False,
        "parent_answer_id": None,
    }
    assert dict(child.hidden_handle.sampling_provenance) == {
        "schema": "gkw_unified_solution_sampling_provenance_v1",
        "solution_sampling_mode": "direct_noise",
        "inherited_from_clean_parent": True,
        "parent_answer_id": parent.hidden_handle.answer_id,
    }
    with pytest.raises(TypeError):
        child.hidden_handle.sampling_provenance["solution_sampling_mode"] = (
            "local_edit"
        )
    assert child.audit.perturbation_qualification_id == (
        "experimental-direct-test"
    )
    assert "solution_sampling_mode" not in child.public_data

    runtime.commit_manufactured((child,))
    runtime.release_manufactured(
        (parent.hidden_handle, child.hidden_handle)
    )
    assert runtime._pending_resident_handles == {}
    assert runtime._handles == {}


@pytest.mark.parametrize("sampling_mode", ("local_edit", "direct_noise"))
@pytest.mark.parametrize("corruption", ("time_nodes", "grid", "arrays"))
def test_perturbation_builder_cannot_change_parent_public_geometry(
    tmp_path,
    sampling_mode,
    corruption,
):
    runtime, _calls = _runtime(tmp_path)
    builder = runtime._symbols.build_qualified_perturbed_observation
    if sampling_mode == "direct_noise":
        runtime.production_qualification = None
        runtime.experimental_direct_noise_compatibility = SimpleNamespace(
            compatibility_id="experimental-direct-test"
        )
        runtime.expected_solution_sampling_mode = "direct_noise"
        runtime.service.sampler.config = SimpleNamespace(
            sampling_mode="direct_noise"
        )

    parent = _manufacture_from_observation(
        runtime,
        _observation(f"{sampling_mode}-geometry-parent"),
        seed=83,
        sampling_mode=sampling_mode,
    )
    runtime._track_pending(parent)
    runtime.commit_manufactured((parent,))

    def malicious_builder(parent_resident, *, case, **kwargs):
        observation = builder(parent_resident, case=case, **kwargs)
        if corruption == "time_nodes":
            observation.case = replace(case, time_nodes=(0.0, 2.0))
        elif corruption == "grid":
            observation.state.contract = SimpleNamespace(
                grid=SimpleNamespace(state_shape=(1, 1, 1, 2, 1))
            )
        else:
            observation.case = replace(
                case,
                public_packet=replace(
                    case.public_packet,
                    arrays=dict(case.public_packet.arrays),
                ),
            )
        return observation

    if sampling_mode == "direct_noise":
        runtime._symbols.build_direct_noise_perturbed_observation = (
            malicious_builder
        )
    else:
        runtime._symbols.build_qualified_perturbed_observation = (
            malicious_builder
        )

    with pytest.raises(
        RuntimeError, match="changed the parent public PDE/domain/grid"
    ):
        runtime.manufacture_perturbed(
            None,
            parent,
            _request(
                GenerationKind.PERTURBED,
                sampling_mode=sampling_mode,
            ),
            seed=89,
        )

    assert runtime._pending_resident_handles == {}
    assert runtime._handles == {parent.case_id: parent.hidden_handle}


def test_direct_noise_failures_roll_back_without_consuming_parent(
    monkeypatch, tmp_path
):
    runtime, _calls = _runtime(tmp_path)
    compatibility = SimpleNamespace(compatibility_id="experimental-direct-test")
    runtime.production_qualification = None
    runtime.experimental_direct_noise_compatibility = compatibility
    runtime.expected_solution_sampling_mode = "direct_noise"
    runtime.service.sampler.config = SimpleNamespace(sampling_mode="direct_noise")
    direct_builder = runtime._symbols.build_qualified_perturbed_observation
    runtime._symbols.build_direct_noise_perturbed_observation = direct_builder
    parent = _manufacture_from_observation(
        runtime,
        _observation("direct-failure-parent"),
        seed=83,
        sampling_mode="direct_noise",
    )
    runtime._track_pending(parent)
    runtime.commit_manufactured((parent,))
    request = _request(
        GenerationKind.PERTURBED,
        sampling_mode="direct_noise",
    )

    builder_failure = RuntimeError("injected direct-noise builder failure")
    runtime._symbols.build_direct_noise_perturbed_observation = (
        lambda *_args, **_kwargs: (_ for _ in ()).throw(builder_failure)
    )
    with pytest.raises(RuntimeError, match="builder failure") as caught:
        runtime.manufacture_perturbed(None, parent, request, seed=89)
    assert caught.value is builder_failure
    assert runtime._pending_resident_handles == {}
    assert runtime._handles == {parent.case_id: parent.hidden_handle}

    runtime._symbols.build_direct_noise_perturbed_observation = direct_builder
    original_capture = runtime._symbols.capture_resident_answer

    def overstate_authority(observation, parent=None, **kwargs):
        resident = original_capture(observation, parent=parent, **kwargs)
        resident.qualification_provenance["perturbation"][
            "production_certificate_eligible"
        ] = True
        return resident

    runtime._symbols.capture_resident_answer = overstate_authority
    with pytest.raises(RuntimeError, match="overstates its authority"):
        runtime.manufacture_perturbed(None, parent, request, seed=97)
    assert runtime._pending_resident_handles == {}
    assert runtime._handles == {parent.case_id: parent.hidden_handle}

    runtime._symbols.capture_resident_answer = original_capture

    def partially_track_then_fail(item):
        runtime._pending_resident_handles[item.case_id] = item.hidden_handle
        raise RuntimeError("injected pending publication failure")

    monkeypatch.setattr(runtime, "_track_pending", partially_track_then_fail)
    with pytest.raises(RuntimeError, match="pending publication failure"):
        runtime.manufacture_perturbed(None, parent, request, seed=101)
    assert runtime._pending_resident_handles == {}
    assert runtime._handles == {parent.case_id: parent.hidden_handle}

    runtime.release_manufactured((parent.hidden_handle,))
    assert runtime._handles == {}


def test_development_completed_solution_oracle_uses_agent_coordinates_and_points(
    tmp_path,
):
    runtime, _calls = _runtime(tmp_path)
    item = _manufacture_from_observation(
        runtime, _observation("completed-clean"), seed=15
    )
    runtime._track_pending(item)
    runtime.commit_manufactured((item,))

    session = runtime.development_solution_session_factory(_public_case(item))
    requested_times = np.asarray([0.125], dtype=np.float64)
    requested_rows = np.asarray([[0, 0, 0], [0, 0, 0]], dtype=np.int64)
    values = session.query_completed_macros(requested_times, requested_rows)
    assert set(values) == set(GKW_MACRO_FIELDS)
    for value in values.values():
        assert value.shape == (1, 2, 2)
        assert value.flags.writeable is False
        np.testing.assert_array_equal(value, np.full((1, 2, 2), 2.0))

    # Accounting observes the Cartesian product without imposing a total quota.
    session.query_completed_macros(requested_times, requested_rows)
    assert session.usage_snapshot["query_calls"] == 2
    assert session.usage_snapshot["evaluated_points"] == 4
    assert session.usage_snapshot["backend_wall_seconds"] >= 0.0
    with pytest.raises(ValueError, match="request point count"):
        session.query_completed_macros(
            np.asarray([0.125, 0.25], dtype=np.float64), requested_rows
        )
    session.close()
    with pytest.raises(RuntimeError, match="closed"):
        session.query_completed_macros(
            requested_times, np.asarray([[0, 0, 0]], dtype=np.int64)
        )


def test_development_completed_solution_oracle_composes_perturbation(tmp_path):
    runtime, _calls = _runtime(tmp_path)
    parent = _manufacture_from_observation(
        runtime, _observation("completed-parent"), seed=16
    )
    runtime._track_pending(parent)
    runtime.commit_manufactured((parent,))
    child = _manufacture_from_observation(
        runtime,
        _observation("completed-child", perturbed=True),
        seed=17,
        parent_handle=parent.hidden_handle,
        panel=None,
    )
    runtime._track_pending(child)
    runtime.commit_manufactured((child,))

    session = runtime.development_solution_session_factory(_public_case(child))
    values = session.query_completed_macros(
        np.asarray([0.125], dtype=np.float64),
        np.asarray([[0, 0, 0]], dtype=np.int64),
    )
    for value in values.values():
        expected = np.full((1, 1, 2), 2.0, dtype=np.float64)
        expected[..., 0] += 0.25
        np.testing.assert_array_equal(value, expected)


def test_completed_solution_oracle_rejects_hidden_case_before_access(tmp_path):
    runtime, _calls = _runtime(tmp_path)
    item = _manufacture_from_observation(
        runtime, _observation("completed-hidden-gate"), seed=18
    )
    runtime._track_pending(item)
    runtime.commit_manufactured((item,))
    hidden = Case(
        case_id=item.case_id,
        source=CaseSource.GENERATED,
        visibility=CaseVisibility.HIDDEN,
        public_data=item.public_data,
    )
    with pytest.raises(PermissionError, match="DEVELOPMENT-only"):
        runtime.development_solution_session_factory(hidden)


def test_capacity_one_backend_is_built_once_per_case_and_sessions_meter_alone(
    tmp_path,
):
    runtime, calls = _runtime(tmp_path)
    first = _manufacture_from_observation(
        runtime, _observation("resident-one"), seed=17
    )
    second = _manufacture_from_observation(
        runtime, _observation("resident-two"), seed=19
    )
    for item in (first, second):
        runtime._track_pending(item)
        runtime.commit_manufactured((item,))

    case_one = _public_case(first)
    session_a = runtime.public_session_factory(case_one)
    session_b = runtime.public_session_factory(case_one)
    phase = np.zeros((1, 5), dtype=np.int64)
    assert session_a.query_source(np.asarray([0.25]), phase).shape == (1, 2)
    assert session_a.query_source(np.asarray([0.25]), phase).shape == (1, 2)
    assert session_b.query_source(np.asarray([0.25]), phase).shape == (1, 2)
    assert len(calls.backend_build) == 1
    assert len(calls.backends[0].sessions) == 2

    # Each wrapper records actual use but imposes no lifetime row quota.
    assert session_b.query_source(np.asarray([0.5]), phase).shape == (1, 2)
    assert session_b.query_source(np.asarray([0.5]), phase).shape == (1, 2)
    assert session_b.query_source(np.asarray([0.75]), phase).shape == (1, 2)
    assert session_b.usage_snapshot["query_calls"] == 4
    assert session_b.usage_snapshot["queried_rows"] == 4

    case_two = _public_case(second)
    session_c = runtime.public_session_factory(case_two)
    assert session_c.query_source(np.asarray([0.25]), phase).shape == (1, 2)
    assert len(calls.backend_build) == 2
    assert runtime._oracle_backend_case_id == second.case_id
    assert runtime.private_oracle_telemetry["backend_loads"] == 2
    assert runtime.private_oracle_telemetry["backend_hits"] >= 1
    assert runtime.private_oracle_telemetry["backend_evictions"] == 1
    assert calls.backends[0].close_calls == 1

    for session in (session_a, session_b, session_c):
        session.close()


@pytest.mark.parametrize("competing_operation", ("switch", "release"))
def test_inflight_oracle_query_blocks_backend_switch_or_release(
    tmp_path,
    competing_operation,
):
    runtime, _calls = _runtime(tmp_path)
    first = _manufacture_from_observation(
        runtime, _observation("resident-concurrent-one"), seed=17
    )
    second = _manufacture_from_observation(
        runtime, _observation("resident-concurrent-two"), seed=19
    )
    for item in (first, second):
        runtime._track_pending(item)
        runtime.commit_manufactured((item,))

    query_entered = Event()
    allow_query_to_finish = Event()
    competing_done = Event()
    errors = []
    results = []
    backends = []

    class BlockingSession(_OracleSession):
        def __init__(self, protocol, backend):
            super().__init__(protocol)
            self.backend = backend

        def query_source(self, times, phase_indices):
            query_entered.set()
            if not allow_query_to_finish.wait(timeout=2.0):
                raise RuntimeError("test did not release the blocked GKW query")
            if self.backend.close_calls:
                raise RuntimeError("GKW backend closed during an active query")
            return super().query_source(times, phase_indices)

    class BlockingBackend(_OracleBackend):
        def open_session(self, *, limits):
            del limits
            session = BlockingSession(self.protocol, self)
            self.sessions.append(session)
            return session

    first_resident = first.hidden_handle.resident

    def build_backend(resident, **kwargs):
        del kwargs
        backend_type = (
            BlockingBackend if resident is first_resident else _OracleBackend
        )
        backend = backend_type(_protocol(resident.case_id))
        backends.append(backend)
        return backend

    runtime._symbols.build_oracle_backend_from_resident = build_backend
    first_session = runtime.public_session_factory(_public_case(first))
    second_session = runtime.public_session_factory(_public_case(second))
    phase = np.zeros((1, 5), dtype=np.int64)

    def run_query():
        try:
            results.append(first_session.query_source(0.25, phase))
        except BaseException as exc:
            errors.append(exc)

    query_thread = Thread(target=run_query)
    query_thread.start()
    assert query_entered.wait(timeout=2.0)

    def run_competing_operation():
        try:
            if competing_operation == "switch":
                second_session.query_source(0.25, phase)
            else:
                runtime.release_manufactured((first.hidden_handle,))
        except BaseException as exc:
            errors.append(exc)
        finally:
            competing_done.set()

    competing_thread = Thread(target=run_competing_operation)
    competing_thread.start()
    assert not competing_done.wait(timeout=0.05)
    assert backends[0].close_calls == 0

    allow_query_to_finish.set()
    query_thread.join(timeout=2.0)
    competing_thread.join(timeout=2.0)
    assert not query_thread.is_alive()
    assert not competing_thread.is_alive()
    assert competing_done.is_set()
    assert errors == []
    assert len(results) == 1
    assert results[0].shape == (1, 2)
    assert backends[0].close_calls == 1

    first_session.close()
    second_session.close()


def test_release_exact_handle_is_idempotent_and_drops_backend_and_answer(
    tmp_path,
):
    runtime, calls = _runtime(tmp_path)
    observation = _observation("resident-release")
    item = _manufacture_from_observation(runtime, observation, seed=23)
    handle = item.hidden_handle
    resident = handle.resident
    state = resident.state
    operator = observation.operator
    resident_ref = weakref.ref(resident)
    state_ref = weakref.ref(state)
    operator_ref = weakref.ref(operator)
    runtime._track_pending(item)
    runtime.commit_manufactured((item,))

    session = runtime.public_session_factory(_public_case(item))
    session.query_source(
        np.asarray([0.25]), np.zeros((1, 5), dtype=np.int64)
    )
    backend = calls.backends[-1]
    session.close()
    runtime.release_manufactured((handle,))
    runtime.release_manufactured((handle,))

    assert runtime._handles == {}
    assert runtime._oracle_backend is None
    assert runtime._oracle_backend_case_id is None
    assert runtime._oracle_backend_resident is None
    assert backend.close_calls == 1
    telemetry = runtime.private_manufacturing_telemetry
    assert telemetry["resident_release_calls"] == 1
    assert telemetry["resident_answers_released"] == 1
    assert telemetry["resident_backend_releases"] == 1
    with pytest.raises(KeyError, match="registered resident"):
        runtime.public_session_factory(_public_case(item))

    del session, backend, operator, state, resident, handle, item, observation
    calls.capture.clear()
    calls.backends.clear()
    gc.collect()
    assert resident_ref() is None
    assert state_ref() is None
    assert operator_ref() is None


def test_clean_cleanup_failure_rolls_back_pending_resident(monkeypatch, tmp_path):
    runtime, _calls = _runtime(tmp_path)
    item = _manufacture_from_observation(
        runtime, _observation("resident-cleanup-failure"), seed=25
    )
    monkeypatch.setattr(runtime, "_prepare_case", lambda *_args: object())
    monkeypatch.setattr(
        runtime,
        "_decode_case",
        lambda *_args, **_kwargs: item,
    )
    monkeypatch.setattr(
        runtime,
        "_discard_transaction_root",
        lambda _root: (_ for _ in ()).throw(
            RuntimeError("synthetic transaction cleanup failure")
        ),
    )

    with pytest.raises(RuntimeError, match="transaction cleanup failure"):
        runtime._manufacture_clean_attempt(
            samples=_Samples(),
            prefix="cleanup-failure",
            seed=25,
            cohort="development",
            panel_profile=runtime_module._DEVELOPMENT_PANEL_PROFILE,
        )

    assert runtime._pending_resident_handles == {}
    assert runtime._handles == {}


def test_release_retries_backend_close_before_dropping_resident(tmp_path):
    runtime, calls = _runtime(tmp_path)
    item = _manufacture_from_observation(
        runtime, _observation("resident-close-retry"), seed=27
    )
    handle = item.hidden_handle
    runtime._track_pending(item)
    runtime.commit_manufactured((item,))
    session = runtime.public_session_factory(_public_case(item))
    session.query_source(
        np.asarray([0.25]), np.zeros((1, 5), dtype=np.int64)
    )
    session.close()
    backend = calls.backends[-1]
    close_calls = 0

    def fail_once():
        nonlocal close_calls
        close_calls += 1
        if close_calls == 1:
            raise RuntimeError("transient backend close")

    backend.close = fail_once
    with pytest.raises(RuntimeError, match="transient backend close"):
        runtime.release_manufactured((handle,))
    assert runtime._handles[item.case_id] is handle
    assert runtime._oracle_backend is backend
    assert runtime.private_oracle_telemetry["backend_evictions"] == 0

    runtime.release_manufactured((handle,))
    assert close_calls == 2
    assert runtime._handles == {}
    assert runtime._oracle_backend is None
    assert runtime.private_oracle_telemetry["backend_evictions"] == 1


def test_clean_batch_failure_aborts_each_pending_resident_once(
    monkeypatch, tmp_path
):
    runtime, _calls = _runtime(tmp_path)
    made = []

    def manufacture(request, seed, index):
        del request
        if index == 1:
            raise RuntimeError("synthetic second-case failure")
        item = _manufacture_from_observation(
            runtime, _observation(f"pending-{seed}"), seed=seed
        )
        runtime._track_pending(item)
        made.append(item)
        return item

    monkeypatch.setattr(runtime, "_manufacture_clean_case", manufacture)
    original_abort = runtime.abort_manufactured
    aborts = []

    def abort(items):
        aborts.append(tuple(item.hidden_handle for item in items))
        original_abort(items)

    monkeypatch.setattr(runtime, "abort_manufactured", abort)
    with pytest.raises(RuntimeError, match="second-case failure"):
        runtime.manufacture_clean(
            None,
            _request(GenerationKind.BASE),
            (29, 31),
        )

    assert aborts == [(made[0].hidden_handle,)]
    assert runtime._pending_resident_handles == {}
    assert runtime._handles == {}
    telemetry = runtime.private_manufacturing_telemetry
    assert telemetry["clean_generation_calls"] == 1
    assert telemetry["clean_generation_cases"] == 0
    assert telemetry["clean_generation_wall_seconds"] >= 0.0


@pytest.mark.parametrize("observation_passed,certificate_passed", [(False, True), (True, False)])
def test_resident_manufacturing_fails_closed_without_completed_certificate(
    tmp_path, observation_passed, certificate_passed
):
    runtime, calls = _runtime(tmp_path)
    observation = _observation("unqualified")
    observation.passed = observation_passed

    def capture(row, parent=None, solution_sampling_mode=None):
        del row, parent, solution_sampling_mode
        return _Resident(qualified=certificate_passed)

    runtime._symbols.capture_resident_answer = capture
    with pytest.raises(RuntimeError, match="audited observation|offline qualification"):
        _manufacture_from_observation(runtime, observation, seed=37)
    assert runtime._pending_resident_handles == {}
    assert runtime._handles == {}
    assert calls.backend_build == []


def _stage_resident(runtime, case_id: str, seed: int):
    item = _manufacture_from_observation(
        runtime,
        _observation(case_id),
        seed=seed,
    )
    runtime._track_pending(item)
    return item


def test_resident_registry_commit_and_abort_are_atomic_and_retryable(tmp_path):
    runtime, _calls = _runtime(tmp_path)
    first = _stage_resident(runtime, "atomic-first", 41)
    second = _stage_resident(runtime, "atomic-second", 43)
    untracked = _manufacture_from_observation(
        runtime,
        _observation("atomic-untracked"),
        seed=47,
    )
    before_handles = dict(runtime._handles)
    before_pending = dict(runtime._pending_resident_handles)

    with pytest.raises(RuntimeError, match="not pending"):
        runtime.commit_manufactured((first, untracked))
    assert runtime._handles == before_handles
    assert runtime._pending_resident_handles == before_pending

    with pytest.raises(RuntimeError, match="duplicate case ids"):
        runtime.commit_manufactured((first, first))
    assert runtime._handles == before_handles
    assert runtime._pending_resident_handles == before_pending

    runtime.commit_manufactured((first, second))
    assert runtime._pending_resident_handles == {}
    assert runtime._handles == {
        first.case_id: first.hidden_handle,
        second.case_id: second.hidden_handle,
    }

    committed = dict(runtime._handles)
    with pytest.raises(RuntimeError, match="cannot abort a committed"):
        runtime.abort_manufactured((first,))
    assert runtime._handles == committed
    assert runtime._pending_resident_handles == {}

    collision = _manufacture_from_observation(
        runtime,
        _observation(first.case_id),
        seed=51,
    )
    fresh = _stage_resident(runtime, "atomic-fresh", 52)
    runtime._pending_resident_handles[collision.case_id] = (
        collision.hidden_handle
    )
    before_collision_handles = dict(runtime._handles)
    before_collision_pending = dict(runtime._pending_resident_handles)
    with pytest.raises(RuntimeError, match="reused a registered case id"):
        runtime.commit_manufactured((fresh, collision))
    assert runtime._handles == before_collision_handles
    assert runtime._pending_resident_handles == before_collision_pending
    runtime.abort_manufactured((fresh, collision))
    assert runtime._handles == committed
    assert runtime._pending_resident_handles == {}

    third = _stage_resident(runtime, "atomic-third", 53)
    fourth = _stage_resident(runtime, "atomic-fourth", 59)
    before_pending = dict(runtime._pending_resident_handles)
    with pytest.raises(RuntimeError, match="not pending"):
        runtime.abort_manufactured((third, untracked))
    assert runtime._handles == committed
    assert runtime._pending_resident_handles == before_pending

    runtime.abort_manufactured((third, fourth))
    assert runtime._handles == committed
    assert runtime._pending_resident_handles == {}


@pytest.mark.parametrize("failure_stage", ("prepare", "decode"))
def test_clean_prepare_or_decode_failure_cleans_root_and_same_prefix_retries(
    monkeypatch,
    tmp_path,
    failure_stage,
):
    runtime, _calls = _runtime(tmp_path)
    samples = _Samples()
    prefix = "same-clean-prefix"
    primary = RuntimeError(f"synthetic {failure_stage} failure")

    def prepare(_samples, requested_prefix):
        assert requested_prefix == prefix
        if failure_stage == "prepare":
            raise primary
        return SimpleNamespace(case_id=f"{requested_prefix}-0000")

    def decode_failure(*_args, run_root, **_kwargs):
        run_root.mkdir(parents=True)
        (run_root / "partial.txt").write_text("partial", encoding="utf-8")
        raise primary

    monkeypatch.setattr(runtime, "_prepare_case", prepare)
    if failure_stage == "decode":
        monkeypatch.setattr(runtime, "_decode_case", decode_failure)
    before = tuple(tmp_path.iterdir())

    with pytest.raises(RuntimeError) as caught:
        runtime._manufacture_clean_attempt(
            samples=samples,
            prefix=prefix,
            seed=61,
            cohort="development",
            panel_profile=runtime_module._DEVELOPMENT_PANEL_PROFILE,
        )
    assert caught.value is primary
    assert tuple(tmp_path.iterdir()) == before
    assert runtime._pending_resident_handles == {}
    assert runtime._handles == {}

    monkeypatch.setattr(
        runtime,
        "_prepare_case",
        lambda _samples, requested_prefix: SimpleNamespace(
            case_id=f"{requested_prefix}-0000"
        ),
    )

    def decode_success(_samples, bridge_case, **kwargs):
        assert kwargs["case_prefix"] == prefix
        assert kwargs["run_root"].parent.parent == tmp_path
        return _manufacture_from_observation(
            runtime,
            _observation(bridge_case.case_id),
            seed=61,
        )

    monkeypatch.setattr(runtime, "_decode_case", decode_success)
    retry = runtime._manufacture_clean_attempt(
        samples=samples,
        prefix=prefix,
        seed=61,
        cohort="development",
        panel_profile=runtime_module._DEVELOPMENT_PANEL_PROFILE,
    )
    assert retry.case_id == f"{prefix}-0000"
    assert runtime._pending_resident_handles == {
        retry.case_id: retry.hidden_handle
    }
    runtime.commit_manufactured((retry,))
    assert runtime._handles == {retry.case_id: retry.hidden_handle}
    assert tuple(tmp_path.iterdir()) == before


def test_clean_failure_preserves_primary_when_transaction_cleanup_also_fails(
    monkeypatch,
    tmp_path,
):
    runtime, _calls = _runtime(tmp_path)
    primary = RuntimeError("synthetic prepare failure")
    cleanup = RuntimeError("synthetic cleanup failure")
    monkeypatch.setattr(
        runtime,
        "_prepare_case",
        lambda *_args: (_ for _ in ()).throw(primary),
    )
    monkeypatch.setattr(
        runtime,
        "_discard_transaction_root",
        lambda _root: (_ for _ in ()).throw(cleanup),
    )

    with pytest.raises(RuntimeError) as caught:
        runtime._manufacture_clean_attempt(
            samples=_Samples(),
            prefix="cleanup-note",
            seed=67,
            cohort="development",
            panel_profile=runtime_module._DEVELOPMENT_PANEL_PROFILE,
        )
    assert caught.value is primary
    assert any(
        "transaction cleanup also failed" in note
        and "synthetic cleanup failure" in note
        for note in getattr(caught.value, "__notes__", ())
    )
    assert runtime._pending_resident_handles == {}
    assert runtime._handles == {}


def _committed_clean_parent(runtime, case_id: str, seed: int):
    parent = _stage_resident(runtime, case_id, seed)
    runtime.commit_manufactured((parent,))
    return parent


def test_qualified_perturbation_high_seed_is_mapped_once_and_never_retried(
    monkeypatch,
    tmp_path,
):
    runtime, _calls = _runtime(tmp_path)
    runtime.maximum_sampling_attempts = 8
    parent = _committed_clean_parent(runtime, "perturb-parent", 71)
    monkeypatch.setattr(
        runtime,
        "_prepare_case_from_anchor_keys",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("perturbation must reuse the exact parent PDE")
        ),
    )
    fatal = RuntimeError("fatal qualified perturbation ABI failure")
    calls = []

    def fail(
        _resident,
        *,
        case,
        qualification,
        seed,
        device,
        private_stage_timings,
    ):
        assert private_stage_timings == {}
        calls.append((case.case_id, qualification, seed, device))
        raise fatal

    runtime._symbols.build_qualified_perturbed_observation = fail
    controller_seeds = ((1 << 63) - 2, (1 << 63) - 1)
    for controller_seed in controller_seeds:
        with pytest.raises(RuntimeError) as caught:
            runtime.manufacture_perturbed(
                None,
                parent,
                _request(GenerationKind.PERTURBED),
                seed=controller_seed,
            )
        assert caught.value is fatal

    modulus = runtime_module.GKW_QUALIFIED_PERTURBATION_SEED_MODULUS
    opaque_ids = {
        controller_seed: runtime._opaque_case_prefix(
            kind='pert',
            seed=controller_seed,
            discriminator=parent.hidden_handle.answer_id,
        ) + "-0000"
        for controller_seed in controller_seeds
    }
    assert calls == [
        (
            opaque_ids[controller_seed],
            runtime.production_qualification,
            controller_seed % modulus,
            "cpu",
        )
        for controller_seed in controller_seeds
    ]
    assert calls[0][2] != calls[1][2]
    assert runtime._handles == {parent.case_id: parent.hidden_handle}
    assert runtime._pending_resident_handles == {}
    telemetry = runtime.private_manufacturing_telemetry
    assert telemetry["perturbed_generation_calls"] == len(controller_seeds)
    assert telemetry["perturbed_generation_cases"] == 0
    assert telemetry["perturbed_online_numerical_audits"] == 0
    assert telemetry["perturbed_online_parameter_hashes"] == 0
    assert telemetry["perturbed_failed_seed_replacements"] == 0
