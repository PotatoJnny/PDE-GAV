from __future__ import annotations

import weakref
from dataclasses import dataclass, replace
from pathlib import Path
from threading import RLock
from types import SimpleNamespace

import numpy as np
import pytest

import pde_agent_verifier_solver.plugins.gkw.selected_torch_runtime as runtime_module
from pde_agent_verifier_solver.sdk import (
    Case,
    CaseSource,
    CaseVisibility,
    GenerationKind,
    GenerationPurpose,
    GenerationRequest,
    RecoverableSamplingRejection,
    GeneratorMode,
)


_IMPLEMENTATION_SHA256 = "a" * 64
_CONTINUOUS_SELECTED_GENERATOR_SCHEMA = (
    "fusion_gkw_selected_generator_full_rank_continuous_condition_codes_v3"
)
_PUBLIC_CONDITION = (5.25, 2.75, 0.875, 1.5)


def _public_packet(condition=_PUBLIC_CONDITION):
    return SimpleNamespace(
        scalars=dict(zip(runtime_module._PUBLIC_CONDITION_NAMES, condition))
    )


class _ValidatedQualification:
    certificate = SimpleNamespace(qualification_id="offline-qualified-test-rule")


class _ValidatedDirectCompatibility:
    compatibility_id = "experimental-direct-test"


class _FakeTensor:
    def __init__(self, values):
        self._values = np.asarray(values)

    def detach(self):
        return self

    def to(self, *, device):
        assert device == "cpu"
        return self

    def contiguous(self):
        return self

    def numpy(self):
        return self._values


@dataclass(frozen=True)
class _StateConfig:
    spectral_width: int


class _Contract:
    def __init__(self) -> None:
        self.grid = SimpleNamespace(
            state_shape=(2, 2, 2, 2, 2),
            ns=2,
            nkx=2,
            nky=2,
        )

    def public_metadata(self):
        return {"operator": "fake-selected-gkw", "grid": [2, 2, 2]}


class _State:
    def __init__(self, *, kx_start=3, config_value=7) -> None:
        self.kx_start = kx_start
        self.ky_count = 2
        self.mode_counts = {"kx": 2, "ky": 2}
        self.config = _StateConfig(config_value)
        self.contract = _Contract()
        self.control_values = _FakeTensor([[1.0, 2.0], [3.0, 4.0]])
        self.nodes = {"time": _FakeTensor([0.5, 0.75, 1.0])}

    def private_provenance(self):
        return {"generator": "fake-selected", "sample": 11}

    @staticmethod
    def operator_view(times, *, time_derivative_order=0):
        import torch

        values = torch.as_tensor(times, dtype=torch.float64).reshape(-1, 1)
        if time_derivative_order == 0:
            real = 1.0 + values.square()
        elif time_derivative_order == 1:
            real = 2.0 * values
        else:  # pragma: no cover - the selected runtime asks only for state/dt
            raise ValueError("unsupported fake derivative order")
        return torch.complex(real, torch.zeros_like(real))


class _Sampler:
    device = "cpu"
    config = SimpleNamespace(sampling_mode="local_edit")

    def __init__(self) -> None:
        self.calls = []

    def sample(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        raise AssertionError("artifact mismatch must fail before selected sampling")


class _Decoder:
    config = SimpleNamespace(neural_inference_precision="bfloat16")

    def decode_and_project(self, *args, **kwargs):  # pragma: no cover - guard only
        raise AssertionError("decoder must not run in these boundary tests")


_DECODER_HARD_CHECKS = {
    "finite",
    "decode_log_std_clipping",
    "pathological_case_correction",
    "exact_moment_projection",
}


def _decoder_report(
    *,
    passed=True,
    enforced=True,
    precision="bfloat16",
    decode_order="sample_major_strict",
    distribution_passed=True,
    case_hard_checks=None,
    case_passed=True,
    report_hard_checks=None,
):
    all_passed = {name: True for name in _DECODER_HARD_CHECKS}
    return {
        "schema": "fusion_gkw_selected_decoder_report_v1",
        "sample_count": 1,
        "cases": [
            {
                "sample_index": 0,
                "passed": case_passed,
                "hard_checks": dict(
                    all_passed if case_hard_checks is None else case_hard_checks
                ),
            }
        ],
        "hard_checks": dict(
            all_passed if report_hard_checks is None else report_hard_checks
        ),
        "distribution_calibration": {
            "enforced": enforced,
            "passed": distribution_passed,
        },
        "passed": passed,
        "dense_state_persisted": False,
        "neural_inference_precision": precision,
        "decode_order": decode_order,
    }


class _ReportingDecoder:
    config = SimpleNamespace(neural_inference_precision="bfloat16")

    def __init__(self, report):
        self.report = report
        self.calls = []

    def decode_and_project(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.report


class _CallbackReportingDecoder(_ReportingDecoder):
    def __init__(self, report, callback_audit):
        super().__init__(report)
        self.callback_audit = callback_audit

    def decode_and_project(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        kwargs["projected_sample_audit_callback"](0, self.callback_audit)
        return self.report


class _FakeBridge:
    def __init__(self) -> None:
        self.finalize_calls = 0
        self.frames = []
        self.audits = []

    def consume_projected_frame(self, *args):
        self.frames.append(args)

    def register_decoder_case_audit(self, *args):
        self.audits.append(args)

    def finalize(self):
        self.finalize_calls += 1
        return {"passed": True, "cases": []}



class _RetentionPolicy:
    @staticmethod
    def audit_only():
        return object()


class _StoredSamples:
    anchor_keys = ("fake-anchor",)
    solution_sampling_mode = "local_edit"
    sampled_public_conditions = np.asarray(
        [_PUBLIC_CONDITION], dtype=np.float32
    )

    @classmethod
    def load_private(cls, path, *, storage_root="/scratch"):
        assert Path(path).is_relative_to(Path(storage_root))
        return cls()



def _service():
    sampler = _Sampler()
    return SimpleNamespace(
        sampler=sampler,
        decoder=_Decoder(),
        config=SimpleNamespace(
            private_root="/scratch/fake-selected-service",
            maximum_representation_relative_l2=1.0,
        ),
        calibrated=SimpleNamespace(
            source=object(),
            hidden_support=object(),
            off_grid=object(),
        ),
        archived_cases={},
        time_registry={},
        selected_positions=(0,),
        window_length=1,
    )


def _selected_symbols(limit_calls):
    def prepare_cases(anchor_keys, *, case_id_prefix, **kwargs):
        del kwargs
        assert tuple(anchor_keys) == _StoredSamples.anchor_keys
        return (SimpleNamespace(case_id=case_id_prefix),)

    def oracle_limits(*, max_total_rows):
        limit_calls.append(max_total_rows)
        return {
            "max_rows_per_request": 65_536,
            "max_total_rows": max_total_rows,
            "max_unique_source_times": 256,
            "source_frame_cache_entries": 1,
        }

    def qualification_clean_retry_seed(*, seed, attempt):
        return (seed % 2_147_483_647 + attempt * 1_000_003) % 2_147_483_647

    return runtime_module._SelectedSymbols(
        prepare_cases=prepare_cases,
        streaming_bridge=object(),
        selected_generator_schema=_CONTINUOUS_SELECTED_GENERATOR_SCHEMA,
        retention_policy=object(),
        validated_production_qualification=_ValidatedQualification,
        qualification_clean_maximum_attempts=8,
        qualification_clean_retry_seed=qualification_clean_retry_seed,
        build_macro_truth=object(),
        build_perturbation_macro_truth=object(),
        oracle_limits=oracle_limits,
        oracle_protocol=object(),
        capture_resident_answer=object(),
        build_qualified_perturbed_observation=object(),
        build_oracle_backend_from_resident=object(),
    )


def _runtime(monkeypatch, *, oracle_max_total_rows=20_000_000):
    limit_calls = []
    symbols = _selected_symbols(limit_calls)
    monkeypatch.setattr(runtime_module, "_load_selected_symbols", lambda: symbols)
    monkeypatch.setattr(
        runtime_module.Path,
        "mkdir",
        lambda self, parents=False, exist_ok=False: None,
    )
    service = _service()
    runtime = runtime_module.GKWSelectedTorchRuntime(
        service,
        implementation_sha256=_IMPLEMENTATION_SHA256,
        private_root="/scratch/gkw-selected-runtime-test",
        production_qualification=_ValidatedQualification(),
        oracle_max_total_rows=oracle_max_total_rows,
    )
    return runtime, service, limit_calls


def test_prepare_case_routes_continuous_condition_and_rejects_packet_drift(
    monkeypatch,
):
    runtime, _service_value, _limit_calls = _runtime(monkeypatch)
    sampled = np.asarray([_PUBLIC_CONDITION], dtype=np.float32)
    samples = SimpleNamespace(
        anchor_keys=_StoredSamples.anchor_keys,
        sampled_public_conditions=sampled,
    )
    calls = []

    def prepare(anchor_keys, **kwargs):
        calls.append((tuple(anchor_keys), kwargs))
        return (
            SimpleNamespace(
                case_id=f"{kwargs['case_id_prefix']}-0000",
                public_packet=_public_packet(
                    tuple(float(value) for value in sampled[0])
                ),
            ),
        )

    runtime._symbols = replace(runtime._symbols, prepare_cases=prepare)
    case = runtime._prepare_case(samples, "fresh-condition")
    assert case.case_id == "fresh-condition-0000"
    assert calls[0][0] == _StoredSamples.anchor_keys
    assert calls[0][1]["sampled_public_conditions"] is sampled

    def drifted(anchor_keys, **kwargs):
        del anchor_keys
        condition = list(float(value) for value in sampled[0])
        condition[-1] += 1.0
        return (
            SimpleNamespace(
                case_id=f"{kwargs['case_id_prefix']}-0000",
                public_packet=_public_packet(tuple(condition)),
            ),
        )

    runtime._symbols = replace(runtime._symbols, prepare_cases=drifted)
    with pytest.raises(RuntimeError, match="continuous coefficients"):
        runtime._prepare_case(samples, "drifted-condition")


def test_constructor_preserves_full_twenty_million_oracle_limit(monkeypatch):
    runtime, service, limit_calls = _runtime(monkeypatch)

    assert runtime.oracle_max_total_rows == 20_000_000
    assert limit_calls == [20_000_000]
    assert runtime._oracle_limits["max_total_rows"] == 20_000_000
    assert service.sampler.calls == []
    assert runtime.formal_macro_time_count == 4
    assert runtime.decoder_precision == "bfloat16"
    assert runtime.transient_storage_root == Path("/dev/shm")
    assert runtime._transaction_parent.parent == Path("/dev/shm")

    service.decoder.config = SimpleNamespace(neural_inference_precision="")
    with pytest.raises(TypeError, match="decoder precision"):
        runtime_module.GKWSelectedTorchRuntime(
            service,
            implementation_sha256=_IMPLEMENTATION_SHA256,
            production_qualification=_ValidatedQualification(),
            private_root="/scratch/gkw-selected-runtime-no-precision",
        )
    service.decoder.config = SimpleNamespace(
        neural_inference_precision="bfloat16"
    )

    with pytest.raises(TypeError, match="validated qualification"):
        runtime_module.GKWSelectedTorchRuntime(
            service,
            implementation_sha256=_IMPLEMENTATION_SHA256,
            production_qualification=object(),
            private_root="/scratch/gkw-selected-runtime-wrong-config",
        )

    with pytest.raises(ValueError, match=r"\[65536, 20000000\]"):
        runtime_module.GKWSelectedTorchRuntime(
            service,
            implementation_sha256=_IMPLEMENTATION_SHA256,
            private_root="/scratch/gkw-selected-runtime-too-large",
            production_qualification=_ValidatedQualification(),
            oracle_max_total_rows=20_000_001,
        )


def test_direct_noise_constructor_uses_only_experimental_authority(monkeypatch):
    limit_calls = []
    symbols = replace(
        _selected_symbols(limit_calls),
        validated_direct_noise_compatibility=_ValidatedDirectCompatibility,
        build_direct_noise_perturbed_observation=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(runtime_module, "_load_selected_symbols", lambda: symbols)
    monkeypatch.setattr(
        runtime_module.Path,
        "mkdir",
        lambda self, parents=False, exist_ok=False: None,
    )
    service = _service()
    service.sampler.config = SimpleNamespace(sampling_mode="direct_noise")
    compatibility = _ValidatedDirectCompatibility()
    runtime = runtime_module.GKWSelectedTorchRuntime(
        service,
        implementation_sha256=_IMPLEMENTATION_SHA256,
        experimental_direct_noise_compatibility=compatibility,
        expected_solution_sampling_mode="direct_noise",
        enforce_distribution_calibration=False,
        private_root="/scratch/gkw-selected-runtime-direct-noise",
    )

    assert runtime.production_qualification is None
    assert runtime.experimental_direct_noise_compatibility is compatibility
    assert runtime.expected_solution_sampling_mode == "direct_noise"
    assert runtime.enforce_distribution_calibration is False
    assert limit_calls == [20_000_000]

    direct_request = GenerationRequest(
        round_index=0,
        count=1,
        mode=GeneratorMode.OFFLINE,
        kind=GenerationKind.BASE,
        visibility=CaseVisibility.DEVELOPMENT,
        settings={
            "fixed_generator_sha256": _IMPLEMENTATION_SHA256,
            "generator_sampling_mode": "direct_noise",
        },
    )
    runtime._validate_request_artifact(direct_request)
    with pytest.raises(ValueError, match="request sampling mode"):
        runtime._validate_request_artifact(
            replace(
                direct_request,
                settings={
                    "fixed_generator_sha256": _IMPLEMENTATION_SHA256,
                    "generator_sampling_mode": "local_edit",
                },
            )
        )

    with pytest.raises(TypeError, match="not production qualification"):
        runtime_module.GKWSelectedTorchRuntime(
            service,
            implementation_sha256=_IMPLEMENTATION_SHA256,
            production_qualification=_ValidatedQualification(),
            experimental_direct_noise_compatibility=compatibility,
            expected_solution_sampling_mode="direct_noise",
            enforce_distribution_calibration=False,
            private_root="/scratch/gkw-selected-runtime-mixed-authority",
        )

    service.sampler.config = SimpleNamespace(sampling_mode="local_edit")
    with pytest.raises(TypeError, match="rejects direct-noise authority"):
        runtime_module.GKWSelectedTorchRuntime(
            service,
            implementation_sha256=_IMPLEMENTATION_SHA256,
            production_qualification=_ValidatedQualification(),
            experimental_direct_noise_compatibility=compatibility,
            private_root="/scratch/gkw-selected-runtime-local-mixed",
        )


def test_clean_sampling_mode_mismatch_fails_before_decode(monkeypatch):
    runtime, service, _limit_calls = _runtime(monkeypatch)
    service.sampler.sample = lambda *_args, **_kwargs: SimpleNamespace(
        solution_sampling_mode="direct_noise"
    )
    monkeypatch.setattr(
        runtime,
        "_manufacture_clean_attempt",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("mode mismatch must fail before decode")
        ),
    )
    request = GenerationRequest(
        round_index=0,
        count=1,
        mode=GeneratorMode.OFFLINE,
        kind=GenerationKind.BASE,
        visibility=CaseVisibility.DEVELOPMENT,
        settings={"cohort": "development"},
    )
    with pytest.raises(RuntimeError, match="expected 'local_edit'"):
        runtime._manufacture_clean_case(request, seed=17, index=0)


def test_constructor_rejects_non_tmpfs_transient_storage(monkeypatch):
    _runtime_value, service, _limit_calls = _runtime(monkeypatch)

    with pytest.raises(ValueError, match="below /dev/shm"):
        runtime_module.GKWSelectedTorchRuntime(
            service,
            implementation_sha256=_IMPLEMENTATION_SHA256,
            private_root="/scratch/gkw-selected-runtime-bad-transient-root",
            production_qualification=_ValidatedQualification(),
            transient_storage_root="/tmp",
        )


def test_retained_tensor_graph_rejects_device_and_storage_aliases():
    import torch

    runtime_module._assert_cpu_tensor_graph(
        torch.empty(0, dtype=torch.float64), label="empty CPU tensor"
    )
    with pytest.raises(RuntimeError, match="non-CPU tensor"):
        runtime_module._assert_cpu_tensor_graph(
            torch.empty(1, device="meta"), label="meta tensor"
        )
    with pytest.raises(RuntimeError, match="non-owning tensor"):
        runtime_module._assert_cpu_tensor_graph(
            torch.ones(2, requires_grad=True), label="autograd tensor"
        )
    with pytest.raises(RuntimeError, match="non-owning tensor"):
        runtime_module._assert_cpu_tensor_graph(
            torch.ones((2, 2)).transpose(0, 1), label="strided tensor"
        )

    backing = torch.arange(8, dtype=torch.float64)
    oversized_view = backing[:2]
    with pytest.raises(RuntimeError, match="oversized tensor view"):
        runtime_module._assert_cpu_tensor_graph(
            oversized_view, label="oversized view"
        )
    owned = runtime_module._owned_cpu_tensor(
        oversized_view, label="copied oversized view"
    )
    assert owned.untyped_storage().nbytes() == owned.numel() * owned.element_size()
    runtime_module._assert_cpu_tensor_graph(owned, label="owned CPU snapshot")


def test_retry_seed_delegates_to_physical_qualification_schedule():
    calls = []

    def physical_retry_seed(*, seed, attempt):
        calls.append((seed, attempt))
        return 1_234_567

    assert runtime_module._retry_seed(
        (1 << 62) + 17,
        3,
        physical_retry_seed=physical_retry_seed,
    ) == 1_234_567
    assert calls == [((1 << 62) + 17, 3)]


@pytest.mark.parametrize("maximum_sampling_attempts", (1, 7, 9))
def test_qualified_constructor_fails_closed_on_retry_budget_mismatch(
    monkeypatch,
    maximum_sampling_attempts,
):
    _runtime_value, service, _limit_calls = _runtime(monkeypatch)

    with pytest.raises(
        ValueError,
        match="maximum_sampling_attempts=8",
    ):
        runtime_module.GKWSelectedTorchRuntime(
            service,
            implementation_sha256=_IMPLEMENTATION_SHA256,
            production_qualification=_ValidatedQualification(),
            private_root="/scratch/gkw-selected-runtime-retry-budget",
            maximum_sampling_attempts=maximum_sampling_attempts,
        )


def test_constructor_keeps_private_root_below_storage_boundary(monkeypatch):
    _unused, service, _limits = _runtime(monkeypatch)
    runtime = runtime_module.GKWSelectedTorchRuntime(
        service,
        implementation_sha256=_IMPLEMENTATION_SHA256,
        production_qualification=_ValidatedQualification(),
        private_root="/project/run/private",
        private_storage_root="/project/run",
    )
    assert runtime.private_root == Path("/project/run/private")
    assert runtime.private_storage_root == Path("/project/run")

    with pytest.raises(ValueError, match="below its storage root"):
        runtime_module.GKWSelectedTorchRuntime(
            service,
            implementation_sha256=_IMPLEMENTATION_SHA256,
            production_qualification=_ValidatedQualification(),
            private_root="/tmp/outside",
            private_storage_root="/project/run",
        )


@pytest.mark.parametrize(
    "name",
    (
        "macro_time_count",
        "macro_field_rows",
        "formal_macro_time_count",
        "maximum_sampling_attempts",
    ),
)
def test_constructor_rejects_non_integer_counts(monkeypatch, name):
    _runtime_value, service, _limit_calls = _runtime(monkeypatch)
    counts = {
        "macro_time_count": 3,
        "macro_field_rows": 1,
        "formal_macro_time_count": 4,
        "maximum_sampling_attempts": 8,
    }
    counts[name] = 3.0

    with pytest.raises(ValueError, match="positive integers"):
        runtime_module.GKWSelectedTorchRuntime(
            service,
            implementation_sha256=_IMPLEMENTATION_SHA256,
            production_qualification=_ValidatedQualification(),
            private_root="/scratch/gkw-selected-runtime-noninteger-count",
            **counts,
        )


@pytest.mark.parametrize(
    ("purpose", "expected_profile"),
    (
        (GenerationPurpose.DEVELOPMENT, runtime_module._DEVELOPMENT_PANEL_PROFILE),
        (
            GenerationPurpose.DEVELOPMENT_SCREEN,
            runtime_module._DEVELOPMENT_PANEL_PROFILE,
        ),
        (GenerationPurpose.SELECTION, runtime_module._FORMAL_PANEL_PROFILE),
        (GenerationPurpose.PROMOTION, runtime_module._FORMAL_PANEL_PROFILE),
        (GenerationPurpose.TERMINAL, runtime_module._FORMAL_PANEL_PROFILE),
        (
            GenerationPurpose.GUIDANCE_FEEDBACK,
            runtime_module._FORMAL_PANEL_PROFILE,
        ),
    ),
)
def test_request_panel_profile_is_driven_only_by_typed_purpose(
    purpose, expected_profile
):
    request = SimpleNamespace(
        purpose=purpose,
        settings={"cohort": "wrong-static-cohort"},
    )
    assert runtime_module._request_panel_profile(request) == (
        purpose.value,
        expected_profile,
    )


@pytest.mark.parametrize("purpose", (None, "", "released_training", 7))
def test_request_panel_profile_rejects_missing_or_unknown_purpose(purpose):
    request = SimpleNamespace(purpose=purpose, settings={})
    with pytest.raises(ValueError, match="request purpose"):
        runtime_module._request_panel_profile(request)


def test_development_and_formal_macro_panels_match_their_declared_strength():
    time_nodes = (0.5, 0.75, 1.0)
    grid = SimpleNamespace(ns=2, nkx=3, nky=4)

    development_times, development_indices = runtime_module._macro_panel(
        time_nodes,
        grid,
        count=5,
        time_count=6,
        seed=17,
        profile=runtime_module._DEVELOPMENT_PANEL_PROFILE,
    )
    expected_flat = np.random.default_rng(17).choice(
        24, size=5, replace=False
    )
    expected_development_indices = np.column_stack(
        np.unravel_index(expected_flat, (2, 3, 4))
    )
    assert np.array_equal(
        development_times,
        np.linspace(0.5, 1.0, num=6, dtype=np.float64),
    )
    assert np.array_equal(
        development_indices, expected_development_indices
    )

    formal_times, formal_indices = runtime_module._macro_panel(
        time_nodes,
        grid,
        count=1,
        time_count=4,
        seed=99,
        profile=runtime_module._FORMAL_PANEL_PROFILE,
    )
    expected_formal_times = np.linspace(
        0.5, 1.0, num=5, dtype=np.float64
    )[1:]
    expected_formal_indices = np.indices(
        (2, 3, 4), dtype=np.int64
    ).reshape(3, -1).T
    assert np.array_equal(formal_times, expected_formal_times)
    assert formal_times[0] > time_nodes[0]
    assert np.array_equal(formal_indices, expected_formal_indices)

    development_audit = runtime_module._macro_panel_audit(
        cohort="development_screen",
        profile=runtime_module._DEVELOPMENT_PANEL_PROFILE,
        times=development_times,
        field_indices=development_indices,
        time_nodes=time_nodes,
        grid=grid,
        development_time_count=6,
        development_field_rows=5,
        formal_time_count=4,
        inherited_from_clean_parent=False,
    )
    assert development_audit["includes_initial_time"] is True
    assert development_audit["full_field_coverage"] is False
    assert development_audit["field_row_count"] == 5

    formal_audit = runtime_module._macro_panel_audit(
        cohort="terminal",
        profile=runtime_module._FORMAL_PANEL_PROFILE,
        times=formal_times,
        field_indices=formal_indices,
        time_nodes=time_nodes,
        grid=grid,
        development_time_count=6,
        development_field_rows=5,
        formal_time_count=4,
        inherited_from_clean_parent=False,
    )
    assert formal_audit["includes_initial_time"] is False
    assert formal_audit["full_field_coverage"] is True
    assert formal_audit["time_count"] == 4
    assert formal_audit["field_row_count"] == 24


def test_sampled_parent_panel_cannot_be_relabelled_as_formal():
    time_nodes = (0.5, 0.75, 1.0)
    grid = SimpleNamespace(ns=2, nkx=3, nky=4)
    times, indices = runtime_module._macro_panel(
        time_nodes,
        grid,
        count=5,
        time_count=6,
        seed=17,
        profile=runtime_module._DEVELOPMENT_PANEL_PROFILE,
    )

    with pytest.raises(ValueError, match="cover every field row and exclude t0"):
        runtime_module._macro_panel_audit(
            cohort="selection",
            profile=runtime_module._FORMAL_PANEL_PROFILE,
            times=times,
            field_indices=indices,
            time_nodes=time_nodes,
            grid=grid,
            development_time_count=6,
            development_field_rows=5,
            formal_time_count=4,
            inherited_from_clean_parent=True,
        )


def test_from_observation_emits_snapshot_safe_float64_scalars(monkeypatch):
    from pde_agent_verifier_solver.plugins.gkw.isolation import (
        snapshot_gkw_case,
    )
    import torch

    runtime, _service_value, _limit_calls = _runtime(monkeypatch)

    def build_macro_truth(state, operator, arrays):
        del operator, arrays
        return SimpleNamespace(
            state=state,
            control_fields={
                name: torch.zeros(
                    (1, 2, 2, 2), dtype=torch.complex128
                )
                for name in runtime_module.GKW_MACRO_FIELDS
            },
            construction_audit={"schema": "test-clean-macro-truth"},
        )

    def capture_resident(observation, parent=None, solution_sampling_mode=None):
        del parent, solution_sampling_mode
        return SimpleNamespace(
            state=observation.state,
            base_state=observation.state,
            qualification_provenance={"passed": True},
        )

    runtime._symbols = replace(
        runtime._symbols,
        oracle_protocol=lambda **kwargs: {"limits": kwargs["limits"]},
        build_macro_truth=build_macro_truth,
        capture_resident_answer=capture_resident,
    )
    state = _State()
    manufactured_times = torch.tensor([0.5, 0.75], dtype=torch.float64)

    observation = SimpleNamespace(
        case=SimpleNamespace(
            case_id="gkw-scalar-contract",
            public_packet=SimpleNamespace(
                arrays={
                    "geometry": np.asarray([1.0, 2.0], dtype=np.float64)
                },
                scalars=dict(
                    zip(
                        runtime_module._PUBLIC_CONDITION_NAMES,
                        _PUBLIC_CONDITION,
                    )
                ),
            ),
            time_nodes=(0.5, 2.0 / 3.0, 5.0 / 6.0, 1.0),
        ),
        state=state,
        operator=object(),
        manufactured=SimpleNamespace(
            times=manufactured_times,
            delta_f_i=state.operator_view(manufactured_times),
            time_derivative=state.operator_view(
                manufactured_times, time_derivative_order=1
            ),
            equation_relative_residual=lambda: 0.0,
        ),
        passed=True,
    )
    manufactured = runtime._from_observation(
        observation,
        case_prefix="gkw-scalar-contract",
        expected_case_id="gkw-scalar-contract",
        anchor_keys=({"trajectory": 0, "start_position": 1, "end_position": 2},),
        sampled_public_condition=_PUBLIC_CONDITION,
        seed=7,
        perturbed=False,
        parent_handle=None,
        panel=None,
        cohort="development",
        panel_profile=runtime_module._DEVELOPMENT_PANEL_PROFILE,
    )
    scalar = manufactured.public_data["scalars"]["q"]
    assert isinstance(scalar, np.float64)
    assert np.asarray(scalar).shape == ()
    assert "macro_query" not in manufactured.public_data
    assert isinstance(manufactured.truth, runtime_module._CompletedMacroTruth)
    assert (
        manufactured.truth.evaluator
        is manufactured.hidden_handle.completed_macro_evaluator

    )
    snapshot = snapshot_gkw_case(
        Case(
            case_id=manufactured.case_id,
            source=CaseSource.GENERATED,
            visibility=CaseVisibility.DEVELOPMENT,
            public_data=manufactured.public_data,
        )
    )
    frozen_scalar = snapshot.public_data["scalars"]["q"]
    assert isinstance(frozen_scalar, np.float64)
    assert np.asarray(frozen_scalar).shape == ()


def _answer_first_observation():
    import torch

    state = _State()
    times = torch.tensor([0.5, 0.75], dtype=torch.float64)
    return SimpleNamespace(
        case=SimpleNamespace(
            case_id="gkw-answer-first-binding",
            public_packet=SimpleNamespace(
                arrays={"geometry": np.asarray([1.0, 2.0], dtype=np.float64)},
                scalars=dict(
                    zip(
                        runtime_module._PUBLIC_CONDITION_NAMES,
                        _PUBLIC_CONDITION,
                    )
                ),
            ),
            time_nodes=(0.5, 0.75, 1.0),
        ),
        state=state,
        operator=object(),
        manufactured=SimpleNamespace(
            times=times,
            delta_f_i=state.operator_view(times),
            time_derivative=state.operator_view(
                times, time_derivative_order=1
            ),
            equation_relative_residual=lambda: 0.0,
        ),
        passed=True,
    )


def test_from_observation_rejects_macro_truth_bound_to_another_state(monkeypatch):
    runtime, _service_value, _limit_calls = _runtime(monkeypatch)

    def capture_resident(observation, parent=None, solution_sampling_mode=None):
        del parent, solution_sampling_mode
        return SimpleNamespace(
            state=observation.state,
            base_state=observation.state,
            qualification_provenance={"passed": True},
        )

    runtime._symbols = replace(
        runtime._symbols,
        oracle_protocol=lambda **kwargs: {"limits": kwargs["limits"]},
        capture_resident_answer=capture_resident,
        build_macro_truth=lambda *args: SimpleNamespace(state=object()),
    )

    with pytest.raises(RuntimeError, match="macro truth was not derived"):
        runtime._from_observation(
            _answer_first_observation(),
            case_prefix="gkw-answer-first-binding",
            expected_case_id="gkw-answer-first-binding",
            anchor_keys=({"trajectory": 0, "start_position": 1, "end_position": 2},),
            sampled_public_condition=_PUBLIC_CONDITION,
            seed=9,
            perturbed=False,
            parent_handle=None,
            panel=None,
            cohort="development",
            panel_profile=runtime_module._DEVELOPMENT_PANEL_PROFILE,
        )


def test_clean_generation_retries_only_typed_decoder_gate_rejections(monkeypatch):
    runtime, service, _limit_calls = _runtime(monkeypatch)
    runtime.maximum_sampling_attempts = 3

    class Sampler:
        device = "cpu"

        def __init__(self):
            self.calls = []

        def sample(self, count, **kwargs):
            assert count == 1
            self.calls.append(kwargs)
            return SimpleNamespace(solution_sampling_mode="local_edit")

    sampler = Sampler()
    service.sampler = sampler
    request = GenerationRequest(
        round_index=3,
        count=1,
        mode=GeneratorMode.OFFLINE,
        kind=GenerationKind.BASE,
        visibility=CaseVisibility.DEVELOPMENT,
        settings={"cohort": "development"},
    )
    attempts = []
    accepted = object()

    def reject_once(**kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            raise runtime_module._DecoderGateRejection("retry sample")
        return accepted

    monkeypatch.setattr(runtime, "_manufacture_clean_attempt", reject_once)
    assert runtime._manufacture_clean_case(request, seed=17, index=0) is accepted
    assert [row["seed"] for row in sampler.calls] == [
        runtime_module._retry_seed(
            17,
            0,
            physical_retry_seed=runtime._symbols.qualification_clean_retry_seed,
        ),
        runtime_module._retry_seed(
            17,
            1,
            physical_retry_seed=runtime._symbols.qualification_clean_retry_seed,
        ),
    ]
    assert attempts[0]["prefix"].startswith("gkw-u-base-")
    assert attempts[1]["prefix"].startswith("gkw-u-base-")
    assert attempts[0]["prefix"] != attempts[1]["prefix"]
    assert f"{17:016x}" not in attempts[0]["prefix"]
    assert {
        (row["cohort"], row["panel_profile"]) for row in attempts
    } == {
        ("development", runtime_module._DEVELOPMENT_PANEL_PROFILE)
    }

    sampler.calls.clear()

    def fatal(**kwargs):
        del kwargs
        raise ValueError("fatal decoder ABI failure")

    monkeypatch.setattr(runtime, "_manufacture_clean_attempt", fatal)
    with pytest.raises(ValueError, match="fatal decoder ABI failure"):
        runtime._manufacture_clean_case(request, seed=23, index=0)
    assert len(sampler.calls) == 1


def test_clean_generation_chains_last_gate_rejection_when_retries_exhaust(
    monkeypatch,
):
    runtime, service, _limit_calls = _runtime(monkeypatch)
    runtime.maximum_sampling_attempts = 2

    class Sampler:
        device = "cpu"

        def __init__(self):
            self.calls = []

        def sample(self, count, **kwargs):
            assert count == 1
            self.calls.append(kwargs)
            return SimpleNamespace(solution_sampling_mode="local_edit")

    sampler = Sampler()
    service.sampler = sampler
    rejection = runtime_module._DecoderGateRejection("retry sample")
    monkeypatch.setattr(
        runtime,
        "_manufacture_clean_attempt",
        lambda **kwargs: (_ for _ in ()).throw(rejection),
    )
    request = GenerationRequest(
        round_index=3,
        count=1,
        mode=GeneratorMode.OFFLINE,
        kind=GenerationKind.BASE,
        visibility=CaseVisibility.DEVELOPMENT,
        settings={"cohort": "development"},
    )

    with pytest.raises(
        RecoverableSamplingRejection,
        match="exhausted physical retries",
    ) as exc_info:
        runtime._manufacture_clean_case(request, seed=29, index=0)
    assert exc_info.value.__cause__ is rejection
    assert exc_info.value.audit == {
        "reason": "gkw_clean_admissibility_exhausted",
        "attempt_count": 2,
    }
    assert len(sampler.calls) == 2

    sampler.calls.clear()
    attempt_count = 0

    def reject_then_return_none(**kwargs):
        nonlocal attempt_count
        del kwargs
        attempt_count += 1
        if attempt_count == 1:
            raise runtime_module._DecoderGateRejection("earlier rejection")
        return None

    monkeypatch.setattr(
        runtime, "_manufacture_clean_attempt", reject_then_return_none
    )
    with pytest.raises(RecoverableSamplingRejection, match="exhausted physical retries") as stale:
        runtime._manufacture_clean_case(request, seed=31, index=0)
    assert stale.value.__cause__ is None
    assert len(sampler.calls) == 2


def test_fixed_artifact_mismatch_fails_before_selected_sampling(monkeypatch):
    runtime, service, _limit_calls = _runtime(monkeypatch)
    request = GenerationRequest(
        round_index=0,
        count=1,
        mode=GeneratorMode.OFFLINE,
        kind=GenerationKind.BASE,
        visibility=CaseVisibility.DEVELOPMENT,
        settings={"fixed_generator_sha256": "b" * 64},
    )

    with pytest.raises(ValueError, match="artifact does not match"):
        runtime.manufacture_clean(None, request, (101,))

    assert service.sampler.calls == []


class _Oracle:
    def __init__(self) -> None:
        self.calls = []
        self.close_calls = 0

    def query_source(self, times, phase_indices):
        self.calls.append("source")
        return np.full(
            (len(np.asarray(phase_indices)), 2), 1.0, dtype=np.float32
        )

    def query_initial_h(self, phase_indices):
        self.calls.append("initial_h")
        return np.full(
            (len(np.asarray(phase_indices)), 2), 2.0, dtype=np.float32
        )

    def query_initial_phi(self, field_indices):
        self.calls.append("initial_phi")
        return np.full(
            (len(np.asarray(field_indices)), 2), 3.0, dtype=np.float32
        )

    def close(self):
        self.close_calls += 1


class _OracleRuntime:
    def __init__(self) -> None:
        self.oracle = _Oracle()
        self.open_calls = 0
        self._oracle_backend_lock = RLock()

    def _open_oracle(self, case):
        self.open_calls += 1
        return self.oracle


def _session_case(
    *,
    max_rows=2,
    max_total=3,
    max_times=2,
    time_interval=(0.0, 1.0),
):
    return Case(
        case_id="gkw-session-test",
        source=CaseSource.GENERATED,
        visibility=CaseVisibility.DEVELOPMENT,
        public_data={
            "arrays": {},
            "scalars": {},
            "metadata": {},
            "oracle_protocol": {
                "time_interval_closed": list(time_interval),
                "phase_grid_shape": [1, 1, 1, 1, 1],
                "field_grid_shape": [1, 1, 1],
                "limits": {
                    "max_rows_per_request": max_rows,
                    "max_total_rows": max_total,
                    "max_unique_source_times": max_times,
                }
            },
        },
    )


def test_generation_rejects_failed_decoder_report_before_bridge_finalize(monkeypatch):
    runtime, _service_value, _limit_calls = _runtime(monkeypatch)
    bridge = _FakeBridge()
    decoder = _ReportingDecoder(_decoder_report(
        passed=False, distribution_passed=False
    ))
    runtime.service.decoder = decoder
    runtime._symbols = replace(
        runtime._symbols,
        streaming_bridge=lambda *args, **kwargs: bridge,
        retention_policy=_RetentionPolicy,
    )

    with pytest.raises(runtime_module._DecoderGateRejection):
        runtime._run_bridge(
            object(),
            SimpleNamespace(public_packet=_public_packet()),
            run_root=Path("/scratch/gkw-decoder-generation-test"),
            callback=lambda observation: {},
        )

    assert bridge.finalize_calls == 0
    assert decoder.calls[0][1]["enforce_distribution_calibration"] is True


@pytest.mark.parametrize("decoder_passes", (True, False))
def test_clean_terminal_frame_waits_for_decoder_report(
    monkeypatch,
    decoder_passes,
):
    import torch

    runtime, service, _limit_calls = _runtime(monkeypatch)
    service.window_length = 4
    bridge = _FakeBridge()
    report = _decoder_report(
        passed=decoder_passes,
        distribution_passed=decoder_passes,
    )
    bridge_case = SimpleNamespace(
        contract=_Contract(),
        public_packet=_public_packet(),
    )
    frame = torch.zeros(
        (2, *bridge_case.contract.grid.state_shape),
        dtype=torch.float32,
    )

    class FrameDecoder:
        config = SimpleNamespace(neural_inference_precision="bfloat16")

        def decode_and_project(self, *args, **kwargs):
            del args
            assert kwargs["public_equation_packets"] == (
                bridge_case.public_packet,
            )
            kwargs["projected_sample_audit_callback"](0, report["cases"][0])
            for frame_index in range(4):
                kwargs["projected_frame_callback"](
                    0,
                    frame_index,
                    frame + frame_index,
                )
            assert [row[1] for row in bridge.frames] == [0, 1, 2]
            return report

    service.decoder = FrameDecoder()
    runtime._symbols = replace(
        runtime._symbols,
        streaming_bridge=lambda *args, **kwargs: bridge,
        retention_policy=_RetentionPolicy,
    )

    if decoder_passes:
        assert runtime._run_bridge(
            object(),
            bridge_case,
            run_root=Path("/scratch/gkw-decoder-deferred-test"),
            callback=lambda observation: None,
        ) == {"passed": True, "cases": []}
    else:
        with pytest.raises(runtime_module._DecoderGateRejection):
            runtime._run_bridge(
                object(),
                bridge_case,
                run_root=Path("/scratch/gkw-decoder-deferred-test"),
                callback=lambda observation: None,
            )

    assert len(bridge.audits) == 1
    assert [row[1] for row in bridge.frames] == (
        [0, 1, 2, 3] if decoder_passes else [0, 1, 2]
    )
    assert bridge.finalize_calls == int(decoder_passes)
    telemetry = runtime.private_manufacturing_telemetry
    assert telemetry[
        "clean_decoder_rejections_before_physical_audit"
    ] == int(not decoder_passes)


def test_singleton_decoder_hard_check_disagreement_is_fatal_contract_error():
    case_checks = {name: True for name in _DECODER_HARD_CHECKS}
    case_checks["finite"] = False
    report = _decoder_report(
        case_passed=False,
        case_hard_checks=case_checks,
    )

    with pytest.raises(
        RuntimeError, match="distribution or physical gates"
    ) as exc_info:
        runtime_module._require_decoder_report(
            report,
            enforce_distribution_calibration=True,
            neural_inference_precision="bfloat16",
        )
    assert not isinstance(exc_info.value, runtime_module._DecoderGateRejection)



def test_generation_rejects_incomplete_decoder_hard_checks_before_finalize(
    monkeypatch,
):
    runtime, _service_value, _limit_calls = _runtime(monkeypatch)
    bridge = _FakeBridge()
    decoder = _ReportingDecoder(
        _decoder_report(case_hard_checks={"finite": True})
    )
    runtime.service.decoder = decoder
    runtime._symbols = replace(
        runtime._symbols,
        streaming_bridge=lambda *args, **kwargs: bridge,
        retention_policy=_RetentionPolicy,
    )

    with pytest.raises(
        RuntimeError, match="distribution or physical gates"
    ) as exc_info:
        runtime._run_bridge(
            object(),
            SimpleNamespace(public_packet=_public_packet()),
            run_root=Path("/scratch/gkw-decoder-hard-check-test"),
            callback=lambda observation: {},
        )

    assert not isinstance(exc_info.value, runtime_module._DecoderGateRejection)
    assert bridge.finalize_calls == 0


def test_resident_public_session_rolls_over_legacy_counters_without_quotas():
    runtime = _OracleRuntime()
    session = runtime_module._ResidentPublicSession(runtime, _session_case())

    source = session.query_source(0.5, np.zeros((1, 5), dtype=np.int64))
    initial_h = session.query_initial_h(np.zeros((1, 5), dtype=np.int64))
    initial_phi = session.query_initial_phi(
        np.zeros((1, 3), dtype=np.int64)
    )
    for value, expected in (
        (source, 1.0),
        (initial_h, 2.0),
        (initial_phi, 3.0),
    ):
        np.testing.assert_array_equal(value, np.full((1, 2), expected))
        assert value.dtype == np.dtype("float64")
        assert not value.flags.writeable
    assert runtime.open_calls == 1
    assert runtime.oracle.calls == ["source", "initial_h", "initial_phi"]

    repeated = session.query_initial_h(np.zeros((1, 5), dtype=np.int64))
    np.testing.assert_array_equal(repeated, np.full((1, 2), 2.0))
    assert runtime.open_calls == 2
    assert runtime.oracle.calls == [
        "source", "initial_h", "initial_phi", "initial_h"
    ]
    assert session.usage_snapshot["query_calls"] == 4
    assert session.usage_snapshot["queried_rows"] == 4
    assert session.usage_snapshot["backend_wall_seconds"] >= 0.0
    assert session.usage_snapshot["underlying_session_rollovers"] == 1

    session.close()
    session.close()
    assert runtime.oracle.close_calls == 2
    assert session._oracle is None
    with pytest.raises(RuntimeError, match="session is closed"):
        session.query_initial_h(np.zeros((1, 5), dtype=np.int64))
    assert runtime.open_calls == 2
    assert session._total_rows == 4

    time_runtime = _OracleRuntime()
    time_session = runtime_module._ResidentPublicSession(
        time_runtime,
        _session_case(max_total=10, max_times=1),
    )
    time_session.query_source(0.5, np.zeros((1, 5), dtype=np.int64))
    time_session.query_source(0.75, np.zeros((1, 5), dtype=np.int64))
    assert time_runtime.open_calls == 2
    assert time_runtime.oracle.calls == ["source", "source"]
    assert time_session._total_rows == 2
    assert time_session.usage_snapshot["underlying_session_rollovers"] == 1


def test_resident_public_session_normalizes_complex_rows_to_read_only_pairs():
    runtime = _OracleRuntime()
    runtime.oracle.query_initial_h = lambda _indices: np.asarray(
        [1.25 + 2.5j], dtype=np.complex64
    )
    session = runtime_module._ResidentPublicSession(
        runtime, _session_case(max_total=10)
    )

    result = session.query_initial_h(np.zeros((1, 5), dtype=np.int64))

    np.testing.assert_array_equal(result, np.asarray([[1.25, 2.5]]))
    assert result.dtype == np.dtype("float64")
    assert not result.flags.writeable


@pytest.mark.parametrize(
    ("value", "error", "message"),
    (
        (np.zeros((1,), dtype=np.float64), RuntimeError, "shape"),
        (
            np.asarray([[np.nan, 0.0]], dtype=np.float64),
            ValueError,
            "finite",
        ),
        (
            np.asarray([["real", "imag"]]),
            TypeError,
            "numeric",
        ),
    ),
)
def test_resident_public_session_rejects_malformed_oracle_output(
    value,
    error,
    message,
):
    runtime = _OracleRuntime()
    runtime.oracle.query_initial_h = lambda _indices: value
    session = runtime_module._ResidentPublicSession(
        runtime, _session_case(max_total=10)
    )

    with pytest.raises(error, match=message):
        session.query_initial_h(np.zeros((1, 5), dtype=np.int64))


def test_resident_public_session_close_releases_oracle_reference():
    class EphemeralOracleRuntime:
        def __init__(self):
            self.open_calls = 0
            self.oracle_ref = None
            self._oracle_backend_lock = RLock()

        def _open_oracle(self, case):
            del case
            self.open_calls += 1
            oracle = _Oracle()
            self.oracle_ref = weakref.ref(oracle)
            return oracle

    runtime = EphemeralOracleRuntime()
    session = runtime_module._ResidentPublicSession(
        runtime, _session_case(max_total=10)
    )
    session.query_initial_h(np.zeros((1, 5), dtype=np.int64))
    oracle_ref = runtime.oracle_ref
    assert oracle_ref is not None
    assert oracle_ref() is not None

    session.close()
    assert session._oracle is None
    assert oracle_ref() is None
    with pytest.raises(RuntimeError, match="session is closed"):
        session.query_initial_h(np.zeros((1, 5), dtype=np.int64))
    assert runtime.open_calls == 1


def test_resident_public_session_close_cleans_up_when_oracle_close_fails():
    class FailingCloseOracle(_Oracle):
        def close(self):
            super().close()
            raise RuntimeError("oracle close failed")

    runtime = _OracleRuntime()
    runtime.oracle = FailingCloseOracle()
    session = runtime_module._ResidentPublicSession(
        runtime, _session_case(max_total=10)
    )
    session.query_initial_h(np.zeros((1, 5), dtype=np.int64))

    with pytest.raises(RuntimeError, match="oracle close failed"):
        session.close()
    assert session._closed
    assert session._oracle is None
    assert runtime.oracle.close_calls == 1

    session.close()
    assert runtime.oracle.close_calls == 1
    with pytest.raises(RuntimeError, match="session is closed"):
        session.query_initial_h(np.zeros((1, 5), dtype=np.int64))
    assert runtime.open_calls == 1


def test_resident_public_session_rejects_shapes_before_opening_or_consuming_budget():
    runtime = _OracleRuntime()
    session = runtime_module._ResidentPublicSession(
        runtime,
        _session_case(max_total=10),
    )

    with pytest.raises(ValueError, match=r"shape \[N,5\]"):
        session.query_initial_h(np.zeros((1, 4), dtype=np.int64))
    with pytest.raises(ValueError, match=r"shape \[N,3\]"):
        session.query_initial_phi(np.zeros((1, 2), dtype=np.int64))
    with pytest.raises(ValueError, match=r"scalar or shape \[N\]"):
        session.query_source(
            np.asarray([0.5]),
            np.zeros((2, 5), dtype=np.int64),
        )
    with pytest.raises(ValueError, match="must be finite"):
        session.query_source(np.nan, np.zeros((1, 5), dtype=np.int64))
    with pytest.raises(ValueError, match="row count exceeds"):
        session.query_source(0.5, np.zeros((0, 5), dtype=np.int64))

    assert runtime.open_calls == 0
    assert runtime.oracle.calls == []
    assert session._total_rows == 0

    session.query_initial_h(np.zeros((1, 5), dtype=np.int64))
    assert runtime.open_calls == 1
    assert session._total_rows == 1


def test_resident_public_session_rejects_queries_outside_public_domain():
    runtime = _OracleRuntime()
    session = runtime_module._ResidentPublicSession(
        runtime,
        _session_case(max_total=10),
    )

    with pytest.raises(ValueError, match="leave the public domain"):
        session.query_source(1.01, np.zeros((1, 5), dtype=np.int64))
    with pytest.raises(ValueError, match="integer shape"):
        session.query_initial_h(np.zeros((1, 5), dtype=np.float64))
    with pytest.raises(ValueError, match="leave the public grid"):
        session.query_initial_h(np.ones((1, 5), dtype=np.int64))
    with pytest.raises(ValueError, match="leave the public grid"):
        session.query_initial_phi(np.ones((1, 3), dtype=np.int64))

    assert runtime.open_calls == 0
    assert runtime.oracle.calls == []
    assert session._total_rows == 0


def test_resident_public_session_rejects_empty_public_time_interval():
    runtime = _OracleRuntime()

    with pytest.raises(ValueError, match="time interval is invalid"):
        runtime_module._ResidentPublicSession(
            runtime,
            _session_case(time_interval=(0.5, 0.5)),
        )

    assert runtime.open_calls == 0


def test_from_observation_rejects_changed_prepared_case_identity(monkeypatch):
    runtime, _service_value, _limit_calls = _runtime(monkeypatch)

    with pytest.raises(RuntimeError, match="changed the prepared case identity"):
        runtime._from_observation(
            _answer_first_observation(),
            case_prefix="gkw-answer-first-binding",
            expected_case_id="gkw-different-prepared-case",
            anchor_keys=(
                {"trajectory": 0, "start_position": 1, "end_position": 2},
            ),
            sampled_public_condition=_PUBLIC_CONDITION,
            seed=97,
            perturbed=False,
            parent_handle=None,
            panel=None,
            cohort="development",
            panel_profile=runtime_module._DEVELOPMENT_PANEL_PROFILE,
        )

    assert runtime._pending_resident_handles == {}
    assert runtime._handles == {}


@pytest.mark.parametrize(
    "report",
    (
        pytest.param(_decoder_report(enforced=False), id="enforcement"),
        pytest.param(_decoder_report(precision="float32"), id="precision"),
    ),
)
def test_generation_rejects_decoder_abi_mismatch_before_bridge_finalize(
    monkeypatch,
    report,
):
    runtime, _service_value, _limit_calls = _runtime(monkeypatch)
    bridge = _FakeBridge()
    decoder = _ReportingDecoder(report)
    runtime.service.decoder = decoder
    runtime._symbols = replace(
        runtime._symbols,
        streaming_bridge=lambda *args, **kwargs: bridge,
        retention_policy=_RetentionPolicy,
    )

    with pytest.raises(
        RuntimeError,
        match="distribution or physical gates",
    ) as caught:
        runtime._run_bridge(
            object(),
            SimpleNamespace(public_packet=_public_packet()),
            run_root=Path("/scratch/gkw-decoder-abi-test"),
            callback=lambda observation: {},
        )

    assert not isinstance(caught.value, runtime_module._DecoderGateRejection)
    assert bridge.finalize_calls == 0
    assert len(decoder.calls) == 1
