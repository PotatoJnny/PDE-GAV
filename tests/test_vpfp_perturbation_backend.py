from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import pde_agent_verifier_solver.plugins.vpfp.perturbation_backend as backend_module

from pde_agent_verifier_solver.plugins.vpfp import (
    VPFPAnchoredPerturbationBackend,
    VPFPContinuousGenerator,
    VPFPLegacyStructuredRuntime,
    VPFPRuntimeCase,
)
from pde_agent_verifier_solver.sdk import (
    Case,
    CaseRecord,
    CaseSource,
    CaseVisibility,
    CohortKind,
    FixedGeneratorCaseSource,
    GenerationKind,
    GenerationPurpose,
    GenerationRequest,
    GeneratorMode,
    RecoverableSamplingRejection,
    SampleRequest,
)


class _Truth:
    def answer(self, query):  # pragma: no cover - this test exercises generation.
        raise AssertionError(query)


_EXPECTED_PHYSICAL_GATES = frozenset({
    "finite",
    "rho_positive_by_sample_and_analytic_bound",
    "macro_displacement_visible",
    "macro_displacement_bounded",
    "microscopic_displacement_visible",
    "microscopic_displacement_bounded",
    "expanded_hermite_energy_visible",
    "source_increment_visible",
    "source_increment_bounded",
    "temporal_derivative_bounded",
    "initial_macro_preserved",
    "initial_coefficients_preserved",
})




def _audit(*, passed=True):
    return {
        "accepted": passed,
        "schema": "vpfp_structured_answer_perturbation_v2",
        "base_answer_embedded_exactly": True,
        "initial_condition_preserved": True,
        "generator_parameters_modified": False,
        "selected_attempt": {
            "accepted": passed,
            "schema": "vpfp_structured_answer_perturbation_audit_v2",
            "gates": {name: passed for name in _EXPECTED_PHYSICAL_GATES},
            "source_recomputed_after_perturbation": True,
            "velocity_normalization_exact_not_quadrature_repaired": True,
        },
    }


def _verification(*, accepted=True, failures=()):
    return {
        "accepted": accepted,
        "failures": failures,
        "metrics": {
            "answer_first_initial_data_rederived_from_primary": True,
            "rho0_materialized_cache_exact": True,
            "g0_materialized_cache_exact": True,
        },
    }


class _PhysicalRuntime:
    supports_perturbations = True

    def __init__(self, *, passing_audit=True):
        self.passing_audit = passing_audit
        self.base_calls = []
        self.perturbed_calls = []

    def generate_base(self, **kwargs):
        self.base_calls.append(kwargs)
        return tuple(
            VPFPRuntimeCase(
                _record(
                    case_id,
                    CaseSource.GENERATED,
                    kwargs["visibility"],
                    kwargs["round_index"],
                ),
                private_handle={"latent_seed": seed},
            )
            for seed, case_id in zip(kwargs["seeds"], kwargs["case_ids"])
        )

    def generate_perturbed(self, **kwargs):
        self.perturbed_calls.append(kwargs)
        return tuple(
            VPFPRuntimeCase(
                _record(
                    case_id,
                    CaseSource.PERTURBED,
                    kwargs["visibility"],
                    kwargs["round_index"],
                    parent_case_id=anchor.case_id,
                ),
                audit=_audit(passed=self.passing_audit),
            )
            for anchor, case_id in zip(kwargs["anchors"], kwargs["case_ids"])
        )


def _record(case_id, source, visibility, round_index, parent_case_id=None):
    return CaseRecord(
        Case(
            case_id=case_id,
            source=source,
            visibility=visibility,
            public_data={
                "equation": "VPFP",
                "question": {"case_id": case_id},
                "arrays": {"t": (0.0, 1.0)},
            },
            round_index=round_index,
            parent_case_id=parent_case_id,
        ),
        _Truth(),
    )


def _source(runtime):
    backend = VPFPAnchoredPerturbationBackend(runtime)
    generator = VPFPContinuousGenerator(backend)
    return FixedGeneratorCaseSource(
        generator_id="vpfp-frozen-test",
        artifact_sha256="a" * 64,
        generator=generator,
        base_state={"external_artifact_token": "ignored-by-frozen-backend"},
    )


def _paired_draw(runtime, *, seed=31):
    source = _source(runtime)
    clean = source.sample(SampleRequest(
        round_index=3,
        cohort=CohortKind.DEVELOPMENT,
        seed=seed,
        count=2,
    ))
    perturbed = source.sample(SampleRequest(
        round_index=3,
        cohort=CohortKind.DEVELOPMENT,
        seed=seed + 4,
        count=2,
        kind=GenerationKind.PERTURBED,
        anchors=tuple(row.case for row in clean),
    ))
    return source, clean, perturbed


def _mock_legacy_transaction_runtime(tmp_path, monkeypatch, *, fail_case_id=None):
    output_root = (tmp_path / "legacy-cases").resolve()
    output_root.mkdir()
    state = {"fail_case_id": fail_case_id}

    class Generator:
        def __init__(self):
            self.calls = []

        def generate(self, seeds):
            self.calls.append(tuple(seeds))
            return {
                "latents": tuple(f"latent:{seed}" for seed in seeds),
                "equation_conditions": tuple((0.1, 1.0) for _ in seeds),
                "equation_seeds": tuple(seeds),
            }

    def materialize(case_dir, **kwargs):
        (case_dir / "payload.txt").write_text(
            kwargs["case_id"], encoding="utf-8"
        )

    def verify(case_dir, **kwargs):
        del kwargs
        case_id = (case_dir / "payload.txt").read_text(encoding="utf-8")
        if state["fail_case_id"] == case_id:
            state["fail_case_id"] = None
            return _verification(accepted=False, failures=("forced-second-case",))
        return _verification()

    def calibrate(base_model, base_latent, **kwargs):
        del base_model
        return SimpleNamespace(
            perturbed_latent=f"perturbed:{base_latent}:{kwargs['seed']}",
            embedded_base_latent=f"embedded:{base_latent}",
            model=object(),
            report=_audit(),
        )

    def load(case_dir, **kwargs):
        case_id = (case_dir / "payload.txt").read_text(encoding="utf-8")
        return _record(
            case_id,
            kwargs["source"],
            kwargs["visibility"],
            kwargs["round_index"],
            kwargs["parent_case_id"],
        )

    runtime = object.__new__(VPFPLegacyStructuredRuntime)
    runtime.output_root = output_root
    runtime.generator = Generator()
    runtime.base_model = SimpleNamespace(config="base-config")
    runtime.expanded_config = "expanded-config"
    runtime.expanded_model = object()
    runtime.perturbation_config = object()
    runtime.maximum_attempts = 1
    runtime.api = SimpleNamespace(
        embed_vpfp_structured_latent=(
            lambda base, expanded, latent: f"embedded:{latent}"
        ),
        calibrate_vpfp_robust_perturbation=calibrate,
        materialize_continuous_case=materialize,
        verify_continuous_case_dir=verify,
        MACRO_OUTPUT_CONTRACT=("rho", "u", "temperature"),
        torch=SimpleNamespace(equal=lambda left, right: left == right),
    )
    runtime.checkpoint = tmp_path / "expanded.pt"
    runtime.grid = object()
    runtime.time = object()
    runtime.chunk_size = 16
    runtime.device = "cpu"
    runtime.verification_samples = 2
    monkeypatch.setattr(backend_module, "load_vpfp_case_record", load)
    return runtime, state


def test_fixed_source_uses_private_handles_and_deterministic_paired_lineage():
    runtime = _PhysicalRuntime()
    source, clean, perturbed = _paired_draw(runtime)

    assert source.supports_perturbations is True
    assert [row.case.source for row in clean] == [CaseSource.GENERATED] * 2
    assert [row.case.source for row in perturbed] == [CaseSource.PERTURBED] * 2
    assert [row.case.parent_case_id for row in perturbed] == [
        row.case.case_id for row in clean
    ]
    assert runtime.perturbed_calls[0]["private_handles"] == tuple(
        {"latent_seed": seed} for seed in runtime.base_calls[0]["seeds"]
    )
    assert all("latent_seed" not in row.case.public_data for row in clean)

    replay_runtime = _PhysicalRuntime()
    _replay_source, replay_clean, replay_perturbed = _paired_draw(replay_runtime)
    assert [row.case.case_id for row in replay_clean] == [
        row.case.case_id for row in clean
    ]
    assert [row.case.case_id for row in replay_perturbed] == [
        row.case.case_id for row in perturbed
    ]
    assert runtime.base_calls[0]["seeds"] == replay_runtime.base_calls[0]["seeds"]
    assert runtime.perturbed_calls[0]["seeds"] == (
        replay_runtime.perturbed_calls[0]["seeds"]
    )
    assert set(runtime.base_calls[0]["seeds"]).isdisjoint(
        runtime.perturbed_calls[0]["seeds"]
    )
    with pytest.raises(ValueError, match="not generated"):
        source.sample(SampleRequest(
            round_index=3,
            cohort=CohortKind.DEVELOPMENT,
            seed=41,
            count=2,
            kind=GenerationKind.PERTURBED,
            anchors=tuple(row.case for row in clean),
        ))

    with pytest.raises(ValueError, match="reuse a case id"):
        source.sample(SampleRequest(
            round_index=3,
            cohort=CohortKind.DEVELOPMENT,
            seed=31,
            count=2,
        ))
    assert len(runtime.base_calls) == 1


def test_released_case_ids_can_be_reused_by_a_later_campaign():
    runtime = _PhysicalRuntime()
    source = _source(runtime)
    request = SampleRequest(
        round_index=0,
        cohort=CohortKind.DEVELOPMENT,
        seed=31,
        count=2,
    )

    first = source.sample(request)
    case_ids = tuple(record.case.case_id for record in first)
    source.release_cases(case_ids)
    second = source.sample(request)

    assert tuple(record.case.case_id for record in second) == case_ids
    assert len(runtime.base_calls) == 2


def test_backend_rejects_unknown_anchor_before_physical_runtime_call():
    runtime = _PhysicalRuntime()
    source = _source(runtime)
    unknown = _record(
        "unknown-clean",
        CaseSource.GENERATED,
        CaseVisibility.DEVELOPMENT,
        1,
    ).case
    with pytest.raises(ValueError, match="not generated"):
        source.sample(SampleRequest(
            round_index=1,
            cohort=CohortKind.DEVELOPMENT,
            seed=9,
            count=1,
            kind=GenerationKind.PERTURBED,
            anchors=(unknown,),
        ))
    assert runtime.perturbed_calls == []




def test_backend_rejects_forged_public_anchor_with_a_cached_case_id():
    runtime = _PhysicalRuntime()
    source = _source(runtime)
    clean = source.sample(SampleRequest(
        round_index=1,
        cohort=CohortKind.DEVELOPMENT,
        seed=3,
        count=1,
    ))
    forged = replace(
        clean[0].case,
        public_data={
            **clean[0].case.public_data,
            "equation": "forged public equation",
        },
    )
    with pytest.raises(ValueError, match="not generated"):
        source.sample(SampleRequest(
            round_index=1,
            cohort=CohortKind.DEVELOPMENT,
            seed=7,
            count=1,
            kind=GenerationKind.PERTURBED,
            anchors=(forged,),
        ))
    assert runtime.perturbed_calls == []


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    (
        ("visibility", CaseVisibility.HIDDEN),
        ("round_index", 99),
    ),
)
def test_backend_rejects_wrong_child_envelope_without_consuming_anchor(
    field, wrong_value
):
    runtime = _PhysicalRuntime()
    source = _source(runtime)
    clean = source.sample(SampleRequest(
        round_index=2,
        cohort=CohortKind.DEVELOPMENT,
        seed=13,
        count=1,
    ))
    anchor = clean[0].case
    valid_generate = runtime.generate_perturbed

    def generate_wrong_envelope(**kwargs):
        rows = valid_generate(**kwargs)
        return tuple(
            replace(
                row,
                record=replace(
                    row.record,
                    case=replace(row.record.case, **{field: wrong_value}),
                ),
            )
            for row in rows
        )

    runtime.generate_perturbed = generate_wrong_envelope
    request = SampleRequest(
        round_index=2,
        cohort=CohortKind.DEVELOPMENT,
        seed=17,
        count=1,
        kind=GenerationKind.PERTURBED,
        anchors=(anchor,),
    )
    with pytest.raises(ValueError, match="requested envelope"):
        source.sample(request)
    attempted_case_id = runtime.perturbed_calls[0]["case_ids"][0]

    runtime.generate_perturbed = valid_generate
    recovered = source.sample(request)
    assert recovered[0].case.case_id == attempted_case_id
    assert recovered[0].case.parent_case_id == anchor.case_id
    assert len(runtime.perturbed_calls) == 2


def test_backend_rejects_changed_parent_time_grid_without_consuming_anchor():
    class ChangedTimeRuntime(_PhysicalRuntime):
        corrupt = True

        def generate_perturbed(self, **kwargs):
            rows = super().generate_perturbed(**kwargs)
            if not self.corrupt:
                return rows
            changed = []
            for row in rows:
                public_data = dict(row.record.case.public_data)
                arrays = dict(public_data["arrays"])
                arrays["t"] = (0.0, 2.0)
                public_data["arrays"] = arrays
                changed.append(
                    replace(
                        row,
                        record=replace(
                            row.record,
                            case=replace(row.record.case, public_data=public_data),
                        ),
                    )
                )
            return tuple(changed)

    runtime = ChangedTimeRuntime()
    source = _source(runtime)
    clean = source.sample(SampleRequest(
        round_index=2,
        cohort=CohortKind.DEVELOPMENT,
        seed=21,
        count=1,
    ))
    anchor = clean[0].case
    request = SampleRequest(
        round_index=2,
        cohort=CohortKind.DEVELOPMENT,
        seed=25,
        count=1,
        kind=GenerationKind.PERTURBED,
        anchors=(anchor,),
    )

    with pytest.raises(ValueError, match="public domain"):
        source.sample(request)
    attempted_case_id = runtime.perturbed_calls[-1]["case_ids"][0]

    runtime.corrupt = False
    recovered = source.sample(request)
    assert recovered[0].case.case_id == attempted_case_id
    assert recovered[0].case.parent_case_id == anchor.case_id
    assert len(runtime.perturbed_calls) == 2


def test_backend_rejects_changed_generator_provenance_and_failed_audit():
    runtime = _PhysicalRuntime(passing_audit=False)
    source = _source(runtime)
    clean = source.sample(SampleRequest(
        round_index=2,
        cohort=CohortKind.DEVELOPMENT,
        seed=4,
        count=1,
    ))
    anchor = clean[0].case
    backend = source.generator.backend
    with pytest.raises(ValueError, match="provenance changed"):
        backend.generate(None, None, GenerationRequest(
            round_index=2,
            count=1,
            mode=GeneratorMode.OFFLINE,
            kind=GenerationKind.PERTURBED,
            visibility=CaseVisibility.DEVELOPMENT,
            seed=8,
            purpose=GenerationPurpose.DEVELOPMENT,
            anchors=(anchor,),
            settings={
                "fixed_generator_id": "vpfp-frozen-test",
                "fixed_generator_sha256": "b" * 64,
            },
        ))
    assert runtime.perturbed_calls == []

    with pytest.raises(ValueError, match="passing physical audit"):
        source.sample(SampleRequest(
            round_index=2,
            cohort=CohortKind.DEVELOPMENT,
            seed=8,
            count=1,
            kind=GenerationKind.PERTURBED,
            anchors=(anchor,),
        ))
    assert len(runtime.perturbed_calls) == 1
    runtime.passing_audit = True
    recovered = source.sample(SampleRequest(
        round_index=2,
        cohort=CohortKind.DEVELOPMENT,
        seed=12,
        count=1,
        kind=GenerationKind.PERTURBED,
        anchors=(anchor,),
    ))
    assert recovered[0].case.parent_case_id == anchor.case_id
    assert len(runtime.perturbed_calls) == 2


def test_vpfp_rejection_audit_is_complete_and_same_anchor_can_retry():
    report = _audit(passed=False)
    attempt = {
        **dict(report["selected_attempt"]),
        "backtrack_index": 0,
        "scale": 1.0,
        "metrics": {"source_increment_relative_l2": 0.91},
    }
    report.update({
        "seed": 123,
        "attempt_count": 1,
        "attempts": [attempt],
    })
    summary = backend_module._physical_rejection_summary(report)
    assert summary["seed"] == 123
    assert summary["attempt_count"] == 1
    assert set(summary["attempts"][0]["failed_gates"]) == _EXPECTED_PHYSICAL_GATES
    assert summary["attempts"][0]["metrics"] == {
        "source_increment_relative_l2": 0.91
    }

    attempt["metrics"] = {
        "finite": np.float32(0.5),
        "nan": float("nan"),
        "nested": {
            "positive": float("inf"),
            "negative": float("-inf"),
        },
    }
    nonfinite = backend_module._physical_rejection_summary(report)
    assert nonfinite["attempts"][0]["metrics"] == {
        "finite": 0.5,
        "nan": {"nonfinite": "nan"},
        "nested": {
            "positive": {"nonfinite": "positive_infinity"},
            "negative": {"nonfinite": "negative_infinity"},
        },
    }
    RecoverableSamplingRejection("nonfinite gate rejection", audit=nonfinite)

    class RejectingRuntime(_PhysicalRuntime):
        def generate_perturbed(self, **kwargs):
            if not self.perturbed_calls:
                self.perturbed_calls.append(kwargs)
                raise RecoverableSamplingRejection(
                    "physical calibration rejected",
                    audit={
                        "schema": "vpfp_perturbation_sampling_rejection_v1",
                        "failed_gates": ["source_increment_bounded"],
                    },
                )
            return super().generate_perturbed(**kwargs)

    runtime = RejectingRuntime()
    source = _source(runtime)
    clean = source.sample(SampleRequest(
        round_index=1,
        cohort=CohortKind.DEVELOPMENT,
        seed=31,
        count=1,
    ))
    backend = source.generator.backend
    assert tuple(backend._anchors) == (clean[0].case.case_id,)
    with pytest.raises(RecoverableSamplingRejection):
        source.sample(SampleRequest(
            round_index=1,
            cohort=CohortKind.DEVELOPMENT,
            seed=35,
            count=1,
            kind=GenerationKind.PERTURBED,
            anchors=(clean[0].case,),
        ))
    assert tuple(backend._anchors) == (clean[0].case.case_id,)

    child = source.sample(SampleRequest(
        round_index=1,
        cohort=CohortKind.DEVELOPMENT,
        seed=39,
        count=1,
        kind=GenerationKind.PERTURBED,
        anchors=(clean[0].case,),
    ))
    assert child[0].case.parent_case_id == clean[0].case.case_id
    assert backend._anchors == {}
    assert len(runtime.perturbed_calls) == 2


@pytest.mark.parametrize("purpose", tuple(GenerationPurpose))
def test_backend_uses_typed_seed_and_purpose_without_static_settings(purpose):
    runtime = _PhysicalRuntime()
    backend = VPFPAnchoredPerturbationBackend(runtime)
    request = GenerationRequest(
        round_index=4,
        count=1,
        mode=GeneratorMode.OFFLINE,
        kind=GenerationKind.BASE,
        visibility=CaseVisibility.DEVELOPMENT,
        seed=101,
        purpose=purpose,
        settings={
            "fixed_generator_id": "vpfp-frozen-test",
            "fixed_generator_sha256": "a" * 64,
            # These stale compatibility keys must not drive canonical runs.
            "seed": 999,
            "cohort": "wrong-static-cohort",
        },
    )

    backend.generate(None, None, request)

    assert runtime.base_calls[0]["cohort"] == purpose.value
    assert runtime.base_calls[0]["seeds"] == backend_module._seeds(request)


def test_generator_advertises_physical_runtime_and_remains_frozen():
    backend = VPFPAnchoredPerturbationBackend(_PhysicalRuntime())
    generator = VPFPContinuousGenerator(backend)
    assert generator.capabilities.perturbations is True
    assert generator.capabilities.guidance is False
    assert generator.capabilities.trainable is False
    fields = {field.name: field.default for field in generator.setup_fields()}
    assert fields["guidance_enabled"] is False
    with pytest.raises(RuntimeError, match="already-frozen"):
        backend.train_base(None, {})
    with pytest.raises(RuntimeError, match="frozen guidance"):
        backend.fit_guidance(None, None, None, {})

    legacy_default = VPFPContinuousGenerator(
        SimpleNamespace(supports_perturbations=False)
    )
    assert legacy_default.capabilities.guidance is True




def test_physical_audit_requires_exact_v2_schema_and_all_twelve_gates():
    assert backend_module._PHYSICAL_GATES == _EXPECTED_PHYSICAL_GATES
    backend_module._require_physical_audit(_audit())

    subset = _audit()
    selected = dict(subset["selected_attempt"])
    gates = dict(selected["gates"])
    gates.pop("source_increment_visible")
    selected["gates"] = gates
    subset["selected_attempt"] = selected
    with pytest.raises(ValueError, match="passing physical audit"):
        backend_module._require_physical_audit(subset)

    wrong_schema = {**_audit(), "schema": "vpfp_structured_answer_perturbation_v1"}
    with pytest.raises(ValueError, match="passing physical audit"):
        backend_module._require_physical_audit(wrong_schema)


def test_vendored_vpfp_module_bundle_identity_and_schema_are_fixed():
    root = (
        Path(__file__).resolve().parents[1]
        / "legacy/vpfp/src/agent_verifier/vpfp"
    )
    expected = {
        "robust_perturbation.py": (
            "38e27d17246e582616f0d73c324ff35d8d68901f8c0dad6815333aace2ad37b6"
        ),
        "analytic_source.py": "c44daa8d87c6a3ff019521dc650798afd88e8c43d6e255b5e487b3873d8734b0",
        "continuous_cases.py": "3e637e4239574c2b82d1b3487cce81c5958a7b830b070fbb06628dfd8ae7ef9d",
        "continuous_macros.py": "b4a22c6af2fe12146d41100d94c2a2a792b3bf30b542cebff7693cb9ace49614",
        "operators.py": "df5188d237cf95eddcb22449bcee252844faf6efca736779f4165abe9f8b7b43",
    }
    for name, expected_sha256 in expected.items():
        assert sha256((root / name).read_bytes()).hexdigest() == expected_sha256
    robust = (root / "robust_perturbation.py").read_bytes()
    assert (
        b'VPFP_ROBUST_PERTURBATION_SCHEMA = "vpfp_structured_answer_perturbation_v2"'
        in robust
    )


def test_legacy_api_rejects_an_unpinned_installed_kernel(tmp_path, monkeypatch):
    source = tmp_path / "robust_perturbation.py"
    source.write_text("# wrong installed kernel\n", encoding="utf-8")
    installed = SimpleNamespace(__file__=str(source))
    imports = []

    def import_module(name):
        imports.append(name)
        return installed

    backend_module._legacy_api.cache_clear()
    monkeypatch.setattr(backend_module.importlib, "import_module", import_module)
    try:
        with pytest.raises(RuntimeError, match="physical module identity mismatch"):
            backend_module._legacy_api()
    finally:
        backend_module._legacy_api.cache_clear()
    assert imports == [
        "agent_verifier.vpfp.robust_perturbation",
        "agent_verifier.vpfp.analytic_source",
        "agent_verifier.vpfp.continuous_cases",
        "agent_verifier.vpfp.continuous_macros",
        "agent_verifier.vpfp.operators",
    ]


@pytest.mark.parametrize(
    "bad_name", ("analytic_source", "continuous_cases")
)
def test_legacy_bundle_rejects_a_wrong_sibling(
    tmp_path, bad_name
):
    source_root = (
        Path(__file__).resolve().parents[1]
        / "legacy/vpfp/src/agent_verifier/vpfp"
    )
    package = tmp_path / "agent_verifier" / "vpfp"
    package.mkdir(parents=True)
    modules = {}
    for module_name in backend_module._EXPECTED_LEGACY_MODULE_SHA256:
        short_name = module_name.rsplit(".", 1)[1]
        target = package / f"{short_name}.py"
        payload = (source_root / f"{short_name}.py").read_bytes()
        if short_name == bad_name:
            payload = b"# wrong sibling module\n"
        target.write_bytes(payload)
        modules[module_name] = SimpleNamespace(__file__=str(target))
    with pytest.raises(
        RuntimeError, match=rf"identity mismatch: .*{bad_name}"
    ):
        backend_module._require_legacy_module_bundle_identity(modules)


def test_legacy_bundle_requires_one_resolved_package_directory(tmp_path):
    source_root = (
        Path(__file__).resolve().parents[1]
        / "legacy/vpfp/src/agent_verifier/vpfp"
    )
    modules = {}
    for index, module_name in enumerate(
        backend_module._EXPECTED_LEGACY_MODULE_SHA256
    ):
        package = tmp_path / f"package-{index}"
        package.mkdir()
        short_name = module_name.rsplit(".", 1)[1]
        target = package / f"{short_name}.py"
        target.write_bytes(
            (source_root / f"{short_name}.py").read_bytes()
        )
        modules[module_name] = SimpleNamespace(__file__=str(target))
    with pytest.raises(RuntimeError, match="share one package directory"):
        backend_module._require_legacy_module_bundle_identity(modules)


@pytest.mark.parametrize(
    "missing_metric", backend_module._ANSWER_FIRST_VERIFICATION_METRICS
)
def test_materialization_verification_requires_answer_first_metrics(missing_metric):
    verification = _verification()
    del verification["metrics"][missing_metric]
    with pytest.raises(RuntimeError, match=missing_metric):
        backend_module._require_answer_first_materialization_verification(verification)


def test_legacy_runtime_passes_exact_seed_vector_and_rejects_wrong_shape(tmp_path, monkeypatch):
    class SeedVector:
        def __init__(self, values, shape):
            self.values = tuple(values)
            self.shape = shape

        def __iter__(self):
            return iter(self.values)

        def __len__(self):
            return len(self.values)

    class Generator:
        malformed = False

        def generate(self, seeds):
            self.received = seeds
            shape = (len(seeds), 1) if self.malformed else (len(seeds),)
            return {
                "latents": tuple(f"latent:{seed}" for seed in seeds),
                "equation_conditions": tuple((0.1, 1.0) for _ in seeds),
                "equation_seeds": SeedVector(seeds, shape),
            }

    generator = Generator()
    runtime = object.__new__(VPFPLegacyStructuredRuntime)
    runtime.generator = generator
    runtime.output_root = (tmp_path / "cases").resolve()
    runtime.output_root.mkdir()
    runtime.api = SimpleNamespace(
        embed_vpfp_structured_latent=lambda base, expanded, latent: f"embedded:{latent}"
    )
    runtime.base_model = SimpleNamespace(config="base-config")
    runtime.expanded_config = "expanded-config"
    runtime.expanded_model = object()
    monkeypatch.setattr(
        runtime,
        "_materialize",
        lambda *args, **kwargs: _record(
            args[0], CaseSource.GENERATED, args[4], args[5]
        ),
    )

    rows = runtime.generate_base(
        seeds=(101, 102),
        case_ids=("clean-a", "clean-b"),
        visibility=CaseVisibility.HIDDEN,
        round_index=4,
        cohort="selection",
    )
    assert generator.received == (101, 102)
    assert [row.private_handle.base_latent for row in rows] == [
        "latent:101",
        "latent:102",
    ]

    generator.malformed = True
    with pytest.raises(ValueError, match="malformed equation seeds"):
        runtime.generate_base(
            seeds=(101, 102),
            case_ids=("clean-c", "clean-d"),
            visibility=CaseVisibility.HIDDEN,
            round_index=4,
            cohort="selection",
        )



def test_legacy_runtime_moves_condition_to_host_for_clean_and_perturbed(
    tmp_path, monkeypatch
):
    runtime, _state = _mock_legacy_transaction_runtime(tmp_path, monkeypatch)
    host_values = np.asarray((0.1, 1.0), dtype=float)

    class CudaCondition:
        def __init__(self):
            self.detach_calls = 0
            self.cpu_calls = 0
            self.numpy_calls = 0

        def __array__(self, dtype=None):
            del dtype
            raise TypeError("can't convert cuda:0 device type tensor to numpy")

        def detach(self):
            self.detach_calls += 1
            return self

        def cpu(self):
            self.cpu_calls += 1
            return self

        def numpy(self):
            self.numpy_calls += 1
            return host_values

    device_condition = CudaCondition()

    def generate(seeds):
        return {
            "latents": tuple(f"latent:{seed}" for seed in seeds),
            "equation_conditions": tuple(device_condition for _ in seeds),
            "equation_seeds": tuple(seeds),
        }

    runtime.generator.generate = generate
    materialized_conditions = []
    calibrated_conditions = []
    ordinary_materialize = runtime.api.materialize_continuous_case
    ordinary_calibrate = runtime.api.calibrate_vpfp_robust_perturbation

    def materialize(case_dir, **kwargs):
        materialized_conditions.append(
            np.asarray(kwargs["equation_condition"], dtype=float).copy()
        )
        return ordinary_materialize(case_dir, **kwargs)

    def calibrate(base_model, base_latent, **kwargs):
        calibrated_conditions.append(
            np.asarray(kwargs["equation_condition"], dtype=float).copy()
        )
        return ordinary_calibrate(base_model, base_latent, **kwargs)

    runtime.api.materialize_continuous_case = materialize
    runtime.api.calibrate_vpfp_robust_perturbation = calibrate
    envelope = {
        "visibility": CaseVisibility.DEVELOPMENT,
        "round_index": 2,
        "cohort": "development",
    }

    clean = runtime.generate_base(
        seeds=(101,), case_ids=("clean-host",), **envelope
    )
    assert isinstance(clean[0].private_handle.equation_condition, np.ndarray)
    child = runtime.generate_perturbed(
        anchors=(clean[0].record.case,),
        private_handles=(clean[0].private_handle,),
        seeds=(201,),
        case_ids=("child-host",),
        **envelope,
    )

    assert device_condition.detach_calls == 1
    assert device_condition.cpu_calls == 1
    assert device_condition.numpy_calls == 1
    assert len(materialized_conditions) == 2
    assert len(calibrated_conditions) == 1
    for condition in materialized_conditions + calibrated_conditions:
        assert np.array_equal(condition, host_values)



def test_legacy_base_batch_rolls_back_first_case_when_second_fails(
    tmp_path, monkeypatch
):
    runtime, _state = _mock_legacy_transaction_runtime(
        tmp_path, monkeypatch, fail_case_id="base-b"
    )
    kwargs = {
        "seeds": (101, 102),
        "case_ids": ("base-a", "base-b"),
        "visibility": CaseVisibility.DEVELOPMENT,
        "round_index": 2,
        "cohort": "development",
    }

    with pytest.raises(RuntimeError, match="failed physical verification"):
        runtime.generate_base(**kwargs)
    assert tuple(runtime.output_root.iterdir()) == ()

    rows = runtime.generate_base(**kwargs)
    assert [row.record.case.case_id for row in rows] == ["base-a", "base-b"]
    assert rows[0].transaction is rows[1].transaction
    rows[0].transaction.commit()
    assert {path.name for path in runtime.output_root.iterdir()} == {
        "base-a", "base-b"
    }



def test_legacy_post_rename_reload_failure_rolls_back_and_retries(
    tmp_path, monkeypatch
):
    runtime, _state = _mock_legacy_transaction_runtime(tmp_path, monkeypatch)
    ordinary_load = backend_module.load_vpfp_case_record
    final_dir = runtime.output_root / "base-a"
    fail_final_once = {"enabled": True}

    def load(case_dir, **kwargs):
        record = ordinary_load(case_dir, **kwargs)
        if case_dir == final_dir and fail_final_once["enabled"]:
            fail_final_once["enabled"] = False
            raise RuntimeError("forced final reload failure")
        return record

    monkeypatch.setattr(backend_module, "load_vpfp_case_record", load)
    kwargs = {
        "seeds": (101,),
        "case_ids": ("base-a",),
        "visibility": CaseVisibility.DEVELOPMENT,
        "round_index": 2,
        "cohort": "development",
    }

    with pytest.raises(RuntimeError, match="forced final reload failure"):
        runtime.generate_base(**kwargs)
    assert tuple(runtime.output_root.iterdir()) == ()

    rows = runtime.generate_base(**kwargs)
    assert rows[0].record.case.case_id == "base-a"
    rows[0].transaction.commit()
    assert final_dir.is_dir()

def test_legacy_perturbed_batch_rolls_back_first_child_and_retries_exactly(
    tmp_path, monkeypatch
):
    runtime, _state = _mock_legacy_transaction_runtime(
        tmp_path, monkeypatch, fail_case_id="child-b"
    )
    anchors = tuple(
        _record(
            f"anchor-{label}",
            CaseSource.GENERATED,
            CaseVisibility.DEVELOPMENT,
            3,
        ).case
        for label in ("a", "b")
    )
    handles = tuple(
        backend_module._LegacyHandle(
            f"latent:{label}", f"embedded:latent:{label}", (0.1, 1.0)
        )
        for label in ("a", "b")
    )
    kwargs = {
        "anchors": anchors,
        "private_handles": handles,
        "seeds": (201, 202),
        "case_ids": ("child-a", "child-b"),
        "visibility": CaseVisibility.DEVELOPMENT,
        "round_index": 3,
        "cohort": "development",
    }

    with pytest.raises(RuntimeError, match="failed physical verification"):
        runtime.generate_perturbed(**kwargs)
    assert tuple(runtime.output_root.iterdir()) == ()

    rows = runtime.generate_perturbed(**kwargs)
    assert [row.record.case.case_id for row in rows] == ["child-a", "child-b"]
    assert [row.record.case.parent_case_id for row in rows] == [
        "anchor-a", "anchor-b"
    ]
    assert rows[0].transaction is rows[1].transaction
    rows[0].transaction.commit()
    assert {path.name for path in runtime.output_root.iterdir()} == {
        "child-a", "child-b"
    }


def test_legacy_batch_preflight_preserves_a_preexisting_later_target(
    tmp_path, monkeypatch
):
    runtime, _state = _mock_legacy_transaction_runtime(tmp_path, monkeypatch)
    existing = runtime.output_root / "base-b"
    existing.mkdir()
    sentinel = existing / "sentinel.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError):
        runtime.generate_base(
            seeds=(1, 2),
            case_ids=("base-a", "base-b"),
            visibility=CaseVisibility.DEVELOPMENT,
            round_index=1,
            cohort="development",
        )
    assert not (runtime.output_root / "base-a").exists()
    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert runtime.generator.calls == []


@pytest.mark.parametrize("corruption", ("audit", "envelope"))
def test_backend_validation_failure_rolls_back_runtime_batch_and_retries(
    tmp_path, monkeypatch, corruption
):
    physical, _state = _mock_legacy_transaction_runtime(tmp_path, monkeypatch)

    class CorruptSecondRow:
        supports_perturbations = True

        def __init__(self):
            self.corrupt = False
            self.last_child_ids = ()

        def generate_base(self, **kwargs):
            return physical.generate_base(**kwargs)

        def generate_perturbed(self, **kwargs):
            rows = physical.generate_perturbed(**kwargs)
            self.last_child_ids = tuple(
                row.record.case.case_id for row in rows
            )
            if not self.corrupt:
                return rows
            if corruption == "audit":
                return rows[:1] + (replace(rows[1], audit={}),)
            wrong_case = replace(
                rows[1].record.case, visibility=CaseVisibility.HIDDEN
            )
            return rows[:1] + (
                replace(rows[1], record=replace(rows[1].record, case=wrong_case)),
            )

    runtime = CorruptSecondRow()
    source = _source(runtime)
    clean = source.sample(SampleRequest(
        round_index=2,
        cohort=CohortKind.DEVELOPMENT,
        seed=19,
        count=2,
    ))
    clean_ids = {row.case.case_id for row in clean}
    runtime.corrupt = True
    request = SampleRequest(
        round_index=2,
        cohort=CohortKind.DEVELOPMENT,
        seed=23,
        count=2,
        kind=GenerationKind.PERTURBED,
        anchors=tuple(row.case for row in clean),
    )
    error = "passing physical audit" if corruption == "audit" else "requested envelope"

    with pytest.raises(ValueError, match=error):
        source.sample(request)
    attempted_ids = runtime.last_child_ids
    assert set(path.name for path in physical.output_root.iterdir()) == clean_ids
    assert all(not (physical.output_root / case_id).exists() for case_id in attempted_ids)

    runtime.corrupt = False
    recovered = source.sample(request)
    assert tuple(row.case.case_id for row in recovered) == attempted_ids
    assert set(path.name for path in physical.output_root.iterdir()) == (
        clean_ids | set(attempted_ids)
    )


@pytest.mark.parametrize("case_id", ("", ".", "..", "../escape", "nested/case"))
def test_legacy_materialization_requires_one_safe_case_path_component(
    tmp_path, case_id
):
    with pytest.raises(ValueError, match="safe path component"):
        backend_module._case_target(tmp_path.resolve(), case_id)


def test_legacy_materialization_failure_does_not_poison_retry(
    tmp_path, monkeypatch
):
    output_root = (tmp_path / "cases").resolve()
    output_root.mkdir()
    staging_paths = []
    verification_calls = []
    load_paths = []

    def materialize(case_dir, **kwargs):
        staging_paths.append(case_dir)
        assert case_dir.parent == output_root
        assert case_dir.name.startswith(".vpfp-case-")
        (case_dir / "payload.txt").write_text(
            kwargs["case_id"], encoding="utf-8"
        )

    def verify(case_dir, **kwargs):
        verification_calls.append((case_dir, kwargs))
        if len(verification_calls) == 1:
            return _verification(accepted=False, failures=("forced-midway-failure",))
        return _verification()

    case_id = "vpfp-r1-base-transaction-test"

    def load(case_dir, **kwargs):
        load_paths.append(case_dir)
        assert (case_dir / backend_module.VPFP_UNIFIED_GENERATION_AUDIT).is_file()
        return _record(
            case_id,
            kwargs["source"],
            kwargs["visibility"],
            kwargs["round_index"],
            kwargs["parent_case_id"],
        )

    runtime = object.__new__(VPFPLegacyStructuredRuntime)
    runtime.output_root = output_root
    runtime.api = SimpleNamespace(
        materialize_continuous_case=materialize,
        verify_continuous_case_dir=verify,
        MACRO_OUTPUT_CONTRACT=("rho", "u", "temperature"),
    )
    runtime.checkpoint = tmp_path / "expanded.pt"
    runtime.grid = object()
    runtime.time = object()
    runtime.chunk_size = 16
    runtime.device = "cpu"
    runtime.verification_samples = 2
    monkeypatch.setattr(backend_module, "load_vpfp_case_record", load)

    def run():
        return runtime._materialize(
            case_id,
            object(),
            object(),
            object(),
            CaseVisibility.DEVELOPMENT,
            1,
            "development",
            None,
            "base",
            {"equation_seed": 7},
        )

    with pytest.raises(RuntimeError, match="failed physical verification"):
        run()
    final_dir = output_root / case_id
    assert not final_dir.exists()
    assert not staging_paths[0].exists()
    assert tuple(output_root.iterdir()) == ()

    record = run()
    assert record.case.case_id == case_id
    assert final_dir.is_dir()
    assert (final_dir / "payload.txt").read_text(encoding="utf-8") == case_id
    assert load_paths == [staging_paths[1], final_dir]
    assert not staging_paths[1].exists()
    assert tuple(output_root.iterdir()) == (final_dir,)

    sentinel = final_dir / "existing-sentinel.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run()
    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert len(staging_paths) == 2


def test_legacy_runtime_preflights_expanded_checkpoint_config(tmp_path, monkeypatch):
    class Config:
        def to_dict(self):
            return {"representation": "expanded"}

    class BaseModel:
        config = object()

        def parameters(self):
            return iter((SimpleNamespace(device="cpu"),))

    class Geometry:
        def validate(self):
            pass

    class Torch:
        payload = {
            "decoder_config": {"representation": "expanded"},
            "robust_answer_first_adapter": {
                "unused_neural_parameters_zeroed": True,
                "continuous_hidden_answer_is_primary": True,
                "generator_parameterization_public": False,
            },
        }

        @classmethod
        def load(cls, *args, **kwargs):
            return cls.payload

    api = SimpleNamespace(
        VPFPRobustPerturbationConfig=lambda: object(),
        expanded_vpfp_perturbation_config=lambda *args: Config(),
        build_deterministic_direct_model=lambda *args, **kwargs: object(),
        torch=Torch,
    )
    monkeypatch.setattr(backend_module, "_legacy_api", lambda: api)
    checkpoint = tmp_path / "expanded.pt"
    checkpoint.write_bytes(b"qualified-checkpoint-placeholder")
    settings = {
        "generator": object(),
        "base_model": BaseModel(),
        "expanded_decoder_checkpoint": checkpoint,
        "output_root": tmp_path / "cases",
        "materialized_grid": Geometry(),
        "materialized_time": Geometry(),
    }
    runtime = VPFPLegacyStructuredRuntime(**settings)
    backend = VPFPAnchoredPerturbationBackend(runtime)
    assert runtime.materialize_cases is True
    assert backend.public_session_methods == ()
    assert backend.development_solution_session_methods == ()

    Torch.payload = {**Torch.payload, "decoder_config": {"representation": "wrong"}}
    with pytest.raises(ValueError, match="not qualified"):
        VPFPLegacyStructuredRuntime(**settings)
