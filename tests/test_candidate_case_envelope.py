from __future__ import annotations

from dataclasses import replace

import pytest
import re
from typing import Mapping

from pde_agent_verifier_solver.engine.candidate_case import (
    candidate_case_id,
    candidate_facing_case,
)
from pde_agent_verifier_solver.engine.macro_evaluator import (
    ControllerMacroEvaluator,
    DevelopmentMacroEvaluator,
    PreparedMacroCohort,
)
from pde_agent_verifier_solver.engine.reference_agent_runtime import (
    ReferenceDevelopmentEvaluatorRPC,
    _PublicValueCodec,
    _reference_controller_cases_for_evaluator,
    _reference_public_case_payload,
)
from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    AnswerView,
    Case,
    CaseRecord,
    CaseSource,
    CaseVisibility,
    MacroMatchResult,
    MacroTarget,
    ProgramInfo,
    QueryKind,
    open_public_case_session,
)


_OPAQUE_CASE_ID = re.compile(r"candidate-case-[0-9a-f]{64}\Z")
_FORBIDDEN_CONTROL_KEYS = {
    "answer_first_perturbation_applied",
    "anchor_case_id",
    "controller_cohort",
    "development_cohort",
    "distribution_lineage",
    "generation_round",
    "logical_cohort_id",
    "logical_root_count",
    "logical_root_index",
    "logical_root_offset",
    "case_source",
    "cohort",
    "cohort_id",
    "cohort_kind",
    "dataset_split",
    "draw_index",
    "draw_kind",
    "generation_kind",
    "generation_variant",
    "is_perturbed",
    "parent_case_id",
    "perturbation",
    "perturbation_applied",
    "perturbation_metadata",
    "policy_variant",
    "robust_variant",
    "round",
    "round_index",
    "sample_kind",
    "source_round_index",
    "tags",
    "target_round_index",
    "visibility",
}


def _recursive_mapping_items(value):
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield key, item
            yield from _recursive_mapping_items(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _recursive_mapping_items(item)


def _assert_control_labels_scrubbed(case: Case, *private_values: str) -> None:
    assert case.source is CaseSource.GENERATED
    assert case.visibility is CaseVisibility.DEVELOPMENT
    assert case.round_index is None
    assert case.parent_case_id is None
    assert case.tags == ()
    assert _OPAQUE_CASE_ID.fullmatch(case.case_id)

    for key, item in _recursive_mapping_items(case.public_data):
        if not isinstance(key, str):
            continue
        normalized = key.strip().lower()
        assert normalized not in _FORBIDDEN_CONTROL_KEYS
        if normalized == "case_id":
            assert item == case.case_id

    public_repr = repr(case.public_data)
    for private_value in private_values:
        assert private_value not in public_repr


def _ns_case() -> Case:
    case_id = "ns-r8-terminal-perturbed-private-91"
    parent_id = "ns-r8-terminal-clean-private-18"
    return Case(
        case_id=case_id,
        source=CaseSource.PERTURBED,
        visibility=CaseVisibility.HIDDEN,
        public_data={
            "equation": "2D incompressible Navier--Stokes",
            "material_case_id": "alloy-A",
            "rotation_round": 3.0,
            "distribution_lineage": {
                "kind": "frozen_reward_guided",
                "snapshot_sha256": "guided-private-snapshot",
                "base_generator_sha256": "guided-private-base",
            },
            "operator": {
                "operator_id": "incompressible_ns_2d_v1",
                "form": (
                    "d_t velocity + velocity dot grad velocity = "
                    "-grad pressure + viscosity laplacian velocity + forcing"
                ),
                "constraint": "divergence(velocity) = 0",
            },
            "parameters": {"viscosity": 0.001},
            "domain": {
                "coordinate_order": ["t", "x", "y"],
                "time_interval": [0.0, 1.0],
                "bounds": {"x": [0.0, 2.0], "y": [0.0, 1.0]},
                "geometry": {"kind": "channel"},
            },
            "boundary_condition": {
                "kind": "no_slip_velocity",
                "velocity": [0.0, 0.0],
            },
            "initial_condition": {
                "field": ["velocity_x", "velocity_y"],
                "access": "query_initial_velocity",
            },
            "forcing": {
                "field": ["forcing_x", "forcing_y"],
                "access": "query_forcing",
                "derivation": "public_NS_operator_applied_to_hidden_answer",
            },
            "answer_first": {
                "contract": "continuous_ns_answer_first_v1",
                "solution_chosen_before_question": True,
                "forcing_rederived_after_perturbation": True,
            },
            "continuous_answer_metadata": {
                "basis_family": "streamfunction_fourier",
                "controller": {
                    "cohort": "terminal",
                    "source_round_index": 7,
                    "policy_variant": "perturbed",
                    "anchor_case_id": parent_id,
                },
            },
            "perturbation": {
                "family": "smooth_spectral_support_complement",
                "randomized_realization": True,
            },
            "audit": [
                {
                    "generation_kind": "perturbed",
                    "round": 8,
                    "visibility": "hidden",
                }
            ],
        },
        round_index=8,
        parent_case_id=parent_id,
        tags=("ns", "answer-first", "perturbed", "cohort:terminal"),
    )


def _vpfp_case() -> Case:
    case_id = "vpfp-r4-selection-perturbed-private-22"
    parent_id = "vpfp-r4-selection-base-private-11"
    return Case(
        case_id=case_id,
        source=CaseSource.PERTURBED,
        visibility=CaseVisibility.HIDDEN,
        public_data={
            "equation": "2D2V Vlasov--Poisson--Fokker--Planck",
            "question": {
                "case_id": case_id,
                "solver_output_contract_version": (
                    "vpfp_continuous_macro_output_v1"
                ),
                "collision_nu": 0.05,
                "temperature": 1.0,
                "time_domain": [0.0, 6.0],
                "source_origin": "generated_continuous_autodiff_residual",
                "source_derivative_backend": "analytic",
                "generator_metadata": {
                    "generator_family": "coarse_projection_answer_first",
                    "robust_variant": "perturbed",
                    "generation_round": 4,
                },
                "case_spec": {
                    "cohort": "selection",
                    "truth_primary": "continuous_hidden_answer",
                    "answer_first_perturbation_applied": True,
                    "parent_case_id": parent_id,
                },
                "nested": [
                    {
                        "variant": "perturbed",
                        "development_cohort": "selection",
                    }
                ],
                "physical_variant": "long_time_weak_collision_v2",
            },
            "arrays": {
                "t": [0.0, 3.0, 6.0],
                "x": [0.0, 0.5, 1.0],
                "rho0": [1.0, 1.1, 0.9],
            },
        },
        round_index=4,
        parent_case_id=parent_id,
        tags=("vpfp", "unified-paired", "perturbed", "cohort:selection"),
    )


def test_real_ns_shape_is_phase_blind_but_keeps_the_public_pde() -> None:
    raw = _ns_case()

    projected = candidate_facing_case(raw)

    _assert_control_labels_scrubbed(
        projected,
        raw.case_id,
        raw.parent_case_id,
        "terminal",
        "perturbed",
    )
    assert projected.public_data["operator"] == raw.public_data["operator"]
    assert projected.public_data["parameters"] == raw.public_data["parameters"]
    assert projected.public_data["material_case_id"] == "alloy-A"
    assert projected.public_data["rotation_round"] == 3.0
    assert "distribution_lineage" not in projected.public_data
    assert projected.public_data["domain"] == raw.public_data["domain"]
    assert (
        projected.public_data["initial_condition"]
        == raw.public_data["initial_condition"]
    )
    assert projected.public_data["forcing"] == raw.public_data["forcing"]
    assert (
        projected.public_data["answer_first"]["forcing_rederived_after_perturbation"]
        is True
    )


def test_real_vpfp_shape_is_recursively_scrubbed_and_projection_is_idempotent() -> None:
    raw = _vpfp_case()

    projected = candidate_facing_case(raw)
    projected_again = candidate_facing_case(projected)

    _assert_control_labels_scrubbed(
        projected,
        raw.case_id,
        raw.parent_case_id,
        "selection",
        "perturbed",
    )
    assert projected_again == projected
    assert candidate_case_id(raw.case_id) == projected.case_id
    assert candidate_case_id(raw.case_id) == candidate_case_id(raw.case_id)
    question = projected.public_data["question"]
    assert question["case_id"] == projected.case_id
    assert question["collision_nu"] == 0.05
    assert question["temperature"] == 1.0
    assert question["source_origin"] == (
        "generated_continuous_autodiff_residual"
    )
    assert question["case_spec"]["truth_primary"] == (
        "continuous_hidden_answer"
    )
    assert question["physical_variant"] == "long_time_weak_collision_v2"
    assert projected.public_data["arrays"] == raw.public_data["arrays"]


def test_piv_projection_hides_controller_distribution_but_keeps_project_artifact() -> None:
    descriptor = {
        "schema": "pde_project_reconstruction_artifact_v1",
        "artifact_id": "ns-piv-trajectory-pod",
        "artifact_kind": "trajectory_pod",
        "format": "numpy_npz",
        "locator": "/project/data/trajectory_pod.npz",
        "sha256": "a" * 64,
        "size_bytes": 158_657_740,
        "selected_method": {
            "method_id": "gappy_pod_map",
            "rank": 64,
            "prior_precision": 10.0,
        },
    }
    raw = Case(
        case_id="private-piv-generated-1",
        source=CaseSource.PERTURBED,
        visibility=CaseVisibility.HIDDEN,
        public_data={
            "equation": "2D incompressible Navier--Stokes",
            "source_recording": "cylinder_vel",
            "source_frames": (4800, 4832),
            "evaluation_reference": "generated_complete_field",
            "generator": {
                "kind": "perturbed",
                "mode": "online",
                "validity_adjusted": True,
            },
            "project_reconstruction": descriptor,
        },
        round_index=3,
    )

    projected = candidate_facing_case(raw)

    assert "generator" not in projected.public_data
    assert "evaluation_reference" not in projected.public_data
    assert projected.public_data["source_recording"] == "cylinder_vel"
    assert projected.public_data["source_frames"] == (4800, 4832)
    assert projected.public_data["project_reconstruction"] == descriptor


def test_prepared_cohort_rejects_reserved_candidate_id_prefix() -> None:
    raw = replace(_ns_case(), case_id="candidate-case-" + "a" * 64)
    matcher = _Matcher(raw)

    with pytest.raises(ValueError, match="candidate id prefix"):
        PreparedMacroCohort.prepare(
            cohort_id="reserved-prefix",
            records=(CaseRecord(raw, _Truth()),),
            matcher=matcher,
            query=AnswerQuery.grid(fields=("macro",), sample=1),
        )


_QUERY = AnswerQuery.grid(fields=("macro",), sample=1)


class _Truth:
    def answer(self, query: AnswerQuery) -> AnswerView:
        return AnswerView(
            kind=query.kind,
            fields={"macro": (2.0,)},
            axes={"sample": (0.0,)},
            shape=(1,),
        )


class _Matcher:
    info = ProgramInfo("candidate-envelope-test", "Envelope", "test")
    contract_id = "candidate_envelope_macro_v1"
    fields = ("macro",)

    def __init__(self, raw_case: Case) -> None:
        self.raw_case = raw_case
        self.match_cases = []

    def prepare_target(
        self,
        case: Case,
        truth: AnswerView,
        query: AnswerQuery,
    ) -> MacroTarget:
        assert case == self.raw_case
        return MacroTarget(case.case_id, self.contract_id, query, truth)

    def match(
        self,
        case: Case,
        prediction: AnswerView,
        target: MacroTarget,
    ) -> MacroMatchResult:
        self.match_cases.append(case)
        assert case == self.raw_case
        assert target.case_id == self.raw_case.case_id
        error = abs(prediction.fields["macro"][0] - target.values.fields["macro"][0])
        return MacroMatchResult(case.case_id, self.contract_id, error)


class _Session:
    def __init__(self, raw_case: Case) -> None:
        self.raw_case = raw_case

    def marker(self) -> str:
        return self.raw_case.case_id


class _ProbeCandidate:
    def __init__(self, raw_case: Case) -> None:
        self.raw_case = raw_case
        self.seen = []

    def predict_macros(
        self,
        case: Case,
        query: AnswerQuery,
        budget=None,
    ) -> AnswerView:
        self.seen.append(case)
        _assert_control_labels_scrubbed(
            case,
            self.raw_case.case_id,
            self.raw_case.parent_case_id,
            "terminal",
            "perturbed",
        )
        assert open_public_case_session(case).marker() == self.raw_case.case_id
        return AnswerView(
            kind=QueryKind.GRID,
            fields={"macro": (2.0,)},
            axes={"sample": (0.0,)},
            shape=(1,),
        )


def _cohort_with_session_probe():
    raw = _ns_case()
    matcher = _Matcher(raw)
    cohort = PreparedMacroCohort.prepare(
        cohort_id="round-8-terminal-private",
        records=(CaseRecord(raw, _Truth()),),
        matcher=matcher,
        query=_QUERY,
    )
    factory_cases = []

    def factory(case: Case) -> _Session:
        factory_cases.append(case)
        return _Session(case)

    return raw, matcher, cohort, factory, factory_cases


def test_macro_evaluator_projects_only_the_candidate_boundary() -> None:
    raw, matcher, cohort, factory, factory_cases = _cohort_with_session_probe()
    candidate = _ProbeCandidate(raw)

    controller_batch = ControllerMacroEvaluator(
        cohort,
        public_session_factory=factory,
        public_session_methods=("marker",),
    ).evaluate("probe", candidate)

    assert controller_batch.case_ids == (raw.case_id,)
    assert controller_batch.accepted
    assert matcher.match_cases == [raw]
    assert factory_cases == [raw]
    assert candidate.seen[0] == cohort.public_cases()[0]

    development = DevelopmentMacroEvaluator(
        cohort,
        case_credit_limit=1,
        public_session_factory=factory,
        public_session_methods=("marker",),
    )
    public_id = development.public_cases[0].case_id
    response = development.evaluate(
        "development-probe",
        candidate,
        case_ids=(public_id,),
    )
    assert response.cases[0].score.case_id == public_id
    assert public_id != raw.case_id
    assert factory_cases[-1] == raw


def test_reference_payload_is_idempotent_and_oracle_alias_binds_raw_case(
    tmp_path,
) -> None:
    raw, _matcher, cohort, factory, factory_cases = _cohort_with_session_probe()
    development = DevelopmentMacroEvaluator(cohort, case_credit_limit=1)
    public_cases = development.public_cases

    controller_cases = _reference_controller_cases_for_evaluator(
        development,
        public_cases,
    )
    assert controller_cases == (raw,)

    payload_root = tmp_path / "payload"
    payload_root.mkdir()
    codec = _PublicValueCodec(payload_root / "arrays")
    raw_payload = _reference_public_case_payload(raw, codec)
    projected_payload = _reference_public_case_payload(public_cases[0], codec)
    assert raw_payload == projected_payload
    assert raw_payload["case_id"] == public_cases[0].case_id
    assert raw_payload["visibility"] == CaseVisibility.DEVELOPMENT.value
    assert raw_payload["source"] == CaseSource.GENERATED.value

    rpc_root = tmp_path / "rpc"
    rpc_root.mkdir()
    (rpc_root / "arrays").mkdir()
    rpc = ReferenceDevelopmentEvaluatorRPC(
        development,
        workspace=rpc_root,
        oracle_cases=controller_cases,
        public_session_factory=factory,
        public_session_methods=("marker",),
    )
    request_codec = _PublicValueCodec(rpc_root / "arrays", create=False)
    try:
        response = rpc._server.call_numerical_oracle(
            {
                "schema": "reference-numerical-oracle-request-v1",
                "namespace": "equation",
                "case_id": public_cases[0].case_id,
                "method": "marker",
                "args": request_codec.encode(()),
                "kwargs": request_codec.encode({}),
            }
        )
        assert request_codec.decode(
            response["result"],
            mmap_arrays=False,
        ) == raw.case_id
    finally:
        rpc._server.server_close()

    assert factory_cases == [raw]
