from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MappingProxyType
from types import SimpleNamespace
import hashlib
import unittest
from unittest.mock import patch

import pde_agent_verifier_solver.plugins.gkw.manufactured_backend as backend_module
from pde_agent_verifier_solver.engine import (
    DevelopmentMacroEvaluator,
    EvidenceStage,
    MacroAgentOutcome,
    MacroAgentSpec,
    MacroClosedLoopRunner,
    MacroLoopConfig,
    PreparedMacroCohort,
)
from pde_agent_verifier_solver.engine.candidate_case import candidate_case_id
from pde_agent_verifier_solver.plugins.gkw import (
    GKWAnswerFirstGenerator,
    GKWManufacturedCase,
    GKWManufacturedGeneratorBackend,
    GKWManufacturingAudit,
    GKWMacroMatcher,
    GKW_DEVELOPMENT_SOLUTION_SESSION_METHODS,
    GKW_PUBLIC_SESSION_METHODS,
    GKW_SOLUTION_SAMPLING_MODES,
    GKW_LEGACY_MACRO_OUTPUT_CONTRACT,
    GKW_MACRO_FIELDS,
    bind_gkw_macro_loop,
    gkw_native_macro_query,
)
from pde_agent_verifier_solver.plugins.gkw.manufactured_backend import (
    GKW_QUALIFIED_PERTURBATION_SEED_MODULUS,
)
from pde_agent_verifier_solver.plugins.gkw.contract import (
    gkw_answer_view_from_pairs,
    gkw_macro_query_panel,
    gkw_point_query_coordinates,
)
from pde_agent_verifier_solver.plugins.gkw.public_contract import (
    canonical_gkw_public_metadata,
)
from pde_agent_verifier_solver.plugins.gkw.release_io import (
    GKW_NUMERICAL_ORACLE_SCHEMA,
)
from pde_agent_verifier_solver.plugins.gkw.isolation import (
    snapshot_gkw_case,
)
from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    Case,
    CaseAccessKind,
    CandidateRef,
    CandidateSubmission,
    CaseSource,
    CaseVisibility,
    CohortKind,
    FixedGeneratorCaseSource,
    GenerationKind,
    GenerationPurpose,
    GenerationRequest,
    GeneratorMode,
    SampleRequest,
    open_public_case_session,
)

from tests.gkw_release_fixture import canonical_gkw_public_components


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _SmoothState:
    baseline: float
    epsilon: float = 0.0

    def value(self, times):
        import numpy as np

        time = np.asarray(times, dtype=np.float64)
        unit = (time - 0.5) / 0.5
        envelope = 16.0 * unit**2 * (1.0 - unit) ** 2
        return self.baseline + 0.1 * (time - 0.5) + self.epsilon * envelope

    def derivative(self, times):
        import numpy as np

        time = np.asarray(times, dtype=np.float64)
        unit = (time - 0.5) / 0.5
        envelope_derivative = 64.0 * unit * (1.0 - unit) * (1.0 - 2.0 * unit)
        return 0.1 + self.epsilon * envelope_derivative

    def source(self, times):
        value = self.value(times)
        return self.derivative(times) - value**2


class _StateTruth:
    _scales = {
        "particle_density": 1.0,
        "parallel_momentum": 0.5,
        "kinetic_energy": 1.5,
        "phi": 0.8,
    }

    def __init__(self, state, public_data, *, raises=False):
        self.state = state
        self.public_data = public_data
        self.raises = raises

    def answer(self, query):
        if self.raises:
            raise AssertionError("truth.answer was called during generation")
        import numpy as np

        case = SimpleNamespace(public_data=self.public_data)
        if query.kind.value == "grid":
            times, indices = gkw_macro_query_panel(case)
            base = self.state.value(times)[:, None]
            values = {}
            for name, scale in self._scales.items():
                pairs = np.zeros((len(times), len(indices), 2), dtype=np.float64)
                pairs[..., 0] = scale * base
                pairs[..., 1] = 0.1 * scale * base
                values[name] = pairs
        else:
            times, indices = gkw_point_query_coordinates(case, query)
            base = self.state.value(times)
            values = {}
            for name, scale in self._scales.items():
                pairs = np.zeros((len(times), 2), dtype=np.float64)
                pairs[:, 0] = scale * base
                pairs[:, 1] = 0.1 * scale * base
                values[name] = pairs
        return gkw_answer_view_from_pairs(
            query, values, times=times, field_indices=indices
        )


def _oracle(case_id):
    return {
        "arbitrary_phase_subsets_supported": True,
        "arbitrary_time_queries_supported": True,
        "case_id": case_id,
        "complex_value_encoding": ["real", "imag"],
        "equation_view": "continuous_time_fixed_phase_grid_method_of_lines",
        "field_axis_order": ["s", "kx", "ky"],
        "field_grid_shape": [16, 85, 32],
        "fixed_phase_coordinates_are_in_public_packet": True,
        "generator_parameters_public": False,
        "hidden_parameterization_public": False,
        "limits": {
            "max_rows_per_request": 65536,
            "max_total_rows": 2000000,
            "max_unique_source_times": 256,
            "source_frame_cache_entries": 1,
        },
        "off_grid_phase_values_are_not_part_of_the_semidiscrete_equation": True,
        "phase_axis_order": ["vpar", "mu", "s", "kx", "ky"],
        "phase_grid_shape": [32, 8, 16, 85, 32],
        "queries": {
            "initial_h": {
                "inputs": {"phase_indices": "int64 [N,5]"},
                "output": "float64 [N,2]",
            },
            "initial_phi": {
                "inputs": {"field_indices": "int64 [N,3]"},
                "output": "float64 [N,2]",
            },
            "source": {
                "definition": "S_balance = d_t h_hidden - R_GKW[h_hidden]",
                "inputs": {
                    "phase_indices": "int64 [N,5]",
                    "times": "float64 scalar or [N]",
                },
                "output": "float64 [N,2]",
            },
        },
        "schema": GKW_NUMERICAL_ORACLE_SCHEMA,
        "time_interval_closed": [0.5, 1.0],
        "truth_precision": "float64",
    }


def _freeze_json(value):
    if isinstance(value, dict):
        return MappingProxyType(
            {name: _freeze_json(item) for name, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _public_data(case_id, *, geometry=None, macro_query=None):
    import numpy as np

    if geometry is None:
        raw_arrays, raw_scalars, _metadata = canonical_gkw_public_components(
            case_id
        )
        for value in raw_arrays.values():
            value.setflags(write=False)
        arrays = MappingProxyType(dict(raw_arrays))
        scalars = MappingProxyType(
            {name: np.float64(value) for name, value in raw_scalars.items()}
        )
    else:
        arrays, scalars = geometry
    metadata = _freeze_json(
        canonical_gkw_public_metadata(
            case_id=case_id,
            time_nodes=(0.5, 2.0 / 3.0, 5.0 / 6.0, 1.0),
            byte_count=sum(value.nbytes for value in arrays.values()),
        )
    )
    if macro_query is None:
        times = np.asarray([0.5, 0.75, 1.0], dtype=np.float64)
        indices = np.asarray([[0, 0, 0], [1, 1, 1]], dtype=np.int64)
        times.setflags(write=False)
        indices.setflags(write=False)
        macro_query = MappingProxyType(
            {
                "times": times,
                "field_indices": indices,
                "output_contract": GKW_LEGACY_MACRO_OUTPUT_CONTRACT,
            }
        )
    return MappingProxyType(
        {
            "equation": "GKW electrostatic gyrokinetics",
            "arrays": arrays,
            "scalars": scalars,
            "metadata": metadata,
            "oracle_protocol": _freeze_json(_oracle(case_id)),
            "macro_query": macro_query,
        }
    )


class _PhysicalRuntime:
    supports_perturbations = True
    implementation_sha256 = _sha("test-gkw-physical-runtime-v1")

    def __init__(
        self,
        *,
        corrupt_child_truth_lineage=False,
        corrupt_child_problem_lineage=None,
        answer_first=True,
        truth_answer_raises=False,
        fail_perturbed_call=None,
        fail_commit_once=False,
    ):
        self.corrupt_child_truth_lineage = corrupt_child_truth_lineage
        self.corrupt_child_problem_lineage = corrupt_child_problem_lineage
        self.answer_first = answer_first
        self.truth_answer_raises = truth_answer_raises
        self.fail_perturbed_call = fail_perturbed_call
        self.fail_commit_once = fail_commit_once
        self.clean_seeds = []
        self.perturbed_seeds = []
        self.perturbed_call_count = 0
        self.states = {}
        self.sources = {}
        self.committed_case_ids = set()
        self.pending_case_ids = set()
        self.commit_batches = []
        self.abort_batches = []
        self.release_batches = []
        self.released_handle_ids = set()
        self.development_solution_factory_calls = []

    def manufacture_clean(self, base_state, request, seeds, progress=None):
        self.clean_seeds.extend(seeds)
        result = []
        for seed in seeds:
            case_id = f"gkw-clean-{seed:016x}"
            state = _SmoothState(1.0 + (seed % 7) * 0.001)
            item = self._case(case_id, state, seed)
            self.pending_case_ids.add(item.case_id)
            result.append(item)
        return tuple(result)

    def manufacture_perturbed(
        self, base_state, parent, request, seed, progress=None
    ):
        self.perturbed_call_count += 1
        if self.perturbed_call_count == self.fail_perturbed_call:
            raise RuntimeError("injected second perturbation failure")
        self.perturbed_seeds.append(seed)
        base = parent.hidden_handle
        epsilon = 0.04 + (seed % 5) * 0.001
        state = _SmoothState(base.baseline, epsilon)
        case_id = f"{parent.case_id}-pert-{seed:016x}"
        item = self._case(
            case_id,
            state,
            seed,
            parent_answer_id=parent.audit.answer_id,
            perturbation_id=f"fourier:{seed}:{epsilon:.17g}",
            parent_public=parent.public_data,
        )
        self.pending_case_ids.add(item.case_id)
        return item

    def commit_manufactured(self, items):
        identifiers = tuple(item.case_id for item in items)
        if self.fail_commit_once:
            self.fail_commit_once = False
            raise RuntimeError("injected fake commit failure")
        if len(set(identifiers)) != len(identifiers):
            raise RuntimeError("duplicate fake commit ids")
        if set(identifiers) & self.committed_case_ids:
            raise RuntimeError("reused fake commit id")
        if not set(identifiers) <= self.pending_case_ids:
            raise RuntimeError("fake commit ids are not pending")
        self.pending_case_ids.difference_update(identifiers)
        self.committed_case_ids.update(identifiers)
        self.commit_batches.append(identifiers)

    def abort_manufactured(self, items):
        identifiers = tuple(item.case_id for item in items)
        if not identifiers:
            return
        if not set(identifiers) <= self.pending_case_ids:
            raise RuntimeError("fake abort ids are not pending")
        self.pending_case_ids.difference_update(identifiers)
        self.abort_batches.append(identifiers)

    def release_manufactured(self, handles):
        batch = tuple(handles)
        for handle in batch:
            identity = id(handle)
            if identity in self.released_handle_ids:
                raise RuntimeError("fake resident was released twice")
            active = {
                case_id
                for case_id in self.committed_case_ids
                if self.states[case_id] is handle
            }
            if not active:
                raise RuntimeError("fake release handle is not active")
            self.committed_case_ids.difference_update(active)
            self.released_handle_ids.add(identity)
        self.release_batches.append(batch)


    def public_session_factory(self, case):
        import numpy as np

        if case.case_id not in self.committed_case_ids:
            raise KeyError("fake GKW case was not committed")
        state = self.states[case.case_id]

        def query_source(times, phase_indices):
            rows = np.asarray(phase_indices).shape[0]
            values = np.asarray(times, dtype=np.float64)
            if values.ndim == 0:
                values = np.full(rows, float(values), dtype=np.float64)
            result = np.zeros((rows, 2), dtype=np.float64)
            result[:, 0] = state.source(values)
            return result

        def query_initial_h(indices):
            rows = np.asarray(indices).shape[0]
            result = np.zeros((rows, 2), dtype=np.float64)
            result[:, 0] = state.value(np.full(rows, 0.5))
            return result

        return SimpleNamespace(
            case_id=case.case_id,
            arrays=case.public_data["arrays"],
            scalars=case.public_data["scalars"],
            metadata=case.public_data["metadata"],
            oracle_protocol=case.public_data["oracle_protocol"],
            query_source=query_source,
            query_initial_h=query_initial_h,
            query_initial_phi=query_initial_h,
        )

    def development_solution_session_factory(self, case):
        import numpy as np

        self.development_solution_factory_calls.append(case.case_id)
        if case.case_id not in self.committed_case_ids:
            raise KeyError("fake GKW case was not committed")
        state = self.states[case.case_id]

        def query_completed_macros(times, field_indices):
            requested_times = np.asarray(times, dtype=np.float64)
            rows = np.asarray(field_indices).shape[0]
            base = state.value(requested_times)[:, None]
            values = {}
            for name, scale in _StateTruth._scales.items():
                pairs = np.zeros(
                    (len(requested_times), rows, 2), dtype=np.float64
                )
                pairs[..., 0] = scale * base
                pairs[..., 1] = 0.1 * scale * base
                values[name] = pairs
            return values

        return SimpleNamespace(
            case_id=case.case_id,
            query_completed_macros=query_completed_macros,
        )

    def _case(
        self,
        case_id,
        state,
        seed,
        *,
        parent_answer_id=None,
        perturbation_id=None,
        parent_public=None,
    ):
        geometry = (
            None
            if parent_public is None
            else (parent_public["arrays"], parent_public["scalars"])
        )
        panel = None if parent_public is None else parent_public["macro_query"]
        public = _public_data(case_id, geometry=geometry, macro_query=panel)
        self.states[case_id] = state
        self.sources[case_id] = state.source(public["macro_query"]["times"])
        answer_id = f"answer:{case_id}"
        truth_answer_id = (
            "answer:wrong-truth"
            if self.corrupt_child_truth_lineage and parent_answer_id is not None
            else answer_id
        )
        problem_answer_id = (
            "answer:wrong-problem"
            if parent_answer_id is not None
            and self.corrupt_child_problem_lineage in {"initial", "boundary"}
            else answer_id
        )
        return GKWManufacturedCase(
            case_id=case_id,
            public_data=public,
            truth=_StateTruth(
                state, public, raises=self.truth_answer_raises
            ),
            hidden_handle=state,
            audit=GKWManufacturingAudit(
                seed=seed,
                answer_id=answer_id,
                source_answer_id=answer_id,
                initial_answer_id=(
                    problem_answer_id
                    if self.corrupt_child_problem_lineage == "initial"
                    else answer_id
                ),
                boundary_answer_id=(
                    problem_answer_id
                    if self.corrupt_child_problem_lineage == "boundary"
                    else answer_id
                ),
                truth_answer_id=truth_answer_id,
                answer_first=self.answer_first,
                passed=True,
                parent_answer_id=parent_answer_id,
                perturbation_id=perturbation_id,
                perturbation_qualification_id=(
                    "test-fourier-qualification-v1"
                    if parent_answer_id is not None
                    else None
                ),
                perturbation_qualification_seed=(
                    seed % GKW_QUALIFIED_PERTURBATION_SEED_MODULUS
                    if parent_answer_id is not None
                    else None
                ),
            ),
        )


class _ConstantProgram:
    def __init__(self, value):
        self.value = float(value)

    def predict_macros(self, case, query, budget=None):
        import numpy as np

        times, indices = gkw_macro_query_panel(case)
        pairs = np.zeros((len(times), len(indices), 2), dtype=np.float64)
        pairs[..., 0] = self.value
        pairs[..., 1] = 0.1 * self.value
        return gkw_answer_view_from_pairs(
            query,
            {name: pairs.copy() for name in GKW_MACRO_FIELDS},
            times=times,
            field_indices=indices,
        )


class _SessionProgram(_ConstantProgram):
    def __init__(self, value):
        super().__init__(value)
        self.opened_case_ids = []

    def predict_macros(self, case, query, budget=None):
        session = open_public_case_session(case)
        _ = session.query_initial_h
        self.opened_case_ids.append(case.case_id)
        return super().predict_macros(case, query, budget)


class _Agent:
    def run(self, context):
        candidate_id = f"gkw-physical-r{context.round_index}"
        candidate = CandidateSubmission(
            CandidateRef(
                candidate_id=candidate_id,
                bundle_sha256=_sha(candidate_id),
                round_index=context.round_index,
                agent_id=context.agent.agent_id,
                parent_sha256=context.incumbent.bundle_sha256,
            ),
            _ConstantProgram(1.0),
        )
        context.evaluator.evaluate(
            candidate_id,
            candidate.program,
            case_ids=tuple(case.case_id for case in context.cases),
        )
        return MacroAgentOutcome(candidate)


class GKWManufacturedBackendTests(unittest.TestCase):
    def _source(self, runtime):
        generator = GKWAnswerFirstGenerator(GKWManufacturedGeneratorBackend(runtime))
        return generator, FixedGeneratorCaseSource(
            generator_id=generator.info.program_id,
            artifact_sha256=runtime.implementation_sha256,
            generator=generator,
            base_state={"checkpoint": "frozen"},
        )

    def test_audit_uses_opaque_answer_lineage_not_state_hashes(self):
        audit = GKWManufacturingAudit(
            seed=7,
            answer_id="answer:opaque",
            source_answer_id="answer:opaque",
            initial_answer_id="answer:opaque",
            boundary_answer_id="answer:opaque",
            truth_answer_id="answer:opaque",
            answer_first=True,
            passed=True,
        )
        self.assertEqual(audit.answer_id, "answer:opaque")
        self.assertIsNone(audit.parent_answer_id)
        self.assertIsNone(audit.perturbation_qualification_seed)

    def test_local_seed_derivation_ignores_global_rng_and_changes_with_request(self):
        import random

        import numpy as np

        random.seed(1)
        np.random.seed(2)
        runtime_a = _PhysicalRuntime()
        _generator_a, source_a = self._source(runtime_a)
        rows_a = source_a.sample(
            SampleRequest(3, CohortKind.DEVELOPMENT, 401, 2)
        )

        random.seed(987654)
        np.random.seed(123456)
        runtime_b = _PhysicalRuntime()
        _generator_b, source_b = self._source(runtime_b)
        rows_b = source_b.sample(
            SampleRequest(3, CohortKind.DEVELOPMENT, 401, 2)
        )
        self.assertEqual(runtime_a.clean_seeds, runtime_b.clean_seeds)
        self.assertEqual(
            [row.case.case_id for row in rows_a],
            [row.case.case_id for row in rows_b],
        )

        runtime_c = _PhysicalRuntime()
        _generator_c, source_c = self._source(runtime_c)
        source_c.sample(SampleRequest(3, CohortKind.DEVELOPMENT, 402, 2))
        self.assertNotEqual(runtime_a.clean_seeds, runtime_c.clean_seeds)

    def test_typed_seed_and_purpose_need_no_static_settings(self):
        for purpose in GenerationPurpose:
            with self.subTest(purpose=purpose.value):
                runtime = _PhysicalRuntime()
                backend = GKWManufacturedGeneratorBackend(runtime)
                generator = GKWAnswerFirstGenerator(backend)
                request = GenerationRequest(
                    round_index=3,
                    count=1,
                    mode=GeneratorMode.OFFLINE,
                    kind=GenerationKind.BASE,
                    visibility=CaseVisibility.DEVELOPMENT,
                    seed=401,
                    purpose=purpose,
                    settings={
                        "fixed_generator_id": generator.info.program_id,
                        "fixed_generator_sha256": runtime.implementation_sha256,
                    },
                )

                records = generator.generate(
                    {"checkpoint": "frozen"}, None, request
                )

                self.assertEqual(
                    runtime.clean_seeds,
                    [backend_module._derived_seed(request, 0)],
                )
                self.assertEqual(
                    backend._roots[records[0].case.case_id].provenance[2],
                    purpose.value,
                )

    def test_backend_requires_distinct_completed_solution_capability(self):
        runtime = _PhysicalRuntime()
        runtime.development_solution_session_factory = None
        with self.assertRaisesRegex(
            TypeError, "DEVELOPMENT completed-solution session"
        ):
            GKWManufacturedGeneratorBackend(runtime)

    def test_backend_requires_atomic_commit_hook(self):
        runtime = _PhysicalRuntime()
        runtime.commit_manufactured = None
        with self.assertRaisesRegex(TypeError, "atomic commit hook"):
            GKWManufacturedGeneratorBackend(runtime)

        runtime = _PhysicalRuntime()
        runtime.abort_manufactured = None
        with self.assertRaisesRegex(TypeError, "atomic abort hook"):
            GKWManufacturedGeneratorBackend(runtime)

        runtime = _PhysicalRuntime()
        runtime.release_manufactured = None
        with self.assertRaisesRegex(TypeError, "resident release hook"):
            GKWManufacturedGeneratorBackend(runtime)

    def test_generation_performs_no_truth_or_large_array_validation(self):
        import numpy as np

        original_array_equal = np.array_equal

        def reject_large_array_equal(left, right, *args, **kwargs):
            if max(np.asarray(left).size, np.asarray(right).size) > 128:
                raise AssertionError("large public array was compared online")
            return original_array_equal(left, right, *args, **kwargs)

        runtime = _PhysicalRuntime(truth_answer_raises=True)
        _generator, source = self._source(runtime)
        with patch(
            "pde_agent_verifier_solver.plugins.gkw.public_contract."
            "validate_gkw_public_components",
            side_effect=AssertionError("full public validation ran online"),
        ), patch.object(
            np, "array_equal", side_effect=reject_large_array_equal
        ):
            clean = source.sample(
                SampleRequest(0, CohortKind.DEVELOPMENT, 71, 1)
            )[0]
            child = source.sample(
                SampleRequest(
                    0,
                    CohortKind.DEVELOPMENT,
                    73,
                    1,
                    kind=GenerationKind.PERTURBED,
                    anchors=(clean.case,),
                )
            )[0]
        self.assertIs(
            clean.case.public_data["arrays"],
            child.case.public_data["arrays"],
        )
        self.assertIs(
            clean.case.public_data["macro_query"],
            child.case.public_data["macro_query"],
        )

    def test_perturbation_rejects_a_changed_parent_time_grid(self):
        class ChangedTimeGridRuntime(_PhysicalRuntime):
            def manufacture_perturbed(
                self, base_state, parent, request, seed, progress=None
            ):
                item = super().manufacture_perturbed(
                    base_state, parent, request, seed, progress
                )
                public = dict(item.public_data)
                metadata = dict(public["metadata"])
                metadata["time_nodes"] = (0.5, 0.55, 0.95, 1.0)
                public["metadata"] = _freeze_json(metadata)
                return replace(item, public_data=MappingProxyType(public))

        runtime = ChangedTimeGridRuntime()
        _generator, source = self._source(runtime)
        clean = source.sample(
            SampleRequest(0, CohortKind.DEVELOPMENT, 79, 1)
        )[0]

        with self.assertRaisesRegex(
            ValueError, "trusted public geometry/panel"
        ):
            source.sample(
                SampleRequest(
                    0,
                    CohortKind.DEVELOPMENT,
                    83,
                    1,
                    kind=GenerationKind.PERTURBED,
                    anchors=(clean.case,),
                )
            )

        self.assertEqual(runtime.pending_case_ids, set())


    def test_request_artifact_must_match_physical_runtime(self):
        runtime = _PhysicalRuntime()
        generator = GKWAnswerFirstGenerator(GKWManufacturedGeneratorBackend(runtime))
        source = FixedGeneratorCaseSource(
            generator_id=generator.info.program_id,
            artifact_sha256=_sha("different-gkw-runtime"),
            generator=generator,
            base_state={"checkpoint": "frozen"},
        )

        with self.assertRaisesRegex(
            ValueError, "artifact does not match the physical runtime"
        ):
            source.sample(SampleRequest(0, CohortKind.DEVELOPMENT, 91, 1))

    def test_empty_wrong_count_preserves_the_original_validation_error(self):
        runtime = _PhysicalRuntime()
        runtime.manufacture_clean = (
            lambda base_state, request, seeds, progress=None: ()
        )
        _generator, source = self._source(runtime)
        with self.assertRaisesRegex(ValueError, "wrong clean case count"):
            source.sample(SampleRequest(0, CohortKind.DEVELOPMENT, 109, 1))
        self.assertEqual(runtime.abort_batches, [])

    def test_commit_failure_aborts_pending_and_exact_retry_succeeds(self):
        runtime = _PhysicalRuntime(fail_commit_once=True)
        _generator, source = self._source(runtime)
        request = SampleRequest(0, CohortKind.DEVELOPMENT, 113, 2)

        with self.assertRaisesRegex(RuntimeError, "fake commit failure"):
            source.sample(request)
        self.assertEqual(runtime.pending_case_ids, set())
        self.assertEqual(runtime.committed_case_ids, set())
        self.assertEqual(len(runtime.abort_batches), 1)

        records = source.sample(request)
        self.assertEqual(len(records), 2)
        self.assertEqual(runtime.pending_case_ids, set())
        self.assertEqual(
            runtime.committed_case_ids,
            {record.case.case_id for record in records},
        )

    def test_backend_rejects_duplicate_perturbation_anchors(self):
        runtime = _PhysicalRuntime()
        generator, source = self._source(runtime)
        clean = source.sample(
            SampleRequest(2, CohortKind.DEVELOPMENT, 93, 1)
        )[0]
        request = GenerationRequest(
            round_index=2,
            count=2,
            mode=GeneratorMode.OFFLINE,
            kind=GenerationKind.PERTURBED,
            visibility=CaseVisibility.DEVELOPMENT,
            anchors=(clean.case, clean.case),
            settings={
                "seed": 97,
                "cohort": CohortKind.DEVELOPMENT.value,
                "fixed_generator_id": generator.info.program_id,
                "fixed_generator_sha256": runtime.implementation_sha256,
            },
        )

        with self.assertRaisesRegex(ValueError, "anchors must be unique"):
            generator.generate(
                {"checkpoint": "frozen"},
                None,
                request,
            )

    def test_two_anchor_failure_does_not_commit_and_exact_retry_succeeds(self):
        runtime = _PhysicalRuntime(fail_perturbed_call=2)
        _generator, source = self._source(runtime)
        roots = source.sample(
            SampleRequest(4, CohortKind.DEVELOPMENT, 301, 2)
        )
        root_ids = {record.case.case_id for record in roots}
        self.assertEqual(runtime.committed_case_ids, root_ids)
        self.assertEqual(len(runtime.commit_batches), 1)
        request = SampleRequest(
            4,
            CohortKind.DEVELOPMENT,
            307,
            2,
            kind=GenerationKind.PERTURBED,
            anchors=tuple(record.case for record in roots),
        )

        with self.assertRaisesRegex(RuntimeError, "second perturbation failure"):
            source.sample(request)

        backend = source.generator.backend
        self.assertEqual(len(runtime.commit_batches), 1)
        self.assertEqual(runtime.committed_case_ids, root_ids)
        self.assertEqual(runtime.pending_case_ids, set())
        self.assertEqual(len(runtime.abort_batches), 1)
        self.assertEqual(set(backend._roots), root_ids)
        staged_ids = set(runtime.states) - root_ids
        self.assertEqual(len(staged_ids), 1)
        self.assertTrue(staged_ids.isdisjoint(runtime.committed_case_ids))

        runtime.fail_perturbed_call = None
        children = source.sample(request)
        child_ids = {record.case.case_id for record in children}
        self.assertEqual(len(children), 2)
        self.assertEqual(runtime.committed_case_ids, root_ids | child_ids)
        self.assertEqual(set(runtime.commit_batches[-1]), child_ids)
        self.assertEqual(backend._roots, {})

    def test_public_session_accepts_exact_snapshots_and_rejects_tampering(self):
        import numpy as np

        runtime = _PhysicalRuntime()
        generator, source = self._source(runtime)
        clean = source.sample(
            SampleRequest(3, CohortKind.DEVELOPMENT, 99, 1)
        )[0]
        issued = generator.backend._issued_cases[clean.case.case_id]
        self.assertIs(issued.case, clean.case)
        self.assertIs(issued.manufactured.hidden_handle, runtime.states[clean.case.case_id])
        immutable_snapshot = snapshot_gkw_case(clean.case)

        self.assertIsNot(immutable_snapshot, clean.case)
        self.assertFalse(
            next(iter(immutable_snapshot.public_data["arrays"].values())).flags.writeable
        )
        session = generator.backend.public_session_factory(immutable_snapshot)
        self.assertEqual(session.case_id, clean.case.case_id)

        query = AnswerQuery.grid(fields=GKW_MACRO_FIELDS, t=3, field_row=2)
        with TemporaryDirectory() as temporary:
            cohort = PreparedMacroCohort.prepare(
                cohort_id="gkw-public-session-snapshot",
                records=(clean,),
                matcher=GKWMacroMatcher(),
                query=query,
                private_store_root=Path(temporary) / "private",
            )
            stored_case = cohort.public_cases()[0]
            self.assertIsNot(stored_case, clean.case)
            self.assertIsInstance(
                stored_case.public_data["scalars"]["eps"], np.float64
            )

            evaluator = DevelopmentMacroEvaluator(
                cohort,
                case_credit_limit=1,
                public_session_factory=generator.public_session_factory,
                public_session_methods=GKW_PUBLIC_SESSION_METHODS,
            )
            candidate = _SessionProgram(1.0)
            response = evaluator.evaluate("gkw-session-candidate", candidate)

            self.assertEqual(
                candidate.opened_case_ids,
                [candidate_case_id(clean.case.case_id)],
            )
            self.assertEqual(response.cases[0].score.status, "valid")

            # The candidate receives only the opaque envelope. The GKW backend
            # is opened by the trusted evaluator with its controller snapshot.
            self.assertEqual(
                stored_case.case_id,
                candidate_case_id(clean.case.case_id),
            )
            tampered = cohort._controller_cases()[0]
            self.assertEqual(tampered.case_id, clean.case.case_id)
            tampered.public_data["metadata"]["byte_count"] += 1
            with self.assertRaisesRegex(ValueError, "metadata differs"):
                generator.backend.public_session_factory(tampered)

    def test_development_solution_session_is_distinct_and_hidden_fails_closed(self):
        import numpy as np

        runtime = _PhysicalRuntime()
        generator, source = self._source(runtime)
        clean = source.sample(
            SampleRequest(3, CohortKind.DEVELOPMENT, 100, 1)
        )[0]
        session = generator.backend.development_solution_session_factory(
            snapshot_gkw_case(clean.case)
        )
        self.assertFalse(hasattr(session, "query_source"))
        values = session.query_completed_macros(
            np.asarray([0.625], dtype=np.float64),
            np.asarray([[0, 0, 0], [1, 1, 1]], dtype=np.int64),
        )
        self.assertEqual(
            values["particle_density"].shape, (1, 2, 2)
        )
        self.assertEqual(
            runtime.development_solution_factory_calls,
            [clean.case.case_id],
        )

        hidden = Case(
            case_id=clean.case.case_id,
            source=clean.case.source,
            visibility=CaseVisibility.HIDDEN,
            public_data=clean.case.public_data,
            round_index=clean.case.round_index,
            parent_case_id=clean.case.parent_case_id,
            tags=clean.case.tags,
        )
        with self.assertRaisesRegex(PermissionError, "DEVELOPMENT-only"):
            generator.backend.development_solution_session_factory(hidden)
        self.assertEqual(
            runtime.development_solution_factory_calls,
            [clean.case.case_id],
        )

    def test_candidate_panel_values_are_ignored_but_headers_remain_bound(self):
        import numpy as np

        runtime = _PhysicalRuntime()
        generator, source = self._source(runtime)
        clean = source.sample(
            SampleRequest(1, CohortKind.DEVELOPMENT, 101, 1)
        )[0]
        cohort = PreparedMacroCohort.prepare(
            cohort_id="gkw-candidate-local-panel",
            records=(clean,),
            matcher=GKWMacroMatcher(),
            query=AnswerQuery.grid(fields=GKW_MACRO_FIELDS, t=3, field_row=2),
        )

        candidate_case = cohort.public_cases()[0]
        self.assertEqual(
            candidate_case.case_id,
            candidate_case_id(clean.case.case_id),
        )
        with self.assertRaisesRegex(ValueError, "active issued case id"):
            generator.backend.public_session_factory(candidate_case)

        controller_snapshot = cohort._controller_cases()[0]
        candidate_panel = controller_snapshot.public_data["macro_query"]
        trusted_panel = clean.case.public_data["macro_query"]
        trusted_times = np.asarray(trusted_panel["times"]).copy()
        trusted_indices = np.asarray(trusted_panel["field_indices"]).copy()
        candidate_panel["times"][0] += np.float64(0.125)
        candidate_panel["field_indices"][0, 0] = (
            int(candidate_panel["field_indices"][0, 0]) + 1
        ) % 16

        self.assertFalse(
            np.array_equal(candidate_panel["times"], trusted_panel["times"])
        )
        self.assertFalse(
            np.array_equal(
                candidate_panel["field_indices"],
                trusted_panel["field_indices"],
            )
        )
        session = generator.backend.public_session_factory(controller_snapshot)
        self.assertIs(session.arrays, clean.case.public_data["arrays"])
        self.assertIs(session.scalars, clean.case.public_data["scalars"])
        self.assertIsNot(
            session.arrays,
            controller_snapshot.public_data["arrays"],
        )
        np.testing.assert_array_equal(trusted_panel["times"], trusted_times)
        np.testing.assert_array_equal(
            trusted_panel["field_indices"], trusted_indices
        )

        header_tampered = cohort._controller_cases()[0]
        header_tampered.public_data["macro_query"]["times"] = (
            header_tampered.public_data["macro_query"]["times"][:-1]
        )
        with self.assertRaisesRegex(ValueError, "macro panel differs"):
            generator.backend.public_session_factory(header_tampered)

        contract_tampered = cohort._controller_cases()[0]
        contract_tampered.public_data["macro_query"][
            "output_contract"
        ] = "tampered-output-contract"
        with self.assertRaisesRegex(ValueError, "macro panel differs"):
            generator.backend.public_session_factory(contract_tampered)


    def test_invalid_runtime_session_is_closed_without_masking_validation(self):
        runtime = _PhysicalRuntime()
        generator, source = self._source(runtime)
        clean = source.sample(
            SampleRequest(1, CohortKind.DEVELOPMENT, 103, 1)
        )[0]

        class InvalidSession:
            def __init__(self, *, close_fails):
                self.case_id = "wrong-runtime-case"
                self.close_fails = close_fails
                self.close_calls = 0

            def close(self):
                self.close_calls += 1
                if self.close_fails:
                    raise RuntimeError("injected invalid-session close failure")

        for close_fails in (False, True):
            with self.subTest(close_fails=close_fails):
                session = InvalidSession(close_fails=close_fails)
                runtime.public_session_factory = lambda _case: session
                with self.assertRaisesRegex(
                    ValueError, "session case id differs"
                ) as caught:
                    generator.backend.public_session_factory(clean.case)

                self.assertEqual(session.close_calls, 1)
                notes = getattr(caught.exception, "__notes__", ())
                if close_fails:
                    self.assertTrue(
                        any(
                            "invalid public-session cleanup also failed" in note
                            and "injected invalid-session close failure" in note
                            for note in notes
                        )
                    )
                else:
                    self.assertEqual(notes, ())


    def test_fixed_source_builds_clean_smooth_child_and_reconstructed_truth(self):
        import numpy as np

        runtime = _PhysicalRuntime()
        generator, source = self._source(runtime)
        self.assertTrue(generator.capabilities.perturbations)
        self.assertFalse(generator.capabilities.trainable)
        fields = {field.name: field for field in generator.setup_fields()}
        self.assertIs(fields["perturbations_enabled"].default, True)
        self.assertEqual(fields["generator_sampling_mode"].default, "local_edit")
        self.assertEqual(
            tuple(value for value, _label in fields["generator_sampling_mode"].choices),
            GKW_SOLUTION_SAMPLING_MODES,
        )
        self.assertEqual(
            generator.case_session_methods(CaseAccessKind.PUBLIC_PROBLEM),
            GKW_PUBLIC_SESSION_METHODS,
        )
        self.assertEqual(
            generator.case_session_methods(
                CaseAccessKind.DEVELOPMENT_SOLUTION
            ),
            GKW_DEVELOPMENT_SOLUTION_SESSION_METHODS,
        )
        clean = source.sample(
            SampleRequest(1, CohortKind.DEVELOPMENT, 101, 1)
        )[0]
        child = source.sample(
            SampleRequest(
                1,
                CohortKind.DEVELOPMENT,
                105,
                1,
                kind=GenerationKind.PERTURBED,
                anchors=(clean.case,),
            )
        )[0]

        self.assertIs(child.case.source, CaseSource.PERTURBED)
        self.assertEqual(child.case.parent_case_id, clean.case.case_id)
        self.assertEqual(len(runtime.clean_seeds), 1)
        self.assertEqual(len(runtime.perturbed_seeds), 1)
        base_state = runtime.states[clean.case.case_id]
        clean_session = generator.backend.public_session_factory(clean.case)
        phase_rows = np.zeros((3, 5), dtype=np.int64)
        clean_source = clean_session.query_source(
            np.asarray([0.5, 0.75, 1.0]), phase_rows
        )
        np.testing.assert_allclose(
            clean_source[:, 0],
            base_state.source(np.asarray([0.5, 0.75, 1.0])),
        )
        child_session = generator.backend.public_session_factory(child.case)
        self.assertEqual(child_session.case_id, child.case.case_id)
        child_state = runtime.states[child.case.case_id]
        probe = np.asarray([0.5, 0.75, 1.0])
        perturbation = child_state.value(probe) - base_state.value(probe)
        self.assertAlmostEqual(float(perturbation[0]), 0.0)
        self.assertAlmostEqual(float(perturbation[-1]), 0.0)
        self.assertGreater(float(abs(perturbation[1])), 0.0)
        np.testing.assert_allclose(
            runtime.sources[child.case.case_id],
            child_state.derivative(probe) - child_state.value(probe) ** 2,
        )
        self.assertFalse(
            np.array_equal(
                runtime.sources[clean.case.case_id],
                runtime.sources[child.case.case_id],
            )
        )
        query = AnswerQuery.grid(
            fields=GKW_MACRO_FIELDS, t=3, field_row=2
        )
        view = child.truth.answer(query)
        expected = child_state.value(probe)
        actual = np.asarray(view.fields["particle_density"]).reshape(3, 2, 2)
        np.testing.assert_allclose(actual[:, 0, 0], expected)

    def test_release_is_exact_idempotent_and_order_independent(self):
        for clean_first in (True, False):
            with self.subTest(clean_first=clean_first):
                runtime = _PhysicalRuntime()
                generator, source = self._source(runtime)
                clean = source.sample(
                    SampleRequest(1, CohortKind.DEVELOPMENT, 181, 1)
                )[0]
                child = source.sample(
                    SampleRequest(
                        1,
                        CohortKind.DEVELOPMENT,
                        183,
                        1,
                        kind=GenerationKind.PERTURBED,
                        anchors=(clean.case,),
                    )
                )[0]
                first, second = (
                    (clean, child) if clean_first else (child, clean)
                )
                backend = generator.backend
                backend.release_cases((first.case.case_id,))
                self.assertEqual(len(runtime.release_batches), 1)
                backend.release_cases(
                    (first.case.case_id, "gkw-never-issued")
                )
                self.assertEqual(len(runtime.release_batches), 1)
                remaining = backend.public_session_factory(second.case)
                self.assertEqual(remaining.case_id, second.case.case_id)

                backend.release_cases((second.case.case_id,))
                self.assertEqual(len(runtime.release_batches), 2)
                self.assertEqual(backend._issued_cases, {})
                self.assertEqual(backend._resident_references, {})
                self.assertEqual(backend._roots, {})
                backend.release_cases(
                    (clean.case.case_id, child.case.case_id)
                )
                self.assertEqual(len(runtime.release_batches), 2)
                with self.assertRaisesRegex(ValueError, "active issued case id"):
                    backend.public_session_factory(clean.case)


    def test_inconsistent_source_truth_lineage_fails_closed(self):
        runtime = _PhysicalRuntime(corrupt_child_truth_lineage=True)
        _generator, source = self._source(runtime)
        clean = source.sample(
            SampleRequest(1, CohortKind.DEVELOPMENT, 201, 1)
        )[0]
        request = SampleRequest(
            1,
            CohortKind.DEVELOPMENT,
            205,
            1,
            kind=GenerationKind.PERTURBED,
            anchors=(clean.case,),
        )
        with self.assertRaisesRegex(
            ValueError, "do not share one answer lineage"
        ):
            source.sample(request)
        self.assertEqual(runtime.pending_case_ids, set())
        self.assertEqual(len(runtime.abort_batches), 1)

        runtime.corrupt_child_truth_lineage = False
        child = source.sample(request)[0]
        self.assertEqual(child.case.parent_case_id, clean.case.case_id)

    def test_wrong_parent_answer_lineage_fails_closed(self):
        runtime = _PhysicalRuntime()
        _generator, source = self._source(runtime)
        clean = source.sample(
            SampleRequest(1, CohortKind.DEVELOPMENT, 207, 1)
        )[0]
        original = runtime.manufacture_perturbed

        def corrupt_parent(base_state, parent, request, seed, progress=None):
            item = original(base_state, parent, request, seed, progress)
            return replace(
                item,
                audit=replace(
                    item.audit, parent_answer_id="answer:wrong-parent"
                ),
            )

        runtime.manufacture_perturbed = corrupt_parent
        with self.assertRaisesRegex(ValueError, "invalid answer lineage"):
            source.sample(
                SampleRequest(
                    1,
                    CohortKind.DEVELOPMENT,
                    209,
                    1,
                    kind=GenerationKind.PERTURBED,
                    anchors=(clean.case,),
                )
            )
        self.assertEqual(runtime.pending_case_ids, set())
        self.assertEqual(len(runtime.abort_batches), 1)


    def test_wrong_qualification_seed_mapping_fails_closed(self):
        runtime = _PhysicalRuntime()
        _generator, source = self._source(runtime)
        clean = source.sample(
            SampleRequest(1, CohortKind.DEVELOPMENT, 210, 1)
        )[0]
        original = runtime.manufacture_perturbed

        def corrupt_seed(base_state, parent, request, seed, progress=None):
            item = original(base_state, parent, request, seed, progress)
            wrong = (
                item.audit.perturbation_qualification_seed + 1
            ) % GKW_QUALIFIED_PERTURBATION_SEED_MODULUS
            return replace(
                item,
                audit=replace(
                    item.audit, perturbation_qualification_seed=wrong
                ),
            )

        runtime.manufacture_perturbed = corrupt_seed
        with self.assertRaisesRegex(ValueError, "invalid answer lineage"):
            source.sample(
                SampleRequest(
                    1,
                    CohortKind.DEVELOPMENT,
                    213,
                    1,
                    kind=GenerationKind.PERTURBED,
                    anchors=(clean.case,),
                )
            )
        self.assertEqual(runtime.pending_case_ids, set())
        self.assertEqual(len(runtime.abort_batches), 1)

    def test_initial_boundary_and_answer_first_provenance_fail_closed(self):
        for offset, field in enumerate(("initial", "boundary")):
            with self.subTest(field=field):
                runtime = _PhysicalRuntime(corrupt_child_problem_lineage=field)
                _generator, source = self._source(runtime)
                clean = source.sample(
                    SampleRequest(
                        1, CohortKind.DEVELOPMENT, 211 + offset, 1
                    )
                )[0]
                request = SampleRequest(
                    1,
                    CohortKind.DEVELOPMENT,
                    221 + offset,
                    1,
                    kind=GenerationKind.PERTURBED,
                    anchors=(clean.case,),
                )
                with self.assertRaisesRegex(
                    ValueError, "do not share one answer lineage"
                ):
                    source.sample(request)
                self.assertEqual(runtime.pending_case_ids, set())
                self.assertEqual(len(runtime.abort_batches), 1)

        runtime = _PhysicalRuntime(answer_first=False)
        _generator, source = self._source(runtime)
        with self.assertRaisesRegex(ValueError, "derived answer-first"):
            source.sample(
                SampleRequest(1, CohortKind.DEVELOPMENT, 231, 1)
            )
        self.assertEqual(runtime.pending_case_ids, set())
        self.assertEqual(len(runtime.abort_batches), 1)

    def test_shared_macro_loop_runs_real_paired_backend_path(self):
        runtime = _PhysicalRuntime()
        generator = GKWAnswerFirstGenerator(GKWManufacturedGeneratorBackend(runtime))
        plugin = bind_gkw_macro_loop(
            generator=generator,
            base_state={"checkpoint": "frozen"},
            generator_sha256=runtime.implementation_sha256,
            query_factory=gkw_native_macro_query,
        )
        cold = CandidateSubmission(
            CandidateRef(
                candidate_id="gkw-cold",
                bundle_sha256="b" * 64,
                round_index=0,
                agent_id="baseline",
            ),
            _ConstantProgram(0.5),
        )
        result = MacroClosedLoopRunner(
            legacy_macro_reference=True,
            config=MacroLoopConfig(
                name="gkw-physical-paired",
                round_count=1,
                agents=(MacroAgentSpec("agent", "Agent"),),
                development_case_count=2,
                selection_case_count=2,
                evidence_stages=(
                    EvidenceStage(
                        2,
                        confidence_z=0.0,
                        min_win_rate=1.0,
                        require_cvar_no_regression=True,
                    ),
                ),
                terminal_case_count=2,
                agent_case_credits=2,
                paired_perturbations=True,
                terminal_confidence_z=0.0,
                terminal_min_win_rate=1.0,
            ),
            plugin=plugin,
            cold_start=cold,
            agents={"agent": _Agent()},
        ).run()

        self.assertTrue(result.paired_perturbations)
        self.assertTrue(result.rounds[0].promotion.promoted)
        self.assertTrue(result.terminal.comparison.passed)
        # Development, the one common selection/leaderboard cohort, and
        # terminal each draw one root. Promotion does not resample.
        self.assertEqual(len(runtime.clean_seeds), 3)
        self.assertEqual(len(runtime.perturbed_seeds), 3)
        self.assertEqual(len(set(runtime.clean_seeds + runtime.perturbed_seeds)), 6)


    def test_candidate_public_values_use_trusted_case_and_headers_stay_bound(self):
        import numpy as np

        runtime = _PhysicalRuntime()
        generator, source = self._source(runtime)
        clean = source.sample(
            SampleRequest(1, CohortKind.DEVELOPMENT, 102, 1)
        )[0]
        cohort = PreparedMacroCohort.prepare(
            cohort_id="gkw-candidate-local-public-values",
            records=(clean,),
            matcher=GKWMacroMatcher(),
            query=AnswerQuery.grid(
                fields=GKW_MACRO_FIELDS,
                t=3,
                field_row=2,
            ),
        )

        candidate = cohort.public_cases()[0]
        self.assertEqual(
            candidate.case_id,
            candidate_case_id(clean.case.case_id),
        )
        with self.assertRaisesRegex(ValueError, "active issued case id"):
            generator.backend.public_session_factory(candidate)

        controller_snapshot = cohort._controller_cases()[0]
        trusted_vpar = np.asarray(
            clean.case.public_data["arrays"]["vpar"]
        ).copy()
        candidate_vpar = np.asarray(
            controller_snapshot.public_data["arrays"]["vpar"]
        ).copy()
        candidate_vpar[0] += np.float64(0.125)
        controller_snapshot.public_data["arrays"]["vpar"] = candidate_vpar
        session = generator.backend.public_session_factory(controller_snapshot)
        self.assertIs(session.arrays, clean.case.public_data["arrays"])
        self.assertIsNot(
            session.arrays,
            controller_snapshot.public_data["arrays"],
        )
        self.assertFalse(
            np.array_equal(
                controller_snapshot.public_data["arrays"]["vpar"],
                clean.case.public_data["arrays"]["vpar"],
            )
        )
        np.testing.assert_array_equal(
            clean.case.public_data["arrays"]["vpar"],
            trusted_vpar,
        )

        array_header = cohort._controller_cases()[0]
        array_header.public_data["arrays"]["vpar"] = np.asarray(
            array_header.public_data["arrays"]["vpar"],
            dtype=np.float32,
        )
        with self.assertRaisesRegex(ValueError, "array headers differ"):
            generator.backend.public_session_factory(array_header)

        scalar = cohort._controller_cases()[0]
        scalar.public_data["scalars"]["eps"] = np.float64(
            scalar.public_data["scalars"]["eps"] + 1.0
        )
        with self.assertRaisesRegex(ValueError, "scalars differ"):
            generator.backend.public_session_factory(scalar)

        oracle = cohort._controller_cases()[0]
        oracle.public_data["oracle_protocol"]["limits"][
            "max_total_rows"
        ] += 1
        with self.assertRaisesRegex(ValueError, "oracle protocol differs"):
            generator.backend.public_session_factory(oracle)

        envelope = replace(
            cohort._controller_cases()[0],
            tags=("gkw", "tampered-envelope"),
        )
        with self.assertRaisesRegex(ValueError, "Case envelope differs"):
            generator.backend.public_session_factory(envelope)


if __name__ == "__main__":
    unittest.main()
