"""Label adapter contracts; no trained model is loaded or evaluated here."""

from dataclasses import dataclass, FrozenInstanceError, replace
import json
from pathlib import Path
import subprocess
import sys
from types import MethodType, SimpleNamespace

import pytest

from pde_agent_verifier_solver.plugins.gkw import legacy_local_edit as adapter


PIN = adapter.LEGACY_LOCAL_EDIT_PHYSICAL_REVISION


@dataclass(frozen=True)
class _Config:
    kinetic_sampling_steps: int = 32
    potential_sampling_steps: int = 32
    sampling_sigma_max: float = .1


@dataclass(frozen=True)
class _Certificate:
    physical_git_commit: str = PIN
    qualification_code_commit: str = PIN
    digest: str = "fixture-not-a-production-qualification"


@dataclass(frozen=True)
class _Qualification:
    certificate: _Certificate
    _token: object
    startup_validation_performed: bool = True
    digest: str = "fixture-not-a-production-qualification"

    def __post_init__(self):
        if self.digest != self.certificate.digest:
            raise ValueError("fixture qualification digest mismatch")


@dataclass(frozen=True)
class _Codes:
    tensor: object
    sampling_seed: int

    def private_payload(self):
        return {"tensor": self.tensor, "sampling_seed": self.sampling_seed}


class _Sampler:
    def __init__(self):
        self.config = _Config()
        self.device = "cpu-fixture"
        self.calls = []
        self.results = []

    def sample(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        result = _Codes(object(), kwargs["seed"])
        self.results.append(result)
        return result


class _Service:
    def __init__(self):
        self.sampler = _Sampler()
        self.decoder = object()
        self.archived_cases = object()
        self.calibrated = object()
        self.config = object()


def _fixture():
    token = object()
    types = adapter._PinnedTypes(
        _Service, _Sampler, _Config, _Codes, _Qualification, _Certificate,
        token, _Sampler.sample,
    )
    return _Service(), _Qualification(_Certificate(), token), types


def test_views_add_only_labels_preserve_arguments_fresh_results_and_payloads():
    raw, qualification, types = _fixture()
    view = adapter._adapt_service(raw, qualification, types)
    sentinel = object()
    assert view.sampler.config.sampling_mode == "local_edit"
    assert not hasattr(raw.sampler.config, "sampling_mode")
    assert view.sampler.config.kinetic_sampling_steps == 32
    assert view.sampler.config.potential_sampling_steps == 32
    assert view.sampler.config.sampling_sigma_max == .1
    for name in ("decoder", "archived_cases", "calibrated", "config"):
        assert getattr(view, name) is getattr(raw, name)
    assert view.sampler.device == raw.sampler.device
    for seed in (181, 192):
        result = view.sampler.sample(1, seed=seed, generator_mode="base",
                                     kinetic_guidance=sentinel, joint_refinement=sentinel)
        original = raw.sampler.results[-1]
        assert result.solution_sampling_mode == "local_edit"
        assert not hasattr(original, "solution_sampling_mode")
        assert result.tensor is original.tensor
        assert result.sampling_seed == seed
        assert result.private_payload() == original.private_payload()
        assert raw.sampler.calls[-1] == ((1,), dict(seed=seed, generator_mode="base",
            kinetic_guidance=sentinel, joint_refinement=sentinel))
        assert raw.sampler.calls[-1][1]["kinetic_guidance"] is sentinel
    assert len(raw.sampler.calls) == 2
    assert raw.sampler.results[0] is not raw.sampler.results[1]
    for target, name in ((view, "sampler"), (view.sampler.config, "sampling_mode"),
                         (result, "solution_sampling_mode")):
        with pytest.raises(FrozenInstanceError):
            setattr(target, name, "direct_noise")


@pytest.mark.parametrize("part", ["service", "sampler", "config", "qualification", "certificate"])
def test_exact_types_are_required_not_strings_or_lookalikes(part):
    service, qualification, types = _fixture()
    if part == "service":
        service = SimpleNamespace(**vars(service))
    elif part == "sampler":
        service.sampler = SimpleNamespace(**vars(service.sampler), sample=service.sampler.sample)
    elif part == "config":
        service.sampler.config = SimpleNamespace(**vars(service.sampler.config))
    elif part == "qualification":
        qualification = SimpleNamespace(**vars(qualification))
    else:
        object.__setattr__(qualification, "certificate", SimpleNamespace(**vars(qualification.certificate)))
    with pytest.raises((TypeError, ValueError), match="legacy GKW"):
        adapter._adapt_service(service, qualification, types)


@pytest.mark.parametrize("change", ["token", "startup", "physical_pin", "qualification_pin", "digest"])
def test_qualification_authority_and_binding_are_required(change):
    service, qualification, types = _fixture()
    if change == "token":
        object.__setattr__(qualification, "_token", object())
    elif change == "startup":
        object.__setattr__(qualification, "startup_validation_performed", False)
    elif change == "digest":
        object.__setattr__(qualification, "digest", "different")
    else:
        name = "physical_git_commit" if change == "physical_pin" else "qualification_code_commit"
        object.__setattr__(qualification.certificate, name, "f" * 40)
    with pytest.raises(ValueError, match="qualification"):
        adapter._adapt_service(service, qualification, types)


@pytest.mark.parametrize("value", ["direct_noise", None, "", True])
def test_conflicting_config_mode_is_rejected(value):
    service, qualification, types = _fixture()
    object.__setattr__(service.sampler.config, "sampling_mode", value)
    with pytest.raises(ValueError, match="sampling_mode conflicts"):
        adapter._adapt_service(service, qualification, types)


@pytest.mark.parametrize("result_kind", ["conflicting", "lookalike", "subclass", "exception"])
def test_result_mode_identity_and_original_exceptions_are_not_hidden(result_kind):
    service, qualification, types = _fixture()
    error = RuntimeError("original physical sampling failure")

    def sample(self, *args, **kwargs):
        if result_kind == "exception":
            raise error
        if result_kind == "lookalike":
            return SimpleNamespace(tensor=object(), sampling_seed=12)
        if result_kind == "subclass":
            class _OtherCodes(_Codes):
                pass
            return _OtherCodes(object(), 12)
        value = _Codes(object(), 12)
        object.__setattr__(value, "solution_sampling_mode", "direct_noise")
        return value

    service.sampler.sample = MethodType(sample, service.sampler)
    view = adapter._adapt_service(service, qualification, replace(types, sample=sample))
    with pytest.raises((TypeError, ValueError, RuntimeError)) as caught:
        view.sampler.sample(1, seed=12)
    if result_kind == "exception":
        assert caught.value is error


@pytest.mark.parametrize("change", ["method", "config", "mode"])
def test_changed_sampler_after_binding_is_rejected(change):
    service, qualification, types = _fixture()
    view = adapter._adapt_service(service, qualification, types)
    if change == "method":
        service.sampler.sample = lambda *a, **k: _Codes(object(), 1)
    elif change == "config":
        service.sampler.config = _Config()
    else:
        object.__setattr__(service.sampler.config, "sampling_mode", "direct_noise")
    with pytest.raises((TypeError, ValueError), match="legacy GKW"):
        view.sampler.sample(1, seed=1)
    assert not service.sampler.calls


def test_public_factory_requires_verified_root_before_adapting(monkeypatch, tmp_path):
    service, qualification, types = _fixture()
    calls = []

    def verified(root, revision):
        calls.append((root, revision))
        return types

    monkeypatch.setattr(adapter, "_verified_pinned_types", verified)
    result = adapter.adapt_qualified_80632f_local_edit_service(
        service, physical_root=tmp_path, physical_revision=PIN,
        production_qualification=qualification,
    )
    assert calls == [(tmp_path.resolve(), PIN)]
    assert result.sampler.config.sampling_mode == "local_edit"
    with pytest.raises(TypeError, match="qualification"):
        adapter.adapt_qualified_80632f_local_edit_service(
            service, physical_root=tmp_path, physical_revision=PIN,
            production_qualification=PIN,
        )


@pytest.mark.parametrize("revision,head,dirty", [("unknown", PIN, ""), (PIN, "a" * 40, ""),
                                                (PIN, PIN, " M changed.py")])
def test_verification_rejects_unknown_revision_or_nonpinned_checkout(monkeypatch, tmp_path,
                                                                  revision, head, dirty):
    responses = iter((head, dirty))
    monkeypatch.setattr(adapter.subprocess, "check_output", lambda *a, **k: next(responses))
    with pytest.raises(ValueError, match="physical"):
        adapter._verified_pinned_types(tmp_path, revision)


def test_imported_module_from_another_root_is_rejected(monkeypatch, tmp_path):
    responses = iter((PIN, ""))
    monkeypatch.setattr(adapter.subprocess, "check_output", lambda *a, **k: next(responses))
    monkeypatch.setattr(adapter.importlib, "import_module",
                        lambda name: SimpleNamespace(__file__="/another/checkout/generator.py"))
    with pytest.raises(ValueError, match="outside the verified physical root"):
        adapter._verified_pinned_types(tmp_path, PIN)


@dataclass(frozen=True)
class _Observation:
    case: object
    state: object


@dataclass(frozen=True)
class _Resident:
    case: object
    state: object
    base_state: object
    qualification_provenance: object


def _bridge_fixture():
    from pde_agent_verifier_solver.plugins.gkw.selected_torch_runtime import _SelectedSymbols
    service, qualification, types = _fixture()
    calls = []

    def capture(observation, parent=None):
        calls.append((observation, parent))
        base = observation.state if parent is None else parent.base_state
        result = _Resident(observation.case, observation.state, base, {"passed": True})
        calls.append(result)
        return result

    expected = {name: object() for name in adapter._SYMBOL_BINDINGS}
    expected["capture_resident_answer"] = capture
    types = replace(types, symbols=expected, observation=_Observation, resident=_Resident)
    return adapter._adapt_service(service, qualification, types), qualification, _SelectedSymbols(**expected), calls


def test_bound_symbol_adapter_preserves_all_other_symbols_and_raw_resident_identity():
    service, qualification, original, calls = _bridge_fixture()
    symbols = adapter.bind_qualified_80632f_symbols(service, original, production_qualification=qualification)
    for name in adapter._SYMBOL_BINDINGS:
        if name != "capture_resident_answer":
            assert getattr(symbols, name) is getattr(original, name)
    observation = _Observation(object(), object())
    parent = symbols.capture_resident_answer(observation, solution_sampling_mode="local_edit")
    assert parent is calls[-1]
    assert calls[-2] == (observation, None)
    assert parent.state is parent.base_state is observation.state
    child_observation = _Observation(object(), object())
    child = symbols.capture_resident_answer(child_observation, parent=parent)
    assert child is calls[-1]
    assert calls[-2] == (child_observation, parent)
    assert child.state is child_observation.state
    assert child.base_state is parent.base_state
    assert not hasattr(parent, "solution_sampling_mode")  # no physical metadata rewrite


@pytest.mark.parametrize("field", tuple(adapter._SYMBOL_BINDINGS))
def test_any_changed_physical_symbol_is_rejected(field):
    service, qualification, symbols, _ = _bridge_fixture()
    with pytest.raises(TypeError, match="symbol identity changed"):
        adapter.bind_qualified_80632f_symbols(service, replace(symbols, **{field: object()}),
                                             production_qualification=qualification)


@pytest.mark.parametrize("kind", ["no_mode", "direct_noise", "unknown_kwarg", "unknown_observation", "unknown_parent", "perturbed_parent"])
def test_capture_never_drops_unrecognized_modes_arguments_or_lineage(kind):
    service, qualification, original, calls = _bridge_fixture()
    capture = adapter.bind_qualified_80632f_symbols(
        service, original, production_qualification=qualification).capture_resident_answer
    observation = _Observation(object(), object())
    kwargs = {"solution_sampling_mode": "local_edit"}
    if kind == "no_mode":
        kwargs.clear()
    elif kind == "direct_noise":
        kwargs["solution_sampling_mode"] = "direct_noise"
    elif kind == "unknown_kwarg":
        kwargs["skip_audit"] = True
    elif kind == "unknown_observation":
        observation = SimpleNamespace(**vars(observation))
    elif kind == "unknown_parent":
        kwargs["parent"] = object()
    else:
        kwargs["parent"] = _Resident(object(), object(), object(), {"passed": True})
    with pytest.raises((ValueError, TypeError)):
        capture(observation, **kwargs)
    assert not calls


def test_symbols_cannot_be_bound_with_a_string_token_or_direct_noise_authority():
    service, qualification, symbols, _ = _bridge_fixture()
    with pytest.raises(TypeError, match="already-qualified"):
        adapter.bind_qualified_80632f_symbols(service, symbols, production_qualification=PIN)
    with pytest.raises(TypeError, match="direct-noise authority"):
        adapter.bind_qualified_80632f_symbols(service,
            replace(symbols, validated_direct_noise_compatibility=object()),
            production_qualification=qualification)


def test_real_pinned_dataclasses_with_cpu_stub_preserve_torch_tensor_identity():
    """Optional local fixture: real imported types, but no service ctor/forward.

    Subprocess isolation keeps the optional physical namespace out of other test
    fixtures. Its synthetic authority is injected only at the private test seam;
    this is not a real-data or production-qualification validation result.
    """
    project = Path(__file__).resolve().parents[1]
    config = json.loads((project / "configs/plasma_canonical_gkw.json").read_text())["factory_settings"]
    try:
        available = (Path(config["physical_root"]).is_dir()
                     and Path(config["python_dependency_root"]).is_dir())
    except PermissionError:
        pytest.skip("optional pinned physical checkout is not readable by this account")
    if not available:
        pytest.skip("pinned physical checkout/dependency bundle is not installed")
    code = r'''
import importlib, json, sys
from dataclasses import replace
from pathlib import Path
from types import MethodType
import torch
torch.set_num_threads(1)
def forbidden(*args, **kwargs):
    raise AssertionError("fixture must not load or run trained models")
torch.load = forbidden
torch.nn.Module._call_impl = forbidden
project = Path(sys.argv[1])
settings = json.loads((project / "configs/plasma_canonical_gkw.json").read_text())["factory_settings"]
sys.path[:0] = [str(project / "src"), str(project), str(Path(settings["physical_root"]) / "src"),
               settings["python_dependency_root"], settings["external_root"], settings["gyaradax_root"]]
from pde_agent_verifier_solver.plugins.gkw import legacy_local_edit as adapter
from tests.test_gkw_legacy_local_edit import _fixture
types = adapter._verified_pinned_types(Path(settings["physical_root"]), adapter.LEGACY_LOCAL_EDIT_PHYSICAL_REVISION)
_, qualification, fake_types = _fixture()
types = replace(types, qualification=fake_types.qualification, certificate=fake_types.certificate,
                qualification_token=fake_types.qualification_token)
paths = settings["artifacts"]
config = types.config(kinetic_edm_checkpoint=Path(paths["kinetic_edm"]),
    kinetic_compact_corpus=Path(paths["kinetic_corpus"]),
    potential_edm_checkpoint=Path(paths["potential_edm"]), potential_corpus=Path(paths["potential_corpus"]))
sample = types.codes(kinetic_code=torch.zeros(1, 2), global_features=torch.ones(1, 3),
    potential_code=torch.zeros(1, 2), potential_conditioning=torch.ones(1, 2),
    sampled_public_conditions=torch.ones(1, 4), anchor_indices=torch.zeros(1, dtype=torch.long),
    potential_corpus_indices=torch.zeros(1, dtype=torch.long), anchor_keys=({"trajectory": 0},),
    sampling_seed=37, generator_mode="base", kinetic_edit_relative_l2=torch.zeros(1),
    potential_edit_relative_l2=torch.zeros(1))
calls = []
def stub_sample(self, *args, **kwargs):
    calls.append((args, kwargs))
    return sample
sampler = object.__new__(types.sampler)
sampler.config, sampler.device = config, "cpu"
sampler.sample = MethodType(stub_sample, sampler)
service = object.__new__(types.service)
service.sampler, service.decoder = sampler, object()
view = adapter._adapt_service(service, qualification, replace(types, sample=stub_sample))
rng_before = torch.get_rng_state().clone()
result = view.sampler.sample(1, seed=37, generator_mode="base")
assert torch.equal(rng_before, torch.get_rng_state())
assert result.solution_sampling_mode == view.sampler.config.sampling_mode == "local_edit"
for field in sample.__dataclass_fields__:
    assert getattr(result, field) is getattr(sample, field)
assert result.private_payload()["sampling_seed"] == 37
assert not hasattr(sample, "solution_sampling_mode") and not hasattr(config, "sampling_mode")
assert calls == [((1,), {"seed": 37, "generator_mode": "base"})]
assert view.decoder is service.decoder
print("real-pinned-types/cpu-stub/identity-only: passed")

# Audit every canonical direct symbol call against the real pin's signature.
# Missing dynamically-spread keywords are checked explicitly below, including
# the qualified perturbation builder alias and clean capture's mode metadata.
import ast, inspect
from threading import Lock
from types import SimpleNamespace
import numpy as np
from pde_agent_verifier_solver.plugins.gkw import selected_torch_runtime as canonical
raw_symbols = canonical._load_selected_symbols()
tree = ast.parse(Path(canonical.__file__).read_text())
checked = set()
for node in ast.walk(tree):
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        continue
    owner = node.func.value
    if not (isinstance(owner, ast.Attribute) and owner.attr == "_symbols"
            and isinstance(owner.value, ast.Name) and owner.value.id == "self"):
        continue
    name = node.func.attr
    values = {kw.arg: object() for kw in node.keywords if kw.arg is not None}
    inspect.signature(getattr(raw_symbols, name)).bind(*[object() for _ in node.args], **values)
    checked.add(name)
assert checked == {"oracle_limits", "prepare_cases", "streaming_bridge", "oracle_protocol",
                  "capture_resident_answer", "build_macro_truth", "build_perturbation_macro_truth",
                  "build_oracle_backend_from_resident"}
inspect.signature(raw_symbols.qualification_clean_retry_seed).bind(seed=1, attempt=0)
inspect.signature(raw_symbols.build_qualified_perturbed_observation).bind(
    object(), case=object(), qualification=object(), seed=1, device="cpu", private_stage_timings={})
inspect.signature(raw_symbols.retention_policy.audit_only).bind()
try:
    inspect.signature(raw_symbols.capture_resident_answer).bind(object(), parent=None,
                                                               solution_sampling_mode="local_edit")
except TypeError:
    pass  # The sole old bridge ABI gap this adapter explicitly repairs.
else:
    raise AssertionError("fixture no longer represents the pinned pre-mode capture ABI")

# Use the physical repository's tiny synthetic state/operator fixture, not its
# trained assets. The authority below is test-only construction, NOT environment
# validation or scientific qualification evidence. Physical source files stay
# untouched; a tiny linear operator replaces only expensive operator setup.
physical_test_path = Path(settings["physical_root"]) / "tests/test_fusion_gkw_smooth_perturbation.py"
spec = importlib.util.spec_from_file_location("_gkw_cpu_bridge_fixture", physical_test_path)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
bridge = fixture.gkw_generated_pde_bridge
qm = fixture.qualification_module
perturbed, arrays = fixture._fixture()
base = perturbed.base_state
arrays = {**arrays, "s": base.nodes["s"].numpy()}
from dataclasses import asdict
dimensions = asdict(base.contract.grid)
for name, axis in base.contract._VECTOR_SHAPES.items():
    arrays.setdefault(name, np.ones(dimensions[axis], dtype=np.float64))
for shapes in (base.contract._MATRIX_SHAPES, base.contract._TOPOLOGY_SHAPES):
    for name, shape in shapes.items():
        arrays.setdefault(name, np.ones(tuple(dimensions[x] if isinstance(x, str) else x
                                             for x in shape), dtype=np.float64))
scalars = {name: 1. for name in base.contract._REQUIRED_SCALARS}
scalars.update(fixture._packet_scalars(arrays))
scalars.update({name: getattr(base.contract.regime, name)
                for name in ("eps", "disp_par", "disp_vp", "disp_x", "disp_y")})
scalars["idisp"] = 2.
assert base.contract.audit_public_coefficients(arrays, scalars).passed
rule = fixture.GKWProductionPerturbationRule(perturbation=perturbed.perturbation.config)
payload = json.loads((Path(settings["qualification_root"]) / "production_certificate.json").read_text())
certificate = replace(qm.gkw_qualification_certificate_from_payload(payload), rule=rule,
    scale_rule_sha256=rule.sha256(), qualification_id="synthetic-bridge-test-not-production")
test_authority = qm.GKWValidatedProductionQualification(
    certificate=certificate, generator_manifest_sha256=certificate.generator_manifest_sha256,
    operator_manifest_sha256=certificate.operator_manifest_sha256,
    physical_source_manifest_sha256=certificate.physical_source_manifest_sha256,
    seed_plan_manifest_sha256=certificate.seed_plan_manifest_sha256,
    qualification_report_sha256=certificate.qualification_report_sha256,
    startup_validation_performed=True, _token=qm._VALIDATED_PRODUCTION_QUALIFICATION_TOKEN)
real_types = replace(types, qualification=qm.GKWValidatedProductionQualification,
    certificate=fixture.GKWQualifiedProductionPerturbationConfig,
    qualification_token=qm._VALIDATED_PRODUCTION_QUALIFICATION_TOKEN, sample=stub_sample)
qualified_view = adapter._adapt_service(service, test_authority, real_types)
symbols = adapter.bind_qualified_80632f_symbols(qualified_view, raw_symbols,
                                              production_qualification=test_authority)
inspect.signature(symbols.capture_resident_answer).bind(object(), parent=None,
                                                       solution_sampling_mode="local_edit")
def case(identity):
    return bridge.GKWGeneratedBridgeCase(case_id=identity,
        anchor_key={"trajectory": 7, "start_position": 0, "end_position": 3},
        time_nodes=tuple(float(x) for x in base.nodes["t"]),
        public_packet=bridge.GKWPublicEquationPacket(arrays=arrays, scalars=scalars,
            metadata={"case_id": identity, "hidden_parameterization_public": False,
                      "source_access": "metered_numerical_oracle"}),
        contract=base.contract)
operator = fixture._LinearPhiOperator(base.contract)
operator.device = torch.device("cpu")
observation = bridge.GKWTransientCaseObservation(case=case("cpu-clean"), state=base,
    operator=operator, manufactured=None, source_admissibility=SimpleNamespace(passed=True,
    calibration_id="synthetic-only"), hidden_support_admissibility=None, off_grid=None,
    representation_relative_l2=0., passed=True)
runtime = object.__new__(canonical.GKWSelectedTorchRuntime)
runtime._symbols, runtime.expected_solution_sampling_mode = symbols, "local_edit"
runtime._oracle_limits, runtime._answer_counter, runtime._lock = symbols.oracle_limits(), 0, Lock()
def canonical_capture(obs, parent=None):
    return runtime._from_observation(obs, case_prefix=obs.case.case_id,
        expected_case_id=obs.case.case_id, anchor_keys=(obs.case.anchor_key,),
        sampled_public_condition=tuple(scalars[k] for k in ("rlt_i", "rln_i", "shat", "q")),
        seed=211, perturbed=parent is not None, parent_handle=parent, panel=None,
        cohort="selection", panel_profile="formal_full", solution_sampling_mode="local_edit")
clean_case = canonical_capture(observation)
parent = clean_case.hidden_handle.resident
assert type(parent) is bridge.GKWProductionResidentAnswer
assert parent.state is parent.base_state
assert not hasattr(parent, "solution_sampling_mode")
assert clean_case.hidden_handle.solution_sampling_mode == "local_edit"
bridge.build_offline_qualified_dataset_exact_gkw_operator = lambda *a, **kw: (operator, object())
bridge.build_dataset_exact_gkw_operator = lambda *a, **kw: (operator, object())
child_observation = symbols.build_qualified_perturbed_observation(parent,
    case=case("cpu-child"), qualification=test_authority, seed=211,
    device="cpu", private_stage_timings={})
child_case = canonical_capture(child_observation, clean_case.hidden_handle)
child = child_case.hidden_handle.resident
assert type(child) is bridge.GKWProductionResidentAnswer
assert child.base_state is parent.base_state
assert child.base_storage_identity == parent.base_storage_identity
assert child.qualification_provenance["perturbation"]["qualification_id"] == certificate.qualification_id
assert child_case.audit.perturbation_qualification_id == certificate.qualification_id
times, indices = np.array([.25, .75]), np.array([[0, 0, 0], [1, 1, 1]])
for value in (clean_case, child_case):
    result = value.hidden_handle.completed_macro_evaluator.evaluate_pairs(times, indices)
    assert set(result) == set(canonical.GKW_MACRO_FIELDS)
    assert all(np.asarray(field).shape == (2, 2, 2) and np.isfinite(field).all()
               for field in result.values())
    resident = value.hidden_handle.resident
    backend = symbols.build_oracle_backend_from_resident(resident, device="cpu",
        limits=runtime._oracle_limits, source_frame_cache_entries=2)
    oracle = backend.open_session(limits=runtime._oracle_limits)
    phase_indices = np.zeros((1, 5), dtype=np.int64)
    assert np.isfinite(oracle.query_initial_h(phase_indices)).all()
    assert np.isfinite(oracle.query_initial_phi(indices)).all()
    assert np.isfinite(oracle.query_source(.5, phase_indices)).all()
    oracle.close()
    backend.close()
print("full-pinned-symbols/clean-child-macro-oracle/cpu-only: passed")
'''
    result = subprocess.run([sys.executable, "-B", "-c", code, str(project)],
                            text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "real-pinned-types/cpu-stub/identity-only: passed" in result.stdout
    assert "full-pinned-symbols/clean-child-macro-oracle/cpu-only: passed" in result.stdout
