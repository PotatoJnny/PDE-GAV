"""Small codec tests; real D5 cross-process GPU qualification is opt-in.

The real test constructs a frozen zero-coefficient reward model using the
production class. It tests restoration/sampling, not Guidance model quality.
"""
from dataclasses import dataclass, replace
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import RLock
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pde_agent_verifier_solver.engine.checkpoint_store import CheckpointStore
from pde_agent_verifier_solver.plugins.vp_1d1v import resume
from pde_agent_verifier_solver.plugins.vp_1d1v.generator import ArtemisBackend, ArtemisGenerator
from pde_agent_verifier_solver.sdk import Case, CaseRecord, CaseSource, CaseVisibility
from pde_agent_verifier_solver.sdk.macro_loop import CohortKind


@dataclass(frozen=True)
class _Options:
    factorized_jvp_mode: str = "forward"
    exact_query_execution_mode: str = "trusted"
    exact_query_kernel_mode: str = "vp_specialized"


@dataclass(frozen=True)
class _Config:
    count: int = 2


@dataclass(frozen=True)
class _Truth:
    fields: dict
    t: tuple
    x: tuple
    continuous_answer: object = None


@dataclass(frozen=True)
class _Audit:
    state_sha256: str
    seed: int


@dataclass(frozen=True)
class _Phase:
    translation: float


@dataclass(frozen=True)
class _Handle:
    runtime_token: object
    state_sha256: str
    field: object
    curvatures: object
    spatial_phase: object


@dataclass(frozen=True)
class _Manufactured:
    case_id: str
    public_data: dict
    truth: object
    hidden_handle: object
    audit: object


@dataclass(frozen=True)
class _IssuedCase:
    case: object
    truth: object
    manufactured: object
    cohort: object


class _Field:
    def __init__(self, model, condition_scores, latent, **kwargs):
        self._condition_scores = torch.as_tensor(condition_scores).clone()
        self._latent = torch.as_tensor(latent).clone()
        self.dtype, self.device = torch.float64, torch.device("cpu")


class _Runtime:
    def __init__(self):
        self.config, self.oracle_options = _Config(), _Options()
        self.conditions = ("condition",)
        self.artifact = SimpleNamespace(model=object())
        self.sensor_candidate = SimpleNamespace(artifact=SimpleNamespace(sha256="sensor"))
        self._runtime_token = object()
        self.materializations = 0

    def _exact_query_handle(self, manufactured):
        return manufactured.hidden_handle

    def _materialize_parent_from_field(self, **kwargs):
        self.materializations += 1
        return SimpleNamespace(case_id="case-a", **kwargs)

    def _manufactured_case(self, handle, parent):
        return handle


class _Source:
    artifact_sha256 = "a" * 64
    supports_perturbations = True
    public_oracle_mode = "exact"
    generator_id = ArtemisGenerator.info.program_id
    maximum_equation_relative_residual = 0.0
    maximum_poisson_relative_residual = 0.0
    maximum_source_cross_backend_relative_l2 = 0.0

    def __init__(self):
        self.runtime = _Runtime()
        self._lock = RLock()
        self._issued_records, self._roots, self._issued_case_ids = {}, {}, set()

    def sample(self, *args):
        raise AssertionError("codec must not sample")

    public_session_factory = sample
    development_solution_session_factory = sample

    def release_cases(self, ids):
        for key in ids:
            self._issued_records.pop(key, None)
            self._roots.pop(key, None)


class _RewardModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.weight = torch.nn.Parameter(torch.randn(2, dtype=torch.float64))


@dataclass(frozen=True)
class _Normalization:
    scale: float

    def to_dict(self):
        return {"scale": self.scale}

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


@dataclass(frozen=True)
class _Coordinate:
    condition: str
    numerator: int


def _fake_guide(generator):
    from pde_agent_verifier_solver.plugins.vp_1d1v.guidance import ArtemisRewardGuidance
    guide = object.__new__(ArtemisRewardGuidance)
    guide.config = SimpleNamespace(sha256="config")
    guide.api = SimpleNamespace(ArtemisRewardModel=_RewardModel, RewardNormalization=_Normalization)
    guide.runtime_api = SimpleNamespace(ArtemisCompactV5GuidanceCoordinateCapsule=_Coordinate)
    guide.device = "cpu"
    guide._audit_lock = RLock()
    guide._coordinates = {}
    guide.last_sampling_report = {}
    guide.training_audits = []
    guide.sampling_audits = []
    generator.backend.guidance_adapter = guide
    return guide


@pytest.fixture
def fake_module(monkeypatch):
    module = SimpleNamespace(VP1DNativeMacroTruth=_Truth, VP1DManufacturingAudit=_Audit,
        VP1DManufacturedCase=_Manufactured, ArtemisCompactFactorizedExactFieldV4=_Field,
        _ExactQueryHandle=_Handle, _PeriodicSpatialPhase=_Phase)
    monkeypatch.setattr(resume, "_module", lambda backend: module)
    return module


def _fake_generator():
    return ArtemisGenerator(ArtemisBackend(_Source(), physical_revision="b" * 40))


def _fake_records(generator):
    source = generator.backend.source
    truth = _Truth({"M0": (1.0, 2.0)}, (0.0,), (0.0, 1.0))
    case = Case("case-a", CaseSource.GENERATED, CaseVisibility.HIDDEN,
                {"arrays": {"t": np.array([0.0]), "x": np.array([0.0, 1.0])}}, 1)
    field = _Field(None, torch.tensor([1.0], dtype=torch.float64), torch.tensor([2.0], dtype=torch.float64))
    handle = _Handle(source.runtime._runtime_token, "state-a", field, None, _Phase(0.25))
    manufactured = _Manufactured(case.case_id, case.public_data, truth, handle, _Audit("state-a", 11))
    record = CaseRecord(case, truth)
    source._issued_records[case.case_id] = _IssuedCase(case, truth, manufactured, CohortKind.SELECTION)
    source._issued_case_ids.update((case.case_id, "previously-released"))
    source._roots[case.case_id] = resume._DeferredRoot(source.runtime, record, manufactured, "condition")
    return record


def test_checkpoint_roundtrip_does_not_materialize_or_pickle_live_pool(tmp_path, fake_module):
    generator = _fake_generator()
    record = _fake_records(generator)
    generator.backend._parallel_oracle_pool = RLock()  # live pool is not serialized
    state = generator.load(generator.backend.checkpoint(), {})
    store = CheckpointStore(tmp_path, identity="test", generator=generator)
    torch.manual_seed(431)
    store.save({"state": state, "record": record})
    expected_rng = torch.rand(3)
    restored = _fake_generator()
    actual = CheckpointStore(tmp_path, identity="test", generator=restored).load()
    assert torch.equal(torch.rand(3), expected_rng)
    assert actual["state"].base == state.base
    np.testing.assert_array_equal(actual["record"].case.public_data["arrays"]["x"], [0, 1])
    assert actual["record"].truth.fields == record.truth.fields
    source = restored.backend.source
    assert source._issued_case_ids == {"case-a", "previously-released"}
    assert source.runtime.materializations == 0
    assert source._roots["case-a"].manufactured.case_id == "case-a"
    assert source._roots["case-a"].manufactured.case_id == "case-a"
    assert source.runtime.materializations == 1


def test_export_reuses_immutable_assets_and_released_truth(tmp_path, fake_module, monkeypatch):
    generator = _fake_generator()
    record = _fake_records(generator)
    runtime = generator.export_runtime_state(tmp_path)
    saved = generator.export_case_record(record, tmp_path)
    before = {path.name: path.stat().st_mtime_ns for path in tmp_path.iterdir()}
    monkeypatch.setattr(resume, "_put", lambda *args: pytest.fail("immutable data rewritten"))
    assert generator.export_runtime_state(tmp_path) == runtime
    assert generator.export_case_record(record, tmp_path) == saved
    assert before == {path.name: path.stat().st_mtime_ns for path in tmp_path.iterdir()}
    # Old round results can retain records even after the resident is released.
    restored = _fake_generator()
    restored.restore_runtime_state(runtime, tmp_path)
    # A resumed process also reuses verified objects at its first checkpoint.
    assert restored.export_runtime_state(tmp_path) == runtime
    restored.backend.source.release_cases((record.case.case_id,))
    assert restored.restore_case_record(saved, tmp_path).truth.fields == record.truth.fields


def test_frozen_guidance_weights_identity_feedback_and_rng_roundtrip(tmp_path, fake_module):
    from pde_agent_verifier_solver.plugins.vp_1d1v.guidance import ArtemisRewardState
    generator = _fake_generator()
    guide = _fake_guide(generator)
    record = _fake_records(generator)
    guide._coordinates[record.case.case_id] = _Coordinate("condition", 123456)
    guide.training_audits = [{"target_solver_id": "winner-r1", "model_reused": False}]
    model = _RewardModel(guide.config).eval()
    model.weight.requires_grad_(False)
    reward = ArtemisRewardState(generator.backend.source.artifact_sha256, "d" * 64,
                               "winner-r1", model, _Normalization(1.3), {"status": "frozen"})
    state = replace(generator.load(generator.backend.checkpoint(), {}), guidance=reward)
    store = CheckpointStore(tmp_path, identity="guide", generator=generator)
    store.save({"state": state, "record": record})
    expected_rng = torch.rand(4)
    restored = _fake_generator()
    _fake_guide(restored)
    value = CheckpointStore(tmp_path, identity="guide", generator=restored).load()
    actual = value["state"].guidance
    assert actual.snapshot_sha256 == reward.snapshot_sha256
    assert actual.target_solver_id == reward.target_solver_id
    assert torch.equal(actual.model.weight, model.weight)
    assert not actual.model.weight.requires_grad and not actual.model.training
    assert actual.normalization == reward.normalization
    assert restored.backend.guidance_adapter._coordinates == guide._coordinates
    assert restored.backend.guidance_adapter.training_audits == guide.training_audits
    assert torch.equal(torch.rand(4), expected_rng)


@pytest.mark.parametrize("failure", ["identity", "asset", "nonempty", "duplicate", "record_binding"])
def test_checkpoint_rejects_drift_and_inconsistent_state(tmp_path, fake_module, failure):
    generator = _fake_generator()
    record = _fake_records(generator)
    payload = generator.export_runtime_state(tmp_path)
    target = _fake_generator()
    if failure == "identity":
        payload["identity"]["checkpoint"]["physical_revision"] = "different"
    elif failure == "asset":
        path = tmp_path / payload["residents"][0]["manufactured"]["asset"]
        path.write_bytes(b"broken")
    elif failure == "nonempty":
        target.backend.source._issued_case_ids.add("old")
    elif failure == "duplicate":
        payload["residents"].append(dict(payload["residents"][0]))
    else:
        target.restore_runtime_state(payload, tmp_path)
        saved = generator.export_case_record(record, tmp_path)
        other = replace(record, truth=replace(record.truth, fields={"M0": (3.0, 4.0)}))
        saved["truth"] = generator.export_case_record(other, tmp_path)["truth"]
        with pytest.raises(ValueError, match="differs from restored resident"):
            target.restore_case_record(saved, tmp_path)
        return
    with pytest.raises(ValueError):
        target.restore_runtime_state(payload, tmp_path)


def _request(kind="base", seed=401, anchors=()):
    from pde_agent_verifier_solver.sdk import GenerationKind, GenerationPurpose, GenerationRequest, GeneratorMode
    return GenerationRequest(1, 1, GeneratorMode.ONLINE, GenerationKind(kind), CaseVisibility.HIDDEN,
                             seed, purpose=GenerationPurpose.SELECTION, anchors=anchors)


def _real_generator(device):
    from pde_agent_verifier_solver.plugins.vp_1d1v.product import load_artemis_backend
    settings = json.loads((Path(__file__).resolve().parents[1] / "configs/plasma_artemis_d5_guided.json").read_text())["factory_settings"]
    settings["device"] = device
    binding = load_artemis_backend(**settings)
    return binding.plugin.generator, binding.checkpoint


def _frozen_engineering_guidance(generator, state):
    from pde_agent_verifier_solver.plugins.vp_1d1v.guidance import ArtemisRewardState
    guide = generator.backend.guidance_adapter
    model = guide.api.ArtemisRewardModel(guide.config).to(guide.device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    normalization = guide.api.RewardNormalization(0.0, 1.0)
    guidance = ArtemisRewardState(generator.backend.source.artifact_sha256, "e" * 64,
        "frozen-engineering-fixture", model, normalization,
        {"qualification": "resume_engineering_fixture_not_guidance_quality"})
    return replace(state, guidance=guidance)


def _real_snapshot(generator, state, record):
    from pde_agent_verifier_solver.sdk import AnswerQuery, CaseAccessKind, QueryKind
    arrays = record.case.public_data["arrays"]
    time, position, velocity = (np.asarray(arrays[name]) for name in ("t", "x", "v"))
    points = np.column_stack((np.resize(time, 32), np.resize(position, 32), np.resize(velocity[::32], 32)))
    session = generator.open_case_session(state, record.case, CaseAccessKind.PUBLIC_PROBLEM)
    try:
        source = np.asarray(session.query_source(points))
        initial = np.asarray(session.query_initial_distribution(points[:, 1:]))
    finally:
        session.close()
    query = AnswerQuery(QueryKind.POINTS, ("M0", "M1", "M2", "M3", "E"),
                        points=tuple({"t": float(t), "x": float(x)} for t, x in points[:, :2]))
    truth = record.truth.answer(query)
    return {"case_id": record.case.case_id, "parent_case_id": record.case.parent_case_id,
        "points": points.tolist(), "source": source.tolist(), "initial": initial.tolist(),
        "truth": {key: list(value) for key, value in truth.fields.items()},
        "public_array_sha256": {key: sha256(np.asarray(value).tobytes()).hexdigest() for key, value in arrays.items()}}


def _future_records(generator, state, pending):
    base = generator.sample(state, _request(seed=801))[0]
    guided = generator.sample(state, _request("guided", seed=901))[0]
    child = generator.sample(state, _request("perturbed", seed=1001, anchors=(pending.case,)))[0]
    return (base, guided, child)


def _real_child(root, device):
    torch.set_num_threads(8)
    root = Path(root)
    generator, _ = _real_generator(device)
    value = CheckpointStore(root / "store", identity="artemis-real-resume", generator=generator).load()
    rng = {"cpu": torch.rand(4).tolist(), "cuda": torch.rand(4, device="cuda:0").cpu().tolist()}
    state = value["state"]
    snapshots = [_real_snapshot(generator, state, record) for record in value["records"]]
    pending = value["pending"]
    future = [_real_snapshot(generator, state, record) for record in _future_records(generator, state, pending)]
    source, guide = generator.backend.source, generator.backend.guidance_adapter
    result = {"records": snapshots, "future": future, "rng": rng,
              "snapshot_sha256": state.guidance.snapshot_sha256,
              "weights": {key: tensor.detach().cpu().tolist() for key, tensor in state.guidance.model.state_dict().items()},
              "issued_ids": sorted(source._issued_case_ids),
              "coordinates": {key: asdict_coordinate(value) for key, value in guide._coordinates.items()},
              "device": str(next(state.guidance.model.parameters()).device)}
    (root / "restored.json").write_text(json.dumps(result, allow_nan=False))


def asdict_coordinate(value):
    from dataclasses import asdict
    return asdict(value)


@pytest.mark.skipif(os.environ.get("ARTEMIS_RESUME_REAL_GPU") != "1", reason="opt-in real D5 GPU checkpoint qualification")
def test_real_d5_checkpoint_in_fresh_process(tmp_path):
    assert os.environ.get("SLURM_JOB_ID") and torch.cuda.is_available(), "requires a GPU compute allocation"
    torch.set_num_threads(8)
    generator, checkpoint = _real_generator("cuda:0")
    state = _frozen_engineering_guidance(generator, generator.load(checkpoint, {}))
    base = generator.sample(state, _request())[0]
    child = generator.sample(state, _request("perturbed", seed=501, anchors=(base.case,)))[0]
    pending = generator.sample(state, _request(seed=601))[0]
    records = (base, child)
    before = [_real_snapshot(generator, state, record) for record in records]
    store = CheckpointStore(tmp_path / "store", identity="artemis-real-resume", generator=generator)
    store.save({"state": state, "records": records, "pending": pending})
    rng = {"cpu": torch.rand(4).tolist(), "cuda": torch.rand(4, device="cuda:0").cpu().tolist()}
    future = [_real_snapshot(generator, state, record) for record in _future_records(generator, state, pending)]
    source, guide = generator.backend.source, generator.backend.guidance_adapter
    expected = {"records": before, "future": future, "rng": rng,
        "snapshot_sha256": state.guidance.snapshot_sha256,
        "weights": {key: tensor.detach().cpu().tolist() for key, tensor in state.guidance.model.state_dict().items()},
        "issued_ids": sorted(source._issued_case_ids),
        "coordinates": {key: asdict_coordinate(value) for key, value in guide._coordinates.items()}}
    (tmp_path / "expected.json").write_text(json.dumps(expected, allow_nan=False))
    target = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    result = subprocess.run([sys.executable, "-c",
        "from tests.test_artemis_resume import _real_child; import sys; _real_child(sys.argv[1], sys.argv[2])",
        str(tmp_path), target], text=True, capture_output=True, timeout=900)
    (tmp_path / "subprocess.stdout").write_text(result.stdout)
    (tmp_path / "subprocess.stderr").write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    actual = json.loads((tmp_path / "restored.json").read_text())
    for key in ("rng", "snapshot_sha256", "weights", "issued_ids", "coordinates"):
        assert actual[key] == expected[key], key
    maximum = 0.0
    for group in ("records", "future"):
        for left, right in zip(expected[group], actual[group], strict=True):
            for key in ("case_id", "parent_case_id", "points", "public_array_sha256", "truth"):
                assert left[key] == right[key], (group, key)
            for key in ("source", "initial"):
                a, b = np.asarray(left[key]), np.asarray(right[key])
                maximum = max(maximum, float(np.max(np.abs(a - b))))
                np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)
    (tmp_path / "resume-qualification.json").write_text(json.dumps({
        "status": "passed", "existing_records": 2, "future_base_guided_perturbed": 3,
        "points_per_source_request": 32, "max_oracle_absolute_error": maximum,
        "restored_device": actual["device"], "logical_device_changed": target != "cuda:0",
        "native_truth_and_public_arrays_exact": True, "guidance_training_performed": False,
        "guidance_quality_claimed": False, "full_iteration_claimed": False,
    }, indent=2))
