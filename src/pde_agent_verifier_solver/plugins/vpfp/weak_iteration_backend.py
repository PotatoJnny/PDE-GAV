"""Thin weak H2/innovation Guidance adapter for the canonical Controller.

The Controller alone owns rounds, Candidates, feedback and promotion. This
module preserves the complete moving-frame public sessions, captures actual
native sampling coordinates, and trains fresh Guides from selected feedback.
Only source evaluation is accelerated; macro/initial fields remain original.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from hashlib import sha256
import io
import json
from pathlib import Path
from threading import RLock

import numpy as np
import torch

from pde_agent_verifier_solver.sdk import (
    Case, CaseRecord, CaseSource, CaseVisibility, GenerationKind,
    GeneratorCheckpoint, GeneratorState, ProgramInfo)
from .innovation_binding import _module_sha
from .moving_frame_backend import (
    _Resident, evaluate_moving_frame_macros, _numpy)
from .moving_frame_binding import MovingFrameGenerator, binding_from_sampler, json_sha, file_sha
from .perturbation_backend import _seeds, _snapshot_vpfp_public_case, _equivalent_vpfp_public_snapshot
from .problem import VPFPContinuousMacroTruth
from .weak_solver_guidance import guidance_payload, load_guidance_payload
from ..plasma_transport_settings import transport_calibration, check_guidance_objective, check_feedback_objective

SCHEMA = "vpfp_weak_canonical_iteration_v1"
FEEDBACK_SELECTION = "first_80_clean_base_feedback_order"
DRAW_TENSORS = ("m", "z", "white_m", "white_z", "condition_m", "conditions", "full_coefficients")


@dataclass
class WeakGuidanceSnapshot:
    heads: dict
    report: dict
    checkpoint_path: object
    checkpoint_sha256: str
    _frozen: object = field(init=False, repr=False)

    def __post_init__(self):
        if set(self.heads) != {"macro", "micro"}:
            raise ValueError("both native weak Guidance heads required")
        self.report = deepcopy(self.report)
        for stage, head in self.heads.items():
            if head.stage != stage:
                raise ValueError("Guide head stage mismatch")
            head.eval().requires_grad_(False)
        self._frozen = self._fingerprint()

    def _fingerprint(self):
        return (json_sha(self.report), str(self.checkpoint_path), self.checkpoint_sha256,
                tuple((stage, _module_sha(head), json_sha(head.identity()),
                       json_sha(head.calibration), float(head.strength)) for stage, head in sorted(self.heads.items())))

    def assert_frozen(self, base):
        base.assert_frozen()
        if self._fingerprint() != self._frozen:
            raise ValueError("frozen weak Guidance snapshot changed")
        for head in self.heads.values():
            if (json_sha(head.base_identity) != json_sha(base.identity) or head.training
                    or any(p.requires_grad for p in head.parameters())):
                raise ValueError("weak Guidance Base binding or frozen mode changed")


class _CapturedSampler:
    def __init__(self, base, draw_function):
        self.base, self.draw_function = base, draw_function
        self.identity = {**deepcopy(base.identity), "actual_draw_capture": True}
        self.pending, self.diagnostics, self.guide = {}, [], None

    def generate(self, seeds):
        # Exactly the native sampler's per-root physical-condition law.
        conditions = []
        for seed in seeds:
            stream = int.from_bytes(sha256(f"vpfp-edm-condition:{seed}".encode()).digest()[:8], "little")
            rng = np.random.default_rng(stream)
            conditions.append([rng.uniform(*self.base.config.collision_nu_range),
                               rng.uniform(*self.base.config.temperature_range)])
        kwargs = {} if self.guide is None else {stage + "_guide": head for stage, head in self.guide.heads.items()}
        draw = self.draw_function(self.base, np.asarray(conditions, dtype=np.float32), seeds, **kwargs)
        if tuple(draw["roots"]) != tuple(seeds) or tuple(draw["equation_seeds"]) != tuple(seeds):
            raise ValueError("native weak draw changed requested root identities")
        if draw.get("captured_original_white_not_reencoded") is not True or draw.get("sampling_steps") != 64:
            raise ValueError("actual native white capture and Heun64 required")
        for index, seed in enumerate(seeds):
            if seed in self.pending:
                raise ValueError("duplicate pending weak root")
            row = {key: draw[key][index:index + 1].detach().cpu().clone() for key in DRAW_TENSORS}
            row.update(roots=[int(seed)], macro_noise_seeds=[int(draw["macro_noise_seeds"][index])],
                micro_noise_seeds=[int(draw["micro_noise_seeds"][index])],
                captured_original_white_not_reencoded=True, sampling_steps=64)
            self.pending[seed] = row
        self.diagnostics = deepcopy(draw.get("guidance_diagnostics", {}))
        return draw


class WeakIterationBackend:
    supports_guidance = True
    supports_perturbations = False

    def __init__(self, backend, capture, *, output_root, trainer, config, initial_guidance=None,
                 perturbation_config=None, metric_profile=None, metric_calibration=None):
        self.backend, self.runtime, self.capture = backend, backend, capture
        self.output_root, self.trainer = Path(output_root), trainer
        self.config = deepcopy(config)
        self._config_sha = json_sha(self.config)
        self.initial_guidance = initial_guidance
        self.score_calibration = transport_calibration(metric_profile, metric_calibration)
        check_guidance_objective(self.score_calibration, initial_guidance, {})
        from .weak_perturbation import perturbation_identity
        self.perturbation_config = perturbation_config
        self.perturbation_identity = (None if perturbation_config is None else perturbation_identity(perturbation_config))
        self.supports_perturbations = perturbation_config is not None
        self._parallel_oracle_pool = None
        self._development_oracle_pool = None
        self._lock = RLock()
        self._draws, self._private_answers, self._restored_records = {}, {}, {}
        self._export_cache, self._restored_cases, self._record_refs = {}, {}, {}
        self.training_audits, self.sampling_audits = [], []
        self.public_session_methods = backend.public_session_methods
        self.development_solution_session_methods = backend.development_solution_session_methods

    def _check(self):
        if json_sha(self.config) != self._config_sha:
            raise ValueError("frozen weak round Guidance configuration changed")
        self.capture.base.assert_frozen()

    def train_base(self, *args, **kwargs):
        raise RuntimeError("accepted weak Base is frozen")

    def public_session_factory(self, case):
        if self._parallel_oracle_pool is not None:
            self.backend._active(case)
            return self._parallel_oracle_pool.session(case)
        if self._development_oracle_pool is not None and case.visibility is CaseVisibility.DEVELOPMENT:
            return self._development_oracle_pool.session(case)
        return self.backend.public_session_factory(case)

    def public_session_operation_lock(self, case):
        pool = self._parallel_oracle_pool
        if pool is None and case.visibility is CaseVisibility.DEVELOPMENT:
            pool = self._development_oracle_pool
        return None if pool is None else pool.operation_lock(case.case_id)

    def development_solution_session_factory(self, case):
        return self.backend.development_solution_session_factory(case)

    def release_cases(self, case_ids):
        with self._lock:
            case_ids = tuple(case_ids)
            if self._development_oracle_pool is not None:
                self._development_oracle_pool.release_cases(case_ids)
            self.backend.release_cases(case_ids)
            for key in case_ids:
                draw = self._draws.pop(key, None)
                if draw is not None:
                    for cache_key in tuple(self._export_cache):
                        if cache_key[1:] == ("draw", id(draw)):
                            self._export_cache.pop(cache_key)

    def generate(self, base_state, guidance_state, request, progress=None):
        with self._lock:
            self._check()
            if self._parallel_oracle_pool is not None:
                raise RuntimeError("weak generation cannot overlap a hidden numerical-worker scope")
            if request.kind is GenerationKind.PERTURBED:
                if self.perturbation_config is None:
                    raise ValueError("weak perturbations unsupported without an explicitly bound configuration")
                if self.capture.pending or self.capture.guide is not None:
                    raise RuntimeError("unfinished weak capture transaction")
                parent_sources = []
                for anchor in request.anchors:
                    draw = self._draws.get(anchor.case_id)
                    if (draw is None or draw["source"] not in ("base", "guided")
                            or anchor.source is not CaseSource.GENERATED or anchor.parent_case_id is not None
                            or anchor.round_index != request.round_index or anchor.visibility is not request.visibility):
                        raise ValueError("weak perturbation requires current-round clean BASE/GUIDED capture anchors")
                    parent_sources.append(draw["source"])
                from .weak_perturbation import generate_perturbed
                records = generate_perturbed(self.backend, request, self.perturbation_config, progress,
                    captured_parent_sources=tuple(parent_sources))
                for record in records:
                    key = record.case.case_id
                    self._private_answers[key] = dict(path=str(self.backend.output_root / key / "private-answer.pt"),
                        sha256=self.backend._resident_cases[key].answer_sha256)
                self.sampling_audits.append(dict(round_index=request.round_index, purpose=request.purpose.value,
                    source="perturbed", count=len(records), case_ids=[row.case.case_id for row in records],
                    parent_case_ids=[row.case.parent_case_id for row in records], root_seeds=list(_seeds(request)),
                    perturbation_identity=deepcopy(self.perturbation_identity), generated_white_coordinates=False))
                self._check()
                return records
            if request.kind not in (GenerationKind.BASE, GenerationKind.GUIDED):
                raise ValueError("weak iteration supports BASE/GUIDED only; perturbations unsupported")
            guided = request.kind is GenerationKind.GUIDED
            if guided:
                if not isinstance(guidance_state, WeakGuidanceSnapshot):
                    raise ValueError("GUIDED requires a trained weak round snapshot")
                check_guidance_objective(self.score_calibration, guidance_state, {})
                guidance_state.assert_frozen(self.capture.base)
                if request.round_index < guidance_state.report["target_round_index"]:
                    raise ValueError("cannot use future-round weak Guidance")
            if self.capture.pending or self.capture.guide is not None:
                raise RuntimeError("unfinished weak capture transaction")
            native_request = request
            if guided:
                seed = int.from_bytes(sha256(f"vpfp-weak-score-guided:{request.seed}".encode()).digest()[:8], "big") % 2**63
                native_request = replace(request, kind=GenerationKind.BASE, seed=seed)
                self.capture.guide = guidance_state
            records = ()
            try:
                seeds = _seeds(native_request)
                records = self.backend.generate(base_state, None, native_request, progress)
                if len(records) != len(seeds) or set(self.capture.pending) != set(seeds):
                    raise ValueError("Case/native capture inventory mismatch")
                for record, seed in zip(records, seeds, strict=True):
                    key = record.case.case_id
                    self._draws[key] = dict(source="guided" if guided else "base", root_seed=seed,
                                            draw=self.capture.pending[seed])
                    self._private_answers[key] = dict(
                        path=str(self.backend.output_root / key / "private-answer.pt"),
                        sha256=self.backend._resident_cases[key].answer_sha256)
                self.sampling_audits.append(dict(round_index=request.round_index, purpose=request.purpose.value,
                    source="guided" if guided else "base", count=len(records),
                    case_ids=[r.case.case_id for r in records], root_seeds=list(seeds),
                    snapshot_sha256=guidance_state.checkpoint_sha256 if guided else None,
                    diagnostics=deepcopy(self.capture.diagnostics)))
                return records
            except BaseException:
                if records:
                    self.release_cases(tuple(r.case.case_id for r in records))
                raise
            finally:
                self.capture.pending.clear(); self.capture.guide = None; self.capture.diagnostics = []
                self._check()
                if guided:
                    guidance_state.assert_frozen(self.capture.base)

    def fit_guidance(self, base_state, previous_guidance, feedback, settings, progress=None):
        del base_state
        with self._lock:
            check_guidance_objective(self.score_calibration, previous_guidance, settings)
            check_feedback_objective(self.score_calibration, feedback)
            self._check()
            if previous_guidance is not None:
                previous_guidance.assert_frozen(self.capture.base)
            solver_sha = getattr(feedback.selected_artifact.program, "tree_sha256", None)
            if not isinstance(solver_sha, str) or len(solver_sha) != 64:
                raise ValueError("selected immutable Candidate bundle SHA required")
            eligible, excluded = [], []
            for index, item in enumerate(feedback.cases):
                key = item.case.case_id
                cached = self._draws.get(key)
                if item.case.source is CaseSource.PERTURBED:
                    excluded.append(dict(case_id=key, reason="not_clean_base")); continue
                if cached is None:
                    raise ValueError("feedback lacks its actual resident native draw")
                if cached["source"] != "base":
                    excluded.append(dict(case_id=key, reason="not_clean_base")); continue
                eligible.append((item, cached, index))
            if (len(eligible) < 80 or len({x[0].case.case_id for x in eligible}) != len(eligible)
                    or len({x[1]["root_seed"] for x in eligible}) != len(eligible)):
                raise ValueError("at least 80 unique actual clean BASE roots required")
            excluded += [dict(case_id=x[0].case.case_id, reason="outside_first80") for x in eligible[80:]]
            selected = eligible[:80]  # Commit identities before reading any errors.
            target = feedback.round_index + 1
            selection = dict(schema=SCHEMA + "_feedback", policy=FEEDBACK_SELECTION,
                source_round_index=feedback.round_index, target_round_index=target,
                target_solver_sha256=solver_sha, selection_uses_rewards=False,
                selected=[dict(case_id=item.case.case_id, root_seed=cached["root_seed"], feedback_index=index,
                    split="fit" if slot < 64 else "validation") for slot, (item, cached, index) in enumerate(selected)],
                excluded=excluded)
            self.output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
            with (self.output_root / f"round-{target:03d}-selection.json").open("x") as stream:
                json.dump(selection, stream, sort_keys=True, allow_nan=False)
            rows = []
            for item, cached, _index in selected:
                error = float(item.primary_error)
                valid = bool(np.isfinite(error) and error >= 0)
                rows.append({**cached, "case_id": item.case.case_id, "error": error if valid else None,
                    "valid": valid, "source_round_index": feedback.round_index, "target_solver_sha256": solver_sha})
            matcher = sha256(b"".join((Path(__file__).parent / name).read_bytes()
                for name in ("macro_matching.py", "tasks/observable.py"))).hexdigest()
            if self.score_calibration is not None:
                matcher = self.score_calibration.objective_hash
            result = self.trainer(base_sampler=self.capture.base, rows=rows,
                source_round_index=feedback.round_index, target_round_index=target,
                target_solver_sha256=solver_sha, reward_matcher_sha256=matcher,
                controller_seed=int(settings["controller_seed"]), output_root=self.output_root / f"round-{target:03d}",
                config=deepcopy(self.config), progress=progress)
            if not isinstance(result, WeakGuidanceSnapshot):
                raise TypeError("trainer must return a frozen WeakGuidanceSnapshot")
            check_guidance_objective(self.score_calibration, result, {})
            if (result.report.get("source_round_index") != feedback.round_index
                    or result.report.get("target_round_index") != target
                    or result.report.get("target_solver_sha256") != solver_sha
                    or result.report.get("execution_passed") is not True):
                raise ValueError("trained weak snapshot belongs to another feedback boundary")
            if previous_guidance is not None and any(result.heads[s] is previous_guidance.heads[s] for s in result.heads):
                raise ValueError("round update reused old Guide weights instead of fresh training")
            result.assert_frozen(self.capture.base)
            self.training_audits.append(dict(selection=selection, checkpoint_path=str(result.checkpoint_path),
                snapshot_sha256=result.checkpoint_sha256, report=deepcopy(result.report),
                **({"scoring_objective": self.score_calibration.objective_identity}
                   if self.score_calibration is not None else {})))
            self._check()
            return result


def _put(directory, payload):
    from pde_agent_verifier_solver.engine.checkpoint_store import atomic_bytes
    stream = io.BytesIO(); torch.save(payload, stream)
    data = stream.getvalue(); digest = sha256(data).hexdigest()
    root = Path(directory); root.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = root / (digest + ".weak.pt")
    if path.is_symlink() or (path.exists() and file_sha(path) != digest):
        raise ValueError("weak continuation object changed")
    if not path.exists():
        atomic_bytes(path, data)
    return dict(name=path.name, sha256=digest)


def _get(directory, descriptor):
    if (set(descriptor) != {"name", "sha256"}
            or not isinstance(descriptor["sha256"], str) or len(descriptor["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in descriptor["sha256"])
            or descriptor["name"] != descriptor["sha256"] + ".weak.pt"):
        raise ValueError("invalid weak continuation descriptor")
    path = Path(directory) / descriptor["name"]
    if path.is_symlink() or file_sha(path) != descriptor["sha256"]:
        raise ValueError("weak continuation object hash differs")
    return torch.load(path, map_location="cpu", weights_only=True)


def _case_payload(case):
    from pde_agent_verifier_solver.engine.common_evaluation import json_value
    public = {key: json_value(value) for key, value in case.public_data.items() if key != "arrays"}
    public["arrays"] = {k: torch.as_tensor(v).detach().cpu().clone() for k, v in case.public_data["arrays"].items()}
    return dict(case_id=case.case_id, source=case.source.value, visibility=case.visibility.value,
        public_data=public, round_index=case.round_index, parent_case_id=case.parent_case_id, tags=case.tags)


def _case(payload):
    data = deepcopy(payload)
    data["source"], data["visibility"] = CaseSource(data["source"]), CaseVisibility(data["visibility"])
    data["public_data"]["arrays"] = {k: v.numpy().copy() for k, v in data["public_data"]["arrays"].items()}
    return Case(**data)


class _StoredEvaluator:
    def __init__(self, runtime, case_id, carrier, condition):
        self.runtime, self.case_id, self.carrier, self.condition = runtime, case_id, carrier, condition

    def __call__(self, case_id, points):
        if case_id != self.case_id:
            raise ValueError("restored continuous truth identity differs")
        chunks = [evaluate_moving_frame_macros(self.runtime.decoder, points[start:start + 32], self.carrier, self.condition)
                  for start in range(0, len(points), 32)]
        return {key: np.concatenate([_numpy(row[key]) for row in chunks])
                for key in ("rho", "momentum", "kinetic_energy", "raw_second_moment")}

    def transport(self, case_id, points):
        from .moving_frame_fast_macros import evaluate_moving_frame_transport_macros
        if case_id != self.case_id:
            raise ValueError("restored continuous truth identity differs")
        chunks = [evaluate_moving_frame_transport_macros(
            self.runtime.decoder, points[start:start + 32], self.carrier, self.condition)
            for start in range(0, len(points), 32)]
        return {key: np.concatenate([_numpy(row[key]) for row in chunks]) for key in chunks[0]}


class WeakIterationGenerator(MovingFrameGenerator):
    info = ProgramInfo("vpfp-weak-canonical-guidance-v1", "VPFP weak guided Generator",
        "Native H2/micro answer-first generation and selected-Solver round Guidance.")

    def load(self, checkpoint, settings, progress=None):
        state = super().load(checkpoint, settings, progress)
        return replace(state, guidance=self.backend.initial_guidance)

    def public_session_operation_lock(self, state, case):
        # The shared evaluator honors this only while this plugin has explicitly
        # activated independent private numerical workers. Otherwise keep its
        # existing global lock; never change the frozen canonical base class.
        return self.backend.public_session_operation_lock(case)

    def _record_data(self, record):
        runtime, key = self.backend.backend, record.case.case_id
        if record.generator_target is not None:
            raise TypeError("weak continuation does not reinterpret Generator training targets")
        if key in runtime._resident_cases:
            row = runtime._resident_cases[key]
            if not _equivalent_vpfp_public_snapshot(row.public_snapshot, record.case):
                raise ValueError("Case snapshot changed before continuation")
            carrier, condition, digest = row.carrier, row.condition, row.answer_sha256
        elif key in self.backend._restored_records:
            saved = self.backend._restored_records[key]
            if not _equivalent_vpfp_public_snapshot(self.backend._restored_cases[key], record.case):
                raise ValueError("restored Case snapshot changed before continuation")
            carrier, condition, digest = saved["carrier"], saved["condition"], saved["answer_sha256"]
        else:
            asset = self.backend._private_answers[key]
            path, digest = Path(asset["path"]), asset["sha256"]
            if path.is_symlink() or file_sha(path) != digest:
                raise ValueError("archived weak answer SHA differs")
            saved = torch.load(path, map_location="cpu", weights_only=True)
            if saved["sampler_identity_sha256"] != runtime.sampler_identity_sha256:
                raise ValueError("archived weak answer Base identity differs")
            carrier = runtime.decoder.carrier_from_payload(saved["carrier"])
            condition = saved["condition"]
            if file_sha(path) != digest:
                raise ValueError("archived weak answer changed during load")
        return dict(case=_case_payload(record.case), carrier={k: v.detach().cpu().clone() for k, v in carrier.items()},
                    condition=condition.detach().cpu().clone(), answer_sha256=digest)

    def export_case_record(self, record, directory):
        key = (str(Path(directory).resolve()), "case", record.case.case_id)
        cached = self.backend._export_cache.get(key)
        if cached is None:
            descriptor = _put(directory, self._record_data(record))
            self.backend._export_cache[key] = (_snapshot_vpfp_public_case(record.case), descriptor)
        else:
            snapshot, descriptor = cached
            if not _equivalent_vpfp_public_snapshot(snapshot, record.case):
                raise ValueError("cached weak Case public data changed")
        return dict(schema=SCHEMA + "_case", binding_sha256=self._identity["binding_sha256"],
                    case_id=record.case.case_id, object=descriptor)

    def restore_case_record(self, payload, directory):
        if payload.get("schema") != SCHEMA + "_case" or payload["binding_sha256"] != self._identity["binding_sha256"]:
            raise ValueError("weak Case continuation binding differs")
        key = payload["case_id"]
        if self.backend._record_refs.get(key) == payload["object"]:
            row, case = self.backend._restored_records[key], self.backend._restored_cases[key]
        else:
            row = _get(directory, payload["object"]); case = _case(row["case"])
            if case.case_id != key or key in self.backend._record_refs:
                raise ValueError("restored weak Case identity/record changed")
            self.backend._restored_records[key], self.backend._restored_cases[key] = row, case
            self.backend._record_refs[key] = payload["object"]
        arrays = case.public_data["arrays"]
        evaluator = _StoredEvaluator(self.backend.backend, case.case_id, row["carrier"], row["condition"])
        truth = VPFPContinuousMacroTruth(case.case_id, tuple(arrays["t"]), tuple(arrays["x"]), tuple(arrays["y"]),
            evaluator, evaluate_transport_points=evaluator.transport)
        return CaseRecord(case, truth)

    def _cached_object(self, directory, kind, value):
        key = (str(Path(directory).resolve()), kind, id(value))
        cached = self.backend._export_cache.get(key)
        if cached is None or cached[0] is not value:
            cached = (value, _put(directory, value))
            self.backend._export_cache[key] = cached
        return cached[1]

    def export_runtime_state(self, directory):
        backend, runtime = self.backend, self.backend.backend
        with backend._lock:
            backend._check()
            if backend.capture.pending or backend.capture.guide is not None:
                raise RuntimeError("weak continuation requires a quiescent capture boundary")
            rows = [self.export_case_record(CaseRecord(row.case, None), directory)
                    for row in runtime._resident_cases.values()]
            value = dict(residents=rows, issued=sorted(runtime._issued),
                draws={key: self._cached_object(directory, "draw", draw) for key, draw in backend._draws.items()},
                private_answers=backend._private_answers, oracle_calls=runtime.oracle_calls,
                generation_receipts=[self._cached_object(directory, "receipt", row) for row in runtime.generation_receipts],
                training_audits=[self._cached_object(directory, "training", row) for row in backend.training_audits],
                sampling_audits=[self._cached_object(directory, "sampling", row) for row in backend.sampling_audits])
            return dict(schema=SCHEMA + "_runtime", binding_sha256=self._identity["binding_sha256"], object=_put(directory, value))

    def restore_runtime_state(self, payload, directory):
        if payload.get("schema") != SCHEMA + "_runtime" or payload["binding_sha256"] != self._identity["binding_sha256"]:
            raise ValueError("weak runtime continuation binding differs")
        backend, runtime = self.backend, self.backend.backend
        with backend._lock:
            backend._check()
            if runtime._resident_cases or runtime._issued:
                raise ValueError("restore weak runtime before issuing any Case")
            value = _get(directory, payload["object"])
            issued = value["issued"]
            if len(set(issued)) != len(issued):
                raise ValueError("duplicate issued weak Case IDs")
            residents = {}
            for descriptor in value["residents"]:
                record = self.restore_case_record(descriptor, directory)
                case = record.case
                row = backend._restored_records[case.case_id]
                if case.case_id not in issued or case.case_id in residents:
                    raise ValueError("invalid restored resident identity")
                residents[case.case_id] = _Resident(case, _snapshot_vpfp_public_case(case),
                    row["carrier"], row["condition"], row["answer_sha256"])
            if not set(value["draws"]) <= set(residents):
                raise ValueError("feedback draw has no restored resident Case")
            runtime._resident_cases, runtime._issued = residents, set(issued)
            runtime.oracle_calls = value["oracle_calls"]
            runtime.generation_receipts = [_get(directory, row) for row in value["generation_receipts"]]
            backend._draws = {key: _get(directory, row) for key, row in value["draws"].items()}
            backend._private_answers = value["private_answers"]
            backend.training_audits = [_get(directory, row) for row in value["training_audits"]]
            backend.sampling_audits = [_get(directory, row) for row in value["sampling_audits"]]

    def export_generator_state(self, state, directory):
        if state.base != self._identity or state.checkpoint.payload != self._identity:
            raise ValueError("weak GeneratorState Base identity differs")
        guide = None
        if state.guidance is not None:
            snapshot = state.guidance; snapshot.assert_frozen(self.backend.capture.base)
            key = (str(Path(directory).resolve()), "guide", id(snapshot))
            cached = self.backend._export_cache.get(key)
            if cached is None or cached[0] is not snapshot:
                cached = (snapshot, _put(directory, dict(heads={k: guidance_payload(v) for k, v in snapshot.heads.items()},
                    report=snapshot.report, checkpoint_path=str(snapshot.checkpoint_path), checkpoint_sha256=snapshot.checkpoint_sha256)))
                self.backend._export_cache[key] = cached
            guide = cached[1]
        return dict(schema=SCHEMA + "_state", binding_sha256=self._identity["binding_sha256"], guidance=guide,
            checkpoint=dict(generator_id=state.checkpoint.generator_id, location=state.checkpoint.location,
                            report=deepcopy(state.checkpoint.report)))

    def restore_generator_state(self, payload, directory):
        if payload.get("schema") != SCHEMA + "_state" or payload["binding_sha256"] != self._identity["binding_sha256"]:
            raise ValueError("weak GeneratorState continuation binding differs")
        descriptor = payload["checkpoint"]
        if descriptor["generator_id"] != self.info.program_id:
            raise ValueError("weak GeneratorState program differs")
        guide = None
        if payload["guidance"] is not None:
            value = _get(directory, payload["guidance"])
            base = self.backend.capture.base
            device = next(base.micro.parameters()).device
            heads = {k: load_guidance_payload(v, base).to(device) for k, v in value["heads"].items()}
            guide = WeakGuidanceSnapshot(heads, value["report"], value["checkpoint_path"], value["checkpoint_sha256"])
            guide.assert_frozen(base)
            check_guidance_objective(self.backend.score_calibration, guide, {})
        checkpoint = GeneratorCheckpoint(self.info.program_id, deepcopy(self._identity),
                                         descriptor["location"], descriptor["report"])
        return GeneratorState(checkpoint, deepcopy(self._identity), guide)


def build_weak_iteration_binding(*, decoder, sampler, draw_function, output_root, trainer,
                                 initial_guidance=None, guidance_config=None, perturbation_config=None,
                                 oracle_execution_options=None, metric_profile=None, metric_calibration=None):
    """Loader/round trainer are explicitly supplied; no legacy IC40 coercion."""
    if initial_guidance is not None:
        initial_guidance.assert_frozen(sampler)
    capture = _CapturedSampler(sampler, draw_function)
    binding = binding_from_sampler(decoder, capture, output_root=Path(output_root) / "resident-cases",
        regime_id="weak", adaptive_source_only=True, source_chunk=32, field_chunk=32,
        **({"oracle_execution_options": oracle_execution_options}
           if oracle_execution_options is not None else {}))
    backend = WeakIterationBackend(binding.plugin.generator.backend, capture,
        output_root=Path(output_root) / "guidance", trainer=trainer,
        config={} if guidance_config is None else guidance_config, initial_guidance=initial_guidance,
        perturbation_config=perturbation_config, metric_profile=metric_profile,
        metric_calibration=metric_calibration)
    identity = deepcopy(dict(binding.report))
    identity.update(schema=SCHEMA, guidance="fresh_selected_solver_native_hierarchical_two_head",
        guidance_config=deepcopy(backend.config), feedback_selection=FEEDBACK_SELECTION,
        guidance_enabled=True, perturbations_enabled=backend.supports_perturbations,
        perturbations=("unsupported" if perturbation_config is None else deepcopy(backend.perturbation_identity)),
        initial_guidance_sha256=None if initial_guidance is None else initial_guidance.checkpoint_sha256,
        implementation_sha256=file_sha(Path(__file__)))
    if backend.score_calibration is not None:
        identity["scoring_objective"] = backend.score_calibration.objective_identity
    identity.pop("binding_sha256"); identity["binding_sha256"] = json_sha(identity)
    generator = WeakIterationGenerator(backend, identity)
    from .product import build_vpfp_plugin
    plugin = (binding.plugin if backend.score_calibration is None else
              build_vpfp_plugin(metric_profile=metric_profile, metric_calibration=metric_calibration))
    return replace(binding, plugin=replace(plugin, generator=generator),
        checkpoint=GeneratorCheckpoint(generator.info.program_id, identity, report=identity), report=identity)
