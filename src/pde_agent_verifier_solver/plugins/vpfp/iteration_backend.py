"""IC40 and round-local score Guidance in the sole canonical Campaign.

There is no lifecycle/controller here. The Controller supplies selected-solver
feedback and owns the frozen round state. Only actual clean BASE draws are
eligible to train E_base[exp(reward) | noisy state]; guided draws never become
BASE labels. Public numerical sessions remain the resident demand-only Oracle.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from threading import RLock
from types import SimpleNamespace

from pde_agent_verifier_solver.sdk import CaseSource, GenerationKind, GeneratorCheckpoint

from .canonical_binding import build_vpfp_online_binding, _BoundVPFPContinuousGenerator
from .solver_error_guidance import draw_frozen_ic40, native_batch_from_draw
from ..plasma_transport_settings import transport_calibration, check_guidance_objective, check_feedback_objective


STRICT_FEEDBACK_SELECTION = "strict_exact_80_clean_base"
FORMAL_FEEDBACK_SELECTION = "first_80_clean_base_feedback_order"


def _validate_feedback_selection(value):
    if value not in (STRICT_FEEDBACK_SELECTION, FORMAL_FEEDBACK_SELECTION):
        raise ValueError("unknown VPFP feedback_selection policy")
    return value


def _file_sha(path):
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


class _CapturedSampler:
    """The decoder and feedback receive the SAME draw, never seed replay."""

    def __init__(self, base, batch_size=4):
        self.base = base
        self.identity = deepcopy(base.identity)
        self.identity["actual_draw_capture"] = True
        self.config = SimpleNamespace(batch_size=batch_size)
        self.guide = None
        self.pending = {}
        self.diagnostics = []

    def generate(self, seeds):
        draw = (draw_frozen_ic40(self.base, seeds) if self.guide is None
                else self.guide.draw(seeds))
        if tuple(draw["root_seeds"]) != tuple(seeds):
            raise ValueError("native draw changed requested root seeds")
        for index, seed in enumerate(seeds):
            if seed in self.pending:
                raise ValueError("duplicate pending native root")
            self.pending[seed] = {
                **{key: draw[key][index:index + 1].detach().cpu().clone()
                   for key in ("full", "initial_context", "future_white", "equation_conditions")},
                **{key: [int(draw[key][index])]
                   for key in ("root_seeds", "known_prior_seeds", "future_noise_seeds")},
            }
        report = None if self.guide is None else deepcopy(self.guide.last_report)
        if report is not None:
            self.diagnostics.append(report)
        return native_batch_from_draw(draw, guidance_report=report)


class VPFPTrainingGuidanceBackend:
    """A PDE adapter for BackendGeneratorProgram.update_guidance, not a loop."""

    supports_guidance = True
    supports_perturbations = True

    def __init__(self, backend, capture, *, output_root, config, trainer=None,
                 feedback_selection=STRICT_FEEDBACK_SELECTION, metric_profile=None,
                 metric_calibration=None):
        from .round_solver_guidance import train_round_solver_guidance
        self.backend, self.capture = backend, capture
        self.runtime = backend.runtime
        self.output_root = Path(output_root)
        self.config = config
        self._config_snapshot = asdict(config)
        self.feedback_selection = _validate_feedback_selection(feedback_selection)
        self._feedback_selection_snapshot = self.feedback_selection
        self.trainer = trainer or train_round_solver_guidance
        self.score_calibration = transport_calibration(metric_profile, metric_calibration)
        self._lock = RLock()
        self._draws = {}
        self.training_audits, self.sampling_audits = [], []
        self.public_session_methods = backend.public_session_methods
        self.development_solution_session_methods = backend.development_solution_session_methods

    def train_base(self, *args, **kwargs):
        raise RuntimeError("IC40 Base is frozen; only Guidance trains at round boundaries")

    def public_session_factory(self, case):
        return self.backend.public_session_factory(case)

    def development_solution_session_factory(self, case):
        return self.backend.development_solution_session_factory(case)

    def release_cases(self, case_ids):
        with self._lock:
            self.backend.release_cases(case_ids)
            for case_id in case_ids:
                self._draws.pop(case_id, None)

    def generate(self, base_state, guidance_state, request, progress=None):
        with self._lock:
            if (asdict(self.config) != self._config_snapshot
                    or self.feedback_selection != self._feedback_selection_snapshot):
                raise ValueError("bound round Guidance recipe changed")
            if request.kind is GenerationKind.PERTURBED:
                return self.backend.generate(base_state, None, request, progress)
            if request.kind not in (GenerationKind.BASE, GenerationKind.GUIDED):
                raise ValueError("unsupported VPFP generation kind")
            guided = request.kind is GenerationKind.GUIDED
            if guided and guidance_state is None:
                raise ValueError("GUIDED sampling requires a trained round snapshot")
            if guided:
                if guidance_state.sampler.base is not self.capture.base:
                    raise ValueError("Guidance snapshot belongs to another Base")
                check_guidance_objective(self.score_calibration, guidance_state, {})
                if request.round_index < guidance_state.report["target_round_index"]:
                    raise ValueError("cannot sample a future round Guidance snapshot")
                guidance_state.sampler.assert_frozen()
            if self.capture.pending or self.capture.guide is not None:
                raise RuntimeError("unfinished native draw capture transaction")
            native_request = request
            if guided:
                # Decoder mathematics and anchor identity are unchanged. Keep
                # a distinct seed stream when using its BASE materializer.
                seed = int.from_bytes(sha256(f"vpfp-score-guided:{request.seed}".encode()).digest()[:8], "big") & ((1 << 63) - 1)
                native_request = replace(request, kind=GenerationKind.BASE, seed=seed)
                self.capture.guide = guidance_state.sampler
            self.capture.diagnostics = []
            records = ()
            try:
                from .perturbation_backend import _seeds
                seeds = _seeds(native_request)
                records = self.backend.generate(base_state, None, native_request, progress)
                if set(self.capture.pending) != set(seeds):
                    raise RuntimeError("resident Case does not have its actual native draw")
                for record, seed in zip(records, seeds, strict=True):
                    self._draws[record.case.case_id] = {
                        "source": "guided" if guided else "base",
                        "root_seed": seed, "draw": self.capture.pending[seed],
                    }
                self.sampling_audits.append({
                    "round_index": request.round_index, "purpose": request.purpose.value,
                    "source": "guided" if guided else "base", "count": len(records),
                    "case_ids": [record.case.case_id for record in records],
                    "root_seeds": list(seeds),
                    "snapshot_sha256": guidance_state.checkpoint_sha256 if guided else None,
                    "diagnostics": deepcopy(self.capture.diagnostics),
                })
                return records
            except BaseException:
                if records:
                    self.release_cases(tuple(record.case.case_id for record in records))
                raise
            finally:
                self.capture.pending.clear()
                self.capture.guide = None
                self.capture.diagnostics = []
                if guided:
                    guidance_state.sampler.assert_frozen()

    def fit_guidance(self, base_state, previous_guidance, feedback, settings, progress=None):
        del base_state
        with self._lock:
            check_guidance_objective(self.score_calibration, previous_guidance, settings)
            check_feedback_objective(self.score_calibration, feedback)
            if (asdict(self.config) != self._config_snapshot
                    or self.feedback_selection != self._feedback_selection_snapshot):
                raise ValueError("bound round Guidance recipe changed")
            if previous_guidance is not None:
                if previous_guidance.sampler.base is not self.capture.base:
                    raise ValueError("previous Guidance belongs to another Base")
                previous_guidance.sampler.assert_frozen()
            artifact = feedback.selected_artifact
            solver_sha = getattr(artifact.program, "tree_sha256", None)
            if not isinstance(solver_sha, str) or len(solver_sha) != 64:
                raise ValueError("round Guidance requires the selected frozen Candidate bundle")
            eligible, excluded = [], []
            formal = self.feedback_selection == FORMAL_FEEDBACK_SELECTION
            for feedback_index, item in enumerate(feedback.cases):
                entry = {"case_id": item.case.case_id, "feedback_index": feedback_index,
                         "case_source": item.case.source.value,
                         "parent_case_id": item.case.parent_case_id}
                if item.case.source is CaseSource.PERTURBED:
                    excluded.append({**entry, "reason": "not_clean_base"})
                    continue
                cached = self._draws.get(item.case.case_id)
                if cached is None:
                    raise ValueError("selected feedback lacks its actual still-resident native draw")
                if cached["source"] != "base":
                    excluded.append({**entry, "root_seed": cached["root_seed"],
                                     "native_source": cached["source"], "reason": "not_clean_base"})
                    continue
                eligible.append((item, cached, {**entry, "root_seed": cached["root_seed"],
                                               "native_source": cached["source"]}))
            target = feedback.round_index + 1
            selection_report = None
            selection_manifest = None
            if formal:
                # Determine the entire identity inventory BEFORE inspecting any
                # reward. An invalid label must retain its preassigned fit/dev
                # slot; it cannot cause a later, more favorable root to enter.
                case_ids, root_seeds = set(), set()
                for _item, cached, entry in eligible:
                    case_id, root = entry["case_id"], cached["root_seed"]
                    if not isinstance(case_id, str) or not case_id.strip() or case_id in case_ids:
                        raise ValueError("formal feedback requires unique actual clean BASE case IDs")
                    if type(root) is not int or not 0 <= root < 2**63 or root in root_seeds:
                        raise ValueError("formal feedback requires unique actual clean BASE physical roots")
                    case_ids.add(case_id)
                    root_seeds.add(root)
                if len(eligible) < 80:
                    raise ValueError("formal feedback requires at least 80 unique actual clean BASE roots")
                clean_base_available = len(eligible)
                excluded.extend({**entry, "reason": "outside_fixed_first_80_clean_base_feedback_order"}
                                for _item, _cached, entry in eligible[80:])
                excluded.sort(key=lambda entry: entry["feedback_index"])
                eligible = eligible[:80]
                selection_report = {
                    "schema": "vpfp_formal_feedback_selection_v1",
                    "feedback_selection": self.feedback_selection,
                    "source_round_index": feedback.round_index, "target_round_index": target,
                    "target_solver_id": artifact.artifact_id, "target_solver_sha256": solver_sha,
                    "feedback_case_count": len(feedback.cases),
                    "available_clean_base_case_count": clean_base_available,
                    "selected_clean_base_case_count": 80,
                    "selection_uses_rewards": False, "invalid_labels_replace_roots": False,
                    "selected": [{**entry, "split": "fit" if index < 64 else "validation",
                        "local_slot": f"fit-{index:04d}" if index < 64 else f"validation-{index - 64:04d}"}
                        for index, (_item, _cached, entry) in enumerate(eligible)],
                    "excluded": deepcopy(excluded),
                }
                # A sibling manifest survives a failed trainer without creating
                # the trainer's directory, which must remain exclusive/fresh.
                self.output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
                path = self.output_root / f"round-{target:03d}-feedback-selection.json"
                with path.open("x") as stream:
                    json.dump(selection_report, stream, indent=2, sort_keys=True, allow_nan=False)
                    stream.write("\n")
                path.chmod(0o400)
                selection_manifest = {"path": str(path), "sha256": _file_sha(path)}
            rows = []
            for item, cached, _entry in eligible:
                error = float(item.primary_error)
                valid = math.isfinite(error) and error >= 0.
                rows.append({**cached, "case_id": item.case.case_id, "error": error if valid else None,
                             "valid": valid,
                             "target_solver_sha256": solver_sha,
                             "source_round_index": feedback.round_index})
            matcher_sha = sha256(b"".join((Path(__file__).parent / name).read_bytes()
                                         for name in ("macro_matching.py", "tasks/observable.py"))).hexdigest()
            if self.score_calibration is not None:
                matcher_sha = self.score_calibration.objective_hash
            trained = self.trainer(
                base_sampler=self.capture.base, rows=rows,
                source_round_index=feedback.round_index, target_round_index=target,
                target_solver_sha256=solver_sha, reward_matcher_sha256=matcher_sha,
                controller_seed=int(settings["controller_seed"]),
                output_root=self.output_root / f"round-{target:03d}",
                config=self.config, progress=progress,
            )
            check_guidance_objective(self.score_calibration, trained, {})
            if previous_guidance is not None and trained.guide is previous_guidance.guide:
                raise RuntimeError("round boundary silently reused previous Guidance weights")
            if previous_guidance is not None:
                previous_guidance.sampler.assert_frozen()
            self.training_audits.append({
                "source_round_index": feedback.round_index, "target_round_index": target,
                "target_solver_id": artifact.artifact_id, "target_solver_sha256": solver_sha,
                "feedback_case_count": len(feedback.cases), "clean_base_case_count": len(rows),
                "feedback_selection": self.feedback_selection,
                "feedback_selection_report": selection_report,
                **({"scoring_objective": self.score_calibration.objective_identity}
                   if self.score_calibration is not None else {}),
                "feedback_selection_manifest": selection_manifest,
                "excluded": excluded, "checkpoint_path": str(trained.checkpoint_path),
                "snapshot_sha256": trained.checkpoint_sha256,
                "previous_snapshot_sha256": None if previous_guidance is None else previous_guidance.checkpoint_sha256,
                "fresh_model_initialization": True, "report": deepcopy(trained.report),
            })
            return trained


def build_vpfp_strong_iteration_binding(*, native_spec, base_checkpoint,
        base_checkpoint_sha256, portable_codec, portable_codec_sha256,
        output_root, device="cuda", guidance_config=None,
        feedback_selection=STRICT_FEEDBACK_SELECTION, oracle_kernel="reference_v1",
        metric_profile=None, metric_calibration=None):
    """Load the frozen learned Base, and register a training RECIPE, not a guide."""
    feedback_selection = _validate_feedback_selection(feedback_selection)
    # This factory owns the dedicated Controller process. Match the validated
    # IC40 numerical recipe; do not silently enable TF32 at a new boundary.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    from .portable_initial_conditioned import load_portable_initial_conditioned_sampler
    from .round_solver_guidance import RoundGuidanceConfig
    config = (RoundGuidanceConfig() if guidance_config is None else
              RoundGuidanceConfig(**guidance_config) if isinstance(guidance_config, dict)
              else guidance_config)
    if not isinstance(config, RoundGuidanceConfig):
        raise TypeError("guidance_config must be RoundGuidanceConfig or its fields")
    base = load_portable_initial_conditioned_sampler(
        base_checkpoint, checkpoint_sha256=base_checkpoint_sha256,
        portable_bundle_path=portable_codec, portable_bundle_sha256=portable_codec_sha256,
        device=device, sample_steps=64, sample_churn=0.)
    base.model.eval().requires_grad_(False)
    base.normalizer.eval().requires_grad_(False)
    capture = _CapturedSampler(base)
    binding = build_vpfp_online_binding(native_spec, output_root=Path(output_root) / "resident-cases",
                                       device=device, latent_sampler=capture, guidance=False,
                                       oracle_kernel=oracle_kernel, metric_profile=metric_profile,
                                       metric_calibration=metric_calibration)
    if tuple(base.identity["collision_nu_range"]) != (10., 30.):
        raise ValueError("this canonical recipe is qualified only for strong IC40")
    backend = VPFPTrainingGuidanceBackend(binding.plugin.generator.backend, capture,
        output_root=Path(output_root) / "guidance", config=config,
        feedback_selection=feedback_selection, metric_profile=metric_profile,
        metric_calibration=metric_calibration)
    identity = deepcopy(dict(binding.report))
    identity.update(schema="vpfp_strong_canonical_round_guidance_binding_v1",
                    guidance="per_round_selected_solver_exponential_score_relative_rescaling",
                    guidance_config=asdict(config), guidance_enabled=True,
                    perturbations_enabled=True, initial_guidance_checkpoint=None,
                    feedback="actual_clean_base_selected_solver_official_errors",
                    feedback_selection=feedback_selection,
                    scientific_qualification="training_recipe_validated_for_prior_solver_not_universal_asset",
                    iteration_implementation_sha256={name: _file_sha(Path(__file__).with_name(name))
                        for name in ("iteration_backend.py", "round_solver_guidance.py",
                                     "standardized_solver_error_guidance.py", "rescaled_solver_error_guidance.py")})
    identity.pop("binding_sha256")
    identity["binding_sha256"] = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    generator = _BoundVPFPContinuousGenerator(backend, identity)
    return replace(binding, plugin=replace(binding.plugin, generator=generator),
                   checkpoint=GeneratorCheckpoint(generator.info.program_id, identity, report=identity),
                   report=identity)
