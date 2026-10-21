"""Canonical feedback bridge to Artemis' existing bounded reward transport.

This remains an experimental guidance arm: the current physical qualification
reports insufficient perturbation hardness evidence. Base admission and the
mathematical Generator are unchanged by fitting this small reward model.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from threading import Lock
from time import perf_counter
from typing import Mapping
from ..plasma_transport_settings import transport_calibration, check_guidance_objective, check_feedback_objective


@dataclass(frozen=True)
class ArtemisRewardState:
    base_sha256: str
    snapshot_sha256: str
    target_solver_id: str
    model: object
    normalization: object
    report: Mapping[str, object]


class ArtemisRewardGuidance:
    """Reuse the pinned reward-model math in canonical update_guidance/sample."""

    def __init__(self, package, source, *, settings, device="cpu", metric_profile=None,
                 metric_calibration=None):
        self.source, self.device = source, device
        self.score_calibration = transport_calibration(metric_profile, metric_calibration)
        self.api = package.module("artemis_reward_guidance")
        self.runtime_api = package.module("artemis_compact_v5_production_runtime")
        self.seed_api = package.module("manufactured_backend")
        self.config = self.api.RewardGuidanceConfig.from_dict({
            **self.api.RewardGuidanceConfig().to_dict(), **dict(settings),
        })
        self.conditions = tuple(source.runtime.conditions)
        if len(self.conditions) != self.config.condition_count:
            raise ValueError("Artemis guidance condition count differs from the admitted runtime")
        if self.config.condition_probabilities is not None:
            raise ValueError("Artemis canonical guidance preserves the production condition stream")
        self._coordinates = {}
        self.last_sampling_report = {}
        self.training_audits = []
        self.sampling_audits = []
        self._audit_lock = Lock()

    def retain_feedback_coordinates(self, records):
        # This is the only private-source-layout bridge, pinned to 7c22b7f.
        # The old seal method restricts cohorts to GUIDANCE_FEEDBACK. Canonical
        # rounds also produce valid selected-Solver feedback from Official;
        # retain just its three scalar primitives, never any answer arrays.
        with self.source._lock:
            roots = [(record.case.case_id, self.source._roots.get(record.case.case_id)) for record in records]
            for identity, root in roots:
                if root is not None:
                    self._coordinates[identity] = self.source.runtime.replay_guidance_feedback_coordinate(root.manufactured)

    def release_feedback_coordinates(self, case_ids):
        for identity in case_ids:
            self._coordinates.pop(identity, None)

    def fit_guidance(self, base_state, previous_guidance, feedback, settings, progress=None):
        # Refit for the entering incumbent on this feedback cohort. The old
        # Guidance model is not loaded or warm-started; only its code/config
        # are reused. The Base Generator checkpoint remains fixed separately.
        import torch
        check_guidance_objective(self.score_calibration, previous_guidance, settings)
        check_feedback_objective(self.score_calibration, feedback)
        started = perf_counter()
        if base_state.artifact_sha256 != self.source.artifact_sha256:
            raise ValueError("Artemis guidance feedback belongs to another base")
        rows, selection = self._feedback_rows(feedback, settings)
        if any(row.case.case_id not in self._coordinates for row in rows):
            raise ValueError("Artemis guidance feedback references a released or unknown root")
        capsules = tuple(self._coordinates[row.case.case_id] for row in rows)
        latents, conditions = self._primitive_tensors(capsules)
        rewards = torch.tensor([row.primary_error for row in rows])
        if selection is None:
            model, report, normalization = self.api.fit_reward_model(
                latents, conditions, rewards,
                seed=int(settings.get("controller_seed", 0)), device=self.device,
                config=self.config, maximum_wall_seconds=settings.get("guidance_maximum_wall_seconds"),
            )
        else:
            model, report, normalization = self._fit_aligned_reward_model(
                latents, conditions, rewards, settings)
            report = {**report, **selection}
        identity_payload = {
            "base": self.source.artifact_sha256, "config": self.config.sha256,
            "source_round": feedback.round_index,
            "target_solver": feedback.selected_artifact.artifact_id,
            "feedback": [(row.case.case_id, row.primary_error) for row in rows],
        }
        if selection is not None:
            identity_payload["feedback_selection"] = selection
        if self.score_calibration is not None:
            identity_payload["scoring_objective"] = self.score_calibration.objective_identity
            report = {**report, "scoring_objective": self.score_calibration.objective_identity,
                      "scoring_objective_hash": self.score_calibration.objective_hash}
        identity = sha256(json.dumps(identity_payload, sort_keys=True, allow_nan=False).encode())
        for key, value in sorted(model.state_dict().items()):
            identity.update(key.encode())
            identity.update(value.detach().cpu().contiguous().numpy().tobytes())
        if progress:
            progress("Artemis reward model fitted; base Generator remains frozen", 1.0)
        state = ArtemisRewardState(self.source.artifact_sha256, identity.hexdigest(),
            feedback.selected_artifact.artifact_id, model, normalization,
            {**report, "qualification": "experimental_insufficient_perturbation_evidence",
             "base_model_updated": False, "feedback_cohort": "canonical_selected_solver"})
        # These are small execution records, not new model-admission checks.
        # Never retain feedback answers, primitive arrays or model parameters.
        with self._audit_lock:
            self.training_audits.append({
                "source_round_index": feedback.round_index,
                "target_solver_id": state.target_solver_id,
                "initialization": "fresh_fit_from_incumbent_feedback",
                "previous_model_reused": previous_guidance is not None and model is previous_guidance.model,
                "feedback_case_count": len(rows),
                "condition_counts": torch.bincount(conditions, minlength=len(self.conditions)).tolist(),
                "controller_seed": int(settings.get("controller_seed", 0)),
                "snapshot_sha256": state.snapshot_sha256,
                "base_sha256": state.base_sha256,
                "guidance_config_sha256": self.config.sha256,
                "wall_seconds": perf_counter() - started,
                "model_report": dict(state.report),
            })
        return state

    def _feedback_rows(self, feedback, settings):
        roots = tuple(row for row in feedback.cases if row.case.parent_case_id is None)
        selection = settings.get("feedback_selection")
        if selection is None:
            return roots, None
        if (selection != "first_80_clean_base_feedback_order"
                or settings.get("feedback_case_count") != 80):
            raise ValueError("Artemis aligned Guidance requires first 80 clean BASE feedback rows")
        # GUIDED roots share the public Case schema with BASE roots. Sampling
        # audit identities, already checkpointed by resume.py, distinguish them
        # without exposing private generator coordinates or inspecting labels.
        with self._audit_lock:
            audits = tuple(self.sampling_audits)
        if any(audit.get("count", 0) and "case_ids" not in audit for audit in audits):
            raise ValueError("Aligned Guidance needs sampled case IDs; start a fresh aligned campaign")
        guided_ids = {identity for audit in audits for identity in audit.get("case_ids", ())}
        clean = tuple(row for row in roots if row.case.case_id not in guided_ids)
        if len(clean) < 80:
            raise ValueError("Artemis aligned Guidance needs at least 80 clean BASE feedback rows")
        rows = clean[:80]  # Selection/order is independent of solver-error labels.
        return rows, {
            "feedback_selection": selection,
            "feedback_case_ids": [row.case.case_id for row in rows],
            "training_case_ids": [row.case.case_id for row in rows[:64]],
            "validation_case_ids": [row.case.case_id for row in rows[64:]],
            "available_clean_base_count": len(clean),
            "excluded_guided_root_count": len(roots) - len(clean),
            "excluded_child_count": len(feedback.cases) - len(roots),
        }

    def _fit_aligned_reward_model(self, latents, conditions, rewards, settings):
        """Fit the first 64 BASE rows; use the last 16 only for activation.

        This reuses the pinned OU-Hermite ridge and bounded transport, not a
        diffusion-network fit. Missing/unsupported conditions retain zero
        gradients; there is no stratified redraw, label sorting or 80-row refit.
        """
        import torch
        started = perf_counter()
        maximum = settings.get("guidance_maximum_wall_seconds")
        if maximum is not None:
            maximum = self.api._finite_float(maximum, "guidance_maximum_wall_seconds")
            if maximum <= 0:
                raise ValueError("guidance_maximum_wall_seconds must be positive")
        if self.config.model_kind != "ou_hermite_ridge_v1":
            raise ValueError("Artemis aligned Guidance requires the pinned OU-Hermite ridge model")
        if not bool(torch.isfinite(rewards).all()) or bool((rewards < 0).any()):
            raise ValueError("Artemis Guidance rewards must be finite and nonnegative")
        train_z, train_c, train_r = latents[:64].float(), conditions[:64], rewards[:64]
        valid_z, valid_c, valid_r = latents[64:].float(), conditions[64:], rewards[64:]
        normalization = self.api._normalization_from_rewards(train_r, self.config)

        # The pinned fitter requires each configured condition to be present.
        # Fit only the observed subset under identical per-condition ridge
        # mathematics, then copy back into the full production condition space.
        present, remapped = torch.unique(train_c, sorted=True, return_inverse=True)
        observed_config = replace(self.config, condition_count=present.numel())
        observed = self.api._fit_hermite_coefficients(
            train_z, remapped, train_r, torch.arange(64), observed_config, normalization)
        model = self.api.build_reward_model(self.config)
        with torch.no_grad():
            model.hermite_coefficients[present] = observed.hermite_coefficients
        prediction = self.api.predict_rewards(
            model, normalization, valid_z, valid_c, device="cpu")
        train_counts = torch.bincount(train_c, minlength=len(self.conditions)).tolist()
        valid_counts = torch.bincount(valid_c, minlength=len(self.conditions)).tolist()
        active, inactive, reliability = [], [], {}
        nonconstant = model.hermite_degrees != 0
        for condition in range(len(self.conditions)):
            mask = valid_c == condition
            supported = train_counts[condition] > 0 and valid_counts[condition] >= 2
            metrics = self.api._ranking_diagnostics(valid_r[mask], prediction[mask]) if supported else {}
            enabled = bool(supported and metrics["spearman"] > 0.0 and metrics["top_quartile_lift"] > 1.0)
            (active if enabled else inactive).append(condition)
            reliability[str(condition)] = {
                "active": enabled, "training_count": train_counts[condition],
                "validation_count": valid_counts[condition], **metrics,
                "reason": ("positive_holdout_ranking_and_lift" if enabled else
                           "insufficient_condition_support" if not supported else "holdout_gate_not_passed"),
            }
            if not enabled:
                with torch.no_grad():
                    model.hermite_coefficients[condition, nonconstant] = 0.0
        report = {
            "schema": self.api.ARTEMIS_REWARD_GUIDANCE_REPORT_SCHEMA,
            "seed": int(settings.get("controller_seed", 0)),
            "sample_count": 80, "primitive_latent_dim": 3,
            "base_model_parameters_seen_or_updated": 0,
            "model_kind": self.config.model_kind,
            "model_selection_on_crossfit": False,
            "normalization": normalization.to_dict(),
            "split": {
                "strategy": "first_64_fit_last_16_holdout_in_clean_base_feedback_order",
                "training_count": 64, "validation_count": 16,
                "training_count_by_condition": train_counts,
                "validation_count_by_condition": valid_counts,
                "normalization_fit_count": 64,
                "validation_labels_used_for_fit": False,
                "validation_labels_used_for_activation": True,
            },
            "fit": {
                "method": "condition_specific_normalized_probabilists_hermite_closed_form_ridge",
                "hermite_degree": self.config.hermite_degree,
                "ridge_alpha": self.config.ridge_alpha,
                "ou_attenuation": "alpha_to_total_degree",
                "model_selection_action": "condition_wise_nonconstant_coefficient_zeroing",
                "final_fit_count": 64, "final_refit": "none_after_holdout_activation",
            },
            "optimization": {"loss": "closed_form_normalized_ridge", "steps": 1},
            "validation_diagnostics": self.api._ranking_diagnostics(valid_r, prediction),
            "validation_diagnostics_role": "held_out_from_fit_used_for_activation_not_unbiased_outer_test",
            "activation_criteria": {"minimum_validation_rows_per_condition": 2,
                                    "spearman_strictly_greater_than": 0.0,
                                    "top_quartile_lift_strictly_greater_than": 1.0},
            "condition_reliability": reliability,
            "active_condition_indices": active, "inactive_condition_indices": inactive,
            "activation_guard_passed": bool(active),
        }
        model = self.api._freeze_reward_model(model.to(self.device))
        if maximum is not None and perf_counter() - started > maximum:
            raise TimeoutError("Artemis aligned Guidance exceeded its fitting wall budget")
        return model, report, normalization

    def generate(self, base_state, state, request, progress=None):
        import torch
        started = perf_counter()
        if not isinstance(state, ArtemisRewardState) or state.base_sha256 != self.source.artifact_sha256:
            raise ValueError("Artemis guided sampling needs a matching frozen reward state")
        check_guidance_objective(self.score_calibration, state, {})
        request_id = f"canonical-{request.purpose.value}-r{request.round_index}-s{request.seed}-n{request.count}"
        seeds = tuple(self.seed_api.vp1d_guided_answer_first_seed(
            controller_seed=request.seed, guide_snapshot_sha256=state.snapshot_sha256,
            request_id=request_id, sample_index=index,
        ) for index in range(request.count))
        raw = tuple(self.source.runtime.guidance_coordinate_from_seed(seed) for seed in seeds)
        raw_latents, raw_conditions = self._primitive_tensors(raw)
        moved, conditions, diagnostics = self.api.sample_guided_primitives(
            state.model, state.normalization, request.count, request.seed,
            self.config, device=self.device, raw_latents=raw_latents, raw_conditions=raw_conditions,
        )
        if not torch.equal(conditions, raw_conditions) or diagnostics.get("raw_primitive_source") != "provided_canonical_base_capsules":
            raise ValueError("Artemis guidance changed the production condition stream")
        unmoved = (moved == raw_latents).all(dim=1)
        uniforms = self.api.primitive_latents_to_uniforms(moved)
        capsules = tuple(original if bool(same) else self.runtime_api.ArtemisCompactV5GuidanceCoordinateCapsule.from_unit_uniforms(
            original.condition, tuple(float(value) for value in uniform))
            for original, uniform, same in zip(raw, uniforms, unmoved, strict=True))
        records = tuple(self.source.issue_guided_answer_first_records(
            capsules, round_index=request.round_index, visibility=request.visibility,
            controller_seed=request.seed, guide_snapshot_sha256=state.snapshot_sha256, request_id=request_id,
        ))
        try:
            bindings = tuple((record.case.case_id, capsule) for record, capsule in zip(records, capsules, strict=True))
        except BaseException:
            self.source.release_cases(tuple(record.case.case_id for record in records))
            raise
        self._coordinates.update(bindings)
        with self._audit_lock:
            self.last_sampling_report = {**diagnostics, "exact_raw_capsule_reuse_count": int(unmoved.sum())}
            self.sampling_audits.append({
                "round_index": request.round_index,
                "purpose": request.purpose.value,
                "visibility": request.visibility.value,
                "controller_seed": request.seed,
                "count": request.count,
                "case_ids": [record.case.case_id for record in records],
                "snapshot_sha256": state.snapshot_sha256,
                "target_solver_id": state.target_solver_id,
                "wall_seconds": perf_counter() - started,
                "diagnostics": dict(self.last_sampling_report),
            })
        return records

    def _primitive_tensors(self, capsules):
        import torch
        numerators = torch.tensor([
            [c.clean_uniform_numerator, c.endpoint_uniform_numerator, c.spatial_phase_numerator]
            for c in capsules], dtype=torch.float64).reshape(-1, 3)
        latents = self.api.primitive_uniforms_to_latents(numerators / float(2**53), endpoint_policy="clamp")
        conditions = torch.tensor([self.conditions.index(c.condition) for c in capsules], dtype=torch.long)
        return latents, conditions
