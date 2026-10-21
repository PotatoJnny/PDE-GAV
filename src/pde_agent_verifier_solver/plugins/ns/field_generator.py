"""Geometry-aware coefficient Generator for completed NS velocity datasets.

The representation owns the geometry: its mean and modes live on the points
provided by the dataset.  The Rectified Flow only learns the distribution of
the corresponding coefficients, so no cylinder, channel, Cartesian grid, or
pressure assumption appears in the Generator.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    Case,
    CaseRecord,
    CaseSource,
    GenerationKind,
    GenerationRequest,
    RoundFeedback,
)

from .datasets import NSPointCloudArrayTruth, NSPointCloudTruth
from .reconstruction.real_piv import TrajectoryPODPrior, load_trajectory_pod


# The product consumes generated samples, so sample-distribution checks carry
# more weight than the local training objective. Ranking first avoids adding
# unlike units (a Wasserstein distance and a loss) as though they were the same
# quantity.
FLOW_CHECKPOINT_SELECTION_WEIGHTS = {
    "validation_loss": 1.0,
    "latent_sliced_wasserstein": 3.0,
    "latent_mmd": 2.0,
    "latent_moment_error": 2.0,
    "latent_energy_quantile_error": 1.0,
    "latent_coverage_error": 1.0,
}


@dataclass(frozen=True)
class LatentFlowConfig:
    """Small Rectified Flow used for the low-dimensional trajectory scores."""

    feature_dim: int
    width: int = 128
    depth: int = 3
    time_features: int = 32

    def validate(self) -> None:
        if min(self.feature_dim, self.width, self.depth, self.time_features) < 1:
            raise ValueError("latent Rectified Flow dimensions must be positive")
        if self.time_features % 2:
            raise ValueError("latent Rectified Flow time_features must be even")


@dataclass
class NSFieldGeneratorState:
    """Base Generator and the dataset-owned information needed to decode it."""

    prior: TrajectoryPODPrior
    model: object
    model_config: LatentFlowConfig
    train_latents: np.ndarray
    valid_anchor_latents: np.ndarray
    validation_latents: np.ndarray
    geometry: Mapping[str, object]
    x: np.ndarray
    y: np.ndarray
    times: np.ndarray
    sensor_indices: np.ndarray
    observed_frames: np.ndarray
    noise_standard_deviation: np.ndarray
    validation_report: Mapping[str, object]
    validity_profile: Mapping[str, object]
    training_history: tuple[Mapping[str, float], ...]
    flow_proposal_fraction: float
    project_reconstruction: Mapping[str, object] | None = None


class NSGeneratorValidationError(RuntimeError):
    """A trained Generator that was rejected by measured-field validation."""

    def __init__(self, state: NSFieldGeneratorState) -> None:
        self.state = state
        failed = ", ".join(state.validation_report["failed_checks"])
        super().__init__(
            "Base Generator failed measured-field validation: " + failed
        )


@dataclass(frozen=True)
class NSLatentGuidance:
    """Gradient of a ridge-fitted linear solver-error model in latent space."""

    direction: np.ndarray
    fit_case_count: int
    error_mean: float
    error_standard_deviation: float
    fitted_r_squared: float


class NSDatasetCoefficientGeneratorBackend:
    """Generator backend for ``pde_completed_case_dataset_v1``.

    The same backend works for any fixed point-cloud geometry whose completed
    manifest supplies a trajectory POD representation.  A different storage
    topology can provide another backend without changing ``NSProblem`` or the
    campaign engine.
    """

    def __init__(self) -> None:
        self._serial = 0
        self._latents_by_case_id: dict[str, np.ndarray] = {}

    supports_guidance = True
    supports_perturbations = True
    trainable = True

    def release_cases(self, case_ids: Sequence[str]) -> None:
        """Release controller-private coefficients with their Case cohort."""

        for case_id in tuple(case_ids):
            self._latents_by_case_id.pop(str(case_id), None)

    def train_base(
        self,
        dataset,
        settings: Mapping[str, object],
        progress=None,
    ) -> NSFieldGeneratorState:
        checkpoint = settings.get("generator_checkpoint")
        if checkpoint:
            return self._load_checkpoint(dataset, Path(str(checkpoint)), settings, progress)

        summary = dataset.training_summary
        if summary.get("dataset_format") != "pde_completed_case_dataset_v1":
            raise ValueError(
                "the local geometry-aware NS Generator expects a completed-field "
                "dataset; use the canonical structured numerical Generator/backend "
                "for ns_d0_v1"
            )
        if summary.get("topology") != "point_cloud":
            raise ValueError("the current local NS Generator expects point_cloud storage")
        representation = summary.get("representation")
        if not representation:
            raise ValueError(
                "completed NS manifest must declare artifacts.representation"
            )
        prior = load_trajectory_pod(Path(str(representation)))
        candidate_ranks = _basis_rank_candidates(settings, prior.rank)
        prior = prior.with_rank(max(candidate_ranks))

        training_records = tuple(dataset.generator_training_cases) or tuple(
            dataset.development_cases
        )
        validation_records = tuple(dataset.generator_validation_cases)
        split_counts = _mapping(summary.get("split_counts"))
        default_minimum = 8 if "generator_training" in split_counts else 2
        minimum_training_cases = int(
            settings.get("minimum_training_cases", default_minimum)
        )
        if len(training_records) < minimum_training_cases:
            raise ValueError(
                f"NS Generator needs at least {minimum_training_cases} training Cases; "
                f"the dataset supplied {len(training_records)}"
            )
        seed = _settings_seed(settings)
        rng = np.random.default_rng(seed)

        training_velocity = np.stack(
            [_record_velocity(record) for record in training_records],
            axis=0,
        )
        train_latents = np.asarray(prior.encode(training_velocity), dtype=np.float32)
        external_validation = len(validation_records) >= 2
        if external_validation:
            validation_velocity = np.stack(
                [_record_velocity(record) for record in validation_records],
                axis=0,
            )
            validation_latents = np.asarray(
                prior.encode(validation_velocity),
                dtype=np.float32,
            )
            fit_latents = train_latents
        else:
            order = rng.permutation(len(train_latents))
            if len(train_latents) < 4:
                validation_indices = order
                fit_latents = train_latents
            else:
                validation_count = max(2, min(len(train_latents) // 5, 64))
                validation_indices = order[:validation_count]
                fit_latents = train_latents[order[validation_count:]]
            validation_latents = train_latents[validation_indices]
            validation_velocity = training_velocity[validation_indices]
        if len(fit_latents) < 2 or len(validation_latents) < 2:
            raise ValueError("NS Generator fit and validation cohorts are too small")
        rank, rank_selection = _select_generator_basis_rank(
            prior=prior,
            candidate_ranks=candidate_ranks,
            fit_latents=fit_latents,
            validation_latents=validation_latents,
            validation_velocity=validation_velocity,
            settings=settings,
            rng=rng,
        )
        if rank != prior.rank:
            prior = prior.with_rank(rank)
            train_latents = train_latents[:, :rank]
            fit_latents = fit_latents[:, :rank]
            validation_latents = validation_latents[:, :rank]
        if progress:
            progress(
                f"Selected Generator basis rank {rank} from isolated validation",
                0.06,
            )

        # Initial/development Cases are not training samples. Their coefficients
        # are registered only so later solver-error guidance and same-round
        # perturbation can refer to them by Case id.
        registered = {}
        for record, latent in zip(training_records, train_latents, strict=True):
            registered[record.case.case_id] = latent
        if external_validation:
            for record, latent in zip(
                validation_records,
                validation_latents,
                strict=True,
            ):
                registered[record.case.case_id] = latent
        for record in (*dataset.development_cases, *dataset.hidden_cases):
            if record.case.case_id not in registered:
                registered[record.case.case_id] = np.asarray(
                    prior.encode(_record_velocity(record)),
                    dtype=np.float32,
                )
        self._latents_by_case_id.update(registered)
        if progress:
            progress(
                f"Encoded {len(train_latents)} training and "
                f"{len(validation_latents)} isolated validation Cases",
                0.08,
            )
        config = LatentFlowConfig(
            feature_dim=rank,
            width=int(settings.get("flow_width", 128)),
            depth=int(settings.get("flow_depth", 3)),
            time_features=int(settings.get("flow_time_features", 32)),
        )
        model, history = train_latent_rectified_flow(
            fit_latents,
            validation_latents,
            config=config,
            steps=int(settings.get("flow_steps", 2_000)),
            batch_size=int(settings.get("flow_batch_size", 64)),
            learning_rate=float(settings.get("flow_learning_rate", 2.0e-4)),
            seed=seed,
            device=str(settings.get("device", "auto")),
            checkpoint_sample_count=int(
                settings.get("checkpoint_sample_count", 256)
            ),
            checkpoint_integration_steps=int(
                settings.get(
                    "checkpoint_integration_steps",
                    settings.get("integration_steps", 32),
                )
            ),
            checkpoint_projection_count=int(
                settings.get("checkpoint_projection_count", 128)
            ),
            progress=progress,
        )

        validation_sample_count = int(settings.get("validation_sample_count", 512))
        flow_proposals = sample_latent_rectified_flow(
            model,
            config,
            validation_sample_count,
            steps=int(settings.get("integration_steps", 64)),
            seed=seed + 404,
            device=str(settings.get("device", "auto")),
            guidance=None,
            guidance_strength=0.0,
            guidance_max_flow_ratio=0.25,
        )
        projected_validation = prior.decode(validation_latents).astype(np.float32)
        projected_training = prior.decode(train_latents).astype(np.float32)
        validity_profile = _build_field_validity_profile(
            projected_training,
            settings,
            projected_validation,
        )
        valid_anchor_latents = train_latents[
            _field_validity_mask(projected_training, validity_profile)
        ]
        if not len(valid_anchor_latents):
            raise RuntimeError("Generator training set has no valid manifold anchors")
        (
            generated_latents,
            generated_velocity,
            flow_proposal_fraction,
            validation_report,
            manifold_mix_selection,
        ) = _select_flow_manifold_mix(
            prior=prior,
            training_latents=train_latents,
            validation_latents=validation_latents,
            validation_velocity=validation_velocity,
            projected_validation=projected_validation,
            flow_proposals=flow_proposals,
            sampling_hz=_sampling_hz(dataset.development_cases[0]),
            settings=settings,
            rng=rng,
            progress=progress,
        )
        validation_report = dict(validation_report)
        validation_report["rank_selection"] = rank_selection
        validation_report["manifold_mix_selection"] = manifold_mix_selection
        template = dataset.development_cases[0].case
        coordinates = _mapping(template.public_data.get("coordinates"))
        observations = _mapping(template.public_data.get("observations"))
        x = np.asarray(coordinates.get("x"), dtype=np.float64)
        y = np.asarray(coordinates.get("y"), dtype=np.float64)
        times = np.asarray(coordinates.get("time"), dtype=np.float64)
        if len(times):
            times = times - times[0]
        state = NSFieldGeneratorState(
            prior=prior,
            model=model,
            model_config=config,
            train_latents=train_latents,
            valid_anchor_latents=valid_anchor_latents,
            validation_latents=validation_latents,
            geometry=dict(_mapping(template.public_data.get("geometry"))),
            x=x,
            y=y,
            times=times,
            sensor_indices=np.asarray(
                observations.get("sensor_indices"), dtype=np.int64
            ),
            observed_frames=np.asarray(
                observations.get("observed_frames"), dtype=np.int64
            ),
            noise_standard_deviation=np.asarray(
                observations.get("noise_standard_deviation", (1.0, 1.0)),
                dtype=np.float64,
            ),
            validation_report=validation_report,
            validity_profile=validity_profile,
            training_history=tuple(history),
            flow_proposal_fraction=flow_proposal_fraction,
            project_reconstruction=_project_reconstruction_from_case(template),
        )
        _validate_state(state)
        if bool(settings.get("require_validation_pass", True)) and not bool(
            validation_report["accepted"]
        ):
            raise NSGeneratorValidationError(state)
        if progress:
            progress("Geometry-aware NS Base Generator is ready", 1.0)
        return state

    def _load_checkpoint(
        self,
        dataset,
        checkpoint: Path,
        settings: Mapping[str, object],
        progress=None,
    ) -> NSFieldGeneratorState:
        """Restore a Base Generator while rebuilding dataset-owned Case latents."""

        import torch

        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        summary = dataset.training_summary
        if summary.get("dataset_format") != "pde_completed_case_dataset_v1":
            raise ValueError("an NS field Generator checkpoint needs a completed dataset")
        representation = summary.get("representation")
        if not representation:
            raise ValueError("completed NS dataset has no representation artifact")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        schema = str(payload.get("schema", ""))
        if schema not in {
            "ns_dataset_latent_flow_v1",
            "ns_dataset_latent_flow_v2",
        }:
            raise ValueError(
                "unsupported NS field Generator checkpoint schema "
                f"{schema!r}"
            )
        raw_config = payload.get("model_config", {})
        config = LatentFlowConfig(
            feature_dim=int(raw_config["feature_dim"]),
            width=int(raw_config["width"]),
            depth=int(raw_config["depth"]),
            time_features=int(raw_config["time_features"]),
        )
        config.validate()
        prior = load_trajectory_pod(Path(str(representation))).with_rank(
            config.feature_dim
        )
        resolved_device = _device(torch, str(settings.get("device", "auto")))
        model = build_latent_flow_model(config)
        model.load_state_dict(payload["model_state"])
        model = model.to(resolved_device).eval()

        if "train_latents" in payload and "validation_latents" in payload:
            train_latents = np.asarray(payload["train_latents"], dtype=np.float32)
            validation_latents = np.asarray(
                payload["validation_latents"],
                dtype=np.float32,
            )
        else:
            # The collaborator's first validated field-flow artifact used the
            # compact v1 schema: model weights and checkpoint-selection
            # history were frozen, while coefficient banks remained
            # dataset-owned.  Rebuild those banks from the immutable completed
            # manifest instead of rejecting or retraining that checkpoint.
            training_records = tuple(dataset.generator_training_cases) or tuple(
                dataset.development_cases
            )
            validation_records = tuple(dataset.generator_validation_cases)
            sidecar = _checkpoint_sidecar(checkpoint, "train_report.json")
            reported_training_count = int(
                sidecar.get("training_cases", len(training_records))
            )
            requested_training_count = int(
                settings.get(
                    "checkpoint_training_cases",
                    reported_training_count,
                )
            )
            requested_validation_count = int(
                settings.get(
                    "checkpoint_validation_cases",
                    len(validation_records),
                )
            )
            training_records = training_records[
                : min(requested_training_count, len(training_records))
            ]
            validation_records = validation_records[
                : min(requested_validation_count, len(validation_records))
            ]
            if len(training_records) < 2 or len(validation_records) < 2:
                raise ValueError(
                    "compact NS Generator checkpoints need at least two "
                    "dataset-owned training and validation Cases"
                )
            train_latents = np.asarray(
                prior.encode(
                    np.stack(
                        [_record_velocity(record) for record in training_records],
                        axis=0,
                    )
                ),
                dtype=np.float32,
            )
            validation_latents = np.asarray(
                prior.encode(
                    np.stack(
                        [_record_velocity(record) for record in validation_records],
                        axis=0,
                    )
                ),
                dtype=np.float32,
            )
        projected_training = prior.decode(train_latents).astype(np.float32)
        projected_validation = prior.decode(validation_latents).astype(np.float32)
        validity_profile = _build_field_validity_profile(
            projected_training,
            settings,
            projected_validation,
        )
        valid_anchor_latents = train_latents[
            _field_validity_mask(projected_training, validity_profile)
        ]
        if not len(valid_anchor_latents):
            raise RuntimeError("Generator checkpoint has no valid manifold anchors")
        for record in (*dataset.development_cases, *dataset.hidden_cases):
            latent = np.asarray(prior.encode(_record_velocity(record)), dtype=np.float32)
            self._latents_by_case_id[record.case.case_id] = latent

        template = dataset.development_cases[0].case
        coordinates = _mapping(template.public_data.get("coordinates"))
        observations = _mapping(template.public_data.get("observations"))
        times = np.asarray(coordinates.get("time"), dtype=np.float64)
        if len(times):
            times = times - times[0]
        history = tuple(dict(row) for row in payload.get("training_history", ()))
        validation_report = dict(payload.get("validation_report", {}))
        if not validation_report:
            selection = _checkpoint_sidecar(checkpoint, "flow_selection.json")
            selected = next(
                (row for row in history if bool(row.get("selected", False))),
                None,
            )
            validation_report = {
                "accepted": selected is not None,
                "validation_kind": (
                    "held_out_generated_distribution_checkpoint_selection"
                ),
                "checkpoint_schema": schema,
                "criterion": selection.get(
                    "criterion",
                    "fixed_held_out_generated_distribution_rank",
                ),
                "selected_step": (
                    int(selected["step"]) if selected is not None else None
                ),
                "measured_field_validation_available": False,
            }
            if not validation_report["accepted"]:
                raise ValueError(
                    "compact NS Generator checkpoint has no selected validation "
                    "candidate"
                )
        state = NSFieldGeneratorState(
            prior=prior,
            model=model,
            model_config=config,
            validation_latents=validation_latents,
            train_latents=train_latents,
            valid_anchor_latents=valid_anchor_latents,
            geometry=dict(_mapping(template.public_data.get("geometry"))),
            x=np.asarray(coordinates.get("x"), dtype=np.float64),
            y=np.asarray(coordinates.get("y"), dtype=np.float64),
            times=times,
            sensor_indices=np.asarray(
                observations.get("sensor_indices"), dtype=np.int64
            ),
            observed_frames=np.asarray(
                observations.get("observed_frames"), dtype=np.int64
            ),
            noise_standard_deviation=np.asarray(
                observations.get("noise_standard_deviation", (1.0, 1.0)),
                dtype=np.float64,
            ),
            validation_report=validation_report,
            validity_profile=validity_profile,
            training_history=history,
            flow_proposal_fraction=float(
                payload.get("flow_proposal_fraction", 1.0)
            ),
            project_reconstruction=_project_reconstruction_from_case(template),
        )
        _validate_state(state)
        if bool(settings.get("require_validation_pass", True)) and not bool(
            validation_report.get("accepted", False)
        ):
            raise NSGeneratorValidationError(state)
        if progress:
            progress(
                f"Loaded the validated Base Generator from {checkpoint.name}",
                1.0,
            )
        return state

    def fit_guidance(
        self,
        base_state: object,
        previous_guidance: object | None,
        feedback: RoundFeedback,
        settings: Mapping[str, object],
        progress=None,
    ) -> NSLatentGuidance:
        state = _expect_state(base_state)
        rows = []
        errors = []
        for item in feedback.cases:
            latent = self._latents_by_case_id.get(item.case.case_id)
            if latent is None:
                continue
            rows.append(latent)
            errors.append(float(item.primary_error))
        minimum = int(settings.get("guidance_minimum_cases", 4))
        if len(rows) < minimum:
            if isinstance(previous_guidance, NSLatentGuidance):
                return previous_guidance
            raise ValueError(
                f"NS guidance needs at least {minimum} scored generated Cases"
            )
        x = np.stack(rows).astype(np.float64)
        y = np.asarray(errors, dtype=np.float64)
        x_centered = x - x.mean(axis=0, keepdims=True)
        y_centered = y - y.mean()
        ridge = float(settings.get("guidance_ridge", 1.0))
        gram = x_centered @ x_centered.T
        dual = np.linalg.solve(
            gram + ridge * np.eye(len(gram), dtype=np.float64),
            y_centered,
        )
        direction = x_centered.T @ dual
        norm = float(np.sqrt(np.mean(np.square(direction))))
        if not np.isfinite(norm) or norm < 1.0e-10:
            direction = x[int(np.argmax(y))] - x.mean(axis=0)
            norm = float(np.sqrt(np.mean(np.square(direction))))
        direction = direction / max(norm, 1.0e-10)
        prediction = x_centered @ (x_centered.T @ dual) + y.mean()
        residual = float(np.sum(np.square(y - prediction)))
        total = float(np.sum(np.square(y_centered)))
        guidance = NSLatentGuidance(
            direction=np.asarray(direction, dtype=np.float32),
            fit_case_count=len(rows),
            error_mean=float(y.mean()),
            error_standard_deviation=float(y.std()),
            fitted_r_squared=1.0 - residual / max(total, 1.0e-30),
        )
        if len(guidance.direction) != state.model_config.feature_dim:
            raise RuntimeError("NS guidance dimension does not match the Generator")
        if progress:
            progress(
                f"Fitted solver-error guidance from {len(rows)} scored Cases",
                1.0,
            )
        return guidance

    def generate(
        self,
        base_state: object,
        guidance_state: object | None,
        request: GenerationRequest,
        progress=None,
    ) -> tuple[CaseRecord, ...]:
        state = _expect_state(base_state)
        # The canonical Controller owns random seed derivation.  Do not mix in
        # mutable call order here: identical GenerationRequests must describe
        # identical scientific draws.
        seed = int(request.seed)
        rng = np.random.default_rng(seed)
        if request.kind is GenerationKind.PERTURBED:
            latents, parents = self._perturb_latents(state, request, rng)
        else:
            guidance = (
                guidance_state
                if request.kind is GenerationKind.GUIDED
                and isinstance(guidance_state, NSLatentGuidance)
                else None
            )
            latents = sample_latent_rectified_flow(
                state.model,
                state.model_config,
                request.count,
                steps=int(request.settings.get("integration_steps", 64)),
                seed=seed,
                device=str(request.settings.get("device", "auto")),
                guidance=guidance,
                guidance_strength=float(
                    request.settings.get("guidance_strength", 0.15)
                ),
                guidance_max_flow_ratio=float(
                    request.settings.get("guidance_max_flow_ratio", 0.25)
                ),
            )
            latents = _mix_flow_proposals_with_training(
                latents,
                state.train_latents,
                state.flow_proposal_fraction,
                rng,
            )
            parents = (None,) * request.count
        latents, velocity, adjusted = _enforce_generated_validity(
            state,
            latents,
        )
        records = tuple(
            self._record(
                state,
                latent=latents[index],
                validity_adjusted=bool(adjusted[index]),
                velocity=velocity[index],
                request=request,
                parent=parents[index],
            )
            for index in range(request.count)
        )
        if progress:
            progress(
                f"Generated {request.count} {request.kind.value} NS Cases; "
                f"{int(np.count_nonzero(adjusted))} adjusted to measured support",
                1.0,
            )
        return records

    def _perturb_latents(self, state, request, rng):
        if not request.anchors:
            raise ValueError("perturbed NS generation requires anchor Cases")
        scale = float(request.settings.get("perturbation_scale", 0.15))
        rows = []
        parents = []
        for index in range(request.count):
            parent = request.anchors[index % len(request.anchors)]
            latent = self._latents_by_case_id.get(parent.case_id)
            if latent is None:
                raise ValueError(
                    f"Generator has no private coefficient for anchor {parent.case_id}"
                )
            direction = rng.normal(size=len(latent))
            direction /= max(float(np.linalg.norm(direction)), 1.0e-12)
            magnitude = scale * math.sqrt(len(latent))
            rows.append(latent + magnitude * direction)
            parents.append(parent)
        return np.asarray(rows, dtype=np.float32), tuple(parents)

    def _record(
        self,
        state,
        *,
        latent,
        velocity,
        request,
        parent,
        validity_adjusted,
    ):
        public_values = velocity[
            np.ix_(state.observed_frames, (0, 1), state.sensor_indices)
        ]
        self._serial += 1
        case_id = (
            f"ns-field-r{request.round_index:02d}-{request.kind.value}-"
            f"{self._serial:06d}"
        )
        source = (
            CaseSource.PERTURBED
            if request.kind is GenerationKind.PERTURBED
            else CaseSource.GENERATED
        )
        public_data = {
            "equation": "2D incompressible Navier--Stokes",
            "geometry": dict(state.geometry),
            "coordinates": {
                "x": tuple(float(value) for value in state.x),
                "y": tuple(float(value) for value in state.y),
                "time": tuple(float(value) for value in state.times),
            },
            "observations": {
                "fields": ("velocity_x", "velocity_y"),
                "values": public_values.tolist(),
                "sensor_indices": tuple(
                    int(value) for value in state.sensor_indices
                ),
                "observed_frames": tuple(
                    int(value) for value in state.observed_frames
                ),
                "noise_standard_deviation": tuple(
                    float(value) for value in state.noise_standard_deviation
                ),
            },
            "generator": {
                "kind": request.kind.value,
                "mode": request.mode.value,
                "validity_adjusted": bool(validity_adjusted),
                "flow_proposal_fraction": float(
                    state.flow_proposal_fraction
                ),
            },
            "evaluation_reference": "generated_complete_field",
        }
        if state.project_reconstruction is not None:
            public_data["project_reconstruction"] = state.project_reconstruction
        case = Case(
            case_id=case_id,
            source=source,
            visibility=request.visibility,
            public_data=public_data,
            round_index=request.round_index,
            parent_case_id=parent.case_id if parent is not None else None,
            tags=("ns", "generated_field", request.kind.value),
        )
        self._latents_by_case_id[case_id] = np.asarray(latent, dtype=np.float32)
        return CaseRecord(
            case,
            NSPointCloudArrayTruth(velocity, state.x, state.y, state.times),
        )


def build_latent_flow_model(config: LatentFlowConfig):
    """Create the small Torch model lazily so importing the NS plugin is cheap."""

    import torch

    config.validate()

    class FlowNetwork(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            input_width = config.feature_dim + config.time_features
            layers: list[torch.nn.Module] = [
                torch.nn.Linear(input_width, config.width),
                torch.nn.SiLU(),
            ]
            for _ in range(config.depth - 1):
                layers.extend(
                    (torch.nn.Linear(config.width, config.width), torch.nn.SiLU())
                )
            layers.append(torch.nn.Linear(config.width, config.feature_dim))
            self.network = torch.nn.Sequential(*layers)
            torch.nn.init.zeros_(self.network[-1].weight)
            torch.nn.init.zeros_(self.network[-1].bias)

        def forward(self, values, time):
            embedding = _time_embedding(torch, time, config.time_features)
            return self.network(torch.cat((values, embedding), dim=1))

    return FlowNetwork()


def train_latent_rectified_flow(
    training_latents: np.ndarray,
    validation_latents: np.ndarray,
    *,
    config: LatentFlowConfig,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: str,
    checkpoint_sample_count: int = 256,
    checkpoint_integration_steps: int = 32,
    checkpoint_projection_count: int = 128,
    progress=None,
) -> tuple[object, list[Mapping[str, float]]]:
    """Train the Flow and restore the checkpoint that generates the best samples.

    Every candidate sees the same validation paths, sampling noise, and sliced
    projections. Selection is driven mainly by the distribution produced
    after ODE integration; held-out flow-matching loss is a supporting signal.
    """

    import torch

    if (
        steps < 1
        or batch_size < 1
        or learning_rate <= 0.0
        or checkpoint_sample_count < 2
        or checkpoint_integration_steps < 1
        or checkpoint_projection_count < 1
    ):
        raise ValueError("Rectified Flow training settings must be positive")
    resolved_device = _device(torch, device)
    torch.manual_seed(seed)
    if resolved_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = build_latent_flow_model(config).to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train = torch.as_tensor(
        training_latents, dtype=torch.float32, device=resolved_device
    )
    validation = torch.as_tensor(
        validation_latents, dtype=torch.float32, device=resolved_device
    )
    generator = torch.Generator(device=resolved_device).manual_seed(seed + 1)
    validation_seed = seed + 2
    history: list[dict[str, float]] = []
    candidate_states: dict[int, dict[str, object]] = {}
    report_every = max(1, steps // 20)
    for step in range(1, steps + 1):
        indices = torch.randint(
            len(train), (batch_size,), device=resolved_device, generator=generator
        )
        clean = train[indices]
        loss = _flow_matching_loss(torch, model, clean, generator)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % report_every == 0 or step == steps:
            with torch.no_grad():
                validation_loss = _deterministic_validation_loss(
                    torch, model, validation, validation_seed
                )
            row = {
                "step": float(step),
                "train_loss": float(loss.detach().cpu()),
                "validation_loss": float(validation_loss),
                "selected": 0.0,
            }
            candidate_states[step] = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            history.append(row)
            if progress:
                progress(
                    f"Rectified Flow step {step}/{steps}; held-out loss "
                    f"{validation_loss:.4f}",
                    0.08 + 0.62 * step / steps,
                )
    if not candidate_states:
        raise RuntimeError("Rectified Flow training produced no validation checkpoint")

    validation_reference = np.asarray(validation_latents, dtype=np.float32)
    candidate_count = len(history)
    for index, row in enumerate(history):
        candidate_step = int(row["step"])
        model.load_state_dict(candidate_states[candidate_step])
        quality = _flow_checkpoint_quality(
            model,
            config,
            validation_reference,
            sample_count=checkpoint_sample_count,
            integration_steps=checkpoint_integration_steps,
            projection_count=checkpoint_projection_count,
            seed=seed + 3,
            device=device,
        )
        row.update(quality)
        if progress:
            progress(
                f"Compared generated samples from Flow step {candidate_step}",
                0.70 + 0.26 * (index + 1) / candidate_count,
            )

    selected = _select_flow_checkpoint(history)
    selected_step = int(selected["step"])
    model.load_state_dict(candidate_states[selected_step])
    model.eval()
    return model, history


def sample_latent_rectified_flow(
    model,
    config: LatentFlowConfig,
    count: int,
    *,
    steps: int,
    seed: int,
    device: str,
    guidance: NSLatentGuidance | None,
    guidance_strength: float,
    guidance_max_flow_ratio: float,
) -> np.ndarray:
    """Integrate the learned flow and optionally add bounded error guidance."""

    import torch

    if count < 1 or steps < 1:
        raise ValueError("Rectified Flow sample count and steps must be positive")
    resolved_device = _device(torch, device)
    model = model.to(resolved_device).eval()
    generator = torch.Generator(device=resolved_device).manual_seed(seed)
    values = torch.randn(
        (count, config.feature_dim),
        generator=generator,
        device=resolved_device,
    )
    direction = None
    if guidance is not None:
        direction = torch.as_tensor(
            guidance.direction, dtype=values.dtype, device=resolved_device
        )[None, :]
    step_size = 1.0 / steps
    with torch.no_grad():
        for index in range(steps):
            time = torch.full(
                (count,), index / steps, dtype=values.dtype, device=resolved_device
            )
            velocity = model(values, time)
            velocity = _guided_velocity(
                torch,
                velocity,
                direction,
                strength=guidance_strength,
                maximum_ratio=guidance_max_flow_ratio,
            )
            values = values + step_size * velocity
    return values.detach().cpu().numpy().astype(np.float32)


def _flow_matching_loss(torch, model, clean, generator):
    noise = torch.randn(
        clean.shape, dtype=clean.dtype, device=clean.device, generator=generator
    )
    time = torch.rand(
        len(clean), dtype=clean.dtype, device=clean.device, generator=generator
    )
    interpolated = (1.0 - time[:, None]) * noise + time[:, None] * clean
    return torch.nn.functional.mse_loss(model(interpolated, time), clean - noise)


def _deterministic_validation_loss(torch, model, clean, seed, repeats=4):
    """Compare checkpoints on the same small bank of validation paths."""

    if repeats < 1:
        raise ValueError("validation repeats must be positive")
    losses = []
    for index in range(repeats):
        generator = torch.Generator(device=clean.device).manual_seed(seed + index)
        loss = _flow_matching_loss(torch, model, clean, generator)
        losses.append(float(loss.cpu()))
    return sum(losses) / len(losses)


def _flow_checkpoint_quality(
    model,
    config: LatentFlowConfig,
    reference: np.ndarray,
    *,
    sample_count: int,
    integration_steps: int,
    projection_count: int,
    seed: int,
    device: str,
) -> dict[str, float]:
    """Measure what a checkpoint generates, not only its training objective."""

    generated = sample_latent_rectified_flow(
        model,
        config,
        sample_count,
        steps=integration_steps,
        seed=seed,
        device=device,
        guidance=None,
        guidance_strength=0.0,
        guidance_max_flow_ratio=1.0,
    )
    return _latent_distribution_quality(
        reference,
        generated,
        projection_count=projection_count,
        seed=seed + 1,
    )


def _latent_distribution_quality(
    reference: np.ndarray,
    generated: np.ndarray,
    *,
    projection_count: int,
    seed: int,
) -> dict[str, float]:
    """Compare generated and held-out coefficients with fixed, readable metrics.

    Sliced Wasserstein compares many one-dimensional views; MMD compares the
    full point clouds; moments and modal energy catch wrong amplitude; coverage
    checks whether each held-out example has generated neighbours. Because POD
    decoding is linear, coefficient mismatch is also field-manifold mismatch.
    """

    expected = np.asarray(reference, dtype=np.float64)
    samples = np.asarray(generated, dtype=np.float64)
    if (
        expected.ndim != 2
        or samples.ndim != 2
        or expected.shape[1] != samples.shape[1]
        or len(expected) < 2
        or len(samples) < 2
    ):
        raise ValueError("Flow sample-quality arrays must be compatible 2D batches")
    finite_fraction = float(np.mean(np.all(np.isfinite(samples), axis=1)))
    if finite_fraction < 1.0:
        raise RuntimeError("a Flow checkpoint generated non-finite coefficients")

    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(projection_count, expected.shape[1]))
    directions /= np.maximum(
        np.linalg.norm(directions, axis=1, keepdims=True),
        1.0e-12,
    )
    expected_projection = expected @ directions.T
    sample_projection = samples @ directions.T
    quantiles = np.linspace(0.0, 1.0, 129)
    projected_difference = np.quantile(
        expected_projection, quantiles, axis=0
    ) - np.quantile(sample_projection, quantiles, axis=0)
    sliced_wasserstein = float(np.mean(np.abs(projected_difference)))

    mean_error = float(
        np.sqrt(np.mean(np.square(expected.mean(axis=0) - samples.mean(axis=0))))
    )
    standard_deviation_error = float(
        np.sqrt(np.mean(np.square(expected.std(axis=0) - samples.std(axis=0))))
    )
    moment_error = mean_error + standard_deviation_error
    expected_energy = np.sqrt(np.mean(np.square(expected), axis=1))
    sample_energy = np.sqrt(np.mean(np.square(samples), axis=1))
    energy_quantiles = (0.1, 0.5, 0.9)
    energy_quantile_error = float(
        np.sqrt(
            np.mean(
                np.square(
                    np.quantile(expected_energy, energy_quantiles)
                    - np.quantile(sample_energy, energy_quantiles)
                )
            )
        )
    )

    expected_distances = _squared_distances(expected, expected)
    positive_distances = expected_distances[expected_distances > 1.0e-12]
    bandwidth_squared = (
        float(np.median(positive_distances))
        if len(positive_distances)
        else float(expected.shape[1])
    )
    bandwidth_squared = max(bandwidth_squared, 1.0e-12)
    sample_distances = _squared_distances(samples, samples)
    cross_distances = _squared_distances(expected, samples)
    latent_mmd = float(
        max(
            0.0,
            np.exp(-expected_distances / (2.0 * bandwidth_squared)).mean()
            + np.exp(-sample_distances / (2.0 * bandwidth_squared)).mean()
            - 2.0
            * np.exp(-cross_distances / (2.0 * bandwidth_squared)).mean(),
        )
    )

    neighbour_count = min(3, len(expected) - 1)
    neighbour_distances = expected_distances.copy()
    np.fill_diagonal(neighbour_distances, np.inf)
    radii = np.partition(
        neighbour_distances,
        neighbour_count - 1,
        axis=1,
    )[:, neighbour_count - 1]
    coverage = float(np.mean(np.min(cross_distances, axis=1) <= radii))
    return {
        "latent_sliced_wasserstein": sliced_wasserstein,
        "latent_mmd": latent_mmd,
        "latent_mean_error": mean_error,
        "latent_standard_deviation_error": standard_deviation_error,
        "latent_moment_error": moment_error,
        "latent_energy_quantile_error": energy_quantile_error,
        "latent_coverage": coverage,
        "latent_coverage_error": 1.0 - coverage,
        "generated_norm_q99": float(
            np.quantile(np.linalg.norm(samples, axis=1), 0.99)
        ),
        "finite_sample_fraction": finite_fraction,
    }


def _squared_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    distances = (
        np.sum(np.square(left), axis=1)[:, None]
        + np.sum(np.square(right), axis=1)[None, :]
        - 2.0 * left @ right.T
    )
    return np.maximum(distances, 0.0)


def _select_flow_checkpoint(
    history: list[dict[str, float]],
) -> dict[str, float]:
    """Choose a candidate by weighted metric ranks on one fixed cohort.

    A rank of one means best for that metric. The weighted sum makes generated
    distribution quality primary while retaining validation loss as a check.
    """

    if not history:
        raise ValueError("Flow checkpoint selection needs candidates")
    for metric, weight in FLOW_CHECKPOINT_SELECTION_WEIGHTS.items():
        ordered = sorted(history, key=lambda row: float(row[metric]))
        previous_value = None
        previous_rank = 0
        for position, row in enumerate(ordered, 1):
            value = float(row[metric])
            rank = previous_rank if previous_value == value else position
            row[f"{metric}_rank"] = float(rank)
            row["selection_rank_score"] = row.get("selection_rank_score", 0.0) + (
                weight * rank
            )
            previous_value = value
            previous_rank = rank
    selected = min(
        history,
        key=lambda row: (
            float(row["selection_rank_score"]),
            float(row["latent_sliced_wasserstein"]),
            float(row["validation_loss"]),
            -float(row["step"]),
        ),
    )
    selected["selected"] = 1.0
    return selected


def _guided_velocity(torch, velocity, direction, *, strength, maximum_ratio):
    if direction is None or strength == 0.0:
        return velocity
    if strength < 0.0 or maximum_ratio <= 0.0:
        raise ValueError("guidance strength/cap is invalid")
    base_rms = velocity.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1.0e-8)
    direction_rms = direction.square().mean(dim=1, keepdim=True).sqrt().clamp_min(
        1.0e-8
    )
    proposed = strength * direction / direction_rms * base_rms
    ratio = proposed.square().mean(dim=1, keepdim=True).sqrt() / base_rms
    cap = torch.clamp(maximum_ratio / ratio.clamp_min(1.0e-8), max=1.0)
    return velocity + proposed * cap


def _time_embedding(torch, time, width):
    half = width // 2
    frequencies = torch.exp(
        math.log(1_000.0)
        * torch.arange(half, dtype=time.dtype, device=time.device)
        / max(half - 1, 1)
    )
    angles = 2.0 * math.pi * time[:, None] * frequencies[None, :]
    return torch.cat((torch.sin(angles), torch.cos(angles)), dim=1)


def _device(torch, requested: str):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for the NS Generator but is unavailable")
    return torch.device(requested)


def _basis_rank_candidates(
    settings: Mapping[str, object],
    available_rank: int,
) -> tuple[int, ...]:
    """Return dataset-configured candidate ranks supported by the POD."""

    raw = settings.get("basis_candidates")
    if raw is None:
        raw = (settings.get("basis_size", available_rank),)
    if not isinstance(raw, (list, tuple)):
        raise ValueError("generator.basis_candidates must be an array of integers")
    candidates = sorted(
        {
            int(value)
            for value in raw
            if 0 < int(value) <= int(available_rank)
        }
    )
    if not candidates:
        raise ValueError(
            "Generator basis candidates must contain a positive rank supported "
            "by the dataset representation"
        )
    return tuple(candidates)


def _select_generator_basis_rank(
    *,
    prior: TrajectoryPODPrior,
    candidate_ranks: tuple[int, ...],
    fit_latents: np.ndarray,
    validation_latents: np.ndarray,
    validation_velocity: np.ndarray,
    settings: Mapping[str, object],
    rng: np.random.Generator,
) -> tuple[int, Mapping[str, object]]:
    """Choose the most detailed rank that still matches held-out data support.

    High POD modes often contain experiment noise. Reconstruction error alone
    therefore keeps asking for more modes, while a Generator becomes unable to
    cover the resulting sparse high-dimensional cloud. We compare real
    training coefficients with isolated validation coefficients before fitting
    the Flow and retain the highest rank that passes both requirements.
    """

    thresholds = _mapping(settings.get("validation_thresholds"))
    maximum_representation_error = float(
        thresholds.get("maximum_representation_relative_l2", 0.12)
    )
    minimum_coverage = float(thresholds.get("minimum_latent_coverage", 0.70))
    sample_count = min(
        len(fit_latents),
        int(settings.get("rank_selection_sample_count", 512)),
    )
    if sample_count < 2:
        raise ValueError("Generator rank selection needs at least two fit Cases")
    selected_indices = rng.choice(len(fit_latents), sample_count, replace=False)
    reference_energy = np.sum(
        np.square(validation_velocity, dtype=np.float64),
        axis=(1, 2, 3),
    )
    candidates: list[dict[str, object]] = []
    for rank in candidate_ranks:
        projected = prior.with_rank(rank).decode(validation_latents[:, :rank])
        representation_error = np.sqrt(
            np.sum(
                np.square(projected - validation_velocity, dtype=np.float64),
                axis=(1, 2, 3),
            )
            / np.maximum(reference_energy, 1.0e-30)
        )
        quality = _latent_distribution_quality(
            validation_latents[:, :rank],
            fit_latents[selected_indices, :rank],
            projection_count=min(128, max(16, rank)),
            seed=_settings_seed(settings) + 505,
        )
        representation_median = float(np.median(representation_error))
        coverage = float(quality["latent_coverage"])
        passed = (
            representation_median <= maximum_representation_error
            and coverage >= minimum_coverage
        )
        candidates.append(
            {
                "rank": int(rank),
                "representation_relative_l2_median": representation_median,
                "empirical_latent_coverage": coverage,
                "empirical_latent_sliced_wasserstein": float(
                    quality["latent_sliced_wasserstein"]
                ),
                "passed": passed,
            }
        )
    passing = [row for row in candidates if bool(row["passed"])]
    if passing:
        selected = max(passing, key=lambda row: int(row["rank"]))
    else:
        selected = min(
            candidates,
            key=lambda row: (
                int(
                    float(row["representation_relative_l2_median"])
                    > maximum_representation_error
                )
                + int(float(row["empirical_latent_coverage"]) < minimum_coverage),
                max(
                    float(row["representation_relative_l2_median"])
                    / max(maximum_representation_error, 1.0e-12),
                    1.0,
                )
                + max(
                    minimum_coverage
                    / max(float(row["empirical_latent_coverage"]), 1.0e-12),
                    1.0,
                ),
            ),
        )
    return int(selected["rank"]), {
        "schema": "ns_generator_basis_selection_v1",
        "criterion": "highest rank passing representation and empirical coverage",
        "selected_rank": int(selected["rank"]),
        "selection_passed": bool(selected["passed"]),
        "fit_case_count": int(len(fit_latents)),
        "validation_case_count": int(len(validation_latents)),
        "sampled_fit_case_count": int(sample_count),
        "thresholds": {
            "maximum_representation_relative_l2": maximum_representation_error,
            "minimum_empirical_latent_coverage": minimum_coverage,
        },
        "candidates": candidates,
    }


def _flow_proposal_fraction_candidates(
    settings: Mapping[str, object],
) -> tuple[float, ...]:
    raw = settings.get("flow_proposal_fractions", (1.0,))
    if not isinstance(raw, (list, tuple)):
        raise ValueError(
            "generator.flow_proposal_fractions must be an array of numbers"
        )
    candidates = sorted({float(value) for value in raw})
    if not candidates or candidates[0] <= 0.0 or candidates[-1] > 1.0:
        raise ValueError("Flow proposal fractions must lie in (0, 1]")
    return tuple(candidates)


def _mix_flow_proposals_with_training(
    proposals: np.ndarray,
    training_latents: np.ndarray,
    flow_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Shrink Flow proposals toward uniformly sampled training-manifold anchors."""

    proposed = np.asarray(proposals, dtype=np.float32)
    training = np.asarray(training_latents, dtype=np.float32)
    if (
        proposed.ndim != 2
        or training.ndim != 2
        or proposed.shape[1] != training.shape[1]
        or len(training) < 2
    ):
        raise ValueError("Flow proposals and training coefficients must be compatible")
    fraction = float(flow_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("Flow proposal fraction must lie in (0, 1]")
    indices = rng.choice(
        len(training),
        size=len(proposed),
        replace=len(proposed) > len(training),
    )
    anchors = training[indices]
    return np.asarray(
        anchors + fraction * (proposed - anchors),
        dtype=np.float32,
    )


def _select_flow_manifold_mix(
    *,
    prior: TrajectoryPODPrior,
    training_latents: np.ndarray,
    validation_latents: np.ndarray,
    validation_velocity: np.ndarray,
    projected_validation: np.ndarray,
    flow_proposals: np.ndarray,
    sampling_hz: float,
    settings: Mapping[str, object],
    rng: np.random.Generator,
    progress=None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
    Mapping[str, object],
    Mapping[str, object],
]:
    """Select the largest Flow contribution that passes physical validation.

    A finite experimental dataset can occupy a thin nonlinear manifold even
    after POD whitening. The Rectified Flow remains the global proposal, while
    a training-only anchor prevents unsupported combinations of POD modes.
    Validation selects how far each new sample may move away from that anchor.
    """

    fractions = _flow_proposal_fraction_candidates(settings)
    proposed = np.asarray(flow_proposals, dtype=np.float32)
    training = np.asarray(training_latents, dtype=np.float32)
    if (
        proposed.ndim != 2
        or training.ndim != 2
        or proposed.shape[1] != training.shape[1]
        or len(training) < 2
    ):
        raise ValueError("Flow proposals and training coefficients must be compatible")
    # Use the same uniformly sampled training anchors for the empirical baseline
    # and every candidate. Differences then come only from the Flow fraction.
    anchor_indices = rng.choice(
        len(training),
        size=len(proposed),
        replace=len(proposed) > len(training),
    )
    anchors = training[anchor_indices]
    anchor_velocity = prior.decode(anchors).astype(np.float32)
    baseline_report = _generator_field_validation(
        reference_velocity=validation_velocity,
        projected_reference=projected_validation,
        generated_velocity=anchor_velocity,
        reference_latents=validation_latents,
        generated_latents=anchors,
        sampling_hz=sampling_hz,
        settings=settings,
    )
    calibrated_settings, threshold_calibration = (
        _calibrate_generator_validation_settings(
            settings,
            baseline_report,
        )
    )
    candidates: list[dict[str, object]] = []
    selected_output = None
    selected_key = None
    for candidate_index, fraction in enumerate(fractions):
        mixed = np.asarray(
            anchors + fraction * (proposed - anchors),
            dtype=np.float32,
        )
        velocity = prior.decode(mixed).astype(np.float32)
        report = _generator_field_validation(
            reference_velocity=validation_velocity,
            projected_reference=projected_validation,
            generated_velocity=velocity,
            reference_latents=validation_latents,
            generated_latents=mixed,
            sampling_hz=sampling_hz,
            settings=calibrated_settings,
        )
        accepted = bool(report["accepted"])
        penalty = _generator_validation_penalty(report)
        row = {
            "flow_proposal_fraction": float(fraction),
            "accepted": accepted,
            "failed_checks": list(report["failed_checks"]),
            "penalty": float(penalty),
            "metrics": dict(report["metrics"]),
            "scalar_quantile_distances": dict(
                report["scalar_quantile_distances"]
            ),
        }
        candidates.append(row)
        if progress:
            progress(
                f"Validated Flow contribution {fraction:g}",
                0.96 + 0.035 * (candidate_index + 1) / len(fractions),
            )
        key = (
            0 if accepted else 1,
            -float(fraction) if accepted else float(penalty),
            -float(fraction),
        )
        if selected_key is None or key < selected_key:
            selected_key = key
            selected_output = (
                mixed,
                velocity,
                float(fraction),
                report,
            )
    if selected_output is None:
        raise RuntimeError("Flow manifold selection produced no candidates")
    mixed, velocity, fraction, report = selected_output
    selection = {
        "schema": "ns_generator_flow_manifold_mix_v1",
        "criterion": "largest Flow fraction passing measured-field validation",
        "selected_flow_proposal_fraction": fraction,
        "selected_accepted": bool(report["accepted"]),
        "anchor_source": "generator_training_coefficients",
        "empirical_training_baseline": {
            "metrics": dict(baseline_report["metrics"]),
            "scalar_quantile_distances": dict(
                baseline_report["scalar_quantile_distances"]
            ),
        },
        "threshold_calibration": threshold_calibration,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    return mixed, velocity, fraction, report, selection


def _calibrate_generator_validation_settings(
    settings: Mapping[str, object],
    baseline_report: Mapping[str, object],
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    """Calibrate distribution gates against real train-to-validation shift."""

    metrics = _mapping(baseline_report.get("metrics"))
    configured = {
        name: float(value)
        for name, value in _mapping(baseline_report.get("thresholds")).items()
    }
    baseline_factor = float(settings.get("validation_baseline_factor", 1.15))
    maximum_multiplier = float(
        settings.get("validation_baseline_maximum_multiplier", 2.0)
    )
    coverage_factor = float(
        settings.get("validation_baseline_coverage_factor", 0.90)
    )
    if (
        baseline_factor < 1.0
        or maximum_multiplier < 1.0
        or not 0.0 < coverage_factor <= 1.0
    ):
        raise ValueError("Generator validation baseline factors are invalid")
    calibrated = dict(configured)
    upper_pairs = (
        ("mean_field_relative_l2", "maximum_mean_field_relative_l2"),
        (
            "standard_deviation_field_relative_l2",
            "maximum_standard_deviation_field_relative_l2",
        ),
        ("maximum_scalar_quantile_distance", "maximum_scalar_quantile_distance"),
        ("v_temporal_spectrum_l1", "maximum_v_temporal_spectrum_l1"),
        (
            "tail_outside_reference_fraction",
            "maximum_tail_outside_reference_fraction",
        ),
    )
    for metric_name, threshold_name in upper_pairs:
        fixed = configured[threshold_name]
        empirical = float(metrics[metric_name]) * baseline_factor
        calibrated[threshold_name] = min(
            fixed * maximum_multiplier,
            max(fixed, empirical),
        )
    fixed_coverage = configured["minimum_latent_coverage"]
    empirical_coverage = float(metrics["latent_coverage"]) * coverage_factor
    calibrated["minimum_latent_coverage"] = max(
        fixed_coverage / maximum_multiplier,
        min(fixed_coverage, empirical_coverage),
    )
    calibrated_settings = dict(settings)
    calibrated_settings["validation_thresholds"] = calibrated
    return calibrated_settings, {
        "schema": "ns_generator_empirical_threshold_calibration_v1",
        "baseline_factor": baseline_factor,
        "maximum_multiplier": maximum_multiplier,
        "coverage_factor": coverage_factor,
        "configured_thresholds": configured,
        "calibrated_thresholds": calibrated,
    }


def _generator_validation_penalty(report: Mapping[str, object]) -> float:
    metrics = _mapping(report.get("metrics"))
    thresholds = _mapping(report.get("thresholds"))
    upper_pairs = (
        ("representation_relative_l2_median", "maximum_representation_relative_l2"),
        ("mean_field_relative_l2", "maximum_mean_field_relative_l2"),
        (
            "standard_deviation_field_relative_l2",
            "maximum_standard_deviation_field_relative_l2",
        ),
        ("maximum_scalar_quantile_distance", "maximum_scalar_quantile_distance"),
        ("v_temporal_spectrum_l1", "maximum_v_temporal_spectrum_l1"),
        (
            "tail_outside_reference_fraction",
            "maximum_tail_outside_reference_fraction",
        ),
    )
    ratios = [
        float(metrics[name]) / max(float(thresholds[limit]), 1.0e-12)
        for name, limit in upper_pairs
    ]
    ratios.append(
        float(thresholds["minimum_latent_coverage"])
        / max(float(metrics["latent_coverage"]), 1.0e-12)
    )
    ratios.append(
        float(thresholds["minimum_finite_fraction"])
        / max(float(metrics["finite_fraction"]), 1.0e-12)
    )
    return float(max(ratios))


def _build_field_validity_profile(
    reference_velocity: np.ndarray,
    settings: Mapping[str, object],
    *additional_velocity: np.ndarray,
) -> Mapping[str, object]:
    """Build geometry-independent bounds from representable data windows.

    Runtime samples are pulled toward the training manifold when a Flow or
    guided proposal leaves measured support. The validity profile therefore
    covers both the training manifold and independent validation manifold;
    otherwise a genuine training window could be rejected as a fallback.
    """

    cohorts = tuple(
        np.asarray(values, dtype=np.float64)
        for values in (reference_velocity, *additional_velocity)
        if len(values)
    )
    if not cohorts:
        raise ValueError("field validity needs at least one reference Case")
    feature_batches = tuple(_window_scalar_features(values) for values in cohorts)
    features = {
        name: np.concatenate([batch[name] for batch in feature_batches])
        for name in feature_batches[0]
    }
    margin = float(settings.get("validity_margin", 0.35))
    bounds = {}
    for name, values in features.items():
        low, high = np.quantile(values, (0.005, 0.995))
        span = max(float(high - low), 0.1 * abs(float(np.median(values))), 1.0e-8)
        bounds[name] = [
            max(0.0, float(low) - margin * span),
            float(high) + margin * span,
        ]
    return {
        "reference_case_count": int(sum(len(values) for values in cohorts)),
        "cohort_case_counts": [int(len(values)) for values in cohorts],
        "quantiles": [0.005, 0.995],
        "margin": margin,
        "bounds": bounds,
    }


def _enforce_generated_validity(state, latents):
    """Keep base, guided, and perturbed samples inside measured support.

    Outlying coefficients are first clipped to the empirical latent envelope.
    A field that still violates measured RMS/energy/roughness bounds is moved
    toward its nearest training coefficient until it becomes valid.
    """

    original = np.asarray(latents, dtype=np.float32)
    constrained = np.nan_to_num(original, nan=0.0, posinf=0.0, neginf=0.0)
    low = np.quantile(state.train_latents, 0.001, axis=0)
    high = np.quantile(state.train_latents, 0.999, axis=0)
    span = np.maximum(high - low, 1.0e-6)
    constrained = np.clip(constrained, low - 0.2 * span, high + 0.2 * span)
    norms = np.linalg.norm(constrained, axis=1)
    maximum_norm = 1.15 * float(np.quantile(np.linalg.norm(state.train_latents, axis=1), 0.995))
    large = norms > maximum_norm
    if np.any(large):
        constrained[large] *= (maximum_norm / norms[large])[:, None]

    velocity = state.prior.decode(constrained).astype(np.float32)
    valid = _field_validity_mask(velocity, state.validity_profile)
    adjusted = np.any(np.abs(constrained - original) > 1.0e-6, axis=1)
    if np.all(valid):
        return constrained, velocity, adjusted

    anchors = np.asarray(state.valid_anchor_latents, dtype=np.float32)
    scale = np.std(state.train_latents, axis=0)
    scale = np.maximum(scale, 1.0e-3)
    for index in np.flatnonzero(~valid):
        distances = np.mean(
            np.square((anchors - constrained[index]) / scale),
            axis=1,
        )
        anchor = anchors[int(np.argmin(distances))]
        candidate = constrained[index]
        for retained_fraction in (0.75, 0.5, 0.25, 0.0):
            proposed = retained_fraction * candidate + (1.0 - retained_fraction) * anchor
            proposed_velocity = state.prior.decode(proposed).astype(np.float32)
            if _field_validity_mask(
                proposed_velocity[None, ...],
                state.validity_profile,
            )[0]:
                constrained[index] = proposed
                velocity[index] = proposed_velocity
                valid[index] = True
                adjusted[index] = True
                break
    if not np.all(valid):
        raise RuntimeError("Generator could not produce fields inside measured support")
    return constrained, velocity, adjusted


def _field_validity_mask(
    velocity: np.ndarray,
    profile: Mapping[str, object],
) -> np.ndarray:
    values = np.asarray(velocity, dtype=np.float64)
    finite = np.all(np.isfinite(values), axis=(1, 2, 3))
    features = _window_scalar_features(values)
    bounds = _mapping(profile.get("bounds"))
    valid = finite.copy()
    for name, rows in features.items():
        limits = bounds.get(name)
        if not isinstance(limits, (list, tuple)) or len(limits) != 2:
            continue
        valid &= rows >= float(limits[0])
        valid &= rows <= float(limits[1])
    return valid


def _generator_field_validation(
    *,
    reference_velocity: np.ndarray,
    projected_reference: np.ndarray,
    generated_velocity: np.ndarray,
    reference_latents: np.ndarray,
    generated_latents: np.ndarray,
    sampling_hz: float,
    settings: Mapping[str, object],
) -> Mapping[str, object]:
    """Check measurement fidelity, then compare generated and projected fields."""

    reference = np.asarray(reference_velocity, dtype=np.float64)
    projected = np.asarray(projected_reference, dtype=np.float64)
    generated = np.asarray(generated_velocity, dtype=np.float64)
    distribution_reference = projected
    reference_energy = np.sum(np.square(reference), axis=(1, 2, 3))
    projection_error = np.sqrt(
        np.sum(np.square(projected - reference), axis=(1, 2, 3))
        / np.maximum(reference_energy, 1.0e-30)
    )
    mean_field_error = _relative_array_error(
        generated.mean(axis=0),
        distribution_reference.mean(axis=0),
    )
    standard_deviation_field_error = _relative_array_error(
        generated.std(axis=0),
        distribution_reference.std(axis=0),
    )
    reference_features = _window_scalar_features(distribution_reference)
    generated_features = _window_scalar_features(generated)
    scalar_distances = {
        name: _normalized_quantile_distance(
            reference_features[name],
            generated_features[name],
        )
        for name in reference_features
    }
    spectrum_distance, reference_frequency, generated_frequency = _v_spectrum_distance(
        distribution_reference,
        generated,
        sampling_hz,
    )
    lower, upper = np.quantile(distribution_reference, (0.001, 0.999), axis=(0, 1, 3))
    outside = (generated < lower[None, None, :, None]) | (
        generated > upper[None, None, :, None]
    )
    latent_quality = _latent_distribution_quality(
        reference_latents,
        generated_latents,
        projection_count=min(128, max(16, reference_latents.shape[1])),
        seed=_settings_seed(settings) + 505,
    )
    finite_fraction = float(np.mean(np.isfinite(generated)))
    metrics = {
        "finite_fraction": finite_fraction,
        "representation_relative_l2_median": float(np.median(projection_error)),
        "mean_field_relative_l2": mean_field_error,
        "standard_deviation_field_relative_l2": standard_deviation_field_error,
        "measurement_mean_field_relative_l2": _relative_array_error(
            generated.mean(axis=0),
            reference.mean(axis=0),
        ),
        "measurement_standard_deviation_field_relative_l2": _relative_array_error(
            generated.std(axis=0),
            reference.std(axis=0),
        ),
        "maximum_scalar_quantile_distance": float(max(scalar_distances.values())),
        "v_temporal_spectrum_l1": spectrum_distance,
        "tail_outside_reference_fraction": float(np.mean(outside)),
        "latent_coverage": float(latent_quality["latent_coverage"]),
        "reference_dominant_v_frequency": reference_frequency,
        "generated_dominant_v_frequency": generated_frequency,
    }
    raw_thresholds = _mapping(settings.get("validation_thresholds"))
    thresholds = {
        "minimum_finite_fraction": float(
            raw_thresholds.get("minimum_finite_fraction", 1.0)
        ),
        "maximum_representation_relative_l2": float(
            raw_thresholds.get("maximum_representation_relative_l2", 0.12)
        ),
        "maximum_mean_field_relative_l2": float(
            raw_thresholds.get("maximum_mean_field_relative_l2", 0.10)
        ),
        "maximum_standard_deviation_field_relative_l2": float(
            raw_thresholds.get(
                "maximum_standard_deviation_field_relative_l2",
                0.35,
            )
        ),
        "maximum_scalar_quantile_distance": float(
            raw_thresholds.get("maximum_scalar_quantile_distance", 0.80)
        ),
        "maximum_v_temporal_spectrum_l1": float(
            raw_thresholds.get("maximum_v_temporal_spectrum_l1", 0.40)
        ),
        "maximum_tail_outside_reference_fraction": float(
            raw_thresholds.get(
                "maximum_tail_outside_reference_fraction",
                0.03,
            )
        ),
        "minimum_latent_coverage": float(
            raw_thresholds.get("minimum_latent_coverage", 0.70)
        ),
    }
    checks = {
        "finite_fraction": metrics["finite_fraction"]
        >= thresholds["minimum_finite_fraction"],
        "representation_relative_l2": metrics[
            "representation_relative_l2_median"
        ]
        <= thresholds["maximum_representation_relative_l2"],
        "mean_field": metrics["mean_field_relative_l2"]
        <= thresholds["maximum_mean_field_relative_l2"],
        "standard_deviation_field": metrics[
            "standard_deviation_field_relative_l2"
        ]
        <= thresholds["maximum_standard_deviation_field_relative_l2"],
        "scalar_distributions": metrics["maximum_scalar_quantile_distance"]
        <= thresholds["maximum_scalar_quantile_distance"],
        "temporal_spectrum": metrics["v_temporal_spectrum_l1"]
        <= thresholds["maximum_v_temporal_spectrum_l1"],
        "velocity_tails": metrics["tail_outside_reference_fraction"]
        <= thresholds["maximum_tail_outside_reference_fraction"],
        "latent_coverage": metrics["latent_coverage"]
        >= thresholds["minimum_latent_coverage"],
    }
    return {
        "schema": "ns_generator_measured_field_validation_v2",
        "measurement_reference": "raw_held_out_measurement",
        "distribution_reference": "held_out_representation_projection",
        "reference_case_count": int(len(reference)),
        "generated_case_count": int(len(generated)),
        "metrics": metrics,
        "scalar_quantile_distances": scalar_distances,
        "latent_metrics": {
            name: float(value) for name, value in latent_quality.items()
        },
        "thresholds": thresholds,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "accepted": all(checks.values()),
    }


def _window_scalar_features(velocity: np.ndarray) -> Mapping[str, np.ndarray]:
    values = np.asarray(velocity, dtype=np.float64)
    if values.ndim != 4 or values.shape[2] != 2:
        raise ValueError("velocity batch must have shape [case,time,2,point]")
    return {
        "u_rms": np.sqrt(np.mean(np.square(values[:, :, 0]), axis=(1, 2))),
        "v_rms": np.sqrt(np.mean(np.square(values[:, :, 1]), axis=(1, 2))),
        "kinetic_energy": 0.5 * np.mean(np.square(values), axis=(1, 2, 3)),
        "temporal_increment_rms": np.sqrt(
            np.mean(np.square(np.diff(values, axis=1)), axis=(1, 2, 3))
        ),
    }


def _relative_array_error(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(
        np.linalg.norm(actual - expected)
        / max(float(np.linalg.norm(expected)), 1.0e-30)
    )


def _normalized_quantile_distance(reference, generated) -> float:
    probabilities = np.linspace(0.01, 0.99, 99)
    left = np.quantile(reference, probabilities)
    right = np.quantile(generated, probabilities)
    scale = max(
        float(np.quantile(reference, 0.75) - np.quantile(reference, 0.25)),
        float(np.std(reference)),
        1.0e-12,
    )
    return float(np.mean(np.abs(left - right)) / scale)


def _v_spectrum_distance(reference, generated, sampling_hz):
    def spectrum(values):
        cross_flow = values[:, :, 1] - values[:, :, 1].mean(axis=1, keepdims=True)
        power = np.mean(
            np.abs(np.fft.rfft(cross_flow, axis=1)) ** 2,
            axis=(0, 2),
        )
        power[0] = 0.0
        return power / max(float(power.sum()), 1.0e-30)

    left = spectrum(reference)
    right = spectrum(generated)
    frequencies = np.fft.rfftfreq(reference.shape[1], d=1.0 / sampling_hz)
    return (
        float(0.5 * np.sum(np.abs(left - right))),
        float(frequencies[int(np.argmax(left))]),
        float(frequencies[int(np.argmax(right))]),
    )


def _sampling_hz(record: CaseRecord) -> float:
    coordinates = _mapping(record.case.public_data.get("coordinates"))
    times = np.asarray(coordinates.get("time"), dtype=np.float64)
    if len(times) < 2:
        return 1.0
    step = float(np.median(np.diff(times)))
    if step <= 0.0:
        raise ValueError("NS Case time coordinates must increase")
    return 1.0 / step

def _record_velocity(record: CaseRecord) -> np.ndarray:
    coordinates = _mapping(record.case.public_data.get("coordinates"))
    times = coordinates.get("time")
    x = coordinates.get("x")
    if not isinstance(times, (list, tuple)) or not isinstance(x, (list, tuple)):
        raise ValueError("completed NS Case is missing public native coordinates")
    training_target = (
        record.generator_target if record.generator_target is not None else record.truth
    )
    if isinstance(training_target, NSPointCloudTruth):
        with np.load(training_target.case_file, allow_pickle=False) as arrays:
            velocity = np.asarray(arrays["velocity"], dtype=np.float32)
        expected = (len(times), 2, len(x))
        if velocity.shape != expected:
            raise ValueError(
                f"{training_target.case_file} velocity must have shape {expected}"
            )
        return velocity
    answer = training_target.answer(
        AnswerQuery.grid(
            fields=("velocity_x", "velocity_y"),
            t=len(times),
            point=len(x),
        )
    )
    shape = (len(times), len(x))
    return np.stack(
        (
            np.asarray(answer.fields["velocity_x"], dtype=np.float32).reshape(shape),
            np.asarray(answer.fields["velocity_y"], dtype=np.float32).reshape(shape),
        ),
        axis=1,
    )


def _validate_state(state: NSFieldGeneratorState) -> None:
    if len(state.x) != state.prior.point_count or len(state.y) != len(state.x):
        raise ValueError("Generator representation and dataset geometry do not match")
    if len(state.times) != state.prior.window_length:
        raise ValueError("Generator representation and dataset time window do not match")
    if len(state.sensor_indices) == 0 or len(state.observed_frames) == 0:
        raise ValueError("Generator needs a public observation layout")
    if state.noise_standard_deviation.shape != (2,):
        raise ValueError("Generator needs one noise scale per velocity component")
    if (
        state.valid_anchor_latents.ndim != 2
        or state.valid_anchor_latents.shape[1] != state.model_config.feature_dim
        or len(state.valid_anchor_latents) < 1
    ):
        raise ValueError("Generator needs at least one valid training-manifold anchor")


def _expect_state(value: object) -> NSFieldGeneratorState:
    if not isinstance(value, NSFieldGeneratorState):
        raise TypeError("unexpected NS field Generator state")
    return value


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _project_reconstruction_from_case(
    case: Case,
) -> Mapping[str, object] | None:
    value = case.public_data.get("project_reconstruction")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("NS Project reconstruction descriptor must be a mapping")
    return dict(value)


def _settings_seed(settings: Mapping[str, object]) -> int:
    """Use the seed namespace assigned by the canonical Controller."""

    raw = settings.get("controller_seed", settings.get("seed", 20260825))
    if isinstance(raw, bool):
        raise TypeError("NS Generator seed must be an integer")
    seed = int(raw)
    if seed < 0:
        raise ValueError("NS Generator seed must be nonnegative")
    return seed


def _checkpoint_sidecar(checkpoint: Path, name: str) -> Mapping[str, object]:
    path = checkpoint.parent / name
    if not path.is_file():
        return {}
    import json

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"NS Generator checkpoint sidecar {path} must be an object")
    return value


def save_field_generator_checkpoint(
    state: NSFieldGeneratorState,
    path: str | Path,
) -> Path:
    """Save the learned flow and its measured-field validity gate."""

    import torch

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    config = state.model_config
    torch.save(
        {
            "schema": "ns_dataset_latent_flow_v2",
            "model_config": {
                "feature_dim": config.feature_dim,
                "width": config.width,
                "depth": config.depth,
                "time_features": config.time_features,
            },
            "model_state": {
                name: value.detach().cpu()
                for name, value in state.model.state_dict().items()
            },
            "training_history": [dict(row) for row in state.training_history],
            "train_latents": np.asarray(state.train_latents, dtype=np.float32),
            "valid_anchor_latents": np.asarray(
                state.valid_anchor_latents,
                dtype=np.float32,
            ),
            "validation_latents": np.asarray(
                state.validation_latents,
                dtype=np.float32,
            ),
            "validation_report": dict(state.validation_report),
            "validity_profile": dict(state.validity_profile),
            "flow_proposal_fraction": float(state.flow_proposal_fraction),
        },
        destination,
    )
    return destination


__all__ = [
    "FLOW_CHECKPOINT_SELECTION_WEIGHTS",
    "LatentFlowConfig",
    "NSDatasetCoefficientGeneratorBackend",
    "NSFieldGeneratorState",
    "NSGeneratorValidationError",
    "NSLatentGuidance",
    "build_latent_flow_model",
    "save_field_generator_checkpoint",
    "sample_latent_rectified_flow",
    "train_latent_rectified_flow",
]
