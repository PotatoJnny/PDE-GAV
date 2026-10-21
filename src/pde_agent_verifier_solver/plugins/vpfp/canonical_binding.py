"""Bind the existing weak/strong numerical models to the canonical Controller.

This module deliberately calls the online coarse-solve generator what it is.
The continuous decoder manufactures an exact answer/source pair, but its old
sampling distribution is not a learned generative model. A learned latent
sampler can be supplied at the same private boundary without changing Tasks.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
import re
from typing import Mapping

import numpy as np

from pde_agent_verifier_solver.sdk import (
    DatasetInput, DatasetKind, FormField, GeneratorCheckpoint, PreparedDataset,
)

from ..reward_guidance import RewardGuidanceConfig

from .generator import VPFPContinuousGenerator
from .perturbation_backend import (
    VPFPAnchoredPerturbationBackend, VPFPLegacyStructuredRuntime, _legacy_api,
)
from .product import build_vpfp_plugin


def runtime_decoder_path(value: str, expected_sha256: str) -> Path:
    """Resolve only a manifest-listed historical decoder locator.

    Frozen checkpoint metadata keeps its original path and SHA; this function
    changes the file opened at runtime, never the serialized prior identity.
    """
    original = Path(value).expanduser()
    asset_root = os.environ.get("VPFP_STRONG_ASSET_ROOT")
    if not asset_root:
        return original
    manifest = json.loads((Path(__file__).resolve().parents[4] / "migration/strong_assets.json").read_text())
    matches = [row for row in manifest["files"] if row.get("role") in {"base_decoder", "expanded_decoder"}
               and row["source"] == str(original)]
    if not matches:
        return original
    if len(matches) != 1 or matches[0]["sha256"] != expected_sha256:
        raise ValueError("embedded decoder locator differs from the immutable migration manifest")
    relative = Path(matches[0]["destination"])
    root = Path(asset_root)
    if not root.is_absolute() or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("strong decoder relocation must remain inside an absolute asset root")
    return root / relative


@dataclass(frozen=True)
class VPFPOnlineModelSpec:
    regime_id: str
    base_decoder_checkpoint: str
    base_decoder_sha256: str
    expanded_decoder_checkpoint: str
    expanded_decoder_sha256: str
    collision_nu_range: tuple[float, float]
    temperature_range: tuple[float, float] = (0.8, 1.2)
    time_domain: tuple[float, float] = (0.0, 1.0)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "VPFPOnlineModelSpec":
        names = set(cls.__dataclass_fields__)
        unknown = set(value) - names
        if unknown:
            raise ValueError(f"unknown VPFP model settings: {sorted(unknown)}")
        copied = dict(value)
        for name in ("collision_nu_range", "temperature_range", "time_domain"):
            if name in copied:
                copied[name] = tuple(copied[name])
        return cls(**copied)

    def preflight(self) -> Mapping[str, object]:
        if not isinstance(self.regime_id, str) or not self.regime_id.strip():
            raise ValueError("VPFP regime_id cannot be empty")
        for name in ("collision_nu_range", "temperature_range", "time_domain"):
            bounds = np.asarray(getattr(self, name), dtype=float)
            if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[1] <= bounds[0]:
                raise ValueError(f"VPFP {name} must be an increasing finite pair")
        if self.collision_nu_range[0] < 0 or self.temperature_range[0] <= 0:
            raise ValueError("VPFP collision/temperature ranges are invalid")
        if self.time_domain[0] != 0:
            raise ValueError("the qualified online VPFP models start at t=0")
        checked = {}
        for label in ("base_decoder", "expanded_decoder"):
            expected = getattr(self, f"{label}_sha256")
            raw = runtime_decoder_path(getattr(self, f"{label}_checkpoint"), expected)
            if raw.is_symlink() or not raw.is_file():
                raise FileNotFoundError(raw)
            checked[label] = {"path": str(raw.resolve()), "sha256": expected}
        return {"regime_id": self.regime_id, "checkpoints": checked}


@dataclass(frozen=True)
class LegacyMacroObservableAdapter:
    """Explicit ABI-only wrapper; it preserves the old program's query limits."""

    program: object

    def __post_init__(self):
        if not callable(getattr(self.program, "predict_macros", None)):
            raise TypeError("legacy VPFP candidate must implement predict_macros")

    def predict_observables(self, case, query, budget=None):
        return self.program.predict_macros(case, query, budget=budget)


class _BoundVPFPContinuousGenerator(VPFPContinuousGenerator):
    """Attach the verified physical identity at the product sampling boundary."""

    def __init__(self, backend, identity):
        super().__init__(backend)
        # Checkpoint payloads are caller-owned mappings. Keep an independent
        # deep snapshot so mutating nested metadata cannot alter this binding.
        self._identity = deepcopy(dict(identity))

    def setup_fields(self):
        return (
            FormField("guidance_enabled", "Experimental reward guidance", "boolean", self.capabilities.guidance),
            FormField("perturbations_enabled", "Qualified physical perturbations", "boolean", self.capabilities.perturbations),
        )

    def _check_settings(self, settings):
        # These keys are bookkeeping inserted by the shared Controller, not
        # requests to rebuild the frozen physical model. Everything else must
        # be selected explicitly at the backend factory boundary.
        allowed = {"basis_size", "master_seed", "controller_seed",
                   "fixed_generator_id", "fixed_generator_sha256",
                   "guidance_enabled", "perturbations_enabled",
                   "metric_profile", "metric_calibration"}
        unknown = set(settings) - allowed
        if unknown:
            raise ValueError("VPFP settings are frozen at backend binding; rebind to change: " + ", ".join(sorted(unknown)))
        if "metric_profile" in settings or "metric_calibration" in settings:
            # The Controller forwards the scoring settings already selected at
            # binding. Accept that identity, never a replacement objective.
            from ..plasma_transport_settings import transport_calibration
            calibration = transport_calibration(
                settings.get("metric_profile"), settings.get("metric_calibration"))
            objective = None if calibration is None else calibration.objective_identity
            if objective != self._identity.get("scoring_objective"):
                raise ValueError("VPFP settings differ from the bound scoring objective")
        for name, available in (("guidance_enabled", self.capabilities.guidance),
                                ("perturbations_enabled", self.capabilities.perturbations)):
            if name in settings and not isinstance(settings[name], bool):
                raise ValueError(f"VPFP {name} must be boolean")
            if settings.get(name) is True and not available:
                raise ValueError(f"VPFP bound backend does not support {name}")

    def load(self, checkpoint, settings, progress=None):
        self._check_settings(settings)
        if checkpoint.payload != self._identity:
            raise ValueError("VPFP checkpoint does not match the bound physical runtime")
        return super().load(checkpoint, settings, progress)

    def public_session_operation_lock(self, state, case):
        """Keep the global Controller lock unless a hidden GPU scope is active."""
        runtime = self._backend().runtime
        provider = getattr(runtime, "public_session_operation_lock", None)
        return None if not callable(provider) else provider(case)

    def sample(self, state, request, progress=None):
        self._check_settings(request.settings)
        if state.checkpoint.payload != self._identity or state.base != self._identity:
            raise ValueError("VPFP loaded state does not match the bound physical runtime")
        fixed = {"fixed_generator_id": self.info.program_id,
                 "fixed_generator_sha256": self._identity["binding_sha256"]}
        for key, value in fixed.items():
            if key in request.settings and request.settings[key] != value:
                raise ValueError(f"VPFP request changes bound {key}")
        return super().sample(state, replace(request, settings={**request.settings, **fixed}), progress)


def vpfp_public_guidance_features(case) -> tuple[float, ...]:
    """Condition and public initial moments; no hidden truth or case identity."""

    question = case.public_data["question"]
    arrays = case.public_data["arrays"]
    features = [float(question["collision_nu"]), float(question["temperature"])]
    for name in ("rho0", "momentum0", "kinetic_energy0"):
        values = np.asarray(arrays[name], dtype=np.float64)
        if not values.size or not np.isfinite(values).all():
            raise ValueError(f"VPFP guidance received invalid public {name}")
        features.extend((float(values.mean()), float(values.std()),
                         float(np.sqrt(np.mean(values * values)))))
    return tuple(features)


def build_vpfp_online_binding(
    spec: VPFPOnlineModelSpec | Mapping[str, object],
    *,
    output_root: str | Path,
    device: str = "cuda",
    guidance: bool = False,
    guidance_config: RewardGuidanceConfig | None = None,
    latent_sampler=None,
    oracle_kernel="reference_v1",
    metric_profile=None,
    metric_calibration=None,
):
    """Construct the real resident Backend and frozen canonical checkpoint.

    Construction checks files and physical identities but does not run a PDE
    solve. Sampling is a compute-node operation. ``latent_sampler`` must expose
    generate(seeds) and a JSON ``identity``; it replaces the coarse solve.
    """

    from pde_agent_verifier_solver.app.backend_run import BackendRunBinding

    if not isinstance(spec, VPFPOnlineModelSpec):
        spec = VPFPOnlineModelSpec.from_mapping(spec)
    if guidance_config is not None and not isinstance(guidance_config, RewardGuidanceConfig):
        raise TypeError("guidance_config must be RewardGuidanceConfig")
    if guidance_config is not None and not guidance:
        raise ValueError("guidance_config requires guidance=True")
    frozen_guidance_config = (guidance_config or RewardGuidanceConfig()) if guidance else None
    checked = spec.preflight()
    api = _legacy_api()  # Verify pinned physical bytes before constructing objects.
    hermite = importlib.import_module("agent_verifier.vpfp.continuous_hermite")
    online = importlib.import_module("agent_verifier.vpfp.online_training")
    high_fidelity = importlib.import_module("agent_verifier.vpfp.high_fidelity")
    operators = importlib.import_module("agent_verifier.vpfp.operators")
    base_payload = api.torch.load(spec.base_decoder_checkpoint, map_location="cpu", weights_only=True)
    adapter = dict(base_payload.get("robust_answer_first_adapter") or {})
    if (adapter.get("unused_neural_parameters_zeroed") is not True
            or adapter.get("continuous_hidden_answer_is_primary") is not True
            or adapter.get("generator_parameterization_public") is not False):
        raise ValueError("VPFP base decoder is not answer-first qualified")
    decoder = hermite.VPFPContinuousHermiteConfig.from_dict(base_payload["decoder_config"])
    for name in ("collision_nu_range", "temperature_range", "time_domain"):
        if tuple(getattr(decoder, name)) != tuple(getattr(spec, name)):
            raise ValueError(f"VPFP requested {name} differs from checkpoint")
    if int(decoder.structured_time_control_count) != 33:
        raise ValueError("VPFP online models require the qualified 33 time controls")
    generation = online.VPFPOnlineGenerationConfig(
        batch_size=16, nx=16, ny=16, velocity_grid_size=21, nt=33,
        t1=spec.time_domain[1], velocity_max=8.0, internal_dt_max=0.001,
        dtype="float32", cache_capacity_cases=128, anchor_case_count=16,
    )
    equation = high_fidelity.VPFPHighFidelityConfig(
        internal_dt_max=0.001, collision_nu_range=spec.collision_nu_range,
        temperature_range=spec.temperature_range,
    )
    if latent_sampler is None:
        sampler = online.VPFPOnlineBatchGenerator(
            decoder_config=decoder, generation_config=generation,
            equation_config=equation, device=device,
        )
        sampling_identity = {
            "kind": "online_coarse_solve_then_continuous_projection",
            "learned_generation": False, "generation": generation.to_dict(),
            "equation": equation.to_dict(),
        }
    else:
        if not callable(getattr(latent_sampler, "generate", None)):
            raise TypeError("VPFP latent_sampler must implement generate(seeds)")
        sampling_identity = dict(latent_sampler.identity)
        if sampling_identity.get("base_decoder_sha256") != spec.base_decoder_sha256:
            raise ValueError("learned VPFP sampler belongs to a different decoder")
        if tuple(sampling_identity.get("collision_nu_range", ())) != spec.collision_nu_range:
            raise ValueError("learned VPFP sampler collision regime differs")
        if tuple(sampling_identity.get("temperature_range", ())) != spec.temperature_range:
            raise ValueError("learned VPFP sampler temperature regime differs")
        if sampling_identity.get("latent_dimension") != decoder.latent_dim:
            raise ValueError("learned VPFP sampler coefficient dimension differs")
        sampler = latent_sampler
    base_model = api.build_deterministic_direct_model(decoder, device=device)
    runtime = VPFPLegacyStructuredRuntime(
        generator=sampler, base_model=base_model,
        expanded_decoder_checkpoint=spec.expanded_decoder_checkpoint,
        output_root=output_root,
        materialized_grid=operators.VPFPGrid(
            nx=33, ny=33, nvx=9, nvy=9,
            x_domain=tuple(decoder.equation.x_domain), v_bounds=(-5., 5., -5., 5.),
        ),
        materialized_time=operators.VPFPTime(t0=0., t1=spec.time_domain[1], nt=13),
        device=device, materialize_cases=False, oracle_kernel=oracle_kernel,
    )
    backend = VPFPAnchoredPerturbationBackend(runtime)
    identity = {
        "schema": "vpfp_canonical_backend_binding_v1", **checked,
        "sampling": sampling_identity,
        "source_derivation": "exact_continuous_analytic_vpfp_residual",
        "public_oracle_evaluation": {
            "schema": "vpfp_requested_coordinate_reuse_v1",
            "mode": "continuous_on_demand",
            "kernel": oracle_kernel,
            "online_result_validation": oracle_kernel == "reference_v1",
            "source_chunk_size": 8192,
            "cross_case_cache": False,
            "cartesian_completion": False,
            "implementation_sha256": {
                name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in (("batched_source.py", "perturbation_backend.py")
                    + (("basic_oracle.py",) if oracle_kernel != "reference_v1" else ())
                    + (("structured_oracle.py",) if oracle_kernel in
                       ("structured_batch_v1", "structured_reuse_v1") else ()))
            },
        },
        "guidance": "reward_surrogate_finite_pool_resampling" if guidance else "disabled",
        "guidance_config": asdict(frozen_guidance_config) if guidance else None,
        "runtime": "resident_continuous_answer", "physical_module_sha256": {
            name: sha256(Path(module.__file__).read_bytes()).hexdigest()
            for name, module in (("online_training", online), ("continuous_hermite", hermite))
        },
        "scientific_qualification": "requires_canonical_held_out_rehearsal",
    }
    if oracle_kernel in ("structured_batch_v1", "structured_reuse_v1"):
        identity["public_oracle_evaluation"]["prepared_intermediates"] = {
            "scope": "immutable_case", "lazy": True,
            "reuse_between_requests": oracle_kernel == "structured_reuse_v1",
            "maximum_cases": 16 if oracle_kernel == "structured_reuse_v1" else 0,
            "final_source_cache": False,
        }
    from ..plasma_transport_settings import transport_calibration
    score_calibration = transport_calibration(metric_profile, metric_calibration)
    if score_calibration is not None:
        identity["scoring_objective"] = score_calibration.objective_identity
    identity["binding_sha256"] = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    # Bind the guide and request provenance to the exact same finalized
    # identity. The guide's runtime snapshot is not part of that identity,
    # so this does not introduce a circular hash dependency.
    if guidance:
        from ..reward_guidance import RewardGuidedBackend
        backend = RewardGuidedBackend(
            backend, feature_extractor=vpfp_public_guidance_features,
            feature_id="vpfp-public-condition-initial-moments-v1",
            base_generator_sha256=identity["binding_sha256"],
            config=frozen_guidance_config,
        )
    generator = _BoundVPFPContinuousGenerator(backend, identity)
    plugin = replace(build_vpfp_plugin(metric_profile=metric_profile,
                                     metric_calibration=metric_calibration), generator=generator)
    checkpoint = GeneratorCheckpoint(generator.info.program_id, identity, report=identity)
    dataset = PreparedDataset(
        DatasetInput(
            dataset_id=f"vpfp-{spec.regime_id}-online", kind=DatasetKind.NUMERICAL,
            problem={"equation": "VPFP", "regime_id": spec.regime_id},
            settings={"online_only": True},
        ), (), (),
        training_summary={"online_only": True, "frozen_checkpoint": True,
                          "experimental_data_validation": False},
    )
    return BackendRunBinding(plugin=plugin, checkpoint=checkpoint, dataset=dataset, report=identity)
