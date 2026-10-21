"""Artifact-backed continuous structured NS sampling.

The adapter loads one frozen PCA/rectified-flow pair, validates their binding
once, and materializes answer-first continuous fields.  Online generation is
batched: one ``flow.sample`` call serves the whole requested case batch.  It
does not perform per-sample hashing, physical audits, or dataset validation.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
from math import isfinite
from pathlib import Path
from threading import Lock
from types import MappingProxyType
from typing import Mapping, Sequence

from .continuous_backend import (
    LegacyStructuredNSSmoothPerturber,
    StructuredNSManufacturedCaseSource,
    _NS_CUDA_QUERY_ADMISSION,
)


@dataclass(frozen=True)
class StructuredNSArtifactBackend:
    """Small assembly returned to the equation-neutral launcher."""

    flow_sampler: "LegacyStructuredNSFlowSampler"
    perturbation_sampler: LegacyStructuredNSSmoothPerturber
    case_source: StructuredNSManufacturedCaseSource

    def build_guidance_backend(
        self,
        *,
        artifact_root: str | Path,
        config=None,
        symbols: object | None = None,
    ):
        from .structured_guidance_backend import (
            StructuredNSMacroGuidanceBackend,
        )

        return StructuredNSMacroGuidanceBackend(
            flow_sampler=self.flow_sampler,
            case_source=self.case_source,
            artifact_root=artifact_root,
            config=config,
            symbols=symbols,
        )

    def bind(self, *, guidance_backend: object | None = None):
        """Bind to the shared Macro loop without constructing a controller."""

        from .macro_loop import bind_ns_macro_loop

        # The runner receives this same backend separately; passing it here
        # only makes the public capability declaration truthful.
        return bind_ns_macro_loop(
            case_source=self.case_source,
            guidance_backend=guidance_backend,
        )


class LegacyStructuredNSFlowSampler:
    """Frozen structured-coefficient RF -> continuous MMS adapter.

    ``time_interval``, ``viscosity``, and ``geometry_spec`` are explicit
    public physical contracts.  They are intentionally not inferred from an
    inaccessible hidden training case at online-use time.
    """

    def __init__(
        self,
        *,
        flow_checkpoint: str | Path,
        pca_checkpoint: str | Path,
        time_interval: tuple[float, float],
        viscosity: float,
        device: str = "cuda",
        integration_steps: int = 64,
        flow_state: str = "ema",
        integration_method: str = "heun",
        cuda_graph_sampling: bool = False,
        geometry_spec: Mapping[str, object] | None = None,
        continuous_mms_abi: str | None = None,
        symbols: object | None = None,
    ) -> None:
        self.flow_checkpoint = Path(flow_checkpoint).expanduser().resolve()
        self.pca_checkpoint = Path(pca_checkpoint).expanduser().resolve()
        for label, path in (
            ("structured NS flow checkpoint", self.flow_checkpoint),
            ("structured NS PCA checkpoint", self.pca_checkpoint),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"missing {label}: {path}")
        try:
            t0, t1 = (float(value) for value in time_interval)
        except (TypeError, ValueError) as exc:
            raise ValueError("structured NS time interval must contain two values") from exc
        if not (isfinite(t0) and isfinite(t1) and t1 > t0):
            raise ValueError("structured NS time interval is invalid")
        viscosity = float(viscosity)
        if not isfinite(viscosity) or viscosity < 0.0:
            raise ValueError("structured NS viscosity must be nonnegative")
        if isinstance(integration_steps, bool) or int(integration_steps) < 1:
            raise ValueError("structured NS flow integration steps must be positive")
        if flow_state not in {"ema", "raw"}:
            raise ValueError("structured NS flow state must be ema or raw")
        if integration_method not in {"euler", "heun"}:
            raise ValueError("structured NS flow method must be euler or heun")

        if not isinstance(cuda_graph_sampling, bool):
            raise TypeError("NS CUDA graph sampling must be a boolean")
        # Opt-in only: the caller must serialize CUDA work in this Controller
        # process during graph capture. The canonical NS runner owns that
        # boundary; arbitrary SDK users retain the eager path by default.
        self._cuda_graph_sampling = cuda_graph_sampling

        resolved = symbols or _legacy_symbols()
        torch = resolved.torch
        self._torch = torch
        self._device = torch.device(device)
        self._steps = int(integration_steps)
        self._method = integration_method
        self._time_interval = (t0, t1)
        self._viscosity = viscosity
        self._symbols = resolved
        self._lock = Lock()

        payload = torch.load(
            self.flow_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        if payload.get("schema") != resolved.flow_schema:
            raise ValueError("unexpected structured NS flow checkpoint schema")
        pca_sha256 = payload.get("pca_sha256")  # Optional historical label.
        rank = int(payload["pca_rank"])
        self.pca = resolved.pca_type.load(
            self.pca_checkpoint,
            map_location=self._device,
        ).with_rank(rank)
        # The PCA declares the coefficient representation. Never
        # silently reinterpret archived v5 weights with a different mask.
        artifact_abi = self.pca.source_signature.get(
            "continuous_mms_abi", resolved.continuous_mms_abi,
        )
        if artifact_abi not in {"continuous_structured_ns_mms_v5", "continuous_structured_ns_mms_v6"}:
            raise ValueError("structured NS PCA declares an unsupported continuous MMS ABI")
        if continuous_mms_abi is not None and continuous_mms_abi != artifact_abi:
            raise ValueError("requested continuous MMS ABI differs from the PCA artifact contract")
        if payload.get("continuous_mms_abi", artifact_abi) != artifact_abi:
            raise ValueError("structured NS flow and PCA continuous MMS ABIs differ")
        self.continuous_mms_abi = artifact_abi
        projection_payload = self.pca.source_signature.get("projection_config")
        if not isinstance(projection_payload, Mapping):
            raise ValueError("structured NS PCA lacks its projection contract")
        projection = resolved.projection_config_type.from_dict(
            dict(projection_payload)
        )
        projection.validate()
        expected_shape = (
            2,
            int(projection.time_coefficient_count),
            int(projection.spatial_mode_x) + 1,
            int(projection.spatial_mode_y) + 1,
        )
        if tuple(int(value) for value in self.pca.coefficient_shape) != expected_shape:
            raise ValueError("structured NS PCA and projection shapes differ")
        self._projection = projection

        model_config = resolved.flow_config_type.from_dict(
            dict(payload["model_config"])
        )
        if int(model_config.feature_dim) != rank:
            raise ValueError("structured NS flow feature dimension differs from PCA rank")
        model = resolved.flow_type.build(model_config)
        state_key = "ema_state_dict" if flow_state == "ema" else "model_state_dict"
        if state_key not in payload:
            raise ValueError(f"structured NS flow checkpoint lacks {state_key}")
        model.load_state_dict(payload[state_key])
        self._model = model.to(self._device).eval()

        if geometry_spec is None:
            geometry = resolved.build_default_geometry()
            spec = resolved.geometry_from_contract(geometry)
        else:
            spec = resolved.geometry_spec_type.from_dict(dict(geometry_spec))
        spec.validate()
        self._geometry_spec = spec
        self._domain_bounds = tuple(float(value) for value in spec.domain_bounds)
        self._artifact_metadata = MappingProxyType(
            {
                "pca_sha256": pca_sha256,
                "continuous_mms_abi": artifact_abi,
                "flow_checkpoint": str(self.flow_checkpoint),
                "pca_checkpoint": str(self.pca_checkpoint),
                "flow_state": flow_state,
                "integration_steps": self._steps,
                "integration_method": self._method,
                "cuda_graph_sampling": self._cuda_graph_sampling,
                "pca_rank": rank,
            }
        )

    @property
    def artifact_metadata(self) -> Mapping[str, object]:
        """Construction-time controller metadata; never recomputed per draw."""

        return self._artifact_metadata

    def __call__(self, seed: int):
        return self.sample_many((seed,))[0]

    def sample_many(self, seeds: Sequence[int]) -> tuple[object, ...]:
        return self.sample_many_with_guidance(seeds, guidance=None)

    def sample_many_with_guidance(
        self,
        seeds: Sequence[int],
        *,
        guidance: object | None,
        guidance_scale: float = 1.0,
    ) -> tuple[object, ...]:
        normalized = _seeds(seeds)
        torch = self._torch
        # Sampling and public derivatives share the same physical GPU. Keep
        # their autograd work from starving one another; waiting stays outside
        # numerical service time and remains inside the Agent hard wall.
        device_index = (self._device.index if self._device.index is not None
                        else torch.cuda.current_device()) if self._device.type == "cuda" else None
        admission = (_NS_CUDA_QUERY_ADMISSION.for_device(device_index)
                     if self._device.type == "cuda" else nullcontext())
        with self._lock, admission:
            generator = torch.Generator(device=self._device).manual_seed(
                _batch_seed(normalized)
            )
            with torch.no_grad():
                initial = torch.randn(
                    (len(normalized), int(self.pca.rank)),
                    device=self._device,
                    generator=generator,
                )
                scores = None
                if self._cuda_graph_sampling:
                    from .flow_cuda_graph import sample_flow_cuda_graph

                    scores = sample_flow_cuda_graph(
                        self._model,
                        initial,
                        steps=self._steps,
                        method=self._method,
                        guidance=guidance,
                        guidance_scale=(float(guidance_scale) if guidance is not None else 0.0),
                    )
                if scores is None:
                    scores = self._model.sample(
                        len(normalized),
                        steps=self._steps,
                        device=self._device,
                        initial_values=initial,
                        guidance=guidance,
                        guidance_scale=(float(guidance_scale) if guidance is not None else 0.0),
                        method=self._method,
                    )
                scores = scores.float()
                coefficients = self.pca.decode(scores).float()
            fields = tuple(
                self._field_from_coefficients(coefficients[index])
                for index in range(len(normalized))
            )
            if self._device.type == "cuda":
                torch.cuda.current_stream(self._device).synchronize()
            return fields

    def encode_fields(self, fields: Sequence[object]):
        """Controller-private coefficient encoding for reward Guidance."""

        if not fields:
            raise ValueError("structured NS guidance fields cannot be empty")
        if any(getattr(field, "continuous_mms_abi", None) != self.continuous_mms_abi for field in fields):
            raise ValueError("guidance field and PCA continuous MMS ABIs differ")
        converter = self._symbols.coefficient_tensor_from_field
        coefficients = self._torch.stack(
            [converter(field) for field in fields],
            dim=0,
        ).to(device=self._device, dtype=self._torch.float32)
        with self._torch.no_grad():
            return self.pca.encode(coefficients).float()

    def _field_from_coefficients(self, coefficients):
        config = self._projection
        shape = (int(config.time_coefficient_count), -1)
        values = self._symbols.case_coefficients_type(
            psi=coefficients[0].reshape(shape),
            pressure=coefficients[1].reshape(shape),
        )
        return self._symbols.field_type(
            values,
            config=config,
            geometry_spec=self._geometry_spec,
            continuous_mms_abi=self.continuous_mms_abi,
            domain_bounds=self._domain_bounds,
            viscosity=self._viscosity,
            t0=self._time_interval[0],
            t1=self._time_interval[1],
            device=self._device,
        )


def build_structured_ns_backend(
    *,
    generator_id: str,
    generator_sha256: str,
    flow_checkpoint: str | Path,
    pca_checkpoint: str | Path,
    time_interval: tuple[float, float],
    viscosity: float,
    perturbation_amplitude: float,
    device: str = "cuda",
    integration_steps: int = 64,
    flow_state: str = "ema",
    integration_method: str = "heun",
    cuda_graph_sampling: bool = False,
    perturbation_family: str = "spectral_complement",
    spectral_max_mode: int = 20,
    geometry_spec: Mapping[str, object] | None = None,
    continuous_mms_abi: str | None = None,
    max_points_per_request: int = 4096,
    symbols: object | None = None,
) -> StructuredNSArtifactBackend:
    """Build the clean/perturbed formal backend from explicit artifacts."""

    sampler = LegacyStructuredNSFlowSampler(
        flow_checkpoint=flow_checkpoint,
        pca_checkpoint=pca_checkpoint,
        time_interval=time_interval,
        viscosity=viscosity,
        device=device,
        integration_steps=integration_steps,
        flow_state=flow_state,
        integration_method=integration_method,
        cuda_graph_sampling=cuda_graph_sampling,
        geometry_spec=geometry_spec,
        continuous_mms_abi=continuous_mms_abi,
        symbols=symbols,
    )
    projection = sampler._projection
    spatial_basis = getattr(projection, "spatial_basis", "cosine")
    if perturbation_family == "spectral_complement" and spatial_basis != "cosine":
        raise ValueError("spectral_complement requires cosine space; explicitly choose a compatible perturbation family")
    if perturbation_family == "spectral_complement" and int(spectral_max_mode) <= max(projection.spatial_mode_x, projection.spatial_mode_y):
        raise ValueError("spectral_max_mode must explicitly exceed the represented spatial band")
    if perturbation_family == "spline_refinement" and spatial_basis != "bspline":
        raise ValueError("spline_refinement requires spatial B-splines")
    perturbation = LegacyStructuredNSSmoothPerturber(
        amplitude=perturbation_amplitude,
        family=perturbation_family,
        spectral_max_mode=spectral_max_mode,
        pca=(sampler.pca if perturbation_family.startswith("pca_") else None),
    )
    source = StructuredNSManufacturedCaseSource(
        generator_id=generator_id,
        artifact_sha256=generator_sha256,
        clean_sampler=sampler,
        perturbation_sampler=perturbation,
        max_points_per_request=max_points_per_request,
        boundary_velocity_trace=(getattr(sampler._geometry_spec, "outer_boundary", "no_slip")
                                 == "manufactured_trace"),
    )
    return StructuredNSArtifactBackend(sampler, perturbation, source)


def _legacy_symbols():
    try:
        import torch

        from agent_verifier.neural_generator.continuous_geometry import (
            ContinuousGeometrySpec,
            continuous_geometry_spec_from_contract,
        )
        from agent_verifier.neural_generator.structured_ns_answer_first import (
            STRUCTURED_NS_MMS_ABI_VERSION,
            ContinuousStructuredNSMMS,
            StructuredNSCaseCoefficients,
            StructuredNSProjectionConfig,
        )
        from agent_verifier.neural_generator.structured_ns_distribution import (
            STRUCTURED_NS_FLOW_SCHEMA,
            StructuredCoefficientFlowConfig,
            StructuredCoefficientPCA,
            StructuredCoefficientRectifiedFlow,
        )
        from agent_verifier.neural_generator.structured_ns_robustness import (
            coefficient_tensor_from_field,
        )
        from agent_verifier.neural_generator.triplane_dataset import (
            build_kite_baffle_geometry,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "the formal structured NS runtime is not importable; add "
            "legacy/ns/src to PYTHONPATH and provide the trained artifacts"
        ) from exc
    from types import SimpleNamespace

    return SimpleNamespace(
        torch=torch,
        flow_schema=STRUCTURED_NS_FLOW_SCHEMA,
        pca_type=StructuredCoefficientPCA,
        flow_config_type=StructuredCoefficientFlowConfig,
        flow_type=StructuredCoefficientRectifiedFlow,
        projection_config_type=StructuredNSProjectionConfig,
        field_type=ContinuousStructuredNSMMS,
        case_coefficients_type=StructuredNSCaseCoefficients,
        continuous_mms_abi=STRUCTURED_NS_MMS_ABI_VERSION,
        geometry_spec_type=ContinuousGeometrySpec,
        build_default_geometry=build_kite_baffle_geometry,
        geometry_from_contract=continuous_geometry_spec_from_contract,
        coefficient_tensor_from_field=coefficient_tensor_from_field,
    )


def _seeds(values: Sequence[int]) -> tuple[int, ...]:
    seeds = tuple(values)
    if not seeds:
        raise ValueError("structured NS batch cannot be empty")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("structured NS seeds must be nonnegative integers")
    return seeds


def _batch_seed(seeds: Sequence[int]) -> int:
    digest = hashlib.sha256(b"structured-ns-flow-batch-v1\0")
    for seed in seeds:
        digest.update(str(seed).encode("ascii"))
        digest.update(b"\0")
    return int.from_bytes(digest.digest()[:8], "big") & ((1 << 63) - 1)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "LegacyStructuredNSFlowSampler",
    "StructuredNSArtifactBackend",
    "build_structured_ns_backend",
]
