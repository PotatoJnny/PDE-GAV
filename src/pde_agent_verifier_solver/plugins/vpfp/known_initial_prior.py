"""Frozen Gaussian-packet initial law and its original native t0 projection.

No PDE solve and no corpus trajectories are used at inference. This samples
the PUSHFORWARD of the known analytic initial law through the original finite
velocity/Hermite/Fourier projection; it is not an exact analytic-f0 decoder.
The parameter law draws initial packets independently of nu,T. At a requested
condition we keep those packet draws and set nu,T before the original packer
and projector. Unobserved source randomness is not a network condition.
"""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import importlib
import json
from pathlib import Path

import numpy as np
import torch

from .structured_diffusion import _seeds, _sha

SCHEMA = "vpfp_frozen_known_initial_pushforward_prior_v1"
FAMILY = "spatially_conditioned_gaussian_packets_v3"
MODULES = ("high_fidelity", "online_training", "torch_reference", "continuous_flow_training", "continuous_hermite")


def _json_sha(payload):
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _file_sha(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"known initial source must be a regular file: {path}")
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class KnownInitialPriorSpec:
    """Serializable, mutation-checked binding; construction does not load assets."""

    def __init__(self, payload):
        fields = {"schema", "model_spec", "corpus_manifest_sha256", "decoder_config", "generation_config",
                  "equation_config", "structured_latent_layout", "physical_module_sha256", "sha256"}
        if not isinstance(payload, dict) or set(payload) != fields or payload["schema"] != SCHEMA:
            raise ValueError("invalid known initial prior fields/schema")
        body = {key: value for key, value in payload.items() if key != "sha256"}
        from .canonical_binding import VPFPOnlineModelSpec
        spec = VPFPOnlineModelSpec.from_mapping(payload["model_spec"])
        _sha(spec.base_decoder_sha256, "known prior decoder")
        _sha(spec.expanded_decoder_sha256, "known prior expanded decoder")
        _sha(payload["corpus_manifest_sha256"], "known prior corpus")
        equation, decoder, generation = (payload[key] for key in ("equation_config", "decoder_config", "generation_config"))
        if equation.get("equation_family") != FAMILY or set(payload["physical_module_sha256"]) != set(MODULES):
            raise ValueError("known prior requires the complete frozen rich Gaussian-packet law")
        for digest in payload["physical_module_sha256"].values():
            _sha(digest, "known prior source module")
        for name in ("collision_nu_range", "temperature_range", "time_domain"):
            if tuple(decoder[name]) != tuple(getattr(spec, name)):
                raise ValueError(f"known prior decoder {name} differs")
        for name in ("collision_nu_range", "temperature_range"):
            if tuple(equation[name]) != tuple(getattr(spec, name)):
                raise ValueError(f"known prior equation {name} differs")
        from .value_control_audit import _layout
        _layout({"provenance": payload, "latent_dimension": decoder["latent_dim"], "time_domain": list(spec.time_domain)})
        if decoder.get("structured_residual_scale") != 0. or generation["nt"] < 3:
            raise ValueError("known prior requires original direct zero-residual time controls")
        self._payload = deepcopy(payload)
        self.sha256 = payload["sha256"]

    @classmethod
    def from_manifest(cls, manifest):
        body = {"schema": SCHEMA, "model_spec": deepcopy(manifest["model_spec"]),
                "corpus_manifest_sha256": manifest["manifest_sha256"]}
        body.update({name: deepcopy(manifest["provenance"][name]) for name in
            ("decoder_config", "generation_config", "equation_config", "structured_latent_layout", "physical_module_sha256")})
        return cls(dict(body, sha256=_json_sha(body)))

    @classmethod
    def from_payload(cls, payload):
        return cls(payload)

    def to_payload(self):
        return deepcopy(self._payload)

    def check_config(self, config):
        payload = self.to_payload()
        decoder = payload["decoder_config"]
        expected = {"structured_time_control_count": config.full_config.control_count,
            "structured_local_channels": config.hermite_channels,
            "rho_max_mode_x": config.rho_max_mode, "rho_max_mode_y": config.rho_max_mode,
            "structured_local_spatial_mode": config.kinetic_max_mode, "latent_dim": config.full_dimension}
        if any(decoder[key] != value for key, value in expected.items()):
            raise ValueError("known initial prior and conditional model layouts differ")
        if any(tuple(decoder[key]) != tuple(getattr(config, key)) for key in
               ("collision_nu_range", "temperature_range", "time_domain")):
            raise ValueError("known initial prior and conditional model physical ranges differ")


class KnownInitialPrior:
    """Lazy original physical runtime; per-case projection preserves grouping."""

    def __init__(self, spec, *, device="cpu"):
        if not isinstance(spec, KnownInitialPriorSpec):
            raise TypeError("known initial sampler requires a frozen prior specification")
        self.spec, self.device, self._runtime = spec, torch.device(device), None

    def _load(self):
        if self._runtime is not None:
            self.spec.to_payload()
            return self._runtime
        from .canonical_binding import VPFPOnlineModelSpec, runtime_decoder_path
        from .perturbation_backend import _legacy_api
        payload = self.spec.to_payload()
        spec = VPFPOnlineModelSpec.from_mapping(payload["model_spec"])
        spec.preflight()
        api = _legacy_api()
        modules = {name: importlib.import_module(f"agent_verifier.vpfp.{name}") for name in MODULES}
        for name, module in modules.items():
            if not Path(module.__file__).is_file():
                raise ValueError(f"known initial prior source identity differs: {name}")
        checkpoint = torch.load(runtime_decoder_path(spec.base_decoder_checkpoint, spec.base_decoder_sha256),
                                map_location="cpu", weights_only=True)
        if _json_sha(checkpoint["decoder_config"]) != _json_sha(payload["decoder_config"]):
            raise ValueError("known initial prior decoder configuration differs from checkpoint")
        adapter = checkpoint.get("robust_answer_first_adapter", {})
        if (adapter.get("unused_neural_parameters_zeroed") is not True
                or adapter.get("continuous_hidden_answer_is_primary") is not True
                or adapter.get("generator_parameterization_public") is not False):
            raise ValueError("known initial decoder is not answer-first qualified")
        decoder = modules["continuous_hermite"].VPFPContinuousHermiteConfig.from_dict(payload["decoder_config"])
        generation = modules["online_training"].VPFPOnlineGenerationConfig(**payload["generation_config"])
        equation = modules["high_fidelity"].VPFPHighFidelityConfig.from_dict(payload["equation_config"])
        if _json_sha(equation.to_dict()) != _json_sha(payload["equation_config"]):
            raise ValueError("known initial equation normalization changed")
        grid, time = generation.grid(), generation.time()
        model = api.build_deterministic_direct_model(decoder, device=self.device)
        with api.ieee_float32_coordinate_matmul(model):
            projector = modules["online_training"].VPFPTorchStructuredProjector(
                decoder, grid=grid, time=time, device=self.device, dtype="float32")
        self._runtime = {"api": api, "modules": modules, "equation": equation, "generation": generation,
            "grid": grid, "time": time, "model": model, "projector": projector, "decoder": decoder}
        return self._runtime

    @torch.no_grad()
    def sample(self, seeds, conditions=None):
        seeds = _seeds(seeds)
        payload = self.spec.to_payload()
        if conditions is not None:
            conditions = np.asarray(conditions, dtype=np.float32)
            bounds = np.asarray([payload["model_spec"][key] for key in ("collision_nu_range", "temperature_range")], dtype=np.float32)
            if (conditions.shape != (len(seeds), 2) or not np.isfinite(conditions).all()
                    or np.any(conditions < bounds[:, 0]) or np.any(conditions > bounds[:, 1])):
                raise ValueError("requested known-prior conditions leave the frozen regime")
        runtime = self._load()
        physical, reference = (runtime["modules"][name] for name in ("high_fidelity", "torch_reference"))
        rows, actual_conditions = [], []
        for index, seed in enumerate(seeds):
            spec = physical.sample_continuous_equation_spec(seed=seed, config=runtime["equation"], defer_rho_log_partition=True)
            if spec.get("family") != FAMILY:
                raise ValueError("known initial sampler returned an unexpected family")
            if conditions is not None:
                spec = dict(spec, collision_nu=float(conditions[index, 0]), temperature=float(conditions[index, 1]))
            parameters = reference.pack_torch_equation_specs([spec], dtype="float32")
            prepared = reference.evaluate_torch_equation_parameter_batch(parameters, grid=runtime["grid"],
                time=replace(runtime["time"], nt=3), device=self.device)
            theta = torch.stack((prepared["collision_nu"], prepared["temperature"]), dim=-1)
            rho, g = reference._uniform_rho_g(torch, prepared["f0"], grid=runtime["grid"],
                rho_floor=runtime["generation"].solver().rho_floor)
            count = runtime["time"].nt
            carrier = {"rho": rho[:, None].expand(-1, count, -1, -1),
                       "g": g[:, None].expand(-1, count, -1, -1, -1, -1)}
            with runtime["api"].ieee_float32_coordinate_matmul(runtime["model"]):
                full = runtime["projector"].project(carrier, theta)["latents"]
            layout = runtime["decoder"].structured_latent_layout()
            r, h, m = 2 * layout["rho_mode_count"], layout["hermite_mode_count"], layout["local_spatial_feature_count"]
            rho_end = layout["rho_latent_dim"]
            start = rho_end + layout["kinetic_global_latent_dim"]
            # Only first values leave this method. No constant future is an
            # output, target, or reverse-diffusion state.
            initial = torch.cat((full[:, :rho_end].reshape(1, 2, count, r)[:, 0, 0],
                full[:, start:].reshape(1, h, 2, count, m)[:, :, 0, 0].flatten(1)), dim=1)
            if not torch.isfinite(initial).all():
                raise ValueError("known initial projection is nonfinite")
            rows.append(initial)
            actual_conditions.append(theta.detach().cpu().numpy()[0])
        return {"initial_controls": torch.cat(rows), "equation_conditions": np.asarray(actual_conditions, dtype=np.float32),
                "prior_equation_seeds": np.asarray(seeds, dtype=np.int64), "pde_solves": 0}
