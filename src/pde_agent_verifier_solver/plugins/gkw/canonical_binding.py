"""Load the existing qualified GKW assets into a canonical Campaign binding."""

from __future__ import annotations

import hashlib
from dataclasses import replace
import importlib
from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys
from typing import Mapping

from pde_agent_verifier_solver.app.backend_run import BackendRunBinding
from pde_agent_verifier_solver.sdk import DatasetInput, DatasetKind, FormField, GeneratorCheckpoint, PreparedDataset
from .manufactured_backend import GKWManufacturedGeneratorBackend
from .product import build_gkw_plugin
from .generator import GKWAnswerFirstGenerator
from .selected_torch_runtime import GKWSelectedTorchRuntime
from .legacy_local_edit import (
    LEGACY_LOCAL_EDIT_ADAPTER_IDENTITY,
    adapt_qualified_80632f_local_edit_service,
)


PHYSICAL_COMMIT = "80632f11a39b8577124ffa652556af1521d9b992"
EXTERNAL_COMMIT = "568411b33be486262b8c5cfe1dfb3e0fe97f6419"
GYARADAX_COMMIT = "318e3d11d7326d99e96f2bd5a8998a462261dcd4"
PYTHON_DEPENDENCIES = {"einops": "0.8.1", "kappamodules": "0.1.112", "omegaconf": "2.3.0"}
ARTIFACT_NAMES = (
    "kinetic_edm", "kinetic_corpus", "potential_edm", "potential_corpus",
    "hierarchy_checkpoint", "parent_hierarchy_checkpoint", "geometry_sidecar",
    "velocity_metric", "correction_calibration", "time_registry", "encode_report",
    "latent_corpus", "equation_calibration",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkout(path: Path, expected: str) -> None:
    observed = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    if observed != expected:
        raise ValueError(f"GKW checkout {path} is {observed}, expected {expected}")
    dirty = subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=all"], text=True
    )
    if dirty.strip():
        raise ValueError(f"GKW physical/dependency checkout is modified: {path}")


def inspect_gkw_assets(settings: Mapping[str, object], *, verify_hashes: bool = True) -> dict:
    """Validate inputs without loading Torch models or generating cases."""
    roots = {name: Path(str(settings[name])).expanduser().resolve()
             for name in ("physical_root", "external_root", "gyaradax_root")}
    _checkout(roots["physical_root"], PHYSICAL_COMMIT)
    _checkout(roots["external_root"], EXTERNAL_COMMIT)
    _checkout(roots["gyaradax_root"], GYARADAX_COMMIT)
    artifacts = settings["artifacts"]
    expected = settings.get("artifact_sha256", {})
    if not isinstance(artifacts, Mapping):
        raise TypeError("GKW artifacts must be a mapping")
    if set(artifacts) != set(ARTIFACT_NAMES):
        raise ValueError("GKW needs all thirteen selected model/operator artifacts")
    files = {}
    for name in ARTIFACT_NAMES:
        path = Path(str(artifacts[name])).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = expected.get(name)  # Optional archived label; no full-file scan.
        files[name] = {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}
    qualification = Path(str(settings["qualification_root"])).expanduser().resolve()
    certificate = json.loads((qualification / "production_certificate.json").read_text())
    if (certificate.get("schema") != "fusion_gkw_offline_perturbation_qualification_v2"
            or certificate.get("physical_git_commit") != PHYSICAL_COMMIT
            or certificate.get("qualification_code_commit") != PHYSICAL_COMMIT
            or certificate.get("qualification_sample_count", 0) < 2700):
        raise ValueError("GKW certificate does not cover the continuous-condition production runtime")
    qualification_report = json.loads((qualification / "qualification_report.json").read_text())
    if qualification_report.get("production_certificate_eligible") is not True:
        raise ValueError("GKW qualification report is not production eligible")
    seed_plan = json.loads((qualification / "seed_plan.json").read_text())
    if (seed_plan.get("schema") != "fusion_gkw_perturbation_qualification_seed_plan_v4"
            or seed_plan.get("campaign_mode") != "production"
            or seed_plan.get("case_count") != certificate["qualification_sample_count"]
            or any(seed_plan.get("bindings", {}).get(key) != certificate.get(key)
                   for key in ("physical_git_commit", "qualification_code_commit", "scale_rule_sha256",
                               "generator_manifest_sha256", "operator_manifest_sha256",
                               "physical_source_manifest_sha256"))):
        raise ValueError("GKW qualification seed plan and certificate differ")
    dependency_report = {}
    dependency_root = settings.get("python_dependency_root")
    if dependency_root is not None:
        dependency_path = Path(str(dependency_root)).expanduser().resolve()
        if not dependency_path.is_dir():
            raise FileNotFoundError(dependency_path)
        installed = {item.metadata["Name"].lower(): item.version
                     for item in metadata.distributions(path=[str(dependency_path)])}
        if any(installed.get(name) != version for name, version in PYTHON_DEPENDENCIES.items()):
            raise ValueError("GKW auxiliary Python dependency versions differ from selected deployment")
        dependency_report = {"path": str(dependency_path), "versions": PYTHON_DEPENDENCIES}
    return {"physical_commit": PHYSICAL_COMMIT, "artifacts": files,
            "qualification": certificate, "artifact_hashes_verified": verify_hashes,
            "assets_ready": verify_hashes, "runtime_loaded": False,
            "qualification_environment_validated": False,
            "python_dependencies": dependency_report}


def _physical_modules(root: Path):
    source = root / "src"
    for name, module in tuple(sys.modules.items()):
        if name == "agent_verifier" or name.startswith("agent_verifier."):
            location = getattr(module, "__file__", None)
            if location and not Path(location).resolve().is_relative_to(source):
                raise RuntimeError("GKW physical package already imported from another checkout; use a fresh process")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    factory = importlib.import_module("agent_verifier.fusion_simulation_first.gkw_selected_service_factory")
    qualification = importlib.import_module("agent_verifier.fusion_simulation_first.gkw_perturbation_qualification")
    return factory, qualification


def gkw_guidance_features(case) -> tuple[float, ...]:
    """Public equation conditions only; no hidden answer or target features."""
    scalars = case.public_data["scalars"]
    return tuple(float(scalars[name]) for name in ("rlt_i", "rln_i", "shat", "q"))


class _BoundGKWGenerator(GKWAnswerFirstGenerator):
    def __init__(self, backend, implementation_sha256):
        super().__init__(backend)
        self.implementation_sha256 = implementation_sha256

    def setup_fields(self):
        return (
            FormField("guidance_enabled", "Experimental reward guidance", "boolean", self.capabilities.guidance),
            FormField("perturbations_enabled", "Qualified physical perturbations", "boolean", self.capabilities.perturbations),
        )

    def _check_settings(self, settings):
        frozen = {"generator_sampling_mode": "local_edit", "decoder_precision": "bfloat16",
                  "interaction_protocol": ("reward_surrogate_finite_pool_resampling" if self.capabilities.guidance
                                           else "unguided_base_diffusion_v1")}
        if any(key in settings and settings[key] != value for key, value in frozen.items()):
            raise ValueError("GKW settings conflict with the loaded physical model")
        if "generation_batch_size" in settings:
            raise ValueError("GKW batch size is owned by the loaded runtime, not a Run setting")

    def load(self, checkpoint, settings, progress=None):
        if checkpoint.payload != self.implementation_sha256:
            raise ValueError("GKW checkpoint does not match the bound physical runtime")
        self._check_settings(settings)
        return super().load(checkpoint, settings, progress)

    def sample(self, state, request, progress=None):
        self._check_settings(request.settings)
        expected = {"fixed_generator_id": self.info.program_id,
                    "fixed_generator_sha256": self.implementation_sha256}
        if any(key in request.settings and request.settings[key] != value
               for key, value in expected.items()):
            raise ValueError("GKW request conflicts with its bound frozen Generator")
        request = replace(request, settings={**request.settings, **expected})
        return super().sample(state, request, progress)


def bind_gkw_runtime(runtime, *, report: Mapping[str, object] | None = None,
                     backend=None, guidance: bool = False) -> BackendRunBinding:
    """Compose an initialized physical runtime for the product Controller."""
    backend = GKWManufacturedGeneratorBackend(runtime) if backend is None else backend
    if guidance:
        from pde_agent_verifier_solver.plugins.reward_guidance import RewardGuidedBackend
        backend = RewardGuidedBackend(
            backend, feature_extractor=gkw_guidance_features,
            feature_id="gkw-public-condition-v1",
            base_generator_sha256=runtime.implementation_sha256,
        )
    plugin = replace(build_gkw_plugin(backend),
                     generator=_BoundGKWGenerator(backend, runtime.implementation_sha256))
    checkpoint = GeneratorCheckpoint(plugin.generator.info.program_id, runtime.implementation_sha256,
                                     report=dict(report or {}))
    dataset = PreparedDataset(
        DatasetInput("gkw-online", DatasetKind.NUMERICAL, settings={"online_only": True}),
        development_cases=(), hidden_cases=(),
        training_summary={"online_only": True, "training_required": False},
    )
    return BackendRunBinding(plugin, checkpoint, dataset,
                             {**dict(report or {}), "guidance_enabled": guidance,
                              "guidance_method": "reward_surrogate_finite_pool_resampling" if guidance else None})


def load_gkw_binding(settings: Mapping[str, object], *, run_root: str | Path,
                     storage_root: str | Path, device: str = "cuda",
                     guidance: bool = False) -> BackendRunBinding:
    """Load the production local-edit model and verify its qualification once."""
    report = inspect_gkw_assets(settings)
    dependency = report["python_dependencies"]
    if dependency:
        for name in PYTHON_DEPENDENCIES:
            existing = sys.modules.get(name)
            if existing is not None:
                path = getattr(existing, "__file__", None)
                if path and not Path(path).resolve().is_relative_to(dependency["path"]):
                    raise RuntimeError("GKW auxiliary dependency already imported elsewhere; use a fresh process")
        sys.path.insert(0, dependency["path"])
    physical = Path(str(settings["physical_root"])).expanduser().resolve()
    factory, qualification = _physical_modules(physical)
    paths = {name: Path(item["path"]) for name, item in report["artifacts"].items()}
    qroot = Path(str(settings["qualification_root"])).expanduser().resolve()
    certificate = qualification.gkw_qualification_certificate_from_payload(report["qualification"])
    token = qualification.validate_gkw_production_qualification_environment(
        certificate,
        qualification_report_path=qroot / "qualification_report.json",
        generator_manifest_path=qroot / "generator_manifest.json",
        operator_manifest_path=qroot / "operator_manifest.json",
        physical_source_manifest_path=qroot / "physical_source_manifest.json",
        seed_plan_path=qroot / "seed_plan.json",
        generator_files={name: paths[name] for name in qualification.GKW_QUALIFICATION_GENERATOR_ARTIFACT_KEYS},
        operator_files={name: paths[name] for name in qualification.GKW_QUALIFICATION_OPERATOR_ARTIFACT_KEYS},
        physical_code_root=physical,
        external_commits={"neural_gyrokinetics": EXTERNAL_COMMIT, "gyaradax": GYARADAX_COMMIT},
    )
    root = Path(run_root).expanduser().resolve()
    storage = Path(storage_root).expanduser().resolve()
    if root == storage or not root.is_relative_to(storage):
        raise ValueError("GKW run_root must be below storage_root")
    runtime_settings = dict(settings.get("runtime", {}))
    allowed = {"macro_time_count", "macro_field_rows", "formal_macro_time_count", "oracle_max_total_rows"}
    if set(runtime_settings) - allowed:
        raise ValueError("unsupported GKW runtime settings")
    identity = {"assets": report, "runtime": runtime_settings, "device": device,
                "sampling_mode": "local_edit", "generator_kind": "empirical_conditional_edm_kernel",
                "compatibility_adapter": dict(LEGACY_LOCAL_EDIT_ADAPTER_IDENTITY),
                "compatibility_adapter_sha256": _file_sha256(Path(__file__).with_name("legacy_local_edit.py"))}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    service = factory.build_gkw_selected_online_service(
        artifacts=factory.GKWSelectedServiceArtifacts(
            external_root=Path(str(settings["external_root"])).resolve(),
            gyaradax_root=Path(str(settings["gyaradax_root"])).resolve(), **paths),
        private_root=root / "selected_service", private_storage_root=storage,
        device=device, decoder_inference_precision="bfloat16",
        reward_bootstrap_samples=32, reward_seed_namespace="gkw-perturbation-qualification",
    )
    service = adapt_qualified_80632f_local_edit_service(
        service, physical_root=physical, physical_revision=PHYSICAL_COMMIT,
        production_qualification=token,
    )
    runtime = GKWSelectedTorchRuntime(
        service, implementation_sha256=digest, production_qualification=token,
        private_root=root / "runtime", private_storage_root=storage,
        **runtime_settings,
    )
    return bind_gkw_runtime(runtime, guidance=guidance,
                            report={**identity, "runtime_loaded": True,
                                    "qualification_environment_validated": True})
