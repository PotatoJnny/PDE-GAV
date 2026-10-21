"""Accepted weak VPFP science in the PDE-GAV Controller, without old launchers.

The old export proof remains evidence about its original runtime. This adapter
separately checks the live model/numerical sources and exact inference method
extraction. A CUDA integration receipt is still required before formal launch.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
SCHEMA = "pde_gav_weak_inference_binding_v1"
HISTORICAL_REVISION = "2ccbc33c2befef9b18636c871ec4285a5b402c86"
PORTABLE_SHA256 = "3b65ec8d04c65940432ea3817de60a684b553611dff6916ec1fd1f78f923b93b"
MACRO_SHA256 = "3262f2d7e65897d871b9eca3b7c388ad70a95735f89c4fc02b81490827d69582"
MICRO_SHA256 = "dad163578cbe7da8bd934ff16dc99bbc4155e66175d2c834db3e69d80df4fd18"


def _digest(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("regular nonsymlink input required: " + str(path))
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _asset(item):
    if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
        raise ValueError("explicit path and SHA256 asset descriptor required")
    path = Path(item["path"])
    if not path.is_absolute() or ".." in path.parts or _digest(path) != item["sha256"]:
        raise ValueError("weak migrated input digest differs: " + str(path))
    return path


def _methods(source, names=("__init__", "identity")):
    tree = ast.parse(source)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ReplayEDM"]
    if len(classes) != 1:
        raise ValueError("one explicit ReplayEDM class required")
    result = {node.name: ast.dump(node, include_attributes=False)
              for node in classes[0].body if isinstance(node, ast.FunctionDef) and node.name in names}
    if set(result) != set(names):
        raise ValueError("ReplayEDM inference constructor/identity methods missing")
    return result


def validate_live_scientific_sources(root=ROOT):
    """Check available modules and inference interfaces without scanning file contents."""
    root = Path(root)
    manifest_path = root / "src/pde_agent_verifier_solver/plugins/vpfp/weak_pde_gav_sources.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("schema") != SCHEMA or manifest.get("historical_revision") != HISTORICAL_REVISION
            or not manifest.get("scientific_sources") or not manifest.get("replay_inference_methods")):
        raise ValueError("explicit weak live-science manifest required")
    for relative in manifest["scientific_sources"]:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or not (root / path).is_file():
            raise ValueError("weak numerical/model source is unavailable: " + relative)
    _methods((root / "scripts/run_vpfp_weak_replay_probe_screen.py").read_text())
    return {"schema": SCHEMA, "source_manifest_path": str(manifest_path),
            "checked_scientific_source_count": len(manifest["scientific_sources"]),
            "replay_constructor_and_identity_available": True,
            "historical_training_driver_imported": False,
            "new_cuda_integration_qualified": False}


@contextmanager
def _historical_locators(evidence):
    path = _asset(evidence["historical_path_map"])
    previous = os.environ.get("VPFP_WEAK_PATH_MAP")
    os.environ["VPFP_WEAK_PATH_MAP"] = str(path)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("VPFP_WEAK_PATH_MAP", None)
        else:
            os.environ["VPFP_WEAK_PATH_MAP"] = previous


def validate_pde_gav_provenance(provenance, evidence):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("full historical asset verification and model loading require a compute allocation")
    historical = evidence["historical_runtime_root"]
    if (not isinstance(historical, dict) or set(historical) != {"path", "revision"}
            or historical["revision"] != HISTORICAL_REVISION):
        raise ValueError("accepted frozen weak runtime revision required")
    historical_root = Path(historical["path"])
    if (not historical_root.is_absolute() or historical_root.is_symlink()
            or historical_root.resolve() == ROOT.resolve()):
        raise ValueError("distinct regular historical evidence source required")
    command = ["git", "-C", str(historical_root)]
    if (subprocess.check_output(command + ["rev-parse", "HEAD"], text=True).strip() != HISTORICAL_REVISION
            or subprocess.check_output(command + ["status", "--porcelain", "--untracked-files=all"], text=True).strip()):
        raise ValueError("historical weak evidence source changed")
    live = validate_live_scientific_sources()
    local_proof = ROOT / "scripts/vpfp_weak_portability_provenance.py"
    spec = importlib.util.spec_from_file_location("_pde_gav_weak_historical_proof", local_proof)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Execute our local, source-checked stdlib proof, not Python from old roots.
    with _historical_locators(evidence):
        historical_checks = module.validate_runtime_portability_provenance(
            provenance, historical_root, "weak_oracle_fast_v1")
    return {"execution_passed": historical_checks.get("execution_passed") is True,
            "historical_export_and_runtime_checks": historical_checks,
            "pde_gav_scientific_sources": live,
            "new_interface_gpu_equivalence_qualified": False,
            "scope": "historical_asset_proof_plus_live_inference_interfaces_not_new_run_qualification"}


def build_pde_gav_weak_binding(config, output_root, *, metric_profile=None, metric_calibration=None):
    """Return (SDK binding, receipt) for the frozen H2/20k fast weak backend."""
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("weak model construction requires a compute allocation")
    if config["portable_base"]["sha256"] != PORTABLE_SHA256:
        raise ValueError("only the accepted H2/seed0 fixed20k portable Base is registered")
    script_root = str(ROOT / "scripts")
    if script_root not in sys.path:
        sys.path.insert(0, script_root)
    import numpy as np  # noqa: F401 -- frozen NumPy-before-Torch convention
    import torch
    from vpfp_weak_guidance_base import draw_base
    from vpfp_weak_checkpoint_compat import install_restore_compat
    from .weak_portable_base import load_portable_base
    from .weak_iteration_backend import build_weak_iteration_binding
    from .weak_round_solver_guidance import train_weak_round_guidance
    from .weak_perturbation import WeakCoherentPerturbationConfig
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Pass an absolute current asset directly; locators apply only to history.
    item = config["portable_base"]
    previous_map = os.environ.pop("VPFP_WEAK_PATH_MAP", None)
    try:
        sampler, decoder, receipt = load_portable_base(item["path"], item["sha256"],
            device=config.get("device", "cuda"), migration_evidence=config)
    finally:
        if previous_map is not None:
            os.environ["VPFP_WEAK_PATH_MAP"] = previous_map
    if (receipt["macro_checkpoint"]["sha256"], receipt["checkpoint"]["sha256"]) != (MACRO_SHA256, MICRO_SHA256):
        raise ValueError("accepted weak macro/micro asset identity changed")
    perturbation = dict(config["perturbation_config"])
    perturbation.pop("schema", None)
    binding = build_weak_iteration_binding(decoder=decoder, sampler=sampler, draw_function=draw_base,
        output_root=output_root, trainer=train_weak_round_guidance, initial_guidance=None,
        guidance_config=config["guidance_config"],
        perturbation_config=WeakCoherentPerturbationConfig(**perturbation),
        oracle_execution_options=config["oracle_execution_options"],
        metric_profile=metric_profile, metric_calibration=metric_calibration)
    install_restore_compat(binding.plugin.generator)
    receipt["pde_gav_binding"] = {"schema": SCHEMA, "historical_campaign_state_imported": False,
        "new_cuda_integration_qualified": False, "uses_shared_controller": True}
    return binding, receipt
