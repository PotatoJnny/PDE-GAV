"""Small source/import contracts, not real model or CUDA qualification."""
import importlib
import json
from pathlib import Path

import pytest

from pde_agent_verifier_solver.plugins.vpfp import weak_pde_gav_binding as api


def test_live_numerical_sources_and_inference_constructor_are_available(monkeypatch):
    def no_digest(path):
        raise AssertionError("live source checks must not rescan file digests")
    monkeypatch.setattr(api, "_digest", no_digest)
    report = api.validate_live_scientific_sources()
    assert report["checked_scientific_source_count"] == 36
    assert report["replay_constructor_and_identity_available"] is True
    assert report["historical_training_driver_imported"] is False
    assert report["new_cuda_integration_qualified"] is False


def test_missing_numerical_source_is_reported(monkeypatch):
    original = Path.is_file
    monkeypatch.setattr(Path, "is_file", lambda p: False if p.name == "moving_frame.py" else original(p))
    with pytest.raises(ValueError, match="numerical/model source is unavailable"):
        api.validate_live_scientific_sources()


def test_inference_adapter_never_trains_or_imports_the_historical_training_driver():
    import sys
    sys.path.insert(0, str(api.ROOT / "scripts"))
    try:
        from run_vpfp_weak_replay_probe_screen import ReplayEDM
        with pytest.raises(RuntimeError, match="inference-only"):
            ReplayEDM.loss(None)
        assert "run_vpfp_weak_source_pilot" not in sys.modules
    finally:
        sys.path.remove(str(api.ROOT / "scripts"))


@pytest.mark.parametrize("module", ["weak_portable_base", "weak_iteration_backend",
    "weak_perturbation", "weak_round_solver_guidance", "weak_solver_guidance",
    "moving_frame_adaptive_source", "innovation_binding", "hybrid_stress_diffusion"])
def test_weak_modules_import_from_the_new_workspace(module):
    loaded = importlib.import_module("pde_agent_verifier_solver.plugins.vpfp." + module)
    assert Path(loaded.__file__).resolve().is_relative_to(api.ROOT)


def test_real_model_load_is_not_allowed_on_login_node(monkeypatch, tmp_path):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="compute allocation"):
        api.build_pde_gav_weak_binding({}, tmp_path)
    with pytest.raises(RuntimeError, match="compute allocation"):
        api.validate_pde_gav_provenance({}, {})


def test_historical_pathmap_scope_restores_environment(monkeypatch, tmp_path):
    from hashlib import sha256
    path = tmp_path / "paths.json"
    path.write_text(json.dumps({"schema": "vpfp-weak-explicit-path-map-v1", "paths": []}))
    evidence = {"historical_path_map": {"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}}
    monkeypatch.setenv("VPFP_WEAK_PATH_MAP", "original-map")
    import os
    with api._historical_locators(evidence):
        assert os.environ["VPFP_WEAK_PATH_MAP"] == str(path)
    assert os.environ["VPFP_WEAK_PATH_MAP"] == "original-map"


def test_unknown_source_options_rejected_and_g0_profile_explicit():
    from pde_agent_verifier_solver.plugins.vpfp.moving_frame_oracle_options import validated_source_execution_options
    options = dict(variant="batched_geometry_g0_v3", source_request_block=8192,
                   g0_request_block=8192, geometry_batch_size=32, geometry_cache_size=512)
    assert validated_source_execution_options(options) == options
    with pytest.raises(ValueError):
        validated_source_execution_options(dict(options, invented_fast_math=True))
