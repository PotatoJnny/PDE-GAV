from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pde_agent_verifier_solver.plugins as plugins
import pde_agent_verifier_solver.plugins.gkw as gkw
import pde_agent_verifier_solver.plugins.ns as ns
import pde_agent_verifier_solver.plugins.vpfp as vpfp
import pde_agent_verifier_solver.sdk as sdk


ROOT = Path(__file__).resolve().parents[1]
LEGACY_RUNNER = ROOT / "scripts" / "run_unified_plasma_agent_iteration.py"
OPT_IN_ENV = "PDE_ENABLE_LEGACY_MACRO_REFERENCE"
OPT_IN_FLAG = "--legacy-macro-reference"

DIRECT_WRAPPERS = (
    ROOT / "scripts" / "slurm_ns_unified_live_agent_iteration.sh",
    ROOT / "scripts" / "slurm_vpfp_unified_live_weak_agent_iteration.sh",
    ROOT / "scripts" / "slurm_vpfp_unified_live_strong_agent_iteration.sh",
    ROOT / "scripts" / "slurm_gkw_unified_live_agent_iteration.sh",
)

ACTIVE_WRAPPERS = DIRECT_WRAPPERS + (
    ROOT / "scripts" / "slurm_ns_unified_formal_template.sh",
    ROOT / "scripts" / "slurm_ns_unified_parallel_worker_smoke.sh",
    ROOT / "scripts" / "slurm_gkw_unified_competitive_full.sh",
    ROOT / "scripts" / "slurm_gkw_unified_competitive_recovery.sh",
    ROOT / "scripts" / "slurm_gkw_unified_competitive_rehearsal.sh",
    ROOT / "scripts" / "slurm_gkw_unified_fresh_objective_ab.sh",
    ROOT / "scripts" / "slurm_gkw_unified_lifecycle_memory_smoke.sh",
    ROOT / "scripts" / "slurm_gkw_unified_parallel_worker_smoke.sh",
    ROOT / "scripts" / "slurm_gkw_unified_runtime_smoke.sh",
    ROOT / "scripts" / "slurm_gkw_unified_strong_agent_iteration.sh",
)


def test_legacy_macro_python_runner_is_disabled_without_explicit_flag() -> None:
    completed = subprocess.run(
        (
            sys.executable,
            str(LEGACY_RUNNER),
            "--config",
            "not-read-while-disabled.json",
        ),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )

    assert completed.returncode == 2
    assert "history/control-arm" in completed.stderr
    assert OPT_IN_FLAG in completed.stderr
    assert "not-read-while-disabled" not in completed.stderr
    assert "Traceback" not in completed.stderr


def test_legacy_macro_python_runner_help_is_explicit() -> None:
    completed = subprocess.run(
        (sys.executable, str(LEGACY_RUNNER), "--help"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )

    assert "legacy formal Macro history/control-arm" in completed.stdout
    assert "Canonical" in completed.stdout
    assert "product runs use CampaignController" in completed.stdout
    source = LEGACY_RUNNER.read_text(encoding="utf-8")
    assert source.count(f'"{OPT_IN_FLAG}"') >= 2


def test_direct_legacy_macro_python_apis_are_disabled_by_default() -> None:
    from pde_agent_verifier_solver.engine.macro_closed_loop import (
        MacroClosedLoopRunner,
    )
    from pde_agent_verifier_solver.engine.reference_agent_iteration import (
        run_reference_agent_iteration,
        run_reference_process_agent_worker,
    )
    from pde_agent_verifier_solver.plugins.vpfp.fresh_u0 import run_fresh_u0

    try:
        MacroClosedLoopRunner(
            config=None,
            plugin=None,
            cold_start=None,
            agents={},
        )
    except PermissionError as exc:
        assert "history/control-arm" in str(exc)
        assert "CampaignController" in str(exc)
        assert "legacy_macro_reference=True" in str(exc)
    else:  # pragma: no cover - fail-closed contract
        raise AssertionError("legacy Macro runner constructed without intent")

    try:
        run_reference_agent_iteration(
            config=None,
            plugin=None,
            provenance={},
        )
    except PermissionError as exc:
        assert "legacy_macro_reference=True" in str(exc)
    else:  # pragma: no cover - fail-closed contract
        raise AssertionError("legacy iteration API ran without explicit intent")

    try:
        run_reference_process_agent_worker(
            config=None,
            plugin=None,
            request={},
            request_sha256="not-read-while-disabled",
            worker_root=ROOT,
            gpu_audit={},
        )
    except PermissionError as exc:
        assert "legacy_macro_reference=True" in str(exc)
    else:  # pragma: no cover - fail-closed contract
        raise AssertionError("legacy worker API ran without explicit intent")

    try:
        run_fresh_u0(None)
    except PermissionError as exc:
        assert "legacy_macro_reference=True" in str(exc)
    else:  # pragma: no cover - fail-closed contract
        raise AssertionError("legacy VPFP API ran without explicit intent")


def test_active_slurm_wrappers_fail_closed_without_environment_opt_in() -> None:
    for wrapper in ACTIVE_WRAPPERS:
        source = wrapper.read_text(encoding="utf-8")
        assert OPT_IN_ENV in source
        completed = subprocess.run(
            ("bash", str(wrapper)),
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
        )
        assert completed.returncode == 2
        assert "history/control-arm" in completed.stderr

    for wrapper in DIRECT_WRAPPERS:
        assert OPT_IN_FLAG in wrapper.read_text(encoding="utf-8")


def test_legacy_macro_contracts_are_not_product_star_imports() -> None:
    legacy_sdk_names = {
        "BoundMacroObjective",
        "CandidateRef",
        "CandidateSubmission",
        "CohortKind",
        "FixedGeneratorCaseSource",
        "LoadedGenerator",
        "MacroLoopPlugin",
        "MacroObjective",
        "MacroQueryFactory",
        "MacroROMProgram",
        "ManufacturedCaseSource",
        "SampleRequest",
    }
    assert legacy_sdk_names.isdisjoint(sdk.__all__)
    assert "RecoverableSamplingRejection" in sdk.__all__
    assert all(hasattr(sdk, name) for name in legacy_sdk_names)

    assert "campaign_plugin" not in plugins.__all__
    assert callable(plugins.campaign_plugin)

    ns_legacy_names = {
        "NSCampaignCompiler",
        "NSCampaignConfig",
        "bind_ns_macro_loop",
        "load_ns_campaign_config",
        "preflight_ns_campaign",
    }
    vpfp_legacy_names = {
        "VPFPCampaignCompiler",
        "VPFPCampaignConfig",
        "bind_vpfp_macro_loop",
        "bind_vpfp_release_bank_reference",
        "load_vpfp_campaign_config",
        "preflight_vpfp_campaign",
    }
    assert ns_legacy_names.isdisjoint(ns.__all__)
    assert vpfp_legacy_names.isdisjoint(vpfp.__all__)
    assert "bind_gkw_macro_loop" not in gkw.__all__
    assert all(hasattr(ns, name) for name in ns_legacy_names)
    assert all(hasattr(vpfp, name) for name in vpfp_legacy_names)
    assert callable(gkw.bind_gkw_macro_loop)
