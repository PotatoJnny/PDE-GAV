from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from textwrap import dedent


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _run_isolated_python(source: str) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    source_root = str(_REPOSITORY_ROOT / "src")
    inherited = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_root
        if not inherited
        else source_root + os.pathsep + inherited
    )
    result = subprocess.run(
        [sys.executable, "-c", dedent(source)],
        cwd=_REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30.0,
    )
    assert result.returncode == 0, (
        f"isolated import probe failed\nstdout:\n{result.stdout}"
        f"\nstderr:\n{result.stderr}"
    )
    return result


def test_canonical_sdk_app_and_registry_imports_stay_pipeline_isolated():
    result = _run_isolated_python(
        """
        import sys

        banned = {
            "pde_agent_verifier_solver.sdk.macro_loop",
            "pde_agent_verifier_solver.sdk.execution_boundary",
            "pde_agent_verifier_solver.engine.staged",
            "pde_agent_verifier_solver.engine.macro_closed_loop",
            "pde_agent_verifier_solver.plugins.ns.campaign",
            "pde_agent_verifier_solver.plugins.vpfp.campaign",
            "pde_agent_verifier_solver.plugins.ns.macro_loop",
            "pde_agent_verifier_solver.plugins.vpfp.macro_loop",
            "pde_agent_verifier_solver.plugins.gkw.macro_loop",
        }

        def require_clean(stage):
            loaded = sorted(banned.intersection(sys.modules))
            if loaded:
                raise AssertionError(f"{stage} loaded legacy modules: {loaded}")

        import pde_agent_verifier_solver.sdk
        require_clean("canonical SDK")

        import pde_agent_verifier_solver.app
        require_clean("canonical app")

        from pde_agent_verifier_solver.plugins.registry import built_in_plugins

        plugins = built_in_plugins()
        assert tuple(plugin.plugin_id for plugin in plugins) == (
            "toy",
            "ns",
            "vpfp",
            "gkw",
            "vp_1d1v",
        )
        require_clean("built-in plugin registry")
        assert not any(name == "agent_verifier" or name.startswith("agent_verifier.")
                       or name.startswith("_pde_artemis_physical_") for name in sys.modules), (
            "plugin discovery must not load resident physical packages"
        )
        print("canonical-imports-isolated")
        """
    )
    assert result.stdout.strip() == "canonical-imports-isolated"


def test_ns_and_vpfp_star_imports_do_not_load_legacy_controllers_or_bindings():
    result = _run_isolated_python(
        """
        import sys

        # Broad exports include the experimental NS assembly, which reuses
        # CohortKind from sdk.macro_loop. Importing that compatibility module
        # must not load a historical Controller, execution boundary or binding.
        banned = {
            "pde_agent_verifier_solver.engine.staged",
            "pde_agent_verifier_solver.engine.macro_closed_loop",
            "pde_agent_verifier_solver.sdk.execution_boundary",
            "pde_agent_verifier_solver.plugins.ns.campaign",
            "pde_agent_verifier_solver.plugins.ns.macro_loop",
            "pde_agent_verifier_solver.plugins.vpfp.campaign",
            "pde_agent_verifier_solver.plugins.vpfp.macro_loop",
        }

        from pde_agent_verifier_solver.plugins.ns import *

        loaded = sorted(banned.intersection(sys.modules))
        assert not loaded, f"NS star import loaded legacy modules: {loaded}"

        from pde_agent_verifier_solver.plugins.vpfp import *

        loaded = sorted(banned.intersection(sys.modules))
        assert not loaded, f"VPFP star import loaded legacy modules: {loaded}"
        assert not any(
            name == "agent_verifier" or name.startswith("agent_verifier.")
            or name.startswith("_pde_artemis_physical_")
            for name in sys.modules
        ), "plugin star imports must not load external physical packages"
        print("plugin-star-imports-isolated")
        """
    )
    assert result.stdout.strip() == "plugin-star-imports-isolated"


def test_explicit_plugin_legacy_attributes_remain_lazy_and_compatible():
    result = _run_isolated_python(
        """
        import sys
        import warnings

        from pde_agent_verifier_solver.plugins import gkw, ns, vpfp

        ns_legacy = {
            "NSCampaignCompiler": "pde_agent_verifier_solver.plugins.ns.campaign",
            "NSCampaignConfig": "pde_agent_verifier_solver.plugins.ns.campaign",
            "load_ns_campaign_config": "pde_agent_verifier_solver.plugins.ns.campaign",
            "preflight_ns_campaign": "pde_agent_verifier_solver.plugins.ns.campaign",
            "bind_ns_macro_loop": "pde_agent_verifier_solver.plugins.ns.macro_loop",
            "NS_VELOCITY_POINT_COUNTS": "pde_agent_verifier_solver.plugins.ns.macro_loop",
            "ns_native_velocity_query": "pde_agent_verifier_solver.plugins.ns.macro_loop",
            "ns_velocity_point_query": "pde_agent_verifier_solver.plugins.ns.macro_loop",
        }

        vpfp_legacy = {
            "VPFPCampaignCompiler": "pde_agent_verifier_solver.plugins.vpfp.campaign",
            "VPFPCampaignConfig": "pde_agent_verifier_solver.plugins.vpfp.campaign",
            "load_vpfp_campaign_config": "pde_agent_verifier_solver.plugins.vpfp.campaign",
            "preflight_vpfp_campaign": "pde_agent_verifier_solver.plugins.vpfp.campaign",
            "VPFPLegacyObservableAdapter": "pde_agent_verifier_solver.plugins.vpfp.legacy_adapter",
            "VPFPLegacyRunner": "pde_agent_verifier_solver.plugins.vpfp.legacy_adapter",
            "bind_vpfp_macro_loop": "pde_agent_verifier_solver.plugins.vpfp.macro_loop",
            "bind_vpfp_release_bank_reference": "pde_agent_verifier_solver.plugins.vpfp.macro_loop",
            "vpfp_native_macro_query": "pde_agent_verifier_solver.plugins.vpfp.macro_loop",
            "VPFPReleaseBankCaseSource": "pde_agent_verifier_solver.plugins.vpfp.release_bank",
            "make_vpfp_hold_initial_cold_start": (
                "pde_agent_verifier_solver.plugins.vpfp.macro_programs"
            ),
        }
        gkw_legacy = {
            "GKWLegacyObservableAdapter": "pde_agent_verifier_solver.plugins.gkw.legacy_adapter",
            "GKWLegacyRuntimeRunner": "pde_agent_verifier_solver.plugins.gkw.legacy_adapter",
            "bind_gkw_macro_loop": "pde_agent_verifier_solver.plugins.gkw.macro_loop",
            "gkw_native_macro_query": "pde_agent_verifier_solver.plugins.gkw.macro_loop",
            "GKW_LEGACY_MACRO_BRIDGE_SCHEMA": (
                "pde_agent_verifier_solver.plugins.gkw.legacy_macro_bridge"
            ),
            "evaluate_gkw_legacy_macro_match": (
                "pde_agent_verifier_solver.plugins.gkw.legacy_macro_bridge"
            ),
            "validate_gkw_legacy_macro_evidence": (
                "pde_agent_verifier_solver.plugins.gkw.legacy_macro_bridge"
            ),
        }

        assert not set(ns_legacy).intersection(ns.__all__)
        assert not set(vpfp_legacy).intersection(vpfp.__all__)
        assert not set(gkw_legacy).intersection(gkw.__all__)
        assert not set(ns_legacy.values()).intersection(sys.modules)
        assert not set(vpfp_legacy.values()).intersection(sys.modules)
        assert not set(gkw_legacy.values()).intersection(sys.modules)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            for package, exports in (
                (ns, ns_legacy),
                (vpfp, vpfp_legacy),
                (gkw, gkw_legacy),
            ):
                for name, module_name in exports.items():
                    assert getattr(package, name) is not None
                    assert module_name in sys.modules

        messages = tuple(str(item.message) for item in caught)
        for name in (*ns_legacy, *vpfp_legacy, *gkw_legacy):
            assert any(name in message for message in messages)
        print("explicit-legacy-imports-compatible")
        """
    )
    assert result.stdout.strip() == "explicit-legacy-imports-compatible"
