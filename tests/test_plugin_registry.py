import unittest
from pathlib import Path

import pde_agent_verifier_solver.plugins.registry as plugin_registry
from pde_agent_verifier_solver.plugins.registry import (
    LegacyStagedCampaignAdapter,
    built_in_plugins,
    get_plugin,
    legacy_staged_plugin,
)
from pde_agent_verifier_solver.sdk import TaskKind


class PluginRegistryTests(unittest.TestCase):
    def test_builtins_expose_problem_generator_and_tasks(self) -> None:
        plugins = {plugin.plugin_id: plugin for plugin in built_in_plugins()}

        self.assertEqual(set(plugins), {"toy", "ns", "vpfp", "gkw", "vp_1d1v"})
        self.assertEqual(plugins["vp_1d1v"].task().kind, TaskKind.OBSERVABLE_MODEL)
        self.assertEqual(plugins["ns"].task().kind, TaskKind.SOLVER)
        self.assertEqual(
            plugins["vpfp"].task("observable_model").kind,
            TaskKind.OBSERVABLE_MODEL,
        )
        self.assertEqual(
            plugins["vpfp"].task("reduced_dynamics").kind,
            TaskKind.REDUCED_DYNAMICS,
        )
        self.assertEqual(
            plugins["vpfp"].task("statistics").kind,
            TaskKind.STATISTICS,
        )

    def test_aliases_find_the_same_plugin(self) -> None:
        self.assertEqual(get_plugin("artemis").plugin_id, "vp_1d1v")
        self.assertEqual(get_plugin("navier-stokes").plugin_id, "ns")
        self.assertEqual(
            get_plugin("vlasov_poisson_fokker_planck").plugin_id,
            "vpfp",
        )
        self.assertEqual(get_plugin("gyrokinetic").plugin_id, "gkw")

    def test_gkw_campaign_and_generator_execution_are_explicitly_gated(self) -> None:
        with self.assertRaisesRegex(ValueError, "no staged campaign adapter"):
            legacy_staged_plugin("gkw")
        generator = get_plugin("gkw").generator
        with self.assertRaisesRegex(RuntimeError, "injected stable backend"):
            generator.train_base(None, {})

    def test_canonical_plugins_do_not_materialize_staged_adapters(self) -> None:
        plugins = {plugin.plugin_id: plugin for plugin in built_in_plugins()}

        self.assertTrue(
            all(plugin.legacy_staged_campaign is None for plugin in plugins.values())
        )
        self.assertIsNotNone(plugins["ns"].legacy_staged_campaign_factory)
        self.assertIsNotNone(plugins["vpfp"].legacy_staged_campaign_factory)
        self.assertIsNone(plugins["toy"].legacy_staged_campaign_factory)
        self.assertIsNone(plugins["gkw"].legacy_staged_campaign_factory)

        self.assertIsNotNone(legacy_staged_plugin("ns").legacy_staged_campaign)
        self.assertIsNotNone(legacy_staged_plugin("vpfp").legacy_staged_campaign)

    def test_old_staged_names_are_deprecated_compatibility_only(self) -> None:
        plugin = legacy_staged_plugin("ns")

        with self.assertWarns(DeprecationWarning):
            self.assertIs(plugin.campaign, plugin.legacy_staged_campaign)
        with self.assertWarns(DeprecationWarning):
            self.assertIs(
                plugin_registry.CampaignAdapter,
                LegacyStagedCampaignAdapter,
            )
        with self.assertWarns(DeprecationWarning):
            self.assertIs(plugin_registry.campaign_plugin("ns"), plugin)


    def test_shared_cli_has_no_equation_specific_import(self) -> None:
        cli = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "pde_agent_verifier_solver"
            / "cli.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("plugins.ns", cli)
        self.assertNotIn("plugins.vpfp", cli)
        self.assertIn("legacy_staged_plugin", cli)
        self.assertNotIn("plugin.campaign", cli)


if __name__ == "__main__":
    unittest.main()
