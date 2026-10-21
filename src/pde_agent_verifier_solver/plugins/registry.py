"""The small registry used to discover PDE plugins.

A new PDE is represented by one PDEPlugin value. The value groups the
scientific programs shown in the panel with the optional adapter that turns a
historical JSON campaign into long-running cluster stages. Keeping this list in
one place means the shared CLI and panel never need problem-specific branches.

The product pipeline is configured and launched through the panel. Staged
campaign adapters remain only for reproducing historical experiments and
control arms; their names make that boundary explicit.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import TYPE_CHECKING, Callable, Mapping

if TYPE_CHECKING:
    from pde_agent_verifier_solver.engine.staged import StagedRunPlan
from pde_agent_verifier_solver.sdk import (
    GeneratorProgram,
    MacroMatcher,
    ProblemProgram,
    PublicProblemBundle,
    TaskProgram,
)


@dataclass(frozen=True)
class LegacyStagedCampaignAdapter:
    """Connect a historical PDE JSON recipe to the staged job runner.

    This is not a second product Controller. It is retained for historical
    experiment reproduction and explicit control-arm execution while the
    scheduler mechanics are reused as infrastructure.
    """

    load_config: Callable[..., object]
    preflight: Callable[[object], Mapping[str, object]]
    compile: Callable[[object], StagedRunPlan]


@dataclass(frozen=True)
class PDEPlugin:
    """Everything a researcher adds for one PDE family.

    tasks is a mapping because one equation can support several outcomes. For
    example, VPFP offers an observable model and a true reduced-dynamics task
    without duplicating its Problem or Generator implementation.
    """

    plugin_id: str
    title: str
    description: str
    aliases: tuple[str, ...]
    problem: ProblemProgram
    generator: GeneratorProgram
    tasks: Mapping[str, TaskProgram]
    default_task: str
    public_problem_bundle: PublicProblemBundle | None = None
    legacy_staged_campaign: LegacyStagedCampaignAdapter | None = None
    legacy_staged_campaign_factory: (
        Callable[[], LegacyStagedCampaignAdapter] | None
    ) = None
    macro_matcher: MacroMatcher | None = None

    def __post_init__(self) -> None:
        names = {self.plugin_id, *self.aliases}
        if any(not name.strip() for name in names):
            raise ValueError("PDE plugin ids and aliases cannot be empty")
        if self.default_task not in self.tasks:
            raise ValueError(
                f"default task {self.default_task!r} is not registered for {self.plugin_id}"
            )
        if (
            self.public_problem_bundle is not None
            and not isinstance(self.public_problem_bundle, PublicProblemBundle)
        ):
            raise TypeError(
                "public_problem_bundle must be a PublicProblemBundle or None"
            )
        if (
            self.legacy_staged_campaign is not None
            and self.legacy_staged_campaign_factory is not None
        ):
            raise ValueError(
                "a PDE plugin cannot provide both an eager and lazy legacy adapter"
            )
        if (
            self.legacy_staged_campaign_factory is not None
            and not callable(self.legacy_staged_campaign_factory)
        ):
            raise TypeError("legacy_staged_campaign_factory must be callable")

    def task(self, task_id: str | None = None) -> TaskProgram:
        selected = task_id or self.default_task
        try:
            return self.tasks[selected]
        except KeyError as exc:
            raise ValueError(
                f"unknown task {selected!r} for {self.plugin_id}; "
                f"choose one of {sorted(self.tasks)}"
            ) from exc

    @property
    def campaign(self) -> LegacyStagedCampaignAdapter | None:
        """Deprecated spelling for the history-only staged adapter."""

        warnings.warn(
            "PDEPlugin.campaign is deprecated; use legacy_staged_campaign for "
            "explicit history/control-arm staged runs",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.legacy_staged_campaign


@lru_cache(maxsize=1)
def built_in_plugins() -> tuple[PDEPlugin, ...]:
    """Return built-ins lazily so importing the SDK stays lightweight."""

    from .gkw.product import build_gkw_plugin
    from .ns.product import build_ns_plugin
    from .toy import build_toy_plugin
    from .vpfp.product import build_vpfp_plugin
    from .vp_1d1v.product import build_artemis_plugin

    return (
        build_toy_plugin(),
        build_ns_plugin(),
        build_vpfp_plugin(),
        build_gkw_plugin(),
        build_artemis_plugin(),
    )


def get_plugin(name: str) -> PDEPlugin:
    """Find a plugin by its public id or a short accepted alias."""

    normalized = name.strip().lower()
    for plugin in built_in_plugins():
        if normalized in {plugin.plugin_id, *plugin.aliases}:
            return plugin
    choices = sorted(plugin.plugin_id for plugin in built_in_plugins())
    raise ValueError(f"unsupported problem plugin {name!r}; choose one of {choices}")


@lru_cache(maxsize=None)
def legacy_staged_plugin(name: str) -> PDEPlugin:
    """Find a plugin and require its history/control-arm staged adapter."""

    plugin = get_plugin(name)
    adapter = plugin.legacy_staged_campaign
    if adapter is None and plugin.legacy_staged_campaign_factory is not None:
        adapter = plugin.legacy_staged_campaign_factory()
    if adapter is None:
        raise ValueError(
            f"problem plugin {plugin.plugin_id!r} has no staged campaign adapter "
            "for legacy/history-only execution"
        )
    # The canonical registry value stays free of the old staged lifecycle.
    # Only this explicitly named compatibility lookup materializes it.
    return replace(
        plugin,
        legacy_staged_campaign=adapter,
        legacy_staged_campaign_factory=None,
    )


def campaign_plugin(name: str) -> PDEPlugin:
    """Deprecated compatibility spelling for :func:`legacy_staged_plugin`."""

    warnings.warn(
        "campaign_plugin() is deprecated; use legacy_staged_plugin() for "
        "explicit history/control-arm staged runs",
        DeprecationWarning,
        stacklevel=2,
    )
    return legacy_staged_plugin(name)


def __getattr__(name: str):
    if name == "CampaignAdapter":
        warnings.warn(
            "CampaignAdapter is deprecated; use LegacyStagedCampaignAdapter",
            DeprecationWarning,
            stacklevel=2,
        )
        return LegacyStagedCampaignAdapter
    raise AttributeError(name)
