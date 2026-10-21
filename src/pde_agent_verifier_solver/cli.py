"""Customer commands for launching and watching PDE campaigns."""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path
from urllib.parse import quote

from pde_agent_verifier_solver.panel.server import run_panel


_LEGACY_STAGED_COMMANDS = {
    "legacy-plan",
    "legacy-run",
    "legacy-resume",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pde-agent-verifier-solver",
        description=(
            "Open the canonical PDE product panel, or explicitly reproduce "
            "a history/control-arm staged campaign."
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    panel = subparsers.add_parser(
        "panel",
        help="Open the canonical product pipeline panel",
    )
    panel.add_argument("--host", default="127.0.0.1")
    panel.add_argument("--port", default=8000, type=int)
    panel_source = panel.add_mutually_exclusive_group()
    panel_source.add_argument("--run-root", default="")
    panel_source.add_argument("--runs-root", default="")

    panel_sync = subparsers.add_parser("_panel-sync")
    panel_sync.add_argument("--runs-root", action="append", required=True)
    panel_sync.add_argument("--token-file", default=".env.panel-sync")
    panel_sync.add_argument("--interval", default=5.0, type=float)
    panel_sync.add_argument("--once", action="store_true")

    for name, help_text in (
        (
            "legacy-plan",
            "Validate/display a history or control-arm staged campaign",
        ),
        (
            "legacy-run",
            "Launch a history or control-arm staged campaign",
        ),
        (
            "legacy-resume",
            "Resume a history or control-arm Slurm staged campaign",
        ),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("config")
        command.add_argument("--dataset", default="")
        command.add_argument("--run-root", default="")
        command.add_argument("--backend", choices=("local", "slurm"), default="")

    status = subparsers.add_parser(
        "status",
        help="Show persistent staged-scheduler state",
    )
    status.add_argument("run_root")

    event = subparsers.add_parser("_stage-event")
    event.add_argument("--run-root", required=True)
    event.add_argument("--stage", required=True)
    event.add_argument("--status", required=True)
    event.add_argument("--message", default="")
    event.add_argument("--exit-code", type=int)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "_panel-sync":
        from pde_agent_verifier_solver.panel.publisher import (
            load_sync_token,
            run_panel_sync,
        )

        run_panel_sync(
            args.runs_root,
            load_sync_token(args.token_file),
            interval_seconds=args.interval,
            once=args.once,
        )
        return
    if args.command in {None, "panel"}:
        if args.command is None:
            run_panel()
        elif args.run_root:
            from pde_agent_verifier_solver.panel.staged_server import run_staged_panel

            run_staged_panel(args.run_root, args.host, args.port)
        elif args.runs_root:
            from pde_agent_verifier_solver.panel.staged_server import (
                run_campaign_dashboard,
            )

            run_campaign_dashboard(args.runs_root, args.host, args.port)
        else:
            run_panel(args.host, args.port)
        return
    if args.command == "status":
        from pde_agent_verifier_solver.engine.staged import read_staged_run

        print(json.dumps(read_staged_run(args.run_root), indent=2, sort_keys=True))
        return
    if args.command == "_stage-event":
        from pde_agent_verifier_solver.engine.staged import record_stage_event

        record_stage_event(
            args.run_root,
            args.stage,
            args.status,
            message=args.message,
            exit_code=args.exit_code,
        )
        return

    if args.command not in _LEGACY_STAGED_COMMANDS:
        parser.error(f"unsupported command {args.command!r}")

    from pde_agent_verifier_solver.engine.staged import (
        LocalStageRunner,
        SlurmStageLauncher,
    )

    plugin, config = _load_legacy_staged_config(
        args.config,
        dataset=args.dataset or None,
        run_root=args.run_root or None,
        backend=args.backend or None,
    )
    # The explicit name prevents this history/control-arm path from being
    # mistaken for the canonical CampaignController product pipeline.
    adapter = plugin.legacy_staged_campaign
    assert adapter is not None
    preflight = adapter.preflight(config)
    plan = adapter.compile(config)
    if args.command == "legacy-plan":
        print(
            json.dumps(
                {"preflight": preflight, "plan": plan.summary()},
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "legacy-resume":
        if config.backend != "slurm":
            raise ValueError("legacy-resume currently supports Slurm campaigns only")
        result = SlurmStageLauncher().resume(plan, legacy_staged_intent=True)
    elif config.backend == "local":
        result = LocalStageRunner().run(plan, legacy_staged_intent=True)
    else:
        result = SlurmStageLauncher().launch(plan, legacy_staged_intent=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    panel_url = os.environ.get("PDE_PANEL_URL", "").rstrip("/")
    if panel_url:
        print(f"Watch this run: {panel_url}?run={quote(plan.run_root.name)}")
    else:
        print(f"Watch all runs: ./pde panel --runs-root {plan.run_root.parent}")


def _load_legacy_staged_config(
    path: Path | str,
    *,
    dataset: Path | str | None,
    run_root: Path | str | None,
    backend: str | None,
):
    from pde_agent_verifier_solver.plugins.registry import legacy_staged_plugin

    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    problem = str(payload.get("problem", "")).lower()
    plugin = legacy_staged_plugin(problem)
    adapter = plugin.legacy_staged_campaign
    assert adapter is not None
    config = adapter.load_config(
        source,
        dataset_override=dataset,
        run_root_override=run_root,
        backend_override=backend,
    )
    return plugin, config


def _load_config(
    path: Path | str,
    *,
    dataset: Path | str | None,
    run_root: Path | str | None,
    backend: str | None,
):
    """Deprecated private compatibility wrapper for old integrations."""

    warnings.warn(
        "_load_config() is a legacy staged helper; use "
        "_load_legacy_staged_config() explicitly",
        DeprecationWarning,
        stacklevel=2,
    )
    return _load_legacy_staged_config(
        path,
        dataset=dataset,
        run_root=run_root,
        backend=backend,
    )


if __name__ == "__main__":
    main()
