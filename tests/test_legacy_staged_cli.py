from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from pde_agent_verifier_solver import cli


def _subcommands(parser: argparse.ArgumentParser) -> set[str]:
    action = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    return set(action.choices)


def test_cli_reserves_staged_science_for_explicit_legacy_commands(capsys) -> None:
    parser = cli._build_parser()

    assert _subcommands(parser) == {
        "panel",
        "_panel-sync",
        "legacy-plan",
        "legacy-run",
        "legacy-resume",
        "status",
        "_stage-event",
    }
    with patch.object(cli, "_load_legacy_staged_config") as legacy_loader:
        with pytest.raises(SystemExit) as error:
            cli.main(["run", "campaign.json"])
        legacy_loader.assert_not_called()
    assert error.value.code == 2
    assert "invalid choice" in capsys.readouterr().err

    help_text = parser.format_help()
    assert "canonical product pipeline panel" in help_text
    assert "history or control-arm staged campaign" in help_text


def test_legacy_plan_still_preflights_and_compiles_without_launching(capsys) -> None:
    plan = SimpleNamespace(
        run_root=Path("/tmp/history-control-arm"),
        summary=lambda: {"stages": ["prepare", "evaluate"]},
    )
    adapter = SimpleNamespace(
        preflight=lambda config: {"config_name": config.name},
        compile=lambda config: plan,
    )
    plugin = SimpleNamespace(legacy_staged_campaign=adapter)
    config = SimpleNamespace(name="history-control-arm", backend="slurm")

    with patch.object(
        cli,
        "_load_legacy_staged_config",
        return_value=(plugin, config),
    ):
        cli.main(["legacy-plan", "campaign.json"])

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "plan": {"stages": ["prepare", "evaluate"]},
        "preflight": {"config_name": "history-control-arm"},
    }


def test_gkw_legacy_staged_config_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "gkw.json"
    source.write_text(json.dumps({"problem": "gkw"}), encoding="utf-8")

    with pytest.raises(ValueError, match="no staged campaign adapter"):
        cli._load_legacy_staged_config(
            source,
            dataset=None,
            run_root=None,
            backend=None,
        )
