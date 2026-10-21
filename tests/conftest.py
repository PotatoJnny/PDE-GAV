"""Shared test configuration.

Historical configurations and launchers refer to external resources through
site placeholders (see docs/SITE_CONFIGURATION.md). The NS canonical launcher
expands environment variables when it loads a JSON configuration, so the tests
provide stable non-existent defaults for those placeholders unless the caller
already set them. The Slurm account placeholder is deliberately left unset.
"""
import os

_SITE_DEFAULTS = {
    "PDE_GAV_SOURCE_ROOT": "/pde-gav-site/source",
    "PDE_GAV_WORKTREES_ROOT": "/pde-gav-site/worktrees",
    "PDE_GAV_RUN_ROOT": "/pde-gav-site/runs",
    "PDE_GAV_DEPS_ROOT": "/pde-gav-site/deps",
    "PDE_GAV_SHARED_DATA_ROOT": "/pde-gav-site/shared-data",
    "PDE_GAV_ARCHIVE_ROOT": "/pde-gav-site/archive",
    "PDE_GAV_AGGREGATE_ROOT": "/pde-gav-site/aggregate",
    "PDE_GAV_VENV": "/pde-gav-site/venv",
    "PDE_GAV_PYTHON": "/pde-gav-site/venv/bin/python",
    "CODEX_BINARY": "/pde-gav-site/bin/codex",
    "CODEX_AUTH_FILE": "/pde-gav-site/codex/auth.json",
}
for _key, _value in _SITE_DEFAULTS.items():
    os.environ.setdefault(_key, _value)
