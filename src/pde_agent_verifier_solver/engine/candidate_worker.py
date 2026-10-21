"""Trusted entry point for one isolated Candidate/Case execution."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .candidate_execution import run_candidate_worker


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--response", required=True, type=Path)
    args = parser.parse_args(argv)
    return run_candidate_worker(args.request, args.response)


if __name__ == "__main__":  # pragma: no cover - subprocess entry point
    raise SystemExit(main())
