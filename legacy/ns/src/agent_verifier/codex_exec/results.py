from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class CodexInvocationRecord:
    round_index: int
    actual_codex_invoked: bool
    codex_binary_path: str | None
    codex_version: str | None
    command_template: str
    returncode: int | None
    stdout_path: str
    stderr_path: str
    wall_time_seconds: float
    prompt_path: str
    workspace_path: str
    generated_submission_path: str | None
    unavailable_reason: str | None = None
    timed_out: bool = False
    submission_exists: bool = False
    model: str | None = None
    reasoning_effort: str | None = None
    environment_overrides_used: bool = True
    network_allowed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CodexRoundRecord:
    round_index: int
    submission_path: str | None
    train_mean_score: float
    fixed_val_mean_score: float
    n_train_rows: int
    n_fixed_val_rows: int
    feedback_path: str
    adversarial_cases_added: int
    invocation: CodexInvocationRecord
    random_cases_added: int = 0

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["invocation"] = self.invocation.as_dict()
        return data


@dataclass
class CodexLoopResult:
    run_dir: str
    rounds: list[CodexRoundRecord] = field(default_factory=list)
    best_submission_path: str | None = None
    best_round_index: int | None = None
    actual_codex_successful_rounds: int = 0
    actual_codex_invoked_rounds: int = 0
    heldout_random_rows: int = 0
    heldout_adversarial_rows: int = 0
    unavailable_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["rounds"] = [r.as_dict() for r in self.rounds]
        return data
