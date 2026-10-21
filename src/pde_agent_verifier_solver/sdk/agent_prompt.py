"""Small, reusable Markdown sections for the user's scientific instructions.

Executable protocol values remain in the run configuration. This module only
composes user-authored task, resource and role descriptions.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class AgentPromptSections:
    task: str
    resources: str = ""
    role: str = ""

    def __post_init__(self) -> None:
        for name in ("task", "resources", "role"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"Prompt {name} must be Markdown text")
            object.__setattr__(self, name, value.strip())
        if not self.task:
            raise ValueError("Prompt task must not be empty")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> AgentPromptSections:
        if not isinstance(value, Mapping) or set(value) - {"task", "resources", "role"}:
            raise ValueError("Prompt sections accept task, resources and role only")
        return cls(task=value.get("task", ""), resources=value.get("resources", ""),
                   role=value.get("role", ""))

    @classmethod
    def from_files(cls, files: Mapping[str, str], *, relative_to: Path) -> AgentPromptSections:
        if not isinstance(files, Mapping) or set(files) - {"task", "resources", "role"}:
            raise ValueError("Prompt files accept task, resources and role only")
        texts = {}
        for name, raw in files.items():
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError(f"Prompt {name} file needs a path")
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = relative_to / path
            texts[name] = path.read_text(encoding="utf-8")
        return cls.from_mapping(texts)

    def as_dict(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in ("task", "resources", "role")}

    def render(self) -> str:
        return "\n\n".join(
            f"## {title}\n\n{text}" for title, text in (
                ("Task", self.task), ("Available resources", self.resources),
                ("Research direction", self.role),
            ) if text
        )


def resolve_agent_prompt_files(config: Mapping[str, object], *, relative_to: Path) -> dict:
    """Snapshot optional shared/role files into the existing Agent prompt field.

    The returned config keeps both the resolved sections and their original file
    references. Resolved sections take precedence on replay, so saved configs
    remain usable when source Markdown files move or are edited.
    """
    result = dict(config)
    shared = config.get("prompt_files")
    if shared is not None and (not isinstance(shared, Mapping)
                              or set(shared) - {"task", "resources"}):
        raise ValueError("Shared prompt_files accept task and resources only")
    agents = config.get("agents", ())
    if not isinstance(agents, (list, tuple)):
        raise TypeError("agents must be a sequence")
    resolved = []
    for original in agents:
        if not isinstance(original, Mapping):
            raise TypeError("Each Agent must be an object")
        row = dict(original)
        if "prompt_sections" in row:
            sections = AgentPromptSections.from_mapping(row["prompt_sections"])
        elif shared is not None or "role_file" in row:
            if "prompt" in row:
                raise ValueError("Choose inline prompt or prompt_files, not both")
            files = dict(shared or {})
            if "role_file" in row:
                files["role"] = row["role_file"]
            sections = AgentPromptSections.from_files(files, relative_to=relative_to)
        else:
            resolved.append(row)
            continue
        row["prompt_sections"] = sections.as_dict()
        row["prompt"] = sections.render()
        resolved.append(row)
    if "agents" in config:
        result["agents"] = resolved
    return result
