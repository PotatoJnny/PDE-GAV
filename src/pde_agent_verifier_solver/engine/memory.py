"""Chronological Campaign memory plus a separate Candidate history."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Mapping, Sequence

from pde_agent_verifier_solver.sdk import Artifact, PromotionMetricSummary


CAMPAIGN_MEMORY_HEADER = "# Campaign Memory Log"


def _positive_round(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("memory round_index must be a positive integer")
    return value


def _nonempty_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be text")
    if not value.strip():
        raise ValueError(f"{name} cannot be empty")
    return value


def _fenced_agent_markdown(markdown: str) -> str:
    """Frame untrusted notes without altering their evidence bytes."""

    longest_run = 0
    current_run = 0
    for character in markdown:
        if character == "~":
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0
    fence = "~" * max(3, longest_run + 1)
    separator = "" if markdown.endswith(("\n", "\r")) else "\n"
    return f"{fence}markdown\n{markdown}{separator}{fence}"


@dataclass(frozen=True)
class AgentRoundLog:
    """One Agent's unedited Markdown notes from one round."""

    round_index: int
    agent_id: str
    agent_name: str
    markdown: str
    status: str = "completed"

    def __post_init__(self) -> None:
        object.__setattr__(self, "round_index", _positive_round(self.round_index))
        object.__setattr__(self, "agent_id", _nonempty_text(self.agent_id, "agent_id"))
        object.__setattr__(
            self,
            "agent_name",
            _nonempty_text(self.agent_name, "agent_name"),
        )
        object.__setattr__(
            self,
            "markdown",
            _nonempty_text(self.markdown, "Agent round log"),
        )
        if self.status not in {"completed", "failed", "timed_out"}:
            raise ValueError("Agent round log status is invalid")


@dataclass(frozen=True)
class CampaignRoundLog:
    """The Agent logs and Controller outcome retained for one round."""

    round_index: int
    agent_logs: tuple[AgentRoundLog, ...]
    controller_markdown: str
    merged_markdown: str
    status: str = "completed"

    def __post_init__(self) -> None:
        round_index = _positive_round(self.round_index)
        agent_logs = tuple(self.agent_logs)
        if any(not isinstance(log, AgentRoundLog) for log in agent_logs):
            raise TypeError("agent_logs must contain AgentRoundLog values")
        if any(log.round_index != round_index for log in agent_logs):
            raise ValueError("Agent logs must belong to the same Campaign round")
        agent_ids = tuple(log.agent_id for log in agent_logs)
        if len(agent_ids) != len(set(agent_ids)):
            raise ValueError("a round cannot contain two logs from the same Agent")
        object.__setattr__(self, "agent_logs", agent_logs)
        object.__setattr__(
            self,
            "controller_markdown",
            _nonempty_text(self.controller_markdown, "Controller round log"),
        )
        object.__setattr__(
            self,
            "merged_markdown",
            _nonempty_text(self.merged_markdown, "merged round log"),
        )
        if self.status not in {"completed", "aborted"}:
            raise ValueError("Campaign round status is invalid")


@dataclass(frozen=True)
class CampaignMemorySnapshot:
    """Immutable full log passed to every Agent entering the next round."""

    rounds: tuple[CampaignRoundLog, ...] = ()

    def __post_init__(self) -> None:
        rounds = tuple(self.rounds)
        if any(not isinstance(round_log, CampaignRoundLog) for round_log in rounds):
            raise TypeError("Campaign memory rounds must be CampaignRoundLog values")
        indices = tuple(round_log.round_index for round_log in rounds)
        if indices != tuple(range(1, len(rounds) + 1)):
            raise ValueError("Campaign memory rounds must be sequential")
        object.__setattr__(self, "rounds", rounds)

    @property
    def markdown(self) -> str:
        """Render the one complete handoff document from the stored rounds."""

        return "\n\n".join(
            (CAMPAIGN_MEMORY_HEADER, *(row.merged_markdown for row in self.rounds))
        )


class CampaignMemoryLog:
    """Append one deterministic, unabridged block after every round."""

    def __init__(self) -> None:
        self._rounds: list[CampaignRoundLog] = []

    def append_round(
        self,
        round_index: int,
        agent_logs: Sequence[AgentRoundLog],
        controller_markdown: str,
        *,
        status: str = "completed",
    ) -> CampaignRoundLog:
        expected = len(self._rounds) + 1
        if round_index != expected:
            raise ValueError(
                f"Campaign memory expected round {expected}, received {round_index}"
            )
        logs = tuple(agent_logs)
        controller_text = _nonempty_text(
            controller_markdown,
            "Controller round log",
        )
        pieces = [f"## Round {round_index}"]
        for log in logs:
            pieces.extend(
                (
                    "### Agent-authored notes (fallible) — "
                    f"{log.agent_name} ({log.agent_id}) — {log.status}",
                    _fenced_agent_markdown(log.markdown),
                )
            )
        pieces.extend(
            (f"### Controller outcome (Official) — {status}", controller_text)
        )
        # Framework headings are normalized at construction time. Agent
        # Markdown remains byte-for-byte evidence inside a content-sized fence;
        # even a forged heading or fence stays inside the fallible Agent block.
        merged = "\n\n".join(pieces)
        round_log = CampaignRoundLog(
            round_index=round_index,
            agent_logs=logs,
            controller_markdown=controller_text,
            merged_markdown=merged,
            status=status,
        )
        self._rounds.append(round_log)
        return round_log

    def snapshot(self) -> CampaignMemorySnapshot:
        return CampaignMemorySnapshot(tuple(self._rounds))


def agent_round_log_template(
    round_index: int,
    agent_name: str,
    *,
    incumbent_name: str | None,
) -> str:
    """Return the small Markdown form placed in an external Agent workspace."""

    _positive_round(round_index)
    name = _nonempty_text(agent_name, "agent_name")
    incumbent = incumbent_name or "None"
    return f"""# Round {round_index} — {name}

## Round goal

State the problem you are trying to solve this round.

## Starting point

- Incumbent: {incumbent}

## Experiments

For each meaningful attempt, record why you tried it, what changed, which
development Cases you evaluated, the feedback you saw, and what you learned.
For public sources you actually read, include the title and URL, the method learned,
why you adopted or rejected it, and its measured outcome when tested.

## Final submission

Record the submitted Candidate(s), or explain why you submitted none.

## Handoff

Record what worked, what failed, and the most useful next experiment.
"""


@dataclass(frozen=True)
class CandidateHistoryNode:
    node_id: str
    round_index: int
    agent_id: str
    artifact_id: str
    parent_node_id: str | None
    title: str
    summary: str
    official_distributions: Mapping[str, PromotionMetricSummary] = field(
        default_factory=dict
    )
    failures: tuple[str, ...] = ()
    status: str = "not_evaluated"
    selected: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _nonempty_text(self.node_id, "node_id"))
        if (
            isinstance(self.round_index, bool)
            or not isinstance(self.round_index, int)
            or self.round_index < 0
        ):
            raise ValueError("Candidate History round_index must be nonnegative")
        object.__setattr__(self, "agent_id", _nonempty_text(self.agent_id, "agent_id"))
        object.__setattr__(
            self,
            "artifact_id",
            _nonempty_text(self.artifact_id, "artifact_id"),
        )
        if self.parent_node_id is not None:
            object.__setattr__(
                self,
                "parent_node_id",
                _nonempty_text(self.parent_node_id, "parent_node_id"),
            )
        object.__setattr__(self, "title", _nonempty_text(self.title, "title"))
        if not isinstance(self.summary, str):
            raise TypeError("summary must be text")
        distributions = dict(self.official_distributions)
        if any(not isinstance(name, str) or not name.strip() for name in distributions):
            raise ValueError("Candidate History distribution names cannot be empty")
        if any(
            not isinstance(summary, PromotionMetricSummary)
            for summary in distributions.values()
        ):
            raise TypeError(
                "Candidate History distributions need PromotionMetricSummary values"
            )
        failures = tuple(self.failures)
        if any(not isinstance(value, str) or not value for value in failures):
            raise ValueError("Candidate History failures must be nonempty strings")
        if self.status not in {"not_evaluated", "eligible", "ineligible"}:
            raise ValueError("Candidate History status is invalid")
        if self.status == "not_evaluated" and (distributions or failures):
            raise ValueError(
                "a not_evaluated Candidate cannot have Official results"
            )
        if not isinstance(self.selected, bool):
            raise TypeError("Candidate History selected must be bool")
        object.__setattr__(
            self,
            "official_distributions",
            MappingProxyType(distributions),
        )
        object.__setattr__(self, "failures", failures)


@dataclass(frozen=True)
class CandidateHistorySnapshot:
    nodes: tuple[CandidateHistoryNode, ...] = ()

    def __post_init__(self) -> None:
        nodes = tuple(self.nodes)
        if any(not isinstance(node, CandidateHistoryNode) for node in nodes):
            raise TypeError(
                "Candidate History snapshot must contain CandidateHistoryNode values"
            )
        node_ids = tuple(node.node_id for node in nodes)
        artifact_ids = tuple(node.artifact_id for node in nodes)
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("Candidate History contains duplicate node ids")
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("Candidate History contains duplicate artifact ids")
        known: set[str] = set()
        for node in nodes:
            if node.parent_node_id is not None and node.parent_node_id not in known:
                raise ValueError(
                    "Candidate History parents must precede their children"
                )
            known.add(node.node_id)
        object.__setattr__(self, "nodes", nodes)

    @property
    def selected_path(self) -> tuple[CandidateHistoryNode, ...]:
        return tuple(node for node in self.nodes if node.selected)


class CandidateHistory:
    def __init__(self) -> None:
        self._nodes: list[CandidateHistoryNode] = []
        self._artifact_to_node: dict[str, str] = {}

    def contains(self, artifact_id: str) -> bool:
        return artifact_id in self._artifact_to_node

    def needs_official_result(self, artifact_id: str) -> bool:
        node_id = self._artifact_to_node.get(artifact_id)
        if node_id is None:
            raise KeyError(f"Candidate History does not contain {artifact_id!r}")
        return next(
            node.status == "not_evaluated"
            for node in self._nodes
            if node.node_id == node_id
        )

    def add_candidate(
        self,
        artifact: Artifact,
    ) -> CandidateHistoryNode:
        """Record one submitted program before its first Official evaluation."""

        if not isinstance(artifact, Artifact):
            raise TypeError("Candidate History requires an Artifact")
        if self.contains(artifact.artifact_id):
            raise ValueError(
                f"Candidate History already contains {artifact.artifact_id!r}"
            )
        parent_node_id = None
        if artifact.parent_artifact_id is not None:
            parent_node_id = self._artifact_to_node.get(artifact.parent_artifact_id)
            if parent_node_id is None:
                raise ValueError(
                    "Candidate History parent is unknown: "
                    f"{artifact.parent_artifact_id!r}"
                )
        node = CandidateHistoryNode(
            node_id=f"candidate-history-{len(self._nodes) + 1}",
            round_index=artifact.round_index,
            agent_id=artifact.agent_id,
            artifact_id=artifact.artifact_id,
            parent_node_id=parent_node_id,
            title=artifact.name,
            summary=artifact.summary,
        )
        self._nodes.append(node)
        self._artifact_to_node[artifact.artifact_id] = node.node_id
        return node

    def record_official_result(
        self,
        artifact_id: str,
        distributions: Mapping[str, PromotionMetricSummary],
        failures: Sequence[str],
        *,
        technically_eligible: bool,
    ) -> CandidateHistoryNode:
        """Attach the Candidate's first common Official result."""

        node_id = self._artifact_to_node.get(artifact_id)
        if node_id is None:
            raise KeyError(f"Candidate History does not contain {artifact_id!r}")
        current = next(node for node in self._nodes if node.node_id == node_id)
        if current.status != "not_evaluated":
            raise ValueError("Candidate History already has an Official result")
        replacement = replace(
            current,
            official_distributions=dict(distributions),
            failures=tuple(failures),
            status="eligible" if technically_eligible else "ineligible",
        )
        self._nodes = [
            replacement if node.node_id == node_id else node for node in self._nodes
        ]
        return replacement

    def mark_selected(self, artifact: Artifact) -> None:
        node_id = self._artifact_to_node.get(artifact.artifact_id)
        if node_id is None:
            raise KeyError(
                f"Candidate History does not contain {artifact.artifact_id!r}"
            )
        self._nodes = [
            replace(node, selected=True) if node.node_id == node_id else node
            for node in self._nodes
        ]

    def snapshot(self) -> CandidateHistorySnapshot:
        return CandidateHistorySnapshot(tuple(self._nodes))


__all__ = [
    "AgentRoundLog",
    "CAMPAIGN_MEMORY_HEADER",
    "CampaignMemoryLog",
    "CampaignMemorySnapshot",
    "CampaignRoundLog",
    "CandidateHistory",
    "CandidateHistoryNode",
    "CandidateHistorySnapshot",
    "agent_round_log_template",
]
