"""Explicit replay of frozen first-round development, never a warm resume.

The operator imports only accepted Candidate snapshots and unedited Agent notes.
Official results, selection, Guidance and all later Agent turns are rebuilt by
the ordinary CampaignController. The caller must verify the regenerated private
cohort before any recovered Official evaluation can start.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable, Mapping

from pde_agent_verifier_solver.sdk import Artifact, CaseRecord, CaseSource

from .campaign import RoundPlan
from .memory import AgentRoundLog


@dataclass(frozen=True)
class RoundDevelopmentRecovery:
    """Audited first-round input; it contains no old Official result or R2 state.

    ``agent_case_counts`` describe historical development, not draws performed
    by this run. ``development_case_sources`` reserves original Controller ids
    when available; an empty mapping is explicitly reported as unavailable.
    The verification callback must return an audit mapping with ``status`` set
    to ``passed`` only if the source cohort and private query plans meet its
    documented verification checks. It must raise or return a different status
    if those checks fail; missing success status is rejected before scoring.
    """

    round_index: int
    artifacts: tuple[Artifact, ...]
    agent_logs: tuple[AgentRoundLog, ...]
    agent_case_counts: Mapping[str, Mapping[str, int]]
    source_run_id: str
    source_manifest_sha256: str
    verify_official: Callable[
        [RoundPlan, Mapping[str, tuple[CaseRecord, ...]], tuple[Artifact, ...]],
        Mapping[str, object],
    ]
    development_case_sources: Mapping[str, CaseSource] = field(default_factory=dict)
    development_case_id_coverage: str = "unavailable"

    def __post_init__(self) -> None:
        if (
            isinstance(self.round_index, bool)
            or not isinstance(self.round_index, int)
            or self.round_index != 1
        ):
            raise ValueError("development recovery supports only first-round replay")
        artifacts = tuple(self.artifacts)
        logs = tuple(self.agent_logs)
        if any(not isinstance(item, Artifact) for item in artifacts):
            raise TypeError("recovery artifacts must be Artifact snapshots")
        if any(not isinstance(item, AgentRoundLog) for item in logs):
            raise TypeError("recovery notes must be AgentRoundLog values")
        if any(log.round_index != self.round_index for log in logs):
            raise ValueError("recovery notes must belong to the recovered round")
        if not isinstance(self.source_run_id, str) or not self.source_run_id.strip():
            raise ValueError("recovery requires a source run id")
        digest = self.source_manifest_sha256
        if not isinstance(digest, str) or len(digest) != 64 or any(
            char not in "0123456789abcdef" for char in digest
        ):
            raise ValueError("recovery requires a lowercase SHA-256 source manifest")
        if not callable(self.verify_official):
            raise TypeError("recovery requires pre-Official cohort verification")
        allowed = {"original", "base", "guided", "base_perturbed", "guided_perturbed"}
        counts = {}
        for agent_id, values in self.agent_case_counts.items():
            if not isinstance(agent_id, str) or not agent_id:
                raise ValueError("recovery case counts require Agent ids")
            values = dict(values)
            if set(values) - allowed:
                raise ValueError("recovery case counts contain an unknown distribution")
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in values.values()
            ):
                raise ValueError("recovery case counts must be nonnegative integers")
            counts[agent_id] = MappingProxyType(
                {name: values.get(name, 0) for name in sorted(allowed)}
            )
        sources = dict(self.development_case_sources)
        if any(not isinstance(key, str) or not key for key in sources):
            raise ValueError("recovery development Case ids must be nonempty strings")
        if any(not isinstance(value, CaseSource) for value in sources.values()):
            raise TypeError("recovery development Case sources must be CaseSource values")
        coverage = self.development_case_id_coverage
        if coverage not in {"unavailable", "partial", "complete"}:
            raise ValueError("recovery development Case id coverage is invalid")
        if sources and coverage == "unavailable":
            raise ValueError("imported development Case ids require explicit coverage")
        if not sources and coverage == "partial":
            raise ValueError("partial development Case id coverage requires saved ids")
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "agent_logs", logs)
        object.__setattr__(self, "agent_case_counts", MappingProxyType(counts))
        object.__setattr__(self, "development_case_sources", MappingProxyType(sources))

    @property
    def audit_metadata(self) -> Mapping[str, object]:
        return {
            "source_run_id": self.source_run_id,
            "source_manifest_sha256": self.source_manifest_sha256,
            "development_replayed": False,
            "agent_case_counts_provenance": "frozen_source_run",
            "development_case_id_reservations": len(self.development_case_sources),
            "development_case_id_coverage": self.development_case_id_coverage,
            "development_case_id_provenance": (
                "frozen_source_run" if self.development_case_sources else "unavailable"
            ),
            "old_official_results_imported": False,
            "later_round_memory_imported": False,
        }
