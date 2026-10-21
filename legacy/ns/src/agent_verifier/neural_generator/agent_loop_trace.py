from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any


EVENTS_FILE = "agent_events.jsonl"
ARTIFACTS_FILE = "agent_artifacts.jsonl"
FEEDBACK_FILE = "agent_feedback.jsonl"
MEMORY_CARDS_FILE = "agent_memory_cards.jsonl"
COMPACT_MEMORY_FILE = "agent_memory_compact.json"


MEMORY_INBOX_SCHEMA: dict[str, Any] = {
    "schema_version": "avf.agent_memory_inbox.v0.1",
    "description": "Optional JSONL notes from the solver agent. The harness validates and distills these into long-term memory cards.",
    "record": {
        "type": "reliable_insight|negative_result|failure_mode|overfit_warning|open_hypothesis",
        "text": "short reusable observation, not a per-case lookup",
        "evidence": "brief public-evaluator or code evidence",
        "confidence": "low|medium|high",
    },
}


def empty_compact_memory() -> dict[str, Any]:
    return {
        "schema_version": "avf.agent_memory_compact.v0.1",
        "rounds_summarized": 0,
        "current_champion": None,
        "reliable_insights": [],
        "negative_results": [],
        "failure_modes": [],
        "overfit_warnings": [],
        "open_hypotheses": [],
        "baseline_facts": [],
        "recent_rounds": [],
        "evidence_index": {},
    }


class AgentLoopRecorder:
    """Append-only local observability store for solver-agent adversarial loops.

    The recorder keeps operational traces separate from artifacts, verifier
    feedback, and distilled memory cards. This mirrors modern agent
    observability systems without adding a heavyweight external dependency.
    """

    def __init__(self, root: str | Path, *, trace_id: str | None = None) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.trace_id = trace_id or f"trace_{uuid.uuid4().hex}"
        self._span_counter = 0
        if not (self.root / COMPACT_MEMORY_FILE).exists():
            self.write_compact_memory(empty_compact_memory())

    def new_span_id(self, name: str) -> str:
        self._span_counter += 1
        slug = "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_")[:48] or "span"
        return f"{slug}_{self._span_counter:06d}"

    def event(
        self,
        *,
        kind: str,
        name: str,
        status: str = "ok",
        span_id: str | None = None,
        parent_span_id: str | None = None,
        attributes: dict[str, Any] | None = None,
        input_artifacts: list[str] | None = None,
        output_artifacts: list[str] | None = None,
    ) -> str:
        sid = span_id or self.new_span_id(name)
        row = {
            "schema_version": "avf.agent_event.v0.1",
            "trace_id": self.trace_id,
            "span_id": sid,
            "parent_span_id": parent_span_id,
            "ts": _utc_now(),
            "kind": kind,
            "name": name,
            "status": status,
            "attributes": attributes or {},
            "input_artifacts": input_artifacts or [],
            "output_artifacts": output_artifacts or [],
        }
        _append_jsonl(self.root / EVENTS_FILE, row)
        return sid

    def artifact(
        self,
        *,
        artifact_id: str,
        kind: str,
        path: str | Path,
        created_by_span: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        p = Path(path)
        row = {
            "schema_version": "avf.agent_artifact.v0.1",
            "artifact_id": artifact_id,
            "kind": kind,
            "uri": _display_path(p, self.root),
            "sha256": _sha256_file(p) if p.is_file() else None,
            "created_by_span": created_by_span,
            "metadata": metadata or {},
        }
        _append_jsonl(self.root / ARTIFACTS_FILE, row)
        return row

    def feedback(
        self,
        *,
        feedback_id: str,
        target_span_id: str | None,
        target_artifact_id: str | None,
        metric: str,
        value: Any,
        source: str,
        split: str,
        direction: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        row = {
            "schema_version": "avf.agent_feedback.v0.1",
            "feedback_id": feedback_id,
            "target_span_id": target_span_id,
            "target_artifact_id": target_artifact_id,
            "metric": metric,
            "value": _jsonable(value),
            "source": source,
            "split": split,
            "direction": direction,
            "metadata": metadata or {},
        }
        _append_jsonl(self.root / FEEDBACK_FILE, row)
        return row

    def memory_card(
        self,
        *,
        memory_id: str,
        card_type: str,
        text: str,
        evidence: list[str],
        confidence: str = "medium",
        scope: str = "triplane_agent_loop",
        status: str = "active",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        row = {
            "schema_version": "avf.agent_memory_card.v0.1",
            "memory_id": memory_id,
            "type": card_type,
            "scope": scope,
            "text": text,
            "evidence": evidence,
            "confidence": confidence,
            "status": status,
            "metadata": metadata or {},
        }
        _append_jsonl(self.root / MEMORY_CARDS_FILE, row)
        return row

    def compact_memory(self) -> dict[str, Any]:
        cards = _read_jsonl(self.root / MEMORY_CARDS_FILE)
        feedback = _read_jsonl(self.root / FEEDBACK_FILE)
        compact = empty_compact_memory()
        rounds = sorted({int(row.get("metadata", {}).get("round")) for row in feedback if _is_int(row.get("metadata", {}).get("round"))})
        compact["rounds_summarized"] = len(rounds)
        compact["current_champion"] = _current_champion(feedback)
        compact["recent_rounds"] = _recent_rounds(feedback)
        compact["evidence_index"] = {str(card.get("memory_id")): card.get("evidence", []) for card in cards[-80:] if card.get("memory_id")}
        buckets = {
            "reliable_insight": "reliable_insights",
            "negative_result": "negative_results",
            "failure_mode": "failure_modes",
            "overfit_warning": "overfit_warnings",
            "open_hypothesis": "open_hypotheses",
            "baseline_fact": "baseline_facts",
            "champion_strategy": "reliable_insights",
        }
        latest_card_indexes = {
            str(card.get("memory_id")): index
            for index, card in enumerate(cards)
            if card.get("memory_id")
        }
        for card_index, card in enumerate(cards):
            memory_id = card.get("memory_id")
            if memory_id and latest_card_indexes[str(memory_id)] != card_index:
                continue
            if card.get("status", "active") != "active":
                continue
            key = buckets.get(str(card.get("type")))
            if key is None:
                continue
            item = {"memory_id": card.get("memory_id"), "text": card.get("text"), "confidence": card.get("confidence")}
            compact[key].append(item)
        for key in ("reliable_insights", "negative_results", "failure_modes", "overfit_warnings", "open_hypotheses", "baseline_facts"):
            compact[key] = compact[key][-12:]
        return compact

    def write_compact_memory(self, payload: dict[str, Any] | None = None) -> Path:
        out = self.root / COMPACT_MEMORY_FILE
        out.write_text(json.dumps(payload or self.compact_memory(), indent=2, sort_keys=True) + "\n")
        return out


def write_workspace_memory_files(workspace: str | Path, compact_memory: dict[str, Any]) -> None:
    root = Path(workspace)
    root.mkdir(parents=True, exist_ok=True)
    (root / COMPACT_MEMORY_FILE).write_text(json.dumps(compact_memory, indent=2, sort_keys=True) + "\n")
    (root / "agent_memory_inbox_schema.json").write_text(json.dumps(MEMORY_INBOX_SCHEMA, indent=2, sort_keys=True) + "\n")
    inbox = root / "agent_memory_inbox.jsonl"
    if not inbox.exists():
        inbox.write_text("")


def ingest_agent_memory_inbox(recorder: AgentLoopRecorder, inbox: str | Path, *, round_index: int, evidence_span: str) -> list[dict[str, Any]]:
    path = Path(inbox)
    if not path.exists():
        return []
    accepted: list[dict[str, Any]] = []
    valid_types = {"reliable_insight", "negative_result", "failure_mode", "overfit_warning", "open_hypothesis"}
    for idx, row in enumerate(_read_jsonl(path)):
        card_type = str(row.get("type", "")).strip()
        text = str(row.get("text", "")).strip()
        if card_type not in valid_types or not text:
            continue
        if len(text) > 1200:
            text = text[:1200] + "..."
        confidence = str(row.get("confidence", "medium")).strip().lower()
        if confidence not in {"low", "medium", "high"}:
            confidence = "medium"
        evidence = [evidence_span]
        if row.get("evidence"):
            evidence.append(str(row.get("evidence"))[:300])
        accepted.append(
            recorder.memory_card(
                memory_id=f"round{round_index:03d}_agent_note_{idx:03d}",
                card_type=card_type,
                text=text,
                evidence=evidence,
                confidence=confidence,
                metadata={"round": int(round_index), "source": "agent_memory_inbox"},
            )
        )
    return accepted


def record_eval_feedback(
    recorder: AgentLoopRecorder,
    *,
    round_index: int,
    eval_payload: dict[str, Any],
    split: str,
    target_span_id: str,
    target_artifact_id: str | None,
) -> None:
    summary = eval_payload.get("summary", {}) if isinstance(eval_payload, dict) else {}
    for metric_name, direction in (
        ("mean_relative_l2", "higher_is_harder_for_generator_lower_is_better_for_solver"),
        ("median_relative_l2", "higher_is_harder_for_generator_lower_is_better_for_solver"),
        ("q90_relative_l2", "higher_is_harder_for_generator_lower_is_better_for_solver"),
        ("max_relative_l2", "higher_is_harder_for_generator_lower_is_better_for_solver"),
        ("ok", "higher_is_better"),
    ):
        if metric_name in summary:
            recorder.feedback(
                feedback_id=f"round{round_index:03d}_{split}_{metric_name}",
                target_span_id=target_span_id,
                target_artifact_id=target_artifact_id,
                metric=metric_name,
                value=summary.get(metric_name),
                source="hidden_verifier",
                split=split,
                direction=direction,
                metadata={"round": int(round_index), "summary_key": metric_name},
            )


def write_round_memory_cards(
    recorder: AgentLoopRecorder,
    *,
    round_index: int,
    summary: dict[str, Any],
    evidence_span: str,
) -> None:
    generated = summary.get("generated_summary") or {}
    guard = summary.get("guard_summary") or {}
    gen_mean = generated.get("mean_relative_l2")
    guard_mean = guard.get("mean_relative_l2")
    verification = summary.get("hard_verification_summary") or {}
    accepted = verification.get("accepted")
    total = verification.get("n")
    verification_mode = str(verification.get("mode") or "")
    if _is_number(gen_mean):
        recorder.memory_card(
            memory_id=f"round{round_index:03d}_generated_difficulty",
            card_type="baseline_fact",
            text=f"Round {round_index:03d} generated current mean relative L2 was {float(gen_mean):.6g}.",
            evidence=[evidence_span],
            confidence="high",
            metadata={"round": int(round_index), "mean_relative_l2": float(gen_mean)},
        )
    if _is_number(gen_mean) and _is_number(guard_mean):
        ratio = float(gen_mean) / max(float(guard_mean), 1.0e-12)
        card_type = "reliable_insight" if ratio >= 0.9 else "open_hypothesis"
        text = (
            f"Round {round_index:03d} generated-vs-guard difficulty ratio was {ratio:.3g}; "
            f"generated mean {float(gen_mean):.6g}, guard mean {float(guard_mean):.6g}."
        )
        if ratio < 0.9:
            text += " Future policy updates should harden samples against the current solver."
        recorder.memory_card(
            memory_id=f"round{round_index:03d}_guard_comparison",
            card_type=card_type,
            text=text,
            evidence=[evidence_span],
            confidence="high",
            metadata={"round": int(round_index), "generated_guard_ratio": ratio},
        )
    if _is_int(accepted) and _is_int(total):
        contract_only = verification_mode == "continuous_mms_contract_only"
        recorder.memory_card(
            memory_id=(
                f"round{round_index:03d}_continuous_mms_contract"
                if contract_only
                else f"round{round_index:03d}_hard_verifier"
            ),
            card_type="baseline_fact",
            text=(
                f"Round {round_index:03d} continuous MMS contract passed for "
                f"{int(accepted)}/{int(total)} generated cases."
                if contract_only
                else f"Round {round_index:03d} hard verifier accepted {int(accepted)}/{int(total)} generated cases."
            ),
            evidence=[evidence_span],
            confidence="high",
            metadata={
                "round": int(round_index),
                "accepted": int(accepted),
                "n": int(total),
                "verification_mode": verification_mode,
            },
        )


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _current_champion(feedback: list[dict[str, Any]]) -> dict[str, Any] | None:
    rows = [
        row
        for row in feedback
        if row.get("split") == "generated_current"
        and row.get("metric") == "mean_relative_l2"
        and _is_number(row.get("value"))
    ]
    if not rows:
        return None
    last = sorted(rows, key=lambda row: int(row.get("metadata", {}).get("round", -1)))[-1]
    return {
        "round": int(last.get("metadata", {}).get("round", -1)),
        "generated_mean_relative_l2": float(last["value"]),
        "evidence": last.get("target_span_id"),
    }


def _recent_rounds(feedback: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_round: dict[int, dict[str, Any]] = {}
    for row in feedback:
        meta = row.get("metadata", {})
        if not _is_int(meta.get("round")) or row.get("metric") != "mean_relative_l2":
            continue
        round_index = int(meta["round"])
        item = by_round.setdefault(round_index, {"round": round_index})
        item[f"{row.get('split')}_mean_relative_l2"] = row.get("value")
    return [by_round[key] for key in sorted(by_round)[-8:]]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value
