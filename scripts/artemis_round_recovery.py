"""Read-only, round-one recovery inputs for the interrupted Artemis campaign.

This is an input adapter, not a second campaign controller.  It imports only
committed R1 nominations and R1 Agent notes, never the later campaign memory.
Successful worker inputs were deleted by the old executor.  Consequently the
adapter checks all ordered public case identities but reports the smaller set
of surviving numerical/public-plan references honestly.  Hidden answers are
not loaded; saved oracle capabilities are ignored, never reused or exported.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping, Sequence

from pde_agent_verifier_solver.engine.agent_wire import AgentWireCodec
from pde_agent_verifier_solver.engine.candidate_case import candidate_facing_case
from pde_agent_verifier_solver.engine.candidate_execution import (
    _case_from_payload, _plan_from_payload,
)
from pde_agent_verifier_solver.engine.memory import AgentRoundLog
from pde_agent_verifier_solver.engine.python_candidate import PythonCandidateProgram
from pde_agent_verifier_solver.sdk import Artifact, Case, EvaluationPlan, TaskKind, freeze_candidate_bundle, inspect_candidate_bundle

_SHA = re.compile(r"[0-9a-f]{64}\Z")
_NOMINATIONS = tuple(f"candidate-r1-{index:04d}" for index in range(1, 5))
_JSON_LIMIT = 8 * 1024 * 1024
_JOURNAL_LIMIT = 128 * 1024 * 1024


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def content_digest(value: object) -> str:
    """Stable exact-value fingerprint, independent of wire sidecar locations."""
    import numpy as np

    def normalized(item: object) -> object:
        if isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            if array.dtype.hasobject:
                raise ValueError("object arrays cannot be recovery evidence")
            return {"array_dtype": array.dtype.str, "shape": list(array.shape),
                    "bytes_sha256": _digest_bytes(array.tobytes())}
        if isinstance(item, np.generic):
            return normalized(item.item())
        if isinstance(item, Enum):
            return normalized(item.value)
        if is_dataclass(item):
            return {field.name: normalized(getattr(item, field.name)) for field in fields(item)}
        if isinstance(item, Mapping):
            if not all(isinstance(key, str) for key in item):
                raise ValueError("recovery evidence keys must be strings")
            return {key: normalized(part) for key, part in item.items()}
        if isinstance(item, (tuple, list)):
            return [normalized(part) for part in item]
        return item

    return _digest_bytes(json.dumps(normalized(value), sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode())


def _inside(path: Path, root: Path) -> Path:
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root.resolve(strict=True)):
        raise ValueError("recovery input escaped its allowed source directory")
    if path.is_symlink():
        raise ValueError("recovery input must not be a symlink")
    return resolved


def _read_json(path: Path, root: Path) -> dict:
    path = _inside(path, root)
    if not path.is_file() or path.stat().st_size > _JSON_LIMIT:
        raise ValueError("recovery JSON input is missing or exceeds its bound")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("recovery JSON input must be an object")
    return value


def _journal(path: Path, root: Path):
    path = _inside(path, root)
    if path.stat().st_size > _JOURNAL_LIMIT:
        raise ValueError("recovery journal exceeds its bound")
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if len(line.encode()) > _JSON_LIMIT:
                raise ValueError("recovery journal row exceeds its bound")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("recovery journal row must be an object")
            yield value


def _readonly_codec(sidecar_root: Path, workspace: Path) -> AgentWireCodec:
    # The normal constructor mkdir/chmods its root, even when used to decode.
    # Avoid those writes to a forensic source; decode itself opens read-only.
    checked = _inside(sidecar_root, workspace)
    if not checked.is_dir():
        raise ValueError("source array sidecars are missing")
    codec = AgentWireCodec(max_sidecar_bytes=16 * 1024 * 1024)
    codec.sidecar_root = checked
    return codec


@dataclass(frozen=True)
class SavedOfficialReference:
    case_index: int
    case_id: str
    public_case_sha256: str
    private_plan_sha256: str


@dataclass(frozen=True)
class FrozenRoundOne:
    artifacts: tuple[Artifact, ...]
    agent_logs: tuple[AgentRoundLog, ...]
    agent_case_counts: Mapping[str, Mapping[str, int]]
    original_case_ids: tuple[str, ...]
    references: tuple[SavedOfficialReference, ...]
    source_manifest_sha256: str
    source_config: Mapping[str, object]
    manifest: Mapping[str, object]


def load_frozen_round_one(
    source_campaign: str | Path,
    *,
    expected_source_head: str,
    expected_config_sha256: str,
    expected_backend_config_sha256: str,
) -> FrozenRoundOne:
    """Validate pinned inputs without importing R2 or reusing old capabilities.

    ``expected_config_sha256`` pins the actual saved resolved_config.json bytes,
    not the original launcher configuration file (whose formatting may differ).
    The source manifest is safe to publish: only hashes, counts and public
    candidate identities, without source paths, private coordinates or tokens.
    """
    root = Path(source_campaign).resolve(strict=True)
    config = _read_json(root / "resolved_config.json", root)
    state = _read_json(root / "source_state.json", root)
    provenance = _read_json(root / "input_provenance.json", root)
    if state.get("commit") != expected_source_head or state.get("status_porcelain"):
        raise ValueError("source code revision is not the expected clean deployment")
    config_digest = _digest_bytes((root / "resolved_config.json").read_bytes())
    if config.get("official", {}).get("base") != 128:
        raise ValueError("R1 recovery requires the original 128-Base Official cohort")
    if any(config["official"].get(key, 0) for key in
           ("original", "guided", "base_perturbed", "guided_perturbed")):
        raise ValueError("R1 recovery supports only the original clean Base cohort")

    memory_event = None
    finish_event = None
    finished_agents = {}
    event_digests = []
    for event in _journal(root / "events.jsonl", root):
        # Stop at the exact R1 boundary: later event contents are not imported.
        event_digests.append(content_digest(event))
        if event.get("round_index") == 1:
            if event.get("event_type") == "agent_finished":
                row = event["payload"]
                if row["agent_id"] in finished_agents:
                    raise ValueError("duplicate R1 Agent completion event")
                finished_agents[row["agent_id"]] = row
            if event.get("event_type") == "campaign_memory_updated":
                if memory_event is not None:
                    raise ValueError("duplicate R1 memory event")
                memory_event = event
            if event.get("event_type") == "round_finished":
                finish_event = event
                break
        elif event.get("round_index", 0) not in (None, 0):
            raise ValueError("later-round events appeared before the R1 boundary")
    if memory_event is None or finish_event is None:
        raise ValueError("source has no complete R1 memory/Official boundary")
    memory = memory_event["payload"]["round_memory"]
    if memory.get("round") != 1 or memory.get("status") != "completed":
        raise ValueError("source R1 memory is not a completed round")
    history = memory_event["payload"]["candidate_history"]
    if {row["artifact_id"] for row in history} != {"candidate-s0", *_NOMINATIONS}:
        raise ValueError("source R1 candidate history is incomplete or contaminated")
    history_by_id = {row["artifact_id"]: row for row in history}
    agents = config.get("agents", [])
    agent_ids = tuple(agent["agent_id"] for agent in agents)
    if len(agent_ids) != 2 or len(set(agent_ids)) != 2 or set(finished_agents) != set(agent_ids):
        raise ValueError("source must contain exactly the two configured R1 Agents")
    logs = tuple(AgentRoundLog(round_index=1, **row) for row in memory["agents"])
    if len(logs) != 2 or {log.agent_id for log in logs} != set(agent_ids):
        raise ValueError("source R1 Agent notes are incomplete")

    artifacts = []
    nomination_digests = []
    for agent_id in agent_ids:
        matches = tuple((root / "agent_turns").glob(f"round-0001-{agent_id}-*"))
        if len(matches) != 1:
            raise ValueError("source R1 Agent workspace is not unique")
        workspace = _inside(matches[0], root / "agent_turns")
        execution_id = workspace.name.rsplit("-", 1)[-1]
        nominations = sorted((workspace / "nominations").glob("nomination-*.json"))
        ids = finished_agents[agent_id].get("artifact_ids", [])
        if len(nominations) != len(ids) or finished_agents[agent_id].get("nomination_count") != len(ids):
            raise ValueError("source R1 committed nominations disagree with Controller completion")
        for index, (path, artifact_id) in enumerate(zip(nominations, ids, strict=True), 1):
            nomination = _read_json(path, workspace)
            if (path.name != f"nomination-{index:08d}.json"
                    or nomination.get("sequence") != index
                    or nomination.get("execution_id") != execution_id
                    or nomination.get("schema") != "canonical-agent-nomination-v1"):
                raise ValueError("source R1 nomination journal identity mismatch")
            if artifact_id not in _NOMINATIONS:
                raise ValueError("non-R1 nomination rejected")
            row = nomination["payload"]
            saved = history_by_id[artifact_id]
            if (saved.get("round") != 1 or saved.get("agent_id") != agent_id
                    or saved.get("title") != row.get("name")
                    or saved.get("summary") != row.get("summary")):
                raise ValueError("source R1 nomination differs from its Official identity")
            raw_bundle = Path(row["bundle_path"])
            if raw_bundle.is_absolute():
                raise ValueError("source nomination must use a workspace-relative frozen bundle")
            bundle_path = _inside(workspace / raw_bundle, workspace / ".submitted-candidates")
            bundle = inspect_candidate_bundle(bundle_path)
            # Prefer the Controller-frozen identical copy and validate both.
            controller_bundle = _inside(workspace / ".controller-candidates" / bundle.tree_sha256,
                                        workspace / ".controller-candidates")
            inspect_candidate_bundle(controller_bundle, recorded_identity=bundle.tree_sha256)
            metadata = {"candidate_bundle_tree_sha256": bundle.tree_sha256,
                        "candidate_bundle_entrypoint": bundle.entrypoint,
                        "candidate_bundle_bytes": bundle.total_bytes,
                        "submitted_artifact_id": f"submitted-{bundle.tree_sha256[:20]}"}
            artifacts.append(Artifact(
                artifact_id=artifact_id, kind=TaskKind.OBSERVABLE_MODEL,
                name=row["name"], summary=row["summary"], usage=row["usage"],
                round_index=1, agent_id=agent_id, parent_artifact_id="candidate-s0",
                metadata=metadata, program=PythonCandidateProgram(
                    controller_bundle / "candidate_bundle.json", expected_tree_sha256=bundle.tree_sha256),
            ))
            nomination_digests.append(content_digest(nomination))
    if tuple(artifact.artifact_id for artifact in artifacts) != _NOMINATIONS:
        raise ValueError("source R1 nomination order differs from the canonical four candidates")

    execution_root = root / "candidate_execution"
    completed = []
    retained = []
    first_official_execution = {}
    for audit in _journal(execution_root / "candidate_execution_audits.jsonl", root):
        if audit.get("phase") != "official" or audit.get("artifact_id") not in _NOMINATIONS:
            continue
        # The old R1 winner is evaluated again as the incumbent in R2.  The
        # canonical source has one first Official execution per nominee; do
        # not import later batches merely because their artifact ID is old.
        artifact_id = audit["artifact_id"]
        execution_id = audit.get("execution_id")
        if not isinstance(execution_id, str) or not execution_id:
            raise ValueError("source Official audit execution ID is invalid")
        first_official_execution.setdefault(artifact_id, execution_id)
        if execution_id != first_official_execution[artifact_id]:
            continue
        if audit["artifact_id"] == "candidate-r1-0002":
            completed.append(audit)
        # Do not scan or decode current R2 worker inputs.
        workspace = Path(audit["workspace"])
        if (workspace / "request.json").is_file():
            retained.append(audit)
    if (len(completed) != 128 or {row["case_index"] for row in completed} != set(range(128))
            or len({row["execution_id"] for row in completed}) != 1
            or any(row.get("status") != "completed" for row in completed)):
        raise ValueError("source complete R1 Official identity audit is missing or ambiguous")
    original_ids = tuple(row["case_id"] for row in sorted(completed, key=lambda row: row["case_index"]))
    if len(set(original_ids)) != 128:
        raise ValueError("source R1 Official Case identities are not unique")
    refs = {}
    by_id = {artifact.artifact_id: artifact for artifact in artifacts}
    for audit in retained:
        workspace = _inside(Path(audit["workspace"]), execution_root)
        request = _read_json(workspace / "request.json", workspace)
        index = request.get("case_index")
        if (isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 128
                or index != audit.get("case_index")
                or request.get("execution_id") != audit.get("execution_id")):
            raise ValueError("retained R1 worker input identity mismatch")
        raw_artifact = request.get("artifact", {})
        artifact = by_id[audit["artifact_id"]]
        if (raw_artifact.get("artifact_id") != artifact.artifact_id
                or raw_artifact.get("round_index") != 1
                or raw_artifact.get("agent_id") != artifact.agent_id
                or raw_artifact.get("bundle_tree_sha256") != artifact.program.tree_sha256):
            raise ValueError("retained worker artifact is not its frozen R1 nomination")
        codec = _readonly_codec(Path(request["sidecar_root"]), workspace)
        case = _case_from_payload(request["case"], codec)
        plan = _plan_from_payload(request["plan"], codec)
        if case.case_id != original_ids[index] or plan.phase.value != "official":
            raise ValueError("retained R1 public Case or private plan identity mismatch")
        reference = SavedOfficialReference(index, case.case_id, content_digest(case), content_digest(plan))
        if index in refs and refs[index] != reference:
            raise ValueError("same R1 Official index has inconsistent saved public/plan values")
        refs[index] = reference
    if not refs:
        raise ValueError("R1 recovery requires retained numerical and private-plan references")
    references = tuple(refs[index] for index in sorted(refs))
    counts = finish_event["payload"]["agent_case_counts"]
    if set(counts) != set(agent_ids):
        raise ValueError("R1 development case counts do not match configured Agents")
    manifest = {
        "schema": "artemis-r1-recovery-input-v1", "round_index": 1,
        "source_head": expected_source_head, "resolved_config_sha256": config_digest,
        "backend_config_sha256": expected_backend_config_sha256,
        "input_provenance_sha256": content_digest(provenance),
        "r1_event_prefix_sha256": content_digest(event_digests),
        "r1_nomination_journals_sha256": content_digest(nomination_digests),
        "r1_agent_logs_sha256": content_digest(logs),
        "nominees": [{"artifact_id": item.artifact_id, "bundle_tree_sha256": item.program.tree_sha256}
                     for item in artifacts],
        "initial_bundle_tree_sha256": provenance["initial_bundle_tree_sha256"],
        "official_case_count": 128, "identity_reference_count": 128,
        "ordered_case_ids_sha256": content_digest(original_ids),
        "numerical_reference_count": len(references),
        "numerical_reference_indices": [row.case_index for row in references],
        "numerical_references_sha256": content_digest(references),
        "full_numerical_reference_saved": len(references) == 128,
        "hidden_answer_saved": False,
        "development_controller_case_registry_restored": False,
        "later_round_data_imported": False,
        "source_writes_performed": False,
    }
    return FrozenRoundOne(tuple(artifacts), logs, counts, original_ids, references,
                          content_digest(manifest), config, manifest)


def stage_frozen_round_one(frozen: FrozenRoundOne, destination: str | Path) -> FrozenRoundOne:
    """Copy only the four verified R1 programs into a new private run root.

    Source evidence and manifest identity remain unchanged.  A new destination
    is mandatory, so this cannot overwrite an active or previously staged run.
    """
    target = Path(destination)
    if target.exists() or target.is_symlink():
        raise FileExistsError("R1 recovery staging requires a new destination")
    if len(frozen.artifacts) != 4 or tuple(a.artifact_id for a in frozen.artifacts) != _NOMINATIONS:
        raise ValueError("only the original four R1 nominees may be staged")
    # Inspect every source before creating any destination artifact.
    inspected = []
    for artifact in frozen.artifacts:
        bundle = inspect_candidate_bundle(artifact.program.bundle_locator,
                                          recorded_identity=artifact.program.tree_sha256)
        inspected.append(bundle)
    target.mkdir(parents=True, mode=0o700)
    staged = []
    for artifact, bundle in zip(frozen.artifacts, inspected, strict=True):
        copied = freeze_candidate_bundle(bundle, target)
        staged.append(replace(artifact, program=PythonCandidateProgram(
            copied.manifest_path, expected_tree_sha256=copied.tree_sha256)))
    return replace(frozen, artifacts=tuple(staged))


def verify_official_replay(
    frozen: FrozenRoundOne,
    cases: Sequence[Case],
    plans: Sequence[EvaluationPlan],
    *,
    candidate_id_seed: int = 0,
) -> Mapping[str, object]:
    """Fail closed before scoring if regenerated R1 order or evidence differs."""
    if len(cases) != 128 or len(plans) != 128:
        raise ValueError("recovered R1 Official cohort must contain exactly 128 Cases and plans")
    references = {row.case_index: row for row in frozen.references}
    for index, (case, plan) in enumerate(zip(cases, plans, strict=True)):
        projected = candidate_facing_case(case, id_seed=candidate_id_seed)
        if projected.case_id != frozen.original_case_ids[index]:
            raise ValueError(f"regenerated R1 Official Case identity/order mismatch at index {index}")
        if plan.phase.value != "official":
            raise ValueError("recovered private plan must be Official")
        reference = references.get(index)
        if reference is not None:
            if content_digest(projected) != reference.public_case_sha256:
                raise ValueError(f"regenerated R1 public numerical content mismatch at index {index}")
            if content_digest(plan) != reference.private_plan_sha256:
                raise ValueError(f"regenerated R1 private plan mismatch at index {index}")
    return {"status": "passed", "identity_order_verified_count": 128,
            "public_numerical_verified_count": len(references),
            "private_plan_verified_count": len(references),
            "total_official_case_count": 128,
            "full_original_numerical_cohort_verified": len(references) == 128,
            "source_manifest_sha256": frozen.source_manifest_sha256,
            "remaining_cases": "deterministically regenerated; original numerical inputs were not retained",
            "hidden_answer_equality_claimed": False}
