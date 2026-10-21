"""Reference-only replay of a verified VPFP answer-first release bank.

This is a PDE-owned manufactured-case source, not an Agent/controller policy.
It is intentionally limited to reference experiments: it replays already
materialized bank cases in-process and therefore makes no formal isolation or
fresh-generation claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from threading import Lock
from typing import Mapping

from pde_agent_verifier_solver.sdk import (
    CaseRecord,
    CaseSource,
    CaseVisibility,
    CohortKind,
    GenerationKind,
    SampleRequest,
)

from .problem import load_vpfp_case_record


VPFP_RELEASE_BANK_SCHEMA = "vpfp_long_time_answer_first_release_bank_v2"
VPFP_RELEASE_BANK_REFERENCE_ID = "vpfp-release-bank-reference-v1"
VPFP_RELEASE_BANK_IDENTITY_SCHEMA = "vpfp_release_bank_reference_identity_v1"
VPFP_MACRO_OUTPUT_CONTRACT = "vpfp_continuous_macro_output_v1"
_REQUIRED_CASE_FILES = (
    "case_spec.json",
    "hidden_truth_cache.npz",
    "hidden_truth_manifest.json",
    "public_arrays.npz",
    "public_question.json",
)


@dataclass(frozen=True)
class _BankCase:
    case_id: str
    case_dir: Path
    file_sha256: Mapping[str, str]


class VPFPReleaseBankCaseSource:
    """Replay verified bank records with deterministic, non-overlapping draws.

    The source hashes the actual report, decoder checkpoint, and every consumed
    public/private case artifact. Adjacent SampleRequest seeds are independent
    stream roots rather than arithmetic starts for per-case seeds. Each bank
    case can be returned at most once for the lifetime of this source.

    This class is reference-only. It does not prove that the historical
    Generator ran in an isolated process, nor does it regenerate a fresh hidden
    answer from the checkpoint.
    """

    generator_id = VPFP_RELEASE_BANK_REFERENCE_ID
    reference_only = True
    formal_claims_allowed = False
    supports_perturbations = False

    def __init__(
        self,
        report_path: Path | str,
        *,
        decoder_checkpoint: Path | str | None = None,
    ) -> None:
        report = _regular_file(report_path, "VPFP release-bank report")
        report_bytes = report.read_bytes()
        payload = _json_object(report_bytes, report)
        if payload.get("schema") != VPFP_RELEASE_BANK_SCHEMA:
            raise ValueError("unexpected VPFP release-bank report schema")
        if payload.get("passed") is not True:
            raise ValueError("VPFP release-bank report is not passing")
        answer_first = payload.get("answer_first_contract")
        if not isinstance(answer_first, Mapping) or not all(
            answer_first.get(name) is True
            for name in (
                "continuous_hidden_answer_is_primary",
                "public_source_is_exact_jet_residual",
                "coarse_equation_first_solve_is_shape_seed_only",
                "grid_arrays_are_numerical_access_views",
            )
        ):
            raise ValueError("VPFP release bank lacks the answer-first contract")

        declared_value = payload.get("decoder_checkpoint")
        if not isinstance(declared_value, str) or not declared_value.strip():
            raise ValueError("VPFP release-bank report lacks decoder_checkpoint")
        declared_checkpoint = _regular_file(
            _relative_to_report(declared_value, report),
            "VPFP release-bank decoder checkpoint",
        )
        checkpoint = declared_checkpoint
        checkpoint_sha256 = _sha256_file(declared_checkpoint)
        if decoder_checkpoint is not None:
            supplied_checkpoint = _regular_file(
                decoder_checkpoint,
                "VPFP supplied decoder checkpoint",
            )
            supplied_sha256 = _sha256_file(supplied_checkpoint)
            if supplied_sha256 != checkpoint_sha256:
                raise ValueError(
                    "supplied VPFP decoder checkpoint bytes differ from the report"
                )
            checkpoint = supplied_checkpoint

        raw_case_dirs: list[object] = []
        for name in (
            "development_case_dirs",
            "released_completed_case_dirs",
        ):
            values = payload.get(name)
            if not isinstance(values, list) or not values:
                raise ValueError(
                    f"VPFP release-bank report needs nonempty {name}"
                )
            raw_case_dirs.extend(values)

        bank_cases: list[_BankCase] = []
        seen_ids: set[str] = set()
        seen_dirs: set[Path] = set()
        identity_rows: list[dict[str, object]] = []
        for raw_case_dir in raw_case_dirs:
            case_dir = _case_directory(
                _relative_to_report(raw_case_dir, report)
            )
            if case_dir in seen_dirs:
                raise ValueError("VPFP release-bank case directory is duplicated")
            seen_dirs.add(case_dir)
            paths = {
                name: _regular_file(
                    case_dir / name,
                    f"VPFP bank case file {name}",
                )
                for name in _REQUIRED_CASE_FILES
            }
            question = _json_object(
                paths["public_question.json"].read_bytes(),
                paths["public_question.json"],
            )
            case_id = str(question.get("case_id") or "")
            if not case_id.strip() or case_id in seen_ids:
                raise ValueError("VPFP release-bank case ids must be unique")
            if (
                question.get("solver_output_contract_version")
                != VPFP_MACRO_OUTPUT_CONTRACT
            ):
                raise ValueError(
                    f"VPFP public question output contract differs for {case_id}"
                )
            seen_ids.add(case_id)

            case_spec = _json_object(
                paths["case_spec.json"].read_bytes(),
                paths["case_spec.json"],
            )
            if (
                case_spec.get("decoder_checkpoint_sha256")
                != checkpoint_sha256
                or case_spec.get("solver_output_contract_version")
                != VPFP_MACRO_OUTPUT_CONTRACT
            ):
                raise ValueError(
                    f"VPFP bank case {case_id} is not bound to the decoder "
                    "and macro-output contract"
                )
            hidden_manifest = _json_object(
                paths["hidden_truth_manifest.json"].read_bytes(),
                paths["hidden_truth_manifest.json"],
            )
            if hidden_manifest.get("case_id") != case_id:
                raise ValueError(
                    f"VPFP hidden manifest case id differs for {case_id}"
                )
            _verify_npz_contract(
                paths["public_arrays.npz"],
                {
                    "t",
                    "x",
                    "y",
                    "rho0",
                    "momentum0",
                    "kinetic_energy0",
                },
            )
            _verify_npz_contract(
                paths["hidden_truth_cache.npz"],
                {
                    "rho_true",
                    "momentum_true",
                    "kinetic_energy_true",
                },
            )
            hashes = {
                name: _sha256_file(path)
                for name, path in sorted(paths.items())
            }
            continuous = hidden_manifest.get("continuous_primary")
            if not isinstance(continuous, Mapping) or (
                continuous.get("decoder_checkpoint_sha256")
                != checkpoint_sha256
                or continuous.get("public_arrays_sha256")
                != hashes["public_arrays.npz"]
                or continuous.get("materialized_cache_sha256")
                != hashes["hidden_truth_cache.npz"]
            ):
                raise ValueError(
                    f"VPFP hidden manifest byte bindings differ for {case_id}"
                )
            bank_cases.append(_BankCase(case_id, case_dir, hashes))
            identity_rows.append(
                {
                    "case_id": case_id,
                    "files": hashes,
                }
            )

        report_sha256 = sha256(report_bytes).hexdigest()
        identity_document = {
            "schema": VPFP_RELEASE_BANK_IDENTITY_SCHEMA,
            "generator_id": self.generator_id,
            "report_sha256": report_sha256,
            "decoder_checkpoint_sha256": checkpoint_sha256,
            "cases": sorted(identity_rows, key=lambda row: str(row["case_id"])),
        }
        encoded_identity = json.dumps(
            identity_document,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

        self.report_path = report
        self.decoder_checkpoint = checkpoint
        self.report_sha256 = report_sha256
        self.decoder_checkpoint_sha256 = checkpoint_sha256
        self.artifact_sha256 = sha256(encoded_identity).hexdigest()
        self._cases = tuple(bank_cases)
        self._used_case_ids: set[str] = set()
        self._lock = Lock()

    @property
    def case_count(self) -> int:
        return len(self._cases)

    @property
    def remaining_case_count(self) -> int:
        with self._lock:
            return self.case_count - len(self._used_case_ids)

    def sample(self, request: SampleRequest) -> tuple[CaseRecord, ...]:
        if not isinstance(request, SampleRequest):
            raise TypeError("VPFP release-bank sample needs a SampleRequest")
        if request.kind is not GenerationKind.BASE:
            raise ValueError("VPFP release bank supports only clean BASE cases")
        visibility = (
            CaseVisibility.DEVELOPMENT
            if request.cohort is CohortKind.DEVELOPMENT
            else CaseVisibility.HIDDEN
        )
        with self._lock:
            available = tuple(
                row
                for row in self._cases
                if row.case_id not in self._used_case_ids
            )
            if len(available) < request.count:
                raise ValueError(
                    "VPFP release bank has "
                    f"{len(available)} unused cases, requested {request.count}"
                )
            selected = tuple(
                sorted(
                    available,
                    key=lambda row: self._draw_rank(request.seed, row.case_id),
                )[: request.count]
            )
            for row in selected:
                self._verify_case_bytes(row)
            records = tuple(
                load_vpfp_case_record(
                    row.case_dir,
                    visibility=visibility,
                    source=CaseSource.GENERATED,
                    round_index=request.round_index,
                    tags=("vpfp", "release-bank-reference"),
                )
                for row in selected
            )
            self._used_case_ids.update(row.case_id for row in selected)
            return records

    def _verify_case_bytes(self, row: _BankCase) -> None:
        for name, expected in row.file_sha256.items():
            path = _regular_file(
                row.case_dir / name,
                f"VPFP sampled bank case file {name}",
            )

    def _draw_rank(self, stream_seed: int, case_id: str) -> bytes:
        payload = (
            f"{VPFP_RELEASE_BANK_IDENTITY_SCHEMA}\0"
            f"{self.artifact_sha256}\0{stream_seed}\0{case_id}"
        ).encode("utf-8")
        return sha256(payload).digest()


def _relative_to_report(value: object, report: Path) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("VPFP release-bank path cannot be empty")
    path = Path(value).expanduser()
    return path if path.is_absolute() else report.parent / path


def _regular_file(value: Path | str, label: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {candidate}")
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is missing: {resolved}")
    return resolved


def _case_directory(value: Path | str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_symlink():
        raise ValueError(
            f"VPFP release-bank case must not be a symlink: {candidate}"
        )
    resolved = candidate.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(
            f"VPFP release-bank case directory is missing: {resolved}"
        )
    return resolved


def _json_object(raw: bytes, path: Path) -> dict[str, object]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON object in {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _verify_npz_contract(path: Path, required: set[str]) -> None:
    import numpy as np

    with np.load(path, allow_pickle=False) as payload:
        missing = required - set(payload.files)
    if missing:
        raise ValueError(
            f"VPFP bank array {path} lacks keys: {sorted(missing)}"
        )


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "VPFP_RELEASE_BANK_IDENTITY_SCHEMA",
    "VPFP_RELEASE_BANK_REFERENCE_ID",
    "VPFP_RELEASE_BANK_SCHEMA",
    "VPFPReleaseBankCaseSource",
]
