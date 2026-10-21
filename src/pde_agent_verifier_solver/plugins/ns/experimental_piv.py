"""Canonical composition helper for completed experimental NS/PIV data.

This module performs no Campaign lifecycle work. It validates one completed
PIV manifest, restores one already-selected coefficient-flow checkpoint, and
returns ordinary SDK objects consumed by ProductApplication and
CampaignController.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from math import isfinite
import os
from pathlib import Path
import stat
from types import MappingProxyType
from typing import Mapping

from pde_agent_verifier_solver.engine import GeneratorRecipe, PythonCandidateCodec
from pde_agent_verifier_solver.sdk import (
    Artifact,
    DatasetInput,
    DatasetKind,
    GeneratorCapabilities,
    GeneratorCheckpoint,
    PreparedDataset,
    TaskKind,
    inspect_candidate_bundle,
)

from .field_generator import NSDatasetCoefficientGeneratorBackend
from .generator import NSStructuredGenerator
from .problem import NSProblem
from .tasks import NSSolverTask


class NSExperimentalPIVGenerator(NSStructuredGenerator):
    """Checkpoint-bound field Generator with fresh online sampling.

    This capability declaration belongs only to the completed-PIV backend.
    The unbound stock NS Generator remains conservative because it does not
    know whether a usable online backend has been supplied.
    """

    capabilities = GeneratorCapabilities(
        offline=True,
        online=True,
        guidance=True,
        perturbations=True,
        trainable=False,
    )

    def __init__(self, backend: NSDatasetCoefficientGeneratorBackend) -> None:
        super().__init__(backend)
        # NSStructuredGenerator derives generic bound-backend capabilities in
        # its constructor. This assembly is narrower: it has already restored
        # the user-selected checkpoint and must never retrain it.
        self.capabilities = type(self).capabilities


@dataclass(frozen=True)
class CanonicalNSPIVAssembly:
    """Validated inputs for one canonical experimental NS Project."""

    dataset: DatasetInput
    prepared_dataset: PreparedDataset
    problem: NSProblem
    task: NSSolverTask
    generator: NSExperimentalPIVGenerator
    checkpoint: GeneratorCheckpoint
    generator_settings: Mapping[str, object]
    representation_path: Path
    project_reconstruction: Mapping[str, object]
    reconstruction_settings: Mapping[str, object]
    terminal_final_test_cases: int
    official_hidden_cases: int

    def programs(self, agents: Mapping[str, object]):
        """Build the normal ProgramBundle after the user supplies Agents."""

        from pde_agent_verifier_solver.app import ProgramBundle

        if not isinstance(agents, Mapping) or not agents:
            raise TypeError("experimental NS assembly needs a nonempty Agent mapping")
        return ProgramBundle(
            problem=self.problem,
            task=self.task,
            generator=self.generator,
            agents=dict(agents),
            # PIV Candidates read only the sparse observations in Case
            # public_data.  Do not advertise the manufactured-flow forcing
            # oracle/session surface used by the numerical NS backend.
            public_problem_bundle=None,
        )

    def generator_recipe(
        self,
        *,
        guidance_enabled: bool = True,
        perturbations_enabled: bool = True,
        settings: Mapping[str, object] | None = None,
    ) -> GeneratorRecipe:
        """Return the canonical recipe fragment that reuses the frozen model."""

        merged = {**dict(self.generator_settings), **dict(settings or {})}
        merged.pop("generator_checkpoint", None)
        return GeneratorRecipe(
            basis_size=int(self.checkpoint.payload.model_config.feature_dim),
            guidance_enabled=guidance_enabled,
            perturbations_enabled=perturbations_enabled,
            settings=merged,
            checkpoint=self.checkpoint,
        )

    def bind_initial_candidate(
        self,
        *,
        bundle_path: str | Path,
        storage_root: str | Path,
        candidate_codec: PythonCandidateCodec | None = None,
    ) -> Artifact:
        """Validate and freeze a lightweight Gappy-POD S0 algorithm bundle.

        The trajectory POD is a Project-owned frozen data dependency exposed
        through ``Case.public_data['project_reconstruction']``.  It must not be
        vendored into an Agent-editable Candidate tree: Candidate snapshots
        freeze algorithm bytes, while the Project descriptor independently
        freezes the representation identity and selected reconstruction
        settings.
        """

        codec = candidate_codec or PythonCandidateCodec(task_kind=TaskKind.SOLVER)
        if codec.task_kind is not TaskKind.SOLVER:
            raise ValueError("PIV initial Candidate codec must bind Solver artifacts")
        bundle = inspect_candidate_bundle(bundle_path, limits=codec.limits)
        members = {member.path: member for member in bundle.members}
        settings_name = "assets/reconstruction_settings.json"
        if settings_name not in members:
            raise ValueError(
                "PIV initial Candidate bundle must declare reconstruction settings"
            )
        embedded_assets = tuple(
            member.path
            for member in bundle.members
            if member.path.lower().endswith((".npz", ".npy", ".h5", ".hdf5"))
        )
        if embedded_assets:
            raise ValueError(
                "PIV Candidate must reference the Project reconstruction "
                f"artifact instead of vendoring array assets: {embedded_assets}"
            )
        raw_settings = json.loads(
            (bundle.root / settings_name).read_text(encoding="utf-8")
        )
        selected = _reconstruction_settings(raw_settings)
        if selected != dict(self.reconstruction_settings):
            raise ValueError(
                "PIV initial Candidate settings differ from the method selected "
                "on reconstruction-validation data"
            )
        artifact = codec.bind_existing(
            bundle.manifest_path,
            storage_root=storage_root,
            artifact_id="ns-piv-gappy-pod-s0",
            name="Validated sparse-PIV Gappy-POD starting solver",
            summary=(
                "Trajectory POD and regularized Gappy-POD coefficients selected "
                "on the reconstruction-validation split."
            ),
            usage=(
                "Call prepare_solution(case, solve_budget), then evaluate the "
                "returned handle at requested physical points."
            ),
        )
        return replace(
            artifact,
            metadata={
                **dict(artifact.metadata),
                "project_reconstruction_artifact_id": self.project_reconstruction[
                    "artifact_id"
                ],
                "project_reconstruction_sha256": self.project_reconstruction[
                    "sha256"
                ],
                "project_reconstruction_size_bytes": self.project_reconstruction[
                    "size_bytes"
                ],
            },
        )


def assemble_completed_piv(
    *,
    manifest_path: str | Path,
    generator_checkpoint_path: str | Path,
    dataset_id: str = "ns-experimental-piv",
    dataset_settings: Mapping[str, object] | None = None,
    generator_settings: Mapping[str, object] | None = None,
) -> CanonicalNSPIVAssembly:
    """Validate and bind completed PIV data to the one canonical pipeline.

    dataset_settings chooses manifest roles and Case counts. In particular,
    final_test_split is placed before hidden_split in the private inventory
    because the canonical Controller seals Terminal cohorts before it
    allocates per-round Official cohorts.

    The checkpoint is restored here once and returned as a GeneratorCheckpoint;
    the subsequent canonical Run therefore calls load and never retrains the
    selected model.
    """

    manifest = Path(manifest_path).expanduser().resolve()
    checkpoint_path = Path(generator_checkpoint_path).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    try:
        manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("completed PIV manifest is not valid JSON") from exc
    if not isinstance(manifest_payload, Mapping):
        raise ValueError("completed PIV manifest must be an object")

    dataset = DatasetInput(
        dataset_id=dataset_id,
        kind=DatasetKind.EXPERIMENTAL,
        location=str(manifest),
        settings=dict(dataset_settings or {}),
    )
    problem = NSProblem()
    prepared = problem.prepare_dataset(dataset)
    representation_value = prepared.training_summary.get("representation")
    if not representation_value:
        raise ValueError("completed PIV manifest has no trajectory POD artifact")
    representation_path = Path(str(representation_value)).expanduser().resolve()
    if not representation_path.is_file():
        raise FileNotFoundError(representation_path)
    reconstruction_settings = _reconstruction_settings(
        manifest_payload.get(
            "selected_method",
            {"rank": 48, "prior_precision": 10.0},
        )
    )
    project_reconstruction = _project_reconstruction_descriptor(
        representation_path,
        reconstruction_settings,
    )
    prepared = _attach_project_reconstruction(
        prepared,
        project_reconstruction,
    )
    terminal_count = int(
        prepared.training_summary.get("terminal_final_test_cases", 0)
    )
    official_count = int(
        prepared.training_summary.get(
            "official_hidden_cases",
            len(prepared.hidden_cases) - terminal_count,
        )
    )
    if terminal_count < 1:
        raise ValueError(
            "canonical experimental NS needs a distinct nonempty final_test split"
        )
    if official_count < 1:
        raise ValueError(
            "canonical experimental NS needs a distinct nonempty verifier split"
        )

    runtime_settings = {
        "device": "cpu",
        "require_validation_pass": True,
        **dict(generator_settings or {}),
    }
    runtime_settings["generator_checkpoint"] = str(checkpoint_path)
    backend = NSDatasetCoefficientGeneratorBackend()
    generator = NSExperimentalPIVGenerator(backend)
    loaded = generator.train(prepared, runtime_settings)
    state = loaded.payload
    checkpoint = GeneratorCheckpoint(
        generator_id=loaded.generator_id,
        payload=state,
        location=str(checkpoint_path),
        report={
            **dict(loaded.report),
            "source_checkpoint_schema": state.validation_report.get(
                "checkpoint_schema",
                "ns_dataset_latent_flow_v2",
            ),
            "validation": dict(state.validation_report),
            "training_case_count": len(state.train_latents),
            "validation_case_count": len(state.validation_latents),
        },
    )
    campaign_settings = dict(runtime_settings)
    campaign_settings.pop("generator_checkpoint", None)
    campaign_settings.pop("checkpoint_training_cases", None)
    campaign_settings.pop("checkpoint_validation_cases", None)
    return CanonicalNSPIVAssembly(
        dataset=dataset,
        prepared_dataset=prepared,
        problem=problem,
        task=NSSolverTask(),
        generator=generator,
        checkpoint=checkpoint,
        generator_settings=MappingProxyType(campaign_settings),
        representation_path=representation_path,
        project_reconstruction=project_reconstruction,
        reconstruction_settings=MappingProxyType(reconstruction_settings),
        terminal_final_test_cases=terminal_count,
        official_hidden_cases=official_count,
    )


def _reconstruction_settings(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("completed PIV selected_method must be an object")
    rank = value.get("rank")
    precision = value.get("prior_precision")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError("completed PIV selected_method.rank must be positive")
    if (
        isinstance(precision, bool)
        or not isinstance(precision, (int, float))
        or not isfinite(float(precision))
        or float(precision) < 0.0
    ):
        raise ValueError(
            "completed PIV selected_method.prior_precision must be nonnegative"
        )
    return {"rank": rank, "prior_precision": float(precision)}


def _project_reconstruction_descriptor(
    representation_path: Path,
    settings: Mapping[str, object],
) -> Mapping[str, object]:
    """Bind one regular representation file to a stable public identity."""

    path = representation_path.expanduser().resolve()
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("PIV Project reconstruction artifact must be a regular file")
    size_bytes, digest = _hash_stable_regular_file(path)
    return MappingProxyType(
        {
            "schema": "pde_project_reconstruction_artifact_v1",
            "artifact_id": "ns-piv-trajectory-pod",
            "artifact_kind": "trajectory_pod",
            "format": "numpy_npz",
            "locator": str(path),
            "sha256": digest,
            "size_bytes": size_bytes,
            "selected_method": MappingProxyType(
                {
                    "method_id": "gappy_pod_map",
                    "rank": int(settings["rank"]),
                    "prior_precision": float(settings["prior_precision"]),
                }
            ),
        }
    )


def _hash_stable_regular_file(path: Path) -> tuple[int, str]:
    before = path.lstat()
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        opened_identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
        )
        if not stat.S_ISREG(opened.st_mode) or opened_identity != before_identity:
            raise ValueError("PIV Project reconstruction artifact changed before read")
        digest = hashlib.sha256()
        bytes_read = 0
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
            bytes_read += len(block)
        after = os.fstat(descriptor)
        current = path.lstat()
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        current_identity = (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
        )
        if (
            bytes_read != opened.st_size
            or after_identity != opened_identity
            or current_identity != opened_identity
        ):
            raise ValueError("PIV Project reconstruction artifact changed while read")
        return int(after.st_size), digest.hexdigest()
    finally:
        os.close(descriptor)


def _attach_project_reconstruction(
    prepared: PreparedDataset,
    descriptor: Mapping[str, object],
) -> PreparedDataset:
    """Attach one Project artifact descriptor to every PIV Case role."""

    def attach(record):
        public_data = dict(record.case.public_data)
        existing = public_data.get("project_reconstruction")
        if existing is not None and existing != descriptor:
            raise ValueError("PIV Case declares a different reconstruction artifact")
        public_data["project_reconstruction"] = descriptor
        return replace(record, case=replace(record.case, public_data=public_data))

    return replace(
        prepared,
        development_cases=tuple(attach(row) for row in prepared.development_cases),
        hidden_cases=tuple(attach(row) for row in prepared.hidden_cases),
        generator_training_cases=tuple(
            attach(row) for row in prepared.generator_training_cases
        ),
        generator_validation_cases=tuple(
            attach(row) for row in prepared.generator_validation_cases
        ),
        training_summary={
            **dict(prepared.training_summary),
            "project_reconstruction": descriptor,
        },
    )


__all__ = [
    "CanonicalNSPIVAssembly",
    "NSExperimentalPIVGenerator",
    "assemble_completed_piv",
]
