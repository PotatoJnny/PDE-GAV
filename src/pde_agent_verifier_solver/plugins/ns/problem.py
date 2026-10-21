"""Dataset-driven Navier--Stokes ProblemProgram.

``NSProblem`` owns the PDE family, not a geometry. It reads the selected
manifest and asks a small dataset adapter to construct public Cases and private
truth providers. Adding another NS storage format therefore means adding an
adapter; a customer never edits the shared engine or panel.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from pde_agent_verifier_solver.sdk import (
    Case,
    DatasetInput,
    FormField,
    PanelCard,
    PreparedDataset,
    ProgramInfo,
)

from .datasets import (
    NSDatasetAdapter,
    default_ns_dataset_adapters,
)


class NSProblem:
    """Two-dimensional incompressible flow on dataset-defined geometry."""

    info = ProgramInfo(
        "ns-2d-dataset-geometry",
        "2D incompressible Navier--Stokes",
        "Velocity or velocity-pressure problems on the geometry and "
        "coordinates declared by the selected dataset.",
    )

    def __init__(
        self,
        adapters: Sequence[NSDatasetAdapter] | None = None,
    ) -> None:
        self._adapters = tuple(adapters or default_ns_dataset_adapters())
        if not self._adapters:
            raise ValueError("NSProblem needs at least one dataset adapter")

    def setup_fields(self) -> tuple[FormField, ...]:
        return (
            FormField(
                "generator_training_split",
                "Generator training split",
                "text",
                "auto",
                "Full-field Cases used only to fit the Generator.",
            ),
            FormField(
                "generator_validation_split",
                "Generator validation split",
                "text",
                "auto",
                "Independent full-field Cases used only to validate generated samples.",
            ),
            FormField(
                "development_split",
                "Agent-visible split",
                "text",
                "auto",
                "Defaults are chosen by the selected dataset adapter.",
            ),
            FormField(
                "hidden_split",
                "Initial hidden split",
                "text",
                "auto",
                "Held-out manifest split used by the controller.",
            ),
            FormField(
                "final_test_split",
                "Final untouched test split",
                "text",
                "auto",
                "Evaluated once after all adversarial rounds finish.",
            ),
            FormField(
                "generator_training_cases",
                "Generator training Cases",
                "integer",
                0,
                "Zero uses every Case in the Generator training split.",
            ),
            FormField(
                "generator_validation_cases",
                "Generator validation Cases",
                "integer",
                0,
                "Zero uses every Case in the Generator validation split.",
            ),
            FormField(
                "development_cases",
                "Initial development cases",
                "integer",
                0,
                "Zero uses every Case in a completed-field split.",
            ),
            FormField(
                "hidden_cases",
                "Initial hidden cases",
                "integer",
                0,
                "Zero uses every Case in a completed-field split.",
            ),
            FormField(
                "final_test_cases",
                "Final untouched test Cases",
                "integer",
                0,
                "Zero uses every Case in the final test split.",
            ),
        )

    def prepare_dataset(self, dataset: DatasetInput) -> PreparedDataset:
        if not dataset.location:
            raise ValueError(
                "NS needs a dataset manifest. Raw experimental observations must "
                "first be completed by a ReconstructionProgram."
            )
        manifest_path = Path(dataset.location).expanduser().resolve()
        manifest = _read_object(manifest_path)
        for adapter in self._adapters:
            if adapter.accepts(manifest):
                normalized = _with_adapter_defaults(
                    dataset,
                    adapter.format_id,
                    manifest,
                )
                return adapter.prepare(normalized, manifest_path, manifest)
        formats = ", ".join(adapter.format_id for adapter in self._adapters)
        raise ValueError(
            f"NS dataset manifest {manifest_path} has no registered adapter; "
            f"built-in formats: {formats}"
        )

    def case_card(self, case: Case) -> PanelCard:
        geometry = _mapping(case.public_data.get("geometry"))
        coordinates = _mapping(case.public_data.get("coordinates"))
        fields = _case_fields(case)
        return PanelCard(
            title=f"NS case {case.case_id}",
            summary=(
                "Incompressible flow on "
                f"{geometry.get('label', 'dataset-defined geometry')}."
            ),
            values={
                "source": case.source.value,
                "geometry": geometry.get("label", "dataset-defined geometry"),
                "topology": geometry.get("topology"),
                "fields": ", ".join(fields),
                "time samples": len(coordinates.get("time", ())) or None,
            },
        )


def _with_adapter_defaults(
    dataset: DatasetInput,
    format_id: str,
    manifest: Mapping[str, object],
) -> DatasetInput:
    """Resolve manifest-aware split defaults shown in the product form."""

    settings = dict(dataset.settings)
    completed = format_id == "pde_completed_case_dataset_v1"
    splits = _mapping(manifest.get("splits"))
    if completed:
        split_defaults = {
            "generator_training_split": (
                "generator_training" if "generator_training" in splits else "train"
            ),
            "generator_validation_split": (
                "generator_validation"
                if "generator_validation" in splits
                else "validation"
            ),
            "development_split": (
                "development" if "development" in splits else "train"
            ),
            "hidden_split": "verifier" if "verifier" in splits else "validation",
            "final_test_split": (
                "final_test" if "final_test" in splits else "validation"
            ),
        }
        for name, default in split_defaults.items():
            if settings.get(name) in (None, "", "auto"):
                settings[name] = default
        for name in (
            "generator_training_cases",
            "generator_validation_cases",
            "development_cases",
            "hidden_cases",
            "final_test_cases",
        ):
            settings.setdefault(name, 0)
    else:
        if settings.get("development_split") in (None, "", "auto"):
            settings["development_split"] = "validation"
        if settings.get("hidden_split") in (None, "", "auto"):
            settings["hidden_split"] = "d0_distribution_test"
        settings.setdefault("development_cases", 16)
        settings.setdefault("hidden_cases", 8)
    return DatasetInput(
        dataset_id=dataset.dataset_id,
        kind=dataset.kind,
        location=dataset.location,
        problem=dataset.problem,
        geometry=dataset.geometry,
        settings=settings,
    )


def _case_fields(case: Case) -> tuple[str, ...]:
    observations = _mapping(case.public_data.get("observations"))
    fields = observations.get("fields")
    if isinstance(fields, (list, tuple)):
        return tuple(str(value) for value in fields)
    arrays = _mapping(case.public_data.get("arrays"))
    return (
        ("velocity_x", "velocity_y", "pressure")
        if arrays
        else ("velocity_x", "velocity_y")
    )


def _read_object(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}
