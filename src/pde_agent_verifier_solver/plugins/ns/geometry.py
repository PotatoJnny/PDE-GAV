"""Dataset-owned geometry descriptions for Navier--Stokes cases.

The NS product does not choose a cylinder, channel, airfoil, or any other
shape.  A dataset declares how its spatial samples are arranged and supplies
the corresponding coordinates.  Dataset adapters use this small object to
give the Problem, Generator, Task, and panel the same description.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence


_SUPPORTED_TOPOLOGIES = {"regular_grid", "point_cloud", "unstructured_mesh"}


@dataclass(frozen=True)
class NSGeometry:
    """Geometry metadata shared by every Case in one prepared dataset.

    ``topology`` describes storage, not the physical shape.  For example, a
    circular-cylinder PIV plane may be stored as a point cloud, while a
    different cylinder dataset may use a Cartesian grid and mask.  ``details``
    stays open so a dataset can publish boundary labels, obstacle parameters,
    mesh regions, or laboratory metadata without changing the shared engine.
    """

    label: str
    topology: str
    spatial_dimension: int
    coordinate_names: tuple[str, ...]
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("NS geometry needs a human-readable label")
        if self.topology not in _SUPPORTED_TOPOLOGIES:
            choices = ", ".join(sorted(_SUPPORTED_TOPOLOGIES))
            raise ValueError(f"unsupported NS geometry topology; choose one of: {choices}")
        if self.spatial_dimension < 1:
            raise ValueError("NS geometry spatial_dimension must be positive")
        if len(self.coordinate_names) != self.spatial_dimension:
            raise ValueError(
                "NS geometry coordinate_names must match spatial_dimension"
            )
        if len(set(self.coordinate_names)) != len(self.coordinate_names):
            raise ValueError("NS geometry coordinate names must be unique")

    def public_description(self) -> dict[str, object]:
        """Return the JSON-friendly description placed in public Cases."""

        return {
            "label": self.label,
            "topology": self.topology,
            "spatial_dimension": self.spatial_dimension,
            "coordinate_names": list(self.coordinate_names),
            **dict(self.details),
        }


def geometry_from_dataset(
    dataset_geometry: Mapping[str, object],
    manifest_geometry: Mapping[str, object],
    *,
    default_topology: str,
    coordinate_names: Sequence[str],
) -> NSGeometry:
    """Merge customer metadata with the storage facts in a manifest.

    Storage facts from the completed dataset take precedence.  Descriptive
    customer fields are retained, so selecting another dataset changes the
    physical geometry without requiring an NS code branch.
    """

    merged = {**dict(dataset_geometry), **dict(manifest_geometry)}
    label = str(
        merged.pop("label", "")
        or merged.get("kind", "")
        or merged.get("geometry_id", "")
        or "dataset-defined geometry"
    )
    topology = str(merged.pop("topology", default_topology))
    raw_names = merged.pop("coordinate_names", coordinate_names)
    if not isinstance(raw_names, (list, tuple)) or not all(
        isinstance(value, str) and value for value in raw_names
    ):
        raise ValueError("geometry.coordinate_names must be a list of names")
    names = tuple(raw_names)
    dimension = int(merged.pop("spatial_dimension", len(names)))
    return NSGeometry(
        label=label,
        topology=topology,
        spatial_dimension=dimension,
        coordinate_names=names,
        details=merged,
    )
