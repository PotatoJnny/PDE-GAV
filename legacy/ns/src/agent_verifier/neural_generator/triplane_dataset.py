from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

Array = np.ndarray

AUTOENCODER_PUBLIC_FIELDS = ("x", "y", "outer_boundary_mask", "obstacle_signed_distance")
AUTOENCODER_HIDDEN_FIELDS = ("u_true", "pressure")

KITE_VERTICES: tuple[tuple[float, float], ...] = (
    (0.315, 0.535),
    (0.438, 0.690),
    (0.565, 0.515),
    (0.405, 0.365),
)

BAFFLE_VERTICES: tuple[tuple[float, float], ...] = (
    (0.660, 0.350),
    (0.790, 0.430),
    (0.690, 0.535),
)


@dataclass(frozen=True)
class KiteBaffleGeometryConfig:
    nx: int = 64
    ny: int = 64
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0
    grid_sample: str = "cell_centered"
    outer_boundary: str = "no_slip_walls"
    obstacle_components: tuple[str, ...] = ("offset_kite", "downstream_triangular_baffle")

    @property
    def domain(self) -> tuple[float, float]:
        return float(self.x_max - self.x_min), float(self.y_max - self.y_min)

    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / float(self.nx)

    @property
    def dy(self) -> float:
        return (self.y_max - self.y_min) / float(self.ny)

    def validate(self) -> None:
        if self.nx < 8 or self.ny < 8:
            raise ValueError("kite+baffle geometry needs nx and ny >= 8")
        if self.grid_sample != "cell_centered":
            raise ValueError("the approved triplane geometry uses cell_centered sampling")
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("invalid domain bounds")


@dataclass(frozen=True)
class GeometryContract:
    config: KiteBaffleGeometryConfig
    x: Array
    y: Array
    fluid_mask: Array
    obstacle_mask: Array
    outer_boundary_mask: Array
    obstacle_signed_distance: Array
    brinkman_indicator: Array
    contract_hash: str

    def public_arrays(self) -> dict[str, Array]:
        return {
            "x": np.asarray(self.x, dtype=np.float64),
            "y": np.asarray(self.y, dtype=np.float64),
            "fluid_mask": np.asarray(self.fluid_mask, dtype=bool),
            "obstacle_mask": np.asarray(self.obstacle_mask, dtype=bool),
            "outer_boundary_mask": np.asarray(self.outer_boundary_mask, dtype=bool),
            "obstacle_signed_distance": np.asarray(self.obstacle_signed_distance, dtype=np.float32),
            "brinkman_indicator": np.asarray(self.brinkman_indicator, dtype=np.float32),
        }

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "geometry_id": "unit_square_offset_kite_downstream_triangular_baffle_v1",
            "contract_hash": self.contract_hash,
            "config": asdict(self.config),
            "vertices": {
                "offset_kite": [list(v) for v in KITE_VERTICES],
                "downstream_triangular_baffle": [list(v) for v in BAFFLE_VERTICES],
            },
            "array_shapes": {name: list(value.shape) for name, value in self.public_arrays().items()},
            "hash_note": "sha256 over canonical geometry metadata and geometry arrays",
        }


@dataclass(frozen=True)
class TriplaneCase:
    case_id: str
    case_dir: Path
    public_question: dict[str, Any]
    public_arrays: dict[str, Array]
    hidden_arrays: dict[str, Array]
    diagnostics: dict[str, Any]
    case_spec: dict[str, Any]


def build_kite_baffle_geometry(config: KiteBaffleGeometryConfig | None = None) -> GeometryContract:
    cfg = config or KiteBaffleGeometryConfig()
    cfg.validate()
    x = cfg.x_min + (np.arange(cfg.nx, dtype=np.float64) + 0.5) * cfg.dx
    y = cfg.y_min + (np.arange(cfg.ny, dtype=np.float64) + 0.5) * cfg.dy
    xx, yy = np.meshgrid(x, y, indexing="ij")
    kite_sdf = _polygon_signed_distance(xx, yy, np.asarray(KITE_VERTICES, dtype=np.float64))
    baffle_sdf = _polygon_signed_distance(xx, yy, np.asarray(BAFFLE_VERTICES, dtype=np.float64))
    obstacle_signed_distance = np.minimum(kite_sdf, baffle_sdf)
    obstacle_mask = obstacle_signed_distance <= 0.0
    outer_boundary_mask = np.zeros_like(obstacle_mask, dtype=bool)
    outer_boundary_mask[0, :] = True
    outer_boundary_mask[-1, :] = True
    outer_boundary_mask[:, 0] = True
    outer_boundary_mask[:, -1] = True
    fluid_mask = ~obstacle_mask
    brinkman_indicator = obstacle_mask.astype(np.float32)
    payload = {
        "config": asdict(cfg),
        "vertices": {
            "offset_kite": [list(v) for v in KITE_VERTICES],
            "downstream_triangular_baffle": [list(v) for v in BAFFLE_VERTICES],
        },
        "arrays": {
            "x": x,
            "y": y,
            "fluid_mask": fluid_mask,
            "obstacle_mask": obstacle_mask,
            "outer_boundary_mask": outer_boundary_mask,
            "obstacle_signed_distance": obstacle_signed_distance.astype(np.float32),
            "brinkman_indicator": brinkman_indicator,
        },
    }
    contract_hash = _hash_payload(payload)
    return GeometryContract(
        config=cfg,
        x=x,
        y=y,
        fluid_mask=fluid_mask,
        obstacle_mask=obstacle_mask,
        outer_boundary_mask=outer_boundary_mask,
        obstacle_signed_distance=obstacle_signed_distance.astype(np.float32),
        brinkman_indicator=brinkman_indicator,
        contract_hash=contract_hash,
    )


def write_geometry_contract(path: str | Path, geometry: GeometryContract) -> None:
    Path(path).write_text(json.dumps(geometry.to_json_dict(), indent=2, sort_keys=True) + "\n")


def write_case(
    case_dir: str | Path,
    *,
    case_id: str,
    geometry: GeometryContract,
    t: Array,
    velocity: Array,
    pressure: Array,
    forcing: Array,
    viscosity: float,
    case_spec: dict[str, Any],
    diagnostics: dict[str, Any] | None = None,
    precomputed_metrics: dict[str, float] | None = None,
    case_source: str = "kite_baffle_d0_large",
    task: str | None = None,
) -> dict[str, float]:
    path = Path(case_dir)
    path.mkdir(parents=True, exist_ok=True)
    vel = np.asarray(velocity, dtype=np.float32)
    prs = np.asarray(pressure, dtype=np.float32)
    frc = np.asarray(forcing, dtype=np.float32)
    times = np.asarray(t, dtype=np.float64)
    if vel.ndim != 4 or vel.shape[1] != 2:
        raise ValueError(f"velocity must have shape [T,2,X,Y], got {vel.shape}")
    if prs.shape != (vel.shape[0], vel.shape[2], vel.shape[3]):
        raise ValueError(f"pressure shape {prs.shape} does not match velocity {vel.shape}")
    if frc.shape != vel.shape:
        raise ValueError(f"forcing shape {frc.shape} does not match velocity {vel.shape}")
    if times.shape != (vel.shape[0],):
        raise ValueError(f"time shape {times.shape} does not match velocity {vel.shape}")
    if vel.shape[-2:] != geometry.fluid_mask.shape:
        raise ValueError(f"case grid {vel.shape[-2:]} does not match geometry {geometry.fluid_mask.shape}")

    diagnostic_arrays = _case_diagnostic_arrays(
        velocity=vel,
        pressure=prs,
        t=times,
        geometry=geometry,
    )
    metric_payload = (
        {str(name): float(value) for name, value in precomputed_metrics.items()}
        if precomputed_metrics is not None
        else case_metrics(
            velocity=vel,
            pressure=prs,
            forcing=frc,
            t=times,
            geometry=geometry,
            diagnostic_arrays=diagnostic_arrays,
        )
    )
    diag = {**metric_payload, **dict(diagnostics or {})}
    public_arrays = {
        **geometry.public_arrays(),
        "initial_velocity": vel[0],
        "forcing": frc,
        "t": times,
    }
    hidden_arrays = {
        "u_true": vel,
        "pressure": prs,
        "t": times,
        "time_derivative": diagnostic_arrays["time_derivative"].astype(np.float32),
        "vorticity": diagnostic_arrays["vorticity"].astype(np.float32),
        "velocity_laplacian": diagnostic_arrays["velocity_laplacian"].astype(np.float32),
        "pressure_gradient": diagnostic_arrays["pressure_gradient"].astype(np.float32),
        "fluid_interior_mask": fluid_interior_mask(geometry).astype(bool),
        "boundary_layer_mask": boundary_layer_mask(geometry).astype(bool),
    }
    public_question = {
        "case_id": case_id,
        "task": task
        or "solve fixed-geometry 2D incompressible flow on the kite+baffle domain from initial velocity and public forcing",
        "case_source": case_source,
        "target_shape": list(vel.shape),
        "geometry": {
            "geometry_id": geometry.to_json_dict()["geometry_id"],
            "contract_hash": geometry.contract_hash,
            "domain": [geometry.config.x_min, geometry.config.x_max, geometry.config.y_min, geometry.config.y_max],
            "grid_sample": geometry.config.grid_sample,
            "outer_boundary": geometry.config.outer_boundary,
        },
        "time": {
            "t0": float(times[0]),
            "t1": float(times[-1]),
            "nt": int(times.size),
            "dt_median": median_spacing(times),
            "values": [float(v) for v in times],
        },
        "viscosity": float(viscosity),
        "forcing_derivation": diag.get("public_forcing_derivation", "provided_public_arrays_forcing"),
        "arrays": {
            "x": "public_arrays.npz:x",
            "y": "public_arrays.npz:y",
            "t": "public_arrays.npz:t",
            "initial_velocity": "public_arrays.npz:initial_velocity",
            "forcing": "public_arrays.npz:forcing",
            "fluid_mask": "public_arrays.npz:fluid_mask",
            "obstacle_mask": "public_arrays.npz:obstacle_mask",
            "outer_boundary_mask": "public_arrays.npz:outer_boundary_mask",
            "obstacle_signed_distance": "public_arrays.npz:obstacle_signed_distance",
            "brinkman_indicator": "public_arrays.npz:brinkman_indicator",
        },
        "solver_contract": "return {'prediction': array} or {'u': array} with target_shape [T,2,X,Y]",
    }
    hidden_manifest = {
        "case_id": case_id,
        "truth_type": case_source,
        "geometry_contract_hash": geometry.contract_hash,
        "hidden_arrays": {
            "u_true": "hidden_truth_cache.npz:u_true",
            "pressure": "hidden_truth_cache.npz:pressure",
            "time_derivative": "hidden_truth_cache.npz:time_derivative",
            "vorticity": "hidden_truth_cache.npz:vorticity",
            "velocity_laplacian": "hidden_truth_cache.npz:velocity_laplacian",
            "pressure_gradient": "hidden_truth_cache.npz:pressure_gradient",
        },
    }
    np.savez_compressed(path / "public_arrays.npz", **public_arrays)
    np.savez_compressed(path / "hidden_truth_cache.npz", **hidden_arrays)
    (path / "public_question.json").write_text(json.dumps(public_question, indent=2, sort_keys=True) + "\n")
    (path / "hidden_truth_manifest.json").write_text(json.dumps(hidden_manifest, indent=2, sort_keys=True) + "\n")
    (path / "diagnostics.json").write_text(json.dumps(diag, indent=2, sort_keys=True) + "\n")
    (path / "case_spec.json").write_text(json.dumps(case_spec, indent=2, sort_keys=True) + "\n")
    return metric_payload


def load_case(
    case_dir: str | Path,
    *,
    public_fields: Iterable[str] | None = None,
    hidden_fields: Iterable[str] | None = None,
    load_metadata: bool = True,
) -> TriplaneCase:
    path = Path(case_dir)
    with np.load(path / "public_arrays.npz") as public_npz:
        public_names = tuple(public_npz.files) if public_fields is None else tuple(public_fields)
        missing_public = sorted(set(public_names) - set(public_npz.files))
        if missing_public:
            raise KeyError(f"missing public arrays in {path}: {missing_public}")
        public_arrays = {name: np.asarray(public_npz[name]) for name in public_names}
    with np.load(path / "hidden_truth_cache.npz") as hidden_npz:
        hidden_names = tuple(hidden_npz.files) if hidden_fields is None else tuple(hidden_fields)
        missing_hidden = sorted(set(hidden_names) - set(hidden_npz.files))
        if missing_hidden:
            raise KeyError(f"missing hidden arrays in {path}: {missing_hidden}")
        hidden_arrays = {name: np.asarray(hidden_npz[name]) for name in hidden_names}
    return TriplaneCase(
        case_id=path.name,
        case_dir=path,
        public_question=json.loads((path / "public_question.json").read_text()) if load_metadata else {},
        public_arrays=public_arrays,
        hidden_arrays=hidden_arrays,
        diagnostics=(
            json.loads((path / "diagnostics.json").read_text())
            if load_metadata and (path / "diagnostics.json").exists()
            else {}
        ),
        case_spec=(
            json.loads((path / "case_spec.json").read_text())
            if load_metadata and (path / "case_spec.json").exists()
            else {}
        ),
    )


def load_autoencoder_case(case_dir: str | Path) -> TriplaneCase:
    return load_case(
        case_dir,
        public_fields=AUTOENCODER_PUBLIC_FIELDS,
        hidden_fields=AUTOENCODER_HIDDEN_FIELDS,
        load_metadata=False,
    )


def iter_case_dirs(root: str | Path) -> list[Path]:
    path = Path(root)
    if not path.exists():
        return []
    case_dirs = {p.parent for p in path.rglob("public_question.json") if p.parent.is_dir()}
    for child in path.iterdir():
        if child.is_dir() and (child / "public_question.json").exists():
            case_dirs.add(child)
    return sorted(case_dirs)


def write_split_manifest(
    path: str | Path,
    case_dirs: Iterable[str | Path],
    *,
    train: int,
    validation: int,
    d0_test: int,
    adversarial_guard: int,
    seed: int,
) -> dict[str, Any]:
    dirs = [Path(p) for p in case_dirs]
    required = int(train) + int(validation) + int(d0_test) + int(adversarial_guard)
    if len(dirs) < required:
        raise ValueError(f"need at least {required} cases for requested split, got {len(dirs)}")
    rng = np.random.default_rng(int(seed))
    order = np.arange(len(dirs))
    rng.shuffle(order)
    chosen = [dirs[int(i)] for i in order[:required]]
    splits = {
        "train": chosen[:train],
        "validation": chosen[train : train + validation],
        "d0_distribution_test": chosen[train + validation : train + validation + d0_test],
        "adversarial_guard": chosen[train + validation + d0_test : required],
    }
    manifest = {
        "seed": int(seed),
        "required_cases": int(required),
        "split_sizes": {name: len(values) for name, values in splits.items()},
        "splits": {
            name: [{"case_id": p.name, "case_dir": str(p)} for p in values] for name, values in splits.items()
        },
    }
    Path(path).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def summarize_dataset(case_dirs: Iterable[str | Path], out_dir: str | Path) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    family_counts: dict[str, int] = {}
    failure_counts: dict[str, int] = {}
    for case_path in case_dirs:
        path = Path(case_path)
        diagnostics = json.loads((path / "diagnostics.json").read_text())
        spec = json.loads((path / "case_spec.json").read_text())
        family = str(spec.get("family", "unknown"))
        family_counts[family] = family_counts.get(family, 0) + 1
        failures = diagnostics.get("filter_failures", [])
        for failure in failures:
            failure_counts[str(failure)] = failure_counts.get(str(failure), 0) + 1
        rows.append({"case_id": path.name, "family": family, **diagnostics})
    metric_names = sorted({k for row in rows for k, v in row.items() if isinstance(v, (int, float)) and np.isfinite(v)})
    metric_summary = {}
    for name in metric_names:
        values = np.asarray([float(row[name]) for row in rows if isinstance(row.get(name), (int, float))], dtype=float)
        if values.size:
            metric_summary[name] = {
                "count": int(values.size),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "q01": float(np.quantile(values, 0.01)),
                "q05": float(np.quantile(values, 0.05)),
                "q50": float(np.quantile(values, 0.50)),
                "q95": float(np.quantile(values, 0.95)),
                "q99": float(np.quantile(values, 0.99)),
            }
    summary = {
        "n_cases": len(rows),
        "family_counts": family_counts,
        "filter_failure_counts": failure_counts,
        "metric_summary": metric_summary,
    }
    (out / "dataset_release_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (out / "family_statistics.json").write_text(json.dumps(family_counts, indent=2, sort_keys=True) + "\n")
    (out / "filter_failure_report.json").write_text(json.dumps(failure_counts, indent=2, sort_keys=True) + "\n")
    with (out / "outlier_table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "family", *metric_names], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (out / "dataset_release_summary.md").write_text(_summary_markdown(summary))
    return summary


def case_metrics(
    *,
    velocity: Array,
    pressure: Array,
    forcing: Array,
    t: Array,
    geometry: GeometryContract,
    diagnostic_arrays: dict[str, Array] | None = None,
) -> dict[str, float]:
    vel = np.asarray(velocity, dtype=np.float64)
    prs = np.asarray(pressure, dtype=np.float64)
    frc = np.asarray(forcing, dtype=np.float64)
    derived = diagnostic_arrays or _case_diagnostic_arrays(
        velocity=vel,
        pressure=prs,
        t=t,
        geometry=geometry,
    )
    td = derived["time_derivative"]
    vort = derived["vorticity"]
    lap = derived["velocity_laplacian"]
    gradp = derived["pressure_gradient"]
    div = derived["divergence"]
    fluid = np.asarray(geometry.fluid_mask, dtype=bool)
    interior = fluid_interior_mask(geometry) & ~boundary_layer_mask(geometry)
    if not np.any(interior):
        interior = fluid
    obstacle = np.asarray(geometry.obstacle_mask, dtype=bool)
    outer = np.asarray(geometry.outer_boundary_mask, dtype=bool)
    velocity_scale = rms(vel[:, :, fluid]) + 1.0e-12
    divergence_scale = velocity_scale / max(min(geometry.config.dx, geometry.config.dy), 1.0e-12)
    forcing_scale = rms(frc[:, :, fluid]) + 1.0e-12
    return {
        "velocity_rms": rms(vel[:, :, fluid]),
        "forcing_rms": rms(frc[:, :, fluid]),
        "time_derivative_rms": rms(td[:, :, fluid]),
        "vorticity_rms": rms(vort[:, fluid]),
        "velocity_laplacian_rms": rms(lap[:, :, fluid]),
        "pressure_gradient_rms": rms(gradp[:, :, fluid]),
        "pressure_rms": rms(prs[:, fluid]),
        "repeat_initial_difficulty": repeat_initial_difficulty(vel, fluid_mask=fluid),
        "divergence_rel": rms(div[:, interior]) / (divergence_scale + 1.0e-12),
        "outer_boundary_velocity_rel": rms(vel[:, :, outer]) / velocity_scale,
        "obstacle_velocity_rel": rms(vel[:, :, obstacle]) / velocity_scale if np.any(obstacle) else 0.0,
        "forcing_inside_obstacle_rel": rms(frc[:, :, obstacle]) / forcing_scale if np.any(obstacle) else 0.0,
    }


def _case_diagnostic_arrays(
    *,
    velocity: Array,
    pressure: Array,
    t: Array,
    geometry: GeometryContract,
) -> dict[str, Array]:
    """Compute finite-difference fields once for metrics and stored case data."""

    vel = np.asarray(velocity, dtype=np.float64)
    prs = np.asarray(pressure, dtype=np.float64)
    times = np.asarray(t, dtype=np.float64)
    return {
        "time_derivative": time_derivative(vel, times),
        "vorticity": vorticity(vel, geometry.x, geometry.y),
        "velocity_laplacian": velocity_laplacian(vel, geometry.x, geometry.y),
        "pressure_gradient": pressure_gradient(prs, geometry.x, geometry.y),
        "divergence": divergence(vel, geometry.x, geometry.y),
    }


def mms_forcing_from_velocity_pressure(*, velocity: Array, pressure: Array, t: Array, viscosity: float, geometry: GeometryContract) -> Array:
    vel = np.asarray(velocity, dtype=np.float64)
    prs = np.asarray(pressure, dtype=np.float64)
    td = time_derivative(vel, t)
    lap = velocity_laplacian(vel, geometry.x, geometry.y)
    gradp = pressure_gradient(prs, geometry.x, geometry.y)
    du_dx = grad(vel[:, 0], geometry.x, axis=1)
    du_dy = grad(vel[:, 0], geometry.y, axis=2)
    dv_dx = grad(vel[:, 1], geometry.x, axis=1)
    dv_dy = grad(vel[:, 1], geometry.y, axis=2)
    adv = np.empty_like(vel, dtype=np.float64)
    adv[:, 0] = vel[:, 0] * du_dx + vel[:, 1] * du_dy
    adv[:, 1] = vel[:, 0] * dv_dx + vel[:, 1] * dv_dy
    forcing = td + adv + gradp - float(viscosity) * lap
    solid = ~np.asarray(geometry.fluid_mask, dtype=bool)
    outer = np.asarray(geometry.outer_boundary_mask, dtype=bool)
    forcing[:, :, solid | outer] = 0.0
    return forcing.astype(np.float32)


def time_derivative(velocity: Array, t: Array) -> Array:
    return np.gradient(np.asarray(velocity, dtype=np.float64), np.asarray(t, dtype=np.float64), axis=0, edge_order=2)


def vorticity(velocity: Array, x: Array, y: Array) -> Array:
    vel = np.asarray(velocity, dtype=np.float64)
    return grad(vel[:, 1], x, axis=1) - grad(vel[:, 0], y, axis=2)


def velocity_laplacian(velocity: Array, x: Array, y: Array) -> Array:
    vel = np.asarray(velocity, dtype=np.float64)
    return np.stack([laplacian(vel[:, component], x, y) for component in range(2)], axis=1)


def pressure_gradient(pressure: Array, x: Array, y: Array) -> Array:
    prs = np.asarray(pressure, dtype=np.float64)
    return np.stack([grad(prs, x, axis=1), grad(prs, y, axis=2)], axis=1)


def divergence(velocity: Array, x: Array, y: Array) -> Array:
    vel = np.asarray(velocity, dtype=np.float64)
    return grad(vel[:, 0], x, axis=1) + grad(vel[:, 1], y, axis=2)


def laplacian(field: Array, x: Array, y: Array) -> Array:
    return grad(grad(field, x, axis=1), x, axis=1) + grad(grad(field, y, axis=2), y, axis=2)


def grad(field: Array, coords: Array, axis: int) -> Array:
    arr = np.asarray(field, dtype=np.float64)
    edge_order = 2 if arr.shape[axis] >= 3 else 1
    return np.gradient(arr, np.asarray(coords, dtype=np.float64), axis=axis, edge_order=edge_order)


def repeat_initial_difficulty(velocity: Array, *, fluid_mask: Array | None = None) -> float:
    vel = np.asarray(velocity, dtype=np.float64)
    pred = np.repeat(vel[:1], vel.shape[0], axis=0)
    if fluid_mask is not None:
        mask = np.asarray(fluid_mask, dtype=bool)
        vel = vel[:, :, mask]
        pred = pred[:, :, mask]
    return float(np.linalg.norm(pred - vel) / (np.linalg.norm(vel) + 1.0e-12))


def fluid_interior_mask(geometry: GeometryContract) -> Array:
    fluid = np.asarray(geometry.fluid_mask, dtype=bool)
    outer = np.asarray(geometry.outer_boundary_mask, dtype=bool)
    return fluid & ~outer


def boundary_layer_mask(geometry: GeometryContract) -> Array:
    sdf = np.asarray(geometry.obstacle_signed_distance, dtype=np.float64)
    near_obstacle = (sdf > 0.0) & (sdf <= 2.5 * max(geometry.config.dx, geometry.config.dy))
    return near_obstacle | np.asarray(geometry.outer_boundary_mask, dtype=bool)


def median_spacing(axis: Array) -> float:
    arr = np.asarray(axis, dtype=np.float64).reshape(-1)
    return 0.0 if arr.size <= 1 else float(np.median(np.diff(arr)))


def rms(arr: Array) -> float:
    vals = np.asarray(arr, dtype=np.float64)
    if vals.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(vals**2)))


def _polygon_signed_distance(xx: Array, yy: Array, vertices: Array) -> Array:
    inside = _points_in_polygon(xx, yy, vertices)
    distance = _distance_to_polygon_edges(xx, yy, vertices)
    return np.where(inside, -distance, distance)


def _points_in_polygon(xx: Array, yy: Array, vertices: Array) -> Array:
    x = np.asarray(xx, dtype=np.float64)
    y = np.asarray(yy, dtype=np.float64)
    inside = np.zeros_like(x, dtype=bool)
    vx = vertices[:, 0]
    vy = vertices[:, 1]
    j = len(vertices) - 1
    for i in range(len(vertices)):
        yi = vy[i]
        yj = vy[j]
        xi = vx[i]
        xj = vx[j]
        intersects = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1.0e-30) + xi)
        inside ^= intersects
        j = i
    return inside


def _distance_to_polygon_edges(xx: Array, yy: Array, vertices: Array) -> Array:
    points_x = np.asarray(xx, dtype=np.float64)
    points_y = np.asarray(yy, dtype=np.float64)
    out = np.full(points_x.shape, np.inf, dtype=np.float64)
    for i in range(len(vertices)):
        ax, ay = vertices[i]
        bx, by = vertices[(i + 1) % len(vertices)]
        abx = bx - ax
        aby = by - ay
        denom = abx * abx + aby * aby
        tau = ((points_x - ax) * abx + (points_y - ay) * aby) / max(denom, 1.0e-30)
        tau = np.clip(tau, 0.0, 1.0)
        closest_x = ax + tau * abx
        closest_y = ay + tau * aby
        out = np.minimum(out, np.sqrt((points_x - closest_x) ** 2 + (points_y - closest_y) ** 2))
    return out


def _hash_payload(payload: dict[str, Any]) -> str:
    digest = hashlib.sha256()

    def update(value: Any) -> None:
        if isinstance(value, dict):
            for key in sorted(value):
                digest.update(str(key).encode("utf-8"))
                update(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value:
                update(item)
        elif isinstance(value, np.ndarray):
            arr = np.ascontiguousarray(value)
            digest.update(str(arr.dtype).encode("utf-8"))
            digest.update(str(arr.shape).encode("utf-8"))
            digest.update(arr.tobytes())
        else:
            digest.update(repr(value).encode("utf-8"))

    update(payload)
    return digest.hexdigest()


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Kite+Baffle D0 Dataset Release Summary",
        "",
        f"- cases: `{summary['n_cases']}`",
        "",
        "## Families",
        "",
    ]
    for name, count in sorted(summary.get("family_counts", {}).items()):
        lines.append(f"- `{name}`: `{count}`")
    lines.extend(["", "## Key Metrics", ""])
    for name in (
        "velocity_rms",
        "forcing_rms",
        "time_derivative_rms",
        "vorticity_rms",
        "velocity_laplacian_rms",
        "pressure_gradient_rms",
        "repeat_initial_difficulty",
    ):
        stats = summary.get("metric_summary", {}).get(name)
        if stats:
            lines.append(f"- `{name}`: mean `{stats['mean']:.6g}`, q05 `{stats['q05']:.6g}`, q95 `{stats['q95']:.6g}`")
    return "\n".join(lines) + "\n"
