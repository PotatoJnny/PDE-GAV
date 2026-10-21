"""Project-owned sparse PIV reconstruction for the experimental NS solver task.

Only this trusted preparation layer reads measured arrays. Candidate Cases are
manufactured by the existing NS source and contain no observations or POD asset.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from pde_agent_verifier_solver.sdk import (
    DatasetInput, DatasetKind, CaseSource, CaseVisibility, FormField, PanelCard,
    PreparedDataset, PreparedReconstructionSource,
    ProgramInfo, ReconstructionEvaluation, ReconstructionInput, ReconstructionOutput,
)
from .problem import NSProblem
from .reconstruction.real_piv import (
    GappyPODPosterior, SparsePIVLayout, SparsePIVObservations, fit_trajectory_pod,
)

MMS_DATASET_SCHEMA = "ns_reconstructed_structured_dataset_v2"
_PREPARED_CASE_SCHEMA = "ns_experimental_frozen_original_cases_v1"
ROLES = ("generator_training", "generator_validation", "development", "verifier", "final_test")


def _object(value, label):
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _positive(value, label):
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and positive")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite and positive") from exc
    if not np.isfinite(number) or number <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return number


def _path(value, base):
    if not isinstance(value, str) or not value:
        raise ValueError("PIV file reference must be nonempty")
    p = Path(value).expanduser()
    return (p if p.is_absolute() else base / p).resolve(strict=True)


def _window_clock(manifest):
    declarations = [section["sampling_hz"] for key in ("source_dataset", "time")
                    if isinstance((section := manifest.get(key)), Mapping)
                    and section.get("sampling_hz") is not None]
    rates = [_positive(value, "declared PIV sampling_hz") for value in declarations]
    if rates and any(not np.isclose(rate, rates[0], rtol=1e-12, atol=0) for rate in rates[1:]):
        raise ValueError("PIV manifest sampling_hz declarations disagree")
    counts = [section[key] for name, key in (("time", "samples_per_case"),
                                             ("split_protocol", "window_length"))
              if isinstance((section := manifest.get(name)), Mapping) and key in section]
    if any(isinstance(count, bool) or not isinstance(count, int) or count < 2 for count in counts):
        raise ValueError("PIV window length must be an integer of at least two")
    if counts and any(count != counts[0] for count in counts[1:]):
        raise ValueError("PIV manifest window lengths disagree")
    return {"sampling_hz": rates[0] if rates else None,
            "samples_per_case": counts[0] if counts else None,
            "time": None, "reference_roundoff": 0.0, "max_adjustment": 0.0}


def _time_roundoff(raw, local, source_frames, sampling_hz):
    """Bound timestamp subtraction error, including legacy float32 provenance.

    Completed PIV files store float64 window-local times even when the acquisition
    clock was float32. Such differences retain a binary lattice; ordinary float64
    timestamps must not inherit the much wider float32 tolerance.
    """
    period = 1.0 / sampling_hz if sampling_hz else float(np.median(np.diff(local)))
    scale = max(float(np.max(np.abs(raw))), float(local[-1]))
    if source_frames is not None:
        # The legacy writer subtracts the window origin before saving. Its source
        # frame range preserves the magnitude needed to bound that cancellation.
        scale = max(scale, (source_frames[1] + 1) * period)
    bound = 32 * np.finfo(np.float64).eps * max(1.0, scale)
    quantum = float(np.spacing(np.float32(np.min(np.diff(local))))) / 8
    quantized = False
    if np.isfinite(quantum) and quantum > 0:
        ticks = local / quantum
        quantized = bool(np.all(np.abs(ticks - np.rint(ticks)) <=
                                32 * np.finfo(np.float64).eps * np.maximum(1, np.abs(ticks))))
    if raw.dtype == np.dtype("float32") or quantized:
        # Two rounded absolute timestamps were subtracted. The legacy loader may
        # also infer the declared period from rounded timestamp differences.
        bound += 2 * float(np.spacing(np.float32(scale)))
        if sampling_hz is not None:
            bound += 16 * np.finfo(np.float32).eps * float(local[-1])
    return bound


def _coordinates(arrays, *, clock, source_frames):
    raw_time = np.asarray(arrays["time"])
    x, y, t = (np.array(arrays[k], dtype=np.float64, copy=True) for k in ("x", "y", "time"))
    if (x.ndim != 1 or x.shape != y.shape or len(x) < 3 or t.ndim != 1 or len(t) < 2
            or not all(np.isfinite(a).all() for a in (x, y, t)) or not np.all(np.diff(t) > 0)):
        raise ValueError("PIV needs finite matching x/y points and increasing time")
    if source_frames is not None and source_frames[1] - source_frames[0] != len(t):
        raise ValueError("PIV source frame range differs from window length")
    if clock["samples_per_case"] is not None and len(t) != clock["samples_per_case"]:
        raise ValueError("one MMS Project needs a common PIV window length")
    local = t - t[0]
    roundoff = _time_roundoff(raw_time, local, source_frames, clock["sampling_hz"])
    if clock["time"] is None:
        clock["time"] = (np.arange(len(t), dtype=np.float64) / clock["sampling_hz"]
                         if clock["sampling_hz"] is not None else local.copy())
        clock["reference_roundoff"] = roundoff if clock["sampling_hz"] is None else 0.0
    common = clock["time"]
    if local.shape != common.shape:
        raise ValueError("one MMS Project needs a common PIV window length")
    adjustment = float(np.max(np.abs(local - common)))
    if adjustment > roundoff + clock["reference_roundoff"]:
        raise ValueError("PIV window times differ beyond timestamp rounding; check sampling_hz and sampling intervals")
    clock["max_adjustment"] = max(clock["max_adjustment"], adjustment)
    return x, y, common.copy()


def _index(values, size, label):
    a = np.asarray(values)
    if (a.ndim != 1 or a.dtype.kind not in "iu" or not len(a)
            or np.any(a < 0) or np.any(a >= size) or len(np.unique(a)) != len(a)):
        raise ValueError(f"invalid {label}")
    return a.astype(np.int64, copy=True)


def prepare_piv_reconstruction_source(dataset, settings, seed):
    """Read calibration fields and public observations, sealing validation truth."""
    if dataset.kind is not DatasetKind.EXPERIMENTAL:
        raise ValueError("experimental MMS reconstruction requires experimental data")
    p = Path(dataset.location).expanduser().resolve(strict=True)
    manifest = _object(json.loads(p.read_text()), "completed PIV manifest")
    if manifest.get("schema") != "pde_completed_case_dataset_v1" or manifest.get("problem") != "ns":
        raise ValueError("experimental MMS expects a completed NS PIV manifest")
    options = {**dict(dataset.settings), **dict(settings)}
    allowed = {"viscosity", "coordinate_units", "mms_geometry", "noise_standard_deviation", "projection_config"}
    if set(options) - allowed:
        raise ValueError(f"unsupported experimental reconstruction settings: {sorted(set(options)-allowed)}")
    # A declared model parameter is required; do not infer viscosity from a filename/Re alone.
    viscosity = (_positive(options["viscosity"], "declared MMS viscosity")
                 if options.get("viscosity") is not None else None)
    declared_geometry = options.get("mms_geometry", dataset.geometry or manifest.get("mms_geometry"))
    if not isinstance(declared_geometry, Mapping) or "obstacle_polygons" not in declared_geometry:
        raise ValueError("declare mms_geometry with obstacle_polygons (empty only for a declared obstacle-free domain)")
    coordinate_units = options.get("coordinate_units")
    if coordinate_units not in ("physical", "nondimensional"):
        raise ValueError("declare coordinate_units as physical or nondimensional")
    splits = _object(manifest.get("splits"), "PIV splits")
    roles = (*ROLES, "reconstruction_validation")
    if len(splits.get("generator_training", ())) < 2:
        raise ValueError("Gappy POD needs at least two independent calibration windows")
    if not splits.get("generator_validation") or not splits.get("reconstruction_validation"):
        raise ValueError("separate generator and reconstruction validation splits are required")
    observation_layout = _object(manifest.get("observation_layout", {}), "observation layout")
    noise = np.asarray(options.get("noise_standard_deviation",
        observation_layout.get("noise_standard_deviation")), dtype=np.float64)
    if noise.shape != (2,) or not np.isfinite(noise).all() or np.any(noise <= 0):
        raise ValueError("declare two finite positive observation noise scales")
    coordinates = None
    clock = _window_clock(manifest)
    calibration = []
    targets = defaultdict(list)
    references = {}
    seen_ids, seen_paths, intervals = set(), set(), defaultdict(list)
    for role in roles:
        rows = splits.get(role, ())
        if not isinstance(rows, (list, tuple)):
            raise ValueError(f"split {role} must be a sequence")
        for index, raw in enumerate(rows):
            row = _object(raw, "PIV case")
            cid = row.get("case_id")
            if not isinstance(cid, str) or not cid or cid in seen_ids:
                raise ValueError("PIV split case IDs must be nonempty and disjoint")
            seen_ids.add(cid)
            file = _path(row.get("case_file"), p.parent)
            if file in seen_paths:
                raise ValueError("PIV splits reuse the same case file")
            seen_paths.add(file)
            recording, frames = row.get("source_recording"), row.get("source_frames")
            # Raw completed-PIV imports need provenance to verify scientific split
            # isolation. Frozen continuous datasets have a separate restore path.
            if not isinstance(recording, str) or not recording or frames is None:
                raise ValueError("each PIV case requires source_recording and source_frames to verify split isolation")
            if (not isinstance(frames, (list, tuple)) or len(frames) != 2
                    or any(isinstance(v, bool) or not isinstance(v, int) for v in frames)
                    or frames[0] < 0 or frames[1] <= frames[0]):
                raise ValueError("invalid PIV source recording/window")
            for old_role, start, stop in intervals[recording]:
                if role != old_role and frames[0] < stop and start < frames[1]:
                    raise ValueError("PIV source frames overlap across scientific splits")
            intervals[recording].append((role, *frames))
            with np.load(file, allow_pickle=False) as a:
                x, y, t = _coordinates(a, clock=clock, source_frames=frames)
                if coordinates is None:
                    coordinates = (x, y, t)
                if any(v.shape != expected.shape or not np.allclose(v, expected, rtol=1e-8, atol=1e-10)
                       for v, expected in zip((x, y), coordinates[:2])):
                    raise ValueError("one MMS Project needs common PIV spatial coordinates")
                sensors = _index(a["sensor_indices"], len(x), "sensor indices")
                frames_observed = _index(a["observed_frames"], len(t), "observed frames")
                values = np.array(a["public_values"], dtype=np.float64, copy=True)
                if not np.isfinite(values).all():
                    raise ValueError("PIV observations must be finite")
                observation = SparsePIVObservations(values, SparsePIVLayout(
                    sensors, frames_observed, len(t), len(x)), noise.copy())
                if role == "generator_training":
                    if row.get("generator_target_kind") != "measured_full_field":
                        raise ValueError("POD calibration must declare measured_full_field training targets")
                    velocity = np.array(a["velocity"], dtype=np.float64, copy=True)
                    if velocity.shape != (len(t), 2, len(x)) or not np.isfinite(velocity).all():
                        raise ValueError("invalid full calibration velocity")
                    calibration.append(velocity)
            alias = f"{role}-{index:05d}"
            targets[role].append({"case_id": cid, "alias": alias, "observations": observation,
                                  "source_recording": recording})
            if role == "reconstruction_validation":
                reference_file = _path(row.get("measurement_reference_file"), p.parent)
                with np.load(reference_file, allow_pickle=False) as a:
                    ref_coords = _coordinates(a, clock=clock, source_frames=frames)
                    if any(v.shape != expected.shape or not np.allclose(v, expected, rtol=1e-8, atol=1e-10)
                           for v, expected in zip(ref_coords[:2], coordinates[:2])):
                        raise ValueError("reconstruction reference coordinates differ")
                    truth = np.array(a["velocity"], dtype=np.float64, copy=True)
                if truth.shape != (len(t), 2, len(x)) or not np.isfinite(truth).all():
                    raise ValueError("invalid reconstruction validation reference")
                mask = ~observation.layout.public_mask()
                if not np.any(mask):
                    raise ValueError("reconstruction validation needs withheld observations")
                references[alias] = {"velocity": truth, "mask": mask}
    x, y, t = coordinates
    geometry = dict(_object(declared_geometry, "MMS geometry"))
    geometry["label"] = geometry.get("label", "PIV measurement region")
    program_input = ReconstructionInput(
        public_split_id="piv_to_continuous_ns",
        fit_data={"calibration_windows": np.stack(calibration), "x": x, "y": y, "time": t,
                  "viscosity": viscosity, "coordinate_units": coordinate_units,
                  "projection_config": dict(options.get("projection_config", {}))},
        target_observations=dict(targets), geometry=geometry,
        noise_model={"standard_deviation": noise.tolist()},
        metadata={"split_counts": {role: len(targets[role]) for role in roles},
                  "calibration_source": "generator_training_only", "time_origin": "window_local",
                  "frame_clock": {"policy": "declared_sampling_hz" if clock["sampling_hz"] else "first_window",
                                  "sampling_hz": clock["sampling_hz"],
                                  "maximum_timestamp_adjustment": clock["max_adjustment"]}},
    )
    return PreparedReconstructionSource(program_input, {"references": references, "coordinates": coordinates})


class ContinuousPIVReconstruction:
    """Compatibility name: Reconstruction now ends at complete discrete data."""
    info = ProgramInfo("ns-gappy-pod-completed-velocity", "PIV velocity reconstruction",
                       "Complete sparse observations on the measured velocity lattice.")

    def setup_fields(self):
        return (FormField("rank", "POD rank", "integer", 64),
                FormField("prior_precision", "POD prior precision", "number", 10.0),
                FormField("noise_scale", "Observation noise scale", "number", 1.0))

    def reconstruct(self, source, settings, seed, progress=None):
        allowed = {"rank", "prior_precision", "noise_scale"}
        if set(settings) - allowed:
            raise ValueError(f"unsupported reconstruction settings: {sorted(set(settings)-allowed)}; continuous fitting belongs to dataset projection_config")
        data = source.fit_data
        rank = settings.get("rank", 64)
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ValueError("POD rank must be positive")
        prior, _ = fit_trajectory_pod(data["calibration_windows"], max_rank=rank)
        precision = _positive(settings.get("prior_precision", 10.0), "prior precision")
        noise_scale = _positive(settings.get("noise_scale", 1.0), "noise scale")
        method = GappyPODPosterior(prior.rank, precision, noise_scale)
        flat = [(role, item) for role, items in source.target_observations.items() for item in items]
        groups = defaultdict(list)
        for role, item in flat:
            obs = item["observations"]
            key = (tuple(obs.layout.sensor_indices), tuple(obs.layout.observed_frames),
                   tuple(obs.noise_standard_deviation))
            groups[key].append((role, item))
        completed, validation = defaultdict(list), {}
        finished = 0
        for group in groups.values():
            estimates = method.reconstruct_many_observations(
                prior, tuple(row[1]["observations"] for row in group), include_uncertainty=False)
            for (role, item), estimate in zip(group, estimates):
                row = {"case_id": item["case_id"], "velocity": estimate.velocity.copy()}
                if item.get("source_recording") is not None:
                    row["source_recording"] = item["source_recording"]
                if role == "reconstruction_validation":
                    validation[item["alias"]] = row
                else:
                    completed[role].append(row)
                finished += 1
                if progress:
                    progress("Completing PIV velocity windows", finished / len(flat))
        return ReconstructionOutput(
            method_id=self.info.program_id,
            completed_data={"split_fields": dict(completed), "validation": validation,
                "x": data["x"].copy(), "y": data["y"].copy(), "time": data["time"].copy(),
                "geometry": dict(source.geometry), "projection_config": dict(data.get("projection_config", {})),
                "physical_settings": {"viscosity": data["viscosity"],
                    "coordinate_units": data["coordinate_units"], "pressure_model": "zero",
                    "time_origin": "window_local", "frame_clock": dict(source.metadata["frame_clock"])}},
            fields=("velocity_x", "velocity_y"),
            case_counts={role: len(completed[role]) for role in ROLES}, selected_rank=prior.rank,
            regularization=precision,
            public_settings={"rank": prior.rank, "prior_precision": precision, "noise_scale": noise_scale},
            diagnostics={"calibration_source": "generator_training_only", "output_kind": "complete_discrete_velocity"},
        )

    def result_card(self, report):
        return PanelCard(title="PIV velocity reconstruction", summary="Complete discrete velocity prepared",
                         values=dict(report.aggregate_metrics))


def project_reconstructed_output(output):
    """Dataset preparation, after Reconstruction, using the numerical projector."""
    from .completed_fields import project_completed_velocity, field_from_coefficients, numerical_symbols
    data = _object(output.completed_data, "completed velocity reconstruction")
    physical = dict(data["physical_settings"])
    if physical.get("pressure_model") != "zero":
        raise ValueError("completed PIV data requires explicit manufactured pressure_model='zero'")
    viscosity = _positive(physical.get("viscosity"), "declared MMS viscosity")
    module = numerical_symbols()
    cfg = module.StructuredNSProjectionConfig.from_dict(dict(data["projection_config"]))
    cfg.validate()
    converted, reports = {}, {}
    for role, rows in data["split_fields"].items():
        if not rows:
            converted[role] = []
            continue
        coefficients, report, geometry = project_completed_velocity(
            np.stack([row["velocity"] for row in rows]), data["x"], data["y"], data["time"],
            geometry=data["geometry"], projection_config=cfg.to_dict(), pressure_model="zero")
        converted[role] = [{**{key: row[key] for key in ("case_id", "source_recording") if key in row},
            "answer": field_from_coefficients(values, config=cfg.to_dict(), geometry=geometry,
                time_interval=(data["time"][0], data["time"][-1]), viscosity=viscosity)}
            for row, values in zip(rows, coefficients)]
        reports[role] = report
    physical["projection_config"] = cfg.to_dict()
    return converted, {"reconstruction_method": output.method_id, "physical_settings": physical,
                       "continuous_projection": {"implementation": "StructuredNSCoefficientProjector",
                                                 "reports": reports}}


class ExperimentalMMSProblem(NSProblem):
    info = ProgramInfo("ns-experimental-mms", "Experimental-data NS solver",
                       "Reconstruct PIV observations once, then solve manufactured NS initial-boundary problems.")

    def __init__(self, backend=None):
        super().__init__()
        self._mms_backend = backend

    def setup_fields(self):
        return (
            FormField("viscosity", "Declared kinematic viscosity", "number", None),
            FormField("coordinate_units", "Coordinate units", "select", "physical",
                      choices=(("physical", "Physical"), ("nondimensional", "Nondimensional"))),
            FormField("mms_geometry", "Declared measurement fluid domain", "json", None,
                      "Include obstacle_polygons; an empty list declares an obstacle-free domain."),
        )

    def prepare_reconstruction_source(self, dataset, settings, seed):
        return prepare_piv_reconstruction_source(dataset, settings, seed)

    def prepare_reconstructed_dataset(self, dataset, output):
        from .experimental_mms import prepare_experimental_mms_answers
        answers, summary = project_reconstructed_output(output)
        return prepare_experimental_mms_answers(dataset, answers,
            training_summary=summary, backend=self._mms_backend)

    def evaluate_reconstruction(self, output, reference, settings):
        per_target = {}
        for alias, ref in reference["references"].items():
            completed = output.completed_data["validation"][alias]["velocity"]
            mask = np.broadcast_to(ref["mask"][:, None, :], completed.shape)
            denominator = max(float(np.linalg.norm(ref["velocity"][mask])), 1e-12)
            per_target[alias] = {"gappy_withheld_relative_l2":
                float(np.linalg.norm(completed[mask]-ref["velocity"][mask])/denominator)}
        if not per_target:
            raise ValueError("reconstruction evaluation needs held-out targets")
        return ReconstructionEvaluation(
            {"gappy_withheld_relative_l2": float(np.mean([v["gappy_withheld_relative_l2"] for v in per_target.values()]))},
            per_target, {"validation_windows": len(per_target)})

    def prepare_dataset(self, dataset):
        p = Path(dataset.location).expanduser().resolve(strict=True)
        manifest = _object(json.loads(p.read_text()), "continuous MMS dataset")
        if manifest.get("schema") != MMS_DATASET_SCHEMA:
            raise ValueError("Select ContinuousPIVReconstruction for a PIV manifest, or load a saved continuous MMS dataset")
        from .completed_fields import field_from_state, field_to_state
        from .experimental_mms import prepare_experimental_mms_answers
        summary = _object(manifest.get("training_summary", {}), "saved MMS metadata")
        physical = _object(summary.get("physical_settings", {}), "saved physical settings")
        supported = {"viscosity", "coordinate_units", "mms_geometry", "projection_config"}
        if set(dataset.settings) - supported:
            raise ValueError("saved continuous data accepts only matching frozen physical settings")
        for key in ("viscosity", "coordinate_units"):
            if key in dataset.settings and dataset.settings[key] != physical.get(key):
                raise ValueError(f"saved continuous dataset has a different frozen {key}")
        if "projection_config" in dataset.settings:
            from .completed_fields import numerical_symbols
            config_type = numerical_symbols().StructuredNSProjectionConfig
            requested = config_type.from_dict(dict(dataset.settings["projection_config"]))
            requested.validate()
            frozen = config_type.from_dict(dict(physical.get("projection_config", {})))
            if requested.to_dict() != frozen.to_dict():
                raise ValueError("saved continuous dataset has a different frozen projection_config")
        answers = {role: tuple({
                        **{key: row[key] for key in ("case_id", "source_recording") if key in row},
                        "answer": field_from_state(row["answer"])} for row in rows)
                   for role, rows in manifest["splits"].items()}
        for rows in answers.values():
            for row in rows:
                expected_geometry = dataset.settings.get("mms_geometry", dataset.geometry or None)
                if expected_geometry is not None:
                    actual = row["answer"].geometry_spec.to_dict()
                    if any(actual.get(key, "no_slip" if key == "outer_boundary" else None) != value
                           for key, value in expected_geometry.items()
                           if key in ("domain_bounds", "obstacle_polygons", "outer_boundary")):
                        raise ValueError("saved continuous dataset has a different frozen geometry")
        if "prepared_case_schema" in manifest:
            if manifest["prepared_case_schema"] != _PREPARED_CASE_SCHEMA:
                raise ValueError("unsupported frozen MMS Case snapshot schema")
            return _restore_prepared_mms_dataset(dataset, manifest, answers, backend=self._mms_backend)
        if any("case" in row for rows in manifest["splits"].values() for row in rows):
            raise ValueError("frozen MMS Case snapshots require their explicit schema")
        return prepare_experimental_mms_answers(dataset, answers,
            training_summary=summary, backend=self._mms_backend)


def save_continuous_mms_dataset(output: ReconstructionOutput, path):
    """Project completed data with the common fitter, then freeze solver answers."""
    from .completed_fields import field_to_state
    answers, summary = project_reconstructed_output(output)
    payload = {"schema": MMS_DATASET_SCHEMA,
        "splits": {role: [{**{key: row[key] for key in ("case_id", "source_recording") if key in row},
                           "answer": field_to_state(row["answer"])} for row in rows]
                   for role, rows in answers.items()}, "training_summary": summary}
    p = Path(path)
    if p.exists():
        raise FileExistsError(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    return p


def save_prepared_mms_dataset(prepared, path):
    """Freeze original Case identities and fields for reuse and continuation."""
    from .experimental_mms import _case_dict, _dataset_digest
    from .completed_fields import field_to_state
    final_count = int(prepared.training_summary.get("terminal_final_test_cases", 0))
    groups = {
        "generator_training": prepared.generator_training_cases,
        "generator_validation": prepared.generator_validation_cases,
        "development": prepared.development_cases,
        "final_test": prepared.hidden_cases[:final_count],
        "verifier": prepared.hidden_cases[final_count:],
    }
    for records in groups.values():
        if any(record.case.source is not CaseSource.INITIAL for record in records):
            raise ValueError("prepared MMS datasets freeze only Original Cases")
    payload = {"schema": MMS_DATASET_SCHEMA,
        "prepared_case_schema": _PREPARED_CASE_SCHEMA,
        "prepared_dataset_sha256": _dataset_digest(prepared),
        "training_summary": dict(prepared.training_summary),
        "splits": {role: [{"case_id": record.case.case_id,
                           "case": _case_dict(record.case),
                           "answer": field_to_state(record.truth._answer)} for record in records]
                   for role, records in groups.items()}}
    p = Path(path)
    if p.exists():
        raise FileExistsError(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    return p


def _restore_prepared_mms_dataset(dataset, manifest, answers, *, backend=None):
    """Restore frozen Originals through the normal manufactured-case validator."""
    from pde_agent_verifier_solver.sdk.macro_loop import CohortKind
    from .experimental_mms import (
        ExperimentalMMSDatasetRegistry, _case_from_dict, _contract, _dataset_digest,
    )
    if set(manifest["splits"]) != set(ROLES):
        raise ValueError("frozen MMS dataset must contain exactly the five solver splits")
    summary = _object(manifest["training_summary"], "frozen MMS summary")
    counts = {role: len(rows) for role, rows in manifest["splits"].items()}
    if (summary.get("split_counts") != counts
            or summary.get("terminal_final_test_cases") != counts["final_test"]
            or summary.get("official_hidden_cases") != counts["verifier"]):
        raise ValueError("frozen MMS split inventory differs from its saved summary")
    plans, seen, contracts = {}, set(), []
    for role, rows in manifest["splits"].items():
        expected_visibility = CaseVisibility.DEVELOPMENT if role == "development" else CaseVisibility.HIDDEN
        plans[role] = []
        for row, parsed in zip(rows, answers[role]):
            case = _case_from_dict(_object(row.get("case"), "frozen public Case"))
            if (case.case_id != row["case_id"] or not case.case_id or case.case_id in seen
                    or case.source is not CaseSource.INITIAL or case.parent_case_id is not None
                    or case.visibility is not expected_visibility):
                raise ValueError("frozen MMS Case identity, visibility or source differs from its split")
            lineage = _object(case.public_data.get("distribution_lineage"), "frozen Case lineage")
            if lineage.get("split") != role:
                raise ValueError("frozen MMS Case lineage differs from its scientific split")
            seen.add(case.case_id)
            answer = parsed["answer"]
            contracts.append(_contract(answer))
            plans[role].append((case, answer))
    if (not contracts or any(value != contracts[0] for value in contracts)
            or summary.get("continuous_contract") != contracts[0]):
        raise ValueError("frozen MMS continuous answer contract differs from its saved summary")
    from .completed_fields import field_to_state
    provenance = _object(summary.get("source_provenance", {}), "frozen source provenance")
    if set(provenance) - seen:
        raise ValueError("frozen source provenance refers to unknown Case identities")
    # A supplied backend may already own these Originals. Reuse only byte-equivalent
    # answers, because the source protects the identity of an active hidden field.
    backend = backend or ExperimentalMMSDatasetRegistry()
    converted = {}
    for role, entries in plans.items():
        cohort = CohortKind.DEVELOPMENT if role == "development" else CohortKind.SELECTION
        records = []
        for case, answer in entries:
            existing = backend._records.get(case.case_id)
            if existing is not None:
                if existing.case != case or field_to_state(existing.truth._answer) != field_to_state(answer):
                    raise ValueError("frozen MMS data attempts to replace an active Case or answer")
                answer = existing.truth._answer
            records.append(backend.case_source.register_existing_answer(
                case, getattr(answer, "raw_answer", answer), cohort=cohort))
        converted[role] = tuple(records)
    prepared = PreparedDataset(
        dataset, converted["development"], converted["final_test"] + converted["verifier"],
        dict(summary), converted["generator_training"], converted["generator_validation"],
    )
    backend.bind_dataset(prepared)
    return prepared
