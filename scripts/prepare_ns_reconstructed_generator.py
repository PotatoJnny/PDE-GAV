#!/usr/bin/env python
"""Prepare measured or reconstructed PIV for the original numerical NS Generator.

Measured mode reads complete measurement references directly. Reconstructed
mode first applies sparse-observation Gappy-POD. Both use the same numerical
continuous projector, PCA, and optional Flow trainer.
No viscosity is inferred or required until an NS MMS Case is constructed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for source_root in (ROOT / "legacy/ns/src", ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from pde_agent_verifier_solver.plugins.ns.completed_fields import (
    completed_grid, numerical_symbols, project_completed_velocity,
)
from pde_agent_verifier_solver.plugins.ns.experimental_mms_project import (
    _coordinates, _path, _window_clock, prepare_piv_reconstruction_source,
)
from pde_agent_verifier_solver.plugins.ns.reconstruction.real_piv import (
    GappyPODPosterior, fit_trajectory_pod,
)
from pde_agent_verifier_solver.sdk import DatasetInput, DatasetKind
from agent_verifier.neural_generator.structured_ns_distribution import StructuredCoefficientPCA


ROLES = ("generator_training", "generator_validation", "development", "verifier", "final_test", "reconstruction_validation")


def load_measured_selected(dataset):
    """Read declared measurement references without opening old completed fields.

    Sparse layouts and reconstruction settings are irrelevant to this path.
    Original acquisition windows still enforce separation of scientific roles.
    """
    path = Path(dataset.location).expanduser().resolve(strict=True)
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "pde_completed_case_dataset_v1" or manifest.get("problem") != "ns":
        raise ValueError("measured mode expects a completed NS PIV manifest")
    splits = manifest.get("splits", {})
    if len(splits.get("generator_training", ())) < 2 or not splits.get("generator_validation"):
        raise ValueError("measured mode requires separate Generator training and validation cases")
    clock = _window_clock(manifest)
    coordinates = None
    completed, records = {}, {}
    seen_ids, seen_paths, intervals = set(), set(), defaultdict(list)
    for role in ROLES:
        rows = splits.get(role, ())
        if not isinstance(rows, (list, tuple)):
            raise ValueError(f"split {role} must be a sequence")
        fields, records[role] = [], []
        for row in rows:
            cid = row.get("case_id")
            if not isinstance(cid, str) or not cid or cid in seen_ids:
                raise ValueError("PIV split case IDs must be nonempty and disjoint")
            seen_ids.add(cid)
            reference = _path(row.get("measurement_reference_file"), path.parent)
            if reference in seen_paths:
                raise ValueError("PIV splits reuse the same measurement reference file")
            seen_paths.add(reference)
            case_file = row.get("case_file")
            if case_file:
                old_case = Path(case_file).expanduser()
                old_case = (old_case if old_case.is_absolute() else path.parent / old_case).resolve()
                if reference == old_case and row.get("generator_target_kind") != "measured_full_field":
                    raise ValueError("measurement reference points to a reconstructed case, not measured_full_field")
            recording, frames = row.get("source_recording"), row.get("source_frames")
            if not isinstance(recording, str) or not recording or frames is None:
                raise ValueError("each measured PIV case requires source_recording and source_frames")
            if (not isinstance(frames, (list, tuple)) or len(frames) != 2
                    or any(isinstance(value, bool) or not isinstance(value, int) for value in frames)
                    or frames[0] < 0 or frames[1] <= frames[0]):
                raise ValueError("invalid measured PIV source recording/window")
            if any(other != role and frames[0] < stop and start < frames[1]
                   for other, start, stop in intervals[recording]):
                raise ValueError("PIV source frames overlap across scientific splits")
            intervals[recording].append((role, *frames))
            with np.load(reference, allow_pickle=False) as arrays:
                x, y, times = _coordinates(arrays, clock=clock, source_frames=frames)
                if coordinates is None:
                    coordinates = x, y, times
                if any(actual.shape != expected.shape or not np.allclose(actual, expected, rtol=1e-8, atol=1e-10)
                       for actual, expected in zip((x, y), coordinates[:2])):
                    raise ValueError("measured PIV cases need common spatial coordinates")
                velocity = np.array(arrays["velocity"], copy=True)
            if velocity.shape != (len(times), 2, len(x)) or not np.isfinite(velocity).all():
                raise ValueError("measured reference requires finite velocity[T,2,P]")
            fields.append(velocity)
            records[role].append({"case_id": cid, "source_recording": recording,
                                  "measurement_reference_file": str(reference)})
        completed[role] = np.stack(fields) if fields else np.empty((0,))
    metadata = {"split_counts": {role: len(records[role]) for role in ROLES},
                "frame_clock": {"policy": "declared_sampling_hz" if clock["sampling_hz"] else "first_window",
                                "sampling_hz": clock["sampling_hz"],
                                "maximum_timestamp_adjustment": clock["max_adjustment"]}}
    return completed, records, coordinates, metadata


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def coefficient_sha256(values):
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.float32).tobytes(order="C")).hexdigest()


def summary(values):
    data = np.asarray(values, dtype=float)
    return {"count": int(data.size), "mean": float(data.mean()), "median": float(np.median(data)),
            "p90": float(np.quantile(data, .9)), "maximum": float(data.max())}


def relative_errors(actual, expected):
    difference = (np.asarray(actual) - expected).reshape(len(actual), -1)
    target = np.asarray(expected).reshape(len(actual), -1)
    return np.linalg.norm(difference, axis=1) / np.maximum(np.linalg.norm(target, axis=1), 1e-12)


def event(stage, **values):
    print(json.dumps({"stage": stage, "unix_time": time.time(), **values}), flush=True)


def export_coefficient_corpus(values, *, case_ids, families, signature, output, split, shard_cases=256):
    """Write exactly the corpus format consumed by the numerical PCA/Flow code."""
    output = Path(output)
    values = np.ascontiguousarray(values, dtype=np.float32)
    if values.ndim != 5 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("coefficient corpus needs finite [N,2,R,Kx,Ky] values")
    if len(case_ids) != len(values) or len(families) != len(values) or len(set(case_ids)) != len(values):
        raise ValueError("coefficient corpus requires aligned unique Case IDs and families")
    if shard_cases < 1:
        raise ValueError("shard_cases must be positive")
    output.mkdir(parents=True, exist_ok=False)
    (output / "shards").mkdir()
    rows = []
    for index, start in enumerate(range(0, len(values), shard_cases)):
        stop = min(start + shard_cases, len(values))
        path = output / "shards" / f"coefficients_{index:05d}.npz"
        np.savez_compressed(path, coefficients=values[start:stop], case_ids=np.asarray(case_ids[start:stop]),
                            families=np.asarray(families[start:stop]))
        rows.append({"path": str(path.relative_to(output)), "count": stop-start, "sha256": file_sha256(path)})
    manifest = {"schema": "ns2d_structured_coefficient_dataset_v2", "split": split,
                "encoded_cases": len(values), "coefficient_shape": list(values.shape[1:]),
                "encoding_signature": {**signature, "split": split, "selected_case_count": len(values)},
                "shards": rows}
    path = output / "coefficient_manifest.json"
    write_json(path, manifest)
    return path


def velocity_from_coefficients(values, *, x, y, times, geometry, projection_config, device="cpu"):
    """Evaluate with the numerical projector's own velocity/temporal operators.

    This does not construct an MMS problem and therefore has no viscosity input.
    Only measured lattice sites are returned, in the original point ordering.
    """
    import torch
    module = numerical_symbols()
    x, y = np.asarray(x), np.asarray(y)
    if "domain_bounds" in geometry:
        left, right, bottom, top = geometry["domain_bounds"]
        keep = (x >= left) & (x <= right) & (y >= bottom) & (y <= top)
        x, y = x[keep], y[keep]
    grid, (ix, iy) = completed_grid(x, y, geometry)
    config = module.StructuredNSProjectionConfig.from_dict(dict(projection_config))
    projector = module.StructuredNSCoefficientProjector(geometry=grid, t=times, config=config, device=device)
    coefficient = torch.as_tensor(values, dtype=torch.float64, device=device)
    if coefficient.ndim == 4:
        coefficient = coefficient[None]
    expected = (2, config.time_coefficient_count, config.spatial_mode_x + 1, config.spatial_mode_y + 1)
    if tuple(coefficient.shape[1:]) != expected:
        raise ValueError("coefficient shape differs from the numerical projection contract")
    spatial = torch.einsum("tr,nrs->nts", projector.temporal_design,
                          coefficient[:, 0].reshape(len(coefficient), config.time_coefficient_count, -1))
    velocity = torch.einsum("fs,nts->ntf", projector.spatial_operators["velocity_design"], spatial)
    velocity = velocity.reshape(len(coefficient), len(times), 2, len(ix))
    flat = ix * len(grid.y) + iy
    source_order = np.searchsorted(np.sort(flat), flat)
    return velocity[..., source_order].detach().cpu().numpy()


def reconstruct_selected(source, *, rank, prior_precision, batch_cases=32):
    """Dense calibration is training-only; all targets use their sparse observations."""
    prior, pod_report = fit_trajectory_pod(source.fit_data["calibration_windows"], max_rank=rank)
    method = GappyPODPosterior(prior.rank, prior_precision, 1.0)
    completed = {}
    records = {}
    for role in ("generator_training", "generator_validation", "development", "verifier", "final_test", "reconstruction_validation"):
        items = tuple(source.target_observations[role])
        groups = defaultdict(list)
        for index, item in enumerate(items):
            obs = item["observations"]
            key = (tuple(obs.layout.sensor_indices), tuple(obs.layout.observed_frames), tuple(obs.noise_standard_deviation))
            groups[key].append((index, item))
        result = np.empty((len(items), *prior.mean.shape), dtype=np.float32)
        for group in groups.values():
            for start in range(0, len(group), batch_cases):
                chunk = group[start:start+batch_cases]
                estimates = method.reconstruct_many_observations(
                    prior, tuple(item["observations"] for _, item in chunk), include_uncertainty=False)
                for (index, _), estimate in zip(chunk, estimates):
                    result[index] = estimate.velocity
        completed[role], records[role] = result, items
        event("reconstructed", split=role, cases=len(items), pod_rank=prior.rank)
    return completed, records, pod_report, prior.rank


def prepare(args):
    started = time.time()
    out = Path(args.out).expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"refusing to reuse Generator preparation directory: {out}")
    if args.viscosity is not None and (not np.isfinite(args.viscosity) or args.viscosity <= 0):
        raise ValueError("viscosity must be explicitly positive when freezing solver Cases")
    if args.flow_steps < 2 or min(args.batch_cases, args.flow_batch_size, args.checkpoint_every) < 1:
        raise ValueError("training requires at least two steps and positive batch/checkpoint sizes")
    manifest = Path(args.manifest).expanduser().resolve(strict=True)
    geometry = json.loads(Path(args.geometry_json).read_text())
    if "obstacle_polygons" not in geometry or "outer_boundary" not in geometry:
        raise ValueError("geometry JSON must explicitly declare obstacle_polygons and outer_boundary")
    projection = numerical_symbols().StructuredNSProjectionConfig(
        spatial_mode_x=args.spatial_mode_x, spatial_mode_y=args.spatial_mode_y,
        time_order=args.time_order, spatial_extension=args.spatial_extension,
        geometry_mask_transition=.05, ridge_relative=args.ridge_relative).to_dict()
    if getattr(args, 'projection_config_json', None):
        projection = numerical_symbols().StructuredNSProjectionConfig.from_dict(
            json.loads(Path(args.projection_config_json).read_text())).to_dict()
        numerical_symbols().StructuredNSProjectionConfig.from_dict(projection).validate()
    dataset = DatasetInput(args.dataset_id, DatasetKind.EXPERIMENTAL, str(manifest),
        settings={"coordinate_units": args.coordinate_units, "mms_geometry": geometry})
    input_mode = getattr(args, "input_mode", "reconstructed")
    if input_mode not in ("measured", "reconstructed"):
        raise ValueError("input_mode must be measured or reconstructed")
    if not np.isfinite(args.pca_relative_tolerance) or args.pca_relative_tolerance < 0:
        raise ValueError("PCA relative tolerance must be finite and nonnegative")
    if input_mode == "measured":
        completed, records, (x, y, times), metadata = load_measured_selected(dataset)
    else:
        prepared_source = prepare_piv_reconstruction_source(dataset, {}, args.seed)
        source = prepared_source.program_input
        metadata = source.metadata
        x, y, times = (source.fit_data[name] for name in ("x", "y", "time"))
    out.mkdir(parents=True)
    spec = {"schema": "ns_reconstructed_generator_preparation_v1", "source_manifest": str(manifest),
        "source_manifest_sha256": file_sha256(manifest), "dataset_id": args.dataset_id,
        "input_mode": input_mode, "field_source": "measurement_reference_file" if input_mode == "measured" else "sparse_observation_reconstruction",
        "coordinate_units": args.coordinate_units, "viscosity": args.viscosity, "pressure_model": "zero",
        "geometry_declaration": geometry, "projection_config": projection,
        "seed": args.seed, "source_clock": metadata["frame_clock"],
        "excluded_from_generator_fitting_and_selection": ["development", "verifier", "final_test", "reconstruction_validation"]}
    if input_mode == "reconstructed":
        spec["reconstruction"] = {"rank": args.pod_rank, "prior_precision": args.prior_precision, "noise_scale": 1.0}
    write_json(out / "preparation_config.json", spec)
    event("source_prepared", input_mode=input_mode, split_counts=metadata["split_counts"])
    if input_mode == "reconstructed":
        completed, records, pod_report, pod_rank = reconstruct_selected(
            source, rank=args.pod_rank, prior_precision=args.prior_precision, batch_cases=args.batch_cases)
        validation_references = prepared_source.reconstruction_validation_reference["references"]
        recon_errors = []
        for row, velocity in zip(records["reconstruction_validation"], completed["reconstruction_validation"]):
            reference = validation_references[row["alias"]]
            mask = np.broadcast_to(reference["mask"][:, None, :], velocity.shape)
            recon_errors.append(float(np.linalg.norm((velocity-reference["velocity"])[mask]) /
                                      max(np.linalg.norm(reference["velocity"][mask]), 1e-12)))
        write_json(out / "reconstruction_report.json", {"pod": pod_report,
            "withheld_velocity_relative_l2": summary(recon_errors), "per_case": dict(zip(
                [item["case_id"] for item in records["reconstruction_validation"]], recon_errors)),
            "evaluation_source": "sealed reconstruction_validation reference only"})
    else:
        write_json(out / "measurement_sources.json", records)
    coefficients, projection_reports = {}, {}
    for role in ("generator_training", "generator_validation", "development", "verifier", "final_test", "reconstruction_validation"):
        if not len(completed[role]):
            continue
        values, report, geometry_spec = project_completed_velocity(completed[role], x, y, times,
            geometry=geometry, projection_config=projection, pressure_model="zero", device=args.device)
        coefficients[role] = np.asarray(values, dtype=np.float32)
        projection_reports[role] = report
        event("projected", split=role, cases=len(values), coefficient_shape=list(values.shape[1:]))
    write_json(out / "projection_report.json", projection_reports)
    signature = {"schema": "ns2d_structured_coefficient_dataset_signature_v2",
        "split_manifest": str(manifest), "split_manifest_sha256": file_sha256(manifest),
        "projection_config": projection, "coefficient_shape": list(coefficients["generator_training"].shape[1:]),
        "geometry": geometry_spec, "time_interval": [float(times[0]), float(times[-1])],
        "training_coefficients_sha256": coefficient_sha256(coefficients["generator_training"]),
        "validation_coefficients_sha256": coefficient_sha256(coefficients["generator_validation"])}
    manifests = {}
    for role, short in (("generator_training", "train"), ("generator_validation", "validation")):
        manifests[short] = export_coefficient_corpus(coefficients[role],
            case_ids=[r["case_id"] for r in records[role]], families=[r.get("source_recording", "experimental") for r in records[role]],
            signature=signature, output=out / "coefficients" / short, split=short)
    write_json(out / "continuous_contract.json", signature)
    fitted_indices = np.asarray(projection_reports["generator_training"]["input_contract"]["fitted_point_indices"], dtype=int)
    validation_arrays = {"x": x[fitted_indices], "y": y[fitted_indices], "time": times,
                         "coefficients": coefficients["generator_validation"]}
    if input_mode == "measured":
        validation_arrays["measured"] = completed["generator_validation"][..., fitted_indices]
    else:
        validation_arrays.update(
            reconstructed=completed["generator_validation"][..., fitted_indices],
            reconstruction_validation=completed["reconstruction_validation"][..., fitted_indices],
            reconstruction_validation_coefficients=coefficients["reconstruction_validation"],
            reconstruction_validation_truth=np.stack([validation_references[r["alias"]]["velocity"] for r in records["reconstruction_validation"]])[..., fitted_indices],
            reconstruction_validation_mask=np.stack([validation_references[r["alias"]]["mask"] for r in records["reconstruction_validation"]])[..., fitted_indices])
    np.savez_compressed(out / "validation_reference.npz", **validation_arrays)
    if args.viscosity is not None:
        from pde_agent_verifier_solver.plugins.ns.completed_fields import field_from_coefficients
        from pde_agent_verifier_solver.plugins.ns.experimental_mms import prepare_experimental_mms_answers
        from pde_agent_verifier_solver.plugins.ns.experimental_mms_project import save_prepared_mms_dataset
        split_answers = {}
        for role in ("generator_training", "generator_validation", "development", "verifier", "final_test"):
            split_answers[role] = [{"case_id": row["case_id"], "source_recording": row.get("source_recording"),
                "answer": field_from_coefficients(value, config=projection, geometry=geometry_spec,
                    time_interval=signature["time_interval"], viscosity=args.viscosity, device="cpu")}
                for row, value in zip(records[role], coefficients.get(role, ()))]
        prepared_dataset = prepare_experimental_mms_answers(dataset, split_answers, training_summary={
            "physical_settings": {"viscosity": args.viscosity, "coordinate_units": args.coordinate_units,
                "mms_geometry": geometry, "projection_config": projection, "pressure_model": "zero"},
            "input_mode": input_mode, "field_source": spec["field_source"],
            "preparation_method": ("measured_to_numerical_structured_projection" if input_mode == "measured"
                                   else "gappy_pod_to_numerical_structured_projection"),
            **({"reconstruction_method": "gappy_pod_to_numerical_structured_projection"}
               if input_mode == "reconstructed" else {}),
            "generator_fitting_split": "generator_training", "generator_selection_split": "generator_validation"})
        save_prepared_mms_dataset(prepared_dataset, out / "prepared_dataset.json")
        event("solver_dataset_frozen", case_counts={role: len(rows) for role, rows in split_answers.items()})
    requested = sorted(set(int(value) for value in args.pca_ranks.split(",")))
    rank_cap = min(len(coefficients["generator_training"])-1, int(np.prod(coefficients["generator_training"].shape[1:])))
    if input_mode == "reconstructed":
        rank_cap = min(rank_cap, pod_rank)
    ranks = [rank for rank in requested if 1 <= rank <= rank_cap]
    if not ranks:
        raise ValueError("PCA rank candidates must fit the training coefficient dimension")
    pca = StructuredCoefficientPCA.fit(coefficients["generator_training"], rank=max(ranks),
        source_signature=signature, oversample=16, power_iterations=4, seed=args.seed)
    reference_velocity = velocity_from_coefficients(coefficients["generator_validation"], x=x, y=y, times=times,
        geometry=geometry, projection_config=projection, device=args.device)
    rank_reports = []
    for rank in ranks:
        selected = pca.with_rank(rank)
        reconstruction = selected.reconstruct(coefficients["generator_validation"]).detach().cpu().numpy()
        velocity = velocity_from_coefficients(reconstruction, x=x, y=y, times=times,
            geometry=geometry, projection_config=projection, device=args.device)
        rank_reports.append({"rank": rank, "manifold_velocity_relative_l2": summary(relative_errors(velocity, reference_velocity)),
            "coefficient_relative_l2": summary(relative_errors(reconstruction, coefficients["generator_validation"]))})
    best = min(row["manifold_velocity_relative_l2"]["median"] for row in rank_reports)
    threshold = best * (1 + args.pca_relative_tolerance)
    selected_rank = min(row["rank"] for row in rank_reports if row["manifold_velocity_relative_l2"]["median"] <= threshold)
    pca.save(out / "pca_checkpoint.pt")
    write_json(out / "pca_selection.json", {"schema": "ns_pca_selection_v1", "selected_rank": selected_rank,
        "rule": "smallest_rank_within_relative_tolerance_of_best_held_out_velocity_error",
        "relative_tolerance": args.pca_relative_tolerance, "candidates": rank_reports,
        "candidate_rank_cap": rank_cap, "selection_split": "generator_validation"})
    flow_command = [sys.executable, str(ROOT / "legacy/ns/scripts/train_ns_structured_coefficient_flow.py"),
        "--train-manifest", str(manifests["train"]), "--validation-manifest", str(manifests["validation"]),
        "--pca-checkpoint", str(out / "pca_checkpoint.pt"), "--rank-selection", str(out / "pca_selection.json"),
        "--out", str(out / "base_generator"), "--steps", str(args.flow_steps), "--batch-size", str(args.flow_batch_size),
        "--validation-cases", str(len(coefficients["generator_validation"])), "--width", str(args.flow_width),
        "--depth", str(args.flow_depth), "--time-width", str(args.flow_time_width), "--lr", "0.0002",
        "--warmup-steps", str(min(1000, max(1, args.flow_steps//5))), "--checkpoint-every", str(args.checkpoint_every),
        "--validate-every", str(args.checkpoint_every), "--seed", str(args.seed),
        "--amp", "bfloat16" if args.device.startswith("cuda") else "none", "--device", args.device]
    write_json(out / "flow_command.json", {"command": flow_command})
    report = {"schema": "ns_reconstructed_generator_prepared_v1", "output": str(out),
        "training_cases": len(coefficients["generator_training"]), "validation_cases": len(coefficients["generator_validation"]),
        "selected_pca_rank": selected_rank, "input_mode": input_mode, "field_source": spec["field_source"],
        "geometry": geometry_spec, "viscosity": args.viscosity, "pressure_model": "zero",
        "fitted_point_count": len(fitted_indices), "input_point_count": len(x),
        "excluded_from_generator_fitting_and_selection": ["development", "verifier", "final_test", "reconstruction_validation"],
        "elapsed_seconds": time.time()-started, "flow_training_requested": args.train_flow}
    if input_mode == "reconstructed":
        report["reconstruction_validation"] = summary(recon_errors)
    write_json(out / "preparation_report.json", report)
    event("generator_prepared", **report)
    if args.train_flow:
        subprocess.run(flow_command, cwd=ROOT, check=True)
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", required=True)
    result.add_argument("--input-mode", choices=("measured", "reconstructed"), default="reconstructed",
                        help="measured reads full measurement references directly; reconstructed preserves sparse Gappy-POD preparation.")
    result.add_argument("--out", required=True)
    result.add_argument("--geometry-json", required=True)
    result.add_argument("--dataset-id", default="cylinder-reconstructed-structured-generator")
    result.add_argument("--coordinate-units", choices=("physical", "nondimensional"), default="physical")
    result.add_argument("--viscosity", type=float, help="Explicit physical viscosity; required only to freeze solver Cases, never inferred.")
    result.add_argument("--pod-rank", type=int, default=64)
    result.add_argument("--prior-precision", type=float, default=10.0)
    result.add_argument("--batch-cases", type=int, default=32)
    result.add_argument("--projection-config-json", help="Explicit frozen continuous-fit config; overrides scalar projection CLI settings.")
    result.add_argument("--spatial-mode-x", type=int, default=16)
    result.add_argument("--spatial-mode-y", type=int, default=16)
    result.add_argument("--time-order", type=int, default=12)
    result.add_argument("--spatial-extension", type=float, default=.25)
    result.add_argument("--ridge-relative", type=float, default=1e-6)
    result.add_argument("--pca-ranks", default="16,32,64")
    result.add_argument("--pca-relative-tolerance", type=float, default=.05)
    result.add_argument("--device", default="cuda")
    result.add_argument("--seed", type=int, default=20260911)
    result.add_argument("--train-flow", action="store_true")
    result.add_argument("--flow-steps", type=int, default=5000)
    result.add_argument("--flow-batch-size", type=int, default=128)
    result.add_argument("--flow-width", type=int, default=768)
    result.add_argument("--flow-depth", type=int, default=8)
    result.add_argument("--flow-time-width", type=int, default=128)
    result.add_argument("--checkpoint-every", type=int, default=1000)
    return result


if __name__ == "__main__":
    prepare(parser().parse_args())
