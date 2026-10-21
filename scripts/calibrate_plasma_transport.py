#!/usr/bin/env python3
"""Collect private real Plasma references on CUDA, then freeze transport scales.

Run each ``collect`` family in a separate Python process: the historical VPFP
and Artemis packages share a legacy import namespace. This command never
submits a job, runs a Candidate, trains Guidance, or replaces a failed draw.
The CPU-only ``fit`` command accepts only complete, hashed collection files.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]
SCHEMA = "plasma_macro_transport_reference_collection_v1"
MASTER_SEED = 202610031701
COUNTS = {"strong": 64, "weak": 64, "artemis": 32}
POINT_COUNT = 128
UNITS = {"density": 1., "velocity": 1., "electric_field": 1.}


def _sha(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path, payload):
    from pde_agent_verifier_solver.engine.checkpoint_store import atomic_json
    atomic_json(Path(path), payload)


def _seed(family, index, kind):
    return int.from_bytes(sha256(
        f"plasma-transport-independent-reference:{MASTER_SEED}:{family}:{index}:{kind}".encode()
    ).digest()[:8], "big") & ((1 << 63)-1)


def _require_compute():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("real reference collection requires a Slurm compute allocation")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("real reference collection requires CUDA; CPU fallback is prohibited")
    return torch


def _condition_ranges(runtime, backend):
    decoder = (runtime.expanded_model if hasattr(runtime, "expanded_model") else
               runtime.decoder.base_decoder if hasattr(runtime, "decoder") else None)
    if decoder is None:
        raise ValueError("cannot locate the real VPFP decoder condition ranges")
    result = {}
    for name in ("collision_nu_range", "temperature_range"):
        raw = getattr(decoder.config, name, None)
        if raw is None or len(raw) != 2:
            raise ValueError("native decoder has no explicit " + name)
        pair = tuple(float(value) for value in raw)
        if not all(math.isfinite(value) for value in pair) or pair[1] <= pair[0]:
            raise ValueError("native decoder condition range is not finite/nondegenerate: " + name)
        result[name] = pair
        capture = getattr(backend, "capture", None)
        base = getattr(capture, "base", None)
        sampling = getattr(getattr(base, "config", None), name, None)
        if sampling is not None and not np.allclose(pair, sampling, rtol=0., atol=0.):
            raise ValueError("native sampler and decoder condition ranges differ: " + name)
    if result["collision_nu_range"][0] < 0 or result["temperature_range"][0] <= 0:
        raise ValueError("invalid physical nu/temperature ranges")
    return result


def _stratum(condition, ranges):
    values = np.asarray(condition, dtype=np.float64)
    if values.shape != (2,) or not np.isfinite(values).all():
        raise ValueError("VPFP reference condition requires finite nu and bath temperature")
    bins = []
    for value, name, count in zip(values, ("collision_nu_range", "temperature_range"), (4, 2)):
        lo, hi = ranges[name]
        # Conditions pass through native FP32 storage; only endpoint roundoff
        # is tolerated. The bin edges and physical range are never fitted.
        tolerance = 4*np.finfo(np.float32).eps*max(abs(lo), abs(hi), 1.)
        if value < lo-tolerance or value > hi+tolerance:
            raise ValueError("sampled condition leaves the native range: " + name)
        bins.append(min(count-1, max(0, int(np.floor(count*(value-lo)/(hi-lo))))))
    return f"nu{bins[0]}_T{bins[1]}"


def _expected_strata(family):
    return {"all"} if family == "artemis" else {f"nu{i}_T{j}" for i in range(4) for j in range(2)}


def _uniform_points(case, family, seed):
    rng = np.random.default_rng(seed)
    if family == "artemis":
        domain = case.public_data["equation"]["domain"]
        intervals = [domain["time_interval"], domain["x_interval"]]
    else:
        question = case.public_data["question"]
        intervals = [question["time_domain"]]
        lengths = question["grid"]["x_domain"]
        if len(lengths) != 2:
            raise ValueError("VPFP reference domain must have two spatial lengths")
        intervals.extend((0., length) for length in lengths)
    bounds = np.asarray(intervals, dtype=np.float64)
    if bounds.shape != (2 if family == "artemis" else 3, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("invalid explicit physical time/space domain")
    return rng.uniform(bounds[:, 0], bounds[:, 1], size=(POINT_COUNT, len(bounds)))


def _canonical_vpfp(values):
    from pde_agent_verifier_solver.plugins.plasma_transport import canonicalize_2d
    packed = {"rho": values["rho"]}
    for name, fields in (("electric", ("electric_x", "electric_y")),
                         ("momentum", ("momentum_x", "momentum_y")),
                         ("pressure", ("pressure_xx", "pressure_xy", "pressure_yy")),
                         ("heat_flux", ("heat_flux_x", "heat_flux_y"))):
        array = np.asarray(values[name], dtype=np.float64)
        if array.shape != (POINT_COUNT, len(fields)):
            raise ValueError("malformed collected VPFP transport field: " + name)
        packed.update({field: array[:, column] for column, field in enumerate(fields)})
    return canonicalize_2d(packed)


def _reference_values(backend, runtime, record, family, coordinates):
    if family != "artemis":
        return _canonical_vpfp(runtime.evaluate_transport_macros(record.case.case_id, coordinates))
    from pde_agent_verifier_solver.plugins.plasma_transport import canonicalize_1d, VP1D_TRANSPORT_FIELDS
    from pde_agent_verifier_solver.sdk import AnswerQuery
    query = AnswerQuery.at_points(fields=VP1D_TRANSPORT_FIELDS,
        points=tuple({"t": float(t), "x": float(x)} for t, x in coordinates))
    return canonicalize_1d(record.truth.answer(query))


def _actual_root(backend, record, family):
    if family == "artemis":
        manufactured = getattr(record.truth, "manufactured", None)
        root = getattr(getattr(manufactured, "audit", None), "seed", None)
    else:
        root = backend._draws[record.case.case_id]["root_seed"]
    if isinstance(root, bool) or not isinstance(root, (int, np.integer)) or int(root) < 0:
        raise ValueError("cannot verify the actual private native reference root seed")
    return int(root)


def collect(family, config_path, output):
    """Collect the fixed reference budget, retaining every failure and its root."""
    if family not in COUNTS:
        raise ValueError("unknown calibration family")
    torch = _require_compute()
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    config = json.loads(config_path.read_text())
    if config.get("schema") != "pde_gav_plasma_gpu_v1" or config.get("equation") != family:
        raise ValueError("collect requires the explicit matching family's legacy/v1 experiment config")
    device = (config["backend"]["factory_settings"].get("device") if family == "artemis"
              else config["backend"].get("device"))
    if not isinstance(device, str) or not device.startswith("cuda"):
        raise ValueError("reference backend must explicitly use CUDA")
    if output.exists():
        raise FileExistsError("reference output must be fresh; failed/partial evidence is never overwritten")
    output.mkdir(parents=True, mode=0o700)
    started = perf_counter()
    report = {"schema": SCHEMA, "family": family, "status": "running",
              "source_role": "independent_reference_calibration", "master_seed": MASTER_SEED,
              "expected_case_count": COUNTS[family], "points_per_case": POINT_COUNT,
              "units": UNITS, "config_path": str(config_path), "config_sha256": _sha(config_path),
              "collection_script_sha256": _sha(__file__), "slurm_job_id": os.environ["SLURM_JOB_ID"],
              "cuda_device": torch.cuda.get_device_name(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "candidate_data_used": False, "guidance_trained": False, "failed_draw_replacement": False,
              "root_isolation": "independent family/index SHA256 seed namespace; actual native roots recorded",
              "records": []}
    path = output/"collection.json"
    _write(path, report)
    try:
        from scripts.plasma_gpu_campaign import _binding, jsonable
        from pde_agent_verifier_solver.sdk import (GenerationRequest, GeneratorMode, GenerationKind,
                                                  GenerationPurpose, CaseVisibility)
        binding, receipt = _binding(config, output/"binding")
        report["binding_report"] = jsonable(binding.report)
        report["binding_receipt"] = jsonable(receipt)
        generator = binding.plugin.generator
        backend = generator._backend()
        runtime = getattr(backend, "runtime", None)
        ranges = {} if family == "artemis" else _condition_ranges(runtime, backend)
        report["condition_ranges"] = ranges
        report["stratification"] = ("all: collisionless Artemis has no nu/bath-temperature condition axes"
                                    if family == "artemis" else "4 equal-width native nu bins x 2 equal-width native bath-temperature bins")
        state = generator.load(binding.checkpoint, {})
        for index in range(COUNTS[family]):
            attempt_started = perf_counter()
            request_seed, point_seed = _seed(family, index, "draw"), _seed(family, index, "points")
            row = {"index": index, "request_seed": request_seed, "point_seed": point_seed,
                   "status": "running", "released": False}
            records = ()
            try:
                records = tuple(generator.sample(state, GenerationRequest(round_index=1, count=1,
                    seed=request_seed, mode=GeneratorMode.ONLINE, kind=GenerationKind.BASE,
                    purpose=GenerationPurpose.PROMOTION, visibility=CaseVisibility.HIDDEN)))
                if len(records) != 1:
                    raise ValueError("reference draw must return exactly one fresh BASE case")
                record, = records
                root = _actual_root(backend, record, family)
                row.update(case_id=record.case.case_id, native_root_seed=root, root_id=f"{family}:{root}")
                if family == "artemis":
                    condition, stratum = [], "all"
                else:
                    question = record.case.public_data["question"]
                    condition = [float(question["collision_nu"]), float(question["temperature"])]
                    stratum = _stratum(condition, ranges)
                row.update(condition=condition, stratum=stratum)
                coordinates = _uniform_points(record.case, family, point_seed)
                torch.cuda.synchronize()
                evaluation_started = perf_counter()
                values = _reference_values(backend, runtime, record, family, coordinates)
                torch.cuda.synchronize()
                row["macro_seconds"] = perf_counter()-evaluation_started
                asset = output/f"reference-{index:03d}.npz"
                with asset.open("xb") as stream:
                    np.savez_compressed(stream, points=coordinates, condition=np.asarray(condition), **values)
                row.update(status="passed", asset=asset.name, asset_sha256=_sha(asset), sample_count=POINT_COUNT)
            except Exception as error:
                row.update(status="failed", error_type=type(error).__name__, error=str(error),
                           traceback=traceback.format_exc())
            finally:
                try:
                    if records:
                        generator.release(state, tuple(record.case.case_id for record in records))
                    row["released"] = True
                except Exception as error:
                    row.update(status="failed", cleanup_error_type=type(error).__name__, cleanup_error=str(error))
                row["wall_seconds"] = perf_counter()-attempt_started
                report["records"].append(row)
                _write(path, report)
        successful = [row for row in report["records"] if row["status"] == "passed"]
        observed = {row["stratum"] for row in successful}
        report["missing_strata"] = sorted(_expected_strata(family)-observed)
        report["successful_case_count"] = len(successful)
        if len(successful) != COUNTS[family] or report["missing_strata"]:
            raise RuntimeError("reference budget incomplete or native condition strata missing; no replacement or reduced-count fit allowed")
        if len({row["native_root_seed"] for row in successful}) != len(successful):
            raise RuntimeError("duplicate native calibration root")
        report["status"] = "passed"
        return report
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        report["wall_seconds"] = perf_counter()-started
        _write(path, report)


def fit(inputs, output):
    from pde_agent_verifier_solver.plugins.plasma_transport import fit_calibration, TRANSPORT_FIELDS
    output = Path(output)
    if output.exists():
        raise FileExistsError("frozen calibration output already exists")
    rows, provenance, seen = [], {}, set()
    for directory in map(Path, inputs):
        manifest_path = directory/"collection.json"
        manifest = json.loads(manifest_path.read_text())
        family = manifest.get("family")
        if (manifest.get("schema") != SCHEMA or family not in COUNTS or family in seen
                or manifest.get("status") != "passed" or manifest.get("master_seed") != MASTER_SEED
                or manifest.get("source_role") != "independent_reference_calibration"
                or manifest.get("points_per_case") != POINT_COUNT or manifest.get("units") != UNITS
                or manifest.get("expected_case_count") != COUNTS[family]
                or manifest.get("candidate_data_used") is not False
                or manifest.get("failed_draw_replacement") is not False):
            raise ValueError("invalid, duplicated, or incomplete independent calibration collection")
        records = manifest.get("records", ())
        if len(records) != COUNTS[family] or {row.get("stratum") for row in records} != _expected_strata(family):
            raise ValueError("fixed reference case count or required physical strata are incomplete")
        for index, row in enumerate(records):
            if (row.get("status") != "passed" or row.get("released") is not True
                    or row.get("sample_count") != POINT_COUNT or row.get("index") != index
                    or row.get("request_seed") != _seed(family, index, "draw")
                    or row.get("point_seed") != _seed(family, index, "points")
                    or row.get("root_id") != f"{family}:{row.get('native_root_seed')}"):
                raise ValueError("reference record failed, was not released, or changed its root namespace")
            name = row["asset"]
            if Path(name).name != name or name != f"reference-{index:03d}.npz":
                raise ValueError("invalid reference asset name")
            asset = directory/name
            if asset.is_symlink() or _sha(asset) != row["asset_sha256"]:
                raise ValueError("private reference asset changed")
            with np.load(asset, allow_pickle=False) as loaded:
                if set(loaded.files) != set(TRANSPORT_FIELDS) | {"points", "condition"}:
                    raise ValueError("reference asset has unexpected fields")
                values = {name: np.array(loaded[name], copy=True) for name in TRANSPORT_FIELDS}
                dimension = 2 if family == "artemis" else 3
                if loaded["points"].shape != (POINT_COUNT, dimension) or not np.isfinite(loaded["points"]).all():
                    raise ValueError("reference point count or coordinates changed")
                if values["density"].shape != (POINT_COUNT, 1):
                    raise ValueError("reference sample count changed")
                condition = np.asarray(loaded["condition"])
                if not np.array_equal(condition, np.asarray(row["condition"])):
                    raise ValueError("reference native condition metadata changed")
                if family != "artemis" and _stratum(condition, manifest["condition_ranges"]) != row["stratum"]:
                    raise ValueError("reference stratum differs from native condition")
            rows.append({"family": family, "stratum": row["stratum"], "root_id": row["root_id"], "values": values})
        seen.add(family)
        provenance[family] = {"manifest_sha256": _sha(manifest_path), "config_sha256": manifest["config_sha256"],
                              "collection_script_sha256": manifest["collection_script_sha256"],
                              "condition_ranges": manifest["condition_ranges"]}
    if seen != set(COUNTS):
        raise ValueError("fit requires all three independent family collections")
    result = fit_calibration(rows, units=UNITS, provenance={"master_seed": MASTER_SEED,
        "points_per_case": POINT_COUNT, "fixed_case_counts": COUNTS,
        "sources": provenance, "candidate_data_used": False,
        "rule": "equal weight per family and per native condition stratum; no candidate-dependent floor adjustment"})
    output.parent.mkdir(parents=True, exist_ok=True)
    _write(output, result.to_dict())
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--family", choices=tuple(COUNTS), required=True)
    collect_parser.add_argument("--config", type=Path, required=True)
    collect_parser.add_argument("--output", type=Path, required=True)
    fit_parser = sub.add_parser("fit")
    fit_parser.add_argument("--input", type=Path, action="append", required=True)
    fit_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "collect":
        report = collect(args.family, args.config, args.output)
        print(json.dumps({key: report[key] for key in ("family", "status", "successful_case_count", "wall_seconds")}))
    else:
        calibration = fit(args.input, args.output)
        print(json.dumps({"status": "frozen", "objective_hash": calibration.objective_hash,
                          "floors": calibration.to_dict()["floors"]}))


if __name__ == "__main__":
    main()
