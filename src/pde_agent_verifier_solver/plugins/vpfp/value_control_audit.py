"""Read-only, whole-corpus gate for the value-only VPFP time representation.

This checks a property of the stored encoding, NOT physical fidelity. It never
projects a target, changes a file, samples rows, or admits a generated model.
Production-sized scans belong in a Slurm compute allocation.
"""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from . import diffusion_data as data


SCHEMA = "vpfp_whole_corpus_value_control_constraint_audit_v1"
_QUANTILES = (0.0, 0.5, 0.9, 0.99, 1.0)
_RMS_FLOOR = float(np.finfo(np.float32).tiny) * 64.0
_SLOPE_RELATIVE_FLOOR = 1e-6


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _layout(manifest):
    """Independently derive and check every published layout dimension."""
    try:
        provenance = manifest["provenance"]
        cfg, generation = provenance["decoder_config"], provenance["generation_config"]
        if (cfg["structured_time_basis"] != "piecewise_cubic_hermite_v1"
                or cfg["structured_direct_coefficients"] is not True
                or cfg["structured_spectral_latent"] is not True):
            raise ValueError("value controls require the direct piecewise cubic Hermite basis")
        k = _integer(cfg["structured_time_control_count"], "control_count", 2)
        rx = _integer(cfg["rho_max_mode_x"], "rho_max_mode_x")
        ry = _integer(cfg["rho_max_mode_y"], "rho_max_mode_y")
        spatial = _integer(cfg["structured_local_spatial_mode"], "local_spatial_mode")
        degree = _integer(cfg["hermite_total_degree"], "hermite_total_degree")
        channels = _integer(cfg["structured_local_channels"], "local_channels", 1)
        inactive = _integer(cfg["structured_kinetic_global_dim"], "inactive_dimension")
        rho_modes = (2 * rx + 1) * (2 * ry + 1) - 1
        kinetic_modes = (2 * spatial + 1) ** 2
        if rho_modes == 0 or channels != (degree + 1) * (degree + 2) // 2:
            raise ValueError("direct coefficient channel/mode layout is unsupported")
        expected = {
            "rho_mode_count": rho_modes // 2,
            "rho_time_feature_count": 2 * k,
            "rho_latent_dim": 2 * k * rho_modes,
            "kinetic_global_latent_dim": inactive,
            "local_spatial_feature_count": kinetic_modes,
            "local_time_feature_count": 2 * k,
            "local_latent_dim": channels * 2 * k * kinetic_modes,
            "hermite_mode_count": channels,
            "total_latent_dim": 2 * k * (rho_modes + channels * kinetic_modes) + inactive,
        }
        supplied = provenance["structured_latent_layout"]
        if not isinstance(supplied, Mapping) or set(supplied) != set(expected):
            raise ValueError("structured latent layout fields differ from the direct source config")
        if any(_integer(supplied[key], key) != value for key, value in expected.items()):
            raise ValueError("structured latent layout differs from the direct source config")
        if (_integer(manifest["latent_dimension"], "latent_dimension", 1) != expected["total_latent_dim"]
                or _integer(cfg["latent_dim"], "decoder latent_dim", 1) != expected["total_latent_dim"]):
            raise ValueError("corpus/decoder latent dimension disagrees with value-control layout")
        domain = tuple(float(value) for value in cfg["time_domain"])
        if (len(domain) != 2 or not all(math.isfinite(value) for value in domain)
                or domain[0] >= domain[1] or list(domain) != list(manifest["time_domain"])):
            raise ValueError("decoder and corpus physical time domains disagree")
        if (_integer(generation["nt"], "generation nt", 2) != k
                or float(generation.get("t0", 0.0)) != domain[0]
                or float(generation["t1"]) != domain[1]):
            raise ValueError("generation controls/time domain differ from the decoder")
        if cfg.get("dtype") != "float32" or generation.get("dtype") != "float32":
            raise ValueError("this gate requires the published float32 source encoding")
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError(f"missing or malformed source value-control configuration: {error}") from error
    return {"control_count": k, "rho_modes": rho_modes, "kinetic_modes": kinetic_modes,
            "hermite_channels": channels, "inactive_dimension": inactive,
            "time_domain": list(domain), "delta_tau": 1.0 / (k - 1),
            "delta_physical_time": (domain[1] - domain[0]) / (k - 1),
            "tau": np.linspace(0.0, 1.0, k, dtype=np.float64).tolist(),
            "structured_latent_layout": expected}


def _slopes(values, tau):
    """Source one-sided endpoints and centered interior, time on axis -2."""
    result = np.empty_like(values)
    result[..., 0, :] = (values[..., 1, :] - values[..., 0, :]) / (tau[1] - tau[0])
    result[..., -1, :] = (values[..., -1, :] - values[..., -2, :]) / (tau[-1] - tau[-2])
    if len(tau) > 2:
        result[..., 1:-1, :] = (values[..., 2:, :] - values[..., :-2, :]) / (tau[2:] - tau[:-2])[:, None]
    return result


def _metrics(values, stored, delta_tau):
    """Per-case/per-channel summaries; no pointwise division near zeros."""
    raw_values, raw_stored = np.asarray(values), np.asarray(stored)
    values = raw_values.astype(np.float64)
    stored = raw_stored.astype(np.float64)
    tau = np.linspace(0.0, 1.0, values.shape[-2], dtype=np.float64)
    expected = _slopes(values, tau)
    difference = stored - expected
    axes = (-2, -1)
    value_rms = np.sqrt(np.mean(values * values, axis=axes))
    slope_rms = np.sqrt(np.mean(expected * expected, axis=axes))
    error_rms = np.sqrt(np.mean(difference * difference, axis=axes))
    maximum = np.max(np.abs(difference), axis=axes)
    value_maximum = np.max(np.abs(values), axis=axes)
    # Float64 checks stored float32 values as exact real numbers. The companion
    # float32 error isolates source arithmetic/roundoff from structural failure.
    expected32 = _slopes(raw_values, tau.astype(np.float32)).astype(np.float64)
    return {
        "scaled_rms_error": delta_tau * error_rms / np.maximum(value_rms, _RMS_FLOOR),
        "scaled_max_error": delta_tau * maximum / np.maximum(value_maximum, _RMS_FLOOR),
        "slope_relative_rms_error": error_rms / np.maximum(
            slope_rms, np.maximum(_SLOPE_RELATIVE_FLOOR * value_rms / delta_tau, _RMS_FLOOR)),
        "absolute_max_slope_error": maximum,
        "float32_reconstruction_scaled_rms_error": delta_tau * np.sqrt(
            np.mean((stored - expected32) ** 2, axis=axes)) / np.maximum(value_rms, _RMS_FLOOR),
    }


def _summarize(metrics, ids, tolerance):
    if not metrics:
        raise ValueError("cannot audit an empty trajectory group")
    merged = {key: np.concatenate([entry[key] for entry in metrics]) for key in metrics[0]}
    if not all(np.isfinite(values).all() for values in merged.values()):
        raise ValueError("nonfinite value-control constraint statistic")
    failures = ((merged["scaled_rms_error"] > tolerance)
                | (merged["scaled_max_error"] > tolerance))
    return {"trajectory_count": len(ids), "passed": not bool(np.any(failures)),
            "failed_case_ids": [str(ids[index]) for index in np.flatnonzero(failures)],
            "failed_case_count": int(np.count_nonzero(failures)),
            "metrics": {key: {"max": float(np.max(values)), "mean": float(np.mean(values)),
                        "quantiles": {str(q): float(np.quantile(values, q)) for q in _QUANTILES}}
                        for key, values in merged.items()}}


def _file_bindings(manifest):
    entries = {"seed_plan": manifest["seed_plan"]}
    for split in data.SPLIT_NAMES:
        for name, entry in manifest["splits"][split]["files"].items():
            entries[f"{split}/{name}"] = entry
    if "source_corpus" in manifest:
        for name in ("manifest_snapshot", "seed_plan_snapshot"):
            entries[f"source_corpus/{name}"] = manifest["source_corpus"][name]
    return {key: {field: entry[field] for field in ("path", "sha256", "size_bytes")
                  if field in entry} for key, entry in entries.items()}


def audit_value_control_corpus(corpus, *, batch_size=8, tolerance=1e-5):
    """Scan every coefficient of all five splits, without modifying the corpus.

    Structural constraint failures return a complete ``passed=False`` report;
    integrity, configuration and nonfinite failures raise ``ValueError``. A
    caller MUST reject a failed report before any value-only target conversion;
    ``require_value_control_audit`` supplies that gate. No silent projection is
    performed here. ``batch_size`` is capped at eight complete trajectories.
    """
    batch_size = _integer(batch_size, "batch_size", 1)
    if batch_size > 8:
        raise ValueError("value-control audit batch_size must not exceed eight trajectories")
    if isinstance(tolerance, bool) or not math.isfinite(tolerance) or not 0 < tolerance <= 1e-3:
        raise ValueError("constraint tolerance must lie in (0, 1e-3]")
    root = Path(corpus).expanduser().resolve(strict=True)
    manifest_file_hash = data._file_sha256(root / "manifest.json")
    manifest = data.read_manifest(root, verify_hashes=True)
    layout = _layout(manifest)
    k, channels = layout["control_count"], layout["hermite_channels"]
    rho_dim = layout["structured_latent_layout"]["rho_latent_dim"]
    kinetic_start = rho_dim + layout["inactive_dimension"]
    reports, total_chunks = {}, 0
    for split in data.SPLIT_NAMES:
        files = manifest["splits"][split]["files"]
        coefficients = np.load(root / files["coefficients"]["path"], mmap_mode="r", allow_pickle=False)
        ids = np.load(root / files["case_ids"]["path"], mmap_mode="r", allow_pickle=False)
        conditions = np.load(root / files["conditions"]["path"], mmap_mode="r", allow_pickle=False)
        groups = {"rho": [], **{f"kinetic_channel_{index:03d}": [] for index in range(channels)}}
        inactive_failures, answer_hashes = [], []
        chunks = 0
        for start in range(0, len(coefficients), batch_size):
            stop = min(start + batch_size, len(coefficients))
            # Copy <=8 rows to avoid mutable mmap aliases during arithmetic.
            batch = np.array(coefficients[start:stop], copy=True)
            condition_batch = np.array(conditions[start:stop], dtype="<f4", copy=True)
            answer_hashes.extend(sha256(np.asarray(row, dtype="<f4").tobytes() + condition.tobytes()).hexdigest()
                                 for row, condition in zip(batch, condition_batch))
            if not np.isfinite(batch).all():
                raise ValueError(f"nonfinite coefficients in {split} rows {start}:{stop}")
            inactive_bad = np.any(batch[:, rho_dim:kinetic_start] != 0, axis=1)
            inactive_failures.extend(str(ids[start + index]) for index in np.flatnonzero(inactive_bad))
            rho = batch[:, :rho_dim].reshape(len(batch), 2, k, layout["rho_modes"])
            kinetic = batch[:, kinetic_start:].reshape(len(batch), channels, 2, k, layout["kinetic_modes"])
            groups["rho"].append(_metrics(rho[:, 0], rho[:, 1], layout["delta_tau"]))
            micro = _metrics(kinetic[:, :, 0], kinetic[:, :, 1], layout["delta_tau"])
            for index in range(channels):
                groups[f"kinetic_channel_{index:03d}"].append({key: value[:, index] for key, value in micro.items()})
            chunks += 1
        summaries = {name: _summarize(values, ids, tolerance) for name, values in groups.items()}
        failed = sorted(set(inactive_failures).union(*(value["failed_case_ids"] for value in summaries.values())))
        reports[split] = {"trajectory_count": len(ids), "chunks_scanned": chunks,
                          "input_answer_sha256": answer_hashes,
                          "passed": not bool(failed), "failed_case_ids": failed,
                          "failed_case_count": len(failed),
                          "inactive_nonzero_case_ids": inactive_failures, "groups": summaries}
        total_chunks += chunks
    # Includes ALL 25 arrays, seed plan, reuse snapshots and complete lineage.
    final_manifest = data.read_manifest(root, verify_hashes=True)
    if final_manifest != manifest:
        raise ValueError('corpus changed during the whole-corpus value-control audit')
    passed = all(value["passed"] for value in reports.values())
    report = {
        "schema": SCHEMA, "status": "passed" if passed else "rejected", "passed": passed,
        "encoding_constraint_verified": passed, "physical_fidelity_verified": False,
        "model_distribution_admitted": False, "corpus_modified": False,
        "target_projection_performed": False, "all_five_splits_fully_scanned": True,
        "sampled_rows_only": False, "source_path": str(root),
        "source_manifest_sha256": manifest["manifest_sha256"],
        "source_manifest_file_sha256": manifest_file_hash,
        "source_decoder_config_sha256": sha256(data._canonical_bytes(manifest["provenance"]["decoder_config"])).hexdigest(),
        "base_decoder_sha256": manifest.get("base_decoder_sha256"),
        "source_files": _file_bindings(manifest), "full_file_hashes_verified_before_and_after": True,
        "input_answer_hash_encoding": "sha256(little_endian_float32_full_coefficients || little_endian_float32_nu_temperature)",
        "layout": layout, "batch_size": batch_size, "chunks_scanned": total_chunks,
        "trajectory_count": sum(value["trajectory_count"] for value in reports.values()),
        "constraint": {"operator": "centered_interior_one_sided_endpoints_uniform_tau",
                       "stored_slope_units": "d/dtau; tau=(physical_t-t0)/(t1-t0)",
                       "gate": "both delta_tau*RMS(s-A*v)/RMS(v) and delta_tau*max_abs(s-A*v)/max_abs(v) <= tolerance, separately per case/rho/Hermite channel",
                       "tolerance": float(tolerance), "absolute_denominator_floor": _RMS_FLOOR,
                       "slope_relative_denominator_floor_fraction": _SLOPE_RELATIVE_FLOOR,
                       "statistics_arithmetic": "float64 on exact stored float32 values; companion float32 reconstruction"},
        "splits": reports,
        "limitations": ["Verifies the coarse projected data encoding, not high-fidelity physics.",
                        "A passed audit does not establish model quality or distributional equivalence.",
                        "Any failed constraint must reject value-only conversion, not silently project targets."],
    }
    # Explicit finite JSON contract, including empty failure lists.
    json.dumps(report, allow_nan=False)
    return report


def require_value_control_audit(report):
    """Reject a missing, partial or failed gate before value-only conversion."""
    if (not isinstance(report, Mapping) or report.get("schema") != SCHEMA
            or report.get("status") != "passed" or report.get("passed") is not True
            or report.get("all_five_splits_fully_scanned") is not True
            or report.get("full_file_hashes_verified_before_and_after") is not True
            or set(report.get("splits", {})) != set(data.SPLIT_NAMES)
            or not all(split.get("passed") is True for split in report["splits"].values())):
        raise ValueError("value-only training rejected: complete value-control constraint audit did not pass")
    return report
