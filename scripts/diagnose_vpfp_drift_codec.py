"""Stage-0 local velocity-drift codec audit; no training or scientific admission.

Decode train/validation answers at fixed (t,x,y) points, calculate each state's
own physical drift, project its translated Hermite AMPLITUDE into a centered
basis, then translate/project back to the complete original Hermite ABI.
The intermediate order never drops an original mode. This does not implement
a spatial/Fourier/time codec, does not generate new answers, and does not read
calibration/reference arrays. Real-answer execution requires Slurm.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
from hashlib import sha256
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SCHEMA = "vpfp_local_velocity_drift_codec_diagnostic_v1"
TIME_FRACTIONS = (0., .140625, .515625, 1.)
SPATIAL_FRACTIONS = ((.113, .227), (.379, .619), (.643, .431), (.827, .883))
DEGREE_OFFSETS = (0, 4, 8)


def _json_bytes(value):
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()


def _file_sha(path):
    result = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def mode_pairs(degree):
    if isinstance(degree, bool) or not isinstance(degree, (int, np.integer)) or degree < 0:
        raise ValueError("Hermite degree must be a nonnegative integer")
    return tuple((nx, total-nx) for total in range(int(degree)+1) for nx in range(total+1))


def _normalized_polynomials(points, degree):
    """H_n(x)/sqrt(2**n*n!); the Gaussian is integrated by Gauss-Hermite."""
    x = np.asarray(points, dtype=np.float64)
    result = np.empty((len(x), degree+1), dtype=np.float64)
    result[:, 0] = 1.
    if degree:
        result[:, 1] = math.sqrt(2.)*x
    for n in range(1, degree):
        result[:, n+1] = math.sqrt(2./(n+1))*x*result[:, n] - math.sqrt(n/(n+1))*result[:, n-1]
    return result


@lru_cache(maxsize=16)
def _quadrature(order):
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or not 2 <= order <= 96:
        raise ValueError("bounded Gauss-Hermite order must be an integer in [2,96]")
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


def translation_overlap(source_degree, target_degree, shift, temperature, quadrature_order):
    """Return S(d)[m,n]=integral phi_m(v)*phi_n(v+d) dv, in float64.

    Completing the Gaussian square moves the quadrature center by -d/2.
    Its polynomial integrand has degree <= source_degree+target_degree in
    each coordinate: this is an overlap integral, not finite-grid least squares
    and not exp(projected_translation_generator). It is not generally unitary.
    """
    source, target = mode_pairs(source_degree), mode_pairs(target_degree)
    d = np.asarray(shift, dtype=np.float64)
    theta = float(temperature)
    if d.shape != (2,) or not np.isfinite(d).all() or not math.isfinite(theta) or theta <= 0:
        raise ValueError("finite two-axis shift and positive temperature required")
    nodes, weights = _quadrature(quadrature_order)
    if 2*quadrature_order-1 < source_degree+target_degree:
        raise ValueError("quadrature does not integrate the declared complete Hermite overlap degree")
    matrices = []
    for delta in d / math.sqrt(2*theta):
        left = _normalized_polynomials(nodes-delta/2, target_degree)
        right = _normalized_polynomials(nodes+delta/2, source_degree)
        matrices.append(math.exp(-delta*delta/4)/math.sqrt(math.pi)*(left.T @ (weights[:, None]*right)))
    target_indices, source_indices = np.asarray(target), np.asarray(source)
    result = (matrices[0][target_indices[:, 0, None], source_indices[None, :, 0]]
        * matrices[1][target_indices[:, 1, None], source_indices[None, :, 1]])
    if not np.isfinite(result).all():
        raise FloatingPointError("Hermite overlap overflow/nonfinite")
    return result


def amplitude_moments(coefficients, temperature, *, rho=1., floor_mixture=0.):
    """Exact moments of the original squared-Hermite + zero-center floor ABI.

    v*a is accumulated in degree N+1, retaining boundary excursions required
    for exact second moments. Never square a truncated N-by-N velocity matrix.
    """
    a = np.asarray(coefficients, dtype=np.float64)
    degree = int((math.sqrt(8*len(a)+1)-3)/2) if a.ndim == 1 else -1
    modes = mode_pairs(degree)
    theta, density, floor = float(temperature), float(rho), float(floor_mixture)
    if (a.shape != (len(modes),) or not np.isfinite(a).all()
            or not all(math.isfinite(x) for x in (theta, density, floor))
            or theta <= 0 or density <= 0 or not 0 <= floor < 1):
        raise ValueError("finite complete total-degree amplitude and physical conditions required")
    norm2 = float(a @ a)
    if not math.isfinite(norm2) or norm2 <= np.finfo(np.float64).tiny:
        raise ValueError("zero/underflow amplitude norm; no silent ground-state substitution")
    extended = mode_pairs(degree+1)
    index = {pair: i for i, pair in enumerate(extended)}
    embedded = np.zeros(len(extended))
    embedded[:len(a)] = a
    raised = np.zeros((2, len(extended)))
    for n, pair in enumerate(modes):
        for axis in (0, 1):
            upper = list(pair); upper[axis] += 1
            raised[axis, index[tuple(upper)]] += math.sqrt(theta*(pair[axis]+1))*a[n]
            if pair[axis]:
                lower = list(pair); lower[axis] -= 1
                raised[axis, index[tuple(lower)]] += math.sqrt(theta*pair[axis])*a[n]
    velocity = (1-floor)*(raised @ embedded)/norm2
    second = (1-floor)*(raised @ raised.T)/norm2 + floor*theta*np.eye(2)
    return {"amplitude_norm_squared": norm2, "velocity": velocity.tolist(),
        "momentum": (density*velocity).tolist(), "raw_second_moment": second.tolist(),
        "central_second_moment": (second-np.outer(velocity, velocity)).tolist(),
        "kinetic_energy": float(.5*density*np.trace(second)),
        "highest_degree_energy_fraction": float(np.sum(a[np.array([sum(pair)==degree for pair in modes])]**2)/norm2),
        "highest_three_degrees_energy_fraction": float(np.sum(a[np.array([sum(pair)>=max(0,degree-2) for pair in modes])]**2)/norm2)}


def quadrature_moments(coefficients, temperature, order, *, rho=1., floor_mixture=0.):
    """Independent tensor velocity quadrature; no clipped velocity window."""
    a = np.asarray(coefficients, dtype=np.float64)
    exact = amplitude_moments(a, temperature, rho=rho, floor_mixture=floor_mixture)
    degree = int((math.sqrt(8*len(a)+1)-3)/2)
    if order < degree+2:
        raise ValueError("moment quadrature must retain degree-extended second moments")
    nodes, weights = _quadrature(order)
    polynomials = _normalized_polynomials(nodes, degree)
    field = np.zeros((order, order))
    for coefficient, (nx, ny) in zip(a, mode_pairs(degree)):
        field += coefficient*np.outer(polynomials[:, nx], polynomials[:, ny])
    mass = ((1-floor_mixture)*field**2/exact["amplitude_norm_squared"]+floor_mixture)
    mass *= np.outer(weights, weights)/math.pi
    vx, vy = np.meshgrid(math.sqrt(2*temperature)*nodes, math.sqrt(2*temperature)*nodes, indexing="ij")
    return {"mass": float(mass.sum()), "momentum": [float(rho*(mass*vx).sum()), float(rho*(mass*vy).sum())],
        "kinetic_energy": float(.5*rho*(mass*(vx*vx+vy*vy)).sum())}


def roundtrip(coefficients, temperature, *, rho=1., floor_mixture=0., intermediate_degree=None,
        quadrature_order=None, zero_shift=False):
    a = np.asarray(coefficients, dtype=np.float64)
    original = amplitude_moments(a, temperature, rho=rho, floor_mixture=floor_mixture)
    degree = int((math.sqrt(8*len(a)+1)-3)/2)
    middle = degree if intermediate_degree is None else intermediate_degree
    mode_pairs(middle)
    if middle < degree:
        raise ValueError("intermediate order may not truncate any original Hermite mode")
    q = middle+2 if quadrature_order is None else quadrature_order
    drift = np.zeros(2) if zero_shift else np.array(original["velocity"])
    center = translation_overlap(degree, middle, drift, temperature, q) @ a
    restored = translation_overlap(middle, degree, -drift, temperature, q) @ center
    centered = amplitude_moments(center, temperature, rho=rho, floor_mixture=floor_mixture)
    returned = amplitude_moments(restored, temperature, rho=rho, floor_mixture=floor_mixture)
    an, rn = math.sqrt(original["amplitude_norm_squared"]), math.sqrt(returned["amplitude_norm_squared"])
    fraction_above_original = float(np.sum(center[np.array([sum(pair)>degree for pair in mode_pairs(middle)])]**2)
        / centered["amplitude_norm_squared"])
    # The original floor stays M_T(v), not M_T(v+u); separate that known ABI
    # residual from finite-Hermite projection error in the centered mean.
    ideal_center_velocity = np.array(original["velocity"])-(1-floor_mixture)*drift
    q_returned = quadrature_moments(restored, temperature, max(q, degree+2), rho=rho, floor_mixture=floor_mixture)
    return {"source_degree": degree, "intermediate_degree": int(middle), "return_degree": degree,
        "quadrature_order_per_velocity_axis": int(q), "zero_shift_control": bool(zero_shift),
        "shift_from_same_state_physical_velocity": drift.tolist(), "original": original,
        "centered": centered, "returned": returned,
        "centered_norm_retained_fraction": centered["amplitude_norm_squared"]/original["amplitude_norm_squared"],
        "returned_norm_retained_fraction": returned["amplitude_norm_squared"]/original["amplitude_norm_squared"],
        "centered_energy_above_original_degree_fraction": fraction_above_original,
        "raw_amplitude_relative_l2_error": float(np.linalg.norm(restored-a)/an),
        "normalized_amplitude_relative_l2_error": float(np.linalg.norm(restored/rn-a/an)),
        "momentum_error": (np.array(returned["momentum"])-original["momentum"]).tolist(),
        "kinetic_energy_error": returned["kinetic_energy"]-original["kinetic_energy"],
        "highest_degree_energy_fraction_error": returned["highest_degree_energy_fraction"]-original["highest_degree_energy_fraction"],
        "ideal_center_velocity_with_unchanged_floor": ideal_center_velocity.tolist(),
        "center_velocity_error_beyond_known_floor": (np.array(centered["velocity"])-ideal_center_velocity).tolist(),
        "velocity_quadrature": q_returned,
        "velocity_quadrature_mass_error": q_returned["mass"]-1.,
        "velocity_quadrature_momentum_error": (np.array(q_returned["momentum"])-returned["momentum"]).tolist(),
        "velocity_quadrature_energy_error": q_returned["kinetic_energy"]-returned["kinetic_energy"]}


def _checked_path(root, entry):
    relative = Path(entry["path"])
    path = root / relative
    if relative.is_absolute() or ".." in relative.parts or path.is_symlink() or not path.is_file():
        raise ValueError("corpus entries must be regular relative files inside the source")
    if not path.resolve().is_relative_to(root):
        raise ValueError('corpus file is outside its root')
    return path


def load_development(corpus_root):
    """Open only train/validation arrays; do not mmap the other split headers."""
    root = Path(corpus_root).resolve(strict=True)
    path = root / "manifest.json"
    if path.is_symlink():
        raise ValueError("manifest must not be a symlink")
    manifest = json.loads(path.read_text())
    digest = manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = digest
    if manifest["schema"] != "vpfp_stratified_coefficient_corpus_v1" or manifest["status"] != "complete":
        raise ValueError("completed VPFP corpus required")
    plan_path = _checked_path(root, manifest["seed_plan"])
    plan = json.loads(plan_path.read_text())
    if (plan["bin_edges"] != manifest["bin_edges"] or plan["regime_id"] != manifest["regime_id"]
            or plan["split_counts"] != manifest["split_counts"]):
        raise ValueError("corpus seed-plan identity differs")
    frozen = {str(path): _file_sha(path), str(plan_path): _file_sha(plan_path)}
    splits, seen = {}, set()
    for role in ("train", "validation"):
        record = manifest["splits"][role]
        count = manifest["split_counts"][role]
        if (record["status"] != "complete" or record["rows_written"] != count or record["count"] != count
                or set(record["files"]) != {"coefficients", "conditions", "equation_seeds", "case_ids", "stratum_ids"}):
            raise ValueError("incomplete development split")
        arrays = {}
        for key, entry in record["files"].items():
            checked = _checked_path(root, entry)
            frozen[str(checked)] = entry["sha256"]
            value = np.load(checked, mmap_mode="r", allow_pickle=False)
            expected = ((count, manifest["latent_dimension"]) if key == "coefficients" else
                (count, 2) if key == "conditions" else (count,))
            kind = "f" if key in ("coefficients", "conditions") else "U" if key == "case_ids" else "i"
            if (value.shape != expected or list(value.shape) != entry["shape"] or value.dtype.str != entry["dtype"]
                    or value.dtype.kind != kind or (kind == "f" and value.dtype != np.float32)):
                raise ValueError("development array shape/dtype mismatch")
            arrays[key] = value
        rows = plan["splits"][role]
        ids = list(map(int, arrays["equation_seeds"]))
        if (len(rows) != count or len(set(ids)) != count or seen.intersection(ids) or min(ids) < 0
                or ids != [row["equation_seed"] for row in rows]
                or not np.array_equal(arrays["conditions"], np.array([row["conditions"] for row in rows], dtype=np.float32))
                or not np.array_equal(arrays["case_ids"], [row["case_id"] for row in rows])
                or not np.array_equal(arrays["stratum_ids"], [row["stratum_id"] for row in rows])):
            raise ValueError("development split seed/condition lineage mismatch")
        seen.update(ids)
        splits[role] = arrays
    return manifest, splits, frozen


def select_rows(split, edges):
    from pde_agent_verifier_solver.plugins.vpfp.diffusion_audit import _strata
    if tuple(map(len, edges)) != (5, 3):
        raise ValueError("diagnostic requires the existing eight joint condition strata")
    bins = _strata(np.asarray(split["conditions"]), tuple(np.asarray(edge) for edge in edges))
    if not np.array_equal(bins, split["stratum_ids"]):
        raise ValueError("declared development strata differ from physical conditions")
    selected = []
    for group in range(8):
        candidates = np.flatnonzero(bins == group)
        if not len(candidates):
            raise ValueError(f"no development case in condition stratum {group}")
        selected.append(int(candidates[np.argmin(split["equation_seeds"][candidates])]))
    return selected


def build_decoder(config_path, manifest, device):
    from pde_agent_verifier_solver.plugins.vpfp.canonical_binding import VPFPOnlineModelSpec
    from pde_agent_verifier_solver.plugins.vpfp.perturbation_backend import _legacy_api
    config = json.loads(Path(config_path).read_text())
    spec = VPFPOnlineModelSpec.from_mapping(config.get("vpfp", config))
    if (_json_bytes(as_plain(spec)) != _json_bytes(manifest["model_spec"])
            or spec.base_decoder_sha256 != manifest["base_decoder_sha256"]):
        raise ValueError("requested decoder/corpus specification differs")
    spec.preflight()
    api = _legacy_api()
    hermite = importlib.import_module("agent_verifier.vpfp.continuous_hermite")
    payload = api.torch.load(spec.base_decoder_checkpoint, map_location="cpu", weights_only=True)
    if _json_bytes(payload["decoder_config"]) != _json_bytes(manifest["provenance"]["decoder_config"]):
        raise ValueError("decoder config differs from source corpus provenance")
    cfg = hermite.VPFPContinuousHermiteConfig.from_dict(payload["decoder_config"])
    if not cfg.structured_direct_coefficients or cfg.structured_residual_scale != 0:
        raise ValueError("diagnostic requires the original direct zero-residual amplitude decoder")
    return api.build_deterministic_direct_model(cfg, device=device)


def as_plain(spec):
    from dataclasses import asdict
    return asdict(spec)


def local_amplitudes(decoder, latent, condition):
    """Use actual decoder fields: original scales, normalization and fallback."""
    torch, cfg = decoder.torch, decoder.config
    times = cfg.time_domain[0]+np.array(TIME_FRACTIONS)*(cfg.time_domain[1]-cfg.time_domain[0])
    points = np.array([[t, x*cfg.equation.x_domain[0], y*cfg.equation.x_domain[1], 0., 0.]
        for t in times for x, y in SPATIAL_FRACTIONS])
    raw = np.array(latent, dtype=np.float32, copy=True)
    conditions = np.asarray(condition, dtype=np.float32)
    if raw.shape != (cfg.latent_dim,) or conditions.shape != (2,) or not np.isfinite(raw).all() or not np.isfinite(conditions).all():
        raise ValueError("nonfinite or invalid development answer snapshot")
    with torch.no_grad():
        fields = decoder.fields(points, raw, equation_condition=conditions)
    a = fields["hermite_coefficients"].detach().cpu().numpy().astype(np.float64)
    rho = fields["rho"].detach().cpu().numpy().astype(np.float64)
    if (a.shape != (len(points), len(mode_pairs(cfg.hermite_total_degree))) or rho.shape != (len(points),)
            or not np.isfinite(a).all() or not np.isfinite(rho).all() or np.any(rho <= 0)):
        raise ValueError("decoder returned invalid local amplitudes/density")
    return points, a, rho, sha256(raw.tobytes(order="C")).hexdigest()


def summarize(cases):
    groups = {}
    for case in cases:
        for row in case.get("rows", []):
            for result in row.get("roundtrips", []):
                key = f"{case['split']}/{'zero' if result['zero_shift_control'] else 'self_drift'}/degree-{result['intermediate_degree']}/q-{result['quadrature_order_per_velocity_axis']}"
                groups.setdefault(key, []).append(result)
    output = {}
    for key, rows in groups.items():
        out = {"local_point_count": len(rows)}
        for metric in ("raw_amplitude_relative_l2_error", "normalized_amplitude_relative_l2_error",
                "centered_norm_retained_fraction", "returned_norm_retained_fraction",
                "centered_energy_above_original_degree_fraction", "kinetic_energy_error",
                "highest_degree_energy_fraction_error", "velocity_quadrature_mass_error",
                "velocity_quadrature_energy_error"):
            x = np.array([row[metric] for row in rows])
            out[metric] = {"mean": float(x.mean()), "median": float(np.median(x)),
                "p95_absolute": float(np.quantile(abs(x), .95)), "maximum_absolute": float(abs(x).max())}
        for axis, name in enumerate(("jx", "jy")):
            x = np.array([row["momentum_error"][axis] for row in rows])
            out[name+"_error"] = {"rms": float(np.sqrt(np.mean(x*x))), "maximum_absolute": float(abs(x).max())}
        output[key] = out
    return output


def run(corpus_root, config_path, output, device="cuda"):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("real-answer drift codec diagnostics require a Slurm compute allocation")
    corpus, config, output = Path(corpus_root).resolve(), Path(config_path).resolve(), Path(output)
    if output.exists() or output.is_symlink() or output.resolve().is_relative_to(corpus):
        raise ValueError("choose a fresh diagnostic output outside the immutable corpus")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started, stage, cases, failures = time.monotonic(), "validate_development_source", [], []
    try:
        manifest, splits, frozen = load_development(corpus)
        frozen[str(config)] = _file_sha(config)
        frozen[str(Path(__file__).resolve())] = _file_sha(__file__)
        selection = {role: select_rows(split, manifest["bin_edges"]) for role, split in splits.items()}
        plan = {"schema": SCHEMA, "corpus_root": str(corpus), "config_path": str(config), "device": device,
            "source_manifest_sha256": manifest["manifest_sha256"], "source_sha256": frozen,
            "selected_rows": selection, "selected_equation_seeds": {role: [int(splits[role]["equation_seeds"][i]) for i in rows]
                for role, rows in selection.items()}, "time_fractions": TIME_FRACTIONS,
            "spatial_fractions": SPATIAL_FRACTIONS, "intermediate_degree_offsets": DEGREE_OFFSETS,
            "quadrature_orders": "intermediate_degree+2 and intermediate_degree+10 per velocity axis",
            "no_original_Hermite_mode_discarded": True, "shift": "same_state_physical_mean_velocity",
            "velocity_floor_policy": "retain_original_zero_center_Maxwellian_floor; report its residual separately",
            "normalization": "no_renormalization_between_projections; report_raw_norm_and_decoder_normalized_moments",
            "physical_reference_arrays_read": False, "new_generated_answers": 0, "new_pde_solves": 0,
            "spatial_or_time_codec_implemented": False, "codec_qualified": False,
            "scientific_admission": False, "population_statistics": False}
        _write(output / "frozen-plan.json", plan)
        stage = "build_original_decoder"
        decoder = build_decoder(config, manifest, device)
        cfg = decoder.config
        if cfg.hermite_total_degree+max(DEGREE_OFFSETS)+10 > 96:
            raise ValueError("decoder order exceeds bounded stage-0 quadrature budget")
        for role, indices in selection.items():
            split = splits[role]
            for index in indices:
                stage = f"local_roundtrip_{role}_row-{index}"
                case = {"split": role, "row_index": index, "equation_seed": int(split["equation_seeds"][index]),
                    "condition": split["conditions"][index].tolist(), "stratum_id": int(split["stratum_ids"][index]), "rows": []}
                cases.append(case)
                try:
                    points, coefficients, rho, digest = local_amplitudes(decoder, split["coefficients"][index], split["conditions"][index])
                    case["source_float32_coefficient_row_sha256"] = digest
                    for point, a, density in zip(points, coefficients, rho):
                        row = {"txy": point[:3].tolist(), "roundtrips": []}
                        case["rows"].append(row)
                        for zero, offsets in ((True, (0,)), (False, DEGREE_OFFSETS)):
                            for offset in offsets:
                                middle = cfg.hermite_total_degree+offset
                                values = []
                                for order in (middle+2, middle+10):
                                    result = roundtrip(a, case["condition"][1], rho=density,
                                        floor_mixture=cfg.velocity_floor_mixture, intermediate_degree=middle,
                                        quadrature_order=order, zero_shift=zero)
                                    row["roundtrips"].append(result)
                                    values.append(result)
                                values[1]["quadrature_refinement_absolute_changes"] = {key: abs(values[1][key]-values[0][key])
                                    for key in ("raw_amplitude_relative_l2_error", "returned_norm_retained_fraction", "kinetic_energy_error")}
                    case["execution_passed"] = True
                except Exception as error:
                    case.update(execution_passed=False, error=repr(error), traceback=traceback.format_exc())
                    failures.append({"stage": stage, "error": repr(error)})
                _write(output / "partial-cases.json", {"schema": SCHEMA, "cases": cases, "failures": failures,
                    "partial_artifacts_preserved": True})
        stage = "verify_frozen_inputs"
        result = {"schema": SCHEMA, "execution_passed": not failures, "cases": cases, "failures": failures,
            "summary": summarize(cases), "wall_seconds": time.monotonic()-started,
            "selected_case_count": len(cases), "local_point_count": sum(len(case["rows"]) for case in cases),
            "physical_reference_arrays_read": False, "new_generated_answers": 0, "new_pde_solves": 0,
            "spatial_or_time_codec_implemented": False, "decoder_ABI_changed": False, "codec_qualified": False,
            "admitted": False, "population_statistics": False, "partial_artifacts_preserved": True,
            "limitations": ["Selected development answers, not generated samples or a distribution audit.",
                "Higher intermediate order does not establish an invertible free generative latent codec.",
                "Pointwise velocity reconstruction says nothing about reprojecting nonlinear shifted fields into fixed Fourier/time support.",
                "The original stationary Maxwellian floor is not translated; its known centering residual is separated from truncation.",
                "No physical failure is silently removed; all case failures remain in the report."]}
    except Exception as error:
        result = {"schema": SCHEMA, "execution_passed": False, "failed_stage": stage,
            "error": repr(error), "traceback": traceback.format_exc(), "cases": cases, "failures": failures,
            "partial_artifacts_preserved": True, "admitted": False, "codec_qualified": False}
    _write(output / "result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    result = run(args.corpus, args.config, args.output, args.device)
    print(json.dumps({"schema": SCHEMA, "execution_passed": result["execution_passed"], "output": str(args.output)}), flush=True)
    return 0 if result["execution_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
