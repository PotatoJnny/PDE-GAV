#!/usr/bin/env python3
"""Capture/check real frozen NS Base/P behavior without training or a solver run.

The harness pins a fixture-only Case-ID key: production sources deliberately
randomize that key, and P's random seed depends on its parent's Case ID. This
does not change production code, keys, checkpoints, or existing campaign IDs.
Arbitrary-time velocity below is a private truth query, not a solver capability.
"""
from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from collections.abc import Mapping

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "ns_real_representation_compatibility_v1"
CODE_FILES = (
    "legacy/ns/src/agent_verifier/neural_generator/structured_ns_answer_first.py",
    "legacy/ns/src/agent_verifier/neural_generator/structured_ns_analytic.py",
    "legacy/ns/src/agent_verifier/neural_generator/structured_ns_robustness.py",
    "legacy/ns/src/agent_verifier/neural_generator/structured_ns_distribution.py",
    "src/pde_agent_verifier_solver/plugins/ns/structured_flow_runtime.py",
    "src/pde_agent_verifier_solver/plugins/ns/continuous_backend.py",
    "src/pde_agent_verifier_solver/plugins/ns/canonical_backend.py",
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plain(value):
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {item.name: plain(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def specifications(workspace):
    config_path = workspace / "configs/ns_canonical_structured_5round.json"
    config = json.loads(config_path.read_text())["generator"]
    numerical = {
        "kind": "numerical", "generator_id": config["generator_id"],
        "generator_sha256": config["generator_sha256"],
        "runtime": {**config["runtime"], "device": "cpu", "cuda_graph_sampling": False},
        **{key: str((config_path.parent / config["artifacts"][key]).resolve())
           for key in ("flow_checkpoint", "pca_checkpoint")},
    }
    prepared = workspace / "project_runs/ns_measured_piv_generators_20260911_v1/cylinder/prepared"
    report = json.loads((prepared / "base_generator/train_report.json").read_text())
    binding = json.loads((prepared.parent / "artifact_binding_check.json").read_text())
    cylinder = {
        "kind": "measured_cylinder", "generator_id": "ns-experimental-mms",
        "dataset_id": "cylinder-measured-structured-generator",
        "prepared_dataset": str(prepared / "prepared_dataset.json"),
        "flow_checkpoint": report["final_checkpoint"],
        "pca_checkpoint": str(prepared / "pca_checkpoint.pt"),
        "runtime": {**binding["settings"], "device": "cpu", "cuda_graph_sampling": False},
    }
    return [numerical, cylinder]


def build(spec):
    from pde_agent_verifier_solver.plugins.ns.canonical_backend import build_canonical_structured_ns_generator
    if spec["kind"] == "numerical":
        bundle = build_canonical_structured_ns_generator(
            generator_id=spec["generator_id"], generator_sha256=spec["generator_sha256"],
            flow_checkpoint=spec["flow_checkpoint"], pca_checkpoint=spec["pca_checkpoint"],
            development_guidance_enabled=False, **spec["runtime"])
        return bundle, ()
    from pde_agent_verifier_solver.plugins.ns.experimental_mms_project import ExperimentalMMSProblem
    from pde_agent_verifier_solver.plugins.ns.experimental_mms import assemble_experimental_mms
    from pde_agent_verifier_solver.sdk import DatasetInput, DatasetKind
    dataset = ExperimentalMMSProblem().prepare_dataset(DatasetInput(
        spec["dataset_id"], DatasetKind.EXPERIMENTAL,
        location=spec["prepared_dataset"], settings={}))
    assembly = assemble_experimental_mms(
        dataset, generator_id=spec["generator_id"], flow_checkpoint=spec["flow_checkpoint"],
        pca_checkpoint=spec["pca_checkpoint"], generator_settings=spec["runtime"],
        development_guidance_enabled=False)
    return assembly.bundle, (dataset.development_cases[0],)


def request(kind, seed, anchors=()):
    from pde_agent_verifier_solver.sdk import (
        GenerationRequest, GeneratorMode, CaseVisibility, GenerationPurpose,
    )
    return GenerationRequest(round_index=0, count=1, mode=GeneratorMode.ONLINE,
        kind=kind, visibility=CaseVisibility.HIDDEN, purpose=GenerationPurpose.PROMOTION,
        seed=seed, anchors=anchors)


def inspect_record(bundle, record, prefix, arrays):
    import numpy as np
    answer = record.truth._answer
    field = answer.raw_answer
    session = bundle.backend.public_session_factory(record.case)
    left, right, bottom, top = field.domain_bounds
    t0, t1 = field.t0, field.t1
    normalized = np.array([[.05, .08], [.15, .85], [.88, .80], [.83, .15]])
    xy = np.array([left, bottom]) + normalized * np.array([right-left, top-bottom])
    points = np.column_stack((t0 + np.array([0., .2, .5, 1.]) * (t1-t0), xy))
    # Use the declared endpoints exactly; t0 + (t1 - t0) can exceed t1 by one ULP.
    points[0, 0], points[-1, 0] = t0, t1
    initial_points = np.column_stack((np.full(len(xy), t0), xy))
    boundary = np.array([[t0, left, (bottom+top)/2], [(t0+t1)/2, right, bottom],
                         [t1, (left+right)/2, top]])
    values = {
        "psi": field.coefficients.psi.detach().cpu().numpy(),
        "pressure": field.coefficients.pressure.detach().cpu().numpy(),
        "query_points_txy": points, "boundary_points_txy": boundary,
        "private_velocity": answer.evaluate_velocity(points),
        "public_forcing": session.query_forcing(points),
        "public_initial_velocity": session.query_initial_velocity(xy),
        "private_boundary_velocity": answer.evaluate_velocity(boundary),
    }
    np.testing.assert_allclose(values["public_initial_velocity"],
        answer.evaluate_velocity(initial_points), rtol=1e-12, atol=1e-13)
    if field.geometry_spec.outer_boundary == "manufactured_trace":
        values["public_boundary_velocity"] = session.query_boundary_velocity(boundary)
        np.testing.assert_allclose(values["public_boundary_velocity"],
            values["private_boundary_velocity"], rtol=1e-12, atol=1e-13)
    else:
        np.testing.assert_allclose(values["private_boundary_velocity"], 0., rtol=0., atol=1e-13)
    # One point exercises the independent original autograd field path too.
    reference = field.quantities(points[:1, (1, 2, 0)])
    for label, keys in (("private_velocity", ("u", "v")),
                        ("public_forcing", ("forcing_u", "forcing_v"))):
        ref = np.stack([reference[key].detach().cpu().numpy() for key in keys], axis=-1)
        np.testing.assert_allclose(np.asarray(values[label])[:1], ref, rtol=1e-10, atol=1e-11)
        values[f"autograd_{label}"] = ref
    for name, value in values.items():
        array = np.asarray(value)
        if not np.isfinite(array).all():
            raise AssertionError(f"nonfinite {prefix}/{name}")
        arrays[f"{prefix}/{name}"] = array.copy()
    public = json.dumps(plain(record.case.public_data))
    assert not any(token in public for token in ('"observations"', '"coefficients"', '"pod_basis"'))
    return {"case": plain(record.case), "projection_config": field.config.to_dict(),
            "geometry": field.geometry_spec.to_dict(), "mms_abi": field.continuous_mms_abi,
            "boundary_condition": plain(field.boundary_condition)}


def collect(specs):
    import numpy as np
    import torch
    from pde_agent_verifier_solver.sdk import GenerationKind
    torch.set_num_threads(1)
    arrays, records, retained = {}, {}, []
    for spec in specs:
        label = spec["kind"]
        print(json.dumps({"event": "load", "fixture": label}), flush=True)
        bundle, originals = build(spec)
        # A local test fixture, never a production campaign key or global patch.
        key = hashlib.sha256(f"{SCHEMA}:{label}:test-only-case-key".encode()).digest()
        bundle.backend.case_source._case_id_key = key
        generator = bundle.generator
        state = generator.load(bundle.checkpoint, {})
        base = generator.sample(state, request(GenerationKind.BASE, 20260913))[0]
        child = generator.sample(state, request(GenerationKind.PERTURBED, 20260914, (base.case,)))[0]
        assert child.case.parent_case_id == base.case.case_id
        assert child.case.public_data["perturbation"]["family"] == "spectral_complement"
        sampler = bundle.components.flow_sampler
        records[label] = {"pca_source_signature": plain(sampler.pca.source_signature),
            "projection_config": sampler._projection.to_dict(), "pca_rank": sampler.pca.rank,
            "base_state": plain(bundle.checkpoint.payload), "cases": {}}
        for role, record in [("base", base), ("perturbed", child)] + [("original", row) for row in originals]:
            records[label]["cases"][role] = inspect_record(bundle, record, f"{label}/{role}", arrays)
        a, b = base.truth._answer.raw_answer, child.truth._answer.raw_answer
        old_shape = (a.config.time_coefficient_count, a.config.spatial_mode_x+1, a.config.spatial_mode_y+1)
        new_shape = (b.config.time_coefficient_count, b.config.spatial_mode_x+1, b.config.spatial_mode_y+1)
        for name in ("psi", "pressure"):
            old = getattr(a.coefficients, name).detach().cpu().numpy().reshape(old_shape)
            new = getattr(b.coefficients, name).detach().cpu().numpy().reshape(new_shape)
            np.testing.assert_array_equal(old, new[:, :old_shape[1], :old_shape[2]])
        np.testing.assert_allclose(arrays[f"{label}/base/public_initial_velocity"],
            arrays[f"{label}/perturbed/public_initial_velocity"], rtol=1e-11, atol=1e-12)
        retained.append((label, bundle, base))
    # Alternating loaded geometries must not contaminate an earlier field.
    for label, bundle, base in retained:
        repeated = {}
        inspect_record(bundle, base, f"{label}/base", repeated)
        for key, value in repeated.items():
            np.testing.assert_array_equal(arrays[key], value)
    return records, arrays


def compare(expected, actual, reference_arrays, actual_arrays):
    import numpy as np
    if expected != actual:
        raise AssertionError("frozen configuration/signature/Case identity changed")
    if set(reference_arrays) != set(actual_arrays):
        raise AssertionError("captured query/array keys changed")
    for key, reference in reference_arrays.items():
        value = actual_arrays[key]
        if reference.dtype != value.dtype or reference.shape != value.shape:
            raise AssertionError(f"array dtype/shape changed: {key}")
        np.testing.assert_array_equal(reference, value, err_msg=key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("capture", "check"))
    parser.add_argument("--baseline-dir", type=Path, default=ROOT / "project_runs/ns_cylinder_representation_20260911_v2/compatibility")
    parser.add_argument("--code-root", type=Path, default=ROOT, help="Import implementation from this checkout/snapshot")
    parser.add_argument("--workspace-root", type=Path, default=ROOT, help="Locate old artifacts only when capturing")
    args = parser.parse_args()
    for directory in (args.code_root / "legacy/ns/src", args.code_root / "src"):
        sys.path.insert(0, str(directory.resolve()))
    import numpy as np
    import torch
    destination = args.baseline_dir.resolve()
    baseline_file, arrays_file = destination / "baseline.json", destination / "baseline_arrays.npz"
    started = time.perf_counter()
    if args.mode == "capture":
        if baseline_file.exists() or arrays_file.exists():
            raise FileExistsError("refusing to overwrite an existing compatibility baseline")
        specs = specifications(args.workspace_root.resolve())
        frozen = {spec[key]: sha(spec[key]) for spec in specs
                  for key in ("flow_checkpoint", "pca_checkpoint", "prepared_dataset") if key in spec}
    else:
        baseline = json.loads(baseline_file.read_text())
        if baseline["schema"] != SCHEMA:
            raise ValueError("unsupported compatibility baseline")
        specs, frozen = baseline["specifications"], baseline["artifact_sha256"]
        for path, digest in frozen.items():
            if sha(path) != digest:
                raise AssertionError(f"frozen artifact changed: {path}")
        if sha(arrays_file) != baseline["arrays_file_sha256"]:
            raise AssertionError("compatibility reference arrays changed")
    records, arrays = collect(specs)
    destination.mkdir(parents=True, exist_ok=True)
    report = {"schema": SCHEMA, "success": True, "mode": args.mode,
        "elapsed_seconds": time.perf_counter()-started, "array_count": len(arrays),
        "code_root": str(args.code_root.resolve()), "code_sha256": {
            name: sha(args.code_root / name) for name in CODE_FILES},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
            "torch": torch.__version__, "device": "cpu", "torch_num_threads": 1},
        "scope": "Two real old checkpoint pairs; one Base and one P each, fixed CPU 256-step Heun/raw. No training, Guidance, solver run, or distribution-quality evaluation.",
        "case_identity_note": "Fixture-only fixed HMAC key; these are reproducible harness Case IDs, not historical campaign IDs.",
        "checks": ["Flow/PCA binding", "frozen projection/signatures", "Case identity and parent",
            "finite velocity/forcing", "public initial/trace agrees with private answer",
            "analytic agrees with original autograd at one point", "P low-band and initial data unchanged",
            "alternating numerical/cylinder queries unchanged"]}
    if args.mode == "capture":
        np.savez_compressed(arrays_file, **arrays)
        write_json(baseline_file, {"schema": SCHEMA, "specifications": specs,
            "artifact_sha256": frozen, "records": records, "arrays_file_sha256": sha(arrays_file)})
    else:
        with np.load(arrays_file) as reference:
            compare(baseline["records"], records, reference, arrays)
        report["checks"].append("all identities and arrays exactly equal capture")
    write_json(destination / f"{args.mode}_report.json", report)
    print(json.dumps({"success": True, "report": str(destination / f"{args.mode}_report.json"),
                      "elapsed_seconds": report["elapsed_seconds"], "array_count": len(arrays)}), flush=True)


if __name__ == "__main__":
    main()
