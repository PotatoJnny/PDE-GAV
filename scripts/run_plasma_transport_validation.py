#!/usr/bin/env python3
"""One bounded compute-node pass: reference calibration and native v2 acceptance.

Never launches Agents, submits jobs, or cancels experiments. Separate child
processes prevent the physical packages' legacy import namespaces from mixing.
Every child log and failed stage survives; success only means engineering and
reference-calibration readiness, not solver accuracy or an experiment result.
"""
from __future__ import annotations

import argparse
import ast
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
REFERENCE_REUSE_EXCLUSIONS = {
    "scripts/run_plasma_transport_validation.py": "orchestration only; all native acceptances run again",
    "src/pde_agent_verifier_solver/plugins/vpfp/weak_portable_base.py":
        "loader consistency tolerance only; config/model asset hashes and checkpoint tensor loading remain exact",
}


# This opt-in scope covers the three reviewed startup/acceptance corrections.
# Calibration collection, fitting, model numerics, imports and all bytes outside
# these definitions remain identical to the original qualified source.
REFERENCE_RUNTIME_REPAIR_FUNCTIONS = {
    "src/pde_agent_verifier_solver/plugins/vpfp/canonical_binding.py":
        ("_BoundVPFPContinuousGenerator._check_settings",),
    "scripts/plasma_gpu_campaign.py": ("compose",),
    "scripts/check_plasma_transport.py": ("run_acceptance",),
}


def _runtime_repair_file(path, names):
    """Hash exact bytes outside explicitly named complete function definitions."""
    raw = Path(path).read_bytes()
    tree = ast.parse(raw, filename=str(path))
    found = {}

    def visit(body, scope=()):
        for node in body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = ".".join((*scope, node.name))
                if qualified in names and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if qualified in found or node.decorator_list:
                        raise ValueError("runtime repair scope is ambiguous or decorated: " + qualified)
                    found[qualified] = node
                visit(node.body, (*scope, node.name))

    visit(tree.body)
    if set(found) != set(names):
        raise ValueError("runtime repair function missing from " + str(path))
    lines = raw.splitlines(keepends=True)
    previous, chunks, functions = 0, [], {}
    for name, node in sorted(found.items(), key=lambda item: item[1].lineno):
        start, end = node.lineno - 1, node.end_lineno
        if start < previous:
            raise ValueError("runtime repair function scopes overlap")
        # Only standalone, undecorated definitions are accepted. Preserve the
        # entire header (parameters/annotations) and permit body edits only.
        body_start = node.body[0].lineno - 1
        if body_start <= start:
            raise ValueError("runtime repair needs a multiline function body")
        chunks.append(b"".join(lines[previous:body_start]))
        chunks.append(("# runtime-repair-body:" + name + "\n").encode())
        functions[name] = sha256(b"".join(lines[body_start:end])).hexdigest()
        previous = end
    chunks.append(b"".join(lines[previous:]))
    return sha256(b"".join(chunks)).hexdigest(), functions


def _science_files(root, *, runtime_repair=False):
    """Exact byte identity, with only the explicitly requested repair scope."""
    excluded = ({"scripts/run_plasma_transport_validation.py"} if runtime_repair
                else set(REFERENCE_REUSE_EXCLUSIONS))
    files = {}
    for folder in ("src", "scripts", "legacy"):
        for path in sorted((root / folder).rglob("*")):
            name = str(path.relative_to(root))
            if not path.is_file() or path.suffix not in {".py", ".sh", ".toml", ".json"} or name in excluded:
                continue
            files[name] = (_runtime_repair_file(path, REFERENCE_RUNTIME_REPAIR_FUNCTIONS[name])[0]
                           if runtime_repair and name in REFERENCE_RUNTIME_REPAIR_FUNCTIONS
                           else sha256(path.read_bytes()).hexdigest())
    if runtime_repair and not set(REFERENCE_RUNTIME_REPAIR_FUNCTIONS) <= files.keys():
        raise ValueError("runtime repair source is missing a required reviewed file")
    return files


def _reuse_references(reference_root, output_root, report, *, runtime_repair=False):
    """Copy immutable references only; independent fit rechecks every seed/asset."""
    from pde_agent_verifier_solver.engine.checkpoint_store import source_identity
    from scripts.calibrate_plasma_transport import COUNTS, _sha
    reference_root = Path(reference_root).resolve(strict=True)
    original_path = reference_root / "validation.json"
    original = json.loads(original_path.read_text())
    source = Path(original["source_root"]).resolve(strict=True)
    required = tuple("collect-" + family for family in COUNTS) + ("fit",)
    stages = original.get("stages", ())[:4]
    if (original.get("schema") != report["schema"]
            or tuple(row.get("name") for row in stages) != required
            or any(row.get("status") != "passed" or row.get("exit_code") != 0 for row in stages)
            or original.get("reference_config_hashes") != report["reference_config_hashes"]):
        raise ValueError("reference reuse requires four passed calibration stages and identical configs")
    if original.get("source_identity") != source_identity(source):
        raise ValueError("original frozen reference source changed")
    files = _science_files(ROOT, runtime_repair=runtime_repair)
    if not files or files != _science_files(source, runtime_repair=runtime_repair):
        raise ValueError("reference scientific source changed beyond the explicit reuse scope")
    calibration = reference_root / "calibration.json"
    frozen = json.loads(calibration.read_text())
    if (calibration.is_symlink() or _sha(calibration) != original.get("calibration_sha256")
            or frozen.get("objective_hash") != original.get("objective_hash")):
        raise ValueError("original frozen calibration bytes or objective changed")
    copied = {}
    for family, count in COUNTS.items():
        directory, target = reference_root / family, output_root / family
        manifest_path = directory / "collection.json"
        manifest = json.loads(manifest_path.read_text())
        if (manifest_path.is_symlink() or manifest.get("family") != family
                or manifest.get("status") != "passed"
                or manifest.get("config_sha256") != report["reference_config_hashes"][family]
                or manifest.get("collection_script_sha256") != _sha(source / "scripts/calibrate_plasma_transport.py")
                or len(manifest.get("records", ())) != count):
            raise ValueError("original collection manifest or configuration changed")
        target.mkdir(mode=0o700, exist_ok=False)
        shutil.copy2(manifest_path, target / "collection.json")
        for index, row in enumerate(manifest["records"]):
            name = f"reference-{index:03d}.npz"
            asset = directory / name
            if row.get("asset") != name or asset.is_symlink() or _sha(asset) != row.get("asset_sha256"):
                raise ValueError("original private reference asset changed")
            shutil.copy2(asset, target / name)
        copied[family] = _sha(manifest_path)
        report["stages"].append({"name": "collect-" + family, "status": "passed", "exit_code": 0,
            "execution": "reused_reference", "source_manifest": str(manifest_path),
            "source_manifest_sha256": copied[family], "fresh_draws": 0})
    report["reference_reuse"] = {"original_root": str(reference_root),
        "original_validation_sha256": _sha(original_path), "original_status": original.get("status"),
        "original_source_identity": original["source_identity"],
        "original_calibration_sha256": _sha(calibration), "collection_manifest_sha256": copied,
        "scientific_files_sha256": sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        "scientific_file_count": len(files), "exceptions": {name: {
            "reason": reason, "original_sha256": _sha(source / name), "current_sha256": _sha(ROOT / name)}
            for name, reason in REFERENCE_REUSE_EXCLUSIONS.items()
            if not runtime_repair or name == "scripts/run_plasma_transport_validation.py"},
        "reference_draws_recomputed": False, "all_native_acceptances_recomputed": True}
    if runtime_repair:
        report["reference_reuse"]["runtime_repair_scope"] = {
            "mode": "reviewed_startup_functions_v1",
            "unchanged_outside_listed_function_bodies": True,
            "calibration_collection_and_fit_source_unchanged": True,
            "files": {name: {
                "functions": list(names), "original_sha256": _sha(source / name),
                "current_sha256": _sha(ROOT / name),
                "unchanged_region_sha256": files[name],
                "original_function_body_sha256": _runtime_repair_file(source / name, names)[1],
                "current_function_body_sha256": _runtime_repair_file(ROOT / name, names)[1],
            } for name, names in REFERENCE_RUNTIME_REPAIR_FUNCTIONS.items()},
        }
    return _sha(calibration)


def run(source_configs, output_root, *, reference_validation_root=None, reference_runtime_repair=False):
    if not os.environ.get("SLURM_JOB_ID", "").isdigit():
        raise RuntimeError("transport validation must run in a Slurm compute allocation")
    from pde_agent_verifier_solver.engine.checkpoint_store import atomic_json, source_identity
    if reference_runtime_repair and reference_validation_root is None:
        raise ValueError("runtime repair reuse requires an original reference validation root")
    source_configs, output_root = Path(source_configs).resolve(), Path(output_root).resolve()
    configs = {family: source_configs / f"{family}-sol.json" for family in ("strong", "weak", "artemis")}
    for family, path in configs.items():
        value = json.loads(path.read_text())
        if value.get("equation") != family or value.get("schema") != "pde_gav_plasma_gpu_v1":
            raise ValueError("expected the three actual frozen reference configurations")
    output_root.mkdir(parents=True, mode=0o700, exist_ok=False)
    report = {"schema": "plasma_transport_v2_validation_pipeline_v1", "status": "running",
              "slurm_job_id": os.environ["SLURM_JOB_ID"], "started_at_unix": time(),
              "source_root": str(ROOT), "source_identity": source_identity(ROOT),
              "source_revision": os.environ.get("PLASMA_VALIDATION_SOURCE_REVISION"),
              "stages": [], "formal_jobs_submitted": False,
              "old_jobs_modified": False, "solver_accuracy_claim": False,
              "reference_config_hashes": {family: sha256(path.read_bytes()).hexdigest()
                                         for family, path in configs.items()}}
    destination = output_root / "validation.json"

    def stage(name, arguments):
        row = {"name": name, "status": "running", "started_at_unix": time(), "arguments": arguments}
        report["stages"].append(row)
        atomic_json(destination, report)
        with (output_root / f"{name}.out").open("x") as out, (output_root / f"{name}.err").open("x") as err:
            result = subprocess.run([sys.executable, *arguments], cwd=ROOT, stdout=out, stderr=err, check=False)
        row.update(exit_code=result.returncode, finished_at_unix=time(),
                   status="passed" if result.returncode == 0 else "failed")
        atomic_json(destination, report)
        if result.returncode:
            raise RuntimeError(f"stage {name} failed with exit {result.returncode}; see retained stdout/stderr")

    try:
        original_calibration_sha = None
        if reference_validation_root is not None:
            original_calibration_sha = _reuse_references(reference_validation_root, output_root, report,
                                                        runtime_repair=reference_runtime_repair)
        else:
            for family, path in configs.items():
                stage(f"collect-{family}", ["scripts/calibrate_plasma_transport.py", "collect",
                      "--family", family, "--config", str(path), "--output", str(output_root / family)])
        calibration = output_root / "calibration.json"
        inputs = [argument for family in configs for argument in ("--input", str(output_root / family))]
        stage("fit", ["scripts/calibrate_plasma_transport.py", "fit", *inputs, "--output", str(calibration)])
        if original_calibration_sha is not None and sha256(calibration.read_bytes()).hexdigest() != original_calibration_sha:
            raise ValueError("independent refit differs from the original frozen calibration")
        frozen = json.loads(calibration.read_text())
        report.update(objective_hash=frozen["objective_hash"], floors=frozen["floors"],
                      calibration_sha256=sha256(calibration.read_bytes()).hexdigest())
        for family, path in configs.items():
            target = output_root / f"acceptance-{family}"
            stage(f"acceptance-{family}", ["scripts/check_plasma_transport.py", "--config", str(path),
                  "--calibration", str(calibration), "--output-root", str(target), "--point-count", "32"])
            receipt = json.loads((target / "report.json").read_text())
            if (receipt.get("status") != "passed" or receipt.get("private_cases_released") is not True
                    or receipt.get("objective_identity", {}).get("objective_hash") != frozen["objective_hash"]):
                raise ValueError(f"{family} acceptance receipt did not bind the frozen objective and clean up")
        report["status"] = "passed"
        return report
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
        raise
    finally:
        report["finished_at_unix"] = time()
        atomic_json(destination, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-configs", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reference-validation-root", type=Path,
                        help="Explicit loader-only retry: verify/reuse prior references; rerun fit and all acceptances")
    parser.add_argument("--reference-runtime-repair", action="store_true",
                        help="Explicit reviewed startup repair: reuse exact references only when all source bytes outside three named function bodies remain unchanged; refit and rerun all acceptances")
    args = parser.parse_args()
    report = run(args.source_configs, args.output_root, reference_validation_root=args.reference_validation_root,
                 reference_runtime_repair=args.reference_runtime_repair)
    print(json.dumps({key: report[key] for key in ("status", "objective_hash", "floors")}))


if __name__ == "__main__":
    main()
