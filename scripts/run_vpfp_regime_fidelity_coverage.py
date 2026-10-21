"""Measure frozen VPFP answers: physical scales and reference morphology coverage.

No generation, training, PDE solve, acceptance threshold tuning, or automatic
admission. The old distribution audit is neither changed nor replaced. Real
extraction requires a compute allocation even on CPU. Paired m/z artifacts are
evaluated directly as A and separately as B, against the same calibration bank.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from pde_agent_verifier_solver.plugins.vpfp.canonical_binding import VPFPOnlineModelSpec
from pde_agent_verifier_solver.plugins.vpfp.diffusion_data import load_split, read_manifest, _source_exclusion_seeds
from pde_agent_verifier_solver.plugins.vpfp.diffusion_audit import _edges, _strata
from pde_agent_verifier_solver.plugins.vpfp.regime_fidelity_coverage import (
    VPFPRegimeFeaturePlan, audit_regime_fidelity_coverage, extract_vpfp_regime_features, sample_from_extraction)
from pde_agent_verifier_solver.plugins.vpfp.structured_diffusion import load_vpfp_diffusion_sampler
from pde_agent_verifier_solver.plugins.vpfp.initial_conditioned_diffusion import CHECKPOINT_SCHEMA as INITIAL_CHECKPOINT_SCHEMA
from diagnose_vpfp_known_initial import _runtime
from train_vpfp_structured_diffusion import balanced_conditions

SCHEMA = "vpfp_frozen_regime_fidelity_coverage_run_v3"


def file_sha(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected regular frozen input file: {path}")
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n"
    temporary = path.with_name(path.name+".partial")
    with temporary.open("x") as stream:
        stream.write(payload)
    temporary.replace(path)


def check_frozen(files):
    pass


def _audit_index(result):
    return {"schema": "vpfp_regime_fidelity_coverage_index_v1",
        "execution_passed": result["execution_passed"], "admitted": False,
        "physical_regime_qualified": False, "confirmation_512_completed": False,
        "model_source_manifest_sha256": result.get("model_source_manifest_sha256"),
        "reference_source_manifest_sha256": result.get("reference_source_manifest_sha256"),
        "stage": result["stage"], "failure": result.get("failure"),
        "targets": {label: {"directory": row["directory"], "status": row["status"],
            "execution_passed": row["status"] == "descriptive_audit_completed", "admitted": False,
            "A_audit_path": row.get("A_audit_path"),
            "B_audit_path": row.get("B_audit_path", row.get("audit_path")),
            "paired_comparison_path": row.get("paired_comparison_path")}
            for label, row in result["targets"].items()}}


def _publish_audit_result(output, result):
    # Save the directory result FIRST so an index I/O failure cannot erase the
    # scientific/execution failure already recorded by the run.
    write_json(output/"result.json", result)
    try:
        write_json(output/"regime-fidelity-coverage.json", _audit_index(result))
    except Exception as error:
        if result["execution_passed"]:
            result["stage"] = "publish_audit_index_failed"
        result["execution_passed"] = False
        result["index_publication_failure"] = {"type": type(error).__name__, "message": str(error)}
        try:
            write_json(output/"result.json", result)
        except Exception:
            pass  # The first result and any .partial are retained if storage failed.
        raise


def select_rows(conditions, count, edges):
    """Select by fixed condition stratum and original row order, never features."""
    if count not in (128, 512):
        raise ValueError("registered extraction counts are exactly 128 or 512, without automatic reduction")
    bins = _strata(conditions, edges)
    groups = [np.flatnonzero(bins == group)[:count//8] for group in range(8)]
    if any(len(group) != count//8 for group in groups):
        raise ValueError(f"source lacks {count//8} frozen attempts in every condition stratum")
    return np.concatenate(groups)


def _regular_directory(path):
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"regular existing directory required: {path}")
    return path.resolve(strict=True)


def _input_hash(raw, condition):
    return sha256(np.asarray(raw, dtype="<f4").tobytes()+np.asarray(condition, dtype="<f4").tobytes()).hexdigest()


def resolve_corpus(path):
    root = _regular_directory(path)
    return root if (root/"manifest.json").is_file() else _regular_directory(root/"corpus")


def corpus_ancestry(source, manifest):
    path = source/manifest["seed_plan"]["path"]
    plan = json.loads(path.read_text())
    return set(_source_exclusion_seeds(plan)), plan


def validate_source_pair(model_source, manifest, reference_source, reference_manifest):
    """Reference roots are fresh; identical training bytes may be reused."""
    for key in ("model_spec", "base_decoder_sha256", "latent_dimension", "regime_id"):
        if not _same_json_value(manifest[key], reference_manifest[key]):
            raise ValueError(f"reference and model-source physical ABI differ: {key}")
    old, _ = corpus_ancestry(model_source, manifest)
    all_reference, ref_plan = corpus_ancestry(reference_source, reference_manifest)
    if model_source != reference_source:
        roots = [row["equation_seed"] for role in ("calibration", "reference_a", "reference_b")
                 for row in ref_plan["splits"][role]]
        if len(set(roots)) != len(roots) or old.intersection(roots):
            raise ValueError("new reference evaluation roots overlap model-source or historical roots")
    return old | all_reference


def _independent_lineage(provenance, count):
    if "model_source_manifest_sha256" not in provenance:
        return []
    result = set()
    for key in ("seeds", "condition_seeds", "future_noise_seeds", "known_initial_seeds",
                "macro_noise_seeds", "micro_noise_seeds"):
        if key not in provenance: continue
        values = provenance[key]
        if (not isinstance(values, list) or len(values) != count or len(set(values)) != count
                or any(type(v) is not int or not 0 <= v < 2**63 for v in values)):
            raise ValueError(f"invalid independent generation ancestry: {key}")
        result.update(values)
    if not {"seeds", "condition_seeds", "future_noise_seeds"} <= set(provenance):
        raise ValueError("independent generation omits ancestry streams")
    return sorted(result)


def _native_stream(tag, seeds):
    """Exact existing sampler SHA tags; no new sampling or RNG calls."""
    return [int.from_bytes(sha256(f"{tag}:{seed}".encode()).digest()[:8], "little") % 2**63 for seed in seeds]


def _validate_independent_attempts(provenance, raw, *, selected_count, macro=None):
    """Cross-check all attempted rows, including failures; legacy stays unchanged."""
    if "model_source_manifest_sha256" not in provenance:
        return
    if (provenance.get("schema") != "vpfp_independent_native_generation_v1"
            and not (provenance.get("schema") == "vpfp_coherent_generated_native_pair_v1"
                     and provenance.get("decoder_views") == ["B_old_full_decoder", "A_continuous_moving_frame"]
                     and provenance.get("full_coefficients_are_A_answer") is False)):
        raise ValueError("independent source marker requires the native512 or complete paired512 receipt")
    count = len(raw)
    fixed = {"attempted": 512, "completed_attempts": 512, "sampler_steps": 64}
    if (count != 512 or selected_count != 512 or any(type(provenance.get(key)) is not int
            or provenance[key] != value for key, value in fixed.items())):
        raise ValueError("independent generation must retain all 512 attempts and sampler_steps=64")
    _independent_lineage(provenance, count)
    failures = provenance.get("failures")
    if not isinstance(failures, list):
        raise ValueError("independent generation needs an explicit failure ledger")
    indices = []
    for failure in failures:
        if not isinstance(failure, dict):
            raise ValueError("independent failure entries must identify attempted rows")
        index = failure.get("attempt")
        if (type(index) is not int or not 0 <= index < count or index in indices
                or failure.get("seed") != provenance["seeds"][index]
                or type(failure.get("seed")) is not int):
            raise ValueError("independent failure index/seed is invalid or repeated")
        indices.append(index)
    for key, expected in (("failed_attempts", len(indices)), ("successful_attempts", count-len(indices))):
        if type(provenance.get(key)) is not int or provenance[key] != expected:
            raise ValueError("independent failure/success counts differ from the failure ledger")
    failed = np.zeros(count, dtype=bool); failed[indices] = True
    arrays = [("raw", raw)] if macro is None else [("raw", raw), ("macro", macro)]
    for name, array in arrays:
        if array.ndim != 2 or len(array) != count:
            raise ValueError("independent raw/macro attempted-row shapes differ")
        # Bounded reads also apply when the frozen array has 650k columns.
        for start in range(0, count, 8):
            block, mask = array[start:start+8], failed[start:start+8]
            if not np.array_equal(np.isnan(block).all(1), mask) or not np.array_equal(np.isfinite(block).all(1), ~mask):
                raise ValueError(f"independent {name} NaN failure rows differ from the failure ledger")


def _validate_sampling_streams(provenance, *, kind):
    if "model_source_manifest_sha256" not in provenance:
        return
    seeds = provenance["seeds"]
    if kind == "initial":
        expected = {"known_initial_seeds": _native_stream("vpfp-initial-conditioned-prior", seeds),
                    "future_noise_seeds": _native_stream("vpfp-initial-conditioned-future", seeds)}
    elif kind == "pair":
        expected = {"macro_noise_seeds": _native_stream("vpfp-coherent-macro", seeds),
                    "micro_noise_seeds": _native_stream("vpfp-coherent-micro", seeds)}
        expected["future_noise_seeds"] = expected["micro_noise_seeds"]
    else:
        expected = {"future_noise_seeds": seeds}
    if kind != "initial" and "known_initial_seeds" in provenance:
        raise ValueError("non-initial sampler receipt unexpectedly claims known-initial ancestry")
    if any(provenance.get(key) != values for key, values in expected.items()):
        raise ValueError("independent known-initial/macro/future sampling streams do not replay native SHA derivation")


def _freeze_independent_native(provenance, manifest, spec, frozen, count):
    if (count != 512 or provenance.get("attempted") != count or provenance.get("completed_attempts") != count
            or provenance.get("sampler_steps") != 64
            or provenance.get("model_source_manifest_sha256") != manifest["manifest_sha256"]):
        raise ValueError("independent generation must bind all 512 attempts and the training source")
    item = provenance.get("checkpoint", {})
    path, digest = item.get("path"), item.get("sha256")
    if path is None:
        raise ValueError('independent native checkpoint differs from receipt')
    frozen[str(Path(path).absolute())] = digest
    # Reuse the real native loader: schema, model/config/normalizer agreement,
    # immutable buffers and complete provenance are checked. Never sample here.
    sampler = load_vpfp_diffusion_sampler(path, digest, device="cpu", steps=64, churn=0.)
    identity = sampler.identity
    if (identity.get("base_decoder_sha256") != spec.base_decoder_sha256
            or identity.get("checkpoint_sha256") != digest
            or identity.get("corpus_manifest_sha256") != manifest["manifest_sha256"]
            or identity.get("latent_dimension") != manifest["latent_dimension"]
            or any(not _same_json_value(identity.get(key), getattr(spec, key))
                   for key in ("collision_nu_range", "temperature_range", "time_domain"))):
        raise ValueError("loaded independent native model/config does not bind the model-source ABI")
    _validate_sampling_streams(provenance, kind="initial" if identity.get("checkpoint_schema") == INITIAL_CHECKPOINT_SCHEMA else "direct")
    del sampler
    return digest


def _same_json_value(left, right):
    """Native tuple-valued configs become lists in JSON generation receipts."""
    options = dict(sort_keys=True, separators=(",", ":"), allow_nan=False)
    return json.dumps(left, **options) == json.dumps(right, **options)


def _freeze_pair(directory, provenance, raw, conditions, selected, frozen, *, manifest, spec, seeds):
    if (provenance.get("schema") != "vpfp_coherent_generated_native_pair_v1"
            or provenance.get("decoder_views") != ["B_old_full_decoder", "A_continuous_moving_frame"]
            or provenance.get("full_coefficients_are_A_answer") is not False
            or provenance.get("conditions_sha256") != frozen[str(directory/"generated-conditions.npy")]):
        raise ValueError("paired A audit requires a complete versioned m/z generation receipt")
    from pde_agent_verifier_solver.plugins.vpfp.moving_frame_binding import validate_hierarchical_payloads
    macro_path = directory/"generated-macro.npy"
    frozen[str(macro_path)] = file_sha(macro_path)
    if provenance.get("macro_sha256") != frozen[str(macro_path)]: raise ValueError("A macro file differs from receipt")
    macro = np.load(macro_path, mmap_mode="r", allow_pickle=False)
    payloads, bindings = {}, {}
    for role in ("macro", "conditional"):
        item = provenance.get("checkpoint_bindings", {}).get(role, {})
        path, digest = item.get("checkpoint"), item.get("checkpoint_sha256")
        if path is None:
            raise ValueError(f'paired {role} checkpoint differs from frozen receipt')
        frozen[str(Path(path).absolute())] = digest
        payloads[role] = torch.load(path, map_location="cpu", weights_only=True)
        bindings[role] = {"checkpoint": str(Path(path).absolute()), "checkpoint_sha256": digest}
    macro_cfg, value_cfg = validate_hierarchical_payloads(payloads["macro"], payloads["conditional"],
        base_decoder_sha256=spec.base_decoder_sha256, macro_checkpoint_sha256=bindings["macro"]["checkpoint_sha256"])
    if (payloads["macro"]["corpus_manifest_sha256"] != manifest["manifest_sha256"]
            or not _same_json_value(provenance.get("macro_config"), macro_cfg.to_dict())
            or not _same_json_value(provenance.get("micro_config"), value_cfg.to_dict())
            or value_cfg.full_dimension != manifest["latent_dimension"]
            or macro.shape != (len(raw), macro_cfg.latent_dimension) or macro.dtype != np.dtype("float32")):
        raise ValueError("A carrier config/shape or hierarchical source corpus mismatch")
    _validate_independent_attempts(provenance, raw, selected_count=len(selected), macro=macro)
    _validate_sampling_streams(provenance, kind="pair")
    pair_sha = sha256(json.dumps(bindings, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    hashes = tuple(_input_hash(raw[i], conditions[i]) for i in selected)
    return {"directory": str(directory), "files": frozen, "selected_rows": selected.tolist(),
        "status": "ready_paired_A_B", "raw": raw, "macro": macro, "conditions": conditions,
        "roots": tuple(f"generated-pair:{pair_sha}:{seed}" for seed in seeds), "selected_hashes": hashes,
        "checkpoint_sha256": None, "checkpoint_pair_sha256": pair_sha, "checkpoint_bindings": bindings,
        "macro_config": macro_cfg.to_dict(), "micro_config": value_cfg.to_dict(),
        "condition_seed_replayed": "condition_seed" in provenance,
        "conditions_file_sha_bound": True, "recorded_generation_failures": provenance.get("failures", []),
        "selected_same_draw_noise_seeds": [seeds[i] for i in selected],
        "MMS_full_velocity_quadrature_completed": False, "physical_calibration_completed": False}


def freeze_generated(directory, *, manifest, spec, count, edges):
    """Resolve only the explicit generated target; no recursive checkpoint search."""
    directory = _regular_directory(directory)
    paths = {name: directory/name for name in
        ("generated-coefficients.npy", "generated-conditions.npy", "generated-provenance.json")}
    frozen = {str(path): file_sha(path) for path in paths.values()}
    provenance = json.loads(paths["generated-provenance.json"].read_text())
    expected = provenance.get("coefficient_sha256", provenance.get("coefficients_sha256"))
    if expected != frozen[str(paths["generated-coefficients.npy"])]:
        raise ValueError("native coefficient file differs from generation receipt")
    if "conditions_sha256" in provenance and provenance["conditions_sha256"] != frozen[str(paths["generated-conditions.npy"])]:
        raise ValueError("native conditions file differs from generation receipt")
    raw = np.load(paths["generated-coefficients.npy"], mmap_mode="r", allow_pickle=False)
    conditions = np.load(paths["generated-conditions.npy"], mmap_mode="r", allow_pickle=False)
    if (raw.ndim != 2 or raw.shape[1] != manifest["latent_dimension"] or conditions.shape != (len(raw), 2)
            or raw.dtype != np.dtype("float32") or conditions.dtype != np.dtype("float32")):
        raise ValueError("frozen generated native array shape/dtype differs from corpus ABI")
    design = provenance.get("design", {})
    if not isinstance(design, dict): raise ValueError("generated design must be an object")
    seeds = provenance.get("noise_seeds", provenance.get("seeds", design.get("sampling_seeds")))
    if (not isinstance(seeds, list) or len(seeds) != len(raw) or len(set(seeds)) != len(seeds)
            or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in seeds)):
        raise ValueError("one unique nonnegative recorded sampling seed per attempted answer required")
    if "condition_seed" in provenance:
        condition_seed = provenance["condition_seed"]
        if isinstance(condition_seed, bool) or not isinstance(condition_seed, int) or condition_seed < 0:
            raise ValueError("invalid condition seed")
        if not np.array_equal(conditions, balanced_conditions(spec, len(raw), condition_seed)):
            raise ValueError("conditions do not replay the frozen balanced condition seed")
    elif "conditions" in design:
        recorded = np.asarray(design["conditions"], dtype=np.float32)
        if not np.array_equal(conditions, recorded):
            raise ValueError("conditions do not replay the frozen known-initial design")
    elif "conditions_sha256" not in provenance:
        raise ValueError("generated conditions need replay seed or explicit file SHA")
    selected = select_rows(conditions, count, edges)
    if "model_source_manifest_sha256" in provenance and seeds != provenance.get("seeds"):
        raise ValueError("independent sampling seed aliases differ from canonical attempted seeds")
    if "model_source_manifest_sha256" in provenance and not np.isfinite(conditions).all():
        raise ValueError("independent attempts require finite conditions even when generation fails")
    _validate_independent_attempts(provenance, raw, selected_count=len(selected))
    lineage = _independent_lineage(provenance, len(raw))
    if "model_source_manifest_sha256" in provenance and provenance["model_source_manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("generation training source differs from declared model source")
    # The paired carrier is NOT projected back into this old full ABI. The
    # A decoder and B decoder are both measured under explicit same-draw IDs.
    macro = directory/"generated-macro.npy"
    if macro.exists():
        pair = _freeze_pair(directory, provenance, raw, conditions, selected, frozen,
            manifest=manifest, spec=spec, seeds=seeds)
        pair["independent_lineage_seeds"] = lineage
        return pair
    old_hashes = None
    if provenance.get("schema") == "vpfp_independent_native_generation_v1":
        checkpoint_sha = _freeze_independent_native(provenance, manifest, spec, frozen, len(raw))
        roots = tuple(f"generated:{checkpoint_sha}:{seed}" for seed in seeds)
    elif "checkpoint_sha256" in provenance:
        checkpoint_sha = provenance["checkpoint_sha256"]
        physical_path, audit_path = directory/"generated-physical.json", directory/"physical-distribution-audit.json"
        for path in (physical_path, audit_path): frozen[str(path)] = file_sha(path)
        physical, old_audit = json.loads(physical_path.read_text()), json.loads(audit_path.read_text())
        if (old_audit.get("checkpoint_sha256") != checkpoint_sha
                or old_audit.get("corpus_manifest_sha256") != manifest["manifest_sha256"]):
            raise ValueError("old physical audit does not bind this checkpoint and corpus")
        roots = tuple(f"generated:{checkpoint_sha}:{seed}" for seed in seeds)
        if tuple(physical.get("root_ids", ())) != roots:
            raise ValueError("generated root IDs do not match recorded checkpoint and noise seeds")
        old_hashes = physical.get("answer_sha256")
        if not isinstance(old_hashes, list) or len(old_hashes) != len(raw):
            raise ValueError("old physical audit lacks per-answer input hashes")
    else:
        if provenance.get("schema") != "vpfp_coherent_generated_native_pair_v1" or provenance.get("decoder_views") != ["old_full_decoder"]:
            raise ValueError("unsupported frozen generation receipt; explicit decoder identity required")
        binding = provenance.get("checkpoint_bindings", {}).get("W0", {})
        checkpoint_sha = binding.get("checkpoint_sha256")
        checkpoint_path = binding.get("checkpoint")
        if checkpoint_path is None:
            raise ValueError('W0 checkpoint file does not match receipt')
        frozen[str(Path(checkpoint_path).absolute())] = checkpoint_sha
        roots = tuple(f"generated:{checkpoint_sha}:{seed}" for seed in seeds)
    if not isinstance(checkpoint_sha, str) or len(checkpoint_sha) != 64 or any(c not in "0123456789abcdef" for c in checkpoint_sha):
        raise ValueError("invalid checkpoint SHA256")
    hashes = tuple(_input_hash(raw[i], conditions[i]) for i in selected)
    if old_hashes is not None and hashes != tuple(old_hashes[i] for i in selected):
        raise ValueError("native answers do not match old physical audit input hashes")
    initial_seeds = (provenance.get("known_initial_seeds") if provenance.get("schema") == "vpfp_independent_native_generation_v1"
        else design.get("known_initial_seeds") if provenance.get("known_initial_seed_applied") is True else None)
    if initial_seeds is not None and (not isinstance(initial_seeds, list) or len(initial_seeds) != len(raw)
            or len(set(initial_seeds)) != len(initial_seeds) or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in initial_seeds)):
        raise ValueError("known-initial generation needs unique integer ancestry per attempted answer")
    if provenance.get("known_initial_seed_applied") is True and initial_seeds is None:
        raise ValueError("known-initial generation receipt omits initial root ancestry")
    return {"directory": str(directory), "files": frozen, "selected_rows": selected.tolist(),
        "status": "ready_native_direct", "raw": raw, "conditions": conditions,
        "roots": roots, "selected_hashes": hashes, "checkpoint_sha256": checkpoint_sha,
        "condition_seed_replayed": "condition_seed" in provenance,
        "conditions_recorded_design_replayed": "conditions" in design,
        "known_initial_seeds": initial_seeds,
        "independent_lineage_seeds": lineage,
        "conditions_file_sha_bound": True, "recorded_generation_failures": provenance.get("failures", [])}


def run(arguments):
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("real regime extraction requires an existing compute allocation, including CPU")
    source = resolve_corpus(getattr(arguments, "model_source", None) or arguments.source)
    reference_source = resolve_corpus(getattr(arguments, "reference_source", None) or source)
    targets = [_regular_directory(path) for path in arguments.generated_dir]
    if len(set(targets)) != len(targets): raise ValueError("duplicate generated target directories")
    output = Path(arguments.output).absolute()
    if output.exists() or output.is_symlink(): raise ValueError("output must be new; no overwrite or implicit resume")
    output = output.resolve()
    if Path("/scratch") not in output.parents:
        raise ValueError("real audit output must be a fresh /scratch run directory")
    if any(root == output or root in output.parents or output in root.parents for root in [source, reference_source, *targets]):
        raise ValueError("output must not overlap frozen inputs")
    output.mkdir(parents=True, exist_ok=False)
    result = {"schema": SCHEMA, "execution_passed": False, "admitted": False,
        "physical_regime_qualified": False, "confirmation_512_completed": False,
        "sample_count": arguments.sample_count, "pde_solves": 0, "new_generated_answers": 0,
        "new_training": False, "stage": "freeze_inputs",
        "targets": {f"target-{index:02d}": {"directory": str(path), "status": "not_started"}
                    for index, path in enumerate(targets)},
        "evidence_role": ("new_reference_descriptive_diagnostics_not_refined_physical_confirmation" if source != reference_source
                          else "previously_opened_development_answers_not_blind_confirmation")}
    frozen = {}
    try:
        manifest = read_manifest(source, verify_hashes=True)
        reference_manifest = read_manifest(reference_source, verify_hashes=True) if reference_source != source else manifest
        result.update(model_source_manifest_sha256=manifest["manifest_sha256"],
            reference_source_manifest_sha256=reference_manifest["manifest_sha256"])
        all_source_seeds = validate_source_pair(source, manifest, reference_source, reference_manifest)
        spec = VPFPOnlineModelSpec.from_mapping(manifest["model_spec"])
        spec.preflight()
        edges = _edges(np.linspace(*spec.collision_nu_range, 5), np.linspace(*spec.temperature_range, 3))
        frozen[str(source/"manifest.json")] = file_sha(source/"manifest.json")
        frozen[str(reference_source/"manifest.json")] = file_sha(reference_source/"manifest.json")
        frozen[str(Path(__file__).absolute())] = file_sha(__file__)
        import pde_agent_verifier_solver.plugins.vpfp.regime_fidelity_coverage as module
        frozen[str(Path(module.__file__).absolute())] = file_sha(module.__file__)
        import pde_agent_verifier_solver.plugins.vpfp.moving_frame_audit as moving_audit
        # Bind precisely the A decoder/measurement graph; no sampler is called.
        for name in ("moving_frame_audit.py", "moving_frame.py", "moving_frame_backend.py", "moving_frame_binding.py", "value_control_diffusion.py"):
            path = Path(moving_audit.__file__).with_name(name)
            frozen[str(path.absolute())] = file_sha(path)
        frozen_targets = [freeze_generated(directory, manifest=manifest, spec=spec, count=arguments.sample_count, edges=edges)
                          for directory in targets]
        for target in frozen_targets: frozen.update(target["files"])
        for target in frozen_targets:
            if all_source_seeds.intersection(target.get("independent_lineage_seeds", [])):
                raise ValueError("independent generated lineage overlaps training, reference or historical roots")
        runtime = _runtime(manifest, spec, arguments.device)
        frozen.update(runtime.frozen_files)
        for label in ("base_decoder", "expanded_decoder"):
            frozen[getattr(spec, label+"_checkpoint")] = getattr(spec, label+"_sha256")
        plan = VPFPRegimeFeaturePlan()
        reference_splits = {role: load_split(reference_source, role, verify_hashes=False)
                            for role in ("calibration", "reference_a", "reference_b")}
        reference_rows = {role: select_rows(split.conditions, arguments.sample_count, edges)
                          for role, split in reference_splits.items()}
        ancestry = [t["known_initial_seeds"] for t in frozen_targets if t.get("known_initial_seeds") is not None]
        if ancestry:
            for seeds in ancestry:
                if all_source_seeds.intersection(seeds):
                    raise ValueError("known-initial generated ancestry overlaps source training or evaluation roots")
        write_json(output/"frozen-plan.json", {"schema": SCHEMA, "source": str(source),
            "source_manifest_sha256": manifest["manifest_sha256"], "files": frozen,
            "model_source": str(source), "reference_source": str(reference_source),
            "model_source_manifest_sha256": manifest["manifest_sha256"],
            "reference_source_manifest_sha256": reference_manifest["manifest_sha256"],
            "plan": asdict(plan), "reference_rows": {k: v.tolist() for k, v in reference_rows.items()},
            "generated_targets": [{k: v for k, v in t.items() if k not in ("raw", "macro", "conditions")} for t in frozen_targets],
            "selection": "condition_stratum_then_original_order_no_feature_selection",
            "sample_count": arguments.sample_count, "confirmation_512_completed": False,
            "physical_acceptance_thresholds": None})
        cached, schema = {}, None
        measured_reports = {}
        def extract(label, raw, conditions, roots, expected_hashes=None, *, moving_decoder=None, macro=None):
            result["stage"] = "extract_"+label; write_json(output/"result.json", result)
            def progress(done, count, case):
                with (output/"progress.jsonl").open("a") as stream:
                    stream.write(json.dumps({"stage": result["stage"], "done": done, "total": count,
                        "hard_valid": case["hard_valid"], "failure": case.get("failure")}, allow_nan=False)+"\n")
            if moving_decoder is None:
                matrix, descriptors, report = extract_vpfp_regime_features(runtime.model, raw, conditions, plan=plan, progress=progress)
            else:
                matrix, descriptors, report = moving_audit.extract_moving_frame_regime_features(moving_decoder,
                    raw, macro, conditions, plan=plan, progress=progress, expected_identity=moving_decoder.identity())
            measured_reports[label] = report
            sample = sample_from_extraction(matrix, conditions, roots, report)
            if expected_hashes is not None and sample.answer_sha256 != tuple(expected_hashes):
                raise ValueError("extraction did not preserve the frozen per-answer identity")
            np.save(output/(label+"-features.npy"), matrix, allow_pickle=False)
            write_json(output/(label+"-measurements.json"), report)
            return sample, descriptors
        for role, split in reference_splits.items():
            rows = reference_rows[role]
            cached[role], current = extract(role, split.latents[rows], split.conditions[rows], tuple(str(split.case_ids[i]) for i in rows))
            if schema is not None and current != schema: raise ValueError("reference descriptor schema changed")
            schema = current
        write_json(output/"descriptors.json", schema)
        for index, target in enumerate(frozen_targets):
            label = f"target-{index:02d}"
            if target["status"] not in ("ready_native_direct", "ready_paired_A_B"):
                result["targets"][label] = {k: v for k, v in target.items() if k != "files"}
                continue
            rows = target["selected_rows"]
            paired = target["status"] == "ready_paired_A_B"
            b_label = label+"-B" if paired else label
            sample, current = extract(b_label, target["raw"][rows], target["conditions"][rows],
                tuple(target["roots"][i] for i in rows), target["selected_hashes"])
            if current != schema: raise ValueError("generated descriptor schema differs from references")
            result["stage"] = "audit_"+label; write_json(output/"result.json", result)
            audit = audit_regime_fidelity_coverage({**cached, "generated": sample}, descriptors=schema,
                nu_edges=edges[0], temperature_edges=edges[1], bootstrap_repeats=arguments.bootstrap_repeats)
            audit.update(source_manifest_sha256=manifest["manifest_sha256"], checkpoint_sha256=target["checkpoint_sha256"],
                model_source_manifest_sha256=manifest["manifest_sha256"], reference_source_manifest_sha256=reference_manifest["manifest_sha256"],
                generated_directory=target["directory"], prior_generation_failures=target["recorded_generation_failures"],
                original_attempt_count=len(target["raw"]), selected_attempt_count=len(rows))
            if paired:
                audit.update(view="B_old_full_decoder_same_joint_draw", checkpoint_pair_sha256=target["checkpoint_pair_sha256"],
                    checkpoint_bindings=target["checkpoint_bindings"], A_answer_report=False)
            write_json(output/(b_label+"-audit.json"), audit)
            result["targets"][label] = {"directory": target["directory"],
                "status": "B_completed_A_pending" if paired else "descriptive_audit_completed",
                "audit_path": str(output/(b_label+"-audit.json")), "admitted": False,
                "attempted_count": audit["cohorts"]["generated"]["attempted_count"],
                "failed_count": audit["cohorts"]["generated"]["failed_count"]}
            if paired:
                from pde_agent_verifier_solver.plugins.vpfp.coherent_diffusion import VPFPCoherentMacroConfig
                from pde_agent_verifier_solver.plugins.vpfp.value_control_diffusion import VPFPValueEDMConfig
                from pde_agent_verifier_solver.plugins.vpfp.moving_frame import MovingFrameDecoder, MovingFrameMacroConfig
                mc, vc = VPFPCoherentMacroConfig.from_dict(target["macro_config"]), VPFPValueEDMConfig.from_dict(target["micro_config"])
                a_decoder = MovingFrameDecoder(runtime.model, MovingFrameMacroConfig(control_count=mc.control_count,
                    max_mode=mc.maximum_mode, time_domain=mc.time_domain,
                    spatial_domain_lengths=runtime.model.config.equation.x_domain), value_config=vc)
                a_label = label+"-A"
                identity = a_decoder.identity()
                expected_a_hashes = tuple(moving_audit.carrier_input_sha256(target["raw"][i], target["macro"][i],
                    target["conditions"][i], identity) for i in rows)
                a_sample, a_schema = extract(a_label, target["raw"][rows], target["conditions"][rows],
                    tuple("A:"+target["roots"][i] for i in rows), expected_a_hashes,
                    moving_decoder=a_decoder, macro=target["macro"][rows])
                if a_schema != schema: raise ValueError("A physical descriptors differ from the frozen reference geometry")
                a_audit = audit_regime_fidelity_coverage({**cached, "generated": a_sample}, descriptors=schema,
                    nu_edges=edges[0], temperature_edges=edges[1], bootstrap_repeats=arguments.bootstrap_repeats)
                if a_audit["geometry"] != audit["geometry"]:
                    raise ValueError("A/B audit did not preserve identical calibration-only feature geometry")
                a_audit.update(source_manifest_sha256=manifest["manifest_sha256"], checkpoint_pair_sha256=target["checkpoint_pair_sha256"],
                    model_source_manifest_sha256=manifest["manifest_sha256"], reference_source_manifest_sha256=reference_manifest["manifest_sha256"],
                    generated_directory=target["directory"], decoder_identity=identity, view="A_continuous_moving_frame",
                    checkpoint_bindings=target["checkpoint_bindings"], prior_generation_failures=target["recorded_generation_failures"],
                    original_attempt_count=len(target["raw"]), selected_attempt_count=len(rows))
                write_json(output/(a_label+"-audit.json"), a_audit)
                paired_report = moving_audit.paired_decoder_comparison(a_sample.features, measured_reports[a_label],
                    sample.features, measured_reports[b_label], schema)
                paired_report.update(checkpoint_pair_sha256=target["checkpoint_pair_sha256"],
                    selected_rows=rows, sampling_seeds=target["selected_same_draw_noise_seeds"],
                    calibration_geometry_fit_sha256=a_audit["geometry"]["fit_identity_sha256"])
                write_json(output/(label+"-paired-decoder-comparison.json"), paired_report)
                result["targets"][label].update(status="descriptive_audit_completed", A_audit_path=str(output/(a_label+"-audit.json")),
                    B_audit_path=str(output/(b_label+"-audit.json")), A_failed_count=a_audit["cohorts"]["generated"]["failed_count"],
                    A_measured=True, B_measured=True, z_only_report_substituted_for_A=False,
                    paired_comparison_path=str(output/(label+"-paired-decoder-comparison.json")))
        result["stage"] = "recheck_frozen_inputs"; write_json(output/"result.json", result)
        check_frozen(frozen)
        final = read_manifest(source, verify_hashes=True)
        if final["manifest_sha256"] != manifest["manifest_sha256"]: raise ValueError("corpus changed during audit")
        final_reference = read_manifest(reference_source, verify_hashes=True) if reference_source != source else final
        if final_reference["manifest_sha256"] != reference_manifest["manifest_sha256"]:
            raise ValueError("reference corpus changed during audit")
        result.update(execution_passed=True, stage="complete", source_manifest_sha256=manifest["manifest_sha256"],
            model_source_manifest_sha256=manifest["manifest_sha256"], reference_source_manifest_sha256=reference_manifest["manifest_sha256"],
            all_requested_physical_adapters_completed=all(t["status"] == "descriptive_audit_completed" for t in result["targets"].values()))
    except Exception as error:
        result["execution_passed"] = False
        result["failure"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        try:
            _publish_audit_result(output, result)
        except Exception:
            pass  # Keep the original run failure; result/partial publication was attempted first.
        raise
    _publish_audit_result(output, result)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--source", type=Path, help="legacy alias for model source; default reference is this same corpus")
    source.add_argument("--model-source", type=Path, help="checkpoint training corpus, not the new reference corpus")
    p.add_argument("--reference-source", type=Path, help="optional fresh reference corpus; training identity is not rebound")
    p.add_argument("--generated-dir", type=Path, action="append", required=True, help="explicit frozen target; repeat for several")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--sample-count", type=int, choices=(128, 512), default=128)
    p.add_argument("--bootstrap-repeats", type=int, default=49)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
