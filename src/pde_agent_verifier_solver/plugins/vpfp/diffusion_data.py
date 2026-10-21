"""Offline, trajectory-disjoint VPFP coefficient corpora for diffusion training.

The sampler is the existing coarse numerical solve plus continuous projection,
not a high-fidelity reference. Rejection sampling changes only which equation
seeds are retained; it never changes an equation's sampled physical parameters.
Collection requires a Slurm allocation and a fresh directory under /scratch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from copy import deepcopy
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
import shutil
from typing import Mapping

import numpy as np

from .canonical_binding import VPFPOnlineModelSpec, build_vpfp_online_binding


SCHEMA = "vpfp_stratified_coefficient_corpus_v1"
SPLIT_NAMES = ("train", "validation", "calibration", "reference_a", "reference_b")
DEFAULT_SPLIT_COUNTS = dict(zip(SPLIT_NAMES, (512, 128, 128, 128, 128)))
_SCRATCH_ROOT = Path("/scratch")
_PILOT_SEEDS = frozenset(range(9_600_000, 9_600_032))
_ARRAY_NAMES = ("coefficients", "conditions", "equation_seeds", "case_ids", "stratum_ids")
_REUSED_SPLITS = ("train", "validation")
_EVALUATION_SPLITS = ("calibration", "reference_a", "reference_b")
_FRESH_EVALUATION_MODE = "reuse_train_validation_fresh_evaluation"
_INDEPENDENT_COLLECTION_MODE = "fresh_all_splits_excluding_cumulative_history"


@dataclass(frozen=True)
class CorpusSplit:
    """Read-only mmap arrays; one row is one complete independent trajectory."""

    coefficients: np.ndarray
    conditions: np.ndarray
    equation_seeds: np.ndarray
    case_ids: np.ndarray
    stratum_ids: np.ndarray
    manifest: Mapping[str, object]

    @property
    def latents(self):
        return self.coefficients

    @property
    def equation_conditions(self):
        return self.conditions

    @property
    def sample_ids(self):
        return self.equation_seeds

    def __len__(self):
        return len(self.equation_seeds)


@dataclass(frozen=True)
class _CollectionRuntime:
    sampler: object
    condition_spec: object
    latent_dimension: int
    provenance: Mapping[str, object]


def _collection_runtime(spec, output_root, device):
    # Reuse the validated factory so equation, decoder and projection settings
    # cannot drift from the canonical physical backend. No generate call here.
    binding = build_vpfp_online_binding(spec, output_root=output_root / "runtime", device=device)
    sampler = binding.plugin.generator.backend.runtime.generator
    physical = importlib.import_module("agent_verifier.vpfp.high_fidelity")
    decoder = sampler.projector.config
    modules = ("high_fidelity", "online_training", "torch_reference",
               "continuous_flow_training", "continuous_hermite")
    provenance = {
        "binding": dict(binding.report),
        "decoder_config": decoder.to_dict(),
        "structured_latent_layout": decoder.structured_latent_layout(),
        "generation_config": sampler.config.to_dict(),
        "equation_config": sampler.equation_config.to_dict(),
        "physical_module_sha256": {
            name: _file_sha256(Path(importlib.import_module(f"agent_verifier.vpfp.{name}").__file__))
            for name in modules
        },
    }

    def condition_spec(seed):
        # This exact call (including deferred normalization) is used by generate.
        return physical.sample_continuous_equation_spec(
            seed=int(seed), config=sampler.equation_config, defer_rho_log_partition=True)

    return _CollectionRuntime(sampler, condition_spec, int(decoder.latent_dim), provenance)


def _file_sha256(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value):
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical_bytes(value):
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _write_json(path, value):
    # Each target is inside the newly created collection directory. A killed
    # writer leaves its partial manifest, never a loadable completed corpus.
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(_canonical_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _validate_counts(split_counts, strata, batch_size):
    if not isinstance(split_counts, Mapping) or set(split_counts) != set(SPLIT_NAMES):
        raise ValueError(f"split_counts must specify exactly {SPLIT_NAMES}")
    if not isinstance(strata, (tuple, list)) or len(strata) != 2:
        raise ValueError("strata must contain nu and temperature bin counts")
    bins = tuple(_positive_integer(value, "strata") for value in strata)
    batch_size = _positive_integer(batch_size, "batch_size")
    counts = {name: _positive_integer(split_counts[name], f"{name} count") for name in SPLIT_NAMES}
    if any(count % (bins[0] * bins[1]) or count % batch_size for count in counts.values()):
        raise ValueError("each split count must be divisible by all strata and by batch_size")
    return counts, bins, batch_size


def _output_path(output_root):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("VPFP corpus collection requires a Slurm compute allocation")
    raw = Path(output_root).expanduser()
    root, scratch = raw.resolve(), _SCRATCH_ROOT.resolve()
    if root == scratch or not root.is_relative_to(scratch):
        raise ValueError("VPFP corpus output must be a fresh child directory under /scratch")
    if raw.is_symlink() or raw.exists() or root.exists():
        raise FileExistsError(f"refusing to reuse or overwrite corpus directory: {raw}")
    return root


def _stratum(conditions, edges):
    values = np.asarray(conditions, dtype=np.float64)
    if values.shape != (2,) or not np.isfinite(values).all():
        raise ValueError("equation conditions must be a finite nu/temperature pair")
    indices = []
    for value, axis in zip(values, edges):
        if value < axis[0] or value > axis[-1]:
            raise ValueError("sampled equation condition is outside the requested regime")
        indices.append(min(int(np.searchsorted(axis, value, side="right")) - 1, len(axis) - 2))
    return indices[0] * (len(edges[1]) - 1) + indices[1]


def _source_exclusion_seeds(plan):
    """Canonical cumulative history from an already SHA-bound local seed plan.

    Only the embedded seed history is consumed: never follow old corpus paths.
    Reused train/validation IDs appear in both lists, so union rather than count
    concatenation. Reject coercible strings/floats/bools and out-of-int64 IDs.
    """
    history = plan.get("excluded_source_equation_seeds", [])
    if not isinstance(history, list):
        raise ValueError("source equation seed history must be a list of int64 IDs")
    current = [row["equation_seed"] for rows in plan["splits"].values() for row in rows]
    seeds = history + current
    if any(isinstance(seed, (bool, np.bool_))
           or not isinstance(seed, (int, np.integer))
           or not 0 <= seed <= np.iinfo(np.int64).max for seed in seeds):
        raise ValueError("source equation seed history contains an invalid int64 ID")
    return sorted({int(seed) for seed in seeds})


def _plan_seeds(spec, counts, master_seed, bins, condition_spec, *,
                split_names=SPLIT_NAMES, excluded_equation_seeds=()):
    edges = [np.linspace(*bounds, count + 1) for bounds, count in
             zip((spec.collision_nu_range, spec.temperature_range), bins)]
    # Stable, regime-specific namespace; no dependence on Python's salted hash.
    namespace = _canonical_bytes({"schema": SCHEMA, "regime_id": spec.regime_id,
                                  "master_seed": master_seed})
    rng = np.random.default_rng(int.from_bytes(sha256(namespace).digest()[:16], "big"))
    seen = set(_PILOT_SEEDS).union(int(seed) for seed in excluded_equation_seeds)
    plans, attempted = {}, 0
    for name in split_names:
        quota = counts[name] // (bins[0] * bins[1])
        filled = np.zeros(bins[0] * bins[1], dtype=np.int64)
        rows = []
        # Fail explicitly if a future physical sampler stops covering the
        # requested regime. No PDE solves occur until the whole plan succeeds.
        for _ in range(max(10_000, counts[name] * 1_000)):
            if len(rows) == counts[name]:
                break
            seed = int(rng.integers(0, np.iinfo(np.int64).max))
            if seed in seen:
                continue
            seen.add(seed)
            attempted += 1
            equation = condition_spec(seed)
            conditions = [float(equation[key]) for key in ("collision_nu", "temperature")]
            if "seed" in equation and int(equation["seed"]) != seed:
                raise ValueError("cheap equation sampler returned a different seed")
            group = _stratum(conditions, edges)
            if filled[group] >= quota:
                continue
            # Keep the original double-precision sampled condition as lineage;
            # the generate output is compared to its exact float32 encoding.
            rows.append({"equation_seed": seed, "conditions": conditions,
                         "stratum_id": group, "case_id": f"{spec.regime_id}:{seed}"})
            filled[group] += 1
        if len(rows) != counts[name]:
            raise RuntimeError(f"cheap seed screening did not cover all strata for {name}")
        plans[name] = rows
    return {"schema": "vpfp_equation_seed_plan_v1", "master_seed": master_seed,
            "regime_id": spec.regime_id, "strata": list(bins),
            "stratum_order": "nu_bin * temperature_bin_count + temperature_bin",
            "condition_order": ["collision_nu", "temperature"],
            "bin_edges": [axis.tolist() for axis in edges],
            "split_counts": counts, "cheap_candidates_screened": attempted,
            "excluded_pilot_equation_seeds": sorted(_PILOT_SEEDS), "splits": plans}


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def collect_corpus(spec, output_root, split_counts, master_seed, strata=(4, 2),
                   batch_size=8, device="cuda"):
    """Collect all five balanced splits, preserving partial output on failure.

    The returned manifest contains a canonical ``manifest_sha256``; the final
    manifest appears only after all arrays have been flushed and hashed.
    Existing partial or complete directories are never reused automatically.
    """
    return _collect_corpus(spec, output_root, split_counts, master_seed, strata,
                           batch_size, device)


def collect_fresh_evaluation_corpus(source_corpus, output_root, master_seed,
                                    batch_size=8, device="cuda", evaluation_count=None):
    """Copy training/validation bytes; collect three newly screened eval splits.

    Source counts and the original 4-by-2 nu/temperature bins are retained by
    default. An explicit evaluation_count replaces only each of the three new
    evaluation counts; copied training/validation bytes remain unchanged.
    Every old equation seed, including the embedded cumulative history of
    previously inspected held-out sets, is excluded from new evaluation. No
    historical corpus path is followed. The complete plan is persisted before any
    numerical solve. Source files are only read, and source hashes are checked
    both before collection and before publication. Failed output is retained.
    """
    if evaluation_count is not None:
        evaluation_count = _positive_integer(evaluation_count, "evaluation_count")
        batch_size = _positive_integer(batch_size, "batch_size")
        if evaluation_count % 8 or evaluation_count % batch_size:
            raise ValueError("evaluation_count must be divisible by eight strata and batch_size")
    spec, source = _frozen_collection_source(source_corpus)
    counts = dict(source["manifest"]["split_counts"])
    if evaluation_count is not None:
        counts.update({name: evaluation_count for name in _EVALUATION_SPLITS})
    return _collect_corpus(spec, output_root, counts, master_seed,
                           (4, 2), batch_size, device, source=source)


def collect_independent_corpus(source_corpus, output_root, split_counts, master_seed,
                               batch_size=8, device="cuda"):
    """Generate five NEW splits, excluding source IDs and their entire history.

    The source binds the physical runtime and an immutable seed-history
    snapshot only. No training/validation answer is copied, and the new split
    counts can differ. All five screened seed lists are published before the
    first numerical solve. Existing collector, Slurm/scratch guards, failure
    journal and native corpus loader remain the only collection lifecycle.
    """
    spec, source = _frozen_collection_source(source_corpus)
    return _collect_corpus(spec, output_root, split_counts, master_seed, (4, 2),
                           batch_size, device, source=source, reused_splits=())


def _frozen_collection_source(source_corpus):
    """Read source metadata and verify every source byte without following history paths."""
    source_root = Path(source_corpus).expanduser().resolve(strict=True)
    manifest_file_hash = _file_sha256(source_root / "manifest.json")
    source_manifest = read_manifest(source_root, verify_hashes=True)
    if source_manifest["strata"] != [4, 2]:
        raise ValueError("fresh evaluation requires the original 4-by-2 joint strata")
    spec = VPFPOnlineModelSpec.from_mapping(source_manifest["model_spec"])
    for key in ("regime_id", "base_decoder_sha256", "collision_nu_range",
                "temperature_range", "time_domain"):
        if _canonical_bytes(source_manifest[key]) != _canonical_bytes(getattr(spec, key)):
            raise ValueError(f"source corpus model identity mismatch: {key}")
    expected_edges = [np.linspace(*bounds, bins + 1).tolist() for bounds, bins in
                      zip((spec.collision_nu_range, spec.temperature_range), (4, 2))]
    if source_manifest["bin_edges"] != expected_edges:
        raise ValueError("source corpus bins differ from the physical model ranges")
    source_plan = json.loads(_checked_file(source_root, source_manifest["seed_plan"], True).read_text())
    source = {"root": source_root, "manifest": source_manifest, "plan": source_plan,
              "manifest_file_sha256": manifest_file_hash}
    return spec, source


def _copy_verified_bytes(source_path, target_path, expected_sha256):
    # Exclusive creation gives an independent inode even on one filesystem;
    # never use hard links, symlinks, or an overwrite-capable convenience copy.
    with source_path.open("rb") as source, target_path.open("xb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)
        target.flush()
        os.fsync(target.fileno())


def _snapshot_source(root, source, *, reused_splits=_REUSED_SPLITS):
    original = source["manifest"]
    directory = root / "source_provenance"
    directory.mkdir()
    entries = {}
    for key, path, digest in (
        ("manifest_snapshot", source["root"] / "manifest.json", source["manifest_file_sha256"]),
        ("seed_plan_snapshot", _checked_file(source["root"], original["seed_plan"], True),
         original["seed_plan"]["sha256"]),
    ):
        target = directory / ("manifest.json" if key == "manifest_snapshot" else "seed_plan.json")
        _copy_verified_bytes(path, target, digest)
        entries[key] = {"path": str(target.relative_to(root)), "sha256": digest,
                        "size_bytes": target.stat().st_size}
    return {"path": str(source["root"]), "manifest_sha256": original["manifest_sha256"],
            "manifest_file_sha256": source["manifest_file_sha256"],
            "seed_plan_sha256": original["seed_plan"]["sha256"],
            "reused_splits": list(reused_splits),
            "regenerated_splits": [name for name in SPLIT_NAMES if name not in reused_splits],
            "excluded_equation_seed_count": len(_source_exclusion_seeds(source["plan"])),
            "copy_method": "bytewise_independent_copy" if reused_splits else "metadata_snapshots_only_all_splits_new", **entries}


def _copy_source_split(root, name, source):
    original = source["manifest"]["splits"][name]
    directory = root / name
    directory.mkdir()
    copied = deepcopy(original)
    copied["batches"] = []  # Source solve diagnostics remain in the snapshot.
    copied["collection_origin"] = {"method": "bytewise_independent_copy", "source_split": name,
        "source_manifest_sha256": source["manifest"]["manifest_sha256"],
        "generated_in_this_collection": False}
    for key, entry in original["files"].items():
        path = _checked_file(source["root"], entry, True)
        target = directory / f"{key}.npy"
        _copy_verified_bytes(path, target, entry["sha256"])
        copied["files"][key]["path"] = str(target.relative_to(root))
    return copied


def _collect_corpus(spec, output_root, split_counts, master_seed, strata,
                    batch_size, device, *, source=None, reused_splits=_REUSED_SPLITS):
    if reused_splits not in (_REUSED_SPLITS, ()):
        raise ValueError("source reuse must be train/validation together or no splits")
    if not isinstance(spec, VPFPOnlineModelSpec):
        spec = VPFPOnlineModelSpec.from_mapping(spec)
    counts, bins, batch_size = _validate_counts(split_counts, strata, batch_size)
    if isinstance(master_seed, bool) or not isinstance(master_seed, (int, np.integer)) or master_seed < 0:
        raise ValueError("master_seed must be a nonnegative integer")
    master_seed = int(master_seed)
    root = _output_path(output_root)
    if source is not None and root.is_relative_to(source["root"]):
        raise ValueError("fresh output cannot be inside the read-only source corpus")
    # Check identity/ranges before creating any output or invoking a sampler.
    checked = spec.preflight()
    if source is not None and checked["checkpoints"] != source["manifest"]["checkpoints"]:
        raise ValueError("source corpus checkpoint identity differs from current preflight")
    root.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": SCHEMA, "status": "collecting", "regime_id": spec.regime_id,
                "model_spec": asdict(spec), "checkpoints": checked["checkpoints"],
                "base_decoder_sha256": spec.base_decoder_sha256,
                "collision_nu_range": list(spec.collision_nu_range),
                "temperature_range": list(spec.temperature_range),
                "time_domain": list(spec.time_domain), "master_seed": master_seed,
                "split_counts": counts, "strata": list(bins), "batch_size": batch_size,
                "sample_unit": "complete_independent_equation_trajectory",
                "data_origin": "coarse_numerical_solve_then_continuous_projection",
                "high_fidelity_validation": False, "experimental_data_validation": False,
                "independent_resolution_calibration": "not_performed",
                "condition_order": ["collision_nu", "temperature"],
                "coefficient_dtype": "float32", "device": str(device),
                "slurm_job_id": os.environ["SLURM_JOB_ID"], "splits": {}}
    _write_json(root / "manifest.partial.json", manifest)
    try:
        if source is not None:
            manifest["collection_mode"] = _FRESH_EVALUATION_MODE if reused_splits else _INDEPENDENT_COLLECTION_MODE
            manifest["source_corpus"] = _snapshot_source(root, source, reused_splits=reused_splits)
            _write_json(root / "manifest.partial.json", manifest)
        runtime = _collection_runtime(spec, root, device)
        dimension = _positive_integer(runtime.latent_dimension, "latent_dimension")
        manifest.update(latent_dimension=dimension, provenance=_jsonable(runtime.provenance))
        if source is None:
            plan = _plan_seeds(spec, counts, master_seed, bins, runtime.condition_spec)
        else:
            original = source["manifest"]
            if dimension != original["latent_dimension"]:
                raise ValueError("source corpus latent dimension differs from current runtime")
            # Binding reports contain output paths. Physical configurations and
            # implementation hashes must match exactly; paths need not match.
            for key in ("decoder_config", "structured_latent_layout", "generation_config",
                        "equation_config", "physical_module_sha256"):
                if _canonical_bytes(runtime.provenance.get(key)) != _canonical_bytes(original["provenance"].get(key)):
                    raise ValueError(f"source corpus physical runtime drift: {key}")
            excluded = _source_exclusion_seeds(source["plan"])
            generated_splits = tuple(name for name in SPLIT_NAMES if name not in reused_splits)
            plan = _plan_seeds(spec, counts, master_seed, bins, runtime.condition_spec,
                               split_names=generated_splits, excluded_equation_seeds=excluded)
            plan["splits"].update({name: deepcopy(source["plan"]["splits"][name]) for name in reused_splits})
            plan.update(reused_splits=list(reused_splits), excluded_source_equation_seeds=excluded,
                        source_manifest_sha256=original["manifest_sha256"])
        _write_json(root / "seed_plan.json", plan)
        manifest["seed_plan"] = {"path": "seed_plan.json", "sha256": _file_sha256(root / "seed_plan.json")}
        manifest["bin_edges"] = plan["bin_edges"]
        for name in SPLIT_NAMES:
            if source is not None and name in reused_splits:
                manifest["splits"][name] = _copy_source_split(root, name, source)
                _write_json(root / "manifest.partial.json", manifest)
                continue
            directory = root / name
            directory.mkdir()
            rows, count = plan["splits"][name], counts[name]
            shapes = {"coefficients": ((count, dimension), np.dtype("float32")),
                      "conditions": ((count, 2), np.dtype("float32")),
                      "equation_seeds": ((count,), np.dtype("int64")),
                      "case_ids": ((count,), np.dtype(f"U{len(spec.regime_id) + 21}")),
                      "stratum_ids": ((count,), np.dtype("int64"))}
            arrays = {key: np.lib.format.open_memmap(directory / f"{key}.npy", mode="w+",
                      dtype=dtype, shape=shape) for key, (shape, dtype) in shapes.items()}
            split = {"count": count, "rows_written": 0, "status": "collecting", "files": {}, "batches": []}
            manifest["splits"][name] = split
            _write_json(root / "manifest.partial.json", manifest)
            for start in range(0, count, batch_size):
                selected = rows[start:start + batch_size]
                seeds = tuple(row["equation_seed"] for row in selected)
                batch = runtime.sampler.generate(seeds)
                coefficients, conditions = _numpy(batch["latents"]), _numpy(batch["equation_conditions"])
                returned_seeds = _numpy(batch["equation_seeds"])
                expected = np.asarray([row["conditions"] for row in selected], dtype=np.float32)
                if coefficients.shape != (len(seeds), dimension) or coefficients.dtype != np.dtype("float32"):
                    raise ValueError("sampler returned an unexpected complete coefficient shape or dtype")
                if not np.isfinite(coefficients).all():
                    raise ValueError("sampler returned nonfinite complete coefficients")
                if conditions.dtype != np.dtype("float32") or not np.array_equal(conditions, expected):
                    raise ValueError("generated conditions differ from the screened physical equation")
                if returned_seeds.dtype != np.dtype("int64") or not np.array_equal(returned_seeds, seeds):
                    raise ValueError("sampler returned different equation seeds")
                stop = start + len(seeds)
                arrays["coefficients"][start:stop] = coefficients
                arrays["conditions"][start:stop] = conditions
                arrays["equation_seeds"][start:stop] = seeds
                arrays["case_ids"][start:stop] = [row["case_id"] for row in selected]
                arrays["stratum_ids"][start:stop] = [row["stratum_id"] for row in selected]
                for array in arrays.values():
                    array.flush()
                split["rows_written"] = stop
                split["batches"].append({"start": start, "count": len(seeds),
                    "diagnostics": _jsonable(batch.get("diagnostics", {}))})
                _write_json(root / "manifest.partial.json", manifest)
                print(json.dumps({"split": name, "rows_written": stop, "split_count": count}), flush=True)
                # A returned batch may hold GPU tensors. Do not retain it while
                # the next numerical solve runs, or build an all-corpus cache.
                del batch, coefficients, conditions, returned_seeds
            for key, (shape, dtype) in shapes.items():
                path = directory / f"{key}.npy"
                split["files"][key] = {"path": str(path.relative_to(root)),
                    "shape": list(shape), "dtype": dtype.str,
                    "size_bytes": path.stat().st_size, "sha256": _file_sha256(path)}
            split["status"] = "complete"
            del arrays
            _write_json(root / "manifest.partial.json", manifest)
        if source is not None:
            current_source = read_manifest(source["root"], verify_hashes=True)
            _validate_reuse_provenance(root, manifest, plan)
            manifest["source_corpus"]["revalidated_before_publication"] = True
        manifest["status"] = "complete"
        manifest["manifest_sha256"] = sha256(_canonical_bytes(manifest)).hexdigest()
        # Retain the collection journal, explicitly marking its completion.
        _write_json(root / "manifest.partial.json", manifest)
        _write_json(root / "manifest.json", manifest)
        return _jsonable(manifest)
    except BaseException as exc:
        manifest.pop("manifest_sha256", None)
        manifest["status"] = "failed"
        manifest["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        try:
            _write_json(root / "manifest.partial.json", manifest)
        except Exception:
            pass  # Preserve the original collection failure (e.g. disk full).
        raise


def _checked_file(root, entry, verify_hashes):
    path = root / entry["path"]
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError("corpus file is missing, symlinked, or outside the corpus")
    if "size_bytes" in entry and path.stat().st_size != entry["size_bytes"]:
        raise ValueError(f"corpus file size mismatch: {path.name}")
    return path


def _validate_reuse_provenance(root, manifest, plan):
    """Validate reuse from local snapshots, without requiring the old path."""
    mode = manifest.get("collection_mode")
    if mode not in (_FRESH_EVALUATION_MODE, _INDEPENDENT_COLLECTION_MODE):
        if "source_corpus" in manifest or "excluded_source_equation_seeds" in plan:
            raise ValueError("source provenance requires fresh-evaluation collection mode")
        return
    reused_splits = _REUSED_SPLITS if mode == _FRESH_EVALUATION_MODE else ()
    generated_splits = tuple(name for name in SPLIT_NAMES if name not in reused_splits)
    source = manifest["source_corpus"]
    original = json.loads(_checked_file(root, source["manifest_snapshot"], True).read_text())
    digest = original.pop("manifest_sha256", None)
    if (original.get("schema") != SCHEMA or original.get("status") != "complete"
            or digest != source["manifest_sha256"] or digest != plan["source_manifest_sha256"]
            or source["manifest_file_sha256"] != source["manifest_snapshot"]["sha256"]
            or source["seed_plan_sha256"] != source["seed_plan_snapshot"]["sha256"]
            or source["seed_plan_sha256"] != original["seed_plan"]["sha256"]):
        raise ValueError("source snapshot identity differs from reuse provenance")
    old_plan = json.loads(_checked_file(root, source["seed_plan_snapshot"], True).read_text())
    if (set(old_plan["splits"]) != set(SPLIT_NAMES)
            or old_plan["split_counts"] != original["split_counts"]
            or old_plan["regime_id"] != original["regime_id"]
            or old_plan["master_seed"] != original["master_seed"]
            or old_plan["bin_edges"] != original["bin_edges"]):
        raise ValueError("source seed-plan identity differs from its snapshot manifest")
    for key in ("regime_id", "model_spec", "checkpoints", "base_decoder_sha256",
                "collision_nu_range", "temperature_range", "time_domain",
                "strata", "bin_edges", "latent_dimension"):
        if _canonical_bytes(manifest[key]) != _canonical_bytes(original[key]):
            raise ValueError(f"source snapshot differs from reused corpus identity: {key}")
    if any(manifest["split_counts"][name] != original["split_counts"][name] for name in reused_splits):
        raise ValueError("source snapshot differs from reused corpus identity: split_counts")
    for key in ("decoder_config", "structured_latent_layout", "generation_config",
                "equation_config", "physical_module_sha256"):
        if _canonical_bytes(manifest["provenance"].get(key)) != _canonical_bytes(original["provenance"].get(key)):
            raise ValueError(f"source snapshot physical runtime drift: {key}")
    excluded = _source_exclusion_seeds(old_plan)
    current = [row["equation_seed"] for rows in old_plan["splits"].values() for row in rows]
    # Validate this plan's stored list as well: Python list equality alone
    # would accept a rehashed float/bool in place of an integer ID.
    _source_exclusion_seeds(plan)
    expected_copy = "bytewise_independent_copy" if reused_splits else "metadata_snapshots_only_all_splits_new"
    if (source["reused_splits"] != list(reused_splits)
            or source["regenerated_splits"] != list(generated_splits)
            or source["copy_method"] != expected_copy
            or plan["reused_splits"] != list(reused_splits)
            or plan["excluded_source_equation_seeds"] != excluded
            or source["excluded_equation_seed_count"] != len(excluded)
            or len(set(current)) != len(current)
            or len(current) != sum(original["split_counts"].values())):
        raise ValueError("fresh evaluation exclusion set differs from cumulative source history")
    for name in generated_splits:
        if set(excluded).intersection(row["equation_seed"] for row in plan["splits"][name]):
            raise ValueError("fresh evaluation overlaps an old source equation seed")
        if "collection_origin" in manifest["splits"][name]:
            raise ValueError("newly generated split cannot claim a copied collection origin")
    for name in reused_splits:
        split = manifest["splits"][name]
        expected_origin = {"method": "bytewise_independent_copy", "source_split": name,
                           "source_manifest_sha256": digest, "generated_in_this_collection": False}
        if (plan["splits"][name] != old_plan["splits"][name]
                or split.get("collection_origin") != expected_origin or split["batches"]):
            raise ValueError("reused split lineage or collection origin differs from source")
        for key in _ARRAY_NAMES:
            before, after = original["splits"][name]["files"][key], split["files"][key]
            if any(before[field] != after[field] for field in ("sha256", "size_bytes", "shape", "dtype")):
                raise ValueError("reused split file metadata differs from original bytes")
    if manifest.get("status") == "complete" and source.get("revalidated_before_publication") is not True:
        raise ValueError("completed fresh corpus lacks final source revalidation")


def read_manifest(output_root, *, verify_hashes=True):
    """Read a completed corpus and check shapes, state and disjoint lineage.

    ``verify_hashes`` remains accepted for older callers but does not trigger
    checksum scans. Seed-plan contents are still matched to the stored rows.
    """
    root = Path(output_root).resolve(strict=True)
    path = root / "manifest.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("no completed corpus manifest; partial output is not training data")
    manifest = json.loads(path.read_text())
    digest = manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = digest
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "complete":
        raise ValueError("corpus schema or completion state is invalid")
    counts, bins, _ = _validate_counts(manifest["split_counts"], manifest["strata"], manifest["batch_size"])
    dimension = _positive_integer(manifest["latent_dimension"], "latent_dimension")
    if set(manifest["splits"]) != set(SPLIT_NAMES):
        raise ValueError("corpus must contain all five disjoint splits")
    plan = json.loads(_checked_file(root, manifest["seed_plan"], True).read_text())
    if (plan["regime_id"] != manifest["regime_id"] or plan["master_seed"] != manifest["master_seed"]
            or plan["split_counts"] != counts or plan["bin_edges"] != manifest["bin_edges"]
            or set(plan["splits"]) != set(SPLIT_NAMES)):
        raise ValueError("corpus seed-plan identity differs from manifest")
    _validate_reuse_provenance(root, manifest, plan)
    seen = set(_PILOT_SEEDS)
    for name in SPLIT_NAMES:
        split, count = manifest["splits"][name], counts[name]
        if split["status"] != "complete" or split["rows_written"] != count or split["count"] != count:
            raise ValueError(f"incomplete corpus split: {name}")
        if set(split["files"]) != set(_ARRAY_NAMES):
            raise ValueError("corpus split is missing complete trajectory arrays")
        arrays = {}
        for key, entry in split["files"].items():
            array = np.load(_checked_file(root, entry, verify_hashes), mmap_mode="r", allow_pickle=False)
            shape = (count, dimension) if key == "coefficients" else ((count, 2) if key == "conditions" else (count,))
            kind = "f" if key in ("coefficients", "conditions") else ("U" if key == "case_ids" else "i")
            if array.shape != shape or list(array.shape) != entry["shape"] or array.dtype.str != entry["dtype"]:
                raise ValueError(f"corpus shape/dtype mismatch: {name}/{key}")
            if array.dtype.kind != kind or (kind == "f" and array.dtype != np.dtype("float32")):
                raise ValueError(f"unexpected corpus array dtype: {key}")
            arrays[key] = array
        rows = plan["splits"][name]
        seeds = [int(value) for value in arrays["equation_seeds"]]
        if len(rows) != count or len(set(seeds)) != count or seen.intersection(seeds) or any(seed < 0 for seed in seeds):
            raise ValueError("equation seed overlap within/across splits or with excluded pilot")
        seen.update(seeds)
        expected_conditions = np.asarray([row["conditions"] for row in rows], dtype=np.float32)
        groups = [_stratum(row["conditions"], plan["bin_edges"]) for row in rows]
        if (seeds != [row["equation_seed"] for row in rows]
                or not np.array_equal(arrays["conditions"], expected_conditions)
                or not np.array_equal(arrays["stratum_ids"], groups)
                or groups != [row["stratum_id"] for row in rows]
                or not np.array_equal(arrays["case_ids"], [f"{manifest['regime_id']}:{seed}" for seed in seeds])
                or [row["case_id"] for row in rows] != arrays["case_ids"].tolist()):
            raise ValueError("corpus equation lineage differs from its screened seed plan")
        if not np.array_equal(np.bincount(groups, minlength=bins[0] * bins[1]),
                              np.full(bins[0] * bins[1], count // (bins[0] * bins[1]))):
            raise ValueError("corpus split is not equally stratified")
    return manifest


def load_split(output_root, name, *, verify_hashes=True):
    """Load one split as read-only arrays, after validating all split lineage."""
    if name not in SPLIT_NAMES:
        raise ValueError(f"unknown corpus split: {name}")
    root = Path(output_root).resolve(strict=True)
    manifest = read_manifest(root, verify_hashes=verify_hashes)
    files = manifest["splits"][name]["files"]
    arrays = {key: np.load(root / files[key]["path"], mmap_mode="r", allow_pickle=False)
              for key in _ARRAY_NAMES}
    return CorpusSplit(**arrays, manifest=manifest)
