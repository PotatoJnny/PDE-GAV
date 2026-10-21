"""Small fake trajectories only: no numerical solve or GPU work on login nodes."""

from hashlib import sha256
import json
from types import SimpleNamespace

import numpy as np
import pytest

from pde_agent_verifier_solver.plugins.vpfp import diffusion_data as data
from pde_agent_verifier_solver.plugins.vpfp.canonical_binding import VPFPOnlineModelSpec


class FakeSampler:
    def __init__(self):
        self.screened = []
        self.generated = []
        self.fail_call = None
        self.corrupt = None

    def condition(self, seed):
        self.screened.append(seed)
        return {"seed": seed, "collision_nu": .02 + ((seed % 4) + .5) * .06 / 4,
                "temperature": .8 + (((seed // 4) % 2) + .5) * .4 / 2}

    def generate(self, seeds):
        self.generated.append(seeds)
        if self.fail_call == len(self.generated):
            raise RuntimeError("deliberate numerical solve failure")
        values = [self.condition(seed) for seed in seeds]
        batch = {"latents": np.asarray([[seed % 101, 2., 3., 4., 5., 6.] for seed in seeds], dtype=np.float32),
                 "equation_conditions": np.asarray([[row["collision_nu"], row["temperature"]]
                                                     for row in values], dtype=np.float32),
                 "equation_seeds": np.asarray(seeds, dtype=np.int64),
                 "diagnostics": {"fake": True, "count": np.int64(len(seeds))}}
        if self.corrupt == "conditions":
            batch["equation_conditions"][0, 0] += .001
        elif self.corrupt == "seed":
            batch["equation_seeds"][0] += 1
        elif self.corrupt == "nonfinite":
            batch["latents"][0, 0] = np.nan
        elif self.corrupt == "shape":
            batch["latents"] = batch["latents"][:, :5]
        elif self.corrupt == "dtype":
            batch["latents"] = batch["latents"].astype(np.float64)
        return batch


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "unit-test-no-compute")
    monkeypatch.setattr(data, "_SCRATCH_ROOT", tmp_path)
    base, expanded = tmp_path / "base.pt", tmp_path / "expanded.pt"
    base.write_bytes(b"test decoder; never loaded as a torch checkpoint")
    expanded.write_bytes(b"test expanded decoder")
    spec = VPFPOnlineModelSpec("weak_fixture", str(base), sha256(base.read_bytes()).hexdigest(),
                             str(expanded), sha256(expanded.read_bytes()).hexdigest(), (.02, .08))
    sampler = FakeSampler()
    runtime = data._CollectionRuntime(sampler, sampler.condition, 6, {
        "generation_config": {"nx": 16, "ny": 16, "velocity_grid_size": 21, "nt": 33},
        "equation_config": {"fake": True}, "structured_latent_layout": {"total_latent_dim": 6}})
    monkeypatch.setattr(data, "_collection_runtime", lambda *_args: runtime)
    return SimpleNamespace(spec=spec, sampler=sampler, root=tmp_path / "corpus",
                           counts={name: 16 for name in data.SPLIT_NAMES})


def collect(fixture, **kwargs):
    return data.collect_corpus(fixture.spec, kwargs.pop("output_root", fixture.root),
        kwargs.pop("split_counts", fixture.counts), kwargs.pop("master_seed", 5312026),
        device="cpu", **kwargs)


def test_balanced_disjoint_complete_trajectory_memmaps(fixture):
    manifest = collect(fixture)
    assert manifest == data.read_manifest(fixture.root)
    assert manifest["status"] == "complete"
    assert manifest["latent_dimension"] == 6
    assert manifest["sample_unit"] == "complete_independent_equation_trajectory"
    assert manifest["data_origin"] == "coarse_numerical_solve_then_continuous_projection"
    assert manifest["high_fidelity_validation"] is False
    assert manifest["experimental_data_validation"] is False
    assert manifest["time_domain"] == [0., 1.]
    assert manifest["base_decoder_sha256"] == fixture.spec.base_decoder_sha256
    assert manifest["provenance"]["generation_config"]["nt"] == 33
    assert len(fixture.sampler.generated) == 10
    seen = set()
    generated = [seed for batch in fixture.sampler.generated for seed in batch]
    plan = json.loads((fixture.root / "seed_plan.json").read_text())
    assert plan["cheap_candidates_screened"] > len(generated)
    for name in data.SPLIT_NAMES:
        split = data.load_split(fixture.root, name, verify_hashes=False)
        assert len(split) == 16
        assert isinstance(split.coefficients, np.memmap)
        assert not split.coefficients.flags.writeable
        assert split.coefficients.shape == (16, 6)
        assert split.conditions.shape == (16, 2)
        assert split.latents is split.coefficients
        assert split.equation_conditions is split.conditions
        assert split.sample_ids is split.equation_seeds
        assert np.bincount(split.stratum_ids, minlength=8).tolist() == [2] * 8
        assert not seen.intersection(split.equation_seeds)
        assert not data._PILOT_SEEDS.intersection(split.equation_seeds)
        seen.update(split.equation_seeds)
        assert split.case_ids.tolist() == [f"weak_fixture:{seed}" for seed in split.equation_seeds]
        assert np.array_equal(split.coefficients[:, 0], split.equation_seeds % 101)
    assert seen == set(generated)


def test_seed_plan_is_reproducible_and_regime_specific(fixture):
    collect(fixture)
    other = fixture.root.parent / "second"
    collect(fixture, output_root=other)
    assert (fixture.root / "seed_plan.json").read_bytes() == (other / "seed_plan.json").read_bytes()
    from dataclasses import replace
    alternate = replace(fixture.spec, regime_id="strong_fixture")
    first = data._plan_seeds(fixture.spec, fixture.counts, 5312026, (4, 2), fixture.sampler.condition)
    second = data._plan_seeds(alternate, fixture.counts, 5312026, (4, 2), fixture.sampler.condition)
    assert {row["equation_seed"] for row in first["splits"]["train"]}.isdisjoint(
           row["equation_seed"] for row in second["splits"]["train"])


def test_partial_failure_is_retained_and_never_loadable_or_reused(fixture):
    fixture.sampler.fail_call = 2
    with pytest.raises(RuntimeError, match="deliberate"):
        collect(fixture)
    partial = json.loads((fixture.root / "manifest.partial.json").read_text())
    assert partial["status"] == "failed"
    assert partial["splits"]["train"]["rows_written"] == 8
    assert partial["failure"]["type"] == "RuntimeError"
    assert (fixture.root / "train" / "coefficients.npy").is_file()
    assert not (fixture.root / "manifest.json").exists()
    with pytest.raises(ValueError, match="no completed corpus"):
        data.read_manifest(fixture.root)
    with pytest.raises(FileExistsError, match="refusing to reuse"):
        collect(fixture)


def test_complete_output_never_overwritten(fixture):
    collect(fixture)
    original = (fixture.root / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        collect(fixture)
    assert (fixture.root / "manifest.json").read_bytes() == original


@pytest.mark.parametrize("bad", ["conditions", "seed", "nonfinite", "shape", "dtype"])
def test_sampler_contract_failure_cannot_produce_training_data(fixture, bad):
    fixture.sampler.corrupt = bad
    with pytest.raises(ValueError):
        collect(fixture)
    assert not (fixture.root / "manifest.json").exists()
    partial = json.loads((fixture.root / "manifest.partial.json").read_text())
    assert partial["status"] == "failed"
    assert partial["splits"]["train"]["rows_written"] == 0


@pytest.mark.parametrize("kwargs", [
    {"split_counts": {"train": 16}},
    {"split_counts": {name: 9 for name in data.SPLIT_NAMES}},
    {"strata": (0, 2)}, {"strata": (4,)}, {"batch_size": 3},
    {"master_seed": -1}, {"master_seed": True}, {"batch_size": True},
])
def test_invalid_collection_arguments_fail_before_runtime(fixture, kwargs):
    with pytest.raises(ValueError):
        collect(fixture, **kwargs)
    assert not fixture.sampler.generated
    assert not fixture.root.exists()


def test_compute_and_scratch_guards(fixture, monkeypatch, tmp_path):
    monkeypatch.delenv("SLURM_JOB_ID")
    with pytest.raises(RuntimeError, match="Slurm"):
        collect(fixture)
    monkeypatch.setenv("SLURM_JOB_ID", "unit-test-no-compute")
    monkeypatch.setattr(data, "_SCRATCH_ROOT", tmp_path / "only-allowed")
    with pytest.raises(ValueError, match="under /scratch"):
        collect(fixture)
    assert not fixture.sampler.generated
    assert not fixture.root.exists()


def test_injected_collection_runtime_does_not_rehash_checkpoint(fixture):
    from pathlib import Path
    Path(fixture.spec.base_decoder_checkpoint).write_bytes(b"changed")
    collect(fixture)
    assert data.read_manifest(fixture.root)["status"] == "complete"


def test_metadata_checksums_optional_but_failed_corpus_is_rejected(fixture):
    collect(fixture)
    path = fixture.root / "train" / "coefficients.npy"
    values = np.load(path, mmap_mode="r+")
    values[0, 0] += 1
    values.flush()
    assert data.read_manifest(fixture.root)["status"] == "complete"
    data.read_manifest(fixture.root, verify_hashes=False)
    manifest_path = fixture.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "failed"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="completion state"):
        data.read_manifest(fixture.root, verify_hashes=False)


def test_lineage_checked_even_when_large_hashes_are_skipped(fixture):
    collect(fixture)
    path = fixture.root / "validation" / "equation_seeds.npy"
    seeds = np.load(path, mmap_mode="r+")
    seeds[0] = np.load(fixture.root / "train" / "equation_seeds.npy")[0]
    seeds.flush()
    with pytest.raises(ValueError, match="overlap"):
        data.read_manifest(fixture.root, verify_hashes=False)


def test_seed_plan_ignores_serialization_whitespace(fixture):
    collect(fixture)
    path = fixture.root / "seed_plan.json"
    path.write_text(path.read_text() + " ")
    assert data.read_manifest(fixture.root, verify_hashes=False)["status"] == "complete"


def test_uncovered_strata_fail_before_any_pde_call(fixture, monkeypatch):
    runtime = data._CollectionRuntime(fixture.sampler,
        lambda seed: {"seed": seed, "collision_nu": .021, "temperature": .81}, 6, {})
    monkeypatch.setattr(data, "_collection_runtime", lambda *_args: runtime)
    with pytest.raises(RuntimeError, match="did not cover all strata"):
        collect(fixture)
    assert not fixture.sampler.generated
    assert json.loads((fixture.root / "manifest.partial.json").read_text())["status"] == "failed"


def test_original_cheap_physics_condition_call_no_forward(tmp_path, monkeypatch):
    """Use actual pinned config/condition helper, never build or run a PDE solver."""
    from pde_agent_verifier_solver.plugins.vpfp.perturbation_backend import _legacy_api
    import importlib
    _legacy_api()
    physical = importlib.import_module("agent_verifier.vpfp.high_fidelity")
    equation = physical.VPFPHighFidelityConfig(internal_dt_max=.001,
        collision_nu_range=(.02, .08), temperature_range=(.8, 1.2))
    class Decoder:
        latent_dim = 6

        def to_dict(self):
            return {"fake_config": True}

        def structured_latent_layout(self):
            return {"total_latent_dim": 6}

    sampler = SimpleNamespace(projector=SimpleNamespace(config=Decoder()), equation_config=equation,
                              config=SimpleNamespace(to_dict=lambda: {"fake_generation_config": True}))
    binding = SimpleNamespace(plugin=SimpleNamespace(generator=SimpleNamespace(
        backend=SimpleNamespace(runtime=SimpleNamespace(generator=sampler)))), report={"factory": "fake"})
    monkeypatch.setattr(data, "build_vpfp_online_binding", lambda *args, **kwargs: binding)
    runtime = data._collection_runtime(object(), tmp_path, "cpu")
    assert runtime.sampler is sampler
    for seed in (8, 2026, 3250126068448031):
        actual = runtime.condition_spec(seed)
        expected = physical.sample_continuous_equation_spec(
            seed=seed, config=equation, defer_rho_log_partition=True)
        assert actual == expected
    assert runtime.provenance["equation_config"]["internal_dt_max"] == .001
    assert set(runtime.provenance["physical_module_sha256"]) == {
        "high_fidelity", "online_training", "torch_reference", "continuous_flow_training", "continuous_hermite"}


def fresh(fixture, **kwargs):
    return data.collect_fresh_evaluation_corpus(
        kwargs.pop("source_corpus", fixture.root),
        kwargs.pop("output_root", fixture.root.parent / "fresh"),
        kwargs.pop("master_seed", 5312026), device="cpu", **kwargs)


def resign_manifest(root):
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = sha256(data._canonical_bytes(manifest)).hexdigest()
    path.write_bytes(data._canonical_bytes(manifest))


def test_explicit_fresh_eval_count_changes_only_new_three_splits(fixture):
    original = collect(fixture)
    output = fixture.root.parent / "larger-evaluation"
    fixture.sampler.generated.clear()
    result = fresh(fixture, output_root=output, evaluation_count=32)
    assert result == data.read_manifest(output, verify_hashes=True)
    assert result["split_counts"] == {**fixture.counts, **{name: 32 for name in data._EVALUATION_SPLITS}}
    assert sum(len(batch) for batch in fixture.sampler.generated) == 96
    for name in ("train", "validation"):
        for key in data._ARRAY_NAMES:
            before = fixture.root / original["splits"][name]["files"][key]["path"]
            after = output / result["splits"][name]["files"][key]["path"]
            assert before.read_bytes() == after.read_bytes()
    original_seeds = data._source_exclusion_seeds(json.loads((fixture.root/"seed_plan.json").read_text()))
    assert set(original_seeds).isdisjoint(seed for batch in fixture.sampler.generated for seed in batch)


def test_explicit_none_fresh_count_is_default_byte_compatible(fixture):
    collect(fixture)
    first, second = fixture.root.parent/"default-count", fixture.root.parent/"none-count"
    left = fresh(fixture, output_root=first)
    right = fresh(fixture, output_root=second, evaluation_count=None)
    assert left["split_counts"] == right["split_counts"] == fixture.counts
    assert (first/"seed_plan.json").read_bytes() == (second/"seed_plan.json").read_bytes()
    for role in data.SPLIT_NAMES:
        for key in data._ARRAY_NAMES:
            assert (first/left["splits"][role]["files"][key]["path"]).read_bytes() == (second/right["splits"][role]["files"][key]["path"]).read_bytes()


@pytest.mark.parametrize("count", [True, False, 0, -8, 7, 9, 8.0, "512", float("nan")])
def test_fresh_eval_count_rejects_malformed_before_source_or_output(fixture, count):
    with pytest.raises(ValueError):
        fresh(fixture, evaluation_count=count)
    assert not fixture.root.exists() and not fixture.sampler.generated


def test_fresh_evaluation_copies_exact_bytes_and_never_solves_reused_splits(fixture, monkeypatch):
    original = collect(fixture)
    original_plan = json.loads((fixture.root / "seed_plan.json").read_text())
    old_seeds = {row["equation_seed"] for rows in original_plan["splits"].values() for row in rows}
    source_bytes = {path.relative_to(fixture.root): (path.read_bytes(), path.stat().st_mtime_ns)
                    for path in fixture.root.rglob("*") if path.is_file()}
    fixture.sampler.generated.clear()
    output = fixture.root.parent / "fresh"
    generate = fixture.sampler.generate

    def checked_generate(seeds):
        plan = json.loads((output / "seed_plan.json").read_text())
        assert set(plan["splits"]) == set(data.SPLIT_NAMES)
        assert plan["excluded_source_equation_seeds"] == sorted(old_seeds)
        assert set(seeds).isdisjoint(old_seeds)
        for name in ("train", "validation"):
            assert (output / name / "coefficients.npy").is_file()
            assert plan["splits"][name] == original_plan["splits"][name]
        return generate(seeds)

    monkeypatch.setattr(fixture.sampler, "generate", checked_generate)
    manifest = fresh(fixture)
    assert manifest == data.read_manifest(output)
    assert manifest["manifest_sha256"] != original["manifest_sha256"]
    assert manifest["collection_mode"] == "reuse_train_validation_fresh_evaluation"
    assert manifest["source_corpus"]["manifest_sha256"] == original["manifest_sha256"]
    assert manifest["source_corpus"]["excluded_equation_seed_count"] == 80
    assert manifest["source_corpus"]["revalidated_before_publication"] is True
    assert len(fixture.sampler.generated) == 6
    new_seeds = {seed for batch in fixture.sampler.generated for seed in batch}
    assert len(new_seeds) == 48
    assert new_seeds.isdisjoint(old_seeds)
    for name in ("train", "validation"):
        split = data.load_split(output, name)
        assert not split.coefficients.flags.writeable
        assert manifest["splits"][name]["batches"] == []
        for key in data._ARRAY_NAMES:
            before = fixture.root / original["splits"][name]["files"][key]["path"]
            after = output / manifest["splits"][name]["files"][key]["path"]
            assert before.read_bytes() == after.read_bytes()
            assert before.stat().st_ino != after.stat().st_ino
            assert not after.is_symlink()
            assert manifest["splits"][name]["files"][key]["sha256"] == original["splits"][name]["files"][key]["sha256"]
    for name in ("calibration", "reference_a", "reference_b"):
        split = data.load_split(output, name)
        assert set(split.equation_seeds).issubset(new_seeds)
        assert np.bincount(split.stratum_ids, minlength=8).tolist() == [2] * 8
    assert source_bytes == {path.relative_to(fixture.root): (path.read_bytes(), path.stat().st_mtime_ns)
                            for path in fixture.root.rglob("*") if path.is_file()}


def test_fresh_seed_plan_reproducible_even_with_same_master_seed_as_source(fixture):
    collect(fixture)
    first, second = fixture.root.parent / "fresh_a", fixture.root.parent / "fresh_b"
    fresh(fixture, output_root=first)
    fresh(fixture, output_root=second)
    assert (first / "seed_plan.json").read_bytes() == (second / "seed_plan.json").read_bytes()
    for name in data.SPLIT_NAMES:
        assert (first / name / "coefficients.npy").read_bytes() == (second / name / "coefficients.npy").read_bytes()


def test_fresh_provenance_snapshots_do_not_require_live_source_path(fixture):
    original = collect(fixture)
    output = fixture.root.parent / "fresh"
    manifest = fresh(fixture)
    for key in ("manifest_snapshot", "seed_plan_snapshot"):
        assert not (output / manifest["source_corpus"][key]["path"]).is_symlink()
    fixture.root.rename(fixture.root.parent / "source-not-at-original-path")
    assert data.read_manifest(output)["source_corpus"]["manifest_sha256"] == original["manifest_sha256"]


@pytest.mark.parametrize("split", data.SPLIT_NAMES)
def test_fresh_uses_existing_source_arrays_without_checksum_gate(fixture, monkeypatch, split):
    collect(fixture)
    path = fixture.root / split / "coefficients.npy"
    values = np.load(path, mmap_mode="r+")
    values[0, 0] += 1
    values.flush()
    fresh(fixture)
    assert (fixture.root.parent / "fresh/manifest.json").is_file()


@pytest.mark.parametrize("key", ["decoder_config", "generation_config", "equation_config", "physical_module_sha256"])
def test_fresh_rejects_physical_runtime_drift_before_any_new_solve(fixture, monkeypatch, key):
    collect(fixture)
    original_runtime = data._collection_runtime(None, None, None)
    provenance = dict(original_runtime.provenance)
    provenance[key] = {"changed": True}
    runtime = data._CollectionRuntime(fixture.sampler, fixture.sampler.condition, 6, provenance)
    monkeypatch.setattr(data, "_collection_runtime", lambda *_args: runtime)
    fixture.sampler.generated.clear()
    with pytest.raises(ValueError, match="physical runtime drift"):
        fresh(fixture)
    assert not fixture.sampler.generated
    output = fixture.root.parent / "fresh"
    assert json.loads((output / "manifest.partial.json").read_text())["status"] == "failed"
    assert not (output / "manifest.json").exists()


def test_fresh_partial_solve_failure_retains_copies_and_plan_without_publishing(fixture):
    collect(fixture)
    fixture.sampler.generated.clear()
    fixture.sampler.fail_call = 2
    with pytest.raises(RuntimeError, match="deliberate"):
        fresh(fixture)
    output = fixture.root.parent / "fresh"
    partial = json.loads((output / "manifest.partial.json").read_text())
    assert partial["status"] == "failed"
    assert partial["splits"]["train"]["status"] == "complete"
    assert partial["splits"]["validation"]["status"] == "complete"
    assert partial["splits"]["calibration"]["rows_written"] == 8
    assert (output / "seed_plan.json").is_file()
    assert not (output / "manifest.json").exists()
    with pytest.raises(FileExistsError):
        fresh(fixture)


def test_fresh_does_not_rescan_old_arrays_after_new_solves(fixture, monkeypatch):
    collect(fixture)
    fixture.sampler.generated.clear()
    generate = fixture.sampler.generate

    def mutate_source_after_solve(seeds):
        batch = generate(seeds)
        if len(fixture.sampler.generated) == 1:
            values = np.load(fixture.root / "reference_b" / "coefficients.npy", mmap_mode="r+")
            values[0, 0] += 1
            values.flush()
        return batch

    monkeypatch.setattr(fixture.sampler, "generate", mutate_source_after_solve)
    fresh(fixture)
    output = fixture.root.parent / "fresh"
    assert len(fixture.sampler.generated) == 6
    assert data.read_manifest(output)["status"] == "complete"


def test_fresh_rejects_output_inside_source_without_creating_it(fixture):
    collect(fixture)
    fixture.sampler.generated.clear()
    with pytest.raises(ValueError, match="inside the read-only source"):
        fresh(fixture, output_root=fixture.root / "nested")
    assert not (fixture.root / "nested").exists()
    assert not fixture.sampler.generated


def test_fresh_rejects_changed_original_joint_bins(fixture):
    collect(fixture, strata=(2, 2))
    with pytest.raises(ValueError, match="4-by-2"):
        fresh(fixture)
    assert not (fixture.root.parent / "fresh").exists()


@pytest.mark.parametrize("kwargs", [{"master_seed": -1}, {"master_seed": True}, {"batch_size": 3}])
def test_fresh_invalid_arguments_fail_before_runtime(fixture, monkeypatch, kwargs):
    collect(fixture)
    monkeypatch.setattr(data, "_collection_runtime", lambda *_args: pytest.fail("must not build runtime"))
    with pytest.raises(ValueError):
        fresh(fixture, **kwargs)
    assert not (fixture.root.parent / "fresh").exists()


def test_fresh_snapshot_hash_is_checked_even_without_array_hashes(fixture):
    collect(fixture)
    fresh(fixture)
    output = fixture.root.parent / "fresh"
    path = output / "source_provenance" / "seed_plan.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="file size mismatch|file SHA256 mismatch"):
        data.read_manifest(output, verify_hashes=False)


def test_fresh_loader_rejects_rehashed_reused_array_change(fixture):
    collect(fixture)
    manifest = fresh(fixture)
    output = fixture.root.parent / "fresh"
    path = output / "train" / "coefficients.npy"
    values = np.load(path, mmap_mode="r+")
    values[0, 0] += 1
    values.flush()
    manifest["splits"]["train"]["files"]["coefficients"]["sha256"] = data._file_sha256(path)
    (output / "manifest.json").write_bytes(data._canonical_bytes(manifest))
    resign_manifest(output)
    with pytest.raises(ValueError, match="reused split file metadata"):
        data.read_manifest(output)


def test_fresh_loader_rejects_old_heldout_seed_even_with_rehashed_plan(fixture):
    collect(fixture)
    manifest = fresh(fixture)
    output = fixture.root.parent / "fresh"
    path = output / "seed_plan.json"
    plan = json.loads(path.read_text())
    original = json.loads((fixture.root / "seed_plan.json").read_text())
    plan["splits"]["calibration"][0] = original["splits"]["reference_b"][0]
    path.write_bytes(data._canonical_bytes(plan))
    manifest["seed_plan"]["sha256"] = data._file_sha256(path)
    (output / "manifest.json").write_bytes(data._canonical_bytes(manifest))
    resign_manifest(output)
    with pytest.raises(ValueError, match="overlaps an old source"):
        data.read_manifest(output, verify_hashes=False)


def _plan_ids(plan):
    return {row["equation_seed"] for rows in plan["splits"].values() for row in rows}


def _resign_seed_plan(root, plan):
    """Re-sign a synthetic fixture to test semantic checks beyond file hashes."""
    path = root / "seed_plan.json"
    path.write_bytes(data._canonical_bytes(plan))
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["seed_plan"]["sha256"] = data._file_sha256(path)
    manifest_path.write_bytes(data._canonical_bytes(manifest))
    resign_manifest(root)


def test_cumulative_seed_history_is_a_sorted_legal_union():
    plan = {"splits": {"train": [{"equation_seed": 9}, {"equation_seed": 0}],
                       "reference_a": [{"equation_seed": np.int64(4)}]},
            "excluded_source_equation_seeds": [9, 3, 9, 4, np.iinfo(np.int64).max]}
    assert data._source_exclusion_seeds(plan) == [0, 3, 4, 9, np.iinfo(np.int64).max]
    del plan["excluded_source_equation_seeds"]
    assert data._source_exclusion_seeds(plan) == [0, 4, 9]


@pytest.mark.parametrize("bad", [None, 7, "7", {}, [True], [False], [np.bool_(True)],
                                  [1.0], ["1"], [-1], [2 ** 63], [float("nan")]])
def test_cumulative_seed_history_rejects_illegal_ids_without_coercion(bad):
    plan = {"splits": {"train": [{"equation_seed": 10}]},
            "excluded_source_equation_seeds": bad}
    with pytest.raises(ValueError, match="source equation seed history"):
        data._source_exclusion_seeds(plan)


@pytest.mark.parametrize("bad", [True, 1.0, "1", -1, 2 ** 63])
def test_cumulative_seed_history_validates_current_ids_too(bad):
    with pytest.raises(ValueError, match="invalid int64 ID"):
        data._source_exclusion_seeds({"splits": {"train": [{"equation_seed": bad}]}})


def test_three_generations_exclude_all_historical_ids_before_any_solve(fixture, monkeypatch):
    collect(fixture)
    original_plan = json.loads((fixture.root / "seed_plan.json").read_text())
    second = fixture.root.parent / "fresh"
    fresh(fixture)
    second_plan = json.loads((second / "seed_plan.json").read_text())
    excluded = _plan_ids(original_plan) | _plan_ids(second_plan)
    historical_only = _plan_ids(original_plan) - _plan_ids(second_plan)
    assert len(excluded) == 128 and len(historical_only) == 48
    # Collection and verification must use only the immediate source and its
    # embedded snapshots, not follow its original source_corpus.path.
    fixture.root.rename(fixture.root.parent / "original-moved")
    source_bytes = {path.relative_to(second): (path.read_bytes(), path.stat().st_mtime_ns)
                    for path in second.rglob("*") if path.is_file()}
    third = fixture.root.parent / "third"
    fixture.sampler.generated.clear()
    generate = fixture.sampler.generate

    def checked_generate(seeds):
        plan = json.loads((third / "seed_plan.json").read_text())
        partial = json.loads((third / "manifest.partial.json").read_text())
        assert plan["excluded_source_equation_seeds"] == sorted(excluded)
        assert partial["source_corpus"]["excluded_equation_seed_count"] == 128
        assert partial["seed_plan"]["sha256"] == data._file_sha256(third / "seed_plan.json")
        assert set(seeds).isdisjoint(excluded)
        return generate(seeds)

    monkeypatch.setattr(fixture.sampler, "generate", checked_generate)
    # All generations intentionally use the same master seed. Without the
    # cumulative fix the third generation replays the original held-out IDs.
    manifest = fresh(fixture, source_corpus=second, output_root=third)
    assert manifest == data.read_manifest(third)
    assert manifest["source_corpus"]["excluded_equation_seed_count"] == 128
    assert len(fixture.sampler.generated) == 6
    generated = {seed for batch in fixture.sampler.generated for seed in batch}
    assert len(generated) == 48 and generated.isdisjoint(excluded)
    assert source_bytes == {path.relative_to(second): (path.read_bytes(), path.stat().st_mtime_ns)
                            for path in second.rglob("*") if path.is_file()}
    for name in ("train", "validation"):
        for key in data._ARRAY_NAMES:
            assert (third / name / f"{key}.npy").read_bytes() == (second / name / f"{key}.npy").read_bytes()
    second.rename(fixture.root.parent / "second-moved")
    assert data.read_manifest(third, verify_hashes=False) == manifest


def test_loader_rejects_rehashed_truncated_cumulative_history(fixture):
    collect(fixture)
    original = json.loads((fixture.root / "seed_plan.json").read_text())
    second = fixture.root.parent / "fresh"
    fresh(fixture)
    second_plan = json.loads((second / "seed_plan.json").read_text())
    third = fixture.root.parent / "third"
    manifest = fresh(fixture, source_corpus=second, output_root=third)
    plan = json.loads((third / "seed_plan.json").read_text())
    dropped = next(iter(_plan_ids(original) - _plan_ids(second_plan)))
    plan["excluded_source_equation_seeds"].remove(dropped)
    # Even if count, seed-plan SHA and manifest SHA are all re-signed, the
    # immediate source snapshot proves that this historical ID was dropped.
    manifest["source_corpus"]["excluded_equation_seed_count"] -= 1
    (third / "manifest.json").write_bytes(data._canonical_bytes(manifest))
    _resign_seed_plan(third, plan)
    with pytest.raises(ValueError, match="cumulative source history"):
        data.read_manifest(third, verify_hashes=False)


@pytest.mark.parametrize("bad", [None, [True], [1.0], ["1"], [-1], [2 ** 63]])
def test_invalid_history_in_source_is_rejected_before_runtime(fixture, monkeypatch, bad):
    collect(fixture)
    fresh(fixture)
    second = fixture.root.parent / "fresh"
    plan = json.loads((second / "seed_plan.json").read_text())
    plan["excluded_source_equation_seeds"] = bad
    _resign_seed_plan(second, plan)
    fixture.sampler.generated.clear()
    monkeypatch.setattr(data, "_collection_runtime", lambda *_args: pytest.fail("must not build runtime"))
    third = fixture.root.parent / "third"
    with pytest.raises(ValueError, match="source equation seed history"):
        fresh(fixture, source_corpus=second, output_root=third)
    assert not fixture.sampler.generated
    assert not third.exists()


@pytest.mark.parametrize("mutation", ["reverse", "duplicate"])
def test_loader_requires_canonical_stored_exclusion_list(fixture, mutation):
    collect(fixture)
    fresh(fixture)
    second = fixture.root.parent / "fresh"
    plan = json.loads((second / "seed_plan.json").read_text())
    history = plan["excluded_source_equation_seeds"]
    plan["excluded_source_equation_seeds"] = history[::-1] if mutation == "reverse" else history + history[:1]
    _resign_seed_plan(second, plan)
    with pytest.raises(ValueError, match="cumulative source history"):
        data.read_manifest(second, verify_hashes=False)


def independent(fixture, **kwargs):
    return data.collect_independent_corpus(
        kwargs.pop("source_corpus", fixture.root),
        kwargs.pop("output_root", fixture.root.parent / "independent"),
        kwargs.pop("split_counts", {**fixture.counts, "train": 32}),
        kwargs.pop("master_seed", 5312026), device="cpu", **kwargs)


def test_independent_five_splits_exclude_all_history_before_first_solve(fixture, monkeypatch):
    collect(fixture)
    first_plan = json.loads((fixture.root / "seed_plan.json").read_text())
    source = fixture.root.parent / "fresh"
    fresh(fixture)
    second_plan = json.loads((source / "seed_plan.json").read_text())
    excluded = _plan_ids(first_plan) | _plan_ids(second_plan)
    before = {path.relative_to(source): (path.read_bytes(), path.stat().st_mtime_ns)
              for path in source.rglob("*") if path.is_file()}
    output = fixture.root.parent / "independent"
    fixture.sampler.generated.clear()
    generate = fixture.sampler.generate

    def checked_generate(seeds):
        plan = json.loads((output / "seed_plan.json").read_text())
        assert set(plan["splits"]) == set(data.SPLIT_NAMES)
        assert sum(map(len, plan["splits"].values())) == 96
        assert plan["reused_splits"] == []
        assert plan["excluded_source_equation_seeds"] == sorted(excluded)
        assert _plan_ids(plan).isdisjoint(excluded)
        return generate(seeds)

    monkeypatch.setattr(fixture.sampler, "generate", checked_generate)
    monkeypatch.setattr(data, "_copy_source_split", lambda *a: pytest.fail("no answer split may be copied"))
    manifest = independent(fixture, source_corpus=source)
    assert manifest == data.read_manifest(output)
    assert manifest["collection_mode"] == data._INDEPENDENT_COLLECTION_MODE
    assert manifest["split_counts"] == {**fixture.counts, "train": 32}
    provenance = manifest["source_corpus"]
    assert provenance["reused_splits"] == [] and provenance["regenerated_splits"] == list(data.SPLIT_NAMES)
    assert provenance["copy_method"] == "metadata_snapshots_only_all_splits_new"
    assert provenance["revalidated_before_publication"] is True
    assert len(fixture.sampler.generated) == 12
    assert {seed for batch in fixture.sampler.generated for seed in batch}.isdisjoint(excluded)
    assert all("collection_origin" not in manifest["splits"][name] for name in data.SPLIT_NAMES)
    assert before == {path.relative_to(source): (path.read_bytes(), path.stat().st_mtime_ns)
                      for path in source.rglob("*") if path.is_file()}
    fixture.root.rename(fixture.root.parent / "original-moved")
    source.rename(fixture.root.parent / "source-moved")
    assert data.read_manifest(output, verify_hashes=False)["manifest_sha256"] == manifest["manifest_sha256"]


def test_independent_collection_is_reproducible_and_can_feed_existing_fresh_collector(fixture):
    collect(fixture)
    first, second = fixture.root.parent / "independent", fixture.root.parent / "independent-two"
    independent(fixture)
    independent(fixture, output_root=second)
    assert (first / "seed_plan.json").read_bytes() == (second / "seed_plan.json").read_bytes()
    downstream = fixture.root.parent / "downstream"
    manifest = fresh(fixture, source_corpus=first, output_root=downstream)
    assert manifest["split_counts"]["train"] == 32
    assert manifest == data.read_manifest(downstream)
    assert (first / "train/coefficients.npy").read_bytes() == (downstream / "train/coefficients.npy").read_bytes()


@pytest.mark.parametrize("role", data.SPLIT_NAMES)
def test_independent_loader_rejects_rehashed_old_id_in_any_new_split(fixture, role):
    collect(fixture)
    independent(fixture)
    output = fixture.root.parent / "independent"
    plan = json.loads((output / "seed_plan.json").read_text())
    old = json.loads((fixture.root / "seed_plan.json").read_text())
    plan["splits"][role][0] = old["splits"]["reference_b"][0]
    _resign_seed_plan(output, plan)
    with pytest.raises(ValueError, match="overlaps an old source"):
        data.read_manifest(output, verify_hashes=False)


def test_independent_loader_rejects_shortened_history_and_false_copy_claim(fixture):
    collect(fixture)
    independent(fixture)
    output = fixture.root.parent / "independent"
    plan = json.loads((output / "seed_plan.json").read_text())
    original = list(plan["excluded_source_equation_seeds"])
    plan["excluded_source_equation_seeds"] = original[1:]
    _resign_seed_plan(output, plan)
    with pytest.raises(ValueError, match="cumulative source history"):
        data.read_manifest(output, verify_hashes=False)
    plan["excluded_source_equation_seeds"] = original
    _resign_seed_plan(output, plan)
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["splits"]["train"]["collection_origin"] = {"generated_in_this_collection": False}
    (output / "manifest.json").write_bytes(data._canonical_bytes(manifest))
    resign_manifest(output)
    with pytest.raises(ValueError, match="cannot claim a copied"):
        data.read_manifest(output, verify_hashes=False)


def test_independent_partial_failure_preserves_plan_and_never_copies_training(fixture):
    collect(fixture)
    fixture.sampler.generated.clear()
    fixture.sampler.fail_call = 2
    with pytest.raises(RuntimeError, match="deliberate"):
        independent(fixture)
    output = fixture.root.parent / "independent"
    partial = json.loads((output / "manifest.partial.json").read_text())
    assert partial["status"] == "failed" and partial["failure"]["type"] == "RuntimeError"
    assert partial["splits"]["train"]["rows_written"] == 8
    assert (output / "seed_plan.json").exists() and not (output / "validation").exists()
    assert not (output / "manifest.json").exists()


def test_independent_ignores_source_json_whitespace(fixture, monkeypatch):
    collect(fixture)
    fixture.sampler.generated.clear()
    generate = fixture.sampler.generate

    def mutate_source(seeds):
        batch = generate(seeds)
        if len(fixture.sampler.generated) == 1:
            with (fixture.root / "manifest.json").open("ab") as stream:
                stream.write(b" ")
        return batch

    monkeypatch.setattr(fixture.sampler, "generate", mutate_source)
    independent(fixture)
    assert data.read_manifest(fixture.root.parent / "independent")["status"] == "complete"
