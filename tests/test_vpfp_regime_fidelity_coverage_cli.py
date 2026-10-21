"""Mock/tiny file checks only; no real array audit or physical qualification."""
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def driver():
    scripts = Path(__file__).resolve().parents[1]/"scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location("regime_coverage_cli_test", scripts/"run_vpfp_regime_fidelity_coverage.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def native_files(tmp_path, driver):
    directory = tmp_path/"generated"; directory.mkdir()
    cfg = SimpleNamespace(collision_nu_range=(0., 1.), temperature_range=(.8, 1.2))
    conditions = driver.balanced_conditions(cfg, 128, 123)
    raw = np.arange(128*3, dtype=np.float32).reshape(128, 3)
    np.save(directory/"generated-coefficients.npy", raw)
    np.save(directory/"generated-conditions.npy", conditions)
    checkpoint_sha, manifest_sha = "a"*64, "b"*64
    seeds = list(range(128))
    provenance = {"checkpoint_sha256": checkpoint_sha, "noise_seeds": seeds, "condition_seed": 123,
        "coefficient_sha256": driver.file_sha(directory/"generated-coefficients.npy")}
    roots = [f"generated:{checkpoint_sha}:{s}" for s in seeds]
    hashes = [driver._input_hash(row, condition) for row, condition in zip(raw, conditions)]
    (directory/"generated-provenance.json").write_text(json.dumps(provenance))
    (directory/"generated-physical.json").write_text(json.dumps({"root_ids": roots, "answer_sha256": hashes}))
    (directory/"physical-distribution-audit.json").write_text(json.dumps({"checkpoint_sha256": checkpoint_sha,
        "corpus_manifest_sha256": manifest_sha}))
    return directory, cfg, {"latent_dimension": 3, "manifest_sha256": manifest_sha}


def freeze(driver, fixture, **kwargs):
    directory, spec, manifest = fixture
    return driver.freeze_generated(directory, spec=spec, manifest=manifest, count=kwargs.get("count", 128),
        edges=driver._edges(np.linspace(0., 1., 5), (.8, 1., 1.2)))


def test_native_receipt_replays_conditions_and_answer_identity(driver, native_files):
    result = freeze(driver, native_files)
    assert result["status"] == "ready_native_direct"
    assert result["condition_seed_replayed"]
    assert len(result["selected_hashes"]) == 128
    driver.check_frozen(result["files"])


def test_file_mutation_fails_before_measurement(driver, native_files):
    result = freeze(driver, native_files)
    directory = native_files[0]
    with (directory/"generated-conditions.npy").open("ab") as stream: stream.write(b"mutation")
    with pytest.raises(ValueError, match="changed"):
        driver.check_frozen(result["files"])


def test_wrong_condition_seed_and_wrong_native_hash_are_refused(driver, native_files):
    directory = native_files[0]
    path = directory/"generated-provenance.json"
    receipt = json.loads(path.read_text()); receipt["condition_seed"] += 1
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="conditions do not replay"):
        freeze(driver, native_files)
    receipt["condition_seed"] -= 1; path.write_text(json.dumps(receipt))
    path = directory/"generated-physical.json"
    physical = json.loads(path.read_text()); physical["answer_sha256"][0] = "f"*64
    path.write_text(json.dumps(physical))
    with pytest.raises(ValueError, match="input hashes"):
        freeze(driver, native_files)


def test_no_512_confirmation_shrink(driver, native_files):
    with pytest.raises(ValueError, match="lacks 64 frozen attempts"):
        freeze(driver, native_files, count=512)


def test_real_strong_nested_design_receipt_format(driver, native_files):
    directory = native_files[0]
    path = directory/"generated-provenance.json"
    receipt = json.loads(path.read_text())
    seeds = receipt.pop("noise_seeds"); receipt.pop("condition_seed")
    receipt["known_initial_seed_applied"] = True
    receipt["design"] = {"sampling_seeds": seeds, "known_initial_seeds": list(range(1000, 1128)),
        "conditions": np.load(directory/"generated-conditions.npy").tolist()}
    path.write_text(json.dumps(receipt))
    result = freeze(driver, native_files)
    assert not result["condition_seed_replayed"]
    assert result["conditions_recorded_design_replayed"]
    assert result["known_initial_seeds"] == list(range(1000, 1128))
    receipt["design"]["conditions"][0][0] += .1
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="known-initial design"):
        freeze(driver, native_files)


def test_incomplete_moving_frame_receipt_is_refused_not_z_only_pass(driver, native_files):
    directory = native_files[0]
    np.save(directory/"generated-macro.npy", np.zeros((128, 4), dtype=np.float32))
    path = directory/"generated-provenance.json"
    receipt = json.loads(path.read_text()); receipt["macro_sha256"] = driver.file_sha(directory/"generated-macro.npy")
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="complete versioned m/z"):
        freeze(driver, native_files)


def test_repeated_explicit_targets_and_registered_count_parser(driver):
    args = driver.parser().parse_args(["--source", "/source", "--generated-dir", "/a", "--generated-dir", "/b", "--output", "/out"])
    assert args.generated_dir == [Path("/a"), Path("/b")]
    assert args.sample_count == 128 and args.bootstrap_repeats == 49
    with pytest.raises(SystemExit):
        driver.parser().parse_args(["--source", "/source", "--generated-dir", "/a", "--output", "/out", "--sample-count", "64"])


def test_json_receipt_comparison_preserves_config_values_not_python_sequence_type(driver):
    native = {"full_config":{"collision_nu_range":(.02,.08),"temporal_dilation_cycle":(1,2,4,9)}}
    receipt = json.loads(json.dumps(native))
    assert driver._same_json_value(receipt,native)
    receipt["full_config"]["temporal_dilation_cycle"][2] = 5
    assert not driver._same_json_value(receipt,native)
    with pytest.raises(ValueError):
        driver._same_json_value({"value":float("nan")},{"value":0.})


def test_separate_model_and_reference_sources_parser(driver):
    args = driver.parser().parse_args(["--model-source", "/model", "--reference-source", "/fresh",
        "--generated-dir", "/a", "--output", "/out", "--sample-count", "512"])
    assert args.model_source == Path("/model") and args.reference_source == Path("/fresh")
    assert args.source is None and args.sample_count == 512
    with pytest.raises(SystemExit):
        driver.parser().parse_args(["--source", "/old", "--model-source", "/model",
            "--generated-dir", "/a", "--output", "/out"])


def test_new_reference_roots_exclude_training_and_cumulative_history(driver, tmp_path):
    def corpus(name, roots, history):
        root = tmp_path/name; root.mkdir()
        plan = {"excluded_source_equation_seeds":history,
            "splits":{role:[{"equation_seed":seed}] for role,seed in zip(
                ("train","validation","calibration","reference_a","reference_b"),roots)}}
        path = root/"seeds.json"; path.write_text(json.dumps(plan))
        manifest = {"seed_plan":{"path":"seeds.json","sha256":driver.file_sha(path)},
            "model_spec":{"physical":"same"},"base_decoder_sha256":"b"*64,"latent_dimension":3,"regime_id":"weak"}
        return root,manifest
    model,m = corpus("old",[1,2,3,4,5],[99])
    ref,r = corpus("fresh",[1,2,11,12,13],[1,2,3,4,5,99])
    assert driver.validate_source_pair(model,m,ref,r) == {1,2,3,4,5,11,12,13,99}
    assert driver.validate_source_pair(model,m,model,m) == {1,2,3,4,5,99}
    bad,b = corpus("bad",[1,2,11,99,13],[])
    with pytest.raises(ValueError,match="historical roots"):
        driver.validate_source_pair(model,m,bad,b)
    r["latent_dimension"] = 4
    with pytest.raises(ValueError,match="physical ABI"):
        driver.validate_source_pair(model,m,ref,r)


@pytest.fixture
def independent_files(driver, tmp_path, monkeypatch):
    """File/receipt fixture with an EXPLICIT mocked real-loader identity.

    The bytes below are not represented as a scientifically valid checkpoint;
    production always calls the complete native loader. No sampling is mocked
    as actual generation or physical validation.
    """
    directory = tmp_path/"independent"; directory.mkdir()
    spec = SimpleNamespace(collision_nu_range=(0.,1.),temperature_range=(.8,1.2),time_domain=(0.,1.),base_decoder_sha256="c"*64)
    manifest = {"latent_dimension":3,"manifest_sha256":"a"*64}
    checkpoint = tmp_path/"native.pt"; checkpoint.write_bytes(b"explicitly mocked native-loader test input")
    identity = {"checkpoint_schema":driver.INITIAL_CHECKPOINT_SCHEMA,
        "base_decoder_sha256":spec.base_decoder_sha256,"checkpoint_sha256":driver.file_sha(checkpoint),
        "corpus_manifest_sha256":manifest["manifest_sha256"],"latent_dimension":3,
        **{name:list(getattr(spec,name)) for name in ("collision_nu_range","temperature_range","time_domain")}}
    loaded = []
    def loader(path, digest, *, device, steps, churn):
        assert Path(path) == checkpoint and digest == driver.file_sha(checkpoint)
        assert (device,steps,churn) == ("cpu",64,0.)
        loaded.append((path,digest))
        return SimpleNamespace(identity=identity)
    monkeypatch.setattr(driver,"load_vpfp_diffusion_sampler",loader)
    raw = np.zeros((512,3),dtype=np.float32); raw[7] = np.nan
    conditions = driver.balanced_conditions(spec,512,123)
    np.save(directory/"generated-coefficients.npy",raw)
    np.save(directory/"generated-conditions.npy",conditions)
    receipt = {"schema":"vpfp_independent_native_generation_v1","attempted":512,"completed_attempts":512,
        "failed_attempts":1,"successful_attempts":511,
        "sampler_steps":64,"checkpoint":{"path":str(checkpoint),"sha256":driver.file_sha(checkpoint)},
        "checkpoint_sha256":driver.file_sha(checkpoint),"model_source_manifest_sha256":manifest["manifest_sha256"],
        "coefficients_sha256":driver.file_sha(directory/"generated-coefficients.npy"),
        "conditions_sha256":driver.file_sha(directory/"generated-conditions.npy"),
        "seeds":list(range(1000,1512)),"condition_seeds":list(range(2000,2512)),
        "future_noise_seeds":driver._native_stream("vpfp-initial-conditioned-future",list(range(1000,1512))),
        "known_initial_seeds":driver._native_stream("vpfp-initial-conditioned-prior",list(range(1000,1512))),
        "failures":[{"attempt":7,"seed":1007,"reason":"synthetic failure retained"}]}
    path = directory/"generated-provenance.json"; path.write_text(json.dumps(receipt))
    return SimpleNamespace(fixture=(directory,spec,manifest),directory=directory,spec=spec,manifest=manifest,
        receipt=receipt,receipt_path=path,identity=identity,loaded=loaded,checkpoint=checkpoint,raw=raw)


def test_independent_native_receipt_does_not_require_old_audit(driver, independent_files):
    f = independent_files
    directory,spec,manifest = f.fixture
    receipt,path = f.receipt,f.receipt_path
    result = freeze(driver,(directory,spec,manifest),count=512)
    assert len(result["selected_rows"]) == 512 and 7 in result["selected_rows"]
    assert len(result["independent_lineage_seeds"]) == 2048
    assert result["known_initial_seeds"] == receipt["known_initial_seeds"]
    assert result["recorded_generation_failures"] == receipt["failures"]
    assert len(f.loaded) == 1
    assert not (directory/"generated-physical.json").exists()
    receipt["completed_attempts"] = 511; path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match="all 512 attempts"):
        freeze(driver,(directory,spec,manifest),count=512)


@pytest.mark.parametrize("key", ["known_initial_seeds", "future_noise_seeds"])
def test_independent_initial_streams_are_required_and_exact_native_derivations(driver, independent_files, key):
    f = independent_files
    values = f.receipt.pop(key)
    f.receipt_path.write_text(json.dumps(f.receipt))
    with pytest.raises(ValueError,match="ancestry|SHA derivation"):
        freeze(driver,f.fixture,count=512)
    f.receipt[key] = values.copy(); f.receipt[key][0] += 1
    f.receipt_path.write_text(json.dumps(f.receipt))
    with pytest.raises(ValueError,match="SHA derivation"):
        freeze(driver,f.fixture,count=512)


@pytest.mark.parametrize("field,bad", [("latent_dimension",4),("time_domain",[0.,2.]),
    ("collision_nu_range",[.01,1.]),("base_decoder_sha256","f"*64),("corpus_manifest_sha256","e"*64)])
def test_loaded_native_identity_must_match_source_config(driver, independent_files, field, bad):
    f = independent_files; f.identity[field] = bad
    with pytest.raises(ValueError,match="model-source ABI"):
        freeze(driver,f.fixture,count=512)


def test_invalid_native_schema_reaches_and_is_rejected_by_real_loader(driver, independent_files, monkeypatch):
    from pde_agent_verifier_solver.plugins.vpfp.structured_diffusion import load_vpfp_diffusion_sampler
    f = independent_files
    # A deliberately invalid seven-field file proves production validation is
    # not a mere header-key check. This is synthetic rejection, not asset proof.
    driver.torch.save({"schema":"synthetic_header_fixture","config":{},"model":{},"normalizer":{},
        "training":{},"base_decoder_sha256":f.spec.base_decoder_sha256,
        "corpus_manifest_sha256":f.manifest["manifest_sha256"]},f.checkpoint)
    f.receipt["checkpoint"]["sha256"] = f.receipt["checkpoint_sha256"] = driver.file_sha(f.checkpoint)
    f.receipt_path.write_text(json.dumps(f.receipt))
    monkeypatch.setattr(driver,"load_vpfp_diffusion_sampler",load_vpfp_diffusion_sampler)
    with pytest.raises(ValueError,match="unexpected VPFP diffusion checkpoint schema"):
        freeze(driver,f.fixture,count=512)


@pytest.mark.parametrize("mutation", ["count","success","duplicate","wrong_seed","out_of_bounds","missing_failure","finite_failed_row","unlisted_nan","infinity","partial_nan"])
def test_all512_failure_counts_and_actual_raw_rows_are_checked(driver, independent_files, mutation):
    f = independent_files
    if mutation == "count": f.receipt["failed_attempts"] = 0
    elif mutation == "success": f.receipt["successful_attempts"] = 512
    elif mutation == "duplicate": f.receipt["failures"] *= 2
    elif mutation == "wrong_seed": f.receipt["failures"][0]["seed"] += 1
    elif mutation == "out_of_bounds": f.receipt["failures"][0]["attempt"] = 512
    elif mutation == "missing_failure":
        f.receipt.update(failures=[],failed_attempts=0,successful_attempts=512)
    else:
        if mutation == "finite_failed_row": f.raw[7] = 0.
        elif mutation == "unlisted_nan": f.raw[8] = np.nan
        elif mutation == "infinity": f.raw[7] = np.inf
        elif mutation == "partial_nan": f.raw[7,0] = 0.
        np.save(f.directory/"generated-coefficients.npy",f.raw)
        f.receipt["coefficients_sha256"] = driver.file_sha(f.directory/"generated-coefficients.npy")
    f.receipt_path.write_text(json.dumps(f.receipt))
    with pytest.raises(ValueError,match="failure|counts"):
        freeze(driver,f.fixture,count=512)


def test_new512_receipt_cannot_be_subsampled_to128(driver, independent_files):
    with pytest.raises(ValueError,match="all 512 attempts"):
        freeze(driver,independent_files.fixture,count=128)


def test_independent_seed_aliases_cannot_relabel_the_canonical_attempts(driver, independent_files):
    f = independent_files
    f.receipt["noise_seeds"] = list(range(5000,5512))
    f.receipt_path.write_text(json.dumps(f.receipt))
    with pytest.raises(ValueError,match="seed aliases"):
        freeze(driver,f.fixture,count=512)


def test_new_source_marker_cannot_enter_unvalidated_old_W0_branch(driver, independent_files):
    f = independent_files
    f.receipt.update(schema="vpfp_coherent_generated_native_pair_v1",decoder_views=["old_full_decoder"])
    f.receipt_path.write_text(json.dumps(f.receipt))
    with pytest.raises(ValueError,match="source marker"):
        freeze(driver,f.fixture,count=512)


def test_new_direct_sampler_noise_is_the_actual_root_not_prior_stream(driver, independent_files):
    f = independent_files
    f.identity["checkpoint_schema"] = "vpfp_temporal_covariance_diffusion_checkpoint_v1"
    f.receipt.pop("known_initial_seeds")
    f.receipt["future_noise_seeds"] = f.receipt["seeds"]
    f.receipt_path.write_text(json.dumps(f.receipt))
    result = freeze(driver,f.fixture,count=512)
    assert result["known_initial_seeds"] is None
    f.receipt["future_noise_seeds"] = driver._native_stream("vpfp-initial-conditioned-future",f.receipt["seeds"])
    f.receipt_path.write_text(json.dumps(f.receipt))
    with pytest.raises(ValueError,match="SHA derivation"):
        freeze(driver,f.fixture,count=512)


def test_new_paired512_receipt_checks_shared_failure_mask_and_native_streams(driver, independent_files, monkeypatch):
    from pde_agent_verifier_solver.plugins.vpfp import moving_frame_binding
    f = independent_files
    macro = np.zeros((512,4),dtype=np.float32); macro[7] = np.nan
    macro_path = f.directory/"generated-macro.npy"; np.save(macro_path,macro)
    f.receipt.pop("known_initial_seeds")
    f.receipt.update(schema="vpfp_coherent_generated_native_pair_v1",
        decoder_views=["B_old_full_decoder","A_continuous_moving_frame"],full_coefficients_are_A_answer=False,
        macro_sha256=driver.file_sha(macro_path), macro_config={"fixture":"macro"},micro_config={"fixture":"value"},
        checkpoint_bindings={role:{"checkpoint":str(f.checkpoint),"checkpoint_sha256":driver.file_sha(f.checkpoint)}
                            for role in ("macro","conditional")},
        macro_noise_seeds=driver._native_stream("vpfp-coherent-macro",f.receipt["seeds"]),
        micro_noise_seeds=driver._native_stream("vpfp-coherent-micro",f.receipt["seeds"]))
    f.receipt["future_noise_seeds"] = f.receipt["micro_noise_seeds"]
    monkeypatch.setattr(driver.torch,"load",lambda *args,**kwargs:{"corpus_manifest_sha256":f.manifest["manifest_sha256"]})
    monkeypatch.setattr(moving_frame_binding,"validate_hierarchical_payloads",lambda *args,**kwargs:(
        SimpleNamespace(latent_dimension=4,to_dict=lambda:{"fixture":"macro"}),
        SimpleNamespace(full_dimension=3,to_dict=lambda:{"fixture":"value"})))
    f.receipt_path.write_text(json.dumps(f.receipt))
    result = freeze(driver,f.fixture,count=512)
    assert result["status"] == "ready_paired_A_B" and len(result["selected_rows"]) == 512
    f.receipt["sampler_steps"] = 32; f.receipt_path.write_text(json.dumps(f.receipt))
    with pytest.raises(ValueError,match="all 512 attempts"):
        freeze(driver,f.fixture,count=512)
    f.receipt["sampler_steps"] = 64
    macro[7] = 0.; np.save(macro_path,macro); f.receipt["macro_sha256"] = driver.file_sha(macro_path)
    f.receipt_path.write_text(json.dumps(f.receipt))
    with pytest.raises(ValueError,match="macro NaN failure rows"):
        freeze(driver,f.fixture,count=512)
    macro[7] = np.nan; np.save(macro_path,macro); f.receipt["macro_sha256"] = driver.file_sha(macro_path)
    f.receipt["macro_noise_seeds"][0] += 1; f.receipt_path.write_text(json.dumps(f.receipt))
    with pytest.raises(ValueError,match="SHA derivation"):
        freeze(driver,f.fixture,count=512)


def test_independent_lineage_does_not_hide_incomplete_or_repeated_stream(driver):
    receipt = {"model_source_manifest_sha256":"a"*64,"seeds":[1,2],"condition_seeds":[3,4],"future_noise_seeds":[1,2]}
    assert driver._independent_lineage(receipt,2) == [1,2,3,4]
    receipt["condition_seeds"] = [3,3]
    with pytest.raises(ValueError,match="ancestry"):
        driver._independent_lineage(receipt,2)
    receipt.pop("condition_seeds")
    with pytest.raises(ValueError,match="omits ancestry"):
        driver._independent_lineage(receipt,2)


def test_no_login_node_extraction(driver, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(ValueError, match="compute allocation"):
        driver.run(SimpleNamespace())


def test_failure_report_saved_before_raise(driver, monkeypatch, tmp_path):
    # Mock only path placement, not the failure-report branch. No source read.
    source, target = tmp_path/"source", tmp_path/"target"
    source.mkdir(); target.mkdir(); (source/"manifest.json").write_text("{}")
    output = tmp_path/"output"
    monkeypatch.setenv("SLURM_JOB_ID", "synthetic-test-only")
    original = driver.Path
    class TestPath(type(original())):
        @property
        def parents(self):
            value = super().parents
            return (*value, original("/scratch")) if self == output else value
    monkeypatch.setattr(driver, "Path", TestPath)
    def fail(*args, **kwargs): raise ValueError("synthetic corpus identity failure")
    monkeypatch.setattr(driver, "read_manifest", fail)
    with pytest.raises(ValueError, match="synthetic corpus identity failure"):
        driver.run(SimpleNamespace(source=source, generated_dir=[target], output=output, sample_count=128))
    result = json.loads((output/"result.json").read_text())
    assert not result["execution_passed"]
    assert result["stage"] == "freeze_inputs"
    assert "synthetic corpus identity failure" in result["failure"]["traceback"]
    index = json.loads((output/"regime-fidelity-coverage.json").read_text())
    assert not index["execution_passed"] and not index["admitted"]
    assert index["model_source_manifest_sha256"] is None
    assert index["targets"]["target-00"]["status"] == "not_started"


def _index_result():
    return {"execution_passed":True,"stage":"complete","model_source_manifest_sha256":"a"*64,
        "reference_source_manifest_sha256":"b"*64,"targets":{
            "target-00":{"directory":"/frozen/native","status":"descriptive_audit_completed","audit_path":"/out/native-audit.json"},
            "target-01":{"directory":"/frozen/pair","status":"descriptive_audit_completed",
                "A_audit_path":"/out/pair-A-audit.json","B_audit_path":"/out/pair-B-audit.json",
                "paired_comparison_path":"/out/pair-comparison.json"}}}


def test_audit_index_lists_explicit_AB_paths_and_keeps_source_identities_separate(driver,tmp_path):
    result = _index_result()
    driver._publish_audit_result(tmp_path,result)
    index = json.loads((tmp_path/"regime-fidelity-coverage.json").read_text())
    assert index["execution_passed"] and not index["admitted"]
    assert index["model_source_manifest_sha256"] != index["reference_source_manifest_sha256"]
    assert index["targets"]["target-00"]["A_audit_path"] is None
    assert index["targets"]["target-00"]["B_audit_path"] == "/out/native-audit.json"
    assert index["targets"]["target-01"]["A_audit_path"] == "/out/pair-A-audit.json"
    result["execution_passed"] = False
    result["targets"]["target-01"].update(status="B_completed_A_pending",A_audit_path=None)
    driver._publish_audit_result(tmp_path,result)
    index = json.loads((tmp_path/"regime-fidelity-coverage.json").read_text())
    assert not index["execution_passed"] and not index["targets"]["target-01"]["execution_passed"]


def test_index_publication_failure_retains_directory_result(driver,tmp_path,monkeypatch):
    original = driver.write_json
    def fail_index(path,value):
        if path.name == "regime-fidelity-coverage.json": raise OSError("synthetic index disk error")
        original(path,value)
    monkeypatch.setattr(driver,"write_json",fail_index)
    with pytest.raises(OSError,match="index disk error"):
        driver._publish_audit_result(tmp_path,_index_result())
    saved = json.loads((tmp_path/"result.json").read_text())
    assert not saved["execution_passed"] and saved["stage"] == "publish_audit_index_failed"
    assert saved["index_publication_failure"]["type"] == "OSError"
    assert len(saved["targets"]) == 2
