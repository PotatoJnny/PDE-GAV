"""Tiny source/lineage and orchestration fixtures; no real assets or Slurm."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("weak_block_driver_tests", ROOT/"scripts/run_vpfp_weak_block_diagnostic.py")
driver = importlib.util.module_from_spec(spec); spec.loader.exec_module(driver)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def old_fixture():
    from tests.test_vpfp_weak_dynamics_driver import tiny_split
    rows = driver.prior.select_validation_rows(tiny_split())
    plan = {"schema": driver.prior.SCHEMA, "source_manifest_sha256": driver.prior.SOURCE_SHA,
        "source": str(driver.prior.SOURCE), "source_split": "validation", "macro_fit_grid": 33,
        "sampling_steps": 64, "master_seed": driver.prior.MASTER_SEED, "selected_roots": rows,
        "noise_plan": driver.prior.fixed_noise_plan(rows), "models": [{"manifest": x} for x in driver.prior.MODEL_MANIFESTS]}
    result = {"schema": driver.prior.SCHEMA, "execution_passed": True, "stage": "complete",
        "registered_carrier_count": 64, "completed_carrier_count": 64, "failures": [],
        "new_training": False, "pde_solves": 0, "units": {f"seed-{s}/root-{g}/{arm}": {"status": "completed"}
            for s in range(2) for g in range(8) for arm in driver.prior.ARMS}}
    return result, plan


def test_original_complete64_and_explicit_four_block_denominator():
    result, plan = old_fixture(); driver.validate_old_run(result, plan)
    units = driver.registered_units()
    assert sum(x["kind"] == "carrier" for x in units.values()) == 64
    assert sum(x["kind"] == "clean" for x in units.values()) == 8
    assert sum(x["kind"] == "comparison" for x in units.values()) == 16
    assert all(x["status"] == "pending" for x in units.values())


@pytest.mark.parametrize("bad", ["partial", "source", "seed", "manifest", "missing_unit", "root"])
def test_old_schedule_mismatch_cannot_be_reused(bad):
    result, plan = old_fixture()
    if bad == "partial": result["completed_carrier_count"] = 63
    if bad == "source": plan["source_manifest_sha256"] = "b"*64
    if bad == "seed": plan["noise_plan"][0]["micro_noise_seed"] += 1
    if bad == "manifest": plan["models"][0]["manifest"] = {"path": "/other", "sha256": "0"*64}
    if bad == "missing_unit": result["units"].pop(next(iter(result["units"])))
    if bad == "root": plan["selected_roots"][0] = {**plan["selected_roots"][0], "root_id": "other-root"}
    with pytest.raises(ValueError): driver.validate_old_run(result, plan)


def baseline_fixture():
    _, plan = old_fixture(); row = plan["noise_plan"][0]; arm = driver.OLD_ARMS[1]
    receipt = {"root_id": row["root_id"], "equation_seed": row["equation_seed"], "row": row["row"],
        "stratum_id": row["stratum_id"], "model_replica": 0, "arm": arm, "view": arm,
        "m_origin": "real", "native_manifest": plan["models"][0]["manifest"],
        "source_manifest_sha256": driver.prior.SOURCE_SHA, "condition_fp32_bits": row["condition_fp32_bits"],
        "complete_floor_preserved": True, "no_Hermite_reprojection": True, "noise_plan": row,
        "micro_conditioning_macro": "real", "carrier_sha256": "c"*64, "decoder_identity": {"fixture": True}}
    design = {"synthetic": True}
    report = {"execution_passed": True, "stage": "complete", "root_id": row["root_id"],
        "condition": row["conditions"],
        "provenance": deepcopy(receipt), "input_carrier_sha256": receipt["carrier_sha256"],
        "decoder_identity": receipt["decoder_identity"], "design": design, "timeline": [{"count": 129, "probe_count": 4}]}
    kwargs = {"seed": 0, "arm": arm, "manifest": plan["models"][0]["manifest"], "design": design}
    return receipt, report, row, kwargs


@pytest.mark.parametrize("bad", [None, "root", "condition", "m_condition", "SHA", "partial129", "design"])
def test_baseline_binding_uses_real_condition_root_decoder_and_completed129(bad):
    receipt, report, row, kwargs = baseline_fixture()
    if bad == "root": receipt["root_id"] = "another"
    if bad == "condition": receipt["condition_fp32_bits"] = [0, 0]
    if bad == "m_condition": receipt["micro_conditioning_macro"] = "generated"
    if bad == "SHA": report["input_carrier_sha256"] = "d"*64
    if bad == "partial129": report["timeline"] = [{"count": 65, "probe_count": 4}]
    if bad == "design": report["design"] = {"other": True}
    if bad is None: assert driver.validate_baseline(receipt, report, row, **kwargs)["count"] == 129
    else:
        with pytest.raises(ValueError): driver.validate_baseline(receipt, report, row, **kwargs)


def test_failure_retains_registered_pending_denominator_and_report(tmp_path):
    progress = driver.Progress(tmp_path, 60)
    name = "seed-0/root-0/"+driver.ARMS[1]
    def fail(path):
        driver.write_json(path, {"execution_passed": False, "partial_timeline": {"count": 129}})
        return {"execution_passed": False}
    assert progress.unit(name, fail) is None
    result = json.loads((tmp_path/"result.json").read_text())
    assert result["units"][name]["status"] == "failed" and result["units"][name]["report_sha256"]
    assert result["completed_carrier_count"] == 0 and len(result["units"]) == 88
    assert sum(x["status"] == "pending" for x in result["units"].values()) == 87


def test_partial_reuse_requires_same_plan_and_matching_root(tmp_path):
    prior = tmp_path/"prior"; prior.mkdir(); plan = {"fixed": "synthetic-input-code-design"}
    driver.write_json(prior/"frozen-plan.json", plan)
    name = "seed-0/root-0/"+driver.ARMS[1]
    unit = prior/name/"result.json"; unit.parent.mkdir(parents=True)
    driver.write_json(unit, {"execution_passed": True, "root_id": "root-0"})
    result = {"schema": driver.SCHEMA, "frozen_plan_sha256": driver.json_sha(plan), "execution_passed": False,
        "units": {name: {"status": "completed", "report": str(unit), "report_sha256": driver.file_sha(unit)},
                  "fail": {"status": "failed", "report": str(unit)}, "wait": {"status": "pending"}}}
    driver.write_json(prior/"result.json", result)
    reused, receipt = driver.reusable_units(prior, plan)
    assert set(reused) == {name} and receipt["failed_pending_units_not_reused"]
    with pytest.raises(ValueError, match="exact same"): driver.reusable_units(prior, {"fixed": "different"})
    current = tmp_path/"fresh"; current.mkdir(); progress = driver.Progress(current, 60); progress.reuse = reused
    assert progress.unit(name, lambda p: pytest.fail("must not recompute"), expected={"root_id": "wrong"}) is None
    driver.write_json(unit, {"execution_passed": True, "root_id": "tampered"})
    reused, _ = driver.reusable_units(prior, plan)
    progress.reuse = reused
    assert progress.unit(name, lambda p: pytest.fail("must not recompute"),
                         expected={"root_id": "root-0"}) is None


def test_guard_no_allocation_fresh_output_and_positive_budget(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(ValueError, match="compute"): driver.run(tmp_path/"output")
    monkeypatch.setenv("SLURM_JOB_ID", "123"); monkeypatch.setenv("SLURM_TMPDIR", str(tmp_path))
    monkeypatch.setattr(driver, "SCRATCH", tmp_path)
    for seconds in (0., -1., float("nan"), float("inf")):
        with pytest.raises(ValueError): driver.run(tmp_path/"output", wall_seconds=seconds)
    present = tmp_path/"present"; present.mkdir()
    with pytest.raises(ValueError, match="fresh"): driver.run(present)
    assert not (tmp_path/"output").exists()


def test_actual_tiny_load_swap_and_execute_no_sampling(tmp_path, monkeypatch):
    from tests.test_vpfp_moving_frame import tiny, constant_macro
    from pde_agent_verifier_solver.plugins.vpfp import moving_frame as mf
    from pde_agent_verifier_solver.plugins.vpfp import moving_frame_binding, weak_block_diagnostic as block
    from pde_agent_verifier_solver.plugins.vpfp.value_control_diffusion import lift_value_controls
    base, value, macro, z, condition = tiny(dtype="float32")
    m = constant_macro(macro, u=(0., 0.), covariance=((float(condition[1]), 0.), (0., float(condition[1])))).float()
    decoder = mf.MovingFrameDecoder(base, macro, value_config=value)
    full = lift_value_controls(z, value).numpy()[0]
    split = SimpleNamespace(latents=np.tile(full, (8, 1)), conditions=np.tile(condition.numpy(), (8, 1)),
        stratum_ids=np.arange(8, dtype=np.int64), case_ids=np.asarray([f"synthetic-{i}" for i in range(8)]),
        sample_ids=np.arange(100, 108, dtype=np.int64))
    rows = driver.prior.select_validation_rows(split); noise = driver.prior.fixed_noise_plan(rows)
    inputs = tmp_path/"inputs"; inputs.mkdir(); entries = []
    real = {"m": m, "z": z}
    generated = {"m": m.clone(), "z": z.clone()}
    generated["z"][:, :value.rho_dimension] = .015
    generated["z"][:, value.rho_dimension:] += .01
    timelines = {}
    for arm, carrier in zip(driver.OLD_ARMS, (real, generated)):
        report = block.measure_block_carrier(decoder, carrier, condition, output=None, root_id="synthetic",
            provenance={"data_origin": "synthetic_fixture"}, design=block.block_design(control_count=3))
        assert report["execution_passed"], report.get("error")
        timelines[arm] = report["timeline"]
    for row in noise:
        entry = {"row": row, "sources": {}}
        for arm, carrier in zip(driver.OLD_ARMS, (real, generated)):
            directory = inputs/f'seed-{row["replica"]}-root-{row["stratum_id"]}-{arm}'
            receipt = driver.prior.save_carrier(directory, decoder, carrier, condition, {"synthetic": True})
            saved = directory/"dynamics.json"
            driver.write_json(saved, {"execution_passed": True, "timeline": timelines[arm], "synthetic_fixture": True})
            entry["sources"][arm] = {"receipt": receipt, "receipt_path": str(directory/"carrier-receipt.json"),
                "fixture_arm": arm, "report_path": str(saved), "report_sha256": driver.file_sha(saved)}
        entries.append(entry)
    plan = {"design": block.block_design(control_count=3), "models": [{"manifest": {"path": "synthetic", "sha256": "0"*64}}]*2,
        "entries": entries, "frozen_files": {}, "selected_roots": rows}
    def forbidden(*args, **kwargs): raise AssertionError("new model sampling forbidden")
    sampler = SimpleNamespace(identity={"corpus_manifest_sha256": driver.prior.SOURCE_SHA}, sample=forbidden)
    monkeypatch.setattr(driver, "freeze_inputs", lambda check: (plan, split))
    monkeypatch.setattr(moving_frame_binding, "load_moving_frame_sampler", lambda *a, **k: (sampler, decoder))
    monkeypatch.setattr(driver, "reused_timeline", lambda source: timelines[source["fixture_arm"]])
    deep_calls = []
    def deep(*args, **kwargs):
        deep_calls.append(kwargs)
        return {"execution_passed": True, "synthetic_deep_fixture": True}
    monkeypatch.setattr(block, "audit_independent_physics", deep)
    monkeypatch.setattr(driver, "SCRATCH", tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "123"); monkeypatch.setenv("SLURM_TMPDIR", str(tmp_path))
    result = driver.run(tmp_path/"output", device="cpu", wall_seconds=300)
    assert result["execution_passed"], result.get("failure", result.get("failures"))
    assert result["completed_carrier_count"] == 64 and result["completed_clean_count"] == 8
    assert result["completed_comparison_count"] == 16 and len(deep_calls) == 4
    assert not result["new_sampling"] and result["new_PDE_solves"] == 0
    for kwargs in deep_calls: assert kwargs["points_txy"].shape == (8, 3)
    assert all((tmp_path/"output"/f"clean/root-{i}/result.json").is_file() for i in range(8))
