"""Synthetic receipt gate tests; fake receipts are not scientific evidence."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts import plasma_gpu_campaign as campaign
from scripts import prepare_plasma_transport_v2 as migration
from pde_agent_verifier_solver.engine import checkpoint_store
from pde_agent_verifier_solver.sdk import inspect_candidate_bundle
from tests.test_plasma_gpu_campaign import config
from tests.test_plasma_transport_scoring import _calibration


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint_store, "source_identity", lambda root: "a" * 64)
    root, sources = tmp_path / "validation", tmp_path / "sources"
    calibration = _calibration()
    references = {}
    for family in migration.FAMILIES:
        for arm in ("sol", "astra"):
            write(sources / f"{family}-{arm}.json", config(family, arm))
        references[family] = campaign.file_sha(sources / f"{family}-sol.json")
    write(root / "calibration.json", calibration.to_dict())
    cal_sha = campaign.file_sha(root / "calibration.json")
    report = {"schema": "plasma_transport_v2_validation_pipeline_v1", "status": "passed",
              "source_identity": "a" * 64, "objective_hash": calibration.objective_hash,
              "calibration_sha256": cal_sha, "reference_config_hashes": references,
              "stages": [{"name": name, "status": "passed", "exit_code": 0}
                         for name in migration.VALIDATION_STAGES]}
    write(root / "validation.json", report)
    for family in migration.FAMILIES:
        bundle = migration.ROOT / "configs/candidates" / (
            "artemis_hold_initial_bundle" if family == "artemis" else "vpfp_transport_simple_bundle")
        write(root / f"acceptance-{family}" / "report.json", {
            "schema": "plasma_macro_transport_v2_native_acceptance_v1", "family": family,
            "status": "passed", "private_cases_released": True,
            "config_sha256": references[family], "config_path": str(sources / f"{family}-sol.json"),
            "calibration_file_sha256": cal_sha, "objective_identity": calibration.objective_identity,
            "candidate_bundle_sha256": inspect_candidate_bundle(bundle).tree_sha256,
            "candidate_audits": [{"status": "completed", "process_group_cleaned": True}] * 2,
        })
    qualification = migration.validation_qualification(root, calibration, reference_config_hashes=references)
    old = config("strong", "astra")
    fresh = migration.transform(old, name="strong-astra", source_path=sources / "strong-astra.json",
                                calibration=calibration, run_tag="unit-only")
    fresh["transport_qualification"] = deepcopy(qualification)
    return root, sources, calibration, qualification, fresh


def test_v2_static_check_requires_real_receipt_even_if_config_is_valid(tmp_path):
    path = tmp_path / "source.json"
    write(path, config("strong"))
    fresh = migration.transform(config("strong"), name="strong-sol", source_path=path,
                                calibration=_calibration(), run_tag="unit-only")
    assert campaign.validate_config(fresh) is fresh
    with pytest.raises(ValueError, match="passed native validation"):
        campaign.static_check(fresh)


def test_consistent_receipts_bind_both_model_arms_to_same_family_science(evidence):
    root, sources, cal, qualification, fresh = evidence
    assert migration.check_transport_qualification(fresh) == qualification
    assert set(qualification["acceptance_sha256"]) == set(migration.FAMILIES)
    assert qualification["source_identity"] == "a" * 64


@pytest.mark.parametrize("change", ["pipeline_failed", "missing_stage", "wrong_source", "wrong_calibration",
                                   "acceptance_failed", "cleanup_failed", "wrong_reference", "receipt_changed"])
def test_invalid_or_mutated_validation_is_not_admitted(evidence, change):
    root, sources, cal, qualification, fresh = evidence
    path = root / ("acceptance-weak/report.json" if change in (
        "acceptance_failed", "cleanup_failed", "wrong_reference", "receipt_changed") else "validation.json")
    row = json.loads(path.read_text())
    if change == "pipeline_failed":
        row["status"] = "failed"
    elif change == "missing_stage":
        row["stages"].pop()
    elif change == "wrong_source":
        row["source_identity"] = "b" * 64
    elif change == "wrong_calibration":
        row["calibration_sha256"] = "b" * 64
    elif change == "acceptance_failed":
        row["status"] = "failed"
    elif change == "cleanup_failed":
        row["candidate_audits"][0]["process_group_cleaned"] = False
    elif change == "wrong_reference":
        row["config_sha256"] = "b" * 64
    else:
        row["extra_change_after_acceptance"] = True
    write(path, row)
    with pytest.raises(ValueError):
        migration.check_transport_qualification(fresh)


def test_changed_scientific_backend_or_unaccepted_s0_rejected(evidence):
    root, sources, cal, qualification, fresh = evidence
    changed = deepcopy(fresh)
    changed["backend"]["changed_after_acceptance"] = True
    with pytest.raises(ValueError, match="model assets"):
        migration.check_transport_qualification(changed)
    changed = deepcopy(fresh)
    changed["candidate"]["initial_bundle"] = str(migration.ROOT / "configs/candidates/artemis_hold_initial_bundle")
    with pytest.raises(ValueError, match="S0 differs"):
        migration.check_transport_qualification(changed)


def test_prepare_records_gate_for_all_six_without_submitting(evidence, tmp_path, monkeypatch):
    root, sources, cal, qualification, fresh = evidence
    monkeypatch.setattr(campaign, "static_check", lambda cfg: {
        "qualification": migration.check_transport_qualification(cfg), "status": "synthetic_static_check"})
    output = tmp_path / "prepared"
    receipt = migration.prepare(sources, root / "calibration.json", output,
                                run_tag="unit-only", validation_root=root)
    assert receipt["formal_jobs_submitted"] is receipt["old_jobs_modified"] is False
    assert receipt["transport_qualification"] == qualification and len(receipt["arms"]) == 6
    for arm in receipt["arms"]:
        assert json.loads(Path(arm["config"]).read_text())["transport_qualification"] == qualification


def test_prepare_refuses_reference_family_config_swap(evidence, tmp_path):
    root, sources, cal, qualification, fresh = evidence
    original = json.loads((sources / "weak-sol.json").read_text())
    original["campaign_id"] = "different-reference"
    write(sources / "weak-sol.json", original)
    with pytest.raises(ValueError, match="reference configurations"):
        migration.prepare(sources, root / "calibration.json", tmp_path / "never-created",
                          run_tag="unit-only", validation_root=root)
    assert not (tmp_path / "never-created").exists()
