"""Synthetic bookkeeping controls for the real two-round integration check."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("vpfp_iteration_check", ROOT / "scripts/check_vpfp_iteration_execution.py")
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)
SOLVER = "c"*64


def _evidence():
    training, sampling, manifests, agents = [], [], [], []
    for index in range(3):
        rows = [{"case_id": f"base-{index}-{j}", "root_seed": 1000*index+j,
            "source": "base", "source_round_index": index, "target_solver_sha256": SOLVER,
            "split": "fit" if j < 64 else "validation"} for j in range(80)]
        sampling.append({"source": "base", "count": 80, "round_index": index,
            "purpose": "guidance_feedback" if index == 0 else "promotion",
            "case_ids": [row["case_id"] for row in rows],
            "root_seeds": [row["root_seed"] for row in rows], "snapshot_sha256": None})
        if index == 2:
            continue
        manifests.append({"rows": rows})
        snapshot = ("a" if index == 0 else "b")*64
        training.append({"source_round_index": index, "target_round_index": index+1,
            "target_solver_sha256": SOLVER, "target_solver_id": "S0",
            "fresh_model_initialization": True, "feedback_case_count": 80,
            "clean_base_case_count": 80, "excluded": [], "snapshot_sha256": snapshot,
            "previous_snapshot_sha256": None if index == 0 else "a"*64,
            "report": {"execution_passed": True, "final_step": 2000, "target_step": 2000,
                "completed_step": 2000, "base_final_verified": True, "feedback_final_verified": True,
                "label_source": "actual_public_solver_macro_error"}})
        sampling.append({"source": "guided", "count": 1, "round_index": index+1,
            "purpose": "development", "case_ids": [f"guided-{index}"],
            "root_seeds": [10000+index], "snapshot_sha256": snapshot})
        agents.append({"round_index": index+1, "nominations": 0, "base_count": 1,
            "guided_count": 1, "perturbed_count": 1, "truth_received": False,
            "source_or_g0_prematerialized": False})
    candidates = [{"execution_id": "batch", "case_index": i, "status": "completed",
        "return_code": 0, "error_code": None, "process_group_cleaned": True,
        "candidate_process_isolated": True, "controller_truth_shared": False,
        "phase": "official" if i < 240 else "development" if i < 246 else "terminal"}
        for i in range(249)]
    return training, sampling, manifests, candidates, agents


def test_complete_real_evidence_shape_passes_without_claiming_llm_or_solver_improvement():
    evidence = _evidence()
    report = CHECK.verify_receipts(*evidence, selected_sha=SOLVER)
    assert report["guidance_rounds"] == 2 and report["actual_feedback_roots"] == 160
    assert report["isolated_candidate_executions"] == 249
    assert not report["llm_invoked"] and not report["solver_improvement_claimed"]


@pytest.mark.parametrize("failure", ["not_retrained", "wrong_solver", "wrong_round", "wrong_predecessor",
    "partial_training", "synthetic_labels", "missing_official", "reused_root", "wrong_draw",
    "wrong_feedback_phase", "leaked_guided", "wrong_snapshot", "wrong_split", "truth", "duplicate_audit",
    "missing_execution", "failed_worker", "missing_cleanup", "agent_nomination"])
def test_bad_or_incomplete_evidence_is_not_accepted(failure):
    train, samples, manifests, audits, agents = _evidence()
    if failure == "not_retrained": train[1]["snapshot_sha256"] = train[0]["snapshot_sha256"]
    elif failure == "wrong_solver": train[1]["target_solver_sha256"] = "d"*64
    elif failure == "wrong_round": train[1]["target_round_index"] = 3
    elif failure == "wrong_predecessor": train[1]["previous_snapshot_sha256"] = None
    elif failure == "partial_training": train[1]["report"]["completed_step"] = 1999
    elif failure == "synthetic_labels": train[0]["report"]["label_source"] = "synthetic_fixture"
    elif failure == "missing_official": samples.pop()
    elif failure == "reused_root": samples[2]["root_seeds"][0] = samples[0]["root_seeds"][0]
    elif failure == "wrong_draw": manifests[0]["rows"][0]["root_seed"] = 999999
    elif failure == "wrong_feedback_phase": samples[0]["purpose"] = "development"
    elif failure == "leaked_guided": manifests[0]["rows"][0]["source"] = "guided"
    elif failure == "wrong_snapshot": samples[1]["snapshot_sha256"] = "b"*64
    elif failure == "wrong_split": manifests[1]["rows"][0]["split"] = "validation"
    elif failure == "truth": audits[0]["controller_truth_shared"] = True
    elif failure == "duplicate_audit": audits[1] = deepcopy(audits[0])
    elif failure == "missing_execution": audits.pop(0)
    elif failure == "failed_worker": audits[0]["status"] = "failed"
    elif failure == "missing_cleanup": audits[0]["process_group_cleaned"] = False
    elif failure == "agent_nomination": agents[0]["nominations"] = 1
    with pytest.raises(ValueError):
        CHECK.verify_receipts(train, samples, manifests, audits, agents, selected_sha=SOLVER)


def test_diagnostic_reduces_only_declared_case_counts_not_training_recipe_or_original_config():
    original = {"campaign_id": "pilot", "round_count": 2, "master_seed": 1,
        "backend": {"guidance_config": {"fit_count": 64, "validation_count": 16, "steps": 2000}},
        "official": {"base": 80}, "task": {"official_point_count": 256},
        "terminal": {"base": 32}, "development": {"starter_base": 4}}
    saved = deepcopy(original)
    selected = CHECK.diagnostic_config(original)
    assert original == saved
    assert selected["backend"] == saved["backend"] and selected["official"] == {"base": 80}
    assert selected["task"] == saved["task"]
    assert selected["terminal"] == {"base": 1, "base_perturbed": 0, "guided_per_round": 1}
    assert selected["development"]["max_cases_per_agent_per_round"] == 3
    assert selected["master_seed"] != original["master_seed"]


def _quick_evidence():
    draws = [{"root_seed": i, "bitwise_equal": True, "maximum_absolute_difference": 0.} for i in range(4)]
    audit = {"status": "completed", "return_code": 0, "error_code": None,
             "process_group_cleaned": True, "candidate_process_isolated": True, "controller_truth_shared": False}
    return draws, [audit], {"error": .02, "retention": {"draws": 0, "pending_draws": 0, "residents": 0, "anchors": 0}}


def test_quick_check_requires_bitwise_per_root_equality_and_clean_isolated_execution():
    draws, audits, keywords = _quick_evidence()
    CHECK.verify_quick_measurements(draws, audits, **keywords)


@pytest.mark.parametrize("failure", ["tiny_mismatch", "duplicate_root", "missing_root", "no_solver", "failed_solver",
                                      "truth", "leak", "nan_error"])
def test_quick_check_does_not_relax_roundtrip_or_accept_partial_work(failure):
    draws, audits, keywords = _quick_evidence()
    if failure == "tiny_mismatch":
        draws[0].update(bitwise_equal=False, maximum_absolute_difference=1e-12)
    elif failure == "duplicate_root": draws[1]["root_seed"] = draws[0]["root_seed"]
    elif failure == "missing_root": draws.pop()
    elif failure == "no_solver": audits.clear()
    elif failure == "failed_solver": audits[0]["return_code"] = 1
    elif failure == "truth": audits[0]["controller_truth_shared"] = True
    elif failure == "leak": keywords["retention"]["draws"] = 1
    elif failure == "nan_error": keywords["error"] = float("nan")
    with pytest.raises(ValueError):
        CHECK.verify_quick_measurements(draws, audits, **keywords)
