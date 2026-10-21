"""Synthetic validation of the real-asset split acceptance bookkeeping."""

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("artemis_split_check", ROOT / "scripts/check_artemis_split_execution.py")
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def _reference(tmp_path):
    (tmp_path / "case.npz").write_bytes(b"synthetic-placeholder-not-scientific-data")
    names = ("R4_winner", "R5_explorer_1", "R5_explorer_2")
    return {"status": "completed", "candidates": [{"name": name} for name in names],
            "cases": [{"candidate": name, "case_key": str(index), "status": "completed",
                       "evaluation_wall_seconds": 2.0} for name in names for index in range(4)],
            "cohort": [{"key": str(index), "public_arrays_path": "case.npz"} for index in range(4)]}


def test_reference_subset_reuses_existing_solutions_and_keeps_source_untouched(tmp_path):
    direct = _reference(tmp_path)
    saved = deepcopy(direct)
    result = CHECK.reference_subset(direct, tmp_path)
    assert direct == saved
    assert len(result["cases"]) == 8
    assert [row["name"] for row in result["candidates"]] == list(CHECK.SELECTED)
    assert result["evaluation_wall_seconds"] == 16.0
    assert all(Path(row["public_arrays_path"]).is_absolute() for row in result["cohort"])


@pytest.mark.parametrize("failure", ["missing", "failed"])
def test_reference_subset_rejects_incomplete_direct_outputs(tmp_path, failure):
    direct = _reference(tmp_path)
    if failure == "missing":
        direct["cases"].pop()
    else:
        direct["cases"][-1]["status"] = "failed"
    with pytest.raises(ValueError, match="four completed cases"):
        CHECK.reference_subset(direct, tmp_path)


def _audits():
    return [{"execution_id": "test", "case_index": index, "status": "completed", "return_code": 0,
             "process_group_cleaned": True, "controller_truth_shared": False,
             "candidate_process_isolated": True, "error_code": None, "timeout_seconds": 900.0}
            for index in range(8)]


def test_clean_audits_require_all_eight_workers():
    CHECK.check_audits(_audits())


@pytest.mark.parametrize("failure", ["missing", "duplicate", "process_group_cleaned", "controller_truth_shared",
                                     "return_code", "error_code", "timeout_seconds"])
def test_worker_failure_privacy_or_cleanup_failure_rejects_acceptance(failure):
    audits = _audits()
    if failure == "missing":
        audits.pop()
    elif failure == "duplicate":
        audits[1] = audits[0]
    else:
        audits[0][failure] = {"process_group_cleaned": False, "controller_truth_shared": True,
                             "return_code": 1, "error_code": "failed", "timeout_seconds": 901.0}[failure]
    with pytest.raises(ValueError):
        CHECK.check_audits(audits)
