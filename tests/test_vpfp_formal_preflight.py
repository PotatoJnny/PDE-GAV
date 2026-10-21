"""Synthetic bookkeeping tests, not numerical or real-asset validation."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("vpfp_formal_preflight", ROOT / "scripts/check_vpfp_formal_preflight.py")
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)
SOLVER = "f" * 64


def evidence():
    inventory = [{"case_id": f"case-{i}", "root_seed": 100+i} for i in range(16)]
    measurements = []
    for workers, elapsed in ((4, 160.), (8, 100.), (16, 120.)):
        rows = [{**root, "point_count": 256, "candidate_query_point_count": 256.,
            "fields": list(CHECK.FIELDS), "query_sha256": f"{i:064x}", "primary_error": .02+i*.001,
            "macro_metrics": {"rho_relative_l2": .01, "momentum_relative_l2": .03,
                              "kinetic_energy_relative_l2": .02}}
            for i, root in enumerate(inventory)]
        audits = [{"execution_id": f"group-{workers}", "case_index": i,
            "artifact_id": CHECK.ARTIFACT_ID, "status": "completed", "phase": "official",
            "return_code": 0, "error_code": None, "process_group_cleaned": True,
            "candidate_process_isolated": True, "controller_truth_shared": False} for i in range(16)]
        measurements.append({"worker_count": workers, "wall_seconds": elapsed, "status": "measured",
            "solver_sha256": SOLVER, "rows": rows, "audits": audits})
    return inventory, measurements


def choose(inventory, measurements):
    return CHECK.choose_worker_count(measurements, inventory, solver_sha=SOLVER, point_count=256)


def test_fastest_consistent_full_group_is_selected_not_largest_worker_count():
    inventory, measurements = evidence()
    saved = deepcopy((inventory, measurements))
    assert choose(inventory, measurements) == 8
    assert (inventory, measurements) == saved


@pytest.mark.parametrize("problem", ["missing_group", "missing_case", "duplicate_root", "duplicate_case",
    "changed_root", "missing_audit", "duplicate_audit", "bad_index", "failed_process", "wrong_solver",
    "wrong_artifact", "truth_leak", "cleanup", "not_isolated", "wrong_phase", "changed_points",
    "fewer_points", "candidate_points", "missing_field", "metric_keys", "macro_drift", "nan_score",
    "nan_metric", "negative_error", "zero_wall", "nan_wall", "group_failed"])
def test_any_failed_or_inconsistent_group_blocks_all_selection(problem):
    inventory, measurements = evidence()
    group = measurements[-1]
    row = group["rows"][0]
    audit = group["audits"][0]
    if problem == "missing_group": measurements.pop()
    elif problem == "missing_case": group["rows"].pop()
    elif problem == "duplicate_root": inventory[1]["root_seed"] = inventory[0]["root_seed"]
    elif problem == "duplicate_case": inventory[1]["case_id"] = inventory[0]["case_id"]
    elif problem == "changed_root": row["root_seed"] += 1
    elif problem == "missing_audit": group["audits"].pop()
    elif problem == "duplicate_audit": group["audits"][1] = deepcopy(audit)
    elif problem == "bad_index": audit["case_index"] = 100
    elif problem == "failed_process": audit["status"] = "timeout"
    elif problem == "wrong_solver": group["solver_sha256"] = "b"*64
    elif problem == "wrong_artifact": audit["artifact_id"] = "different"
    elif problem == "truth_leak": audit["controller_truth_shared"] = True
    elif problem == "cleanup": audit["process_group_cleaned"] = False
    elif problem == "not_isolated": audit["candidate_process_isolated"] = False
    elif problem == "wrong_phase": audit["phase"] = "development"
    elif problem == "changed_points": row["query_sha256"] = "e"*64
    elif problem == "fewer_points": row["point_count"] = 128
    elif problem == "candidate_points": row["candidate_query_point_count"] = 128
    elif problem == "missing_field": row["fields"].pop()
    elif problem == "metric_keys": row["macro_metrics"].pop("momentum_relative_l2")
    elif problem == "macro_drift": row["macro_metrics"]["momentum_relative_l2"] += 1e-8
    elif problem == "nan_score": row["primary_error"] = float("nan")
    elif problem == "nan_metric": row["macro_metrics"]["rho_relative_l2"] = float("nan")
    elif problem == "negative_error": row["primary_error"] = -.1
    elif problem == "zero_wall": group["wall_seconds"] = 0
    elif problem == "nan_wall": group["wall_seconds"] = float("nan")
    elif problem == "group_failed": group["status"] = "failed"
    with pytest.raises(ValueError):
        choose(inventory, measurements)


def test_small_roundoff_within_fixed_tolerance_is_accepted():
    inventory, measurements = evidence()
    measurements[1]["rows"][0]["primary_error"] += 1e-13
    assert choose(inventory, measurements) == 8


def test_query_receipt_uses_actual_coordinates_and_fields():
    from pde_agent_verifier_solver.sdk import AnswerQuery
    query = AnswerQuery.at_points(fields=CHECK.FIELDS, points=({"t": .5, "x": .1, "y": .2},))
    receipt = CHECK._query_receipt(SimpleNamespace(query=query))
    changed = AnswerQuery.at_points(fields=CHECK.FIELDS, points=({"t": .5, "x": .10001, "y": .2},))
    assert receipt["point_count"] == 1
    assert receipt["fields"] == list(CHECK.FIELDS)
    assert receipt["query_sha256"] != CHECK._query_receipt(SimpleNamespace(query=changed))["query_sha256"]


@pytest.mark.parametrize("counts", ["", "1,4", "4,4,8", "4,8,16,32", "4,8,0", "4,x,16"])
def test_bounded_worker_grid_cannot_be_silently_changed(counts):
    with pytest.raises(ValueError):
        CHECK.parse_worker_counts(counts)


def test_compute_execution_is_rejected_on_login_node(monkeypatch, tmp_path):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="Slurm"):
        CHECK.run_preflight(tmp_path / "not_read.json", tmp_path / "not_created")
    assert not (tmp_path / "not_created").exists()
