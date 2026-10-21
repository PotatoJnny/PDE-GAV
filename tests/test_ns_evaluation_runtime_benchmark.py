"""Benchmark safety: unchanged scientific inputs and honest paired comparisons."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "ns_runtime_benchmark", Path(__file__).resolve().parents[1] / "scripts/benchmark_ns_evaluation_runtime.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_subset_preserves_source_fields_cases_and_plans(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    rows = []
    for index, stratum in enumerate(("base", "guided", "base_perturbed", "base")):
        field = root / f"field{index}" / "hidden_continuous_case.pt"
        field.parent.mkdir()
        field.write_bytes(f"frozen bytes {index}".encode())
        rows.append({"case": {"case_id": f"case{index}"}, "stratum": stratum,
                     "state_path": str(field.relative_to(root)), "state_sha256": benchmark.digest(field),
                     "plan": {"seed": index, "points": [[.01, .2, .3]]}, "reference": [index / 10]})
    data = {"cases": rows, "case_count": len(rows), "strata_counts": {"base": 2, "guided": 1, "base_perturbed": 1}}
    data["content_sha256"] = benchmark.content_digest(data)
    source = root / "bank.json"
    source.write_text(json.dumps(data))
    before = source.read_bytes()
    destination = tmp_path / "subset" / "bank.json"
    result = benchmark.prepare_subset(source, destination, ("base", "guided", "base_perturbed"), 1)
    assert source.read_bytes() == before
    assert result["case_ids"] == ["case0", "case1", "case2"]
    subset = json.loads(destination.read_text())
    for original, selected in zip(rows, subset["cases"]):
        assert {k: v for k, v in original.items() if k != "state_path"} == {
            k: v for k, v in selected.items() if k != "state_path"}
        assert (root / original["state_path"]).read_bytes() == (destination.parent / selected["state_path"]).read_bytes()
    subset.pop("content_sha256")
    assert benchmark.content_digest(subset) == json.loads(destination.read_text())["content_sha256"]


@pytest.mark.parametrize("mutation", ("errors", "candidate", "query_points", "case_ids"))
def test_comparison_rejects_faster_but_inequivalent_execution(mutation):
    original = {"candidate_label": "solver", "device": "cpu", "label": "baseline", "status": "passed",
                "candidate_tree_sha256": "identity", "cases": [{"case_id": "a", "primary_error": .25}],
                "oracle_usage_totals": {"query_calls": 4, "queried_points": 16000}}
    updated = copy.deepcopy(original)
    updated["label"] = "optimized"
    if mutation == "errors":
        updated["cases"][0]["primary_error"] = .2
    elif mutation == "candidate":
        updated["candidate_tree_sha256"] = "other"
    elif mutation == "query_points":
        updated["oracle_usage_totals"]["queried_points"] -= 1
    else:
        updated["cases"][0]["case_id"] = "b"
    assert benchmark.compare_results([original, copy.deepcopy(original)])[1]["passed"]
    assert not benchmark.compare_results([original, updated])[1]["passed"]
