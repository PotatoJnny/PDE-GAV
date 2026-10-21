"""Tiny explicit reports for plotting; not model validation."""
from hashlib import sha256
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("vpfp_coverage_plots", ROOT/"scripts/plot_vpfp_regime_coverage.py")
plotter = importlib.util.module_from_spec(spec); spec.loader.exec_module(plotter)


def test_incomplete_or_mutated_result_cannot_be_plotted(tmp_path):
    path = tmp_path/"result.json"
    path.write_text(json.dumps({"execution_passed":False,"stage":"extract","targets":{}}))
    digest = sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match="completed"):
        plotter.load_completed(path,digest)
    path.write_text("changed")
    with pytest.raises(ValueError,match="SHA"):
        plotter.load_completed(path,digest)


def test_plot_keeps_eight_strata_and_publishes_provenance_without_admission(tmp_path, monkeypatch):
    from matplotlib.figure import Figure
    original_savefig = Figure.savefig
    displayed_ranges = []
    def record_budget_ranges(figure, filename, *args, **kwargs):
        if Path(filename).name == "fidelity-coverage-curves.png":
            assert len(figure.axes) == 6
            for axis in figure.axes:
                assert axis.get_xscale() == "symlog"
                displayed_ranges.append(axis.get_xlim())
        return original_savefig(figure, filename, *args, **kwargs)
    monkeypatch.setattr(Figure, "savefig", record_budget_ranges)
    rows = [{"stratum":i,"neighbors":{"5":{"reference_coverage":i/7}}} for i in range(8)]
    audit = {"execution_passed":True,"admitted":False,"geometry":{"fit_identity_sha256":"a"*64},
        "source_manifest_sha256":"b"*64,"comparisons":{role:{"joint_strata":rows} for role in ("generated","reference_null")},
        "budget_curves":{axis:{"curve":[{"maximum_dimensionless_budget":b,"k5_reference_coverage_by_stratum":[b/100]*8,
            "retained_fraction_all_attempts":b/100} for b in (0,.01,1,100)]} for axis in plotter.AXES}}
    target = tmp_path/"target.json"; target.write_text(json.dumps(audit))
    result = {"execution_passed":True,"stage":"complete","targets":{"target-0":{
        "status":"descriptive_audit_completed","directory":"/synthetic/seed-0/base/evaluation","audit_path":str(target)}}}
    path = tmp_path/"result.json"; path.write_text(json.dumps(result))
    receipt = plotter.plot(path,sha256(path.read_bytes()).hexdigest(),tmp_path/"figures")
    assert receipt["execution_passed"] and not receipt["admitted"]
    assert receipt["new_field_evaluations"] == 0
    assert receipt["models"] == ["seed-0/base"]
    assert displayed_ranges == [(0.0, 100.0)]*6
    assert receipt["budget_display_range"] == [0.0, 100.0]
    assert receipt["budget_display_range_shared_by_all_panels"] is True
    for name in receipt["figures"]:
        assert (tmp_path/"figures"/name).stat().st_size > 1000


def budget_report(curves):
    return {"budget_curves": {axis: {"curve": [
        {"maximum_dimensionless_budget": value} for value in curve]}
        for axis, curve in zip(plotter.AXES, curves)}}


def test_display_range_uses_all_models_and_budget_axes_without_changing_curves():
    reports = [("first", budget_report(([0, .1], [0, 4], [0, 2]))),
               ("second", budget_report(([0, 1], [0, 3], [0, 9])))]
    before = json.dumps(reports, sort_keys=True)
    assert plotter.budget_display_limits(reports) == (0.0, 9.0)
    assert json.dumps(reports, sort_keys=True) == before


@pytest.mark.parametrize("values", ([-1, 1], [0, float("nan")], [0, float("inf")],
                                   [0, True], [0, "1"], [1, 0], [0, 0], [], [0]))
def test_budget_display_rejects_invalid_or_degenerate_registered_values(values):
    report = budget_report((values, values, values))
    with pytest.raises(ValueError, match="budget"):
        plotter.budget_display_limits([("fixture", report)])
