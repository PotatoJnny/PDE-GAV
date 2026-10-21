"""Tiny JSON mechanics and mocked rendering; no physical feature extraction."""
from copy import deepcopy
import csv
import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("strong_audit_summary_tests",ROOT/"scripts/summarize_vpfp_strong_audit.py")
driver=importlib.util.module_from_spec(spec); spec.loader.exec_module(driver)


def measurement(count=128,retained=26):
    cases=[]
    for index in range(count):
        budgets=[.5,.7,.9] if index<retained else [2.,.1,.1]
        cases.append({"hard_valid":True,"budget_defined_mask":[True]*3,"budgets":budgets,
            "budget_ledger":{"collision_normalized_budget_by_time":[[10.,20.,30.]]*5+[[1.,2.,3.],[2.,3.,4.],[3.,4.,5.]]}})
    return {"complete":True,"plan_sha256":"a"*64,"budget_axes":list(driver.AXES),
        "plan":{"time_fractions":[0.,.015625,.0625,.125,.25,.5,.75,1.]},"cases":cases}


def audit(seed=0,arm="cov_base",step=20000):
    descriptors=[{"name":f"time{t}.{field}","mean":float(t)} for t in (5,6,7)
        for field in ("dt_K_rms","dt_f_fixed_probe_rms","source_fixed_probe_rms")]
    descriptors += [{"name":"time7.rho_spectrum_high","mean":.003},{"name":"time7.K_spectrum_high","mean":.004}]
    comparisons=[]
    for index in range(8):
        comparisons.append({"stratum":index,"neighbors":{str(k):{"precision_all_attempts":.5,
            "reference_coverage":index/8,"recall_generated_neighborhood":.25} for k in (3,5,10)}})
    grid=[0.,1.,math.sqrt(10.),10.]
    return {"generated_directory":f"/synthetic/seed-{seed}/{arm}/step-{step:06d}/evaluation",
        "comparisons":{"generated":{"global_frechet_descriptive":{"squared_distance":2.},"joint_strata":comparisons},
            "reference_null":{"global_frechet_descriptive":{"squared_distance":.1},"joint_strata":deepcopy(comparisons)}},
        "cohorts":{"generated":{"attempted_count":128,"hard_valid_count":128,"numerically_uncertain_count":128,
            "physical_descriptor_summary":descriptors}},
        "morphology_occupancy":{"rows":[{"cohort":"generated","stratum":i,"covered_reference_cells":2,
            "reference_cells":3,"same_cell_labels":[[0],[1]]} for i in range(8)]},
        "joint_budget_curve":{"grid":grid,"curve":[{"maximum_dimensionless_budget_on_each_axis":value,
            "retained_count":32 if value else 0,"retained_fraction_all_attempts":.25 if value else 0.,
            "k5_reference_coverage_by_stratum":[.5 if value else 0.]*8} for value in grid]}}


def test_stratified_metrics_and_cells_are_not_globally_merged():
    row=driver.summarize_model("example",audit(),measurement())
    assert row["k5_reference_coverage"]==pytest.approx(3.5/8)
    assert row["k3_reference_coverage"]==row["k10_reference_coverage"]
    assert row["k5_coverage_stratum_7"]==.875
    assert row["covered_morphology_stratum_cells"]==16
    assert row["reference_morphology_stratum_cells"]==24
    assert row["joint_one_count"]==32 and row["joint_sqrt10_count"]==32
    assert row["late_dtK_mean"]==6 and row["late_dtf_probe_mean"]==6


def test_late_max_then_median_not_mean_or_all_time_max():
    values=measurement(count=2,retained=1)
    values["cases"][1]["budget_ledger"]["collision_normalized_budget_by_time"][7]=[7.,8.,9.]
    result=driver.late_budget_medians(values)
    assert [result[key]["median"] for key in driver.AXES]==[5.,6.,7.]
    values["cases"][1]["budget_ledger"]["collision_normalized_budget_by_time"][7][0]=None
    result=driver.late_budget_medians(values)
    assert result[driver.AXES[0]]=={"median":3.,"defined_count":1,"undefined_or_failed_count":1}


def test_joint_reference_requires_all_axes_and_counts_undefined():
    for count in (26,30):
        result=driver.reference_joint_curve(measurement(retained=count),[1.,math.sqrt(10.)])
        assert result[0]["retained_count"]==count and result[1]["retained_count"]==128
        assert result[0]["retained_fraction_all_attempts"]==count/128
    values=measurement(count=2,retained=2); values["cases"][0]["budget_defined_mask"][2]=False
    result=driver.reference_joint_curve(values,[1.])[0]
    assert result["retained_count"]==1 and result["undefined_or_failed_count"]==1


@pytest.mark.parametrize("problem",["strata","missing_lambda","duplicate_lambda","nonfinite","wrong_axis"])
def test_bad_or_incomparable_statistics_rejected(problem):
    report=audit(); values=measurement()
    if problem=="strata": report["comparisons"]["generated"]["joint_strata"][0]["stratum"]=1
    elif problem=="missing_lambda": report["joint_budget_curve"]["curve"].pop(1)
    elif problem=="duplicate_lambda": report["joint_budget_curve"]["curve"].append(deepcopy(report["joint_budget_curve"]["curve"][1]))
    elif problem=="nonfinite": report["cohorts"]["generated"]["physical_descriptor_summary"][0]["mean"]=float("nan")
    else: values["budget_axes"].reverse()
    with pytest.raises(ValueError): driver.summarize_model("fixture",report,values)


def test_explicit14_layout_and_unknown_aggregate_rejected(tmp_path):
    states=driver.expected_states()
    assert len(states)==14 and len(set(states))==14
    assert sum(state[2]==40000 for state in states)==4
    with pytest.raises(ValueError,match="pinned"):
        driver.run(tmp_path/"missing.json","0"*64,tmp_path/"output")
    assert not (tmp_path/"output").exists()


def test_full14_csv_markdown_and_existing_plot_calls_are_preserved(tmp_path,monkeypatch):
    source=tmp_path/"completed"; source.mkdir()
    reports=[]; targets={}
    for index,(seed,arm,step) in enumerate(driver.expected_states()):
        report=audit(seed,arm,step); target=f"target-{index:02d}"
        reports.append((f"seed-{seed}/{arm}/{step//1000}k",report))
        targets[target]={"directory":report["generated_directory"]}
        (source/(target+"-measurements.json")).write_text(json.dumps(measurement()))
    for role,count in (("reference_a",26),("reference_b",30)):
        (source/(role+"-measurements.json")).write_text(json.dumps(measurement(retained=count)))
    aggregate=source/"result.json"; aggregate.write_text(json.dumps({"targets":targets}))
    digest=driver.coverage_plot.file_sha(aggregate); monkeypatch.setattr(driver,"EXPECTED_SHA256",digest)
    monkeypatch.setattr(driver.coverage_plot,"load_completed",lambda *args:(reports,{str(aggregate):digest}))
    calls=[]
    def plot(path,expected,output):
        calls.append((path,expected,output)); output.mkdir()
        return {"execution_passed":True,"synthetic_render_fixture":True}
    monkeypatch.setattr(driver.coverage_plot,"plot",plot)
    monkeypatch.setattr(driver,"joint_plot",lambda *args:calls.append("joint_three_axes"))
    result=driver.run(aggregate,digest,tmp_path/"summary")
    assert result["execution_passed"] and not result["admitted"]
    assert len(result["models"])==14 and len(calls)==2
    assert result["references"]["reference_a"]["joint_one_count"]==26
    assert result["references"]["reference_b"]["joint_one_count"]==30
    with (tmp_path/"summary/models.csv").open() as stream: rows=list(csv.DictReader(stream))
    assert len(rows)==14 and "late_source_max_budget_median" in rows[0]
    text=(tmp_path/"summary/comparisons.md").read_text()
    assert "not independent" in text and "NOT acceptance thresholds" in text
    assert result["automatic_model_or_threshold_selection"] is False
    assert result["new_sampling"] is False and result["new_PDE_solves"]==0
    with pytest.raises(ValueError,match="fresh"): driver.run(aggregate,digest,tmp_path/"summary")
