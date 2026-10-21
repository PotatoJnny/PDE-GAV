"""Summarize the frozen14-state strong audit; never choose a model or cutoff."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import plot_vpfp_regime_coverage as coverage_plot

EXPECTED_SHA256="6af6ea4418a9ec5eadd6dbc52209e1e5380c4505013bceabe04135551074b5f4"
AXES=coverage_plot.AXES
EXAMPLE_BUDGETS=(1.,math.sqrt(10.))
NOTES={
    "FD":"Descriptive squared Frechet distance in reference-calibration physical PCA; not image FID or an admission score.",
    "neighbors":"k3/k5/k10 precision, reference coverage and generated-neighborhood recall are arithmetic means of eight condition strata; no frequency weighting.",
    "morphology":"Sum distinct (stratum, morphology cell) counts; identical cell labels in different strata are never merged.",
    "joint_budgets":"All three dimensionless collision-relative axes must be <=lambda at all eight registered times. lambda=1 and sqrt(10) are examples, NOT acceptance thresholds.",
    "late":"Arithmetic mean of per-time cohort means at fractions .5/.75/1; not a trajectory time integral or RMS pooled across all probes/times.",
    "dt_K":"Existing horizon-scaled dt_K_rms descriptor: horizon/(rho_background*condition_temperature).",
    "dt_f_and_source":"Existing fixed-probe RMS descriptors scaled by condition_temperature*horizon/rho_background. Neither is a full-space velocity L2 integral or an unscaled physical S norm.",
    "late_max_budgets":"For each original trajectory and each collision-normalized budget axis, take max across late .5/.75/1, then cohort median. Undefined entries are excluded with their counts retained.",
    "terminal_spectra":"Saved last-time rho/K high spatial-band powers in original audit descriptor scaling; grid/refinement limitations remain unchanged.",
    "sample_unit":"128 attempts per state are 128 trajectory roots; states reuse paired roots/noise and summing14 counts does NOT give1792 independent answers.",
    "uncertainty":"Sample-count curves are nested, not independent replicate datasets. 49 exploratory trajectory bootstraps do not establish equivalence, adequate power or numerical convergence.",
    "limits":"Previously opened coarse development references; no fresh512 confirmation, physical thresholds or scientific admission established.",
    "provenance":"The aggregate has its predeclared SHA. Child audit/measurement and plotting-source SHAs are frozen when this summary reads them and rechecked after rendering; the aggregate SHA alone does not authenticate their historical bytes.",
}


def expected_states():
    return [(seed,arm,step) for seed in range(2)
        for arm in ("cov_base","initial_conditioned","initial_conditioned_rho_cov")
        for step in ((20000,) if arm.endswith("rho_cov") else (20000,30000,40000))]


def state_identity(directory):
    match=re.search(r"/seed-([01])/(cov_base|initial_conditioned|initial_conditioned_rho_cov)/step-([0-9]{6})/evaluation$",str(directory))
    if match is None: raise ValueError("explicit strong fixed-step generated directory required")
    return int(match[1]),match[2],int(match[3])


def finite(value):
    if type(value) not in (int,float) or not math.isfinite(value): raise ValueError("finite stored scalar required")
    return float(value)


def strata(rows):
    if sorted(row["stratum"] for row in rows)!=list(range(8)):
        raise ValueError("exactly eight distinct condition strata required")
    return sorted(rows,key=lambda row:row["stratum"])


def descriptors(audit,role):
    items=audit["cohorts"][role]["physical_descriptor_summary"]
    values={item["name"]:finite(item["mean"]) for item in items}
    if len(values)!=len(items): raise ValueError("duplicate physical descriptor")
    return values


def joint_point(audit,budget):
    points=[row for row in audit["joint_budget_curve"]["curve"]
        if math.isclose(finite(row["maximum_dimensionless_budget_on_each_axis"]),budget,rel_tol=1.e-12,abs_tol=0.)]
    if len(points)!=1: raise ValueError("registered joint budget example missing or duplicated; no nearest-value selection")
    row=points[0]
    if len(row["k5_reference_coverage_by_stratum"])!=8: raise ValueError("eight joint-budget coverage values required")
    return row


def late_budget_medians(measurements):
    times=measurements["plan"]["time_fractions"]
    indices=[i for i,value in enumerate(times) if value>=.5]
    if [times[i] for i in indices]!=[.5,.75,1.]: raise ValueError("registered late three-time plan required")
    if measurements["budget_axes"]!=list(AXES): raise ValueError("source/momentum/energy budget order changed")
    output={}
    for column,name in enumerate(AXES):
        values=[]
        for case in measurements["cases"]:
            if case.get("hard_valid") is not True: continue
            ledger=case["budget_ledger"]["collision_normalized_budget_by_time"]
            if len(ledger)!=len(times): raise ValueError("budget ledger time count differs")
            selected=[ledger[i][column] for i in indices]
            if any(value is None for value in selected): continue
            values.append(max(map(finite,selected)))
        output[name]={"median":statistics.median(values) if values else None,
            "defined_count":len(values),"undefined_or_failed_count":len(measurements["cases"])-len(values)}
    return output


def reference_joint_curve(measurements,grid):
    if measurements["budget_axes"]!=list(AXES): raise ValueError("reference budget order changed")
    count=len(measurements["cases"]); valid=[]
    if count==0: raise ValueError("reference cohort must not be empty")
    for case in measurements["cases"]:
        if len(case["budget_defined_mask"])!=3: raise ValueError("three explicit reference budget definition flags required")
        if case.get("hard_valid") is not True or not all(case["budget_defined_mask"]): continue
        if len(case["budgets"])!=3: raise ValueError("three reference budget axes required")
        valid.append(max(map(finite,case["budgets"])))
    return [{"maximum_dimensionless_budget_on_each_axis":finite(limit),
        "retained_count":sum(value<=limit for value in valid),
        "retained_fraction_all_attempts":sum(value<=limit for value in valid)/count,
        "attempted_count":count,"undefined_or_failed_count":count-len(valid)} for limit in grid]


def summarize_model(label,audit,measurements):
    seed,arm,step=state_identity(audit["generated_directory"])
    row={"model":label,"seed":seed,"arm":arm,"global_step":step,
        "attempted_count":audit["cohorts"]["generated"]["attempted_count"],
        "hard_valid_count":audit["cohorts"]["generated"]["hard_valid_count"],
        "numerically_uncertain_count":audit["cohorts"]["generated"]["numerically_uncertain_count"],
        "FD":finite(audit["comparisons"]["generated"]["global_frechet_descriptive"]["squared_distance"])}
    comparisons=strata(audit["comparisons"]["generated"]["joint_strata"])
    for k in (3,5,10):
        for field in ("precision_all_attempts","reference_coverage","recall_generated_neighborhood"):
            row[f"k{k}_{field}"]=statistics.mean(finite(x["neighbors"][str(k)][field]) for x in comparisons)
    for i,item in enumerate(comparisons): row[f"k5_coverage_stratum_{i}"]=finite(item["neighbors"]["5"]["reference_coverage"])
    morphology=strata([item for item in audit["morphology_occupancy"]["rows"] if item["cohort"]=="generated"])
    row["covered_morphology_stratum_cells"]=sum(item["covered_reference_cells"] for item in morphology)
    row["reference_morphology_stratum_cells"]=sum(item["reference_cells"] for item in morphology)
    for name,budget in zip(("one","sqrt10"),EXAMPLE_BUDGETS):
        point=joint_point(audit,budget)
        row[f"joint_{name}_count"]=point["retained_count"]
        row[f"joint_{name}_fraction"]=finite(point["retained_fraction_all_attempts"])
        row[f"joint_{name}_k5_coverage"]=statistics.mean(map(finite,point["k5_reference_coverage_by_stratum"]))
    values=descriptors(audit,"generated")
    for suffix,key in (("dt_K_rms","late_dtK_mean"),("dt_f_fixed_probe_rms","late_dtf_probe_mean"),
                       ("source_fixed_probe_rms","late_source_probe_mean")):
        row[key]=statistics.mean(values[f"time{i}.{suffix}"] for i in (5,6,7))
    late_budgets=late_budget_medians(measurements)
    for short,axis in zip(("source","momentum","energy"),AXES):
        stats=late_budgets[axis]
        row[f"late_{short}_max_budget_median"]=stats["median"]
        row[f"late_{short}_budget_defined_count"]=stats["defined_count"]
        row[f"late_{short}_budget_undefined_or_failed_count"]=stats["undefined_or_failed_count"]
    row["terminal_rho_high_power"]=values["time7.rho_spectrum_high"]
    row["terminal_K_high_power"]=values["time7.K_spectrum_high"]
    return row


def joint_plot(reports,reference_curves,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure,panels=plt.subplots(2,1,figsize=(11,8),sharex=True,constrained_layout=True)
    colors=plt.get_cmap("tab20").colors
    maximum=0.
    for i,(label,report) in enumerate(reports):
        curve=report["joint_budget_curve"]["curve"]
        x=[point["maximum_dimensionless_budget_on_each_axis"] for point in curve]
        maximum=max(maximum,max(x))
        panels[0].plot(x,[statistics.mean(point["k5_reference_coverage_by_stratum"]) for point in curve],label=label,color=colors[i])
        panels[1].plot(x,[point["retained_fraction_all_attempts"] for point in curve],color=colors[i])
    reference_handles=[]
    for role,curve in reference_curves.items():
        handle,=panels[1].plot([point["maximum_dimensionless_budget_on_each_axis"] for point in curve],
            [point["retained_fraction_all_attempts"] for point in curve],"--",linewidth=2,label=role)
        reference_handles.append(handle)
    for panel in panels:
        panel.set_xscale("symlog",linthresh=1.e-4); panel.set_xlim(0,maximum)
        panel.set_ylim(-.02,1.02); panel.grid(alpha=.2)
    panels[0].set_ylabel("Mean k5 reference coverage\n(equal eight-stratum weight)")
    panels[1].set_ylabel("Retained / all original attempts")
    panels[1].set_xlabel("Joint bound on ALL THREE axes (examples, not admission thresholds)")
    handles,labels=panels[0].get_legend_handles_labels()
    figure.legend(handles+reference_handles,labels+list(reference_curves),loc="upper center",bbox_to_anchor=(.5,-.01),ncol=3,fontsize=8)
    figure.suptitle("Strong VPFP: joint source/momentum/energy budget\nReference A/B retained fractions only; filtered reference coverage was not recomputed")
    figure.savefig(Path(output)/"joint-three-budget-curves.png",dpi=160,bbox_inches="tight"); plt.close(figure)


def markdown(summary):
    columns=("model","FD","k3_reference_coverage","k5_reference_coverage","k10_reference_coverage",
        "covered_morphology_stratum_cells","joint_one_count","joint_one_k5_coverage","joint_sqrt10_count",
        "late_dtK_mean","late_dtf_probe_mean","late_source_probe_mean")
    text=["# Strong fixed-step audit comparison","", "Machine summary of frozen artifacts; no selected winner or acceptance threshold.","",
        "| "+" | ".join(columns)+" |","| "+" | ".join("---" for _ in columns)+" |"]
    for row in summary["models"]:
        text.append("| "+" | ".join(f"{row[key]:.6g}" if isinstance(row[key],float) else str(row[key]) for key in columns)+" |")
    text.extend(["","Reference joint-budget examples (not physically accepted counts):",""])
    for role,item in summary["references"].items():
        text.append(f"- {role}: lambda1 {item['joint_one_count']}/{item['attempted_count']}; sqrt10 {item['joint_sqrt10_count']}/{item['attempted_count']}.")
    text.extend(["","## Definitions and limits",""])
    text.extend(f"- {key}: {value}" for key,value in NOTES.items())
    text.extend(["", "The CSV contains the full fourteen-row scalar table, including k5 by stratum, late maximum-budget medians/defined counts and terminal high-band powers.",""])
    return "\n".join(text)


def run(audit_result,expected_sha256,output):
    if expected_sha256!=EXPECTED_SHA256: raise ValueError("this summary is pinned to the authorized completed5286256 aggregate SHA")
    output=Path(output).absolute(); source=Path(audit_result).absolute()
    if output.exists() or output.is_symlink() or output.is_relative_to(source.parent):
        raise ValueError("fresh output outside completed audit inputs required")
    output.mkdir(parents=True)
    summary={"schema":"vpfp_strong_fixed14_audit_summary_v1","execution_passed":False,"admitted":False,
        "automatic_model_or_threshold_selection":False,"new_sampling":False,"new_PDE_solves":0,"new_training":False,
        "audit_result":str(source),"audit_result_sha256":expected_sha256,"definitions":NOTES,"models":[]}
    def save():
        path=output/"summary.json.partial"; path.write_text(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False)+"\n")
        path.replace(output/"summary.json")
    save()
    try:
        reports,frozen=coverage_plot.load_completed(source,expected_sha256)
        for path in (Path(__file__),ROOT/"scripts/plot_vpfp_regime_coverage.py"):
            frozen[str(path)]=coverage_plot.file_sha(path)
        if [state_identity(audit["generated_directory"]) for _,audit in reports]!=expected_states():
            raise ValueError("exact ordered14 fixed20/30/40k states required")
        aggregate=json.loads(source.read_text())
        reference_reports={}
        def measurements(path):
            frozen[str(path)]=coverage_plot.file_sha(path)
            value=json.loads(path.read_text())
            if value.get("complete") is not True or len(value["cases"])!=128:
                raise ValueError("all128 stored measurement rows required")
            if value["plan"]["time_fractions"]!=[0.,.015625,.0625,.125,.25,.5,.75,1.]:
                raise ValueError("original fixed eight-time plan changed")
            return value
        for role in ("reference_a","reference_b"):
            reference_reports[role]=measurements(source.parent/(role+"-measurements.json"))
        if reference_reports["reference_a"]["plan_sha256"]!=reference_reports["reference_b"]["plan_sha256"]:
            raise ValueError("reference A/B measurement plans differ")
        for (target,item),(label,audit) in zip(sorted(aggregate["targets"].items()),reports):
            if item["directory"]!=audit["generated_directory"] or audit["cohorts"]["generated"]["attempted_count"]!=128:
                raise ValueError("target directory/128 attempt identity differs")
            values=measurements(source.parent/(target+"-measurements.json"))
            if values["plan_sha256"]!=reference_reports["reference_a"]["plan_sha256"]:
                raise ValueError("measurement plans differ from the reference")
            summary["models"].append(summarize_model(label,audit,values))
        grid=reports[0][1]["joint_budget_curve"]["grid"]
        if any(audit["joint_budget_curve"]["grid"]!=grid for _,audit in reports):
            raise ValueError("all14 models must share the same preregistered joint budget grid")
        reference_curves={role:reference_joint_curve(value,grid) for role,value in reference_reports.items()}
        summary["references"]={}
        for role,curve in reference_curves.items():
            result={"attempted_count":128,"joint_budget_curve":curve,"late_budget_medians":late_budget_medians(reference_reports[role])}
            for label,budget in zip(("one","sqrt10"),EXAMPLE_BUDGETS):
                points=[x for x in curve if math.isclose(x["maximum_dimensionless_budget_on_each_axis"],budget,rel_tol=1.e-12)]
                if len(points)!=1: raise ValueError("reference joint example must be on the registered grid")
                result[f"joint_{label}_count"]=points[0]["retained_count"]
            summary["references"][role]=result
        summary["reference_null_FD"]=reports[0][1]["comparisons"]["reference_null"]["global_frechet_descriptive"]["squared_distance"]
        summary["reference_null_k5_coverage_by_stratum"]=coverage_plot.coverage_vector(reports[0][1],"reference_null")
        summary["frozen_input_sha256"]=frozen; save()
        with (output/"models.csv").open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(summary["models"][0])); writer.writeheader(); writer.writerows(summary["models"])
        (output/"comparisons.md").write_text(markdown(summary))
        summary["reused_plot_receipt"]=coverage_plot.plot(source,expected_sha256,output/"figures")
        joint_plot(reports,reference_curves,output/"figures")
        summary["figures"]=["figures/fidelity-coverage-curves.png","figures/condition-coverage.png","figures/joint-three-budget-curves.png"]
        summary["execution_passed"]=True
    except Exception as error:
        summary["failure"]={"type":type(error).__name__,"reason":str(error)}
        save(); raise
    save()
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-result",type=Path,required=True)
    parser.add_argument("--expected-sha256",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(); run(args.audit_result,args.expected_sha256,args.output)
