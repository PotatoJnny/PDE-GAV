#!/usr/bin/env python3
"""No-training moving-frame mechanics and macro compression research curves.

Only eight predeclared train and eight validation cases (one per condition
stratum) are read. No physical reference arrays, generated samples, PDE solves,
or scientific admission are involved. Finite Fourier/time compression errors
are reported, never converted to a silently fitted pass threshold.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "src"))
from pde_agent_verifier_solver.plugins.vpfp import moving_frame as mf
from pde_agent_verifier_solver.plugins.vpfp.value_control_diffusion import extract_value_controls

_spec = importlib.util.spec_from_file_location("vpfp_moving_frame_source_helpers", ROOT / "scripts/diagnose_vpfp_drift_codec.py")
source = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(source)
SCHEMA = "vpfp_moving_frame_no_training_preflight_v1"
MODES, FIT_GRIDS, EVAL_GRIDS = (4,6), (33,65), (33,65)
JET_FIELDS = ("f","ux","uy","Cxx","Cxy","Cyy","jx","jy","K")
JET_AXES = ("dt","dx","dy")


def time_samples(control_count):
    """Five nodes plus first/interior/last interval off-node probes, not all t."""
    nodes = sorted(set([0, (control_count-1)//4, (control_count-1)//2,
                        3*(control_count-1)//4, control_count-1]))
    intervals = sorted(set([0, (control_count-2)//4, (control_count-2)//2,
                            3*(control_count-2)//4, control_count-2]))
    return ([{"kind":"node", "tau":i/(control_count-1)} for i in nodes]
            +[{"kind":"midpoint", "tau":(i+.5)/(control_count-1)} for i in intervals]
            +[{"kind":"quarter", "tau":(i+q)/(control_count-1)} for i in intervals for q in (.25,.75)])


def _summary(delta, reference):
    delta, reference = delta.double(), reference.double()
    error = delta.square().mean().sqrt()
    scale = reference.square().mean().sqrt()
    return {"count":delta.numel(), "absolute_rms":float(error), "reference_rms":float(scale),
            "signed_mean":float(delta.mean()), "maximum_absolute":float(delta.abs().max()),
            "relative_rms":float(error/scale) if float(scale)>0 else None}


def _constant_macro(config, velocity, covariance):
    t,m = config.control_count,config.modes
    u = velocity[None,:,None].expand(1,2,t).clone()
    local = torch.zeros((1,2,t,m-1),dtype=u.dtype,device=u.device)
    chol = torch.zeros((1,3,t,m),dtype=u.dtype,device=u.device)
    chol[...,0] = mf.log_cholesky(covariance)[None,:,None]
    return mf.pack_macro_controls(u,local,chol,config)


def derivative_time_samples(samples):
    """First, middle and last listed time in each kind; not a distribution."""
    selected = []
    for kind in ("node","midpoint","quarter"):
        rows = sorted((row for row in samples if row["kind"]==kind),key=lambda row:row["tau"])
        if rows:
            selected.extend(rows[i] for i in sorted(set((0,len(rows)//2,len(rows)-1))))
    return selected


@torch.enable_grad()
def _probe_jets(decoder,z,condition,points,macro=None):
    """Separate pointwise autodiff graphs for old density and moving-frame A.

    A is evaluated after compression, not differentiated through an oracle
    replacement of m. The native graph handles its own amplitude normalization,
    gauge and fallback; no finite-difference derivative or invented zero macro
    channel replaces that graph. Returned columns use physical t,x,y units.
    """
    q = points.detach().clone().requires_grad_(True)
    if macro is None:
        fields = decoder.source_fields(q,z,condition)
        source_moments = mf.source_local_moments(fields,decoder.base_decoder)
        rho = fields["rho"].double()
        u,c = source_moments["mean"],source_moments["covariance"]
        j = rho[:,None]*u
        k = .5*rho*(u.square().sum(-1)+c.diagonal(dim1=-2,dim2=-1).sum(-1))
        f = fields["f"].double()
    else:
        fields = decoder.fields(q,{"m":macro,"z":z},equation_condition=condition)
        u,c,j,k,f = (fields[name] for name in
            ("velocity","central_covariance","momentum","kinetic_energy","f"))
    scalars = torch.cat((f[:,None],u,c[:,0,:],c[:,1,1,None],j,k[:,None]),1)
    jets = torch.stack([torch.autograd.grad(scalars[:,i].sum(),q,
        retain_graph=i+1<len(JET_FIELDS),create_graph=False)[0][:,:3]
        for i in range(len(JET_FIELDS))],1).detach()
    if not torch.isfinite(jets).all():
        raise ValueError("nonfinite actual continuous derivative probe; no clipping or missing-row removal")
    return jets


def derivative_probe_diagnostics(base,z,condition,fits,samples):
    """Two independent spatial/thermal-velocity probes per selected time.

    First/interior/last nodes, midpoints and quarters stay separately labelled.
    These are clean-pair reconstruction probes, NOT a generated derivative
    distribution audit or a space/velocity integrated Sobolev norm.
    """
    decoder = next(iter(fits.values()))[0]
    config = decoder.macro_config
    parameter = next(base.parameters())
    fractions = torch.tensor([[.173,.389,.61,-.37],[.613,.827,-.93,.52]],
        device=parameter.device,dtype=torch.float64)
    rows = []
    for sample in derivative_time_samples(samples):
        t = config.time_domain[0]+sample["tau"]*(config.time_domain[1]-config.time_domain[0])
        xy = fractions[:,:2]*fractions.new_tensor(config.spatial_domain_lengths)
        velocity = fractions[:,2:]*math.sqrt(float(condition[1]))
        points = torch.cat((torch.full((len(fractions),1),t,device=parameter.device,dtype=torch.float64),xy,velocity),1)
        original = _probe_jets(decoder,z,condition,points)
        for (mode,grid),(candidate,macro) in fits.items():
            transformed = _probe_jets(candidate,z,condition,points,macro)
            metrics = {f"{axis}_{field}":_summary(transformed[:,i,j]-original[:,i,j],original[:,i,j])
                for i,field in enumerate(JET_FIELDS) for j,axis in enumerate(JET_AXES)}
            rows.append({"maximum_mode":mode,"fit_grid":grid,"time_kind":sample["kind"],
                "tau":sample["tau"],"physical_time":t,"points_txyvxvy":points.tolist(),
                "macro_coordinate_storage_dtype":str(macro.dtype),"field_order":list(JET_FIELDS),
                "derivative_axis_order":list(JET_AXES),"physical_coordinate_units":True,
                "original_derivative_values":original.tolist(),"A_derivative_values":transformed.tolist(),
                "metrics":metrics,"scientific_distribution_audit":False,
                "node_convention":"native_C1_Hermite_right_interval_at_interior_knots_left_at_final_endpoint; no_second_time_derivative_claim"})
    return rows


@torch.no_grad()
def mechanical_check(base,z,condition,value_config,config):
    """Full-H transformed quadrature plus no-compression pointwise identity.

    FP64 Hermite-Gaussian integration is in the source velocity variable;
    orders N+3/N+5 integrate the full squared N amplitude and second moments.
    This checks an actual nonidentity transform, including the whole floor.
    """
    decoder = mf.MovingFrameDecoder(base,config,value_config=value_config)
    parameter = next(base.parameters())
    q0 = torch.tensor([[config.time_domain[0]+.371*(config.time_domain[1]-config.time_domain[0]),
        .213*config.spatial_domain_lengths[0], .417*config.spatial_domain_lengths[1], .3,-.4]],
        device=parameter.device,dtype=torch.float64)
    original = decoder.source_fields(q0,z,condition)
    src = mf.source_local_moments(original,base)
    exact_m = _constant_macro(config,src["mean"][0],src["covariance"][0])
    # At this one spatial point m is uncompressed. This is not a spatial codec.
    identity = decoder.fields(q0,{"m":exact_m,"z":z},equation_condition=condition)
    ident_abs = float((identity["f"]-original["f"].double()).abs().max())
    ident_scale = float(original["f"].double().abs().max())
    dtype_eps = torch.finfo(parameter.dtype).eps
    identity_budget = 512*dtype_eps*max(ident_scale,torch.finfo(parameter.dtype).tiny)
    velocity = torch.tensor([.47,-.31],device=parameter.device,dtype=torch.float64)
    covariance = torch.tensor([[1.4,.23],[.23,.73]],device=parameter.device,dtype=torch.float64)
    macro = _constant_macro(config,velocity,covariance)
    target = decoder.macro_at(q0,macro,original)
    theta = float(condition[1])
    orders = (base.config.hermite_total_degree+3,base.config.hermite_total_degree+5)
    rows = []
    for order in orders:
        nodes,weights = np.polynomial.hermite.hermgauss(order)
        mesh = np.stack(np.meshgrid(nodes,nodes,indexing="ij"),-1).reshape(-1,2)
        vsrc = torch.tensor(mesh*math.sqrt(2*theta),device=parameter.device,dtype=torch.float64)
        standard = torch.linalg.solve_triangular(src["factor"][0],(vsrc-src["mean"][0]).T,upper=False).T
        velocity_nodes = velocity+standard@target["factor"][0].T
        query = q0.expand(len(mesh),-1).clone(); query[:,3:5] = velocity_nodes
        # Bound point batches even when N14 -> 19^2 nodes.
        g = torch.cat([decoder.fields(part,{"m":macro,"z":z},equation_condition=condition)["g"]
                       for part in query.split(128)])
        jac = torch.linalg.det(target["factor"])/torch.linalg.det(src["factor"])
        measure = torch.tensor(np.outer(weights,weights).ravel()*np.exp((mesh*mesh).sum(1))*2*theta,
                               device=parameter.device,dtype=torch.float64)*jac
        mass = (g*measure).sum()
        mean = torch.einsum("n,ni->i",g*measure,velocity_nodes)
        centered = velocity_nodes-velocity
        cov = torch.einsum("n,ni,nj->ij",g*measure,centered,centered)
        kinetic = .5*(g*measure*velocity_nodes.square().sum(-1)).sum()
        expected_kinetic = .5*(velocity.square().sum()+covariance.trace())
        rows.append({"order":order,"mass_absolute_error":float(abs(mass-1)),
            "mean_maximum_absolute_error":float((mean-velocity).abs().max()),
            "covariance_maximum_absolute_error":float((cov-covariance).abs().max()),
            "kinetic_energy_absolute_error":float(abs(kinetic-expected_kinetic)),
            "mass":float(mass),"mean":mean.tolist(),"covariance":cov.tolist()})
    # The continuous A algebra is FP64; native coefficients may be FP32 but
    # remain constant with v. This is an engineering roundoff check, not a
    # regime or compression tolerance, and its limit is fixed before data.
    quadrature_budget = 2e-9
    passed = ident_abs <= identity_budget and all(max(r[key] for key in
        ("mass_absolute_error","mean_maximum_absolute_error","covariance_maximum_absolute_error",
         "kinetic_energy_absolute_error")) <= quadrature_budget for r in rows)
    return {"passed":bool(passed),"uncompressed_point_f_absolute_error":ident_abs,
        "uncompressed_point_f_roundoff_budget":identity_budget,
        "quadrature_absolute_engineering_budget":quadrature_budget,"quadrature":rows,
        "full_source_Hermite_degree":base.config.hermite_total_degree,
        "complete_floor_transformed":True,"source_dtype":str(parameter.dtype),
        "pointwise_identity_is_not_spatial_reconstruction":True}


@torch.no_grad()
def diagnose_case(base,z,condition,*,modes=MODES,fit_grids=FIT_GRIDS,evaluation_grids=EVAL_GRIDS,
                  samples=None,chunk_size=256):
    value_config = mf.value_config_from_decoder(base)
    config = mf.MovingFrameMacroConfig(control_count=value_config.control_count,max_mode=max(modes),
        time_domain=base.config.time_domain,spatial_domain_lengths=base.config.equation.x_domain)
    samples = time_samples(config.control_count) if samples is None else samples
    mechanics = mechanical_check(base,z,condition,value_config,config)
    fits,extraction = {},[]
    for grid in fit_grids:
        high,report = mf.extract_macro_controls(base,z,condition,config,value_config=value_config,
                                                fit_grid=grid,chunk_size=chunk_size)
        extraction.append(report)
        for mode in modes:
            low_config = replace(config,max_mode=mode)
            # The downstream macro target cache and generated coordinates are
            # FP32. Include that storage quantization in compression curves,
            # while the separate exact-identity mechanics test stays FP64.
            low = mf.restrict_macro_controls(high,config,low_config).float()
            fits[(mode,grid)] = (mf.MovingFrameDecoder(base,low_config,value_config=value_config),low)
    rows = []
    parameter = next(base.parameters())
    for grid in evaluation_grids:
        # Offset grid is not the fit grid, even for count=33 or 65.
        xy = mf.spatial_grid(config,grid,device=parameter.device,offset=.371)
        for sample in samples:
            t = config.time_domain[0]+sample["tau"]*(config.time_domain[1]-config.time_domain[0])
            q = torch.cat((torch.full((len(xy),1),t,device=xy.device,dtype=xy.dtype),xy,torch.zeros_like(xy)),1)
            reference, predictions = [],{key:[] for key in fits}
            rho_rows = []
            for part in q.split(chunk_size):
                native = next(iter(fits.values()))[0].source_fields(part,z,condition)
                moments = mf.source_local_moments(native,base)
                rho = native["rho"].double(); rho_rows.append(rho)
                mean,cov = moments["mean"],moments["covariance"]
                ref_k = .5*rho*(mean.square().sum(-1)+cov.diagonal(dim1=-2,dim2=-1).sum(-1))
                reference.append(torch.cat((mean,cov[:,0,:],cov[:,1,1,None],rho[:,None]*mean,ref_k[:,None]),1))
                for key,(decoder,macro) in fits.items():
                    target = decoder.macro_at(part,macro,native)
                    u,c = target["mean"],target["covariance"]
                    k = .5*rho*(u.square().sum(-1)+c.diagonal(dim1=-2,dim2=-1).sum(-1))
                    predictions[key].append(torch.cat((u,c[:,0,:],c[:,1,1,None],rho[:,None]*u,k[:,None]),1))
            ref,rho = torch.cat(reference),torch.cat(rho_rows)
            for key,(decoder,macro) in fits.items():
                pred = torch.cat(predictions[key])
                metrics = {name:_summary(pred[:,i]-ref[:,i],ref[:,i]) for i,name in
                           enumerate(("ux","uy","Cxx","Cxy","Cyy","jx","jy","K"))}
                local_u = mf.split_macro_controls(macro.double(),decoder.macro_config)[0]
                prescribed = mf.interpolate_controls(local_u[...,None],q[:1,0],decoder.macro_config)[0,:,0]
                grid_global = pred[:,5:7].sum(0)/rho.sum()
                # Three fixed, nonzero thermal velocity probes at three spatial
                # rows; f error is a probe error, never a fullspace norm.
                probe = q[torch.tensor([0,len(q)//3,2*len(q)//3],device=q.device)].clone()
                probe[:,3:5] = math.sqrt(float(condition[1]))*torch.tensor([[0.,0.],[.7,-.4],[-1.1,.6]],device=q.device)
                old_f = decoder.source_fields(probe,z,condition)["f"].double()
                new_f = decoder.fields(probe,{"m":macro,"z":z},equation_condition=condition)["f"]
                rows.append({"maximum_mode":key[0],"fit_grid":key[1],"evaluation_grid":grid,
                    "macro_coordinate_storage_dtype":"torch.float32",
                    "evaluation_grid_cell_offset":.371,"time_kind":sample["kind"],"tau":sample["tau"],
                    "physical_time":t,"metrics":metrics,"f_probe_error":_summary(new_f-old_f,old_f),
                    "density_weighted_global_U_grid_check_maximum_absolute":float((grid_global-prescribed).abs().max())})
    derivatives = derivative_probe_diagnostics(base,z,condition,fits,samples)
    return {"numerical_mechanics":mechanics,"extraction":extraction,"compression_diagnostics":rows,
            "derivative_probe_diagnostics":derivatives,
            "numerical_mechanics_passed":mechanics["passed"],"execution_passed":True}


def run(corpus_root,config_path,output,device="cuda"):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("moving-frame development diagnostics require a Slurm compute allocation")
    corpus,config,output = Path(corpus_root).resolve(),Path(config_path).resolve(),Path(output)
    if not (corpus/"manifest.json").exists() and (corpus/"corpus/manifest.json").exists():
        corpus = corpus/"corpus"
    if output.exists() or output.is_symlink() or output.resolve().is_relative_to(corpus):
        raise ValueError("choose a fresh output outside immutable corpus")
    output = output.resolve(); output.mkdir(parents=True,exist_ok=False)
    started,stage,cases,failures = time.monotonic(),"source_binding",[],[]
    try:
        manifest,splits,frozen = source.load_development(corpus)
        code = [Path(__file__).resolve(),Path(source.__file__).resolve(),Path(mf.__file__).resolve(),config]
        plugin = ROOT/"src/pde_agent_verifier_solver/plugins/vpfp"
        code += [plugin/name for name in ("value_control_diffusion.py","structured_diffusion.py","canonical_binding.py",
                                           "perturbation_backend.py","diffusion_audit.py")]
        code += list((ROOT/"legacy/vpfp/src/agent_verifier/vpfp").glob("*.py"))
        frozen.update({str(path):source._file_sha(path) for path in code})
        selection = {role:source.select_rows(split,manifest["bin_edges"]) for role,split in splits.items()}
        source._write(output/"frozen-plan.json",{"schema":SCHEMA,"source_sha256":frozen,
            "source_manifest_sha256":manifest["manifest_sha256"],"selected_rows":selection,
            "selected_equation_seeds":{role:[int(splits[role]["equation_seeds"][i]) for i in rows] for role,rows in selection.items()},
            "modes":MODES,"fit_grids":FIT_GRIDS,"evaluation_grids":EVAL_GRIDS,
            "evaluation_time_rule":"five_control_nodes_and_five_intervals_mid_and_both_quarter_points",
            "derivative_probe_rule":"two_independent_txyv_probes_at_first_middle_last_listed_time_of_each_kind",
            "derivative_fields":JET_FIELDS,"derivative_axes_physical_units":JET_AXES,
            "node_derivative_convention":"native_C1_Hermite_right_interval_interior_left_interval_final_endpoint",
            "physical_reference_arrays_read":False,"new_pde_solves":0,"new_training_steps":0,
            "compression_is_research_curve_not_scientific_admission":True})
        stage = "build_bound_decoder"
        base = source.build_decoder(config,manifest,device)
        value = mf.value_config_from_decoder(base)
        parameter = next(base.parameters())
        for role,indices in selection.items():
            for index in indices:
                stage = f"{role}/row-{index}"
                case = {"split":role,"row":index,"equation_seed":int(splits[role]["equation_seeds"][index]),
                        "stratum_id":int(splits[role]["stratum_ids"][index])}
                try:
                    full = torch.tensor(np.array(splits[role]["coefficients"][index:index+1],copy=True),device=parameter.device)
                    z = extract_value_controls(full,value,validate_slopes=True).to(parameter)
                    cond = torch.tensor(np.array(splits[role]["conditions"][index],copy=True),device=parameter.device).to(parameter)
                    case["condition"] = cond.tolist()
                    case.update(diagnose_case(base,z,cond))
                except Exception as error:
                    case.update(execution_passed=False,numerical_mechanics_passed=False,error=repr(error),traceback=traceback.format_exc())
                    failures.append({"stage":stage,"error":repr(error)})
                cases.append(case)
                source._write(output/"partial-cases.json",{"cases":cases,"failures":failures})
                print(json.dumps({"stage":stage,"execution_passed":case["execution_passed"],
                                  "numerical_mechanics_passed":case["numerical_mechanics_passed"]}),flush=True)
        stage = "verify_frozen_inputs"
        result = {"schema":SCHEMA,"execution_passed":not failures,
            "source_manifest_sha256":manifest["manifest_sha256"],
            "frozen_plan_sha256":source._file_sha(output/"frozen-plan.json"),
            "numerical_mechanics_passed":bool(len(cases)==16 and all(c["numerical_mechanics_passed"] for c in cases)),
            "cases":cases,"failures":failures,"compression_diagnostics":"per_case_raw_unit_curves_in_cases",
            "selected_case_count":len(cases),"physical_reference_arrays_read":False,"new_pde_solves":0,
            "new_training_steps":0,"new_generated_answers":0,"scientific_regime_qualified":False,
            "training_recommended_automatically":False,"wall_seconds":time.monotonic()-started,
            "limitations":["Sixteen selected development cases, not population statistics or independent confirmation.",
                "Only listed spatial grids and time probes; no proof of uniform convergence between them.",
                "Macro Fourier and value-time interpolation errors are research curves, not a regime validity threshold.",
                "A retains full continuous carrier and moves the entire floor; it is not projected to fixed N14.",
                "Point f errors are thermal velocity probes, not a fullspace density norm."]}
    except Exception as error:
        result = {"schema":SCHEMA,"execution_passed":False,"numerical_mechanics_passed":False,
            "failed_stage":stage,"error":repr(error),"traceback":traceback.format_exc(),"cases":cases,"failures":failures}
    result["partial_artifacts_preserved"] = True
    source._write(output/"result.json",result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",default="cuda")
    args = parser.parse_args(argv)
    result = run(args.source,args.config,args.output,args.device)
    print(json.dumps({"execution_passed":result["execution_passed"],"numerical_mechanics_passed":result["numerical_mechanics_passed"],
                      "output":str(args.output)}),flush=True)
    return 0 if result["execution_passed"] and result["numerical_mechanics_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
