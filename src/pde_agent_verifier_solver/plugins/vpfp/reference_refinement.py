"""Few-root, same-equation raw numerical refinement; never an N14 truth gate.

Re-solve the manifest's continuous equation before its projection. Separate
stored N14 versus replay-coarse differences from replay-coarse versus refined
differences. Two resolutions do not establish convergence, domain-tail error,
an observed order or a GCI. All real computation belongs in a Slurm allocation.
"""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import importlib
from itertools import product
import json
import math
import os
from pathlib import Path
import time as walltime

import numpy as np
import torch

SCHEMA = "vpfp_same_equation_reference_refinement_v1"
ROLES = ("calibration","reference_a","reference_b")


def _jsonable(value):
    if isinstance(value,dict): return {str(k):_jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [_jsonable(v) for v in value]
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,torch.Tensor): return value.detach().cpu().tolist()
    return value


def _sha_json(value):
    return sha256(json.dumps(_jsonable(value),sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def _file_sha(path):
    digest = sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda:source.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()


def _write(path,value):
    path = Path(path)
    pending = path.with_name(path.name+".partial")
    pending.write_text(json.dumps(_jsonable(value),sort_keys=True,indent=2,allow_nan=False))
    pending.replace(path)


def select_stratified_rows(stratum_ids,*,rows=None,max_cases=1):
    """Predetermined first row of each condition stratum; no feature access."""
    groups = np.asarray(stratum_ids)
    if groups.ndim!=1 or groups.dtype.kind not in "iu" or not len(groups) or (groups<0).any():
        raise ValueError("nonempty nonnegative integer strata required")
    if type(max_cases) is not int or not 1<=max_cases<=8:
        raise ValueError("refinement is explicitly limited to 1..8 roots per call")
    if rows is None:
        rows = [int(np.flatnonzero(groups==group)[0]) for group in sorted(set(groups.tolist()))][:max_cases]
    else:
        rows = list(rows)
        if (not rows or len(rows)>max_cases or len(set(rows))!=len(rows)
                or any(type(i) is not int or not 0<=i<len(groups) for i in rows)):
            raise ValueError("explicit unique root rows must fit the declared few-case budget")
    return rows


def default_levels(generation_config):
    """Actual baseline ABI, then 1.5x space/velocity intervals and 2x frames."""
    generation = dict(generation_config)
    n = int(generation["velocity_grid_size"])
    level = {"name":"replay_coarse","nx":int(generation["nx"]),"ny":int(generation["ny"]),
        "nvx":n,"nvy":n,"nt":int(generation["nt"]),"t0":0.,"t1":float(generation["t1"]),
        "x_domain":[2*math.pi,2*math.pi],"v_bounds":[-float(generation["velocity_max"]),
            float(generation["velocity_max"]),-float(generation["velocity_max"]),float(generation["velocity_max"])],
        "internal_dt_max":float(generation["internal_dt_max"]),"dtype":generation["dtype"],
        "cfl":.18,"positivity_floor":1e-13,"rho_floor":1e-10,"source_interpolation":"linear_output_frames"}
    refined = {**level,"name":"refined_same_domain",
        "nx":math.ceil(1.5*level["nx"]),"ny":math.ceil(1.5*level["ny"]),
        "nvx":math.ceil(1.5*(n-1))+1,"nvy":math.ceil(1.5*(n-1))+1,
        "nt":2*(level["nt"]-1)+1,"internal_dt_max":level["internal_dt_max"]/2,"dtype":"float64"}
    return [level,refined]


def validate_levels(levels,generation_config):
    levels = _jsonable(levels)
    expected = default_levels(generation_config)[0]
    if not isinstance(levels,list) or len(levels)!=2 or levels[0]!=expected:
        raise ValueError("exact manifest coarse replay plus one explicitly refined level required")
    refined = levels[1]
    if not isinstance(refined,dict) or set(refined)!=set(expected):
        raise ValueError("unexpected refinement level fields")
    for key in ("t0","t1","x_domain","v_bounds","cfl","positivity_floor","rho_floor","source_interpolation"):
        if refined[key]!=expected[key]:
            raise ValueError("same-equation refinement may not change domains/boundaries/floors: "+key)
    if (refined["name"]!="refined_same_domain" or refined["dtype"]!="float64"
            or not math.isfinite(refined["internal_dt_max"]) or not 0<refined["internal_dt_max"]<expected["internal_dt_max"]
            or any(type(refined[k]) is not int or refined[k]<=expected[k] for k in ("nx","ny","nvx","nvy","nt"))):
        raise ValueError("refinement must increase each grid/frame count, lower dt cap and use FP64")
    return levels


def reference_domain_receipt(generation_config, decoder_config, *, levels=None):
    """Cheap, exact legacy coordinate qualification; no checkpoint or solver load.

    Legacy VPFPGrid used the analytic 2*pi box and FP32 linspace axes. The
    representation infers its period after promoting those axes to FP64 as
    last-first+first_spacing. Only THAT exact reconstructed value is accepted,
    never an allclose neighbourhood. Asset/config/module hashes remain checked
    separately by refine_reference_samples before any measurement.
    """
    generation = dict(generation_config)
    levels = validate_levels(default_levels(generation) if levels is None else levels,generation)
    cfg = decoder_config.to_dict() if hasattr(decoder_config,"to_dict") else decoder_config
    numerical = list(levels[0]["x_domain"])
    actual = list(cfg["equation"]["x_domain"])
    times = [levels[0]["t0"],levels[0]["t1"]]
    if list(cfg["time_domain"])!=times:
        raise ValueError("manifest numerical and decoder time domains differ")
    if "x_domain" in generation and list(generation["x_domain"])!=numerical:
        raise ValueError("generation contains an unsupported changed numerical domain")
    if len(actual)!=2 or not all(math.isfinite(x) and x>0 for x in actual):
        raise ValueError("finite positive decoder periods required")
    reconstructed,axes = [],[]
    for size,length in zip((generation["nx"],generation["ny"]),numerical):
        if type(size) is not int or size<3: raise ValueError("legacy periodic grid size must be an integer >=3")
        axis = np.linspace(0.,length,size,endpoint=False,dtype=np.float32)
        promoted = axis.astype(np.float64)
        reconstructed.append(float(promoted[-1]-promoted[0]+promoted[1]-promoted[0]))
        axes.append({"size":size,"fp32_axis_sha256":sha256(axis.tobytes()).hexdigest(),
            "first":float(promoted[0]),"last":float(promoted[-1]),"first_spacing":float(promoted[1]-promoted[0])})
    if actual==numerical:
        reason = "exact_identical_domains"
    elif generation["dtype"]=="float32" and actual==reconstructed:
        reason = "exact_legacy_FP32_linspace_promoted_period_last_minus_first_plus_first_spacing"
    else:
        raise ValueError("manifest numerical and decoder physical domains differ beyond exact legacy FP32 period reconstruction")
    return {"schema":"vpfp_exact_legacy_period_coordinate_receipt_v1","execution_passed":True,
        "time_domain":times,"numerical_x_domain":numerical,"decoder_x_domain":actual,
        "fp32_axis_inferred_x_domain":reconstructed,"axis_reconstruction":axes,"reason":reason,
        "absolute_period_difference":[a-b for a,b in zip(actual,numerical)],
        "native_query_dx_dnumerical_x":[a/b for a,b in zip(actual,numerical)],
        "comparison_coordinates":"raw coarse/refined at identical original numerical coordinates; native evaluated at equal periodic fractions",
        "native_spatial_derivative_policy":"pullback chain rule only; no density/J/K amplitude or velocity rescaling",
        "numerical_solver_domain_modified":False,"decoder_asset_modified":False,
        "domain_equality_tolerance_used":False,"same_physical_point_claimed_for_native_comparison":actual==numerical,
        "evidence":"VPFPGrid default 2*pi; make_axes FP32 linspace; representation_diagnostics._periodic_length on promoted axes"}


def _mapped_native_observations(model,latent,condition,points,modules,receipt):
    points = np.asarray(points,dtype=np.float64)
    mapped = points.copy()
    scales = np.asarray(receipt["native_query_dx_dnumerical_x"],dtype=np.float64)
    mapped[:,1:3] *= scales
    result = _native_observations(model,latent,condition,mapped,modules)
    for name,values in result["values"].items():
        if name.startswith("dx_"): result["values"][name] = (np.asarray(values)*scales[0]).tolist()
        elif name.startswith("dy_"): result["values"][name] = (np.asarray(values)*scales[1]).tolist()
    result.update(numerical_comparison_points=points.tolist(),native_decoder_query_points=mapped.tolist(),
        coordinate_receipt=receipt,derivative_method="native autodiff with explicit numerical-coordinate pullback for dx/dy")
    return result


def _modules():
    from .perturbation_backend import _legacy_api
    api = _legacy_api()
    return api,{name:importlib.import_module("agent_verifier.vpfp."+name) for name in
                ("high_fidelity","online_training","torch_reference","continuous_hermite","operators","continuous_macros")}


def _solve_level(spec,level,modules,device):
    """The actual original Torch numerical solver; no projection is invoked."""
    operators,solver = modules["operators"],modules["torch_reference"]
    grid = operators.VPFPGrid(nx=level["nx"],ny=level["ny"],nvx=level["nvx"],nvy=level["nvy"],
        x_domain=tuple(level["x_domain"]),v_bounds=tuple(level["v_bounds"]),dtype=level["dtype"])
    time = operators.VPFPTime(t0=level["t0"],t1=level["t1"],nt=level["nt"])
    cfg = solver.VPFPTorchSolverConfig(**{k:level[k] for k in
        ("internal_dt_max","dtype","cfl","positivity_floor","rho_floor","source_interpolation")})
    parameters = solver.pack_torch_equation_specs([spec],dtype=level["dtype"])
    prepared = solver.evaluate_torch_equation_parameter_batch(parameters,grid=grid,time=time,device=device)
    solution = solver.solve_torch_vpfp_batch(prepared,grid=grid,time=time,config=cfg,device=device)
    return solution,{"level":level,"same_unrounded_equation_spec_sha256":_sha_json(spec),
        "packed_precision_parameter_sha256":_sha_json({k:v for k,v in parameters.items() if "wall_time" not in k}),
        "solver_diagnostics":_jsonable(solution["diagnostics"]),
        "raw_f_shape":list(solution["f"].shape),"raw_f_bytes":solution["f"].numel()*solution["f"].element_size(),
        "projected_back_to_N14":False,"source_grid_interpolation":"linear at this level's own output frames",
        "equation_parameter_rounding_changes_with_dtype":True}


def _interpolate(values,level,points):
    """Explicit periodic-x, bounded t/v multilinear interpolation of raw grids."""
    points = torch.as_tensor(points,dtype=torch.float64,device=values.device)
    dims = 5 if values.ndim==5 else 3
    if points.ndim!=2 or points.shape[1]!=dims or not torch.isfinite(points).all():
        raise ValueError("raw interpolation needs finite physical t/x/y[/vx/vy] points")
    sizes = [level["nt"],level["nx"],level["ny"]]+([level["nvx"],level["nvy"]] if dims==5 else [])
    bounds = [(level["t0"],level["t1"]),(0,level["x_domain"][0]),(0,level["x_domain"][1])]
    if dims==5: bounds += [(level["v_bounds"][0],level["v_bounds"][1]),(level["v_bounds"][2],level["v_bounds"][3])]
    low,high,fraction = [],[],[]
    for axis,(size,(lower,upper)) in enumerate(zip(sizes,bounds)):
        if axis in (1,2):
            position = (points[:,axis]-lower)/(upper-lower)*size
            index = torch.floor(position).long()
            low.append(index.remainder(size)); high.append((index+1).remainder(size))
        else:
            if ((points[:,axis]<lower)|(points[:,axis]>upper)).any():
                raise ValueError("raw comparison may not extrapolate beyond common physical t/v domain")
            position = (points[:,axis]-lower)/(upper-lower)*(size-1)
            index = torch.floor(position).long().clamp(max=size-2)
            low.append(index); high.append(index+1)
        fraction.append(position-index)
    trailing = values.shape[dims:]
    result = torch.zeros((len(points),*trailing),dtype=torch.float64,device=values.device)
    for corner in product((0,1),repeat=dims):
        weight = torch.ones(len(points),dtype=torch.float64,device=values.device)
        for axis,bit in enumerate(corner):
            weight *= fraction[axis] if bit else 1-fraction[axis]
        sample = values[tuple(high[axis] if bit else low[axis] for axis,bit in enumerate(corner))].double()
        result += weight.reshape(len(points),*((1,)*len(trailing)))*sample
    return result


def _point_jets(values,level,points):
    points = np.asarray(points,dtype=float)
    dims = points.shape[1]
    steps = [(level["t1"]-level["t0"])/(level["nt"]-1),
        level["x_domain"][0]/level["nx"],level["x_domain"][1]/level["ny"]]
    bounds = [(level["t0"],level["t1"]),None,None]
    if dims==5:
        steps += [(level["v_bounds"][1]-level["v_bounds"][0])/(level["nvx"]-1),
                  (level["v_bounds"][3]-level["v_bounds"][2])/(level["nvy"]-1)]
        bounds += [(level["v_bounds"][0],level["v_bounds"][1]),(level["v_bounds"][2],level["v_bounds"][3])]
    jets = []
    for axis,h in enumerate(steps):
        offsets = np.tile([-1.,0.,1.],(len(points),1))
        weights = np.tile([-1.,0.,1.],(len(points),1))/(2*h)
        if bounds[axis] is not None:
            left = points[:,axis]-h<bounds[axis][0]
            right = points[:,axis]+h>bounds[axis][1]
            offsets[left],weights[left] = [0.,1.,2.],np.array([-3.,4.,-1.])/(2*h)
            offsets[right],weights[right] = [-2.,-1.,0.],np.array([1.,-4.,3.])/(2*h)
        samples = []
        for column in range(3):
            q = points.copy(); q[:,axis] += offsets[:,column]*h
            # Decimal arithmetic at the endpoint can overshoot by one ulp.
            if bounds[axis] is not None:
                lower,upper = bounds[axis]
                roundoff = 8*np.finfo(float).eps*max(1.,abs(lower),abs(upper))
                if ((q[:,axis]<lower-roundoff)|(q[:,axis]>upper+roundoff)).any():
                    raise ValueError("requested grid-step finite-difference stencil does not fit common domain")
                q[:,axis] = np.clip(q[:,axis],lower,upper)
            sampled = _interpolate(values,level,q)
            w = torch.as_tensor(weights[:,column],device=sampled.device)
            samples.append(sampled*w.reshape(len(q),*((1,)*(sampled.ndim-1))))
        jets.append(sum(samples))
    return torch.stack(jets,dim=-1),steps


def _raw_observations(solution,level,points,*,tail_width):
    f = solution["f"][0]
    if tuple(f.shape)!=(level["nt"],level["nx"],level["ny"],level["nvx"],level["nvy"]) or not torch.isfinite(f).all():
        raise ValueError("finite complete raw numerical trajectory required")
    vx = torch.linspace(level["v_bounds"][0],level["v_bounds"][1],level["nvx"],device=f.device,dtype=torch.float64)
    vy = torch.linspace(level["v_bounds"][2],level["v_bounds"][3],level["nvy"],device=f.device,dtype=torch.float64)
    wx,wy = torch.ones_like(vx),torch.ones_like(vy)
    wx[[0,-1]] *= .5; wy[[0,-1]] *= .5
    weights = wx[:,None]*wy[None,:]*(vx[1]-vx[0])*(vy[1]-vy[0])
    speed2 = vx[:,None]**2+vy[None,:]**2
    edge = ((vx[:,None]<=vx[0]+tail_width)|(vx[:,None]>=vx[-1]-tail_width)
            |(vy[None,:]<=vy[0]+tail_width)|(vy[None,:]>=vy[-1]-tail_width))
    fields,tail = [],[]
    for frame in f:
        frame = frame.double()
        moments = [torch.sum(frame*weights*factor,dim=(-2,-1)) for factor in
                   (1.,vx[:,None],vy[None,:],.5*speed2)]
        fields.append(torch.stack(moments,-1))
        tail.append({"box_mass_spatial_mean":float(moments[0].mean()),
            "box_K_spatial_mean":float(moments[3].mean()),
            "fixed_tail_band_mass_spatial_mean":float((frame*weights*edge).sum((-2,-1)).mean()),
            "fixed_tail_band_K_spatial_mean":float((frame*weights*edge*.5*speed2).sum((-2,-1)).mean()),
            "boundary_f_maximum":float(torch.stack((frame[...,0,:].max(),frame[...,-1,:].max(),
                frame[...,:,0].max(),frame[...,:,-1].max())).max())})
    macro = torch.stack(fields)
    f_values = _interpolate(f,level,points)
    macro_values = _interpolate(macro,level,np.asarray(points)[:,:3])
    f_jets,f_steps = _point_jets(f,level,points)
    macro_jets,macro_steps = _point_jets(macro,level,np.asarray(points)[:,:3])
    values = {"f":f_values,**{name:macro_values[:,i] for i,name in enumerate(("rho","jx","jy","K"))}}
    for axis,name in enumerate(("dt","dx","dy","dvx","dvy")): values[name+"_f"] = f_jets[:,axis]
    for i,name in enumerate(("rho","jx","jy","K")):
        for axis,coordinate in enumerate(("dt","dx","dy")): values[coordinate+"_"+name] = macro_jets[:,i,axis]
    if any(not torch.isfinite(x).all() for x in values.values()): raise ValueError("nonfinite raw comparison values")
    uniform = solution.get("rho")
    discrepancy = None if uniform is None else float(((uniform[0].double()-macro[...,0])**2).mean().sqrt())
    return {"values":_jsonable(values),"point_count":len(points),
        "raw_derivative_method":"2nd-order grid-step coordinate differences of explicitly multilinear-interpolated raw f/macros; one-sided t/v endpoints",
        "f_jet_steps":f_steps,"macro_jet_steps":macro_steps,
        "velocity_moment_scope":"trapezoidal on identical finite v box, not full R2",
        "returned_solver_uniform_rho_minus_trapezoid_rms":discrepancy,
        "tail":{"fixed_physical_band_width":tail_width,"frames":tail,
                "outside_domain_tail_error_bounded":False,"domain_expansion_performed":False},
        "full_native_projection_used":False}


def _native_observations(model,latent,condition,points,modules):
    q = torch.as_tensor(points,dtype=torch.float64,device=next(model.parameters()).device).clone().requires_grad_(True)
    with torch.enable_grad():
        fields = model.fields(q,torch.as_tensor(latent,device=q.device),equation_condition=condition)
        moments = modules["continuous_macros"].torch_squared_hermite_moments(torch,
            fields["hermite_coefficients"].double(),modes=model.hermite_modes,
            temperature=fields["equation_condition"][:,1].double(),
            floor_mixture=model.config.velocity_floor_mixture)
        rho = fields["rho"].double()
        values = {"f":fields["f"].double(),"rho":rho,
            "jx":rho*moments["mean_velocity"][:,0],"jy":rho*moments["mean_velocity"][:,1],
            "K":rho*moments["kinetic_energy"]}
        for name in tuple(values):
            gradient = torch.autograd.grad(values[name].sum(),q,retain_graph=True)[0]
            for axis,coordinate in enumerate(("dt","dx","dy","dvx","dvy") if name=="f" else ("dt","dx","dy")):
                values[coordinate+"_"+name] = gradient[:,axis]
    if any(not torch.isfinite(x).all() for x in values.values()): raise ValueError("nonfinite stored N14 reference observation")
    return {"values":_jsonable(values),"macro_velocity_scope":"full R2 analytic Hermite moments",
        "derivative_method":"native continuous physical-coordinate autodiff",
        "comparison_caveat":"raw finite-box versus N14 fullspace moments combines projection, tail and quadrature errors"}


def compare_observations(reference,candidate):
    if set(reference["values"])!=set(candidate["values"]): raise ValueError("comparison field/derivative keys differ")
    report = {}
    for name,x in reference["values"].items():
        a,b = np.asarray(x,dtype=float),np.asarray(candidate["values"][name],dtype=float)
        if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all(): raise ValueError("invalid shared-query comparison")
        gap = b-a
        scale = float(np.sqrt(np.mean(a*a)))
        error = float(np.sqrt(np.mean(gap*gap)))
        report[name] = {"absolute_rms":error,"reference_rms":scale,"signed_mean":float(gap.mean()),
            "maximum_absolute":float(np.max(np.abs(gap))),"relative_rms":None if scale==0 else error/scale,
            "count":int(a.size)}
    return report


def _default_points(level,condition):
    t0,t1 = level["t0"],level["t1"]
    theta = math.sqrt(float(condition[1]))
    return np.asarray([[t0+fraction*(t1-t0),x*level["x_domain"][0],y*level["x_domain"][1],vx*theta,vy*theta]
        for fraction in (0.,.25,.5,.75,1.)
        for x,y,vx,vy in ((.173,.389,.41,-.29),(.613,.827,-.61,.19))])


def refine_reference_samples(corpus_root,role,*,expected_manifest_sha256,output,
        rows=None,max_cases=1,device="cuda",points_txyvv=None,levels=None,resume=False):
    """Scheduler-only few-root benchmark/refinement with incremental receipts.

    Default max_cases=1 must be explicitly raised to inspect more strata.
    Failed root/level receipts are retained; no automatic third level or retries.
    """
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_TMPDIR"):
        raise RuntimeError("reference refinement requires a Slurm compute allocation")
    if role not in ROLES: raise ValueError("only explicit physical-reference roles may be refined")
    output,corpus = Path(output),Path(corpus_root).resolve(strict=True)
    if output.is_symlink() or (output.exists() and not resume): raise ValueError("fresh refinement output required unless explicit resume")
    if resume and not (output/"frozen-plan.json").is_file(): raise ValueError("resume requires an existing frozen refinement plan")
    if output.resolve().is_relative_to(corpus): raise ValueError("refinement must not write inside reference corpus")
    from .diffusion_data import read_manifest,_checked_file
    manifest = read_manifest(corpus,verify_hashes=False)
    frozen = {str(corpus/"manifest.json"):_file_sha(corpus/"manifest.json")}
    seed_plan = _checked_file(corpus,manifest["seed_plan"],True)
    frozen[str(seed_plan)] = manifest["seed_plan"]["sha256"]
    files,arrays = manifest["splits"][role]["files"],{}
    for key,entry in files.items():
        path = _checked_file(corpus,entry,True)
        frozen[str(path)] = entry["sha256"]
        arrays[key] = np.load(path,mmap_mode="r",allow_pickle=False)
    selected = select_stratified_rows(arrays["stratum_ids"],rows=rows,max_cases=max_cases)
    levels = validate_levels(default_levels(manifest["provenance"]["generation_config"]) if levels is None else levels,
                             manifest["provenance"]["generation_config"])
    api,modules = _modules()
    for name,digest in manifest["provenance"]["physical_module_sha256"].items():
        module = importlib.import_module("agent_verifier.vpfp."+name)
        frozen[str(Path(module.__file__).resolve())] = digest
    base_asset = manifest["checkpoints"]["base_decoder"]
    if not Path(base_asset["path"]).is_absolute() or Path(base_asset["path"]).is_symlink():
        raise ValueError("base decoder must be an explicit nonsymlinked frozen asset")
    frozen[str(Path(base_asset["path"]).resolve())] = base_asset["sha256"]
    payload = torch.load(base_asset["path"],map_location="cpu",weights_only=True)
    decoder_config = modules["continuous_hermite"].VPFPContinuousHermiteConfig.from_dict(payload["decoder_config"])
    if _jsonable(decoder_config.to_dict())!=_jsonable(manifest["provenance"]["decoder_config"]):
        raise ValueError("stored reference decoder config differs from pinned base")
    domain_receipt = reference_domain_receipt(manifest["provenance"]["generation_config"],decoder_config,levels=levels)
    for module in modules.values():
        frozen[str(Path(module.__file__).resolve())] = _file_sha(module.__file__)
    frozen[str(Path(__file__).resolve())] = _file_sha(__file__)
    model = api.build_deterministic_direct_model(decoder_config,device=device)
    equation_config = modules["high_fidelity"].VPFPHighFidelityConfig.from_dict(manifest["provenance"]["equation_config"])
    if equation_config.equation_family!=modules["high_fidelity"].RICH_EQUATION_FAMILY:
        raise ValueError("refinement must use the original supported rich equation family")
    roots = [{"row":i,"equation_seed":int(arrays["equation_seeds"][i]),
              "case_id":str(arrays["case_ids"][i]),"stratum_id":int(arrays["stratum_ids"][i]),
              "condition":arrays["conditions"][i].tolist()} for i in selected]
    for root in roots:
        points = _default_points(levels[0],root["condition"]) if points_txyvv is None else np.array(points_txyvv,dtype=float,copy=True)
        if points.ndim!=2 or points.shape[1]!=5 or not 1<=len(points)<=64 or not np.isfinite(points).all():
            raise ValueError("1..64 finite common physical phase points required")
        root["points_txyvxvy"] = points.tolist()
        root["stored_N14_row_sha256"] = sha256(np.asarray(arrays["coefficients"][root["row"]]).tobytes()).hexdigest()
        batches = [b for b in manifest["splits"][role].get("batches",[])
                   if b["start"]<=root["row"]<b["start"]+b["count"]]
        root["original_collection_batch"] = None if not batches else {
            "start":batches[0]["start"],"count":batches[0]["count"],
            "solver":batches[0].get("diagnostics",{}).get("solver")}
    plan = {"schema":SCHEMA,"source_manifest_sha256":expected_manifest_sha256,"role":role,
        "selection":"explicit rows" if rows is not None else "first row per sorted condition stratum, then max_cases",
        "roots":roots,"levels":levels,"domain_coordinate_receipt":domain_receipt,"frozen_source_files":frozen,"max_cases":max_cases,
        "feature_selection_used":False,"raw_numerical_solution_not_N14_is_refinement_comparator":True,
        "points_policy":"explicit common physical points" if points_txyvv is not None else "fixed two probes at five physical times"}
    if resume:
        if json.loads((output/"frozen-plan.json").read_text())!=_jsonable(plan):
            raise ValueError("refinement resume plan/input/code identity changed")
    else:
        output.mkdir(parents=True)
        _write(output/"frozen-plan.json",plan)
    checkpoint_path = output/"level-checkpoints.json"
    if resume:
        checkpoint = json.loads(checkpoint_path.read_text())
        for item in checkpoint["completed_levels"].values():
            path = output/item["file"]
            if path.parent!=output or not path.is_file():
                raise ValueError("completed refinement level report changed")
    else:
        checkpoint = {"frozen_plan_sha256":_file_sha(output/"frozen-plan.json"),
            "completed_levels":{},"solve_attempts":[],"new_pde_solves":0}
        _write(checkpoint_path,checkpoint)
    started = walltime.monotonic()
    report = {"schema":SCHEMA,"execution_passed":True,"numerical_convergence_unresolved":True,
        "scientific_regime_qualified":False,"all_reference_rows_refined":False,
        "frozen_plan_sha256":_file_sha(output/"frozen-plan.json"),"source_manifest_sha256":expected_manifest_sha256,
        "role":role,"domain_coordinate_receipt":domain_receipt,"requested_root_count":len(roots),"cases":[],
        "new_pde_solves":checkpoint["new_pde_solves"],"new_pde_solve_attempts":len(checkpoint["solve_attempts"]),
        "explicit_resume":resume,"previous_attempts_preserved":checkpoint["solve_attempts"],
        "limitations":["Two simultaneous grid/time/source-interpolation/precision changes are not isolated factors.",
            "Two levels cannot establish asymptotic order/GCI or a full-domain tail bound.",
            "The solver family is shared; this is independent resolution, not an independent numerical algorithm.",
            "Positive repair mass/steps and rho quadrature differences must be inspected, not hidden.",
            "Stored N14 versus raw coarse is reported separately and is not called pure projection error.",
            "Single-root replay can differ from original batch-minimum CFL stepping, even at the same grid/cap.",
            "Only selected roots, not all512 references, are refined."],
        "not_executed":["third_resolution","velocity_domain_expansion","candidate_adapted_tolerance"]}
    for root in roots:
        row = {**root,"execution_passed":False,"levels":[],"numerical_convergence_unresolved":True}
        try:
            spec = modules["high_fidelity"].sample_continuous_equation_spec(seed=root["equation_seed"],
                config=equation_config,defer_rho_log_partition=True)
            if int(spec["seed"])!=root["equation_seed"] or not np.array_equal(
                    np.asarray([spec["collision_nu"],spec["temperature"]],dtype=np.float32),
                    np.asarray(root["condition"],dtype=np.float32)):
                raise ValueError("same equation seed no longer reconstructs original physical conditions")
            row["continuous_equation_spec_sha256"] = _sha_json(spec)
            row["continuous_equation_spec"] = spec
            _write(output/f"root-{root['row']}-equation.json",spec)
            points = np.asarray(root["points_txyvxvy"],dtype=float)
            tail_width = 2*max((levels[0]["v_bounds"][1]-levels[0]["v_bounds"][0])/(levels[0]["nvx"]-1),
                              (levels[0]["v_bounds"][3]-levels[0]["v_bounds"][2])/(levels[0]["nvy"]-1))
            observations = []
            for level in levels:
                key = f"root-{root['row']}-{level['name']}"
                if key in checkpoint["completed_levels"]:
                    cached = json.loads((output/checkpoint["completed_levels"][key]["file"]).read_text())
                    if (cached.get("execution_passed") is not True or cached["metadata"]["level"]!=level
                            or cached["metadata"]["same_unrounded_equation_spec_sha256"]!=row["continuous_equation_spec_sha256"]):
                        raise ValueError("completed refinement level configuration/spec changed")
                    row["levels"].append(cached); observations.append(cached["observations"])
                    continue
                if (output/(key+".json")).exists():
                    raise ValueError("unregistered level artifact retained; refusing overwrite or automatic duplicate solve")
                item = {"name":level["name"],"execution_passed":False}
                row["levels"].append(item)
                phase_started = walltime.monotonic()
                report["new_pde_solve_attempts"] += 1
                attempt = {"root_row":root["row"],"level":level["name"],"status":"running",
                    "index":len(checkpoint["solve_attempts"])}
                checkpoint["solve_attempts"].append(attempt); _write(checkpoint_path,checkpoint)
                solution,metadata = _solve_level(spec,level,modules,device)
                report["new_pde_solves"] += 1
                checkpoint["new_pde_solves"] += 1
                item.update(metadata=metadata,numerical_solver_returned=True)
                measured = _raw_observations(solution,level,points,tail_width=tail_width)
                item.update(execution_passed=True,metadata=metadata,observations=measured,
                    wall_seconds=walltime.monotonic()-phase_started)
                observations.append(measured)
                del solution
                saved = output/(key+".json"); _write(saved,item)
                checkpoint["completed_levels"][key] = {"file":saved.name,"sha256":_file_sha(saved)}
                attempt["status"] = "completed"; _write(checkpoint_path,checkpoint)
            original = (root.get("original_collection_batch") or {}).get("solver")
            replay = row["levels"][0]["metadata"]["solver_diagnostics"]
            row["coarse_replay_batch_check"] = {
                "original_batch_solver_receipt_available":original is not None,
                "same_substeps_by_frame":None if original is None else original.get("substeps_by_frame")==replay.get("substeps_by_frame"),
                "original_batch_bitwise_replay_claimed":False,
                "interpretation":"projection/interpolation/tail/roundoff and possibly single-root versus batch CFL stepping are confounded"}
            stored = _mapped_native_observations(model,np.array(arrays["coefficients"][root["row"]],copy=True),
                root["condition"],points,modules,domain_receipt)
            row["stored_N14_observations"] = stored
            row["stored_N14_vs_replayed_raw_coarse"] = compare_observations(observations[0],stored)
            row["raw_coarse_vs_refined_same_domain"] = compare_observations(observations[0],observations[1])
            row["execution_passed"] = True
        except Exception as error:
            row["failure"] = {"type":type(error).__name__,"message":str(error)}
            report["execution_passed"] = False
            if row["levels"] and not row["levels"][-1]["execution_passed"]:
                row["levels"][-1]["failure"] = row["failure"]
            if checkpoint["solve_attempts"] and checkpoint["solve_attempts"][-1]["status"]=="running":
                checkpoint["solve_attempts"][-1].update(status="failed",failure=row["failure"])
                _write(checkpoint_path,checkpoint)
        report["cases"].append(row)
        _write(output/"result.json",report)
    report["wall_seconds"] = walltime.monotonic()-started
    report["completed_root_count"] = sum(row["execution_passed"] for row in report["cases"])
    _write(output/"result.json",report)
    return _jsonable(report)
