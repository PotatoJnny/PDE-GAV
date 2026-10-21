"""Two predeclared weak roots: raw dense-time U and global finite-box budgets.

This is a development representation check, not new training or 512-root
confirmation. Retain raw-solver discretization/tail/repair uncertainty instead
of treating the old cubic macro interpolant as physical truth.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import importlib
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
sys.path.insert(0, str(ROOT / "src"))
from pde_agent_verifier_solver.plugins.vpfp import reference_refinement as ref
from pde_agent_verifier_solver.plugins.vpfp.coherent_velocity_dynamics import (
    VPFPVelocityDynamicsConfig, fit_velocity_dynamics, evaluate_velocity_dynamics)

SOURCE = Path("${PDE_GAV_RUN_ROOT}/vpfp-scaleup-weak-5241246/corpus")
SOURCE_SHA = "0e322cd70bdd2558e0512911e2199a45acff964499ba3ba6a18b901b432e6db8"
CROSS = Path("${PDE_GAV_RUN_ROOT}/vpfp-weak-dynamics-diagnostic-5287092/pipeline/cross-diagnostic")
CACHE = Path("${PDE_GAV_RUN_ROOT}/vpfp-weak-regime-coherent-5268892/macro-targets")
STRATA = (0, 7)
DEGREES = (4, 8, 16)
SCHEMA = "vpfp_weak_two_root_raw_time_velocity_preflight_v1"


def file_sha(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("regular nonsymlinked bound input required")
    return ref._file_sha(path)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ref._write(path, value)


def levels_for(generation):
    coarse, fine = ref.default_levels(generation)
    if coarse["nt"] != 33:
        raise ValueError("original weak33-frame reference required")
    dense = {**coarse, "name": "same_grid_dense_time", "nt": 129}
    fine = {**fine, "name": "refined_grid_dense_time", "nt": 257}
    return [coarse, dense, fine]


def freeze_inputs():
    from pde_agent_verifier_solver.plugins.vpfp.diffusion_data import read_manifest
    manifest = read_manifest(SOURCE, verify_hashes=False)
    if manifest["manifest_sha256"] != SOURCE_SHA:
        raise ValueError("original weak corpus identity changed")
    frozen = {}
    def bind(path, expected=None):
        digest = file_sha(path)
        if expected is not None and digest != expected:
            raise ValueError("bound input changed: " + str(path))
        frozen[str(path)] = digest
    bind(SOURCE / "manifest.json")
    bind(CROSS / "frozen-plan.json")
    cross = json.loads((CROSS / "frozen-plan.json").read_text())
    if cross["source_manifest_sha256"] != SOURCE_SHA or cross["source_split"] != "validation":
        raise ValueError("same opened cross-diagnostic validation roots required")
    roots = [row for row in cross["selected_roots"] if row["stratum_id"] in STRATA]
    if [r["stratum_id"] for r in roots] != list(STRATA):
        raise ValueError("exact predeclared low/high condition roots required")
    arrays = {}
    for key in ("conditions", "equation_seeds", "case_ids", "stratum_ids"):
        entry = manifest["splits"]["validation"]["files"][key]
        bind(SOURCE / entry["path"], entry["sha256"])
        arrays[key] = np.load(SOURCE / entry["path"], mmap_mode="r", allow_pickle=False)
    cache_receipt = json.loads((CACHE / "validation.json").read_text())
    bind(CACHE / "validation.json")
    bind(CACHE / "validation.npy", cache_receipt["sha256"])
    prior = json.loads((CACHE.parent / "frozen-plan.json").read_text())
    bind(CACHE.parent / "frozen-plan.json")
    if prior["source_manifest_sha256"] != SOURCE_SHA or prior["macro_targets"]["validation"] != cache_receipt:
        raise ValueError("macro cache must be the original frozen extraction")
    macro = np.load(CACHE / "validation.npy", mmap_mode="r", allow_pickle=False)
    if (macro.shape != (128,5*33*169) or macro.dtype != np.dtype("float32")
            or len(cache_receipt["case_ids"]) != 128 or len(cache_receipt["sample_ids"]) != 128):
        raise ValueError("exact128x33-node original macro cache layout required")
    for row in roots:
        i = row["row"]
        if (str(arrays["case_ids"][i]) != row["root_id"]
                or int(arrays["equation_seeds"][i]) != row["equation_seed"]
                or int(arrays["stratum_ids"][i]) != row["stratum_id"]
                or cache_receipt["case_ids"][i] != row["root_id"]
                or cache_receipt["sample_ids"][i] != row["equation_seed"]
                or not np.array_equal(arrays["conditions"][i], np.asarray(row["conditions"], dtype=np.float32))):
            raise ValueError("cross root ancestry or condition differs")
    _, modules = ref._modules()
    for name, digest in manifest["provenance"]["physical_module_sha256"].items():
        bind(Path(importlib.import_module("agent_verifier.vpfp." + name).__file__), digest)
    for module in modules.values():
        bind(Path(module.__file__))
    bind(Path(__file__))
    bind(ROOT / "src/pde_agent_verifier_solver/plugins/vpfp/reference_refinement.py")
    bind(ROOT / "src/pde_agent_verifier_solver/plugins/vpfp/coherent_velocity_dynamics.py")
    domain = ref.reference_domain_receipt(manifest["provenance"]["generation_config"],
                                          manifest["provenance"]["decoder_config"])
    return manifest, roots, macro, modules, frozen, domain


def solve(spec, level, modules, device):
    operators, solver = modules["operators"], modules["torch_reference"]
    grid = operators.VPFPGrid(nx=level["nx"], ny=level["ny"], nvx=level["nvx"], nvy=level["nvy"],
        x_domain=tuple(level["x_domain"]), v_bounds=tuple(level["v_bounds"]), dtype=level["dtype"])
    times = operators.VPFPTime(t0=level["t0"], t1=level["t1"], nt=level["nt"])
    cfg = solver.VPFPTorchSolverConfig(**{k:level[k] for k in
        ("internal_dt_max", "dtype", "cfl", "positivity_floor", "rho_floor", "source_interpolation")})
    packed = solver.pack_torch_equation_specs([spec], dtype=level["dtype"])
    prepared = solver.evaluate_torch_equation_parameter_batch(packed, grid=grid, time=times, device=device)
    solution = solver.solve_torch_vpfp_batch(prepared, grid=grid, time=times, config=cfg, device=device)
    return solution, prepared


def global_moments(fields, level):
    """Independent trapezoid-v, uniform-periodic-x mean; physical volume noted."""
    if tuple(fields.shape) != (level["nt"], level["nx"], level["ny"], level["nvx"], level["nvy"]):
        raise ValueError("complete raw trajectory required")
    vx = torch.linspace(*level["v_bounds"][:2], level["nvx"], dtype=torch.float64, device=fields.device)
    vy = torch.linspace(*level["v_bounds"][2:], level["nvy"], dtype=torch.float64, device=fields.device)
    wx, wy = torch.ones_like(vx), torch.ones_like(vy)
    wx[[0,-1]] *= .5; wy[[0,-1]] *= .5
    weight = wx[:,None]*wy[None,:]*(vx[1]-vx[0])*(vy[1]-vy[0])
    factors = (torch.ones_like(weight), vx[:,None], vy[None,:], .5*(vx[:,None]**2+vy[None,:]**2))
    values = []
    for frame in fields:
        spatial_mean = frame.double().mean(dim=(0,1))
        values.append(torch.stack([(spatial_mean*weight*factor).sum() for factor in factors]))
    result = torch.stack(values).cpu().numpy()
    if not np.isfinite(result).all():
        raise ValueError("nonfinite independent full-spatial finite-v moments")
    return result


def rms(values):
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values*values)))


def tail_report(fields,level):
    """Fixed physical1.6 tail band across levels; not a domain-expansion bound."""
    vx = torch.linspace(*level["v_bounds"][:2],level["nvx"],device=fields.device,dtype=torch.float64)
    vy = torch.linspace(*level["v_bounds"][2:],level["nvy"],device=fields.device,dtype=torch.float64)
    wx,wy = torch.ones_like(vx),torch.ones_like(vy)
    wx[[0,-1]] *= .5; wy[[0,-1]] *= .5
    weight = wx[:,None]*wy[None,:]*(vx[1]-vx[0])*(vy[1]-vy[0])
    band = ((vx[:,None] <= vx[0]+1.6) | (vx[:,None] >= vx[-1]-1.6)
            | (vy[None,:] <= vy[0]+1.6) | (vy[None,:] >= vy[-1]-1.6))
    speed2 = vx[:,None]**2+vy[None,:]**2
    entries = []
    for frame in fields:
        spatial_mean = frame.double().mean(dim=(0,1))
        entries.append({"tail_mass_spatial_mean":float((spatial_mean*weight*band).sum()),
            "tail_K_spatial_mean":float((spatial_mean*weight*band*.5*speed2).sum()),
            "boundary_f_maximum":float(torch.stack((frame[...,0,:].max(),frame[...,-1,:].max(),
                frame[...,:,0].max(),frame[...,:,-1].max())).max())})
    return {"physical_tail_band_width":1.6,"by_time":entries,"bounds":level["v_bounds"],
            "velocity_domain_expanded":False}


def measure(solution, prepared, level, nu):
    moments = global_moments(solution["f"][0], level)
    source = global_moments(prepared["source_frames"][0], level)
    times = np.linspace(level["t0"], level["t1"], level["nt"])
    if (moments[:,0] <= 0).any():
        raise ValueError("positive global mass required")
    U = moments[:,1:3]/moments[:,:1]
    dt = np.gradient(moments, times, axis=0, edge_order=2)
    dU = dt[:,1:3]/moments[:,:1]-U*dt[:,:1]/moments[:,:1]
    # Explicit mass-source correction; not silently assume numerical M'=0.
    a_source = (source[:,1:3]-U*source[:,:1])/moments[:,:1]
    a_proxy = dU+nu*U
    return {"times":times.tolist(), "moment_order":["M/area","Jx/area","Jy/area","K/area"],
        "spatial_area":math.prod(level["x_domain"]), "moments":moments.tolist(), "source_moments":source.tolist(),
        "U":U.tolist(), "dU_fd":dU.tolist(), "U_drive_proxy":a_proxy.tolist(),
        "source_acceleration_mass_corrected":a_source.tolist(),
        "global_mass_balance_residual":(dt[:,0]-source[:,0]).tolist(),
        "momentum_balance_remainder":(dt[:,1:3]+nu*moments[:,1:3]-source[:,1:3]).tolist(),
        "velocity_drive_minus_source_acceleration":(a_proxy-a_source).tolist(),
        "U_rms":rms(U), "drive_proxy_rms":rms(a_proxy), "source_acceleration_rms":rms(a_source),
        "minimum_f":float(solution["f"].min()),
        "collision_nu_used_by_solver":nu,"finite_box_tail":tail_report(solution["f"][0],level),
        "source_from_original_equation_not_manufactured_from_solution":True,
        "scope":"entire_periodic_spatial_grid_trapezoidal_finite_velocity_box_not_R2",
        "fd_method":"2nd_order_moment_differences_then_quotient_rule;one_sided_endpoints",
        "remainder_not_automatically_physics_failure":"includes_discretization_velocity_boundary_flux_electric_discretization_and_positive_repair",
        "solver_diagnostics":ref._jsonable(solution["diagnostics"])}


def compare_fit(node_U, dense, nu, degree):
    cfg = VPFPVelocityDynamicsConfig(drive_degree=degree,t0=dense["times"][0],t1=dense["times"][-1])
    values = torch.tensor(np.asarray(node_U).T[None], dtype=torch.float64)
    collision = torch.tensor([nu], dtype=torch.float64)
    fitted = fit_velocity_dynamics(values, collision, cfg)
    prediction = evaluate_velocity_dynamics(fitted.coordinates, collision,
        torch.tensor(dense["times"], dtype=torch.float64), cfg)
    U, dU, drive = [prediction[k][0].T.detach().numpy() for k in ("U","dU_dt","a")]
    truth = np.asarray(dense["U"])
    off = np.arange(len(truth)) % ((len(truth)-1)//32) != 0
    return {"degree":degree, "coordinate_count":cfg.latent_dimension,
        "U_dense_rms_gap":rms(U-truth), "U_offnode_rms_gap":rms((U-truth)[off]),
        "dU_vs_dense_moment_fd_rms_gap":rms(dU-np.asarray(dense["dU_fd"])),
        "a_vs_independently_integrated_source_rms_gap":rms(drive-np.asarray(dense["source_acceleration_mass_corrected"])),
        "a_rms":rms(drive), "a_derivative_energy":float(prediction["drive_derivative_mean_square"][0]),
        "U0_absolute_gap":rms(U[0]-truth[0]), "scientific_admission":False}


def run(arguments):
    if not os.environ.get("SLURM_JOB_ID", "").isdigit() or not os.environ.get("SLURM_TMPDIR"):
        raise ValueError("raw references require a Slurm compute allocation")
    output = Path(arguments.output).absolute()
    if (output.exists() or output.is_symlink() or not output.is_relative_to(Path("${PDE_GAV_RUN_ROOT}"))
            or any(output == p or output.is_relative_to(p) or p.is_relative_to(output) for p in (SOURCE,CROSS,CACHE))):
        raise ValueError("fresh separate scratch output required")
    if not math.isfinite(arguments.wall_seconds) or arguments.wall_seconds <= 0:
        raise ValueError("positive allocation wall limit required")
    output.mkdir(parents=True)
    started = time.monotonic()
    result = {"schema":SCHEMA, "execution_passed":False, "scientific_admission":False,
        "new_training":False, "attempted_pde_solves":0, "completed_pde_solves":0, "cases":[],
        "strata":list(STRATA), "degrees":list(DEGREES), "independent_root_count":2,
        "limitations":["two_opened_validation_roots_not_population_confirmation",
            "same_solver_family_and_finite_velocity_box_no_domain_expansion",
            "dense_output_also_refines_source_interpolation_and_changes_frame_aligned_substeps",
            "raw_refined_changes_spatial_velocity_precision_and_time_factors_jointly",
            "no_threshold_fit_and_no_automatic_training_qualification"]}
    def save():
        result["wall_seconds"] = time.monotonic()-started
        write(output/"result.json",result)
    save()
    try:
        manifest, roots, macro, modules, frozen, domain = freeze_inputs()
        levels = levels_for(manifest["provenance"]["generation_config"])
        write(output/"frozen-plan.json", {"schema":SCHEMA,"roots":roots,"levels":levels,
            "files":frozen,"domain_representation":domain,"selection":"existing_cross_first_validation_roots_strata0_and7_before_raw_result_access",
            "fit_targets":["existing_cached33_macro_U","same_grid_dense_raw_subsample33","refined_dense_raw_subsample33"]})
        equation_cfg = modules["high_fidelity"].VPFPHighFidelityConfig.from_dict(manifest["provenance"]["equation_config"])
        for root in roots:
            case = {"root":root,"levels":[],"fits":[],"execution_passed":False}
            result["cases"].append(case); save()
            spec = modules["high_fidelity"].sample_continuous_equation_spec(seed=root["equation_seed"],
                config=equation_cfg,defer_rho_log_partition=True)
            if not np.array_equal(np.asarray([spec["collision_nu"],spec["temperature"]],dtype=np.float32),
                                  np.asarray(root["conditions"],dtype=np.float32)):
                raise ValueError("equation seed/condition replay changed")
            write(output/f"stratum-{root['stratum_id']}/equation.json",spec)
            for level in levels:
                if time.monotonic()-started >= arguments.wall_seconds-60:
                    raise TimeoutError("allocation deadline; unstarted roots/levels not replaced")
                result["stage"] = f"stratum-{root['stratum_id']}/{level['name']}"
                result["attempted_pde_solves"] += 1; save()
                solution, prepared = solve(spec,level,modules,arguments.device)
                actual_nu = float(prepared["collision_nu"][0])
                measured = measure(solution,prepared,level,actual_nu)
                measured.update(unrounded_equation_nu=float(spec["collision_nu"]),
                    collision_nu_roundoff=actual_nu-float(spec["collision_nu"]),
                    bath_temperature_used_by_solver=float(prepared["temperature"][0]))
                item = {"level":level,"observations":measured,"execution_passed":True}
                case["levels"].append(item); result["completed_pde_solves"] += 1
                write(output/f"stratum-{root['stratum_id']}/{level['name']}.json",item)
                del solution, prepared
                save()
            cached = np.asarray(macro[root["row"],:66]).reshape(2,33).T
            for item in case["levels"][1:]:
                dense = item["observations"]
                for origin,nodes in (("cached_native33",cached),
                        ("raw_same_level_subsample33",np.asarray(dense["U"])[::(len(dense["U"])-1)//32])):
                    for degree in DEGREES:
                        case["fits"].append({"target":origin,"dense_level":item["level"]["name"],
                            **compare_fit(nodes,dense,dense["collision_nu_used_by_solver"],degree)})
            case["execution_passed"] = True; save()
        result.update(execution_passed=True,stage="complete")
    except Exception as error:
        result["failure"] = {"type":type(error).__name__,"reason":str(error),"traceback":traceback.format_exc()}
    finally:
        save()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    parser.add_argument("--wall-seconds",type=float,default=7200)
    return 0 if run(parser.parse_args(argv))["execution_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
