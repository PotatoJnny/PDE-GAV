"""Carrier-level VPFP research controls and fixed-probe time diagnostics.

A researcher supplies frozen physical roots, then decodes changed answers and
manufactures their actual sources. No feature vector is edited. Replicas and
descendants retain their physical-root identity and are not independent data.
These controls calibrate a research measurement's sensitivity, not thresholds
chosen against a candidate, and not scientific regime/admission definitions.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import math

import numpy as np
import torch

from .moving_frame import (MovingFrameDecoder, extract_macro_controls, half_modes,
    macro_fourier_basis, pack_macro_controls, spatial_grid, split_macro_controls)
from .moving_frame_backend import evaluate_moving_frame_macros, manufacture_moving_frame_source
from .value_control_diffusion import extract_value_controls, lift_value_controls

SCHEMA = "vpfp_physical_carrier_control_v1"
TIME_SCHEMA = "vpfp_fixed_probe_temporal_frequency_v1"


class UndefinedPhysicalControl(ValueError):
    """An undefined intervention, with root/hash receipt for caller persistence."""
    def __init__(self,message,receipt):
        super().__init__(message)
        self.receipt = receipt


def _json_copy(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _tensor_sha(value):
    value = value.detach().cpu().contiguous()
    header = json.dumps({"dtype":str(value.dtype), "shape":list(value.shape)}, sort_keys=True).encode()
    return sha256(header+b"\0"+value.numpy().tobytes()).hexdigest()


def _inputs(decoder, carrier, conditions, root_ids, provenance, *, allow_generated=False):
    if not isinstance(decoder, MovingFrameDecoder) or set(carrier)!={"m","z"}:
        raise ValueError("an actual versioned A decoder and complete m/z carrier are required")
    if any(not isinstance(v,torch.Tensor) or v.dtype not in (torch.float32,torch.float64)
           or v.ndim!=2 or not len(v) or not torch.isfinite(v).all() for v in carrier.values()):
        raise ValueError("finite nonempty FP32/FP64 native carrier matrices required")
    decoder.carrier_to_payload(carrier)
    conditions = torch.as_tensor(conditions)
    if (conditions.dtype not in (torch.float32,torch.float64)
            or conditions.shape!=(len(carrier["m"]),2)
            or not torch.isfinite(conditions).all() or (conditions[:,0]<0).any()
            or (conditions[:,1]<=0).any()):
        raise ValueError("finite physical [nu>=0,temperature>0] rows required")
    roots = tuple(root_ids)
    if len(roots)!=len(conditions) or any(not isinstance(x,str) or not x.strip() for x in roots):
        raise ValueError("one explicit physical root ID per carrier row required")
    provenance = _json_copy(provenance)
    digest = provenance.get("source_manifest_sha256","")
    if (len(digest)!=64 or any(x not in "0123456789abcdef" for x in digest)
            or provenance.get("data_origin") not in (("physical_reference","synthetic_fixture","generated_answer")
                if allow_generated else ("physical_reference","synthetic_fixture"))
            or not isinstance(provenance.get("split"),str) or not provenance["split"]):
        raise ValueError("explicit frozen source SHA, split and physical/synthetic origin required")
    return {k:v.detach().clone() for k,v in carrier.items()}, conditions.detach().clone(), roots, provenance


def carrier_hashes(decoder, carrier, conditions):
    """Hash actual stored precision, condition and versioned decoder identity."""
    identity = _json_copy(decoder.identity())
    return [sha256(json.dumps({"schema":SCHEMA,"decoder":identity,
        "m":_tensor_sha(carrier["m"][i]),"z":_tensor_sha(carrier["z"][i]),
        "condition":_tensor_sha(conditions[i])},sort_keys=True,separators=(",",":")).encode()).hexdigest()
        for i in range(len(conditions))]


def _receipt(decoder,before,conditions,roots,provenance,after,indices,rule):
    original = carrier_hashes(decoder,before,conditions)
    selected_conditions = conditions[list(indices)].clone()
    output = carrier_hashes(decoder,after,selected_conditions)
    selected_roots = [roots[i] for i in indices]
    return {"schema":SCHEMA,"execution_passed":True,"control_defined":True,
        "decoder_identity":_json_copy(decoder.identity()),
        "source_provenance":provenance,"rule":_json_copy(rule),
        "lineage":[{"output_row":j,"source_row":i,"physical_root_id":roots[i],
                    "source_carrier_sha256":original[i],"output_carrier_sha256":output[j]}
                   for j,i in enumerate(indices)],
        "input_physical_root_count":len(set(roots)),"output_row_count":len(indices),
        "output_distinct_physical_root_count":len(set(selected_roots)),
        "output_root_multiplicities":dict(Counter(selected_roots)),
        "output_distinct_carrier_count":len(set(output)),
        "independence_unit":"original physical root, never control row; distinct IDs alone do not prove independence",
        "feature_arrays_modified":False,"full_native_z_retained":True,
        "new_training_steps":0,"new_pde_solves":0,"scientific_regime_qualified":False,
        "candidate_adapted_threshold":False}


def _result(decoder,before,conditions,roots,provenance,after,rule):
    indices = tuple(range(len(conditions)))
    return {"carrier":after,"conditions":conditions.clone(),
        "receipt":_receipt(decoder,before,conditions,roots,provenance,after,indices,rule)}


def _undefined(decoder,before,conditions,roots,provenance,kind,message):
    receipt = _receipt(decoder,before,conditions,roots,provenance,before,range(len(roots)),{"kind":kind})
    for row in receipt["lineage"]:
        row["output_carrier_sha256"] = None
    receipt.update(execution_passed=False,control_defined=False,failure=message,
        output_distinct_carrier_count=0,output_row_count=0)
    raise UndefinedPhysicalControl(message,receipt)


def index_carrier_control(decoder,carrier,conditions,*,root_ids,provenance,indices,
                          selection_rule,source_sampling_weights=None):
    """Apply a declared root index schedule, not a feature-space corruption.

    weights describe the source sampling law, NOT a claim that a finite index
    sequence exactly realizes it. Report both the law and realized counts.
    E.g. changed mixture frequencies, repeated roots or one-root collapse.
    """
    before,conditions,roots,provenance = _inputs(decoder,carrier,conditions,root_ids,provenance)
    indices = tuple(indices)
    if (not indices or any(type(i) is not int or not 0<=i<len(roots) for i in indices)
            or not isinstance(selection_rule,str) or not selection_rule.strip()):
        raise ValueError("explicit valid source indices and a declared selection rule required")
    weights = None
    if source_sampling_weights is not None:
        weights = np.asarray(source_sampling_weights,dtype=np.float64)
        if (weights.shape!=(len(roots),) or not np.isfinite(weights).all() or (weights<0).any()
                or not np.isfinite(weights.sum()) or weights.sum()<=0):
            raise ValueError("nonnegative finite root sampling weights with positive sum required")
        if any(weights[i]==0 for i in indices):
            raise ValueError("index schedule selects a root with zero declared probability")
        weights = (weights/weights.sum()).tolist()
    after = {k:v[list(indices)].clone() for k,v in before.items()}
    receipt = _receipt(decoder,before,conditions,roots,provenance,after,indices,
        {"kind":"root_index_selection","selection_rule":selection_rule,
         "indices":list(indices),"declared_source_probabilities":weights,
         "realized_source_counts":dict(Counter(indices)),
         "physical_change":"none within a selected root; population frequencies/support change",
         "same_root_copies_are_not_new_independent_samples":True})
    return {"carrier":after,"conditions":conditions[list(indices)].clone(),"receipt":receipt}


def reference_carrier_from_full(decoder,full,conditions,*,root_ids,provenance,fit_grid=33,chunk_size=128):
    """Extract A-reference m from the SAME full native root without discarding H.

    This is a compressed A(reference) control anchor, not a lossless copy of B.
    The full-value lift must be bitwise exact; m extraction errors remain those
    measured by the separate moving-frame representation preflight.
    """
    full = torch.as_tensor(full)
    conditions = torch.as_tensor(conditions)
    z = extract_value_controls(full,decoder.value_config,validate_slopes=True)
    if not torch.equal(lift_value_controls(z,decoder.value_config),full):
        raise ValueError("reference full-to-value lift is not bitwise exact")
    m,extraction = extract_macro_controls(decoder.base_decoder,z,conditions,decoder.macro_config,
        value_config=decoder.value_config,fit_grid=fit_grid,chunk_size=chunk_size)
    carrier = {"m":m.to(z),"z":z.clone()}
    carrier,conditions,roots,provenance = _inputs(decoder,carrier,conditions,root_ids,provenance)
    result = _result(decoder,carrier,conditions,roots,provenance,carrier,
        {"kind":"A_reference_extraction","fit_grid":fit_grid,"extraction":extraction,
         "full_native_root_sha256":[_tensor_sha(row) for row in full],
         "full_native_value_lift_bitwise":True,"A_equals_B_claimed":False})
    return result


def _seed(seed,root,digest):
    if type(seed) is not int or seed<0:
        raise ValueError("nonnegative integer control seed required")
    return int.from_bytes(sha256(f"{SCHEMA}:{seed}:{root}:{digest}".encode()).digest()[:8],"little")


def _node_geometry(decoder,carrier,conditions,grid_count,chunk_size):
    """Native rho grid and full macro coordinates; polynomial flow integrals.

    A rho has native finite Fourier support. The grid exceeds the maximum
    coordinate-wise wave number in rho * |local_velocity|^2, so its mean has
    no exact-arithmetic Fourier alias. This is NOT a quadrature claim for C/f/S.
    """
    cfg = decoder.macro_config
    band = decoder.value_config.rho_max_mode+2*cfg.max_mode
    if grid_count is None:
        grid_count = max(3,2*band+1)
    if type(grid_count) is not int or grid_count<=band or grid_count<2*cfg.max_mode+1:
        raise ValueError("node flow grid aliases rho times local-velocity-squared")
    if type(chunk_size) is not int or chunk_size<1:
        raise ValueError("positive chunk_size required")
    device = next(decoder.parameters()).device
    xy = spatial_grid(cfg,grid_count,device=device)
    basis = macro_fourier_basis(xy,cfg)[:,1:]
    times = torch.linspace(*cfg.time_domain,cfg.control_count,dtype=torch.float64,device=device)
    rows = []
    with torch.no_grad():
        for case in range(len(conditions)):
            density = []
            for t in times:
                q = torch.cat((t.expand(len(xy),1),xy,torch.zeros_like(xy)),1)
                pieces = [decoder.source_fields(part,carrier["z"][case:case+1],
                            conditions[case])["rho"].double() for part in q.split(chunk_size)]
                rho = torch.cat(pieces)
                if not torch.isfinite(rho).all() or (rho<=0).any():
                    raise ValueError("positive finite native rho required for flow energy")
                density.append(rho)
            rows.append(torch.stack(density))
    return torch.stack(rows),basis,{"grid_size":grid_count,"control_times":times.cpu().tolist(),
        "flow_product_maximum_coordinate_mode":band,
        "quadrature_scope":"rho-weighted polynomial flow moments only, not C/f/source",
        "no_epsilon_jitter_or_variance_floor":True}


def _centered_field(coefficients,basis,rho):
    field = torch.einsum("ctm,nm->tnc",coefficients,basis)
    mean = (rho[...,None]*field).sum(1)/rho.sum(1)[:,None]
    return field-mean[:,None,:]


def _inner(a,b,rho):
    return (rho*(a*b).sum(-1)).sum(-1)/rho.sum(-1)


def _probe_check(decoder,before,after,conditions):
    """Actual continuous A+MMS, never old Hermite source or modified features."""
    cfg = decoder.macro_config
    t0,t1 = cfg.time_domain
    h = (t1-t0)/(cfg.control_count-1)
    times = [t0,t0+.25*h,t0+.5*h,(t0+t1)/2,t1]
    reports = []
    for case in range(len(conditions)):
        q = torch.tensor([[t,.173*cfg.spatial_domain_lengths[0],.389*cfg.spatial_domain_lengths[1],
                           .41,-.29] for t in times],dtype=torch.float64,
                         device=next(decoder.parameters()).device)
        measurements = {}
        row = {"case_row":case,"points_txyvxvy":q.cpu().tolist(),"execution_passed":False,
               "actual_A_MMS_executed":False,"independent_residual_verification":False}
        try:
            for label,carrier in (("source",before),("control",after)):
                c = {k:v[case:case+1] for k,v in carrier.items()}
                fields = manufacture_moving_frame_source(decoder,q,c,conditions[case])
                measurements[label] = {k:fields[k].detach().double().cpu().tolist()
                    for k in ("rho","momentum","kinetic_energy","f","source","central_covariance")}
            row.update(execution_passed=True,actual_A_MMS_executed=True)
        except Exception as error:
            row["failure"] = {"type":type(error).__name__,"message":str(error)}
        reports.append({**row,**measurements})
    return reports


def _finish_control(decoder,before,after,conditions,roots,provenance,rule,check_mms):
    decoder.carrier_to_payload(after)
    result = _result(decoder,before,conditions,roots,provenance,after,rule)
    result["receipt"]["physics_probe_checks"] = _probe_check(decoder,before,after,conditions) if check_mms else []
    passed = all(row["execution_passed"] for row in result["receipt"]["physics_probe_checks"])
    result["receipt"].update(execution_passed=passed,control_defined=True)
    result["receipt"]["physics_probe_check_status"] = ("executed" if passed else "failed") if check_mms else "not_executed_explicitly_deferred"
    return result


def relative_phase_scramble(decoder,carrier,conditions,*,root_ids,provenance,seed,
                            preserve_node_flow_energy=True,grid_count=None,chunk_size=128,check_mms=True):
    """Scramble local-u relative phases; NOT a guaranteed coherence destroyer.

    Common angle per mode/row across velocity components and all times.
    Before optional energy matching this preserves each mode's unweighted
    squared amplitude at ALL times. Global U/low-k power can remain unchanged,
    so an audit of those alone need not detect this deliberately weaker control.
    """
    before,conditions,roots,provenance = _inputs(decoder,carrier,conditions,root_ids,provenance)
    u,local,chol = [x.double().clone() for x in split_macro_controls(before["m"],decoder.macro_config)]
    if not local.shape[-1]:
        raise ValueError("relative phase control has no non-DC modes")
    hashes = carrier_hashes(decoder,before,conditions)
    angles = np.stack([np.random.default_rng(_seed(seed,root,digest)).uniform(-np.pi,np.pi,local.shape[-1]//2)
                       for root,digest in zip(roots,hashes)])
    phase = torch.as_tensor(angles,device=local.device,dtype=local.dtype)[:,None,None,:]
    pairs = local.reshape(*local.shape[:-1],-1,2)
    cosine,sine = pairs[...,0],pairs[...,1]
    rotated = torch.stack((cosine*phase.cos()-sine*phase.sin(),
                          cosine*phase.sin()+sine*phase.cos()),-1).flatten(-2)
    geometry,ratios,energy_checks = None,None,[]
    if preserve_node_flow_energy:
        rho,basis,geometry = _node_geometry(decoder,before,conditions,grid_count,chunk_size)
        factors = []
        for case in range(len(conditions)):
            old = _centered_field(local[case].to(basis),basis,rho[case])
            new = _centered_field(rotated[case].to(basis),basis,rho[case])
            a,b = _inner(old,old,rho[case]),_inner(new,new,rho[case])
            if not torch.isfinite(a).all() or not torch.isfinite(b).all() or (a<0).any() or (b<0).any():
                raise ValueError("invalid flow energy; no clipping")
            if ((b==0)&(a!=0)).any():
                raise ValueError("undefined energy rematch: controlled local variance is zero")
            factor = torch.ones_like(a)
            mask = b>0
            factor[mask] = (a[mask]/b[mask]).sqrt()
            factors.append(factor)
        ratios = torch.stack(factors).to(rotated)
        rotated = rotated*ratios[:,None,:,None]
    after = {"m":pack_macro_controls(u,rotated,chol,decoder.macro_config).to(before["m"]),"z":before["z"].clone()}
    if preserve_node_flow_energy:
        final = split_macro_controls(after["m"],decoder.macro_config)[1].double()
        for case in range(len(conditions)):
            old = _centered_field(local[case].to(basis),basis,rho[case])
            new = _centered_field(final[case].to(basis),basis,rho[case])
            energy_checks.append({"source_local_velocity_squared":_inner(old,old,rho[case]).cpu().tolist(),
                                  "control_local_velocity_squared":_inner(new,new,rho[case]).cpu().tolist()})
    return _finish_control(decoder,before,after,conditions,roots,provenance,
        {"kind":"relative_phase_scramble","seed":seed,"half_modes":list(half_modes(decoder.macro_config.max_mode)),
         "angles_radians":angles.tolist(),"rotation":"[cos,sin] -> [cos*cos(phi)-sin*sin(phi),cos*sin(phi)+sin*cos(phi)]",
         "phase_shared_across_time_and_velocity_components":True,"guaranteed_detectable":False,
         "global_U_C_rho_full_z_unchanged":True,"node_flow_energy_rematched":preserve_node_flow_energy,
         "node_amplitude_factors":None if ratios is None else ratios.cpu().tolist(),
         "node_energy_checks":energy_checks,"geometry":geometry,
         "changes":"local waveform, phase relation to rho/C/shape, j, raw higher moments and MMS; optional node gain also changes time curves",
         "preserves":"full z and C; therefore local standardized/central shape; global rho-weighted U",
         "not_claimed":"single-factor intervention; destroyed global coherence; exact nonnode energy preservation"},check_mms)


def coherent_to_incoherent_flow(decoder,carrier,conditions,*,root_ids,provenance,
                                minimum_mode=3,grid_count=None,chunk_size=128,check_mms=True):
    """Transfer |U|^2 to rho-centered flow orthogonal to existing local flow.

    At each macro node use a predeclared high-k template and weighted
    Gram--Schmidt; no new floor or dimension removal. U becomes exactly zero.
    C/z stay unchanged. Energy equality is numerical at nodes, not all times.
    """
    before,conditions,roots,provenance = _inputs(decoder,carrier,conditions,root_ids,provenance)
    cfg = decoder.macro_config
    if type(minimum_mode) is not int or minimum_mode<3:
        raise ValueError("incoherent template must be outside coherent k<=2")
    modes = half_modes(cfg.max_mode)
    available = [i for i,(kx,ky) in enumerate(modes) if max(abs(kx),abs(ky))>=minimum_mode]
    if not available:
        _undefined(decoder,before,conditions,roots,provenance,"coherent_to_incoherent_flow",
            "undefined coherent transfer: macro layout has no requested high modes")
    u,local,chol = [x.double().clone() for x in split_macro_controls(before["m"],cfg)]
    rho,basis,geometry = _node_geometry(decoder,before,conditions,grid_count,chunk_size)
    changed = local.clone()
    rows = []
    # Four linearly independent templates at the first eligible mode; after
    # subtracting one local direction at least one remains nonzero in positive rho.
    mode = available[0]
    for case in range(len(conditions)):
        old_coeff = local[case].to(basis)
        old = _centered_field(old_coeff,basis,rho[case])
        old_energy = _inner(old,old,rho[case])
        if not torch.isfinite(old_energy).all() or (old_energy<0).any():
            raise ValueError("invalid original local flow energy")
        candidates = []
        for component in (0,1):
            for phase_index in (0,1):
                template = torch.zeros_like(old_coeff)
                template[component,:,2*mode+phase_index] = 1.
                h = _centered_field(template,basis,rho[case])
                coefficient = torch.zeros_like(old_energy)
                mask = old_energy>0
                coefficient[mask] = _inner(old,h,rho[case])[mask]/old_energy[mask]
                residual_coeff = template-coefficient[None,:,None]*old_coeff
                residual = _centered_field(residual_coeff,basis,rho[case])
                norm = _inner(residual,residual,rho[case])
                candidates.append((residual_coeff,residual,norm,component,phase_index,coefficient))
        norms = torch.stack([x[2] for x in candidates])
        chosen = norms.argmax(0)  # Maximum norm minimizes cancellation, not candidate-driven calibration.
        output = old_coeff.clone()
        records = []
        for ti in range(cfg.control_count):
            index = int(chosen[ti])
            residual_coeff,residual,norm,component,phase_index,projection = candidates[index]
            target = u[case,:,ti].square().sum().to(norm)
            if not torch.isfinite(norm[ti]) or norm[ti]<=0 or not torch.isfinite(target):
                _undefined(decoder,before,conditions,roots,provenance,"coherent_to_incoherent_flow",
                    "undefined coherent transfer: nonpositive/nonfinite orthogonal template energy")
            gain = (target/norm[ti]).sqrt()
            output[:,ti] += gain*residual_coeff[:,ti]
            records.append({"time_index":ti,"template_mode":list(modes[mode]),"component":component,
                "phase":"cos" if phase_index==0 else "sin","gain":float(gain),
                "gram_schmidt_projection_coefficient":float(projection[ti]),
                "template_local_inner_product":float(_inner(old,residual,rho[case])[ti]),
                "removed_U_squared":float(target)})
        changed[case] = output.to(changed)
        rows.append(records)
    after = {"m":pack_macro_controls(torch.zeros_like(u),changed,chol,cfg).to(before["m"]),"z":before["z"].clone()}
    actual_local = split_macro_controls(after["m"],cfg)[1].double()
    node_checks = []
    for case in range(len(conditions)):
        old = _centered_field(local[case].to(basis),basis,rho[case])
        new = _centered_field(actual_local[case].to(basis),basis,rho[case])
        old_energy = _inner(old,old,rho[case])+u[case].to(basis).square().sum(0)
        new_energy = _inner(new,new,rho[case])
        node_checks.append({"source_total_velocity_squared":old_energy.cpu().tolist(),
            "control_total_velocity_squared":new_energy.cpu().tolist(),
            "source_spatial_mean_flow_K":(.5*rho[case].mean(1)*old_energy).cpu().tolist(),
            "control_spatial_mean_flow_K":(.5*rho[case].mean(1)*new_energy).cpu().tolist(),
            "thermal_K_unchanged_pointwise_because_rho_and_C_unchanged":True,
            "maximum_absolute_energy_defect":float((old_energy-new_energy).abs().max())})
    return _finish_control(decoder,before,after,conditions,roots,provenance,
        {"kind":"coherent_to_incoherent_flow","minimum_mode":minimum_mode,"geometry":geometry,
         "template_and_gain":rows,"node_energy_checks":node_checks,"global_U_exactly_zero":True,
         "C_rho_full_z_unchanged":True,"new_epsilon_or_floor":False,
         "construction":"rho-centered high-mode template, weighted Gram-Schmidt against original local field, gain sqrt(|U|^2/template_variance)",
         "changes":"coherent momentum removed, local flow and Fourier spectrum increased/reorganized; raw higher moments, f and source change",
         "not_claimed":"pure high-frequency addition after Gram-Schmidt; single-factor intervention; exact nonnode K equality",
         "template_selection":"largest residual norm per time node among four fixed candidates; switching can alter time shape/source",
         "zero_original_U_is_noop_for_that_time":True},check_mms)


def drive_velocity(decoder,carrier,conditions,*,root_ids,provenance,
                   global_amplitude=(0.,0.),local_amplitude=(0.,0.),mode=(1,0),
                   cycles=1.,phase=0.,check_mms=True):
    """Add a physical-velocity sinusoid to U and a declared low-k cos mode.

    The sinusoid is sampled at existing macro controls; the actual continuous
    curve is the native C1 Hermite interpolation, not the ideal analytic sine.
    Nyquist-underresolved requested drives are rejected, never silently aliased.
    Amplitudes/frequencies are researcher-selected controls, not regime cutoffs.
    """
    before,conditions,roots,provenance = _inputs(decoder,carrier,conditions,root_ids,provenance)
    cfg = decoder.macro_config
    ga,la = np.asarray(global_amplitude,dtype=float),np.asarray(local_amplitude,dtype=float)
    mode = tuple(mode)
    if (ga.shape!=(2,) or la.shape!=(2,) or not np.isfinite(ga).all() or not np.isfinite(la).all()
            or not math.isfinite(cycles) or cycles<=0 or cycles>=(cfg.control_count-1)/2
            or not math.isfinite(phase) or mode not in half_modes(cfg.max_mode)
            or max(abs(x) for x in mode)>2):
        raise ValueError("finite amplitudes, available low mode and strictly sub-Nyquist positive cycles required")
    u,local,chol = [x.clone() for x in split_macro_controls(before["m"],cfg)]
    tau = torch.linspace(0,1,cfg.control_count,device=u.device,dtype=torch.float64)
    curve = torch.sin(2*math.pi*float(cycles)*tau+float(phase))
    u += torch.as_tensor(ga,device=u.device,dtype=u.dtype)[None,:,None]*curve.to(u)[None,None]
    index = half_modes(cfg.max_mode).index(mode)*2
    local[...,index] += torch.as_tensor(la,device=local.device,dtype=local.dtype)[None,:,None]*curve.to(local)[None,None]
    after = {"m":pack_macro_controls(u,local,chol,cfg),"z":before["z"].clone()}
    return _finish_control(decoder,before,after,conditions,roots,provenance,
        {"kind":"velocity_drive","global_amplitude_physical_velocity":ga.tolist(),
         "local_cos_amplitude_physical_velocity":la.tolist(),"local_mode":list(mode),
         "cycles_per_time_window":float(cycles),"phase_radians":float(phase),
         "control_node_curve":curve.cpu().tolist(),"continuous_curve":"native value-to-slope C1 Hermite, not an analytic sine",
         "maximum_requested_cycles_for_unaliased_node_sampling":(cfg.control_count-1)/2,
         "z_rho_C_unchanged":True,"energy_preserved":False,
         "changes":"U/local velocity, j, bulk kinetic energy, raw higher moments, f and actual MMS driving",
         "physical_regime_is_not_assigned_from_nu_or_this_control_label":True},check_mms)


def _time_spectrum(values,times):
    """One-sided Parseval power on endpoint-excluded finite-window samples."""
    values = np.asarray(values,dtype=np.float64)
    times = np.asarray(times,dtype=np.float64)
    if (values.ndim!=2 or not values.shape[1] or len(values)!=len(times) or len(times)<3
            or times.ndim!=1 or not np.isfinite(times).all() or not np.isfinite(values).all()
            or not (np.diff(times)>0).all() or not np.allclose(np.diff(times),times[1]-times[0],rtol=1e-12,atol=0)):
        raise ValueError("finite [time,probe] values required for time spectrum")
    sampled = values[:-1]
    count = len(sampled)
    frequency = np.fft.rfftfreq(count,d=float(times[1]-times[0]))
    factors = np.full(len(frequency),2.)
    factors[0] = 1.
    if count%2==0: factors[-1] = 1.
    def power(array):
        transformed = np.fft.rfft(array,axis=0)/count
        return np.mean(np.abs(transformed)**2,axis=1)*factors
    raw = power(sampled)
    tau = (times[:-1]-times[0])/(times[-1]-times[0])
    detrended = sampled-(values[0]+tau[:,None]*(values[-1]-values[0]))
    window = np.hanning(count+1)[:-1]
    tapered = power(detrended*window[:,None])/np.mean(window**2)
    if not np.isfinite(raw).all() or not np.isfinite(tapered).all():
        raise ValueError("nonfinite temporal spectral energy; no clipping")
    return {"frequency_cycles_per_physical_time":frequency.tolist(),
        "rectangular_one_sided_mean_square_power":raw.tolist(),
        "linear_endpoint_detrended_Hann_power_window_corrected":tapered.tolist(),
        "raw_parseval_mean_square":float(np.mean(sampled**2)),
        "raw_power_sum":float(raw.sum()),
        "endpoint_jump_rms":float(np.sqrt(np.mean((values[-1]-values[0])**2))),
        "rectangular_top_quarter_Nyquist_power":float(raw[frequency>=.75*frequency[-1]].sum()),
        "window":"rectangular periodic extension plus separate endpoint-linear-detrended periodic Hann",
        "endpoint_handling":"last physical endpoint retained in values/jump but excluded from FFT to represent exact window length",
        "tapered_power_is_not_the_original_unwindowed_energy":True}


def _derivative_energy(values,times):
    values = np.asarray(values,dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite derivative energy")
    mean_square = np.mean(values**2,axis=1)
    integral = np.sum(.5*(mean_square[1:]+mean_square[:-1])*np.diff(times))
    if not math.isfinite(integral):
        raise ValueError("nonfinite integrated derivative energy; no clipping")
    return {"time_integrated_probe_mean_square":float(integral),
            "time_mean_probe_mean_square":float(integral/(times[-1]-times[0])),
            "spatial_or_velocity_integral":False}


def temporal_frequency_diagnostics(decoder,carrier,conditions,*,root_ids,provenance,
        spatial_probes,velocity_probes,time_counts=None,chunk_size=16):
    """Actual A source and macros on nested midpoint/quarter time grids.

    Default 33-control state -> 65 and 129 *inclusive* uniform queries. Spatial
    and velocity probes are fixed physical coordinates, not moving with u.
    Store raw values, exact macro/f dt, and explicitly finite-difference dt S.
    No fullspace norm, dynamics certification or population gate is inferred.
    """
    before,conditions,roots,provenance = _inputs(decoder,carrier,conditions,root_ids,provenance,allow_generated=True)
    cfg = decoder.macro_config
    expected = (2*(cfg.control_count-1)+1,4*(cfg.control_count-1)+1)
    counts = expected if time_counts is None else tuple(time_counts)
    if counts!=expected or type(chunk_size) is not int or chunk_size<1:
        raise ValueError("use the nested control/midpoint and control/quarter grids and positive chunks")
    xy,vv = np.asarray(spatial_probes,dtype=float),np.asarray(velocity_probes,dtype=float)
    if (xy.ndim!=2 or xy.shape[1]!=2 or not len(xy) or vv.shape!=xy.shape
            or not np.isfinite(xy).all() or not np.isfinite(vv).all()):
        raise ValueError("paired nonempty finite [probe,2] physical x/y and vx/vy required")
    hashes = carrier_hashes(decoder,before,conditions)
    report = {"schema":TIME_SCHEMA,"execution_passed":True,"source_provenance":provenance,
        "decoder_identity":_json_copy(decoder.identity()),"control_count":cfg.control_count,
        "time_counts":list(counts),"spatial_probes":xy.tolist(),"velocity_probes":vv.tolist(),
        "cases":[],"physical_regime_inferred_from_nu":False,
        "scientific_regime_qualified":False,"new_training_steps":0,"new_pde_solves":0,
        "independence_unit":"original physical root, not repeated/control rows",
        "limitations":["Fixed physical probes, not spatial/velocity integrated Sobolev norms.",
            "Nonperiodic finite-window leakage and unresolved frequencies above each Nyquist remain.",
            "65/129 agreement is a sampled diagnostic, not an alias-free or uniform convergence proof.",
            "Source is manufactured from A autodiff, not an independent PDE residual check.",
            "dt_source uses finite differences, not exact third-order autodiff.",
            "Point source does not measure net/integrated injection; no regime threshold is fitted."]}
    device = next(decoder.parameters()).device
    for case in range(len(conditions)):
        row = {"physical_root_id":roots[case],"input_carrier_sha256":hashes[case],
               "condition":conditions[case].cpu().tolist(),"execution_passed":False,"grids":[]}
        try:
            c = {k:value[case:case+1] for k,value in before.items()}
            for count in counts:
                times = np.linspace(*cfg.time_domain,count)
                points = np.column_stack((np.repeat(times,len(xy)),np.tile(xy,(count,1)),np.tile(vv,(count,1))))
                raw,derivatives = {},{}
                for start in range(0,len(points),chunk_size):
                    q = torch.as_tensor(points[start:start+chunk_size],dtype=torch.float64,device=device).clone().requires_grad_(True)
                    with torch.enable_grad():
                        macros = evaluate_moving_frame_macros(decoder,q[:,:3],c,conditions[case])
                        fields = {"rho":macros["rho"],"jx":macros["momentum"][:,0],
                                  "jy":macros["momentum"][:,1],"K":macros["kinetic_energy"]}
                        for name,value in fields.items():
                            raw.setdefault(name,[]).append(value.detach().cpu().numpy())
                            derivative = torch.autograd.grad(value.sum(),q,retain_graph=True)[0][:,0]
                            derivatives.setdefault(name,[]).append(derivative.detach().cpu().numpy())
                        source = manufacture_moving_frame_source(decoder,q,c,conditions[case])
                        for name in ("f","source"):
                            raw.setdefault(name,[]).append(source[name].detach().cpu().numpy())
                        derivatives.setdefault("f",[]).append(source["f_gradient"][:,0].detach().cpu().numpy())
                raw = {name:np.concatenate(parts).reshape(count,len(xy)) for name,parts in raw.items()}
                derivatives = {name:np.concatenate(parts).reshape(count,len(xy)) for name,parts in derivatives.items()}
                derivatives["source"] = np.gradient(raw["source"],times,axis=0,edge_order=2)
                stride = (count-1)//(cfg.control_count-1)
                index = np.arange(count)
                labels = ["node" if i%stride==0 else "midpoint" if (2*i)%stride==0 else "quarter" for i in index]
                grid = {"count":count,"times":times.tolist(),"time_kind":labels,
                    "values":{k:v.tolist() for k,v in raw.items()},
                    "dt_values":{k:v.tolist() for k,v in derivatives.items()},
                    "spectra":{k:_time_spectrum(v,times) for k,v in raw.items()},
                    "derivative_energy":{k:_derivative_energy(v,times) for k,v in derivatives.items()},
                    "dt_source_method":"second-order centered finite difference; second-order one-sided endpoints",
                    "other_dt_method":"actual A physical-coordinate autodiff; native C1 knot convention",
                    "Nyquist_cycles_per_physical_time":.5/(times[1]-times[0]),
                    "unresolved_aliasing_possible":True}
                row["grids"].append(grid)
            coarse,fine = row["grids"]
            refinements = {}
            for name in coarse["values"]:
                a,b = np.asarray(coarse["values"][name]),np.asarray(fine["values"][name])[::2]
                ea = coarse["derivative_energy"][name]["time_mean_probe_mean_square"]
                eb = fine["derivative_energy"][name]["time_mean_probe_mean_square"]
                refinements[name] = {"shared_node_value_maximum_absolute_difference":float(np.max(np.abs(a-b))),
                    "derivative_energy_coarse":ea,"derivative_energy_fine":eb,
                    "derivative_energy_absolute_difference":abs(ea-eb),
                    "derivative_energy_relative_to_fine":None if eb==0 else abs(ea-eb)/eb}
            row.update(execution_passed=True,refinement=refinements)
        except Exception as error:
            row["failure"] = {"type":type(error).__name__,"message":str(error)}
            report["execution_passed"] = False
        report["cases"].append(row)
    return _json_copy(report)
