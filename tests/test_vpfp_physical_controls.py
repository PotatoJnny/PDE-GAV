"""Tiny actual-decoder controls. Run numerical tests only in an allocation."""
from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from tests.test_vpfp_moving_frame import tiny, constant_macro
from pde_agent_verifier_solver.plugins.vpfp import moving_frame as mf
from pde_agent_verifier_solver.plugins.vpfp import physical_controls as pc


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def fixture(max_mode=3,nonuniform=True,local=True):
    base,value,original,z,theta = tiny(degree=2,dtype="float64")
    cfg = replace(original,max_mode=max_mode)
    decoder = mf.MovingFrameDecoder(base,cfg,value_config=value)
    m = constant_macro(cfg)
    u,flow,chol = mf.split_macro_controls(m,cfg)
    if local:
        index = mf.half_modes(cfg.max_mode).index((1,0))*2
        flow[0,0,:,index] = torch.tensor([.2,.25,.18])
        flow[0,1,:,index+1] = torch.tensor([.1,.14,.11])
    if nonuniform:
        z[:,:value.rho_dimension].reshape(1,value.control_count,value.rho_modes)[0,:,0] = .13
    # Non-Gaussian higher-H carrier, never removed by a control.
    z[:,value.rho_dimension:].reshape(1,value.hermite_channels,value.control_count,value.kinetic_modes)[0,-1,:,0] = .07
    m = mf.pack_macro_controls(u,flow,chol,cfg)
    kwargs = {"root_ids":["physical-root-0"],"provenance":{"source_manifest_sha256":"a"*64,
              "split":"synthetic","data_origin":"synthetic_fixture"}}
    return decoder,{"m":m,"z":z},theta[None],kwargs


def node_macros(decoder,carrier,conditions,size=17):
    cfg = decoder.macro_config
    xy = mf.spatial_grid(cfg,size,device="cpu")
    reports = []
    for t in torch.linspace(*cfg.time_domain,cfg.control_count,dtype=torch.float64):
        q = torch.cat((t.expand(len(xy),1),xy),1)
        with torch.no_grad():
            values = pc.evaluate_moving_frame_macros(decoder,q,carrier,conditions[0])
        reports.append({"J":values["momentum"].mean(0),"K":values["kinetic_energy"].mean(),
                        "rho":values["rho"],"C":values["central_covariance"]})
    return reports


def test_index_control_changes_population_not_root_bytes_or_independence():
    decoder,carrier,condition,kw = fixture()
    carrier = {k:v.expand(2,-1).clone() for k,v in carrier.items()}
    carrier["m"][1,0] += .2
    condition = condition.expand(2,-1).clone()
    kw["root_ids"] = ["root-A","root-B"]
    original = {k:v.clone() for k,v in carrier.items()}
    result = pc.index_carrier_control(decoder,carrier,condition,**kw,indices=[1,1,0],
        selection_rule="predeclared mixture-frequency/repetition control",source_sampling_weights=[.25,.75])
    receipt = result["receipt"]
    assert receipt["output_row_count"]==3 and receipt["output_distinct_physical_root_count"]==2
    assert receipt["output_root_multiplicities"]=={"root-B":2,"root-A":1}
    assert receipt["lineage"][0]["source_carrier_sha256"]==receipt["lineage"][1]["output_carrier_sha256"]
    for j,i in enumerate((1,1,0)):
        for key in carrier:
            assert torch.equal(result["carrier"][key][j],original[key][i])
    collapsed = pc.index_carrier_control(decoder,carrier,condition,**kw,indices=[0]*4,
        selection_rule="one physical-root mode collapse")
    assert collapsed["receipt"]["output_distinct_physical_root_count"]==1
    assert collapsed["receipt"]["output_distinct_carrier_count"]==1
    result["carrier"]["z"][0,0] = 19.
    assert torch.equal(carrier["z"],original["z"])
    json.dumps(receipt,allow_nan=False)


@pytest.mark.parametrize("indices,weights", [([True],None),([2],None),([],None),([0],[0.,1.]),([0],[1.,-1.])])
def test_index_control_rejects_undefined_selection(indices,weights):
    decoder,carrier,condition,kw = fixture()
    carrier = {k:v.expand(2,-1).clone() for k,v in carrier.items()}
    condition = condition.expand(2,-1).clone(); kw["root_ids"] = ["A","B"]
    with pytest.raises(ValueError):
        pc.index_carrier_control(decoder,carrier,condition,**kw,indices=indices,
            source_sampling_weights=weights,selection_rule="fixed negative fixture")


def test_relative_phase_has_explicit_weaker_semantics_and_node_energy_matching():
    decoder,carrier,condition,kw = fixture()
    result = pc.relative_phase_scramble(decoder,carrier,condition,**kw,seed=12,
        preserve_node_flow_energy=True,grid_count=17,check_mms=False)
    assert not result["receipt"]["rule"]["guaranteed_detectable"]
    assert torch.equal(carrier["z"],result["carrier"]["z"])
    old_u,_,old_chol = mf.split_macro_controls(carrier["m"],decoder.macro_config)
    new_u,_,new_chol = mf.split_macro_controls(result["carrier"]["m"],decoder.macro_config)
    assert torch.equal(old_u,new_u) and torch.equal(old_chol,new_chol)
    a,b = node_macros(decoder,carrier,condition),node_macros(decoder,result["carrier"],condition)
    for old,new in zip(a,b):
        torch.testing.assert_close(old["K"],new["K"],rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(old["J"],new["J"],rtol=1e-12,atol=1e-12)
        assert torch.equal(old["rho"],new["rho"]) and torch.equal(old["C"],new["C"])
    again = pc.relative_phase_scramble(decoder,carrier,condition,**kw,seed=12,
        preserve_node_flow_energy=True,grid_count=17,check_mms=False)
    assert torch.equal(again["carrier"]["m"],result["carrier"]["m"])


def test_phase_only_rotation_preserves_each_mode_power_at_nonnodes():
    decoder,carrier,condition,kw = fixture()
    result = pc.relative_phase_scramble(decoder,carrier,condition,**kw,seed=4,
        preserve_node_flow_energy=False,check_mms=False)
    cfg = decoder.macro_config
    old = mf.split_macro_controls(carrier["m"],cfg)[1]
    new = mf.split_macro_controls(result["carrier"]["m"],cfg)[1]
    t = torch.tensor([2.,2.27,3.11,4.68,5.],dtype=torch.float64)
    a,b = [mf.interpolate_controls(v,t,cfg).reshape(len(t),2,-1,2) for v in (old,new)]
    torch.testing.assert_close(a.square().sum(-1),b.square().sum(-1),rtol=2e-13,atol=2e-14)
    assert not torch.equal(old,new)


@pytest.mark.parametrize("with_local",[False,True])
def test_coherent_to_incoherent_preserves_node_K_but_removes_global_J(with_local):
    decoder,carrier,condition,kw = fixture(local=with_local)
    if with_local:
        u,flow,chol = mf.split_macro_controls(carrier["m"],decoder.macro_config)
        high = mf.half_modes(decoder.macro_config.max_mode).index((0,3))*2
        # Every fixed candidate overlaps the original local flow: genuinely
        # exercise nonzero Gram-Schmidt projections, not only an orthogonal toy.
        flow[0,0,:,high] = .12; flow[0,0,:,high+1] = .18
        flow[0,1,:,high] = -.13; flow[0,1,:,high+1] = .09
        carrier["m"] = mf.pack_macro_controls(u,flow,chol,decoder.macro_config)
    result = pc.coherent_to_incoherent_flow(decoder,carrier,condition,**kw,grid_count=17,check_mms=False)
    assert torch.equal(result["carrier"]["z"],carrier["z"])
    u,flow,chol = mf.split_macro_controls(result["carrier"]["m"],decoder.macro_config)
    assert torch.count_nonzero(u)==0 and torch.count_nonzero(flow)>0
    assert torch.equal(chol,mf.split_macro_controls(carrier["m"],decoder.macro_config)[2])
    before,after = node_macros(decoder,carrier,condition),node_macros(decoder,result["carrier"],condition)
    for a,b in zip(before,after):
        assert torch.linalg.vector_norm(a["J"])>.1
        torch.testing.assert_close(b["J"],torch.zeros_like(b["J"]),rtol=0,atol=2e-13)
        torch.testing.assert_close(a["K"],b["K"],rtol=2e-12,atol=2e-12)
        assert torch.equal(a["rho"],b["rho"]) and torch.equal(a["C"],b["C"])
    for row in result["receipt"]["rule"]["node_energy_checks"]:
        assert row["maximum_absolute_energy_defect"]<2e-12
    if with_local:
        assert any(abs(row["gram_schmidt_projection_coefficient"])>1e-4
            for row in result["receipt"]["rule"]["template_and_gain"][0])
    assert not result["receipt"]["scientific_regime_qualified"]
    json.dumps(result["receipt"],allow_nan=False)


def test_undefined_coherent_transfer_keeps_source_hash_and_no_fake_output():
    decoder,carrier,condition,kw = fixture(max_mode=1)
    with pytest.raises(pc.UndefinedPhysicalControl,match="no requested high modes") as failure:
        pc.coherent_to_incoherent_flow(decoder,carrier,condition,**kw,check_mms=False)
    receipt = failure.value.receipt
    assert not receipt["execution_passed"] and not receipt["control_defined"]
    assert receipt["lineage"][0]["physical_root_id"]==kw["root_ids"][0]
    assert receipt["lineage"][0]["source_carrier_sha256"]
    assert receipt["lineage"][0]["output_carrier_sha256"] is None
    json.dumps(receipt,allow_nan=False)


def test_real_drive_changes_actual_A_source_and_retains_complete_z_C():
    decoder,carrier,condition,kw = fixture(max_mode=1)
    result = pc.drive_velocity(decoder,carrier,condition,**kw,global_amplitude=(1.,-.2),
        local_amplitude=(.15,0.),cycles=.5,mode=(1,0))
    receipt = result["receipt"]
    assert receipt["execution_passed"] and receipt["physics_probe_check_status"]=="executed"
    check = receipt["physics_probe_checks"][0]
    assert check["actual_A_MMS_executed"] and not check["independent_residual_verification"]
    assert not np.allclose(check["source"]["source"],check["control"]["source"])
    assert torch.equal(carrier["z"],result["carrier"]["z"])
    old_c = mf.split_macro_controls(carrier["m"],decoder.macro_config)[2]
    new_c = mf.split_macro_controls(result["carrier"]["m"],decoder.macro_config)[2]
    assert torch.equal(old_c,new_c)
    assert receipt["lineage"][0]["source_carrier_sha256"]!=receipt["lineage"][0]["output_carrier_sha256"]
    assert not receipt["rule"]["energy_preserved"]


def test_drive_rejects_native_time_alias_and_probe_failure_keeps_carrier(monkeypatch):
    decoder,carrier,condition,kw = fixture(max_mode=1)
    with pytest.raises(ValueError,match="sub-Nyquist"):
        pc.drive_velocity(decoder,carrier,condition,**kw,cycles=1.,check_mms=False)
    def fail(*args,**kwargs): raise FloatingPointError("synthetic failure, never a valid source")
    monkeypatch.setattr(pc,"manufacture_moving_frame_source",fail)
    result = pc.drive_velocity(decoder,carrier,condition,**kw,global_amplitude=(1.,0.),cycles=.5)
    assert not result["receipt"]["execution_passed"]
    assert result["receipt"]["physics_probe_check_status"]=="failed"
    assert result["receipt"]["lineage"][0]["output_carrier_sha256"]
    assert result["receipt"]["physics_probe_checks"][0]["failure"]["type"]=="FloatingPointError"


def test_reference_extract_retains_full_high_H_state_and_rejects_slope_edit():
    decoder,carrier,condition,kw = fixture(max_mode=1,nonuniform=False)
    full = pc.lift_value_controls(carrier["z"],decoder.value_config)
    result = pc.reference_carrier_from_full(decoder,full,condition,**kw,fit_grid=5,chunk_size=17)
    assert torch.equal(result["carrier"]["z"],carrier["z"])
    assert torch.equal(pc.lift_value_controls(result["carrier"]["z"],decoder.value_config),full)
    assert not result["receipt"]["rule"]["A_equals_B_claimed"]
    changed = full.clone()
    changed[0,decoder.value_config.rho_dimension] += .2
    with pytest.raises(ValueError):
        pc.reference_carrier_from_full(decoder,changed,condition,**kw,fit_grid=5)


def test_spectrum_exact_one_sided_power_physical_frequency_and_window_caveats():
    times = np.linspace(2.,5.,65)
    values = (2*np.sin(2*np.pi*3*(times-2.)/3.))[:,None]
    spectrum = pc._time_spectrum(values,times)
    power = np.asarray(spectrum["rectangular_one_sided_mean_square_power"])
    frequencies = np.asarray(spectrum["frequency_cycles_per_physical_time"])
    assert frequencies[power.argmax()]==pytest.approx(1.)
    assert spectrum["raw_power_sum"]==pytest.approx(2.,abs=1e-12)
    assert spectrum["raw_parseval_mean_square"]==pytest.approx(2.,abs=1e-12)
    assert spectrum["tapered_power_is_not_the_original_unwindowed_energy"]
    derivative = (4*np.pi*np.cos(2*np.pi*(times-2.)))[:,None]
    energy = pc._derivative_energy(derivative,times)
    assert energy["time_mean_probe_mean_square"]==pytest.approx(8*np.pi**2)
    assert energy["time_integrated_probe_mean_square"]==pytest.approx(24*np.pi**2)


@pytest.mark.parametrize("bad",["nonuniform","nan"])
def test_spectrum_rejects_undefined_samples(bad):
    t = np.linspace(0,1,5); x = np.ones((5,1))
    if bad=="nonuniform": t[2]+=.01
    else: x[2]=np.nan
    with pytest.raises(ValueError): pc._time_spectrum(x,t)


def test_temporal_diagnostic_actual_physical_dt_nesting_and_chunk_invariance():
    decoder,carrier,condition,kw = fixture(max_mode=1,nonuniform=False,local=False)
    u,flow,chol = mf.split_macro_controls(carrier["m"],decoder.macro_config)
    u[0,0] = torch.tensor([.2,.35,.5]); u[0,1] = 0.
    carrier["m"] = mf.pack_macro_controls(u,flow,chol,decoder.macro_config)
    args = dict(spatial_probes=[[.2,.3]],velocity_probes=[[.4,-.2]])
    whole = pc.temporal_frequency_diagnostics(decoder,carrier,condition,**kw,**args,chunk_size=32)
    chunked = pc.temporal_frequency_diagnostics(decoder,carrier,condition,**kw,**args,chunk_size=1)
    assert whole["execution_passed"] and chunked["execution_passed"]
    assert whole["time_counts"]==[5,9]  # The tiny fixture has 3 native controls; production 33 -> 65/129.
    bg = decoder.base_decoder.config.equation.background_density
    for a,b in zip(whole["cases"][0]["grids"],chunked["cases"][0]["grids"]):
        assert a["time_kind"].count("node")==3
        assert a["time_kind"].count("midpoint")==2
        np.testing.assert_allclose(a["dt_values"]["jx"],.1*bg,rtol=2e-6,atol=2e-8)
        np.testing.assert_allclose(a["dt_values"]["K"],.01*bg*np.asarray(a["times"])[:,None],rtol=2e-6,atol=2e-8)
        for name in a["values"]:
            np.testing.assert_allclose(a["values"][name],b["values"][name],rtol=3e-12,atol=3e-13)
        assert a["unresolved_aliasing_possible"]
        assert "finite difference" in a["dt_source_method"]
    assert whole["cases"][0]["grids"][1]["time_kind"].count("quarter")==4
    assert not whole["physical_regime_inferred_from_nu"] and not whole["scientific_regime_qualified"]
    json.dumps(whole,allow_nan=False)


def test_temporal_failures_stay_as_root_rows_and_wrong_grid_rejected(monkeypatch):
    decoder,carrier,condition,kw = fixture(max_mode=1)
    kw["provenance"]["data_origin"] = "generated_answer"
    kw["provenance"]["split"] = "generated"
    def fail(*args,**kwargs): raise FloatingPointError("synthetic failed MMS")
    monkeypatch.setattr(pc,"manufacture_moving_frame_source",fail)
    args = dict(spatial_probes=[[.1,.2]],velocity_probes=[[.3,.4]])
    report = pc.temporal_frequency_diagnostics(decoder,carrier,condition,**kw,**args)
    assert not report["execution_passed"] and len(report["cases"])==1
    assert report["cases"][0]["physical_root_id"]==kw["root_ids"][0]
    assert report["cases"][0]["failure"]["type"]=="FloatingPointError"
    assert report["cases"][0]["input_carrier_sha256"]
    assert report["source_provenance"]["data_origin"]=="generated_answer"
    # Measurements may inspect generated answers; reference calibration
    # interventions may not silently relabel them as physical roots.
    with pytest.raises(ValueError,match="origin"):
        pc.drive_velocity(decoder,carrier,condition,**kw,cycles=.5,check_mms=False)
    with pytest.raises(ValueError,match="nested"):
        pc.temporal_frequency_diagnostics(decoder,carrier,condition,**kw,**args,time_counts=(65,129))


def test_carrier_hash_binds_condition_precision_and_m():
    decoder,carrier,condition,kw = fixture()
    old = pc.carrier_hashes(decoder,carrier,condition)
    changed = {k:v.clone() for k,v in carrier.items()}; changed["m"][0,0]+=.01
    assert old!=pc.carrier_hashes(decoder,changed,condition)
    changed_condition = condition.clone(); changed_condition[0,0]+=.001
    assert old!=pc.carrier_hashes(decoder,carrier,changed_condition)
    assert old!=pc.carrier_hashes(decoder,{k:v.float() for k,v in carrier.items()},condition)
