"""Tiny mechanics for raw-time U checks; numerical tests run only on compute."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
loader = importlib.util.spec_from_file_location("raw_velocity_under_test",ROOT/"scripts/run_vpfp_velocity_raw_reference.py")
raw = importlib.util.module_from_spec(loader)
loader.loader.exec_module(raw)


def generation():
    return dict(velocity_grid_size=21,nx=16,ny=16,nt=33,t1=1.,velocity_max=8.,
                internal_dt_max=.001,dtype="float32")


def test_three_declared_levels_keep_physical_domain_and_original_coarse():
    levels = raw.levels_for(generation())
    assert [x["nt"] for x in levels] == [33,129,257]
    assert levels[0] == raw.ref.default_levels(generation())[0]
    for level in levels[1:]:
        for key in ("t0","t1","x_domain","v_bounds"):
            assert level[key] == levels[0][key]
    assert levels[1]["nx"] == 16 and levels[2]["nx"] == 24
    assert raw.STRATA == (0,7) and raw.DEGREES == (4,8,16)
    with pytest.raises(ValueError,match="33-frame"):
        raw.levels_for({**generation(),"nt":17})


def test_global_moments_use_trapezoid_and_spatial_mean():
    level = dict(nt=3,nx=2,ny=3,nvx=3,nvy=3,v_bounds=[-1.,1.,-1.,1.])
    moments = raw.global_moments(torch.ones((3,2,3,3,3),dtype=torch.float64),level)
    assert np.allclose(moments,np.tile([4.,0.,0.,2.],(3,1)))
    with pytest.raises(ValueError,match="trajectory"):
        raw.global_moments(torch.ones((2,2,3,3,3)),level)


def test_source_acceleration_includes_mass_injection(monkeypatch):
    times = np.linspace(0.,1.,5)
    M = 1+times
    U = np.tile([.3,-.2],(5,1))
    moments = np.column_stack((M,M[:,None]*U,np.ones(5)))
    source = np.column_stack((np.ones(5),U,np.zeros(5)))
    values = iter([moments,source])
    monkeypatch.setattr(raw,"global_moments",lambda *args:next(values))
    monkeypatch.setattr(raw,"tail_report",lambda *args:{"synthetic_fixture":True})
    report = raw.measure({"f":torch.ones((1,5,1,1,1,1)),"diagnostics":{}},
        {"source_frames":torch.zeros((1,5,1,1,1,1))},dict(t0=0.,t1=1.,nt=5,x_domain=[2.,3.]),0.)
    assert np.allclose(report["source_acceleration_mass_corrected"],0.)
    assert np.allclose(report["U_drive_proxy"],0.,atol=1e-15)
    assert np.allclose(report["global_mass_balance_residual"],0.)
    assert report["source_from_original_equation_not_manufactured_from_solution"]


def test_exact_driven_U_fit_against_dense_raw_fixture():
    nu = .04
    t = np.linspace(0.,1.,129)
    U = np.exp(-nu*t)[:,None]*np.array([[.4,-.2]])
    dense = dict(times=t.tolist(),U=U.tolist(),dU_fd=(-nu*U).tolist(),
                 source_acceleration_mass_corrected=np.zeros_like(U).tolist())
    result = raw.compare_fit(U[::4],dense,nu,4)
    assert result["coordinate_count"] == 12
    assert result["U_offnode_rms_gap"] < 1e-12
    assert result["a_vs_independently_integrated_source_rms_gap"] < 1e-11
    assert result["scientific_admission"] is False


def test_real_solve_rejected_without_allocation(tmp_path,monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID",raising=False)
    with pytest.raises(ValueError,match="allocation"):
        raw.run(SimpleNamespace(output=tmp_path,wall_seconds=2,device="cpu"))
    assert not (tmp_path/"result.json").exists()
