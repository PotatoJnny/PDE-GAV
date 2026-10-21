"""Measured Fixed must keep raw samples and use the numerical array contract."""
import json
from pathlib import Path

import numpy as np
import pytest

from pde_agent_verifier_solver.plugins.ns.discrete_measured import write_discrete_measured_case
from pde_agent_verifier_solver.plugins.ns.common_evaluation import load_numerical_split
from pde_agent_verifier_solver.sdk import AnswerQuery, CaseVisibility
from scripts.prepare_ns_measured_fixed_full import discrete_fixed_config


def measured_case(tmp_path):
    x=np.linspace(-.3,.7,5);y=np.linspace(-.2,.8,4);t=np.linspace(0,1,6)
    xx,yy=np.meshgrid(x,y,indexing='ij')
    order=np.random.default_rng(3).permutation(xx.size)
    px,py=xx.ravel()[order],yy.ravel()[order]
    velocity=np.broadcast_to(np.stack([1+2*t,3-t],axis=1)[:,:,None],(6,2,20)).copy()
    directory=tmp_path/'measured'
    diagnostic=write_discrete_measured_case(directory,case_id='measured',velocity=velocity,
        point_x=px,point_y=py,times=t,viscosity=.001,provenance={'test':True})
    manifest=tmp_path/'data.json'
    manifest.write_text(json.dumps({'schema':'ns_d0_v1','splits':{'development':[
        {'case_id':'measured','case_dir':str(directory)}]}}))
    record,=load_numerical_split(manifest,split='development',visibility=CaseVisibility.DEVELOPMENT)
    return record,velocity,diagnostic


def test_native_samples_and_boundary_survive_numerical_loader(tmp_path):
    record,velocity,diagnostic=measured_case(tmp_path)
    arrays=record.case.public_data['arrays']
    assert 'domain' not in record.case.public_data
    assert not {'u_true','velocity','pressure','coefficients'} & arrays.keys()
    assert arrays['initial_velocity'].shape==(2,5,4)
    assert arrays['boundary_velocity'].shape==(6,2,14)
    np.testing.assert_array_equal(arrays['initial_velocity'][:,2,2],velocity[0,:,0])
    np.testing.assert_array_equal(arrays['boundary_velocity'][:,:,0],velocity[:,:,0])
    with np.load(record.truth.case_dir/'hidden_truth_cache.npz') as f:
        np.testing.assert_array_equal(f['u_true'][:,:,2,2],velocity[:,:,0])
    assert diagnostic['continuous_projection'] is False
    query=AnswerQuery.at_points(fields=('velocity_x','velocity_y'),points=({'t':.35,'x':.15,'y':.1},))
    answer=record.truth.answer(query)
    np.testing.assert_allclose([answer.fields['velocity_x'][0],answer.fields['velocity_y'][0]], [1.7,2.65])


def test_discrete_mms_matches_known_acceleration(tmp_path):
    record,_,_=measured_case(tmp_path)
    a=record.case.public_data['arrays'];f=a['forcing']
    np.testing.assert_allclose(f[:,0,1:-1,1:-1],2,atol=1e-6)
    np.testing.assert_allclose(f[:,1,1:-1,1:-1],-1,atol=1e-6)
    assert not np.any(f[:,:,a['outer_boundary_mask']])


def test_duplicate_native_coordinates_rejected(tmp_path):
    x=np.tile(np.arange(3.),3);y=np.repeat(np.arange(3.),3);x[0]=x[1]
    with pytest.raises(ValueError,match='duplicate'):
        write_discrete_measured_case(tmp_path/'bad',case_id='bad',velocity=np.ones((3,2,9)),
            point_x=x,point_y=y,times=np.arange(3.),viscosity=.001,provenance={})
    assert not (tmp_path/'bad').exists()


def test_fixed_uses_numerical_loader_and_retains_all_non_data_budgets(tmp_path):
    root=Path(__file__).resolve().parents[1]
    original=json.loads((root/'configs/paper/ns_bgp_v3/fixed_s101.json').read_text())
    fixed=discrete_fixed_config(original,tmp_path/'development.json')
    assert 'kind' not in fixed['dataset']
    assert fixed['agent']==original['agent']
    for key in ('schema','method','seed','search','evaluation','initial_candidate'):
        assert fixed[key]==original[key]
    assert fixed['dataset']['selection_manifest']==original['dataset']['selection_manifest']
    from pde_agent_verifier_solver.plugins.ns.baselines.cli import _public_problem_bundle
    bundle=_public_problem_bundle(continuous_selection=True)
    assert bundle.oracle_methods==()
    assert bundle.development_solution_oracle_methods==()
    assert 'boundary_velocity[T,2,Nb]' in fixed['prompt']
    assert 'query_boundary_velocity' in fixed['prompt']
