"""Import saved continuous Cases without resuming CUDA or sampling a new pool."""
from hashlib import sha256
import json
import pickle
from pathlib import Path
from dataclasses import replace

import numpy as np
import pytest

from tests.test_ns_geometry_v6 import field, V6, single_thread
from pde_agent_verifier_solver.engine.common_evaluation import json_value
from pde_agent_verifier_solver.plugins.ns.continuous_backend import StructuredNSManufacturedCaseSource
from pde_agent_verifier_solver.plugins.ns.resume import _field_payload, _put_tensors
from pde_agent_verifier_solver.plugins.ns.baselines.preparation import load_checkpoint_starter
from pde_agent_verifier_solver.sdk.macro_loop import CohortKind, SampleRequest


@pytest.fixture
def received(tmp_path):
    root=tmp_path/'received';objects=root/'common/objects';objects.mkdir(parents=True)
    raw=field(V6)
    source=StructuredNSManufacturedCaseSource(generator_id='original',artifact_sha256='a'*64,
        clean_sampler=lambda seed:raw,perturbation_sampler=None)
    records=source.sample(SampleRequest(1,CohortKind.DEVELOPMENT,seed=101,count=64))
    descriptor=_put_tensors(objects,_field_payload(raw))
    weights={}
    paths={}
    for key in ('flow_checkpoint','pca_checkpoint'):
        path=root/(key+'.bin');path.write_bytes(key.encode());paths[key]=str(path)
        weights[key]=sha256(path.read_bytes()).hexdigest()
    identity={'test':'frozen-original-pool'}
    checkpoint={'schema':'campaign_checkpoint_v1','identity':identity,'generator_runtime':{
        'schema':'ns_private_round_runtime_v1',
        'base':{'generator_id':'original','artifact_sha256':'a'*64},
        'flow_identity':{k+'_sha256':v for k,v in weights.items()},
        'residents':[{'case':json_value(r.case),'cohort':'development','field':descriptor} for r in records]}}
    path=root/'common/checkpoint.json';path.write_text(json.dumps(checkpoint))
    checksum=sha256(path.read_bytes()).hexdigest()
    index={'case_ids':[r.case.case_id for r in records],'seed':101,'checkpoint_sha256':checksum}
    (root/'index.json').write_text(json.dumps(index))
    config={'schema':'ns_v6_asset_recovery_v1','asset_root':str(root),
        'common_development':{'case_count':64,'continuous_mms_abi':V6,'manifest':'index.json',
            'checkpoint_directory':'common','checkpoint_sha256':checksum,'checkpoint_identity':identity},
        'generator':{'generator_id':'original','generator_sha256':'a'*64,
            'artifacts':paths,'artifact_sha256':weights,'runtime':{'device':'cuda','integration_steps':256}}}
    p=root/'restore.json';p.write_text(json.dumps(config))
    return p,records,raw


def test_cpu_import_preserves_original_fields_order_and_does_not_restore_runtime(received,monkeypatch):
    import pde_agent_verifier_solver.plugins.ns.canonical_backend as canonical
    def forbidden(*args,**kwargs):raise AssertionError('data export must not construct a live Generator')
    monkeypatch.setattr(canonical,'build_canonical_structured_ns_generator',forbidden)
    p,original,field=received
    before={f:sha256(f.read_bytes()).hexdigest() for f in p.parent.rglob('*') if f.is_file()}
    result=load_checkpoint_starter(p)
    assert [r.case.case_id for r in result.records]==[r.case.case_id for r in original]
    declared = json.loads(p.read_text())
    assert result.provenance['source_checkpoint_sha256'] == declared['common_development']['checkpoint_sha256']
    assert result.provenance['generator_artifact_sha256'] == declared['generator']['artifact_sha256']
    for record in result.records:
        actual=record.truth._answer.raw_answer
        assert actual.device.type=='cpu' and actual.continuous_mms_abi==V6
        np.testing.assert_array_equal(actual.coefficients.psi,field.coefficients.psi)
        np.testing.assert_array_equal(actual.coefficients.pressure,field.coefficients.pressure)
    assert before=={f:sha256(f.read_bytes()).hexdigest() for f in p.parent.rglob('*') if f.is_file()}


@pytest.mark.parametrize('corruption',['field','index','outside'])
def test_changed_or_missing_assets_fail_before_export(received,corruption):
    p,_,_=received
    config=json.loads(p.read_text())
    if corruption=='index':
        path=p.parent/'index.json';data=json.loads(path.read_text());data['case_ids'].pop();path.write_text(json.dumps(data))
    elif corruption=='field':
        next((p.parent/'common/objects').glob('*.bin')).write_bytes(b'corrupt')
    else:
        config['common_development']['manifest']='../elsewhere.json';p.write_text(json.dumps(config))
    with pytest.raises((ValueError, pickle.UnpicklingError, EOFError)):load_checkpoint_starter(p)


def test_saved_fields_do_not_require_loading_or_rescanning_generator_weights(received):
    path, original, _ = received
    config = json.loads(path.read_text())
    for location in config['generator']['artifacts'].values():
        Path(location).unlink()
    result = load_checkpoint_starter(path)
    assert [record.case.case_id for record in result.records] == [record.case.case_id for record in original]
    assert result.provenance['generator_artifact_sha256'] == config['generator']['artifact_sha256']
