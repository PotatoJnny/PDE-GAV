#!/usr/bin/env python3
"""Check package contents, all eight templates and receiver dependencies on CPU."""
from __future__ import annotations
import argparse
from hashlib import sha256
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'legacy/ns/src')]


def check(root=ROOT):
    root=Path(root)
    manifest=json.loads((root/'PACKAGE_MANIFEST.json').read_text())
    for name,expected in manifest['files'].items():
        path=root/name
        if path.is_symlink() or not path.is_file():
            raise ValueError('package file missing or changed: ' + name)
    from pde_agent_verifier_solver.plugins.ns.baselines.search import NSBaselineConfig
    loaded=Path(sys.modules[NSBaselineConfig.__module__].__file__).resolve()
    if not loaded.is_relative_to((root/'src').resolve()):
        raise ValueError('imports resolved outside the extracted source package')
    for method in ('codepde','autonumerics','autopde','pde_sharp'):
        pair=[]
        for arm in ('fixed_sampled','continuous_online_base'):
            p=root/'configs/paper/ns_literature_v2'/f'{method}_{arm}.template.json'
            config=json.loads(p.read_text())
            NSBaselineConfig(config['method'],root/'unused-check-only',seed=config['seed'],
                agent_budget_seconds=config['agent']['hard_wall_seconds'],**config['search'])
            pair.append(config)
        for key in ('search','agent','evaluation','initial_candidate','seed'):
            if pair[0][key]!=pair[1][key]: raise ValueError('pair mismatch: '+method+'/'+key)
    versions={}
    for module,dist in [('numpy','numpy'),('scipy','scipy'),('torch','torch'),('yaml','PyYAML')]:
        importlib.import_module(module)
        versions[dist]=importlib.metadata.version(dist)
    # No device initialization, model request, data sampling or scheduling.
    fields=importlib.import_module('agent_verifier.neural_generator.structured_ns_answer_first')
    if 'continuous_structured_ns_mms_v6' not in fields.SUPPORTED_STRUCTURED_NS_MMS_ABI_VERSIONS:
        raise ValueError('this experiment requires the reviewed v6 field/restore implementation')
    return {'status':'ok','source_files_verified':len(manifest['files']),
            'paired_configs_verified':8,'dependencies':versions,
            'real_assets':'checked separately by prepare_ns_literature_baselines.py --binding'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(check(),indent=2))
