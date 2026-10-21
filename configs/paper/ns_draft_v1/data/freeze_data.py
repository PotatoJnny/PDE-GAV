#!/usr/bin/env python3
"""Freeze the approved NS banks using metadata only; no simulation or training.

Run from the repository with .venv/bin/python. Existing differing outputs are
rejected. The case directories remain in the shared release; no arrays copied.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SOURCE = Path('${PDE_GAV_SHARED_DATA_ROOT}/runs/server_triplane_generator_20260628_full100k/d0_large')
PRIOR = Path('${PDE_GAV_RUN_ROOT}/pde_runs/ns_strong_prompt_5round_20260828_v1/artifacts')
SEED = 20260907


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    content = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != content:
        raise RuntimeError(f'refusing to replace a different frozen file: {path}')
    path.write_text(content)


def ids_sha(ids):
    return hashlib.sha256(b''.join(key.encode() + b'\0' for key in ids)).hexdigest()


def main():
    source_manifest = SOURCE / 'split_manifest.json'
    splits = read(source_manifest)['splits']
    assembled = read(SOURCE / 'assembled_d0_manifest.json')
    family = {r['case_id']: r['family'] for r in assembled['cases']}
    families = sorted(set(family.values()))
    assert len(families) == 10
    source_ids = {k: {r['case_id'] for r in rows} for k, rows in splits.items()}
    for a in source_ids:
        for b in source_ids:
            if a != b:
                assert source_ids[a].isdisjoint(source_ids[b])

    audit_path = REPO / 'assets/generators/ns/structured_flow_v1/answer_first_report.json'
    prior_audit = read(audit_path)
    exposed = {r['case_id'] for key in ('reference_full_projected_cases', 'reference_pca_reconstructed_cases')
               for r in prior_audit[key]}
    assert len(exposed) == 128 and exposed <= source_ids['d0_distribution_test']

    rng = random.Random(SEED)
    validation_ids = sorted(source_ids['validation'])
    rng.shuffle(validation_ids)
    development = validation_ids[:512]
    selection = validation_ids[512:1024]
    used = set(development + selection)
    calibration = []
    final_test = []
    draft_test = []
    for name in families:
        eligible = sorted(k for k in source_ids['validation'] - used if family[k] == name)
        rng.shuffle(eligible)
        calibration.extend(eligible[:5])
        test_eligible = sorted(k for k in source_ids['d0_distribution_test'] - exposed if family[k] == name)
        rng.shuffle(test_eligible)
        assert len(test_eligible) >= 100
        final_test.extend(test_eligible[:100])
        draft_test.extend(test_eligible[:50])
    banks = {'development': development, 'validation': selection, 'calibration': calibration,
             'test_draft': draft_test, 'test_final': final_test}
    exclusive = ('development', 'validation', 'calibration', 'test_final')
    for a in exclusive:
        for b in exclusive:
            if a != b:
                assert set(banks[a]).isdisjoint(banks[b])
    assert set(draft_test) <= set(final_test)
    selected_ids = sorted(set().union(*(set(ids) for ids in banks.values())))

    def inspect(key):
        directory = (SOURCE / 'cases' / key).resolve(strict=True)
        metadata = {}
        for name in ('case_spec.json', 'public_question.json', 'public_arrays.npz', 'hidden_truth_cache.npz'):
            path = directory / name
            assert path.is_file(), path
            metadata[name] = {'bytes': path.stat().st_size}
            if path.suffix == '.json':
                metadata[name]['sha256'] = sha(path)
        spec = read(directory / 'case_spec.json')
        assert spec['case_id'] == key and spec['family'] == family[key]
        return key, str(directory), metadata

    with ThreadPoolExecutor(max_workers=8) as pool:
        inspected = list(pool.map(inspect, selected_ids))
    directories = {key: directory for key, directory, _ in inspected}

    common = {
        'schema': 'ns_d0_v1', 'dataset_id': 'ns-d0-paper-draft-v1',
        'selection_seed': SEED, 'source_manifest': str(source_manifest),
        'source_manifest_sha256': sha(source_manifest),
        'reference_kind': 'numerical_trajectory_derived_discrete_mms',
        'selection_algorithm': 'Python random.Random(20260907); sorted IDs; development/validation first, then family-sorted calibration and nested test',
        'test_eligibility': 'D0 d0_distribution_test excluding the 128 case IDs in the packaged post-selection answer_first_report',
    }
    files = {}
    for filename, names in {
        'development_manifest.json': ('development', 'validation'),
        'calibration_manifest.json': ('calibration',),
        'test_draft_manifest.json': ('test_draft',),
        'test_final_manifest.json': ('test_final',),
    }.items():
        payload = {**common, 'splits': {name: [
            {'case_id': key, 'case_dir': directories[key], 'family': family[key],
             'source_split': 'd0_distribution_test' if name.startswith('test_') else 'validation'}
            for key in banks[name]] for name in names}}
        write(HERE / filename, payload)
        files[filename] = {'sha256': sha(HERE / filename), 'splits': list(names)}

    provenance = {}
    for role, expected in (('train', 'train'), ('validation', 'validation'), ('hidden_reference', 'd0_distribution_test')):
        path = PRIOR / 'coefficients' / role / 'coefficient_manifest.json'
        manifest = read(path)
        signature = manifest['encoding_signature']
        expected_ids = [r['case_id'] for r in splits[expected]]
        assert signature['split'] == expected
        assert signature['split_manifest_sha256'] == sha(source_manifest)
        assert signature['selected_case_ids_sha256'] == ids_sha(expected_ids)
        provenance[role] = {'manifest': str(path), 'manifest_sha256': sha(path),
                            'encoding_signature': signature, 'signature_matches_release': True,
                            'encoded_cases': manifest['encoded_cases']}
    selection_evidence = []
    for step in (5000, 10000, 15000, 20000, 25000, 30000):
        path = PRIOR / 'generator_validation' / f'step_{step:07d}' / 'materialized/answer_first_report.json'
        report = read(path)
        ids = [r['case_id'] for r in report['reference_full_projected_cases']]
        assert set(ids) <= source_ids['validation']
        selection_evidence.append({'step': step, 'report': str(path), 'sha256': sha(path),
                                   'reference_manifest': report['reference_manifest'],
                                   'reference_ids_all_original_validation': True, 'reference_count': len(ids)})
    train_config = PRIOR / 'base_generator/train_config.json'
    pca_audit = REPO / 'assets/generators/ns/structured_flow_v1/pca_audit.json'
    flow_selection = REPO / 'assets/generators/ns/structured_flow_v1/flow_selection.json'
    index = {
        **common, 'files': files,
        'banks': {name: {'count': len(ids), 'ordered_ids_sha256': ids_sha(ids),
                         'family_counts': dict(sorted(Counter(family[k] for k in ids).items()))}
                  for name, ids in banks.items()},
        'excluded_prior_audit_test_ids': sorted(exposed),
        'validation_banks_pairwise_disjoint': True, 'draft_test_nested_in_final': True,
        'all_selected_cases_outside_generator_training_split': True,
        'large_case_arrays_hashed': False,
        'fingerprint_scope': 'Frozen row IDs and paths; selected public_question/case_spec SHA256; NPZ existence and byte sizes. Case NPZ byte content is not fully pinned by this metadata-only freeze.',
    }
    write(HERE / 'bank_index.json', index)
    evidence = {
        'schema': 'ns_paper_data_freeze_audit_v1', 'selection_seed': SEED,
        'bank_index': str(HERE / 'bank_index.json'), 'bank_index_sha256': sha(HERE / 'bank_index.json'),
        'source_metadata': {name: {'path': str(SOURCE / name), 'sha256': sha(SOURCE / name)}
                            for name in ('split_manifest.json', 'assembled_d0_manifest.json', 'dataset_release_summary.json', 'geometry_contract.json')},
        'selected_case_file_checks': {key: {'case_dir': directory, 'files': metadata} for key, directory, metadata in inspected},
        'generator_provenance': provenance,
        'pca_audit': {'path': str(pca_audit), 'sha256': sha(pca_audit), 'train_cases': read(pca_audit)['train_cases'],
                      'validation_cases': read(pca_audit)['validation_cases']},
        'flow_train_config': {'path': str(train_config), 'sha256': sha(train_config), 'content': read(train_config)},
        'flow_selection': {'path': str(flow_selection), 'sha256': sha(flow_selection), 'selected_step': read(flow_selection)['selected_step'],
                           'candidate_reference_reports': selection_evidence},
        'post_selection_test_audit': {'path': str(audit_path), 'sha256': sha(audit_path),
                                      'exposed_case_count': len(exposed), 'all_exposed_cases_excluded_from_new_test': True},
        'confirmed': {
            'all_selected_directories_and_required_files_exist': True,
            'selected_case_spec_ids_and_families_match': True,
            'all_banks_disjoint_from_train_80000': True,
            'pca_and_flow_training_reports_reference_train_80000': True,
            'six_flow_selection_physical_reports_reference_validation': True,
            'entire_5000_test_was_previously_encoded': True,
            'prior_128_test_reference_audit_cases_excluded': True,
        },
        'unknown_or_pending': {
            'complete_historical_test_access_audit': 'not performed; do not describe these IDs as never previously processed',
            'independent_forward_solver_and_refinement_qualification': 'pending; the D0 release is a numerical-trajectory-derived discrete MMS source',
            'all_100000_case_array_content_hashes': 'not computed; metadata/source manifests only',
            'generator_training_reexecuted': 'not performed; provenance verified through existing signed-by-hash manifests/reports',
        },
        'interpretation': 'Encoding a held-out coefficient bank is not itself fitting the Generator. Existing train/config and checkpoint-selection evidence use train/validation. Prior post-selection test audit is disclosed and its 128 IDs are excluded. This does not prove absence of all historical test-informed decisions.',
    }
    path = REPO / 'docs/paper/evidence/ns_draft_v1_data_freeze_audit.json'
    write(path, evidence)
    print(json.dumps({'counts': {name: len(ids) for name, ids in banks.items()}, 'excluded_test': len(exposed),
                      'index': str(HERE / 'bank_index.json'), 'audit': str(path)}, indent=2))


if __name__ == '__main__':
    main()
