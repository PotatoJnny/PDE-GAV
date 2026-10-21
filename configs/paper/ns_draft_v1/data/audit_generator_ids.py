#!/usr/bin/env python3
"""Read only case_ids from coefficient NPZ members, never coefficient arrays."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from freeze_data import HERE, PRIOR, REPO, SOURCE, ids_sha, read, sha


def main():
    release = read(SOURCE / 'split_manifest.json')
    reports = {}
    bank_ids = {}
    for filename in ('development_manifest.json', 'calibration_manifest.json',
                     'test_draft_manifest.json', 'test_final_manifest.json'):
        for name, rows in read(HERE / filename)['splits'].items():
            bank_ids[name] = {row['case_id'] for row in rows}
    for role in ('train', 'validation', 'hidden_reference'):
        path = PRIOR / 'coefficients' / role / 'coefficient_manifest.json'
        manifest = read(path)
        actual = []
        for shard in manifest['shards']:
            with np.load(path.parent / shard['path'], allow_pickle=False) as data:
                actual.extend(str(value) for value in data['case_ids'])
        signature = manifest['encoding_signature']
        expected = [row['case_id'] for row in release['splits'][signature['split']]]
        assert actual == expected
        assert ids_sha(actual) == signature['selected_case_ids_sha256']
        # Actual PCA/flow loading applies max_cases to the first ordered members.
        used = actual if role == 'train' else actual[:2048] if role == 'validation' else []
        reports[role] = {
            'manifest': str(path), 'manifest_sha256': sha(path),
            'coefficient_shards_read_metadata_only': len(manifest['shards']),
            'actual_case_id_count': len(actual), 'ordered_ids_sha256': ids_sha(actual),
            'actual_order_exactly_matches_original_split': True,
            'declared_pca_flow_fit_or_loss_cases': len(used),
            'fit_or_loss_case_overlap_with_new_banks': {
                name: len(set(used) & ids) for name, ids in bank_ids.items()},
        }
    result = {
        'schema': 'ns_paper_coefficient_id_audit_v1', 'corpora': reports,
        'coefficient_numeric_arrays_read': False,
        'large_shard_content_hashes_recomputed': False,
        'loader_semantics': 'load_structured_coefficient_corpus takes first max_cases in shard order; PCA audit and flow train_config declare 80000train and2048validation.',
        'validation_overlap_interpretation': 'Development/search-validation cases can overlap Generator hyperparameter-validation data. The Generator has not fitted on those cases by gradient training; this is shared prior model-selection information, not untouched validation. Final test remains disjoint from both train and validation.',
        'hidden_reference_interpretation': 'The entire5000test coefficient bank exists. It is not in the verified fit/loss configuration. The128 known post-selection diagnostic IDs were excluded by the frozenbank eligibility rule.',
        'checker_sha256': sha(Path(__file__)),
    }
    output = REPO / 'docs/paper/evidence/ns_draft_v1_data_coefficient_ids_audit.json'
    if output.exists():
        raise RuntimeError(f'refusing to overwrite {output}')
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'output': str(output), 'corpora': reports}, indent=2))


if __name__ == '__main__':
    main()
