"""Terminal protocol revision preserves existing scientific objects and source files."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from pde_agent_verifier_solver.plugins.ns.case_bank import load_ns_continuous_case_bank
from tests.test_ns_mixed_evaluation_preparation import cpu_field_factory, prepared_suite

ROOT = Path(__file__).resolve().parents[1]


def script():
    spec = importlib.util.spec_from_file_location('base_only_terminal', ROOT / 'scripts/prepare_ns_base_only_terminal.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def settings():
    return {'generator_id': 'cpu-test-ns', 'generator_sha256': 'a' * 64,
            'runtime': {'perturbation_family': 'spectral_complement',
                        'perturbation_amplitude': .001, 'spectral_max_mode': 3}}


def test_revision_preserves_frozen_rows_adds_only_unused_base_parents_and_replays(prepared_suite, tmp_path):
    old = prepared_suite['path'].parent / 'terminal.json'
    original = json.loads(old.read_text())
    old_bytes = {p: p.read_bytes() for p in old.parent.rglob('*') if p.is_file()}
    config = settings()
    row = next(row for row in original['cases'] if row['stratum'] == 'base_perturbed')
    config['runtime']['perturbation_amplitude'] = row['generation_provenance']['perturbation_audit']['amplitude_magnitude']
    first = script().prepare(old, tmp_path / 'v4', generator_config=config)
    second = script().prepare(old, tmp_path / 'replay', generator_config=config)
    assert first['strata_counts'] == {'base': 2, 'guided': 2, 'base_perturbed': 2}
    assert first['retained_cases_unchanged'] == 5 and first['added_base_perturbed'] == 1
    assert first['removed_guided_perturbed'] == 1
    a, b = [load_ns_continuous_case_bank(r['manifest'], expected_purpose='terminal') for r in (first, second)]
    # The relative bank paths and all scientific data are independent of output root.
    assert a.manifest['cases'] == b.manifest['cases']
    before = {row['case']['case_id']: row for row in original['cases']}
    base_ids = {row.case.case_id for row in a.records if a.strata[row.case.case_id] == 'base'}
    parent_ids = [row.case.parent_case_id for row in a.records if a.strata[row.case.case_id] == 'base_perturbed']
    assert set(parent_ids) == base_ids and len(parent_ids) == len(set(parent_ids))
    for row in a.manifest['cases']:
        if row['case']['case_id'] in before:
            assert row == before[row['case']['case_id']]
    added = next(row for row in a.records if row.case.case_id not in before)
    query = a.plans[added.case.case_id].query
    points = np.array([[p['t'], p['x'], p['y']] for p in query.points])
    values = a._answers[added.case.case_id].evaluate_velocity(points)
    frozen = added.truth.answer(query)
    for column, key in enumerate(('velocity_x', 'velocity_y')):
        np.testing.assert_allclose(values[:, column], frozen.fields[key], rtol=0, atol=0)
    assert all(p.read_bytes() == data for p, data in old_bytes.items())
    with pytest.raises(FileExistsError):
        script().prepare(old, tmp_path / 'v4', generator_config=config)


def test_revision_rejects_selection_bank_and_changed_perturbation(prepared_suite, tmp_path):
    root = prepared_suite['path'].parent
    with pytest.raises(ValueError, match='purpose'):
        script().prepare(root / 'fixed_selection.json', tmp_path / 'bad-purpose', generator_config=settings())
    config = settings()
    config['runtime']['perturbation_amplitude'] = 999.
    with pytest.raises(ValueError, match='family/amplitude'):
        script().prepare(root / 'terminal.json', tmp_path / 'bad-amplitude', generator_config=config)
    assert not (tmp_path / 'bad-purpose').exists() and not (tmp_path / 'bad-amplitude').exists()
