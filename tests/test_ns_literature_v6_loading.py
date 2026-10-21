"""Load the declared MMS version; retain legacy defaults and reject mixed assets."""
from dataclasses import replace
from hashlib import sha256

import pytest

from tests.test_ns_geometry_v6 import V5, V6, artifact_pair, single_thread
from tests.test_ns_resume import _request
from pde_agent_verifier_solver.plugins.ns.baselines.environment import build_online_generator
from pde_agent_verifier_solver.sdk import CaseVisibility, GenerationPurpose


def config(directory, actual, expected):
    values = artifact_pair(directory, actual)
    return {**values, "generator_id": "literature-v6-cpu-test", "generator_sha256": "a" * 64,
            "perturbation_amplitude": 1e-5, "spectral_max_mode": 2,
            "artifact_sha256": {key: sha256(values[key].read_bytes()).hexdigest()
                                for key in ("flow_checkpoint", "pca_checkpoint")},
            "expected_continuous_mms_abi": expected}


@pytest.mark.parametrize("abi", [V5, V6])
def test_valid_version_loads_and_new_case_uses_declared_geometry(tmp_path, abi):
    generator, state = build_online_generator(config(tmp_path, abi, abi), tmp_path)
    request = replace(_request(count=1), visibility=CaseVisibility.DEVELOPMENT,
                      purpose=GenerationPurpose.DEVELOPMENT)
    record, = generator.sample(state, request)
    assert record.truth._answer.raw_answer.continuous_mms_abi == abi


@pytest.mark.parametrize("actual,expected", [(V5, V6), (V6, V5)])
def test_version_supported_but_different_weights_are_rejected(tmp_path, actual, expected):
    with pytest.raises(ValueError, match="Generator artifacts use a different continuous MMS version"):
        build_online_generator(config(tmp_path, actual, expected), tmp_path)
