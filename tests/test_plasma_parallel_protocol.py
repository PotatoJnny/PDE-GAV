"""Small configuration checks for the shared queue; no model or GPU execution."""
import pytest
from scripts.plasma_gpu_campaign import parallel_execution_settings


def configuration(equation="weak"):
    return {"equation": equation, "execution": {
        "candidate_case_workers": 8, "development_workers_per_gpu": 4,
        "solver_batch_size": 1, "gpu_slots_per_device": 2,
        "oracle_max_workers": 8, "oracle_concurrency_per_device": 1}}


@pytest.mark.parametrize("equation", ["strong", "weak", "artemis"])
def test_shared_gpu_slots_do_not_claim_batch_support(equation):
    result = parallel_execution_settings(configuration(equation))
    assert result == dict(solver_batch_size=1, gpu_slots_per_device=2,
                         oracle_max_workers=8, oracle_concurrency_per_device=1)


@pytest.mark.parametrize("key,value,reason", [
    ("solver_batch_size", 4, "prepare_solutions"),
    ("oracle_concurrency_per_device", 2, "concurrent-stream"),
    ("oracle_max_workers", 9, "worker bound"),
    ("gpu_slots_per_device", 9, "worker bound"),
    ("oracle_max_workers", 2, "development lanes"),
    ("gpu_slots_per_device", True, "positive integers"),
    ("oracle_max_workers", 0, "positive integers"),
])
def test_rejects_unsupported_or_misleading_parallel_settings(key, value, reason):
    config = configuration()
    config["execution"][key] = value
    with pytest.raises(ValueError, match=reason):
        parallel_execution_settings(config)
