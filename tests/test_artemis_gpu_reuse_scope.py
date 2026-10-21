from threading import RLock
from types import SimpleNamespace

import pytest

from pde_agent_verifier_solver.engine.evaluator import _CaseSessionBindings
from pde_agent_verifier_solver.engine.unified_runtime import CampaignController
from pde_agent_verifier_solver.plugins.vp_1d1v.evaluation_gpu_reuse import ArtemisHiddenEvaluationScope
from scripts.run_canonical_artemis_campaign import _execution_inputs


def test_artemis_preserves_paired_hidden_plan_seeds():
    from pde_agent_verifier_solver.plugins.vp_1d1v.task import ArtemisEvaluation

    assert ArtemisEvaluation.pair_hidden_plan_seeds is True


class FakePool:
    def __init__(self, source, **kwargs):
        self.kwargs = kwargs
        self.started = self.closed = False

    def start(self):
        self.started = True

    def close(self):
        self.closed = True

    def stats(self):
        return {"started": self.started, "closed": self.closed, "cleanup_complete": self.closed}


def make_scope(factory=FakePool):
    backend = SimpleNamespace(source=object(), _parallel_oracle_pool=None)
    binding = SimpleNamespace(plugin=SimpleNamespace(generator=SimpleNamespace(_backend=lambda: backend)))
    execution = SimpleNamespace(oracle_host_count=1)
    scope = ArtemisHiddenEvaluationScope(binding, execution, backend_settings={},
                                        devices=("2", "0", "1"), pool_factory=factory)
    return scope, backend, execution


@pytest.mark.parametrize("fail", [False, True])
def test_scope_reuses_only_idle_devices_and_restores_single_gpu(fail):
    scope, backend, execution = make_scope()
    try:
        with scope(cases=("case-a", "case-b"), unavailable_gpu_devices=("1",)) as pool:
            assert pool.kwargs["devices"] == ("2", "0")
            assert pool.kwargs["cases"] == ("case-a", "case-b")
            assert backend._parallel_oracle_pool is pool
            assert execution.oracle_host_count == 2
            if fail:
                raise LookupError("candidate failed")
    except LookupError:
        assert fail
    assert pool.closed
    assert backend._parallel_oracle_pool is None
    assert execution.oracle_host_count == 1
    assert scope.last_report["pool"]["cleanup_complete"]


def test_scope_cleans_partial_startup():
    class BrokenPool(FakePool):
        def start(self):
            super().start()
            raise RuntimeError("startup failed")
    scope, backend, execution = make_scope(BrokenPool)
    with pytest.raises(RuntimeError, match="startup failed"):
        with scope(cases=("a",)):
            pytest.fail("must not enter")
    assert backend._parallel_oracle_pool is None
    assert execution.oracle_host_count == 1
    assert scope.last_report["pool"]["cleanup_complete"]


def test_unavailable_all_devices_and_overlapping_scopes_fail():
    scope, backend, _ = make_scope()
    with pytest.raises(RuntimeError, match="no healthy GPU"):
        with scope(cases=("a",), unavailable_gpu_devices=("0", "1", "2")):
            pass
    with scope(cases=("a",)):
        with pytest.raises(RuntimeError, match="overlap"):
            with scope(cases=("a",)):
                pass


@pytest.mark.parametrize("replicated", [False, True])
def test_session_lock_is_explicit_and_defaults_to_global(replicated):
    global_lock, lane_lock = RLock(), RLock()
    generator = SimpleNamespace(
        open_case_session=lambda *_: None,
        case_session_methods=lambda *_: ("query_source",),
        public_session_operation_lock=lambda *_: lane_lock if replicated else None,
    )
    bindings = _CaseSessionBindings(generator, object(), operation_lock=global_lock)
    session = bindings.persistent_public_session(SimpleNamespace(case_id="private"), SimpleNamespace(case_id="public"))
    assert session._operation_lock is (lane_lock if replicated else global_lock)


def test_controller_scope_keeps_order_and_forwards_quarantine():
    from contextlib import contextmanager
    events = []
    @contextmanager
    def scope(**kwargs):
        events.append(("enter", kwargs))
        yield
        events.append(("exit",))
    controller = object.__new__(CampaignController)
    controller.candidate_execution = SimpleNamespace(hidden_evaluation_scope=scope)
    controller.agent_execution = SimpleNamespace(quarantined_gpu_devices=("1",))
    controller._official_reports_in_scope = lambda *a, **k: events.append(("evaluate", a)) or "result"
    a, b = object(), object()
    hidden = {"base": (SimpleNamespace(case=a),), "pert": (SimpleNamespace(case=b),)}
    assert controller._official_reports(("candidate",), hidden) == "result"
    assert [row[0] for row in events] == ["enter", "evaluate", "exit"]
    assert events[0][1] == {"cases": (a, b), "unavailable_gpu_devices": ("1",)}


def test_execution_override_preserves_guidance_and_assets():
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    original = json.loads((root / "configs/plasma_canonical_artemis.json").read_text())["factory_settings"]
    inputs = {"factory_settings": original, "backend_config_sha256": "original"}
    effective = _execution_inputs(inputs, root / "configs/plasma_artemis_d5_guided.json")
    assert effective["factory_settings"]["reward_guidance_settings"] == original["reward_guidance_settings"]
    assert effective["factory_settings"]["oracle_settings"]["exact_query_kernel_mode"] == "vp_specialized"
    assert effective["original_backend_config_sha256"] == "original"


def test_execution_override_rejects_science_changes(tmp_path):
    import json
    path = tmp_path / "override.json"
    path.write_text(json.dumps({"physics": "artemis", "factory_settings": {"science_config_path": "different"}}))
    with pytest.raises(ValueError, match="scientific assets"):
        _execution_inputs({"factory_settings": {"science_config_path": "original"}}, path)
