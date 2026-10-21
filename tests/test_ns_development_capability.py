from __future__ import annotations

from types import SimpleNamespace
import sys

import pytest

from pde_agent_verifier_solver.engine.agent_capability import (
    AgentCapabilityClientError, ControllerAgentCapabilityClient,
)
from pde_agent_verifier_solver.engine.agent_execution import CommandAgentProgram
from pde_agent_verifier_solver.engine.campaign import AgentSpec
from pde_agent_verifier_solver.engine.memory import CampaignMemorySnapshot
from pde_agent_verifier_solver.engine.runtime import AgentContext
from pde_agent_verifier_solver.engine.unified_runtime import _InteractiveSampler
from pde_agent_verifier_solver.plugins.ns.development_capability import (
    NSDevelopmentCapabilityFacade, NSDevelopmentSamplerView,
)
from pde_agent_verifier_solver.sdk import (
    Case, CaseSource, CaseVisibility, DevelopmentCoordinateMode,
    PairedSampleBatch, SampleBatch, SampleCasesRequest, SampleSource,
)


class _Sampler:
    def __init__(self):
        self.calls = []

    def _case(self, source, *, parent=None):
        return Case(
            case_id=f"case-{len(self.calls)}-{source.value}",
            source=source,
            visibility=CaseVisibility.DEVELOPMENT,
            public_data={},
            round_index=1,
            parent_case_id=parent,
        )

    def sample_cases(self, request):
        self.calls.append(("sample", request.source))
        return SampleBatch((self._case(CaseSource.GENERATED),), 20)

    def sample_paired_cases(self, request):
        self.calls.append(("paired", request.source))
        root = self._case(CaseSource.GENERATED)
        child = self._case(CaseSource.PERTURBED, parent=root.case_id)
        return PairedSampleBatch((root,), (child,), 18, ())

    def perturb_cases(self, request):
        self.calls.append(("perturb", request.parent_case_ids))
        return SampleBatch((self._case(CaseSource.PERTURBED, parent=request.parent_case_ids[0]),), 19)


@pytest.mark.parametrize("guidance,perturbations", [(False, False), (True, False), (False, True), (True, True)])
def test_ns_development_rpc_enforces_tools_before_touching_underlying_sampler(tmp_path, guidance, perturbations):
    sampler = _Sampler()
    evaluator = SimpleNamespace(
        cases=(), fixed_query=None,
        development_coordinate_mode=DevelopmentCoordinateMode.PRIVATE_RANDOM_POINTS,
        development_point_count=8, allow_grid_truth=False, allow_point_truth=False,
    )
    context = AgentContext(
        round_plan=None,
        agent=AgentSpec("test-agent", "Test"),
        prompt="test", evaluator=evaluator, incumbent=None,
        memory=CampaignMemorySnapshot(), sampler=sampler,
    )
    facade = NSDevelopmentCapabilityFacade(
        guidance_enabled=guidance, perturbations_enabled=perturbations,
    )
    lease = facade.open(context, tmp_path, "ns-permission-test", CommandAgentProgram(
        (sys.executable, "-c", "pass"), lambda *args: None,
    ))
    try:
        client = ControllerAgentCapabilityClient(lease.manifest, workspace=tmp_path)
        assert len(client.sample_base(1).cases) == 1
        operations = (
            (lambda: client.sample_guided(1), guidance),
            (lambda: client.sample_paired_cases(SampleSource.BASE, 1), perturbations),
            (lambda: client.sample_paired_cases(SampleSource.GUIDED, 1), guidance and perturbations),
            (lambda: client.perturb_cases(("existing-clean-root",)), perturbations),
        )
        for operation, permitted in operations:
            calls_before = list(sampler.calls)
            if permitted:
                operation()
                assert len(sampler.calls) == len(calls_before) + 1
            else:
                with pytest.raises(AgentCapabilityClientError) as denied:
                    operation()
                assert denied.value.code == "access_denied"
                assert sampler.calls == calls_before
    finally:
        lease.close()
    # The hidden-capable Controller context itself is never mutated.
    assert context.sampler is sampler


def test_ns_permission_view_keeps_controller_allowed_sources_authoritative():
    sampler = _Sampler()
    inner = _InteractiveSampler(
        sampler, SimpleNamespace(), (SampleSource.BASE,),
        allow_guidance=True, allow_perturbations=True,
    )
    view = NSDevelopmentSamplerView(inner, guidance_enabled=True, perturbations_enabled=True)
    with pytest.raises(PermissionError, match="not enabled in this round"):
        view.sample_cases(SampleCasesRequest(SampleSource.GUIDED, 1))
    assert sampler.calls == []
