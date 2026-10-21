from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from pde_agent_verifier_solver.app import ProductApplication, ProgramBundle, RunStatus
from pde_agent_verifier_solver.engine.agent_capability import (
    AgentCapabilityClientError,
    ControllerAgentCapabilityClient,
    ControllerAgentCapabilityFacade,
)
from pde_agent_verifier_solver.engine.agent_execution import (
    CommandAgentProgram,
    SubprocessAgentExecutionService,
)
from pde_agent_verifier_solver.engine.campaign import (
    AgentSpec,
    DevelopmentSamplingPolicy,
    EvaluatorPolicy,
    RoundPlan,
)
from pde_agent_verifier_solver.engine.evaluator import AgentEvaluator
from pde_agent_verifier_solver.engine.memory import CampaignMemorySnapshot
from pde_agent_verifier_solver.engine.runtime import AgentContext
from pde_agent_verifier_solver.panel.controller import PanelController
from pde_agent_verifier_solver.plugins.registry import get_plugin
from pde_agent_verifier_solver.plugins.toy import (
    ToyPoissonGenerator,
    ToyPoissonProblem,
    ToyResearchAgent,
    ToySolverEvaluation,
    ToySolverTask,
)
from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    CaseAccessKind,
    DatasetInput,
    DatasetKind,
    GeneratorCheckpoint,
    GeneratorState,
    HiddenSuiteSpec,
    PublicOracleMethod,
    PublicProblemBundle,
    PublicResource,
    SampleSource,
)


_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"


def _oracle_method(name: str) -> PublicOracleMethod:
    return PublicOracleMethod(
        name=name,
        description=f"Test method {name}",
        input_schema={"arguments": "Agent-selected"},
        output_schema={"value": "finite numeric data"},
        cost_unit="call",
    )


def _test_bundle() -> PublicProblemBundle:
    return PublicProblemBundle(
        bundle_id="canonical-public-capability-test-v1",
        equation_id="canonical-public-capability-equation-v1",
        description="Public equation resources and two separated oracles.",
        resources=(
            PublicResource(
                "reference/operator.py",
                b"RESOURCE_BYTES_SENTINEL = 7\n",
                media_type="text/x-python",
            ),
        ),
        oracle_methods=(_oracle_method("public_values"),),
        development_solution_oracle_methods=(
            _oracle_method("solution_values"),
        ),
    )


class _OracleSession:
    def __init__(self, access: CaseAccessKind) -> None:
        self.access = access
        self.closed = False

    def public_values(self, values, *, offset=0.0):
        assert self.access is CaseAccessKind.PUBLIC_PROBLEM
        return np.asarray(values, dtype=np.float64) + float(offset)

    def solution_values(self, values, *, scale=1.0):
        assert self.access is CaseAccessKind.DEVELOPMENT_SOLUTION
        return np.asarray(values, dtype=np.float64) * float(scale)

    def close(self) -> None:
        self.closed = True


class _OracleToyGenerator(ToyPoissonGenerator):
    def __init__(self) -> None:
        super().__init__()
        self.sessions: list[_OracleSession] = []

    def case_session_methods(self, access: CaseAccessKind) -> tuple[str, ...]:
        if access is CaseAccessKind.PUBLIC_PROBLEM:
            return ("public_values",)
        return ("solution_values",)

    def open_case_session(self, state, case, access):
        del state, case
        session = _OracleSession(access)
        self.sessions.append(session)
        return session


def _context(
    evaluator: AgentEvaluator,
    bundle: PublicProblemBundle,
) -> AgentContext:
    agent = AgentSpec("oracle-agent", "Oracle Agent")
    return AgentContext(
        round_plan=RoundPlan(
            round_index=1,
            agents=(agent,),
            agent_budget_seconds=5,
            development_sampling=DevelopmentSamplingPolicy(
                max_cases_per_agent_per_round=2,
                allowed_sources=(SampleSource.BASE,),
            ),
            round_hidden=HiddenSuiteSpec(base=1),
        ),
        agent=agent,
        prompt="Use the public equation.",
        evaluator=evaluator,
        incumbent=None,
        memory=CampaignMemorySnapshot(),
        public_problem_bundle=bundle,
    )


def _evaluator(
    generator: _OracleToyGenerator,
    *,
    allow_solution: bool,
) -> tuple[AgentEvaluator, str]:
    dataset = ToyPoissonProblem().prepare_dataset(
        DatasetInput(
            "public-capability-dataset",
            DatasetKind.NUMERICAL,
        )
    )
    record = dataset.development_cases[0]
    query = AnswerQuery.grid(fields=("u",), x=9)
    checkpoint = GeneratorCheckpoint(generator.info.program_id, object())
    state = GeneratorState(checkpoint=checkpoint, base=object())
    evaluator = AgentEvaluator(
        (record,),
        ToySolverEvaluation(query),
        EvaluatorPolicy(
            query,
            allow_development_solution_session=allow_solution,
        ),
        generator=generator,
        generator_state=state,
    )
    return evaluator, record.case.case_id


def test_rpc_uses_opaque_case_ids_allowlists_wire_codec_and_revocation(tmp_path):
    generator = _OracleToyGenerator()
    bundle = _test_bundle()
    evaluator, raw_id = _evaluator(generator, allow_solution=True)
    context = _context(evaluator, bundle)
    program = CommandAgentProgram((sys.executable, "-c", "pass"), lambda *_: None)
    workspace = tmp_path / "rpc"
    workspace.mkdir()
    lease = ControllerAgentCapabilityFacade().open(
        context,
        workspace,
        "public-problem-rpc",
        program,
    )
    client = ControllerAgentCapabilityClient(lease.manifest, workspace=workspace)
    opaque_id = evaluator.cases[0].case_id

    np.testing.assert_allclose(
        client.invoke_public_problem(
            opaque_id,
            "public_values",
            np.asarray((1.0, 2.0)),
            offset=0.5,
        ),
        (1.5, 2.5),
    )
    np.testing.assert_allclose(
        client.invoke_development_solution(
            opaque_id,
            "solution_values",
            np.asarray((2.0, 3.0)),
            scale=4.0,
        ),
        (8.0, 12.0),
    )
    with pytest.raises(AgentCapabilityClientError) as undeclared:
        client.invoke_public_problem(opaque_id, "solution_values", (1.0,))
    assert undeclared.value.code == "capability_unavailable"
    with pytest.raises(AgentCapabilityClientError) as raw_denied:
        client.invoke_public_problem(raw_id, "public_values", (1.0,))
    assert raw_denied.value.code == "operation_failed"
    assert raw_id not in str(raw_denied.value)
    assert lease.controller_service_seconds > 0.0

    lease.close()
    assert generator.sessions
    assert all(session.closed for session in generator.sessions)
    with pytest.raises(AgentCapabilityClientError) as expired:
        client.invoke_public_problem(opaque_id, "public_values", (1.0,))
    assert expired.value.code == "capability_unavailable"


def test_development_solution_rpc_requires_user_authorization(tmp_path):
    generator = _OracleToyGenerator()
    bundle = _test_bundle()
    evaluator, _raw_id = _evaluator(generator, allow_solution=False)
    context = _context(evaluator, bundle)
    workspace = tmp_path / "rpc-denied"
    workspace.mkdir()
    lease = ControllerAgentCapabilityFacade().open(
        context,
        workspace,
        "public-problem-rpc-denied",
        CommandAgentProgram((sys.executable, "-c", "pass"), lambda *_: None),
    )
    try:
        client = ControllerAgentCapabilityClient(lease.manifest, workspace=workspace)
        with pytest.raises(AgentCapabilityClientError) as denied:
            client.invoke_development_solution(
                evaluator.cases[0].case_id,
                "solution_values",
                (1.0,),
            )
        assert denied.value.code == "capability_unavailable"
    finally:
        lease.close()


def _decode_toy_nomination(payload, context, workspace):
    del workspace
    return ToyResearchAgent._artifact(
        context,
        int(payload["mode"]),
        suffix="public-problem-e2e",
    )


def test_vpfp_and_gkw_bundles_reach_subprocess_through_the_canonical_run(tmp_path):
    worker = tmp_path / "public_problem_worker.py"
    worker.write_text(
        "\n".join(
            (
                "import json, os, stat",
                "from pathlib import Path",
                "from pde_agent_verifier_solver.engine.agent_execution import (",
                "    load_agent_worker_request, submit_agent_nomination,",
                "    write_agent_worker_response,",
                ")",
                "request = load_agent_worker_request()",
                "public = request['public_problem']",
                (
                    "assert public['manifest']['bundle_id'] == "
                    "request['worker_settings']['bundle_id']"
                ),
                "assert not Path(public['resources_path']).is_absolute()",
                "assert not Path(public['manifest_path']).is_absolute()",
                (
                    "root = Path(os.environ['PDE_AGENT_WORKSPACE']) / "
                    "public['resources_path']"
                ),
                "assert stat.S_IMODE(root.stat().st_mode) == 0o555",
                (
                    "assert json.loads((root / 'manifest.json').read_text()) "
                    "== public['manifest']"
                ),
                "for item in public['manifest']['resources']:",
                "    assert 'data' not in item",
                "    path = root / item['path']",
                "    assert path.is_file()",
                "    assert stat.S_IMODE(path.stat().st_mode) == 0o444",
                "submit_agent_nomination({'mode': 3}, sequence=1)",
                (
                    "write_agent_worker_response("
                    "observations=('public problem received',))"
                ),
            )
        ),
        encoding="utf-8",
    )

    for plugin_id in ("vpfp", "gkw"):
        source_plugin = get_plugin(plugin_id)
        assert source_plugin.public_problem_bundle is not None
        runnable_plugin = replace(
            source_plugin,
            problem=ToyPoissonProblem(),
            generator=ToyPoissonGenerator(),
            tasks={"solver": ToySolverTask()},
            default_task="solver",
        )
        command_agent = CommandAgentProgram(
            (sys.executable, str(worker)),
            _decode_toy_nomination,
            environment={"PYTHONPATH": str(_SOURCE_ROOT)},
            worker_settings={
                "bundle_id": source_plugin.public_problem_bundle.bundle_id
            },
        )
        programs = ProgramBundle.from_plugin(
            runnable_plugin,
            {"agent": command_agent},
        )
        assert programs.public_problem_bundle is source_plugin.public_problem_bundle
        service = SubprocessAgentExecutionService(
            tmp_path / f"{plugin_id}-turns",
            hard_wall_seconds=15,
        )
        panel = PanelController(
            ProductApplication.local_development(agent_execution=service),
            programs,
        )
        launched = panel.create_and_launch(
            {
                "project_name": f"{plugin_id} public problem E2E",
                "round_count": 1,
                "agents": ["agent"],
                "starter_cases": {"base": 1},
                "max_sampled_cases_per_agent_per_round": 2,
                "agent_budget_seconds": 8,
                "generator_settings": {
                    "guidance_enabled": False,
                    "perturbations_enabled": False,
                },
                "round_hidden": {"base": 1},
                "terminal_hidden": {"base": 1},
            }
        )
        session = panel.app.wait(launched["run_id"], timeout=20)
        assert session.status is RunStatus.COMPLETED, session.error
        request = json.loads(Path(service.audits[-1].request_path).read_text())
        assert request["public_problem"]["manifest"]["bundle_id"] == (
            source_plugin.public_problem_bundle.bundle_id
        )
        assert "data" not in request["public_problem"]["manifest"]["resources"][0]


def test_program_bundle_from_plugin_validates_agents_and_none_is_compatible():
    plugin = get_plugin("toy")
    with pytest.raises(TypeError, match="nonempty Agent mapping"):
        ProgramBundle.from_plugin(plugin, {})
    programs = ProgramBundle.from_plugin(
        plugin,
        {"agent": ToyResearchAgent()},
    )
    assert programs.public_problem_bundle is None
