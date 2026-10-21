from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import textwrap
import tomllib
from fnmatch import fnmatchcase
from types import MappingProxyType
from typing import Mapping
from unittest.mock import patch

import numpy as np
import pytest

from pde_agent_verifier_solver.engine import agent_tool
from pde_agent_verifier_solver.engine.agent_capability import (
    ControllerAgentCapabilityClient,
)
from pde_agent_verifier_solver.engine.agent_execution import (
    AGENT_CAMPAIGN_MEMORY_ENV,
    AGENT_CANDIDATE_HISTORY_ENV,
    AGENT_EXECUTION_ID_ENV,
    AGENT_NOMINATIONS_ENV,
    AGENT_PROTOCOL_MAX_BYTES_ENV,
    AGENT_REQUEST_ENV,
    AGENT_RESPONSE_ENV,
    AGENT_ROUND_LOG_ENV,
    AGENT_ROUND_LOG_MAX_BYTES_ENV,
    AGENT_WORKSPACE_ENV,
    AgentExecutionError,
    CommandAgentProgram,
    SubprocessAgentExecutionService,
    UnavailableAgentCapabilityFacade,
    submit_agent_nomination,
)
from pde_agent_verifier_solver.engine.agent_wire import AgentWireCodec
from pde_agent_verifier_solver.engine.campaign import (
    AgentSpec,
    DevelopmentSamplingPolicy,
    RoundPlan,
)
from pde_agent_verifier_solver.engine.evaluator import (
    EvaluationCaseResult,
    EvaluationResponse,
)
from pde_agent_verifier_solver.engine.memory import CampaignMemorySnapshot
from pde_agent_verifier_solver.engine.python_candidate import PythonCandidateCodec
from pde_agent_verifier_solver.engine.runtime import AgentContext
from pde_agent_verifier_solver.plugins.toy import ToyResearchAgent
from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    AnswerView,
    Case,
    CaseSource,
    CaseVisibility,
    HiddenSuiteSpec,
    QueryKind,
    SampleSource,
    TaskKind,
    freeze_candidate_bundle,
    inspect_candidate_bundle,
    write_candidate_bundle_manifest,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
WORKER_SCRIPT = REPOSITORY_ROOT / "scripts" / "run_canonical_codex_agent.py"
CONFIG = REPOSITORY_ROOT / "scripts" / "canonical_codex.config.toml"
SPEC = importlib.util.spec_from_file_location(
    "_canonical_codex_worker_under_test",
    WORKER_SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
WORKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKER)


def test_remove_runtime_preserves_external_symlink_targets(tmp_path: Path) -> None:
    executable = tmp_path / "shared-codex"
    executable.write_bytes(b"#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)
    external_file = tmp_path / "shared-readonly-file"
    external_file.write_bytes(b"shared data must not change\n")
    external_file.chmod(0o400)
    external_directory = tmp_path / "shared-directory"
    external_directory.mkdir()
    nested_target = external_directory / "must-survive"
    nested_target.write_bytes(b"external directory contents\n")
    nested_target.chmod(0o400)
    external_directory.chmod(0o500)
    targets = (executable, external_file, external_directory, nested_target)
    modes_before = {target: stat.S_IMODE(target.stat().st_mode) for target in targets}
    contents_before = {
        target: target.read_bytes() for target in targets if target.is_file()
    }

    runtime = tmp_path / "runtime"
    cache = runtime / "codex-home" / "cache"
    cache.mkdir(parents=True)
    (cache / "codex").symlink_to(executable)
    (cache / "file").symlink_to(external_file)
    (cache / "directory").symlink_to(external_directory, target_is_directory=True)
    missing_target = tmp_path / "must-stay-missing"
    (cache / "dangling").symlink_to(missing_target)
    readonly_file = cache / "local-readonly-file"
    readonly_file.write_bytes(b"remove only this runtime copy\n")
    readonly_file.chmod(0o400)
    for directory in (cache, cache.parent, runtime):
        directory.chmod(0o500)

    WORKER._remove_runtime(runtime)

    assert not os.path.lexists(runtime)
    assert not os.path.lexists(missing_target)
    assert {target: stat.S_IMODE(target.stat().st_mode) for target in targets} == modes_before
    assert {target: target.read_bytes() for target in contents_before} == contents_before
    assert tuple(external_directory.iterdir()) == (nested_target,)


def test_remove_runtime_does_not_chmod_hardlinked_external_file(tmp_path: Path) -> None:
    target = tmp_path / "shared-executable"
    target.write_bytes(b"unchanged executable bytes\n")
    target.chmod(0o700)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    os.link(target, runtime / "hardlink")
    runtime.chmod(0o500)

    WORKER._remove_runtime(runtime)

    assert not os.path.lexists(runtime)
    assert stat.S_IMODE(target.stat().st_mode) == 0o700
    assert target.read_bytes() == b"unchanged executable bytes\n"


@pytest.mark.parametrize("target_kind", ("file", "directory", "missing"))
def test_remove_runtime_unlinks_root_symlink_only(tmp_path: Path, target_kind: str) -> None:
    target = tmp_path / "external-target"
    if target_kind == "directory":
        target.mkdir()
        (target / "keep").write_bytes(b"external contents\n")
        target.chmod(0o500)
    elif target_kind == "file":
        target.write_bytes(b"external contents\n")
        target.chmod(0o400)
    mode_before = stat.S_IMODE(target.stat().st_mode) if target.exists() else None
    runtime = tmp_path / "runtime-link"
    runtime.symlink_to(target, target_is_directory=target_kind == "directory")

    WORKER._remove_runtime(runtime)

    assert not os.path.lexists(runtime)
    if target_kind == "missing":
        assert not os.path.lexists(target)
    else:
        assert stat.S_IMODE(target.stat().st_mode) == mode_before
        contents = target / "keep" if target_kind == "directory" else target
        assert contents.read_bytes() == b"external contents\n"


def test_remove_runtime_missing_directory_is_noop(tmp_path: Path) -> None:
    runtime = tmp_path / "missing-runtime"
    WORKER._remove_runtime(runtime)
    assert not os.path.lexists(runtime)


class CasesOnlyEvaluator:
    def __init__(self, cases):
        self.cases = tuple(cases)


def agent_tool_request(
    *,
    inline_array_bytes: int = 64,
    max_sidecar_bytes: int = 1024 * 1024,
    max_rpc_bytes: int = 1024 * 1024,
) -> Mapping[str, object]:
    return MappingProxyType(
        {
            "worker_settings": MappingProxyType(
                {
                    "candidate_execution": MappingProxyType(
                        {"device": "cpu", "cuda_available": False}
                    )
                }
            ),
            "capabilities": MappingProxyType(
                {
                    "wire": MappingProxyType(
                        {
                            "codec": "safe-json-npy-v1",
                            "sidecar_directory": "capability-arrays",
                            "inline_array_bytes": inline_array_bytes,
                            "max_sidecar_bytes": max_sidecar_bytes,
                            "max_rpc_bytes": max_rpc_bytes,
                        }
                    )
                }
            ),
        }
    )


def agent_context() -> AgentContext:
    agent = AgentSpec(agent_id="canonical-codex", name="Canonical Codex")
    plan = RoundPlan(
        round_index=2,
        agents=(agent,),
        agent_budget_seconds=3,
        development_sampling=DevelopmentSamplingPolicy(
            max_cases_per_agent_per_round=7,
            allowed_sources=(SampleSource.BASE, SampleSource.GUIDED),
        ),
        round_hidden=HiddenSuiteSpec(base=1),
    )
    case = Case(
        case_id="development-case-1",
        source=CaseSource.GENERATED,
        visibility=CaseVisibility.DEVELOPMENT,
        public_data={"frequency": 3},
        round_index=2,
        tags=("base",),
    )
    return AgentContext(
        round_plan=plan,
        agent=agent,
        prompt="Improve the public solver without assuming a solver family.",
        evaluator=CasesOnlyEvaluator((case,)),
        incumbent=None,
        memory=CampaignMemorySnapshot(),
        seed=29,
    )


def decode_nomination(payload, context, workspace):
    del workspace
    return ToyResearchAgent._artifact(
        context,
        int(payload["mode"]),
        suffix=str(payload["suffix"]),
    )


def write_fake_codex(
    root: Path,
    *,
    capture_path: Path,
    exit_code: int,
    nomination_count: int,
    native_events: tuple[dict, ...] = (),
    stderr_text: str = "",
    before_first_event: bool = False,
    invalid_auth_audit: str | None = None,
) -> Path:
    executable = root / f"fake-codex-{exit_code}-{nomination_count}.py"
    executable.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env python3
            import json
            import os
            from pathlib import Path
            import subprocess
            import sys
            import time

            from pde_agent_verifier_solver.engine.agent_execution import (
                submit_agent_nomination,
                write_agent_round_log,
            )

            with Path({str(capture_path.with_suffix('.invocations'))!r}).open("a") as count:
                count.write("invoked\\n")
            if {before_first_event!r}:
                sys.stderr.write({stderr_text!r})
                raise SystemExit({exit_code})
            print(json.dumps({{"type": "thread.started"}}), flush=True)
            prompt = sys.stdin.read()
            codex_home = Path(os.environ["CODEX_HOME"])
            for _ in range(1000):
                if (
                    not (codex_home / "auth.json").exists()
                    and (codex_home / "auth-lifecycle.json").is_file()
                ):
                    break
                time.sleep(0.001)
            auth_missing = not (codex_home / "auth.json").exists()
            audit_present = (codex_home / "auth-lifecycle.json").is_file()
            candidate = Path(os.environ["PDE_AGENT_WORKSPACE"]) / "bytecode-probe"
            candidate.mkdir()
            candidate_source = candidate / "submission.py"
            candidate_source.write_text("VALUE = 1\\n", encoding="utf-8")
            subprocess.run(
                [sys.executable, "-m", "py_compile", str(candidate_source)],
                check=True,
            )
            request = json.loads(
                Path(os.environ["PDE_AGENT_REQUEST"]).read_text(encoding="utf-8")
            )
            nomination_count = {nomination_count}
            round_log = (
                "# Mock Codex log\\n\\n"
                f"Committed {{nomination_count}} nomination(s)."
            )
            write_agent_round_log(round_log)
            for sequence in range(1, nomination_count + 1):
                submit_agent_nomination(
                    {{
                        "mode": sequence,
                        "suffix": f"mock-{{sequence}}",
                    }},
                    sequence=sequence,
                )
            output_index = sys.argv.index("--output-last-message") + 1
            Path(sys.argv[output_index]).write_text(
                "Mock Codex finished.\\n",
                encoding="utf-8",
            )
            capture = {{
                "argv": sys.argv[1:],
                "prompt": prompt,
                "config_text": (codex_home / "config.toml").read_text(),
                "execution_id": os.environ["PDE_AGENT_EXECUTION_ID"],
                "request_execution_id": request["execution_id"],
                "round_log_max_bytes": os.environ[
                    "PDE_AGENT_ROUND_LOG_MAX_BYTES"
                ],
                "protocol_max_bytes": os.environ[
                    "PDE_AGENT_PROTOCOL_MAX_BYTES"
                ],
                "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                "pycache_prefix": os.environ.get("PYTHONPYCACHEPREFIX"),
                "candidate_pycache_exists": (candidate / "__pycache__").exists(),
                "redirected_pyc_exists": any(
                    Path(os.environ["PYTHONPYCACHEPREFIX"]).rglob(
                        "submission.*.pyc"
                    )
                ),
                "response_exposed": "PDE_AGENT_RESPONSE" in os.environ,
                "unrelated_exposed": "UNRELATED_MARKER" in os.environ,
                "auth_missing": auth_missing,
                "auth_audit_present": audit_present,
            }}
            Path({str(capture_path)!r}).write_text(
                json.dumps(capture, indent=2, sort_keys=True) + "\\n",
                encoding="utf-8",
            )
            sys.stderr.write({stderr_text!r})
            for event in {native_events!r}:
                print(json.dumps(event), flush=True)
            if {invalid_auth_audit!r} is not None:
                (Path(os.environ["PDE_AGENT_WORKSPACE"]) / "codex-auth-lifecycle.json").write_text({invalid_auth_audit!r})
            raise SystemExit({exit_code})
            """
        ),
        encoding="utf-8",
    )
    executable.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    return executable


def command_program(
    codex_binary: Path,
    auth_file: Path,
    runtime_root: Path,
    *,
    gpu_devices: tuple[str, ...],
) -> CommandAgentProgram:
    return CommandAgentProgram(
        (
            sys.executable,
            str(WORKER_SCRIPT),
            "--codex-binary",
            str(codex_binary),
            "--auth-file",
            str(auth_file),
            "--runtime-root",
            str(runtime_root),
        ),
        decode_nomination,
        environment={
            "PYTHONPATH": str(SOURCE_ROOT),
            "UNRELATED_MARKER": "must-not-reach-codex",
        },
        gpu_devices=gpu_devices,
    )


def write_auth(root: Path) -> tuple[Path, str]:
    auth = root / "auth.json"
    contents = '{"test_only": true}\n'
    auth.write_text(contents, encoding="utf-8")
    auth.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return auth, contents


def staged_incumbent(tmp_path: Path):
    author = tmp_path / "author"
    author.mkdir()
    (author / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    (author / "submission.py").write_text(
        "def prepare_solution(case, solve_budget=None):\n"
        "    return object()\n",
        encoding="utf-8",
    )
    write_candidate_bundle_manifest(
        author,
        files=("submission.py", "helper.py"),
    )
    frozen = freeze_candidate_bundle(author, tmp_path / "sealed")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    shutil.copytree(frozen.root, workspace / "incumbent")
    request = {
        "incumbent": {
            "program_available": True,
            "program": {
                "bundle_path": "incumbent",
                "entrypoint": frozen.entrypoint,
                "tree_sha256": frozen.tree_sha256,
            },
        }
    }
    return workspace, request, frozen


def test_total_time_prompt_includes_generator_and_evaluator_waits() -> None:
    prompt = WORKER._research_prompt({
        "round": {
            "round_index": 1, "active_budget_seconds": None,
            "hard_wall_seconds": 14400, "time_budget_mode": "hard_wall",
            "development_sampling": {"max_cases": 512, "allowed_sources": ["base", "guided"]},
        },
        "agent": {"prompt": "Improve the solver."},
        "candidate_execution": {"case_charged_wall_seconds": 900.0},
    })
    assert "14400 seconds of total elapsed time" in prompt
    assert "generating Cases, evaluating Candidates, and waiting for tools" in prompt
    assert "the timer never pauses" in prompt
    assert "seconds of Agent-active time" not in prompt
    assert "900-second per-Case charged-wall limit" in prompt
    assert "prepare_solution, handle.evaluate, and returning its result" in prompt
    assert "queue and shared-lock waits are excluded" in prompt
    assert "actual Oracle numerical computation" in prompt
    assert "ALL of these waits still count" in prompt


def test_config_and_prompt_fix_model_policy_and_neutral_protocol() -> None:
    config = CONFIG.read_text(encoding="utf-8")
    assert WORKER.MODEL == "gpt-5.6-sol"
    assert WORKER.REASONING_EFFORT == "max"
    assert 'approval_policy = "never"' in config
    assert 'sandbox_mode = "workspace-write"' in config
    assert 'web_search = "live"' in config
    assert "network_access = true" in config
    allowlist = config.split("include_only = [", 1)[1].split("]", 1)[0]
    assert '"PDE_AGENT_EXECUTION_ID"' in allowlist
    assert '"PDE_AGENT_ROUND_LOG_MAX_BYTES"' in allowlist
    assert '"PDE_AGENT_PROTOCOL_MAX_BYTES"' in allowlist
    assert '"PYTHONPYCACHEPREFIX"' in allowlist
    assert "PDE_AGENT_RESPONSE" not in allowlist
    assert "CODEX_HOME" not in allowlist
    assert "PDE_MACRO_" not in allowlist

    prompt = WORKER._research_prompt(
        {
            "round": {
                "round_index": 3,
                "active_budget_seconds": 21600,
                "hard_wall_seconds": 28800,
                "development_sampling": {
                    "max_cases": 200,
                    "allowed_sources": ["base", "guided"],
                },
            },
            "agent": {"prompt": "Pursue the strongest defensible solver."},
        }
    )
    assert prompt.count("Pursue the strongest defensible solver.") == 1
    assert "prepare_solution(case, solve_budget=None) -> handle" in prompt
    assert "handle.evaluate(points_query) -> AnswerView" in prompt
    assert "undecorated, synchronous module-level function" in prompt
    assert "must not rebind it later" in prompt
    assert "CPU-only subprocesses" in prompt
    assert "must execute correctly without CUDA" in prompt
    assert "zero, one, or multiple distinct Candidates" in prompt
    assert "python -m pde_agent_verifier_solver.engine.agent_tool" in prompt
    assert "clone-incumbent DEST" in prompt
    assert "perturb CASE_ID..." in prompt
    assert "export-cases DEST" in prompt
    assert "AgentWireCodec decoder hint" in prompt
    assert "evaluate-many" in prompt
    assert "shared_plan: true" in prompt
    assert "Sequential `evaluate` calls" in prompt
    assert "private_random_points" in prompt
    assert "user_fixed_query" in prompt
    assert "agent_selected_query" in prompt
    assert "`development_point_count`" in prompt
    assert "`allow_grid_truth`" in prompt
    assert "`allow_point_truth`" in prompt
    assert "phase-blind public projection" in prompt
    assert "`roots[i]` corresponds to `children[i]`" in prompt
    assert "21600 seconds of Agent-active time" in prompt
    assert "28800" in prompt
    assert "second real-time hard wall" in prompt
    assert "Macro" not in prompt


@pytest.mark.parametrize("has_public_resources", (False, True))
@pytest.mark.parametrize("has_project_dependency", (False, True))
def test_prompt_scopes_public_resources_and_project_dependencies(
    tmp_path, has_public_resources, has_project_dependency,
):
    from pde_agent_verifier_solver.plugins.ns.public_problem import (
        ns_public_problem_bundle,
    )

    context = agent_context()
    case = context.evaluator.cases[0]
    public_data = dict(case.public_data)
    if has_project_dependency:
        public_data["project_reconstruction"] = {
            "schema": "pde_project_reconstruction_artifact_v1",
            "artifact_id": "ns-piv-trajectory-pod",
            "artifact_kind": "trajectory_pod",
            "format": "numpy_npz",
            "locator": "/project/data/trajectory_pod.npz",
            "sha256": "a" * 64,
            "size_bytes": 100,
        }
    context = replace(
        context,
        evaluator=CasesOnlyEvaluator((replace(case, public_data=public_data),)),
        public_problem_bundle=ns_public_problem_bundle() if has_public_resources else None,
    )
    service = SubprocessAgentExecutionService(
        tmp_path / "turns", hard_wall_seconds=14400,
        capability_facade=UnavailableAgentCapabilityFacade(),
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    program = CommandAgentProgram((sys.executable, "-c", "pass"), decode_nomination)
    request = service._build_request(program, context, "execution", workspace, 14400, {})
    prompt = WORKER._research_prompt(request)

    assert ("public_data.project_reconstruction" in prompt) is has_project_dependency
    if has_project_dependency:
        assert "Read it through its `locator`" in prompt
        assert "do not copy, vendor, rename, or modify" in prompt
    assert ("Public Task resources are in" in prompt) is has_public_resources
    if has_public_resources:
        readme = "public_resources/ns_reference/README.md"
        assert f"`{readme}`" in prompt
        assert "`public_resources/manifest.json`" in prompt
        assert "du/dx + dv/dy = 0" in (workspace / readme).read_text()
    assert '`fields`, `resolution`, and `points`' in prompt
    assert '"fields":["u"]' not in prompt
    assert "Agent-authored notes are fallible" not in prompt
    assert "after meaningful experiments" not in request["round_log"]["instruction"]
    assert "before finishing" in request["round_log"]["instruction"]


def test_canonical_paths_require_execution_id_and_positive_integer_limits(
    tmp_path: Path,
) -> None:
    workspace = tmp_path.resolve()
    nominations = workspace / "nominations"
    nominations.mkdir()
    files = {
        AGENT_REQUEST_ENV: workspace / "request.json",
        AGENT_CAMPAIGN_MEMORY_ENV: workspace / "CAMPAIGN_MEMORY.md",
        AGENT_CANDIDATE_HISTORY_ENV: workspace / "CANDIDATE_HISTORY.json",
        AGENT_ROUND_LOG_ENV: workspace / "ROUND_LOG.md",
    }
    for path in files.values():
        path.write_text("{}\n", encoding="utf-8")
    environment = {
        AGENT_WORKSPACE_ENV: str(workspace),
        **{name: str(path) for name, path in files.items()},
        AGENT_NOMINATIONS_ENV: str(nominations),
        AGENT_RESPONSE_ENV: str(workspace / "response.json"),
        AGENT_EXECUTION_ID_ENV: "execution-123",
        AGENT_ROUND_LOG_MAX_BYTES_ENV: "4096",
        AGENT_PROTOCOL_MAX_BYTES_ENV: "8192",
    }
    with patch.dict(os.environ, environment, clear=True):
        paths = WORKER._canonical_paths(workspace)
    assert paths[AGENT_REQUEST_ENV] == files[AGENT_REQUEST_ENV]
    assert paths[AGENT_RESPONSE_ENV] == workspace / "response.json"

    missing_execution_id = dict(environment)
    del missing_execution_id[AGENT_EXECUTION_ID_ENV]
    with patch.dict(os.environ, missing_execution_id, clear=True):
        with pytest.raises(ValueError, match=AGENT_EXECUTION_ID_ENV):
            WORKER._canonical_paths(workspace)

    for name, invalid in (
        (AGENT_ROUND_LOG_MAX_BYTES_ENV, "not-an-integer"),
        (AGENT_PROTOCOL_MAX_BYTES_ENV, "0"),
    ):
        malformed = dict(environment)
        malformed[name] = invalid
        with patch.dict(os.environ, malformed, clear=True):
            with pytest.raises(ValueError, match="positive integer"):
                WORKER._canonical_paths(workspace)


def test_codex_environment_inherits_the_worker_virtual_environment(
    tmp_path: Path,
) -> None:
    virtual_environment = tmp_path / "worker-venv"
    worker_python = virtual_environment / "bin" / "python"
    inherited_path = os.pathsep.join(("/usr/local/bin", "/usr/bin"))
    environment = {
        **{
            name: f"canonical-{index}"
            for index, name in enumerate(WORKER._CANONICAL_AGENT_ENVIRONMENT)
        },
        "CUDA_VISIBLE_DEVICES": "2",
        "PATH": inherited_path,
        "VIRTUAL_ENV": "/stale/inherited-venv",
        "UNRELATED_MARKER": "must-not-reach-codex",
    }
    with (
        patch.dict(os.environ, environment, clear=True),
        patch.object(WORKER.sys, "executable", str(worker_python)),
        patch.object(WORKER.sys, "prefix", str(virtual_environment)),
        patch.object(WORKER.sys, "base_prefix", "/usr"),
    ):
        codex_environment = WORKER._codex_environment(
            repository_root=REPOSITORY_ROOT,
            workspace=tmp_path / "workspace",
            codex_home=tmp_path / "codex-home",
            agent_home=tmp_path / "agent-home",
        )

    assert codex_environment["PATH"] == os.pathsep.join(
        (str(worker_python.parent), inherited_path)
    )
    assert codex_environment["VIRTUAL_ENV"] == str(virtual_environment)
    assert "UNRELATED_MARKER" not in codex_environment


@pytest.mark.parametrize("agent_threads", ("1", "8"))
def test_codex_numeric_pools_use_agent_allowance_not_controller_default(tmp_path, agent_threads):
    environment = {
        **{name: f"canonical-{index}" for index, name in enumerate(WORKER._CANONICAL_AGENT_ENVIRONMENT)},
        "CUDA_VISIBLE_DEVICES": "2", "OMP_NUM_THREADS": agent_threads, "MKL_NUM_THREADS": agent_threads,
        # Parent shell caps belong to Controller kernels and must not silently
        # reduce the separately configured Agent's local numerical allowance.
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
    }
    with patch.dict(os.environ, environment, clear=True):
        actual = WORKER._codex_environment(
            repository_root=REPOSITORY_ROOT, workspace=tmp_path / "workspace",
            codex_home=tmp_path / "codex-home", agent_home=tmp_path / "agent-home")
    shell_policy = tomllib.loads(CONFIG.read_text())["shell_environment_policy"]
    assert shell_policy["inherit"] == "all"
    # Codex applies this allowlist after receiving _codex_environment. Checking
    # only the wrapper's dictionary misses caps silently dropped before shell
    # tools start NumPy/SciPy or a nested Python process.
    shell_environment = {name: value for name, value in actual.items()
                         if any(fnmatchcase(name, pattern) for pattern in shell_policy["include_only"])}
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        assert actual[name] == agent_threads
        assert shell_environment[name] == agent_threads


def test_clone_incumbent_makes_writable_author_bundle_and_preserves_source(
    tmp_path: Path,
) -> None:
    workspace, request, frozen = staged_incumbent(tmp_path)
    source = workspace / "incumbent"
    source_manifest_before = json.loads(
        (source / "candidate_bundle.json").read_text(encoding="utf-8")
    )
    with patch.dict(os.environ, {AGENT_WORKSPACE_ENV: str(workspace)}, clear=True):
        result = agent_tool._clone_incumbent(request, "candidate-next")

    clone = workspace / "candidate-next"
    assert result["cloned"] == "candidate-next"
    assert result["source"] == "incumbent"
    assert inspect_candidate_bundle(clone).tree_sha256 == frozen.tree_sha256
    author_manifest = json.loads(
        (clone / "candidate_bundle.json").read_text(encoding="utf-8")
    )
    assert set(author_manifest) == {"schema", "entrypoint", "files"}
    assert set(source_manifest_before) == {
        "schema",
        "entrypoint",
        "files",
        "total_bytes",
        "tree_sha256",
    }
    assert json.loads(
        (source / "candidate_bundle.json").read_text(encoding="utf-8")
    ) == source_manifest_before
    for path in (clone, *clone.rglob("*")):
        assert path.stat().st_mode & stat.S_IWUSR
    for path in (source, *source.rglob("*")):
        assert not path.stat().st_mode & stat.S_IWUSR

    (clone / "helper.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert inspect_candidate_bundle(clone).tree_sha256 != frozen.tree_sha256


def test_clone_incumbent_rejects_unsafe_or_existing_destinations(
    tmp_path: Path,
) -> None:
    workspace, request, _ = staged_incumbent(tmp_path)
    existing = workspace / "existing"
    existing.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (workspace / "link").symlink_to(outside, target_is_directory=True)
    with patch.dict(os.environ, {AGENT_WORKSPACE_ENV: str(workspace)}, clear=True):
        for destination in (
            "../escape",
            str(outside / "absolute"),
            "existing",
            "incumbent/child",
            "link/child",
        ):
            with pytest.raises((OSError, ValueError)):
                agent_tool._clone_incumbent(request, destination)
    assert not (tmp_path / "escape").exists()
    assert not (outside / "absolute").exists()
    assert not (outside / "child").exists()


def test_clone_incumbent_reuses_recorded_identity_without_rescanning(tmp_path, monkeypatch):
    from pde_agent_verifier_solver.sdk import candidate_bundle

    workspace, request, _ = staged_incumbent(tmp_path)

    def no_rescan(*args, **kwargs):
        raise AssertionError("cloning a recorded program must not rescan its contents")

    monkeypatch.setattr(candidate_bundle, "_hash_regular_member", no_rescan)
    with patch.dict(os.environ, {AGENT_WORKSPACE_ENV: str(workspace)}, clear=True):
        result = agent_tool._clone_incumbent(request, "candidate-next")
    assert result["cloned"] == "candidate-next"
    assert (workspace / "candidate-next/helper.py").read_bytes() == (workspace / "incumbent/helper.py").read_bytes()


def test_clone_incumbent_removes_its_destination_after_copy_failure(
    tmp_path: Path,
) -> None:
    workspace, request, _ = staged_incumbent(tmp_path)
    with (
        patch.dict(os.environ, {AGENT_WORKSPACE_ENV: str(workspace)}, clear=True),
        patch.object(agent_tool.shutil, "copytree", side_effect=OSError("copy failed")),
    ):
        with pytest.raises(OSError, match="copy failed"):
            agent_tool._clone_incumbent(request, "candidate-partial")
    assert not (workspace / "candidate-partial").exists()


def test_clone_incumbent_cli_routes_without_opening_capability(
    tmp_path: Path,
    capsys,
) -> None:
    workspace, request, frozen = staged_incumbent(tmp_path)
    with (
        patch.dict(os.environ, {AGENT_WORKSPACE_ENV: str(workspace)}, clear=True),
        patch.object(agent_tool, "load_agent_worker_request", return_value=request),
        patch.object(agent_tool, "ControllerAgentCapabilityClient") as capability,
    ):
        assert agent_tool.main(["clone-incumbent", "candidate-next"]) == 0
    capability.assert_not_called()
    result = json.loads(capsys.readouterr().out)
    assert result["cloned"] == "candidate-next"
    assert result["source"] == "incumbent"
    assert (
        inspect_candidate_bundle(workspace / "candidate-next").tree_sha256
        == frozen.tree_sha256
    )


def test_agent_tool_submit_freezes_exact_candidate_bytes_before_journaling(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    nominations = workspace / "nominations"
    nominations.mkdir()
    candidate = workspace / "candidate"
    candidate.mkdir()
    source = candidate / "submission.py"
    source.write_text("VALUE = 'submitted'\n", encoding="utf-8")
    write_candidate_bundle_manifest(candidate, files=("submission.py",))
    submitted_identity = inspect_candidate_bundle(candidate).tree_sha256
    environment = {
        AGENT_WORKSPACE_ENV: str(workspace.resolve()),
        AGENT_NOMINATIONS_ENV: str(nominations.resolve()),
        AGENT_EXECUTION_ID_ENV: "execution-freeze",
    }

    with patch.dict(os.environ, environment, clear=True):
        journal, snapshot_value = agent_tool._commit_candidate_nomination(
            "candidate",
            sequence=1,
            name="Frozen candidate",
            summary="Exact bytes at submission time.",
            usage="Call prepare_solution.",
        )

    snapshot = workspace / snapshot_value
    assert inspect_candidate_bundle(snapshot).tree_sha256 == submitted_identity
    assert (snapshot / "submission.py").read_text(encoding="utf-8") == (
        "VALUE = 'submitted'\n"
    )
    committed = json.loads(journal.read_text(encoding="utf-8"))
    assert committed["payload"] == {
        "bundle_path": snapshot_value,
        "name": "Frozen candidate",
        "summary": "Exact bytes at submission time.",
        "usage": "Call prepare_solution.",
    }
    for path in (snapshot, *snapshot.rglob("*")):
        assert not path.stat().st_mode & stat.S_IWUSR

    source.write_text("VALUE = 'changed later'\n", encoding="utf-8")
    shutil.rmtree(candidate)
    assert inspect_candidate_bundle(snapshot).tree_sha256 == submitted_identity
    assert (snapshot / "submission.py").read_text(encoding="utf-8") == (
        "VALUE = 'submitted'\n"
    )


def test_agent_tool_submit_removes_snapshot_when_journal_publish_fails(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    nominations = workspace / "nominations"
    nominations.mkdir()
    candidate = workspace / "candidate"
    candidate.mkdir()
    (candidate / "submission.py").write_text("VALUE = 1\n", encoding="utf-8")
    write_candidate_bundle_manifest(candidate, files=("submission.py",))
    environment = {
        AGENT_WORKSPACE_ENV: str(workspace.resolve()),
        AGENT_NOMINATIONS_ENV: str(nominations.resolve()),
        AGENT_EXECUTION_ID_ENV: "execution-freeze",
    }

    with patch.dict(os.environ, environment, clear=True):
        submit_agent_nomination(
            {"already": "committed"},
            sequence=1,
        )
        with pytest.raises(FileExistsError):
            agent_tool._commit_candidate_nomination(
                "candidate",
                sequence=1,
                name="Candidate",
                summary="Should not publish.",
                usage="None",
            )

    assert not (
        workspace / ".submitted-candidates" / "nomination-00000001"
    ).exists()
    assert not tuple(
        (workspace / ".submitted-candidates").glob(".staging-*")
    )


def test_timeout_recovers_cli_submission_bytes_not_later_author_edits(
    tmp_path: Path,
) -> None:
    worker = tmp_path / "submit_then_mutate.py"
    worker.write_text(
        textwrap.dedent(
            """\
            import os
            from pathlib import Path
            import time

            from pde_agent_verifier_solver.engine import agent_tool
            from pde_agent_verifier_solver.sdk import write_candidate_bundle_manifest

            workspace = Path(os.environ["PDE_AGENT_WORKSPACE"])
            candidate = workspace / "candidate"
            candidate.mkdir()
            source = candidate / "submission.py"
            source.write_text(
                "MARKER = 'submitted-before-timeout'\\n"
                "def prepare_solution(case, solve_budget=None):\\n"
                "    return object()\\n",
                encoding="utf-8",
            )
            write_candidate_bundle_manifest(candidate, files=("submission.py",))
            assert agent_tool.main([
                "submit", "candidate", "--sequence", "1",
                "--name", "Timeout snapshot", "--summary", "Frozen first",
                "--usage", "Call prepare_solution",
            ]) == 0
            source.write_text(
                "raise RuntimeError('later author edit must not be recovered')\\n",
                encoding="utf-8",
            )
            time.sleep(60)
            """
        ),
        encoding="utf-8",
    )
    codec = PythonCandidateCodec(task_kind=TaskKind.SOLVER)
    program = CommandAgentProgram(
        (sys.executable, str(worker)),
        codec.decode_nomination,
        environment={"PYTHONPATH": str(SOURCE_ROOT)},
    )
    service = SubprocessAgentExecutionService(
        tmp_path / "turns",
        hard_wall_seconds=5,
        capability_facade=UnavailableAgentCapabilityFacade(),
        termination_grace_seconds=0.2,
        poll_interval_seconds=0.01,
    )

    outcome = service.execute(program, agent_context(), 2.0)

    assert outcome.execution_status == "timed_out"
    assert len(outcome.nominations) == 1
    artifact = outcome.nominations[0]
    recovered = inspect_candidate_bundle(artifact.program.bundle_locator)
    assert (recovered.root / "submission.py").read_text(encoding="utf-8").startswith(
        "MARKER = 'submitted-before-timeout'\n"
    )
    audit = service.audits[-1]
    assert audit.status == "hard_wall_timeout"
    assert audit.nomination_count == 1
    assert audit.rejected_nomination_count == 0
    author = Path(audit.workspace) / "candidate" / "submission.py"
    assert "later author edit" in author.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("worker_settings", "expected_timeout"),
    (
        ({"controller_rpc_timeout_seconds": 7_200}, 7_200.0),
        ({}, 300.0),
    ),
)
def test_agent_tool_passes_controller_authored_rpc_timeout(
    worker_settings: dict[str, object],
    expected_timeout: float,
    capsys,
) -> None:
    request = MappingProxyType(
        {"worker_settings": MappingProxyType(worker_settings)}
    )
    with (
        patch.object(agent_tool, "load_agent_worker_request", return_value=request),
        patch.object(agent_tool, "ControllerAgentCapabilityClient") as capability,
    ):
        capability.from_request.return_value.current_status.return_value = ((), [])
        assert agent_tool.main(["status"]) == 0

    capability.from_request.assert_called_once_with(
        request,
        timeout_seconds=expected_timeout,
    )
    assert json.loads(capsys.readouterr().out)["cases"] == []


def test_agent_tool_status_reports_cpu_only_candidate_execution() -> None:
    status = agent_tool._status(agent_tool_request(), ())

    assert status["candidate_execution"] == {
        "device": "cpu",
        "cuda_available": False,
    }


@pytest.mark.parametrize(
    "invalid_timeout",
    (True, 0, -1, float("nan"), float("inf"), "7200"),
)
def test_agent_tool_rejects_non_positive_non_finite_or_non_numeric_rpc_timeout(
    invalid_timeout: object,
) -> None:
    request = MappingProxyType(
        {
            "worker_settings": MappingProxyType(
                {"controller_rpc_timeout_seconds": invalid_timeout}
            )
        }
    )

    with pytest.raises((TypeError, ValueError), match="must be"):
        agent_tool._controller_rpc_timeout_seconds(request)


@pytest.mark.parametrize("query_kind", (QueryKind.POINTS, QueryKind.GRID))
def test_agent_tool_evaluate_many_uses_one_shared_batch_and_returns_full_views(
    tmp_path: Path,
    capsys,
    query_kind: QueryKind,
) -> None:
    workspace = tmp_path.resolve()
    if query_kind is QueryKind.POINTS:
        query = AnswerQuery.at_points(
            fields=("velocity_x", "velocity_y"),
            points=(
                {"t": 0.0, "x": -0.5, "y": 0.25},
                {"t": 0.1, "x": 0.5, "y": -0.25},
            ),
        )
        view = AnswerView(
            kind=query_kind,
            fields={"velocity_x": (1.0, 2.0), "velocity_y": (-1.0, -2.0)},
            points=query.points,
            shape=(2,),
        )
    else:
        query = AnswerQuery.grid(
            fields=("velocity_x", "velocity_y"),
            x=2,
            y=1,
        )
        view = AnswerView(
            kind=query_kind,
            fields={"velocity_x": (1.0, 2.0), "velocity_y": (-1.0, -2.0)},
            axes={"x": (-0.5, 0.5), "y": (0.25,)},
            shape=(2, 1),
        )
    query_path = workspace / "query.json"
    query_path.write_text(
        json.dumps(
            {
                "kind": query.kind.value,
                "fields": list(query.fields),
                "resolution": dict(query.resolution),
                "points": [dict(point) for point in query.points],
            }
        ),
        encoding="utf-8",
    )
    responses = tuple(
        EvaluationResponse(
            artifact_id=f"artifact-{index}",
            primary_error=0.1 * index,
            metrics=MappingProxyType({"relative_l2": 0.2 * index}),
            cases=(
                EvaluationCaseResult(
                    case_id="opaque-case-a",
                    primary_error=0.3 * index,
                    metrics=MappingProxyType({"rmse": 0.4 * index}),
                    prediction=view,
                    truth=view,
                ),
            ),
        )
        for index in (1, 2)
    )
    request = agent_tool_request()
    with (
        patch.dict(
            os.environ,
            {
                AGENT_WORKSPACE_ENV: str(workspace),
                AGENT_PROTOCOL_MAX_BYTES_ENV: str(1024 * 1024),
            },
            clear=True,
        ),
        patch.object(agent_tool, "load_agent_worker_request", return_value=request),
        patch.object(
            agent_tool,
            "ControllerAgentCapabilityClient",
            autospec=True,
        ) as capability,
    ):
        client = capability.from_request.return_value
        client.evaluate_many.return_value = responses
        assert (
            agent_tool.main(
                [
                    "evaluate-many",
                    "--bundle",
                    "candidate-a",
                    "--bundle",
                    "candidate-b",
                    "--case-id",
                    "opaque-case-a",
                    "--case-id",
                    "opaque-case-b",
                    "--query-json",
                    "query.json",
                    "--return-truth",
                ]
            )
            == 0
        )

    client.evaluate_many.assert_called_once_with(
        ({"bundle_path": "candidate-a"}, {"bundle_path": "candidate-b"}),
        case_ids=("opaque-case-a", "opaque-case-b"),
        query=query,
        return_truth=True,
    )
    client.evaluate.assert_not_called()
    client.current_cases.assert_not_called()
    output = json.loads(capsys.readouterr().out)
    assert output["shared_plan"] is True
    assert [item["bundle_path"] for item in output["evaluations"]] == [
        "candidate-a",
        "candidate-b",
    ]
    assert [item["artifact_id"] for item in output["evaluations"]] == [
        "artifact-1",
        "artifact-2",
    ]
    disclosed = output["evaluations"][0]["cases"][0]
    assert disclosed["prediction"]["fields"]["velocity_x"] == [1.0, 2.0]
    assert disclosed["truth"]["fields"]["velocity_y"] == [-1.0, -2.0]
    assert disclosed["truth"]["kind"] == query_kind.value
    if query_kind is QueryKind.POINTS:
        assert disclosed["truth"]["points"] == [dict(point) for point in query.points]
        assert disclosed["truth"]["axes"] == {}
    else:
        assert disclosed["truth"]["axes"] == {"x": [-0.5, 0.5], "y": [0.25]}
        assert disclosed["truth"]["points"] == []


@pytest.mark.parametrize("operation", ("evaluate", "evaluate-many"))
def test_agent_tool_returns_case_scores_without_serializing_answer_views(
    tmp_path: Path, capsys, operation: str,
) -> None:
    # Even if a client object contains answer views, the CLI must not disclose
    # them unless the user explicitly requested them under the existing policy.
    hidden = AnswerView(
        kind=QueryKind.POINTS,
        fields={"u": (987.654,)},
        points=({"x": 0.123456},),
        shape=(1,),
    )
    response = EvaluationResponse(
        artifact_id="candidate-result", primary_error=0.2,
        metrics={"relative_l2": 0.2},
        cases=tuple(
            EvaluationCaseResult(
                case_id=f"public-case-{i}", primary_error=error,
                metrics={"relative_l2": error, "candidate_solve_wall_seconds": i + 1.0},
                prediction=hidden, truth=hidden,
            ) for i, error in enumerate((0.1, 0.3))
        ),
    )
    request = agent_tool_request()
    with (
        patch.dict(os.environ, {AGENT_WORKSPACE_ENV: str(tmp_path.resolve())}, clear=True),
        patch.object(agent_tool, "load_agent_worker_request", return_value=request),
        patch.object(agent_tool, "ControllerAgentCapabilityClient", autospec=True) as capability,
    ):
        client = capability.from_request.return_value
        client.current_cases.return_value = ()
        client.evaluate.return_value = response
        client.evaluate_many.return_value = (response, response)
        args = (
            ["evaluate", "candidate", "public-case-0", "public-case-1"]
            if operation == "evaluate" else
            ["evaluate-many", "--bundle", "candidate", "--bundle", "other",
             "--case-id", "public-case-0", "--case-id", "public-case-1"]
        )
        assert agent_tool.main(args) == 0
    raw = capsys.readouterr().out
    payload = json.loads(raw)
    results = payload["evaluations"] if operation == "evaluate-many" else [payload]
    for result in results:
        assert [row["case_id"] for row in result["cases"]] == ["public-case-0", "public-case-1"]
        assert [row["primary_error"] for row in result["cases"]] == [0.1, 0.3]
        assert all(row["truth"] is None and row["prediction"] is None for row in result["cases"])
        assert result["cases"][1]["metrics"]["candidate_solve_wall_seconds"] == 2.0
    assert "987.654" not in raw and "0.123456" not in raw


def test_agent_tool_single_evaluate_without_truth_keeps_aggregate_schema(
    tmp_path: Path,
    capsys,
) -> None:
    request = agent_tool_request()
    response = EvaluationResponse(
        artifact_id="aggregate-only",
        primary_error=0.125,
        metrics=MappingProxyType({"relative_l2": 0.25}),
        cases=(),
    )
    with (
        patch.dict(
            os.environ,
            {AGENT_WORKSPACE_ENV: str(tmp_path.resolve())},
            clear=True,
        ),
        patch.object(agent_tool, "load_agent_worker_request", return_value=request),
        patch.object(
            agent_tool,
            "ControllerAgentCapabilityClient",
            autospec=True,
        ) as capability,
    ):
        client = capability.from_request.return_value
        client.current_cases.return_value = ()
        client.evaluate.return_value = response
        assert agent_tool.main(["evaluate", "candidate", "opaque-case"]) == 0

    client.evaluate.assert_called_once_with(
        {"bundle_path": "candidate"},
        case_ids=("opaque-case",),
        query=None,
        return_truth=False,
    )
    assert json.loads(capsys.readouterr().out) == {
        "artifact_id": "aggregate-only",
        "cases": [],
        "metrics": {"relative_l2": 0.25},
        "primary_error": 0.125,
    }


@pytest.mark.parametrize(
    "payload",
    (
        "[]",
        '{"kind":"points","fields":["u"],"resolution":{},"points":[],"extra":1}',
        '{"kind":"points","fields":["u","u"],"resolution":{},"points":[{"x":0.0}]}',
        '{"kind":"points","kind":"grid","fields":["u"],"resolution":{"x":2},"points":[]}',
        '{"kind":"unknown","fields":["u"],"resolution":{},"points":[{"x":0.0}]}',
        '{"kind":"grid","fields":["u"],"resolution":{},"points":[]}',
        '{"kind":"grid","fields":["u"],"resolution":{"x":2},"points":[{"x":0.0}]}',
        '{"kind":"points","fields":["u"],"resolution":{"x":2},"points":[{"x":0.0}]}',
        '{"kind":"points","fields":["u"],"resolution":{},"points":[]}',
        '{"kind":"grid","fields":["u"],"resolution":{"x":true},"points":[]}',
        '{"kind":"grid","fields":["u"],"resolution":{"x":0},"points":[]}',
        '{"kind":"points","fields":["u"],"resolution":{},"points":[{"x":true}]}',
        '{"kind":"points","fields":["u"],"resolution":{},"points":[{"x":NaN}]}',
        '{"kind":"points","fields":[""],"resolution":{},"points":[{"x":0.0}]}',
        '{"kind":"grid","fields":["u"],"resolution":{"":2},"points":[]}',
    ),
)
def test_agent_tool_query_json_rejects_malformed_or_ambiguous_queries(
    tmp_path: Path,
    payload: str,
) -> None:
    query = tmp_path / "query.json"
    query.write_text(payload, encoding="utf-8")
    with patch.dict(
        os.environ,
        {
            AGENT_WORKSPACE_ENV: str(tmp_path.resolve()),
            AGENT_PROTOCOL_MAX_BYTES_ENV: "4096",
        },
        clear=True,
    ):
        with pytest.raises((TypeError, ValueError)):
            agent_tool._load_query_json("query.json")


def test_agent_tool_query_json_rejects_unsafe_missing_and_oversized_files(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text(
        '{"kind":"grid","fields":["u"],"resolution":{"x":2},"points":[]}',
        encoding="utf-8",
    )
    (workspace / "query-link.json").symlink_to(outside)
    (workspace / "too-large.json").write_bytes(b" " * 257)
    with patch.dict(
        os.environ,
        {
            AGENT_WORKSPACE_ENV: str(workspace.resolve()),
            AGENT_PROTOCOL_MAX_BYTES_ENV: "256",
        },
        clear=True,
    ):
        for path in (
            str(outside),
            "../outside.json",
            "query-link.json",
            "missing.json",
            "too-large.json",
        ):
            with pytest.raises((OSError, ValueError)):
                agent_tool._load_query_json(path)


def test_agent_tool_exports_lossless_phase_blind_public_cases_atomically(
    tmp_path: Path,
) -> None:
    workspace = tmp_path.resolve()
    public_data = {
        "array": np.arange(64, dtype=np.float64).reshape(8, 8),
        "complex": 1.5 - 2.25j,
        "bytes": b"public-bytes",
        "tuple": ("value", 3),
        "nested": {"finite": 0.5},
    }
    case = Case(
        case_id="candidate-case-opaque",
        source=CaseSource.GENERATED,
        visibility=CaseVisibility.DEVELOPMENT,
        public_data=public_data,
    )
    request = agent_tool_request(inline_array_bytes=32)
    with patch.dict(
        os.environ,
        {AGENT_WORKSPACE_ENV: str(workspace)},
        clear=True,
    ):
        result = agent_tool._export_cases(
            request,
            (case,),
            (case.case_id,),
            "public-case",
        )
        second = agent_tool._export_cases(
            request,
            (case,),
            (case.case_id,),
            "public-case-copy",
        )

    destination = workspace / "public-case"
    manifest_path = destination / "cases.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(manifest) == {"schema", "cases"}
    assert manifest["schema"] == "canonical-agent-public-cases-v1"
    assert len(manifest["cases"]) == 1
    exported_case = manifest["cases"][0]
    assert set(exported_case) == {"case_id", "public_data"}
    assert exported_case["case_id"] == case.case_id
    private_fields = {
        "source",
        "visibility",
        "round_index",
        "parent_case_id",
        "tags",
        "truth",
    }
    assert not (private_fields & set(exported_case))
    codec = AgentWireCodec(
        destination / "arrays",
        inline_array_bytes=32,
        max_sidecar_bytes=1024 * 1024,
    )
    decoded = codec.decode(exported_case["public_data"])
    np.testing.assert_array_equal(decoded["array"], public_data["array"])
    assert decoded["complex"] == public_data["complex"]
    assert decoded["bytes"] == public_data["bytes"]
    assert decoded["tuple"] == public_data["tuple"]
    assert decoded["nested"] == public_data["nested"]
    assert result["case_ids"] == [case.case_id]
    assert result["array_sidecars"] == 1
    assert "AgentWireCodec" in result["decoder"]
    assert stat.S_IMODE(destination.stat().st_mode) == 0o700
    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o400
    sidecar = next((destination / "arrays").glob("*.npy"))
    assert stat.S_IMODE(sidecar.stat().st_mode) == 0o400
    assert manifest_path.read_bytes() == (
        workspace / second["manifest"]
    ).read_bytes()


def test_agent_tool_export_cases_rejects_unsafe_ids_and_cleans_partial_output(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (workspace / "link").symlink_to(outside, target_is_directory=True)
    (workspace / "existing").mkdir()
    case = Case(
        case_id="candidate-case-opaque",
        source=CaseSource.GENERATED,
        visibility=CaseVisibility.DEVELOPMENT,
        public_data={"value": 1.0},
    )
    request = agent_tool_request()
    with patch.dict(
        os.environ,
        {AGENT_WORKSPACE_ENV: str(workspace.resolve())},
        clear=True,
    ):
        for case_ids, destination in (
            (("unknown",), "unknown"),
            ((case.case_id, case.case_id), "duplicate"),
            ((case.case_id,), "../escape"),
            ((case.case_id,), str(outside / "absolute")),
            ((case.case_id,), "link/child"),
            ((case.case_id,), "existing"),
        ):
            with pytest.raises((OSError, ValueError)):
                agent_tool._export_cases(
                    request,
                    (case,),
                    case_ids,
                    destination,
                )
        with (
            patch.object(
                agent_tool.AgentWireCodec,
                "encode",
                side_effect=KeyboardInterrupt,
            ),
            pytest.raises(KeyboardInterrupt),
        ):
            agent_tool._export_cases(
                request,
                (case,),
                (case.case_id,),
                "interrupted",
            )
    assert not (tmp_path / "escape").exists()
    assert not (outside / "absolute").exists()
    assert not (outside / "child").exists()
    assert not (workspace / "interrupted").exists()
    assert not tuple(workspace.glob(".interrupted-partial-*"))


@pytest.mark.parametrize("settings", (
    {},
    {"model": "configured-backbone", "reasoning_effort": "low"},
    {"model": "gpt-6-astra", "reasoning_effort": "ultra"},
    {"nomination_policy": "latest"},
))
def test_mock_codex_success_preserves_response_nominations_and_log(
    tmp_path: Path,
    settings: dict[str, str],
) -> None:
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    capture_path = tmp_path / "capture.json"
    codex = write_fake_codex(
        tmp_path,
        capture_path=capture_path,
        exit_code=0,
        nomination_count=2,
    )
    auth, auth_contents = write_auth(tmp_path)
    # This is a protocol test; allow for shared-filesystem interpreter startup.
    service = SubprocessAgentExecutionService(
        tmp_path / "turns",
        hard_wall_seconds=20,
        capability_facade=UnavailableAgentCapabilityFacade(),
    )

    outcome = service.execute(
        replace(command_program(
            codex,
            auth,
            runtime_root,
            gpu_devices=("7",),
        ), worker_settings=settings),
        agent_context(),
        15,
    )

    assert outcome.execution_status == "completed"
    expected_modes = [2] if settings.get("nomination_policy") == "latest" else [1, 2]
    assert len(outcome.nominations) == len(expected_modes)
    assert [
        item.metadata["max_resolved_mode"] for item in outcome.nominations
    ] == expected_modes
    assert outcome.round_log == "# Mock Codex log\n\nCommitted 2 nomination(s)."
    audit = service.audits[-1]
    assert audit.status == "completed"
    assert audit.nomination_count == len(expected_modes)
    assert audit.nomination_policy == settings.get("nomination_policy", "all")
    assert audit.nomination_journal_entry_count == 2
    assert audit.final_nomination_sequence == (2 if settings.get("nomination_policy") == "latest" else None)
    workspace = Path(audit.workspace)
    response = json.loads(
        (workspace / "response.json").read_text(encoding="utf-8")
    )
    assert response["status"] == "completed"
    assert response["execution_id"] == audit.execution_id
    assert len(tuple((workspace / "nominations").glob("nomination-*.json"))) == 2
    assert (workspace / "codex-auth-lifecycle.json").is_file()
    assert (workspace / "codex-last-message.txt").is_file()

    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    assert capture["execution_id"] == audit.execution_id
    assert capture["request_execution_id"] == audit.execution_id
    assert capture["round_log_max_bytes"] == str(8 * 1024 * 1024)
    assert capture["protocol_max_bytes"] == str(8 * 1024 * 1024)
    assert capture["cuda_visible_devices"] == "7"
    assert capture["pycache_prefix"] == str(workspace / "pycache")
    assert capture["candidate_pycache_exists"] is False
    assert capture["redirected_pyc_exists"] is True
    assert capture["response_exposed"] is False
    assert capture["unrelated_exposed"] is False
    assert capture["auth_missing"] is True
    assert capture["auth_audit_present"] is True
    assert capture["argv"][capture["argv"].index("--model") + 1] == settings.get("model", "gpt-5.6-sol")
    assert f'model_reasoning_effort={json.dumps(settings.get("reasoning_effort", "max"))}' in capture["argv"]
    assert "--ephemeral" not in capture["argv"]
    assert json.loads((workspace / "codex-recovery.json").read_text())["until_deadline"] is True
    assert "--strict-config" in capture["argv"]
    assert capture["config_text"] == CONFIG.read_text()
    assert "restricted_namespace_workspace_edit_v1" in capture["config_text"]
    assert not any(value in capture["argv"] for value in (
        "--yolo", "--dangerously-bypass-approvals-and-sandbox", "danger-full-access",
    ))
    if settings.get("nomination_policy") == "latest":
        assert "Choose exactly one final Candidate" in capture["prompt"]
        assert "explicitly submit that version again" in capture["prompt"]
        assert "zero, one, or multiple distinct Candidates" not in capture["prompt"]
    assert "prepare_solution(case, solve_budget=None)" in capture["prompt"]
    assert "clone-incumbent DEST" in capture["prompt"]
    assert (
        capture["prompt"].count(
            "Improve the public solver without assuming a solver family."
        )
        == 1
    )
    assert auth.read_text(encoding="utf-8") == auth_contents
    assert tuple(runtime_root.iterdir()) == ()


def test_mock_codex_failure_retains_committed_nomination_and_round_log(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    codex = write_fake_codex(
        tmp_path,
        capture_path=tmp_path / "capture.json",
        exit_code=23,
        nomination_count=1,
    )
    auth, _ = write_auth(tmp_path)
    service = SubprocessAgentExecutionService(
        tmp_path / "turns",
        hard_wall_seconds=20,
        capability_facade=UnavailableAgentCapabilityFacade(),
    )

    outcome = service.execute(
        command_program(
            codex,
            auth,
            runtime_root,
            gpu_devices=("3",),
        ),
        agent_context(),
        15,
    )

    assert outcome.execution_status == "failed"
    assert len(outcome.nominations) == 1
    assert outcome.round_log == "# Mock Codex log\n\nCommitted 1 nomination(s)."
    audit = service.audits[-1]
    assert audit.status == "worker_failed"
    assert audit.nomination_count == 1
    workspace = Path(audit.workspace)
    response = json.loads(
        (workspace / "response.json").read_text(encoding="utf-8")
    )
    assert response["status"] == "failed"
    assert response["execution_id"] == audit.execution_id
    assert "status 23" in response["error"]
    assert (workspace / "ROUND_LOG.md").is_file()
    assert tuple(runtime_root.iterdir()) == ()


@pytest.mark.parametrize("events,code,expected", [
    (({"type": "error", "message": "Invalid prompt: potentially violating our usage policy; Bearer synthetic-secret", "request": {"prompt": "PRIVATE_REQUEST"}},
      {"type": "turn.failed", "error": {"message": "Invalid prompt: potentially violating our usage policy; synthetic-secret"}}),
     1, "native turn.failed: prompt rejected by policy"),
    (({"type": "turn.failed", "error": {"code": "content_policy_violation", "message": "PRIVATE_REQUEST"}},),
     0, "native turn.failed: prompt rejected by policy"),
    (({"type": "error", "message": "stream disconnected"},
      {"type": "turn.failed", "error": {"code": "content_policy_violation"}}),
     1, "native turn.failed: prompt rejected by policy"),
    (({"type": "turn.failed", "error": {"message": "Invalid prompt: malformed input; synthetic-secret"}},),
     1, "native turn.failed: invalid prompt / request"),
    (({"type": "turn.failed", "error": {"message": "synthetic-secret PRIVATE_REQUEST"}},),
     1, "native turn.failed: unclassified error"),
    (({"type": "error", "message": "stream disconnected"},
      {"type": "turn.failed", "error": {"message": "connection reset"}},
      {"type": "turn.completed"}), 0, None),
    (({"type": "item.completed", "item": {"type": "command_execution", "exit_code": 1,
                                          "aggregated_output": "Invalid prompt: violating our usage policy"}},
      {"type": "turn.completed"}), 0, None),
])
def test_mock_codex_native_failure_attribution_preserves_commits_without_retry(
    tmp_path, events, code, expected,
):
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    capture_path = tmp_path / "capture.json"
    codex = write_fake_codex(
        tmp_path, capture_path=capture_path, exit_code=code, nomination_count=1,
        native_events=events, stderr_text="helper: No space left on device\n",
    )
    auth, _ = write_auth(tmp_path)
    # Failure attribution, not a three-second startup deadline, is under test.
    service = SubprocessAgentExecutionService(
        tmp_path / "turns", hard_wall_seconds=20,
        capability_facade=UnavailableAgentCapabilityFacade(),
    )
    outcome = service.execute(command_program(codex, auth, runtime_root, gpu_devices=("3",)),
                              agent_context(), 15)
    assert len(outcome.nominations) == 1
    assert outcome.round_log == "# Mock Codex log\n\nCommitted 1 nomination(s)."
    assert outcome.execution_status == ("failed" if expected else "completed")
    workspace = Path(service.audits[-1].workspace)
    response = json.loads((workspace / "response.json").read_text())
    error = response.get("error") or ""
    if expected:
        assert f"status {code}" in error
        assert expected in error
    else:
        assert not error
    if "invalid prompt / request" in error:
        assert "policy" not in error
    if "policy" in error:
        assert "not automatically retryable" in error
        assert "connection interrupted" not in error
    assert len(error) < 1024
    assert "synthetic-secret" not in error and "PRIVATE_REQUEST" not in error
    assert capture_path.with_suffix(".invocations").read_text().splitlines() == ["invoked"]
    assert tuple(runtime_root.iterdir()) == ()
    # Inspection must not consume, rewrite, or suppress the raw native stream.
    stdout = Path(service.audits[-1].stdout_path).read_text()
    assert [json.loads(line) for line in stdout.splitlines()] == [{"type": "thread.started"}, *events]
    assert "No space left on device" in Path(service.audits[-1].stderr_path).read_text()
    assert json.loads((workspace / "codex-auth-lifecycle.json").read_text())["auth_removed_before_first_event_forwarded"] is True


@pytest.mark.parametrize("code", [1, 0])
def test_mock_codex_failure_before_first_event_keeps_cause_and_auth_failure(tmp_path, code):
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    capture_path = tmp_path / "capture.json"
    # Exceed a pipe buffer to verify that stderr is drained before stdout starts.
    stderr_text = "authentication failed; synthetic-secret PRIVATE_REQUEST\n" + "x" * 100000
    codex = write_fake_codex(
        tmp_path, capture_path=capture_path, exit_code=code, nomination_count=0,
        before_first_event=True, stderr_text=stderr_text,
    )
    auth, _ = write_auth(tmp_path)
    service = SubprocessAgentExecutionService(
        tmp_path / "turns", hard_wall_seconds=20,
        capability_facade=UnavailableAgentCapabilityFacade(),
    )
    with pytest.raises(AgentExecutionError, match="before its first event: authentication rejected"):
        service.execute(command_program(codex, auth, runtime_root, gpu_devices=("3",)),
                        agent_context(), 15)
    audit = service.audits[-1]
    response = json.loads((Path(audit.workspace) / "response.json").read_text())
    assert response["status"] == "failed"
    assert f"status {code if code else 97}" in response["error"]
    assert "auth safety check incomplete" in response["error"]
    assert "synthetic-secret" not in response["error"] and "PRIVATE_REQUEST" not in response["error"]
    assert len(response["error"]) < 1024
    assert Path(audit.stderr_path).read_text() == stderr_text
    assert capture_path.with_suffix(".invocations").read_text().splitlines() == ["invoked"]
    assert tuple(runtime_root.iterdir()) == ()


@pytest.mark.parametrize("audit_text", ["{}", "[]"])
def test_mock_codex_success_cannot_hide_invalid_auth_audit(tmp_path, audit_text):
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    codex = write_fake_codex(
        tmp_path, capture_path=tmp_path / "capture.json", exit_code=0, nomination_count=1,
        native_events=({"type": "turn.completed"},), invalid_auth_audit=audit_text,
    )
    auth, _ = write_auth(tmp_path)
    service = SubprocessAgentExecutionService(
        tmp_path / "turns", hard_wall_seconds=20,
        capability_facade=UnavailableAgentCapabilityFacade(),
    )
    outcome = service.execute(command_program(codex, auth, runtime_root, gpu_devices=("3",)),
                              agent_context(), 15)
    assert outcome.execution_status == "failed"
    assert len(outcome.nominations) == 1
    assert "auth safety check incomplete" in service.audits[-1].error
    assert tuple(runtime_root.iterdir()) == ()


def test_mock_codex_policy_failure_survives_non_object_auth_audit(tmp_path):
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    codex = write_fake_codex(
        tmp_path, capture_path=tmp_path / "capture.json", exit_code=1, nomination_count=1,
        native_events=({"type": "turn.failed", "error": {"code": "content_policy_violation"}},),
        invalid_auth_audit="[]",
    )
    auth, _ = write_auth(tmp_path)
    service = SubprocessAgentExecutionService(
        tmp_path / "turns", hard_wall_seconds=20,
        capability_facade=UnavailableAgentCapabilityFacade(),
    )
    outcome = service.execute(command_program(codex, auth, runtime_root, gpu_devices=("3",)),
                              agent_context(), 15)
    assert outcome.execution_status == "failed"
    assert len(outcome.nominations) == 1
    assert "prompt rejected by policy" in service.audits[-1].error
    assert "auth safety check incomplete" in service.audits[-1].error
    assert "AttributeError" not in service.audits[-1].error
    assert tuple(runtime_root.iterdir()) == ()


def test_native_event_inspection_is_bounded_and_ignores_non_events():
    spec = importlib.util.spec_from_file_location(
        "_auth_supervisor_under_test", REPOSITORY_ROOT / "scripts/run_codex_with_ephemeral_auth.py")
    supervisor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(supervisor)
    events = supervisor._NativeEvents()
    # Oversized and malformed lines must not corrupt parsing of following events.
    events.feed(b'{"type":"error","message":"' + b"x" * (supervisor._EVENT_MAX_BYTES + 1))
    assert not events.pending and events.discarding
    events.feed(b'"}\n[]\n{"type":[]}\n\xff\n')
    events.feed(b'{"type":"turn.failed","error":{"code":"content_')
    events.feed(b'policy_violation"}}')
    events.finish()
    assert events.failure == "native turn.failed: prompt rejected by policy (not automatically retryable)"


def test_auth_supervisor_thread_start_failure_removes_auth_and_stops_child(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "_auth_supervisor_start_failure", REPOSITORY_ROOT / "scripts/run_codex_with_ephemeral_auth.py")
    supervisor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(supervisor)
    auth = tmp_path / "auth.json"
    auth.write_text("synthetic-auth")
    monkeypatch.setattr(sys, "argv", ["supervisor", "--auth", str(auth),
                                     "--audit", str(tmp_path / "audit.json"), "--", "fake-codex"])
    process = object()
    stopped = []
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(supervisor.signal, "signal", lambda *args: None)
    monkeypatch.setattr(supervisor, "_terminate_process_group", lambda child: stopped.append(child))
    def fail_start(_thread):
        raise RuntimeError("cannot start a new thread")
    monkeypatch.setattr(supervisor.threading.Thread, "start", fail_start)
    with pytest.raises(RuntimeError, match="cannot start a new thread"):
        supervisor.main()
    assert not auth.exists()
    assert stopped == [process]


def test_multiple_gpu_assignment_is_rejected_and_runtime_is_removed(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    capture_path = tmp_path / "capture.json"
    codex = write_fake_codex(
        tmp_path,
        capture_path=capture_path,
        exit_code=0,
        nomination_count=1,
    )
    auth, _ = write_auth(tmp_path)
    service = SubprocessAgentExecutionService(
        tmp_path / "turns",
        hard_wall_seconds=20,
        capability_facade=UnavailableAgentCapabilityFacade(),
    )

    with pytest.raises(AgentExecutionError, match="CUDA_VISIBLE_DEVICES"):
        service.execute(
            command_program(
                codex,
                auth,
                runtime_root,
                gpu_devices=("0", "1"),
            ),
            agent_context(),
            15,
        )

    audit = service.audits[-1]
    assert audit.status == "worker_failed"
    workspace = Path(audit.workspace)
    response = json.loads(
        (workspace / "response.json").read_text(encoding="utf-8")
    )
    assert response["status"] == "failed"
    assert "exactly one GPU" in response["error"]
    assert not capture_path.exists()
    assert tuple(runtime_root.iterdir()) == ()


@pytest.mark.parametrize("settings", ({"model": ""}, {"reasoning_effort": None}, []))
def test_model_settings_rejects_invalid_worker_configuration(settings):
    with pytest.raises((TypeError, ValueError)):
        WORKER._model_settings({"worker_settings": settings})


@pytest.mark.parametrize("budget", (0, -1, True, float("nan"), float("inf"), "900"))
def test_prompt_rejects_invalid_candidate_execution_budget(budget):
    with pytest.raises(ValueError, match="per-Case execution limit"):
        WORKER._research_prompt({
            "round": {"round_index": 1, "hard_wall_seconds": 14400},
            "agent": {"prompt": "Improve the solver."},
            "candidate_execution": {"case_charged_wall_seconds": budget},
        })


def test_public_candidate_budget_reaches_request_status_and_prompt(tmp_path):
    # Scientific launch configuration comes from the actual Controller service,
    # never a stale worker_settings copy or a hardcoded prompt constant.
    context = replace(agent_context(), candidate_solve_budget_seconds=37.5)
    service = SubprocessAgentExecutionService(
        tmp_path / "turns", hard_wall_seconds=14400,
        capability_facade=UnavailableAgentCapabilityFacade(),
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    program = CommandAgentProgram(
        (sys.executable, "-c", "pass"), decode_nomination,
        worker_settings={"candidate_execution": {
            "device": "cpu", "cuda_available": False,
            "case_charged_wall_seconds": 999,
        }},
    )
    request = service._build_request(program, context, "execution", workspace, 14400, {})
    status = agent_tool._status(request, ())
    assert status["candidate_execution"] == {
        "device": "cpu", "cuda_available": False,
        "case_charged_wall_seconds": 37.5,
        "budget_scope": "entire_candidate_worker_including_prepare_and_query",
        "excluded_time": "controller_metered_queue_and_lock_wait_only",
        "oracle_numerical_execution_is_charged": True,
    }
    assert "37.5-second per-Case charged-wall limit" in WORKER._research_prompt(request)


@pytest.mark.parametrize("budget", (0, -1, True, float("nan"), float("inf"), "900"))
def test_agent_context_rejects_invalid_candidate_budget(budget):
    with pytest.raises(ValueError, match="candidate_solve_budget_seconds"):
        replace(agent_context(), candidate_solve_budget_seconds=budget)
