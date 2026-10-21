"""Canonical Python Candidate bundle binding for subprocess Agent runs."""

from __future__ import annotations

import ast
from contextlib import contextmanager
import importlib.util
import json
import secrets
import shutil
import sys
from pathlib import Path, PurePosixPath
from threading import Lock, RLock
from time import perf_counter
from typing import Mapping

from .candidate_diagnostics import candidate_exception_diagnostic

from pde_agent_verifier_solver.sdk import Artifact, TaskKind
from pde_agent_verifier_solver.sdk.candidate_bundle import (
    CandidateBundleLimits,
    freeze_candidate_bundle,
    inspect_candidate_bundle,
)


class PythonCandidateError(ValueError):
    """A submitted Python Candidate cannot be safely bound."""


_IMPORT_LOCK = Lock()


class PythonCandidateProgram:
    """Lazy, deepcopy-safe binding to one frozen Python Candidate bundle."""

    _deferred_solution_abi = True

    def __init__(
        self,
        locator: str | Path,
        *,
        expected_tree_sha256: str,
    ) -> None:
        self._locator = str(Path(locator).resolve())
        self._expected_tree_sha256 = expected_tree_sha256
        self._loaded: object | None = None
        self._load_lock = Lock()
        self._batch_solution_capability = None
        self._clock_lock = RLock()

    @property
    def bundle_locator(self) -> Path:
        return Path(self._locator)

    @property
    def tree_sha256(self) -> str:
        return self._expected_tree_sha256

    def __deepcopy__(self, memo: dict[int, object]) -> PythonCandidateProgram:
        del memo
        return type(self)(
            self._locator,
            expected_tree_sha256=self._expected_tree_sha256,
        )

    @property
    def supports_batch_solution(self) -> bool:
        # Only read the tiny entrypoint declaration once. Do not rehash model
        # weights on every evaluation; workers reuse the recorded program ID.
        with self._load_lock:
            if self._batch_solution_capability is None:
                locator = self.bundle_locator
                if locator.suffix == ".py":
                    entrypoint = locator
                else:
                    manifest = locator / "candidate_bundle.json" if locator.is_dir() else locator
                    name = json.loads(manifest.read_text())["entrypoint"]
                    path = PurePosixPath(name)
                    if path.is_absolute() or ".." in path.parts:
                        raise PythonCandidateError("batch entrypoint leaves its frozen bundle")
                    entrypoint = manifest.parent.joinpath(*path.parts)
                tree = ast.parse(entrypoint.read_text())
                self._batch_solution_capability = any(
                    isinstance(node, ast.FunctionDef) and node.name == "prepare_solutions"
                    for node in tree.body)
            return self._batch_solution_capability

    def _module(self) -> object:
        with self._load_lock:
            if self._loaded is None:
                self._loaded = _load_python_candidate_module(
                    self.bundle_locator,
                    expected_tree_sha256=self._expected_tree_sha256,
                )
            return self._loaded

    def __getattr__(self, name: str) -> object:
        if name not in {"prepare_solution", "prepare_solutions"}:
            raise AttributeError(name)
        prepare = getattr(self._module(), name, None)
        if not callable(prepare):
            raise AttributeError(name)
        return prepare

    @contextmanager
    def _solve_budget_clock_scope(self):
        """Adapt legacy module-level stopwatch aliases without changing bytes.

        Frozen Candidates historically imported ``time.perf_counter`` for
        cooperative preparation deadlines. Only those exact aliases in this
        Candidate's entry module are rebound, for the duration of execution.
        The process-wide time module, helper modules, and bundle bytes remain
        untouched. New Candidates should import the SDK clock explicitly.
        """
        from pde_agent_verifier_solver.sdk.solve_budget import candidate_solve_clock

        with self._clock_lock:
            module = self._module()
            namespace = vars(module)
            aliases = tuple(name for name, value in namespace.items() if value is perf_counter)
            for name in aliases:
                namespace[name] = candidate_solve_clock
            try:
                yield aliases
            finally:
                for name in aliases:
                    if namespace.get(name) is candidate_solve_clock:
                        namespace[name] = perf_counter


class PythonCandidateCodec:
    """Freeze Agent bundles and translate nominations into Artifacts.

    A nomination contains a workspace-relative ``bundle_path`` and optional
    ``name``, ``summary``, and ``usage`` text. Bundle directories use the
    public ``candidate_bundle.json`` format; one-file candidates also work.
    """

    def __init__(
        self,
        *,
        task_kind: TaskKind,
        limits: CandidateBundleLimits | None = None,
    ) -> None:
        if not isinstance(task_kind, TaskKind):
            raise TypeError("task_kind must be a TaskKind")
        self.task_kind = task_kind
        self.limits = limits or CandidateBundleLimits()

    def decode_nomination(
        self,
        payload: Mapping[str, object],
        context,
        workspace: Path,
    ) -> Artifact:
        path_value = payload.get("bundle_path")
        if not isinstance(path_value, str) or not path_value.strip():
            raise PythonCandidateError("nomination needs a nonempty bundle_path")
        source = _workspace_path(workspace, path_value, label="bundle_path")
        inspected = inspect_candidate_bundle(source, limits=self.limits)
        frozen = freeze_candidate_bundle(
            inspected,
            workspace / ".controller-candidates",
            limits=self.limits,
        )
        program = PythonCandidateProgram(
            frozen.manifest_path,
            expected_tree_sha256=frozen.tree_sha256,
        )
        _validate_python_candidate_entrypoint(
            frozen.entrypoint_path,
            error_message="Python Candidate must export callable prepare_solution",
        )
        return Artifact(
            artifact_id=f"submitted-{frozen.tree_sha256[:20]}",
            kind=self.task_kind,
            name=_optional_text(
                payload,
                "name",
                default=f"Python Candidate {frozen.tree_sha256[:10]}",
            ),
            program=program,
            summary=_optional_text(
                payload,
                "summary",
                default="Agent-submitted Python scientific program.",
            ),
            usage=_optional_text(
                payload,
                "usage",
                default=(
                    "Call prepare_solution(case, solve_budget), then "
                    "handle.evaluate(query)."
                ),
            ),
            round_index=context.round_plan.round_index,
            agent_id=context.agent.agent_id,
            parent_artifact_id=(
                None if context.incumbent is None else context.incumbent.artifact_id
            ),
            metadata={
                "candidate_bundle_tree_sha256": frozen.tree_sha256,
                "candidate_bundle_entrypoint": frozen.entrypoint,
                "candidate_bundle_bytes": frozen.total_bytes,
            },
        )

    def encode_incumbent(
        self,
        incumbent: Artifact,
        workspace: Path,
    ) -> Mapping[str, object]:
        program = incumbent.program
        if not isinstance(program, PythonCandidateProgram):
            raise PythonCandidateError(
                "canonical Python Agent needs a PythonCandidateProgram incumbent"
            )
        source = inspect_candidate_bundle(program.bundle_locator, limits=self.limits,
                                          recorded_identity=program.tree_sha256)
        destination = workspace / "incumbent"
        if destination.exists():
            raise PythonCandidateError("incumbent staging destination already exists")
        shutil.copytree(source.root, destination, symlinks=False)
        staged = inspect_candidate_bundle(destination, limits=self.limits,
                                          recorded_identity=source.tree_sha256)
        return {
            "bundle_path": "incumbent",
            "entrypoint": staged.entrypoint,
            "tree_sha256": staged.tree_sha256,
        }

    def bind_existing(
        self,
        locator: str | Path,
        *,
        storage_root: str | Path,
        artifact_id: str,
        name: str,
        summary: str,
        usage: str,
        round_index: int = 0,
        agent_id: str = "user",
    ) -> Artifact:
        """Create a Controller-frozen S0 from a user-provided bundle."""

        frozen = freeze_candidate_bundle(locator, storage_root, limits=self.limits)
        program = PythonCandidateProgram(
            frozen.manifest_path,
            expected_tree_sha256=frozen.tree_sha256,
        )
        _validate_python_candidate_entrypoint(
            frozen.entrypoint_path,
            error_message=(
                "initial Python Candidate must export callable prepare_solution"
            ),
        )
        return Artifact(
            artifact_id=artifact_id,
            kind=self.task_kind,
            name=name,
            program=program,
            summary=summary,
            usage=usage,
            round_index=round_index,
            agent_id=agent_id,
            metadata={
                "candidate_bundle_tree_sha256": frozen.tree_sha256,
                "candidate_bundle_entrypoint": frozen.entrypoint,
                "candidate_bundle_bytes": frozen.total_bytes,
            },
        )


def _validate_python_candidate_entrypoint(
    entrypoint: Path,
    *,
    error_message: str,
) -> None:
    """Validate the required ABI without importing Candidate-controlled code.

    Binding only establishes the exact source-level function contract. Runtime
    imports and callable resolution stay deferred until the program is actually
    invoked (normally inside the Candidate execution boundary).
    """

    try:
        tree = ast.parse(entrypoint.read_bytes(), filename=str(entrypoint))
    except (OSError, SyntaxError, UnicodeDecodeError, ValueError) as exc:
        error = PythonCandidateError(
            f"Python Candidate entrypoint is not valid Python: {type(exc).__name__}"
        )
        if isinstance(exc, SyntaxError):
            error.candidate_diagnostic = candidate_exception_diagnostic(
                exc, bundle_root=entrypoint.parent,
            )
        raise error from exc
    declarations = [
        (index, node)
        for index, node in enumerate(tree.body)
        if isinstance(node, ast.FunctionDef) and node.name == "prepare_solution"
    ]
    if not declarations:
        raise PythonCandidateError(
            f"{error_message}; expected a module-level synchronous "
            "def prepare_solution(case, solve_budget=None)"
        )
    declaration_index, declaration = declarations[-1]
    if declaration.decorator_list:
        raise PythonCandidateError(
            f"{error_message}; prepare_solution must be undecorated"
        )
    if not _is_prepare_solution_signature(declaration.args):
        raise PythonCandidateError(
            f"{error_message}; expected exactly "
            "prepare_solution(case, solve_budget=None)"
        )
    if any(
        _statement_rebinds_module_name(statement, "prepare_solution")
        for statement in tree.body[declaration_index + 1 :]
    ):
        raise PythonCandidateError(
            f"{error_message}; prepare_solution cannot be rebound after its declaration"
        )


def _is_prepare_solution_signature(arguments: ast.arguments) -> bool:
    positional = (*arguments.posonlyargs, *arguments.args)
    return (
        [argument.arg for argument in positional] == ["case", "solve_budget"]
        and len(arguments.defaults) == 1
        and isinstance(arguments.defaults[0], ast.Constant)
        and arguments.defaults[0].value is None
        and not arguments.kwonlyargs
        and not arguments.kw_defaults
        and arguments.vararg is None
        and arguments.kwarg is None
    )


def _statement_rebinds_module_name(statement: ast.stmt, name: str) -> bool:
    """Return whether a later module statement can replace the ABI binding."""

    visitor = _ModuleBindingVisitor(name)
    visitor.visit(statement)
    return visitor.found


class _ModuleBindingVisitor(ast.NodeVisitor):
    """Find bindings executed in module scope without entering nested scopes."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.found = False

    def visit_Name(self, node: ast.Name) -> None:
        if node.id == self.name and isinstance(node.ctx, (ast.Store, ast.Del)):
            self.found = True

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name == self.name:
            self.found = True

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if node.name == self.name:
            self.found = True

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if node.name == self.name:
            self.found = True

    def visit_Import(self, node: ast.Import) -> None:
        if any(
            (alias.asname or alias.name.split(".", 1)[0]) == self.name
            for alias in node.names
        ):
            self.found = True

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if any(
            alias.name == "*" or (alias.asname or alias.name) == self.name
            for alias in node.names
        ):
            self.found = True

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name == self.name:
            self.found = True
        self.generic_visit(node)

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        if node.name == self.name:
            self.found = True
        self.generic_visit(node)

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        if node.name == self.name:
            self.found = True

    def visit_MatchMapping(self, node: ast.MatchMapping) -> None:
        if node.rest == self.name:
            self.found = True
        self.generic_visit(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        del node

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node, node.elt)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node, node.elt)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension(node, node.elt)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node, node.key, node.value)

    def _visit_comprehension(
        self,
        node: ast.ListComp | ast.SetComp | ast.GeneratorExp | ast.DictComp,
        *values: ast.expr,
    ) -> None:
        # Comprehension iteration targets live in an implicit nested scope, but
        # assignment expressions bind in the containing module scope.
        for generator in node.generators:
            self.visit(generator.iter)
            for condition in generator.ifs:
                self.visit(condition)
        for value in values:
            self.visit(value)


def _load_python_candidate_module(
    locator: Path,
    *,
    expected_tree_sha256: str,
) -> object:
    bundle = inspect_candidate_bundle(locator, recorded_identity=expected_tree_sha256)
    entrypoint = bundle.entrypoint_path
    module_name = f"_canonical_candidate_{bundle.tree_sha256}_{secrets.token_hex(16)}"
    try:
        spec = importlib.util.spec_from_file_location(
            module_name,
            entrypoint,
            submodule_search_locations=[str(entrypoint.parent)],
        )
        if spec is None or spec.loader is None:
            raise ImportError("Candidate entrypoint has no Python loader")
        module = importlib.util.module_from_spec(spec)
    except BaseException as exc:
        raise PythonCandidateError(
            f"Python Candidate could not be prepared: {type(exc).__name__}"
        ) from exc

    local_top_levels = {
        member.path.split("/", 1)[0].removesuffix(".py")
        for member in bundle.members
        if member.path.endswith(".py")
    }
    with _IMPORT_LOCK:
        original_path = list(sys.path)
        original_bytecode = sys.dont_write_bytecode
        saved_modules = {
            name: sys.modules.pop(name)
            for name in tuple(sys.modules)
            if name.split(".", 1)[0] in local_top_levels
        }
        sys.modules[module_name] = module
        sys.path[:0] = list(dict.fromkeys((str(entrypoint.parent), str(bundle.root))))
        sys.dont_write_bytecode = True
        try:
            spec.loader.exec_module(module)
        except (KeyboardInterrupt, GeneratorExit):
            _discard_candidate_imports(module_name, local_top_levels, saved_modules)
            raise
        except BaseException as exc:
            _discard_candidate_imports(module_name, local_top_levels, saved_modules)
            error = PythonCandidateError(
                f"Python Candidate could not be loaded: {type(exc).__name__}: {exc}"
            )
            error.candidate_diagnostic = candidate_exception_diagnostic(
                exc, bundle_root=bundle.root,
            )
            raise error from exc
        finally:
            sys.path[:] = original_path
            sys.dont_write_bytecode = original_bytecode
        _discard_candidate_imports(module_name, local_top_levels, saved_modules)
    return module


def _discard_candidate_imports(
    module_name: str,
    local_top_levels: set[str],
    saved_modules: Mapping[str, object],
) -> None:
    for name in tuple(sys.modules):
        if (
            name == module_name
            or name.startswith(module_name + ".")
            or name.split(".", 1)[0] in local_top_levels
        ):
            sys.modules.pop(name, None)
    sys.modules.update(saved_modules)


def _workspace_path(workspace: Path, raw: str, *, label: str) -> Path:
    pure = PurePosixPath(raw)
    if pure.is_absolute() or not pure.parts or any(
        part in {"", ".", ".."} for part in pure.parts
    ):
        raise PythonCandidateError(f"{label} must be a safe relative path")
    resolved_workspace = workspace.resolve()
    resolved = workspace.joinpath(*pure.parts).resolve()
    if resolved != resolved_workspace and resolved_workspace not in resolved.parents:
        raise PythonCandidateError(f"{label} escaped the Agent workspace")
    return resolved


def _optional_text(
    payload: Mapping[str, object],
    name: str,
    *,
    default: str,
) -> str:
    value = payload.get(name, default)
    if not isinstance(value, str) or not value.strip():
        raise PythonCandidateError(f"{name} must be nonempty text")
    if len(value) > 4096:
        raise PythonCandidateError(f"{name} is too long")
    return value.strip()


__all__ = [
    "PythonCandidateCodec",
    "PythonCandidateError",
    "PythonCandidateProgram",
]
