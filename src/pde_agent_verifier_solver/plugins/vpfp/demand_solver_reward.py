"""One frozen VPFP solver call with on-demand, public numerical queries.

The controller retains the continuous answer and the actual public session.
Only a clean Case and a numerical pipe proxy cross the spawn boundary. This
is a trusted-code experiment, not a filesystem/security sandbox. The existing
SDK session and four-macro GRID output contract are unchanged. An explicit
legacy adapter can lazily supply arrays; it does not promise fewer queries
when an old solver asks for the entire source array.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from hashlib import sha256
import inspect
import io
import json
import math
import multiprocessing
from pathlib import Path
from time import perf_counter
from types import ModuleType

import numpy as np

from pde_agent_verifier_solver.sdk import (
    AnswerQuery, Case, CaseSource, CaseVisibility, open_public_case_session,
)
from pde_agent_verifier_solver.sdk.public_sessions import bind_public_session_factory
from .frozen_solver_reward import (
    ANONYMOUS_CASE_ID, PUBLIC_SHAPE, FrozenSolver, load_frozen_solver,
    _question, _axes, _array, _array_sha, _point_chunks, _validate_view,
    _NonfinitePrediction,
)
from .macro_matching import VPFP_MACRO_FIELDS
from .public_problem import VPFP_MAX_ORACLE_POINTS_PER_CALL, VPFP_OPERATOR_ID

SCHEMA = "vpfp_demand_solver_reward_v1"
METHOD_WIDTHS = {"rho0": 2, "g0": 4, "source": 5}
LEGACY_CHUNK_SIZE = 8192


def build_demand_public_case(case, *, legacy_array_adapter=False):
    """Copy only small physical inputs; never inspect source/g0/rho0 values.

    Axes describe this experiment's fixed OUTPUT query, not an Oracle query
    restriction. Native candidates may query any allowed physical points.
    Public initial J/K and quadrature weights are copied only for the explicit
    old-array adapter. Its three large/data-oracle keys are added in the child.
    """
    if not isinstance(case, Case) or type(legacy_array_adapter) is not bool:
        raise TypeError("a public Case and explicit boolean legacy adapter are required")
    question = _question(case.public_data["question"])
    arrays = _axes(case.public_data["arrays"])
    question.update(equation_version=VPFP_OPERATOR_ID,
        time_domain=[float(arrays["t"][0]), float(arrays["t"][-1])],
        x_domain=[float((arrays[name][1]-arrays[name][0])*len(arrays[name])) for name in ("x", "y")],
        velocity_domain="R2")
    if legacy_array_adapter:
        _, nx, ny, nvx, nvy = PUBLIC_SHAPE
        original = case.public_data["arrays"]
        for name, shape in {"velocity_weights": (nvx, nvy),
                "momentum0": (nx, ny, 2), "kinetic_energy0": (nx, ny)}.items():
            arrays[name] = _array(original[name], shape, name)
        if np.any(arrays["velocity_weights"] <= 0):
            raise ValueError("legacy public quadrature weights must be positive")
    return Case(case_id=ANONYMOUS_CASE_ID, source=CaseSource.PUBLIC,
        visibility=CaseVisibility.HIDDEN, public_data={"question": question, "arrays": arrays},
        parent_case_id=None, round_index=None, tags=())


def _light_identity(case):
    arrays = {name: _array_sha(value) for name, value in case.public_data["arrays"].items()}
    body = {"question": case.public_data["question"], "arrays": arrays}
    return {"public_input_sha256": sha256(json.dumps(body, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
        "array_sha256": arrays, "public_question": dict(body["question"]),
        "public_input_identity_scope": "small_initial_public_inputs; requested_values_bound_separately",
        "array_shape_t_x_y_vx_vy": list(PUBLIC_SHAPE)}


def _points(value, method):
    raw = np.asarray(value)
    if (method not in METHOD_WIDTHS or raw.dtype.kind not in "fiu"
            or raw.ndim != 2 or raw.shape[1] != METHOD_WIDTHS[method]
            or not 1 <= len(raw) <= VPFP_MAX_ORACLE_POINTS_PER_CALL):
        raise ValueError("Oracle request requires an allowlisted finite [N,D] point array within the point limit")
    return _array(raw, raw.shape, method+"_points")


class _PublicOracleError(RuntimeError):
    """A numerical public-input service failure, never a solver-error label."""


class _PipePublicSession:
    """Child-only numerical proxy, with no reference to the controller object."""
    def __init__(self, connection):
        self.connection, self.next_id, self.closed = connection, 0, False

    def _call(self, method, points):
        if self.closed:
            raise _PublicOracleError("public numerical session is closed")
        # Parent validates independently; sending bad numeric requests still
        # preserves a failed-request receipt in the controller.
        request_id = self.next_id
        self.next_id += 1
        self.connection.send({"kind": "oracle_request", "request_id": request_id,
            "method": method, "points": points})
        response = self.connection.recv()
        if (not isinstance(response, dict) or response.get("kind") != "oracle_response"
                or response.get("request_id") != request_id):
            raise _PublicOracleError("public numerical response identity differs")
        if response.get("ok") is not True:
            raise _PublicOracleError("public numerical request failed; see controller receipt")
        return _array(response["values"], (len(points),), method)

    def rho0(self, points_xy):
        return self._call("rho0", points_xy)

    def g0(self, points_xyv):
        return self._call("g0", points_xyv)

    def source(self, points):
        return self._call("source", points)

    def close(self):
        # The SDK revokes the proxy; the worker owns the pipe lifetime.
        self.closed = True


class _LazyPublicArrays(Mapping):
    """Compatibility only: iter/len never evaluate or convert lazy arrays."""
    def __init__(self, small_arrays):
        self.cache = dict(small_arrays)
        self.names = tuple(self.cache) + tuple(METHOD_WIDTHS)
        self.case = None

    def __len__(self):
        return len(self.names)

    def __iter__(self):
        return iter(self.names)

    def __contains__(self, name):
        return name in self.names

    def __getitem__(self, name):
        if name in self.cache:
            return self.cache[name]
        if name not in METHOD_WIDTHS:
            raise KeyError(name)
        session = open_public_case_session(self.case)
        axis_names = {"rho0": ("x", "y"), "g0": ("x", "y", "vx", "vy"),
            "source": ("t", "x", "y", "vx", "vy")}[name]
        axes = tuple(self.cache[key] for key in axis_names)
        shape = tuple(map(len, axes))
        result = np.empty(math.prod(shape), dtype=np.float64)
        operation = getattr(session, name)
        for start, points in _point_chunks(axes, LEGACY_CHUNK_SIZE):
            result[start:start+len(points)] = operation(points)
        value = _array(result.reshape(shape), shape, name)
        if (name == "rho0" and np.any(value <= 0)) or (name == "g0" and np.any(value < 0)):
            raise _PublicOracleError("public initial density violates its numerical input contract")
        self.cache[name] = value
        return value


def _worker(connection, source_bytes, source_name, expected_sha, public_case,
        query, budget, legacy_array_adapter, attempted_event):
    stdout, stderr = io.StringIO(), io.StringIO()
    phase = "solver_setup"
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            case = public_case
            if legacy_array_adapter:
                arrays = _LazyPublicArrays(case.public_data["arrays"])
                case = replace(case, public_data={"question": dict(case.public_data["question"]), "arrays": arrays})
                arrays.case = case
            module = ModuleType("demand_vpfp_solver_"+expected_sha)
            module.__file__ = source_name
            exec(compile(source_bytes, source_name, "exec"), module.__dict__)
            predictor = getattr(module, "predict_macros", None)
            if not callable(predictor):
                raise TypeError("frozen solver has no predict_macros entrypoint")
            inspect.signature(predictor).bind(case, query, budget=budget)
            proxy = _PipePublicSession(connection)
            with bind_public_session_factory(lambda _case: proxy, case,
                    allowed_methods=tuple(METHOD_WIDTHS)):
                phase = "solver_prediction"
                attempted_event.set()
                prediction = predictor(case, query, budget=budget)
                phase = "solver_output_contract"
                _validate_view(prediction, query, public_case.public_data["arrays"])
        connection.send({"kind": "result", "prediction": prediction, "error": None,
            "failure_origin": None, "validity_label_available": False,
            "stdout": stdout.getvalue()[-8192:], "stderr": stderr.getvalue()[-8192:]})
    except BaseException as exc:
        numeric = ((phase == "solver_prediction" and isinstance(exc, (ArithmeticError, np.linalg.LinAlgError)))
            or (phase == "solver_output_contract" and isinstance(exc, _NonfinitePrediction)))
        origin = "public_oracle" if isinstance(exc, _PublicOracleError) else (
            "solver_numerical_failure" if numeric else phase)
        try:
            connection.send({"kind": "result", "prediction": None,
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "failure_origin": origin, "validity_label_available": numeric,
                "stdout": stdout.getvalue()[-8192:], "stderr": stderr.getvalue()[-8192:]})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


def run_demand_solver(asset, case, query, *, public_session, budget_seconds=180.,
        legacy_array_adapter=False, check=lambda: None):
    """Run one fixed solver; parent services only its actual public requests.

    The active-wall budget includes child startup/solver/IPC/validation, minus
    wall time spent inside the parent's Oracle backend callback. This is not
    process CPU time: transport and synchronization outside that callback are
    still charged. The caller's allocation check runs
    before/after each request and while waiting; no extra sampling occurs.
    The caller owns and closes public_session. Only this spawned process may
    be terminated on failure. No reward is assigned here.
    """
    started = perf_counter()
    process = parent_pipe = child_pipe = attempted = None
    paused_seconds = 0.0
    allocation_error = None
    usage = {name: {"calls": 0, "points": 0, "evaluated_points": 0,
        "completed_points": 0, "backend_seconds": 0., "failures": 0}
        for name in METHOD_WIDTHS}
    requests = []
    receipt = {"schema": SCHEMA, "solver_sha256": getattr(asset, "sha256", None),
        "execution_assurance": "trusted_frozen_code_spawn_public_arguments_and_numerical_proxy_not_formal_sandbox",
        "solver_received_hidden_truth": False, "solver_received_origin_labels": False,
        "solver_received_controller_oracle_object": False, "public_oracle_methods": list(METHOD_WIDTHS),
        "legacy_array_adapter": legacy_array_adapter, "prematerialized_points": 0,
        "prematerialized_source_or_g0": False, "oracle_usage": usage, "oracle_requests": requests,
        "solver_prediction_attempted": False, "validity_label_available": False,
        "failure_origin": None, "reward_eligible": False, "reward": None,
        "oracle_points_are_agent_selected": True, "output_scope": "existing_four_macro_GRID_only",
        "point_accounting": "points=requested; evaluated_points=entered_backend; completed_points=finite_shape_valid_backend_result",
        "scientific_admission": False,
        "budget_policy": "elapsed_wall_minus_parent_backend_callback_wall; startup_validation_IPC_charged; not_process_CPU_time; allocation_check_separate"}

    def allocation_check():
        nonlocal allocation_error
        try:
            check()
        except Exception as exc:
            allocation_error = exc
            raise

    def finish(stage, *, prediction=None, error=None, eligible=False, origin=None):
        wall = perf_counter()-started
        receipt.update(stage=stage, execution_passed=prediction is not None,
            error=error, validity_label_available=bool(eligible), failure_origin=origin,
            solver_prediction_attempted=attempted is not None and attempted.is_set(),
            wall_seconds=wall, backend_seconds=sum(item["backend_seconds"] for item in usage.values()),
            oracle_wait_seconds=paused_seconds, solver_active_wall_seconds=max(0., wall-paused_seconds),
            requested_points=sum(item["points"] for item in usage.values()),
            evaluated_points=sum(item["evaluated_points"] for item in usage.values()),
            completed_points=sum(item["completed_points"] for item in usage.values()),
            request_count=len(requests))
        if prediction is not None:
            receipt["reward_pending_verifier_comparison"] = True
        return prediction, receipt

    def error_dict(exc):
        return {"type": type(exc).__name__, "message": str(exc)}

    def active_remaining():
        return float(budget_seconds)-(perf_counter()-started-paused_seconds)

    def deadline(message=None):
        load_frozen_solver(asset.path, asset.sha256)
        entered = attempted is not None and attempted.is_set()
        eligible = entered and (message is None or message["error"] is None
            or message["validity_label_available"] is True)
        return finish("solver_timeout", error={"type": "TimeoutError", "message": "frozen solver active-wall deadline exceeded"},
            eligible=eligible, origin="solver_prediction_deadline" if eligible else "setup_or_unclassified_deadline")

    try:
        if (isinstance(budget_seconds, bool) or not isinstance(budget_seconds, (int, float))
                or not math.isfinite(float(budget_seconds)) or budget_seconds <= 0
                or type(legacy_array_adapter) is not bool or not callable(check)):
            raise ValueError("positive finite active-wall budget and explicit adapter/check required")
        if not isinstance(asset, FrozenSolver):
            raise TypeError("a verified FrozenSolver asset is required")
        if not all(callable(getattr(public_session, name, None)) for name in METHOD_WIDTHS):
            raise TypeError("the current case public session must expose rho0/g0/source")
        allocation_check()
        load_frozen_solver(asset.path, asset.sha256)
        source_bytes = Path(asset.path).read_bytes()
        clean = build_demand_public_case(case, legacy_array_adapter=legacy_array_adapter)
        axes = clean.public_data["arrays"]
        expected_query = AnswerQuery.grid(fields=VPFP_MACRO_FIELDS,
            **{name: len(axes[name]) for name in ("t", "x", "y")})
        if query != expected_query:
            raise ValueError("this experiment retains its registered macro GRID output query")
        receipt.update(_light_identity(clean), budget_seconds=float(budget_seconds))
        context = multiprocessing.get_context("spawn")
        parent_pipe, child_pipe = context.Pipe(duplex=True)
        attempted = context.Event()
        process = context.Process(target=_worker, args=(child_pipe, source_bytes, asset.path,
            asset.sha256, clean, query, float(budget_seconds), legacy_array_adapter, attempted), daemon=True)
        process.start()
        child_pipe.close()
        next_request = 0
        while True:
            allocation_check()
            remaining = active_remaining()
            if remaining <= 0:
                return deadline()
            if not parent_pipe.poll(min(.05, remaining)):
                if not process.is_alive():
                    raise RuntimeError("solver worker exited without a final numerical response")
                continue
            message = parent_pipe.recv()
            if not isinstance(message, dict):
                raise ValueError("malformed numerical worker message")
            if message.get("kind") == "oracle_request":
                method = message.get("method")
                unit = {"request_id": message.get("request_id"), "method": method,
                    "points": 0, "input_sha256": None, "output_sha256": None,
                    "status": "received", "backend_seconds": 0.}
                requests.append(unit)
                counted = type(method) is str and method in usage
                if counted:
                    usage[method]["calls"] += 1
                try:
                    if (set(message) != {"kind", "request_id", "method", "points"}
                            or type(message["request_id"]) is not int
                            or message["request_id"] != next_request or not counted):
                        raise ValueError("unknown, repeated or non-allowlisted public numerical request")
                    next_request += 1
                    raw = np.asarray(message["points"])
                    if raw.dtype.kind in "fiu":
                        unit["input_sha256"] = _array_sha(raw)
                    if raw.ndim > 0:
                        unit["points"] = len(raw)
                        usage[method]["points"] += len(raw)
                    points = _points(raw, method)
                    allocation_check()
                    backend_start = perf_counter()
                    usage[method]["evaluated_points"] += len(points)
                    try:
                        values = getattr(public_session, method)(points)
                    finally:
                        unit["backend_seconds"] = perf_counter()-backend_start
                        usage[method]["backend_seconds"] += unit["backend_seconds"]
                        paused_seconds += unit["backend_seconds"]
                    # Hash returned numerical bytes even when nonfinite or
                    # wrongly shaped, then retain the failed validation.
                    returned = np.asarray(values)
                    if returned.dtype.kind in "fiu":
                        unit["output_sha256"] = _array_sha(returned)
                    result = _array(returned, (len(points),), method)
                    usage[method]["completed_points"] += len(points)
                    allocation_check()
                    unit["status"] = "completed"
                    parent_pipe.send({"kind": "oracle_response", "request_id": message["request_id"],
                        "ok": True, "values": result})
                except Exception as exc:
                    unit.update(status="failed", error=error_dict(exc))
                    if counted:
                        usage[method]["failures"] += 1
                    if exc is allocation_error:
                        raise
                    # Do not let a candidate reinterpret an Oracle failure
                    # as a difficult generated equation or catch and hide it.
                    return finish("oracle_failed", error=error_dict(exc), origin="public_oracle")
                continue
            if message.get("kind") != "result" or set(message) != {
                    "kind", "prediction", "error", "failure_origin", "validity_label_available", "stdout", "stderr"}:
                raise ValueError("unexpected numerical worker response")
            while process.is_alive():
                allocation_check()
                remaining = active_remaining()
                if remaining <= 0:
                    return deadline(message)
                process.join(timeout=min(.05, remaining))
            receipt.update(child_exitcode=process.exitcode,
                stdout_tail=message["stdout"], stderr_tail=message["stderr"])
            if active_remaining() <= 0:
                return deadline(message)
            load_frozen_solver(asset.path, asset.sha256)
            if message["error"] is not None or process.exitcode != 0:
                eligible = (process.exitcode == 0 and attempted.is_set()
                    and message["validity_label_available"] is True)
                return finish("solver_failed", error=message["error"], eligible=eligible,
                    origin=message["failure_origin"] if process.exitcode == 0 else "worker_process_exit")
            prediction = message["prediction"]
            _validate_view(prediction, query, axes)
            return finish("solver_completed", prediction=prediction)
    except Exception as exc:
        result = finish("allocation_interrupted" if exc is allocation_error else "solver_failed",
            error=error_dict(exc), origin="allocation_check" if exc is allocation_error else "parent_verification_or_setup")
        if exc is allocation_error:
            # Preserve the exact exception identity used by the owning rollout
            # to distinguish a paused allocation from a completed failed draw.
            exc.demand_solver_receipt = result[1]
            raise
        return result
    finally:
        if process is not None and process.pid is not None:
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.)
            if process.is_alive():
                process.kill()
                process.join(timeout=1.)
        for connection in (parent_pipe, child_pipe):
            if connection is not None:
                connection.close()


__all__ = ["SCHEMA", "build_demand_public_case", "run_demand_solver"]
