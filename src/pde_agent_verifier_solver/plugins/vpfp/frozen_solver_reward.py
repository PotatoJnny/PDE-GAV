"""Frozen, public-input strong VPFP solver calls and verifier-only rewards.

This is a bounded guidance experiment, not a new solver or an isolation claim.
The solver receives one anonymous numerical public Case in a fresh spawned
process, never a runtime, truth provider, oracle, parent label, or old score.
Only the controller calls ``score_solver_prediction`` after prediction returns.
The historical solver's empirical corrections are preserved byte-for-byte.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from hashlib import sha256
import io
import inspect
import json
import math
import multiprocessing
from pathlib import Path
import re
from time import perf_counter
from types import ModuleType
from typing import Mapping

import numpy as np

from pde_agent_verifier_solver.sdk import (
    AnswerQuery, AnswerView, Case, CaseSource, CaseVisibility, QueryKind,
)
from .macro_matching import VPFP_MACRO_FIELDS, VPFPMacroMatcher


SCHEMA = "vpfp_frozen_solver_reward_v1"
PUBLIC_ARRAY_KEYS = (
    "t", "x", "y", "vx", "vy", "velocity_weights", "rho0", "g0",
    "source", "momentum0", "kinetic_energy0",
)
PUBLIC_QUESTION_KEYS = ("collision_nu", "temperature", "background_density")
PUBLIC_SHAPE = (13, 33, 33, 9, 9)  # t, x, y, vx, vy; not a hidden grid.
ANONYMOUS_CASE_ID = "frozen-strong-public-question"


@dataclass(frozen=True)
class FrozenSolver:
    path: str
    sha256: str


def load_frozen_solver(path, expected_sha256: str) -> FrozenSolver:
    """Check actual source bytes; do not import or execute the solver here."""
    if not isinstance(expected_sha256, str) or re.fullmatch(
        r"[0-9a-f]{64}", expected_sha256
    ) is None:
        raise ValueError("solver requires an explicit SHA256")
    raw = Path(path)
    if raw.is_symlink() or not raw.is_file():
        raise ValueError("solver must be an existing non-symlink source file")
    payload = raw.read_bytes()
    compile(payload, str(raw), "exec")  # syntax only; execution is child-only.
    return FrozenSolver(str(raw.resolve()), expected_sha256)


def _array(value, shape, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu" or raw.shape != tuple(shape):
        raise ValueError(f"{name} must be a real numeric array of shape {shape}")
    result = np.array(raw, dtype=np.float64, copy=True, order="C")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains nonfinite values")
    result.flags.writeable = False
    return result


def _question(value):
    if not isinstance(value, Mapping):
        raise TypeError("public question must be a mapping")
    result = {}
    for name in PUBLIC_QUESTION_KEYS:
        raw = value[name]
        if isinstance(raw, (bool, np.bool_)) or not isinstance(
            raw, (int, float, np.integer, np.floating)
        ):
            raise ValueError(f"public {name} must be a real scalar")
        number = float(raw)
        if not math.isfinite(number) or number <= 0:
            raise ValueError(f"public {name} must be finite and positive")
        result[name] = number
    if not 10.0 <= result["collision_nu"] <= 30.0:
        raise ValueError("frozen historical strong solver scope is nu in [10,30]")
    # Native equation conditions are FP32; admit its exact representation of
    # the stated endpoint, not a generic fuzzy extension of the regime.
    if not 0.8 <= result["temperature"] <= float(np.float32(1.2)):
        raise ValueError("frozen historical strong solver scope is T in [.8,1.2]")
    return result


def _axes(arrays):
    axes = {
        name: _array(arrays[name], (count,), name)
        for name, count in zip(("t", "x", "y", "vx", "vy"), PUBLIC_SHAPE)
    }
    for name, axis in axes.items():
        if not np.all(np.diff(axis) > 0):
            raise ValueError(f"{name} must be strictly increasing")
        # Only a numerical uniform-grid check, never physical acceptance.
        if not np.allclose(np.diff(axis), axis[1]-axis[0], rtol=1e-12, atol=0):
            raise ValueError(f"historical solver requires a uniform {name} axis")
    if axes["t"][0] != 0.0 or axes["t"][-1] != 1.0:
        raise ValueError("frozen solver time domain must be exactly [0,1]")
    for name in ("x", "y"):
        if axes[name][0] != 0:
            raise ValueError("frozen solver spatial axes must start at zero")
    for name in ("vx", "vy"):
        if not np.array_equal(axes[name], np.linspace(-5.0, 5.0, 9)):
            raise ValueError("frozen solver velocity grid must be 9 points on [-5,5]")
    return axes


def build_solver_public_case(*, question, arrays) -> Case:
    """Copy only the declared physical inputs; discard all origin metadata.

    Parent/source labels are never supplied, including for guided candidates.
    Public initial momentum and energy are part of the existing VPFP ABI; they
    are initial data, not future hidden targets. Input arrays are never mutated.
    """
    if not isinstance(arrays, Mapping):
        raise TypeError("public arrays must be a mapping")
    q = _question(question)
    data = _axes(arrays)
    nt, nx, ny, nvx, nvy = PUBLIC_SHAPE
    shapes = {
        "velocity_weights": (nvx, nvy), "rho0": (nx, ny),
        "g0": (nx, ny, nvx, nvy), "source": (nt, nx, ny, nvx, nvy),
        "momentum0": (nx, ny, 2), "kinetic_energy0": (nx, ny),
    }
    data.update({name: _array(arrays[name], shape, name)
                 for name, shape in shapes.items()})
    if np.any(data["rho0"] <= 0) or np.any(data["g0"] < 0):
        raise ValueError("public initial density must be positive and g0 nonnegative")
    if np.any(data["velocity_weights"] <= 0):
        raise ValueError("public quadrature weights must be positive")
    return Case(
        case_id=ANONYMOUS_CASE_ID, source=CaseSource.PUBLIC,
        visibility=CaseVisibility.HIDDEN,  # no development-solution capability.
        public_data={"question": q, "arrays": data},
        parent_case_id=None, round_index=None, tags=(),
    )


def _array_sha(value):
    array = np.ascontiguousarray(value, dtype=np.float64)
    digest = sha256()
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(b"float64-C\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def public_case_receipt(case):
    """Data identity is physical inputs only, not an ancestry substitute."""
    arrays = case.public_data["arrays"]
    identities = {name: _array_sha(arrays[name]) for name in PUBLIC_ARRAY_KEYS}
    body = {"question": dict(case.public_data["question"]), "arrays": identities}
    return {
        "public_input_sha256": sha256(json.dumps(
            body, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()).hexdigest(),
        "array_sha256": identities,
        "public_question": body["question"],
        "array_shape_t_x_y_vx_vy": list(PUBLIC_SHAPE),
        "solver_received_hidden_truth": False,
        "solver_received_origin_labels": False,
        "numerical_resolution_qualified": False,
    }


def _point_chunks(axes, chunk_size):
    shape = tuple(len(axis) for axis in axes)
    count = math.prod(shape)
    for start in range(0, count, chunk_size):
        indices = np.unravel_index(np.arange(start, min(start+chunk_size, count)), shape)
        yield start, np.column_stack([axis[index] for axis, index in zip(axes, indices)])


def materialize_public_case(case: Case, public_session, *, chunk_size=8192):
    """Sample public rho0/g0/source, then return (anonymous Case, receipt).

    ``public_session`` is the issued PUBLIC_PROBLEM session, obtained from
    ``backend.public_session_factory(case)``. It is used only by this parent
    materializer and is never serialized to the solver worker. No completed-
    solution method is called. The caller owns and closes its session.
    """
    if not isinstance(case, Case):
        raise TypeError("materialization requires an SDK public Case, not CaseRecord")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    callbacks = {name: getattr(public_session, name, None) for name in ("rho0", "g0", "source")}
    if not all(callable(value) for value in callbacks.values()):
        raise TypeError("public session must implement rho0, g0, source")
    question = _question(case.public_data["question"])
    original = case.public_data["arrays"]
    axes = _axes(original)
    data = dict(axes)
    for name in ("velocity_weights", "momentum0", "kinetic_energy0"):
        data[name] = original[name]
    requests = {
        "rho0": (axes["x"], axes["y"]),
        "g0": (axes["x"], axes["y"], axes["vx"], axes["vy"]),
        "source": tuple(axes[name] for name in ("t", "x", "y", "vx", "vy")),
    }
    usage = {}
    started = perf_counter()
    for name, coordinates in requests.items():
        shape = tuple(len(axis) for axis in coordinates)
        result = np.empty(math.prod(shape), dtype=np.float64)
        calls = 0
        for start, points in _point_chunks(coordinates, chunk_size):
            values = _array(callbacks[name](points), (len(points),), name)
            result[start:start+len(points)] = values
            calls += 1
        data[name] = result.reshape(shape)
        usage[name] = {"calls": calls, "points": int(result.size)}
    sanitized = build_solver_public_case(question=question, arrays=data)
    receipt = public_case_receipt(sanitized)
    receipt.update({
        "schema": SCHEMA, "stage": "public_materialized", "execution_passed": True,
        "oracle_usage": usage, "oracle_methods": ["rho0", "g0", "source"],
        "wall_seconds": perf_counter()-started,
        "initial_momentum_and_energy": "existing_public_initial_macro_arrays",
        "source_and_initial_density": "fresh_public_numerical_oracle_values",
    })
    return sanitized, receipt


def native_macro_query(case: Case) -> AnswerQuery:
    axes = _axes(case.public_data["arrays"])
    return AnswerQuery.grid(fields=VPFP_MACRO_FIELDS,
                            **{name: len(axes[name]) for name in ("t", "x", "y")})


class _NonfinitePrediction(ValueError):
    """Numerical output failure after the entire result ABI was checked."""


def _validate_view(view, query, axes):
    if not isinstance(view, AnswerView) or view.kind is not QueryKind.GRID:
        raise ValueError("solver must return an SDK GRID AnswerView")
    if query.kind is not QueryKind.GRID or tuple(query.resolution) != ("t", "x", "y"):
        raise ValueError("frozen solver requires native t/x/y GRID query")
    if (len(query.fields) != len(VPFP_MACRO_FIELDS)
            or set(query.fields) != set(VPFP_MACRO_FIELDS)
            or set(view.fields) != set(VPFP_MACRO_FIELDS)):
        raise ValueError("solver must return exactly the four VPFP macro fields")
    shape = tuple(query.resolution[name] for name in ("t", "x", "y"))
    if tuple(view.shape) != shape or tuple(view.axes) != ("t", "x", "y"):
        raise ValueError("solver output shape/axis order differs from query")
    for name in ("t", "x", "y"):
        if not np.array_equal(np.asarray(view.axes[name]), np.asarray(axes[name])):
            raise ValueError(f"solver output {name} coordinates differ from public query")
    # Check every field shape/type before identifying any numerical failure;
    # malformed APIs must not become negative solver-validity labels.
    for name in VPFP_MACRO_FIELDS:
        raw = np.asarray(view.fields[name])
        if raw.dtype.kind not in "fiu" or raw.shape != (math.prod(shape),):
            raise ValueError(f"solver output {name} numeric shape differs from query")
    if view.points:
        raise ValueError("GRID solver output must not contain unrelated points")
    for name in VPFP_MACRO_FIELDS:
        if not np.isfinite(np.asarray(view.fields[name], dtype=np.float64)).all():
            raise _NonfinitePrediction(f"solver output {name} contains nonfinite values")


def _worker(connection, source_bytes, source_name, expected_sha, public_case, query, budget, attempted_event):
    """Spawn target: arguments contain no truth provider or public oracle."""
    stdout, stderr = io.StringIO(), io.StringIO()
    phase = "solver_setup"
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            case = build_solver_public_case(**public_case.public_data)
            module = ModuleType("frozen_vpfp_solver_"+expected_sha)
            module.__file__ = source_name
            exec(compile(source_bytes, source_name, "exec"), module.__dict__)
            predictor = getattr(module, "predict_macros", None)
            if not callable(predictor):
                raise TypeError("frozen solver has no predict_macros entrypoint")
            inspect.signature(predictor).bind(case, query, budget=budget)
            # Shared event records actual entry, including when the child never
            # returns. It contains no controller state or truth information.
            phase = "solver_prediction"
            attempted_event.set()
            prediction = predictor(case, query, budget=budget)
            phase = "solver_output_contract"
            _validate_view(prediction, query, case.public_data["arrays"])
        connection.send({"prediction": prediction, "error": None,
                         "failure_origin": None, "validity_label_available": False,
                         "stdout": stdout.getvalue()[-8192:], "stderr": stderr.getvalue()[-8192:]})
    except BaseException as exc:
        numeric = ((phase == "solver_prediction" and isinstance(exc, (ArithmeticError, np.linalg.LinAlgError)))
                   or (phase == "solver_output_contract" and isinstance(exc, _NonfinitePrediction)))
        connection.send({"prediction": None,
                         "error": {"type": type(exc).__name__, "message": str(exc)},
                         "failure_origin": "solver_numerical_failure" if numeric else phase,
                         "validity_label_available": numeric,
                         "stdout": stdout.getvalue()[-8192:], "stderr": stderr.getvalue()[-8192:]})
    finally:
        connection.close()


def _invalid(stage, exc=None, **extra):
    return {"schema": SCHEMA, "stage": stage, "execution_passed": False,
            "reward_eligible": False, "reward": None,
            "error": None if exc is None else {"type": type(exc).__name__, "message": str(exc)},
            **extra}


def run_frozen_solver(asset: FrozenSolver, case: Case, query: AnswerQuery, *, budget_seconds=180.0):
    """Return (prediction or None, finite receipt); never score failures.

    Deadline includes child startup and solve. Only the Process object created
    here may be terminated/killed. Spawn avoids inheriting controller truth
    memory; same-user filesystem access is NOT a formal security sandbox.
    """
    started = perf_counter()
    process = None
    receive = send = attempted_event = None
    receipt = {"solver_sha256": getattr(asset, "sha256", None),
               "execution_assurance": "trusted_frozen_code_spawn_public_arguments_not_formal_sandbox",
               "solver_received_hidden_truth": False, "solver_received_origin_labels": False,
               "solver_prediction_attempted": False, "validity_label_available": False,
               "failure_origin": None,
               "validity_label_policy": "finite_scored_output_or_explicit_numeric_failure_or_prediction_started_deadline; setup_API_unknown_exceptions_excluded"}
    try:
        if isinstance(budget_seconds, bool) or not math.isfinite(float(budget_seconds)) or budget_seconds <= 0:
            raise ValueError("solver deadline must be finite and positive")
        if not isinstance(asset, FrozenSolver):
            raise TypeError("solver asset must be a verified FrozenSolver")
        load_frozen_solver(asset.path, asset.sha256)
        source_bytes = Path(asset.path).read_bytes()
        sanitized = build_solver_public_case(**{
            name: case.public_data[name] for name in ("question", "arrays")
        })
        expected_query = native_macro_query(sanitized)
        if query != expected_query:
            raise ValueError("requested solver query differs from frozen public grid")
        receipt.update(public_case_receipt(sanitized))
        receipt["budget_seconds"] = float(budget_seconds)
        context = multiprocessing.get_context("spawn")
        receive, send = context.Pipe(duplex=False)
        attempted_event = context.Event()
        process = context.Process(target=_worker, args=(send, source_bytes, asset.path,
            asset.sha256, sanitized, query, float(budget_seconds), attempted_event), daemon=True)
        process.start()
        send.close()
        def timeout_receipt(message=None):
            # A setup timeout is not evidence about this generated problem.
            # Recheck asset integrity even if prediction exceeded its deadline.
            load_frozen_solver(asset.path, asset.sha256)
            attempted = attempted_event.is_set()
            eligible = attempted and (message is None or message["error"] is None
                                       or message["validity_label_available"] is True)
            receipt.update(solver_prediction_attempted=attempted,
                validity_label_available=eligible,
                failure_origin="solver_prediction_deadline" if eligible else "setup_or_unclassified_deadline")
            return _invalid("solver_timeout", TimeoutError("frozen solver deadline exceeded"),
                            **receipt, wall_seconds=perf_counter()-started)
        remaining = max(0.0, float(budget_seconds)-(perf_counter()-started))
        if not receive.poll(remaining):
            return None, timeout_receipt()
        message = receive.recv()
        process.join(timeout=max(0.0, float(budget_seconds)-(perf_counter()-started)))
        if process.is_alive() or perf_counter()-started > budget_seconds:
            return None, timeout_receipt(message)
        receipt.update({"child_exitcode": process.exitcode,
                        "solver_prediction_attempted": attempted_event.is_set(),
                        "stdout_tail": message["stdout"], "stderr_tail": message["stderr"]})
        load_frozen_solver(asset.path, asset.sha256)  # failed mutations stay failed.
        if message["error"] is not None or process.exitcode != 0:
            receipt.update(validity_label_available=(process.exitcode == 0
                and receipt["solver_prediction_attempted"] and message["validity_label_available"] is True),
                failure_origin=message["failure_origin"] if process.exitcode == 0 else "worker_process_exit")
            return None, _invalid("solver_failed", **receipt, error=message["error"],
                                  wall_seconds=perf_counter()-started)
        prediction = message["prediction"]
        _validate_view(prediction, query, sanitized.public_data["arrays"])
        return prediction, {"schema": SCHEMA, "stage": "solver_completed",
            "execution_passed": True, "reward_eligible": False, "reward": None,
            "reward_pending_verifier_comparison": True, **receipt,
            "wall_seconds": perf_counter()-started}
    except Exception as exc:
        receipt.update(validity_label_available=False, failure_origin="parent_verification_or_setup",
                       solver_prediction_attempted=attempted_event is not None and attempted_event.is_set())
        return None, _invalid("solver_failed", exc, **receipt, wall_seconds=perf_counter()-started)
    finally:
        if process is not None and process.pid is not None:
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=1.0)
        for connection in (receive, send):
            if connection is not None:
                connection.close()


def score_solver_prediction(prediction, truth_view, query, *, background_density, temperature):
    """Verifier-only fresh macro error, not a reference-distribution reward.

    E = sqrt(mean over four fields of (RMSE / physical scale)**2), with scales
    rho_bg, rho_bg*sqrt(T), rho_bg*sqrt(T), rho_bg*T. Reward is the fixed bounded
    monotone transform log1p(E)/(1+log1p(E)). There is no fitted reward scale,
    relative-truth floor, candidate-derived clipping, or physics-pass threshold.
    The old SDK relative-L2 contract is retained only as a labeled diagnostic.
    """
    try:
        if prediction is None:
            raise ValueError("missing solver prediction cannot receive a reward")
        rho, theta = float(background_density), float(temperature)
        if isinstance(background_density, bool) or isinstance(temperature, bool) or not (
            math.isfinite(rho) and math.isfinite(theta) and rho > 0 and theta > 0
        ):
            raise ValueError("physical reward scales must be finite and positive")
        _validate_view(truth_view, query, truth_view.axes)
        _validate_view(prediction, query, truth_view.axes)
        scale = {"rho": rho, "momentum_x": rho*math.sqrt(theta),
                 "momentum_y": rho*math.sqrt(theta), "kinetic_energy": rho*theta}
        physical, raw, predicted_hashes, truth_hashes = {}, {}, {}, {}
        for name in VPFP_MACRO_FIELDS:
            predicted = np.asarray(prediction.fields[name], dtype=np.float64)
            expected = np.asarray(truth_view.fields[name], dtype=np.float64)
            delta = predicted-expected
            # hypot avoids overflowing the sum of squares of finite errors.
            rms = math.hypot(*delta)/math.sqrt(delta.size)
            value = rms/scale[name]
            if not math.isfinite(value):
                raise ValueError("nonfinite numerical error cannot receive a reward")
            raw[name], physical[name] = rms, value
            predicted_hashes[name], truth_hashes[name] = _array_sha(predicted), _array_sha(expected)
        error = math.hypot(*physical.values())/2.0
        transformed = math.log1p(error)
        reward = transformed/(1.0+transformed)
        matcher = VPFPMacroMatcher()
        bare = Case(ANONYMOUS_CASE_ID, CaseSource.PUBLIC, CaseVisibility.HIDDEN, {})
        target = matcher.prepare_target(bare, truth_view, query)
        try:
            matched = matcher.match(bare, prediction, target)
            compatible = {name: float(value) if math.isfinite(float(value)) else None
                          for name, value in matched.metrics.items()}
            compatible_error = None
        except (ArithmeticError, ValueError) as exc:
            compatible, compatible_error = None, {"type": type(exc).__name__, "message": str(exc)}
        report = {"schema": SCHEMA, "stage": "fresh_solver_error_scored",
            "execution_passed": True, "reward_eligible": True, "reward": reward,
            "physical_scaled_rmse": error, "per_field_rmse_raw_units": raw,
            "per_field_physical_scaled_rmse": physical, "physical_scales": scale,
            "aggregation": "sqrt(mean_of_four_physical_scaled_field_MSEs)",
            "reward_transform": "log1p(E)/(1+log1p(E))",
            "reward_target": "this_call_solver_prediction_error_against_this_generated_hidden_answer",
            "sdk_relative_l2_diagnostic_only": compatible,
            "sdk_relative_l2_diagnostic_error": compatible_error,
            "prediction_macro_sha256": predicted_hashes, "truth_macro_sha256": truth_hashes,
            "physical_regime_qualified": False, "numerical_resolution_qualified": False,
            "old_error_labels_used": False, "reference_similarity_used": False}
        json.dumps(report, allow_nan=False)
        return report
    except Exception as exc:
        return _invalid("solver_error_scoring_failed", exc)


__all__ = ["FrozenSolver", "load_frozen_solver", "build_solver_public_case",
           "materialize_public_case", "public_case_receipt", "native_macro_query",
           "run_frozen_solver", "score_solver_prediction"]
