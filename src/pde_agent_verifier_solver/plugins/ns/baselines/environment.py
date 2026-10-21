"""Paired finite-input / continuous-online NS environments for external Agents.

The starter bank is exported once from DEVELOPMENT records. Both arms restore
exactly those fields; the finite arm receives samples of their public inputs,
never an independently manufactured discrete forcing. v2 additionally exports finite
development reference samples; private validation and test fields remain hidden.
"""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from threading import RLock
from types import SimpleNamespace

import numpy as np

from pde_agent_verifier_solver.engine.common_evaluation import content_sha256, json_value
from pde_agent_verifier_solver.engine.checkpoint_store import atomic_json
from pde_agent_verifier_solver.sdk import (
    CaseAccessKind, CaseRecord, CaseSource, CaseVisibility, GenerationKind,
    GenerationPurpose, GenerationRequest, GeneratorMode, SampleBatch, SampleSource,
)
from ..case_bank import NSBankSessionProvider, _case
from ..continuous_backend import StructuredNSMMSAnswer, _NSVelocityTruth

DEVELOPMENT_SCHEMA = "ns_baseline_continuous_development_v1"
ENVIRONMENT_ARMS = ("fixed_sampled", "continuous_online_base")


def _digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def continuous_field_identity(answer):
    """Scientific identity independent of tensor archive serialization and file names."""
    field = answer.raw_answer
    return content_sha256({"config": field.config.to_dict(), "geometry": answer.public_geometry,
        "abi": field.continuous_mms_abi, "bounds": answer.domain_bounds,
        "time": answer.time_interval, "viscosity": answer.viscosity,
        "psi": field.coefficients.psi.detach().cpu().numpy(),
        "pressure": field.coefficients.pressure.detach().cpu().numpy()})


def sample_public_inputs(answer, *, grid_size=64, forcing_times=32, include_reference=False):
    """Sample continuous MMS public inputs, with float64 cell-centre coordinates."""
    from ..tasks.solver import _inside_polygon, _obstacle_polygons
    for number in (grid_size, forcing_times):
        if type(number) is not int or number < 2:
            raise ValueError("input grid/time counts must be integers >=2")
    x0, x1, y0, y1 = answer.domain_bounds
    x = x0 + (np.arange(grid_size, dtype=np.float64) + .5) * (x1 - x0) / grid_size
    y = y0 + (np.arange(grid_size, dtype=np.float64) + .5) * (y1 - y0) / grid_size
    t = np.linspace(*answer.time_interval, forcing_times, dtype=np.float64)
    xy = np.stack(np.meshgrid(x, y, indexing="ij"), axis=-1).reshape(-1, 2)
    polygons = _obstacle_polygons(answer.public_geometry)
    fluid = np.array([not any(_inside_polygon(a, b, p) for p in polygons) for a, b in xy]).reshape(grid_size, grid_size)
    def at_time(method, when, coordinates):
        points = np.column_stack((np.full(len(coordinates), when), coordinates))
        result = np.concatenate([np.asarray(method(points[i:i+4096]), dtype=np.float64)
                                 for i in range(0, len(points), 4096)])
        if result.shape != (len(points), 2) or not np.isfinite(result).all():
            raise ValueError("public input sampling returned an incomplete/nonfinite field")
        return result
    initial = at_time(answer.evaluate_velocity, t[0], xy).T.reshape(2, grid_size, grid_size)
    forcing = np.stack([at_time(answer.evaluate_forcing, when, xy).T.reshape(2, grid_size, grid_size) for when in t])
    arrays = {"x": x, "y": y, "t": t, "fluid_mask": fluid,
              "initial_velocity": initial, "forcing": forcing}
    # Include finite boundary input samples for manufactured outer traces too.
    outer = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    edges = []
    for polygon in (outer, *polygons):
        for left, right in zip(polygon, (*polygon[1:], polygon[0])):
            edges.extend(np.asarray(left) + fraction * (np.asarray(right) - left)
                         for fraction in np.linspace(0, 1, grid_size, endpoint=False))
    boundary = np.asarray(edges, dtype=np.float64)
    arrays["boundary_xy"] = boundary
    arrays["boundary_velocity"] = np.stack([at_time(answer.evaluate_velocity, when, boundary).T for when in t])
    if include_reference:
        arrays["reference_velocity"] = np.stack([
            at_time(answer.evaluate_velocity, when, xy).T.reshape(2, grid_size, grid_size) for when in t])
    return arrays


def write_development_bank(manifest_path, records, *, provenance, grid_size=64, forcing_times=32, include_reference=False):
    """Export existing development Cases for BOTH arms. Does not train/sample a Generator."""
    from agent_verifier.neural_generator.structured_ns_answer_first import write_structured_continuous_case_state
    path = Path(manifest_path).expanduser().resolve()
    records = tuple(records)
    if (not records or len({r.case.case_id for r in records}) != len(records)
            or any(r.case.visibility is not CaseVisibility.DEVELOPMENT
                   or r.case.source is not CaseSource.GENERATED or r.case.parent_case_id is not None
                   or not isinstance(getattr(r.truth, "_answer", None), StructuredNSMMSAnswer)
                   for r in records)):
        raise ValueError("starter bank requires unique clean continuous DEVELOPMENT Cases")
    assets = path.with_suffix(".cases")
    if path.exists() or assets.exists():
        raise FileExistsError("starter export requires a new manifest and asset directory")
    assets.mkdir(parents=True, mode=0o700)
    rows = []
    for index, record in enumerate(records):
        directory = assets / f"{index:06d}"
        field = write_structured_continuous_case_state(directory, field=record.truth._answer.raw_answer)
        arrays = directory / "public_inputs.npz"
        np.savez(arrays, **sample_public_inputs(record.truth._answer, grid_size=grid_size, forcing_times=forcing_times, include_reference=include_reference))
        rows.append({"case": json_value(record.case), "state_path": str(field.relative_to(path.parent)),
                     "state_sha256": _digest(field), "arrays_path": str(arrays.relative_to(path.parent)),
                     "arrays_sha256": _digest(arrays), "public_case_sha256": content_sha256(record.case),
                     "continuous_mms_abi": record.truth._answer.raw_answer.continuous_mms_abi})
        for item in directory.iterdir():
            item.chmod(0o600)
    payload = {"schema": DEVELOPMENT_SCHEMA, "purpose": "development", "cases": rows,
               "case_count": len(rows), "provenance": dict(provenance),
               "finite_sampling": {"grid_size": grid_size, "forcing_times": forcing_times,
                                   "dtype": "float64", "spatial_nodes": "cell_centres"}}
    if include_reference:
        payload["development_reference"] = "finite_velocity_same_continuous_mms"
    payload["content_sha256"] = content_sha256(payload)
    atomic_json(path, payload)
    return path


def _asset(path, row, name):
    result = (path.parent / row[name + "_path"]).resolve()
    if not result.is_relative_to(path.parent):
        raise ValueError("starter asset leaves its directory: " + name)
    return result


def load_development_bank(manifest_path, *, arm, device="cpu", expected_cases=None, expected_abi=None, require_reference=False):
    from agent_verifier.neural_generator.structured_ns_answer_first import load_structured_continuous_case
    if arm not in ENVIRONMENT_ARMS:
        raise ValueError("unknown paired NS input environment")
    path = Path(manifest_path).expanduser().resolve()
    payload = json.loads(path.read_text())
    if payload.get("schema") != DEVELOPMENT_SCHEMA or payload.get("purpose") != "development":
        raise ValueError("invalid or changed continuous starter manifest")
    has_reference = payload.get("development_reference") == "finite_velocity_same_continuous_mms"
    if require_reference and not has_reference:
        raise ValueError("this protocol requires finite development reference arrays; re-export the starter bank")
    rows = payload["cases"]
    if (not rows or len(rows) != payload["case_count"] or
            (expected_cases is not None and len(rows) != expected_cases)):
        raise ValueError("starter Case count differs from the declared experiment")
    records, answers = [], {}
    n, nt = payload["finite_sampling"]["grid_size"], payload["finite_sampling"]["forcing_times"]
    for row in rows:
        case = _case(row["case"])
        if (case.visibility is not CaseVisibility.DEVELOPMENT or case.source is not CaseSource.GENERATED
                or case.parent_case_id is not None or case.case_id in answers):
            raise ValueError("starter must contain unique clean DEVELOPMENT Cases")
        field_path, arrays_path = _asset(path, row, "state"), _asset(path, row, "arrays")
        if field_path.name != "hidden_continuous_case.pt":
            raise ValueError("unsupported continuous starter field filename")
        answer = StructuredNSMMSAnswer(load_structured_continuous_case(field_path.parent, device=device))
        actual_abi = answer.raw_answer.continuous_mms_abi
        if actual_abi != row["continuous_mms_abi"] or (expected_abi is not None and actual_abi != expected_abi):
            raise ValueError("starter continuous MMS version differs from the declared physics")
        domain = case.public_data["domain"]
        if (list(answer.domain_bounds) != [*domain["bounds"]["x"], *domain["bounds"]["y"]]
                or list(answer.time_interval) != list(domain["time_interval"])
                or answer.viscosity != case.public_data["parameters"]["viscosity"]
                or content_sha256(answer.public_geometry) != content_sha256(domain["geometry"])):
            raise ValueError("starter public problem and continuous field disagree")
        if arm == "fixed_sampled" or has_reference:
            with np.load(arrays_path, allow_pickle=False) as data:
                arrays = {k: data[k].copy() for k in data.files}
            shapes = {"x": (n,), "y": (n,), "t": (nt,), "fluid_mask": (n, n),
                      "initial_velocity": (2, n, n), "forcing": (nt, 2, n, n)}
            if has_reference:
                shapes["reference_velocity"] = (nt, 2, n, n)
            if any(arrays.get(k, np.empty(0)).shape != shape for k, shape in shapes.items()):
                raise ValueError("finite public input shapes disagree with the starter manifest")
            if any(not np.isfinite(value).all() for value in arrays.values()):
                raise ValueError("nonfinite starter public inputs")
            if has_reference:
                reference = {k: arrays[k] for k in ("x", "y", "t", "fluid_mask")}
                reference["velocity"] = arrays.pop("reference_velocity")
                case = replace(case, public_data={**case.public_data, "development_reference": reference,
                                                   "development_input_samples": arrays})
        if arm == "fixed_sampled":
            public = {k: v for k, v in case.public_data.items()
                      if k not in ("oracle_protocol", "development_solution_protocol")}
            public["arrays"] = arrays
            public["input_representation"] = "finite_samples_of_continuous_mms_public_inputs"
            case = replace(case, public_data=public)
        answers[case.case_id] = answer
        records.append(CaseRecord(case, _NSVelocityTruth(answer)))
    return SimpleNamespace(records=tuple(records), _answers=answers,
        manifest_sha256=_digest(path), manifest_path=path, manifest=payload,
        state_sha256=frozenset(row["state_sha256"] for row in rows),
        field_identities=frozenset(continuous_field_identity(a) for a in answers.values()))


class PairedSessionProvider:
    """Route frozen and newly issued Cases through the existing scoped public ABI."""
    def __init__(self, starter, selection_bank, *, dynamic=None, dynamic_state=None, arm):
        self.static = NSBankSessionProvider((starter, selection_bank))
        self.dynamic, self.dynamic_state, self.arm = dynamic, dynamic_state, arm
        self.static_ids = frozenset(r.case.case_id for r in (*starter.records, *selection_bank.records))
        self.array_ids = frozenset(r.case.case_id for r in starter.records) if arm == "fixed_sampled" else frozenset()
        self.state = dynamic_state if dynamic is not None else self.static.state

    def __getattr__(self, name):
        # Checkpoint codecs remain owned by the existing canonical NS Generator.
        if name in {"export_runtime_state", "restore_runtime_state", "export_case_record", "restore_case_record",
                    "export_generator_state", "restore_generator_state"} and self.dynamic is not None:
            return getattr(self.dynamic, name)
        raise AttributeError(name)

    def case_session_methods(self, access):
        if access is not CaseAccessKind.PUBLIC_PROBLEM:
            return ()
        methods = self.static.case_session_methods(access)
        if self.dynamic is not None:
            methods = tuple(dict.fromkeys((*methods, *self.dynamic.case_session_methods(access))))
        return methods

    def open_case_session(self, state, case, access):
        if state is not self.state or access is not CaseAccessKind.PUBLIC_PROBLEM or case.case_id in self.array_ids:
            raise PermissionError("this Case/access has no continuous public input permission")
        if case.case_id in self.static_ids:
            return self.static.open_case_session(self.static.state, case, access)
        if self.dynamic is None:
            raise PermissionError("unknown Case")
        return self.dynamic.open_case_session(self.dynamic_state, case, access)


class OnlineBaseEnvironment:
    """One persistent allowance per solver branch, shared by all its repair roles."""
    def __init__(self, generator, state, *, new_cases_per_branch=64, reserved_case_ids=(), expected_abi=None):
        if new_cases_per_branch is not None and (type(new_cases_per_branch) is not int or new_cases_per_branch < 1):
            raise ValueError("new_cases_per_branch must be a positive integer")
        self.generator, self.generator_state, self.limit = generator, state, new_cases_per_branch
        self.reserved = frozenset(reserved_case_ids)
        self.expected_abi = expected_abi
        self.progress, self.lock = {}, RLock()
        self.description = {"arm": "continuous_online_base", "new_cases_per_branch": self.limit,
                            "quota_unit": "solver_development_branch_including_repairs_and_restarts"}

    def bind(self, state):
        self.progress = state
        state.setdefault("branches", {})
        for branch in state["branches"].values():
            if self.limit is not None and len(branch["records"]) > self.limit:
                raise ValueError("resumed branch exceeds its declared Case allowance")

    def _branch(self, name):
        return self.progress.setdefault("branches", {}).setdefault(name, {"records": [], "next_operation": 0, "requests": []})

    def records(self, branch):
        return tuple(self._branch(branch)["records"])

    def sampler(self, branch, evaluator, seed):
        return _BranchSampler(self, branch, evaluator, seed)


class _BranchSampler:
    def __init__(self, environment, branch, evaluator, seed):
        self.env, self.branch, self.evaluator, self.seed = environment, branch, evaluator, seed

    @property
    def remaining(self):
        return None if self.env.limit is None else self.env.limit - len(self.env.records(self.branch))

    def sample_cases(self, request):
        import time
        if request.source is not SampleSource.BASE:
            raise PermissionError("only fresh Base acquisition is enabled")
        with self.env.lock:
            if self.remaining is not None and request.count > self.remaining:
                raise ValueError("branch Case allowance exceeded; repairs/restarts do not reset it")
            state = self.env._branch(self.branch)
            operation = state["next_operation"]
            state["next_operation"] += 1
            seed = int.from_bytes(sha256(f"{self.seed}:{self.branch}:{operation}".encode()).digest()[:8], "big")
            generation = GenerationRequest(1, request.count, GeneratorMode.ONLINE, GenerationKind.BASE,
                CaseVisibility.DEVELOPMENT, seed=seed, purpose=GenerationPurpose.DEVELOPMENT)
            started = time.monotonic()
            records = tuple(self.env.generator.sample(self.env.generator_state, generation))
            existing = self.env.reserved | {r.case.case_id for b in self.env.progress["branches"].values() for r in b["records"]}
            ids = [r.case.case_id for r in records]
            if (len(records) != request.count or len(set(ids)) != len(ids) or set(ids) & existing
                    or any(r.case.visibility is not CaseVisibility.DEVELOPMENT or
                           r.case.source is not CaseSource.GENERATED or r.case.parent_case_id is not None for r in records)):
                raise ValueError("Generator returned a non-unique/non-Base/non-development batch")
            if self.env.expected_abi is not None and any(
                getattr(getattr(getattr(r.truth, "_answer", None), "raw_answer", None), "continuous_mms_abi", None)
                != self.env.expected_abi for r in records):
                raise ValueError("fresh Base Cases changed the declared continuous MMS version")
            self.evaluator.register(records)
            state["records"].extend(records)
            state["requests"].append({"operation": operation, "seed": seed, "count": len(records),
                                      "case_ids": ids, "wall_seconds": time.monotonic() - started})
            cases = self.evaluator._candidate_cases_for_controller_ids(ids)
            return SampleBatch(cases, self.remaining)

    def perturb_cases(self, request):
        raise PermissionError("Perturbation is disabled in the paired Base environment")

    def sample_paired_cases(self, request):
        raise PermissionError("paired perturbation is disabled")


def paired_public_problem_bundle(arm):
    from ..public_problem import ns_public_problem_bundle
    from pde_agent_verifier_solver.sdk import PublicResource
    if arm not in ENVIRONMENT_ARMS:
        raise ValueError("unknown paired environment")
    bundle = ns_public_problem_bundle(guidance_supported=False, boundary_velocity_trace=True)
    if arm == "continuous_online_base":
        note = ("\n\nThe starter Cases may also expose development_input_samples and "
                "development_reference (velocity[T,2,X,Y] plus x/y/t/fluid_mask). These "
                "are finite samples of the same starter fields supplied in the paired "
                "fixed environment. They do not authorize continuous solution queries; "
                "fresh Cases expose public inputs and numerical evaluation feedback.\n")
        return replace(bundle, resources=tuple(
            replace(r, data=r.data + note.encode()) if r.relative_path == "ns_reference/README.md" else r
            for r in bundle.resources))
    note = """# NS finite public-input development environment

The equation, physical domain, boundary conditions and viscosity are declared by
case.public_data. arrays supplies x[X], y[Y], t[T], initial_velocity[2,X,Y],
forcing[T,2,X,Y], fluid_mask[X,Y], boundary_xy[N,2], boundary_velocity[T,2,N].
These are float64 samples of the SAME continuous MMS problems used by the paired
continuous environment, on cell-centre spatial nodes and uniform time nodes.
They are equation inputs. When development_reference is advertised, its velocity
array [T,2,X,Y] and x/y/t/fluid_mask provide finite reference samples of these
same development Cases. They do not grant continuous truth-query access. Choose your numerical
method and interpolate these inputs as needed. Continuous public-input queries
and new Case acquisition are unavailable for these development Cases.

Implement prepare_solution(case, solve_budget=None), returning evaluate(query).
Hidden validation/test Cases use continuous public initial/forcing (and declared
boundary) queries instead of arrays. The delivered solver must accept both input
paths. The shared SDK helper open_public_case_session(case) is available DURING
candidate execution on those hidden Cases. No hidden Case/answer is exposed to
the coding Agent. Solution-truth queries are disabled. The evaluation interface,
output fields and solver time cap are identical for both environment arms.
"""
    return replace(bundle, bundle_id="ns-paired-finite-inputs-v1", resources=(
        PublicResource("ns_reference/README.md", note.encode(), "text/markdown"),
        *(r for r in bundle.resources if r.relative_path != "ns_reference/README.md")),
        oracle_methods=(), development_solution_oracle_methods=())


def build_online_generator(config, base):
    """Load pinned RF/PCA artifacts using the existing canonical Base generator."""
    from ..canonical_backend import build_canonical_structured_ns_generator
    values = dict(config)
    values.pop("artifact_sha256", None)
    expected_abi = values.pop("expected_continuous_mms_abi", None)
    if expected_abi is not None:
        from agent_verifier.neural_generator.structured_ns_answer_first import SUPPORTED_STRUCTURED_NS_MMS_ABI_VERSIONS
        if expected_abi not in SUPPORTED_STRUCTURED_NS_MMS_ABI_VERSIONS:
            raise ValueError("this checkout cannot generate the requested continuous MMS version")
    for key in ("flow_checkpoint", "pca_checkpoint"):
        path = Path(values[key]).expanduser()
        path = (Path(base) / path).resolve() if not path.is_absolute() else path.resolve()
        values[key] = path
    for key in ("development_guidance_enabled", "development_perturbations_enabled", "guidance_config",
                "guidance_protocol", "guidance_artifact_root"):
        if key in values:
            raise ValueError("paired Base generator cannot configure Guidance/Perturbation")
    bundle = build_canonical_structured_ns_generator(**values,
        development_guidance_enabled=False, development_perturbations_enabled=False)
    if (expected_abi is not None
            and bundle.components.flow_sampler.continuous_mms_abi != expected_abi):
        raise ValueError("Generator artifacts use a different continuous MMS version than the starter bank")
    return bundle.generator, bundle.generator.load(bundle.checkpoint, settings={})


from ..tasks.solver import NSSolverTask, NSEvaluationProgram, _private_development_query


def _paired_development_query(case, settings, seed):
    # Representation is the intervention. Keep feedback coordinates continuous
    # and identical across arms instead of selecting M1's finite array nodes.
    public = {key: value for key, value in case.public_data.items() if key != "arrays"}
    return _private_development_query(replace(case, public_data=public), settings, seed)


class PairedNSSolverTask(NSSolverTask):
    def create_evaluation(self, settings):
        return NSEvaluationProgram.from_settings(
            program_id="ns-paired-continuous-feedback-v1", settings=settings,
            private_development_query=_paired_development_query,
            execute_candidate=self.execute_candidate, score_candidate=self.score_candidate)
