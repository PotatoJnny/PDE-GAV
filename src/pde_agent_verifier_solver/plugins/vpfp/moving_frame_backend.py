"""BASE-only resident moving-frame answers for the existing VPFP macro Task.

No Hermite embedding, old analytic source, perturbation, guide or online PDE
solve is used. Public initial data and every Oracle come from the same A.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from threading import RLock

import numpy as np
import torch

from pde_agent_verifier_solver.sdk import Case, CaseRecord, CaseSource, CaseVisibility, GenerationKind
from .problem import VPFPContinuousMacroTruth
from .public_problem import VPFP_PUBLIC_SESSION_METHODS, VPFP_DEVELOPMENT_SOLUTION_SESSION_METHODS
from .perturbation_backend import (_VPFPPublicSession, _VPFPDevelopmentSession,
    _validated_oracle_points, _validated_release_case_ids, _snapshot_vpfp_public_case,
    _equivalent_vpfp_public_snapshot, _validate_request, _provenance, _seeds, _case_ids)


def _query(decoder,points,width):
    device = next(decoder.parameters()).device
    query = torch.as_tensor(points,device=device,dtype=torch.float64)
    if query.ndim != 2 or query.shape[1] != width or not len(query) or not torch.isfinite(query).all():
        raise ValueError(f"moving-frame query must be finite nonempty [N,{width}]")
    return query


def evaluate_moving_frame_macros(decoder,points_txy,carrier,condition):
    """Differentiable full-velocity moments; raw second moment INCLUDES rho."""
    points = _query(decoder,points_txy,3)
    phase = torch.cat((points,points.new_zeros((len(points),2))),1)
    fields = decoder.fields(phase,carrier,equation_condition=condition)
    rho,u,c = fields["rho"],fields["velocity"],fields["central_covariance"]
    raw = rho[:,None,None]*(c+u[:,:,None]*u[:,None,:])
    return {"rho":rho,"momentum":fields["momentum"],"kinetic_energy":fields["kinetic_energy"],
            "raw_second_moment":torch.stack((raw[:,0,0],raw[:,0,1],raw[:,1,1]),1),
            "electric":fields["electric"],"velocity":u,"central_covariance":c}


@torch.enable_grad()
def manufacture_moving_frame_source(decoder,points,carrier,condition,*,create_graph=False):
    """Actual A residual, with FP64 coordinates and bath theta (NOT stress C).

    S=f_t+v.grad_x f+E.grad_v f-nu(2f+v.grad_v f+theta Delta_v f).
    Existing differentiable query tensors stay connected when create_graph=True.
    Manufacturing S is not an independent verification of this same residual.
    """
    query = _query(decoder,points,5)
    if not query.requires_grad:
        query = query.detach().clone().requires_grad_(True)
    if create_graph:
        decoder.validate_source_gradient_domain(query,carrier)
    values = decoder.fields(query,carrier,equation_condition=condition)
    f = values["f"]
    gradient = torch.autograd.grad(f.sum(),query,create_graph=True,retain_graph=True)[0]
    second_x = torch.autograd.grad(gradient[:,3].sum(),query,create_graph=create_graph,retain_graph=True)[0][:,3]
    second_y = torch.autograd.grad(gradient[:,4].sum(),query,create_graph=create_graph,retain_graph=create_graph)[0][:,4]
    physical = values["equation_condition"].double()
    collision = 2*f+(query[:,3:5]*gradient[:,3:5]).sum(1)+physical[:,1]*(second_x+second_y)
    source = gradient[:,0]+(query[:,3:5]*gradient[:,1:3]).sum(1)+(values["electric"].double()*gradient[:,3:5]).sum(1)-physical[:,0]*collision
    result = {**values,"source":source,"f_gradient":gradient,"f_velocity_laplacian":second_x+second_y,
              "collision":collision,"points":query}
    if any(not torch.isfinite(value).all() for value in result.values()):
        raise FloatingPointError("nonfinite moving-frame source; no clipping or dropped queries")
    return result


def _numpy(value):
    return value.detach().cpu().numpy().astype(np.float64,copy=True)


@dataclass(frozen=True)
class _Resident:
    case: Case
    public_snapshot: Case
    carrier: dict
    condition: torch.Tensor
    answer_sha256: str


class MovingFrameBackend:
    """Plugin-owned backend + resident runtime; intentionally BASE-only."""
    supports_perturbations = False
    supports_guidance = False
    public_session_methods = VPFP_PUBLIC_SESSION_METHODS
    development_solution_session_methods = VPFP_DEVELOPMENT_SOLUTION_SESSION_METHODS

    def __init__(self,decoder,sampler,*,output_root,public_grid=33,public_times=13,source_chunk=32,field_chunk=256,
            batched_source=False, adaptive_source_only=False, oracle_execution_options=None):
        if type(batched_source) is not bool or type(adaptive_source_only) is not bool:
            raise TypeError("source implementation flags must be explicit booleans")
        if batched_source and adaptive_source_only:
            raise ValueError("source-only and all-field acceleration are mutually exclusive")
        from .moving_frame_oracle_options import validated_source_execution_options
        self.oracle_execution_options = validated_source_execution_options(oracle_execution_options)
        if self.oracle_execution_options is not None and not adaptive_source_only:
            raise ValueError("batched geometry requires source-only acceleration")
        if any(type(n) is not int or n<1 for n in (public_grid,public_times,source_chunk,field_chunk)) or min(public_grid,public_times)<2:
            raise ValueError("positive bounded runtime grid/chunk settings required")
        self.decoder,self.sampler = decoder,sampler
        self.batched_source = batched_source
        self.adaptive_source_only = adaptive_source_only
        self._source_evaluators = {}
        self.source_implementation = None
        if batched_source or adaptive_source_only:
            from .moving_frame_adaptive_source import SOURCE_POLICY
            self.source_implementation = {"name": "adaptive_exact_v1", "source_policy": SOURCE_POLICY,
                "adaptive_source_only": adaptive_source_only,
                "macro_evaluator": "original" if adaptive_source_only else "accelerated",
                "source_derivation": "actual_A_adaptive_autodiff_or_batched_coordinate_jets_analytic_velocity_OU_Poisson_residual",
                "physical_module_sha256": {name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                    for name in ("moving_frame_adaptive_source.py", "moving_frame_batched_source.py",
                                 "moving_frame_fast_source.py", "moving_frame_fast_macros.py")}}
            if self.oracle_execution_options is not None:
                self.source_implementation["oracle_execution_options"] = dict(self.oracle_execution_options)
                if "g0_request_block" in self.oracle_execution_options:
                    self.source_implementation["g0_evaluator"] = "batched_exact_geometry_values_v1"
                self.source_implementation["physical_module_sha256"]["moving_frame_oracle_options.py"] = (
                    sha256(Path(__file__).with_name("moving_frame_oracle_options.py").read_bytes()).hexdigest())
        self.sampler_identity = json.loads(json.dumps(sampler.identity,sort_keys=True,allow_nan=False))
        self.sampler_identity_sha256 = sha256(json.dumps(self.sampler_identity,sort_keys=True,
            separators=(",",":"),allow_nan=False).encode()).hexdigest()
        if Path(output_root).is_symlink():
            raise ValueError("private runtime output must not be symlinked")
        self.output_root = Path(output_root).resolve()
        if self.output_root.exists():
            raise ValueError("moving-frame runtime requires fresh private output")
        self.output_root.mkdir(parents=True,mode=0o700)
        self.public_grid,self.public_times = public_grid,public_times
        self.source_chunk,self.field_chunk = source_chunk,field_chunk
        self._runtime_lock = RLock()
        self._resident_cases,self._issued = {},set()
        self.generation_receipts = []
        self.oracle_calls = {name:0 for name in ("source","rho0","g0","macros")}

    def train_base(self,*args,**kwargs):
        raise RuntimeError("moving-frame Campaign requires frozen models")

    def fit_guidance(self,*args,**kwargs):
        raise RuntimeError("moving-frame guidance is not implemented")

    def _resident_for_id(self,case_id):
        resident = self._resident_cases.get(case_id)
        if resident is None:
            raise ValueError("moving-frame answer is no longer active")
        return resident

    def _active(self,case):
        if not isinstance(case,Case):
            raise TypeError("moving-frame session requires Case")
        resident = self._resident_for_id(case.case_id)
        if not _equivalent_vpfp_public_snapshot(resident.public_snapshot,case):
            raise ValueError("moving-frame public snapshot changed")
        return resident

    def public_session_factory(self,case):
        with self._runtime_lock:
            return _VPFPPublicSession(self,self._active(case).case)

    def development_solution_session_factory(self,case):
        if not isinstance(case,Case) or case.visibility is not CaseVisibility.DEVELOPMENT:
            raise PermissionError("moving-frame completed macros are DEVELOPMENT-only")
        with self._runtime_lock:
            return _VPFPDevelopmentSession(self,self._active(case).case)

    def release_cases(self,case_ids):
        with self._runtime_lock:
            for identifier in _validated_release_case_ids(case_ids):
                self._resident_cases.pop(identifier,None)
                self._source_evaluators.pop(identifier,None)
        # Retain private immutable replay files and issued IDs, not GPU tensors.

    def _source_evaluator(self,identifier,carrier,condition):
        if not (self.batched_source or self.adaptive_source_only):
            return None
        if identifier not in self._source_evaluators:
            from .moving_frame_adaptive_source import MovingFrameAdaptiveSourceEvaluator
            options = getattr(self, "oracle_execution_options", None)
            tuning = {} if options is None else {
                "geometry_cache_size": options["geometry_cache_size"],
                "geometry_batch_size": options["geometry_batch_size"],
                "phase_chunk_size": max(options["source_request_block"], options.get("g0_request_block", 0)),
            }
            self._source_evaluators[identifier] = MovingFrameAdaptiveSourceEvaluator(
                self.decoder, carrier, condition, **tuning)
        return self._source_evaluators[identifier]

    def source_execution_stats(self):
        """Controller-only work counters; no coordinates or final source values."""
        with self._runtime_lock:
            options = getattr(self, "oracle_execution_options", None)
            return {"oracle_execution_options": options,
                    "source_outer_block": self.source_chunk if options is None else options["source_request_block"],
                    "g0_outer_block": (self.field_chunk if options is None else
                                       options.get("g0_request_block", self.field_chunk)),
                    "prepared_case_count": len(self._source_evaluators),
                    "cases": {key: value.cache_info() for key, value in self._source_evaluators.items()}}

    def _evaluate(self,case_id,points,width,kind):
        query = _validated_oracle_points(points,width=width,name=kind)
        with self._runtime_lock:
            resident = self._resident_for_id(case_id)
            options = getattr(self, "oracle_execution_options", None)
            batched_g0 = kind == "g0" and options is not None and "g0_request_block" in options
            evaluator = (self._source_evaluator(case_id,resident.carrier,resident.condition)
                         if kind == "source" or self.batched_source or batched_g0 else None)
            self.oracle_calls[kind] += 1
            if kind=="rho0":
                query = np.column_stack((np.full(len(query),self.decoder.macro_config.time_domain[0]),query,np.zeros((len(query),2))))
            elif kind=="g0":
                query = np.column_stack((np.full(len(query),self.decoder.macro_config.time_domain[0]),query))
            rows = []
            chunk = self.source_chunk if kind=="source" else self.field_chunk
            if kind == "source" and options is not None:
                # Group requested velocities before adaptive routing. The old
                # 32-point outer slices hid opportunities to share geometry.
                chunk = options["source_request_block"]
            elif batched_g0:
                # Preserve exact rows/order while sharing t0/x/y value geometry
                # across velocities. fields() prepares no coordinate jets;
                # later source calls upgrade cached entries when necessary.
                chunk = options["g0_request_block"]
            for start in range(0,len(query),chunk):
                part = query[start:start+chunk]
                if kind=="source":
                    value = (evaluator.source(part,create_graph=False) if evaluator is not None else
                        manufacture_moving_frame_source(self.decoder,part,resident.carrier,resident.condition))["source"]
                elif kind=="macros":
                    with torch.no_grad():
                        value = (evaluator.macros(part) if evaluator is not None else
                            evaluate_moving_frame_macros(self.decoder,part,resident.carrier,resident.condition))
                    rows.append({name:_numpy(value[name]) for name in ("rho","momentum","kinetic_energy","raw_second_moment")})
                    continue
                else:
                    with torch.no_grad():
                        if evaluator is not None:
                            value = (evaluator.macros(part[:, :3])["rho"] if kind=="rho0"
                                     else evaluator.fields(part)["g"])
                        else:
                            value = self.decoder.fields(_query(self.decoder,part,5),resident.carrier,
                                equation_condition=resident.condition)["rho" if kind=="rho0" else "g"]
                rows.append(_numpy(value))
            if kind=="macros":
                return {name:np.concatenate([row[name] for row in rows]) for name in rows[0]}
            return np.concatenate(rows)

    def evaluate_source(self,case_id,points):
        return self._evaluate(case_id,points,5,"source")

    def evaluate_rho0(self,case_id,points):
        return self._evaluate(case_id,points,2,"rho0")

    def evaluate_g0(self,case_id,points):
        return self._evaluate(case_id,points,4,"g0")

    def evaluate_macros(self,case_id,points):
        return self._evaluate(case_id,points,3,"macros")

    def evaluate_transport_macros(self, case_id, points):
        """Opt-in v2 query: no change to legacy macros or manufactured source."""
        from .moving_frame_fast_macros import evaluate_moving_frame_transport_macros
        query = _validated_oracle_points(points, width=3, name="transport_macros")
        with self._runtime_lock, torch.no_grad():
            resident = self._resident_for_id(case_id)
            evaluator = self._source_evaluator(case_id, resident.carrier, resident.condition)
            self.oracle_calls["transport_macros"] = self.oracle_calls.get("transport_macros", 0)+1
            rows = []
            for start in range(0, len(query), self.field_chunk):
                part = query[start:start+self.field_chunk]
                values = (evaluator.transport_macros(part) if evaluator is not None else
                    evaluate_moving_frame_transport_macros(self.decoder, part, resident.carrier, resident.condition))
                rows.append({name: _numpy(value) for name, value in values.items()})
            return {name: np.concatenate([row[name] for row in rows]) for name in rows[0]}

    def _record(self,identifier,carrier,condition,request,cohort):
        cfg = self.decoder.macro_config
        evaluator = self._source_evaluator(identifier,carrier,condition) if self.batched_source else None
        x,y = [np.arange(self.public_grid,dtype=np.float64)*length/self.public_grid for length in cfg.spatial_domain_lengths]
        t = np.linspace(*cfg.time_domain,self.public_times)
        mesh = np.meshgrid(x,y,indexing="ij")
        points = np.column_stack((np.full(mesh[0].size,t[0]),mesh[0].ravel(),mesh[1].ravel()))
        with torch.no_grad():
            parts = [(evaluator.macros(points[i:i+self.field_chunk]) if evaluator is not None else
                      evaluate_moving_frame_macros(self.decoder,points[i:i+self.field_chunk],carrier,condition))
                     for i in range(0,len(points),self.field_chunk)]
        initial = {name:np.concatenate([_numpy(part[name]) for part in parts]) for name in ("rho","momentum","kinetic_energy")}
        equation = self.decoder.base_decoder.public_equation_metadata()
        equation.update(collision_nu=float(condition[0]),temperature=float(condition[1]))
        question = {**equation,"case_id":identifier,"equation":equation.get("equation_version"),
            "task":"predict exact full-space VPFP macroscopic quantities","time_domain":list(cfg.time_domain),
            "solver_output_contract_version":"vpfp_continuous_macro_output_v1",
            "source_origin":("generated_continuous_moving_frame_batched_exact_residual" if self.batched_source or self.adaptive_source_only else
                             "generated_continuous_moving_frame_autodiff_residual"),
            "factorization":"f=rho*g; integral_R2 g dv=1",
            "grid":{"nx":self.public_grid,"ny":self.public_grid,"nvx":9,"nvy":9,
                    "x_domain":list(cfg.spatial_domain_lengths),"v_bounds":[-5.,5.,-5.,5.]},
            "time":{"t0":float(t[0]),"t1":float(t[-1]),"nt":len(t)},
            "materialization":{"role":"public_initial_macro_view_only","primary_truth":"controller_private_continuous_answer",
                               "agent_internal_grid_may_be_finer":True}}
        velocity = np.linspace(-5.,5.,9)
        weights = np.full(9,1.25); weights[[0,-1]] *= .5
        arrays = {"t":t,"x":x,"y":y,"vx":velocity.copy(),"vy":velocity.copy(),"velocity_weights":np.outer(weights,weights),
            "rho0":initial["rho"].reshape(self.public_grid,self.public_grid),
            "momentum0":initial["momentum"].reshape(self.public_grid,self.public_grid,2),
            "kinetic_energy0":initial["kinetic_energy"].reshape(self.public_grid,self.public_grid)}
        case = Case(case_id=identifier,source=CaseSource.GENERATED,visibility=request.visibility,
            public_data={"equation":"2D2V Vlasov--Poisson--Fokker--Planck","question":question,"arrays":arrays},
            round_index=request.round_index,tags=("vpfp","moving-frame-A","base",f"cohort:{cohort}"))
        truth = VPFPContinuousMacroTruth(identifier,tuple(t),tuple(x),tuple(y),self.evaluate_macros,
                                        evaluate_transport_points=self.evaluate_transport_macros)
        return CaseRecord(case,truth)

    def generate(self,base_state,guidance_state,request,progress=None):
        del base_state,progress
        _validate_request(request)
        if request.kind is not GenerationKind.BASE or guidance_state is not None:
            raise ValueError("moving-frame backend supports unguided BASE only")
        _,_,cohort = _provenance(request)
        seeds,identifiers = _seeds(request),_case_ids(request,_seeds(request))
        with self._runtime_lock:
            if any(identifier in self._issued or (self.output_root/identifier).exists() for identifier in identifiers):
                raise ValueError("moving-frame case identifiers cannot be reused")
            batch = self.sampler.generate(seeds)
            required = {"schema","m","z","equation_conditions","equation_seeds"}
            if (not isinstance(batch,dict) or not required.issubset(batch)
                    or batch["schema"]!="vpfp_moving_frame_native_answer_batch_v1"
                    or tuple(map(int,batch["equation_seeds"]))!=seeds
                    or any(len(batch[name])!=len(seeds) for name in ("m","z","equation_conditions"))):
                raise ValueError("moving-frame sampler changed the native batch contract")
            pending,records = {},[]
            try:
                for index,identifier in enumerate(identifiers):
                    carrier = {name:torch.as_tensor(batch[name][index:index+1]).detach().cpu().clone() for name in ("m","z")}
                    condition = torch.as_tensor(batch["equation_conditions"][index],dtype=torch.float32).detach().cpu().clone()
                    if condition.shape!=(2,) or not torch.isfinite(condition).all():
                        raise ValueError("invalid moving-frame condition row")
                    payload = self.decoder.carrier_to_payload(carrier)
                    record = self._record(identifier,carrier,condition,request,cohort)
                    directory = self.output_root/identifier
                    directory.mkdir(mode=0o700)
                    answer = directory/"private-answer.pt"
                    with answer.open("xb") as handle:
                        torch.save({"schema":"vpfp_moving_frame_private_answer_v1","carrier":payload,
                                    "condition":condition,"equation_seed":seeds[index],
                                    "sampler_identity":self.sampler_identity,
                                    "sampler_identity_sha256":self.sampler_identity_sha256},handle)
                    digest = sha256(answer.read_bytes()).hexdigest()
                    receipt = {"case_id":identifier,"private_answer_sha256":digest,"decoder_schema":self.decoder.schema,
                               "sampler_identity_sha256":self.sampler_identity_sha256,
                               "source":"actual_A_autodiff","pde_solves":0,"scientific_admission":False}
                    if self.batched_source or self.adaptive_source_only:
                        receipt.update(source="actual_A_batched_exact",numerical_implementation=self.source_implementation)
                    (directory/"private-receipt.json").write_text(json.dumps(receipt,sort_keys=True,allow_nan=False))
                    pending[identifier] = _Resident(record.case,_snapshot_vpfp_public_case(record.case),carrier,condition,digest)
                    records.append(record)
                    self.generation_receipts.append(receipt)
            except Exception as error:
                # Keep failed files for diagnosis; do not publish partial resident cases.
                for identifier in identifiers:
                    self._source_evaluators.pop(identifier,None)
                (self.output_root/f"failed-batch-{request.seed}.json").write_text(json.dumps({"error":repr(error),"case_ids":identifiers}))
                raise
            self._resident_cases.update(pending); self._issued.update(identifiers)
            return tuple(records)
