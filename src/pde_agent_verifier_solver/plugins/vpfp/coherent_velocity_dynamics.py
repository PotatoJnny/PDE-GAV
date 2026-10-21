"""Research-only coordinates for a driven, spatially coherent velocity.

For a periodic domain with constant total mass, U' + nu U = a(t). This
module represents U0 and a nonperiodic Legendre drive, NOT a source-free
constraint or a new decoder. Fixed-nu fitting/evaluation and standardization
are linear/affine, so an EDM may learn these coordinates without imposing a
nonlinear physical constraint at noisy states. Low polynomial degrees ARE
compression: support in the original 33-frame velocity space is not claimed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from hashlib import sha256
import json
import math

import numpy as np
import torch


SCHEMA = "vpfp_coherent_velocity_driven_ou_legendre_v1"


@dataclass(frozen=True)
class VPFPVelocityDynamicsConfig:
    drive_degree: int = 4  # Degree d means d+1 coefficients, including DC.
    t0: float = 0.
    t1: float = 1.
    node_count: int = 33
    quadrature_order: int = 64
    schema: str = SCHEMA

    def __post_init__(self):
        if (self.schema != SCHEMA or type(self.drive_degree) is not int
                or not 0 <= self.drive_degree <= 16 or self.node_count != 33
                or type(self.node_count) is not int or type(self.quadrature_order) is not int
                or self.quadrature_order not in (64,96,128)
                or isinstance(self.t0,bool) or isinstance(self.t1,bool)
                or not math.isfinite(self.t0) or not math.isfinite(self.t1)
                or self.t1 <= self.t0 or not math.isfinite(self.t1-self.t0)):
            raise ValueError("finite increasing time domain, 33 nodes and degree 0..16 required")

    @property
    def duration(self):
        return self.t1-self.t0

    @property
    def latent_dimension(self):
        return 2*(self.drive_degree+2)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, payload):
        if set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("exact velocity dynamics config fields required")
        return cls(**payload)


def _tensor(value, *, device=None):
    # NumPy keeps JSON/Python floats in FP64 instead of first quantizing them
    # through torch's default FP32 dtype (which could also spuriously overflow).
    raw = value if isinstance(value,torch.Tensor) else np.asarray(value)
    result = torch.as_tensor(raw,device=device)
    if result.dtype not in (torch.float32,torch.float64) or not torch.isfinite(result).all():
        raise ValueError("finite FP32/FP64 physical arrays required")
    return result.double()


def _nu(value, batch, device):
    result = _tensor(value,device=device)
    if result.shape != (batch,) or (result < 0).any():
        raise ValueError("nu must have shape [batch], be finite and nonnegative")
    return result


def _times(value, config, device):
    result = _tensor(value,device=device)
    if (result.ndim != 1 or len(result)==0 or (result < config.t0).any()
            or (result > config.t1).any() or (len(result)>1 and (result.diff()<=0).any())):
        raise ValueError("evaluation times must increase within the physical time domain")
    return result


def _coordinates(value, config):
    result = _tensor(value)
    if result.ndim != 2 or len(result)==0 or result.shape[1] != config.latent_dimension:
        raise ValueError("coordinates must be [batch, 2 + 2*(degree+1)]")
    return result


def _legendre(x, degree):
    values = [torch.ones_like(x)]
    if degree:
        values.append(x)
    for n in range(1,degree):
        values.append(((2*n+1)*x*values[-1]-n*values[-2])/(n+1))
    return torch.stack(values,dim=-1)


@lru_cache(maxsize=3)
def _quadrature(order):
    nodes,weights = np.polynomial.legendre.leggauss(order)
    return (nodes+1)/2,weights/2


def _kernel(nu, times, config, order=None):
    """OU convolution of each Legendre basis; no small-nu reciprocal.

    Integrate backwards from t. For nu*(t-t0)>64, omit only the remote tail:
    |omitted integral of P_n| <= exp(-64)/nu since |P_n|<=1 on [-1,1].
    This is independent of fitted residuals. Quadrature error is separately
    measured by a higher-order rule, not certified by agreement of two rules.
    """
    order = config.quadrature_order if order is None else order
    if order not in (64,96,128):
        raise ValueError("supported quadrature orders are 64/96/128")
    elapsed = times-config.t0
    collision_time = nu[:,None]*elapsed[None,:]
    if not torch.isfinite(collision_time).all():
        raise ValueError("nu*time overflow is unsupported")
    truncated = collision_time > 64.
    # clamp_min affects only the unused division in the small-nu branch.
    width = torch.where(truncated,64./nu[:,None].clamp_min(1./config.duration),elapsed[None,:])
    nodes,weights = _quadrature(order)
    nodes = torch.as_tensor(nodes,device=nu.device,dtype=torch.float64)
    weights = torch.as_tensor(weights,device=nu.device,dtype=torch.float64)
    backward = width[...,None]*nodes
    location = 2*((elapsed[None,:,None]-backward)/config.duration)-1
    basis = _legendre(location,config.drive_degree)
    weight = torch.exp(-nu[:,None,None]*backward)*weights
    kernel = width[...,None]*(weight[...,None]*basis).sum(dim=-2)
    tail = torch.where(truncated,math.exp(-64.)/nu[:,None].clamp_min(1./config.duration),0.)
    if not torch.isfinite(kernel).all():
        raise ValueError("OU kernel nonfinite; no numerical qualification")
    return kernel,tail


def _drive_diagnostics(drive, config):
    # Time-average squared drive: orthogonality gives sum c_n^2/(2n+1).
    n = torch.arange(config.drive_degree+1,device=drive.device,dtype=torch.float64)
    energy = drive.square().sum(dim=1)/(2*n+1)
    derivative_matrix = torch.zeros((len(n),len(n)),device=drive.device,dtype=torch.float64)
    for row in range(len(n)):
        for column in range(row):
            if (row-column)%2:
                derivative_matrix[row,column] = 2*(2*column+1)/config.duration
    derivative_coefficients = drive@derivative_matrix
    derivative_energy = (derivative_coefficients.square()/(2*n+1)).sum(dim=(1,2))
    return energy,derivative_coefficients,derivative_energy


def evaluate_velocity_dynamics(coordinates, nu, times, config, *, quadrature_order=None):
    """Return FP64 U/Udot/a/adot [B,2,Q]; gradients to coordinates are retained.

    Layout: [U0_x,U0_y,a_x0,...,a_xd,a_y0,...,a_yd]. nu has [B];
    times has [Q] in physical units. a is acceleration, not a sampled MMS
    source family. Its interpretation requires constant mass and periodicity.
    """
    values = _coordinates(coordinates,config)
    collisions = _nu(nu,len(values),values.device)
    points = _times(times,config,values.device)
    u0,drive = values[:,:2],values[:,2:].reshape(len(values),2,config.drive_degree+1)
    kernel,tail = _kernel(collisions,points,config,quadrature_order)
    basis = _legendre(2*(points-config.t0)/config.duration-1,config.drive_degree)
    decay = torch.exp(-collisions[:,None]*(points-config.t0)[None,:])
    velocity = u0[:,:,None]*decay[:,None,:]+torch.einsum("bcn,btn->bct",drive,kernel)
    acceleration = torch.einsum("bcn,tn->bct",drive,basis)
    dt = acceleration-collisions[:,None,None]*velocity
    energy,derivative_coefficients,derivative_energy = _drive_diagnostics(drive,config)
    high_start = max(1,(config.drive_degree+2)//2)
    result = {"U":velocity,"dU_dt":dt,"a":acceleration,
        "da_dt":torch.einsum("bcn,tn->bct",derivative_coefficients,basis),
        "drive_legendre_energy":energy,"drive_mean_square":energy.sum(dim=1),
        "drive_derivative_mean_square":derivative_energy,
        "drive_high_order_energy":energy[:,high_start:].sum(dim=1),
        "velocity_tail_absolute_bound":drive.abs().sum(dim=-1)[:,:,None]*tail[:,None,:]}
    if not all(torch.isfinite(value).all() for value in result.values()):
        raise ValueError("nonfinite driven-velocity evaluation")
    return result


@dataclass(frozen=True)
class VPFPVelocityDynamicsFit:
    coordinates: torch.Tensor
    diagnostics: dict


def fit_velocity_dynamics(velocity, nu, config):
    """Fit every trajectory independently, keeping its measured U(t0) exactly.

    Input U is [B,2,33] on uniform physical times. This is a representation
    fit, not shared training. Validation may measure its compression error,
    but may never enter the shared normalizer's fit. No PDE solve is used.
    """
    values = _tensor(velocity)
    if values.ndim!=3 or len(values)==0 or values.shape[1:]!=(2,config.node_count):
        raise ValueError("uniform 33-frame U array [B,2,33] required")
    collisions = _nu(nu,len(values),values.device)
    points = torch.linspace(config.t0,config.t1,config.node_count,device=values.device,dtype=torch.float64)
    kernel,_ = _kernel(collisions,points,config)
    matrix = kernel[:,1:,:]
    left,singular,right = torch.linalg.svd(matrix,full_matrices=False)
    cutoff = torch.finfo(torch.float64).eps*max(matrix.shape[1:])*singular[:,0]
    if (singular[:,-1]<=cutoff).any():
        raise ValueError("rank-deficient drive design; no silent truncation or ridge fitted")
    decay = torch.exp(-collisions[:,None]*(points-config.t0)[None,:])
    target = values[:,:,1:]-values[:,:,:1]*decay[:,None,1:]
    drive = torch.einsum("bnq,bcq->bcn",right.transpose(-2,-1),
        torch.einsum("btq,bct->bcq",left,target)/singular[:,None,:])
    coordinates = torch.cat((values[:,:,0],drive.flatten(1)),dim=1)
    output = evaluate_velocity_dynamics(coordinates,collisions,points,config)
    residual = output["U"]-values
    alternate_order = 96 if config.quadrature_order in (64,128) else 128
    alternate = evaluate_velocity_dynamics(coordinates,collisions,points,config,quadrature_order=alternate_order)
    def rows(tensor): return tensor.detach().cpu().tolist()
    diagnostics = {"schema":SCHEMA+"_fit_report","config":config.to_dict(),
        "original_coordinates":66,"retained_coordinates":config.latent_dimension,
        "U0_preserved_exactly":bool(torch.equal(output["U"][:,:,0],values[:,:,0])),
        "node_rms_error":rows(residual.square().mean(dim=(1,2)).sqrt()),
        "future_node_rms_error":rows(residual[:,:,1:].square().mean(dim=(1,2)).sqrt()),
        "node_max_absolute_error":rows(residual.abs().amax(dim=(1,2))),
        "design_condition_number":rows(singular[:,0]/singular[:,-1]),
        "design_smallest_singular_value":rows(singular[:,-1]),
        "quadrature_orders":[config.quadrature_order,alternate_order],
        "quadrature_velocity_max_difference":rows((alternate["U"]-output["U"]).abs().amax(dim=(1,2))),
        "velocity_tail_absolute_bound":rows(output["velocity_tail_absolute_bound"].amax(dim=(1,2))),
        "drive_legendre_energy":rows(output["drive_legendre_energy"]),
        "drive_mean_square":rows(output["drive_mean_square"]),
        "drive_derivative_mean_square":rows(output["drive_derivative_mean_square"]),
        "drive_high_order_energy":rows(output["drive_high_order_energy"]),
        "high_order_meaning":"Legendre degrees >= max(1,ceil((d+1)/2)); NOT periodic Fourier power",
        "drive_periodic_or_zero_mean_required":False,"source_free_required":False,
        "full_support_in_original_33_node_space":False,
        "off_node_reference_error_measured":False,"two_quadrature_orders_prove_convergence":False,
        "scientific_admission":False,"new_PDE_solves":0}
    return VPFPVelocityDynamicsFit(coordinates,diagnostics)


class VPFPVelocityDynamicsNormalizer:
    """Train-only reversible affine coordinates; separate velocity/drive floors.

    Positive scales preserve Gaussian support in the retained coefficient
    space. They do not restore the degrees discarded by polynomial fitting.
    This prototype does not implement an EDM or change any sampler schema.
    """
    schema = SCHEMA+"_train_normalizer"

    def __init__(self,config,center,scales,metadata):
        center,scales = _tensor(center),_tensor(scales)
        if (center.shape!=(config.latent_dimension,) or scales.shape!=center.shape
                or (scales<=0).any() or metadata.get("split")!="train"
                or metadata.get("generated_samples_used_for_fit") is not False):
            raise ValueError("positive train-only dynamics normalization required")
        self.config,self.center,self.scales,self.fit_metadata = config,center.clone(),scales.clone(),dict(metadata)

    @classmethod
    def fit(cls,config,coordinates,*,sample_ids,split="train",absolute_floor=1.e-6,relative_floor=1.e-3):
        values = _coordinates(coordinates,config)
        ids = list(sample_ids)
        if (split!="train" or len(values)<2 or len(ids)!=len(values)
                or any(isinstance(x,bool) or not isinstance(x,(str,int)) or str(x)=="" for x in ids)
                or len(set(map(str,ids)))!=len(ids)
                or not all(not isinstance(x,bool) and math.isfinite(x) and x>0 for x in (absolute_floor,relative_floor))):
            raise ValueError("distinct train IDs and positive finite floors required")
        center = values.mean(dim=0)
        std = (values-center).square().mean(dim=0).sqrt()
        floors=[]
        for block in (values[:,:2],values[:,2:]):
            floors.append(torch.maximum(torch.tensor(absolute_floor,device=values.device,dtype=torch.float64),relative_floor*block.square().mean().sqrt()))
        scales = torch.cat((std[:2].clamp_min(floors[0]),std[2:].clamp_min(floors[1])))
        return cls(config,center,scales,{"split":"train","sample_ids":list(map(str,ids)),
            "generated_samples_used_for_fit":False,"absolute_floor":absolute_floor,"relative_floor":relative_floor,
            "floor_blocks":"U0_velocity_and_drive_acceleration_separately",
            "coordinate_SHA256":sha256(values.detach().cpu().contiguous().numpy().tobytes()).hexdigest()})

    def encode(self,coordinates):
        values = _coordinates(coordinates,self.config)
        return (values-self.center.to(values))/self.scales.to(values)

    def decode(self,coordinates):
        values = _coordinates(coordinates,self.config)
        return values*self.scales.to(values)+self.center.to(values)

    def to_payload(self):
        data = {"schema":self.schema,"config":self.config.to_dict(),"center":self.center.cpu().tolist(),
            "scales":self.scales.cpu().tolist(),"fit_metadata":self.fit_metadata}
        return {**data,"sha256":sha256(json.dumps(data,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()}

    @classmethod
    def from_payload(cls,payload):
        if set(payload)-{"sha256"}!={"schema","config","center","scales","fit_metadata"} or payload["schema"]!=cls.schema:
            raise ValueError("explicit velocity normalizer schema required")
        return cls(VPFPVelocityDynamicsConfig.from_dict(payload["config"]),payload["center"],payload["scales"],payload["fit_metadata"])
