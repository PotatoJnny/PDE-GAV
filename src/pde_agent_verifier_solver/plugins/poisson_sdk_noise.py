"""Poisson boundary/MMS adapter for the existing SDK perturbation helper.

Noise draws and amplitude selection are the SDK implementation. This module
only supplies the PDE-specific continuous composition, derivatives and checks.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
from pde_agent_verifier_solver.sdk.perturbation import (
    SmoothRandomFieldSettings, SmoothRandomFourierField, draw_smooth_random_field,
    MultiplicativePerturbationSettings, RelativeChangeRange,
    PerturbationValidation, sample_multiplicative_mms_perturbation,
)
from .poisson_diagnostic import PoissonField

LENGTH_SCALES = (1/64, 1/128, 1/256, 1/512)
SOLUTION_RANGE = (1e-5, 2e-5)
MAX_SOURCE_CHANGE = 1.0
NORM_X = np.linspace(0., 1., 8193)


@dataclass(frozen=True)
class SDKPoissonField:
    amplitude: float
    noise: SmoothRandomFourierField | None = None
    alpha: float = 0.
    epsilon_fraction: float = .001

    def random_derivatives(self, x):
        x = np.asarray(x,dtype=float).reshape(-1)
        if self.noise is None:
            return (np.zeros_like(x),)*3
        frequencies = np.asarray(self.noise.frequencies)[:,0]
        weights = np.asarray(self.noise.weights)
        phase = x[:,None]*frequencies+np.asarray(self.noise.phases)
        # Same terms and analytic derivative formula as SDK field.derivative;
        # vectorization changes neither its draw nor its spatial spectrum.
        return tuple(np.sum(np.cos(phase+order*np.pi/2)*(weights*frequencies**order),axis=1)
                     for order in (0,1,2))

    def solution(self, x):
        x = np.asarray(x,dtype=float).reshape(-1)
        u = self.amplitude*np.sin(np.pi*x)
        if self.noise is None or self.alpha == 0: return u
        delta,_,_ = self.random_derivatives(x)
        # SDK epsilon-shifted multiplication, with a PDE-owned boundary envelope.
        return u+self.alpha*(u+self.epsilon_fraction*abs(self.amplitude))*x*(1-x)*delta

    def forcing(self, x):
        x = np.asarray(x,dtype=float).reshape(-1)
        u = self.amplitude*np.sin(np.pi*x)
        up = self.amplitude*np.pi*np.cos(np.pi*x)
        upp = -np.pi**2*u
        if self.noise is None or self.alpha == 0: return -upp
        d,dp,dpp = self.random_derivatives(x)
        w,wp,wpp = x*(1-x),1-2*x,-2.
        shifted = u+self.epsilon_fraction*abs(self.amplitude)
        vpp = (upp*w+2*up*wp+shifted*wpp)*d+2*(up*w+shifted*wp)*dp+shifted*w*dpp
        return -upp-self.alpha*vpp

    def answer(self, query):
        return PoissonField.answer(self,query)

    def to_payload(self):
        return {'amplitude':self.amplitude,'alpha':self.alpha,
                'epsilon_fraction':self.epsilon_fraction,
                'noise':None if self.noise is None else asdict(self.noise)}

    @classmethod
    def from_payload(cls, payload):
        item=dict(payload);noise=item.pop('noise')
        return cls(noise=None if noise is None else SmoothRandomFourierField(**noise),**item)


def relative_change(left, right):
    left,right=np.asarray(left),np.asarray(right)
    return float(np.sqrt(np.trapezoid((right-left)**2,NORM_X)/np.trapezoid(left*left,NORM_X)))


def physical_checks(parent, child):
    source_change=relative_change(parent.forcing(NORM_X),child.forcing(NORM_X))
    x=np.linspace(.001,.999,513);h=3e-5;u=child.solution
    second=(-u(x+2*h)+16*u(x+h)-30*u(x)+16*u(x-h)-u(x-2*h))/(12*h*h)
    f=child.forcing(x)
    residual=float(np.linalg.norm(-second-f)/np.linalg.norm(f))
    boundary=float(np.max(np.abs(u([0.,1.]))))
    return {'source_relative_change':source_change,'mms_residual':residual,
            'boundary_residual':boundary}


def perturb(parent, *, seed, length_scale):
    if parent.noise is not None:
        raise ValueError('Only clean roots may be perturbed')
    config=SmoothRandomFieldSettings(length_scales=(length_scale,))
    settings=MultiplicativePerturbationSettings(RelativeChangeRange(*SOLUTION_RANGE))
    def validate(original, candidate, manufactured):
        if manufactured is not candidate:
            raise ValueError('MMS field changed identity')
        metrics=physical_checks(original,candidate)
        good=(metrics['source_relative_change']<=MAX_SOURCE_CHANGE
              and metrics['mms_residual']<1e-5 and metrics['boundary_residual']<1e-12)
        return PerturbationValidation(good,'' if good else 'Poisson physical qualification failed',metrics)
    outcome=sample_multiplicative_mms_perturbation(parent,
        settings=settings,seed=seed,
        draw_random_field=lambda field_seed:draw_smooth_random_field(config,seed=field_seed),
        compose_candidate=lambda original,noise,alpha,epsilon:SDKPoissonField(original.amplitude,noise,alpha,epsilon),
        measure_relative_change=lambda original,candidate:relative_change(original.solution(NORM_X),candidate.solution(NORM_X)),
        manufacture_problem=lambda candidate:candidate,validate=validate)
    report={'method':'sdk_epsilon_shifted_multiplicative_mms',
            'noise_sampler':'sdk.draw_smooth_random_field',
            'noise_settings':asdict(config),'settings':asdict(settings),
            'alpha':outcome.alpha,'relative_solution_change':outcome.relative_change,
            'random_field_seed':outcome.random_field_seed,
            'random_field_attempt':outcome.random_field_attempt,'alpha_attempt':outcome.alpha_attempt,
            'physical_checks':dict(outcome.validation.metrics)}
    return outcome.solution,report
