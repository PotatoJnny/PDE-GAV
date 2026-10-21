"""Opt-in, answer-first coherent perturbations of complete weak VPFP carriers.

The frozen moving-frame/innovation physics files are deliberately untouched.
This is an in-band macro perturbation, NOT an NS spectral-complement test or a
projection into the old fixed Hermite basis. All of z, including rho and the
positive floor, is preserved; the same continuous decoder manufactures S.

Like the mature NS adapter, online generation uses a frozen amplitude, not an
expensive per-case source calibration loop. Its physical admission must come
from a separately recorded source-scale smoke, not from this constructor.
"""
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np
import torch

from pde_agent_verifier_solver.sdk import CaseSource, GenerationKind
from .moving_frame import half_modes, split_macro_controls, pack_macro_controls
from .moving_frame_backend import _Resident
from .perturbation_backend import (_validate_request, _provenance, _seeds, _case_ids,
    _snapshot_vpfp_public_case, _validate_pair_public_contract)


SCHEMA = "vpfp_weak_coherent_answer_perturbation_v1"


def _finite_private_metadata(value, path="metadata"):
    """Keep archive/receipt metadata finite and weights-only-loadable.

    NumPy scalar arithmetic is useful for the envelope bounds, but torch.save
    otherwise pickles its scalar constructor into both private answers and
    runtime receipts. Convert metadata only; never transform carrier tensors.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path}: finite private metadata required")
        return value
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise TypeError(f"{path}: private metadata keys must be strings")
        return {key: _finite_private_metadata(item, f"{path}.{key}")
            for key, item in value.items()}
    if type(value) in (list, tuple):
        return type(value)(_finite_private_metadata(item, f"{path}[{index}]")
            for index, item in enumerate(value))
    raise TypeError(f"{path}: builtin private metadata required, got {type(value).__name__}")


@dataclass(frozen=True)
class WeakCoherentPerturbationConfig:
    velocity_over_sqrt_bath: float = .03
    log_diagonal: float = .02
    offdiagonal_over_sqrt_bath: float = .02
    maximum_spatial_mode: int = 2
    schema: str = SCHEMA

    def __post_init__(self):
        if (self.schema != SCHEMA or type(self.maximum_spatial_mode) is not int
                or not 1 <= self.maximum_spatial_mode <= 2):
            raise ValueError("registered coherent weak perturbation layout required")
        for name in ("velocity_over_sqrt_bath", "log_diagonal", "offdiagonal_over_sqrt_bath"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= .1:
                raise ValueError("finite predeclared small weak perturbation budgets required")


def perturbation_identity(config):
    if not isinstance(config, WeakCoherentPerturbationConfig):
        raise TypeError("explicit WeakCoherentPerturbationConfig required")
    return dict(schema=SCHEMA, config=asdict(config),
        implementation_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        family="in_band_coherent_velocity_and_log_cholesky",
        spectral_support_complement=False, full_native_z_preserved=True,
        initial_value_and_first_time_derivative_preserved=True,
        online_source_calibration=False, scientific_admission=False,
        admission_note="requires independent frozen-budget source-scale smoke",
        guidance_training_eligible=False)


def _time_envelope(config):
    """Frozen C1 Hermite controls, normalized by their actual continuous sup.

    The first two values are zero, so native one-sided slope at t0 is exactly
    zero as well. The native interpolation between knots is retained unchanged.
    A small negative first-interval undershoot is included in the absolute
    bound; the interpolant is not claimed to be monotone or nonnegative.
"""
    n = config.control_count
    if n < 3:
        raise ValueError("weak perturbation requires at least three time controls")
    values = np.linspace(0., 1., n)**2
    values[:2] = 0.
    h = 1. / (n - 1)
    slopes = np.concatenate(((values[1:2]-values[:1])/h,
        (values[2:]-values[:-2])/(2*h), (values[-1:]-values[-2:-1])/h))
    maximum, derivative = 0., 0.
    for left, right, dl, dr in zip(values[:-1], values[1:], slopes[:-1], slopes[1:]):
        a, b, c, d = 2*left-2*right+h*(dl+dr), -3*left+3*right-h*(2*dl+dr), h*dl, left
        critical = [0., 1.]
        if abs(a) > 1e-15:
            discriminant = 4*b*b-12*a*c
            if discriminant >= 0:
                critical += [(-2*b+sign*math.sqrt(discriminant))/(6*a) for sign in (-1, 1)]
        elif abs(b) > 1e-15:
            critical.append(-c/(2*b))
        maximum = max(maximum, *(abs(((a*q+b)*q+c)*q+d) for q in critical if 0 <= q <= 1))
        derivative_points = [0., 1.] + ([-b/(3*a)] if abs(a) > 1e-15 else [])
        derivative = max(derivative, *(abs(3*a*q*q+2*b*q+c)/h
            for q in derivative_points if 0 <= q <= 1))
    return values/maximum, derivative/(maximum*(config.time_domain[1]-config.time_domain[0]))


def perturb_carrier(decoder, carrier, condition, *, seed, config):
    """Return a new complete carrier plus a private deterministic budget record.

Velocity budget bounds the pointwise final velocity change, including the
decoder's density-weighted centering (local delta is bounded twice). The
off-diagonal Cholesky coordinate L10 has velocity units; it is NOT a log.
"""
    perturbation_identity(config)
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("private nonnegative int64 perturbation seed required")
    if not isinstance(carrier, dict) or set(carrier) != {"m", "z"}:
        raise ValueError("complete native m/z carrier required")
    if any(not isinstance(v, torch.Tensor) or v.ndim != 2 or len(v) != 1
           or not torch.is_floating_point(v) or not torch.isfinite(v).all() for v in carrier.values()):
        raise ValueError("one finite complete native carrier required")
    physical = torch.as_tensor(condition).detach().cpu().double()
    if physical.shape != (2,) or not torch.isfinite(physical).all() or (physical <= 0).any():
        raise ValueError("positive finite nu and bath temperature required")
    cfg = decoder.macro_config
    if cfg.max_mode < 1:
        raise ValueError("weak coherent perturbation needs a nonconstant spatial mode")
    m = carrier["m"].detach().clone()
    u, local, chol = split_macro_controls(m, cfg)
    values, derivative_bound = _time_envelope(cfg)
    envelope = torch.as_tensor(values, dtype=m.dtype, device=m.device)
    rng = np.random.default_rng(seed)
    modes = half_modes(min(config.maximum_spatial_mode, cfg.max_mode))
    mode = modes[int(rng.integers(len(modes)))]
    index = half_modes(cfg.max_mode).index(mode)
    phase = float(rng.uniform(-math.pi, math.pi))
    wave = m.new_zeros(cfg.modes)
    wave[1+2*index] = math.cos(phase)/math.sqrt(2)
    wave[2+2*index] = math.sin(phase)/math.sqrt(2)
    angles = rng.uniform(-math.pi, math.pi, 2)
    directions = m.new_tensor(np.stack((np.cos(angles), np.sin(angles)), 1))
    thermal_speed = math.sqrt(float(physical[1]))
    velocity_budget = config.velocity_over_sqrt_bath*thermal_speed
    u += .5*velocity_budget*directions[0, :, None]*envelope
    local += .25*velocity_budget*directions[1, :, None, None]*envelope[None, :, None]*wave[None, None, 1:]
    # Independent symmetric directions; no selection using Solver rewards.
    signs = m.new_tensor(rng.choice([-1., 1.], (3, 2)))
    stress_shape = .5*signs[:, 1, None]*wave[None, :]
    stress_shape[:, 0] += .5*signs[:, 0]
    scales = m.new_tensor([config.log_diagonal,
        config.offdiagonal_over_sqrt_bath*thermal_speed, config.log_diagonal])
    chol += scales[:, None, None]*envelope[None, :, None]*stress_shape[:, None, :]
    child_m = pack_macro_controls(u, local, chol, cfg)
    if not torch.isfinite(child_m).all():
        raise FloatingPointError("nonfinite perturbed macro controls; no clipping")
    child = {"m": child_m, "z": carrier["z"].detach().clone()}
    wave_numbers = [abs(2*math.pi*k/length) for k, length in zip(mode, cfg.spatial_domain_lengths)]
    return child, _finite_private_metadata(dict(schema=SCHEMA, seed=seed, spatial_mode=list(mode), spatial_phase=phase,
        config=asdict(config), actual_velocity_bound=velocity_budget,
        actual_log_diagonal_bound=config.log_diagonal,
        actual_cholesky_offdiagonal_bound=config.offdiagonal_over_sqrt_bath*thermal_speed,
        time_envelope_derivative_bound=derivative_bound,
        actual_time_domain=list(cfg.time_domain), actual_time_control_count=cfg.control_count,
        absolute_physical_wave_numbers=wave_numbers,
        coordinate_derivative_bounds=dict(
            velocity_spatial_norm_per_axis=[.25*velocity_budget*k for k in wave_numbers],
            velocity_time_envelope_contribution=velocity_budget*derivative_bound,
            velocity_time_additional_density_weighted_wave_derivative_coefficient=.25*velocity_budget,
            velocity_time_additional_term="coefficient * abs(d_t mean_rho(wave)); not bounded by the envelope alone",
            log_diagonal_spatial_abs_per_axis=[.5*config.log_diagonal*k for k in wave_numbers],
            log_diagonal_time_abs=config.log_diagonal*derivative_bound,
            offdiagonal_spatial_abs_per_axis=[.5*config.offdiagonal_over_sqrt_bath*thermal_speed*k for k in wave_numbers],
            offdiagonal_time_abs=config.offdiagonal_over_sqrt_bath*thermal_speed*derivative_bound),
        full_distribution_derivative_and_source_bound_guaranteed=False,
        online_source_evaluations=0, online_pde_solves=0,
        original_native_z_unchanged=True, density_and_mass_unchanged=True,
        initial_macro_controls_and_slopes_unchanged=True, scientific_admission=False), "perturbation")


def generate_perturbed(runtime, request, config, progress=None, *, captured_parent_sources=None):
    """Compose with a frozen MovingFrameBackend; never invoke its sampler.

The outer WeakIterationBackend verifies actual clean BASE/GUIDED capture
lineage and passes its private source tuple. Native GUIDED records retain the
inner runtime's BASE metadata; never retag their immutable public snapshots.
Without this private adapter argument the standalone runtime remains BASE-only.
Failed files remain private, and no partial batch becomes an active resident.
"""
    del progress
    identity = _finite_private_metadata(perturbation_identity(config), "perturbation_identity")
    _validate_request(request)
    if request.kind is not GenerationKind.PERTURBED:
        raise ValueError("weak perturbation requires a PERTURBED request")
    parent_sources = (("base",) * request.count if captured_parent_sources is None
                      else captured_parent_sources)
    if (type(parent_sources) is not tuple or len(parent_sources) != request.count
            or any(type(source) is not str or source not in ("base", "guided") for source in parent_sources)):
        raise ValueError("one captured clean BASE/GUIDED source is required per perturbation anchor")
    _, _, cohort = _provenance(request)
    seeds = _seeds(request)
    identifiers = _case_ids(request, seeds)
    with runtime._runtime_lock:
        parents = []
        for anchor in request.anchors:
            row = runtime._active(anchor)
            if (anchor.source is not CaseSource.GENERATED
                    or anchor.parent_case_id is not None or "base" not in anchor.tags
                    or anchor.round_index != request.round_index
                    or anchor.visibility is not request.visibility or f"cohort:{cohort}" not in anchor.tags):
                raise ValueError("weak perturbation requires active same-cohort clean BASE anchors")
            parents.append(row)
        if any(key in runtime._issued or (runtime.output_root/key).exists() for key in identifiers):
            raise ValueError("weak perturbed case identifiers cannot be reused")
        pending, records, receipts = {}, [], []
        try:
            for identifier, seed, parent, parent_source in zip(identifiers, seeds, parents, parent_sources, strict=True):
                carrier, budget = perturb_carrier(runtime.decoder, parent.carrier, parent.condition,
                    seed=seed, config=config)
                # Validate before numerical record construction or any archive
                # is written, including when a custom perturbation hook is used.
                budget = _finite_private_metadata(budget, "perturbation")
                condition = parent.condition.detach().cpu().clone()
                record = runtime._record(identifier, carrier, condition, request, cohort)
                case = replace(record.case, source=CaseSource.PERTURBED, parent_case_id=parent.case.case_id,
                    tags=("vpfp", "moving-frame-A", "perturbed", f"{parent_source}-perturbed", f"cohort:{cohort}"))
                record = replace(record, case=case)
                _validate_pair_public_contract(parent.case, case)
                for name in ("rho0", "momentum0", "kinetic_energy0"):
                    if not np.array_equal(parent.case.public_data["arrays"][name], case.public_data["arrays"][name]):
                        raise ValueError("weak perturbation changed its initial public moments")
                directory = runtime.output_root/identifier
                directory.mkdir(mode=0o700)
                lineage = dict(parent_case_id=parent.case.case_id, parent_answer_sha256=parent.answer_sha256,
                    parent_source=parent_source, generation_kind="perturbed", perturbation=budget)
                answer = directory/"private-answer.pt"
                with answer.open("xb") as handle:
                    torch.save(dict(schema=SCHEMA+"_private_answer", carrier=runtime.decoder.carrier_to_payload(carrier),
                        condition=condition, sampler_identity=runtime.sampler_identity,
                        sampler_identity_sha256=runtime.sampler_identity_sha256,
                        perturbation_identity=identity, **lineage), handle)
                digest = sha256(answer.read_bytes()).hexdigest()
                receipt = dict(case_id=identifier, private_answer_sha256=digest, decoder_schema=runtime.decoder.schema,
                    sampler_identity_sha256=runtime.sampler_identity_sha256, source="actual_A_autodiff",
                    pde_solves=0, scientific_admission=False, perturbation_identity=identity, **lineage)
                if runtime.batched_source or runtime.adaptive_source_only:
                    receipt.update(source="actual_A_batched_exact", numerical_implementation=runtime.source_implementation)
                (directory/"private-receipt.json").write_text(json.dumps(receipt, sort_keys=True, allow_nan=False))
                pending[identifier] = _Resident(case, _snapshot_vpfp_public_case(case), carrier, condition, digest)
                records.append(record); receipts.append(receipt)
        except BaseException as error:
            for identifier in identifiers:
                runtime._source_evaluators.pop(identifier, None)
            failure = runtime.output_root/f"failed-weak-perturbation-batch-{request.seed}.json"
            with failure.open("x") as handle:
                json.dump(dict(schema=SCHEMA, error=repr(error), case_ids=identifiers,
                    parent_case_ids=[p.case.case_id for p in parents], attempted_count=len(records)+1,
                    requested_count=request.count, published_count=0, perturbation_identity=identity), handle)
            raise
        runtime._resident_cases.update(pending); runtime._issued.update(identifiers)
        runtime.generation_receipts.extend(receipts)
        return tuple(records)
