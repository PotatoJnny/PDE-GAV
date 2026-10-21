"""Versioned continuous moving-frame density; no finite-Hermite re-projection.

The full native value state z, including rho, is retained.  Its *whole* velocity
density (squared amplitude AND Maxwellian floor) is standardized with exact
Hermite-neighbour moments, then affinely assigned the macro coordinate's mean
and central covariance.  This is a decoder-only A/B intervention, not a claim
that the carrier is a pure shape quotient or that macro truncation is lossless.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import importlib
import json
import math

import torch
from torch import nn

from .value_control_diffusion import VPFPValueEDMConfig, lift_value_controls
from .structured_diffusion import VPFPStructuredEDMConfig


SCHEMA = "vpfp_continuous_moving_frame_carrier_v1"


@dataclass(frozen=True)
class MovingFrameMacroConfig:
    control_count: int = 33
    max_mode: int = 6
    time_domain: tuple = (0., 1.)
    spatial_domain_lengths: tuple = (2*math.pi, 2*math.pi)
    schema: str = "vpfp_physical_moving_frame_macro_v1"

    def __post_init__(self):
        object.__setattr__(self, "time_domain", tuple(self.time_domain))
        object.__setattr__(self, "spatial_domain_lengths", tuple(self.spatial_domain_lengths))
        if (type(self.control_count) is not int or self.control_count < 2
                or type(self.max_mode) is not int or self.max_mode < 0
                or len(self.time_domain) != 2 or len(self.spatial_domain_lengths) != 2
                or not all(math.isfinite(float(x)) for x in (*self.time_domain, *self.spatial_domain_lengths))
                or self.time_domain[1] <= self.time_domain[0]
                or min(self.spatial_domain_lengths) <= 0
                or self.schema != "vpfp_physical_moving_frame_macro_v1"):
            raise ValueError("invalid moving-frame macro config")

    @property
    def modes(self):
        return (2*self.max_mode+1)**2

    @property
    def macro_dimension(self):
        return 5*self.control_count*self.modes

    @property
    def latent_dimension(self):
        return self.macro_dimension

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, payload):
        if set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("unexpected moving-frame config fields")
        return cls(**payload)


def _finite(values, name):
    if not torch.is_floating_point(values) or not bool(torch.isfinite(values).all()):
        raise ValueError(f"{name} must be finite floating point")
    return values


def half_modes(max_mode):
    return tuple((kx, ky) for kx in range(-max_mode, max_mode+1)
                 for ky in range(-max_mode, max_mode+1)
                 if kx > 0 or (kx == 0 and ky > 0))


def macro_fourier_basis(points_xy, config):
    """[N,M], same normalized DC,cos,sin ordering as _fourier_basis."""
    _finite(points_xy, "xy")
    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        raise ValueError("xy must have shape [N,2]")
    dc = torch.ones_like(points_xy[:, :1])
    if config.max_mode == 0:
        return dc
    modes = torch.tensor(half_modes(config.max_mode), device=points_xy.device, dtype=points_xy.dtype)
    phase = 2*torch.pi*(points_xy[:, :1]*modes[None, :, 0]/config.spatial_domain_lengths[0]
                       +points_xy[:, 1:2]*modes[None, :, 1]/config.spatial_domain_lengths[1])
    return torch.cat((dc, math.sqrt(2)*torch.stack((phase.cos(),phase.sin()), -1).flatten(1)), 1)


def spatial_grid(config, count, *, device, dtype=torch.float64, offset=0.):
    if type(count) is not int or count < 2*config.max_mode+1:
        raise ValueError("spatial grid aliases macro Fourier coordinates")
    axes = [(torch.arange(count, device=device, dtype=dtype)+offset)*length/count
            for length in config.spatial_domain_lengths]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 2)


def split_macro_controls(values, config):
    _finite(values, "macro controls")
    if values.ndim != 2 or values.shape[1] != config.macro_dimension:
        raise ValueError("invalid macro coordinate shape")
    b, t, m = len(values), config.control_count, config.modes
    return (values[:, :2*t].reshape(b, 2, t),
            values[:, 2*t:2*t*m].reshape(b, 2, t, m-1),
            values[:, 2*t*m:].reshape(b, 3, t, m))


def pack_macro_controls(global_velocity, local_velocity, log_cholesky, config):
    b, t, m = len(global_velocity), config.control_count, config.modes
    if (global_velocity.shape != (b, 2, t) or local_velocity.shape != (b, 2, t, m-1)
            or log_cholesky.shape != (b, 3, t, m)):
        raise ValueError("invalid macro block shapes")
    result = torch.cat((global_velocity.flatten(1), local_velocity.flatten(1), log_cholesky.flatten(1)), 1)
    split_macro_controls(result, config)
    return result


def restrict_macro_controls(values, source_config, target_config):
    """Exact coordinate subset for the predeclared low-to-high mode sweep."""
    if (target_config.max_mode > source_config.max_mode
            or target_config.control_count != source_config.control_count
            or target_config.time_domain != source_config.time_domain
            or target_config.spatial_domain_lengths != source_config.spatial_domain_lengths):
        raise ValueError("macro restriction requires identical time/domain and lower spatial order")
    u, local, chol = split_macro_controls(values,source_config)
    lookup = {pair:i for i,pair in enumerate(half_modes(source_config.max_mode))}
    indices = [2*lookup[pair]+phase for pair in half_modes(target_config.max_mode) for phase in (0,1)]
    take = torch.tensor(indices,device=values.device,dtype=torch.long)
    take_chol = torch.cat((torch.zeros(1,device=values.device,dtype=torch.long),take+1))
    return pack_macro_controls(u,local.index_select(-1,take),chol.index_select(-1,take_chol),target_config)


def interpolate_controls(values, time, config):
    """[B,C,T,M] -> [N,C,M]; same value-to-slope lift as native z.

    No unique-time caching: independent point rows retain their own dt graph.
    """
    _finite(values, "time controls")
    _finite(time, "physical time")
    if (values.ndim != 4 or values.shape[-2] != config.control_count
            or time.ndim != 1 or values.shape[0] not in (1, len(time))):
        raise ValueError("invalid time/control row layout")
    t0, t1 = config.time_domain
    if bool(((time < t0) | (time > t1)).any()):
        raise ValueError("moving-frame query outside time domain")
    h = 1./(config.control_count-1)
    slopes = torch.cat(((values[..., 1:2, :]-values[..., :1, :])/h,
                        (values[..., 2:, :]-values[..., :-2, :])/(2*h),
                        (values[..., -1:, :]-values[..., -2:-1, :])/h), -2)
    p = (time-t0)/(t1-t0)*(config.control_count-1)
    lo = p.floor().long().clamp(max=config.control_count-2)
    q = p-lo
    row = torch.zeros_like(lo) if len(values) == 1 else torch.arange(len(time), device=time.device)
    def at(v, shift):
        return v[row, :, lo+shift, :]
    return ((2*q**3-3*q**2+1)[:, None, None]*at(values, 0)
            +(q**3-2*q**2+q)[:, None, None]*h*at(slopes, 0)
            +(-2*q**3+3*q**2)[:, None, None]*at(values, 1)
            +(q**3-q**2)[:, None, None]*h*at(slopes, 1))


def log_cholesky(covariance):
    _finite(covariance, "central covariance")
    if covariance.shape[-2:] != (2, 2):
        raise ValueError("central covariance must end in 2x2")
    if not torch.allclose(covariance, covariance.transpose(-1, -2), rtol=0., atol=0.):
        raise ValueError("central covariance must be symmetric")
    factor, info = torch.linalg.cholesky_ex(covariance)
    if bool(info.any()):
        raise ValueError("central covariance is not SPD; no jitter or clipping allowed")
    return torch.stack((factor[..., 0, 0].log(), factor[..., 1, 0], factor[..., 1, 1].log()), -1)


def cholesky_from_log(values):
    _finite(values, "log Cholesky")
    if values.shape[-1] != 3:
        raise ValueError("log Cholesky order is [log L00,L10,log L11]")
    d0, off, d1 = values[..., 0].exp(), values[..., 1], values[..., 2].exp()
    factor = torch.stack((d0, torch.zeros_like(d0), off, d1), -1).reshape(*values.shape[:-1], 2, 2)
    _finite(factor, "Cholesky factor")
    if bool(((d0 <= 0) | (d1 <= 0)).any()):
        raise ValueError("Cholesky exponential underflow; no floor or clipping allowed")
    return factor


def _legacy_modules():
    from .perturbation_backend import _legacy_api
    _legacy_api()
    return (importlib.import_module("agent_verifier.vpfp.continuous_macros"),
            importlib.import_module("agent_verifier.vpfp.hermite_density"))


def value_config_from_decoder(base_decoder):
    """Layout-only adapter; no fitted geometry or physical scaling is changed."""
    c = base_decoder.config
    return VPFPValueEDMConfig(VPFPStructuredEDMConfig(control_count=c.structured_time_control_count,
        hermite_channels=c.structured_local_channels, rho_max_mode=c.rho_max_mode_x,
        kinetic_max_mode=c.structured_local_spatial_mode, inactive_dim=c.structured_kinetic_global_dim,
        grid_size=2*max(c.rho_max_mode_x, c.structured_local_spatial_mode)+1,
        width=max(256, 2*(c.structured_local_channels+1)), time_domain=c.time_domain,
        collision_nu_range=c.collision_nu_range, temperature_range=c.temperature_range))


def source_local_moments(base_fields, base_decoder):
    """Exact N+1 neighbour recurrence in FP64, with the source's full floor."""
    macros, _ = _legacy_modules()
    moments = macros.torch_squared_hermite_moments(torch,
        base_fields["hermite_coefficients"].double(), modes=base_decoder.hermite_modes,
        temperature=base_fields["equation_condition"][:, 1].double(),
        floor_mixture=base_decoder.config.velocity_floor_mixture)
    mean = moments["mean_velocity"]
    second = moments["raw_second_moment"]
    second = torch.stack((second[:, 0], second[:, 1], second[:, 1], second[:, 2]), -1).reshape(-1, 2, 2)
    cov = second-mean[..., :, None]*mean[..., None, :]
    chol_coordinates = log_cholesky(cov)
    return {"mean": mean, "covariance": cov, "factor": cholesky_from_log(chol_coordinates),
            "log_cholesky": chol_coordinates, "second_moment": second}


def weighted_velocity_offset(local_coefficients, rho_coefficients, base_decoder, config):
    """Exact spatial Parseval integral; native rho coefficients multiply cos/sin.

    The density gauge is spatially constant. Rho's non-DC coefficients may be
    nonlinear in time, but its spatial support is finite, so no quadrature or
    shared-time approximation is needed for this weighted center.
    """
    modes = {pair: i for i, pair in enumerate(half_modes(config.max_mode))}
    local_indices, rho_indices = [], []
    for j, pair in enumerate(base_decoder.rho_half_modes):
        if tuple(pair) in modes:
            local_indices.extend((2*modes[tuple(pair)],2*modes[tuple(pair)]+1))
            rho_indices.extend((2*j,2*j+1))
    left = torch.tensor(local_indices,device=local_coefficients.device,dtype=torch.long)
    right = torch.tensor(rho_indices,device=local_coefficients.device,dtype=torch.long)
    rho = rho_coefficients.flatten(1).to(local_coefficients).index_select(-1,right)
    return (local_coefficients.index_select(-1,left)*rho[:,None,:]).sum(-1)/(math.sqrt(2)*float(base_decoder.config.equation.background_density))


class MovingFrameDecoder(nn.Module):
    """Point-query decoder for {'m': physical macro coordinates, 'z': native values}.

    This has a NEW density contract. In particular it deliberately does not
    return `hermite_coefficients` falsely describing the transformed density.
    Rho/phi/electric are unchanged. `momentum`, `central_covariance` and K are
    analytic full-velocity integrals, not finite velocity-box estimates.
    """
    schema = SCHEMA

    def __init__(self, base_decoder, config, *, value_config):
        super().__init__()
        self.base_decoder, self.macro_config, self.value_config = base_decoder, config, value_config
        if not isinstance(config, MovingFrameMacroConfig) or not isinstance(value_config, VPFPValueEDMConfig):
            raise TypeError("explicit macro and native value config required")
        base = base_decoder.config
        if (not base.structured_spectral_latent or not base.structured_direct_coefficients
                or base.structured_residual_scale != 0 or base.structured_time_basis != "piecewise_cubic_hermite_v1"
                or value_config.full_dimension != base.latent_dim
                or value_config.control_count != base.structured_time_control_count
                or value_config.hermite_channels != base.structured_local_channels
                or value_config.kinetic_max_mode != base.structured_local_spatial_mode
                or value_config.rho_max_mode != base.rho_max_mode_x or base.rho_max_mode_x != base.rho_max_mode_y
                or tuple(config.time_domain) != tuple(base.time_domain)
                or tuple(value_config.time_domain) != tuple(base.time_domain)
                or tuple(config.spatial_domain_lengths) != tuple(base.equation.x_domain)):
            raise ValueError("moving-frame decoder requires the bound direct native value ABI")
        if float(base.equation.background_density) <= 0:
            raise ValueError("positive background density required")

    def source_fields(self, points, z, condition):
        if z.ndim == 1:
            z = z[None]
        full = lift_value_controls(z, self.value_config)
        native = self.base_decoder.fields(points, full, equation_condition=condition)
        if self.base_decoder.config.rho_positivity_gauge == "phase_invariant_group_l1_v2":
            query, raw, constant_zero = self._native_rho_groups(points, full)
            if bool(constant_zero.any()):
                rho, phi, electric, coefficients = self._rho_with_constant_zero_groups(query, raw, constant_zero)
                native = {**native, "rho": rho, "phi": phi, "electric": electric,
                          "rho_coefficients": coefficients, "f": rho*native["g"]}
        return native

    def _native_rho_groups(self, points, full):
        """Replay ONLY the native rho contraction, including native dtype/scale.

        A fixed zero Fourier pair throughout all time controls is a constant
        zero function of coordinates. Its Euclidean norm has zero coordinate
        derivatives, although differentiating torch's norm at zero twice gives
        0/0. This is not a smoothing of a nonconstant trajectory crossing zero.
        """
        base = self.base_decoder
        parameter = next(base.parameters())
        query = points.to(device=parameter.device, dtype=parameter.dtype)
        full = full.to(parameter)
        count, groups = len(query), len(base.rho_half_modes)
        if full.ndim != 2 or len(full) not in (1, count):
            raise ValueError("native rho replay requires one or one-per-query full row")
        shared = len(full) == 1 and count > 1
        if len(full) == 1 and not shared:
            full = full.expand(count, -1)
        end = int(base.structured_layout["rho_latent_dim"])
        rho_values = (float(base.config.latent_input_scale)*full[:, :end]).reshape(
            len(full), int(base.structured_layout["rho_time_feature_count"]), 2*groups)
        tau = (query[:, :1]-float(base.config.time_domain[0]))/float(
            base.config.time_domain[1]-base.config.time_domain[0])
        basis, normalization = base._structured_time_basis(tau, int(base.config.structured_rho_time_order))
        if shared:
            raw = torch.einsum("nt,tm->nm", basis, rho_values[0])
        else:
            raw = torch.einsum("nt,ntm->nm", basis, rho_values)
        raw = (raw/float(normalization)).reshape(count, groups, 2)
        constant_zero = (rho_values.reshape(len(full), -1, groups, 2) == 0).all(1).all(-1)
        if shared:
            constant_zero = constant_zero.expand(count, -1)
        return query, raw, constant_zero

    def _rho_with_constant_zero_groups(self, points, raw, constant_zero):
        # Mask BEFORE norm; masking a norm(0) result afterwards still sends
        # 0/0 through its double backward. Nonzero groups keep the native path.
        dummy = torch.zeros_like(raw)
        dummy[..., 0] = 1.
        safe_raw = torch.where(constant_zero[..., None], dummy, raw)
        norms = torch.linalg.vector_norm(safe_raw, dim=-1)
        norms = torch.where(constant_zero, torch.zeros_like(norms), norms)
        denominator = 1.0+torch.sum(norms, dim=1, keepdim=True)[..., None]
        base = self.base_decoder
        scale = float(base.config.rho_contrast)*float(base.config.equation.background_density)
        coefficients = scale*raw/denominator
        kappa = base._rho_kappa.to(dtype=points.dtype, device=points.device)
        phase = points[:, 1:2]*kappa[None, :, 0]+points[:, 2:3]*kappa[None, :, 1]
        cosine, sine = torch.cos(phase), torch.sin(phase)
        wave = coefficients[:, :, 0]*cosine+coefficients[:, :, 1]*sine
        rho = float(base.config.equation.background_density)+torch.sum(wave, dim=-1)
        kappa_squared = torch.sum(kappa.square(), dim=-1)
        phi = torch.sum(wave/kappa_squared[None, :], dim=-1)
        electric = torch.stack([torch.sum(
            (coefficients[:, :, 0]*kappa[None, :, axis]*sine
             -coefficients[:, :, 1]*kappa[None, :, axis]*cosine)/kappa_squared[None, :], dim=-1)
            for axis in (0, 1)], dim=-1)
        return rho, phi, electric, coefficients

    def validate_source_gradient_domain(self, points, carrier):
        """Reject a possible nonsmooth rho-gauge crossing for d(source)/dq.

        r/(1+||r||) is C1 but generally not C2 when a nonconstant r crosses
        zero. A finite source value is not a certificate for its derivative.
        No epsilon or invented higher derivative is supplied at that crossing.
        """
        if self.base_decoder.config.rho_positivity_gauge != "phase_invariant_group_l1_v2":
            return
        z = carrier["z"]
        if z.ndim == 1:
            z = z[None]
        _, raw, constant_zero = self._native_rho_groups(points, lift_value_controls(z, self.value_config))
        norms = torch.linalg.vector_norm(raw.detach(), dim=-1)
        if bool(((norms == 0) & ~constant_zero).any()):
            raise ValueError("source gradient undefined/unsupported at a nonconstant rho-group zero crossing")

    def macro_at(self, points, m, base_fields):
        config = self.macro_config
        if m.ndim == 1:
            m = m[None]
        m = m.to(device=points.device, dtype=torch.float64)
        u, local, chol = split_macro_controls(m, config)
        p = points.double()
        global_u = interpolate_controls(u[..., None], p[:, 0], config)[..., 0]
        local = interpolate_controls(local, p[:, 0], config)
        basis = macro_fourier_basis(p[:, 1:3], config)
        offset = weighted_velocity_offset(local, base_fields["rho_coefficients"], self.base_decoder, config)
        velocity = global_u+torch.einsum("ncm,nm->nc", local, basis[:, 1:])-offset
        chol_values = torch.einsum("ncm,nm->nc", interpolate_controls(chol, p[:, 0], config), basis)
        factor = cholesky_from_log(chol_values)
        covariance = factor@factor.transpose(-1, -2)
        _finite(covariance, "generated central covariance")
        return {"mean": velocity, "global_velocity": global_u, "weighted_offset": offset,
                "factor": factor, "covariance": covariance}

    def fields(self, points, carrier, *, equation_condition):
        _finite(points, "points")
        if points.ndim != 2 or points.shape[1] != 5 or set(carrier) != {"m", "z"}:
            raise ValueError("moving-frame query needs [N,5] points and exact m/z carrier")
        points = points.to(device=next(self.base_decoder.parameters()).device)
        native = self.source_fields(points, carrier["z"], equation_condition)
        source = source_local_moments(native, self.base_decoder)
        target = self.macro_at(points, carrier["m"], native)
        # v_source = mu_z + L_z L_target^-1 (v-u_target), with both
        # distributions' complete floor transformed by the same affine map.
        centered = torch.linalg.solve_triangular(target["factor"],
            (points[:, 3:5].double()-target["mean"])[..., None], upper=False)[..., 0]
        source_v = source["mean"]+(source["factor"]@centered[..., None])[..., 0]
        _, density = _legacy_modules()
        g_source = density.torch_squared_hermite_density(torch, source_v,
            native["hermite_coefficients"].double(), modes=self.base_decoder.hermite_modes,
            temperature=native["equation_condition"][:, 1].double(),
            floor_mixture=self.base_decoder.config.velocity_floor_mixture)
        jacobian = source["factor"].diagonal(dim1=-2, dim2=-1).prod(-1)/target["factor"].diagonal(dim1=-2, dim2=-1).prod(-1)
        rho = native["rho"].double()
        g = g_source*jacobian
        momentum = rho[:, None]*target["mean"]
        kinetic = .5*rho*(target["mean"].square().sum(-1)+target["covariance"].diagonal(dim1=-2, dim2=-1).sum(-1))
        result = {"rho": rho, "g": g, "f": rho*g, "phi": native["phi"], "electric": native["electric"],
            "momentum": momentum, "kinetic_energy": kinetic, "velocity": target["mean"],
            "central_covariance": target["covariance"], "global_velocity": target["global_velocity"],
            "source_mean_velocity": source["mean"], "source_central_covariance": source["covariance"],
            "source_hermite_coefficients": native["hermite_coefficients"],
            "rho_coefficients": native["rho_coefficients"], "equation_condition": native["equation_condition"]}
        for key, value in result.items():
            _finite(value, key)
        return result

    def identity(self):
        data = self.base_decoder.config.to_dict()
        return {"schema": self.schema, "macro_config": self.macro_config.to_dict(),
                "value_config": self.value_config.to_dict(),
                "source_decoder_config_sha256": sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                "floor": "entire_source_density_affine_pushforward",
                "global_velocity": "rho_weighted_Parseval_exact",
                "shape_projection": "none_full_continuous_native_carrier"}

    def carrier_to_payload(self, carrier):
        if set(carrier) != {"m", "z"}:
            raise ValueError("unexpected carrier fields")
        split_macro_controls(carrier["m"], self.macro_config)
        lift_value_controls(carrier["z"], self.value_config)
        if len(carrier["m"]) != len(carrier["z"]):
            raise ValueError("macro/micro batch mismatch")
        return {"identity": self.identity(), "m": carrier["m"].detach().cpu().clone(),
                "z": carrier["z"].detach().cpu().clone()}

    def carrier_from_payload(self, payload):
        if set(payload) != {"identity", "m", "z"} or payload["identity"] != self.identity():
            raise ValueError("moving-frame carrier identity mismatch")
        carrier = {"m": payload["m"], "z": payload["z"]}
        self.carrier_to_payload(carrier)
        parameter = next(self.base_decoder.parameters())
        return {key: value.to(device=parameter.device).clone() for key, value in carrier.items()}


@torch.no_grad()
def extract_macro_controls(base_decoder, z_value, condition, config, *, value_config, fit_grid=33, chunk_size=256):
    """Fit physical m(z) on an explicit spatial grid at macro control times.

    Global U and projection of non-bandlimited u/log-Cholesky use the stated
    grid; the decoder's rho-weighted recentering itself is analytic. No
    eigenvalue floor/clipping, held-out reference data, or learned fit enters.
    """
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("positive extraction chunk size required")
    decoder = MovingFrameDecoder(base_decoder, config, value_config=value_config)
    if z_value.ndim == 1:
        z_value = z_value[None]
    if condition.ndim == 1:
        condition = condition[None].expand(len(z_value), -1)
    if condition.shape != (len(z_value), 2):
        raise ValueError("one physical condition per source case required")
    parameter = next(base_decoder.parameters())
    xy = spatial_grid(config, fit_grid, device=parameter.device)
    basis = macro_fourier_basis(xy, config)
    times = torch.linspace(*config.time_domain, config.control_count, device=parameter.device, dtype=torch.float64)
    packed, cov_min, cov_max = [], math.inf, 0.
    for case in range(len(z_value)):
        us, locals_, chols = [], [], []
        z = z_value[case:case+1].to(parameter)
        for time in times:
            query = torch.cat((time.expand(len(xy), 1), xy, torch.zeros_like(xy)), 1)
            rs, ms, cs = [], [], []
            for part in query.split(chunk_size):
                fields = decoder.source_fields(part, z, condition[case].to(parameter))
                moments = source_local_moments(fields, base_decoder)
                rs.append(fields["rho"].double()); ms.append(moments["mean"]); cs.append(moments["log_cholesky"])
                eig = torch.linalg.eigvalsh(moments["covariance"])
                cov_min, cov_max = min(cov_min, float(eig.min())), max(cov_max, float(eig.max()))
            rho, mean, chol = torch.cat(rs), torch.cat(ms), torch.cat(cs)
            us.append((rho[:, None]*mean).sum(0)/rho.sum())
            locals_.append(mean.T@basis[:, 1:]/len(xy))
            chols.append(chol.T@basis/len(xy))
        packed.append(pack_macro_controls(torch.stack(us, 1)[None], torch.stack(locals_, 1)[None],
                                          torch.stack(chols, 1)[None], config))
    return torch.cat(packed), {"schema": "vpfp_moving_frame_macro_extraction_v1", "fit_grid": fit_grid,
        "control_count": config.control_count, "maximum_mode": config.max_mode,
        "source_covariance_eigenvalue_min": cov_min, "source_covariance_eigenvalue_max": cov_max,
        "log_cholesky_order": ["log_L00", "L10", "log_L11"], "physical_velocity_units": True,
        "macro_reconstruction_exact": False, "scientific_regime_qualified": False,
        "macro_fit": "uniform_spatial_Fourier_projection_at_time_controls",
        "velocity_floor": "included_in_source_exact_moments_and_whole_density_pushforward"}
