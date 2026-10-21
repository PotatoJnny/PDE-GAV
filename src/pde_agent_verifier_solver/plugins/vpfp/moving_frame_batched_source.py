"""Case-local numerical A source with reusable t/x/y geometry and first jets.

The first implementation deliberately JVPs the existing geometry, preserving
its native-FP32 / affine-FP64 split. Only the complete velocity density and its
chain rule are assembled analytically. It does not cache answers, round query
coordinates, complete Cartesian products, or approximate the represented f.
"""
from __future__ import annotations

from collections import OrderedDict
import importlib
import numpy as np  # Preserve the frozen physical-module initialization order.
import torch

from .moving_frame import MovingFrameDecoder, source_local_moments
from .moving_frame_backend import _query, manufacture_moving_frame_source
from .value_control_diffusion import lift_value_controls
from .perturbation_backend import _legacy_api


def _finite_fields(fields):
    for name, value in fields.items():
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f'nonfinite moving-frame numerical field: {name}')
    return fields


class MovingFrameSourceEvaluator:
    """One immutable cloned m/z/condition; caller serializes concurrent access.

    Numerical source/fields results are detached. ``macros`` retains ordinary
    coordinate autograd, and ``source(create_graph=True)`` delegates to the old
    implementation for higher coordinate derivatives of the frozen case. It
    does NOT reconnect gradients to the caller's original m/z tensors; parameter
    guidance must use the original function with its original inputs. The
    decoder is shared, but mutations invalidate this evaluator rather than
    silently reusing stale geometry.
    """

    def __init__(self, decoder, carrier, condition, *, geometry_cache_size=512,
                 geometry_batch_size=32, phase_chunk_size=8192):
        if type(decoder) is not MovingFrameDecoder:
            raise TypeError('requires the original MovingFrameDecoder')
        for name, value in (('geometry_cache_size', geometry_cache_size),
                            ('geometry_batch_size', geometry_batch_size), ('phase_chunk_size', phase_chunk_size)):
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if not isinstance(carrier, dict) or set(carrier) != {'m', 'z'}:
            raise ValueError('one complete m/z carrier required')
        cfg = decoder.base_decoder.config
        if (not cfg.structured_direct_coefficients or not cfg.structured_spectral_latent
                or cfg.structured_residual_scale != 0 or cfg.structured_time_basis != 'piecewise_cubic_hermite_v1'
                or cfg.rho_positivity_gauge not in ('phase_invariant_group_l1_v2', 'coordinate_l1_v1')):
            raise NotImplementedError('numerical batching requires the direct cubic-Hermite carrier')
        parameter = next(decoder.parameters())
        self.decoder = decoder
        self._carrier = {}
        for name, value in carrier.items():
            tensor = torch.as_tensor(value, device=parameter.device)
            if tensor.ndim == 1:
                tensor = tensor[None]
            if tensor.ndim != 2 or len(tensor) != 1 or not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
                raise ValueError('batch evaluator owns exactly one finite floating carrier')
            self._carrier[name] = tensor.detach().clone()
        decoder.carrier_to_payload(self._carrier)  # Original complete-state validation once.
        physical = torch.as_tensor(condition, device=parameter.device)
        if physical.shape not in ((2,), (1, 2)):
            raise ValueError('one nu/T condition required; per-query cases are unsupported')
        self._condition = physical.reshape(2).detach().clone()
        decoder.base_decoder._equation_condition_rows(self._condition, count=1)
        if not bool(torch.isfinite(self._condition).all()):
            raise ValueError('nonfinite equation condition')
        self.geometry_cache_size, self.geometry_batch_size = geometry_cache_size, geometry_batch_size
        self.phase_chunk_size = phase_chunk_size
        self._cache = OrderedDict()
        self._stats = dict(geometry_hits=0, geometry_misses=0, geometry_evictions=0,
                           geometry_value_batches=0, geometry_jet_batches=0, geometry_rows_computed=0,
                           requested_phase_points=0, assembled_phase_points=0)
        modes = decoder.base_decoder.hermite_modes
        self._indices = tuple(torch.tensor([p[axis] for p in modes], device=parameter.device, dtype=torch.long) for axis in (0, 1))
        self._degree = max(max(pair) for pair in modes)
        _legacy_api()
        self._analytic = importlib.import_module('agent_verifier.vpfp.analytic_source')
        self._decoder_version = self._version()

    def _version(self):
        model = self.decoder
        return (id(model.base_decoder.config), id(model.value_config), id(model.macro_config),
                tuple((name, id(value), value._version, value.dtype, value.device)
                      for name, value in (*model.named_parameters(), *model.named_buffers())))

    def _points(self, points, width):
        if self._version() != self._decoder_version:
            raise RuntimeError('shared decoder changed; construct a new immutable-case evaluator')
        query = _query(self.decoder, points, width)
        t0, t1 = self.decoder.macro_config.time_domain
        if bool(((query[:, 0] < t0) | (query[:, 0] > t1)).any()):
            raise ValueError('query outside the original time domain')
        return query

    def _geometry(self, txy):
        phase = torch.cat((txy, txy.new_zeros((len(txy), 2))), dim=1)
        native = self.decoder.source_fields(phase, self._carrier['z'], self._condition)
        original = source_local_moments(native, self.decoder.base_decoder)
        target = self.decoder.macro_at(phase, self._carrier['m'], native)
        rho = native['rho'].double()
        u, cov = target['mean'], target['covariance']
        return {'rho': rho, 'phi': native['phi'], 'electric': native['electric'],
                'momentum': rho[:, None]*u,
                'kinetic_energy': .5*rho*(u.square().sum(-1)+cov.diagonal(dim1=-2, dim2=-1).sum(-1)),
                'velocity': u, 'central_covariance': cov, 'global_velocity': target['global_velocity'],
                'source_mean_velocity': original['mean'], 'source_central_covariance': original['covariance'],
                'source_hermite_coefficients': native['hermite_coefficients'],
                'rho_coefficients': native['rho_coefficients'], 'equation_condition': native['equation_condition'],
                '_source_factor': original['factor'], '_target_factor': target['factor']}

    def _prepare(self, txy, derivatives):
        with torch.no_grad():
            if derivatives:
                jets = []
                for axis in range(3):
                    tangent = torch.zeros_like(txy)
                    tangent[:, axis] = 1.
                    values, jet = torch.func.jvp(self._geometry, (txy,), (tangent,))
                    jets.append(jet)
                derivative = {key: torch.stack([jet[key] for jet in jets], 1).detach() for key in values}
                _finite_fields(derivative)
                self._stats['geometry_jet_batches'] += 1
            else:
                values, derivative = self._geometry(txy), None
                self._stats['geometry_value_batches'] += 1
            values = {key: value.detach() for key, value in values.items()}
            _finite_fields(values)
            self._stats['geometry_rows_computed'] += len(txy)
            return values, derivative

    def _lookup(self, query, derivatives):
        exact = np.ascontiguousarray(query.detach().cpu().numpy(), dtype=np.float64)
        unique, inverse = np.unique(exact, axis=0, return_inverse=True)
        keys = [row.tobytes() for row in unique]
        entries, missing = [None]*len(unique), []
        for index, key in enumerate(keys):
            entry = self._cache.get(key)
            if entry is not None and (not derivatives or entry[1] is not None):
                entries[index] = entry
                self._cache.move_to_end(key)
                self._stats['geometry_hits'] += 1
            else:
                missing.append(index)
                self._stats['geometry_misses'] += 1
        for start in range(0, len(missing), self.geometry_batch_size):
            positions = missing[start:start+self.geometry_batch_size]
            coords = torch.as_tensor(unique[positions], device=query.device, dtype=torch.float64)
            values, jets = self._prepare(coords, derivatives)
            for row, index in enumerate(positions):
                # Clone each row: a single LRU entry must not retain a whole batch.
                entry = ({k: v[row].clone() for k, v in values.items()},
                         None if jets is None else {k: v[row].clone() for k, v in jets.items()})
                entries[index] = entry
                self._cache[keys[index]] = entry
                self._cache.move_to_end(keys[index])
                if len(self._cache) > self.geometry_cache_size:
                    self._cache.popitem(last=False)
                    self._stats['geometry_evictions'] += 1
        take = torch.as_tensor(inverse, device=query.device, dtype=torch.long)
        values = {k: torch.stack([entry[0][k] for entry in entries]).index_select(0, take) for k in entries[0][0]}
        jets = ({k: torch.stack([entry[1][k] for entry in entries]).index_select(0, take) for k in entries[0][1]}
                if derivatives else None)
        return values, jets

    def _velocity(self, query, geometry, jets):
        source, target = geometry['_source_factor'], geometry['_target_factor']
        centered = torch.linalg.solve_triangular(target, (query[:, 3:5]-geometry['velocity'])[..., None], upper=False)[..., 0]
        w = geometry['source_mean_velocity']+(source@centered[..., None])[..., 0]
        theta = geometry['equation_condition'][:, 1].double()
        c = geometry['source_hermite_coefficients'].double()
        x = self._analytic._hermite_1d_jet(torch, w[:, 0], self._degree, torch.sqrt(2*theta))
        y = self._analytic._hermite_1d_jet(torch, w[:, 1], self._degree, torch.sqrt(2*theta))
        x, y = tuple(v.index_select(1, self._indices[0]) for v in x), tuple(v.index_select(1, self._indices[1]) for v in y)
        basis = x[0]*y[0]
        amplitude = (c*basis).sum(-1)
        norm = c.square().sum(-1).clamp_min(torch.finfo(c.dtype).tiny)
        floor = float(self.decoder.base_decoder.config.velocity_floor_mixture)
        gaussian = torch.exp(-.5*w.square().sum(-1)/theta)/(2*torch.pi*theta)
        g = (1-floor)*amplitude.square()/norm+floor*gaussian
        jacobian = source.diagonal(dim1=-2, dim2=-1).prod(-1)/target.diagonal(dim1=-2, dim2=-1).prod(-1)
        result = {key: value for key, value in geometry.items() if not key.startswith('_')}
        result.update(g=g*jacobian, f=geometry['rho']*(g*jacobian))
        if jets is None:
            return result
        def dot(dx, dy):
            return (c*x[dx]*y[dy]).sum(-1)
        ax, ay, axx, axy, ayy = dot(1,0), dot(0,1), dot(2,0), dot(1,1), dot(0,2)
        gw = (1-floor)*2*amplitude[:, None]*torch.stack((ax, ay), 1)/norm[:, None]-floor*gaussian[:, None]*w/theta[:, None]
        xx = (1-floor)*2*(ax.square()+amplitude*axx)/norm+floor*gaussian*(w[:,0].square()/theta.square()-1/theta)
        xy = (1-floor)*2*(ax*ay+amplitude*axy)/norm+floor*gaussian*w[:,0]*w[:,1]/theta.square()
        yy = (1-floor)*2*(ay.square()+amplitude*ayy)/norm+floor*gaussian*(w[:,1].square()/theta.square()-1/theta)
        affine = torch.linalg.solve_triangular(target.transpose(-1,-2), source.transpose(-1,-2), upper=True).transpose(-1,-2)
        metric = affine@affine.transpose(-1,-2)
        rho = geometry['rho']
        laplacian = rho*jacobian*(xx*metric[:,0,0]+2*xy*metric[:,0,1]+yy*metric[:,1,1])
        dc = jets['source_hermite_coefficients'].double()
        da = (dc*basis[:, None, :]).sum(-1)
        dn = 2*(dc*c[:, None, :]).sum(-1)
        gc = (1-floor)*(2*amplitude[:,None]*da/norm[:,None]-amplitude.square()[:,None]*dn/norm.square()[:,None])
        ds, dt = jets['_source_factor'], jets['_target_factor']
        rhs = -jets['velocity']-(dt@centered[:,None,:,None])[...,0]
        dr = torch.linalg.solve_triangular(target[:,None], rhs[...,None], upper=False)[...,0]
        dw = jets['source_mean_velocity']+(ds@centered[:,None,:,None])[...,0]+(source[:,None]@dr[...,None])[...,0]
        dj = jacobian[:,None]*((ds.diagonal(dim1=-2,dim2=-1)/source.diagonal(dim1=-2,dim2=-1)[:,None]).sum(-1)
                              -(dt.diagonal(dim1=-2,dim2=-1)/target.diagonal(dim1=-2,dim2=-1)[:,None]).sum(-1))
        dg = gc+(gw[:,None,:]*dw).sum(-1)
        spacetime = jets['rho']*(g*jacobian)[:,None]+rho[:,None]*(dg*jacobian[:,None]+g[:,None]*dj)
        velocity = rho[:,None]*jacobian[:,None]*(affine.transpose(-1,-2)@gw[...,None])[...,0]
        gradient = torch.cat((spacetime, velocity), dim=1)
        collision = 2*result['f']+(query[:,3:5]*velocity).sum(1)+theta*laplacian
        forcing = gradient[:,0]+(query[:,3:5]*gradient[:,1:3]).sum(1)+(geometry['electric'].double()*velocity).sum(1)-geometry['equation_condition'][:,0].double()*collision
        result.update(f_gradient=gradient, f_velocity_laplacian=laplacian, collision=collision, source=forcing, points=query)
        return result

    @torch.no_grad()
    def fields(self, points):
        query = self._points(points, 5).detach()
        chunks = [self._velocity(part, *self._lookup(part[:,:3], False)) for part in query.split(self.phase_chunk_size)]
        return _finite_fields({key: torch.cat([row[key] for row in chunks]) for key in chunks[0]})

    def source(self, points, *, create_graph=False):
        query = self._points(points, 5)
        if create_graph:
            return manufacture_moving_frame_source(self.decoder, query, self._carrier, self._condition, create_graph=True)
        with torch.no_grad():
            query = query.detach()
            self._stats['requested_phase_points'] += len(query)
            rows = []
            for part in query.split(self.phase_chunk_size):
                rows.append(self._velocity(part, *self._lookup(part[:,:3], True)))
                self._stats['assembled_phase_points'] += len(part)
            return _finite_fields({key: torch.cat([row[key] for row in rows]) for key in rows[0]})

    def macros(self, points_txy):
        """No microscopic moments; retain ordinary coordinate first derivatives."""
        p = self._points(points_txy, 3)
        phase = torch.cat((p, p.new_zeros((len(p),2))), 1)
        base = self.decoder.base_decoder
        full = lift_value_controls(self._carrier['z'], self.decoder.value_config)
        query, raw, zero = self.decoder._native_rho_groups(phase, full)
        if base.config.rho_positivity_gauge == 'phase_invariant_group_l1_v2' and bool(zero.any()):
            rho, phi, electric, coefficients = self.decoder._rho_with_constant_zero_groups(query, raw, zero)
        else:
            rho, phi, electric, coefficients = base._rho_poisson(query, raw)
        target = self.decoder.macro_at(phase, self._carrier['m'], {'rho_coefficients':coefficients})
        rho, u, c = rho.double(), target['mean'], target['covariance']
        raw_second = rho[:,None,None]*(c+u[:,:,None]*u[:,None,:])
        return _finite_fields({'rho':rho, 'momentum':rho[:,None]*u,
            'kinetic_energy':.5*rho*(u.square().sum(-1)+c.diagonal(dim1=-2,dim2=-1).sum(-1)),
            'raw_second_moment':torch.stack((raw_second[:,0,0],raw_second[:,0,1],raw_second[:,1,1]),1),
            'electric':electric, 'velocity':u, 'central_covariance':c,
            'global_velocity':target['global_velocity'], 'phi':phi})

    @torch.no_grad()
    def transport_macros(self, points_txy):
        """V2 values only, sharing cached geometry but never differentiating T3."""
        from .moving_frame_fast_macros import transport_from_geometry
        points = self._points(points_txy, 3).detach()
        rows = [transport_from_geometry(self._lookup(part, False)[0], self.decoder)
                for part in points.split(self.phase_chunk_size)]
        return _finite_fields({name: torch.cat([row[name] for row in rows]) for name in rows[0]})

    def clear_cache(self):
        self._cache.clear()

    def cache_info(self):
        size = sum(v.numel()*v.element_size() for entry in self._cache.values() for part in entry if part is not None for v in part.values())
        return {**self._stats, 'geometry_entries':len(self._cache), 'geometry_limit':self.geometry_cache_size,
                'geometry_cache_bytes':size, 'cached_final_source_arrays':False,
                'coordinate_cache_key_dtype':'float64', 'geometry_method':'original_operations_three_direction_JVP'}
