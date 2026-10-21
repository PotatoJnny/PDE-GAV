"""Case-local structured VPFP Oracle with bounded coordinate-factor reuse.

Only requested coordinates are evaluated. Exact host FP64 keys precede device
conversion; immutable case state, time projections and one velocity basis are
private. No final source arrays, Cartesian completion, result scans, asset
reads or global cache are introduced. The caller serializes access and owns the
existing IEEE-FP32 context/physics qualification at its factory boundary.
"""
from __future__ import annotations

from collections import OrderedDict
import math
import numpy as np

from . import basic_oracle as basic


class StructuredOracle:
    """Trusted immutable Case representation; numerical inputs checked upstream."""

    TIME_LIMIT = 8
    VELOCITY_LIMIT = 8192
    POINT_BLOCK = 512
    TIME_FALLBACK_THRESHOLD = 32

    def __init__(self, model, latent, equation_condition):
        self.model, self.torch = model, model.torch
        parameter = next(model.parameters())
        self.dtype, self.device = parameter.dtype, parameter.device
        self.row = self.torch.as_tensor(latent, dtype=self.dtype, device=self.device).reshape(-1).detach().clone()
        self.condition = self.torch.as_tensor(equation_condition, dtype=self.dtype, device=self.device).reshape(-1).detach().clone()
        if self.row.numel() != model.config.latent_dim or self.condition.numel() != 2:
            raise ValueError("one full Case latent and one nu/T condition required")
        scale = float(model.config.latent_input_scale)
        if scale != 1.:
            self.row = self.row * scale
        cfg, layout = model.config, model.structured_layout
        end = layout["rho_latent_dim"]
        self.rho_controls = self.row[:end].reshape(layout["rho_time_feature_count"], -1)
        self.local_controls = self.row[end + cfg.structured_kinetic_global_dim:].reshape(
            cfg.structured_local_channels, layout["local_time_feature_count"],
            layout["local_spatial_feature_count"])
        self.rho_kappa = model._rho_kappa.to(dtype=self.dtype, device=self.device).detach().clone()
        lx, ly = map(float, cfg.equation.x_domain)
        self.local_kappa = self._tensor([(2.*math.pi*kx/lx, 2.*math.pi*ky/ly)
                                        for kx, ky in model.local_half_modes]).reshape(-1, 2)
        self.modes = self._tensor(model.hermite_modes, indices=True)
        self.maximum_degree = max(max(pair) for pair in model.hermite_modes)
        degree = range(self.maximum_degree + 1)
        self.degree = self._tensor(list(degree), indices=True)
        self.first_left = self._tensor([math.sqrt(n/2.) for n in degree])
        self.first_right = self._tensor([math.sqrt((n+1)/2.) for n in degree])
        self.second_left = self._tensor([math.sqrt(n*(n-1)) if n >= 2 else 0. for n in degree])
        self.second_right = self._tensor([math.sqrt((n+1)*(n+2)) for n in degree])
        self._time = OrderedDict()
        self._velocity = None
        self._stats = dict(time_hits=0, time_misses=0, time_evictions=0,
            velocity_hits=0, velocity_misses=0, cartesian_blocks=0, sparse_blocks=0,
            mixed_cartesian_blocks=0, cartesian_pairs=0, sparse_pairs=0,
            macro_fallback_blocks=0, requested_points=0, assembled_pairs=0,
            time_projection_batches=0, maximum_projection_batch=0)

    def _tensor(self, value, *, indices=False):
        return self.torch.as_tensor(value, device=self.device,
                                    dtype=self.torch.long if indices else self.dtype)

    @staticmethod
    def _query(points):
        query = np.asarray(points, dtype=np.float64)
        if query.ndim != 2 or query.shape[1] != 5 or not len(query):
            raise ValueError("StructuredOracle needs nonempty [N,5] points")
        return query

    @staticmethod
    def _time_key(value):
        return np.float64(value).tobytes()

    def _project_batch(self, times):
        """Contract only active cubic controls before any space expansion."""
        torch, cfg = self.torch, self.model.config
        t = self._tensor(times)
        if cfg.structured_time_basis == "piecewise_cubic_hermite_v1":
            count = cfg.structured_time_control_count
            t0, t1 = map(float, cfg.time_domain)
            tau = ((t-t0)/(t1-t0)).clamp(0., 1.)
            position = tau * float(count-1)
            lower = torch.floor(position).to(torch.long).clamp(max=count-2)
            local = position-lower.to(t.dtype)
            l2, l3 = local.square(), local.square()*local
            h00, h10 = 2.*l3-3.*l2+1., l3-2.*l2+local
            h01, h11 = -2.*l3+3.*l2, l3-l2
            d00, d10 = 6.*l2-6.*local, 3.*l2-4.*local+1.
            d01, d11 = -6.*l2+6.*local, 3.*l2-2.*local
            interval = 1./float(count-1)
            rate = float(count-1)/(t1-t0)
            indices = torch.stack((lower, lower+1, count+lower, count+lower+1), dim=1)
            weights = torch.stack((h00,h01,interval*h10,interval*h11), dim=1)
            derivative = torch.stack((rate*d00,rate*d01,rate*interval*d10,rate*interval*d11), dim=1)
            rho = self.rho_controls[indices]
            local_values = self.local_controls[:, indices, :].permute(1, 2, 0, 3)
            # [B,4,C,S], never [all-query-points,C,66].
            rv = (rho*weights[:, :, None]).sum(1)
            rd = (rho*derivative[:, :, None]).sum(1)
            lv = (local_values*weights[:, :, None, None]).sum(1)
            ld = (local_values*derivative[:, :, None, None]).sum(1)
        else:
            rb, rdb, rn = basic._time_basis_jet(torch,cfg,t,order=cfg.structured_rho_time_order)
            if cfg.structured_rho_time_order == cfg.structured_local_time_order:
                lb, ldb, ln = rb, rdb, rn
            else:
                lb, ldb, ln = basic._time_basis_jet(torch,cfg,t,order=cfg.structured_local_time_order)
            rv, rd = rb@self.rho_controls/rn, rdb@self.rho_controls/rn
            lv = torch.einsum("bt,cts->bcs",lb,self.local_controls)/ln
            ld = torch.einsum("bt,cts->bcs",ldb,self.local_controls)/ln
        self._stats["time_projection_batches"] += 1
        self._stats["maximum_projection_batch"] = max(self._stats["maximum_projection_batch"],len(times))
        # Clone rows: cache eviction must not retain a larger batch allocation.
        return [tuple(value[index].clone() for value in (rv,rd,lv,ld))
                for index in range(len(times))]

    def _time_entries(self, times):
        if len(times) > self.TIME_LIMIT:
            raise ValueError("time projection batches are bounded by eight")
        entries, missing, missing_indices = [None]*len(times), [], []
        for index,value in enumerate(times):
            key = self._time_key(value)
            if key in self._time:
                entries[index] = self._time.pop(key)
                self._time[key] = entries[index]
                self._stats["time_hits"] += 1
            else:
                missing.append(value)
                missing_indices.append(index)
                self._stats["time_misses"] += 1
        if missing:
            for index,entry in zip(missing_indices,self._project_batch(missing),strict=True):
                entries[index] = entry
                key = self._time_key(times[index])
                self._time[key] = entry
                if len(self._time) > self.TIME_LIMIT:
                    self._time.popitem(last=False)
                    self._stats["time_evictions"] += 1
        return entries

    def _spatial(self, points, *, derivatives=True):
        torch, k = self.torch, self.local_kappa
        phase = points[:, :1]*k[:,0] + points[:,1:2]*k[:,1]
        cosine,sine = math.sqrt(2.)*torch.cos(phase),math.sqrt(2.)*torch.sin(phase)
        interleave=lambda a,b:torch.stack((a,b),dim=-1).flatten(1)
        ones=torch.ones_like(points[:,:1])
        value=torch.cat((ones,interleave(cosine,sine)),1)
        if not derivatives:
            return value
        zeros=torch.zeros_like(ones)
        return value,torch.cat((zeros,interleave(-k[:,0]*sine,k[:,0]*cosine)),1),torch.cat((zeros,interleave(-k[:,1]*sine,k[:,1]*cosine)),1)

    def _velocity_basis(self, points, *, derivatives=True):
        exact=np.ascontiguousarray(points,dtype=np.float64)
        key=(exact.shape,exact.tobytes())
        if self._velocity is not None and self._velocity[0] == key and (not derivatives or len(self._velocity[2]) == 5):
            self._stats["velocity_hits"] += 1
            return self._velocity[1],self._velocity[2] if derivatives else self._velocity[2][:1]
        self._stats["velocity_misses"] += 1
        velocity=self._tensor(exact)
        torch=self.torch
        scale=torch.sqrt(2.*self.condition[1])
        maximum=self.maximum_degree
        def one(values):
            xi=values/scale
            f=[torch.pi**(-.25)/torch.sqrt(scale)*torch.exp(-.5*xi.square())]
            final=maximum+2 if derivatives else maximum
            if final>=1:
                f.append(math.sqrt(2.)*xi*f[0])
            for n in range(1,final):
                f.append(math.sqrt(2./(n+1))*xi*f[n]-math.sqrt(n/(n+1))*f[n-1])
            table=torch.stack(f,1)
            value=table[:,:maximum+1]
            if not derivatives:
                return (value,)
            padded=torch.cat((torch.zeros_like(table[:,:1]).expand(-1,2),table),1)
            degree=self.degree
            d=(padded[:,degree+1]*self.first_left-padded[:,degree+3]*self.first_right)/scale
            dd=(padded[:,degree]*self.second_left-(2.*degree+1.)*value+padded[:,degree+4]*self.second_right)/(2.*scale.square())
            return value,d,dd
        x,y=one(velocity[:,0]),one(velocity[:,1])
        i,j=self.modes[:,0],self.modes[:,1]
        basis=(x[0][:,i]*y[0][:,j],)
        if derivatives:
            basis+=(x[1][:,i]*y[0][:,j],x[0][:,i]*y[1][:,j],x[2][:,i]*y[0][:,j],x[0][:,i]*y[2][:,j])
        if len(exact)<=self.VELOCITY_LIMIT:
            self._velocity=(key,velocity,basis)
        return velocity,basis

    def _macro(self, spacetime):
        torch=self.torch
        times,time_inverse=np.unique(spacetime[:,0],return_inverse=True)
        dimensions={"rho":0,"rho_derivatives":1,"electric":0,"coefficients":0,"coefficient_derivatives":1}
        # Many unrelated sparse times should not become thousands of tiny
        # Python groups. Use the verified dense kernel with bounded Q instead.
        if len(times)>self.TIME_FALLBACK_THRESHOLD:
            self._stats["macro_fallback_blocks"] += 1
            coords=self._tensor(spacetime)
            parts=[basic._macro_jet(self.model,coords[i:i+self.POINT_BLOCK],self.row)
                   for i in range(0,len(coords),self.POINT_BLOCK)]
            return {name:torch.cat([part[name] for part in parts],dim=axis) for name,axis in dimensions.items()}
        parts,positions=[],[]
        cfg=self.model.config
        scale=float(cfg.structured_local_scale)/math.sqrt(float(self.local_controls.shape[-1]))
        for first in range(0,len(times),self.TIME_LIMIT):
            selected_times=times[first:first+self.TIME_LIMIT]
            entries=self._time_entries(selected_times)
            stacked=[torch.stack([entry[index] for entry in entries]) for index in range(4)]
            selected=np.flatnonzero((time_inverse>=first)&(time_inverse<first+len(selected_times)))
            for start in range(0,len(selected),self.POINT_BLOCK):
                indices=selected[start:start+self.POINT_BLOCK]
                host_local=time_inverse[indices]-first
                coords=self._tensor(spacetime[indices])
                spatial,sx,sy=self._spatial(coords[:,1:])
                # All x/y/value projections in one bmm; time uses its already
                # projected coefficient derivative. No 66-time broadcast.
                if np.all(host_local==host_local[0]):
                    rho,rhod,local,locald=entries[int(host_local[0])]
                    rho=rho.expand(len(indices),-1)
                    rhod=rhod.expand(len(indices),-1)
                    # Shared-time blocks use four dense matrix products,
                    # never replicate [C,S] coefficients per spatial point.
                    projected=torch.bmm(torch.stack((local,locald,local,local)),
                        torch.stack((spatial.T,spatial.T,sx.T,sy.T)))*scale
                    raw=projected[0].T
                    derivatives=projected[1:].transpose(1,2)
                else:
                    local_indices=self._tensor(host_local,indices=True)
                    rho,rhod,local,locald=[value.index_select(0,local_indices) for value in stacked]
                    projection=torch.bmm(local,torch.stack((spatial,sx,sy),dim=-1))*scale
                    time_projection=torch.bmm(locald,spatial[:,:,None])[:,:,0]*scale
                    raw=projection[:,:,0]
                    derivatives=torch.stack((time_projection,projection[:,:,1],projection[:,:,2]),0)
                norm=torch.linalg.vector_norm(raw,dim=-1,keepdim=True)
                safe=norm.clamp_min(torch.finfo(raw.dtype).tiny)
                normalized=raw/safe
                ground=torch.zeros_like(raw); ground[:,0]=1.
                valid=norm>torch.finfo(raw.dtype).eps
                normalized=torch.where(valid,normalized,ground)
                parallel=(normalized[None]*derivatives).sum(-1,keepdim=True)
                cd=torch.where(valid[None],(derivatives-normalized[None]*parallel)/safe[None],torch.zeros_like(derivatives))
                macro=basic._rho_coordinate_jet(self.model,coords,rho,rhod)
                macro.update(coefficients=normalized,coefficient_derivatives=cd)
                parts.append(macro);positions.extend(indices.tolist())
        restore=self._tensor(np.argsort(positions),indices=True)
        return {name:torch.cat([part[name] for part in parts],dim=axis).index_select(axis,restore)
                for name,axis in dimensions.items()}

    @staticmethod
    def _full_pairs(space_inverse,velocity_inverse,space_count,velocity_count):
        pairs=space_inverse*velocity_count+velocity_inverse
        # GEMM is permitted only when every pair already occurs in the input.
        full=space_count*velocity_count<=len(pairs) and len(np.unique(pairs))==space_count*velocity_count
        return full,pairs

    @staticmethod
    def _macro_subset(macro,indices):
        dims={"rho":0,"rho_derivatives":1,"electric":0,"coefficients":0,"coefficient_derivatives":1}
        return {name:macro[name].index_select(axis,indices) for name,axis in dims.items()}

    def _cartesian_source(self,macro,velocity,basis):
        """Only call for rows whose every velocity pair was actually requested."""
        torch=self.torch
        c,cd=macro["coefficients"],macro["coefficient_derivatives"]
        amplitude=c@basis[0].T
        amplitude_d=torch.matmul(cd,basis[0].T)
        jets=torch.matmul(c,torch.stack(basis[1:]).transpose(1,2))
        return basic._source_from_amplitudes(torch,model=self.model,
            velocity_x=velocity[:,0][None,:],velocity_y=velocity[:,1][None,:],condition=self.condition,
            macro=macro,amplitude=amplitude,amplitude_d=amplitude_d,
            amplitude_vx=jets[0],amplitude_vy=jets[1],
            amplitude_vx2=jets[2],amplitude_vy2=jets[3],cartesian=True)

    def _sparse_source(self,macro,velocity,basis,space_inverse,velocity_inverse):
        si,vi=self._tensor(space_inverse,indices=True),self._tensor(velocity_inverse,indices=True)
        return basic._source_pointwise(self.torch,model=self.model,
            velocity=velocity.index_select(0,vi),condition=self.condition.expand(len(space_inverse),-1),
            macro=self._macro_subset(macro,si),
            basis=tuple(value.index_select(0,vi) for value in basis))

    def source(self, points, chunk_size=8192):
        from .perturbation_backend import _legacy_api
        query=self._query(points)
        if type(chunk_size) is not int or not 1<=chunk_size<=8192:
            raise ValueError("structured source chunk_size must be in 1..8192")
        torch=self.torch;outputs=[]
        with _legacy_api().ieee_float32_coordinate_matmul(self.model):
            for start in range(0,len(query),chunk_size):
                requested=query[start:start+chunk_size]
                spacetime,si=np.unique(requested[:,:3],axis=0,return_inverse=True)
                velocity,vi=np.unique(requested[:,3:],axis=0,return_inverse=True)
                macro=self._macro(spacetime)
                v,basis=self._velocity_basis(velocity)
                full,pairs=self._full_pairs(si,vi,len(spacetime),len(velocity))
                self._stats["requested_points"]+=len(requested)
                if full:
                    self._stats["cartesian_blocks"]+=1
                    self._stats["assembled_pairs"]+=len(spacetime)*len(velocity)
                    self._stats["cartesian_pairs"]+=len(spacetime)*len(velocity)
                    source=self._cartesian_source(macro,v,basis)
                    outputs.append(source.flatten().index_select(0,self._tensor(pairs,indices=True)))
                else:
                    # Fixed-size Oracle chunks need not align with a Solver's
                    # velocity count (8192 versus S0's 81). Use GEMM for every
                    # complete requested row, leaving only incomplete rows to
                    # pointwise assembly. Duplicates cannot fake completeness.
                    independent=np.unique(pairs)
                    counts=np.bincount(independent//len(velocity),minlength=len(spacetime))
                    complete=np.flatnonzero(counts==len(velocity))
                    if len(complete):
                        self._stats["mixed_cartesian_blocks"]+=1
                        row_map=np.full(len(spacetime),-1,dtype=np.int64)
                        row_map[complete]=np.arange(len(complete))
                        full_positions=np.flatnonzero(row_map[si]>=0)
                        sparse_positions=np.flatnonzero(row_map[si]<0)
                        compact=self._macro_subset(macro,self._tensor(complete,indices=True))
                        full_source=self._cartesian_source(compact,v,basis)
                        full_pairs=row_map[si[full_positions]]*len(velocity)+vi[full_positions]
                        result=self.row.new_empty(len(requested))
                        result.index_copy_(0,self._tensor(full_positions,indices=True),
                            full_source.flatten().index_select(0,self._tensor(full_pairs,indices=True)))
                        remaining=self._sparse_source(macro,v,basis,si[sparse_positions],vi[sparse_positions])
                        result.index_copy_(0,self._tensor(sparse_positions,indices=True),remaining)
                        self._stats["cartesian_pairs"]+=len(complete)*len(velocity)
                        self._stats["sparse_pairs"]+=len(sparse_positions)
                        self._stats["assembled_pairs"]+=len(complete)*len(velocity)+len(sparse_positions)
                        outputs.append(result)
                    else:
                        self._stats["sparse_blocks"]+=1
                        self._stats["sparse_pairs"]+=len(requested)
                        self._stats["assembled_pairs"]+=len(requested)
                        outputs.append(self._sparse_source(macro,v,basis,si,vi))
        return torch.cat(outputs)

    def initial(self, points, field_name="rho"):
        from .perturbation_backend import _legacy_api
        query=self._query(points)
        if field_name not in ("rho","g"):
            raise ValueError("initial field must be rho or g")
        torch,cfg=self.torch,self.model.config;outputs=[]
        with _legacy_api().ieee_float32_coordinate_matmul(self.model):
            rho,_,local,_=self._time_entries([float(cfg.time_domain[0])])[0]
            if field_name=="rho":
                raw=rho.reshape(len(self.model.rho_half_modes),2)
                c=float(cfg.rho_contrast)*float(cfg.equation.background_density)*raw/(1.+torch.linalg.vector_norm(raw,dim=-1).sum())
            for start in range(0,len(query),8192):
                requested=query[start:start+8192]
                xy,si=np.unique(requested[:,1:3],axis=0,return_inverse=True)
                coords=self._tensor(xy);sidx=self._tensor(si,indices=True)
                if field_name=="rho":
                    phase=coords[:,:1]*self.rho_kappa[:,0]+coords[:,1:2]*self.rho_kappa[:,1]
                    values=float(cfg.equation.background_density)+(c[:,0]*torch.cos(phase)+c[:,1]*torch.sin(phase)).sum(-1)
                    outputs.append(values.index_select(0,sidx));continue
                spatial=self._spatial(coords,derivatives=False)
                raw=float(cfg.structured_local_scale)*(spatial@local.T)/math.sqrt(float(spatial.shape[1]))
                c=basic._normalized(torch,raw)
                velocity,vi=np.unique(requested[:,3:],axis=0,return_inverse=True)
                v,basis=self._velocity_basis(velocity,derivatives=False)
                full,pairs=self._full_pairs(si,vi,len(xy),len(velocity))
                norm=c.square().sum(-1).clamp_min(torch.finfo(c.dtype).tiny)
                maxwellian=torch.exp(-.5*v.square().sum(-1)/self.condition[1])/(2.*torch.pi*self.condition[1])
                floor=float(cfg.velocity_floor_mixture)
                if full:
                    amplitude=c@basis[0].T
                    values=(1.-floor)*amplitude.square()/norm[:,None]+floor*maxwellian[None,:]
                    outputs.append(values.flatten().index_select(0,self._tensor(pairs,indices=True)))
                else:
                    vidx=self._tensor(vi,indices=True)
                    amplitude=(c.index_select(0,sidx)*basis[0].index_select(0,vidx)).sum(-1)
                    outputs.append((1.-floor)*amplitude.square()/norm.index_select(0,sidx)+floor*maxwellian.index_select(0,vidx))
        return torch.cat(outputs)

    def cache_info(self):
        def size(value):
            return value.numel()*value.element_size()
        time_bytes=sum(size(value) for entry in self._time.values() for value in entry)
        velocity_bytes=0 if self._velocity is None else (len(self._velocity[0][1])+size(self._velocity[1])+sum(size(value) for value in self._velocity[2]))
        owned=sum(size(value) for value in (self.row,self.condition,self.rho_kappa,self.local_kappa,
            self.modes,self.degree,self.first_left,self.first_right,self.second_left,self.second_right))
        return dict(self._stats,time_entries=len(self._time),time_limit=self.TIME_LIMIT,
            velocity_entries=int(self._velocity is not None),velocity_limit=self.VELOCITY_LIMIT,
            velocity_points=0 if self._velocity is None else len(self._velocity[1]),
            time_cache_bytes=time_bytes,velocity_cache_bytes=velocity_bytes,
            cache_bytes=time_bytes+velocity_bytes,owned_bytes=owned,
            exact_coordinate_key_dtype="float64",cached_final_source_arrays=False)
