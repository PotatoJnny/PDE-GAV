"""Conservative source-node omission with independent calibration and phase bounds.

Only new source acquisitions may be reduced. Retain complete g0, stationary
operator, and source-pilot arrays; expand omitted source nodes with zeros before
using the existing tensor Green response. This helper never queries the Oracle.
"""
from __future__ import annotations
import math
import numpy as np
from green_ops import PAIRS,ou_parameters
from moments import central_tangent

GROUPS=(((0,),(1,)),((1,2),(1,1)),((3,4,5),(1,2,1)),((6,7),(1,1)))


def _multiply(left,right):
    out={}
    for (p,q),x in left.items():
        for (a,b),y in right.items():out[p+a,q+b]=out.get((p+a,q+b),0.)+x*y
    return out


def absolute_cardinal_integrals(model):
    """Exact integral of |cardinal_i(z)| |z|**p, p=0..3, over its box.

    Roots split each cubic cell into intervals with a known sign. The absolute
    z power is split at zero as well. All remaining integrals are polynomials.
    A rounding margin is applied to every nonzero result.
    """
    z=np.asarray(model.z,float);result=np.zeros((4,len(z)))
    for cell,(left,right) in enumerate(zip(z[:-1],z[1:])):
        width=right-left
        for node in range(len(z)):
            coefficient=np.asarray(model.cardinal.c[:,cell,node][::-1],float)
            trimmed=np.trim_zeros(coefficient,'b')
            if not len(trimmed):continue
            roots=np.polynomial.polynomial.polyroots(trimmed)
            cuts=[0.,width]
            cuts.extend(float(r.real) for r in roots if abs(r.imag)<1e-9*(1+abs(r.real)) and 0<r.real<width)
            if left<0<right:cuts.append(-left)
            cuts=sorted(set(cuts))
            for p in range(4):
                power=np.array([math.comb(p,a)*left**(p-a) for a in range(p+1)])
                product=np.polynomial.polynomial.polymul(coefficient,power)
                integral=np.concatenate(([0.],product/np.arange(1,len(product)+1)))
                for lo,hi in zip(cuts[:-1],cuts[1:]):
                    middle=(lo+hi)/2
                    sign=np.sign(np.polynomial.polynomial.polyval(middle,coefficient))*np.sign(left+middle)**p
                    value=np.polynomial.polynomial.polyval(hi,integral)-np.polynomial.polynomial.polyval(lo,integral)
                    result[p,node]+=max(0.,sign*value)
    return result*(1+1e-10)+1e-14


def phase_envelope(model,duration,wavevector_max):
    """Bound |K_m(t,k,node)| for 0<=t<=duration and componentwise |k|<=max.

    Bound both GH and continuous cardinal kernels before taking their maximum.
    Any convex lag-dependent GH/spline blend is therefore covered. The OU
    Gaussian spatial damping has magnitude <=1 and may safely be omitted.
    """
    r,A,V,C,W=ou_parameters(np.array([duration]),model.nu,model.theta)
    r=float(r[0]);variance=float(V[0]);cross=float(C[0]);kmax=np.asarray(wavevector_max,float)
    absolute=absolute_cardinal_integrals(model)
    if getattr(model,'axis_blend',False):
        gh_axis=np.asarray(model.gh_axis_integration)
        absolute=np.maximum(absolute,np.stack([gh_axis*abs(model.z)**p for p in range(4)]))
    constant=abs(model.center-model.bulk)+(1-r)*abs(model.bulk)+cross*kmax
    polynomials=[]
    for axis in range(2):
        linear={(0,0):constant[axis],(1,0):abs(model.transform[axis,0]),(0,1):abs(model.transform[axis,1])}
        square=_multiply(linear,linear);second=dict(square);second[0,0]=second.get((0,0),0.)+variance
        third=_multiply(square,linear)
        for power,value in linear.items():third[power]=third.get(power,0.)+3*variance*value
        polynomials.append([{(0,0):1.},linear,second,third])
    v=model.physical_velocity
    mean=np.maximum(abs(v-model.bulk),abs(r*v-model.bulk))+cross*kmax
    gp=[[np.ones(len(v)),mean[:,a],mean[:,a]**2+variance,mean[:,a]**3+3*variance*mean[:,a]] for a in range(2)]
    rows=[]
    for p,q in PAIRS:
        polynomial=_multiply(polynomials[0][p],polynomials[1][q])
        spline=np.zeros((len(model.z),len(model.z)))
        for (a,b),coefficient in polynomial.items():spline+=coefficient*np.outer(absolute[a],absolute[b])
        spline=spline.ravel()*model.jacobian
        gh=model.gh_integration*gp[0][p]*gp[1][q]
        rows.append(np.maximum(spline,gh) if model.gh_blend else spline)
    result=np.asarray(rows)*(1+1e-10)
    if not np.isfinite(result).all() or np.any(result<0):raise FloatingPointError('invalid phase envelope')
    return result


def _node_costs(values,raw,relative_velocity,integration,center,theta,duration,envelope):
    nv=len(relative_velocity);ns=len(values)
    pairs=tuple((d-b,b) for d in range(5) for b in range(d+1))
    rule=np.stack([integration*relative_velocity[:,0]**p*relative_velocity[:,1]**q for p,q in pairs])
    contribution=rule[:,:,None]*values.T[None]
    full=contribution.sum(axis=1);costs=[]
    for degree in range(5):
        ids=[i for i,(p,q) in enumerate(pairs) if p+q==degree]
        floor=.02*theta**(degree/2)/max(duration,.25)
        scale=max(float(np.sqrt(np.mean(np.sum(full[ids]**2,axis=0)))),floor)
        costs.append(np.sqrt(np.mean(np.sum(abs(contribution[ids])**2,axis=0),axis=-1))/scale)
    central=central_tangent(raw[:,None,:],contribution[:10],center)
    total=central.sum(axis=1)
    jacobian=central_tangent(raw[:,None,:],np.broadcast_to(np.eye(10)[:,:,None],(10,10,ns)),center)
    bounded=np.einsum('gms,mv,sv->gvs',abs(jacobian),envelope,abs(values),optimize=True)
    floors=(1.0041253663782175,.4788345668393978,2.930763214320111,.9900774471118267)
    for (group,weight),floor in zip(GROUPS,floors):
        ids=list(group);w=np.asarray(weight)[:,None]
        scale=max(float(np.sqrt(np.mean(np.sum(w*abs(total[ids])**2,axis=0)))),.02*floor/max(duration,.25))
        costs.append(np.sqrt(np.mean(np.sum(w[:,None]*abs(central[ids])**2,axis=0),axis=-1))/scale)
        costs.append(np.sqrt(np.mean(np.sum(w[:,None]*bounded[ids]**2,axis=0),axis=-1))/scale)
    return np.asarray(costs)


def conservative_source_mask(pilot_values,calibration_values,relative_velocity,integration,
                             pilot_raw,calibration_raw,center,theta,duration,kernel_model,
                             wavevector_max,*,training_tolerance=1e-4,
                             calibration_tolerance=1e-3,maximum_drop_fraction=.25):
    """Return (retained Boolean mask, diagnostics); every failure keeps all nodes.

    The mask uses only pilot samples. Calibration samples are an independent
    acceptance gate, never used to train the ranking. Both gates control an RMS
    triangle bound for degree0--4 raw source moments, degree0--3 central moments,
    and every phased OU adjoint in the declared lag/wavevector box. Field-error
    amplification by Poisson is outside this source-quadrature diagnostic.
    """
    velocity=np.asarray(relative_velocity,float);all_nodes=np.ones(len(velocity),bool)
    diagnostics={'status':'unchanged','node_count':len(velocity),'retained_nodes':len(velocity)}
    try:
        p=np.asarray(pilot_values,float);c=np.asarray(calibration_values,float);wt=np.asarray(integration,float)
        pr=np.asarray(pilot_raw,float)[:10];cr=np.asarray(calibration_raw,float)[:10];center=np.asarray(center,float);kmax=np.asarray(wavevector_max,float)
        if (velocity.ndim!=2 or velocity.shape[1]!=2 or p.ndim!=2 or c.ndim!=2 or p.shape[1]!=len(velocity)
            or c.shape[1]!=len(velocity) or wt.shape!=(len(velocity),) or pr.shape!=(10,len(p)) or cr.shape!=(10,len(c))
            or center.shape!=(2,) or kmax.shape!=(2,) or np.any(kmax<0) or np.any(wt<=0) or np.any(pr[0]<=0) or np.any(cr[0]<=0)
            or duration<=0 or theta<=0 or not 0<maximum_drop_fraction<1 or min(training_tolerance,calibration_tolerance)<=0
            or not all(np.isfinite(x).all() for x in (p,c,velocity,wt,pr,cr,center,kmax,theta,duration,training_tolerance,calibration_tolerance,maximum_drop_fraction))):
            raise ValueError('finite source, positive raw density, compatible axes and controls required')
        envelope=phase_envelope(kernel_model,float(duration),kmax)
        if envelope.shape!=(10,len(velocity)):raise ValueError('kernel tensor does not match source velocity rule')
        train=_node_costs(p,pr,velocity,wt,center,float(theta),float(duration),envelope)
        valid=_node_costs(c,cr,velocity,wt,center,float(theta),float(duration),envelope)
        if not np.isfinite(train).all() or not np.isfinite(valid).all():raise FloatingPointError('nonfinite pruning costs')
        ordered=np.argsort(train.max(axis=0),kind='stable');cumulative=np.cumsum(train[:,ordered],axis=1)
        count=min(int(np.sum(cumulative.max(axis=0)<=training_tolerance)),int(maximum_drop_fraction*len(velocity)))
        if count==0:diagnostics['reason']='no nodes satisfy the phase and moment bound';return all_nodes,diagnostics
        dropped=ordered[:count];train_bound=train[:,dropped].sum(axis=1);calibration_bound=valid[:,dropped].sum(axis=1)
        diagnostics.update(training_bound_max=float(train_bound.max()),calibration_bound_max=float(calibration_bound.max()),
            training_bounds=train_bound.tolist(),calibration_bounds=calibration_bound.tolist(),training_tolerance=float(training_tolerance),
            calibration_tolerance=float(calibration_tolerance),proposed_dropped_nodes=count,
            phase_duration=float(duration),phase_wavevector_max=kmax.tolist())
        if np.max(calibration_bound)>calibration_tolerance:
            diagnostics.update(status='calibration_rejected',reason='independent phase or moment triangle bound exceeds tolerance')
            return all_nodes,diagnostics
        mask=all_nodes.copy();mask[dropped]=False
        diagnostics.update(status='applied',retained_nodes=int(mask.sum()))
        return mask,diagnostics
    except (ValueError,TypeError,AttributeError,FloatingPointError,ArithmeticError,np.linalg.LinAlgError,MemoryError) as exc:
        diagnostics.update(status='invalid_fallback',reason=str(exc));return all_nodes,diagnostics
