"""Charge-response empirical tensor interpolation; no Oracle access.

Covariance-conditioned interpolation keeps contributions from every sampled
velocity-axis mode.  Greedy nodes minimize posterior uncertainty of smooth
Fourier density functionals, rather than only interpolating POD vectors.
Heldout validation is required: the separable covariance is an approximation.
"""
from __future__ import annotations

import numpy as np
from tensor_cubature import TensorCubature


def ou_response(nodes, max_wave=8., horizon=1., nu=0., theta=1.,
                n_wave=12, n_lag=13):
    """One-axis density response functionals of the homogeneous OU process.

    Includes both signs and zero wavenumber.  Spatial wavenumbers have modest
    1/sqrt(1+k^2) emphasis, reflecting Poisson attenuation of electric error.
    This is a target metric, not a closure for the nonlinear kinetic solve.
    """
    v=np.asarray(nodes,float)
    k=np.linspace(0.,max_wave,n_wave)
    lag=np.linspace(0.,horizon,n_lag)
    kk,tt=np.meshgrid(k,lag,indexing='ij')
    if nu:
        a=-np.expm1(-nu*tt)/nu
        z=nu*tt
        poly=z**3*(2/3+z*(-.5+z*(7/30+z*(-1/12+z*31/1260))))
        direct=2*z+4*np.expm1(-z)-np.expm1(-2*z)
        noise=theta/nu**2*np.where(z<1e-3,poly,direct)
    else:
        a=tt;noise=np.zeros_like(tt)
    weight=np.exp(-.5*noise*kk*kk)/(1+kk*kk)**.25
    phase=(kk*a).ravel()[:,None]*v[None,:]
    weights=weight.ravel()[:,None]
    # Real/imaginary rows keep all covariance algebra real.
    return np.concatenate([weights*np.cos(phase),weights*np.sin(phase)])


def _covariance(samples,axis):
    if axis==0:return np.einsum('rxy,rzy->xz',samples,samples,optimize=True)
    return np.einsum('rxy,rxz->yz',samples,samples,optimize=True)


def _condition(c,goal,count,ridge):
    c=(c+c.T)*.5
    scale=max(float(np.max(np.diag(c))),np.finfo(float).tiny)
    c=c/scale
    regularization=float(ridge)
    residual=c.copy();selected=[];variance=[]
    for _ in range(count):
        numerator=np.sum((goal@residual)**2,axis=0)
        denominator=np.maximum(np.diag(residual)+regularization,1e-30)
        merit=numerator/denominator
        merit[selected]=-np.inf
        i=int(np.argmax(merit));selected.append(i)
        column=residual[:,i].copy()
        residual-=np.outer(column,column)/denominator[i]
        residual=(residual+residual.T)*.5
        variance.append(float(np.sum((goal@residual)*goal)))
    selected=np.asarray(selected,dtype=int)
    system=c[np.ix_(selected,selected)]+regularization*np.eye(count)
    interpolation=np.linalg.solve(system,c[selected,:]).T
    return selected,interpolation,{
        'condition':float(np.linalg.cond(system)),
        'posterior_goal_trace':variance,
        'initial_goal_trace':float(np.sum((goal@c)*goal)),
    }


def fit_goal(samples,nodes,integration_weights,*,count=20,ridge=1e-8,
             max_wave=8.,horizon=1.,nu=0.,theta=1.,degree=4,
             validation=None):
    f=np.asarray(samples,float);v=np.asarray(nodes,float)
    q=np.asarray(integration_weights,float)
    if f.ndim!=3 or f.shape[1:]!=(len(v),len(v)):
        raise ValueError('samples and velocity axis disagree')
    if not np.isfinite(f).all() or count<1 or count>len(v) or ridge<=0:
        raise ValueError('finite samples, valid count and positive ridge required')
    goal=ou_response(v,max_wave,horizon,nu,theta)*q[None,:]
    goal/=np.linalg.norm(goal)
    parts=[_condition(_covariance(f,a),goal,count,ridge) for a in range(2)]
    ix,tx,dx=parts[0];iy,ty,dy=parts[1]
    pairs=tuple((d-j,j) for d in range(degree+1) for j in range(d+1))
    wx=np.asarray([(q*v**j)@tx for j in range(degree+1)])
    wy=np.asarray([(q*v**j)@ty for j in range(degree+1)])
    weights=np.stack([wx[i,:,None]*wy[j,None,:] for i,j in pairs])
    diagnostics={'method':'goal_covariance_tensor','ridge':ridge,'node_count':count**2,
        'nodes_per_axis':[count,count],'axis_conditioning':[dx,dy],
        'goal':{'max_wave':max_wave,'horizon':horizon,'nu':nu,'theta':theta}}
    rule=TensorCubature(v.copy(),v.copy(),q.copy(),q.copy(),ix,iy,tx,ty,pairs,weights,diagnostics)
    if validation is not None:rule.diagnostics['heldout']=rule.validate(validation)
    return rule


def validate_charge(rule,values,*,max_wave=8.,horizon=1.,nu=0.,theta=1.,
                    n_wave=6,n_lag=13):
    """Transported charge error across independent 2D wave/lag directions.

    Includes zero-wave charge and the public collisional response. No completed
    solution data enter this diagnostic; instantaneous current is checked by
    the rule's separate ordinary moment validation.
    """
    f=np.asarray(values,float)
    reconstruction=rule.reconstruct(rule.sample(f))
    dv=rule.integration_x[:,None]*rule.integration_y[None,:]
    vx=rule.nodes_x[:,None];vy=rule.nodes_y[None,:]
    ks=np.linspace(0.,max_wave,n_wave)
    angles=np.arange(8)*np.pi/8
    lag=np.linspace(0,horizon,n_lag)
    if nu:
        a=-np.expm1(-nu*lag)/nu
        z=nu*lag
        poly=z**3*(2/3+z*(-.5+z*(7/30+z*(-1/12+z*31/1260))))
        direct=2*z+4*np.expm1(-z)-np.expm1(-2*z)
        noise=theta/nu**2*np.where(z<1e-3,poly,direct)
    else:a=lag;noise=np.zeros_like(lag)
    errors=[];targets=[];wave_errors={}
    for k in ks:
        functions=np.asarray([dv*np.exp(-1j*k*aa*(np.cos(angle)*vx+np.sin(angle)*vy))
                     for aa in a for angle in angles])
        target=np.einsum('rxy,mxy->rm',f,functions,optimize=True)
        delta=np.einsum('rxy,mxy->rm',reconstruction-f,functions,optimize=True)
        damp=np.repeat(np.exp(-.5*noise*k*k),len(angles))/(1+k*k)**.25
        errors.append(np.sum(abs(delta*damp)**2));targets.append(np.sum(abs(target*damp)**2))
        wave_errors[str(float(k))]=float(np.linalg.norm(delta)/max(np.linalg.norm(target),1e-30))
    return {'charge_response_relative_rms':float(np.sqrt(sum(errors)/max(sum(targets),1e-30))),
            'wave_relative_rms':wave_errors,
            'density_relative_rms':wave_errors['0.0']}
