"""Choose cubic/quintic forcing interpolation using source-pilot residuals.

The response proxy is an isolated Ornstein--Uhlenbeck response, not a VPFP
solution-error estimate. All inputs are candidate-owned public source values.
"""
from __future__ import annotations
import math
import numpy as np


def _channel_map(theta):
    matrix=np.zeros((10,8))
    scale=math.sqrt(theta)
    matrix[0,0]=4/1.0041253663782175
    matrix[1,1]=matrix[2,2]=scale/.4788345668393978
    matrix[0,3]=matrix[0,5]=theta/2.930763214320111
    matrix[3,3]=matrix[4,4]=matrix[5,5]=theta*math.sqrt(2)/2.930763214320111
    heat=.5*scale**3/.9900774471118267
    matrix[6,6],matrix[8,6],matrix[1,6]=heat*math.sqrt(6),heat*math.sqrt(2),4*heat
    matrix[9,7],matrix[7,7],matrix[2,7]=heat*math.sqrt(6),heat*math.sqrt(2),4*heat
    return matrix


def residual_response_diagnostic(times, residual, theta, nu):
    """Integrate linearly interpolated pilot residuals through OU damping."""
    times=np.asarray(times,float);residual=np.asarray(residual,float)
    if (times.ndim!=1 or len(times)<2 or np.any(np.diff(times)<=0)
            or residual.ndim!=3 or residual.shape[0]!=len(times)
            or residual.shape[-1]!=10 or not np.isfinite(residual).all()
            or not math.isfinite(theta) or theta<=0
            or not math.isfinite(nu) or nu<0):
        raise ValueError('finite ordered pilot times and (time,probe,10) residuals required')
    mapping=_channel_map(theta)
    rates=nu*np.array([0,1,1,2,2,2,3,3,3,3])
    widths=np.diff(times)
    x=widths[:,None]*rates
    damping=np.exp(-x)
    small=x<1.e-3
    first=np.empty_like(x);second=np.empty_like(x)
    xs=x[small]
    first[small]=1-xs/2+xs*xs/6-xs**3/24+xs**4/120
    second[small]=.5-xs/6+xs*xs/24-xs**3/120+xs**4/720
    xl=x[~small]
    first[~small]=-np.expm1(-xl)/xl
    second[~small]=(xl+np.expm1(-xl))/(xl*xl)
    response=np.zeros_like(residual[0]);energy=0.;density_energy=0.
    for i,h in enumerate(widths):
        nxt=(damping[i]*response+h*(first[i]*residual[i]
                                  +second[i]*(residual[i+1]-residual[i])))
        a=response@mapping;b=nxt@mapping
        energy+=h/2*np.mean(a*a+b*b)
        density_energy+=h/2*np.mean(response[:,0]**2+nxt[:,0]**2)
        response=nxt
    channels=residual@mapping
    return {'response_rms':float(math.sqrt(energy/(times[-1]-times[0]))),
            'density_primitive_rms':float(math.sqrt(density_energy/(times[-1]-times[0]))),
            'density_final_rms':float(np.sqrt(np.mean(response[:,0]**2))),
            'max_defect':float(np.sqrt(np.max(np.mean(channels*channels,axis=1))))}


def choose_source_spline_degree(times, values, selected_indices, corner_indices,
                               theta, nu, cubic_builder, quintic_builder,
                               improvement=.75, max_defect_ratio=2.):
    """Return degree and diagnostics; a strict improvement selects quintic.

    ``corner_indices`` refers to the complete pilot. Builders receive the
    acquired source-time vector, pilot values at those times, and its local
    break indices. They must return ordinary vector-valued interpolants.
    """
    times=np.asarray(times,float);values=np.asarray(values,float)
    selected=np.unique(np.asarray(selected_indices,dtype=int))
    corners=np.unique(np.asarray(corner_indices,dtype=int))
    if (len(selected)<2 or selected[0]!=0 or selected[-1]!=len(times)-1
            or np.any(selected<0) or np.any(selected>=len(times))):
        raise ValueError('selected source knots must cover the pilot endpoints')
    breaks=np.flatnonzero(np.isin(selected,corners))
    diagnostics={}
    for degree,builder in [(3,cubic_builder),(5,quintic_builder)]:
        try:
            interpolant=builder(times[selected],values[selected],breaks)
            residual=interpolant(times)-values
            diagnostics[str(degree)]=residual_response_diagnostic(times,residual,theta,nu)
        except (ValueError,ArithmeticError,np.linalg.LinAlgError) as error:
            if degree==3:raise
            diagnostics['5']={'failure':str(error)}
            return 3,diagnostics
    cubic=diagnostics['3'];quintic=diagnostics['5']
    roundoff=128*np.finfo(float).eps*max(1.,float(np.max(abs(values@_channel_map(theta)))))
    degree=5 if (cubic['response_rms']>roundoff*(times[-1]-times[0])
                 and quintic['response_rms']<improvement*cubic['response_rms']
                 and quintic['max_defect']<=max_defect_ratio*cubic['max_defect']) else 3
    return degree,diagnostics
