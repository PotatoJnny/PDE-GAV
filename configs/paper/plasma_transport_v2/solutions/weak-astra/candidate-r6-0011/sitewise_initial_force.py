"""Evaluate the initial electric-force product before spatial projection.

All inputs are candidate-owned initial Fourier representations. This helper
does not query the public Oracle. The correction replaces interpolation of the
NX-grid product by multiplication of the separately interpolated factors.
"""
from __future__ import annotations
import numpy as np


def fourier_values(modes, xy, periods):
    """Evaluate real Fourier fields; preserve arbitrary leading field axes."""
    modes=np.asarray(modes);xy=np.asarray(xy,float);periods=np.asarray(periods,float)
    n=modes.shape[-1]
    if modes.shape[-2:]!=(n,n) or xy.ndim!=2 or xy.shape[1]!=2 or periods.shape!=(2,):
        raise ValueError('square Fourier modes, (sites,2) coordinates and 2 periods required')
    leading=modes.shape[:-2]
    frequencies=[2*np.pi*np.fft.fftfreq(n,d=float(period)/n) for period in periods]
    unique_x,index_x=np.unique(xy[:,0],return_inverse=True)
    unique_y,index_y=np.unique(xy[:,1],return_inverse=True)
    if len(unique_x)*len(unique_y)<=2*len(xy):
        # Source meshes (and row blocks thereof) have repeated axes. Avoid
        # forming one nv*NX*NX contraction independently for every mesh cell.
        phase_x=np.exp(1j*unique_x[:,None]*frequencies[0])
        phase_y=np.exp(1j*unique_y[:,None]*frequencies[1])
        grid=np.einsum('cij,ai,bj->cab',modes.reshape(-1,n,n),phase_x,phase_y,optimize=True)
        values=grid[:,index_x,index_y].real
    else:
        phase_x=np.exp(1j*xy[:,0,None]*frequencies[0]);phase_y=np.exp(1j*xy[:,1,None]*frequencies[1])
        values=np.einsum('cij,si,sj->cs',modes.reshape(-1,n,n),phase_x,phase_y,optimize=True).real
    return values.reshape(leading+(len(xy),))


def point_values(modes,xy,periods):
    """Drop-in incumbent point projector, with Cartesian contraction reuse."""
    return fourier_values(modes,xy,periods)


def sitewise_initial_operator(fine_operator_modes, initial_force_modes,
                              field_modes, velocity_gradient_modes, xy, periods,
                              *,check=lambda:None,site_block=256):
    """Correct NX-grid electric-product aliasing without changing linear terms.

`fine_operator_modes` and `initial_force_modes` have (nv,n,n) axes; the
initial force is -E0.grad_v(f0). `velocity_gradient_modes` has (2,nv,n,n)
axes. `field_modes` has (2,n,n) axes. Return the corrected full Lf0 with
(sites,nv) axes, matching the incumbent's source sample layout.

The difference is zero on the original initial grid, up to roundoff, when
the product's Nyquist lines vanish. On a nondivisor coarse mesh it avoids
aliasing the same product a second time through the initial NX mesh.
"""
    check();xy=np.asarray(xy,float)
    fine=np.asarray(fine_operator_modes);force=np.asarray(initial_force_modes)
    gradient=np.asarray(velocity_gradient_modes);field=np.asarray(field_modes)
    if fine.ndim!=3 or force.shape!=fine.shape or gradient.shape!=(2,)+fine.shape or field.shape!=(2,)+fine.shape[-2:]:
        raise ValueError('compatible initial operator, force, gradient and field modes required')
    output=np.empty((len(xy),fine.shape[0]),float)
    linear=fine-force
    for first in range(0,len(xy),int(site_block)):
        check();sites=xy[first:first+int(site_block)]
        resolved=fourier_values(linear,sites,periods)
        e=fourier_values(field,sites,periods)
        dv=fourier_values(gradient,sites,periods)
        output[first:first+len(sites)]=(resolved-np.einsum('ds,dvs->vs',e,dv)).T
    check();return output
