"""Separable continuous-velocity OU kernels for tensor cardinal cubic splines.

Research helper only. Source samples are unweighted physical densities on a
rotated/scaled tensor grid v = physical_center + transform @ z. The spline is
zero outside the rectangle bounded by its outer nodes. Kernel entries already
include the physical Lebesgue Jacobian; no GH integration weights are used.
"""
from __future__ import annotations
import numpy as np
from scipy.interpolate import CubicSpline
from green_ops import ou_parameters, PAIRS, _kernel_terms


def _multiply(left,right):
    result={}
    for (a,b),x in left.items():
        for (c,d),y in right.items():
            pair=(a+c,b+d)
            result[pair]=result.get(pair,0.)+x*y
    return result


def _gaussian_polynomials(linear,variance):
    one={(0,0):np.ones_like(variance)}
    square=_multiply(linear,linear)
    second=dict(square);second[0,0]=second.get((0,0),0.)+variance
    third=_multiply(square,linear)
    for power,value in linear.items():third[power]=third.get(power,0.)+3*variance*value
    return [one,linear,second,third]


class SplineOUKernel:
    def __init__(self,z_nodes,transform,physical_center,bulk_center,nu,theta,
                 *,quadrature_order=8,bc_type='natural',gh_blend=False,axis_blend=False,
                 blend_scale=1.,blend_power=8.):
        self.z=np.asarray(z_nodes,float)
        self.transform=np.asarray(transform,float)
        self.center=np.asarray(physical_center,float)
        self.bulk=np.asarray(bulk_center,float)
        self.nu=float(nu);self.theta=float(theta);self.quadrature_order=int(quadrature_order)
        self.bc_type=bc_type
        self.gh_blend=bool(gh_blend)
        self.axis_blend=bool(axis_blend)
        self.blend_scale=float(blend_scale);self.blend_power=float(blend_power)
        if min(self.blend_scale,self.blend_power)<=0 or not np.isfinite(self.blend_scale+self.blend_power):
            raise ValueError('finite positive blend scale and exponent required')
        if self.axis_blend and not self.gh_blend:
            raise ValueError('axis blending requires GH blending')
        if (self.z.ndim!=1 or len(self.z)<4 or np.any(np.diff(self.z)<=0)
                or self.transform.shape!=(2,2) or self.center.shape!=(2,) or self.bulk.shape!=(2,)
                or self.quadrature_order<1):
            raise ValueError('ordered spline nodes, nonsingular 2x2 transform, and valid quadrature required')
        self.jacobian=float(abs(np.linalg.det(self.transform)))
        if not np.isfinite(self.jacobian) or self.jacobian<=0:raise ValueError('nonsingular finite transform required')
        self.cardinal=CubicSpline(self.z,np.eye(len(self.z)),axis=0,bc_type=bc_type,extrapolate=False)
        self.second_derivatives=self.cardinal(self.z,2)
        zz=np.stack(np.meshgrid(self.z,self.z,indexing='ij'),-1).reshape(-1,2)
        self.physical_velocity=self.center+zz@self.transform.T
        gh_roots,gh_weights=np.polynomial.hermite.hermgauss(len(self.z))
        if self.gh_blend and not np.allclose(self.z,np.sqrt(2)*gh_roots,rtol=1e-12,atol=1e-12):
            raise ValueError('GH blending requires standard probabilists GH nodes')
        one=np.sqrt(2)*gh_weights*np.exp(self.z**2/2)
        self.gh_axis_integration=one
        self.gh_integration=self.jacobian*np.outer(one,one).ravel()
        self._quadratures={}

    def _quadrature(self,order):
        if order not in self._quadratures:
            roots,weights=np.polynomial.legendre.leggauss(order)
            width=np.diff(self.z);midpoint=.5*(self.z[1:]+self.z[:-1])
            locations=midpoint[:,None]+.5*width[:,None]*roots
            weights=.5*width[:,None]*np.broadcast_to(weights,(len(width),order))
            r=(roots+1)/2
            local=np.stack((np.broadcast_to(1-r,locations.shape),np.broadcast_to(r,locations.shape),
                            width[:,None]**2*((1-r)**3-(1-r))/6,
                            width[:,None]**2*(r**3-r)/6))
            tensor=np.stack([weights[None]*locations[None]**p*local for p in range(4)])
            self._quadratures[order]=(locations,tensor)
        return self._quadratures[order]

    def _integrals(self,frequency):
        # Each integrand is a polynomial times an oscillatory exponential.
        # Increase the per-cell rule when necessary; no upper cap is imposed.
        phase=float(np.max(abs(frequency)))*float(np.max(np.diff(self.z)))
        order=max(self.quadrature_order,int(np.ceil(phase/2))+4)
        locations,tensor=self._quadrature(order)
        shape=frequency.shape
        phase=np.exp(-1j*np.asarray(frequency).reshape(-1,1,1)*locations[None])
        local=np.einsum('fcq,plcq->fplc',phase,tensor,optimize=True)
        direct=np.zeros((phase.shape[0],4,len(self.z)),complex)
        curvature=np.zeros_like(direct)
        direct[:,:,:-1]+=local[:,:,0];direct[:,:,1:]+=local[:,:,1]
        curvature[:,:,:-1]+=local[:,:,2];curvature[:,:,1:]+=local[:,:,3]
        values=(direct+curvature@self.second_derivatives).reshape(shape+(4,len(self.z)))
        if self.axis_blend:
            omega=np.asarray(frequency)
            weight=-np.expm1(-(np.abs(omega)*float(np.min(np.diff(self.z)))/(np.pi*self.blend_scale))**self.blend_power)
            gh=np.exp(-1j*omega[...,None]*self.z)*self.gh_axis_integration
            gh=gh[...,None,:]*np.stack([self.z**p for p in range(4)])
            values=weight[...,None,None]*values+(1-weight[...,None,None])*gh
        return values

    def iter_kernel_terms(self,tau,wavevectors,*,density_only=False):
        tau=np.atleast_1d(np.asarray(tau,float));k=np.asarray(wavevectors,float)
        if tau.ndim!=1 or k.ndim!=2 or k.shape[1]!=2:raise ValueError('1D times and (modes,2) wavevectors required')
        decay,a,variance,cross,displacement=ou_parameters(tau,self.nu,self.theta)
        rotated=k@self.transform
        ix=self._integrals(a[:,None]*rotated[None,:,0])
        iy=self._integrals(a[:,None]*rotated[None,:,1])
        phase=self.jacobian*np.exp(-1j*a[:,None]*(k@self.center)[None]-.5*displacement[:,None]*np.sum(k*k,axis=1)[None])
        if self.gh_blend and not self.axis_blend:
            dimensionless=np.max(abs(a[:,None,None]*rotated[None]),axis=-1)*float(np.min(np.diff(self.z)))
            spline_weight=-np.expm1(-(dimensionless/(np.pi*self.blend_scale))**self.blend_power)
            gh_terms=_kernel_terms(tau,k,self.physical_velocity,self.bulk,self.nu,self.theta,density_only)
        def blend(kernel):
            if not self.gh_blend or self.axis_blend:return kernel
            gh=next(gh_terms)*self.gh_integration[None,None]
            return spline_weight[:,:,None]*kernel+(1-spline_weight[:,:,None])*gh
        count=1 if density_only else len(PAIRS)
        if density_only:
            yield blend((phase[:,:,None,None]*ix[:,:,0,:,None]*iy[:,:,0,None,:]).reshape(len(tau),len(k),-1))
            return
        variance=np.broadcast_to(variance[:,None],phase.shape)
        polynomials=[]
        for axis in range(2):
            linear={(0,0):decay[:,None]*self.center[axis]-self.bulk[axis]-1j*cross[:,None]*k[None,:,axis],
                    (1,0):np.broadcast_to(decay[:,None]*self.transform[axis,0],phase.shape),
                    (0,1):np.broadcast_to(decay[:,None]*self.transform[axis,1],phase.shape)}
            polynomials.append(_gaussian_polynomials(linear,variance))
        for degree_x,degree_y in PAIRS[:count]:
            polynomial=_multiply(polynomials[0][degree_x],polynomials[1][degree_y])
            result=np.zeros(phase.shape+(len(self.z),len(self.z)),complex)
            # First contract the small polynomial in the x-axis node index.
            # Grouping by y power halves the expensive full tensor products.
            for q in range(degree_x+degree_y+1):
                left=np.zeros(phase.shape+(len(self.z),),complex)
                for (p,power_y),coefficient in polynomial.items():
                    if power_y==q:left+=coefficient[:,:,None]*ix[:,:,p,:]
                result+=left[:,:,:,None]*iy[:,:,q,None,:]
            yield blend((phase[:,:,None,None]*result).reshape(len(tau),len(k),-1))

    def kernel(self,tau,wavevectors,*,density_only=False):
        """Return (moments,times,modes,n_nodes**2), including physical dv weights."""
        return np.stack(list(self.iter_kernel_terms(tau,wavevectors,density_only=density_only)))
