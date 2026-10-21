"""Common-velocity Fourier/Hermite correction operators, with no Oracle access."""
from __future__ import annotations
import math
import numpy as np

PAIRS=tuple((degree-b,b) for degree in range(4) for b in range(degree+1))


def _product(left,right):
    answer={}
    for (a,b),x in left.items():
        for (c,d),y in right.items():answer[a+c,b+d]=answer.get((a+c,b+d),0.)+x*y
    return answer


class HermiteTensor:
    def __init__(self,order,covariance,scale,origin_relative,bulk_center,nu,theta):
        self.order=int(order);self.nu=float(nu);self.theta=float(theta)
        self.origin=np.asarray(origin_relative,float);self.bulk=np.asarray(bulk_center,float)
        self.center=self.origin+self.bulk
        eigen,self.rotation=np.linalg.eigh(np.asarray(covariance,float))
        if self.order<4 or np.min(eigen)<=0 or scale<=0 or nu<0 or theta<=0:
            raise ValueError('positive covariance, scale, theta and valid order/nu required')
        self.sigma=float(scale)*np.sqrt(eigen)
        nodes,weights=np.polynomial.hermite.hermgauss(self.order)
        self.z=np.sqrt(2.)*nodes;self.w=weights/np.sqrt(np.pi)
        he=np.empty((self.order,self.order));he[0]=1.;he[1]=self.z
        for degree in range(1,self.order-1):
            he[degree+1]=(self.z*he[degree]-np.sqrt(degree)*he[degree-1])/np.sqrt(degree+1)
        self.G=he*np.sqrt(self.w)[None]
        zz=np.stack(np.meshgrid(self.z,self.z,indexing='ij'),-1)
        self.relative_velocity=self.origin+np.einsum('ij,abj->abi',self.rotation,zz*self.sigma)
        self.physical_velocity=self.relative_velocity+self.bulk
        one=np.sqrt(2*np.pi)*self.w*np.exp(self.z**2/2)
        self.integration=self.sigma.prod()*one[:,None]*one[None,:]
        self.sqrt_weight=np.sqrt(self.w[:,None]*self.w[None,:])
        self.value_to_nodal=self.integration/self.sqrt_weight
        self.moment_weights=np.stack([self.sqrt_weight*self.relative_velocity[...,0]**a*
                                      self.relative_velocity[...,1]**b for a,b in PAIRS])
        self.degree=np.arange(self.order,dtype=float)
        self.ladder=np.sqrt(self.degree[1:])
        self.second=np.sqrt(self.degree[2:]*(self.degree[2:]-1))
        self._low_transform=self._make_low_transform()
        orthogonality=float(np.max(abs(self.G@self.G.T-np.eye(self.order))))
        if orthogonality>1e-10:raise ValueError('Hermite quadrature transform is not orthogonal')

    def _make_low_transform(self):
        polynomials=[]
        for axis in range(2):
            direction=self.rotation[:,axis]/self.sigma[axis]
            linear={(0,0):-float(direction@self.origin),(1,0):direction[0],(0,1):direction[1]}
            polys=[{(0,0):1.},linear]
            for degree in range(1,3):
                next_poly=_product(linear,polys[-1])
                for power,coefficient in polys[-2].items():
                    next_poly[power]=next_poly.get(power,0.)-np.sqrt(degree)*coefficient
                polys.append({power:coefficient/np.sqrt(degree+1) for power,coefficient in next_poly.items()})
            polynomials.append(polys)
        result=[]
        for a,b in PAIRS:
            poly=_product(polynomials[0][a],polynomials[1][b])
            result.append([poly.get(pair,0.) for pair in PAIRS])
        return np.asarray(result)

    def to_hermite(self,nodal):return self.G@np.asarray(nodal)@self.G.T
    def to_nodal(self,hermite):return self.G.T@np.asarray(hermite)@self.G
    def from_values(self,values):return np.asarray(values)*self.value_to_nodal
    def raw_moments(self,nodal):return np.einsum('...ij,mij->m...',nodal,self.moment_weights,optimize=True)

    def patch_low_moments(self,nodal,raw):
        hermite=self.to_hermite(nodal)
        target=np.einsum('ab,b...->a...',self._low_transform,np.asarray(raw)[:10],optimize=True)
        for row,(a,b) in enumerate(PAIRS):hermite[...,a,b]=target[row]
        return self.to_nodal(hermite)

    def force_rhs(self,hermite,field):
        h=np.asarray(hermite);field=np.asarray(field)
        acceleration=np.einsum('...i,ij->...j',field,self.rotation)/self.sigma
        result=np.zeros_like(h)
        result[...,1:,:]+=acceleration[...,0,None,None]*self.ladder[:,None]*h[...,:-1,:]
        result[...,:,1:]+=acceleration[...,1,None,None]*self.ladder[None,:]*h[...,:,:-1]
        return result

    def velocity_rhs(self,hermite,field):
        h=np.asarray(hermite)
        result=-self.nu*(self.degree[:,None]+self.degree[None,:])*h
        result+=self.force_rhs(h,np.asarray(field)-self.nu*self.center)
        result[...,2:,:]+=self.nu*(self.theta/self.sigma[0]**2-1)*self.second[:,None]*h[...,:-2,:]
        result[...,:,2:]+=self.nu*(self.theta/self.sigma[1]**2-1)*self.second[None,:]*h[...,:,:-2]
        return result

    def absorber(self,strength=40.,power=8):
        rates=float(strength)*((self.degree[:,None]/(self.order-1))**power+
                               (self.degree[None,:]/(self.order-1))**power)
        rates[self.degree[:,None]+self.degree[None,:]<=3]=0.
        return rates

    def streaming_rhs(self,nodal,periods):
        nx,ny=nodal.shape[:2]
        kx,ky=np.meshgrid(2*np.pi*np.fft.fftfreq(nx,d=float(periods[0])/nx),
                          2*np.pi*np.fft.fftfreq(ny,d=float(periods[1])/ny),indexing='ij')
        frequency=kx[...,None,None]*self.physical_velocity[...,0]+ky[...,None,None]*self.physical_velocity[...,1]
        modes=np.fft.fft2(nodal,axes=(0,1))
        if nx%2==0:modes[nx//2]=0.
        if ny%2==0:modes[:,ny//2]=0.
        return np.fft.ifft2(-1j*frequency*modes,axes=(0,1)).real


def electric_field(density,periods):
    nx,ny=density.shape
    kx,ky=np.meshgrid(2*np.pi*np.fft.fftfreq(nx,d=float(periods[0])/nx),
                      2*np.pi*np.fft.fftfreq(ny,d=float(periods[1])/ny),indexing='ij')
    k2=kx*kx+ky*ky;inverse=np.divide(1.,k2,out=np.zeros_like(k2),where=k2>0)
    rh=np.fft.fft2(density)
    return np.stack((np.fft.ifft2(-1j*kx*inverse*rh).real,
                     np.fft.ifft2(-1j*ky*inverse*rh).real),axis=-1)


def spatial_project(values,n):
    """Project a real field with leading two spatial axes to an n² mesh."""
    old=values.shape[0]
    if values.shape[1]!=old or n<old:raise ValueError('spatial projection only prolongs square meshes')
    modes=np.fft.fft2(values,axes=(0,1))/old**2
    if old%2==0:modes[old//2]=0.;modes[:,old//2]=0.
    frequencies=np.rint(np.fft.fftfreq(old)*old).astype(int)
    out=np.zeros((n,n)+values.shape[2:],complex)
    out[frequencies[:,None]%n,frequencies[None,:]%n]=modes
    return np.fft.ifft2(out,axes=(0,1)).real*n*n


def evolve_correction(operator,initial_nodal,source_times,defect_nodal,periods,
                      *,dt_max=.005,absorber_strength=0.,check=None):
    """Strang stream/RK2 velocity evolution of delta=f-f0 around static f0.

    defect_nodal is S+L_fine(f0) at each source time, already projected onto
    the evolution mesh. Only delta receives the high-Hermite absorber.
    """
    check=check or (lambda:None);check()
    initial_nodal=np.asarray(initial_nodal,float);times=np.asarray(source_times,float)
    source=np.asarray(defect_nodal,float)
    if (len(times)<2 or np.any(np.diff(times)<=0) or source.shape!=(len(times),)+initial_nodal.shape
            or not np.isfinite(source).all() or not np.isfinite(initial_nodal).all() or dt_max<=0):
        raise ValueError('finite initial/source tensors and ordered times required')
    nx,ny=initial_nodal.shape[:2]
    h0=operator.to_hermite(initial_nodal);field0=electric_field(h0[...,0,0],periods)
    source_h=operator.to_hermite(source)
    # Periodic Poisson compatibility requires constant total particle number.
    source_h[...,0,0]-=source_h[...,0,0].mean(axis=(1,2))[:,None,None]
    kx,ky=np.meshgrid(2*np.pi*np.fft.fftfreq(nx,d=float(periods[0])/nx),
                      2*np.pi*np.fft.fftfreq(ny,d=float(periods[1])/ny),indexing='ij')
    frequency=kx[...,None,None]*operator.physical_velocity[...,0]+ky[...,None,None]*operator.physical_velocity[...,1]
    rates=operator.absorber(absorber_strength)
    delta=np.zeros_like(initial_nodal)
    moments=np.zeros((len(times),10,nx,ny));steps=0;largest=0.;largest_h=0.;largest_u=0.
    for frame in range(len(times)-1):
        check();width=float(times[frame+1]-times[frame]);count=max(1,int(np.ceil(width/dt_max)));dt=width/count
        phase=np.exp(-.5j*dt*frequency);damping=np.exp(-.5*dt*rates)
        if nx%2==0:phase[nx//2]=0.
        if ny%2==0:phase[:,ny//2]=0.
        for step in range(count):
            check()
            delta=np.fft.ifft2(phase*np.fft.fft2(delta,axes=(0,1)),axes=(0,1)).real
            h=operator.to_hermite(delta)*damping
            alpha=(step+.5)/count;forcing=(1-alpha)*source_h[frame]+alpha*source_h[frame+1]
            field=electric_field(h[...,0,0],periods)
            k1=operator.velocity_rhs(h,field0+field)+operator.force_rhs(h0,field)+forcing
            midpoint=h+.5*dt*k1
            field=electric_field(midpoint[...,0,0],periods)
            k2=operator.velocity_rhs(midpoint,field0+field)+operator.force_rhs(h0,field)+forcing
            h=(h+dt*k2)*damping
            delta=operator.to_nodal(h)
            delta=np.fft.ifft2(phase*np.fft.fft2(delta,axes=(0,1)),axes=(0,1)).real
            if not np.isfinite(delta).all():raise FloatingPointError('nonfinite kinetic correction')
            steps+=1
        moments[frame+1]=operator.raw_moments(delta)
        largest=max(largest,float(np.max(abs(moments[frame+1]))))
        largest_h=max(largest_h,float(np.max(abs(h))));largest_u=max(largest_u,float(np.max(abs(delta))))
    check();return moments,{'kinetic_steps':steps,'kinetic_max_raw_increment':largest,
                           'kinetic_dt_max':float(dt_max),'kinetic_absorber_strength':float(absorber_strength),
                           'kinetic_initial_max_abs_hermite':float(np.max(abs(h0))),
                           'kinetic_forcing_max_abs_hermite':float(np.max(abs(source_h))),
                           'kinetic_max_abs_hermite_delta':largest_h,'kinetic_max_abs_nodal_delta':largest_u}
