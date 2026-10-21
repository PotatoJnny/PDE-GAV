"""Guarded512-point source queries for a fixed1024-point evolution grid."""
import numpy as np


class SourceQueryGrid:
    def __init__(self,x,velocity,quadrature,tmin,tmax):
        self.x=np.asarray(x);self.v=np.asarray(velocity)
        self.nx=len(x);self.nv=len(velocity)
        if self.nx!=1024:raise ValueError('SourceQueryGrid requires fixed1024 evolution')
        self.coarse_nx=512;self.duration=tmax-tmin
        self.weight=np.asarray(quadrature)*(1+np.abs(velocity)**3)
        length=(x[1]-x[0])*self.nx
        self.check_x=x[0]+length*(np.arange(17)+.3819660112501051)/17
        self.frequencies=np.fft.fftfreq(512,1/512)
        self.interpolation=np.exp(2j*np.pi*(self.check_x-x[0])[:,None]/length*self.frequencies)
        self.info=dict(solver_nx=1024,source_query_nx=512,slices=0,interpolated_slices=0,
                       exact_slices=0,probe_points=0,tail_tolerance=1e-6,
                       shifted_tolerance=1e-8,nyquist_tolerance=1e-12,cutoff=128)

    def matches(self,points):
        if points.ndim!=2 or points.shape!=(self.nx*self.nv,3):return False
        if not np.all(points[:,0]==points[0,0]):return False
        shaped=points.reshape(self.nx,self.nv,3)
        return np.array_equal(shaped[:,0,1],self.x) and np.array_equal(shaped[0,:,2],self.v)

    def sample(self,points,raw_query,reserve):
        shaped=points.reshape(self.nx,self.nv,3)
        even=raw_query(shaped[::2].reshape(-1,3),reserve=reserve).reshape(512,self.nv)
        coefficients=np.fft.fft(even,axis=0)/512
        scale=max(float(np.mean(np.abs(even)@self.weight))*self.duration,1e-12)
        tail=float(np.sum(np.abs(coefficients[np.abs(self.frequencies)>=128])@self.weight))*self.duration/scale
        nyquist=float(np.abs(coefficients[256])@self.weight)*self.duration/scale
        self.info.update(last_time=float(points[0,0]),last_tail=tail,last_nyquist=nyquist)
        accepted=np.isfinite(even).all() and np.isfinite(tail) and tail<=1e-6 and nyquist<=1e-12
        coefficients[256]=0.
        if accepted:
            # Probe even when every coarse value is zero: hidden512 modes can
            # vanish on the entire even lattice while remaining nonzero here.
            probe=np.column_stack((np.full(17*self.nv,points[0,0]),
                                   np.repeat(self.check_x,self.nv),np.tile(self.v,17)))
            exact=raw_query(probe,reserve=reserve).reshape(17,self.nv)
            predicted=(self.interpolation@coefficients).real
            defect=float(np.mean(np.abs(predicted-exact)@self.weight))*self.duration/scale
            self.info['probe_points']+=len(probe)
            self.info['last_shifted_defect']=defect
            accepted=np.isfinite(defect) and defect<=1e-8
        self.info['slices']+=1
        if not accepted:
            # Both halves are exact Oracle values, including all high modes.
            values=np.empty((1024,self.nv))
            values[::2]=even
            values[1::2]=raw_query(shaped[1::2].reshape(-1,3),reserve=reserve).reshape(512,self.nv)
            self.info['exact_slices']+=1
            return values.ravel()
        padded=np.zeros((1024,self.nv),dtype=np.complex128)
        padded[:256]=coefficients[:256]
        padded[-255:]=coefficients[-255:]
        values=np.fft.ifft(padded,axis=0).real*1024
        values[::2]=even
        self.info['interpolated_slices']+=1
        return values.ravel()
