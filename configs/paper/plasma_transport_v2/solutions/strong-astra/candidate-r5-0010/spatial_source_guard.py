"""Independent public-pilot guard for a reduced late source spatial mesh.

The velocity quadrature and Hermite variance must agree in both inputs. This
guard calibrates against an already acquired full spatial source snapshot;
it performs no Oracle access and assumes no source frequency support.
"""
from __future__ import annotations
import numpy as np


def pad_fourier_modes(modes, target_n):
    """Embed odd-grid unnormalized FFT modes at the same physical period."""
    modes=np.asarray(modes);source_n=modes.shape[-1]
    if (modes.shape[-2]!=source_n or source_n%2!=1 or target_n%2!=1
            or target_n<source_n):
        raise ValueError('ordered odd square Fourier meshes required')
    if target_n==source_n:return modes.copy()
    k=np.rint(np.fft.fftfreq(source_n)*source_n).astype(int)%target_n
    result=np.zeros(modes.shape[:-2]+(target_n,target_n),dtype=modes.dtype)
    result[...,k[:,None],k[None,:]]=modes*(target_n/source_n)**2
    return result


class SpatialSourceGuard:
    def __init__(self,n,xy,periods,degree,reference_modes,reference_values):
        self.n=int(n);self.degree=int(degree)
        xy=np.asarray(xy);periods=np.asarray(periods)
        k=np.rint(np.fft.fftfreq(n)*n).astype(int)
        self.phase=np.exp(2j*np.pi*(xy[:,0,None,None]/periods[0]*k[:,None]
                                  +xy[:,1,None,None]/periods[1]*k[None,:]))/n**2
        powers=np.array([d for d in range(degree+1) for _ in range(d+1)])
        self.bands=[(powers>=lo)&(powers<=hi) for lo,hi in [(0,0),(1,3),(4,8),(9,degree)]]
        self.bands=[band for band in self.bands if np.any(band)]
        reference_values=np.asarray(reference_values)
        defect=self.project(reference_modes)-reference_values
        self.thresholds=[];self.reference_maxima=[]
        for band in self.bands:
            block=defect[:,band]
            maximum=float(np.max(abs(block)));rms=float(np.sqrt(np.mean(block*block)))
            roundoff=256*np.finfo(float).eps*(1+float(np.max(abs(reference_values[:,band]))))
            self.thresholds.append(max(3*maximum,8*rms,roundoff))
            self.reference_maxima.append(maximum)
        # A poorly resolved calibration snapshot does not establish a small
        # precision floor. In that case preserve the full source mesh.
        caps=[2.e-5,1.e-4,3.e-4,1.e-3][:len(self.bands)]
        self.enabled=all(value<=cap for value,cap in zip(self.reference_maxima,caps))
        self.accepted=self.rejected=self.consecutive_rejections=0

    def project(self,modes):
        return np.einsum('cij,pij->pc',modes,self.phase).real

    def accept(self,modes,pilot_values):
        defect=self.project(modes)-np.asarray(pilot_values)
        maxima=[float(np.max(abs(defect[:,band]))) for band in self.bands]
        accepted=(self.enabled and np.isfinite(defect).all()
                  and all(value<=threshold for value,threshold in zip(maxima,self.thresholds)))
        if accepted:
            self.accepted+=1;self.consecutive_rejections=0
        else:
            self.rejected+=1;self.consecutive_rejections+=1
            if self.consecutive_rejections>=3:self.enabled=False
        return accepted

    def diagnostics(self):
        return {'enabled':self.enabled,'accepted':self.accepted,'rejected':self.rejected,
                'reference_maxima':self.reference_maxima,'thresholds':self.thresholds}
