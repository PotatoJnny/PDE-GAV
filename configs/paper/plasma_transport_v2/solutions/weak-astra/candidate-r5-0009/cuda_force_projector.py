"""Double-precision GPU projection of the frozen-initial electric force.

Keeps the fine-grid velocity gradient on the assigned device across field
iterations. Products, FFT normalization, frequency extraction and the two
Nyquist conventions match resample_modes(project(-sum(gradient*field)), nx).
Only the field and final coarse Fourier coefficients cross the device boundary.
"""
from __future__ import annotations

import numpy as np
import torch


class CudaForceProjector:
    def __init__(self, gradient_oversampled, nx, *, velocity_block=256,
                 prefer_cuda=True, check=None):
        self.check = check or (lambda: None)
        self.check()
        self.gradient_host = np.asarray(gradient_oversampled, dtype=np.float64)
        self.nx = int(nx)
        self.velocity_block = int(velocity_block)
        if (self.gradient_host.ndim != 4 or self.gradient_host.shape[0] != 2
                or self.gradient_host.shape[-2] != self.gradient_host.shape[-1]
                or self.nx < 1 or self.nx > self.gradient_host.shape[-1]
                or self.velocity_block < 1 or not np.isfinite(self.gradient_host).all()):
            raise ValueError('finite (2,nv,n,n) gradient and valid coarse grid/block required')
        self.nv, self.fine = self.gradient_host.shape[1], self.gradient_host.shape[-1]
        self.indices = np.rint(np.fft.fftfreq(self.nx)*self.nx).astype(np.int64) % self.fine
        self.gradient_device = None
        self.indices_device = None
        self.closed = False
        self.backend = 'numpy'
        self.fallback_reason = None
        if prefer_cuda and torch.cuda.is_available():
            try:
                self.gradient_device = torch.as_tensor(self.gradient_host, device='cuda', dtype=torch.float64)
                self.indices_device = torch.as_tensor(self.indices, device='cuda', dtype=torch.int64)
                self.backend = 'cuda'
            except torch.cuda.OutOfMemoryError as error:
                self._drop_device()
                torch.cuda.empty_cache()
                self.fallback_reason = type(error).__name__+': '+str(error)
        try:
            self.check()
        except BaseException:
            self.close()
            raise

    def _drop_device(self):
        self.gradient_device = None
        self.indices_device = None
        self.backend = 'numpy'

    def _numpy(self, field):
        result = np.empty((self.nv, self.nx, self.nx), complex)
        for first in range(0, self.nv, self.velocity_block):
            self.check()
            last = min(first+self.velocity_block, self.nv)
            force = -np.sum(self.gradient_host[:, first:last]*field[:, None], axis=0)
            modes = np.fft.fft2(force, axes=(-2, -1))/self.fine**2
            if self.fine % 2 == 0:
                modes[..., self.fine//2, :] = 0.
                modes[..., :, self.fine//2] = 0.
            coarse = modes[..., self.indices[:, None], self.indices[None, :]].copy()
            if self.nx % 2 == 0:
                coarse[..., self.nx//2, :] = 0.
                coarse[..., :, self.nx//2] = 0.
            result[first:last] = coarse
        return result

    def _cuda(self, field):
        value = torch.as_tensor(field, device='cuda', dtype=torch.float64)
        result = np.empty((self.nv, self.nx, self.nx), complex)
        for first in range(0, self.nv, self.velocity_block):
            self.check()
            last = min(first+self.velocity_block, self.nv)
            force = -torch.sum(self.gradient_device[:, first:last]*value[:, None], dim=0)
            modes = torch.fft.fft2(force, dim=(-2, -1))/self.fine**2
            if self.fine % 2 == 0:
                modes[..., self.fine//2, :] = 0.
                modes[..., :, self.fine//2] = 0.
            coarse = modes.index_select(-2, self.indices_device).index_select(-1, self.indices_device)
            if self.nx % 2 == 0:
                coarse[..., self.nx//2, :] = 0.
                coarse[..., :, self.nx//2] = 0.
            result[first:last] = coarse.detach().cpu().numpy()
        return result

    def __call__(self, field):
        if self.closed:
            raise RuntimeError('force projector is closed')
        self.check()
        field = np.asarray(field, dtype=np.float64)
        if field.shape != (2, self.fine, self.fine) or not np.isfinite(field).all():
            raise ValueError('finite electric field matching fine gradient grid required')
        if self.backend == 'cuda':
            try:
                return self._cuda(field)
            except torch.cuda.OutOfMemoryError as error:
                # Preserve completed Green work: the stored host gradient gives
                # the identical operation on CPU if a large cache crowds CUDA.
                self._drop_device()
                torch.cuda.empty_cache()
                self.fallback_reason = type(error).__name__+': '+str(error)
        return self._numpy(field)

    @property
    def diagnostics(self):
        return {'backend': self.backend, 'velocity_block': self.velocity_block,
                'gradient_device_bytes': int(self.gradient_host.nbytes) if self.backend == 'cuda' else 0,
                'fallback_reason': self.fallback_reason}

    def close(self):
        self._drop_device()
        self.gradient_host = None
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
