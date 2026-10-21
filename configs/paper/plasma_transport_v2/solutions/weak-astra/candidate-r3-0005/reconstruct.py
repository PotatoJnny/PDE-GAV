"""Batched tensor-source reconstruction with bounded intermediate storage.

This module does not import Torch unless a Torch reconstructor is constructed.
The interpolation factors must come from the uncorrected tensor rule: empirical
moment corrections do not define a corresponding full-source interpolant.
"""
from __future__ import annotations

import numpy as np


def reconstruct_numpy(values, left, right, *, out=None, batch_size=128,
                      dtype=np.float32):
    """Return (...,nvx,nvy) from (...,selected_x,selected_y).

    If ``out`` is supplied, write there and return that exact object.  A NumPy
    memmap is supported.  The input/output must be C contiguous; no full-size
    dtype conversion is made.  Each batch uses two dense matrix multiplications
    and at most a batch-size-by-nvx-by-selected_y intermediate.
    """
    a=np.asarray(values)
    lx=np.asarray(left,dtype=dtype);ly=np.asarray(right,dtype=dtype)
    if a.ndim < 2 or a.shape[-2:] != (lx.shape[1],ly.shape[1]):
        raise ValueError("sample values and interpolation factors disagree")
    if not a.flags.c_contiguous or batch_size < 1:
        raise ValueError("C-contiguous input and positive batch_size required")
    shape=a.shape[:-2]+(lx.shape[0],ly.shape[0])
    if out is None:
        out=np.empty(shape,dtype=dtype)
    if out.shape != shape or not out.flags.c_contiguous:
        raise ValueError("out must have correct shape and be C contiguous")
    src=a.reshape((-1,)+a.shape[-2:])
    dst=out.reshape((-1,)+shape[-2:])
    for start in range(0,len(src),batch_size):
        end=min(start+batch_size,len(src))
        block=np.asarray(src[start:end],dtype=dtype)
        intermediate=np.matmul(lx,block)
        np.matmul(intermediate,ly.T,out=dst[start:end])
    return out


class TorchCPUReconstructor:
    """Cached Torch CPU factors; reconstruct host arrays into host storage.

    Torch may use its configured CPU thread count.  The caller controls that
    global setting; constructing this object does not alter it or select a GPU.
    """
    def __init__(self,left,right,*,dtype=np.float32):
        import torch
        self.torch=torch
        self.dtype=np.dtype(dtype)
        if self.dtype not in (np.dtype(np.float32),np.dtype(np.float64)):
            raise ValueError("Torch CPU reconstruction supports float32/64")
        self.left=torch.from_numpy(np.ascontiguousarray(left,dtype=self.dtype))
        self.right=torch.from_numpy(np.ascontiguousarray(right,dtype=self.dtype)).T

    def __call__(self,values,*,out=None,batch_size=128):
        a=np.asarray(values)
        if a.ndim < 2 or a.shape[-2:] != (self.left.shape[1],self.right.shape[0]):
            raise ValueError("sample values and interpolation factors disagree")
        if not a.flags.c_contiguous or batch_size < 1:
            raise ValueError("C-contiguous input and positive batch_size required")
        shape=a.shape[:-2]+(self.left.shape[0],self.right.shape[1])
        if out is None:
            out=np.empty(shape,dtype=self.dtype)
        if out.shape != shape or not out.flags.c_contiguous or out.dtype != self.dtype:
            raise ValueError("out must have correct shape/dtype and be C contiguous")
        src=a.reshape((-1,)+a.shape[-2:]);dst=out.reshape((-1,)+shape[-2:])
        with self.torch.no_grad():
            for start in range(0,len(src),batch_size):
                end=min(start+batch_size,len(src))
                block=self.torch.from_numpy(np.ascontiguousarray(src[start:end],dtype=self.dtype))
                destination=self.torch.from_numpy(dst[start:end])
                self.torch.matmul(self.torch.matmul(self.left,block),self.right,out=destination)
        return out

