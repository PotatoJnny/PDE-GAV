"""Optional float64 CUDA arithmetic kernels; Torch remains the fallback.

Compile the bundled source in memory with the installed CUDA NVRTC library.
All kernels launch in Torch's current stream and retain the original method's
precision, discretization, and boundary conditions. No external assets/cache.
"""
from __future__ import annotations
import ctypes as ct
import ctypes.util
from pathlib import Path
import torch


_SOURCE = r'''
struct Z { double x, y; };
__device__ __forceinline__ Z add(Z a,Z b){return {a.x+b.x,a.y+b.y};}
__device__ __forceinline__ Z sub(Z a,Z b){return {a.x-b.x,a.y-b.y};}
__device__ __forceinline__ Z mul(Z a,Z b){return {a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x};}
__device__ __forceinline__ Z twice(Z a){return {2.*a.x,2.*a.y};}
__device__ __forceinline__ long ix(long i,int nv,long s0,long s1){return (i/nv)*s0+(i%nv)*s1;}

template<int G> __device__ void dg_impl(
 const double* v,const double* ef,const double* der,const double* left,
 const double* right,const double* ll,const double* lr,const double* ji,
 double* out,int nx,int nc,long s0,long s1,long se){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=nx*nc)return;
 int x=i%nx,c=i/nx;
 double a[G], al=0.,ar=0.,nl=0.,nr=0.;
 #pragma unroll
 for(int p=0;p<G;++p){
   a[p]=v[x*s0+(c*G+p)*s1];
   al+=a[p]*left[p]; ar+=a[p]*right[p];
   if(c>0)nl+=v[x*s0+((c-1)*G+p)*s1]*right[p];
   if(c+1<nc)nr+=v[x*s0+((c+1)*G+p)*s1]*left[p];
 }
 double e=ef[x*se],pos=e>0.?e:0.,neg=e<0.?e:0.;
 double fl=pos*nl+neg*al,fr=pos*ar+neg*nr;
 #pragma unroll
 for(int p=0;p<G;++p){
   double d=0.;
   #pragma unroll
   for(int q=0;q<G;++q)d+=a[q]*der[p*G+q];
   double corr=(fr-e*ar)*lr[p]-(fl-e*al)*ll[p];
   out[x+nx*(c*G+p)]=-(e*d+corr)*ji[c];
 }
}
#define DG(G) extern "C" __global__ void dg##G(const double* v,const double* ef,const double* der,const double* l,const double* r,const double* ll,const double* lr,const double* ji,double* out,int nx,int nc,long s0,long s1,long se){dg_impl<G>(v,ef,der,l,r,ll,lr,ji,out,nx,nc,s0,s1,se);}
DG(4) DG(6) DG(8) DG(12)

extern "C" __global__ void stage2(const Z* p,const Z* f,const Z* q,const Z* k,Z* out,
 int nv,long n,long fs0,long fs1,long ks0,long ks1){
 long i=(long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 out[i]=add(mul(p[i],f[ix(i,nv,fs0,fs1)]),mul(q[i],k[ix(i,nv,ks0,ks1)]));
}
extern "C" __global__ void stage3(const Z* p,const Z* a,const Z* q,const Z* k3,const Z* k1,Z* out,
 int nv,long n,long as0,long as1,long k3s0,long k3s1,long k1s0,long k1s1){
 long i=(long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 out[i]=add(mul(p[i],a[ix(i,nv,as0,as1)]),mul(q[i],sub(twice(k3[ix(i,nv,k3s0,k3s1)]),k1[ix(i,nv,k1s0,k1s1)])));
}
extern "C" __global__ void stage4(const Z* full,const Z* f,const Z* c1,const Z* k1,const Z* c2,const Z* k2,const Z* k3,const Z* c3,const Z* k4,Z* out,
 int nv,long n,long fs0,long fs1,long k1s0,long k1s1,long k2s0,long k2s1,long k3s0,long k3s1,long k4s0,long k4s1){
 long i=(long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 Z v=add(mul(full[i],f[ix(i,nv,fs0,fs1)]),mul(c1[i],k1[ix(i,nv,k1s0,k1s1)]));
 v=add(v,mul(twice(c2[i]),add(k2[ix(i,nv,k2s0,k2s1)],k3[ix(i,nv,k3s0,k3s1)])));
 out[i]=add(v,mul(c3[i],k4[ix(i,nv,k4s0,k4s1)]));
}
'''


def _check(code, operation):
    if code:
        raise RuntimeError('%s returned CUDA status %d' % (operation, code))


def _load_nvrtc():
    """Find the installed CUDA library without assuming a toolkit location."""
    candidates=[ctypes.util.find_library('nvrtc') or 'libnvrtc.so','libnvrtc.so']
    package=Path(torch.__file__).parent
    roots=[package/'lib',package.parent/'nvidia'/'cuda_nvrtc'/'lib']
    # Some HPC Torch builds use RUNPATH rather than LD_LIBRARY_PATH. Discover
    # the already loaded CUDA runtime's directory from this process only.
    try:
        for line in Path('/proc/self/maps').read_text().splitlines():
            parts=line.split()
            if parts and '/libcudart.so' in parts[-1]:
                roots.append(Path(parts[-1]).parent)
    except OSError:
        pass
    for root in dict.fromkeys(roots):
        candidates.extend(str(p) for p in sorted(root.glob('libnvrtc.so*')))
    for path in candidates:
        try:
            return ct.CDLL(path)
        except OSError:
            pass
    raise RuntimeError('Installed NVRTC library unavailable')


class _Runtime:
    def __init__(self):
        self.nvrtc = _load_nvrtc()
        self.cuda = ct.CDLL(ctypes.util.find_library('cuda') or 'libcuda.so.1')
        self.nvrtc.nvrtcCreateProgram.argtypes = [ct.POINTER(ct.c_void_p),ct.c_char_p,ct.c_char_p,ct.c_int,ct.c_void_p,ct.c_void_p]
        self.nvrtc.nvrtcCompileProgram.argtypes = [ct.c_void_p,ct.c_int,ct.POINTER(ct.c_char_p)]
        for name in ('nvrtcGetPTXSize','nvrtcGetProgramLogSize'):
            getattr(self.nvrtc,name).argtypes = [ct.c_void_p,ct.POINTER(ct.c_size_t)]
        for name in ('nvrtcGetPTX','nvrtcGetProgramLog'):
            getattr(self.nvrtc,name).argtypes = [ct.c_void_p,ct.c_void_p]
        self.nvrtc.nvrtcDestroyProgram.argtypes = [ct.POINTER(ct.c_void_p)]
        self.cuda.cuModuleLoadData.argtypes = [ct.POINTER(ct.c_void_p),ct.c_void_p]
        self.cuda.cuModuleGetFunction.argtypes = [ct.POINTER(ct.c_void_p),ct.c_void_p,ct.c_char_p]
        self.cuda.cuLaunchKernel.argtypes = [ct.c_void_p,ct.c_uint,ct.c_uint,ct.c_uint,ct.c_uint,ct.c_uint,ct.c_uint,ct.c_uint,ct.c_void_p,ct.POINTER(ct.c_void_p),ct.c_void_p]
        program=ct.c_void_p()
        _check(self.nvrtc.nvrtcCreateProgram(ct.byref(program),_SOURCE.encode(),b'transport_fused.cu',0,None,None),'nvrtcCreateProgram')
        try:
            capability=torch.cuda.get_device_capability()
            options=[b'--std=c++11',('--gpu-architecture=compute_%d%d'%capability).encode(),b'--fmad=false']
            raw=(ct.c_char_p*len(options))(*options)
            result=self.nvrtc.nvrtcCompileProgram(program,len(options),raw)
            if result:
                size=ct.c_size_t()
                self.nvrtc.nvrtcGetProgramLogSize(program,ct.byref(size))
                log=ct.create_string_buffer(size.value)
                self.nvrtc.nvrtcGetProgramLog(program,log)
                raise RuntimeError('NVRTC: '+log.value.decode(errors='replace'))
            size=ct.c_size_t()
            _check(self.nvrtc.nvrtcGetPTXSize(program,ct.byref(size)),'nvrtcGetPTXSize')
            ptx=ct.create_string_buffer(size.value)
            _check(self.nvrtc.nvrtcGetPTX(program,ptx),'nvrtcGetPTX')
        finally:
            self.nvrtc.nvrtcDestroyProgram(ct.byref(program))
        self.module=ct.c_void_p()
        _check(self.cuda.cuModuleLoadData(ct.byref(self.module),ptx),'cuModuleLoadData')
        self.functions={}
        for name in ('dg4','dg6','dg8','dg12','stage2','stage3','stage4'):
            function=ct.c_void_p()
            _check(self.cuda.cuModuleGetFunction(ct.byref(function),self.module,name.encode()),'cuModuleGetFunction')
            self.functions[name]=function

    def launch(self,name,count,args):
        pointer_args=(ct.c_void_p*len(args))(*[ct.cast(ct.byref(v),ct.c_void_p) for v in args])
        stream=ct.c_void_p(torch.cuda.current_stream().cuda_stream)
        _check(self.cuda.cuLaunchKernel(self.functions[name],(int(count)+255)//256,1,1,256,1,1,0,stream,pointer_args,None),'cuLaunchKernel')


def _p(tensor):
    return ct.c_void_p(tensor.data_ptr())


def _strides(tensor):
    return [ct.c_long(tensor.stride(0)),ct.c_long(tensor.stride(1))]


_runtime=None
_disabled=False
status={'enabled':False,'reason':'not_initialized'}


def _get_runtime():
    global _runtime,_disabled
    if _disabled:
        return None
    if _runtime is None:
        try:
            _runtime=_Runtime()
            status.update(enabled=True,reason='compiled')
        except Exception as exc:
            _disabled=True
            status.update(enabled=False,reason=str(exc))
    return _runtime


def _try(name,count,args):
    global _disabled
    runtime=_get_runtime()
    if runtime is None:
        return False
    try:
        runtime.launch(name,count,args)
        return True
    except Exception as exc:
        _disabled=True
        status.update(enabled=False,reason=str(exc))
        return False


def acceleration(dg,values,electric,ngauss):
    """Return None if optional compilation/launch is unavailable."""
    if values.dtype != torch.float64 or electric.dtype != torch.float64:
        return None
    nx,nv=values.shape
    out=torch.empty_strided((nx,nv),(1,nx),dtype=values.dtype,device=values.device)
    args=[_p(v) for v in (values,electric,dg.derivative,dg.left,dg.right,dg.lift_left,dg.lift_right,dg.inverse_jacobian,out)]
    args += [ct.c_int(nx),ct.c_int(nv//ngauss),*_strides(values),ct.c_long(electric.stride(0))]
    return out if _try('dg%d'%ngauss,nx*(nv//ngauss),args) else None


def stage2(ph,f,q,k):
    out=torch.empty_like(ph)
    args=[_p(v) for v in (ph,f,q,k,out)]
    args += [ct.c_int(f.shape[1]),ct.c_long(f.numel()),*_strides(f),*_strides(k)]
    if _try('stage2',f.numel(),args):
        return out
    return ph*f+q*k


def stage3(ph,a,q,k3,k1):
    out=torch.empty_like(ph)
    args=[_p(v) for v in (ph,a,q,k3,k1,out)]
    args += [ct.c_int(a.shape[1]),ct.c_long(a.numel()),*_strides(a),*_strides(k3),*_strides(k1)]
    if _try('stage3',a.numel(),args):
        return out
    return ph*a+q*(2*k3-k1)


def stage4(full,f,f1,k1,f2,k2,k3,f3,k4):
    out=torch.empty_like(full)
    args=[_p(v) for v in (full,f,f1,k1,f2,k2,k3,f3,k4,out)]
    args += [ct.c_int(f.shape[1]),ct.c_long(f.numel())]
    for v in (f,k1,k2,k3,k4):
        args += _strides(v)
    if _try('stage4',f.numel(),args):
        return out
    return full*f+f1*k1+2*f2*(k2+k3)+f3*k4
