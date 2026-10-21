"""Small NVRTC-compiled float64 DG acceleration kernel; optional research path."""
import ctypes as C
import ctypes.util
import time

import torch


SOURCE = r'''
extern "C" __global__ void dg_acceleration(
    const double* values, const double* electric, double* output,
    const double* derivative, const double* left, const double* right,
    const double* lift_left, const double* lift_right,
    const double* inverse_jacobian, int ncells, int nx,
    int value_stride0, int value_stride1, int electric_stride) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int n=nx*ncells*6;
  if(i>=n) return;
  int g=i%6;
  int cell=(i/6)%ncells;
  int x=i/(ncells*6);
  int start=x*value_stride0+cell*6*value_stride1;
  double field=electric[x*electric_stride];
  double own=0., deriv=0., neighbor=0.;
  bool positive=field>=0.;
  const double* own_endpoint=positive?left:right;
  const double* other_endpoint=positive?right:left;
  bool present=positive?(cell>0):(cell+1<ncells);
  int offset=(positive?-6:6)*value_stride1;
  #pragma unroll
  for(int j=0;j<6;j++) {
    double value=values[start+j*value_stride1];
    deriv += derivative[g*6+j]*value;
    own += own_endpoint[j]*value;
    if(present) neighbor += other_endpoint[j]*values[start+offset+j*value_stride1];
  }
  double correction=positive?(own-neighbor)*lift_left[g]:(neighbor-own)*lift_right[g];
  output[i]=-field*(deriv+correction)*inverse_jacobian[cell];
}

struct Cx { double r; double i; };
__device__ Cx add(Cx a,Cx b) { return {a.r+b.r,a.i+b.i}; }
__device__ Cx sub(Cx a,Cx b) { return {a.r-b.r,a.i-b.i}; }
__device__ Cx scale(Cx a,double s) { return {s*a.r,s*a.i}; }
__device__ Cx mul(Cx a,Cx b) { return {a.r*b.r-a.i*b.i,a.r*b.i+a.i*b.r}; }
extern "C" __global__ void stage_linear(
    const Cx* ph,const Cx* f,const Cx* q,const Cx* k,Cx* out,
    int nx,int nv,int fs0,int fs1,int ks0,int ks1) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=nx*nv)return;
  int x=i/nv,v=i%nv,fi=x*fs0+v*fs1,ki=x*ks0+v*ks1;
  out[i]=add(mul(ph[i],f[fi]),mul(q[i],k[ki]));
}
extern "C" __global__ void stage_third(
    const Cx* ph,const Cx* a,const Cx* q,const Cx* k3,const Cx* k1,Cx* out,
    int nx,int nv,int fs0,int fs1,int ks0,int ks1) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=nx*nv)return;
  int x=i/nv,v=i%nv,fi=x*fs0+v*fs1,ki=x*ks0+v*ks1;
  out[i]=add(mul(ph[i],a[fi]),mul(q[i],sub(scale(k3[ki],2.),k1[ki])));
}
extern "C" __global__ void stage_finish(
    const Cx* full,const Cx* f,const Cx* f1,const Cx* k1,const Cx* f2,
    const Cx* k2,const Cx* k3,const Cx* f3,const Cx* k4,Cx* out,
    int nx,int nv,int fs0,int fs1,int ks0,int ks1) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=nx*nv)return;
  int x=i/nv,v=i%nv,fi=x*fs0+v*fs1,ki=x*ks0+v*ks1;
  out[i]=add(add(add(mul(full[i],f[fi]),mul(f1[i],k1[ki])),
                  mul(scale(f2[i],2.),add(k2[ki],k3[ki]))),mul(f3[i],k4[ki]));
}
'''


class FusedAcceleration:
    def __init__(self, dg, nx):
        began=time.perf_counter()
        self.nx=int(nx)
        self.ncells=len(dg.edges)-1
        if self.nx<1 or self.ncells<1 or dg.derivative.shape!=(6,6):
            raise ValueError('FusedAcceleration supports Gauss6 only')
        self.device=dg.derivative.device
        if self.device!=torch.device('cuda:0'):
            raise ValueError('FusedAcceleration requires the assigned cuda:0')
        geometry=(dg.derivative,dg.left,dg.right,dg.lift_left,dg.lift_right,dg.inverse_jacobian)
        if any(t.device!=self.device or t.dtype!=torch.float64 or not t.is_contiguous()
               or t.is_conj() or t.is_neg() for t in geometry):
            raise ValueError('FusedAcceleration requires contiguous float64 CUDA geometry')
        if (any(t.shape!=(6,) for t in geometry[1:5])
                or dg.inverse_jacobian.shape!=(1,self.ncells,1)):
            raise ValueError('FusedAcceleration geometry has incompatible shapes')
        try:
            self.nvrtc=C.CDLL(C.util.find_library('nvrtc') or 'libnvrtc.so.12')
        except OSError:
            compiler_library=getattr(C.util,'_findLib_gcc',lambda name:None)('nvrtc')
            if compiler_library is None:raise
            self.nvrtc=C.CDLL(compiler_library)
        self.cuda=C.CDLL(C.util.find_library('cuda') or 'libcuda.so.1')
        self.nvrtc.nvrtcCreateProgram.argtypes=[C.POINTER(C.c_void_p),C.c_char_p,C.c_char_p,C.c_int,C.c_void_p,C.c_void_p]
        self.nvrtc.nvrtcCompileProgram.argtypes=[C.c_void_p,C.c_int,C.POINTER(C.c_char_p)]
        self.nvrtc.nvrtcGetPTXSize.argtypes=[C.c_void_p,C.POINTER(C.c_size_t)]
        self.nvrtc.nvrtcGetPTX.argtypes=[C.c_void_p,C.c_void_p]
        self.nvrtc.nvrtcGetProgramLogSize.argtypes=[C.c_void_p,C.POINTER(C.c_size_t)]
        self.nvrtc.nvrtcGetProgramLog.argtypes=[C.c_void_p,C.c_void_p]
        self.nvrtc.nvrtcDestroyProgram.argtypes=[C.POINTER(C.c_void_p)]
        self.cuda.cuModuleLoadData.argtypes=[C.POINTER(C.c_void_p),C.c_void_p]
        self.cuda.cuModuleGetFunction.argtypes=[C.POINTER(C.c_void_p),C.c_void_p,C.c_char_p]
        self.cuda.cuLaunchKernel.argtypes=[C.c_void_p,C.c_uint,C.c_uint,C.c_uint,
                                         C.c_uint,C.c_uint,C.c_uint,C.c_uint,
                                         C.c_void_p,C.POINTER(C.c_void_p),C.c_void_p]
        program=C.c_void_p()
        self.check(self.nvrtc.nvrtcCreateProgram(C.byref(program),SOURCE.encode(),b'dg.cu',0,None,None),'nvrtcCreateProgram')
        try:
            major,minor=torch.cuda.get_device_capability()
            options=(C.c_char_p*2)(f'--gpu-architecture=compute_{major}{minor}'.encode(),b'--fmad=false')
            status=self.nvrtc.nvrtcCompileProgram(program,len(options),options)
            if status:
                size=C.c_size_t();self.nvrtc.nvrtcGetProgramLogSize(program,C.byref(size))
                log=C.create_string_buffer(size.value);self.nvrtc.nvrtcGetProgramLog(program,log)
                raise RuntimeError(f'NVRTC compilation failed ({status}): {log.value.decode()}')
            size=C.c_size_t();self.check(self.nvrtc.nvrtcGetPTXSize(program,C.byref(size)),'nvrtcGetPTXSize')
            ptx=C.create_string_buffer(size.value);self.check(self.nvrtc.nvrtcGetPTX(program,ptx),'nvrtcGetPTX')
        finally:
            self.nvrtc.nvrtcDestroyProgram(C.byref(program))
        # PyTorch owns the assigned primary CUDA context and current stream.
        torch.cuda.current_stream()
        self.module=C.c_void_p()
        self.check(self.cuda.cuModuleLoadData(C.byref(self.module),ptx),'cuModuleLoadData')
        self.function=C.c_void_p()
        self.check(self.cuda.cuModuleGetFunction(C.byref(self.function),self.module,b'dg_acceleration'),'cuModuleGetFunction')
        self.tensors=(dg.derivative,dg.left,dg.right,dg.lift_left,dg.lift_right,dg.inverse_jacobian)
        self.arguments=[C.c_void_p(0),C.c_void_p(0),C.c_void_p(0),
                        *[C.c_void_p(t.data_ptr()) for t in self.tensors],C.c_int(self.ncells),C.c_int(self.nx),
                        C.c_int(0),C.c_int(0),C.c_int(0)]
        self.parameters=(C.c_void_p*len(self.arguments))(*[C.cast(C.pointer(v),C.c_void_p) for v in self.arguments])
        self.block=256;self.grid=(self.nx*self.ncells*6+self.block-1)//self.block
        self.startup_seconds=time.perf_counter()-began

    @staticmethod
    def check(status, operation):
        if status:raise RuntimeError(f'{operation} returned CUDA status{status}')

    def __call__(self, values, electric):
        if (values.dtype!=torch.float64 or electric.dtype!=torch.float64
                or values.device!=self.device or electric.device!=self.device
                or values.is_conj() or values.is_neg() or electric.is_conj() or electric.is_neg()
                or values.shape!=(self.nx,self.ncells*6) or electric.shape!=(self.nx,)):
            raise ValueError('FusedAcceleration requires the configured float64 CUDA shapes')
        output=torch.empty(values.shape,dtype=values.dtype,device=values.device)
        self.arguments[0].value=values.data_ptr()
        self.arguments[1].value=electric.data_ptr()
        self.arguments[2].value=output.data_ptr()
        self.arguments[-3].value=values.stride(0)
        self.arguments[-2].value=values.stride(1)
        self.arguments[-1].value=electric.stride(0)
        stream=torch.cuda.current_stream().cuda_stream
        self.check(self.cuda.cuLaunchKernel(self.function,self.grid,1,1,self.block,1,1,0,
                                           C.c_void_p(stream),self.parameters,None),'cuLaunchKernel')
        return output


class FusedStages:
    """Optional complex128 ETDRK4 pointwise stages using the same CUDA module."""
    def __init__(self, acceleration):
        self.acceleration=acceleration
        self.cuda=acceleration.cuda
        self.nx,self.nv=acceleration.nx,acceleration.ncells*6
        self.plans={}
        for name,pointer_count in [('stage_linear',5),('stage_third',6),('stage_finish',10)]:
            fn=C.c_void_p()
            acceleration.check(self.cuda.cuModuleGetFunction(C.byref(fn),acceleration.module,name.encode()),'cuModuleGetFunction')
            arguments=[*[C.c_void_p(0) for _ in range(pointer_count)],
                       C.c_int(self.nx),C.c_int(self.nv),C.c_int(0),C.c_int(0),C.c_int(0),C.c_int(0)]
            params=(C.c_void_p*len(arguments))(*[C.cast(C.pointer(a),C.c_void_p) for a in arguments])
            self.plans[name]=(fn,arguments,params)

    def launch(self,name,inputs,state,rhs):
        if any(t.dtype!=torch.complex128 or t.device!=self.acceleration.device or t.is_conj() or t.is_neg()
               or t.shape!=(self.nx,self.nv) for t in inputs):
            raise ValueError('FusedStages requires the configured complex128 CUDA shapes')
        output=torch.empty((self.nx,self.nv),dtype=torch.complex128,device=state.device)
        fn,args,params=self.plans[name]
        for slot,tensor in zip(args,(*inputs,output)):slot.value=tensor.data_ptr()
        for slot,value in zip(args[-4:],(*state.stride(),*rhs.stride())):slot.value=value
        a=self.acceleration
        a.check(self.cuda.cuLaunchKernel(fn,a.grid,1,1,a.block,1,1,0,
                                         C.c_void_p(torch.cuda.current_stream().cuda_stream),params,None),'cuLaunchKernel')
        return output

    def linear(self,ph,f,q,k):
        if not ph.is_contiguous() or not q.is_contiguous():return ph*f+q*k
        return self.launch('stage_linear',(ph,f,q,k),f,k)

    def third(self,ph,a,q,k3,k1):
        if not ph.is_contiguous() or not q.is_contiguous() or k1.stride()!=k3.stride():
            return ph*a+q*(2*k3-k1)
        return self.launch('stage_third',(ph,a,q,k3,k1),a,k3)

    def finish(self,full,f,f1,k1,f2,k2,k3,f3,k4):
        if (not all(c.is_contiguous() for c in (full,f1,f2,f3))
                or not all(k.stride()==k1.stride() for k in (k2,k3,k4))):
            return full*f+f1*k1+2*f2*(k2+k3)+f3*k4
        return self.launch('stage_finish',(full,f,f1,k1,f2,k2,k3,f3,k4),f,k1)


def prepare_acceleration(dg,nx):
    """An unavailable compiler/runtime retains the original PyTorch path."""
    try:
        accelerator=FusedAcceleration(dg,nx)
    except (OSError,RuntimeError,ValueError,AttributeError) as error:
        return None,dict(enabled=False,reason=str(error))
    return accelerator,dict(enabled=True,startup_seconds=accelerator.startup_seconds,
                            method='nvrtc_float64_gauss6')


def prepare_stages(accelerator):
    if accelerator is None:return None,dict(enabled=False,reason='acceleration_unavailable')
    try:stages=FusedStages(accelerator)
    except (OSError,RuntimeError,ValueError,AttributeError) as error:
        return None,dict(enabled=False,reason=str(error))
    return stages,dict(enabled=True,method='nvrtc_complex128_etdrk4')
