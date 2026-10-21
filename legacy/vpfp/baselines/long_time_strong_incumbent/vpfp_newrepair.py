"""Safe compact CNN inference for source-knot and continuous-time repairs."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch import nn
from vpfp_hierarchy import build_features

class RepairNet(nn.Module):
    def __init__(self,cin,width,depth,dilated=True):
        super().__init__();self.stem=nn.Conv2d(cin,width,1);self.blocks=nn.ModuleList();pattern=(1,2,3,1,2,4,1,3)
        for j in range(depth):
            d=pattern[j%len(pattern)] if dilated else 1
            self.blocks.append(nn.Sequential(nn.GroupNorm(min(8,width),width),nn.GELU(),nn.Conv2d(width,width,3,padding=d,dilation=d,padding_mode='circular'),nn.GroupNorm(min(8,width),width),nn.GELU(),nn.Conv2d(width,width,3,padding=d,dilation=d,padding_mode='circular')))
        self.head=nn.Sequential(nn.GroupNorm(min(8,width),width),nn.GELU(),nn.Conv2d(width,4,1))
    def forward(self,x):
        x=self.stem(x)
        for block in self.blocks:x=x+.2*block(x)
        return self.head(x)
_CACHE={}
def load_model(filename):
    if filename in _CACHE:return _CACHE[filename]
    torch.set_num_threads(1);path=Path(__file__).resolve().parent/filename
    with np.load(path,allow_pickle=False) as z:
        ind=np.asarray(z['indices'],int);net=RepairNet(len(ind),int(z['width'][0]),int(z['depth'][0]),bool(z['dilated'][0]));marker='state__';state={k[len(marker):]:torch.from_numpy(np.asarray(z[k])) for k in z.files if k.startswith(marker)};net.load_state_dict(state);net.eval();value=(net,ind,np.asarray(z['x_mean'],np.float32),np.asarray(z['x_std'],np.float32),np.asarray(z['y_scale'],np.float32))
    _CACHE[filename]=value;return value

def _augment_phase(case,times,features):
    st=np.asarray(case['arrays']['t'],float);q=np.asarray(times,float)
    if len(st)<2:
        phase=np.zeros(len(q))
    else:
        j=np.searchsorted(st,q,side='right')-1;j=np.clip(j,0,len(st)-2);phase=(q-st[j])/np.maximum(st[j+1]-st[j],1e-14);phase=np.clip(phase,0,1);phase[np.isclose(phase,1,rtol=0,atol=2e-12)]=0
    shape=16*phase**2*(1-phase)**2
    extra=np.stack((shape,shape*(2*phase-1),np.sin(2*np.pi*phase),shape*(2*phase-1)**2),axis=1).astype(np.float32)
    extra=np.broadcast_to(extra[:,:,None,None],(len(q),4,features.shape[2],features.shape[3]))
    return np.concatenate((features,extra),axis=1)

def correction(case,times,hierarchy_result,filename):
    feat=build_features(case,times,hierarchy_result).astype(np.float32);net,ind,mean,std,scale=load_model(filename)
    if int(np.max(ind))>=feat.shape[1]:feat=_augment_phase(case,times,feat)
    x=(feat[:,ind]-mean[None,:,None,None])/std[None,:,None,None];values=[]
    with torch.no_grad():
        for first in range(0,len(x),16):values.append(net(torch.from_numpy(np.clip(x[first:first+16],-12,12))).numpy())
    value=np.concatenate(values)*scale[None,:,None,None];value[:,0]-=value[:,0].mean(axis=(-2,-1),keepdims=True);at0=np.isclose(np.asarray(times,float),float(case['time'].t0),rtol=0,atol=1e-13);value[at0]=0
    return value
