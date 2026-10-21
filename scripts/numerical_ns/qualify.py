"""Independent smooth manufactured NS correctness/convergence checks, CPU only."""
import argparse,json,os,time
from pathlib import Path
import numpy as np

class Analytic:
    def __init__(self,trace=False):
        self.trace=trace
        self.metadata={'domain':{'bounds':{'x':[0.,1.],'y':[0.,1.]},'time_interval':[0.,.1],
            'geometry':{'domain_bounds':[0.,1.,0.,1.],'obstacle_polygons':[]}},
            'parameters':{'viscosity':.01},'boundary_condition':{'kind':'manufactured_velocity_trace' if trace else 'no_slip_velocity'}}
    def g(self,x):
        k=np.pi
        if self.trace:
            v=k*(x+.137)
            return np.sin(v),k*np.cos(v),-k*k*np.sin(v),-k**3*np.cos(v)
        return np.sin(k*x)**2,k*np.sin(2*k*x),2*k*k*np.cos(2*k*x),-4*k**3*np.sin(2*k*x)
    def values(self,t,x,y):
        a=.03*np.exp(-t)
        gx,dx,dxx,dxxx=self.g(x);gy,dy,dyy,dyyy=self.g(y)
        u=a*gx*dy;v=-a*dx*gy
        ux=a*dx*dy;uy=a*gx*dyy;vx=-a*dxx*gy;vy=-ux
        fx=-u+u*ux+v*uy-.01*a*(dxx*dy+gx*dyyy)
        fy=-v+u*vx+v*vy+.01*a*(dxxx*gy+dx*dyy)
        return np.column_stack((u,v)),np.column_stack((fx,fy))
    def query(self,op,points):
        if op=='I':t=np.zeros(len(points));x,y=points.T
        else:t,x,y=points.T
        uv,force=self.values(t,x,y)
        return force if op=='F' else uv

def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['mac','fem','nektar'],required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();config=json.loads((root/'control/runtime.json').read_text())
    from importlib import import_module
    solve=import_module(a.method).solve
    results=[]
    for trace in ([False,True] if a.method!='nektar' else [False]):
        errors=[]
        for level in (0,1):
            profile={**config,'method':a.method,'resolution':8*2**level,'steps':20*2**level,
                     'modes':4+level,'quadrature_extra':2,'force_quadrature_factor':2,'quadrature_degree':6}
            out=root/'qualification'/f'{a.method}-{trace}-{level}';out.mkdir(parents=True,exist_ok=True)
            os.chdir(out)
            problem=Analytic(trace)
            start=time.monotonic();sol=solve(problem,profile)
            rng=np.random.default_rng(1729)
            points=np.column_stack((rng.uniform(.01,.1,200),rng.uniform(.001,.999,(200,2))))
            predicted=sol.evaluate(points);truth=problem.values(*points.T)[0]
            error=float(np.linalg.norm(predicted-truth)/np.linalg.norm(truth))
            assert np.isfinite(predicted).all()
            row={'method':a.method,'trace':trace,'level':level,'relative_l2':error,
                 'wall_seconds':time.monotonic()-start,'diagnostics':sol.diagnostics}
            print(json.dumps(row),flush=True);results.append(row);errors.append(error);sol.close()
        if not errors[1]<errors[0] or errors[1]>.10:
            raise AssertionError(f'{a.method}, dynamic trace={trace}: convergence failed {errors}')
    (root/'qualification'/f'{a.method}.json').write_text(json.dumps({'status':'passed','results':results},indent=2)+'\n')
if __name__=='__main__':main()
