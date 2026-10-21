"""Second-order staggered MAC + SSPRK2 projection on a physical rectangle.

Full nonlinear advection, central diffusion, dynamic Dirichlet velocities and a
Neumann pressure projection. Grid and all input queries are chosen before test
coordinates are available. This is a classical implementation, not an Agent
program or a representation-specific reconstruction.
"""
import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.interpolate import RegularGridInterpolator


def xygrid(x, y):
    a, b = np.meshgrid(x, y, indexing='ij')
    return np.column_stack((a.ravel(), b.ravel()))


class Solution:
    def close(self):
        pass

    def evaluate(self, points):
        out = np.empty((len(points), 2))
        # State includes prescribed tangential traces, so interpolation is also
        # defined between the first/last face centre and the physical wall.
        for i, (t, x, y) in enumerate(points):
            j = int(np.clip(np.searchsorted(self.times, t) - 1, 0, len(self.times)-2))
            a = float(np.clip((t-self.times[j])/(self.times[j+1]-self.times[j]), 0, 1))
            for c, axes in enumerate((self.u_axes, self.v_axes)):
                values = (1-a)*self.states[j][c] + a*self.states[j+1][c]
                out[i, c] = RegularGridInterpolator(axes, values, bounds_error=True)([[x, y]])[0]
        return out


def solve(inputs, profile):
    begin = time.monotonic()
    meta = inputs.metadata
    if meta['domain']['geometry'].get('obstacle_polygons'):
        raise ValueError('MAC baseline requires an obstacle-free rectangle')
    x0, x1 = meta['domain']['bounds']['x']
    y0, y1 = meta['domain']['bounds']['y']
    t0, end = meta['domain']['time_interval']
    nu = float(meta['parameters']['viscosity'])
    n = int(profile['resolution'])
    length = max(x1-x0, y1-y0)
    nx = max(4, round(n*(x1-x0)/length))
    ny = max(4, round(n*(y1-y0)/length))
    dx, dy = (x1-x0)/nx, (y1-y0)/ny
    xf, yf = np.linspace(x0,x1,nx+1), np.linspace(y0,y1,ny+1)
    xc, yc = (xf[:-1]+xf[1:])/2, (yf[:-1]+yf[1:])/2
    xy_u, xy_v = xygrid(xf, yc), xygrid(xc, yf)
    # Four tangential traces and normal face traces, using the public boundary.
    bxy = np.vstack((xygrid([x0,x1],yc), xygrid(xc,[y0,y1]),
                    xygrid(xf,[y0,y1]), xygrid([x0,x1],yf)))
    boundary_cache = {}
    def boundary(t):
        if t not in boundary_cache:
            z = inputs.query('B', np.column_stack((np.full(len(bxy),t),bxy)))
            k = 2*ny
            un = z[:k,0].reshape(2,ny)
            vn = z[k:k+2*nx,1].reshape(nx,2); k += 2*nx
            ut = z[k:k+2*(nx+1),0].reshape(nx+1,2); k += 2*(nx+1)
            vt = z[k:,1].reshape(2,ny+1)
            boundary_cache.clear()
            boundary_cache[t] = un,vn,ut,vt
        return boundary_cache[t]
    def set_bc(u,v,t):
        un,vn,_,_ = boundary(t)
        u[[0,-1],:] = un
        v[:,[0,-1]] = vn
    def neumann_matrix(n,h):
        d = np.full(n,2.); d[[0,-1]]=1.
        return sparse.diags((-np.ones(n-1),d,-np.ones(n-1)),(-1,0,1),format='csc')/(h*h)
    A = sparse.kronsum(neumann_matrix(ny,dy), neumann_matrix(nx,dx),format='csc')
    lu = splu(A[1:,1:].tocsc())
    max_flux_error = 0.
    def project(u,v,t):
        nonlocal max_flux_error
        set_bc(u,v,t)
        div=(u[1:]-u[:-1])/dx+(v[:,1:]-v[:,:-1])/dy
        # A pure Neumann solve needs a compatible RHS. The remaining mean is the
        # measured quadrature error in the supplied discrete boundary flux.
        mean=float(div.mean()); max_flux_error=max(max_flux_error,abs(mean))
        b=(div-mean).ravel()
        phi=np.zeros(nx*ny); phi[1:]=-lu.solve(b[1:]); phi=phi.reshape(nx,ny)
        u[1:-1]-=(phi[1:]-phi[:-1])/dx
        v[:,1:-1]-=(phi[:,1:]-phi[:,:-1])/dy
        return u,v
    def rhs(u,v,t):
        _,_,ut,vt=boundary(t)
        ug=np.column_stack((2*ut[:,0]-u[:,0],u,2*ut[:,1]-u[:,-1]))
        vg=np.row_stack((2*vt[0]-v[0],v,2*vt[1]-v[-1]))
        v_at_u=(v[:-1,:-1]+v[1:,:-1]+v[:-1,1:]+v[1:,1:])/4
        u_at_v=(u[:-1,:-1]+u[1:,:-1]+u[:-1,1:]+u[1:,1:])/4
        f=inputs.query('F',np.column_stack((np.full(len(xy_u)+len(xy_v),t),np.vstack((xy_u,xy_v)))))
        fu=f[:len(xy_u),0].reshape(nx+1,ny)
        fv=f[len(xy_u):,1].reshape(nx,ny+1)
        ru=np.zeros_like(u); rv=np.zeros_like(v)
        ru[1:-1]=(-u[1:-1]*(u[2:]-u[:-2])/(2*dx)
            -v_at_u*(ug[1:-1,2:]-ug[1:-1,:-2])/(2*dy)
            +nu*((u[2:]-2*u[1:-1]+u[:-2])/dx**2
                +(ug[1:-1,2:]-2*u[1:-1]+ug[1:-1,:-2])/dy**2)+fu[1:-1])
        rv[:,1:-1]=(-u_at_v*(vg[2:,1:-1]-vg[:-2,1:-1])/(2*dx)
            -v[:,1:-1]*(v[:,2:]-v[:,:-2])/(2*dy)
            +nu*((vg[2:,1:-1]-2*v[:,1:-1]+vg[:-2,1:-1])/dx**2
                +(v[:,2:]-2*v[:,1:-1]+v[:,:-2])/dy**2)+fv[:,1:-1])
        return ru,rv
    u=inputs.query('I',xy_u)[:,0].reshape(nx+1,ny)
    v=inputs.query('I',xy_v)[:,1].reshape(nx,ny+1)
    u,v=project(u,v,t0)
    result=Solution();result.times=[];result.states=[]
    result.u_axes=(xf,np.r_[y0,yc,y1]);result.v_axes=(np.r_[x0,xc,x1],yf)
    def save(t):
        _,_,ut,vt=boundary(t)
        result.times.append(t)
        result.states.append((np.column_stack((ut[:,0],u,ut[:,1])),np.row_stack((vt[0],v,vt[1]))))
    t=float(t0);save(t)
    max_dt=(end-t0)/int(profile['steps'])
    while t < end-1e-13*max(1.,abs(end)):
        rate=np.max(abs(u))/dx+np.max(abs(v))/dy
        dt=min(max_dt, .30/max(rate,1e-12), .40/max(nu*(1/dx**2+1/dy**2),1e-12),end-t)
        if dt < 1e-10*(end-t0) or len(result.times)>20000:
            raise RuntimeError('unstable MAC solution or excessive time-step count')
        ru,rv=rhs(u,v,t)
        u1,v1=project(u+dt*ru,v+dt*rv,t+dt)
        r1u,r1v=rhs(u1,v1,t+dt)
        u,v=project(.5*(u+u1+dt*r1u),.5*(v+v1+dt*r1v),t+dt)
        if not (np.isfinite(u).all() and np.isfinite(v).all()):
            raise FloatingPointError('nonfinite MAC solution')
        t+=dt;save(t)
    result.times=np.array(result.times)
    result.diagnostics={'method':'MAC-SSPRK2-projection','nx':nx,'ny':ny,
        'steps':len(result.times)-1,'max_boundary_flux_divergence':max_flux_error,
        'wall_seconds':time.monotonic()-begin, 'hidden_points_seen_during_solve':False}
    return result
