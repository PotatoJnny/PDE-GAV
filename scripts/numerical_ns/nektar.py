"""Nektar++ IMEX2 public-input solve with post-solve coefficient evaluation."""
import os
from pathlib import Path
import shutil
import socket
import struct
import subprocess
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from types import SimpleNamespace
import numpy as np
from nektar_session import gmsh_geometry,write_session

def serial_mpi_environment():
    """Keep single-rank native tools off the cluster's restricted RDMA fabric."""
    env=os.environ.copy()
    # Every Case is an independent singleton. MPI initialization otherwise
    # probes mlx5/UD even for ns_initial_field and ns_evaluate_fields, where
    # a denied QP transition can stall before any numerical work is performed.
    env.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,tcp',
               OMPI_MCA_osc='pt2pt',OMPI_MCA_coll='^hcoll,ucc',
               UCX_TLS='self,tcp')
    return env

class Server:
    def __init__(self,inputs):
        self.inputs=inputs
        self.temp=tempfile.TemporaryDirectory(prefix='ns-input-',dir='/tmp')
        self.path=self.temp.name+'/i.sock'
        self.sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
        self.sock.bind(self.path);self.sock.listen(2);self.sock.settimeout(.2)
        self.stop=threading.Event();self.error=None;self.connection=None
        self.thread=threading.Thread(target=self.serve,daemon=True)
    def read(self,c,n):
        out=bytearray()
        while len(out)<n:
            if self.stop.is_set():raise EOFError('stopped')
            try:b=c.recv(n-len(out))
            except socket.timeout:continue
            if not b:raise EOFError('disconnected')
            out.extend(b)
        return bytes(out)
    def serve(self):
        try:
            while not self.stop.is_set():
                try:c,_=self.sock.accept()
                except socket.timeout:continue
                self.connection=c
                with c:
                    c.settimeout(.2)
                    if self.read(c,8)!=b'NSREF01\n':raise ValueError('bad bridge protocol')
                    n=struct.unpack('<Q',self.read(c,8))[0]
                    if not 0<n<5_000_000:raise ValueError('bad point count')
                    xy=np.frombuffer(self.read(c,16*n),dtype='<f8').reshape(-1,2).copy()
                    bounds=self.inputs.metadata['domain']['bounds']
                    lo=np.array([bounds['x'][0],bounds['y'][0]]);hi=np.array([bounds['x'][1],bounds['y'][1]])
                    if (xy<lo-1e-12).any() or (xy>hi+1e-12).any():raise ValueError('mesh outside public domain')
                    xy=np.clip(xy,lo,hi);c.sendall(b'K')
                    while not self.stop.is_set():
                        try:op=self.read(c,1).decode()
                        except EOFError:break
                        t=struct.unpack('<d',self.read(c,8))[0]
                        t0,t1=self.inputs.metadata['domain']['time_interval']
                        if not t0-1e-12<=t<=t1+1e-12:raise ValueError('input time outside interval')
                        if op not in ('I','F'):raise ValueError('nonpublic operation')
                        pts=xy if op=='I' else np.column_stack((np.full(n,np.clip(t,t0,t1)),xy))
                        values=self.inputs.query(op,pts)
                        c.settimeout(None);c.sendall(b'K'+np.asarray(values,dtype='<f8').tobytes());c.settimeout(.2)
                self.connection=None
        except BaseException as e:
            if not self.stop.is_set():self.error=e
    def __enter__(self):self.thread.start();return self
    def __exit__(self,*args):
        self.stop.set()
        if self.connection:
            try:self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:pass
        self.sock.close();self.thread.join(2);self.temp.cleanup()

def run(argv,directory,env,name,server=None):
    with (directory/(name+'.log')).open('wb') as log:
        p=subprocess.Popen(argv,cwd=directory,env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            while p.poll() is None:
                if server and server.error:raise RuntimeError('public input bridge failed') from server.error
                try:p.wait(timeout=.2)
                except subprocess.TimeoutExpired:pass
            if p.returncode:raise RuntimeError(name+' failed: '+(directory/(name+'.log')).read_text()[-5000:])
        finally:
            if p.poll() is None:
                p.terminate()
                try:p.wait(timeout=1)
                except subprocess.TimeoutExpired:p.kill();p.wait()

class Solution:
    def close(self):
        if not os.environ.get('NS_NUMERICAL_KEEP_STATES'):shutil.rmtree(self.directory)
    def evaluate(self,points):
        t0,t1=self.interval;n=self.steps
        pos=np.clip((points[:,0]-t0)/(t1-t0)*n,0,n)
        low=np.minimum(np.floor(pos).astype(int),n-1);weight=pos-low
        requests=[]
        for offset in (0,1):
            for i,(_,x,y) in enumerate(points):
                k=int(low[i]+offset)
                name='initial.fld' if k==0 else self.snapshots[k]
                requests.append(f'{offset*len(points)+i} {name} {x:.17g} {y:.17g}')
        (self.directory/'evaluation_requests.txt').write_text('\n'.join(requests)+'\n')
        run([str(Path(self.profile['native_build'])/'ns_evaluate_fields'),'mesh.xml','session.xml'],self.directory,self.env,'evaluate')
        data=np.loadtxt(self.directory/'evaluation_values.txt',ndmin=2)
        if len(data)!=2*len(points):raise ValueError('incomplete spectral evaluation')
        values=np.empty((len(data),2));values[data[:,0].astype(int)]=data[:,1:]
        return (1-weight[:,None])*values[:len(points)]+weight[:,None]*values[len(points):]

def solve(inputs,profile):
    begin=time.monotonic();meta=inputs.metadata
    if meta['boundary_condition']['kind']!='no_slip_velocity':raise ValueError('Nektar baseline currently supports no-slip boundaries')
    directory=Path(tempfile.mkdtemp(prefix='nektar_states-',dir=Path.cwd())).resolve()
    geometry=meta['domain']['geometry'];bounds=geometry['domain_bounds']
    h=max(bounds[1]-bounds[0],bounds[3]-bounds[2])/profile['resolution']
    (directory/'mesh.geo').write_text(gmsh_geometry(geometry,h))
    env=serial_mpi_environment()
    run([profile['gmsh'],'-2','mesh.geo','-format','msh2','-o','mesh.msh','-v','2'],directory,env,'gmsh')
    nektar=Path(profile['nektar_root'])/'bin';native=Path(profile['native_build'])
    run([str(nektar/'NekMesh'),'mesh.msh','mesh.xml'],directory,env,'mesh')
    t0,t1=meta['domain']['time_interval'];steps=int(profile['steps'])
    config={'modes':profile['modes'],'quadrature_extra':profile.get('quadrature_extra',2),'time_step':(t1-t0)/steps}
    write_session(directory,SimpleNamespace(public_metadata=meta),config,[])
    # All fields at all solver time steps are saved, independently of test points.
    with Server(inputs) as server:
        env['NS_PUBLIC_INPUT_SOCKET']=server.path
        env['NS_REFERENCE_FORCE_QUADRATURE_FACTOR']=str(profile.get('force_quadrature_factor',2))
        run([str(native/'ns_initial_field'),'mesh.xml','session.xml'],directory,env,'initial',server)
        env['LD_PRELOAD']=str(native/'libNSPublicInputForce.so')
        run([str(nektar/'IncNavierStokesSolver'),'mesh.xml','session.xml'],directory,env,'solve',server)
    env.pop('LD_PRELOAD',None);env.pop('NS_PUBLIC_INPUT_SOCKET',None)
    files=list(directory.glob('*_*.chk'))
    snapshots={int(f.stem.rsplit('_',1)[1]):f.name for f in files}
    if not all(k in snapshots for k in range(1,steps+1)):
        raise ValueError('missing numerical coefficient snapshots: '+str(sorted(snapshots)))
    solution=Solution();solution.directory=directory;solution.steps=steps;solution.interval=(t0,t1)
    solution.snapshots=snapshots;solution.profile=profile;solution.env=env
    solution.diagnostics={'method':'Nektar++5.7.0-IMEX2','modes':profile['modes'],'steps':steps,
        'wall_seconds':time.monotonic()-begin,'hidden_points_seen_during_solve':False,
        'cfl_log':(directory/'solve.log').read_text()[-2000:]}
    return solution
