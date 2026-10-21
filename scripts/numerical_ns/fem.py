"""Taylor--Hood P2/P1 BDF2 incremental pressure correction.

Variational splitting follows Jorgen S. Dokken's MIT-licensed dolfinx_ipcs
(2021); see DOKKEN_LICENSE. Adaptations: public quadrature forcing, arbitrary
polygonal holes, backward-Euler startup, time-dependent velocity boundaries,
and a numerical-state-only, post-solve point evaluator.
"""
import time
import numpy as np


class Solution:
    def close(self):
        pass

    def evaluate(self, points):
        from dolfinx import geometry
        xyz=np.column_stack((points[:,1:],np.zeros(len(points))))
        tree=geometry.bb_tree(self.mesh,self.mesh.topology.dim)
        candidates=geometry.compute_collisions_points(tree,xyz)
        hits=geometry.compute_colliding_cells(self.mesh,candidates,xyz)
        cells=np.array([hits.links(i)[0] if len(hits.links(i)) else -1 for i in range(len(points))],dtype=np.int32)
        if (cells<0).any():
            raise ValueError('query outside the physical finite-element mesh')
        j=np.clip(np.searchsorted(self.times,points[:,0])-1,0,len(self.times)-2)
        weight=np.clip((points[:,0]-self.times[j])/(self.times[j+1]-self.times[j]),0,1)
        out=np.zeros((len(points),2))
        for offset in (0,1):
            for k in np.unique(j+offset):
                ids=np.flatnonzero(j+offset==k)
                self.field.x.array[:]=self.states[k]
                values=self.field.eval(xyz[ids],cells[ids]).reshape(-1,2)
                w=weight[ids] if offset else 1-weight[ids]
                out[ids]+=w[:,None]*values
        return out


def solve(inputs, profile):
    begin=time.monotonic()
    import basix
    import basix.ufl
    import dolfinx
    from dolfinx import fem, mesh as dmesh
    from dolfinx.io import gmshio
    import dolfinx.fem.petsc as fp
    import gmsh
    from mpi4py import MPI
    from petsc4py import PETSc
    import ufl
    meta=inputs.metadata
    bounds=meta['domain']['bounds'];x0,x1=bounds['x'];y0,y1=bounds['y']
    t0,end=meta['domain']['time_interval'];steps=int(profile['steps']);dt=(end-t0)/steps
    nu=float(meta['parameters']['viscosity'])
    def public(op,points):
        points=np.asarray(points,dtype=float).copy()
        lower=np.array([x0,y0] if op=='I' else [t0,x0,y0])
        upper=np.array([x1,y1] if op=='I' else [end,x1,y1])
        tolerance=64*np.finfo(float).eps*max(1.,float(np.max(abs(np.r_[lower,upper]))))
        if (points<lower-tolerance).any() or (points>upper+tolerance).any():
            raise ValueError('FEM coordinates genuinely leave the public domain')
        return inputs.query(op,np.clip(points,lower,upper))

    h=max(x1-x0,y1-y0)/int(profile['resolution'])
    polygons=[[[x0,y0],[x1,y0],[x1,y1],[x0,y1]],*meta['domain']['geometry'].get('obstacle_polygons',[])]
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal',0);gmsh.option.setNumber('General.NumThreads',1)
        gmsh.model.add('physical_ns')
        loops=[];edges=[]
        for index,poly in enumerate(polygons):
            vertices=[gmsh.model.geo.addPoint(float(x),float(y),0,h*(.65 if index else 1.)) for x,y in poly]
            lines=[gmsh.model.geo.addLine(a,b) for a,b in zip(vertices,vertices[1:]+vertices[:1])]
            loops.append(gmsh.model.geo.addCurveLoop(lines));edges+=lines
        surface=gmsh.model.geo.addPlaneSurface(loops)
        gmsh.model.geo.synchronize()
        gmsh.model.addPhysicalGroup(2,[surface],1);gmsh.model.addPhysicalGroup(1,edges,2)
        gmsh.model.mesh.generate(2)
        msh,_,_=gmshio.model_to_mesh(gmsh.model,MPI.COMM_SELF,0,gdim=2)
    finally:
        gmsh.finalize()
    V=fem.functionspace(msh,('Lagrange',2,(2,)))
    Q=fem.functionspace(msh,('Lagrange',1))
    un=fem.Function(V);old=fem.Function(V);adv=fem.Function(V);star=fem.Function(V)
    uD=fem.Function(V);p=fem.Function(Q);phi=fem.Function(Q)
    xyz=V.tabulate_dof_coordinates();xy=xyz[:,:2]
    un.x.array[:]=public('I',xy).ravel();old.x.array[:]=un.x.array
    msh.topology.create_connectivity(1,2)
    facets=dmesh.exterior_facet_indices(msh.topology)
    bdofs=fem.locate_dofs_topological(V,1,facets)
    boundary_xy=xy[bdofs]
    vbcs=[fem.dirichletbc(uD,bdofs)]
    pbcs=[fem.dirichletbc(PETSc.ScalarType(0),np.array([0],dtype=np.int32),Q)]
    qdegree=int(profile.get('quadrature_degree',6))
    qp,_=basix.make_quadrature(basix.CellType.triangle,qdegree)
    W=fem.functionspace(msh,basix.ufl.quadrature_element(msh.basix_cell(),value_shape=(2,),degree=qdegree))
    force=fem.Function(W)
    cells=np.arange(msh.topology.index_map(2).size_local,dtype=np.int32)
    jit={'cache_dir':profile['jit_cache'],'cffi_extra_compile_args':['-O2']}
    expr=fem.Expression(ufl.SpatialCoordinate(msh),qp,jit_options=jit)
    qxy=expr.eval(msh,cells).reshape(-1,2)
    # Map quadrature data through dofmaps rather than assuming global cell order.
    qdofs=np.array([W.dofmap.cell_dofs(int(c)) for c in cells],dtype=np.int32)
    u,v=ufl.TrialFunction(V),ufl.TestFunction(V)
    r,q=ufl.TrialFunction(Q),ufl.TestFunction(Q)
    dx=ufl.Measure('dx',domain=msh,metadata={'quadrature_degree':qdegree})
    alpha=fem.Constant(msh,PETSc.ScalarType(1./dt))
    b1=fem.Constant(msh,PETSc.ScalarType(1./dt));b2=fem.Constant(msh,PETSc.ScalarType(0.))
    a1=(alpha*ufl.inner(u,v)+nu*ufl.inner(ufl.grad(u),ufl.grad(v))
        +ufl.inner(ufl.grad(u)*adv,v)+.5*ufl.div(adv)*ufl.inner(u,v))*dx
    L1=(ufl.inner(b1*un+b2*old,v)+p*ufl.div(v)+ufl.inner(force,v))*dx
    a2=ufl.inner(ufl.grad(r),ufl.grad(q))*dx
    L2=-alpha*ufl.div(star)*q*dx
    a3=ufl.inner(u,v)*dx
    L3=(ufl.inner(star,v)-ufl.inner(ufl.grad(phi),v)/alpha)*dx
    forms=[fem.form(f,jit_options=jit) for f in (a1,L1,a2,L2,a3,L3)]
    a1,L1,a2,L2,a3,L3=forms
    A1=fp.create_matrix(a1);A2=fp.assemble_matrix(a2,bcs=pbcs);A2.assemble()
    A3=fp.assemble_matrix(a3,bcs=vbcs);A3.assemble()
    def ksp(A):
        k=PETSc.KSP().create(msh.comm);k.setType('preonly');k.getPC().setType('lu')
        k.getPC().setFactorSolverType('mumps');k.setTolerances(rtol=1e-9);k.setOperators(A)
        return k
    k1,k2,k3=ksp(A1),ksp(A2),ksp(A3)
    bs=[fp.create_vector(L) for L in (L1,L2,L3)]
    def linear(a,L,A,b,k,target,bcs):
        with b.localForm() as local:local.set(0.)
        fp.assemble_vector(b,L);fp.apply_lifting(b,[a],[bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,mode=PETSc.ScatterMode.REVERSE)
        fp.set_bc(b,bcs);k.solve(b,target.x.petsc_vec);target.x.scatter_forward()
        if k.getConvergedReason()<0:raise RuntimeError('FEM linear solve failed')
    result=Solution();result.mesh=msh;result.field=fem.Function(V)
    result.times=np.linspace(t0,end,steps+1);result.states=[un.x.array.copy()]
    for i,t in enumerate(result.times[1:],1):
        bc=public('B',np.column_stack((np.full(len(boundary_xy),t),boundary_xy)))
        uD.x.array.reshape(-1,2)[bdofs]=bc
        values=public('F',np.column_stack((np.full(len(qxy),t),qxy)))
        force.x.array.reshape(-1,2)[qdofs.ravel()]=values
        if i==1:
            adv.x.array[:]=un.x.array
        else:
            alpha.value=1.5/dt;b1.value=2./dt;b2.value=-.5/dt
            adv.x.array[:]=2*un.x.array-old.x.array
        A1.zeroEntries();fp.assemble_matrix(A1,a1,bcs=vbcs);A1.assemble()
        linear(a1,L1,A1,bs[0],k1,star,vbcs)
        linear(a2,L2,A2,bs[1],k2,phi,pbcs)
        old.x.array[:]=un.x.array
        linear(a3,L3,A3,bs[2],k3,un,vbcs)
        p.x.array[:]+=phi.x.array
        if not np.isfinite(un.x.array).all():raise FloatingPointError('nonfinite FEM solution')
        result.states.append(un.x.array.copy())
    result.diagnostics={'method':'Taylor-Hood-P2-P1-BDF2-IPCS','dolfinx':dolfinx.__version__,
        'cells':len(cells),'velocity_dofs':len(un.x.array),'steps':steps,'quadrature_degree':qdegree,
        'wall_seconds':time.monotonic()-begin,'hidden_points_seen_during_solve':False}
    for k in (k1,k2,k3):k.destroy()
    for a in (A1,A2,A3):a.destroy()
    for b in bs:b.destroy()
    return result
