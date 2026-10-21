"""Geometry/session construction adapted from the received public NS bridge."""
import math
import xml.etree.ElementTree as ET
def gmsh_geometry(geometry, mesh_size, *, corner_mesh_size=None, corner_distance=None):
    x0, x1, y0, y1 = map(float, geometry['domain_bounds'])
    if not math.isfinite(mesh_size) or mesh_size <= 0: raise ValueError('invalid mesh size')
    polygons = [[[x0,y0],[x1,y0],[x1,y1],[x0,y1]], *geometry['obstacle_polygons']]
    text = ['Mesh.MshFileVersion = 2.2;', 'Mesh.ElementOrder = 1;',
            'Mesh.Algorithm = 6;', 'General.NumThreads = 1;']
    vertex, edge = 1, 1
    for loop, points in enumerate(polygons, 1):
        first = vertex; lines = []
        for x,y in points:
            text.append(f'Point({vertex}) = {{{x:.17g}, {y:.17g}, 0, {mesh_size:.17g}}};')
            vertex += 1
        for i in range(len(points)):
            text.append(f'Line({edge}) = {{{first+i}, {first+(i+1)%len(points)}}};')
            lines.append(str(edge)); edge += 1
        edges = ','.join(lines)
        text.extend([f'Curve Loop({loop}) = {{{edges}}};', f'Physical Curve({loop}) = {{{edges}}};'])
    loops = ','.join(map(str, range(1,len(polygons)+1)))
    text.extend([f'Plane Surface(100) = {{{loops}}};', 'Physical Surface(100) = {100};'])
    if corner_mesh_size is not None:
        hmin, radius = float(corner_mesh_size), float(corner_distance)
        if not 0 < hmin < mesh_size or not mesh_size < radius < 1:
            raise ValueError('invalid corner refinement')
        vertices=','.join(map(str,range(5,vertex)))
        if not vertices: raise ValueError('corner refinement needs obstacle vertices')
        text.extend(['Field[1] = Distance;', f'Field[1].PointsList = {{{vertices}}};',
            'Field[2] = Threshold;', 'Field[2].InField = 1;',
            f'Field[2].SizeMin = {hmin:.17g};', f'Field[2].SizeMax = {mesh_size:.17g};',
            'Field[2].DistMin = 0;', f'Field[2].DistMax = {radius:.17g};',
            'Background Field = 2;', 'Mesh.MeshSizeExtendFromBoundary = 0;',
            'Mesh.MeshSizeFromPoints = 0;', 'Mesh.MeshSizeFromCurvature = 0;'])
    return '\n'.join(text)+'\n'


def write_session(directory, problem, config, probes):
    root = ET.parse(directory/'mesh.xml').getroot()
    comp = root.find('./GEOMETRY/COMPOSITE')
    if comp is None: raise ValueError('NekMesh returned no composites')
    volume, walls = [], []
    for c in comp:
        content = (c.text or '').strip()
        if content.startswith(('T[','Q[')): volume.append(c.attrib['ID'])
        elif content.startswith(('E[','S[')): walls.append(c.attrib['ID'])
        else: raise ValueError('unexpected mesh composite: '+content[:80])
    if not volume or len(walls) != 1+len(problem.public_metadata['domain']['geometry']['obstacle_polygons']):
        raise ValueError('mesh lost fluid or obstacle boundary composites')
    document = ET.Element('NEKTAR')
    expansion = ET.SubElement(document,'EXPANSIONS')
    expansion_attributes = dict(COMPOSITE='C['+','.join(volume)+']',
                                NUMMODES=str(config['modes']), FIELDS='u,v,p', TYPE='MODIFIED')
    if config.get('quadrature_extra', 0):
        if not all((c.text or '').strip().startswith('T[') for c in comp if c.attrib['ID'] in volume):
            raise ValueError('explicit quadrature currently qualified only for triangular elements')
        extra = int(config['quadrature_extra'])
        if not 1 <= extra <= 16: raise ValueError('invalid overintegration count')
        modes = int(config['modes'])
        expansion_attributes = dict(COMPOSITE='C['+','.join(volume)+']', FIELDS='u,v,p',
            NUMMODES=f'{modes},{modes}', BASISTYPE='Modified_A,Modified_B',
            POINTSTYPE='GaussLobattoLegendre,GaussRadauMAlpha1Beta0',
            NUMPOINTS=f'{modes+1+extra},{modes+extra}')
    ET.SubElement(expansion,'E',**expansion_attributes)
    conditions = ET.SubElement(document,'CONDITIONS')
    scheme = ET.SubElement(conditions,'TIMEINTEGRATIONSCHEME')
    ET.SubElement(scheme,'METHOD').text = 'IMEX'
    ET.SubElement(scheme,'ORDER').text = '2'
    info = ET.SubElement(conditions,'SOLVERINFO')
    solver_type = config.get('solver_type', 'VelocityCorrectionScheme')
    if solver_type not in ('VelocityCorrectionScheme', 'VCSWeakPressure'):
        raise ValueError('unqualified numerical solver type')
    for key,value in {'EQTYPE':'UnsteadyNavierStokes','SolverType':solver_type,
                      'AdvectionForm':'Convective','Projection':'Galerkin','GlobalSysSoln':'DirectStaticCond'}.items():
        ET.SubElement(info,'I',PROPERTY=key,VALUE=value)
    parameters = ET.SubElement(conditions,'PARAMETERS')
    t0,t1 = map(float,problem.public_metadata['domain']['time_interval'])
    steps = int(round((t1-t0)/float(config['time_step'])))
    if steps < 1 or not math.isclose(steps*config['time_step'],t1-t0,abs_tol=1e-13):
        raise ValueError('time step must divide the physical time interval')
    for key,value in {'Time':t0,'TimeStep':config['time_step'],'NumSteps':steps,
                      'Kinvis':problem.public_metadata['parameters']['viscosity'],
                      'IO_InfoSteps':max(1,steps//10),'IO_CFLSteps':10,'IO_CheckSteps':1}.items():
        ET.SubElement(parameters,'P').text = f'{key} = {value}'
    variables = ET.SubElement(conditions,'VARIABLES')
    for index,name in enumerate(('u','v','p')): ET.SubElement(variables,'V',ID=str(index)).text=name
    regions = ET.SubElement(conditions,'BOUNDARYREGIONS')
    ET.SubElement(regions,'B',ID='0').text='C['+','.join(walls)+']'
    boundaries=ET.SubElement(conditions,'BOUNDARYCONDITIONS')
    region=ET.SubElement(boundaries,'REGION',REF='0')
    for name in ('u','v'): ET.SubElement(region,'D',VAR=name,VALUE='0')
    ET.SubElement(region,'N',VAR='p',USERDEFINEDTYPE='H',VALUE='0')
    initial=ET.SubElement(conditions,'FUNCTION',NAME='InitialConditions')
    ET.SubElement(initial,'F',VAR='u,v,p',FILE='initial.fld')
    forcing=ET.SubElement(document,'FORCING')
    ET.SubElement(forcing,'FORCE',TYPE='NSExactPublicInput')
    # Coefficient snapshots are evaluated only after hidden coordinates arrive.
    ET.indent(document)
    ET.ElementTree(document).write(directory/'session.xml',encoding='unicode',xml_declaration=True)
    return steps
