"""Source-only velocity interpolation, independent of DG state resolution.

Research helper: source v nodes are Gauss3/4 per already-selected DG cell.
High-order local interpolation crosses cell interfaces because the source is
a continuous function of v. Every source time uses the same linear operator.
"""
import numpy as np


def gauss_nodes(edges, order):
    xi, qw = np.polynomial.legendre.leggauss(order)
    half = np.diff(edges) / 2
    return ((edges[:-1] + half)[:, None] + half[:, None]*xi).ravel(), (half[:, None]*qw).ravel()


def local_interpolation(source_v, target_v, degree=11):
    """Contiguous local barycentric stencils; returns sparse rows idx, weights."""
    m = min(degree + 1, len(source_v))
    starts = np.clip(np.searchsorted(source_v, target_v)-m//2, 0, len(source_v)-m)
    idx = starts[:, None] + np.arange(m)
    nodes = source_v[idx]
    span = nodes[:, -1] - nodes[:, 0]
    z = (nodes - nodes[:, :1])/span[:, None]
    y = (target_v - nodes[:, 0])/span
    dif = z[:, :, None] - z[:, None, :]
    dif[:, np.arange(m), np.arange(m)] = 1
    bary = 1/np.prod(dif, axis=-1)
    dif = y[:, None]-z
    hits = dif == 0
    dif[hits] = 1
    weights = bary/dif
    weights /= weights.sum(axis=-1, keepdims=True)
    exact_rows = hits.any(axis=1)
    weights[exact_rows] = hits[exact_rows]
    return idx, weights


def interpolate(values, stencil):
    idx, weights = stencil
    return np.sum(values[..., idx]*weights, axis=-1)


def conditioned_interpolation(source_v, target_v, degree=11, lebesgue_limit=3.):
    """Select a containing stencil with minimal amplification; reduce degree.

    Unequal adjacent DG cell widths can make a centered high-degree stencil
    catastrophically unstable. Test all containing stencils, and only retain
    a higher-order rule if its Lebesgue weight sum stays below a fixed bound.
    """
    out_idx = np.zeros((len(target_v), degree+1), dtype=int)
    out_weights = np.zeros((len(target_v), degree+1))
    accepted_degrees = np.zeros(len(target_v),dtype=int)
    for row,target in enumerate(target_v):
        j=int(np.searchsorted(source_v,target))
        for deg in range(degree,0,-2):
            m=deg+1
            starts=np.unique(np.clip(np.arange(j-m+1,j),0,len(source_v)-m))
            best=None
            for start in starts:
                idx=start+np.arange(m)
                nodes=source_v[idx]
                scale=nodes[-1]-nodes[0]
                z=(nodes-nodes[0])/scale; y=(target-nodes[0])/scale
                d=z[:,None]-z[None,:]; np.fill_diagonal(d,1.)
                bary=1/np.prod(d,axis=1)
                delta=y-z
                if (delta==0).any():
                    weights=(delta==0).astype(float)
                else:
                    weights=bary/delta; weights/=sum(weights)
                condition=sum(abs(weights))
                if best is None or condition<best[0]:best=(condition,idx,weights)
            if best[0]<=lebesgue_limit or deg==1:
                out_idx[row,:m]=best[1];out_weights[row,:m]=best[2]
                accepted_degrees[row]=deg
                break
    return (out_idx,out_weights),accepted_degrees


def interpolation_matrix(source_v, target_v, degree=11):
    idx, weights = local_interpolation(source_v, target_v, degree)
    matrix = np.zeros((len(target_v), len(source_v)))
    matrix[np.arange(len(target_v))[:, None], idx] = weights
    return matrix


def element_interpolation(edges, order, target_v):
    """Stable local polynomial source elements, independent of state cells."""
    xi, _ = np.polynomial.legendre.leggauss(order)
    j=np.clip(np.searchsorted(edges,target_v,side='right')-1,0,len(edges)-2)
    z=2*(target_v-edges[j])/(edges[j+1]-edges[j])-1
    diff=xi[:,None]-xi[None,:];np.fill_diagonal(diff,1.)
    bary=1/np.prod(diff,axis=1)
    d=z[:,None]-xi
    hit=d==0;d[hit]=1.
    weights=bary/d;weights/=weights.sum(axis=1,keepdims=True)
    weights[hit.any(axis=1)]=hit[hit.any(axis=1)]
    return j[:,None]*order+np.arange(order),weights


def adapt_elements(sample, lower, upper, order=10, max_nodes=360, tolerance=1e-5,
                   target_v=None, target_weights=None):
    """Greedy bisection using exact independent 2*order Gauss probes.

    sample(v) returns all fixed pilot (time,x) rows, final dimension velocity.
    Error is max over those rows of weighted L2(v) kinetic interpolation,
    using (1+v^6). The tolerance is not an error certificate between probes.
    """
    coarse,w0=np.polynomial.legendre.leggauss(order)
    fine,w1=np.polynomial.legendre.leggauss(2*order)
    test=element_interpolation(np.array([-1.,1.]),order,fine)
    queried=0
    target_values=None if target_v is None else sample(target_v)
    if target_v is not None:
        queried+=len(target_v)
    def assess(lo,hi,vals):
        mid=(lo+hi)/2;half=(hi-lo)/2
        cv=mid+half*coarse;fv=mid+half*fine
        defect=interpolate(vals[...,:order],test)-vals[...,order:]
        score=np.max(np.sum(abs(defect)**2*(half*w1*(1+fv**6)),axis=-1))
        if target_v is not None:
            mask=(target_v>=lo)&(target_v<hi)
            if mask.any():
                interp=element_interpolation(np.array([lo,hi]),order,target_v[mask])
                difference=interpolate(vals[...,:order],interp)-target_values[...,mask]
                norm=np.max(np.sum(abs(difference)**2*(target_weights[mask]*(1+target_v[mask]**6)),axis=-1))
                score=max(score,norm)
        return [lo,hi,float(score)]
    def assess_many(intervals):
        nonlocal queried
        probe_v=[]
        for lo,hi in intervals:
            mid=(lo+hi)/2;half=(hi-lo)/2
            probe_v.append(np.r_[mid+half*coarse,mid+half*fine])
        probe_v=np.concatenate(probe_v)
        values=sample(probe_v);queried+=len(probe_v)
        return [assess(lo,hi,values[...,i*3*order:(i+1)*3*order])
                for i,(lo,hi) in enumerate(intervals)]
    # Several equal starter cells avoid a single-node blind spot for narrow beams.
    edges=np.linspace(lower,upper,9)
    cells=assess_many(list(zip(edges[:-1],edges[1:])))
    while (len(cells)+1)*order<=max_nodes and sum(c[2] for c in cells)>tolerance**2:
        i=int(np.argmax([c[2] for c in cells]));lo,hi,_=cells.pop(i);mid=(lo+hi)/2
        cells.extend(assess_many([(lo,mid),(mid,hi)]))
    cells.sort()
    edges=np.r_[[c[0] for c in cells],cells[-1][1]]
    return edges,dict(nodes=len(cells)*order,pilot_velocity_values=queried,
                      estimated_defect=float(np.sqrt(sum(c[2] for c in cells))),
                      passed=sum(c[2] for c in cells)<=tolerance**2)
