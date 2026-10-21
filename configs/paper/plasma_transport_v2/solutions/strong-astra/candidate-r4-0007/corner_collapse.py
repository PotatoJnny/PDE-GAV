"""Empirically guarded removal of redundant brackets around trusted corners.

Only existing independently detected trusted joins are eligible. A candidate
must have a tight three-node bracket, one-sided polynomial predictability,
and continuity at that trusted join. Final source-only OU response checks
compare the original and proposed joins at equal full-frame budgets. Failure
at either stage preserves the original bracket. No time-grid frequency is
assumed, and no additional public-source calls are made.
"""
from __future__ import annotations
import numpy as np


def _local_center_test(times,values,center,radius,bracket=None):
    # Fit each side independently; two heldout folds always retain the outer
    # endpoints, so validation measures interpolation rather than extrapolation.
    fits=[];rms_errors=[];max_errors=[];component_errors=[];conditions=[];gains=[]
    if bracket is None:
        bracket=(center,center)
    for side in (-1,1):
        outside=(times<=bracket[0]) if side<0 else (times>=bracket[1])
        indices=np.flatnonzero(outside&((times-center)*side>0)&((times-center)*side<=radius))
        if len(indices)<8:
            return False,{'reason':'insufficient one-sided source pilot points'}
        x=(times[indices]-center)/radius
        y=values[indices].reshape(len(indices),-1)
        powers=np.polynomial.polynomial.polyvander(x,3)
        side_fits=[]
        for offset in (1,2):
            test=(np.arange(len(indices))%3==offset)
            test[[0,-1]]=False
            train=~test
            if np.count_nonzero(test)<2:
                return False,{'reason':'insufficient heldout source pilot points'}
            condition=float(np.linalg.cond(powers[train]))
            intercept_gain=float(np.linalg.norm(np.linalg.pinv(powers[train],rcond=1.e-12)[0]))
            if not np.isfinite(condition) or condition>1.e4 or intercept_gain>4.:
                return False,{'reason':'ill-conditioned one-sided source extrapolation',
                              'condition':condition,'intercept_noise_gain':intercept_gain}
            fit=np.linalg.lstsq(powers[train],y[train],rcond=1.e-12)[0]
            defect=powers[test]@fit-y[test]
            rms_errors.append(float(np.sqrt(np.mean(defect*defect))))
            max_errors.append(float(np.max(abs(defect))))
            component_errors.append(np.sqrt(np.mean(defect*defect,axis=0)))
            conditions.append(condition);gains.append(intercept_gain)
            side_fits.append(fit)
        fits.append(np.mean(side_fits,axis=0))
    intercepts=np.stack([fit[0] for fit in fits])
    center_value=values[np.searchsorted(times,center)].ravel()
    contact=intercepts-center_value
    difference=intercepts[0]-intercepts[1]
    continuity=float(np.sqrt(np.mean(difference*difference)))
    contact_rms=float(np.max(np.sqrt(np.mean(contact*contact,axis=1))))
    contact_max=float(np.max(abs(contact)))
    amplitude=max(1.,float(np.sqrt(np.mean(center_value*center_value))))
    noise=max(max(rms_errors),256*np.finfo(float).eps*amplitude)
    max_noise=max(max(max_errors),256*np.finfo(float).eps*amplitude)
    component_noise=np.maximum(np.max(component_errors,axis=0),
                              256*np.finfo(float).eps*np.maximum(1.,abs(center_value)))
    amplification=1.+max(gains)
    component_ratio=float(np.max(np.maximum(np.max(abs(contact),axis=0),abs(difference))
                                  /(component_noise*amplification)))
    # The absolute scale cap prevents a poorly fitted local feature from
    # legitimizing a large discontinuity via an inflated empirical noise level.
    passed=(noise<=2.e-5*amplitude and continuity<=8*noise
            and contact_rms<=8*noise and contact_max<=8*max_noise and component_ratio<=8.)
    return passed,{'continuity_rms':continuity,'contact_rms':contact_rms,
                   'contact_max':contact_max,'heldout_rms':noise,'heldout_max':max_noise,
                   'component_noise_ratio':component_ratio,'condition':max(conditions),
                   'intercept_noise_gain':max(gains),
                   'source_amplitude':amplitude,'passed':bool(passed)}


def collapse_corner_brackets(times,values,corner_indices,trusted_times,*,theta,nu,
        base_spacing,select_knots,spline_builder,response_diagnostic,
        knot_counts=(129,257,513,769),should_continue=None,validation_values=None):
    """Use optional extra source moments only for local corner validation.

    The first-ten ``values`` continue to drive knot selection and the OU
    response gate. ``validation_values`` may contain all available same-site
    pilot moments; it never changes the global diagnostic's physical meaning.
    """
    times=np.asarray(times,float);values=np.asarray(values,float)
    original=np.unique(np.asarray(corner_indices,int));trusted=np.unique(np.asarray(trusted_times,float))
    if (times.ndim!=1 or len(times)<2 or np.any(np.diff(times)<=0)
            or values.shape[0]!=len(times) or not np.isfinite(values).all()
            or base_spacing<=0 or not np.all(np.isin(trusted,times))):
        raise ValueError('finite source pilot, trusted sampled corners, and base spacing required')
    kept=set(map(int,original));diagnostics={'enabled':False,'original_count':len(original),
                                            'local':[],'budgets':[]}
    def continuing():
        if should_continue is None or should_continue():return True
        diagnostics['reason']='cooperative corner-collapse deadline'
        return False
    if not continuing():return original,diagnostics
    local_values=values
    if validation_values is not None:
        local_values=np.asarray(validation_values,float)
        if (local_values.ndim!=3 or local_values.shape[:2]!=values.shape[:2]
                or local_values.shape[2]<1 or not np.isfinite(local_values).all()):
            raise ValueError('validation source moments must share pilot times and sites')
        if not continuing():return original,diagnostics
        diagnostics['validation_channels']=int(local_values.shape[2])
    corner_times=times[original]
    if not len(trusted):return original,diagnostics
    ownership=np.argmin(abs(corner_times[:,None]-trusted[None,:]),axis=1)
    for owner,center in enumerate(trusted):
        if not continuing():return original,diagnostics
        # Preserve the logarithmically sampled initial layer independently of
        # local fit quality; its source derivatives can vary on tiny time scales.
        if center-times[0]<=8*base_spacing:
            continue
        group=original[ownership==owner]
        if len(group)!=3 or center not in times[group]:continue
        left,right=times[group[[0,-1]]]
        if not left<center<right or max(center-left,right-center)>.125*base_spacing:
            continue
        passed,detail=_local_center_test(times,local_values,center,.5*base_spacing,(left,right))
        entry=dict(center=float(center),bracket=[float(left),float(right)],**detail)
        diagnostics['local'].append(entry)
        if passed:
            kept.remove(int(group[0]));kept.remove(int(group[-1]))
    proposed=np.array(sorted(kept),int)
    diagnostics['proposed_count']=len(proposed)
    if len(proposed)==len(original):return original,diagnostics
    # Reuse the progressive acquisition order across all prescribed budgets.
    counts=sorted(set(min(int(count),len(times)) for count in knot_counts
                      if int(count)>=max(len(original),len(proposed))+2))
    if not counts:
        diagnostics['reason']='insufficient source-frame budget for original joins'
        return original,diagnostics
    orders=[]
    for joins in (original,proposed):
        if not continuing():return original,diagnostics
        orders.append(select_knots(times,values,theta,nu,max(counts),corners=joins))
    for count in counts:
        if not continuing():return original,diagnostics
        metrics=[]
        for joins,order in zip((original,proposed),orders):
            selected=np.sort(order[:count]);breaks=np.flatnonzero(np.isin(selected,joins))
            residual=spline_builder(times[selected],values[selected],breaks)(times)-values
            metrics.append(response_diagnostic(times,residual,theta,nu))
        old,new=metrics
        passed=(new['response_rms']<=.95*old['response_rms']
                and new['density_primitive_rms']<=1.05*old['density_primitive_rms']
                and new['density_final_rms']<=1.10*old['density_final_rms']
                and new['max_defect']<=1.25*old['max_defect'])
        diagnostics['budgets'].append({'count':count,'original':old,'proposed':new,'passed':bool(passed)})
        if not passed:
            diagnostics['reason']='equal-budget source interpolation response gate rejected collapse'
            return original,diagnostics
    diagnostics.update(enabled=True,selected_count=len(proposed),removed_count=len(original)-len(proposed))
    diagnostics['acquisition_order']=orders[1].tolist()
    return proposed,diagnostics
