"""Explicit transport-score configuration shared by the Plasma plugins.

Missing settings retain historical scoring; a transport run must name its
frozen reference calibration. No candidate results or implicit epsilon set it.
"""
from __future__ import annotations

from .plasma_transport import TRANSPORT_PROTOCOL_ID, load_transport_calibration


def transport_calibration(metric_profile=None, metric_calibration=None):
    if metric_profile is None:
        if metric_calibration is not None:
            raise ValueError("metric_calibration requires an explicit metric_profile")
        return None
    if metric_profile != TRANSPORT_PROTOCOL_ID:
        raise ValueError("unknown Plasma metric_profile")
    if metric_calibration is None:
        raise ValueError("transport scoring requires a frozen metric_calibration")
    return load_transport_calibration(metric_calibration)


def configured_matcher(settings, fallback, family):
    """Resolve task settings without silently changing a bound run objective."""
    if "metric_profile" not in settings and "metric_calibration" not in settings:
        return fallback
    calibration = transport_calibration(settings.get("metric_profile"), settings.get("metric_calibration"))
    if calibration is None:
        if getattr(fallback, "calibration", None) is not None:
            raise ValueError("cannot remove the bound transport scoring objective")
        return fallback
    if family == "vpfp":
        from .vpfp.macro_matching import VPFPTransportMacroMatcher
        matcher = VPFPTransportMacroMatcher(calibration=calibration)
    elif family == "vp_1d1v":
        from .vp_1d1v.macro_matching import VP1DTransportMacroMatcher
        matcher = VP1DTransportMacroMatcher(calibration=calibration)
    else:
        raise ValueError("unknown Plasma family")
    if (getattr(fallback, "calibration", None) is not None
            and fallback.contract_id != matcher.contract_id):
        raise ValueError("task calibration differs from the bound transport objective")
    return matcher


def transport_brief(calibration):
    """Public score definition and actual frozen scales, not hidden answers."""
    if calibration is None:
        return ""
    import json
    checked = calibration.to_dict()
    public = {key: checked[key] for key in ("protocol_id", "units", "floors")}
    return (
        " Scoring uses plasma_macro_transport_v2: S=(e_n+e_E)/8+"
        "(e_j+e_P+e_qh)/4, lower is better. Each e_A is the RMS error over "
        "all scoring points divided by sqrt(reference RMS squared + frozen "
        "scale floor squared), never a pointwise relative error. Vector norms "
        "are Euclidean; pressure uses the full Frobenius norm, so Pxy counts "
        "twice in 2D. P is central pressure and q_h=0.5*integral "
        "|v-u|^2*(v-u)*f dv; raw third moments are not heat flux. Prediction "
        "and reference are centered using their own velocity. Density/field, "
        "particle transport, stress and heat transport each receive 25%. "
        "Failures retain the formal ineligibility rule; successful-subset "
        "averages cannot promote a failed candidate. Frozen score calibration: "
        + json.dumps(public, sort_keys=True, separators=(",", ":"))
        + ". Objective SHA256=" + calibration.objective_hash + "."
    )


def check_guidance_objective(calibration, previous_guidance, settings):
    """Reject mixing legacy/v2 labels or continuing an incompatible Guide."""
    requested = transport_calibration(settings.get("metric_profile"), settings.get("metric_calibration"))
    if requested is not None and (calibration is None or requested.objective_hash != calibration.objective_hash):
        raise ValueError("Guidance settings differ from the bound scoring objective")
    if previous_guidance is None:
        return
    report = previous_guidance.report
    previous = report.get("scoring_objective_hash", report.get("reward_matcher_sha256"))
    if calibration is not None and previous != calibration.objective_hash:
        raise ValueError("previous Guidance was trained for another scoring objective")
    if calibration is None and report.get("scoring_objective_hash") is not None:
        raise ValueError("transport Guidance cannot be resumed with legacy scoring")


def check_feedback_objective(calibration, feedback):
    """Use the exact promoted score as reward; never recompute a surrogate."""
    from math import isfinite, isclose
    for item in feedback.cases:
        score = getattr(item, "metrics", {}).get(TRANSPORT_PROTOCOL_ID)
        if calibration is None:
            if score is not None:
                raise ValueError("transport labels cannot train a legacy Guidance objective")
            continue
        error = float(item.primary_error)
        if not isfinite(error) or error < 0.0:
            # Failed attempts retain invalid labels rather than being dropped
            # or replaced with more favorable roots.
            continue
        if score is None or not isclose(float(score), error, rel_tol=1.e-12, abs_tol=1.e-15):
            raise ValueError("Guidance feedback is not the promoted transport score")
