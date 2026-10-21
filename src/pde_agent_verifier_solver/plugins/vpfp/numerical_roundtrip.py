"""Restricted IEEE roundtrip gate for the two strong VPFP value codecs.

Budget inputs are ONLY frozen normalization constants and source FP32 values;
the decoded answer is never supplied to ``value_roundtrip_budget``. Its small
arithmetic trace propagates operation-count bounds, not observed residuals.
This is a representation-mechanics gate, not a physical tolerance or admission.

At each node E bounds BOTH the replay and any supported native arithmetic
path relative to the same real-arithmetic graph. Thus a native operand has
magnitude <= |replay|+2E, even if layout changes the legal summation order.
Propagation is |A|E plus gamma_k(d) times that operand envelope, plus k*tiny(d).
Stored FP32 q=cast(scale), p=reciprocal(q) are exact graph constants.
The ideal roundtrip's fixed affine defect is bounded separately using R*W-I
and q*p-1. FP64 construction of that defect and all positive budget arithmetic
are themselves rounded outward. The tiny terms conservatively cover hidden
underflow in sums/products; nonzero subnormal source/trace/output tensors and
overflow, non-nearest rounding, AMP, or CUDA TF32 are explicitly unsupported.
Only the existing CPU-built immutable payload operators are accepted: all
buffers must match a canonical payload reconstruction bit for bit. A small
R*W-I budget is not evidence that an arbitrary inverse is the right operator.
"""
from dataclasses import dataclass
import ctypes
import hashlib
import json
import math
import platform
import sys

import torch

from .structured_diffusion import _split
from .value_control_diffusion import extract_value_controls, lift_value_controls, _value_split
from .temporal_covariance_diffusion import VPFPTemporalCovarianceNormalizer
from .initial_conditioned_diffusion import VPFPInitialConditionedNormalizer

SCHEMA = "vpfp_restricted_IEEE_value_and_lift_roundtrip_gate_v1"
VALUE_RTOL, VALUE_ATOL = 2.e-4, 2.e-5


class UnsupportedRoundtripArithmetic(ValueError):
    pass


def _normal(value, name):
    if value.dtype == torch.float32:
        bits, exponent, mantissa = value.view(torch.int32), 0x7f800000, 0x007fffff
    elif value.dtype == torch.float64:
        bits, exponent, mantissa = value.view(torch.int64), 0x7ff0000000000000, 0x000fffffffffffff
    else:
        raise UnsupportedRoundtripArithmetic(f"{name}: unsupported arithmetic precision")
    # Floating comparisons themselves can flush denormal operands to zero.
    if torch.any((bits & exponent) == exponent):
        raise UnsupportedRoundtripArithmetic(f"{name}: nonfinite/overflow arithmetic")
    if torch.any(((bits & exponent) == 0) & ((bits & mantissa) != 0)):
        raise UnsupportedRoundtripArithmetic(f"{name}: nonzero subnormal arithmetic is unsupported")


def _nearest_rounding(device):
    # FE_TONEAREST=0 is the supported Linux x86_64 ABI, not a portable guess.
    # Check host fenv as well: Python scalar gamma/outward-bound construction
    # uses host FP64 even when the answer and tensor trace live on CUDA.
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise UnsupportedRoundtripArithmetic("rounding-mode verification only supports Linux x86_64")
    try:
        current = ctypes.CDLL(None).fegetround
        current.argtypes, current.restype = [], ctypes.c_int
        mode = current()
    except (AttributeError, OSError) as error:
        raise UnsupportedRoundtripArithmetic("host rounding mode cannot be verified") from error
    if mode != 0:
        raise UnsupportedRoundtripArithmetic("round-to-nearest-even is required for host arithmetic")
    for target in {torch.device("cpu"), device}:
        for dtype, integer, one_bits in ((torch.float32, torch.int32, 0x3f800000),
                (torch.float64, torch.int64, 0x3ff0000000000000)):
            left = torch.tensor([one_bits, one_bits+1], dtype=integer, device=target).view(dtype)
            half = torch.full_like(left, torch.finfo(dtype).eps/2)
            result = (left+half).view(integer)
            expected = torch.tensor([one_bits, one_bits+2], dtype=integer, device=target)
            if not torch.equal(result, expected):
                raise UnsupportedRoundtripArithmetic("round-to-nearest-even device tie probes failed")


def _frozen_operator_integrity(normalizer):
    """Validate construction independently of both residual and RW-I budget.

    Fit and native reload build these operators on CPU before device copies.
    Rebuild the same payload on CPU: a replaced version-zero buffer must not
    become acceptable merely because the affine-defect allowance grows.
    """
    payload = normalizer.to_payload()
    canonical = type(normalizer).from_payload(payload)
    actual, expected = dict(normalizer.named_buffers()), dict(canonical.named_buffers())
    if set(actual) != set(expected):
        raise ValueError("frozen operator reconstruction buffer names differ")
    for name, value in actual.items():
        _normal(value, f"frozen operator {name}")
        other = expected[name]
        if (value.dtype != torch.float64 or value.shape != other.shape
                or not torch.equal(value.detach().cpu().view(torch.int64), other.view(torch.int64))):
            raise ValueError(f"frozen operator differs from canonical CPU payload reconstruction: {name}")
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return {"canonical_CPU_payload_rebuild_bitwise_passed": True,
        "normalizer_payload_sha256": digest, "verified_buffer_count": len(actual),
        "supported_construction": "CPU_fit_or_native_payload_reload_then_exact_device_copy"}


def _gamma(count, dtype):
    unit = torch.finfo(dtype).eps/2
    if count*unit >= 1:
        raise UnsupportedRoundtripArithmetic("invalid IEEE operation-count bound")
    return math.nextafter(count*unit/(1-count*unit), math.inf)


def _up(value, *, positive=None):
    """Upper rounding for a nonnegative FP64 elementary operation.

    A positive result flushed below normal range is replaced by tiny64;
    otherwise nextafter bounds the one correctly-rounded operation. Exact
    zeros stay zero only when operands establish that the result is zero.
    """
    if not torch.isfinite(value).all() or torch.any(value < 0):
        raise UnsupportedRoundtripArithmetic("positive budget arithmetic overflowed")
    positive = value > 0 if positive is None else positive
    value = torch.where(positive, value.clamp_min(torch.finfo(torch.float64).tiny), value)
    result = torch.where(positive, torch.nextafter(value, torch.full_like(value, float("inf"))), value)
    if not torch.isfinite(result).all():
        raise UnsupportedRoundtripArithmetic("outward-rounded budget overflowed")
    return result


def _add(a, b):
    return _up(a+b, positive=(a > 0) | (b > 0))


def _mul(a, b):
    return _up(a*b, positive=(a > 0) & (b > 0))


def _dot(matrix, values):
    """Upper bound of the exact NONNEGATIVE group matrix-vector product."""
    count = matrix.shape[-1]
    result = torch.einsum("hts,bhsm->bhtm", matrix, values)
    # Positive dot products may have rounded down or flushed tiny products.
    numerator = _add(result, 2*count*torch.finfo(torch.float64).tiny)
    denominator = math.nextafter(1-_gamma(2*count, torch.float64), -math.inf)
    return _up(numerator/denominator, positive=numerator > 0)


def _matrix_abs_product(left, right):
    # Reuse exactly the same certified positive dot routine, not an unchecked
    # FP64 matmul inside the budget construction.
    return _dot(left.abs(), right.abs()[None])[0]


@dataclass
class _Node:
    value: torch.Tensor
    error: torch.Tensor

    def slice(self, selection):
        return _Node(self.value[selection], self.error[selection])


def _constant(value):
    _normal(value, "trace constant")
    return _Node(value, torch.zeros_like(value, dtype=torch.float64))


def _envelope(node):
    return _add(node.value.double().abs(), _mul(2., node.error))


def _range(magnitude, dtype):
    # A positive-term upper envelope must also exclude overflow in any legal
    # path, not merely in this replay's particular cancellation/summation.
    safe_maximum = math.nextafter(torch.finfo(dtype).max/(1+_gamma(1, dtype)), 0.)
    if torch.any(magnitude > safe_maximum):
        raise UnsupportedRoundtripArithmetic("arithmetic envelope cannot exclude overflow")


def _cast(node, dtype):
    if dtype == torch.float32:
        magnitude = _envelope(node)
        _range(magnitude, dtype)
        if torch.any((node.value != 0) & (node.value.abs() < torch.finfo(dtype).tiny)):
            raise UnsupportedRoundtripArithmetic("trace cast enters unsupported FP32 subnormal range")
    value = node.value.to(dtype)
    _normal(value, "trace cast")
    if node.value.dtype == dtype or dtype == torch.float64:
        return _Node(value, node.error)  # FP32 -> FP64 is exact.
    error = _add(node.error, _add(_mul(_gamma(1, dtype), magnitude), torch.finfo(dtype).tiny))
    return _Node(value, error)


def _scale(node, coefficient):
    _normal(coefficient, "trace multiplier")
    magnitude = _mul(_envelope(node), coefficient.double().abs())
    _range(magnitude, node.value.dtype)
    if node.value.dtype == torch.float32:
        exact_product = node.value.double()*coefficient.double()
        if torch.any((exact_product != 0) & (exact_product.abs() < torch.finfo(torch.float32).tiny)):
            raise UnsupportedRoundtripArithmetic("trace product enters unsupported FP32 subnormal range")
    value = node.value*coefficient
    _normal(value, "trace multiplication")
    local = _add(_mul(_gamma(1, value.dtype), magnitude), torch.finfo(value.dtype).tiny)
    return _Node(value, _add(_mul(coefficient.double().abs(), node.error), local))


def _sum(left, right):
    if left.value.dtype != right.value.dtype:
        raise UnsupportedRoundtripArithmetic("trace addition precision differs")
    magnitude = _add(_envelope(left), _envelope(right))
    _range(magnitude, left.value.dtype)
    value = left.value+right.value
    _normal(value, "trace addition")
    local = _add(_mul(_gamma(1, value.dtype), magnitude), torch.finfo(value.dtype).tiny)
    return _Node(value, _add(_add(left.error, right.error), local))


def _linear(node, matrix):
    if matrix.dtype != torch.float64 or node.value.dtype != torch.float64:
        raise UnsupportedRoundtripArithmetic("only FP64 frozen temporal contractions are supported")
    _normal(matrix, "trace temporal matrix")
    magnitude = _dot(matrix.abs(), _envelope(node))
    _range(magnitude, torch.float64)
    value = torch.einsum("hts,bhsm->bhtm", matrix, node.value)
    _normal(value, "trace temporal product")
    count = matrix.shape[-1]
    local = _add(_mul(_gamma(2*count, torch.float64), magnitude), 2*count*torch.finfo(torch.float64).tiny)
    return _Node(value, _add(_dot(matrix.abs(), node.error), local))


def _cat(left, right):
    return _Node(torch.cat((left.value, right.value), dim=-2), torch.cat((left.error, right.error), dim=-2))


def _inverse_defect(root, inverse):
    """Outward |R*W-I|, including FP64 product and identity subtraction."""
    count = root.shape[-1]
    product = root@inverse
    _normal(product, "fixed inverse product")
    identity = torch.eye(count, device=root.device, dtype=torch.float64)[None]
    difference = product-identity
    product_error = _add(_mul(_gamma(2*count, torch.float64), _matrix_abs_product(root, inverse)),
        2*count*torch.finfo(torch.float64).tiny)
    subtract_error = _add(_mul(_gamma(1, torch.float64), _add(product.abs(), identity)), torch.finfo(torch.float64).tiny)
    return _add(difference.abs(), _add(product_error, subtract_error))


def _group_budget(values, q, p, mean, root=None, inverse=None, regression=None):
    """One actual dtype-preserving arithmetic trace; no output residual."""
    exact_input = values.double()
    node = _scale(_constant(values), p)
    if root is not None:
        centered = _sum(_cast(node, torch.float64), _constant(-mean))
        if regression is None:
            white = _cast(_linear(centered, inverse), torch.float32)
            decoded = _sum(_linear(_cast(white, torch.float64), root), _constant(mean))
        else:
            initial = centered.slice((slice(None), slice(None), slice(0, 1), slice(None)))
            future = centered.slice((slice(None), slice(None), slice(1, None), slice(None)))
            residual = _sum(future, _scale(initial, -regression))
            white = _cast(_linear(residual, inverse), torch.float32)
            decoded_future = _sum(_linear(_cast(white, torch.float64), root), _scale(initial, regression))
            decoded = _sum(_cat(initial, decoded_future), _constant(mean))
        node = _cast(decoded, torch.float32)
    node = _scale(node, q)
    # The real graph need not be the identity: reciprocal32 and the stored
    # inverse factors are rounded constants, not ideal reciprocal operators.
    q64, p64 = q.double(), p.double()
    qp = q64*p64
    scalar_defect = _add((qp-1).abs(), _add(
        _mul(_gamma(1, torch.float64), _mul(q64.abs(), p64.abs())),
        _add(_mul(_gamma(1, torch.float64), _add(qp.abs(), 1.)), torch.finfo(torch.float64).tiny)))
    defect = _mul(scalar_defect, exact_input.abs())
    if root is not None:
        delta = _inverse_defect(root, inverse)
        if regression is None:
            magnitude = _add(_mul(p64, exact_input.abs()), mean.abs())
            defect = _add(defect, _mul(q64, _dot(delta, magnitude)))
        else:
            initial, future = exact_input[..., :1, :], exact_input[..., 1:, :]
            magnitude = _add(_mul(p64, _add(future.abs(), _mul(regression.abs(), initial.abs()))),
                _add(mean[..., 1:, :].abs(), _mul(regression.abs(), mean[..., :1, :].abs())))
            future_defect = _mul(q64, _dot(delta, magnitude))
            defect = _add(defect, torch.cat((torch.zeros_like(defect[..., :1, :]), future_defect), dim=-2))
    return _add(node.error, defect), defect


def _configuration(normalizer, raw):
    if not isinstance(raw, torch.Tensor) or raw.dtype != torch.float32 or raw.ndim != 2 or len(raw) == 0:
        raise UnsupportedRoundtripArithmetic("source must be complete FP32 rows")
    _normal(raw, "source")
    device = raw.device
    _nearest_rounding(device)
    if torch.is_autocast_enabled(device.type):
        raise UnsupportedRoundtripArithmetic("AMP arithmetic is unsupported")
    if device.type == "cuda" and (torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32):
        raise UnsupportedRoundtripArithmetic("CUDA TF32 is unsupported")
    for dtype, integer, half_minimum_bits in ((torch.float32, torch.int32, 0x00400000),
            (torch.float64, torch.int64, 0x0008000000000000)):
        probe = torch.tensor(torch.finfo(dtype).tiny, device=device, dtype=dtype)*.5
        # Test output flushing AND subnormal-input denormals-are-zero; these
        # flags need not be enabled together. Integer construction avoids DAZ.
        subnormal_input = torch.tensor(half_minimum_bits, device=device, dtype=integer).view(dtype)
        input_probe = subnormal_input*2.
        if int(probe.view(integer)) != half_minimum_bits or int(input_probe.view(integer)) != 2*half_minimum_bits:
            raise UnsupportedRoundtripArithmetic("FTZ/DAZ arithmetic mode is unsupported")
    if type(normalizer) is VPFPInitialConditionedNormalizer:
        normalizer._check()
        temporal, config, conditional = normalizer.temporal, normalizer.config.value_config, True
    elif type(normalizer) is VPFPTemporalCovarianceNormalizer:
        normalizer._check_buffer_dtype()
        temporal, config, conditional = normalizer, normalizer.config, False
    else:
        raise UnsupportedRoundtripArithmetic("only the frozen strong cov and conditional codecs are supported")
    # Conditional rho uses the same Schur affine graph already budgeted below;
    # only its frozen constants differ. The unconditional rho bypass remains
    # the originally reviewed scope. No acceptance tolerance changes here.
    rho_supported = temporal.rho_policy == "identity" or (conditional and temporal.rho_policy == "covariance")
    if (not rho_supported or temporal.covariance_strength != 1.
            or temporal.covariance_recipe != "mixture" or not temporal.temporal.fit_metadata["center_kinetic_dc"]):
        raise UnsupportedRoundtripArithmetic("unsupported covariance/centering recipe")
    intervals = config.control_count-1
    if intervals < 2 or intervals & (intervals-1):
        raise UnsupportedRoundtripArithmetic("only dyadic uniform control intervals are covered")
    if next(temporal.buffers()).device != device:
        raise UnsupportedRoundtripArithmetic("normalizer and source devices differ")
    return temporal, config, conditional


@torch.no_grad()
def value_roundtrip_budget(normalizer, source_raw):
    """Return (reduced Bv, metadata); deliberately no restored/output input."""
    temporal, config, conditional = _configuration(normalizer, source_raw)
    integrity = _frozen_operator_integrity(normalizer)
    values = extract_value_controls(source_raw, config)  # Original 8*eps check.
    rho, kinetic = _value_split(values, config)
    q = temporal.parent.group_scales.to(source_raw)
    p = q.reciprocal()  # Same native-device rounded constants as parent.encode.
    _normal(q, "q32 normalization scale")
    _normal(p, "p32 reciprocal")
    if torch.any(q <= 0) or torch.any(p <= 0):
        raise UnsupportedRoundtripArithmetic("normalization scale is not positive")
    if conditional:
        roots, inverses = normalizer.square_root, normalizer.inverse_square_root
        regressions = normalizer.regression
    else:
        roots, inverses = temporal.temporal.square_root, temporal.temporal.inverse_square_root
        regressions = None
    budgets, fixed_defects = [], {}
    for group, block, section in (("rho", rho[:, None], slice(0, 1)), ("kinetic", kinetic, slice(1, None))):
        mean = torch.zeros_like(block[:1], dtype=torch.float64)
        if group == "kinetic":
            mean[..., 0] = temporal.temporal.kinetic_dc_mean[None]
        root, inverse = roots[section], inverses[section]
        if group == "rho" and not conditional:
            root = inverse = None  # Native rho bypass: never apply fitted C.
        regression = None if regressions is None else regressions[section][None, :, :, None]
        budget, defect = _group_budget(block, q[section][None, :, None, None],
            p[section][None, :, None, None], mean, root, inverse, regression)
        budgets.append(budget.flatten(1))
        fixed_defects[group] = {"maximum_fixed_affine_defect_budget": float(defect.max()),
            "definition": "rounded_constant_qp_minus_I_and_temporal_RW_minus_I_not_observed_output"}
    bound = torch.cat(budgets, dim=1)
    return bound, {"schema": SCHEMA, "budget_inputs": "source_FP32_values_and_frozen_codec_constants_only",
        "observed_residual_used_for_budget": False, "u32": 2.**-24, "u64": 2.**-53,
        "dot_product_operation_count": "2*inner_dimension", "positive_FP64_budget_arithmetic": "nextafter_upper_plus_dot_gamma_and_tiny",
        "fixed_operator_defect_included": True, "fixed_affine_defect": fixed_defects,
        "operator_integrity": integrity, "q32": q.cpu().tolist(), "p32": p.cpu().tolist(),
        "conditional_clean_context_branch": conditional, "rho_policy": temporal.rho_policy,
        "control_count": config.control_count,
        "native_operand_envelope": "absolute_replay_value_plus_two_propagated_errors",
        "FTZ_DAZ_probe_passed": True, "host_and_device_round_to_nearest_even_verified": True,
        "subnormal_or_nonfinite_tensors": "unsupported", "physical_tolerance": False}


def _absolute_difference_operator(values, config):
    h_inv = float(config.control_count-1)  # Dyadic: coefficients exact in FP32/64.
    return torch.cat((_mul(_add(values[..., 1:2, :], values[..., :1, :]), h_inv),
        _mul(_add(values[..., 2:, :], values[..., :-2, :]), h_inv/2),
        _mul(_add(values[..., -1:, :], values[..., -2:-1, :]), h_inv)), dim=-2)


def _source_support_budget(values, slopes, config):
    """Bound |stored_s-real_Dv| implied by the UNCHANGED FP64 8eps check.

    Besides its advertised 8eps32 allowance, include the check's own FP64
    Dv, neighboring-magnitude, tolerance, and comparison-subtraction errors.
    Inputs are source values/slopes, never the observed check residual.
    """
    tiny = torch.finfo(torch.float64).tiny
    absolute_Dv = _absolute_difference_operator(values.double().abs(), config)
    slope = slopes.double().abs()
    difference_error = _add(_mul(_gamma(2, torch.float64), absolute_Dv), 2*tiny)
    # Upper bound (1+gamma)*A without rounding 1+gamma back down to one.
    neighboring = _add(absolute_Dv, _add(_mul(_gamma(3, torch.float64), absolute_Dv), 3*tiny))
    threshold = _add(_mul(8*torch.finfo(torch.float32).eps, _add(neighboring, slope)),
        8*torch.finfo(torch.float32).tiny)
    threshold = _add(threshold, _add(_mul(_gamma(3, torch.float64), threshold), 3*tiny))
    subtraction = _add(_mul(_gamma(1, torch.float64),
        _add(slope, _add(absolute_Dv, difference_error))), tiny)
    return _add(threshold, _add(difference_error, subtraction))


def _error_upper(observed, source):
    # Source/output FP32 are exact in FP64; account for final subtraction too.
    a, b = observed.double(), source.double()
    return _add((a-b).abs(), _add(_mul(_gamma(1, torch.float64), _add(a.abs(), b.abs())), torch.finfo(torch.float64).tiny))


@torch.no_grad()
def audit_normalizer_roundtrip(normalizer, source_raw, restored):
    """Finite JSON report; rejection/unsupported status never changes answers."""
    report = {"schema": SCHEMA, "passed": False, "status": "rejected", "groups": {},
        "source_original_8eps_support_passed": False,
        "value_tolerance": {"rtol": VALUE_RTOL, "atol": VALUE_ATOL},
        "physical_admission": False, "answer_modified": False}
    try:
        temporal, config, _ = _configuration(normalizer, source_raw)
        if restored.shape != source_raw.shape or restored.dtype != source_raw.dtype or restored.device != source_raw.device:
            raise UnsupportedRoundtripArithmetic("restored shape/precision/device mismatch")
        _normal(restored, "restored")
        bound, metadata = value_roundtrip_budget(normalizer, source_raw)
        report["budget"] = metadata
        report["source_original_8eps_support_passed"] = True
        new_values = extract_value_controls(restored, config, validate_slopes=False)
        expected = lift_value_controls(new_values, config)
        exact = torch.eq(restored.view(torch.int32), expected.view(torch.int32))
        report["restored_native_lift_bitwise_passed"] = bool(exact.all())
        report["restored_native_lift_bitwise_mismatches"] = int((~exact).sum())
        start = config.full_config.rho_dimension
        inactive = restored[:, start:start+config.full_config.inactive_dim]
        report["inactive_positive_zero_bitwise_passed"] = bool((inactive.view(torch.int32) == 0).all())
        left, right = _split(source_raw, config.full_config), _split(restored, config.full_config)
        budgets = _value_split(bound, config)
        passed = report["restored_native_lift_bitwise_passed"] and report["inactive_positive_zero_bitwise_passed"]
        for name, source, output, budget in zip(("rho", "kinetic"), left, right, budgets):
            if name == "rho":
                source, output, budget = source[:, None], output[:, None], budget[:, None]
            v, s = source[:, :, 0], source[:, :, 1]
            new_v, new_s = output[:, :, 0], output[:, :, 1]
            value_error = _error_upper(new_v, v)
            old = torch.isclose(new_v, v, rtol=VALUE_RTOL, atol=VALUE_ATOL)
            source_support = _source_support_budget(v, s, config)
            magnitude = _absolute_difference_operator(_add(v.double().abs(), budget), config)
            output_lift = _add(_mul(_gamma(2, torch.float32), magnitude), 2*torch.finfo(torch.float32).tiny)
            slope_budget = _add(_absolute_difference_operator(budget, config), _add(source_support, output_lift))
            slope_error = _error_upper(new_s, s)
            current = {"value_old_allclose_passed": bool(old.all()), "value_IEEE_budget_passed": bool((value_error <= budget).all()),
                "slope_propagated_budget_passed": bool((slope_error <= slope_budget).all()),
                "value_old_tolerance_failures": int((~old).sum()), "value_IEEE_budget_failures": int((value_error > budget).sum()),
                "slope_budget_failures": int((slope_error > slope_budget).sum()),
                "maximum_value_error": float((new_v.double()-v.double()).abs().max()), "maximum_value_budget": float(budget.max()),
                "maximum_value_error_over_budget": float((value_error/budget).max()),
                "maximum_slope_error": float((new_s.double()-s.double()).abs().max()), "maximum_slope_budget": float(slope_budget.max()),
                "source_FP64_support_check_rounding_included": True,
                "maximum_source_support_budget": float(source_support.max()),
                "maximum_output_lift_rounding_budget": float(output_lift.max()),
                "slope_units": "d/dtau_not_physical_time", "budget_fitted_to_residual": False}
            report["groups"][name] = current
            passed = passed and all(current[key] for key in ("value_old_allclose_passed", "value_IEEE_budget_passed", "slope_propagated_budget_passed"))
        report.update(passed=passed, status="passed" if passed else "rejected")
    except UnsupportedRoundtripArithmetic as error:
        report.update(status="unsupported", reason=str(error))
    except ValueError as error:
        report.update(reason=str(error))  # Preserve original source support rejection.
    return report


__all__ = ["value_roundtrip_budget", "audit_normalizer_roundtrip", "UnsupportedRoundtripArithmetic"]
