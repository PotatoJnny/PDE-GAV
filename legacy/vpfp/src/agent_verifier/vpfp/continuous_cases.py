from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from agent_verifier.vpfp.analytic_source import (
    ANALYTIC_SOURCE_BACKEND,
    IEEE_COORDINATE_MATMUL_PRECISION,
    ieee_float32_coordinate_matmul,
    manufacture_standard_vpfp_cartesian_source_analytic,
    manufacture_standard_vpfp_source_analytic,
    prepare_standard_vpfp_cartesian_velocity_jet,
    resolve_structured_source_backend,
)
from agent_verifier.vpfp.continuous_training import (
    load_continuous_autoencoder_checkpoint,
)
from agent_verifier.vpfp.continuous_macros import (
    MACRO_OUTPUT_CONTRACT,
    MACRO_TRUTH_SCHEMA,
    evaluate_continuous_hermite_macro_grid,
    is_continuous_macro_question,
)
from agent_verifier.vpfp.continuous_v2 import (
    STANDARD_VPFP_EQUATION_VERSION,
    manufacture_standard_vpfp_source,
)
from agent_verifier.vpfp.io import VPFPCase, load_case, write_case
from agent_verifier.vpfp.operators import (
    EPS,
    VPFPGrid,
    VPFPTime,
    make_axes,
    poisson_electric_field,
    relative_l2,
)


CONTINUOUS_PRIVATE_STATE = "continuous_private_state.npz"
CONTINUOUS_PRIMARY_SCHEMA = "vpfp_continuous_primary_truth_v1"
CONTINUOUS_CASE_SCHEMA = "vpfp_continuous_materialized_case_v1"


def is_standard_continuous_question(question: dict[str, Any]) -> bool:
    return str(question.get("equation_version") or question.get("equation")) == (
        STANDARD_VPFP_EQUATION_VERSION
    )


def materialize_continuous_case(
    case_dir: str | Path,
    *,
    case_id: str,
    model,
    decoder_checkpoint: str | Path,
    latent,
    grid: VPFPGrid,
    time: VPFPTime,
    micro_latent=None,
    equation_condition=None,
    source_origin: str = "generated_continuous_autodiff_residual",
    solver_output_contract: str = "full_state",
    generator_metadata: dict[str, Any] | None = None,
    case_spec: dict[str, Any] | None = None,
    evaluation_chunk_size: int = 2048,
    source_derivative_backend: str = "auto",
) -> Path:
    """Cache a finite evaluation view of a continuous answer-first VPFP case.

    The decoder checkpoint plus latent remains the primary hidden object. The
    arrays written by this function are explicitly a controller-selected query
    view used by the current solver ABI and verifier.
    """

    grid.validate()
    time.validate()
    if int(evaluation_chunk_size) < 1:
        raise ValueError("evaluation_chunk_size must be positive")
    if solver_output_contract not in {"full_state", MACRO_OUTPUT_CONTRACT}:
        raise ValueError("unsupported continuous solver output contract")
    macro_output = solver_output_contract == MACRO_OUTPUT_CONTRACT
    config = model.config
    condition_array = _validated_equation_condition(equation_condition)
    collision_nu = (
        float(config.equation.collision_nu)
        if condition_array is None
        else float(condition_array[0])
    )
    temperature = (
        float(config.equation.temperature)
        if condition_array is None
        else float(condition_array[1])
    )
    if tuple(float(value) for value in config.equation.x_domain) != tuple(
        float(value) for value in grid.x_domain
    ):
        raise ValueError("materialized x domain does not match the continuous model")
    if not np.allclose(
        [time.t0, time.t1], config.time_domain, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("materialized time domain does not match the continuous model")
    minimum_nx = 4 * int(config.rho_max_mode_x) + 1
    minimum_ny = 4 * int(config.rho_max_mode_y) + 1
    if grid.nx < minimum_nx or grid.ny < minimum_ny:
        raise ValueError(
            "materialized spatial grid aliases the squared rho Fourier support: "
            f"need at least ({minimum_nx}, {minimum_ny}), "
            f"got ({grid.nx}, {grid.ny})"
        )

    axes = make_axes(grid, time)
    mesh_shape = (time.nt, grid.nx, grid.ny, grid.nvx, grid.nvy)
    model.eval()
    resolved_source_backend = resolve_structured_source_backend(
        model,
        micro_latent=micro_latent,
        requested=source_derivative_backend,
    )
    if resolved_source_backend == ANALYTIC_SOURCE_BACKEND:
        with ieee_float32_coordinate_matmul(model):
            rho, g, source, phi, electric = _materialize_analytic_cartesian(
                model,
                latent,
                axes=axes,
                mesh_shape=mesh_shape,
                equation_condition=condition_array,
                evaluation_chunk_size=int(evaluation_chunk_size),
            )
    else:
        point_count = int(np.prod(mesh_shape))
        rho_flat = np.empty(point_count, dtype=float)
        g_flat = np.empty(point_count, dtype=float)
        source_flat = np.empty(point_count, dtype=float)
        phi_flat = np.empty(point_count, dtype=float)
        electric_flat = np.empty((point_count, 2), dtype=float)
        for start, stop, points in _mesh_point_chunks(
            axes, mesh_shape, int(evaluation_chunk_size)
        ):
            values = manufacture_standard_vpfp_source(
                model,
                points,
                latent,
                micro_latent=micro_latent,
                equation_condition=condition_array,
                create_graph=False,
            )
            rho_flat[start:stop] = _numpy(values["rho"])
            g_flat[start:stop] = _numpy(values["g"])
            source_flat[start:stop] = _numpy(values["source"])
            phi_flat[start:stop] = _numpy(values["phi"])
            electric_flat[start:stop] = _numpy(values["electric"])
        rho = rho_flat.reshape(mesh_shape)[..., 0, 0]
        g = g_flat.reshape(mesh_shape)
        source = source_flat.reshape(mesh_shape)
        phi = phi_flat.reshape(mesh_shape)[..., 0, 0]
        electric = electric_flat.reshape((*mesh_shape, 2))[..., 0, 0, :]
        electric = np.moveaxis(electric, -1, 1)
    weights = trapezoidal_velocity_weights(grid)
    window_mass = np.sum(g * weights, axis=(-2, -1))
    macro_truth = None
    if macro_output:
        with ieee_float32_coordinate_matmul(model):
            macro_truth = evaluate_continuous_hermite_macro_grid(
                model,
                latent,
                grid=grid,
                time=time,
                micro_latent=micro_latent,
                equation_condition=condition_array,
                chunk_size=int(evaluation_chunk_size),
            )
        if not np.allclose(
            macro_truth["rho"], rho, rtol=2.0e-5, atol=2.0e-6
        ):
            raise RuntimeError("macro rho query disagrees with materialized rho")
    checkpoint = Path(decoder_checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_sha = _sha256(checkpoint)
    metadata = {
        "schema": CONTINUOUS_CASE_SCHEMA,
        "backend": "continuous_neural_field",
        "representation": str(config.representation),
        **dict(generator_metadata or {}),
        "source_derivative_backend": resolved_source_backend,
        "coordinate_matmul_precision": (
            IEEE_COORDINATE_MATMUL_PRECISION
            if resolved_source_backend == ANALYTIC_SOURCE_BACKEND
            else "caller_default"
        ),
    }
    spec = {
        "schema": CONTINUOUS_CASE_SCHEMA,
        "source_origin": str(source_origin),
        "mathematical_velocity_domain": "R^2",
        "materialized_velocity_window": [float(value) for value in grid.v_bounds],
        "materialized_view_is_primary_truth": False,
        "decoder_checkpoint_sha256": checkpoint_sha,
        "solver_output_contract_version": solver_output_contract,
        **dict(case_spec or {}),
        "source_derivative_backend": resolved_source_backend,
        "source_materialization_backend": (
            "factorized_cartesian"
            if resolved_source_backend == ANALYTIC_SOURCE_BACKEND
            else "pointwise"
        ),
        "coordinate_matmul_precision": (
            IEEE_COORDINATE_MATMUL_PRECISION
            if resolved_source_backend == ANALYTIC_SOURCE_BACKEND
            else "caller_default"
        ),
    }
    diagnostics = {
        "all_finite": bool(
            all(np.all(np.isfinite(value)) for value in (rho, g, source, phi, electric))
        ),
        "rho_min": float(np.min(rho)),
        "g_min": float(np.min(g)),
        "source_rms": float(np.sqrt(np.mean(source**2))),
        "g_fullspace_normalization": "analytic_by_continuous_representation",
        "g_window_mass_min": float(np.min(window_mass)),
        "g_window_mass_max": float(np.max(window_mass)),
        "coordinate_matmul_precision": (
            IEEE_COORDINATE_MATMUL_PRECISION
            if resolved_source_backend == ANALYTIC_SOURCE_BACKEND
            else "caller_default"
        ),
    }
    if macro_truth is not None:
        diagnostics.update(
            {
                "macro_truth_schema": MACRO_TRUTH_SCHEMA,
                "macro_truth_fullspace_exact_from_hermite_coefficients": True,
            }
        )
    output = Path(case_dir)
    case = VPFPCase(
        case_id=str(case_id),
        grid=grid,
        time=time,
        collision_nu=collision_nu,
        rho=rho,
        g=g,
        source=source,
        phi=phi,
        electric=electric,
        diagnostics=diagnostics,
        case_spec=spec,
        generator_metadata=metadata,
        source_origin=str(source_origin),
    )
    write_case(output, case)

    latent_array = _numpy_array(latent)
    micro_array = (
        np.empty((0,), dtype=float)
        if micro_latent is None
        else _numpy_array(micro_latent)
    )
    np.savez_compressed(
        output / CONTINUOUS_PRIVATE_STATE,
        latent=latent_array,
        micro_latent=micro_array,
        has_micro_latent=np.asarray(micro_latent is not None, dtype=np.bool_),
        equation_condition=(
            np.empty((0,), dtype=float)
            if condition_array is None
            else condition_array
        ),
        has_equation_condition=np.asarray(
            condition_array is not None, dtype=np.bool_
        ),
    )
    with np.load(output / "public_arrays.npz") as payload:
        public_arrays = {name: np.asarray(payload[name]) for name in payload.files}
    public_arrays["velocity_weights"] = weights
    if macro_truth is not None:
        public_arrays["momentum0"] = macro_truth["momentum"][0]
        public_arrays["kinetic_energy0"] = macro_truth["kinetic_energy"][0]
    np.savez_compressed(output / "public_arrays.npz", **public_arrays)

    if macro_truth is not None:
        with np.load(output / "hidden_truth_cache.npz") as payload:
            hidden_arrays = {
                name: np.asarray(payload[name]) for name in payload.files
            }
        hidden_arrays.update(
            {
                "momentum_true": macro_truth["momentum"],
                "kinetic_energy_true": macro_truth["kinetic_energy"],
                "raw_second_moment_true": macro_truth["raw_second_moment"],
            }
        )
        np.savez_compressed(output / "hidden_truth_cache.npz", **hidden_arrays)

    public_question = _read_json(output / "public_question.json")
    public_question.update(
        {
            "equation": STANDARD_VPFP_EQUATION_VERSION,
            "equation_version": STANDARD_VPFP_EQUATION_VERSION,
            "task": (
                "predict full-space macroscopic rho, momentum, and kinetic energy"
                if macro_output
                else "predict a materialized evaluation view of continuous rho(t,x,y) and g(t,x,y,vx,vy)"
            ),
            "factorization": "f(t,x,v)=rho(t,x)*g(t,v|x), with integral over R^2_v of g equal to one",
            "temperature": temperature,
            "background_density": float(config.equation.background_density),
            "domain": model.public_equation_metadata()["domain"],
            "collision_operator": model.public_equation_metadata()["collision_operator"],
            "normalization": model.public_equation_metadata()["normalization"],
            "poisson": {
                "equation": "-Delta_x phi = rho - background_density",
                "electric_field": "E = -grad_x phi",
                "x_boundary": "periodic",
                "implementation": "Fourier multiplier on the spatial torus",
            },
            "velocity_boundary": "none_in_mathematical_problem; v_bounds is an evaluation query window",
            "materialization": {
                "role": "controller_selected_evaluation_view_of_primary_continuous_truth",
                "velocity_window_is_boundary_condition": False,
                "agent_internal_grid_may_be_finer": True,
                "submission_is_evaluated_on_declared_view": True,
            },
            "basis": None,
            "solver_contract": (
                "return {'rho': rho_pred, 'momentum': momentum_pred, "
                "'kinetic_energy': kinetic_energy_pred} on target_shapes"
                if macro_output
                else "return {'rho': rho_pred, 'g': g_pred} on target_shapes; basis_coeffs are not supported for this continuous case"
            ),
        }
    )
    if macro_truth is not None:
        rho_shape = list(macro_truth["rho"].shape)
        public_question.update(
            {
                "solver_output_contract_version": MACRO_OUTPUT_CONTRACT,
                "target_shapes": {
                    "rho": rho_shape,
                    "momentum": [*rho_shape, 2],
                    "kinetic_energy": rho_shape,
                },
                "macroscopic_quantities": {
                    "rho": "integral_R2 f dv",
                    "momentum": "integral_R2 v f dv; final axis is (x,y)",
                    "kinetic_energy": "0.5*integral_R2 |v|^2 f dv",
                    "velocity_domain": "R^2",
                },
            }
        )
        public_question["arrays"].update(
            {
                "momentum0": "public_arrays.npz:momentum0",
                "kinetic_energy0": "public_arrays.npz:kinetic_energy0",
            }
        )
    _write_json(output / "public_question.json", public_question)

    state_sha = _sha256(output / CONTINUOUS_PRIVATE_STATE)
    cache_sha = _sha256(output / "hidden_truth_cache.npz")
    public_arrays_sha = _sha256(output / "public_arrays.npz")
    hidden_manifest = _read_json(output / "hidden_truth_manifest.json")
    if macro_truth is not None:
        hidden_manifest["truth_type"] = (
            "continuous_vpfp_answer_with_exact_fullspace_macro_cache"
        )
        hidden_manifest["hidden_arrays"].update(
            {
                "momentum_true": "hidden_truth_cache.npz:momentum_true",
                "kinetic_energy_true": "hidden_truth_cache.npz:kinetic_energy_true",
                "raw_second_moment_true": "hidden_truth_cache.npz:raw_second_moment_true",
            }
        )
    hidden_manifest.update(
        {
            "truth_type": (
                "continuous_vpfp_answer_with_exact_fullspace_macro_cache"
                if macro_truth is not None
                else "continuous_vpfp_answer_with_materialized_cache"
            ),
            "continuous_primary": {
                "schema": CONTINUOUS_PRIMARY_SCHEMA,
                "decoder_checkpoint": str(checkpoint),
                "decoder_checkpoint_sha256": checkpoint_sha,
                "private_state": CONTINUOUS_PRIVATE_STATE,
                "private_state_sha256": state_sha,
                "materialized_cache_sha256": cache_sha,
                "public_arrays_sha256": public_arrays_sha,
                "materialization_evaluation_chunk_size": int(
                    evaluation_chunk_size
                ),
                "source_derivative_backend": resolved_source_backend,
                "coordinate_matmul_precision": (
                    IEEE_COORDINATE_MATMUL_PRECISION
                    if resolved_source_backend == ANALYTIC_SOURCE_BACKEND
                    else "caller_default"
                ),
                "equation_condition": (
                    None
                    if condition_array is None
                    else condition_array.tolist()
                ),
                "primary_object": "decoder_checkpoint_plus_latent_plus_condition",
                "cache_role": "controller_selected_evaluation_view",
                "macro_truth": (
                    None
                    if macro_truth is None
                    else {
                        "schema": MACRO_TRUTH_SCHEMA,
                        "method": "exact_fullspace_squared_hermite_moments",
                        "solver_output_contract_version": MACRO_OUTPUT_CONTRACT,
                    }
                ),
            },
        }
    )
    _write_json(output / "hidden_truth_manifest.json", hidden_manifest)
    return output


def verify_continuous_case_dir(
    case_dir: str | Path,
    *,
    residual_tol: float = 1.0e-8,
    normalization_tol: float = 5.0e-6,
    require_residual: bool | None = None,
    device: str = "cpu",
    truth_sample_count: int = 512,
    source_sample_count: int = 64,
) -> dict[str, Any]:
    case = load_case(case_dir)
    if _is_equation_first_numeric_reference(case):
        return _verify_equation_first_numeric_reference(
            case,
            normalization_tol=float(normalization_tol),
        )
    del residual_tol, normalization_tol, require_residual
    failures: list[str] = []
    try:
        (
            model,
            latent,
            micro_latent,
            equation_condition,
            manifest,
        ) = load_continuous_primary_state(case_dir, device=device)
    except Exception as exc:  # noqa: BLE001 - audit reports exact integrity failure.
        return {
            "accepted": False,
            "case_id": case["case_id"],
            "metrics": {},
            "failures": [f"continuous_primary_integrity:{type(exc).__name__}:{exc}"],
        }

    public = case["public_arrays"]
    hidden = case["hidden_arrays"]
    rho = np.asarray(hidden["rho_true"], dtype=float)
    g = np.asarray(hidden["g_true"], dtype=float)
    source = np.asarray(public["source"], dtype=float)
    expected_rho = (case["time"].nt, case["grid"].nx, case["grid"].ny)
    expected_g = expected_rho + (case["grid"].nvx, case["grid"].nvy)
    if rho.shape != expected_rho or g.shape != expected_g or source.shape != expected_g:
        failures.append("shape")
    if not all(np.all(np.isfinite(value)) for value in (rho, g, source)):
        failures.append("nonfinite")
    if np.min(rho) < -1.0e-10:
        failures.append("rho_negative")
    if np.min(g) < -1.0e-10:
        failures.append("g_negative")

    macro_case = is_continuous_macro_question(case["public_question"])
    if macro_case:
        expected_momentum = (*expected_rho, 2)
        required_macro = {
            "momentum_true": expected_momentum,
            "kinetic_energy_true": expected_rho,
            "raw_second_moment_true": (*expected_rho, 3),
        }
        for name, expected_shape in required_macro.items():
            value = hidden.get(name)
            if value is None or tuple(np.asarray(value).shape) != expected_shape:
                failures.append(f"{name}_shape")
            elif not np.all(np.isfinite(value)):
                failures.append(f"{name}_nonfinite")
        momentum_truth = hidden.get("momentum_true")
        if (
            momentum_truth is None
            or public.get("momentum0") is None
            or not np.array_equal(public["momentum0"], momentum_truth[0])
        ):
            failures.append("momentum0_mismatch")
        kinetic_truth = hidden.get("kinetic_energy_true")
        if (
            kinetic_truth is None
            or public.get("kinetic_energy0") is None
            or not np.array_equal(
                public["kinetic_energy0"], kinetic_truth[0]
            )
        ):
            failures.append("kinetic_energy0_mismatch")

    axes = case["public_arrays"]
    mesh_shape = expected_g
    point_count = int(np.prod(mesh_shape))
    truth_indices = _audit_indices(point_count, truth_sample_count)
    truth_points = _points_from_flat_indices(axes, mesh_shape, truth_indices)
    import torch

    primary = manifest["continuous_primary"]
    spec_source_backend = str(
        case.get("case_spec", {}).get("source_derivative_backend") or ""
    )
    source_backend = str(
        primary.get("source_derivative_backend") or spec_source_backend
    )
    coordinate_precision = str(
        primary.get("coordinate_matmul_precision") or "caller_default"
    )
    if primary.get("source_derivative_backend") and (
        source_backend != spec_source_backend
    ):
        failures.append("source_backend_metadata_mismatch")
    field_kwargs = {"micro_latent": micro_latent}
    if equation_condition is not None:
        field_kwargs["equation_condition"] = equation_condition
    if coordinate_precision == IEEE_COORDINATE_MATMUL_PRECISION:
        with ieee_float32_coordinate_matmul(model), torch.no_grad():
            truth = model.fields(truth_points, latent, **field_kwargs)
    else:
        with torch.no_grad():
            truth = model.fields(truth_points, latent, **field_kwargs)
    if str(model.config.dtype) == "float32":
        cache_rtol = 2.0e-5
        cache_atol = 2.0e-6
        source_rtol = 2.0e-5
        source_atol = 2.0e-6
    else:
        cache_rtol = 5.0e-7
        cache_atol = 1.0e-9
        source_rtol = 2.0e-6
        source_atol = 2.0e-8
    initial_chunk_size = int(
        primary.get("materialization_evaluation_chunk_size") or 2048
    )
    primary_rho0, primary_g0 = _materialize_primary_initial_answer(
        model,
        latent,
        axes=axes,
        mesh_shape=mesh_shape,
        micro_latent=micro_latent,
        equation_condition=equation_condition,
        source_backend=source_backend,
        coordinate_precision=coordinate_precision,
        evaluation_chunk_size=initial_chunk_size,
    )
    public_rho0 = public.get("rho0")
    public_g0 = public.get("g0")
    rho0_cache_exact = False
    g0_cache_exact = False
    rho0_primary_max_abs = float("inf")
    g0_primary_max_abs = float("inf")
    if (
        public_rho0 is None
        or tuple(np.asarray(public_rho0).shape) != expected_rho[1:]
        or not np.all(np.isfinite(public_rho0))
    ):
        failures.append("rho0_shape_or_nonfinite")
    else:
        public_rho0 = np.asarray(public_rho0, dtype=float)
        rho0_cache_exact = bool(np.array_equal(public_rho0, rho[0]))
        rho0_primary_max_abs = float(
            np.max(np.abs(public_rho0 - primary_rho0))
        )
        if not rho0_cache_exact:
            failures.append("rho0_materialized_cache_mismatch")
        if not np.allclose(
            public_rho0, primary_rho0, rtol=cache_rtol, atol=cache_atol
        ):
            failures.append("rho0_primary_answer_mismatch")
    if (
        public_g0 is None
        or tuple(np.asarray(public_g0).shape) != expected_g[1:]
        or not np.all(np.isfinite(public_g0))
    ):
        failures.append("g0_shape_or_nonfinite")
    else:
        public_g0 = np.asarray(public_g0, dtype=float)
        g0_cache_exact = bool(np.array_equal(public_g0, g[0]))
        g0_primary_max_abs = float(np.max(np.abs(public_g0 - primary_g0)))
        if not g0_cache_exact:
            failures.append("g0_materialized_cache_mismatch")
        if not np.allclose(
            public_g0, primary_g0, rtol=cache_rtol, atol=cache_atol
        ):
            failures.append("g0_primary_answer_mismatch")

    index_tuple = np.unravel_index(truth_indices, mesh_shape)
    cached_values = {
        "rho": rho[index_tuple[:3]],
        "g": g[index_tuple],
        "phi": np.asarray(hidden["phi_true"])[index_tuple[:3]],
        "electric": np.asarray(hidden["E_true"])[
            index_tuple[0], :, index_tuple[1], index_tuple[2]
        ],
    }
    truth_cache_max_abs = 0.0
    for name, cached in cached_values.items():
        represented = _numpy(truth[name])
        truth_cache_max_abs = max(
            truth_cache_max_abs,
            float(np.max(np.abs(represented - np.asarray(cached)))),
        )
        if not np.allclose(represented, cached, rtol=cache_rtol, atol=cache_atol):
            failures.append(f"materialized_{name}_mismatch")

    source_indices = _audit_indices(point_count, source_sample_count)
    source_points = _points_from_flat_indices(axes, mesh_shape, source_indices)
    if source_backend == ANALYTIC_SOURCE_BACKEND:
        if coordinate_precision == IEEE_COORDINATE_MATMUL_PRECISION:
            with ieee_float32_coordinate_matmul(model):
                represented_source = manufacture_standard_vpfp_source_analytic(
                    model,
                    source_points,
                    latent,
                    micro_latent=micro_latent,
                    equation_condition=equation_condition,
                )["source"]
        else:
            represented_source = manufacture_standard_vpfp_source_analytic(
                model,
                source_points,
                latent,
                micro_latent=micro_latent,
                equation_condition=equation_condition,
            )["source"]
    else:
        represented_source = manufacture_standard_vpfp_source(
            model,
            source_points,
            latent,
            micro_latent=micro_latent,
            equation_condition=equation_condition,
            create_graph=False,
        )["source"]
    cached_source = source[np.unravel_index(source_indices, mesh_shape)]
    source_cache_max_abs = float(
        np.max(np.abs(_numpy(represented_source) - cached_source))
    )
    if not np.allclose(
        _numpy(represented_source), cached_source, rtol=source_rtol, atol=source_atol
    ):
        failures.append("materialized_source_mismatch")

    truth_residual = standard_ou_residual(
        rho,
        g,
        source,
        grid=case["grid"],
        time=case["time"],
        collision_nu=case["collision_nu"],
        temperature=float(case["public_question"]["temperature"]),
    )
    weights = np.asarray(public["velocity_weights"], dtype=float)
    window_mass = np.sum(g * weights, axis=(-2, -1))
    metrics = {
        "primary_schema": manifest["continuous_primary"]["schema"],
        "truth_vpfp_residual_rel": relative_l2(truth_residual, source),
        "finite_difference_residual_is_hard_gate": False,
        "truth_cache_max_abs_error": truth_cache_max_abs,
        "truth_cache_rtol": cache_rtol,
        "truth_cache_atol": cache_atol,
        "source_cache_max_abs_error": source_cache_max_abs,
        "source_cache_rtol": source_rtol,
        "source_cache_atol": source_atol,
        "source_cache_verification_backend": (
            source_backend
            if source_backend == ANALYTIC_SOURCE_BACKEND
            else "torch_nested_coordinate_autodiff_v1"
        ),
        "coordinate_matmul_precision": coordinate_precision,
        "public_arrays_sha256_verified": "public_arrays_sha256" in primary,
        "answer_first_initial_data_rederived_from_primary": True,
        "rho0_materialized_cache_exact": rho0_cache_exact,
        "g0_materialized_cache_exact": g0_cache_exact,
        "rho0_primary_answer_max_abs_error": rho0_primary_max_abs,
        "g0_primary_answer_max_abs_error": g0_primary_max_abs,
        "g_normalization_contract": "analytic_fullspace_not_materialized_window",
        "g_window_mass_min": float(np.min(window_mass)),
        "g_window_mass_max": float(np.max(window_mass)),
        "rho_min": float(np.min(rho)),
        "g_min": float(np.min(g)),
        "source_rms": float(np.sqrt(np.mean(source**2))),
        "source_origin": str(case["public_question"].get("source_origin", "")),
        "macro_output_contract": bool(macro_case),
        "macro_truth_fullspace_exact_from_hermite_coefficients": bool(macro_case),
    }
    return {
        "accepted": not failures,
        "metrics": metrics,
        "failures": sorted(set(failures)),
        "case_id": case["case_id"],
    }


def _verify_equation_first_numeric_reference(
    case: dict[str, Any],
    *,
    normalization_tol: float,
) -> dict[str, Any]:
    failures: list[str] = []
    public = case["public_arrays"]
    hidden = case["hidden_arrays"]
    rho = np.asarray(hidden["rho_true"], dtype=float)
    g = np.asarray(hidden["g_true"], dtype=float)
    source = np.asarray(public["source"], dtype=float)
    expected_rho = (case["time"].nt, case["grid"].nx, case["grid"].ny)
    expected_g = expected_rho + (case["grid"].nvx, case["grid"].nvy)
    shapes_valid = (
        rho.shape == expected_rho
        and g.shape == expected_g
        and source.shape == expected_g
    )
    if not shapes_valid:
        failures.append("shape")
    finite = all(np.all(np.isfinite(value)) for value in (rho, g, source))
    if not finite:
        failures.append("nonfinite")
    weights = np.asarray(public.get("velocity_weights"), dtype=float)
    if weights.shape != (case["grid"].nvx, case["grid"].nvy):
        failures.append("velocity_weights_shape")
        window_mass = np.asarray([np.nan])
    elif shapes_valid and finite:
        window_mass = np.sum(g * weights, axis=(-2, -1))
        if float(np.max(np.abs(window_mass - 1.0))) > normalization_tol:
            failures.append("g_window_normalization")
    else:
        window_mass = np.asarray([np.nan])
    if finite:
        if float(np.min(rho)) < -1.0e-10:
            failures.append("rho_negative")
        if float(np.min(g)) < -1.0e-10:
            failures.append("g_negative")

    macro_case = is_continuous_macro_question(case["public_question"])
    if macro_case:
        expected_macro = {
            "momentum_true": (*expected_rho, 2),
            "kinetic_energy_true": expected_rho,
            "raw_second_moment_true": (*expected_rho, 3),
        }
        for name, expected_shape in expected_macro.items():
            value = hidden.get(name)
            if value is None or tuple(np.asarray(value).shape) != expected_shape:
                failures.append(f"{name}_shape")
            elif not np.all(np.isfinite(value)):
                failures.append(f"{name}_nonfinite")
        momentum_truth = hidden.get("momentum_true")
        if (
            momentum_truth is None
            or public.get("momentum0") is None
            or not np.array_equal(public["momentum0"], momentum_truth[0])
        ):
            failures.append("momentum0_mismatch")
        kinetic_truth = hidden.get("kinetic_energy_true")
        if (
            kinetic_truth is None
            or public.get("kinetic_energy0") is None
            or not np.array_equal(
                public["kinetic_energy0"], kinetic_truth[0]
            )
        ):
            failures.append("kinetic_energy0_mismatch")

    if shapes_valid and finite:
        truth_residual = standard_ou_residual(
            rho,
            g,
            source,
            grid=case["grid"],
            time=case["time"],
            collision_nu=case["collision_nu"],
            temperature=float(case["public_question"]["temperature"]),
        )
        residual_rel = relative_l2(truth_residual, source)
    else:
        residual_rel = float("inf")
    metrics = {
        "primary_schema": "vpfp_standard_ou_equation_first_reference_v1",
        "truth_vpfp_residual_rel": float(residual_rel),
        "finite_difference_residual_is_hard_gate": False,
        "g_normalization_contract": "declared_numerical_velocity_quadrature",
        "g_window_mass_min": float(np.nanmin(window_mass)),
        "g_window_mass_max": float(np.nanmax(window_mass)),
        "rho_min": float(np.min(rho)) if finite else float("nan"),
        "g_min": float(np.min(g)) if finite else float("nan"),
        "source_rms": (
            float(np.sqrt(np.mean(source**2))) if finite else float("nan")
        ),
        "source_origin": str(case["public_question"].get("source_origin", "")),
        "macro_output_contract": bool(macro_case),
        "macro_truth_fullspace_exact_from_hermite_coefficients": False,
        "macro_truth_reference_velocity_quadrature": bool(macro_case),
    }
    return {
        "accepted": not failures,
        "metrics": metrics,
        "failures": sorted(set(failures)),
        "case_id": case["case_id"],
    }


def evaluate_continuous_macro_prediction(
    case_dir: str | Path, output: dict[str, Any]
) -> dict[str, Any]:
    case = load_case(case_dir)
    if not is_continuous_macro_question(case["public_question"]):
        return _failed_macro(case["case_id"], "not_macro_output_contract")
    try:
        if _is_equation_first_numeric_reference(case):
            _assert_equation_first_numeric_reference_integrity(case_dir)
        else:
            _assert_primary_integrity(case_dir)
    except Exception as exc:  # noqa: BLE001
        return _failed_macro(
            case["case_id"], f"private_case_integrity:{type(exc).__name__}"
        )
    status, rho_pred, momentum_pred, kinetic_pred = (
        _coerce_continuous_macro_output(case, output)
    )
    if (
        status != "ok"
        or rho_pred is None
        or momentum_pred is None
        or kinetic_pred is None
    ):
        return _failed_macro(case["case_id"], status)

    hidden = case["hidden_arrays"]
    rho_true = np.asarray(hidden["rho_true"], dtype=float)
    momentum_true = np.asarray(hidden["momentum_true"], dtype=float)
    kinetic_true = np.asarray(hidden["kinetic_energy_true"], dtype=float)
    _phi_pred, electric_pred = poisson_electric_field(rho_pred, case["grid"])
    electric_true = np.asarray(hidden["E_true"], dtype=float)
    rho_l2 = relative_l2(rho_pred - rho_true, rho_true)
    momentum_l2 = relative_l2(
        momentum_pred - momentum_true, momentum_true
    )
    kinetic_l2 = relative_l2(kinetic_pred - kinetic_true, kinetic_true)
    field_l2 = relative_l2(electric_pred - electric_true, electric_true)
    rho_negative = relative_l2(np.minimum(rho_pred, 0.0), rho_pred)
    kinetic_negative = relative_l2(
        np.minimum(kinetic_pred, 0.0), kinetic_pred
    )
    rho_initial = relative_l2(rho_pred[0] - rho_true[0], rho_true[0])
    momentum_initial = relative_l2(
        momentum_pred[0] - momentum_true[0], momentum_true[0]
    )
    kinetic_initial = relative_l2(
        kinetic_pred[0] - kinetic_true[0], kinetic_true[0]
    )
    score = (
        (rho_l2 + momentum_l2 + kinetic_l2) / 3.0
        + rho_negative
        + kinetic_negative
    )
    return {
        "case_id": case["case_id"],
        "agent_status": "ok",
        "equation_version": STANDARD_VPFP_EQUATION_VERSION,
        "solver_output_contract_version": MACRO_OUTPUT_CONTRACT,
        "rho_rel_l2": float(rho_l2),
        "momentum_density_rel_l2": float(momentum_l2),
        "kinetic_energy_density_rel_l2": float(kinetic_l2),
        "electric_field_rel_l2": float(field_l2),
        "rho_initial_rel_l2": float(rho_initial),
        "momentum_initial_rel_l2": float(momentum_initial),
        "kinetic_energy_initial_rel_l2": float(kinetic_initial),
        "rho_positivity_violation": float(rho_negative),
        "kinetic_energy_positivity_violation": float(kinetic_negative),
        "score_smaller_is_better": float(score),
        "adversarial_score_larger_is_harder": float(score),
    }


def _coerce_continuous_macro_output(case: dict[str, Any], output):
    if not isinstance(output, dict):
        return "output_not_dict", None, None, None
    rho_value = output.get("rho", output.get("rho_pred"))
    momentum_value = output.get("momentum", output.get("j"))
    kinetic_value = output.get("kinetic_energy", output.get("K"))
    if rho_value is None or momentum_value is None or kinetic_value is None:
        return "missing_macro_fields", None, None, None
    rho = np.asarray(rho_value, dtype=float)
    momentum = np.asarray(momentum_value, dtype=float)
    kinetic = np.asarray(kinetic_value, dtype=float)
    expected_rho = tuple(case["hidden_arrays"]["rho_true"].shape)
    if (
        rho.shape != expected_rho
        or momentum.shape != (*expected_rho, 2)
        or kinetic.shape != expected_rho
    ):
        return "bad_shape", None, None, None
    if not all(np.all(np.isfinite(value)) for value in (rho, momentum, kinetic)):
        return "nan", None, None, None
    return "ok", rho, momentum, kinetic


def _failed_macro(case_id: str, status: str) -> dict[str, Any]:
    keys = (
        "rho_rel_l2",
        "momentum_density_rel_l2",
        "kinetic_energy_density_rel_l2",
        "electric_field_rel_l2",
        "score_smaller_is_better",
        "adversarial_score_larger_is_harder",
    )
    return {
        "case_id": case_id,
        "agent_status": status,
        **{key: float("inf") for key in keys},
    }


def evaluate_continuous_prediction(
    case_dir: str | Path, output: dict[str, Any]
) -> dict[str, Any]:
    case = load_case(case_dir)
    if is_continuous_macro_question(case["public_question"]):
        return evaluate_continuous_macro_prediction(case_dir, output)
    try:
        if _is_equation_first_numeric_reference(case):
            _assert_equation_first_numeric_reference_integrity(case_dir)
        else:
            _assert_primary_integrity(case_dir)
    except Exception as exc:  # noqa: BLE001
        return _failed(case["case_id"], f"private_case_integrity:{type(exc).__name__}")
    status, rho_pred, g_pred = _coerce_continuous_output(case, output)
    if status != "ok" or rho_pred is None or g_pred is None:
        return _failed(case["case_id"], status)

    grid = case["grid"]
    time = case["time"]
    hidden = case["hidden_arrays"]
    public = case["public_arrays"]
    rho_true = np.asarray(hidden["rho_true"], dtype=float)
    g_true = np.asarray(hidden["g_true"], dtype=float)
    f_true = rho_true[..., None, None] * g_true
    f_pred = rho_pred[..., None, None] * g_pred
    _phi_pred, electric_pred = poisson_electric_field(rho_pred, grid)
    electric_true = np.asarray(hidden["E_true"], dtype=float)
    source = np.asarray(public["source"], dtype=float)
    temperature = float(case["public_question"]["temperature"])
    pred_residual = standard_ou_residual(
        rho_pred,
        g_pred,
        source,
        grid=grid,
        time=time,
        collision_nu=case["collision_nu"],
        temperature=temperature,
    )
    truth_residual = standard_ou_residual(
        rho_true,
        g_true,
        source,
        grid=grid,
        time=time,
        collision_nu=case["collision_nu"],
        temperature=temperature,
    )
    rho_l2 = relative_l2(rho_pred - rho_true, rho_true)
    g_l2 = relative_l2(g_pred - g_true, g_true)
    f_l2 = relative_l2(f_pred - f_true, f_true)
    field_l2 = relative_l2(electric_pred - electric_true, electric_true)
    residual_rel = relative_l2(pred_residual, source)
    reference_residual_rel = relative_l2(truth_residual, source)
    residual_excess_rel = relative_l2(pred_residual - truth_residual, source)
    weights = np.asarray(public["velocity_weights"], dtype=float)
    mass_pred = np.sum(g_pred * weights, axis=(-2, -1))
    mass_true = np.sum(g_true * weights, axis=(-2, -1))
    window_mass_error = relative_l2(mass_pred - mass_true, mass_true)
    window_mass_max_abs = float(np.max(np.abs(mass_pred - mass_true)))
    rho_negative = relative_l2(np.minimum(rho_pred, 0.0), rho_pred)
    g_negative = relative_l2(np.minimum(g_pred, 0.0), g_pred)
    kinetic = _kinetic_error_metrics(
        f_pred,
        f_true,
        grid=grid,
        weights=weights,
    )
    score = (
        f_l2
        + 0.25 * rho_l2
        + 0.10 * g_l2
        + 0.10 * field_l2
        + 0.10 * residual_excess_rel
        + 0.25 * window_mass_error
        + rho_negative
        + g_negative
    )
    return {
        "case_id": case["case_id"],
        "agent_status": "ok",
        "equation_version": STANDARD_VPFP_EQUATION_VERSION,
        "rho_rel_l2": float(rho_l2),
        "g_rel_l2": float(g_l2),
        "f_rel_l2": float(f_l2),
        "electric_field_rel_l2": float(field_l2),
        "vpfp_residual_rel": float(residual_rel),
        "reference_vpfp_residual_rel": float(reference_residual_rel),
        "vpfp_residual_excess_rel": float(residual_excess_rel),
        "g_window_mass_rel_l2": float(window_mass_error),
        "g_window_mass_max_abs_error": window_mass_max_abs,
        "g_normalization_contract": "analytic_fullspace_not_materialized_window",
        "rho_positivity_violation": float(rho_negative),
        "g_positivity_violation": float(g_negative),
        "score_smaller_is_better": float(score),
        "adversarial_score_larger_is_harder": float(score),
        **kinetic,
    }


def load_continuous_primary_state(case_dir: str | Path, *, device: str = "cpu"):
    case_path = Path(case_dir)
    manifest = _assert_primary_integrity(case_path)
    primary = manifest["continuous_primary"]
    checkpoint = _resolve_private_path(case_path, primary["decoder_checkpoint"])
    decoder = _load_continuous_decoder_checkpoint(checkpoint, device=device)
    with np.load(case_path / primary["private_state"]) as payload:
        latent = np.asarray(payload["latent"])
        has_micro = bool(np.asarray(payload["has_micro_latent"]).item())
        micro_latent = np.asarray(payload["micro_latent"]) if has_micro else None
        has_condition = (
            "has_equation_condition" in payload.files
            and bool(np.asarray(payload["has_equation_condition"]).item())
        )
        equation_condition = (
            np.asarray(payload["equation_condition"])
            if has_condition
            else None
        )
    if equation_condition is None and primary.get("equation_condition") is not None:
        equation_condition = _validated_equation_condition(
            primary["equation_condition"]
        )
    return decoder, latent, micro_latent, equation_condition, manifest


def _load_continuous_decoder_checkpoint(checkpoint: Path, *, device: str):
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    schema = payload.get("schema")
    del payload
    if schema == "vpfp_continuous_autoencoder_checkpoint_v1":
        loader = load_continuous_autoencoder_checkpoint
    elif schema in {
        "vpfp_continuous_flow_autoencoder_checkpoint_v1",
        "vpfp_continuous_hermite_autoencoder_checkpoint_v1",
    }:
        from agent_verifier.vpfp.continuous_flow_training import (
            load_flow_autoencoder_checkpoint,
            load_hermite_autoencoder_checkpoint,
        )

        loader = (
            load_flow_autoencoder_checkpoint
            if schema == "vpfp_continuous_flow_autoencoder_checkpoint_v1"
            else load_hermite_autoencoder_checkpoint
        )
    else:
        raise ValueError(f"unsupported continuous decoder schema: {schema!r}")
    _encoder, decoder, _payload = loader(checkpoint, device=device)
    return decoder


def _validated_equation_condition(value):
    if value is None:
        return None
    condition = np.asarray(value, dtype=float)
    if condition.shape != (2,) or not np.all(np.isfinite(condition)):
        raise ValueError("equation_condition must be a finite (collision_nu, temperature) pair")
    if np.any(condition <= 0.0):
        raise ValueError("collision_nu and temperature must be positive")
    return condition


def standard_ou_residual(
    rho: np.ndarray,
    g: np.ndarray,
    source: np.ndarray,
    *,
    grid: VPFPGrid,
    time: VPFPTime,
    collision_nu: float,
    temperature: float,
) -> np.ndarray:
    """Independent finite-grid diagnostic for the full-space OU VPFP PDE."""

    f = np.asarray(rho, dtype=float)[..., None, None] * np.asarray(g, dtype=float)
    _phi, electric = poisson_electric_field(rho, grid)
    edge_t = 2 if time.nt >= 3 else 1
    df_dt = np.gradient(f, time.dt, axis=0, edge_order=edge_t)
    df_dx = (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2.0 * grid.dx)
    df_dy = (np.roll(f, -1, axis=2) - np.roll(f, 1, axis=2)) / (2.0 * grid.dy)
    edge_vx = 2 if grid.nvx >= 3 else 1
    edge_vy = 2 if grid.nvy >= 3 else 1
    df_dvx = np.gradient(f, grid.dvx, axis=3, edge_order=edge_vx)
    df_dvy = np.gradient(f, grid.dvy, axis=4, edge_order=edge_vy)
    d2f_dvx2 = np.gradient(df_dvx, grid.dvx, axis=3, edge_order=edge_vx)
    d2f_dvy2 = np.gradient(df_dvy, grid.dvy, axis=4, edge_order=edge_vy)
    axes = make_axes(grid)
    vx = axes["vx"][None, None, None, :, None]
    vy = axes["vy"][None, None, None, None, :]
    field = (
        electric[:, 0, :, :, None, None] * df_dvx
        + electric[:, 1, :, :, None, None] * df_dvy
    )
    collision = 2.0 * f + vx * df_dvx + vy * df_dvy + float(temperature) * (
        d2f_dvx2 + d2f_dvy2
    )
    computed = (
        df_dt
        + vx * df_dx
        + vy * df_dy
        + field
        - float(collision_nu) * collision
    )
    return computed - np.asarray(source, dtype=float)


def trapezoidal_velocity_weights(grid: VPFPGrid) -> np.ndarray:
    wx = np.full(grid.nvx, grid.dvx, dtype=float)
    wy = np.full(grid.nvy, grid.dvy, dtype=float)
    wx[[0, -1]] *= 0.5
    wy[[0, -1]] *= 0.5
    return wx[:, None] * wy[None, :]


def _is_equation_first_numeric_reference(case: dict[str, Any]) -> bool:
    return bool(
        is_standard_continuous_question(case["public_question"])
        and case.get("case_spec", {}).get("schema")
        == "vpfp_standard_ou_equation_first_reference_v1"
        and case.get("case_spec", {}).get("truth_primary")
        == "equation_first_refined_numerical_solution"
        and case["public_question"].get("source_origin")
        == "randomized_public_equation"
    )


def _assert_equation_first_numeric_reference_integrity(
    case_dir: str | Path,
) -> None:
    path = Path(case_dir)
    case = load_case(path)
    if not _is_equation_first_numeric_reference(case):
        raise ValueError("case is not an equation-first standard-OU reference")
    if float(case["public_question"].get("temperature", 0.0)) <= 0.0:
        raise ValueError("equation-first OU reference has invalid temperature")
    required = (
        "public_question.json",
        "public_arrays.npz",
        "hidden_truth_cache.npz",
        "hidden_truth_manifest.json",
        "case_spec.json",
    )
    for name in required:
        if not (path / name).is_file():
            raise FileNotFoundError(path / name)


def _assert_primary_integrity(case_dir: str | Path) -> dict[str, Any]:
    path = Path(case_dir)
    question = _read_json(path / "public_question.json")
    if not is_standard_continuous_question(question):
        raise ValueError("case is not standard continuous VPFP")
    manifest = _read_json(path / "hidden_truth_manifest.json")
    primary = manifest.get("continuous_primary")
    if not isinstance(primary, dict) or primary.get("schema") != CONTINUOUS_PRIMARY_SCHEMA:
        raise ValueError("missing continuous primary truth manifest")
    checkpoint = _resolve_private_path(path, primary["decoder_checkpoint"])
    state = _resolve_private_path(path, primary["private_state"])
    cache = path / "hidden_truth_cache.npz"
    expected = {
        checkpoint: primary["decoder_checkpoint_sha256"],
        state: primary["private_state_sha256"],
        cache: primary["materialized_cache_sha256"],
    }
    public_arrays_sha = primary.get("public_arrays_sha256")
    if public_arrays_sha is not None:
        expected[path / "public_arrays.npz"] = public_arrays_sha
    for item, expected_sha in expected.items():
        if not item.is_file():
            raise FileNotFoundError(item)
    return manifest


def _coerce_continuous_output(case: dict[str, Any], output: dict[str, Any]):
    if not isinstance(output, dict):
        return "failed", None, None
    rho_value = output.get("rho", output.get("rho_pred"))
    g_value = output.get("g", output.get("g_pred"))
    if rho_value is None or g_value is None:
        return "missing_rho_or_g", None, None
    rho = np.asarray(rho_value, dtype=float)
    g = np.asarray(g_value, dtype=float)
    if rho.shape != tuple(case["hidden_arrays"]["rho_true"].shape):
        return "bad_shape", None, None
    if g.shape != tuple(case["hidden_arrays"]["g_true"].shape):
        return "bad_shape", None, None
    if not np.all(np.isfinite(rho)) or not np.all(np.isfinite(g)):
        return "nan", None, None
    return "ok", rho, g


def _kinetic_error_metrics(
    f_pred: np.ndarray,
    f_true: np.ndarray,
    *,
    grid: VPFPGrid,
    weights: np.ndarray,
) -> dict[str, float]:
    axes = make_axes(grid)
    vx = np.asarray(axes["vx"], dtype=float)[:, None]
    vy = np.asarray(axes["vy"], dtype=float)[None, :]
    radius_squared = vx**2 + vy**2
    vmax_x = max(abs(grid.v_bounds[0]), abs(grid.v_bounds[1]))
    vmax_y = max(abs(grid.v_bounds[2]), abs(grid.v_bounds[3]))
    tail = (np.abs(vx) >= 0.80 * vmax_x) | (np.abs(vy) >= 0.80 * vmax_y)

    def fields(values: np.ndarray) -> dict[str, np.ndarray]:
        density = np.sum(values * weights, axis=(-2, -1))
        mx = np.sum(values * vx * weights, axis=(-2, -1))
        my = np.sum(values * vy * weights, axis=(-2, -1))
        safe = np.maximum(density, EPS)
        xx = np.sum(values * vx**2 * weights, axis=(-2, -1)) - mx**2 / safe
        xy = np.sum(values * vx * vy * weights, axis=(-2, -1)) - mx * my / safe
        yy = np.sum(values * vy**2 * weights, axis=(-2, -1)) - my**2 / safe
        return {
            "momentum": np.stack((mx, my), axis=-1),
            "kinetic": 0.5 * np.sum(values * radius_squared * weights, axis=(-2, -1)),
            "pressure": np.stack((xx, xy, yy), axis=-1),
            "entropy": np.sum(
                np.maximum(values, EPS) * np.log(np.maximum(values, EPS)) * weights,
                axis=(-2, -1),
            ),
            "tail": np.sum(values * tail * weights, axis=(-2, -1)),
        }

    predicted = fields(np.asarray(f_pred, dtype=float))
    reference = fields(np.asarray(f_true, dtype=float))
    return {
        "momentum_density_rel_l2": relative_l2(
            predicted["momentum"] - reference["momentum"], reference["momentum"]
        ),
        "kinetic_energy_density_rel_l2": relative_l2(
            predicted["kinetic"] - reference["kinetic"], reference["kinetic"]
        ),
        "pressure_tensor_rel_l2": relative_l2(
            predicted["pressure"] - reference["pressure"], reference["pressure"]
        ),
        "entropy_density_rel_l2": relative_l2(
            predicted["entropy"] - reference["entropy"], reference["entropy"]
        ),
        "velocity_tail_mass_rel_l2": relative_l2(
            predicted["tail"] - reference["tail"], reference["tail"]
        ),
    }


def _failed(case_id: str, status: str) -> dict[str, Any]:
    keys = (
        "rho_rel_l2",
        "g_rel_l2",
        "f_rel_l2",
        "electric_field_rel_l2",
        "vpfp_residual_rel",
        "reference_vpfp_residual_rel",
        "vpfp_residual_excess_rel",
        "momentum_density_rel_l2",
        "kinetic_energy_density_rel_l2",
        "pressure_tensor_rel_l2",
        "entropy_density_rel_l2",
        "velocity_tail_mass_rel_l2",
        "score_smaller_is_better",
        "adversarial_score_larger_is_harder",
    )
    return {"case_id": case_id, "agent_status": status, **{key: float("inf") for key in keys}}


def _materialize_primary_initial_answer(
    model,
    latent,
    *,
    axes: dict[str, np.ndarray],
    mesh_shape: tuple[int, int, int, int, int],
    micro_latent,
    equation_condition,
    source_backend: str,
    coordinate_precision: str,
    evaluation_chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Re-evaluate public initial data from the primary hidden answer."""

    initial_axes = {
        name: np.asarray(value)
        for name, value in axes.items()
    }
    initial_axes["t"] = np.asarray(initial_axes["t"][:1])
    initial_shape = (1, *mesh_shape[1:])
    chunk_size = max(1, int(evaluation_chunk_size))
    if source_backend == ANALYTIC_SOURCE_BACKEND:
        if micro_latent is not None:
            raise ValueError("analytic VPFP primary cannot carry a micro latent")
        if coordinate_precision == IEEE_COORDINATE_MATMUL_PRECISION:
            with ieee_float32_coordinate_matmul(model):
                rho, g, _source, _phi, _electric = (
                    _materialize_analytic_cartesian(
                        model,
                        latent,
                        axes=initial_axes,
                        mesh_shape=initial_shape,
                        equation_condition=equation_condition,
                        evaluation_chunk_size=chunk_size,
                    )
                )
        else:
            rho, g, _source, _phi, _electric = _materialize_analytic_cartesian(
                model,
                latent,
                axes=initial_axes,
                mesh_shape=initial_shape,
                equation_condition=equation_condition,
                evaluation_chunk_size=chunk_size,
            )
        return rho[0], g[0]

    point_count = int(np.prod(initial_shape))
    rho_flat = np.empty(point_count, dtype=float)
    g_flat = np.empty(point_count, dtype=float)
    for start, stop, points in _mesh_point_chunks(
        initial_axes, initial_shape, chunk_size
    ):
        values = manufacture_standard_vpfp_source(
            model,
            points,
            latent,
            micro_latent=micro_latent,
            equation_condition=equation_condition,
            create_graph=False,
        )
        rho_flat[start:stop] = _numpy(values["rho"])
        g_flat[start:stop] = _numpy(values["g"])
    rho = rho_flat.reshape(initial_shape)[..., 0, 0]
    return rho[0], g_flat.reshape(initial_shape)[0]


def _materialize_analytic_cartesian(
    model,
    latent,
    *,
    axes: dict[str, np.ndarray],
    mesh_shape: tuple[int, int, int, int, int],
    equation_condition,
    evaluation_chunk_size: int,
):
    spacetime_shape = mesh_shape[:3]
    velocity_shape = mesh_shape[3:]
    spacetime_count = int(np.prod(spacetime_shape))
    velocity_count = int(np.prod(velocity_shape))
    spacetime_chunk_size = max(
        1, int(evaluation_chunk_size) // velocity_count
    )
    rho = np.empty(spacetime_count, dtype=float)
    phi = np.empty(spacetime_count, dtype=float)
    electric = np.empty((spacetime_count, 2), dtype=float)
    g = np.empty((spacetime_count, velocity_count), dtype=float)
    source = np.empty_like(g)
    velocity_x, velocity_y = np.meshgrid(
        axes["vx"], axes["vy"], indexing="ij"
    )
    velocity = np.column_stack(
        (velocity_x.reshape(-1), velocity_y.reshape(-1))
    )
    prepared_velocity = prepare_standard_vpfp_cartesian_velocity_jet(
        model,
        velocity,
        equation_condition=equation_condition,
    )
    torch = model.torch
    with torch.no_grad():
        for start in range(0, spacetime_count, spacetime_chunk_size):
            stop = min(spacetime_count, start + spacetime_chunk_size)
            indices = np.arange(start, stop, dtype=np.int64)
            it, ix, iy = np.unravel_index(indices, spacetime_shape)
            spacetime = np.column_stack(
                (axes["t"][it], axes["x"][ix], axes["y"][iy])
            )
            values = manufacture_standard_vpfp_cartesian_source_analytic(
                model,
                spacetime,
                None,
                latent,
                prepared_velocity=prepared_velocity,
            )
            rho[start:stop] = _numpy(values["rho"])
            phi[start:stop] = _numpy(values["phi"])
            electric[start:stop] = _numpy(values["electric"])
            g[start:stop] = _numpy(values["g"])
            source[start:stop] = _numpy(values["source"])
    electric_grid = electric.reshape(*spacetime_shape, 2)
    return (
        rho.reshape(spacetime_shape),
        g.reshape(mesh_shape),
        source.reshape(mesh_shape),
        phi.reshape(spacetime_shape),
        np.moveaxis(electric_grid, -1, 1),
    )


def _mesh_point_chunks(
    axes: dict[str, np.ndarray], shape: tuple[int, ...], chunk_size: int
) -> Iterable[tuple[int, int, np.ndarray]]:
    total = int(np.prod(shape))
    for start in range(0, total, int(chunk_size)):
        stop = min(total, start + int(chunk_size))
        indices = np.arange(start, stop, dtype=np.int64)
        yield start, stop, _points_from_flat_indices(axes, shape, indices)


def _points_from_flat_indices(
    axes: dict[str, np.ndarray], shape: tuple[int, ...], indices: np.ndarray
) -> np.ndarray:
    it, ix, iy, ivx, ivy = np.unravel_index(indices, shape)
    return np.column_stack(
        (
            axes["t"][it],
            axes["x"][ix],
            axes["y"][iy],
            axes["vx"][ivx],
            axes["vy"][ivy],
        )
    )


def _audit_indices(point_count: int, requested: int) -> np.ndarray:
    count = max(1, min(int(point_count), int(requested)))
    return np.unique(np.linspace(0, point_count - 1, count, dtype=np.int64))


def _resolve_private_path(case_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else case_dir / path


def _numpy(value) -> np.ndarray:
    return value.detach().cpu().numpy().astype(float, copy=False)


def _numpy_array(value) -> np.ndarray:
    if hasattr(value, "detach"):
        return _numpy(value)
    return np.asarray(value, dtype=float)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=256)
def _cached_sha256_key(path: str, size: int, mtime_ns: int) -> str:
    del size, mtime_ns
    return _sha256(Path(path))


def _cached_sha256(path: Path) -> str:
    stat = path.stat()
    return _cached_sha256_key(str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns))
