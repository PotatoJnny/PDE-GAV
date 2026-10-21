"""Fixed-root physical/control subset of the 512-attempt confirmation run.

No candidate selection, sampling or training occurs here. The enclosing driver
owns the hard process deadline; this helper publishes every completed unit and
does not start another unit after its remaining wall budget is exhausted.
"""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from run_vpfp_independent_confirmation import (
    file_sha, write_json, stream, json_sha, assert_frozen, checked_asset,
    NATIVE_KINDS, validate_candidate_kind,
)

SCHEMA = "vpfp_independent_physics_subset_run_v1"


def first_stratum_rows(rows):
    """First metadata row per registered stratum; no success/feature input."""
    first = {}
    for index, row in enumerate(rows):
        group = row["stratum_id"]
        if type(group) is not int or not 0 <= group < 8:
            raise ValueError("eight integer strata required")
        first.setdefault(group, index)
    if set(first) != set(range(8)):
        raise ValueError("every pre-registered stratum must be present")
    return [first[group] for group in range(8)]


def physics_design(references, candidates, master_seed):
    controls = [{"id": "phase", "kind": "relative_phase_scramble"},
                {"id": "incoherent", "kind": "coherent_to_incoherent", "minimum_mode": 3},
                {"id": "collapse", "kind": "root_index", "indices": [0]*8},
                {"id": "reweighted", "kind": "root_index", "indices": [0,0,0,0,0,1,1,1,2,2,3,3,4,5,6,7],
                 "source_sampling_weights": (np.exp(-np.arange(8))/np.exp(-np.arange(8)).sum()).tolist(),
                 "weight_law": "exp(-original_stratum_index)_normalized_not_feature_selected",
                 "finite_schedule_is_not_exact_realization_of_weights": True}]
    for amplitude in (.125, .5, 2.):
        for cycles in (1., 8.):
            controls.append({"id": f"drive-a{amplitude:g}-c{cycles:g}", "kind": "drive_velocity",
                "global_amplitude": [amplitude, 0.], "local_amplitude": [amplitude/2, 0.],
                "cycles": cycles, "mode": [1, 0], "phase": 0.,
                "amplitude_units": "physical_velocity_not_thermal_scaled"})
    return {"schema": SCHEMA, "selection_rule": "first_original_metadata_row_per_stratum_before_measurement_no_failed_row_replacement",
        "reference_rows": {regime: first_stratum_rows(item["expected_splits"]["calibration"])
                           for regime, item in references.items()},
        "candidate_rows": {item["id"]: first_stratum_rows(item["rows"]) for item in candidates},
        # Off-node quarter controls: h=1/256 and h/2 remain in their33-frame
        # interval. Node accuracy must not stand in for between-node behavior.
        "reference_role": "calibration", "time_fractions": [.0703125, .7578125],
        "spatial_probe_fractions": [[.173, .389]], "physical_velocity_probes": [[0., 0.]],
        "spatial_mass_grids": [17, 33], "relative_fd_step": 2.**-8, "include_l2": True,
        "time_counts": [65, 129], "control_reference_macro_maximum_mode": 6,
        "control_reference_fit_grid": 33, "controls": controls,
        "phase_seed_by_regime": {regime: stream("independent-control-phase-"+regime, master_seed) for regime in references},
        "control_geometry": "same_remaining_504_calibration_fit_and_512_refA_refB_all_controls_no_anchor_leakage",
        "control_statistics": "one_row_per_original_physical_root_duplicates_retained_in_control_receipt_not_new_N",
        "control_full_velocity_rows": [0], "control_time_spectrum_rows": [0],
        "reference_time_spectrum_rows": list(range(8)),
        "candidate_time_spectrum_rows": list(range(8)),
        "benchmark": "first_registered_reference_root_two_points_reused_as_its_measurement_not_new_N",
        "refinement": {"role": "calibration", "anchor_positions": [0], "max_cases": 1,
            "levels": "original_coarse_and_same_domain_1p5x_space_velocity_intervals_2x_time_half_dt_FP64",
            "source_family_and_seed_unchanged": True, "refined_512_population": False},
        "bootstrap_repeats": 49, "budget_grid": "existing_fixed_regime_fidelity_coverage_BUDGET_GRID_not_tuned",
        "physics_population_scope": "8_fixed_roots_per_reference_regime_and_candidate_not_all_512",
        "scientific_admission": False}


def subset_sample(sample, rows):
    rows = np.asarray(rows, dtype=np.int64)
    return replace(sample, features=np.asarray(sample.features)[rows], conditions=np.asarray(sample.conditions)[rows],
        root_ids=tuple(sample.root_ids[i] for i in rows), answer_sha256=tuple(sample.answer_sha256[i] for i in rows),
        valid_mask=np.asarray(sample.valid_mask)[rows], numerical_uncertain_mask=np.asarray(sample.numerical_uncertain_mask)[rows],
        budgets=np.asarray(sample.budgets)[rows],
        budget_defined_mask=None if sample.budget_defined_mask is None else np.asarray(sample.budget_defined_mask)[rows],
        lineage_ids=None if sample.lineage_ids is None else tuple(sample.lineage_ids[i] for i in rows),
        failures=tuple({**entry, "row": int(np.flatnonzero(rows == entry["row"])[0])}
                       for entry in sample.failures if entry["row"] in rows))


def independent_control_sample(sample):
    """Collapse literal same-root copies, never assign invented independent IDs."""
    first = {}
    for index, root in enumerate(sample.root_ids):
        if root in first:
            previous = first[root]
            if (sample.answer_sha256[index] != sample.answer_sha256[previous]
                    or not np.array_equal(np.asarray(sample.conditions)[index], np.asarray(sample.conditions)[previous])
                    or not np.array_equal(np.asarray(sample.features)[index], np.asarray(sample.features)[previous])):
                raise ValueError("multiple distinct descendants of one root require a cluster estimator, not this thin control")
        else:
            first[root] = index
    return subset_sample(sample, list(first.values())), {"control_rows": len(sample.root_ids),
        "distinct_physical_roots": len(first), "independent_statistical_rows": list(first.values()),
        "same_root_replicas_are_not_new_N": True}


def control_status(units, names, carrier_present):
    if not carrier_present:
        return "failed"
    return "completed" if names and all(units[name]["status"] == "completed" for name in names) else "completed_with_measurement_failures"


def physical_points(model, design):
    start, stop = model.config.time_domain
    lengths = model.config.equation.x_domain
    xy = np.asarray(design["spatial_probe_fractions"], dtype=np.float64)*np.asarray(lengths)
    times = start+(stop-start)*np.asarray(design["time_fractions"], dtype=np.float64)
    points = np.column_stack((np.repeat(times, len(xy)), np.tile(xy, (len(times), 1))))
    return points, xy


def native_temporal_frequency(model, full, condition, *, root_id, provenance, design):
    """B's OWN source/macros at fixed physical probes; never A-reconstruct B."""
    import importlib
    import torch
    from pde_agent_verifier_solver.plugins.vpfp.perturbation_backend import _legacy_api
    from pde_agent_verifier_solver.plugins.vpfp.regime_fidelity_coverage import _macro_values
    from pde_agent_verifier_solver.plugins.vpfp.physical_controls import _time_spectrum, _derivative_energy
    _legacy_api()
    analytic = importlib.import_module("agent_verifier.vpfp.analytic_source")
    count = model.config.structured_time_control_count
    expected = (2*(count-1)+1, 4*(count-1)+1)
    if tuple(design["time_counts"]) != expected or not bool(torch.isfinite(full).all()):
        raise ValueError("finite B answer and registered nested native time grids required")
    _, xy = physical_points(model, design)
    vv = np.asarray(design["physical_velocity_probes"])
    if vv.shape != xy.shape:
        raise ValueError("one fixed physical velocity per spatial probe required")
    report = {"schema":"vpfp_native_B_fixed_probe_temporal_frequency_v1", "execution_passed":True,
        "physical_root_id":root_id,"source_provenance":provenance,"decoder_kind":"old_full_B",
        "input_full_sha256":sha256(full.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
        "condition":condition.detach().cpu().tolist(),"time_counts":list(expected),"grids":[],
        "scientific_regime_qualified":False,"independent_PDE_residual":False,
        "limitations":["fixed physical probes not spatial/velocity norms", "finite-window leakage and unresolved frequencies",
            "65/129 agreement does not establish uniform convergence", "source time derivative uses coordinate finite differences"]}
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad(), analytic.ieee_float32_coordinate_matmul(model):
            for size in expected:
                times = np.linspace(*model.config.time_domain,size)
                points = np.column_stack((np.repeat(times,len(xy)),np.tile(xy,(size,1)),np.tile(vv,(size,1))))
                values, derivatives = {}, {}
                for offset in range(0,len(points),16):
                    query = points[offset:offset+16]
                    jet = analytic.analytic_structured_macro_jet(model,query[:,:3],full)
                    macros = _macro_values(jet,model.hermite_modes,float(condition[1]),model.config.velocity_floor_mixture)
                    source = analytic.manufacture_standard_vpfp_source_analytic(model,query,full,equation_condition=condition)
                    current = {key:macros[key] for key in ("rho","jx","jy","K")}
                    current.update({key:source[key].detach().double().cpu().numpy() for key in ("f","source")})
                    dt = {key:macros["dt_"+key] for key in ("jx","jy","K")}
                    dt.update(rho=jet["rho_derivatives"][0].detach().double().cpu().numpy(),
                        f=source["f_gradient"][:,0].detach().double().cpu().numpy())
                    for key,value in current.items(): values.setdefault(key,[]).append(value)
                    for key,value in dt.items(): derivatives.setdefault(key,[]).append(value)
                values = {key:np.concatenate(parts).reshape(size,len(xy)) for key,parts in values.items()}
                derivatives = {key:np.concatenate(parts).reshape(size,len(xy)) for key,parts in derivatives.items()}
                derivatives["source"] = np.gradient(values["source"],times,axis=0,edge_order=2)
                report["grids"].append({"count":size,"times":times.tolist(),
                    "values":{key:value.tolist() for key,value in values.items()},
                    "dt_values":{key:value.tolist() for key,value in derivatives.items()},
                    "spectra":{key:_time_spectrum(value,times) for key,value in values.items()},
                    "derivative_energy":{key:_derivative_energy(value,times) for key,value in derivatives.items()}})
    finally:
        model.train(was_training)
    coarse, fine = report["grids"]
    report["fine_minus_coarse_derivative_energy"] = {key:
        fine["derivative_energy"][key]["time_integrated_probe_mean_square"]-coarse["derivative_energy"][key]["time_integrated_probe_mean_square"]
        for key in coarse["derivative_energy"]}
    json.dumps(report,allow_nan=False)
    return report


class _Progress:
    def __init__(self, output, plan_sha, design, regime, wall_seconds):
        if not math.isfinite(wall_seconds) or wall_seconds <= 0:
            raise ValueError("positive remaining physical-stage wall budget required")
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        self.deadline = self.started+wall_seconds
        self.report = {"schema": SCHEMA, "regime": regime, "frozen_plan_sha256": plan_sha, "design": design,
            "execution_passed": False, "physical_regime_qualified": False, "scientific_admission": False,
            "refined_512_population": False, "wall_budget_seconds": wall_seconds, "units": {}, "stage": "load_bound_inputs",
            "controls": {row["id"]: {"status": "pending"} for row in design["controls"]},
            "pending_scientific_questions": ["uniform_space_time_and_velocity_error_budget",
                "third_reference_resolution_and_velocity_domain_sensitivity", "population_wide_physical_qualification"],
            "note": "Numerical differences and budget curves are evidence for review, not thresholds fitted to candidates."}
        self.save()

    def save(self):
        self.report["actual_wall_seconds"] = time.monotonic()-self.started
        write_json(self.output/"result.json", self.report)

    def check(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("physical-stage deadline; unstarted registered units remain pending")

    def unit(self, name, function):
        self.check()
        self.report["stage"] = name
        self.report["units"][name] = {"status": "running"}
        self.save()
        started = time.monotonic()
        try:
            value = function()
            path = self.output/(name+".json")
            write_json(path, value)
            passed = value.get("execution_passed", True) is True
            self.report["units"][name] = {"status": "completed" if passed else "measurement_failed",
                "report": str(path), "sha256": file_sha(path), "actual_wall_seconds": time.monotonic()-started}
            self.save()
            return value
        except TimeoutError:
            raise
        except Exception as error:
            self.report["units"][name] = {"status": "failed", "failure": {"type": type(error).__name__,
                "reason": str(error), "traceback": traceback.format_exc(), "control_receipt": getattr(error, "receipt", None)},
                "actual_wall_seconds": time.monotonic()-started}
            self.save()
            return None


def _reference_cache(directory, corpus, design, frozen):
    from pde_agent_verifier_solver.plugins.vpfp.diffusion_data import load_split
    from pde_agent_verifier_solver.plugins.vpfp.regime_fidelity_coverage import sample_from_extraction, VPFPRegimeFeaturePlan
    directory = Path(directory)
    def document(path):
        frozen[str(path)] = file_sha(path)
        return json.loads(path.read_text())
    result = document(directory/"result.json")
    if result.get("execution_passed") is not True or result.get("sample_count") != 512:
        raise ValueError("all512 descriptive audit must complete before physical controls")
    audit_plan = document(directory/"frozen-plan.json")
    if Path(audit_plan["reference_source"]) != Path(corpus):
        raise ValueError("cached physical features reference the wrong corpus")
    descriptors = document(directory/"descriptors.json")
    cached, splits = {}, {}
    for role in ("calibration", "reference_a", "reference_b"):
        split = load_split(corpus, role, verify_hashes=False)
        rows = audit_plan["reference_rows"][role]
        if len(rows) != 512 or sorted(rows) != list(range(512)):
            raise ValueError("complete512 cached references are required")
        path = directory/(role+"-features.npy")
        frozen[str(path)] = file_sha(path)
        matrix = np.load(path, allow_pickle=False)
        report = document(directory/(role+"-measurements.json"))
        cached[role] = sample_from_extraction(matrix, split.conditions[rows], [str(split.case_ids[i]) for i in rows], report)
        splits[role] = split
    feature_options = dict(audit_plan["plan"])
    for key in ("time_fractions", "grid_sizes"):
        feature_options[key] = tuple(feature_options[key])
    return cached, splits, descriptors, VPFPRegimeFeaturePlan(**feature_options), audit_plan


def validate_candidate_schedule(candidate, plan, target):
    """Bind a self-consistent native receipt to THIS pre-registered candidate.

    Native loading/source/stream checks are performed by freeze_generated.
    This additional comparison rejects a different otherwise legal model or
    512-root schedule from the same training corpus before any measurement.
    """
    validate_candidate_kind(candidate)
    rows = candidate["rows"]
    if len(rows) != 512:
        raise ValueError("candidate plan requires all512 registered attempts")
    # Match freeze_generated's resolved directory keys, including an allowed
    # regular directory below a canonicalized scratch ancestor.
    path = Path(candidate["generated_dir"]).resolve(strict=True)/"generated-provenance.json"
    provenance = checked_asset({"path":str(path),"sha256":target["files"][str(path)]},json_only=True)
    if provenance.get("selection_sha256") != plan["selection_sha256"]:
        raise ValueError("generated selection SHA differs from the registered candidate plan")
    for field, row_field in (("seeds","seed"),("condition_seeds","condition_seed")):
        actual, expected = provenance.get(field), [row[row_field] for row in rows]
        if (not isinstance(actual,list) or any(type(seed) is not int for seed in actual)
                or actual != expected):
            raise ValueError("generated "+field+" differ from the registered512 attempt schedule")
    actual = np.asarray(target["conditions"])
    expected = np.asarray([row["conditions"] for row in rows],dtype=np.float32)
    if (actual.dtype != np.dtype("float32") or actual.shape != (512,2) or expected.shape != actual.shape
            or not np.array_equal(actual.view(np.uint32),expected.view(np.uint32))):
        raise ValueError("generated FP32 condition bits differ from the registered512 attempt schedule")
    if candidate["kind"] in NATIVE_KINDS:
        checked_asset(candidate["asset"])
        if (target["status"] != "ready_native_direct" or provenance.get("checkpoint") != candidate["asset"]
                or provenance.get("checkpoint_sha256") != candidate["asset"]["sha256"]):
            raise ValueError("native generated checkpoint differs from the explicitly selected asset")
        binding = dict(candidate["asset"])
    elif candidate["kind"] == "weak_hierarchy":
        manifest = checked_asset(candidate["asset"],json_only=True)
        if (manifest.get("schema") != "vpfp_moving_frame_native_model_manifest_v1"
                or target["status"] != "ready_paired_A_B"):
            raise ValueError("selected weak asset must be the explicit paired native manifest")
        binding = {role:{"checkpoint":manifest[role+"_checkpoint"]["path"],
            "checkpoint_sha256":manifest[role+"_checkpoint"]["sha256"]} for role in ("macro","conditional")}
        if provenance.get("checkpoint_bindings") != binding or target.get("checkpoint_bindings") != binding:
            raise ValueError("generated macro/conditional bindings differ from the explicitly selected pair")
    else:
        raise ValueError("unsupported explicit candidate kind")
    return {"schema":"vpfp_registered_candidate_generation_binding_v1","execution_passed":True,
        "candidate":candidate["id"],"selection_sha256":plan["selection_sha256"],
        "generated_provenance_sha256":target["files"][str(path)],"registered_attempt_count":512,
        "root_and_condition_seed_lists_equal":True,"condition_fp32_bits_equal":True,
        "selected_asset":candidate["asset"],"actual_checkpoint_binding":binding,
        "self_consistent_other_model_or_schedule_is_not_accepted":True}


def freeze_candidate_targets(plan, regime, source, spec, edges, *, frozen, check_deadline=lambda:None):
    """Freeze every target before the first expensive physics/reference unit."""
    from run_vpfp_regime_fidelity_coverage import freeze_generated
    targets,receipts = {},{}
    for candidate in plan["candidates"]:
        if candidate["regime"] != regime:
            continue
        check_deadline()
        target = freeze_generated(Path(candidate["generated_dir"]),manifest=source,spec=spec,count=512,edges=edges)
        receipts[candidate["id"]] = validate_candidate_schedule(candidate,plan,target)
        frozen.update(target["files"])
        targets[candidate["id"]] = target
    return targets,receipts


def run(plan, *, plan_sha256, regime, output, device, wall_seconds):
    """Actual fixed subset; caller's process deadline also bounds in-flight calls."""
    import torch
    from pde_agent_verifier_solver.plugins.vpfp.diffusion_data import read_manifest, VPFPOnlineModelSpec
    from pde_agent_verifier_solver.plugins.vpfp.moving_frame import MovingFrameDecoder, MovingFrameMacroConfig, value_config_from_decoder
    from pde_agent_verifier_solver.plugins.vpfp.value_control_diffusion import extract_value_controls, lift_value_controls
    from pde_agent_verifier_solver.plugins.vpfp.independent_physics import audit_independent_physics
    from pde_agent_verifier_solver.plugins.vpfp import physical_controls as controls
    from pde_agent_verifier_solver.plugins.vpfp.regime_fidelity_coverage import sample_from_extraction, audit_regime_fidelity_coverage
    from pde_agent_verifier_solver.plugins.vpfp.moving_frame_audit import extract_moving_frame_regime_features
    from diagnose_vpfp_known_initial import _runtime
    design = plan["physics_design"]
    if design != physics_design(plan["references"], plan["candidates"], plan["master_seed"]):
        raise ValueError("physical design differs from the pre-measurement registered protocol")
    progress = _Progress(output, plan_sha256, design, regime, wall_seconds)
    frozen = dict(plan["frozen_files"])
    try:
        assert_frozen(plan)
        item = plan["references"][regime]
        source = read_manifest(item["model_source"]["corpus"], verify_hashes=False)
        reference = read_manifest(item["reference_source"], verify_hashes=False)
        if source["manifest_sha256"] != item["model_source"]["manifest_sha256"] or source["manifest_sha256"] == reference["manifest_sha256"]:
            raise ValueError("distinct declared model/reference identities required")
        frozen[str(Path(item["reference_source"])/"manifest.json")] = file_sha(Path(item["reference_source"])/"manifest.json")
        # The subset consumes calibration answers plus the cached banks' root/
        # condition arrays. Bind their ORIGINAL manifest hashes before reading;
        # do not bless changed data by hashing its current bytes after the fact.
        for role in ("calibration","reference_a","reference_b"):
            for key, entry in reference["splits"][role]["files"].items():
                if role == "calibration" or key != "coefficients":
                    frozen[str(Path(item["reference_source"])/entry["path"])] = entry["sha256"]
        assert_frozen({"frozen_files":frozen})
        spec = VPFPOnlineModelSpec.from_mapping(source["model_spec"])
        edges = (np.linspace(*spec.collision_nu_range,5),np.linspace(*spec.temperature_range,3))
        progress.report["stage"] = "freeze_all_registered_candidate_targets"
        progress.save()
        targets,target_bindings = freeze_candidate_targets(plan,regime,source,spec,edges,
            frozen=frozen,check_deadline=progress.check)
        write_json(progress.output/"registered-candidate-bindings.json",target_bindings)
        runtime = _runtime(source, spec, device)
        frozen.update(runtime.frozen_files)
        points, xy = physical_points(runtime.model, design)
        vc = value_config_from_decoder(runtime.model)
        mc = MovingFrameMacroConfig(control_count=vc.control_count, max_mode=design["control_reference_macro_maximum_mode"],
            time_domain=vc.time_domain, spatial_domain_lengths=runtime.model.config.equation.x_domain)
        moving = MovingFrameDecoder(runtime.model, mc, value_config=vc)
        if vc.control_count != 33:
            raise ValueError("registered65/129 physical protocol requires33 native controls")
        audit_directory = Path(item["reference_source"]).parents[1]/(regime+"-audit")
        cached, splits, descriptors, feature_plan, audit_plan = _reference_cache(audit_directory, item["reference_source"], design, frozen)
        if audit_plan["reference_source_manifest_sha256"] != reference["manifest_sha256"]:
            raise ValueError("cached reference SHA differs from fresh corpus")
        selected = design["reference_rows"][regime]
        split = splits["calibration"]
        expected = item["expected_splits"]["calibration"]
        if any(str(split.case_ids[i]) != expected[i]["case_id"] or int(split.stratum_ids[i]) != expected[i]["stratum_id"] for i in selected):
            raise ValueError("fixed calibration anchor metadata changed")
        roots = [str(split.case_ids[i]) for i in selected]
        raw = torch.tensor(np.asarray(split.latents[selected]), device=device)
        conditions = torch.tensor(np.asarray(split.conditions[selected]), device=device)
        provenance = {"source_manifest_sha256": reference["manifest_sha256"], "split": "calibration", "data_origin": "physical_reference"}
        # Hold the physical control ancestors OUT of the calibration fit; use
        # exactly the same504/512/512 geometry for B, A(reference), all controls.
        rows_by_root = {root: i for i, root in enumerate(cached["calibration"].root_ids)}
        b_sample = subset_sample(cached["calibration"], [rows_by_root[root] for root in roots])
        cached["calibration"] = subset_sample(cached["calibration"],
            [i for i, root in enumerate(cached["calibration"].root_ids) if root not in set(roots)])
        if len(cached["calibration"].root_ids) != 504:
            raise ValueError("control calibration must exclude exactly8 selected physical roots")
        write_json(progress.output/"measurement-inputs.json", {"files": frozen, "model_source_manifest_sha256": source["manifest_sha256"],
            "reference_source_manifest_sha256": reference["manifest_sha256"], "points_txy": points.tolist(),
            "calibration_anchor_rows": selected, "calibration_anchor_roots": roots, "control_decoder_identity": moving.identity(),
            "controls_calibration_rows": len(cached["calibration"].root_ids), "refined_512_population": False})

        def deep(name, decoder, answer, condition, root):
            return progress.unit(name, lambda: {"physical_root_id": root, **audit_independent_physics(
                decoder, answer, condition, points_txy=points, spatial_times=points[:, 0],
                spatial_grids=tuple(design["spatial_mass_grids"]), relative_fd_step=design["relative_fd_step"],
                chunk_size=32, include_l2=design["include_l2"])})

        # This first fixed root is both timing benchmark and its own B result.
        deep("reference-B-00", runtime.model, raw[:1], conditions[0], roots[0])
        from pde_agent_verifier_solver.plugins.vpfp.reference_refinement import refine_reference_samples
        progress.unit("reference-refinement", lambda: refine_reference_samples(item["reference_source"], "calibration",
            expected_manifest_sha256=reference["manifest_sha256"], output=progress.output/"reference-refinement-artifacts",
            rows=[selected[0]], max_cases=1, device=device,
            points_txyvv=np.column_stack((points, np.zeros((len(points), 2)))).tolist()))
        for i in range(1, 8):
            deep(f"reference-B-{i:02d}", runtime.model, raw[i:i+1], conditions[i], roots[i])
        for i in design["reference_time_spectrum_rows"]:
            progress.unit(f"reference-B-time-{i:02d}", lambda i=i: native_temporal_frequency(runtime.model,raw[i:i+1],
                conditions[i],root_id=roots[i],provenance=provenance,design=design))

        def score(name, sample):
            effective, multiplicity = independent_control_sample(sample)
            def audit():
                from pde_agent_verifier_solver.plugins.vpfp.regime_fidelity_coverage import _frechet
                report = audit_regime_fidelity_coverage({**cached, "generated": effective}, descriptors=descriptors,
                    nu_edges=edges[0], temperature_edges=edges[1], bootstrap_repeats=design["bootstrap_repeats"])
                center,axes = np.asarray(report["geometry"]["center"]),np.asarray(report["geometry"]["axes"])
                coordinate = lambda cohort:(np.arcsinh(np.asarray(cohort.features)[cohort.valid_mask])-center)@axes.T
                report.update(control_multiplicity=multiplicity,
                    control_all_output_rows={"count":len(sample.root_ids),"valid_count":int(np.asarray(sample.valid_mask).sum()),
                        "failed_count":int((~np.asarray(sample.valid_mask)).sum())},
                    core_attempted_count_means_unique_roots_not_control_output_yield=True,
                    attempt_weighted_Frechet={**_frechet(coordinate(cached["reference_a"]),coordinate(sample)),
                        "fit":"same504_calibration_PCA_no_refit", "independent_sample_count":len(effective.root_ids),
                        "repeated_outputs_are_frequency_weights_not_independent_samples":True,"uncertainty_interval":None},
                    evidence_role="fixed_physical_control_sensitivity_not_independent_model_confirmation")
                return report
            return progress.unit(name+"-budget-fidelity-coverage",audit)

        score("reference-B", b_sample)
        anchor_box = {}
        def reconstruct():
            anchor_box.update(controls.reference_carrier_from_full(moving, raw, conditions, root_ids=roots,
                provenance=provenance, fit_grid=design["control_reference_fit_grid"]))
            return anchor_box["receipt"]
        progress.unit("reference-A-reconstruction", reconstruct)

        def measure_control(name, controlled, original=None):
            carrier, cond = controlled["carrier"], controlled["conditions"]
            root_ids = [row["physical_root_id"] for row in controlled["receipt"]["lineage"]]
            box = {}
            def features():
                full = lift_value_controls(carrier["z"], moving.value_config).detach().cpu().numpy()
                matrix, schema, report = extract_moving_frame_regime_features(moving, full,
                    carrier["m"].detach().cpu().numpy(), cond.detach().cpu().numpy(), plan=feature_plan, expected_identity=moving.identity())
                if schema != descriptors:
                    raise ValueError("actual A/control descriptor schema differs from original B reference bank")
                np.save(progress.output/(name+"-features.npy"), matrix, allow_pickle=False)
                box["sample"] = sample_from_extraction(matrix, cond.detach().cpu().numpy(), root_ids, report)
                report.update(control_receipt=controlled["receipt"], evidence_role="actual_A_control_not_feature_corruption",
                    execution_passed=report["complete"] and all("failure" not in case for case in report["cases"]))
                if original is not None:
                    # Pure representation baseline, not a perturbation effect.
                    difference = matrix-np.asarray(original.features)
                    report["A_reference_minus_B_representation"] = {"descriptor_names": [d["name"] for d in descriptors],
                        "per_root": difference.tolist(), "valid_pair_mask": (box["sample"].valid_mask & original.valid_mask).tolist(),
                        "not_a_control_effect": True}
                elif (progress.output/"reference-A-features.npy").is_file():
                    original_a = np.load(progress.output/"reference-A-features.npy", allow_pickle=False)
                    source_rows = [row["source_row"] for row in controlled["receipt"]["lineage"]]
                    original_report = json.loads((progress.output/"reference-A-measurements.json").read_text())
                    report["control_minus_A_reference"] = {"descriptor_names": [d["name"] for d in descriptors],
                        "source_anchor_rows":source_rows,"per_output_row":(matrix-original_a[source_rows]).tolist(),
                        "valid_pair_mask":[bool(case["hard_valid"] and original_report["cases"][row]["hard_valid"])
                                           for case,row in zip(report["cases"],source_rows)],
                        "A_to_B_representation_error_not_counted_as_control_effect":True}
                return report
            progress.unit(name+"-measurements", features)
            if "sample" in box:
                score(name, box["sample"])
            return carrier, cond, root_ids

        def frequency(name, decoder, carrier, cond, root_ids, positions, source_provenance=None):
            for i in positions:
                progress.unit(f"{name}-time-{i:02d}", lambda i=i: controls.temporal_frequency_diagnostics(decoder,
                    {k: v[i:i+1] for k, v in carrier.items()}, cond[i:i+1], root_ids=[root_ids[i]],
                    provenance=provenance if source_provenance is None else source_provenance,
                    spatial_probes=xy, velocity_probes=design["physical_velocity_probes"], time_counts=tuple(design["time_counts"])))

        if anchor_box:
            carrier, cond, ids = measure_control("reference-A", anchor_box, b_sample)
            for i in range(8):
                deep(f"reference-A-{i:02d}", moving, {k: v[i:i+1] for k, v in carrier.items()}, cond[i], ids[i])
            frequency("reference-A", moving, carrier, cond, ids, design["reference_time_spectrum_rows"])
            for setting in design["controls"]:
                label = setting["id"]
                progress.report["controls"][label] = {"status": "running", "setting": setting}
                before_units = set(progress.report["units"])
                box = {}
                def transform(setting=setting):
                    kwargs = dict(root_ids=roots, provenance=provenance)
                    args = (moving, anchor_box["carrier"], anchor_box["conditions"])
                    if setting["kind"] == "relative_phase_scramble":
                        value = controls.relative_phase_scramble(*args, **kwargs, seed=design["phase_seed_by_regime"][regime])
                    elif setting["kind"] == "coherent_to_incoherent":
                        value = controls.coherent_to_incoherent_flow(*args, **kwargs, minimum_mode=setting["minimum_mode"])
                    elif setting["kind"] == "root_index":
                        value = controls.index_carrier_control(*args, **kwargs, indices=setting["indices"],
                            selection_rule="pre_registered_"+setting["id"]+"_root_frequency_schedule_no_new_independent_roots",
                            source_sampling_weights=setting.get("source_sampling_weights"))
                    else:
                        value = controls.drive_velocity(*args, **kwargs, **{key: setting[key] for key in
                            ("global_amplitude", "local_amplitude", "mode", "cycles", "phase")})
                    box.update(value)
                    return value["receipt"]
                progress.unit(label+"-carrier", transform)
                if box:
                    carrier, cond, ids = measure_control(label, box)
                    for i in design["control_full_velocity_rows"]:
                        deep(f"{label}-physical-{i:02d}", moving, {k: v[i:i+1] for k, v in carrier.items()}, cond[i], ids[i])
                    frequency(label, moving, carrier, cond, ids, design["control_time_spectrum_rows"])
                new_units = sorted(set(progress.report["units"])-before_units)
                progress.report["controls"][label] = {"status": control_status(progress.report["units"],new_units,bool(box)),
                    "setting": setting, "units":new_units}
                progress.save()

        for candidate in plan["candidates"]:
            if candidate["regime"] != regime:
                continue
            progress.check()
            target = targets[candidate["id"]]
            generated_moving = None
            if target["status"] == "ready_paired_A_B":
                from pde_agent_verifier_solver.plugins.vpfp.coherent_diffusion import VPFPCoherentMacroConfig
                from pde_agent_verifier_solver.plugins.vpfp.value_control_diffusion import VPFPValueEDMConfig
                config = VPFPCoherentMacroConfig.from_dict(target["macro_config"])
                generated_moving = MovingFrameDecoder(runtime.model, MovingFrameMacroConfig(control_count=config.control_count,
                    max_mode=config.maximum_mode, time_domain=config.time_domain,
                    spatial_domain_lengths=runtime.model.config.equation.x_domain),
                    value_config=VPFPValueEDMConfig.from_dict(target["micro_config"]))
            for position, row in enumerate(design["candidate_rows"][candidate["id"]]):
                # Never replace a selected failed attempt with another row.
                full = torch.tensor(np.asarray(target["raw"][row:row+1]), device=device)
                condition = torch.tensor(np.asarray(target["conditions"][row]), device=device)
                root = target["roots"][row]
                name = candidate["id"]+f"-physical-{position:02d}"
                if generated_moving is None:
                    deep(name, runtime.model, full, condition, root)
                    progress.unit(name+"-time", lambda: native_temporal_frequency(runtime.model,full,condition,root_id=root,
                        provenance={"source_manifest_sha256":source["manifest_sha256"],"split":"generated",
                            "data_origin":"generated_answer"},design=design))
                else:
                    # Same draw/root, independent B numerical measurement. This
                    # is a decoder counterfactual, never an additional sample.
                    deep(name+"-B",runtime.model,full,condition,root)
                    progress.unit(name+"-B-time",lambda: native_temporal_frequency(runtime.model,full,condition,root_id=root,
                        provenance={"source_manifest_sha256":source["manifest_sha256"],"split":"generated",
                            "data_origin":"generated_answer"},design=design))
                    answer_box = {}
                    def native_answer():
                        z = extract_value_controls(full, generated_moving.value_config, validate_slopes=True)
                        if not torch.equal(lift_value_controls(z, generated_moving.value_config), full):
                            raise ValueError("generated A carrier full-to-value relift is not bitwise exact")
                        answer_box.update(m=torch.tensor(np.asarray(target["macro"][row:row+1]), device=device), z=z)
                        return {"execution_passed": True, "physical_root_id": root, "attempt": row,
                            "native_relift_bitwise": True, "decoder_identity": generated_moving.identity()}
                    progress.unit(name+"-binding", native_answer)
                    if answer_box:
                        deep(name+"-A", generated_moving, answer_box, condition, root)
                        frequency(name+"-A",generated_moving,answer_box,condition[None],[root],[0],
                            {"source_manifest_sha256":source["manifest_sha256"],"split":"generated","data_origin":"generated_answer"})
                    progress.report.setdefault("paired_generated_roots",[]).append({"candidate":candidate["id"],
                        "attempt":row,"root":root,"B_unit":name+"-B","A_unit":name+"-A",
                        "same_sample_not_two_independent_roots":True})
        assert_frozen({"frozen_files": frozen})
        write_json(progress.output/"measurement-inputs-final.json", {"files": frozen})
        progress.check()
        statuses = [entry["status"] for entry in progress.report["units"].values()]
        progress.report.update(execution_passed=bool(statuses) and all(status == "completed" for status in statuses),
            stage="registered_subset_execution_finished", refinement_root_count=1,
            confirmation_512_completed=False, population_512_is_descriptive_only=True)
    except Exception as error:
        progress.report["failure"] = {"type": type(error).__name__, "reason": str(error), "traceback": traceback.format_exc()}
    finally:
        progress.save()
    return progress.report
