"""Reuse frozen5287092 draws for density/kinetic interventions; never sample.

Fresh output is mandatory. --reuse-from may reference completed units of an
identical block plan, including an otherwise partial run; failed/pending units
are never relabeled complete. The Slurm wrapper owns the hard process timeout.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
sys.path.insert(0, str(ROOT/"scripts"))
import run_vpfp_weak_dynamics_diagnostic as prior
from run_vpfp_independent_confirmation import checked_asset, file_sha, json_sha, write_json

SCHEMA = "vpfp_frozen_weak_block_run_v1"
OLD = Path("${PDE_GAV_RUN_ROOT}/vpfp-weak-dynamics-diagnostic-5287092/pipeline/cross-diagnostic")
OLD_RESULT_SHA = "46c4e543b209ed4e7b4305aba4bad97a938414a90c1a712ffab8fabbc2a8e761"
OLD_PLAN_SHA = "85b7a1fee91d7081d2dfb124cd2037d0dfce7073bd9886db0a0214e6873fa531"
OLD_ARMS = ("real_m_real_z", "real_m_generated_z")
ARMS = ("rho_real_kin_real", "rho_gen_kin_real", "rho_real_kin_gen", "rho_gen_kin_gen")
SCRATCH = Path("${PDE_GAV_RUN_ROOT}")
BASELINE_PHYSICAL_MODULES = {"weak_dynamics_diagnostic.py", "moving_frame.py", "moving_frame_backend.py",
    "moving_frame_binding.py", "physical_controls.py", "independent_physics.py", "value_control_diffusion.py",
    "structured_diffusion.py", "coherent_diffusion.py", "temporal_covariance_diffusion.py",
    "canonical_binding.py", "perturbation_backend.py"}


def validate_old_run(result, plan):
    if (result.get("schema") != prior.SCHEMA or result.get("execution_passed") is not True
            or result.get("stage") != "complete" or result.get("completed_carrier_count") != 64
            or result.get("registered_carrier_count") != 64 or result.get("failures") != []
            or result.get("new_training") is not False or result.get("pde_solves") != 0):
        raise ValueError("completed unchanged5287092 64-carrier run required")
    if (plan.get("schema") != prior.SCHEMA or plan.get("source_manifest_sha256") != prior.SOURCE_SHA
            or plan.get("source") != str(prior.SOURCE) or plan.get("source_split") != "validation"
            or plan.get("macro_fit_grid") != 33 or plan.get("sampling_steps") != 64
            or plan.get("master_seed") != prior.MASTER_SEED or len(plan.get("selected_roots", [])) != 8
            or plan.get("noise_plan") != prior.fixed_noise_plan(plan["selected_roots"])
            or [m["manifest"] for m in plan.get("models", [])] != list(prior.MODEL_MANIFESTS)):
        raise ValueError("original source/roots/two models/noise plan required")
    for seed in range(2):
        for group in range(8):
            for arm in prior.ARMS:
                if result["units"].get(f"seed-{seed}/root-{group}/{arm}", {}).get("status") != "completed":
                    raise ValueError("original registered carrier was not completed")


def validate_baseline(receipt, dynamics, row, *, seed, arm, manifest, design):
    for key, expected in {"root_id": row["root_id"], "equation_seed": row["equation_seed"],
            "row": row["row"], "stratum_id": row["stratum_id"], "model_replica": seed,
            "arm": arm, "view": arm, "m_origin": "real", "native_manifest": manifest,
            "source_manifest_sha256": prior.SOURCE_SHA, "condition_fp32_bits": row["condition_fp32_bits"],
            "complete_floor_preserved": True, "no_Hermite_reprojection": True}.items():
        if receipt.get(key) != expected:
            raise ValueError("baseline lineage/condition/carrier binding differs: "+key)
    if (receipt.get("noise_plan") != row or (arm == OLD_ARMS[1] and
            receipt.get("micro_conditioning_macro") != "real")):
        raise ValueError("baseline must use the frozen real-m conditional draw")
    if (dynamics.get("execution_passed") is not True or dynamics.get("stage") != "complete"
            or dynamics.get("root_id") != row["root_id"] or dynamics.get("provenance") != receipt
            or dynamics.get("input_carrier_sha256") != receipt["carrier_sha256"]
            or dynamics.get("decoder_identity") != receipt["decoder_identity"]
            or dynamics.get("design") != design):
        raise ValueError("completed dynamics does not describe this exact source carrier")
    theta = np.asarray(dynamics.get("condition"), dtype=np.float32)
    if theta.shape != (2,) or theta.view(np.uint32).tolist() != row["condition_fp32_bits"]:
        raise ValueError("saved dynamics physical condition differs from the frozen row")
    timeline = [r for r in dynamics.get("timeline", []) if r.get("count") == 129]
    if len(timeline) != 1 or timeline[0].get("probe_count") != 4:
        raise ValueError("original full129/four-probe timeline required, not partial arrays")
    return timeline[0]


def registered_units():
    units = {f"seed-{s}/root-{g}/{arm}": {"status": "pending", "kind": "carrier"}
        for s in range(2) for g in range(8) for arm in ARMS}
    units.update({f"clean/root-{g}": {"status": "pending", "kind": "clean"} for g in range(8)})
    units.update({f"seed-{s}/root-{g}/paired-comparison": {"status": "pending", "kind": "comparison"}
        for s in range(2) for g in range(8)})
    return units


def freeze_inputs(check_deadline):
    from pde_agent_verifier_solver.plugins.vpfp.diffusion_data import read_manifest, load_split
    from pde_agent_verifier_solver.plugins.vpfp.weak_block_diagnostic import block_design
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all", "--",
        "src", "scripts", "configs", "tests", "legacy/vpfp"], cwd=ROOT, text=True)
    if dirty.strip():
        raise ValueError("freeze source/configs/tests before compute")
    frozen = {}
    def freeze(path, expected=None, *, document=False):
        check_deadline(); path = Path(path).absolute()
        digest = file_sha(path)
        frozen[str(path)] = digest
        return json.loads(path.read_text()) if document else digest
    old_result = freeze(OLD/"result.json", OLD_RESULT_SHA, document=True)
    old_plan = freeze(OLD/"frozen-plan.json", OLD_PLAN_SHA, document=True)
    validate_old_run(old_result, old_plan)
    compatibility = {}
    # The legacy physical computation is identical, not merely a matching ABI.
    for old_path, expected in old_plan["frozen_files"].items():
        relative = None
        for marker in ("/src/pde_agent_verifier_solver/plugins/vpfp/", "/legacy/vpfp/"):
            if marker in old_path:
                if marker == "/legacy/vpfp/" or Path(old_path).name in BASELINE_PHYSICAL_MODULES:
                    relative = marker[1:]+old_path.split(marker, 1)[1]
        if Path(old_path).name in ("run_vpfp_weak_dynamics_diagnostic.py", "run_vpfp_independent_confirmation.py"):
            relative = "scripts/"+Path(old_path).name
        if relative is not None:
            freeze(ROOT/relative, expected)
            compatibility[relative] = expected
    if "src/pde_agent_verifier_solver/plugins/vpfp/weak_dynamics_diagnostic.py" not in compatibility:
        raise ValueError("old physical source fingerprints missing")
    manifest = read_manifest(prior.SOURCE, verify_hashes=False)
    if manifest["manifest_sha256"] != prior.SOURCE_SHA:
        raise ValueError("original weak corpus identity required")
    freeze(prior.SOURCE/"manifest.json")
    for entry in [manifest["seed_plan"], *manifest["splits"]["validation"]["files"].values()]:
        path = (prior.SOURCE/entry["path"]).resolve(strict=True)
        if not path.is_relative_to(prior.SOURCE.resolve()): raise ValueError("validation escaped original corpus")
        freeze(path, entry["sha256"])
    split = load_split(prior.SOURCE, "validation", verify_hashes=False)
    if prior.select_validation_rows(split) != old_plan["selected_roots"]:
        raise ValueError("original metadata-first roots or FP32 condition bits changed")
    for item in old_plan["models"]:
        header = freeze(item["manifest"]["path"], item["manifest"]["sha256"], document=True)
        if header != item["assets"]: raise ValueError("native manifest payload changed")
        for role in ("base_decoder", "macro_checkpoint", "conditional_checkpoint"):
            freeze(header[role]["path"], header[role]["sha256"])
    design = block_design()
    if old_plan["design"] != design["phase_design"]: raise ValueError("old129 source/probe design differs")
    entries = []
    for row in old_plan["noise_plan"]:
        seed, group = row["replica"], row["stratum_id"]
        entry = {"row": row, "sources": {}}
        for arm in OLD_ARMS:
            directory = OLD/f"seed-{seed}/root-{group}/{arm}"
            receipt = freeze(directory/"carrier-receipt.json", document=True)
            dynamics = freeze(directory/"dynamics.json", document=True)
            validate_baseline(receipt, dynamics, row, seed=seed, arm=arm,
                manifest=old_plan["models"][seed]["manifest"], design=old_plan["design"])
            for key in ("m", "z"):
                item = receipt["files"][key]
                if item["path"] != str(directory/(key+".npy")): raise ValueError("carrier block escaped root directory")
                freeze(item["path"], item["sha256"])
            entry["sources"][arm] = {"receipt": receipt, "receipt_path": str(directory/"carrier-receipt.json"),
                "report_path": str(directory/"dynamics.json"), "report_sha256": frozen[str(directory/"dynamics.json")]}
        a, b = [entry["sources"][arm]["receipt"] for arm in OLD_ARMS]
        if a["files"]["m"]["tensor_sha256"] != b["files"]["m"]["tensor_sha256"]:
            raise ValueError("real macro differs between source blocks")
        entries.append(entry)
    # Real/real is repeated across model seeds, not eight additional roots.
    for group in range(8):
        pair = [next(e for e in entries if e["row"]["replica"] == s and e["row"]["stratum_id"] == group)
            ["sources"][OLD_ARMS[0]]["receipt"] for s in (0, 1)]
        if pair[0]["carrier_sha256"] != pair[1]["carrier_sha256"] or pair[0]["decoder_identity"] != pair[1]["decoder_identity"]:
            raise ValueError("real clean pair differs across seeds; cannot reuse eight clean measurements")
    for path in [*sorted((ROOT/"src/pde_agent_verifier_solver/plugins/vpfp").glob("*.py")),
                 Path(__file__),
                 ROOT/"tests/test_vpfp_weak_block_diagnostic.py", ROOT/"tests/test_vpfp_weak_block_driver.py"]:
        freeze(path)
    plan = {"schema": SCHEMA, "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "design": design, "old_result": {"path": str(OLD/"result.json"), "sha256": OLD_RESULT_SHA},
        "old_plan_sha256": OLD_PLAN_SHA, "source_manifest_sha256": prior.SOURCE_SHA,
        "selected_roots": old_plan["selected_roots"], "entries": entries, "models": old_plan["models"],
        "physical_source_compatibility_sha256": compatibility, "frozen_files": frozen,
        "historical_compatibility_scope": "actual_A_decoder_source_and_phase_Runtime; unused_reference_refinement_not_part_of_129baseline",
        "registered_carrier_count": 64, "reused_baseline_count": 32, "new_shallow_carrier_count": 32,
        "new_clean_root_count": 8, "new_deep_carrier_count": 4, "new_training": False,
        "new_sampling": False, "new_PDE_solves": 0, "scientific_admission": False}
    prior.check_frozen(plan)
    return plan, split


def load_carrier(decoder, source, row, device):
    import torch
    from pde_agent_verifier_solver.plugins.vpfp.physical_controls import carrier_hashes
    receipt = source["receipt"]; carrier = {}
    for key in ("m", "z"):
        item = receipt["files"][key]
        checked_asset({key: item[key] for key in ("path", "sha256")})
        value = np.load(item["path"], allow_pickle=False)
        if value.dtype != np.dtype('float32'):
            raise ValueError('native stored block dtype/tensor SHA changed')
        carrier[key] = torch.from_numpy(np.array(value, copy=True)).to(device)
    condition = torch.tensor(row["conditions"], dtype=torch.float32, device=device)
    if condition.cpu().numpy().view(np.uint32).tolist() != row["condition_fp32_bits"]:
        raise ValueError("physical condition FP32 bits changed")
    decoder.carrier_to_payload(carrier)
    if decoder.identity() != receipt["decoder_identity"] or carrier_hashes(decoder, carrier, condition[None])[0] != receipt["carrier_sha256"]:
        raise ValueError("native decoder or complete carrier differs from completed baseline")
    return carrier, condition


def reused_timeline(source):
    report = json.loads(Path(source["report_path"]).read_text())
    return next(row for row in report["timeline"] if row["count"] == 129)


def reusable_units(path, plan):
    if path is None: return {}, None
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_dir(): raise ValueError("regular existing block output required for reuse")
    old_plan = json.loads((path/"frozen-plan.json").read_text())
    result = json.loads((path/"result.json").read_text())
    identity = json_sha(plan)
    if json_sha(old_plan) != identity or result.get("frozen_plan_sha256") != identity or result.get("schema") != SCHEMA:
        raise ValueError("partial reuse requires the exact same input/design/code frozen plan")
    outputs = {}
    for name, state in result.get("units", {}).items():
        if state.get("status") != "completed" or not state.get("report"): continue
        if name not in registered_units(): raise ValueError("unknown completed reuse slot")
        file = Path(state["report"]).absolute()
        # Do not recursively follow a previous reuse chain or an arbitrary file.
        if not file.is_relative_to(path) or file.is_symlink(): continue
        if file != path/name/"result.json": raise ValueError("reuse report belongs to another registered slot")
        report = json.loads(file.read_text())
        if report.get("execution_passed") is not True: raise ValueError("cached unit is not completed")
        outputs[name] = {"report": str(file), "report_sha256": state["report_sha256"]}
    return outputs, {"path": str(path), "result_sha256": file_sha(path/"result.json"),
        "plan_sha256": identity, "failed_pending_units_not_reused": True}


class Progress:
    def __init__(self, output, seconds):
        self.output, self.started = output, time.monotonic(); self.deadline = self.started+seconds
        self.report = {"schema": SCHEMA, "execution_passed": False, "stage": "freeze_inputs", "failures": [],
            "units": registered_units(), "registered_carrier_count": 64, "registered_clean_count": 8,
            "registered_deep_count": 4, "registered_comparison_count": 16,
            "new_training": False, "new_sampling": False, "new_PDE_solves": 0, "scientific_admission": False,
            "evidence_role": "eight_opened_development_roots_nested_block_interventions", "wall_seconds_limit": seconds}
        self.reuse = {}; self.save()
    def save(self):
        self.report["elapsed_seconds"] = time.monotonic()-self.started
        for kind in ("carrier", "clean", "comparison"):
            self.report["completed_"+kind+"_count"] = sum(v.get("kind") == kind and v["status"] == "completed" for v in self.report["units"].values())
        write_json(self.output/"result.json", self.report)
    def check(self):
        if time.monotonic() >= self.deadline: raise TimeoutError("block diagnostic budget exhausted; no shortened protocol")
    def unit(self, name, function, *, expected=None):
        self.check(); state = self.report["units"].setdefault(name, {})
        state.update(status="running"); self.report["stage"] = name; self.save(); start = time.monotonic()
        try:
            if name in self.reuse:
                cached = self.reuse[name]
                value = json.loads(Path(cached["report"]).read_text())
                if any(value.get(k) != v for k, v in (expected or {}).items()): raise ValueError("reuse unit/root/carrier identity differs")
                path = Path(cached["report"]); reused = True
            else:
                path = self.output/name/"result.json"; path.parent.mkdir(parents=True, exist_ok=True)
                value = function(path); reused = False
                if value.get("execution_passed") is not True: raise ValueError("measurement failed; partial report retained")
                if not path.exists(): write_json(path, value)
            state.update(status="completed", report=str(path), report_sha256=file_sha(path), reused_complete_unit=reused,
                elapsed_seconds=time.monotonic()-start)
            return value
        except Exception as error:
            state.update(status="failed", elapsed_seconds=time.monotonic()-start,
                failure={"type": type(error).__name__, "reason": str(error), "traceback": traceback.format_exc()})
            partial = self.output/name/"result.json"
            if partial.is_file() and not partial.is_symlink():
                state.update(report=str(partial), report_sha256=file_sha(partial))
            self.report["failures"].append({"unit": name, **state["failure"]})
            if isinstance(error, TimeoutError): raise
            return None
        finally: self.save()


def run(output, *, device="cuda", wall_seconds=10800, reuse_from=None):
    if not os.environ.get("SLURM_JOB_ID", "").isdigit() or not os.environ.get("SLURM_TMPDIR"):
        raise ValueError("compute allocation required; no login numerical execution")
    if device not in ("cpu", "cuda") or not math.isfinite(wall_seconds) or wall_seconds <= 0:
        raise ValueError("cpu/cuda and positive finite allocation wall budget required")
    output = Path(output).absolute()
    if output.exists() or output.is_symlink() or not output.is_relative_to(SCRATCH): raise ValueError("fresh scratch output required")
    for root in (OLD, prior.SOURCE, prior.MODEL_ROOT):
        if output == root or output.is_relative_to(root) or root.is_relative_to(output): raise ValueError("output overlaps immutable input root")
    if reuse_from is not None and (Path(reuse_from).absolute() == output or Path(reuse_from).absolute().is_relative_to(output)):
        raise ValueError("fresh output cannot contain previous reuse source")
    output.mkdir(parents=True); progress = Progress(output, wall_seconds); plan = None
    try:
        import torch
        from pde_agent_verifier_solver.plugins.vpfp.moving_frame_binding import load_moving_frame_sampler
        from pde_agent_verifier_solver.plugins.vpfp.weak_block_diagnostic import (
            make_block_carriers, measure_block_carrier, measure_clean_derivatives, compare_timelines)
        plan, split = freeze_inputs(progress.check)
        write_json(output/"frozen-plan.json", plan)
        progress.report["frozen_plan_sha256"] = json_sha(plan)
        progress.reuse, progress.report["reuse_from"] = reusable_units(reuse_from, plan); progress.save()
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        for seed, item in enumerate(plan["models"]):
            progress.check()
            sampler, decoder = load_moving_frame_sampler(item["manifest"]["path"], item["manifest"]["sha256"], device=device)
            if sampler.identity["corpus_manifest_sha256"] != prior.SOURCE_SHA: raise ValueError("native source differs")
            for entry in (e for e in plan["entries"] if e["row"]["replica"] == seed):
                row = entry["row"]; group = row["stratum_id"]; progress.check()
                progress.report["stage"] = f"seed-{seed}/root-{group}/load_frozen_blocks"; progress.save()
                real, condition = load_carrier(decoder, entry["sources"][OLD_ARMS[0]], row, device)
                generated, _ = load_carrier(decoder, entry["sources"][OLD_ARMS[1]], row, device)
                carriers, receipts = make_block_carriers(decoder, real, generated, condition)
                timelines = {}
                for arm in ARMS:
                    name = f"seed-{seed}/root-{group}/{arm}"; carrier = carriers[arm]
                    provenance = {**receipts[arm], "root_id": row["root_id"], "row": row["row"], "stratum_id": group,
                        "model_replica": seed, "source_manifest_sha256": prior.SOURCE_SHA, "split": "validation",
                        "data_origin": "physical_reference" if arm == ARMS[0] else "counterfactual_block_intervention",
                        "source_carrier_receipts": [e["receipt_path"] for e in entry["sources"].values()],
                        "condition_fp32_bits": row["condition_fp32_bits"], "original_noise_plan": row}
                    directory = output/name; directory.mkdir(parents=True, exist_ok=True)
                    prior.save_carrier(directory/"carrier", decoder, carrier, condition, provenance)
                    deep = seed == 0 and group in (0, 1) and arm in ARMS[1:3]
                    def measure(path):
                        if arm in (ARMS[0], ARMS[3]):
                            source = entry["sources"][OLD_ARMS[0 if arm == ARMS[0] else 1]]
                            timeline = reused_timeline(source)
                            result = {"schema": SCHEMA+"_baseline_reuse", "execution_passed": True,
                                "root_id": row["root_id"], "input_carrier_sha256": provenance["output_carrier_sha256"],
                                "original_report": {"path": source["report_path"], "sha256": source["report_sha256"]},
                                "decoder_identity": decoder.identity(), "timeline": timeline,
                                "recomputed_here": False, "new_independent_sample": False, "scientific_admission": False}
                            write_json(path, result); return result
                        return measure_block_carrier(decoder, carrier, condition, output=path, root_id=row["root_id"],
                            provenance=provenance, design=plan["design"], deep=deep, check_deadline=progress.check)
                    result = progress.unit(name, measure, expected={"root_id": row["root_id"],
                        "input_carrier_sha256": receipts[arm]["output_carrier_sha256"]})
                    if result is not None: timelines[arm] = result["timeline"]
                    if arm in ARMS[1:3] and result is not None:
                        key = "first_deep_benchmark" if deep else "first_shallow_benchmark"
                        progress.report.setdefault(key, {"unit": name, "elapsed_seconds": progress.report["units"][name]["elapsed_seconds"],
                            "reused": progress.report["units"][name]["reused_complete_unit"], "projection_is_not_runtime_guarantee": True})
                if len(timelines) == 4:
                    def compare(path):
                        result = {"schema": SCHEMA+"_paired", "execution_passed": True, "root_id": row["root_id"],
                            "scientific_admission": False, "comparisons": {arm: compare_timelines(timelines[ARMS[0]], timelines[arm]) for arm in ARMS[1:]}}
                        write_json(path, result); return result
                    progress.unit(f"seed-{seed}/root-{group}/paired-comparison", compare, expected={"root_id": row["root_id"]})
                if seed == 0:
                    _, full = prior.condition_for_row(split, row, device)
                    provenance = {"data_origin": "physical_reference", "view": "real_m_real_z", "split": "validation",
                        "source_manifest_sha256": prior.SOURCE_SHA, "same_real_carrier_across_seeds_checked": True,
                        "source_carrier_receipt": entry["sources"][OLD_ARMS[0]]["receipt_path"]}
                    progress.unit(f"clean/root-{group}", lambda path: measure_clean_derivatives(decoder, real, condition,
                        reference_full=full, output=path, root_id=row["root_id"], provenance=provenance,
                        design=plan["design"], check_deadline=progress.check), expected={"root_id": row["root_id"],
                            "input_carrier_sha256": receipts[ARMS[0]]["output_carrier_sha256"]})
            del sampler, decoder
        progress.check(); prior.check_frozen(plan)
        progress.report.update(stage="complete", execution_passed=not progress.report["failures"] and all(
            state["status"] == "completed" for state in progress.report["units"].values()))
    except Exception as error:
        progress.report.update(execution_passed=False, stage="stopped", failure={"type": type(error).__name__,
            "reason": str(error), "traceback": traceback.format_exc()})
    finally: progress.save()
    return progress.report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--wall-seconds", type=float, default=10800.)
    parser.add_argument("--reuse-from", type=Path)
    args = parser.parse_args(argv)
    return 0 if run(args.output, device=args.device, wall_seconds=args.wall_seconds,
        reuse_from=args.reuse_from)["execution_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
