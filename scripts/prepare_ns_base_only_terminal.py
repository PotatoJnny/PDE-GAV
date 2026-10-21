#!/usr/bin/env python3
"""Freeze Base-only perturbations for NS terminal evaluation, preserving the old bank."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import shutil

from pde_agent_verifier_solver.engine.common_evaluation import content_sha256
from pde_agent_verifier_solver.plugins.ns.case_bank import (
    load_ns_continuous_case_bank, write_ns_continuous_case_bank,
)
from pde_agent_verifier_solver.plugins.ns.continuous_backend import (
    LegacyStructuredNSSmoothPerturber, StructuredNSManufacturedCaseSource,
)
from pde_agent_verifier_solver.plugins.ns.tasks import NSSolverTask
from pde_agent_verifier_solver.sdk import EvaluationPhase, GenerationKind, GenerationPurpose
from pde_agent_verifier_solver.sdk.macro_loop import CohortKind, SampleRequest

PROTOCOL = "iclr_ns_bgp_1024_terminal_base_perturb_v4"


def prepare(source_manifest, output_directory, *, generator_config, seed=101):
    """Keep B/G/existing BP unchanged; replace GP with BP from unused Base parents.

    Counts scale with the frozen input bank, allowing the same operation to be
    qualified on small actual fields. No candidate or learned sampler is run.
    """
    source_path = Path(source_manifest).resolve()
    original_bytes = source_path.read_bytes()
    bank = load_ns_continuous_case_bank(source_path, expected_purpose=GenerationPurpose.TERMINAL)
    original = bank.manifest
    rows = original["cases"]
    base = [row for row in rows if row["stratum"] == "base"]
    guided = [row for row in rows if row["stratum"] == "guided"]
    bp = [row for row in rows if row["stratum"] == "base_perturbed"]
    gp = [row for row in rows if row["stratum"] == "guided_perturbed"]
    if (not base or len(base) != len(guided) or len(bp) + len(gp) != len(base)
            or len(rows) != 3 * len(base) or not gp):
        raise ValueError("source must be an equal B/G/P terminal bank containing Guided-P")
    base_ids = {row["case"]["case_id"] for row in base}
    used_parents = [row["case"]["parent_case_id"] for row in bp]
    if len(set(used_parents)) != len(used_parents) or not set(used_parents) <= base_ids:
        raise ValueError("existing Base-P must have distinct parents in the frozen Base bank")
    parents = [row for row in base if row["case"]["case_id"] not in set(used_parents)]
    runtime = generator_config["runtime"]
    family, amplitude = runtime["perturbation_family"], runtime["perturbation_amplitude"]
    for row in bp:
        audit = row["generation_provenance"]["perturbation_audit"]
        if audit["family"] != family or audit["amplitude_magnitude"] != amplitude:
            raise ValueError("new perturbations must use the existing calibrated family/amplitude")
    output = Path(output_directory).resolve()
    if output.is_relative_to(source_path.parent):
        raise ValueError("output must be outside the frozen source bank directory")
    output.mkdir(parents=True, exist_ok=False)
    def no_clean_sampling(seed):
        raise AssertionError("terminal revision must preserve frozen Base/Guided fields")
    source = StructuredNSManufacturedCaseSource(
        generator_id=bank.provenance.get("generator_id", generator_config["generator_id"]),
        artifact_sha256=generator_config["generator_sha256"], clean_sampler=no_clean_sampling,
        perturbation_sampler=LegacyStructuredNSSmoothPerturber(
            amplitude=amplitude, family=family, spectral_max_mode=runtime["spectral_max_mode"]),
        max_points_per_request=runtime.get("max_points_per_request", 4096))
    def derive(kind, parent_id):
        material = f"{PROTOCOL}:{seed}:{bank.manifest_sha256}:{kind}:{parent_id}".encode()
        return int.from_bytes(sha256(material).digest()[:4], "big") & 0x7fffffff
    records_by_id = {record.case.case_id: record for record in bank.records}
    added, plans, details = [], {}, {}
    for index, row in enumerate(parents):
        parent_id = row["case"]["case_id"]
        parent = records_by_id[parent_id]
        # Restore the already validated frozen answer through the original
        # resident factory; this also works with the running batch's old source.
        answer = bank._answers[parent_id]
        source._commit((source._resident(
            parent.case, answer.raw_answer, answer, CohortKind.TERMINAL),))
        request_seed = derive("perturbation", parent_id)
        child = source.sample(SampleRequest(
            parent.case.round_index or 0, CohortKind.TERMINAL, seed=request_seed,
            count=1, kind=GenerationKind.PERTURBED, anchors=(parent.case,)))[0]
        audit = dict(source.controller_perturbation_audit(child.case.case_id))
        # Public IDs are opaque but reproducible across preparations; the
        # underlying source's random per-instance ID key is not serialized.
        child_id = "ns-case-" + content_sha256((PROTOCOL, bank.manifest_sha256, seed, parent_id))
        child = replace(child, case=replace(child.case, case_id=child_id))
        settings = dict(bank.plans[parent_id].settings)
        evaluation = NSSolverTask().create_evaluation(settings)
        plans[child_id] = evaluation.make_plan(child.case, EvaluationPhase.TERMINAL,
                                               settings, derive("query", parent_id))
        details[child_id] = {"controller_distribution": "common_base/base_perturbed",
                             "case_position": len(bp) + index, "request_seed": request_seed,
                             "perturbation_audit": audit}
        added.append(child)
    provenance = deepcopy(dict(bank.provenance))
    provenance.update(protocol_revision=PROTOCOL, revision_seed=seed,
        source_manifest=str(source_path), source_manifest_sha256=bank.manifest_sha256,
        change="retain B/G/BP and replace Guided-P with Base-P from unused Base parents",
        terminal_suites={
            "common_base": {"original": 0, "base": len(base), "guided": 0,
                            "base_perturbed": len(base), "guided_perturbed": 0},
            "common_guided": {"original": 0, "base": 0, "guided": len(guided),
                              "base_perturbed": 0, "guided_perturbed": 0}})
    added_path = write_ns_continuous_case_bank(output / "added_base_perturbed.json",
        bank_id="terminal_base_only_added", purpose=GenerationPurpose.TERMINAL,
        records=added, plans=plans, strata={r.case.case_id: "base_perturbed" for r in added},
        provenance=provenance, case_provenance=details)
    added_rows = json.loads(added_path.read_text())["cases"]
    retained = [deepcopy(row) for row in rows if row["stratum"] != "guided_perturbed"]
    for row in retained:
        old_state = source_path.parent / row["state_path"]
        new_state = output / row["state_path"]
        new_state.parent.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(old_state.parent, new_state.parent)
    result = deepcopy(dict(original))
    result.pop("content_sha256")
    result.update(bank_id="terminal_base_only_v4", provenance=provenance,
                  cases=retained + added_rows)
    result["case_count"] = len(result["cases"])
    result["strata_counts"] = dict(Counter(row["stratum"] for row in result["cases"]))
    result["content_sha256"] = content_sha256(result)
    manifest = output / "terminal.json"
    with manifest.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    manifest.chmod(0o600)
    checked = load_ns_continuous_case_bank(manifest, expected_purpose=GenerationPurpose.TERMINAL)
    assert dict(Counter(checked.strata.values())) == {
        "base": len(base), "guided": len(base), "base_perturbed": len(base)}
    verified_parents = [r.case.parent_case_id for r in checked.records
                        if checked.strata[r.case.case_id] == "base_perturbed"]
    assert len(verified_parents) == len(set(verified_parents)) and set(verified_parents) == base_ids
    assert source_path.read_bytes() == original_bytes
    receipt = {"protocol_revision": PROTOCOL, "status": "frozen",
        "source_manifest": str(source_path), "source_manifest_sha256": bank.manifest_sha256,
        "manifest": str(manifest), "manifest_sha256": checked.manifest_sha256,
        "strata_counts": result["strata_counts"], "retained_cases_unchanged": len(retained),
        "added_base_perturbed": len(added), "removed_guided_perturbed": len(gp),
        "all_base_parents_used_once": True,
        "generator_config_sha256": content_sha256(generator_config),
        "preparation_script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "coding_agent_calls": 0,
        "generator_training_calls": 0, "candidate_evaluations": 0}
    (output / "revision_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    config = json.loads(args.config.read_text())
    print(json.dumps(prepare(args.source_manifest, args.output,
                            generator_config=config["generator"], seed=args.seed)))


if __name__ == "__main__":
    main()
