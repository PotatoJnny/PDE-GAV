"""Collect actual numerical answer coefficients and fit a learned sampler.

Both operations require a Slurm allocation. ``collect`` performs the old PDE
solves offline once; ``train`` fits the answer distribution. Runtime draws from
the resulting checkpoint do not solve an initial-value problem.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pde_agent_verifier_solver.plugins.vpfp.canonical_binding import (
    VPFPOnlineModelSpec, build_vpfp_online_binding,
)
from pde_agent_verifier_solver.plugins.vpfp.learned_latent import (
    VPFPConditionalLatentDistribution, _file_sha256,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    collect = subparsers.add_parser("collect")
    collect.add_argument("--config", type=Path, required=True)
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--samples", type=int, default=256)
    collect.add_argument("--batch-size", type=int, default=4)
    collect.add_argument("--seed-start", type=int, default=9600000)
    collect.add_argument("--device", default="cuda")
    train = subparsers.add_parser("train")
    train.add_argument("--corpus", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--rank", type=int, default=16)
    arguments = parser.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("offline VPFP collection/training requires a Slurm compute allocation")
    if arguments.output.exists():
        parser.error("output already exists; select a fresh artifact path")
    if arguments.operation == "collect":
        if arguments.samples < 12 or arguments.batch_size < 1 or arguments.seed_start < 0:
            parser.error("collect requires >=12 samples, a positive batch size and nonnegative seeds")
        payload = json.loads(arguments.config.read_text())
        spec = VPFPOnlineModelSpec.from_mapping(payload.get("vpfp", payload))
        binding = build_vpfp_online_binding(
            spec, output_root=arguments.output.parent / "collection-resident-runtime",
            device=arguments.device,
        )
        sampler = binding.plugin.generator.backend.runtime.generator
        latents, conditions, identifiers = [], [], []
        for start in range(0, arguments.samples, arguments.batch_size):
            seeds = tuple(range(arguments.seed_start + start,
                                arguments.seed_start + min(arguments.samples, start + arguments.batch_size)))
            batch = sampler.generate(seeds)
            latents.append(batch["latents"].detach().cpu().numpy())
            conditions.append(batch["equation_conditions"].detach().cpu().numpy())
            identifiers.extend(seeds)
            print(json.dumps({"completed_answers": len(identifiers),
                              "requested_answers": arguments.samples}), flush=True)
        metadata = {
            "schema": "vpfp_physical_answer_corpus_v1",
            "data_origin": "numerical_simulation",
            "base_decoder_sha256": spec.base_decoder_sha256,
            "collision_nu_range": list(spec.collision_nu_range),
            "temperature_range": list(spec.temperature_range),
            "binding": dict(binding.report),
            "training_resolution": {"nx": 16, "ny": 16, "nv": 21, "nt": 33},
            "high_fidelity_validation": False,
            "experimental_data_validation": False,
        }
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        with arguments.output.open("xb") as target:
            np.savez_compressed(target, latents=np.concatenate(latents),
                equation_conditions=np.concatenate(conditions), sample_ids=np.asarray(identifiers),
                metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
        print(json.dumps({"corpus": str(arguments.output), "sha256": _file_sha256(arguments.output)}))
    else:
        corpus_sha = _file_sha256(arguments.corpus)
        with np.load(arguments.corpus, allow_pickle=False) as corpus:
            metadata = json.loads(str(corpus["metadata"]))
            if metadata.get("schema") != "vpfp_physical_answer_corpus_v1":
                raise ValueError("training needs a provenance-bearing physical answer corpus")
            model = VPFPConditionalLatentDistribution.fit(
                corpus["latents"], corpus["equation_conditions"],
                sample_ids=corpus["sample_ids"],
                base_decoder_sha256=metadata["base_decoder_sha256"],
                collision_nu_range=metadata["collision_nu_range"],
                temperature_range=metadata["temperature_range"], rank=arguments.rank,
                provenance={**metadata, "source_artifact_sha256": corpus_sha},
            )
        model.save(arguments.output)
        print(json.dumps({"checkpoint": str(arguments.output),
                          "sha256": _file_sha256(arguments.output),
                          "report": dict(model.metadata)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
