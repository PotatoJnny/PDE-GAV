#!/usr/bin/env python
"""Create the direct Fourier--Hermite checkpoints used by long-time VPFP.

The robust long-time experiment does not train a neural decoder.  It uses a
deterministic direct-coefficient representation and learns a distribution over
its continuous coefficients online.  Historical checkpoints lived on scratch;
this script reconstructs the same representation from its explicit settings so
the paper experiment can be run again after scratch cleanup.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from agent_verifier.vpfp.continuous_flow_training import (
    VPFPFlowObservationEncoder,
    VPFPFlowObservationEncoderConfig,
    save_hermite_autoencoder_checkpoint,
)
from agent_verifier.vpfp.continuous_hermite import (
    VPFPContinuousHermiteConfig,
    VPFPContinuousHermiteField,
)
from agent_verifier.vpfp.hermite_density import total_degree_mode_pairs
from agent_verifier.vpfp.robust_perturbation import (
    VPFPRobustPerturbationConfig,
    build_deterministic_direct_model,
    expanded_vpfp_perturbation_config,
    write_vpfp_direct_decoder_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--collision-nu-min", type=float, default=10.0)
    parser.add_argument("--collision-nu-max", type=float, default=30.0)
    parser.add_argument("--temperature-min", type=float, default=0.8)
    parser.add_argument("--temperature-max", type=float, default=1.2)
    parser.add_argument("--t1", type=float, default=1.0)
    parser.add_argument("--time-controls", type=int, default=33)
    parser.add_argument("--hermite-degree", type=int, default=14)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.out).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    source = _write_small_source_checkpoint(output / "source_template.pt")
    base_config = _direct_config(args)
    base_model = build_deterministic_direct_model(base_config, device="cpu")
    base = write_vpfp_direct_decoder_checkpoint(
        source,
        output / "base_online_decoder.pt",
        model=base_model,
        metadata={
            "role": "base_online_answer_first_generator",
            "time_control_count": int(args.time_controls),
            "time_domain": [0.0, float(args.t1)],
            "regime_id": "long_time_strong_collision_v2",
        },
    )

    perturbation = VPFPRobustPerturbationConfig(
        base_hermite_total_degree=int(args.hermite_degree)
    )
    expanded_config = expanded_vpfp_perturbation_config(
        base_config, perturbation
    )
    expanded_model = build_deterministic_direct_model(
        expanded_config, device="cpu"
    )
    expanded = write_vpfp_direct_decoder_checkpoint(
        source,
        output / "expanded_robust_decoder.pt",
        model=expanded_model,
        metadata={
            "role": "expanded_structure_preserving_generator",
            "time_control_count": int(args.time_controls),
            "time_domain": [0.0, float(args.t1)],
            "regime_id": "long_time_strong_collision_v2",
        },
    )
    report = {
        "base_decoder_checkpoint": str(base),
        "expanded_decoder_checkpoint": str(expanded),
        "base_latent_dimension": int(base_config.latent_dim),
        "expanded_latent_dimension": int(expanded_config.latent_dim),
        "time_controls": int(args.time_controls),
        "collision_nu_range": [
            float(args.collision_nu_min),
            float(args.collision_nu_max),
        ],
        "temperature_range": [
            float(args.temperature_min),
            float(args.temperature_max),
        ],
        "time_domain": [0.0, float(args.t1)],
    }
    (output / "checkpoint_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def _direct_config(args: argparse.Namespace) -> VPFPContinuousHermiteConfig:
    hermite_modes = len(total_degree_mode_pairs(int(args.hermite_degree)))
    provisional = replace(
        VPFPContinuousHermiteConfig(),
        latent_dim=1,
        rho_positivity_gauge="phase_invariant_group_l1_v2",
        hermite_total_degree=int(args.hermite_degree),
        structured_spectral_latent=True,
        structured_local_channels=hermite_modes,
        structured_local_spatial_mode=4,
        structured_time_basis="piecewise_cubic_hermite_v1",
        structured_time_control_count=int(args.time_controls),
        structured_direct_coefficients=True,
        structured_residual_scale=0.0,
        collision_nu_range=(
            float(args.collision_nu_min),
            float(args.collision_nu_max),
        ),
        temperature_range=(
            float(args.temperature_min),
            float(args.temperature_max),
        ),
        time_domain=(0.0, float(args.t1)),
    )
    config = replace(
        provisional,
        latent_dim=int(
            provisional.structured_latent_layout()["total_latent_dim"]
        ),
    )
    config.validate()
    return config


def _write_small_source_checkpoint(path: Path) -> Path:
    """Write only the schema template needed by the direct checkpoint writer."""

    decoder = VPFPContinuousHermiteField.build(VPFPContinuousHermiteConfig())
    encoder = VPFPFlowObservationEncoder.build(
        VPFPFlowObservationEncoderConfig(
            latent_dim=128,
            width=8,
            point_layers=1,
            attention_heads=1,
            query_tokens=1,
            self_attention_layers=1,
        )
    )
    save_hermite_autoencoder_checkpoint(
        path,
        decoder=decoder,
        encoder=encoder,
        feature_normalization={},
        latent_normalization={},
        case_latents=torch.empty((0, 128)),
        case_ids=(),
        train_indices=(),
        validation_indices=(),
        train_config={"role": "direct_checkpoint_schema_template"},
        step=0,
    )
    return path


if __name__ == "__main__":
    main()
