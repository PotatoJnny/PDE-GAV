"""The shared numerical trainer's advertised seed fixes model initialization."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_same_training_seed_reproduces_parameters_despite_ambient_rng(tmp_path):
    torch = pytest.importorskip("torch")
    from pde_agent_verifier_solver.plugins.ns.completed_fields import numerical_symbols
    numerical_symbols()
    from agent_verifier.neural_generator.structured_ns_distribution import StructuredCoefficientPCA
    spec = importlib.util.spec_from_file_location("_seed_prepare", ROOT / "scripts/prepare_ns_reconstructed_generator.py")
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    coefficients = np.random.default_rng(91).normal(size=(6, 2, 2, 2, 2)).astype(np.float32)
    signature = {"split_manifest_sha256": "seed-test", "projection_config": {},
                 "coefficient_shape": [2, 2, 2, 2]}
    train = prepare.export_coefficient_corpus(coefficients[:4], case_ids=list("abcd"), families=["f"]*4,
        signature=signature, output=tmp_path/"train", split="train")
    validation = prepare.export_coefficient_corpus(coefficients[4:], case_ids=list("ef"), families=["f"]*2,
        signature=signature, output=tmp_path/"validation", split="validation")
    pca = tmp_path/"pca.pt"
    StructuredCoefficientPCA.fit(coefficients[:4], rank=2, source_signature=signature, seed=8).save(pca)
    spec = importlib.util.spec_from_file_location("_shared_numerical_trainer_seed", ROOT/"legacy/ns/scripts/train_ns_structured_coefficient_flow.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        results = []
        for ambient in (13, 79):
            torch.manual_seed(ambient)
            out = tmp_path/str(ambient)
            trainer.main(["--train-manifest", str(train), "--validation-manifest", str(validation),
                "--pca-checkpoint", str(pca), "--out", str(out), "--rank", "2", "--steps", "2",
                "--batch-size", "2", "--validation-cases", "2", "--width", "8", "--depth", "1",
                "--time-width", "4", "--warmup-steps", "1", "--checkpoint-every", "2",
                "--validate-every", "2", "--seed", "101", "--amp", "none", "--device", "cpu"])
            results.append(torch.load(out/"checkpoints/flow_step_0000002.pt", weights_only=False))
        for name in ("model_state_dict", "ema_state_dict"):
            for key, first in results[0][name].items():
                torch.testing.assert_close(first, results[1][name][key], rtol=0, atol=0)
    finally:
        torch.set_num_threads(previous_threads)
