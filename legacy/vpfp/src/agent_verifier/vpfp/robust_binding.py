"""Immutable policy binding for one VPFP robust-verifier round."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

from .runtime_paths import is_persistent_run_path


VPFP_ROBUST_BINDING_SCHEMA = "vpfp_robust_verifier_binding_v2"


def bind_vpfp_robust_verifier_policy(
    *,
    run_root: str | Path,
    generator_provenance: Mapping[str, Any],
    perturbation_config: Any,
    external_authorization: Mapping[str, Any],
    create: bool,
) -> Path:
    """Create once, then require byte-equivalent experiment semantics."""

    root = Path(run_root).resolve()
    if not is_persistent_run_path(root):
        raise ValueError("VPFP robust verifier binding needs configured run storage")
    generator = dict(generator_provenance)
    if (
        generator.get("generator_family") != "coarse_projection_answer_first"
        or generator.get("truth_primary") != "continuous_hidden_answer"
        or generator.get("low_fidelity_equation_solve_is_truth") is not False
        or generator.get("source_derivative_backend")
        != "structured_fourier_hermite_jet_v1"
    ):
        raise ValueError("VPFP procedural generator provenance is invalid")
    authorization = dict(external_authorization)
    if (
        authorization.get("authorized") is not True
        or authorization.get("agent_count") != 2
        or authorization.get("model") != "gpt-5.6-sol"
        or authorization.get("reasoning_effort") != "max"
    ):
        raise ValueError("VPFP external Agent authorization is incomplete")
    if is_dataclass(perturbation_config):
        perturbation = asdict(perturbation_config)
    elif isinstance(perturbation_config, Mapping):
        perturbation = dict(perturbation_config)
    else:
        raise TypeError("VPFP perturbation config must be a dataclass or mapping")

    payload = {
        "schema": VPFP_ROBUST_BINDING_SCHEMA,
        "enabled": True,
        "generator_provenance": generator,
        "external_agent_authorization": authorization,
        "answer_first_structured_perturbation": {
            "enabled": True,
            "config": perturbation,
            "base_generator_parameters_modified": False,
            "hidden_parameterization_public": False,
        },
        "equation_first_candidate_self_convergence": {
            "enabled": True,
            "reference_solution_used_for_candidate_metric": False,
            "cross_solver_error_used": False,
            "strict_monotonicity_is_acceptance_gate": False,
            "selection_role": "auxiliary_tiebreak_only",
        },
        "answer_first_exact_error_primary": True,
    }
    canonical = json.loads(json.dumps(payload, sort_keys=True))
    serialized = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    canonical["policy_sha256"] = hashlib.sha256(serialized.encode()).hexdigest()
    path = root / "controller_private" / "robust_verifier_binding.json"
    if path.exists():
        if not path.is_file() or path.is_symlink():
            raise ValueError("VPFP robust verifier binding path is unsafe")
        if json.loads(path.read_text(encoding="utf-8")) != canonical:
            raise ValueError("VPFP robust verifier policy changed within a round")
    elif create:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        temporary.write_text(
            json.dumps(canonical, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    else:
        raise FileNotFoundError(
            "VPFP robust verifier policy was not bound at round creation"
        )
    return path
