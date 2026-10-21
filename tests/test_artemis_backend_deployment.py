"""Cold Artemis deployment guards; no model execution or physical imports.

This revision-specific acceptance test preserves physical science while checking
the explicitly requested latest shared protocol and publication defaults.
It is not a scientific qualification test.
"""

import ast
from hashlib import sha256
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
BASELINE = "b1eed0b3748ca4ccbaaff82880bb7f21538104d9"
MERGED_PROTOCOL = "aa5665d"
UPSTREAM_PROTOCOL = "b2cd7b8"
REVIEWED_BUDGET_REPAIR = "7b20c2aecaae13af43917b0f3c535e35c92c2923"
REPAIRED_SHARED_PATHS = {
    "src/pde_agent_verifier_solver/engine/candidate_execution.py",
    "src/pde_agent_verifier_solver/engine/evaluator.py",
    "src/pde_agent_verifier_solver/engine/python_candidate.py",
    "src/pde_agent_verifier_solver/engine/unified_runtime.py",
    "src/pde_agent_verifier_solver/engine/round_recovery.py",
    "src/pde_agent_verifier_solver/sdk/continuous_solution.py",
    "src/pde_agent_verifier_solver/sdk/solve_budget.py",
}
PHYSICAL_REVISION = "91e2889dcd3f17137a13d4fd349b49539d3b5629"
OPTIMIZED = {
    "exact_query_tx_reuse": "request",
    "exact_query_source_mode": "source_only",
    "factorized_jvp_mode": "forward",
    "exact_query_tx_mode": "demand",
    "exact_query_transfer_mode": "request",
    "generation_sensor_geometry": "precomputed",
}


def _git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True, capture_output=True).stdout


def _baseline(path):
    return _git("show", f"{BASELINE}:{path}")


def _assignment(tree, name):
    return next(node for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))


def test_backend_configuration_and_physical_pin_are_explicit_and_consistent():
    relative = "configs/plasma_canonical_artemis.json"
    raw = (ROOT / relative).read_bytes()
    backend, old = json.loads(raw), json.loads(_baseline(relative))
    settings, old_settings = backend["factory_settings"], old["factory_settings"]
    assert settings.pop("oracle_settings") == OPTIMIZED
    assert settings.pop("physical_checkout") == str(ROOT.parent / "artemis-oracle-opt2-20260907")
    guidance = settings.pop("reward_guidance_settings")
    previous_guidance = json.loads(_git("show", "7c22b7f:configs/vp_1d1v_artemis_compact_v5_reward_guidance_causal_agent_iteration.json"))
    assert guidance == previous_guidance["vp_1d1v"]["guidance"]["backend"]
    old_settings.pop("physical_checkout")
    assert backend == old  # Base science/assets unchanged; G reuses the trained path.
    formal = json.loads((ROOT / "configs/artemis_canonical_5round.json").read_bytes())
    launcher = ast.parse((ROOT / "scripts/run_canonical_artemis_campaign.py").read_bytes())
    digest = sha256(raw).hexdigest()
    assert formal["backend"] == {"config": "plasma_canonical_artemis.json", "sha256": digest}
    assert ast.literal_eval(_assignment(launcher, "BACKEND_CONFIG_SHA256").value) == digest
    physical = ast.parse((ROOT / "src/pde_agent_verifier_solver/plugins/vp_1d1v/physical.py").read_bytes())
    assert ast.literal_eval(_assignment(physical, "ARTEMIS_PHYSICAL_REVISION").value) == PHYSICAL_REVISION


def test_physical_recipe_stays_fixed_while_execution_follows_latest_protocol():
    relative = "configs/artemis_canonical_5round.json"
    current, old = json.loads((ROOT / relative).read_bytes()), json.loads(_baseline(relative))
    reference = json.loads((ROOT / "configs/paper/ns_draft_v1/base_s101.json").read_bytes())
    assert current["execution"] == reference["execution"]
    assert current["promotion"] == reference["promotion"]
    assert current["master_seed"] == reference["master_seed"] == 101
    assert current["official"] == {"original": 0, "base": 128, "guided": 0,
                                    "base_perturbed": 0, "guided_perturbed": 0}
    assert current["generator"]["guidance_enabled"] is True
    assert current["generator"]["perturbations_enabled"] is True
    assert current["development"]["allowed_sources"] == ["base", "guided"]
    current["development"]["allowed_sources"] = old["development"]["allowed_sources"]
    for current_agent, old_agent in zip(current["agents"], old["agents"], strict=True):
        current_agent.pop("prompt")
        old_agent.pop("prompt")
    for key in ("backend", "execution", "campaign_id", "master_seed", "official", "promotion", "profile", "generator"):
        current.pop(key)
        old.pop(key)
    assert current == old  # S0, macros, query/task, budgets and Terminal unchanged.


def _without_opt4_transport(source, relative):
    tree = ast.parse(source)
    class Normalize(ast.NodeTransformer):
        def visit_Assign(self, node):
            if any(isinstance(t, ast.Name) and t.id == "_CANDIDATE_ORACLE_INLINE_ARRAY_BYTES" for t in node.targets):
                return None
            return self.generic_visit(node)
        def visit_ClassDef(self, node):
            permitted = ({"AgentWireCodec": {"__init__", "_encode_array", "_decode_inline_array", "_decode_sidecar_array"}}
                if relative.endswith("agent_wire.py") else
                {"_CandidateOracleHost": {"__init__"}, "CandidatePublicSessionClient": {"__init__"}})
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name in permitted.get(node.name, set()):
                    child.body = [ast.Pass()]
                    child.args = ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[])
            return self.generic_visit(node)
    return ast.dump(Normalize().visit(tree), include_attributes=False)


def test_only_reviewed_budget_recovery_and_opt4_transport_leaves_change():
    protected = (
        "src/pde_agent_verifier_solver/engine",
        "src/pde_agent_verifier_solver/sdk",
        "src/pde_agent_verifier_solver/app",
        "configs/candidates/artemis_hold_initial_bundle",
        "scripts/canonical_codex.config.toml",
    )
    paths = _git("ls-tree", "-r", "--name-only", MERGED_PROTOCOL, "--", *protected).decode().splitlines()
    assert paths
    for relative in sorted(set(paths) | REPAIRED_SHARED_PATHS):
        if relative == "src/pde_agent_verifier_solver/app/backend_run.py":
            continue  # Generic injection of the upstream Candidate service, tested separately.
        revision = REVIEWED_BUDGET_REPAIR if relative in REPAIRED_SHARED_PATHS else MERGED_PROTOCOL
        current, previous = (ROOT / relative).read_bytes(), _git("show", f"{revision}:{relative}")
        if relative.endswith(("/agent_wire.py", "/candidate_execution.py")):
            # Endpoint/codec leaf behavior is covered by test_artemis_opt4_rpc;
            # all other execution, authorization, wait-budget and lifecycle code
            # remains identical to the reviewed protocol.
            assert _without_opt4_transport(current, relative) == _without_opt4_transport(previous, relative), relative
        else:
            assert current == previous, relative


def test_worker_is_exactly_the_latest_upstream_worker():
    relative = "scripts/run_canonical_codex_agent.py"
    assert (ROOT / relative).read_bytes() == _git("show", f"{UPSTREAM_PROTOCOL}:{relative}")
