"""Exact provenance for the accepted weak Base's one-time portable export.

Only standard-library code: importing this helper cannot mix old/new Torch
modules. Historical loaders and their strict checks are never modified.
Six predeclared runtime-interface deltas are not model-graph equivalence.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import subprocess

from vpfp_weak_migration_paths import resolve_asset_path

SCHEMA = "vpfp_weak_portability_provenance_v1"
OLD_REVISION = "48dacc30e4bcef30560148c28566a1e305eb4c97"
OLD_ROOT = Path("${PDE_GAV_WORKTREES_ROOT}/vpfp-weak-guidance-20260911")
ORIGIN_RECEIPT_ASSET = {
    "path": "${PDE_GAV_RUN_ROOT}/vpfp-weak-guidance-5386292/pilot/base-binding.json",
    "sha256": "6bfd1bd409821229e53a273dbfb21f0d18647b0f7f66d19155eb7fa981238c54"}
ORIGIN_RECEIPT_SHA256 = "bcdb8eb63fc4a98edc6672c07dd3001ede4fbce5cbef2177895c53a66ffa69ea"
INPUT_COUNT, RELOCATED_COUNT = 559, 196
PREFIX = "src/pde_agent_verifier_solver/plugins/vpfp/"
# Registered before exporting or evaluating the portable candidate. Exact
# old+new pairs only: a seventh difference or any future edit must fail.
INTERFACE_DELTAS = {PREFIX + name: {"old_sha256": old, "new_sha256": new, "scope": scope}
    for name, old, new, scope in (
    ("canonical_binding.py", "0cc4a6132ef11de577316834959ccef66a51549fd46580a83864735b7d11a87e",
     "d983207f4689e70f8d41b94b52c5b33d9884ff20d7af13101f44b0745edd0ff9", "legacy_binding_oracle_metadata"),
    ("moving_frame_backend.py", "643788474e5c0360ff5c3e586c46a539ec38045060dfc2daf59d83f6ca8c64e3",
     "e47429cec5b865abf5944f884773d098266c13d76f34f3a545ac9b97ed840df4", "source_only_dispatch_and_JSON_time_metadata_not_source_formula"),
    ("moving_frame_binding.py", "12a4b9a262010e88a6a8a6357b9057a09b8eb3ce887a22bcea4f5710c18bae38",
     "33fffacd10aca0c70be5382b091bfeddadd3641d9f781a3ead19ddd1d071e4b4", "source_only_dispatch_identity"),
    ("perturbation_backend.py", "159f6dd36556a01c1e5f90906a33b06147e489c97a934115e3898181417dcdd9",
     "285b25aa3b402ca53a00d2e6551ea23412805e4bd51a76b1b8b03b9c3f166aae", "session_accounting_and_strong_legacy_source_path_not_weak_source_math"),
    ("public_problem.py", "711491a73663c462c4bc58e61704a6cb0e9bcd27e182431f52bcfdd1a3da4f9e",
     "e6e70d8bfb2cfdfbb1eb7e1dbb094661cf7073a3d839c0dcb0e6cacf864c6165", "public_oracle_contract_documentation"),
    ("tasks/observable.py", "8629b2ce70baed1f21362a50ed75590dcaf423a6e721f4ad4f1a44d666cfee0f",
     "f1e75e281c2604160f6d9e4498c5f83d88b854a8b6fa1bf022e25d814136f96f", "canonical_two_phase_candidate_execution_and_private_plans"))}
MODEL_GRAPH = tuple(PREFIX + name for name in (
    "innovation_diffusion.py", "temporal_innovation.py", "bulk_transport_innovation.py",
    "coherent_diffusion.py", "structured_diffusion.py", "value_control_diffusion.py",
    "moving_frame.py", "coherent_velocity_diffusion.py", "coherent_velocity_dynamics.py")) + (
    "scripts/vpfp_innovation_training_view.py", "scripts/train_vpfp_structured_diffusion.py",
    "scripts/vpfp_innovation_precision_gate.py")
# These three are registered by the strictly loaded conditioning context,
# not as individual paths in frozen_inputs. Keep that provenance explicit.
CONTEXT_GRAPH = MODEL_GRAPH[:7] + MODEL_GRAPH[-3:]
CONTEXT_ONLY_GRAPH = MODEL_GRAPH[-3:]

# A separate execution upgrade, not a rewrite of the immutable export proof.
# The model graph and original six export-time interface pairs stay unchanged.
RUNTIME_INTERFACE_PROFILE = "weak_fair_v4"
RUNTIME_INTERFACE_UPGRADES = {
    PREFIX + "moving_frame_backend.py": {
        "export_sha256": "e47429cec5b865abf5944f884773d098266c13d76f34f3a545ac9b97ed840df4",
        "runtime_sha256": "b0921b2ad19d322b52dfea33a5f03c8a4fa3e87aaec5d197ac751ec489fc79f6",
        "scope": "opt_in_batched_source_dispatch_original_math_and_legacy_default_preserved"},
    PREFIX + "tasks/observable.py": {
        "export_sha256": "f1e75e281c2604160f6d9e4498c5f83d88b854a8b6fa1bf022e25d814136f96f",
        "runtime_sha256": "4ee77dc4c28324e2cf1957908fb13f630e847bf39e8e583b3e3b1b59f38cb17a",
        "scope": "opt_in_queue_aware_candidate_clock_no_score_or_truth_change"},
}
RUNTIME_ADDED_SOURCES = {
    PREFIX + "moving_frame_oracle_options.py":
        "1a75136053c3dd9e8c0ab8ff1b1debe761439de0b56fbc4089edfea542b3263b",
}
NS900_RUNTIME_INTERFACE_PROFILE = "weak_fair_ns900_v1"
NS900_RUNTIME_INTERFACE_UPGRADES = {
    **deepcopy(RUNTIME_INTERFACE_UPGRADES),
    PREFIX + "tasks/observable.py": {
        "export_sha256": "f1e75e281c2604160f6d9e4498c5f83d88b854a8b6fa1bf022e25d814136f96f",
        "runtime_sha256": "3e6565448bd4d891e4baf9d051cf831f30f95eb5d5e528d1fc388480d95d4800",
        "scope": "opt_in_NS_remaining_case_budget_no_separate_prepare_limit_no_score_or_truth_change"},
}
# The accepted export and both older execution registrations stay immutable.
# This opt-in profile reuses exact initial-value geometry while preserving the
# complete carrier, decoder/model graph, physical source and NS900 task. New
# development-pool code is outside the export inventory and is bound by the
# fresh Campaign's full runtime source fingerprint, not by a model-proof delta.
ORACLE_FAST_RUNTIME_INTERFACE_PROFILE = "weak_oracle_fast_v1"
ORACLE_FAST_RUNTIME_INTERFACE_UPGRADES = {
    **deepcopy(NS900_RUNTIME_INTERFACE_UPGRADES),
    PREFIX + "moving_frame_backend.py": {
        "export_sha256": "e47429cec5b865abf5944f884773d098266c13d76f34f3a545ac9b97ed840df4",
        "runtime_sha256": "59365933f6beb5be4cd0f2490a79a60cff3f8000bf66468e9a4ec3a5ebd1390c",
        "scope": "opt_in_exact_g0_value_geometry_reuse_source_math_and_legacy_dispatch_preserved"},
}
ORACLE_FAST_RUNTIME_ADDED_SOURCES = {
    PREFIX + "moving_frame_oracle_options.py":
        "dd4798e57f63f97ebe906eccbe51d94e411508591d721227598f4a5c6241742c",
}


def canonical_json_sha(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _file_sha(path):
    path = resolve_asset_path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("regular immutable provenance file required: " + str(path))
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_source(root):
    root = resolve_asset_path(root)
    if (Path(root) / "EXECUTION_SOURCE.json").exists():
        # This verifier is standard-library-only. Load it directly to preserve
        # the provenance helper's numerical-import isolation contract.
        import importlib.util
        verifier = Path(__file__).resolve().parents[1] / "src/pde_agent_verifier_solver/engine/execution_source.py"
        spec = importlib.util.spec_from_file_location("_weak_snapshot_provenance", verifier)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.execution_source_identity(root, require_clean=True)["source_sha256"]
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    revision = git("rev-parse", "HEAD")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("portable provenance requires a clean committed source root")
    return revision


def _context_model_graph(receipt, old_root, new_root, relocated):
    """Follow the original context asset's registered graph, without inventing paths."""
    identity = receipt["sampler_identity"]
    asset = identity["assets"]["conditioning_context"]
    path = Path(asset["path"])
    if (set(asset) != {"path", "sha256"} or not path.is_absolute()
            or receipt["frozen_inputs"].get(str(path)) != asset["sha256"]
            or _file_sha(path) != asset["sha256"]):
        raise ValueError("original conditioning context must be a frozen input")
    context = json.loads(resolve_asset_path(path).read_text())
    view = context["training_view_binding"]
    if (context.get("schema") != "vpfp_innovation_conditioning_context_v2"
            or view.get("schema") != "vpfp_complete_innovation_training_view_binding_v2"
            or canonical_json_sha(view) != identity["training_view_binding_sha256"]
            or _file_sha(path) != asset["sha256"]):
        raise ValueError("original conditioning context graph binding changed")
    registered = view["source_graph_sha256"]
    if set(registered) != set(CONTEXT_GRAPH):
        raise ValueError("complete original conditioning context source graph required")
    supplemental = {}
    for relative, expected in registered.items():
        if (_file_sha(old_root / relative) != expected
                or _file_sha(new_root / relative) != expected):
            raise ValueError("original context-registered model source changed: " + relative)
        if relative in relocated:
            if relocated[relative]["old_sha256"] != expected:
                raise ValueError("direct/context model source registration differs: " + relative)
        else:
            supplemental[relative] = {"old_path": str(old_root / relative),
                "new_path": str(new_root / relative), "sha256": expected}
    if set(supplemental) != set(CONTEXT_ONLY_GRAPH):
        raise ValueError("exact context-only helper inventory required")
    return registered, {"context_asset": dict(asset),
        "training_view_binding_sha256": identity["training_view_binding_sha256"],
        "sources": supplemental}


def _proof(old_root, new_root, receipt, *, interface_deltas=None):
    deltas = INTERFACE_DELTAS if interface_deltas is None else interface_deltas
    old_root, new_root = Path(old_root).absolute(), Path(new_root).absolute()
    if old_root != OLD_ROOT or old_root == new_root or resolve_asset_path(old_root).is_symlink() or new_root.is_symlink():
        raise ValueError("explicit distinct pinned old and current source roots required")
    if canonical_json_sha(receipt) != ORIGIN_RECEIPT_SHA256:
        raise ValueError("fresh strict-load receipt differs from original accepted Base")
    origin = Path(ORIGIN_RECEIPT_ASSET["path"])
    if (_file_sha(origin) != ORIGIN_RECEIPT_ASSET["sha256"]
            or canonical_json_sha(json.loads(resolve_asset_path(origin).read_text())) != ORIGIN_RECEIPT_SHA256):
        raise ValueError("original accepted Base receipt anchor changed")
    files = receipt["frozen_inputs"]
    if len(files) != INPUT_COUNT or canonical_json_sha(files) != receipt["frozen_input_sha256"]:
        raise ValueError("complete original frozen-input inventory required")
    relocated = {}
    for raw, expected in sorted(files.items()):
        old = Path(raw)
        if not old.is_absolute() or ".." in old.parts or _file_sha(old) != expected:
            raise ValueError("original frozen input changed: " + raw)
        if not old.is_relative_to(old_root):
            continue
        relative = str(old.relative_to(old_root))
        current = _file_sha(new_root / relative)
        delta = deltas.get(relative)
        if delta is None:
            if current != expected:
                raise ValueError("unregistered physical/helper source difference: " + relative)
        elif (expected, current) != (delta["old_sha256"], delta["new_sha256"]):
            raise ValueError("registered interface old/new SHA pair differs: " + relative)
        relocated[relative] = {"old_sha256": expected, "new_sha256": current,
            "scope": "byte_identical" if delta is None else delta["scope"]}
    if (len(relocated) != RELOCATED_COUNT or not set(deltas) <= set(relocated)
            or set(MODEL_GRAPH) - set(relocated) != set(CONTEXT_ONLY_GRAPH)
            or set(MODEL_GRAPH) & set(deltas)):
        raise ValueError("complete relocated source/model/interface inventory required")
    registered, supplemental = _context_model_graph(receipt, old_root, new_root, relocated)
    graph = {name: registered[name] if name in registered else relocated[name]["old_sha256"]
             for name in MODEL_GRAPH}
    interfaces = {name: relocated[name]["new_sha256"] for name in deltas}
    return relocated, dict(execution_passed=True, origin_receipt_sha256=ORIGIN_RECEIPT_SHA256,
        checked_input_count=len(files), relocated_source_count=len(relocated),
        source_graph_sha256=graph, supplemental_model_sources=supplemental,
        runtime_interface_sha256=interfaces,
        interface_delta_count=len(interfaces), physical_admission=False)


def create_portability_provenance(old_root, new_root, receipt):
    old_revision, new_revision = _git_source(old_root), _git_source(new_root)
    if old_revision != OLD_REVISION:
        raise ValueError("original strict loader must use pinned48dacc3 source")
    inventory, checks = _proof(old_root, new_root, receipt)
    if (_git_source(old_root), _git_source(new_root)) != (old_revision, new_revision):
        raise ValueError("source revision changed during provenance creation")
    return dict(schema=SCHEMA, old_source_root=str(Path(old_root).absolute()),
        new_source_root=str(Path(new_root).absolute()), old_source_revision=old_revision,
        new_source_revision=new_revision, origin_receipt_asset=dict(ORIGIN_RECEIPT_ASSET),
        origin_receipt_sha256=ORIGIN_RECEIPT_SHA256,
        origin_receipt=json.loads(json.dumps(receipt, allow_nan=False)),
        relocated_sources=inventory, checks=checks, historical_loaders_modified=False,
        sampler_identity_changed=False, scientific_admission=False)


def validate_portability_provenance(provenance, new_root):
    fields = {"schema", "old_source_root", "new_source_root", "old_source_revision", "new_source_revision",
        "origin_receipt_asset", "origin_receipt_sha256", "origin_receipt", "relocated_sources", "checks",
        "historical_loaders_modified", "sampler_identity_changed", "scientific_admission"}
    if (not isinstance(provenance, dict) or set(provenance) != fields or provenance["schema"] != SCHEMA
            or provenance["old_source_revision"] != OLD_REVISION
            or provenance["origin_receipt_asset"] != ORIGIN_RECEIPT_ASSET
            or provenance["origin_receipt_sha256"] != ORIGIN_RECEIPT_SHA256
            or any(provenance[key] is not False for key in (
                "historical_loaders_modified", "sampler_identity_changed", "scientific_admission"))):
        raise ValueError("explicit pinned portable provenance schema required")
    observed = provenance["new_source_revision"]
    if not isinstance(observed, str) or len(observed) != 40 or any(c not in "0123456789abcdef" for c in observed):
        raise ValueError("export-time current source revision required")
    old_root = Path(provenance["old_source_root"])
    before = (_git_source(old_root), _git_source(new_root))
    if before[0] != OLD_REVISION:
        raise ValueError("original strict-load source revision changed")
    inventory, checks = _proof(old_root, new_root, provenance["origin_receipt"])
    expected_checks = provenance["checks"]
    relocation = None
    if not isinstance(provenance["new_source_root"], str):
        raise ValueError("absolute export source root string required")
    export_root = Path(provenance["new_source_root"])
    current_root = Path(new_root).absolute()
    if (not export_root.is_absolute() or ".." in export_root.parts
            or resolve_asset_path(export_root).is_symlink() or not resolve_asset_path(export_root).is_dir()):
        raise ValueError("regular absolute export source root required")
    if export_root != current_root:
        # The immutable export binds three context-only helper locators to
        # its original canonical checkout. _proof above has already checked
        # all original inputs and all current physical/helper source bytes.
        # Relocate only these declared path strings for exact comparison;
        # neither the stored provenance nor any model asset is rewritten.
        expected_checks = deepcopy(expected_checks)
        supplemental = expected_checks.get("supplemental_model_sources") if isinstance(expected_checks, dict) else None
        sources = supplemental.get("sources") if isinstance(supplemental, dict) else None
        if not isinstance(sources, dict) or set(sources) != set(CONTEXT_ONLY_GRAPH):
            raise ValueError("exact supplemental locator inventory required")
        paths = {}
        for relative in CONTEXT_ONLY_GRAPH:
            row = sources[relative]
            if (not isinstance(row, dict) or set(row) != {"old_path", "new_path", "sha256"}
                    or any(not isinstance(value, str) for value in row.values())
                    or row["new_path"] != str(export_root / relative)):
                raise ValueError("supplemental export locator changed: " + relative)
            paths[relative] = {"export_path": row["new_path"],
                "current_path": str(current_root / relative), "sha256": row["sha256"]}
            row["new_path"] = str(current_root / relative)
        relocation = dict(schema="vpfp_weak_portable_source_relocation_v1",
            export_source_root=str(export_root), current_source_root=str(current_root),
            export_source_revision=observed, current_source_revision=before[1],
            original_provenance_sha256=canonical_json_sha(provenance), path_fields=paths,
            original_provenance_changed=False, model_source_bytes_changed=False)
    if (inventory != provenance["relocated_sources"] or checks != expected_checks
            or canonical_json_sha(checks) != canonical_json_sha(expected_checks)):
        raise ValueError("portable provenance inventory/checks changed")
    if (_git_source(old_root), _git_source(new_root)) != before:
        raise ValueError("source revision changed during provenance validation")
    # Current whole-Run source identity is independently frozen by the formal
    # entry point; unrelated commits do not redefine physical model identity.
    return checks if relocation is None else {**checks, "source_root_relocation": relocation}


def validate_runtime_portability_provenance(provenance, new_root, profile):
    """Validate the unchanged historical proof, then an exact execution upgrade.

    This permits only two predeclared interface byte transitions. It does not
    qualify their numerical equivalence: the independent native GPU preflight
    must still pass before a formal run can use this execution profile.
    """
    if profile == RUNTIME_INTERFACE_PROFILE:
        upgrades = RUNTIME_INTERFACE_UPGRADES
        added_sources = RUNTIME_ADDED_SOURCES
    elif profile == NS900_RUNTIME_INTERFACE_PROFILE:
        upgrades = NS900_RUNTIME_INTERFACE_UPGRADES
        added_sources = RUNTIME_ADDED_SOURCES
    elif profile == ORACLE_FAST_RUNTIME_INTERFACE_PROFILE:
        upgrades = ORACLE_FAST_RUNTIME_INTERFACE_UPGRADES
        added_sources = ORACLE_FAST_RUNTIME_ADDED_SOURCES
    else:
        raise ValueError("unknown weak runtime interface profile")
    if not isinstance(provenance, dict) or not isinstance(provenance.get("new_source_root"), str):
        raise ValueError("original portable export provenance required")
    original_sha = canonical_json_sha(provenance)
    export_root = resolve_asset_path(Path(provenance["new_source_root"]))
    # Retain all original receipt, asset, model, interface and inventory checks.
    original_checks = validate_portability_provenance(provenance, export_root)
    new_root = Path(new_root).absolute()
    old_root = Path(provenance["old_source_root"])
    before = (_git_source(old_root), _git_source(export_root), _git_source(new_root))
    if before[0] != OLD_REVISION:
        raise ValueError("original strict-load source revision changed")
    deltas = deepcopy(INTERFACE_DELTAS)
    if (not upgrades
            or not set(upgrades) <= set(deltas)
            or set(upgrades) & set(MODEL_GRAPH)):
        raise ValueError("runtime upgrades must be predeclared non-model interfaces")
    for relative, upgrade in upgrades.items():
        if (set(upgrade) != {"export_sha256", "runtime_sha256", "scope"}
                or upgrade["export_sha256"] != deltas[relative]["new_sha256"]
                or _file_sha(export_root / relative) != upgrade["export_sha256"]):
            raise ValueError("runtime interface export anchor differs: " + relative)
        deltas[relative].update(new_sha256=upgrade["runtime_sha256"], scope=upgrade["scope"])
    inventory, checks = _proof(old_root, new_root, provenance["origin_receipt"],
        interface_deltas=deltas)
    for relative, expected in added_sources.items():
        path = Path(relative)
        if (path.is_absolute() or ".." in path.parts or relative in inventory or relative in MODEL_GRAPH
                or _file_sha(new_root / relative) != expected):
            raise ValueError("runtime added source differs: " + relative)
    if (canonical_json_sha(provenance) != original_sha
            or (_git_source(old_root), _git_source(export_root), _git_source(new_root)) != before
            or checks["source_graph_sha256"] != original_checks["source_graph_sha256"]):
        raise ValueError("provenance or physical source changed during runtime validation")
    return {**checks, "runtime_interface_upgrade": {
        "schema": "vpfp_weak_runtime_interface_upgrade_v1", "profile": profile,
        "export_source_root": str(export_root), "current_source_root": str(new_root),
        "export_source_revision": before[1], "current_source_revision": before[2],
        "original_provenance_sha256": original_sha,
        "original_provenance_validated": True, "original_provenance_changed": False,
        "model_source_bytes_changed": False, "numerical_equivalence_qualified": False,
        "interfaces": deepcopy(upgrades),
        "added_sources": dict(added_sources)}}
