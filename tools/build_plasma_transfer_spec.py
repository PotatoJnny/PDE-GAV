#!/usr/bin/env python3
"""Describe the frozen six-arm deployment's files without loading or hashing models.

This inventories dependencies, not a runnable cross-cluster deployment. Original
evidence bytes remain immutable; target paths, scheduler, GPU/runtime compatibility
and native acceptance must still be checked separately. No secrets or Git history
are copied. Large-file checksums are delegated to plasma_transfer_manifest seal.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import stat
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import plasma_transfer_manifest as inventory

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "dbdd6c91ea95273480c937eea1fb5615ae4ac86d"
RUNTIME_SHA = "13167b35b7f31ec79358bae623229a0bd43dd01364805af5d16e6c0c12f01962"
WEAK_EXPORT_RELOCATION = ("73fea1b947d2c6691502aa02ed4eb4f29eb515c1",
                          "7301ac3675d2464699d168a30147eb17869dc405")
ARMS = tuple(f"{family}-{model}" for family in ("strong", "weak", "artemis") for model in ("sol", "astra"))


def read_json(path):
    path = inventory._path(str(path))
    info = inventory._checked(path)
    if not stat.S_ISREG(info.st_mode) or info.st_size > 16 * 1024 * 1024:
        raise ValueError("only small regular JSON documents may be read: " + str(path))
    return json.loads(path.read_text())


def git_head(root):
    inventory._checked(inventory._path(str(root)))
    value = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Git HEAD must be a full commit ID")
    return value


def bound_files(value):
    """Yield only explicitly paired locator/hash fields, never arbitrary paths."""
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            yield value["path"], value["sha256"]
        for key, item in value.items():
            if key.endswith("_path") and key[:-5] + "_sha256" in value:
                yield item, value[key[:-5] + "_sha256"]
            yield from bound_files(item)
    elif isinstance(value, list):
        for item in value:
            yield from bound_files(item)


def resolve_path(value, mapping):
    logical = inventory._path(str(value))
    if mapping.get("schema") != "vpfp-weak-explicit-path-map-v1":
        raise ValueError("unsupported weak path-map schema")
    matches = []
    for row in mapping["paths"]:
        old = inventory._path(row["source"])
        new = inventory._path(row["destination"])
        if logical == old or (row.get("directory") is True and logical.is_relative_to(old)):
            matches.append((len(old.parts), new / logical.relative_to(old)))
    if matches:
        matches.sort(key=lambda item: item[0], reverse=True)
        if len(matches) > 1 and matches[0][0] == matches[1][0]:
            raise ValueError("ambiguous weak path mapping")
        return matches[0][1]
    if str(logical).startswith(("/scratch/", "/project/", "/home/")):
        raise ValueError("unmapped original-account dependency: " + str(logical))
    return logical


class Builder:
    def __init__(self, commit_lookup=git_head):
        self.files = {}
        self.repositories = {}
        self.commit_lookup = commit_lookup

    def add(self, path, role, digest=None):
        path = inventory._path(str(path))
        info = inventory._checked(path)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("asset must be a regular file: " + str(path))
        if digest is not None:
            inventory._digest(digest)
        key = str(path)
        if key in self.files:
            old = self.files[key]
            if digest is not None and old.get("sha256") not in (None, digest):
                raise ValueError("conflicting SHA256 pins: " + key)
            if digest is not None:
                old["sha256"] = digest
            old["roles"] = sorted(set(old["roles"]) | {role})
            return
        self.files[key] = dict(source=key, destination="fs/" + key.lstrip("/"),
                               role=role, roles=[role], bytes=info.st_size)
        if digest is not None:
            self.files[key]["sha256"] = digest

    def tree(self, root, role):
        root = inventory._path(str(root))
        if not stat.S_ISDIR(inventory._checked(root).st_mode):
            raise ValueError("evidence root must be a regular directory")
        for path in sorted(root.iterdir()):
            info = inventory._checked(path)
            if path.name in inventory.CACHES:
                continue
            if stat.S_ISDIR(info.st_mode):
                self.tree(path, role)
            else:
                self.add(path, role)

    def repository(self, path, role, expected):
        path = inventory._path(str(path))
        head = self.commit_lookup(path)
        key = str(path)
        if head != expected and not (role == "weak-export" and (expected, head) == WEAK_EXPORT_RELOCATION):
            raise ValueError("unregistered historical Git revision difference: " + key)
        row = dict(path=key, repository="${PDE_GAV_PHYSICAL_REPOSITORY}",
                   head=head, snapshot_revision=head, historical_recorded_revision=expected, role=role, source_files=[],
                   action="export and verify exact shallow Git snapshot; do not copy .git or entire checkout")
        if head != expected:
            row["revision_difference"] = {
                "kind": "existing accepted weak export relocation; not a new qualification",
                "verification": "scripts/vpfp_weak_portability_provenance.py validate_portability_provenance checks source bytes and unchanged original proof",
                "preserve_historical_record": True}
        if key in self.repositories and self.repositories[key]["historical_recorded_revision"] != expected:
            raise ValueError("conflicting required Git revisions")
        self.repositories.setdefault(key, row)
        return self.repositories[key]

    def weak(self, backend):
        portable = backend["portable_base"]
        self.add(portable["path"], "weak portable inference model", portable["sha256"])
        locator = backend["historical_path_map"]
        self.add(locator["path"], "weak immutable locator map", locator["sha256"])
        mapping = read_json(locator["path"])
        receipt_path = Path(portable["path"]).parent / "receipt.json"
        self.add(receipt_path, "weak portable export receipt")
        receipt = read_json(receipt_path)
        provenance = receipt["provenance"]
        for field in ("old", "new"):
            self.repository(resolve_path(provenance[field + "_source_root"], mapping),
                            "weak-origin" if field == "old" else "weak-export",
                            provenance[field + "_source_revision"])
        runtime = backend["historical_runtime_root"]
        self.repository(runtime["path"], "weak-runtime", runtime["revision"])
        for key in ("original_base_receipt", "portable"):
            item = receipt[key]
            self.add(resolve_path(item["path"], mapping), "weak export " + key, item["sha256"])
        origin_asset = provenance["origin_receipt_asset"]
        origin_path = resolve_path(origin_asset["path"], mapping)
        self.add(origin_path, "weak origin strict-load receipt", origin_asset["sha256"])
        origin = read_json(origin_path)
        frozen = origin["frozen_inputs"]
        if len(frozen) != 559:
            raise ValueError("current weak freeze must contain exactly 559 frozen_inputs")
        for original, digest in frozen.items():
            current = resolve_path(original, mapping)
            owners = [row for root, row in self.repositories.items() if current.is_relative_to(root)]
            if len(owners) > 1:
                raise ValueError("ambiguous Git ownership of frozen input")
            if owners:
                info = inventory._checked(current)
                if not stat.S_ISREG(info.st_mode):
                    raise ValueError("frozen Git source must be regular")
                inventory._digest(digest)
                owners[0]["source_files"].append(dict(path=str(current.relative_to(owners[0]["path"])),
                                                       original_path=original, sha256=digest))
            else:
                self.add(current, "weak frozen input required by historical proof", digest)
        for relative, pins in provenance.get("relocated_sources", {}).items():
            relative_path = inventory._path(relative, absolute=False)
            for field in ("old", "new"):
                logical = Path(provenance[field + "_source_root"]) / relative_path
                current = resolve_path(logical, mapping)
                root = str(resolve_path(provenance[field + "_source_root"], mapping))
                if not stat.S_ISREG(inventory._checked(current).st_mode):
                    raise ValueError("relocated scientific source must be regular")
                digest = inventory._digest(pins[field + "_sha256"])
                builder_row = self.repositories[root]
                builder_row["source_files"].append(dict(path=relative, original_path=str(logical), sha256=digest))
        supplemental = provenance.get("checks", {}).get("supplemental_model_sources", {}).get("sources", {})
        for row in supplemental.values():
            for key in ("old_path", "new_path"):
                current = resolve_path(row[key], mapping)
                owners = [item for root, item in self.repositories.items() if current.is_relative_to(root)]
                if len(owners) != 1 or not stat.S_ISREG(inventory._checked(current).st_mode):
                    raise ValueError("supplemental model helper must belong to an exact Git snapshot")
                inventory._digest(row["sha256"])
                owners[0]["source_files"].append(dict(path=str(current.relative_to(owners[0]["path"])),
                                                       original_path=row[key], sha256=row["sha256"]))


def build(config_directory, output, runtime_directory=None, *, code_root=ROOT, commit_lookup=git_head):
    config_directory = Path(config_directory).absolute()
    output = Path(output).absolute()
    inventory._path(str(output))
    inventory._checked(output.parent)
    if output.exists() or output.is_symlink():
        raise FileExistsError("output must be a new directory")
    builder = Builder(commit_lookup)
    configs = {}
    validation_roots = set()
    for arm in ARMS:
        path = config_directory / (arm + ".json")
        config = read_json(path)
        if config["equation"] != arm.rsplit("-", 1)[0]:
            raise ValueError("arm filename and equation differ")
        configs[arm] = config
        builder.add(path, "current six-arm configuration")
        source = config["source_experiment"]
        builder.add(source["config"], "original experiment configuration", source["sha256"])
        validation_roots.add(config["transport_qualification"]["validation_root"])
    builder.add(config_directory / "migration.json", "six-arm preparation receipt")
    if len(validation_roots) != 1:
        raise ValueError("six arms must share one frozen validation root")
    validation = Path(next(iter(validation_roots)))
    builder.tree(validation, "immutable transport-v2 calibration and native acceptance evidence")
    for family in ("strong", "weak", "artemis"):
        accepted = read_json(validation / ("acceptance-" + family) / "report.json")
        builder.add(accepted["config_path"], "accepted reference configuration", accepted["config_sha256"])
    for arm, config in configs.items():
        backend = config["backend"]
        if config["equation"] == "strong":
            for key in ("base_checkpoint", "portable_codec"):
                builder.add(backend[key], "strong " + key, backend[key + "_sha256"])
            for kind in ("base", "expanded"):
                native = backend["native_spec"]
                builder.add(native[kind + "_decoder_checkpoint"], "strong " + kind + " decoder",
                            native[kind + "_decoder_sha256"])
        elif config["equation"] == "weak":
            builder.weak(backend)
        else:
            settings = backend["factory_settings"]
            for key, digest_key in (("admission_record_path", "expected_admission_record_sha256"),
                                    ("science_config_path", "expected_science_config_sha256")):
                builder.add(settings[key], "Artemis " + key, settings[digest_key])
            admission = read_json(settings["admission_record_path"])
            for path, digest in bound_files(admission["evidence_bindings"]):
                builder.add(path, "Artemis frozen admission dependency", digest)
            builder.add(backend["source_config"]["path"], "Artemis original backend configuration",
                        backend["source_config"]["sha256"])
            row = builder.repository(settings["physical_checkout"], "artemis-physical",
                                     settings["expected_physical_revision"])
            row["required_source_paths"] = ["src/pde_agent_verifier_solver/plugins/vp_1d1v",
                                             "scripts/train_vp_1d1v_artemis_compact_pca_v5.py"]
    if runtime_directory is not None:
        runtime = Path(runtime_directory).absolute()
        for name in ("bin/codex", "bin/codex-code-mode-host", "codex-package.json",
                     "codex-path/rg", "codex-resources/bwrap"):
            builder.add(runtime / name, "optional pinned native Agent runtime",
                        RUNTIME_SHA if name == "bin/codex" else None)
    revision = commit_lookup(Path(code_root))
    entries = [builder.files[key] for key in sorted(builder.files)]
    repositories = [builder.repositories[key] for key in sorted(builder.repositories)]
    for row in repositories:
        # Sol/Astra may share precisely the same frozen source evidence.
        by_path = {}
        for item in row["source_files"]:
            if item["path"] in by_path and by_path[item["path"]]["sha256"] != item["sha256"]:
                raise ValueError("conflicting frozen Git source SHA256 pins")
            by_path[item["path"]] = item
        row["source_files"] = sorted(by_path.values(), key=lambda item: item["path"])
    spec = dict(schema="plasma_transfer_spec_v1", code_repository="https://github.com/PotatoJnny/PDE-GAV",
                code_revision=revision, scientific_baseline_revision=BASELINE, entries=entries)
    evidence = dict(schema="plasma_git_evidence_v1", repositories=repositories,
                    copied=False, qualification_granted=False)
    summary = dict(schema="plasma_transfer_spec_summary_v1", file_count=len(entries),
                   total_bytes=sum(row["bytes"] for row in entries), git_repository_count=len(repositories),
                   pinned_file_count=sum("sha256" in row for row in entries),
                   scientific_baseline_revision=BASELINE, code_revision=revision,
                   file_contents_verified=False, target_runtime_qualified=False,
                   warnings=["Metadata-only inventory: seal and verify checksums before use.",
                             "Git snapshots use snapshot_revision, not historical_recorded_revision; an existing weak export relocation is recorded separately.",
                             "Original absolute paths in evidence/checkpoints are not rewritten.",
                             "Target relocation, GPU/runtime and native qualification remain required.",
                             "Do not copy credentials, old campaigns, .git directories or SSH keys."])
    output.mkdir(mode=0o700)
    for filename, value in (("spec.json", spec), ("git-evidence.json", evidence), ("summary.json", summary)):
        inventory._write_new(output / filename, inventory._json(value))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-directory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--runtime-directory")
    args = parser.parse_args()
    print(json.dumps(build(args.config_directory, args.output, args.runtime_directory), indent=2))


if __name__ == "__main__":
    main()
