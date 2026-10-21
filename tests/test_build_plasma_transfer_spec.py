"""Metadata-only, synthetic transfer-spec tests: no real scientific assets."""
from copy import deepcopy
import json
from pathlib import Path
import stat

import pytest

from tools import build_plasma_transfer_spec as api

SHA = "a" * 64
REV = "b" * 40


def file(path, value=b"fixture"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


def document(path, value):
    return file(path, json.dumps(value).encode())


def fixture_configs(root):
    configs = root / "configs"
    configs.mkdir()
    validation = root / "validation"
    artifact = file(root / "asset.pt")
    portable = file(root / "portable/native-inference-base.pt")
    origin_root = root / "git-origin"
    export_root = root / "git-export"
    runtime_root = root / "git-runtime"
    artemis_root = root / "git-artemis"
    for path in (origin_root, export_root, runtime_root, artemis_root):
        path.mkdir()
    git_source = file(origin_root / "src/frozen.py")
    frozen = {str(git_source): SHA}
    mappings = []
    for number in range(558):
        logical = str(root / "historical" / str(number))
        frozen[logical] = SHA
        mappings.append(dict(source=logical, destination=str(artifact), directory=False))
    origin = document(root / "origin.json", {"frozen_inputs": frozen})
    mapping = document(root / "path-map.json", {"schema": "vpfp-weak-explicit-path-map-v1", "paths": mappings})
    provenance = dict(old_source_root=str(origin_root), new_source_root=str(export_root),
                      old_source_revision=REV, new_source_revision=REV,
                      origin_receipt_asset=dict(path=str(origin), sha256=SHA))
    document(portable.parent / "receipt.json", dict(provenance=provenance,
        original_base_receipt=dict(path=str(origin), sha256=SHA),
        original_reference=dict(path=str(artifact), sha256=SHA),
        portable=dict(path=str(portable), sha256=SHA)))
    admission = document(root / "admission.json", {"evidence_bindings": {
        "bundle": {"checkpoint_path": str(artifact), "checkpoint_sha256": SHA}}})
    science = document(root / "science.json", {})
    source_backend = document(root / "backend.json", {})
    for arm in api.ARMS:
        family = arm.rsplit("-", 1)[0]
        old = document(root / "old" / (arm + ".json"), {})
        config = dict(equation=family, source_experiment=dict(config=str(old), sha256=SHA),
                      transport_qualification=dict(validation_root=str(validation)))
        if family == "strong":
            config["backend"] = dict(base_checkpoint=str(artifact), base_checkpoint_sha256=SHA,
                portable_codec=str(artifact), portable_codec_sha256=SHA, native_spec=dict(
                    base_decoder_checkpoint=str(artifact), base_decoder_sha256=SHA,
                    expanded_decoder_checkpoint=str(artifact), expanded_decoder_sha256=SHA))
        elif family == "weak":
            config["backend"] = dict(portable_base=dict(path=str(portable), sha256=SHA),
                historical_path_map=dict(path=str(mapping), sha256=SHA),
                historical_runtime_root=dict(path=str(runtime_root), revision=REV))
        else:
            config["backend"] = dict(source_config=dict(path=str(source_backend), sha256=SHA), factory_settings=dict(
                admission_record_path=str(admission), expected_admission_record_sha256=SHA,
                science_config_path=str(science), expected_science_config_sha256=SHA,
                physical_checkout=str(artemis_root), expected_physical_revision=REV))
        document(configs / (arm + ".json"), config)
        if arm.endswith("-sol"):
            document(validation / ("acceptance-" + family) / "report.json", dict(config_path=str(old), config_sha256=SHA))
    document(configs / "migration.json", {})
    document(validation / "calibration.json", {})
    document(validation / "validation.json", {})
    return configs, git_source


def test_build_real_shape_metadata_only_and_git_exclusion(tmp_path, monkeypatch):
    configs, git_source = fixture_configs(tmp_path)
    monkeypatch.setattr(api.inventory, "_hash_file", lambda *a, **k: pytest.fail("must not hash large assets"))
    summary = api.build(configs, tmp_path / "out", code_root=tmp_path, commit_lookup=lambda _: REV)
    assert summary["git_repository_count"] == 4
    assert summary["target_runtime_qualified"] is False
    assert summary["file_contents_verified"] is False
    spec = json.loads((tmp_path / "out/spec.json").read_text())
    assert spec["scientific_baseline_revision"] == api.BASELINE
    assert spec["code_revision"] == REV
    assert len({item["source"] for item in spec["entries"]}) == len(spec["entries"])
    assert str(git_source) not in {item["source"] for item in spec["entries"]}
    evidence = json.loads((tmp_path / "out/git-evidence.json").read_text())
    origin = next(row for row in evidence["repositories"] if row["role"] == "weak-origin")
    assert len(origin["source_files"]) == 1
    assert origin["source_files"][0]["path"] == "src/frozen.py"
    assert stat.S_IMODE((tmp_path / "out").stat().st_mode) == 0o700
    with pytest.raises(FileExistsError):
        api.build(configs, tmp_path / "out", code_root=tmp_path, commit_lookup=lambda _: REV)


def test_explicit_runtime_allowlist_excludes_adjacent_secrets(tmp_path):
    configs, _ = fixture_configs(tmp_path)
    runtime = tmp_path / "runtime"
    names = ("bin/codex", "bin/codex-code-mode-host", "codex-package.json", "codex-path/rg", "codex-resources/bwrap")
    for name in names:
        file(runtime / name)
    file(runtime / "auth.json", b"must not be read")
    api.build(configs, tmp_path / "out", runtime, code_root=tmp_path, commit_lookup=lambda _: REV)
    entries = json.loads((tmp_path / "out/spec.json").read_text())["entries"]
    runtime_entries = [row for row in entries if row["source"].startswith(str(runtime) + "/")]
    assert len(runtime_entries) == 5
    codex = next(row for row in runtime_entries if row["source"].endswith("/bin/codex"))
    assert codex["sha256"] == api.RUNTIME_SHA
    assert not any("auth.json" in row["source"] for row in entries)


def test_longest_prefix_and_ambiguous_path_map(tmp_path):
    mapping = {"schema": "vpfp-weak-explicit-path-map-v1", "paths": [
        dict(source=str(tmp_path / "old"), destination=str(tmp_path / "new"), directory=True),
        dict(source=str(tmp_path / "old/deep"), destination=str(tmp_path / "override"), directory=True)]}
    assert api.resolve_path(tmp_path / "old/deep/x", mapping) == tmp_path / "override/x"
    mapping["paths"].append(deepcopy(mapping["paths"][-1]))
    with pytest.raises(ValueError, match="ambiguous"):
        api.resolve_path(tmp_path / "old/deep/x", mapping)


def test_bound_file_pairs_only():
    assert list(api.bound_files({"root": {"path": "/a", "sha256": SHA},
        "other": [{"model_path": "/b", "model_sha256": SHA}], "directory": "/not-opened"})) == [("/a", SHA), ("/b", SHA)]


def test_exact_export_relocation_preserves_recorded_revision(tmp_path):
    historical, deployed = api.WEAK_EXPORT_RELOCATION
    builder = api.Builder(lambda _: deployed)
    row = builder.repository(tmp_path, "weak-export", historical)
    assert row["snapshot_revision"] == deployed
    assert row["historical_recorded_revision"] == historical
    assert row["revision_difference"]["preserve_historical_record"] is True
    with pytest.raises(ValueError, match="unregistered"):
        builder.repository(tmp_path, "weak-export", REV)
    with pytest.raises(ValueError, match="unregistered"):
        builder.repository(tmp_path, "artemis-physical", historical)


def test_duplicate_conflicting_pins_fail_and_symlinks_fail(tmp_path):
    path = file(tmp_path / "weights.pt")
    builder = api.Builder(lambda _: REV)
    builder.add(path, "a", SHA)
    builder.add(path, "b", SHA)
    assert len(builder.files) == 1
    with pytest.raises(ValueError, match="conflicting SHA256"):
        builder.add(path, "c", "c" * 64)
    link = tmp_path / "link.pt"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="symlink"):
        builder.add(link, "no")


def test_missing_runtime_fails_before_output_and_json_size_is_bounded(tmp_path):
    configs, _ = fixture_configs(tmp_path)
    with pytest.raises(FileNotFoundError):
        api.build(configs, tmp_path / "out", tmp_path / "missing", code_root=tmp_path, commit_lookup=lambda _: REV)
    assert not (tmp_path / "out").exists()
    big = file(tmp_path / "big.json", b"x" * (16 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match="small regular JSON"):
        api.read_json(big)
