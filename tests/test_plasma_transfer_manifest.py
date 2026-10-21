"""Small synthetic files only: no real assets, credentials, transfer or Slurm."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil
import stat
import subprocess
import sys

import pytest

from tools import plasma_transfer_manifest as api


def write_json(path, value):
    path.write_text(json.dumps(value))
    return path


def spec_for(tmp_path, source, **entry_fields):
    entry = {"source": str(source), "destination": "fs/" + str(source).lstrip("/"), "role": "synthetic asset"}
    entry.update(entry_fields)
    return write_json(tmp_path / "spec.json", {"schema": "plasma_transfer_spec_v1",
        "code_repository": "https://github.com/example/repo.git", "code_revision": "a" * 40, "entries": [entry]})


def fixture_plan(tmp_path):
    source = tmp_path / "model"
    source.mkdir()
    (source / "weights.pt").write_bytes(b"synthetic weights")
    (source / "native-runtime").write_bytes(b"synthetic binary")
    (source / "native-runtime").chmod(0o750)
    inventory = api.plan(spec_for(tmp_path, source), tmp_path / "plan")
    return source, inventory, tmp_path / "plan/inventory.json"


def payload_copy(tmp_path, inventory):
    root = tmp_path / "payload"
    root.mkdir()
    for row in inventory["files"]:
        target = root / row["destination"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row["source"], target)
    return root


def test_plan_is_metadata_only_then_seal_verify_preserve_original(tmp_path, monkeypatch):
    original_hash = api._hash_file
    monkeypatch.setattr(api, "_hash_file", lambda *args, **kwargs: pytest.fail("plan must not hash asset contents"))
    source, inventory, manifest = fixture_plan(tmp_path)
    assert inventory["total_bytes"] == len(b"synthetic weights") + len(b"synthetic binary")
    assert inventory["file_count"] == 2
    assert (tmp_path / "plan/rsync-files.txt").read_text().splitlines() == [row["source"].lstrip("/") for row in inventory["files"]]
    monkeypatch.setattr(api, "_hash_file", original_hash)
    sealed = tmp_path / "sealed.json"
    result = api.seal(manifest, sealed)
    assert json.loads(manifest.read_text())["status"] == "planned"
    assert result["status"] == "sealed"
    root = payload_copy(tmp_path, inventory)
    assert api.verify(sealed, root) == {"status": "verified", "file_count": 2, "total_bytes": inventory["total_bytes"]}
    assert next(row for row in result["files"] if row["source"].endswith("native-runtime"))["mode"] == 0o750


@pytest.mark.parametrize("part", [".ssh", ".aws", ".codex", "auth.json", "credentials.json", ".env", ".env.local", ".secret"])
def test_secret_paths_are_rejected_without_reading(tmp_path, part, monkeypatch):
    source = tmp_path / "assets"
    source.mkdir()
    (source / part).write_bytes(b"not a real secret")
    spec = spec_for(tmp_path, source)
    with pytest.raises(ValueError, match="secret path"):
        api.plan(spec, tmp_path / "plan")
    assert not (tmp_path / "plan").exists()


@pytest.mark.parametrize("ancestor", [False, True])
def test_symlink_entry_or_ancestor_is_rejected(tmp_path, ancestor):
    actual = tmp_path / "actual"
    actual.mkdir()
    (actual / "file.pt").write_bytes(b"x")
    link = tmp_path / "link"
    link.symlink_to(actual if ancestor else actual / "file.pt")
    with pytest.raises(ValueError, match="symlink"):
        api.plan(spec_for(tmp_path, link / "file.pt" if ancestor else link), tmp_path / "plan")


def test_nested_symlink_rejected_even_if_cache_name(tmp_path):
    source = tmp_path / "assets"
    source.mkdir()
    (source / ".git").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="symlink"):
        api.plan(spec_for(tmp_path, source), tmp_path / "plan")


def test_cache_exclusions_are_explicit(tmp_path):
    source = tmp_path / "assets"
    source.mkdir()
    (source / "model.pt").write_bytes(b"x")
    for name in (".git", "__pycache__"):
        (source / name).mkdir()
        (source / name / "skip").write_bytes(b"ignored")
    result = api.plan(spec_for(tmp_path, source), tmp_path / "plan")
    assert result["file_count"] == 1
    assert {Path(row["source"]).name for row in result["excluded"]} == {".git", "__pycache__"}


@pytest.mark.parametrize("value", ["relative/file", "/tmp/../model", "/tmp/new\nline", "/tmp/carriage\rreturn"])
def test_noncanonical_sources_rejected(tmp_path, value):
    with pytest.raises(ValueError):
        api.plan(spec_for(tmp_path, value), tmp_path / "plan")


@pytest.mark.parametrize("destination", ["../escape", "/absolute", "arbitrary/model"])
def test_destination_must_match_rsync_layout(tmp_path, destination):
    source = tmp_path / "model.pt"
    source.write_bytes(b"x")
    with pytest.raises(ValueError, match="destination"):
        api.plan(spec_for(tmp_path, source, destination=destination), tmp_path / "plan")


def test_overlapping_entries_rejected(tmp_path):
    source = tmp_path / "assets"
    source.mkdir()
    child = source / "weights.pt"
    child.write_bytes(b"x")
    spec = spec_for(tmp_path, source)
    data = json.loads(spec.read_text())
    data["entries"].append({"source": str(child), "destination": "fs/" + str(child).lstrip("/"), "role": "overlap"})
    write_json(spec, data)
    with pytest.raises(ValueError, match="duplicate"):
        api.plan(spec, tmp_path / "plan")


@pytest.mark.parametrize("source", ["/", "/tmp", "/project", "/scratch", "/home"])
def test_broad_roots_rejected(tmp_path, source):
    if not Path(source).exists():
        pytest.skip("root does not exist on this host")
    with pytest.raises(ValueError, match="too broad"):
        api.plan(spec_for(tmp_path, source), tmp_path / "plan")


def test_pinned_sha_and_no_overwrite(tmp_path):
    source = tmp_path / "asset"
    source.write_bytes(b"x")
    spec = spec_for(tmp_path, source, sha256=sha256(b"x").hexdigest())
    api.plan(spec, tmp_path / "plan")
    with pytest.raises(FileExistsError):
        api.plan(spec, tmp_path / "plan")
    sealed = tmp_path / "sealed.json"
    api.seal(tmp_path / "plan/inventory.json", sealed)
    before = sealed.read_bytes()
    with pytest.raises(FileExistsError):
        api.seal(tmp_path / "plan/inventory.json", sealed)
    assert sealed.read_bytes() == before


def test_bad_pinned_sha_and_directory_sha_rejected(tmp_path):
    source = tmp_path / "asset"
    source.write_bytes(b"x")
    api.plan(spec_for(tmp_path, source, sha256="f" * 64), tmp_path / "plan")
    with pytest.raises(ValueError, match="pinned SHA"):
        api.seal(tmp_path / "plan/inventory.json", tmp_path / "sealed.json")
    assert not (tmp_path / "sealed.json").exists()
    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(ValueError, match="only for file"):
        api.plan(spec_for(tmp_path, directory, sha256="f" * 64), tmp_path / "plan-directory")


def test_changed_since_plan_rejected(tmp_path):
    source, inventory, manifest = fixture_plan(tmp_path)
    (source / "weights.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed since planning"):
        api.seal(manifest, tmp_path / "sealed.json")


def test_mutation_during_hash_rejected(tmp_path, monkeypatch):
    source, inventory, manifest = fixture_plan(tmp_path)
    original = api.sha256
    changed = False

    class MutatingHash:
        def __init__(self):
            self.digest = original()

        def update(self, value):
            nonlocal changed
            self.digest.update(value)
            if not changed:
                (source / "native-runtime").write_bytes(b"changed during hashing")
                changed = True

        def hexdigest(self):
            return self.digest.hexdigest()

    monkeypatch.setattr(api, "sha256", MutatingHash)
    with pytest.raises(ValueError, match="changed while hashing"):
        api.seal(manifest, tmp_path / "sealed.json")


@pytest.mark.parametrize("failure", ["missing", "tampered", "symlink", "ancestor", "executable"])
def test_verify_rejects_bad_payload(tmp_path, failure):
    source, inventory, manifest = fixture_plan(tmp_path)
    sealed = tmp_path / "sealed.json"
    api.seal(manifest, sealed)
    root = payload_copy(tmp_path, inventory)
    target = root / inventory["files"][0]["destination"]
    if failure == "missing":
        target.unlink()
    elif failure == "tampered":
        target.write_bytes(b"z" * target.stat().st_size)
    elif failure == "symlink":
        target.unlink()
        target.symlink_to(source / "native-runtime")
    elif failure == "ancestor":
        displaced = target.parent.with_name("displaced")
        target.parent.rename(displaced)
        target.parent.symlink_to(displaced)
    else:
        target.chmod(stat.S_IMODE(target.stat().st_mode) & ~0o111)
    with pytest.raises((ValueError, FileNotFoundError)):
        api.verify(sealed, root)


def test_manifest_traversal_duplicate_and_totals_rejected(tmp_path):
    source, inventory, manifest = fixture_plan(tmp_path)
    sealed = tmp_path / "sealed.json"
    data = api.seal(manifest, sealed)
    root = payload_copy(tmp_path, inventory)
    for kind in ("traversal", "duplicate", "total"):
        bad = deepcopy(data)
        if kind == "traversal":
            bad["files"][0]["destination"] = "../escape"
        elif kind == "duplicate":
            bad["files"].append(deepcopy(bad["files"][0]))
        else:
            bad["total_bytes"] += 1
        with pytest.raises(ValueError):
            api.verify(write_json(tmp_path / "bad.json", bad), root)


def test_large_seal_refused_only_on_login_host(tmp_path, monkeypatch):
    source, inventory, manifest = fixture_plan(tmp_path)
    monkeypatch.setattr(api, "_inventory", lambda *args: {**inventory, "total_bytes": 257 * 1024 * 1024})
    monkeypatch.setattr(api.socket, "gethostname", lambda: "login01")
    with pytest.raises(ValueError, match="allocated CPU compute"):
        api.seal(manifest, tmp_path / "sealed.json")
    monkeypatch.setattr(api.socket, "gethostname", lambda: "local-laptop")
    assert api.seal(manifest, tmp_path / "sealed.json")["status"] == "sealed"


@pytest.mark.parametrize("change", [{"snapshot": None}, {"mode": -1}, {"size": True}, {"pinned_sha256": "bad"}])
def test_malformed_snapshot_and_permissions_rejected(tmp_path, change):
    source, inventory, manifest = fixture_plan(tmp_path)
    inventory["files"][0].update(change)
    write_json(manifest, inventory)
    with pytest.raises(ValueError):
        api.seal(manifest, tmp_path / "sealed.json")


def test_cli_plan_seal_verify(tmp_path):
    source = tmp_path / "one.pt"
    source.write_bytes(b"small CLI smoke")
    spec = spec_for(tmp_path, source)
    script = str(Path(api.__file__).resolve())
    output = tmp_path / "plan"
    sealed = tmp_path / "sealed.json"
    commands = [
        ["plan", "--spec", str(spec), "--output", str(output)],
        ["seal", "--manifest", str(output / "inventory.json"), "--output", str(sealed)],
    ]
    for arguments in commands:
        completed = subprocess.run([sys.executable, script, *arguments], capture_output=True, text=True, check=True)
        assert json.loads(completed.stdout)["file_count"] == 1
    root = payload_copy(tmp_path, json.loads(sealed.read_text()))
    completed = subprocess.run([sys.executable, script, "verify", "--manifest", str(sealed), "--root", str(root)],
                               capture_output=True, text=True, check=True)
    assert json.loads(completed.stdout)["status"] == "verified"


def test_github_credentials_path_rejected_before_read(tmp_path):
    protected = tmp_path / ".config/gh/hosts.yml"
    protected.parent.mkdir(parents=True)
    protected.write_text("not a credential")
    with pytest.raises(ValueError, match="secret path"):
        api.plan(spec_for(tmp_path, protected), tmp_path / "plan")


def test_verify_requires_sealed_inventory(tmp_path):
    source, inventory, manifest = fixture_plan(tmp_path)
    with pytest.raises(ValueError, match="expected sealed"):
        api.verify(manifest, tmp_path)
