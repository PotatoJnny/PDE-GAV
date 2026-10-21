"""Small offline Git fixtures; no actual models, cluster jobs or networking."""
from hashlib import sha256
import json
import subprocess

import pytest

from tools import plasma_git_snapshot as snapshot


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], env=snapshot._environment()).decode().strip()


@pytest.fixture
def history(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    git(root, "init", "--quiet", "--template=")
    git(root, "config", "user.name", "Snapshot fixture")
    git(root, "config", "user.email", "snapshot@example.invalid")
    (root / "obsolete-history.txt").write_text("must not travel in the selected pack")
    git(root, "add", ".")
    git(root, "commit", "--quiet", "-m", "parent")
    parent = git(root, "rev-parse", "HEAD")
    old_blob = git(root, "rev-parse", "HEAD:obsolete-history.txt")
    (root / "obsolete-history.txt").unlink()
    (root / "nested").mkdir()
    (root / "nested/data.bin").write_bytes(b"\x00exact\r\nbytes\xff")
    (root / "run.sh").write_text("#!/bin/sh\nexit 0\n")
    (root / "run.sh").chmod(0o755)
    git(root, "add", "-A")
    git(root, "commit", "--quiet", "-m", "selected commit")
    return root, git(root, "rev-parse", "HEAD"), parent, old_blob


def test_exact_commit_offline_restore_excludes_parent_history(history, tmp_path):
    root, revision, parent, old_blob = history
    target = tmp_path / "snapshot"
    receipt = snapshot.export_snapshot(root, revision, target)
    assert receipt["has_parents"] is True
    assert receipt["parents"] == [parent]
    assert receipt["file_count"] == 2
    original_commit = snapshot._git(root, "cat-file", "commit", revision)
    root.rename(tmp_path / "source-not-at-original-location")
    restored = tmp_path / "restored"
    result = snapshot.restore_snapshot(target, restored)
    assert result["clean"] is True
    assert git(restored, "rev-parse", "HEAD") == revision
    assert git(restored, "status", "--porcelain", "--untracked-files=all") == ""
    assert snapshot._git(restored, "cat-file", "commit", revision) == original_commit
    assert git(restored, "rev-list", "--count", "HEAD") == "1"
    assert git(restored, "rev-parse", "--is-shallow-repository") == "true"
    assert (restored / "nested/data.bin").read_bytes() == b"\x00exact\r\nbytes\xff"
    assert (restored / "run.sh").stat().st_mode & 0o111
    for absent in (parent, old_blob):
        with pytest.raises(ValueError, match="git command failed"):
            snapshot._git(restored, "cat-file", "-e", absent)


def test_root_commit_does_not_need_shallow_boundary(tmp_path):
    root = tmp_path / "root-only"
    root.mkdir()
    git(root, "init", "--quiet", "--template=")
    git(root, "-c", "user.name=fixture", "-c", "user.email=f@invalid", "commit", "--quiet", "--allow-empty", "-m", "root")
    receipt = snapshot.export_snapshot(root, git(root, "rev-parse", "HEAD"), tmp_path / "pack")
    assert receipt["object_count"] == 2
    snapshot.restore_snapshot(tmp_path / "pack", tmp_path / "restore")
    assert git(tmp_path / "restore", "rev-parse", "--is-shallow-repository") == "false"


@pytest.mark.parametrize("name", [".env", ".env.production", ".ssh/id_rsa", ".aws/config", "auth.json", "nested/private.key",
                                  ".codex/state.json", ".git-credentials", ".config/gh/hosts.yml", ".config/gcloud/state.json"])
def test_secret_like_tracked_path_refused(history, tmp_path, name):
    root, _, _, _ = history
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("fake fixture, not a credential")
    git(root, "add", "-f", name)
    git(root, "commit", "--quiet", "-m", "forbidden fixture")
    with pytest.raises(ValueError, match="secret-like"):
        snapshot.export_snapshot(root, git(root, "rev-parse", "HEAD"), tmp_path / "pack")
    assert not (tmp_path / "pack").exists()


def test_tracked_symlink_refused(history, tmp_path):
    root, _, _, _ = history
    (root / "linked").symlink_to("nested/data.bin")
    git(root, "add", "linked")
    git(root, "commit", "--quiet", "-m", "symlink fixture")
    with pytest.raises(ValueError, match="symlinks, gitlinks"):
        snapshot.export_snapshot(root, git(root, "rev-parse", "HEAD"), tmp_path / "pack")


def test_gitlink_refused(history, tmp_path):
    root, revision, _, _ = history
    git(root, "update-index", "--add", "--cacheinfo", "160000," + revision + ",subrepo")
    (root / "subrepo").mkdir()
    git(root, "commit", "--quiet", "-m", "gitlink fixture")
    with pytest.raises(ValueError, match="symlinks, gitlinks"):
        snapshot.export_snapshot(root, git(root, "rev-parse", "HEAD"), tmp_path / "pack")


def test_dirty_head_and_output_guards(history, tmp_path):
    root, revision, parent, _ = history
    with pytest.raises(ValueError, match="explicit checkout root"):
        snapshot.export_snapshot(root / "nested", revision, tmp_path / "pack")
    with pytest.raises(ValueError, match="HEAD"):
        snapshot.export_snapshot(root, parent, tmp_path / "pack")
    with pytest.raises(ValueError, match="outside"):
        snapshot.export_snapshot(root, revision, root / "pack")
    (root / "untracked").write_text("fixture")
    with pytest.raises(ValueError, match="clean"):
        snapshot.export_snapshot(root, revision, tmp_path / "pack")


def test_pack_tampering_refused_before_restore(history, tmp_path):
    root, revision, _, _ = history
    snapshot.export_snapshot(root, revision, tmp_path / "pack")
    with (tmp_path / "pack/snapshot.pack").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="digest"):
        snapshot.restore_snapshot(tmp_path / "pack", tmp_path / "restore")
    assert not (tmp_path / "restore").exists()


def test_snapshot_symlink_refused(history, tmp_path):
    root, revision, _, _ = history
    snapshot.export_snapshot(root, revision, tmp_path / "pack")
    (tmp_path / "linked-pack").symlink_to(tmp_path / "pack", target_is_directory=True)
    with pytest.raises(ValueError, match="nonsymlink"):
        snapshot.restore_snapshot(tmp_path / "linked-pack", tmp_path / "restore")


def test_receipt_tree_tampering_refused(history, tmp_path):
    root, revision, _, _ = history
    snapshot.export_snapshot(root, revision, tmp_path / "pack")
    path = tmp_path / "pack/receipt.json"
    receipt = json.loads(path.read_text())
    receipt["root_tree"] = "0" * 40
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="inventory differs"):
        snapshot.restore_snapshot(tmp_path / "pack", tmp_path / "restore")


def test_large_login_export_requires_allocation(history, tmp_path, monkeypatch):
    root, revision, _, _ = history
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(snapshot, "MAX_LOGIN_OBJECT_BYTES", 1)
    with pytest.raises(RuntimeError, match="compute allocation"):
        snapshot.export_snapshot(root, revision, tmp_path / "pack")
    assert not (tmp_path / "pack").exists()


def test_extra_parent_object_refused_even_if_pack_digest_updated(history, tmp_path):
    root, revision, parent, _ = history
    target = tmp_path / "pack"
    receipt = snapshot.export_snapshot(root, revision, target)
    _, objects = snapshot._inventory(root, revision)
    payload = snapshot._git(root, "pack-objects", "--stdout", "--window=0", "--no-reuse-delta",
                            input=("\n".join(objects + [parent]) + "\n").encode())
    (target / "snapshot.pack").write_bytes(payload)
    receipt.update(pack_bytes=len(payload), pack_sha256=sha256(payload).hexdigest())
    (target / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="extra objects"):
        snapshot.restore_snapshot(target, tmp_path / "restore")


def test_git_environment_does_not_redirect_source(history, tmp_path, monkeypatch):
    root, revision, _, _ = history
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "unrelated"))
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.worktree")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", str(tmp_path / "unrelated"))
    assert snapshot.export_snapshot(root, revision, tmp_path / "pack")["revision"] == revision
