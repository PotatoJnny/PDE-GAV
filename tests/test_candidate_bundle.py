import json
import stat
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from pde_agent_verifier_solver.sdk.candidate_bundle import (
    CANDIDATE_BUNDLE_MANIFEST,
    CANDIDATE_BUNDLE_SCHEMA,
    CandidateBundleError,
    CandidateBundleLimits,
    freeze_candidate_bundle,
    inspect_candidate_bundle,
    write_candidate_bundle_manifest,
)


def _write_tree(root: Path, *, model: bytes = b"trained weights") -> None:
    (root / "lib").mkdir(parents=True)
    (root / "assets").mkdir()
    (root / "submission.py").write_text(
        "from lib.rom import predict_macros\n", encoding="utf-8"
    )
    (root / "lib" / "rom.py").write_text(
        "def predict_macros(*args): return 1\n", encoding="utf-8"
    )
    (root / "assets" / "model.bin").write_bytes(model)


def _write_raw_manifest(root: Path, **changes: object) -> Path:
    payload = {
        "schema": CANDIDATE_BUNDLE_SCHEMA,
        "entrypoint": "submission.py",
        "files": ["submission.py"],
    }
    payload.update(changes)
    path = root / CANDIDATE_BUNDLE_MANIFEST
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_multifile_identity_is_root_and_manifest_order_independent(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    _write_tree(first_root)
    _write_tree(second_root)
    files = ["submission.py", "lib/rom.py", "assets/model.bin"]
    write_candidate_bundle_manifest(
        first_root,
        files=files,
    )
    write_candidate_bundle_manifest(
        second_root,
        files=reversed(files),
    )

    first = inspect_candidate_bundle(first_root)
    second = inspect_candidate_bundle(second_root / CANDIDATE_BUNDLE_MANIFEST)

    assert first.tree_sha256 == second.tree_sha256
    assert first.members == second.members
    assert first.total_bytes == second.total_bytes
    assert first.root != second.root
    assert first.entrypoint_path == first_root / "submission.py"
    assert tuple(member.path for member in first.members) == (
        "assets/model.bin",
        "lib/rom.py",
        "submission.py",
    )


def test_tree_identity_covers_contents_paths_and_entrypoint(tmp_path: Path) -> None:
    roots = [tmp_path / name for name in ("base", "contents", "entrypoint")]
    for root in roots:
        root.mkdir()
        _write_tree(root)
    write_candidate_bundle_manifest(
        roots[0],
        files=["submission.py", "lib/rom.py", "assets/model.bin"],
    )
    (roots[1] / "assets" / "model.bin").write_bytes(b"different weights")
    write_candidate_bundle_manifest(
        roots[1],
        files=["submission.py", "lib/rom.py", "assets/model.bin"],
    )
    (roots[2] / "alternative.py").write_text("pass\n", encoding="utf-8")
    write_candidate_bundle_manifest(
        roots[2],
        entrypoint="alternative.py",
        files=[
            "submission.py",
            "alternative.py",
            "lib/rom.py",
            "assets/model.bin",
        ],
    )

    identities = {inspect_candidate_bundle(root).tree_sha256 for root in roots}

    assert len(identities) == 3


def test_single_python_file_remains_a_supported_bundle(tmp_path: Path) -> None:
    source = tmp_path / "candidate.py"
    source.write_text("def predict_macros(*args): return 0\n", encoding="utf-8")

    bundle = inspect_candidate_bundle(source)

    assert bundle.entrypoint == "candidate.py"
    assert tuple(member.path for member in bundle.members) == ("candidate.py",)
    assert bundle.total_bytes == source.stat().st_size


@pytest.mark.parametrize(
    "unsafe",
    (
        "../outside.py",
        "/absolute.py",
        "assets//weights.pt",
        "assets/./weights.pt",
        "assets\\weights.pt",
    ),
)
def test_manifest_rejects_unsafe_member_paths(tmp_path: Path, unsafe: str) -> None:
    root = tmp_path / "candidate"
    root.mkdir()
    (root / "submission.py").write_text("pass\n", encoding="utf-8")
    manifest = _write_raw_manifest(root, files=["submission.py", unsafe])

    with pytest.raises(CandidateBundleError, match="unsafe candidate bundle path"):
        inspect_candidate_bundle(manifest)


def test_closed_tree_rejects_undeclared_files_and_symlinks(tmp_path: Path) -> None:
    undeclared_root = tmp_path / "undeclared"
    undeclared_root.mkdir()
    (undeclared_root / "submission.py").write_text("pass\n", encoding="utf-8")
    (undeclared_root / "forgotten.pt").write_bytes(b"asset")
    _write_raw_manifest(undeclared_root)

    with pytest.raises(CandidateBundleError, match="undeclared=.*forgotten.pt"):
        inspect_candidate_bundle(undeclared_root)

    symlink_root = tmp_path / "symlink"
    symlink_root.mkdir()
    (symlink_root / "submission.py").write_text("pass\n", encoding="utf-8")
    (symlink_root / "real.pt").write_bytes(b"weights")
    (symlink_root / "linked.pt").symlink_to("real.pt")
    _write_raw_manifest(
        symlink_root,
        files=["submission.py", "real.pt", "linked.pt"],
    )

    with pytest.raises(CandidateBundleError, match="cannot contain symlink"):
        inspect_candidate_bundle(symlink_root)


def test_limits_apply_to_file_count_individual_files_and_total(tmp_path: Path) -> None:
    root = tmp_path / "candidate"
    root.mkdir()
    (root / "submission.py").write_bytes(b"1234")
    (root / "weights.bin").write_bytes(b"5678")
    _write_raw_manifest(root, files=["submission.py", "weights.bin"])

    with pytest.raises(CandidateBundleError, match="max_files"):
        inspect_candidate_bundle(root, limits=CandidateBundleLimits(max_files=1))
    with pytest.raises(CandidateBundleError, match="max_file_bytes"):
        inspect_candidate_bundle(
            root,
            limits=CandidateBundleLimits(max_file_bytes=3),
        )
    with pytest.raises(CandidateBundleError, match="max_total_bytes"):
        inspect_candidate_bundle(
            root,
            limits=CandidateBundleLimits(max_total_bytes=7),
        )


def test_agent_declared_digest_and_size_are_optional_metadata(tmp_path: Path) -> None:
    root = tmp_path / "candidate"
    root.mkdir()
    (root / "submission.py").write_text("pass\n", encoding="utf-8")
    _write_raw_manifest(root, tree_sha256="0" * 64)
    assert inspect_candidate_bundle(root).tree_sha256 != "0" * 64

    _write_raw_manifest(root, total_bytes=999)
    assert inspect_candidate_bundle(root).total_bytes == 5


def test_freeze_copies_complete_tree_under_digest_and_makes_it_read_only(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    _write_tree(source_root)
    write_candidate_bundle_manifest(
        source_root,
        files=["submission.py", "lib/rom.py", "assets/model.bin"],
    )
    source = inspect_candidate_bundle(source_root)

    frozen = freeze_candidate_bundle(source, tmp_path / "sealed")

    assert frozen.root == (tmp_path / "sealed" / source.tree_sha256).resolve()
    assert frozen.tree_sha256 == source.tree_sha256
    assert frozen.normalized_manifest() == json.loads(
        frozen.manifest_path.read_text(encoding="utf-8")
    )
    for member in frozen.members:
        mode = frozen.root.joinpath(*member.path.split("/")).stat().st_mode
        assert not mode & stat.S_IWUSR
    assert not frozen.root.stat().st_mode & stat.S_IWUSR

    (source_root / "assets" / "model.bin").write_bytes(b"source changed later")
    assert inspect_candidate_bundle(frozen.root).tree_sha256 == source.tree_sha256
    assert freeze_candidate_bundle(frozen.root, tmp_path / "sealed") == frozen


def test_candidate_bundle_value_is_frozen(tmp_path: Path) -> None:
    source = tmp_path / "candidate.py"
    source.write_text("pass\n", encoding="utf-8")
    bundle = inspect_candidate_bundle(source)

    with pytest.raises(FrozenInstanceError):
        bundle.tree_sha256 = "0" * 64



def test_freeze_reuses_inspection_without_rescanning_model(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "candidate.py"
    source.write_text("pass\n", encoding="utf-8")
    inspected = inspect_candidate_bundle(source)
    from pde_agent_verifier_solver.sdk import candidate_bundle as module
    monkeypatch.setattr(module, "_hash_regular_member",
                        lambda *args: pytest.fail("repeated full-file hashing"))
    frozen = freeze_candidate_bundle(inspected, tmp_path / "sealed")
    loaded = inspect_candidate_bundle(frozen.root, recorded_identity=frozen.tree_sha256)
    assert loaded.entrypoint_path.read_bytes() == source.read_bytes()
    assert loaded.tree_sha256 == inspected.tree_sha256
