"""Small CPU/mock builder checks. These never compile Codex or qualify CUDA."""
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


SPEC = importlib.util.spec_from_file_location(
    "compatible_gpu_codex_build",
    Path(__file__).resolve().parents[1] / "scripts/build_compatible_gpu_codex.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize(("environment", "host", "jobs"), [
    ({}, "kn123", 1),
    ({"SLURM_JOB_ID": "fake", "SLURM_CPUS_PER_TASK": "4"}, "kn123", 1),
    ({"SLURM_JOB_ID": "123", "SLURM_CPUS_PER_TASK": "4"}, "login1", 1),
    ({"SLURM_JOB_ID": "123", "SLURM_CPUS_PER_TASK": "4"}, "kn123", 5),
    ({"SLURM_JOB_ID": "123", "SLURM_CPUS_PER_TASK": "4"}, "kn123", 0),
])
def test_compute_guard_rejects(environment, host, jobs):
    with pytest.raises(RuntimeError):
        MODULE.require_compute(environment, host, jobs)


def test_compute_guard_accepts_scoped_allocation():
    MODULE.require_compute({"SLURM_JOB_ID": "123", "SLURM_CPUS_PER_TASK": "4"}, "kn123", 4)


def test_environment_isolated_and_production_guards_fixed(tmp_path):
    environment = MODULE.build_environment({
        "PATH": "/usr/bin", "OPENAI_API_KEY": "must-not-copy", "CODEX_HOME": "/private",
        "RUSTFLAGS": "-C debug-assertions=yes", "CARGO_BUILD_TARGET": "arbitrary",
        "CARGO_PROFILE_DEV_SMALL_DEBUG_ASSERTIONS": "true"},
        tmp_path / "cargo", tmp_path / "rustup", tmp_path / "target")
    assert "OPENAI_API_KEY" not in environment
    assert "CODEX_HOME" not in environment
    assert "RUSTFLAGS" not in environment
    assert "CARGO_BUILD_TARGET" not in environment
    assert environment["CARGO_PROFILE_DEV_SMALL_DEBUG_ASSERTIONS"] == "false"
    assert environment["CODEX_BWRAP_SHA256"] == MODULE.ORIGINAL_HASHES["codex-resources/bwrap"]


def lock_bytes(version, *, external_version="1.0.0", count=152):
    text = "version = 4\n"
    for index in range(count):
        text += f'[[package]]\nname = "workspace-{index}"\nversion = "{version}"\n'
    text += ('[[package]]\nname = "external"\n'
             f'version = "{external_version}"\nsource = "registry+test"\nchecksum = "abc"\n')
    return text.encode()


def test_workspace_only_lock_stamp_accepted():
    assert MODULE.verify_lock_version_stamp(lock_bytes("0.0.0"), lock_bytes(MODULE.VERSION)) == 152


@pytest.mark.parametrize("derived", [
    lock_bytes(MODULE.VERSION, external_version="2.0.0"),
    lock_bytes(MODULE.VERSION, count=151),
    lock_bytes("0.155.2"),
])
def test_dependency_or_stamp_drift_rejected(derived):
    with pytest.raises(RuntimeError, match="152 workspace"):
        MODULE.verify_lock_version_stamp(lock_bytes("0.0.0"), derived)


def test_inventory_rejects_symlink(tmp_path):
    (tmp_path / "linked").symlink_to("/dev/null")
    with pytest.raises(RuntimeError, match="symlink"):
        MODULE.inventory(tmp_path)


def test_inventory_rejects_special_file(tmp_path):
    import os
    os.mkfifo(tmp_path / "fifo")
    with pytest.raises(RuntimeError, match="special file"):
        MODULE.inventory(tmp_path)


@pytest.fixture
def mocked_build(tmp_path, monkeypatch):
    source, bundle = tmp_path / "source", tmp_path / "original"
    source.mkdir()
    (source / "codex-rs").mkdir()
    bundle.mkdir()
    metadata = {"layoutVersion": 1, "version": MODULE.VERSION,
                "target": "x86_64-unknown-linux-musl", "variant": "codex",
                "entrypoint": "bin/codex", "resourcesDir": "codex-resources",
                "pathDir": "codex-path"}
    (bundle / "codex-package.json").write_text(json.dumps(metadata))
    for name in MODULE.ORIGINAL_HASHES:
        path = bundle / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("original-" + name).encode())
        path.chmod(0o755)
    monkeypatch.setattr(MODULE, "ORIGINAL_HASHES", {
        name: MODULE.digest(bundle / name) for name in MODULE.ORIGINAL_HASHES})
    cargo_home, rustup_home = tmp_path / "cargo", tmp_path / "rustup"
    cargo_home.mkdir()
    rustup_home.mkdir()
    compiler_directory = rustup_home / "toolchains" / (MODULE.TOOLCHAIN + "-x86_64-unknown-linux-gnu") / "bin"
    compiler_directory.mkdir(parents=True)
    (compiler_directory / "rustc").write_bytes(b"compiler fixture")
    (compiler_directory / "cargo").write_bytes(b"cargo fixture")
    patch, lock_patch = tmp_path / "patch", tmp_path / "lock-patch"
    patch.write_text("reviewed sandbox patch")
    lock_patch.write_text("reviewed lock patch")
    args = SimpleNamespace(source=source, original_bundle=bundle,
        patch=patch, patch_sha256=MODULE.digest(patch),
        lock_patch=lock_patch, lock_patch_sha256=MODULE.digest(lock_patch),
        output=tmp_path / "output", cargo_home=cargo_home, rustup_home=rustup_home,
        target_dir=tmp_path / "target", jobs=4)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    monkeypatch.setattr(MODULE.platform, "node", lambda: "kn123")
    monkeypatch.setattr(MODULE, "verify_source", lambda *args: {"verified": True})
    monkeypatch.setattr(MODULE, "run_text", lambda command, **kwargs:
        b"rustc 1.95.0\nrelease: 1.95.0\nhost: x86_64-unknown-linux-gnu\n"
        if command[-1] == "-vV" else b"cargo 1.95.0\n")
    calls = []
    def fake_cargo(command, **kwargs):
        calls.append((command, kwargs))
        binary = args.target_dir / MODULE.PROFILE / "codex"
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b"new compiled binary fixture")
        binary.chmod(0o755)
        kwargs["stdout"].write(b"mock build only\n")
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(MODULE.subprocess, "run", fake_cargo)
    return args, calls


def test_mock_build_preserves_original_and_records_unqualified(mocked_build):
    args, calls = mocked_build
    original = MODULE.inventory(args.original_bundle)
    result = MODULE.build(args)
    assert result["status"] == "built_unqualified"
    assert result["cuda_qualified"] is False
    assert result["release_equivalent"] is False
    assert result["debug_assertions"] is False
    assert result["target"] == "x86_64-unknown-linux-gnu"
    assert result["reused_resource_target"] == "x86_64-unknown-linux-musl"
    assert MODULE.inventory(args.original_bundle) == original
    assert (args.output / "source.patch").read_bytes() == args.patch.read_bytes()
    assert (args.output / "workspace-version-lock.patch").read_bytes() == args.lock_patch.read_bytes()
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert "--locked" in command
    assert command[command.index("--bin") + 1] == "codex"
    assert command[command.index("--jobs") + 1] == "4"
    assert kwargs["env"]["CARGO_PROFILE_DEV_SMALL_DEBUG_ASSERTIONS"] == "false"
    assert (args.output / "runtime/bin/codex").read_bytes() == b"new compiled binary fixture"


def test_existing_output_rejected_before_build(mocked_build):
    args, calls = mocked_build
    args.output.mkdir()
    with pytest.raises(RuntimeError, match="must not exist"):
        MODULE.build(args)
    assert calls == []


def test_nonexecutable_package_component_rejected(mocked_build):
    args, calls = mocked_build
    (args.original_bundle / "codex-resources/bwrap").chmod(0o600)
    with pytest.raises(RuntimeError, match="component mismatch"):
        MODULE.build(args)
    assert calls == []
    assert not args.output.exists()


def test_failed_build_retains_log_manifest_and_no_runtime(mocked_build, monkeypatch):
    args, _ = mocked_build
    def failing(command, **kwargs):
        kwargs["stdout"].write(b"compile failure fixture\n")
        return subprocess.CompletedProcess(command, 23)
    monkeypatch.setattr(MODULE.subprocess, "run", failing)
    with pytest.raises(RuntimeError, match="Cargo build failed"):
        MODULE.build(args)
    report = json.loads((args.output / "build-provenance.json").read_text())
    assert report["status"] == "failed"
    assert report["build_returncode"] == 23
    assert not report["cuda_qualified"]
    assert not (args.output / "runtime").exists()
    assert "compile failure" in (args.output / "build.log").read_text()


def test_build_checks_source_once_before_compilation(mocked_build, monkeypatch):
    args, _ = mocked_build
    checks = iter(({"revision": "before"}, {"revision": "after"}))
    monkeypatch.setattr(MODULE, "verify_source", lambda *args: next(checks))
    MODULE.build(args)
    assert (args.output / "runtime/bin/codex").is_file()
    assert next(checks) == {"revision": "after"}


def test_cargo_configuration_override_rejected(mocked_build):
    args, calls = mocked_build
    (args.cargo_home / "config.toml").write_text('[build]\ntarget = "other"\n')
    with pytest.raises(RuntimeError, match="configuration overrides"):
        MODULE.build(args)
    assert calls == []


def test_missing_pinned_toolchain_not_installed(mocked_build):
    args, calls = mocked_build
    args.rustup_home = args.rustup_home.parent / "empty-rustup"
    args.rustup_home.mkdir()
    with pytest.raises(RuntimeError, match="already be installed"):
        MODULE.build(args)
    assert calls == []


def test_cache_cannot_write_into_source_or_install(mocked_build):
    args, calls = mocked_build
    args.target_dir = args.original_bundle / "target"
    with pytest.raises(RuntimeError, match="must not modify"):
        MODULE.build(args)
    assert calls == []


def test_official_root_alias_preserved_and_points_to_new_cli(mocked_build):
    args, _ = mocked_build
    (args.original_bundle / "codex").symlink_to("bin/codex")
    original = MODULE.inventory(args.original_bundle)
    report = MODULE.build(args)
    alias = args.output / "runtime/codex"
    assert alias.is_symlink()
    assert alias.readlink() == Path("bin/codex")
    assert alias.read_bytes() == b"new compiled binary fixture"
    assert report["runtime_files"]["codex"]["resolved_member"] == "bin/codex"
    assert MODULE.inventory(args.original_bundle) == original


@pytest.mark.parametrize("target", ["/dev/null", "../outside", "codex-resources/bwrap"])
def test_unapproved_root_alias_rejected_before_build(mocked_build, target):
    args, calls = mocked_build
    (args.original_bundle / "codex").symlink_to(target)
    with pytest.raises(RuntimeError, match="symlink"):
        MODULE.build(args)
    assert calls == []
