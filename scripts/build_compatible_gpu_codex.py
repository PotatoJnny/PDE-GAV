"""Build an isolated, pinned Codex GPU-compatibility package on Slurm only.

This never installs software, submits jobs, reads authentication, or invokes a
model. The resulting package is NOT CUDA-qualified or release-equivalent.
An existing build cache is reusable, but adopting an arbitrary binary is not.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import subprocess
import time
import tomllib


UPSTREAM_COMMIT = "be2951ea34f0d295ed0becf97079f92fa5f6950e"
LOCK_SHA256 = "81eb29f6276d5b3d1b39f1588a26059dadfbba717952247efeb8a575f271f569"
DERIVED_LOCK_SHA256 = "df88a71b82843c6f092610fb07589f7a40032ddc25f50718354546ca541eb9b7"
TOOLCHAIN = "1.95.0"
VERSION = "0.155.1"
PROFILE = "dev-small"
PATCHED_FILE = "codex-rs/linux-sandbox/src/bwrap.rs"
NEW_TEST_FILE = "codex-rs/linux-sandbox/src/bwrap_nvidia_tests.rs"
LOCK_FILE = "codex-rs/Cargo.lock"
ORIGINAL_HASHES = {
    "bin/codex": "0753dfe1d8b87a52436deb13eb1c549661ef4c84fee2c5aa688385eebeccb761",
    "bin/codex-code-mode-host": "210ab8ebaebf4bc1421d9e30339c858354ca35fa91e2f87f4c6204e5382f8a63",
    "codex-path/rg": "e62198eb19b136b88c330af83647b5a962cb99b6b1f066758568f12de1974849",
    "codex-resources/bwrap": "77360cb751ccedc5971391444ac86a8a33c15b04d6b4a6fe45f5d25496e62c4c",
    "codex-resources/zsh/bin/zsh": "67faaaa89242c4a332e16e508a1977cffc24bf7fca31d4411cdfd101f3831ef3",
}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def run_text(command, *, cwd=None, env=None, allowed=(0,)):
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, check=False)
    if result.returncode not in allowed:
        raise RuntimeError(f"Command failed ({result.returncode}): {command[0]}")
    return result.stdout


def git(source, *arguments, allowed=(0,)):
    return run_text(["git", "--no-optional-locks", "-C", str(source), *arguments], allowed=allowed)


def require_compute(environment, hostname, jobs):
    if not re.fullmatch(r"[0-9]+", environment.get("SLURM_JOB_ID", "")):
        raise RuntimeError("A real Slurm compute allocation is required")
    if "login" in hostname.lower().split(".")[0]:
        raise RuntimeError("Building on a login node is prohibited")
    cpus = environment.get("SLURM_CPUS_PER_TASK", "")
    if not cpus.isdigit() or not 1 <= jobs <= int(cpus):
        raise RuntimeError("--jobs must fit positive SLURM_CPUS_PER_TASK")


def inventory(directory):
    """Allow only the official root CLI alias; reject all other symlinks."""
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise RuntimeError("Package root must be a real directory")
    result = {}
    for base, directories, files in os.walk(directory, followlinks=False):
        for name in directories + files:
            path = Path(base) / name
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                relative = path.relative_to(directory).as_posix()
                target = os.readlink(path)
                approved = directory / "bin/codex"
                if (relative != "codex" or target != "bin/codex"
                        or approved.is_symlink() or not approved.is_file()
                        or path.resolve(strict=True) != approved.resolve(strict=True)):
                    raise RuntimeError(f"Package symlink is not allowed: {relative}")
                # The target is also inventoried/pinned as a regular package member.
                result[relative] = {"symlink_target": target, "resolved_member": "bin/codex",
                                    "bytes": approved.stat().st_size,
                                    "mode": stat.S_IMODE(metadata.st_mode)}
                continue
            if stat.S_ISDIR(metadata.st_mode):
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise RuntimeError(f"Package special file is not allowed: {path.relative_to(directory)}")
            result[path.relative_to(directory).as_posix()] = {
                "bytes": metadata.st_size,
                "mode": stat.S_IMODE(metadata.st_mode),
            }
    return result


def verify_bundle(bundle):
    files = inventory(bundle)
    metadata = json.loads((bundle / "codex-package.json").read_text())
    expected = {"layoutVersion": 1, "version": VERSION,
                "target": "x86_64-unknown-linux-musl", "variant": "codex",
                "entrypoint": "bin/codex", "resourcesDir": "codex-resources",
                "pathDir": "codex-path"}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Original bundle is not the pinned 0.155.1 Linux package")
    for name, expected_hash in ORIGINAL_HASHES.items():
        item = files.get(name, {})
        if not item or not item.get("mode", 0) & 0o111:
            raise RuntimeError(f"Original bundle component mismatch: {name}")
    return metadata, files


def verify_lock_version_stamp(original, derived):
    expected = tomllib.loads(original.decode())
    changed = 0
    for package in expected["package"]:
        if "source" not in package:
            if package["version"] != "0.0.0":
                raise RuntimeError("Unexpected upstream workspace package version")
            package["version"] = VERSION
            changed += 1
    if changed != 152 or expected != tomllib.loads(derived.decode()):
        raise RuntimeError("Lock patch must change only the 152 workspace version stamps")
    return changed


def verify_source(source, patch, patch_sha256, lock_patch, lock_patch_sha256):
    if patch.is_symlink() or not patch.is_file():
        raise RuntimeError("Patch must be a regular file")
    if git(source, "rev-parse", "HEAD").decode().strip() != UPSTREAM_COMMIT:
        raise RuntimeError("Upstream commit mismatch")
    original_lock = git(source, "show", "HEAD:" + LOCK_FILE)
    stamp_count = verify_lock_version_stamp(original_lock, (source / LOCK_FILE).read_bytes())
    if lock_patch.is_symlink() or not lock_patch.is_file():
        raise RuntimeError("Lock patch must be a regular file")
    if git(source, "diff", "--binary", "--no-ext-diff", "--no-color",
           "HEAD", "--", LOCK_FILE) != lock_patch.read_bytes():
        raise RuntimeError("Lock patch artifact differs from the exact source diff")
    tracked = git(source, "diff", "--name-only", "-z", "HEAD", "--").decode().split("\0")
    untracked = git(source, "ls-files", "--others", "--exclude-standard", "-z").decode().split("\0")
    if {name for name in tracked if name} != {PATCHED_FILE, LOCK_FILE}:
        raise RuntimeError("Only the approved bwrap.rs and workspace lock patches are allowed")
    if {name for name in untracked if name} != {NEW_TEST_FILE}:
        raise RuntimeError("Only the approved new bwrap test file is allowed")
    if any((source / name).is_symlink() for name in (PATCHED_FILE, NEW_TEST_FILE)):
        raise RuntimeError("Patched source must not be a symlink")
    canonical_patch = git(source, "diff", "--binary", "--no-ext-diff", "--no-color",
                          "HEAD", "--", PATCHED_FILE)
    canonical_patch += git(source, "diff", "--no-index", "--binary", "--no-ext-diff",
                           "--no-color", "--", "/dev/null", NEW_TEST_FILE, allowed=(1,))
    if canonical_patch != patch.read_bytes():
        raise RuntimeError("Patch artifact differs from the exact source diff")
    return {"upstream_commit": UPSTREAM_COMMIT, "upstream_cargo_lock_sha256": LOCK_SHA256,
            "derived_cargo_lock_sha256": DERIVED_LOCK_SHA256,
            "lock_patch_sha256": lock_patch_sha256,
            "workspace_version_stamps_changed": stamp_count,
            "external_dependencies_unchanged": True,
            "patch_sha256": patch_sha256,
            "modified_files": {name: digest(source / name)
                               for name in (PATCHED_FILE, NEW_TEST_FILE)},
            "rust_toolchain_file_sha256": digest(source / "codex-rs/rust-toolchain.toml")}


def build_environment(environment, cargo_home, rustup_home, target_dir):
    # Do not pass API/auth/session variables or caller-supplied Rust/Cargo flags.
    allowed = ("PATH", "LANG", "LC_ALL", "TMPDIR", "LD_LIBRARY_PATH", "LIBRARY_PATH",
               "CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "PKG_CONFIG_PATH",
               "CC", "CXX", "AR", "RANLIB", "SSL_CERT_FILE", "SSL_CERT_DIR")
    result = {key: environment[key] for key in allowed if key in environment}
    result.update(CARGO_HOME=str(cargo_home), RUSTUP_HOME=str(rustup_home),
                  RUSTUP_TOOLCHAIN=TOOLCHAIN, CARGO_TARGET_DIR=str(target_dir),
                  CARGO_PROFILE_DEV_SMALL_DEBUG_ASSERTIONS="false",
                  CODEX_BWRAP_SHA256=ORIGINAL_HASHES["codex-resources/bwrap"],
                  GIT_TERMINAL_PROMPT="0")
    result["PATH"] = str(cargo_home / "bin") + os.pathsep + result.get("PATH", os.defpath)
    return result


def validate_toolchain_layout(cargo_home, rustup_home):
    # Cargo home configuration can silently override build target/rustflags.
    # A controlled cache does not need either user configuration file.
    if any((cargo_home / name).exists() for name in ("config", "config.toml")):
        raise RuntimeError("Isolated Cargo home must not contain configuration overrides")
    compiler = rustup_home / "toolchains" / (TOOLCHAIN + "-x86_64-unknown-linux-gnu") / "bin/rustc"
    cargo = compiler.with_name("cargo")
    if not compiler.is_file() or not cargo.is_file():
        raise RuntimeError("Pinned toolchain must already be installed; builder will not install it")
    return {"rustc_binary_sha256": digest(compiler), "cargo_binary_sha256": digest(cargo)}


def assemble(bundle, binary, runtime, original_metadata, original_files, host, patch_hash):
    if runtime.exists() or runtime.is_symlink():
        raise RuntimeError("Runtime destination must not exist")
    # Copy private bytes, never link or modify the original installed components.
    shutil.copytree(bundle, runtime, symlinks=True,
                    ignore=lambda directory, names: ["codex"]
                    if Path(directory) == bundle / "bin" else [])
    shutil.copyfile(binary, runtime / "bin/codex")
    (runtime / "bin/codex").chmod(0o755)
    metadata = dict(original_metadata)
    metadata.update(version=VERSION + "+gpu.leaves." + patch_hash[:12], target=host)
    (runtime / "codex-package.json").write_text(json.dumps(metadata, indent=2) + "\n")
    files = inventory(runtime)
    expected_names = set(original_files)
    if set(files) != expected_names:
        raise RuntimeError("Assembled package file set changed")
    for name, original in original_files.items():
        if name == "codex" and "symlink_target" in original:
            if (files[name].get("symlink_target") != original["symlink_target"]
                    or files[name].get("resolved_member") != "bin/codex"):
                raise RuntimeError("Root CLI alias did not follow the new bundled executable")
        elif name not in ("bin/codex", "codex-package.json") and files[name] != original:
            raise RuntimeError(f"Copied resource changed: {name}")
    return files


def write_manifest(output, report):
    (output / "build-provenance.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def build(args):
    require_compute(os.environ, platform.node(), args.jobs)
    source = args.source.resolve(strict=True)
    bundle = args.original_bundle.resolve(strict=True)
    patch = args.patch.absolute()
    lock_patch = args.lock_patch.absolute()
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        raise RuntimeError("Output must not exist; failed outputs are retained")
    output = output.parent.resolve(strict=True) / output.name
    cargo_home = args.cargo_home.resolve(strict=True)
    rustup_home = args.rustup_home.resolve(strict=True)
    target_dir = args.target_dir.resolve()
    for protected in (source, bundle, cargo_home, rustup_home):
        if output.is_relative_to(protected) or target_dir.is_relative_to(protected):
            raise RuntimeError("Output/cache must not modify source or installed runtime")
    if target_dir.is_relative_to(output) or output.is_relative_to(target_dir):
        raise RuntimeError("Use a separate build cache, not the source/output root")
    source_info = verify_source(source, patch, args.patch_sha256,
                                lock_patch, args.lock_patch_sha256)
    original_metadata, original_files = verify_bundle(bundle)
    toolchain_hashes = validate_toolchain_layout(cargo_home, rustup_home)
    environment = build_environment(os.environ, cargo_home, rustup_home, target_dir)
    cargo = cargo_home / "bin/cargo"
    rustc = cargo_home / "bin/rustc"
    rustc_version = run_text([str(rustc), "+" + TOOLCHAIN, "-vV"], env=environment).decode()
    if f"release: {TOOLCHAIN}\n" not in rustc_version:
        raise RuntimeError("Rust toolchain version mismatch")
    host_match = re.search(r"^host: (\S+)$", rustc_version, re.MULTILINE)
    if not host_match or host_match[1] != "x86_64-unknown-linux-gnu":
        raise RuntimeError("This builder qualifies only the native x86_64 Linux GNU target")
    cargo_version = run_text([str(cargo), "+" + TOOLCHAIN, "--version"], env=environment).decode().strip()
    command = [str(cargo), "+" + TOOLCHAIN, "build", "--locked", "-p", "codex-cli",
               "--bin", "codex", "--profile", PROFILE, "--jobs", str(args.jobs)]
    report = {"schema": "isolated_codex_gpu_build_v1", "status": "building",
              "started_at_unix": time.time(), "hostname": platform.node(),
              "slurm_job_id": os.environ["SLURM_JOB_ID"],
              "source": str(source), "source_verification": source_info,
              "original_bundle": str(bundle), "original_bundle_files": original_files,
              "profile": PROFILE, "debug_assertions": False,
              "release_equivalent": False, "cuda_qualified": False,
              "model_invoked": False, "authentication_read": False,
              "installed_runtime_modified": False, "target": host_match[1],
              "reused_resource_target": original_metadata["target"],
              "rustc_version": rustc_version, "cargo_version": cargo_version,
              "toolchain_binary_hashes": toolchain_hashes,
              "build_command": command, "build_cache": str(target_dir.resolve()),
              "builder_sha256": digest(__file__),
              "explicit_build_environment": {key: environment[key] for key in (
                  "CARGO_HOME", "RUSTUP_HOME", "RUSTUP_TOOLCHAIN", "CARGO_TARGET_DIR",
                  "CARGO_PROFILE_DEV_SMALL_DEBUG_ASSERTIONS", "CODEX_BWRAP_SHA256")}}
    output.mkdir(mode=0o700)
    shutil.copyfile(patch, output / "source.patch")
    shutil.copyfile(lock_patch, output / "workspace-version-lock.patch")
    write_manifest(output, report)
    try:
        with (output / "build.log").open("wb") as log:
            result = subprocess.run(command, cwd=source / "codex-rs", env=environment,
                                    stdout=log, stderr=subprocess.STDOUT, check=False)
        report["build_returncode"] = result.returncode
        if result.returncode:
            raise RuntimeError(f"Cargo build failed ({result.returncode}); see build.log")
        binary = target_dir / PROFILE / "codex"
        if binary.is_symlink() or not binary.is_file() or not os.access(binary, os.X_OK):
            raise RuntimeError("Cargo did not produce the expected executable")
        report["runtime_files"] = assemble(bundle, binary, output / "runtime",
                                            original_metadata, original_files,
                                            host_match[1], args.patch_sha256)
        report["runtime_binary"] = str(output / "runtime/bin/codex")
        report["status"] = "built_unqualified"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["finished_at_unix"] = time.time()
        write_manifest(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "original-bundle", "patch", "lock-patch", "output", "cargo-home",
                 "rustup-home", "target-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--patch-sha256", default="unrecorded")
    parser.add_argument("--lock-patch-sha256", default="unrecorded")
    parser.add_argument("--jobs", type=int, default=4)
    report = build(parser.parse_args())
    print(json.dumps({"status": report["status"], "runtime_binary": report["runtime_binary"]}))


if __name__ == "__main__":
    main()
