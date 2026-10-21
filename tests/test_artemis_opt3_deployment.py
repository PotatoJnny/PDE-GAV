"""Offline profile experiment retains explicit source pins and protocol defaults."""

from hashlib import sha1
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pde_agent_verifier_solver.plugins.vp_1d1v import physical, product

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "91e2889dcd3f17137a13d4fd349b49539d3b5629"


def test_benchmark_configs_change_only_explicit_physical_profile_options():
    original = json.loads((ROOT / "configs/plasma_canonical_artemis.json").read_text())
    baseline = json.loads((ROOT / "configs/plasma_artemis_opt3_benchmark_baseline.json").read_text())
    optimized = json.loads((ROOT / "configs/plasma_artemis_opt3_benchmark.json").read_text())
    del original["factory_settings"]["reward_guidance_settings"]
    assert original == baseline
    options = optimized["factory_settings"]
    assert options.pop("expected_physical_revision")
    assert options.pop("physical_checkout").endswith("/artemis-oracle-opt3-20260909")
    assert options["oracle_settings"].pop("exact_query_profile_mode") == "frozen_basis"
    baseline["factory_settings"].pop("physical_checkout")
    assert optimized == baseline
    assert physical.ARTEMIS_PHYSICAL_REVISION == BASELINE


@pytest.mark.parametrize("revision", ["main", "HEAD", "a"*39, "A"*40, "", 10, False])
def test_nonexact_physical_pin_is_rejected_before_loading(tmp_path, revision):
    with pytest.raises(ValueError, match="exact lowercase"):
        physical.ArtemisPhysicalPackage(tmp_path, expected_revision=revision)


def _mock_package(tmp_path, monkeypatch):
    folder = tmp_path / physical._PACKAGE_PATH
    folder.mkdir(parents=True)
    module = folder / "probe.py"
    content = b"value = 1\n"
    module.write_bytes(content)
    digest = sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
    relative = module.relative_to(tmp_path).as_posix()
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if "cat-file" in command:
            return SimpleNamespace(stdout=b"commit\n")
        return SimpleNamespace(stdout=f"100644 blob {digest}\t{relative}\0".encode())
    monkeypatch.setattr(physical.subprocess, "run", run)
    return module, calls


def test_revisions_separate_namespaces_without_file_content_checks(tmp_path, monkeypatch):
    module, calls = _mock_package(tmp_path, monkeypatch)
    default = physical.ArtemisPhysicalPackage(tmp_path)
    explicit = physical.ArtemisPhysicalPackage(tmp_path, expected_revision="2"*40)
    assert default.revision == BASELINE
    assert explicit.revision == "2"*40
    assert default.namespace != explicit.namespace
    assert any(BASELINE in call for call in calls)
    assert any("2"*40 in call for call in calls)
    module.write_text("value = 2\n")
    physical.ArtemisPhysicalPackage(tmp_path, expected_revision="2"*40)


def test_explicit_revision_does_not_allow_unpinned_python_files(tmp_path, monkeypatch):
    module, _ = _mock_package(tmp_path, monkeypatch)
    (module.parent / "unexpected.py").write_text("value = 2\n")
    with pytest.raises(ValueError, match="unpinned"):
        physical.ArtemisPhysicalPackage(tmp_path, expected_revision="2"*40)


def test_tree_object_is_not_an_accepted_commit_pin(tmp_path, monkeypatch):
    monkeypatch.setattr(physical.subprocess, "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=b"tree\n"))
    with pytest.raises(ValueError, match="Git commit"):
        physical.ArtemisPhysicalPackage(tmp_path, expected_revision="2"*40)


def test_preflight_forwards_pin_only_to_physical_source_verification(monkeypatch):
    captured = {}
    startup = object()
    def preflight(**kwargs):
        assert "expected_physical_revision" not in kwargs
        return startup
    def package(checkout, **kwargs):
        captured.update(checkout=checkout, **kwargs)
        return SimpleNamespace(module=lambda name: SimpleNamespace(
            preflight_artemis_compact_v5_startup=preflight))
    monkeypatch.setattr(product, "ArtemisPhysicalPackage", package)
    _, loaded = product.preflight_artemis_backend(
        physical_checkout="physical", admission_record_path="admission",
        expected_admission_record_sha256="a"*64, science_config_path="science",
        expected_science_config_sha256="b"*64, expected_physical_revision="2"*40)
    assert loaded is startup
    assert captured == {"checkout": "physical", "expected_revision": "2"*40}
