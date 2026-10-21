"""Shared host notice, not a permission change or live CUDA qualification."""
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "scripts/canonical_codex.config.toml"
# Complete semantic identity of the ff7ec78 shared config, before this notice.
# Deliberate future policy changes must explicitly update this regression pin.
BASE_CONFIG_SHA256 = "01e3fe619bcf50bf39310616c37b786155edf3639f0a3c848fd15ff7d882874c"


def test_only_developer_notice_changes_shared_config():
    config = tomllib.loads(CONFIG.read_text())
    notice = config.pop("developer_instructions")
    assert "restricted_namespace_workspace_edit_v1" in notice
    assert sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest() == BASE_CONFIG_SHA256


def test_notice_is_host_scoped_and_never_grants_escalation():
    notice = tomllib.loads(CONFIG.read_text())["developer_instructions"]
    for text in (
        "only on hosts where", "/proc/sys/user/max_net_namespaces is 0",
        "On other hosts", "neither condition is confirmed",
        "do not invoke or retry", "normal sandboxed shell tool",
        "only inside the already authorized Agent workspace",
        "Keep the current sandbox, approval policy, network policy and writable roots",
        "Do not request unsandboxed execution", "report the blocker",
        "Pass this host constraint to any child agents", "administrator task",
    ):
        assert text in notice


@pytest.mark.parametrize("filename", (
    "ns_canonical_structured_5round.json",
    "vpfp_strong_formal_5round_s101.json",
    "artemis_canonical_5round.json",
))
def test_pde_runners_select_the_same_shared_config(filename):
    path = ROOT / "configs" / filename
    config = json.loads(path.read_text())
    assert (path.parent / config["worker"]["codex_config"]).resolve() == CONFIG


def test_gpu_config_derivation_preserves_notice_and_existing_permissions():
    spec = importlib.util.spec_from_file_location("restricted_namespace_shared_gpu_test", ROOT / "scripts/shared_agent_gpu.py")
    module = importlib.util.module_from_spec(spec)
    # Dataclass processing consults the defining module's globals.
    import sys
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        text = CONFIG.read_text()
        notice_block = text.split('developer_instructions = """', 1)[1].split('"""', 1)[0]
        baseline_text = text.replace('developer_instructions = """' + notice_block + '"""', '', 1)
        devices = ["/dev/nvidia0", "/dev/nvidiactl", "/dev/nvidia-uvm"]
        baseline = tomllib.loads(module._gpu_native_config(baseline_text, devices))
        actual = tomllib.loads(module._gpu_native_config(text, devices))
        assert actual.pop("developer_instructions") == tomllib.loads(text)["developer_instructions"]
        assert actual == baseline
    finally:
        sys.modules.pop(spec.name, None)
