"""Pinned, lazy access to the existing Artemis mathematical implementation.

The research package keeps its own relative imports in a private namespace.
Only its PDE modules are loaded: its package initializer, old Controller and
launch scripts are never executed by this adapter.
"""

from __future__ import annotations

from hashlib import sha256
from importlib import import_module
from pathlib import Path
import subprocess
import sys
from threading import RLock
from types import ModuleType

# Cold-deployed Artemis oracle revision; existing campaigns retain their own pin.
ARTEMIS_PHYSICAL_REVISION = "91e2889dcd3f17137a13d4fd349b49539d3b5629"
_PACKAGE_PATH = "src/pde_agent_verifier_solver/plugins/vp_1d1v"
_LOCK = RLock()


class ArtemisPhysicalPackage:
    def __init__(self, checkout: Path | str, *, expected_revision: str | None = None):
        self.revision = ARTEMIS_PHYSICAL_REVISION if expected_revision is None else expected_revision
        if (not isinstance(self.revision, str) or len(self.revision) != 40
                or any(char not in "0123456789abcdef" for char in self.revision)):
            raise ValueError("Artemis physical revision must be an exact lowercase Git commit SHA")
        self.checkout = Path(checkout).expanduser().resolve(strict=True)
        self.package_root = self.checkout / _PACKAGE_PATH
        self._verify()
        self.namespace = "_pde_artemis_physical_" + sha256(
            (str(self.checkout) + "\0" + self.revision).encode()
        ).hexdigest()[:16]
        with _LOCK:
            if self.namespace not in sys.modules:
                package = ModuleType(self.namespace)
                package.__path__ = [str(self.package_root)]
                package.__package__ = self.namespace
                sys.modules[self.namespace] = package

    def _verify(self):
        kind = subprocess.run(
            ["git", "-C", str(self.checkout), "cat-file", "-t", self.revision],
            check=True, capture_output=True,
        ).stdout.strip()
        if kind != b"commit":
            raise ValueError("Artemis physical revision must identify a Git commit")
        result = subprocess.run(
            ["git", "-C", str(self.checkout), "ls-tree", "-r", "-z",
             self.revision, "--", _PACKAGE_PATH],
            check=True, capture_output=True,
        )
        if not result.stdout:
            raise ValueError("Artemis checkout lacks the pinned physical package")
        expected = set()
        for row in result.stdout.split(b"\0"):
            if not row:
                continue
            metadata, relative = row.split(b"\t", 1)
            mode, kind, digest = metadata.split()
            path = self.checkout / relative.decode()
            if kind != b"blob" or mode not in (b"100644", b"100755") or path.is_symlink():
                raise ValueError("Artemis physical source must contain regular files")
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.suffix == ".py":
                expected.add(path)
        unexpected = set(self.package_root.rglob("*.py")) - expected
        if unexpected:
            raise ValueError("Artemis physical package contains unpinned Python modules")

    def module(self, name: str):
        if not name.isidentifier() or name.startswith("_"):
            raise ValueError("Artemis module must be a public module name")
        with _LOCK:
            return import_module(f"{self.namespace}.{name}")
