"""Controller-owned devices and plugin resources for one hidden evaluation.

The scope carries no Candidate capability and never changes process CUDA
visibility. Plugins may lazily use the devices; ordinary CPU plugins need not.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from hashlib import sha256
from threading import RLock
from typing import Callable, Hashable, Iterator, Sequence, TypeVar


_Resource = TypeVar("_Resource")


class EvaluationDevicePool:
    def __init__(self, devices: Sequence[str]) -> None:
        if isinstance(devices, (str, bytes)):
            raise TypeError("evaluation devices must be a sequence")
        prepared = tuple(devices)
        if any(not isinstance(device, str) or not device for device in prepared):
            raise ValueError("evaluation devices must be nonempty device strings")
        if len(prepared) != len(set(prepared)):
            raise ValueError("evaluation devices cannot repeat")
        self._devices = prepared
        self._lock = RLock()
        self._resources: dict[Hashable, object] = {}
        self._closed = False
        self._close_error: BaseException | None = None

    @property
    def devices(self) -> tuple[str, ...]:
        return self._devices

    @property
    def cleanup_confirmed(self) -> bool:
        with self._lock:
            return self._closed and self._close_error is None

    def device_for(self, case_id: str) -> str:
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("device allocation requires a nonempty Case id")
        with self._lock:
            self._require_open()
            if not self._devices:
                raise RuntimeError("evaluation scope has no devices")
            index = int.from_bytes(sha256(case_id.encode("utf-8")).digest()[:8], "big")
            return self._devices[index % len(self._devices)]

    def get_resource(self, key: Hashable, factory: Callable[[], _Resource]) -> _Resource:
        """Construct one plugin-owned resource per key, even across threads."""
        with self._lock:
            self._require_open()
            if key not in self._resources:
                try:
                    resource = factory()
                except BaseException as error:
                    # A partially initialized plugin may already own device
                    # allocations, without returning an object we can close.
                    self._close_error = self._close_error or error
                    raise
                if not callable(getattr(resource, "close", None)):
                    error = TypeError("evaluation resources must implement close()")
                    self._close_error = self._close_error or error
                    raise error
                self._resources[key] = resource
            return self._resources[key]  # type: ignore[return-value]

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("evaluation device scope is closed")

    def close(self) -> None:
        """Close every resource before its GPU reservation may be returned.

        Plugins own synchronization and release of their device allocations.
        A failed close remains failed on repeated calls; it cannot become a
        successful cleanup merely because this scope is already closed.
        """
        with self._lock:
            if not self._closed:
                self._closed = True
                closed_ids: set[int] = set()
                for resource in reversed(tuple(self._resources.values())):
                    if id(resource) in closed_ids:
                        continue
                    closed_ids.add(id(resource))
                    try:
                        resource.close()
                    except BaseException as error:
                        if self._close_error is None:
                            self._close_error = error
                        else:
                            add_note = getattr(self._close_error, "add_note", None)
                            if callable(add_note):
                                add_note("Another evaluation resource also failed to close: " + type(error).__name__)
                self._resources.clear()
            if self._close_error is not None:
                raise self._close_error

    def __enter__(self) -> "EvaluationDevicePool":
        with self._lock:
            self._require_open()
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


_CUDA_WORKSPACE_RESOURCE = object()


class _CUDAWorkspaceCleanup:
    def __init__(self, pool: EvaluationDevicePool, torch_module: object) -> None:
        self.pool = pool
        self.torch = torch_module
        self.clear_workspaces = getattr(
            getattr(torch_module, "_C", None), "_cuda_clearCublasWorkspaces", None,
        )
        if not callable(self.clear_workspaces):
            raise RuntimeError("PyTorch cannot release the borrowed CUDA workspaces")

    def close(self) -> None:
        # Pool.close calls this last. A failed plugin close or partial factory
        # leaves ownership uncertain: keep its workspaces and quarantine cards.
        if self.pool._close_error is not None:
            return
        for device in self.pool.devices:
            self.torch.cuda.synchronize(device)
        # This PyTorch cache is process-wide, so never clear it per plugin or
        # per device while another resource may still be using a cuBLAS handle.
        self.clear_workspaces()
        for device in self.pool.devices:
            with self.torch.cuda.device(device):
                self.torch.cuda.empty_cache()


def register_cuda_workspace_cleanup(pool: EvaluationDevicePool, torch_module: object) -> None:
    """Register once, before any GPU resource, to release library workspaces last.

    Torch is passed by an opt-in plugin; importing this SDK module never loads
    CUDA. The scope must own every device whose workspaces it will release.
    """
    def first_resource() -> _CUDAWorkspaceCleanup:
        if pool._resources:
            raise RuntimeError("CUDA workspace cleanup must precede GPU resource registration")
        return _CUDAWorkspaceCleanup(pool, torch_module)

    pool.get_resource(_CUDA_WORKSPACE_RESOURCE, first_resource)


_CURRENT: ContextVar[EvaluationDevicePool | None] = ContextVar(
    "controller_evaluation_device_pool", default=None,
)


def current_evaluation_device_pool() -> EvaluationDevicePool | None:
    return _CURRENT.get()


@contextmanager
def bind_evaluation_device_pool(pool: EvaluationDevicePool | None) -> Iterator[None]:
    if pool is not None and not isinstance(pool, EvaluationDevicePool):
        raise TypeError("evaluation device pool must be EvaluationDevicePool or None")
    token = _CURRENT.set(pool)
    try:
        yield
    finally:
        _CURRENT.reset(token)
