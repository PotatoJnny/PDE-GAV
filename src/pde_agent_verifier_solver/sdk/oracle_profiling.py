"""Opt-in measurements attached to the current Controller Oracle invocation."""
from contextlib import contextmanager
from contextvars import ContextVar

_CURRENT = ContextVar('oracle_profile', default=None)


def current_oracle_profile():
    return _CURRENT.get()


@contextmanager
def capture_oracle_profile(enabled):
    measurements = {} if enabled else None
    token = _CURRENT.set(measurements)
    try:
        yield measurements
    finally:
        _CURRENT.reset(token)
