"""Compatibility namespace for retained stateless voice-conversion helpers."""

from importlib import import_module


_MODULES = {"pipeline", "utils"}
__all__ = sorted(_MODULES)


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = import_module(f"{__name__}.{name}")
    globals()[name] = value
    return value
