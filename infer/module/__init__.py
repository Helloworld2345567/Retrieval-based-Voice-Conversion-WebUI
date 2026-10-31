"""Inference-only compatibility namespace for retained RVC model modules."""

from importlib import import_module


_MODULES = {"attentions", "commons", "models", "modules"}
__all__ = sorted(_MODULES)


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = import_module(f"{__name__}.{name}")
    globals()[name] = value
    return value
