"""Compatibility access to the retained FCPE inference implementation."""

from importlib import import_module


__all__ = ["FCPEInfer"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("rvc.vc.fcpe"), name)
    globals()[name] = value
    return value
