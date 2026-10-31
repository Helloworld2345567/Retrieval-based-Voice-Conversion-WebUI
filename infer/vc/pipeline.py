"""Compatibility access to the retained offline conversion pipeline."""

from importlib import import_module


__all__ = ["Pipeline", "change_rms"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("rvc.vc.pipeline"), name)
    globals()[name] = value
    return value
