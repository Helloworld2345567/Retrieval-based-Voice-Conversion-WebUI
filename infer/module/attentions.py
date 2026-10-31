"""Compatibility access to retained inference attention layers."""

from importlib import import_module


__all__ = ["Encoder", "FFN", "MultiHeadAttention"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("rvc.vc.attentions"), name)
    globals()[name] = value
    return value
