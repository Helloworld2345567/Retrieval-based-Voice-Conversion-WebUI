"""Compatibility access to retained inference tensor helpers."""

from importlib import import_module


__all__ = [
    "fused_add_tanh_sigmoid_multiply",
    "get_padding",
    "init_weights",
    "sequence_mask",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("rvc.vc.commons"), name)
    globals()[name] = value
    return value
