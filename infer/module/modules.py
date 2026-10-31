"""Compatibility access to retained RVC inference building blocks."""

from importlib import import_module


__all__ = [
    "Flip",
    "LRELU_SLOPE",
    "LayerNorm",
    "ResBlock1",
    "ResBlock2",
    "ResidualCouplingLayer",
    "WN",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("rvc.vc.modules"), name)
    globals()[name] = value
    return value
