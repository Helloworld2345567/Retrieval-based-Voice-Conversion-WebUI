"""Compatibility access to retained RVC inference model classes."""

from importlib import import_module


__all__ = [
    "Generator",
    "GeneratorNSF",
    "ResidualCouplingBlock",
    "SineGen",
    "SourceModuleHnNSF",
    "SynthesizerTrnMs256NSFsid",
    "SynthesizerTrnMs256NSFsid_nono",
    "SynthesizerTrnMs768NSFsid",
    "SynthesizerTrnMs768NSFsid_nono",
    "TextEncoder",
    "sr2sr",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("rvc.vc.models"), name)
    globals()[name] = value
    return value
