"""Compatibility access to retained voice-conversion audio helpers."""

from importlib import import_module


__all__ = ["load_audio", "wav2"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("rvc.vc.audio"), name)
    globals()[name] = value
    return value
