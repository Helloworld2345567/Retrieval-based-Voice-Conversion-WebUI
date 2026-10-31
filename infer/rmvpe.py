"""Compatibility access to retained RMVPE inference classes."""

from importlib import import_module


__all__ = [
    "BiGRU",
    "ConvBlockRes",
    "Decoder",
    "DeepUnet",
    "E2E",
    "Encoder",
    "Intermediate",
    "MelSpectrogram",
    "RMVPE",
    "ResDecoderBlock",
    "ResEncoderBlock",
    "STFT",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("rvc.vc.rmvpe"), name)
    globals()[name] = value
    return value
