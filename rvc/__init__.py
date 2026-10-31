"""Stable public interfaces for offline voice conversion and separation.

All exports are resolved lazily so importing :mod:`rvc` does not initialize an
inference framework or an audio codec.
"""

from importlib import import_module


_EXPORTS = {
    "AudioEncodingParams": ("rvc.separation", "AudioEncodingParams"),
    "AudioParams": ("rvc.separation", "AudioParams"),
    "AudioResult": ("rvc.vc", "AudioResult"),
    "InferenceParams": ("rvc.separation", "InferenceParams"),
    "MSSeparator": ("rvc.separation", "MSSeparator"),
    "ModelMetadata": ("rvc.vc", "ModelMetadata"),
    "ModelSpec": ("rvc.separation", "ModelSpec"),
    "RequestParams": ("rvc.separation", "RequestParams"),
    "SeparationRequest": ("rvc.separation", "SeparationRequest"),
    "Speaker": ("rvc.vc", "Speaker"),
    "VoiceConversionConfig": ("rvc.vc", "VoiceConversionConfig"),
    "VoiceConverter": ("rvc.vc", "VoiceConverter"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
