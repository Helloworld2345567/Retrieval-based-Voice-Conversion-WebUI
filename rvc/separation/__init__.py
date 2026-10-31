"""Public interfaces for the retained five-model separation runtime.

Torch and audio codecs remain unloaded until their runtime classes are used.
"""

from importlib import import_module


_EXPORTS = {
    "AudioEncodingParams": (".request", "AudioEncodingParams"),
    "AudioParams": (".request", "AudioParams"),
    "InferenceParams": (".request", "InferenceParams"),
    "MSSeparator": (".separator", "MSSeparator"),
    "ModelSpec": (".catalog", "ModelSpec"),
    "RequestParams": (".request", "RequestParams"),
    "SeparationRequest": (".request", "SeparationRequest"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
