"""Public interfaces for offline voice conversion.

Inference dependencies are imported only when an exported value is accessed.
"""

from importlib import import_module


_EXPORTS = {
    "AudioResult": (".engine", "AudioResult"),
    "ModelMetadata": (".checkpoints", "ModelMetadata"),
    "Speaker": (".checkpoints", "Speaker"),
    "VoiceConversionConfig": (".config", "VoiceConversionConfig"),
    "VoiceConverter": (".engine", "VoiceConverter"),
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
