"""Compatibility API for the retained fixed five separation presets.

This namespace intentionally excludes the removed catalog, ensemble, workflow,
and server implementations.
"""

from importlib import import_module


_EXPORTS = {
    "MODELS": ("rvc.separation.catalog", "MODELS"),
    "MSSeparator": ("rvc.separation.separator", "MSSeparator"),
    "ModelSpec": ("rvc.separation.catalog", "ModelSpec"),
    "download_model": ("rvc.separation.download", "download_model"),
    "get_model": ("rvc.separation.catalog", "get_model"),
    "load_audio": ("rvc.separation.audio_io", "load_audio"),
    "model_path": ("rvc.separation.catalog", "model_path"),
    "save_audio": ("rvc.separation.audio_io", "save_audio"),
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
