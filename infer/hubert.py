"""Compatibility access to retained HuBERT feature extraction.

The legacy loader signature is preserved while model loading remains deferred.
"""

from importlib import import_module

from rvc.core.paths import resource_path


HUBERT_MODEL_PATH = resource_path("hubert_root", "assets/hubert_base")

__all__ = [
    "HUBERT_MODEL_PATH",
    "HubertModelWithFinalProj",
    "extract_hubert_features",
    "load_hubert_model",
]


def load_hubert_model(device, is_half=False):
    retained = import_module("rvc.vc.hubert")
    return retained.load_hubert_model(HUBERT_MODEL_PATH, device, is_half)


def __getattr__(name):
    if name not in {"HubertModelWithFinalProj", "extract_hubert_features"}:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("rvc.vc.hubert"), name)
    globals()[name] = value
    return value
