"""Compatibility helpers with retained index and HuBERT equivalents only."""

import os
from importlib import import_module

from rvc.core.paths import index_roots, resource_path
from rvc.vc.checkpoints import find_index


__all__ = ["get_index_path_from_model", "load_hubert"]


def get_index_path_from_model(sid, speaker_id=None):
    """Return the best retained feature index, or the legacy empty sentinel."""
    try:
        selected_speaker = None if speaker_id is None else int(speaker_id)
    except (TypeError, ValueError):
        selected_speaker = None
    path = find_index(str(sid or ""), selected_speaker, index_roots())
    return os.fspath(path) if path is not None else ""


def load_hubert(config):
    """Load HuBERT from the explicit root carried by the retained VC config."""
    retained = import_module("rvc.vc.hubert")
    model_path = getattr(config, "hubert_root", resource_path("hubert_root", "assets/hubert_base"))
    return retained.load_hubert_model(model_path, config.device, config.is_half)
