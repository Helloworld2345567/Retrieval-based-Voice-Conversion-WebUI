"""Compatibility access to retained voice-conversion CUDA Graph helpers."""

from importlib import import_module


__all__ = [
    "ENV_NAME",
    "MAX_CACHE_ENV",
    "clear_cuda_graph_cache",
    "configure_cuda_graph",
    "cuda_graph_enabled",
    "detect_cuda_graph_support",
    "get_cuda_graph_stats",
    "run_cuda_graph",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("rvc.vc.cuda_graph"), name)
    globals()[name] = value
    return value
