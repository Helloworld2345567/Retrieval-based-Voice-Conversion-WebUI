"""Shared, domain-neutral inference infrastructure."""

from .devices import DeviceConfig, select_device
from .executables import find_executable
from .paths import PROJECT_ROOT, cache_root, data_root, index_roots, resource_path


__all__ = [
    "DeviceConfig",
    "PROJECT_ROOT",
    "cache_root",
    "data_root",
    "find_executable",
    "index_roots",
    "resource_path",
    "select_device",
]
