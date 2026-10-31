"""Shared project, resource, and writable runtime path resolution."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def is_source_tree():
    """Return whether the package is running from a project source tree."""
    return (PROJECT_ROOT / "pyproject.toml").is_file() and (PROJECT_ROOT / "rvc").is_dir()


def data_root():
    """Return the root for mutable model data without writing into site-packages."""
    override = os.environ.get("RVC_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    if is_source_tree():
        return PROJECT_ROOT
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return (base / "RVC").resolve()
    base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return (base / "rvc").resolve()


def cache_root():
    """Return a writable cache root for downloads and transient artifacts."""
    override = os.environ.get("RVC_CACHE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    if is_source_tree():
        return PROJECT_ROOT / ".codex"
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return (base / "RVC" / "Cache").resolve()
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return (base / "rvc").resolve()


def resource_path(variable, relative, *, base=None):
    """Resolve a resource override without changing the process environment."""
    value = Path(os.environ.get(variable, str(data_root() / relative))).expanduser()
    if not value.is_absolute() and base is not None:
        value = Path(base) / value
    return value.resolve()


def index_roots(*, base=None):
    return (
        resource_path("outside_index_root", "assets/indices", base=base),
        resource_path("index_root", "logs", base=base),
    )
