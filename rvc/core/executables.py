"""Discovery of executables bundled at the project root."""

import os
from pathlib import Path

from .paths import PROJECT_ROOT


def find_executable(name, *, root=None):
    """Return a bundled executable path, or its name for normal PATH lookup."""
    root = PROJECT_ROOT if root is None else Path(root)
    suffix = ".exe" if os.name == "nt" else ""
    local = root / (name + suffix)
    return str(local) if local.is_file() else name
