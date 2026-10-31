"""Keep test-created files below the project workspace."""

from pathlib import Path


def pytest_configure(config):
    if config.option.basetemp is None:
        config.option.basetemp = str(Path(__file__).resolve().parents[1] / ".codex" / "pytest")
