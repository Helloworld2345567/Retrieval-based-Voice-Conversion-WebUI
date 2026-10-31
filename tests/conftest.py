"""Keep test-created files below the project workspace."""

from pathlib import Path


def pytest_configure(config):
    if config.option.basetemp is None:
        base_temp = Path(__file__).resolve().parents[1] / ".codex" / "pytest"
        base_temp.parent.mkdir(parents=True, exist_ok=True)
        config.option.basetemp = str(base_temp)
