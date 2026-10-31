"""Compatibility entry point for python -m tools.pymss.cli."""
from rvc.separation.cli import main as _main


def main(argv=None):
    return _main(argv, legacy=True)


if __name__ == "__main__":
    raise SystemExit(main())
