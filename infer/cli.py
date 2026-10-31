"""Compatibility entry point for the original offline conversion command."""

from pathlib import Path
import sys


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rvc.vc.cli import main as convert_main


def main(argv=None):
    return convert_main(argv, legacy=True)


if __name__ == "__main__":
    raise SystemExit(main())
