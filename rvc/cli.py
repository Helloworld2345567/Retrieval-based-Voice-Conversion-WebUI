"""Public CLI dispatcher. Import inference engines only after parsing commands."""

import argparse
import sys


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        if args and args[0] == "convert":
            from .convert_cli import main as convert

            return convert(args[1:])
        if args and args[0] == "separate":
            from .separation.cli import main as separate

            return separate(args[1:])
        parser = argparse.ArgumentParser(
            prog="python -m rvc", description="Offline RVC conversion and vocal separation."
        )
        commands = parser.add_subparsers(dest="command", required=True)
        commands.add_parser("convert", help="Convert audio with an RVC model.")
        commands.add_parser("separate", help="Separate vocals or manage the five supported models.")
        parser.parse_args(args)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(f"rvc: error: {error}", file=sys.stderr)
        return 1
