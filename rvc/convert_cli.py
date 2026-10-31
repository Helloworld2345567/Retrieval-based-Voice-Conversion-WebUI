"""Compatibility alias for :mod:`rvc.vc.cli`."""

from importlib import import_module
import sys


sys.modules[__name__] = import_module("rvc.vc.cli")
