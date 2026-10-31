"""Compatibility alias for :mod:`rvc.vc.cuda_graph`."""

from importlib import import_module
import sys


sys.modules[__name__] = import_module("rvc.vc.cuda_graph")
