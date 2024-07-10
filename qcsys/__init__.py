"""
qcsys
"""

import os

from .common import *
from .devices import *
from .analysis import *

with open(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "VERSION.txt")), "r"
) as _ver_file:
    __version__ = _ver_file.read().rstrip()

__author__ = "Shantanu Jha, Shoumik Chowdhury"
__credits__ = "EQuS"
